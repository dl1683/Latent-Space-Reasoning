"""Canonical GCP dispatcher for latent-space-reasoning jobs (one VM per shard).

    python experiments/cloud/lsr_cloud.py launch experiments/cloud/configs/<cfg>.json [--only i,j] [--dry-run]
    python experiments/cloud/lsr_cloud.py status [run]      # VMs, shard states, relaunchable shards
    python experiments/cloud/lsr_cloud.py relaunch <run>    # re-create VMs for unfinished shards (preempted)
    python experiments/cloud/lsr_cloud.py pull <run>        # copy results to experiments/results/cloud/<run>/
    python experiments/cloud/lsr_cloud.py spend             # recompute experiments/cloud/SPEND.md
    python experiments/cloud/lsr_cloud.py stop <run|ALL>    # delete this program's VMs (never anything else)

Config: {"run", "kind", "machine", "accelerator" ("nvidia-l4:1"), "usd_per_hour", "cost_cap_usd" (per VM),
"zones": [...], "requirements": [...], "common": {...}, "shards": [{...}, ...]}. Each VM gets common|shard.
Only resources named sc-bench-lsr-* with label app=stateful-bench,role=lsr are created or deleted.
"""
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_SDK = r"C:\Users\devan\AppData\Local\Google\Cloud SDK\google-cloud-sdk"
GCLOUD = [_SDK + r"\platform\bundledpython\python.exe", _SDK + r"\lib\gcloud.py"]
PROJECT = "iqidis-developer"
BUCKET = "gs://sc-bench-iqidis-artifacts/lsr"
SA = "sc-bench-worker@iqidis-developer.iam.gserviceaccount.com"
IMAGE_FAMILY, IMAGE_PROJECT = "pytorch-2-9-cu129-ubuntu-2204-nvidia-580", "deeplearning-platform-release"
PREFIX = "sc-bench-lsr-"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REGISTRY = HERE / "launches.jsonl"  # one row per VM creation: run, shard, vm, zone, rate, created


def gcloud(*args, check=True):
    proj = [] if args[0] == "storage" else [f"--project={PROJECT}"]
    p = subprocess.run([*GCLOUD, *args, *proj], capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(p.stderr.strip()[-2000:])
    return p.stdout


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def vm_name(run, i):
    return f"{PREFIX}{run}-{i:03d}".lower().replace("_", "-")[:63]


def upload_bundle(run, cfg):
    with tempfile.TemporaryDirectory() as td:
        reqs = Path(td) / "requirements.txt"
        reqs.write_text("\n".join(cfg.get("requirements", ["transformers>=5.12,<6", "accelerate", "hf_transfer", "jinja2>=3.1"])) + "\n")
        gcloud("storage", "cp", "-r", str(HERE / "lsr_jobs"), str(HERE / "automon.sh"), str(reqs), f"{BUCKET}/runs/{run}/bundle/")


def create_vm(cfg, i, shard_path):
    run = cfg["run"]
    name = vm_name(run, i)
    hours = max(0.25, cfg["cost_cap_usd"] / cfg["usd_per_hour"])
    for v in list_vms(run):  # stale VM of the same name (e.g. powered off after a failure): replace it
        if v["name"] == name:
            gcloud("compute", "instances", "delete", name, f"--zone={v['zone'].split('/')[-1]}", "--quiet", check=False)
    gcloud("storage", "rm", f"{shard_path}/out/done.json", check=False)
    for zone in cfg["zones"]:
        args = ["compute", "instances", "create", name, f"--zone={zone}", f"--machine-type={cfg['machine']}",
                f"--service-account={SA}", "--scopes=https://www.googleapis.com/auth/cloud-platform",
                f"--image-family={IMAGE_FAMILY}", f"--image-project={IMAGE_PROJECT}",
                f"--boot-disk-size={cfg.get('disk_gb', 200)}GB", "--boot-disk-type=pd-balanced",
                f"--labels=app=stateful-bench,role=lsr,run={run.lower()}",
                "--provisioning-model=SPOT", "--instance-termination-action=DELETE",
                f"--max-run-duration={int(hours * 3600)}s", "--maintenance-policy=TERMINATE",
                f"--metadata=lsr-job={shard_path},lsr-idle-min={cfg.get('idle_min', 30)},lsr-grace-min={cfg.get('grace_min', 45)}",
                f"--metadata-from-file=startup-script={HERE / 'vm_startup.sh'},shutdown-script={HERE / 'vm_shutdown.sh'}"]
        if cfg.get("accelerator"):
            t, c = cfg["accelerator"].split(":")
            args.insert(6, f"--accelerator=type={t},count={c}")
        try:
            gcloud(*args)
        except RuntimeError as e:
            el = str(e).lower()
            if any(s in el for s in ("stockout", "enough resources", "not available", "does not exist in zone",
                                      "zone_resource_pool_exhausted", "quota", "unsupported")):
                print(f"  {name}: {zone} unavailable ({el.splitlines()[-1][:120]})")
                continue
            raise
        with REGISTRY.open("a") as f:
            f.write(json.dumps({"run": run, "shard": i, "vm": name, "zone": zone, "usd_per_hour": cfg["usd_per_hour"],
                                "machine": cfg["machine"], "accelerator": cfg.get("accelerator"), "created": now()}) + "\n")
        print(f"  launched {name} in {zone} (cap {hours:.1f} h, ${cfg['cost_cap_usd']})")
        return zone
    print(f"  {name}: NO CAPACITY in {cfg['zones']}")
    return None


def shard_state(run, i):
    out = gcloud("storage", "ls", f"{BUCKET}/runs/{run}/shards/{i:03d}/out/", check=False)
    if "done.json" in out:
        rc = json.loads(gcloud("storage", "cat", f"{BUCKET}/runs/{run}/shards/{i:03d}/out/done.json"))["rc"]
        return "done" if rc == 0 else f"failed(rc={rc})"
    return "started" if "boot_" in out else "pending"


def launch(cfg_path, only=None, dry=False):
    cfg = json.loads(Path(cfg_path).read_text())
    run = cfg["run"]
    idx = [int(x) for x in only.split(",")] if only else range(len(cfg["shards"]))
    total_cap = sum(cfg["cost_cap_usd"] for _ in idx)
    print(f"run {run}: {len(list(idx))} VMs x {cfg['machine']} {cfg.get('accelerator')} cap ${total_cap:.0f} total")
    if dry:
        return
    upload_bundle(run, cfg)
    gcloud("storage", "cp", str(cfg_path), f"{BUCKET}/runs/{run}/config.json")

    def one(i):
        shard = {**cfg.get("common", {}), **cfg["shards"][i], "kind": cfg["kind"], "run": run, "shard": i}
        p = f"{BUCKET}/runs/{run}/shards/{i:03d}"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(shard, f)
        gcloud("storage", "rm", "-r", f"{p}/out", check=False)  # launch = fresh; only relaunch resumes
        gcloud("storage", "cp", f.name, f"{p}/shard.json")
        os.unlink(f.name)
        return create_vm(cfg, i, p)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(one, idx))


def list_vms(run=None):
    flt = "labels.app=stateful-bench AND labels.role=lsr" + (f" AND labels.run={run.lower()}" if run else "")
    vms = json.loads(gcloud("compute", "instances", "list", f"--filter={flt}", "--format=json"))
    return [v for v in vms if v["name"].startswith(PREFIX)]


def status(run=None):
    for v in list_vms(run):
        print(f"{v['name']:45s} {v['zone'].split('/')[-1]:15s} {v['status']:12s} {v['labels'].get('run')}")
    if run:
        cfg = json.loads(gcloud("storage", "cat", f"{BUCKET}/runs/{run}/config.json"))
        with ThreadPoolExecutor(16) as ex:
            states = list(ex.map(lambda i: shard_state(run, i), range(len(cfg["shards"]))))
        for i, s in enumerate(states):
            print(f"  shard {i:03d}: {s}")


def relaunch(run, rebundle=False, retry_failed=False):
    cfg = json.loads(gcloud("storage", "cat", f"{BUCKET}/runs/{run}/config.json"))
    if rebundle:  # running VMs keep the code they booted with; only (re)launched shards get the new bundle
        upload_bundle(run, cfg)
    live = {v["name"] for v in list_vms(run) if v["status"] not in ("TERMINATED",)}
    for i in range(len(cfg["shards"])):
        if vm_name(run, i) in live:
            continue
        s = shard_state(run, i)
        if retry_failed and s.startswith("failed"):
            gcloud("storage", "rm", f"{BUCKET}/runs/{run}/shards/{i:03d}/out/done.json", check=False)
            s = "started"
        if s in ("pending", "started"):
            print(f"shard {i}: {s}, relaunching")
            create_vm(cfg, i, f"{BUCKET}/runs/{run}/shards/{i:03d}")


def reap():
    """Delete this program's powered-off (TERMINATED) VMs; their GPUs are already released."""
    for v in list_vms():
        if v["status"] == "TERMINATED":
            gcloud("compute", "instances", "delete", v["name"], f"--zone={v['zone'].split('/')[-1]}", "--quiet", check=False)
            print(f"reaped {v['name']}")


def stop(run):
    for v in list_vms(None if run == "ALL" else run):
        gcloud("compute", "instances", "delete", v["name"], f"--zone={v['zone'].split('/')[-1]}", "--quiet", check=False)
        print(f"deleted {v['name']}")


def pull(run):
    dst = REPO / "experiments" / "results" / "cloud" / run
    dst.mkdir(parents=True, exist_ok=True)
    gcloud("storage", "rsync", "-r", "-x", r".*heartbeat$|.*\.pt$", f"{BUCKET}/runs/{run}", str(dst))
    print(f"pulled to {dst}")


def spend():
    """Accrued cost per launch = (end or now - created) x all-in hourly rate. End = VM gone: last end_/done
    marker in GCS for that shard after creation, else last-seen time recorded at reap. Conservative (overestimates)."""
    rows = [json.loads(l) for l in REGISTRY.read_text().splitlines()] if REGISTRY.exists() else []
    live = {v["name"]: v for v in list_vms()}
    t_now = dt.datetime.now(dt.timezone.utc)
    by_run, total = {}, 0.0
    ends = {}
    for run in {r["run"] for r in rows}:
        ls = gcloud("storage", "ls", "-l", "-r", f"{BUCKET}/runs/{run}/shards/", check=False)
        for line in ls.splitlines():
            parts = line.split()
            if len(parts) == 3 and ("/out/end_" in parts[2] or parts[2].endswith("/out/done.json")):
                shard = int(parts[2].split("/shards/")[1][:3])
                ts = dt.datetime.fromisoformat(parts[1].replace("Z", "+00:00"))
                ends.setdefault((run, shard), []).append(ts)
    for r in rows:
        c = dt.datetime.fromisoformat(r["created"])
        latest = max((x for x in rows if x["vm"] == r["vm"]), key=lambda x: x["created"])
        later = [e for e in ends.get((r["run"], r["shard"]), []) if e > c]
        if r is latest and r["vm"] in live and live[r["vm"]]["status"] == "RUNNING":
            end = t_now
        elif later:
            end = min(later)
        else:
            end = c + dt.timedelta(minutes=10)  # gone with no end marker (failed create/boot): charge 10 min
        usd = max(0.0, (end - c).total_seconds() / 3600) * r["usd_per_hour"]
        by_run[r["run"]] = by_run.get(r["run"], 0.0) + usd
        total += usd
    lines = ["# LSR GCP spend ledger", "", f"Updated {now()}. Estimated from VM lifetimes x all-in Spot rate "
             "(Cloud Billing catalog, 2026-10-11). Budget 5,000 USD.", "",
             "| run | VMs launched | est. USD |", "|---|---:|---:|"]
    for run in sorted(by_run):
        lines.append(f"| {run} | {sum(1 for r in rows if r['run'] == run)} | {by_run[run]:.2f} |")
    lines += [f"| **total** | {len(rows)} | **{total:.2f}** |", "",
              f"Live VMs now: {sum(1 for v in live.values() if v['status'] == 'RUNNING')}"]
    (HERE / "SPEND.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    if cmd == "launch":
        only = rest[rest.index("--only") + 1] if "--only" in rest else None
        launch(rest[0], only, "--dry-run" in rest)
    elif cmd == "status":
        status(rest[0] if rest else None)
    elif cmd == "relaunch":
        relaunch(rest[0], "--rebundle" in rest, "--retry-failed" in rest)
    elif cmd == "pull":
        pull(rest[0])
    elif cmd == "spend":
        spend()
    elif cmd == "reap":
        reap()
    elif cmd == "stop":
        stop(rest[0])
