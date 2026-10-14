"""Canonical GCP dispatcher for latent-space-reasoning jobs (one VM per shard).

    python experiments/cloud/lsr_cloud.py launch experiments/cloud/configs/<cfg>.json [--only i,j] [--dry-run]
    python experiments/cloud/lsr_cloud.py status [run]      # VMs and two-layer terminal receipt states
    python experiments/cloud/lsr_cloud.py relaunch <run>    # refused: request a fresh gated attempt
    python experiments/cloud/lsr_cloud.py pull <run>        # copy results to experiments/results/cloud/<run>/
    python experiments/cloud/lsr_cloud.py spend             # recompute experiments/cloud/SPEND.md
    python experiments/cloud/lsr_cloud.py stop <run|ALL>    # delete this program's VMs (never anything else)

Config: {"run", "kind", "machine", "accelerator" ("nvidia-l4:1"), "usd_per_hour", "cost_cap_usd" (per VM),
"zones": [...], "requirements": [...], "common": {...}, "shards": [{...}, ...]}. Each VM gets common|shard.
Requires explicit has_gpu/guest_accelerators, immutable image, exact lock/hash, separate control argv,
revision-bound mirror manifest for model jobs, and a bound 1/3/4-hour allocation (four hours by default).
Only resources named sc-bench-lsr-* with label app=stateful-bench,role=lsr are created or deleted.
"""
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
import hashlib
import math
import re
import tarfile
import time

try:
    from .lsr_jobs.run import assemble_shard
except ImportError:
    from lsr_jobs.run import assemble_shard
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


def gcloud_env():
    """Pass only SDK essentials; shell-injected variables can exceed Windows limits."""
    keep = {"systemroot", "temp", "tmp", "userprofile", "appdata", "localappdata", "home",
            "cloudsdk_active_config_name", "cloudsdk_config",
            "http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    env = {k: v for k, v in os.environ.items() if k.lower() in keep}
    entries, seen, size = [], set(), 0
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        entry = entry.strip().strip('"')
        if not entry or not os.path.isdir(entry):
            continue
        key = os.path.normcase(os.path.abspath(entry))
        added = len(entry) + bool(entries)
        if key in seen or size + added >= 8000:
            continue
        entries.append(entry)
        seen.add(key)
        size += added
    env["PATH"] = os.pathsep.join(entries)
    return env


def gcloud(*args, check=True):
    proj = [] if args[0] == "storage" else [f"--project={PROJECT}"]
    p = subprocess.run([*GCLOUD, *args, *proj], capture_output=True, text=True, env=gcloud_env())
    if check and p.returncode != 0:
        raise RuntimeError(p.stderr.strip()[-2000:])
    return p.stdout


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


LIFETIME_SECONDS = 14400
WORK_STOP_SECONDS = 13500
SHUTDOWN_RESERVE_SECONDS = 900
CONFIG_KEYS = {"run", "kind", "entry", "machine", "accelerator", "usd_per_hour", "cost_cap_usd",
               "aggregate_cost_ceiling_usd", "zones", "requirements", "common", "shards", "disk_gb",
               "idle_min", "grace_min", "note", "boot_disk_type", "nic_type", "image_name",
               "has_gpu", "guest_accelerators", "weights_manifest_uri", "weights_manifest_sha256",
               "image_project", "image_id", "release", "requirements_lock_sha256", "weights_mirror_uri", "weights_local_path",
               "allow_hf_download", "rate_verified", "shape_verified", "lifetime_seconds",
               "work_stop_seconds", "shutdown_reserve_seconds", "max_hours", "shutdown_margin_hours", "ancillary_allowance_usd", "credit_guard_epoch"}


# Conservative administrative cutoff; operator must supply an earlier bound if needed.
CREDIT_GUARD_EPOCH = 1792195200  # 2026-10-17 00:00:00 UTC


def lifetime(cfg):
    values = tuple(cfg.get(k, v) for k, v in (("lifetime_seconds", 14400),
                   ("work_stop_seconds", 13500), ("shutdown_reserve_seconds", 900)))
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError("positive integer lifetime fields required")
    total, work, reserve = values
    if total not in (3600, 10800, 14400) or reserve < 900 or work + reserve != total:
        raise ValueError("invalid lifetime/reserve contract")
    return values


def ancillary(cfg):
    value = cfg.get("ancillary_allowance_usd", 0)
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid ancillary allowance")
    return value


def vm_name(run, i):
    # Refuse instead of truncating/normalizing names into collisions.
    name = f"{PREFIX}{run}-{i:03d}"
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,61}[a-z0-9]", name):
        raise ValueError("run must form a literal valid unique instance name (<=63 characters)")
    return name


def positive(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {field}")
    return value


# Operator receipt: RESEARCH_QUEUE.md, Immutable image and lock (2026-10-14).
# Release authority is independent of the candidate config; syntactically valid
# substitutions must not authorize a different deployment.
RELEASE_PINS = {
    "image_project": "deeplearning-platform-release",
    "image_name": "pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20261001",
    "image_id": "1805511023404495091",
    "requirements_lock_sha256": "1d7f449541ab71e0f87a2978537ad88795226e4d63d36fb27d654601c6903df9",
}
RELEASE_REVISION = "5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89"
RELEASE_MIRROR = "gs://sc-bench-iqidis-artifacts/lsr/models/gemma-4-31B/" + RELEASE_REVISION
RELEASE_MANIFEST_SHA256 = "8ea23dc0845967880fa8ce2661f256d13e59129d5fd78c048fbe95481da32d46"

# Operator-retained measured receipts, keyed by their exact byte SHA-256.
# Deliberately empty: no measured placement has been supplied. Candidate config
# bytes (including a self-declared digest) are never release authority.
RELEASE_PLACEMENT_EVIDENCE = {}


def effective_workload_sha256(shard):
    """Bind operation inputs, not a separately named recipe file.

    The evidence's own digest is excluded to avoid a circular hash. Allocation
    identity/clock fields are added by assembly and are not model operations.
    """
    excluded = {"placement_contract", "config", "config_sha256", "kind", "run", "shard",
                "attempt", "dispatch", "runtime_control_sha256", "requirements_lock_sha256",
                "weights_manifest_sha256", "entry", "entry_defaulted"}
    return hashlib.sha256(json.dumps({k: v for k, v in shard.items() if k not in excluded},
                                    sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_placement(cfg, shard):
    p = shard.get("placement_contract")
    if not isinstance(p, dict) or p.get("status") != "VERIFIED" or p.get("autograd_tested") is not True:
        raise ValueError("31B autograd placement/headroom UNVERIFIED")
    mapping, memory = p.get("device_map"), p.get("max_memory")
    devices = set(range(sum(a["count"] for a in cfg["guest_accelerators"])))
    if (not isinstance(mapping, dict) or not mapping or not devices
            or any(not isinstance(k, str) or not isinstance(v, str)
                   or v not in {f"cuda:{i}" for i in devices} for k, v in mapping.items())
            or not isinstance(memory, dict) or set(memory) != {str(i) for i in devices}
            or any(type(v) is not int or v <= 0 for v in memory.values())
            or p.get("allow_offload") is not False):
        raise ValueError("placement explicit mapping/memory/no-offload required")
    threshold = p.get("min_device_headroom_fraction")
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0.1 <= threshold < 1:
        raise ValueError("placement headroom threshold")
    evidence_hash = p.get("evidence_sha256")
    raw = RELEASE_PLACEMENT_EVIDENCE.get(evidence_hash) if isinstance(evidence_hash, str) else None
    if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != evidence_hash:
        raise ValueError("placement independent measured evidence missing/hash mismatch")
    evidence = json.loads(raw)
    runner = REPO / "experiments/memory/prototype/runner.py" if cfg["kind"] == "latent_memory_prototype" else HERE / "lsr_jobs" / (cfg["kind"] + ".py")
    config = Path(shard.get("config") or "")
    if not config.is_absolute():
        config = REPO / config
    if runner.is_symlink() or config.is_symlink() or not runner.is_file() or not config.is_file():
        raise ValueError("placement runner/config missing")
    binding = {"model": shard["model"], "revision": shard["revision"],
               "tokenizer_revision": shard["tokenizer_revision"], "machine": cfg["machine"],
               "guest_accelerators": cfg["guest_accelerators"], "kind": cfg["kind"],
               "phase": shard.get("phase"), "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
               "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
               "requirements_lock_sha256": cfg["requirements_lock_sha256"],
               "weights_manifest_sha256": cfg["weights_manifest_sha256"],
               "device_map": mapping, "max_memory": memory, "allow_offload": False,
               "effective_workload_sha256": effective_workload_sha256(shard),
               "min_device_headroom_fraction": threshold}
    if evidence.get("binding") != binding or shard.get("config_sha256") != binding["config_sha256"]:
        raise ValueError("placement measurement provenance mismatch")
    measured = evidence.get("headroom_fraction")
    procedure = evidence.get("procedure")
    if (evidence.get("status") != "PASS" or not isinstance(procedure, dict)
            or any(procedure.get(k) is not True for k in ("load", "backward", "decode"))
            or not isinstance(procedure.get("command"), list) or not procedure["command"]
            or any(not isinstance(x, str) or not x for x in procedure["command"])
            or not isinstance(procedure.get("input_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", procedure["input_sha256"])
            or not isinstance(measured, dict) or set(measured) != set(memory)
            or any(type(v) not in (int, float) or not math.isfinite(v) or not threshold <= v <= 1 for v in measured.values())):
        raise ValueError("placement measured load/backward/decode/headroom required")
    return binding


def validate_release_pins(cfg):
    """Validate only operator-authorized resources, separately from operation gates."""
    for field, value in RELEASE_PINS.items():
        if cfg.get(field) != value:
            raise ValueError(f"unreviewed release {field}")
    reqs = cfg.get("requirements")
    if (not isinstance(reqs, list) or not reqs
            or any(not isinstance(r, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+", r) for r in reqs)
            or hashlib.sha256(("\n".join(reqs) + "\n").encode()).hexdigest() != RELEASE_PINS["requirements_lock_sha256"]):
        raise ValueError("release dependency bytes mismatch")
    for part in cfg.get("shards", []):
        shard = {**cfg.get("common", {}), **part}
        if shard.get("model"):
            expected = {"weights_mirror_uri": RELEASE_MIRROR,
                        "weights_manifest_uri": RELEASE_MIRROR + "/_manifest_sha256.txt",
                        "weights_manifest_sha256": RELEASE_MANIFEST_SHA256,
                        "weights_local_path": "/opt/lsr/weights"}
            if shard["model"] != "google/gemma-4-31B" or any(cfg.get(k) != v for k, v in expected.items()):
                raise ValueError("unreviewed release model assets")
            if any(shard.get(k) != RELEASE_REVISION for k in ("revision", "tokenizer_revision")):
                raise ValueError("unreviewed release revision")
        elif ("model" not in shard or shard["model"] is not None
                or any(k not in cfg or cfg[k] is not None for k in ("weights_mirror_uri", "weights_manifest_uri", "weights_manifest_sha256", "weights_local_path"))):
            raise ValueError("model-free release requires null assets")
    return cfg


def validate_config(cfg):
    total, work, reserve = lifetime(cfg)
    if cfg.get("release") is True:
        validate_release_pins(cfg)
    for part in cfg.get("shards", []):
        shard = {**cfg.get("common", {}), **part}
        placement = shard.get("placement_contract")
        if shard.get("model") == "google/gemma-4-31B" and (cfg.get("release") is True or placement is not None):
            validate_placement(cfg, shard)
    unknown = set(cfg) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    if not cfg.get("shards") or not isinstance(cfg["shards"], list):
        raise ValueError("nonempty shards required")
    for i in range(len(cfg["shards"])):
        vm_name(cfg["run"], i)
        shard = assemble_shard(cfg, i)
        if shard.get("missing_operator_inputs"):
            raise ValueError("MISSING-operator: " + ", ".join(shard["missing_operator_inputs"]))
        if cfg.get("release") is True and cfg["kind"] == "s0_memory_port":
            if shard.get("phase") not in {"smoke", "fit"}:
                raise ValueError("MISSING-operator: S0 phase")
            for field in ("smoke_cases", "fit_cases"):
                cases = shard.get(field)
                if not isinstance(cases, list) or not cases or (field == "fit_cases" and len(cases) != 4):
                    raise ValueError("MISSING-operator: pinned " + field)
                digest = hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                if shard.get(field + "_sha256") != digest:
                    raise ValueError("S0 case digest mismatch: " + field)
            if (shard.get("model_source") != shard.get("model")
                    or shard.get("model_local") != cfg.get("weights_local_path")
                    or shard.get("allow_hf_download") is not False):
                raise ValueError("S0 flat receiver identity/root/download contract")
            placement = shard.get("placement")
            if (not isinstance(placement, dict) or not placement.get("device_map")
                    or placement["device_map"] != shard["placement_contract"]["device_map"]
                    or not placement.get("measurement_id")):
                raise ValueError("MISSING-operator: S0 receiver placement")
            evidence = json.loads(RELEASE_PLACEMENT_EVIDENCE[shard["placement_contract"]["evidence_sha256"]])
            if (placement.get("measurement_id") != evidence.get("measurement_id")
                    or not isinstance(placement.get("measurement_id"), str)
                    or not placement["measurement_id"].strip()
                    or placement.get("min_free_fraction") != shard["placement_contract"]["min_device_headroom_fraction"]):
                raise ValueError("S0 receiver measurement/headroom aliases mismatch")
            if shard.get("dtype") != "bfloat16" or shard.get("attn_implementation") not in {"eager", "sdpa"}:
                raise ValueError("MISSING-operator: S0 BF16/backend contract")
        if cfg["kind"] == "latent_memory_prototype" and not shard.get("model"):
            if cfg.get("has_gpu") is not False or shard.get("model") is not None or any(cfg.get(k) is not None for k in ("weights_local_path", "weights_mirror_uri", "weights_manifest_uri", "weights_manifest_sha256")):
                raise ValueError("model-free prototype requires CPU/null assets")
        for field in ("usd_per_hour", "cost_cap_usd"):
            if field in shard and shard[field] != cfg[field]:
                raise ValueError(f"nested {field} conflicts with allocation")
        for key, value in (("lifetime_seconds", total), ("work_stop_seconds", work), ("shutdown_reserve_seconds", reserve)):
            if key in shard and shard[key] != value:
                raise ValueError("nested lifetime conflicts with allocation")
        if "job_stop_seconds" in shard and shard["job_stop_seconds"] != work:
            raise ValueError("shard job_stop_seconds conflicts with creation clock")
    if cfg.get("idle_min",30) != 30 or cfg.get("grace_min",45) != 45:
        raise ValueError("runtime automon contract is fixed at idle=30/grace=45 minutes")
    if cfg.get("boot_disk_type") not in {"pd-standard", "pd-balanced", "pd-ssd"}:
        raise ValueError("explicit boot_disk_type required")
    if cfg.get("nic_type") not in {"GVNIC", "VIRTIO_NET"}:
        raise ValueError("explicit nic_type required")
    for field in ("usd_per_hour", "cost_cap_usd", "aggregate_cost_ceiling_usd", "disk_gb"):
        positive(cfg.get(field), field)
    for field in ("rate_verified", "shape_verified", "allow_hf_download"):
        if type(cfg.get(field)) is not bool:
            raise ValueError(f"explicit boolean {field} required")
    if cfg["allow_hf_download"]:
        raise ValueError("this dispatcher implements mirror-only launches")
    if cfg.get("max_hours", total / 3600) != total / 3600 or cfg.get("shutdown_margin_hours", reserve / 3600) != reserve / 3600:
        raise ValueError("legacy duration fields conflict with deadline contract")
    if cfg["cost_cap_usd"] < cfg["usd_per_hour"] * (total / 3600) + ancillary(cfg):
        raise ValueError("per-instance cost ceiling below complete lifetime estimate")
    if not isinstance(cfg.get("zones"), list) or len(cfg["zones"]) != 1 or not re.fullmatch(r"[a-z]+-[a-z0-9]+[0-9]-[a-z]", cfg["zones"][0]):
        raise ValueError("exactly one explicit zone; no fallback")
    if not re.fullmatch(r"[a-z][a-z0-9-]+", cfg.get("machine", "")):
        raise ValueError("explicit machine required")
    if type(cfg.get("has_gpu")) is not bool:
        raise ValueError("explicit has_gpu required")
    expected = cfg.get("guest_accelerators")
    if not isinstance(expected, list) or any(not isinstance(a, dict) or set(a) != {"type", "count"}
            or not re.fullmatch(r"nvidia-[a-z0-9-]+", a.get("type", ""))
            or type(a.get("count")) is not int or a["count"] <= 0 for a in expected):
        raise ValueError("explicit accelerator type/count contract required")
    if bool(expected) != cfg["has_gpu"]:
        raise ValueError("GPU expectation conflicts with accelerators")
    integrated = {"a3-highgpu-2g": [{"type": "nvidia-h100-80gb", "count": 2}],
                  "a2-highgpu-2g": [{"type": "nvidia-tesla-a100", "count": 2}]}
    if cfg["machine"] in integrated:
        if expected != integrated[cfg["machine"]] or cfg.get("accelerator"):
            raise ValueError("integrated shape accelerator mismatch")
    elif cfg["has_gpu"]:
        if not cfg["machine"].startswith("n1-") or len(expected) != 1 or cfg.get("accelerator") != f"{expected[0]['type']}:{expected[0]['count']}":
            raise ValueError("unreviewed GPU shape/attachment")
    elif not re.fullmatch(r"(?:n1|n2|n2d|e2|c2|c3)-[a-z0-9-]+", cfg["machine"]):
        raise ValueError("unreviewed CPU shape")
    if not cfg["has_gpu"] and cfg.get("accelerator"):
        raise ValueError("CPU shape cannot request accelerator")
    if cfg.get("accelerator") and not re.fullmatch(r"nvidia-[a-z0-9-]+:[1-9][0-9]*", cfg["accelerator"]):
        raise ValueError("invalid accelerator")
    for field in ("image_name", "image_project"):
        if not isinstance(cfg.get(field), str) or not re.fullmatch(r"[a-z][a-z0-9-]+", cfg[field]):
            raise ValueError(f"immutable {field} missing (families forbidden)")
    if type(cfg.get("release", False)) is not bool:
        raise ValueError("release must be boolean")
    if cfg.get("release", False) or "image_id" in cfg:
        if not isinstance(cfg.get("image_id"), str) or not re.fullmatch(r"[1-9][0-9]*", cfg["image_id"]):
            raise ValueError("resolved immutable image_id required")
    reqs = cfg.get("requirements")
    if not isinstance(reqs, list) or not reqs or any(not re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+", r) for r in reqs):
        raise ValueError("exact dependency lock missing")
    lock = hashlib.sha256(("\n".join(reqs) + "\n").encode()).hexdigest()
    if cfg.get("requirements_lock_sha256") != lock:
        raise ValueError("dependency lock hash mismatch")
    if any(assemble_shard(cfg, i).get("model") for i in range(len(cfg["shards"]))):
        if not re.fullmatch(r"gs://[a-z0-9._-]+/[A-Za-z0-9/_.-]+", cfg.get("weights_mirror_uri") or ""):
            raise ValueError("pre-staged weights_mirror_uri missing")
        if cfg.get("weights_local_path") != "/opt/lsr/weights":
            raise ValueError("weights_local_path must be /opt/lsr/weights")
        if not re.fullmatch(r"gs://[a-z0-9._-]+/[A-Za-z0-9/_.-]+", cfg.get("weights_manifest_uri") or "") or not re.fullmatch(r"[a-f0-9]{64}", cfg.get("weights_manifest_sha256") or ""):
            raise ValueError("revision-bound mirror manifest missing")
        for i in range(len(cfg["shards"])):
            shard = assemble_shard(cfg, i)
            if any(not re.fullmatch(r"[a-f0-9]{40}", shard.get(field) or "") for field in ("revision", "tokenizer_revision")):
                raise ValueError("model and tokenizer revision required")
    return cfg


def launch_plan(cfg, only=None):
    validate_config(cfg)
    idx = list(range(len(cfg["shards"]))) if only is None else [int(x) for x in only.split(",")]
    if not idx or len(set(idx)) != len(idx) or any(i < 0 or i >= len(cfg["shards"]) for i in idx):
        raise ValueError("invalid or duplicate shard selection")
    estimate = (cfg["usd_per_hour"] * (lifetime(cfg)[0] / 3600) + ancillary(cfg)) * len(idx)
    if estimate > cfg["aggregate_cost_ceiling_usd"]:
        raise ValueError("aggregate cost ceiling exceeded")
    return idx, estimate


def bundle_paths():
    paths = [HERE / "automon.sh"]
    forbidden = {".git", "__pycache__", "weights", "checkpoints"}
    for root in (HERE / "lsr_jobs", REPO / "experiments" / "memory"):
        if not root.is_dir():
            raise ValueError(f"bundle root missing: {root}")
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("bundle symlinks forbidden")
            if path.is_file() and not (set(path.relative_to(root).parts) & forbidden) and path.suffix.lower() in {".py", ".json", ".sh"}:
                paths.append(path)
    return paths


def bundle_name(path):
    return path.relative_to(HERE).as_posix() if path.is_relative_to(HERE) else path.relative_to(REPO).as_posix()


def upload_bundle(run, cfg):
    with tempfile.TemporaryDirectory() as td:
        reqs = Path(td) / "requirements.txt"
        reqs.write_text("\n".join(cfg["requirements"]) + "\n", encoding="utf-8", newline="\n")
        config = Path(td) / "config.json"
        config.write_text(json.dumps(cfg, sort_keys=True, separators=(",", ":")) + "\n",
                          encoding="utf-8", newline="\n")
        archive = Path(td) / "code.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for path in bundle_paths():
                tar.add(path, arcname=bundle_name(path), recursive=False)
            tar.add(reqs, arcname="requirements.txt")
            tar.add(config, arcname="config.json")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        gcloud("storage", "cp", str(archive), f"{BUCKET}/runs/{run}/bundle/code.tar.gz", "--if-generation-match=0")
        return digest


def refuse_collisions(cfg, idx):
    # All-instance name check, including unlabelled and terminated instances.
    existing = json.loads(gcloud("compute", "instances", "list", "--format=json"))
    names = {v["name"] for v in existing}
    if any(vm_name(cfg["run"], i) in names for i in idx):
        raise ValueError("instance name collision; prior VM preserved")
    # This is an existence check, not a swallowed ls error. Auth/network errors raise.
    listing = gcloud("storage", "ls", f"{BUCKET}/runs/")
    target = f"{BUCKET}/runs/{cfg['run']}"
    if any(line.strip().rstrip("/") == target or line.strip().startswith(target + "/")
           for line in listing.splitlines()):
        raise ValueError("output prefix collision; prior output preserved")


def dispatch_contract(cfg, created_at, bundle_sha256, i=0):
    if type(created_at) is not int or created_at <= 0 or not re.fullmatch(r"[a-f0-9]{64}", bundle_sha256):
        raise ValueError("invalid creation clock or bundle hash")
    total, work, reserve = lifetime(cfg)
    guard = cfg.get("credit_guard_epoch", CREDIT_GUARD_EPOCH)
    if type(guard) is not int or guard > CREDIT_GUARD_EPOCH or created_at + total > guard:
        raise ValueError("absolute credit guard")
    model_job = bool(assemble_shard(cfg, i).get("model"))
    return {"created_at": created_at, "work_stop_at": created_at + work,
            "run": cfg["run"], "shard": i, "attempt": vm_name(cfg["run"], i),
            "shutdown_at": created_at + total, "lifetime_seconds": total,
            "work_stop_seconds": work, "shutdown_reserve_seconds": reserve, "credit_guard_epoch": guard,
            "bundle_sha256": bundle_sha256, "weights_mirror_uri": cfg.get("weights_mirror_uri") if model_job else None,
            "weights_local_path": cfg.get("weights_local_path") if model_job else None, "allow_hf_download": False,
            "has_gpu": cfg["has_gpu"],
            "runtime_control_sha256": hashlib.sha256(json.dumps(assemble_shard(cfg, i)["runtime_control_command"], separators=(",", ":")).encode()).hexdigest(),
            "weights_manifest_uri": cfg.get("weights_manifest_uri") if model_job else None,
            "weights_manifest_sha256": cfg.get("weights_manifest_sha256") if model_job else None,
            "model_revision": assemble_shard(cfg, i).get("revision"),
            "tokenizer_revision": assemble_shard(cfg, i).get("tokenizer_revision"),
            "requirements_lock_sha256": cfg["requirements_lock_sha256"]}


def verify_instance(actual, cfg, name, zone, labels, metadata, deadline):
    def tail(value):
        return value.rsplit("/", 1)[-1]
    if actual["name"] != name or tail(actual["machineType"]) != cfg["machine"] or tail(actual["zone"]) != zone:
        raise ValueError("instance identity/shape mismatch")
    sched = actual["scheduling"]
    if sched.get("provisioningModel") != "SPOT" or sched.get("instanceTerminationAction") != "DELETE" or sched.get("onHostMaintenance") != "TERMINATE":
        raise ValueError("scheduling mismatch")
    if dt.datetime.fromisoformat(sched["terminationTime"].replace("Z", "+00:00")).timestamp() != deadline:
        raise ValueError("lifetime mismatch")
    if [s["email"] for s in actual["serviceAccounts"]] != [SA] or actual.get("labels") != labels:
        raise ValueError("service account/labels mismatch")
    got = {x["key"]: x["value"] for x in actual["metadata"]["items"]}
    if got != metadata:
        raise ValueError("metadata mismatch")
    if actual["networkInterfaces"][0].get("nicType") != cfg["nic_type"]:
        raise ValueError("NIC mismatch")
    disk = next(d for d in actual["disks"] if d.get("boot"))
    described = json.loads(gcloud("compute", "disks", "describe", tail(disk["source"]), f"--zone={zone}", "--format=json"))
    if tail(described["type"]) != cfg["boot_disk_type"] or int(described["sizeGb"]) != cfg["disk_gb"]:
        raise ValueError("boot disk mismatch")
    source = described["sourceImage"].split("/projects/", 1)[-1]
    expected_image = f"{cfg['image_project']}/global/images/{cfg['image_name']}"
    if source.removeprefix("projects/") != expected_image:
        raise ValueError("image mismatch")
    if "image_id" in cfg and described.get("sourceImageId") != cfg["image_id"]:
        raise ValueError("immutable image ID mismatch")
    accelerators = [{"type": tail(a["acceleratorType"]), "count": a["acceleratorCount"]}
                    for a in actual.get("guestAccelerators", [])]
    if sorted(accelerators, key=lambda a: a["type"]) != sorted(cfg["guest_accelerators"], key=lambda a: a["type"]):
        raise ValueError("accelerator mismatch")


def create_vm(cfg, i, shard_path, contract):
    validate_config(cfg)
    name, zone = vm_name(cfg["run"], i), cfg["zones"][0]
    expected_path = f"{BUCKET}/runs/{cfg['run']}/shards/{i:03d}"
    if shard_path != expected_path:
        raise ValueError("unexpected shard prefix")
    expected_contract = dispatch_contract(cfg, contract["created_at"], contract["bundle_sha256"], i)
    pin = contract.get("shard_sha256")
    if not isinstance(pin, str) or not re.fullmatch(r"[a-f0-9]{64}", pin):
        raise ValueError("trusted launch shard hash required")
    expected_contract["shard_sha256"] = pin
    if contract != expected_contract:
        raise ValueError("dispatch contract mismatch")
    if time.time() >= contract["work_stop_at"]:
        raise ValueError("creation clock exhausted")
    labels = {"app": "stateful-bench", "role": "lsr", "run": cfg["run"]}
    metadata = {"lsr-job": shard_path, "lsr-idle-min": str(cfg.get("idle_min", 30)),
                "lsr-grace-min": str(cfg.get("grace_min", 45)),
                **{f"lsr-{k.replace('_', '-')}": str(v).lower() if isinstance(v, bool) else str(v)
                   for k, v in contract.items() if v is not None}}
    metadata["lsr-created-epoch"] = str(contract["created_at"])
    metadata["lsr-accelerator"] = cfg.get("accelerator") or ("integrated" if contract["has_gpu"] else "")
    metadata["lsr-model-mirror"] = cfg.get("weights_local_path") or "/opt/lsr/offline-hf"
    metadata.update({"startup-script": (HERE / "vm_startup.sh").read_text(),
                     "shutdown-script": (HERE / "vm_shutdown.sh").read_text()})
    deadline = dt.datetime.fromtimestamp(contract["shutdown_at"], dt.timezone.utc).isoformat()
    args = ["compute", "instances", "create", name, f"--zone={zone}", f"--machine-type={cfg['machine']}",
            f"--service-account={SA}", "--scopes=https://www.googleapis.com/auth/cloud-platform",
            f"--image={cfg['image_name']}", f"--image-project={cfg['image_project']}",
            f"--boot-disk-size={cfg['disk_gb']}GB", f"--boot-disk-type={cfg['boot_disk_type']}",
            f"--network-interface=nic-type={cfg['nic_type']}",
            "--labels=" + ",".join(f"{k}={v}" for k,v in labels.items()),
            "--provisioning-model=SPOT", "--instance-termination-action=DELETE",
            f"--termination-time={deadline}", "--maintenance-policy=TERMINATE",
            "--metadata=" + ",".join(f"{k}={v}" for k,v in metadata.items() if k not in {"startup-script", "shutdown-script"}),
            f"--metadata-from-file=startup-script={HERE / 'vm_startup.sh'},shutdown-script={HERE / 'vm_shutdown.sh'}"]
    if cfg.get("accelerator"):
        t, c = cfg["accelerator"].split(":")
        args.append(f"--accelerator=type={t},count={c}")
    # GCE creation is atomic on name: a raced collision raises, never deletes an incumbent.
    gcloud(*args)
    try:
        actual = json.loads(gcloud("compute", "instances", "describe", name, f"--zone={zone}", "--format=json"))
        verify_instance(actual, cfg, name, zone, labels, metadata, contract["shutdown_at"])
    except Exception:
        gcloud("compute", "instances", "delete", name, f"--zone={zone}", "--quiet", check=False)
        raise
    with REGISTRY.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"run": cfg["run"], "shard": i, "vm": name, "zone": zone,
                            "usd_per_hour": cfg["usd_per_hour"], "rate_status": "VERIFIED" if cfg["rate_verified"] else "UNVERIFIED",
                            "shape_status": "VERIFIED" if cfg["shape_verified"] else "UNVERIFIED",
                            "machine": cfg["machine"], "accelerator": cfg.get("accelerator"),
                            "created": dt.datetime.fromtimestamp(contract["created_at"], dt.timezone.utc).isoformat(),
                            "ancillary_allowance_usd": ancillary(cfg), "dispatch": contract}) + "\n")
    print(f"verified {name} in {zone}; finite creation-clock deadline {deadline}")
    return zone


def expected_attempt(run, i):
    """Local owner-held launch journal is authority, never a downloaded receipt."""
    try:
        rows = [json.loads(line) for line in REGISTRY.read_text(encoding="utf-8").splitlines()]
        matches = [r["expected"] for r in rows if r.get("record_type") == "attempt_expected"
                   and r.get("run") == run and type(r.get("shard")) is int and r["shard"] == i]
        if len(matches) == 1:
            return matches[0]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def attempt_manifest(shard, raw):
    d = shard["dispatch"]
    return {"run": shard["run"], "shard": shard["shard"], "attempt": shard["attempt"],
            "shard_sha256": hashlib.sha256(raw).hexdigest(),
            "bundle_sha256": d["bundle_sha256"], "created_epoch": d["created_at"]}


def receipt_state(receipts, started=False, expected=None):
    """Failure dominates. Success requires both work and finalized payload custody."""
    parsed = {}
    for name in ("runtime_receipt.json", "shutdown_receipt.json"):
        raw = receipts.get(name)
        if raw is None:
            continue
        try:
            r = json.loads(raw)
            if not isinstance(r, dict) or type(r.get("rc")) is not int or r["rc"] < 0 or r.get("status") not in {"completed", "failed"} or type(r.get("finalized")) is not bool or not isinstance(r.get("reason"), str) or not isinstance(r.get("hashes"), dict) or not isinstance(r.get("clock"), dict):
                raise ValueError("invalid receipt")
            if (r["rc"] == 0) != (r["status"] == "completed"):
                raise ValueError("inconsistent receipt")
            parsed[name] = r
        except (ValueError, TypeError):
            return f"failed(malformed:{name})"
    for name, r in parsed.items():
        if r["rc"] or r["status"] == "failed":
            return f"failed({name}:{r['reason']},rc={r['rc']})"
    if len(parsed) != 2:
        return "incomplete(missing-receipt)" if receipts else ("started" if started else "pending")
    # A finalized shell receipt attests to custody of the deferred runtime receipt.
    if not parsed["shutdown_receipt.json"]["finalized"]:
        return "incomplete(custody-unfinalized)"
    for key in ("bundle_sha256",):
        values = [r.get("hashes", {}).get(key) for r in parsed.values()]
        if not all(isinstance(v, str) and re.fullmatch(r"[a-f0-9]{64}", v) for v in values) or len(set(values)) != 1:
            return "failed(receipt-binding)"
    clocks = [r.get("clock", {}).get("created_epoch") for r in parsed.values()]
    if any(type(c) is not int or c <= 0 for c in clocks) or len(set(clocks)) != 1:
        return "failed(receipt-clock-binding)"
    if not isinstance(expected, dict):
        return "failed(expected-attempt-missing)"
    if (not isinstance(expected.get("run"), str) or type(expected.get("shard")) is not int
            or expected["shard"] < 0 or not isinstance(expected.get("attempt"), str)
            or type(expected.get("created_epoch")) is not int or expected["created_epoch"] <= 0
            or any(not isinstance(expected.get(k), str) or not re.fullmatch(r"[a-f0-9]{64}", expected[k])
                   for k in ("shard_sha256", "bundle_sha256"))):
        return "failed(expected-attempt-malformed)"
    runtime, shutdown = parsed["runtime_receipt.json"], parsed["shutdown_receipt.json"]
    for r in (runtime, shutdown):
        identity = r.get("identity")
        if not isinstance(identity, dict) or any(type(identity.get(k)) is not type(expected[k])
                or identity[k] != expected[k] for k in ("run", "shard", "attempt")):
            return "failed(receipt-identity-binding)"
        if any(r["hashes"].get(k) != expected[k] for k in ("shard_sha256", "bundle_sha256")):
            return "failed(receipt-attempt-binding)"
        if r["clock"]["created_epoch"] != expected["created_epoch"]:
            return "failed(receipt-attempt-clock-binding)"
    # Bind custody to the exact finalized runtime bytes, not just common pins.
    if shutdown["hashes"].get("runtime_receipt_sha256") != hashlib.sha256(
            receipts["runtime_receipt.json"].encode("utf-8")).hexdigest():
        return "failed(receipt-custody-binding)"
    return "done"


def shard_state(run, i):
    prefix = f"{BUCKET}/runs/{run}/shards/{i:03d}/out/"
    listing = gcloud("storage", "ls", prefix, check=False)
    names = {line.strip().rsplit("/", 1)[-1] for line in listing.splitlines()}
    receipts = {name: gcloud("storage", "cat", prefix + name)
                for name in ("runtime_receipt.json", "shutdown_receipt.json") if name in names}
    if "done.json" in names and not receipts:
        return "incomplete(legacy-receipt)"
    return receipt_state(receipts, bool(names), expected_attempt(run, i))


def launch(cfg_path, only=None, dry=False):
    cfg = json.loads(Path(cfg_path).read_text())
    idx, estimate = launch_plan(cfg, only)
    print(f"run {cfg['run']}: {len(idx)} VMs; aggregate estimate ${estimate:.4f}; "
          f"rate {'VERIFIED' if cfg['rate_verified'] else 'UNVERIFIED'}; "
          f"shape {'VERIFIED' if cfg['shape_verified'] else 'UNVERIFIED'}")
    if dry:
        return
    refuse_collisions(cfg, idx)
    # Reserve the run prefix atomically. Losing racers cannot upload/delete anything.
    with tempfile.TemporaryDirectory() as td:
        claim = Path(td) / "claim.json"
        claim.write_text(json.dumps({"run": cfg["run"], "created": now()}))
        gcloud("storage", "cp", str(claim), f"{BUCKET}/runs/{cfg['run']}/claim.json", "--if-generation-match=0")
        digest = upload_bundle(cfg["run"], cfg)
        gcloud("storage", "cp", str(cfg_path), f"{BUCKET}/runs/{cfg['run']}/config.json", "--if-generation-match=0")
        for i in idx:
            contract = dispatch_contract(cfg, int(time.time()), digest, i)
            shard = assemble_shard(cfg, i)
            shard["job_stop_seconds"] = contract["work_stop_seconds"]
            shard["requirements"] = cfg["requirements"]
            shard["dispatch"] = contract
            path = Path(td) / f"{i:03d}.json"
            raw = (json.dumps(shard, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            path.write_bytes(raw)
            expected = attempt_manifest(shard, raw)
            # Persist before upload/create: failed launches retain their intended attempt.
            with REGISTRY.open("a", encoding="utf-8") as journal:
                journal.write(json.dumps({"record_type": "attempt_expected", "run": cfg["run"],
                                          "shard": i, "expected": expected}) + "\n")
            prefix = f"{BUCKET}/runs/{cfg['run']}/shards/{i:03d}"
            gcloud("storage", "cp", str(path), f"{prefix}/shard.json", "--if-generation-match=0")
            # Transport the journal hash outside the hashed shard (no self-hash).
            launch_contract = {**contract, "shard_sha256": expected["shard_sha256"]}
            create_vm(cfg, i, prefix, launch_contract)


def list_vms(run=None):
    flt = "labels.app=stateful-bench AND labels.role=lsr" + (f" AND labels.run={run.lower()}" if run else "")
    vms = json.loads(gcloud("compute", "instances", "list", f"--filter={flt}", "--format=json"))
    return [v for v in vms if v["name"].startswith(PREFIX)]


def status(run=None):
    for v in list_vms(run):
        print(f"{v['name']:45s} {v['zone'].split('/')[-1]:15s} {v['status']:12s} {v['labels'].get('run')}")
    if run:
        cfg = json.loads(gcloud("storage", "cat", f"{BUCKET}/runs/{run}/config.json"))
        with ThreadPoolExecutor(2) as ex:
            states = list(ex.map(lambda i: shard_state(run, i), range(len(cfg["shards"]))))
        for i, s in enumerate(states):
            print(f"  shard {i:03d}: {s}")


def relaunch(run, rebundle=False, retry_failed=False):
    raise ValueError("relaunch refused: preserve prior run and request a fresh gated unique run name")


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
    cfg = json.loads((dst / "config.json").read_text())
    for i in range(len(cfg["shards"])):
        root = dst / "shards" / f"{i:03d}" / "out"
        receipts = {p.name: p.read_bytes().decode("utf-8") for p in root.glob("*_receipt.json")}
        print(f"  shard {i:03d}: {receipt_state(receipts, root.exists(), expected_attempt(run, i))}")


def spend():
    """Planning estimates only. Missing terminal evidence uses the full bounded lifetime."""
    rows = [json.loads(l) for l in REGISTRY.read_text().splitlines()] if REGISTRY.exists() else []
    rows = [r for r in rows if r.get("record_type") != "attempt_expected"]
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
            end = c + dt.timedelta(seconds=r.get("dispatch", {}).get("lifetime_seconds", 0))
            if end == c:
                raise ValueError("UNKNOWN-HIGH: legacy missing terminal evidence and lifetime")
        usd = max(0.0, (end - c).total_seconds() / 3600) * r["usd_per_hour"] + r.get("ancillary_allowance_usd", 0)
        by_run[r["run"]] = by_run.get(r["run"], 0.0) + usd
        total += usd
    lines = ["# LSR GCP spend ledger", "", f"Updated {now()}. Planning estimate from VM lifetimes x recorded rate; missing ends charge full bounded lifetime. "
             "Rates may be UNVERIFIED; declared ancillary allowances included (not a billing reconciliation). Owner authorization 2026-10-15: per-program spend caps removed; compute credit only; Scale Y1 - AI expires 2026-10-17.", "",
             "| run | VMs launched | est. USD | rate provenance |", "|---|---:|---:|---|"]
    for run in sorted(by_run):
        provenance=",".join(sorted({r.get("rate_status","UNKNOWN") for r in rows if r["run"]==run}))
        lines.append(f"| {run} | {sum(1 for r in rows if r['run'] == run)} | {by_run[run]:.2f} | {provenance} |")
    lines += [f"| **total** | {len(rows)} | **{total:.2f}** | mixed; see rows |", "",
              f"Live VMs now: {sum(1 for v in live.values() if v['status'] == 'RUNNING')}; burn rate now: "
              f"${sum(next((r['usd_per_hour'] for r in reversed(rows) if r['vm'] == n), 0) for n, v in live.items() if v['status'] == 'RUNNING'):.1f}/h"]
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
