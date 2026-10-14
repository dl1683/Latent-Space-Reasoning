"""Finite VM work and custody, stdlib only. No external effect on import/check.

Production: --run --shard FILE --out DIR --created-epoch UNIX_SECONDS
            --bundle-sha256 SHA256 --job gs://.../shards/NNN
Shard must supply runtime_control_command (argv, no shell), whose zero exit
certifies the separate operation-specific positive control before work starts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import importlib.metadata as metadata
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

LIFETIME = 14_400
WORK_STOP = 13_500
RESERVE = 900
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_DATASETS_OFFLINE": "1", "PIP_NO_INDEX": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}


class Stop(RuntimeError):
    pass


def heartbeat_epoch(value):
    """Malformed progress fails closed, including JSON NaN and numeric strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def verify_deployment(cfg, raw, args, runtime):
    """No effects except local reads/progress; independent launch authority is mandatory."""
    checks = runtime.verification
    d = cfg.get("dispatch")
    if not isinstance(d, dict):
        raise Stop("dispatch_contract")
    expected = {"run": getattr(args, "expected_run", None),
                "shard": getattr(args, "expected_shard", None),
                "attempt": getattr(args, "expected_attempt", None)}
    if (not isinstance(expected["run"], str) or not expected["run"]
            or type(expected["shard"]) is not int or expected["shard"] < 0
            or not isinstance(expected["attempt"], str) or not expected["attempt"]):
        raise Stop("launch_identity")
    runtime.identity.update(expected)
    pin = getattr(args, "expected_shard_sha256", None)
    if not valid_hash(pin) or hashlib.sha256(raw).hexdigest() != pin:
        raise Stop("shard_hash")
    for key, value in expected.items():
        if any(type(part.get(key)) is not type(value) or part[key] != value for part in (cfg, d)):
            raise Stop("shard_identity")
    if (type(args.created_epoch) is not int or args.created_epoch <= 0
            or not valid_hash(args.bundle_sha256)
            or d.get("bundle_sha256") != args.bundle_sha256
            or type(d.get("created_at")) is not int or d["created_at"] != args.created_epoch):
        raise Stop("launch_binding")
    CREDIT_GUARD_EPOCH = 1792195200  # same administrative cutoff as dispatcher/bootstrap
    values = tuple(d.get(k) for k in ("lifetime_seconds", "work_stop_seconds", "shutdown_reserve_seconds"))
    if (any(type(v) is not int or v <= 0 for v in values) or values[0] not in (3600,10800,14400)
            or values[2] < 900 or values[1] + values[2] != values[0]):
        raise Stop("dispatch_clock")
    total, work, reserve = values
    guard = d.get("credit_guard_epoch", CREDIT_GUARD_EPOCH)
    if type(guard) is not int or guard > CREDIT_GUARD_EPOCH or args.created_epoch + total > guard:
        raise Stop("credit_guard")
    for field, value in (("work_stop_at", args.created_epoch + work), ("shutdown_at", args.created_epoch + total)):
        if type(d.get(field)) is not int or d[field] != value:
            raise Stop("dispatch_clock")
    runtime.lifetime, runtime.work_stop, runtime.reserve = total, work, reserve
    checks["identity"] = dict(expected)
    control = cfg.get("runtime_control_command")
    if (not isinstance(control, list) or not control
            or any(not isinstance(x, str) or not x.strip() or "\x00" in x for x in control)):
        raise Stop("control_missing")
    def both(field, actual):
        if not valid_hash(actual) or any(part.get(field) != actual for part in (cfg, d)):
            raise Stop(field)
        checks[field] = actual
    both("runtime_control_sha256", hashlib.sha256(json.dumps(control, separators=(",", ":")).encode()).hexdigest())
    lock_path = Path(getattr(args, "requirements_path", "/opt/lsr/requirements.txt"))
    if lock_path.is_symlink():
        raise Stop("requirements_symlink")
    lock = lock_path.read_bytes()
    both("requirements_lock_sha256", hashlib.sha256(lock).hexdigest())
    reqs = cfg.get("requirements")
    if (not isinstance(reqs, list) or not reqs or not all(isinstance(x, str) for x in reqs)
            or ("\n".join(reqs) + "\n").encode() != lock):
        raise Stop("requirements_bytes")
    normalize = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    locked, installed = {}, {}
    for line in lock.decode("utf-8").splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
        if not match or normalize(match[1]) in locked:
            raise Stop("requirements_format")
        locked[normalize(match[1])] = match[2]
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name or normalize(name) in installed:
            raise Stop("installed_metadata")
        installed[normalize(name)] = dist.version
    checks["installed_dependencies"] = {"expected": locked, "actual": installed,
        "missing": sorted(locked.keys() - installed.keys()),
        "unexpected": sorted(installed.keys() - locked.keys()),
        "mismatched": sorted(k for k in locked.keys() & installed.keys() if locked[k] != installed[k])}
    # OS-provided distributions in the image freeze are part of the full contract.
    # No image-id waiver: ALL installed metadata must match, including extras.
    if installed != locked:
        raise Stop("installed_dependencies")
    if d.get("allow_hf_download") is not False:
        raise Stop("download_fallback")
    if not cfg.get("model"):
        if ("model" in cfg and cfg["model"] not in (None, "")
                or any("weights_manifest_sha256" not in p or p["weights_manifest_sha256"] is not None for p in (cfg, d))
                or any(d.get(k) is not None for k in ("weights_local_path", "weights_mirror_uri", "weights_manifest_uri", "model_revision", "tokenizer_revision"))):
            raise Stop("no_model_contract")
        checks["assets"] = {"model": None, "manifest": None}
        return control
    if not isinstance(cfg["model"], str) or not cfg["model"]:
        raise Stop("model_identity")
    for field, target in (("revision", "model_revision"), ("tokenizer_revision", "tokenizer_revision")):
        value = cfg.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value) or d.get(target) != value:
            raise Stop("asset_revision")
    if d.get("weights_local_path") != "/opt/lsr/weights":
        raise Stop("asset_root")
    root = Path(getattr(args, "weights_path", d["weights_local_path"]))
    if root.is_symlink() or not root.is_dir():
        raise Stop("asset_root")
    # Symlinks anywhere below the root are prohibited, not merely listed links.
    for path in root.rglob("*"):
        runtime.progress()
        if path.is_symlink():
            raise Stop("asset_symlink")
    manifest = root / "_manifest_sha256.txt"
    manifest_raw = manifest.read_bytes()
    both("weights_manifest_sha256", hashlib.sha256(manifest_raw).hexdigest())
    if not (root / "_COMPLETE").is_file():
        raise Stop("asset_incomplete")
    lines = manifest_raw.decode("utf-8").splitlines()
    if not lines:
        raise Stop("asset_manifest_format")
    listed = set()
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise Stop("asset_manifest_format")
        name = match[2]
        if name.startswith("./"):
            name = name[2:]  # canonical sha256sum output; raw manifest remains pin-bound
        parts = name.split("/")
        if (any(p in ("", ".", "..") for p in parts) or "\\" in name or ":" in name
                or name in listed or name in ("_manifest_sha256.txt", "_COMPLETE")):
            raise Stop("asset_path")
        listed.add(name)
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise Stop("asset_missing")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                runtime.progress()
                chunk = stream.read(4 * 1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        if hasher.hexdigest() != match[1]:
            raise Stop("asset_content")
    # All files must be listed except the two custody markers; covers new weight formats.
    if {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} != listed | {"_manifest_sha256.txt", "_COMPLETE"}:
        raise Stop("asset_unlisted")
    if not {"config.json", "tokenizer.json"}.issubset(listed) or any((root/n).stat().st_size == 0 for n in listed):
        raise Stop("asset_required")
    try:
        if not isinstance(json.loads((root/"config.json").read_text()), dict):
            raise ValueError("config")
        if not isinstance(json.loads((root/"tokenizer.json").read_text()), dict):
            raise ValueError("tokenizer")
        index = "model.safetensors.index.json"
        if index in listed:
            weight_map = json.loads((root/index).read_text())["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("empty index")
            weights = set(weight_map.values())
            if not weights.issubset(listed) or any(not n.endswith(".safetensors") for n in weights):
                raise ValueError("missing indexed weight")
        elif "model.safetensors" in listed:
            weights = {"model.safetensors"}
        else:
            raise ValueError("unsupported weights")
        if {n for n in listed if n.endswith((".safetensors", ".bin"))} != weights:
            raise ValueError("unindexed weights")
    except (ValueError, KeyError, TypeError):
        raise Stop("asset_inventory")
    checks["assets"] = {"model": cfg["model"], "model_revision": cfg["revision"],
                        "tokenizer_revision": cfg["tokenizer_revision"], "files": sorted(listed)}
    return control


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


class Runtime:
    """Injected effects let tests exercise this same production state machine."""

    def __init__(self, out, created, hashes, *, clock=time.time, sleep=time.sleep,
                 spawn=None, upload=None, heartbeat=None, stall_seconds=300, publish=None, deferred=False):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.created = created
        self.lifetime, self.work_stop, self.reserve = LIFETIME, WORK_STOP, RESERVE
        self.hashes = hashes
        self.clock, self.sleep = clock, sleep
        self.spawn = spawn
        self.upload = upload
        self.publish = publish
        self.deferred = deferred
        self.active_stage = "bootstrap"
        self.heartbeat = heartbeat
        self.stall_seconds = stall_seconds
        self.identity, self.verification = {}, {}

    def progress(self):
        # Actual validation work advances bootstrap progress; never a blind timer.
        now = self.clock()
        if now < self.created or now >= self.created + self.work_stop:
            raise Stop("deadline")
        write_json(self.out / "heartbeat_bootstrap", {"stage": "bootstrap", "epoch": now})
        self.guard("bootstrap")

    def guard(self, stage):
        now = self.clock()
        if self.created <= 0 or now < self.created:
            raise Stop("creation_clock")
        if now >= self.created + self.work_stop:
            raise Stop("deadline")
        beat = heartbeat_epoch(self.heartbeat())
        if beat is None or beat > now or now - beat >= self.stall_seconds:
            raise Stop("stall")

    def wait(self, argv, stage):
        self.active_stage = stage
        write_json(self.out / "runtime_stage.json", {"stage": stage, "epoch": self.clock()})
        self.guard(stage)
        proc = self.spawn(argv)
        try:
            while True:
                self.guard(stage)
                rc = proc.poll()
                if rc is not None:
                    if rc:
                        raise Stop(stage)
                    phase_out = self.out / stage
                    phase_out.mkdir(exist_ok=True)
                    phase_receipt = self.receipt("completed", 0)
                    phase_receipt["phase"] = stage
                    phase_receipt["argv_sha256"] = hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest()
                    write_json(phase_out / "terminal_receipt.json", phase_receipt)
                    return
                self.sleep(1)
        except Stop as exc:
            phase_out = self.out / stage
            phase_out.mkdir(exist_ok=True)
            phase_receipt = self.receipt(str(exc), 1)
            phase_receipt["phase"] = stage
            write_json(phase_out / "terminal_receipt.json", phase_receipt)
            raise
        finally:
            # Group kill includes descendants; never leave a fitting child alive.
            proc.stop()

    def execute(self, bootstrap, control, work):
        reason, rc = "completed", 0
        stage = "bootstrap"
        try:
            self.guard(stage)
            bootstrap()
            stage = "control"
            self.wait(control, stage)
            stage = "work"
            self.wait(work, stage)
        except Stop as exc:
            reason, rc = str(exc), 1
        except Exception:
            reason, rc = stage, 1
        receipt = self.receipt(reason, rc)
        def persist(value):
            try:
                write_json(self.out / "runtime_receipt.json", value)
                return True
            except Exception:
                # Atomic publication failed: never mask this with a successful upload.
                fallback = self.receipt("receipt", 1)
                fallback["prior_reason"] = reason
                try:
                    (self.out / "runtime_receipt.json").write_text(json.dumps(fallback) + "\n", encoding="utf-8")
                except OSError:
                    pass  # Total disk failure cannot guarantee any local receipt.
                return False
        if not persist(receipt):
            return 1
        if self.deferred:
            return rc  # Shell parent stops its session, then performs all custody.
        try:
            budget = min(self.reserve - 30, int(self.created + self.lifetime - self.clock() - 30))
            if budget <= 0:
                raise Stop("custody_deadline")
            self.upload(self.out, budget)
            receipt["finalized"] = True
            if not persist(receipt):
                return 1
            if self.publish is not None:
                budget = min(self.reserve - 30, int(self.created + self.lifetime - self.clock() - 30))
                if budget <= 0:
                    raise Stop("custody_deadline")
                self.publish(self.out / "runtime_receipt.json", budget)
        except Exception:
            rc = 1
            receipt = self.receipt("custody", rc)
            receipt["prior_reason"] = reason
            persist(receipt)
        return rc

    def receipt(self, reason, rc):
        now = self.clock()
        return {"status": "completed" if rc == 0 else "failed", "reason": reason, "rc": rc,
                "finalized": False, "hashes": self.hashes, "identity": self.identity,
                "verification": self.verification,
                "clock": {"created_epoch": self.created, "now_epoch": now,
                          "elapsed_seconds": now - self.created,
                          "work_stop_epoch": self.created + self.work_stop,
                          "lifetime_end_epoch": self.created + self.lifetime,
                          "reserve_seconds": self.reserve}}


def restrict_worker_writes(out):
    """Linux Landlock ABI >=3: inherited read-only code/assets/dependencies.

    Only the attempt output tree and existing character devices are writable.
    No fallback on an unsupported kernel. Administrative writers outside this
    process tree must still be excluded by the operator's exclusive VM custody.
    """
    import ctypes
    import platform
    import stat
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise Stop("custody_platform")
    root = Path(out)
    if root.is_symlink() or not root.is_dir() or root.resolve() != root.absolute():
        raise Stop("custody_output")
    for entry in sys.path:
        if Path(entry or ".").resolve().is_relative_to(root):
            raise Stop("custody_code_in_output")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    def call(number, *args):
        result = libc.syscall(number, *args)
        if result < 0:
            raise Stop("custody_landlock:" + str(ctypes.get_errno()))
        return result
    abi = call(444, 0, 0, 1)
    if abi < 3:
        raise Stop("custody_landlock_abi")
    # WRITE_FILE, REMOVE_DIR/FILE, MAKE_*, REFER and TRUNCATE. Read/execute
    # remain unrestricted; writes outside allowed roots fail even as uid 0.
    access = (1 << 1) | sum(1 << bit for bit in range(4, 15))
    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]
    class Beneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]
    rule = Ruleset(access)
    fd = call(444, ctypes.byref(rule), ctypes.sizeof(rule), 0)
    try:
        paths = [(root, access)]
        # CUDA/terminal/null devices are runtime interfaces, never asset roots.
        paths.extend((path, 1 << 1) for path in Path("/dev").rglob("*")
                     if not path.is_symlink() and stat.S_ISCHR(path.stat().st_mode))
        for path, rights in paths:
            handle = os.open(path, os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                item = Beneath(rights, handle)
                call(445, fd, 1, ctypes.byref(item), 0)
            finally:
                os.close(handle)
        if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
            raise Stop("custody_no_new_privs")
        call(446, fd, 0)
    finally:
        os.close(fd)


class Child:
    def __init__(self, argv, *, own_session=None):
        self.own_session = (os.environ.get("LSR_WORK_GROUP") != "1") if own_session is None else own_session
        env = {**os.environ, **OFFLINE}
        root = env.get("LSR_CUSTODY_WRITE_ROOT") if own_session is None else None
        if root:
            argv = [sys.executable, "-B", "-m", __package__ + ".s0_runtime",
                    "--custody-exec", root, "--", *argv]
        self.proc = subprocess.Popen(argv, env=env, start_new_session=self.own_session)

    def poll(self):
        return self.proc.poll()

    def stop(self):
        try:
            if self.own_session:
                os.killpg(self.proc.pid, signal.SIGKILL)
            elif self.proc.poll() is None:
                self.proc.kill()
        except ProcessLookupError:
            pass
        self.proc.wait(timeout=10)


def bounded_command(argv, budget):
    # Custody has its own session, killed in full on timeout AND successful exit.
    child = Child(argv, own_session=True)
    try:
        rc = child.proc.wait(timeout=budget)
        if rc:
            raise subprocess.CalledProcessError(rc, argv)
    finally:
        child.stop()


def run(args):
    out = Path(args.out)
    def heartbeat():
        try:
            # Worker emits progress at control/load/fit boundaries, never a blind timer.
            # Until that producer is integrated, legacy result heartbeat remains supported.
            for phase in ("fit", "load", runtime.active_stage):
                path = out / ("heartbeat_" + phase)
                if path.exists():
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if value["stage"] == runtime.active_stage:
                        return heartbeat_epoch(value["epoch"])
            return (out / "heartbeat").stat().st_mtime
        except (FileNotFoundError, ValueError, TypeError, KeyError):
            return None

    def upload(path, budget):
        bounded_command(["gcloud", "storage", "rsync", "-r", str(path), args.job + "/out"], budget)

    def publish(path, budget):
        bounded_command(["gcloud", "storage", "cp", str(path), args.job + "/out/runtime_receipt.json"], budget)

    # Even malformed shard/control metadata gets a terminal receipt.
    cfg = {}
    hashes = {"runtime_sha256": digest(__file__), "bundle_sha256": args.bundle_sha256}
    def bootstrap():
        if len(args.bundle_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.bundle_sha256):
            raise Stop("bundle_hash")
        raw = Path(args.shard).read_bytes()
        hashes["shard_sha256"] = hashlib.sha256(raw).hexdigest()
        cfg.update(json.loads(raw))
        control = cfg.get("runtime_control_command")
        if not isinstance(control, list) or not control or not all(isinstance(x, str) and x for x in control):
            raise Stop("control_missing")
        control_argv.extend(verify_deployment(cfg, raw, args, runtime))
        if len(raw) > 60000:
            raise Stop("shard_argv_limit")
        work_argv.extend([sys.executable, "-B", "-m", __package__ + ".run", "--verified-json",
                          hashes["shard_sha256"], raw.decode("utf-8"), str(out)])
        runtime.verification["work_input"] = {"transport": "immutable-argv", "sha256": hashes["shard_sha256"]}
    control_argv = []
    work_argv = []
    runtime = Runtime(out, args.created_epoch, hashes, spawn=Child, upload=upload,
                      heartbeat=heartbeat, stall_seconds=300, publish=publish,
                      deferred=os.environ.get("LSR_CUSTODY_DEFERRED") == "1")
    def interrupted(_signum, _frame):
        raise Stop("signal")
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    prior_custody = os.environ.get("LSR_CUSTODY_WRITE_ROOT")
    os.environ["LSR_CUSTODY_WRITE_ROOT"] = str(out.resolve())
    try:
        return runtime.execute(bootstrap, control_argv,
                               work_argv)
    finally:
        if prior_custody is None:
            os.environ.pop("LSR_CUSTODY_WRITE_ROOT", None)
        else:
            os.environ["LSR_CUSTODY_WRITE_ROOT"] = prior_custody
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def fault_checks(runtime_type=Runtime, verbose=True):
    """Real state machine; fake time/process/storage, no network or models."""
    class FakeChild:
        def __init__(self, rc):
            self.rc, self.stopped = rc, False

        def poll(self):
            return self.rc

        def stop(self):
            self.stopped = True
            self.rc = -9

    for fault in ("bootstrap", "control", "custody", "stall", "deadline", "running_stall", "running_deadline", "happy"):
        with tempfile.TemporaryDirectory() as td:
            now = [1000.0 + (WORK_STOP if fault == "deadline" else 0)]
            children, uploads = [], []
            def forbidden(*_args, **_kwargs):
                raise AssertionError("real external effect forbidden")
            def spawn(argv):
                rc = 1 if fault == "control" else None if fault.startswith("running_") else 0
                child = FakeChild(rc)
                children.append(child)
                return child
            def upload(path, budget):
                assert 0 < budget <= 870
                uploads.append(json.loads((path / "runtime_receipt.json").read_text()))
                if fault == "custody":
                    raise OSError("injected upload failure")
            def bootstrap():
                if fault == "bootstrap":
                    raise OSError("injected bootstrap failure")
            def sleep(_seconds):
                now[0] += WORK_STOP if fault == "running_deadline" else 301
            hashes = {"runtime_sha256": digest(__file__), "bundle_sha256": "a" * 64,
                      "shard_sha256": "b" * 64}
            runtime = runtime_type(td, 1000, hashes, clock=lambda: now[0], sleep=sleep,
                                   spawn=spawn, upload=upload,
                                   heartbeat=lambda: None if fault == "stall" else 1000.0)
            # Any accidental real subprocess/gsutil path fails the check immediately.
            from unittest.mock import patch
            with patch.object(subprocess, "Popen", forbidden), patch.object(subprocess, "run", forbidden):
                rc = runtime.execute(bootstrap, ["fake-control"], ["fake-work"])
            receipt = json.loads((Path(td) / "runtime_receipt.json").read_text())
            expected = "stall" if fault == "running_stall" else "deadline" if fault == "running_deadline" else fault
            assert (rc == 0) == (fault == "happy"), (fault, rc, receipt)
            assert receipt["status"] == ("completed" if fault == "happy" else "failed")
            assert receipt["reason"] == ("completed" if fault == "happy" else expected), receipt
            assert receipt["rc"] == rc and receipt["hashes"] == hashes
            assert receipt["clock"]["lifetime_end_epoch"] - receipt["clock"]["work_stop_epoch"] == RESERVE
            assert uploads, "custody never attempted"
            if fault.startswith("running_"):
                assert children and all(p.stopped for p in children)
            if verbose:
                print(f"PASS {fault} child_exit={rc} terminal={receipt['status']} reason={receipt['reason']}")


def custody_checks():
    """Actual kernel enforcement at exact receiver loader body; fixture loaders.

    No torch/weights/network. The production loader function is AST-extracted;
    its pretrained call is a local finite byte-reader/mutation fixture.
    """
    if os.name == "nt":
        repo = Path(__file__).resolve().parents[3].as_posix()
        posix = "/mnt/" + repo[0].lower() + repo[2:]
        subprocess.run(["wsl", "-d", "Ubuntu", "--cd", posix, "env", "OMP_NUM_THREADS=2",
                        "MKL_NUM_THREADS=2", "python3", "-B", "-c",
                        "from experiments.cloud.lsr_jobs.s0_runtime import custody_checks; custody_checks()"], check=True)
        return
    script = r'''
import ast, copy, hashlib, json, math, os, re, socket, sys, tempfile, types
from pathlib import Path
from contextlib import nullcontext
from experiments.cloud.lsr_jobs.s0_runtime import restrict_worker_writes
def forbidden(*args,**kwargs):raise AssertionError('external effect forbidden')
socket.socket=socket.create_connection=forbidden
source=Path('experiments/cloud/lsr_jobs/gemma4_receiver.py').read_text()
tree=ast.parse(source)
body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('_placement','_load_original','_require')]
with nullcontext(sys.argv[1]) as td:
    root=Path(td);out=root/'out';out.mkdir();assets=root/'weights';assets.mkdir()
    target=assets/'model.safetensors';target.write_bytes(b'bounded fixture, not model weights')
    code=root/'runner.py';code.write_bytes(b'fixture code');deps=root/'dependency.py';deps.write_bytes(b'fixture dependency')
    paths=[target,code,deps];expected={p:p.read_bytes() for p in paths};events=[]
    # Prove the attempted write actually works without enforcement.
    target.write_bytes(b'CHANGED');assert target.read_bytes()!=expected[target];target.write_bytes(expected[target])
    class Model:
        def state_dict(self):return {}
        def parameters(self):return []
        def named_parameters(self,**kw):return []
        def eval(self):return self
        def requires_grad_(self,_):return self
    class Loader:
        def __new__(cls,config):return Model()
        @classmethod
        def from_pretrained(cls,path,**kw):
            assert Path(path)==assets
            for p in paths:
                for operation in (lambda p=p:p.write_bytes(b'CHANGED'),lambda p=p:p.unlink(),
                                  lambda p=p:p.rename(out/p.name)):
                    try:operation()
                    except PermissionError:events.append('denied')
                    else:raise AssertionError('loader-boundary mutation admitted')
                assert p.read_bytes()==expected[p]
            return Model(),{}
    config=types.SimpleNamespace(model_type='gemma4',tie_word_embeddings=False,
                                 text_config=types.SimpleNamespace(tie_word_embeddings=False))
    transformers=types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**kw:'fixture-tokenizer'),
        Gemma4Config=types.SimpleNamespace(from_pretrained=lambda *a,**kw:config),Gemma4ForConditionalGeneration=Loader)
    sys.modules['accelerate']=types.SimpleNamespace(init_empty_weights=nullcontext)
    sys.modules['transformers']=transformers
    scope=dict(Path=Path,copy=copy,re=re,math=math,torch=types.SimpleNamespace(bfloat16='fixture-dtype'),
        _offline=lambda:None,_checkpoint_keys=lambda root:set(),_key_contract=lambda *a:set(),_check_tied_values=lambda *a:None)
    def verified(cfg,path):
        for p in paths:assert p.read_bytes()==expected[p]
        return ['fixture']
    scope['_validate_staged']=verified
    exec(compile(ast.Module(body=body,type_ignores=[]),'<exact receiver loader AST>','exec'),scope)
    cfg=dict(model='google/gemma-4-31B',model_local='/opt/lsr/weights',revision='5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89',
        tokenizer_revision='5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89',allow_hf_download=False,
        placement=dict(device_map={'':'cuda:0'},measurement_id='fixture-only',min_free_fraction=.1),dtype='bfloat16',attn_implementation='eager')
    # The exact identity guard is retained; resolve only the staged fixture path.
    scope['Path']=lambda value: assets if value=='/opt/lsr/weights' else Path(value)
    restrict_worker_writes(out)
    tok,model,receipt=scope['_load_original'](cfg)
    assert tok=='fixture-tokenizer' and len(events)==9
    (out/'receipt.json').write_text(json.dumps(dict(mutations_denied=len(events),unchanged_bytes=True)))
    assert json.loads((out/'receipt.json').read_text())['unchanged_bytes']
    print('PASS exact receiver loading-boundary custody: 9 write/unlink/rename mutations denied; unchanged assets/code/dependencies readable; output writable')
    # Cleanup is owned by unsandboxed parent; this child cannot remove assets.
    os._exit(0)
'''
    # Child must not inherit arbitrary write handles. No network or cloud effect.
    with tempfile.TemporaryDirectory() as td:
        result = subprocess.run([sys.executable, "-B", "-u", "-c", script, td], capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode:
        raise AssertionError(result.stderr)
    # Exercise actual production Child wrapping/exec and inherited enforcement.
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); out = root / "out"; out.mkdir()
        asset = root / "asset"; asset.write_bytes(b"unchanged")
        worker = """import sys, subprocess, socket
from pathlib import Path
def forbidden(*args,**kwargs):raise AssertionError('external effect forbidden')
socket.socket=socket.create_connection=forbidden
asset, out = map(Path, sys.argv[1:])
try: asset.write_bytes(b'CHANGED')
except PermissionError: pass
else: raise AssertionError('production Child did not enforce custody')
assert asset.read_bytes()==b'unchanged'
child=subprocess.run([sys.executable,'-B','-c',"from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b'CHANGED')",str(asset)],capture_output=True)
assert child.returncode!=0 and b'PermissionError' in child.stderr
(out/'worker.json').write_text('output permitted')
print('PASS production Child custody wrapper + exec + inherited descendant write refusal',flush=True)
"""
        with patch.dict(os.environ, {"LSR_CUSTODY_WRITE_ROOT": str(out)}):
            child = Child([sys.executable, "-B", "-c", worker, str(asset), str(out)])
            try:
                assert child.proc.wait(timeout=30) == 0
                assert asset.read_bytes() == b"unchanged" and (out / "worker.json").is_file()
            finally:
                child.stop()
        with patch.dict(os.environ, {"LSR_CUSTODY_WRITE_ROOT": str(root / "absent")}):
            child = Child([sys.executable, "-B", "-c", "raise AssertionError('unguarded command executed')"])
            try:
                assert child.proc.wait(timeout=30) == 1
            finally:
                child.stop()
        print("PASS production custody setup failure refuses command; no writable-root fallback")


def shell_checks():
    if os.name == "nt":
        # Git Bash lacks setsid/Unix group semantics; run exact scripts in local WSL.
        repo = Path(__file__).resolve().parents[3]
        posix = "/mnt/" + repo.drive[0].lower() + repo.as_posix()[2:]
        subprocess.run(["wsl", "-d", "Ubuntu", "--cd", posix, "env",
                        "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "PYTHONDONTWRITEBYTECODE=1",
                        "python3", "-B", "-c",
                        "from experiments.cloud.lsr_jobs.s0_runtime import shell_checks; shell_checks()"], check=True)
        return
    bash = shutil.which("bash")
    if not bash:
        raise AssertionError("bash required for shell checks")
    cloud = Path(__file__).resolve().parent.parent
    for name in ("vm_startup.sh", "vm_shutdown.sh"):
        subprocess.run([bash, "-n", (cloud / name).as_posix()], check=True)
    print("PASS bash -n vm_startup.sh vm_shutdown.sh")
    shellcheck = shutil.which("shellcheck")
    if shellcheck:
        subprocess.run([shellcheck, str(cloud / "vm_startup.sh"), str(cloud / "vm_shutdown.sh")], check=True)
        print("PASS shellcheck")
    else:
        print("SKIP shellcheck unavailable")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Deliberately tiny PATH: all potential cloud/driver effects are stubs.
        def script(name, body):
            p = root / name
            p.write_text("#!/bin/bash\n" + body + "\n", encoding="utf-8")
            p.chmod(0o755)
        script("gcloud", "echo forbidden >&2; exit 99")
        script("gsutil", "echo forbidden >&2; exit 99")
        for name in ("bash", "timeout", "grep", "mkdir", "mv", "cat", "setsid", "sha256sum"):
            target = shutil.which(name)
            assert target, name
            script(name, f'exec "{Path(target).as_posix()}" "$@"')
        script("automon.sh", 'echo automon >> "$LOG"; echo "IDLE_POWEROFF ready=0 reason=never_ready"')
        env = {**os.environ, "PATH": root.as_posix(), "W": root.as_posix(),
               "OUT": root.as_posix(), "LOG": (root / "calls").as_posix()}
        startup = (cloud / "vm_startup.sh").as_posix()
        for metadata, driver, expected in (("", False, False), ("gpu", False, False),
                                            ("", True, False), ("false", True, False), ("true", True, True)):
            if driver:
                script("nvidia-smi", 'echo nvidia >> "$LOG"')
            log = root / "calls"
            log.unlink(missing_ok=True)
            subprocess.run([bash, "-c", 'source "$1" --branch-library; gpu_checks "$2"; wait',
                            "check", startup, metadata], env=env, check=True, timeout=10, capture_output=True)
            calls = log.read_text() if log.exists() else ""
            assert ("nvidia" in calls) == expected, calls
            assert ("automon" in calls) == expected, calls
            print(f"PASS shell branch metadata={bool(metadata)} driver={driver} gpu={expected}")
        for fault in ("driver", "automon"):
            script("nvidia-smi", 'exit 7' if fault == "driver" else 'echo nvidia >> "$LOG"')
            script("automon.sh", 'exit 8')
            result = subprocess.run([bash, "-c", 'source "$1" --branch-library; gpu_checks true',
                                     "check", startup], env=env, timeout=10, capture_output=True)
            assert result.returncode != 0, fault
            print(f"PASS shell GPU {fault} fault exit={result.returncode}")
        # Exercise the actual shutdown script, including its fresh-environment path.
        script("date", 'echo "$FAKE_NOW"')
        script("gcloud", 'echo upload >> "$LOG"; exit "${UPLOAD_RC:-0}"')
        for case in ("happy", "custody", "expired", "preemption", "rename", "write", "final_publish"):
            log = root / "calls"
            log.unlink(missing_ok=True)
            for name, value in (("runtime_created", "1000"), ("runtime_job", "gs://fake/shards/0"),
                                ("runtime_bundle_hash", "a" * 64)):
                (root / name).write_text(value)
            for name, value in (("run", "offline"), ("shard", "0"), ("attempt", "offline-0"), ("shard_hash", "b" * 64)):
                (root / ("runtime_" + name)).write_text(value)
            inner = {"status": "completed", "rc": 0, "reason": "completed", "finalized": False,
                     "identity": {"run": "offline", "shard": 0, "attempt": "offline-0"},
                     "hashes": {"bundle_sha256": "a" * 64, "shard_sha256": "b" * 64},
                     "clock": {"created_epoch": 1000}}
            (root / "runtime_receipt.json").write_text(json.dumps(inner))
            script("mv", 'exit 9' if case == "rename" else f'exec "{shutil.which("mv")}" "$@"')
            script("gcloud", 'echo upload >> "$LOG"; if [[ "$2" == cp && "${FINAL_FAIL:-0}" == 1 ]]; then exit 9; fi; exit "${UPLOAD_RC:-0}"')
            (root / "shutdown_receipt.tmp").unlink(missing_ok=True)
            if case == "write":
                (root / "shutdown_receipt.tmp").mkdir()
            test_env = {**env, "FAKE_NOW": "15380" if case == "expired" else "14500",
                        "LSR_LIFETIME_SECONDS":"14400", "LSR_WORK_STOP_SECONDS":"13500",
                        "LSR_SHUTDOWN_RESERVE_SECONDS":"900", "LSR_CREDIT_GUARD_EPOCH":"1792195200",
                        "UPLOAD_RC": "1" if case == "custody" else "0", "LSR_OUT": root.as_posix(),
                        "LSR_LIFETIME_SECONDS":"14400", "LSR_WORK_STOP_SECONDS":"13500", "LSR_SHUTDOWN_RESERVE_SECONDS":"900", "LSR_CREDIT_GUARD_EPOCH":"1792195200", "LSR_EXIT_CODE": "0", "LSR_END_REASON": "shutdown", "FINAL_FAIL": "1" if case == "final_publish" else "0"}
            for key in ("LSR_JOB", "LSR_CREATED_EPOCH", "LSR_BUNDLE_SHA256"):
                test_env.pop(key, None)
            if case == "preemption":
                test_env.pop("LSR_EXIT_CODE")
                test_env.pop("LSR_END_REASON")
            result = subprocess.run([bash, (cloud / "vm_shutdown.sh").as_posix()],
                                    env=test_env, timeout=10, capture_output=True)
            receipt = json.loads((root / "shutdown_receipt.json").read_text())
            assert (result.returncode == 0) == (case == "happy"), result.stderr
            assert receipt["status"] == ("completed" if case == "happy" else "failed")
            assert receipt["rc"] == result.returncode
            if case not in ("rename", "write"):
                assert receipt["hashes"]["bundle_sha256"] == "a" * 64
            assert log.exists() == (case not in ("expired", "rename", "write"))
            assert receipt["finalized"] == (case in ("happy", "preemption"))
            if case == "happy":
                from experiments.cloud.lsr_cloud import receipt_state
                raw = (root / "runtime_receipt.json").read_text()
                assert receipt["hashes"]["runtime_receipt_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
                expected = {**inner["identity"], **inner["hashes"], "created_epoch": 1000}
                assert receipt_state({"runtime_receipt.json": raw,
                    "shutdown_receipt.json": (root / "shutdown_receipt.json").read_text()}, expected=expected) == "done"
                print("PASS exact shutdown script -> receipt consumer done; exact runtime bytes bound")
            if case == "write":
                (root / "shutdown_receipt.tmp").rmdir()
            print(f"PASS shell shutdown {case} exit={result.returncode} terminal={receipt['status']}")

    full_bootstrap_checks(bash, cloud)
    custody_group_checks()

def full_bootstrap_checks(bash, cloud):
    """Exact scripts; every external command on PATH is a bounded offline stub."""
    source = (cloud / "vm_startup.sh").read_text()
    shutdown_source = (cloud / "vm_shutdown.sh").read_text()

    def probe(case, created="1000", startup_source=source, shutdown_text=shutdown_source, lifetime_sec=14400):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            startup = root / "startup.sh"
            startup.write_text(startup_source)
            (root / "shutdown.source").write_text(shutdown_text)

            def stub(name, body):
                path = bin_dir / name
                path.write_text("#!/bin/bash\nset -eu\n" + body + "\n")
                path.chmod(0o755)

            for name in ("bash", "mkdir", "mv", "cat", "tee", "grep", "touch", "setsid"):
                real = shutil.which(name)
                assert real, name
                stub(name, f'exec "{real}" "$@"')
            stub("date", r'''if [[ -f "$LSR_ROOT/expired" || "$CASE" == expired_launch ]]; then echo "$STOP_AT";
                elif [[ "$CASE" == late_metadata ]]; then echo "$((STOP_AT-2))";
                elif [[ "$CASE" == watchdog_late && "$PPID" != "$(/bin/cat "$LSR_ROOT/parent")" ]]; then
                    : > "$LSR_ROOT/expired"; echo "$STOP_AT";
                else echo "$((STOP_AT-1))"; fi''')
            stub("curl", r'''case "${!#}" in
                */lsr-created-at) echo "$PPID" > "$LSR_ROOT/parent"; printf '%s\n' "$CREATED_TEXT";;
                */lsr-job) echo gs://fake/runs/offline/shards/0;;
                */lsr-bundle-sha256) printf '%064d\n' 0;;
                */shutdown-script) [[ "$CASE" != late_metadata ]] || : > "$LSR_ROOT/expired"; /bin/cat "$LSR_ROOT/shutdown.source";;
                */lsr-lifetime-seconds) echo "$FAKE_LIFETIME";;
                */lsr-work-stop-seconds) echo "$((FAKE_LIFETIME-900))";;
                */lsr-shutdown-reserve-seconds) echo 900;;
                */lsr-credit-guard-epoch) echo 1792195200;;
                */lsr-run) echo offline;;
                */lsr-shard) echo 0;;
                */lsr-attempt) echo offline-0;;
                */lsr-shard-sha256) printf '%064d\n' 0;;
                */lsr-has-gpu) echo false;;
                */lsr-weights-mirror-uri) echo '';;
                *) exit 98;; esac''')
            stub("timeout", f'exec "{shutil.which("timeout")}" "$@"')
            stub("sha256sum", 'if [[ "${1:-}" != -c ]]; then exec /usr/bin/sha256sum "$@"; fi; while read -r line; do :; done; [[ "$CASE" != hash_fault ]] || exit 9; echo offline-hash-ok')
            stub("tar", '/bin/mkdir -p "$LSR_ROOT/experiments/memory" "$LSR_ROOT/lsr_jobs"; : > "$LSR_ROOT/lsr_jobs/s0_runtime.py"')
            stub("sleep", r'''echo "watchdog_sleep=$1 current=$(date +%s)" >> "$LSR_ROOT/calls"
            for ((i=0;i<200;i++)); do
                [[ ! -f "$LSR_ROOT/descendant" ]] || break
                /bin/sleep 0.01
            done
            /bin/sleep 0.1
            : > "$LSR_ROOT/expired"''')
            blocked = r'''/bin/sleep 60 &
                echo $! > "$LSR_ROOT/descendant"
                echo blocked >> "$LSR_ROOT/calls"
                wait'''
            stub("python3", blocked if case == "supervisor" else
                 'printf \'{"status":"completed","reason":"completed","rc":0,"finalized":false,"hashes":{},"clock":{}}\\n\' > "$OUT/runtime_receipt.json"; exit 0')
            stub("gcloud", r'''echo "gcloud_$2 current=$(date +%s)" >> "$LSR_ROOT/calls"
            if [[ "$2" == rsync || "${!#}" == */shutdown_receipt.json ]]; then
                if [[ -f "$LSR_ROOT/descendant" ]]; then
                    pid=$(/bin/cat "$LSR_ROOT/descendant")
                    state=$(/bin/ps -o stat= -p "$pid" || true)
                    if [[ -n "$state" && "$state" != Z* ]]; then
                        echo CUSTODY_BEFORE_STOP >> "$LSR_ROOT/calls"; exit 77
                    fi
                fi
                echo stopped-before-custody >> "$LSR_ROOT/calls"
                echo custody >> "$LSR_ROOT/calls"
            elif [[ "$2" == cp && ( "$CASE" == bootstrap || "$CASE" == late_metadata ) ]]; then
            ''' + blocked + r'''
            else
                : > "${!#}"
            fi''')
            stub("shutdown", 'echo poweroff >> "$LSR_ROOT/calls"')
            env = {**os.environ, "PATH": str(bin_dir), "LSR_ROOT": str(root),
                   "CASE": case, "CREATED_TEXT": created, "FAKE_LIFETIME":str(lifetime_sec), "STOP_AT":str(1000+lifetime_sec-900)}
            for key in ("LSR_CREATED_EPOCH", "LSR_JOB", "LSR_BUNDLE_SHA256", "LSR_END_REASON",
                        "LSR_EXIT_CODE", "LSR_CUSTODY_DEFERRED", "LSR_WORK_GROUP"):
                env.pop(key, None)
            try:
                result = subprocess.run([bash, str(startup)], env=env, capture_output=True,
                                        text=True, timeout=12)
                calls = (root / "calls").read_text() if (root / "calls").exists() else ""
                receipt = json.loads((root / "out/shutdown_receipt.json").read_text())
                return result, calls, receipt
            finally:
                path = root / "out/runtime_work_pgid"
                if path.exists():
                    try:
                        os.killpg(int(path.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    for total in (3600,10800,14400):
        result, calls, receipt = probe("happy", lifetime_sec=total)
        assert result.returncode == 0, (total,result,calls)
        assert receipt["clock"]["work_stop_epoch"] == 1000+total-900
        assert receipt["clock"]["lifetime_end_epoch"] == 1000+total
        print(f"PASS actual startup/shutdown lifetime={total}")

    marker = 'BASH_VERSINFO[$(echo METADATA_ARITHMETIC_EXECUTED >&2)0]'
    def assert_no_late_work(result, calls, receipt):
        assert result.returncode != 0 and receipt["reason"] == "deadline", (result, calls, receipt)
        assert "blocked" not in calls and "watchdog_sleep=" not in calls, calls
        assert "gcloud_cp current=14500" in calls, calls  # Final receipt custody is allowed.
        assert "CUSTODY_BEFORE_STOP" not in calls

    for case in ("late_metadata", "expired_launch"):
        assert_no_late_work(*probe(case))
        print(f"PASS full bootstrap {case}: no work or positive watchdog delay at cutoff")
    result, calls, receipt = probe("watchdog_late")
    assert result.returncode != 0 and receipt["reason"] == "deadline", (result, calls, receipt)
    assert "watchdog_sleep=" not in calls and "blocked" not in calls, calls
    print("PASS watchdog fresh clock: expired child/watchdog never sleeps or bootstraps")
    # Permanent version of review A's 14498 -> 14500 metadata reproduction.
    old_clock = source.replace(
        'if ! work_budget; then export LSR_END_REASON=deadline; exit 1; fi',
        'LEFT=$((LSR_STOP_EPOCH - NOW)); if (( LEFT <= 0 )); then export LSR_END_REASON=deadline; exit 1; fi'
    ).replace('    work_budget || exit 1', '    LEFT=2').replace(
        "    now=$(date +%s)\n    if [[", "    now=$NOW\n    if [[")
    # NOW must be inherited by the old relative-countdown mutant.
    old_clock = old_clock.replace('export LSR_STOP_EPOCH=', 'export NOW\nexport LSR_STOP_EPOCH=')
    reproduction = probe("late_metadata", startup_source=old_clock)
    assert "watchdog_sleep=2 current=14500" in reproduction[1], reproduction
    assert "blocked" in reproduction[1], reproduction
    try:
        assert_no_late_work(*reproduction)
    except AssertionError:
        print("PASS negative self-test: review stale-clock reproduction rejected")
    else:
        raise AssertionError("old stale-clock behavior escaped regression check")
    for value in ("0", "01000", "999999999999999999999999", "14501", "nope", marker):
        result, calls, receipt = probe("metadata", value)
        assert result.returncode != 0 and receipt["status"] == "failed", (value, result, calls)
        assert "METADATA_ARITHMETIC_EXECUTED" not in result.stderr + result.stdout
        assert "custody" not in calls
        print(f"PASS full bootstrap invalid epoch={value!r} exit={result.returncode} no-evaluation")
    for case in ("bootstrap", "supervisor"):
        result, calls, receipt = probe(case)
        assert result.returncode != 0 and receipt["reason"] == "deadline", (result, calls, receipt)
        assert "CUSTODY_BEFORE_STOP" not in calls and "blocked" in calls, calls
        assert calls.index("stopped-before-custody") < calls.index("custody\n"), calls
        print(f"PASS full blocked {case}: descendants stopped before first custody")
    for case in ("happy", "hash_fault"):
        result, calls, receipt = probe(case)
        assert (result.returncode == 0) == (case == "happy"), (result, calls, receipt)
        assert receipt["status"] == ("completed" if case == "happy" else "failed")
        assert receipt["finalized"] and "custody" in calls
        if case == "hash_fault":
            assert receipt["reason"] == "bootstrap"
        print(f"PASS full bootstrap {case} exit={result.returncode} finalized={receipt['finalized']}")
    old_validation = source.replace(
        'if ! [[ "$LSR_CREATED_EPOCH" =~ ^[1-9][0-9]{0,9}$ ]]; then exit 1; fi',
        '[[ "$LSR_CREATED_EPOCH" =~ ^[0-9]+$ ]] && (( LSR_CREATED_EPOCH <= $(date +%s) ))')
    result, _, _ = probe("metadata", marker, old_validation)
    assert "METADATA_ARITHMETIC_EXECUTED" in result.stdout + result.stderr, result
    print("PASS negative self-test: old metadata arithmetic evaluates marker and is rejected")
    old_stop = source.replace('    kill -KILL -- "-$2"',
                              '    bash "$W/shutdown.sh"; kill -KILL -- "-$2"')
    old_shutdown = shutdown_source.replace('kill -KILL -- "-$PGID" 2>/dev/null || true', ':')
    result, calls, _ = probe("bootstrap", startup_source=old_stop, shutdown_text=old_shutdown)
    assert "CUSTODY_BEFORE_STOP" in calls, (result, calls)
    print("PASS negative self-test: custody-before-stop rejected")
    old_receipt = shutdown_source.replace('write_receipt "$STATUS" false || { fallback; exit 1; }',
                                          'write_receipt "$STATUS" false').replace(
        'write_receipt "$STATUS" true || { fallback; exit 1; }', 'write_receipt "$STATUS" true')
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        script = root / "shutdown.sh"
        script.write_text(old_receipt)
        for name, body in (("date", "echo 14500"), ("mv", "exit 9"), ("timeout", "exit 0"),
                           ("mkdir", f'exec "{shutil.which("mkdir")}" "$@"'),
                           ("setsid", f'exec "{shutil.which("setsid")}" "$@"'),
                           ("sha256sum", f'exec "{shutil.which("sha256sum")}" "$@"')):
            p = root / name
            p.write_text("#!/bin/bash\n" + body + "\n")
            p.chmod(0o755)
        (root / "runtime_receipt.json").write_text('{}')
        result = subprocess.run([bash, str(script)], env={**os.environ, "PATH": str(root),
            "LSR_RUN": "offline", "LSR_SHARD": "0", "LSR_ATTEMPT": "offline-0", "LSR_SHARD_SHA256": "b" * 64,
            "LSR_OUT": str(root), "LSR_CREATED_EPOCH": "1000", "LSR_LIFETIME_SECONDS":"14400", "LSR_WORK_STOP_SECONDS":"13500", "LSR_SHUTDOWN_RESERVE_SECONDS":"900", "LSR_CREDIT_GUARD_EPOCH":"1792195200", "LSR_EXIT_CODE": "0",
            "LSR_BUNDLE_SHA256": "a" * 64,
            "LSR_JOB": "gs://fake/shards/0"}, capture_output=True, timeout=10)
        assert result.returncode == 0 and not (root / "shutdown_receipt.json").exists()
    print("PASS negative self-test: unchecked rename exits zero and is rejected")


def custody_group_checks():
    for blocked in (False, True):
        with tempfile.TemporaryDirectory() as td:
            marker = Path(td) / "pid"
            code = ("import subprocess,time; from pathlib import Path; "
                    f"p=subprocess.Popen([{sys.executable!r},'-B','-c','import time; time.sleep(60)']); "
                    f"Path({str(marker)!r}).write_text(str(p.pid)); " +
                    ("time.sleep(60)" if blocked else "time.sleep(.05)"))
            try:
                bounded_command([sys.executable, "-B", "-c", code], 0.5)
                assert not blocked
            except subprocess.TimeoutExpired:
                assert blocked
            pid = int(marker.read_text())
            for _ in range(100):
                stat = Path(f"/proc/{pid}/stat")
                if not stat.exists() or stat.read_text().split()[2] == "Z":
                    break
                time.sleep(.01)
            else:
                os.kill(pid, signal.SIGKILL)
                raise AssertionError("custody descendant survived")
            print(f"PASS custody descendant cleanup blocked={blocked}")


def receipt_and_progress_checks():
    from unittest.mock import patch
    for fault in ("rename", "write", "final_publish"):
        with tempfile.TemporaryDirectory() as td:
            uploads = []
            runtime = Runtime(td, 1000, {}, clock=lambda: 1001,
                              heartbeat=lambda: 1001, spawn=lambda _: type("P", (), {
                                  "poll": lambda self: 0, "stop": lambda self: None})(),
                              upload=lambda path, budget: uploads.append(True),
                              publish=lambda *_: (_ for _ in ()).throw(OSError("publish")))
            if fault in ("rename", "write"):
                original = write_json
                def fail(path, value):
                    if Path(path).name == "runtime_receipt.json":
                        raise OSError(fault)
                    return original(path, value)
                with patch(__name__ + ".write_json", fail):
                    rc = runtime.execute(lambda: None, ["control"], ["work"])
            else:
                rc = runtime.execute(lambda: None, ["control"], ["work"])
            receipt = json.loads((Path(td) / "runtime_receipt.json").read_text())
            assert rc != 0 and receipt["status"] == "failed" and not receipt["finalized"]
            assert bool(uploads) == (fault == "final_publish")
            print(f"PASS runtime receipt fault={fault} exit={rc}")
    for advancing in (True, False):
        with tempfile.TemporaryDirectory() as td:
            now = [1000]
            beat = [1000]
            class Phase:
                def poll(self):
                    return 0 if now[0] >= 1600 else None
                def stop(self):
                    pass
            def tick(_):
                now[0] += 100
                if advancing:
                    beat[0] = now[0]
            runtime = Runtime(td, 1000, {}, clock=lambda: now[0], sleep=tick,
                              heartbeat=lambda: beat[0], spawn=lambda _: Phase(), upload=lambda *_: None)
            rc = runtime.execute(lambda: None, ["control"], ["fit"])
            assert (rc == 0) == advancing
            print(f"PASS long phase advancing={advancing} exit={rc} (no blind heartbeat timer)")


def save_fixture(cfg, args):
    raw = (json.dumps(cfg, sort_keys=True) + "\n").encode()
    Path(args.shard).write_bytes(raw)
    args.expected_shard_sha256 = hashlib.sha256(raw).hexdigest()


def deployment_fixture(root):
    from types import SimpleNamespace
    control = ["fake-control"]
    lock = b"Image_OS_Package==1.0\n"
    cfg = {"run": "offline", "shard": 0, "attempt": "offline-0", "kind": "associative_continuation_cpu", "model": None,
           "runtime_control_command": control, "requirements": [lock.decode().strip()],
           "runtime_control_sha256": hashlib.sha256(json.dumps(control, separators=(",", ":")).encode()).hexdigest(),
           "requirements_lock_sha256": hashlib.sha256(lock).hexdigest(), "weights_manifest_sha256": None}
    cfg["dispatch"] = {**{k: cfg[k] for k in ("run", "shard", "attempt", "runtime_control_sha256", "requirements_lock_sha256", "weights_manifest_sha256")},
                       "bundle_sha256": "a" * 64, "created_at": 1000, "work_stop_at": 14500,
                       "shutdown_at": 15400, "work_stop_seconds": WORK_STOP, "lifetime_seconds": LIFETIME,
                       "shutdown_reserve_seconds": RESERVE, "allow_hf_download": False,
                       "weights_local_path": None, "weights_manifest_uri": None, "weights_mirror_uri": None,
                       "model_revision": None, "tokenizer_revision": None}
    root.mkdir(exist_ok=True)
    (root / "requirements.txt").write_bytes(lock)
    args = SimpleNamespace(out=str(root), shard=str(root / "shard.json"), created_epoch=1000,
                           bundle_sha256="a" * 64, job="gs://offline/shards/000",
                           expected_run="offline", expected_shard=0, expected_attempt="offline-0",
                           requirements_path=str(root / "requirements.txt"), weights_path=str(root / "weights"))
    installed = [SimpleNamespace(metadata={"Name": "image-os-package"}, version="1.0")]
    save_fixture(cfg, args)
    return cfg, args, installed


def deployment_checks():
    """Production verifier/run with raising effects FIRST; no cloud/network/model."""
    from unittest.mock import patch
    import socket
    from experiments.cloud.lsr_cloud import receipt_state
    def forbidden(*_a, **_kw):
        raise AssertionError("external effect forbidden")
    cases = ("cpu", "model", "model_one", "model_three", "model_single", "mutate_shard", "index_missing", "argv", "argv_copy", "lock_bytes", "lock_list", "version", "extra_package",
             "missing_package", "manifest_pin", "revision", "asset_content", "missing_asset", "traversal",
             "extra_asset", "symlink", "no_complete", "null_model", "swapped", "bool_shard", "shard_bytes",
             "missing_authority", "bundle", "clock", "download", "hash_deadline")
    reasons = {"index_missing":"asset_inventory", "argv":"runtime_control_sha256", "argv_copy":"runtime_control_sha256",
        "lock_bytes":"requirements_lock_sha256", "lock_list":"requirements_bytes", "version":"installed_dependencies",
        "extra_package":"installed_dependencies", "missing_package":"installed_dependencies", "manifest_pin":"weights_manifest_sha256",
        "revision":"asset_revision", "asset_content":"asset_content", "missing_asset":"asset_missing", "traversal":"asset_path",
        "extra_asset":"asset_unlisted", "symlink":"asset_symlink", "no_complete":"asset_incomplete",
        "null_model":"weights_manifest_sha256", "swapped":"shard_identity", "bool_shard":"shard_identity",
        "shard_bytes":"shard_hash", "missing_authority":"shard_hash", "bundle":"launch_binding", "clock":"launch_binding",
        "download":"download_fallback", "hash_deadline":"deadline"}
    with patch.object(subprocess, "Popen", forbidden), patch.object(subprocess, "run", forbidden), \
         patch.object(socket, "socket", forbidden), patch.object(socket, "create_connection", forbidden):
        for case in cases:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                cfg, args, installed = deployment_fixture(root)
                if case not in ("cpu", "mutate_shard", "argv", "argv_copy", "lock_bytes", "lock_list", "version", "extra_package", "missing_package",
                                "swapped", "bool_shard", "shard_bytes", "missing_authority", "bundle", "clock", "download"):
                    weights = root / "weights"
                    weights.mkdir()
                    names = ["config.json", "tokenizer.json", "model.safetensors.index.json"] + [f"weight-{i}.safetensors" for i in range(1 if case in ("model_one", "model_single") else 3 if case == "model_three" else 8)]
                    for name in names:
                        (weights / name).write_bytes(b"reviewed-local-fixture")
                    (weights / "config.json").write_text('{}')
                    (weights / "tokenizer.json").write_text('{}')
                    (weights / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {str(i):n for i,n in enumerate(names[3:])}}))
                    if case == "model_single":
                        (weights / names[-1]).rename(weights / "model.safetensors")
                        (weights / "model.safetensors.index.json").unlink()
                        names = ["config.json", "tokenizer.json", "model.safetensors"]
                    if case == "index_missing":
                        (weights / "model.safetensors.index.json").write_text(json.dumps({"weight_map":{"x":"absent.safetensors"}}))
                    manifest = "".join(digest(weights / n) + "  " + n + "\n" for n in names)
                    (weights / "_manifest_sha256.txt").write_text(manifest)
                    (weights / "_COMPLETE").touch()
                    cfg.update(model="reviewed/model", revision="c" * 40, tokenizer_revision="d" * 40,
                               weights_manifest_sha256=digest(weights / "_manifest_sha256.txt"))
                    cfg["dispatch"].update(weights_manifest_sha256=cfg["weights_manifest_sha256"],
                        model_revision=cfg["revision"], tokenizer_revision=cfg["tokenizer_revision"],
                        weights_local_path="/opt/lsr/weights", weights_mirror_uri="gs://offline/model",
                        weights_manifest_uri="gs://offline/model/_manifest_sha256.txt")
                    if case == "manifest_pin": (weights / "_manifest_sha256.txt").write_text(manifest + "\n")
                    if case == "revision": cfg["revision"] = "e" * 40
                    if case == "asset_content": (weights / names[2]).write_bytes(b"altered")
                    if case == "missing_asset": (weights / names[2]).unlink()
                    if case == "extra_asset": (weights / "extra.bin").touch()
                    if case == "no_complete": (weights / "_COMPLETE").unlink()
                    if case == "null_model":
                        cfg["weights_manifest_sha256"] = cfg["dispatch"]["weights_manifest_sha256"] = None
                    if case == "traversal":
                        (weights / "_manifest_sha256.txt").write_text(manifest.replace(names[2], "../escape"))
                        cfg["weights_manifest_sha256"] = cfg["dispatch"]["weights_manifest_sha256"] = digest(weights / "_manifest_sha256.txt")
                    if case == "symlink":
                        (weights / names[2]).unlink()
                        try: (weights / names[2]).symlink_to(weights / names[3])
                        except OSError:
                            print("SKIP symlink fixture: OS privilege unavailable")
                            continue
                if case == "argv": cfg["runtime_control_command"] = ["altered-control"]
                if case == "argv_copy": cfg["dispatch"]["runtime_control_sha256"] = "f" * 64
                if case == "lock_bytes": (root / "requirements.txt").write_bytes(b"changed==2\n")
                if case == "lock_list": cfg["requirements"] = ["changed==2"]
                if case == "version": installed[0].version = "2.0"
                if case == "extra_package": installed.append(type(installed[0])(metadata={"Name": "extra"}, version="1"))
                if case == "missing_package": installed.clear()
                if case == "swapped": cfg["attempt"] = "another-attempt"
                if case == "bool_shard": cfg["shard"] = False
                if case == "bundle": cfg["dispatch"]["bundle_sha256"] = "f" * 64
                if case == "clock": cfg["dispatch"]["created_at"] = 999
                if case == "download": cfg["dispatch"]["allow_hf_download"] = True
                save_fixture(cfg, args)
                if case == "shard_bytes": Path(args.shard).write_bytes(Path(args.shard).read_bytes() + b" ")
                if case == "missing_authority": args.expected_shard_sha256 = None
                (root / "heartbeat").touch()
                os.utime(root / "heartbeat", (1001, 1001))
                spawned, now = [], [1001]
                class FakeChild:
                    def __init__(self, argv):
                        spawned.append(list(argv))
                        if case == "mutate_shard":
                            if len(spawned) == 1:
                                Path(args.shard).write_text(json.dumps(dict(cfg,kind="unreviewed-kind")))
                            else:
                                assert argv[4] == "--verified-json"
                                assert hashlib.sha256(argv[6].encode()).hexdigest() == args.expected_shard_sha256
                                assert json.loads(argv[6])["kind"] == cfg["kind"]
                    def poll(self): return 0
                    def stop(self): pass
                base = Runtime
                def factory(*a, **kw):
                    obj = base(*a, clock=lambda: now[0], sleep=forbidden, **kw)
                    if case == "hash_deadline":
                        progress = obj.progress
                        def expire():
                            now[0] = 14500
                            progress()
                        obj.progress = expire
                    return obj
                with patch.object(metadata, "distributions", return_value=installed), \
                     patch(__name__ + ".Runtime", factory), patch(__name__ + ".Child", FakeChild), \
                     patch.dict(os.environ, {"LSR_CUSTODY_DEFERRED": "1"}):
                    rc = run(args)
                raw = (root / "runtime_receipt.json").read_text()
                receipt = json.loads(raw)
                if case in ("cpu", "model", "model_one", "model_three", "model_single", "mutate_shard"):
                    assert rc == 0 and len(spawned) == 2, receipt
                    assert receipt["verification"]["installed_dependencies"]["actual"] == {"image-os-package": "1.0"}
                    expected = {**receipt["identity"], "shard_sha256": args.expected_shard_sha256,
                                "bundle_sha256": args.bundle_sha256, "created_epoch": args.created_epoch}
                    outer = {**receipt, "finalized": True, "hashes": {**receipt["hashes"],
                        "runtime_receipt_sha256": hashlib.sha256(raw.encode()).hexdigest()}}
                    pair = {"runtime_receipt.json": raw, "shutdown_receipt.json": json.dumps(outer)}
                    assert receipt_state(pair, expected=expected) == "done"
                    outer["hashes"]["runtime_receipt_sha256"] = "f" * 64
                    pair["shutdown_receipt.json"] = json.dumps(outer)
                    assert receipt_state(pair, expected=expected) == "failed(receipt-custody-binding)"
                    print(f"PASS verified {case}: production runtime -> consumer done; unbound receipt rejected")
                else:
                    assert rc == 1 and receipt["status"] == "failed" and not spawned, (case, receipt, spawned)
                    assert receipt["reason"] == reasons[case], ("wrong deployment refusal reason", case, receipt)
                    if case in ("version", "extra_package", "missing_package"):
                        field = {"version":"mismatched", "extra_package":"unexpected", "missing_package":"missing"}[case]
                        assert receipt["verification"]["installed_dependencies"][field], (case, receipt)
                    print(f"PASS deployment {case}: nonzero terminal {receipt['reason']}; zero children")


def deployment_reason_mutant():
    """An unrelated refusal must not satisfy the argv integrity negative."""
    from unittest.mock import patch
    from contextlib import redirect_stdout
    import io
    original = verify_deployment
    def mutant(cfg, *args):
        if cfg.get("runtime_control_command") == ["altered-control"]:
            raise Stop("dispatch_contract")
        return original(cfg, *args)
    with patch(__name__ + ".verify_deployment", mutant), redirect_stdout(io.StringIO()):
        try:
            deployment_checks()
        except AssertionError as exc:
            assert exc.args[0][0:2] == ("wrong deployment refusal reason", "argv"), exc
        else:
            raise AssertionError("wrong-reason mutant escaped deployment suite")
    print("PASS negative self-test: unrelated argv refusal rejected by intended-reason assertion")


def cli_checks():
    result = subprocess.run([sys.executable, "-B", "-m", __package__ + ".s0_runtime", "--check", "unknown"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2, result
    print("PASS unknown check exit=2")
    # Missing operation control must stop before spawning any actual dispatcher.
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "shard.json").write_text('{}')
        (root / "heartbeat").touch()
        def forbidden(*_args, **_kwargs):
            raise AssertionError("real external effect forbidden")
        args = argparse.Namespace(out=td, created_epoch=int(time.time()),
                                  bundle_sha256="a" * 64, shard=str(root / "shard.json"), job="gs://fake")
        with patch.object(subprocess, "Popen", forbidden), patch.object(subprocess, "run", forbidden):
            rc = run(args)
        receipt = json.loads((root / "runtime_receipt.json").read_text())
        assert rc == 1 and receipt["prior_reason"] == "control_missing", receipt
        assert receipt["hashes"]["shard_sha256"] == digest(root / "shard.json")
        print("PASS production run missing control rejected before dispatch; upload blocked by raising mock")
    # Execute the production JSON reader and guard together; a frozen control
    # cannot be admitted by NaN, infinities, strings, null, bool or malformed JSON.
    for epoch in (float("nan"), float("inf"), -float("inf"), "NaN", "1000", "bad", None, True, 10**1000):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg, args, installed = deployment_fixture(root)
            (root / "heartbeat").touch()
            (root / "heartbeat_control").write_text(json.dumps({"stage": "control", "epoch": epoch}))
            args.created_epoch = int(time.time())
            cfg["dispatch"].update(created_at=args.created_epoch, work_stop_at=args.created_epoch + WORK_STOP, shutdown_at=args.created_epoch + LIFETIME)
            save_fixture(cfg, args)
            original = Runtime
            spawned = []
            def construct(*pos, **kw):
                kw.update(spawn=lambda argv: spawned.append(argv) or forbidden(),
                          upload=lambda *_: None, publish=lambda *_: None, deferred=False)
                return original(*pos, **kw)
            with patch.object(metadata, "distributions", return_value=installed), patch(__name__ + ".Runtime", construct), patch.object(subprocess, "Popen", forbidden), patch.object(subprocess, "run", forbidden):
                rc = run(args)
            receipt = json.loads((root / "runtime_receipt.json").read_text())
            assert rc == 1 and receipt["reason"] == "stall" and not spawned, receipt
            print(f"PASS production malformed heartbeat type={type(epoch).__name__} rejected before frozen control")
    for beat in (float("nan"), float("inf"), -float("inf"), "bad", None, True, 10**1000):
        with tempfile.TemporaryDirectory() as td:
            runtime = Runtime(td, 1000, {}, clock=lambda: 1600, heartbeat=lambda: beat)
            try:
                runtime.guard("control")
            except Stop as exc:
                assert str(exc) == "stall"
            else:
                raise AssertionError("malformed heartbeat escaped guard")
    print("PASS direct guard malformed heartbeat negative controls")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--custody-exec"]:
        if len(argv) < 4 or argv[2] != "--":
            return 2
        try:
            restrict_worker_writes(argv[1])
            root = Path(argv[1])
            for name in ("tmp", "offline-hf"):
                (root / name).mkdir(exist_ok=True)
            os.environ.update(TMPDIR=str(root / "tmp"), HF_HOME=str(root / "offline-hf"))
            os.execvpe(argv[3], argv[3:], os.environ)
        except (Stop, OSError) as exc:
            print("FAIL use-time custody: " + str(exc), file=sys.stderr)
            return 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", choices=["fault-injection"])
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--shard")
    parser.add_argument("--out")
    parser.add_argument("--created-epoch", type=int)
    parser.add_argument("--bundle-sha256")
    parser.add_argument("--job")
    parser.add_argument("--expected-run")
    parser.add_argument("--expected-shard", type=int)
    parser.add_argument("--expected-attempt")
    parser.add_argument("--expected-shard-sha256")
    args = parser.parse_args(argv)
    if args.check:
        fault_checks()
        class BrokenGuard(Runtime):
            def guard(self, stage):
                pass
        try:
            fault_checks(BrokenGuard, verbose=False)
        except AssertionError:
            print("PASS negative self-test: broken guard rejected")
        else:
            raise AssertionError("broken guard escaped fault suite")
        receipt_and_progress_checks()
        deployment_checks()
        deployment_reason_mutant()
        custody_checks()
        shell_checks()
        cli_checks()
        return 0
    if not args.run or not all((args.shard, args.out, args.created_epoch, args.bundle_sha256, args.job)):
        parser.error("--run requires --shard --out --created-epoch --bundle-sha256 --job")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
