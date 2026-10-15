"""VM entry point: python -m lsr_jobs.run --verified-json SHA256 JSON OUT. Dispatches on shard["kind"]."""
import importlib
import json
import sys
import os
import hashlib
import re
import time
from contextlib import contextmanager
from pathlib import Path

KINDS = {"smoke": "smoke", "recur": "recur", "search": "search", "branch": "branch", "p2": "p2", "fd": "fd", "native": "native", "q1pp": "q1pp", "readout": "readout", "gemma_controls": "gemma_controls", "bindq": "bindq", "formation": "formation", "e1_ledger": "e1_ledger", "s0_memory_port": "s0_memory_port"}

KINDS["associative_continuation_cpu"] = "experiments.memory.associative_continuation_cpu"

KINDS["latent_memory_prototype"] = "experiments.memory.prototype.runner"
PROTOTYPE_PHASES = {"prepare", "smoke", "overfit", "train", "evaluate", "demo"}

ENTRIES = {kind: {"main"} for kind in KINDS}
ENTRIES["e1_ledger"].add("alignment_hook_smoke")


def assemble_shard(cfg, i):
    """Single production assembly boundary; never silently discard an entry."""
    kind = cfg["kind"]
    if kind not in KINDS:
        raise ValueError(f"unknown kind: {kind}")
    common, shard = cfg.get("common", {}), cfg["shards"][i]
    entries = [part["entry"] for part in (cfg, common, shard) if "entry" in part]
    if entries and any(e != entries[0] for e in entries):
        raise ValueError("conflicting entry declarations")
    entry = entries[0] if entries else "main"
    if entry not in ENTRIES[kind]:
        raise ValueError(f"unknown entry for {kind}: {entry}")
    for part in (common, shard):
        if any(k in part for k in ("kind", "run", "shard", "attempt", "dispatch", "runtime_control_sha256",
                                   "requirements_lock_sha256", "weights_manifest_sha256", "requirements")):
            raise ValueError("reserved shard field")
    merged = {**common, **shard}
    if kind == "latent_memory_prototype":
        if merged.get("phase") not in PROTOTYPE_PHASES:
            raise ValueError("unknown prototype phase")
        if not isinstance(merged.get("config"), str) or not merged["config"]:
            raise ValueError("prototype config required")
        if not isinstance(merged.get("config_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", merged["config_sha256"]):
            raise ValueError("prototype config digest required")
        if not merged.get("model") and merged["phase"] != "prepare":
            raise ValueError("model-free prototype only supports prepare")
    control = {**common, **shard}.get("runtime_control_command")
    if not isinstance(control, list) or not control or any(not isinstance(x, str) or not x.strip() or "\x00" in x for x in control):
        raise ValueError("nonempty runtime_control_command argv required")
    return {**common, **shard, "kind": kind, "run": cfg["run"], "shard": i,
            "attempt": f"sc-bench-lsr-{cfg['run']}-{i:03d}",
            "runtime_control_sha256": hashlib.sha256(json.dumps(control, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "requirements_lock_sha256": cfg.get("requirements_lock_sha256"),
            "weights_manifest_sha256": cfg.get("weights_manifest_sha256") if {**common, **shard}.get("model") else None,
            "entry": entry, "entry_defaulted": not bool(entries)}


def prepare_runtime(cfg):
    """Mirror staging is owned by bootstrap; forbid a hub fallback here."""
    dispatch = cfg["dispatch"]
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    if cfg.get("model_local") is not None and cfg.get("model_local") != dispatch.get("weights_local_path"):
        raise ValueError("model_local conflicts with trusted root")
    if cfg.get("model"):
        if cfg["kind"] == "s0_memory_port":
            if cfg["model"] not in {"google/gemma-4-31B", "google/gemma-4-E2B", "google/gemma-4-E4B"}:
                raise ValueError("S0 requires admitted model identity; path rewrite forbidden")
            if cfg.get("model_source", cfg["model"]) != cfg["model"]:
                raise ValueError("S0 conflicting model identity aliases")
            for field, bound in (("revision", "model_revision"), ("tokenizer_revision", "tokenizer_revision")):
                if cfg.get(field) != dispatch.get(bound) or not re.fullmatch(r"[0-9a-f]{40}", cfg.get(field) or ""):
                    raise ValueError("S0 revision conflicts with trusted dispatch")
        root = Path(dispatch["weights_local_path"])
        if not root.is_dir() or not (root / "config.json").is_file():
            raise ValueError("staged mirror missing; refusing hub download")
        if dispatch["weights_local_path"] != "/opt/lsr/weights" or cfg.get("model_local", dispatch["weights_local_path"]) != dispatch["weights_local_path"]:
            raise ValueError("untrusted model_local")
        cfg["model_source"] = cfg.get("model_source", cfg["model"])
        cfg["model_local"] = dispatch["weights_local_path"]
        if cfg["kind"] not in {"latent_memory_prototype", "s0_memory_port"}:
            cfg["model"] = str(root)  # legacy loader API; model_source retains identity


@contextmanager
def cpu_work_progress(out_dir):
    """Emit progress only after a real optimizer step; remove hook on every exit."""
    from torch.optim.optimizer import register_optimizer_step_post_hook
    root=Path(out_dir)
    steps=0
    def completed_step(optimizer, args, kwargs):
        nonlocal steps
        steps += 1
        raw=json.dumps({"stage":"work", "epoch":time.time(), "optimizer_steps":steps,
                        "source":"optimizer-step-completed"})
        temporary=root/"heartbeat_work.pending"
        temporary.write_text(raw,encoding="utf-8")
        os.replace(temporary,root/"heartbeat_work")
    handle=register_optimizer_step_post_hook(completed_step)
    try:
        yield
    finally:
        handle.remove()


def execute(cfg, out_dir):
    prepare_runtime(cfg)
    if cfg["kind"] == "latent_memory_prototype":
        if cfg.get("phase") not in PROTOTYPE_PHASES:
            raise ValueError("unknown prototype phase")
        if not cfg.get("model") and (cfg["phase"] != "prepare" or cfg["dispatch"].get("has_gpu") is not False):
            raise ValueError("invalid model-free prototype")
        path = Path(cfg["config"])
        if path.is_symlink():
            raise ValueError("prototype config symlink forbidden")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != cfg.get("config_sha256"):
            raise ValueError("prototype config hash mismatch")
        prototype_config = json.loads(raw)
        if not isinstance(prototype_config, dict):
            raise ValueError("prototype config object required")
        # Only one effective assembly reaches the runner; the verified dispatch
        # wins over file aliases. No path reread or legacy CLI fallback is allowed.
        effective = {**prototype_config, **cfg}
        for key in ("model", "model_source", "model_local", "revision", "tokenizer_revision", "placement_contract"):
            if key in prototype_config and prototype_config[key] != cfg.get(key):
                raise ValueError(f"prototype config conflicts with verified {key}")
        if cfg.get("model") and not isinstance(cfg.get("placement_contract"), dict):
            raise ValueError("prototype explicit placement required")
        mod = importlib.import_module(KINDS[cfg["kind"]])
        entry = getattr(mod, "run_dispatch", None)
        if not callable(entry):
            raise ValueError("prototype runner requires run_dispatch(effective_config, config_bytes, out_dir)")
        root = Path(out_dir) / cfg["phase"]
        root.mkdir(parents=True, exist_ok=True)
        # The runner must enforce placement, local-only loading and emit genuine
        # resident control/fit receipts. Missing implementation fails closed.
        return entry(effective_config=effective, config_bytes=raw, out_dir=str(root))
    if cfg["kind"] == "associative_continuation_cpu":
        # The CPU CLI owns its config/control and never imports common.load.
        from contextlib import redirect_stdout
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "shard.json").write_text(json.dumps(cfg), encoding="utf-8")
        mod = importlib.import_module(KINDS[cfg["kind"]])
        with cpu_work_progress(root), (root / "cpu_receipt.json").open("w", encoding="utf-8") as output, redirect_stdout(output):
            return mod._cli(["--config", cfg.get("config", "experiments/memory/associative_continuation_cpu.json"), "--check", cfg.get("cpu_check", "receiver")])
    from .common import Out
    out = Out(out_dir)
    out.meta("shard.json", cfg)
    mod = importlib.import_module(f".{KINDS[cfg['kind']]}", __package__)
    getattr(mod, cfg.get("entry", "main"))(cfg, out)
    return int(any(k in out.done for k in ("e1|smoke_fail", "align|smoke_fail")))


if __name__ == "__main__":
    if sys.argv[1] == "--verified-json":
        raw = sys.argv[3].encode("utf-8")
        if hashlib.sha256(raw).hexdigest() != sys.argv[2]:
            raise SystemExit("verified input hash mismatch")
        sys.exit(execute(json.loads(raw), sys.argv[4]))
    raise SystemExit("production dispatcher requires immutable verified input")
