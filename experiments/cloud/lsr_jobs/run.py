"""VM entry point: python -m lsr_jobs.run <shard.json> <out_dir>. Dispatches on shard["kind"]."""
import importlib
import json
import sys

from .common import Out

KINDS = {"smoke": "smoke", "recur": "recur", "search": "search", "branch": "branch", "p2": "p2", "fd": "fd", "native": "native", "q1pp": "q1pp", "readout": "readout", "gemma_controls": "gemma_controls", "bindq": "bindq", "formation": "formation"}

if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1]))
    out = Out(sys.argv[2])
    out.meta("shard.json", cfg)
    mod = importlib.import_module(f".{KINDS[cfg['kind']]}", __package__)
    getattr(mod, cfg.get("entry", "main"))(cfg, out)
