"""VM entry point: python -m lsr_jobs.run <shard.json> <out_dir>. Dispatches on shard["kind"]."""
import importlib
import json
import sys

from .common import Out

KINDS = {"smoke": "smoke", "recur": "recur", "search": "search", "branch": "branch", "p2": "p2", "fd": "fd", "native": "native", "formation": "formation"}

if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1]))
    out = Out(sys.argv[2])
    out.meta("shard.json", cfg)
    importlib.import_module(f".{KINDS[cfg['kind']]}", __package__).main(cfg, out)
