"""Batch runner for experiment round 5: multi-pair splits (analysis only) + pair x depth."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=== BATCH 5: MULTI-PAIR SPLITS (analysis only) ===", flush=True)
from run_omega_multi_pair_splits import run_multi_pair_splits
splits = run_multi_pair_splits()
print(f"\nMulti-pair splits done\n", flush=True)

print("=== BATCH 5: PAIR x DEPTH ===", flush=True)
from run_omega_pair_depth import run_pair_depth
depth_obs = run_pair_depth()
print(f"\nPair x depth done: {len(depth_obs)} observations\n", flush=True)

print("=== ALL BATCH 5 EXPERIMENTS COMPLETE ===", flush=True)
