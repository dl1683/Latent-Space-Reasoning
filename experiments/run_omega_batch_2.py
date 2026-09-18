"""Batch runner for experiment round 2: pairs, ablation, repair."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=== BATCH 2: PAIRS ===", flush=True)
from run_omega_pairs import run_pair_discovery
pair_stats = run_pair_discovery()
print(f"\nPairs done: {len(pair_stats)} pairs tested\n", flush=True)

print("=== BATCH 2: ABLATION ===", flush=True)
from run_omega_ablation import run_ablation
ablation_stats = run_ablation()
print(f"\nAblation done\n", flush=True)

print("=== BATCH 2: REPAIR ===", flush=True)
from run_omega_repair import run_repair
repair_obs = run_repair()
print(f"\nRepair done: {len(repair_obs)} observations\n", flush=True)

print("=== ALL BATCH 2 EXPERIMENTS COMPLETE ===", flush=True)
