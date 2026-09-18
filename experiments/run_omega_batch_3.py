"""Batch runner for experiment round 3: ablation, repair, stability."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=== BATCH 3: ABLATION ===", flush=True)
from run_omega_ablation import run_ablation
ablation_stats = run_ablation()
print(f"\nAblation done\n", flush=True)

print("=== BATCH 3: REPAIR ===", flush=True)
from run_omega_repair import run_repair
repair_obs = run_repair()
print(f"\nRepair done: {len(repair_obs)} observations\n", flush=True)

print("=== BATCH 3: STABILITY ===", flush=True)
from run_omega_stability import run_stability
stability = run_stability()
print(f"\nStability done\n", flush=True)

print("=== ALL BATCH 3 EXPERIMENTS COMPLETE ===", flush=True)
