"""Batch runner for experiment round 6: scaling (0.6B, 1.7B, 4B)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=== BATCH 6: SCALING ===", flush=True)
from run_omega_scaling import run_scaling
all_obs = run_scaling()
print(f"\nScaling done: {len(all_obs)} observations\n", flush=True)

print("=== ALL BATCH 6 EXPERIMENTS COMPLETE ===", flush=True)
