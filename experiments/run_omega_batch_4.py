"""Batch runner for experiment round 4: pair transfer, query transfer, cross-model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=== BATCH 4: PAIR TRANSFER ===", flush=True)
from run_omega_pair_transfer import run_pair_transfer
pair_obs = run_pair_transfer()
print(f"\nPair transfer done: {len(pair_obs)} observations\n", flush=True)

print("=== BATCH 4: QUERY TRANSFER ===", flush=True)
from run_omega_query_transfer import run_query_transfer
query_obs = run_query_transfer()
print(f"\nQuery transfer done: {len(query_obs)} observations\n", flush=True)

print("=== BATCH 4: CROSS-MODEL ===", flush=True)
from run_omega_crossmodel import run_crossmodel
crossmodel = run_crossmodel()
print(f"\nCross-model done\n", flush=True)

print("=== ALL BATCH 4 EXPERIMENTS COMPLETE ===", flush=True)
