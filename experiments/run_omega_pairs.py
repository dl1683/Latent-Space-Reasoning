"""Surface Pair Discovery: Find pairs with small baseline behavioral distance.

The omega framework is data-starved because most equiv pairs have baseline
d > tau, yielding zero eligible cells. This experiment systematically tests
many candidate pairs across all 6 depth-4 cells to find pairs with d ≤ tau
at various thresholds.

Design principle: pairs should be semantically equivalent assertions that
the state hasn't changed, differing only in surface phrasing.

6 cells × 30 pairs × 2 sides = 360 readouts (~1.5 min)
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix


MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B-Base",
    "query_template": "f()\nprint({var})  # Output: ",
    "templates": {
        "depth4_single": (
            "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
            "    def g():\n        {var} = 999\n        def h():\n"
            "            {var} = 9999\n            def k():\n"
            "                {var} = 99999\n                return {var}\n"
            "            k()\n            return {var}\n"
            "        h()\n        return {var}\n"
            "    g()\n    return {var}\n"
        ),
    },
}

CANDIDATE_PAIRS = [
    # Known working pair (strong_assert_2) — control
    ("# State unchanged.\n", "# {var} stays the same.\n", "ctrl_strong_assert_2"),
    # Known working pair (moderate_assert) — control
    ("# Values preserved.\n", "# {var} retains its value.\n", "ctrl_moderate_assert"),

    # Minimal comment variants
    ("# OK.\n", "# Fine.\n", "minimal_ok_fine"),
    ("# OK.\n", "# Good.\n", "minimal_ok_good"),
    ("# Done.\n", "# OK.\n", "minimal_done_ok"),
    ("# Pass.\n", "# OK.\n", "minimal_pass_ok"),

    # Variable-free assertions
    ("# No change.\n", "# Nothing changed.\n", "novar_nochange"),
    ("# All good.\n", "# Everything OK.\n", "novar_allgood"),
    ("# Confirmed.\n", "# Verified.\n", "novar_confirmed"),
    ("# Checked.\n", "# Verified.\n", "novar_checked"),

    # Short variable assertions
    ("# {var} OK.\n", "# {var} fine.\n", "var_ok_fine"),
    ("# {var} set.\n", "# {var} ready.\n", "var_set_ready"),
    ("# {var} valid.\n", "# {var} OK.\n", "var_valid_ok"),

    # Structural variants (same meaning, different structure)
    ("# Check: {var}.\n", "# Verify: {var}.\n", "struct_check_verify"),
    ("# See {var}.\n", "# Note {var}.\n", "struct_see_note"),

    # Pure whitespace / formatting variants
    ("#\n", "# \n", "ws_hash_hashspace"),
    ("# .\n", "#.\n", "ws_dotspace"),
    ("pass\n", "pass \n", "ws_pass_trailing"),

    # Code-like assertions
    ("assert True\n", "assert 1\n", "code_assert_true_1"),
    ("_ = None\n", "_ = 0\n", "code_none_zero"),
    ("pass\n", "...\n", "code_pass_ellipsis"),

    # Longer but equivalent
    ("# The state is unchanged.\n", "# State remains as before.\n", "long_unchanged"),
    ("# No modifications made.\n", "# Nothing was modified.\n", "long_nomod"),
    ("# {var} has not changed.\n", "# {var} remains unchanged.\n", "long_var_unchanged"),

    # Terse vs verbose (same meaning)
    ("# ok\n", "# okay\n", "terse_ok_okay"),
    ("# nop\n", "# no-op\n", "terse_nop"),

    # Hash-only variants
    ("##\n", "###\n", "hash_double_triple"),
    ("# -\n", "# --\n", "hash_dash"),

    # Single-word assertions
    ("# Stable.\n", "# Steady.\n", "word_stable_steady"),
    ("# Safe.\n", "# Secure.\n", "word_safe_secure"),
]

CELLS = [
    (4, "x", 1), (4, "x", 5), (4, "x", 7),
    (4, "y", 2), (4, "y", 6), (4, "y", 8),
]


def run_pair_discovery():
    print("=== Surface Pair Discovery Sweep ===")
    n_total = len(CELLS) * len(CANDIDATE_PAIRS) * 2
    print(f"Cells: {len(CELLS)}, Pairs: {len(CANDIDATE_PAIRS)}")
    print(f"Total readouts: {n_total}")
    print()

    adapter = ModelAdapter(MODEL_CFG)
    query_tmpl = MODEL_CFG["query_template"]
    observations = []
    t0 = time.time()
    n = 0

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_a_tmpl, surf_b_tmpl, pair_name in CANDIDATE_PAIRS:
            for side, surf_tmpl in [("a", surf_a_tmpl), ("b", surf_b_tmpl)]:
                surface_text = surf_tmpl.replace("{var}", var)
                full_suffix = surface_text + query

                dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                n += 1

                observations.append({
                    "depth": depth, "var": var, "val": val,
                    "pair": pair_name, "side": side,
                    "dist": dist.tolist(),
                })

                if n % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"  {n}/{n_total} readouts, {elapsed:.0f}s, "
                          f"{n/elapsed:.1f}/s", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {n} readouts in {elapsed:.1f}s ({n/elapsed:.1f}/s)")

    result_dir = Path("experiments/results/omega_pairs")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    obs_idx = {}
    for obs in observations:
        key = (obs["var"], obs["val"], obs["pair"], obs["side"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    print("\n=== Pair Baseline Distances ===")
    print(f"{'Pair':<30s} {'mean_d':>8s} {'max_d':>8s} {'n_elig01':>8s} "
          f"{'n_elig02':>8s} {'n_elig05':>8s} {'n_cells':>7s}")
    print("-" * 90)

    pair_stats = {}
    for _, _, pair_name in CANDIDATE_PAIRS:
        tvs = []
        for _, var, val in CELLS:
            ka = (var, val, pair_name, "a")
            kb = (var, val, pair_name, "b")
            if ka in obs_idx and kb in obs_idx:
                d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                tvs.append(d)

        if tvs:
            mean_d = float(np.mean(tvs))
            max_d = float(np.max(tvs))
            n_01 = sum(1 for d in tvs if d <= 0.01)
            n_02 = sum(1 for d in tvs if d <= 0.02)
            n_05 = sum(1 for d in tvs if d <= 0.05)
            print(f"  {pair_name:<28s} {mean_d:8.5f} {max_d:8.5f} "
                  f"{n_01:>8d} {n_02:>8d} {n_05:>8d} {len(tvs):>7d}")
            pair_stats[pair_name] = {
                "mean_d": mean_d, "max_d": max_d,
                "n_cells": len(tvs), "distances": tvs,
                "n_eligible_01": n_01, "n_eligible_02": n_02,
                "n_eligible_05": n_05,
            }

    with open(result_dir / "pair_stats.json", "w") as f:
        json.dump(pair_stats, f, indent=2)

    good_pairs = {k: v for k, v in pair_stats.items() if v["n_eligible_02"] >= 3}
    print(f"\n=== Pairs with >=3 eligible cells at tau=0.02: {len(good_pairs)} ===")
    for name, stats in sorted(good_pairs.items(), key=lambda x: -x[1]["n_eligible_02"]):
        print(f"  {name}: {stats['n_eligible_02']}/6 cells, mean_d={stats['mean_d']:.5f}")

    print(f"\nSaved {n} observations to {result_dir}/")
    del adapter
    return pair_stats


if __name__ == "__main__":
    run_pair_discovery()
