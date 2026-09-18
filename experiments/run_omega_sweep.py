"""Omega Sweep: High-volume perturbation dataset builder.

Uses KV cache to share prefix computation across all suffixes per cell.
Each cell's prefix (~200 tokens) is computed once; each suffix variation
is only ~5-30 tokens of incremental work.

Target: 2000+ readouts in <15 minutes on CPU.
"""

import json
import sys
import time
import copy
import numpy as np
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix, build_query


MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B-Base",
    "query_template": "f()\nprint({var})  # Output: ",
    "templates": {
        "depth2_single": (
            "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
            "    def g():\n        {var} = 999\n        return {var}\n"
            "    g()\n    return {var}\n"
        ),
        "depth3_single": (
            "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
            "    def g():\n        {var} = 999\n        def h():\n"
            "            {var} = 9999\n            return {var}\n"
            "        h()\n        return {var}\n"
            "    g()\n    return {var}\n"
        ),
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

SURFACES = {
    "none": "",
    "assert_novar_1": "# No changes.\n",
    "assert_novar_2": "# State unchanged.\n",
    "assert_novar_3": "# Nothing modified.\n",
    "assert_novar_4": "# Values preserved.\n",
    "assert_novar_5": "# Same as before.\n",
    "assert_var_1": "# {var} is unchanged.\n",
    "assert_var_2": "# {var} stays the same.\n",
    "assert_var_3": "# {var} not modified.\n",
    "assert_var_4": "# {var} retains its value.\n",
    "mislead_var_1": "# Reassigning {var} now.\n",
    "mislead_var_2": "# Updating {var}.\n",
    "mislead_var_3": "# Setting {var} to new value.\n",
    "mislead_var_4": "# Changing {var}.\n",
    "mislead_novar_1": "# Reassigning the value now.\n",
    "mislead_novar_2": "# Overwriting the variable.\n",
    "mislead_novar_3": "# Updating to a new value.\n",
    "rewrite_1": "{var} = {var}\n",
    "rewrite_2": "{var} = +{var}\n",
    "rewrite_3": "{var} = int({var})\n",
    "rewrite_4": "{var} = ({var})\n",
    "observe_1": "_ = {var}\n",
    "observe_2": "type({var})\n",
    "observe_3": "isinstance({var}, int)\n",
    "boundary_1": "\n",
    "boundary_2": "pass\n",
    "boundary_3": "...\n",
    "boundary_4": "\n\n",
}

CONTINUATIONS = {
    "none": "",
    "neutral_1": "# No changes.\n",
    "neutral_2": "# State unchanged.\n",
    "neutral_3": "# Values preserved.\n",
    "heavy_1": "# Reassigning {var} now.\n",
    "heavy_2": "# Updating {var}.\n",
    "heavy_3": "# Changing {var}.\n",
    "bare_nl": "\n",
    "pass": "pass\n",
    "ellipsis": "...\n",
    "identity": "{var} = {var}\n",
    "observe": "_ = {var}\n",
    "verified": "# Verified.\n",
    "ok": "# OK.\n",
    "double_neutral": "# No changes.\n# No changes.\n",
    "neutral_then_heavy": "# No changes.\n# Reassigning {var} now.\n",
    "heavy_then_neutral": "# Reassigning {var} now.\n# No changes.\n",
    "type_check": "type({var})\n",
    "assert_true": "assert True\n",
    "comment_var": "# {var}\n",
    "print_var": "print({var})\n",
}

CELLS_FAST = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

CELLS_FULL = [
    (d, v, val)
    for d in [2, 3, 4]
    for v in ["x", "y", "z"]
    for val in [1, 3, 5, 7]
]

EQUIV_PAIRS = [
    ("assert_novar_1", "assert_var_1", "strong_assert"),
    ("assert_novar_2", "assert_var_2", "strong_assert_2"),
    ("assert_novar_4", "assert_var_4", "moderate_assert"),
    ("mislead_var_1", "mislead_novar_1", "misleading"),
    ("mislead_var_4", "mislead_novar_3", "misleading_2"),
    ("rewrite_1", "rewrite_4", "rewrite_pair"),
    ("boundary_1", "boundary_2", "boundary_pair"),
    ("observe_1", "observe_2", "observe_pair"),
]


def run_sweep(cells, surfaces, continuations, cfg, equiv_pairs=None):
    """Run a full sweep and return structured data.

    For each cell, compute prefix state once. Then for each surface,
    compute surface state (prefix + surface). Then for each continuation,
    compute the final readout (prefix + surface + continuation + query).

    Returns list of observation dicts.
    """
    adapter = ModelAdapter(cfg)
    query_tmpl = cfg["query_template"]
    observations = []
    t0 = time.time()
    n_readouts = 0

    for depth, var, val in cells:
        tmpl_key = f"depth{depth}_single"
        tmpl = cfg["templates"][tmpl_key]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)

        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_name, surf_tmpl in surfaces.items():
            surface_text = surf_tmpl.replace("{var}", var)
            suffix_for_query = surface_text + query if surface_text else query

            if surface_text:
                surf_state = adapter.get_state_after_prefix(prefix_text + surface_text)
            else:
                surf_state = prefix_state

            for cont_name, cont_tmpl in continuations.items():
                cont_text = cont_tmpl.replace("{var}", var)

                if cont_text:
                    full_suffix = surface_text + cont_text + query
                else:
                    full_suffix = suffix_for_query

                dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                n_readouts += 1

                observations.append({
                    "depth": depth,
                    "var": var,
                    "val": val,
                    "surface": surf_name,
                    "continuation": cont_name,
                    "dist": dist.tolist(),
                })

                if n_readouts % 100 == 0:
                    elapsed = time.time() - t0
                    rate = n_readouts / elapsed
                    print(f"  {n_readouts} readouts, {elapsed:.0f}s, "
                          f"{rate:.1f}/s", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {n_readouts} readouts in {elapsed:.1f}s "
          f"({n_readouts/elapsed:.1f}/s), "
          f"{adapter.call_count} model calls", flush=True)

    del adapter
    return observations


def compute_pair_distances(observations, equiv_pairs, continuations):
    """For each equiv pair, compute TV distances across all cells and continuations."""
    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    results = {}
    for surf_a, surf_b, pair_name in equiv_pairs:
        pair_data = []
        for obs in observations:
            if obs["surface"] != surf_a:
                continue
            key_a = (obs["depth"], obs["var"], obs["val"],
                     surf_a, obs["continuation"])
            key_b = (obs["depth"], obs["var"], obs["val"],
                     surf_b, obs["continuation"])
            if key_a in obs_idx and key_b in obs_idx:
                dist_a = obs_idx[key_a]
                dist_b = obs_idx[key_b]
                d = 0.5 * np.abs(dist_a - dist_b).sum()
                pair_data.append({
                    "depth": obs["depth"], "var": obs["var"], "val": obs["val"],
                    "continuation": obs["continuation"],
                    "tv": float(d),
                })
        results[pair_name] = pair_data
    return results


def compute_omega_table(pair_distances, tau):
    """Compute omega modulus for each continuation across each pair type."""
    table = {}
    for pair_name, pairs in pair_distances.items():
        conts = sorted(set(p["continuation"] for p in pairs))

        immediate = [p for p in pairs if p["continuation"] == "none"]
        eligible_cells = set()
        for p in immediate:
            if p["tv"] <= tau:
                eligible_cells.add((p["depth"], p["var"], p["val"]))

        cont_results = {}
        for cn in conts:
            post = [p for p in pairs
                    if p["continuation"] == cn
                    and (p["depth"], p["var"], p["val"]) in eligible_cells]
            if not post:
                cont_results[cn] = {
                    "omega": None, "n_eligible": 0, "assessable": False,
                }
                continue
            omega = max(p["tv"] for p in post)
            preserved = sum(1 for p in post if p["tv"] <= tau)
            cont_results[cn] = {
                "omega": float(omega),
                "preserves": omega <= tau,
                "n_eligible": len(post),
                "preserved_count": preserved,
                "preservation_rate": preserved / len(post),
                "mean_tv": float(np.mean([p["tv"] for p in post])),
                "max_tv": float(omega),
                "assessable": True,
            }
        table[pair_name] = {
            "n_eligible_cells": len(eligible_cells),
            "continuations": cont_results,
        }
    return table


def run_fast_sweep():
    """Quick 6-cell sweep with full surface × continuation grid."""
    print("=== Omega Sweep: Fast (6 cells × 28 surfaces × 21 continuations) ===")
    n = len(CELLS_FAST) * len(SURFACES) * len(CONTINUATIONS)
    print(f"Target: {n} readouts")
    print()

    observations = run_sweep(CELLS_FAST, SURFACES, CONTINUATIONS, MODEL_CFG)

    result_dir = Path("experiments/results/omega_sweep_fast")
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    pair_dists = compute_pair_distances(observations, EQUIV_PAIRS, CONTINUATIONS)
    with open(result_dir / "pair_distances.json", "w") as f:
        json.dump(pair_dists, f, indent=2)

    print("\n=== Omega Table (tau=0.01) ===")
    omega_table = compute_omega_table(pair_dists, 0.01)
    with open(result_dir / "omega_table.json", "w") as f:
        json.dump(omega_table, f, indent=2)

    for pair_name, pt in omega_table.items():
        print(f"\n--- {pair_name} ({pt['n_eligible_cells']} eligible cells) ---")
        if pt["n_eligible_cells"] == 0:
            print("  No eligible cells at tau=0.01")
            continue
        for cn, cr in sorted(pt["continuations"].items()):
            if not cr["assessable"]:
                continue
            status = "PRESERVE" if cr["preserves"] else "AMPLIFY"
            print(f"  {cn:25s} omega={cr['omega']:.5f} "
                  f"mean={cr['mean_tv']:.5f} "
                  f"{cr['preserved_count']}/{cr['n_eligible']} "
                  f"{status}")

    print(f"\nSaved {len(observations)} observations to {result_dir}/")
    return observations, omega_table


def run_full_sweep():
    """Full 36-cell sweep across depths 2-4."""
    print("=== Omega Sweep: Full (36 cells × 28 surfaces × 21 continuations) ===")
    n = len(CELLS_FULL) * len(SURFACES) * len(CONTINUATIONS)
    print(f"Target: {n} readouts")
    print()

    observations = run_sweep(CELLS_FULL, SURFACES, CONTINUATIONS, MODEL_CFG)

    result_dir = Path("experiments/results/omega_sweep_full")
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    pair_dists = compute_pair_distances(observations, EQUIV_PAIRS, CONTINUATIONS)
    with open(result_dir / "pair_distances.json", "w") as f:
        json.dump(pair_dists, f, indent=2)

    for tau in [0.005, 0.01, 0.02, 0.05]:
        print(f"\n=== Omega Table (tau={tau}) ===")
        omega_table = compute_omega_table(pair_dists, tau)
        with open(result_dir / f"omega_table_tau{tau}.json", "w") as f:
            json.dump(omega_table, f, indent=2)

        for pair_name, pt in omega_table.items():
            print(f"\n--- {pair_name} ({pt['n_eligible_cells']} eligible cells) ---")
            if pt["n_eligible_cells"] == 0:
                print("  No eligible cells")
                continue
            for cn, cr in sorted(pt["continuations"].items()):
                if not cr["assessable"]:
                    continue
                status = "PRESERVE" if cr["preserves"] else "AMPLIFY"
                print(f"  {cn:25s} omega={cr['omega']:.5f} "
                      f"mean={cr['mean_tv']:.5f} "
                      f"{cr['preserved_count']}/{cr['n_eligible']} "
                      f"{status}")

    print(f"\nSaved {len(observations)} observations to {result_dir}/")
    return observations


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fast"
    if mode == "fast":
        run_fast_sweep()
    elif mode == "full":
        run_full_sweep()
    else:
        print(f"Unknown mode: {mode}. Use 'fast' or 'full'.")
