"""Omega Fixed-Cell Comparison (Experiment #48).

Resolves the tau-dependent eligibility confound from exp #46.
The problem: different tau thresholds include different cells, and
cells differ in which continuations amplify most. This makes it
impossible to compare continuations cleanly across thresholds.

Solution: fix the eligible cell set BEFORE comparing continuations.
For each pair, identify cells eligible at tau=0.05 under the EMPTY
baseline, then compare ALL continuations on those SAME cells.

All continuations end with newline to prevent boundary gluing (exp #46).
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv


MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B",
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

SURFACES = {
    "assert_var_2": "# {var} stays the same.\n",
    "assert_novar_2": "# State unchanged.\n",
    "boundary_1": "\n",
    "boundary_2": "pass\n",
    "minimal_ok": "# OK.\n",
    "minimal_good": "# Good.\n",
}

EQUIV_PAIRS = [
    ("assert_var_2", "assert_novar_2", "strong_assert_2"),
    ("boundary_1", "boundary_2", "boundary_pair"),
    ("minimal_ok", "minimal_good", "minimal_ok_good"),
]

CONTINUATIONS = {
    "empty":           "",
    "identity":        "{var} = {var}\n",
    "nonvar_42":       "_ = 42\n",
    "nonvar_99":       "_ = 99\n",
    "heavy":           "{var} = {var} + 1  # WARNING: intentional mutation\n",
    "neutral_pass":    "pass\n",
    "neutral_comment": "# check.\n",
    "type_check":      "type({var})\n",
    "double_nonvar":   "_ = 42\n_ = 43\n",
    "double_identity": "{var} = {var}\n{var} = {var}\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]


def run_fixed_cell_experiment():
    print("=== Omega Fixed-Cell Comparison (Exp #48) ===")
    n_target = len(CELLS) * len(SURFACES) * len(CONTINUATIONS)
    print(f"Target: {n_target} readouts")
    print(f"Cells: {len(CELLS)}, Surfaces: {len(SURFACES)}, "
          f"Continuations: {len(CONTINUATIONS)}")

    adapter = ModelAdapter(MODEL_CFG)
    query_tmpl = MODEL_CFG["query_template"]
    observations = []
    t0 = time.time()
    n_readouts = 0

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_name, surf_tmpl in SURFACES.items():
            surface_text = surf_tmpl.replace("{var}", var)

            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)
                if cont_text == "" or cont_text.endswith("\n"):
                    full_suffix = surface_text + cont_text + query
                else:
                    full_suffix = surface_text + cont_text + "\n" + query

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
                    print(f"  {n_readouts} readouts, {elapsed:.0f}s",
                          flush=True)

    elapsed = time.time() - t0
    print(f"Done: {n_readouts} readouts in {elapsed:.1f}s")

    result_dir = Path("experiments/results/omega_fixed_cell")
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    print("\n=== Fixed-Cell Analysis ===")
    summary = {"n_readouts": n_readouts, "elapsed_s": elapsed, "pairs": {}}

    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        print(f"\n--- {pair_name} ---")

        # Step 1: compute baseline TV per cell under empty continuation
        baseline_tvs = {}
        for depth, var, val in CELLS:
            key_a = (depth, var, val, surf_a, "empty")
            key_b = (depth, var, val, surf_b, "empty")
            if key_a in obs_idx and key_b in obs_idx:
                baseline_tvs[(depth, var, val)] = tv(obs_idx[key_a],
                                                     obs_idx[key_b])

        # Step 2: identify eligible cells at tau=0.05
        eligible_cells = [c for c, btv in baseline_tvs.items() if btv <= 0.05]
        eligible_cells_001 = [c for c, btv in baseline_tvs.items()
                              if btv <= 0.01]

        print(f"  Baseline TVs: {', '.join(f'{c[1]}{c[2]}={baseline_tvs[c]:.4f}' for c in sorted(baseline_tvs))}")
        print(f"  Eligible at tau=0.05: {len(eligible_cells)} cells")
        print(f"  Eligible at tau=0.01: {len(eligible_cells_001)} cells")

        if not eligible_cells:
            print("  NO eligible cells — skipping pair")
            summary["pairs"][pair_name] = {
                "n_eligible_005": 0,
                "n_eligible_001": 0,
                "baseline_tvs": {f"{c[1]}{c[2]}": float(v)
                                 for c, v in baseline_tvs.items()},
            }
            continue

        # Step 3: for FIXED eligible cells, compute omega for each continuation
        pair_results = {
            "n_eligible_005": len(eligible_cells),
            "n_eligible_001": len(eligible_cells_001),
            "baseline_tvs": {f"{c[1]}{c[2]}": float(v)
                             for c, v in sorted(baseline_tvs.items())},
            "fixed_cell_ranking_005": {},
            "fixed_cell_ranking_001": {},
            "per_cell_data": {},
        }

        for cont_name in CONTINUATIONS:
            # Post-action TVs on fixed eligible cells (tau=0.05)
            post_tvs_005 = []
            for cell in eligible_cells:
                depth, var, val = cell
                key_a = (depth, var, val, surf_a, cont_name)
                key_b = (depth, var, val, surf_b, cont_name)
                if key_a in obs_idx and key_b in obs_idx:
                    post_tvs_005.append(tv(obs_idx[key_a], obs_idx[key_b]))

            # Post-action TVs on fixed eligible cells (tau=0.01)
            post_tvs_001 = []
            for cell in eligible_cells_001:
                depth, var, val = cell
                key_a = (depth, var, val, surf_a, cont_name)
                key_b = (depth, var, val, surf_b, cont_name)
                if key_a in obs_idx and key_b in obs_idx:
                    post_tvs_001.append(tv(obs_idx[key_a], obs_idx[key_b]))

            if post_tvs_005:
                pair_results["fixed_cell_ranking_005"][cont_name] = {
                    "omega_max": float(max(post_tvs_005)),
                    "omega_mean": float(np.mean(post_tvs_005)),
                    "all_tvs": [float(t) for t in post_tvs_005],
                }

            if post_tvs_001:
                pair_results["fixed_cell_ranking_001"][cont_name] = {
                    "omega_max": float(max(post_tvs_001)),
                    "omega_mean": float(np.mean(post_tvs_001)),
                    "all_tvs": [float(t) for t in post_tvs_001],
                }

            # Per-cell data (all cells, not just eligible)
            all_post_tvs = []
            for cell in sorted(baseline_tvs.keys()):
                depth, var, val = cell
                key_a = (depth, var, val, surf_a, cont_name)
                key_b = (depth, var, val, surf_b, cont_name)
                if key_a in obs_idx and key_b in obs_idx:
                    all_post_tvs.append(float(tv(obs_idx[key_a],
                                                 obs_idx[key_b])))
            pair_results["per_cell_data"][cont_name] = all_post_tvs

        # Sort rankings by omega_max
        rank_005 = sorted(pair_results["fixed_cell_ranking_005"].items(),
                          key=lambda x: x[1]["omega_max"], reverse=True)
        rank_001 = sorted(pair_results["fixed_cell_ranking_001"].items(),
                          key=lambda x: x[1]["omega_max"], reverse=True)

        print(f"\n  Fixed-cell ranking (tau=0.05 eligible, {len(eligible_cells)} cells):")
        for name, data in rank_005:
            print(f"    {name:20s}  max={data['omega_max']:.4f}  "
                  f"mean={data['omega_mean']:.4f}")

        if rank_001:
            print(f"\n  Fixed-cell ranking (tau=0.01 eligible, {len(eligible_cells_001)} cells):")
            for name, data in rank_001:
                print(f"    {name:20s}  max={data['omega_max']:.4f}  "
                      f"mean={data['omega_mean']:.4f}")

        # Comparisons
        r005 = pair_results["fixed_cell_ranking_005"]
        comparisons = {}
        if "identity" in r005 and "nonvar_42" in r005:
            id_max = r005["identity"]["omega_max"]
            nv_max = r005["nonvar_42"]["omega_max"]
            comparisons["identity_vs_nonvar42_max"] = {
                "identity": id_max,
                "nonvar_42": nv_max,
                "ratio": id_max / nv_max if nv_max > 0 else None,
                "identity_wins": id_max > nv_max,
            }
            id_mean = r005["identity"]["omega_mean"]
            nv_mean = r005["nonvar_42"]["omega_mean"]
            comparisons["identity_vs_nonvar42_mean"] = {
                "identity": id_mean,
                "nonvar_42": nv_mean,
                "ratio": id_mean / nv_mean if nv_mean > 0 else None,
                "identity_wins": id_mean > nv_mean,
            }

        if "nonvar_42" in r005 and "nonvar_99" in r005:
            comparisons["nonvar42_vs_nonvar99"] = {
                "nonvar_42": r005["nonvar_42"]["omega_max"],
                "nonvar_99": r005["nonvar_99"]["omega_max"],
                "value_matters": abs(r005["nonvar_42"]["omega_max"] -
                                     r005["nonvar_99"]["omega_max"]) > 0.01,
            }

        if "double_nonvar" in r005 and "nonvar_42" in r005:
            comparisons["double_vs_single_nonvar"] = {
                "double": r005["double_nonvar"]["omega_max"],
                "single": r005["nonvar_42"]["omega_max"],
                "more_computation_more_amplification":
                    r005["double_nonvar"]["omega_max"] >
                    r005["nonvar_42"]["omega_max"],
            }

        if "double_identity" in r005 and "identity" in r005:
            comparisons["double_vs_single_identity"] = {
                "double": r005["double_identity"]["omega_max"],
                "single": r005["identity"]["omega_max"],
                "more_computation_more_amplification":
                    r005["double_identity"]["omega_max"] >
                    r005["identity"]["omega_max"],
            }

        pair_results["comparisons"] = comparisons
        summary["pairs"][pair_name] = pair_results

        if comparisons:
            print(f"\n  Comparisons:")
            for k, v in comparisons.items():
                print(f"    {k}: {json.dumps(v, default=str)}")

    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {result_dir}")
    return summary


if __name__ == "__main__":
    run_fixed_cell_experiment()
