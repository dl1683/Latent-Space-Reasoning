"""Omega Partial Identity Analysis (Experiment #38).

Tests which tokens in `x = x\n` trigger the omega phase transition.
The phase experiment (#31) found a step function at rep=1 — omega jumps
from 0.003 to 0.047 with one identity statement. This experiment
decomposes the identity into sub-token fragments to localize the trigger.

Fragments tested:
  - x           (variable name only)
  - x =         (assignment operator)
  - x = x       (complete identity, no newline)
  - x = x\n     (full identity)
  - x =\n       (dangling assignment)
  - = x\n       (no LHS)
  - x = y\n     (cross-variable assignment)
  - x = 1\n     (literal assignment)
  - x = None\n  (None assignment)
  - # x\n       (variable in comment)
  - \n           (bare newline, baseline)
  - pass\n      (pass statement, baseline)
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
    "assert_novar_1": "# No changes.\n",
    "assert_var_1": "# {var} is unchanged.\n",
    "assert_novar_2": "# State unchanged.\n",
    "assert_var_2": "# {var} stays the same.\n",
    "boundary_1": "\n",
    "boundary_2": "pass\n",
    "minimal_ok": "# OK.\n",
    "minimal_good": "# Good.\n",
}

EQUIV_PAIRS = [
    ("assert_novar_1", "assert_var_1", "strong_assert"),
    ("assert_novar_2", "assert_var_2", "strong_assert_2"),
    ("boundary_1", "boundary_2", "boundary_pair"),
    ("minimal_ok", "minimal_good", "minimal_ok_good"),
]

PARTIAL_CONTINUATIONS = {
    "baseline_none": "",
    "bare_newline": "\n",
    "pass_stmt": "pass\n",
    "var_only": "{var}",
    "var_equals": "{var} = ",
    "identity_no_nl": "{var} = {var}",
    "identity_full": "{var} = {var}\n",
    "dangling_assign": "{var} =\n",
    "no_lhs": "= {var}\n",
    "cross_var_assign": "{var} = y\n",
    "literal_assign": "{var} = 1\n",
    "none_assign": "{var} = None\n",
    "comment_var": "# {var}\n",
    "type_check": "type({var})\n",
    "observe": "_ = {var}\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]


def run_partial_experiment():
    print("=== Omega Partial Identity Analysis ===")
    n_target = len(CELLS) * len(SURFACES) * len(PARTIAL_CONTINUATIONS)
    print(f"Target: {n_target} readouts")
    print(f"Cells: {len(CELLS)}, Surfaces: {len(SURFACES)}, "
          f"Continuations: {len(PARTIAL_CONTINUATIONS)}")

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

            for cont_name, cont_tmpl in PARTIAL_CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)
                full_suffix = surface_text + cont_text + query
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
                    print(f"  {n_readouts} readouts, {elapsed:.0f}s, "
                          f"{n_readouts/elapsed:.1f}/s", flush=True)

    elapsed = time.time() - t0
    print(f"Done: {n_readouts} readouts in {elapsed:.1f}s "
          f"({n_readouts/elapsed:.1f}/s)")

    result_dir = Path("experiments/results/omega_partial")
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    print("\n=== Pair Distances by Continuation Fragment ===")
    summary = {}
    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        print(f"\n--- {pair_name} ---")
        cont_results = {}
        for cont_name in PARTIAL_CONTINUATIONS:
            tvs = []
            for depth, var, val in CELLS:
                key_a = (depth, var, val, surf_a, cont_name)
                key_b = (depth, var, val, surf_b, cont_name)
                if key_a in obs_idx and key_b in obs_idx:
                    tvs.append(tv(obs_idx[key_a], obs_idx[key_b]))
            if tvs:
                mean_tv = float(np.mean(tvs))
                max_tv = float(np.max(tvs))
                cont_results[cont_name] = {
                    "mean_tv": mean_tv,
                    "max_tv": max_tv,
                    "n": len(tvs),
                }
        cont_results_sorted = sorted(
            cont_results.items(), key=lambda x: x[1]["mean_tv"], reverse=True)
        for cn, cr in cont_results_sorted:
            print(f"  {cn:22s}: mean_tv={cr['mean_tv']:.5f} "
                  f"max={cr['max_tv']:.5f}")
        summary[pair_name] = cont_results

    baseline_conts = ["baseline_none", "bare_newline", "pass_stmt"]
    print("\n=== Omega at tau=0.01 (eligible cells only) ===")
    omega_summary = {}
    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        eligible_cells = set()
        for depth, var, val in CELLS:
            key_a = (depth, var, val, surf_a, "baseline_none")
            key_b = (depth, var, val, surf_b, "baseline_none")
            if key_a in obs_idx and key_b in obs_idx:
                d = tv(obs_idx[key_a], obs_idx[key_b])
                if d <= 0.01:
                    eligible_cells.add((depth, var, val))

        if not eligible_cells:
            for depth, var, val in CELLS:
                key_a = (depth, var, val, surf_a, "baseline_none")
                key_b = (depth, var, val, surf_b, "baseline_none")
                if key_a in obs_idx and key_b in obs_idx:
                    d = tv(obs_idx[key_a], obs_idx[key_b])
                    if d <= 0.05:
                        eligible_cells.add((depth, var, val))
            tau_used = 0.05
        else:
            tau_used = 0.01

        print(f"\n--- {pair_name} ({len(eligible_cells)} eligible at "
              f"tau={tau_used}) ---")
        pair_omega = {}
        for cont_name in PARTIAL_CONTINUATIONS:
            tvs = []
            for cell in eligible_cells:
                depth, var, val = cell
                key_a = (depth, var, val, surf_a, cont_name)
                key_b = (depth, var, val, surf_b, cont_name)
                if key_a in obs_idx and key_b in obs_idx:
                    tvs.append(tv(obs_idx[key_a], obs_idx[key_b]))
            if tvs:
                omega = float(np.max(tvs))
                mean = float(np.mean(tvs))
                status = "PRESERVE" if omega <= tau_used else "AMPLIFY"
                pair_omega[cont_name] = {
                    "omega": omega, "mean": mean, "status": status
                }

        pair_omega_sorted = sorted(
            pair_omega.items(), key=lambda x: x[1]["omega"], reverse=True)
        for cn, cr in pair_omega_sorted:
            print(f"  {cn:22s}: omega={cr['omega']:.5f} "
                  f"mean={cr['mean']:.5f} {cr['status']}")
        omega_summary[pair_name] = pair_omega

    with open(result_dir / "summary.json", "w") as f:
        json.dump({
            "n_readouts": n_readouts,
            "elapsed_s": elapsed,
            "pair_distances": summary,
            "omega_eligible": omega_summary,
        }, f, indent=2)

    print(f"\nSaved to {result_dir}/")
    return observations, summary, omega_summary


if __name__ == "__main__":
    run_partial_experiment()
