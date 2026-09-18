"""Pair-Level Out-of-Sample Test: Does omega generalize across surface pairs?

THE decisive test for omega's predictive power:
- Compute omega using one set of surface pairs (calibration pairs)
- Predict preservation for a DIFFERENT set of surface pairs (evaluation pairs)
- If omega transfers, it measures the continuation's effect on state,
  not artifacts of specific pair phrasings

Uses the newly discovered pairs from run_omega_pairs.py:
- terse_ok_okay ("# ok\n" / "# okay\n") — 6/6 eligible at tau=0.02
- minimal_ok_good ("# OK.\n" / "# Good.\n") — 6/6 eligible
- minimal_ok_fine ("# OK.\n" / "# Fine.\n") — 6/6 eligible
- novar_confirmed ("# Confirmed.\n" / "# Verified.\n") — 6/6 eligible
- ctrl_strong_assert_2 ("# State unchanged.\n" / "# {var} stays the same.\n") — 6/6 eligible

Strategy: leave-one-pair-out cross-validation.
For each pair: train omega on the other 4, predict on this one.

5 pairs × 6 cells × 2 sides × 8 continuations = 480 readouts (~2 min)
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

SURFACE_PAIRS = [
    ("# State unchanged.\n", "# {var} stays the same.\n", "strong_assert_2"),
    ("# ok\n", "# okay\n", "terse_ok_okay"),
    ("# OK.\n", "# Good.\n", "minimal_ok_good"),
    ("# OK.\n", "# Fine.\n", "minimal_ok_fine"),
    ("# Confirmed.\n", "# Verified.\n", "novar_confirmed"),
]

CONTINUATIONS = {
    "none": "",
    "neutral": "# State unchanged.\n",
    "heavy": "# Reassigning {var} now.\n",
    "pass": "pass\n",
    "identity": "{var} = {var}\n",
    "type_check": "type({var})\n",
    "observe": "_ = {var}\n",
    "bare_nl": "\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5), (4, "x", 7),
    (4, "y", 2), (4, "y", 6), (4, "y", 8),
]

HEAVY_KW = ["heavy", "reassign", "updat", "chang"]


def run_pair_transfer():
    print("=== Pair-Level Out-of-Sample Test ===")
    n_total = len(CELLS) * len(SURFACE_PAIRS) * 2 * len(CONTINUATIONS)
    print(f"Cells: {len(CELLS)}, Pairs: {len(SURFACE_PAIRS)}, "
          f"Continuations: {len(CONTINUATIONS)}")
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

        for surf_a_tmpl, surf_b_tmpl, pair_name in SURFACE_PAIRS:
            for side, surf_tmpl in [("a", surf_a_tmpl), ("b", surf_b_tmpl)]:
                surface_text = surf_tmpl.replace("{var}", var)

                for cont_name, cont_tmpl in CONTINUATIONS.items():
                    cont_text = cont_tmpl.replace("{var}", var)
                    full_suffix = surface_text + cont_text + query

                    dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                    n += 1

                    observations.append({
                        "depth": depth, "var": var, "val": val,
                        "pair": pair_name, "side": side,
                        "continuation": cont_name,
                        "dist": dist.tolist(),
                    })

                    if n % 100 == 0:
                        elapsed = time.time() - t0
                        print(f"  {n}/{n_total} readouts, {elapsed:.0f}s, "
                              f"{n/elapsed:.1f}/s", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {n} readouts in {elapsed:.1f}s ({n/elapsed:.1f}/s)")

    result_dir = Path("experiments/results/omega_pair_transfer")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    obs_idx = {}
    for obs in observations:
        key = (obs["var"], obs["val"], obs["pair"], obs["side"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    pair_names = [p[2] for p in SURFACE_PAIRS]

    for tau in [0.01, 0.02, 0.05]:
        print(f"\n=== Leave-One-Pair-Out Cross-Validation (tau={tau}) ===")

        all_omega_correct = 0
        all_baseline_correct = 0
        all_total = 0

        for eval_pair in pair_names:
            train_pairs = [p for p in pair_names if p != eval_pair]

            train_omega = {}
            for cont_name in CONTINUATIONS:
                max_tvs_across_pairs = []
                for tp in train_pairs:
                    tvs = []
                    for _, var, val in CELLS:
                        ka_base = (var, val, tp, "a", "none")
                        kb_base = (var, val, tp, "b", "none")
                        if ka_base not in obs_idx or kb_base not in obs_idx:
                            continue
                        d_pre = 0.5 * np.abs(obs_idx[ka_base] - obs_idx[kb_base]).sum()
                        if d_pre > tau:
                            continue
                        ka = (var, val, tp, "a", cont_name)
                        kb = (var, val, tp, "b", cont_name)
                        if ka in obs_idx and kb in obs_idx:
                            d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                            tvs.append(d)
                    if tvs:
                        max_tvs_across_pairs.append(max(tvs))

                if max_tvs_across_pairs:
                    train_omega[cont_name] = max(max_tvs_across_pairs)

            omega_correct = 0
            baseline_correct = 0
            total = 0

            for cont_name in CONTINUATIONS:
                if cont_name not in train_omega:
                    continue

                eval_tvs = []
                for _, var, val in CELLS:
                    ka_base = (var, val, eval_pair, "a", "none")
                    kb_base = (var, val, eval_pair, "b", "none")
                    if ka_base not in obs_idx or kb_base not in obs_idx:
                        continue
                    d_pre = 0.5 * np.abs(obs_idx[ka_base] - obs_idx[kb_base]).sum()
                    if d_pre > tau:
                        continue
                    ka = (var, val, eval_pair, "a", cont_name)
                    kb = (var, val, eval_pair, "b", cont_name)
                    if ka in obs_idx and kb in obs_idx:
                        d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                        eval_tvs.append(d)

                if not eval_tvs:
                    continue

                actual = "preserve" if max(eval_tvs) <= tau else "separate"
                omega_pred = "preserve" if train_omega[cont_name] <= tau else "separate"

                cn_lower = cont_name.lower()
                is_heavy = any(k in cn_lower for k in HEAVY_KW)
                baseline_pred = "separate" if is_heavy else "preserve"

                total += 1
                omega_correct += int(omega_pred == actual)
                baseline_correct += int(baseline_pred == actual)

                if omega_pred != baseline_pred:
                    marker = "omega" if omega_pred == actual else "baseline"
                    print(f"  eval={eval_pair:20s} {cont_name:12s} "
                          f"omega={omega_pred:8s} base={baseline_pred:8s} "
                          f"actual={actual:8s} [{marker}]")

            all_omega_correct += omega_correct
            all_baseline_correct += baseline_correct
            all_total += total

            if total > 0:
                print(f"  eval={eval_pair}: omega={omega_correct}/{total} "
                      f"({100*omega_correct/total:.0f}%), "
                      f"baseline={baseline_correct}/{total} "
                      f"({100*baseline_correct/total:.0f}%)")

        if all_total > 0:
            print(f"\n  OVERALL (tau={tau}): omega={all_omega_correct}/{all_total} "
                  f"({100*all_omega_correct/all_total:.1f}%), "
                  f"baseline={all_baseline_correct}/{all_total} "
                  f"({100*all_baseline_correct/all_total:.1f}%)")
            print(f"  Advantage: +{100*(all_omega_correct-all_baseline_correct)/all_total:.1f}pp")

    with open(result_dir / "cross_validation.json", "w") as f:
        json.dump({
            "method": "leave-one-pair-out",
            "pairs": pair_names,
            "n_observations": n,
        }, f, indent=2)

    print(f"\nSaved {n} observations to {result_dir}/")
    del adapter
    return observations


if __name__ == "__main__":
    run_pair_transfer()
