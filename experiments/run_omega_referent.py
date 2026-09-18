"""Omega Referent Test (Experiment #50).

Does amplification follow the REFERENT (tracked variable), the TOKEN NAME,
or is it just ANY extra computation context?

Exp #46 showed `_ = 42` (unrelated to tracked variable) amplifies as much
as identity (`x = x`). This experiment tests whether the NAME of the variable
in the continuation matters.

Continuations tested:
- tracked_identity:   {var} = {var}       (reassigns tracked var to itself)
- tracked_value:      {var} = {outer_val} (reassigns tracked var to its value)
- other_var_identity: w = w               (reassigns DIFFERENT var to itself)
- other_var_assign:   w = 42              (assigns unrelated var a value)
- nonvar_42:          _ = 42              (no variable at all)
- query_func:         f()                 (calls the function that query calls)
- empty:              <nothing>           (baseline)
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

# Continuations to test — grouped by referent type
CONTINUATIONS = {
    "tracked_identity":   "{var} = {var}\n",          # tracked var -> itself
    "tracked_value":      "{var} = {outer_val}\n",    # tracked var -> its value
    "other_var_identity": "w = w\n",                  # different var -> itself
    "other_var_assign":   "w = 42\n",                 # different var -> literal
    "nonvar_42":          "_ = 42\n",                 # no real variable
    "query_func":         "f()\n",                    # calls query function
    "empty":              "",                         # baseline (no continuation)
}

# Surface pairs (standard set)
SURFACES = {
    "assert_var_2":   "# {var} stays the same.\n",
    "assert_novar_2": "# State unchanged.\n",
    "boundary_1":     "\n",
    "boundary_2":     "pass\n",
    "minimal_ok":     "# OK.\n",
    "minimal_good":   "# Good.\n",
}

EQUIV_PAIRS = [
    ("assert_var_2", "assert_novar_2", "strong_assert_2"),
    ("boundary_1", "boundary_2", "boundary_pair"),
    ("minimal_ok", "minimal_good", "minimal_ok_good"),
]

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def run_referent_experiment():
    print("=== Omega Referent Test (Exp #50) ===")
    n_target = len(CELLS) * len(SURFACES) * len(CONTINUATIONS)
    print(f"Target: {n_target} readouts ({len(CELLS)} cells x "
          f"{len(SURFACES)} surfaces x {len(CONTINUATIONS)} continuations)")

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
                cont_text = (cont_tmpl
                             .replace("{var}", var)
                             .replace("{outer_val}", str(val)))
                full_suffix = surface_text + cont_text + query
                dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                n_readouts += 1

                argmax_idx = int(np.argmax(dist))
                argmax_label = DIGIT_LABELS[argmax_idx]
                argmax_prob = float(dist[argmax_idx])
                correct_digit = str(val)
                is_correct = argmax_label == correct_digit

                observations.append({
                    "depth": depth,
                    "var": var,
                    "val": val,
                    "surface": surf_name,
                    "continuation": cont_name,
                    "dist": dist.tolist(),
                    "argmax_label": argmax_label,
                    "argmax_prob": argmax_prob,
                    "correct_digit": correct_digit,
                    "is_correct": is_correct,
                })

                if n_readouts % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  {n_readouts} readouts, {elapsed:.0f}s",
                          flush=True)

    elapsed = time.time() - t0
    print(f"Done: {n_readouts} readouts in {elapsed:.1f}s")

    result_dir = Path("experiments/results/omega_referent")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    analyze(observations, result_dir)


def analyze(observations, result_dir):
    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = obs

    summary = {"n_readouts": len(observations)}

    # Per continuation, per pair: compute omega (max TV across cells)
    pair_results = {}
    all_omega_data = []

    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        cont_results = {}
        for cont_name in CONTINUATIONS:
            tvs = []
            flips = []
            correct_a_list = []
            correct_b_list = []
            for depth, var, val in CELLS:
                key_a = (depth, var, val, surf_a, cont_name)
                key_b = (depth, var, val, surf_b, cont_name)
                oa = obs_idx[key_a]
                ob = obs_idx[key_b]
                cell_tv = tv(np.array(oa["dist"]), np.array(ob["dist"]))
                tvs.append(float(cell_tv))

                flip = oa["argmax_label"] != ob["argmax_label"]
                flips.append(flip)
                correct_a_list.append(oa["is_correct"])
                correct_b_list.append(ob["is_correct"])

                all_omega_data.append({
                    "pair": pair_name,
                    "continuation": cont_name,
                    "cell": f"{var}_{val}",
                    "tv": float(cell_tv),
                    "flip": flip,
                })

            omega = max(tvs) if tvs else 0
            mean_tv = float(np.mean(tvs))
            flip_rate = sum(flips) / len(flips) if flips else 0

            cont_results[cont_name] = {
                "omega": omega,
                "omega_mean": mean_tv,
                "flip_rate": flip_rate,
                "n_flips": sum(flips),
                "n_cells": len(flips),
                "accuracy_surface_a": sum(correct_a_list) / len(correct_a_list),
                "accuracy_surface_b": sum(correct_b_list) / len(correct_b_list),
                "tvs": tvs,
            }

        pair_results[pair_name] = cont_results

    summary["pair_results"] = pair_results

    # Continuation category analysis: tracked vs other vs none
    categories = {
        "tracked_var": ["tracked_identity", "tracked_value"],
        "other_var":   ["other_var_identity", "other_var_assign"],
        "no_var":      ["nonvar_42", "query_func"],
        "baseline":    ["empty"],
    }

    category_summary = {}
    for cat_name, cont_names in categories.items():
        omegas = []
        mean_tvs = []
        for pair_name in pair_results:
            for cont_name in cont_names:
                if cont_name in pair_results[pair_name]:
                    r = pair_results[pair_name][cont_name]
                    omegas.append(r["omega"])
                    mean_tvs.append(r["omega_mean"])
        category_summary[cat_name] = {
            "mean_omega": float(np.mean(omegas)) if omegas else 0,
            "max_omega":  float(np.max(omegas)) if omegas else 0,
            "min_omega":  float(np.min(omegas)) if omegas else 0,
            "std_omega":  float(np.std(omegas)) if omegas else 0,
            "mean_tv_mean": float(np.mean(mean_tvs)) if mean_tvs else 0,
            "n": len(omegas),
        }

    summary["category_summary"] = category_summary

    # Grand comparison table: each continuation, averaged across pairs
    grand_table = {}
    for cont_name in CONTINUATIONS:
        omegas = []
        mean_tvs = []
        for pair_name in pair_results:
            r = pair_results[pair_name][cont_name]
            omegas.append(r["omega"])
            mean_tvs.append(r["omega_mean"])
        grand_table[cont_name] = {
            "mean_omega": float(np.mean(omegas)),
            "max_omega":  float(np.max(omegas)),
            "min_omega":  float(np.min(omegas)),
            "mean_tv":    float(np.mean(mean_tvs)),
        }
    summary["grand_table"] = grand_table

    # Overall accuracy
    total_correct = sum(1 for o in observations if o["is_correct"])
    summary["overall_accuracy"] = round(total_correct / len(observations), 3)
    summary["overall_correct"] = total_correct
    summary["overall_total"] = len(observations)

    # Gradient test: tracked > other > none?
    tracked_omegas = []
    other_omegas = []
    none_omegas = []
    baseline_omegas = []
    for pair_name in pair_results:
        for cn in ["tracked_identity", "tracked_value"]:
            tracked_omegas.append(pair_results[pair_name][cn]["omega"])
        for cn in ["other_var_identity", "other_var_assign"]:
            other_omegas.append(pair_results[pair_name][cn]["omega"])
        for cn in ["nonvar_42", "query_func"]:
            none_omegas.append(pair_results[pair_name][cn]["omega"])
        baseline_omegas.append(pair_results[pair_name]["empty"]["omega"])

    gradient_test = {
        "tracked_mean": float(np.mean(tracked_omegas)),
        "other_mean":   float(np.mean(other_omegas)),
        "none_mean":    float(np.mean(none_omegas)),
        "baseline_mean": float(np.mean(baseline_omegas)),
        "tracked_gt_other": float(np.mean(tracked_omegas)) > float(np.mean(other_omegas)),
        "other_gt_none":    float(np.mean(other_omegas)) > float(np.mean(none_omegas)),
        "none_gt_baseline": float(np.mean(none_omegas)) > float(np.mean(baseline_omegas)),
        "gradient_holds": (
            float(np.mean(tracked_omegas)) > float(np.mean(other_omegas)) >
            float(np.mean(none_omegas)) > float(np.mean(baseline_omegas))
        ),
    }
    summary["gradient_test"] = gradient_test

    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print results
    print("\n=== Results ===")
    print(f"Overall accuracy: {summary['overall_accuracy']:.1%} "
          f"({summary['overall_correct']}/{summary['overall_total']})")

    print("\n--- Grand Table: Continuation x Omega ---")
    print(f"  {'Continuation':25s} {'mean_omega':>12s} {'max_omega':>12s} "
          f"{'min_omega':>12s} {'mean_tv':>12s}")
    for c in sorted(grand_table.keys(),
                    key=lambda k: -grand_table[k]["mean_omega"]):
        v = grand_table[c]
        print(f"  {c:25s} {v['mean_omega']:12.4f} {v['max_omega']:12.4f} "
              f"{v['min_omega']:12.4f} {v['mean_tv']:12.4f}")

    print("\n--- Category Summary ---")
    print(f"  {'Category':15s} {'mean_omega':>12s} {'max_omega':>12s} "
          f"{'std_omega':>12s} {'mean_tv':>12s}")
    for cat in ["tracked_var", "other_var", "no_var", "baseline"]:
        v = category_summary[cat]
        print(f"  {cat:15s} {v['mean_omega']:12.4f} {v['max_omega']:12.4f} "
              f"{v['std_omega']:12.4f} {v['mean_tv_mean']:12.4f}")

    print("\n--- Gradient Test ---")
    g = gradient_test
    print(f"  tracked:  {g['tracked_mean']:.4f}")
    print(f"  other:    {g['other_mean']:.4f}")
    print(f"  none:     {g['none_mean']:.4f}")
    print(f"  baseline: {g['baseline_mean']:.4f}")
    print(f"  tracked > other: {g['tracked_gt_other']}")
    print(f"  other > none:    {g['other_gt_none']}")
    print(f"  none > baseline: {g['none_gt_baseline']}")
    print(f"  Full gradient (tracked > other > none > baseline): "
          f"{g['gradient_holds']}")

    print("\n--- Per-Pair Detail ---")
    for pair_name in pair_results:
        print(f"\n  {pair_name}:")
        for c in sorted(pair_results[pair_name].keys(),
                        key=lambda k: -pair_results[pair_name][k]["omega"]):
            v = pair_results[pair_name][c]
            print(f"    {c:25s}: omega={v['omega']:.4f}, "
                  f"mean_tv={v['omega_mean']:.4f}, "
                  f"flips={v['n_flips']}/{v['n_cells']}, "
                  f"acc_a={v['accuracy_surface_a']:.1%}, "
                  f"acc_b={v['accuracy_surface_b']:.1%}")


if __name__ == "__main__":
    run_referent_experiment()
