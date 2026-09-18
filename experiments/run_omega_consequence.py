"""Omega Consequence Prediction (Experiment #49).

Does omega predict actual model failures? Tests whether cells with high
omega (TV between surface-pair readouts) also show different argmax
predictions — bridging distribution distance to correctness.
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

SURFACES = {
    "assert_novar_2": "# State unchanged.\n",
    "assert_var_2": "# {var} stays the same.\n",
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
    "empty": "",
    "identity": "{var} = {var}\n",
    "nonvar_42": "_ = 42\n",
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
    "neutral_pass": "pass\n",
    "type_check": "type({var})\n",
    "safety_comment": "# Do not modify {var}.\n",
    "deprecation": "# Deprecated: {var} is no longer used.\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def run_consequence_experiment():
    print("=== Omega Consequence Prediction ===")
    n_target = len(CELLS) * len(SURFACES) * len(CONTINUATIONS)
    print(f"Target: {n_target} readouts ({len(CELLS)} cells × "
          f"{len(SURFACES)} surfaces × {len(CONTINUATIONS)} continuations)")

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

    result_dir = Path("experiments/results/omega_consequence")
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

    # Per continuation, per pair: compute omega and prediction flips
    pair_results = {}
    all_omega_flip_pairs = []

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
                cell_tv = tv(oa["dist"], ob["dist"])
                tvs.append(float(cell_tv))

                flip = oa["argmax_label"] != ob["argmax_label"]
                flips.append(flip)
                correct_a_list.append(oa["is_correct"])
                correct_b_list.append(ob["is_correct"])

                all_omega_flip_pairs.append({
                    "pair": pair_name,
                    "continuation": cont_name,
                    "cell": f"{var}_{val}",
                    "tv": float(cell_tv),
                    "flip": flip,
                    "correct_a": oa["is_correct"],
                    "correct_b": ob["is_correct"],
                    "argmax_a": oa["argmax_label"],
                    "argmax_b": ob["argmax_label"],
                })

            omega = max(tvs) if tvs else 0
            flip_rate = sum(flips) / len(flips) if flips else 0
            acc_a = sum(correct_a_list) / len(correct_a_list)
            acc_b = sum(correct_b_list) / len(correct_b_list)

            cont_results[cont_name] = {
                "omega": omega,
                "omega_mean": float(np.mean(tvs)),
                "flip_rate": flip_rate,
                "n_flips": sum(flips),
                "n_cells": len(flips),
                "accuracy_surface_a": acc_a,
                "accuracy_surface_b": acc_b,
                "tvs": tvs,
            }

        pair_results[pair_name] = cont_results

    summary["pair_results"] = pair_results

    # Threshold analysis: at what omega does flip become common?
    thresholds = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    threshold_analysis = {}
    for thresh in thresholds:
        above = [p for p in all_omega_flip_pairs if p["tv"] >= thresh]
        below = [p for p in all_omega_flip_pairs if p["tv"] < thresh]
        tp = sum(1 for p in above if p["flip"])
        fp = sum(1 for p in above if not p["flip"])
        fn = sum(1 for p in below if p["flip"])
        tn = sum(1 for p in below if not p["flip"])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0)
        threshold_analysis[str(thresh)] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "n_above": len(above),
            "n_below": len(below),
        }

    summary["threshold_analysis"] = threshold_analysis

    # By continuation type: flip rate and mean omega
    cont_summary = {}
    for cont_name in CONTINUATIONS:
        cells = [p for p in all_omega_flip_pairs
                 if p["continuation"] == cont_name]
        mean_tv = float(np.mean([c["tv"] for c in cells]))
        flip_rate = sum(1 for c in cells if c["flip"]) / len(cells)
        cont_summary[cont_name] = {
            "mean_tv": round(mean_tv, 4),
            "flip_rate": round(flip_rate, 3),
            "n_flips": sum(1 for c in cells if c["flip"]),
            "n_cells": len(cells),
        }
    summary["continuation_flip_rates"] = cont_summary

    # Overall accuracy
    total_correct = sum(1 for o in observations if o["is_correct"])
    summary["overall_accuracy"] = round(total_correct / len(observations), 3)
    summary["overall_correct"] = total_correct
    summary["overall_total"] = len(observations)

    # Point-biserial correlation between TV and flip
    tvs_arr = np.array([p["tv"] for p in all_omega_flip_pairs])
    flips_arr = np.array([1 if p["flip"] else 0
                          for p in all_omega_flip_pairs])
    if flips_arr.std() > 0 and tvs_arr.std() > 0:
        corr = float(np.corrcoef(tvs_arr, flips_arr)[0, 1])
    else:
        corr = 0.0
    summary["tv_flip_correlation"] = round(corr, 3)

    # Best F1 threshold
    best_f1 = 0
    best_thresh = None
    for t, v in threshold_analysis.items():
        if v["f1"] > best_f1:
            best_f1 = v["f1"]
            best_thresh = t
    summary["best_threshold"] = best_thresh
    summary["best_f1"] = best_f1

    # Per-cell detail
    summary["all_cells"] = all_omega_flip_pairs

    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print results
    print("\n=== Results ===")
    print(f"Overall accuracy: {summary['overall_accuracy']:.1%} "
          f"({summary['overall_correct']}/{summary['overall_total']})")
    print(f"TV-flip correlation: {corr:.3f}")
    print(f"Best threshold: {best_thresh} (F1={best_f1:.3f})")

    print("\n--- Continuation Flip Rates ---")
    for c, v in sorted(cont_summary.items(),
                       key=lambda x: -x[1]["flip_rate"]):
        print(f"  {c:20s}: flip={v['flip_rate']:.1%} "
              f"({v['n_flips']}/{v['n_cells']}), "
              f"mean_tv={v['mean_tv']:.4f}")

    print("\n--- Threshold Analysis ---")
    for t in thresholds:
        v = threshold_analysis[str(t)]
        print(f"  tau={t:.2f}: P={v['precision']:.2f} R={v['recall']:.2f} "
              f"F1={v['f1']:.2f} (above={v['n_above']})")

    print("\n--- Per-Pair Results ---")
    for pair_name in pair_results:
        print(f"\n  {pair_name}:")
        for c, v in sorted(pair_results[pair_name].items(),
                           key=lambda x: -x[1]["omega"]):
            print(f"    {c:20s}: omega={v['omega']:.4f}, "
                  f"flips={v['n_flips']}/{v['n_cells']}, "
                  f"acc_a={v['accuracy_surface_a']:.1%}, "
                  f"acc_b={v['accuracy_surface_b']:.1%}")


if __name__ == "__main__":
    run_consequence_experiment()
