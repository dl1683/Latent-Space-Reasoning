"""Omega Cross-Question Consequence Test (Experiment #58).

The REAL transfer test: measure TV under Q1, predict flips under Q2.
Fable R4 priority #1. Previous consequence tests (#49) used the same
readout for both TV and flip — structural coupling. This experiment
measures TV under one question and tests whether it predicts argmax
disagreement under a DIFFERENT question.

Eligible pairs only (baseline TV <= tau).
Reports AUC, LOPO threshold, and single-surface margin baseline.
"""

import gc
import json
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

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
    "safety_comment": "# Do not modify {var}.\n",
    "deprecation": "# Deprecated: {var} is no longer used.\n",
}

QUESTIONS = {
    "print_output": "f()\nprint({var})  # Output: ",
    "assert_eq": "f()\nassert {var} == ",
    "addition": "f()\nresult = {var} + 1\nprint(result)  # Output: ",
    "comparison": "f()\nif {var} > 3:\n    print(\"big\")\nelse:\n    print(\"small\")\n# Output: ",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]
TAU = 0.05


def run_cross_question():
    print("=== Cross-Question Consequence Test ===")
    n_readouts_est = len(CELLS) * len(SURFACES) * len(CONTINUATIONS) * len(QUESTIONS)
    print(f"Target: ~{n_readouts_est} readouts")

    ckpt_dir = Path("experiments/results/omega_cross_question")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "checkpoint.json"

    # Resume from checkpoint if available
    observations = []
    done_cells = set()
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        all_obs = ckpt["obs"]
        # Determine which cells are fully done (144 readouts each)
        cell_counts = defaultdict(int)
        for obs in all_obs:
            cell_counts[(obs["depth"], obs["var"], obs["val"])] += 1
        expected_per_cell = len(SURFACES) * len(CONTINUATIONS) * len(QUESTIONS)
        for cell_key, count in cell_counts.items():
            if count >= expected_per_cell:
                done_cells.add(cell_key)
        # Keep only observations from fully completed cells
        observations = [obs for obs in all_obs
                        if (obs["depth"], obs["var"], obs["val"])
                        in done_cells]
        print(f"  Resumed: {len(observations)} observations from "
              f"{len(done_cells)}/{len(CELLS)} complete cells "
              f"(dropped {len(all_obs) - len(observations)} partial)",
              flush=True)

    remaining = [(d, v, val) for d, v, val in CELLS
                 if (d, v, val) not in done_cells]
    if not remaining:
        print("  All cells already done, skipping to analysis")
    else:
        adapter = ModelAdapter(MODEL_CFG)
        gc.collect()
        print("  Model loaded (float32)", flush=True)

        t0 = time.time()
        n_readouts = len(observations)

        for ci, (depth, var, val) in enumerate(remaining):
            print(f"  Cell {ci+1}/{len(remaining)}: "
                  f"{var}={val}", flush=True)
            tmpl = MODEL_CFG["templates"]["depth4_single"]
            prefix_text = build_prefix(tmpl, var=var, outer_val=val)
            prefix_state = adapter.get_state_after_prefix(prefix_text)

            for q_name, q_tmpl in QUESTIONS.items():
                query = q_tmpl.replace("{var}", var)

                for surf_name, surf_tmpl in SURFACES.items():
                    surface_text = surf_tmpl.replace("{var}", var)

                    for cont_name, cont_tmpl in CONTINUATIONS.items():
                        cont_text = cont_tmpl.replace("{var}", var)
                        full_suffix = surface_text + cont_text + query
                        dist = adapter.get_dist_from_state(
                            prefix_state, full_suffix)
                        n_readouts += 1

                        argmax_idx = int(np.argmax(dist))
                        argmax_label = DIGIT_LABELS[argmax_idx]
                        correct_digit = str(val)
                        top2 = np.sort(dist)[-2:]
                        margin = float(top2[1] - top2[0])

                        observations.append({
                            "depth": depth, "var": var, "val": val,
                            "question": q_name,
                            "surface": surf_name,
                            "continuation": cont_name,
                            "dist": dist.tolist(),
                            "argmax_label": argmax_label,
                            "argmax_prob": float(dist[argmax_idx]),
                            "correct_digit": correct_digit,
                            "is_correct": argmax_label == correct_digit,
                            "margin": margin,
                        })

                        if n_readouts % 50 == 0:
                            gc.collect()

            # Checkpoint after each cell
            del prefix_state
            gc.collect()
            with open(ckpt_path, "w") as cf:
                json.dump({"n": n_readouts, "obs": observations}, cf)
            elapsed = time.time() - t0
            print(f"  Checkpoint: {n_readouts} readouts, {elapsed:.0f}s",
                  flush=True)

        # Free model after all readouts
        del adapter
        gc.collect()
        elapsed = time.time() - t0
        print(f"Done: {n_readouts} readouts in {elapsed:.1f}s")

    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["question"], obs["surface"], obs["continuation"])
        obs_idx[key] = obs

    # For each question pair (Q_tv, Q_flip):
    # 1. Compute TV under Q_tv for each (pair, cell, continuation)
    # 2. Check eligibility: baseline TV under Q_tv <= TAU
    # 3. Determine flip under Q_flip
    # 4. Compute AUC, correlation, threshold metrics
    q_names = list(QUESTIONS.keys())
    cross_results = []

    for q_tv in q_names:
        for q_flip in q_names:
            rows = []
            for surf_a, surf_b, pair_name in EQUIV_PAIRS:
                for cont_name in CONTINUATIONS:
                    # Check eligibility: empty continuation baseline under q_tv
                    eligible_cells = []
                    for depth, var, val in CELLS:
                        key_a_base = (depth, var, val, q_tv, surf_a, "empty")
                        key_b_base = (depth, var, val, q_tv, surf_b, "empty")
                        if key_a_base in obs_idx and key_b_base in obs_idx:
                            base_tv = tv(
                                obs_idx[key_a_base]["dist"],
                                obs_idx[key_b_base]["dist"])
                            if base_tv <= TAU:
                                eligible_cells.append((depth, var, val))

                    for depth, var, val in eligible_cells:
                        # TV under Q_tv
                        oa_tv = obs_idx[(depth, var, val, q_tv, surf_a, cont_name)]
                        ob_tv = obs_idx[(depth, var, val, q_tv, surf_b, cont_name)]
                        cell_tv = float(tv(oa_tv["dist"], ob_tv["dist"]))

                        # Flip under Q_flip
                        oa_fl = obs_idx[(depth, var, val, q_flip, surf_a, cont_name)]
                        ob_fl = obs_idx[(depth, var, val, q_flip, surf_b, cont_name)]
                        flip = oa_fl["argmax_label"] != ob_fl["argmax_label"]

                        # Single-surface margins under Q_flip
                        margin_a = oa_fl["margin"]
                        margin_b = ob_fl["margin"]
                        min_margin = min(margin_a, margin_b)

                        rows.append({
                            "pair": pair_name,
                            "continuation": cont_name,
                            "cell": f"{var}_{val}",
                            "tv_q_tv": cell_tv,
                            "flip_q_flip": flip,
                            "margin_a": margin_a,
                            "margin_b": margin_b,
                            "min_margin": min_margin,
                        })

            if not rows:
                continue

            tvs = np.array([r["tv_q_tv"] for r in rows])
            flips = np.array([1 if r["flip_q_flip"] else 0 for r in rows])
            margins = np.array([r["min_margin"] for r in rows])

            n_flips = int(flips.sum())
            n_eligible = len(rows)

            # Correlation
            if flips.std() > 0 and tvs.std() > 0:
                corr = float(np.corrcoef(tvs, flips)[0, 1])
            else:
                corr = 0.0

            # Margin correlation
            if flips.std() > 0 and margins.std() > 0:
                margin_corr = float(np.corrcoef(-margins, flips)[0, 1])
            else:
                margin_corr = 0.0

            # AUC (manual for small n)
            def compute_auc(scores, labels):
                pos = scores[labels == 1]
                neg = scores[labels == 0]
                if len(pos) == 0 or len(neg) == 0:
                    return 0.5
                auc = 0
                for p in pos:
                    auc += np.sum(p > neg) + 0.5 * np.sum(p == neg)
                return float(auc / (len(pos) * len(neg)))

            tv_auc = compute_auc(tvs, flips)
            margin_auc = compute_auc(-margins, flips)

            # Threshold sweep
            thresholds = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
            best_f1 = 0
            best_t = None
            threshold_detail = {}
            for t in thresholds:
                above = flips[tvs >= t]
                below = flips[tvs < t]
                tp = int(above.sum())
                fp = int(len(above) - tp)
                fn = int(below.sum())
                tn = int(len(below) - fn)
                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                threshold_detail[str(t)] = {
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "precision": round(p, 3), "recall": round(r, 3),
                    "f1": round(f1, 3),
                }
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t

            is_transfer = q_tv != q_flip

            cross_results.append({
                "q_tv": q_tv,
                "q_flip": q_flip,
                "is_transfer": is_transfer,
                "n_eligible": n_eligible,
                "n_flips": n_flips,
                "tv_flip_correlation": round(corr, 4),
                "margin_flip_correlation": round(margin_corr, 4),
                "tv_auc": round(tv_auc, 4),
                "margin_auc": round(margin_auc, 4),
                "tv_auc_minus_margin_auc": round(tv_auc - margin_auc, 4),
                "best_threshold": best_t,
                "best_f1": round(best_f1, 3),
                "threshold_detail": threshold_detail,
            })

    # Summarize
    same_q = [r for r in cross_results if not r["is_transfer"]]
    cross_q = [r for r in cross_results if r["is_transfer"]]

    summary = {
        "n_readouts": n_readouts,
        "tau": TAU,
        "questions": list(QUESTIONS.keys()),
        "same_question_results": same_q,
        "cross_question_results": cross_q,
        "headline": {
            "same_q_mean_tv_auc": round(float(np.mean(
                [r["tv_auc"] for r in same_q])), 4) if same_q else None,
            "same_q_mean_margin_auc": round(float(np.mean(
                [r["margin_auc"] for r in same_q])), 4) if same_q else None,
            "cross_q_mean_tv_auc": round(float(np.mean(
                [r["tv_auc"] for r in cross_q])), 4) if cross_q else None,
            "cross_q_mean_margin_auc": round(float(np.mean(
                [r["margin_auc"] for r in cross_q])), 4) if cross_q else None,
        },
    }

    result_dir = Path("experiments/results/omega_cross_question")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    # Print
    print(f"\n=== CROSS-QUESTION RESULTS ===")
    print(f"{'Q_tv':20s} {'Q_flip':20s} {'Xfer':>5s} {'N':>4s} "
          f"{'Flips':>5s} {'TV_AUC':>7s} {'M_AUC':>7s} {'Delta':>7s} "
          f"{'r':>6s} {'BestF1':>7s}")
    for r in sorted(cross_results,
                    key=lambda x: (-x["is_transfer"], -x["tv_auc"])):
        print(f"{r['q_tv']:20s} {r['q_flip']:20s} "
              f"{'YES' if r['is_transfer'] else 'no':>5s} "
              f"{r['n_eligible']:4d} {r['n_flips']:5d} "
              f"{r['tv_auc']:7.3f} {r['margin_auc']:7.3f} "
              f"{r['tv_auc_minus_margin_auc']:+7.3f} "
              f"{r['tv_flip_correlation']:6.3f} {r['best_f1']:7.3f}")

    h = summary["headline"]
    print(f"\n--- Headline ---")
    print(f"Same-question:  TV AUC={h['same_q_mean_tv_auc']}, "
          f"Margin AUC={h['same_q_mean_margin_auc']}")
    print(f"Cross-question: TV AUC={h['cross_q_mean_tv_auc']}, "
          f"Margin AUC={h['cross_q_mean_margin_auc']}")
    if h['cross_q_mean_tv_auc'] and h['cross_q_mean_margin_auc']:
        delta = h['cross_q_mean_tv_auc'] - h['cross_q_mean_margin_auc']
        print(f"Cross-Q TV advantage: {delta:+.4f}")
        if delta > 0.05:
            print("=> TV carries cross-question transfer information beyond margin")
        elif delta > 0:
            print("=> TV marginally better than margin on cross-question")
        else:
            print("=> Margin matches or beats TV on cross-question — omega claim weakened")


if __name__ == "__main__":
    import traceback
    try:
        run_cross_question()
    except Exception as e:
        print(f"\n!!! EXCEPTION: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
