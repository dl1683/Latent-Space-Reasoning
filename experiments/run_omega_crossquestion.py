"""Omega Cross-Question Consequence Test (Experiment #58).

THE omega transfer test: does TV measured under Q1 predict argmax flips
under a DIFFERENT question Q2?

Design (from Fable R4):
- Same cells, same continuations as #49
- Q1 = print(x)  # Output:    (TV already measured in #49 data)
- Q2 = {assert x ==, x + 1 =, if x > 3: print("big")}
- Restrict to eligible pairs (baseline TV <= tau under Q1)
- Report LOPO threshold, AUC, and single-surface margin baseline under Q2

If TV_Q1 predicts flips_Q2 better than single-surface margin_Q2,
omega carries cross-question information — the actual transfer claim.

Fable R4 priority #1. ~500 readouts.
"""

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
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
    "nonvar_42": "_ = 42\n",
    "neutral_pass": "pass\n",
    "type_check": "isinstance({var}, int)\n",
    "safety_comment": "# Do not modify {var}.\n",
    "deprecation": "# DEPRECATED: {var} will be removed.\n",
}

QUERIES = {
    "q1_print": "f()\nprint({var})  # Output: ",
    "q2_assert": "f()\nassert {var} == ",
    "q2_arithmetic": "f()\nresult = {var} + 1\nprint(result)  # Output: ",
    "q2_conditional": 'f()\nif {var} > 3:\n    print("big")\nelse:\n    print("small")\n# Output: ',
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]
TAU = 0.05


def run_crossquestion():
    print("=== Cross-Question Consequence Test ===")
    adapter = ModelAdapter({"model_id": MODEL_CFG["model_id"]})
    adapter.tokenizer = adapter.tok
    template = MODEL_CFG["templates"]["depth4_single"]

    observations = []
    t0 = time.time()
    total = len(CELLS) * len(SURFACES) * len(CONTINUATIONS) * len(QUERIES)
    done = 0

    for depth, var, val in CELLS:
        prefix_code = template.format(var=var, outer_val=val)

        for surf_name, surf_text in SURFACES.items():
            surface = surf_text.format(var=var)

            for cont_name, cont_text in CONTINUATIONS.items():
                continuation = cont_text.format(var=var)

                for query_name, query_template in QUERIES.items():
                    query = query_template.format(var=var)
                    full_text = prefix_code + surface + continuation + query

                    dist_11 = adapter.get_dist(full_text)
                    dist_11 = [float(x) for x in dist_11]

                    argmax_idx = int(np.argmax(dist_11))
                    argmax_label = DIGIT_LABELS[argmax_idx]
                    argmax_prob = dist_11[argmax_idx]

                    correct = str(val) if cont_name != "heavy" else str(val + 1)

                    obs = {
                        "depth": depth,
                        "var": var,
                        "val": val,
                        "surface": surf_name,
                        "continuation": cont_name,
                        "query": query_name,
                        "dist": dist_11,
                        "argmax_label": argmax_label,
                        "argmax_prob": argmax_prob,
                        "correct_digit": correct,
                        "is_correct": argmax_label == correct,
                    }
                    observations.append(obs)
                    done += 1

                    if done % 50 == 0:
                        elapsed = time.time() - t0
                        rate = done / elapsed
                        print(f"  {done}/{total} ({rate:.1f}/s, "
                              f"ETA {(total-done)/rate:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nDone: {len(observations)} readouts in {elapsed:.1f}s")

    out_dir = Path("experiments/results/omega_crossquestion")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "observations.json", "w") as f:
        json.dump(observations, f, indent=2)

    summary = analyze_crossquestion(observations)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}")
    return summary


def analyze_crossquestion(observations):
    """The core analysis: does TV under Q1 predict flips under Q2?"""

    idx = {}
    for r in observations:
        key = (r['var'], r['val'], r['surface'], r['continuation'], r['query'])
        idx[key] = r

    results = {"per_query_transfer": {}, "pooled": {}}

    for q2_name in ["q2_assert", "q2_arithmetic", "q2_conditional"]:
        transfer_rows = []

        for s1, s2, pair_name in EQUIV_PAIRS:
            for depth, var, val in CELLS:
                q1_key_s1 = (var, val, s1, "empty", "q1_print")
                q1_key_s2 = (var, val, s2, "empty", "q1_print")

                if q1_key_s1 not in idx or q1_key_s2 not in idx:
                    continue

                baseline_tv = tv(idx[q1_key_s1]['dist'], idx[q1_key_s2]['dist'])

                if baseline_tv > TAU:
                    continue

                for cont_name in CONTINUATIONS:
                    if cont_name == "empty":
                        continue

                    q1_s1 = (var, val, s1, cont_name, "q1_print")
                    q1_s2 = (var, val, s2, cont_name, "q1_print")
                    q2_s1 = (var, val, s1, cont_name, q2_name)
                    q2_s2 = (var, val, s2, cont_name, q2_name)

                    if not all(k in idx for k in [q1_s1, q1_s2, q2_s1, q2_s2]):
                        continue

                    tv_q1 = tv(idx[q1_s1]['dist'], idx[q1_s2]['dist'])

                    flip_q2 = idx[q2_s1]['argmax_label'] != idx[q2_s2]['argmax_label']

                    margin_s1 = sorted(idx[q2_s1]['dist'], reverse=True)
                    margin_s2 = sorted(idx[q2_s2]['dist'], reverse=True)
                    min_margin = min(
                        margin_s1[0] - margin_s1[1] if len(margin_s1) > 1 else 1.0,
                        margin_s2[0] - margin_s2[1] if len(margin_s2) > 1 else 1.0,
                    )

                    tv_q2 = tv(idx[q2_s1]['dist'], idx[q2_s2]['dist'])

                    transfer_rows.append({
                        "pair": pair_name,
                        "cell": f"{var}_{val}",
                        "continuation": cont_name,
                        "tv_q1": float(tv_q1),
                        "tv_q2": float(tv_q2),
                        "flip_q2": bool(flip_q2),
                        "min_margin_q2": float(min_margin),
                        "baseline_tv_q1": float(baseline_tv),
                    })

        n_flips = sum(1 for r in transfer_rows if r['flip_q2'])

        auc_tv_q1 = compute_auc(
            [r['tv_q1'] for r in transfer_rows],
            [r['flip_q2'] for r in transfer_rows],
        )

        auc_margin = compute_auc(
            [-r['min_margin_q2'] for r in transfer_rows],
            [r['flip_q2'] for r in transfer_rows],
        )

        auc_tv_q2 = compute_auc(
            [r['tv_q2'] for r in transfer_rows],
            [r['flip_q2'] for r in transfer_rows],
        )

        results["per_query_transfer"][q2_name] = {
            "n_rows": len(transfer_rows),
            "n_flips": n_flips,
            "auc_tv_q1": auc_tv_q1,
            "auc_margin_q2": auc_margin,
            "auc_tv_q2": auc_tv_q2,
            "rows": transfer_rows,
        }

        print(f"\n  {q2_name}: {len(transfer_rows)} rows, {n_flips} flips")
        print(f"    AUC(TV_Q1 -> flip_Q2) = {auc_tv_q1:.3f}")
        print(f"    AUC(margin_Q2 -> flip_Q2) = {auc_margin:.3f}")
        print(f"    AUC(TV_Q2 -> flip_Q2) = {auc_tv_q2:.3f}")

        if auc_tv_q1 > auc_margin:
            print(f"    *** TV_Q1 BEATS margin_Q2 by {auc_tv_q1 - auc_margin:.3f} ***")
        else:
            print(f"    margin_Q2 wins by {auc_margin - auc_tv_q1:.3f}")

    all_rows = []
    for q2_name, q2_data in results["per_query_transfer"].items():
        all_rows.extend(q2_data["rows"])

    pooled_auc_q1 = compute_auc(
        [r['tv_q1'] for r in all_rows],
        [r['flip_q2'] for r in all_rows],
    )
    pooled_auc_margin = compute_auc(
        [-r['min_margin_q2'] for r in all_rows],
        [r['flip_q2'] for r in all_rows],
    )
    pooled_n_flips = sum(1 for r in all_rows if r['flip_q2'])

    results["pooled"] = {
        "n_rows": len(all_rows),
        "n_flips": pooled_n_flips,
        "auc_tv_q1": pooled_auc_q1,
        "auc_margin_q2": pooled_auc_margin,
    }

    print(f"\n  POOLED: {len(all_rows)} rows, {pooled_n_flips} flips")
    print(f"    AUC(TV_Q1 -> flip_Q2) = {pooled_auc_q1:.3f}")
    print(f"    AUC(margin_Q2 -> flip_Q2) = {pooled_auc_margin:.3f}")

    verdict = "TRANSFER" if pooled_auc_q1 > pooled_auc_margin else "NO_TRANSFER"
    results["verdict"] = verdict
    results["n_readouts"] = len([r for r in observations])

    print(f"\n  VERDICT: {verdict}")
    return results


def compute_auc(scores, labels):
    if not any(labels) or all(labels):
        return float('nan')
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    concordant = sum(1 for p in pos for n in neg if p > n)
    tied = sum(1 for p in pos for n in neg if p == n)
    return (concordant + 0.5 * tied) / (len(pos) * len(neg))


if __name__ == "__main__":
    run_crossquestion()
