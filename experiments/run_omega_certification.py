"""Omega Certification Budget Test (Experiment #63).

Tests omega's value as a CERTIFICATION tool: given a budget of N
questions, how well does omega estimated from those N questions predict
behavior under a held-out question?

This is distinct from same-question prediction (where margin wins).
The question is: if I measure omega under {Q1, Q2, Q3}, does that
predict behavior under Q4 that I HAVEN'T run yet?

Margin can't do this because margin is per-question. You'd need to run
Q4 to compute margin under Q4. Omega gives you a bound on ALL questions
in the family, including ones you haven't tested.

Design:
- 4 questions: print, assert, arithmetic, conditional
- For each leave-one-out split: train omega on 3 questions, test on 4th
- Compare: omega_train predicts test-question flips vs margin_test (oracle)
- Budget analysis: how does prediction quality scale with N=1,2,3?

~1152 readouts (same data as exp #58, reanalyzed differently).
Uses exp #58 data if available, otherwise collects fresh.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from itertools import combinations

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
    "q_print": "f()\nprint({var})  # Output: ",
    "q_assert": "f()\nassert {var} == ",
    "q_arithmetic": "f()\nresult = {var} + 1\nprint(result)  # Output: ",
    "q_conditional": 'f()\nif {var} > 3:\n    print("big")\nelse:\n    print("small")\n# Output: ',
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]
TAU = 0.05


def load_or_collect():
    crossq_path = Path("experiments/results/omega_crossquestion/observations.json")
    if crossq_path.exists():
        print("Loading existing cross-question data...")
        with open(crossq_path) as f:
            return json.load(f)

    print("Collecting fresh data...")
    adapter = ModelAdapter({"model_id": MODEL_CFG["model_id"]})
    template = MODEL_CFG["templates"]["depth4_single"]
    observations = []
    t0 = time.time()
    total = len(CELLS) * len(SURFACES) * len(CONTINUATIONS) * len(QUERIES)
    done = 0

    for depth, var, val in CELLS:
        prefix_code = template.format(var=var, outer_val=val)
        prefix_state = adapter.get_state_after_prefix(prefix_code)

        for surf_name, surf_text in SURFACES.items():
            surface = surf_text.format(var=var)
            for cont_name, cont_text in CONTINUATIONS.items():
                continuation = cont_text.format(var=var)
                for query_name, query_template in QUERIES.items():
                    query = query_template.format(var=var)
                    suffix = surface + continuation + query
                    dist = adapter.get_dist_from_state(prefix_state, suffix)
                    done += 1

                    argmax_idx = int(np.argmax(dist))
                    observations.append({
                        "depth": depth, "var": var, "val": val,
                        "surface": surf_name, "continuation": cont_name,
                        "query": query_name,
                        "dist": dist.tolist(),
                        "argmax_label": DIGIT_LABELS[argmax_idx],
                        "argmax_prob": float(dist[argmax_idx]),
                        "correct_digit": str(val),
                    })

                    if done % 100 == 0:
                        elapsed = time.time() - t0
                        rate = done / elapsed
                        print(f"  {done}/{total} ({rate:.1f}/s)")

    print(f"Collected {len(observations)} readouts in {time.time()-t0:.1f}s")
    return observations


def compute_auc(scores, labels):
    if not any(labels) or all(labels):
        return float('nan')
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    concordant = sum(1 for p in pos for n in neg if p > n)
    tied = sum(1 for p in pos for n in neg if p == n)
    return (concordant + 0.5 * tied) / (len(pos) * len(neg))


def run_certification():
    print("=== Certification Budget Test (Exp #63) ===")

    observations = load_or_collect()

    idx = {}
    for r in observations:
        key = (r['var'], r['val'], r['surface'], r['continuation'], r['query'])
        idx[key] = r

    query_names = list(QUERIES.keys())
    results = {"leave_one_out": {}, "budget_curves": {}}

    for test_q in query_names:
        train_qs = [q for q in query_names if q != test_q]
        transfer_rows = []

        for s1, s2, pair_name in EQUIV_PAIRS:
            for depth, var, val in CELLS:
                base_s1 = (var, val, s1, "empty", train_qs[0])
                base_s2 = (var, val, s2, "empty", train_qs[0])
                if base_s1 not in idx or base_s2 not in idx:
                    continue
                baseline = tv(idx[base_s1]['dist'], idx[base_s2]['dist'])
                if baseline > TAU:
                    continue

                for cont_name in CONTINUATIONS:
                    if cont_name == "empty":
                        continue

                    train_tvs = []
                    for tq in train_qs:
                        k1 = (var, val, s1, cont_name, tq)
                        k2 = (var, val, s2, cont_name, tq)
                        if k1 in idx and k2 in idx:
                            train_tvs.append(
                                float(tv(idx[k1]['dist'], idx[k2]['dist'])))

                    if not train_tvs:
                        continue

                    omega_train = max(train_tvs)
                    mean_tv_train = float(np.mean(train_tvs))

                    test_k1 = (var, val, s1, cont_name, test_q)
                    test_k2 = (var, val, s2, cont_name, test_q)
                    if test_k1 not in idx or test_k2 not in idx:
                        continue

                    flip_test = (idx[test_k1]['argmax_label'] !=
                                 idx[test_k2]['argmax_label'])

                    margin_s1 = sorted(idx[test_k1]['dist'], reverse=True)
                    margin_s2 = sorted(idx[test_k2]['dist'], reverse=True)
                    min_margin_test = min(
                        margin_s1[0] - margin_s1[1] if len(margin_s1) > 1 else 1,
                        margin_s2[0] - margin_s2[1] if len(margin_s2) > 1 else 1,
                    )

                    tv_test = float(tv(idx[test_k1]['dist'],
                                       idx[test_k2]['dist']))

                    transfer_rows.append({
                        "pair": pair_name,
                        "cell": f"{var}_{val}",
                        "continuation": cont_name,
                        "omega_train": omega_train,
                        "mean_tv_train": mean_tv_train,
                        "tv_test": tv_test,
                        "flip_test": bool(flip_test),
                        "min_margin_test": float(min_margin_test),
                    })

        n_flips = sum(1 for r in transfer_rows if r['flip_test'])

        auc_omega = compute_auc(
            [r['omega_train'] for r in transfer_rows],
            [r['flip_test'] for r in transfer_rows])
        auc_margin = compute_auc(
            [-r['min_margin_test'] for r in transfer_rows],
            [r['flip_test'] for r in transfer_rows])
        auc_tv_test = compute_auc(
            [r['tv_test'] for r in transfer_rows],
            [r['flip_test'] for r in transfer_rows])

        results["leave_one_out"][test_q] = {
            "n_rows": len(transfer_rows),
            "n_flips": n_flips,
            "auc_omega_train": auc_omega,
            "auc_margin_test_oracle": auc_margin,
            "auc_tv_test_oracle": auc_tv_test,
        }

        print(f"\n  Hold-out: {test_q} ({len(transfer_rows)} rows, {n_flips} flips)")
        print(f"    AUC(omega_train)        = {auc_omega:.3f} [TRANSFER]")
        print(f"    AUC(margin_test oracle) = {auc_margin:.3f} [ORACLE]")
        print(f"    AUC(TV_test oracle)     = {auc_tv_test:.3f} [ORACLE]")

        if not np.isnan(auc_omega) and not np.isnan(auc_margin):
            if auc_omega > auc_margin:
                print(f"    *** OMEGA BEATS ORACLE MARGIN by {auc_omega-auc_margin:.3f} ***")

    for budget in [1, 2, 3]:
        budget_aucs = []
        for test_q in query_names:
            avail = [q for q in query_names if q != test_q]
            combo_aucs = []
            for combo in combinations(avail, budget):
                rows = []
                for s1, s2, pair_name in EQUIV_PAIRS:
                    for depth, var, val in CELLS:
                        base_s1 = (var, val, s1, "empty", combo[0])
                        base_s2 = (var, val, s2, "empty", combo[0])
                        if base_s1 not in idx or base_s2 not in idx:
                            continue
                        bl = tv(idx[base_s1]['dist'], idx[base_s2]['dist'])
                        if bl > TAU:
                            continue

                        for cont_name in CONTINUATIONS:
                            if cont_name == "empty":
                                continue
                            tvs_budget = []
                            for tq in combo:
                                k1 = (var, val, s1, cont_name, tq)
                                k2 = (var, val, s2, cont_name, tq)
                                if k1 in idx and k2 in idx:
                                    tvs_budget.append(float(
                                        tv(idx[k1]['dist'], idx[k2]['dist'])))
                            if not tvs_budget:
                                continue

                            test_k1 = (var, val, s1, cont_name, test_q)
                            test_k2 = (var, val, s2, cont_name, test_q)
                            if test_k1 not in idx or test_k2 not in idx:
                                continue
                            flip = (idx[test_k1]['argmax_label'] !=
                                    idx[test_k2]['argmax_label'])
                            rows.append({
                                "omega": max(tvs_budget),
                                "flip": bool(flip),
                            })

                if rows:
                    a = compute_auc(
                        [r['omega'] for r in rows],
                        [r['flip'] for r in rows])
                    if not np.isnan(a):
                        combo_aucs.append(a)

            if combo_aucs:
                budget_aucs.append(float(np.mean(combo_aucs)))

        if budget_aucs:
            results["budget_curves"][budget] = {
                "mean_auc": round(float(np.mean(budget_aucs)), 4),
                "min_auc": round(float(np.min(budget_aucs)), 4),
                "max_auc": round(float(np.max(budget_aucs)), 4),
            }
            print(f"\n  Budget N={budget}: mean AUC = {np.mean(budget_aucs):.3f} "
                  f"[{np.min(budget_aucs):.3f}, {np.max(budget_aucs):.3f}]")

    pooled_omega_aucs = [v["auc_omega_train"] for v in results["leave_one_out"].values()
                         if not np.isnan(v["auc_omega_train"])]
    pooled_margin_aucs = [v["auc_margin_test_oracle"] for v in results["leave_one_out"].values()
                          if not np.isnan(v["auc_margin_test_oracle"])]

    if pooled_omega_aucs and pooled_margin_aucs:
        mean_omega = float(np.mean(pooled_omega_aucs))
        mean_margin = float(np.mean(pooled_margin_aucs))

        if mean_omega > mean_margin:
            verdict = f"OMEGA_CERTIFIES: mean AUC {mean_omega:.3f} > oracle margin {mean_margin:.3f}"
        elif mean_omega > 0.7:
            verdict = f"OMEGA_USEFUL: mean AUC {mean_omega:.3f} (oracle margin {mean_margin:.3f})"
        else:
            verdict = f"OMEGA_WEAK: mean AUC {mean_omega:.3f} < oracle margin {mean_margin:.3f}"
    else:
        verdict = "INSUFFICIENT_FLIPS"

    results["verdict"] = verdict
    results["n_observations"] = len(observations)

    print(f"\nVERDICT: {verdict}")

    out_dir = Path("experiments/results/omega_certification")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_dir}")

    return results


if __name__ == "__main__":
    run_certification()
