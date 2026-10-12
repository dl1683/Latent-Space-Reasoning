# Historical variant only: this analyzer expects query/q1_print rows; canonical Exp #58 is experiments/run_omega_cross_question.py (question/print_output rows), so no behavior-changing redirect.
"""R6-corrected reanalysis of cross-question data (Experiment #58).

Fixes 5 problems R6 identified:
1. Exclude q2_conditional (maps to OTHER in 11-bin, TV≈0 always)
2. Fair margin baseline: use margin under Q_tv (q1_print), not Q_flip
3. Report permutation-p and per-combo exact CIs, not delta>0.05
4. Filter rows where pair already flips under Q_flip with empty continuation
5. PRIMARY QUANTITY: question-invariance (Spearman of TV across questions)
   and certificate rate (TV < margin/2 → no flip)
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent))
from omega import tv


def spearman_rho(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3:
        return float('nan'), 1.0
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def bootstrap_auc_ci(scores, labels, n_boot=2000, alpha=0.05):
    n = len(scores)
    if not any(labels) or all(labels) or n < 5:
        return float('nan'), float('nan'), float('nan')
    aucs = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        s = [scores[i] for i in idx]
        l = [labels[i] for i in idx]
        if not any(l) or all(l):
            continue
        aucs.append(compute_auc(s, l))
    aucs = sorted(aucs)
    lo = aucs[int(alpha/2 * len(aucs))]
    hi = aucs[int((1 - alpha/2) * len(aucs))]
    return float(np.mean(aucs)), lo, hi


def compute_auc(scores, labels):
    if not any(labels) or all(labels):
        return float('nan')
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    concordant = sum(1 for p in pos for n in neg if p > n)
    tied = sum(1 for p in pos for n in neg if p == n)
    return (concordant + 0.5 * tied) / (len(pos) * len(neg))


def permutation_test(scores, labels, n_perm=5000):
    if not any(labels) or all(labels):
        return float('nan')
    observed = compute_auc(scores, labels)
    labels_arr = list(labels)
    count = 0
    for _ in range(n_perm):
        np.random.shuffle(labels_arr)
        if compute_auc(scores, labels_arr) >= observed:
            count += 1
    return count / n_perm


def reanalyze(obs_path):
    with open(obs_path) as f:
        observations = json.load(f)

    idx = {}
    for r in observations:
        key = (r['var'], r['val'], r['surface'], r['continuation'], r['query'])
        idx[key] = r

    EQUIV_PAIRS = [
        ("assert_var_2", "assert_novar_2", "strong_assert_2"),
        ("boundary_1", "boundary_2", "boundary_pair"),
        ("minimal_ok", "minimal_good", "minimal_ok_good"),
    ]
    CELLS = [(4, "x", 1), (4, "x", 5), (4, "y", 2), (4, "y", 6),
             (4, "z", 3), (4, "z", 7)]
    CONTINUATIONS = ["empty", "identity", "heavy", "nonvar_42", "neutral_pass",
                     "type_check", "safety_comment", "deprecation"]
    TAU = 0.05

    results = {}

    # === ANALYSIS 1: Question-invariance (R6 §3.5) ===
    # Spearman of TV_Qi vs TV_Qj across all rows, continuous
    all_queries = ["q1_print", "q2_assert", "q2_arithmetic"]  # exclude conditional
    tv_by_query = defaultdict(list)
    row_keys = []

    for s1, s2, pair_name in EQUIV_PAIRS:
        for depth, var, val in CELLS:
            base_s1 = (var, val, s1, "empty", "q1_print")
            base_s2 = (var, val, s2, "empty", "q1_print")
            if base_s1 not in idx or base_s2 not in idx:
                continue
            baseline = tv(idx[base_s1]['dist'], idx[base_s2]['dist'])
            if baseline > TAU:
                continue

            for cont_name in CONTINUATIONS:
                if cont_name == "empty":
                    continue
                valid = True
                tvs = {}
                for q in all_queries:
                    k1 = (var, val, s1, cont_name, q)
                    k2 = (var, val, s2, cont_name, q)
                    if k1 not in idx or k2 not in idx:
                        valid = False
                        break
                    tvs[q] = float(tv(idx[k1]['dist'], idx[k2]['dist']))
                if not valid:
                    continue

                row_keys.append((pair_name, f"{var}_{val}", cont_name))
                for q in all_queries:
                    tv_by_query[q].append(tvs[q])

    qi_results = {}
    for qi, qj in combinations(all_queries, 2):
        rho, p = spearman_rho(tv_by_query[qi], tv_by_query[qj])
        qi_results[f"{qi}_vs_{qj}"] = {"rho": round(rho, 4), "p": round(p, 6)}

    results["question_invariance"] = {
        "n_rows": len(row_keys),
        "pairwise_spearman": qi_results,
    }
    print(f"\n=== QUESTION INVARIANCE ({len(row_keys)} rows) ===")
    for k, v in qi_results.items():
        print(f"  {k}: rho={v['rho']:.3f}, p={v['p']:.4f}")

    # === ANALYSIS 2: Transfer AUC with fair baselines ===
    for q2_name in ["q2_assert", "q2_arithmetic"]:  # exclude conditional
        transfer_rows = []
        for s1, s2, pair_name in EQUIV_PAIRS:
            for depth, var, val in CELLS:
                base_s1 = (var, val, s1, "empty", "q1_print")
                base_s2 = (var, val, s2, "empty", "q1_print")
                if base_s1 not in idx or base_s2 not in idx:
                    continue
                baseline = tv(idx[base_s1]['dist'], idx[base_s2]['dist'])
                if baseline > TAU:
                    continue

                # FIX 4: filter baseline flips under q2
                base_q2_s1 = (var, val, s1, "empty", q2_name)
                base_q2_s2 = (var, val, s2, "empty", q2_name)
                if base_q2_s1 in idx and base_q2_s2 in idx:
                    if idx[base_q2_s1]['argmax_label'] != idx[base_q2_s2]['argmax_label']:
                        continue  # pair already flips at baseline under q2

                for cont_name in CONTINUATIONS:
                    if cont_name == "empty":
                        continue

                    q1_s1 = (var, val, s1, cont_name, "q1_print")
                    q1_s2 = (var, val, s2, cont_name, "q1_print")
                    q2_s1 = (var, val, s1, cont_name, q2_name)
                    q2_s2 = (var, val, s2, cont_name, q2_name)
                    if not all(k in idx for k in [q1_s1, q1_s2, q2_s1, q2_s2]):
                        continue

                    tv_q1 = float(tv(idx[q1_s1]['dist'], idx[q1_s2]['dist']))
                    tv_q2 = float(tv(idx[q2_s1]['dist'], idx[q2_s2]['dist']))
                    flip_q2 = idx[q2_s1]['argmax_label'] != idx[q2_s2]['argmax_label']

                    # FIX 2: fair margin baseline under Q1, not Q_flip
                    margin_q1_s1 = sorted(idx[q1_s1]['dist'], reverse=True)
                    margin_q1_s2 = sorted(idx[q1_s2]['dist'], reverse=True)
                    min_margin_q1 = min(
                        margin_q1_s1[0] - margin_q1_s1[1],
                        margin_q1_s2[0] - margin_q1_s2[1],
                    )

                    # Also compute margin under q2 for comparison (the rigged one)
                    margin_q2_s1 = sorted(idx[q2_s1]['dist'], reverse=True)
                    margin_q2_s2 = sorted(idx[q2_s2]['dist'], reverse=True)
                    min_margin_q2 = min(
                        margin_q2_s1[0] - margin_q2_s1[1],
                        margin_q2_s2[0] - margin_q2_s2[1],
                    )

                    transfer_rows.append({
                        "pair": pair_name, "cell": f"{var}_{val}",
                        "continuation": cont_name,
                        "tv_q1": tv_q1, "tv_q2": tv_q2,
                        "flip_q2": bool(flip_q2),
                        "min_margin_q1": float(min_margin_q1),
                        "min_margin_q2": float(min_margin_q2),
                    })

        n_flips = sum(1 for r in transfer_rows if r['flip_q2'])
        auc_tv_q1 = compute_auc([r['tv_q1'] for r in transfer_rows],
                                [r['flip_q2'] for r in transfer_rows])
        auc_margin_q1 = compute_auc([-r['min_margin_q1'] for r in transfer_rows],
                                    [r['flip_q2'] for r in transfer_rows])
        auc_margin_q2 = compute_auc([-r['min_margin_q2'] for r in transfer_rows],
                                    [r['flip_q2'] for r in transfer_rows])
        auc_tv_q2 = compute_auc([r['tv_q2'] for r in transfer_rows],
                                [r['flip_q2'] for r in transfer_rows])

        # Bootstrap CIs
        _, tv_q1_lo, tv_q1_hi = bootstrap_auc_ci(
            [r['tv_q1'] for r in transfer_rows],
            [r['flip_q2'] for r in transfer_rows])
        perm_p = permutation_test(
            [r['tv_q1'] for r in transfer_rows],
            [r['flip_q2'] for r in transfer_rows])

        results[f"transfer_{q2_name}"] = {
            "n_rows": len(transfer_rows),
            "n_flips": n_flips,
            "auc_tv_q1": round(auc_tv_q1, 4),
            "auc_tv_q1_ci": [round(tv_q1_lo, 4), round(tv_q1_hi, 4)],
            "auc_tv_q1_perm_p": round(perm_p, 4),
            "auc_margin_q1_fair": round(auc_margin_q1, 4),
            "auc_margin_q2_rigged": round(auc_margin_q2, 4),
            "auc_tv_q2": round(auc_tv_q2, 4),
        }

        print(f"\n=== TRANSFER → {q2_name} ({len(transfer_rows)} rows, {n_flips} flips) ===")
        print(f"  AUC(TV_Q1→flip_Q2) = {auc_tv_q1:.3f} [{tv_q1_lo:.3f}, {tv_q1_hi:.3f}] (perm p={perm_p:.3f})")
        print(f"  AUC(margin_Q1→flip_Q2) = {auc_margin_q1:.3f} [FAIR baseline]")
        print(f"  AUC(margin_Q2→flip_Q2) = {auc_margin_q2:.3f} [RIGGED, for comparison]")
        print(f"  AUC(TV_Q2→flip_Q2) = {auc_tv_q2:.3f} [same-Q upper bound]")

    # === ANALYSIS 3: Certificate rate ===
    cert_rows = []
    cert_violations = 0
    for q2_name in ["q2_assert", "q2_arithmetic"]:
        tr = results.get(f"transfer_{q2_name}", {})
        # Need the raw rows - rebuild from observations
        pass

    # Certificate: TV_Q1 < margin_Q1/2 → should NOT flip under Q2
    all_transfer = []
    for q2_name in ["q2_assert", "q2_arithmetic"]:
        for s1, s2, pair_name in EQUIV_PAIRS:
            for depth, var, val in CELLS:
                base_s1 = (var, val, s1, "empty", "q1_print")
                base_s2 = (var, val, s2, "empty", "q1_print")
                if base_s1 not in idx or base_s2 not in idx:
                    continue
                baseline = tv(idx[base_s1]['dist'], idx[base_s2]['dist'])
                if baseline > TAU:
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

                    tv_q1 = float(tv(idx[q1_s1]['dist'], idx[q1_s2]['dist']))
                    margin_q1_s1 = sorted(idx[q1_s1]['dist'], reverse=True)
                    margin_q1_s2 = sorted(idx[q1_s2]['dist'], reverse=True)
                    min_margin = min(margin_q1_s1[0] - margin_q1_s1[1],
                                     margin_q1_s2[0] - margin_q1_s2[1])
                    flip_q2 = idx[q2_s1]['argmax_label'] != idx[q2_s2]['argmax_label']

                    certified = tv_q1 < min_margin / 2
                    all_transfer.append({
                        "q2": q2_name, "pair": pair_name,
                        "cell": f"{var}_{val}", "continuation": cont_name,
                        "tv_q1": tv_q1, "min_margin_q1": float(min_margin),
                        "certified": certified, "flip_q2": bool(flip_q2),
                        "violation": certified and flip_q2,
                    })

    n_certified = sum(1 for r in all_transfer if r['certified'])
    n_violations = sum(1 for r in all_transfer if r['violation'])
    results["certificate"] = {
        "n_total": len(all_transfer),
        "n_certified": n_certified,
        "n_violations": n_violations,
        "violation_rate": round(n_violations / max(n_certified, 1), 4),
    }
    print(f"\n=== CERTIFICATE RATE ===")
    print(f"  {n_certified}/{len(all_transfer)} rows certified (TV_Q1 < margin_Q1/2)")
    print(f"  {n_violations} violations (certified but flipped under Q2)")
    print(f"  Violation rate: {n_violations/max(n_certified,1):.1%}")

    # === VERDICT ===
    qi_rhos = [v['rho'] for v in results['question_invariance']['pairwise_spearman'].values()
               if not np.isnan(v['rho'])]
    mean_qi_rho = float(np.mean(qi_rhos)) if qi_rhos else 0

    if mean_qi_rho > 0.7 and n_violations == 0:
        verdict = "STRONG_TRANSFER: question-invariant omega, zero certificate violations"
    elif mean_qi_rho > 0.5:
        verdict = f"MODERATE_TRANSFER: mean QI rho={mean_qi_rho:.2f}, {n_violations} violations"
    elif mean_qi_rho > 0.3:
        verdict = f"WEAK_TRANSFER: mean QI rho={mean_qi_rho:.2f}, insufficient invariance"
    else:
        verdict = f"NO_TRANSFER: mean QI rho={mean_qi_rho:.2f}, omega is question-dependent"

    results["verdict"] = verdict
    results["mean_question_invariance_rho"] = round(mean_qi_rho, 4)
    print(f"\n=== VERDICT: {verdict} ===")

    return results


if __name__ == "__main__":
    obs_path = Path("experiments/results/omega_crossquestion/observations.json")
    if not obs_path.exists():
        print(f"No data at {obs_path}. Run run_omega_crossquestion.py first.")
        sys.exit(1)

    results = reanalyze(obs_path)
    out_dir = obs_path.parent
    with open(out_dir / "reanalysis_r6.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir / 'reanalysis_r6.json'}")
