"""R6-recommended free reanalyses of cross-question #58 data.

Addresses R6's five criticisms:
1. Exclude comparison Q (readout-mismatched, answers map to OTHER)
2. Fair margin: margin_Q_tv instead of margin_Q_flip
3. Power reporting: exact CIs per combo
4. Baseline-flip filter: exclude pairs that flip with empty continuation
5. Direct Spearman: TV rank stability across questions (the native quantity)

Plus: certificate recheck with proper per-question margins.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations


def tv(a, b):
    return 0.5 * np.abs(np.array(a) - np.array(b)).sum()


SURFACE_PAIRS = {
    ("assert_var_2", "assert_novar_2"): "strong_assert_2",
    ("boundary_1", "boundary_2"): "boundary_pair",
    ("minimal_good", "minimal_ok"): "minimal_pair",
}

PAIR_LOOKUP = {}
for (sa, sb), name in SURFACE_PAIRS.items():
    PAIR_LOOKUP[sa] = (sb, name)
    PAIR_LOOKUP[sb] = (sa, name)


def load_data():
    with open("experiments/results/omega_cross_question/checkpoint.json") as f:
        data = json.load(f)
    return data["obs"]


def build_lookup(obs):
    """Index: (cell, surface, continuation, question) -> observation."""
    idx = {}
    for o in obs:
        key = (f"{o['var']}_{o['val']}", o["surface"], o["continuation"], o["question"])
        idx[key] = o
    return idx


def compute_pair_metrics(idx):
    """For each (cell, pair, continuation, question): TV, margin_a, margin_b, flip."""
    actual_cells = sorted(set(k[0] for k in idx))
    actual_conts = sorted(set(k[2] for k in idx))
    actual_qs = sorted(set(k[3] for k in idx))
    rows = []
    for (sa, sb), pair_name in SURFACE_PAIRS.items():
        for cell in actual_cells:
            for cont in actual_conts:
                for q in actual_qs:
                    key_a = (cell, sa, cont, q)
                    key_b = (cell, sb, cont, q)
                    if key_a not in idx or key_b not in idx:
                        continue
                    oa = idx[key_a]
                    ob = idx[key_b]
                    pair_tv = tv(oa["dist"], ob["dist"])
                    min_margin = min(oa["margin"], ob["margin"])
                    flip = oa["argmax_label"] != ob["argmax_label"]
                    rows.append({
                        "cell": cell,
                        "pair": pair_name,
                        "continuation": cont,
                        "question": q,
                        "tv": float(pair_tv),
                        "margin_a": float(oa["margin"]),
                        "margin_b": float(ob["margin"]),
                        "min_margin": float(min_margin),
                        "argmax_a": oa["argmax_label"],
                        "argmax_b": ob["argmax_label"],
                        "flip": flip,
                        "correct_a": oa["is_correct"],
                        "correct_b": ob["is_correct"],
                    })
    return rows


def auc(scores, labels):
    if not any(labels) or all(labels):
        return float("nan")
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    c = sum(1 for p in pos for n in neg if p > n)
    t = sum(1 for p in pos for n in neg if p == n)
    return (c + 0.5 * t) / (len(pos) * len(neg))


def bootstrap_auc_ci(scores, labels, n_boot=2000, alpha=0.05):
    if not any(labels) or all(labels):
        return float("nan"), float("nan"), float("nan")
    scores = np.array(scores)
    labels = np.array(labels)
    n = len(scores)
    rng = np.random.RandomState(42)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        a = auc(scores[idx].tolist(), labels[idx].tolist())
        if not np.isnan(a):
            aucs.append(a)
    if len(aucs) < 10:
        return float("nan"), float("nan"), float("nan")
    lo = np.percentile(aucs, 100 * alpha / 2)
    hi = np.percentile(aucs, 100 * (1 - alpha / 2))
    return float(np.mean(aucs)), float(lo), float(hi)


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3:
        return float("nan"), float("nan")
    r, p = spearmanr(x, y)
    return float(r), float(p)


def run():
    obs = load_data()
    idx = build_lookup(obs)
    rows = compute_pair_metrics(idx)

    questions = sorted(set(r["question"] for r in rows))
    conts = sorted(set(r["continuation"] for r in rows))
    print(f"Total pair-rows: {len(rows)}")
    print(f"Questions: {questions}")
    print(f"Continuations: {conts}")

    # ================================================================
    # REANALYSIS 1: Exclude comparison question
    # ================================================================
    rows_no_comp = [r for r in rows if r["question"] != "comparison"]
    qs_clean = [q for q in questions if q != "comparison"]
    print(f"\n=== R6 FIX 1: Excluding comparison ===")
    print(f"Rows: {len(rows)} -> {len(rows_no_comp)}")

    comp_rows = [r for r in rows if r["question"] == "comparison"]
    comp_flips = sum(1 for r in comp_rows if r["flip"])
    comp_mean_tv = np.mean([r["tv"] for r in comp_rows]) if comp_rows else 0
    print(f"Comparison stats: {len(comp_rows)} rows, {comp_flips} flips, mean TV={comp_mean_tv:.4f}")

    # ================================================================
    # REANALYSIS 4: Baseline-flip filter
    # ================================================================
    baseline_flippers = set()
    for r in rows_no_comp:
        if r["continuation"] == "empty" and r["flip"]:
            baseline_flippers.add((r["cell"], r["pair"], r["question"]))

    rows_filtered = [r for r in rows_no_comp
                     if (r["cell"], r["pair"], r["question"]) not in baseline_flippers]
    print(f"\n=== R6 FIX 4: Baseline-flip filter ===")
    print(f"Baseline flippers: {len(baseline_flippers)} (cell, pair, Q) combos")
    print(f"Rows: {len(rows_no_comp)} -> {len(rows_filtered)}")

    # ================================================================
    # REANALYSIS 2: Fair margin (transfer AUC with margin_Q_tv)
    # ================================================================
    print(f"\n=== R6 FIX 2: Fair margin comparison ===")
    print(f"{'Q_tv':>15} -> {'Q_flip':>15} | {'tv_AUC':>8} {'CI':>20} | {'margin_qtv':>10} {'CI':>20} | {'margin_qflip':>12} | {'n':>3} {'flips':>5}")

    transfer_results = []
    for q_tv in qs_clean:
        for q_flip in qs_clean:
            is_transfer = q_tv != q_flip
            sub = [r for r in rows_filtered if r["question"] == q_tv]
            flip_lookup = {}
            for r in rows_filtered:
                if r["question"] == q_flip:
                    flip_lookup[(r["cell"], r["pair"], r["continuation"])] = r["flip"]

            tvs, margins_qtv, margins_qflip, flips = [], [], [], []
            for r in sub:
                key = (r["cell"], r["pair"], r["continuation"])
                if key in flip_lookup:
                    tvs.append(r["tv"])
                    margins_qtv.append(r["min_margin"])
                    flip_row = [rr for rr in rows_filtered
                                if rr["question"] == q_flip
                                and rr["cell"] == r["cell"]
                                and rr["pair"] == r["pair"]
                                and rr["continuation"] == r["continuation"]]
                    margins_qflip.append(flip_row[0]["min_margin"] if flip_row else 0)
                    flips.append(flip_lookup[key])

            n = len(tvs)
            n_flips = sum(flips)
            tv_auc, tv_lo, tv_hi = bootstrap_auc_ci(tvs, flips)
            mg_qtv_auc, mg_qtv_lo, mg_qtv_hi = bootstrap_auc_ci(
                [-m for m in margins_qtv], flips)
            mg_qflip_auc = auc([-m for m in margins_qflip], flips)

            print(f"{q_tv:>15} -> {q_flip:>15} | {tv_auc:8.3f} [{tv_lo:.2f},{tv_hi:.2f}] | "
                  f"{mg_qtv_auc:10.3f} [{mg_qtv_lo:.2f},{mg_qtv_hi:.2f}] | "
                  f"{mg_qflip_auc:12.3f} | {n:3d} {n_flips:5d}"
                  f" {'TRANSFER' if is_transfer else 'same-Q'}")

            transfer_results.append({
                "q_tv": q_tv, "q_flip": q_flip, "is_transfer": is_transfer,
                "n": n, "n_flips": n_flips,
                "tv_auc": tv_auc, "tv_ci": [tv_lo, tv_hi],
                "margin_qtv_auc": mg_qtv_auc, "margin_qtv_ci": [mg_qtv_lo, mg_qtv_hi],
                "margin_qflip_auc": mg_qflip_auc,
            })

    # ================================================================
    # REANALYSIS 5: Direct Spearman (TV rank stability across questions)
    # ================================================================
    print(f"\n=== R6 FIX 5: Direct Spearman (TV rank stability) ===")
    tv_by_q = defaultdict(lambda: defaultdict(float))
    for r in rows_filtered:
        key = (r["cell"], r["pair"], r["continuation"])
        tv_by_q[key][r["question"]] = r["tv"]

    keys_all_qs = [k for k, qdict in tv_by_q.items()
                   if all(q in qdict for q in qs_clean)]

    print(f"Rows with all {len(qs_clean)} questions: {len(keys_all_qs)}")

    spearman_pairs = []
    for qa, qb in combinations(qs_clean, 2):
        x = [tv_by_q[k][qa] for k in keys_all_qs]
        y = [tv_by_q[k][qb] for k in keys_all_qs]
        rho, p = spearman(x, y)
        print(f"  {qa} vs {qb}: rho={rho:.3f}, p={p:.4f}, n={len(x)}")
        spearman_pairs.append({"qa": qa, "qb": qb, "rho": rho, "p": p, "n": len(x)})

    all_x = []
    all_y = []
    for qa, qb in combinations(qs_clean, 2):
        x = [tv_by_q[k][qa] for k in keys_all_qs]
        y = [tv_by_q[k][qb] for k in keys_all_qs]
        all_x.extend(x)
        all_y.extend(y)
    pooled_rho, pooled_p = spearman(all_x, all_y)
    print(f"  Pooled: rho={pooled_rho:.3f}, p={pooled_p:.6f}, n={len(all_x)}")

    # ================================================================
    # CERTIFICATE RECHECK
    # ================================================================
    print(f"\n=== Certificate recheck (TV < margin/2 theorem) ===")
    n_cert = 0
    n_viol = 0
    n_pass = 0
    violations = []
    for r in rows_filtered:
        if r["continuation"] == "empty":
            continue
        if r["tv"] < r["min_margin"] / 2:
            n_cert += 1
            if r["flip"]:
                n_viol += 1
                violations.append(r)
        else:
            n_pass += 1

    total = n_cert + n_pass
    print(f"Certified (TV < margin/2): {n_cert}/{total} ({100*n_cert/max(total,1):.1f}%)")
    print(f"Violations among certified: {n_viol}/{n_cert} ({100*n_viol/max(n_cert,1):.1f}%)")
    if violations:
        print("Violation details:")
        for v in violations:
            print(f"  {v['cell']}/{v['pair']}/{v['continuation']}/{v['question']}: "
                  f"TV={v['tv']:.4f}, margin={v['min_margin']:.4f}, "
                  f"argmax {v['argmax_a']} vs {v['argmax_b']}")

    # Cross-question certificate
    print(f"\n=== Cross-Q certificate ===")
    for q_tv in qs_clean:
        for q_flip in qs_clean:
            if q_tv == q_flip:
                continue
            sub_tv = {(r["cell"], r["pair"], r["continuation"]): r
                      for r in rows_filtered if r["question"] == q_tv
                      and r["continuation"] != "empty"}
            sub_flip = {(r["cell"], r["pair"], r["continuation"]): r
                        for r in rows_filtered if r["question"] == q_flip
                        and r["continuation"] != "empty"}
            common = set(sub_tv) & set(sub_flip)
            cert = 0
            viol = 0
            for k in common:
                r_tv = sub_tv[k]
                r_flip = sub_flip[k]
                if r_tv["tv"] < r_tv["min_margin"] / 2:
                    cert += 1
                    if r_flip["flip"]:
                        viol += 1
            print(f"  TV@{q_tv} -> flip@{q_flip}: {cert} certified, {viol} violated "
                  f"({100*viol/max(cert,1):.1f}%), n_common={len(common)}")

    # ================================================================
    # SUMMARY
    # ================================================================
    transfer_only = [r for r in transfer_results if r["is_transfer"]]
    same_q = [r for r in transfer_results if not r["is_transfer"]]

    mean_transfer_tv = np.nanmean([r["tv_auc"] for r in transfer_only])
    mean_same_tv = np.nanmean([r["tv_auc"] for r in same_q])
    mean_transfer_mg = np.nanmean([r["margin_qtv_auc"] for r in transfer_only])

    summary = {
        "n_obs": len(obs),
        "n_pairs_no_comp": len(rows_no_comp),
        "n_pairs_filtered": len(rows_filtered),
        "n_baseline_flippers": len(baseline_flippers),
        "comparison_excluded": {
            "n_rows": len(comp_rows),
            "n_flips": comp_flips,
            "mean_tv": float(comp_mean_tv),
        },
        "transfer_auc_mean": float(mean_transfer_tv),
        "same_q_auc_mean": float(mean_same_tv),
        "fair_margin_transfer_mean": float(mean_transfer_mg),
        "spearman_pairs": spearman_pairs,
        "pooled_spearman": {"rho": pooled_rho, "p": pooled_p, "n": len(all_x)},
        "certificate": {
            "n_certified": n_cert,
            "n_violated": n_viol,
            "rate": float(n_viol / max(n_cert, 1)),
        },
        "transfer_results": transfer_results,
        "verdict": None,
    }

    # Determine verdict
    if mean_transfer_tv > 0.7:
        summary["verdict"] = "TRANSFER: omega predicts cross-question flips"
    elif pooled_rho > 0.4 and n_viol == 0:
        summary["verdict"] = "CERTIFICATE: omega is question-stable certificate (no flip prediction but stable ranks)"
    elif pooled_rho > 0.3:
        summary["verdict"] = "PARTIAL: moderate rank stability, transfer AUC at chance"
    else:
        summary["verdict"] = "NEGATIVE: omega is question-specific, no transfer"

    print(f"\n=== VERDICT ===")
    print(f"Mean transfer TV AUC: {mean_transfer_tv:.3f}")
    print(f"Mean same-Q TV AUC: {mean_same_tv:.3f}")
    print(f"Mean transfer margin(Q_tv) AUC: {mean_transfer_mg:.3f}")
    print(f"Pooled Spearman: rho={pooled_rho:.3f}")
    print(f"Certificate violations: {n_viol}/{n_cert}")
    print(f"VERDICT: {summary['verdict']}")

    out_dir = Path("experiments/results/omega_cross_question")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "r6_reanalysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_dir / 'r6_reanalysis.json'}")

    return summary


if __name__ == "__main__":
    run()
