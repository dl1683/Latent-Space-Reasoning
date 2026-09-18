"""Full-Vocab Incremental Value Test (Experiment #63).

After exp #55b runs, this tests whether full-vocab TV predicts full-vocab
flips better than margin does. The 11-bin incremental test (exp #62) showed
omega is epiphenomenal — but that was in the 11-bin world where most
distributional information is collapsed.

Reads exp #55b results. Zero compute.
"""

import json
import numpy as np
from pathlib import Path


def load_data():
    p = Path("experiments/results/omega_consequence_fullvocab/summary.json")
    if not p.exists():
        print("ERROR: exp #55b results not found. Run run_omega_consequence_fullvocab.py first.")
        return None
    with open(p) as f:
        return json.load(f)


def safe_corr(x, y):
    if x.std() > 0 and y.std() > 0:
        return float(np.corrcoef(x, y)[0, 1])
    return 0.0


def auc(scores, labels):
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    a = 0
    for pv in pos:
        a += np.sum(pv > neg) + 0.5 * np.sum(pv == neg)
    return float(a / (len(pos) * len(neg)))


def analyze(summary):
    pairs = summary["all_pairs"]
    n = len(pairs)

    tv_bin11 = np.array([p["tv_bin11"] for p in pairs])
    tv_fv = np.array([p["tv_fullvocab"] for p in pairs])
    fv_flip = np.array([1 if p["fullvocab_flip"] else 0 for p in pairs])
    b11_flip = np.array([1 if p["bin11_flip"] else 0 for p in pairs])

    n_fv_flips = int(fv_flip.sum())
    n_b11_flips = int(b11_flip.sum())

    print(f"\n=== FULL-VOCAB INCREMENTAL ANALYSIS (Exp #63) ===\n")
    print(f"Pairs: {n}")
    print(f"Full-vocab flips: {n_fv_flips} ({n_fv_flips/n:.1%})")
    print(f"11-bin flips: {n_b11_flips} ({n_b11_flips/n:.1%})")

    print(f"\n--- Correlations ---")
    print(f"  bin11-TV vs bin11-flip:     r={safe_corr(tv_bin11, b11_flip):+.3f}")
    print(f"  bin11-TV vs fullvocab-flip: r={safe_corr(tv_bin11, fv_flip):+.3f}")
    print(f"  fullvocab-TV vs fullvocab-flip: r={safe_corr(tv_fv, fv_flip):+.3f}  <-- KEY")
    print(f"  fullvocab-TV vs bin11-flip: r={safe_corr(tv_fv, b11_flip):+.3f}")
    print(f"  bin11-TV vs fullvocab-TV:   r={safe_corr(tv_bin11, tv_fv):+.3f}")

    print(f"\n--- AUC for flip prediction ---")
    if n_fv_flips > 0 and n_fv_flips < n:
        auc_fv_tv_fv_flip = auc(tv_fv, fv_flip)
        auc_bin_tv_fv_flip = auc(tv_bin11, fv_flip)
        print(f"  fullvocab-TV -> fullvocab-flip: AUC={auc_fv_tv_fv_flip:.3f}")
        print(f"  bin11-TV -> fullvocab-flip:     AUC={auc_bin_tv_fv_flip:.3f}")
    else:
        print(f"  Cannot compute AUC (all same class for fullvocab-flip)")

    if n_b11_flips > 0 and n_b11_flips < n:
        auc_fv_tv_b11_flip = auc(tv_fv, b11_flip)
        auc_bin_tv_b11_flip = auc(tv_bin11, b11_flip)
        print(f"  fullvocab-TV -> bin11-flip:     AUC={auc_fv_tv_b11_flip:.3f}")
        print(f"  bin11-TV -> bin11-flip:         AUC={auc_bin_tv_b11_flip:.3f}")
    else:
        print(f"  Cannot compute AUC (all same class for bin11-flip)")

    print(f"\n--- TV magnitude comparison ---")
    print(f"  Mean full-vocab TV: {tv_fv.mean():.4f} (std {tv_fv.std():.4f})")
    print(f"  Mean 11-bin TV:     {tv_bin11.mean():.4f} (std {tv_bin11.std():.4f})")
    ratio = tv_fv.mean() / max(tv_bin11.mean(), 1e-10)
    print(f"  Ratio: {ratio:.1f}x")

    if n_fv_flips > 0:
        fv_flip_idx = fv_flip == 1
        nf_flip_idx = fv_flip == 0
        print(f"\n--- Full-vocab TV at flip vs no-flip ---")
        print(f"  Flip mean TV:    {tv_fv[fv_flip_idx].mean():.4f}")
        print(f"  No-flip mean TV: {tv_fv[nf_flip_idx].mean():.4f}")
        print(f"  Ratio: {tv_fv[fv_flip_idx].mean() / max(tv_fv[nf_flip_idx].mean(), 1e-10):.2f}x")

    print(f"\n--- Per-continuation full-vocab flip rate ---")
    by_cont = {}
    for p in pairs:
        c = p["continuation"]
        if c not in by_cont:
            by_cont[c] = {"fv_flips": 0, "b11_flips": 0, "n": 0,
                          "tv_fv": [], "tv_b11": []}
        by_cont[c]["n"] += 1
        by_cont[c]["fv_flips"] += 1 if p["fullvocab_flip"] else 0
        by_cont[c]["b11_flips"] += 1 if p["bin11_flip"] else 0
        by_cont[c]["tv_fv"].append(p["tv_fullvocab"])
        by_cont[c]["tv_b11"].append(p["tv_bin11"])

    for c in sorted(by_cont.keys()):
        d = by_cont[c]
        fv_fr = d["fv_flips"] / d["n"]
        b11_fr = d["b11_flips"] / d["n"]
        mtv_fv = np.mean(d["tv_fv"])
        mtv_b11 = np.mean(d["tv_b11"])
        print(f"  {c:20s}: fv_flip={fv_fr:.1%} b11_flip={b11_fr:.1%} "
              f"tvfv={mtv_fv:.4f} tvb11={mtv_b11:.4f}")

    results = {
        "n_pairs": n,
        "n_fv_flips": n_fv_flips,
        "n_b11_flips": n_b11_flips,
        "corr_fv_tv_fv_flip": safe_corr(tv_fv, fv_flip),
        "corr_bin_tv_fv_flip": safe_corr(tv_bin11, fv_flip),
        "corr_fv_tv_bin_flip": safe_corr(tv_fv, b11_flip),
        "corr_bin_tv_bin_flip": safe_corr(tv_bin11, b11_flip),
        "mean_fv_tv": float(tv_fv.mean()),
        "mean_bin_tv": float(tv_bin11.mean()),
        "tv_ratio": float(ratio),
    }
    return results


if __name__ == "__main__":
    print("=== Full-Vocab Incremental Analysis (Exp #63) ===\n")

    summary = load_data()
    if summary is None:
        exit(1)

    results = analyze(summary)

    out_dir = Path("experiments/results/omega_fullvocab_incremental")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir}")
