"""HP-D s=1 gap diagnosis: is the 6x Fisher-law underestimate at (B=8, rho=0.3,
s=1, d=2/4) an optimizer artifact or real higher-order KL excess?
20 seeds x 12000 steps, lr in {0.05, 0.2}. Decision (frozen):
- if best excess falls below 2x the Fisher prediction (0.014 -> below 0.028
  at d=2), the locator rows were optimizer-limited: strengthen and re-score.
- if best excess stays above 3x the Fisher prediction (> 0.042 at d=2) across
  all 20 seeds, the gap is genuine large-deviation KL: the nonlinear theory
  is required (open unit confirmed).
"""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).parent / "results" / "hpd_compression"


def block_kernel(n, B, rho, s):
    ranks = np.arange(1, n + 1, dtype=np.float64)
    p = ranks ** (-s); p /= p.sum()
    block = np.arange(n) * B // n
    same = block[:, None] == block[None, :]
    P = (1 - rho) * p[None, :] + rho * same / (n / B)
    return P, p


def run(n, P, p, d, seed, steps=12000, lr=0.05):
    rng = np.random.default_rng(seed)
    H = 0.1 * rng.standard_normal((d, n))
    W = 0.1 * rng.standard_normal((d, n))
    mH = np.zeros_like(H); vH = np.zeros_like(H)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    b1, b2, eps = 0.9, 0.999, 1e-8
    PT = P.T * p[None, :]
    for t in range(1, steps + 1):
        Z = W.T @ H
        Zs = Z - Z.max(axis=0, keepdims=True)
        Q = np.exp(Zs); Q /= Q.sum(axis=0, keepdims=True)
        G = (Q - P.T) * p[None, :]
        gW = H @ G.T; gH = W @ G
        mH = b1 * mH + (1 - b1) * gH; vH = b2 * vH + (1 - b2) * gH * gH
        mW = b1 * mW + (1 - b1) * gW; vW = b2 * vW + (1 - b2) * gW * gW
        H -= lr * (mH / (1 - b1 ** t)) / (np.sqrt(vH / (1 - b2 ** t)) + eps)
        W -= lr * (mW / (1 - b1 ** t)) / (np.sqrt(vW / (1 - b2 ** t)) + eps)
    Z = W.T @ H
    Zs = Z - Z.max(axis=0, keepdims=True)
    Q = np.exp(Zs); Q /= Q.sum(axis=0, keepdims=True)
    return float(-(PT * np.log(Q + 1e-300)).sum())


def main():
    n, B, rho, s = 512, 8, 0.3, 1.0
    P, p = block_kernel(n, B, rho, s)
    floor = float(-(p[:, None] * P * np.log(P)).sum())
    results = {}
    for d in [2, 4]:
        vals = []
        for lr in [0.05, 0.2]:
            for seed in range(20):
                vals.append(run(n, P, p, d, 1000 + seed, lr=lr) - floor)
        vals = sorted(vals)
        results[d] = dict(best=min(vals), median=vals[len(vals) // 2],
                          worst=max(vals), fisher_pred=(0.0139 if d == 2 else 0.0054))
        print(f"d={d}: best={min(vals):.4f} median={vals[len(vals)//2]:.4f} "
              f"worst={max(vals):.4f} fisher={results[d]['fisher_pred']}")
    json.dump(results, open(OUT / "s1_gap_diagnosis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
