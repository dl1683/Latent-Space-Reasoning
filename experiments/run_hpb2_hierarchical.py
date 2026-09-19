"""HP-B2 (staged, NOT RUN): hierarchical-kernel compression cycle.
Frozen predictions (theory/HPD_ALLOCATION_THEOREM.md sections 13-13e):

Config: b=2 tree, L=6, n=64 leaves, Zipf(s=1) leaf masses, rho in {0.3, 0.7},
pi geometric (decay 1/2). d grid: cumulative level ranks {1,3,7,15,31,63}
plus midpoints {2,5,11,23,47}. CE optimization: Adam 8000 steps, 6 seeds,
best-of. Frozen verdicts:
- P1 (emergent knee): excess drops by >=5x at EACH cumulative rank boundary
  (stepwise structure) in both rho arms.
- P2 (within-bundle tails): between consecutive ranks, excess decays
  following the block-mass tail (log-convex in d; curvature negative).
- P3 (CGF crossover): at rho=0.7, the measured excess at the FIRST interior
  d (d=2, after dropping the root-1 gap only... the coarsest strongly-gapped
  level) exceeds the Fisher recursive-tail prediction by >=2x; at rho=0.3
  the same cell agrees within 1.5x.
Scoring: P1 10/10 boundary cells; P2 curvature sign in >=8/10 interior
segments; P3 both arms as stated. Any failure is informative-recorded.
"""
import argparse, json
from pathlib import Path
import numpy as np

OUT = Path(__file__).parent / "results" / "hpb2_hierarchical"
OUT.mkdir(parents=True, exist_ok=True)


def hier_P(n, b, L, rho, s, pi_decay):
    ranks = np.arange(1, n + 1, dtype=np.float64)
    p = ranks ** (-s); p /= p.sum()
    pi = np.array([pi_decay ** (L - 1 - ell) for ell in range(L)])
    pi /= pi.sum()
    lvl = np.zeros((n, n))
    for ell in range(L):
        A = np.array([t >> (L - ell) for t in range(n)])  # ell=0 root (1 block), ell=L-1 pairs
        same = A[:, None] == A[None, :]
        lvl += pi[ell] * same / max(1, n >> ell)  # block size at level ell = n/2^ell (root = n)
    return (1 - rho) * p[None, :] + rho * lvl, p


def run(n, P, p, d, seed, steps=8000, lr=0.05):
    rng = np.random.default_rng(seed)
    H = 0.1 * rng.standard_normal((d, n)); W = 0.1 * rng.standard_normal((d, n))
    mH = np.zeros_like(H); vH = np.zeros_like(H); mW = np.zeros_like(W); vW = np.zeros_like(W)
    b1, b2, e0 = 0.9, 0.999, 1e-8
    PT = P.T * p[None, :]
    for t in range(1, steps + 1):
        Z = W.T @ H; Zs = Z - Z.max(axis=0, keepdims=True)
        Q = np.exp(Zs); Q /= Q.sum(axis=0, keepdims=True)
        G = (Q - P.T) * p[None, :]
        gW = H @ G.T; gH = W @ G
        mH = b1*mH+(1-b1)*gH; vH = b2*vH+(1-b2)*gH*gH
        mW = b1*mW+(1-b1)*gW; vW = b2*vW+(1-b2)*gW*gW
        H -= lr*(mH/(1-b1**t))/(np.sqrt(vH/(1-b2**t))+e0)
        W -= lr*(mW/(1-b1**t))/(np.sqrt(vW/(1-b2**t))+e0)
    Z = W.T @ H; Zs = Z - Z.max(axis=0, keepdims=True)
    Q = np.exp(Zs); Q /= Q.sum(axis=0, keepdims=True)
    return float(-(PT * np.log(Q + 1e-300)).sum())


def main():
    rows = []
    n, b, L, s = 64, 2, 6, 1.0
    for rho in [0.3, 0.7]:
        P, p = hier_P(n, b, L, rho, s, 0.5)
        floor = float(-(p[:, None] * P * np.log(P)).sum())
        for d in [1, 2, 3, 5, 7, 11, 15, 23, 31, 47, 63]:
            best = min(run(n, P, p, d, sd) for sd in range(6))
            rows.append(dict(rho=rho, d=d, floor=floor, best=best, excess=best - floor))
            print(f"rho={rho} d={d}: excess={best-floor:.4f}", flush=True)
    json.dump(rows, open(OUT / "hierarchical_curve.json", "w"), indent=1)


if __name__ == "__main__":
    main()
