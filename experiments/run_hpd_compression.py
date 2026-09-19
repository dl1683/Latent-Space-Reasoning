"""HP-D compression locator: excess CE of the best rank-d factorization of a
planted-block conditional kernel. Exploratory derivation-support (logged as
such in theory/HPD_ALLOCATION_THEOREM.md section 8); the eventual phase-boundary
CLAIM is pre-registered separately after the structure is seen.

Config: n=512, B in {8,32}, rho in {0.3,0.7}, s in {0.5,1.0},
d in {2,4,8,16,32,64}, Adam 4000 steps, 3 seeds, best-of (nonconvex).
Measure: excess CE = L - conditional entropy H(P(.|I)).
"""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).parent / "results" / "hpd_compression"
OUT.mkdir(parents=True, exist_ok=True)


def block_kernel(n, B, rho, s):
    ranks = np.arange(1, n + 1, dtype=np.float64)
    p = ranks ** (-s); p /= p.sum()
    block = np.arange(n) * B // n
    same = block[:, None] == block[None, :]
    P = (1 - rho) * p[None, :] + rho * same / (n / B)
    return P, p


def ce_floor(P, p):
    return float(-(p[:, None] * P * np.log(P)).sum())


def run(n, B, rho, s, d, seed, steps=4000, lr=0.05):
    P, p = block_kernel(n, B, rho, s)
    logP = np.log(P)
    rng = np.random.default_rng(seed)
    H = 0.1 * rng.standard_normal((d, n))
    W = 0.1 * rng.standard_normal((d, n))
    mH = np.zeros_like(H); vH = np.zeros_like(H)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    b1, b2, eps = 0.9, 0.999, 1e-8
    I = np.arange(n)  # full-batch rows (n x n logits fine at n=512)
    for t in range(1, steps + 1):
        Z = W.T @ H                     # n x n: logits Z[j, i]
        Zs = Z - Z.max(axis=0, keepdims=True)
        Q = np.exp(Zs); Q /= Q.sum(axis=0, keepdims=True)
        G = (Q - P.T) * p[None, :]      # dL/dZ[j,i], column-weighted by p_i
        gW = H @ G.T
        gH = W @ G
        for Pm, gm, m, v in ((H, gH, mH, vH), (W, gW, mW, vW)):
            pass
        mH = b1 * mH + (1 - b1) * gH; vH = b2 * vH + (1 - b2) * gH * gH
        mW = b1 * mW + (1 - b1) * gW; vW = b2 * vW + (1 - b2) * gW * gW
        H -= lr * (mH / (1 - b1 ** t)) / (np.sqrt(vH / (1 - b2 ** t)) + eps)
        W -= lr * (mW / (1 - b1 ** t)) / (np.sqrt(vW / (1 - b2 ** t)) + eps)
    Z = W.T @ H
    Zs = Z - Z.max(axis=0, keepdims=True)
    Q = np.exp(Zs); Q /= Q.sum(axis=0, keepdims=True)
    loss = float(-(p[:, None] * P * np.log(Q.T + 1e-300)).sum())
    return loss


def main():
    n = 512
    rows = []
    for B in [8, 32]:
        for rho in [0.3, 0.7]:
            for s in [0.5, 1.0]:
                P, p = block_kernel(n, B, rho, s)
                floor = ce_floor(P, p)
                for d in [2, 4, 8, 16, 32, 64]:
                    if d >= n:
                        continue
                    best = min(run(n, B, rho, s, d, sd) for sd in [0, 1, 2])
                    rows.append(dict(B=B, rho=rho, s=s, d=d, floor=floor,
                                     best_loss=best, excess=best - floor))
                    print(f"B={B} rho={rho} s={s} d={d}: excess={best-floor:.4f}")
    json.dump(rows, open(OUT / "compression_locator.json", "w"), indent=1)


if __name__ == "__main__":
    main()
