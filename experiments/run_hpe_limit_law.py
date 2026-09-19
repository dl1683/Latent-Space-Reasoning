"""HP-E limit-law validation: two-corner phase diagram (theory 4b).

Pre-registered (frozen before running). Coordinate-sparse directions,
Sigma = diag(r^{-alpha}), d = 512, REPS = 300.
Grid: k/d in {0.05, 5} x alpha in {0, 0.5, 0.9}.
Predicted classes:
- dense corner (k/d=5), alpha=0: channel-PR large -> concentrated
  (cv <= 3/(sqrt of effective channels scale); registered: cv < 0.15);
- dense, alpha=0.9: channel-PR ~ O(1) -> broad (cv > 0.3);
- sparse corner (k/d=0.05), alpha=0: third-moment PR large -> cv < 0.25;
- sparse, alpha=0.9: third-moment PR O(1) -> broad (cv > 0.4);
- alpha=0.5 cells: intermediate (no registered threshold; reported).
Verdict = 4/4 corner cells matching registered bands.
"""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).parent / "results" / "hpe_aggregate"
REPS = 300


def a2_spare(d, alpha, k, rng):
    lam = np.arange(1, d + 1, dtype=np.float64) ** (-alpha / 2.0)
    idx = rng.integers(0, d, size=k)
    N = np.bincount(idx, minlength=d)
    sig2 = lam ** 2
    w = sig2 * (N * (N - 1) / 2.0)
    return 2.0 * w.sum() * (d / (sig2.sum()))  # c^4 = (d/trSigma)^2 with trSigma=sig2.sum()


def main():
    d = 512
    rng = np.random.default_rng(99)
    rows = []
    for ratio in [0.05, 5.0]:
        k = max(8, int(ratio * d))
        for alpha in [0.0, 0.5, 0.9]:
            vals = np.array([a2_spare(d, alpha, k, rng) for _ in range(REPS)])
            cv = float(vals.std() / vals.mean())
            lam = np.arange(1, d + 1, dtype=np.float64) ** (-alpha)
            pr2 = lam.sum() ** 2 / (lam ** 2).sum()
            pr3 = (lam ** 1.5).sum() ** 2 / (lam ** 3).sum()
            rows.append(dict(koverd=ratio, k=k, alpha=alpha, cv=cv,
                             channel_PR=float(pr2), third_moment_PR=float(pr3)))
            print(f"k/d={ratio} alpha={alpha}: cv={cv:.3f} channelPR={pr2:.0f} thirdPR={pr3:.0f}")
    json.dump(rows, open(OUT / "limit_law_validation.json", "w"), indent=1)
    get = lambda r, a: next(x["cv"] for x in rows if x["koverd"] == r and x["alpha"] == a)
    v = dict(dense_a0=get(5.0, 0.0) < 0.15, dense_a09=get(5.0, 0.9) > 0.3,
             sparse_a0=get(0.05, 0.0) < 0.25, sparse_a09=get(0.05, 0.9) > 0.4)
    json.dump(v, open(OUT / "limit_law_verdicts.json", "w"), indent=1)
    print("VERDICTS:", v, "=>", "PASS 4/4" if all(v.values()) else "see cells")


if __name__ == "__main__":
    main()
