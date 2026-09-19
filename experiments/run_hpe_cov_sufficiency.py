"""HP-E covariance-sufficiency test (review-6 lead unit).

Two feature ensembles with IDENTICAL covariance Sigma but different
higher-moment structure (Gaussian-delocalized vs coordinate-sparse
directions). Pair-level isotropy identity: E[(f_i.f_j)^2] = tr(Sigma^2)/d^2
for BOTH -> identical E[A_2]. The test is CONCENTRATION.

Pre-registered (frozen before running; expectation-normalized features
f = Sigma^{1/2} u, u Gaussian vs u = sqrt(d)*e_r, both isotropic):
- P1: E[A_2]/(k(k-1) tr(Sigma^2)/d^2) in [0.9, 1.1] for both ensembles
  (identity check; should PASS and does not discriminate).
- P2: cv(A_2, sparse) >= 3 x cv(A_2, Gaussian) at (d=512, alpha=0.5,
  k=64) -- higher moments break concentration.
- P3: at the crossing k ~ sqrt(PR): Gaussian A_2 spread (p90/p10) <= 3;
  sparse spread >= 3x Gaussian's -- the frontier is sharp for Gaussian
  and broad for sparse, i.e. covariance does NOT suffice for the
  high-probability statement.
Interpretations pre-registered: P2+P3 hold -> PR is expectation-level only;
the high-probability frontier needs a fourth-moment/localization functional
(next invariant target). P2/P3 fail -> PR suffices (robustness result).
"""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).parent / "results" / "hpe_aggregate"
OUT.mkdir(parents=True, exist_ok=True)
REPS = 200


def features(d, k, alpha, ens, rng):
    lam = np.arange(1, d + 1, dtype=np.float64) ** (-alpha / 2.0)
    if ens == "gauss":
        U = rng.standard_normal((k, d))
    else:  # coordinate-sparse: random axis per feature
        idx = rng.integers(0, d, size=k)
        U = np.zeros((k, d))
        U[np.arange(k), idx] = np.sqrt(d)
    return (U * lam[None, :]) * np.sqrt(d / ((lam ** 2).sum()))  # E||f||^2 = 1


def main():
    rng = np.random.default_rng(11)
    rows = []
    for alpha in [0.0, 0.5]:
        d = 512
        lam = np.arange(1, d + 1, dtype=np.float64) ** (-alpha)
        PR = lam.sum() ** 2 / (lam ** 2).sum()
        pred_pair = (lam ** 2).sum() / d ** 2
        for k in [32, 64, 128]:
            for ens in ["gauss", "sparse"]:
                a2 = []
                for r in range(REPS):
                    F = features(d, k, alpha, ens, rng)
                    G = F @ F.T
                    off = G - np.diag(np.diag(G))
                    a2.append((off ** 2).sum())
                a2 = np.array(a2)
                pred_mean = k * (k - 1) * pred_pair
                rows.append(dict(alpha=alpha, d=d, k=k, ens=ens, PR=float(PR),
                                 mean=float(a2.mean()), pred_mean=float(pred_mean),
                                 ratio=float(a2.mean() / pred_mean),
                                 cv=float(a2.std() / a2.mean()),
                                 spread_p90_p10=float(np.percentile(a2, 90) / np.percentile(a2, 10))))
                print(f"a={alpha} k={k} {ens}: ratio={a2.mean()/pred_mean:.3f} "
                      f"cv={a2.std()/a2.mean():.3f} spread={np.percentile(a2,90)/np.percentile(a2,10):.2f}")
    json.dump(rows, open(OUT / "cov_sufficiency.json", "w"), indent=1)
    g = {(r["alpha"], r["k"]): r for r in rows if r["ens"] == "gauss"}
    sp = {(r["alpha"], r["k"]): r for r in rows if r["ens"] == "sparse"}
    v = dict(
        P1=all(0.9 <= r["ratio"] <= 1.1 for r in rows),
        P2=(sp[(0.5, 64)]["cv"] >= 3 * g[(0.5, 64)]["cv"]),
        P3=(sp[(0.5, 64)]["spread_p90_p10"] >= 3 * g[(0.5, 64)]["spread_p90_p10"]),
    )
    json.dump(v, open(OUT / "cov_sufficiency_verdicts.json", "w"), indent=1)
    print("VERDICTS:", v)


if __name__ == "__main__":
    main()
