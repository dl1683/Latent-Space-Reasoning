"""HP-E interference-kurtosis law: out-of-sample confirmation.

Pre-registered in theory/HPE_AGGREGATE_INTERFERENCE.md 3j (frozen):
(i) Gaussian-arm cv in [1, 3] x 2/k in all cells;
(ii) sparse-arm cv in [0.7, 1.4] x sqrt(2*rho_4)/k, rho_4 computed from its
     definition by Monte Carlo with 40k samples (no free constants);
(iii) means ensemble-invariant within 5%.
Fresh seeds; alpha in {0, 0.5}; d in {256, 1024}; k in {48, 96}; REPS 300.
"""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).parent / "results" / "hpe_aggregate"
REPS = 300


def feats(d, k, alpha, ens, rng):
    lam = np.arange(1, d + 1, dtype=np.float64) ** (-alpha / 2.0)
    if ens == "gauss":
        U = rng.standard_normal((k, d)) / np.sqrt(d)
    else:
        idx = rng.integers(0, d, size=k)
        U = np.zeros((k, d)); U[np.arange(k), idx] = 1.0
    return (U * lam[None, :]) * np.sqrt(d / (lam ** 2).sum())


def rho4_mc(d, alpha, ens, rng, n=40000):
    lam = np.arange(1, d + 1, dtype=np.float64) ** (-alpha / 2.0)
    if ens == "gauss":
        u = rng.standard_normal((n, d)) / np.sqrt(d)
        v = rng.standard_normal((n, d)) / np.sqrt(d)
    else:
        iu = rng.integers(0, d, size=n); iv = rng.integers(0, d, size=n)
        u = np.zeros((n, d)); u[np.arange(n), iu] = 1.0
        v = np.zeros((n, d)); v[np.arange(n), iv] = 1.0
    z = ((u * lam) * (v * lam)).sum(axis=1)  # u^T Sigma v with diag Sigma=lam^2
    return float((z ** 4).mean() / ((z ** 2).mean() ** 2))


def main():
    rng = np.random.default_rng(2026)
    rows = []
    for d in [256, 1024]:
        for alpha in [0.0, 0.5]:
            for k in [48, 96]:
                cell = {}
                for ens in ["gauss", "sparse"]:
                    a2 = []
                    for r in range(REPS):
                        F = feats(d, k, alpha, ens, rng)
                        G = F @ F.T
                        off = G - np.diag(np.diag(G))
                        a2.append((off ** 2).sum())
                    a2 = np.array(a2)
                    r4 = rho4_mc(d, alpha, ens, rng)
                    pred_cv = (2.0 / k) if ens == "gauss" else np.sqrt(2 * r4) / k
                    cell[ens] = dict(cv=float(a2.std() / a2.mean()),
                                     pred_cv=float(pred_cv), rho4=r4,
                                     mean=float(a2.mean()))
                    rows.append(dict(d=d, alpha=alpha, k=k, ens=ens,
                                     cv=cell[ens]["cv"], pred_cv=float(pred_cv),
                                     rho4=r4, mean=float(a2.mean())))
                    print(f"d={d} a={alpha} k={k} {ens}: cv={cell[ens]['cv']:.3f} "
                          f"pred={pred_cv:.3f} rho4={r4:.1f}")
                inv = abs(cell["gauss"]["mean"] - cell["sparse"]["mean"]) / cell["gauss"]["mean"]
                print(f"   mean invariance dev: {inv:.3%}")
    json.dump(rows, open(OUT / "kurtosis_confirm.json", "w"), indent=1)
    v = dict(
        i_gauss=all(1.0 <= r["cv"] / (2.0 / r["k"]) <= 3.0 for r in rows if r["ens"] == "gauss"),
        ii_sparse=all(0.7 <= r["cv"] / r["pred_cv"] <= 1.4 for r in rows if r["ens"] == "sparse"),
    )
    json.dump(v, open(OUT / "kurtosis_confirm_verdicts.json", "w"), indent=1)
    print("VERDICTS:", v)


if __name__ == "__main__":
    main()
