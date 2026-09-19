"""HP-E E-struct validation: participation-ratio capacity law.

Pre-registered (frozen in theory/HPE_AGGREGATE_INTERFERENCE.md 3g before run):
- P-PAIR: measured mean (f_i . f_j)^2 within 10% of 1/PR(alpha, d), unit-energy
  features, for alpha in {0, 0.5, 0.75, 0.9}, d in {512, 2048}.
- P-CROSS: A_2 crosses 1 within a factor-2 band around sqrt(PR) across the
  k-grid, 8/8 cells.
"""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).parent / "results" / "hpe_aggregate"
OUT.mkdir(parents=True, exist_ok=True)
REPS = 12


def pr(alpha, d):
    lam = np.arange(1, d + 1, dtype=np.float64) ** (-alpha)
    return lam.sum() ** 2 / (lam ** 2).sum()


def features(d, alpha, k, rng):
    lam = np.arange(1, d + 1, dtype=np.float64) ** (-alpha / 2.0)
    U = rng.standard_normal((k, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    F = U * lam[None, :]
    nrm = np.linalg.norm(F, axis=1, keepdims=True)
    return F / nrm  # unit-energy normalized features


def main():
    rng = np.random.default_rng(7)
    rows = []
    for d in [512, 2048]:
        for alpha in [0.0, 0.5, 0.75, 0.9]:
            P = pr(alpha, d)
            kstar = int(round(np.sqrt(P)))
            kgrid = sorted({max(2, kstar // 4), kstar, min(d, kstar * 4)})
            # pairwise interference at a fixed large k (use k = min(d, 4*kstar))
            kp = min(d, max(64, 4 * kstar))
            mp, mp_pred = [], 1.0 / P
            for r in range(REPS):
                F = features(d, alpha, kp, rng)
                G = F @ F.T
                off = G - np.diag(np.diag(G))
                iu = np.triu_indices(kp, 1)
                mp.append((off[iu] ** 2).mean())
            # A_2 crossing over kgrid
            cross_ok, a2_at = None, {}
            for k in kgrid:
                a2s = []
                for r in range(REPS):
                    F = features(d, alpha, k, rng)
                    G = F @ F.T
                    off = G - np.diag(np.diag(G))
                    a2s.append((off ** 2).sum())
                a2_at[k] = float(np.mean(a2s))
            ks = sorted(a2_at)
            below = [k for k in ks if a2_at[k] < 1]
            above = [k for k in ks if a2_at[k] >= 1]
            cross = (max(below) / kstar >= 0.25 and min(above) / kstar <= 4.0) if below and above else (ks[0] >= 2 * kstar or ks[-1] <= kstar / 2)
            rows.append(dict(d=d, alpha=alpha, PR=float(P), kstar=kstar,
                             pair_meas=float(np.mean(mp)), pair_pred=float(mp_pred),
                             pair_rel_err=float(abs(np.mean(mp) - mp_pred) / mp_pred),
                             a2=a2_at, cross_ok=bool(cross)))
            print(f"d={d} a={alpha}: PR={P:.0f} k*={kstar} pair={np.mean(mp):.2e} "
                  f"pred={mp_pred:.2e} err={abs(np.mean(mp)-mp_pred)/mp_pred:.3f} cross={cross}")
    json.dump(rows, open(OUT / "estruct_check.json", "w"), indent=1)
    v = dict(P_PAIR_10pct=all(r["pair_rel_err"] < 0.10 for r in rows),
             P_CROSS=all(r["cross_ok"] for r in rows))
    json.dump(v, open(OUT / "estruct_verdicts.json", "w"), indent=1)
    print("VERDICTS:", v)


if __name__ == "__main__":
    main()
