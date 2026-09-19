"""HP-E E5 anchor validation: empirically check the PROVED quadratic anchor and
the t=3 ladder ratio. Pre-registered predictions (frozen before running):

P1: For M = I_d, unit-sphere features: E[A_2] = k(k-1)/d EXACTLY (theory);
    simulation means within 5% for d >= 128.
P2: Concentration: std(A_2)/mean(A_2) <= 5/sqrt(k) (variance route predicts
    O(1/sqrt(kd)); looser bound registered to be safe).
P3: Random tensor arms (||T||_F = ||M||_F = 1): mean(pair^2) * d in [0.5, 2];
    mean(triple^2) * d^2 in [0.5, 2]  (ladder ratio test).
P4: Absolute-error crossing: A_2 crosses 1 near k = sqrt(d) (within the
    tested k grid: A_2 < 1 at k = sqrt(d)/4; A_2 > 1 at k = 4*sqrt(d)).
"""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).parent / "results" / "hpe_aggregate"
OUT.mkdir(parents=True, exist_ok=True)
REPS = 20


def unit_sphere(d, k, rng):
    F = rng.standard_normal((k, d))
    return F / np.linalg.norm(F, axis=1, keepdims=True)


def main():
    rng = np.random.default_rng(42)
    res = {"identity_anchor": [], "random_pair": [], "random_triple": []}
    for d in [128, 512, 2048]:
        for k in sorted({max(4, int(np.sqrt(d) / 4)), int(np.sqrt(d)), 4 * int(np.sqrt(d))}):
            a2, sig = [], []
            for r in range(REPS):
                F = unit_sphere(d, k, rng)
                G = F @ F.T
                off = G - np.diag(np.diag(G))
                a2.append((off ** 2).sum())
                sig.append((np.diag(G) ** 2).sum())
            a2 = np.array(a2)
            res["identity_anchor"].append(dict(
                d=d, k=k, mean_A2=float(a2.mean()), std_A2=float(a2.std()),
                pred=float(k * (k - 1) / d), signal_mean=float(np.mean(sig)),
                rel_err=float(abs(a2.mean() - k * (k - 1) / d) / (k * (k - 1) / d)),
                cv=float(a2.std() / a2.mean())))
            print(f"d={d} k={k}: A2={a2.mean():.3f} pred={k*(k-1)/d:.3f} "
                  f"relerr={abs(a2.mean()-k*(k-1)/d)/(k*(k-1)/d):.4f} cv={a2.std()/a2.mean():.4f}")
    # random-tensor arms at d in {128, 512}
    for d in [128, 512]:
        M = rng.standard_normal((d, d)); M /= np.linalg.norm(M)
        T = rng.standard_normal((d, d, d)); T /= np.linalg.norm(T)
        k = 64
        F = unit_sphere(d, k, rng)
        pair, triple = [], []
        for r in range(400):
            i, j = rng.integers(0, k, 2)
            while j == i:
                j = rng.integers(0, k)
            pair.append((F[i] @ M @ F[j]) ** 2)
            a, b, c = rng.choice(k, 3, replace=False)
            triple.append(float(np.einsum('a,b,c,abc->', F[a], F[b], F[c], T)) ** 2)
        res["random_pair"].append(dict(d=d, mean_pair2=float(np.mean(pair)),
                                       scaled=float(np.mean(pair) * d)))
        res["random_triple"].append(dict(d=d, mean_triple2=float(np.mean(triple)),
                                         scaled=float(np.mean(triple) * d * d)))
        print(f"d={d}: pair^2*d={np.mean(pair)*d:.3f}  triple^2*d^2={np.mean(triple)*d*d:.3f}")
    json.dump(res, open(OUT / "anchor_check.json", "w"), indent=1)

    ia = res["identity_anchor"]
    verdicts = {
        "P1_pass_5pct": all(x["rel_err"] < 0.05 for x in ia if x["d"] >= 128),
        "P2_pass": all(x["cv"] <= 5 / np.sqrt(x["k"]) for x in ia),
        "P3_pass": all(0.5 <= x["scaled"] <= 2 for x in res["random_pair"] + res["random_triple"]),
        "P4_pass": all(x["mean_A2"] < 1 for x in ia if x["k"] <= np.sqrt(x["d"]) / 2) and
                   all(x["mean_A2"] > 1 for x in ia if x["k"] >= 2 * np.sqrt(x["d"])),
    }
    json.dump(verdicts, open(OUT / "anchor_verdicts.json", "w"), indent=1)
    print("VERDICTS:", verdicts)


if __name__ == "__main__":
    main()
