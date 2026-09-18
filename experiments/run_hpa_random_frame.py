"""HP-A: analytic + Monte-Carlo foundation cycle (v2 pre-registration, theory/HD_POWER_LAW_ALLOCATION.md).

Phases: a1 (mean-field fixed point vs L-BFGS optima), a2 (random-frame
spectral map + estimator validity), a3 (exact-flow pilot + transient curves).
CPU numpy float64, no SGD grid runs. Reuses QSAE primitives from the HP runner.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from hp_spectral_lib import zipf_p, qsae_loss_grad, spectrum, rand_frame

OUT = Path(__file__).parent / "results" / "hpa_random_frame"
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-14


def mf_fixed_point(n, d, s, iters=40000, tol=1e-13):
    p = zipf_p(n, s)
    X, M = 0.5 * d, 0.5
    it = 0
    for it in range(iters):
        x = (d / (d - 1.0)) * np.maximum(0.0, 1.0 - (X + M / p) / (2.0 * d))
        Xn, Mn = x.sum(), (p * x).sum()
        if abs(Xn - X) < tol and abs(Mn - M) < tol:
            X, M = Xn, Mn
            break
        X, M = 0.7 * X + 0.3 * Xn, 0.7 * M + 0.3 * Mn
    x = (d / (d - 1.0)) * np.maximum(0.0, 1.0 - (X + M / p) / (2.0 * d))
    resid = max(abs(x.sum() - X), abs((p * x).sum() - M))
    return dict(x=x, X=X, M=M, k0=int((x > 0).sum()), iters=it, residual=resid)


def fit_window(lam, frac):
    m = max(3, int(frac * len(lam)))
    k = np.arange(1, m + 1, dtype=np.float64)
    y = np.log(np.maximum(lam[:m], EPS))
    lk = np.log(k)
    A = np.vstack([lk, np.ones_like(lk)]).T
    c1, *_ = np.linalg.lstsq(A, y, rcond=None)
    r1 = 1 - ((y - A @ c1) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    B = np.vstack([np.ones_like(lk), lk, lk ** 2]).T
    c2, *_ = np.linalg.lstsq(B, y, rcond=None)
    r2 = 1 - ((y - B @ c2) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(alpha=-c1[0], r2_pl=r1, r2_quad=r2, dR2=r1 - r2, m=m)


def battery(lam):
    return {w: fit_window(lam, f) for w, f in
            [("top50", 0.5), ("top90", 0.9), ("full", 1.0)]}


# ---------------- A1 ----------------

def phase_a1():
    res = {}
    for s in [0.5, 1.0]:
        for (n, d) in [(64, 16), (256, 64), (512, 128)]:
            fp = mf_fixed_point(n, d, s)
            p = zipf_p(n, s)
            inits = []
            for seed in range(5):
                rng = np.random.default_rng(1000 + seed)
                W0 = 0.05 * rng.standard_normal((d, n)) / np.sqrt(d)
                f = lambda W: qsae_loss_grad(W.reshape(d, n), p)[0]
                g = lambda W: qsae_loss_grad(W.reshape(d, n), p)[1].ravel()
                r = minimize(f, W0.ravel(), jac=g, method="L-BFGS-B",
                             options=dict(maxiter=20000, maxfun=40000, ftol=1e-15, gtol=1e-10))
                Wopt = r.x.reshape(d, n)
                xm = (Wopt ** 2).sum(axis=0)
                k = np.arange(1, n + 1, dtype=np.float64)
                H = (k ** (-s)).sum()
                mask = xm > 1e-8 * xm.max()
                X = np.vstack([np.ones(mask.sum()), -(H * (k ** s))[mask]]).T
                coef, *_ = np.linalg.lstsq(X, xm[mask], rcond=None)
                ss = ((xm[mask] - xm[mask].mean()) ** 2).sum()
                r2 = 1 - (((X @ coef - xm[mask]) ** 2).sum() / ss if ss > 0 else np.nan)
                r2_id = 1 - ((xm[mask] - fp["x"][mask]) ** 2).sum() / ((xm - xm.mean()) ** 2).sum()
                lam = spectrum(Wopt)
                inits.append(dict(seed=seed, loss=float(r.fun), gnorm=float(np.abs(g(r.x)).max()),
                                  norm_fit_r2=float(r2), c0=float(coef[0]), c1H=float(coef[1]),
                                  r2_identity_vs_theory=float(r2_id),
                                  alpha=battery(lam)["top90"]["alpha"],
                                  k0_fit=int(mask.sum())))
            med = lambda key: float(np.median([i[key] for i in inits]))
            res[f"s{s}_n{n}_d{d}"] = dict(fixed_point=dict(k0=fp["k0"], X=fp["X"], M=fp["M"],
                                                           residual=fp["residual"]),
                                          median=dict(norm_fit_r2=med("norm_fit_r2"),
                                                      r2_identity_vs_theory=med("r2_identity_vs_theory"),
                                                      alpha=med("alpha"), k0_fit=med("k0_fit")),
                                          runs=inits)
            print(f"A1 s={s} n={n} d={d}: fitR2={med('norm_fit_r2'):.3f} "
                  f"idR2={med('r2_identity_vs_theory'):.3f} k0th={fp['k0']} k0fit={med('k0_fit'):.0f}")
    json.dump(res, open(OUT / "a1_meanfield.json", "w"))
    return res


# ---------------- A2 ----------------

def phase_a2(a1=None):
    a1 = a1 or json.load(open(OUT / "a1_meanfield.json"))
    res = {}
    for s in [0.5, 1.0]:
        for gamma in [4, 16]:
            for d in [32, 64, 128, 256, 512, 1024]:
                n = gamma * d
                profiles = {}
                fp = mf_fixed_point(n, d, s)
                profiles["bulge"] = fp["x"]
                for beta in [0.25, 0.5, 1.0]:
                    k = np.arange(1, n + 1, dtype=np.float64)
                    profiles[f"pl{beta}"] = k ** (-beta)
                profiles["uniform"] = np.ones(n)
                tag = f"s{s}_g{gamma}_d{d}"
                res[tag] = {}
                for pname, prof in profiles.items():
                    rows = []
                    for rep in range(20):
                        rng = np.random.default_rng(hash((tag, pname, rep)) % (2 ** 32))
                        lam = rand_frame(prof, d, rng)
                        b = battery(lam)
                        rows.append({w: dict(alpha=v["alpha"], r2_pl=v["r2_pl"], dR2=v["dR2"])
                                     for w, v in b.items()})
                    med = {}
                    for w in ["top50", "top90", "full"]:
                        med[w] = dict(alpha=float(np.median([r[w]["alpha"] for r in rows])),
                                      r2_pl=float(np.median([r[w]["r2_pl"] for r in rows])),
                                      dR2=float(np.median([r[w]["dR2"] for r in rows])))
                    res[tag][pname] = med
                line = " ".join(f"{p}:{res[tag][p]['top90']['alpha']:.2f}" for p in res[tag])
                print(f"A2 {tag}  {line}")
    json.dump(res, open(OUT / "a2_spectral_map.json", "w"))
    return res


# ---------------- A3 ----------------

def alpha_of_norms(x, d, seed, reps=5):
    vals = []
    for rep in range(reps):
        rng = np.random.default_rng(seed * 977 + rep)
        lam = rand_frame(x, d, rng)
        vals.append(fit_window(lam, 0.9)["alpha"])
    return float(np.median(vals))


def phase_a3():
    n, d, s = 64, 16, 1.0
    p = zipf_p(n, s)
    # A1 theorem endpoint (sparse-exact), NOT the refuted mean-field bulge
    x_star_se = np.zeros(n); x_star_se[:d] = 1.0
    pilot = {}
    for seed in [90, 91, 92]:
        rng = np.random.default_rng(seed)
        W = 0.05 * rng.standard_normal((d, n)) / np.sqrt(d)
        W[:, 0] = np.abs(W[:, 0])
        lr, traj = 2e-3, []
        for t in range(1, 4001):
            _, grad = qsae_loss_grad(W, p)
            W = W - lr * grad
            if t % 250 == 0:
                traj.append((t, fit_window(spectrum(W), 0.9)["alpha"]))
        pilot[seed] = traj
    t1 = 250
    a_pilot = float(np.median([dict(traj)[t1] for traj in pilot.values()]))
    # calibrate rho on first checkpoint only
    rng0 = np.random.default_rng(90)
    x0 = ((0.05 * rng0.standard_normal((d, n)) / np.sqrt(d)) ** 2).sum(axis=0)
    x_star = x_star_se
    best, rho = 1e9, None
    for r in np.logspace(-4, 1, 260):
        xt = x_star / (1 + np.maximum(x_star / np.maximum(x0, EPS) - 1, 0) * np.exp(-r * p * t1))
        if abs(alpha_of_norms(xt, d, 7, 3) - a_pilot) < best:
            best, rho = abs(alpha_of_norms(xt, d, 7, 3) - a_pilot), r
    # predictions at (1024,64,s=1) for HP-B
    n2, d2 = 1024, 64
    p2 = zipf_p(n2, s)
    x_star2 = np.zeros(n2); x_star2[:d2] = 1.0  # A1 theorem endpoint
    rng = np.random.default_rng(5)
    x0b = (0.05 ** 2) * (rng.standard_normal((d2, n2)) ** 2).sum(axis=0) / d2
    preds = {}
    for t in [1000, 10000, 50000, 100000, 200000]:
        xt = x_star2 / (1 + np.maximum(x_star2 / np.maximum(x0b, EPS) - 1, 0) * np.exp(-rho * p2 * t))
        preds[t] = alpha_of_norms(xt, d2, 11)
    out = dict(pilot={str(sd): tr for sd, tr in pilot.items()},
               rho_cal=float(rho), rho_abs_err=float(best),
               alpha_pred_1024_64_s1=preds, x_star="sparse_exact_topd")
    json.dump(out, open(OUT / "a3_transient.json", "w"))
    print("A3 rho", rho, "preds", preds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["a1", "a2", "a3", "all"], default="all")
    a = ap.parse_args()
    t0 = time.time()
    if a.phase in ("a1", "all"):
        a1 = phase_a1()
    if a.phase in ("a2", "all"):
        phase_a2()
    if a.phase in ("a3", "all"):
        phase_a3()
    print(f"wall {time.time()-t0:.1f}s")
