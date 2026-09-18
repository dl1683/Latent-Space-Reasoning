"""HP-1: Equilibrium or transient? Spectral law of Zipf-trained superposition codes.

Implements the frozen pre-registration in theory/HD_POWER_LAW_ALLOCATION.md.
CPU numpy only (float64). One canonical runner; phases explore the frozen grid.

Usage:
  python experiments/run_hp1_spectral_law.py --phase lrcal   # lr calibration (loss-based)
  python experiments/run_hp1_spectral_law.py --phase main    # QSAE grid, 3 seeds
  python experiments/run_hp1_spectral_law.py --phase xe      # XE replication arm
  python experiments/run_hp1_spectral_law.py --phase nulls   # N0/N1/N2
  python experiments/run_hp1_spectral_law.py --phase smoke   # instrument check (not in grid)
"""
import argparse, json, time
from pathlib import Path
import numpy as np

OUT = Path(__file__).parent / "results" / "hp1_spectral_law"
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-12


def zipf_p(n, s):
    k = np.arange(1, n + 1, dtype=np.float64)
    w = k ** (-s) if s > 0 else np.ones(n)
    return w / w.sum()


def spectrum(W):
    S = W @ W.T
    lam = np.linalg.eigvalsh(S)[::-1]
    return np.maximum(lam, EPS)


def pl_fit(lam, frac=0.9):
    """OLS of log lam_k vs log k over k=1..floor(frac*d). Returns (alpha, R2)."""
    m = max(3, int(frac * len(lam)))
    x = np.log(np.arange(1, m + 1))
    y = np.log(lam[:m])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (res[0] / ss) if len(res) and ss > 0 else np.nan
    return -coef[0], r2


def norm_profile_fit(x, s, n):
    """Fit x_(k) = c0 - c1*H*k^s on the unclipped (x>0) region, usage-ranked."""
    k = np.arange(1, n + 1, dtype=np.float64)
    H = (k ** (-s)).sum() if s > 0 else float(n)
    mask = x > 0
    if mask.sum() < 5:
        return dict(c0=np.nan, c1H=np.nan, r2=np.nan, n_unclipped=int(mask.sum()))
    X = np.vstack([np.ones(mask.sum()), -(H * (k ** s))[mask]]).T
    coef, res, *_ = np.linalg.lstsq(X, x[mask], rcond=None)
    ss = ((x[mask] - x[mask].mean()) ** 2).sum()
    r2 = 1.0 - (res[0] / ss) if len(res) and ss > 0 else np.nan
    return dict(c0=coef[0], c1H=coef[1], r2=r2, n_unclipped=int(mask.sum()))


def mf_spectrum(x_profile, rng):
    """Spectrum from random directions with the given norm profile."""
    n = len(x_profile)
    d = int((x_profile > 0).sum())
    d = max(d, 2)
    keep = np.sort(x_profile)[::-1][:d]  # top-d norms get directions
    W = rng.standard_normal((d, n))
    W = W / np.linalg.norm(W, axis=0, keepdims=True)
    W = W * np.sqrt(np.maximum(x_profile, 0))[None, :]
    return spectrum(W)


# ---------------- QSAE ----------------

def qsae_loss_grad(W, p):
    S = W @ W.T
    Wp = W * p[None, :]
    A = Wp @ W.T
    loss = np.trace(A @ S) - 2.0 * np.trace(A) + p.sum()
    grad = 2.0 * (S @ Wp) + 2.0 * (A @ W) - 4.0 * Wp
    return loss, grad


def run_qsae(n, d, s, seed, lr, checkpoints, Tmax, p=None, record_from=0):
    rng = np.random.default_rng(seed)
    p = zipf_p(n, s) if p is None else p
    W = 0.05 * rng.standard_normal((d, n)) / np.sqrt(d)
    W[:, 0] = np.abs(W[:, 0])  # break sign symmetry
    ck = sorted(checkpoints)
    recs, losses = [], []
    t0 = time.time()
    for t in range(1, Tmax + 1):
        loss, grad = qsae_loss_grad(W, p)
        W = W - lr * grad
        losses.append(loss)
        if t in ck and t >= record_from:
            lam = spectrum(W)
            alpha, r2 = pl_fit(lam)
            x = (W ** 2).sum(axis=0)
            recs.append(dict(t=t, loss=loss, alpha=alpha, r2_pl=r2,
                             cond=lam[0] / max(lam[-1], EPS),
                             norm_fit=norm_profile_fit(x, s, n),
                             norms=x.tolist(), lam=lam.tolist()))
    slope = (losses[-1] - losses[-1000]) / 1000.0 if len(losses) > 1000 else np.nan
    return dict(n=n, d=d, s=s, seed=seed, lr=lr, Tmax=Tmax, converged_slope=slope,
                final_loss=losses[-1], records=recs, wall_s=time.time() - t0), W


def alpha_pred_curve(profile_star, p, checkpoints, kappa):
    """P3 closed form: x_i(t)=x*_i(1-e^{-kappa p_i t}); proxy spectrum = sorted + mean."""
    preds = {}
    for t in checkpoints:
        x = profile_star * (1.0 - np.exp(-kappa * p * t))
        proxy = np.sort(x)[::-1] + max(x.sum() / len(profile_star), EPS)
        preds[t] = pl_fit(np.maximum(proxy, EPS))[0]
    return preds


def calibrate_kappa(profile_star, p, checkpoints, alpha_at_t1):
    best, bk = 1e9, None
    for kappa in np.logspace(-3, 3, 240):
        a = alpha_pred_curve(profile_star, p, [checkpoints[0]], kappa)[checkpoints[0]]
        if abs(a - alpha_at_t1) < best:
            best, bk = abs(a - alpha_at_t1), kappa
    return bk


# ---------------- XE (tied softmax, entry-101 replication) ----------------

def run_xe(n, d, s, seed, lr, checkpoints, Tmax, batch=256):
    rng = np.random.default_rng(seed)
    p = zipf_p(n, s)
    W = 0.01 * rng.standard_normal((d, n))
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    b1, b2, adeps = 0.9, 0.999, 1e-8
    recs = []
    t0 = time.time()
    for t in range(1, Tmax + 1):
        I = rng.choice(n, size=batch, p=p)
        J = rng.choice(n, size=batch, p=p)
        H = W[:, I]                          # d x B
        Z = W.T @ H                          # n x B
        Z = Z - Z.max(axis=0, keepdims=True)
        P = np.exp(Z); P /= P.sum(axis=0, keepdims=True)
        P[J, np.arange(batch)] -= 1.0
        P /= batch                           # dL/dZ
        gW = H @ P.T                         # d x n
        contrib = W @ P                      # d x B (gradient into context cols)
        np.add.at(gW, (slice(None), I), contrib)
        mW = b1 * mW + (1 - b1) * gW
        vW = b2 * vW + (1 - b2) * gW * gW
        mh = mW / (1 - b1 ** t); vh = vW / (1 - b2 ** t)
        W -= lr * mh / (np.sqrt(vh) + adeps)
        if t in sorted(checkpoints):
            # eval loss on fixed probe batch
            rngE = np.random.default_rng(12345)
            Ie = rngE.choice(n, size=1024, p=p); Je = rngE.choice(n, size=1024, p=p)
            Ze = W.T @ W[:, Ie]; Ze -= Ze.max(axis=0, keepdims=True)
            Pe = np.exp(Ze); Pe /= Pe.sum(axis=0, keepdims=True)
            loss = -np.log(Pe[Je, np.arange(1024)] + EPS).mean()
            lam = spectrum(W)
            alpha, r2 = pl_fit(lam)
            recs.append(dict(t=t, loss=float(loss), alpha=alpha, r2_pl=r2,
                             cond=lam[0] / max(lam[-1], EPS), lam=lam.tolist()))
    return dict(n=n, d=d, s=s, seed=seed, lr=lr, Tmax=Tmax, records=recs,
                wall_s=time.time() - t0)


# ---------------- nulls ----------------

def null_n0_lbfgs(n, d, s, seed, gd_lam):
    from scipy.optimize import minimize
    p = zipf_p(n, s)
    rng = np.random.default_rng(seed)
    out = []
    for k in range(3):
        W0 = 0.05 * rng.standard_normal((d, n)) / np.sqrt(d)
        f = lambda W: qsae_loss_grad(W.reshape(d, n), p)[0]
        g = lambda W: qsae_loss_grad(W.reshape(d, n), p)[1].ravel()
        r = minimize(f, W0.ravel(), jac=g, method="L-BFGS-B",
                     options=dict(maxiter=20000, maxfun=40000, ftol=1e-14, gtol=1e-10))
        lam = spectrum(r.x.reshape(d, n))
        out.append(dict(seed=k, success=bool(r.success), loss=float(r.fun),
                        alpha=pl_fit(lam)[0], lam=lam.tolist()))
    m = min(len(gd_lam), min(len(o["lam"]) for o in out))
    dev = max(abs(a - b) / max(abs(b), EPS)
              for o in out for a, b in zip(gd_lam[:m], o["lam"][:m]))
    return dict(runs=out, max_rel_dev_vs_gd=dev)


def null_n1_random_dirs(x_profile, seed, s):
    rng = np.random.default_rng(seed)
    lam_mf = mf_spectrum(np.array(x_profile), rng)
    return dict(alpha_mf=pl_fit(lam_mf)[0], r2_mf=pl_fit(lam_mf)[1],
                lam_mf=lam_mf.tolist(), lam_mf_desc=lam_mf.tolist())


def r2_identity(log_actual, log_model):
    ss = ((log_actual - log_actual.mean()) ** 2).sum()
    return 1.0 - ((log_actual - log_model) ** 2).sum() / ss if ss > 0 else np.nan


# ---------------- phases ----------------

GRID = [dict(n=256, d=64, s=0.5), dict(n=256, d=64, s=1.0),
        dict(n=1024, d=64, s=0.5), dict(n=1024, d=64, s=1.0),
        dict(n=1024, d=256, s=0.5), dict(n=1024, d=256, s=1.0)]
CK = [1000, 10000, 50000, 100000, 200000]


def phase_lrcal():
    arm = dict(n=1024, d=64, s=1.0)
    res = {}
    for lr in [0.002, 0.01, 0.05, 0.2]:
        r, _ = run_qsae(**arm, seed=0, lr=lr, checkpoints=[2000], Tmax=2000)
        res[lr] = dict(final_loss=r["final_loss"], finite=bool(np.isfinite(r["final_loss"])))
        print(lr, res[lr])
    ok = {k: v for k, v in res.items() if v["finite"] and v["final_loss"] < 1.0}
    best = min(ok, key=ok.get) if ok else 0.01
    json.dump(dict(results=res, chosen_lr=best), open(OUT / "lr_calibration.json", "w"), indent=1)
    print("chosen:", best)


def phase_main(lr):
    for arm in GRID:
        for seed in [0, 1, 2]:
            tag = f"qsae_n{arm['n']}_d{arm['d']}_s{arm['s']}_seed{seed}"
            out, W = run_qsae(**arm, seed=seed, lr=lr, checkpoints=CK, Tmax=200000)
            # P-C: kappa calibrated at FIRST checkpoint only, later = prediction
            end = out["records"][-1]
            star = np.array(end["norms"])
            prof = np.maximum(star, 0)  # endpoint norms as x* for the P-C calibration
            kap = calibrate_kappa(prof, zipf_p(arm["n"], arm["s"]), CK, out["records"][0]["alpha"])
            out["kappa_cal"] = kap
            out["alpha_pred"] = alpha_pred_curve(prof, zipf_p(arm["n"], arm["s"]), CK, kap)
            # P-B: mean-field spectrum comparison at endpoint
            rng = np.random.default_rng(seed + 100)
            lam_act = np.array(end["lam"])
            r2mf = []
            for _ in range(3):
                lam_mf = mf_spectrum(prof, rng)
                m = min(len(lam_act), len(lam_mf))
                r2mf.append(r2_identity(np.log(lam_act[:m]), np.log(np.maximum(lam_mf[:m], EPS))))
            out["r2_mf_endpoint"] = float(np.mean(r2mf))
            out["delta_r2_PB"] = float(end["r2_pl"] - out["r2_mf_endpoint"])
            json.dump(out, open(OUT / f"{tag}.json", "w"))
            print(tag, "alpha_end", round(end["alpha"], 3), "dR2", round(out["delta_r2_PB"], 3))


def phase_xe():
    for seed in [0, 1, 2]:
        out = run_xe(1024, 64, 1.0, seed, 3e-3, [3000, 12000, 48000, 192000], 192000)
        json.dump(out, open(OUT / f"xe_n1024_d64_s1_seed{seed}.json", "w"))
        print("xe seed", seed, [(r["t"], round(r["alpha"], 3)) for r in out["records"]])


def phase_nulls(lr):
    # N2: uniform usage
    for seed in [0, 1, 2]:
        out, _ = run_qsae(1024, 64, 0.0, seed, lr, CK, 200000)
        json.dump(out, open(OUT / f"null_n2_uniform_seed{seed}.json", "w"))
        print("N2 seed", seed, "alpha_end", round(out["records"][-1]["alpha"], 3))
    # N0: L-BFGS multi-init vs GD endpoint
    gd, Wgd = run_qsae(1024, 64, 1.0, 0, lr, CK, 200000)
    lam_gd = np.array(gd["records"][-1]["lam"])
    n0 = null_n0_lbfgs(1024, 64, 1.0, 7, lam_gd)
    json.dump(n0, open(OUT / "null_n0_lbfgs.json", "w"))
    print("N0 max_rel_dev", n0["max_rel_dev_vs_gd"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["lrcal", "main", "xe", "nulls", "smoke"])
    ap.add_argument("--lr", type=float, default=None)
    a = ap.parse_args()
    if a.phase == "lrcal":
        phase_lrcal()
    elif a.phase == "main":
        phase_main(a.lr or json.load(open(OUT / "lr_calibration.json"))["chosen_lr"])
    elif a.phase == "xe":
        phase_xe()
    elif a.phase == "nulls":
        phase_nulls(a.lr or json.load(open(OUT / "lr_calibration.json"))["chosen_lr"])
    elif a.phase == "smoke":
        # instrument check only: finite-difference gradient + tiny run NOT in grid
        rng = np.random.default_rng(0)
        W = rng.standard_normal((8, 20)); p = zipf_p(20, 1.0)
        _, g = qsae_loss_grad(W, p)
        num = np.zeros_like(W); h = 1e-6
        for i in range(8):
            for j in range(20):
                Wp, Wm = W.copy(), W.copy()
                Wp[i, j] += h; Wm[i, j] -= h
                num[i, j] = (qsae_loss_grad(Wp, p)[0] - qsae_loss_grad(Wm, p)[0]) / (2 * h)
        err = np.abs(num - g).max() / (np.abs(g).max() + EPS)
        print("grad check max rel err:", err)
        r, _ = run_qsae(64, 16, 1.0, 0, 0.05, [100, 300], 300)
        print("tiny run alpha:", [round(x["alpha"], 3) for x in r["records"]])
        assert err < 1e-4, "gradient check FAILED"
        print("SMOKE OK")
