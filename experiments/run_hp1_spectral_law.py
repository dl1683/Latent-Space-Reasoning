"""HP-B: optimizer cycle (v2 repaired design, theory/HD_POWER_LAW_ALLOCATION.md §5).

Phases:
  pilot   lr calibration on INDEPENDENT seeds 90-92 (loss-based only)
  main    QSAE grid: s in {0.5,1.0}, (n,d) in {(256,64),(1024,64),(1024,256)}
          plus (4096,64) s=1 only; 3 seeds; convergence gate + frozen extension
          schedule; N1-t at every checkpoint; P-C' scored on non-calibration
          checkpoints against A3's frozen rho.
  xe      entry-101 replication with the repaired equilibrium gate (capped
          extensions; wall-clock bound logged).
  nulls   N0 L-BFGS multi-init, N1 endpoint randomization, N2 uniform usage.

Small arms run CPU float64 numpy. The (1024,256) and (4096,64) arms run
torch CUDA float32 when available (engineering backend change, logged in each
JSON as "device"; measurement tolerances dwarf float32 roundoff).
"""
import argparse, json, time
from pathlib import Path
import numpy as np

from hp_spectral_lib import (zipf_p, qsae_loss_grad, spectrum, battery, fit_window,
                             rand_frame, n1t_alpha, mf_fixed_point, alpha_pred, EPS)

OUT = Path(__file__).parent / "results" / "hp1_spectral_law"
OUT.mkdir(parents=True, exist_ok=True)
A3 = OUT.parent / "hpa_random_frame" / "a3_transient.json"

GRID = [dict(n=256, d=64, s=0.5), dict(n=256, d=64, s=1.0),
        dict(n=1024, d=64, s=0.5), dict(n=1024, d=64, s=1.0),
        dict(n=1024, d=256, s=0.5), dict(n=1024, d=256, s=1.0),
        dict(n=4096, d=64, s=1.0)]
CK = [1000, 10000, 50000, 100000, 200000]
PC_CK = [10000, 50000, 100000, 200000]  # 1e3 is burn-in, excluded from P-C' scoring


def torch_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def run_qsae(n, d, s, seed, lr, checkpoints, Tmax, device="cpu",
             wall_cap_s=7200, max_ext=3):
    t_start = time.time()
    if device == "cuda":
        import torch
        dev = "cuda"
        p_t = torch.tensor(zipf_p(n, s), device=dev, dtype=torch.float32)

        def step(W, record=False):
            S = W @ W.T
            Wp = W * p_t[None, :]
            A = Wp @ W.T
            loss = torch.trace(A @ S) - 2 * torch.trace(A) + float(p_t.sum().item())
            grad = 2.0 * (S @ Wp) + 2.0 * (A @ W) - 4.0 * Wp
            if record:
                gn = float(grad.abs().max().item())
                Wn = W.detach().cpu().numpy().astype(np.float64)
                return float(loss.item()), gn, Wn
            return W - lr * grad, None, None

        def new_W():
            g = torch.Generator(device=dev).manual_seed(seed)
            W = 0.05 * torch.randn((d, n), generator=g, device=dev) / np.sqrt(d)
            W[:, 0] = W[:, 0].abs()
            return W
    else:
        p = zipf_p(n, s)
        rng = np.random.default_rng(seed)

        def new_W():
            W = 0.05 * rng.standard_normal((d, n)) / np.sqrt(d)
            W[:, 0] = np.abs(W[:, 0])
            return W

        def step(W, record=False):
            loss, grad = qsae_loss_grad(W, p)
            if record:
                return float(loss), float(np.abs(grad).max()), W.copy()
            return W - lr * grad, None, None

    W = new_W()
    _, gnorm0, _ = step(W, record=True)
    T, ext = Tmax, 0
    recs, losses, gnorms = [], [], []
    rng_n1t = np.random.default_rng(seed + 500)
    while True:
        for t in range(1, T + 1):
            if device == "cuda":
                if t in checkpoints:
                    loss, gn, Wn = step(W, record=True)
                    recs.append(_record(Wn, t, loss, gn, d, s, n, rng_n1t))
                else:
                    W, _, _ = step(W)
            else:
                loss, grad = qsae_loss_grad(W, p)
                if t in checkpoints:
                    gn = float(np.abs(grad).max())
                    recs.append(_record(W.copy(), t, float(loss), gn, d, s, n, rng_n1t))
                W = W - lr * grad
            losses.append(loss)
            gnorms.append(gn if t in checkpoints else None)
        last = recs[-1]
        prev = recs[-2] if len(recs) >= 2 else last
        slope = (losses[-1] - losses[-1000]) / 1000.0 if len(losses) > 1000 else np.inf
        conv = (abs(last["bat"]["top90"]["alpha"] - prev["bat"]["top90"]["alpha"]) <= 0.05
                and slope < 1e-5 and last["gnorm"] < 1e-3 * gnorm0)
        if conv or ext >= max_ext or (time.time() - t_start) > wall_cap_s:
            return dict(n=n, d=d, s=s, seed=seed, lr=lr, device=device, Tmax=T,
                        extensions_used=ext, converged=bool(conv),
                        capped_by_wall=(time.time() - t_start) > wall_cap_s,
                        gnorm0=gnorm0, final_loss_slope=slope, records=recs,
                        wall_s=time.time() - t_start)
        T *= 4
        ext += 1
        checkpoints = sorted(set(checkpoints + [T // 4, T // 2]))


def _record(W, t, loss, gn, d, s, n, rng_n1t):
    lam = spectrum(W)
    bat = battery(lam)
    x = (W ** 2).sum(axis=0)
    return dict(t=t, loss=loss, gnorm=gn, bat={k: v for k, v in bat.items()},
                n1t_alpha=n1t_alpha(W, rng_n1t),
                norms=x.tolist(), lam=lam.tolist())


def phase_pilot():
    arm = dict(n=1024, d=64, s=1.0)
    res = {}
    for lr in [0.002, 0.01, 0.05, 0.2]:
        ls = []
        for seed in [90, 91, 92]:
            r = run_qsae(**arm, seed=seed, lr=lr, checkpoints=[2000], Tmax=2000,
                         max_ext=0)
            ls.append(r["records"][-1]["loss"])
        res[lr] = dict(median_loss=float(np.median(ls)), finite=bool(np.isfinite(ls).all()))
        print(lr, res[lr])
    ok = {k: v["median_loss"] for k, v in res.items() if v["finite"] and v["median_loss"] < 1.0}
    best = min(ok, key=ok.get) if ok else 0.01
    json.dump(dict(results={str(k): v for k, v in res.items()}, chosen_lr=best),
              open(OUT / "lr_pilot.json", "w"), indent=1)
    print("chosen lr:", best)


def phase_main(lr):
    a3 = json.load(open(A3))
    rho = a3["rho_cal"]
    for arm in GRID:
        heavy = arm["d"] * arm["n"] >= 200000 and arm["n"] > 2048
        device = "cuda" if (heavy and torch_available()) else "cpu"
        for seed in [0, 1, 2]:
            tag = f"qsae_n{arm['n']}_d{arm['d']}_s{arm['s']}_seed{seed}"
            out = run_qsae(**arm, seed=seed, lr=lr, checkpoints=list(CK), Tmax=200000,
                           device=device)
            # P-C' transient scoring (frozen rho from A3 pilot; A1-theorem
            # sparse-exact endpoint as x*: ones on top-d usage ranks, zero tail)
            x_star_se = np.zeros(arm["n"]); x_star_se[:arm["d"]] = 1.0
            rng0 = np.random.default_rng(seed + 900)
            x0 = ((0.05 * rng0.standard_normal((arm["d"], arm["n"])) / np.sqrt(arm["d"])) ** 2
                  ).sum(axis=0)
            devs = {}
            for t in PC_CK:
                rec = next((r for r in out["records"] if r["t"] == t), None)
                if rec:
                    ap = alpha_pred(arm, t, rho, x_star_se, x0)
                    devs[t] = dict(alpha_pred=ap, alpha_meas=rec["bat"]["top90"]["alpha"],
                                   dev=abs(ap - rec["bat"]["top90"]["alpha"]))
            out["pc_scoring"] = devs
            out["pc_pass"] = (sum(1 for v in devs.values() if v["dev"] <= 0.10) >= 3
                              if devs else None)
            # P-A'/P-B' only if converged
            if out["converged"]:
                end = out["records"][-1]
                x = np.array(end["norms"])
                k = np.arange(1, arm["n"] + 1, dtype=np.float64)
                H = (k ** (-arm["s"])).sum()
                mask = x > 1e-8 * x.max()
                X = np.vstack([np.ones(mask.sum()), -(H * (k ** arm["s"]))[mask]]).T
                coef, *_ = np.linalg.lstsq(X, x[mask], rcond=None)
                ss = ((x[mask] - x[mask].mean()) ** 2).sum()
                out["pa_norm_fit"] = dict(r2=float(1 - ((X @ coef - x[mask]) ** 2).sum() / ss),
                                          c0=float(coef[0]), c1H=float(coef[1]),
                                          n_unclipped=int(mask.sum()))
                rngb = np.random.default_rng(seed + 300)
                lam_act = np.array(end["lam"])
                rids = []
                for _ in range(5):
                    lam_mf = rand_frame(x, rngb)
                    m = min(len(lam_act), len(lam_mf))
                    la, lm = np.log(np.maximum(lam_act[:m], EPS)), np.log(np.maximum(lam_mf[:m], EPS))
                    rids.append(1 - ((la - lm) ** 2).sum() / ((la - la.mean()) ** 2).sum())
                out["pb_r2_identity"] = float(np.median(rids))
                out["pb_adequate"] = bool(end["bat"]["top90"]["r2_pl"] - out["pb_r2_identity"] <= 0.05)
            json.dump(out, open(OUT / f"{tag}.json", "w"))
            print(tag, "conv", out["converged"], "ext", out["extensions_used"],
                  "a_end", round(out["records"][-1]["bat"]["top90"]["alpha"], 3),
                  "pc_pass", out["pc_pass"])


def run_xe(n, d, s, seed, lr, checkpoints, Tmax, batch=256, wall_cap_s=7200):
    rng = np.random.default_rng(seed)
    p = zipf_p(n, s)
    W = 0.01 * rng.standard_normal((d, n))
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    b1, b2, adeps = 0.9, 0.999, 1e-8
    recs = []
    t0 = time.time()
    rngE = np.random.default_rng(12345)
    Ie = rngE.choice(n, size=1024, p=p); Je = rngE.choice(n, size=1024, p=p)
    for t in range(1, Tmax + 1):
        I = rng.choice(n, size=batch, p=p)
        J = rng.choice(n, size=batch, p=p)
        H = W[:, I]
        Z = W.T @ H
        Z = Z - Z.max(axis=0, keepdims=True)
        P = np.exp(Z); P /= P.sum(axis=0, keepdims=True)
        P[J, np.arange(batch)] -= 1.0
        P /= batch
        gW = H @ P.T
        np.add.at(gW, (slice(None), I), W @ P)
        mW = b1 * mW + (1 - b1) * gW
        vW = b2 * vW + (1 - b2) * gW * gW
        mh = mW / (1 - b1 ** t); vh = vW / (1 - b2 ** t)
        W -= lr * mh / (np.sqrt(vh) + adeps)
        if t in checkpoints:
            Ze = W.T @ W[:, Ie]; Ze -= Ze.max(axis=0, keepdims=True)
            Pe = np.exp(Ze); Pe /= Pe.sum(axis=0, keepdims=True)
            loss = float(-np.log(Pe[Je, np.arange(1024)] + 1e-12).mean())
            lam = spectrum(W)
            bat = battery(lam)
            rng_n1t = np.random.default_rng(seed + 500 + t)
            recs.append(dict(t=t, loss=loss, gnorm=float(np.abs(gW).max()), bat=bat,
                             n1t_alpha=n1t_alpha(W, rng_n1t), lam=lam.tolist(),
                             norms=(W ** 2).sum(axis=0).tolist()))
        if (time.time() - t0) > wall_cap_s:
            break
    return dict(n=n, d=d, s=s, seed=seed, lr=lr, Tmax=Tmax, records=recs,
                wall_s=time.time() - t0)


def phase_xe():
    for seed in [0, 1, 2]:
        out = run_xe(1024, 64, 1.0, seed, 3e-3, [3000, 12000, 48000, 192000], 192000)
        recs = out["records"]
        if len(recs) >= 2:
            a_l, a_p = recs[-1]["bat"]["top90"]["alpha"], recs[-2]["bat"]["top90"]["alpha"]
            out["gate_stationary"] = bool(abs(a_l - a_p) <= 0.05)
        json.dump(out, open(OUT / f"xe_n1024_d64_s1_seed{seed}.json", "w"))
        print("xe seed", seed, [(r["t"], round(r["bat"]["top90"]["alpha"], 3)) for r in recs])


def phase_nulls(lr):
    for seed in [0, 1, 2]:
        out = run_qsae(1024, 64, 0.0, seed, lr, list(CK), 200000, max_ext=1)
        json.dump(out, open(OUT / f"null_n2_uniform_seed{seed}.json", "w"))
        print("N2 seed", seed, "alpha_end", round(out["records"][-1]["bat"]["top90"]["alpha"], 3))
    from scipy.optimize import minimize
    n, d, s = 1024, 64, 1.0
    p = zipf_p(n, s)
    gd = run_qsae(n, d, s, 0, lr, list(CK), 200000, max_ext=1)
    lam_gd = np.array(gd["records"][-1]["lam"])
    runs = []
    for k in range(3):
        rng = np.random.default_rng(7000 + k)
        W0 = 0.05 * rng.standard_normal((d, n)) / np.sqrt(d)
        f = lambda W: qsae_loss_grad(W.reshape(d, n), p)[0]
        g = lambda W: qsae_loss_grad(W.reshape(d, n), p)[1].ravel()
        r = minimize(f, W0.ravel(), jac=g, method="L-BFGS-B",
                     options=dict(maxiter=20000, maxfun=40000, ftol=1e-15, gtol=1e-10))
        lam = spectrum(r.x.reshape(d, n))
        runs.append(dict(seed=k, loss=float(r.fun), alpha=fit_window(lam, 0.9)["alpha"],
                         lam=lam.tolist()))
    m = min(len(lam_gd), min(len(o["lam"]) for o in runs))
    dev = max(abs(a - b) / max(abs(b), EPS) for o in runs for a, b in zip(lam_gd[:m], o["lam"][:m]))
    json.dump(dict(runs=runs, max_rel_dev_vs_gd=dev),
              open(OUT / "null_n0_lbfgs.json", "w"))
    print("N0 max_rel_dev", dev)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["pilot", "main", "xe", "nulls"])
    ap.add_argument("--lr", type=float, default=None)
    a = ap.parse_args()
    if a.phase == "pilot":
        phase_pilot()
    elif a.phase == "main":
        phase_main(a.lr or json.load(open(OUT / "lr_pilot.json"))["chosen_lr"])
    elif a.phase == "xe":
        phase_xe()
    elif a.phase == "nulls":
        phase_nulls(a.lr or json.load(open(OUT / "lr_pilot.json"))["chosen_lr"])
