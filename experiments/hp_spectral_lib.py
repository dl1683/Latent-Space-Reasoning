"""Shared primitives for the HP spectral-law cycles (QSAE, fits, frames, mean field)."""
import numpy as np

EPS = 1e-14


def zipf_p(n, s):
    k = np.arange(1, n + 1, dtype=np.float64)
    w = k ** (-s) if s > 0 else np.ones(n)
    return w / w.sum()


def qsae_loss_grad(W, p):
    """W: d x n float array; p: usage weights. Returns (loss, grad)."""
    S = W @ W.T
    Wp = W * p[None, :]
    A = Wp @ W.T
    loss = np.trace(A @ S) - 2.0 * np.trace(A) + p.sum()
    grad = 2.0 * (S @ Wp) + 2.0 * (A @ W) - 4.0 * Wp
    return loss, grad


def spectrum(W):
    lam = np.linalg.eigvalsh(np.asarray(W) @ np.asarray(W).T)[::-1]
    return np.maximum(lam, EPS)


def fit_window(lam, frac):
    """Power-law vs curved(quadratic-in-log) fits on the top-`frac` ranked eigenvalues."""
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
    return dict(alpha=-float(c1[0]), r2_pl=float(r1), r2_quad=float(r2), dR2=float(r1 - r2), m=m)


def battery(lam):
    return {w: fit_window(lam, f) for w, f in [("top50", 0.5), ("top90", 0.9), ("full", 1.0)]}


def rand_frame(x_profile, d, rng):
    """d x n frame with the given column-norm profile and fresh random directions."""
    x = np.maximum(np.asarray(x_profile, dtype=np.float64), 0.0)
    n = len(x)
    U = rng.standard_normal((d, n))
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    return spectrum(U * np.sqrt(x)[None, :])


def n1t_alpha(W, rng, reps=3):
    """N1-t control: preserve learned norms, randomize directions; median top90 alpha."""
    W = np.asarray(W)
    x = (W ** 2).sum(axis=0)
    d = W.shape[0]
    return float(np.median([fit_window(rand_frame(x, d, rng), 0.9)["alpha"] for _ in range(reps)]))


def mf_fixed_point(n, d, s, iters=40000, tol=1e-13):
    """Exact random-mixing stationarity of the QSAE (P1')."""
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


def alpha_pred(arm, t, rho, x_star, x0, reps=5, seed=7):
    """P3' logistic norms mapped through the random-frame spectral operator."""
    p = zipf_p(arm["n"], arm["s"])
    xt = x_star / (1 + np.maximum(x_star / np.maximum(x0, EPS) - 1, 0) * np.exp(-rho * p * t))
    rng = np.random.default_rng(seed)
    return float(np.median([fit_window(rand_frame(xt, arm["d"], rng), 0.9)["alpha"]
                            for _ in range(reps)]))
