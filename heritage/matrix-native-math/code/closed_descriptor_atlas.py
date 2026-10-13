"""Closed-descriptor atlas v1 (research/CLOSED_DESCRIPTOR_ATLAS_V1.md).  Entry 1: MPO-jet / truncated-series Kronecker product (B359).
Exact verification over F_P (P = 65521) against dense matrices: product table, degree-state extraction recurrence, contraction count,
negative controls (reversed order, omitted mixed term, unsafe in-place update).  CPU only.  Run: python code/closed_descriptor_atlas.py [--selftest]"""
import sys, json, itertools
import numpy as np

P = 65521


def rand_jet(rng, d, k, p=2):
    """d local jets: F[j][r] is a p x p matrix over F_P (degree r coefficient of site j)."""
    return [[rng.integers(0, P, (p, p)).astype(np.int64) for _ in range(k + 1)] for _ in range(d)]


def compose(A, B, k):
    """Native product: per site truncated convolution  C_j[u+v] += A_j[u] @ B_j[v]  (A applied AFTER B: C = A B)."""
    out = []
    for Aj, Bj in zip(A, B):
        Cj = [np.zeros_like(Aj[0]) for _ in range(k + 1)]
        for u in range(k + 1):
            for v in range(k + 1 - u):
                Cj[u + v] = (Cj[u + v] + Aj[u] @ Bj[v]) % P
        out.append(Cj)
    return out


def kron_list(mats):
    M = np.array([[1]], dtype=np.int64)
    for X in mats:
        M = np.kron(M, X) % P
    return M


def dense_coeffs(A, k):
    """H_l = sum_{r_1+..+r_d = l} kron_j A_j[r_j]."""
    d = len(A)
    n = A[0][0].shape[0] ** d
    H = [np.zeros((n, n), dtype=np.int64) for _ in range(k + 1)]
    for rs in itertools.product(range(k + 1), repeat=d):
        l = sum(rs)
        if l <= k:
            H[l] = (H[l] + kron_list([A[j][rs[j]] for j in range(d)])) % P
    return H


def apply_mode(M, x, j, d, p):
    """Apply the p x p matrix M along tensor mode j of x (shape p^d vector, site 0 most significant)."""
    X = x.reshape((p,) * d)
    X = np.moveaxis(np.tensordot(M, X, axes=([1], [j])), 0, j)
    return (X.reshape(-1)) % P


def degree_state_action(A, x, k, p=2, count=None):
    """Return [H_0 x, ..., H_k x] with the degree-state recurrence: y_l = sum_{r<=l} A_j[r] (x)_{j} x_{l-r}.  Safe double buffering."""
    d = len(A)
    cur = [x.copy()] + [np.zeros_like(x) for _ in range(k)]
    for j in range(d):
        new = [np.zeros_like(x) for _ in range(k + 1)]
        for l in range(k + 1):
            for r in range(l + 1):
                new[l] = (new[l] + apply_mode(A[j][r], cur[l - r], j, d, p)) % P
                if count is not None:
                    count[0] += 1
        cur = new
    return cur


def verify(d, k, p=2, seed=0, trials=3):
    rng = np.random.default_rng(seed)
    out = {"d": d, "k": k, "p": p}
    ok_table = ok_act = True
    cnt = [0]
    for _ in range(trials):
        A, B = rand_jet(rng, d, k, p), rand_jet(rng, d, k, p)
        C = compose(A, B, k)
        HA, HB, HC = dense_coeffs(A, k), dense_coeffs(B, k), dense_coeffs(C, k)
        for l in range(k + 1):                                   # coefficient of eps^l of the product matrix
            prod = sum((HA[u] @ HB[l - u]) % P for u in range(l + 1)) % P
            ok_table &= bool(np.array_equal(prod % P, HC[l]))
        x = rng.integers(0, P, p ** d).astype(np.int64)
        cnt[0] = 0
        act = degree_state_action(C, x, k, p, cnt)
        for l in range(k + 1):
            ok_act &= bool(np.array_equal(act[l], (HC[l] @ x) % P))
    out.update(product_table_exact=ok_table, degree_state_action_exact=ok_act, mode_contractions=cnt[0],
               predicted_contractions=d * (k + 1) * (k + 2) // 2, descriptor_scalars=d * (k + 1) * p * p)
    return out


def negative_controls(seed=1):
    rng = np.random.default_rng(seed)
    d, k, p = 3, 2, 2
    A, B = rand_jet(rng, d, k, p), rand_jet(rng, d, k, p)
    C = compose(A, B, k); Crev = compose(B, A, k)
    HA, HB, HC = dense_coeffs(A, k), dense_coeffs(B, k), dense_coeffs(C, k)
    l = 2
    right = sum((HA[u] @ HB[l - u]) % P for u in range(l + 1)) % P
    rev = dense_coeffs(Crev, k)[l]
    omitted = (HA[0] @ HB[2] + HA[2] @ HB[0]) % P                 # drops the mixed term A1 B1
    x = rng.integers(0, P, p ** d).astype(np.int64)
    # unsafe in-place update (reads already-updated lower degrees)
    cur = [x.copy()] + [np.zeros_like(x) for _ in range(k)]
    for j in range(d):
        for l2 in range(k, -1, -1):                               # wrong direction: in-place in ascending would overwrite; use ascending in-place
            pass
    cur = [x.copy()] + [np.zeros_like(x) for _ in range(k)]
    for j in range(d):
        for l2 in range(k + 1):
            acc = np.zeros_like(x)
            for r in range(l2 + 1):
                acc = (acc + apply_mode(C[j][r], cur[l2 - r], j, d, p)) % P
            cur[l2] = acc                                         # in-place: later degrees read updated lower degrees
    inplace_ok = all(np.array_equal(cur[l2], (dense_coeffs(C, k)[l2] @ x) % P) for l2 in range(k + 1))
    return {"reversed_order_detected": not np.array_equal(rev, right), "omitted_mixed_term_detected": not np.array_equal(omitted, right),
            "unsafe_inplace_detected": not inplace_ok}


def selftest():
    r = verify(2, 2, trials=2)
    assert r["product_table_exact"] and r["degree_state_action_exact"], r
    nc = negative_controls()
    assert all(nc.values()), nc
    print("selftest PASSED")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    res = {"entry1_mpo_jet": [verify(d, k) for d in (2, 3, 4) for k in (1, 2)], "negative_controls": negative_controls()}
    import os; os.makedirs("results/closed_descriptor_atlas_v1", exist_ok=True)
    json.dump(res, open("results/closed_descriptor_atlas_v1/entry1_mpo_jet.json", "w"), indent=1)
    for r in res["entry1_mpo_jet"]: print(r)
    print(res["negative_controls"])
