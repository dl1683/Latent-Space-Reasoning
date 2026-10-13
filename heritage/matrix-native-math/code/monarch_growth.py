"""Monarch composition-growth law (research/MONARCH_GROWTH_V1.md): dim closure(Mon^k) for n = q^2, Mon^k = image of
Phi_k(B_0..B_k) = B_0 P B_1 P ... P B_k, B_i block-diagonal (q blocks of q x q), P = coordinate swap (x,y)->(y,x).
Claimed law (Codex gpt-6.1-sol, round 1 of the 14:49 meditation): dim = min(q^4, (k+1) q^3 - k q^2).
Exact rank of the differential D Phi_k over F_65521, (a) at GENERIC random invertible B_i and (b) at Codex's permutation witness (affine-plane
directions).  rank over F_p at a point is a LOWER BOUND for the generic rank over Q; the upper bound is the diagonal-junction gauge count.
Run: python code/monarch_growth.py [--selftest]"""
import sys, json, time, itertools
import numpy as np

P = 65521


def rank_mod(M):
    M = (M.copy() % P).astype(np.int64)
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        if r >= rows:
            break
        piv = np.nonzero(M[r:, c])[0]
        if len(piv) == 0:
            continue
        p = piv[0] + r
        if p != r:
            M[[r, p]] = M[[p, r]]
        invp = pow(int(M[r, c]), P - 2, P)
        M[r] = (M[r] * invp) % P
        nz = np.nonzero(M[:, c])[0]
        nz = nz[nz != r]
        if len(nz):
            M[nz] = (M[nz] - np.outer(M[nz, c], M[r])) % P
        r += 1
    return r


def swap_perm(q):
    n = q * q
    Pm = np.zeros((n, n), dtype=np.int64)
    for x in range(q):
        for y in range(q):
            Pm[x * q + y, y * q + x] = 1                 # index = y*q + x (block y first) -> (x,y)->(y,x)
    return Pm


def idx(x, y, q):
    return y * q + x


def blockdiag_units(q):
    """Basis of H: block y, entry (x, x').  Returns list of (row, col) pairs of the n x n matrix."""
    return [(idx(x, y, q), idx(xp, y, q)) for y in range(q) for x in range(q) for xp in range(q)]


def jacobian_rank(Bs, q):
    n = q * q
    Pm = swap_perm(q)
    k = len(Bs) - 1
    # left_i = B_0 P ... B_{i-1} P  ; right_i = P B_{i+1} ... P B_k
    left = [np.eye(n, dtype=np.int64)]
    for i in range(k):
        left.append(((left[-1] @ Bs[i]) % P @ Pm) % P)
    # right_i = P B_{i+1} P B_{i+2} ... P B_k  (for i < k)
    right = [None] * (k + 1)
    right[k] = np.eye(n, dtype=np.int64)
    for i in range(k - 1, -1, -1):
        right[i] = (((Pm @ Bs[i + 1]) % P) @ right[i + 1]) % P
    units = blockdiag_units(q)
    cols = []
    for i in range(k + 1):
        L, R = left[i], right[i]
        for (a, b) in units:
            col = np.outer(L[:, a], R[b, :]) % P
            cols.append(col.reshape(-1))
    J = np.stack(cols, axis=1)                           # n^2 x (k+1) q^3
    return rank_mod(J.T)


def random_H(q, rng):
    n = q * q
    B = np.zeros((n, n), dtype=np.int64)
    for y in range(q):
        while True:
            blk = rng.integers(0, P, (q, q))
            if rank_mod(blk) == q:
                break
        B[y * q:(y + 1) * q, y * q:(y + 1) * q] = blk
    return B


def predicted(q, k):
    return min(q ** 4, (k + 1) * q ** 3 - k * q ** 2)


def run(qs=(2, 3, 4, 5), kmax=None, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for q in qs:
        for k in range(1, (kmax or q + 2) + 1):
            Bs = [random_H(q, rng) for _ in range(k + 1)]
            t = time.time()
            r = jacobian_rank(Bs, q)
            rows.append({"q": q, "n": q * q, "k": k, "generic_rank_modp": r, "predicted": predicted(q, k), "match": r == predicted(q, k),
                         "seconds": round(time.time() - t, 2)})
            print(rows[-1], flush=True)
    return rows


def selftest():
    rng = np.random.default_rng(0)
    # k = 1 (L P R): dim = 2 q^3 - q^2 (known from FLOOR_LDR_V1 M2: dim Mon = 2 m^3 - m^2)
    for q in (2, 3):
        r = jacobian_rank([random_H(q, rng), random_H(q, rng)], q)
        assert r == 2 * q ** 3 - q ** 2, (q, r)
    # k = 2: dim = 3 q^3 - 2 q^2 (FLOOR_LDR_V1 M2)
    q = 3
    r = jacobian_rank([random_H(q, rng) for _ in range(3)], q)
    assert r == 3 * q ** 3 - 2 * q ** 2, r
    # negative control: wrong swap (identity instead of P) collapses to the block-diagonal algebra, rank q^3
    global swap_perm
    orig = swap_perm
    swap_perm = lambda qq: np.eye(qq * qq, dtype=np.int64)
    try:
        r0 = jacobian_rank([random_H(3, rng), random_H(3, rng)], 3)
    finally:
        swap_perm = orig
    assert r0 == 3 ** 3, r0
    print("selftest PASSED")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    import os
    rows = run()
    os.makedirs("results/monarch_growth_v1", exist_ok=True)
    json.dump(rows, open("results/monarch_growth_v1/generic_rank.json", "w"), indent=1)
