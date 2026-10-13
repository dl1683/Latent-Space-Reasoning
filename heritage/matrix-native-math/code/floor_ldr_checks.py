"""floor_ldr_v1 checks (single thread, small exact linear algebra). Usage: python code/floor_ldr_checks.py <suite>
suites: monarch | butterfly | ldr | dao | m2_witness | dao_exact | ldr_kron_family | qs_kron_exact | r3 | monarch_extra |
butterfly_extra. Writes results/floor_ldr_v1/<suite>.json.
Dimension of a product-of-sparse-factors variety = generic rank of the parameter Jacobian; a rank mod a prime at one
random integer point is a rigorous lower bound (rank can only drop mod p / at special points); the upper bounds
quoted in research/FLOOR_LDR_V1.md come from explicit gauge symmetries.

Repair round 2 (gate 5, 2026-09-29) predictions, derived from the proofs in the doc and written BEFORE the four new
suites were run (a mismatch withdraws the statement, not the prediction):
  R2-P1 (m2_witness): at L = R = I, S_{a'}[a,b] = 2^(a a' b) the Jacobian rank is 3m^3 - 2m^2 (kernel = the 2m^2
        gauge) for m = 2..5 mod 32003, and exactly over Q for m = 2, 3; M = Pi S Pi has M[(a,e),(b,c)] = delta_ec 2^(aeb).
  R2-P2 (dao_exact): for D = Pi L Pi R with rational L, R (m = 3): every slice of Pi D has all 2x2 minors 0 (rank <= 1)
        while Pi D^-1 and Pi D^T each have a slice with a nonzero 2x2 minor (certificate stored as exact Fractions).
  R2-P3 (ldr_kron_family): rank of nabla_{Ze(x)Ze', Zf(x)Zf'}(I(x)I) = [e'!=f'](n1-1) + [e!=f](n2-1) + [ee'!=ff'],
        i.e. n1+n2-1 generically, n1, n2 in 2..6.
  R2-P4 (qs_kron_exact): order(J_k) = order(I_k + J_k) = 1 and order(J_k (x) (I_k + J_k)) = k, k = 2..6.

Repair round 3 (cold no-memory referee, 2026-09-30) predictions, derived from the proofs in the doc and written BEFORE
the suites r3 / monarch_extra / butterfly_extra were run (a mismatch withdraws the statement, not the prediction):
  R3-P1 (r3.kron_reorder): for (m1,m2) in {(2,2),(2,3),(3,2),(3,3)} and random M_s in Mon(m_s), Q(M1 (x) M2)Q^T is in
        Mon(m1 m2) (all slices rank <= 1) while the plain lexicographic M1 (x) M2 is not; Q(Pi1 (x) Pi2)Q^T = Pi_{m1 m2};
        Pi_{m1m2} Q(Pi1 M1 (x) Pi2 M2)Q^T is in Mon(m1 m2); a random dense matrix is not in Mon (negative control).
  R3-P2 (r3.monotone): for m = 2, 3 and k = 1, 2, 3, replacing the last factor pair (L, R) of X = B_0 Pi ... Pi B_k by
        (L', S, R') with the Latin square S_x[a,b] = [x = a+b mod m] reproduces X exactly (now with k+1 Pi's); the shifted
        (wrong) Latin square does not.
  R3-P3 (r3.dao_piside): on the stored certificate D^-1 Pi = R^-1 Pi L^-1 and D^T Pi = R^T Pi L^T exactly, both are in
        Mon_LPiR, while Pi D^-1 and Pi D^T are not.
  R3-P4 (r3.slice_formula): slice_{ab}(L Pi S Pi R) = L_a diag(S_{a'}[a,b]) R_b exactly (m = 2, 3, 4); the variant with
        R_a in place of R_b fails.
  R3-P5 (r3.reconstructible): for (e,f,e',f') = (2,3,5,7) and (n1,n2) in {(2,2),(2,3),(3,2),(3,3)} the map
        M -> A M - M B of the Kronecker pair is bijective (rank N^2); for (2,3,3,2) at (2,2) it is singular; A = B = I
        gives the zero map.
  R3-P6 (r3.closed_algebras): for coprime (n1,n2) the Kronecker product of circulants is a circulant after the CRT
        reindexing and is not in lexicographic order; for block-diagonal A, B the product is block diagonal after the
        reordering (i,x,j,y) -> (i,j,x,y) and is not in the plain order.
  R3-P7 (monarch_extra): dim = min(m^4, (k+1) m^3 - k m^2) at (m,k) in {(2,3),(3,4),(5,3),(5,5),(6,1),(6,2),(6,5),(6,6)},
        i.e. 16, 81, 425, 625, 396, 576, 1116, 1296.
  R3-P8 (butterfly_extra): n = 32, L = 5: rank n(kL+1) = 192, 352, 992 at k = 1, 2, 6 and dense 1024 at k = 7."""
import json
import math
import os
import random
import sys
import time
from fractions import Fraction as Fr

import numpy as np

P = 32003
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "floor_ldr_v1")


def rank_mod(A, p=P):
    A = np.array(A, dtype=np.int64) % p
    r = 0
    rows, cols = A.shape
    for c in range(cols):
        if r == rows:
            break
        nz = np.nonzero(A[r:, c])[0]
        if nz.size == 0:
            continue
        i = r + nz[0]
        A[[r, i]] = A[[i, r]]
        A[r] = (A[r] * pow(int(A[r, c]), p - 2, p)) % p
        m = np.nonzero(A[:, c])[0]
        m = m[m != r]
        if m.size:
            A[m] = (A[m] - np.outer(A[m, c], A[r])) % p
        r += 1
    return r


def rank_mod_fwd(A, p=P):
    """Forward-elimination rank mod p: same value as rank_mod, about half the work (used for the larger R3 sizes)."""
    A = np.array(A, dtype=np.int64) % p
    r = 0
    rows, cols = A.shape
    for c in range(cols):
        if r == rows:
            break
        nz = np.nonzero(A[r:, c])[0]
        if nz.size == 0:
            continue
        i = r + nz[0]
        if i != r:
            A[[r, i]] = A[[i, r]]
        A[r] = (A[r] * pow(int(A[r, c]), p - 2, p)) % p
        below = np.nonzero(A[r + 1:, c])[0] + r + 1
        if below.size:
            A[below] = (A[below] - np.outer(A[below, c], A[r])) % p
        r += 1
    return r


def jac_rank(factors, masks, rank_fn=None):
    """factors: list of n x n int matrices; masks: boolean free-entry matrices, or None for a fixed factor."""
    n = factors[0].shape[0]
    pre = [np.eye(n, dtype=np.int64)]
    for F in factors:
        pre.append((pre[-1] @ F) % P)
    suf = [np.eye(n, dtype=np.int64)]
    for F in reversed(factors):
        suf.append((F @ suf[-1]) % P)
    suf = suf[::-1]  # suf[j] = F_j ... F_{N-1}
    cols = []
    for j, mk in enumerate(masks):
        if mk is None:
            continue
        L, R = pre[j], suf[j + 1]
        for (r, c) in zip(*np.nonzero(mk)):
            cols.append(np.outer(L[:, r], R[c, :]).ravel() % P)
    J = np.array(cols)  # (params, n^2); rank is the same as its transpose
    return (rank_fn or rank_mod)(J), J.shape[0]


def blockdiag_mask(m):
    n = m * m
    M = np.zeros((n, n), dtype=bool)
    for b in range(m):
        M[b * m:(b + 1) * m, b * m:(b + 1) * m] = True
    return M


def perm_mat(m):
    n = m * m
    Pm = np.zeros((n, n), dtype=np.int64)
    for a in range(m):
        for b in range(m):
            Pm[b * m + a, a * m + b] = 1
    return Pm


MONARCH_PAIRS = [(2, 1), (3, 1), (4, 1), (5, 1), (2, 2), (3, 2), (4, 2), (5, 2), (3, 3), (4, 3), (4, 4), (5, 4)]
MONARCH_EXTRA_PAIRS = [(2, 3), (3, 4), (5, 3), (5, 5), (6, 1), (6, 2), (6, 5), (6, 6)]


def monarch_suite(pairs=MONARCH_PAIRS, seed=1, rank_fn=None):
    rng = np.random.default_rng(seed)
    res = []
    for m, k in pairs:
        n = m * m
        bd = blockdiag_mask(m)
        Pm = perm_mat(m)
        fac, msk = [], []
        for _ in range(k):
            fac += [rng.integers(1, P, (n, n)) * bd, Pm, rng.integers(1, P, (n, n)) * bd]
            msk += [bd, None, bd]
        rk, npar = jac_rank(fac, msk, rank_fn)
        pred = min(m ** 4, (k + 1) * m ** 3 - k * m ** 2)
        res.append(dict(m=m, n=n, k=k, params=npar, jac_rank_mod_p=rk, predicted=pred, dense=m ** 4, match=rk == pred))
        print(res[-1], flush=True)
    return res


def butterfly_suite(ns=(4, 8, 16), ks=(1, 2, 3, 4, 5), seed=2, rank_fn=None):
    rng = np.random.default_rng(seed)
    res = []
    for n in ns:
        L = n.bit_length() - 1
        masks = []
        for j in range(L):
            M = np.zeros((n, n), dtype=bool)
            for i in range(n):
                M[i, i] = True
                M[i, i ^ (1 << j)] = True
            masks.append(M)
        for k in ks:
            ms = masks * k
            fac = [rng.integers(1, P, (n, n)) * mk for mk in ms]
            rk, npar = jac_rank(fac, ms, rank_fn)
            pred = min(n * n, n * (k * L + 1))
            res.append(dict(n=n, L=L, k=k, params=npar, jac_rank_mod_p=rk, predicted=pred, match=rk == pred))
            print(res[-1], flush=True)
            if rk == n * n:
                break
    return res


# ---------------- exact rational linear algebra (small) ----------------
def mat(n, m, f):
    return [[Fr(f(i, j)) for j in range(m)] for i in range(n)]


def mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def sub(A, B):
    return [[a - b for a, b in zip(r, s)] for r, s in zip(A, B)]


def add(A, B):
    return [[a + b for a, b in zip(r, s)] for r, s in zip(A, B)]


def rank_q(A):
    A = [r[:] for r in A]
    rk, rows, cols = 0, len(A), len(A[0])
    for c in range(cols):
        piv = next((i for i in range(rk, rows) if A[i][c] != 0), None)
        if piv is None:
            continue
        A[rk], A[piv] = A[piv], A[rk]
        for i in range(rk + 1, rows):
            if A[i][c] != 0:
                f = A[i][c] / A[rk][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[rk])]
        rk += 1
    return rk


def inv_q(A):
    n = len(A)
    M = [A[i][:] + [Fr(int(i == j)) for j in range(n)] for i in range(n)]
    for c in range(n):
        piv = next(i for i in range(c, n) if M[i][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        M[c] = [x / M[c][c] for x in M[c]]
        for i in range(n):
            if i != c and M[i][c] != 0:
                f = M[i][c]
                M[i] = [x - f * y for x, y in zip(M[i], M[c])]
    return [r[n:] for r in M]


def kron(A, B):
    return [[a * b for a in ra for b in rb] for ra in A for rb in B]


def Z(n, f):  # (Z_f x)[i] = x[i-1], (Z_f x)[0] = f x[n-1]
    return mat(n, n, lambda i, j: 1 if i == j + 1 else (f if (i == 0 and j == n - 1) else 0))


def toeplitz(n, rnd):
    t = {d: rnd() for d in range(-n + 1, n)}
    return mat(n, n, lambda i, j: t[i - j])


def nabla(A, M, B):
    return rank_q(sub(mul(A, M), mul(M, B)))


def qs_order(Mx):
    """Quasiseparable order: max over cuts t of max(rank M[t:, :t], rank M[:t, t:])."""
    n_ = len(Mx)
    return max(max(rank_q([r[:k] for r in Mx[k:]]), rank_q([r[k:] for r in Mx[:k]])) for k in range(1, n_))


def ldr_suite():
    rnd = random.Random(3)
    R = lambda: rnd.randint(-9, 9) or 1
    out = {}
    n = 6
    e, f, g = Fr(2), Fr(3), Fr(5)
    T1, T2 = toeplitz(n, R), toeplitz(n, R)
    out["toeplitz_rank_Ze_Zf"] = nabla(Z(n, e), T1, Z(n, f))
    out["product_rank_Ze_Zg"] = nabla(Z(n, e), mul(T1, T2), Z(n, g))  # T1 under (Ze,Zf), T2 under (Zf,Zg): bound 4
    out["inverse_rank_swapped_pair"] = nabla(Z(n, f), inv_q(T1), Z(n, e))
    out["sum_rank_same_pair"] = nabla(Z(n, e), add(T1, T2), Z(n, f))
    kr = []
    for n1, n2 in [(3, 3), (3, 4), (4, 4), (4, 5), (5, 5)]:
        A, B = toeplitz(n1, R), toeplitz(n2, R)
        rk = nabla(kron(Z(n1, e), Z(n2, e)), kron(A, B), kron(Z(n1, f), Z(n2, f)))
        kr.append(dict(n1=n1, n2=n2, rank=rk, two_n1_plus_two_n2=2 * n1 + 2 * n2, n1n2=n1 * n2))
    out["kron_toeplitz_kron_pair"] = kr
    N, h = 8, 4
    M = mat(N, N, lambda i, j: rnd.randint(-9, 9))
    a12 = [[rnd.randint(-3, 3) for _ in range(2)] for _ in range(h)]
    b12 = [[rnd.randint(-3, 3) for _ in range(N - h)] for _ in range(2)]
    a21 = [[rnd.randint(-3, 3)] for _ in range(N - h)]
    b21 = [[rnd.randint(-3, 3) for _ in range(h)]]
    B12, B21 = mul(mat(h, 2, lambda i, j: a12[i][j]), mat(2, N - h, lambda i, j: b12[i][j])), mul(
        mat(N - h, 1, lambda i, j: a21[i][j]), mat(1, h, lambda i, j: b21[i][j]))
    for i in range(h):
        for j in range(N - h):
            M[i][h + j] = B12[i][j]
            M[h + j][i] = B21[j][i]
    Pm = mat(N, N, lambda i, j: 1 if (i == j and i < h) else 0)
    out["projector_commutator_rank"] = rank_q(sub(mul(Pm, M), mul(M, Pm)))
    out["projector_blocks_rank_sum"] = rank_q([r[h:] for r in M[:h]]) + rank_q([r[:h] for r in M[h:]])

    def qs(n):
        u, v, w, z = ([rnd.randint(1, 5) for _ in range(n)] for _ in range(4))
        return mat(n, n, lambda i, j: (u[i] * v[j] if i > j else (w[i] * z[j] if i < j else rnd.randint(20, 40))))

    order = qs_order
    A, B = qs(8), qs(8)
    out["qs_order_A_B_AB_A+B_Ainv"] = [order(A), order(B), order(mul(A, B)), order(add(A, B)), order(inv_q(A))]
    # Kronecker of two order-1 quasiseparable matrices in lexicographic order: order grows with the block size
    out["qs_kron_order_n1_n2_order"] = [(k1, k2, order(kron(qs(k1), qs(k2)))) for k1, k2 in [(3, 3), (4, 4), (5, 5), (6, 6)]]
    return out


def dao_suite():
    """Dao 2022 class Pi.Mon = {Pi L Pi R} vs the doc class Mon = {L Pi R}; Latin-square containment Mon in Mon^2."""
    rng = np.random.default_rng(5)
    out = {"jacobian": [], "latin_square": [], "dao_closure": []}
    for m in (2, 3, 4):
        n = m * m
        bd, Pm = blockdiag_mask(m), perm_mat(m)
        fac1 = [Pm, rng.integers(1, P, (n, n)) * bd, Pm, rng.integers(1, P, (n, n)) * bd]
        msk1 = [None, bd, None, bd]
        r1, _ = jac_rank(fac1, msk1)
        fac2 = fac1 + [Pm, rng.integers(1, P, (n, n)) * bd, Pm, rng.integers(1, P, (n, n)) * bd]
        r2, np2 = jac_rank(fac2, msk1 + msk1)
        out["jacobian"].append(dict(m=m, dao_single=r1, mon_pred=2 * m ** 3 - m ** 2, dao_product=r2, params=np2,
                                    dense=m ** 4, four_m3_minus_3m2=4 * m ** 3 - 3 * m ** 2))
        print(out["jacobian"][-1], flush=True)
    for m in (3, 4):
        n = m * m
        Pm = perm_mat(m).astype(float)
        rg = np.random.default_rng(m)
        u, v = rg.standard_normal((m, m, m)), rg.standard_normal((m, m, m))  # u[a,b,e], v[a,b,c]
        M = np.zeros((n, n))
        for a in range(m):
            for b in range(m):
                for e in range(m):
                    for c in range(m):
                        M[a * m + e, b * m + c] = u[a, b, e] * v[a, b, c]
        L, R, S = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        for a in range(m):
            for b in range(m):
                ap = (a + b) % m
                L[a * m:(a + 1) * m, a * m + ap] = u[a, b]          # block a, column a' = a+b
                R[b * m + ap, b * m:(b + 1) * m] = v[a, b]           # block b, row a' = a+b
                S[ap * m + a, ap * m + b] = 1                        # block a', entry [a,b]
        err = float(np.abs(L @ Pm @ S @ Pm @ R - M).max())
        # rank-1 slices of M (M in Mon) and S invertibility not needed; record error
        out["latin_square"].append(dict(m=m, max_abs_err=err, ok=err < 1e-10))
        print(out["latin_square"][-1], flush=True)
    for m in (3,):
        n = m * m
        Pm = perm_mat(m).astype(float)
        rg = np.random.default_rng(11)
        bdr = lambda: np.block([[rg.standard_normal((m, m)) if i == j else np.zeros((m, m)) for j in range(m)] for i in range(m)])
        D = Pm @ bdr() @ Pm @ bdr()

        def maxslice(X):
            T = X.reshape(m, m, m, m)
            return max(np.linalg.matrix_rank(T[a, :, b, :], tol=1e-9) for a in range(m) for b in range(m))
        out["dao_closure"].append(dict(m=m, member=maxslice(Pm @ D), inverse=maxslice(Pm @ np.linalg.inv(D)),
                                       transpose=maxslice(Pm @ D.T)))
        print(out["dao_closure"][-1], flush=True)
    return out


# ---------------- repair round 2 (gate 5) ----------------
def jac_rank_exact(factors, masks):
    """Exact (Fraction) analogue of jac_rank for small sizes; factors are lists-of-lists, masks boolean arrays or None."""
    n = len(factors[0])
    eye = mat(n, n, lambda i, j: int(i == j))
    pre = [eye]
    for F in factors:
        pre.append(mul(pre[-1], F))
    suf = [eye]
    for F in reversed(factors):
        suf.append(mul(F, suf[-1]))
    suf = suf[::-1]
    rows = []
    for j, mk in enumerate(masks):
        if mk is None:
            continue
        L, R = pre[j], suf[j + 1]
        for r in range(n):
            for c in range(n):
                if mk[r][c]:
                    rows.append([L[i][r] * R[c][k] for i in range(n) for k in range(n)])
    return rank_q(rows), len(rows)


def m2_witness_suite():
    """Explicit witness for M2(c): L = R = I, S_{a'}[a,b] = 2^(a a' b) (block a', entry [a,b]); Jacobian rank of
    (L, S, R) -> L Pi S Pi R must be 3m^3 - 2m^2. mod-p for m = 2..5 (supplementary), exact over Q for m = 2, 3."""
    out = {"mod_p": [], "exact_Q": []}
    for m in (2, 3, 4, 5):
        n = m * m
        bd, Pm = blockdiag_mask(m), perm_mat(m)
        S = np.zeros((n, n), dtype=np.int64)
        for c in range(m):
            for a in range(m):
                for b in range(m):
                    S[c * m + a, c * m + b] = pow(2, a * b * c, P)
        I = np.eye(n, dtype=np.int64)
        rk, npar = jac_rank([I, Pm, S, Pm, I], [bd, None, bd, None, bd])
        want = 3 * m ** 3 - 2 * m ** 2
        out["mod_p"].append(dict(m=m, params=npar, jac_rank_mod_p=rk, predicted=want, kernel=npar - rk,
                                 gauge=2 * m * m, match=rk == want))
        print(out["mod_p"][-1], flush=True)
    for m in (2, 3):
        n = m * m
        bd = blockdiag_mask(m)
        Pm = mat(n, n, lambda i, j: int(j == (i % m) * m + i // m))
        S = mat(n, n, lambda i, j: 2 ** ((i % m) * (j % m) * (i // m)) if i // m == j // m else 0)
        eye = mat(n, n, lambda i, j: int(i == j))
        M = mul(mul(Pm, S), Pm)
        slices_ok = all(M[a * m + e][b * m + c] == (2 ** (a * e * b) if e == c else 0)
                        for a in range(m) for e in range(m) for b in range(m) for c in range(m))
        rk, npar = jac_rank_exact([eye, Pm, S, Pm, eye], [bd, None, bd, None, bd])
        want = 3 * m ** 3 - 2 * m ** 2
        out["exact_Q"].append(dict(m=m, params=npar, rank_over_Q=rk, predicted=want, kernel=npar - rk,
                                   gauge=2 * m * m, match=rk == want, witness_slice_form_ok=slices_ok))
        print(out["exact_Q"][-1], flush=True)
    return out


def _slices(T, m):
    return {(a, b): [[T[a * m + e][b * m + c] for c in range(m)] for e in range(m)] for a in range(m) for b in range(m)}


def _minors2(Sl, m):
    for e1 in range(m):
        for e2 in range(e1 + 1, m):
            for c1 in range(m):
                for c2 in range(c1 + 1, m):
                    yield (e1, e2, c1, c2), Sl[e1][c1] * Sl[e2][c2] - Sl[e1][c2] * Sl[e2][c1]


def _det3(S):
    return (S[0][0] * (S[1][1] * S[2][2] - S[1][2] * S[2][1]) - S[0][1] * (S[1][0] * S[2][2] - S[1][2] * S[2][0])
            + S[0][2] * (S[1][0] * S[2][1] - S[1][1] * S[2][0]))


def _first_offender(T, m):
    """First slice (a,b) of T (already Pi-premultiplied) with a nonzero 2x2 minor; prefers a rank-3 slice."""
    sl = _slices(T, m)
    for want_det in (True, False):
        for (a, b), S in sl.items():
            if want_det and _det3(S) == 0:
                continue
            for (e1, e2, c1, c2), v in _minors2(S, m):
                if v != 0:
                    return dict(slice=[a, b], rows=[e1, e2], cols=[c1, c2],
                                entries=[[str(S[e1][c1]), str(S[e1][c2])], [str(S[e2][c1]), str(S[e2][c2])]],
                                minor=str(v), slice_det3=str(_det3(S)))
    return None


def dao_exact_suite():
    """Exact certificate that Dao's class Pi.Mon = {Pi L Pi R} is not closed under inverse or transpose (m = 3):
    D = Pi L Pi R with integer blocks; the class test is 'all slices of Pi X have rank <= 1' (M1)."""
    m, n = 3, 9
    rnd = random.Random(20260929)

    def blk():
        while True:
            B = mat(m, m, lambda i, j: rnd.randint(-3, 3))
            if rank_q(B) == m:
                return B
    Lb = [blk() for _ in range(m)]
    Rb = [blk() for _ in range(m)]
    dg = lambda bl: mat(n, n, lambda i, j: bl[i // m][i % m][j % m] if i // m == j // m else 0)
    Lf, Rf = dg(Lb), dg(Rb)
    Pm = mat(n, n, lambda i, j: int(j == (i % m) * m + i // m))
    D = mul(mul(Pm, Lf), mul(Pm, Rf))
    Dinv = inv_q(D)
    Dt = [list(r) for r in zip(*D)]
    PD, PDinv, PDt = mul(Pm, D), mul(Pm, Dinv), mul(Pm, Dt)
    member_minors_zero = all(v == 0 for S in _slices(PD, m).values() for _, v in _minors2(S, m))
    cert = dict(
        m=m, seed=20260929,
        convention="index (a,e) -> a*m+e; Pi[(b,a),(a,b)] = 1; D = Pi L Pi R, L,R block diagonal with m blocks m x m; "
                   "closed form D[(i,j),(b,c)] = L_j[i][b] * R_b[j][c]; slice(a,b)[e][c] = (Pi X)[(a,e),(b,c)]; "
                   "X is in Dao's class iff every slice of Pi X has rank <= 1 (M1).",
        L_blocks=[[[str(x) for x in r] for r in B] for B in Lb], R_blocks=[[[str(x) for x in r] for r in B] for B in Rb],
        D=[[str(x) for x in r] for r in D], D_inverse=[[str(x) for x in r] for r in Dinv],
        D_times_Dinv_is_identity=mul(D, Dinv) == mat(n, n, lambda i, j: int(i == j)),
        member_D_all_2x2_minors_zero=member_minors_zero,
        inverse_offender=_first_offender(PDinv, m), transpose_offender=_first_offender(PDt, m),
        max_slice_rank={k: max(3 if _det3(S) != 0 else (2 if any(v != 0 for _, v in _minors2(S, m)) else
                                                        (1 if any(x != 0 for r in S for x in r) else 0))
                               for S in _slices(T, m).values())
                        for k, T in (("Pi_D", PD), ("Pi_Dinv", PDinv), ("Pi_Dtranspose", PDt))})
    print({k: cert[k] for k in ("D_times_Dinv_is_identity", "member_D_all_2x2_minors_zero", "max_slice_rank")}, flush=True)
    print("inverse offender:", cert["inverse_offender"], flush=True)
    print("transpose offender:", cert["transpose_offender"], flush=True)
    return cert


def verify_dao_exact(path):
    """Independent re-verification from the stored JSON only (closed-form D, entrywise sums, direct minors)."""
    d = json.load(open(path))
    m, n = d["m"], d["m"] ** 2
    L = [[[Fr(x) for x in r] for r in B] for B in d["L_blocks"]]
    R = [[[Fr(x) for x in r] for r in B] for B in d["R_blocks"]]
    D = [[L[j][i][b] * R[b][j][c] for b in range(m) for c in range(m)] for i in range(m) for j in range(m)]
    assert D == [[Fr(x) for x in r] for r in d["D"]], "stored D differs from closed form"
    Di = [[Fr(x) for x in r] for r in d["D_inverse"]]
    for i in range(n):
        for j in range(n):
            assert sum(D[i][k] * Di[k][j] for k in range(n)) == int(i == j), "D * D^-1 != I"
    swap = lambda idx: (idx % m) * m + idx // m  # Pi as an index map on (row) pairs: (Pi X)[i] = X[swap(i)]
    for key, X in (("inverse_offender", Di), ("transpose_offender", [list(r) for r in zip(*D)])):
        o = d[key]
        a, b = o["slice"]
        (e1, e2), (c1, c2) = o["rows"], o["cols"]
        g = lambda e, c: X[swap(a * m + e)][b * m + c]
        assert [[str(g(e1, c1)), str(g(e1, c2))], [str(g(e2, c1)), str(g(e2, c2))]] == o["entries"], key + " entries"
        minor = g(e1, c1) * g(e2, c2) - g(e1, c2) * g(e2, c1)
        assert str(minor) == o["minor"] and minor != 0, key + " minor"
    for a in range(m):
        for b in range(m):
            g = lambda e, c: D[swap(a * m + e)][b * m + c]
            assert all(g(e1, c1) * g(e2, c2) - g(e1, c2) * g(e2, c1) == 0
                       for e1 in range(m) for e2 in range(m) for c1 in range(m) for c2 in range(m)), "member not rank<=1"
    print("dao_exact independently verified from JSON", flush=True)


def ldr_kron_family_suite():
    """rank nabla_{Ze(x)Ze', Zf(x)Zf'}(I(x)I) = [e'!=f'](n1-1) + [e!=f](n2-1) + [ee'!=ff'] (R2-P3)."""
    out = []
    for (e, f, e2, f2), rng_ in [((2, 3, 5, 7), range(2, 7)), ((2, 3, 3, 2), range(2, 5)), ((2, 3, 4, 4), range(2, 5)),
                                 ((2, 2, 5, 7), range(2, 5))]:
        for n1 in rng_:
            for n2 in rng_:
                N = n1 * n2
                A, B = kron(Z(n1, Fr(e)), Z(n2, Fr(e2))), kron(Z(n1, Fr(f)), Z(n2, Fr(f2)))
                rk = rank_q(sub(A, B))  # nabla(I) = A I - I B
                pred = int(e2 != f2) * (n1 - 1) + int(e != f) * (n2 - 1) + int(e * e2 != f * f2)
                fac = (nabla(Z(n1, Fr(e)), mat(n1, n1, lambda i, j: int(i == j)), Z(n1, Fr(f))),
                       nabla(Z(n2, Fr(e2)), mat(n2, n2, lambda i, j: int(i == j)), Z(n2, Fr(f2))))
                out.append(dict(e=e, f=f, e2=e2, f2=f2, n1=n1, n2=n2, N=N, rank=rk, predicted=pred,
                                factor_ranks=list(fac), match=rk == pred))
        print("params", (e, f, e2, f2), "all match:", all(r["match"] for r in out if (r["e"], r["f"], r["e2"], r["f2"]) == (e, f, e2, f2)), flush=True)
    return out


def qs_kron_exact_suite():
    """order(J_k) = order(I_k + J_k) = 1 and order(J_k (x) (I_k + J_k)) = k (R2-P4); J_k = k x k nilpotent shift."""
    out = []
    for k in range(2, 7):
        J = mat(k, k, lambda i, j: int(i == j + 1))
        B = add(mat(k, k, lambda i, j: int(i == j)), J)
        K = kron(J, B)
        out.append(dict(k=k, order_J=qs_order(J), order_IplusJ=qs_order(B), order_kron=qs_order(K), predicted=k,
                        match=(qs_order(J), qs_order(B), qs_order(K)) == (1, 1, k)))
        print(out[-1], flush=True)
    return out


# ---------------- repair round 3 (cold no-memory referee, 2026-09-30) ----------------
def _pi(m):
    return mat(m * m, m * m, lambda i, j: int(j == (i % m) * m + i // m))


def _permrows(X, m):
    """Pi X on the m x m grid: (Pi X)[i] = X[swap(i)], swap((a,e)) = (e,a)."""
    return [X[(i % m) * m + i // m][:] for i in range(m * m)]


def _tr(A):
    return [list(r) for r in zip(*A)]


def _dg(bl, m):
    """Block-diagonal m^2 x m^2 matrix from a list of m blocks, each m x m."""
    return mat(m * m, m * m, lambda i, j: bl[i // m][i % m][j % m] if i // m == j // m else 0)


def _in_mon(X, m):
    """M1 membership test: X (m^2 x m^2) is in Mon_LPiR iff every slice X[(a,.),(b,.)] has all 2x2 minors zero."""
    return all(v == 0 for S in _slices(X, m).values() for _, v in _minors2(S, m))


def _rand_mon(m, rnd):
    u = {(a, b): [rnd.randint(1, 9) for _ in range(m)] for a in range(m) for b in range(m)}
    v = {(a, b): [rnd.randint(1, 9) for _ in range(m)] for a in range(m) for b in range(m)}
    return mat(m * m, m * m, lambda i, j: u[(i // m, j // m)][i % m] * v[(i // m, j // m)][j % m])


def _q_map(m1, m2):
    """Fixed reordering ((a1,e1),(a2,e2)) -> ((a1,a2),(e1,e2)) of the lexicographic index of M1 (x) M2."""
    m, n2 = m1 * m2, m2 * m2
    return {(a1 * m1 + e1) * n2 + a2 * m2 + e2: (a1 * m2 + a2) * m + e1 * m2 + e2
            for a1 in range(m1) for e1 in range(m1) for a2 in range(m2) for e2 in range(m2)}


def _reorder(K, q):
    """Q K Q^T for the permutation i -> q[i] (the same map on rows and columns)."""
    out = [[None] * len(K) for _ in K]
    for i in range(len(K)):
        for j in range(len(K)):
            out[q[i]][q[j]] = K[i][j]
    return out


def r3_kron_reorder():
    rnd = random.Random(20260930)
    out = []
    for m1, m2 in [(2, 2), (2, 3), (3, 2), (3, 3)]:
        m = m1 * m2
        M1, M2 = _rand_mon(m1, rnd), _rand_mon(m2, rnd)
        K, q = kron(M1, M2), _q_map(m1, m2)
        Pi1, Pi2 = _pi(m1), _pi(m2)
        Dq = _reorder(kron(mul(Pi1, M1), mul(Pi2, M2)), q)
        row = dict(m1=m1, m2=m2, m=m, sampler_in_Mon=_in_mon(M1, m1) and _in_mon(M2, m2),
                   q_is_permutation=sorted(q.values()) == list(range(m * m)),
                   reordered_kron_in_Mon=_in_mon(_reorder(K, q), m),
                   NEG_plain_lexicographic_kron_in_Mon=_in_mon(K, m),
                   Q_Pi1xPi2_Qt_equals_Pi=_reorder(kron(Pi1, Pi2), q) == _pi(m),
                   dao_class_kron_reordered_PiTimes_in_Mon=_in_mon(_permrows(Dq, m), m),
                   NEG_random_dense_in_Mon=_in_mon(mat(m * m, m * m, lambda i, j: rnd.randint(1, 9)), m))
        row["match"] = (row["sampler_in_Mon"] and row["q_is_permutation"] and row["reordered_kron_in_Mon"]
                        and not row["NEG_plain_lexicographic_kron_in_Mon"] and row["Q_Pi1xPi2_Qt_equals_Pi"]
                        and row["dao_class_kron_reordered_PiTimes_in_Mon"] and not row["NEG_random_dense_in_Mon"])
        out.append(row)
        print(row, flush=True)
    return out


def _chain(blocks, m):
    """B_0 Pi B_1 Pi ... Pi B_k for a list of block lists (each a list of m blocks)."""
    Pm = _pi(m)
    X = _dg(blocks[0], m)
    for bl in blocks[1:]:
        X = mul(mul(X, Pm), _dg(bl, m))
    return X


def _latin(Lb, Rb, m, shift=0):
    """Blocks (L', S, R') with L' Pi S Pi R' = L Pi R when shift = 0 (Latin-square step); shift != 0 is the wrong square."""
    Lp = [mat(m, m, lambda e, x: Lb[a][e][(x - a) % m]) for a in range(m)]
    Rp = [mat(m, m, lambda x, c: Rb[b][(x - b) % m][c]) for b in range(m)]
    Sb = [mat(m, m, lambda a, b: int((a + b + shift) % m == x)) for x in range(m)]
    return Lp, Sb, Rp


def r3_monotone():
    rnd = random.Random(20260931)
    out = []
    for m in (2, 3):
        for k in (1, 2, 3):
            blocks = [[mat(m, m, lambda i, j: rnd.randint(1, 9)) for _ in range(m)] for _ in range(k + 1)]
            X = _chain(blocks, m)
            same = {}
            for tag, shift in (("latin", 0), ("wrong_shift", 1)):
                Lp, Sb, Rp = _latin(blocks[k - 1], blocks[k], m, shift)
                same[tag] = _chain(blocks[:k - 1] + [Lp, Sb, Rp], m) == X
            row = dict(m=m, k=k, pis_before=k, pis_after=k + 1, latin_reproduces_X=same["latin"],
                       NEG_wrong_shift_reproduces_X=same["wrong_shift"], match=same["latin"] and not same["wrong_shift"])
            out.append(row)
            print(row, flush=True)
    return out


def r3_dao_piside():
    d = json.load(open(os.path.join(OUT, "dao_exact.json")))
    m = d["m"]
    Lb = [[[Fr(x) for x in r] for r in B] for B in d["L_blocks"]]
    Rb = [[[Fr(x) for x in r] for r in B] for B in d["R_blocks"]]
    L, R, Pm = _dg(Lb, m), _dg(Rb, m), _pi(m)
    D = mul(mul(Pm, L), mul(Pm, R))
    Dinv, Dt = [[Fr(x) for x in r] for r in d["D_inverse"]], _tr(D)
    Linv, Rinv = _dg([inv_q(B) for B in Lb], m), _dg([inv_q(B) for B in Rb], m)
    Lt, Rt = _dg([_tr(B) for B in Lb], m), _dg([_tr(B) for B in Rb], m)
    X1, X2 = mul(Dinv, Pm), mul(Dt, Pm)
    out = dict(stored_D_matches=D == [[Fr(x) for x in r] for r in d["D"]], D_in_Dao_class=_in_mon(_permrows(D, m), m),
               Dinv_Pi_equals_Rinv_Pi_Linv=X1 == mul(mul(Rinv, Pm), Linv),
               Dt_Pi_equals_Rt_Pi_Lt=X2 == mul(mul(Rt, Pm), Lt),
               Dinv_Pi_in_Mon=_in_mon(X1, m), Dt_Pi_in_Mon=_in_mon(X2, m),
               NEG_Pi_Dinv_in_Mon=_in_mon(_permrows(Dinv, m), m), NEG_Pi_Dt_in_Mon=_in_mon(_permrows(Dt, m), m),
               recorded_Dinv_in_Mon_plain=_in_mon(Dinv, m), recorded_Dt_in_Mon_plain=_in_mon(Dt, m))
    out["match"] = (out["stored_D_matches"] and out["D_in_Dao_class"] and out["Dinv_Pi_equals_Rinv_Pi_Linv"]
                    and out["Dt_Pi_equals_Rt_Pi_Lt"] and out["Dinv_Pi_in_Mon"] and out["Dt_Pi_in_Mon"]
                    and not out["NEG_Pi_Dinv_in_Mon"] and not out["NEG_Pi_Dt_in_Mon"])
    print(out, flush=True)
    return out


def r3_slice_formula():
    rnd = random.Random(20260932)
    out = []
    for m in (2, 3, 4):
        Pm = _pi(m)
        rb = lambda: [mat(m, m, lambda i, j: rnd.randint(-4, 4)) for _ in range(m)]
        Lb, Sb, Rb = rb(), rb(), rb()
        M = mul(mul(mul(_dg(Lb, m), Pm), mul(_dg(Sb, m), Pm)), _dg(Rb, m))
        ok, wrong_fails = True, False
        for a in range(m):
            for b in range(m):
                sl = [[M[a * m + e][b * m + c] for c in range(m)] for e in range(m)]
                dgn = mat(m, m, lambda x, y: Sb[x][a][b] if x == y else 0)  # diag over a' of S_{a'}[a,b]
                ok = ok and sl == mul(mul(Lb[a], dgn), Rb[b])
                wrong_fails = wrong_fails or sl != mul(mul(Lb[a], dgn), Rb[a])
        out.append(dict(m=m, formula_holds_all_slices=ok, NEG_variant_R_a_fails=wrong_fails, match=ok and wrong_fails))
        print(out[-1], flush=True)
    return out


def _nabla_rank(A, B):
    """Rank of the linear map M -> A M - M B on N x N matrices (exact)."""
    N = len(A)
    imgs = []
    for i in range(N):
        for j in range(N):
            E = mat(N, N, lambda p, q: int(p == i and q == j))
            imgs.append([x for r in sub(mul(A, E), mul(E, B)) for x in r])
    return rank_q(imgs)


def r3_reconstructible():
    out = {"bijective_family": []}
    for n1, n2 in [(2, 2), (2, 3), (3, 2), (3, 3)]:
        A, B = kron(Z(n1, Fr(2)), Z(n2, Fr(5))), kron(Z(n1, Fr(3)), Z(n2, Fr(7)))
        N = n1 * n2
        rk = _nabla_rank(A, B)
        out["bijective_family"].append(dict(n1=n1, n2=n2, N=N, rank=rk, full=N * N, match=rk == N * N))
        print(out["bijective_family"][-1], flush=True)
    A, B = kron(Z(2, Fr(2)), Z(2, Fr(3))), kron(Z(2, Fr(3)), Z(2, Fr(2)))
    rk = _nabla_rank(A, B)
    out["NEG_common_spectrum_ee'=ff'"] = dict(params="(e,f,e',f')=(2,3,3,2), n1=n2=2", rank=rk, full=16, match=rk < 16)
    I4 = mat(4, 4, lambda i, j: int(i == j))
    rk0 = _nabla_rank(I4, I4)
    out["trivial_A_eq_B_eq_I"] = dict(params="A = B = I_4", rank=rk0, match=rk0 == 0)
    print(out["NEG_common_spectrum_ee'=ff'"], out["trivial_A_eq_B_eq_I"], flush=True)
    return out


def r3_closed_algebras():
    rnd = random.Random(20260933)
    circ = lambda c: mat(len(c), len(c), lambda i, j: c[(i - j) % len(c)])
    is_circ = lambda K: all(K[i][j] == K[(i + 1) % len(K)][(j + 1) % len(K)] for i in range(len(K)) for j in range(len(K)))
    out = {"circulant": [], "blockdiag": []}
    for n1, n2 in [(2, 3), (3, 2), (3, 4), (4, 3), (2, 2), (2, 4)]:
        K = kron(circ([rnd.randint(1, 9) for _ in range(n1)]), circ([rnd.randint(1, 9) for _ in range(n2)]))
        N, cop = n1 * n2, math.gcd(n1, n2) == 1
        row = dict(n1=n1, n2=n2, coprime=cop, NEG_plain_lexicographic_is_circulant=is_circ(K))
        if cop:
            crt = {i1 * n2 + i2: next(r for r in range(N) if r % n1 == i1 and r % n2 == i2)
                   for i1 in range(n1) for i2 in range(n2)}
            row["crt_reordered_is_circulant"] = is_circ(_reorder(K, crt))
        row["match"] = (not row["NEG_plain_lexicographic_is_circulant"]) and (row["crt_reordered_is_circulant"] if cop else True)
        out["circulant"].append(row)
        print(row, flush=True)

    def bdrand(sizes):
        M, blocks, off = mat(sum(sizes), sum(sizes), lambda i, j: 0), [], 0
        for s in sizes:
            b = mat(s, s, lambda i, j: rnd.randint(1, 9))
            for i in range(s):
                for j in range(s):
                    M[off + i][off + j] = b[i][j]
            blocks.append(b)
            off += s
        return M, blocks

    def contiguous_ok(K, sizes):
        which, o = {}, 0
        for t, s in enumerate(sizes):
            which.update({i: t for i in range(o, o + s)})
            o += s
        return all(K[i][j] == 0 for i in range(len(K)) for j in range(len(K)) if which[i] != which[j])

    for sA, sB in [((2, 3), (1, 2)), ((1, 1, 2), (3, 2))]:
        A, bA = bdrand(sA)
        B, bB = bdrand(sB)
        nB = len(B)
        idA = [(t, x) for t, s in enumerate(sA) for x in range(s)]
        idB = [(t, y) for t, s in enumerate(sB) for y in range(s)]
        K = kron(A, B)
        order = sorted(range(len(A) * nB), key=lambda i: (idA[i // nB][0], idB[i % nB][0], idA[i // nB][1], idB[i % nB][1]))
        K2 = _reorder(K, {old: new for new, old in enumerate(order)})
        sizes = [a * b for a in sA for b in sB]
        blocks_equal, o = True, 0
        for i, a in enumerate(sA):
            for j, b in enumerate(sB):
                blocks_equal = blocks_equal and [r[o:o + a * b] for r in K2[o:o + a * b]] == kron(bA[i], bB[j])
                o += a * b
        row = dict(sizes_A=list(sA), sizes_B=list(sB), reordered_contiguous_blocks=contiguous_ok(K2, sizes),
                   diagonal_blocks_equal_kron_Ai_Bj=blocks_equal, NEG_plain_order_contiguous_blocks=contiguous_ok(K, sizes))
        row["match"] = row["reordered_contiguous_blocks"] and blocks_equal and not row["NEG_plain_order_contiguous_blocks"]
        out["blockdiag"].append(row)
        print(row, flush=True)
    return out


def r3_helper_controls():
    """Positive/negative controls for the helpers used by the R3 suites."""
    rng = np.random.default_rng(3)
    A = (rng.integers(0, P, (30, 17)) @ rng.integers(0, P, (17, 40))) % P  # rank 17 (mod p, w.h.p.)
    X = mat(9, 9, lambda i, j: (i * 7 + j * 3) % 5)
    out = dict(rank_fwd=rank_mod_fwd(A), rank_full=rank_mod(A), expected_rank=17,
               zero_matrix_rank=rank_mod_fwd(np.zeros((5, 5), dtype=np.int64)),
               permrows_equals_Pi_mul=_permrows(X, 3) == mul(_pi(3), X))
    out["match"] = out["rank_fwd"] == out["rank_full"] == 17 and out["zero_matrix_rank"] == 0 and out["permrows_equals_Pi_mul"]
    print(out, flush=True)
    return out


def _all_match(o):
    if isinstance(o, dict):
        return all(_all_match(v) for v in o.values()) and o.get("match", True) is not False
    if isinstance(o, list):
        return all(_all_match(v) for v in o)
    return True


def r3_suite():
    out = dict(helper_controls=r3_helper_controls(), kron_reorder=r3_kron_reorder(), monotone=r3_monotone(),
               dao_piside=r3_dao_piside(), slice_formula=r3_slice_formula(), reconstructible=r3_reconstructible(),
               closed_algebras=r3_closed_algebras())
    out["all_match"] = _all_match(out)
    print("r3 all_match:", out["all_match"], flush=True)
    return out


def monarch_extra_suite():
    return monarch_suite(MONARCH_EXTRA_PAIRS, 7, rank_mod_fwd)


def butterfly_extra_suite():
    return butterfly_suite((32,), (1, 2, 6, 7), 9, rank_mod_fwd)


SUITES = dict(monarch=monarch_suite, butterfly=butterfly_suite, ldr=ldr_suite, dao=dao_suite, m2_witness=m2_witness_suite,
              dao_exact=dao_exact_suite, ldr_kron_family=ldr_kron_family_suite, qs_kron_exact=qs_kron_exact_suite,
              r3=r3_suite, monarch_extra=monarch_extra_suite, butterfly_extra=butterfly_extra_suite)

if __name__ == "__main__":
    suite = sys.argv[1]
    t0 = time.time()
    res = SUITES[suite]()
    print(f"[{suite}] {time.time() - t0:.1f} s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, suite + ".json")
    json.dump(res, open(path, "w"), default=int, indent=1)
    if suite == "ldr":
        print(json.dumps(res, default=int, indent=1))
    if suite == "dao_exact":
        verify_dao_exact(path)
