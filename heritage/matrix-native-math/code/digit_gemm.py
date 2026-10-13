"""digit_gemm_complexity_v1: exact lower bounds and exhaustive searches for the number of low-precision GEMMs
needed to compute one exact higher-precision matrix product.

Unit doc: research/DIGIT_GEMM_COMPLEXITY_V1.md (models M1, M2, M3, M4, Mpoly; lemmas and theorems cited by name).
Data: results/digit_gemm_complexity_v1/.

Subcommands (python code/digit_gemm.py <cmd>):
  awe        reproduce AWE's INT8xINT8 (6) and INT4xINT8 (4) schemes exactly (identity, value set, coverage, K*)
  ell        exact ell(V, X): least j with theta . V^j covering a symmetric progression of length 2X+1
  rows       candidate dependent rows (ratio lemma) and |G(f)| for the tight-case search
  tight      exhaustive decision of the tight case r = ell_A + ell_B - 1 (MDS structure theorem)
  table      lower-bound table vs AWE's published counts
  modular    per-modulus bounds / constructions and moduli-selection integer programs (FP4, FP8, INT8)
  mpoly      SAT/ILP minimum rank of the value-constrained polynomial-product tensor
All arithmetic that decides a claim is exact (integers / fractions). Floating point is used only for screening,
and every screened-in candidate is re-verified exactly.
"""
import itertools
import json
import math
import os
import sys
from fractions import Fraction

import numpy as np

OUT = os.path.join("results", "digit_gemm_complexity_v1")

# ----------------------------------------------------------------------------------------------------------
# Value sets (each: sorted tuple of ints; a per-plane scale is absorbed into the reconstruction coefficients)
# ----------------------------------------------------------------------------------------------------------
S_FP4 = (-12, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12)          # 2 x E2M1 (AWE section III-A)
V_FP8_16 = tuple(range(-16, 17))                                     # E4M3 integers with |v| <= 16 (2603.10634)
V_INT8 = tuple(range(-128, 128))
V_INT4 = tuple(range(-8, 8))


def e4m3_integers(cap):
    """All integers representable in FP8 E4M3 (max finite 448) with |v| <= cap."""
    vals = {0}
    for e in range(-9, 9):                       # normals 2^-6..2^8 with 3 mantissa bits, subnormals below
        for m in range(8, 16):
            v = Fraction(m, 8) * Fraction(2) ** e
            if v.denominator == 1 and v <= 448:
                vals.add(int(v))
    return tuple(sorted(x for x in {v for v in vals} | {-v for v in vals} if abs(x) <= cap))


def width_range(M):
    """AWE convention: integer width M is the symmetric range [-(2^(M-1)-1), 2^(M-1)-1]; returns X."""
    return 2 ** (M - 1) - 1


def diff_ratio_set(V):
    """R = D/D with D = (V - V) minus 0, as a set of Fractions (ratio lemma)."""
    D = sorted({a - b for a in V for b in V if a != b})
    return {Fraction(p, q) for p in D for q in D}


def _save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=str)


# ----------------------------------------------------------------------------------------------------------
# Exact verification of limb/plane schemes in AWE's Definition IV.1 form (planes linear in limbs)
# ----------------------------------------------------------------------------------------------------------
def admissible(V, planes, p):
    """T = {a in V^p : alpha_k . a in V for every plane alpha_k} (planes: list of integer/Fraction tuples)."""
    Vs = set(V)
    return [a for a in itertools.product(V, repeat=p)
            if all(sum(Fraction(c) * x for c, x in zip(al, a)) in Vs for al in planes)]


def coverage(weights, T):
    """Largest X such that {w . a : a in T} contains every integer in [-X, X]."""
    vals = {sum(Fraction(w) * x for w, x in zip(weights, a)) for a in T}
    X = 0
    while Fraction(X + 1) in vals and Fraction(-(X + 1)) in vals:
        X += 1
    return X if 0 in vals else -1


def verify_limb_scheme(V, alpha, beta, lam, w, v, XA, XB, check_all_pairs=True):
    """Exact check of a scheme: matrix identity sum lam_k alpha_k beta_k^T = w v^T, coverages >= targets, and the
    scalar identity over every admissible pair (independent of the matrix identity), plus c_A, c_B and K*."""
    p, q = len(w), len(v)
    lam = [Fraction(x) for x in lam]
    Mx = [[sum(lam[k] * Fraction(alpha[k][i]) * Fraction(beta[k][j]) for k in range(len(lam)))
           for j in range(q)] for i in range(p)]
    mat_ok = all(Mx[i][j] == Fraction(w[i]) * Fraction(v[j]) for i in range(p) for j in range(q))
    TA, TB = admissible(V, alpha, p), admissible(V, beta, q)
    covA, covB = coverage(w, TA), coverage(v, TB)
    # representatives used for the target ranges: one admissible tuple per value
    repA, repB = {}, {}
    for a in TA:
        repA.setdefault(sum(Fraction(x) * y for x, y in zip(w, a)), a)
    for b in TB:
        repB.setdefault(sum(Fraction(x) * y for x, y in zip(v, b)), b)
    pairs_ok = None
    if check_all_pairs:
        PA = [[sum(Fraction(c) * y for c, y in zip(al, a)) for al in alpha] for a in TA]
        QB = [[sum(Fraction(c) * y for c, y in zip(be, b)) for be in beta] for b in TB]
        xa = [sum(Fraction(x) * y for x, y in zip(w, a)) for a in TA]
        yb = [sum(Fraction(x) * y for x, y in zip(v, b)) for b in TB]
        # vectorised exact check with integers: scale lam to a common denominator
        Dn = math.lcm(*[x.denominator for x in lam])
        L = [int(x * Dn) for x in lam]
        PAi = np.array([[int(x) for x in row] for row in PA], dtype=np.int64)
        QBi = np.array([[int(x) for x in row] for row in QB], dtype=np.int64)
        lhs = (PAi * np.array(L, dtype=np.int64)) @ QBi.T
        rhs = Dn * np.outer(np.array([int(x) for x in xa], dtype=np.int64), np.array([int(x) for x in yb], dtype=np.int64))
        pairs_ok = bool(np.array_equal(lhs, rhs))
    cA = max(abs(int(x)) for a in TA for x in [sum(Fraction(c) * y for c, y in zip(al, a)) for al in alpha]
             if abs(sum(Fraction(z) * y for z, y in zip(w, a))) <= XA)
    cB = max(abs(int(x)) for b in TB for x in [sum(Fraction(c) * y for c, y in zip(be, b)) for be in beta]
             if abs(sum(Fraction(z) * y for z, y in zip(v, b))) <= XB)
    return {"matrix_identity": mat_ok, "pairs_identity": pairs_ok, "|T_A|": len(TA), "|T_B|": len(TB),
            "coverage_A": covA, "coverage_B": covB, "target_A": XA, "target_B": XB,
            "covers": covA >= XA and covB >= XB, "c_A_max_all_admissible": cA, "c_B_max_all_admissible": cB,
            "K_star_fp32_all_admissible": (2 ** 24) // (cA * cB), "r": len(lam)}


def cmd_awe():
    """Reproduce AWE Appendix A: INT8xINT8 in 6 products and INT4xINT8 in 4 products."""
    S = S_FP4
    e = lambda *c: tuple(c)
    # INT8 x INT8: w = v = (1, 29, 37); planes a_i and a_i + a_j (order P00, P11, P22, P01, P02, P12)
    planes = [e(1, 0, 0), e(0, 1, 0), e(0, 0, 1), e(1, 1, 0), e(1, 0, 1), e(0, 1, 1)]
    lam = [-65, -261, 259, 29, 37, 1073]
    r1 = verify_limb_scheme(S, planes, planes, lam, (1, 29, 37), (1, 29, 37), 127, 127)
    alt = verify_limb_scheme(S, planes, planes, [-49, -319, 1053, 11, 39, 429], (1, 11, 39), (1, 11, 39), 127, 127)
    # INT4 x INT8: A = 3a0 + 5a1, B = 5b0 + 3b1 + 42b2; 2AB = 35P1 - 42P2 + 210P3 + 15P4
    al4 = [e(0, 1), e(1, 0), e(1, 2), e(2, 1)]
    be4 = [e(1, 0, 0), e(0, 1, -1), e(0, 0, 1), e(1, 2, 0)]
    r2 = verify_limb_scheme(S, al4, be4, [Fraction(35, 2), Fraction(-42, 2), Fraction(210, 2), Fraction(15, 2)],
                            (3, 5), (5, 3, 42), 7, 127)
    out = {"INT8xINT8_6_w(1,29,37)": r1, "INT8xINT8_6_w(1,11,39)": alt, "INT4xINT8_4": r2}
    _save("awe_reproduction.json", out)
    print(json.dumps(out, indent=1, default=str))


# ----------------------------------------------------------------------------------------------------------
# Tight case: dependent rows (ratio lemma) and the point sets G(f) = {u in V^rho : f.u in V}
# ----------------------------------------------------------------------------------------------------------
def box_points(V, rho):
    """All u in V^rho as an int64 array (rows in itertools.product order)."""
    return np.array(list(itertools.product(V, repeat=rho)), dtype=np.int64)


def row_mask(V, U, f):
    """Boolean mask over the rows u of U with f.u in V (exact: f scaled to integers)."""
    L = math.lcm(*[Fraction(x).denominator for x in f])
    F = np.array([int(Fraction(x) * L) for x in f], dtype=np.int64)
    return np.isin(U @ F, np.array([L * x for x in V], dtype=np.int64))


def candidate_rows(V, rho, N):
    """Canonical dependent rows f (all entries > 0, sorted descending) that can occur in a tight scheme:
    every ratio among (f_1..f_rho, -1) lies in R = D/D (ratio lemma) and |G(f)| >= N.  Returns list of (f, |G|)."""
    R = diff_ratio_set(V)
    Rp = sorted(x for x in R if x > 0)
    Rs = set(R)
    U = box_points(V, rho)
    out = []
    for f in itertools.combinations_with_replacement(sorted(Rp, reverse=True), rho):
        if any((a / b) not in Rs for a, b in itertools.combinations(f, 2)):
            continue
        g = int(row_mask(V, U, f).sum())
        if g >= N:
            out.append((f, g))
    return out, len(Rp)


def row_variants(f):
    """All sign/permutation variants of a row, normalised so the first entry is positive (a plane sign flip is a
    symmetry because V = -V)."""
    out = set()
    for perm in itertools.permutations(f):
        for signs in itertools.product((1, -1), repeat=len(f)):
            g = tuple(s * x for s, x in zip(signs, perm))
            if g[0] < 0:
                g = tuple(-x for x in g)
            out.add(g)
    return sorted(out)


def systematic_generator(rows, ell):
    """Generator matrix (r x ell, rows = coordinate functionals) of the code {(u, f1.u, f2.u, ...)}."""
    return [tuple(Fraction(int(i == j)) for j in range(ell)) for i in range(ell)] + [tuple(Fraction(x) for x in f) for f in rows]


def _nullvec(rowsM, n):
    """A nonzero vector in the right null space of the (exact) matrix rowsM (list of length-n Fraction rows);
    assumes the null space is 1-dimensional (returns None otherwise)."""
    import sympy
    M = sympy.Matrix([[sympy.Rational(x.numerator, x.denominator) for x in r] for r in rowsM]) if rowsM else sympy.zeros(0, n)
    ns = M.nullspace() if rowsM else [sympy.eye(n)[:, i] for i in range(n)]
    if len(ns) != 1:
        return None
    v = ns[0]
    return tuple(Fraction(int(sympy.fraction(x)[0]), int(sympy.fraction(x)[1])) for x in v)


def min_weight_words(G, ell):
    """Primal minimal-weight codewords of the MDS code with generator G (r x ell): for each (ell-1)-subset Z of
    coordinates, the codeword G u with (G u)_z = 0 for z in Z. Returns list of (Z, word) or None if not MDS."""
    r = len(G)
    out = []
    for Z in itertools.combinations(range(r), ell - 1):
        u = _nullvec([G[z] for z in Z], ell)
        if u is None:
            return None
        w = tuple(sum(G[i][j] * u[j] for j in range(ell)) for i in range(r))
        if sum(1 for x in w if x != 0) != r - ell + 1:
            return None
        out.append((Z, w))
    return out


def ratio_ok(word, Rs):
    nz = [x for x in word if x != 0]
    return all((a / b) in Rs for a, b in itertools.combinations(nz, 2))


def _packbits_rows(masks):
    """Pack a (n, m) boolean array into (n, ceil(m/64)) uint64 words."""
    n, m = masks.shape
    pad = (-m) % 64
    b = np.packbits(np.concatenate([masks, np.zeros((n, pad), bool)], axis=1), axis=1, bitorder="little")
    return b.view(np.uint64)


def good_codes_53(V=S_FP4, N=255, save=True):
    """All [5,3] MDS plane codes U = {(u, f.u, g.u)} (systematic over coordinates 1,2,3) that a TIGHT r = 5 scheme can
    use on one side with N input values (section 3, Lemmas R1 and R2): entries of f, g in R, f_k/g_k in R, all 2x2
    minors m_i nonzero with m_a/m_b in R and m_i/f_j, m_i/g_j in R (j != i), and |X| = |{u in V^3 : f.u, g.u in V}| >= N.
    f runs over canonical rows (u-coordinate permutations and signs fixed), g over all labelled rows.
    Returns list of (f, g, |X|)."""
    R = sorted(diff_ratio_set(V))
    Rs = set(R)
    idx = {x: i for i, x in enumerate(R)}
    QR = np.array([[(a / b) in Rs for b in R] for a in R], dtype=bool)
    canon, _ = candidate_rows(V, 3, N)
    U = box_points(V, 3)
    lab = sorted({g for f, _ in canon for g in row_variants(f)})
    G = np.array([[idx[x] for x in g] for g in lab], dtype=np.int64)
    bits = _packbits_rows(np.array([row_mask(V, U, g) for g in lab]))
    labpos = {g: i for i, g in enumerate(lab)}
    out = []
    n_col = n_bits = 0
    for f, _ in canon:
        a = [idx[x] for x in f]
        ok = QR[a[0], G[:, 0]] & QR[a[1], G[:, 1]] & QR[a[2], G[:, 2]]          # f_k / g_k in R
        cand = np.nonzero(ok)[0]
        n_col += len(cand)
        if len(cand) == 0:
            continue
        fb = bits[labpos[f] if f in labpos else labpos[row_variants(f)[0]]]
        cnt = np.bitwise_count(bits[cand] & fb).sum(axis=1)
        for i in cand[cnt >= N]:
            n_bits += 1
            g = lab[i]
            m = (f[1] * g[2] - f[2] * g[1], f[2] * g[0] - f[0] * g[2], f[0] * g[1] - f[1] * g[0])
            if any(x == 0 for x in m):
                continue
            if not all((m[x] / m[y]) in Rs for x, y in itertools.combinations(range(3), 2)):
                continue
            if not all((m[i2] / f[j]) in Rs and (m[i2] / g[j]) in Rs for i2 in range(3) for j in range(3) if j != i2):
                continue
            out.append((f, g, int(np.bitwise_count(bits[i] & fb).sum())))
    info = {"V": list(V), "N": N, "canonical_rows": len(canon), "labelled_rows": len(lab),
            "pairs_passing_column_ratios": int(n_col), "pairs_with_|X|>=N": int(n_bits), "R_good_codes": len(out)}
    if save:
        _save("codes53_N%d.json" % N, {"info": info, "codes": [([str(x) for x in f], [str(x) for x in g], s) for f, g, s in out]})
    return out, info


def load_codes53(N=255):
    d = json.load(open(os.path.join(OUT, "codes53_N%d.json" % N), encoding="utf-8"))
    return [(tuple(Fraction(x) for x in f), tuple(Fraction(x) for x in g), s) for f, g, s in d["codes"]]


def plucker_key_ok(f, g):
    """Orbit dedupe (coordinate permutations x sign flips act on the 5 coordinates of the plane code).
    Plucker coordinates of U = rowspace-dual of (e1,e2,e3,f,g): p_T = det of the rows in T.  The systematic form over
    triple T has |entries| = |p_(T-i+j)| / |p_T|; key(T) = sorted multiset of these 6 ratios.  Keep a listing iff the
    free triple {1,2,3} minimises the key (ties kept) -- every orbit keeps >= 1 listing (safe dedupe)."""
    m = (f[1] * g[2] - f[2] * g[1], f[2] * g[0] - f[0] * g[2], f[0] * g[1] - f[1] * g[0])
    p = {(0, 1, 2): Fraction(1), (0, 1, 3): f[2], (0, 1, 4): g[2], (0, 2, 3): -f[1], (0, 2, 4): -g[1],
         (1, 2, 3): f[0], (1, 2, 4): g[0], (0, 3, 4): m[0], (1, 3, 4): m[1], (2, 3, 4): m[2]}
    keys = {}
    for T in p:
        rest = [j for j in range(5) if j not in T]
        ks = sorted(abs(p[tuple(sorted(set(T) - {i} | {j}))]) / abs(p[T]) for i in T for j in rest)
        keys[T] = tuple(ks)
    return keys[(0, 1, 2)] == min(keys.values())


def _prim(v):
    L = math.lcm(*[Fraction(x).denominator for x in v])
    w = [int(Fraction(x) * L) for x in v]
    gg = math.gcd(*w)
    return tuple(x // gg for x in w)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _egcd(a, b):
    if b == 0:
        return (abs(a), (1 if a >= 0 else -1), 0)
    g, x, y = _egcd(b, a % b)
    return (g, y, x - (a // b) * y)


def plane_lattice(u):
    """For primitive integer u, return (n0, b1, b2) with n0.u = 1 and {b1, b2} a basis of {n in Z^3 : n.u = 0}."""
    a, b, c = u
    if a == 0 and b == 0:
        s = 1 if c > 0 else -1
        return (0, 0, s), (1, 0, 0), (0, 1, 0)
    g1, x, y = _egcd(a, b)
    g2, s, t = _egcd(g1, c)
    assert g2 == 1
    n0 = (x * s, y * s, t)
    b1 = (b // g1, -a // g1, 0)
    b2 = (c * x, c * y, -g1)
    return n0, b1, b2


def covering_functionals(X, Dirs, q, K=127, want_all=True):
    """All integer n with n.u = q*k solvable in X for every k = 1..K (so c = n/q covers [-K, K] with step 1, X = -X)
    and |n.d| <= 2*K*q for every collision direction d in Dirs (pigeonhole bound, section 3).  X: (m,3) int array.
    Complete: the representative of value 1 is some u in X, so n lies on a lattice plane n.u = q (u in X)."""
    B = 2 * K * q
    Dm = np.array(Dirs, dtype=np.int64)
    cands = []
    # Float bounding with a safety margin of 2 lattice steps generates a SUPERSET of the lattice points of the
    # polygon; the exact integer filter |n.d| <= B below removes the excess, so completeness is not affected.
    for u in {tuple(int(t) for t in r) for r in X if any(r)}:
        gu = math.gcd(*u)
        if q % gu:
            continue
        up = tuple(t // gu for t in u)
        n0, b1, b2 = plane_lattice(up)
        n0 = np.array(n0, dtype=np.int64) * (q // gu)
        b1, b2 = np.array(b1, dtype=np.int64), np.array(b2, dtype=np.int64)
        c0, c1, c2 = (Dm @ n0).astype(float), (Dm @ b1).astype(float), (Dm @ b2).astype(float)
        P1 = np.concatenate([c1, c1]); P2 = np.concatenate([c2, c2]); RR = np.concatenate([B - c0, -B - c0])
        i1, i2 = np.triu_indices(len(P1), 1)
        det = P1[i1] * P2[i2] - P2[i1] * P1[i2]
        nz = np.abs(det) > 1e-12
        if not nz.any():
            continue
        av = (RR[i1] * P2[i2] - P2[i1] * RR[i2])[nz] / det[nz]
        bv = (P1[i1] * RR[i2] - RR[i1] * P1[i2])[nz] / det[nz]
        feas = np.all(np.abs(c0[None, :] + av[:, None] * c1[None, :] + bv[:, None] * c2[None, :]) <= B * (1 + 1e-9) + 1e-6, axis=1)
        if not feas.any():
            continue
        amin, amax = av[feas].min(), av[feas].max()
        A_ = np.arange(math.floor(amin) - 2, math.ceil(amax) + 3)
        base = c0[None, :] + A_[:, None] * c1[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            l1 = (-B - base) / c2[None, :]
            l2 = (B - base) / c2[None, :]
        lo = np.where(c2[None, :] != 0, np.minimum(l1, l2), -np.inf).max(axis=1)
        hi = np.where(c2[None, :] != 0, np.maximum(l1, l2), np.inf).min(axis=1)
        for a, l, h in zip(A_, lo, hi):
            if not (np.isfinite(l) and np.isfinite(h)) or h < l - 2:
                continue
            bs = np.arange(math.floor(l) - 2, math.ceil(h) + 3, dtype=np.int64)
            cands.append(n0[None, :] + int(a) * b1[None, :] + bs[:, None] * b2[None, :])
    if not cands:
        return np.zeros((0, 3), np.int64)
    N = np.unique(np.concatenate(cands), axis=0)
    N = N[np.all(np.abs(N @ Dm.T) <= B, axis=1)]                     # exact integer polytope filter
    alive = np.ones(len(N), bool)
    for k in range(2, K + 1):
        idx = np.nonzero(alive)[0]
        if len(idx) == 0:
            break
        for s in range(0, len(idx), 20000):
            sl = idx[s:s + 20000]
            alive[sl] = (N[sl] @ X.T == k * q).any(axis=1)
    return N[alive]


def _vegcd(a, b):
    """Vectorised extended gcd: returns g, x, y with a x + b y = g >= 0 (int64 arrays)."""
    a = a.astype(np.int64).copy(); b = b.astype(np.int64).copy()
    x0, y0, x1, y1 = np.ones_like(a), np.zeros_like(a), np.zeros_like(a), np.ones_like(a)
    while np.any(b != 0):
        nz = b != 0
        qd = np.zeros_like(a)
        qd[nz] = np.floor_divide(a[nz], b[nz])
        a, b = np.where(nz, b, a), np.where(nz, a - qd * b, b)
        x0, x1 = np.where(nz, x1, x0), np.where(nz, x0 - qd * x1, x1)
        y0, y1 = np.where(nz, y1, y0), np.where(nz, y0 - qd * y1, y1)
    s = np.where(a < 0, -1, 1)
    return a * s, x0 * s, y0 * s


SIEVE = 6
SIEVE_NB = 4      # non-full sieve values per line in the compiled kernel

try:
    import numba as _nb

    @_nb.njit(cache=True)
    def _egcd_nb(a, b):
        x0, y0, x1, y1 = 1, 0, 0, 1
        while b != 0:
            qd = a // b
            a, b = b, a - qd * b
            x0, x1 = x1, x0 - qd * x1
            y0, y1 = y1, y0 - qd * y1
        if a < 0:
            return -a, -x0, -y0
        return a, x0, y0

    @_nb.njit(cache=True)
    def _t_values(bu, vu, kq, lo, hi, m):
        """Sorted t in [lo, hi] with bu[i] + t vu[i] = kq for some i; returns (array, full_line_flag)."""
        out = np.empty(m, np.int64)
        c = 0
        full = False
        for i in range(m):
            if vu[i] == 0:
                if bu[i] == kq:
                    full = True
            else:
                num = kq - bu[i]
                if num % vu[i] == 0:
                    t = num // vu[i]
                    if lo <= t <= hi:
                        out[c] = t
                        c += 1
        return np.unique(out[:c]), full

    @_nb.njit(cache=True)
    def _covers(n0, n1, n2, Xa, q, K):
        present = np.zeros(K + 1, np.bool_)
        for i in range(Xa.shape[0]):
            val = n0 * Xa[i, 0] + n1 * Xa[i, 1] + n2 * Xa[i, 2]
            if val > 0 and val <= K * q and val % q == 0:
                present[val // q] = True
        for k in range(1, K + 1):
            if not present[k]:
                return False
        return True

    @_nb.njit(cache=True)
    def _lines_kernel(u1, nb, b1, b2, Xa, Dm, q, K, B, auto):
        """All covering n (up to sign fixed by u1 -> +q) on the lattice lines n.u1 = q, n.u2 = k2 q."""
        m = Xa.shape[0]
        nd = Dm.shape[0]
        res = np.empty((0, 3), np.int64)
        found = []
        WMAX = 1 << 16
        bu = np.empty(m, np.int64)
        vu = np.empty(m, np.int64)
        cntarr = np.zeros(WMAX, np.int32)
        stamparr = np.zeros(WMAX, np.int32)
        touched = np.empty(m, np.int64)
        for j in range(m):
            u2 = Xa[j]
            cx = u2[1] * u1[2] - u2[2] * u1[1]
            cy = u2[2] * u1[0] - u2[0] * u1[2]
            cz = u2[0] * u1[1] - u2[1] * u1[0]
            if cx == 0 and cy == 0 and cz == 0:
                continue
            A = u2[0] * b1[0] + u2[1] * b1[1] + u2[2] * b1[2]
            Bc = u2[0] * b2[0] + u2[1] * b2[1] + u2[2] * b2[2]
            b0 = u2[0] * nb[0] + u2[1] * nb[1] + u2[2] * nb[2]
            for k2 in range(2, 14):
                C = k2 * q - b0
                g, xa, xb = _egcd_nb(A, Bc)
                if g != 0 and C % g == 0:
                    al = xa * (C // g)
                    be = xb * (C // g)
                    base = nb + al * b1 + be * b2
                    v = (Bc // g) * b1 - (A // g) * b2
                    lo = -(1 << 62)
                    hi = 1 << 62
                    ok = True
                    for di in range(nd):
                        bd = base[0] * Dm[di, 0] + base[1] * Dm[di, 1] + base[2] * Dm[di, 2]
                        vd = v[0] * Dm[di, 0] + v[1] * Dm[di, 1] + v[2] * Dm[di, 2]
                        if vd == 0:
                            if bd > B or bd < -B:
                                ok = False
                                break
                        elif vd > 0:
                            l_ = -((B + bd) // vd)          # ceil((-B - bd)/vd)
                            h_ = (B - bd) // vd
                            lo = max(lo, l_)
                            hi = min(hi, h_)
                        else:
                            l_ = -((B - bd) // (-vd))       # ceil((B - bd)/vd) = ceil((bd - B)/w), w = -vd > 0
                            h_ = (-B - bd) // vd
                            lo = max(lo, l_)
                            hi = min(hi, h_)
                    if ok and lo <= hi and hi - lo + 1 <= WMAX:
                        # fast path: count array over the t-interval (no allocation, no sorting)
                        W = hi - lo + 1
                        for i in range(m):
                            bu[i] = Xa[i, 0] * base[0] + Xa[i, 1] * base[1] + Xa[i, 2] * base[2]
                            vu[i] = Xa[i, 0] * v[0] + Xa[i, 1] * v[1] + Xa[i, 2] * v[2]
                        nreq = 0
                        ntouch = 0
                        kk = k2 + 1
                        cnt_vals = 0
                        alive_line = True
                        while cnt_vals < SIEVE_NB and kk <= K:
                            if not auto[kk]:
                                kq = kk * q
                                full = False
                                for i in range(m):
                                    if vu[i] == 0 and bu[i] == kq:
                                        full = True
                                        break
                                if not full:
                                    cnt_vals += 1
                                    nreq += 1
                                    hit = 0
                                    for i in range(m):
                                        if vu[i] != 0:
                                            num = kq - bu[i]
                                            if num % vu[i] == 0:
                                                t = num // vu[i]
                                                if lo <= t <= hi:
                                                    o = t - lo
                                                    if stamparr[o] != nreq and cntarr[o] == nreq - 1:
                                                        stamparr[o] = nreq
                                                        cntarr[o] = nreq
                                                        hit += 1
                                                        if nreq == 1:
                                                            touched[ntouch] = o
                                                            ntouch += 1
                                    if hit == 0:
                                        alive_line = False
                                        break
                            kk += 1
                        if alive_line:
                            if nreq == 0:
                                for o in range(W):
                                    t = lo + o
                                    n0_ = base[0] + t * v[0]
                                    n1_ = base[1] + t * v[1]
                                    n2_ = base[2] + t * v[2]
                                    if _covers(n0_, n1_, n2_, Xa, q, K):
                                        found.append((n0_, n1_, n2_))
                            else:
                                for ii in range(ntouch):
                                    o = touched[ii]
                                    if cntarr[o] == nreq:
                                        t = lo + o
                                        n0_ = base[0] + t * v[0]
                                        n1_ = base[1] + t * v[1]
                                        n2_ = base[2] + t * v[2]
                                        if _covers(n0_, n1_, n2_, Xa, q, K):
                                            found.append((n0_, n1_, n2_))
                        for ii in range(ntouch):              # reset only touched cells
                            cntarr[touched[ii]] = 0
                            stamparr[touched[ii]] = 0
                    elif ok and lo <= hi:
                        for i in range(m):
                            bu[i] = Xa[i, 0] * base[0] + Xa[i, 1] * base[1] + Xa[i, 2] * base[2]
                            vu[i] = Xa[i, 0] * v[0] + Xa[i, 1] * v[1] + Xa[i, 2] * v[2]
                        # sieve with the next three values not attained automatically by multiples of u1; the
                        # candidate list is seeded by the first constraining value (<= |X| entries)
                        cand = np.empty(0, np.int64)
                        seeded = False
                        cnt = 0
                        kk = k2 + 1
                        while cnt < 3 and kk <= K and (not seeded or cand.shape[0] > 0):
                            if not auto[kk]:
                                tv, full = _t_values(bu, vu, kk * q, lo, hi, m)
                                if not full:
                                    if not seeded:
                                        cand = tv
                                        seeded = True
                                    else:
                                        keep = np.zeros(cand.shape[0], np.bool_)
                                        p = 0
                                        for ii in range(cand.shape[0]):
                                            while p < tv.shape[0] and tv[p] < cand[ii]:
                                                p += 1
                                            if p < tv.shape[0] and tv[p] == cand[ii]:
                                                keep[ii] = True
                                        cand = cand[keep]
                                cnt += 1
                            kk += 1
                        if not seeded:
                            cand = np.arange(lo, hi + 1)
                        for t in cand:
                            n0_ = base[0] + t * v[0]
                            n1_ = base[1] + t * v[1]
                            n2_ = base[2] + t * v[2]
                            if _covers(n0_, n1_, n2_, Xa, q, K):
                                found.append((n0_, n1_, n2_))
                if not auto[k2]:
                    break
        if len(found) > 0:
            res = np.empty((len(found), 3), np.int64)
            for i in range(len(found)):
                res[i, 0] = found[i][0]
                res[i, 1] = found[i][1]
                res[i, 2] = found[i][2]
        return res

    @_nb.njit(cache=True)
    def _cover_code_nb(Xa, Dm, q, K, B, grid):
        """Whole covering search for one code in compiled code (same enumeration as covering_functionals_nb).
        grid[x+12, y+12, z+12] = 1 + index of the point in Xa (0 if absent)."""
        m = Xa.shape[0]
        outs = []
        auto = np.zeros(K + 2, np.bool_)
        for idx in range(m):
            a, b, c = Xa[idx, 0], Xa[idx, 1], Xa[idx, 2]
            if a == 0 and b == 0 and c == 0:
                continue
            j = grid[-a + 12, -b + 12, -c + 12] - 1              # index of -u1 (X = -X)
            if j >= 0 and j < idx:
                continue
            g1, _x, _y = _egcd_nb(a, b)
            g, _s, _t = _egcd_nb(g1, c)
            if q % g != 0:
                continue
            pa, pb, pc = a // g, b // g, c // g
            n0 = np.zeros(3, np.int64)
            b1 = np.zeros(3, np.int64)
            b2 = np.zeros(3, np.int64)
            if pa == 0 and pb == 0:
                n0[2] = 1 if pc > 0 else -1
                b1[0] = 1
                b2[1] = 1
            else:
                h1, x, y = _egcd_nb(pa, pb)
                h2, s, t = _egcd_nb(h1, pc)
                n0[0] = x * s
                n0[1] = y * s
                n0[2] = t
                b1[0] = pb // h1
                b1[1] = -pa // h1
                b2[0] = pc * x
                b2[1] = pc * y
                b2[2] = -h1
            nb = n0 * (q // g)
            for k in range(K + 2):
                xa, xb, xc = k * a, k * b, k * c
                if k > 0 and -12 <= xa <= 12 and -12 <= xb <= 12 and -12 <= xc <= 12:
                    auto[k] = grid[xa + 12, xb + 12, xc + 12] > 0
                else:
                    auto[k] = False
            u1 = Xa[idx].copy()
            res = _lines_kernel(u1, nb, b1, b2, Xa, Dm, q, K, B, auto)
            for r in range(res.shape[0]):
                outs.append((res[r, 0], res[r, 1], res[r, 2]))
        out = np.empty((len(outs), 3), np.int64)
        for i in range(len(outs)):
            out[i, 0] = outs[i][0]
            out[i, 1] = outs[i][1]
            out[i, 2] = outs[i][2]
        return out

    @_nb.njit(cache=True)
    def _gcd(a, b):
        a = abs(a)
        b = abs(b)
        while b:
            a, b = b, a % b
        return a

    @_nb.njit(cache=True)
    def _rat_keys(C0, D0, w, Xa, kk, lo, hi, m, keys):
        """Normalised keys of t = (kk D0 - C0.u) / (D0 w.u) for u in X with w.u != 0 and t in [lo, hi];
        returns (count, full) where full means some u with w.u = 0 attains kk on the whole line."""
        c = 0
        full = False
        for i in range(m):
            wu = w[0] * Xa[i, 0] + w[1] * Xa[i, 1] + w[2] * Xa[i, 2]
            cu = C0[0] * Xa[i, 0] + C0[1] * Xa[i, 1] + C0[2] * Xa[i, 2]
            if wu == 0:
                if cu == kk * D0:
                    full = True
                continue
            num = kk * D0 - cu
            den = D0 * wu
            if den < 0:
                num = -num
                den = -den
            tv = num / den
            if tv < lo - 1e-6 * (1.0 + abs(lo)) or tv > hi + 1e-6 * (1.0 + abs(hi)):
                continue
            g = _gcd(num, den)
            if g > 1:
                num //= g
                den //= g
            keys[c] = (num + (1 << 30)) * (1 << 32) + den
            c += 1
        return c, full

    @_nb.njit(cache=True)
    def _rat_fkeys(C0, D0, w, Xa, kk, lo, hi, m, fk, nums, dens):
        """As _rat_keys, without gcd: float keys num/den plus the exact (num, den > 0)."""
        c = 0
        full = False
        for i in range(m):
            wu = w[0] * Xa[i, 0] + w[1] * Xa[i, 1] + w[2] * Xa[i, 2]
            cu = C0[0] * Xa[i, 0] + C0[1] * Xa[i, 1] + C0[2] * Xa[i, 2]
            if wu == 0:
                if cu == kk * D0:
                    full = True
                continue
            num = kk * D0 - cu
            den = D0 * wu
            if den < 0:
                num = -num
                den = -den
            tv = num / den
            if tv < lo - 1e-6 * (1.0 + abs(lo)) or tv > hi + 1e-6 * (1.0 + abs(hi)):
                continue
            fk[c] = tv
            nums[c] = num
            dens[c] = den
            c += 1
        return c, full

    @_nb.njit(cache=True)
    def _cover_code_rat(Xa, Dm, K, grid):
        """All rational covering functionals c (up to sign) of one code: X covers every integer in [-K, K] under c and
        |c.d| <= 2K for every collision direction d. Returns rows (N0, N1, N2, Q) with c = (N0, N1, N2)/Q."""
        m = Xa.shape[0]
        nd = Dm.shape[0]
        B = 2 * K
        outs = []
        auto = np.zeros(K + 2, np.bool_)
        fkA = np.empty(m, np.float64)
        fkB = np.empty(m, np.float64)
        numA = np.empty(m, np.int64)
        denA = np.empty(m, np.int64)
        numB = np.empty(m, np.int64)
        denB = np.empty(m, np.int64)
        fullv = np.zeros(K + 2, np.bool_)
        present = np.zeros(K + 1, np.bool_)
        for idx in range(m):
            a, b, cc = Xa[idx, 0], Xa[idx, 1], Xa[idx, 2]
            if a == 0 and b == 0 and cc == 0:
                continue
            j0 = grid[-a + 12, -b + 12, -cc + 12] - 1
            if j0 >= 0 and j0 < idx:
                continue
            for k in range(K + 2):
                xa, xb, xc = k * a, k * b, k * cc
                if k > 0 and -12 <= xa <= 12 and -12 <= xb <= 12 and -12 <= xc <= 12:
                    auto[k] = grid[xa + 12, xb + 12, xc + 12] > 0
                else:
                    auto[k] = False
            for jdx in range(m):
                u2x, u2y, u2z = Xa[jdx, 0], Xa[jdx, 1], Xa[jdx, 2]
                w0 = b * u2z - cc * u2y
                w1 = cc * u2x - a * u2z
                w2 = a * u2y - b * u2x
                if w0 == 0 and w1 == 0 and w2 == 0:
                    continue
                w = np.array([w0, w1, w2], np.int64)
                # third equation c.e_j = 0 with |w_j| maximal: det[u1; u2; e_j] = +-w_j != 0
                jj = 0
                if abs(w1) > abs(w[jj]):
                    jj = 1
                if abs(w2) > abs(w[jj]):
                    jj = 2
                k2first = 2
                while auto[k2first]:
                    k2first += 1          # first value not attained by k u1 (exists: 13 u1 is outside [-12,12]^3)
                for k2 in range(k2first, k2first + 1):
                    # solve c.u1 = 1, c.u2 = k2, c.e_jj = 0 by Cramer: C0 / D0
                    Mx = np.zeros((3, 3), np.int64)
                    Mx[0, 0], Mx[0, 1], Mx[0, 2] = a, b, cc
                    Mx[1, 0], Mx[1, 1], Mx[1, 2] = u2x, u2y, u2z
                    Mx[2, jj] = 1
                    D0 = (Mx[0, 0] * (Mx[1, 1] * Mx[2, 2] - Mx[1, 2] * Mx[2, 1])
                          - Mx[0, 1] * (Mx[1, 0] * Mx[2, 2] - Mx[1, 2] * Mx[2, 0])
                          + Mx[0, 2] * (Mx[1, 0] * Mx[2, 1] - Mx[1, 1] * Mx[2, 0]))
                    rhs0, rhs1 = 1, k2
                    C0 = np.zeros(3, np.int64)
                    for col in range(3):
                        # replace column col by rhs (1, k2, 0)
                        M2 = Mx.copy()
                        M2[0, col], M2[1, col], M2[2, col] = rhs0, rhs1, 0
                        C0[col] = (M2[0, 0] * (M2[1, 1] * M2[2, 2] - M2[1, 2] * M2[2, 1])
                                   - M2[0, 1] * (M2[1, 0] * M2[2, 2] - M2[1, 2] * M2[2, 0])
                                   + M2[0, 2] * (M2[1, 0] * M2[2, 1] - M2[1, 1] * M2[2, 0]))
                    # t-interval from |c.d| <= B, c = C0/D0 + t w
                    lo = -1e300
                    hi = 1e300
                    ok = True
                    for di in range(nd):
                        cd = (C0[0] * Dm[di, 0] + C0[1] * Dm[di, 1] + C0[2] * Dm[di, 2]) / D0
                        wd = w0 * Dm[di, 0] + w1 * Dm[di, 1] + w2 * Dm[di, 2]
                        if wd == 0:
                            if abs(cd) > B * (1.0 + 1e-9) + 1e-6:
                                ok = False
                                break
                        else:
                            l1 = (-B - cd) / wd
                            h1 = (B - cd) / wd
                            if l1 > h1:
                                l1, h1 = h1, l1
                            lo = max(lo, l1)
                            hi = min(hi, h1)
                    if ok and lo <= hi + 1e-6 * (1.0 + abs(hi)):
                        nA = -1
                        cntv = 0
                        kk = k2 + 1
                        dead = False
                        # sieve on float64 keys num/den (correctly rounded division: equal rationals give identical
                        # doubles, so the float intersection is a superset of the exact one); the exact (num, den) of
                        # the first value is carried along and every survivor is checked exactly below
                        # values attained on the whole line (by points of span(u1, u2)): one pass, then skipped
                        for kx in range(K + 2):
                            fullv[kx] = False
                        for i in range(m):
                            if w0 * Xa[i, 0] + w1 * Xa[i, 1] + w2 * Xa[i, 2] == 0:
                                cu = C0[0] * Xa[i, 0] + C0[1] * Xa[i, 1] + C0[2] * Xa[i, 2]
                                if cu % D0 == 0:
                                    vv = cu // D0
                                    if 1 <= vv <= K:
                                        fullv[vv] = True
                        while cntv < SIEVE_NB and kk <= K:
                            if not auto[kk] and not fullv[kk]:
                                nB, full = _rat_fkeys(C0, D0, w, Xa, kk, lo, hi, m, fkB, numB, denB)
                                if not full:
                                    cntv += 1
                                    if nA < 0:
                                        order = np.argsort(fkB[:nB])
                                        nA = nB
                                        for ii in range(nB):
                                            fkA[ii] = fkB[order[ii]]
                                            numA[ii] = numB[order[ii]]
                                            denA[ii] = denB[order[ii]]
                                    else:
                                        sB = np.sort(fkB[:nB])
                                        nn = 0
                                        p = 0
                                        for ii in range(nA):
                                            while p < nB and sB[p] < fkA[ii]:
                                                p += 1
                                            if p < nB and sB[p] == fkA[ii]:
                                                fkA[nn] = fkA[ii]
                                                numA[nn] = numA[ii]
                                                denA[nn] = denA[ii]
                                                nn += 1
                                        nA = nn
                                    if nA == 0:
                                        dead = True
                                        break
                            kk += 1
                        if not dead and nA > 0:
                            for ii in range(nA):
                                num = numA[ii]
                                den = denA[ii]
                                # c = (C0 den + num D0 w) / (D0 den)
                                N0 = C0[0] * den + num * D0 * w0
                                N1 = C0[1] * den + num * D0 * w1
                                N2 = C0[2] * den + num * D0 * w2
                                Q = D0 * den
                                if Q < 0:
                                    N0, N1, N2, Q = -N0, -N1, -N2, -Q
                                g = _gcd(_gcd(_gcd(N0, N1), N2), Q)
                                N0 //= g
                                N1 //= g
                                N2 //= g
                                Q //= g
                                for kx in range(K + 1):
                                    present[kx] = False
                                for i in range(m):
                                    val = N0 * Xa[i, 0] + N1 * Xa[i, 1] + N2 * Xa[i, 2]
                                    if val > 0 and val <= K * Q and val % Q == 0:
                                        present[val // Q] = True
                                allp = True
                                for kx in range(1, K + 1):
                                    if not present[kx]:
                                        allp = False
                                        break
                                if allp:
                                    outs.append((N0, N1, N2, Q))
                    if not auto[k2]:
                        break
        out = np.empty((len(outs), 4), np.int64)
        for i in range(len(outs)):
            out[i, 0] = outs[i][0]
            out[i, 1] = outs[i][1]
            out[i, 2] = outs[i][2]
            out[i, 3] = outs[i][3]
        return out

    HAVE_NUMBA = True
except Exception:          # pragma: no cover
    HAVE_NUMBA = False


def covering_functionals_rat(X, Dirs, K=127):
    """All rational covering functionals (every denominator at once), as rows (N0, N1, N2, Q) up to sign."""
    Xa = np.ascontiguousarray(np.array(X, dtype=np.int64))
    Dm = np.ascontiguousarray(np.array(Dirs, dtype=np.int64))
    grid = np.zeros((25, 25, 25), np.int64)
    for i, r in enumerate(Xa):
        grid[r[0] + 12, r[1] + 12, r[2] + 12] = i + 1
    res = _cover_code_rat(Xa, Dm, int(K), grid)
    if len(res) == 0:
        return set()
    out = set()
    for r in res:
        t = tuple(int(x) for x in r)
        neg = (-t[0], -t[1], -t[2], t[3])
        out.add(max(t, neg))
    return out


def covering_functionals_nb2(X, Dirs, q, K=127):
    """Fully compiled covering search (driver loop in numba too); same output as covering_functionals_nb."""
    Xa = np.ascontiguousarray(np.array(X, dtype=np.int64))
    Dm = np.ascontiguousarray(np.array(Dirs, dtype=np.int64))
    grid = np.zeros((25, 25, 25), np.int64)
    for i, r in enumerate(Xa):
        grid[r[0] + 12, r[1] + 12, r[2] + 12] = i + 1
    res = _cover_code_nb(Xa, Dm, int(q), int(K), int(2 * K * q), grid)
    if len(res) == 0:
        return np.zeros((0, 3), np.int64)
    return np.unique(res, axis=0)


def covering_functionals_nb(X, Dirs, q, K=127):
    """Compiled version of covering_functionals_lines (same enumeration: value-1 representative u1 in X/+-, value-k2
    representative u2 independent of u1, k2 <= 13; t-interval from the collision slabs; 3-value sieve; full check)."""
    Xa = np.ascontiguousarray(np.array(X, dtype=np.int64))
    Dm = np.ascontiguousarray(np.array(Dirs, dtype=np.int64))
    Xs = {tuple(int(t) for t in r) for r in Xa}
    B = 2 * K * q
    seen, outs = set(), []
    for r in Xa:
        u1 = tuple(int(t) for t in r)
        if not any(u1) or tuple(-t for t in u1) in seen:
            continue
        seen.add(u1)
        gu = math.gcd(*u1)
        if q % gu:
            continue
        n0, b1, b2 = plane_lattice(tuple(t // gu for t in u1))
        nb = np.array(n0, dtype=np.int64) * (q // gu)
        auto = np.array([k > 0 and tuple(k * t for t in u1) in Xs for k in range(K + 2)], dtype=np.bool_)
        res = _lines_kernel(np.array(u1, dtype=np.int64), nb, np.array(b1, dtype=np.int64),
                            np.array(b2, dtype=np.int64), Xa, Dm, q, K, B, auto)
        if len(res):
            outs.append(res)
    if not outs:
        return np.zeros((0, 3), np.int64)
    return np.unique(np.concatenate(outs), axis=0)


def covering_functionals_lines(X, Dirs, q, K=127):
    """Same output as covering_functionals (all integer n, up to sign, with n.X covering q*[1..K] and |n.d| <= 2Kq),
    generated on lattice LINES: u1 = the value-1 representative (one of each +-pair; the sign of n absorbs the other),
    u2 = a representative of k2, the smallest value >= 2 with a representative independent of u1 (k2 <= 13 because
    13 u1 is not in V^3); every smaller value k must then be represented by k u1, so k u1 must lie in X."""
    Bp = 2 * K * q
    Dm = np.array(Dirs, dtype=np.int64)
    Xl = [tuple(int(t) for t in r) for r in X]
    Xs = set(Xl)
    Xa = np.array(Xl, dtype=np.int64)
    XTf = Xa.T.astype(np.float64)            # float64 BLAS products are exact here (all |values| << 2^53)
    DmTf = Dm.T.astype(np.float64)
    seen, cands, survivors = set(), [], []

    def _flush():
        # streaming filter: polytope then coverage of values 1..K; bounded memory per value-1 representative
        if not cands:
            return
        N = np.concatenate(cands)
        cands.clear()
        N = N[np.all(np.abs(N @ Dm.T) <= Bp, axis=1)]
        out = []
        for s in range(0, len(N), 20000):
            Nc = N[s:s + 20000]
            V = Nc @ Xa.T                                  # exact int64 values n.u, computed once per chunk
            valid = (V > 0) & (V <= K * q) & (V % q == 0)
            rows, cols = np.nonzero(valid)
            present = np.zeros((len(Nc), K + 1), bool)
            present[rows, V[rows, cols] // q] = True
            ok = present[:, 1:].all(axis=1)
            if ok.any():
                out.append(Nc[ok])
        if out:
            survivors.append(np.concatenate(out))

    for u1 in Xl:
        if not any(u1) or tuple(-t for t in u1) in seen:
            continue
        seen.add(u1)
        gu = math.gcd(*u1)
        if q % gu:
            continue
        n0, b1, b2 = (np.array(v, dtype=np.int64) for v in plane_lattice(tuple(t // gu for t in u1)))
        nb = n0 * (q // gu)
        cr = np.stack([Xa[:, 1] * u1[2] - Xa[:, 2] * u1[1], Xa[:, 2] * u1[0] - Xa[:, 0] * u1[2],
                       Xa[:, 0] * u1[1] - Xa[:, 1] * u1[0]], 1)
        indep = np.any(cr != 0, axis=1)
        U2 = Xa[indep]
        A, Bc = U2 @ b1, U2 @ b2
        for k2 in range(2, 14):
            C = k2 * q - U2 @ nb
            g, xa, xb = _vegcd(A, Bc)
            ok = (g != 0) & (C % np.where(g == 0, 1, g) == 0)
            if ok.any():
                gg, cc = g[ok], C[ok] // g[ok]
                al, be = xa[ok] * cc, xb[ok] * cc
                base = nb[None, :] + al[:, None] * b1[None, :] + be[:, None] * b2[None, :]
                v = (Bc[ok] // gg)[:, None] * b1[None, :] - (A[ok] // gg)[:, None] * b2[None, :]
                bd, vd = base @ Dm.T, v @ Dm.T                      # |bd + t vd| <= Bp for every direction
                with np.errstate(divide="ignore", invalid="ignore"):
                    l1, l2 = (-Bp - bd) / vd, (Bp - bd) / vd
                lo = np.where(vd != 0, np.minimum(l1, l2), -np.inf).max(axis=1)
                hi = np.where(vd != 0, np.maximum(l1, l2), np.inf).min(axis=1)
                fin = np.isfinite(lo) & np.isfinite(hi) & (hi >= lo - 1e-9)
                if fin.any():
                    t0 = np.ceil(lo[fin] - 1e-9).astype(np.int64)
                    t1 = np.floor(hi[fin] + 1e-9).astype(np.int64)
                    keep = t1 >= t0
                    if keep.any():
                        t0, t1 = t0[keep], t1[keep]
                        bb, vv = base[fin][keep], v[fin][keep]
                        # multi-value sieve: for the next SIEVE values k not automatically attained by multiples of
                        # u1, mark the t in [t0, t1] where some u in X has n(t).u = k q (t = (kq - base.u)/(v.u), or the
                        # whole line when v.u = 0 and base.u = kq); keep the t marked for every sieved value.
                        BU, VU = bb @ Xa.T, vv @ Xa.T
                        nzv = VU != 0
                        Vs1 = np.where(nzv, VU, 1)
                        ks, k = [], k2 + 1
                        while len(ks) < SIEVE and k <= K:
                            if tuple(k * t for t in u1) not in Xs:
                                ks.append(k)
                            k += 1
                        SH = np.int64(1) << np.int64(32)
                        C = None                                   # sparse candidate keys line*SH + (t - t0 + 1)
                        for kk in ks:
                            num = kk * q - BU
                            div = nzv & (num % Vs1 == 0)
                            tt = np.where(div, num // Vs1, 0) - t0[:, None]
                            inr = div & (tt >= 0) & (tt <= (t1 - t0)[:, None])
                            li, ci = np.nonzero(inr)
                            keys = np.unique(li.astype(np.int64) * SH + tt[li, ci] + 1)
                            full = np.nonzero(np.any((~nzv) & (BU == kk * q), axis=1))[0]
                            if C is None:
                                parts = [keys]
                                for r in full:                     # whole line attains kk: expand its interval
                                    parts.append(np.int64(r) * SH + np.arange(1, t1[r] - t0[r] + 2, dtype=np.int64))
                                C = np.unique(np.concatenate(parts))
                            else:
                                C = C[np.isin(C, keys) | np.isin(C // SH, full)]
                            if len(C) == 0:
                                break
                        if C is not None and len(C):
                            li, oi = C // SH, C % SH - 1
                            cands.append(bb[li] + (t0[li] + oi)[:, None] * vv[li])
            if tuple(k2 * t for t in u1) not in Xs:        # k2 u1 absent: k2 needs an independent representative
                break
        _flush()
    if not survivors:
        return np.zeros((0, 3), np.int64)
    return np.unique(np.concatenate(survivors), axis=0)


def cmd_lines_controls():
    """The line generator must reproduce the plane generator (up to sign) on the controls."""
    planes = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    X = np.array(admissible(S_FP4, planes, 3), dtype=np.int64)
    Dirs = [_prim(_cross([Fraction(x) for x in a], [Fraction(x) for x in b])) for a, b in itertools.combinations(planes, 2)]
    canon = lambda S_: {tuple(r) if tuple(r) > tuple(-t for t in r) else tuple(-t for t in r) for r in S_}
    import time
    t0 = time.time(); L = canon(covering_functionals_lines(X, Dirs, 1)); t1 = time.time()
    P = canon(covering_functionals(X, Dirs, 1)); t2 = time.time()
    rows = [{"control": "AWE6", "lines": len(L), "planes": len(P), "equal": L == P, "t_lines": round(t1 - t0, 2), "t_planes": round(t2 - t1, 2)}]
    for f, g, _ in load_codes53()[:6]:
        Xc, Dc = code_X_dirs(f, g)
        for K in (12, 127):
            t0 = time.time(); L = canon(covering_functionals_lines(Xc, Dc, 1, K=K)); t1 = time.time()
            P = canon(covering_functionals(Xc, Dc, 1, K=K)); t2 = time.time()
            rows.append({"K": K, "lines": len(L), "planes": len(P), "equal": L == P, "t_lines": round(t1 - t0, 2), "t_planes": round(t2 - t1, 2)})
    _save("lines_controls.json", rows)
    print(json.dumps(rows))


def code_X_dirs(f, g, V=S_FP4):
    U = box_points(V, 3)
    X = U[row_mask(V, U, f) & row_mask(V, U, g)]
    funcs = [(1, 0, 0), (0, 1, 0), (0, 0, 1), f, g]
    Dirs = [_prim(_cross([Fraction(x) for x in funcs[j]], [Fraction(x) for x in funcs[l]]))
            for j, l in itertools.combinations(range(5), 2)]
    return X, Dirs


def q_allowed(X, qmax=16):
    """Denominators q (c = n/q in lowest terms) compatible with >= 255 covering points: for every prime power p^e || q
    some coordinate k is invertible, so u_k is fixed mod p^e by the other two; bound = sum over (u_i,u_j)-classes of the
    largest residue group.  Returns q <= qmax all of whose prime-power parts pass (composites checked conservatively)."""
    def ok_pe(pe):
        best = 0
        for k in range(3):
            i, j = [t for t in range(3) if t != k]
            groups = {}
            for r in X:
                groups.setdefault((int(r[i]), int(r[j]), int(r[k]) % pe), 0)
                groups[(int(r[i]), int(r[j]), int(r[k]) % pe)] += 1
            cls = {}
            for (a, b, _), cnt in groups.items():
                cls[(a, b)] = max(cls.get((a, b), 0), cnt)
            best = max(best, sum(cls.values()))
        return best >= 255
    # every prime power p^e > 24 fails automatically (S has pairwise distinct residues mod p^e, so the bound is at most
    # the number of (u_i,u_j) classes <= 225 < 255); the bound is monotone in e, so keep the largest passing exponent
    best_e = {}
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23):
        e = 0
        while p ** (e + 1) <= 24 and ok_pe(p ** (e + 1)):
            e += 1
        if e:
            best_e[p] = e
    qs = [1]
    for p, e in best_e.items():
        qs = [q * p ** i for q in qs for i in range(e + 1)]
    return sorted(qs)


def q_allowed_exact(X, N=255):
    """Exact denominator filter: c = n/q with gcd(n, q) = 1 needs >= N covering points with n.u == 0 (mod q). For every
    prime power p^e <= 24 compute the exact maximum over n mod p^e (some n_k a unit mod p) of #{u in X : n.u == 0};
    prime powers > 24 fail automatically (S has distinct residues mod p^e, so <= 225 points). Composite q (products
    of passing prime powers, one per prime) are checked exactly when q <= 64 and kept otherwise (conservative)."""
    Xa = np.array(X, dtype=np.int64)

    def maxcount(md):
        vals = np.arange(md)
        Nv = np.array(list(itertools.product(vals, repeat=3)), dtype=np.int64)
        prim = np.ones(len(Nv), bool)
        for p in range(2, md + 1):
            if md % p == 0 and all(p % d for d in range(2, p)):
                prim &= np.any(Nv % p != 0, axis=1)
        Nv = Nv[prim]
        best = 0
        for s in range(0, len(Nv), 4096):
            best = max(best, int(((Nv[s:s + 4096] @ Xa.T) % md == 0).sum(axis=1).max()))
        return best

    best_e = {}
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23):
        e = 0
        while p ** (e + 1) <= 24 and maxcount(p ** (e + 1)) >= N:
            e += 1
        if e:
            best_e[p] = e
    qs = [1]
    for p, e in best_e.items():
        qs = [q * p ** i for q in qs for i in range(e + 1)]
    pp_ok = {p ** i for p, e in best_e.items() for i in range(1, e + 1)}
    return sorted(q for q in qs if q == 1 or q in pp_ok or q > 64 or maxcount(q) >= N)


def cmd_cover_controls():
    """Positive/negative controls for covering_functionals (pre-run instrumentation gate, AGENTS.md section 12)."""
    # positive control: AWE's 6-plane INT8 code (planes a_i, a_i + a_j); its admissible set must admit the covering
    # functionals w = (1,29,37) and (1,11,39) (coverages 148 and 134 >= 127); collision directions from all 15 plane pairs
    planes = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    X = np.array(admissible(S_FP4, planes, 3), dtype=np.int64)
    Dirs = [_prim(_cross([Fraction(x) for x in a], [Fraction(x) for x in b])) for a, b in itertools.combinations(planes, 2)]
    N = covering_functionals(X, Dirs, 1, K=127)
    Ns = {tuple(int(t) for t in r) for r in N}
    has = (1, 29, 37) in Ns and (1, 11, 39) in Ns
    # small-target control on real codes: compare with brute force over the full polytope
    codes = load_codes53()
    rows = []
    for f, g, _ in codes[:6]:
        Xc, Dc = code_X_dirs(f, g)
        K = 12
        fast = {tuple(r) for r in covering_functionals(Xc, Dc, 1, K=K)}
        Bd = 2 * K
        grid = np.array(list(itertools.product(range(-Bd, Bd + 1), repeat=3)), dtype=np.int64)
        grid = grid[np.all(np.abs(grid @ np.array(Dc).T) <= Bd, axis=1)]
        V = grid @ Xc.T
        okm = np.ones(len(grid), bool)
        for k in range(1, K + 1):
            okm &= (V == k).any(1)
        brute = {tuple(r) for r in grid[okm]}
        rows.append({"f": [str(x) for x in f], "g": [str(x) for x in g], "K": K, "fast": len(fast), "brute": len(brute),
                     "equal": fast == brute})
    out = {"positive_control_AWE6_finds_(1,29,37)_and_(1,11,39)": has, "AWE6_covering_functionals_q1": sorted(Ns),
           "small_target_vs_brute": rows}
    _save("cover_controls.json", out)
    print(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------------------------------------
# Modular side: per-modulus schemes, coprime-moduli integer programs, and the engine-facing config table
# ----------------------------------------------------------------------------------------------------------
def covers_residues(V, m):
    return len({v % m for v in V}) == m


def karatsuba_modulus_ok(V, m, smax=None):
    """Certified 3-product scheme mod m: every residue x has x == a0 + s*a1 (mod m) with a0, a1, a0+a1 in V for some
    fixed s; then x*y == (1-s) a0 b0 + s (a0+a1)(b0+b1) + (s^2-s) a1 b1 (mod m). Returns the smallest working s or None."""
    Vs = set(V)
    pairs = [(a0, a1) for a0 in V for a1 in V if a0 + a1 in Vs]
    for s in range(2, (smax or m) + 1):
        if len({(a0 + s * a1) % m for a0, a1 in pairs}) == m:
            return s
    return None


def coprime_ip(cands, target_log2, bound_int=None):
    """Minimise sum cost over pairwise-coprime moduli with sum log2(m) >= target_log2 (scipy HiGHS MILP); then verify
    the exact product bound. cands: list of (m, cost). Returns (moduli, total_cost)."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    ms = [m for m, _ in cands]
    cost = np.array([c for _, c in cands], dtype=float)
    lg = np.array([math.log2(m) for m in ms])
    primes = sorted({p for m in ms for p in range(2, m + 1) if m % p == 0 and all(p % d for d in range(2, int(p ** .5) + 1))})
    A = [[1.0 if m % p == 0 else 0.0 for m in ms] for p in primes]
    cons = [LinearConstraint(np.array(A), -np.inf, 1.0), LinearConstraint(lg[None, :], target_log2 + 1e-9, np.inf)]
    obj = cost - 1e-6 * lg                     # tie-break: larger product
    res = milp(obj, constraints=cons, integrality=np.ones(len(ms)), bounds=Bounds(0, 1))
    if res.x is None:
        return None, None
    chosen = [m for m, xv in zip(ms, res.x) if xv > 0.5]
    if bound_int is not None:
        assert math.prod(chosen) > bound_int, "log rounding broke the exact bound"
    return sorted(chosen, reverse=True), int(round(sum(c for (m, c), xv in zip(cands, res.x) if xv > 0.5)))


def int8_crt(b, k):
    """Exact product of b-bit-magnitude integer matrices (|x| <= 2^b - 1) with inner dimension k by INT8 CRT
    (1 GEMM per modulus m <= 256, symmetric residues in [-128,127], INT32 exact while k*128^2 < 2^31)."""
    bound = 2 * k * (2 ** b - 1) ** 2               # need prod(m) > 2 max|C|
    mods, n = coprime_ip([(m, 1) for m in range(2, 257)], math.log2(bound), bound)
    return mods, n


def cmd_chain():
    """Range / base-extension cost model for exact CHAINS A_1 A_2 ... A_q in persistent INT8 residue arithmetic.
    |entries of A_s| <= H = 2^b - 1, every inner dimension k: |(A_1..A_s)_ij| <= R_s = H^s k^(s-1).  One INT8 GEMM per
    modulus per product (moduli <= 256, pairwise coprime, residues reduced after every product).  N(R) = minimum number of
    moduli with prod(m) > 2R (exact MILP).  Fixed base: every product uses the final base, (q-1) N(R_q) GEMMs.  Growing
    base (base extension before product s): sum_s N(R_s) GEMMs plus elementwise extension work.  Hard cap: prod of all
    pairwise-coprime m <= 256 is lcm(1..256), so exact INT8 residue chains need log2(2 R_q) <= log2 lcm(1..256)."""
    cap = math.log2(math.lcm(*range(1, 257)))
    rows = []
    for b, k in [(7, 4096), (8, 4096), (16, 4096), (24, 4096), (53, 4096)]:
        H = 2 ** b - 1
        Ncache = {}
        def N(s):
            if s not in Ncache:
                R = H ** s * k ** (s - 1)
                if math.log2(2 * R) > cap:
                    Ncache[s] = None
                else:
                    Ncache[s] = coprime_ip([(m, 1) for m in range(2, 257)], math.log2(2 * R), 2 * R)[1]
            return Ncache[s]
        for q in range(2, 41):
            if N(q) is None:
                rows.append({"b": b, "k": k, "q": q, "exact_cap_exceeded": True})
                break
            fixed = (q - 1) * N(q)
            grow = sum(N(s) for s in range(2, q + 1))
            rows.append({"b": b, "k": k, "q": q, "log2_R_q": round(math.log2(H ** q * k ** (q - 1)), 1), "N_final": N(q),
                         "gemms_fixed_base": fixed, "gemms_growing_base": grow,
                         "rounded_chain_crt": (q - 1) * N(2),
                         "rounded_chain_M1_floor": (q - 1) * (2 * ((b + 1 + 7) // 8) - 1)})
    out = {"cap_log2_lcm_1_256": round(cap, 2), "rows": rows}
    _save("chain_cost_model.json", out)
    for r in rows:
        if r["q"] in (2, 3, 4, 8, 16) or r.get("exact_cap_exceeded"):
            print(r)
    print("cap", round(cap, 2))


def sylvester_int(nbits_signed_values, Vsize):
    """ell >= ceil(log_|V| N) for N input values."""
    N, ell = nbits_signed_values, 1
    while Vsize ** ell < N:
        ell += 1
    return ell


def cmd_modular():
    out = {}
    # INT8 CRT vs digit schemes, lower bounds; the engine table
    table = []
    for b in (7, 8, 12, 15, 16, 20, 24, 27, 31, 32, 40, 48, 53, 56, 60):
        for k in (1024, 4096, 16384, 65536, 131071):
            mods, n = int8_crt(b, k)
            ell = sylvester_int(2 ** (b + 1) - 1, 256)
            s8 = math.ceil((b + 1) / 8)                     # signed 8-bit digits, schoolbook
            entry = {"value_set": "INT8[-128,127]", "bits_b": b, "k_max": k,
                     "crt": {"moduli": mods, "gemms": n, "recon_read_bytes_per_output": 4 * n,
                             "recon_macs_per_output": n, "exact_if": "k*128^2 < 2^31 and prod(moduli) > 2k(2^b-1)^2",
                             "rigor": "PROVED exact (CRT + INT32 bound); count optimal among 1-GEMM-per-modulus CRT (MILP, exact product re-verified)"},
                     "digits_schoolbook": {"limbs": s8, "gemms": s8 * s8, "recon_read_bytes_per_output": 4 * s8 * s8,
                                           "exact_if": "k*128^2 < 2^31", "rigor": "PROVED exact"},
                     "lower_bound_M1_M2prime": 2 * ell - 1, "lower_bound_M4_injectivity": ell,
                     "lower_bound_rigor": "Theorem 2 (PROVED-pending-gate)"}
            best = min(("crt", n), ("digits_schoolbook", s8 * s8), key=lambda t: t[1])
            entry["recommended"] = best[0]
            table.append(entry)
    out["int8_table_rows"] = len(table)
    # FP8 (E4M3 integers |v| <= 16) and FP4 (2 x E2M1) per-modulus analysis
    for name, V, one_cap in (("FP8_E4M3_16", V_FP8_16, 64), ("FP4_E2M1x2", S_FP4, 64)):
        ones = [m for m in range(2, 400) if covers_residues(V, m)]
        threes = {}
        for m in range(2, 1200):
            if m in ones:
                continue
            s = karatsuba_modulus_ok(V, m, smax=80)
            if s:
                threes[m] = s
        out[name] = {"one_product_moduli": ones, "three_product_moduli_count": len(threes),
                     "three_product_moduli_max": max(threes) if threes else None}
        # targets as in the literature: FP4 = AWE width 54 exact at K = 16384 (published 59); FP8 = 2603.10634's
        # modulus product P > 2^111 (P/2 > 2^110; published 3N = 36 with N = 12)
        targets = {"FP4_E2M1x2": ("width54_exact_K16384", 2 * 16384 * (2 ** 53 - 1) ** 2),
                   "FP8_E4M3_16": ("P_gt_2^111", 2 ** 111)}
        tname, bound = targets[name]
        cands = [(m, 1) for m in ones] + [(m, 3) for m in threes]
        mods, cost = coprime_ip(cands, math.log2(bound), bound)
        out[name][tname] = {"moduli": mods, "products": cost}
        mods3, cost3 = coprime_ip([(m, 3) for m in threes], math.log2(bound), bound)
        out[name][tname + "_three_product_moduli_only"] = {"moduli": mods3, "products": cost3}
    _save("modular.json", out)
    cfg = {"description": "engine-facing certified schemes: exact integer product of b-bit-magnitude integer matrices "
                          "(after power-of-two row/column scaling and round-to-nearest truncation, the product error is "
                          "<= 2^(1-b) (max_l|A_il| sum_l|B_lj| + (1+2^(1-b)) sum_l|A_il| max_l|B_lj|) componentwise - "
                          "PROVED elementary bound; mapping a target error to b is the engine's contract)",
           "unit": "digit_gemm_complexity_v1", "rows": table}
    _save("config_table.json", cfg)
    print(json.dumps(out, indent=1, default=str)[:4000])


_CODES = None


def _init_codes():
    global _CODES
    _CODES = load_codes53()


def _tight_worker(args):
    idxs, K = args
    codes = _CODES
    surv, stats = [], {"codes": 0, "q_extra": 0}
    for i in idxs:
        f, g, _ = codes[i]
        X, Dirs = code_X_dirs(f, g)
        stats["codes"] += 1
        for r in covering_functionals_rat(X, Dirs, K=K):         # every denominator in one pass
            surv.append({"code": i, "f": [str(x) for x in f], "g": [str(x) for x in g], "n": list(r[:3]), "q": r[3]})
    return surv, stats


def cmd_tight53():
    """Exhaustive decision of the tight INT8xINT8 case r = 5 over V_FP4 in model M1 (section 3): every R-good [5,3]
    plane code (orbit representatives) is searched for all rational covering functionals.
    Usage: tight53 <part> <nparts> [workers] [only_chunks e.g. 16,17 | -].  Resumable: each finished chunk is appended
    to tight53_part<p>_of_<n>.progress.jsonl with its chunk id and code indices; finished chunks are skipped."""
    import multiprocessing as mp
    import time
    part, nparts = int(sys.argv[2]), int(sys.argv[3])
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    only = None
    if len(sys.argv) > 5 and sys.argv[5] != "-":
        only = {int(x) for x in sys.argv[5].split(",")}
    reps = json.load(open(os.path.join(OUT, "orbits53.json")))      # exact orbit representatives (canonical53)
    mine = reps[part::nparts]
    W = 24
    pf = os.path.join(OUT, "tight53_part%d_of_%d.progress.jsonl" % (part, nparts))
    done = set()
    if os.path.exists(pf):
        done = {json.loads(l)["chunk"] for l in open(pf, encoding="utf-8") if l.strip()}
    todo = [j for j in range(W) if j not in done and (only is None or j in only)]
    print("part %d: %d chunks done, %d to run" % (part, len(done), len(todo)), flush=True)
    t0 = time.time()
    with mp.Pool(workers, initializer=_init_codes) as pool:
        for j, o in pool.imap_unordered(_tight_chunk, [(j, mine[j::W]) for j in todo]):
            with open(pf, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"chunk": j, "idxs": mine[j::W], "codes": o[1]["codes"], "survivors": o[0]}) + chr(10))
            print("chunk %d done, codes %d, survivors %d, %.0fs" % (j, o[1]["codes"], len(o[0]), time.time() - t0), flush=True)


def _tight_chunk(a):
    j, idxs = a
    return j, _tight_worker((idxs, 127))


AWE_TABLE = {  # AWE arXiv 2609.24519 Fig. 4 (counts found so far; rows M_A = 4..17, columns M_B = M_A..17)
    4: [3, 3, 4, 4, 4, 5, 6, 7, 7, 7, 8, 9, 9, 9], 5: [3, 4, 4, 5, 5, 6, 7, 7, 7, 8, 9, 9, 10],
    6: [4, 5, 6, 6, 6, 8, 8, 8, 8, 10, 10, 10], 7: [5, 6, 8, 8, 8, 9, 11, 11, 11, 13, 14],
    8: [6, 8, 9, 9, 9, 11, 12, 12, 13, 14], 9: [8, 9, 10, 10, 11, 12, 13, 14, 14], 10: [9, 11, 12, 12, 12, 15, 15, 15],
    11: [11, 12, 15, 15, 16, 18, 19], 12: [12, 15, 16, 16, 18, 19], 13: [15, 16, 19, 19, 19], 14: [16, 20, 20, 20],
    15: [20, 23, 24], 16: [23, 24], 17: [24]}


def ell_two_digit(V, X):
    """Exact test of ell(V, [-X, X]) <= 2: is there theta = n/q (q in {1,2}, |n_k| <= 2Xq) with theta.V^2 covering
    [-X, X]?  (q in {1,2} and the n bound are forced when 2X+1 > 8|V| - see doc section 3; the caller checks that.)"""
    Vv = np.array(V, dtype=np.int64)
    P = np.array(list(itertools.product(V, repeat=2)), dtype=np.int64)
    found = []
    for q in (1, 2):
        Bn = 2 * X * q
        n1 = np.arange(1, Bn + 1)
        for a in range(0, Bn + 1):          # n = (a, n2) up to sign symmetries of V (V = -V)
            n2 = n1
            vals = a * P[:, 0][None, :] + n2[:, None] * P[:, 1][None, :]      # (len(n2), |V|^2)
            ok = np.ones(len(n2), bool)
            for k in range(1, X + 1):
                ok &= (vals == k * q).any(axis=1)
                if not ok.any():
                    break
            for b in n2[ok]:
                if math.gcd(math.gcd(a, int(b)), q) == 1:
                    found.append((q, a, int(b)))
    return found


def cmd_ell():
    res = {}
    for M in (4, 5, 6, 7):
        X = width_range(M)
        f2 = ell_two_digit(S_FP4, X)
        res[M] = {"X": X, "log_bound": sylvester_int(2 * X + 1, 15), "two_digit_witnesses": len(f2),
                  "example": f2[:3], "ell": 2 if f2 else 3}
    # lower-bound table vs AWE
    def ell(M):
        if M in res:
            return res[M]["ell"]
        return max(sylvester_int(2 ** M - 1, 15), 2)
    table = []
    for MA, row in AWE_TABLE.items():
        for j, awe in enumerate(row):
            MB = MA + j
            lb = ell(MA) + ell(MB) - 1
            table.append({"MA": MA, "MB": MB, "lower_bound": lb, "awe": awe, "gap": awe - lb})
    out = {"ell": res, "table": table, "optimal_cells": [(t["MA"], t["MB"]) for t in table if t["gap"] == 0],
           "gap1_cells": [(t["MA"], t["MB"]) for t in table if t["gap"] == 1]}
    _save("ell_and_bound_table.json", out)
    print(json.dumps({"ell": res, "optimal_cells": out["optimal_cells"], "gap1_cells": out["gap1_cells"]}, default=str))


def modulus_recipe(V, m):
    """Certified per-modulus scheme: 1 product if V covers Z/m, else the 3-product Karatsuba recipe (weight s).
    Returns dict with the representation table and an exhaustive exact check over all residue pairs."""
    Vs = set(V)
    if covers_residues(V, m):
        rep = {}
        for v in sorted(V, key=abs):
            rep.setdefault(v % m, v)
        ok = all((rep[x] * rep[y] - x * y) % m == 0 for x in range(m) for y in range(m))
        return {"m": m, "products": 1, "planes": ["P"], "recon": "P*Q", "exact_check_all_pairs": ok,
                "max_abs_plane": max(abs(v) for v in rep.values()), "table": [rep[x] for x in range(m)]}
    s = karatsuba_modulus_ok(V, m, smax=m)
    if s is None:
        return None
    rep = {}
    for a0 in sorted(V, key=abs):
        for a1 in sorted(V, key=abs):
            if a0 + a1 in Vs:
                rep.setdefault((a0 + s * a1) % m, (a0, a1))
    R0 = np.array([rep[x][0] for x in range(m)], dtype=np.int64)
    R1 = np.array([rep[x][1] for x in range(m)], dtype=np.int64)
    G0 = np.outer(R0, R0); G1 = np.outer(R1, R1); G2 = np.outer(R0 + R1, R0 + R1)
    lhs = ((1 - s) * G0 + s * G2 + (s * s - s) * G1) % m
    ok = bool(np.array_equal(lhs, np.outer(np.arange(m), np.arange(m)) % m))
    return {"m": m, "products": 3, "weight_s": s, "planes": ["a0", "a1", "a0+a1"],
            "recon": "(1-s)*G(a0,b0) + s*G(a0+a1,b0+b1) + (s^2-s)*G(a1,b1)  (mod m)", "exact_check_all_pairs": ok,
            "max_abs_plane": int(max(np.abs(R0).max(), np.abs(R1).max(), np.abs(R0 + R1).max())),
            "table": [list(rep[x]) for x in range(m)]}


def cmd_verify_crt():
    """Exact verification of the FP8 mixed-moduli CRT scheme and the INT8 CRT rows; writes the engine config."""
    mod = json.load(open(os.path.join(OUT, "modular.json"), encoding="utf-8"))
    out = {}
    for name, V, key in (("FP8_E4M3_16", V_FP8_16, "P_gt_2^111"), ("FP4_E2M1x2", S_FP4, "width54_exact_K16384")):
        mods = mod[name][key]["moduli"]
        recipes = [modulus_recipe(V, m) for m in mods]
        coprime = all(math.gcd(a, b) == 1 for a, b in itertools.combinations(mods, 2))
        P = math.prod(mods)
        cmax = max(r["max_abs_plane"] for r in recipes)
        out[name] = {"target": key, "moduli": mods, "gemms": sum(r["products"] for r in recipes),
                     "all_recipes_exact": all(r["exact_check_all_pairs"] for r in recipes), "pairwise_coprime": coprime,
                     "log2_P": math.log2(P), "max_abs_plane_value": cmax,
                     "fp32_exact_if_k_le": (2 ** 24) // (cmax * cmax),
                     "exact_integer_product_if": "k * max|A_int| * max|B_int| < P/2 (CRT uniqueness)",
                     "recipes": recipes}
    _save("crt_verified.json", out)
    cfg = json.load(open(os.path.join(OUT, "config_table.json"), encoding="utf-8"))
    cfg["fp8_fp4_fp64_level"] = [
        {"value_set": nm, "target": o["target"], "moduli": o["moduli"], "gemms": o["gemms"],
         "gemm_inner_dim_multiple": 1, "recon_read_bytes_per_output": 4 * o["gemms"],
         "crt_terms": len(o["moduli"]), "exact_if": [o["exact_integer_product_if"], "FP32 accumulation exact while k <= %d" % o["fp32_exact_if_k_le"]],
         "per_modulus": [{k: r[k] for k in r if k != "table"} for r in o["recipes"]],
         "rigor": "PROVED exact (per-modulus identities checked over all residue pairs; CRT); count optimal only within the "
                  "declared candidate family (1-product moduli where V covers Z/m, 3-product 2-limb Karatsuba moduli) by MILP",
         "published_best": {"FP8_E4M3_16": "36 (3N, N=12; arXiv 2603.10634 fast mode)", "FP4_E2M1x2": "59 (AWE arXiv 2609.24519)"}[nm]}
        for nm, o in out.items()]
    _save("config_table.json", cfg)
    print(json.dumps({k: {kk: v[kk] for kk in ("gemms", "all_recipes_exact", "pairwise_coprime", "log2_P", "fp32_exact_if_k_le")}
                      for k, v in out.items()}, indent=1))


def cmd_mpoly():
    """Mpoly control (doc 3.5): admissible planes alpha (alpha . a in V for all digit tuples a in D^s) and admissible
    evaluation points (p:q) with scalings c (plane c*(q^(s-1),...,p^(s-1)))."""
    out = {}
    for name, V, D in (("FP4_D[-4,4]", S_FP4, range(-4, 5)), ("FP4_D[-2,2]", S_FP4, range(-2, 3)),
                       ("FP4_D[-1,1]", S_FP4, range(-1, 2)), ("INT4_D[-2,1]", V_INT4, range(-2, 2))):
        Vs = set(V)
        for s in (2, 3):
            tuples = list(itertools.product(D, repeat=s))
            multi = [al for al in itertools.product(range(-12, 13), repeat=s)
                     if sum(1 for x in al if x) >= 2 and all(sum(x * y for x, y in zip(al, a)) in Vs for a in tuples)]
            pts = set()
            for p in range(-12, 13):
                for qq in range(0, 13):
                    if (p == 0 and qq == 0) or math.gcd(p, qq) != 1 or (qq == 0 and p != 1):
                        continue
                    ev = [qq ** (s - 1 - i) * p ** i for i in range(s)]
                    for c in (Fraction(1, 4), Fraction(1, 3), Fraction(1, 2), 1, 2, 3, 4):
                        if all(c * sum(x * y for x, y in zip(ev, a)) in Vs for a in tuples):
                            pts.add((p, qq))
                            break
            out["%s_s%d" % (name, s)] = {"multi_limb_planes": len(multi), "example_multi": multi[:6],
                                          "admissible_points": sorted(pts), "rank_2s-1_possible": len(pts) >= 2 * s - 1}
    _save("mpoly.json", out)
    print(json.dumps(out))


def cmd_two_product_moduli():
    """Generation lemma (doc Theorem 6(d)): a 2-product scheme mod p^e needs the A-representations (>= p^e distinct
    points of V^2) to lie in a maximal submodule {a : g1 a1 + g2 a2 == 0 mod p}; so p^e <= max_g |V^2 cap M_g|."""
    out = {}
    for name, V, cover in (("FP8_E4M3_16", V_FP8_16, 33), ("FP4_E2M1x2", S_FP4, 13)):
        res = {}
        for p in (2, 3, 5, 7, 11, 13):
            gam = [(1, 0)] + [(t, 1) for t in range(p)]
            mx = max(sum(1 for a1 in V for a2 in V if (g1 * a1 + g2 * a2) % p == 0) for g1, g2 in gam)
            e_max = 0
            while p ** (e_max + 1) <= mx:
                e_max += 1
            cands = [p ** e for e in range(2, e_max + 1) if not covers_residues(V, p ** e)]
            res[p] = {"max_submodule_points": mx, "open_2product_prime_powers": cands}
        out[name] = res
    _save("two_product_moduli.json", out)
    print(json.dumps(out))


# ---------------- tight cells with ell_A = 2 (exact, complete given Theorem 3 and Lemma 4) ----------------------
def _rows2(V, N):
    R = diff_ratio_set(V)
    Rp = sorted(x for x in R if x > 0)
    U = box_points(V, 2)
    out = []
    for f1, f2 in itertools.combinations_with_replacement(sorted(Rp, reverse=True), 2):
        if (f1 / f2) in R and row_mask(V, U, (f1, f2)).sum() >= N:
            out.append((f1, f2))
    return out, R


def _codes2(V, N, r):
    """R-good [r,2] MDS codes (r = 3: one dependent row; r = 4: two) with |X| >= N; f canonical, g labelled."""
    canon, R = _rows2(V, N)
    if r == 3:
        return [((f,), ) for f in canon], R
    lab = sorted({g for f in canon for g in row_variants(f)})
    U = box_points(V, 2)
    out = []
    for f in canon:
        mf = row_mask(V, U, f)
        for g in lab:
            if (f[0] / g[0]) not in R or (f[1] / g[1]) not in R:
                continue
            m = f[0] * g[1] - f[1] * g[0]
            if m == 0 or not all((m / x) in R for x in (f[0], f[1], g[0], g[1])):
                continue
            if (mf & row_mask(V, U, g)).sum() >= N:
                out.append(((f, g),))
    return out, R


def _cover2(X, Dirs, q, K):
    """All integer n in Z^2 (up to sign) with n.X covering q*[-K..K] and |n.d| <= 2Kq for d in Dirs."""
    B = 2 * K * q
    Dm = np.array(Dirs, dtype=np.int64)
    n1 = np.arange(0, B + 1, dtype=np.int64)
    res = []
    for a in n1:
        n2 = np.arange(-B, B + 1, dtype=np.int64)
        N = np.stack([np.full(len(n2), a), n2], 1)
        if a == 0:
            N = N[n2 > 0]
        N = N[np.all(np.abs(N @ Dm.T) <= B, axis=1)]
        for k in range(1, K + 1):
            if len(N) == 0:
                break
            N = N[(N @ X.T == k * q).any(axis=1)]
        res.extend(tuple(int(t) for t in r) for r in N)
    return res


def _q_allowed2(X, N):
    best_e = {}
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23):
        e = 0
        while p ** (e + 1) <= 24:
            pe = p ** (e + 1)
            ok = False
            for k in range(2):
                o = 1 - k
                cls = {}
                for r_ in X:
                    key = (int(r_[o]), int(r_[k]) % pe)
                    cls[key] = cls.get(key, 0) + 1
                per = {}
                for (a, _), c in cls.items():
                    per[a] = max(per.get(a, 0), c)
                ok |= sum(per.values()) >= N
            if not ok:
                break
            e += 1
        if e:
            best_e[p] = e
    qs = [1]
    for p, e in best_e.items():
        qs = [q * p ** i for q in qs for i in range(e + 1)]
    return sorted(qs)


def tight_cell_l2(V, MA, MB, ellB):
    """Decide the tight case r = 2 + ellB - 1 for width pair (MA with ell = 2, MB with ell = ellB); lambda pinned by the
    B-side ratio lemma (requires 2^MB - 1 > |V|^(ellB-1))."""
    XA, XB = width_range(MA), width_range(MB)
    NA, NB = 2 * XA + 1, 2 * XB + 1
    assert NA > len(V) and NB > len(V) ** (ellB - 1)
    r = 1 + ellB
    codes, R = _codes2(V, NA, r)
    U2 = box_points(V, 2)
    Rl = sorted(R)
    stats = {"A_codes": len(codes), "A_covered": 0, "lambda_candidates": 0, "B_checked": 0, "schemes": []}
    for (rows,) in codes:
        gam = [(Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))] + [tuple(Fraction(x) for x in row) for row in rows]
        X = U2[np.all([row_mask(V, U2, row) for row in rows], axis=0)]
        Dirs = [_prim((g[1], -g[0])) for g in gam]              # codeword vanishing at each coordinate
        for q in _q_allowed2(X, NA):
            for n in _cover2(X, Dirs, q, XA):
                stats["A_covered"] += 1
                c = (Fraction(n[0], q), Fraction(n[1], q))
                # theta^T for pairs T = (i, j): c = th_i gam_i + th_j gam_j
                th = {}
                for i, j in itertools.combinations(range(r), 2):
                    a, b = gam[i], gam[j]
                    det = a[0] * b[1] - a[1] * b[0]
                    th[(i, j)] = ((c[0] * b[1] - c[1] * b[0]) / det, (a[0] * c[1] - a[1] * c[0]) / det)
                # zero-representation kappa in U_A (u with c.u = 0) -> planes kappa_k = gam_k . u0
                u0 = (c[1], -c[0])
                kap = [g[0] * u0[0] + g[1] * u0[1] for g in gam]
                allowed = {(i, j): {th[(i, j)][1] / th[(i, j)][0] * x for x in Rl} for (i, j) in th}   # lambda_j/lambda_i
                lam_sets = [[Fraction(1)]]
                for j in range(1, r):
                    new = []
                    for lam in lam_sets:
                        for v in allowed[(0, j)]:
                            if all((v / lam[i]) in allowed[(i, j)] for i in range(1, j)):
                                new.append(lam + [v])
                    lam_sets = new
                for lam in lam_sets:
                    # B-side dual lemma: Lambda*kappa has ratios in R
                    lk = [lam[k] * kap[k] for k in range(r)]
                    if any(x == 0 for x in lk) or not all((x / y) in R for x, y in itertools.combinations(lk, 2)):
                        continue
                    stats["lambda_candidates"] += 1
                    res = _check_B(V, r, gam, c, lam, NB, XB)
                    stats["B_checked"] += 1
                    if res:
                        stats["schemes"].append({"rows": [[str(x) for x in row] for row in rows], "n": n, "q": q,
                                                 "lambda": [str(x) for x in lam], **res})
    return stats


def _check_B(V, r, gam, c, lam, NB, XB):
    """Given U_A (coordinate functionals gam on Q^2), c and lambda, build U_B = Lambda^-1 {z : z|U_A in span c},
    enumerate X_B = U_B cap V^r and test whether d covers an AP of 2 XB + 1 terms."""
    import sympy
    # z|U_A = sum_k z_k gam_k (as a functional on Q^2) must be proportional to c: 2 x r condition matrix
    Mz = sympy.Matrix([[sympy.Rational(g[t].numerator, g[t].denominator) for g in gam] for t in range(2)])
    cvec = sympy.Matrix([sympy.Rational(c[0].numerator, c[0].denominator), sympy.Rational(c[1].numerator, c[1].denominator)])
    # solve sum_k z_k gam_k = s * c  <=>  [Mz | -c] (z, s) = 0
    ns = Mz.row_join(-cvec).nullspace()
    Zb = [v[:r, 0] for v in ns]                            # basis of Z_A (dimension r - 1)
    Ub = [sympy.Matrix([v[k] / sympy.Rational(lam[k].numerator, lam[k].denominator) for k in range(r)]) for v in Zb]
    ellB = len(Ub)
    G = sympy.Matrix.hstack(*Ub)                           # r x ellB generator of U_B
    # free coordinates 0..ellB-1: points Q = G * G[:ellB]^-1 * t for t in V^ellB
    Gf = G[:ellB, :]
    if Gf.det() == 0:
        return None
    T = G * Gf.inv()
    Tq = [[Fraction(int(sympy.fraction(x)[0]), int(sympy.fraction(x)[1])) for x in T.row(i)] for i in range(r)]
    Vs = set(V)
    pts = []
    for t in itertools.product(V, repeat=ellB):
        Q = [sum(Tq[i][j] * t[j] for j in range(ellB)) for i in range(r)]
        if all(x in Vs for x in Q):
            pts.append(Q)
    if len(pts) < NB:
        return None
    # d(Q) = beta(P*, Q) / c(P*) with P* = U_A-vector of u* = (1, 0) or (0, 1) having c != 0
    us = (Fraction(1), Fraction(0)) if c[0] != 0 else (Fraction(0), Fraction(1))
    Pst = [g[0] * us[0] + g[1] * us[1] for g in gam]
    cst = c[0] * us[0] + c[1] * us[1]
    vals = {sum(lam[k] * Pst[k] * Q[k] for k in range(r)) / cst for Q in pts}
    for dlt in sorted(v for v in vals if v > 0):
        if all((k * dlt) in vals and (-k * dlt) in vals for k in range(1, XB + 1)):
            return {"X_B": len(pts), "delta_B": str(dlt)}
    return None


def pin_partner(V, gam, c, r, NE, XE):
    """Given a covered hard-side code (coordinate functionals gam in Q^ellH, covering functional c) of a tight scheme
    with r products, pin lambda by the easy side's primal ratio lemma (its min-weight codewords are Lambda^-1 theta^T,
    |T| = ellH) and dual lemma (Lambda * kappa, kappa in ker c vanishing on ellH - 2 coordinates), then build and test
    the easy side exactly. Returns list of schemes."""
    import sympy
    ellH = len(c)
    R = diff_ratio_set(V)
    Rl = sorted(R)
    def solve(cols, rhs):
        Mx = sympy.Matrix([[sympy.Rational(col[t].numerator, col[t].denominator) for col in cols] for t in range(ellH)])
        sol = Mx.solve(sympy.Matrix([sympy.Rational(x.numerator, x.denominator) for x in rhs]))
        return [Fraction(int(sympy.fraction(x)[0]), int(sympy.fraction(x)[1])) for x in sol]
    th = {T: solve([gam[i] for i in T], c) for T in itertools.combinations(range(r), ellH)}
    if any(x == 0 for T in th for x in th[T]):
        return []
    allowed = {}
    for T, t in th.items():
        for a, b in itertools.combinations(range(ellH), 2):
            i, j = T[a], T[b]
            s = {t[b] / t[a] * x for x in Rl}
            allowed[(i, j)] = allowed[(i, j)] & s if (i, j) in allowed else s
    lam_sets = [[Fraction(1)]]
    for j in range(1, r):
        new = []
        for lam in lam_sets:
            for v in allowed.get((0, j), set()):
                if all(((i, j) not in allowed) or (v / lam[i]) in allowed[(i, j)] for i in range(1, j)):
                    new.append(lam + [v])
        lam_sets = new
    # dual lemma: kappa in ker c (as functional on Q^ellH), vanishing on ellH - 2 coordinates
    Kbasis = sympy.Matrix([[sympy.Rational(x.numerator, x.denominator) for x in c]]).nullspace()
    out = []
    for lam in lam_sets:
        ok = True
        for Z in itertools.combinations(range(r), ellH - 2):
            # u in ker c with gam_z . u = 0 for z in Z
            rows_ = [[sympy.Rational(x.numerator, x.denominator) for x in c]] + \
                    [[sympy.Rational(x.numerator, x.denominator) for x in gam[z]] for z in Z]
            ns = sympy.Matrix(rows_).nullspace()
            if len(ns) != 1:
                continue
            u = [Fraction(int(sympy.fraction(x)[0]), int(sympy.fraction(x)[1])) for x in ns[0]]
            kap = [lam[k] * sum(g[t] * u[t] for t in range(ellH)) for k, g in enumerate(gam)]
            nz = [x for k, x in enumerate(kap) if k not in Z]
            if any(x == 0 for x in nz) or not all((a / b) in R for a, b in itertools.combinations(nz, 2)):
                ok = False
                break
        if not ok:
            continue
        res = _check_B_gen(V, r, gam, c, lam, NE, XE)
        if res:
            out.append({"lambda": [str(x) for x in lam], **res})
    return out


def _check_B_gen(V, r, gam, c, lam, NB, XB):
    """General-dimension version of _check_B (hard side functionals gam in Q^ellH)."""
    import sympy
    ellH = len(c)
    Mz = sympy.Matrix([[sympy.Rational(g[t].numerator, g[t].denominator) for g in gam] for t in range(ellH)])
    cvec = sympy.Matrix([sympy.Rational(x.numerator, x.denominator) for x in c])
    ns = Mz.row_join(-cvec).nullspace()
    Zb = [v[:r, 0] for v in ns]
    Ub = [sympy.Matrix([v[k] / sympy.Rational(lam[k].numerator, lam[k].denominator) for k in range(r)]) for v in Zb]
    ellB = len(Ub)
    G = sympy.Matrix.hstack(*Ub)
    Gf = G[:ellB, :]
    if Gf.det() == 0:
        return None
    T = G * Gf.inv()
    Tq = [[Fraction(int(sympy.fraction(x)[0]), int(sympy.fraction(x)[1])) for x in T.row(i)] for i in range(r)]
    Vs = set(V)
    pts = []
    for t in itertools.product(V, repeat=ellB):
        Q = [sum(Tq[i][j] * t[j] for j in range(ellB)) for i in range(r)]
        if all(x in Vs for x in Q):
            pts.append(Q)
    if len(pts) < NB:
        return None
    us = [Fraction(0)] * ellH
    for t in range(ellH):
        if c[t] != 0:
            us[t] = Fraction(1)
            break
    Pst = [sum(g[t] * us[t] for t in range(ellH)) for g in gam]
    cst = sum(c[t] * us[t] for t in range(ellH))
    vals = {sum(lam[k] * Pst[k] * Q[k] for k in range(r)) / cst for Q in pts}
    for dlt in sorted(v for v in vals if v > 0):
        if all((k * dlt) in vals and (-k * dlt) in vals for k in range(1, XB + 1)):
            return {"X_B": len(pts), "delta_B": str(dlt)}
    return None


def cmd_small_cells():
    out = {}
    for MA, MB, ellB in ((6, 6, 2), (5, 6, 2), (5, 8, 3), (5, 9, 3)):
        import time
        t0 = time.time()
        st = tight_cell_l2(S_FP4, MA, MB, ellB)
        st["seconds"] = round(time.time() - t0, 1)
        out["%dx%d" % (MA, MB)] = st
        print(MA, MB, {k: v for k, v in st.items() if k != "schemes"}, "schemes:", len(st["schemes"]), flush=True)
    _save("small_cells.json", out)


def cmd_e2e_fp8():
    """End-to-end exactness test of the verified FP8 CRT scheme: random integer A (m x k), B (k x p) with
    k max|A| max|B| < P/2; residues -> FP8-exact planes (integers in [-16,16]) -> 32 float32-accumulated GEMMs ->
    per-modulus recombination -> CRT -> compare with the exact integer product."""
    d = json.load(open(os.path.join(OUT, "crt_verified.json"), encoding="utf-8"))["FP8_E4M3_16"]
    mods, recipes = d["moduli"], d["recipes"]
    P = math.prod(mods)
    rng = np.random.default_rng(7)
    out = []
    for (m_, k_, p_, bits) in ((8, 256, 8, 50), (16, 1024, 16, 48), (8, 65536, 8, 40)):
        A = [[int(x) for x in row] for row in rng.integers(-(2 ** bits), 2 ** bits, size=(m_, k_), dtype=np.int64)]
        Bm = [[int(x) for x in row] for row in rng.integers(-(2 ** bits), 2 ** bits, size=(k_, p_), dtype=np.int64)]
        assert k_ * (2 ** bits) ** 2 < P // 2
        exact = [[sum(A[i][l] * Bm[l][j] for l in range(k_)) for j in range(p_)] for i in range(m_)]
        gemms, residues = 0, []
        for mod, rc in zip(mods, recipes):
            Ar = np.array([[x % mod for x in row] for row in A], dtype=np.int64)
            Br = np.array([[x % mod for x in row] for row in Bm], dtype=np.int64)
            tab = rc["table"]
            if rc["products"] == 1:
                tA = np.array(tab, dtype=np.float32)
                G = tA[Ar] @ tA[Br]                                   # one FP8-exact GEMM, float32 accumulation
                R_ = np.rint(G).astype(np.int64) % mod
                gemms += 1
            else:
                s = rc["weight_s"]
                t0 = np.array([t[0] for t in tab], dtype=np.float32)
                t1 = np.array([t[1] for t in tab], dtype=np.float32)
                A0, A1, B0, B1 = t0[Ar], t1[Ar], t0[Br], t1[Br]
                G0, G1, G2 = A0 @ B0, A1 @ B1, (A0 + A1) @ (B0 + B1)      # three FP8-exact GEMMs
                G0, G1, G2 = (np.rint(x).astype(np.int64) for x in (G0, G1, G2))
                R_ = ((1 - s) * G0 + s * G2 + (s * s - s) * G1) % mod
                gemms += 3
            residues.append(R_)
        ok = True
        for i in range(m_):
            for j in range(p_):
                # CRT reconstruction to the symmetric range
                x = 0
                for mod, R_ in zip(mods, residues):
                    Mi = P // mod
                    x += int(R_[i, j]) * Mi * pow(Mi, -1, mod)
                x %= P
                if x > P // 2:
                    x -= P
                ok &= (x == exact[i][j])
        out.append({"shape": [m_, k_, p_], "input_bits": bits, "gemms": gemms, "exact_match": bool(ok)})
    _save("fp8_e2e.json", out)
    print(json.dumps(out))


def cmd_pairshare53():
    """Second, independent route for the INT8xINT8 r = 5 decision (monotonicity): a covered R-good [5,3] code (f, g)
    has covered [4,3] sub-codes (u, f.u) and (u, g.u) sharing the functional c. Reads codes43_M8*.json (covering
    functionals of every canonical [4,3] row), transports them to all sign/permutation variants, and tests every
    R-good pair of rows (column ratios, minors) with a common functional for coverage of X_f cap X_g."""
    import glob
    files = sorted(glob.glob(os.path.join(OUT, "codes43_M8_part*.json")))
    cov = {}
    for fn in files:
        for e in json.load(open(fn, encoding="utf-8"))["covered_all"]:
            h = tuple(Fraction(x) for x in e["h"])
            cov.setdefault(h, set()).add((e["q"], tuple(e["n"])))
    R = diff_ratio_set(S_FP4)
    U = box_points(S_FP4, 3)
    # transport to labelled variants: variant = sign/permutation of the canonical row applied to u-coordinates
    lab = {}
    for h, fs in cov.items():
        for perm in itertools.permutations(range(3)):
            for sg in itertools.product((1, -1), repeat=3):
                hv = tuple(sg[i] * h[perm[i]] for i in range(3))
                fl = 1 if hv[0] > 0 else -1
                hv = tuple(fl * x for x in hv)
                # u' with u'_i = sg_i u_perm(i): functional n on u becomes n'_i = sg_i n_perm(i)
                lab.setdefault(hv, set()).update((q, tuple(sg[i] * n[perm[i]] for i in range(3))) for q, n in fs)
    canon = [h for h in cov if all(x > 0 for x in h) and list(h) == sorted(h, reverse=True)]
    tested, hits = 0, []
    for f in canon:
        Ff = {(q, n) for q, n in lab.get(f, set())} | {(q, tuple(-t for t in n)) for q, n in lab.get(f, set())}
        for g, Fg in lab.items():
            if (f[0] / g[0]) not in R or (f[1] / g[1]) not in R or (f[2] / g[2]) not in R:
                continue
            m = (f[1] * g[2] - f[2] * g[1], f[2] * g[0] - f[0] * g[2], f[0] * g[1] - f[1] * g[0])
            if any(x == 0 for x in m):
                continue
            common = Ff & (Fg | {(q, tuple(-t for t in n)) for q, n in Fg})
            if not common:
                continue
            X = U[row_mask(S_FP4, U, f) & row_mask(S_FP4, U, g)]
            for q, n in common:
                tested += 1
                vals = set((X @ np.array(n, dtype=np.int64)).tolist())
                if all(k * q in vals and -k * q in vals for k in range(1, 128)):
                    hits.append({"f": [str(x) for x in f], "g": [str(x) for x in g], "q": q, "n": list(n)})
    out = {"covered_canonical_43_rows": len(cov), "pairs_with_common_functional_tested": tested, "hits": hits}
    _save("pairshare53.json", out)
    print(json.dumps({k: (v if k != "hits" else len(v)) for k, v in out.items()}))


def cmd_codes43():
    """One-sided decision for [4,3] plane codes: which R-good MDS [4,3] codes (u, h.u) carry a functional covering
    [-X, X] for width M (argument M, default 9)?  Positive control first: AWE's INT4xINT8 B-side code h = (1,2,2) must
    carry w = (5,3,45) at M = 8.  Usage: codes43 M [limit]."""
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    limit = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    part = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    nparts = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    X_ = width_range(M)
    N = 2 * X_ + 1
    U = box_points(S_FP4, 3)

    def dirs43(h):
        funcs = [(1, 0, 0), (0, 1, 0), (0, 0, 1), tuple(h)]
        return [_prim(_cross([Fraction(x) for x in funcs[j]], [Fraction(x) for x in funcs[l]]))
                for j, l in itertools.combinations(range(4), 2)]
    h0 = (Fraction(1), Fraction(2), Fraction(2))
    X0 = U[row_mask(S_FP4, U, h0)]
    ctrl = {tuple(int(t) for t in r) for r in covering_functionals_nb(X0, dirs43(h0), 1, K=127)}
    ctrl_ok = (5, 3, 45) in ctrl or (-5, -3, -45) in ctrl
    rows, _ = candidate_rows(S_FP4, 3, N)
    rows = rows[part::nparts][:limit]
    covered = []
    import time
    t0 = time.time()
    for h, g in rows:
        X = U[row_mask(S_FP4, U, h)]
        D = dirs43(h)
        for q in q_allowed(X):
            for n in covering_functionals_nb(X, D, q, K=X_):
                covered.append({"h": [str(x) for x in h], "q": q, "n": [int(t) for t in n]})
    out = {"M": M, "N": N, "positive_control_INT8_h122_finds_(5,3,45)": ctrl_ok, "control_functionals": len(ctrl),
           "rows_searched": len(rows), "covered": len(covered), "covered_rows": len({tuple(e["h"]) for e in covered}),
           "covered_all": covered, "seconds": round(time.time() - t0, 1)}
    tag = ("_limit%d" % limit if limit else "") + ("_part%dof%d" % (part, nparts) if nparts > 1 else "")
    _save("codes43_M%d%s.json" % (M, tag), out)
    print(json.dumps({k: v for k, v in out.items() if k != "covered_all"}))


def canonical53(f, g):
    """Exact canonical form of the [5,3] code (e1,e2,e3,f,g) under coordinate permutations and sign changes: the
    lexicographic minimum over the 120 (free triple, dependent pair) orderings of the sign-normalised systematic matrix."""
    one, zero = Fraction(1), Fraction(0)
    gam = [(one, zero, zero), (zero, one, zero), (zero, zero, one), tuple(f), tuple(g)]

    def inv3(Mx):
        a, b, c = Mx[0]; d, e, f_ = Mx[1]; g_, h, i = Mx[2]
        det = a * (e * i - f_ * h) - b * (d * i - f_ * g_) + c * (d * h - e * g_)
        adj = [[e * i - f_ * h, c * h - b * i, b * f_ - c * e],
               [f_ * g_ - d * i, a * i - c * g_, c * d - a * f_],
               [d * h - e * g_, b * g_ - a * h, a * e - b * d]]
        return [[x / det for x in row] for row in adj]
    best = None
    for T in itertools.permutations(range(5), 3):
        Minv = inv3([gam[t] for t in T])
        rest = [k for k in range(5) if k not in T]
        for d1, d2 in (rest, rest[::-1]):
            A = [[sum(gam[d][s] * Minv[s][c] for s in range(3)) for c in range(3)] for d in (d1, d2)]
            A = [[x if row[0] > 0 else -x for x in row] for row in A]
            for c in (1, 2):
                if A[0][c] < 0:
                    A[0][c], A[1][c] = -A[0][c], -A[1][c]
            key = tuple(A[0] + A[1])
            if best is None or key < best:
                best = key
    return best


def cmd_orbits53():
    codes = load_codes53()
    reps = json.load(open(os.path.join(OUT, "reps53.json")))
    seen, orb = {}, []
    import time
    t0 = time.time()
    for i in reps:
        f, g, _ = codes[i]
        k = canonical53(f, g)
        if k not in seen:
            seen[k] = i
            orb.append(i)
    _save("orbits53.json", orb)
    print(json.dumps({"representatives": len(reps), "orbits": len(orb), "seconds": round(time.time() - t0, 1)}))


def cmd_tight53_summary():
    """Aggregate the resumable progress files of the 10-way split; check that the searched codes are all orbits."""
    reps = json.load(open(os.path.join(OUT, "orbits53.json")))
    got, surv = set(), []
    for p in range(10):
        pf = os.path.join(OUT, "tight53_part%d_of_10.progress.jsonl" % p)
        if not os.path.exists(pf):
            continue
        for l in open(pf, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("idxs"):
                    got |= set(r["idxs"])
                surv += r["survivors"]
    keys = {}
    for s_ in surv:
        k = (s_["code"], tuple(s_["n"]), s_["q"])
        keys[k] = s_
    surv = list(keys.values())
    out = {"representatives": len(reps), "searched": len(got), "complete": got >= set(reps),
           "covered_codes": len({s_["code"] for s_ in surv}), "functionals_up_to_sign": len(surv), "survivor_list": surv}
    _save("tight53_summary.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "survivor_list"}))


def cmd_codes():
    out, info = good_codes_53()
    print(json.dumps(info, indent=1))
    print("largest |X|:", sorted(out, key=lambda t: -t[2])[:10])


def cmd_rows():
    res = {}
    for name, V, rho, N in [("FP4_rho3_N255", S_FP4, 3, 255), ("FP4_rho2_N31", S_FP4, 2, 31)]:
        rows, nR = candidate_rows(V, rho, N)
        res[name] = {"|R+|": nR, "n_rows": len(rows),
                     "rows": [([str(x) for x in f], g) for f, g in sorted(rows, key=lambda t: -t[1])]}
        print(name, "|R+| =", nR, "rows kept:", len(rows), "top:", [([str(x) for x in f], g) for f, g in sorted(rows, key=lambda t: -t[1])[:12]])
    _save("candidate_rows.json", res)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "awe"
    globals()["cmd_" + cmd]()
