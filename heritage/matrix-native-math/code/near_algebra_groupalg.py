"""Blockwise defect law on a NON-COMMUTATIVE exact algebra: the left-regular representation of S_n (C[S_n], dimension n!).
A0 = span{L_g}; its isotypic components Pi_i (dimension d_i^2) are the eigenspaces of a random central element (class sums);
elements of A0 preserve every Pi_i.  Perturbation E (random symmetric sparse) is stored through the nb x nb matrix of block
spectral norms M_ij = ||Pi_i E Pi_j||_2.  Law: D' = diag(||C Pi_i||) eta M + D diag(||A Pi_j||) + D eta M, radius = ||D||_2.
Compared with the TRUE distance of the exact product from A0 (Frobenius-orthogonal projection onto span{L_g}) and the plain norm
law.  Run: python code/near_algebra_groupalg.py [n] [--selftest]"""
import sys, json, itertools
import numpy as np

def perm_group(n):
    elems = list(itertools.permutations(range(n)))
    index = {p: i for i, p in enumerate(elems)}
    return elems, index

def left_regular(elems, index):
    N = len(elems)
    mats = []
    for g in elems:
        L = np.zeros((N, N))
        for j, h in enumerate(elems):
            gh = tuple(g[h[k]] for k in range(len(g)))
            L[index[gh], j] = 1.0
        mats.append(L)
    return mats

def cycle_type(p):
    seen = set(); ct = []
    for i in range(len(p)):
        if i not in seen:
            c = 0; j = i
            while j not in seen:
                seen.add(j); j = p[j]; c += 1
            ct.append(c)
    return tuple(sorted(ct))

def isotypic_projectors(elems, mats, rng):
    classes = {}
    for g, L in zip(elems, mats):
        classes.setdefault(cycle_type(g), np.zeros_like(L))
        classes[cycle_type(g)] += L
    Z = sum(rng.standard_normal() * K for K in classes.values())
    Z = (Z + Z.T) / 2
    w, V = np.linalg.eigh(Z)
    groups = []
    cur = [0]
    for i in range(1, len(w)):
        if abs(w[i] - w[cur[0]]) < 1e-7 * max(1, abs(w[i])):
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)
    return V, groups

def block_norms(Ehat, groups):
    nb = len(groups)
    M = np.zeros((nb, nb))
    for a in range(nb):
        for b in range(nb):
            M[a, b] = np.linalg.norm(Ehat[np.ix_(groups[a], groups[b])], 2)
    return M

def run(n=4, etas=(1e-8, 1e-6, 1e-4, 1e-2), depths=(1, 2, 4, 8), seed=0):
    rng = np.random.default_rng(seed)
    elems, index = perm_group(n)
    mats = left_regular(elems, index)
    N = len(elems)
    V, groups = isotypic_projectors(elems, mats, rng)
    dims = [len(g) for g in groups]
    out = []
    Lstack = np.stack(mats)                                  # N x N x N
    for eta in etas:
        P = np.eye(N); Cmat = np.eye(N)
        D = np.zeros((len(groups), len(groups))); rplain = 0.0
        for L in range(1, max(depths) + 1):
            c = rng.standard_normal(N) / np.sqrt(N)
            A = np.tensordot(c, Lstack, axes=1)
            E = np.zeros((N, N))
            for i in range(N):
                cols = rng.choice(N, min(5, N), replace=False)
                E[i, cols] = rng.standard_normal(len(cols))
            E = (E + E.T) / 2
            E /= np.linalg.norm(E, 2)
            Ehat = V.T @ E @ V
            M = block_norms(Ehat, groups)
            Ahat = V.T @ A @ V; Chat = V.T @ Cmat @ V
            nA = np.array([np.linalg.norm(Ahat[np.ix_(g, g)], 2) for g in groups])
            nC = np.array([np.linalg.norm(Chat[np.ix_(g, g)], 2) for g in groups])
            D = np.diag(nC) @ (eta * M) + D @ np.diag(nA) + D @ (eta * M)
            rblock = np.linalg.norm(D, 2)
            rplain = np.linalg.norm(Cmat, 2) * eta + rplain * np.linalg.norm(A, 2) + rplain * eta
            Cmat = Cmat @ A
            P = P @ (A + eta * E)
            if L in depths:
                coef = np.tensordot(Lstack, P, axes=([1, 2], [0, 1])) / N     # <L_g, P>/|G|
                res = P - np.tensordot(coef, Lstack, axes=1)
                true = np.linalg.norm(res, 2); nP = np.linalg.norm(P, 2)
                out.append({"n": n, "G": N, "blocks": len(groups), "eta": eta, "depth": L, "true_rel": true / nP,
                            "block_rel": rblock / nP, "plain_rel": rplain / nP,
                            "block_over_true": rblock / true if true > 0 else None,
                            "plain_over_true": rplain / true if true > 0 else None})
    return out

def selftest():
    elems, index = perm_group(3)
    mats = left_regular(elems, index)
    assert len(mats) == 6
    # closure: L_g L_h = L_{gh}
    for a in range(6):
        for b in range(6):
            P = mats[a] @ mats[b]
            assert any(np.array_equal(P, M) for M in mats)
    rng = np.random.default_rng(0)
    V, groups = isotypic_projectors(elems, mats, rng)
    assert sorted(len(g) for g in groups) == [1, 1, 4]       # S_3: irreps of dim 1,1,2 -> d^2 = 1,1,4
    print("selftest PASSED")

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    res = run(n=n)
    print(f"S_{n}: |G| = {res[0]['G']}, isotypic blocks = {res[0]['blocks']}")
    print(f"{'eta':>8} {'L':>2} {'true_rel':>10} {'block_rel':>10} {'plain_rel':>10} {'block/true':>10} {'plain/true':>10}")
    for r in res:
        print(f"{r['eta']:8.0e} {r['depth']:2d} {r['true_rel']:10.2e} {r['block_rel']:10.2e} {r['plain_rel']:10.2e} {r['block_over_true']:10.1f} {r['plain_over_true']:10.1f}")
    import os
    os.makedirs("results/near_algebra_probe_v1", exist_ok=True)
    json.dump(res, open(f"results/near_algebra_probe_v1/groupalg_S{n}.json", "w"), indent=1)
