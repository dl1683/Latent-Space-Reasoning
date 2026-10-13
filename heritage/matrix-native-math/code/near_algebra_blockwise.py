"""Blockwise-spectral defect law for a NEAR-ALGEBRA (follow-up to near_algebra_probe.py).
In the common eigenbasis of the exact small algebra A0 (Hamming scheme H(8,2): Walsh-Hadamard characters grouped by weight
a = 0..8, eigenspace dimension binom(8,a)) every A in A0 is diagonal with only d = 9 distinct eigenvalues.  A perturbation
E is stored ONLY through its d x d matrix of block spectral norms M_ab = ||Pi_a E Pi_b||_2 (d^2 numbers, certified).
Near-algebra element = (lambda in R^d, defect matrix D in R^{d x d} of block-norm bounds).  Product (law, rigorous by
submultiplicativity on blocks):  lambda' = lambda_C * lambda_A (pointwise: the 'Fourier trivialisation');
D' = diag(|lambda_C|) (eta M) + D diag(|lambda_A|) + D (eta M);  certified radius r = ||D||_2 (spectral norm of the d x d
matrix of block-norm bounds upper-bounds the operator norm of the block matrix).
Compare with the true distance from A0 and with the plain norm law.  Run: python code/near_algebra_blockwise.py [--selftest]"""
import sys, json
import numpy as np

def setup(m):
    n = 2 ** m
    idx = np.arange(n)
    pop = np.array([bin(i).count("1") for i in range(n)])
    W = np.array([[(-1) ** bin(i & j).count("1") for j in range(n)] for i in range(n)], dtype=float) / np.sqrt(n)
    groups = [np.where(pop == a)[0] for a in range(m + 1)]
    # eigenvalue of distance-k adjacency on weight-a characters = Krawtchouk K_k(a)
    from math import comb
    def kraw(k, a):
        return sum((-1) ** j * comb(a, j) * comb(m - a, k - j) for j in range(k + 1))
    Lam = np.array([[kraw(k, a) for a in range(m + 1)] for k in range(m + 1)], dtype=float)  # [k, a]
    return n, W, groups, Lam

def block_norms(Ehat, groups):
    d = len(groups)
    M = np.zeros((d, d))
    for a in range(d):
        for b in range(d):
            M[a, b] = np.linalg.norm(Ehat[np.ix_(groups[a], groups[b])], 2)
    return M

def run(m=8, etas=(1e-8, 1e-6, 1e-4, 1e-2), depths=(1, 2, 4, 8), seed=0, kind='sparse'):
    rng = np.random.default_rng(seed)
    n, W, groups, Lam = setup(m)
    d = m + 1
    out = []
    for eta in etas:
        P = np.eye(n)
        lamC = np.ones(d)
        D = np.zeros((d, d))
        rplain = 0.0
        Cn = 1.0
        for L in range(1, max(depths) + 1):
            coef = rng.standard_normal(d) / d
            lamA = coef @ Lam                      # eigenvalue vector of A = sum coef_k B_k
            A = W @ np.diag(np.repeat(0.0, n)) @ W  # placeholder (overwritten below)
            diag_full = np.zeros(n)
            for a in range(d):
                diag_full[groups[a]] = lamA[a]
            A = W @ np.diag(diag_full) @ W         # W symmetric orthogonal -> A = W diag W
            if kind == 'sparse':
                E = np.zeros((n, n))
                for i in range(n):
                    cols = rng.choice(n, 5, replace=False)
                    E[i, cols] = rng.standard_normal(5)
                E = (E + E.T) / 2
            elif kind == 'smooth_potential':      # diagonal potential cos(2 pi i / n): couples neighbouring weight classes
                E = np.diag(np.cos(2 * np.pi * np.arange(n) / n + rng.uniform(0, 6.28)))
            elif kind == 'lowrank':               # rank-2 perturbation u u^T + v v^T with random vectors
                u = rng.standard_normal(n); v = rng.standard_normal(n)
                E = np.outer(u, u) + np.outer(v, v)
            elif kind == 'aligned':               # adversarial: perturbation concentrated on the top eigenspaces (weights 0 and 1)
                g = np.concatenate([groups[0], groups[1]])
                Eh = np.zeros((n, n)); Eh[np.ix_(g, g)] = rng.standard_normal((len(g), len(g))); Eh = (Eh + Eh.T) / 2
                E = W @ Eh @ W
            E /= np.linalg.norm(E, 2)
            Ehat = W @ E @ W
            M = block_norms(Ehat, groups)
            X = A + eta * E
            P = P @ X
            # blockwise law
            D = np.diag(np.abs(lamC)) @ (eta * M) + D @ np.diag(np.abs(lamA)) + D @ (eta * M)
            rblock = np.linalg.norm(D, 2)
            # plain norm law
            rplain = Cn * eta + rplain * np.max(np.abs(lamA)) + rplain * eta
            lamC = lamC * lamA
            Cn = np.max(np.abs(lamC))
            if L in depths:
                # true distance of P from A0: in the eigenbasis, drop everything off the eigenspace-diagonal-constant structure:
                Phat = W @ P @ W
                # projection onto A0 = block-diagonal with scalar blocks equal to the block trace mean
                R = Phat.copy()
                for a in range(d):
                    g = groups[a]
                    R[np.ix_(g, g)] -= np.eye(len(g)) * (np.trace(Phat[np.ix_(g, g)]) / len(g))
                for a in range(d):
                    for b in range(d):
                        if a != b:
                            pass
                true = np.linalg.norm(R, 2)
                nP = np.linalg.norm(P, 2)
                out.append({"eta": eta, "depth": L, "true_rel": true / nP, "block_rel": rblock / nP, "plain_rel": rplain / nP,
                            "block_over_true": rblock / true if true > 0 else None, "plain_over_true": rplain / true if true > 0 else None})
    return out

def selftest():
    n, W, groups, Lam = setup(5)
    assert np.allclose(W @ W, np.eye(n))
    assert [len(g) for g in groups] == [1, 5, 10, 10, 5, 1]
    # exactness: elements of A0 are diagonal in the character basis with constant eigenvalue per weight class
    k = 2
    pop = np.array([bin(i).count("1") for i in range(n)])
    D = pop[np.arange(n)[:, None] ^ np.arange(n)[None, :]]
    B2 = (D == k).astype(float)
    Bh = W @ B2 @ W
    assert np.allclose(Bh, np.diag(np.diag(Bh)), atol=1e-9)
    print("selftest PASSED")

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    kind = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'sparse'
    res = run(kind=kind)
    print(kind)
    print(f"{'eta':>8} {'L':>2} {'true_rel':>10} {'block_rel':>10} {'plain_rel':>10} {'block/true':>10} {'plain/true':>10}")
    for r in res:
        print(f"{r['eta']:8.0e} {r['depth']:2d} {r['true_rel']:10.2e} {r['block_rel']:10.2e} {r['plain_rel']:10.2e} {r['block_over_true']:10.1f} {r['plain_over_true']:10.1f}")
    import os
    os.makedirs("results/near_algebra_probe_v1", exist_ok=True)
    json.dump(res, open(f"results/near_algebra_probe_v1/blockwise_{kind}.json", "w"), indent=1)
