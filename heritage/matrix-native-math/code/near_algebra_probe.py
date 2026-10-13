"""Near-algebra probe: a small exact algebra A0 (Bose-Mesner algebra of the Hamming scheme H(8,2), d = 9, n = 256)
plus a priced error budget.  A matrix X is stored as (coefficients a in R^9, radius r) meaning ||X - sum_i a_i B_i||_2 <= r.

Question (kill test from the 2026-10-02 theory meditation): for generators X_j = A_j + eta*E_j (A_j in A0, E_j a random
symmetric sparse perturbation of unit spectral norm), compose L of them.  Compare
  (1) the TRUE distance of the exact product from A0 (Frobenius-orthogonal projection residual, spectral norm),
  (2) the certified radius from the composition law  r_xy <= |C_x| r_y + r_x |C_y| + r_x r_y   (A0 is exactly closed, so the
      structure-constant defect is 0),
and report the largest depth at which the certified radius stays below tolerance times the product norm.
H0: the certified radius is within a small factor of the truth and useful depth is large for eta <= 1e-4.
H1 (kill): the certified radius is vacuous (>= tol) after a few products, or grows much faster than the truth.
Run: python code/near_algebra_probe.py [--selftest]
"""
import sys, json, itertools
import numpy as np

def hamming_basis(m):
    n = 2 ** m
    idx = np.arange(n)
    pop = np.array([bin(i).count("1") for i in range(n)])
    D = pop[idx[:, None] ^ idx[None, :]]
    return [(D == k).astype(np.float64) for k in range(m + 1)], n

def proj_coeffs(B, X):
    return np.array([np.sum(b * X) / np.sum(b * b) for b in B])

def synth(B, a):
    return sum(ai * bi for ai, bi in zip(a, B))

def spec(X):
    return np.linalg.norm(X, 2)

def run(m=8, etas=(0.0, 1e-8, 1e-6, 1e-4, 1e-2), depths=range(1, 9), seed=0, tol=1e-6):
    rng = np.random.default_rng(seed)
    B, n = hamming_basis(m)
    out = []
    for eta in etas:
        gens = []
        for j in range(max(depths)):
            a = rng.standard_normal(len(B)) / len(B)
            A = synth(B, a)
            E = np.zeros((n, n))
            for i in range(n):  # sparse symmetric perturbation, 5 per row, then unit spectral norm
                cols = rng.choice(n, 5, replace=False)
                E[i, cols] = rng.standard_normal(5)
            E = (E + E.T) / 2
            E /= spec(E)
            gens.append((a, A, A + eta * E, eta))
        # running exact product, A0 coefficient algebra (exact via structure constants = matrix products projected), radius law
        P = np.eye(n)
        Pa = np.zeros(len(B)); Pa[0] = 1.0   # coefficients of the A0-part (identity = B_0)
        Cmat = np.eye(n)
        r = 0.0
        rows = []
        for L, (a, A, X, e) in enumerate(gens, 1):
            P = P @ X
            rX = e  # ||X - A||_2 <= eta since ||E||_2 = 1
            prevC = spec(Cmat)                      # norm of the A0-part BEFORE multiplying (law: P_L = (C+D)(A+dX))
            r = prevC * rX + r * spec(A) + r * rX   # r_L <= |C_{L-1}| eta + r_{L-1} |A_L| + r_{L-1} eta (starts at r_0 = 0)
            Cmat = Cmat @ A                         # exact algebra: the A0-part stays in A0 (defect 0)
            Pa = proj_coeffs(B, Cmat)
            res = P - synth(B, proj_coeffs(B, P))
            true_dist = spec(res)
            nP = spec(P)
            if L in depths:
                out.append({"eta": eta, "depth": L, "true_dist": true_dist, "cert_radius": r, "norm_P": nP,
                            "true_rel": true_dist / nP, "cert_rel": r / nP, "useful": bool(r / nP < tol)})
    return out

def selftest():
    B, n = hamming_basis(5)
    assert len(B) == 6 and n == 32
    a = np.arange(1, 7, dtype=float)
    A = synth(B, a)
    assert np.allclose(proj_coeffs(B, A), a)
    # exact closure: product of two elements stays in the algebra
    P = synth(B, a) @ synth(B, a[::-1])
    assert np.linalg.norm(P - synth(B, proj_coeffs(B, P))) < 1e-9
    # negative control: a generic perturbation leaves the algebra
    rng = np.random.default_rng(0)
    E = rng.standard_normal((n, n)); E = E + E.T
    assert np.linalg.norm(E - synth(B, proj_coeffs(B, E))) > 1.0
    print("selftest PASSED")

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    res = run()
    print(f"{'eta':>8} {'L':>2} {'true_rel':>10} {'cert_rel':>10} {'ratio':>9} useful(<1e-6)")
    for r in res:
        ratio = (r['cert_rel'] / r['true_rel']) if r['true_rel'] > 0 else float('nan')
        print(f"{r['eta']:8.0e} {r['depth']:2d} {r['true_rel']:10.2e} {r['cert_rel']:10.2e} {ratio:9.2f} {r['useful']}")
    import os
    os.makedirs("results/near_algebra_probe_v1", exist_ok=True)
    json.dump(res, open("results/near_algebra_probe_v1/probe.json", "w"), indent=1)
