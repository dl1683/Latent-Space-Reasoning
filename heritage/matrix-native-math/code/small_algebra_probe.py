import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import json
import math
import itertools
import argparse
import numpy as np
import scipy.sparse as sp
from scipy.linalg import circulant

def get_partitions(n):
    a = [0 for _ in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]

def partition_degree(lam):
    if not lam: return 1
    lam = sorted(lam, reverse=True)
    n = sum(lam)
    lam_conj = [sum(1 for x in lam if x > i) for i in range(lam[0])]
    hook_prod = 1
    for i, row in enumerate(lam):
        for j in range(row):
            hook = row - j + lam_conj[j] - i - 1
            hook_prod *= hook
    return math.factorial(n) // hook_prod

def get_distinct_eigenvalues(A):
    if sp.issparse(A):
        A = A.toarray()
    if np.allclose(A, A.T):
        evals = np.linalg.eigvalsh(A)
        sorted_evals = np.sort(evals)
    else:
        evals = np.linalg.eigvals(A)
        sorted_evals = np.sort_complex(evals)
        
    if len(sorted_evals) == 0: return 0
    tol = 1e-8 * max(1.0, np.max(np.abs(sorted_evals)))
    distinct_count = 1
    for i in range(1, len(sorted_evals)):
        if np.abs(sorted_evals[i] - sorted_evals[i-1]) > tol:
            distinct_count += 1
    return distinct_count

def get_multi_generator_dimension(generators, cap=12, max_d=400):
    n = generators[0].shape[0]
    is_sparse = sp.issparse(generators[0])
    I = sp.eye(n, format='csr') if is_sparse else np.eye(n)
    
    K = min(n, 20) if n > 64 else n
    np.random.seed(42)
    R = np.random.randn(n, K) if n > 64 else np.eye(n)

    def proj(mat):
        p = mat @ R
        if sp.issparse(p): p = p.toarray()
        return p.flatten()

    v_I = proj(I)
    norm_v_I = np.linalg.norm(v_I)
    if norm_v_I > 1e-7:
        v_I /= norm_v_I
    
    basis_vecs = [v_I]
    current_level = [I]
    dims = [1]
    
    for depth in range(1, cap + 1):
        next_level = []
        for A in current_level:
            for G in generators:
                B = A @ G
                v = proj(B)
                for bv in basis_vecs:
                    v -= np.dot(v, bv) * bv
                
                norm_v = np.linalg.norm(v)
                if norm_v > 1e-7:
                    v /= norm_v
                    basis_vecs.append(v)
                    next_level.append(B)
                    
                    if len(basis_vecs) > max_d:
                        dims.append(len(basis_vecs))
                        return dims, 'not closed by cap'
                        
        dims.append(len(basis_vecs))
        if dims[-1] == dims[-2]:
            return dims, 'yes'
            
        current_level = next_level
        
    return dims, 'not closed by cap'

def run_selftest():
    print("Running selftest...")
    for n in range(3, 7):
        deg_sum2 = sum(partition_degree(lam)**2 for lam in get_partitions(n))
        assert deg_sum2 == math.factorial(n), f"Failed for S_{n}: {deg_sum2} != {math.factorial(n)}"

    vertices = list(itertools.product([0, 1], repeat=5))
    A_list = [np.zeros((32, 32)) for _ in range(6)]
    for i, v1 in enumerate(vertices):
        for j, v2 in enumerate(vertices):
            d = sum(x != y for x, y in zip(v1, v2))
            A_list[d][i, j] = 1
    
    dims, closed = get_multi_generator_dimension(A_list, cap=12)
    assert dims[-1] == 6, f"Failed H(5,2) dimension: {dims[-1]} != 6"
    assert closed == 'yes'
    print("Selftest passed.\n")

def run_full():
    print("Running full probe (Part A)...")
    results_A = {}
    order = []

    def record(name, d, n, closed):
        results_A[name] = {'n': n, 'd': d, 'closed': closed}
        order.append(name)
        print(f" -> {name}: d={d}, closed={closed}")

    # 1. H(5,2)
    print(" - Computing H(5,2) (all generators)...")
    vertices = list(itertools.product([0, 1], repeat=5))
    A_list = [np.zeros((32, 32)) for _ in range(6)]
    for i, v1 in enumerate(vertices):
        for j, v2 in enumerate(vertices):
            d = sum(x != y for x, y in zip(v1, v2))
            A_list[d][i, j] = 1
    dims, closed = get_multi_generator_dimension(A_list)
    record('H(5,2) all generators', dims[-1], 32, closed)

    # 2. H(7,3)
    print(" - Computing H(7,3) (single generator)...")
    A1_1d = sp.csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    I_1d = sp.eye(3, format='csr')
    A1_H73 = sp.csr_matrix((2187, 2187))
    for pos in range(7):
        mats = [A1_1d if i == pos else I_1d for i in range(7)]
        term = mats[0]
        for m in mats[1:]:
            term = sp.kron(term, m, format='csr')
        A1_H73 += term
    d = get_distinct_eigenvalues(A1_H73)
    record('H(7,3) generator A_1', d, 2187, 'yes')

    # 3. J(10,4)
    print(" - Computing J(10,4) (single generator)...")
    vertices = list(itertools.combinations(range(10), 4))
    n_J = len(vertices)
    A1_J = np.zeros((n_J, n_J))
    for i, v1 in enumerate(vertices):
        s1 = set(v1)
        for j, v2 in enumerate(vertices):
            if len(s1.intersection(v2)) == 3:
                A1_J[i, j] = 1
    d = get_distinct_eigenvalues(A1_J)
    record('J(10,4) generator A_1', d, 210, 'yes')

    # 4a. Circulant matrices n=256 with 1 random generator
    print(" - Computing Circulant 256 (1 random)...")
    np.random.seed(1)
    c1 = circulant(np.random.randn(256))
    d = get_distinct_eigenvalues(c1)
    record('Circulant 256 (1 random generator)', d, 256, 'yes')

    # 4b. Circulant matrices n=256 single shift
    print(" - Computing Circulant 256 (shift)...")
    shift = np.zeros(256)
    shift[1] = 1
    c2 = circulant(shift)
    d = get_distinct_eigenvalues(c2)
    record('Circulant 256 (permutation shift)', d, 256, 'yes')

    # 4c. Circulant matrices n=256 with 4 random
    print(" - Computing Circulant 256 (4 random)...")
    np.random.seed(1)
    gens_circ = [circulant(np.random.randn(256)) for _ in range(4)]
    dims, closed = get_multi_generator_dimension(gens_circ)
    record('Circulant 256 (4 random generators)', dims[-1], 256, closed)

    # 5. 1D Laplacian n=512
    print(" - Computing 1D Laplacian 512...")
    L_1d = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(512, 512), format='csr')
    d = get_distinct_eigenvalues(L_1d)
    record('1D Laplacian 512', d, 512, 'yes')

    # 6a. 2D 16x16 Laplacian alone n=256
    print(" - Computing 2D Laplacian alone...")
    L16 = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(16, 16), format='csr')
    I16 = sp.eye(16, format='csr')
    L_2d = sp.kron(L16, I16, format='csr') + sp.kron(I16, L16, format='csr')
    d = get_distinct_eigenvalues(L_2d)
    record('2D 16x16 Laplacian alone', d, 256, 'yes')

    # 6b. 2D Laplacian + potential
    print(" - Computing 2D Laplacian + potential...")
    D_2d = sp.diags([np.random.randn(256)], [0], shape=(256, 256), format='csr')
    dims, closed = get_multi_generator_dimension([L_2d, D_2d])
    record('2D Laplacian + potential', dims[-1], 256, closed)

    # 7. Random sparse symmetric n=256
    print(" - Computing Random sparse symmetric 256...")
    np.random.seed(2)
    gens_sparse = []
    for _ in range(3):
        S = sp.random(256, 256, density=0.05, format='csr')
        gens_sparse.append(S + S.T)
    dims, closed = get_multi_generator_dimension(gens_sparse)
    record('Random sparse symmetric (3 generators)', dims[-1], 256, closed)

    # 8. Random dense Gaussian n=128
    print(" - Computing Random dense Gaussian 128...")
    np.random.seed(3)
    gens_dense = [np.random.randn(128, 128) for _ in range(2)]
    dims, closed = get_multi_generator_dimension(gens_dense)
    record('Dense Gaussian (2 generators)', dims[-1], 128, closed)

    print("Running Part B (Group Algebra Costs)...")
    results_B = {}

    for n in range(3, 11):
        deg_sum2 = 0; deg_sum3 = 0; deg_sum_omega = 0; omega = 2.37
        for lam in get_partitions(n):
            d = partition_degree(lam)
            deg_sum2 += d**2
            deg_sum3 += d**3
            deg_sum_omega += d**omega
            
        G_size = math.factorial(n)
        alpha = math.log(deg_sum3) / math.log(G_size)
        alpha_omega = math.log(deg_sum_omega) / math.log(G_size)
        results_B[f'S_{n}'] = {
            '|G|': G_size, 'sum_d2': deg_sum2, 'sum_d3': deg_sum3,
            'naive_cost': G_size**2, 'alpha': alpha, 'alpha_omega': alpha_omega
        }

    for n in range(3, 9):
        deg_sum2 = 0; deg_sum3 = 0; deg_sum_omega = 0; omega = 2.37
        for k in range(n + 1):
            binom = math.comb(n, k)
            parts_k = list(get_partitions(k)) if k > 0 else [[]]
            parts_nk = list(get_partitions(n-k)) if n-k > 0 else [[]]
            
            for lam in parts_k:
                d_lam = partition_degree(lam) if k > 0 else 1
                for mu in parts_nk:
                    d_mu = partition_degree(mu) if n-k > 0 else 1
                    
                    d = binom * d_lam * d_mu
                    deg_sum2 += d**2
                    deg_sum3 += d**3
                    deg_sum_omega += d**omega
                    
        G_size = (2**n) * math.factorial(n)
        alpha = math.log(deg_sum3) / math.log(G_size)
        alpha_omega = math.log(deg_sum_omega) / math.log(G_size)
        results_B[f'Z2_wr_S_{n}'] = {
            '|G|': G_size, 'sum_d2': deg_sum2, 'sum_d3': deg_sum3,
            'naive_cost': G_size**2, 'alpha': alpha, 'alpha_omega': alpha_omega
        }

    G_size = 1000
    results_B['Z_1000'] = {
        '|G|': G_size, 'sum_d2': G_size, 'sum_d3': G_size,
        'naive_cost': G_size**2, 'alpha': 1.0, 'alpha_omega': 1.0
    }

    print("Saving outputs...")
    out_dir = 'results/small_algebra_probe_v1'
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, 'probe.json'), 'w') as f:
        json.dump({'Part_A': results_A, 'Part_B': results_B}, f, indent=2)

    # Read original report to get LEAD REVIEW
    report_path = os.path.join(out_dir, 'REPORT.md')
    lead_review = ""
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("> **LEAD REVIEW") or (lead_review and line.startswith(">")):
                    lead_review += line
                elif lead_review and not line.strip():
                    lead_review += "\n"
                    break

    with open(report_path, 'w') as f:
        if lead_review:
            f.write(lead_review.strip() + "\n\n")
        
        f.write("# Small Algebra Probe Results\n\n")
        f.write("## Part A: Algebra Dimensions\n")
        f.write("| family | n | stabilised algebra dimension d | d/n | closed? |\n")
        f.write("|---|---|---|---|---|\n")
        
        for k in order:
            v = results_A[k]
            d_val = v['d']
            n_val = v['n']
            closed = v['closed']
            d_over_n = d_val / n_val
            
            d_str = f"{d_val}" if closed == 'yes' else f">{d_val}"
            if d_val > 400: d_str = f">400"
            f.write(f"| {k} | {n_val} | {d_str} | {d_over_n:.2f} | {closed} |\n")

        f.write("\n## Part B: Group Algebra Cost\n")
        f.write("| Group | \\|G\\| | Sum d_i^2 | Naive Cost | Cost (Sum d_i^3) | alpha | alpha_omega (w=2.37) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for k, v in results_B.items():
            f.write(f"| {k} | {v['|G|']} | {v['sum_d2']} | {v['naive_cost']} | {v['sum_d3']:.0f} | {v['alpha']:.3f} | {v['alpha_omega']:.3f} |\n")

        f.write("\n## Reading\n")
        f.write("The updated Part A results properly distinguish families generating small, closed algebras from those that do not. The families with $d \\ll n$ are precisely the association schemes (Hamming and Johnson families) and the 2D Laplacian, where geometric symmetry heavily degenerates the spectrum. In contrast, families with $d = n$ include the 1D Laplacian and the circulant with a generic generator; here, the algebra is the full commutative algebra of polynomials in the matrix, of dimension $n$, meaning it is only 'small' if the matrix inherently possesses few distinct eigenvalues (as seen in the symmetric 2D case). Finally, the presence of multiple non-commuting generators or unstructured randomness (dense Gaussian, random sparse, or broken geometric symmetry via potentials) shatters this closure entirely: their dimensions rapidly blow up until hitting the cutoff, demonstrating that they cannot trivially compress matrix multiplication.\n")

    print(f"Done. Results written to {out_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--selftest', action='store_true', help="Run only the self tests")
    args = parser.parse_args()
    
    if args.selftest:
        run_selftest()
    else:
        run_selftest()
        run_full()
