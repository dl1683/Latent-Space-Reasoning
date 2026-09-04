"""Operator algebra analysis: extract K_A, K_M as 3x3 stochastic matrices on (C,L,R).

From Gate 2 v2 data (8 arms per context), infer the operator matrices and test:
1. Compositional prediction: does K_A K_M match the observed AM distribution?
2. Commutativity gap: how far is K_A K_M from K_M K_A?
3. Idempotency: is K_A^2 close to K_A?
4. Any stable algebraic relations between the operators?

The (C,L,R) simplex: C = P(correct digit), L = P(digit 9 = shadow), R = 1 - C - L.
"""
import json
import sys
import numpy as np


def dist_to_clr(dist, correct_digit):
    C = dist[correct_digit]
    L = dist[9]
    if correct_digit == 9:
        return None
    R = 1.0 - C - L
    return np.array([C, L, R])


def fit_operator(baseline_clrs, operated_clrs):
    """Fit a 3x3 stochastic matrix K such that operated ≈ K @ baseline.

    K[i,j] = P(output component i | input component j).
    Each column of K sums to 1 (column-stochastic).

    We fit by least squares: min ||operated - K @ baseline||^2
    subject to K >= 0, columns sum to 1.

    For simplicity, use unconstrained least squares and project.
    """
    n = len(baseline_clrs)
    B = np.column_stack(baseline_clrs)  # 3 x n
    O = np.column_stack(operated_clrs)  # 3 x n

    K, _, _, _ = np.linalg.lstsq(B.T, O.T, rcond=None)
    K = K.T  # 3x3

    K = np.maximum(K, 0)
    for j in range(3):
        s = K[:, j].sum()
        if s > 0:
            K[:, j] /= s

    return K


def matrix_tv(K1, K2):
    """Max column-wise TV between two stochastic matrices."""
    return np.max(0.5 * np.sum(np.abs(K1 - K2), axis=0))


def clr_tv(a, b):
    return 0.5 * np.sum(np.abs(a - b))


def analyze():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "experiments/results/svb_qwen3_composition_v2/result.json"
    with open(path) as f:
        data = json.load(f)

    obs = data["observations"]

    arm_clr = {}
    for key, o in obs.items():
        if o["val"] == 9:
            continue
        d = np.array(o["dist"])
        clr = dist_to_clr(d, o["val"])
        if clr is None:
            continue
        ctx = (o["depth"], o["var"], o["val"])
        arm_clr.setdefault(o["arm"], {})[ctx] = clr

    contexts = sorted(arm_clr.get("AM", {}).keys())
    n = len(contexts)
    print(f"=== OPERATOR ALGEBRA ANALYSIS ===\n")
    print(f"Contexts: {n}\n")

    baseline_clrs = [arm_clr["F_AF_M"][c] for c in contexts]
    a_alone_clrs = [arm_clr["AF_M"][c] for c in contexts]
    m_alone_clrs = [arm_clr["F_AM"][c] for c in contexts]
    am_clrs = [arm_clr["AM"][c] for c in contexts]
    ma_clrs = [arm_clr["MA"][c] for c in contexts]

    K_A = fit_operator(baseline_clrs, a_alone_clrs)
    K_M = fit_operator(baseline_clrs, m_alone_clrs)

    print("K_A (ASSERT operator, fitted on baseline -> A-alone):")
    print(f"  C  L  R")
    for i, label in enumerate(["C", "L", "R"]):
        print(f"  {label}: [{K_A[i,0]:.4f}  {K_A[i,1]:.4f}  {K_A[i,2]:.4f}]")

    print(f"\nK_M (MISLEADING operator, fitted on baseline -> M-alone):")
    print(f"  C  L  R")
    for i, label in enumerate(["C", "L", "R"]):
        print(f"  {label}: [{K_M[i,0]:.4f}  {K_M[i,1]:.4f}  {K_M[i,2]:.4f}]")

    K_AM = K_A @ K_M
    K_MA = K_M @ K_A

    print(f"\n--- COMPOSITIONAL PREDICTION ---")
    print(f"\nK_A @ K_M (predicted AM):")
    for i, label in enumerate(["C", "L", "R"]):
        print(f"  {label}: [{K_AM[i,0]:.4f}  {K_AM[i,1]:.4f}  {K_AM[i,2]:.4f}]")

    print(f"\nK_M @ K_A (predicted MA):")
    for i, label in enumerate(["C", "L", "R"]):
        print(f"  {label}: [{K_MA[i,0]:.4f}  {K_MA[i,1]:.4f}  {K_MA[i,2]:.4f}]")

    commutator_tv = matrix_tv(K_AM, K_MA)
    print(f"\nMatrix commutator TV: {commutator_tv:.4f}")

    am_pred_tvs = []
    ma_pred_tvs = []
    am_direct_tvs = []

    print(f"\n{'Context':<15} {'TV(AM,pred)':>11} {'TV(MA,pred)':>11} {'TV(AM,MA)':>10}")

    for i, ctx in enumerate(contexts):
        b = baseline_clrs[i]
        pred_am = K_AM @ b
        pred_am = np.maximum(pred_am, 0)
        pred_am /= pred_am.sum()

        pred_ma = K_MA @ b
        pred_ma = np.maximum(pred_ma, 0)
        pred_ma /= pred_ma.sum()

        actual_am = am_clrs[i]
        actual_ma = ma_clrs[i]

        tv_am = clr_tv(actual_am, pred_am)
        tv_ma = clr_tv(actual_ma, pred_ma)
        tv_direct = clr_tv(actual_am, actual_ma)

        am_pred_tvs.append(tv_am)
        ma_pred_tvs.append(tv_ma)
        am_direct_tvs.append(tv_direct)

        label = f"d{ctx[0]}_{ctx[1]}_{ctx[2]}"
        print(f"  {label:<13} {tv_am:11.4f} {tv_ma:11.4f} {tv_direct:10.4f}")

    print(f"\n  Compositional prediction quality:")
    print(f"    TV(AM, K_A@K_M@baseline): mean={np.mean(am_pred_tvs):.4f}, "
          f"median={np.median(am_pred_tvs):.4f}")
    print(f"    TV(MA, K_M@K_A@baseline): mean={np.mean(ma_pred_tvs):.4f}, "
          f"median={np.median(ma_pred_tvs):.4f}")
    print(f"    TV(AM, MA) direct:        mean={np.mean(am_direct_tvs):.4f}")

    print(f"\n--- ALGEBRAIC RELATIONS ---")

    K_A2 = K_A @ K_A
    K_M2 = K_M @ K_M
    idemp_A = matrix_tv(K_A2, K_A)
    idemp_M = matrix_tv(K_M2, K_M)
    print(f"\n  Idempotency test (K^2 vs K):")
    print(f"    K_A: TV = {idemp_A:.4f} {'(near-idempotent)' if idemp_A < 0.05 else ''}")
    print(f"    K_M: TV = {idemp_M:.4f} {'(near-idempotent)' if idemp_M < 0.05 else ''}")

    K_AMA = K_A @ K_M @ K_A
    K_MAM = K_M @ K_A @ K_M
    ama_vs_ma = matrix_tv(K_AMA, K_MA)
    ama_vs_a = matrix_tv(K_AMA, K_A)
    mam_vs_am = matrix_tv(K_MAM, K_AM)
    mam_vs_m = matrix_tv(K_MAM, K_M)
    print(f"\n  Braid-like relations:")
    print(f"    TV(AMA, MA) = {ama_vs_ma:.4f}")
    print(f"    TV(AMA, A)  = {ama_vs_a:.4f}")
    print(f"    TV(MAM, AM) = {mam_vs_am:.4f}")
    print(f"    TV(MAM, M)  = {mam_vs_m:.4f}")

    eigenvalues_A = np.linalg.eigvals(K_A)
    eigenvalues_M = np.linalg.eigvals(K_M)
    print(f"\n  Eigenvalues:")
    print(f"    K_A: {np.sort(np.abs(eigenvalues_A))[::-1]}")
    print(f"    K_M: {np.sort(np.abs(eigenvalues_M))[::-1]}")

    # Dominant eigenvalue should be 1 (stochastic matrix)
    print(f"\n  Spectral radius: K_A={max(abs(eigenvalues_A)):.4f}, "
          f"K_M={max(abs(eigenvalues_M)):.4f}")

    K_AM_eig = np.linalg.eigvals(K_AM)
    K_MA_eig = np.linalg.eigvals(K_MA)
    print(f"    K_AM: {np.sort(np.abs(K_AM_eig))[::-1]}")
    print(f"    K_MA: {np.sort(np.abs(K_MA_eig))[::-1]}")

    print(f"\n--- SUMMARY ---")
    comp_quality = np.mean(am_pred_tvs + ma_pred_tvs)
    print(f"  Compositional prediction error: {comp_quality:.4f}")
    if comp_quality < 0.02:
        print(f"  -> STRONG: matrix composition predicts observations well")
    elif comp_quality < 0.05:
        print(f"  -> MODERATE: matrix composition captures main structure")
    else:
        print(f"  -> WEAK: operators are context-dependent, not constant matrices")


if __name__ == "__main__":
    analyze()
