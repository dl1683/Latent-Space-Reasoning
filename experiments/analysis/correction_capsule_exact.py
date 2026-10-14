#!/usr/bin/env python3
"""
Exact-Arithmetic Correction Capsule Runner (Canonical Stdlib-Only CPU Implementation)

Grounding:
- Droid Round 1: internal/analysis_2026_10_14/droid_r1.md (Candidate 1, Section 2)
- Root Candidate Geometry: internal/analysis_2026_10_14/root_response_geometry.md
- Correction Ready Response: theory/CORRECTION_READY_RESPONSE.md (Sections 2 and 4)

Specifications:
- Stdlib-only: fractions.Fraction, json, sys, os, argparse. No numpy/torch/CUDA.
- Small Gauss-Jordan matrix inversion with explicit singular-pivot refusal.
- Explicit fixed JSON configuration: at most 12 tiny cases (12 cases implemented),
  dimension <= 8, no timing or statistical claim.
- Emits exact JSON arithmetic facts and assertion checks.
- Finite witness checks are NOT proofs of general theorems, quadratic bit lower bounds,
  or claims of novelty.
"""

from fractions import Fraction
import argparse
import json
import os
import sys


class SingularMatrixError(ValueError):
    """Raised when Gauss-Jordan elimination encounters a column with no non-zero pivot."""
    pass


def parse_fraction(val):
    """Parse integer, string fraction 'n/d', or Fraction into Fraction."""
    if isinstance(val, Fraction):
        return val
    if isinstance(val, int):
        return Fraction(val, 1)
    if isinstance(val, str):
        val = val.strip()
        if "/" in val:
            num, den = val.split("/", 1)
            return Fraction(int(num.strip()), int(den.strip()))
        return Fraction(int(val), 1)
    raise TypeError(f"Cannot parse value {val!r} of type {type(val)} to Fraction")


def format_fraction(f: Fraction) -> str:
    """Format Fraction as exact string 'n/d' or 'n'."""
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def mat_from_json(raw_matrix):
    """Convert JSON array of arrays of strings/ints into 2D Fraction matrix."""
    return [[parse_fraction(cell) for cell in row] for row in raw_matrix]


def mat_to_json(matrix):
    """Convert 2D Fraction matrix into JSON array of formatted fraction strings."""
    return [[format_fraction(cell) for cell in row] for row in matrix]


def identity(n: int):
    """Generate n x n identity matrix of Fractions."""
    return [[Fraction(1 if i == j else 0) for j in range(n)] for i in range(n)]


def zeros(rows: int, cols: int):
    """Generate rows x cols zero matrix of Fractions."""
    return [[Fraction(0) for _ in range(cols)] for _ in range(rows)]


def mat_add(A, B):
    """Matrix addition A + B."""
    r, c = len(A), len(A[0])
    return [[A[i][j] + B[i][j] for j in range(c)] for i in range(r)]


def mat_sub(A, B):
    """Matrix subtraction A - B."""
    r, c = len(A), len(A[0])
    return [[A[i][j] - B[i][j] for j in range(c)] for i in range(r)]


def mat_scale(scalar, A):
    """Scalar multiplication s * A."""
    s = parse_fraction(scalar)
    return [[s * cell for cell in row] for row in A]


def mat_transpose(A):
    """Matrix transpose A^T."""
    r, c = len(A), len(A[0])
    return [[A[i][j] for i in range(r)] for j in range(c)]


def mat_mul(A, B):
    """Matrix multiplication A * B."""
    rA, cA = len(A), len(A[0])
    rB, cB = len(B), len(B[0])
    if cA != rB:
        raise ValueError(f"Matrix dimension mismatch: {rA}x{cA} * {rB}x{cB}")
    return [[sum(A[i][k] * B[k][j] for k in range(cA)) for j in range(cB)] for i in range(rA)]


def outer_product(u, v):
    """Rank-one outer product u * v^T where u and v are vectors (lists of Fractions)."""
    u_f = [parse_fraction(x) for x in u]
    v_f = [parse_fraction(x) for x in v]
    return [[u_f[i] * v_f[j] for j in range(len(v_f))] for i in range(len(u_f))]


def mat_inv(A):
    """
    Gauss-Jordan elimination with explicit singular-pivot refusal.
    Inverts square matrix A with Fraction entries.
    """
    n = len(A)
    for row in A:
        if len(row) != n:
            raise ValueError(f"Matrix must be square: {len(A)}x{len(row)}")

    # Augmented matrix [A | I_n]
    aug = [
        [Fraction(A[i][j]) for j in range(n)] + [Fraction(1 if i == j else 0) for j in range(n)]
        for i in range(n)
    ]

    for col in range(n):
        # Pivot selection at or below row col
        pivot_row = None
        for r in range(col, n):
            if aug[r][col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            raise SingularMatrixError(f"Singular matrix: pivot refusal at column {col}")

        # Swap rows
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        # Normalize pivot row
        pivot_val = aug[col][col]
        aug[col] = [val / pivot_val for val in aug[col]]

        # Eliminate other rows
        for r in range(n):
            if r != col and aug[r][col] != 0:
                factor = aug[r][col]
                aug[r] = [aug[r][c] - factor * aug[col][c] for c in range(2 * n)]

    return [[aug[i][n + j] for j in range(n)] for i in range(n)]


def test_singular_pivot_refusal():
    """Verify that singular-pivot refusal triggers on a known singular matrix."""
    singular_mat = [
        [Fraction(1), Fraction(1)],
        [Fraction(1), Fraction(1)]
    ]
    try:
        mat_inv(singular_mat)
        return False, "Failed to refuse singular pivot"
    except SingularMatrixError as e:
        return True, str(e)


def execute_fixture_1(case_cfg):
    """
    Fixture 1:
    A1=[[2,1],[1,1]], A2=[[3,2],[2,2]], C=1: compute S,F,G; verify common S=F=1,
    G=1 and 1/2; corrected interior +1 gives 3/2 and 5/3.
    """
    inputs = case_cfg["inputs"]
    A = mat_from_json(inputs["A"])
    C = mat_from_json(inputs["C"])
    b_idx = inputs["boundary_indices"]
    i_idx = inputs["interior_indices"]
    D = mat_from_json(inputs["interior_perturbation_D"])

    # Partition blocks
    A_BB = [[A[i][j] for j in b_idx] for i in b_idx]
    A_BI = [[A[i][j] for j in i_idx] for i in b_idx]
    A_IB = [[A[i][j] for j in b_idx] for i in i_idx]
    A_II = [[A[i][j] for j in i_idx] for i in i_idx]

    inv_A_II = mat_inv(A_II)

    # S = A_BB - A_BI * A_II^{-1} * A_IB
    S_mat = mat_sub(A_BB, mat_mul(mat_mul(A_BI, inv_A_II), A_IB))
    S_val = S_mat[0][0]

    # F = A_BI * A_II^{-1} * C
    F_mat = mat_mul(mat_mul(A_BI, inv_A_II), C)

    # G = C^T * A_II^{-1} * C
    G_mat = mat_mul(mat_mul(mat_transpose(C), inv_A_II), C)

    # Corrected interior: A_II_corr = A_II + C * D * C^T
    CDC = mat_mul(mat_mul(C, D), mat_transpose(C))
    A_II_corr = mat_add(A_II, CDC)
    inv_A_II_corr = mat_inv(A_II_corr)

    # Direct Schur: S_direct = A_BB - A_BI * (A_II_corr)^{-1} * A_IB
    S_direct_mat = mat_sub(A_BB, mat_mul(mat_mul(A_BI, inv_A_II_corr), A_IB))
    S_direct_val = S_direct_mat[0][0]

    # Capsule response: R(G, D) = D * (I + G * D)^{-1}
    m = len(G_mat)
    I_m = identity(m)
    I_plus_GD = mat_add(I_m, mat_mul(G_mat, D))
    inv_I_plus_GD = mat_inv(I_plus_GD)
    R = mat_mul(D, inv_I_plus_GD)

    # S_capsule(D) = S + F * R * F^T
    FRFT = mat_mul(mat_mul(F_mat, R), mat_transpose(F_mat))
    S_capsule_mat = mat_add(S_mat, FRFT)
    S_capsule_val = S_capsule_mat[0][0]

    assertions = {
        "S_is_1": S_val == Fraction(1),
        "F_is_1": F_mat[0][0] == Fraction(1),
        "direct_equals_capsule": S_direct_val == S_capsule_val,
    }

    if case_cfg["case_id"] == "case_01_fixture1_A1":
        assertions["G_is_1"] = G_mat[0][0] == Fraction(1)
        assertions["S_direct_corrected_is_3_2"] = S_direct_val == Fraction(3, 2)
        assertions["S_capsule_corrected_is_3_2"] = S_capsule_val == Fraction(3, 2)
    elif case_cfg["case_id"] == "case_02_fixture1_A2":
        assertions["G_is_1_2"] = G_mat[0][0] == Fraction(1, 2)
        assertions["S_direct_corrected_is_5_3"] = S_direct_val == Fraction(5, 3)
        assertions["S_capsule_corrected_is_5_3"] = S_capsule_val == Fraction(5, 3)

    return {
        "case_id": case_cfg["case_id"],
        "S": format_fraction(S_val),
        "F": mat_to_json(F_mat),
        "G": mat_to_json(G_mat),
        "S_direct_corrected": format_fraction(S_direct_val),
        "S_capsule_corrected": format_fraction(S_capsule_val),
        "assertions": assertions,
        "all_passed": all(assertions.values()),
    }


def execute_fixture_2(case_cfg):
    """
    Fixture 2:
    G=[[1,1/32],[1/32,1]], F=e1, S=1, D diagonal 0, partial dyadic and I.
    Verify actual S(D)=1+e1^T D(I+GD)^(-1)e1, proxy 1+d1/(1+d1), absolute error <= 1/7168.
    """
    inputs = case_cfg["inputs"]
    G = mat_from_json(inputs["G"])
    F = mat_from_json(inputs["F"])
    S = parse_fraction(inputs["S"])
    D = mat_from_json(inputs["D"])

    m = len(G)
    I_m = identity(m)
    I_plus_GD = mat_add(I_m, mat_mul(G, D))
    inv_I_plus_GD = mat_inv(I_plus_GD)
    R = mat_mul(D, inv_I_plus_GD)

    # Actual S(D) = S + F * R * F^T
    FRFT = mat_mul(mat_mul(F, R), mat_transpose(F))
    S_actual = S + FRFT[0][0]

    # Proxy: S_proxy = 1 + d1 / (1 + d1)
    d1 = D[0][0]
    S_proxy = Fraction(1) + d1 / (Fraction(1) + d1)

    abs_error = abs(S_actual - S_proxy)
    bound_7168 = Fraction(1, 7168)

    assertions = {
        "error_within_1_7168": abs_error <= bound_7168
    }

    return {
        "case_id": case_cfg["case_id"],
        "S_actual": format_fraction(S_actual),
        "S_proxy": format_fraction(S_proxy),
        "absolute_error": format_fraction(abs_error),
        "bound_1_7168": format_fraction(bound_7168),
        "assertions": assertions,
        "all_passed": all(assertions.values()),
    }


def execute_fixture_3(case_cfg):
    """
    Fixture 3:
    Same G with offdiag negative and D=1/2*[1,1][1,1]^T; verify two true responses differ
    and share baseline. Report exact gap; this finite witness is NOT quadratic bit proof.
    """
    inputs = case_cfg["inputs"]
    G_plus = mat_from_json(inputs["G_plus"])
    G_minus = mat_from_json(inputs["G_minus"])
    F = mat_from_json(inputs["F"])
    S = parse_fraction(inputs["S"])
    D = mat_from_json(inputs["D"])

    # Baseline D=0 responses
    D_zero = zeros(2, 2)
    R_plus_base = mat_mul(D_zero, mat_inv(identity(2)))
    R_minus_base = mat_mul(D_zero, mat_inv(identity(2)))
    S_base_plus = S + mat_mul(mat_mul(F, R_plus_base), mat_transpose(F))[0][0]
    S_base_minus = S + mat_mul(mat_mul(F, R_minus_base), mat_transpose(F))[0][0]

    # True response for G_plus
    I_plus_GplusD = mat_add(identity(2), mat_mul(G_plus, D))
    R_plus = mat_mul(D, mat_inv(I_plus_GplusD))
    S_plus = S + mat_mul(mat_mul(F, R_plus), mat_transpose(F))[0][0]

    # True response for G_minus
    I_plus_GminusD = mat_add(identity(2), mat_mul(G_minus, D))
    R_minus = mat_mul(D, mat_inv(I_plus_GminusD))
    S_minus = S + mat_mul(mat_mul(F, R_minus), mat_transpose(F))[0][0]

    exact_gap = abs(S_plus - S_minus)
    epsilon0 = Fraction(1, 4096)

    assertions = {
        "share_baseline": S_base_plus == Fraction(1) and S_base_minus == Fraction(1),
        "responses_differ": S_plus != S_minus,
        "gap_is_32_4095": exact_gap == Fraction(32, 4095),
        "gap_exceeds_2_epsilon0": exact_gap > Fraction(2, 4096),
    }

    return {
        "case_id": case_cfg["case_id"],
        "S_G_plus_baseline": format_fraction(S_base_plus),
        "S_G_minus_baseline": format_fraction(S_base_minus),
        "S_G_plus_D": format_fraction(S_plus),
        "S_G_minus_D": format_fraction(S_minus),
        "exact_gap": format_fraction(exact_gap),
        "two_epsilon0": format_fraction(Fraction(2, 4096)),
        "assertions": assertions,
        "all_passed": all(assertions.values()),
        "scientific_note": (
            "Finite witness separating two packed worlds by exact gap 32/4095 > 2*epsilon0 "
            "(epsilon0=1/4096); this finite witness is NOT quadratic bit proof."
        ),
    }


def execute_fixture_4(case_cfg):
    """
    Fixture 4:
    G built from 4x4 Hadamard H and offblock H/64, diagonals I, two fixed diagonal D
    fixtures in [0,1]. Check H^T H=4I and exact responses within 1/7168.
    """
    inputs = case_cfg["inputs"]
    H = mat_from_json(inputs["H"])
    scale = parse_fraction(inputs["offblock_scale"])
    F = mat_from_json(inputs["F"])
    S = parse_fraction(inputs["S"])

    # Verify H^T H = 4 * I_4
    HT = mat_transpose(H)
    HTH = mat_mul(HT, H)
    four_I4 = mat_scale(Fraction(4), identity(4))
    hth_is_4I = (HTH == four_I4)

    # Build 8x8 G
    G = identity(8)
    for i in range(4):
        for j in range(4):
            G[i][4 + j] = H[i][j] * scale
            G[4 + i][j] = HT[i][j] * scale

    if "Da_diagonal" in inputs:
        d_diag = [parse_fraction(x) for x in inputs["Da_diagonal"]]
    else:
        d_diag = [parse_fraction(x) for x in inputs["Db_diagonal"]]

    D = [[d_diag[i] if i == j else Fraction(0) for j in range(8)] for i in range(8)]

    I8 = identity(8)
    I_plus_GD = mat_add(I8, mat_mul(G, D))
    inv_I_plus_GD = mat_inv(I_plus_GD)
    R = mat_mul(D, inv_I_plus_GD)

    S_actual = S + mat_mul(mat_mul(F, R), mat_transpose(F))[0][0]
    d1 = d_diag[0]
    S_proxy = Fraction(1) + d1 / (Fraction(1) + d1)
    abs_error = abs(S_actual - S_proxy)
    bound_7168 = Fraction(1, 7168)

    assertions = {
        "hadamard_orthogonal_HTH_is_4I": hth_is_4I,
        "error_within_1_7168": abs_error <= bound_7168,
    }

    return {
        "case_id": case_cfg["case_id"],
        "HTH_is_4I": hth_is_4I,
        "S_actual": format_fraction(S_actual),
        "S_proxy": format_fraction(S_proxy),
        "absolute_error": format_fraction(abs_error),
        "bound_1_7168": format_fraction(bound_7168),
        "assertions": assertions,
        "all_passed": all(assertions.values()),
    }


def execute_fixture_5(case_cfg):
    """
    Fixture 5:
    Verify the ordered resolvent identity R=R0-R0 Delta R0+R0 Delta R0 Delta R
    for a singular PSD D fixture and a noncommuting rational PSD D fixture in 2x2.
    Construct D by explicit rational rankone outerproducts so PSD is not assumed from
    entry signs. R0=D(I+D)^(-1), Delta=G-I. Check equality in exact fractions.
    """
    inputs = case_cfg["inputs"]
    G = mat_from_json(inputs["G"])
    Delta = mat_from_json(inputs["Delta"])
    case_id = case_cfg["case_id"]

    I2 = identity(2)

    if case_id == "case_09_fixture5_resolvent_singular_PSD":
        u = [parse_fraction(x) for x in inputs["rank_one_generator_u"]]
        # Explicit rank-one outer product
        D = outer_product(u, u)
        det_D = D[0][0] * D[1][1] - D[0][1] * D[1][0]
        singular_check = (det_D == Fraction(0))
        noncomm_check = None
    else:
        # Noncommuting: sum of two explicit rank-one outer products
        gen = inputs["generators"]
        u1 = [parse_fraction(x) for x in gen[0]]
        u2 = [parse_fraction(x) for x in gen[1]]
        D = mat_add(outer_product(u1, u1), outer_product(u2, u2))
        D_Delta = mat_mul(D, Delta)
        Delta_D = mat_mul(Delta, D)
        singular_check = None
        noncomm_check = (D_Delta != Delta_D)

    # R0 = D * (I + D)^{-1}
    inv_I_plus_D = mat_inv(mat_add(I2, D))
    R0 = mat_mul(D, inv_I_plus_D)

    # R = D * (I + G * D)^{-1}
    inv_I_plus_GD = mat_inv(mat_add(I2, mat_mul(G, D)))
    R = mat_mul(D, inv_I_plus_GD)

    # Ordered resolvent RHS = R0 - R0 * Delta * R0 + R0 * Delta * R0 * Delta * R
    t1 = R0
    t2 = mat_mul(mat_mul(R0, Delta), R0)
    t3 = mat_mul(mat_mul(mat_mul(mat_mul(R0, Delta), R0), Delta), R)
    RHS = mat_add(mat_sub(t1, t2), t3)

    resolvent_identity_holds = (R == RHS)

    assertions = {
        "ordered_resolvent_identity_exact": resolvent_identity_holds
    }

    res = {
        "case_id": case_id,
        "resolvent_R": mat_to_json(R),
        "resolvent_RHS": mat_to_json(RHS),
        "resolvent_identity_holds": resolvent_identity_holds,
    }

    if singular_check is not None:
        assertions["det_D_is_zero"] = singular_check
        res["det_D"] = format_fraction(det_D)
        res["is_singular"] = singular_check
    if noncomm_check is not None:
        assertions["noncommuting_with_Delta"] = noncomm_check
        res["noncommuting_with_Delta"] = noncomm_check

    res["assertions"] = assertions
    res["all_passed"] = all(assertions.values())
    return res


def execute_fixture_6(case_cfg):
    """
    Fixture 6:
    Ordered additions and a supplied-term retraction keep actual current D and yield
    the same response as direct solve of that D; state historical terms not supplied free.
    """
    inputs = case_cfg["inputs"]
    G = mat_from_json(inputs["G"])
    F = mat_from_json(inputs["F"])
    S = parse_fraction(inputs["S"])

    s1 = inputs["step_1_add_T1"]
    alpha1 = parse_fraction(s1["alpha"])
    v1 = [parse_fraction(x) for x in s1["v"]]
    T1 = mat_scale(alpha1, outer_product(v1, v1))

    s2 = inputs["step_2_add_T2"]
    alpha2 = parse_fraction(s2["alpha"])
    v2 = [parse_fraction(x) for x in s2["v"]]
    T2 = mat_scale(alpha2, outer_product(v2, v2))

    # Initial state D_0 = 0
    D = zeros(2, 2)
    # Step 1: add T1
    D = mat_add(D, T1)
    # Step 2: add T2
    D = mat_add(D, T2)
    # Step 3: retract supplied term T1
    supplied_T1 = mat_from_json(inputs["step_3_retract_supplied_T1"]["supplied_term"])
    D_current = mat_sub(D, supplied_T1)

    # Response evaluated at D_current
    I2 = identity(2)
    inv_current = mat_inv(mat_add(I2, mat_mul(G, D_current)))
    R_current = mat_mul(D_current, inv_current)
    S_current = S + mat_mul(mat_mul(F, R_current), mat_transpose(F))[0][0]

    # Direct solve of T2
    inv_direct = mat_inv(mat_add(I2, mat_mul(G, T2)))
    R_direct = mat_mul(T2, inv_direct)
    S_direct = S + mat_mul(mat_mul(F, R_direct), mat_transpose(F))[0][0]

    assertions = {
        "current_D_equals_T2": (D_current == T2),
        "S_current_equals_S_direct": (S_current == S_direct),
        "S_current_is_113_97": (S_current == Fraction(113, 97)),
    }

    return {
        "case_id": case_cfg["case_id"],
        "D_current": mat_to_json(D_current),
        "T2": mat_to_json(T2),
        "S_current": format_fraction(S_current),
        "S_direct_T2": format_fraction(S_direct),
        "assertions": assertions,
        "all_passed": all(assertions.values()),
        "accounting_statement": (
            "Historical terms are not supplied free: retraction requires caller to supply "
            "the term (or charged event log); current state retains only cumulative D."
        ),
    }


def execute_fixture_7(case_cfg):
    """
    Fixture 7:
    Eight-port acquisition witness from theory/CORRECTION_READY_RESPONSE.md sections 2 and 4.
    Demonstrates certificate non-monotonicity: acquisition of port 2 improves proxy response
    to exact true world (error 1/496 -> 0), but certified bound k+b strictly worsens
    from 7/64 to 744/6727 (increase of 527/430528).
    Interval intersection repair preserves containment and nonincreasing width.
    Completions G +/- offdiag34/4 yield responses 17/16 and 16/15 with gap 1/240 > 2*epsilon0.
    """
    inputs = case_cfg["inputs"]
    dim = inputs.get("dimension", 8)
    G = mat_from_json(inputs["G"])
    F = mat_from_json(inputs["F"])
    S = parse_fraction(inputs["S"])
    D = mat_from_json(inputs["D"])
    envelope = parse_fraction(inputs["generic_offdiag_envelope"])
    K_initial = inputs["K_initial"]
    K_promoted = inputs["K_promoted"]
    G_plus = mat_from_json(inputs["G_plus"])
    G_minus = mat_from_json(inputs["G_minus"])

    I_dim = identity(dim)

    # True world response y = S + F * R_G(D) * F^T
    inv_I_plus_GD = mat_inv(mat_add(I_dim, mat_mul(G, D)))
    R_G = mat_mul(D, inv_I_plus_GD)
    y = S + mat_mul(mat_mul(F, R_G), mat_transpose(F))[0][0]

    def solve_certificate(K_indices):
        # Q = G_{KK} \oplus I_{J \setminus K}
        Q = identity(dim)
        for i in K_indices:
            for j in K_indices:
                Q[i][j] = G[i][j]

        # H envelope: 0 on diagonal, 0 on acquired principal block K x K,
        # envelope on all other unacquired off-diagonals
        H = zeros(dim, dim)
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    H[i][j] = Fraction(0)
                elif i in K_indices and j in K_indices:
                    H[i][j] = Fraction(0)
                else:
                    H[i][j] = envelope

        # Proxy solve: x = R_Q(D) f = D (I + Q D)^{-1} f
        inv_I_plus_QD = mat_inv(mat_add(I_dim, mat_mul(Q, D)))
        R_Q = mat_mul(D, inv_I_plus_QD)
        f_col = mat_transpose(F)
        x_col = mat_mul(R_Q, f_col)
        x = [x_col[i][0] for i in range(dim)]

        # q = H |x|
        abs_x = [abs(val) for val in x]
        q = [sum(H[i][j] * abs_x[j] for j in range(dim)) for i in range(dim)]

        # k = |x|^T H |x| = |x|^T q
        k = sum(abs_x[i] * q[i] for i in range(dim))

        # b = min((4/7) ||q||_2^2, q^T |D| q)
        norm_q_sq = sum(q[i] * q[i] for i in range(dim))
        b_cand_1 = Fraction(4, 7) * norm_q_sq
        abs_D = [[abs(D[i][j]) for j in range(dim)] for i in range(dim)]
        D_q = [sum(abs_D[i][j] * q[j] for j in range(dim)) for i in range(dim)]
        b_cand_2 = sum(q[i] * D_q[i] for i in range(dim))
        b = min(b_cand_1, b_cand_2)

        # Proxy response p = S + f^T x
        p = S + sum(F[0][i] * x[i] for i in range(dim))

        # Output interval [p - k, p + k + b]
        interval = (p - k, p + k + b)
        width = interval[1] - interval[0]

        return {
            "Q": Q,
            "H": H,
            "x": x,
            "q": q,
            "k": k,
            "b": b,
            "k_plus_b": k + b,
            "p": p,
            "interval": interval,
            "width": width,
        }

    cert1 = solve_certificate(K_initial)
    cert2 = solve_certificate(K_promoted)

    true_initial_error = abs(y - cert1["p"])
    true_promoted_error = abs(y - cert2["p"])

    # Output intervals and intersection
    L1, U1 = cert1["interval"]
    L2, U2 = cert2["interval"]
    L_int = max(L1, L2)
    U_int = min(U1, U2)
    intersection_interval = (L_int, U_int)
    width_intersection = U_int - L_int

    # Completions G+ and G-
    inv_Gplus = mat_inv(mat_add(I_dim, mat_mul(G_plus, D)))
    R_plus = mat_mul(D, inv_Gplus)
    S_plus = S + mat_mul(mat_mul(F, R_plus), mat_transpose(F))[0][0]

    inv_Gminus = mat_inv(mat_add(I_dim, mat_mul(G_minus, D)))
    R_minus = mat_mul(D, inv_Gminus)
    S_minus = S + mat_mul(mat_mul(F, R_minus), mat_transpose(F))[0][0]

    completions_gap = abs(S_plus - S_minus)

    # Containment and nonincreasing width
    interval_1_contains_y = (L1 <= y <= U1)
    interval_2_contains_y = (L2 <= y <= U2)
    intersection_contains_y = (L_int <= y <= U_int)
    intersection_contained_in_interval_1 = (L1 <= L_int and U_int <= U1)
    intersection_nonincreasing_width = (width_intersection <= cert1["width"])

    assertions = {
        "k1_plus_b1_is_7_64": cert1["k_plus_b"] == Fraction(7, 64),
        "k2_plus_b2_is_744_6727": cert2["k_plus_b"] == Fraction(744, 6727),
        "strict_increase_k_plus_b": (cert2["k_plus_b"] > cert1["k_plus_b"]) and ((cert2["k_plus_b"] - cert1["k_plus_b"]) == Fraction(527, 430528)),
        "true_initial_error_is_1_496": true_initial_error == Fraction(1, 496),
        "true_promoted_error_is_0": true_promoted_error == Fraction(0),
        "interval_1_contains_y": interval_1_contains_y,
        "interval_2_contains_y": interval_2_contains_y,
        "intersection_contains_y": intersection_contains_y,
        "intersection_contained_in_interval_1": intersection_contained_in_interval_1,
        "intersection_nonincreasing_width": intersection_nonincreasing_width,
        "response_G_plus_is_17_16": S_plus == Fraction(17, 16),
        "response_G_minus_is_16_15": S_minus == Fraction(16, 15),
        "completions_gap_is_1_240": completions_gap == Fraction(1, 240),
        "completions_gap_exceeds_2_epsilon0": completions_gap > Fraction(2, 4096),
    }

    return {
        "case_id": case_cfg["case_id"],
        "y_true": format_fraction(y),
        "K1_initial": {
            "x": [format_fraction(val) for val in cert1["x"]],
            "H": mat_to_json(cert1["H"]),
            "q": [format_fraction(val) for val in cert1["q"]],
            "k": format_fraction(cert1["k"]),
            "b": format_fraction(cert1["b"]),
            "k_plus_b": format_fraction(cert1["k_plus_b"]),
            "p": format_fraction(cert1["p"]),
            "interval": [format_fraction(L1), format_fraction(U1)],
            "width": format_fraction(cert1["width"]),
            "true_error": format_fraction(true_initial_error),
        },
        "K2_promoted": {
            "x": [format_fraction(val) for val in cert2["x"]],
            "H": mat_to_json(cert2["H"]),
            "q": [format_fraction(val) for val in cert2["q"]],
            "k": format_fraction(cert2["k"]),
            "b": format_fraction(cert2["b"]),
            "k_plus_b": format_fraction(cert2["k_plus_b"]),
            "p": format_fraction(cert2["p"]),
            "interval": [format_fraction(L2), format_fraction(U2)],
            "width": format_fraction(cert2["width"]),
            "true_error": format_fraction(true_promoted_error),
        },
        "intersection_interval": [format_fraction(L_int), format_fraction(U_int)],
        "intersection_width": format_fraction(width_intersection),
        "completions": {
            "S_G_plus": format_fraction(S_plus),
            "S_G_minus": format_fraction(S_minus),
            "gap": format_fraction(completions_gap),
        },
        "assertions": assertions,
        "all_passed": all(assertions.values()),
        "scientific_note": (
            "Finite witness demonstrating certificate non-monotonicity under port acquisition "
            "and repair via interval intersection; completions separated by exact gap 1/240 > 2*epsilon0 "
            "(epsilon0=1/4096). This finite witness is NOT proof of universal propositions."
        ),
    }


def run_all_cases(config_data):
    """Execute all fixtures in config_data and collect results and assertions."""
    # Smoke check: singular pivot refusal
    refusal_ok, refusal_msg = test_singular_pivot_refusal()
    if not refusal_ok:
        raise RuntimeError(f"Singular-pivot refusal sanity check failed: {refusal_msg}")

    results = []
    cases = config_data.get("cases", [])
    if len(cases) > 12:
        raise ValueError(f"Exceeded max 12 tiny cases bound: got {len(cases)}")

    for case in cases:
        f_num = case.get("fixture_num")
        if f_num == 1:
            res = execute_fixture_1(case)
        elif f_num == 2:
            res = execute_fixture_2(case)
        elif f_num == 3:
            res = execute_fixture_3(case)
        elif f_num == 4:
            res = execute_fixture_4(case)
        elif f_num == 5:
            res = execute_fixture_5(case)
        elif f_num == 6:
            res = execute_fixture_6(case)
        elif f_num == 7:
            res = execute_fixture_7(case)
        else:
            raise ValueError(f"Unknown fixture number: {f_num} in case {case.get('case_id')}")
        results.append(res)

    all_passed = all(r["all_passed"] for r in results)

    summary = {
        "status": "PASS" if all_passed else "FAIL",
        "total_cases": len(results),
        "singular_pivot_refusal_verified": refusal_ok,
        "tolerance_bounds_used": ["1/7168", "1/4096"],
        "cases_executed": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Exact-Arithmetic Correction Capsule Runner (Pure Python fractions.Fraction)"
    )
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "correction_capsule_exact.json")
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help=f"Path to JSON config (default: {default_config})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSON execution facts/assertions",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    summary = run_all_cases(config_data)

    json_str = json.dumps(summary, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str + "\n")
        print(f"Results written to {args.output}")

    print(json_str)

    if summary["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
