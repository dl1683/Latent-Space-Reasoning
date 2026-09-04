"""Operator algebra v2: position-specific operators on (C,L,R).

The 8-arm design gives us 4 position-specific operators:
  K_A^1 from AF_M (A in pos1, filler in pos2)
  K_M^2 from F_AM (filler in pos1, M in pos2)
  K_M^1 from MF_A (M in pos1, filler in pos2)
  K_A^2 from F_MA (filler in pos1, A in pos2)

Compositional predictions:
  AM = K_M^2 o K_A^1 (A first, M second)
  MA = K_A^2 o K_M^1 (M first, A second)

The genuine interaction = TV(observed, position-correct prediction).
If this is small, operators compose linearly. If large, there is
genuine state-dependent interaction.
"""
import json
import sys
import numpy as np


def dist_to_clr(dist, correct_digit):
    if correct_digit == 9:
        return None
    C = dist[correct_digit]
    L = dist[9]
    R = 1.0 - C - L
    return np.array([C, L, R])


def fit_operator(baseline_clrs, operated_clrs):
    n = len(baseline_clrs)
    B = np.column_stack(baseline_clrs)
    O = np.column_stack(operated_clrs)
    K, _, _, _ = np.linalg.lstsq(B.T, O.T, rcond=None)
    K = K.T
    K = np.maximum(K, 0)
    for j in range(3):
        s = K[:, j].sum()
        if s > 0:
            K[:, j] /= s
    return K


def clr_tv(a, b):
    return 0.5 * np.sum(np.abs(a - b))


def print_matrix(name, K):
    print(f"  {name}:")
    print(f"         C->     L->     R->")
    for i, label in enumerate(["C", "L", "R"]):
        print(f"    {label}: [{K[i,0]:.4f}  {K[i,1]:.4f}  {K[i,2]:.4f}]")


def main():
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
    print(f"=== POSITION-SPECIFIC OPERATOR ALGEBRA ===\n")
    print(f"Contexts: {n}\n")

    baseline = [arm_clr["F_AF_M"][c] for c in contexts]
    af_m = [arm_clr["AF_M"][c] for c in contexts]
    f_am = [arm_clr["F_AM"][c] for c in contexts]
    mf_a = [arm_clr["MF_A"][c] for c in contexts]
    f_ma = [arm_clr["F_MA"][c] for c in contexts]
    am = [arm_clr["AM"][c] for c in contexts]
    ma = [arm_clr["MA"][c] for c in contexts]

    K_A1 = fit_operator(baseline, af_m)
    K_M2 = fit_operator(baseline, f_am)
    K_M1 = fit_operator(baseline, mf_a)
    K_A2 = fit_operator(baseline, f_ma)

    print("--- POSITION-SPECIFIC OPERATORS ---\n")
    print_matrix("K_A^1 (A in pos 1)", K_A1)
    print_matrix("K_A^2 (A in pos 2)", K_A2)
    print_matrix("K_M^1 (M in pos 1)", K_M1)
    print_matrix("K_M^2 (M in pos 2)", K_M2)

    print(f"\n--- POSITION DEPENDENCE ---\n")
    tv_A_pos = 0.5 * np.sum(np.abs(K_A1 - K_A2))
    tv_M_pos = 0.5 * np.sum(np.abs(K_M1 - K_M2))
    print(f"  TV(K_A^1, K_A^2) = {tv_A_pos:.4f}  (A's position dependence)")
    print(f"  TV(K_M^1, K_M^2) = {tv_M_pos:.4f}  (M's position dependence)")

    print(f"\n--- COMPOSITIONAL PREDICTIONS ---\n")

    K_AM_pred = K_M2 @ K_A1
    K_MA_pred = K_A2 @ K_M1

    print_matrix("Predicted AM = K_M^2 @ K_A^1", K_AM_pred)
    print_matrix("Predicted MA = K_A^2 @ K_M^1", K_MA_pred)

    am_tvs = []
    ma_tvs = []
    am_direct = []

    print(f"\n  {'Context':<15} {'TV(AM,pred)':>11} {'TV(MA,pred)':>11} {'TV(AM,MA)':>10}")

    for i, ctx in enumerate(contexts):
        b = baseline[i]
        pred_am = K_AM_pred @ b
        pred_am = np.maximum(pred_am, 0)
        if pred_am.sum() > 0:
            pred_am /= pred_am.sum()

        pred_ma = K_MA_pred @ b
        pred_ma = np.maximum(pred_ma, 0)
        if pred_ma.sum() > 0:
            pred_ma /= pred_ma.sum()

        tv_am = clr_tv(am[i], pred_am)
        tv_ma = clr_tv(ma[i], pred_ma)
        tv_dir = clr_tv(am[i], ma[i])

        am_tvs.append(tv_am)
        ma_tvs.append(tv_ma)
        am_direct.append(tv_dir)

        label = f"d{ctx[0]}_{ctx[1]}_{ctx[2]}"
        print(f"  {label:<13} {tv_am:11.4f} {tv_ma:11.4f} {tv_dir:10.4f}")

    am_arr = np.array(am_tvs)
    ma_arr = np.array(ma_tvs)
    both = np.concatenate([am_arr, ma_arr])

    print(f"\n  Summary:")
    print(f"    TV(AM, pred AM): mean={am_arr.mean():.4f}, median={np.median(am_arr):.4f}")
    print(f"    TV(MA, pred MA): mean={ma_arr.mean():.4f}, median={np.median(ma_arr):.4f}")
    print(f"    Combined:        mean={both.mean():.4f}")
    print(f"    TV(AM, MA) direct: mean={np.mean(am_direct):.4f}")

    # Genuine interaction = residual after position-correct prediction
    print(f"\n--- GENUINE INTERACTION (residual after position-correct composition) ---\n")
    if both.mean() < 0.02:
        print(f"  -> STRONG COMPOSITIONALITY: position-specific operators compose well")
        print(f"     Operators are approximately constant (state-independent)")
    elif both.mean() < 0.05:
        print(f"  -> MODERATE COMPOSITIONALITY: main structure captured")
        print(f"     Some state-dependent residual interaction")
    else:
        print(f"  -> WEAK COMPOSITIONALITY: operators are genuinely state-dependent")
        print(f"     The output of one operator changes how the next one acts")

    # Check the commutator at the matrix level
    print(f"\n--- MATRIX COMMUTATOR (position-corrected) ---\n")
    comm_pred_tv = 0.5 * np.sum(np.abs(K_AM_pred - K_MA_pred))
    print(f"  TV(predicted AM matrix, predicted MA matrix) = {comm_pred_tv:.4f}")
    print(f"  This is the STRUCTURAL noncommutativity (position-corrected)")

    # Per-variable
    print(f"\n--- PER-VARIABLE ---\n")
    for var in ["x", "y", "z"]:
        idxs = [i for i, ctx in enumerate(contexts) if ctx[1] == var]
        am_v = am_arr[idxs]
        ma_v = ma_arr[idxs]
        dir_v = np.array(am_direct)[idxs]
        print(f"  {var}: pred_AM_err={am_v.mean():.4f}, pred_MA_err={ma_v.mean():.4f}, "
              f"direct={dir_v.mean():.4f}")


if __name__ == "__main__":
    main()
