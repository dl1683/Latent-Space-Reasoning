"""Re-analyze Gate 2 v2 results with both probability-space and log-ratio predictions.

Codex specified additivity in log-ratio coordinates. The original runner used
probability-space additivity. This checks whether the coordinate choice matters.
"""
import json
import sys
import numpy as np


def analyze(observations, eps_eq=0.01):
    arm_data = {}
    for key, obs in observations.items():
        ctx = (obs["depth"], obs["var"], obs["val"])
        if obs["val"] == 9:
            continue
        arm_data.setdefault(obs["arm"], {})[ctx] = np.array(obs["dist"])

    contexts = sorted(arm_data.get("AM", {}).keys())
    n = len(contexts)

    print(f"  Contexts: {n}\n")
    print(f"  {'Context':<15} {'TV(AM,MA)':>9} {'exc_prob':>9} {'exc_logr':>9}")

    direct_tvs = []
    exc_prob_list = []
    exc_logr_list = []

    for ctx in contexts:
        d_am = arm_data["AM"][ctx]
        d_ma = arm_data["MA"][ctx]
        d_afm = arm_data["AF_M"][ctx]
        d_fma = arm_data["F_MA"][ctx]
        d_mfa = arm_data["MF_A"][ctx]
        d_fam = arm_data["F_AM"][ctx]
        d_fafm = arm_data["F_AF_M"][ctx]
        d_fmfa = arm_data["F_MF_A"][ctx]

        tv_direct = 0.5 * np.sum(np.abs(d_am - d_ma))
        direct_tvs.append(tv_direct)

        # Probability-space additive prediction
        pred_am_p = d_afm + d_fam - d_fafm
        pred_am_p = np.maximum(pred_am_p, 0)
        if pred_am_p.sum() > 0:
            pred_am_p /= pred_am_p.sum()

        pred_ma_p = d_mfa + d_fma - d_fmfa
        pred_ma_p = np.maximum(pred_ma_p, 0)
        if pred_ma_p.sum() > 0:
            pred_ma_p /= pred_ma_p.sum()

        exc_p = 0.5 * (
            0.5 * np.sum(np.abs(d_am - pred_am_p)) +
            0.5 * np.sum(np.abs(d_ma - pred_ma_p))
        )
        exc_prob_list.append(exc_p)

        # Log-ratio (multiplicative) prediction
        # p_AM_i proportional to p_AFM_i * p_FAM_i / p_FAFM_i
        floor = 1e-10
        pred_am_lr = (d_afm + floor) * (d_fam + floor) / (d_fafm + floor)
        pred_am_lr /= pred_am_lr.sum()
        pred_ma_lr = (d_mfa + floor) * (d_fma + floor) / (d_fmfa + floor)
        pred_ma_lr /= pred_ma_lr.sum()

        exc_lr = 0.5 * (
            0.5 * np.sum(np.abs(d_am - pred_am_lr)) +
            0.5 * np.sum(np.abs(d_ma - pred_ma_lr))
        )
        exc_logr_list.append(exc_lr)

        label = f"d{ctx[0]}_{ctx[1]}_{ctx[2]}"
        print(f"  {label:<15} {tv_direct:9.4f} {exc_p:9.4f} {exc_lr:9.4f}")

    direct_arr = np.array(direct_tvs)
    exc_prob_arr = np.array(exc_prob_list)
    exc_logr_arr = np.array(exc_logr_list)

    print(f"\n  === SUMMARY ===")
    print(f"  Direct TV(AM,MA): mean={direct_arr.mean():.4f}, median={np.median(direct_arr):.4f}")
    print(f"  Excess (prob):    mean={exc_prob_arr.mean():.4f}, median={np.median(exc_prob_arr):.4f}")
    print(f"  Excess (log-ratio): mean={exc_logr_arr.mean():.4f}, median={np.median(exc_logr_arr):.4f}")

    print(f"\n  === GATE 2 (eps_eq={eps_eq}) ===")
    for label, arr in [("prob-space", exc_prob_arr), ("log-ratio", exc_logr_arr)]:
        m = arr.mean()
        passes = m > eps_eq
        print(f"  {label:<12}: mean excess = {m:.4f} {'> ' if passes else '<='} {eps_eq} "
              f"-> {'PASS' if passes else 'FAIL'}")

    # Variable-level breakdown
    print(f"\n  === PER-VARIABLE (log-ratio) ===")
    for var in ["x", "y", "z"]:
        var_idxs = [i for i, ctx in enumerate(contexts) if ctx[1] == var]
        if var_idxs:
            var_direct = direct_arr[var_idxs]
            var_exc = exc_logr_arr[var_idxs]
            print(f"  {var}: direct={var_direct.mean():.4f}, excess={var_exc.mean():.4f}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "experiments/results/svb_qwen3_composition_v2/result.json"
    with open(path) as f:
        data = json.load(f)

    print("=== GATE 2 v2: COORDINATE ROBUSTNESS CHECK ===\n")
    analyze(data["observations"], data["config"].get("eps_eq", 0.01))


if __name__ == "__main__":
    main()
