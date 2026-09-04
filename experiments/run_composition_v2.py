"""Gate 2 v2: Filler-based noncommutativity test.

Codex-designed 8-arm experiment that absorbs any additive position/recency
model algebraically. Token-length-matched neutral fillers replace same-role
controls. Reuses ModelAdapter from run_svb_0.
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

from run_svb_0 import ModelAdapter, build_prefix, build_query


def expand_surface(surface, var):
    return surface.replace("{var}", var)


def run_filler_composition(adapter, cfg):
    fc = cfg["filler_composition"]
    surface_map = {
        "A": fc["surface_A"],
        "M": fc["surface_M"],
        "F_A": fc["filler_A"],
        "F_M": fc["filler_M"],
    }
    arms = fc["arms"]
    observations = {}
    call_count_start = adapter.call_count

    for depth in cfg["depths"]:
        tmpl_key = f"depth{depth}_single"
        tmpl = cfg["templates"][tmpl_key]
        t0 = time.time()

        for var in cfg["variables"]:
            for val in cfg["outer_values"]:
                prefix = build_prefix(tmpl, var=var, outer_val=val)
                state = adapter.get_state_after_prefix(prefix)
                query = build_query(var, cfg)

                suffixes = []
                for arm in arms:
                    s1_key = arm["suffix_1"]
                    s2_key = arm["suffix_2"]
                    s1 = expand_surface(surface_map[s1_key], var)
                    s2 = expand_surface(surface_map[s2_key], var)
                    suffixes.append((arm["name"], s1 + s2 + query))

                for i, (arm_name, suffix_str) in enumerate(suffixes):
                    last = (i == len(suffixes) - 1)
                    dist = adapter.get_dist_from_state(
                        state, suffix_str, deepcopy=not last)
                    key = f"d{depth}_{var}_{val}_{arm_name}"
                    observations[key] = {
                        "depth": depth, "var": var, "val": val,
                        "arm": arm_name, "dist": dist.tolist(),
                    }

                del state
            gc.collect()

        calls = adapter.call_count - call_count_start
        print(f"  d{depth}: {time.time()-t0:.1f}s ({calls} calls total)", flush=True)

    return observations


def analyze_filler_composition(observations, cfg):
    """Codex filler-based noncommutativity analysis.

    Filler null model in additive log-ratio coordinates:
      predicted_phi(AM) = phi(AF_M) + phi(F_AM) - phi(F_AF_M)
      predicted_phi(MA) = phi(MF_A) + phi(F_MA) - phi(F_MF_A)

    Two pre-registered gates:
    1. Direct order effect: TV(AM, MA) > eps_eq
    2. Excess interaction: order contrast beyond filler null > eps_eq
    """
    print("\n=== GATE 2 v2: FILLER-BASED NONCOMMUTATIVITY ===\n")

    eps_eq = cfg.get("eps_eq", 0.01)

    arm_data = {}
    for key, obs in observations.items():
        ctx = (obs["depth"], obs["var"], obs["val"])
        if obs["val"] == 9:
            continue
        arm_data.setdefault(obs["arm"], {})[ctx] = np.array(obs["dist"])

    contexts = sorted(arm_data.get("AM", {}).keys())
    n = len(contexts)
    print(f"  Contexts: {n}")

    direct_tvs = []
    excess_am = []
    excess_ma = []

    print(f"\n  {'Context':<20} {'TV(AM,MA)':>10} {'excess_AM':>10} {'excess_MA':>10}")

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

        predicted_am = d_afm + d_fam - d_fafm
        predicted_am = np.maximum(predicted_am, 0)
        if predicted_am.sum() > 0:
            predicted_am /= predicted_am.sum()

        predicted_ma = d_mfa + d_fma - d_fmfa
        predicted_ma = np.maximum(predicted_ma, 0)
        if predicted_ma.sum() > 0:
            predicted_ma /= predicted_ma.sum()

        e_am = 0.5 * np.sum(np.abs(d_am - predicted_am))
        e_ma = 0.5 * np.sum(np.abs(d_ma - predicted_ma))
        excess_am.append(e_am)
        excess_ma.append(e_ma)

        label = f"d{ctx[0]}_{ctx[1]}_{ctx[2]}"
        print(f"  {label:<20} {tv_direct:10.4f} {e_am:10.4f} {e_ma:10.4f}")

    direct_arr = np.array(direct_tvs)
    excess_am_arr = np.array(excess_am)
    excess_ma_arr = np.array(excess_ma)
    excess_mean = np.mean([excess_am_arr.mean(), excess_ma_arr.mean()])

    print(f"\n  === SUMMARY ===")
    print(f"  Direct TV(AM,MA):   mean={direct_arr.mean():.4f}, "
          f"median={np.median(direct_arr):.4f}, max={direct_arr.max():.4f}")
    print(f"  Excess AM:          mean={excess_am_arr.mean():.4f}, "
          f"median={np.median(excess_am_arr):.4f}")
    print(f"  Excess MA:          mean={excess_ma_arr.mean():.4f}, "
          f"median={np.median(excess_ma_arr):.4f}")
    print(f"  Mean excess:        {excess_mean:.4f}")

    print(f"\n  === GATE RESULTS (eps_eq={eps_eq}) ===")

    gate1 = bool(np.median(direct_arr) > eps_eq)
    gate2 = bool(excess_mean > eps_eq)

    print(f"  Gate 1 (direct order): median TV = {np.median(direct_arr):.4f} "
          f"{'> ' if gate1 else '<='} {eps_eq} -> {'PASS' if gate1 else 'FAIL'}")
    print(f"  Gate 2 (excess):       mean excess = {excess_mean:.4f} "
          f"{'> ' if gate2 else '<='} {eps_eq} -> {'PASS' if gate2 else 'FAIL'}")

    if gate1 and gate2:
        verdict = "NONCOMMUTATIVE_CONTROLLED"
        print(f"\n  -> {verdict}")
        print(f"     Both gates pass. Genuine interaction beyond additive position effects.")
        print(f"     Rejects K_a, logit-bias, and position-weighted additive decay.")
    elif gate1 and not gate2:
        verdict = "POSITION_EXPLAINED"
        print(f"\n  -> {verdict}")
        print(f"     Direct order effect exists but filler null absorbs it.")
        print(f"     Ordinary recency/position explains the order effect.")
    elif not gate1:
        verdict = "COMMUTATIVE"
        print(f"\n  -> {verdict}")
        print(f"     Selected generators commute within tolerance.")
    else:
        verdict = "INCONCLUSIVE"

    robust_gamma = direct_arr - excess_am_arr - excess_ma_arr
    print(f"\n  Robust commutator margin (Gamma):")
    print(f"    mean={robust_gamma.mean():.4f}, median={np.median(robust_gamma):.4f}")
    print(f"    positive in {np.sum(robust_gamma > 0)}/{n} contexts")

    return {
        "direct_tv_mean": float(direct_arr.mean()),
        "direct_tv_median": float(np.median(direct_arr)),
        "direct_tv_max": float(direct_arr.max()),
        "excess_am_mean": float(excess_am_arr.mean()),
        "excess_ma_mean": float(excess_ma_arr.mean()),
        "excess_mean": float(excess_mean),
        "gate1_pass": gate1,
        "gate2_pass": gate2,
        "verdict": verdict,
        "robust_gamma_mean": float(robust_gamma.mean()),
        "robust_gamma_positive_frac": float(np.sum(robust_gamma > 0) / n),
        "eps_eq": eps_eq,
    }


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else \
        "experiments/config/svb_qwen3_composition_v2.json"
    with open(config_path) as f:
        cfg = json.load(f)

    result_dir = Path(cfg["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    print("Gate 2 v2: Filler-Based Noncommutativity Test", flush=True)
    print(f"Config: {config_path}", flush=True)
    t_start = time.time()

    adapter = ModelAdapter(cfg)
    print(f"Model loaded. Digit tokens: {adapter.digit_token_ids}", flush=True)

    observations = run_filler_composition(adapter, cfg)
    summary = analyze_filler_composition(observations, cfg)

    elapsed = time.time() - t_start
    print(f"\nTotal: {adapter.call_count} calls, {elapsed:.1f}s ({elapsed/60:.1f} min)",
          flush=True)

    result_file = result_dir / "result.json"
    with open(result_file, "w") as f:
        json.dump({
            "config": cfg,
            "summary": summary,
            "observations": observations,
            "calls": adapter.call_count,
            "elapsed_s": elapsed,
        }, f, indent=2)
    print(f"Saved to {result_file}", flush=True)


if __name__ == "__main__":
    main()
