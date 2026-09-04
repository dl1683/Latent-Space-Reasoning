"""Gate 2 v2 repair: multi-filler robustness test.

Reuses AM/MA and filler pair 1 from the original v2 experiment.
Collects filler pairs 2 and 3. Analyzes all with the correct
residual-order-contrast statistic.
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


def run_filler_arms(adapter, cfg, filler_A, filler_M, pair_name):
    """Run just the 6 filler arms for one filler pair."""
    surface_map = {
        "A": cfg["surfaces"]["surface_A"],
        "M": cfg["surfaces"]["surface_M"],
        "F_A": filler_A,
        "F_M": filler_M,
    }
    filler_arms = [a for a in cfg["arms"] if a["name"] not in ("AM", "MA")]
    observations = {}
    t0 = time.time()

    for depth in cfg["depths"]:
        tmpl_key = f"depth{depth}_single"
        tmpl = cfg["templates"][tmpl_key]

        for var in cfg["variables"]:
            for val in cfg["outer_values"]:
                prefix = build_prefix(tmpl, var=var, outer_val=val)
                state = adapter.get_state_after_prefix(prefix)
                query = build_query(var, cfg)

                for arm in filler_arms:
                    s1 = expand_surface(surface_map[arm["suffix_1"]], var)
                    s2 = expand_surface(surface_map[arm["suffix_2"]], var)
                    suffix_str = s1 + s2 + query
                    dist = adapter.get_dist_from_state(state, suffix_str, deepcopy=True)
                    key = f"d{depth}_{var}_{val}_{arm['name']}"
                    observations[key] = {
                        "depth": depth, "var": var, "val": val,
                        "arm": arm["name"], "dist": dist.tolist(),
                        "pair": pair_name,
                    }

                del state
            gc.collect()

    elapsed = time.time() - t0
    print(f"  {pair_name}: {len(observations)} obs, {elapsed:.1f}s", flush=True)
    return observations


def analyze_residual_order(am_ma_obs, filler_obs_list, pair_names, eps_eq=0.01):
    """Correct statistic: residual order contrast for each filler pair."""
    print("\n=== RESIDUAL ORDER CONTRAST (CORRECT STATISTIC) ===\n")

    am_data = {}
    ma_data = {}
    for key, o in am_ma_obs.items():
        ctx = (o["depth"], o["var"], o["val"])
        if o["val"] == 9:
            continue
        if o["arm"] == "AM":
            am_data[ctx] = np.array(o["dist"])
        elif o["arm"] == "MA":
            ma_data[ctx] = np.array(o["dist"])

    contexts = sorted(am_data.keys())
    n = len(contexts)
    print(f"  Contexts: {n}")

    all_residuals = {}
    all_gains = {}

    for pair_idx, (filler_obs, pair_name) in enumerate(zip(filler_obs_list, pair_names)):
        filler_data = {}
        for key, o in filler_obs.items():
            ctx = (o["depth"], o["var"], o["val"])
            if o["val"] == 9:
                continue
            filler_data.setdefault(o["arm"], {})[ctx] = np.array(o["dist"])

        residuals = []
        gain_residuals = []

        for ctx in contexts:
            d_am = am_data[ctx]
            d_ma = ma_data[ctx]
            d_afm = filler_data["AF_M"][ctx]
            d_fma = filler_data["F_MA"][ctx]
            d_mfa = filler_data["MF_A"][ctx]
            d_fam = filler_data["F_AM"][ctx]
            d_fafm = filler_data["F_AF_M"][ctx]
            d_fmfa = filler_data["F_MF_A"][ctx]

            floor = 1e-10
            pred_am = (d_afm + floor) * (d_fam + floor) / (d_fafm + floor)
            pred_am /= pred_am.sum()
            pred_ma = (d_mfa + floor) * (d_fma + floor) / (d_fmfa + floor)
            pred_ma /= pred_ma.sum()

            obs_order = d_am - d_ma
            pred_order = pred_am - pred_ma
            resid = obs_order - pred_order
            resid_tv = 0.5 * np.sum(np.abs(resid))
            residuals.append(resid_tv)

            if np.dot(pred_order, pred_order) > 0:
                g = np.dot(obs_order, pred_order) / np.dot(pred_order, pred_order)
                gain_resid = obs_order - g * pred_order
                gain_tv = 0.5 * np.sum(np.abs(gain_resid))
            else:
                gain_tv = 0.5 * np.sum(np.abs(obs_order))
            gain_residuals.append(gain_tv)

        all_residuals[pair_name] = np.array(residuals)
        all_gains[pair_name] = np.array(gain_residuals)

    print(f"\n  {'Pair':<8} {'ResidOrd_mean':>13} {'ResidOrd_med':>12} {'GainRes_mean':>13} {'>0.01':>6}")
    for pn in pair_names:
        r = all_residuals[pn]
        g = all_gains[pn]
        print(f"  {pn:<8} {r.mean():13.4f} {np.median(r):12.4f} {g.mean():13.4f} {np.sum(g>0.01):>3}/24")

    # Consistency: pairwise TV between residual vectors
    print(f"\n  --- CROSS-FILLER CONSISTENCY ---")
    for i in range(len(pair_names)):
        for j in range(i + 1, len(pair_names)):
            pi, pj = pair_names[i], pair_names[j]
            diff = np.abs(all_residuals[pi] - all_residuals[pj])
            corr = np.corrcoef(all_residuals[pi], all_residuals[pj])[0, 1]
            print(f"  {pi} vs {pj}: mean abs diff={diff.mean():.4f}, corr={corr:.4f}")

    # Gate: median residual > eps_eq, consistent across pairs
    medians = [np.median(all_residuals[pn]) for pn in pair_names]
    all_pass = all(m > eps_eq for m in medians)
    print(f"\n  --- GATE ---")
    print(f"  Medians: {[f'{m:.4f}' for m in medians]}")
    print(f"  All > {eps_eq}: {all_pass}")

    # Gain-fit gate: after scalar gain, does residual still pass?
    gain_medians = [np.median(all_gains[pn]) for pn in pair_names]
    gain_pass = all(m > eps_eq for m in gain_medians)
    print(f"  Gain-fit medians: {[f'{m:.4f}' for m in gain_medians]}")
    print(f"  Gain-fit all > {eps_eq}: {gain_pass}")

    if all_pass and not gain_pass:
        print(f"\n  -> POSITION_WITH_NONLINEAR_GAIN")
        print(f"     Residual order contrast exists but scalar gain absorbs it.")
    elif all_pass and gain_pass:
        print(f"\n  -> GENUINE_INTERACTION")
        print(f"     Residual survives scalar gain fit, consistent across fillers.")
    else:
        print(f"\n  -> FILLER_DEPENDENT or COMMUTATIVE")

    return {
        "residual_medians": {pn: float(np.median(all_residuals[pn])) for pn in pair_names},
        "gain_medians": {pn: float(np.median(all_gains[pn])) for pn in pair_names},
        "all_pass_residual": all_pass,
        "all_pass_gain": gain_pass,
    }


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else \
        "config/svb_qwen3_composition_v2_multifiller.json"
    with open(config_path) as f:
        cfg = json.load(f)

    result_dir = Path(cfg["result_dir"].replace("experiments/", "", 1))
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load existing AM/MA and pair 1 filler data
    with open("results/svb_qwen3_composition_v2/result.json") as f:
        v2_data = json.load(f)

    am_ma_obs = {}
    pair1_obs = {}
    for key, o in v2_data["observations"].items():
        if o["arm"] in ("AM", "MA"):
            am_ma_obs[key] = o
        else:
            pair1_obs[key] = o

    print(f"Loaded {len(am_ma_obs)} AM/MA obs and {len(pair1_obs)} pair1 filler obs",
          flush=True)

    # Run filler pairs 2 and 3
    t_start = time.time()
    adapter = ModelAdapter(cfg)
    print(f"Model loaded.", flush=True)

    pair2_obs = run_filler_arms(
        adapter, cfg,
        cfg["filler_pairs"][1]["filler_A"],
        cfg["filler_pairs"][1]["filler_M"],
        "pair2")

    pair3_obs = run_filler_arms(
        adapter, cfg,
        cfg["filler_pairs"][2]["filler_A"],
        cfg["filler_pairs"][2]["filler_M"],
        "pair3")

    elapsed = time.time() - t_start
    print(f"\nTotal: {adapter.call_count} calls, {elapsed:.1f}s ({elapsed/60:.1f} min)",
          flush=True)

    summary = analyze_residual_order(
        am_ma_obs,
        [pair1_obs, pair2_obs, pair3_obs],
        ["pair1", "pair2", "pair3"],
        cfg.get("eps_eq", 0.01))

    result_file = result_dir / "result.json"
    with open(result_file, "w") as f:
        json.dump({
            "config": cfg,
            "summary": summary,
            "am_ma_observations": am_ma_obs,
            "pair1_observations": pair1_obs,
            "pair2_observations": pair2_obs,
            "pair3_observations": pair3_obs,
            "calls": adapter.call_count,
            "elapsed_s": elapsed,
        }, f, indent=2)
    print(f"Saved to {result_file}", flush=True)


if __name__ == "__main__":
    main()
