"""Omega Lyapunov Exponent Estimation (Experiment #68).

Frames autoregressive bifurcation as a dynamical systems phenomenon.
For each (pair, cell, continuation), computes the LOCAL Lyapunov exponent:
the exponential rate of growth or decay of TV in the first N tokens.

λ = (1/N) * ln(TV_N / TV_0)

If λ > 0: diverging (positive Lyapunov exponent)
If λ < 0: converging (negative exponent)
If λ ≈ 0: marginal stability

The bimodality hypothesis predicts a BIMODAL distribution of λ with
a gap around zero — trajectories are either strongly positive or
strongly negative, with few near zero.

Additionally tests:
- Is the distribution of λ significantly bimodal (Hartigan's dip test)?
- Does continuation type predict the sign of λ?
- Does the exponent stabilize or drift over generation length?
- What is the relationship between λ and margin/entropy?

~54 conditions (3 pairs × 6 cells × 3 continuations).
Short trajectories (15 tokens) to focus on the growth rate.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
import copy
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv

import torch

MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B-Base",
    "query_template": "f()\nprint({var})  # Output: ",
    "templates": {
        "depth4_single": (
            "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
            "    def g():\n        {var} = 999\n        def h():\n"
            "            {var} = 9999\n            def k():\n"
            "                {var} = 99999\n                return {var}\n"
            "            k()\n            return {var}\n"
            "        h()\n        return {var}\n"
            "    g()\n    return {var}\n"
        ),
    },
}

SURFACES = {
    "assert_novar_2": "# State unchanged.\n",
    "assert_var_2": "# {var} stays the same.\n",
    "boundary_1": "\n",
    "boundary_2": "pass\n",
    "minimal_ok": "# OK.\n",
    "minimal_good": "# Good.\n",
}

EQUIV_PAIRS = [
    ("assert_var_2", "assert_novar_2", "strong_assert_2"),
    ("boundary_1", "boundary_2", "boundary_pair"),
    ("minimal_ok", "minimal_good", "minimal_ok_good"),
]

CONTINUATIONS = {
    "empty": "",
    "identity": "{var} = {var}\n",
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

N_GENERATE = 15
TAU = 0.05
EPS = 1e-10


def generate_trajectory(adapter, prefix_state, suffix_text, n_tokens):
    ids = adapter.tok.encode(suffix_text, add_special_tokens=False,
                             return_tensors="pt")
    st = copy.deepcopy(prefix_state)

    with torch.no_grad():
        out = adapter._forward_with_cache(ids, st)
    st = adapter._get_cache(out)

    trajectory = []
    logits = out.logits[0, -1, :]
    dist = adapter._extract_11bin(logits)
    trajectory.append(dist)

    for step in range(1, n_tokens + 1):
        next_token_id = int(torch.argmax(logits))
        next_ids = torch.tensor([[next_token_id]])
        with torch.no_grad():
            out = adapter._forward_with_cache(next_ids, st)
        st = adapter._get_cache(out)
        logits = out.logits[0, -1, :]
        dist = adapter._extract_11bin(logits)
        trajectory.append(dist)

    adapter.call_count += 1
    return trajectory


def compute_lyapunov(tvs, window=None):
    """Estimate local Lyapunov exponent from TV trajectory."""
    if window is None:
        window = len(tvs) - 1

    tv0 = max(tvs[0], EPS)
    tvn = max(tvs[min(window, len(tvs) - 1)], EPS)

    lam = np.log(tvn / tv0) / window
    return float(lam)


def compute_windowed_lyapunov(tvs, window_size=5):
    """Compute Lyapunov exponents over sliding windows."""
    exponents = []
    for i in range(len(tvs) - window_size):
        tv_start = max(tvs[i], EPS)
        tv_end = max(tvs[i + window_size], EPS)
        lam = np.log(tv_end / tv_start) / window_size
        exponents.append(float(lam))
    return exponents


def hartigans_dip_statistic(data):
    """Simple bimodality test: ratio of inter-mode gap to range."""
    if len(data) < 10:
        return 0.0, False
    sorted_d = np.sort(data)
    n = len(sorted_d)
    mid = n // 2
    gap = sorted_d[mid] - sorted_d[mid - 1]
    rng = sorted_d[-1] - sorted_d[0]
    if rng < EPS:
        return 0.0, False
    dip = gap / rng
    return float(dip), dip > 0.1


def run():
    print("=== Lyapunov Exponent Estimation ===")
    n_cond = len(CELLS) * len(EQUIV_PAIRS) * len(CONTINUATIONS)
    print(f"Conditions: {n_cond}, Tokens: {N_GENERATE}")

    adapter = ModelAdapter(MODEL_CFG)
    query_tmpl = MODEL_CFG["query_template"]
    t0 = time.time()
    records = []

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)
        cell_id = f"d{depth}_{var}_{val}"

        for surf_a, surf_b, pair_name in EQUIV_PAIRS:
            sa = SURFACES[surf_a].replace("{var}", var)
            sb = SURFACES[surf_b].replace("{var}", var)

            dist_a_base = adapter.get_dist_from_state(prefix_state, sa + query)
            dist_b_base = adapter.get_dist_from_state(prefix_state, sb + query)
            base_tv = float(tv(dist_a_base, dist_b_base))
            eligible = base_tv <= TAU

            margin_a = float(np.sort(dist_a_base)[-1] - np.sort(dist_a_base)[-2])
            entropy_a = float(-np.sum(dist_a_base[dist_a_base > 0] *
                                      np.log2(dist_a_base[dist_a_base > 0])))

            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)
                suffix_a = sa + cont_text + query
                suffix_b = sb + cont_text + query

                traj_a = generate_trajectory(
                    adapter, prefix_state, suffix_a, N_GENERATE)
                traj_b = generate_trajectory(
                    adapter, prefix_state, suffix_b, N_GENERATE)

                step_tvs = [float(tv(traj_a[s], traj_b[s]))
                            for s in range(N_GENERATE + 1)]

                lam_full = compute_lyapunov(step_tvs)
                lam_5 = compute_lyapunov(step_tvs, window=5)
                lam_10 = compute_lyapunov(step_tvs, window=10)
                windowed = compute_windowed_lyapunov(step_tvs, window_size=5)

                final_tv = step_tvs[-1]
                fate = "grow" if final_tv > 0.5 else "collapse"

                rec = {
                    "cell": cell_id,
                    "pair": pair_name,
                    "continuation": cont_name,
                    "eligible": eligible,
                    "base_tv": base_tv,
                    "margin_a": margin_a,
                    "entropy_a": entropy_a,
                    "initial_tv": step_tvs[0],
                    "final_tv": final_tv,
                    "fate": fate,
                    "lyapunov_full": lam_full,
                    "lyapunov_5": lam_5,
                    "lyapunov_10": lam_10,
                    "windowed_lyapunov": windowed,
                    "trajectory": step_tvs,
                }
                records.append(rec)
                tag = "[elig]" if eligible else "[INELIG]"
                print(f"  {cell_id}/{pair_name}/{cont_name}: "
                      f"lam={lam_full:+.3f} ({fate}) {tag}")

    elapsed = time.time() - t0

    elig = [r for r in records if r["eligible"]]
    lambdas = [r["lyapunov_full"] for r in elig]
    pos = [l for l in lambdas if l > 0]
    neg = [l for l in lambdas if l <= 0]

    dip, is_bimodal = hartigans_dip_statistic(lambdas)

    by_cont = defaultdict(list)
    for r in elig:
        by_cont[r["continuation"]].append(r["lyapunov_full"])

    cont_stats = {}
    for cont, lams in by_cont.items():
        cont_stats[cont] = {
            "mean": float(np.mean(lams)),
            "std": float(np.std(lams)),
            "n_pos": sum(1 for l in lams if l > 0),
            "n_neg": sum(1 for l in lams if l <= 0),
            "n": len(lams),
        }

    corr_margin = float(np.corrcoef(
        [r["margin_a"] for r in elig], lambdas)[0, 1]) if len(elig) > 2 else 0
    corr_entropy = float(np.corrcoef(
        [r["entropy_a"] for r in elig], lambdas)[0, 1]) if len(elig) > 2 else 0

    summary = {
        "n_total": len(records),
        "n_eligible": len(elig),
        "n_positive": len(pos),
        "n_negative": len(neg),
        "mean_lambda": float(np.mean(lambdas)) if lambdas else 0,
        "std_lambda": float(np.std(lambdas)) if lambdas else 0,
        "dip_statistic": dip,
        "bimodal": is_bimodal,
        "by_continuation": cont_stats,
        "corr_margin_lambda": corr_margin,
        "corr_entropy_lambda": corr_entropy,
        "elapsed_s": round(elapsed, 1),
    }

    outdir = Path("experiments/results/omega_lyapunov")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "rows.json", "w") as f:
        json.dump(records, f, indent=2)
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== SUMMARY ({elapsed:.0f}s) ===")
    print(f"Eligible: {len(elig)}/{len(records)}")
    print(f"Positive lam: {len(pos)}, Negative lam: {len(neg)}")
    print(f"Mean lam: {np.mean(lambdas):.3f} +/- {np.std(lambdas):.3f}")
    print(f"Dip statistic: {dip:.3f} (bimodal={is_bimodal})")
    print(f"Correlation with margin: {corr_margin:.3f}")
    print(f"Correlation with entropy: {corr_entropy:.3f}")
    for cont, stats in cont_stats.items():
        print(f"  {cont}: lam={stats['mean']:+.3f}+/-{stats['std']:.3f} "
              f"({stats['n_pos']}+/{stats['n_neg']}-)")
    sys.stdout.flush()


if __name__ == "__main__":
    run()
