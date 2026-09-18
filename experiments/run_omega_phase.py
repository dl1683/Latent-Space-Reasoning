"""Omega Phase Transition: Does omega show a sharp threshold or smooth gradient
as perturbation intensity varies continuously?

Two axes of continuous variation:
1. Repetition count: x = x repeated 1..10 times
2. Comment dilution: ratio of identity to neutral lines in a 4-line block

~360 readouts. Target: <10 min CPU.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix


TEMPLATE = (
    "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
    "    def g():\n        {var} = 999\n        def h():\n"
    "            {var} = 9999\n            def k():\n"
    "                {var} = 99999\n                return {var}\n"
    "            k()\n            return {var}\n"
    "        h()\n        return {var}\n"
    "    g()\n    return {var}\n"
)

QUERY = "f()\nprint({var})  # Output: "

SURFACE_PAIRS = [
    ("# OK.\n", "# Good.\n", "minimal_ok_good"),
]

CELLS = [
    (4, "x", 1), (4, "x", 5), (4, "x", 7),
    (4, "y", 2), (4, "y", 6), (4, "y", 8),
]


def make_repetition_continuations(var):
    """Identity statement repeated 1..10 times."""
    conts = {}
    for n in range(1, 11):
        conts[f"rep_{n}"] = (f"{var} = {var}\n") * n
    conts["rep_0"] = ""
    return conts


def make_dilution_continuations(var):
    """4-line blocks with varying identity-to-neutral ratio."""
    neutral = f"# State unchanged.\n"
    identity = f"{var} = {var}\n"
    conts = {
        "dil_0pct": neutral * 4,
        "dil_25pct": neutral * 3 + identity,
        "dil_50pct_early": identity + neutral + identity + neutral,
        "dil_50pct_late": neutral + identity + neutral + identity,
        "dil_75pct": identity * 3 + neutral,
        "dil_100pct": identity * 4,
    }
    return conts


def run_phase():
    print("=== Omega Phase Transition Experiment ===")
    result_dir = Path("experiments/results/omega_phase")
    result_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "model_id": "Qwen/Qwen3-1.7B",
        "query_template": QUERY,
        "templates": {"depth4_single": TEMPLATE},
    }
    adapter = ModelAdapter(cfg)

    observations = []
    t0 = time.time()
    n = 0

    for depth, var, val in CELLS:
        prefix_text = build_prefix(TEMPLATE, var=var, outer_val=val)
        query = QUERY.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        rep_conts = make_repetition_continuations(var)
        dil_conts = make_dilution_continuations(var)
        all_conts = {**rep_conts, **dil_conts}

        for pair_surf_a, pair_surf_b, pair_name in SURFACE_PAIRS:
            for cont_name, cont_text in all_conts.items():
                for side, surf_tmpl in [("a", pair_surf_a), ("b", pair_surf_b)]:
                    surf = surf_tmpl.replace("{var}", var)
                    full_suffix = surf + cont_text + query
                    dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                    n += 1
                    observations.append({
                        "depth": depth, "var": var, "val": val,
                        "pair": pair_name, "side": side,
                        "continuation": cont_name,
                        "dist": dist.tolist(),
                    })

    elapsed = time.time() - t0
    print(f"Done: {n} readouts in {elapsed:.1f}s ({n/elapsed:.1f}/s)")

    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    analyze_phase(observations)
    print(f"\nSaved {len(observations)} observations to {result_dir}/")
    return observations


def analyze_phase(all_obs):
    print("\n=== Phase Transition Analysis ===")

    obs_idx = {}
    for o in all_obs:
        key = (o["var"], o["val"], o["pair"], o["side"], o["continuation"])
        obs_idx[key] = np.array(o["dist"], dtype=np.float64)

    tau = 0.05

    print("\n--- Repetition axis (identity repeated 0..10 times) ---")
    print(f"{'Reps':<8s} {'eligible':>8s} {'mean_tv':>10s} {'max_tv':>10s} {'status':>10s}")

    for rep in range(0, 11):
        cont_name = f"rep_{rep}"
        tvs = []
        for _, var, val in CELLS:
            ka_base = (var, val, "minimal_ok_good", "a", "rep_0")
            kb_base = (var, val, "minimal_ok_good", "b", "rep_0")
            if ka_base not in obs_idx or kb_base not in obs_idx:
                continue
            d_pre = 0.5 * np.abs(obs_idx[ka_base] - obs_idx[kb_base]).sum()
            if d_pre > tau:
                continue
            ka = (var, val, "minimal_ok_good", "a", cont_name)
            kb = (var, val, "minimal_ok_good", "b", cont_name)
            if ka in obs_idx and kb in obs_idx:
                d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                tvs.append(d)

        if tvs:
            mean_tv = float(np.mean(tvs))
            max_tv = float(np.max(tvs))
            status = "PRESERVE" if max_tv <= tau else "AMPLIFY"
            print(f"{rep:<8d} {len(tvs):>8d} {mean_tv:10.5f} {max_tv:10.5f} {status:>10s}")

    print("\n--- Dilution axis (identity fraction in 4-line block) ---")
    print(f"{'Fraction':<16s} {'eligible':>8s} {'mean_tv':>10s} {'max_tv':>10s} {'status':>10s}")

    dil_names = ["dil_0pct", "dil_25pct", "dil_50pct_early", "dil_50pct_late", "dil_75pct", "dil_100pct"]
    dil_labels = ["0%", "25%", "50% (early)", "50% (late)", "75%", "100%"]

    for dil_name, dil_label in zip(dil_names, dil_labels):
        tvs = []
        for _, var, val in CELLS:
            ka_base = (var, val, "minimal_ok_good", "a", "rep_0")
            kb_base = (var, val, "minimal_ok_good", "b", "rep_0")
            if ka_base not in obs_idx or kb_base not in obs_idx:
                continue
            d_pre = 0.5 * np.abs(obs_idx[ka_base] - obs_idx[kb_base]).sum()
            if d_pre > tau:
                continue
            ka = (var, val, "minimal_ok_good", "a", dil_name)
            kb = (var, val, "minimal_ok_good", "b", dil_name)
            if ka in obs_idx and kb in obs_idx:
                d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                tvs.append(d)

        if tvs:
            mean_tv = float(np.mean(tvs))
            max_tv = float(np.max(tvs))
            status = "PRESERVE" if max_tv <= tau else "AMPLIFY"
            print(f"{dil_label:<16s} {len(tvs):>8d} {mean_tv:10.5f} {max_tv:10.5f} {status:>10s}")

    # Also print at tau=0.01 for comparison
    print("\n--- Repetition axis at tau=0.01 ---")
    print(f"{'Reps':<8s} {'eligible':>8s} {'mean_tv':>10s} {'max_tv':>10s} {'status':>10s}")

    for rep in range(0, 11):
        cont_name = f"rep_{rep}"
        tvs = []
        for _, var, val in CELLS:
            ka_base = (var, val, "minimal_ok_good", "a", "rep_0")
            kb_base = (var, val, "minimal_ok_good", "b", "rep_0")
            if ka_base not in obs_idx or kb_base not in obs_idx:
                continue
            d_pre = 0.5 * np.abs(obs_idx[ka_base] - obs_idx[kb_base]).sum()
            if d_pre > 0.01:
                continue
            ka = (var, val, "minimal_ok_good", "a", cont_name)
            kb = (var, val, "minimal_ok_good", "b", cont_name)
            if ka in obs_idx and kb in obs_idx:
                d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                tvs.append(d)

        if tvs:
            mean_tv = float(np.mean(tvs))
            max_tv = float(np.max(tvs))
            status = "PRESERVE" if max_tv <= 0.01 else "AMPLIFY"
            print(f"{rep:<8d} {len(tvs):>8d} {mean_tv:10.5f} {max_tv:10.5f} {status:>10s}")


if __name__ == "__main__":
    run_phase()
