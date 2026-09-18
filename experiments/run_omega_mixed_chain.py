"""Omega Mixed Continuation Chains (Experiment #54).

Exp #47 tested same-continuation chains (N→NN→NNN, I→II→III).
This tests MIXED chains: what happens when we combine different
continuation types in sequence?

Key questions:
- Does safety_donot + heavy amplify more than heavy alone?
- Does identity + heavy cancel or compose?
- Does deprecation suppress heavy's amplification?

Tests all 2-step combinations of {identity, heavy, neutral, safety, depr}.
~300 readouts (25 combos x 2 surfaces x 6 cells).
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv


CFG = {
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
}

EQUIV_PAIRS = [
    ("assert_novar_2", "assert_var_2", "strong_assert_2"),
]

BASE_CONTS = {
    "I": "{var} = {var}\n",
    "H": "{var} = {var} + 1\n",
    "N": "pass\n",
    "S": "# SAFETY: do not modify\n{var} = {var}\n",
    "D": "# DEPRECATED: will be removed\n{var} = {var}\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]


def run_mixed_chain():
    print("=== Omega Mixed Chains (Exp #54) ===\n")

    adapter = ModelAdapter(CFG)
    query_tmpl = CFG["query_template"]
    observations = []
    t0 = time.time()
    n = 0

    chains = {}
    for a_name, a_tmpl in BASE_CONTS.items():
        chains[a_name] = a_tmpl
        for b_name, b_tmpl in BASE_CONTS.items():
            chains[f"{a_name}{b_name}"] = a_tmpl + "{sep}" + b_tmpl

    for depth, var, val in CELLS:
        tmpl = CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_name, surf_tmpl in SURFACES.items():
            surface_text = surf_tmpl.replace("{var}", var)

            for chain_name, chain_tmpl in chains.items():
                chain_text = chain_tmpl.replace("{var}", var).replace(
                    "{sep}", "")
                full_suffix = surface_text + chain_text + query
                dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                n += 1

                observations.append({
                    "depth": depth,
                    "var": var,
                    "val": val,
                    "surface": surf_name,
                    "chain": chain_name,
                    "dist": dist.tolist(),
                })

                if n % 100 == 0:
                    print(f"  {n} readouts, {time.time()-t0:.0f}s",
                          flush=True)

    elapsed = time.time() - t0
    print(f"Done: {n} readouts in {elapsed:.1f}s")

    result_dir = Path("experiments/results/omega_mixed_chain")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    # Analysis
    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["chain"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    summary = {"n_readouts": n, "elapsed_s": elapsed, "chains": {}}

    print("\n=== Omega by Chain ===")
    for tau in [0.01, 0.05]:
        eligible = set()
        for depth, var, val in CELLS:
            ka = (depth, var, val, "assert_novar_2", "I")
            kb = (depth, var, val, "assert_var_2", "I")
            if ka in obs_idx and kb in obs_idx:
                d = tv(obs_idx[ka], obs_idx[kb])
                if d <= tau:
                    eligible.add((depth, var, val))

        if not eligible:
            continue

        print(f"\n  tau={tau} ({len(eligible)} eligible):")
        chain_omega = {}
        for chain_name in chains:
            tvs = []
            for cell in eligible:
                d, v, vl = cell
                ka = (d, v, vl, "assert_novar_2", chain_name)
                kb = (d, v, vl, "assert_var_2", chain_name)
                if ka in obs_idx and kb in obs_idx:
                    tvs.append(tv(obs_idx[ka], obs_idx[kb]))
            if tvs:
                chain_omega[chain_name] = {
                    "omega": float(np.max(tvs)),
                    "mean": float(np.mean(tvs)),
                }

        summary["chains"][f"tau{tau}"] = chain_omega

        # Sort by omega
        for cn, cd in sorted(chain_omega.items(),
                             key=lambda x: x[1]["omega"], reverse=True):
            print(f"    {cn:6s}: omega={cd['omega']:.5f} "
                  f"mean={cd['mean']:.5f}")

    # Interaction analysis: is AB > max(A, B)?
    print("\n=== Interaction Effects (tau=0.05) ===")
    print("  Chain  | omega | A_alone | B_alone | superadditive?")
    tau_key = "tau0.05"
    if tau_key in summary["chains"]:
        co = summary["chains"][tau_key]
        for a_name in BASE_CONTS:
            for b_name in BASE_CONTS:
                ab_name = f"{a_name}{b_name}"
                if ab_name in co and a_name in co and b_name in co:
                    ab_omega = co[ab_name]["omega"]
                    a_omega = co[a_name]["omega"]
                    b_omega = co[b_name]["omega"]
                    max_single = max(a_omega, b_omega)
                    superadd = ab_omega > max_single * 1.1
                    flag = "YES" if superadd else "no"
                    print(f"  {ab_name:6s} | {ab_omega:.4f} | "
                          f"{a_omega:.4f} | {b_omega:.4f} | {flag}")

    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir}/")
    return summary


if __name__ == "__main__":
    run_mixed_chain()
