"""Repair/Undo Experiment: Can neutral continuations undo heavy amplification?

The composition experiment showed bare_then_neutral (0.004) < bare (0.016),
suggesting neutral can repair. This experiment systematically tests:
- heavy → neutral (1-3 neutrals)
- identity → neutral (1-3 neutrals)
- heavy → pass (1-3 passes)
- heavy → neutral → identity (does identity re-amplify?)
- neutral → heavy → neutral (sandwich)

Tests whether omega is reversible or if some amplifications are permanent.

6 cells × 2 pairs × 2 sides × 18 chains = 432 readouts (~2 min)
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix


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

SURFACE_PAIRS = [
    ("# State unchanged.\n", "# {var} stays the same.\n", "strong_assert_2"),
    ("# Values preserved.\n", "# {var} retains its value.\n", "moderate_assert"),
]

N = "# State unchanged.\n"
H = "# Reassigning {var} now.\n"
I = "{var} = {var}\n"
P = "pass\n"
T = "type({var})\n"

CHAINS = {
    # Controls
    "none": "",
    "neutral_1": N,
    "heavy_1": H,
    "identity_1": I,
    "pass_1": P,

    # Heavy then repair
    "heavy_neutral_1": H + N,
    "heavy_neutral_2": H + N + N,
    "heavy_neutral_3": H + N + N + N,
    "heavy_pass_1": H + P,
    "heavy_pass_2": H + P + P,
    "heavy_pass_3": H + P + P + P,

    # Identity then repair
    "identity_neutral_1": I + N,
    "identity_neutral_2": I + N + N,
    "identity_neutral_3": I + N + N + N,

    # Re-amplification after repair
    "heavy_neutral_identity": H + N + I,
    "identity_neutral_heavy": I + N + H,

    # Sandwich patterns
    "neutral_heavy_neutral": N + H + N,
    "neutral_identity_neutral": N + I + N,
}

CELLS = [
    (4, "x", 1), (4, "x", 5), (4, "x", 7),
    (4, "y", 2), (4, "y", 6), (4, "y", 8),
]


def run_repair():
    print("=== Repair/Undo Experiment ===")
    n_total = len(CELLS) * len(SURFACE_PAIRS) * 2 * len(CHAINS)
    print(f"Cells: {len(CELLS)}, Pairs: {len(SURFACE_PAIRS)}, "
          f"Chains: {len(CHAINS)}")
    print(f"Total readouts: {n_total}")
    print()

    adapter = ModelAdapter(MODEL_CFG)
    query_tmpl = MODEL_CFG["query_template"]
    observations = []
    t0 = time.time()
    n = 0

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_a_tmpl, surf_b_tmpl, pair_name in SURFACE_PAIRS:
            for side, surf_tmpl in [("a", surf_a_tmpl), ("b", surf_b_tmpl)]:
                surface_text = surf_tmpl.replace("{var}", var)

                for chain_name, chain_tmpl in CHAINS.items():
                    chain_text = chain_tmpl.replace("{var}", var)
                    full_suffix = surface_text + chain_text + query

                    dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                    n += 1

                    observations.append({
                        "depth": depth, "var": var, "val": val,
                        "pair": pair_name, "side": side,
                        "chain": chain_name,
                        "dist": dist.tolist(),
                    })

                    if n % 100 == 0:
                        elapsed = time.time() - t0
                        print(f"  {n}/{n_total} readouts, {elapsed:.0f}s, "
                              f"{n/elapsed:.1f}/s", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {n} readouts in {elapsed:.1f}s ({n/elapsed:.1f}/s)")

    result_dir = Path("experiments/results/omega_repair")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    obs_idx = {}
    for obs in observations:
        key = (obs["var"], obs["val"], obs["pair"], obs["side"], obs["chain"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    for tau in [0.01, 0.02, 0.05]:
        print(f"\n=== Repair Chains (tau={tau}) ===")
        print(f"{'Chain':<30s} {'pair':<20s} {'mean_tv':>8s} {'max_tv':>8s} "
              f"{'n':>3s} {'status':>8s}")
        print("-" * 80)

        for pair_name in [p[2] for p in SURFACE_PAIRS]:
            for chain_name in CHAINS:
                tvs = []
                for _, var, val in CELLS:
                    ka_base = (var, val, pair_name, "a", "none")
                    kb_base = (var, val, pair_name, "b", "none")
                    if ka_base not in obs_idx or kb_base not in obs_idx:
                        continue
                    d_pre = 0.5 * np.abs(obs_idx[ka_base] - obs_idx[kb_base]).sum()
                    if d_pre > tau:
                        continue

                    ka = (var, val, pair_name, "a", chain_name)
                    kb = (var, val, pair_name, "b", chain_name)
                    if ka in obs_idx and kb in obs_idx:
                        d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                        tvs.append(d)

                if tvs:
                    mean_tv = float(np.mean(tvs))
                    max_tv = float(np.max(tvs))
                    status = "PRESERVE" if max_tv <= tau else "AMPLIFY"
                    print(f"  {chain_name:<28s} {pair_name:<20s} "
                          f"{mean_tv:8.5f} {max_tv:8.5f} {len(tvs):3d} "
                          f"{status:>8s}")

    print(f"\nSaved {n} observations to {result_dir}/")
    del adapter
    return observations


if __name__ == "__main__":
    run_repair()
