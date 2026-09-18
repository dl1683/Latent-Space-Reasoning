"""Omega Revival Scaling (Experiment #72).

Does latent information persistence scale with model size?

Tests revival (contraction -> probe) on Qwen3 0.6B, 1.7B, 4B Base.
If revival ratio increases with size, larger models retain more
latent information through contraction steps.

~600 forwards total (~200 per model), ~10 minutes.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv

BASE_CFG = {
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

MODELS = [
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B-Base",
]

SURFACES = {
    "assert_novar_2": "# State unchanged.\n",
    "assert_var_2": "# {var} stays the same.\n",
    "boundary_1": "\n",
    "boundary_2": "pass\n",
}

EQUIV_PAIRS = [
    ("assert_var_2", "assert_novar_2", "strong_assert_2"),
    ("boundary_1", "boundary_2", "boundary_pair"),
]

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "z", 3),
]

TAU = 0.05

CHAINS = {
    "base": "",
    "identity": "{var} = {var}\n",
    "contract": "{var} = None\n",
    "contract_then_probe": "{var} = None\nisinstance({var}, int)\n",
    "contract_x2_then_probe": "{var} = None\n{var} = None\nisinstance({var}, int)\n",
}


def run():
    print("=== Revival Scaling Test ===", flush=True)
    t0 = time.time()
    all_results = {}

    for model_id in MODELS:
        model_name = model_id.split("/")[-1]
        print(f"\n--- {model_name} ---", flush=True)
        cfg = {**BASE_CFG, "model_id": model_id}
        adapter = ModelAdapter(cfg)
        query_tmpl = cfg["query_template"]
        records = []

        for depth, var, val in CELLS:
            tmpl = cfg["templates"]["depth4_single"]
            prefix_text = build_prefix(tmpl, var=var, outer_val=val)
            query = query_tmpl.replace("{var}", var)
            prefix_state = adapter.get_state_after_prefix(prefix_text)
            cell_id = f"d{depth}_{var}_{val}"

            for surf_a, surf_b, pair_name in EQUIV_PAIRS:
                sa = SURFACES[surf_a].replace("{var}", var)
                sb = SURFACES[surf_b].replace("{var}", var)

                dist_a_base = adapter.get_dist_from_state(
                    prefix_state, sa + query)
                dist_b_base = adapter.get_dist_from_state(
                    prefix_state, sb + query)
                base_tv = float(tv(dist_a_base, dist_b_base))
                eligible = base_tv <= TAU
                if not eligible:
                    continue

                for chain_name, chain_tmpl in CHAINS.items():
                    chain_text = chain_tmpl.replace("{var}", var)
                    suffix_a = sa + chain_text + query
                    suffix_b = sb + chain_text + query
                    dist_a = adapter.get_dist_from_state(
                        prefix_state, suffix_a)
                    dist_b = adapter.get_dist_from_state(
                        prefix_state, suffix_b)
                    omega_val = float(tv(dist_a, dist_b))

                    rec = {
                        "cell": cell_id,
                        "pair": pair_name,
                        "chain": chain_name,
                        "base_tv": base_tv,
                        "omega": omega_val,
                    }
                    records.append(rec)

                print(f"  {cell_id}/{pair_name}: done", flush=True)

        by_chain = {}
        for chain_name in CHAINS:
            recs = [r for r in records if r["chain"] == chain_name]
            if recs:
                omegas = [r["omega"] for r in recs]
                by_chain[chain_name] = {
                    "n": len(recs),
                    "mean": float(np.mean(omegas)),
                    "std": float(np.std(omegas)),
                }

        contract_mean = by_chain.get("contract", {}).get("mean", 0)
        revival_mean = by_chain.get("contract_then_probe", {}).get("mean", 0)
        revival_ratio = revival_mean / max(contract_mean, 1e-10)

        revival2_mean = by_chain.get(
            "contract_x2_then_probe", {}).get("mean", 0)
        revival2_ratio = revival2_mean / max(contract_mean, 1e-10)

        model_result = {
            "model": model_name,
            "n_records": len(records),
            "by_chain": by_chain,
            "revival_ratio_1x": revival_ratio,
            "revival_ratio_2x": revival2_ratio,
        }
        all_results[model_name] = model_result
        print(f"  Revival ratio (1x contract): {revival_ratio:.2f}x",
              flush=True)
        print(f"  Revival ratio (2x contract): {revival2_ratio:.2f}x",
              flush=True)

        del adapter

    elapsed = time.time() - t0

    summary = {
        "models": all_results,
        "elapsed_s": round(elapsed, 1),
    }

    outdir = Path("experiments/results/omega_revival_scaling")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== SCALING SUMMARY ({elapsed:.0f}s) ===", flush=True)
    for name, res in all_results.items():
        print(f"  {name}: revival_1x={res['revival_ratio_1x']:.2f}x "
              f"revival_2x={res['revival_ratio_2x']:.2f}x", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    run()
