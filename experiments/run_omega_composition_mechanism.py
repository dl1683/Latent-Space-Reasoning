"""Omega Composition Mechanism (Experiment #70).

WHY does neutral_then_identity amplify 2.1x above identity alone?
What is the mechanism of non-commutative composition?

Tests:
1. Per-step readout: measure omega after JUST the first continuation,
   then after the second. Does neutral "prime" a state that identity
   then amplifies?
2. Margin tracking: does neutral reduce margin (increase uncertainty)
   so identity has more room to differentiate?
3. Argmax stability: does neutral change the argmax, or only the
   distribution shape?
4. Three-way comparison: neutral_then_identity vs identity vs
   neutral+identity (both together, no ordering)

Quick: ~200 forwards, ~5 minutes.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv

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

STEPS = {
    "neutral": "# Processing...\n",
    "identity": "{var} = {var}\n",
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
    "verified": "# Verified.\n",
}

CHAINS = [
    ("none", []),
    ("neutral", ["neutral"]),
    ("identity", ["identity"]),
    ("heavy", ["heavy"]),
    ("neutral_then_identity", ["neutral", "identity"]),
    ("identity_then_neutral", ["identity", "neutral"]),
    ("neutral_then_heavy", ["neutral", "heavy"]),
    ("heavy_then_neutral", ["heavy", "neutral"]),
    ("neutral_then_neutral", ["neutral", "neutral"]),
    ("identity_then_identity", ["identity", "identity"]),
]

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

TAU = 0.05


def run():
    print("=== Composition Mechanism ===", flush=True)
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

            for chain_name, chain_steps in CHAINS:
                cont_text = ""
                for step_name in chain_steps:
                    cont_text += STEPS[step_name].replace("{var}", var)

                suffix_a = sa + cont_text + query
                suffix_b = sb + cont_text + query

                dist_a = adapter.get_dist_from_state(prefix_state, suffix_a)
                dist_b = adapter.get_dist_from_state(prefix_state, suffix_b)
                omega_val = float(tv(dist_a, dist_b))

                argmax_a = int(np.argmax(dist_a))
                argmax_b = int(np.argmax(dist_b))
                margin_a = float(np.sort(dist_a)[-1] - np.sort(dist_a)[-2])
                margin_b = float(np.sort(dist_b)[-1] - np.sort(dist_b)[-2])
                entropy_a = float(-np.sum(dist_a[dist_a > 0] *
                                          np.log2(dist_a[dist_a > 0])))
                entropy_b = float(-np.sum(dist_b[dist_b > 0] *
                                          np.log2(dist_b[dist_b > 0])))

                intermediate_tvs = []
                if len(chain_steps) == 2:
                    step1_text = STEPS[chain_steps[0]].replace("{var}", var)
                    mid_a = sa + step1_text + query
                    mid_b = sb + step1_text + query
                    dist_ma = adapter.get_dist_from_state(prefix_state, mid_a)
                    dist_mb = adapter.get_dist_from_state(prefix_state, mid_b)
                    intermediate_tvs.append({
                        "after_step": chain_steps[0],
                        "tv": float(tv(dist_ma, dist_mb)),
                        "margin_a": float(np.sort(dist_ma)[-1] - np.sort(dist_ma)[-2]),
                        "margin_b": float(np.sort(dist_mb)[-1] - np.sort(dist_mb)[-2]),
                        "entropy_a": float(-np.sum(dist_ma[dist_ma > 0] *
                                                    np.log2(dist_ma[dist_ma > 0]))),
                    })

                rec = {
                    "cell": cell_id,
                    "pair": pair_name,
                    "chain": chain_name,
                    "eligible": eligible,
                    "base_tv": base_tv,
                    "omega": omega_val,
                    "argmax_a": argmax_a,
                    "argmax_b": argmax_b,
                    "argmax_same": argmax_a == argmax_b,
                    "margin_a": margin_a,
                    "margin_b": margin_b,
                    "entropy_a": entropy_a,
                    "entropy_b": entropy_b,
                    "intermediates": intermediate_tvs,
                }
                records.append(rec)
                tag = "[elig]" if eligible else "[INELIG]"
                mid_info = ""
                if intermediate_tvs:
                    mid_info = f" mid={intermediate_tvs[0]['tv']:.4f}"
                print(f"  {cell_id}/{pair_name}/{chain_name}: "
                      f"omega={omega_val:.4f} m={margin_a:.3f}/{margin_b:.3f}"
                      f"{mid_info} {tag}", flush=True)

    elapsed = time.time() - t0

    elig = [r for r in records if r["eligible"]]
    by_chain = {}
    for chain_name, _ in CHAINS:
        ce = [r for r in elig if r["chain"] == chain_name]
        if ce:
            omegas = [r["omega"] for r in ce]
            margins_a = [r["margin_a"] for r in ce]
            entropies_a = [r["entropy_a"] for r in ce]
            by_chain[chain_name] = {
                "n": len(ce),
                "mean_omega": float(np.mean(omegas)),
                "mean_margin_a": float(np.mean(margins_a)),
                "mean_entropy_a": float(np.mean(entropies_a)),
            }

    chain_2step = [r for r in elig if r["intermediates"]]
    priming = []
    for r in chain_2step:
        mid = r["intermediates"][0]
        priming.append({
            "chain": r["chain"],
            "mid_tv": mid["tv"],
            "final_tv": r["omega"],
            "mid_margin_a": mid["margin_a"],
            "final_margin_a": r["margin_a"],
            "amplification": r["omega"] / max(mid["tv"], 1e-10),
        })

    summary = {
        "n_total": len(records),
        "n_eligible": len(elig),
        "by_chain": by_chain,
        "priming_effects": priming,
        "elapsed_s": round(elapsed, 1),
    }

    outdir = Path("experiments/results/omega_composition_mechanism")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "rows.json", "w") as f:
        json.dump(records, f, indent=2)
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== SUMMARY ({elapsed:.0f}s) ===", flush=True)
    print(f"Eligible: {len(elig)}/{len(records)}", flush=True)
    for chain, stats in by_chain.items():
        print(f"  {chain:30s}: omega={stats['mean_omega']:.4f} "
              f"margin={stats['mean_margin_a']:.3f} "
              f"entropy={stats['mean_entropy_a']:.3f}", flush=True)

    if priming:
        print(f"\nPriming effects (2-step chains):", flush=True)
        for p in priming[:10]:
            print(f"  {p['chain']:30s}: mid={p['mid_tv']:.4f} -> "
                  f"final={p['final_tv']:.4f} "
                  f"(amp={p['amplification']:.1f}x) "
                  f"margin: {p['mid_margin_a']:.3f}->{p['final_margin_a']:.3f}",
                  flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    run()
