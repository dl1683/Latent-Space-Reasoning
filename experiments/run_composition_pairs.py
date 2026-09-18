"""Composition Ordering Across Pair Families (R6 Experiment #2).

Tests whether the super-linear composition effect (neutral_then_identity = 1.8x
sum of parts) holds across multiple surface pair families, or is pair-specific.

3 pair families x 6 cells x 3 chains x 2 members = 108 readouts per chain set.
Plus baselines (empty) = ~132 total. ~3 minutes.
"""

import json, sys, time, gc
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix

MODEL_CFG = {"model_id": "Qwen/Qwen3-1.7B-Base"}

TEMPLATE = (
    "{var} = {outer_val}\n"
    "def f():\n"
    "    {var} = 99\n"
    "    def g():\n"
    "        {var} = 999\n"
    "        def h():\n"
    "            {var} = 9999\n"
    "            def k():\n"
    "                {var} = 99999\n"
    "                return {var}\n"
    "            k()\n"
    "            return {var}\n"
    "        h()\n"
    "        return {var}\n"
    "    g()\n"
    "    return {var}\n"
)

PAIR_FAMILIES = {
    "assert": ("# {var} stays the same.\n", "# State unchanged.\n"),
    "boundary": ("\n", "pass\n"),
    "minimal": ("# OK.\n", "# Good.\n"),
}

CHAINS = {
    "neutral_then_identity": ("_ = 0\n", "{var} = {var}\n"),
    "identity_then_neutral": ("{var} = {var}\n", "_ = 0\n"),
    "neutral_then_heavy": ("_ = 0\n", "{var} = {var} + 0\n"),
}

SINGLE_CONTS = {
    "empty": "",
    "identity": "{var} = {var}\n",
    "neutral": "_ = 0\n",
    "heavy": "{var} = {var} + 0\n",
}

CELLS = [
    ("x", 1), ("x", 5),
    ("y", 2), ("y", 6),
    ("z", 3), ("z", 7),
]

QUERY = "f()\nprint({var})  # Output: "
DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def tv(a, b):
    return 0.5 * np.abs(np.array(a) - np.array(b)).sum()


def run():
    print("=== Composition Across Pair Families ===", flush=True)
    adapter = ModelAdapter(MODEL_CFG)
    observations = []
    t0 = time.time()
    n = 0

    for var, val in CELLS:
        prefix = build_prefix(TEMPLATE, var=var, outer_val=val)
        query = QUERY.format(var=var)
        ps = adapter.get_state_after_prefix(prefix)

        for pf_name, (s1, s2) in PAIR_FAMILIES.items():
            surf1 = s1.replace("{var}", var)
            surf2 = s2.replace("{var}", var)

            for surf_label, surf in [("member_1", surf1), ("member_2", surf2)]:
                # Singles (baselines)
                for cn, ct in SINGLE_CONTS.items():
                    cont = ct.replace("{var}", var)
                    dist = adapter.get_dist_from_state(ps, surf + cont + query)
                    n += 1
                    ai = int(np.argmax(dist))
                    observations.append({
                        "pair_family": pf_name,
                        "surface_member": surf_label,
                        "var": var, "val": val,
                        "chain_type": "single",
                        "continuation": cn,
                        "dist": dist.tolist(),
                        "argmax_label": DIGIT_LABELS[ai],
                        "argmax_prob": float(dist[ai]),
                        "correct_digit": str(val),
                        "is_correct": DIGIT_LABELS[ai] == str(val),
                    })

                # Chains
                for chain_name, (c1, c2) in CHAINS.items():
                    cont1 = c1.replace("{var}", var)
                    cont2 = c2.replace("{var}", var)
                    dist = adapter.get_dist_from_state(
                        ps, surf + cont1 + cont2 + query)
                    n += 1
                    ai = int(np.argmax(dist))
                    observations.append({
                        "pair_family": pf_name,
                        "surface_member": surf_label,
                        "var": var, "val": val,
                        "chain_type": "chain",
                        "continuation": chain_name,
                        "dist": dist.tolist(),
                        "argmax_label": DIGIT_LABELS[ai],
                        "argmax_prob": float(dist[ai]),
                        "correct_digit": str(val),
                        "is_correct": DIGIT_LABELS[ai] == str(val),
                    })

        if n % 84 == 0:
            print(f"  {n} readouts, {time.time()-t0:.0f}s", flush=True)

        del ps
        gc.collect()

    elapsed = time.time() - t0
    print(f"Done: {n} readouts in {elapsed:.1f}s")

    # Analysis: per pair family, compute TV between members for each chain
    results = {}
    for pf_name in PAIR_FAMILIES:
        results[pf_name] = {}
        all_conts = list(SINGLE_CONTS.keys()) + list(CHAINS.keys())

        for cn in all_conts:
            tvs = []
            for var, val in CELLS:
                m1 = [o for o in observations
                      if o["pair_family"] == pf_name
                      and o["surface_member"] == "member_1"
                      and o["continuation"] == cn
                      and o["var"] == var and o["val"] == val]
                m2 = [o for o in observations
                      if o["pair_family"] == pf_name
                      and o["surface_member"] == "member_2"
                      and o["continuation"] == cn
                      and o["var"] == var and o["val"] == val]
                if m1 and m2:
                    tvs.append(tv(m1[0]["dist"], m2[0]["dist"]))

            if tvs:
                results[pf_name][cn] = {
                    "mean_tv": round(float(np.mean(tvs)), 4),
                    "max_tv": round(float(np.max(tvs)), 4),
                    "n_cells": len(tvs),
                }

    # Super-linearity check per pair family
    for pf_name in PAIR_FAMILIES:
        pf = results[pf_name]
        if "neutral" in pf and "identity" in pf and "neutral_then_identity" in pf:
            sum_parts = pf["neutral"]["mean_tv"] + pf["identity"]["mean_tv"]
            chain_tv = pf["neutral_then_identity"]["mean_tv"]
            pf["super_linearity"] = {
                "sum_of_parts": round(sum_parts, 4),
                "chain": round(chain_tv, 4),
                "ratio": round(chain_tv / max(sum_parts, 0.0001), 2),
                "is_super_linear": chain_tv > 1.2 * sum_parts,
            }

        # Order asymmetry
        if "neutral_then_identity" in pf and "identity_then_neutral" in pf:
            fwd = pf["neutral_then_identity"]["mean_tv"]
            rev = pf["identity_then_neutral"]["mean_tv"]
            pf["order_asymmetry"] = {
                "neutral_first": round(fwd, 4),
                "identity_first": round(rev, 4),
                "ratio": round(fwd / max(rev, 0.0001), 2),
            }

    summary = {
        "n_readouts": n,
        "per_pair_family": results,
        "elapsed_s": round(elapsed, 1),
    }

    # Verdict
    super_linear_count = sum(
        1 for pf in results.values()
        if "super_linearity" in pf and pf["super_linearity"]["is_super_linear"])

    if super_linear_count == len(PAIR_FAMILIES):
        summary["verdict"] = (
            "GENERAL: super-linearity holds across all pair families — "
            "genuine transformer composition property")
    elif super_linear_count > 0:
        summary["verdict"] = (
            f"PARTIAL: super-linearity holds in {super_linear_count}/"
            f"{len(PAIR_FAMILIES)} pair families — partially general")
    else:
        summary["verdict"] = (
            "PAIR_SPECIFIC: super-linearity is pair-dependent — "
            "descriptive observation, not general property")

    out_dir = Path("experiments/results/omega_composition_pairs")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    print(f"\n=== COMPOSITION ACROSS PAIRS ===")
    for pf_name, pf in results.items():
        print(f"\n  {pf_name}:")
        for cn, data in pf.items():
            if isinstance(data, dict) and "mean_tv" in data:
                print(f"    {cn:25s}: TV={data['mean_tv']:.4f}")
        if "super_linearity" in pf:
            sl = pf["super_linearity"]
            print(f"    super-linear: chain={sl['chain']:.4f} "
                  f"sum={sl['sum_of_parts']:.4f} "
                  f"ratio={sl['ratio']:.1f}x "
                  f"{'YES' if sl['is_super_linear'] else 'NO'}")
        if "order_asymmetry" in pf:
            oa = pf["order_asymmetry"]
            print(f"    order: neutral_first={oa['neutral_first']:.4f} "
                  f"identity_first={oa['identity_first']:.4f} "
                  f"ratio={oa['ratio']:.1f}x")

    print(f"\n  VERDICT: {summary['verdict']}")
    del adapter
    gc.collect()
    return summary


if __name__ == "__main__":
    run()
