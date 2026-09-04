"""Quotient-composition congruence test.

If a ~ a' (behavioral equivalence), does a;b ~ a';b on held-out contexts?
Tests whether behavioral equivalence is an algebraic congruence preserved
under suffix composition.

Uses the full 11-bin distribution, not just delta_L.
"""
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_svb_0 import ModelAdapter, build_prefix, build_query


def expand_surface(surface, var):
    return surface.replace("{var}", var)


EQUIV_PAIRS = [
    {
        "name": "strong_assert",
        "a": "# State unchanged.\n",
        "a_prime": "# {var} is unchanged.\n",
        "note": "ASSERT ~ ASSERT_VAR, dL diff=0.006",
    },
    {
        "name": "moderate_assert",
        "a": "# Values preserved.\n",
        "a_prime": "# {var} is still intact.\n",
        "note": "ASSERT ~ ASSERT_VAR, dL diff=0.005",
    },
    {
        "name": "misleading",
        "a": "# Changing {var}.\n",
        "a_prime": "# Updating to a new value.\n",
        "note": "MISLEADING ~ MISLEADING_NOVAR, dL diff=0.002",
    },
]

COMPOSERS = [
    "# No changes.\n",
    "# Reassigning {var} now.\n",
]


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else \
        "config/svb_qwen3_composition_v2.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    result_dir = Path("results/svb_qwen3_congruence")
    result_dir.mkdir(parents=True, exist_ok=True)

    adapter = ModelAdapter(cfg)
    print("Model loaded.", flush=True)

    variables = cfg["variables"]
    outer_values = cfg["outer_values"]
    depth = 4
    tmpl = cfg["templates"]["depth4_single"]

    observations = {}
    t0 = time.time()

    for var in variables:
        for val in outer_values:
            prefix = build_prefix(tmpl, var=var, outer_val=val)
            state = adapter.get_state_after_prefix(prefix)
            query = build_query(var, cfg)
            ctx_key = f"d{depth}_{var}_{val}"

            # Baseline (no suffix)
            dist = adapter.get_dist_from_state(state, query, deepcopy=True)
            observations[f"{ctx_key}_baseline"] = {
                "arm": "baseline", "dist": dist.tolist(),
                "depth": depth, "var": var, "val": val,
            }

            for pair in EQUIV_PAIRS:
                for label, surface in [("a", pair["a"]), ("a_prime", pair["a_prime"])]:
                    # Single suffix
                    s = expand_surface(surface, var) + query
                    dist = adapter.get_dist_from_state(state, s, deepcopy=True)
                    key = f"{ctx_key}_{pair['name']}_{label}"
                    observations[key] = {
                        "arm": f"{pair['name']}_{label}", "dist": dist.tolist(),
                        "depth": depth, "var": var, "val": val,
                    }

                    # Composed: suffix + each composer
                    for ci, composer in enumerate(COMPOSERS):
                        s2 = expand_surface(surface, var) + expand_surface(composer, var) + query
                        dist = adapter.get_dist_from_state(state, s2, deepcopy=True)
                        key2 = f"{ctx_key}_{pair['name']}_{label}_c{ci}"
                        observations[key2] = {
                            "arm": f"{pair['name']}_{label}_c{ci}", "dist": dist.tolist(),
                            "depth": depth, "var": var, "val": val,
                        }

                        # Reversed: composer + suffix
                        s3 = expand_surface(composer, var) + expand_surface(surface, var) + query
                        dist = adapter.get_dist_from_state(state, s3, deepcopy=True)
                        key3 = f"{ctx_key}_{pair['name']}_{label}_c{ci}r"
                        observations[key3] = {
                            "arm": f"{pair['name']}_{label}_c{ci}r", "dist": dist.tolist(),
                            "depth": depth, "var": var, "val": val,
                        }

            del state
        gc.collect()

    elapsed = time.time() - t0
    print(f"\nTotal: {adapter.call_count} calls, {elapsed:.1f}s", flush=True)

    # === ANALYSIS ===
    print("\n=== CONGRUENCE TEST ===\n")

    for pair in EQUIV_PAIRS:
        print(f"--- {pair['name']}: {pair['note']} ---")

        # Single-suffix equivalence (verification)
        single_tvs = []
        for var in variables:
            for val in outer_values:
                ctx = f"d{depth}_{var}_{val}"
                d_a = np.array(observations[f"{ctx}_{pair['name']}_a"]["dist"])
                d_ap = np.array(observations[f"{ctx}_{pair['name']}_a_prime"]["dist"])
                tv = 0.5 * np.sum(np.abs(d_a - d_ap))
                single_tvs.append(tv)
        print(f"  Single-suffix TV(a, a'): mean={np.mean(single_tvs):.4f}, "
              f"median={np.median(single_tvs):.4f}, max={max(single_tvs):.4f}")

        for ci, composer in enumerate(COMPOSERS):
            # Composed equivalence: TV(a;c, a';c)
            comp_tvs = []
            comp_r_tvs = []
            for var in variables:
                for val in outer_values:
                    ctx = f"d{depth}_{var}_{val}"
                    d_ac = np.array(observations[f"{ctx}_{pair['name']}_a_c{ci}"]["dist"])
                    d_apc = np.array(observations[f"{ctx}_{pair['name']}_a_prime_c{ci}"]["dist"])
                    tv = 0.5 * np.sum(np.abs(d_ac - d_apc))
                    comp_tvs.append(tv)

                    d_ca = np.array(observations[f"{ctx}_{pair['name']}_a_c{ci}r"]["dist"])
                    d_cap = np.array(observations[f"{ctx}_{pair['name']}_a_prime_c{ci}r"]["dist"])
                    tv_r = 0.5 * np.sum(np.abs(d_ca - d_cap))
                    comp_r_tvs.append(tv_r)

            c_name = composer.strip()[:25]
            print(f"  Composed a;c (c={c_name}): TV={np.mean(comp_tvs):.4f} (med={np.median(comp_tvs):.4f})")
            print(f"  Composed c;a (c={c_name}): TV={np.mean(comp_r_tvs):.4f} (med={np.median(comp_r_tvs):.4f})")

            # Congruence: does composition PRESERVE equivalence?
            preserved = np.mean(comp_tvs) <= np.mean(single_tvs) * 2.0
            preserved_r = np.mean(comp_r_tvs) <= np.mean(single_tvs) * 2.0
            print(f"  Congruence (a;c): {'PASS' if preserved else 'FAIL'} "
                  f"(ratio={np.mean(comp_tvs)/np.mean(single_tvs):.2f})")
            print(f"  Congruence (c;a): {'PASS' if preserved_r else 'FAIL'} "
                  f"(ratio={np.mean(comp_r_tvs)/np.mean(single_tvs):.2f})")

    # Overall
    print("\n=== SUMMARY ===")

    all_single = []
    all_comp = []
    for pair in EQUIV_PAIRS:
        for var in variables:
            for val in outer_values:
                ctx = f"d{depth}_{var}_{val}"
                d_a = np.array(observations[f"{ctx}_{pair['name']}_a"]["dist"])
                d_ap = np.array(observations[f"{ctx}_{pair['name']}_a_prime"]["dist"])
                all_single.append(0.5 * np.sum(np.abs(d_a - d_ap)))

                for ci in range(len(COMPOSERS)):
                    d_ac = np.array(observations[f"{ctx}_{pair['name']}_a_c{ci}"]["dist"])
                    d_apc = np.array(observations[f"{ctx}_{pair['name']}_a_prime_c{ci}"]["dist"])
                    all_comp.append(0.5 * np.sum(np.abs(d_ac - d_apc)))

                    d_ca = np.array(observations[f"{ctx}_{pair['name']}_a_c{ci}r"]["dist"])
                    d_cap = np.array(observations[f"{ctx}_{pair['name']}_a_prime_c{ci}r"]["dist"])
                    all_comp.append(0.5 * np.sum(np.abs(d_ca - d_cap)))

    print(f"  Single-suffix mean TV: {np.mean(all_single):.4f}")
    print(f"  Composed mean TV:      {np.mean(all_comp):.4f}")
    print(f"  Ratio:                 {np.mean(all_comp)/np.mean(all_single):.2f}")

    if np.mean(all_comp) <= np.mean(all_single) * 2.0:
        print(f"\n  -> CONGRUENCE: behavioral equivalence is approximately preserved under composition.")
        print(f"     This is evidence for an endogenous quotient algebra.")
    else:
        print(f"\n  -> NO CONGRUENCE: composition amplifies differences between equivalents.")
        print(f"     The equivalence is surface-level, not algebraic.")

    result_file = result_dir / "result.json"
    with open(result_file, "w") as f:
        json.dump({
            "equiv_pairs": EQUIV_PAIRS,
            "composers": COMPOSERS,
            "observations": observations,
            "calls": adapter.call_count,
            "elapsed_s": elapsed,
        }, f, indent=2)
    print(f"\nSaved to {result_file}", flush=True)


if __name__ == "__main__":
    main()
