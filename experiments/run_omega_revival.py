"""Omega Revival: Can amplified distance be contracted, then re-exposed?

Experiment #42. Tests whether identity-induced amplification is truly
irreversible, or whether a contracting continuation merely hides the
distinction from one readout while a subsequent probe can re-expose it.

Chain: surface_pair → identity (amplify) → contraction → revival_probe → query
"""
import json
import sys
import time
import copy
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix, build_query

MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B",
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

EQUIV_PAIRS = [
    ("# No changes.\n", "# {var} is unchanged.\n", "strong_assert"),
    ("# State unchanged.\n", "# {var} stays the same.\n", "strong_assert_2"),
    ("\n", "pass\n", "boundary_pair"),
]

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

CONTRACTIONS = {
    "blank_newlines": "\n\n",
    "reassign_42": "{var} = 42\n",
    "del_var": "del {var}\n",
    "long_neutral": "# This is a standard comment that has nothing to do with variables.\n",
    "pass_x3": "pass\npass\npass\n",
    "none_assign": "{var} = None\n",
    "empty_string": "",
}

REVIVAL_PROBES = {
    "print_var": "print({var})\n",
    "type_var": "type({var})\n",
    "add_one": "{var} + 1\n",
    "bare_var": "{var}\n",
    "isinstance_check": "isinstance({var}, int)\n",
    "none_probe": "",
}


def tv(a, b):
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum()


def run_revival():
    adapter = ModelAdapter(MODEL_CFG)
    query_tmpl = MODEL_CFG["query_template"]
    observations = []
    t0 = time.time()
    n_readouts = 0

    identity_tmpl = "{var} = {var}\n"

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        identity = identity_tmpl.replace("{var}", var)

        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_a_tmpl, surf_b_tmpl, pair_name in EQUIV_PAIRS:
            surf_a = surf_a_tmpl.replace("{var}", var)
            surf_b = surf_b_tmpl.replace("{var}", var)

            # Step 0: Baseline (surface only, no continuation)
            dist_a_base = adapter.get_dist_from_state(prefix_state, surf_a + query)
            dist_b_base = adapter.get_dist_from_state(prefix_state, surf_b + query)
            tv_baseline = tv(dist_a_base, dist_b_base)
            n_readouts += 2

            # Step 1: Identity amplification
            dist_a_id = adapter.get_dist_from_state(prefix_state, surf_a + identity + query)
            dist_b_id = adapter.get_dist_from_state(prefix_state, surf_b + identity + query)
            tv_identity = tv(dist_a_id, dist_b_id)
            n_readouts += 2

            for cont_name, cont_tmpl in CONTRACTIONS.items():
                cont = cont_tmpl.replace("{var}", var)

                # Step 2: Identity + contraction
                chain_a_cont = surf_a + identity + cont + query
                chain_b_cont = surf_b + identity + cont + query
                dist_a_cont = adapter.get_dist_from_state(prefix_state, chain_a_cont)
                dist_b_cont = adapter.get_dist_from_state(prefix_state, chain_b_cont)
                tv_contracted = tv(dist_a_cont, dist_b_cont)
                n_readouts += 2

                for probe_name, probe_tmpl in REVIVAL_PROBES.items():
                    probe = probe_tmpl.replace("{var}", var)

                    if not probe:
                        # No revival probe; just record the contraction result
                        if probe_name == "none_probe":
                            observations.append({
                                "depth": depth, "var": var, "val": val,
                                "pair": pair_name,
                                "contraction": cont_name,
                                "probe": "none",
                                "tv_baseline": float(tv_baseline),
                                "tv_identity": float(tv_identity),
                                "tv_contracted": float(tv_contracted),
                                "tv_revived": float(tv_contracted),
                                "amplified": bool(tv_identity > tv_baseline * 1.5),
                                "contracted": bool(tv_contracted < tv_identity * 0.7),
                                "revived": False,
                            })
                        continue

                    # Step 3: Identity + contraction + revival probe
                    chain_a_full = surf_a + identity + cont + probe + query
                    chain_b_full = surf_b + identity + cont + probe + query
                    dist_a_rev = adapter.get_dist_from_state(prefix_state, chain_a_full)
                    dist_b_rev = adapter.get_dist_from_state(prefix_state, chain_b_full)
                    tv_revived = tv(dist_a_rev, dist_b_rev)
                    n_readouts += 2

                    observations.append({
                        "depth": depth, "var": var, "val": val,
                        "pair": pair_name,
                        "contraction": cont_name,
                        "probe": probe_name,
                        "tv_baseline": float(tv_baseline),
                        "tv_identity": float(tv_identity),
                        "tv_contracted": float(tv_contracted),
                        "tv_revived": float(tv_revived),
                        "amplified": bool(tv_identity > tv_baseline * 1.5),
                        "contracted": bool(tv_contracted < tv_identity * 0.7),
                        "revived": bool(tv_revived > tv_contracted * 1.5 if tv_contracted > 0.001 else tv_revived > 0.01),
                    })

                if n_readouts % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  {n_readouts} readouts, {elapsed:.0f}s, "
                          f"{n_readouts/elapsed:.1f}/s", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Done: {n_readouts} readouts in {elapsed:.1f}s", flush=True)

    del adapter
    return observations, elapsed


def analyze(observations):
    """Analyze which contractions reduce and which probes revive."""
    print("\n=== REVIVAL ANALYSIS ===\n")

    # Filter to eligible cells (where identity amplifies)
    eligible = [o for o in observations if o["amplified"]]
    print(f"Eligible cells (identity amplifies): {len(eligible)}/{len(observations)}")

    if not eligible:
        print("No amplification detected; cannot test revival.")
        return {}

    # Contraction effectiveness
    print("\n--- Contraction effectiveness ---")
    from collections import defaultdict
    cont_stats = defaultdict(list)
    for o in eligible:
        if o["probe"] == "none":
            cont_stats[o["contraction"]].append(o)

    for cont_name in sorted(cont_stats.keys()):
        rows = cont_stats[cont_name]
        mean_id = np.mean([r["tv_identity"] for r in rows])
        mean_cont = np.mean([r["tv_contracted"] for r in rows])
        ratio = mean_cont / mean_id if mean_id > 0 else 0
        n_contracted = sum(1 for r in rows if r["contracted"])
        print(f"  {cont_name:20s}: id={mean_id:.4f} -> cont={mean_cont:.4f} "
              f"(ratio={ratio:.2f}, {n_contracted}/{len(rows)} contracted)")

    # Revival by probe after contraction
    print("\n--- Revival by probe (after best contraction) ---")
    probe_stats = defaultdict(list)
    for o in eligible:
        if o["probe"] != "none" and o["contracted"]:
            probe_stats[o["probe"]].append(o)

    for probe_name in sorted(probe_stats.keys()):
        rows = probe_stats[probe_name]
        if not rows:
            continue
        mean_cont = np.mean([r["tv_contracted"] for r in rows])
        mean_rev = np.mean([r["tv_revived"] for r in rows])
        ratio = mean_rev / mean_cont if mean_cont > 0 else 0
        n_revived = sum(1 for r in rows if r["revived"])
        print(f"  {probe_name:20s}: cont={mean_cont:.4f} -> rev={mean_rev:.4f} "
              f"(ratio={ratio:.2f}, {n_revived}/{len(rows)} revived)")

    # Full chain analysis: which (contraction, probe) combos show revival?
    print("\n--- Best revival chains ---")
    chain_stats = defaultdict(list)
    for o in eligible:
        if o["probe"] != "none":
            chain_stats[(o["contraction"], o["probe"])].append(o)

    best_chains = []
    for (cont, probe), rows in chain_stats.items():
        mean_base = np.mean([r["tv_baseline"] for r in rows])
        mean_id = np.mean([r["tv_identity"] for r in rows])
        mean_cont = np.mean([r["tv_contracted"] for r in rows])
        mean_rev = np.mean([r["tv_revived"] for r in rows])
        revival_ratio = mean_rev / mean_cont if mean_cont > 0 else 0
        n_revived = sum(1 for r in rows if r["revived"])
        best_chains.append({
            "contraction": cont, "probe": probe,
            "mean_base": mean_base, "mean_id": mean_id,
            "mean_cont": mean_cont, "mean_rev": mean_rev,
            "revival_ratio": revival_ratio,
            "n_revived": n_revived, "n_total": len(rows),
        })

    best_chains.sort(key=lambda x: -x["revival_ratio"])
    for c in best_chains[:15]:
        tag = "REVIVED" if c["n_revived"] > 0 else ""
        print(f"  {c['contraction']:20s} + {c['probe']:20s}: "
              f"base={c['mean_base']:.4f} -> id={c['mean_id']:.4f} "
              f"-> cont={c['mean_cont']:.4f} -> rev={c['mean_rev']:.4f} "
              f"(x{c['revival_ratio']:.1f}) [{c['n_revived']}/{c['n_total']}] {tag}")

    return {
        "n_eligible": len(eligible),
        "best_chains": best_chains[:15],
    }


if __name__ == "__main__":
    out_dir = Path("experiments/results/omega_revival")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Omega Revival Experiment ===")
    print("Testing: amplify -> contract -> probe -> re-expose?\n")

    observations, elapsed = run_revival()

    with open(out_dir / "observations.json", "w") as f:
        json.dump(observations, f, indent=2)

    summary = analyze(observations)
    summary["n_readouts"] = len(observations)
    summary["elapsed_s"] = elapsed

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}")
