"""Omega Composition: Does amplification compose across continuation chains?

Tests whether omega of (c1 then c2) relates to omega of c1 and c2 separately.
Uses KV cache for efficiency.

Key questions:
- Does neutral+neutral = still neutral? (contraction composed)
- Does neutral+heavy = heavy? (contraction then amplification)
- Does heavy+neutral = heavy? (amplification then contraction)
- Does heavy+heavy = super-heavy? (double amplification)
- Order effects: is c1;c2 different from c2;c1?
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix, build_query


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

SURFACES_A = {
    "assert_novar_2": "# State unchanged.\n",
    "assert_var_2": "# {var} stays the same.\n",
}

SINGLE_CONTS = {
    "none": "",
    "neutral": "# No changes.\n",
    "heavy": "# Reassigning {var} now.\n",
    "bare": "\n",
    "pass": "pass\n",
    "identity": "{var} = {var}\n",
    "verified": "# Verified.\n",
}

CHAIN_CONTS = {}
for n1, t1 in SINGLE_CONTS.items():
    if n1 == "none":
        continue
    for n2, t2 in SINGLE_CONTS.items():
        if n2 == "none":
            continue
        CHAIN_CONTS[f"{n1}_then_{n2}"] = t1 + t2

CELLS = [
    (4, "x", 1), (4, "x", 3), (4, "x", 5), (4, "x", 7),
    (4, "y", 2), (4, "y", 4), (4, "y", 6), (4, "y", 8),
    (4, "z", 1), (4, "z", 3), (4, "z", 5), (4, "z", 7),
]

ALL_CONTS = {}
ALL_CONTS.update(SINGLE_CONTS)
ALL_CONTS.update(CHAIN_CONTS)


def run_composition_sweep():
    print("=== Omega Composition Sweep ===")
    print(f"Cells: {len(CELLS)}")
    print(f"Surfaces: {len(SURFACES_A)}")
    print(f"Single continuations: {len(SINGLE_CONTS)}")
    print(f"Chain continuations: {len(CHAIN_CONTS)}")
    total_conts = len(ALL_CONTS)
    n_total = len(CELLS) * len(SURFACES_A) * total_conts
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

        for surf_name, surf_tmpl in SURFACES_A.items():
            surface_text = surf_tmpl.replace("{var}", var)

            for cont_name, cont_tmpl in ALL_CONTS.items():
                cont_text = cont_tmpl.replace("{var}", var)
                full_suffix = surface_text + cont_text + query

                dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                n += 1

                observations.append({
                    "depth": depth, "var": var, "val": val,
                    "surface": surf_name, "continuation": cont_name,
                    "dist": dist.tolist(),
                })

                if n % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"  {n}/{n_total} readouts, {elapsed:.0f}s, "
                          f"{n/elapsed:.1f}/s", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {n} readouts in {elapsed:.1f}s ({n/elapsed:.1f}/s)")

    result_dir = Path("experiments/results/omega_composition")
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    obs_idx = {}
    for obs in observations:
        key = (obs["var"], obs["val"], obs["surface"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    tau = 0.01
    print(f"\n=== Composition Analysis (tau={tau}) ===")
    print()

    for (var, val) in sorted(set((c[1], c[2]) for c in CELLS)):
        key_a = (var, val, "assert_novar_2", "none")
        key_b = (var, val, "assert_var_2", "none")
        if key_a not in obs_idx or key_b not in obs_idx:
            continue
        d_pre = 0.5 * np.abs(obs_idx[key_a] - obs_idx[key_b]).sum()
        if d_pre > tau:
            continue

        print(f"Cell ({var},{val}): d_pre={d_pre:.6f}")
        row = []
        for cn in sorted(ALL_CONTS.keys()):
            ka = (var, val, "assert_novar", cn)
            kb = (var, val, "assert_var", cn)
            if ka in obs_idx and kb in obs_idx:
                d = 0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum()
                row.append((cn, float(d)))
        for cn, d in sorted(row, key=lambda x: x[1], reverse=True)[:15]:
            marker = "<<" if d > tau else ""
            print(f"  {cn:35s} tv={d:.6f} {marker}")
        print()

    singles = {}
    chains = {}
    for (var, val) in sorted(set((c[1], c[2]) for c in CELLS)):
        key_a_none = (var, val, "assert_novar_2", "none")
        key_b_none = (var, val, "assert_var_2", "none")
        if key_a_none not in obs_idx or key_b_none not in obs_idx:
            continue
        d_pre = 0.5 * np.abs(obs_idx[key_a_none] - obs_idx[key_b_none]).sum()
        if d_pre > tau:
            continue

        for cn, _ in ALL_CONTS.items():
            ka = (var, val, "assert_novar_2", cn)
            kb = (var, val, "assert_var_2", cn)
            if ka not in obs_idx or kb not in obs_idx:
                continue
            d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
            if "_then_" in cn:
                chains.setdefault(cn, []).append(d)
            else:
                singles.setdefault(cn, []).append(d)

    print("\n=== Composition Rules ===")
    print(f"{'Continuation':35s} {'mean_tv':>10s} {'max_tv':>10s} {'n':>4s}")
    for cn in sorted(singles.keys()):
        vals = singles[cn]
        print(f"{cn:35s} {np.mean(vals):10.6f} {np.max(vals):10.6f} {len(vals):4d}")
    print()
    for cn in sorted(chains.keys(), key=lambda c: np.mean(chains[c]), reverse=True)[:20]:
        vals = chains[cn]
        print(f"{cn:35s} {np.mean(vals):10.6f} {np.max(vals):10.6f} {len(vals):4d}")

    with open(result_dir / "summary.json", "w") as f:
        json.dump({
            "singles": {k: {"mean": float(np.mean(v)), "max": float(np.max(v)),
                            "n": len(v)} for k, v in singles.items()},
            "chains": {k: {"mean": float(np.mean(v)), "max": float(np.max(v)),
                           "n": len(v)} for k, v in chains.items()},
        }, f, indent=2)

    print(f"\nSaved {n} observations to {result_dir}/")
    del adapter
    return observations


if __name__ == "__main__":
    run_composition_sweep()
