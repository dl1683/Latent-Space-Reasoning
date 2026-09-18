"""Omega Chain Amplification (Experiment #47).

Do sequences of continuations create larger divergence than single ones?
Exp #46 found that the safest-looking cell (lowest baseline TV=0.00022)
has the highest max omega (0.687). This experiment tests whether
CHAINING continuations on that cell creates even larger divergence.

Design:
  - Focus on strong_assert_2 pair (the one with eligible cells)
  - Test chains of 1, 2, 3 continuations
  - Mix amplifying and preserving continuations
  - Measure: does omega grow, saturate, or decay with chain length?

Chains tested:
  Single: I, H, N, T (identity, heavy, neutral, type_check)
  Double: II, IH, IN, HI, HH, HN, NI, NH, NN, TI, TT
  Triple: III, IHI, INI, HIH, NIN, TTT
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

SURFACES = {
    "assert_novar_2": "# State unchanged.\n",
    "assert_var_2": "# {var} stays the same.\n",
}

EQUIV_PAIRS = [
    ("assert_novar_2", "assert_var_2", "strong_assert_2"),
]

ATOMIC = {
    "I": "{var} = {var}\n",
    "H": "{var} = {var} + 1\n",
    "N": "pass\n",
    "T": "type({var})\n",
    "O": "_ = {var}\n",
}

CHAINS = {
    # Single
    "I": ["I"],
    "H": ["H"],
    "N": ["N"],
    "T": ["T"],
    "O": ["O"],
    # Double
    "II": ["I", "I"],
    "IH": ["I", "H"],
    "IN": ["I", "N"],
    "HI": ["H", "I"],
    "HH": ["H", "H"],
    "HN": ["H", "N"],
    "NI": ["N", "I"],
    "NH": ["N", "H"],
    "NN": ["N", "N"],
    "TI": ["T", "I"],
    "TT": ["T", "T"],
    "OI": ["O", "I"],
    # Triple
    "III": ["I", "I", "I"],
    "IHI": ["I", "H", "I"],
    "INI": ["I", "N", "I"],
    "HIH": ["H", "I", "H"],
    "NIN": ["N", "I", "N"],
    "TTT": ["T", "T", "T"],
    "NNN": ["N", "N", "N"],
    "HHH": ["H", "H", "H"],
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]


def chain_to_text(chain, var):
    parts = []
    for code in chain:
        parts.append(ATOMIC[code].replace("{var}", var))
    return "".join(parts)


def run_chain_experiment():
    print("=== Omega Chain Amplification (Exp #47) ===")
    n_target = len(CELLS) * len(SURFACES) * len(CHAINS)
    print(f"Target: {n_target} readouts")

    adapter = ModelAdapter(MODEL_CFG)
    query_tmpl = MODEL_CFG["query_template"]
    observations = []
    t0 = time.time()
    n_readouts = 0

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for surf_name, surf_tmpl in SURFACES.items():
            surface_text = surf_tmpl.replace("{var}", var)

            for chain_name, chain_codes in CHAINS.items():
                chain_text = chain_to_text(chain_codes, var)
                full_suffix = surface_text + chain_text + query
                dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                n_readouts += 1

                observations.append({
                    "depth": depth,
                    "var": var,
                    "val": val,
                    "surface": surf_name,
                    "chain": chain_name,
                    "chain_length": len(chain_codes),
                    "dist": dist.tolist(),
                })

                if n_readouts % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  {n_readouts}/{n_target}, {elapsed:.0f}s",
                          flush=True)

    elapsed = time.time() - t0
    print(f"Done: {n_readouts} readouts in {elapsed:.1f}s")

    result_dir = Path("experiments/results/omega_chain")
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["chain"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    summary = {"n_readouts": n_readouts, "elapsed_s": elapsed, "pairs": {}}

    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        for tau in [0.01, 0.05]:
            eligible_cells = set()
            for depth, var, val in CELLS:
                key_a = (depth, var, val, surf_a, "N")
                key_b = (depth, var, val, surf_b, "N")
                if key_a in obs_idx and key_b in obs_idx:
                    d = tv(obs_idx[key_a], obs_idx[key_b])
                    if d <= tau:
                        eligible_cells.add((depth, var, val))

            tau_key = f"tau_{tau}"
            if pair_name not in summary["pairs"]:
                summary["pairs"][pair_name] = {}

            print(f"\n--- {pair_name} at tau={tau} "
                  f"({len(eligible_cells)} eligible) ---")

            chain_omega = {}
            for chain_name in CHAINS:
                tvs = []
                for cell in eligible_cells:
                    depth, var, val = cell
                    key_a = (depth, var, val, surf_a, chain_name)
                    key_b = (depth, var, val, surf_b, chain_name)
                    if key_a in obs_idx and key_b in obs_idx:
                        tvs.append(tv(obs_idx[key_a], obs_idx[key_b]))
                if tvs:
                    omega = float(np.max(tvs))
                    chain_omega[chain_name] = {
                        "omega": omega,
                        "length": len(CHAINS[chain_name]),
                        "status": "AMPLIFY" if omega > tau else "PRESERVE",
                    }

            by_length = {}
            for cn, cd in sorted(chain_omega.items(),
                                 key=lambda x: x[1]["omega"], reverse=True):
                l = cd["length"]
                if l not in by_length:
                    by_length[l] = []
                by_length[l].append((cn, cd["omega"]))
                print(f"  {cn:6s} (len={l}): omega={cd['omega']:.5f} "
                      f"{cd['status']}")

            print(f"\n  By chain length:")
            for l in sorted(by_length):
                omegas = [o for _, o in by_length[l]]
                print(f"    Length {l}: mean={np.mean(omegas):.5f} "
                      f"max={np.max(omegas):.5f} "
                      f"min={np.min(omegas):.5f}")

            summary["pairs"][pair_name][tau_key] = chain_omega

    # Growth analysis
    print("\n=== Growth Analysis: does chaining increase omega? ===")
    growth = {}
    for pair_name in [p[2] for p in EQUIV_PAIRS]:
        for tau_key in summary["pairs"].get(pair_name, {}):
            pd = summary["pairs"][pair_name][tau_key]
            comparisons = [
                ("I->II->III", ["I", "II", "III"]),
                ("H->HH->HHH", ["H", "HH", "HHH"]),
                ("N->NN->NNN", ["N", "NN", "NNN"]),
                ("T->TT->TTT", ["T", "TT", "TTT"]),
            ]
            print(f"\n  {pair_name} {tau_key}:")
            for label, chain_names in comparisons:
                vals = [pd[cn]["omega"] for cn in chain_names if cn in pd]
                if len(vals) >= 2:
                    trend = "GROWING" if vals[-1] > vals[0] else "SHRINKING"
                    ratio = vals[-1] / vals[0] if vals[0] > 0.001 else None
                    print(f"    {label:20s}: "
                          f"{' -> '.join(f'{v:.4f}' for v in vals)} "
                          f"({trend}"
                          f"{f', {ratio:.2f}x' if ratio else ''})")
                    growth[f"{pair_name}_{tau_key}_{label}"] = {
                        "values": vals, "trend": trend, "ratio": ratio
                    }

    summary["growth"] = growth

    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {result_dir}/")
    return observations, summary


if __name__ == "__main__":
    run_chain_experiment()
