"""Omega Authority Framing (Experiment #43).

Tests whether source-framing comments change omega. Does adding
"According to the documentation:" or "The API spec states:" before
a continuation alter its omega? This separates content effects from
frame effects — relevant to whether instruct amplification is
content-dependent or frame-dependent.

Continuations:
  identity_bare:   x = x\n
  identity_doc:    # According to the documentation:\nx = x\n
  identity_spec:   # The API spec states:\nx = x\n
  identity_todo:   # TODO: verify this\nx = x\n
  heavy_bare:      x = x + 1  # WARNING: intentional mutation\n
  heavy_doc:       # According to the documentation:\nx = x + 1  # WARNING: intentional mutation\n
  neutral_bare:    pass\n
  neutral_doc:     # According to the documentation:\npass\n
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
    "assert_novar_1": "# No changes.\n",
    "assert_var_1": "# {var} is unchanged.\n",
    "assert_novar_2": "# State unchanged.\n",
    "assert_var_2": "# {var} stays the same.\n",
    "minimal_ok": "# OK.\n",
    "minimal_good": "# Good.\n",
}

EQUIV_PAIRS = [
    ("assert_novar_1", "assert_var_1", "strong_assert"),
    ("assert_novar_2", "assert_var_2", "strong_assert_2"),
    ("minimal_ok", "minimal_good", "minimal_ok_good"),
]

CONTINUATIONS = {
    "identity_bare": "{var} = {var}\n",
    "identity_doc": "# According to the documentation:\n{var} = {var}\n",
    "identity_spec": "# The API spec states:\n{var} = {var}\n",
    "identity_todo": "# TODO: verify this\n{var} = {var}\n",
    "heavy_bare": "{var} = {var} + 1  # WARNING: intentional mutation\n",
    "heavy_doc": "# According to the documentation:\n{var} = {var} + 1  # WARNING: intentional mutation\n",
    "neutral_bare": "pass\n",
    "neutral_doc": "# According to the documentation:\npass\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]


def run_authority_experiment():
    print("=== Omega Authority Framing (Exp #43) ===")
    n_target = len(CELLS) * len(SURFACES) * len(CONTINUATIONS)
    print(f"Target: {n_target} readouts")
    print(f"Cells: {len(CELLS)}, Surfaces: {len(SURFACES)}, "
          f"Continuations: {len(CONTINUATIONS)}")

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

            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)
                full_suffix = surface_text + cont_text + query
                dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                n_readouts += 1

                observations.append({
                    "depth": depth,
                    "var": var,
                    "val": val,
                    "surface": surf_name,
                    "continuation": cont_name,
                    "dist": dist.tolist(),
                })

                if n_readouts % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  {n_readouts}/{n_target} readouts, {elapsed:.0f}s",
                          flush=True)

    elapsed = time.time() - t0
    print(f"Done: {n_readouts} readouts in {elapsed:.1f}s "
          f"({n_readouts/elapsed:.1f}/s)")

    result_dir = Path("experiments/results/omega_authority")
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    print("\n=== Omega at tau=0.01 and tau=0.05 ===")
    summary = {"n_readouts": n_readouts, "elapsed_s": elapsed, "pairs": {}}

    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        for tau in [0.01, 0.05]:
            eligible_cells = set()
            for depth, var, val in CELLS:
                key_a = (depth, var, val, surf_a, "identity_bare")
                key_b = (depth, var, val, surf_b, "identity_bare")
                if key_a in obs_idx and key_b in obs_idx:
                    d = tv(obs_idx[key_a], obs_idx[key_b])
                    if d <= tau:
                        eligible_cells.add((depth, var, val))

            tau_key = f"tau_{tau}"
            if pair_name not in summary["pairs"]:
                summary["pairs"][pair_name] = {}
            if tau_key not in summary["pairs"][pair_name]:
                summary["pairs"][pair_name][tau_key] = {}

            print(f"\n--- {pair_name} at tau={tau} "
                  f"({len(eligible_cells)} eligible) ---")

            pair_omega = {}
            for cont_name in CONTINUATIONS:
                tvs = []
                for cell in eligible_cells:
                    depth, var, val = cell
                    key_a = (depth, var, val, surf_a, cont_name)
                    key_b = (depth, var, val, surf_b, cont_name)
                    if key_a in obs_idx and key_b in obs_idx:
                        tvs.append(tv(obs_idx[key_a], obs_idx[key_b]))
                if tvs:
                    omega = float(np.max(tvs))
                    mean = float(np.mean(tvs))
                    status = "PRESERVE" if omega <= tau else "AMPLIFY"
                    pair_omega[cont_name] = {
                        "omega": omega, "mean": mean, "status": status,
                        "n_cells": len(tvs),
                    }

            pair_omega_sorted = sorted(
                pair_omega.items(), key=lambda x: x[1]["omega"], reverse=True)
            for cn, cr in pair_omega_sorted:
                print(f"  {cn:22s}: omega={cr['omega']:.5f} "
                      f"mean={cr['mean']:.5f} {cr['status']}")
            summary["pairs"][pair_name][tau_key] = pair_omega

    # Compute framing ratios
    print("\n=== Framing Ratios (framed/bare) ===")
    framing_comparisons = [
        ("identity_doc", "identity_bare", "doc_frame"),
        ("identity_spec", "identity_bare", "spec_frame"),
        ("identity_todo", "identity_bare", "todo_frame"),
        ("heavy_doc", "heavy_bare", "heavy_doc_frame"),
        ("neutral_doc", "neutral_bare", "neutral_doc_frame"),
    ]

    summary["framing_ratios"] = {}
    for pair_name in [p[2] for p in EQUIV_PAIRS]:
        print(f"\n--- {pair_name} ---")
        ratios = {}
        for tau in [0.01, 0.05]:
            tau_key = f"tau_{tau}"
            pair_data = summary["pairs"].get(pair_name, {}).get(tau_key, {})
            if not pair_data:
                continue
            print(f"  tau={tau}:")
            for framed, bare, label in framing_comparisons:
                if framed in pair_data and bare in pair_data:
                    bare_omega = pair_data[bare]["omega"]
                    framed_omega = pair_data[framed]["omega"]
                    if bare_omega > 0.001:
                        ratio = framed_omega / bare_omega
                    else:
                        ratio = None
                    ratios[f"{label}_tau{tau}"] = {
                        "bare_omega": bare_omega,
                        "framed_omega": framed_omega,
                        "ratio": ratio,
                    }
                    ratio_str = f"{ratio:.3f}" if ratio is not None else "N/A"
                    print(f"    {label:22s}: bare={bare_omega:.5f} "
                          f"framed={framed_omega:.5f} ratio={ratio_str}")
        summary["framing_ratios"][pair_name] = ratios

    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {result_dir}/")
    return observations, summary


if __name__ == "__main__":
    run_authority_experiment()
