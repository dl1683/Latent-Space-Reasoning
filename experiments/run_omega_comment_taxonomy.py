"""Omega Comment Taxonomy (Experiment #44).

Tests a systematic taxonomy of code comments for their omega effect.
Exp #43 showed doc-frame comments dramatically change omega. This
experiment maps which TYPES of comments amplify vs suppress, organized
by semantic category:

Categories:
  1. Authority: "According to the documentation:", "The API spec states:"
  2. Safety: "# SAFETY: do not modify", "# WARNING: critical value"
  3. Deprecation: "# DEPRECATED:", "# TODO: remove"
  4. Certainty: "# Verified correct", "# This is guaranteed"
  5. Uncertainty: "# Not sure about this", "# Needs review"
  6. Explanation: "# This assigns the value", "# Setting variable"
  7. Control: bare identity (no comment), pass (neutral)

Each comment precedes the same identity continuation (x = x\n).
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
    "minimal_ok": "# OK.\n",
    "minimal_good": "# Good.\n",
}

EQUIV_PAIRS = [
    ("assert_novar_2", "assert_var_2", "strong_assert_2"),
    ("minimal_ok", "minimal_good", "minimal_ok_good"),
]

CONTINUATIONS = {
    # Controls
    "identity_bare": "{var} = {var}\n",
    "neutral_bare": "pass\n",
    "heavy_bare": "{var} = {var} + 1\n",
    # Authority
    "auth_doc": "# According to the documentation:\n{var} = {var}\n",
    "auth_spec": "# The API spec states:\n{var} = {var}\n",
    "auth_official": "# Per official guidelines:\n{var} = {var}\n",
    # Safety
    "safety_donot": "# SAFETY: do not modify\n{var} = {var}\n",
    "safety_warning": "# WARNING: critical value\n{var} = {var}\n",
    "safety_critical": "# CRITICAL: preserve this\n{var} = {var}\n",
    # Deprecation
    "depr_deprecated": "# DEPRECATED: will be removed\n{var} = {var}\n",
    "depr_todo": "# TODO: remove this\n{var} = {var}\n",
    "depr_legacy": "# Legacy code, do not extend\n{var} = {var}\n",
    # Certainty
    "cert_verified": "# Verified correct\n{var} = {var}\n",
    "cert_guaranteed": "# This is guaranteed\n{var} = {var}\n",
    "cert_proven": "# Proven by tests\n{var} = {var}\n",
    # Uncertainty
    "uncert_notsure": "# Not sure about this\n{var} = {var}\n",
    "uncert_review": "# Needs review\n{var} = {var}\n",
    "uncert_fixme": "# FIXME: might be wrong\n{var} = {var}\n",
    # Explanation
    "explain_assigns": "# This assigns the value\n{var} = {var}\n",
    "explain_setting": "# Setting variable\n{var} = {var}\n",
    "explain_identity": "# Identity operation\n{var} = {var}\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

CATEGORIES = {
    "control": ["identity_bare", "neutral_bare", "heavy_bare"],
    "authority": ["auth_doc", "auth_spec", "auth_official"],
    "safety": ["safety_donot", "safety_warning", "safety_critical"],
    "deprecation": ["depr_deprecated", "depr_todo", "depr_legacy"],
    "certainty": ["cert_verified", "cert_guaranteed", "cert_proven"],
    "uncertainty": ["uncert_notsure", "uncert_review", "uncert_fixme"],
    "explanation": ["explain_assigns", "explain_setting", "explain_identity"],
}


def run_comment_taxonomy():
    print("=== Omega Comment Taxonomy (Exp #44) ===")
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

                if n_readouts % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"  {n_readouts}/{n_target} readouts, {elapsed:.0f}s",
                          flush=True)

    elapsed = time.time() - t0
    print(f"Done: {n_readouts} readouts in {elapsed:.1f}s")

    result_dir = Path("experiments/results/omega_comment_taxonomy")
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    summary = {"n_readouts": n_readouts, "elapsed_s": elapsed,
               "pairs": {}, "category_summary": {}}

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
                    }

            pair_omega_sorted = sorted(
                pair_omega.items(), key=lambda x: x[1]["omega"], reverse=True)
            for cn, cr in pair_omega_sorted:
                print(f"  {cn:22s}: omega={cr['omega']:.5f} "
                      f"mean={cr['mean']:.5f} {cr['status']}")
            summary["pairs"][pair_name][tau_key] = pair_omega

            # Category averages
            cat_avgs = {}
            for cat_name, members in CATEGORIES.items():
                omegas = [pair_omega[m]["omega"] for m in members
                          if m in pair_omega]
                if omegas:
                    cat_avgs[cat_name] = {
                        "mean_omega": float(np.mean(omegas)),
                        "max_omega": float(np.max(omegas)),
                        "min_omega": float(np.min(omegas)),
                    }
            summary["category_summary"][f"{pair_name}_{tau_key}"] = cat_avgs

            print(f"\n  Category averages:")
            for cat_name, ca in sorted(cat_avgs.items(),
                                       key=lambda x: x[1]["mean_omega"],
                                       reverse=True):
                print(f"    {cat_name:15s}: mean={ca['mean_omega']:.5f} "
                      f"range=[{ca['min_omega']:.5f}, {ca['max_omega']:.5f}]")

    # Framing ratios vs bare identity
    print("\n=== Framing Ratios vs identity_bare ===")
    summary["framing_ratios"] = {}
    for pair_name in [p[2] for p in EQUIV_PAIRS]:
        for tau in [0.01, 0.05]:
            tau_key = f"tau_{tau}"
            pair_data = summary["pairs"].get(pair_name, {}).get(tau_key, {})
            if not pair_data or "identity_bare" not in pair_data:
                continue
            bare_omega = pair_data["identity_bare"]["omega"]
            if bare_omega < 0.0001:
                continue
            ratios = {}
            for cont_name, cd in pair_data.items():
                if cont_name == "identity_bare":
                    continue
                ratios[cont_name] = cd["omega"] / bare_omega
            summary["framing_ratios"][f"{pair_name}_{tau_key}"] = ratios
            print(f"\n{pair_name} tau={tau} (bare={bare_omega:.5f}):")
            for cn, r in sorted(ratios.items(), key=lambda x: x[1],
                                reverse=True):
                print(f"  {cn:22s}: {r:.3f}x")

    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {result_dir}/")
    return observations, summary


if __name__ == "__main__":
    run_comment_taxonomy()
