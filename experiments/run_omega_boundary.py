"""Omega Boundary-Preserving Syntax Crossover (Experiment #46).

Fable R3 Rank 1: Tests whether variable-reference amplification survives
proper prompt assembly. Exp #38 had a concatenation boundary bug where
continuations without trailing newlines got glued to the query
(x became xf(), x= became x=f(), x=x became x=xf()).

This experiment ensures every continuation arm produces valid Python
when followed by the query, and includes the glued form as a diagnostic.
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
    "assert_var_2": "# {var} stays the same.\n",
    "assert_novar_2": "# State unchanged.\n",
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

# All valid arms end with \n so they don't glue to the query.
# The glued_identity arm deliberately omits \n to reproduce the #38 bug.
CONTINUATIONS = {
    "empty":              "",
    "var_newline":        "{var}\n",
    "var_semicolon":      "{var};\n",
    "identity_newline":   "{var} = {var}\n",
    "identity_semicolon": "{var} = {var};\n",
    "glued_identity":     "{var} = {var}",
    "nonvar_expr":        "_ = 42\n",
    "pass_stmt":          "pass\n",
}

VALID_ARMS = [k for k in CONTINUATIONS if k != "glued_identity"]

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
]


def run_boundary_experiment():
    print("=== Omega Boundary-Preserving Syntax Crossover (Exp #46) ===")
    n_target = len(CELLS) * len(SURFACES) * len(CONTINUATIONS)
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

            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)

                # CRITICAL: ensure separator between continuation and query
                # For glued_identity, deliberately skip the separator
                if cont_name == "glued_identity":
                    full_suffix = surface_text + cont_text + query
                elif cont_text == "" or cont_text.endswith("\n"):
                    full_suffix = surface_text + cont_text + query
                else:
                    full_suffix = surface_text + cont_text + "\n" + query

                dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                n_readouts += 1

                observations.append({
                    "depth": depth,
                    "var": var,
                    "val": val,
                    "surface": surf_name,
                    "continuation": cont_name,
                    "dist": dist.tolist(),
                    "full_suffix_preview": full_suffix[:120],
                    "valid_syntax": cont_name != "glued_identity",
                })

                if n_readouts % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  {n_readouts}/{n_target} readouts, {elapsed:.0f}s",
                          flush=True)

    elapsed = time.time() - t0
    print(f"Done: {n_readouts} readouts in {elapsed:.1f}s")

    result_dir = Path("experiments/results/omega_boundary")
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    # --- Analysis ---
    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    print("\n=== Pair Analysis ===")
    summary = {"n_readouts": n_readouts, "elapsed_s": elapsed, "pairs": {}}

    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        print(f"\n--- {pair_name} ---")
        pair_result = {"continuations": {}}

        for cont_name in CONTINUATIONS:
            tvs = []
            for depth, var, val in CELLS:
                key_a = (depth, var, val, surf_a, cont_name)
                key_b = (depth, var, val, surf_b, cont_name)
                if key_a in obs_idx and key_b in obs_idx:
                    d = tv(obs_idx[key_a], obs_idx[key_b])
                    tvs.append(d)

            if tvs:
                pair_result["continuations"][cont_name] = {
                    "tvs": [float(v) for v in tvs],
                    "max": float(max(tvs)),
                    "mean": float(np.mean(tvs)),
                    "valid_syntax": cont_name != "glued_identity",
                }

        summary["pairs"][pair_name] = pair_result

    # Compute omega at tau=0.01 and tau=0.05 for each pair/continuation
    for tau in [0.01, 0.05]:
        tau_key = f"tau_{tau}"
        print(f"\n=== Omega at tau={tau} ===")

        for pair_name in summary["pairs"]:
            pair_data = summary["pairs"][pair_name]
            baseline_tvs = pair_data["continuations"]["empty"]["tvs"]
            n_eligible = sum(1 for v in baseline_tvs if v <= tau)

            if tau_key not in pair_data:
                pair_data[tau_key] = {"n_eligible": n_eligible}

            print(f"\n  {pair_name}: {n_eligible} eligible cells at tau={tau}")

            for cont_name in CONTINUATIONS:
                cont_data = pair_data["continuations"][cont_name]
                eligible_post_tvs = [
                    cont_data["tvs"][i]
                    for i in range(len(baseline_tvs))
                    if baseline_tvs[i] <= tau
                ]

                if eligible_post_tvs:
                    omega = max(eligible_post_tvs)
                    omega_mean = float(np.mean(eligible_post_tvs))
                else:
                    omega = None
                    omega_mean = None

                cont_data[tau_key] = {
                    "omega": omega,
                    "omega_mean": omega_mean,
                    "n_eligible": len(eligible_post_tvs),
                }

                tag = "" if cont_data["valid_syntax"] else " [GLUED/INVALID]"
                if omega is not None:
                    print(f"    {cont_name}: omega={omega:.4f} "
                          f"(mean={omega_mean:.4f}){tag}")
                else:
                    print(f"    {cont_name}: no eligible cells{tag}")

    # Key comparisons
    print("\n=== KEY COMPARISONS ===")
    comparisons = {}
    for pair_name in summary["pairs"]:
        pair_data = summary["pairs"][pair_name]
        for tau in [0.01, 0.05]:
            tau_key = f"tau_{tau}"
            n_eligible = pair_data[tau_key]["n_eligible"]
            if n_eligible == 0:
                continue

            print(f"\n{pair_name} at tau={tau} ({n_eligible} eligible):")
            comp_key = f"{pair_name}_{tau_key}"
            comparisons[comp_key] = {}

            # Valid arms only ranking
            valid_omegas = {}
            for cont_name in VALID_ARMS:
                o = pair_data["continuations"][cont_name][tau_key].get("omega")
                if o is not None:
                    valid_omegas[cont_name] = o

            ranked = sorted(valid_omegas.items(), key=lambda x: -x[1])
            print("  Valid arms ranking:")
            for i, (name, omega) in enumerate(ranked):
                print(f"    #{i+1}: {name} = {omega:.4f}")
            comparisons[comp_key]["valid_ranking"] = [
                {"arm": n, "omega": o} for n, o in ranked
            ]

            # Glued diagnostic
            glued = pair_data["continuations"]["glued_identity"][tau_key].get("omega")
            identity = pair_data["continuations"]["identity_newline"][tau_key].get("omega")
            if glued is not None and identity is not None:
                ratio = glued / identity if identity > 0 else float('inf')
                print(f"  Glued vs valid identity: {glued:.4f} / {identity:.4f} "
                      f"= {ratio:.2f}x")
                comparisons[comp_key]["glued_vs_valid"] = {
                    "glued": glued, "valid": identity, "ratio": ratio
                }

            # Var-specific vs non-var
            var_nl = valid_omegas.get("var_newline")
            nonvar = valid_omegas.get("nonvar_expr")
            if var_nl is not None and nonvar is not None:
                ratio = var_nl / nonvar if nonvar > 0 else float('inf')
                print(f"  Var-specific vs nonvar: {var_nl:.4f} / {nonvar:.4f} "
                      f"= {ratio:.2f}x")
                comparisons[comp_key]["var_vs_nonvar"] = {
                    "var": var_nl, "nonvar": nonvar, "ratio": ratio
                }

            # Identity vs pass
            identity_val = valid_omegas.get("identity_newline")
            pass_val = valid_omegas.get("pass_stmt")
            if identity_val is not None and pass_val is not None:
                ratio = identity_val / pass_val if pass_val > 0 else float('inf')
                print(f"  Identity vs pass: {identity_val:.4f} / {pass_val:.4f} "
                      f"= {ratio:.2f}x")

    summary["comparisons"] = comparisons

    # Verdict
    print("\n=== VERDICT ===")
    boundary_artifact = False
    valid_amplification = False

    for pair_name in summary["pairs"]:
        for tau in [0.01, 0.05]:
            tau_key = f"tau_{tau}"
            pair_data = summary["pairs"][pair_name]
            if pair_data[tau_key]["n_eligible"] == 0:
                continue

            glued_o = pair_data["continuations"]["glued_identity"][tau_key].get("omega")
            valid_id = pair_data["continuations"]["identity_newline"][tau_key].get("omega")
            empty_o = pair_data["continuations"]["empty"][tau_key].get("omega")

            if glued_o and valid_id and glued_o > valid_id * 1.5:
                boundary_artifact = True
            if valid_id and empty_o and valid_id > empty_o * 1.5:
                valid_amplification = True

    if boundary_artifact:
        print("CONFIRMED: Glued identity inflates omega (boundary artifact exists)")
    else:
        print("NOT CONFIRMED: Glued identity does NOT inflate omega significantly")

    if valid_amplification:
        print("CONFIRMED: Valid identity still amplifies omega vs baseline")
    else:
        print("NOT CONFIRMED: Valid identity does NOT amplify vs baseline")

    summary["verdict"] = {
        "boundary_artifact_confirmed": boundary_artifact,
        "valid_amplification_survives": valid_amplification,
    }

    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir}")


if __name__ == "__main__":
    run_boundary_experiment()
