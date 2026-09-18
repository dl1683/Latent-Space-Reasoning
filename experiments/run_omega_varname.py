"""Omega Variable Name Sensitivity (Experiment #52).

All prior experiments use x/y/z as variable names. Are omega findings
invariant to the token identity of the variable? If omega changes with
variable name, there's a token-embedding-level effect beyond code structure.

Tests: x, y, z (standard) + data, result, val, foo, _tmp, count
Same depth4 template, same continuations, same surface pairs.
~270 readouts (9 vars x 5 continuations x 2 surfaces x 3 value slots).
"""

import json
import sys
import time
import copy
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv


CFG = {
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

VARNAMES = ["x", "y", "z", "data", "result", "val", "foo", "_tmp", "count"]

SURFACES = {
    "assert_novar_2": "# State unchanged.\n",
    "assert_var_2": "# {var} stays the same.\n",
}

EQUIV_PAIRS = [
    ("assert_novar_2", "assert_var_2", "strong_assert_2"),
]

CONTINUATIONS = {
    "identity_bare": "{var} = {var}\n",
    "heavy_bare": "{var} = {var} + 1\n",
    "neutral_bare": "pass\n",
    "safety_donot": "# SAFETY: do not modify\n{var} = {var}\n",
    "depr_deprecated": "# DEPRECATED: will be removed\n{var} = {var}\n",
}

OUTER_VALS = [1, 5, 3]


def run_varname():
    print("=== Omega Variable Name Sensitivity (Exp #52) ===\n")

    adapter = ModelAdapter(CFG)
    query_tmpl = CFG["query_template"]
    observations = []
    t0 = time.time()
    n = 0

    for var in VARNAMES:
        for val in OUTER_VALS:
            tmpl = CFG["templates"]["depth4_single"]
            prefix_text = build_prefix(tmpl, var=var, outer_val=val)
            query = query_tmpl.replace("{var}", var)
            prefix_state = adapter.get_state_after_prefix(prefix_text)

            for surf_name, surf_tmpl in SURFACES.items():
                surface_text = surf_tmpl.replace("{var}", var)

                for cont_name, cont_tmpl in CONTINUATIONS.items():
                    cont_text = cont_tmpl.replace("{var}", var)
                    full_suffix = surface_text + cont_text + query
                    dist = adapter.get_dist_from_state(prefix_state,
                                                       full_suffix)
                    n += 1
                    observations.append({
                        "var": var,
                        "val": val,
                        "surface": surf_name,
                        "continuation": cont_name,
                        "dist": dist.tolist(),
                    })

                    if n % 50 == 0:
                        elapsed = time.time() - t0
                        print(f"  {n} readouts, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"Done: {n} readouts in {elapsed:.1f}s")

    result_dir = Path("experiments/results/omega_varname")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    # Analysis
    obs_idx = {}
    for obs in observations:
        key = (obs["var"], obs["val"], obs["surface"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    summary = {"n_readouts": n, "elapsed_s": elapsed, "by_var": {}}

    print("\n=== Omega by Variable Name ===")
    for var in VARNAMES:
        for tau in [0.01, 0.05]:
            eligible_vals = set()
            for val in OUTER_VALS:
                ka = (var, val, "assert_novar_2", "identity_bare")
                kb = (var, val, "assert_var_2", "identity_bare")
                if ka in obs_idx and kb in obs_idx:
                    d = tv(obs_idx[ka], obs_idx[kb])
                    if d <= tau:
                        eligible_vals.add(val)

            if not eligible_vals:
                continue

            print(f"\n  var={var} tau={tau} ({len(eligible_vals)} eligible):")
            var_omega = {}
            for cont in CONTINUATIONS:
                tvs = []
                for val in eligible_vals:
                    ka = (var, val, "assert_novar_2", cont)
                    kb = (var, val, "assert_var_2", cont)
                    if ka in obs_idx and kb in obs_idx:
                        tvs.append(tv(obs_idx[ka], obs_idx[kb]))
                if tvs:
                    omega = float(np.max(tvs))
                    mean = float(np.mean(tvs))
                    var_omega[cont] = {"omega": omega, "mean": mean}
                    print(f"    {cont:22s}: omega={omega:.5f}")

            summary["by_var"][f"{var}_tau{tau}"] = var_omega

    # Cross-variable comparison
    print("\n=== Cross-Variable heavy_bare Omega (tau=0.05) ===")
    heavy_omegas = []
    for var in VARNAMES:
        eligible_vals = set()
        for val in OUTER_VALS:
            ka = (var, val, "assert_novar_2", "identity_bare")
            kb = (var, val, "assert_var_2", "identity_bare")
            if ka in obs_idx and kb in obs_idx:
                d = tv(obs_idx[ka], obs_idx[kb])
                if d <= 0.05:
                    eligible_vals.add(val)
        tvs = []
        for val in eligible_vals:
            ka = (var, val, "assert_novar_2", "heavy_bare")
            kb = (var, val, "assert_var_2", "heavy_bare")
            if ka in obs_idx and kb in obs_idx:
                tvs.append(tv(obs_idx[ka], obs_idx[kb]))
        omega = float(np.max(tvs)) if tvs else None
        heavy_omegas.append((var, omega, len(eligible_vals)))
        status = f"omega={omega:.5f}" if omega else "no eligible cells"
        print(f"  {var:10s}: {status} ({len(eligible_vals)} elig)")

    valid = [(v, o) for v, o, _ in heavy_omegas if o is not None]
    if len(valid) >= 2:
        vals = [o for _, o in valid]
        cv = float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 0
        print(f"\n  CV across variables: {cv:.3f} "
              f"(mean={np.mean(vals):.5f}, std={np.std(vals):.5f})")
        summary["heavy_bare_cv"] = cv
        summary["heavy_bare_mean"] = float(np.mean(vals))

    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir}/")
    return summary


if __name__ == "__main__":
    run_varname()
