"""Omega Safety Comments on Instruct (Experiment #48).

Exp #44 found safety comments amplify omega 7.45x on base model.
Does this hold on instruct model? If so, it's a production-relevant
safety finding: the comments developers add to mark code as critical
actually make the instruct model MORE likely to diverge.

Tests: safety_donot, depr_deprecated, identity_bare, heavy_bare, neutral_bare
on both base and instruct Qwen3-1.7B. Focused subset of exp #44.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv


BASE_CFG = {
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

INSTRUCT_CFG = {
    "model_id": "Qwen/Qwen3-1.7B",
    "query_template": "f()\nprint({var})  # Output: ",
    "templates": BASE_CFG["templates"],
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
    "identity_bare": "{var} = {var}\n",
    "heavy_bare": "{var} = {var} + 1\n",
    "neutral_bare": "pass\n",
    "safety_donot": "# SAFETY: do not modify\n{var} = {var}\n",
    "safety_warning": "# WARNING: critical value\n{var} = {var}\n",
    "depr_deprecated": "# DEPRECATED: will be removed\n{var} = {var}\n",
    "depr_todo": "# TODO: remove this\n{var} = {var}\n",
    "cert_proven": "# Proven by tests\n{var} = {var}\n",
    "uncert_fixme": "# FIXME: might be wrong\n{var} = {var}\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]


def run_model(cfg, model_label):
    print(f"\n{'='*50}")
    print(f"  Model: {model_label} ({cfg['model_id']})")
    print(f"{'='*50}")

    adapter = ModelAdapter(cfg)
    query_tmpl = cfg["query_template"]
    observations = []
    t0 = time.time()
    n_readouts = 0
    n_target = len(CELLS) * len(SURFACES) * len(CONTINUATIONS)

    for depth, var, val in CELLS:
        tmpl = cfg["templates"]["depth4_single"]
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
                    "model": model_label,
                    "dist": dist.tolist(),
                })

                if n_readouts % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  {n_readouts}/{n_target}, {elapsed:.0f}s",
                          flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {n_readouts} readouts in {elapsed:.1f}s")
    del adapter
    import gc; gc.collect()
    return observations, elapsed


def analyze(observations, model_label):
    obs_idx = {}
    for obs in observations:
        key = (obs["depth"], obs["var"], obs["val"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    results = {}
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

            if not eligible_cells:
                continue

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
                    pair_omega[cont_name] = {
                        "omega": float(np.max(tvs)),
                        "mean": float(np.mean(tvs)),
                    }

            key = f"{pair_name}_tau{tau}"
            results[key] = {
                "n_eligible": len(eligible_cells),
                "omegas": pair_omega,
            }

            print(f"\n  {pair_name} tau={tau} "
                  f"({len(eligible_cells)} eligible, {model_label}):")
            for cn, cd in sorted(pair_omega.items(),
                                 key=lambda x: x[1]["omega"], reverse=True):
                print(f"    {cn:22s}: omega={cd['omega']:.5f}")

    return results


def run_safety_instruct():
    print("=== Omega Safety Comments on Instruct (Exp #48) ===")

    base_obs, base_time = run_model(BASE_CFG, "base")
    instruct_obs, instruct_time = run_model(INSTRUCT_CFG, "instruct")

    print("\n\n=== Analysis ===")
    base_results = analyze(base_obs, "base")
    instruct_results = analyze(instruct_obs, "instruct")

    print("\n\n=== Base vs Instruct Comparison ===")
    comparison = {}
    for key in base_results:
        if key not in instruct_results:
            continue
        br = base_results[key]["omegas"]
        ir = instruct_results[key]["omegas"]
        print(f"\n  {key}:")
        comp = {}
        for cont in CONTINUATIONS:
            if cont in br and cont in ir:
                ratio = ir[cont]["omega"] / br[cont]["omega"] if br[cont]["omega"] > 0.001 else None
                comp[cont] = {
                    "base": br[cont]["omega"],
                    "instruct": ir[cont]["omega"],
                    "ratio": ratio,
                }
                ratio_str = f"{ratio:.3f}" if ratio else "N/A"
                print(f"    {cont:22s}: base={br[cont]['omega']:.5f} "
                      f"instruct={ir[cont]['omega']:.5f} "
                      f"ratio={ratio_str}")
        comparison[key] = comp

    # Safety-specific analysis
    print("\n\n=== Safety Comment Effect ===")
    for key in comparison:
        print(f"\n  {key}:")
        comp = comparison[key]
        if "identity_bare" in comp and "safety_donot" in comp:
            base_safety_ratio = (comp["safety_donot"]["base"] /
                                 comp["identity_bare"]["base"]
                                 if comp["identity_bare"]["base"] > 0.001
                                 else None)
            inst_safety_ratio = (comp["safety_donot"]["instruct"] /
                                 comp["identity_bare"]["instruct"]
                                 if comp["identity_bare"]["instruct"] > 0.001
                                 else None)
            if base_safety_ratio:
                print(f"    Base safety_donot/identity: {base_safety_ratio:.3f}x")
            if inst_safety_ratio:
                print(f"    Instruct safety_donot/identity: {inst_safety_ratio:.3f}x")

    result_dir = Path("experiments/results/omega_safety_instruct")
    result_dir.mkdir(parents=True, exist_ok=True)

    all_obs = base_obs + instruct_obs
    with open(result_dir / "observations.json", "w") as f:
        json.dump(all_obs, f)

    summary = {
        "n_readouts": len(all_obs),
        "base_time": base_time,
        "instruct_time": instruct_time,
        "base_results": base_results,
        "instruct_results": instruct_results,
        "comparison": comparison,
    }
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {result_dir}/")
    return summary


if __name__ == "__main__":
    run_safety_instruct()
