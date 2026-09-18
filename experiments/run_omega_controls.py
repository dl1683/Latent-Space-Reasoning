"""Omega Controls: Adversarial and matched controls for the omega effect.

Tests:
1. Token-count matched: same length, different meaning
2. Semantic negation: "unchanged" vs "changed"
3. Language controls: comments in different "styles"
4. Whitespace variations: tabs, spaces, multiple newlines
5. Cross-variable: continuation mentions a different variable
6. Nonsense: random but syntactically valid comments
7. Code vs comment: executable vs non-executable with same meaning

Each test produces a matched pair for omega computation.
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

SURFACE_PAIRS = [
    ("# State unchanged.\n", "# {var} stays the same.\n", "strong_assert_2"),
    ("# Values preserved.\n", "# {var} retains its value.\n", "moderate_assert"),
]

CONTROL_CONTINUATIONS = {
    "neutral_original": "# No changes.\n",
    "heavy_original": "# Reassigning {var} now.\n",

    "matched_len_neutral": "# All is fine.\n",
    "matched_len_heavy_a": "# Now we change it.\n",
    "matched_len_heavy_b": "# Delete {var} now.\n",

    "negated_neutral": "# Changes made.\n",
    "negated_heavy": "# Not reassigning.\n",

    "formal_neutral": "# INVARIANT: {var} preserved.\n",
    "casual_neutral": "# {var} is ok lol.\n",
    "terse_neutral": "# ok\n",
    "verbose_neutral": "# The variable {var} has not been modified in any way.\n",

    "whitespace_1": " \n",
    "whitespace_2": "  \n",
    "whitespace_3": "\t\n",
    "whitespace_4": "\n\n",
    "whitespace_5": "\n\n\n",

    "cross_var_neutral_x": "# x is unchanged.\n",
    "cross_var_neutral_y": "# y is unchanged.\n",
    "cross_var_neutral_z": "# z is unchanged.\n",
    "cross_var_heavy_x": "# Reassigning x now.\n",
    "cross_var_heavy_y": "# Reassigning y now.\n",
    "cross_var_heavy_z": "# Reassigning z now.\n",

    "nonsense_1": "# Flurble gax.\n",
    "nonsense_2": "# Zzz qqq www.\n",
    "nonsense_3": "# 42 17 99.\n",

    "code_identity": "{var} = {var}\n",
    "code_noop": "pass\n",
    "code_type": "type({var})\n",
    "code_assert_true": "assert True\n",
    "code_assert_var": "assert {var} == {var}\n",
    "code_del_restore": "_{var}_tmp = {var}\n",

    "empty": "",
    "just_hash": "#\n",
    "hash_space": "# \n",
    "hash_var": "# {var}\n",
    "hash_number": "# 1\n",

    "multi_neutral_2": "# No changes.\n# No changes.\n",
    "multi_neutral_3": "# No changes.\n# No changes.\n# No changes.\n",
    "multi_heavy_2": "# Reassigning {var} now.\n# Reassigning {var} now.\n",
    "multi_mixed_nh": "# No changes.\n# Reassigning {var} now.\n",
    "multi_mixed_hn": "# Reassigning {var} now.\n# No changes.\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]


def run_controls():
    print("=== Omega Controls: Adversarial & Matched ===")
    n_total = len(CELLS) * len(SURFACE_PAIRS) * 2 * len(CONTROL_CONTINUATIONS)
    print(f"Cells: {len(CELLS)}, Pairs: {len(SURFACE_PAIRS)}, "
          f"Continuations: {len(CONTROL_CONTINUATIONS)}")
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

        for surf_a_tmpl, surf_b_tmpl, pair_name in SURFACE_PAIRS:
            for side, surf_tmpl in [("a", surf_a_tmpl), ("b", surf_b_tmpl)]:
                surface_text = surf_tmpl.replace("{var}", var)

                for cont_name, cont_tmpl in CONTROL_CONTINUATIONS.items():
                    cont_text = cont_tmpl.replace("{var}", var)
                    full_suffix = surface_text + cont_text + query

                    dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                    n += 1

                    observations.append({
                        "depth": depth, "var": var, "val": val,
                        "pair": pair_name, "side": side,
                        "continuation": cont_name,
                        "dist": dist.tolist(),
                    })

                    if n % 100 == 0:
                        elapsed = time.time() - t0
                        print(f"  {n}/{n_total} readouts, {elapsed:.0f}s, "
                              f"{n/elapsed:.1f}/s", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {n} readouts in {elapsed:.1f}s ({n/elapsed:.1f}/s)")

    result_dir = Path("experiments/results/omega_controls")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    obs_idx = {}
    for obs in observations:
        key = (obs["var"], obs["val"], obs["side"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    tau = 0.01
    print(f"\n=== Control Omega Table (tau={tau}) ===")
    print(f"{'Continuation':35s} {'mean_tv':>10s} {'max_tv':>10s} {'n_elig':>7s} {'status':>10s}")

    cont_stats = {}
    for cont_name in sorted(CONTROL_CONTINUATIONS.keys()):
        tvs = []
        for depth, var, val in CELLS:
            ka_none = (var, val, "a", "empty")
            kb_none = (var, val, "b", "empty")
            if ka_none not in obs_idx or kb_none not in obs_idx:
                continue
            d_pre = 0.5 * np.abs(obs_idx[ka_none] - obs_idx[kb_none]).sum()
            if d_pre > tau:
                continue

            ka = (var, val, "a", cont_name)
            kb = (var, val, "b", cont_name)
            if ka in obs_idx and kb in obs_idx:
                d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                tvs.append(d)

        if tvs:
            mean_tv = np.mean(tvs)
            max_tv = np.max(tvs)
            status = "PRESERVE" if max_tv <= tau else "AMPLIFY"
            print(f"{cont_name:35s} {mean_tv:10.6f} {max_tv:10.6f} {len(tvs):7d} {status:>10s}")
            cont_stats[cont_name] = {
                "mean_tv": float(mean_tv), "max_tv": float(max_tv),
                "n": len(tvs), "preserves": bool(max_tv <= tau),
            }
        else:
            print(f"{cont_name:35s}       (no eligible cells)")
            cont_stats[cont_name] = {"n": 0}

    with open(result_dir / "control_stats.json", "w") as f:
        json.dump(cont_stats, f, indent=2)

    categories = {
        "neutral_variants": ["neutral_original", "matched_len_neutral", "formal_neutral",
                             "casual_neutral", "terse_neutral", "verbose_neutral"],
        "heavy_variants": ["heavy_original", "matched_len_heavy_a", "matched_len_heavy_b"],
        "negated": ["negated_neutral", "negated_heavy"],
        "whitespace": ["whitespace_1", "whitespace_2", "whitespace_3",
                       "whitespace_4", "whitespace_5"],
        "cross_var": [k for k in CONTROL_CONTINUATIONS if k.startswith("cross_var")],
        "nonsense": ["nonsense_1", "nonsense_2", "nonsense_3"],
        "code": [k for k in CONTROL_CONTINUATIONS if k.startswith("code_")],
        "minimal": ["empty", "just_hash", "hash_space", "hash_var", "hash_number"],
        "multi": [k for k in CONTROL_CONTINUATIONS if k.startswith("multi_")],
    }

    print("\n=== Category Summary ===")
    for cat_name, members in categories.items():
        preserving = sum(1 for m in members
                         if m in cont_stats and cont_stats[m].get("preserves", False))
        amplifying = sum(1 for m in members
                         if m in cont_stats and cont_stats[m].get("n", 0) > 0
                         and not cont_stats[m].get("preserves", True))
        total = sum(1 for m in members
                    if m in cont_stats and cont_stats[m].get("n", 0) > 0)
        print(f"  {cat_name:20s}: {preserving}/{total} preserving, "
              f"{amplifying}/{total} amplifying")

    with open(result_dir / "categories.json", "w") as f:
        json.dump(categories, f, indent=2)

    print(f"\nSaved {n} observations to {result_dir}/")
    del adapter
    return observations


if __name__ == "__main__":
    run_controls()
