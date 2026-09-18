"""Omega Gradient: Does omega vary continuously with perturbation strength?

Tests a semantic gradient from fully-neutral to fully-heavy:
  Level 0: no continuation (empty)
  Level 1: neutral comment (generic)
  Level 2: neutral comment (mentions variable)
  Level 3: ambiguous comment
  Level 4: light hint at change
  Level 5: explicit mention of reassignment
  Level 6: strong reassignment claim
  Level 7: multiple reassignment claims
  Level 8: reassignment + code

Key question: Is omega a smooth function of perturbation strength,
or is there a sharp transition?

Fast: 6 cells × 2 pairs × 2 sides × 16 gradients = 384 readouts (~2 min)
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix


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

GRADIENT = {
    "g00_empty": "",
    "g01_newline": "\n",
    "g02_pass": "pass\n",
    "g03_hash": "#\n",
    "g04_comment_ok": "# OK.\n",
    "g05_verified": "# Verified.\n",
    "g06_state_unchanged": "# State unchanged.\n",
    "g07_var_unchanged": "# {var} is unchanged.\n",
    "g08_var_preserved": "# {var} retains its value.\n",
    "g09_ambiguous": "# {var} processing done.\n",
    "g10_hint_update": "# {var} may need updating.\n",
    "g11_update_mention": "# Updating {var}.\n",
    "g12_reassign": "# Reassigning {var} now.\n",
    "g13_strong_reassign": "# {var} has been reassigned.\n",
    "g14_double_reassign": "# Reassigning {var}.\n# {var} changed.\n",
    "g15_reassign_code": "# Reassigning {var}.\n{var} = {var}\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]


def run_gradient():
    print("=== Omega Gradient Sweep ===")
    n_total = len(CELLS) * len(SURFACE_PAIRS) * 2 * len(GRADIENT)
    print(f"Cells: {len(CELLS)}, Pairs: {len(SURFACE_PAIRS)}, "
          f"Gradient levels: {len(GRADIENT)}")
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

                for grad_name, grad_tmpl in GRADIENT.items():
                    grad_text = grad_tmpl.replace("{var}", var)
                    full_suffix = surface_text + grad_text + query

                    dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                    n += 1

                    observations.append({
                        "depth": depth, "var": var, "val": val,
                        "pair": pair_name, "side": side,
                        "gradient": grad_name,
                        "dist": dist.tolist(),
                    })

                    if n % 100 == 0:
                        elapsed = time.time() - t0
                        print(f"  {n}/{n_total} readouts, {elapsed:.0f}s, "
                              f"{n/elapsed:.1f}/s", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {n} readouts in {elapsed:.1f}s ({n/elapsed:.1f}/s)")

    result_dir = Path("experiments/results/omega_gradient")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    obs_idx = {}
    for obs in observations:
        key = (obs["var"], obs["val"], obs["pair"], obs["side"], obs["gradient"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    for tau in [0.01, 0.02, 0.05]:
        print(f"\n=== Gradient Omega (tau={tau}) ===")
        for pair_name in [p[2] for p in SURFACE_PAIRS]:
            grad_stats = {}
            for grad_name in GRADIENT:
                tvs = []
                for depth, var, val in CELLS:
                    ka_base = (var, val, pair_name, "a", "g00_empty")
                    kb_base = (var, val, pair_name, "b", "g00_empty")
                    if ka_base not in obs_idx or kb_base not in obs_idx:
                        continue
                    d_pre = 0.5 * np.abs(obs_idx[ka_base] - obs_idx[kb_base]).sum()
                    if d_pre > tau:
                        continue

                    ka = (var, val, pair_name, "a", grad_name)
                    kb = (var, val, pair_name, "b", grad_name)
                    if ka in obs_idx and kb in obs_idx:
                        d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                        tvs.append(d)

                if tvs:
                    grad_stats[grad_name] = {
                        "mean": float(np.mean(tvs)),
                        "max": float(np.max(tvs)),
                        "n": len(tvs),
                        "preserves": max(tvs) <= tau,
                    }

            if not grad_stats:
                print(f"  {pair_name}: no eligible cells")
                continue

            print(f"\n  {pair_name} ({list(grad_stats.values())[0]['n']} cells):")
            for gn in sorted(GRADIENT.keys()):
                if gn in grad_stats:
                    gs = grad_stats[gn]
                    status = "PRESERVE" if gs["preserves"] else "AMPLIFY"
                    bar = "|" + "#" * min(int(gs["mean"] * 200), 40)
                    print(f"    {gn:25s} mean={gs['mean']:.6f} max={gs['max']:.6f} "
                          f"{status:8s} {bar}")

            with open(result_dir / f"gradient_{pair_name}_tau{tau}.json", "w") as f:
                json.dump(grad_stats, f, indent=2)

    print(f"\nSaved {n} observations to {result_dir}/")
    del adapter
    return observations


if __name__ == "__main__":
    run_gradient()
