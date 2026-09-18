"""Omega Query Sensitivity: Does the readout question format affect the omega effect?

Tests different query formats to see if omega is robust to how we ask.
Key question: Is the amplification an artifact of a specific query, or a property
of the state?

Fast: 6 cells × 4 surfaces × 3 conts × 8 queries = 576 readouts (~3 min).
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

QUERY_FORMATS = {
    "print_output": "f()\nprint({var})  # Output: ",
    "print_bare": "f()\nprint({var})\n# ",
    "result_eq": "result = f()\n# result = ",
    "assert_eq": "f()\nassert {var} == ",
    "return_val": "answer = f()\n# The function returned {var} = ",
    "inline": "f(); print({var})  # ",
    "docstring": 'f()\n"""{var} is now: ',
    "comment_val": "f()\n# {var} is ",
}

SURFACES = {
    "assert_novar": "# State unchanged.\n",
    "assert_var": "# {var} stays the same.\n",
    "mislead_var": "# Reassigning {var} now.\n",
    "none": "",
}

CONTINUATIONS = {
    "none": "",
    "neutral": "# State unchanged.\n",
    "heavy": "# Reassigning {var} now.\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]


def run_query_sweep():
    print("=== Omega Query Format Sensitivity ===")
    n_total = len(CELLS) * len(SURFACES) * len(CONTINUATIONS) * len(QUERY_FORMATS)
    print(f"Cells: {len(CELLS)}, Surfaces: {len(SURFACES)}, "
          f"Continuations: {len(CONTINUATIONS)}, Queries: {len(QUERY_FORMATS)}")
    print(f"Total readouts: {n_total}")
    print()

    adapter = ModelAdapter(MODEL_CFG)
    observations = []
    t0 = time.time()
    n = 0

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for qname, qtmpl in QUERY_FORMATS.items():
            query = qtmpl.replace("{var}", var)

            for surf_name, surf_tmpl in SURFACES.items():
                surface_text = surf_tmpl.replace("{var}", var)

                for cont_name, cont_tmpl in CONTINUATIONS.items():
                    cont_text = cont_tmpl.replace("{var}", var)
                    full_suffix = surface_text + cont_text + query

                    dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                    n += 1

                    observations.append({
                        "depth": depth, "var": var, "val": val,
                        "query": qname, "surface": surf_name,
                        "continuation": cont_name,
                        "dist": dist.tolist(),
                    })

                    if n % 100 == 0:
                        elapsed = time.time() - t0
                        print(f"  {n}/{n_total} readouts, {elapsed:.0f}s, "
                              f"{n/elapsed:.1f}/s", flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {n} readouts in {elapsed:.1f}s ({n/elapsed:.1f}/s)")

    result_dir = Path("experiments/results/omega_queries")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    obs_idx = {}
    for obs in observations:
        key = (obs["var"], obs["val"], obs["query"],
               obs["surface"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    tau = 0.01
    pair_a, pair_b = "assert_novar", "assert_var"

    print(f"\n=== Query × Continuation Omega (tau={tau}) ===\n")
    print(f"{'Query':20s} {'none_tv':>10s} {'neutral_tv':>10s} {'heavy_tv':>10s} "
          f"{'n_elig':>7s} {'neutral':>10s} {'heavy':>10s}")

    query_stats = {}
    for qname in sorted(QUERY_FORMATS.keys()):
        for cont_name in sorted(CONTINUATIONS.keys()):
            tvs = []
            for depth, var, val in CELLS:
                ka_none = (var, val, qname, pair_a, "none")
                kb_none = (var, val, qname, pair_b, "none")
                if ka_none not in obs_idx or kb_none not in obs_idx:
                    continue
                d_pre = 0.5 * np.abs(obs_idx[ka_none] - obs_idx[kb_none]).sum()
                if d_pre > tau:
                    continue

                ka = (var, val, qname, pair_a, cont_name)
                kb = (var, val, qname, pair_b, cont_name)
                if ka in obs_idx and kb in obs_idx:
                    d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                    tvs.append(d)

            query_stats.setdefault(qname, {})[cont_name] = {
                "mean": float(np.mean(tvs)) if tvs else None,
                "max": float(np.max(tvs)) if tvs else None,
                "n": len(tvs),
            }

        qs = query_stats[qname]
        none_mean = f"{qs['none']['mean']:.6f}" if qs['none']['mean'] is not None else "n/a"
        neut_mean = f"{qs['neutral']['mean']:.6f}" if qs['neutral']['mean'] is not None else "n/a"
        heavy_mean = f"{qs['heavy']['mean']:.6f}" if qs['heavy']['mean'] is not None else "n/a"
        n_elig = qs['none']['n']
        neut_status = "PRESERVE" if qs['neutral']['max'] is not None and qs['neutral']['max'] <= tau else "AMPLIFY" if qs['neutral']['max'] is not None else "n/a"
        heavy_status = "PRESERVE" if qs['heavy']['max'] is not None and qs['heavy']['max'] <= tau else "AMPLIFY" if qs['heavy']['max'] is not None else "n/a"
        print(f"{qname:20s} {none_mean:>10s} {neut_mean:>10s} {heavy_mean:>10s} "
              f"{n_elig:7d} {neut_status:>10s} {heavy_status:>10s}")

    with open(result_dir / "query_stats.json", "w") as f:
        json.dump(query_stats, f, indent=2)

    print(f"\nSaved {n} observations to {result_dir}/")
    del adapter
    return observations


if __name__ == "__main__":
    run_query_sweep()
