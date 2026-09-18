"""Query Format Transfer: Does omega from one query predict another?

The query sensitivity experiment showed only print_output and assert_eq
produce eligible cells. This tests: compute omega under print_output,
predict preservation under assert_eq (and vice versa).

If omega transfers across query formats, it captures a state property
independent of the probing method — a strong universality claim.

6 cells x 2 queries x 2 pairs x 2 sides x 6 continuations = 288 readouts
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
    "assert_eq": "f()\nassert {var} == ",
}

SURFACE_PAIRS = [
    ("# State unchanged.\n", "# {var} stays the same.\n", "strong_assert_2"),
    ("# ok\n", "# okay\n", "terse_ok_okay"),
]

CONTINUATIONS = {
    "none": "",
    "neutral": "# State unchanged.\n",
    "heavy": "# Reassigning {var} now.\n",
    "identity": "{var} = {var}\n",
    "type_check": "type({var})\n",
    "pass": "pass\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5), (4, "x", 7),
    (4, "y", 2), (4, "y", 6), (4, "y", 8),
]

HEAVY_KW = ["heavy", "reassign", "updat", "chang"]


def run_query_transfer():
    print("=== Query Format Transfer Test ===")
    n_total = len(CELLS) * len(QUERY_FORMATS) * len(SURFACE_PAIRS) * 2 * len(CONTINUATIONS)
    print(f"Cells: {len(CELLS)}, Queries: {len(QUERY_FORMATS)}, "
          f"Pairs: {len(SURFACE_PAIRS)}, Continuations: {len(CONTINUATIONS)}")
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

        for q_name, q_tmpl in QUERY_FORMATS.items():
            query = q_tmpl.replace("{var}", var)

            for surf_a_tmpl, surf_b_tmpl, pair_name in SURFACE_PAIRS:
                for side, surf_tmpl in [("a", surf_a_tmpl), ("b", surf_b_tmpl)]:
                    surface_text = surf_tmpl.replace("{var}", var)

                    for cont_name, cont_tmpl in CONTINUATIONS.items():
                        cont_text = cont_tmpl.replace("{var}", var)
                        full_suffix = surface_text + cont_text + query

                        dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                        n += 1

                        observations.append({
                            "depth": depth, "var": var, "val": val,
                            "query": q_name,
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

    result_dir = Path("experiments/results/omega_query_transfer")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    obs_idx = {}
    for obs in observations:
        key = (obs["var"], obs["val"], obs["query"], obs["pair"],
               obs["side"], obs["continuation"])
        obs_idx[key] = np.array(obs["dist"], dtype=np.float64)

    q_names = list(QUERY_FORMATS.keys())

    for tau in [0.01, 0.02, 0.05]:
        print(f"\n=== Query Transfer (tau={tau}) ===")

        for train_q in q_names:
            eval_q = [q for q in q_names if q != train_q][0]

            train_omega = {}
            for pair_name in [p[2] for p in SURFACE_PAIRS]:
                for cont_name in CONTINUATIONS:
                    tvs = []
                    for _, var, val in CELLS:
                        ka_base = (var, val, train_q, pair_name, "a", "none")
                        kb_base = (var, val, train_q, pair_name, "b", "none")
                        if ka_base not in obs_idx or kb_base not in obs_idx:
                            continue
                        d_pre = 0.5 * np.abs(obs_idx[ka_base] - obs_idx[kb_base]).sum()
                        if d_pre > tau:
                            continue
                        ka = (var, val, train_q, pair_name, "a", cont_name)
                        kb = (var, val, train_q, pair_name, "b", cont_name)
                        if ka in obs_idx and kb in obs_idx:
                            d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                            tvs.append(d)
                    if tvs:
                        key = (pair_name, cont_name)
                        train_omega[key] = max(
                            train_omega.get(key, 0), max(tvs))

            omega_correct = 0
            baseline_correct = 0
            total = 0

            for pair_name in [p[2] for p in SURFACE_PAIRS]:
                for cont_name in CONTINUATIONS:
                    key = (pair_name, cont_name)
                    if key not in train_omega:
                        continue

                    eval_tvs = []
                    for _, var, val in CELLS:
                        ka_base = (var, val, eval_q, pair_name, "a", "none")
                        kb_base = (var, val, eval_q, pair_name, "b", "none")
                        if ka_base not in obs_idx or kb_base not in obs_idx:
                            continue
                        d_pre = 0.5 * np.abs(obs_idx[ka_base] - obs_idx[kb_base]).sum()
                        if d_pre > tau:
                            continue
                        ka = (var, val, eval_q, pair_name, "a", cont_name)
                        kb = (var, val, eval_q, pair_name, "b", cont_name)
                        if ka in obs_idx and kb in obs_idx:
                            d = float(0.5 * np.abs(obs_idx[ka] - obs_idx[kb]).sum())
                            eval_tvs.append(d)

                    if not eval_tvs:
                        continue

                    actual = "preserve" if max(eval_tvs) <= tau else "separate"
                    omega_pred = "preserve" if train_omega[key] <= tau else "separate"

                    cn_lower = cont_name.lower()
                    is_heavy = any(k in cn_lower for k in HEAVY_KW)
                    baseline_pred = "separate" if is_heavy else "preserve"

                    total += 1
                    omega_correct += int(omega_pred == actual)
                    baseline_correct += int(baseline_pred == actual)

                    if omega_pred != baseline_pred:
                        marker = "omega" if omega_pred == actual else "baseline"
                        print(f"  train={train_q:15s} eval={eval_q:15s} "
                              f"{pair_name:20s} {cont_name:12s} "
                              f"omega={omega_pred:8s} base={baseline_pred:8s} "
                              f"actual={actual:8s} [{marker}]")

            if total > 0:
                print(f"  train={train_q} -> eval={eval_q}: "
                      f"omega={omega_correct}/{total} ({100*omega_correct/total:.0f}%), "
                      f"baseline={baseline_correct}/{total} ({100*baseline_correct/total:.0f}%)")

    print(f"\nSaved {n} observations to {result_dir}/")
    del adapter
    return observations


if __name__ == "__main__":
    run_query_transfer()
