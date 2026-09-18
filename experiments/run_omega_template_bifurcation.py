"""Omega Template Bifurcation (Experiment #62).

Is the autoregressive bimodality specific to the nested-99 depth-4 template?
Tests bifurcation on 3 different code structures to see if the grow/collapse
pattern is template-dependent or general.

Addresses signal s194.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv

import torch
import copy


TEMPLATES = {
    "depth4_nested99": {
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "query_template": "f()\nprint({var})  # Output: ",
        "code": (
            "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
            "    def g():\n        {var} = 999\n        def h():\n"
            "            {var} = 9999\n            def k():\n"
            "                {var} = 99999\n                return {var}\n"
            "            k()\n            return {var}\n"
            "        h()\n        return {var}\n"
            "    g()\n    return {var}\n"
        ),
    },
    "simple_if_else": {
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "query_template": "result = check({var})\nprint(result)  # Output: ",
        "code": (
            "{var} = {outer_val}\n"
            "def check(val):\n"
            "    if val > 3:\n"
            "        return val * 2\n"
            "    else:\n"
            "        return val + 10\n"
        ),
    },
    "list_comprehension": {
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "query_template": "print(data[0])  # Output: ",
        "code": (
            "{var} = {outer_val}\n"
            "data = [{var}, {var}+1, {var}+2]\n"
            "data = [d * 2 for d in data]\n"
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
    ("assert_var_2", "assert_novar_2", "strong_assert_2"),
    ("minimal_ok", "minimal_good", "minimal_ok_good"),
]

CONTINUATIONS = {
    "empty": "",
    "identity": "{var} = {var}\n",
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
}

CELLS = [
    ("x", 1), ("x", 5),
    ("y", 2), ("y", 6),
]

N_GENERATE = 20
DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def generate_trajectory(adapter, prefix_state, suffix_text, n_tokens):
    ids = adapter.tok.encode(suffix_text, add_special_tokens=False,
                              return_tensors="pt")
    st = copy.deepcopy(prefix_state)
    with torch.no_grad():
        out = adapter._forward_with_cache(ids, st)
    st = adapter._get_cache(out)

    trajectory = []
    logits = out.logits[0, -1, :]
    dist = adapter._extract_11bin(logits)
    trajectory.append(dist)

    for step in range(1, n_tokens + 1):
        next_token_id = int(torch.argmax(logits))
        next_ids = torch.tensor([[next_token_id]])
        with torch.no_grad():
            out = adapter._forward_with_cache(next_ids, st)
        st = adapter._get_cache(out)
        logits = out.logits[0, -1, :]
        dist = adapter._extract_11bin(logits)
        trajectory.append(dist)

    adapter.call_count += 1
    return trajectory


def run_template_bifurcation():
    print("=== Template Bifurcation ===")

    all_records = []
    template_summaries = {}

    for tmpl_name, tmpl_cfg in TEMPLATES.items():
        print(f"\n--- Template: {tmpl_name} ---")
        cfg = {
            "model_id": tmpl_cfg["model_id"],
            "query_template": tmpl_cfg["query_template"],
            "templates": {"main": tmpl_cfg["code"]},
        }
        adapter = ModelAdapter(cfg)
        query_tmpl = tmpl_cfg["query_template"]
        t0 = time.time()

        for var, val in CELLS:
            prefix_text = build_prefix(tmpl_cfg["code"], var=var, outer_val=val)
            query = query_tmpl.replace("{var}", var)
            prefix_state = adapter.get_state_after_prefix(prefix_text)

            for surf_a, surf_b, pair_name in EQUIV_PAIRS:
                for cont_name, cont_tmpl in CONTINUATIONS.items():
                    cont_text = cont_tmpl.replace("{var}", var)
                    suffix_a = SURFACES[surf_a].replace("{var}", var) + cont_text + query
                    suffix_b = SURFACES[surf_b].replace("{var}", var) + cont_text + query

                    traj_a = generate_trajectory(adapter, prefix_state, suffix_a, N_GENERATE)
                    traj_b = generate_trajectory(adapter, prefix_state, suffix_b, N_GENERATE)

                    step_tvs = [float(tv(traj_a[s], traj_b[s]))
                                for s in range(N_GENERATE + 1)]

                    final_tv = step_tvs[-1]
                    fate = "grow" if final_tv > 0.5 else "collapse"

                    all_records.append({
                        "template": tmpl_name,
                        "cell": f"{var}_{val}",
                        "pair": pair_name,
                        "continuation": cont_name,
                        "initial_tv": step_tvs[0],
                        "final_tv": final_tv,
                        "max_tv": max(step_tvs),
                        "fate": fate,
                    })

                    print(f"  {var}_{val}/{pair_name}/{cont_name}: "
                          f"t0={step_tvs[0]:.4f} -> t{N_GENERATE}={final_tv:.4f} "
                          f"({fate})", flush=True)

        elapsed = time.time() - t0
        recs = [r for r in all_records if r["template"] == tmpl_name]
        n_grow = sum(1 for r in recs if r["fate"] == "grow")
        template_summaries[tmpl_name] = {
            "n": len(recs),
            "n_grow": n_grow,
            "grow_rate": round(n_grow / len(recs), 3) if recs else 0,
            "mean_final_tv": round(float(np.mean([r["final_tv"] for r in recs])), 4),
            "time_s": round(elapsed, 1),
        }

        import gc
        del adapter
        gc.collect()

    # Compare templates
    summary = {
        "n_records": len(all_records),
        "n_generate": N_GENERATE,
        "templates": list(TEMPLATES.keys()),
        "template_summaries": template_summaries,
        "records": all_records,
    }

    result_dir = Path("experiments/results/omega_template_bifurcation")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== TEMPLATE BIFURCATION RESULTS ===")
    print(f"{'Template':25s} {'N':>4s} {'Grow':>5s} {'Rate':>6s} {'MeanFinal':>10s}")
    for t, s in template_summaries.items():
        print(f"{t:25s} {s['n']:4d} {s['n_grow']:5d} "
              f"{s['grow_rate']:6.1%} {s['mean_final_tv']:10.4f}")

    # Is bimodality general?
    all_have_both = all(
        s["n_grow"] > 0 and s["n_grow"] < s["n"]
        for s in template_summaries.values()
    )
    if all_have_both:
        print("\n=> BIMODALITY IS GENERAL across templates")
    else:
        uniform = [t for t, s in template_summaries.items()
                   if s["n_grow"] == 0 or s["n_grow"] == s["n"]]
        print(f"\n=> Templates with uniform fate: {uniform}")
        print("=> Bimodality may be template-specific")


if __name__ == "__main__":
    run_template_bifurcation()
