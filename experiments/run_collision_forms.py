"""Experiment #72: Collision Form Generalization.

Exp #71 showed that `_ = val` (value-matching literal) amplifies 8.1x.
Is this about ASSIGNMENT PATH AMBIGUITY (two assignments to same value) or
about VALUE PRESENCE (any mention of the matching value)?

Tests multiple syntactic forms of value reference:
1. _ = val (assignment, from #71 — baseline collision)
2. print(val) (value read, no assignment)
3. assert var == val (explicit comparison, forces evaluation)
4. type(val) (value read, different context)
5. _ = other_val (mismatch control)
6. empty (no-insertion control)

If assignment-path: only forms 1 should amplify.
If value-presence: all matching forms (1-4) should amplify similarly.
If comparison-specific: form 3 should dominate.

Design: 6 forms x 3 cells x 2 pairs x 2 surfaces = 72 readouts. ~2 min.
"""
import gc
import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv

MODEL_CFG = {
    "model_id": "Qwen/Qwen3-1.7B-Base",
    "query_template": "f()\nprint({var})  # Output: ",
    "templates": {},
}

CELLS = [("x", 1), ("y", 2), ("y", 6)]

SURFACES = {
    "assert_var": "# {var} stays the same.\n",
    "assert_novar": "# State unchanged.\n",
    "boundary_1": "\n",
    "boundary_2": "pass\n",
}

EQUIV_PAIRS = [
    ("assert_var", "assert_novar", "strong_assert"),
    ("boundary_1", "boundary_2", "boundary_pair"),
]

PREFIX_TEMPLATE = (
    "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
    "    def g():\n        {var} = 999\n        def h():\n"
    "            {var} = 9999\n            return {var}\n"
    "        h()\n        return {var}\n"
    "    g()\n    return {var}\n"
)

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def get_forms(var, val):
    mismatch_val = [v for v in [1, 2, 3, 5, 6, 7] if v != val][0]
    return {
        "empty": "",
        "assign_match": f"_ = {val}\n",
        "assign_mismatch": f"_ = {mismatch_val}\n",
        "print_match": f"print({val})\n",
        "assert_eq": f"assert {var} == {val}\n",
        "type_match": f"type({val})\n",
        "isinstance_match": f"isinstance({var}, int)\n",
        "eval_var": f"_ = {var}\n",
    }


def run():
    print("=== Exp #72: Collision Form Generalization ===")

    adapter = ModelAdapter(MODEL_CFG)
    import torch
    adapter.mdl.half()
    gc.collect()
    print("Model loaded (float16)")

    query_tmpl = MODEL_CFG["query_template"]
    t0 = time.time()

    all_dists = {}
    n_done = 0

    for var, val in CELLS:
        prefix_text = build_prefix(PREFIX_TEMPLATE, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        forms = get_forms(var, val)

        for surf_name, surf_tmpl in SURFACES.items():
            surf_text = surf_tmpl.replace("{var}", var)
            for form_name, form_text in forms.items():
                full_suffix = surf_text + form_text + query
                dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                all_dists[(var, val, surf_name, form_name)] = dist
                n_done += 1

        del prefix_state
        gc.collect()
        elapsed = time.time() - t0
        print(f"  Cell {var}_{val} done, {n_done} readouts, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {n_done} readouts in {elapsed:.1f}s")

    results = []
    for surf_a, surf_b, pair_name in EQUIV_PAIRS:
        for var, val in CELLS:
            forms = get_forms(var, val)
            for form_name in forms:
                ka = (var, val, surf_a, form_name)
                kb = (var, val, surf_b, form_name)
                if ka in all_dists and kb in all_dists:
                    cell_tv = float(tv(all_dists[ka], all_dists[kb]))
                    flip = (DIGIT_LABELS[int(np.argmax(all_dists[ka]))] !=
                            DIGIT_LABELS[int(np.argmax(all_dists[kb]))])
                    results.append({
                        "pair": pair_name,
                        "cell": f"{var}_{val}",
                        "form": form_name,
                        "tv": cell_tv,
                        "flip": flip,
                    })

    # Form comparison
    print("\n=== FORM AMPLIFICATION (strong_assert) ===")
    print(f"{'Form':>20s} {'mean_TV':>10s} {'vs_empty':>10s}")
    print("-" * 45)
    empty_tvs = [r["tv"] for r in results
                 if r["pair"] == "strong_assert" and r["form"] == "empty"]
    empty_mean = np.mean(empty_tvs) if empty_tvs else 0.001

    form_order = ["empty", "assign_mismatch", "type_match", "print_match",
                  "isinstance_match", "eval_var", "assign_match", "assert_eq"]
    for form in form_order:
        tvs = [r["tv"] for r in results
               if r["pair"] == "strong_assert" and r["form"] == form]
        if tvs:
            m = np.mean(tvs)
            ratio = m / empty_mean if empty_mean > 0.001 else float('inf')
            print(f"{form:>20s} {m:10.4f} {ratio:10.1f}x")

    # Same on boundary
    print("\n=== FORM AMPLIFICATION (boundary_pair) ===")
    print(f"{'Form':>20s} {'mean_TV':>10s} {'vs_empty':>10s}")
    print("-" * 45)
    empty_tvs_b = [r["tv"] for r in results
                   if r["pair"] == "boundary_pair" and r["form"] == "empty"]
    empty_mean_b = np.mean(empty_tvs_b) if empty_tvs_b else 0.001

    for form in form_order:
        tvs = [r["tv"] for r in results
               if r["pair"] == "boundary_pair" and r["form"] == form]
        if tvs:
            m = np.mean(tvs)
            ratio = m / empty_mean_b if empty_mean_b > 0.001 else float('inf')
            print(f"{form:>20s} {m:10.4f} {ratio:10.1f}x")

    # Categorize: value-matching vs var-referencing vs control
    print("\n=== CATEGORY ANALYSIS (strong_assert) ===")
    cats = {
        "control": ["empty", "assign_mismatch"],
        "value_mention": ["type_match", "print_match", "assign_match"],
        "var_reference": ["eval_var", "isinstance_match"],
        "explicit_comparison": ["assert_eq"],
    }
    for cat_name, forms in cats.items():
        tvs = [r["tv"] for r in results
               if r["pair"] == "strong_assert" and r["form"] in forms]
        if tvs:
            m = np.mean(tvs)
            ratio = m / empty_mean if empty_mean > 0.001 else float('inf')
            print(f"  {cat_name:>25s}: mean_TV={m:.4f} ({ratio:.1f}x)")

    summary = {
        "experiment": 72,
        "name": "collision_forms",
        "n_readouts": n_done,
        "n_comparisons": len(results),
        "results": results,
    }

    result_dir = Path("experiments/results/omega_collision_forms")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {result_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
