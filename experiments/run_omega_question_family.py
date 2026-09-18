"""Exp #65: Question-family topology mapping.

R6 reanalysis showed omega is family-dependent: assert↔print ρ=0.651 (same family)
but addition shows no correlation (different family). This experiment maps the full
family structure by measuring TV rank stability across 10 diverse questions.

Design:
- 10 questions spanning 4 hypothesized families:
  - Return-value family: print, assert, conditional, type-check
  - Arithmetic family: addition, subtraction, comparison
  - String family: repr, f-string format
  - Side-effect family: assignment target
- 2 cells × 3 surface pairs × 5 continuations × 10 questions = 300 readouts
- Pairwise Spearman across all 45 question pairs
- Output: 10×10 correlation matrix → family clustering

~300 model forwards (fast, <5 min on CPU).
"""

import gc
import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv

MODEL_ID = "Qwen/Qwen3-1.7B-Base"

TEMPLATE = (
    "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
    "    def g():\n        {var} = 999\n        def h():\n"
    "            {var} = 9999\n            def k():\n"
    "                {var} = 99999\n                return {var}\n"
    "            k()\n            return {var}\n"
    "        h()\n        return {var}\n"
    "    g()\n    return {var}\n"
)

QUESTIONS = {
    # Return-value family (probes f()'s return value directly)
    "q_print": "f()\nprint({var})  # Output: ",
    "q_assert": "f()\nassert {var} == ",
    "q_conditional": "f()\nif {var} > 5:\n    print(",
    "q_type": "f()\ntype({var})  # ",

    # Arithmetic family (probes numeric operations on the result)
    "q_add": "f()\nprint({var} + 1)  # Output: ",
    "q_subtract": "f()\nprint({var} - 1)  # Output: ",
    "q_compare": "f()\nprint({var} > 5)  # Output: ",

    # String family (probes string representation)
    "q_repr": "f()\nprint(repr({var}))  # Output: ",
    "q_fstring": "f()\nprint(f'val={{{var}}}')  # Output: ",

    # Side-effect family (probes assignment/mutation)
    "q_assign": "f()\ny = {var}\nprint(y)  # Output: ",
}

SURFACES = {
    "assert_novar_2": "# State unchanged.\n",
    "assert_var_2": "# {var} stays the same.\n",
    "boundary_1": "\n",
    "boundary_2": "pass\n",
    "minimal_good": "# ok\n",
    "minimal_ok": "# good\n",
}

SURFACE_PAIRS = [
    ("assert_var_2", "assert_novar_2", "strong_assert"),
    ("boundary_1", "boundary_2", "boundary"),
    ("minimal_good", "minimal_ok", "minimal"),
]

CONTINUATIONS = {
    "empty": "",
    "identity": "{var} = {var}\n",
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
    "nonvar_42": "_ = 42\n",
    "safety_comment": "# Safety: variable unchanged\n",
}

CELLS = [
    (4, "x", 1),
    (4, "x", 5),
]

TAU = 0.05
DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def run():
    print("=== Exp #65: Question-Family Topology Mapping ===")
    adapter = ModelAdapter({"model_id": MODEL_ID})
    t0 = time.time()

    observations = []
    n_total = len(CELLS) * len(SURFACE_PAIRS) * len(CONTINUATIONS) * len(QUESTIONS)
    print(f"Planned: {n_total} pair-readouts ({len(CELLS)} cells × {len(SURFACE_PAIRS)} pairs × "
          f"{len(CONTINUATIONS)} conts × {len(QUESTIONS)} questions)")

    for depth, var, val in CELLS:
        prefix_text = build_prefix(TEMPLATE, var=var, outer_val=val)
        prefix_state = adapter.get_state_after_prefix(prefix_text)

        for sa, sb, pair_name in SURFACE_PAIRS:
            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)

                for q_name, q_tmpl in QUESTIONS.items():
                    query = q_tmpl.replace("{var}", var)

                    dists = {}
                    argmaxes = {}
                    margins = {}
                    for surf_name, surf_tmpl in [(sa, SURFACES[sa]), (sb, SURFACES[sb])]:
                        surface_text = surf_tmpl.replace("{var}", var)
                        full_suffix = surface_text + cont_text + query
                        dist = adapter.get_dist_from_state(prefix_state, full_suffix)
                        dists[surf_name] = dist
                        am = int(np.argmax(dist))
                        argmaxes[surf_name] = DIGIT_LABELS[am]
                        sorted_probs = sorted(dist, reverse=True)
                        margins[surf_name] = float(sorted_probs[0] - sorted_probs[1])

                    pair_tv = float(tv(dists[sa], dists[sb]))
                    flip = argmaxes[sa] != argmaxes[sb]
                    min_margin = min(margins[sa], margins[sb])

                    observations.append({
                        "cell": f"{var}_{val}",
                        "pair": pair_name,
                        "continuation": cont_name,
                        "question": q_name,
                        "tv": pair_tv,
                        "flip": flip,
                        "min_margin": min_margin,
                        "argmax_a": argmaxes[sa],
                        "argmax_b": argmaxes[sb],
                    })

                    status = "FLIP" if flip else "ok"
                    print(f"  {var}_{val}/{pair_name}/{cont_name}/{q_name}: "
                          f"TV={pair_tv:.4f} margin={min_margin:.4f} [{status}]",
                          flush=True)

            gc.collect()

        del prefix_state
        gc.collect()

    elapsed = time.time() - t0
    print(f"\nDone: {len(observations)} readouts in {elapsed:.1f}s "
          f"({adapter.call_count} forwards)")

    summary = analyze(observations)

    out_dir = Path("experiments/results/omega_question_family")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "observations.json", "w") as f:
        json.dump(observations, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_dir}")
    return summary


def analyze(observations):
    from scipy.stats import spearmanr

    questions = sorted(set(o["question"] for o in observations))
    conts = sorted(set(o["continuation"] for o in observations))

    # Build TV matrix: (cell, pair, continuation) → {question: tv}
    tv_by_q = defaultdict(lambda: {})
    for o in observations:
        key = (o["cell"], o["pair"], o["continuation"])
        tv_by_q[key][o["question"]] = o["tv"]

    # Only keep rows with all questions
    complete = {k: v for k, v in tv_by_q.items() if len(v) == len(questions)}
    n_complete = len(complete)
    print(f"\n=== ANALYSIS: {n_complete} complete rows ===")

    # Pairwise Spearman
    corr_matrix = {}
    print(f"\n{'':>15}", end="")
    for q in questions:
        print(f" {q:>12}", end="")
    print()

    for qa in questions:
        print(f"{qa:>15}", end="")
        for qb in questions:
            if qa == qb:
                corr_matrix[(qa, qb)] = {"rho": 1.0, "p": 0.0}
                print(f" {'1.000':>12}", end="")
                continue
            x = [v[qa] for v in complete.values()]
            y = [v[qb] for v in complete.values()]
            if len(x) < 3:
                corr_matrix[(qa, qb)] = {"rho": float("nan"), "p": float("nan")}
                print(f" {'nan':>12}", end="")
                continue
            rho, p = spearmanr(x, y)
            corr_matrix[(qa, qb)] = {"rho": float(rho), "p": float(p)}
            sig = "*" if p < 0.05 else " "
            print(f" {rho:>5.3f}{sig:1s}{p:5.3f}", end="")
        print()

    # Family clustering
    print("\n=== FAMILY CLUSTERING ===")
    hypothesized = {
        "return_value": ["q_print", "q_assert", "q_conditional", "q_type"],
        "arithmetic": ["q_add", "q_subtract", "q_compare"],
        "string": ["q_repr", "q_fstring"],
        "side_effect": ["q_assign"],
    }

    family_results = {}
    for fname, members in hypothesized.items():
        within = []
        for qa, qb in combinations(members, 2):
            if (qa, qb) in corr_matrix:
                within.append(corr_matrix[(qa, qb)]["rho"])
        between = []
        non_members = [q for q in questions if q not in members]
        for qa in members:
            for qb in non_members:
                if (qa, qb) in corr_matrix:
                    between.append(corr_matrix[(qa, qb)]["rho"])

        mean_within = float(np.nanmean(within)) if within else float("nan")
        mean_between = float(np.nanmean(between)) if between else float("nan")
        family_results[fname] = {
            "members": members,
            "mean_within_rho": mean_within,
            "mean_between_rho": mean_between,
            "gap": mean_within - mean_between if not (np.isnan(mean_within) or np.isnan(mean_between)) else float("nan"),
            "n_within_pairs": len(within),
            "n_between_pairs": len(between),
        }
        print(f"  {fname}: within={mean_within:.3f} between={mean_between:.3f} "
              f"gap={mean_within-mean_between:.3f}" if not np.isnan(mean_within) else f"  {fname}: insufficient data")

    # Flip rates per question
    flip_rates = {}
    for q in questions:
        q_obs = [o for o in observations if o["question"] == q]
        flip_rates[q] = {
            "n": len(q_obs),
            "n_flips": sum(1 for o in q_obs if o["flip"]),
            "rate": sum(1 for o in q_obs if o["flip"]) / max(len(q_obs), 1),
            "mean_tv": float(np.mean([o["tv"] for o in q_obs])),
        }
    print(f"\n=== FLIP RATES ===")
    for q in questions:
        fr = flip_rates[q]
        print(f"  {q}: {fr['n_flips']}/{fr['n']} flips ({fr['rate']:.2%}), "
              f"mean TV={fr['mean_tv']:.4f}")

    # Overall statistics
    all_rhos = [v["rho"] for v in corr_matrix.values() if not np.isnan(v["rho"]) and v["rho"] < 1.0]
    mean_rho = float(np.mean(all_rhos)) if all_rhos else float("nan")
    n_sig = sum(1 for v in corr_matrix.values() if v["p"] < 0.05 and v["rho"] < 1.0)

    summary = {
        "n_observations": len(observations),
        "n_complete_rows": n_complete,
        "n_questions": len(questions),
        "mean_pairwise_rho": mean_rho,
        "n_significant_pairs": n_sig,
        "n_total_pairs": len(all_rhos),
        "correlation_matrix": {f"{qa}|{qb}": v for (qa, qb), v in corr_matrix.items()},
        "family_results": family_results,
        "flip_rates": flip_rates,
    }

    # Verdict
    return_rhos = [corr_matrix[(qa, qb)]["rho"]
                   for qa, qb in combinations(hypothesized["return_value"], 2)
                   if (qa, qb) in corr_matrix and not np.isnan(corr_matrix[(qa, qb)]["rho"])]
    mean_return = float(np.mean(return_rhos)) if return_rhos else float("nan")

    if not np.isnan(mean_return) and mean_return > 0.5:
        verdict = f"FAMILY_CONFIRMED: return-value family mean rho={mean_return:.3f}"
    elif n_sig > len(all_rhos) * 0.3:
        verdict = f"MODERATE_STRUCTURE: {n_sig}/{len(all_rhos)} significant pairs"
    else:
        verdict = f"WEAK_STRUCTURE: only {n_sig}/{len(all_rhos)} significant pairs"

    summary["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")
    return summary


if __name__ == "__main__":
    run()
