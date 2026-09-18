"""Attractor Basin Mapping (Experiment E4/R5-derived).

Tests the two-basin competition mechanism revealed by #59 reanalysis:
- Identity continuation creates competition between outer value and
  innermost-scope attractor ("9" from nested 99/999/9999/99999)
- Prediction: attractor digit comes from innermost scope leading digit
- Test: change inner values to 11/111/1111/11111 → attractor should be "1"
- Test: vary outer_val 0-9 → higher initial margin = more flip-resistant

Key measurement: p(outer_val) vs p(attractor_digit) after identity.
If attractor_digit changes with template → mechanism confirmed.
If margin predicts flipping → quantitative basin model.

~240 readouts (2 templates × 10 values × 6 surfaces × 2 continuations).
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter
from omega import tv


MODEL_CFG = {"model_id": "Qwen/Qwen3-1.7B-Base"}

TEMPLATES = {
    "nines": {
        "code": (
            "{var} = {outer_val}\ndef f():\n    {var} = 99\n"
            "    def g():\n        {var} = 999\n        def h():\n"
            "            {var} = 9999\n            def k():\n"
            "                {var} = 99999\n                return {var}\n"
            "            k()\n            return {var}\n"
            "        h()\n        return {var}\n"
            "    g()\n    return {var}\n"
        ),
        "expected_attractor": "9",
    },
    "ones": {
        "code": (
            "{var} = {outer_val}\ndef f():\n    {var} = 11\n"
            "    def g():\n        {var} = 111\n        def h():\n"
            "            {var} = 1111\n            def k():\n"
            "                {var} = 11111\n                return {var}\n"
            "            k()\n            return {var}\n"
            "        h()\n        return {var}\n"
            "    g()\n    return {var}\n"
        ),
        "expected_attractor": "1",
    },
}

SURFACES = {
    "assert_novar": "# State unchanged.\n",
    "assert_var": "# {var} stays the same.\n",
    "boundary_1": "\n",
    "boundary_2": "pass\n",
    "minimal_ok": "# OK.\n",
    "minimal_good": "# Good.\n",
}

CONTINUATIONS = {
    "empty": "",
    "identity": "{var} = {var}\n",
}

OUTER_VALS = list(range(10))
DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def run():
    print("=== Attractor Basin Mapping ===")
    adapter = ModelAdapter({"model_id": MODEL_CFG["model_id"]})
    var = "x"

    observations = []
    t0 = time.time()
    total = len(TEMPLATES) * len(OUTER_VALS) * len(SURFACES) * len(CONTINUATIONS)
    done = 0

    for tmpl_name, tmpl_cfg in TEMPLATES.items():
        template = tmpl_cfg["code"]
        expected_attractor = tmpl_cfg["expected_attractor"]

        for outer_val in OUTER_VALS:
            prefix_code = template.format(var=var, outer_val=outer_val)
            query = f"f()\nprint({var})  # Output: "
            prefix_state = adapter.get_state_after_prefix(prefix_code)

            for surf_name, surf_text in SURFACES.items():
                surface = surf_text.format(var=var)

                for cont_name, cont_text in CONTINUATIONS.items():
                    continuation = cont_text.format(var=var)
                    suffix = surface + continuation + query
                    dist = adapter.get_dist_from_state(prefix_state, suffix)

                    argmax_idx = int(np.argmax(dist))
                    correct_digit = str(outer_val)
                    attractor_idx = int(expected_attractor)

                    observations.append({
                        "template": tmpl_name,
                        "expected_attractor": expected_attractor,
                        "outer_val": outer_val,
                        "var": var,
                        "surface": surf_name,
                        "continuation": cont_name,
                        "dist": dist.tolist(),
                        "argmax_label": DIGIT_LABELS[argmax_idx],
                        "argmax_prob": float(dist[argmax_idx]),
                        "p_correct": float(dist[outer_val]),
                        "p_attractor": float(dist[attractor_idx]),
                        "is_correct": DIGIT_LABELS[argmax_idx] == correct_digit,
                        "is_attractor": DIGIT_LABELS[argmax_idx] == expected_attractor,
                        "margin": float(sorted(dist, reverse=True)[0] - sorted(dist, reverse=True)[1]),
                    })
                    done += 1
                    if done % 50 == 0:
                        elapsed = time.time() - t0
                        print(f"  {done}/{total} ({done/elapsed:.1f}/s)")

    print(f"\nDone: {len(observations)} readouts in {time.time()-t0:.0f}s")
    out_dir = Path("experiments/results/omega_attractor_basin")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "observations.json", "w") as f:
        json.dump(observations, f, indent=2)

    summary = analyze(observations)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_dir}")
    return summary


def analyze(observations):
    results = {}

    for tmpl_name in TEMPLATES:
        expected_attr = TEMPLATES[tmpl_name]["expected_attractor"]
        tmpl_obs = [r for r in observations if r["template"] == tmpl_name]

        # Baseline (empty) margin by outer_val
        empty_margins = {}
        for val in OUTER_VALS:
            empty_sub = [r for r in tmpl_obs
                        if r["continuation"] == "empty" and r["outer_val"] == val]
            if empty_sub:
                empty_margins[val] = {
                    "mean_margin": round(float(np.mean([r["margin"] for r in empty_sub])), 4),
                    "correct_rate": round(sum(1 for r in empty_sub if r["is_correct"]) / len(empty_sub), 4),
                }

        # Identity flip analysis
        identity_flips = {}
        for val in OUTER_VALS:
            id_sub = [r for r in tmpl_obs
                     if r["continuation"] == "identity" and r["outer_val"] == val
                     and r["surface"] not in ["assert_novar", "assert_var"]]
            if id_sub:
                n_correct = sum(1 for r in id_sub if r["is_correct"])
                n_attractor = sum(1 for r in id_sub if r["is_attractor"])
                mean_p_correct = float(np.mean([r["p_correct"] for r in id_sub]))
                mean_p_attractor = float(np.mean([r["p_attractor"] for r in id_sub]))

                identity_flips[val] = {
                    "n": len(id_sub),
                    "correct_rate": round(n_correct / len(id_sub), 4),
                    "attractor_rate": round(n_attractor / len(id_sub), 4),
                    "mean_p_correct": round(mean_p_correct, 4),
                    "mean_p_attractor": round(mean_p_attractor, 4),
                    "empty_margin": empty_margins.get(val, {}).get("mean_margin", None),
                }

        # Does attractor match template?
        non_assert_identity = [r for r in tmpl_obs
                               if r["continuation"] == "identity"
                               and r["surface"] not in ["assert_novar", "assert_var"]]
        n_attractor_match = sum(1 for r in non_assert_identity if r["is_attractor"])
        attractor_rate = round(n_attractor_match / max(len(non_assert_identity), 1), 4)

        results[tmpl_name] = {
            "expected_attractor": expected_attr,
            "attractor_rate_non_assert": attractor_rate,
            "n_non_assert_identity": len(non_assert_identity),
            "empty_margins": empty_margins,
            "identity_flips": identity_flips,
        }

        print(f"\n=== {tmpl_name} (attractor={expected_attr}) ===")
        print(f"  Attractor rate (non-assert, identity): {attractor_rate:.1%}")
        print(f"  val | correct% | attractor% | p(correct) | p(attractor) | empty_margin")
        for val in OUTER_VALS:
            if val in identity_flips:
                f = identity_flips[val]
                em = f.get("empty_margin", 0) or 0
                print(f"   {val}  | {f['correct_rate']:7.1%}  | {f['attractor_rate']:9.1%}  | "
                      f"{f['mean_p_correct']:10.4f}  | {f['mean_p_attractor']:12.4f}  | {em:.4f}")

    # Cross-template comparison: does attractor digit track template?
    nines_attr = results["nines"]["attractor_rate_non_assert"]
    ones_attr = results["ones"]["attractor_rate_non_assert"]
    if nines_attr > 0.3 and ones_attr > 0.3:
        verdict = "ATTRACTOR_TRACKS_TEMPLATE: both templates create their expected attractor"
    elif nines_attr > 0.3 and ones_attr < 0.1:
        verdict = "9_IS_DEFAULT: only nines template creates attractor; 1 doesn't attract"
    elif nines_attr < 0.1 and ones_attr < 0.1:
        verdict = "NO_ATTRACTOR: identity doesn't create competing attractors"
    else:
        verdict = f"MIXED: nines_attractor={nines_attr:.1%}, ones_attractor={ones_attr:.1%}"

    # Margin prediction: does empty margin predict identity flip resistance?
    margins_and_flips = []
    for tmpl_name in TEMPLATES:
        for val, f in results[tmpl_name].get("identity_flips", {}).items():
            if f.get("empty_margin") is not None:
                margins_and_flips.append((f["empty_margin"], f["correct_rate"]))

    if len(margins_and_flips) >= 5:
        ms, cs = zip(*margins_and_flips)
        from scipy.stats import spearmanr
        rho, p = spearmanr(ms, cs)
        margin_predicts = f"rho={rho:.3f}, p={p:.4f}"
    else:
        margin_predicts = "insufficient data"

    results["verdict"] = verdict
    results["margin_predicts_resistance"] = margin_predicts
    print(f"\n=== VERDICT: {verdict} ===")
    print(f"  Margin predicts resistance: {margin_predicts}")

    return results


if __name__ == "__main__":
    run()
