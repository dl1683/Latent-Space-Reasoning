"""Attractor-Free Template (R6 Experiment #1).

THE decisive test: replace nested 99/999/9999/99999 with non-digit strings.
If identity amplification collapses, the effect is digit-basin competition.
If it survives, the effect is about variable state tracking.

6 cells x 6 surfaces x 6 continuations = 216 readouts. ~4 minutes.
"""

import json, sys, time, gc
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix

MODEL_CFG = {"model_id": "Qwen/Qwen3-1.7B-Base"}

TEMPLATE_DIGIT = (
    "{var} = {outer_val}\n"
    "def f():\n"
    "    {var} = 99\n"
    "    def g():\n"
    "        {var} = 999\n"
    "        def h():\n"
    "            {var} = 9999\n"
    "            def k():\n"
    "                {var} = 99999\n"
    "                return {var}\n"
    "            k()\n"
    "            return {var}\n"
    "        h()\n"
    "        return {var}\n"
    "    g()\n"
    "    return {var}\n"
)

TEMPLATE_STRING = (
    '{var} = {outer_val}\n'
    'def f():\n'
    '    {var} = "cat"\n'
    '    def g():\n'
    '        {var} = "dog"\n'
    '        def h():\n'
    '            {var} = "fox"\n'
    '            def k():\n'
    '                {var} = "owl"\n'
    '                return {var}\n'
    '            k()\n'
    '            return {var}\n'
    '        h()\n'
    '        return {var}\n'
    '    g()\n'
    '    return {var}\n'
)

TEMPLATES = {"digit_99": TEMPLATE_DIGIT, "string_animals": TEMPLATE_STRING}

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
    "heavy": "{var} = {var} + 0\n",
    "neutral": "_ = 0\n",
    "nonvar_42": "_ = 42\n",
    "bare": "{var}\n",
}

CELLS = [
    ("x", 1), ("x", 5),
    ("y", 2), ("y", 6),
    ("z", 3), ("z", 7),
]

QUERY = "f()\nprint({var})  # Output: "
DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def tv(a, b):
    return 0.5 * np.abs(np.array(a) - np.array(b)).sum()


def run():
    print("=== Attractor-Free Template Test ===", flush=True)
    adapter = ModelAdapter(MODEL_CFG)
    observations = []
    t0 = time.time()
    n = 0

    for tmpl_name, tmpl in TEMPLATES.items():
        for var, val in CELLS:
            prefix = build_prefix(tmpl, var=var, outer_val=val)
            query = QUERY.format(var=var)
            ps = adapter.get_state_after_prefix(prefix)

            for sn, st in SURFACES.items():
                surf = st.replace("{var}", var)
                for cn, ct in CONTINUATIONS.items():
                    cont = ct.replace("{var}", var)
                    dist = adapter.get_dist_from_state(ps, surf + cont + query)
                    n += 1
                    ai = int(np.argmax(dist))
                    observations.append({
                        "template": tmpl_name,
                        "var": var, "val": val,
                        "surface": sn, "continuation": cn,
                        "dist": dist.tolist(),
                        "argmax_label": DIGIT_LABELS[ai],
                        "argmax_prob": float(dist[ai]),
                        "correct_digit": str(val),
                        "is_correct": DIGIT_LABELS[ai] == str(val),
                    })
                    if n % 72 == 0:
                        print(f"  {n}/{216*2} readouts, {time.time()-t0:.0f}s",
                              flush=True)

            del ps
            gc.collect()

    elapsed = time.time() - t0
    print(f"Done: {n} readouts in {elapsed:.1f}s")

    from collections import Counter
    per_template = {}
    for tn in TEMPLATES:
        rows = [o for o in observations if o["template"] == tn]
        argmax_dist = Counter(o["argmax_label"] for o in rows)
        correct_rate = sum(1 for o in rows if o["is_correct"]) / len(rows)
        per_template[tn] = {
            "n": len(rows),
            "accuracy": round(correct_rate, 3),
            "argmax_distribution": dict(argmax_dist),
        }

    # Identity amplification comparison
    for tn in TEMPLATES:
        rows = [o for o in observations if o["template"] == tn]
        empty = [o for o in rows if o["continuation"] == "empty"]
        identity = [o for o in rows if o["continuation"] == "identity"]

        if empty and identity:
            empty_correct = sum(1 for o in empty if o["is_correct"]) / len(empty)
            id_correct = sum(1 for o in identity if o["is_correct"]) / len(identity)
            per_template[tn]["empty_accuracy"] = round(empty_correct, 3)
            per_template[tn]["identity_accuracy"] = round(id_correct, 3)
            per_template[tn]["identity_amplification"] = round(
                id_correct / max(empty_correct, 0.001), 2)

            # TV between surface pairs under identity vs empty
            pairs = [
                ("assert_var", "assert_novar"),
                ("boundary_1", "boundary_2"),
                ("minimal_ok", "minimal_good"),
            ]
            for cont_name in ["empty", "identity", "heavy", "neutral"]:
                tvs = []
                for s1, s2 in pairs:
                    for var, val in CELLS:
                        r1 = [o for o in rows if o["surface"] == s1
                              and o["continuation"] == cont_name
                              and o["var"] == var and o["val"] == val]
                        r2 = [o for o in rows if o["surface"] == s2
                              and o["continuation"] == cont_name
                              and o["var"] == var and o["val"] == val]
                        if r1 and r2:
                            tvs.append(tv(r1[0]["dist"], r2[0]["dist"]))
                if tvs:
                    per_template[tn][f"mean_tv_{cont_name}"] = round(
                        float(np.mean(tvs)), 4)

    # Flip analysis: how many surface pairs flip argmax under identity?
    flip_analysis = {}
    pairs = [
        ("assert_var", "assert_novar"),
        ("boundary_1", "boundary_2"),
        ("minimal_ok", "minimal_good"),
    ]
    for tn in TEMPLATES:
        rows = [o for o in observations if o["template"] == tn]
        n_flips = 0
        n_pairs = 0
        for s1, s2 in pairs:
            for var, val in CELLS:
                for cn in CONTINUATIONS:
                    if cn == "empty":
                        continue
                    r1 = [o for o in rows if o["surface"] == s1
                          and o["continuation"] == cn
                          and o["var"] == var and o["val"] == val]
                    r2 = [o for o in rows if o["surface"] == s2
                          and o["continuation"] == cn
                          and o["var"] == var and o["val"] == val]
                    if r1 and r2:
                        n_pairs += 1
                        if r1[0]["argmax_label"] != r2[0]["argmax_label"]:
                            n_flips += 1
        flip_analysis[tn] = {
            "n_pairs": n_pairs,
            "n_flips": n_flips,
            "flip_rate": round(n_flips / max(n_pairs, 1), 3),
        }

    summary = {
        "n_readouts": n,
        "per_template": per_template,
        "flip_analysis": flip_analysis,
        "elapsed_s": round(elapsed, 1),
    }

    # Verdict
    digit_id_amp = per_template.get("digit_99", {}).get(
        "identity_amplification", 1.0)
    string_id_amp = per_template.get("string_animals", {}).get(
        "identity_amplification", 1.0)

    if string_id_amp < 1.5 and digit_id_amp >= 1.5:
        summary["verdict"] = (
            "ATTRACTOR_EXPLAINS: identity amplification collapses without "
            "digit literals — the effect is digit-basin competition")
    elif string_id_amp >= 1.5:
        summary["verdict"] = (
            "VARIABLE_STATE: identity amplification survives without digits "
            "— the effect is about variable state tracking")
    else:
        summary["verdict"] = (
            "NO_AMPLIFICATION: identity does not amplify in either template")

    out_dir = Path("experiments/results/omega_attractor_free")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    print(f"\n=== ATTRACTOR-FREE RESULTS ===")
    for tn, data in per_template.items():
        print(f"\n  {tn}:")
        print(f"    accuracy={data['accuracy']:.0%}  "
              f"argmax={data['argmax_distribution']}")
        if "identity_amplification" in data:
            print(f"    empty_acc={data['empty_accuracy']:.3f}  "
                  f"identity_acc={data['identity_accuracy']:.3f}  "
                  f"amplification={data['identity_amplification']:.1f}x")
        for k, v in data.items():
            if k.startswith("mean_tv_"):
                print(f"    {k}={v:.4f}")

    print(f"\n  Flip analysis:")
    for tn, fa in flip_analysis.items():
        print(f"    {tn}: {fa['n_flips']}/{fa['n_pairs']} "
              f"({fa['flip_rate']:.1%})")

    print(f"\n  VERDICT: {summary['verdict']}")
    del adapter
    gc.collect()
    return summary


if __name__ == "__main__":
    import traceback
    try:
        run()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
