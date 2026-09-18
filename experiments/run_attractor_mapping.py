"""Attractor Mapping (Experiment #68).

The 9-attractor hypothesis: Qwen3-1.7B-Base's argmax is pulled toward "9"
because the nested template uses 99/999/9999/99999. If we change the nested
values, does the attractor shift?

Tests:
1. Standard template: nested 99/999/9999/99999 -> expect 9-attractor
2. 3-template: nested 33/333/3333/33333 -> does attractor shift to 3?
3. 7-template: nested 77/777/7777/77777 -> shift to 7?
4. Mixed: nested 12/345/6789/10111 -> no dominant digit?
5. Clean: no nested values, just flat x = val -> no attractor?

6 cells x 5 templates x 6 surfaces x 2 continuations = 360 readouts.
~5 minutes.
"""

import json, sys, time, gc
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix

MODEL_CFG = {"model_id": "Qwen/Qwen3-1.7B-Base"}

TEMPLATES = {
    "nested_9": (
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
    ),
    "nested_3": (
        "{var} = {outer_val}\n"
        "def f():\n"
        "    {var} = 33\n"
        "    def g():\n"
        "        {var} = 333\n"
        "        def h():\n"
        "            {var} = 3333\n"
        "            def k():\n"
        "                {var} = 33333\n"
        "                return {var}\n"
        "            k()\n"
        "            return {var}\n"
        "        h()\n"
        "        return {var}\n"
        "    g()\n"
        "    return {var}\n"
    ),
    "nested_7": (
        "{var} = {outer_val}\n"
        "def f():\n"
        "    {var} = 77\n"
        "    def g():\n"
        "        {var} = 777\n"
        "        def h():\n"
        "            {var} = 7777\n"
        "            def k():\n"
        "                {var} = 77777\n"
        "                return {var}\n"
        "            k()\n"
        "            return {var}\n"
        "        h()\n"
        "        return {var}\n"
        "    g()\n"
        "    return {var}\n"
    ),
    "nested_mixed": (
        "{var} = {outer_val}\n"
        "def f():\n"
        "    {var} = 12\n"
        "    def g():\n"
        "        {var} = 345\n"
        "        def h():\n"
        "            {var} = 6789\n"
        "            def k():\n"
        "                {var} = 10111\n"
        "                return {var}\n"
        "            k()\n"
        "            return {var}\n"
        "        h()\n"
        "        return {var}\n"
        "    g()\n"
        "    return {var}\n"
    ),
    "flat": (
        "{var} = {outer_val}\n"
        "def f():\n"
        "    return {var}\n"
    ),
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
    "nonvar_42": "_ = 42\n",
}

CELLS = [
    ("x", 1), ("x", 5),
    ("y", 2), ("y", 6),
    ("z", 3), ("z", 7),
]

QUERY = "f()\nprint({var})  # Output: "
DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def run_attractor_mapping():
    print("=== Attractor Mapping ===", flush=True)
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
                    if n % 60 == 0:
                        print(f"  {n} readouts, {time.time()-t0:.0f}s", flush=True)

            del ps
            gc.collect()

    elapsed = time.time() - t0
    print(f"Done: {n} readouts in {elapsed:.1f}s")

    # Analysis
    from collections import Counter
    per_template = {}
    for tn in TEMPLATES:
        rows = [o for o in observations if o["template"] == tn]
        argmax_dist = Counter(o["argmax_label"] for o in rows)
        correct_rate = sum(1 for o in rows if o["is_correct"]) / len(rows)
        dominant = argmax_dist.most_common(1)[0] if argmax_dist else ("?", 0)
        per_template[tn] = {
            "n": len(rows),
            "accuracy": round(correct_rate, 3),
            "argmax_distribution": dict(argmax_dist),
            "dominant_digit": dominant[0],
            "dominant_count": dominant[1],
            "dominant_fraction": round(dominant[1] / len(rows), 3),
        }

    # Does the attractor shift with the template?
    expected_attractors = {
        "nested_9": "9", "nested_3": "3", "nested_7": "7",
        "nested_mixed": None, "flat": None,
    }
    attractor_shifts = {}
    for tn, expected in expected_attractors.items():
        actual = per_template[tn]["dominant_digit"]
        if expected:
            attractor_shifts[tn] = {
                "expected": expected,
                "actual": actual,
                "matches": actual == expected,
            }

    summary = {
        "n_readouts": n,
        "per_template": per_template,
        "attractor_shifts": attractor_shifts,
    }

    n_match = sum(1 for v in attractor_shifts.values() if v["matches"])
    if n_match == len(attractor_shifts):
        summary["verdict"] = "ATTRACTOR_SHIFTS: dominant digit tracks template nested values"
    elif n_match > 0:
        summary["verdict"] = f"PARTIAL_SHIFT: {n_match}/{len(attractor_shifts)} templates shift attractor"
    else:
        summary["verdict"] = "NO_SHIFT: attractor does not track template values"

    out_dir = Path("experiments/results/omega_attractor_mapping")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "observations.json", "w") as f:
        json.dump(observations, f)

    print(f"\n=== ATTRACTOR MAPPING RESULTS ===")
    for tn, data in per_template.items():
        exp = expected_attractors.get(tn, "?")
        print(f"  {tn:15s}: dominant={data['dominant_digit']} "
              f"({data['dominant_fraction']:.0%}) "
              f"expected={exp} acc={data['accuracy']:.0%}")
        top3 = sorted(data["argmax_distribution"].items(),
                       key=lambda x: -x[1])[:3]
        print(f"    top3: {', '.join(f'{d}:{c}' for d,c in top3)}")

    print(f"\nVERDICT: {summary['verdict']}")
    del adapter
    gc.collect()
    return summary


if __name__ == "__main__":
    run_attractor_mapping()
