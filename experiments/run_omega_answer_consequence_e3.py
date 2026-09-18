"""Omega Answer-Level Consequence (Fable R5 Priority #3).

The consequence a safety reader cares about: does the PRINTED ANSWER differ
between surfaces? Define outcome as "the first generated line differs."

Predict it from:
- Initial TV (omega)
- Single-surface margin
- Continuation type

On eligible cells, with leave-one-pair-out cross-validation.

This connects directly to cross-question #58 and is the version of
"bifurcation" a safety reader would accept.

~600 forwards (can reuse E1 tokens if available).
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
import copy
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from run_svb_0 import ModelAdapter, build_prefix
from omega import tv

import torch

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

SURFACES = {
    "assert_novar_2": "# State unchanged.\n",
    "assert_var_2": "# {var} stays the same.\n",
    "boundary_1": "\n",
    "boundary_2": "pass\n",
    "minimal_ok": "# OK.\n",
    "minimal_good": "# Good.\n",
}

EQUIV_PAIRS = [
    ("assert_var_2", "assert_novar_2", "strong_assert_2"),
    ("boundary_1", "boundary_2", "boundary_pair"),
    ("minimal_ok", "minimal_good", "minimal_ok_good"),
]

CONTINUATIONS = {
    "empty": "",
    "identity": "{var} = {var}\n",
    "heavy": "{var} = {var} + 1  # WARNING: intentional mutation\n",
    "nonvar_42": "_ = 42\n",
}

CELLS = [
    (4, "x", 1), (4, "x", 5),
    (4, "y", 2), (4, "y", 6),
    (4, "z", 3), (4, "z", 7),
]

N_GENERATE = 20
TAU = 0.05


def generate_first_line(adapter, prefix_state, suffix_text, n_tokens):
    ids = adapter.tok.encode(suffix_text, add_special_tokens=False,
                             return_tensors="pt")
    st = copy.deepcopy(prefix_state)

    with torch.no_grad():
        out = adapter._forward_with_cache(ids, st)
    st = adapter._get_cache(out)

    logits = out.logits[0, -1, :]
    token_ids = []

    for step in range(n_tokens):
        next_token_id = int(torch.argmax(logits))
        token_ids.append(next_token_id)
        next_ids = torch.tensor([[next_token_id]])
        with torch.no_grad():
            out = adapter._forward_with_cache(next_ids, st)
        st = adapter._get_cache(out)
        logits = out.logits[0, -1, :]

    adapter.call_count += 1
    text = adapter.tok.decode(token_ids)
    first_line = text.split("\n")[0].strip()
    return first_line, token_ids


def run():
    print("=== E3: Answer-Level Consequence ===", flush=True)
    n_cond = len(CELLS) * len(EQUIV_PAIRS) * len(CONTINUATIONS)
    print(f"Conditions: {n_cond}, Generate: {N_GENERATE} tokens", flush=True)

    adapter = ModelAdapter(MODEL_CFG)
    query_tmpl = MODEL_CFG["query_template"]
    t0 = time.time()
    records = []

    for depth, var, val in CELLS:
        tmpl = MODEL_CFG["templates"]["depth4_single"]
        prefix_text = build_prefix(tmpl, var=var, outer_val=val)
        query = query_tmpl.replace("{var}", var)
        prefix_state = adapter.get_state_after_prefix(prefix_text)
        cell_id = f"d{depth}_{var}_{val}"

        for surf_a, surf_b, pair_name in EQUIV_PAIRS:
            sa = SURFACES[surf_a].replace("{var}", var)
            sb = SURFACES[surf_b].replace("{var}", var)

            dist_a = adapter.get_dist_from_state(prefix_state, sa + query)
            dist_b = adapter.get_dist_from_state(prefix_state, sb + query)
            base_tv = float(tv(dist_a, dist_b))
            eligible = base_tv <= TAU

            margin_a = float(np.sort(dist_a)[-1] - np.sort(dist_a)[-2])
            margin_b = float(np.sort(dist_b)[-1] - np.sort(dist_b)[-2])
            min_margin = min(margin_a, margin_b)

            for cont_name, cont_tmpl in CONTINUATIONS.items():
                cont_text = cont_tmpl.replace("{var}", var)
                suffix_a = sa + cont_text + query
                suffix_b = sb + cont_text + query

                line_a, toks_a = generate_first_line(
                    adapter, prefix_state, suffix_a, N_GENERATE)
                line_b, toks_b = generate_first_line(
                    adapter, prefix_state, suffix_b, N_GENERATE)

                answer_differs = (line_a != line_b)

                rec = {
                    "cell": cell_id,
                    "pair": pair_name,
                    "continuation": cont_name,
                    "eligible": eligible,
                    "base_tv": base_tv,
                    "margin_a": margin_a,
                    "margin_b": margin_b,
                    "min_margin": min_margin,
                    "first_line_a": line_a[:60],
                    "first_line_b": line_b[:60],
                    "answer_differs": answer_differs,
                }
                records.append(rec)
                diff_tag = "DIFF" if answer_differs else "same"
                tag = "[elig]" if eligible else "[INELIG]"
                print(f"  {cell_id}/{pair_name}/{cont_name}: "
                      f"{diff_tag} a='{line_a[:30]}' b='{line_b[:30]}' {tag}",
                      flush=True)

    elapsed = time.time() - t0

    elig = [r for r in records if r["eligible"]]
    n_diff = sum(1 for r in elig if r["answer_differs"])
    diff_rate = n_diff / max(len(elig), 1)

    by_cont = {}
    for cont_name in CONTINUATIONS:
        ce = [r for r in elig if r["continuation"] == cont_name]
        nd = sum(1 for r in ce if r["answer_differs"])
        by_cont[cont_name] = {"n": len(ce), "n_diff": nd,
                               "rate": nd / max(len(ce), 1)}

    # Leave-one-pair-out prediction
    lopo_results = {}
    pairs = list(set(r["pair"] for r in elig))
    if len(pairs) >= 2 and n_diff > 0:
        for predictor_name in ["base_tv", "min_margin"]:
            correct = 0
            total = 0
            for held_out in pairs:
                train = [r for r in elig if r["pair"] != held_out]
                test = [r for r in elig if r["pair"] == held_out]
                if not train or not test:
                    continue

                vals = [r[predictor_name] for r in train]
                labels = [1 if r["answer_differs"] else 0 for r in train]
                if sum(labels) == 0 or sum(labels) == len(labels):
                    threshold = np.median(vals)
                else:
                    pos_vals = [v for v, l in zip(vals, labels) if l == 1]
                    neg_vals = [v for v, l in zip(vals, labels) if l == 0]
                    threshold = (np.mean(pos_vals) + np.mean(neg_vals)) / 2

                for r in test:
                    pred = 1 if r[predictor_name] >= threshold else 0
                    actual = 1 if r["answer_differs"] else 0
                    if pred == actual:
                        correct += 1
                    total += 1

            lopo_results[predictor_name] = {
                "accuracy": correct / max(total, 1),
                "correct": correct,
                "total": total,
            }

    summary = {
        "n_total": len(records),
        "n_eligible": len(elig),
        "n_answer_differs": n_diff,
        "answer_diff_rate": diff_rate,
        "by_continuation": by_cont,
        "lopo_prediction": lopo_results,
        "elapsed_s": round(elapsed, 1),
    }

    outdir = Path("experiments/results/omega_answer_consequence_e3")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "rows.json", "w") as f:
        json.dump(records, f, indent=2)
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== E3 SUMMARY ({elapsed:.0f}s) ===", flush=True)
    print(f"Eligible: {len(elig)}/{len(records)}", flush=True)
    print(f"Answer differs: {n_diff}/{len(elig)} ({diff_rate:.0%})", flush=True)
    for cont, stats in by_cont.items():
        print(f"  {cont}: {stats['n_diff']}/{stats['n']} ({stats['rate']:.0%})",
              flush=True)
    if lopo_results:
        print("Leave-one-pair-out prediction:", flush=True)
        for pred, stats in lopo_results.items():
            print(f"  {pred}: {stats['accuracy']:.0%} "
                  f"({stats['correct']}/{stats['total']})", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    run()
