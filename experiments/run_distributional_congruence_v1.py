"""
Distributional continuation congruence v1.

Strengthens continuation_congruence_v1 from greedy-equality to distributional-equality.
Instead of checking if two same-place histories give the same greedy answer after a
continuation, checks if they give the same FULL NEXT-TOKEN DISTRIBUTION (measured by
sqrt(JSD) between their output distributions).

If same-place histories produce distributions that are negligibly different after the
same continuation, the continuation is a well-defined operation on behavioral places
at the distributional level, not just the greedy level.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import json
import os
from datetime import datetime
from itertools import product as iterproduct

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cpu"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "distributional_congruence_v1")

ENTITIES = ["ZOG", "MIP", "PLIM"]
VALUES = {"ZOG": ("big", "small"), "MIP": ("hot", "cold"), "PLIM": ("red", "blue")}

JSD_THRESHOLD = 0.05


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()
    return model, tok


def get_next_token_dist(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    return F.softmax(logits, dim=-1)


def js_dist(p, q):
    m = (p + q) / 2
    eps = 1e-10
    jsd = (
        0.5 * ((p + eps) * ((p + eps) / (m + eps)).log()).sum()
        + 0.5 * ((q + eps) * ((q + eps) / (m + eps)).log()).sum()
    )
    return math.sqrt(max(0, float(jsd)))


def greedy_answer(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    top_idx = torch.argmax(logits).item()
    return tok.decode([top_idx]).strip()


def get_signature(model, tok, history):
    sig = {}
    for entity in ENTITIES:
        prompt = history + f"\n{entity}:"
        sig[entity] = greedy_answer(model, tok, prompt)
    return sig


def make_histories():
    assignments = list(iterproduct(range(2), range(2), range(2)))
    histories = []
    for a, b, c in assignments:
        vals = (VALUES["ZOG"][a], VALUES["MIP"][b], VALUES["PLIM"][c])
        tag = f"w{a}{b}{c}"
        h1 = f"ZOG: {vals[0]}. MIP: {vals[1]}. PLIM: {vals[2]}."
        histories.append({"tag": tag, "variant": "std", "history": h1, "vals": vals})
        h2 = f"PLIM: {vals[2]}. MIP: {vals[1]}. ZOG: {vals[0]}."
        histories.append({"tag": tag, "variant": "rev", "history": h2, "vals": vals})
        h3 = f"ZOG: {vals[0]}. MIP: {vals[1]}. PLIM: {vals[2]}. ZOG: {vals[0]}."
        histories.append({"tag": tag, "variant": "dup", "history": h3, "vals": vals})
    return histories


def make_continuations_for_history(vals):
    return [
        {"name": "neutral_distractor", "text": " The sky is clear."},
        {"name": "new_entity_commit", "text": " KROT: fast."},
        {"name": "repeat_zog", "text": f" ZOG: {vals[0]}."},
        {"name": "correct_zog",
         "text": f" Actually, ZOG: {VALUES['ZOG'][1 - VALUES['ZOG'].index(vals[0])]}."},
    ]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, tok = load_model()

    print("Phase 1: Baseline")
    histories = make_histories()
    baseline_results = []
    for h in histories:
        sig = get_signature(model, tok, h["history"])
        correct = all(sig[e] == h["vals"][i] for i, e in enumerate(ENTITIES))
        baseline_results.append({
            "tag": h["tag"], "variant": h["variant"],
            "signature": sig,
            "expected": {e: h["vals"][i] for i, e in enumerate(ENTITIES)},
            "all_correct": correct,
        })
        mark = "OK" if correct else "FAIL"
        print(f"  {h['tag']}/{h['variant']}: {sig} [{mark}]")

    n_correct = sum(1 for r in baseline_results if r["all_correct"])
    total = len(baseline_results)
    baseline_rate = n_correct / total
    print(f"\nBaseline: {n_correct}/{total} = {baseline_rate:.1%}")

    if baseline_rate < 0.70:
        print("BASELINE BELOW 70% -> KILL")
        out_path = os.path.join(RESULTS_DIR, "results.json")
        with open(out_path, "w") as f:
            json.dump({
                "experiment": "distributional_congruence_v1",
                "timestamp": datetime.now().isoformat(),
                "model": MODEL_ID,
                "status": "KILLED_BASELINE",
                "baseline_rate": baseline_rate,
            }, f, indent=2)
        return

    print("\nPhase 2: Group by behavioral signature")
    sig_groups = {}
    for h, br in zip(histories, baseline_results):
        if not br["all_correct"]:
            continue
        sig_key = tuple(br["signature"][e] for e in ENTITIES)
        if sig_key not in sig_groups:
            sig_groups[sig_key] = []
        sig_groups[sig_key].append(h)

    print(f"  {len(sig_groups)} distinct behavioral places")

    print("\nPhase 3: Distributional congruence")
    print(f"  JSD threshold for 'same distribution': {JSD_THRESHOLD}")
    congruence_results = []

    for sig_key, group in sig_groups.items():
        if len(group) < 2:
            continue

        vals = group[0]["vals"]
        continuations = make_continuations_for_history(vals)

        for cont in continuations:
            for query_entity in ENTITIES:
                query_prompt_suffix = f"\n{query_entity}:"
                dists = []
                for h in group:
                    extended = h["history"] + cont["text"] + query_prompt_suffix
                    dist = get_next_token_dist(model, tok, extended)
                    dists.append({
                        "tag": h["tag"],
                        "variant": h["variant"],
                        "dist": dist,
                    })

                max_jsd = 0.0
                pair_jsds = []
                for i in range(len(dists)):
                    for j in range(i + 1, len(dists)):
                        jsd = js_dist(dists[i]["dist"], dists[j]["dist"])
                        pair_jsds.append({
                            "pair": f"{dists[i]['variant']}-{dists[j]['variant']}",
                            "jsd": round(jsd, 6),
                        })
                        if jsd > max_jsd:
                            max_jsd = jsd

                congruent = max_jsd < JSD_THRESHOLD
                congruence_results.append({
                    "pre_sig": list(sig_key),
                    "continuation": cont["name"],
                    "query_entity": query_entity,
                    "max_jsd": round(max_jsd, 6),
                    "congruent": congruent,
                    "pair_jsds": pair_jsds,
                })

                status = f"CONGRUENT (max_jsd={max_jsd:.4f})" if congruent else f"DEFECT (max_jsd={max_jsd:.4f})"
                print(f"  sig={sig_key} + {cont['name']:25s} query={query_entity:4s} -> {status}")

    n_tests = len(congruence_results)
    n_congruent = sum(1 for r in congruence_results if r["congruent"])
    n_defects = n_tests - n_congruent
    defect_rate = n_defects / max(n_tests, 1)

    print(f"\n=== SUMMARY ===")
    print(f"  Total distributional congruence tests: {n_tests}")
    print(f"  Congruent (max_jsd < {JSD_THRESHOLD}): {n_congruent}")
    print(f"  Defects: {n_defects}")
    print(f"  Defect rate: {defect_rate:.1%}")

    by_continuation = {}
    for r in congruence_results:
        c = r["continuation"]
        if c not in by_continuation:
            by_continuation[c] = {"total": 0, "defects": 0, "max_jsd_vals": []}
        by_continuation[c]["total"] += 1
        by_continuation[c]["defects"] += 1 if not r["congruent"] else 0
        by_continuation[c]["max_jsd_vals"].append(r["max_jsd"])

    print("\n  By continuation type:")
    for c, stats in by_continuation.items():
        rate = stats["defects"] / max(stats["total"], 1)
        avg_jsd = sum(stats["max_jsd_vals"]) / max(len(stats["max_jsd_vals"]), 1)
        print(f"    {c:25s}: {stats['defects']}/{stats['total']} defects ({rate:.0%}), avg max_jsd={avg_jsd:.4f}")

    for c in by_continuation:
        by_continuation[c]["avg_max_jsd"] = round(
            sum(by_continuation[c]["max_jsd_vals"]) / max(len(by_continuation[c]["max_jsd_vals"]), 1), 6)
        del by_continuation[c]["max_jsd_vals"]

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "distributional_congruence_v1",
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_ID,
            "purpose": "Test whether continuations produce the same DISTRIBUTION (not just greedy answer) across same-place histories",
            "status": "COMPLETE",
            "jsd_threshold": JSD_THRESHOLD,
            "baseline_rate": baseline_rate,
            "n_congruence_tests": n_tests,
            "n_congruent": n_congruent,
            "n_defects": n_defects,
            "defect_rate": defect_rate,
            "by_continuation": by_continuation,
            "congruence_results": congruence_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
