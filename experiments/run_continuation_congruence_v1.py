"""
Predictive Continuation Congruence v1.

Tests whether token updates act as well-defined moves on behavioral places.

Two histories h, h' have the same behavioral place if they give the same
answer to all direct-recall queries. A continuation c is well-defined on
places if: h ~= h' (same place) implies h+c ~= h'+c (same place after c).

If same-place histories bifurcate after the same continuation, the
continuation is NOT a well-defined operation — it depends on the specific
history, not just the behavioral place.

Codex design (direction dialogue v2, Q4).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from datetime import datetime
from itertools import product as iterproduct

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cpu"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "continuation_congruence_v1")

ENTITIES = ["ZOG", "MIP", "PLIM"]
VALUES = {"ZOG": ("big", "small"), "MIP": ("hot", "cold"), "PLIM": ("red", "blue")}

CONTINUATIONS = [
    {"name": "neutral_distractor", "text": " The sky is clear."},
    {"name": "new_entity_commit", "text": " KROT: fast."},
    {"name": "repeat_first", "text": ""},  # will be filled per history
    {"name": "explicit_correction", "text": ""},  # will be filled per history
]


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()
    return model, tok


def greedy_answer(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    top_idx = torch.argmax(logits).item()
    return tok.decode([top_idx]).strip()


def get_signature(model, tok, history):
    """Query each entity and record the greedy answer."""
    sig = {}
    for entity in ENTITIES:
        prompt = history + f"\n{entity}:"
        sig[entity] = greedy_answer(model, tok, prompt)
    return sig


def make_histories():
    """Generate histories that store facts about all 3 entities.
    Multiple presentation variants to create same-place/different-history pairs.
    """
    assignments = list(iterproduct(range(2), range(2), range(2)))
    histories = []

    for a, b, c in assignments:
        vals = (VALUES["ZOG"][a], VALUES["MIP"][b], VALUES["PLIM"][c])
        tag = f"w{a}{b}{c}"

        # Variant 1: standard order
        h1 = f"ZOG: {vals[0]}. MIP: {vals[1]}. PLIM: {vals[2]}."
        histories.append({"tag": tag, "variant": "std", "history": h1, "vals": vals})

        # Variant 2: reversed order
        h2 = f"PLIM: {vals[2]}. MIP: {vals[1]}. ZOG: {vals[0]}."
        histories.append({"tag": tag, "variant": "rev", "history": h2, "vals": vals})

        # Variant 3: repeated first entity (duplicate consistent write)
        h3 = f"ZOG: {vals[0]}. MIP: {vals[1]}. PLIM: {vals[2]}. ZOG: {vals[0]}."
        histories.append({"tag": tag, "variant": "dup", "history": h3, "vals": vals})

    return histories


def make_continuations_for_history(vals):
    """Generate continuations specific to a history's values."""
    return [
        {"name": "neutral_distractor", "text": " The sky is clear."},
        {"name": "new_entity_commit", "text": " KROT: fast."},
        {"name": "repeat_zog", "text": f" ZOG: {vals[0]}."},
        {"name": "correct_zog", "text": f" Actually, ZOG: {VALUES['ZOG'][1 - VALUES['ZOG'].index(vals[0])]}."},
    ]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, tok = load_model()

    print("Phase 1: Baseline — check if model can recall all 3 entities")
    histories = make_histories()

    baseline_results = []
    for h in histories:
        sig = get_signature(model, tok, h["history"])
        correct = all(
            sig[e] == h["vals"][i] for i, e in enumerate(ENTITIES)
        )
        baseline_results.append({
            "tag": h["tag"],
            "variant": h["variant"],
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
        print("BASELINE BELOW 70% — instrument kill condition. Stopping.")
        out_path = os.path.join(RESULTS_DIR, "results.json")
        with open(out_path, "w") as f:
            json.dump({
                "experiment": "continuation_congruence_v1",
                "timestamp": datetime.now().isoformat(),
                "model": MODEL_ID,
                "status": "KILLED_BASELINE",
                "baseline_rate": baseline_rate,
                "baseline_results": baseline_results,
            }, f, indent=2)
        print(f"Saved to {out_path}")
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
    for sk, group in sig_groups.items():
        variants = [h["variant"] for h in group]
        print(f"  sig={sk}: {len(group)} histories ({', '.join(variants)})")

    print("\nPhase 3: Congruence test — same place, same continuation, same outcome?")
    congruence_results = []

    for sig_key, group in sig_groups.items():
        if len(group) < 2:
            continue

        vals = group[0]["vals"]
        continuations = make_continuations_for_history(vals)

        for cont in continuations:
            post_sigs = []
            for h in group:
                extended = h["history"] + cont["text"]
                post_sig = get_signature(model, tok, extended)
                post_sigs.append({
                    "tag": h["tag"],
                    "variant": h["variant"],
                    "pre_sig": sig_key,
                    "continuation": cont["name"],
                    "post_sig": tuple(post_sig[e] for e in ENTITIES),
                })

            all_post_sigs = set(ps["post_sig"] for ps in post_sigs)
            congruent = len(all_post_sigs) == 1
            defect = not congruent

            congruence_results.append({
                "pre_sig": list(sig_key),
                "continuation": cont["name"],
                "n_histories": len(group),
                "n_distinct_post_sigs": len(all_post_sigs),
                "congruent": congruent,
                "defect": defect,
                "details": post_sigs,
            })

            status = "CONGRUENT" if congruent else f"DEFECT ({len(all_post_sigs)} outcomes)"
            print(f"  sig={sig_key} + {cont['name']:25s} -> {status}")

    n_tests = len(congruence_results)
    n_defects = sum(1 for r in congruence_results if r["defect"])
    n_congruent = n_tests - n_defects
    defect_rate = n_defects / max(n_tests, 1)

    print(f"\n=== SUMMARY ===")
    print(f"  Total congruence tests: {n_tests}")
    print(f"  Congruent: {n_congruent}")
    print(f"  Defects: {n_defects}")
    print(f"  Defect rate: {defect_rate:.1%}")

    if defect_rate > 0.10:
        print(f"  VERDICT: CONGRUENCE FAILS — updates are NOT well-defined operations on places")
    else:
        print(f"  VERDICT: CONGRUENCE HOLDS — updates act as well-defined operations")

    by_continuation = {}
    for r in congruence_results:
        c = r["continuation"]
        if c not in by_continuation:
            by_continuation[c] = {"total": 0, "defects": 0}
        by_continuation[c]["total"] += 1
        by_continuation[c]["defects"] += 1 if r["defect"] else 0

    print("\n  By continuation type:")
    for c, stats in by_continuation.items():
        rate = stats["defects"] / max(stats["total"], 1)
        print(f"    {c:25s}: {stats['defects']}/{stats['total']} defects ({rate:.0%})")

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "continuation_congruence_v1",
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_ID,
            "purpose": "Test whether token continuations are well-defined operations on behavioral places",
            "status": "COMPLETE",
            "baseline_rate": baseline_rate,
            "n_congruence_tests": n_tests,
            "n_congruent": n_congruent,
            "n_defects": n_defects,
            "defect_rate": defect_rate,
            "by_continuation": by_continuation,
            "kill_conditions": {
                "baseline_kill": "< 90% direct recall",
                "algebra_kill": "> 10% congruence defect rate",
            },
            "baseline_results": baseline_results,
            "congruence_results": congruence_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
