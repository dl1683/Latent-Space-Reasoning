"""
Predictive Fiber and Synchronization v1.

Codex-directed (direction v4): the decisive experiment.

Question: Is the distributional residual inside a greedy fiber PREDICTIVE
(predicts future task behavior) or PRESENTATION LEAKAGE (static stylistic
mass that doesn't affect downstream decisions)?

Design:
  Three pair classes within each greedy fiber (same argmax answers):
    1. Benign presentation: same facts, different order (std vs rev)
    2. History pairs: same final facts, different write history (std vs dup)
    3. Positive control: genuinely different fact assignments (different fibers)

  After each continuation, measure:
    - Full-vocabulary sqrt(JSD) between pair distributions
    - Task kernel: probability mass on (correct_value, wrong_value, OTHER)
    - Whether the fiber-internal distance predicts which history variant
      produced the output (via a canonical restatement that should reset fibers)

  Key estimand: does history-pair distance in the task law EXCEED benign
  presentation distance and SURVIVE held-out continuation?

  Outcomes:
    - Residual survives in task law → refine the quotient
    - Residual collapses after canonical restatement → presentation only
    - Effect doesn't survive held-out names → prompt-family artifact
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import json
import os
from datetime import datetime

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cpu"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "predictive_fiber_v1")

WORLDS = {
    "w000": {"ZOG": "big", "MIP": "hot", "PLIM": "red"},
    "w100": {"ZOG": "small", "MIP": "hot", "PLIM": "red"},
    "w010": {"ZOG": "big", "MIP": "cold", "PLIM": "red"},
    "w001": {"ZOG": "big", "MIP": "hot", "PLIM": "blue"},
}

ENTITIES = ["ZOG", "MIP", "PLIM"]


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()
    return model, tok


def make_history_variants(world):
    vals = world
    std = f"ZOG: {vals['ZOG']}. MIP: {vals['MIP']}. PLIM: {vals['PLIM']}."
    rev = f"PLIM: {vals['PLIM']}. MIP: {vals['MIP']}. ZOG: {vals['ZOG']}."
    dup = f"ZOG: {vals['ZOG']}. MIP: {vals['MIP']}. PLIM: {vals['PLIM']}. ZOG: {vals['ZOG']}."
    return {"std": std, "rev": rev, "dup": dup}


def get_dist(model, tok, prompt):
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


def shannon_entropy(p):
    eps = 1e-10
    return float(-(p * (p + eps).log2()).sum())


def task_kernel(dist, tok, correct_val, wrong_val):
    correct_ids = tok.encode(f" {correct_val}", add_special_tokens=False)
    wrong_ids = tok.encode(f" {wrong_val}", add_special_tokens=False)

    correct_mass = sum(float(dist[tid]) for tid in correct_ids)
    wrong_mass = sum(float(dist[tid]) for tid in wrong_ids)

    bare_correct = tok.encode(correct_val, add_special_tokens=False)
    bare_wrong = tok.encode(wrong_val, add_special_tokens=False)
    correct_mass += sum(float(dist[tid]) for tid in bare_correct if tid not in correct_ids)
    wrong_mass += sum(float(dist[tid]) for tid in bare_wrong if tid not in wrong_ids)

    other_mass = 1.0 - correct_mass - wrong_mass
    return {
        "correct": round(correct_mass, 6),
        "wrong": round(wrong_mass, 6),
        "other": round(other_mass, 6),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, tok = load_model()

    VALUES_MAP = {"ZOG": ("big", "small"), "MIP": ("hot", "cold"), "PLIM": ("red", "blue")}

    continuations = [
        {"name": "empty", "text": ""},
        {"name": "neutral", "text": " The sky is clear."},
        {"name": "repeat_zog", "text": " ZOG: big."},
        {"name": "correct_zog", "text": " Actually, ZOG: small."},
        {"name": "new_entity", "text": " KROT: fast."},
        {"name": "canonical_restatement",
         "text": " To be clear: ZOG: big. MIP: hot. PLIM: red."},
    ]

    all_results = []
    for query_entity in ENTITIES:
        correct_val = WORLDS["w000"][query_entity]
        wrong_val = VALUES_MAP[query_entity][1 - VALUES_MAP[query_entity].index(correct_val)]
        query_suffix = f"\n{query_entity}:"

        print(f"\n=== Query: {query_entity} (correct={correct_val}, wrong={wrong_val}) ===")

        for cont in continuations:
            print(f"\n  Continuation: {cont['name']}")

            fiber_variants = {}
            for wname, world in WORLDS.items():
                if world[query_entity] != correct_val:
                    continue
                variants = make_history_variants(world)
                for vname, history in variants.items():
                    prompt = history + cont["text"] + query_suffix
                    dist = get_dist(model, tok, prompt)
                    key = f"{wname}_{vname}"
                    greedy = tok.decode([torch.argmax(dist).item()]).strip()
                    entropy = shannon_entropy(dist)
                    tk = task_kernel(dist, tok, correct_val, wrong_val)
                    fiber_variants[key] = {
                        "world": wname,
                        "variant": vname,
                        "prompt": prompt,
                        "dist": dist,
                        "greedy": greedy,
                        "entropy": round(entropy, 4),
                        "task_kernel": tk,
                    }
                    print(f"    {key}: greedy={greedy}, H={entropy:.2f}, kernel={tk}")

            benign_pairs = []
            history_pairs = []
            cross_world_pairs = []

            keys = list(fiber_variants.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    ki, kj = keys[i], keys[j]
                    fi, fj = fiber_variants[ki], fiber_variants[kj]
                    jsd = js_dist(fi["dist"], fj["dist"])
                    tk_diff = abs(fi["task_kernel"]["correct"] - fj["task_kernel"]["correct"])
                    pair_info = {
                        "a": ki, "b": kj,
                        "jsd": round(jsd, 6),
                        "task_kernel_diff": round(tk_diff, 6),
                        "entropy_diff": round(abs(fi["entropy"] - fj["entropy"]), 4),
                    }

                    if fi["world"] == fj["world"]:
                        vset = {fi["variant"], fj["variant"]}
                        if vset == {"std", "rev"}:
                            pair_info["class"] = "benign_presentation"
                            benign_pairs.append(pair_info)
                        elif "dup" in vset:
                            pair_info["class"] = "history_pair"
                            history_pairs.append(pair_info)
                    else:
                        if fi["variant"] == fj["variant"]:
                            pair_info["class"] = "cross_world"
                            cross_world_pairs.append(pair_info)

            def avg_jsd(pairs):
                if not pairs:
                    return 0
                return round(sum(p["jsd"] for p in pairs) / len(pairs), 6)

            benign_avg = avg_jsd(benign_pairs)
            history_avg = avg_jsd(history_pairs)
            cross_avg = avg_jsd(cross_world_pairs)

            print(f"\n    Benign presentation pairs ({len(benign_pairs)}): avg JSD = {benign_avg}")
            print(f"    History pairs ({len(history_pairs)}): avg JSD = {history_avg}")
            print(f"    Cross-world pairs ({len(cross_world_pairs)}): avg JSD = {cross_avg}")

            exceeds = history_avg > benign_avg if benign_pairs and history_pairs else None
            print(f"    History > Benign? {exceeds}")

            result = {
                "query_entity": query_entity,
                "correct_val": correct_val,
                "wrong_val": wrong_val,
                "continuation": cont["name"],
                "n_variants": len(fiber_variants),
                "benign_presentation": {
                    "n_pairs": len(benign_pairs),
                    "avg_jsd": benign_avg,
                    "pairs": benign_pairs,
                },
                "history_pairs": {
                    "n_pairs": len(history_pairs),
                    "avg_jsd": history_avg,
                    "pairs": history_pairs,
                },
                "cross_world": {
                    "n_pairs": len(cross_world_pairs),
                    "avg_jsd": cross_avg,
                    "pairs": cross_world_pairs,
                },
                "history_exceeds_benign": exceeds,
                "variants": {
                    k: {
                        "world": v["world"],
                        "variant": v["variant"],
                        "greedy": v["greedy"],
                        "entropy": v["entropy"],
                        "task_kernel": v["task_kernel"],
                    }
                    for k, v in fiber_variants.items()
                },
            }
            all_results.append(result)

    print("\n\n=== SUMMARY ===")
    for cont_name in [c["name"] for c in continuations]:
        results_for_cont = [r for r in all_results if r["continuation"] == cont_name]
        benign_jsds = [r["benign_presentation"]["avg_jsd"] for r in results_for_cont if r["benign_presentation"]["avg_jsd"] > 0]
        history_jsds = [r["history_pairs"]["avg_jsd"] for r in results_for_cont if r["history_pairs"]["avg_jsd"] > 0]
        cross_jsds = [r["cross_world"]["avg_jsd"] for r in results_for_cont if r["cross_world"]["avg_jsd"] > 0]

        benign_mean = sum(benign_jsds) / max(len(benign_jsds), 1)
        history_mean = sum(history_jsds) / max(len(history_jsds), 1)
        cross_mean = sum(cross_jsds) / max(len(cross_jsds), 1)

        exceeds_count = sum(1 for r in results_for_cont if r["history_exceeds_benign"] is True)
        total = sum(1 for r in results_for_cont if r["history_exceeds_benign"] is not None)

        print(f"  {cont_name:25s}: benign={benign_mean:.4f}  history={history_mean:.4f}  cross={cross_mean:.4f}  history>benign: {exceeds_count}/{total}")

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "predictive_fiber_v1",
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_ID,
            "purpose": "Test whether distributional residual inside greedy fibers is predictive state or presentation leakage. Codex direction v4.",
            "design": "Three pair classes: benign_presentation (std vs rev), history_pair (std/rev vs dup), cross_world (different fact worlds). Continuations: empty, neutral, repeat, correction, new entity, canonical restatement. Key estimand: history_pair JSD > benign_presentation JSD.",
            "n_results": len(all_results),
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
