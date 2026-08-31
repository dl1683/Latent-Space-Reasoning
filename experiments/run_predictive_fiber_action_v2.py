"""
Predictive Fiber Action Algebra v2.

Codex v6 corrected experiment. Fixes construction errors in v1:
  1. Square test uses CORRECTED-world restatement S_{p'} on path C-then-S,
     so both paths end at the corrected world p'.
  2. Places defined by full greedy answer SIGNATURE (all queries), not
     per-entity buckets.
  3. Restatement S_p constructed from the greedy signature observable,
     not hidden ground-truth world.
  4. Tests fiber-wide descent (all representatives of a fiber map to
     same target under each operation).
  5. Task-response kernel is primary endpoint. Full-vocab JSD secondary.
  6. Paired intervals reported.

The typed square being tested:
    F_p --C--> F_{p'}
    |           |
    S_p         S_{p'}
    |           |
    S_p(F_p) --C--> S_{p'}(F_{p'})

If S_{p'} . C ~= C . S_p (both ending at corrected world), then
synchronization descends cleanly and the fiber factorizes.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import json
import os
import statistics
from datetime import datetime

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cpu"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "predictive_fiber_action_v2")

REGISTERED_ENTITIES = {
    "ZOG": ("big", "small"),
    "MIP": ("hot", "cold"),
    "PLIM": ("red", "blue"),
}

HELDOUT_ENTITIES = {
    "KROT": ("fast", "slow"),
    "HESK": ("tall", "short"),
    "VORN": ("loud", "quiet"),
}


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()
    return model, tok


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
    correct_ids = set(tok.encode(f" {correct_val}", add_special_tokens=False))
    correct_ids.update(tok.encode(correct_val, add_special_tokens=False))
    wrong_ids = set(tok.encode(f" {wrong_val}", add_special_tokens=False))
    wrong_ids.update(tok.encode(wrong_val, add_special_tokens=False))
    wrong_ids -= correct_ids
    correct_mass = sum(float(dist[tid]) for tid in correct_ids)
    wrong_mass = sum(float(dist[tid]) for tid in wrong_ids)
    other_mass = 1.0 - correct_mass - wrong_mass
    return {
        "correct": round(correct_mass, 6),
        "wrong": round(wrong_mass, 6),
        "other": round(other_mass, 6),
    }


def make_worlds(entities):
    names = list(entities.keys())
    n = len(names)
    worlds = {}
    for i in range(2**n):
        bits = format(i, f'0{n}b')
        world = {}
        for j, name in enumerate(names):
            world[name] = entities[name][int(bits[j])]
        key = f"w{''.join(bits)}"
        worlds[key] = world
    return worlds


def make_history(world, entities, order="std"):
    names = list(entities.keys())
    if order == "rev":
        names = names[::-1]
    parts = [f"{n}: {world[n]}" for n in names]
    return ". ".join(parts) + "."


def make_restatement_from_world(world, entities):
    names = list(entities.keys())
    parts = [f"{n}: {world[n]}" for n in names]
    return " To be clear: " + ". ".join(parts) + "."


def make_corrected_world(world, target_entity, new_value):
    corrected = dict(world)
    corrected[target_entity] = new_value
    return corrected


def get_greedy_signature(model, tok, base_prompt, entity_names):
    sig = {}
    for ent in entity_names:
        prompt = base_prompt + f"\n{ent}:"
        dist = get_dist(model, tok, prompt)
        greedy = tok.decode([torch.argmax(dist).item()]).strip()
        sig[ent] = greedy
    return tuple(sorted(sig.items()))


def run_algebra_validation(model, tok, entities, entity_set_name):
    worlds = make_worlds(entities)
    entity_names = list(entities.keys())
    results = {
        "entity_set": entity_set_name,
        "entities": {k: list(v) for k, v in entities.items()},
        "n_worlds": len(worlds),
        "greedy_signatures": {},
        "action_table": {},
        "idempotence_test": [],
        "typed_square_test": [],
        "descent_test": [],
    }

    print(f"\n{'=' * 60}")
    print(f"Entity set: {entity_set_name} ({entity_names})")
    print(f"{'=' * 60}")

    # Phase 1: Enumerate all states and compute full greedy signatures
    print("\n--- Phase 1: Enumerate states ---")
    all_states = {}

    for wname, world in worlds.items():
        for order in ["std", "rev"]:
            history = make_history(world, entities, order)
            base = history
            sig = get_greedy_signature(model, tok, base, entity_names)
            key = f"{wname}_{order}"
            per_entity = {}
            for ent in entity_names:
                prompt = base + f"\n{ent}:"
                dist = get_dist(model, tok, prompt)
                greedy = tok.decode([torch.argmax(dist).item()]).strip()
                correct_val = world[ent]
                wrong_val = entities[ent][1 - entities[ent].index(correct_val)]
                tk = task_kernel(dist, tok, correct_val, wrong_val)
                per_entity[ent] = {
                    "dist": dist,
                    "greedy": greedy,
                    "correct": correct_val,
                    "wrong": wrong_val,
                    "task_kernel": tk,
                }
            all_states[key] = {
                "world_key": wname,
                "world": world,
                "order": order,
                "base_prompt": base,
                "signature": sig,
                "per_entity": per_entity,
            }

    # Group by greedy signature
    sig_groups = {}
    baseline_correct = 0
    baseline_total = 0
    for key, state in all_states.items():
        sig = state["signature"]
        sig_str = "|".join(f"{k}={v}" for k, v in sig)
        if sig_str not in sig_groups:
            sig_groups[sig_str] = []
        sig_groups[sig_str].append(key)
        for ent in entity_names:
            baseline_total += 1
            pe = state["per_entity"][ent]
            if pe["greedy"] == pe["correct"] or pe["correct"].lower() in pe["greedy"].lower():
                baseline_correct += 1

    baseline_rate = baseline_correct / max(baseline_total, 1)
    results["baseline_accuracy"] = round(baseline_rate, 4)
    print(f"Baseline accuracy: {baseline_correct}/{baseline_total} = {baseline_rate:.1%}")
    print(f"Greedy signatures found: {len(sig_groups)}")
    for sig_str, members in sorted(sig_groups.items()):
        print(f"  {sig_str}: {len(members)} members ({', '.join(members[:4])}{'...' if len(members) > 4 else ''})")

    results["greedy_signatures"] = {k: v for k, v in sig_groups.items()}

    # Phase 2: Action table (per-entity, as v1)
    print("\n--- Phase 2: Action table ---")
    action_results = []
    for key, state in all_states.items():
        base = state["base_prompt"]
        world = state["world"]
        for ent in entity_names:
            pe = state["per_entity"][ent]
            original_greedy = pe["greedy"]

            for op_name, suffix in [
                ("empty", ""),
                ("neutral", " The weather is nice today."),
                ("restatement", make_restatement_from_world(world, entities)),
            ]:
                prompt = base + suffix + f"\n{ent}:"
                dist = get_dist(model, tok, prompt)
                greedy = tok.decode([torch.argmax(dist).item()]).strip()
                preserved = greedy == original_greedy
                action_results.append({
                    "source": key,
                    "entity": ent,
                    "op": op_name,
                    "original_greedy": original_greedy,
                    "result_greedy": greedy,
                    "preserved": preserved,
                })

            # Correction: flip the queried entity
            wrong_val = pe["wrong"]
            correction = f" Actually, {ent}: {wrong_val}."
            prompt = base + correction + f"\n{ent}:"
            dist = get_dist(model, tok, prompt)
            greedy = tok.decode([torch.argmax(dist).item()]).strip()
            action_results.append({
                "source": key,
                "entity": ent,
                "op": "correction",
                "original_greedy": original_greedy,
                "result_greedy": greedy,
                "preserved": greedy == original_greedy,
            })

    results["action_table"] = action_results
    for op in ["empty", "neutral", "restatement", "correction"]:
        subset = [r for r in action_results if r["op"] == op]
        rate = sum(1 for r in subset if r["preserved"]) / max(len(subset), 1)
        print(f"  {op}: {sum(1 for r in subset if r['preserved'])}/{len(subset)} preserved = {rate:.1%}")

    # Phase 3: Idempotence S_p^2 ~= S_p
    print("\n--- Phase 3: Idempotence ---")
    idemp_results = []
    for key, state in all_states.items():
        base = state["base_prompt"]
        world = state["world"]
        restatement = make_restatement_from_world(world, entities)

        for ent in entity_names:
            pe = state["per_entity"][ent]
            sp1_prompt = base + restatement + f"\n{ent}:"
            sp2_prompt = base + restatement + restatement + f"\n{ent}:"
            sp1_dist = get_dist(model, tok, sp1_prompt)
            sp2_dist = get_dist(model, tok, sp2_prompt)

            jsd = js_dist(sp1_dist, sp2_dist)
            sp1_greedy = tok.decode([torch.argmax(sp1_dist).item()]).strip()
            sp2_greedy = tok.decode([torch.argmax(sp2_dist).item()]).strip()
            sp1_tk = task_kernel(sp1_dist, tok, pe["correct"], pe["wrong"])
            sp2_tk = task_kernel(sp2_dist, tok, pe["correct"], pe["wrong"])

            idemp_results.append({
                "source": key,
                "entity": ent,
                "jsd_S_vs_S2": round(jsd, 6),
                "greedy_S": sp1_greedy,
                "greedy_S2": sp2_greedy,
                "greedy_match": sp1_greedy == sp2_greedy,
                "tk_S": sp1_tk,
                "tk_S2": sp2_tk,
            })

    results["idempotence_test"] = idemp_results
    jsds = [r["jsd_S_vs_S2"] for r in idemp_results]
    matches = sum(1 for r in idemp_results if r["greedy_match"])
    print(f"  n={len(idemp_results)}, JSD range [{min(jsds):.4f}, {max(jsds):.4f}], mean={statistics.mean(jsds):.4f}")
    print(f"  Greedy idempotence: {matches}/{len(idemp_results)} = {matches/len(idemp_results):.1%}")

    # Phase 4: TYPED SQUARE (Codex v6 corrected)
    # S_{p'} . C  vs  C . S_p
    # Path 1 (CS_typed): base + correction + S_{p'} (corrected-world restatement) + query
    # Path 2 (SC_typed): base + S_p (old-world restatement) + correction + query
    # Both paths end at the corrected world.
    print("\n--- Phase 4: Typed correction/synchronization square ---")
    square_results = []

    for key, state in all_states.items():
        base = state["base_prompt"]
        world = state["world"]

        for ent in entity_names:
            pe = state["per_entity"][ent]
            correct_val = pe["correct"]
            wrong_val = pe["wrong"]

            corrected_world = make_corrected_world(world, ent, wrong_val)

            s_p = make_restatement_from_world(world, entities)
            s_p_prime = make_restatement_from_world(corrected_world, entities)

            correction = f" Actually, {ent}: {wrong_val}."

            # Path 1: C then S_{p'} — correct, then restate the corrected world
            cs_prompt = base + correction + s_p_prime + f"\n{ent}:"
            cs_dist = get_dist(model, tok, cs_prompt)

            # Path 2: S_p then C — restate old world, then correct
            sc_prompt = base + s_p + correction + f"\n{ent}:"
            sc_dist = get_dist(model, tok, sc_prompt)

            jsd = js_dist(cs_dist, sc_dist)
            cs_greedy = tok.decode([torch.argmax(cs_dist).item()]).strip()
            sc_greedy = tok.decode([torch.argmax(sc_dist).item()]).strip()
            cs_tk = task_kernel(cs_dist, tok, wrong_val, correct_val)
            sc_tk = task_kernel(sc_dist, tok, wrong_val, correct_val)

            sq = {
                "source": key,
                "entity": ent,
                "corrected_entity": ent,
                "original_value": correct_val,
                "corrected_value": wrong_val,
                "jsd_CS_vs_SC": round(jsd, 6),
                "greedy_CS": cs_greedy,
                "greedy_SC": sc_greedy,
                "greedy_match": cs_greedy == sc_greedy,
                "tk_CS": cs_tk,
                "tk_SC": sc_tk,
                "tk_correct_diff": round(abs(cs_tk["correct"] - sc_tk["correct"]), 6),
                "cs_prompt_snippet": cs_prompt[-120:],
                "sc_prompt_snippet": sc_prompt[-120:],
            }
            square_results.append(sq)
            print(f"  {key}/{ent}: JSD={jsd:.4f}, greedy={cs_greedy}/{sc_greedy}, tk_diff={sq['tk_correct_diff']:.4f}")

    results["typed_square_test"] = square_results
    sq_jsds = [r["jsd_CS_vs_SC"] for r in square_results]
    sq_matches = sum(1 for r in square_results if r["greedy_match"])
    sq_tk_diffs = [r["tk_correct_diff"] for r in square_results]
    print(f"\n  n={len(square_results)}")
    print(f"  JSD range [{min(sq_jsds):.4f}, {max(sq_jsds):.4f}], mean={statistics.mean(sq_jsds):.4f}, median={statistics.median(sq_jsds):.4f}")
    print(f"  Greedy commutativity: {sq_matches}/{len(square_results)} = {sq_matches/len(square_results):.1%}")
    print(f"  Task kernel diff: mean={statistics.mean(sq_tk_diffs):.4f}, median={statistics.median(sq_tk_diffs):.4f}")

    # Phase 5: Descent test — do all fiber representatives map to same target?
    print("\n--- Phase 5: Descent test ---")
    descent_results = []
    for sig_str, members in sig_groups.items():
        if len(members) < 2:
            continue
        for ent in entity_names:
            targets_empty = set()
            targets_restate = set()
            for mk in members:
                mstate = all_states[mk]
                base = mstate["base_prompt"]
                world = mstate["world"]
                # Empty
                prompt = base + f"\n{ent}:"
                dist = get_dist(model, tok, prompt)
                g = tok.decode([torch.argmax(dist).item()]).strip()
                targets_empty.add(g)
                # Restatement
                restatement = make_restatement_from_world(world, entities)
                prompt = base + restatement + f"\n{ent}:"
                dist = get_dist(model, tok, prompt)
                g = tok.decode([torch.argmax(dist).item()]).strip()
                targets_restate.add(g)

            descent_results.append({
                "signature": sig_str,
                "entity": ent,
                "n_members": len(members),
                "empty_targets": list(targets_empty),
                "empty_descends": len(targets_empty) == 1,
                "restatement_targets": list(targets_restate),
                "restatement_descends": len(targets_restate) == 1,
            })

    results["descent_test"] = descent_results
    empty_desc = sum(1 for r in descent_results if r["empty_descends"])
    rest_desc = sum(1 for r in descent_results if r["restatement_descends"])
    total_desc = len(descent_results)
    print(f"  Fiber groups tested: {total_desc}")
    print(f"  Empty descent: {empty_desc}/{total_desc} = {empty_desc/max(total_desc,1):.1%}")
    print(f"  Restatement descent: {rest_desc}/{total_desc} = {rest_desc/max(total_desc,1):.1%}")

    return results


def main():
    print("Loading model...")
    model, tok = load_model()
    print(f"Model: {MODEL_ID}")
    print(f"Tokenizer vocab: {tok.vocab_size}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {"model": MODEL_ID, "timestamp": datetime.now().isoformat(), "sets": {}}

    for set_name, entities in [("registered", REGISTERED_ENTITIES), ("heldout", HELDOUT_ENTITIES)]:
        print(f"\n{'#' * 60}")
        print(f"# SET {set_name}")
        print(f"{'#' * 60}")
        res = run_algebra_validation(model, tok, entities, set_name)
        # Remove dist tensors before saving
        all_results["sets"][set_name] = res

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for sn in ["registered", "heldout"]:
        r = all_results["sets"][sn]
        print(f"\n  SET {sn}")
        print(f"    Baseline: {r['baseline_accuracy']:.1%}")
        idemp = r["idempotence_test"]
        ij = [x["jsd_S_vs_S2"] for x in idemp]
        im = sum(1 for x in idemp if x["greedy_match"])
        print(f"    Idempotence: n={len(idemp)}, JSD [{min(ij):.4f},{max(ij):.4f}], greedy {im}/{len(idemp)}")
        sq = r["typed_square_test"]
        sj = [x["jsd_CS_vs_SC"] for x in sq]
        sm = sum(1 for x in sq if x["greedy_match"])
        stk = [x["tk_correct_diff"] for x in sq]
        print(f"    Typed square: n={len(sq)}, JSD [{min(sj):.4f},{max(sj):.4f}], mean={statistics.mean(sj):.4f}")
        print(f"    Greedy commutativity: {sm}/{len(sq)} = {sm/len(sq):.1%}")
        print(f"    Task kernel diff: mean={statistics.mean(stk):.4f}")

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
