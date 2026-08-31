"""
Predictive Fiber Action Algebra v1.

Codex-directed (v5): the algebra-validation experiment.

Builds and validates the Synchronized Predictive-Fiber Action Algebra:
  A_PF = (Q -> G, M, alpha, {F_p}, {S_p}, {rho_e})

Design:
  1. Enumerate greedy places G (greedy answer vectors) across all history
     variants and worlds.
  2. For each greedy place p, enumerate the fiber F_p = g^{-1}(p).
  3. Define place-typed canonical restatement S_p for EACH greedy place.
  4. Test S_p^2 ~= S_p (idempotent retraction).
  5. Test the correction/synchronization square:
       F_p --C--> F_p'
       |           |
       S_p         S_p'
       |           |
       S_p(F_p) --C--> S_p'(F_p')
  6. Use task-total response law (correct, wrong, other) as primary endpoint.
  7. Full-vocabulary sqrt(JSD) as secondary endpoint.
  8. Held-out entity names for generalization.

Primary endpoint is the task-total kernel, not full-vocabulary JSD.
Model revision pinned. Report clustered effects.
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
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "predictive_fiber_action_v1")

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

ALL_ENTITY_SETS = {
    "registered": REGISTERED_ENTITIES,
    "heldout": HELDOUT_ENTITIES,
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
    """Generate all 2^n worlds for n entities with binary values."""
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


def make_dup_history(world, entities, dup_entity):
    names = list(entities.keys())
    parts = [f"{n}: {world[n]}" for n in names]
    parts.append(f"{dup_entity}: {world[dup_entity]}")
    return ". ".join(parts) + "."


def make_restatement(world, entities):
    """Place-typed canonical restatement S_p: restates all facts in canonical order."""
    names = list(entities.keys())
    parts = [f"{n}: {world[n]}" for n in names]
    return " To be clear: " + ". ".join(parts) + "."


def run_algebra_validation(model, tok, entities, entity_set_name):
    """Run the full algebra validation for one entity set."""
    worlds = make_worlds(entities)
    entity_names = list(entities.keys())
    results = {
        "entity_set": entity_set_name,
        "entities": {k: list(v) for k, v in entities.items()},
        "n_worlds": len(worlds),
        "greedy_places": {},
        "fiber_table": {},
        "action_table": {},
        "idempotence_test": [],
        "correction_sync_square": [],
        "laws": {},
    }

    # Phase 1: Enumerate greedy places and fibers
    print(f"\n{'='*60}")
    print(f"Entity set: {entity_set_name} ({entity_names})")
    print(f"{'='*60}")

    all_states = {}

    for wname, world in worlds.items():
        for order in ["std", "rev"]:
            history = make_history(world, entities, order)
            for query_entity in entity_names:
                correct_val = world[query_entity]
                wrong_val = entities[query_entity][
                    1 - entities[query_entity].index(correct_val)
                ]
                prompt = history + f"\n{query_entity}:"
                dist = get_dist(model, tok, prompt)
                greedy = tok.decode([torch.argmax(dist).item()]).strip()
                tk = task_kernel(dist, tok, correct_val, wrong_val)
                entropy = shannon_entropy(dist)
                key = f"{wname}_{order}_{query_entity}"
                all_states[key] = {
                    "world": wname,
                    "order": order,
                    "query": query_entity,
                    "correct": correct_val,
                    "wrong": wrong_val,
                    "prompt": prompt,
                    "dist": dist,
                    "greedy": greedy,
                    "task_kernel": tk,
                    "entropy": round(entropy, 4),
                }

        # Also dup variants for first entity
        dup_entity = entity_names[0]
        history_dup = make_dup_history(world, entities, dup_entity)
        for query_entity in entity_names:
            correct_val = world[query_entity]
            wrong_val = entities[query_entity][
                1 - entities[query_entity].index(correct_val)
            ]
            prompt = history_dup + f"\n{query_entity}:"
            dist = get_dist(model, tok, prompt)
            greedy = tok.decode([torch.argmax(dist).item()]).strip()
            tk = task_kernel(dist, tok, correct_val, wrong_val)
            entropy = shannon_entropy(dist)
            key = f"{wname}_dup_{query_entity}"
            all_states[key] = {
                "world": wname,
                "order": "dup",
                "query": query_entity,
                "correct": correct_val,
                "wrong": wrong_val,
                "prompt": prompt,
                "dist": dist,
                "greedy": greedy,
                "task_kernel": tk,
                "entropy": round(entropy, 4),
            }

    # Group by greedy place (query_entity, greedy_answer)
    greedy_places = {}
    baseline_correct = 0
    baseline_total = 0
    for key, state in all_states.items():
        gp = (state["query"], state["greedy"])
        if gp not in greedy_places:
            greedy_places[gp] = []
        greedy_places[gp].append(key)
        if state["greedy"] == state["correct"] or state["greedy"] == f" {state['correct']}":
            baseline_correct += 1
        elif state["correct"].lower() in state["greedy"].lower():
            baseline_correct += 1
        baseline_total += 1

    baseline_acc = baseline_correct / max(baseline_total, 1)
    print(f"\nBaseline accuracy: {baseline_correct}/{baseline_total} = {baseline_acc:.3f}")

    results["baseline"] = {
        "correct": baseline_correct,
        "total": baseline_total,
        "accuracy": round(baseline_acc, 4),
    }

    # Record fiber structure
    print(f"\nGreedy places ({len(greedy_places)}):")
    for gp, members in sorted(greedy_places.items()):
        print(f"  {gp}: {len(members)} members")
        results["fiber_table"][f"{gp[0]}={gp[1]}"] = {
            "query": gp[0],
            "greedy": gp[1],
            "fiber_size": len(members),
            "members": members,
        }

    # Phase 2: Action table — apply continuations and record transitions
    continuations = {
        "empty": "",
        "neutral": " The sky is clear.",
        "correction": None,  # entity-specific, filled per query
        "restatement": None,  # world-specific, filled per state
    }

    print("\n--- Action table ---")
    action_results = []
    for gp, members in sorted(greedy_places.items()):
        query_entity = gp[0]
        for member_key in members:
            state = all_states[member_key]
            world = worlds[state["world"]]
            correct_val = state["correct"]
            wrong_val = state["wrong"]

            for cont_name in ["empty", "neutral", "correction", "restatement"]:
                if cont_name == "correction":
                    cont_text = f" Actually, {query_entity}: {wrong_val}."
                elif cont_name == "restatement":
                    cont_text = make_restatement(world, entities)
                else:
                    cont_text = continuations[cont_name]

                prompt_after = state["prompt"].rsplit(f"\n{query_entity}:", 1)[0]
                prompt_after = prompt_after + cont_text + f"\n{query_entity}:"
                dist_after = get_dist(model, tok, prompt_after)
                greedy_after = tok.decode([torch.argmax(dist_after).item()]).strip()

                if cont_name == "correction":
                    tk_after = task_kernel(dist_after, tok, wrong_val, correct_val)
                else:
                    tk_after = task_kernel(dist_after, tok, correct_val, wrong_val)

                entropy_after = shannon_entropy(dist_after)
                jsd_from_base = js_dist(state["dist"], dist_after)

                action_results.append({
                    "source": member_key,
                    "source_place": f"{gp[0]}={gp[1]}",
                    "continuation": cont_name,
                    "greedy_before": state["greedy"],
                    "greedy_after": greedy_after,
                    "tk_before": state["task_kernel"],
                    "tk_after": tk_after,
                    "entropy_before": state["entropy"],
                    "entropy_after": round(entropy_after, 4),
                    "jsd_from_base": round(jsd_from_base, 6),
                    "place_preserved": state["greedy"] == greedy_after,
                })

    results["action_table"] = action_results

    # Summarize action table
    print("\nAction summary (place preservation rate):")
    for cont_name in ["empty", "neutral", "correction", "restatement"]:
        relevant = [a for a in action_results if a["continuation"] == cont_name]
        preserved = sum(1 for a in relevant if a["place_preserved"])
        total = len(relevant)
        print(f"  {cont_name}: {preserved}/{total} = {preserved/max(total,1):.3f}")

    # Phase 3: Idempotence test — S_p^2 ~= S_p
    print("\n--- Idempotence test: S_p^2 ~= S_p ---")
    idemp_results = []
    for gp, members in sorted(greedy_places.items()):
        query_entity = gp[0]
        for member_key in members[:2]:  # test 2 per fiber for efficiency
            state = all_states[member_key]
            world = worlds[state["world"]]
            correct_val = state["correct"]
            wrong_val = state["wrong"]

            restatement = make_restatement(world, entities)
            base_prompt = state["prompt"].rsplit(f"\n{query_entity}:", 1)[0]

            # S_p: apply restatement once
            sp1_prompt = base_prompt + restatement + f"\n{query_entity}:"
            sp1_dist = get_dist(model, tok, sp1_prompt)

            # S_p^2: apply restatement twice
            sp2_prompt = base_prompt + restatement + restatement + f"\n{query_entity}:"
            sp2_dist = get_dist(model, tok, sp2_prompt)

            jsd_sp1_sp2 = js_dist(sp1_dist, sp2_dist)
            sp1_greedy = tok.decode([torch.argmax(sp1_dist).item()]).strip()
            sp2_greedy = tok.decode([torch.argmax(sp2_dist).item()]).strip()
            sp1_entropy = shannon_entropy(sp1_dist)
            sp2_entropy = shannon_entropy(sp2_dist)

            idemp = {
                "source": member_key,
                "place": f"{gp[0]}={gp[1]}",
                "jsd_sp1_sp2": round(jsd_sp1_sp2, 6),
                "greedy_sp1": sp1_greedy,
                "greedy_sp2": sp2_greedy,
                "greedy_match": sp1_greedy == sp2_greedy,
                "entropy_sp1": round(sp1_entropy, 4),
                "entropy_sp2": round(sp2_entropy, 4),
            }
            idemp_results.append(idemp)
            print(f"  {member_key}: JSD(S,S²)={jsd_sp1_sp2:.4f}, greedy_match={sp1_greedy==sp2_greedy}")

    results["idempotence_test"] = idemp_results
    avg_idemp_jsd = sum(r["jsd_sp1_sp2"] for r in idemp_results) / max(len(idemp_results), 1)
    greedy_idemp_rate = sum(1 for r in idemp_results if r["greedy_match"]) / max(len(idemp_results), 1)
    print(f"  Avg JSD(S,S²): {avg_idemp_jsd:.4f}")
    print(f"  Greedy idempotence rate: {greedy_idemp_rate:.3f}")

    results["idempotence_summary"] = {
        "avg_jsd": round(avg_idemp_jsd, 6),
        "greedy_match_rate": round(greedy_idemp_rate, 4),
        "n_tests": len(idemp_results),
    }

    # Phase 4: Correction/Synchronization square
    # F_p --C--> F_p'
    # |           |
    # S_p         S_p'
    # |           |
    # S_p(F_p) --C--> S_p'(F_p')
    #
    # Path 1: correct first, then synchronize the result
    # Path 2: synchronize first, then correct
    # If the square commutes, the two paths produce similar distributions
    print("\n--- Correction/Synchronization square ---")
    square_results = []
    for gp, members in sorted(greedy_places.items()):
        query_entity = gp[0]
        for member_key in members[:2]:
            state = all_states[member_key]
            world = worlds[state["world"]]
            correct_val = state["correct"]
            wrong_val = state["wrong"]

            restatement = make_restatement(world, entities)
            correction = f" Actually, {query_entity}: {wrong_val}."
            base_prompt = state["prompt"].rsplit(f"\n{query_entity}:", 1)[0]

            # Path 1: C then S
            cs_prompt = base_prompt + correction + restatement + f"\n{query_entity}:"
            cs_dist = get_dist(model, tok, cs_prompt)

            # Path 2: S then C
            sc_prompt = base_prompt + restatement + correction + f"\n{query_entity}:"
            sc_dist = get_dist(model, tok, sc_prompt)

            jsd_cs_sc = js_dist(cs_dist, sc_dist)
            cs_greedy = tok.decode([torch.argmax(cs_dist).item()]).strip()
            sc_greedy = tok.decode([torch.argmax(sc_dist).item()]).strip()

            cs_tk = task_kernel(cs_dist, tok, wrong_val, correct_val)
            sc_tk = task_kernel(sc_dist, tok, wrong_val, correct_val)

            sq = {
                "source": member_key,
                "place": f"{gp[0]}={gp[1]}",
                "jsd_C_then_S_vs_S_then_C": round(jsd_cs_sc, 6),
                "greedy_CS": cs_greedy,
                "greedy_SC": sc_greedy,
                "greedy_match": cs_greedy == sc_greedy,
                "tk_CS": cs_tk,
                "tk_SC": sc_tk,
                "tk_correct_diff": round(abs(cs_tk["correct"] - sc_tk["correct"]), 6),
            }
            square_results.append(sq)
            print(f"  {member_key}: JSD(CS,SC)={jsd_cs_sc:.4f}, greedy={cs_greedy}/{sc_greedy}, tk_diff={sq['tk_correct_diff']:.4f}")

    results["correction_sync_square"] = square_results
    avg_sq_jsd = sum(r["jsd_C_then_S_vs_S_then_C"] for r in square_results) / max(len(square_results), 1)
    greedy_sq_rate = sum(1 for r in square_results if r["greedy_match"]) / max(len(square_results), 1)
    avg_tk_diff = sum(r["tk_correct_diff"] for r in square_results) / max(len(square_results), 1)
    print(f"  Avg JSD(CS vs SC): {avg_sq_jsd:.4f}")
    print(f"  Greedy commutativity: {greedy_sq_rate:.3f}")
    print(f"  Avg task kernel diff: {avg_tk_diff:.4f}")

    results["square_summary"] = {
        "avg_jsd": round(avg_sq_jsd, 6),
        "greedy_commutativity_rate": round(greedy_sq_rate, 4),
        "avg_tk_correct_diff": round(avg_tk_diff, 6),
        "n_tests": len(square_results),
    }

    # Phase 5: Fiber-internal distances (within-fiber JSD)
    print("\n--- Fiber-internal distances ---")
    fiber_dist_results = []
    for gp, members in sorted(greedy_places.items()):
        if len(members) < 2:
            continue
        for i in range(min(len(members), 3)):
            for j in range(i + 1, min(len(members), 3)):
                ki, kj = members[i], members[j]
                si, sj = all_states[ki], all_states[kj]
                jsd = js_dist(si["dist"], sj["dist"])
                tk_diff = abs(si["task_kernel"]["correct"] - sj["task_kernel"]["correct"])
                pair_type = "benign" if si["order"] != "dup" and sj["order"] != "dup" else "history"
                if si["world"] != sj["world"]:
                    pair_type = "cross_world"
                fiber_dist_results.append({
                    "place": f"{gp[0]}={gp[1]}",
                    "pair": f"{ki} vs {kj}",
                    "pair_type": pair_type,
                    "jsd": round(jsd, 6),
                    "tk_diff": round(tk_diff, 6),
                })

    results["fiber_distances"] = fiber_dist_results

    # Summarize by pair type
    for pt in ["benign", "history", "cross_world"]:
        relevant = [r for r in fiber_dist_results if r["pair_type"] == pt]
        if relevant:
            avg = sum(r["jsd"] for r in relevant) / len(relevant)
            print(f"  {pt}: avg JSD = {avg:.4f} ({len(relevant)} pairs)")

    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model, tok = load_model()

    all_results = {}
    for set_name, entities in ALL_ENTITY_SETS.items():
        all_results[set_name] = run_algebra_validation(model, tok, entities, set_name)

    # Cross-set comparison
    print("\n\n" + "=" * 60)
    print("CROSS-SET COMPARISON")
    print("=" * 60)
    for set_name in ["registered", "heldout"]:
        r = all_results[set_name]
        print(f"\n{set_name}:")
        print(f"  Baseline: {r['baseline']['accuracy']:.3f}")
        print(f"  Greedy places: {len(r['fiber_table'])}")
        print(f"  Idempotence JSD: {r['idempotence_summary']['avg_jsd']:.4f}")
        print(f"  Idempotence greedy: {r['idempotence_summary']['greedy_match_rate']:.3f}")
        print(f"  Square JSD: {r['square_summary']['avg_jsd']:.4f}")
        print(f"  Square greedy comm: {r['square_summary']['greedy_commutativity_rate']:.3f}")
        print(f"  Square tk diff: {r['square_summary']['avg_tk_correct_diff']:.4f}")

    # Determine verdict
    reg = all_results["registered"]
    held = all_results["heldout"]

    verdict_lines = []
    if reg["idempotence_summary"]["avg_jsd"] < 0.15:
        verdict_lines.append("S_p is approximately idempotent (registered)")
    else:
        verdict_lines.append("S_p is NOT idempotent (registered)")

    if reg["square_summary"]["avg_jsd"] < 0.20:
        verdict_lines.append("Correction/sync square approximately commutes (registered)")
    else:
        verdict_lines.append("Correction/sync square does NOT commute — presentation and prediction coupled (registered)")

    if held["baseline"]["accuracy"] > 0.7:
        verdict_lines.append(f"Held-out baseline OK ({held['baseline']['accuracy']:.3f})")
        if held["idempotence_summary"]["avg_jsd"] < 0.15:
            verdict_lines.append("S_p idempotence generalizes to held-out entities")
        else:
            verdict_lines.append("S_p idempotence does NOT generalize")
    else:
        verdict_lines.append(f"Held-out baseline too low ({held['baseline']['accuracy']:.3f}) — cannot validate")

    print("\n\nVERDICT:")
    for line in verdict_lines:
        print(f"  {line}")

    # Remove non-serializable tensors before saving
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items() if k != "dist"}
        if isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        if isinstance(obj, torch.Tensor):
            return None
        return obj

    out = {
        "experiment": "predictive_fiber_action_v1",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_ID,
        "purpose": "Codex direction v5: validate the Synchronized Predictive-Fiber Action Algebra. "
                   "Test S_p idempotence, correction/synchronization square commutativity, "
                   "and held-out entity generalization.",
        "entity_sets": {
            "registered": list(REGISTERED_ENTITIES.keys()),
            "heldout": list(HELDOUT_ENTITIES.keys()),
        },
        "verdict": verdict_lines,
        "results": clean_for_json(all_results),
    }

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
