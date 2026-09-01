"""
Signature-indexed restatement S^G_g (v1).

Tests whether a restatement constructed from the model's own observable
greedy signature — not the experimenter-known world — retains the algebraic
properties established for S^W_w in v2:
  1. Approximate idempotence: (S^G_g)^2 ~= S^G_g
  2. Descent to greedy quotient G: all fiber members map to same target
  3. Non-naturality with correction: typed square S^G_{g'} . C vs C . S^G_g

The prediction: S^G should fix the one descent failure in v2 (fiber spanning
worlds w101/w111 with different hidden worlds but same greedy signature),
because S^G uses only the shared observable, not the divergent hidden world.

For the typed square, S^G_{g'} after correction requires running the
corrected history through the model to get its NEW greedy signature g',
then building the restatement from g'. This is the key construction.
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
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "signature_restatement_v1")

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
    """S^W_w — world-conditioned (kept for comparison only)."""
    names = list(entities.keys())
    parts = [f"{n}: {world[n]}" for n in names]
    return " To be clear: " + ". ".join(parts) + "."


def make_restatement_from_signature(sig, entity_names):
    """S^G_g — signature-indexed restatement from observable greedy answers.

    sig: tuple of (entity_name, greedy_answer) pairs, sorted by entity name.
    entity_names: list of entity names (for ordering).
    """
    sig_dict = dict(sig)
    parts = [f"{n}: {sig_dict[n]}" for n in entity_names]
    return " To be clear: " + ". ".join(parts) + "."


def make_shuffled_restatement(sig, entity_names):
    """Control: same format, same tokens, but values rotated among entities."""
    sig_dict = dict(sig)
    values = [sig_dict[n] for n in entity_names]
    rotated = values[1:] + values[:1]
    parts = [f"{n}: {rotated[i]}" for i, n in enumerate(entity_names)]
    return " To be clear: " + ". ".join(parts) + "."


def make_corrected_world(world, target_entity, new_value):
    corrected = dict(world)
    corrected[target_entity] = new_value
    return corrected


def get_greedy_signature(model, tok, base_prompt, entity_names):
    sig = {}
    margins = {}
    for ent in entity_names:
        prompt = base_prompt + f"\n{ent}:"
        dist = get_dist(model, tok, prompt)
        topk = torch.topk(dist, 2)
        greedy = tok.decode([topk.indices[0].item()]).strip()
        second = tok.decode([topk.indices[1].item()]).strip()
        margin = float(topk.values[0] - topk.values[1])
        sig[ent] = greedy
        margins[ent] = {"top1": greedy, "top2": second, "margin": round(margin, 6)}
    return tuple(sorted(sig.items())), margins


def run_sg_validation(model, tok, entities, entity_set_name):
    worlds = make_worlds(entities)
    entity_names = list(entities.keys())
    results = {
        "entity_set": entity_set_name,
        "entities": {k: list(v) for k, v in entities.items()},
        "n_worlds": len(worlds),
        "greedy_signatures": {},
        "sw_vs_sg_comparison": [],
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
            sig, sig_margins = get_greedy_signature(model, tok, base, entity_names)
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
                "sig_margins": sig_margins,
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
        print(f"  {sig_str}: {len(members)} members ({', '.join(members[:4])})")

    results["greedy_signatures"] = {k: v for k, v in sig_groups.items()}

    # Phase 2: S^W vs S^G comparison — how often do they produce different text?
    print("\n--- Phase 2: S^W vs S^G comparison ---")
    sw_sg_comparison = []
    for key, state in all_states.items():
        base = state["base_prompt"]
        world = state["world"]
        sig = state["signature"]

        sw = make_restatement_from_world(world, entities)
        sg = make_restatement_from_signature(sig, entity_names)
        match = sw == sg

        for ent in entity_names:
            sw_prompt = base + sw + f"\n{ent}:"
            sg_prompt = base + sg + f"\n{ent}:"
            sw_dist = get_dist(model, tok, sw_prompt)
            sg_dist = get_dist(model, tok, sg_prompt)
            jsd = js_dist(sw_dist, sg_dist)
            sw_greedy = tok.decode([torch.argmax(sw_dist).item()]).strip()
            sg_greedy = tok.decode([torch.argmax(sg_dist).item()]).strip()

            sw_sg_comparison.append({
                "source": key,
                "entity": ent,
                "sw_text_equals_sg": match,
                "sw_greedy": sw_greedy,
                "sg_greedy": sg_greedy,
                "greedy_match": sw_greedy == sg_greedy,
                "jsd_sw_vs_sg": round(jsd, 6),
            })

    results["sw_vs_sg_comparison"] = sw_sg_comparison
    text_matches = sum(1 for r in sw_sg_comparison if r["sw_text_equals_sg"]) // len(entity_names)
    greedy_matches = sum(1 for r in sw_sg_comparison if r["greedy_match"])
    jsds_comp = [r["jsd_sw_vs_sg"] for r in sw_sg_comparison]
    n_total = len(sw_sg_comparison)
    print(f"  Text identical (S^W == S^G): {text_matches}/{len(all_states)}")
    print(f"  Greedy identical: {greedy_matches}/{n_total}")
    if jsds_comp:
        divergent = [r for r in sw_sg_comparison if not r["sw_text_equals_sg"]]
        if divergent:
            div_jsds = [r["jsd_sw_vs_sg"] for r in divergent]
            print(f"  When S^W != S^G: JSD mean={statistics.mean(div_jsds):.4f}, max={max(div_jsds):.4f}")

    # Phase 3: Idempotence of S^G
    print("\n--- Phase 3: Idempotence of S^G ---")
    idemp_results = []
    for key, state in all_states.items():
        base = state["base_prompt"]
        sig = state["signature"]
        sg = make_restatement_from_signature(sig, entity_names)

        for ent in entity_names:
            pe = state["per_entity"][ent]
            sg1_prompt = base + sg + f"\n{ent}:"
            sg2_prompt = base + sg + sg + f"\n{ent}:"
            sg1_dist = get_dist(model, tok, sg1_prompt)
            sg2_dist = get_dist(model, tok, sg2_prompt)

            jsd = js_dist(sg1_dist, sg2_dist)
            sg1_greedy = tok.decode([torch.argmax(sg1_dist).item()]).strip()
            sg2_greedy = tok.decode([torch.argmax(sg2_dist).item()]).strip()
            sg1_tk = task_kernel(sg1_dist, tok, pe["correct"], pe["wrong"])
            sg2_tk = task_kernel(sg2_dist, tok, pe["correct"], pe["wrong"])

            idemp_results.append({
                "source": key,
                "entity": ent,
                "jsd_SG_vs_SG2": round(jsd, 6),
                "greedy_SG": sg1_greedy,
                "greedy_SG2": sg2_greedy,
                "greedy_match": sg1_greedy == sg2_greedy,
                "tk_SG": sg1_tk,
                "tk_SG2": sg2_tk,
            })

    results["idempotence_test"] = idemp_results
    jsds_i = [r["jsd_SG_vs_SG2"] for r in idemp_results]
    matches_i = sum(1 for r in idemp_results if r["greedy_match"])
    print(f"  n={len(idemp_results)}, JSD range [{min(jsds_i):.4f}, {max(jsds_i):.4f}], mean={statistics.mean(jsds_i):.4f}")
    print(f"  Greedy idempotence: {matches_i}/{len(idemp_results)} = {matches_i/len(idemp_results):.1%}")

    # Phase 4: Place preservation under S^G
    print("\n--- Phase 4: Place preservation under S^G ---")
    preservation_results = []
    for key, state in all_states.items():
        base = state["base_prompt"]
        sig = state["signature"]
        sg = make_restatement_from_signature(sig, entity_names)
        after_prompt = base + sg
        sig_after, _ = get_greedy_signature(model, tok, after_prompt, entity_names)
        preserved = sig == sig_after
        preservation_results.append({
            "source": key,
            "sig_before": "|".join(f"{k}={v}" for k, v in sig),
            "sig_after": "|".join(f"{k}={v}" for k, v in sig_after),
            "preserved": preserved,
        })
        if not preserved:
            print(f"  CHANGED: {key}: {sig} -> {sig_after}")

    results["place_preservation"] = preservation_results
    pres_rate = sum(1 for r in preservation_results if r["preserved"]) / len(preservation_results)
    print(f"  S^G place preservation: {sum(1 for r in preservation_results if r['preserved'])}/{len(preservation_results)} = {pres_rate:.1%}")

    # Phase 4b: Shuffled-renderer control (Codex confound: textual echo)
    print("\n--- Phase 4b: Shuffled-renderer control ---")
    shuffled_results = []
    for key, state in all_states.items():
        base = state["base_prompt"]
        sig = state["signature"]
        shuffled = make_shuffled_restatement(sig, entity_names)
        after_prompt = base + shuffled
        sig_after, _ = get_greedy_signature(model, tok, after_prompt, entity_names)
        preserved = sig == sig_after
        shuffled_results.append({
            "source": key,
            "sig_before": "|".join(f"{k}={v}" for k, v in sig),
            "sig_after": "|".join(f"{k}={v}" for k, v in sig_after),
            "preserved": preserved,
            "shuffled_text": shuffled[:60],
        })
        if not preserved:
            print(f"  DISRUPTED: {key}: {dict(sig)} -> {dict(sig_after)}")

    results["shuffled_control"] = shuffled_results
    shuf_pres = sum(1 for r in shuffled_results if r["preserved"]) / len(shuffled_results)
    print(f"  Shuffled place preservation: {sum(1 for r in shuffled_results if r['preserved'])}/{len(shuffled_results)} = {shuf_pres:.1%}")
    print(f"  (S^G was {pres_rate:.1%} — shuffled should be lower if content matters)")

    # Phase 5a: Correction descent — does C_{e<-v} produce same g' for all fiber members?
    # (Codex: prerequisite for a genuine quotient-level typed square)
    print("\n--- Phase 5a: Correction descent ---")
    correction_descent = []
    for sig_str, members in sig_groups.items():
        if len(members) < 2:
            continue
        for ent in entity_names:
            # Use a fixed correction value: flip to the "wrong" value of the first member
            first_pe = all_states[members[0]]["per_entity"][ent]
            correction_val = first_pe["wrong"]
            correction = f" Actually, {ent}: {correction_val}."

            post_correction_sigs = {}
            for mk in members:
                mstate = all_states[mk]
                corrected_base = mstate["base_prompt"] + correction
                sig_after, _ = get_greedy_signature(model, tok, corrected_base, entity_names)
                sig_str_after = "|".join(f"{k}={v}" for k, v in sig_after)
                post_correction_sigs[mk] = sig_str_after

            unique_targets = set(post_correction_sigs.values())
            descends = len(unique_targets) == 1
            correction_descent.append({
                "fiber": sig_str,
                "entity": ent,
                "correction_val": correction_val,
                "n_members": len(members),
                "post_correction_sigs": post_correction_sigs,
                "descends": descends,
            })
            if not descends:
                print(f"  FAIL: {sig_str}/{ent}->'{correction_val}': {post_correction_sigs}")

    results["correction_descent"] = correction_descent
    if correction_descent:
        cd_pass = sum(1 for r in correction_descent if r["descends"])
        print(f"  Correction descent: {cd_pass}/{len(correction_descent)} = {cd_pass/len(correction_descent):.1%}")
    else:
        print(f"  No multi-member fibers to test")

    # Phase 6: TYPED SQUARE with S^G
    # Path CS: base + correction + S^G_{g'} + query
    #   where g' = greedy signature of (base + correction)
    # Path SC: base + S^G_g + correction + query
    print("\n--- Phase 6: Typed correction/S^G square ---")
    square_results = []

    for key, state in all_states.items():
        base = state["base_prompt"]
        sig_g = state["signature"]

        for ent in entity_names:
            pe = state["per_entity"][ent]
            correct_val = pe["correct"]
            wrong_val = pe["wrong"]

            correction = f" Actually, {ent}: {wrong_val}."

            # Get the greedy signature AFTER correction (Codex: g'_x = γ(C_{e←v}x))
            corrected_base = base + correction
            sig_g_prime, _ = get_greedy_signature(model, tok, corrected_base, entity_names)

            sg_g = make_restatement_from_signature(sig_g, entity_names)
            sg_g_prime = make_restatement_from_signature(sig_g_prime, entity_names)

            # Path 1: C then S^G_{g'} — correct, then restate from post-correction signature
            cs_prompt = corrected_base + sg_g_prime + f"\n{ent}:"
            cs_dist = get_dist(model, tok, cs_prompt)

            # Path 2: S^G_g then C — restate from pre-correction signature, then correct
            sc_prompt = base + sg_g + correction + f"\n{ent}:"
            sc_dist = get_dist(model, tok, sc_prompt)

            jsd = js_dist(cs_dist, sc_dist)
            cs_greedy = tok.decode([torch.argmax(cs_dist).item()]).strip()
            sc_greedy = tok.decode([torch.argmax(sc_dist).item()]).strip()
            cs_tk = task_kernel(cs_dist, tok, wrong_val, correct_val)
            sc_tk = task_kernel(sc_dist, tok, wrong_val, correct_val)

            # Also compare with S^W square for the same pair
            corrected_world = make_corrected_world(state["world"], ent, wrong_val)
            sw_old = make_restatement_from_world(state["world"], entities)
            sw_new = make_restatement_from_world(corrected_world, entities)
            cs_sw_prompt = base + correction + sw_new + f"\n{ent}:"
            sc_sw_prompt = base + sw_old + correction + f"\n{ent}:"
            cs_sw_dist = get_dist(model, tok, cs_sw_prompt)
            sc_sw_dist = get_dist(model, tok, sc_sw_prompt)
            jsd_sw = js_dist(cs_sw_dist, sc_sw_dist)

            sq = {
                "source": key,
                "entity": ent,
                "corrected_value": wrong_val,
                "original_value": correct_val,
                "sig_g": "|".join(f"{k}={v}" for k, v in sig_g),
                "sig_g_prime": "|".join(f"{k}={v}" for k, v in sig_g_prime),
                "sg_text": sg_g[:60],
                "sg_prime_text": sg_g_prime[:60],
                "jsd_CS_vs_SC_SG": round(jsd, 6),
                "jsd_CS_vs_SC_SW": round(jsd_sw, 6),
                "greedy_CS": cs_greedy,
                "greedy_SC": sc_greedy,
                "greedy_match": cs_greedy == sc_greedy,
                "tk_CS": cs_tk,
                "tk_SC": sc_tk,
                "tk_correct_diff": round(abs(cs_tk["correct"] - sc_tk["correct"]), 6),
            }
            square_results.append(sq)
            delta_label = "SG<SW" if jsd < jsd_sw else ("SG>SW" if jsd > jsd_sw else "SG==SW")
            print(f"  {key}/{ent}: JSD_SG={jsd:.4f}, JSD_SW={jsd_sw:.4f} [{delta_label}], greedy={cs_greedy}/{sc_greedy}")

    results["typed_square_test"] = square_results
    sq_jsds_sg = [r["jsd_CS_vs_SC_SG"] for r in square_results]
    sq_jsds_sw = [r["jsd_CS_vs_SC_SW"] for r in square_results]
    sq_matches = sum(1 for r in square_results if r["greedy_match"])
    sq_tk_diffs = [r["tk_correct_diff"] for r in square_results]
    print(f"\n  n={len(square_results)}")
    print(f"  S^G JSD: [{min(sq_jsds_sg):.4f}, {max(sq_jsds_sg):.4f}], mean={statistics.mean(sq_jsds_sg):.4f}")
    print(f"  S^W JSD: [{min(sq_jsds_sw):.4f}, {max(sq_jsds_sw):.4f}], mean={statistics.mean(sq_jsds_sw):.4f}")
    print(f"  Greedy commutativity (S^G): {sq_matches}/{len(square_results)} = {sq_matches/len(square_results):.1%}")
    print(f"  Task kernel diff: mean={statistics.mean(sq_tk_diffs):.4f}")
    sg_less = sum(1 for s, w in zip(sq_jsds_sg, sq_jsds_sw) if s < w)
    print(f"  S^G < S^W (less non-naturality): {sg_less}/{len(square_results)}")

    # Phase 5: Descent test with S^G (Codex: test FULL signature, not entity-by-entity)
    print("\n--- Phase 5: Descent test (S^G vs S^W) ---")
    descent_results = []
    for sig_str, members in sig_groups.items():
        if len(members) < 2:
            continue
        sig_tuple = all_states[members[0]]["signature"]
        sg = make_restatement_from_signature(sig_tuple, entity_names)

        for ent in entity_names:
            targets_empty = set()
            targets_sg = set()
            targets_sw = set()
            for mk in members:
                mstate = all_states[mk]
                base = mstate["base_prompt"]
                world = mstate["world"]
                # Empty
                prompt = base + f"\n{ent}:"
                dist = get_dist(model, tok, prompt)
                g = tok.decode([torch.argmax(dist).item()]).strip()
                targets_empty.add(g)
                # S^G (same for all members — that's the point)
                prompt = base + sg + f"\n{ent}:"
                dist = get_dist(model, tok, prompt)
                g = tok.decode([torch.argmax(dist).item()]).strip()
                targets_sg.add(g)
                # S^W (for comparison — uses hidden world, differs across members)
                sw = make_restatement_from_world(world, entities)
                prompt = base + sw + f"\n{ent}:"
                dist = get_dist(model, tok, prompt)
                g = tok.decode([torch.argmax(dist).item()]).strip()
                targets_sw.add(g)

            descent_results.append({
                "signature": sig_str,
                "entity": ent,
                "n_members": len(members),
                "member_worlds": [all_states[mk]["world_key"] for mk in members],
                "empty_targets": list(targets_empty),
                "empty_descends": len(targets_empty) == 1,
                "sg_targets": list(targets_sg),
                "sg_descends": len(targets_sg) == 1,
                "sw_targets": list(targets_sw),
                "sw_descends": len(targets_sw) == 1,
            })

    results["descent_test"] = descent_results
    empty_desc = sum(1 for r in descent_results if r["empty_descends"])
    sg_desc = sum(1 for r in descent_results if r["sg_descends"])
    sw_desc = sum(1 for r in descent_results if r["sw_descends"])
    total_desc = len(descent_results)
    print(f"  Fiber groups tested: {total_desc}")
    print(f"  Empty descent:       {empty_desc}/{total_desc} = {empty_desc/max(total_desc,1):.1%}")
    print(f"  S^G descent:         {sg_desc}/{total_desc} = {sg_desc/max(total_desc,1):.1%}")
    print(f"  S^W descent:         {sw_desc}/{total_desc} = {sw_desc/max(total_desc,1):.1%}")
    failures_sg = [r for r in descent_results if not r["sg_descends"]]
    failures_sw = [r for r in descent_results if not r["sw_descends"]]
    if failures_sg:
        print(f"  S^G failures:")
        for f in failures_sg:
            print(f"    {f['signature']}/{f['entity']}: worlds={f['member_worlds']}, targets={f['sg_targets']}")
    if failures_sw:
        print(f"  S^W failures:")
        for f in failures_sw:
            print(f"    {f['signature']}/{f['entity']}: worlds={f['member_worlds']}, targets={f['sw_targets']}")

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
        res = run_sg_validation(model, tok, entities, set_name)
        all_results["sets"][set_name] = res

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for sn in ["registered", "heldout"]:
        r = all_results["sets"][sn]
        print(f"\n  SET {sn}")
        print(f"    Baseline: {r['baseline_accuracy']:.1%}")

        comp = r["sw_vs_sg_comparison"]
        text_same = sum(1 for c in comp if c["sw_text_equals_sg"])
        greedy_same = sum(1 for c in comp if c["greedy_match"])
        print(f"    S^W vs S^G text identical: {text_same}/{len(comp)}")
        print(f"    S^W vs S^G greedy identical: {greedy_same}/{len(comp)}")

        idemp = r["idempotence_test"]
        ij = [x["jsd_SG_vs_SG2"] for x in idemp]
        im = sum(1 for x in idemp if x["greedy_match"])
        print(f"    S^G idempotence: n={len(idemp)}, JSD [{min(ij):.4f},{max(ij):.4f}], greedy {im}/{len(idemp)}")

        sq = r["typed_square_test"]
        sj_sg = [x["jsd_CS_vs_SC_SG"] for x in sq]
        sj_sw = [x["jsd_CS_vs_SC_SW"] for x in sq]
        sm = sum(1 for x in sq if x["greedy_match"])
        stk = [x["tk_correct_diff"] for x in sq]
        print(f"    S^G square: JSD [{min(sj_sg):.4f},{max(sj_sg):.4f}], mean={statistics.mean(sj_sg):.4f}")
        print(f"    S^W square: JSD [{min(sj_sw):.4f},{max(sj_sw):.4f}], mean={statistics.mean(sj_sw):.4f}")
        print(f"    Greedy commutativity (S^G): {sm}/{len(sq)} = {sm/len(sq):.1%}")

        desc = r["descent_test"]
        if desc:
            sg_d = sum(1 for d in desc if d["sg_descends"])
            sw_d = sum(1 for d in desc if d["sw_descends"])
            print(f"    Descent S^G: {sg_d}/{len(desc)}, S^W: {sw_d}/{len(desc)}")

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
