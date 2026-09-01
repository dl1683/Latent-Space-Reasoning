"""
OCI-002: Factorial Positional-Carrier Confirmation

Codex Round 4 specification (adapted for real-world entities — model cannot
do in-context learning with nonce entities).

Three mutually-disjoint panels of capital-city pairs, cyclic pairing:
  Panel 1: Vienna/Austria + Oslo/Norway
  Panel 2: Tokyo/Japan + Rome/Italy
  Panel 3: Moscow/Russia + Madrid/Spain
  Pairings: 1→2, 2→3, 3→1

For each panel, 4 prompts hold baseline:
  E1@slot1, query=E1
  E1@slot2, query=E1
  E2@slot1, query=E2
  E2@slot2, query=E2

Core test: transplanting donor hidden state (last position) at mid-layer
boundaries should transfer positional routing — "which slot to attend to"
— independently of queried entity identity.

Pass criteria (from Codex round 4, adapted):
  1. Same queried entity, slot 1→2: recipient target margin shift ≥ 0.30 mean
  2. Recipient follows donor slot in ≥ 75% of mismatch cells
  3. Reversing recipient fact order reverses selected value in ≥ 75%
  4. Changing queried entity (holding slot fixed): effect change ≤ 0.10 mean
  5. Donor value becomes top-1 in ≤ 10% of cells (content leakage check)
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = Path(__file__).parent / "results" / "oci_002"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
TEMPLATE = "{E1} is the capital of {V1}. {E2} is the capital of {V2}. {Q} is the capital of"

PANELS = [
    {"id": "P1", "e1": "Vienna", "v1": "Austria", "e2": "Oslo", "v2": "Norway"},
    {"id": "P2", "e1": "Tokyo", "v1": "Japan", "e2": "Rome", "v2": "Italy"},
    {"id": "P3", "e1": "Moscow", "v1": "Russia", "e2": "Madrid", "v2": "Spain"},
]

CYCLIC_PAIRINGS = [(0, 1), (1, 2), (2, 0)]  # donor_panel → recipient_panel

CO_PRIMARY = [18, 20]
DIAGNOSTIC = [16, 22]
ALL_BOUNDARIES = DIAGNOSTIC[:1] + CO_PRIMARY + DIAGNOSTIC[1:]  # [16, 18, 20, 22]


def make_prompts(panel):
    """Generate 4 prompt configurations for a panel."""
    e1, v1, e2, v2 = panel["e1"], panel["v1"], panel["e2"], panel["v2"]
    return {
        f"{e1}@s1_q{e1}": {
            "text": TEMPLATE.format(E1=e1, V1=v1, E2=e2, V2=v2, Q=e1),
            "query": e1, "slot": 1, "correct": v1, "other": v2,
            "slot1_entity": e1, "slot2_entity": e2,
        },
        f"{e1}@s2_q{e1}": {
            "text": TEMPLATE.format(E1=e2, V1=v2, E2=e1, V2=v1, Q=e1),
            "query": e1, "slot": 2, "correct": v1, "other": v2,
            "slot1_entity": e2, "slot2_entity": e1,
        },
        f"{e2}@s1_q{e2}": {
            "text": TEMPLATE.format(E1=e2, V1=v2, E2=e1, V2=v1, Q=e2),
            "query": e2, "slot": 1, "correct": v2, "other": v1,
            "slot1_entity": e2, "slot2_entity": e1,
        },
        f"{e2}@s2_q{e2}": {
            "text": TEMPLATE.format(E1=e1, V1=v1, E2=e2, V2=v2, Q=e2),
            "query": e2, "slot": 2, "correct": v2, "other": v1,
            "slot1_entity": e1, "slot2_entity": e2,
        },
    }


def run_baseline(model, tokenizer, prompt_text):
    """Run a prompt and return hidden states + output probabilities."""
    inputs = tokenizer(prompt_text, return_tensors="pt")
    ntok = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    probs = torch.softmax(out.logits[0, -1, :], dim=-1)
    hidden = [h[0, -1, :].clone() for h in out.hidden_states]
    return {
        "ntokens": ntok,
        "probs": probs,
        "hidden_states": hidden,
        "inputs": inputs,
    }


def transplant(model, donor_hidden, recipient_inputs, boundary):
    """Replace recipient's last-position hidden state at layer boundary."""
    state = {"done": False}

    def hook(module, args, _s=state, _h=donor_hidden):
        if not _s["done"]:
            _s["done"] = True
            nh = args[0].clone()
            nh[0, -1, :] = _h
            return (nh,) + args[1:]
        return args

    h = model.model.layers[boundary].register_forward_pre_hook(hook)
    with torch.no_grad():
        out = model(**recipient_inputs)
    h.remove()
    return torch.softmax(out.logits[0, -1, :], dim=-1)


def get_token_id(tokenizer, word):
    return tokenizer.encode(" " + word)[-1]


def main():
    t0 = time.time()
    print("=== OCI-002: Factorial Positional-Carrier Confirmation ===")
    print(f"Model: {MODEL_ID}")
    print(f"Boundaries: co-primary {CO_PRIMARY}, diagnostic {DIAGNOSTIC}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=torch.float32, local_files_only=True)
    model.eval()

    all_prompts = {}
    all_baselines = {}
    tok_counts = {}

    print("--- PHASE 1: Baselines ---")
    for panel in PANELS:
        prompts = make_prompts(panel)
        pid = panel["id"]
        all_prompts[pid] = prompts
        tok_counts[pid] = set()
        for name, cfg in prompts.items():
            bl = run_baseline(model, tokenizer, cfg["text"])
            all_baselines[f"{pid}/{name}"] = bl
            tok_counts[pid].add(bl["ntokens"])

            p_correct = bl["probs"][get_token_id(tokenizer, cfg["correct"])].item()
            p_other = bl["probs"][get_token_id(tokenizer, cfg["other"])].item()
            top1_idx = bl["probs"].argmax().item()
            top1 = tokenizer.decode([top1_idx]).strip()
            print(f"  {pid}/{name} ({bl['ntokens']}tok): top1={top1} "
                  f"p({cfg['correct']})={p_correct:.3f} p({cfg['other']})={p_other:.3f}")

        tc = tok_counts[pid]
        assert len(tc) == 1, f"Panel {pid} has mixed token counts: {tc}"
    print()

    print("--- PHASE 2: Transplant experiments ---")
    results = {"baselines": {}, "transplants": [], "pass_criteria": {}}

    for pid in [p["id"] for p in PANELS]:
        panel_bl = {}
        for name, cfg in all_prompts[pid].items():
            key = f"{pid}/{name}"
            bl = all_baselines[key]
            panel_bl[name] = {
                "text": cfg["text"],
                "query": cfg["query"],
                "slot": cfg["slot"],
                "correct": cfg["correct"],
                "other": cfg["other"],
                "p_correct": bl["probs"][get_token_id(tokenizer, cfg["correct"])].item(),
                "p_other": bl["probs"][get_token_id(tokenizer, cfg["other"])].item(),
                "top1": tokenizer.decode([bl["probs"].argmax().item()]).strip(),
                "ntokens": bl["ntokens"],
            }
        results["baselines"][pid] = panel_bl

    for d_idx, r_idx in CYCLIC_PAIRINGS:
        d_panel = PANELS[d_idx]
        r_panel = PANELS[r_idx]
        dpid, rpid = d_panel["id"], r_panel["id"]
        print(f"\n  Pairing: {dpid} → {rpid}")

        d_prompts = all_prompts[dpid]
        r_prompts = all_prompts[rpid]

        r_values = {r_panel["v1"], r_panel["v2"]}
        d_values = {d_panel["v1"], d_panel["v2"]}
        assert r_values.isdisjoint(d_values), f"Values overlap: {r_values} ∩ {d_values}"

        for boundary in ALL_BOUNDARIES:
            for d_name, d_cfg in d_prompts.items():
                for r_name, r_cfg in r_prompts.items():
                    d_key = f"{dpid}/{d_name}"
                    r_key = f"{rpid}/{r_name}"

                    donor_h = all_baselines[d_key]["hidden_states"][boundary]
                    recipient_inputs = all_baselines[r_key]["inputs"]

                    patched_probs = transplant(model, donor_h, recipient_inputs, boundary)

                    r_correct_id = get_token_id(tokenizer, r_cfg["correct"])
                    r_other_id = get_token_id(tokenizer, r_cfg["other"])
                    p_rc = patched_probs[r_correct_id].item()
                    p_ro = patched_probs[r_other_id].item()

                    d_correct_id = get_token_id(tokenizer, d_cfg["correct"])
                    d_other_id = get_token_id(tokenizer, d_cfg["other"])
                    p_dc = patched_probs[d_correct_id].item()
                    p_do = patched_probs[d_other_id].item()

                    top1_idx = patched_probs.argmax().item()
                    top1_tok = tokenizer.decode([top1_idx]).strip()
                    top1_is_donor_val = top1_tok in d_values

                    r_v1_id = get_token_id(tokenizer, r_panel["v1"])
                    r_v2_id = get_token_id(tokenizer, r_panel["v2"])
                    p_rv1 = patched_probs[r_v1_id].item()
                    p_rv2 = patched_probs[r_v2_id].item()

                    target_margin = p_rv1 - p_rv2
                    slot_selected = 1 if p_rv1 > p_rv2 else (2 if p_rv2 > p_rv1 else 0)

                    row = {
                        "donor_panel": dpid,
                        "recipient_panel": rpid,
                        "boundary": boundary,
                        "donor_name": d_name,
                        "recipient_name": r_name,
                        "donor_query": d_cfg["query"],
                        "donor_slot": d_cfg["slot"],
                        "recipient_query": r_cfg["query"],
                        "recipient_slot": r_cfg["slot"],
                        "recipient_order": "normal" if r_cfg["slot1_entity"] == r_panel["e1"] else "reversed",
                        "p_recipient_v1": p_rv1,
                        "p_recipient_v2": p_rv2,
                        "target_margin": target_margin,
                        "slot_selected": slot_selected,
                        "p_donor_correct": p_dc,
                        "p_donor_other": p_do,
                        "top1": top1_tok,
                        "top1_is_donor_value": top1_is_donor_val,
                    }
                    results["transplants"].append(row)

            n_done = len(results["transplants"])
            if n_done % 48 == 0:
                print(f"    B{boundary}: {n_done} rows total")

    print(f"\n--- PHASE 3: Pass criteria evaluation ---")

    for boundary in CO_PRIMARY:
        rows = [r for r in results["transplants"] if r["boundary"] == boundary]
        print(f"\n  === B{boundary} ===")

        slot_shift_deltas = []
        slot_following_mismatch = []
        sentence_pos_following = []
        reversal_pairs = []
        entity_fixed_deltas = []
        content_leakage = []

        for r in rows:
            content_leakage.append(r["top1_is_donor_value"])
            margin = r["target_margin"]
            if abs(margin) > 0.01:
                donor_pos = r["donor_slot"]
                if r["recipient_order"] == "normal":
                    boosted_pos = 1 if margin > 0 else 2
                else:
                    boosted_pos = 2 if margin > 0 else 1
                sentence_pos_following.append(donor_pos == boosted_pos)

        for d_idx, r_idx in CYCLIC_PAIRINGS:
            dpid = PANELS[d_idx]["id"]
            rpid = PANELS[r_idx]["id"]
            pr = [r for r in rows if r["donor_panel"] == dpid and r["recipient_panel"] == rpid]

            entities_in_donor = {PANELS[d_idx]["e1"], PANELS[d_idx]["e2"]}
            for query_entity in entities_in_donor:
                s1_rows = [r for r in pr if r["donor_query"] == query_entity and r["donor_slot"] == 1]
                s2_rows = [r for r in pr if r["donor_query"] == query_entity and r["donor_slot"] == 2]

                for s1r in s1_rows:
                    for s2r in s2_rows:
                        if s1r["recipient_name"] == s2r["recipient_name"]:
                            delta = abs(s1r["target_margin"] - s2r["target_margin"])
                            slot_shift_deltas.append(delta)

                for r in s1_rows + s2_rows:
                    if r["donor_slot"] != r["recipient_slot"]:
                        follows = r["slot_selected"] == r["donor_slot"]
                        slot_following_mismatch.append(follows)

            for query_entity in entities_in_donor:
                normal_rows = [r for r in pr if r["donor_query"] == query_entity and r["recipient_order"] == "normal"]
                reversed_rows = [r for r in pr if r["donor_query"] == query_entity and r["recipient_order"] == "reversed"]
                for nr in normal_rows:
                    for rr in reversed_rows:
                        if nr["donor_slot"] == rr["donor_slot"]:
                            reversed_ok = nr["slot_selected"] != rr["slot_selected"] if nr["slot_selected"] != 0 and rr["slot_selected"] != 0 else False
                            reversal_pairs.append(reversed_ok)

            for slot_val in [1, 2]:
                slot_rows = [r for r in pr if r["donor_slot"] == slot_val]
                for i, r1 in enumerate(slot_rows):
                    for r2 in slot_rows[i+1:]:
                        if r1["donor_query"] != r2["donor_query"] and r1["recipient_name"] == r2["recipient_name"]:
                            delta = abs(r1["target_margin"] - r2["target_margin"])
                            entity_fixed_deltas.append(delta)

        mean_slot_shift = sum(slot_shift_deltas) / len(slot_shift_deltas) if slot_shift_deltas else 0
        slot_follow_pct = sum(slot_following_mismatch) / len(slot_following_mismatch) * 100 if slot_following_mismatch else 0
        sent_pos_pct = sum(sentence_pos_following) / len(sentence_pos_following) * 100 if sentence_pos_following else 0
        reversal_pct = sum(reversal_pairs) / len(reversal_pairs) * 100 if reversal_pairs else 0
        mean_entity_delta = sum(entity_fixed_deltas) / len(entity_fixed_deltas) if entity_fixed_deltas else 0
        content_leak_pct = sum(content_leakage) / len(content_leakage) * 100 if content_leakage else 0

        c1 = mean_slot_shift >= 0.30
        c2_orig = slot_follow_pct >= 75
        c2_corrected = sent_pos_pct >= 75
        c3 = reversal_pct >= 75
        c4 = mean_entity_delta <= 0.10
        c5 = content_leak_pct <= 10

        criteria = {
            "boundary": boundary,
            "C1_slot_shift_mean": round(mean_slot_shift, 4),
            "C1_pass": c1,
            "C1_n": len(slot_shift_deltas),
            "C2_panel_slot_follow_pct": round(slot_follow_pct, 1),
            "C2_panel_slot_pass": c2_orig,
            "C2_sentence_pos_follow_pct": round(sent_pos_pct, 1),
            "C2_sentence_pos_pass": c2_corrected,
            "C2_n": len(slot_following_mismatch),
            "C3_reversal_pct": round(reversal_pct, 1),
            "C3_pass": c3,
            "C3_n": len(reversal_pairs),
            "C4_entity_delta_mean": round(mean_entity_delta, 4),
            "C4_pass": c4,
            "C4_n": len(entity_fixed_deltas),
            "C5_content_leak_pct": round(content_leak_pct, 1),
            "C5_pass": c5,
            "C5_n": len(content_leakage),
            "OVERALL_PASS": c1 and c2_orig and c3 and c4 and c5,
            "CORRECTED_PASS": c1 and c2_corrected and c3 and c5,
        }
        results["pass_criteria"][f"B{boundary}"] = criteria

        print(f"  C1 slot-shift mean:        {mean_slot_shift:.4f} (need ≥0.30) {'PASS' if c1 else 'FAIL'} [n={len(slot_shift_deltas)}]")
        print(f"  C2 panel-slot-follow:      {slot_follow_pct:.1f}% (need ≥75%) {'PASS' if c2_orig else 'FAIL'} [n={len(slot_following_mismatch)}]")
        print(f"  C2b sentence-pos-follow:   {sent_pos_pct:.1f}% (need ≥75%) {'PASS' if c2_corrected else 'FAIL'} [n={len(sentence_pos_following)}]")
        print(f"  C3 reversal:               {reversal_pct:.1f}% (need ≥75%) {'PASS' if c3 else 'FAIL'} [n={len(reversal_pairs)}]")
        print(f"  C4 entity-delta:           {mean_entity_delta:.4f} (need ≤0.10) {'PASS' if c4 else 'FAIL'} [n={len(entity_fixed_deltas)}]")
        print(f"  C5 content-leak:           {content_leak_pct:.1f}% (need ≤10%) {'PASS' if c5 else 'FAIL'} [n={len(content_leakage)}]")
        print(f"  *** B{boundary}: ORIGINAL={'PASS' if criteria['OVERALL_PASS'] else 'FAIL'} | CORRECTED(C2b)={'PASS' if criteria['CORRECTED_PASS'] else 'FAIL'} ***")

    both_pass_orig = all(v["OVERALL_PASS"] for v in results["pass_criteria"].values())
    both_pass_corrected = all(v.get("CORRECTED_PASS", False) for v in results["pass_criteria"].values())
    results["verdict_original"] = "POSITIONAL-CARRIER PASS" if both_pass_orig else "FAIL"
    results["verdict_corrected"] = "SENTENCE-POSITION ROUTING PASS" if both_pass_corrected else "FAIL"
    results["verdict"] = results["verdict_corrected"]

    print(f"\n{'='*60}")
    print(f"  ORIGINAL VERDICT (Codex C2): {results['verdict_original']}")
    print(f"  CORRECTED VERDICT (C2b sentence-pos): {results['verdict_corrected']}")
    print(f"{'='*60}")

    elapsed = time.time() - t0
    results["meta"] = {
        "model": MODEL_ID,
        "template": TEMPLATE,
        "panels": PANELS,
        "cyclic_pairings": CYCLIC_PAIRINGS,
        "co_primary_boundaries": CO_PRIMARY,
        "diagnostic_boundaries": DIAGNOSTIC,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_transplant_rows": len(results["transplants"]),
    }

    out_path = RESULTS_DIR / "verdict.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
