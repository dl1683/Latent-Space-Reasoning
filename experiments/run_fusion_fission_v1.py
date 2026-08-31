"""
Fusion-Fission v1: test whether two independently controllable facts
become indivisible during computation and later separate.

DUAL METHODOLOGY (breakpoint hunting):
  1. Opaque whole-record transplants -> categorical behavioral outcomes
  2. R^n probes (linear, cosine) -> continuous geometric measures
  Where they agree: R^n captures native structure.
  Where they disagree: BREAKPOINT revealing native structure R^n can't see.

CPU only. ~700-1100 forwards for the core experiment.
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cpu"
DTYPE = torch.float32


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
    return model, tok


# --- World construction ---
# Two independent facts: color of HESK and color of VORN
# 4 worlds: (red,red), (red,blue), (blue,red), (blue,blue)

ENTITIES = ["HESK", "VORN"]
VALUES = ["red", "blue"]

def make_prompt(hesk_color, vorn_color):
    return (
        f"HESK is {hesk_color}. VORN is {vorn_color}. "
        f"Remember these facts.\n"
    )

def make_query_a():
    return "What color is HESK? Answer with one word:"

def make_query_b():
    return "What color is VORN? Answer with one word:"


WORLDS = [
    {"hesk": "red",  "vorn": "red",  "label": "00"},
    {"hesk": "red",  "vorn": "blue", "label": "01"},
    {"hesk": "blue", "vorn": "red",  "label": "10"},
    {"hesk": "blue", "vorn": "blue", "label": "11"},
]


def get_answer_logits(model, tok, prompt, query):
    """Get logits for 'red' vs 'blue' after prompt+query."""
    text = prompt + query
    ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(ids)
    logits = out.logits[0, -1]
    red_id = tok.encode(" red", add_special_tokens=False)[0]
    blue_id = tok.encode(" blue", add_special_tokens=False)[0]
    return {
        "red": logits[red_id].item(),
        "blue": logits[blue_id].item(),
        "pred": "red" if logits[red_id] > logits[blue_id] else "blue",
    }


def get_hidden_states(model, tok, prompt):
    """Get all layer hidden states for a prompt."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return out.hidden_states, ids


def transplant_at_layer(model, tok, donor_prompt, host_prompt, query, layer_idx):
    """Transplant donor's hidden state at layer_idx into host, then query.

    Uses a forward hook to replace the host's hidden state with the donor's
    at a specific layer, then continues the host's computation.
    """
    # Get donor hidden states
    donor_hs, donor_ids = get_hidden_states(model, tok, donor_prompt)
    donor_state = donor_hs[layer_idx]  # [1, seq_len, d]

    # Get host input
    host_text = host_prompt + query
    host_ids = tok(host_text, return_tensors="pt").input_ids.to(DEVICE)

    # Length of prompt (without query) for transplant scope
    prompt_ids = tok(host_prompt, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = prompt_ids.shape[1]
    donor_prompt_ids = tok(donor_prompt, return_tensors="pt").input_ids.to(DEVICE)
    donor_len = donor_prompt_ids.shape[1]

    # Only transplant if prompt lengths match
    if prompt_len != donor_len:
        return None

    # Hook to replace hidden states at the target layer
    replaced = [False]

    def hook_fn(module, input, output):
        if replaced[0]:
            return output
        replaced[0] = True
        # output is a tuple; first element is the hidden state
        if isinstance(output, tuple):
            hs = output[0]
            new_hs = hs.clone()
            new_hs[0, :prompt_len] = donor_state[0, :donor_len]
            return (new_hs,) + output[1:]
        else:
            new_hs = output.clone()
            new_hs[0, :prompt_len] = donor_state[0, :donor_len]
            return new_hs

    # Register hook on the target layer
    layers = model.model.layers
    handle = layers[layer_idx].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            out = model(host_ids)
        logits = out.logits[0, -1]
        red_id = tok.encode(" red", add_special_tokens=False)[0]
        blue_id = tok.encode(" blue", add_special_tokens=False)[0]
        return {
            "red": logits[red_id].item(),
            "blue": logits[blue_id].item(),
            "pred": "red" if logits[red_id] > logits[blue_id] else "blue",
        }
    finally:
        handle.remove()


def cosine_sim(a, b):
    """Cosine similarity between two tensors."""
    return F.cosine_similarity(a.flatten().unsqueeze(0),
                                b.flatten().unsqueeze(0)).item()


def run_baseline(model, tok):
    """Check clean accuracy on all 4 worlds x 2 queries."""
    print("=== Baseline accuracy check ===")
    results = {}
    correct = 0
    total = 0
    for w in WORLDS:
        prompt = make_prompt(w["hesk"], w["vorn"])
        for fact, query_fn, expected in [
            ("A", make_query_a, w["hesk"]),
            ("B", make_query_b, w["vorn"]),
        ]:
            r = get_answer_logits(model, tok, prompt, query_fn())
            ok = r["pred"] == expected
            correct += ok
            total += 1
            results[f"{w['label']}_{fact}"] = {
                "expected": expected, "pred": r["pred"], "ok": ok,
                "margin": abs(r["red"] - r["blue"]),
            }
            print(f"  World {w['label']}, query {fact}: "
                  f"expected={expected}, pred={r['pred']}, "
                  f"margin={abs(r['red'] - r['blue']):.2f} {'OK' if ok else 'FAIL'}")
    acc = correct / total
    print(f"\n  Accuracy: {correct}/{total} = {acc:.1%}")
    return results, acc


def run_fusion_fission_scan(model, tok):
    """Scan all layers for fusion-fission transitions.

    At each layer, transplant from counterfactual worlds and check:
    - A-only transplant (change HESK, keep VORN): does only query-A change?
    - B-only transplant (keep HESK, change VORN): does only query-B change?
    - Joint transplant (change both): do both change?

    SEPARATE = A-only changes only A, B-only changes only B
    FUSED = A-only also changes B (or vice versa)
    """
    n_layers = model.config.num_hidden_layers
    print(f"\n=== Fusion-fission scan across {n_layers} layers ===")

    # Use world 00 (red,red) as host
    host = WORLDS[0]  # hesk=red, vorn=red
    host_prompt = make_prompt(host["hesk"], host["vorn"])

    # Counterfactual donors
    donor_a = WORLDS[2]  # hesk=blue, vorn=red -> changes A only
    donor_b = WORLDS[1]  # hesk=red, vorn=blue -> changes B only
    donor_ab = WORLDS[3]  # hesk=blue, vorn=blue -> changes both

    layer_results = []

    for layer in range(n_layers):
        result = {"layer": layer}

        # A-only transplant: donor has different HESK, same VORN
        donor_a_prompt = make_prompt(donor_a["hesk"], donor_a["vorn"])
        ta_qa = transplant_at_layer(model, tok, donor_a_prompt, host_prompt,
                                     make_query_a(), layer)
        ta_qb = transplant_at_layer(model, tok, donor_a_prompt, host_prompt,
                                     make_query_b(), layer)

        # B-only transplant: donor has same HESK, different VORN
        donor_b_prompt = make_prompt(donor_b["hesk"], donor_b["vorn"])
        tb_qa = transplant_at_layer(model, tok, donor_b_prompt, host_prompt,
                                     make_query_a(), layer)
        tb_qb = transplant_at_layer(model, tok, donor_b_prompt, host_prompt,
                                     make_query_b(), layer)

        # Joint transplant
        donor_ab_prompt = make_prompt(donor_ab["hesk"], donor_ab["vorn"])
        tab_qa = transplant_at_layer(model, tok, donor_ab_prompt, host_prompt,
                                      make_query_a(), layer)
        tab_qb = transplant_at_layer(model, tok, donor_ab_prompt, host_prompt,
                                      make_query_b(), layer)

        if any(x is None for x in [ta_qa, ta_qb, tb_qa, tb_qb, tab_qa, tab_qb]):
            print(f"  Layer {layer:2d}: SKIPPED (length mismatch)")
            continue

        # Classify: did the transplant change the answer?
        a_only_changes_a = ta_qa["pred"] != host["hesk"]  # Should be True (blue)
        a_only_changes_b = ta_qb["pred"] != host["vorn"]  # Should be False
        b_only_changes_a = tb_qa["pred"] != host["hesk"]  # Should be False
        b_only_changes_b = tb_qb["pred"] != host["vorn"]  # Should be True (blue)
        joint_changes_a = tab_qa["pred"] != host["hesk"]
        joint_changes_b = tab_qb["pred"] != host["vorn"]

        # Determine fusion status
        a_leaks = a_only_changes_b  # A transplant changed B -> fused
        b_leaks = b_only_changes_a  # B transplant changed A -> fused

        if not a_leaks and not b_leaks:
            status = "SEPARATE"
        elif a_leaks and b_leaks:
            status = "FUSED"
        else:
            status = "PARTIAL"

        result.update({
            "a_transplant": {
                "changes_a": a_only_changes_a, "changes_b": a_only_changes_b,
                "qa_pred": ta_qa["pred"], "qb_pred": ta_qb["pred"],
            },
            "b_transplant": {
                "changes_a": b_only_changes_a, "changes_b": b_only_changes_b,
                "qa_pred": tb_qa["pred"], "qb_pred": tb_qb["pred"],
            },
            "joint_transplant": {
                "changes_a": joint_changes_a, "changes_b": joint_changes_b,
                "qa_pred": tab_qa["pred"], "qb_pred": tab_qb["pred"],
            },
            "status": status,
        })

        layer_results.append(result)

        sym = {"SEPARATE": "S", "FUSED": "F", "PARTIAL": "P"}[status]
        print(f"  Layer {layer:2d}: [{sym}] "
              f"A-transplant->(A:{ta_qa['pred']},B:{ta_qb['pred']}) "
              f"B-transplant->(A:{tb_qa['pred']},B:{tb_qb['pred']}) "
              f"Joint->(A:{tab_qa['pred']},B:{tab_qb['pred']})")

    return layer_results


def run_rn_comparison(model, tok):
    """R^n probe comparison: cosine similarity between world representations.

    For each layer, compute cosine similarity between the 4 worlds' hidden
    states. If R^n structure matches native structure, cosine-similar worlds
    should have similar behavioral effects under transplant.
    """
    print("\n=== R^n comparison: cosine similarity structure ===")

    prompts = {w["label"]: make_prompt(w["hesk"], w["vorn"]) for w in WORLDS}

    # Get hidden states for all worlds
    all_hs = {}
    for label, prompt in prompts.items():
        hs, _ = get_hidden_states(model, tok, prompt)
        all_hs[label] = hs

    n_layers = len(all_hs["00"])
    rn_results = []

    for layer in range(0, n_layers, 4):  # Sample every 4th layer
        # Mean-pool over sequence for a single representation per world
        vecs = {}
        for label in ["00", "01", "10", "11"]:
            vecs[label] = all_hs[label][layer][0].mean(dim=0)

        # Compute pairwise cosine similarities
        sims = {}
        for l1 in ["00", "01", "10", "11"]:
            for l2 in ["00", "01", "10", "11"]:
                if l1 < l2:
                    sims[f"{l1}-{l2}"] = cosine_sim(
                        vecs[l1].unsqueeze(0), vecs[l2].unsqueeze(0)
                    )

        # Key comparison: does 00-10 (differ on A) have same sim as 00-01 (differ on B)?
        # In R^n with orthogonal facts, they should be equal.
        # In R^n with correlated facts, one may dominate.
        rn_results.append({
            "layer": layer,
            "sim_differ_A": sims.get("00-10", 0),  # same B, different A
            "sim_differ_B": sims.get("00-01", 0),  # same A, different B
            "sim_differ_both": sims.get("00-11", 0),  # different both
            "sim_same_A": sims.get("01-11", 0),  # same A=red? No, 01=red, 11=blue
        })

        print(f"  Layer {layer:2d}: "
              f"differ-A={sims.get('00-10', 0):.4f}  "
              f"differ-B={sims.get('00-01', 0):.4f}  "
              f"differ-both={sims.get('00-11', 0):.4f}")

    return rn_results


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/results/fusion_fission_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Fusion-Fission v1 ===")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"Output: {out_dir}\n")

    t0 = time.time()
    model, tok = load_model()
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # Step 1: Baseline
    baseline, acc = run_baseline(model, tok)
    if acc < 0.75:
        print(f"\nWARNING: Baseline accuracy {acc:.1%} < 75%. "
              "Results may not be interpretable. Continuing anyway for signal.")

    # Step 2: Fusion-fission scan
    t1 = time.time()
    ff_results = run_fusion_fission_scan(model, tok)
    print(f"\nFusion-fission scan: {time.time() - t1:.1f}s")

    # Step 3: R^n comparison
    t2 = time.time()
    rn_results = run_rn_comparison(model, tok)
    print(f"R^n comparison: {time.time() - t2:.1f}s")

    # Step 4: Find transitions
    print("\n=== Transition analysis ===")
    transitions = []
    for i in range(len(ff_results) - 1):
        if ff_results[i]["status"] != ff_results[i+1]["status"]:
            transitions.append({
                "from_layer": ff_results[i]["layer"],
                "to_layer": ff_results[i+1]["layer"],
                "from_status": ff_results[i]["status"],
                "to_status": ff_results[i+1]["status"],
            })
            print(f"  Layer {ff_results[i]['layer']}->{ff_results[i+1]['layer']}: "
                  f"{ff_results[i]['status']}->{ff_results[i+1]['status']}")

    if not transitions:
        print("  No transitions detected — facts remain in same status across all layers.")

    # Save
    results = {
        "model": MODEL_ID,
        "device": DEVICE,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline": baseline,
        "baseline_accuracy": acc,
        "fusion_fission": ff_results,
        "rn_comparison": rn_results,
        "transitions": transitions,
        "total_forwards": len(ff_results) * 6 + 8 + len(rn_results) * 4,
    }

    out_path = out_dir / "fusion_fission_v1_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")
    print(f"Total time: {time.time() - t0:.1f}s")
    print(f"Total forwards: ~{results['total_forwards']}")

    return results


if __name__ == "__main__":
    main()
