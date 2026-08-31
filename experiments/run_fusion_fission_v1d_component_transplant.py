"""
Fusion-Fission v1d: Component-level transplant.

Question: Is fusion caused by the MLP or by attention?
Method: At each layer, transplant ONLY the attention output or ONLY
the MLP output from a world that differs in one fact, and check if
the other fact's answer changes.

- Transplant attention output from A-different donor: if B changes,
  attention carries fusion-causing info
- Transplant MLP output from A-different donor: if B changes, MLP
  carries fusion-causing info
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cpu"
DTYPE = torch.float32

SEPARATE_LAYERS = [5, 6, 8, 22, 23, 24]
FUSED_LAYERS = [2, 7, 13, 14, 18]


def make_query(hesk_color, vorn_color, ask_which):
    return (f"HESK is {hesk_color}. VORN is {vorn_color}. "
            f"Remember these facts.\nWhat color is {ask_which}?")


def get_logit_diff(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(ids)
    logits = out.logits[0, -1]
    red_id = tok.encode("red", add_special_tokens=False)[0]
    blue_id = tok.encode("blue", add_special_tokens=False)[0]
    return float(logits[red_id] - logits[blue_id])


def component_transplant(model, tok, base_prompt, donor_prompt, layer, component):
    """
    Run base_prompt but at `layer`, replace one component's output
    with what it would produce on donor_prompt.

    component: "attn", "mlp", or "full" (whole hidden state)
    """
    ids_base = tok(base_prompt, return_tensors="pt").input_ids.to(DEVICE)
    ids_donor = tok(donor_prompt, return_tensors="pt").input_ids.to(DEVICE)

    # First pass: collect donor's component outputs
    donor_attn_out = {}
    donor_mlp_out = {}
    donor_full_out = {}

    def capture_attn(module, input, output):
        if isinstance(output, tuple):
            donor_attn_out["val"] = output[0].detach().clone()
        else:
            donor_attn_out["val"] = output.detach().clone()

    def capture_mlp(module, input, output):
        donor_mlp_out["val"] = output.detach().clone()

    target_layer = model.model.layers[layer]
    h1 = target_layer.self_attn.register_forward_hook(capture_attn)
    h2 = target_layer.mlp.register_forward_hook(capture_mlp)

    with torch.no_grad():
        out_donor = model(ids_donor, output_hidden_states=True)
    donor_full_out["val"] = out_donor.hidden_states[layer + 1].detach().clone()

    h1.remove()
    h2.remove()

    # Second pass: run base but patch in donor's component
    hooks = []

    if component == "attn":
        def patch_attn(module, input, output):
            if isinstance(output, tuple):
                new = list(output)
                new[0] = donor_attn_out["val"]
                return tuple(new)
            return donor_attn_out["val"]
        hooks.append(target_layer.self_attn.register_forward_hook(patch_attn))

    elif component == "mlp":
        def patch_mlp(module, input, output):
            return donor_mlp_out["val"]
        hooks.append(target_layer.mlp.register_forward_hook(patch_mlp))

    elif component == "full":
        def patch_full(module, input, output):
            if isinstance(output, tuple):
                new = list(output)
                new[0] = donor_full_out["val"]
                return tuple(new)
            return donor_full_out["val"]
        hooks.append(target_layer.register_forward_hook(patch_full))

    with torch.no_grad():
        out = model(ids_base)

    for h in hooks:
        h.remove()

    logits = out.logits[0, -1]
    red_id = tok.encode("red", add_special_tokens=False)[0]
    blue_id = tok.encode("blue", add_special_tokens=False)[0]
    pred = "red" if logits[red_id] > logits[blue_id] else "blue"
    margin = float(logits[red_id] - logits[blue_id])

    return pred, margin


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/results/fusion_fission_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Fusion-Fission v1d: Component-Level Transplant ===\n")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()

    # Base world: A=red, B=red
    # A-different donor: A=blue, B=red (only A differs)
    # We query about B (VORN) to detect fusion:
    # If transplanting A-different donor changes B-answer -> FUSED
    base_qb = make_query("red", "red", "VORN")      # expect: red
    donor_qb = make_query("blue", "red", "VORN")     # expect: red (B unchanged)

    # Also query A to verify transplant works
    base_qa = make_query("red", "red", "HESK")       # expect: red
    donor_qa = make_query("blue", "red", "HESK")     # expect: blue

    # Baseline
    base_a_diff = get_logit_diff(model, tok, base_qa)
    base_b_diff = get_logit_diff(model, tok, base_qb)
    print(f"  Baseline A: red-blue = {base_a_diff:.2f} ({'red' if base_a_diff > 0 else 'blue'})")
    print(f"  Baseline B: red-blue = {base_b_diff:.2f} ({'red' if base_b_diff > 0 else 'blue'})")

    test_layers = sorted(set(FUSED_LAYERS + SEPARATE_LAYERS + [3, 4, 9, 10, 15, 20]))
    results = []

    for layer in test_layers:
        tag = "FUSED" if layer in FUSED_LAYERS else "SEPARATE" if layer in SEPARATE_LAYERS else "PARTIAL"

        # Full transplant: does B change?
        full_b_pred, full_b_margin = component_transplant(model, tok, base_qb, donor_qb, layer, "full")
        # Attention-only transplant: does B change?
        attn_b_pred, attn_b_margin = component_transplant(model, tok, base_qb, donor_qb, layer, "attn")
        # MLP-only transplant: does B change?
        mlp_b_pred, mlp_b_margin = component_transplant(model, tok, base_qb, donor_qb, layer, "mlp")

        # Also check A-query with full transplant to verify it works
        full_a_pred, full_a_margin = component_transplant(model, tok, base_qa, donor_qa, layer, "full")

        b_changed_full = (full_b_pred != "red")
        b_changed_attn = (attn_b_pred != "red")
        b_changed_mlp = (mlp_b_pred != "red")

        lr = {
            "layer": layer,
            "status": tag,
            "full_transplant_b": {"pred": full_b_pred, "margin": full_b_margin, "b_changed": b_changed_full},
            "attn_transplant_b": {"pred": attn_b_pred, "margin": attn_b_margin, "b_changed": b_changed_attn},
            "mlp_transplant_b": {"pred": mlp_b_pred, "margin": mlp_b_margin, "b_changed": b_changed_mlp},
            "full_transplant_a": {"pred": full_a_pred, "margin": full_a_margin},
        }
        results.append(lr)

        b_shift_full = base_b_diff - full_b_margin
        b_shift_attn = base_b_diff - attn_b_margin
        b_shift_mlp = base_b_diff - mlp_b_margin

        driver = ""
        if b_changed_attn and not b_changed_mlp:
            driver = "ATTN drives B-change"
        elif b_changed_mlp and not b_changed_attn:
            driver = "MLP drives B-change"
        elif b_changed_attn and b_changed_mlp:
            driver = "BOTH drive B-change"
        elif b_changed_full:
            driver = "FULL changes B but neither component alone does"
        else:
            driver = "B unchanged (no fusion)"

        print(f"  Layer {layer:2d} [{tag:8s}]  "
              f"full_B={full_b_pred}({b_shift_full:+.1f})  "
              f"attn_B={attn_b_pred}({b_shift_attn:+.1f})  "
              f"mlp_B={mlp_b_pred}({b_shift_mlp:+.1f})  "
              f"| {driver}")

    # Summary
    print("\n=== Summary ===")
    fused_results = [r for r in results if r["status"] == "FUSED"]
    sep_results = [r for r in results if r["status"] == "SEPARATE"]

    for group_name, group in [("FUSED", fused_results), ("SEPARATE", sep_results)]:
        n = len(group)
        n_b_full = sum(1 for r in group if r["full_transplant_b"]["b_changed"])
        n_b_attn = sum(1 for r in group if r["attn_transplant_b"]["b_changed"])
        n_b_mlp = sum(1 for r in group if r["mlp_transplant_b"]["b_changed"])
        print(f"  {group_name} ({n}): B changes with full={n_b_full}, attn_only={n_b_attn}, mlp_only={n_b_mlp}")

    out_path = out_dir / "fusion_fission_v1d_component_transplant.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "baseline_a_diff": base_a_diff, "baseline_b_diff": base_b_diff}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
