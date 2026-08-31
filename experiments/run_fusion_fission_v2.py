"""
Fusion-Fission v2: Repaired instrument per Codex audit.

Fixes all five Codex-identified defects:
1. 100% baseline retrieval (template/entity/value sweep)
2. Continuous 2x2 causal transfer matrix K_l (not binary labels)
3. Correct layer alignment (post-block hook for both capture and patch)
4. Fact-position-only transplant (query tokens untouched)
5. Self-patch identity control

The transfer matrix at each layer:
  K_l = [[dA/dA, dA/dB],
         [dB/dA, dB/dB]]
where dX/dY = logit shift in X-answer when Y-fact is changed via transplant.
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

CONFIGS = [
    {
        "storage_template": "{a} has the color {va}. {b} has the color {vb}. To summarize: {a} is {va} and {b} is {vb}.",
        "query_template": "\nThe color of {q} is",
        "entities": ("HESK", "VORN"),
        "values": ("red", "blue"),
        "id": "T3_HESK_VORN_redblue",
    },
    {
        "storage_template": "{a}: {va}. {b}: {vb}.",
        "query_template": "\n{q}:",
        "entities": ("PLIM", "KROT"),
        "values": ("hot", "cold"),
        "id": "T4_PLIM_KROT_hotcold",
    },
    {
        "storage_template": "{a}: {va}. {b}: {vb}.",
        "query_template": "\n{q}:",
        "entities": ("ZOG", "MIP"),
        "values": ("big", "small"),
        "id": "T4_ZOG_MIP_bigsmall",
    },
]


def make_storage(cfg, va, vb):
    return cfg["storage_template"].format(
        a=cfg["entities"][0], b=cfg["entities"][1], va=va, vb=vb
    )


def make_full(cfg, va, vb, q_name):
    storage = make_storage(cfg, va, vb)
    query = cfg["query_template"].format(q=q_name)
    return storage + query


def get_logit_diff(model, tok, prompt, v0, v1):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    v0_id = tok.encode(v0, add_special_tokens=False)[0]
    v1_id = tok.encode(v1, add_special_tokens=False)[0]
    return float(logits[v0_id] - logits[v1_id])


def transplant_facts_at_layer(model, tok, cfg, host_va, host_vb, donor_va, donor_vb, q_name, layer):
    """
    Fact-position-only transplant with correct layer alignment.

    Both donor and host run the FULL prompt (storage + query) to avoid BPE
    boundary mismatch (storage-only tokenization merges the final period
    differently). Only storage-token positions are transplanted.
    """
    v0, v1 = cfg["values"]

    donor_full = make_full(cfg, donor_va, donor_vb, q_name)
    host_full = make_full(cfg, host_va, host_vb, q_name)

    donor_ids = tok(donor_full, return_tensors="pt").input_ids.to(DEVICE)
    host_ids = tok(host_full, return_tensors="pt").input_ids.to(DEVICE)

    if donor_ids.shape[1] != host_ids.shape[1]:
        return None, f"length mismatch: donor={donor_ids.shape[1]} host={host_ids.shape[1]}"

    host_storage_ids = tok(make_storage(cfg, host_va, host_vb), return_tensors="pt").input_ids.to(DEVICE)
    storage_len = host_storage_ids.shape[1]

    captured = {}

    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            captured["val"] = output[0].detach().clone()
        else:
            captured["val"] = output.detach().clone()

    target_layer = model.model.layers[layer]
    h = target_layer.register_forward_hook(capture_hook)
    with torch.no_grad():
        model(donor_ids)
    h.remove()

    def patch_hook(module, input, output):
        if isinstance(output, tuple):
            hs = output[0].clone()
            hs[0, :storage_len] = captured["val"][0, :storage_len]
            return (hs,) + output[1:]
        else:
            hs = output.clone()
            hs[0, :storage_len] = captured["val"][0, :storage_len]
            return hs

    h = target_layer.register_forward_hook(patch_hook)
    with torch.no_grad():
        out = model(host_ids)
    h.remove()

    logits = out.logits[0, -1]
    v0_id = tok.encode(v0, add_special_tokens=False)[0]
    v1_id = tok.encode(v1, add_special_tokens=False)[0]
    return float(logits[v0_id] - logits[v1_id]), None


def check_baseline(model, tok, cfg):
    v0, v1 = cfg["values"]
    nameA, nameB = cfg["entities"]
    worlds = [(0, 0), (0, 1), (1, 0), (1, 1)]
    results = []
    all_correct = True
    min_margin = float("inf")

    for wa, wb in worlds:
        va = v0 if wa == 0 else v1
        vb = v0 if wb == 0 else v1
        for q_name, q_idx in [(nameA, 0), (nameB, 1)]:
            expected_idx = wa if q_idx == 0 else wb
            expected = v0 if expected_idx == 0 else v1
            prompt = make_full(cfg, va, vb, q_name)
            diff = get_logit_diff(model, tok, prompt, v0, v1)
            pred = v0 if diff > 0 else v1
            margin = diff if expected_idx == 0 else -diff
            if pred != expected:
                all_correct = False
            min_margin = min(min_margin, margin)
            results.append({
                "world": f"{wa}{wb}", "query": q_name,
                "expected": expected, "pred": pred,
                "margin": round(margin, 3),
            })

    return all_correct, min_margin, results


def compute_transfer_matrix(model, tok, cfg, layer):
    """
    Compute 2x2 causal transfer matrix K_l.

    For each base world (diagonal: 00 and 11), change one fact via
    fact-position transplant and measure the logit shift in both answers.
    """
    v0, v1 = cfg["values"]
    nameA, nameB = cfg["entities"]
    entries = []

    for change_which in [0, 1]:
        for from_val in [0, 1]:
            base_a = from_val
            base_b = from_val
            donor_a = base_a
            donor_b = base_b
            if change_which == 0:
                donor_a = 1 - base_a
            else:
                donor_b = 1 - base_b

            base_va = v0 if base_a == 0 else v1
            base_vb = v0 if base_b == 0 else v1
            donor_va = v0 if donor_a == 0 else v1
            donor_vb = v0 if donor_b == 0 else v1

            for q_name, q_idx in [(nameA, 0), (nameB, 1)]:
                base_prompt = make_full(cfg, base_va, base_vb, q_name)
                base_diff = get_logit_diff(model, tok, base_prompt, v0, v1)

                transplant_diff, err = transplant_facts_at_layer(
                    model, tok, cfg, base_va, base_vb, donor_va, donor_vb, q_name, layer
                )
                if transplant_diff is None:
                    return None, entries, err

                shift = transplant_diff - base_diff

                entries.append({
                    "change": "A" if change_which == 0 else "B",
                    "from_val": from_val,
                    "query": "A" if q_idx == 0 else "B",
                    "base_diff": round(base_diff, 3),
                    "transplant_diff": round(transplant_diff, 3),
                    "shift": round(shift, 3),
                })

    dA_when_changeA = np.mean([e["shift"] for e in entries if e["change"] == "A" and e["query"] == "A"])
    dB_when_changeA = np.mean([e["shift"] for e in entries if e["change"] == "A" and e["query"] == "B"])
    dA_when_changeB = np.mean([e["shift"] for e in entries if e["change"] == "B" and e["query"] == "A"])
    dB_when_changeB = np.mean([e["shift"] for e in entries if e["change"] == "B" and e["query"] == "B"])

    K = [[dA_when_changeA, dA_when_changeB],
         [dB_when_changeA, dB_when_changeB]]

    return K, entries, None


def classify_from_K(K, diag_thresh=1.0, offdiag_thresh=0.5):
    diag_A = abs(K[0][0])
    diag_B = abs(K[1][1])
    offdiag_AB = abs(K[1][0])
    offdiag_BA = abs(K[0][1])

    strong_diag = diag_A > diag_thresh and diag_B > diag_thresh
    weak_offdiag = offdiag_AB < offdiag_thresh and offdiag_BA < offdiag_thresh
    strong_offdiag = offdiag_AB > diag_thresh or offdiag_BA > diag_thresh
    weak_diag = diag_A < offdiag_thresh and diag_B < offdiag_thresh

    if strong_diag and weak_offdiag:
        return "SEPARATE"
    elif strong_offdiag:
        return "FUSED"
    elif weak_diag:
        return "NO_CONTROL"
    else:
        return "PARTIAL"


def self_patch_control(model, tok, cfg, layer):
    """Self-patch: transplant from identical world. Must produce ~0 shift."""
    v0, v1 = cfg["values"]
    nameA = cfg["entities"][0]
    base_prompt = make_full(cfg, v0, v0, nameA)
    base_diff = get_logit_diff(model, tok, base_prompt, v0, v1)
    transplant_diff, _ = transplant_facts_at_layer(
        model, tok, cfg, v0, v0, v0, v0, nameA, layer
    )
    if transplant_diff is None:
        return float("inf")
    return abs(transplant_diff - base_diff)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/results/fusion_fission_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Fusion-Fission v2: Repaired Instrument ===\n")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()

    all_results = {}

    for cfg in CONFIGS:
        print(f"\n--- Config: {cfg['id']} ---")

        ok, min_margin, baseline = check_baseline(model, tok, cfg)
        print(f"  Baseline: {'PASS' if ok else 'FAIL'} (min margin {min_margin:.2f})")
        if not ok:
            print("  SKIPPING — baseline failed")
            fails = [r for r in baseline if r["pred"] != r["expected"]]
            for f in fails:
                print(f"    FAIL: w{f['world']}/{f['query']} pred={f['pred']} exp={f['expected']} m={f['margin']:.2f}")
            all_results[cfg["id"]] = {"baseline": "FAIL", "details": baseline}
            continue

        n_layers = model.config.num_hidden_layers

        sp0 = self_patch_control(model, tok, cfg, 0)
        sp_mid = self_patch_control(model, tok, cfg, n_layers // 2)
        print(f"  Self-patch: L0={sp0:.4f}, L{n_layers//2}={sp_mid:.4f}")
        if sp0 > 0.1 or sp_mid > 0.1:
            print("  WARNING: self-patch non-zero — instrument suspect")

        layer_results = []
        for layer in range(n_layers):
            K, entries, err = compute_transfer_matrix(model, tok, cfg, layer)
            if K is None:
                print(f"  L{layer:2d}: SKIP ({err})")
                continue

            status = classify_from_K(K)
            lr = {
                "layer": layer,
                "K": [[round(x, 3) for x in row] for row in K],
                "status": status,
            }
            layer_results.append(lr)

            print(f"  L{layer:2d} [{status:11s}]  "
                  f"K=[{K[0][0]:+6.2f} {K[0][1]:+6.2f}; "
                  f"{K[1][0]:+6.2f} {K[1][1]:+6.2f}]")

        statuses = [lr["status"] for lr in layer_results]
        n_sep = statuses.count("SEPARATE")
        n_fus = statuses.count("FUSED")
        n_par = statuses.count("PARTIAL")
        n_noc = statuses.count("NO_CONTROL")
        transitions = sum(1 for i in range(1, len(statuses)) if statuses[i] != statuses[i - 1])

        print(f"\n  Summary: SEP={n_sep} FUS={n_fus} PAR={n_par} NOC={n_noc} transitions={transitions}")

        all_results[cfg["id"]] = {
            "baseline": "PASS",
            "min_margin": round(min_margin, 3),
            "self_patch_l0": round(sp0, 4),
            "self_patch_mid": round(sp_mid, 4),
            "layer_results": layer_results,
            "summary": {
                "SEPARATE": n_sep, "FUSED": n_fus,
                "PARTIAL": n_par, "NO_CONTROL": n_noc,
                "transitions": transitions,
            },
        }

    print("\n\n=== Cross-config replication ===")
    passing = [cid for cid, r in all_results.items() if r.get("baseline") == "PASS" and r.get("layer_results")]
    if len(passing) >= 2:
        for i, c1 in enumerate(passing):
            for c2 in passing[i + 1:]:
                r1 = all_results[c1]["layer_results"]
                r2 = all_results[c2]["layer_results"]
                n = min(len(r1), len(r2))
                agree = sum(1 for j in range(n) if r1[j]["status"] == r2[j]["status"])
                print(f"  {c1} vs {c2}: {agree}/{n} ({100 * agree / n:.0f}%)")
    else:
        print("  Fewer than 2 passing configs")

    out_path = out_dir / "fusion_fission_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
