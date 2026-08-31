"""
Fusion-Sensitivity v1: Can a static metric PREDICT transplant outcomes?

We know from v1-v1e that fusion = world-insensitive computation and
fission = world-sensitive computation. This experiment tests whether a
quantitative world-sensitivity score computed from hidden states can
predict the binary transplant outcome (does A-transplant change B?).

Metric: World-Sensitivity Score (WSS) at layer L
  WSS(L) = relative_divergence(L) × update_dissimilarity(L)
  where:
    relative_divergence = |convergence_A| / avg_update_magnitude
    update_dissimilarity = 1 - cosine(update_00, update_10)

Prediction target: a_transplant.changes_b from v1 (binary).

Win condition:
  - Point-biserial correlation >= 0.40 between WSS and changes_b
  - Threshold accuracy >= 0.75 (best-threshold binary classification)
  Both must hold for PASS.

CPU only, Qwen3-0.6B, same 4-world setup.
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

WORLDS = [
    {"hesk": "red",  "vorn": "red",  "label": "00"},
    {"hesk": "red",  "vorn": "blue", "label": "01"},
    {"hesk": "blue", "vorn": "red",  "label": "10"},
    {"hesk": "blue", "vorn": "blue", "label": "11"},
]


def make_prompt(hesk_color, vorn_color):
    return f"HESK is {hesk_color}. VORN is {vorn_color}. Remember these facts.\n"


def make_query(which):
    return f"What color is {which}? Answer with one word:"


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def transplant_at_layer(model, tok, donor_prompt, host_prompt, query, layer_idx):
    donor_ids = tok(donor_prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        donor_out = model(donor_ids, output_hidden_states=True)
    donor_state = donor_out.hidden_states[layer_idx]

    host_text = host_prompt + query
    host_ids = tok(host_text, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = tok(host_prompt, return_tensors="pt").input_ids.shape[1]

    replaced = [False]
    def hook_fn(module, input, output):
        if replaced[0]:
            return output
        replaced[0] = True
        if isinstance(output, tuple):
            hs = output[0].clone()
            hs[0, :prompt_len] = donor_state[0, :prompt_len]
            return (hs,) + output[1:]
        hs = output.clone()
        hs[0, :prompt_len] = donor_state[0, :prompt_len]
        return hs

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            out = model(host_ids)
        logits = out.logits[0, -1]
        red_id = tok.encode(" red", add_special_tokens=False)[0]
        blue_id = tok.encode(" blue", add_special_tokens=False)[0]
        return "red" if logits[red_id] > logits[blue_id] else "blue"
    finally:
        handle.remove()


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/results/fusion_fission_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Fusion-Sensitivity v1: Predictive World-Sensitivity Score ===\n")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()

    # --- Step 1: Compute hidden states for all 4 worlds ---
    all_hs = {}
    for w in WORLDS:
        prompt = make_prompt(w["hesk"], w["vorn"])
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        all_hs[w["label"]] = [h[0, -1].numpy() for h in out.hidden_states]

    n_layers = len(all_hs["00"]) - 1
    print(f"Model layers: {n_layers}\n")

    # --- Step 2: Compute WSS at each layer ---
    wss_scores = []
    for layer in range(1, n_layers + 1):
        prev = {lb: all_hs[lb][layer - 1] for lb in ["00", "01", "10", "11"]}
        curr = {lb: all_hs[lb][layer] for lb in ["00", "01", "10", "11"]}
        updates = {lb: curr[lb] - prev[lb] for lb in ["00", "01", "10", "11"]}

        # Update dissimilarity for A-different pairs: (00,10) and (01,11)
        ud1 = 1.0 - cosine_sim(updates["00"], updates["10"])
        ud2 = 1.0 - cosine_sim(updates["01"], updates["11"])
        update_dissim = (ud1 + ud2) / 2.0

        # Divergence rate for A-different pairs
        dist_a_curr = (np.linalg.norm(curr["00"] - curr["10"])
                       + np.linalg.norm(curr["01"] - curr["11"])) / 2.0
        dist_a_prev = (np.linalg.norm(prev["00"] - prev["10"])
                       + np.linalg.norm(prev["01"] - prev["11"])) / 2.0
        divergence = dist_a_curr - dist_a_prev

        avg_mag = np.mean([np.linalg.norm(updates[lb]) for lb in ["00", "01", "10", "11"]])
        rel_divergence = divergence / avg_mag if avg_mag > 1e-8 else 0.0

        wss = rel_divergence * update_dissim
        wss_scores.append({
            "layer": layer,
            "wss": float(wss),
            "update_dissim": float(update_dissim),
            "rel_divergence": float(rel_divergence),
            "divergence": float(divergence),
            "avg_update_mag": float(avg_mag),
        })

    # --- Step 3: Run transplant test at every layer ---
    host = WORLDS[0]
    host_prompt = make_prompt(host["hesk"], host["vorn"])
    donor_a = WORLDS[2]
    donor_a_prompt = make_prompt(donor_a["hesk"], donor_a["vorn"])
    query_b = make_query("VORN")

    transplant_results = []
    for layer in range(n_layers):
        b_pred = transplant_at_layer(model, tok, donor_a_prompt, host_prompt, query_b, layer)
        changes_b = (b_pred != host["vorn"])
        transplant_results.append({"layer": layer, "b_pred": b_pred, "changes_b": changes_b})

    # --- Step 4: Align and compute correlation ---
    # WSS is for layers 1..28, transplant is for layers 0..27
    # Match on layer index (WSS layer L measures the transformation INTO layer L)
    paired = []
    for tr in transplant_results:
        wss_entry = next((w for w in wss_scores if w["layer"] == tr["layer"]), None)
        if wss_entry is None:
            wss_entry = next((w for w in wss_scores if w["layer"] == tr["layer"] + 1), None)
        if wss_entry:
            paired.append({
                "layer": tr["layer"],
                "wss": wss_entry["wss"],
                "update_dissim": wss_entry["update_dissim"],
                "rel_divergence": wss_entry["rel_divergence"],
                "changes_b": tr["changes_b"],
            })

    print(f"Paired {len(paired)} layers for prediction test.\n")

    wss_vals = np.array([p["wss"] for p in paired])
    labels = np.array([int(p["changes_b"]) for p in paired])

    # Point-biserial correlation
    n0 = np.sum(labels == 0)
    n1 = np.sum(labels == 1)
    if n0 > 0 and n1 > 0:
        m0 = np.mean(wss_vals[labels == 0])
        m1 = np.mean(wss_vals[labels == 1])
        s = np.std(wss_vals, ddof=1)
        if s > 1e-12:
            rpb = (m1 - m0) / s * np.sqrt(n0 * n1 / len(labels)**2)
        else:
            rpb = 0.0
    else:
        rpb = 0.0

    # Best-threshold accuracy
    best_acc = 0.0
    best_thresh = 0.0
    best_direction = ">"
    sorted_wss = np.sort(np.unique(wss_vals))
    for t in sorted_wss:
        for direction in [">", "<"]:
            if direction == ">":
                preds = (wss_vals > t).astype(int)
            else:
                preds = (wss_vals < t).astype(int)
            acc = np.mean(preds == labels)
            if acc > best_acc:
                best_acc = acc
                best_thresh = t
                best_direction = direction

    # Also try each sub-metric alone
    ud_vals = np.array([p["update_dissim"] for p in paired])
    rd_vals = np.array([p["rel_divergence"] for p in paired])

    def best_threshold_acc(vals, labs):
        ba, bt = 0.0, 0.0
        for t in np.sort(np.unique(vals)):
            for d in [">", "<"]:
                p = (vals > t if d == ">" else vals < t).astype(int)
                a = np.mean(p == labs)
                if a > ba:
                    ba, bt = a, t
        return ba, bt

    ud_acc, _ = best_threshold_acc(ud_vals, labels)
    rd_acc, _ = best_threshold_acc(rd_vals, labels)

    # --- Print results ---
    print("--- Per-layer results ---")
    for p in paired:
        tag = "FUSED(B)" if p["changes_b"] else "SEPAR(B)"
        print(f"  Layer {p['layer']:2d}  WSS={p['wss']:.6f}  "
              f"ud={p['update_dissim']:.6f}  rd={p['rel_divergence']:.4f}  [{tag}]")

    print(f"\n--- Prediction metrics ---")
    print(f"  Point-biserial r(WSS, changes_b) = {rpb:.4f}  (gate >= 0.40)")
    print(f"  Best-threshold accuracy (WSS)     = {best_acc:.3f}  (gate >= 0.75)")
    print(f"  Best-threshold accuracy (ud only)  = {ud_acc:.3f}")
    print(f"  Best-threshold accuracy (rd only)  = {rd_acc:.3f}")
    print(f"  Fused count: {n1}, Separate count: {n0}, Majority baseline: {max(n0,n1)/len(labels):.3f}")

    pass_rpb = abs(rpb) >= 0.40
    pass_acc = best_acc >= 0.75
    if pass_rpb and pass_acc:
        verdict = "PASS: WSS predicts transplant outcomes"
    elif pass_rpb or pass_acc:
        verdict = "PARTIAL: one gate passed, one failed"
    else:
        verdict = "FAIL: WSS does not predict transplant outcomes"

    print(f"\n  Verdict: {verdict}")

    out_path = out_dir / "fusion_sensitivity_v1.json"
    results = {
        "wss_scores": wss_scores,
        "transplant_results": transplant_results,
        "paired": paired,
        "rpb": float(rpb),
        "best_threshold_acc": float(best_acc),
        "best_threshold": float(best_thresh),
        "best_direction": best_direction,
        "ud_acc": float(ud_acc),
        "rd_acc": float(rd_acc),
        "n_fused": int(n1),
        "n_separate": int(n0),
        "majority_baseline": float(max(n0, n1) / len(labels)),
        "verdict": verdict,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
