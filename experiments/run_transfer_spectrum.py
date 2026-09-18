"""Experiment: Transfer Rho Across Language Model Architectures and Training.

Core question: Is the low behavioral transfer (rho=0.158) found on
Qwen3-1.7B universal to language models, or model-specific?

If universal: it's evidence that fine-tuning/RLHF fundamentally cannot
guarantee held-out behavioral preservation in transformers.

Protocol (parallel to yeast epistasis transfer test):
- 20 code cells with known digit answers (analog of yeast genes)
- 2 surface pairs x 6 perturbation types = 12 conditions (analog of gene partners)
- Transfer test: split conditions 8/4, correlate cell-pair distances
- Compare transfer rho across models

Models tested:
- GPT-2 small (124M) - different architecture, different training
- Qwen3-1.7B-Base (1.7B) - our original finding
- Pythia-70M at training checkpoints - same architecture, different training stage
"""
import copy
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

DIGIT_LABELS = [str(d) for d in range(10)] + ["OTHER"]


def tv(a, b):
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum()


class LightAdapter:
    """Minimal model adapter for transfer test. Handles any HF causal LM."""

    def __init__(self, model_id, revision=None, device="cpu", dtype=torch.float16):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
        tok_kwargs = {"trust_remote_code": True}
        if revision:
            kwargs["revision"] = revision
            tok_kwargs["revision"] = revision

        self.tok = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
        self.mdl = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.mdl.eval()
        self.device = device
        if device != "cpu":
            self.mdl = self.mdl.to(device)

        self.digit_ids = {}
        for d in range(10):
            toks = self.tok.encode(str(d), add_special_tokens=False)
            if len(toks) == 1:
                self.digit_ids[d] = toks[0]
            else:
                alt = self.tok.encode(f" {d}", add_special_tokens=False)
                if len(alt) >= 1:
                    self.digit_ids[d] = alt[-1]
                else:
                    raise ValueError(f"Cannot get single token for digit {d}")

        self.n_calls = 0

    def _extract_11bin(self, logits):
        probs = torch.softmax(logits.float(), dim=0).cpu().numpy().astype(np.float64)
        bins = np.zeros(11, dtype=np.float64)
        for d in range(10):
            bins[d] = probs[self.digit_ids[d]]
        bins[10] = max(0.0, 1.0 - bins[:10].sum())
        return bins

    def get_state(self, text):
        ids = self.tok.encode(text, add_special_tokens=False, return_tensors="pt")
        ids = ids.to(self.device)
        with torch.no_grad():
            out = self.mdl(ids, use_cache=True)
        self.n_calls += 1
        return copy.deepcopy(out.past_key_values)

    def get_dist(self, state, suffix):
        ids = self.tok.encode(suffix, add_special_tokens=False, return_tensors="pt")
        ids = ids.to(self.device)
        st = copy.deepcopy(state)
        with torch.no_grad():
            out = self.mdl(ids, past_key_values=st, use_cache=True)
        self.n_calls += 1
        return self._extract_11bin(out.logits[0, -1, :])

    def cleanup(self):
        del self.mdl
        del self.tok
        gc.collect()
        if self.device != "cpu":
            torch.cuda.empty_cache()


# --- Cell Generation ---

def make_cells():
    """Generate 20 diverse code cells with known digit answers."""
    cells = []
    configs = [
        ("x", 3, "simple assignment"),
        ("y", 7, "simple assignment"),
        ("z", 1, "simple assignment"),
        ("n", 5, "simple assignment"),
        ("count", 9, "simple assignment"),
        ("val", 0, "simple assignment"),
        ("num", 4, "simple assignment"),
        ("total", 6, "simple assignment"),
        ("score", 8, "simple assignment"),
        ("result", 2, "simple assignment"),
        ("a", 5, "with arithmetic context"),
        ("b", 3, "with arithmetic context"),
        ("temp", 7, "with indirection"),
        ("data", 1, "with indirection"),
        ("idx", 4, "loop context"),
        ("flag", 0, "conditional context"),
        ("width", 6, "domain variable"),
        ("depth", 9, "domain variable"),
        ("level", 2, "domain variable"),
        ("step", 8, "domain variable"),
    ]

    for var, digit, style in configs:
        if style == "simple assignment":
            prefix = f"{var} = {digit}\n"
            query = f"print({var})\n# Output: "
        elif style == "with arithmetic context":
            other = (digit + 3) % 10
            prefix = f"{var} = {digit}\n_ = {other} + 1\n"
            query = f"print({var})\n# Output: "
        elif style == "with indirection":
            prefix = f"_{var} = {digit}\n{var} = _{var}\n"
            query = f"print({var})\n# Output: "
        elif style == "loop context":
            prefix = f"for i in range(1):\n    {var} = {digit}\n"
            query = f"print({var})\n# Output: "
        elif style == "conditional context":
            prefix = f"if True:\n    {var} = {digit}\n"
            query = f"print({var})\n# Output: "
        elif style == "domain variable":
            prefix = f"{var} = {digit}\n"
            query = f"print({var})\n# Output: "

        collision_digit = (digit + 5) % 10
        cells.append({
            "name": f"{var}_{digit}",
            "prefix": prefix,
            "query": query,
            "answer": digit,
            "perturbations": {
                "empty": "",
                "identity": f"{var} = {var}\n",
                "heavy": f"{var} = {var} + 1\n",
                "collision": f"# note: {collision_digit}\n",
                "pass": "pass\n",
                "reassign_same": f"{var} = {digit}\n",
            },
        })
    return cells


SURFACE_PAIRS = [
    ("minimal", "", "\n"),
    ("comment", "# setup\n", "# init\n"),
]


def run_model(model_id, cells, revision=None, device="cpu", label=None):
    """Run full transfer test on one model. Returns dict of results."""
    label = label or model_id.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"MODEL: {label}")
    print(f"  id: {model_id}, revision: {revision}, device: {device}")
    print(f"{'='*60}")

    t0 = time.time()
    adapter = LightAdapter(model_id, revision=revision, device=device)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    n_cells = len(cells)
    perturbation_names = list(cells[0]["perturbations"].keys())
    n_perts = len(perturbation_names)
    n_pairs = len(SURFACE_PAIRS)

    condition_names = []
    for sp_name, _, _ in SURFACE_PAIRS:
        for p_name in perturbation_names:
            condition_names.append(f"{sp_name}_{p_name}")
    n_conditions = len(condition_names)

    # Matrix: cells x conditions, each entry = TV distance between surfaces
    tv_matrix = np.zeros((n_cells, n_conditions))

    # Also track validation: does model predict correct digit?
    correct = 0
    total = 0

    t1 = time.time()
    for ci, cell in enumerate(cells):
        prefix_state = adapter.get_state(cell["prefix"])

        # Validation: bare query
        bare_dist = adapter.get_dist(prefix_state, cell["query"])
        predicted = int(np.argmax(bare_dist[:10]))
        if predicted == cell["answer"]:
            correct += 1
        total += 1

        cond_idx = 0
        for sp_name, surf_a, surf_b in SURFACE_PAIRS:
            for p_name in perturbation_names:
                pert_text = cell["perturbations"][p_name]
                suffix_a = surf_a + pert_text + cell["query"]
                suffix_b = surf_b + pert_text + cell["query"]

                dist_a = adapter.get_dist(prefix_state, suffix_a)
                dist_b = adapter.get_dist(prefix_state, suffix_b)
                tv_matrix[ci, cond_idx] = tv(dist_a, dist_b)
                cond_idx += 1

        del prefix_state
        gc.collect()

        elapsed = time.time() - t1
        rate = (ci + 1) / elapsed if elapsed > 0 else 0
        eta = (n_cells - ci - 1) / rate if rate > 0 else 0
        if (ci + 1) % 5 == 0 or ci == n_cells - 1:
            print(f"  Cell {ci+1}/{n_cells} done ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    inference_time = time.time() - t1
    print(f"  Inference: {inference_time:.1f}s, {adapter.n_calls} forward passes")
    print(f"  Validation: {correct}/{total} cells predicted correctly ({correct/total:.0%})")

    # --- Transfer Test ---
    # Split conditions into train (8) and test (4)
    # Use a fixed split: first 8 conditions = train, last 4 = test
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_conditions)
    n_train_cond = n_conditions * 2 // 3
    train_cond_idx = sorted(perm[:n_train_cond])
    test_cond_idx = sorted(perm[n_train_cond:])

    train_profiles = tv_matrix[:, train_cond_idx]
    test_profiles = tv_matrix[:, test_cond_idx]

    # Filter out cells with zero variance (no signal)
    train_vars = np.var(train_profiles, axis=1)
    test_vars = np.var(test_profiles, axis=1)
    active_cells = (train_vars > 1e-10) & (test_vars > 1e-10)
    n_active = np.sum(active_cells)

    if n_active < 5:
        print(f"  WARNING: Only {n_active} active cells. Transfer test unreliable.")
        transfer_rho = float('nan')
        transfer_p = float('nan')
    else:
        train_p = train_profiles[active_cells]
        test_p = test_profiles[active_cells]

        train_dists = pdist(train_p, metric='euclidean')
        test_dists = pdist(test_p, metric='euclidean')

        transfer_rho, transfer_p_val = spearmanr(train_dists, test_dists)
        n_pairs_used = len(train_dists)
        print(f"\n  TRANSFER TEST:")
        print(f"    Active cells: {n_active}/{n_cells}")
        print(f"    Cell pairs: {n_pairs_used}")
        print(f"    Train conditions: {[condition_names[i] for i in train_cond_idx]}")
        print(f"    Test conditions:  {[condition_names[i] for i in test_cond_idx]}")
        print(f"    Transfer rho: {transfer_rho:.3f} (p={transfer_p_val:.2e})")

    # Also compute per-condition mean TV (sensitivity profile)
    mean_tvs = np.mean(tv_matrix, axis=0)
    print(f"\n  Mean TV by condition:")
    for i, name in enumerate(condition_names):
        print(f"    {name:>25s}: {mean_tvs[i]:.4f}")

    # Multiple random train/test splits for robustness
    rho_samples = []
    for seed in range(20):
        rng_s = np.random.RandomState(seed * 7 + 13)
        perm_s = rng_s.permutation(n_conditions)
        tr_idx = sorted(perm_s[:n_train_cond])
        te_idx = sorted(perm_s[n_train_cond:])
        tr_p = tv_matrix[active_cells][:, tr_idx]
        te_p = tv_matrix[active_cells][:, te_idx]
        if tr_p.shape[0] >= 3:
            tr_d = pdist(tr_p, metric='euclidean')
            te_d = pdist(te_p, metric='euclidean')
            rho_s, _ = spearmanr(tr_d, te_d)
            if not np.isnan(rho_s):
                rho_samples.append(rho_s)

    if rho_samples:
        mean_rho = np.mean(rho_samples)
        std_rho = np.std(rho_samples)
        print(f"\n  Robustness (20 random splits):")
        print(f"    Mean rho: {mean_rho:.3f} +/- {std_rho:.3f}")
        print(f"    Range: [{min(rho_samples):.3f}, {max(rho_samples):.3f}]")
    else:
        mean_rho = float('nan')
        std_rho = float('nan')

    adapter.cleanup()

    return {
        "model_id": model_id,
        "label": label,
        "revision": revision,
        "n_cells": n_cells,
        "n_active_cells": int(n_active),
        "n_conditions": n_conditions,
        "correct_predictions": correct,
        "total_predictions": total,
        "transfer_rho": float(transfer_rho) if not np.isnan(transfer_rho) else None,
        "transfer_rho_mean": float(mean_rho) if not np.isnan(mean_rho) else None,
        "transfer_rho_std": float(std_rho) if not np.isnan(std_rho) else None,
        "rho_samples": [float(r) for r in rho_samples],
        "mean_tv_by_condition": {name: float(mean_tvs[i]) for i, name in enumerate(condition_names)},
        "tv_matrix": tv_matrix.tolist(),
        "condition_names": condition_names,
        "cell_names": [c["name"] for c in cells],
        "load_time": load_time,
        "inference_time": inference_time,
        "n_forward_passes": adapter.n_calls if hasattr(adapter, 'n_calls') else 0,
    }


def main():
    print("=" * 60)
    print("TRANSFER RHO ACROSS LANGUAGE MODEL ARCHITECTURES")
    print("=" * 60)
    print(f"Testing whether rho=0.158 (Qwen3-1.7B) is universal\n")

    cells = make_cells()
    print(f"Generated {len(cells)} code cells")
    print(f"Surface pairs: {len(SURFACE_PAIRS)}")
    print(f"Perturbation types: {len(cells[0]['perturbations'])}")
    print(f"Total conditions: {len(SURFACE_PAIRS) * len(cells[0]['perturbations'])}")

    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    print(f"Device: {device}")

    results = []

    # --- Model 1: GPT-2 small (fast, CPU is fine) ---
    r = run_model("openai-community/gpt2", cells, device="cpu", label="GPT2-small-124M")
    results.append(r)
    gc.collect()

    # --- Model 2: Pythia-70M final checkpoint ---
    r = run_model("EleutherAI/pythia-70m", cells, device="cpu", label="Pythia-70M-final")
    results.append(r)
    gc.collect()

    # --- Model 3: Pythia-70M early checkpoints (training dynamics) ---
    pythia_checkpoints = ["step1", "step512", "step1000", "step4000",
                          "step16000", "step64000", "step143000"]
    for ckpt in pythia_checkpoints:
        try:
            r = run_model("EleutherAI/pythia-70m", cells, revision=ckpt,
                          device="cpu", label=f"Pythia-70M-{ckpt}")
            results.append(r)
            gc.collect()
        except Exception as e:
            print(f"  SKIP {ckpt}: {e}")

    # --- Model 4: Qwen3-1.7B (our original, use GPU for speed) ---
    qwen_device = device  # use GPU if available
    r = run_model("Qwen/Qwen3-1.7B-Base", cells, device=qwen_device,
                  label="Qwen3-1.7B-Base")
    results.append(r)
    gc.collect()
    if has_cuda:
        torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY: Transfer Rho Across Models")
    print(f"{'='*60}")
    print(f"\n{'Model':>30s} {'Params':>8s} {'Correct':>8s} {'Rho':>8s} {'Rho_std':>8s}")
    print("-" * 66)
    for r in results:
        params = ""
        if "70m" in r["label"].lower() or "70M" in r["label"]:
            params = "70M"
        elif "124" in r["label"] or "gpt2" in r["label"].lower():
            params = "124M"
        elif "1.7" in r["label"]:
            params = "1.7B"
        rho_str = f"{r['transfer_rho_mean']:.3f}" if r.get('transfer_rho_mean') else "N/A"
        std_str = f"{r.get('transfer_rho_std', 0):.3f}" if r.get('transfer_rho_std') else ""
        correct_str = f"{r['correct_predictions']}/{r['total_predictions']}"
        print(f"{r['label']:>30s} {params:>8s} {correct_str:>8s} {rho_str:>8s} {std_str:>8s}")

    # Biology comparison
    print(f"\n--- Comparison to biological systems ---")
    print(f"{'System':>30s} {'Transfer Rho':>12s}")
    print("-" * 44)
    print(f"{'GB1 protein (536K doubles)':>30s} {'0.975':>12s}")
    print(f"{'Yeast ExN (775K test pts)':>30s} {'0.965':>12s}")
    print(f"{'Yeast ExE (121K test pts)':>30s} {'0.877':>12s}")
    for r in results:
        rho_str = f"{r['transfer_rho_mean']:.3f}" if r.get('transfer_rho_mean') else "N/A"
        print(f"{r['label']:>30s} {rho_str:>12s}")

    # Save results
    out_dir = Path("experiments/results/transfer_spectrum")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # Training dynamics plot data
    training_results = [r for r in results if "Pythia-70M-step" in r.get("label", "")]
    if training_results:
        print(f"\n--- Pythia-70M Training Dynamics ---")
        print(f"{'Checkpoint':>25s} {'Correct':>8s} {'Rho_mean':>10s} {'Rho_std':>10s}")
        print("-" * 55)
        for r in training_results:
            rho_str = f"{r['transfer_rho_mean']:.3f}" if r.get('transfer_rho_mean') else "N/A"
            std_str = f"{r.get('transfer_rho_std', 0):.3f}" if r.get('transfer_rho_std') else ""
            correct_str = f"{r['correct_predictions']}/{r['total_predictions']}"
            print(f"{r['label']:>25s} {correct_str:>8s} {rho_str:>10s} {std_str:>10s}")


if __name__ == "__main__":
    main()
