"""Activation Probe: Hidden-state scouting (A) + MLP redistribution analysis (C).

Codex-recommended probes for NeurIPS paper.
- A: Compare residual stream per layer for baseline vs perturbation (identify target layers)
- C: Analyze MLP activation patterns (bridges to Shi et al.'s activation redistribution)

Uses teacher-forced forward passes to control for different generated text.
Measures: per-layer residual norm/cosine, MLP output norm, activation sparsity, top-k overlap.
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from latent_reasoning.core.encoder import LLMEncoder


THINK_TOKEN_ID = 151667


def build_prompt(tokenizer, query: str) -> str:
    """Build chat-formatted prompt."""
    system_msg = "Answer to the best of your ability."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


class MLPHookCollector:
    """Registers hooks on MLP sublayers to collect activation data."""

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.mlp_outputs = {}  # layer_idx -> tensor
        self.mlp_gate_acts = {}  # layer_idx -> gate activations (pre-nonlinearity)

    def register(self):
        """Register forward hooks on all MLP layers."""
        for i, layer in enumerate(self.model.model.layers):
            hook = layer.mlp.register_forward_hook(
                self._make_hook(i)
            )
            self.hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # output is the MLP output tensor
            self.mlp_outputs[layer_idx] = output.detach().cpu().float()
            # Try to capture gate activations if available
            if hasattr(module, 'gate_proj') and hasattr(module, 'act_fn'):
                # For Qwen3 SwiGLU: gate = act_fn(gate_proj(x)) * up_proj(x)
                # We capture the full MLP output; gate acts need deeper hooks
                pass
        return hook_fn

    def clear(self):
        """Clear collected data."""
        self.mlp_outputs.clear()
        self.mlp_gate_acts.clear()

    def remove_hooks(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def compute_layer_stats(hidden_states_base, hidden_states_pert, mlp_base, mlp_pert,
                        position_idx: int = -1) -> List[dict]:
    """Compare hidden states and MLP outputs per layer at a given position."""
    n_layers = len(hidden_states_base) - 1  # first entry is embedding output
    stats = []

    for li in range(n_layers):
        # Residual stream comparison (layer li+1 since index 0 = embedding)
        h_base = hidden_states_base[li + 1][0, position_idx, :].float()
        h_pert = hidden_states_pert[li + 1][0, position_idx, :].float()

        # L2 distance and cosine similarity
        l2_dist = (h_base - h_pert).norm().item()
        cosine = F.cosine_similarity(h_base.unsqueeze(0), h_pert.unsqueeze(0)).item()
        base_norm = h_base.norm().item()
        pert_norm = h_pert.norm().item()
        norm_ratio = pert_norm / max(base_norm, 1e-8)

        layer_stat = {
            "layer": li,
            "residual_l2": l2_dist,
            "residual_cosine": cosine,
            "residual_base_norm": base_norm,
            "residual_pert_norm": pert_norm,
            "residual_norm_ratio": norm_ratio,
        }

        # MLP output comparison
        if li in mlp_base and li in mlp_pert:
            m_base = mlp_base[li][0, position_idx, :].float()
            m_pert = mlp_pert[li][0, position_idx, :].float()

            mlp_l2 = (m_base - m_pert).norm().item()
            mlp_cosine = F.cosine_similarity(
                m_base.unsqueeze(0), m_pert.unsqueeze(0)
            ).item()
            mlp_base_norm = m_base.norm().item()
            mlp_pert_norm = m_pert.norm().item()

            # Sparsity: fraction of near-zero activations
            base_sparsity = (m_base.abs() < 0.01).float().mean().item()
            pert_sparsity = (m_pert.abs() < 0.01).float().mean().item()

            # Top-k overlap (k=100)
            k = min(100, m_base.shape[0])
            base_topk = set(m_base.abs().topk(k).indices.tolist())
            pert_topk = set(m_pert.abs().topk(k).indices.tolist())
            topk_overlap = len(base_topk & pert_topk) / k

            # Activation entropy (on absolute values, normalized)
            base_abs = m_base.abs()
            base_probs = base_abs / base_abs.sum().clamp_min(1e-8)
            base_entropy = -(base_probs * base_probs.clamp_min(1e-10).log()).sum().item()

            pert_abs = m_pert.abs()
            pert_probs = pert_abs / pert_abs.sum().clamp_min(1e-8)
            pert_entropy = -(pert_probs * pert_probs.clamp_min(1e-10).log()).sum().item()

            layer_stat.update({
                "mlp_l2": mlp_l2,
                "mlp_cosine": mlp_cosine,
                "mlp_base_norm": mlp_base_norm,
                "mlp_pert_norm": mlp_pert_norm,
                "mlp_base_sparsity": base_sparsity,
                "mlp_pert_sparsity": pert_sparsity,
                "mlp_topk100_overlap": topk_overlap,
                "mlp_base_entropy": base_entropy,
                "mlp_pert_entropy": pert_entropy,
            })

        stats.append(layer_stat)

    return stats


def run_forward(model, tokenizer, prompt, soft_prompt, device, collector):
    """Run forward pass with optional soft prompt, collecting hidden states + MLP outputs."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        text_embeds = model.get_input_embeddings()(input_ids)

        if soft_prompt is not None:
            sp = soft_prompt.to(model.dtype).to(device)
            combined_embeds = torch.cat([sp, text_embeds], dim=1)
            soft_mask = torch.ones(
                1, sp.size(1), dtype=attention_mask.dtype, device=device
            )
            combined_mask = torch.cat([soft_mask, attention_mask], dim=1)
        else:
            combined_embeds = text_embeds
            combined_mask = attention_mask

        collector.clear()
        outputs = model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    return outputs, collector.mlp_outputs.copy()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Activation probe (A+C)")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--n-tasks", type=int, default=5,
                        help="Number of tasks to probe (default 5 for speed)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("ACTIVATION PROBE (Hidden-state scouting + MLP redistribution)")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    encoder = LLMEncoder(args.model, quantization="4bit")
    model = encoder.model
    tokenizer = encoder.tokenizer
    device = encoder._device

    # Calibrate
    with torch.no_grad():
        embed_weight = model.get_input_embeddings().weight
        target_rms = embed_weight.float().square().mean().sqrt().item()
    embed_dim = embed_weight.shape[1]
    n_layers = len(model.model.layers)
    print(f"  Embed dim: {embed_dim}, Layers: {n_layers}, Target RMS: {target_rms:.5f}")

    # Generate tasks
    from run_latent_sensitivity import generate_nested_tasks
    tasks = generate_nested_tasks(n_tasks=args.n_tasks, seed=42,
                                  difficulty_filter="sweet_spot")
    print(f"  Tasks: {len(tasks)}")

    # Set up hooks
    collector = MLPHookCollector(model)
    collector.register()

    # Prepare conditions
    noise_gen = torch.Generator().manual_seed(2024)

    def make_noise(n_tokens):
        sp = torch.randn(1, n_tokens, embed_dim, generator=noise_gen)
        current_rms = sp.square().mean().sqrt().clamp_min(1e-8)
        return sp * (target_rms / current_rms)

    conditions = {
        "baseline": None,
        "noise_2tok": make_noise(2),
        "noise_8tok": make_noise(8),
        "zero_2tok": torch.zeros(1, 2, embed_dim),
    }

    # Run probes
    all_results = []
    for ti, task in enumerate(tasks):
        prompt = build_prompt(tokenizer, task.prompt)
        task_result = {"task_id": task.task_id, "layers": []}

        # Run baseline first
        print(f"\n  [{ti+1}/{len(tasks)}] {task.task_id}:")
        base_out, base_mlp = run_forward(
            model, tokenizer, prompt, None, device, collector
        )
        base_hidden = [h.cpu().float() for h in base_out.hidden_states]

        # Run each perturbation condition
        for cond_name, sp in conditions.items():
            if cond_name == "baseline":
                continue

            pert_out, pert_mlp = run_forward(
                model, tokenizer, prompt, sp, device, collector
            )
            pert_hidden = [h.cpu().float() for h in pert_out.hidden_states]

            # Analyze last prompt position
            # For baseline, last position is -1
            # For perturbation, last position is offset by number of soft tokens
            n_soft = sp.size(1) if sp is not None else 0

            # Compare at the LAST prompt token position (where generation logits come from)
            stats = compute_layer_stats(
                base_hidden, pert_hidden, base_mlp, pert_mlp,
                position_idx=-1  # last position in each case
            )

            # Find layers with largest divergence
            max_l2_layer = max(stats, key=lambda x: x["residual_l2"])
            min_cos_layer = min(stats, key=lambda x: x["residual_cosine"])

            print(f"    {cond_name}: max_l2=L{max_l2_layer['layer']} "
                  f"({max_l2_layer['residual_l2']:.2f}), "
                  f"min_cos=L{min_cos_layer['layer']} "
                  f"({min_cos_layer['residual_cosine']:.4f})")

            task_result["layers"].append({
                "condition": cond_name,
                "per_layer_stats": stats,
            })

            del pert_out, pert_hidden, pert_mlp
            gc.collect()

        del base_out, base_hidden, base_mlp
        gc.collect()
        torch.cuda.empty_cache()

        all_results.append(task_result)

    # Aggregate: find which layers show largest perturbation effects
    print("\n" + "=" * 60)
    print("AGGREGATE: Layer-wise perturbation effects (noise_2tok)")
    print("=" * 60)

    for li in range(n_layers):
        l2s = []
        coss = []
        mlp_l2s = []
        for r in all_results:
            for c in r["layers"]:
                if c["condition"] == "noise_2tok":
                    s = c["per_layer_stats"][li]
                    l2s.append(s["residual_l2"])
                    coss.append(s["residual_cosine"])
                    if "mlp_l2" in s:
                        mlp_l2s.append(s["mlp_l2"])

        mean_l2 = np.mean(l2s)
        mean_cos = np.mean(coss)
        mean_mlp_l2 = np.mean(mlp_l2s) if mlp_l2s else 0

        if mean_l2 > 1.0 or mean_cos < 0.95:  # Only show significant layers
            print(f"  Layer {li:2d}: "
                  f"resid_l2={mean_l2:7.2f}, cos={mean_cos:.4f}, "
                  f"mlp_l2={mean_mlp_l2:7.2f}")

    # Save
    collector.remove_hooks()
    output_path = args.output or str(
        Path(__file__).parent / "activation_probe_results.json"
    )
    output = {
        "experiment": "activation_probe",
        "model": args.model,
        "n_tasks": len(tasks),
        "n_layers": n_layers,
        "conditions": list(conditions.keys()),
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
