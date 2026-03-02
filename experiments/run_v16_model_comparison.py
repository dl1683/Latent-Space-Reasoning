"""V16: Model Exploration — Hybrid SSM vs Pure Transformer.

Tests whether SSM-hybrid architectures respond differently to latent
conditioning via soft prompt injection.

Models tested (all fit in 24GB VRAM):
- Qwen3-4B: Pure Transformer, ~4GB Q4
- Granite-4.0-Tiny (~1B): IBM Hybrid SSM, ~1GB FP16
- Falcon-H1-1.5B: TII Transformer+Mamba hybrid, ~1.5GB FP16

Per model:
1. Load + auto_calibrate() — model-specific embed_dim, RMS, norms
2. check_soft_prompt_compatibility() — verify inputs_embeds support
3. Run V15-style geometry isolation (Euclidean vs Hyperbolic + soft prompt)
4. 3 seeds per model (budget: ~2h per model x 3 seeds = ~6h total)

Risk: SSM models may not support inputs_embeds.
Mitigation: check_soft_prompt_compatibility() detects at startup.
Incompatible models fall back to RNG-seed conditioning.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from harness import (
    DecodeConfig,
    DecodeMode,
    EvolutionParams,
    ExperimentCondition,
    ExperimentSpec,
    auto_calibrate,
    check_soft_prompt_compatibility,
    experiment_cli,
    generate_all_unique_tasks,
    print_results,
    run_experiment,
    save_results,
    split_train_test,
)
from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.decode.projection import make_row_orthonormal_W


@dataclass
class ModelConfig:
    """Configuration for a model under test."""
    name: str
    hf_id: str
    quantization: str
    arch_type: str  # "transformer", "hybrid_ssm", "hybrid_mamba"


# Models from MODEL_DIRECTORY.md
MODELS = [
    ModelConfig(
        name="qwen3_4b",
        hf_id="Qwen/Qwen3-4B",
        quantization="4bit",
        arch_type="transformer",
    ),
    ModelConfig(
        name="granite_tiny",
        hf_id="ibm-granite/granite-4.0-tiny-preview",
        quantization="none",
        arch_type="hybrid_ssm",
    ),
    ModelConfig(
        name="falcon_h1_1.5b",
        hf_id="tiiuae/Falcon-H1-1.5B-Instruct",
        quantization="none",
        arch_type="hybrid_mamba",
    ),
]


def run_single_model(
    model_cfg: ModelConfig,
    args,
    n_seeds: int,
    curvature: float,
) -> Optional[dict]:
    """Run the V15-style geometry isolation for a single model."""

    print(f"\n{'=' * 70}", flush=True)
    print(f"MODEL: {model_cfg.name} ({model_cfg.hf_id})", flush=True)
    print(f"  Arch: {model_cfg.arch_type}, Quant: {model_cfg.quantization}", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Load
    try:
        encoder = LLMEncoder(
            model_name=model_cfg.hf_id,
            quantization=model_cfg.quantization,
        )
    except Exception as e:
        print(f"ERROR loading {model_cfg.hf_id}: {e}", flush=True)
        return None

    # Calibrate
    cal = auto_calibrate(encoder)
    print(f"Calibration: {json.dumps(cal, indent=2)}", flush=True)

    # Check soft prompt compatibility
    soft_prompt_ok = check_soft_prompt_compatibility(encoder)
    print(f"Soft prompt compatible: {soft_prompt_ok}", flush=True)

    embed_dim = cal["embed_dim"]
    target_rms = cal["embedding_rms"]
    d_latent = encoder.latent_dim

    # Shared W
    num_soft_tokens = 8
    d_out = num_soft_tokens * embed_dim
    W = make_row_orthonormal_W(d_latent, d_out, seed=1234)
    W = W.to(device=encoder._device, dtype=encoder.model.dtype)
    print(f"W shape: {W.shape}, device: {W.device}", flush=True)

    # Tasks
    depths = [2] if args.diagnostic else [2, 3]
    tasks_by_depth = generate_all_unique_tasks(args.branching, depths)
    train_tasks, test_tasks = split_train_test(
        tasks_by_depth, args.test_tasks_per_depth, args.train_tasks_per_depth,
    )

    # Choose decode mode based on compatibility
    if soft_prompt_ok:
        decode_mode = DecodeMode.SOFT_PROMPT
        print("Using SOFT_PROMPT decode", flush=True)
    else:
        decode_mode = DecodeMode.RNG_SEED
        print("WARNING: Falling back to RNG_SEED decode", flush=True)

    # Build conditions
    base_kwargs = dict(
        mode=decode_mode,
        curvature=curvature,
        max_new_tokens=1024,  # Models need room for chain-of-thought
        temperature=0.3,
    )

    if decode_mode == DecodeMode.SOFT_PROMPT:
        base_kwargs.update(
            W_soft=W,
            embed_dim=embed_dim,
            num_soft_tokens=num_soft_tokens,
            target_rms=target_rms,
        )

    conditions = [
        ExperimentCondition(
            name="no_evolution",
            decode_cfg=DecodeConfig(geometry="euclidean", **base_kwargs),
            evolve=False,
        ),
        ExperimentCondition(
            name="euclidean",
            decode_cfg=DecodeConfig(geometry="euclidean", **base_kwargs),
            evolve=True,
        ),
        ExperimentCondition(
            name="hyperbolic",
            decode_cfg=DecodeConfig(geometry="hyperbolic", **base_kwargs),
            evolve=True,
        ),
    ]

    spec = ExperimentSpec(
        name=f"v16_{model_cfg.name}",
        conditions=conditions,
        primary_comparison=("euclidean", "hyperbolic"),
        bonferroni_n=1,
        depths=depths,
    )

    evo = EvolutionParams(
        generations=args.evo_gens,
        population_size=args.evo_pop,
        tasks_per_gen=args.evo_tasks,
        noise_scale=args.noise_scale,
        curvature=curvature,
    )

    start = time.time()
    results = run_experiment(
        spec=spec,
        encoder=encoder,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        evo=evo,
        n_seeds=n_seeds,
    )
    elapsed = time.time() - start
    results["elapsed_seconds"] = elapsed
    results["model_config"] = {
        "name": model_cfg.name,
        "hf_id": model_cfg.hf_id,
        "quantization": model_cfg.quantization,
        "arch_type": model_cfg.arch_type,
        "calibration": cal,
        "soft_prompt_compatible": soft_prompt_ok,
        "decode_mode": decode_mode.value,
    }

    print_results(results)

    # Free GPU memory before next model
    del encoder
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    args = experiment_cli("V16: Model comparison (Transformer vs Hybrid)")
    n_seeds = 1 if args.diagnostic else min(args.seeds, 3)
    curvature = args.curvature

    print("=" * 70, flush=True)
    print("V16: MODEL COMPARISON — TRANSFORMER vs HYBRID SSM", flush=True)
    print("=" * 70, flush=True)
    print(f"Models: {[m.name for m in MODELS]}", flush=True)
    print(f"Seeds: {n_seeds}", flush=True)

    all_results = {}
    for model_cfg in MODELS:
        result = run_single_model(model_cfg, args, n_seeds, curvature)
        if result is not None:
            all_results[model_cfg.name] = result

    # Summary
    print(f"\n{'#' * 70}", flush=True)
    print("CROSS-MODEL SUMMARY", flush=True)
    print(f"{'#' * 70}", flush=True)
    for name, res in all_results.items():
        stats = res["statistics"]["per_condition"]
        print(f"\n{name}:", flush=True)
        for cond, s in stats.items():
            print(f"  {cond:25s}: {s['mean']*100:.1f}% +/- {s['std']*100:.1f}%", flush=True)
        print(f"  Verdict: {res['primary_verdict']}", flush=True)

    # Save combined
    suffix = "_diagnostic" if args.diagnostic else ""
    out_path = Path(args.output) if args.output else (
        Path(__file__).parent / f"v16_model_comparison{suffix}.json"
    )
    combined = {
        "experiment": "v16_model_comparison",
        "models": list(all_results.keys()),
        "per_model": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
