"""V17: Active Inference Surrogate Ablation.

Tests whether surrogate-guided exploration/exploitation (Active Inference)
improves evolution compared to standard random mutation + elitist selection.

Four conditions, all using soft prompt decode + accuracy fitness:
- C0: no_evolution       -- seed latent, no optimisation (baseline)
- C1: standard_evolution -- standard Gaussian mutation, no surrogate
- C2: active_inference   -- surrogate screens 4x candidates by EFE
- C3: active_inference_8x -- surrogate screens 8x candidates by EFE

The surrogate predicts accuracy from latent vectors using a small MLP.
EFE = -predicted_accuracy - beta * prediction_uncertainty
Beta anneals across generations (explore -> exploit).

Critical isolation:
- Same W matrix, same seed latents, same task pools
- C1 and C2 use IDENTICAL mutation geometry (hyperbolic Mobius)
- The ONLY variable is whether surrogate screening is applied
- C3 tests whether more aggressive screening helps further
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

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
    generate_nested_expression_tasks,
    print_results,
    run_experiment,
    save_results,
)
from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.decode.projection import make_row_orthonormal_W


def main():
    args = experiment_cli("V17: Active Inference surrogate ablation")

    n_seeds = 1 if args.diagnostic else args.seeds
    curvature = args.curvature

    # V17 always uses nested tasks + accuracy fitness
    nested_difficulty = args.difficulty if args.difficulty in ("easy_nested", "sweet_spot") else "easy_nested"
    fitness_mode = "accuracy"

    print("=" * 70, flush=True)
    print("V17: ACTIVE INFERENCE SURROGATE ABLATION", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Seeds: {n_seeds}", flush=True)
    print(f"Curvature: {curvature}", flush=True)
    print(f"Difficulty: {nested_difficulty}", flush=True)
    print(f"Fitness mode: {fitness_mode}", flush=True)
    print(f"Diagnostic: {args.diagnostic}", flush=True)

    # Load model
    print("\nLoading model...", flush=True)
    encoder = LLMEncoder(
        model_name=args.model,
        quantization=args.quantization,
    )

    # Auto-calibrate
    cal = auto_calibrate(encoder)
    print(f"Calibration: {json.dumps(cal, indent=2)}", flush=True)

    embed_dim = cal["embed_dim"]
    target_rms = cal["embedding_rms"]
    d_latent = encoder.latent_dim

    # Verify soft prompt compatibility
    compatible = check_soft_prompt_compatibility(encoder)
    if not compatible:
        print("ERROR: Model does not support inputs_embeds. Cannot run V17.", flush=True)
        sys.exit(1)

    # Shared W matrix -- same for ALL conditions
    num_soft_tokens = 8
    d_out = num_soft_tokens * embed_dim
    W = make_row_orthonormal_W(d_latent, d_out, seed=1234)
    W = W.to(device=encoder._device)
    print(f"W shape: {W.shape} (d_latent={d_latent}, d_out={d_out})", flush=True)

    # Task generation
    n_test = 25 if args.diagnostic else args.test_tasks_per_depth
    n_train = 80 if args.diagnostic else args.train_tasks_per_depth
    train_tasks, test_tasks = generate_nested_expression_tasks(
        n_train=n_train, n_test=n_test,
        seed=42, difficulty=nested_difficulty,
    )
    print(f"Train: {len(train_tasks)}, Test: {len(test_tasks)}", flush=True)

    # Build conditions -- all use hyperbolic geometry (proven architecture)
    base_decode_kwargs = dict(
        mode=DecodeMode.SOFT_PROMPT,
        W_soft=W,
        embed_dim=embed_dim,
        num_soft_tokens=num_soft_tokens,
        target_rms=target_rms,
        curvature=curvature,
        max_new_tokens=1024,
        temperature=0.3,
    )

    decode_cfg = DecodeConfig(geometry="hyperbolic", **base_decode_kwargs)

    conditions = [
        ExperimentCondition(
            name="no_evolution",
            decode_cfg=decode_cfg,
            evolve=False,
        ),
        ExperimentCondition(
            name="standard_evolution",
            decode_cfg=decode_cfg,
            evolve=True,
            use_surrogate=False,
        ),
        ExperimentCondition(
            name="active_inference_4x",
            decode_cfg=decode_cfg,
            evolve=True,
            use_surrogate=True,
        ),
    ]

    spec = ExperimentSpec(
        name="v17_active_inference",
        conditions=conditions,
        primary_comparison=("standard_evolution", "active_inference_4x"),
        secondary_comparisons=[
            ("no_evolution", "standard_evolution"),
            ("no_evolution", "active_inference_4x"),
        ],
        bonferroni_n=1,
        depths=[2],
    )

    evo = EvolutionParams(
        generations=args.evo_gens,
        population_size=args.evo_pop,
        tasks_per_gen=args.evo_tasks,
        noise_scale=args.noise_scale,
        curvature=curvature,
        fitness_mode=fitness_mode,
        surrogate_expansion=4,
    )

    # Print config summary
    ball_radius = (1.0 / math.sqrt(curvature)) * 0.95
    print(f"\nEvolution: {evo.generations} gens x {evo.population_size} pop", flush=True)
    print(f"Surrogate expansion: {evo.surrogate_expansion}x", flush=True)
    print(f"Ball radius: {ball_radius:.4f}", flush=True)
    print(f"Target RMS: {target_rms:.6f}", flush=True)

    # Run
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

    # Config for reproducibility
    config = {
        "model": args.model,
        "quantization": args.quantization,
        "seeds": n_seeds,
        "curvature": curvature,
        "task_type": "nested",
        "difficulty": nested_difficulty,
        "fitness_mode": fitness_mode,
        "depths": [2],
        "n_train": len(train_tasks),
        "n_test": len(test_tasks),
        "evo_gens": evo.generations,
        "evo_pop": evo.population_size,
        "evo_tasks_per_gen": evo.tasks_per_gen,
        "noise_scale": evo.noise_scale,
        "surrogate_expansion": evo.surrogate_expansion,
        "num_soft_tokens": num_soft_tokens,
        "d_latent": d_latent,
        "embed_dim": embed_dim,
        "target_rms": target_rms,
        "ball_radius": ball_radius,
        "calibration": cal,
    }

    # Output
    print_results(results)

    suffix = "_diagnostic" if args.diagnostic else ""
    out_path = Path(args.output) if args.output else (
        Path(__file__).parent / f"v17_active_inference{suffix}.json"
    )
    save_results(results, out_path, config=config)

    print(f"\nElapsed: {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
