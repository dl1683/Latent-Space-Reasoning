"""V15: Geometry Isolation Under Same Conditioning Channel.

Answers the core Codex C+ question: does hyperbolic geometry improve
reasoning when the conditioning channel is identical (soft prompt)?

Three conditions, all using soft prompt decode:
- C0: no_evolution       -- seed latent, no optimisation (baseline)
- C1: euclidean_softprompt -- Euclidean Gaussian mutation constrained to L2 ball
- C2: hyperbolic_softprompt -- Hyperbolic Mobius mutation in Poincare ball

Critical isolation:
- Same W matrix (seed=1234) for all conditions
- Same model, same tasks, same RNG seeds per condition
- Matched ball radius: Euclidean L2 ball radius = 1/sqrt(c) * 0.95
- Both paths apply radial_tanh_squash before W projection
- The ONLY variable is the mutation geometry (flat vs curved)
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
    generate_all_unique_tasks,
    print_results,
    run_experiment,
    save_results,
    split_train_test,
)
from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.decode.projection import make_row_orthonormal_W


def main():
    args = experiment_cli("V15: Geometry isolation under same conditioning")

    # Override defaults for V15
    n_seeds = 1 if args.diagnostic else args.seeds
    curvature = args.curvature

    print("=" * 70, flush=True)
    print("V15: GEOMETRY ISOLATION UNDER SAME CONDITIONING", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Seeds: {n_seeds}", flush=True)
    print(f"Curvature: {curvature}", flush=True)
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
        print("ERROR: Model does not support inputs_embeds. Cannot run V15.", flush=True)
        sys.exit(1)
    print(f"Soft prompt compatible: {compatible}", flush=True)

    # Shared W matrix -- same for ALL conditions
    num_soft_tokens = 8
    d_out = num_soft_tokens * embed_dim
    W = make_row_orthonormal_W(d_latent, d_out, seed=1234)
    W = W.to(device=encoder._device)  # Keep float32 for precision
    print(f"W shape: {W.shape} (d_latent={d_latent}, d_out={d_out})", flush=True)
    print(f"W device: {W.device}", flush=True)

    # Task generation
    depths = args.diagnostic and [2] or [2, 3]
    tasks_by_depth = generate_all_unique_tasks(args.branching, depths)
    total_tasks = sum(len(v) for v in tasks_by_depth.values())
    print(f"Tasks: {total_tasks} total across depths {depths}", flush=True)

    train_tasks, test_tasks = split_train_test(
        tasks_by_depth, args.test_tasks_per_depth, args.train_tasks_per_depth,
    )
    print(f"Train: {len(train_tasks)}, Test: {len(test_tasks)}", flush=True)

    # Build conditions -- identical decode config except geometry
    base_decode_kwargs = dict(
        mode=DecodeMode.SOFT_PROMPT,
        W_soft=W,
        embed_dim=embed_dim,
        num_soft_tokens=num_soft_tokens,
        target_rms=target_rms,
        curvature=curvature,
        max_new_tokens=1024,  # Models need room for chain-of-thought
        temperature=0.3,
    )

    conditions = [
        ExperimentCondition(
            name="no_evolution",
            decode_cfg=DecodeConfig(geometry="euclidean", **base_decode_kwargs),
            evolve=False,
        ),
        ExperimentCondition(
            name="euclidean_softprompt",
            decode_cfg=DecodeConfig(geometry="euclidean", **base_decode_kwargs),
            evolve=True,
        ),
        ExperimentCondition(
            name="hyperbolic_softprompt",
            decode_cfg=DecodeConfig(geometry="hyperbolic", **base_decode_kwargs),
            evolve=True,
        ),
    ]

    spec = ExperimentSpec(
        name="v15_geometry_isolation",
        conditions=conditions,
        primary_comparison=("euclidean_softprompt", "hyperbolic_softprompt"),
        secondary_comparisons=[
            ("no_evolution", "euclidean_softprompt"),
            ("no_evolution", "hyperbolic_softprompt"),
        ],
        bonferroni_n=1,  # Only 1 primary comparison
        depths=depths,
    )

    evo = EvolutionParams(
        generations=args.evo_gens,
        population_size=args.evo_pop,
        tasks_per_gen=args.evo_tasks,
        noise_scale=args.noise_scale,
        curvature=curvature,
    )

    # Print config summary
    ball_radius = (1.0 / math.sqrt(curvature)) * 0.95
    print(f"\nEvolution: {evo.generations} gens x {evo.population_size} pop", flush=True)
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
        "branching": args.branching,
        "depths": depths,
        "train_per_depth": args.train_tasks_per_depth,
        "test_per_depth": args.test_tasks_per_depth,
        "evo_gens": evo.generations,
        "evo_pop": evo.population_size,
        "evo_tasks_per_gen": evo.tasks_per_gen,
        "noise_scale": evo.noise_scale,
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
        Path(__file__).parent / f"v15_geometry_isolation{suffix}.json"
    )
    save_results(results, out_path, config=config)

    print(f"\nElapsed: {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
