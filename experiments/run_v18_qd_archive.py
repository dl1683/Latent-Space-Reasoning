"""V18: Quality-Diversity Archive Evolution.

Tests whether QD (Dominated Novelty Search) improves evolution compared to
standard elitist selection by maintaining diverse high-quality solutions.

Three conditions, all using soft prompt decode + accuracy fitness:
- C0: no_evolution       -- seed latent, no optimisation (baseline)
- C1: standard_evolution -- standard elitist selection (no archive)
- C2: qd_evolution       -- DNS archive with novelty-weighted selection

The QD archive stores diverse solutions using Dominated Novelty Search.
Parents are sampled from the archive using farthest-point sampling,
providing stepping stones for exploration instead of greedy convergence.

Key hypothesis: QD should find MORE correct latents for different task types,
exploiting the proven result that different latents fix different failure modes
(32% accuracy range, p=0.006 from sensitivity analysis).

Novel combination: QD + Poincare ball geometry (no existing work).
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
    QDParams,
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
    args = experiment_cli("V18: Quality-Diversity archive evolution")

    n_seeds = 1 if args.diagnostic else args.seeds
    curvature = args.curvature

    # V18 always uses nested tasks + accuracy fitness
    nested_difficulty = args.difficulty if args.difficulty in ("easy_nested", "sweet_spot") else "easy_nested"

    print("=" * 70, flush=True)
    print("V18: QUALITY-DIVERSITY ARCHIVE EVOLUTION", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Seeds: {n_seeds}", flush=True)
    print(f"Curvature: {curvature}", flush=True)
    print(f"Difficulty: {nested_difficulty}", flush=True)
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
        print("ERROR: Model does not support inputs_embeds. Cannot run V18.", flush=True)
        sys.exit(1)

    # Shared W matrix
    num_soft_tokens = 8
    d_out = num_soft_tokens * embed_dim
    W = make_row_orthonormal_W(d_latent, d_out, seed=1234)
    W = W.to(device=encoder._device)
    print(f"W shape: {W.shape}", flush=True)

    # Task generation
    n_test = 25 if args.diagnostic else args.test_tasks_per_depth
    n_train = 80 if args.diagnostic else args.train_tasks_per_depth
    train_tasks, test_tasks = generate_nested_expression_tasks(
        n_train=n_train, n_test=n_test,
        seed=42, difficulty=nested_difficulty,
    )
    print(f"Train: {len(train_tasks)}, Test: {len(test_tasks)}", flush=True)

    # All conditions use hyperbolic geometry
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

    qd_params = QDParams(
        bd_dim=16,
        rff_gamma=0.1,
        novelty_weight=0.3,
        novelty_k=5,
        archive_size=100,
        domination_threshold=0.15,
    )

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
        ),
        ExperimentCondition(
            name="qd_evolution",
            decode_cfg=decode_cfg,
            evolve=True,
            use_qd=True,
            qd_params=qd_params,
        ),
    ]

    spec = ExperimentSpec(
        name="v18_qd_archive",
        conditions=conditions,
        primary_comparison=("standard_evolution", "qd_evolution"),
        secondary_comparisons=[
            ("no_evolution", "standard_evolution"),
            ("no_evolution", "qd_evolution"),
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
        fitness_mode="accuracy",
    )

    # Print config summary
    ball_radius = (1.0 / math.sqrt(curvature)) * 0.95
    print(f"\nEvolution: {evo.generations} gens x {evo.population_size} pop", flush=True)
    print(f"QD: bd_dim={qd_params.bd_dim}, novelty_weight={qd_params.novelty_weight}", flush=True)
    print(f"QD: archive_size={qd_params.archive_size}", flush=True)
    print(f"Ball radius: {ball_radius:.4f}", flush=True)

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

    config = {
        "model": args.model,
        "quantization": args.quantization,
        "seeds": n_seeds,
        "curvature": curvature,
        "task_type": "nested",
        "difficulty": nested_difficulty,
        "fitness_mode": "accuracy",
        "depths": [2],
        "n_train": len(train_tasks),
        "n_test": len(test_tasks),
        "evo_gens": evo.generations,
        "evo_pop": evo.population_size,
        "evo_tasks_per_gen": evo.tasks_per_gen,
        "noise_scale": evo.noise_scale,
        "num_soft_tokens": num_soft_tokens,
        "d_latent": d_latent,
        "embed_dim": embed_dim,
        "target_rms": target_rms,
        "ball_radius": ball_radius,
        "qd_params": {
            "bd_dim": qd_params.bd_dim,
            "rff_gamma": qd_params.rff_gamma,
            "novelty_weight": qd_params.novelty_weight,
            "novelty_k": qd_params.novelty_k,
            "archive_size": qd_params.archive_size,
            "domination_threshold": qd_params.domination_threshold,
        },
        "calibration": cal,
    }

    print_results(results)

    suffix = "_diagnostic" if args.diagnostic else ""
    out_path = Path(args.output) if args.output else (
        Path(__file__).parent / f"v18_qd_archive{suffix}.json"
    )
    save_results(results, out_path, config=config)

    print(f"\nElapsed: {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
