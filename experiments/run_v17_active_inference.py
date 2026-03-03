"""V17: Active Inference Surrogate Ablation.

Tests whether surrogate-guided exploration/exploitation (Active Inference)
improves evolution compared to standard random mutation + elitist selection.

Three conditions, all using soft prompt decode + accuracy fitness:
- C0: no_evolution       -- seed latent, no optimisation (baseline)
- C1: standard_evolution -- standard Gaussian mutation, no surrogate
- C2: active_inference   -- surrogate screens 4x candidates by EFE

The surrogate predicts accuracy from latent vectors using a small MLP.
EFE = -predicted_accuracy - beta * prediction_uncertainty
Beta anneals across generations (explore -> exploit).

Critical isolation:
- Same W matrix, same seed latents, same task pools
- C1 and C2 use IDENTICAL mutation geometry (hyperbolic Mobius)
- The ONLY variable is whether surrogate screening is applied
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from harness import (
    DecodeConfig,
    EvolutionParams,
    ExperimentCondition,
    ExperimentSpec,
    experiment_cli,
    make_base_decode_kwargs,
    print_results,
    run_experiment,
    save_results,
    setup_soft_prompt_experiment,
)


def main():
    args = experiment_cli("V17: Active Inference surrogate ablation")
    nested_difficulty = args.difficulty if args.difficulty in ("easy_nested", "sweet_spot") else "easy_nested"

    setup = setup_soft_prompt_experiment(
        args, "V17: ACTIVE INFERENCE SURROGATE ABLATION",
        nested_difficulty=nested_difficulty,
    )

    decode_cfg = DecodeConfig(geometry="hyperbolic", **make_base_decode_kwargs(setup))

    conditions = [
        ExperimentCondition(name="no_evolution", decode_cfg=decode_cfg, evolve=False),
        ExperimentCondition(name="standard_evolution", decode_cfg=decode_cfg, evolve=True),
        ExperimentCondition(name="active_inference_4x", decode_cfg=decode_cfg, evolve=True, use_surrogate=True),
    ]

    spec = ExperimentSpec(
        name="v17_active_inference",
        conditions=conditions,
        primary_comparison=("standard_evolution", "active_inference_4x"),
        secondary_comparisons=[
            ("no_evolution", "standard_evolution"),
            ("no_evolution", "active_inference_4x"),
        ],
        bonferroni_n=1, depths=[2],
    )

    evo = EvolutionParams(
        generations=args.evo_gens, population_size=args.evo_pop,
        tasks_per_gen=args.evo_tasks, noise_scale=args.noise_scale,
        curvature=setup.curvature, fitness_mode="accuracy",
        surrogate_expansion=4,
    )

    ball_radius = (1.0 / math.sqrt(setup.curvature)) * 0.95
    print(f"\nEvolution: {evo.generations} gens x {evo.population_size} pop", flush=True)
    print(f"Surrogate expansion: {evo.surrogate_expansion}x", flush=True)
    print(f"Ball radius: {ball_radius:.4f}", flush=True)

    start = time.time()
    results = run_experiment(
        spec=spec, encoder=setup.encoder,
        train_tasks=setup.train_tasks, test_tasks=setup.test_tasks,
        evo=evo, n_seeds=setup.n_seeds,
    )
    elapsed = time.time() - start
    results["elapsed_seconds"] = elapsed

    config = {
        "model": args.model, "quantization": args.quantization,
        "seeds": setup.n_seeds, "curvature": setup.curvature,
        "task_type": "nested", "difficulty": nested_difficulty,
        "fitness_mode": "accuracy", "depths": [2],
        "n_train": len(setup.train_tasks), "n_test": len(setup.test_tasks),
        "evo_gens": evo.generations, "evo_pop": evo.population_size,
        "surrogate_expansion": evo.surrogate_expansion,
        "ball_radius": ball_radius, "calibration": setup.cal,
    }

    print_results(results)

    suffix = "_diagnostic" if args.diagnostic else ""
    out_path = Path(args.output) if args.output else (
        Path(__file__).parent / f"v17_active_inference{suffix}.json"
    )
    save_results(results, out_path, config=config)
    print(f"\nElapsed: {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
