"""V18: Quality-Diversity Archive Evolution.

Tests whether QD (Dominated Novelty Search) improves evolution compared to
standard elitist selection by maintaining diverse high-quality solutions.

Three conditions, all using soft prompt decode + accuracy fitness:
- C0: no_evolution       -- seed latent, no optimisation (baseline)
- C1: standard_evolution -- standard elitist selection (no archive)
- C2: qd_evolution       -- DNS archive with novelty-weighted selection

Key hypothesis: QD should find MORE correct latents for different task types,
exploiting the proven result that different latents fix different failure modes
(32% accuracy range, p=0.006 from sensitivity analysis).

Novel combination: QD + Poincare ball geometry (no existing work).
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
    QDParams,
    experiment_cli,
    make_base_decode_kwargs,
    print_results,
    run_experiment,
    save_results,
    setup_soft_prompt_experiment,
)


def main():
    args = experiment_cli("V18: Quality-Diversity archive evolution")
    nested_difficulty = args.difficulty if args.difficulty in ("easy_nested", "sweet_spot") else "easy_nested"

    setup = setup_soft_prompt_experiment(
        args, "V18: QUALITY-DIVERSITY ARCHIVE EVOLUTION",
        nested_difficulty=nested_difficulty,
    )

    decode_cfg = DecodeConfig(geometry="hyperbolic", **make_base_decode_kwargs(setup))

    qd_params = QDParams(
        bd_dim=16, rff_gamma=0.1, novelty_weight=0.3,
        novelty_k=5, archive_size=100, domination_threshold=0.15,
    )

    conditions = [
        ExperimentCondition(name="no_evolution", decode_cfg=decode_cfg, evolve=False),
        ExperimentCondition(name="standard_evolution", decode_cfg=decode_cfg, evolve=True),
        ExperimentCondition(
            name="qd_evolution", decode_cfg=decode_cfg,
            evolve=True, use_qd=True, qd_params=qd_params,
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
        bonferroni_n=1, depths=[2],
    )

    evo = EvolutionParams(
        generations=args.evo_gens, population_size=args.evo_pop,
        tasks_per_gen=args.evo_tasks, noise_scale=args.noise_scale,
        curvature=setup.curvature, fitness_mode="accuracy",
    )

    ball_radius = (1.0 / math.sqrt(setup.curvature)) * 0.95
    print(f"\nEvolution: {evo.generations} gens x {evo.population_size} pop", flush=True)
    print(f"QD: bd_dim={qd_params.bd_dim}, novelty_weight={qd_params.novelty_weight}", flush=True)
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
        "ball_radius": ball_radius,
        "qd_params": {
            "bd_dim": qd_params.bd_dim, "novelty_weight": qd_params.novelty_weight,
            "archive_size": qd_params.archive_size,
        },
        "calibration": setup.cal,
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
