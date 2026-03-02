"""
Verifiable Evolution V8 - Fast Validation of c=0.5 Finding

V7 Key Finding: c=0.5 shows +70% improvement at depth 3!
This is a fast validation experiment to confirm across multiple seeds.

Design:
1. Focus on c=0.5 vs Euclidean (skip curvature sweep)
2. Reduced validation (30 tasks) for faster runs
3. 5 seeds for statistical confidence
4. Smaller model (1.7B) for 3x faster inference
"""

import argparse
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.config import GeometryConfig
from latent_reasoning.core.encoder import LLMEncoder


@dataclass
class FocusedTask:
    task_id: str
    prompt: str
    correct_answer: any
    verifier: callable
    depth: int
    difficulty: str


class FocusedTaskGenerator:
    def __init__(self, branching: int = 4, seed: int = 42):
        self.branching = branching
        self.rng = random.Random(seed)

    def generate_focused(self, tasks_per_depth: int = 100) -> list[FocusedTask]:
        tasks = []
        for depth in [2, 3]:
            for i in range(tasks_per_depth):
                task_id = f"d{depth}_t{i}"
                task = self._generate_task(depth, task_id)
                tasks.append(task)
        return tasks

    def _generate_task(self, depth: int, task_id: str) -> FocusedTask:
        path = [self.rng.randint(0, self.branching - 1) for _ in range(depth)]
        path_sum = sum(path)
        answer = path_sum * (depth + 1) + depth * 7

        prompt = (
            f"Calculate: sum([{','.join(map(str, path))}]) * {depth + 1} + {depth} * 7 = ?\n"
            f"Answer with just the number."
        )

        return FocusedTask(
            task_id=task_id,
            prompt=prompt,
            correct_answer=answer,
            verifier=self._verify_number,
            depth=depth,
            difficulty="medium",
        )

    def _verify_number(self, response: str, expected: int) -> bool:
        import re
        for num in re.findall(r'\d+', response):
            if int(num) == expected:
                return True
        return False


class FocusedTaskPool:
    def __init__(self, tasks_per_depth: int = 100, val_per_depth: int = 15, seed: int = 42):
        self.rng = random.Random(seed)
        gen = FocusedTaskGenerator(seed=seed)
        all_tasks = gen.generate_focused(tasks_per_depth)
        self.rng.shuffle(all_tasks)

        depth_2 = [t for t in all_tasks if t.depth == 2]
        depth_3 = [t for t in all_tasks if t.depth == 3]

        self.val_tasks = depth_2[:val_per_depth] + depth_3[:val_per_depth]
        self.train_tasks = depth_2[val_per_depth:] + depth_3[val_per_depth:]

    def sample_train(self, n: int, seed: int = None) -> list[FocusedTask]:
        rng = random.Random(seed) if seed else self.rng
        return rng.sample(self.train_tasks, min(n, len(self.train_tasks)))

    def get_validation(self) -> list[FocusedTask]:
        return self.val_tasks

    def stats(self) -> dict:
        return {
            'train_size': len(self.train_tasks),
            'val_size': len(self.val_tasks),
        }


@dataclass
class Candidate:
    latent: Tensor
    raw_fitness: float = 0.0
    correct: int = 0
    total: int = 0
    depth_correct: Dict[int, int] = field(default_factory=dict)
    depth_total: Dict[int, int] = field(default_factory=dict)
    task_results: Dict[str, bool] = field(default_factory=dict)


def evaluate_candidate(
    candidate: Candidate,
    tasks: list[FocusedTask],
    encoder: LLMEncoder,
    hyp_module,
    geometry_config: GeometryConfig,
) -> None:
    import sys
    candidate.correct = 0
    candidate.total = len(tasks)
    candidate.depth_correct = defaultdict(int)
    candidate.depth_total = defaultdict(int)
    candidate.task_results = {}

    for i, task in enumerate(tasks):
        sys.stdout.flush()
        candidate.depth_total[task.depth] += 1

        response = encoder.decode(
            candidate.latent,
            query=task.prompt,
            max_new_tokens=200,
            temperature=0.3,
            hyperbolic=hyp_module is not None,
            curvature=geometry_config.curvature if hyp_module else 1.0,
        )

        is_correct = task.verifier(response, task.correct_answer)
        candidate.task_results[task.task_id] = is_correct

        if is_correct:
            candidate.correct += 1
            candidate.depth_correct[task.depth] += 1

    candidate.raw_fitness = candidate.correct / candidate.total if candidate.total > 0 else 0.0


def mutate(latent, scale, hyp_module, geometry_config):
    noise = torch.randn_like(latent) * scale

    if hyp_module is not None:
        lat = latent.squeeze()
        norm = lat.norm()
        if norm > 0.95:
            lat = lat * (0.95 / norm)

        tangent = hyp_module.logmap0(lat, geometry_config.curvature)
        tangent = tangent + noise.squeeze()
        mutated = hyp_module.expmap0(tangent, geometry_config.curvature)
        mutated = hyp_module.project_to_ball(
            mutated, geometry_config.curvature, geometry_config.max_norm
        )
        return mutated.unsqueeze(0) if mutated.dim() == 1 else mutated
    else:
        return latent + noise


def crossover(latent_a, latent_b, hyp_module, geometry_config, t=0.5):
    if hyp_module is not None:
        parent_a = latent_a.squeeze()
        parent_b = latent_b.squeeze()

        norm_a, norm_b = parent_a.norm(), parent_b.norm()
        if norm_a > 0.95:
            parent_a = parent_a * (0.95 / norm_a)
        if norm_b > 0.95:
            parent_b = parent_b * (0.95 / norm_b)

        tan_a = hyp_module.logmap0(parent_a, geometry_config.curvature)
        tan_b = hyp_module.logmap0(parent_b, geometry_config.curvature)
        midpoint_tan = t * tan_a + (1 - t) * tan_b
        child = hyp_module.expmap0(midpoint_tan, geometry_config.curvature)
        child = hyp_module.project_to_ball(
            child, geometry_config.curvature, geometry_config.max_norm
        )
        return child.unsqueeze(0) if child.dim() == 1 else child
    else:
        return t * latent_a + (1 - t) * latent_b


def run_evolution(
    encoder: LLMEncoder,
    pool: FocusedTaskPool,
    geometry: str,
    seed_latent: Tensor,
    curvature: float = 0.5,  # Default to c=0.5 (our best finding)
    generations: int = 3,
    population_size: int = 2,
    tasks_per_gen: int = 4,
    mutation_scale: float = 0.1,
) -> Candidate:
    if geometry == "hyperbolic":
        from latent_reasoning.utils import hyperbolic as hyp
        geometry_config = GeometryConfig(
            space="hyperbolic",
            curvature=curvature,
            tangent_scale=0.35,
            max_norm=0.95,
        )
        seed_latent = hyp.expmap0(
            seed_latent.squeeze() * geometry_config.tangent_scale,
            curvature,
        ).unsqueeze(0)
        hyp_module = hyp
    else:
        geometry_config = GeometryConfig(space="euclidean")
        hyp_module = None

    population = [Candidate(latent=seed_latent.clone())]
    for _ in range(population_size - 1):
        mutated = mutate(seed_latent, mutation_scale, hyp_module, geometry_config)
        population.append(Candidate(latent=mutated))

    for gen in range(generations):
        tasks = pool.sample_train(tasks_per_gen, seed=gen * 1000 + gen)

        for cand in population:
            evaluate_candidate(cand, tasks, encoder, hyp_module, geometry_config)

        population.sort(key=lambda c: c.raw_fitness, reverse=True)
        elite = population[:max(1, population_size // 2)]

        new_pop = [Candidate(latent=e.latent.clone()) for e in elite]

        while len(new_pop) < population_size:
            p1 = elite[random.randint(0, len(elite) - 1)]
            p2 = elite[random.randint(0, len(elite) - 1)]
            child_latent = crossover(p1.latent, p2.latent, hyp_module, geometry_config)
            child_latent = mutate(child_latent, mutation_scale, hyp_module, geometry_config)
            new_pop.append(Candidate(latent=child_latent))

        population = new_pop

        best = max(population, key=lambda c: c.raw_fitness)
        print(f"[GEN {gen+1:02d}] best={best.raw_fitness:.3f}", flush=True)

    return max(population, key=lambda c: c.raw_fitness)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=2)
    parser.add_argument("--tasks-per-gen", type=int, default=4)
    parser.add_argument("--val-tasks", type=int, default=15)
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("VERIFIABLE EVOLUTION V8 - FAST c=0.5 VALIDATION", flush=True)
    print("Hypothesis: c=0.5 hyperbolic > Euclidean on depth 2-3", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Generations: {args.generations}", flush=True)
    print(f"Population: {args.population}", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    print(f"Validation tasks: {args.val_tasks * 2} ({args.val_tasks} per depth)", flush=True)
    print("=" * 70, flush=True)

    print("\nCreating task pool...", flush=True)
    pool = FocusedTaskPool(tasks_per_depth=100, val_per_depth=args.val_tasks, seed=42)
    print(f"Pool stats: {pool.stats()}", flush=True)

    print("\nLoading model...", flush=True)
    encoder = LLMEncoder(model_name=args.model, quantization="4bit")

    all_results = []

    for seed_idx in range(args.seeds):
        seed = 42 + seed_idx * 1000
        random.seed(seed)
        torch.manual_seed(seed)

        print(f"\n{'#' * 70}", flush=True)
        print(f"# SEED {seed_idx + 1}/{args.seeds} (seed={seed})", flush=True)
        print(f"{'#' * 70}", flush=True)

        system_prompt = "You calculate expressions step by step and give numeric answers."
        seed_latent = encoder.encode(system_prompt)

        # Euclidean baseline
        print("\n[EUCLIDEAN] Running evolution...", flush=True)
        euc_best = run_evolution(
            encoder, pool, "euclidean", seed_latent.clone(),
            generations=args.generations,
            population_size=args.population,
            tasks_per_gen=args.tasks_per_gen,
        )

        val_tasks = pool.get_validation()
        from latent_reasoning.config import GeometryConfig
        euc_config = GeometryConfig(space="euclidean")
        evaluate_candidate(euc_best, val_tasks, encoder, None, euc_config)
        euc_acc = euc_best.raw_fitness
        euc_d2 = euc_best.depth_correct.get(2, 0)
        euc_d3 = euc_best.depth_correct.get(3, 0)

        print(f"\n[EUCLIDEAN] Validation: {euc_acc * 100:.1f}%", flush=True)
        print(f"  Depth 2: {euc_d2}/{args.val_tasks}", flush=True)
        print(f"  Depth 3: {euc_d3}/{args.val_tasks}", flush=True)

        # Hyperbolic c=0.5
        print("\n[HYPERBOLIC c=0.5] Running evolution...", flush=True)
        hyp_best = run_evolution(
            encoder, pool, "hyperbolic", seed_latent.clone(),
            curvature=0.5,
            generations=args.generations,
            population_size=args.population,
            tasks_per_gen=args.tasks_per_gen,
        )

        from latent_reasoning.utils import hyperbolic as hyp
        hyp_config = GeometryConfig(
            space="hyperbolic",
            curvature=0.5,
            tangent_scale=0.35,
            max_norm=0.95,
        )
        evaluate_candidate(hyp_best, val_tasks, encoder, hyp, hyp_config)
        hyp_acc = hyp_best.raw_fitness
        hyp_d2 = hyp_best.depth_correct.get(2, 0)
        hyp_d3 = hyp_best.depth_correct.get(3, 0)

        print(f"\n[HYPERBOLIC c=0.5] Validation: {hyp_acc * 100:.1f}%", flush=True)
        print(f"  Depth 2: {hyp_d2}/{args.val_tasks}", flush=True)
        print(f"  Depth 3: {hyp_d3}/{args.val_tasks}", flush=True)

        margin = hyp_acc - euc_acc
        winner = "HYPERBOLIC" if margin > 0 else "EUCLIDEAN" if margin < 0 else "TIE"
        print(f"\n[SEED {seed_idx + 1}] Winner: {winner} (margin: {margin * 100:+.1f}%)", flush=True)

        all_results.append({
            'seed': seed,
            'euc_acc': euc_acc,
            'euc_d2': euc_d2,
            'euc_d3': euc_d3,
            'hyp_acc': hyp_acc,
            'hyp_d2': hyp_d2,
            'hyp_d3': hyp_d3,
            'margin': margin,
            'winner': winner,
        })

    # Final summary
    print(f"\n{'=' * 70}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)

    hyp_wins = sum(1 for r in all_results if r['winner'] == 'HYPERBOLIC')
    euc_wins = sum(1 for r in all_results if r['winner'] == 'EUCLIDEAN')
    ties = sum(1 for r in all_results if r['winner'] == 'TIE')

    avg_margin = sum(r['margin'] for r in all_results) / len(all_results)
    avg_euc = sum(r['euc_acc'] for r in all_results) / len(all_results)
    avg_hyp = sum(r['hyp_acc'] for r in all_results) / len(all_results)

    avg_euc_d3 = sum(r['euc_d3'] for r in all_results) / len(all_results)
    avg_hyp_d3 = sum(r['hyp_d3'] for r in all_results) / len(all_results)

    print(f"\nHyperbolic wins: {hyp_wins}/{args.seeds}", flush=True)
    print(f"Euclidean wins: {euc_wins}/{args.seeds}", flush=True)
    print(f"Ties: {ties}/{args.seeds}", flush=True)

    print(f"\nAverage accuracy:", flush=True)
    print(f"  Euclidean: {avg_euc * 100:.1f}%", flush=True)
    print(f"  Hyperbolic c=0.5: {avg_hyp * 100:.1f}%", flush=True)
    print(f"  Margin: {avg_margin * 100:+.1f}%", flush=True)

    print(f"\nDepth 3 average (per {args.val_tasks} tasks):", flush=True)
    print(f"  Euclidean: {avg_euc_d3:.1f}", flush=True)
    print(f"  Hyperbolic: {avg_hyp_d3:.1f}", flush=True)
    print(f"  Advantage: {(avg_hyp_d3 - avg_euc_d3) / args.val_tasks * 100:+.1f}%", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    overall = "HYPERBOLIC c=0.5 WINS" if hyp_wins > euc_wins else "EUCLIDEAN WINS" if euc_wins > hyp_wins else "TIE"
    print(f"VERDICT: {overall}", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
