"""
Verifiable Evolution V5 - Qwen3-4B + Calibrated Tasks

Per Codex analysis:
"Geometry helps representation of hierarchical structure, not raw algorithmic competence.
 You need to raise the model's baseline ability to get non-zero signal."

V5 Changes from V4:
1. Use Qwen3-4B (larger model for better deep task performance)
2. Calibrate depth to 20-60% accuracy band (max depth 5, focus on 3-5)
3. Add explicit algorithmic scaffolding in prompt (few-shot example)
4. Adjust tail metric to use model's frontier depth (3-5 instead of 5-8)
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.config import GeometryConfig
from latent_reasoning.core.encoder import LLMEncoder


@dataclass
class CalibratedTreeTask:
    """Tree task calibrated to model capability."""
    prompt: str
    correct_answer: any
    verifier: callable
    depth: int
    category: str
    difficulty: str


class CalibratedTreeTaskGenerator:
    """
    Generate tree tasks calibrated for 20-60% accuracy.

    Focus on depths 2-5 (instead of 5-8) to get usable signal.
    Add few-shot example in prompt for algorithmic scaffolding.
    """

    FEW_SHOT_EXAMPLE = """Example:
Q: Starting at root (value=0), follow: child 1 -> child 2. Value = sum(path) * (depth+1) + len(path) * 7.
A: Path = [1, 2], depth = 2. sum([1,2]) = 3. 3 * (2+1) + 2*7 = 3*3 + 14 = 9 + 14 = 23.

Now solve:
"""

    def __init__(self, max_depth: int = 5, branching: int = 4, seed: int = 42):
        self.max_depth = max_depth
        self.branching = branching
        self.rng = random.Random(seed)

    def generate(self, n: int = 10) -> list[CalibratedTreeTask]:
        """Generate n calibrated tasks."""
        tasks = []

        for _ in range(n):
            # Bias toward moderate depths (3-4) where signal is usable
            depth_choice = self.rng.random()
            if depth_choice < 0.2:
                depth = self.rng.randint(1, 2)  # Easy
            elif depth_choice < 0.6:
                depth = self.rng.randint(3, 4)  # Target range
            else:
                depth = min(self.rng.randint(4, 5), self.max_depth)  # Harder

            task = self._generate_task(depth)
            tasks.append(task)

        return tasks

    def _generate_task(self, depth: int) -> CalibratedTreeTask:
        """Generate a single task."""
        path = [self.rng.randint(0, self.branching - 1) for _ in range(depth)]

        # Compute answer: sum(path) * (depth+1) + len(path) * 7
        answer = sum(path) * (depth + 1) + len(path) * 7

        # Build prompt with few-shot example
        path_str = " -> ".join([f"child {p}" for p in path])
        prompt = (
            f"{self.FEW_SHOT_EXAMPLE}"
            f"Q: Starting at root (value=0), follow: {path_str}. "
            f"Value = sum(path) * (depth+1) + len(path) * 7. What is the value? "
            f"Just give the number."
        )

        if depth <= 2:
            difficulty = "easy"
        elif depth <= 4:
            difficulty = "medium"
        else:
            difficulty = "hard"

        return CalibratedTreeTask(
            prompt=prompt,
            correct_answer=answer,
            verifier=self._verify_number,
            depth=depth,
            category="tree_traversal",
            difficulty=difficulty,
        )

    def _verify_number(self, response: str, expected: int) -> bool:
        """Verify numeric response."""
        import re
        numbers = re.findall(r'-?\d+', response)
        if not numbers:
            return False
        try:
            return int(numbers[-1]) == expected
        except (ValueError, IndexError):
            return False


class CalibratedTaskPool:
    """Task pool calibrated for 20-60% accuracy."""

    def __init__(self, pool_size: int = 100, val_ratio: float = 0.2, seed: int = 42):
        random.seed(seed)

        gen = CalibratedTreeTaskGenerator(max_depth=5, seed=seed)
        all_tasks = gen.generate(pool_size)

        random.shuffle(all_tasks)

        val_size = int(len(all_tasks) * val_ratio)
        self.val_tasks = all_tasks[:val_size]
        self.train_tasks = all_tasks[val_size:]

    def sample_train(self, n: int, seed: int | None = None) -> list[CalibratedTreeTask]:
        if seed is not None:
            random.seed(seed)
        return random.sample(self.train_tasks, min(n, len(self.train_tasks)))

    def get_validation(self) -> list[CalibratedTreeTask]:
        return self.val_tasks

    def stats(self) -> dict:
        train_depths = defaultdict(int)
        train_diff = defaultdict(int)
        for t in self.train_tasks:
            train_depths[t.depth] += 1
            train_diff[t.difficulty] += 1

        return {
            "train_size": len(self.train_tasks),
            "val_size": len(self.val_tasks),
            "train_depths": dict(train_depths),
            "train_difficulty": dict(train_diff),
        }


@dataclass
class TreeCandidate:
    """Candidate with depth-weighted fitness."""
    latent: Tensor
    depth_correct: dict = field(default_factory=dict)
    depth_total: dict = field(default_factory=dict)
    correct: int = 0
    total: int = 0

    @property
    def raw_fitness(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def weighted_fitness(self) -> float:
        """Depth-weighted fitness with 2^depth scaling."""
        weighted_sum = 0.0
        weight_total = 0.0
        for depth, total in self.depth_total.items():
            weight = 2 ** depth
            correct = self.depth_correct.get(depth, 0)
            weighted_sum += correct * weight
            weight_total += total * weight
        return weighted_sum / weight_total if weight_total > 0 else 0.0

    def get_tail_metric(self) -> float:
        """Tail metric: accuracy on depths 3-5 (calibrated frontier)."""
        deep_correct = sum(self.depth_correct.get(d, 0) for d in range(3, 6))
        deep_total = sum(self.depth_total.get(d, 0) for d in range(3, 6))
        return deep_correct / deep_total if deep_total > 0 else 0.0


def evaluate_candidate(
    candidate: TreeCandidate,
    tasks: list[CalibratedTreeTask],
    encoder: LLMEncoder,
    hyp_module,
    geometry_config: GeometryConfig,
) -> None:
    """Evaluate candidate."""
    candidate.correct = 0
    candidate.total = len(tasks)
    candidate.depth_correct = defaultdict(int)
    candidate.depth_total = defaultdict(int)

    for task in tasks:
        candidate.depth_total[task.depth] += 1

        response = encoder.decode(
            candidate.latent,
            query=task.prompt,
            max_new_tokens=200,  # Longer for chain-of-thought
            temperature=0.3,
            hyperbolic=hyp_module is not None,
            curvature=geometry_config.curvature if hyp_module else 1.0,
        )

        if task.verifier(response, task.correct_answer):
            candidate.correct += 1
            candidate.depth_correct[task.depth] += 1


def compute_diversity(population, hyp_module, geometry_config) -> float:
    """Compute diversity with numerical safeguards."""
    n = len(population)
    if n <= 1:
        return 0.0

    total_dist = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            try:
                lat_i = population[i].latent.squeeze()
                lat_j = population[j].latent.squeeze()

                if hyp_module is not None:
                    norm_i, norm_j = lat_i.norm(), lat_j.norm()
                    if norm_i > 0.99 or norm_j > 0.99:
                        dist = torch.norm(lat_i - lat_j).item()
                    else:
                        dist = hyp_module.hyperbolic_distance(
                            lat_i, lat_j, geometry_config.curvature
                        ).item()
                else:
                    dist = torch.norm(lat_i - lat_j).item()

                if not (math.isnan(dist) or math.isinf(dist)):
                    total_dist += dist
                    count += 1
            except Exception:
                continue

    return total_dist / count if count > 0 else 0.0


def mutate(latent, scale, hyp_module, geometry_config):
    """Mutate with safeguards."""
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

        if torch.isnan(mutated).any() or torch.isinf(mutated).any():
            mutated = latent.squeeze() + noise.squeeze() * 0.1
            mutated = hyp_module.project_to_ball(
                mutated, geometry_config.curvature, geometry_config.max_norm
            )

        return mutated.unsqueeze(0) if mutated.dim() == 1 else mutated
    else:
        return latent + noise


def crossover(parent_a, parent_b, hyp_module, geometry_config):
    """Crossover with safeguards."""
    t = random.random()

    if hyp_module is not None:
        a = parent_a.squeeze()
        b = parent_b.squeeze()
        norm_a, norm_b = a.norm(), b.norm()
        if norm_a > 0.95:
            a = a * (0.95 / norm_a)
        if norm_b > 0.95:
            b = b * (0.95 / norm_b)

        try:
            child = hyp_module.hyperbolic_interpolate(a, b, t, geometry_config.curvature)
            if torch.isnan(child).any() or torch.isinf(child).any():
                child = t * a + (1 - t) * b
                child = hyp_module.project_to_ball(
                    child, geometry_config.curvature, geometry_config.max_norm
                )
        except Exception:
            child = t * a + (1 - t) * b
            child = hyp_module.project_to_ball(
                child, geometry_config.curvature, geometry_config.max_norm
            )

        return child.unsqueeze(0) if child.dim() == 1 else child
    else:
        return t * parent_a + (1 - t) * parent_b


def tournament_select(population, k: int = 3):
    """Tournament selection using weighted fitness."""
    contestants = random.sample(population, min(k, len(population)))
    return max(contestants, key=lambda c: c.weighted_fitness)


def run_evolution(
    encoder: LLMEncoder,
    pool: CalibratedTaskPool,
    geometry: str,
    seed_latent: Tensor,
    generations: int = 6,
    population_size: int = 6,
    tasks_per_gen: int = 10,
    elite_count: int = 2,
    mutation_scale: float = 0.1,
) -> dict:
    """Run evolution with calibrated tasks."""

    if geometry == "hyperbolic":
        from latent_reasoning.utils import hyperbolic as hyp
        geometry_config = GeometryConfig(
            space="hyperbolic",
            curvature=1.0,
            tangent_scale=0.35,
            max_norm=0.95,
        )
        seed_latent = hyp.expmap0(
            seed_latent.squeeze() * geometry_config.tangent_scale,
            geometry_config.curvature,
        ).unsqueeze(0)
        hyp_module = hyp
    else:
        geometry_config = GeometryConfig(space="euclidean")
        hyp_module = None

    # Initialize population
    population = [TreeCandidate(latent=seed_latent.clone())]
    for _ in range(population_size - 1):
        mutated = mutate(seed_latent, mutation_scale, hyp_module, geometry_config)
        population.append(TreeCandidate(latent=mutated))

    history = []
    rolling_fitness = []

    for gen in range(generations):
        tasks = pool.sample_train(tasks_per_gen, seed=gen * 1000 + gen)

        for candidate in population:
            evaluate_candidate(candidate, tasks, encoder, hyp_module, geometry_config)

        population.sort(key=lambda c: c.weighted_fitness, reverse=True)

        rolling_fitness.append(population[0].raw_fitness)
        if len(rolling_fitness) > 3:
            rolling_fitness.pop(0)
        roll_avg = sum(rolling_fitness) / len(rolling_fitness)

        diversity = compute_diversity(population, hyp_module, geometry_config)
        avg_fitness = sum(c.raw_fitness for c in population) / len(population)
        tail_metric = population[0].get_tail_metric()

        history.append({
            "generation": gen + 1,
            "raw_fitness": population[0].raw_fitness,
            "weighted_fitness": population[0].weighted_fitness,
            "roll_avg": roll_avg,
            "diversity": diversity,
            "tail_metric": tail_metric,
        })

        print(
            f"[GEN {gen+1:02d}] raw={population[0].raw_fitness:.3f} "
            f"wgt={population[0].weighted_fitness:.3f} "
            f"roll={roll_avg:.3f} "
            f"tail={tail_metric:.3f} "
            f"div={diversity:.3f}",
            flush=True
        )

        # Create next generation
        next_gen = []
        for elite in population[:elite_count]:
            next_gen.append(TreeCandidate(latent=elite.latent.clone()))

        while len(next_gen) < population_size:
            parent_a = tournament_select(population)
            parent_b = tournament_select(population)
            child_latent = crossover(parent_a.latent, parent_b.latent, hyp_module, geometry_config)
            if random.random() < 0.8:
                child_latent = mutate(child_latent, mutation_scale, hyp_module, geometry_config)
            next_gen.append(TreeCandidate(latent=child_latent))

        population = next_gen

    return {
        "final_raw": population[0].raw_fitness,
        "final_weighted": population[0].weighted_fitness,
        "final_tail": population[0].get_tail_metric(),
        "history": history,
        "best_latent": population[0].latent,
    }


def evaluate_on_validation(latent, val_tasks, encoder, geometry):
    """Evaluate on validation set."""
    if geometry == "hyperbolic":
        from latent_reasoning.utils import hyperbolic as hyp
        geometry_config = GeometryConfig(
            space="hyperbolic", curvature=1.0, tangent_scale=0.35, max_norm=0.95
        )
        hyp_module = hyp
    else:
        geometry_config = GeometryConfig(space="euclidean")
        hyp_module = None

    candidate = TreeCandidate(latent=latent)
    evaluate_candidate(candidate, val_tasks, encoder, hyp_module, geometry_config)

    return {
        "raw_accuracy": candidate.raw_fitness,
        "weighted_accuracy": candidate.weighted_fitness,
        "tail_metric": candidate.get_tail_metric(),
        "correct": candidate.correct,
        "total": candidate.total,
        "depth_breakdown": {
            d: f"{candidate.depth_correct.get(d, 0)}/{candidate.depth_total.get(d, 0)}"
            for d in sorted(candidate.depth_total.keys())
        },
    }


def main():
    parser = argparse.ArgumentParser(description="V5: Qwen3-4B + Calibrated Tasks")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--tasks-per-gen", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("VERIFIABLE EVOLUTION V5 - QWEN3-4B + CALIBRATED TASKS", flush=True)
    print("Larger model | Few-shot prompt | Calibrated depth (2-5) | Tail at 3-5", flush=True)
    print("=" * 70, flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Generations: {args.generations}", flush=True)
    print(f"Population: {args.population}", flush=True)
    print(f"Tasks/gen: {args.tasks_per_gen}", flush=True)
    print(f"Pool size: {args.pool_size}", flush=True)
    print(f"Runs: {args.runs}", flush=True)
    print("=" * 70, flush=True)

    print("\nCreating calibrated task pool...", flush=True)
    pool = CalibratedTaskPool(pool_size=args.pool_size, seed=args.seed)
    print(f"Pool stats: {pool.stats()}", flush=True)

    print("\nLoading model...", flush=True)
    encoder = LLMEncoder(model_name=args.model, quantization="4bit")

    prompts = [
        "You follow tree paths step by step, computing values at each node.",
        "You solve hierarchical traversal problems by tracking depth and path sums.",
    ]

    results = []

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx * 1000
        random.seed(run_seed)
        torch.manual_seed(run_seed)

        print(f"\n{'#' * 70}", flush=True)
        print(f"# RUN {run_idx + 1}/{args.runs} (seed={run_seed})", flush=True)
        print("#" * 70, flush=True)

        for prompt_idx, prompt in enumerate(prompts):
            print(f"\nPrompt: {prompt[:60]}...", flush=True)

            seed_latent = encoder.encode(prompt)

            print(f"\n[HYPERBOLIC] Running evolution...", flush=True)
            hyp_result = run_evolution(
                encoder=encoder,
                pool=pool,
                geometry="hyperbolic",
                seed_latent=seed_latent.clone(),
                generations=args.generations,
                population_size=args.population,
                tasks_per_gen=args.tasks_per_gen,
            )

            random.seed(run_seed)
            torch.manual_seed(run_seed)

            print(f"\n[EUCLIDEAN] Running evolution...", flush=True)
            euc_result = run_evolution(
                encoder=encoder,
                pool=pool,
                geometry="euclidean",
                seed_latent=seed_latent.clone(),
                generations=args.generations,
                population_size=args.population,
                tasks_per_gen=args.tasks_per_gen,
            )

            print(f"\n[VALIDATION] Evaluating on held-out set...", flush=True)
            val_tasks = pool.get_validation()

            hyp_val = evaluate_on_validation(hyp_result["best_latent"], val_tasks, encoder, "hyperbolic")
            euc_val = evaluate_on_validation(euc_result["best_latent"], val_tasks, encoder, "euclidean")

            print(f"  Hyperbolic: raw={hyp_val['raw_accuracy']*100:.1f}% wgt={hyp_val['weighted_accuracy']:.3f} tail={hyp_val['tail_metric']:.3f}", flush=True)
            print(f"  Euclidean:  raw={euc_val['raw_accuracy']*100:.1f}% wgt={euc_val['weighted_accuracy']:.3f} tail={euc_val['tail_metric']:.3f}", flush=True)
            print(f"  Hyp depths: {hyp_val['depth_breakdown']}", flush=True)
            print(f"  Euc depths: {euc_val['depth_breakdown']}", flush=True)

            raw_margin = (hyp_val['raw_accuracy'] - euc_val['raw_accuracy']) * 100
            wgt_margin = (hyp_val['weighted_accuracy'] - euc_val['weighted_accuracy']) * 100
            tail_margin = (hyp_val['tail_metric'] - euc_val['tail_metric']) * 100

            if wgt_margin > 5:
                winner = "HYPERBOLIC"
            elif wgt_margin < -5:
                winner = "EUCLIDEAN"
            else:
                winner = "TIE"

            print(f"\n[RESULT] {winner}", flush=True)
            print(f"  Raw margin: {raw_margin:+.1f}%", flush=True)
            print(f"  Weighted margin: {wgt_margin:+.1f}%", flush=True)
            print(f"  Tail margin: {tail_margin:+.1f}%", flush=True)

            results.append({
                "run": run_idx + 1,
                "prompt_idx": prompt_idx,
                "hyperbolic": hyp_val,
                "euclidean": euc_val,
                "winner": winner,
                "raw_margin": raw_margin,
                "wgt_margin": wgt_margin,
                "tail_margin": tail_margin,
            })

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY - V5 CALIBRATED TASKS (QWEN3-4B)", flush=True)
    print("=" * 70, flush=True)

    hyp_wins = sum(1 for r in results if r['winner'] == "HYPERBOLIC")
    euc_wins = sum(1 for r in results if r['winner'] == "EUCLIDEAN")
    ties = sum(1 for r in results if r['winner'] == "TIE")

    avg_raw_margin = sum(r['raw_margin'] for r in results) / len(results)
    avg_wgt_margin = sum(r['wgt_margin'] for r in results) / len(results)
    avg_tail_margin = sum(r['tail_margin'] for r in results) / len(results)

    print(f"Hyperbolic wins: {hyp_wins}", flush=True)
    print(f"Euclidean wins: {euc_wins}", flush=True)
    print(f"Ties: {ties}", flush=True)
    print(f"Average raw margin: {avg_raw_margin:+.1f}%", flush=True)
    print(f"Average weighted margin: {avg_wgt_margin:+.1f}%", flush=True)
    print(f"Average tail margin: {avg_tail_margin:+.1f}%", flush=True)

    output_path = Path(__file__).parent / "v5_calibrated_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
