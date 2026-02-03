"""
Verifiable Evolution Loop

This replaces the broken latent scorer with GROUND TRUTH fitness.
Evolution is driven by actual correctness on verifiable tasks.

Key insight from Codex:
"The breakthrough isn't better scoring - it's making truth observable."

Fitness = number of correct answers on verifiable tasks
Selection = keep candidates that solve the most tasks
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor

from latent_reasoning.config import Config, GeometryConfig
from latent_reasoning.core.encoder import LLMEncoder
from latent_reasoning.verification.verifiable_tasks import (
    VerifiableTask,
    VerifiableTaskSuite,
    create_task_suite,
)


@dataclass
class VerifiableCandidate:
    """A candidate with its fitness based on verifiable tasks."""
    latent: Tensor
    correct: int = 0
    total: int = 0
    responses: List[str] = field(default_factory=list)

    @property
    def fitness(self) -> float:
        """Fitness is accuracy on verifiable tasks."""
        if self.total == 0:
            return 0.0
        return self.correct / self.total


@dataclass
class VerifiableEvolutionResult:
    """Result of verifiable evolution."""
    best_latent: Tensor
    best_fitness: float
    best_correct: int
    best_total: int
    generations: int
    final_population: List[VerifiableCandidate]
    history: List[dict] = field(default_factory=list)


class VerifiableEvolutionLoop:
    """
    Evolution loop with ground-truth fitness.

    Instead of using a broken latent scorer, we:
    1. Generate diverse candidates using hyperbolic geometry
    2. Evaluate each candidate on verifiable tasks
    3. Select survivors based on actual correctness
    4. Repeat until convergence or max generations
    """

    def __init__(
        self,
        encoder: LLMEncoder,
        geometry_config: GeometryConfig,
        population_size: int = 10,
        tasks_per_evaluation: int = 10,
        mutation_scale: float = 0.1,
        elite_count: int = 2,
        seed: int | None = None,
    ):
        self.encoder = encoder
        self.geometry_config = geometry_config
        self.population_size = population_size
        self.tasks_per_evaluation = tasks_per_evaluation
        self.mutation_scale = mutation_scale
        self.elite_count = elite_count

        # Task suite for generating verifiable tasks
        self.task_suite = create_task_suite(seed=seed)

        # Hyperbolic utilities if using hyperbolic space
        self._hyp = None
        if geometry_config.space == "hyperbolic":
            from latent_reasoning.utils import hyperbolic as hyp
            self._hyp = hyp

    def run(
        self,
        seed_latent: Tensor,
        max_generations: int = 50,
        target_fitness: float = 0.9,
        patience: int = 10,
    ) -> VerifiableEvolutionResult:
        """
        Run verifiable evolution.

        Args:
            seed_latent: Initial latent to evolve from
            max_generations: Maximum generations to run
            target_fitness: Stop if this fitness is reached
            patience: Stop if no improvement for this many generations

        Returns:
            VerifiableEvolutionResult with best candidate
        """
        # Initialize population from seed
        population = self._initialize_population(seed_latent)

        best_fitness = 0.0
        best_candidate = population[0]
        no_improvement = 0
        history = []

        for gen in range(max_generations):
            # Generate tasks for this generation
            tasks = self.task_suite.generate_batch(self.tasks_per_evaluation)

            # Evaluate all candidates
            for candidate in population:
                self._evaluate_candidate(candidate, tasks)

            # Sort by fitness (descending)
            population.sort(key=lambda c: c.fitness, reverse=True)

            # Track best
            if population[0].fitness > best_fitness:
                best_fitness = population[0].fitness
                best_candidate = population[0]
                no_improvement = 0
            else:
                no_improvement += 1

            # Log progress
            avg_fitness = sum(c.fitness for c in population) / len(population)
            history.append({
                "generation": gen,
                "best_fitness": population[0].fitness,
                "avg_fitness": avg_fitness,
                "best_correct": population[0].correct,
                "best_total": population[0].total,
            })

            print(
                f"[GEN {gen+1:02d}] best={population[0].fitness:.3f} "
                f"({population[0].correct}/{population[0].total}) "
                f"avg={avg_fitness:.3f} survivors={len(population)}"
            )

            # Check stopping conditions
            if best_fitness >= target_fitness:
                print(f"Target fitness {target_fitness} reached!")
                break

            if no_improvement >= patience:
                print(f"No improvement for {patience} generations, stopping.")
                break

            # Create next generation
            population = self._create_next_generation(population)

        return VerifiableEvolutionResult(
            best_latent=best_candidate.latent,
            best_fitness=best_candidate.fitness,
            best_correct=best_candidate.correct,
            best_total=best_candidate.total,
            generations=gen + 1,
            final_population=population,
            history=history,
        )

    def _initialize_population(self, seed_latent: Tensor) -> List[VerifiableCandidate]:
        """Initialize population with mutations of seed."""
        population = [VerifiableCandidate(latent=seed_latent.clone())]

        for _ in range(self.population_size - 1):
            mutated = self._mutate(seed_latent)
            population.append(VerifiableCandidate(latent=mutated))

        return population

    def _mutate(self, latent: Tensor) -> Tensor:
        """Mutate a latent vector."""
        noise = torch.randn_like(latent) * self.mutation_scale

        if self._hyp is not None:
            # Hyperbolic mutation: add noise in tangent space, map back
            tangent = self._hyp.logmap0(latent.squeeze(), self.geometry_config.curvature)
            tangent = tangent + noise.squeeze()
            mutated = self._hyp.expmap0(tangent, self.geometry_config.curvature)
            mutated = self._hyp.project_to_ball(
                mutated, self.geometry_config.curvature, self.geometry_config.max_norm
            )
            return mutated.unsqueeze(0) if mutated.dim() == 1 else mutated
        else:
            # Euclidean mutation: simple addition
            return latent + noise

    def _crossover(self, parent_a: Tensor, parent_b: Tensor) -> Tensor:
        """Crossover two parents."""
        t = random.random()

        if self._hyp is not None:
            # Hyperbolic interpolation
            child = self._hyp.hyperbolic_interpolate(
                parent_a.squeeze(),
                parent_b.squeeze(),
                t,
                self.geometry_config.curvature,
            )
            return child.unsqueeze(0) if child.dim() == 1 else child
        else:
            # Euclidean interpolation
            return t * parent_a + (1 - t) * parent_b

    def _evaluate_candidate(self, candidate: VerifiableCandidate, tasks: List[VerifiableTask]) -> None:
        """Evaluate candidate on verifiable tasks."""
        candidate.responses = []
        candidate.correct = 0
        candidate.total = len(tasks)

        for task in tasks:
            # Decode and generate response
            response = self.encoder.decode(
                candidate.latent,
                query=task.prompt,
                max_new_tokens=100,  # Short responses for verification tasks
                temperature=0.3,  # Low temperature for more deterministic answers
                hyperbolic=self._hyp is not None,
                curvature=self.geometry_config.curvature if self._hyp else 1.0,
            )

            candidate.responses.append(response)

            # Verify correctness
            if self.task_suite.evaluate_response(task, response):
                candidate.correct += 1

    def _create_next_generation(self, population: List[VerifiableCandidate]) -> List[VerifiableCandidate]:
        """Create next generation through selection, crossover, mutation."""
        next_gen = []

        # Keep elites
        elites = population[:self.elite_count]
        for elite in elites:
            next_gen.append(VerifiableCandidate(latent=elite.latent.clone()))

        # Fill rest with offspring
        while len(next_gen) < self.population_size:
            # Tournament selection
            parent_a = self._tournament_select(population)
            parent_b = self._tournament_select(population)

            # Crossover
            child_latent = self._crossover(parent_a.latent, parent_b.latent)

            # Mutation
            if random.random() < 0.8:  # 80% mutation rate
                child_latent = self._mutate(child_latent)

            next_gen.append(VerifiableCandidate(latent=child_latent))

        return next_gen

    def _tournament_select(self, population: List[VerifiableCandidate], k: int = 3) -> VerifiableCandidate:
        """Tournament selection."""
        contestants = random.sample(population, min(k, len(population)))
        return max(contestants, key=lambda c: c.fitness)


def run_verifiable_evolution(
    prompt: str,
    encoder: LLMEncoder,
    geometry: str = "hyperbolic",
    max_generations: int = 50,
    population_size: int = 10,
    tasks_per_evaluation: int = 10,
) -> VerifiableEvolutionResult:
    """
    Run verifiable evolution on a prompt.

    This is the main entry point for ground-truth evolution.
    """
    # Create geometry config
    if geometry == "hyperbolic":
        geometry_config = GeometryConfig(
            space="hyperbolic",
            curvature=1.0,
            tangent_scale=0.35,
            max_norm=0.98,
        )
    else:
        geometry_config = GeometryConfig(space="euclidean")

    # Encode seed
    seed_latent = encoder.encode(prompt)

    # Map to hyperbolic if needed
    if geometry == "hyperbolic":
        from latent_reasoning.utils import hyperbolic as hyp
        seed_latent = hyp.expmap0(
            seed_latent.squeeze() * geometry_config.tangent_scale,
            geometry_config.curvature,
        )
        seed_latent = seed_latent.unsqueeze(0)

    # Create and run evolution loop
    loop = VerifiableEvolutionLoop(
        encoder=encoder,
        geometry_config=geometry_config,
        population_size=population_size,
        tasks_per_evaluation=tasks_per_evaluation,
    )

    return loop.run(seed_latent, max_generations=max_generations)
