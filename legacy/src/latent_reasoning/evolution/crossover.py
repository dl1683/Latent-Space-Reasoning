"""Crossover strategies for evolutionary optimization."""

from __future__ import annotations

import random

import torch
from torch import Tensor

from latent_reasoning.evolution.operators import CrossoverStrategy


class MeanCrossover(CrossoverStrategy):
    """
    Mean crossover strategy.

    Simply averages the two parents.
    """

    def crossover(
        self,
        parent_a: Tensor,
        parent_b: Tensor,
        score_a: float = 0.0,
        score_b: float = 0.0,
    ) -> Tensor:
        return (parent_a + parent_b) / 2


class WeightedCrossover(CrossoverStrategy):
    """
    Weighted crossover strategy.

    Weights parents by their scores.
    """

    def crossover(
        self,
        parent_a: Tensor,
        parent_b: Tensor,
        score_a: float = 0.0,
        score_b: float = 0.0,
    ) -> Tensor:
        # Shift scores to be positive
        min_score = min(score_a, score_b)
        shifted_a = score_a - min_score + 0.1
        shifted_b = score_b - min_score + 0.1

        total = shifted_a + shifted_b
        weight_a = shifted_a / total
        weight_b = shifted_b / total

        return weight_a * parent_a + weight_b * parent_b


class InterpolationCrossover(CrossoverStrategy):
    """
    Interpolation crossover strategy.

    Picks a random point on the line between the two parents.
    """

    def crossover(
        self,
        parent_a: Tensor,
        parent_b: Tensor,
        score_a: float = 0.0,
        score_b: float = 0.0,
    ) -> Tensor:
        # Random interpolation factor
        alpha = random.random()

        return alpha * parent_a + (1 - alpha) * parent_b


class SliceCrossover(CrossoverStrategy):
    """
    Slice crossover strategy.

    Takes the first half of one parent and second half of the other.
    """

    def crossover(
        self,
        parent_a: Tensor,
        parent_b: Tensor,
        score_a: float = 0.0,
        score_b: float = 0.0,
    ) -> Tensor:
        # Find midpoint
        mid = parent_a.shape[-1] // 2

        # Combine slices
        child = torch.cat([parent_a[..., :mid], parent_b[..., mid:]], dim=-1)

        return child


class BlendCrossover(CrossoverStrategy):
    """
    Blend crossover strategy.

    Creates offspring by blending parent genes with some randomness.
    """

    def __init__(self, alpha: float = 0.5):
        """
        Initialize blend crossover.

        Args:
            alpha: Blend factor (higher = more exploration beyond parents)
        """
        self.alpha = alpha

    def crossover(
        self,
        parent_a: Tensor,
        parent_b: Tensor,
        score_a: float = 0.0,
        score_b: float = 0.0,
    ) -> Tensor:
        # For each dimension, sample uniformly from extended range
        diff = parent_b - parent_a
        min_val = torch.minimum(parent_a, parent_b) - self.alpha * torch.abs(diff)
        max_val = torch.maximum(parent_a, parent_b) + self.alpha * torch.abs(diff)

        # Sample uniformly
        child = min_val + torch.rand_like(parent_a) * (max_val - min_val)

        return child


class HyperbolicCrossover(CrossoverStrategy):
    """
    Hyperbolic crossover using Karcher mean (Fréchet barycenter).

    Combines parents using the hyperbolic barycenter, which respects
    the curved geometry of the Poincaré ball. This produces offspring
    that lie on the geodesic between parents.

    The hyperbolic mean naturally:
    - Stays on the manifold (no projection needed)
    - Weights by score (higher score = more influence)
    - Preserves hierarchical structure of reasoning branches
    """

    def __init__(
        self,
        curvature: float = 1.0,
        max_norm: float = 0.98,
        max_iterations: int = 7,
    ):
        """
        Initialize hyperbolic crossover.

        Args:
            curvature: Poincaré ball curvature
            max_norm: Maximum norm for stability
            max_iterations: Iterations for Karcher mean computation
        """
        self.curvature = curvature
        self.max_norm = max_norm
        self.max_iterations = max_iterations
        self._hyp = None

    def _get_hyperbolic(self):
        if self._hyp is None:
            from latent_reasoning.utils import hyperbolic as hyp
            self._hyp = hyp
        return self._hyp

    def crossover(
        self,
        parent_a: Tensor,
        parent_b: Tensor,
        score_a: float = 0.0,
        score_b: float = 0.0,
    ) -> Tensor:
        """
        Produce offspring via hyperbolic barycenter.

        Args:
            parent_a: First parent in Poincaré ball
            parent_b: Second parent in Poincaré ball
            score_a: Score of first parent
            score_b: Score of second parent

        Returns:
            Offspring point (Karcher mean weighted by scores)
        """
        hyp = self._get_hyperbolic()

        # Ensure parents are inside ball
        parent_a = hyp.project_to_ball(parent_a, self.curvature, self.max_norm)
        parent_b = hyp.project_to_ball(parent_b, self.curvature, self.max_norm)

        # Compute weights from scores (shifted to be positive)
        min_score = min(score_a, score_b)
        weight_a = score_a - min_score + 0.1
        weight_b = score_b - min_score + 0.1
        total = weight_a + weight_b
        weight_a /= total
        weight_b /= total

        # Stack parents and weights
        parents = torch.stack([parent_a.squeeze(), parent_b.squeeze()])
        weights = torch.tensor([weight_a, weight_b], device=parent_a.device, dtype=parent_a.dtype)

        # Compute Karcher mean
        child = hyp.karcher_mean(
            parents,
            weights=weights,
            c=self.curvature,
            max_iters=self.max_iterations,
        )

        return child

    def update_curvature(self, curvature: float) -> None:
        """Update curvature (for annealing)."""
        self.curvature = curvature


class HyperbolicInterpolationCrossover(CrossoverStrategy):
    """
    Hyperbolic geodesic interpolation crossover.

    Picks a random point on the geodesic between two parents.
    This is the hyperbolic analog of linear interpolation.
    """

    def __init__(
        self,
        curvature: float = 1.0,
        max_norm: float = 0.98,
    ):
        self.curvature = curvature
        self.max_norm = max_norm
        self._hyp = None

    def _get_hyperbolic(self):
        if self._hyp is None:
            from latent_reasoning.utils import hyperbolic as hyp
            self._hyp = hyp
        return self._hyp

    def crossover(
        self,
        parent_a: Tensor,
        parent_b: Tensor,
        score_a: float = 0.0,
        score_b: float = 0.0,
    ) -> Tensor:
        hyp = self._get_hyperbolic()

        # Random interpolation parameter biased by scores
        # Higher score = closer to that parent
        total_score = abs(score_a) + abs(score_b) + 0.1
        t_mean = abs(score_b) / total_score  # t=0 -> parent_a, t=1 -> parent_b

        # Add some randomness around the score-biased mean
        t = max(0.0, min(1.0, t_mean + (random.random() - 0.5) * 0.4))

        # Geodesic interpolation
        child = hyp.hyperbolic_interpolate(parent_a, parent_b, t, self.curvature)
        return hyp.project_to_ball(child, self.curvature, self.max_norm)

    def update_curvature(self, curvature: float) -> None:
        self.curvature = curvature


def get_crossover_strategy(name: str, **kwargs) -> CrossoverStrategy:
    """Factory function to get a crossover strategy by name."""
    strategies = {
        "mean": MeanCrossover,
        "weighted": WeightedCrossover,
        "interpolation": InterpolationCrossover,
        "slice": SliceCrossover,
        "blend": BlendCrossover,
        "hyperbolic": HyperbolicCrossover,
        "hyperbolic_interpolation": HyperbolicInterpolationCrossover,
    }

    if name not in strategies:
        raise ValueError(f"Unknown crossover strategy: {name}")

    return strategies[name](**kwargs)


def select_crossover_pairs(
    population: list[Tensor],
    scores: list[float],
    n_pairs: int,
    diversity_threshold: float = 0.3,
) -> list[tuple[int, int]]:
    """
    Select pairs of candidates for crossover.

    Prefers diverse pairs (not too similar) with good scores.

    Args:
        population: List of latent vectors
        scores: Corresponding scores
        n_pairs: Number of pairs to select
        diversity_threshold: Minimum cosine distance for pair selection

    Returns:
        List of (index_a, index_b) tuples
    """
    if len(population) < 2:
        return []

    pairs = []
    used = set()

    # Sort by score to prefer good candidates
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    for i in sorted_indices:
        if len(pairs) >= n_pairs:
            break

        for j in sorted_indices:
            if i >= j or (i, j) in used:
                continue

            # Check diversity
            cos_sim = torch.nn.functional.cosine_similarity(
                population[i].flatten().unsqueeze(0).float(),
                population[j].flatten().unsqueeze(0).float(),
            ).item()

            if cos_sim < (1 - diversity_threshold):
                pairs.append((i, j))
                used.add((i, j))
                used.add((j, i))
                break

    return pairs


def population_diversity(population: list[Tensor]) -> float:
    """
    Compute the diversity of a population.

    Returns:
        Float between 0 (all identical) and 1 (maximally diverse)
    """
    if len(population) < 2:
        return 0.0

    # Compute average pairwise cosine distance
    distances = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            cos_sim = torch.nn.functional.cosine_similarity(
                population[i].flatten().unsqueeze(0).float(),
                population[j].flatten().unsqueeze(0).float(),
            ).item()
            distances.append(1 - cos_sim)  # Convert similarity to distance

    return sum(distances) / len(distances)


def select_crossover_pairs_hyperbolic(
    population: list[Tensor],
    scores: list[float],
    n_pairs: int,
    curvature: float = 1.0,
    diversity_threshold: float = 0.5,
) -> list[tuple[int, int]]:
    """
    Select pairs for crossover using hyperbolic distance.

    Prefers pairs that are diverse (far apart in hyperbolic space)
    and have good scores.

    Args:
        population: List of latent vectors in Poincaré ball
        scores: Corresponding scores
        n_pairs: Number of pairs to select
        curvature: Hyperbolic curvature
        diversity_threshold: Minimum hyperbolic distance for pair selection

    Returns:
        List of (index_a, index_b) tuples
    """
    from latent_reasoning.utils import hyperbolic as hyp

    if len(population) < 2:
        return []

    pairs = []
    used = set()

    # Sort by score to prefer good candidates
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    for i in sorted_indices:
        if len(pairs) >= n_pairs:
            break

        for j in sorted_indices:
            if i >= j or (i, j) in used:
                continue

            # Compute hyperbolic distance
            h_dist = hyp.hyperbolic_distance(
                population[i].squeeze(),
                population[j].squeeze(),
                curvature,
            ).item()

            # Select if sufficiently diverse
            if h_dist > diversity_threshold:
                pairs.append((i, j))
                used.add((i, j))
                used.add((j, i))
                break

    return pairs


def population_diversity_hyperbolic(
    population: list[Tensor],
    curvature: float = 1.0,
) -> float:
    """
    Compute population diversity using hyperbolic distance.

    Returns:
        Average pairwise hyperbolic distance
    """
    from latent_reasoning.utils import hyperbolic as hyp

    if len(population) < 2:
        return 0.0

    distances = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            h_dist = hyp.hyperbolic_distance(
                population[i].squeeze(),
                population[j].squeeze(),
                curvature,
            ).item()
            distances.append(h_dist)

    return sum(distances) / len(distances) if distances else 0.0
