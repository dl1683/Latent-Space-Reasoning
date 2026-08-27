"""
Novelty computation for Quality Diversity.

Novelty measures how different a solution is from others in the archive.
Higher novelty encourages exploration of underexplored regions of the
behavioral space.

Key Concepts:
- Novelty = mean distance to k nearest neighbors in BD space
- Higher novelty = more different from existing solutions
- QD fitness = (1-α)*fitness + α*novelty balances quality and diversity

Reference: "Novelty Search and the Problem with Objectives" (Lehman & Stanley, 2011)
"""

from __future__ import annotations

import torch
from torch import Tensor


class NoveltyComputer:
    """
    Computes novelty scores using k-nearest neighbor distances.

    Novelty(bd) = mean distance to k nearest neighbors in BD space

    Higher novelty indicates the solution is more different from existing
    solutions, which encourages exploration of underexplored regions.

    Args:
        k: Number of nearest neighbors for novelty computation
        distance_metric: Distance metric ("euclidean" or "cosine")

    Usage:
        >>> computer = NoveltyComputer(k=10)
        >>> novelty = computer.compute_novelty(bd, archive_bds)
    """

    def __init__(self, k: int = 10, distance_metric: str = "euclidean"):
        if k < 1:
            raise ValueError("k must be at least 1")

        self.k = k
        self.distance_metric = distance_metric

    def compute_novelty(
        self,
        bd: Tensor,
        archive_bds: list[Tensor],
        population_bds: list[Tensor] | None = None,
    ) -> float:
        """
        Compute novelty of a single BD.

        Novelty is the mean distance to the k nearest neighbors from
        the combined archive and current population.

        Args:
            bd: Behavioral descriptor to evaluate
            archive_bds: BDs currently in the archive
            population_bds: Optional current population BDs

        Returns:
            Novelty score (mean distance to k nearest neighbors)
        """
        # Combine archive and population for neighbor search
        all_bds = list(archive_bds)
        if population_bds:
            all_bds.extend(population_bds)

        if not all_bds:
            return 1.0  # Maximum novelty if no comparison points

        # Stack for efficient computation
        device = bd.device
        all_bds_tensor = torch.stack([b.to(device) for b in all_bds])

        # Compute distances
        distances = self._compute_distances(bd.unsqueeze(0), all_bds_tensor)

        # Get k nearest (excluding zero-distance if bd is in the set)
        k_actual = min(self.k, len(all_bds))
        top_k_distances, _ = torch.topk(distances, k_actual, largest=False)

        return top_k_distances.mean().item()

    def compute_novelty_batch(
        self,
        bds: list[Tensor],
        archive_bds: list[Tensor],
    ) -> list[float]:
        """
        Compute novelty for a batch of BDs efficiently.

        More efficient than calling compute_novelty repeatedly due to
        batched distance computation.

        Args:
            bds: List of behavioral descriptors to evaluate
            archive_bds: BDs currently in the archive

        Returns:
            List of novelty scores
        """
        if not bds:
            return []

        if not archive_bds:
            return [1.0] * len(bds)

        device = bds[0].device
        bds_tensor = torch.stack([b.to(device) for b in bds])
        archive_tensor = torch.stack([b.to(device) for b in archive_bds])

        # Pairwise distances (batch_size, archive_size)
        all_distances = self._compute_pairwise_distances(bds_tensor, archive_tensor)

        novelty_scores = []
        k_actual = min(self.k, len(archive_bds))

        for i in range(len(bds)):
            top_k, _ = torch.topk(all_distances[i], k_actual, largest=False)
            novelty_scores.append(top_k.mean().item())

        return novelty_scores

    def compute_local_competition(
        self,
        bd: Tensor,
        fitness: float,
        archive_bds: list[Tensor],
        archive_fitnesses: list[float],
        neighborhood_size: int | None = None,
    ) -> float:
        """
        Compute local competition score within BD neighborhood.

        This is useful for MAP-Elites style algorithms where fitness
        comparison is local to a neighborhood.

        Args:
            bd: Behavioral descriptor
            fitness: Fitness score of this solution
            archive_bds: BDs in the archive
            archive_fitnesses: Corresponding fitness scores
            neighborhood_size: Size of local neighborhood (defaults to k)

        Returns:
            Rank within local neighborhood (0 = best, 1 = worst)
        """
        if not archive_bds:
            return 0.0  # Best if no competition

        k = neighborhood_size or self.k
        device = bd.device

        archive_tensor = torch.stack([b.to(device) for b in archive_bds])
        distances = self._compute_distances(bd.unsqueeze(0), archive_tensor)

        # Get k nearest neighbors
        k_actual = min(k, len(archive_bds))
        _, neighbor_indices = torch.topk(distances, k_actual, largest=False)

        # Count how many neighbors have higher fitness
        neighbor_fitnesses = [archive_fitnesses[i] for i in neighbor_indices.tolist()]
        better_count = sum(1 for f in neighbor_fitnesses if f > fitness)

        return better_count / k_actual

    def _compute_distances(self, query: Tensor, targets: Tensor) -> Tensor:
        """Compute distances from query to all targets."""
        if self.distance_metric == "euclidean":
            return torch.cdist(query, targets).squeeze(0)
        elif self.distance_metric == "cosine":
            # Cosine distance = 1 - cosine_similarity
            query_norm = torch.nn.functional.normalize(query, dim=-1)
            targets_norm = torch.nn.functional.normalize(targets, dim=-1)
            similarity = torch.mm(query_norm, targets_norm.t())
            return (1 - similarity).squeeze(0)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _compute_pairwise_distances(
        self,
        queries: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Compute pairwise distances between queries and targets."""
        if self.distance_metric == "euclidean":
            return torch.cdist(queries, targets)
        elif self.distance_metric == "cosine":
            queries_norm = torch.nn.functional.normalize(queries, dim=-1)
            targets_norm = torch.nn.functional.normalize(targets, dim=-1)
            similarity = torch.mm(queries_norm, targets_norm.t())
            return 1 - similarity
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def __repr__(self) -> str:
        return f"NoveltyComputer(k={self.k}, metric={self.distance_metric})"


def combine_fitness_novelty(
    fitness: float,
    novelty: float,
    alpha: float = 0.3,
    fitness_weight: float = 1.0,
    novelty_weight: float = 1.0,
) -> float:
    """
    Combine fitness and novelty into QD fitness score.

    qd_fitness = (1 - alpha) * fitness * fitness_weight + alpha * novelty * novelty_weight

    This is the core QD scoring function that balances quality (fitness)
    with diversity (novelty).

    Args:
        fitness: Raw fitness score (higher = better quality)
        novelty: Novelty score (higher = more different from archive)
        alpha: Balance parameter (0 = pure fitness, 1 = pure novelty)
               Recommended: 0.2-0.4 for most tasks
        fitness_weight: Optional scaling for fitness component
        novelty_weight: Optional scaling for novelty component

    Returns:
        Combined QD fitness score

    Example:
        >>> qd_score = combine_fitness_novelty(0.8, 0.5, alpha=0.3)
        >>> # qd_score = 0.7 * 0.8 + 0.3 * 0.5 = 0.56 + 0.15 = 0.71
    """
    return (1 - alpha) * fitness * fitness_weight + alpha * novelty * novelty_weight


def normalize_novelty_scores(
    novelty_scores: list[float],
    method: str = "minmax",
) -> list[float]:
    """
    Normalize novelty scores to [0, 1] range.

    Useful when combining with fitness scores that are already normalized.

    Args:
        novelty_scores: Raw novelty scores
        method: Normalization method ("minmax" or "zscore")

    Returns:
        Normalized novelty scores
    """
    if not novelty_scores:
        return []

    if method == "minmax":
        min_val = min(novelty_scores)
        max_val = max(novelty_scores)
        if max_val - min_val < 1e-8:
            return [0.5] * len(novelty_scores)
        return [(n - min_val) / (max_val - min_val) for n in novelty_scores]

    elif method == "zscore":
        import statistics
        mean = statistics.mean(novelty_scores)
        std = statistics.stdev(novelty_scores) if len(novelty_scores) > 1 else 1.0
        if std < 1e-8:
            return [0.5] * len(novelty_scores)
        # Convert z-scores to [0, 1] using sigmoid-like mapping
        z_scores = [(n - mean) / std for n in novelty_scores]
        return [1 / (1 + 2.71828 ** (-z)) for z in z_scores]

    else:
        raise ValueError(f"Unknown normalization method: {method}")
