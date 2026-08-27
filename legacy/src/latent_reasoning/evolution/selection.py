"""
Selection strategies for evolutionary optimization in latent space.

This module implements various selection strategies that determine which candidates
survive from one generation to the next. Selection is crucial for balancing
exploitation (keeping the best solutions) with exploration (maintaining diversity).

Available strategies:
- ElitistSelection: Always keeps top performers, adds randomness for diversity
- TournamentSelection: Runs competitions between random subsets
- RouletteSelection: Probability proportional to fitness scores

Each strategy has different characteristics in terms of selection pressure,
diversity maintenance, and computational efficiency.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Set, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from latent_reasoning.evolution.operators import SelectionStrategy


@dataclass
class SelectionResult:
    """Result of selection with diversity tracking."""
    latents: List[Tensor]
    scores: List[float]
    diversity_indices: Set[int]  # Indices of chains selected for diversity (protected from merge)


def _compute_diversity_distance(
    candidate: Tensor,
    selected: List[Tensor],
    use_hyperbolic: bool = False,
    curvature: float = 1.0,
) -> float:
    """
    Compute minimum distance from candidate to all selected latents.

    Args:
        candidate: Candidate latent vector
        selected: List of already selected latents
        use_hyperbolic: If True, use hyperbolic distance in Poincaré ball
        curvature: Hyperbolic curvature (only used if use_hyperbolic=True)

    Returns:
        Minimum distance to selected latents
    """
    if not selected:
        return float('inf')

    if use_hyperbolic:
        # Use hyperbolic distance
        from latent_reasoning.utils import hyperbolic as hyp
        distances = []
        for s in selected:
            d = hyp.hyperbolic_distance(candidate.squeeze(), s.squeeze(), curvature).item()
            distances.append(d)
        return min(distances)
    else:
        # Use Euclidean cosine distance
        candidate_flat = candidate.flatten().unsqueeze(0)
        selected_stack = torch.stack([s.flatten() for s in selected])

        # Cosine similarity
        similarities = F.cosine_similarity(candidate_flat, selected_stack)

        # Return minimum distance (1 - max similarity)
        return (1 - similarities.max()).item()


def _select_for_diversity(
    candidates: List[Tensor],
    candidate_indices: List[int],
    already_selected: List[Tensor],
    n_select: int,
    use_hyperbolic: bool = False,
    curvature: float = 1.0,
) -> List[int]:
    """
    Select candidates that maximize diversity from already selected.

    Args:
        candidates: List of candidate latent vectors
        candidate_indices: Original indices of candidates
        already_selected: Already selected latents
        n_select: Number to select for diversity
        use_hyperbolic: If True, use hyperbolic distance
        curvature: Hyperbolic curvature

    Returns:
        Indices of selected candidates
    """
    if not candidates or n_select <= 0:
        return []

    selected_indices = []
    current_selected = list(already_selected)
    available = list(zip(candidates, candidate_indices))

    for _ in range(min(n_select, len(available))):
        if not available:
            break

        # Find candidate with maximum distance from current selection
        best_idx = -1
        best_dist = -1.0

        for i, (cand, orig_idx) in enumerate(available):
            dist = _compute_diversity_distance(
                cand, current_selected, use_hyperbolic, curvature
            )
            if dist > best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0:
            cand, orig_idx = available.pop(best_idx)
            selected_indices.append(orig_idx)
            current_selected.append(cand)

    return selected_indices


def compute_population_spread_hyperbolic(
    population: List[Tensor],
    curvature: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute population spread using hyperbolic distance.

    Returns:
        (mean_distance, max_distance) - statistics about population spread
    """
    from latent_reasoning.utils import hyperbolic as hyp

    if len(population) < 2:
        return 0.0, 0.0

    distances = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            d = hyp.hyperbolic_distance(
                population[i].squeeze(),
                population[j].squeeze(),
                curvature
            ).item()
            distances.append(d)

    if not distances:
        return 0.0, 0.0

    return sum(distances) / len(distances), max(distances)


class ElitistSelection(SelectionStrategy):
    """
    Elitist selection strategy with fitness-proportional selection for remaining slots.

    This strategy guarantees that the best candidates always survive (elitism) while
    filling remaining slots through weighted random selection. This balances strong
    exploitation of good solutions with exploration through diversity.

    Algorithm:
    1. Sort all candidates by fitness score (descending)
    2. Always select the top elite_k candidates
    3. For remaining slots, use fitness-proportional random selection
    4. Ensure no duplicates in the final selection

    Characteristics:
    - **High exploitation**: Best solutions always survive
    - **Moderate exploration**: Random selection maintains some diversity
    - **Stable convergence**: Prevents loss of good solutions
    - **Configurable pressure**: elite_k controls selection pressure

    Best for:
    - Problems where preserving the best solutions is critical
    - Situations requiring steady progress without backtracking
    - Balancing convergence speed with diversity maintenance
    """

    def __init__(self, elite_k: int = 2):
        """
        Initialize elitist selection strategy.

        Args:
            elite_k: Number of elite candidates to always keep. Should be
                less than the typical population size. Higher values increase
                selection pressure but may reduce diversity.
                - 1-2: Moderate elitism (recommended)
                - 3-5: Strong elitism (faster convergence, less diversity)
                - 0: No elitism (pure fitness-proportional selection)

        Example:
            >>> # Moderate elitism - keeps top 2, selects rest randomly
            >>> strategy = ElitistSelection(elite_k=2)
            >>>
            >>> # Strong elitism - keeps top 4 for faster convergence
            >>> strategy = ElitistSelection(elite_k=4)
        """
        self.elite_k = elite_k

    def select(
        self,
        population: List[Tensor],
        scores: List[float],
        n_survivors: int,
    ) -> Tuple[List[Tensor], List[float]]:
        if len(population) <= n_survivors:
            return population.copy(), scores.copy()

        # Sort by score
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # Always keep elite
        elite_count = min(self.elite_k, n_survivors)
        selected_indices = sorted_indices[:elite_count]

        # Select remaining from non-elite, weighted by score
        remaining_needed = n_survivors - elite_count
        if remaining_needed > 0:
            non_elite_indices = sorted_indices[elite_count:]
            non_elite_scores = [scores[i] for i in non_elite_indices]

            # Convert scores to selection probabilities
            # Shift scores to be positive
            min_score = min(non_elite_scores) if non_elite_scores else 0
            shifted_scores = [s - min_score + 0.1 for s in non_elite_scores]
            total = sum(shifted_scores)

            if total > 0:
                probs = [s / total for s in shifted_scores]
                additional = random.choices(
                    non_elite_indices,
                    weights=probs,
                    k=min(remaining_needed, len(non_elite_indices)),
                )
                selected_indices.extend(additional)

        selected_latents = [population[i].clone() for i in selected_indices]
        selected_scores = [scores[i] for i in selected_indices]

        return selected_latents, selected_scores

    def select_with_diversity(
        self,
        population: List[Tensor],
        scores: List[float],
        n_survivors: int,
        diversity_quota: float = 0.33,
    ) -> SelectionResult:
        """
        Select survivors with a diversity quota.

        Args:
            population: All candidate latent vectors
            scores: Fitness scores for each candidate
            n_survivors: Total number of survivors to select
            diversity_quota: Fraction of slots reserved for diversity (0-1)

        Returns:
            SelectionResult with selected latents and indices of diversity-protected chains
        """
        if len(population) <= n_survivors:
            # Even when keeping all, still protect some for diversity
            n_diversity_protect = max(1, int(len(population) * diversity_quota)) if len(population) > 1 else 0
            # Protect the most diverse chains (those with lowest similarity to others)
            diversity_indices = set()
            if n_diversity_protect > 0 and len(population) > 1:
                # Select chains that are most different from the top scorer
                sorted_by_score = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                top_latent = population[sorted_by_score[0]]
                # Compute distances from top
                distances = []
                for i, lat in enumerate(population):
                    if i == sorted_by_score[0]:
                        distances.append((i, -1.0))  # Top scorer not for diversity
                    else:
                        dist = _compute_diversity_distance(lat, [top_latent])
                        distances.append((i, dist))
                # Sort by distance (descending) and take top n_diversity_protect
                distances.sort(key=lambda x: x[1], reverse=True)
                for idx, _ in distances[:n_diversity_protect]:
                    if idx != sorted_by_score[0]:
                        diversity_indices.add(idx)
            return SelectionResult(
                latents=population.copy(),
                scores=scores.copy(),
                diversity_indices=diversity_indices,
            )

        # Calculate slots - ensure at least 1 diversity slot when population > 1
        n_diversity = max(1, int(n_survivors * diversity_quota)) if n_survivors > 1 else 0
        n_score_based = n_survivors - n_diversity

        # Sort by score
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # Always keep elite (score-based)
        elite_count = min(self.elite_k, n_score_based)
        score_selected = sorted_indices[:elite_count]

        # Fill remaining score-based slots
        score_remaining = n_score_based - elite_count
        if score_remaining > 0:
            non_elite_indices = sorted_indices[elite_count:]
            non_elite_scores = [scores[i] for i in non_elite_indices]

            min_score = min(non_elite_scores) if non_elite_scores else 0
            shifted_scores = [s - min_score + 0.1 for s in non_elite_scores]
            total = sum(shifted_scores)

            if total > 0:
                probs = [s / total for s in shifted_scores]
                additional = random.choices(
                    non_elite_indices,
                    weights=probs,
                    k=min(score_remaining, len(non_elite_indices)),
                )
                score_selected.extend(additional)

        # Get already selected latents for diversity computation
        already_selected_latents = [population[i] for i in score_selected]

        # Find candidates not yet selected
        selected_set = set(score_selected)
        remaining_indices = [i for i in range(len(population)) if i not in selected_set]
        remaining_latents = [population[i] for i in remaining_indices]

        # Select diversity candidates
        diversity_original_indices = _select_for_diversity(
            remaining_latents,
            remaining_indices,
            already_selected_latents,
            n_diversity,
        )

        # Combine selections
        all_selected_indices = list(score_selected) + diversity_original_indices

        # Track which final indices are diversity-protected
        diversity_final_indices = set(range(len(score_selected), len(all_selected_indices)))

        selected_latents = [population[i].clone() for i in all_selected_indices]
        selected_scores = [scores[i] for i in all_selected_indices]

        return SelectionResult(
            latents=selected_latents,
            scores=selected_scores,
            diversity_indices=diversity_final_indices,
        )


class TournamentSelection(SelectionStrategy):
    """
    Tournament selection strategy.

    Randomly selects groups and picks the best from each group.
    """

    def __init__(self, tournament_size: int = 3):
        """
        Initialize tournament selection.

        Args:
            tournament_size: Number of candidates in each tournament
        """
        self.tournament_size = tournament_size

    def select(
        self,
        population: List[Tensor],
        scores: List[float],
        n_survivors: int,
    ) -> Tuple[List[Tensor], List[float]]:
        if len(population) <= n_survivors:
            return population.copy(), scores.copy()

        selected_indices = []
        used_indices = set()

        while len(selected_indices) < n_survivors:
            # Random tournament
            available = [i for i in range(len(population)) if i not in used_indices]
            if not available:
                # Reset if we've used all
                used_indices.clear()
                available = list(range(len(population)))

            tournament = random.sample(available, min(self.tournament_size, len(available)))

            # Pick winner
            winner = max(tournament, key=lambda i: scores[i])
            selected_indices.append(winner)
            used_indices.add(winner)

        selected_latents = [population[i].clone() for i in selected_indices]
        selected_scores = [scores[i] for i in selected_indices]

        return selected_latents, selected_scores


class RankSelection(SelectionStrategy):
    """
    Rank selection strategy.

    Simply selects the top N by score.
    """

    def select(
        self,
        population: List[Tensor],
        scores: List[float],
        n_survivors: int,
    ) -> Tuple[List[Tensor], List[float]]:
        if len(population) <= n_survivors:
            return population.copy(), scores.copy()

        # Sort by score descending
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted_indices[:n_survivors]

        selected_latents = [population[i].clone() for i in top_indices]
        selected_scores = [scores[i] for i in top_indices]

        return selected_latents, selected_scores


class RouletteSelection(SelectionStrategy):
    """
    Roulette wheel selection strategy.

    Selection probability proportional to fitness score.
    """

    def select(
        self,
        population: List[Tensor],
        scores: List[float],
        n_survivors: int,
    ) -> Tuple[List[Tensor], List[float]]:
        if len(population) <= n_survivors:
            return population.copy(), scores.copy()

        # Shift scores to be positive
        min_score = min(scores)
        shifted_scores = [s - min_score + 0.1 for s in scores]
        total = sum(shifted_scores)

        probs = [s / total for s in shifted_scores]

        # Select with replacement allowed
        selected_indices = random.choices(
            range(len(population)),
            weights=probs,
            k=n_survivors,
        )

        selected_latents = [population[i].clone() for i in selected_indices]
        selected_scores = [scores[i] for i in selected_indices]

        return selected_latents, selected_scores


def get_selection_strategy(name: str, **kwargs) -> SelectionStrategy:
    """Factory function to get a selection strategy by name."""
    strategies = {
        "elitist": ElitistSelection,
        "tournament": TournamentSelection,
        "rank": RankSelection,
        "roulette": RouletteSelection,
    }

    if name not in strategies:
        raise ValueError(f"Unknown selection strategy: {name}")

    return strategies[name](**kwargs)
