"""
Base protocols and abstract classes for evolutionary operators.

This module defines the interfaces that all evolutionary operators must implement.
These operators form the core of the evolutionary algorithm:

- SelectionStrategy: Chooses which candidates survive to the next generation
- MutationStrategy: Creates variations of existing candidates
- CrossoverStrategy: Combines multiple candidates to create offspring

Each operator type has specific responsibilities and design patterns that
implementations should follow for effective evolution in latent space.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from torch import Tensor


class SelectionStrategy(ABC):
    """
    Abstract base class for selection strategies in evolutionary algorithms.

    Selection strategies determine which candidates from the current population
    survive to reproduce in the next generation. This is a crucial component
    that balances exploitation (keeping the best solutions) with exploration
    (maintaining diversity for continued evolution).

    Key design principles:
    - **Fitness-based**: Better scores should increase survival probability
    - **Diversity-aware**: Avoid premature convergence by maintaining variety
    - **Scalable**: Should work efficiently with different population sizes
    - **Deterministic**: Same inputs should produce same outputs for reproducibility

    Common strategies:
    - Elitist: Always keep top K candidates, select rest randomly
    - Tournament: Run mini-competitions between random subsets
    - Roulette: Probability proportional to fitness scores
    - Rank-based: Selection based on rank rather than raw scores
    """

    @abstractmethod
    def select(
        self,
        population: List[Tensor],
        scores: List[float],
        n_survivors: int,
    ) -> Tuple[List[Tensor], List[float]]:
        """
        Select survivors from the current population for the next generation.

        This method implements the core selection logic that determines which
        latent vectors continue to evolve. The selection should balance keeping
        high-quality solutions while maintaining enough diversity for continued
        improvement.

        Args:
            population: List of latent vectors representing current candidates.
                All tensors should have the same shape and be on the same device.
            scores: Corresponding fitness scores for each candidate. Higher scores
                indicate better quality. Scores may be negative.
            n_survivors: Number of candidates to select for survival. Should be
                less than or equal to len(population).

        Returns:
            Tuple containing:
            - selected_latents: List of surviving latent vectors
            - selected_scores: Corresponding scores for the survivors

            Both lists should have exactly n_survivors elements and maintain
            the correspondence between latents and scores.

        Raises:
            ValueError: If n_survivors > len(population) or if population/scores
                have mismatched lengths
            RuntimeError: If selection fails due to invalid scores (e.g., all NaN)

        Example:
            >>> strategy = ElitistSelection(elite_k=2)
            >>> survivors, scores = strategy.select(
            ...     population=[latent1, latent2, latent3, latent4],
            ...     scores=[0.8, 0.6, 0.9, 0.7],
            ...     n_survivors=3
            ... )
            >>> # Returns top 3: [latent3, latent1, latent4] with scores [0.9, 0.8, 0.7]
        """
        pass


class MutationStrategy(ABC):
    """
    Abstract base class for mutation strategies in evolutionary algorithms.

    Mutation strategies create variations of existing candidates by introducing
    controlled random changes to latent vectors. This is the primary source of
    exploration in the evolutionary process, allowing the algorithm to discover
    new regions of the latent space.

    Key design principles:
    - **Controlled Randomness**: Changes should be meaningful but not destructive
    - **Temperature-aware**: Mutation strength should be adjustable over time
    - **Hint-responsive**: Should utilize guidance from modifier judges when available
    - **Preserving Structure**: Maintain the general structure of latent representations

    Mutation strategies balance:
    - **Exploration**: Finding new areas of latent space
    - **Exploitation**: Making small improvements to existing solutions
    - **Diversity**: Preventing population from becoming too similar
    - **Stability**: Avoiding changes that break the latent representation

    Common approaches:
    - Gaussian: Add random noise from normal distribution
    - Directed: Follow hints from modifier judges about improvement directions
    - Adaptive: Adjust mutation based on population diversity or progress
    - Structured: Respect the structure of the latent space
    """

    @abstractmethod
    def mutate(
        self,
        candidate: Tensor,
        modification_hint: Tensor | None,
        temperature: float,
    ) -> Tensor:
        """
        Create a mutated version of a candidate latent vector.

        This method applies controlled random changes to explore the latent space
        around the current candidate. The mutation should be influenced by the
        temperature (controlling strength) and optional hints from judges.

        Args:
            candidate: Latent vector to mutate. Shape should be preserved in output.
                The input tensor should not be modified in-place.
            modification_hint: Optional direction vector from modifier judges
                indicating promising directions for improvement. If provided,
                should have the same shape as candidate. May be None if no
                guidance is available.
            temperature: Mutation strength parameter, typically in range [0, 2].
                - 0.0: No mutation (return copy of candidate)
                - 1.0: Standard mutation strength
                - 2.0: Strong mutation for exploration
                Higher values increase exploration, lower values focus on exploitation.

        Returns:
            Mutated latent vector with the same shape and device as the input.
            Should be a new tensor, not a modified version of the input.

        Example:
            >>> strategy = GaussianMutation(noise_scale=0.1)
            >>> mutated = strategy.mutate(
            ...     candidate=original_latent,
            ...     modification_hint=improvement_direction,
            ...     temperature=0.8
            ... )
            >>> # mutated is similar to original_latent but with controlled changes

        Note:
            - Should be deterministic given the same random seed
            - Output should remain a valid latent representation
            - Consider the magnitude and distribution of the input latent
        """
        pass


class CrossoverStrategy(ABC):
    """
    Abstract base class for crossover strategies in evolutionary algorithms.

    Crossover strategies combine information from multiple parent solutions to
    create offspring that potentially inherit the best characteristics from
    each parent. This is a key mechanism for combining successful features
    discovered in different parts of the population.

    Key design principles:
    - **Information Preservation**: Maintain valuable features from both parents
    - **Meaningful Combination**: Create coherent latent representations
    - **Fitness-aware**: Better parents should contribute more to offspring
    - **Diversity Generation**: Produce children different from both parents

    Crossover in latent space considerations:
    - **Semantic Coherence**: Combined latents should represent meaningful concepts
    - **Dimensional Consistency**: Respect the structure of the latent space
    - **Interpolation vs Recombination**: Choose appropriate mixing strategies
    - **Stability**: Avoid creating invalid or degenerate representations

    Common strategies:
    - Blend: Weighted average based on fitness scores
    - Uniform: Randomly select elements from each parent
    - Arithmetic: Simple averaging or linear interpolation
    - Semantic: Combine based on semantic similarity
    """

    @abstractmethod
    def crossover(
        self,
        parent_a: Tensor,
        parent_b: Tensor,
        score_a: float = 0.0,
        score_b: float = 0.0,
    ) -> Tensor:
        """
        Combine two parent latent vectors to create an offspring.

        This method implements the core crossover logic that combines information
        from two successful candidates to create a new candidate that potentially
        inherits beneficial characteristics from both parents.

        Args:
            parent_a: First parent latent vector. Should be a valid latent
                representation that has been evaluated and selected for reproduction.
            parent_b: Second parent latent vector. Should have the same shape
                and be on the same device as parent_a.
            score_a: Fitness score of first parent. Higher scores indicate
                better quality. Used by weighted crossover strategies to
                determine relative contribution.
            score_b: Fitness score of second parent. Used for weighting
                the contribution of parent_b in the offspring.

        Returns:
            Child latent vector with the same shape and device as the parents.
            Should represent a meaningful combination of the parent characteristics.

        Raises:
            ValueError: If parents have incompatible shapes or are on different devices
            RuntimeError: If crossover fails due to invalid parent representations

        Example:
            >>> strategy = BlendCrossover()
            >>> child = strategy.crossover(
            ...     parent_a=high_quality_latent,    # score: 0.9
            ...     parent_b=diverse_latent,         # score: 0.7
            ...     score_a=0.9,
            ...     score_b=0.7
            ... )
            >>> # child inherits more from high_quality_latent due to higher score

        Note:
            - Should be deterministic given the same inputs and random seed
            - Output should be a valid latent representation
            - Consider both fitness scores and semantic compatibility
        """
        pass
