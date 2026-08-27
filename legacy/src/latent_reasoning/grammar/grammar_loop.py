"""
Grammar-based evolution loop.

This module provides an evolution loop that operates on FractalGrammars
instead of raw latent vectors. The key insight is that evolving grammars
provides:
- Better compression (grammars are smaller than latents)
- More structure (AND/OR trees encode composition)
- Natural diversity (different structures → different behaviors)

The loop integrates with QD for diversity maintenance and uses
depth-adaptive mutation for structured exploration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
import random

import torch
from torch import Tensor

from latent_reasoning.grammar.grammar import FractalGrammar, GrammarStats
from latent_reasoning.grammar.mutation import GrammarMutationStrategy, GrammarCrossoverStrategy

if TYPE_CHECKING:
    from latent_reasoning.config import GrammarConfig, QDConfig
    from latent_reasoning.core.scorer import LatentScorer
    from latent_reasoning.qd.manager import QDManager


@dataclass
class GrammarEvolutionResult:
    """Result of grammar evolution."""
    best_grammar: FractalGrammar
    best_latent: Tensor
    best_score: float
    final_population: list[FractalGrammar]
    history: list[dict] = field(default_factory=list)

    @property
    def grammar_stats(self) -> GrammarStats:
        """Get stats for the best grammar."""
        return self.best_grammar.stats


@dataclass
class GrammarIndividual:
    """An individual in the grammar population."""
    grammar: FractalGrammar
    latent: Tensor | None = None
    score: float = 0.0
    generation_created: int = 0


class GrammarEvolutionLoop:
    """
    Evolution loop operating on FractalGrammars.

    Instead of evolving raw latent vectors, this loop evolves grammars
    that GENERATE latents. This provides:
    - Compression: Grammars are more compact than high-dim latents
    - Structure: AND/OR trees encode meaningful composition
    - Diversity: Different structures produce different behaviors

    The loop follows a (μ + λ) evolution strategy:
    1. Initialize population of random grammars
    2. Expand grammars to get latents
    3. Score latents (quality assessment)
    4. Select top grammars
    5. Mutate/crossover to create offspring
    6. Repeat

    Args:
        grammar_config: Configuration for grammars
        latent_dim: Dimension of latent vectors
        population_size: Number of grammars in population
        offspring_size: Number of offspring per generation
        mutation_rate: Base mutation rate
        crossover_rate: Probability of crossover vs mutation
        tournament_size: Tournament selection size
        device: Device for computations

    Usage:
        >>> loop = GrammarEvolutionLoop(config, latent_dim=1024)
        >>> result = loop.run(
        ...     scorer=scorer,
        ...     query="Explain quantum computing",
        ...     num_generations=20,
        ... )
        >>> print(f"Best grammar: {result.best_grammar}")
    """

    def __init__(
        self,
        grammar_config: "GrammarConfig",
        latent_dim: int,
        population_size: int = 20,
        offspring_size: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.3,
        tournament_size: int = 3,
        device: torch.device | str = "cpu",
        qd_manager: "QDManager | None" = None,
    ):
        self.grammar_config = grammar_config
        self.latent_dim = latent_dim
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # QD integration
        self.qd_manager = qd_manager

        # Mutation and crossover strategies
        self.mutation_strategy = GrammarMutationStrategy(
            grammar_config,
            base_mutation_rate=mutation_rate,
        )
        self.crossover_strategy = GrammarCrossoverStrategy(grammar_config)

        # Population
        self.population: list[GrammarIndividual] = []
        self.generation = 0
        self.best_ever: GrammarIndividual | None = None

        # History
        self.history: list[dict] = []

    def initialize_population(self, seed_grammar: FractalGrammar | None = None) -> None:
        """
        Initialize the grammar population.

        Args:
            seed_grammar: Optional seed grammar to clone and mutate
        """
        self.population = []

        for i in range(self.population_size):
            if seed_grammar is not None and i == 0:
                # Keep original seed
                grammar = seed_grammar.clone()
            elif seed_grammar is not None and i < self.population_size // 2:
                # Mutate seed for diversity
                grammar = self.mutation_strategy.mutate(
                    seed_grammar,
                    generation=0,
                    temperature=1.0 + i * 0.1,
                )
            else:
                # Create grammar based on init_strategy from config
                init_strategy = getattr(self.grammar_config, 'init_strategy', 'random')
                if init_strategy == "balanced":
                    grammar = FractalGrammar.balanced(
                        config=self.grammar_config,
                        latent_dim=self.latent_dim,
                        depth=self.grammar_config.min_depth,
                        branching=self.grammar_config.branching_factor,
                        device=self.device,
                    )
                elif init_strategy == "deep":
                    # Use balanced with deeper depth
                    grammar = FractalGrammar.balanced(
                        config=self.grammar_config,
                        latent_dim=self.latent_dim,
                        depth=self.grammar_config.max_depth,
                        branching=self.grammar_config.branching_factor,
                        device=self.device,
                    )
                else:  # "random" or default
                    grammar = FractalGrammar.random(
                        self.grammar_config,
                        self.latent_dim,
                        self.device,
                    )

            individual = GrammarIndividual(
                grammar=grammar,
                generation_created=0,
            )
            self.population.append(individual)

        self.generation = 0
        self.best_ever = None
        self.history = []

    def run(
        self,
        scorer: "LatentScorer",
        query: str,
        num_generations: int = 20,
        seed_latent: Tensor | None = None,
        early_stop_threshold: float = 0.95,
        verbose: bool = False,
    ) -> GrammarEvolutionResult:
        """
        Run grammar evolution.

        Args:
            scorer: Latent scorer for quality assessment
            query: Query string for scoring context
            num_generations: Number of generations to evolve
            seed_latent: Optional seed for grammar expansion
            early_stop_threshold: Stop if score exceeds this
            verbose: Print progress

        Returns:
            GrammarEvolutionResult with best grammar and history
        """
        # Initialize if needed
        if not self.population:
            self.initialize_population()

        # Evolution loop
        for gen in range(num_generations):
            self.generation = gen

            # 1. Expand grammars to latents
            self._expand_population(seed_latent)

            # 2. Score latents
            self._score_population(scorer, query)

            # 3. Update best ever
            self._update_best()

            # 4. Record history
            self._record_history()

            if verbose:
                best = max(self.population, key=lambda ind: ind.score)
                print(f"Gen {gen}: best={best.score:.4f}, avg={self._avg_score():.4f}")

            # 5. Early stopping
            if self.best_ever and self.best_ever.score >= early_stop_threshold:
                if verbose:
                    print(f"Early stop at generation {gen}")
                break

            # 6. Selection and reproduction
            self._selection_and_reproduction()

        # Return result
        if self.best_ever is None:
            best = max(self.population, key=lambda ind: ind.score)
            self.best_ever = best

        return GrammarEvolutionResult(
            best_grammar=self.best_ever.grammar,
            best_latent=self.best_ever.latent,
            best_score=self.best_ever.score,
            final_population=[ind.grammar for ind in self.population],
            history=self.history,
        )

    def _expand_population(self, seed_latent: Tensor | None) -> None:
        """Expand all grammars to generate latents."""
        for individual in self.population:
            with torch.no_grad():
                individual.latent = individual.grammar.expand(
                    seed=seed_latent,
                    temperature=1.0,
                )

    def _score_population(self, scorer: "LatentScorer", query: str) -> None:
        """Score all individuals using the latent scorer."""
        for individual in self.population:
            if individual.latent is not None:
                with torch.no_grad():
                    # Handle different scorer interfaces
                    try:
                        # Try new interface with query
                        score_result = scorer.score(individual.latent, query)
                    except TypeError:
                        # Fallback to old interface without query
                        score_result = scorer.score(individual.latent)

                    # Handle both ScoreResult and float returns
                    if hasattr(score_result, 'overall'):
                        individual.score = score_result.overall
                    else:
                        individual.score = float(score_result)

                # Update QD archive if available
                if self.qd_manager is not None:
                    self.qd_manager.update(
                        individual.latent,
                        individual.score,
                        metadata={
                            "grammar": individual.grammar.to_dict(),
                            "generation": self.generation,
                        },
                    )

    def _update_best(self) -> None:
        """Update best ever individual."""
        current_best = max(self.population, key=lambda ind: ind.score)

        if self.best_ever is None or current_best.score > self.best_ever.score:
            self.best_ever = GrammarIndividual(
                grammar=current_best.grammar.clone(),
                latent=current_best.latent.clone() if current_best.latent is not None else None,
                score=current_best.score,
                generation_created=current_best.generation_created,
            )

    def _avg_score(self) -> float:
        """Compute average population score."""
        if not self.population:
            return 0.0
        return sum(ind.score for ind in self.population) / len(self.population)

    def _record_history(self) -> None:
        """Record generation statistics."""
        scores = [ind.score for ind in self.population]
        stats = [ind.grammar.stats for ind in self.population]

        record = {
            "generation": self.generation,
            "best_score": max(scores),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "best_ever_score": self.best_ever.score if self.best_ever else 0.0,
            "avg_nodes": sum(s.num_nodes for s in stats) / len(stats),
            "avg_depth": sum(s.max_depth for s in stats) / len(stats),
            "avg_compression": sum(s.compression_ratio for s in stats) / len(stats),
        }

        if self.qd_manager is not None:
            record["qd_archive_size"] = len(self.qd_manager.archive)
            record["qd_coverage"] = getattr(self.qd_manager.archive, 'coverage', 0.0)

        self.history.append(record)

    def _selection_and_reproduction(self) -> None:
        """Select parents and create offspring."""
        # Sort by score (descending)
        self.population.sort(key=lambda ind: ind.score, reverse=True)

        # Keep top individuals (elitism)
        elite_size = max(2, self.population_size // 5)
        new_population = self.population[:elite_size]

        # Fill rest with offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(self.population) >= 2:
                # Crossover
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                child1, child2 = self.crossover_strategy.crossover(
                    parent1.grammar,
                    parent2.grammar,
                )
                new_population.append(GrammarIndividual(
                    grammar=child1,
                    generation_created=self.generation + 1,
                ))
                if len(new_population) < self.population_size:
                    new_population.append(GrammarIndividual(
                        grammar=child2,
                        generation_created=self.generation + 1,
                    ))
            else:
                # Mutation
                parent = self._tournament_select()
                child = self.mutation_strategy.mutate(
                    parent.grammar,
                    generation=self.generation,
                    temperature=self._compute_temperature(),
                )
                new_population.append(GrammarIndividual(
                    grammar=child,
                    generation_created=self.generation + 1,
                ))

        self.population = new_population[:self.population_size]

    def _tournament_select(self) -> GrammarIndividual:
        """Select individual via tournament selection."""
        candidates = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(candidates, key=lambda ind: ind.score)

    def _compute_temperature(self) -> float:
        """Compute mutation temperature based on generation."""
        # Start high, decay over time
        base_temp = 1.0
        decay = 0.95
        return base_temp * (decay ** self.generation)

    def reset(self) -> None:
        """Reset the evolution state."""
        self.population = []
        self.generation = 0
        self.best_ever = None
        self.history = []

    def inject_grammar(self, grammar: FractalGrammar) -> None:
        """Inject a grammar into the population."""
        individual = GrammarIndividual(
            grammar=grammar.clone(),
            generation_created=self.generation,
        )
        self.population.append(individual)

        # Keep population size bounded
        if len(self.population) > self.population_size:
            # Remove worst
            self.population.sort(key=lambda ind: ind.score, reverse=True)
            self.population = self.population[:self.population_size]

    def get_diverse_grammars(self, n: int = 5) -> list[FractalGrammar]:
        """
        Get n diverse grammars from the population.

        Diversity is measured by tree structure difference.
        """
        if not self.population:
            return []

        # Sort by score
        sorted_pop = sorted(self.population, key=lambda ind: ind.score, reverse=True)

        # Greedily select diverse grammars
        selected = [sorted_pop[0].grammar]

        for ind in sorted_pop[1:]:
            if len(selected) >= n:
                break

            # Check structural diversity
            is_diverse = True
            for sel in selected:
                # Simple diversity: different tree structure
                if ind.grammar.tree.num_nodes == sel.tree.num_nodes and \
                   ind.grammar.tree.max_depth == sel.tree.max_depth:
                    # Check rule overlap
                    overlap = len(ind.grammar.tree.rules_used & sel.tree.rules_used)
                    total = len(ind.grammar.tree.rules_used | sel.tree.rules_used)
                    if total > 0 and overlap / total > 0.8:
                        is_diverse = False
                        break

            if is_diverse:
                selected.append(ind.grammar)

        return selected

    @classmethod
    def from_config(
        cls,
        grammar_config: "GrammarConfig",
        latent_dim: int,
        device: torch.device | str = "cpu",
        qd_manager: "QDManager | None" = None,
    ) -> "GrammarEvolutionLoop":
        """Create evolution loop from config."""
        return cls(
            grammar_config=grammar_config,
            latent_dim=latent_dim,
            population_size=grammar_config.population_size,
            offspring_size=grammar_config.offspring_size,
            mutation_rate=grammar_config.mutation_rate,
            crossover_rate=grammar_config.crossover_rate,
            tournament_size=grammar_config.tournament_size,
            device=device,
            qd_manager=qd_manager,
        )
