"""Main evolution loop for latent space reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor

from latent_reasoning.config import EvolutionConfig
from latent_reasoning.core.chain import ChainState, ChainTracker, compute_cross_chain_summary
from latent_reasoning.core.panel import JudgePanel
from latent_reasoning.evolution.operators import SelectionStrategy, MutationStrategy, CrossoverStrategy
from latent_reasoning.evolution.selection import get_selection_strategy
from latent_reasoning.evolution.mutation import get_mutation_strategy, AdaptiveMutation
from latent_reasoning.evolution.crossover import get_crossover_strategy, select_crossover_pairs, population_diversity
from latent_reasoning.utils.logging import log_generation, log_event, LogLevel


@dataclass
class EvolutionResult:
    """Result of the evolution process."""

    best_latent: Tensor
    best_score: float
    survivors: List[ChainState]
    generations: int
    total_evaluations: int
    history: List[dict] = field(default_factory=list)
    converged: bool = False
    stop_reason: str = ""


@dataclass
class GenerationSnapshot:
    """Snapshot of a single generation."""

    generation: int
    latents: List[Tensor]
    scores: List[float]
    best_score: float
    mean_score: float
    diversity: float


class EvolutionLoop:
    """
    Main evolution loop for optimizing latent vectors through evolutionary algorithms.

    This class implements the core evolutionary optimization process that improves
    latent representations through iterative selection, mutation, and crossover
    operations. It's the heart of the latent space reasoning system.

    The evolutionary process:
    1. **Initialization**: Create initial population from seed latent vector
    2. **Evaluation**: Score all candidates using the judge panel
    3. **Selection**: Choose the best candidates for reproduction
    4. **Reproduction**: Create new candidates through mutation and crossover
    5. **Replacement**: Replace old population with new candidates
    6. **Repeat**: Continue until convergence or maximum generations

    Key features:
    - **Adaptive Temperature**: Gradually reduces mutation strength over time
    - **Convergence Detection**: Stops early when no improvement is found
    - **Diversity Maintenance**: Prevents premature convergence through diversity metrics
    - **Budget Management**: Respects evaluation limits for computational efficiency
    - **Chain Tracking**: Maintains history for analysis and debugging

    The algorithm balances exploration (finding new regions of latent space) with
    exploitation (refining promising solutions) to find high-quality reasoning
    representations.

    Example:
        >>> from latent_reasoning.core.panel import JudgePanel
        >>> from latent_reasoning.config import EvolutionConfig
        >>>
        >>> # Set up evolution
        >>> config = EvolutionConfig(generations=10, chains=8)
        >>> loop = EvolutionLoop(judge_panel, config)
        >>>
        >>> # Run evolution on a seed latent
        >>> result = loop.run(seed_latent, max_evaluations=100)
        >>> print(f"Best score: {result.best_score:.3f}")
        >>> print(f"Generations: {result.generations}")
        >>> print(f"Converged: {result.converged}")
    """

    def __init__(
        self,
        judge_panel: JudgePanel,
        config: EvolutionConfig,
        selection: SelectionStrategy | None = None,
        mutation: MutationStrategy | None = None,
        crossover: CrossoverStrategy | None = None,
    ):
        """
        Initialize the evolution loop with strategies and configuration.

        Sets up the evolutionary algorithm with the specified strategies and
        parameters. If strategies are not provided, they will be created
        automatically based on the configuration.

        Args:
            judge_panel: Panel of judges for evaluating latent vectors. This
                determines how fitness is calculated and should include both
                scoring judges (for fitness) and modifier judges (for guidance).
            config: Evolution configuration containing all parameters:
                - generations: Maximum number of evolution cycles
                - chains: Population size (number of parallel evolution chains)
                - temperature: Initial mutation strength
                - selection/mutation/crossover settings
                - convergence criteria
            selection: Custom selection strategy. If None, creates strategy
                from config.selection (e.g., ElitistSelection, TournamentSelection).
            mutation: Custom mutation strategy. If None, creates strategy
                from config.mutation (e.g., GaussianMutation, DirectedMutation).
            crossover: Custom crossover strategy. If None, creates strategy
                from config.crossover (e.g., BlendCrossover, UniformCrossover).

        Example:
            >>> from latent_reasoning.config import EvolutionConfig
            >>> from latent_reasoning.evolution.selection import ElitistSelection
            >>>
            >>> # Use default strategies from config
            >>> config = EvolutionConfig(generations=15, chains=10)
            >>> loop = EvolutionLoop(judge_panel, config)
            >>>
            >>> # Use custom selection strategy
            >>> custom_selection = ElitistSelection(elite_k=3)
            >>> loop = EvolutionLoop(judge_panel, config, selection=custom_selection)

        Note:
            - Strategies are created lazily based on configuration if not provided
            - Temperature starts at config.temperature and decays over time
            - All strategies must be compatible with the latent vector dimensions
        """
        self.judge_panel = judge_panel
        self.config = config

        # Set up strategies with appropriate kwargs
        if selection is not None:
            self.selection = selection
        else:
            selection_kwargs = self._get_selection_kwargs(config.selection)
            self.selection = get_selection_strategy(
                config.selection.strategy,
                **selection_kwargs,
            )

        if mutation is not None:
            self.mutation = mutation
        else:
            mutation_kwargs = self._get_mutation_kwargs(config.mutation)
            self.mutation = get_mutation_strategy(
                config.mutation.strategy,
                **mutation_kwargs,
            )

        if crossover is not None:
            self.crossover = crossover
        else:
            self.crossover = get_crossover_strategy(config.crossover.strategy)

        # State
        self.current_temperature = config.temperature
        self.total_evaluations = 0

    def run(
        self,
        seed: Tensor,
        max_evaluations: int | None = None,
    ) -> EvolutionResult:
        """
        Run the complete evolutionary optimization process.

        This method executes the main evolutionary loop, starting from a seed latent
        vector and iteratively improving it through selection, mutation, and crossover
        until convergence or resource limits are reached.

        The process:
        1. Initialize population by mutating the seed vector
        2. For each generation:
           a. Evaluate all candidates using the judge panel
           b. Track best solution and convergence metrics
           c. Select survivors based on fitness scores
           d. Generate new candidates through mutation and crossover
           e. Apply diversity maintenance and temperature decay
        3. Return the best solution found with detailed statistics

        Args:
            seed: Initial latent vector to start evolution from. This should be
                the encoded representation of the original query. Shape: (latent_dim,)
            max_evaluations: Maximum number of judge evaluations to perform.
                If None, uses the budget from configuration. Useful for limiting
                computational cost in resource-constrained environments.

        Returns:
            EvolutionResult containing:
            - best_latent: The highest-scoring latent vector found
            - best_score: Score of the best solution
            - survivors: Final population of high-quality solutions
            - generations: Number of generations actually run
            - total_evaluations: Total judge evaluations performed
            - history: Per-generation statistics for analysis
            - converged: Whether the algorithm converged early
            - stop_reason: Why the evolution stopped

        Example:
            >>> # Basic evolution
            >>> result = loop.run(seed_latent)
            >>> print(f"Improved from seed to {result.best_score:.3f}")
            >>>
            >>> # Limited budget evolution
            >>> result = loop.run(seed_latent, max_evaluations=50)
            >>> print(f"Used {result.total_evaluations} evaluations")
            >>>
            >>> # Analyze evolution progress
            >>> for gen_stats in result.history:
            ...     print(f"Gen {gen_stats['generation']}: {gen_stats['best_score']:.3f}")

        Note:
            - Evolution may stop early if convergence is detected
            - The seed vector is always included in the initial population
            - Temperature automatically decays to reduce mutation strength over time
            - All returned tensors are on the same device as the input seed
        """
        device = seed.device

        # Initialize population
        chains = self._initialize_population(seed)
        trackers = [ChainTracker() for _ in chains]

        history = []
        best_latent = seed.clone()
        best_score = float("-inf")
        generations_without_improvement = 0

        for gen in range(self.config.generations):
            # Check evaluation budget
            if max_evaluations and self.total_evaluations >= max_evaluations:
                log_event("BUDGET_EXHAUSTED", level=LogLevel.NORMAL)
                break

            # Compute cross-chain summary
            cross_chain = compute_cross_chain_summary(chains)

            # Evaluate all chains
            raw_scores = []
            for i, chain in enumerate(chains):
                context = trackers[i].get_context(chain.latent, cross_chain)
                verdict = self.judge_panel.evaluate(chain.latent, context)
                raw_scores.append(verdict.score)
                trackers[i].record(chain.latent, verdict.score)
                self.total_evaluations += 1

            # Add diversity bonus to encourage exploration
            scores = self._apply_diversity_bonus(chains, raw_scores)
            for i, chain in enumerate(chains):
                chain.score = scores[i]

            # Track best
            gen_best_idx = max(range(len(scores)), key=lambda i: scores[i])
            gen_best_score = scores[gen_best_idx]
            gen_mean_score = sum(scores) / len(scores)

            if gen_best_score > best_score:
                best_score = gen_best_score
                best_latent = chains[gen_best_idx].latent.clone()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Log progress
            log_generation(
                gen=gen + 1,
                chains=len(chains),
                best_score=gen_best_score,
                mean_score=gen_mean_score,
            )

            # Record history
            history.append({
                "generation": gen + 1,
                "best_score": gen_best_score,
                "mean_score": gen_mean_score,
                "num_chains": len(chains),
                "diversity": population_diversity([c.latent for c in chains]),
            })

            # Check convergence
            if gen_best_score >= self.config.convergence.threshold:
                log_event("CONVERGED", level=LogLevel.NORMAL, score=gen_best_score)
                return EvolutionResult(
                    best_latent=best_latent,
                    best_score=best_score,
                    survivors=self._get_top_k(chains, scores, self.config.selection.survivors),
                    generations=gen + 1,
                    total_evaluations=self.total_evaluations,
                    history=history,
                    converged=True,
                    stop_reason="score_threshold",
                )

            # Check patience
            if generations_without_improvement >= self.config.convergence.patience:
                log_event("OPTIMAL_FOUND", level=LogLevel.NORMAL,
                         message="No improvement found after several generations. Stopping evolution.")
                return EvolutionResult(
                    best_latent=best_latent,
                    best_score=best_score,
                    survivors=self._get_top_k(chains, scores, self.config.selection.survivors),
                    generations=gen + 1,
                    total_evaluations=self.total_evaluations,
                    history=history,
                    converged=True,
                    stop_reason="patience",
                )

            # Selection
            selected_latents, selected_scores = self.selection.select(
                [c.latent for c in chains],
                scores,
                self.config.selection.survivors,
            )

            # Get modifications for survivors
            modifications = []
            for i, latent in enumerate(selected_latents):
                context = trackers[min(i, len(trackers) - 1)].get_context(latent, cross_chain)
                mod = self.judge_panel.get_modification(latent, context)
                modifications.append(mod)

            # Mutation
            mutants = []
            for latent, mod in zip(selected_latents, modifications):
                mutated = self.mutation.mutate(latent, mod, self.current_temperature)
                mutants.append(mutated)

            # Update adaptive mutation if applicable
            if isinstance(self.mutation, AdaptiveMutation):
                self.mutation.update_adaptation(gen_best_score)

            # Crossover (if diverse enough)
            diversity = population_diversity(selected_latents)
            if diversity > self.config.crossover.threshold:
                pairs = select_crossover_pairs(
                    selected_latents,
                    selected_scores,
                    n_pairs=len(selected_latents) // 2,
                    diversity_threshold=self.config.crossover.threshold,
                )
                for idx_a, idx_b in pairs:
                    child = self.crossover.crossover(
                        selected_latents[idx_a],
                        selected_latents[idx_b],
                        selected_scores[idx_a],
                        selected_scores[idx_b],
                    )
                    mutants.append(child)

            # Merge similar chains
            mutants = self._merge_similar(mutants, self.config.merge.threshold)

            # Create new chains
            chains = [ChainState(latent=m, generation=gen + 1) for m in mutants]

            # Update trackers
            while len(trackers) < len(chains):
                trackers.append(ChainTracker())
            trackers = trackers[:len(chains)]

            # Decay temperature
            self.current_temperature *= self.config.temperature_decay

        # Final evaluation - actually score the final population
        cross_chain = compute_cross_chain_summary(chains)
        scores = []
        for i, chain in enumerate(chains):
            if chain.score == 0.0:  # Only evaluate if not already scored
                context = trackers[min(i, len(trackers) - 1)].get_context(chain.latent, cross_chain)
                verdict = self.judge_panel.evaluate(chain.latent, context)
                chain.score = verdict.score
                self.total_evaluations += 1
            scores.append(chain.score)

        # Update best if final evaluation found something better
        final_best_idx = max(range(len(scores)), key=lambda i: scores[i])
        if scores[final_best_idx] > best_score:
            best_score = scores[final_best_idx]
            best_latent = chains[final_best_idx].latent.clone()

        return EvolutionResult(
            best_latent=best_latent,
            best_score=best_score,
            survivors=self._get_top_k(chains, scores, self.config.selection.survivors),
            generations=self.config.generations,
            total_evaluations=self.total_evaluations,
            history=history,
            converged=False,
            stop_reason="max_generations",
        )

    def _initialize_population(self, seed: Tensor) -> List[ChainState]:
        """Initialize the population from a seed vector with high diversity."""
        chains = []
        # Use initial_diversity multiplier for much more varied starting population
        init_noise_scale = self.config.temperature * self.config.initial_diversity
        for _ in range(self.config.chains):
            # Add substantial noise to create diverse initial population
            noise = torch.randn_like(seed) * init_noise_scale
            latent = seed + noise
            chains.append(ChainState(latent=latent))
        return chains

    def _apply_diversity_bonus(
        self,
        chains: List[ChainState],
        raw_scores: List[float],
    ) -> List[float]:
        """Apply diversity bonus to scores to encourage exploration.

        Chains that are more different from others get a bonus, preventing
        premature convergence to a single solution.
        """
        if len(chains) <= 1 or self.config.diversity_weight <= 0:
            return raw_scores

        # Stack all latents for efficient computation
        latents = torch.stack([c.latent.flatten() for c in chains])

        # Compute pairwise cosine similarities
        norms = latents.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = latents / norms
        similarities = torch.mm(normalized, normalized.t())

        # For each chain, compute average similarity to OTHER chains
        # (exclude self-similarity on diagonal)
        n = len(chains)
        mask = 1.0 - torch.eye(n, device=latents.device)
        avg_similarity = (similarities * mask).sum(dim=1) / (n - 1)

        # Diversity bonus = 1 - avg_similarity (more different = higher bonus)
        diversity_bonus = (1.0 - avg_similarity).tolist()

        # Combine: final_score = raw_score + diversity_weight * diversity_bonus
        final_scores = []
        for raw, bonus in zip(raw_scores, diversity_bonus):
            final_scores.append(raw + self.config.diversity_weight * bonus)

        return final_scores

    def _get_top_k(
        self,
        chains: List[ChainState],
        scores: List[float],
        k: int,
    ) -> List[ChainState]:
        """Get the top K chains by score."""
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [chains[i] for i in sorted_indices[:k]]

    def _merge_similar(
        self,
        latents: List[Tensor],
        threshold: float,
    ) -> List[Tensor]:
        """Merge latents that are very similar."""
        if len(latents) < 2:
            return latents

        merged = []
        used = set()

        for i in range(len(latents)):
            if i in used:
                continue

            # Check for similar latents
            similar = [i]
            for j in range(i + 1, len(latents)):
                if j in used:
                    continue

                cos_sim = torch.nn.functional.cosine_similarity(
                    latents[i].flatten().unsqueeze(0).float(),
                    latents[j].flatten().unsqueeze(0).float(),
                ).item()

                if cos_sim > threshold:
                    similar.append(j)
                    used.add(j)

            # Merge similar latents by averaging
            if len(similar) > 1:
                merged_latent = torch.stack([latents[idx] for idx in similar]).mean(dim=0)
                merged.append(merged_latent)
            else:
                merged.append(latents[i])

            used.add(i)

        return merged

    def reset(self) -> None:
        """Reset the evolution loop state."""
        self.current_temperature = self.config.temperature
        self.total_evaluations = 0
        if isinstance(self.mutation, AdaptiveMutation):
            self.mutation.reset()

    @staticmethod
    def _get_selection_kwargs(config) -> dict:
        """Get kwargs for selection strategy based on strategy type."""
        strategy = config.strategy
        if strategy == "elitist":
            return {"elite_k": config.elite}
        elif strategy == "tournament":
            return {"tournament_size": config.elite}  # Use elite as tournament size
        elif strategy in ("rank", "roulette"):
            return {}
        return {}

    @staticmethod
    def _get_mutation_kwargs(config) -> dict:
        """Get kwargs for mutation strategy based on strategy type."""
        strategy = config.strategy
        if strategy == "gaussian":
            return {"noise_scale": config.trust}  # Use trust as noise scale
        elif strategy == "directed":
            return {"trust": config.trust}
        elif strategy == "adaptive":
            return {"base_trust": config.trust}
        return {}
