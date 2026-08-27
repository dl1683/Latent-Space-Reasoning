"""Main evolution loop for latent space reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Set

import torch
from torch import Tensor

from latent_reasoning.config import EvolutionConfig, GeometryConfig
from latent_reasoning.core.chain import ChainState, ChainTracker, compute_cross_chain_summary
from latent_reasoning.core.panel import JudgePanel
from latent_reasoning.evolution.operators import SelectionStrategy, MutationStrategy, CrossoverStrategy
from latent_reasoning.evolution.selection import get_selection_strategy, SelectionResult, ElitistSelection
from latent_reasoning.evolution.mutation import get_mutation_strategy, AdaptiveMutation, HyperbolicMutation, HyperbolicAdaptiveMutation
from latent_reasoning.evolution.crossover import (
    get_crossover_strategy,
    select_crossover_pairs,
    population_diversity,
    select_crossover_pairs_hyperbolic,
    population_diversity_hyperbolic,
    HyperbolicCrossover,
)
from latent_reasoning.utils.logging import log_generation, log_event, LogLevel

if TYPE_CHECKING:
    from latent_reasoning.qd import QDManager
    from latent_reasoning.core.autopoietic import AutopoieticPanel


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
    # QD fields (populated when QD is enabled)
    qd_archive_size: int = 0
    qd_archive_stats: dict = field(default_factory=dict)


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
        qd_manager: "QDManager | None" = None,
        autopoietic_panel: "AutopoieticPanel | None" = None,
        geometry_config: GeometryConfig | None = None,
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
            qd_manager: Optional QDManager for Quality Diversity integration.
                When provided, enables QD scoring which combines raw fitness
                with novelty to maintain diverse solutions in an archive.
            autopoietic_panel: Optional AutopoieticPanel for self-updating judge.
                When provided, enables homeostatic temperature control and
                periodic grounding against external evaluator.
            geometry_config: Optional GeometryConfig for hyperbolic latent space.
                When space="hyperbolic", enables Poincaré ball operations for
                mutation, crossover, and diversity computation.

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
        self.qd_manager = qd_manager
        self.autopoietic_panel = autopoietic_panel
        self.geometry_config = geometry_config or GeometryConfig()

        # Hyperbolic space setup (lazy loaded)
        self._hyperbolic = None
        self._use_hyperbolic = self.geometry_config.space == "hyperbolic"
        self._current_curvature = (
            self.geometry_config.initial_curvature
            if self.geometry_config.anneal_curvature
            else self.geometry_config.curvature
        )

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
        elif self._use_hyperbolic:
            # Use hyperbolic mutation when in hyperbolic space
            if config.mutation.strategy == "adaptive":
                self.mutation = HyperbolicAdaptiveMutation(
                    base_noise_scale=self.geometry_config.mutation_noise_scale,
                    curvature=self._current_curvature,
                    max_norm=self.geometry_config.max_norm,
                    base_trust=config.mutation.trust,
                )
            else:
                self.mutation = HyperbolicMutation(
                    noise_scale=self.geometry_config.mutation_noise_scale,
                    curvature=self._current_curvature,
                    max_norm=self.geometry_config.max_norm,
                    trust=config.mutation.trust,
                )
        else:
            mutation_kwargs = self._get_mutation_kwargs(config.mutation)
            self.mutation = get_mutation_strategy(
                config.mutation.strategy,
                **mutation_kwargs,
            )

        if crossover is not None:
            self.crossover = crossover
        elif self._use_hyperbolic:
            # Use hyperbolic crossover when in hyperbolic space
            self.crossover = HyperbolicCrossover(
                curvature=self._current_curvature,
                max_norm=self.geometry_config.max_norm,
                max_iterations=self.geometry_config.barycenter_iterations,
            )
        else:
            self.crossover = get_crossover_strategy(config.crossover.strategy)

        # State
        self.current_temperature = config.temperature
        self.total_evaluations = 0
        self._current_survivors = max(1, min(config.selection.survivors, config.chains))

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

        # Reset per-run state to avoid cross-query leakage when an orchestrator
        # is reused for multiple queries.
        self.total_evaluations = 0
        self.current_temperature = self.config.temperature

        # Initialize population
        chains = self._initialize_population(seed)
        trackers = [ChainTracker() for _ in chains]

        history = []
        best_latent = seed.clone()
        best_score = float("-inf")
        generations_without_improvement = 0
        self._current_survivors = max(1, min(self.config.selection.survivors, len(chains)))
        score_cache: dict[bytes, float] = {}

        for gen in range(self.config.generations):
            # Check evaluation budget
            if max_evaluations and self.total_evaluations >= max_evaluations:
                log_event("BUDGET_EXHAUSTED", level=LogLevel.NORMAL)
                break

            # Compute cross-chain summary
            cross_chain = compute_cross_chain_summary(chains)

            # Evaluate all chains
            raw_scores = []
            cache_hits = 0
            for i, chain in enumerate(chains):
                cached_score = None
                cache_key = None
                if self.config.score_cache:
                    cache_key = self._score_cache_key(chain.latent, self.config.score_cache_precision)
                    cached_score = score_cache.get(cache_key)

                if cached_score is None:
                    context = trackers[i].get_context(chain.latent, cross_chain)
                    verdict = self.judge_panel.evaluate(chain.latent, context)
                    score = verdict.score
                    # Guard against NaN scores (prevents evolution from stalling)
                    if score != score:  # NaN check
                        score = 0.0
                    if cache_key is not None:
                        score_cache[cache_key] = score
                    self.total_evaluations += 1
                else:
                    score = cached_score
                    cache_hits += 1

                raw_scores.append(score)
                trackers[i].record(chain.latent, score)

            # Apply QD scoring if manager is available, otherwise use diversity bonus
            if self.qd_manager is not None:
                # Compute behavioral descriptors
                bds = self.qd_manager.compute_bds(chains)

                # Compute novelty scores
                novelty_scores = self.qd_manager.compute_novelty(bds)

                # Combine fitness with novelty for QD scoring (for selection/exploration)
                qd_scores = self.qd_manager.combine_fitness(raw_scores, novelty_scores)
                # Use QD scores for selection (encourages exploration)
                scores = qd_scores

                # Update archive with this generation's solutions
                added, rejected = self.qd_manager.update_archive(
                    chains=chains,
                    bds=bds,
                    raw_scores=raw_scores,
                    qd_scores=qd_scores,
                    generation=gen,
                )
            else:
                # Fall back to simple diversity bonus
                scores = self._apply_diversity_bonus(chains, raw_scores)

            for i, chain in enumerate(chains):
                chain.score = scores[i]
                # Store raw score separately for final selection
                chain.raw_score = raw_scores[i]

            # Track best by RAW fitness (not QD score!)
            # QD scores are useful for exploration/selection, but the final
            # "best" answer should be the one with highest actual quality
            gen_best_raw_idx = max(range(len(raw_scores)), key=lambda i: raw_scores[i])
            gen_best_raw_score = raw_scores[gen_best_raw_idx]
            gen_mean_raw_score = sum(raw_scores) / len(raw_scores)

            # For logging, show QD scores (what drives selection)
            gen_best_idx = max(range(len(scores)), key=lambda i: scores[i])
            gen_best_score = scores[gen_best_idx]
            gen_mean_score = sum(scores) / len(scores)

            # Track best solution by RAW fitness (actual quality)
            if gen_best_raw_score > best_score:
                best_score = gen_best_raw_score
                best_latent = chains[gen_best_raw_idx].latent.clone()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Dynamically adjust survivor budget when enabled.
            current_survivor_budget = self._update_survivor_budget(
                generations_without_improvement=generations_without_improvement,
                current_population=len(chains),
                generation=gen + 1,
            )

            # Log progress
            log_generation(
                gen=gen + 1,
                chains=len(chains),
                best_score=gen_best_score,
                mean_score=gen_mean_score,
            )

            # Record history
            if self._use_hyperbolic:
                current_diversity = population_diversity_hyperbolic(
                    [c.latent for c in chains], self._current_curvature
                )
            else:
                current_diversity = population_diversity([c.latent for c in chains])

            gen_history = {
                "generation": gen + 1,
                "best_score": gen_best_score,  # QD score (for selection)
                "mean_score": gen_mean_score,  # QD mean
                "best_raw_score": gen_best_raw_score,  # Raw fitness (actual quality)
                "mean_raw_score": gen_mean_raw_score,  # Raw mean
                "num_chains": len(chains),
                "diversity": current_diversity,
                "survivor_budget": current_survivor_budget,
            }
            if self.config.score_cache:
                gen_history["score_cache_hits"] = cache_hits
            # Add geometry stats if hyperbolic
            if self._use_hyperbolic:
                gen_history["curvature"] = self._current_curvature
                gen_history["geometry"] = "hyperbolic"
            # Add QD stats if available
            if self.qd_manager is not None:
                qd_stats = self.qd_manager.get_archive_statistics()
                gen_history["qd_archive_size"] = qd_stats.get("size", 0)
                gen_history["qd_coverage"] = qd_stats.get("coverage", 0.0)
                gen_history["qd_added"] = added
                gen_history["qd_rejected"] = rejected
            # Add autopoietic stats if available
            if self.autopoietic_panel is not None:
                panel_stats = self.autopoietic_panel.get_statistics()
                gen_history["autopoietic_trust"] = panel_stats.judge_trust
                gen_history["autopoietic_correlation"] = panel_stats.judge_correlation
                gen_history["autopoietic_temperature"] = panel_stats.temperature
            history.append(gen_history)

            # Check convergence
            if gen_best_score >= self.config.convergence.threshold:
                log_event("CONVERGED", level=LogLevel.NORMAL, score=gen_best_score)
                qd_size, qd_stats = self._get_qd_stats()
                return EvolutionResult(
                    best_latent=best_latent,
                    best_score=best_score,
                    survivors=self._get_top_k(chains, scores, current_survivor_budget),
                    generations=gen + 1,
                    total_evaluations=self.total_evaluations,
                    history=history,
                    converged=True,
                    stop_reason="score_threshold",
                    qd_archive_size=qd_size,
                    qd_archive_stats=qd_stats,
                )

            # Check patience
            if generations_without_improvement >= self.config.convergence.patience:
                log_event("OPTIMAL_FOUND", level=LogLevel.NORMAL,
                         message="No improvement found after several generations. Stopping evolution.")
                qd_size, qd_stats = self._get_qd_stats()
                return EvolutionResult(
                    best_latent=best_latent,
                    best_score=best_score,
                    survivors=self._get_top_k(chains, scores, current_survivor_budget),
                    generations=gen + 1,
                    total_evaluations=self.total_evaluations,
                    history=history,
                    converged=True,
                    stop_reason="patience",
                    qd_archive_size=qd_size,
                    qd_archive_stats=qd_stats,
                )

            # Selection with diversity quota
            diversity_indices: Set[int] = set()
            use_diversity = isinstance(self.selection, ElitistSelection) and self.config.selection.diversity_quota > 0
            if use_diversity:
                selection_result = self.selection.select_with_diversity(
                    [c.latent for c in chains],
                    scores,
                    current_survivor_budget,
                    diversity_quota=self.config.selection.diversity_quota,
                )
                selected_latents = selection_result.latents
                selected_scores = selection_result.scores
                diversity_indices = selection_result.diversity_indices
            else:
                selected_latents, selected_scores = self.selection.select(
                    [c.latent for c in chains],
                    scores,
                    current_survivor_budget,
                )

            # Get modifications for survivors
            modifications = []
            for i, latent in enumerate(selected_latents):
                context = trackers[min(i, len(trackers) - 1)].get_context(latent, cross_chain)
                mod = self.judge_panel.get_modification(latent, context)
                modifications.append(mod)

            # Mutation - track both mutant and parent for history propagation
            # Also track which mutants came from diversity-protected parents
            mutants_with_parents = []  # List of (mutated_latent, parent_latent, is_diversity) tuples
            for i, (latent, mod) in enumerate(zip(selected_latents, modifications)):
                mutated = self.mutation.mutate(latent, mod, self.current_temperature)
                is_diversity = i in diversity_indices
                mutants_with_parents.append((mutated, latent, is_diversity))

            # Update adaptive mutation if applicable
            if isinstance(self.mutation, (AdaptiveMutation, HyperbolicAdaptiveMutation)):
                self.mutation.update_adaptation(gen_best_score)

            # Crossover (if diverse enough) - crossover children are not diversity-protected
            if self._use_hyperbolic:
                diversity = population_diversity_hyperbolic(selected_latents, self._current_curvature)
            else:
                diversity = population_diversity(selected_latents)

            if diversity > self.config.crossover.threshold:
                if self._use_hyperbolic:
                    pairs = select_crossover_pairs_hyperbolic(
                        selected_latents,
                        selected_scores,
                        n_pairs=len(selected_latents) // 2,
                        curvature=self._current_curvature,
                        diversity_threshold=self.config.crossover.threshold,
                    )
                else:
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
                    # For crossover, use the average of both parents as history
                    if self._use_hyperbolic:
                        # Use Karcher mean for hyperbolic parent averaging
                        hyp = self._get_hyperbolic()
                        parent_avg = hyp.karcher_mean(
                            torch.stack([selected_latents[idx_a].squeeze(), selected_latents[idx_b].squeeze()]),
                            c=self._current_curvature,
                        )
                    else:
                        parent_avg = (selected_latents[idx_a] + selected_latents[idx_b]) / 2
                    mutants_with_parents.append((child, parent_avg, False))

            # Build protected indices set for merge
            protected_for_merge = {i for i, (_, _, is_div) in enumerate(mutants_with_parents) if is_div}

            # Merge similar chains (preserve first parent for merged ones)
            # Diversity-protected chains are not merged away
            mutants_only = [m for m, _, _ in mutants_with_parents]
            merged_indices = self._merge_similar_with_indices(
                mutants_only, self.config.merge.threshold, protected_for_merge
            )
            merged_with_parents = [(mutants_only[i], mutants_with_parents[i][1]) for i in merged_indices]

            # Create new chains with history from parents
            chains = []
            for mutant, parent in merged_with_parents:
                # Create history with parent latent to enable trajectory-based BDs
                chain_history = [parent.clone()] if parent is not None else []
                chain = ChainState(latent=mutant, generation=gen + 1, history=chain_history)
                chains.append(chain)

            # Update trackers
            while len(trackers) < len(chains):
                trackers.append(ChainTracker())
            trackers = trackers[:len(chains)]

            # Decay temperature (or use autopoietic homeostatic control)
            if self.autopoietic_panel is not None:
                # Let autopoietic panel control temperature via homeostasis
                panel_stats = self.autopoietic_panel.step_generation(chains, gen)
                self.current_temperature = self.autopoietic_panel.temperature
            else:
                self.current_temperature *= self.config.temperature_decay

            # Update curvature if annealing is enabled
            self._update_curvature(gen + 1)

        # Final evaluation - actually score the final population
        cross_chain = compute_cross_chain_summary(chains)
        scores = []
        for i, chain in enumerate(chains):
            if chain.score == 0.0:  # Only evaluate if not already scored
                cached_score = None
                cache_key = None
                if self.config.score_cache:
                    cache_key = self._score_cache_key(chain.latent, self.config.score_cache_precision)
                    cached_score = score_cache.get(cache_key)

                if cached_score is None:
                    context = trackers[min(i, len(trackers) - 1)].get_context(chain.latent, cross_chain)
                    verdict = self.judge_panel.evaluate(chain.latent, context)
                    chain.score = verdict.score
                    if cache_key is not None:
                        score_cache[cache_key] = chain.score
                    self.total_evaluations += 1
                else:
                    chain.score = cached_score
            scores.append(chain.score)

        # Update best if final evaluation found something better
        final_best_idx = max(range(len(scores)), key=lambda i: scores[i])
        if scores[final_best_idx] > best_score:
            best_score = scores[final_best_idx]
            best_latent = chains[final_best_idx].latent.clone()

        qd_size, qd_stats = self._get_qd_stats()
        return EvolutionResult(
            best_latent=best_latent,
            best_score=best_score,
            survivors=self._get_top_k(chains, scores, self._current_survivors),
            generations=self.config.generations,
            total_evaluations=self.total_evaluations,
            history=history,
            converged=False,
            stop_reason="max_generations",
            qd_archive_size=qd_size,
            qd_archive_stats=qd_stats,
        )

    def _get_qd_stats(self) -> tuple[int, dict]:
        """Get QD archive statistics if manager is available."""
        if self.qd_manager is not None:
            stats = self.qd_manager.get_archive_statistics()
            return stats.get("size", 0), stats
        return 0, {}

    def _initialize_population(self, seed: Tensor) -> List[ChainState]:
        """Initialize the population from a seed vector with high diversity."""
        chains = []
        # Use initial_diversity multiplier for much more varied starting population
        init_noise_scale = self.config.temperature * self.config.initial_diversity

        if self._use_hyperbolic:
            hyp = self._get_hyperbolic()
            # Map seed to hyperbolic space
            seed_hyp = hyp.expmap0(
                seed * self.geometry_config.tangent_scale,
                self._current_curvature,
            )
            seed_hyp = hyp.project_to_ball(seed_hyp, self._current_curvature, self.geometry_config.max_norm)

            for _ in range(self.config.chains):
                # Generate noise in tangent space and map to ball
                tangent_noise = torch.randn_like(seed) * init_noise_scale * self.geometry_config.mutation_noise_scale
                # Apply noise via expmap at seed position
                latent = hyp.expmap(
                    tangent_noise.unsqueeze(0),
                    seed_hyp.unsqueeze(0),
                    self._current_curvature,
                ).squeeze(0)
                latent = hyp.project_to_ball(latent, self._current_curvature, self.geometry_config.max_norm)
                chains.append(ChainState(latent=latent))
        else:
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

        n = len(chains)

        if self._use_hyperbolic:
            # Use hyperbolic distance for diversity computation
            hyp = self._get_hyperbolic()
            latents = [c.latent for c in chains]

            # Compute pairwise hyperbolic distances
            distances = torch.zeros(n, n, device=chains[0].latent.device)
            for i in range(n):
                for j in range(i + 1, n):
                    d = hyp.hyperbolic_distance(
                        latents[i].squeeze(),
                        latents[j].squeeze(),
                        self._current_curvature,
                    )
                    distances[i, j] = d
                    distances[j, i] = d

            # Average distance to other chains (higher = more diverse)
            mask = 1.0 - torch.eye(n, device=distances.device)
            avg_distance = (distances * mask).sum(dim=1) / (n - 1)

            # Normalize to [0, 1] range approximately
            max_dist = avg_distance.max().item() + 1e-8
            diversity_bonus = (avg_distance / max_dist).tolist()
        else:
            # Stack all latents for efficient computation
            latents = torch.stack([c.latent.flatten() for c in chains])

            # Compute pairwise cosine similarities
            norms = latents.norm(dim=1, keepdim=True).clamp(min=1e-8)
            normalized = latents / norms
            similarities = torch.mm(normalized, normalized.t())

            # For each chain, compute average similarity to OTHER chains
            # (exclude self-similarity on diagonal)
            mask = 1.0 - torch.eye(n, device=latents.device)
            avg_similarity = (similarities * mask).sum(dim=1) / (n - 1)

            # Diversity bonus = 1 - avg_similarity (more different = higher bonus)
            diversity_bonus = (1.0 - avg_similarity).tolist()

        # Combine: final_score = raw_score + diversity_weight * diversity_bonus
        final_scores = []
        for raw, bonus in zip(raw_scores, diversity_bonus):
            final_scores.append(raw + self.config.diversity_weight * bonus)

        return final_scores

    def _score_cache_key(self, latent: Tensor, precision: int) -> bytes:
        """Build a quantized cache key for scorer reuse."""
        scale = float(10**precision)
        quantized = torch.round(latent.detach().float().flatten() * scale).to(torch.int32)
        return quantized.cpu().numpy().tobytes()

    def _update_survivor_budget(
        self,
        generations_without_improvement: int,
        current_population: int,
        generation: int,
    ) -> int:
        """
        Adapt survivor budget to reduce compute during plateaus.

        When progress stalls, shrink survivor count to save evaluations.
        When progress resumes, restore survivors gradually up to baseline.
        """
        cfg = self.config.selection
        baseline = max(1, min(cfg.survivors, current_population))

        if not cfg.adaptive_survivors:
            self._current_survivors = baseline
            return self._current_survivors

        min_survivors = max(1, min(cfg.min_survivors, baseline))
        self._current_survivors = max(min_survivors, min(self._current_survivors, current_population))

        # Improvement phase: slowly restore exploration budget.
        if generations_without_improvement == 0:
            if self._current_survivors < baseline:
                self._current_survivors = min(baseline, self._current_survivors + 1)
            return self._current_survivors

        # Plateau phase: decay survivor budget every configured patience window.
        should_decay = generations_without_improvement % cfg.survivor_decay_patience == 0
        if should_decay:
            decayed = int(round(self._current_survivors * cfg.survivor_decay))
            decayed = max(min_survivors, min(decayed, current_population))
            if decayed < self._current_survivors:
                log_event(
                    "SURVIVOR_DECAY",
                    level=LogLevel.VERBOSE,
                    generation=generation,
                    old=self._current_survivors,
                    new=decayed,
                )
                self._current_survivors = decayed

        return self._current_survivors

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

    def _merge_similar_with_indices(
        self,
        latents: List[Tensor],
        threshold: float,
        protected_indices: Set[int] | None = None,
    ) -> List[int]:
        """
        Find which latents to keep after merging similar ones.

        Returns indices of representative latents (first one in each similar group).
        This allows preserving parent tracking through the merge operation.

        Args:
            latents: List of latent tensors to potentially merge
            threshold: Cosine similarity threshold for merging (or hyperbolic distance threshold)
            protected_indices: Indices that should never be merged away (diversity chains)
        """
        if len(latents) < 2:
            return list(range(len(latents)))

        protected = protected_indices or set()
        keep_indices = []
        used = set()

        if self._use_hyperbolic:
            # Use hyperbolic distance for similarity check
            hyp = self._get_hyperbolic()
            # In hyperbolic mode, threshold is a distance (lower = more similar)
            # We use geometry_config.merge_threshold instead of cosine threshold
            hyp_threshold = self.geometry_config.merge_threshold

            for i in range(len(latents)):
                if i in used:
                    continue

                # Check for similar latents and mark them as used
                for j in range(i + 1, len(latents)):
                    if j in used:
                        continue

                    # Never merge away a protected (diversity) chain
                    if j in protected:
                        continue

                    h_dist = hyp.hyperbolic_distance(
                        latents[i].squeeze(),
                        latents[j].squeeze(),
                        self._current_curvature,
                    ).item()

                    # Merge if hyperbolic distance is below threshold
                    if h_dist < hyp_threshold:
                        used.add(j)

                # Keep the first one from each similar group
                keep_indices.append(i)
                used.add(i)
        else:
            for i in range(len(latents)):
                if i in used:
                    continue

                # Check for similar latents and mark them as used
                for j in range(i + 1, len(latents)):
                    if j in used:
                        continue

                    # Never merge away a protected (diversity) chain
                    if j in protected:
                        continue

                    cos_sim = torch.nn.functional.cosine_similarity(
                        latents[i].flatten().unsqueeze(0).float(),
                        latents[j].flatten().unsqueeze(0).float(),
                    ).item()

                    if cos_sim > threshold:
                        used.add(j)

                # Keep the first one from each similar group
                keep_indices.append(i)
                used.add(i)

        # Ensure all protected indices are included
        for p in protected:
            if p not in keep_indices and p < len(latents):
                keep_indices.append(p)

        return sorted(keep_indices)

    def reset(self) -> None:
        """Reset the evolution loop state."""
        self.current_temperature = self.config.temperature
        self.total_evaluations = 0
        self._current_survivors = max(1, min(self.config.selection.survivors, self.config.chains))
        if isinstance(self.mutation, (AdaptiveMutation, HyperbolicAdaptiveMutation)):
            self.mutation.reset()
        if self.qd_manager is not None:
            self.qd_manager.reset()
        if self.autopoietic_panel is not None:
            self.autopoietic_panel.reset()
        # Reset curvature to initial value
        if self._use_hyperbolic and self.geometry_config.anneal_curvature:
            self._current_curvature = self.geometry_config.initial_curvature
            self._update_curvature(0)

    def _get_hyperbolic(self):
        """Lazy load hyperbolic module."""
        if self._hyperbolic is None:
            from latent_reasoning.utils import hyperbolic as hyp
            self._hyperbolic = hyp
        return self._hyperbolic

    def _update_curvature(self, generation: int) -> None:
        """Update curvature for annealing if enabled."""
        if not self._use_hyperbolic or not self.geometry_config.anneal_curvature:
            return

        # Linear interpolation from initial to final curvature
        total_gens = self.geometry_config.anneal_generations
        progress = min(generation / total_gens, 1.0)
        self._current_curvature = (
            self.geometry_config.initial_curvature +
            progress * (self.geometry_config.final_curvature - self.geometry_config.initial_curvature)
        )

        # Update strategies with new curvature
        if hasattr(self.mutation, 'update_curvature'):
            self.mutation.update_curvature(self._current_curvature)
        if hasattr(self.crossover, 'update_curvature'):
            self.crossover.update_curvature(self._current_curvature)

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
        noise_scale = getattr(config, 'noise_scale', 0.5)
        if strategy == "gaussian":
            return {"noise_scale": noise_scale}
        elif strategy == "directed":
            return {"trust": config.trust, "noise_scale": noise_scale}
        elif strategy == "adaptive":
            return {"base_trust": config.trust, "noise_scale": noise_scale}
        return {}
