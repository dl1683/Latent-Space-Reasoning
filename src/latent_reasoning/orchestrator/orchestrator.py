"""Main orchestrator for coordinating the full reasoning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import List

import torch
from torch import Tensor

from latent_reasoning.config import Config
from latent_reasoning.core.encoder import Encoder, LLMEncoder
from latent_reasoning.core.judge import ScorerJudge, ModifierJudge, create_scorer_from_config
from latent_reasoning.core.panel import JudgePanel
from latent_reasoning.core.chain import ChainState
from latent_reasoning.core.autopoietic import create_autopoietic_panel, AutopoieticPanel
from latent_reasoning.evolution.loop import EvolutionLoop, EvolutionResult
from latent_reasoning.grammar import GrammarEvolutionLoop, GrammarEvolutionResult, FractalGrammar
from latent_reasoning.qd import create_qd_manager, QDManager
from latent_reasoning.orchestrator.budget import ComputeBudget
from latent_reasoning.orchestrator.checkpoint import CheckpointManager
from latent_reasoning.utils.logging import log_event, print_header, print_result, LogLevel, set_verbosity


def _tag_history(history: list, source: str) -> list:
    """
    Tag history entries with a source identifier.

    Handles cases where history entries might not be dicts (defensive coding).
    """
    tagged = []
    for h in history:
        if isinstance(h, dict):
            tagged.append({**h, "source": source})
        else:
            # Fallback for non-dict entries (should not happen, but defensive)
            tagged.append({"data": str(h), "source": source})
    return tagged


@dataclass
class OrchestrationResult:
    """Result of the orchestration process."""

    # Final outputs
    final_latent: Tensor
    decoded_outputs: List[str]
    best_score: float

    # Survivors
    survivors: List[ChainState]

    # Stats
    generations: int
    total_evaluations: int
    stop_reason: str

    # History
    evolution_history: List[dict] = field(default_factory=list)

    # Stage timings (seconds)
    encode_duration_s: float = 0.0
    evolution_duration_s: float = 0.0
    decode_duration_s: float = 0.0
    total_run_duration_s: float = 0.0


class Orchestrator:
    """
    Main orchestrator for the complete latent space reasoning pipeline.

    The Orchestrator is the central coordinator that manages the entire reasoning
    process from start to finish. It integrates all components of the system and
    handles the complex workflow of latent space optimization.

    Complete Pipeline:
    1. **Initialization**: Set up encoder, judges, evolution loop, and budget
    2. **Encoding**: Convert input query to latent vector representation
    3. **Evolution**: Optimize latent through evolutionary algorithms
    4. **Budget Management**: Track and enforce computational limits
    5. **Checkpointing**: Save/restore state for fault tolerance
    6. **Decoding**: Convert optimized latent back to text response
    7. **Result Assembly**: Package results with statistics and metadata

    Key Responsibilities:
    - **Component Integration**: Coordinates encoder, judges, and evolution
    - **Resource Management**: Enforces compute budgets and time limits
    - **Error Handling**: Graceful degradation and recovery from failures
    - **Progress Tracking**: Detailed logging and statistics collection
    - **State Management**: Checkpointing for long-running optimizations

    The orchestrator abstracts away the complexity of the multi-component system,
    providing a clean interface for running complete reasoning workflows while
    handling all the coordination, error recovery, and resource management
    behind the scenes.

    Example:
        >>> from latent_reasoning.config import Config
        >>> config = Config()  # Use defaults
        >>> orchestrator = Orchestrator(config)
        >>> result = orchestrator.run("How to implement caching?")
        >>> print(f"Best response: {result.decoded_outputs[0]}")
        >>> print(f"Confidence: {result.best_score:.3f}")
    """

    def __init__(
        self,
        config: Config,
        encoder: Encoder | None = None,
        judge_panel: JudgePanel | None = None,
    ):
        """
        Initialize the orchestrator with configuration and optional components.

        Sets up all components needed for the reasoning pipeline. Components can
        be provided explicitly for custom setups, or will be created automatically
        from the configuration for standard usage.

        Args:
            config: Complete configuration object containing settings for all
                components (encoder, judges, evolution, budget, output). This
                defines the behavior of the entire reasoning system.
            encoder: Pre-initialized encoder instance. If None, creates an
                LLMEncoder from config.encoder settings. Useful for sharing
                encoders across multiple orchestrators or custom implementations.
            judge_panel: Pre-initialized judge panel. If None, creates a panel
                from config.judges settings. Allows custom judge configurations
                or sharing panels across runs.

        Example:
            Standard initialization:
            >>> config = Config()  # Use defaults
            >>> orchestrator = Orchestrator(config)

            Custom encoder:
            >>> encoder = LLMEncoder("Qwen/Qwen3-4B", device_preference="cuda:1")
            >>> orchestrator = Orchestrator(config, encoder=encoder)

            Shared components:
            >>> # Share expensive components across multiple runs
            >>> shared_encoder = LLMEncoder("Qwen/Qwen3-4B")
            >>> shared_panel = create_judge_panel(config.judges)
            >>> orchestrator1 = Orchestrator(config1, shared_encoder, shared_panel)
            >>> orchestrator2 = Orchestrator(config2, shared_encoder, shared_panel)

        Note:
            - Component creation can be expensive (model loading), so sharing
              is recommended for multiple runs
            - All components must be compatible with the configuration
            - Verbosity is set globally based on config.output.verbosity
        """
        self.config = config

        # Set verbosity
        set_verbosity(config.output.verbosity)

        # Initialize encoder
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = self._create_encoder()
        self._baseline_encoder: Encoder | None = None

        # Initialize judge panel
        if judge_panel is not None:
            self.judge_panel = judge_panel
        else:
            self.judge_panel = self._create_judge_panel()

        # Initialize budget
        self.budget = ComputeBudget(
            max_generations=config.evolution.generations,
            max_evaluations=config.budget.max_evaluations,
            max_time=config.budget.max_time,
        )

        # Initialize checkpoint manager
        checkpoint_dir = config.output.history_path if config.output.save_history else None
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)

        # Initialize QD manager if enabled
        self.qd_manager: QDManager | None = None
        if config.qd.enabled:
            self.qd_manager = create_qd_manager(
                config=config.qd,
                latent_dim=self.encoder.latent_dim,
            )

        # Initialize autopoietic panel if enabled
        self.autopoietic_panel: AutopoieticPanel | None = None
        if config.autopoietic.enabled:
            # Create internal scorer callable from judge panel
            def internal_scorer(latent: Tensor) -> float:
                # Use the first scorer in the panel
                if self.judge_panel.scorers:
                    from latent_reasoning.core.judge import ScoreResult
                    result = self.judge_panel.scorers[0].score(latent)
                    return result.overall if isinstance(result, ScoreResult) else float(result)
                return 0.5  # Neutral score if no scorers

            # Create decoder callable from encoder
            def decoder(latent: Tensor, query: str) -> str:
                return self.encoder.decode(
                    latent,
                    query=query,
                    max_new_tokens=config.synthesis.max_tokens,
                    temperature=config.synthesis.temperature,
                )

            self.autopoietic_panel = create_autopoietic_panel(
                config=config.autopoietic,
                internal_scorer=internal_scorer,
                decoder=decoder,
                device=config.encoder.device,
            )

        # Initialize evolution loop
        self.evolution_loop = EvolutionLoop(
            judge_panel=self.judge_panel,
            config=config.evolution,
            qd_manager=self.qd_manager,
            autopoietic_panel=self.autopoietic_panel,
            geometry_config=config.geometry,
        )

        # Initialize grammar evolution loop if enabled
        self.grammar_loop: GrammarEvolutionLoop | None = None
        if config.grammar.enabled:
            # Use encoder's resolved device (handles "auto" -> actual device)
            encoder_device = self.encoder.device if hasattr(self.encoder, 'device') else "cpu"
            self.grammar_loop = GrammarEvolutionLoop.from_config(
                grammar_config=config.grammar,
                latent_dim=self.encoder.latent_dim,
                device=encoder_device,
                qd_manager=self.qd_manager,
            )

    def _create_encoder(self) -> Encoder:
        """Create encoder from config."""
        return LLMEncoder(
            model_name=self.config.encoder.model,
            extraction_layer=self.config.encoder.layer,
            pooling=self.config.encoder.pooling,
            device_preference=self.config.encoder.device,
            quantization=self.config.encoder.quantization,
            latent_dim=self.config.encoder.latent_dim,
        )

    def _create_baseline_encoder(self) -> Encoder:
        """Create baseline encoder without quantization."""
        return LLMEncoder(
            model_name=self.config.encoder.model,
            extraction_layer=self.config.encoder.layer,
            pooling=self.config.encoder.pooling,
            device_preference=self.config.encoder.device,
            quantization="none",
            latent_dim=self.config.encoder.latent_dim,
        )

    def _get_baseline_encoder(self) -> Encoder:
        # If a custom encoder is injected, treat it as authoritative for baseline
        # to avoid forcing model downloads in test/offline environments.
        if not isinstance(self.encoder, LLMEncoder):
            return self.encoder
        if self.config.encoder.quantization == "none":
            return self.encoder
        if self._baseline_encoder is None:
            self._baseline_encoder = self._create_baseline_encoder()
        return self._baseline_encoder

    def _create_judge_panel(self) -> JudgePanel:
        """Create judge panel from config."""
        scorers = []
        for scorer_config in self.config.judges.scorers:
            # Use factory function to handle different scorer types
            scorer = create_scorer_from_config(
                scorer_config,
                device=self.config.encoder.device,
                encoder_latent_dim=self.encoder.latent_dim,
            )
            scorers.append(scorer)

        modifiers = []
        for modifier_config in self.config.judges.modifiers:
            modifier = ModifierJudge(
                model_name=modifier_config.model,
                layers=tuple(modifier_config.layers),
                canonical_dim=self.encoder.latent_dim,
                device_preference=self.config.encoder.device,
                quantization=modifier_config.quantization,
            )
            modifiers.append(modifier)

        return JudgePanel(
            scorers=scorers,
            modifiers=modifiers,
            aggregation=self.config.judges.aggregation,
            calibrate=self.config.judges.calibrate,
        )

    def _combine_latents(self, survivors: List[ChainState]) -> Tensor:
        """Combine top survivor latents into a single representation."""
        if not survivors:
            raise ValueError("No survivors to combine")

        ordered = sorted(survivors, key=lambda s: s.score, reverse=True)
        top = ordered[:self.config.synthesis.max_survivors]

        stacked = torch.stack([s.latent.float() for s in top])
        scores = torch.tensor([s.score for s in top], device=stacked.device, dtype=torch.float32)
        if torch.allclose(scores, scores[0]) or scores.abs().sum().item() == 0.0:
            weights = torch.ones(len(top), device=stacked.device) / len(top)
        else:
            weights = torch.softmax(scores, dim=0)

        # Use Karcher mean for hyperbolic space
        if self.config.geometry.space == "hyperbolic":
            from latent_reasoning.utils import hyperbolic as hyp
            # Flatten latents for Karcher mean
            flat_latents = stacked.view(len(top), -1)
            combined = hyp.karcher_mean(
                flat_latents,
                weights=weights,
                c=self.config.geometry.curvature,
                max_iters=self.config.geometry.barycenter_iterations,
            )
            combined = combined.view_as(top[0].latent)
        else:
            view_shape = [len(top)] + [1] * (stacked.dim() - 1)
            combined = (stacked * weights.view(*view_shape)).sum(dim=0)

        return combined.to(top[0].latent.dtype)

    def _select_decode_latent(self, evolution_result: EvolutionResult) -> Tensor:
        """Select the latent to decode based on synthesis strategy."""
        strategy = self.config.synthesis.decode_strategy
        if strategy == "combined":
            if evolution_result.survivors:
                return self._combine_latents(evolution_result.survivors)
            return evolution_result.best_latent
        if strategy == "best":
            return evolution_result.best_latent
        raise ValueError(f"Unsupported decode strategy: {strategy}")

    def _run_grammar_evolution(
        self,
        query: str,
        seed: Tensor,
    ) -> tuple[Tensor, float, list[dict]]:
        """
        Run grammar-based evolution.

        Args:
            query: Input query for scoring
            seed: Seed latent from encoding

        Returns:
            Tuple of (best_latent, best_score, history)
        """
        if self.grammar_loop is None:
            raise RuntimeError("Grammar loop not initialized")

        # Get scorer from judge panel
        if not self.judge_panel.scorers:
            raise RuntimeError("No scorers available for grammar evolution")

        scorer = self.judge_panel.scorers[0]

        log_event("GRAMMAR_EVOLVE", level=LogLevel.NORMAL)

        # Run grammar evolution
        # Handle verbosity which can be int or str
        verbosity = self.config.output.verbosity
        is_verbose = verbosity >= 2 if isinstance(verbosity, int) else verbosity == "verbose"

        grammar_result = self.grammar_loop.run(
            scorer=scorer,
            query=query,
            num_generations=self.config.evolution.generations,
            seed_latent=seed,
            early_stop_threshold=0.95,
            verbose=is_verbose,
        )

        log_event(
            "GRAMMAR_DONE",
            level=LogLevel.NORMAL,
            score=f"{grammar_result.best_score:.3f}",
            nodes=grammar_result.grammar_stats.num_nodes,
            compression=f"{grammar_result.grammar_stats.compression_ratio:.2f}x",
        )

        return grammar_result.best_latent, grammar_result.best_score, grammar_result.history

    def run(self, query: str) -> OrchestrationResult:
        """
        Execute the complete latent space reasoning pipeline on a query.

        This is the main method that orchestrates the entire reasoning process
        from encoding the input query to producing the final optimized response.
        It coordinates all components and manages the complex workflow.

        Pipeline Steps:
        1. **Budget Initialization**: Start tracking computational resources
        2. **Query Encoding**: Convert input text to latent vector representation
        3. **Judge Setup**: Configure scorers with query reference for evaluation
        4. **Evolution**: Optimize latent through evolutionary algorithms
        5. **Budget Update**: Track resource usage and enforce limits
        6. **Checkpointing**: Save final state for recovery/analysis
        7. **Decoding**: Convert optimized latent(s) back to text responses
        8. **Result Assembly**: Package outputs with statistics and metadata

        Args:
            query: Input query to reason about. Can be any text that benefits
                from structured reasoning:
                - Questions: "How to implement user authentication?"
                - Problems: "Design a scalable microservices architecture"
                - Requests: "Create a plan for database optimization"
                - Scenarios: "Handle high traffic during peak hours"

        Returns:
            OrchestrationResult containing:
            - final_latent: Best latent vector found through evolution
            - decoded_outputs: List of decoded responses (currently one)        
            - best_score: Highest fitness score achieved
            - survivors: Final population of high-quality latent vectors        
            - generations: Number of evolution cycles completed
            - total_evaluations: Total judge evaluations performed
            - stop_reason: Why the evolution process terminated
            - evolution_history: Detailed per-generation statistics

        Example:
            >>> orchestrator = Orchestrator(config)
            >>> result = orchestrator.run("How to implement caching?")
            >>>
            >>> # Access the best response
            >>> best_response = result.decoded_outputs[0]
            >>> print(f"Response: {best_response}")
            >>>
            >>> # Check quality and efficiency
            >>> print(f"Quality score: {result.best_score:.3f}")
            >>> print(f"Generations: {result.generations}")
            >>> print(f"Evaluations: {result.total_evaluations}")
            >>>
            >>> # Analyze evolution progress
            >>> for gen_stats in result.evolution_history:
            ...     print(f"Gen {gen_stats['generation']}: {gen_stats['best_score']:.3f}")

        Raises:
            RuntimeError: If encoding, evolution, or decoding fails
            TimeoutError: If budget time limit is exceeded
            ValueError: If query is empty or invalid

        Note:
            - The method handles all error recovery and resource management     
            - Progress is logged according to configured verbosity level        
            - Checkpoints are saved automatically for fault tolerance
            - Decoding strategy is controlled by synthesis.decode_strategy
        """
        run_start = perf_counter()
        verbosity = self.config.output.verbosity
        if isinstance(verbosity, int):
            render_output = verbosity >= int(LogLevel.NORMAL)
        else:
            render_output = verbosity not in {"silent", "minimal"}
        if render_output:
            print_header("Latent Space Reasoning Engine")

        log_event("START", query=query[:50] + "..." if len(query) > 50 else query)

        # Reset budget counters for per-query isolation, then start timer.
        self.budget.reset()
        self.budget.start()

        # Encode query
        log_event("ENCODE", level=LogLevel.VERBOSE)
        encode_start = perf_counter()
        seed = self.encoder.encode(query)
        encode_duration_s = perf_counter() - encode_start
        log_event(
            "ENCODED",
            level=LogLevel.VERBOSE,
            shape=tuple(seed.shape),
            norm=f"{seed.norm().item():.2f}",
        )

        # Set scorer reference to query latent (same latent space for meaningful scoring)
        for scorer in self.judge_panel.scorers:
            scorer.set_reference(embedding=seed)

        # Set query context for autopoietic panel if enabled
        if self.autopoietic_panel is not None:
            self.autopoietic_panel.set_query(query)

        # Determine evolution mode
        use_grammar = self.config.grammar.enabled and self.grammar_loop is not None
        use_standard = not use_grammar or self.config.qd.enabled  # Hybrid if both enabled

        # Storage for results
        best_latent = seed
        best_score = 0.0
        total_evaluations = 0
        generations = 0
        stop_reason = "none"
        evolution_history = []
        survivors = []
        evolution_duration_s = 0.0

        # Run standard evolution if applicable
        if use_standard:
            log_event("EVOLVE", level=LogLevel.NORMAL)
            evolution_start = perf_counter()
            evolution_result = self.evolution_loop.run(
                seed=seed,
                max_evaluations=self.budget.max_evaluations - self.budget.evaluations_used,
            )
            evolution_duration_s += perf_counter() - evolution_start

            best_latent = evolution_result.best_latent
            best_score = evolution_result.best_score
            total_evaluations = evolution_result.total_evaluations
            generations = evolution_result.generations
            stop_reason = evolution_result.stop_reason
            evolution_history = evolution_result.history
            survivors = evolution_result.survivors

            # Update budget
            self.budget.evaluations_used += evolution_result.total_evaluations
            self.budget.generations_used = evolution_result.generations

        # Run grammar evolution if enabled
        if use_grammar:
            grammar_start = perf_counter()
            grammar_latent, grammar_score, grammar_history = self._run_grammar_evolution(
                query=query,
                seed=seed,
            )
            evolution_duration_s += perf_counter() - grammar_start

            # In hybrid mode, take the better result
            if use_standard:
                log_event(
                    "HYBRID_COMPARE",
                    level=LogLevel.NORMAL,
                    standard_score=f"{best_score:.3f}",
                    grammar_score=f"{grammar_score:.3f}",
                )
                if grammar_score > best_score:
                    log_event("HYBRID_WINNER", level=LogLevel.NORMAL, winner="grammar")
                    best_latent = grammar_latent
                    best_score = grammar_score
                    stop_reason = "grammar_better"
                    # Merge histories
                    evolution_history = evolution_history + _tag_history(grammar_history, "grammar")
                else:
                    log_event("HYBRID_WINNER", level=LogLevel.NORMAL, winner="standard")
                    evolution_history = _tag_history(evolution_history, "standard") + _tag_history(grammar_history, "grammar")
            else:
                # Grammar only mode
                best_latent = grammar_latent
                best_score = grammar_score
                evolution_history = grammar_history
                generations = len(grammar_history)
                stop_reason = "grammar_complete"

        # Checkpoint final state
        self.checkpoint_manager.save_checkpoint(
            chains=survivors,
            generation=generations,
            best_latent=best_latent,
            best_score=best_score,
        )

        # Decode final latent (pass query for context)
        log_event("DECODE", level=LogLevel.VERBOSE)
        decode_latent = best_latent
        if survivors and self.config.synthesis.decode_strategy == "combined":
            decode_latent = self._combine_latents(survivors)

        decode_start = perf_counter()
        decoded_outputs = [
            self.encoder.decode(
                decode_latent,
                query=query,
                max_new_tokens=self.config.synthesis.max_tokens,
                temperature=self.config.synthesis.temperature,
            )
        ]
        decode_duration_s = perf_counter() - decode_start
        total_run_duration_s = perf_counter() - run_start

        # Log completion
        log_event(
            "DONE",
            level=LogLevel.NORMAL,
            score=f"{best_score:.3f}",
            generations=generations,
            reason=stop_reason,
        )

        result = OrchestrationResult(
            final_latent=decode_latent,
            decoded_outputs=decoded_outputs,
            best_score=best_score,
            survivors=survivors,
            generations=generations,
            total_evaluations=total_evaluations,
            stop_reason=stop_reason,
            evolution_history=evolution_history,
            encode_duration_s=encode_duration_s,
            evolution_duration_s=evolution_duration_s,
            decode_duration_s=decode_duration_s,
            total_run_duration_s=total_run_duration_s,
        )

        # Print result
        if decoded_outputs and render_output:
            print_result(
                decoded_outputs[0],
                best_score,
                generations=generations,
                evaluations=total_evaluations,
            )

        return result

    def run_baseline(self, query: str) -> str:
        """
        Run baseline generation without latent space reasoning.

        Args:
            query: Input query

        Returns:
            Baseline generated output
        """
        baseline_encoder = self._get_baseline_encoder()
        if hasattr(baseline_encoder, "generate_baseline"):
            return baseline_encoder.generate_baseline(
                query=query,
                max_new_tokens=self.config.synthesis.max_tokens,
                temperature=self.config.synthesis.temperature,
            )
        # Fallback: encode/decode without evolution if encoder lacks direct baseline
        seed = baseline_encoder.encode(query)
        return baseline_encoder.decode(
            seed,
            query=query,
            max_new_tokens=self.config.synthesis.max_tokens,
            temperature=self.config.synthesis.temperature,
        )

    def compare(self, query: str) -> dict:
        """
        Compare baseline vs latent space reasoning.

        Args:
            query: Input query

        Returns:
            Dict with both outputs for comparison
        """
        compare_start = perf_counter()

        # Run baseline
        baseline_start = perf_counter()
        baseline_output = self.run_baseline(query)
        baseline_duration_s = perf_counter() - baseline_start

        # Run latent reasoning
        latent_start = perf_counter()
        result = self.run(query)
        latent_duration_s = perf_counter() - latent_start
        total_duration_s = perf_counter() - compare_start

        latency_overhead_ratio = None
        if baseline_duration_s > 0:
            latency_overhead_ratio = latent_duration_s / baseline_duration_s

        return {
            "query": query,
            "baseline": baseline_output,
            "latent_reasoning": result.decoded_outputs[0] if result.decoded_outputs else "",
            "latent_score": result.best_score,
            "generations": result.generations,
            "evaluations": result.total_evaluations,
            "baseline_duration_s": baseline_duration_s,
            "latent_duration_s": latent_duration_s,
            "latent_run_duration_s": result.total_run_duration_s,
            "latent_encode_duration_s": result.encode_duration_s,
            "latent_evolution_duration_s": result.evolution_duration_s,
            "latent_decode_duration_s": result.decode_duration_s,
            "latent_non_evolution_duration_s": max(
                0.0, result.total_run_duration_s - result.evolution_duration_s
            ),
            "total_compare_duration_s": total_duration_s,
            "latency_overhead_ratio": latency_overhead_ratio,
        }

    def reset(self) -> None:
        """Reset the orchestrator state."""
        self.budget.reset()
        self.checkpoint_manager.clear()
        self.evolution_loop.reset()
