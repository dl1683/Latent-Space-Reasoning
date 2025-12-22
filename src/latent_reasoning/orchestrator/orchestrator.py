"""Main orchestrator for coordinating the full reasoning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
from torch import Tensor

from latent_reasoning.config import Config
from latent_reasoning.core.encoder import Encoder, LLMEncoder
from latent_reasoning.core.judge import ScorerJudge, ModifierJudge, create_scorer_from_config
from latent_reasoning.core.panel import JudgePanel
from latent_reasoning.core.chain import ChainState
from latent_reasoning.evolution.loop import EvolutionLoop, EvolutionResult
from latent_reasoning.orchestrator.budget import ComputeBudget
from latent_reasoning.orchestrator.checkpoint import CheckpointManager
from latent_reasoning.utils.logging import log_event, print_header, print_result, LogLevel, set_verbosity


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

        # Initialize evolution loop
        self.evolution_loop = EvolutionLoop(
            judge_panel=self.judge_panel,
            config=config.evolution,
        )

    def _create_encoder(self) -> Encoder:
        """Create encoder from config."""
        return LLMEncoder(
            model_name=self.config.encoder.model,
            extraction_layer=self.config.encoder.layer,
            pooling=self.config.encoder.pooling,
            device_preference=self.config.encoder.device,
            quantization=self.config.encoder.quantization,
        )

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
        print_header("Latent Space Reasoning Engine")

        log_event("START", query=query[:50] + "..." if len(query) > 50 else query)

        # Start budget timer
        self.budget.start()

        # Encode query
        log_event("ENCODE", level=LogLevel.VERBOSE)
        seed = self.encoder.encode(query)
        log_event(
            "ENCODED",
            level=LogLevel.VERBOSE,
            shape=tuple(seed.shape),
            norm=f"{seed.norm().item():.2f}",
        )

        # Set scorer reference to query latent (same latent space for meaningful scoring)
        for scorer in self.judge_panel.scorers:
            scorer.set_reference(embedding=seed)

        # Run evolution
        log_event("EVOLVE", level=LogLevel.NORMAL)
        evolution_result = self.evolution_loop.run(
            seed=seed,
            max_evaluations=self.budget.max_evaluations - self.budget.evaluations_used,
        )

        # Update budget
        self.budget.evaluations_used += evolution_result.total_evaluations
        self.budget.generations_used = evolution_result.generations

        # Checkpoint final state
        self.checkpoint_manager.save_checkpoint(
            chains=evolution_result.survivors,
            generation=evolution_result.generations,
            best_latent=evolution_result.best_latent,
            best_score=evolution_result.best_score,
        )

        # Decode final latent (pass query for context)
        log_event("DECODE", level=LogLevel.VERBOSE)
        decode_latent = self._select_decode_latent(evolution_result)
        decoded_outputs = [
            self.encoder.decode(
                decode_latent,
                query=query,
                max_new_tokens=self.config.synthesis.max_tokens,
                temperature=self.config.synthesis.temperature,
            )
        ]

        # Log completion
        log_event(
            "DONE",
            level=LogLevel.NORMAL,
            score=f"{evolution_result.best_score:.3f}",
            generations=evolution_result.generations,
            reason=evolution_result.stop_reason,
        )

        result = OrchestrationResult(
            final_latent=decode_latent,
            decoded_outputs=decoded_outputs,
            best_score=evolution_result.best_score,
            survivors=evolution_result.survivors,
            generations=evolution_result.generations,
            total_evaluations=evolution_result.total_evaluations,
            stop_reason=evolution_result.stop_reason,
            evolution_history=evolution_result.history,
        )

        # Print result
        if decoded_outputs:
            print_result(
                decoded_outputs[0],
                evolution_result.best_score,
                generations=evolution_result.generations,
                evaluations=evolution_result.total_evaluations,
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
        # For baseline, we just encode and decode directly
        # without any evolution
        seed = self.encoder.encode(query)
        return self.encoder.decode(
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
        # Run baseline
        baseline_output = self.run_baseline(query)

        # Run latent reasoning
        result = self.run(query)

        return {
            "query": query,
            "baseline": baseline_output,
            "latent_reasoning": result.decoded_outputs[0] if result.decoded_outputs else "",
            "latent_score": result.best_score,
            "generations": result.generations,
            "evaluations": result.total_evaluations,
        }

    def reset(self) -> None:
        """Reset the orchestrator state."""
        self.budget.reset()
        self.checkpoint_manager.clear()
        self.evolution_loop.reset()
