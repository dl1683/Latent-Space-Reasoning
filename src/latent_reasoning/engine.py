"""
Main Engine class for the Latent Space Reasoning system.

This module provides the Engine class, which is the primary interface for users
who need more control than the simple reason() function provides. The Engine
allows custom configuration, multiple queries, and access to internal state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Any

from torch import Tensor

from latent_reasoning.config import Config, get_default_config
from latent_reasoning.core.encoder import Encoder, LLMEncoder
from latent_reasoning.core.judge import ScorerJudge, ModifierJudge, create_scorer_from_config
from latent_reasoning.core.panel import JudgePanel
from latent_reasoning.core.chain import ChainState
from latent_reasoning.evolution.loop import EvolutionLoop
from latent_reasoning.evolution.operators import SelectionStrategy, MutationStrategy, CrossoverStrategy
from latent_reasoning.orchestrator.orchestrator import Orchestrator, OrchestrationResult
from latent_reasoning.utils.logging import set_verbosity, LogLevel


@dataclass
class ReasoningResult:
    """
    Result from running latent space reasoning on a query.

    This dataclass contains all the information about a reasoning run,
    including the final output, quality metrics, and evolution statistics.

    Attributes:
        plan: The final generated reasoning plan/response
        confidence: Quality score from 0-1 (higher = better quality)
        survivors: All final candidate solutions from evolution
        all_plans: Text outputs from all surviving candidates
        generations: Number of evolution generations that ran
        evaluations: Total number of latent vectors evaluated
        stop_reason: Why evolution stopped ("max_generations", "converged", etc.)
        history: Detailed evolution progress (generation-by-generation stats)
    """

    # Main outputs - what users care about most
    plan: str                                    # The best generated response
    confidence: float                            # Quality score 0-1

    # Detailed results - for analysis and debugging
    survivors: List[ChainState] = field(default_factory=list)      # All final candidates
    all_plans: List[str] = field(default_factory=list)             # Decoded outputs (usually one)

    # Evolution metadata - for understanding performance
    generations: int = 0                         # How many evolution cycles ran
    evaluations: int = 0                         # Total latent vectors scored
    stop_reason: str = ""                        # Why evolution stopped

    # Detailed history - for research and debugging
    history: List[dict] = field(default_factory=list)              # Per-generation statistics


class Engine:
    """
    Main engine for latent space reasoning with full customization options.

    The Engine class is the primary interface for the latent space reasoning system,
    providing complete control over the reasoning process. It's designed for users
    who need more flexibility than the simple reason() function, allowing deep
    customization of models, algorithms, and parameters.

    Architecture:
    - **High-level Interface**: Abstracts away orchestrator complexity
    - **Component Management**: Handles encoder, judges, and evolution setup
    - **Configuration Integration**: Seamlessly works with YAML configs
    - **Performance Optimization**: Reuses expensive components across runs
    - **Research-friendly**: Provides access to internal state and statistics

    Key capabilities:
    - **Multi-query Processing**: Run multiple queries with shared components
    - **Baseline Comparison**: Built-in A/B testing against direct generation
    - **Evolution Analysis**: Detailed statistics and history tracking
    - **Custom Strategies**: Plug in custom selection, mutation, crossover algorithms
    - **Model Flexibility**: Support for any HuggingFace transformer model
    - **Research Access**: Direct encoding/decoding for experimentation

    Use Cases:
    - **Production Systems**: Consistent high-quality reasoning at scale
    - **Research**: Experimenting with different evolutionary strategies
    - **Benchmarking**: Comparing multiple models and configurations
    - **Integration**: Embedding reasoning into larger applications

    Examples:
        Basic usage:
        >>> engine = Engine(encoder="Qwen/Qwen3-4B")
        >>> result = engine.run("How to implement caching?")
        >>> print(result.plan)
        >>> print(f"Confidence: {result.confidence:.3f}")

        Custom configuration:
        >>> from latent_reasoning.config import Config
        >>> config = Config()
        >>> config.evolution.chains = 10
        >>> config.evolution.generations = 20
        >>> engine = Engine(config=config)

        Multiple queries with shared components:
        >>> engine = Engine(encoder="Qwen/Qwen3-1.7B", verbosity="minimal")
        >>> queries = ["How to cache data?", "Design an API", "Optimize queries"]
        >>> for query in queries:
        ...     result = engine.run(query)
        ...     print(f"{query}: {result.confidence:.3f}")

        Comparison mode:
        >>> comparison = engine.compare("How to implement authentication?")
        >>> print("Baseline:", comparison["baseline"])
        >>> print("Latent Reasoning:", comparison["latent_reasoning"])
        >>> print("Improvement:", comparison["improvement_factor"])
    """

    def __init__(
        self,
        config: Config | None = None,
        encoder: Encoder | str | None = None,
        scorers: List[ScorerJudge | str] | None = None,
        modifiers: List[ModifierJudge | str] | None = None,
        selection_strategy: SelectionStrategy | None = None,
        mutation_strategy: MutationStrategy | None = None,
        crossover_strategy: CrossoverStrategy | None = None,
        verbosity: str | LogLevel = "normal",
    ):
        """
        Initialize the reasoning engine with custom settings.

        Args:
            config: Complete configuration object (uses defaults if None)
            encoder: Model name (e.g., "Qwen/Qwen3-4B") or Encoder instance
            scorers: List of scorer models/instances for quality evaluation
            modifiers: List of modifier models for suggesting improvements
            selection_strategy: Custom selection algorithm (elitist, tournament, etc.)
            mutation_strategy: Custom mutation algorithm (gaussian, directed, adaptive)
            crossover_strategy: Custom crossover algorithm (weighted, interpolation, etc.)
            verbosity: Logging level ("silent", "minimal", "normal", "verbose", "debug")

        Note:
            If you provide both config and individual parameters (like encoder),
            the individual parameters will override the config settings.
        """
        # Use provided config or create default
        self.config = config or get_default_config()

        # Set verbosity
        if isinstance(verbosity, str):
            self.config.output.verbosity = verbosity
        set_verbosity(self.config.output.verbosity)

        # Override config with explicit arguments
        if encoder is not None:
            if isinstance(encoder, str):
                self.config.encoder.model = encoder
            else:
                self._encoder = encoder

        # Build or store components
        self._encoder: Encoder | None = encoder if isinstance(encoder, Encoder) else None
        self._scorers: List[ScorerJudge] | None = None
        self._modifiers: List[ModifierJudge] | None = None

        if scorers is not None:
            self._scorers = []
            for scorer in scorers:
                if isinstance(scorer, str):
                    self.config.judges.scorers.append(
                        type(self.config.judges.scorers[0])(model=scorer)
                        if self.config.judges.scorers
                        else type("ScorerConfig", (), {"model": scorer})()
                    )
                else:
                    self._scorers.append(scorer)

        if modifiers is not None:
            self._modifiers = []
            for modifier in modifiers:
                if isinstance(modifier, str):
                    self.config.judges.modifiers.append(
                        type(self.config.judges.modifiers[0])(model=modifier)
                        if self.config.judges.modifiers
                        else type("ModifierConfig", (), {"model": modifier})()
                    )
                else:
                    self._modifiers.append(modifier)

        # Store custom strategies
        self._selection_strategy = selection_strategy
        self._mutation_strategy = mutation_strategy
        self._crossover_strategy = crossover_strategy

        # Lazy-initialized orchestrator
        self._orchestrator: Orchestrator | None = None

    def _get_orchestrator(self) -> Orchestrator:
        """Get or create the orchestrator."""
        if self._orchestrator is None:
            # Create encoder if not provided
            encoder = self._encoder
            if encoder is None:
                encoder = LLMEncoder(
                    model_name=self.config.encoder.model,
                    extraction_layer=self.config.encoder.layer,
                    pooling=self.config.encoder.pooling,
                    device_preference=self.config.encoder.device,
                    quantization=self.config.encoder.quantization,
                )

            # Create judge panel
            scorers = self._scorers or []
            if not scorers:
                for scorer_config in self.config.judges.scorers:
                    scorer = create_scorer_from_config(
                        scorer_config,
                        device=self.config.encoder.device,
                        encoder_latent_dim=encoder.latent_dim,
                    )
                    scorers.append(scorer)

            modifiers = self._modifiers or []
            if not modifiers:
                for modifier_config in self.config.judges.modifiers:
                    modifiers.append(ModifierJudge(
                        model_name=modifier_config.model,
                        layers=tuple(modifier_config.layers),
                        canonical_dim=encoder.latent_dim,
                        device_preference=self.config.encoder.device,
                        quantization=modifier_config.quantization,
                    ))

            judge_panel = JudgePanel(
                scorers=scorers,
                modifiers=modifiers,
                aggregation=self.config.judges.aggregation,
                calibrate=self.config.judges.calibrate,
            )

            self._orchestrator = Orchestrator(
                config=self.config,
                encoder=encoder,
                judge_panel=judge_panel,
            )

            # Override strategies if provided
            if self._selection_strategy:
                self._orchestrator.evolution_loop.selection = self._selection_strategy
            if self._mutation_strategy:
                self._orchestrator.evolution_loop.mutation = self._mutation_strategy
            if self._crossover_strategy:
                self._orchestrator.evolution_loop.crossover = self._crossover_strategy

        return self._orchestrator

    def run(self, query: str) -> ReasoningResult:
        """
        Run latent space reasoning on a query.

        This is the main method for generating high-quality responses using
        evolutionary optimization in latent space. The process:
        1. Encodes your query into a latent vector
        2. Evolves the latent through multiple generations
        3. Scores candidates using trained neural networks
        4. Decodes the best result back to text

        Args:
            query: The question or problem to reason about

        Returns:
            ReasoningResult containing the generated plan, confidence score,
            and detailed evolution statistics

        Examples:
            >>> engine = Engine()
            >>> result = engine.run("How to implement user authentication?")
            >>> print(result.plan)
            >>> print(f"Quality: {result.confidence:.2f}")
            >>> print(f"Evolved over {result.generations} generations")

            >>> # Check if result is high quality
            >>> if result.confidence > 0.8:
            ...     print("High quality result!")
            >>> else:
            ...     print("Consider trying with more generations or a larger model")
        """
        orchestrator = self._get_orchestrator()
        result = orchestrator.run(query)

        return ReasoningResult(
            plan=result.decoded_outputs[0] if result.decoded_outputs else "",
            confidence=result.best_score,
            survivors=result.survivors,
            all_plans=result.decoded_outputs,
            generations=result.generations,
            evaluations=result.total_evaluations,
            stop_reason=result.stop_reason,
            history=result.evolution_history,
        )

    def run_baseline(self, query: str) -> str:
        """
        Run baseline (direct) generation without latent space evolution.

        This generates a response using standard LLM generation for comparison
        purposes. Useful for A/B testing to see the improvement from evolution.

        Args:
            query: The query to process with baseline generation

        Returns:
            Generated text using direct LLM generation (no evolution)

        Example:
            >>> engine = Engine()
            >>> baseline = engine.run_baseline("How to implement caching?")
            >>> evolved = engine.run("How to implement caching?")
            >>> print("Baseline:", baseline)
            >>> print("Evolved:", evolved.plan)
        """
        orchestrator = self._get_orchestrator()
        return orchestrator.run_baseline(query)

    def compare(self, query: str) -> dict:
        """
        Compare latent reasoning vs baseline generation side-by-side.

        This is the best way to evaluate the effectiveness of latent space
        reasoning. It runs both methods on the same query and returns both
        outputs along with quality metrics.

        Args:
            query: The query to compare both methods on

        Returns:
            Dictionary containing:
                - "baseline": Direct LLM generation output
                - "latent_reasoning": Evolved latent space output
                - "latent_score": Quality score for latent reasoning result
                - "generations": Number of evolution generations
                - "evaluations": Total evaluations performed

        Example:
            >>> engine = Engine(encoder="Qwen/Qwen3-4B")
            >>> result = engine.compare("How to optimize database queries?")
            >>> print("=== Baseline ===")
            >>> print(result["baseline"])
            >>> print("\\n=== Latent Reasoning ===")
            >>> print(result["latent_reasoning"])
            >>> print(f"\\nImprovement score: {result['latent_score']:.3f}")
        """
        orchestrator = self._get_orchestrator()
        return orchestrator.compare(query)

    def encode(self, text: str) -> Tensor:
        """
        Encode text into the model's latent space representation.

        This method is primarily for research and debugging. It converts
        text into the latent vector that the evolution process operates on.

        Args:
            text: Text to encode into latent space

        Returns:
            Latent vector (torch.Tensor) representing the text

        Example:
            >>> engine = Engine()
            >>> latent = engine.encode("How to implement authentication?")
            >>> print(f"Latent shape: {latent.shape}")
            >>> print(f"Latent norm: {latent.norm().item():.3f}")
        """
        orchestrator = self._get_orchestrator()
        return orchestrator.encoder.encode(text)

    def decode(self, latent: Tensor, query: str | None = None) -> str:
        """
        Decode a latent vector back to text.

        This method is primarily for research and debugging. It converts
        a latent vector back into text using the model's generation process.

        Args:
            latent: Latent vector to decode back to text
            query: Optional original query for context

        Returns:
            Generated text from the latent vector

        Example:
            >>> engine = Engine()
            >>> # Encode then decode (should be similar to original)
            >>> original = "How to implement caching?"
            >>> latent = engine.encode(original)
            >>> decoded = engine.decode(latent, query=original)
            >>> print("Decoded:", decoded)
        """
        orchestrator = self._get_orchestrator()
        return orchestrator.encoder.decode(latent, query=query)

    def reset(self) -> None:
        """
        Reset the engine state and clear any cached components.

        This clears the internal orchestrator and forces reinitialization
        on the next method call. Useful when changing configuration or
        clearing cached state.
        """
        if self._orchestrator is not None:
            if hasattr(self._orchestrator, 'reset'):
                self._orchestrator.reset()
        self._orchestrator = None

    @classmethod
    def from_config(cls, config: Config) -> "Engine":
        """
        Create an engine from a configuration object.

        Args:
            config: Complete configuration object

        Returns:
            New Engine instance with the specified configuration

        Example:
            >>> from latent_reasoning.config import Config
            >>> config = Config()
            >>> config.encoder.model = "Qwen/Qwen3-4B"
            >>> config.evolution.chains = 10
            >>> engine = Engine.from_config(config)
        """
        return cls(config=config)

    @classmethod
    def from_yaml(cls, path: str) -> "Engine":
        """
        Create an engine from a YAML configuration file.

        Args:
            path: Path to YAML configuration file

        Returns:
            New Engine instance with configuration loaded from file

        Example:
            >>> # Create config.yaml with your settings
            >>> engine = Engine.from_yaml("config.yaml")
            >>> result = engine.run("Your query here")
        """
        config = Config.from_yaml(path)
        return cls(config=config)
