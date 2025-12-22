"""
Configuration schema for Latent Space Reasoning Engine.

This module defines all configuration classes using Pydantic for validation
and type safety. The main Config class combines all settings and can be
loaded from YAML files for easy customization.

Key configuration areas:
- EncoderConfig: Model selection and encoding parameters
- JudgeConfig: Scoring and evaluation settings
- EvolutionConfig: Evolutionary algorithm parameters
- BudgetConfig: Compute limits and timeouts
- SynthesisConfig: Text generation settings
- OutputConfig: Logging and formatting options
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class Verbosity(IntEnum):
    """
    Verbosity levels for logging output.

    Controls how much information is displayed during reasoning:
    - SILENT (0): Only errors
    - MINIMAL (1): Just final results
    - NORMAL (2): Progress updates (default)
    - VERBOSE (3): Detailed evolution info
    - DEBUG (4): Everything including internal state
    """
    SILENT = 0
    MINIMAL = 1
    NORMAL = 2
    VERBOSE = 3
    DEBUG = 4


class EncoderConfig(BaseModel):
    """
    Configuration for the encoder model that converts text to latent vectors.

    The encoder is the foundation of the entire reasoning system - it defines the
    latent space where evolutionary optimization occurs. The choice of encoder
    model and parameters significantly impacts both quality and performance.

    Model Selection Guidelines:
    - **Qwen/Qwen3-4B**: Best quality, requires ~8GB VRAM (recommended)
    - **Qwen/Qwen3-1.7B**: Good balance of quality/speed, ~4GB VRAM
    - **Qwen/Qwen3-0.6B**: Fastest, minimal VRAM (~2GB), lower quality
    - **microsoft/Phi-3.5-mini-instruct**: Alternative, good for reasoning
    - **Local models**: Use absolute path to local model directory

    Layer Selection:
    - **-4 (default)**: 4th layer from end, optimal for most reasoning tasks
    - **-2 to -6**: Earlier layers capture syntax, later capture semantics
    - **-1**: Final layer, most semantic but may be over-specialized

    Pooling Strategies:
    - **mean**: Average all tokens, captures full context (recommended)
    - **last**: Use final token, good for completion-style tasks
    - **cls**: Use first token, for models with explicit CLS tokens

    Performance Considerations:
    - Larger models produce higher quality but require more compute
    - Layer choice affects both quality and evolution effectiveness
    - max_length impacts memory usage and processing time
    """

    # Model selection - choose based on your hardware and quality needs
    model: str = "Qwen/Qwen3-4B"  # HuggingFace model ID or local path

    # Latent extraction settings
    layer: int = -4                # Which hidden layer to extract (-4 = 4th from last, usually optimal)
    pooling: Literal["mean", "last", "cls"] = "mean"  # How to pool sequence dimension

    # Hardware settings
    device: str = "auto"           # Device: "auto", "cuda", "cuda:0", "cpu"
    quantization: Literal["auto", "none", "4bit"] = "auto"
    max_length: int = Field(default=2048, ge=1, le=32768)  # Maximum input sequence length

    class Config:
        extra = "forbid"  # Prevent typos in config files


class ScorerConfig(BaseModel):
    """
    Configuration for a scorer judge that evaluates latent vector quality.

    Scorers are the critical component that drives evolution - they determine
    which latent vectors represent higher-quality responses and should survive
    to the next generation. The quality of scoring directly impacts the
    effectiveness of the evolutionary process.

    Scorer Types:

    **semantic**: Cosine similarity-based scoring
    - Uses a separate embedding model to encode the query
    - Scores latents based on similarity to query embedding
    - Fast and general-purpose, works out-of-the-box
    - Best for: Initial experiments, general reasoning tasks
    - Limitations: May not capture task-specific quality nuances

    **trained_latent**: Neural network trained on quality data
    - Uses a specialized network trained to predict response quality
    - Works directly in the encoder's latent space
    - More accurate and task-specific than semantic similarity
    - Best for: Production use, specific domains with training data
    - Requirements: Trained checkpoint file matching encoder dimension

    Configuration Guidelines:
    - Start with semantic for quick experiments
    - Use trained_latent for best quality when available
    - Multiple scorers can be combined for robust evaluation
    - Ensure latent_dim matches your encoder's output dimension
    """

    # Scorer type selection
    type: Literal["semantic", "trained_latent"] = "semantic"

    # For semantic scorer (cosine similarity based)
    model: str | None = None       # HuggingFace model for encoding (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    layer: int = -1                # Which layer to extract from semantic model
    head: str | None = None        # Optional path to trained regression head weights
    quantization: Literal["auto", "none", "4bit"] = "none"

    # For trained_latent scorer (neural network trained on quality data)
    checkpoint: str | None = None  # Path to trained LatentScorer checkpoint
    latent_dim: int = 1024         # Latent dimension (must match encoder)

    class Config:
        extra = "forbid"


class ModifierConfig(BaseModel):
    """
    Configuration for a modifier judge that suggests improvements.

    Modifiers analyze latent vectors and suggest directions for mutation
    to improve quality. They use LLM hidden states to understand what
    changes might help.
    """

    model: str                     # HuggingFace model for modification suggestions
    layers: list[int] = Field(default_factory=lambda: [-8, -4])  # Hidden layers to extract from
    quantization: Literal["auto", "none", "4bit"] = "auto"

    class Config:
        extra = "forbid"


class JudgeConfig(BaseModel):
    """
    Configuration for the judge panel that evaluates and improves latents.

    The judge panel combines multiple scorers and modifiers to evaluate
    latent vector quality and suggest improvements during evolution.
    """

    # Scorer configuration - determines quality evaluation
    scorers: list[ScorerConfig] = Field(
        default_factory=lambda: [ScorerConfig(
            type="trained_latent",
            checkpoint="checkpoints/latent_scorer/final_model.pt",
            latent_dim=1024,
        )]
    )

    # Modifier configuration - suggests improvements (optional)
    modifiers: list[ModifierConfig] = Field(default_factory=list)

    # Judge panel behavior
    aggregation: Literal["mean", "weighted", "max", "min"] = "mean"  # How to combine multiple scorer outputs
    calibrate: bool = True         # Whether to normalize scores to zero-mean

    class Config:
        extra = "forbid"


class SelectionConfig(BaseModel):
    """
    Configuration for selection strategy in evolution.

    Selection determines which latent vectors survive to the next generation.
    Different strategies balance exploration vs exploitation.
    """

    strategy: Literal["elitist", "tournament", "roulette", "rank"] = "elitist"
    survivors: int = Field(default=5, ge=1)    # How many chains to keep each generation
    elite: int = Field(default=2, ge=0)        # Always keep top N (for elitist strategy)

    class Config:
        extra = "forbid"


class MutationConfig(BaseModel):
    """
    Configuration for mutation strategy in evolution.

    Mutation adds variation to latent vectors to explore new regions
    of the latent space. Different strategies balance random exploration
    with directed improvement.
    """

    strategy: Literal["directed", "gaussian", "adaptive"] = "directed"
    trust: float = Field(default=0.7, ge=0, le=1)  # How much to trust modifier suggestions (0=random, 1=fully directed)

    class Config:
        extra = "forbid"


class CrossoverConfig(BaseModel):
    """
    Configuration for crossover strategy in evolution.

    Crossover combines high-scoring parent latents to create children
    that inherit beneficial traits from both parents.
    """

    strategy: Literal["weighted", "mean", "interpolation"] = "weighted"
    threshold: float = Field(default=0.3, ge=0, le=1)  # Minimum population diversity required for crossover

    class Config:
        extra = "forbid"


class MergeConfig(BaseModel):
    """
    Configuration for merging similar chains.

    When chains become too similar, they can be merged to avoid
    wasting computation on redundant exploration.
    """

    threshold: float = Field(default=0.9, ge=0, le=1)  # Cosine similarity threshold for merging

    class Config:
        extra = "forbid"


class ConvergenceConfig(BaseModel):
    """
    Configuration for convergence detection.

    Evolution stops early when convergence criteria are met,
    saving computation when further improvement is unlikely.
    """

    threshold: float = Field(default=0.95, ge=0, le=1)  # Score threshold for early stopping
    patience: int = Field(default=5, ge=1)              # Generations without improvement before stopping

    class Config:
        extra = "forbid"


class EvolutionConfig(BaseModel):
    """
    Configuration for the evolutionary optimization process.

    Controls the core evolutionary algorithm that optimizes latent vectors
    through selection, mutation, and crossover operations. These parameters
    significantly impact both quality and computational efficiency.

    Key Parameters:

    **Population Settings**:
    - chains: Population size (more = better exploration, slower)
    - generations: Evolution cycles (more = better optimization, slower)

    **Mutation Control**:
    - temperature: Mutation strength (higher = more exploration)
    - temperature_decay: Cooling schedule (gradual focus on exploitation)
    - min_viable_score: Quality threshold for survival

    **Algorithm Components**:
    - selection: How survivors are chosen each generation
    - mutation: How variations are created
    - crossover: How parents are combined
    - merge: When similar chains are consolidated
    - convergence: Early stopping criteria

    Tuning Guidelines:
    - **Fast experiments**: chains=5, generations=10
    - **Balanced (default)**: chains=10, generations=30
    - **High quality**: chains=20, generations=50
    - **Research**: chains=50, generations=100+

    Temperature Schedule:
    - Start high (0.8-1.0) for exploration
    - Decay gradually (0.95-0.98) to focus search
    - End low (0.1-0.3) for fine-tuning
    """

    chains: int = Field(default=10, ge=1)
    generations: int = Field(default=30, ge=1)
    temperature: float = Field(default=0.5, ge=0, le=2)
    temperature_decay: float = Field(default=0.95, ge=0, le=1)
    min_viable_score: float = Field(default=0.1, ge=0, le=1)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    mutation: MutationConfig = Field(default_factory=MutationConfig)
    crossover: CrossoverConfig = Field(default_factory=CrossoverConfig)
    merge: MergeConfig = Field(default_factory=MergeConfig)
    convergence: ConvergenceConfig = Field(default_factory=ConvergenceConfig)

    class Config:
        extra = "forbid"


class BudgetConfig(BaseModel):
    """Configuration for compute budget."""
    max_evaluations: int = Field(default=500, ge=1)
    max_time: float | None = None  # Seconds, None = unlimited

    class Config:
        extra = "forbid"


class SynthesisConfig(BaseModel):
    """Configuration for final synthesis."""
    decoder: str | None = None  # None = use encoder model
    model: str = "gemini-2.5-flash"  # Frontier model for synthesis
    decode_strategy: Literal["best", "combined"] = "best"
    max_survivors: int = Field(default=5, ge=1)
    max_tokens: int = Field(default=2048, ge=1, le=32768)  # Max output tokens  
    temperature: float = Field(default=0.7, ge=0, le=2.0)  # Decode temperature

    class Config:
        extra = "forbid"


class OutputConfig(BaseModel):
    """Configuration for output settings."""
    verbosity: Literal["silent", "minimal", "normal", "verbose", "debug"] = "normal"
    format: Literal["text", "json", "markdown"] = "text"
    save_history: bool = False
    history_path: str | None = None

    class Config:
        extra = "forbid"


class BenchmarkConfig(BaseModel):
    """Configuration for benchmarks/evaluation."""
    judge_model: str = "gemini-2.5-flash"
    blind: bool = True
    temperature: float = 0  # Deterministic judging

    class Config:
        extra = "forbid"


class Config(BaseModel):
    """
    Main configuration for the Latent Space Reasoning Engine.

    This is the central configuration object that controls all aspects of the
    reasoning system. It provides a hierarchical structure for organizing
    settings and supports both programmatic and file-based configuration.

    Configuration Sections:
    - **encoder**: Model and parameters for text-to-latent conversion
    - **judges**: Scoring and modification strategies for evaluation
    - **evolution**: Evolutionary algorithm parameters and strategies
    - **budget**: Computational limits and resource management
    - **synthesis**: Final response generation and decoding
    - **output**: Logging, formatting, and result storage
    - **benchmark**: Evaluation and comparison settings

    Usage Patterns:

    **Default Configuration**:
    >>> config = Config()  # Uses sensible defaults
    >>> engine = Engine(config=config)

    **Programmatic Customization**:
    >>> config = Config()
    >>> config.encoder.model = "Qwen/Qwen3-1.7B"  # Smaller model
    >>> config.evolution.chains = 20              # More exploration
    >>> config.output.verbosity = "verbose"       # Detailed logging

    **YAML Configuration**:
    >>> config = Config.from_yaml("my_config.yaml")
    >>> # Modify if needed
    >>> config.evolution.generations = 50
    >>> config.to_yaml("updated_config.yaml")

    **Environment-specific Configs**:
    >>> # Development: fast iteration
    >>> dev_config = Config()
    >>> dev_config.evolution.chains = 5
    >>> dev_config.evolution.generations = 10
    >>>
    >>> # Production: high quality
    >>> prod_config = Config()
    >>> prod_config.evolution.chains = 20
    >>> prod_config.evolution.generations = 50
    >>> prod_config.budget.max_time = 300  # 5 minute limit

    The configuration system uses Pydantic for validation, ensuring that
    all parameters are valid and providing helpful error messages for
    invalid configurations.
    """

    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    judges: JudgeConfig = Field(default_factory=JudgeConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    class Config:
        extra = "forbid"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Load configuration from a dictionary."""
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        return self.model_dump()


def get_default_config() -> Config:
    """Get a default configuration with sensible defaults for testing."""
    return Config(
        encoder=EncoderConfig(model="Qwen/Qwen3-4B"),
        judges=JudgeConfig(
            # Use smaller encoder model for scoring (faster)
            scorers=[ScorerConfig(type="semantic", model="sentence-transformers/all-MiniLM-L6-v2")],
            # Use smaller Qwen3 for modifications
            modifiers=[ModifierConfig(model="Qwen/Qwen3-1.7B")],
        ),
        evolution=EvolutionConfig(
            chains=5,
            generations=10,
            temperature=0.5,
        ),
        budget=BudgetConfig(max_evaluations=200),
    )


def get_trained_scorer_config(checkpoint_path: str = "checkpoints/latent_scorer/final_model.pt") -> Config:
    """Get a configuration that uses the trained latent scorer."""
    return Config(
        encoder=EncoderConfig(model="Qwen/Qwen3-0.6B"),  # Smaller for faster iteration
        judges=JudgeConfig(
            scorers=[ScorerConfig(
                type="trained_latent",
                checkpoint=checkpoint_path,
                latent_dim=1024,
            )],
            modifiers=[],  # No modifier for cleaner experiments
        ),
        evolution=EvolutionConfig(
            chains=5,
            generations=15,
            temperature=0.5,
        ),
        budget=BudgetConfig(max_evaluations=300),
    )
