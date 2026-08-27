"""
Latent Space Reasoning Engine with Evolutionary Optimization

A novel approach to LLM reasoning that optimizes in latent space using evolutionary
algorithms to generate higher-quality, more specific responses than standard text generation.

## Core Concept

Instead of generating text directly, this engine:
1. **Encodes** your query into latent space using transformer hidden states
2. **Evolves** the representation through selection, mutation, and crossover
3. **Scores** candidates using trained neural networks or semantic similarity
4. **Decodes** the best result to structured, specific text

This approach consistently produces more detailed, accurate, and useful responses
compared to standard text generation, especially for complex reasoning tasks.

## API Reference

### Simple Interface
```python
from latent_reasoning import reason

# One-line reasoning with automatic configuration
result = reason("How do I implement user authentication?")
print(result.plan)
print(f"Confidence: {result.confidence:.2f}")
```

### Advanced Interface
```python
from latent_reasoning import Engine, Config

# Custom configuration for specific needs
config = Config()
config.encoder.model = "Qwen/Qwen3-4B"  # Higher quality model
config.evolution.chains = 20            # More exploration
config.evolution.generations = 50       # Longer optimization

engine = Engine(config=config)
result = engine.run("Design a REST API")
print(f"Plan: {result.plan}")
print(f"Evolution stats: {result.generations} generations, {result.total_evaluations} evaluations")
```

### Comparison Mode
```python
# Compare baseline vs latent reasoning side-by-side
comparison = engine.compare("How to optimize database queries?")
print("Baseline Response:")
print(comparison["baseline"])
print("\nLatent Reasoning Response:")
print(comparison["latent_reasoning"])
print(f"\nImprovement Factor: {comparison['improvement_factor']:.2f}x")
```

### CLI Usage
```bash
# Quick comparison
latent-reason compare "How to implement caching?"

# Custom model and parameters
latent-reason run "Design microservices" --encoder Qwen/Qwen3-1.7B --chains 15

# Verbose output with evolution details
latent-reason run "Optimize SQL queries" --verbosity verbose
```

## Key Components

- **Engine**: Main interface for reasoning operations
- **Config**: Hierarchical configuration system
- **ReasoningResult**: Structured result with metadata
- **reason()**: Simple one-function interface
- **compare()**: Side-by-side baseline comparison
"""

from latent_reasoning.config import (
    Config,
    EncoderConfig,
    JudgeConfig,
    ScorerConfig,
    ModifierConfig,
    EvolutionConfig,
    BudgetConfig,
    SynthesisConfig,
    OutputConfig,
)
from latent_reasoning.engine import Engine, ReasoningResult
from latent_reasoning.reason import reason, compare

__version__ = "0.2.0"

__all__ = [
    # Main interfaces
    "reason",           # Simple one-liner function
    "compare",          # Compare baseline vs latent reasoning
    "Engine",           # Full-featured engine class
    "ReasoningResult",  # Result dataclass

    # Configuration classes
    "Config",           # Main configuration
    "EncoderConfig",    # Model and encoding settings
    "JudgeConfig",      # Scoring and evaluation settings
    "ScorerConfig",     # Individual scorer configuration
    "ModifierConfig",   # Modification suggestion settings
    "EvolutionConfig",  # Evolution algorithm parameters
    "BudgetConfig",     # Compute budget limits
    "SynthesisConfig",  # Text generation settings
    "OutputConfig",     # Logging and output formatting
]
