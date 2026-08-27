"""
Simple interface for latent space reasoning.

This module provides the easiest way to use the Latent Space Reasoning Engine
with minimal setup. Perfect for quick experiments and simple use cases.

The main functions are:
- reason(): Run latent space reasoning on a single query
- compare(): Compare baseline vs latent reasoning side-by-side
"""

from __future__ import annotations

from latent_reasoning.config import Config, get_default_config
from latent_reasoning.engine import Engine, ReasoningResult


# Global engine for simple API - reused across calls for efficiency
_engine: Engine | None = None


def reason(
    query: str,
    config: Config | None = None,
    encoder: str | None = None,
    verbosity: str = "normal",
) -> ReasoningResult:
    """
    Run latent space reasoning on a query - the simplest interface.

    This function provides a one-liner way to get high-quality reasoning results
    using evolutionary optimization in latent space. Perfect for quick experiments.

    Args:
        query: The question or problem to reason about
        config: Optional custom configuration (uses sensible defaults if None)
        encoder: Model name to use (e.g., "Qwen/Qwen3-4B", "Qwen/Qwen3-0.6B")
        verbosity: How much output to show:
            - "silent": No output except errors
            - "minimal": Just final result
            - "normal": Progress updates (default)
            - "verbose": Detailed evolution info
            - "debug": Everything including internal state

    Returns:
        ReasoningResult containing:
            - plan: The generated reasoning plan/response
            - confidence: Quality score from 0-1 (higher = better)
            - generations: Number of evolution cycles run
            - evaluations: Total latent vectors evaluated
            - survivors: All final candidate solutions
            - history: Evolution progress (if verbosity >= "verbose")

    Examples:
        Basic usage:
        >>> from latent_reasoning import reason
        >>> result = reason("How do I implement user authentication?")
        >>> print(result.plan)
        >>> print(f"Confidence: {result.confidence:.2f}")

        With specific model:
        >>> result = reason(
        ...     "Design a REST API for a todo app",
        ...     encoder="Qwen/Qwen3-4B",
        ...     verbosity="verbose"
        ... )
        >>> print(f"Generated in {result.generations} generations")

        Quick and quiet:
        >>> result = reason("How to optimize SQL queries?", verbosity="silent")
        >>> print(result.plan)  # Just the result, no progress output
    """
    global _engine

    # Create or reuse engine
    if _engine is None or config is not None or encoder is not None:
        engine_config = config or get_default_config()
        if encoder:
            engine_config.encoder.model = encoder
        engine_config.output.verbosity = verbosity
        _engine = Engine(config=engine_config)

    return _engine.run(query)


def compare(
    query: str,
    config: Config | None = None,
    encoder: str | None = None,
) -> dict:
    """
    Compare latent reasoning vs baseline generation side-by-side.

    This is the best way to see the difference between standard LLM generation
    and latent space reasoning. It runs both methods on the same query and
    returns both outputs for comparison.

    Args:
        query: The question or problem to compare methods on
        config: Optional custom configuration
        encoder: Model name to use for both methods

    Returns:
        Dictionary containing:
            - "baseline": Direct LLM generation output
            - "latent_reasoning": Evolved latent space output
            - "latent_score": Quality score for the latent reasoning result
            - "generations": Number of evolution generations run
            - "evaluations": Total evaluations performed

    Examples:
        Basic comparison:
        >>> from latent_reasoning import compare
        >>> result = compare("How to implement rate limiting?")
        >>> print("=== Baseline ===")
        >>> print(result["baseline"])
        >>> print("\\n=== Latent Reasoning ===")
        >>> print(result["latent_reasoning"])
        >>> print(f"\\nScore: {result['latent_score']:.3f}")

        With larger model:
        >>> result = compare(
        ...     "Design a microservices architecture",
        ...     encoder="Qwen/Qwen3-4B"
        ... )
        >>> if result["latent_score"] > 0.8:
        ...     print("High quality result!")

        Programmatic evaluation:
        >>> queries = [
        ...     "How to optimize database queries?",
        ...     "Implement user authentication",
        ...     "Design a caching strategy"
        ... ]
        >>> for q in queries:
        ...     result = compare(q)
        ...     print(f"{q}: Score {result['latent_score']:.3f}")
    """
    global _engine

    if _engine is None or config is not None or encoder is not None:
        engine_config = config or get_default_config()
        if encoder:
            engine_config.encoder.model = encoder
        _engine = Engine(config=engine_config)

    return _engine.compare(query)


def reset() -> None:
    """
    Reset the global engine state.

    This clears the cached engine instance, forcing the next call to reason()
    or compare() to create a fresh engine. Useful when you want to change
    configuration or clear any cached state.

    Example:
        >>> from latent_reasoning import reason, reset
        >>> result1 = reason("Query 1", encoder="Qwen/Qwen3-0.6B")
        >>> reset()  # Clear cached engine
        >>> result2 = reason("Query 2", encoder="Qwen/Qwen3-4B")  # Uses new model
    """
    global _engine
    if _engine is not None:
        # Clear any cached state in the engine
        if hasattr(_engine, 'reset'):
            _engine.reset()
    _engine = None
