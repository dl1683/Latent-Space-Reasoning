"""
Autopoietic (self-updating) Judge for latent space reasoning.

The autopoietic judge maintains itself through continuous learning:
1. Fast updates: EMA of internal scorer on each evaluation
2. Slow updates: Periodic grounding against external evaluator
3. Trust adaptation: Shifts weight to internal as correlation improves

This addresses the known scorer weakness (0.07 correlation) by treating
the judge as a living system that adapts to maintain quality alignment.

Key Concepts:

**Two-Time-Scale Learning**:
- Fast: Exponential moving average of scorer predictions (every eval)
- Slow: External grounding and weight adjustment (every N generations)

**Trust Weighting**:
- final_score = trust * internal_score + (1-trust) * external_score
- Trust starts low, increases as correlation improves

**Drift Prevention**:
- Periodic external grounding prevents scorer drift
- Anchor set provides stable correlation measurement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import torch
from torch import Tensor

from latent_reasoning.core.autopoietic.experience_buffer import ExperienceBuffer, ExperienceEntry
from latent_reasoning.core.autopoietic.external_evaluator import ExternalEvaluator, ExternalScore

if TYPE_CHECKING:
    from latent_reasoning.config import AutopoieticJudgeConfig


@dataclass
class JudgeState:
    """Snapshot of judge state for monitoring."""
    internal_trust: float
    correlation: float | None
    buffer_size: int
    grounded_count: int
    total_evaluations: int
    generation: int


@dataclass
class AnchorEntry:
    """Entry in the anchor set for correlation monitoring."""
    latent: Tensor
    query: str
    decoded_text: str
    external_score: float


class AutopoieticJudge:
    """
    Self-updating judge with two-time-scale learning.

    The judge combines an internal scorer (fast, cheap) with an external
    evaluator (slow, expensive) using adaptive trust weighting. Over time,
    as the internal scorer improves, trust shifts toward it.

    Two-Time-Scale Updates:
    1. **Fast (every evaluation)**: EMA update of running score statistics
    2. **Slow (every N generations)**: External grounding, trust adjustment

    Trust Evolution:
    - Starts at initial_internal_trust (e.g., 0.3)
    - Increases by trust_growth_rate when correlation > threshold
    - Bounded by max_internal_trust

    Args:
        config: AutopoieticJudgeConfig
        internal_scorer: Callable that scores latent vectors (returns float)
        external_evaluator: ExternalEvaluator for grounding
        decoder: Callable to decode latent to text (for external eval)
        device: Device for tensor operations

    Usage:
        >>> judge = AutopoieticJudge(config, scorer, evaluator, decoder)
        >>> score = judge.evaluate(latent, query="What is AI?")
        >>> # After N generations:
        >>> judge.ground(generation=5)
    """

    def __init__(
        self,
        config: "AutopoieticJudgeConfig",
        internal_scorer: Callable[[Tensor], float],
        external_evaluator: ExternalEvaluator,
        decoder: Callable[[Tensor, str], str],
        device: torch.device | str = "cpu",
    ):
        self.config = config

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Core components
        self.internal_scorer = internal_scorer
        self.external_evaluator = external_evaluator
        self.decoder = decoder

        # Experience buffer
        self.buffer = ExperienceBuffer(
            max_size=config.buffer_size,
            device=device,
        )

        # State
        self._internal_trust = config.initial_internal_trust
        self._correlation: float | None = None
        self._total_evaluations = 0
        self._generation = 0

        # EMA statistics
        self._ema_internal = 0.5
        self._ema_external = 0.5
        self._ema_count = 0

        # Anchor set for stable correlation measurement
        self._anchor_set: list[AnchorEntry] = []

    @property
    def internal_trust(self) -> float:
        """Current trust in internal scorer (0 to 1)."""
        return self._internal_trust

    @property
    def correlation(self) -> float | None:
        """Current correlation between internal and external scores."""
        return self._correlation

    def evaluate(
        self,
        latent: Tensor,
        query: str = "",
        use_external: bool = False,
    ) -> float:
        """
        Evaluate a latent vector and return quality score.

        By default, uses internal scorer only for speed. Set use_external=True
        for grounding evaluation (slower but provides ground truth).

        Args:
            latent: Latent vector to evaluate
            query: Query text for context (needed for external eval)
            use_external: Whether to also call external evaluator

        Returns:
            Quality score (0 to 1)
        """
        self._total_evaluations += 1

        # Internal score (fast)
        latent = latent.to(self.device)
        internal_score = self.internal_scorer(latent)

        # Update EMA
        self._update_ema(internal_score, None)

        # Add to buffer (external score will be filled later if grounded)
        idx = self.buffer.add(
            latent=latent,
            internal_score=internal_score,
            generation=self._generation,
            query=query,
        )

        if use_external and query:
            # Expensive external evaluation
            decoded = self.decoder(latent, query)
            external_result = self.external_evaluator.evaluate(query, decoded)

            if external_result.is_valid:
                self.buffer.update_external(idx, external_result.score)
                self._update_ema(internal_score, external_result.score)

                # Return trust-weighted combination
                return self._combine_scores(internal_score, external_result.score)

        return internal_score

    def evaluate_batch(
        self,
        latents: list[Tensor],
        queries: list[str] | None = None,
    ) -> list[float]:
        """
        Evaluate a batch of latent vectors (internal only for speed).

        Args:
            latents: List of latent vectors
            queries: Optional list of query texts

        Returns:
            List of quality scores
        """
        queries = queries or [""] * len(latents)
        return [
            self.evaluate(lat, query, use_external=False)
            for lat, query in zip(latents, queries)
        ]

    def ground(self, generation: int) -> dict:
        """
        Perform grounding update against external evaluator.

        Samples recent entries from the buffer, evaluates them externally,
        and updates trust based on correlation improvement.

        Args:
            generation: Current generation number

        Returns:
            Dictionary with grounding statistics
        """
        self._generation = generation

        # Sample entries for grounding
        entries = self.buffer.sample_recent(
            n=self.config.external_sample_size,
            recency_weight=0.7,
        )

        if not entries:
            return {"grounded": 0, "skipped": "no_entries"}

        grounded_count = 0
        for entry in entries:
            if entry.has_external:
                continue  # Already grounded

            if not entry.query:
                continue  # Need query for external eval

            # Decode and evaluate externally
            decoded = self.decoder(entry.latent, entry.query)
            external_result = self.external_evaluator.evaluate(entry.query, decoded)

            if external_result.is_valid:
                # Find entry in buffer and update
                for i, buf_entry in enumerate(self.buffer.entries):
                    if torch.allclose(buf_entry.latent, entry.latent):
                        self.buffer.update_external(i, external_result.score)
                        grounded_count += 1
                        break

                self._update_ema(entry.internal_score, external_result.score)

        # Update correlation
        self._correlation = self.buffer.compute_correlation()

        # Update trust based on correlation
        if self._correlation is not None and self._correlation > self.config.correlation_threshold:
            self._internal_trust = min(
                self.config.max_internal_trust,
                self._internal_trust + self.config.trust_growth_rate,
            )

        return {
            "grounded": grounded_count,
            "correlation": self._correlation,
            "trust": self._internal_trust,
            "buffer_size": len(self.buffer),
        }

    def build_anchor_set(
        self,
        latents: list[Tensor],
        queries: list[str],
    ) -> int:
        """
        Build anchor set for stable correlation monitoring.

        The anchor set is a fixed set of (latent, query, external_score) tuples
        that provides a stable reference for measuring correlation over time.

        Args:
            latents: Latent vectors for anchor set
            queries: Corresponding queries

        Returns:
            Number of anchors added
        """
        added = 0
        for latent, query in zip(latents, queries):
            if len(self._anchor_set) >= self.config.anchor_set_size:
                break

            # Decode and evaluate externally
            decoded = self.decoder(latent, query)
            external_result = self.external_evaluator.evaluate(query, decoded)

            if external_result.is_valid:
                self._anchor_set.append(AnchorEntry(
                    latent=latent.clone().detach().to(self.device),
                    query=query,
                    decoded_text=decoded,
                    external_score=external_result.score,
                ))
                added += 1

        return added

    def evaluate_anchor_correlation(self) -> float | None:
        """
        Compute correlation on anchor set.

        Uses the fixed anchor set for stable correlation measurement,
        unaffected by buffer churn.

        Returns:
            Pearson correlation, or None if anchor set too small
        """
        if len(self._anchor_set) < 3:
            return None

        internal_scores = []
        external_scores = []

        for anchor in self._anchor_set:
            internal = self.internal_scorer(anchor.latent)
            internal_scores.append(internal)
            external_scores.append(anchor.external_score)

        # Compute Pearson correlation
        internal = torch.tensor(internal_scores)
        external = torch.tensor(external_scores)

        internal_centered = internal - internal.mean()
        external_centered = external - external.mean()

        numerator = (internal_centered * external_centered).sum()
        denominator = torch.sqrt(
            (internal_centered ** 2).sum() * (external_centered ** 2).sum()
        )

        if denominator < 1e-8:
            return 0.0

        return (numerator / denominator).item()

    def _combine_scores(self, internal: float, external: float) -> float:
        """Combine internal and external scores using trust weighting."""
        return self._internal_trust * internal + (1 - self._internal_trust) * external

    def _update_ema(self, internal: float, external: float | None) -> None:
        """Update exponential moving averages."""
        decay = self.config.ema_decay
        self._ema_internal = decay * self._ema_internal + (1 - decay) * internal
        self._ema_count += 1

        if external is not None:
            self._ema_external = decay * self._ema_external + (1 - decay) * external

    def get_state(self) -> JudgeState:
        """Get current judge state for monitoring."""
        return JudgeState(
            internal_trust=self._internal_trust,
            correlation=self._correlation,
            buffer_size=len(self.buffer),
            grounded_count=len(self.buffer.get_grounded_entries()),
            total_evaluations=self._total_evaluations,
            generation=self._generation,
        )

    def get_statistics(self) -> dict:
        """Get detailed statistics for monitoring."""
        stats = {
            "internal_trust": self._internal_trust,
            "correlation": self._correlation,
            "ema_internal": self._ema_internal,
            "ema_external": self._ema_external,
            "total_evaluations": self._total_evaluations,
            "generation": self._generation,
            "anchor_set_size": len(self._anchor_set),
        }

        buffer_stats = self.buffer.get_statistics()
        stats.update({f"buffer_{k}": v for k, v in buffer_stats.items()})

        # Anchor correlation if available
        anchor_corr = self.evaluate_anchor_correlation()
        if anchor_corr is not None:
            stats["anchor_correlation"] = anchor_corr

        return stats

    def step_generation(self, generation: int) -> None:
        """
        Called at end of generation.

        Triggers grounding if it's time based on judge_update_freq.
        """
        self._generation = generation

        # Check if it's time to ground
        if generation > 0 and generation % self.config.judge_update_freq == 0:
            self.ground(generation)

    def reset(self) -> None:
        """Reset judge state (for new evolution run)."""
        self.buffer.clear()
        self._internal_trust = self.config.initial_internal_trust
        self._correlation = None
        self._total_evaluations = 0
        self._generation = 0
        self._ema_internal = 0.5
        self._ema_external = 0.5
        self._ema_count = 0
        self._anchor_set.clear()

    def to(self, device: torch.device | str) -> "AutopoieticJudge":
        """Move judge to device."""
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.buffer.to(device)
        self._anchor_set = [
            AnchorEntry(
                latent=a.latent.to(device),
                query=a.query,
                decoded_text=a.decoded_text,
                external_score=a.external_score,
            )
            for a in self._anchor_set
        ]
        return self

    def __repr__(self) -> str:
        return (
            f"AutopoieticJudge("
            f"trust={self._internal_trust:.2f}, "
            f"correlation={self._correlation}, "
            f"buffer={len(self.buffer)}/{self.config.buffer_size})"
        )
