"""
Autopoietic Panel for integrating self-updating judge with evolution.

The AutopoieticPanel wraps the AutopoieticJudge and HomeostasisController
to provide a unified interface compatible with the existing JudgePanel API.
This enables drop-in replacement for standard evolution workflows.

Key Features:
- Compatible with existing JudgePanel interface
- Automatic homeostatic temperature control
- Periodic external grounding
- Statistics tracking for monitoring
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
from torch import Tensor

from latent_reasoning.core.autopoietic.autopoietic_judge import AutopoieticJudge
from latent_reasoning.core.autopoietic.homeostasis import HomeostasisController
from latent_reasoning.core.autopoietic.external_evaluator import (
    ExternalEvaluator,
    create_external_evaluator,
)
from latent_reasoning.core.autopoietic.experience_buffer import ExperienceBuffer

if TYPE_CHECKING:
    from latent_reasoning.config import AutopoieticJudgeConfig
    from latent_reasoning.core.chain import ChainState


@dataclass
class Verdict:
    """
    Evaluation verdict compatible with existing JudgePanel.

    Mirrors the structure expected by EvolutionLoop.
    """
    score: float
    modification: Tensor | None = None
    reasoning: str = ""


@dataclass
class PanelStatistics:
    """Statistics from the autopoietic panel."""
    judge_trust: float
    judge_correlation: float | None
    temperature: float
    diversity: float
    buffer_size: int
    grounded_count: int
    generation: int


class AutopoieticPanel:
    """
    Panel integrating autopoietic judge with homeostatic control.

    Provides a JudgePanel-compatible interface for use with EvolutionLoop,
    while adding self-updating and temperature regulation capabilities.

    The panel coordinates:
    1. AutopoieticJudge: Scoring with trust-weighted internal/external
    2. HomeostasisController: Adaptive temperature based on diversity
    3. External grounding: Periodic alignment with frontier model

    Args:
        config: AutopoieticJudgeConfig
        internal_scorer: Callable that scores latent vectors
        decoder: Callable to decode latent to text
        device: Device for computations
        use_mock_external: Use mock evaluator (for testing)

    Usage:
        >>> panel = AutopoieticPanel(config, scorer, decoder)
        >>> verdict = panel.evaluate(latent, context)
        >>> # At end of generation:
        >>> panel.step_generation(chains, generation)
        >>> new_temp = panel.temperature  # Use for mutation
    """

    def __init__(
        self,
        config: "AutopoieticJudgeConfig",
        internal_scorer: Callable[[Tensor], float],
        decoder: Callable[[Tensor, str], str],
        device: torch.device | str = "cpu",
        use_mock_external: bool = False,
    ):
        self.config = config

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Create external evaluator
        external_evaluator = create_external_evaluator(
            model=config.external_model,
            temperature=config.external_temperature,
            mock=use_mock_external,
        )

        # Create autopoietic judge
        self.judge = AutopoieticJudge(
            config=config,
            internal_scorer=internal_scorer,
            external_evaluator=external_evaluator,
            decoder=decoder,
            device=device,
        )

        # Create homeostasis controller
        self.homeostasis = HomeostasisController(
            target_diversity=config.target_diversity,
            control_gain=config.homeostasis_k,
            min_temperature=config.min_temperature,
            max_temperature=config.max_temperature,
            initial_temperature=0.5,  # Will be updated from evolution config
        )

        # State
        self._generation = 0
        self._current_query = ""

    @property
    def temperature(self) -> float:
        """Current temperature from homeostasis controller."""
        return self.homeostasis.temperature

    @property
    def internal_trust(self) -> float:
        """Current trust in internal scorer."""
        return self.judge.internal_trust

    @property
    def correlation(self) -> float | None:
        """Current correlation between internal and external."""
        return self.judge.correlation

    def evaluate(
        self,
        latent: Tensor,
        context: dict | None = None,
    ) -> Verdict:
        """
        Evaluate a latent vector.

        Compatible with JudgePanel.evaluate interface.

        Args:
            latent: Latent vector to evaluate
            context: Optional context dict (may contain 'query')

        Returns:
            Verdict with score
        """
        context = context or {}
        query = context.get("query", self._current_query)

        score = self.judge.evaluate(
            latent=latent,
            query=query,
            use_external=False,  # Fast path for evolution
        )

        return Verdict(score=score)

    def evaluate_batch(
        self,
        latents: list[Tensor],
        contexts: list[dict] | None = None,
    ) -> list[Verdict]:
        """
        Evaluate a batch of latent vectors.

        Args:
            latents: List of latent vectors
            contexts: Optional list of context dicts

        Returns:
            List of Verdict objects
        """
        contexts = contexts or [{}] * len(latents)
        return [self.evaluate(lat, ctx) for lat, ctx in zip(latents, contexts)]

    def get_modification(
        self,
        latent: Tensor,
        context: dict | None = None,
    ) -> Tensor | None:
        """
        Get modification suggestion for a latent.

        Note: Autopoietic panel doesn't provide modifications by default.
        Returns None to let mutation strategy handle exploration.

        Args:
            latent: Latent vector
            context: Optional context dict

        Returns:
            None (no modification suggestion)
        """
        return None

    def step_generation(
        self,
        chains: list["ChainState"],
        generation: int,
    ) -> dict:
        """
        Called at end of each generation.

        Updates homeostatic temperature and triggers judge grounding
        if it's time based on config.judge_update_freq.

        Args:
            chains: Current population of chains
            generation: Current generation number

        Returns:
            Dictionary with update statistics
        """
        self._generation = generation

        # Update homeostasis based on population diversity
        new_temp = self.homeostasis.update_from_population(
            chains=chains,
            generation=generation,
        )

        # Update judge (may trigger grounding)
        self.judge.step_generation(generation)

        # Build statistics
        stats = {
            "generation": generation,
            "temperature": new_temp,
            "diversity": self.homeostasis.diversity,
            "judge_trust": self.judge.internal_trust,
            "judge_correlation": self.judge.correlation,
        }

        return stats

    def set_query(self, query: str) -> None:
        """
        Set the current query for context.

        Should be called before running evolution on a new query.

        Args:
            query: The query text
        """
        self._current_query = query

    def ground_judge(self) -> dict:
        """
        Manually trigger judge grounding.

        Useful for initial calibration or periodic forced updates.

        Returns:
            Grounding statistics
        """
        return self.judge.ground(self._generation)

    def build_anchor_set(
        self,
        latents: list[Tensor],
        queries: list[str],
    ) -> int:
        """
        Build anchor set for stable correlation monitoring.

        Args:
            latents: Latent vectors
            queries: Corresponding queries

        Returns:
            Number of anchors added
        """
        return self.judge.build_anchor_set(latents, queries)

    def get_statistics(self) -> PanelStatistics:
        """Get current panel statistics."""
        return PanelStatistics(
            judge_trust=self.judge.internal_trust,
            judge_correlation=self.judge.correlation,
            temperature=self.homeostasis.temperature,
            diversity=self.homeostasis.diversity,
            buffer_size=len(self.judge.buffer),
            grounded_count=len(self.judge.buffer.get_grounded_entries()),
            generation=self._generation,
        )

    def get_detailed_statistics(self) -> dict:
        """Get detailed statistics from all components."""
        stats = {
            "generation": self._generation,
            "query": self._current_query[:50] if self._current_query else "",
        }

        # Judge stats
        judge_stats = self.judge.get_statistics()
        stats.update({f"judge_{k}": v for k, v in judge_stats.items()})

        # Homeostasis stats
        homeo_stats = self.homeostasis.get_statistics()
        stats.update({f"homeo_{k}": v for k, v in homeo_stats.items()})

        return stats

    def reset(self) -> None:
        """Reset panel state for new evolution run."""
        self.judge.reset()
        self.homeostasis.reset()
        self._generation = 0
        self._current_query = ""

    def to(self, device: torch.device | str) -> "AutopoieticPanel":
        """Move panel to device."""
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.judge.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"AutopoieticPanel("
            f"trust={self.internal_trust:.2f}, "
            f"temp={self.temperature:.3f}, "
            f"gen={self._generation})"
        )


def create_autopoietic_panel(
    config: "AutopoieticJudgeConfig",
    internal_scorer: Callable[[Tensor], float],
    decoder: Callable[[Tensor, str], str],
    device: str = "auto",
    use_mock_external: bool = False,
) -> AutopoieticPanel | None:
    """
    Factory function to create an AutopoieticPanel.

    Args:
        config: AutopoieticJudgeConfig
        internal_scorer: Callable that scores latent vectors
        decoder: Callable to decode latent to text
        device: Device for computations ("auto", "cuda", "cpu")
        use_mock_external: Use mock evaluator (for testing)

    Returns:
        AutopoieticPanel if config.enabled, else None
    """
    if not config.enabled:
        return None

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return AutopoieticPanel(
        config=config,
        internal_scorer=internal_scorer,
        decoder=decoder,
        device=device,
        use_mock_external=use_mock_external,
    )
