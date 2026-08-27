"""Judge panel for aggregating multiple judges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
from torch import Tensor

from latent_reasoning.core.judge import ScorerJudge, ModifierJudge
from latent_reasoning.core.chain import ModificationContext


@dataclass
class PanelVerdict:
    """Result from the judge panel evaluation."""

    score: float
    modification: Tensor | None
    individual_scores: List[float]
    individual_modifications: List[Tensor] | None = None


class JudgePanel:
    """
    Panel of judges that evaluates latent vectors.

    Aggregates scores from multiple scorer judges and modifications
    from multiple modifier judges.
    """

    def __init__(
        self,
        scorers: List[ScorerJudge] | None = None,
        modifiers: List[ModifierJudge] | None = None,
        aggregation: Literal["mean", "weighted", "max", "min"] = "mean",
        calibrate: bool = True,
    ):
        """
        Initialize the judge panel.

        Args:
            scorers: List of scorer judges
            modifiers: List of modifier judges
            aggregation: How to aggregate scores
            calibrate: Whether to calibrate scores
        """
        self.scorers = scorers or []
        self.modifiers = modifiers or []
        self.aggregation = aggregation
        self.should_calibrate = calibrate

        # Calibration data
        self.scorer_weights: List[float] = [1.0] * len(self.scorers)

    def add_scorer(self, scorer: ScorerJudge, weight: float = 1.0) -> None:
        """Add a scorer judge to the panel."""
        self.scorers.append(scorer)
        self.scorer_weights.append(weight)

    def add_modifier(self, modifier: ModifierJudge) -> None:
        """Add a modifier judge to the panel."""
        self.modifiers.append(modifier)

    def evaluate(
        self,
        latent: Tensor,
        context: ModificationContext | None = None,
    ) -> PanelVerdict:
        """
        Evaluate a latent vector.

        Args:
            latent: Latent vector to evaluate
            context: Optional context for modifiers

        Returns:
            PanelVerdict with aggregated score and modification
        """
        # Get scores from all scorers
        individual_scores = []
        for scorer in self.scorers:
            score = scorer.score(latent)
            individual_scores.append(score)

        # Aggregate scores
        if not individual_scores:
            aggregated_score = 0.0
        elif self.aggregation == "mean":
            aggregated_score = sum(individual_scores) / len(individual_scores)
        elif self.aggregation == "weighted":
            total_weight = sum(self.scorer_weights[:len(individual_scores)])
            aggregated_score = sum(
                s * w for s, w in zip(individual_scores, self.scorer_weights)
            ) / max(total_weight, 1e-6)
        elif self.aggregation == "max":
            aggregated_score = max(individual_scores)
        elif self.aggregation == "min":
            aggregated_score = min(individual_scores)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Get modifications from all modifiers
        individual_modifications = []
        combined_modification = None

        if self.modifiers:
            for modifier in self.modifiers:
                mod = modifier.suggest_modification(latent, context)
                individual_modifications.append(mod)

            # Average the modifications
            if individual_modifications:
                device = individual_modifications[0].device
                stacked = torch.stack([m.to(device) for m in individual_modifications])
                combined_modification = stacked.mean(dim=0)

        return PanelVerdict(
            score=aggregated_score,
            modification=combined_modification,
            individual_scores=individual_scores,
            individual_modifications=individual_modifications if individual_modifications else None,
        )

    def score(self, latent: Tensor) -> float:
        """Get just the score for a latent vector."""
        verdict = self.evaluate(latent)
        return verdict.score

    def get_modification(
        self,
        latent: Tensor,
        context: ModificationContext | None = None,
    ) -> Tensor | None:
        """Get just the modification for a latent vector."""
        verdict = self.evaluate(latent, context)
        return verdict.modification

    def calibrate(self, latents: Tensor) -> None:
        """
        Calibrate all scorers on a set of latents.

        Args:
            latents: Batch of latent vectors for calibration
        """
        if not self.should_calibrate:
            return

        for scorer in self.scorers:
            scorer.calibrate(latents)

    def evaluate_batch(
        self,
        latents: List[Tensor],
        contexts: List[ModificationContext] | None = None,
    ) -> List[PanelVerdict]:
        """
        Evaluate multiple latent vectors.

        Args:
            latents: List of latent vectors
            contexts: Optional list of contexts (one per latent)

        Returns:
            List of PanelVerdicts
        """
        results = []
        for i, latent in enumerate(latents):
            context = contexts[i] if contexts else None
            verdict = self.evaluate(latent, context)
            results.append(verdict)
        return results

    def __len__(self) -> int:
        """Return total number of judges."""
        return len(self.scorers) + len(self.modifiers)

    @property
    def num_scorers(self) -> int:
        """Return number of scorer judges."""
        return len(self.scorers)

    @property
    def num_modifiers(self) -> int:
        """Return number of modifier judges."""
        return len(self.modifiers)
