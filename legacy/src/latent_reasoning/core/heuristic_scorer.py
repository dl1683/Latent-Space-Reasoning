"""Heuristic-based scoring for decoded text quality.

This scorer evaluates the quality of decoded plans based on structural
heuristics rather than latent similarity. This encourages evolution
to produce better-structured outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TextQualityScores:
    """Breakdown of text quality scores."""
    structure_score: float  # Has numbered steps, headers, etc.
    depth_score: float  # Length and detail
    action_score: float  # Contains actionable verbs
    coherence_score: float  # Logical flow indicators
    overall_score: float  # Weighted combination


class HeuristicScorer:
    """
    Scores decoded text based on quality heuristics.

    This complements or replaces latent similarity scoring by
    evaluating the actual output quality.
    """

    def __init__(
        self,
        structure_weight: float = 0.3,
        depth_weight: float = 0.2,
        action_weight: float = 0.3,
        coherence_weight: float = 0.2,
    ):
        """
        Initialize the heuristic scorer.

        Args:
            structure_weight: Weight for structure score
            depth_weight: Weight for depth score
            action_weight: Weight for action verb presence
            coherence_weight: Weight for coherence indicators
        """
        self.weights = {
            "structure": structure_weight,
            "depth": depth_weight,
            "action": action_weight,
            "coherence": coherence_weight,
        }

        # Action verbs that indicate concrete steps
        self.action_verbs = [
            "create", "implement", "design", "develop", "build", "deploy",
            "configure", "set up", "install", "test", "validate", "verify",
            "analyze", "assess", "evaluate", "monitor", "review", "optimize",
            "define", "specify", "document", "plan", "schedule", "prioritize",
            "integrate", "connect", "migrate", "update", "refactor", "fix",
            "automate", "scale", "secure", "backup", "restore", "debug",
        ]

        # Coherence indicators (words that show logical flow)
        self.coherence_markers = [
            "first", "then", "next", "after", "before", "finally", "lastly",
            "once", "when", "while", "during", "following", "subsequently",
            "therefore", "thus", "hence", "because", "since", "as a result",
            "in order to", "to ensure", "to achieve", "to accomplish",
        ]

    def score(self, text: str) -> TextQualityScores:
        """
        Score the quality of decoded text.

        Args:
            text: Decoded plan text

        Returns:
            TextQualityScores with breakdown and overall score
        """
        structure = self._score_structure(text)
        depth = self._score_depth(text)
        action = self._score_action(text)
        coherence = self._score_coherence(text)

        overall = (
            structure * self.weights["structure"]
            + depth * self.weights["depth"]
            + action * self.weights["action"]
            + coherence * self.weights["coherence"]
        )

        return TextQualityScores(
            structure_score=structure,
            depth_score=depth,
            action_score=action,
            coherence_score=coherence,
            overall_score=overall,
        )

    def _score_structure(self, text: str) -> float:
        """Score based on structural elements."""
        score = 0.0

        # Check for numbered steps (1., 2., etc. or Step 1, Step 2, etc.)
        numbered_steps = len(re.findall(r'(?:^|\n)\s*(?:\d+\.|Step \d+)', text))
        if numbered_steps >= 5:
            score += 0.4
        elif numbered_steps >= 3:
            score += 0.3
        elif numbered_steps >= 1:
            score += 0.2

        # Check for headers (###, ##, **, etc.)
        headers = len(re.findall(r'(?:^|\n)\s*(?:#{1,3}|[*]{2})', text))
        if headers >= 5:
            score += 0.3
        elif headers >= 3:
            score += 0.2
        elif headers >= 1:
            score += 0.1

        # Check for bullet points
        bullets = len(re.findall(r'(?:^|\n)\s*[-*•]', text))
        if bullets >= 5:
            score += 0.2
        elif bullets >= 3:
            score += 0.15
        elif bullets >= 1:
            score += 0.1

        # Bonus for consistent structure
        if numbered_steps >= 3 and (headers >= 3 or bullets >= 5):
            score += 0.1

        return min(1.0, score)

    def _score_depth(self, text: str) -> float:
        """Score based on depth and detail."""
        # Word count (target: 300-1000 words for a good plan)
        words = len(text.split())

        if words < 100:
            word_score = words / 200  # Linear ramp up to 100
        elif words < 300:
            word_score = 0.5 + (words - 100) / 400  # Ramp to 1.0 at 300
        elif words <= 1000:
            word_score = 1.0
        else:
            # Slight penalty for being too verbose
            word_score = max(0.7, 1.0 - (words - 1000) / 2000)

        # Paragraph diversity
        paragraphs = text.split('\n\n')
        para_score = min(1.0, len(paragraphs) / 10)

        return (word_score * 0.7 + para_score * 0.3)

    def _score_action(self, text: str) -> float:
        """Score based on presence of action verbs."""
        text_lower = text.lower()

        # Count unique action verbs used
        used_verbs = set()
        for verb in self.action_verbs:
            # Match verb as whole word
            if re.search(rf'\b{verb}\b', text_lower):
                used_verbs.add(verb)

        # Score based on variety of action verbs
        verb_count = len(used_verbs)

        if verb_count >= 15:
            return 1.0
        elif verb_count >= 10:
            return 0.85
        elif verb_count >= 7:
            return 0.7
        elif verb_count >= 5:
            return 0.55
        elif verb_count >= 3:
            return 0.4
        else:
            return verb_count / 7.5  # Linear for low counts

    def _score_coherence(self, text: str) -> float:
        """Score based on coherence indicators."""
        text_lower = text.lower()

        # Count coherence markers
        marker_count = 0
        for marker in self.coherence_markers:
            marker_count += len(re.findall(rf'\b{re.escape(marker)}\b', text_lower))

        # Score based on marker presence
        if marker_count >= 10:
            return 1.0
        elif marker_count >= 7:
            return 0.85
        elif marker_count >= 5:
            return 0.7
        elif marker_count >= 3:
            return 0.5
        else:
            return marker_count / 6  # Linear for low counts


class DiversityBonusScorer:
    """
    Adds a diversity bonus to encourage exploration.

    Penalizes latents that are too similar to the seed,
    encouraging the evolution to explore different regions.
    """

    def __init__(
        self,
        diversity_weight: float = 0.3,
        min_diversity: float = 0.05,
        max_diversity: float = 0.5,
    ):
        """
        Initialize the diversity scorer.

        Args:
            diversity_weight: Weight of diversity in final score
            min_diversity: Minimum useful diversity (cosine distance)
            max_diversity: Maximum useful diversity before penalty
        """
        self.diversity_weight = diversity_weight
        self.min_diversity = min_diversity
        self.max_diversity = max_diversity
        self._seed: Tensor | None = None

    def set_seed(self, seed: Tensor) -> None:
        """Set the seed latent for diversity comparison."""
        self._seed = seed.clone()
        if self._seed.dim() == 1:
            self._seed = self._seed.unsqueeze(0)
        # Normalize for cosine similarity
        self._seed = torch.nn.functional.normalize(self._seed.float(), dim=-1)

    def score_diversity(self, latent: Tensor) -> float:
        """
        Score the diversity of a latent from the seed.

        Args:
            latent: Latent vector to score

        Returns:
            Diversity score between 0 and 1
        """
        if self._seed is None:
            return 0.5  # Neutral if no seed set

        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        latent_norm = torch.nn.functional.normalize(latent.float(), dim=-1)

        # Cosine similarity
        sim = torch.nn.functional.cosine_similarity(latent_norm, self._seed).item()

        # Convert to distance (0 = identical, 1 = orthogonal)
        distance = 1.0 - sim

        # Map distance to score
        if distance < self.min_diversity:
            # Too similar - low score
            return distance / self.min_diversity * 0.5
        elif distance <= self.max_diversity:
            # Sweet spot - high score
            range_size = self.max_diversity - self.min_diversity
            normalized = (distance - self.min_diversity) / range_size
            return 0.5 + normalized * 0.5
        else:
            # Too different - slight penalty
            excess = distance - self.max_diversity
            return max(0.3, 1.0 - excess)

    def combine_scores(
        self,
        base_score: float,
        latent: Tensor,
    ) -> float:
        """
        Combine base score with diversity bonus.

        Args:
            base_score: Original score (e.g., from similarity)
            latent: Latent vector being scored

        Returns:
            Combined score with diversity bonus
        """
        diversity = self.score_diversity(latent)

        combined = (
            base_score * (1 - self.diversity_weight)
            + diversity * self.diversity_weight
        )

        return combined
