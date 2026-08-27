"""Chain state management and tracking for evolutionary reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor

from latent_reasoning.utils.device import ensure_tensor_device


@dataclass
class CrossChainSummary:
    """Summary of cross-chain state for context-aware modification."""

    num_active_chains: int
    best_global_score: float
    score_distribution: tuple[float, float, float]  # min, median, max
    centroid: Tensor | None = None
    spread: float = 0.0
    recently_explored: Tensor | None = None


@dataclass
class ModificationContext:
    """Context information for modification judges."""

    # Chain-local context
    current: Tensor
    momentum: Tensor | None = None
    recent_modifications: Tensor | None = None
    stuck_signal: float = 0.0

    # Cross-chain context
    cross_chain: CrossChainSummary | None = None


@dataclass
class ChainState:
    """State of a single reasoning chain."""

    latent: Tensor
    score: float = 0.0
    raw_score: float = 0.0  # Raw fitness score (without QD novelty bonus)
    generation: int = 0
    history: List[Tensor] = field(default_factory=list)
    modification_history: List[Tensor] = field(default_factory=list)
    is_converged: bool = False
    is_dead: bool = False

    def update(self, new_latent: Tensor, new_score: float, modification: Tensor | None = None):
        """Update chain with new state."""
        self.history.append(self.latent.clone())
        if modification is not None:
            self.modification_history.append(modification.clone())
        self.latent = new_latent
        self.score = new_score
        self.generation += 1

    def clone(self) -> "ChainState":
        """Create a copy of this chain state."""
        return ChainState(
            latent=self.latent.clone(),
            score=self.score,
            raw_score=self.raw_score,
            generation=self.generation,
            history=[h.clone() for h in self.history],
            modification_history=[m.clone() for m in self.modification_history],
            is_converged=self.is_converged,
            is_dead=self.is_dead,
        )


class ChainTracker:
    """Tracks chain history and detects patterns like loops and stagnation."""

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        stagnation_threshold: float = 0.01,
        history_window: int = 10,
    ):
        """
        Initialize the chain tracker.

        Args:
            similarity_threshold: Cosine similarity threshold for loop detection
            stagnation_threshold: Minimum delta norm to not be considered stagnating
            history_window: Number of recent states to track
        """
        self.similarity_threshold = similarity_threshold
        self.stagnation_threshold = stagnation_threshold
        self.history_window = history_window

        self.history: List[Tensor] = []
        self.modification_history: List[Tensor] = []
        self.score_history: List[float] = []

    def record(self, latent: Tensor, score: float, modification: Tensor | None = None):
        """Record a new state."""
        self.history.append(latent.clone())
        self.score_history.append(score)

        if modification is not None:
            self.modification_history.append(modification.clone())

        # Keep only recent history
        if len(self.history) > self.history_window:
            self.history = self.history[-self.history_window:]
            self.score_history = self.score_history[-self.history_window:]
            self.modification_history = self.modification_history[-self.history_window:]

    def check_progress(self, current: Tensor) -> dict:
        """
        Check if the chain is making progress.

        Returns:
            Dict with 'ok' bool and optional 'issue' string
        """
        if len(self.history) < 2:
            return {"ok": True}

        device = current.device

        # Check for loops (high similarity to past states)
        for past in self.history[:-1]:
            past = ensure_tensor_device(past, device)
            sim = self._cosine_similarity(current, past)
            if sim > self.similarity_threshold:
                return {"ok": False, "issue": "loop_detected", "similarity": sim}

        # Check for stagnation (small deltas)
        if len(self.history) >= 3:
            recent_deltas = []
            for i in range(-2, 0):
                prev = ensure_tensor_device(self.history[i - 1], device)
                curr = ensure_tensor_device(self.history[i], device)
                delta = torch.norm(curr - prev).item()
                recent_deltas.append(delta)

            if all(d < self.stagnation_threshold for d in recent_deltas):
                return {"ok": False, "issue": "stagnation", "deltas": recent_deltas}

        return {"ok": True}

    def compute_momentum(self, window: int = 5) -> Tensor | None:
        """Compute the average direction of recent moves."""
        if len(self.history) < 2:
            return None

        device = self.history[-1].device
        recent = self.history[-window:] if len(self.history) >= window else self.history

        deltas = []
        for i in range(len(recent) - 1):
            prev = ensure_tensor_device(recent[i], device)
            curr = ensure_tensor_device(recent[i + 1], device)
            deltas.append(curr - prev)

        if not deltas:
            return None

        return torch.stack(deltas).mean(dim=0)

    def compute_stuck_signal(self) -> float:
        """
        Compute how "stuck" the chain is.

        Returns:
            Float between 0 (progressing) and 1 (stuck)
        """
        if len(self.history) < 3:
            return 0.0

        # Based on recent score improvement
        recent_scores = self.score_history[-5:] if len(self.score_history) >= 5 else self.score_history
        if len(recent_scores) < 2:
            return 0.0

        # Check if scores are improving
        improvements = [
            recent_scores[i + 1] - recent_scores[i]
            for i in range(len(recent_scores) - 1)
        ]

        avg_improvement = sum(improvements) / len(improvements)

        # Also check latent space movement
        device = self.history[-1].device
        recent = self.history[-3:]
        deltas = []
        for i in range(len(recent) - 1):
            prev = ensure_tensor_device(recent[i], device)
            curr = ensure_tensor_device(recent[i + 1], device)
            deltas.append(torch.norm(curr - prev).item())

        avg_delta = sum(deltas) / len(deltas) if deltas else 0

        # Combine signals
        # No improvement and small deltas = stuck
        if avg_improvement <= 0 and avg_delta < self.stagnation_threshold:
            return 1.0
        elif avg_improvement <= 0:
            return 0.7
        elif avg_delta < self.stagnation_threshold:
            return 0.5
        else:
            return 0.0

    def get_recent_modifications(self, n: int = 3) -> Tensor | None:
        """Get the last N modification vectors."""
        if not self.modification_history:
            return None

        recent = self.modification_history[-n:]
        if not recent:
            return None

        return torch.stack(recent)

    def get_context(self, current: Tensor, cross_chain: CrossChainSummary | None = None) -> ModificationContext:
        """Build a modification context for the current state."""
        return ModificationContext(
            current=current,
            momentum=self.compute_momentum(),
            recent_modifications=self.get_recent_modifications(),
            stuck_signal=self.compute_stuck_signal(),
            cross_chain=cross_chain,
        )

    def _cosine_similarity(self, a: Tensor, b: Tensor) -> float:
        """Compute cosine similarity between two tensors."""
        a = a.flatten().float()
        b = b.flatten().float()
        return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    def reset(self):
        """Reset the tracker."""
        self.history.clear()
        self.modification_history.clear()
        self.score_history.clear()


def compute_cross_chain_summary(chains: List[ChainState]) -> CrossChainSummary:
    """Compute a summary of cross-chain state."""
    if not chains:
        return CrossChainSummary(
            num_active_chains=0,
            best_global_score=0.0,
            score_distribution=(0.0, 0.0, 0.0),
        )

    scores = [c.score for c in chains]
    scores_sorted = sorted(scores)

    # Compute median
    n = len(scores_sorted)
    if n % 2 == 0:
        median = (scores_sorted[n // 2 - 1] + scores_sorted[n // 2]) / 2
    else:
        median = scores_sorted[n // 2]

    # Compute centroid
    device = chains[0].latent.device
    latents = torch.stack([ensure_tensor_device(c.latent, device) for c in chains])
    centroid = latents.mean(dim=0)

    # Compute spread (average pairwise distance)
    spread = 0.0
    if len(chains) > 1:
        distances = []
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                dist = torch.norm(chains[i].latent - chains[j].latent).item()
                distances.append(dist)
        spread = sum(distances) / len(distances)

    # Recently explored = rolling average of recent positions
    all_recent = []
    for chain in chains:
        if chain.history:
            recent = chain.history[-3:] if len(chain.history) >= 3 else chain.history
            all_recent.extend(recent)

    recently_explored = None
    if all_recent:
        all_recent = [ensure_tensor_device(r, device) for r in all_recent]
        recently_explored = torch.stack(all_recent).mean(dim=0)

    return CrossChainSummary(
        num_active_chains=len(chains),
        best_global_score=max(scores),
        score_distribution=(min(scores), median, max(scores)),
        centroid=centroid,
        spread=spread,
        recently_explored=recently_explored,
    )
