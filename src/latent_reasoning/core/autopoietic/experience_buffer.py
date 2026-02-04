"""
Experience buffer for autopoietic judge learning.

The experience buffer stores (latent, internal_score, external_score) tuples
that can be used to update the internal scorer based on external feedback.
This enables online learning and drift correction.

Key Features:
- Ring buffer implementation for fixed memory footprint
- Random sampling for mini-batch training
- Priority sampling based on score discrepancy
- Statistics tracking for analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator
import random

import torch
from torch import Tensor


@dataclass
class ExperienceEntry:
    """
    Single entry in the experience buffer.

    Stores a latent vector along with both internal (scorer) and external
    (grounding judge) scores. The discrepancy between these scores drives
    learning updates.

    Attributes:
        latent: The latent vector
        internal_score: Score from internal scorer
        external_score: Score from external evaluator (may be None if not evaluated)
        generation: Generation when this entry was created
        query: Optional query text for context
        timestamp: Entry creation timestamp (for recency weighting)
    """
    latent: Tensor
    internal_score: float
    external_score: float | None = None
    generation: int = 0
    query: str = ""
    timestamp: int = 0

    @property
    def has_external(self) -> bool:
        """Whether this entry has been externally evaluated."""
        return self.external_score is not None

    @property
    def discrepancy(self) -> float | None:
        """Score discrepancy (external - internal). None if no external score."""
        if self.external_score is None:
            return None
        return self.external_score - self.internal_score

    def to(self, device: torch.device | str) -> "ExperienceEntry":
        """Move latent to specified device."""
        return ExperienceEntry(
            latent=self.latent.to(device),
            internal_score=self.internal_score,
            external_score=self.external_score,
            generation=self.generation,
            query=self.query,
            timestamp=self.timestamp,
        )


class ExperienceBuffer:
    """
    Ring buffer for storing experience tuples.

    Maintains a fixed-size buffer of (latent, internal_score, external_score)
    entries that can be sampled for training updates. Supports both random
    and priority-based sampling.

    The buffer implements a ring buffer strategy: when full, new entries
    overwrite the oldest entries. This ensures constant memory usage while
    keeping the most recent experiences available.

    Args:
        max_size: Maximum number of entries to store
        device: Device for tensor operations

    Usage:
        >>> buffer = ExperienceBuffer(max_size=1000)
        >>> buffer.add(latent, internal_score=0.7, query="test")
        >>> buffer.update_external(0, external_score=0.8)
        >>> batch = buffer.sample(n=32)
    """

    def __init__(
        self,
        max_size: int = 1000,
        device: torch.device | str = "cpu",
    ):
        self.max_size = max_size
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.entries: list[ExperienceEntry] = []
        self._next_idx = 0  # For ring buffer insertion
        self._timestamp = 0

    def add(
        self,
        latent: Tensor,
        internal_score: float,
        external_score: float | None = None,
        generation: int = 0,
        query: str = "",
    ) -> int:
        """
        Add an experience entry to the buffer.

        Args:
            latent: Latent vector
            internal_score: Score from internal scorer
            external_score: Optional score from external evaluator
            generation: Current generation number
            query: Optional query text for context

        Returns:
            Index where entry was stored
        """
        self._timestamp += 1
        entry = ExperienceEntry(
            latent=latent.clone().detach().to(self.device),
            internal_score=internal_score,
            external_score=external_score,
            generation=generation,
            query=query,
            timestamp=self._timestamp,
        )

        if len(self.entries) < self.max_size:
            self.entries.append(entry)
            return len(self.entries) - 1
        else:
            # Ring buffer: overwrite oldest
            idx = self._next_idx
            self.entries[idx] = entry
            self._next_idx = (self._next_idx + 1) % self.max_size
            return idx

    def update_external(self, idx: int, external_score: float) -> None:
        """
        Update the external score for an existing entry.

        Args:
            idx: Index of the entry to update
            external_score: Score from external evaluator
        """
        if 0 <= idx < len(self.entries):
            entry = self.entries[idx]
            self.entries[idx] = ExperienceEntry(
                latent=entry.latent,
                internal_score=entry.internal_score,
                external_score=external_score,
                generation=entry.generation,
                query=entry.query,
                timestamp=entry.timestamp,
            )

    def sample(self, n: int, with_external_only: bool = False) -> list[ExperienceEntry]:
        """
        Sample entries uniformly at random.

        Args:
            n: Number of entries to sample
            with_external_only: If True, only sample entries with external scores

        Returns:
            List of sampled entries
        """
        if with_external_only:
            candidates = [e for e in self.entries if e.has_external]
        else:
            candidates = self.entries

        if not candidates:
            return []

        n = min(n, len(candidates))
        return random.sample(candidates, n)

    def sample_priority(
        self,
        n: int,
        temperature: float = 1.0,
    ) -> list[ExperienceEntry]:
        """
        Sample entries with priority based on score discrepancy.

        Entries with larger discrepancy (external - internal) are more
        likely to be sampled, as they represent opportunities for learning.

        Args:
            n: Number of entries to sample
            temperature: Sampling temperature (higher = more uniform)

        Returns:
            List of sampled entries
        """
        candidates = [e for e in self.entries if e.has_external]
        if not candidates:
            return []

        n = min(n, len(candidates))

        # Compute priority weights based on absolute discrepancy
        discrepancies = [abs(e.discrepancy or 0) for e in candidates]
        max_disc = max(discrepancies) if discrepancies else 1.0

        # Normalize and apply temperature
        weights = torch.tensor([d / max(max_disc, 1e-8) for d in discrepancies])
        weights = torch.softmax(weights / temperature, dim=0)

        # Sample without replacement
        indices = torch.multinomial(weights, n, replacement=False)
        return [candidates[i] for i in indices.tolist()]

    def sample_recent(self, n: int, recency_weight: float = 0.5) -> list[ExperienceEntry]:
        """
        Sample entries with recency bias.

        Args:
            n: Number of entries to sample
            recency_weight: How much to weight recency (0 = uniform, 1 = most recent only)

        Returns:
            List of sampled entries
        """
        if not self.entries:
            return []

        n = min(n, len(self.entries))

        # Compute recency weights
        max_ts = self._timestamp
        weights = torch.tensor([
            recency_weight * (e.timestamp / max_ts) + (1 - recency_weight)
            for e in self.entries
        ])
        weights = weights / weights.sum()

        # Sample without replacement
        indices = torch.multinomial(weights, n, replacement=False)
        return [self.entries[i] for i in indices.tolist()]

    def get_grounded_entries(self) -> list[ExperienceEntry]:
        """Get all entries that have external scores."""
        return [e for e in self.entries if e.has_external]

    def get_ungrounded_entries(self) -> list[ExperienceEntry]:
        """Get all entries without external scores."""
        return [e for e in self.entries if not e.has_external]

    def get_statistics(self) -> dict:
        """Get buffer statistics for monitoring."""
        grounded = self.get_grounded_entries()

        stats = {
            "size": len(self.entries),
            "max_size": self.max_size,
            "fill_ratio": len(self.entries) / self.max_size,
            "grounded_count": len(grounded),
            "grounded_ratio": len(grounded) / max(len(self.entries), 1),
        }

        if grounded:
            discrepancies = [e.discrepancy for e in grounded]
            stats["mean_discrepancy"] = sum(discrepancies) / len(discrepancies)
            stats["max_discrepancy"] = max(abs(d) for d in discrepancies)
            stats["mean_internal"] = sum(e.internal_score for e in grounded) / len(grounded)
            stats["mean_external"] = sum(e.external_score for e in grounded) / len(grounded)

        return stats

    def compute_correlation(self) -> float | None:
        """
        Compute Pearson correlation between internal and external scores.

        Returns:
            Correlation coefficient, or None if insufficient data
        """
        grounded = self.get_grounded_entries()
        if len(grounded) < 3:
            return None

        internal = torch.tensor([e.internal_score for e in grounded])
        external = torch.tensor([e.external_score for e in grounded])

        # Pearson correlation
        internal_centered = internal - internal.mean()
        external_centered = external - external.mean()

        numerator = (internal_centered * external_centered).sum()
        denominator = torch.sqrt((internal_centered ** 2).sum() * (external_centered ** 2).sum())

        if denominator < 1e-8:
            return 0.0

        return (numerator / denominator).item()

    def clear(self) -> None:
        """Clear all entries from the buffer."""
        self.entries.clear()
        self._next_idx = 0
        self._timestamp = 0

    def to(self, device: torch.device | str) -> "ExperienceBuffer":
        """Move all tensors to specified device."""
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.entries = [e.to(device) for e in self.entries]
        return self

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[ExperienceEntry]:
        return iter(self.entries)

    def __getitem__(self, idx: int) -> ExperienceEntry:
        return self.entries[idx]

    def __repr__(self) -> str:
        grounded = sum(1 for e in self.entries if e.has_external)
        return f"ExperienceBuffer(size={len(self)}/{self.max_size}, grounded={grounded})"
