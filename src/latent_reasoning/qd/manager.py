"""
QD Manager for orchestrating Quality Diversity in evolution.

The QDManager integrates behavioral descriptor computation, novelty
scoring, and archive management into a cohesive interface for the
EvolutionLoop.

This is the main entry point for QD functionality - it coordinates
all QD components and provides a simple API for the evolution loop.

Usage:
    >>> from latent_reasoning.qd import QDManager
    >>> from latent_reasoning.config import QDConfig
    >>>
    >>> config = QDConfig(enabled=True, bd_dim=16)
    >>> manager = QDManager(config, latent_dim=1024)
    >>>
    >>> # In evolution loop:
    >>> bds = manager.compute_bds(chains)
    >>> novelty = manager.compute_novelty(bds)
    >>> qd_scores = manager.combine_fitness(raw_scores, novelty)
    >>> manager.update_archive(chains, bds, raw_scores, qd_scores, gen)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from latent_reasoning.qd.behavior import BehaviorComputer, BehaviorDescriptor
from latent_reasoning.qd.novelty import NoveltyComputer, combine_fitness_novelty, normalize_novelty_scores
from latent_reasoning.qd.archive import DNSArchive, ArchiveEntry

if TYPE_CHECKING:
    from latent_reasoning.config import QDConfig
    from latent_reasoning.core.chain import ChainState


class QDManager:
    """
    Orchestrates Quality Diversity integration for evolution.

    Provides a unified interface for:
    - Computing behavioral descriptors for chains
    - Computing novelty scores against the archive
    - Combining fitness with novelty for QD scoring
    - Managing the archive of diverse solutions
    - Sampling diverse parents for reproduction

    The QDManager is designed to be used as an optional enhancement
    to the standard EvolutionLoop - when enabled, it replaces the
    simple diversity bonus with a full QD system.

    Args:
        config: QD configuration
        latent_dim: Dimension of latent vectors
        device: Device for computations ("auto", "cuda", "cpu")

    Example:
        >>> config = QDConfig(enabled=True, novelty_weight=0.3)
        >>> manager = QDManager(config, latent_dim=1024)
        >>>
        >>> # Each generation:
        >>> bds = manager.compute_bds(chains)
        >>> novelty = manager.compute_novelty(bds)
        >>> qd_scores = manager.combine_fitness(raw_scores, novelty)
        >>> added, rejected = manager.update_archive(...)
    """

    def __init__(
        self,
        config: "QDConfig",
        latent_dim: int,
        device: torch.device | str = "auto",
    ):
        self.config = config
        self.latent_dim = latent_dim

        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Initialize behavior computer
        self.behavior_computer = BehaviorComputer(
            latent_dim=latent_dim,
            bd_dim=config.bd_dim,
            rff_gamma=config.rff_gamma,
            latent_weight=config.bd_latent_weight,
            structural_weight=config.bd_structural_weight,
            trajectory_weight=config.bd_trajectory_weight,
            device=self.device,
        )

        # Initialize novelty computer
        self.novelty_computer = NoveltyComputer(
            k=config.novelty_k,
            distance_metric="euclidean",
        )

        # Initialize archive
        self.archive = DNSArchive(
            max_size=config.archive_size,
            domination_threshold=config.domination_threshold,
            novelty_threshold=config.novelty_threshold,
        )

        # Statistics tracking
        self._generation = 0
        self._total_added = 0
        self._total_rejected = 0

    def compute_bds(
        self,
        chains: list["ChainState"],
    ) -> list[BehaviorDescriptor]:
        """
        Compute behavioral descriptors for all chains.

        Args:
            chains: List of chain states from evolution

        Returns:
            List of BehaviorDescriptors
        """
        bds = []
        for chain in chains:
            # Get history if available
            history = None
            if hasattr(chain, 'history') and chain.history:
                history = chain.history

            # Get generation if available
            generation = getattr(chain, 'generation', self._generation)

            bd = self.behavior_computer.compute(
                latent=chain.latent,
                generation=generation,
                history=history,
            )
            bds.append(bd)
        return bds

    def compute_novelty(
        self,
        bds: list[BehaviorDescriptor],
    ) -> list[float]:
        """
        Compute novelty scores for all BDs against the archive.

        Args:
            bds: List of behavioral descriptors

        Returns:
            List of novelty scores (higher = more novel)
        """
        bd_vectors = [bd.vector for bd in bds]
        archive_bds = self.archive.get_bds()

        return self.novelty_computer.compute_novelty_batch(
            bds=bd_vectors,
            archive_bds=archive_bds,
        )

    def combine_fitness(
        self,
        raw_scores: list[float],
        novelty_scores: list[float],
        normalize_novelty: bool = True,
    ) -> list[float]:
        """
        Combine raw fitness with novelty for QD fitness.

        Uses the formula: qd_score = (1-α)*fitness + α*novelty
        where α is config.novelty_weight

        IMPORTANT: Novelty is normalized to [0, 1] before combining to ensure
        it doesn't dominate fitness scores. Without normalization, raw novelty
        (which is a distance metric) can be much larger than fitness (which is
        typically in [0, 1]), causing the search to prioritize novelty over
        actual quality.

        Args:
            raw_scores: Raw fitness scores from judges (typically [0, 1])
            novelty_scores: Novelty scores from compute_novelty (raw distances)
            normalize_novelty: Whether to normalize novelty to [0, 1] (default: True)

        Returns:
            List of combined QD fitness scores
        """
        # Normalize novelty scores to [0, 1] to match fitness scale
        # This prevents novelty from dominating the combined score
        if normalize_novelty and novelty_scores:
            normalized_novelty = normalize_novelty_scores(novelty_scores, method="minmax")
        else:
            normalized_novelty = novelty_scores

        return [
            combine_fitness_novelty(
                fitness=fitness,
                novelty=novelty,
                alpha=self.config.novelty_weight,
            )
            for fitness, novelty in zip(raw_scores, normalized_novelty)
        ]

    def update_archive(
        self,
        chains: list["ChainState"],
        bds: list[BehaviorDescriptor],
        raw_scores: list[float],
        qd_scores: list[float],
        generation: int,
        metadata: list[dict] | None = None,
    ) -> tuple[int, int]:
        """
        Update the archive with current generation's solutions.

        Attempts to add each solution to the archive based on
        domination rules. Solutions that dominate existing entries
        replace them; solutions dominated by existing entries are
        rejected.

        Args:
            chains: Chain states with latent vectors
            bds: Behavioral descriptors
            raw_scores: Raw fitness scores
            qd_scores: Combined QD fitness scores
            generation: Current generation number
            metadata: Optional list of metadata dicts per chain

        Returns:
            Tuple of (added_count, rejected_count)
        """
        self._generation = generation
        metadata = metadata or [{}] * len(chains)

        added = 0
        rejected = 0

        for chain, bd, raw, qd, meta in zip(chains, bds, raw_scores, qd_scores, metadata):
            was_added, reason = self.archive.try_add(
                latent=chain.latent,
                bd=bd.vector,
                fitness=raw,
                qd_fitness=qd,
                generation=generation,
                metadata=meta,
            )
            if was_added:
                added += 1
                self._total_added += 1
            else:
                rejected += 1
                self._total_rejected += 1

        return added, rejected

    def sample_parents(self, n: int, method: str = "diverse") -> list[Tensor]:
        """
        Sample parent latents from the archive for reproduction.

        Args:
            n: Number of parents to sample
            method: Sampling method ("diverse", "random", "weighted")

        Returns:
            List of latent vectors
        """
        if len(self.archive) == 0:
            return []

        if method == "diverse":
            entries = self.archive.sample_diverse(n)
        elif method == "random":
            entries = self.archive.sample_random(n)
        elif method == "weighted":
            entries = self.archive.sample_weighted(n)
        else:
            entries = self.archive.sample_diverse(n)

        return [entry.latent for entry in entries]

    def get_archive_statistics(self) -> dict:
        """Get current archive statistics."""
        stats = self.archive.get_statistics()
        stats.update({
            "total_added": self._total_added,
            "total_rejected": self._total_rejected,
            "add_rate": self._total_added / max(1, self._total_added + self._total_rejected),
        })
        return stats

    def get_best_solutions(self, n: int = 5) -> list[ArchiveEntry]:
        """Get the n best solutions by raw fitness."""
        return self.archive.get_best(n)

    def get_diverse_solutions(self, n: int = 5) -> list[ArchiveEntry]:
        """Get n diverse solutions using farthest-point sampling."""
        return self.archive.sample_diverse(n)

    def get_all_latents(self) -> list[Tensor]:
        """Get all latent vectors in the archive."""
        return self.archive.get_latents()

    def update(
        self,
        latent: Tensor,
        score: float,
        metadata: dict | None = None,
    ) -> bool:
        """
        Convenience method to update archive with a single latent/score pair.

        This is a simpler interface than update_archive() for cases where you
        have individual solutions rather than batches (e.g., grammar evolution).

        Args:
            latent: Latent vector to add
            score: Fitness score for the latent
            metadata: Optional metadata dict

        Returns:
            True if the solution was added to the archive, False if rejected
        """
        # Create a minimal ChainState-like object for BD computation
        from latent_reasoning.core.chain import ChainState
        chain = ChainState(latent=latent, score=score, raw_score=score)

        # Compute BD
        bd = self.behavior_computer.compute(
            latent=latent,
            generation=self._generation,
            history=None,
        )

        # Try to add directly to archive
        was_added, reason = self.archive.try_add(
            latent=latent,
            bd=bd.vector,
            fitness=score,
            qd_fitness=score,  # For single updates, use raw score as QD score
            generation=self._generation,
            metadata=metadata or {},
        )

        if was_added:
            self._total_added += 1
        else:
            self._total_rejected += 1

        return was_added

    def step_generation(self) -> None:
        """Called at end of generation to update internal state."""
        self._generation += 1

    def reset(self) -> None:
        """Reset the archive and statistics (for new evolution run)."""
        self.archive.clear()
        self._generation = 0
        self._total_added = 0
        self._total_rejected = 0

    def to(self, device: torch.device | str) -> "QDManager":
        """Move manager components to device."""
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.behavior_computer.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"QDManager(bd_dim={self.config.bd_dim}, "
            f"archive_size={len(self.archive)}/{self.config.archive_size}, "
            f"novelty_weight={self.config.novelty_weight})"
        )


def create_qd_manager(
    config: "QDConfig",
    latent_dim: int,
    device: str = "auto",
) -> QDManager | None:
    """
    Factory function to create a QDManager if QD is enabled.

    Args:
        config: QD configuration
        latent_dim: Dimension of latent vectors
        device: Device for computations

    Returns:
        QDManager if config.enabled, else None
    """
    if not config.enabled:
        return None
    return QDManager(config, latent_dim, device)
