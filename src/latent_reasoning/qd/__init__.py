"""
Quality Diversity (QD) module for Latent Space Reasoning.

This module provides QD algorithms that maintain archives of diverse,
high-quality solutions rather than converging to a single optimum.
QD enables the system to explore multiple valid reasoning approaches
and provides stepping stones for further exploration.

Key Components:

**Behavioral Descriptors (behavior.py)**:
- RFFProjector: Random Fourier Features for dimensionality reduction
- BehaviorComputer: Computes BDs from latent + structure + trajectory
- BehaviorDescriptor: Container for BD vectors with metadata

**Novelty Computation (novelty.py)**:
- NoveltyComputer: k-NN based novelty scoring
- combine_fitness_novelty: QD fitness function

**Archive Management (archive.py)**:
- DNSArchive: Dominated Novelty Search (gridless, recommended)
- MapElitesArchive: Grid-based alternative
- ArchiveEntry: Container for archived solutions

**Orchestration (manager.py)**:
- QDManager: Main interface integrating all QD components
- create_qd_manager: Factory function

Quick Start:
    >>> from latent_reasoning.qd import QDManager
    >>> from latent_reasoning.config import Config
    >>>
    >>> config = Config()
    >>> config.qd.enabled = True
    >>> config.qd.novelty_weight = 0.3
    >>>
    >>> manager = QDManager(config.qd, latent_dim=1024)
    >>>
    >>> # In evolution loop:
    >>> bds = manager.compute_bds(chains)
    >>> novelty = manager.compute_novelty(bds)
    >>> qd_scores = manager.combine_fitness(raw_scores, novelty)
    >>> manager.update_archive(chains, bds, raw_scores, qd_scores, gen)

References:
- "Dominated Novelty Search" (Feb 2025) - DNS archive algorithm
- "AutoQD" (June 2025) - RFF for behavioral descriptors
- "Novelty Search and the Problem with Objectives" (Lehman & Stanley, 2011)
"""

from latent_reasoning.qd.behavior import (
    BehaviorComputer,
    BehaviorDescriptor,
    RFFProjector,
)
from latent_reasoning.qd.novelty import (
    NoveltyComputer,
    combine_fitness_novelty,
    normalize_novelty_scores,
)
from latent_reasoning.qd.archive import (
    DNSArchive,
    MapElitesArchive,
    ArchiveEntry,
)
from latent_reasoning.qd.manager import (
    QDManager,
    create_qd_manager,
)

__all__ = [
    # Behavior
    "BehaviorComputer",
    "BehaviorDescriptor",
    "RFFProjector",
    # Novelty
    "NoveltyComputer",
    "combine_fitness_novelty",
    "normalize_novelty_scores",
    # Archive
    "DNSArchive",
    "MapElitesArchive",
    "ArchiveEntry",
    # Manager
    "QDManager",
    "create_qd_manager",
]
