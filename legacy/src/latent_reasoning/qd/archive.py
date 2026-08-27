"""
QD Archive implementations for storing diverse solutions.

Archives are the core data structure of Quality Diversity algorithms.
They store a collection of high-quality, diverse solutions found during
evolution, enabling:
- Diverse output generation (multiple valid approaches)
- Recovery from local optima through diversity pressure
- Stepping stones for further exploration

Key Implementations:
- DNSArchive: Dominated Novelty Search (gridless, high-dim friendly)
- ArchiveEntry: Container for archived solutions

Reference: "Dominated Novelty Search" (Feb 2025)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import torch
from torch import Tensor


@dataclass
class ArchiveEntry:
    """
    Single entry in the QD archive.

    Stores a solution along with its behavioral descriptor, fitness scores,
    and metadata for analysis.

    Attributes:
        latent: The latent vector (solution)
        bd: Behavioral descriptor characterizing the solution
        fitness: Raw fitness score from judge
        qd_fitness: Combined QD fitness (fitness + novelty)
        generation: Generation when this entry was added
        metadata: Optional extra information (query, decoded text, etc.)
    """
    latent: Tensor
    bd: Tensor
    fitness: float
    qd_fitness: float
    generation: int
    metadata: dict = field(default_factory=dict)

    def __lt__(self, other: "ArchiveEntry") -> bool:
        """For sorting - higher qd_fitness is 'greater'."""
        return self.qd_fitness < other.qd_fitness

    def to(self, device: torch.device | str) -> "ArchiveEntry":
        """Move tensors to specified device."""
        return ArchiveEntry(
            latent=self.latent.to(device),
            bd=self.bd.to(device),
            fitness=self.fitness,
            qd_fitness=self.qd_fitness,
            generation=self.generation,
            metadata=self.metadata,
        )


class DNSArchive:
    """
    Dominated Novelty Search archive.

    Gridless QD archive that maintains diversity through novelty-based
    domination checks rather than fixed grid cells. This approach:
    - Doesn't require predefined BD boundaries
    - Works well in high-dimensional BD spaces
    - Dynamically adapts to the distribution of solutions

    Domination Rules:
    - A solution dominates another if it's in the same BD neighborhood
      AND has higher fitness
    - When adding a new solution, it replaces dominated solutions
    - If the new solution is dominated, it's rejected

    Args:
        max_size: Maximum number of entries to store
        domination_threshold: BD distance below which domination is checked
        novelty_threshold: Minimum novelty for automatic addition

    Usage:
        >>> archive = DNSArchive(max_size=500)
        >>> added, reason = archive.try_add(latent, bd, fitness, qd_fitness, gen)
        >>> if added:
        ...     print(f"Added! Archive size: {len(archive)}")
    """

    def __init__(
        self,
        max_size: int = 500,
        domination_threshold: float = 0.1,
        novelty_threshold: float = 0.05,
    ):
        self.max_size = max_size
        self.domination_threshold = domination_threshold
        self.novelty_threshold = novelty_threshold

        self.entries: list[ArchiveEntry] = []
        self._bd_cache: list[Tensor] = []  # Cache for fast BD lookup

    def try_add(
        self,
        latent: Tensor,
        bd: Tensor,
        fitness: float,
        qd_fitness: float,
        generation: int,
        metadata: dict | None = None,
    ) -> tuple[bool, str]:
        """
        Try to add a solution to the archive.

        Checks domination relationships and adds if the solution is
        either novel enough or dominates existing entries.

        Args:
            latent: Latent vector (solution)
            bd: Behavioral descriptor
            fitness: Raw fitness score
            qd_fitness: Combined QD fitness
            generation: Current generation
            metadata: Optional metadata

        Returns:
            Tuple of (was_added: bool, reason: str)
        """
        entry = ArchiveEntry(
            latent=latent.clone().detach(),
            bd=bd.clone().detach(),
            fitness=fitness,
            qd_fitness=qd_fitness,
            generation=generation,
            metadata=metadata or {},
        )

        # Check if dominated by existing entries
        dominated_by = self._find_dominating(bd, fitness)
        if dominated_by is not None:
            return False, f"dominated_by_entry_{dominated_by}"

        # Check if this dominates existing entries
        dominated_indices = self._find_dominated(bd, fitness)

        # Remove dominated entries
        if dominated_indices:
            for idx in sorted(dominated_indices, reverse=True):
                self.entries.pop(idx)
                self._bd_cache.pop(idx)

        # Add new entry
        self.entries.append(entry)
        self._bd_cache.append(bd.clone().detach())

        # Prune if over capacity
        if len(self.entries) > self.max_size:
            self._prune()

        return True, "added"

    def _find_dominating(self, bd: Tensor, fitness: float) -> int | None:
        """Find an entry that dominates the given solution."""
        for i, entry in enumerate(self.entries):
            bd_dist = torch.norm(bd.to(entry.bd.device) - entry.bd).item()
            if bd_dist < self.domination_threshold:
                # In similar BD region - check fitness domination
                if entry.fitness > fitness:
                    return i
        return None

    def _find_dominated(self, bd: Tensor, fitness: float) -> list[int]:
        """Find entries dominated by the given solution."""
        dominated = []
        for i, entry in enumerate(self.entries):
            bd_dist = torch.norm(bd.to(entry.bd.device) - entry.bd).item()
            if bd_dist < self.domination_threshold:
                if fitness > entry.fitness:
                    dominated.append(i)
        return dominated

    def _prune(self) -> None:
        """Remove lowest QD fitness entries to maintain max_size."""
        while len(self.entries) > self.max_size:
            # Find entry with lowest QD fitness
            min_idx = min(range(len(self.entries)),
                         key=lambda i: self.entries[i].qd_fitness)
            self.entries.pop(min_idx)
            self._bd_cache.pop(min_idx)

    def get_bds(self) -> list[Tensor]:
        """Get all behavioral descriptors in the archive."""
        return self._bd_cache.copy()

    def get_latents(self) -> list[Tensor]:
        """Get all latent vectors in the archive."""
        return [e.latent for e in self.entries]

    def get_entries(self) -> list[ArchiveEntry]:
        """Get all archive entries."""
        return self.entries.copy()

    def get_best(self, n: int = 1) -> list[ArchiveEntry]:
        """Get the n best entries by raw fitness."""
        sorted_entries = sorted(self.entries, key=lambda e: e.fitness, reverse=True)
        return sorted_entries[:n]

    def sample_diverse(self, n: int) -> list[ArchiveEntry]:
        """
        Sample n diverse entries from the archive.

        Uses a greedy farthest-point sampling approach to maximize
        diversity in the returned set.

        Args:
            n: Number of entries to sample

        Returns:
            List of diverse ArchiveEntry objects
        """
        if len(self.entries) <= n:
            return self.entries.copy()

        # Start with highest fitness entry
        selected_indices = [max(range(len(self.entries)),
                               key=lambda i: self.entries[i].fitness)]

        while len(selected_indices) < n:
            # Find entry farthest from all selected
            max_min_dist = -1
            best_idx = -1

            for i in range(len(self.entries)):
                if i in selected_indices:
                    continue

                # Min distance to any selected entry
                min_dist = min(
                    torch.norm(self.entries[i].bd.to(self.entries[j].bd.device) - self.entries[j].bd).item()
                    for j in selected_indices
                )

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i

            if best_idx >= 0:
                selected_indices.append(best_idx)
            else:
                break

        return [self.entries[i] for i in selected_indices]

    def sample_random(self, n: int) -> list[ArchiveEntry]:
        """Sample n random entries from the archive."""
        import random
        if len(self.entries) <= n:
            return self.entries.copy()
        return random.sample(self.entries, n)

    def sample_weighted(self, n: int, temperature: float = 1.0) -> list[ArchiveEntry]:
        """
        Sample entries weighted by QD fitness (softmax).

        Args:
            n: Number of entries to sample
            temperature: Sampling temperature (higher = more uniform)

        Returns:
            List of sampled entries
        """
        if len(self.entries) <= n:
            return self.entries.copy()

        # Compute softmax weights
        qd_scores = torch.tensor([e.qd_fitness for e in self.entries])
        weights = torch.softmax(qd_scores / temperature, dim=0)

        # Sample without replacement
        indices = torch.multinomial(weights, n, replacement=False)
        return [self.entries[i] for i in indices.tolist()]

    def get_statistics(self) -> dict:
        """Get archive statistics for monitoring."""
        if not self.entries:
            return {
                "size": 0,
                "mean_fitness": 0.0,
                "max_fitness": 0.0,
                "min_fitness": 0.0,
                "mean_qd_fitness": 0.0,
                "coverage": 0.0,
                "mean_generation": 0.0,
            }

        fitnesses = [e.fitness for e in self.entries]
        qd_fitnesses = [e.qd_fitness for e in self.entries]
        generations = [e.generation for e in self.entries]

        # Compute coverage (spread in BD space using variance)
        if len(self._bd_cache) > 1:
            device = self._bd_cache[0].device
            all_bds = torch.stack([bd.to(device) for bd in self._bd_cache])
            coverage = all_bds.var(dim=0).sum().item()
        else:
            coverage = 0.0

        return {
            "size": len(self.entries),
            "mean_fitness": sum(fitnesses) / len(fitnesses),
            "max_fitness": max(fitnesses),
            "min_fitness": min(fitnesses),
            "mean_qd_fitness": sum(qd_fitnesses) / len(qd_fitnesses),
            "coverage": coverage,
            "mean_generation": sum(generations) / len(generations),
        }

    def get_coverage_by_dimension(self) -> dict[int, tuple[float, float]]:
        """
        Get min/max range for each BD dimension.

        Useful for understanding which regions of BD space are covered.
        """
        if not self._bd_cache:
            return {}

        device = self._bd_cache[0].device
        all_bds = torch.stack([bd.to(device) for bd in self._bd_cache])
        bd_dim = all_bds.shape[1]

        coverage = {}
        for d in range(bd_dim):
            min_val = all_bds[:, d].min().item()
            max_val = all_bds[:, d].max().item()
            coverage[d] = (min_val, max_val)

        return coverage

    def clear(self) -> None:
        """Clear the archive."""
        self.entries.clear()
        self._bd_cache.clear()

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[ArchiveEntry]:
        return iter(self.entries)

    def __getitem__(self, idx: int) -> ArchiveEntry:
        return self.entries[idx]

    def __repr__(self) -> str:
        return f"DNSArchive(size={len(self)}/{self.max_size}, threshold={self.domination_threshold})"


class MapElitesArchive:
    """
    MAP-Elites style grid-based archive.

    Divides the BD space into a grid and stores the best solution
    in each cell. Simpler than DNS but requires predefined boundaries.

    Note: Primarily included for comparison. DNS is recommended for
    high-dimensional BD spaces.
    """

    def __init__(
        self,
        bd_dim: int,
        grid_size: int = 10,
        bd_bounds: tuple[float, float] = (0.0, 1.0),
    ):
        self.bd_dim = bd_dim
        self.grid_size = grid_size
        self.bd_bounds = bd_bounds

        # Grid storage: maps grid cell tuple to ArchiveEntry
        self.grid: dict[tuple[int, ...], ArchiveEntry] = {}

    def _bd_to_cell(self, bd: Tensor) -> tuple[int, ...]:
        """Convert BD to grid cell coordinates."""
        bd_np = bd.cpu().numpy()
        min_val, max_val = self.bd_bounds
        normalized = (bd_np - min_val) / (max_val - min_val)
        cell = tuple(
            min(self.grid_size - 1, max(0, int(v * self.grid_size)))
            for v in normalized
        )
        return cell

    def try_add(
        self,
        latent: Tensor,
        bd: Tensor,
        fitness: float,
        qd_fitness: float,
        generation: int,
        metadata: dict | None = None,
    ) -> tuple[bool, str]:
        """Try to add to the appropriate grid cell."""
        cell = self._bd_to_cell(bd)

        entry = ArchiveEntry(
            latent=latent.clone().detach(),
            bd=bd.clone().detach(),
            fitness=fitness,
            qd_fitness=qd_fitness,
            generation=generation,
            metadata=metadata or {},
        )

        if cell not in self.grid or fitness > self.grid[cell].fitness:
            self.grid[cell] = entry
            return True, "added"
        else:
            return False, "cell_occupied_by_better"

    def get_bds(self) -> list[Tensor]:
        return [e.bd for e in self.grid.values()]

    def get_entries(self) -> list[ArchiveEntry]:
        return list(self.grid.values())

    def get_statistics(self) -> dict:
        if not self.grid:
            return {"size": 0, "coverage": 0.0}

        total_cells = self.grid_size ** self.bd_dim
        return {
            "size": len(self.grid),
            "coverage": len(self.grid) / total_cells,
            "mean_fitness": sum(e.fitness for e in self.grid.values()) / len(self.grid),
        }

    def clear(self) -> None:
        self.grid.clear()

    def __len__(self) -> int:
        return len(self.grid)
