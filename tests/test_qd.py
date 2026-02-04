"""Tests for Quality Diversity (QD) module."""

import pytest
import torch
from torch import Tensor

from latent_reasoning.config import QDConfig
from latent_reasoning.qd.behavior import (
    RFFProjector,
    BehaviorComputer,
    BehaviorDescriptor,
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
from latent_reasoning.qd.manager import QDManager, create_qd_manager


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def latent_dim():
    """Standard latent dimension for tests."""
    return 256


@pytest.fixture
def bd_dim():
    """Standard behavioral descriptor dimension."""
    return 16


@pytest.fixture
def device():
    """Device for tensor operations."""
    return torch.device("cpu")


@pytest.fixture
def sample_latents(latent_dim, device):
    """Generate sample latent vectors for testing."""
    torch.manual_seed(42)
    return [torch.randn(latent_dim, device=device) for _ in range(10)]


@pytest.fixture
def qd_config():
    """Standard QD configuration for tests."""
    return QDConfig(
        enabled=True,
        bd_dim=16,
        rff_gamma=0.1,
        novelty_k=5,
        novelty_weight=0.3,
        archive_size=100,
        domination_threshold=0.1,
    )


# =============================================================================
# RFFProjector Tests
# =============================================================================

class TestRFFProjector:
    """Tests for Random Fourier Features projector."""

    def test_initialization(self, latent_dim, bd_dim, device):
        """Test projector initializes with correct dimensions."""
        projector = RFFProjector(
            input_dim=latent_dim,
            output_dim=bd_dim,
            gamma=0.1,
            device=device,
        )
        assert projector.input_dim == latent_dim
        assert projector.output_dim == bd_dim
        # W is output_dim // 2 because RFF uses cos + sin pairs
        assert projector.W.shape == (latent_dim, bd_dim // 2)
        assert projector.b.shape == (bd_dim // 2,)

    def test_project_single(self, latent_dim, bd_dim, device):
        """Test projection of single latent vector."""
        projector = RFFProjector(latent_dim, bd_dim, device=device)
        latent = torch.randn(latent_dim, device=device)

        result = projector.project(latent)

        assert result.shape == (bd_dim,)
        assert result.device == device
        # RFF output should be bounded due to cos function
        assert result.abs().max() <= 2.0  # sqrt(2/bd_dim) * bd_dim

    def test_project_batch(self, latent_dim, bd_dim, device):
        """Test projection of batch of latents (project handles both)."""
        projector = RFFProjector(latent_dim, bd_dim, device=device)
        latents = torch.randn(5, latent_dim, device=device)

        # project() handles batched input
        result = projector.project(latents)

        assert result.shape == (5, bd_dim)

    def test_deterministic_with_seed(self, latent_dim, bd_dim, device):
        """Test that same seed produces same projections."""
        latent = torch.randn(latent_dim, device=device)

        proj1 = RFFProjector(latent_dim, bd_dim, seed=123, device=device)
        proj2 = RFFProjector(latent_dim, bd_dim, seed=123, device=device)

        result1 = proj1.project(latent)
        result2 = proj2.project(latent)

        assert torch.allclose(result1, result2)

    def test_different_seeds_different_projections(self, latent_dim, bd_dim, device):
        """Test that different seeds produce different projections."""
        latent = torch.randn(latent_dim, device=device)

        proj1 = RFFProjector(latent_dim, bd_dim, seed=123, device=device)
        proj2 = RFFProjector(latent_dim, bd_dim, seed=456, device=device)

        result1 = proj1.project(latent)
        result2 = proj2.project(latent)

        assert not torch.allclose(result1, result2)


# =============================================================================
# BehaviorComputer Tests
# =============================================================================

class TestBehaviorComputer:
    """Tests for behavioral descriptor computation."""

    def test_initialization(self, latent_dim, bd_dim, device):
        """Test behavior computer initializes correctly."""
        computer = BehaviorComputer(
            latent_dim=latent_dim,
            bd_dim=bd_dim,
            device=device,
        )
        assert computer.bd_dim == bd_dim

    def test_compute_bd_minimal(self, latent_dim, bd_dim, device):
        """Test computing BD with minimal inputs."""
        computer = BehaviorComputer(latent_dim, bd_dim, device=device)
        latent = torch.randn(latent_dim, device=device)

        bd = computer.compute(latent, generation=0)

        assert isinstance(bd, BehaviorDescriptor)
        assert bd.vector.shape == (bd_dim,)
        # BD has component tensors, not generation
        assert bd.latent_component is not None
        assert bd.structural_component is not None
        assert bd.trajectory_component is not None

    def test_compute_bd_with_history(self, latent_dim, bd_dim, device):
        """Test computing BD with trajectory history."""
        computer = BehaviorComputer(latent_dim, bd_dim, device=device)
        latent = torch.randn(latent_dim, device=device)
        history = [torch.randn(latent_dim, device=device) for _ in range(3)]

        bd = computer.compute(latent, generation=5, history=history)

        assert isinstance(bd, BehaviorDescriptor)
        assert bd.vector.shape == (bd_dim,)
        # BD has component tensors
        assert bd.latent_component is not None

    def test_different_latents_different_bds(self, latent_dim, bd_dim, device):
        """Test that different latents produce different BDs."""
        computer = BehaviorComputer(latent_dim, bd_dim, device=device)
        latent1 = torch.randn(latent_dim, device=device)
        latent2 = torch.randn(latent_dim, device=device)

        bd1 = computer.compute(latent1, generation=0)
        bd2 = computer.compute(latent2, generation=0)

        # BDs should be different for different latents
        assert not torch.allclose(bd1.vector, bd2.vector)


# =============================================================================
# NoveltyComputer Tests
# =============================================================================

class TestNoveltyComputer:
    """Tests for novelty computation."""

    def test_initialization(self):
        """Test novelty computer initializes with correct parameters."""
        computer = NoveltyComputer(k=10, distance_metric="euclidean")
        assert computer.k == 10
        assert computer.distance_metric == "euclidean"

    def test_invalid_k_raises(self):
        """Test that invalid k value raises error."""
        with pytest.raises(ValueError):
            NoveltyComputer(k=0)

    def test_novelty_empty_archive(self, bd_dim, device):
        """Test novelty is maximal with empty archive."""
        computer = NoveltyComputer(k=5)
        bd = torch.randn(bd_dim, device=device)

        novelty = computer.compute_novelty(bd, archive_bds=[])

        assert novelty == 1.0  # Maximum novelty for empty archive

    def test_novelty_single_element(self, bd_dim, device):
        """Test novelty with single archive element."""
        computer = NoveltyComputer(k=5)
        bd = torch.randn(bd_dim, device=device)
        archive_bd = torch.randn(bd_dim, device=device)

        novelty = computer.compute_novelty(bd, archive_bds=[archive_bd])

        assert novelty > 0  # Should have some novelty

    def test_novelty_identical_bd_is_low(self, bd_dim, device):
        """Test that identical BD has low novelty."""
        computer = NoveltyComputer(k=5)
        bd = torch.randn(bd_dim, device=device)
        archive_bds = [bd.clone() for _ in range(5)]

        novelty = computer.compute_novelty(bd, archive_bds=archive_bds)

        assert novelty < 0.01  # Should be very low (essentially zero)

    def test_novelty_batch(self, bd_dim, device):
        """Test batch novelty computation."""
        computer = NoveltyComputer(k=3)
        bds = [torch.randn(bd_dim, device=device) for _ in range(5)]
        archive_bds = [torch.randn(bd_dim, device=device) for _ in range(10)]

        novelty_scores = computer.compute_novelty_batch(bds, archive_bds)

        assert len(novelty_scores) == 5
        assert all(n >= 0 for n in novelty_scores)

    def test_cosine_distance(self, bd_dim, device):
        """Test novelty with cosine distance metric."""
        computer = NoveltyComputer(k=3, distance_metric="cosine")
        bds = [torch.randn(bd_dim, device=device) for _ in range(3)]
        archive_bds = [torch.randn(bd_dim, device=device) for _ in range(5)]

        novelty_scores = computer.compute_novelty_batch(bds, archive_bds)

        assert len(novelty_scores) == 3


class TestCombineFitnessNovelty:
    """Tests for fitness-novelty combination."""

    def test_pure_fitness(self):
        """Test alpha=0 gives pure fitness."""
        result = combine_fitness_novelty(0.8, 0.5, alpha=0.0)
        assert abs(result - 0.8) < 1e-6

    def test_pure_novelty(self):
        """Test alpha=1 gives pure novelty."""
        result = combine_fitness_novelty(0.8, 0.5, alpha=1.0)
        assert abs(result - 0.5) < 1e-6

    def test_balanced(self):
        """Test alpha=0.5 gives balanced combination."""
        result = combine_fitness_novelty(0.8, 0.4, alpha=0.5)
        expected = 0.5 * 0.8 + 0.5 * 0.4  # 0.6
        assert abs(result - expected) < 1e-6

    def test_typical_alpha(self):
        """Test typical alpha=0.3 value."""
        result = combine_fitness_novelty(1.0, 0.5, alpha=0.3)
        expected = 0.7 * 1.0 + 0.3 * 0.5  # 0.85
        assert abs(result - expected) < 1e-6


class TestNormalizeNovelty:
    """Tests for novelty normalization."""

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        scores = [0.2, 0.5, 0.8, 1.0]
        normalized = normalize_novelty_scores(scores, method="minmax")

        assert abs(min(normalized)) < 1e-6  # Min should be 0
        assert abs(max(normalized) - 1.0) < 1e-6  # Max should be 1

    def test_empty_list(self):
        """Test empty list returns empty."""
        assert normalize_novelty_scores([]) == []

    def test_constant_values(self):
        """Test constant values get normalized to 0.5."""
        scores = [0.5, 0.5, 0.5]
        normalized = normalize_novelty_scores(scores)
        assert all(abs(n - 0.5) < 1e-6 for n in normalized)


# =============================================================================
# DNSArchive Tests
# =============================================================================

class TestDNSArchive:
    """Tests for Dominated Novelty Search archive."""

    def test_initialization(self):
        """Test archive initializes with correct parameters."""
        archive = DNSArchive(max_size=100, domination_threshold=0.1)
        assert archive.max_size == 100
        assert len(archive) == 0

    def test_add_first_entry(self, latent_dim, bd_dim, device):
        """Test adding first entry to empty archive."""
        archive = DNSArchive(max_size=100)
        latent = torch.randn(latent_dim, device=device)
        bd = torch.randn(bd_dim, device=device)

        added, reason = archive.try_add(
            latent=latent,
            bd=bd,
            fitness=0.8,
            qd_fitness=0.9,
            generation=0,
        )

        assert added is True
        assert reason == "added"
        assert len(archive) == 1

    def test_dominated_entry_rejected(self, latent_dim, bd_dim, device):
        """Test that dominated entries are rejected."""
        archive = DNSArchive(max_size=100, domination_threshold=0.5)

        # Add high-fitness entry
        latent1 = torch.randn(latent_dim, device=device)
        bd1 = torch.zeros(bd_dim, device=device)  # Fixed BD for testing
        archive.try_add(latent1, bd1, fitness=0.9, qd_fitness=0.9, generation=0)

        # Try to add lower-fitness entry with similar BD
        latent2 = torch.randn(latent_dim, device=device)
        bd2 = torch.zeros(bd_dim, device=device) + 0.01  # Very similar BD
        added, reason = archive.try_add(latent2, bd2, fitness=0.5, qd_fitness=0.5, generation=1)

        assert added is False
        assert "dominated" in reason

    def test_dominating_entry_replaces(self, latent_dim, bd_dim, device):
        """Test that dominating entries replace existing."""
        archive = DNSArchive(max_size=100, domination_threshold=0.5)

        # Add low-fitness entry
        latent1 = torch.randn(latent_dim, device=device)
        bd1 = torch.zeros(bd_dim, device=device)
        archive.try_add(latent1, bd1, fitness=0.5, qd_fitness=0.5, generation=0)
        assert len(archive) == 1

        # Add higher-fitness entry with similar BD
        latent2 = torch.randn(latent_dim, device=device)
        bd2 = torch.zeros(bd_dim, device=device) + 0.01
        added, reason = archive.try_add(latent2, bd2, fitness=0.9, qd_fitness=0.9, generation=1)

        assert added is True
        assert len(archive) == 1  # Should have replaced, not added

    def test_diverse_entries_coexist(self, latent_dim, bd_dim, device):
        """Test that diverse entries can coexist."""
        archive = DNSArchive(max_size=100, domination_threshold=0.1)

        # Add entries with very different BDs
        for i in range(5):
            latent = torch.randn(latent_dim, device=device)
            bd = torch.zeros(bd_dim, device=device)
            bd[i % bd_dim] = 10.0  # Very different BDs
            archive.try_add(latent, bd, fitness=0.5 + i * 0.05, qd_fitness=0.5, generation=i)

        assert len(archive) == 5  # All should be added

    def test_max_size_enforcement(self, latent_dim, bd_dim, device):
        """Test that archive doesn't exceed max size."""
        max_size = 5
        archive = DNSArchive(max_size=max_size, domination_threshold=0.01)

        # Add more entries than max_size with diverse BDs
        for i in range(10):
            latent = torch.randn(latent_dim, device=device)
            bd = torch.randn(bd_dim, device=device) * 10  # Ensure diverse
            archive.try_add(latent, bd, fitness=0.1 * i, qd_fitness=0.1 * i, generation=i)

        assert len(archive) <= max_size

    def test_get_best(self, latent_dim, bd_dim, device):
        """Test getting best entries by fitness."""
        archive = DNSArchive(max_size=100, domination_threshold=0.01)

        # Add entries with different fitness
        for i in range(5):
            latent = torch.randn(latent_dim, device=device)
            bd = torch.randn(bd_dim, device=device) * 10
            archive.try_add(latent, bd, fitness=0.1 * (i + 1), qd_fitness=0.5, generation=i)

        best = archive.get_best(n=2)

        assert len(best) == 2
        assert best[0].fitness >= best[1].fitness  # Sorted by fitness

    def test_sample_diverse(self, latent_dim, bd_dim, device):
        """Test diverse sampling from archive."""
        archive = DNSArchive(max_size=100, domination_threshold=0.01)

        for i in range(10):
            latent = torch.randn(latent_dim, device=device)
            bd = torch.randn(bd_dim, device=device) * 10
            archive.try_add(latent, bd, fitness=0.5, qd_fitness=0.5, generation=i)

        diverse = archive.sample_diverse(n=3)

        assert len(diverse) == 3

    def test_get_statistics(self, latent_dim, bd_dim, device):
        """Test archive statistics computation."""
        archive = DNSArchive(max_size=100)

        # Empty archive
        stats = archive.get_statistics()
        assert stats["size"] == 0

        # Add entries
        for i in range(5):
            latent = torch.randn(latent_dim, device=device)
            bd = torch.randn(bd_dim, device=device) * 10
            archive.try_add(latent, bd, fitness=0.1 * (i + 1), qd_fitness=0.5, generation=i)

        stats = archive.get_statistics()
        assert stats["size"] == 5
        assert "mean_fitness" in stats
        assert "max_fitness" in stats
        assert "coverage" in stats


# =============================================================================
# MapElitesArchive Tests
# =============================================================================

class TestMapElitesArchive:
    """Tests for MAP-Elites archive."""

    def test_initialization(self, bd_dim):
        """Test archive initializes correctly."""
        archive = MapElitesArchive(bd_dim=bd_dim, grid_size=10)
        assert archive.bd_dim == bd_dim
        assert len(archive) == 0

    def test_add_entry(self, latent_dim, bd_dim, device):
        """Test adding entry to grid cell."""
        archive = MapElitesArchive(bd_dim=bd_dim, grid_size=10)
        latent = torch.randn(latent_dim, device=device)
        bd = torch.rand(bd_dim, device=device)  # In [0, 1] range

        added, reason = archive.try_add(latent, bd, fitness=0.8, qd_fitness=0.9, generation=0)

        assert added is True
        assert len(archive) == 1

    def test_better_fitness_replaces(self, latent_dim, bd_dim, device):
        """Test that better fitness replaces cell occupant."""
        archive = MapElitesArchive(bd_dim=bd_dim, grid_size=10)
        bd = torch.ones(bd_dim, device=device) * 0.5  # Same cell

        # Add first entry
        latent1 = torch.randn(latent_dim, device=device)
        archive.try_add(latent1, bd, fitness=0.5, qd_fitness=0.5, generation=0)

        # Add better entry to same cell
        latent2 = torch.randn(latent_dim, device=device)
        added, reason = archive.try_add(latent2, bd.clone(), fitness=0.8, qd_fitness=0.8, generation=1)

        assert added is True
        assert len(archive) == 1


# =============================================================================
# QDManager Tests
# =============================================================================

class TestQDManager:
    """Tests for QD Manager orchestration."""

    def test_initialization(self, qd_config, latent_dim, device):
        """Test manager initializes all components."""
        manager = QDManager(qd_config, latent_dim, device=device)

        assert manager.config == qd_config
        assert manager.latent_dim == latent_dim
        assert manager.behavior_computer is not None
        assert manager.novelty_computer is not None
        assert manager.archive is not None

    def test_compute_bds(self, qd_config, latent_dim, device):
        """Test computing BDs for chains."""
        from latent_reasoning.core.chain import ChainState

        manager = QDManager(qd_config, latent_dim, device=device)
        chains = [ChainState(latent=torch.randn(latent_dim, device=device)) for _ in range(5)]

        bds = manager.compute_bds(chains)

        assert len(bds) == 5
        assert all(isinstance(bd, BehaviorDescriptor) for bd in bds)
        assert all(bd.vector.shape == (qd_config.bd_dim,) for bd in bds)

    def test_compute_novelty(self, qd_config, latent_dim, device):
        """Test computing novelty scores."""
        manager = QDManager(qd_config, latent_dim, device=device)

        # Create BDs with proper structure
        bds = [
            BehaviorDescriptor(
                vector=torch.randn(qd_config.bd_dim, device=device),
                latent_component=torch.randn(8, device=device),
                structural_component=torch.randn(4, device=device),
                trajectory_component=torch.randn(4, device=device),
            )
            for _ in range(5)
        ]

        novelty_scores = manager.compute_novelty(bds)

        assert len(novelty_scores) == 5
        # With empty archive, all should have max novelty
        assert all(n == 1.0 for n in novelty_scores)

    def test_combine_fitness(self, qd_config, latent_dim, device):
        """Test fitness-novelty combination."""
        manager = QDManager(qd_config, latent_dim, device=device)

        raw_scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        novelty_scores = [1.0, 0.8, 0.6, 0.4, 0.2]

        qd_scores = manager.combine_fitness(raw_scores, novelty_scores)

        assert len(qd_scores) == 5
        # With alpha=0.3: qd = 0.7*fitness + 0.3*novelty
        expected_first = 0.7 * 0.5 + 0.3 * 1.0  # 0.65
        assert abs(qd_scores[0] - expected_first) < 1e-6

    def test_update_archive(self, qd_config, latent_dim, device):
        """Test archive update."""
        from latent_reasoning.core.chain import ChainState

        manager = QDManager(qd_config, latent_dim, device=device)
        chains = [ChainState(latent=torch.randn(latent_dim, device=device)) for _ in range(5)]
        bds = manager.compute_bds(chains)
        raw_scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        qd_scores = [0.5, 0.6, 0.7, 0.8, 0.9]

        added, rejected = manager.update_archive(
            chains=chains,
            bds=bds,
            raw_scores=raw_scores,
            qd_scores=qd_scores,
            generation=0,
        )

        assert added >= 0
        assert rejected >= 0
        assert added + rejected == 5

    def test_sample_parents(self, qd_config, latent_dim, device):
        """Test sampling parents from archive."""
        from latent_reasoning.core.chain import ChainState

        manager = QDManager(qd_config, latent_dim, device=device)

        # Empty archive
        parents = manager.sample_parents(n=3)
        assert parents == []

        # Add entries
        chains = [ChainState(latent=torch.randn(latent_dim, device=device)) for _ in range(10)]
        bds = manager.compute_bds(chains)
        raw_scores = [0.1 * i for i in range(10)]
        qd_scores = raw_scores
        manager.update_archive(chains, bds, raw_scores, qd_scores, generation=0)

        # Sample parents
        parents = manager.sample_parents(n=3, method="diverse")
        assert len(parents) <= 3

    def test_get_archive_statistics(self, qd_config, latent_dim, device):
        """Test getting archive statistics."""
        manager = QDManager(qd_config, latent_dim, device=device)

        stats = manager.get_archive_statistics()

        assert "size" in stats
        assert "total_added" in stats
        assert "total_rejected" in stats

    def test_reset(self, qd_config, latent_dim, device):
        """Test resetting manager state."""
        from latent_reasoning.core.chain import ChainState

        manager = QDManager(qd_config, latent_dim, device=device)

        # Add some entries
        chains = [ChainState(latent=torch.randn(latent_dim, device=device)) for _ in range(5)]
        bds = manager.compute_bds(chains)
        manager.update_archive(chains, bds, [0.5] * 5, [0.5] * 5, generation=0)

        assert len(manager.archive) > 0

        # Reset
        manager.reset()

        assert len(manager.archive) == 0
        assert manager._generation == 0
        assert manager._total_added == 0


class TestCreateQDManager:
    """Tests for QD manager factory function."""

    def test_disabled_returns_none(self, latent_dim):
        """Test that disabled config returns None."""
        config = QDConfig(enabled=False)
        manager = create_qd_manager(config, latent_dim)
        assert manager is None

    def test_enabled_returns_manager(self, latent_dim):
        """Test that enabled config returns manager."""
        config = QDConfig(enabled=True)
        manager = create_qd_manager(config, latent_dim)
        assert manager is not None
        assert isinstance(manager, QDManager)


# =============================================================================
# Integration Tests
# =============================================================================

class TestQDIntegration:
    """Integration tests for QD pipeline."""

    def test_full_qd_pipeline(self, qd_config, latent_dim, device):
        """Test complete QD pipeline from latents to archive."""
        from latent_reasoning.core.chain import ChainState

        # Initialize
        manager = QDManager(qd_config, latent_dim, device=device)

        # Simulate multiple generations
        for gen in range(5):
            # Create population
            chains = [
                ChainState(latent=torch.randn(latent_dim, device=device), generation=gen)
                for _ in range(10)
            ]

            # Compute BDs
            bds = manager.compute_bds(chains)

            # Compute novelty
            novelty = manager.compute_novelty(bds)

            # Simulate raw fitness scores
            raw_scores = [torch.rand(1).item() for _ in chains]

            # Combine fitness
            qd_scores = manager.combine_fitness(raw_scores, novelty)

            # Update archive
            added, rejected = manager.update_archive(
                chains, bds, raw_scores, qd_scores, gen
            )

            # Step generation
            manager.step_generation()

        # Verify archive has entries
        assert len(manager.archive) > 0

        # Verify statistics
        stats = manager.get_archive_statistics()
        assert stats["total_added"] > 0

    def test_qd_improves_diversity(self, latent_dim, device):
        """Test that QD maintains diversity in archive via domination."""
        from latent_reasoning.core.chain import ChainState

        # Use high domination threshold so similar solutions dominate each other
        config = QDConfig(
            enabled=True,
            bd_dim=16,
            domination_threshold=0.5,  # High threshold for domination
            archive_size=100,
        )
        manager = QDManager(config, latent_dim, device=device)

        # Add many solutions with varying fitness in same BD region
        base_latent = torch.randn(latent_dim, device=device)
        for i in range(20):
            # Small perturbation - BDs will be similar
            latent = base_latent + torch.randn(latent_dim, device=device) * 0.001
            chain = ChainState(latent=latent)
            bd = manager.compute_bds([chain])[0]
            # Increasing fitness - later solutions should dominate earlier
            fitness = 0.1 + i * 0.04
            manager.update_archive([chain], [bd], [fitness], [fitness], generation=i)

        # Archive should have fewer than 20 due to domination
        # (higher fitness solutions replace lower fitness in same BD region)
        assert len(manager.archive) < 20
