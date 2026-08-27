"""Tests for Autopoietic Judge module."""

import pytest
import torch
from torch import Tensor

from latent_reasoning.config import AutopoieticJudgeConfig
from latent_reasoning.core.autopoietic.experience_buffer import (
    ExperienceBuffer,
    ExperienceEntry,
)
from latent_reasoning.core.autopoietic.homeostasis import (
    HomeostasisController,
)
from latent_reasoning.core.autopoietic.external_evaluator import (
    MockExternalEvaluator,
    ExternalScore,
    create_external_evaluator,
)
from latent_reasoning.core.autopoietic.autopoietic_judge import (
    AutopoieticJudge,
)
from latent_reasoning.core.autopoietic.autopoietic_panel import (
    AutopoieticPanel,
    create_autopoietic_panel,
    Verdict,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def latent_dim():
    """Standard latent dimension for tests."""
    return 256


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
def autopoietic_config():
    """Standard autopoietic configuration for tests."""
    return AutopoieticJudgeConfig(
        enabled=True,
        judge_update_freq=5,
        external_sample_size=3,
        target_diversity=0.4,
        homeostasis_k=0.1,
        ema_decay=0.99,
        buffer_size=100,
        initial_internal_trust=0.3,
        max_internal_trust=0.9,
        trust_growth_rate=0.05,
        correlation_threshold=0.3,
    )


@pytest.fixture
def mock_scorer():
    """Mock internal scorer that returns random-ish scores."""
    def scorer(latent: Tensor) -> float:
        # Simple scorer based on latent norm
        return 0.5 + 0.1 * (latent.norm().item() % 1)
    return scorer


@pytest.fixture
def mock_decoder():
    """Mock decoder that returns placeholder text."""
    def decoder(latent: Tensor, query: str) -> str:
        return f"Response to '{query}' with latent norm {latent.norm().item():.2f}"
    return decoder


# =============================================================================
# ExperienceBuffer Tests
# =============================================================================

class TestExperienceBuffer:
    """Tests for experience buffer."""

    def test_initialization(self, device):
        """Test buffer initializes correctly."""
        buffer = ExperienceBuffer(max_size=100, device=device)
        assert buffer.max_size == 100
        assert len(buffer) == 0

    def test_add_entry(self, latent_dim, device):
        """Test adding entries to buffer."""
        buffer = ExperienceBuffer(max_size=100, device=device)
        latent = torch.randn(latent_dim, device=device)

        idx = buffer.add(latent, internal_score=0.7, query="test")

        assert len(buffer) == 1
        assert idx == 0
        assert buffer[0].internal_score == 0.7
        assert buffer[0].query == "test"
        assert not buffer[0].has_external

    def test_update_external(self, latent_dim, device):
        """Test updating external score."""
        buffer = ExperienceBuffer(max_size=100, device=device)
        latent = torch.randn(latent_dim, device=device)

        idx = buffer.add(latent, internal_score=0.7)
        buffer.update_external(idx, external_score=0.8)

        assert buffer[idx].has_external
        assert buffer[idx].external_score == 0.8
        assert buffer[idx].discrepancy == pytest.approx(0.1)

    def test_ring_buffer_behavior(self, latent_dim, device):
        """Test ring buffer overwrites oldest entries."""
        buffer = ExperienceBuffer(max_size=5, device=device)

        # Add 7 entries to a buffer of size 5
        for i in range(7):
            latent = torch.randn(latent_dim, device=device)
            buffer.add(latent, internal_score=0.1 * i)

        assert len(buffer) == 5
        # Oldest entries should have been overwritten

    def test_sample_random(self, latent_dim, device):
        """Test random sampling."""
        buffer = ExperienceBuffer(max_size=100, device=device)

        for i in range(10):
            latent = torch.randn(latent_dim, device=device)
            buffer.add(latent, internal_score=0.1 * i)

        samples = buffer.sample(n=5)
        assert len(samples) == 5

    def test_sample_with_external_only(self, latent_dim, device):
        """Test sampling only grounded entries."""
        buffer = ExperienceBuffer(max_size=100, device=device)

        # Add 5 entries, ground only 2
        for i in range(5):
            latent = torch.randn(latent_dim, device=device)
            idx = buffer.add(latent, internal_score=0.5)
            if i < 2:
                buffer.update_external(idx, external_score=0.6)

        grounded_samples = buffer.sample(n=10, with_external_only=True)
        assert len(grounded_samples) == 2
        assert all(s.has_external for s in grounded_samples)

    def test_compute_correlation(self, latent_dim, device):
        """Test correlation computation."""
        buffer = ExperienceBuffer(max_size=100, device=device)

        # Add entries with correlated internal/external scores
        for i in range(10):
            latent = torch.randn(latent_dim, device=device)
            internal = 0.1 * i
            external = 0.1 * i + 0.05  # Highly correlated
            idx = buffer.add(latent, internal_score=internal)
            buffer.update_external(idx, external_score=external)

        correlation = buffer.compute_correlation()
        assert correlation is not None
        assert correlation > 0.9  # Should be highly correlated

    def test_get_statistics(self, latent_dim, device):
        """Test statistics computation."""
        buffer = ExperienceBuffer(max_size=100, device=device)

        for i in range(5):
            latent = torch.randn(latent_dim, device=device)
            idx = buffer.add(latent, internal_score=0.5)
            if i < 3:
                buffer.update_external(idx, external_score=0.6)

        stats = buffer.get_statistics()
        assert stats["size"] == 5
        assert stats["grounded_count"] == 3
        assert "mean_discrepancy" in stats


# =============================================================================
# HomeostasisController Tests
# =============================================================================

class TestHomeostasisController:
    """Tests for homeostatic temperature control."""

    def test_initialization(self):
        """Test controller initializes correctly."""
        controller = HomeostasisController(
            target_diversity=0.4,
            control_gain=0.1,
            min_temperature=0.1,
            max_temperature=2.0,
            initial_temperature=0.5,
        )
        assert controller.temperature == 0.5
        assert controller.target_diversity == 0.4

    def test_update_increases_temp_when_diversity_low(self):
        """Test temperature increases when diversity is below target."""
        controller = HomeostasisController(
            target_diversity=0.5,
            control_gain=0.5,
            initial_temperature=0.5,
        )

        # Low diversity should increase temperature
        new_temp = controller.update(diversity=0.2, generation=1)
        assert new_temp > 0.5

    def test_update_decreases_temp_when_diversity_high(self):
        """Test temperature decreases when diversity is above target."""
        controller = HomeostasisController(
            target_diversity=0.3,
            control_gain=0.5,
            initial_temperature=0.5,
        )

        # High diversity should decrease temperature
        new_temp = controller.update(diversity=0.7, generation=1)
        assert new_temp < 0.5

    def test_temperature_bounded(self):
        """Test temperature stays within bounds."""
        controller = HomeostasisController(
            target_diversity=0.5,
            control_gain=10.0,  # Very aggressive
            min_temperature=0.1,
            max_temperature=2.0,
            initial_temperature=0.5,
        )

        # Try to push temp very low
        new_temp = controller.update(diversity=0.9)
        assert new_temp >= 0.1

        # Try to push temp very high
        new_temp = controller.update(diversity=0.01)
        assert new_temp <= 2.0

    def test_compute_diversity(self, latent_dim, device):
        """Test diversity computation from latents."""
        controller = HomeostasisController()

        # Identical latents should have low diversity
        latent = torch.randn(latent_dim, device=device)
        identical_latents = [latent.clone() for _ in range(5)]
        diversity = controller.compute_diversity(latents=identical_latents)
        assert diversity < 0.01

        # Random latents should have moderate diversity
        random_latents = [torch.randn(latent_dim, device=device) for _ in range(5)]
        diversity = controller.compute_diversity(latents=random_latents)
        assert diversity > 0.1

    def test_get_statistics(self):
        """Test statistics retrieval."""
        controller = HomeostasisController(initial_temperature=0.5)
        controller.update(0.3, generation=1)
        controller.update(0.4, generation=2)

        stats = controller.get_statistics()
        assert stats["temperature"] == controller.temperature
        assert stats["updates"] == 2


# =============================================================================
# MockExternalEvaluator Tests
# =============================================================================

class TestMockExternalEvaluator:
    """Tests for mock external evaluator."""

    def test_evaluate(self):
        """Test mock evaluation."""
        evaluator = MockExternalEvaluator(base_score=0.6)
        result = evaluator.evaluate("What is AI?", "AI is artificial intelligence.")

        assert isinstance(result, ExternalScore)
        assert result.is_valid
        assert 0 < result.score < 1

    def test_evaluate_batch(self):
        """Test batch evaluation."""
        evaluator = MockExternalEvaluator()
        pairs = [
            ("Q1", "A1"),
            ("Q2", "A2"),
            ("Q3", "A3"),
        ]

        results = evaluator.evaluate_batch(pairs)
        assert len(results) == 3
        assert all(r.is_valid for r in results)


class TestCreateExternalEvaluator:
    """Tests for external evaluator factory."""

    def test_create_mock(self):
        """Test creating mock evaluator."""
        evaluator = create_external_evaluator(mock=True)
        assert isinstance(evaluator, MockExternalEvaluator)

    def test_fallback_to_mock_without_api_key(self, monkeypatch):
        """Test fallback to mock when no API key."""
        # Remove API key
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        # Should fall back to mock with warning
        import warnings
        with warnings.catch_warnings(record=True):
            evaluator = create_external_evaluator(mock=False)
            assert isinstance(evaluator, MockExternalEvaluator)


# =============================================================================
# AutopoieticJudge Tests
# =============================================================================

class TestAutopoieticJudge:
    """Tests for autopoietic judge."""

    def test_initialization(self, autopoietic_config, mock_scorer, mock_decoder, device):
        """Test judge initializes correctly."""
        evaluator = MockExternalEvaluator()
        judge = AutopoieticJudge(
            config=autopoietic_config,
            internal_scorer=mock_scorer,
            external_evaluator=evaluator,
            decoder=mock_decoder,
            device=device,
        )

        assert judge.internal_trust == autopoietic_config.initial_internal_trust
        assert judge.correlation is None
        assert len(judge.buffer) == 0

    def test_evaluate_internal_only(self, autopoietic_config, mock_scorer, mock_decoder, latent_dim, device):
        """Test evaluation with internal scorer only."""
        evaluator = MockExternalEvaluator()
        judge = AutopoieticJudge(
            config=autopoietic_config,
            internal_scorer=mock_scorer,
            external_evaluator=evaluator,
            decoder=mock_decoder,
            device=device,
        )

        latent = torch.randn(latent_dim, device=device)
        score = judge.evaluate(latent, query="test", use_external=False)

        assert 0 <= score <= 1
        assert len(judge.buffer) == 1

    def test_evaluate_with_external(self, autopoietic_config, mock_scorer, mock_decoder, latent_dim, device):
        """Test evaluation with external grounding."""
        evaluator = MockExternalEvaluator()
        judge = AutopoieticJudge(
            config=autopoietic_config,
            internal_scorer=mock_scorer,
            external_evaluator=evaluator,
            decoder=mock_decoder,
            device=device,
        )

        latent = torch.randn(latent_dim, device=device)
        score = judge.evaluate(latent, query="test query", use_external=True)

        assert 0 <= score <= 1
        # Buffer entry should have external score
        assert judge.buffer[0].has_external

    def test_ground_updates_trust(self, autopoietic_config, mock_scorer, mock_decoder, latent_dim, device):
        """Test that grounding can update trust."""
        evaluator = MockExternalEvaluator()
        judge = AutopoieticJudge(
            config=autopoietic_config,
            internal_scorer=mock_scorer,
            external_evaluator=evaluator,
            decoder=mock_decoder,
            device=device,
        )

        # Add some entries
        for i in range(10):
            latent = torch.randn(latent_dim, device=device)
            judge.evaluate(latent, query=f"query {i}")

        initial_trust = judge.internal_trust

        # Ground (may or may not increase trust depending on correlation)
        stats = judge.ground(generation=5)
        assert "grounded" in stats

    def test_get_state(self, autopoietic_config, mock_scorer, mock_decoder, device):
        """Test state retrieval."""
        evaluator = MockExternalEvaluator()
        judge = AutopoieticJudge(
            config=autopoietic_config,
            internal_scorer=mock_scorer,
            external_evaluator=evaluator,
            decoder=mock_decoder,
            device=device,
        )

        state = judge.get_state()
        assert state.internal_trust == autopoietic_config.initial_internal_trust
        assert state.buffer_size == 0

    def test_reset(self, autopoietic_config, mock_scorer, mock_decoder, latent_dim, device):
        """Test judge reset."""
        evaluator = MockExternalEvaluator()
        judge = AutopoieticJudge(
            config=autopoietic_config,
            internal_scorer=mock_scorer,
            external_evaluator=evaluator,
            decoder=mock_decoder,
            device=device,
        )

        # Add entries
        for i in range(5):
            latent = torch.randn(latent_dim, device=device)
            judge.evaluate(latent)

        assert len(judge.buffer) == 5

        # Reset
        judge.reset()

        assert len(judge.buffer) == 0
        assert judge.internal_trust == autopoietic_config.initial_internal_trust


# =============================================================================
# AutopoieticPanel Tests
# =============================================================================

class TestAutopoieticPanel:
    """Tests for autopoietic panel integration."""

    def test_initialization(self, autopoietic_config, mock_scorer, mock_decoder, device):
        """Test panel initializes correctly."""
        panel = AutopoieticPanel(
            config=autopoietic_config,
            internal_scorer=mock_scorer,
            decoder=mock_decoder,
            device=device,
            use_mock_external=True,
        )

        assert panel.internal_trust == autopoietic_config.initial_internal_trust
        assert panel.temperature > 0

    def test_evaluate_returns_verdict(self, autopoietic_config, mock_scorer, mock_decoder, latent_dim, device):
        """Test evaluate returns proper Verdict."""
        panel = AutopoieticPanel(
            config=autopoietic_config,
            internal_scorer=mock_scorer,
            decoder=mock_decoder,
            device=device,
            use_mock_external=True,
        )

        latent = torch.randn(latent_dim, device=device)
        verdict = panel.evaluate(latent, context={"query": "test"})

        assert isinstance(verdict, Verdict)
        assert 0 <= verdict.score <= 1

    def test_step_generation(self, autopoietic_config, mock_scorer, mock_decoder, latent_dim, device):
        """Test step_generation updates state."""
        from latent_reasoning.core.chain import ChainState

        panel = AutopoieticPanel(
            config=autopoietic_config,
            internal_scorer=mock_scorer,
            decoder=mock_decoder,
            device=device,
            use_mock_external=True,
        )

        chains = [ChainState(latent=torch.randn(latent_dim, device=device)) for _ in range(10)]
        initial_temp = panel.temperature

        stats = panel.step_generation(chains, generation=1)

        assert "temperature" in stats
        assert "diversity" in stats
        # Temperature should have changed based on diversity

    def test_get_statistics(self, autopoietic_config, mock_scorer, mock_decoder, device):
        """Test statistics retrieval."""
        panel = AutopoieticPanel(
            config=autopoietic_config,
            internal_scorer=mock_scorer,
            decoder=mock_decoder,
            device=device,
            use_mock_external=True,
        )

        stats = panel.get_statistics()
        assert stats.judge_trust == autopoietic_config.initial_internal_trust
        assert stats.temperature > 0

    def test_reset(self, autopoietic_config, mock_scorer, mock_decoder, device):
        """Test panel reset."""
        panel = AutopoieticPanel(
            config=autopoietic_config,
            internal_scorer=mock_scorer,
            decoder=mock_decoder,
            device=device,
            use_mock_external=True,
        )

        panel._generation = 10
        panel.reset()

        assert panel._generation == 0


class TestCreateAutopoieticPanel:
    """Tests for panel factory function."""

    def test_disabled_returns_none(self, mock_scorer, mock_decoder):
        """Test disabled config returns None."""
        config = AutopoieticJudgeConfig(enabled=False)
        panel = create_autopoietic_panel(config, mock_scorer, mock_decoder)
        assert panel is None

    def test_enabled_returns_panel(self, autopoietic_config, mock_scorer, mock_decoder):
        """Test enabled config returns panel."""
        panel = create_autopoietic_panel(
            autopoietic_config,
            mock_scorer,
            mock_decoder,
            use_mock_external=True,
        )
        assert panel is not None
        assert isinstance(panel, AutopoieticPanel)


# =============================================================================
# Integration Tests
# =============================================================================

class TestAutopoieticIntegration:
    """Integration tests for autopoietic system."""

    def test_full_autopoietic_loop(self, autopoietic_config, mock_scorer, mock_decoder, latent_dim, device):
        """Test complete autopoietic workflow."""
        from latent_reasoning.core.chain import ChainState

        panel = AutopoieticPanel(
            config=autopoietic_config,
            internal_scorer=mock_scorer,
            decoder=mock_decoder,
            device=device,
            use_mock_external=True,
        )
        panel.set_query("Test query for integration")

        # Simulate multiple generations
        for gen in range(10):
            # Create population
            chains = [
                ChainState(latent=torch.randn(latent_dim, device=device))
                for _ in range(10)
            ]

            # Evaluate each chain
            for chain in chains:
                verdict = panel.evaluate(chain.latent)
                chain.score = verdict.score

            # Step generation
            stats = panel.step_generation(chains, generation=gen)

        # Check final state
        final_stats = panel.get_statistics()
        assert final_stats.generation == 9
        assert panel.judge.buffer is not None

    def test_homeostasis_maintains_diversity(self, autopoietic_config, mock_scorer, mock_decoder, latent_dim, device):
        """Test that homeostasis maintains target diversity."""
        from latent_reasoning.core.chain import ChainState

        # Set a specific target diversity
        autopoietic_config.target_diversity = 0.5

        panel = AutopoieticPanel(
            config=autopoietic_config,
            internal_scorer=mock_scorer,
            decoder=mock_decoder,
            device=device,
            use_mock_external=True,
        )

        temperatures = []

        # Run with varying diversity
        for gen in range(20):
            # Alternate between low and high diversity populations
            if gen % 2 == 0:
                # Low diversity - similar latents
                base = torch.randn(latent_dim, device=device)
                chains = [ChainState(latent=base + torch.randn(latent_dim, device=device) * 0.01) for _ in range(10)]
            else:
                # High diversity - random latents
                chains = [ChainState(latent=torch.randn(latent_dim, device=device)) for _ in range(10)]

            stats = panel.step_generation(chains, generation=gen)
            temperatures.append(stats["temperature"])

        # Temperature should vary in response to diversity changes
        assert max(temperatures) > min(temperatures)
