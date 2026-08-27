"""Integration tests for the full latent space reasoning pipeline."""

import pytest
import torch
import tempfile
import os

from latent_reasoning.config import (
    Config, EncoderConfig, JudgeConfig, ScorerConfig, ModifierConfig,
    EvolutionConfig, SelectionConfig, MutationConfig, CrossoverConfig,
    ConvergenceConfig, MergeConfig, BudgetConfig, SynthesisConfig, OutputConfig,
)
from latent_reasoning.core.encoder import Encoder
from latent_reasoning.core.judge import Judge
from latent_reasoning.core.chain import ChainState, ChainTracker, ModificationContext
from latent_reasoning.core.panel import JudgePanel
from latent_reasoning.evolution.selection import get_selection_strategy
from latent_reasoning.evolution.mutation import get_mutation_strategy
from latent_reasoning.evolution.crossover import get_crossover_strategy, population_diversity
from latent_reasoning.evolution.loop import EvolutionLoop
from latent_reasoning.engine import Engine
from latent_reasoning.orchestrator.budget import ComputeBudget
from latent_reasoning.orchestrator.checkpoint import CheckpointManager
from latent_reasoning.orchestrator.orchestrator import Orchestrator


class MockEncoder(Encoder):
    """Mock encoder for testing."""

    def __init__(self, dim: int = 768):
        self._latent_dim = dim
        self._device = torch.device("cpu")

    def encode(self, text: str) -> torch.Tensor:
        """Create a deterministic latent based on text hash."""
        torch.manual_seed(hash(text) % (2**32))
        return torch.randn(self._latent_dim)

    def decode(self, latent: torch.Tensor, query: str | None = None, **kwargs) -> str:
        """Decode latent to text (mock)."""
        return f"Decoded output from latent with norm {latent.norm():.4f}"

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        return self


class MockScorerJudge(Judge):
    """Mock scorer that returns consistent scores."""

    def __init__(self, bias: float = 0.0, dim: int = 768):
        self.bias = bias
        self._latent_dim = dim
        self._device = torch.device("cpu")

    def score(self, latent: torch.Tensor) -> float:
        """Score based on latent magnitude with bias."""
        magnitude = latent.norm().item()
        score = torch.sigmoid(torch.tensor(magnitude - 25 + self.bias * 10)).item()
        return score * 2 - 1  # Map to -1 to 1

    def calibrate(self, latents: torch.Tensor) -> None:
        """No-op calibration."""
        pass

    def set_reference(self, text: str | None = None, embedding: torch.Tensor | None = None) -> None:
        """No-op reference setting."""
        pass

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        return self


class MockModifierJudge(Judge):
    """Mock modifier that suggests improvements."""

    def __init__(self, strength: float = 0.1, dim: int = 768):
        self.strength = strength
        self._latent_dim = dim
        self._device = torch.device("cpu")

    def suggest_modification(
        self,
        latent: torch.Tensor,
        context: ModificationContext | None = None,
    ) -> torch.Tensor:
        """Suggest moving toward unit norm."""
        current_norm = latent.norm()
        if current_norm > 0:
            target = latent / current_norm
            delta = (target - latent) * self.strength
            return delta
        return torch.zeros_like(latent)

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        return self


class ConstantScorerJudge(Judge):
    """Scorer that returns a constant value to force plateau behavior."""

    def __init__(self, score: float = 0.0, dim: int = 768):
        self._score = score
        self._latent_dim = dim
        self._device = torch.device("cpu")

    def score(self, latent: torch.Tensor) -> float:
        return self._score

    def calibrate(self, latents: torch.Tensor) -> None:
        pass

    def set_reference(self, text: str | None = None, embedding: torch.Tensor | None = None) -> None:
        pass

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        return self


class TestEvolutionComponents:
    """Test evolution components work together."""

    def test_selection_with_population(self):
        """Test selection produces valid subset."""
        population = [torch.randn(64) for _ in range(8)]
        scores = [0.1 * i for i in range(8)]

        selection = get_selection_strategy("elitist", elite_k=4)
        selected, selected_scores = selection.select(population, scores, 4)

        assert len(selected) == 4
        assert len(selected_scores) == 4
        # Elitist should pick the highest scores
        assert min(selected_scores) >= 0.4

    def test_mutation_modifies_latent(self):
        """Test mutation changes the latent vector."""
        latent = torch.randn(64)
        original = latent.clone()

        mutation = get_mutation_strategy("gaussian", noise_scale=0.1)
        mutated = mutation.mutate(latent, None, temperature=1.0)

        assert not torch.allclose(original, mutated), "Mutation should change latent"

    def test_crossover_combines_parents(self):
        """Test crossover produces valid child."""
        parent1 = torch.ones(64)
        parent2 = torch.zeros(64)

        crossover = get_crossover_strategy("mean")
        child = crossover.crossover(parent1, parent2, 0.5, 0.5)

        # Mean should be 0.5
        assert torch.allclose(child, torch.ones(64) * 0.5)


class TestJudgePanelIntegration:
    """Test judge panel with mock judges."""

    def test_panel_aggregates_scores(self):
        """Test panel correctly aggregates scores."""
        scorer1 = MockScorerJudge(bias=0.0)
        scorer2 = MockScorerJudge(bias=0.5)

        panel = JudgePanel(scorers=[scorer1, scorer2], aggregation="mean")

        latent = torch.randn(768)
        verdict = panel.evaluate(latent)

        assert -1 <= verdict.score <= 1
        assert len(verdict.individual_scores) == 2

    def test_panel_with_modifiers(self):
        """Test panel produces modifications."""
        scorer = MockScorerJudge()
        modifier = MockModifierJudge(strength=0.1)

        panel = JudgePanel(scorers=[scorer], modifiers=[modifier])

        latent = torch.randn(768)
        verdict = panel.evaluate(latent)

        assert verdict.modification is not None
        assert verdict.modification.shape == latent.shape


class TestEvolutionLoopIntegration:
    """Test the evolution loop with mocks."""

    def test_evolution_loop_runs(self):
        """Test evolution loop completes without errors."""
        latent_dim = 64

        # Create judge panel
        panel = JudgePanel(
            scorers=[MockScorerJudge(dim=latent_dim)],
            modifiers=[MockModifierJudge(dim=latent_dim)],
        )

        # Create config
        config = EvolutionConfig(
            chains=4,
            generations=3,
            selection=SelectionConfig(strategy="elitist", elite=2, survivors=2),
            mutation=MutationConfig(strategy="gaussian", trust=0.1),
            crossover=CrossoverConfig(strategy="mean", threshold=0.8),
            convergence=ConvergenceConfig(threshold=0.99, patience=50),
            merge=MergeConfig(threshold=0.98),
        )

        loop = EvolutionLoop(judge_panel=panel, config=config)

        # Run evolution
        seed = torch.randn(latent_dim)
        result = loop.run(seed, max_evaluations=100)

        assert result is not None
        assert result.best_latent.shape == seed.shape
        assert result.generations <= 3
        assert len(result.history) > 0

    def test_evolution_improves_or_maintains_score(self):
        """Test evolution generally improves scores."""
        latent_dim = 64
        torch.manual_seed(42)

        # Create a scorer that rewards higher norms
        class NormScorer(Judge):
            def __init__(self):
                self._device = torch.device("cpu")

            def score(self, latent):
                return min(1.0, latent.norm().item() / 10.0)

            def calibrate(self, latents):
                pass

            @property
            def device(self):
                return self._device

            def to(self, device):
                return self

        panel = JudgePanel(scorers=[NormScorer()])

        config = EvolutionConfig(
            chains=8,
            generations=10,
            selection=SelectionConfig(strategy="tournament", elite=4, survivors=4),
            mutation=MutationConfig(strategy="gaussian", trust=0.3),
            crossover=CrossoverConfig(strategy="mean", threshold=0.5),
            convergence=ConvergenceConfig(threshold=0.99, patience=50),
        )

        loop = EvolutionLoop(judge_panel=panel, config=config)

        # Start with a low-norm seed
        seed = torch.randn(latent_dim) * 0.1
        initial_score = panel.score(seed)

        result = loop.run(seed, max_evaluations=500)

        # Best score should be at least as good as initial
        assert result.best_score >= initial_score * 0.9, "Evolution should not degrade significantly"


class TestBudgetTracking:
    """Test budget management."""

    def test_budget_tracks_generations(self):
        """Test budget tracks generations correctly."""
        budget = ComputeBudget(max_generations=10, max_evaluations=1000)

        assert budget.can_continue()
        assert budget.generations_used == 0

        for i in range(5):
            budget.record_generation()

        assert budget.generations_used == 5
        assert budget.can_continue()

        for i in range(5):
            budget.record_generation()

        assert not budget.can_continue()

    def test_budget_tracks_evaluations(self):
        """Test budget tracks evaluations correctly."""
        budget = ComputeBudget(max_generations=100, max_evaluations=100)

        budget.record_evaluation(50)
        assert budget.evaluations_used == 50
        assert budget.can_continue()

        budget.record_evaluation(50)
        assert not budget.can_continue()

    def test_budget_time_tracking(self):
        """Test budget tracks time correctly."""
        budget = ComputeBudget(max_generations=10, max_evaluations=100, max_time=60.0)
        budget.start()

        assert budget.can_continue()
        usage = budget.get_usage()
        assert "elapsed_time" in usage


class TestCheckpointManager:
    """Test checkpoint saving and loading."""

    def test_checkpoint_save_load(self, tmp_path):
        """Test checkpoint saving and loading."""
        checkpoint_dir = tmp_path / "checkpoints"
        manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))

        # Create chains to save
        chains = [
            ChainState(latent=torch.randn(64), score=0.1),
            ChainState(latent=torch.randn(64), score=0.2),
        ]

        # Save checkpoint
        manager.save_checkpoint(
            chains=chains,
            generation=5,
            best_latent=chains[1].latent,
            best_score=0.2,
        )

        # Load checkpoint
        loaded = manager.get_latest_checkpoint()

        assert loaded is not None
        assert loaded.generation == 5
        assert loaded.best_score == 0.2


class TestOrchestratorIntegration:
    """Test the full orchestrator pipeline."""

    def test_orchestrator_with_mocks(self, tmp_path):
        """Test orchestrator runs with mock components."""
        latent_dim = 64

        # Create mock encoder and judges
        encoder = MockEncoder(dim=latent_dim)
        panel = JudgePanel(
            scorers=[MockScorerJudge(dim=latent_dim)],
            modifiers=[MockModifierJudge(dim=latent_dim)],
        )

        # Create config
        config = Config(
            encoder=EncoderConfig(
                model="mock",
            ),
            evolution=EvolutionConfig(
                chains=4,
                generations=3,
            ),
            budget=BudgetConfig(max_evaluations=50),
            output=OutputConfig(
                save_history=True,
                history_path=str(tmp_path / "history"),
            ),
        )

        # Create orchestrator with injected components
        orchestrator = Orchestrator(
            config=config,
            encoder=encoder,
            judge_panel=panel,
        )

        # Run
        result = orchestrator.run("What is the meaning of life?")

        assert result is not None
        assert result.generations <= 3
        assert len(result.decoded_outputs) > 0

    def test_orchestrator_baseline(self, tmp_path):
        """Test baseline comparison mode."""
        latent_dim = 64

        encoder = MockEncoder(dim=latent_dim)
        panel = JudgePanel(scorers=[MockScorerJudge(dim=latent_dim)])

        config = Config(
            encoder=EncoderConfig(model="mock"),
            evolution=EvolutionConfig(chains=4, generations=2),
            output=OutputConfig(save_history=False),
        )

        orchestrator = Orchestrator(config=config, encoder=encoder, judge_panel=panel)

        # Get baseline output
        baseline = orchestrator.run_baseline("Test query")
        assert baseline is not None
        assert len(baseline) > 0

    def test_orchestrator_compare(self, tmp_path):
        """Test comparison mode."""
        latent_dim = 64

        encoder = MockEncoder(dim=latent_dim)
        panel = JudgePanel(scorers=[MockScorerJudge(dim=latent_dim)])

        config = Config(
            encoder=EncoderConfig(model="mock"),
            evolution=EvolutionConfig(chains=4, generations=2),
            output=OutputConfig(save_history=False),
        )

        orchestrator = Orchestrator(config=config, encoder=encoder, judge_panel=panel)

        # Compare outputs
        comparison = orchestrator.compare("Test query")

        assert "baseline" in comparison
        assert "latent_reasoning" in comparison
        assert "latent_score" in comparison
        assert "baseline_duration_s" in comparison
        assert "latent_duration_s" in comparison
        assert "latent_run_duration_s" in comparison
        assert "latent_encode_duration_s" in comparison
        assert "latent_evolution_duration_s" in comparison
        assert "latent_decode_duration_s" in comparison
        assert "latent_non_evolution_duration_s" in comparison
        assert "total_compare_duration_s" in comparison
        assert "latency_overhead_ratio" in comparison
        assert comparison["baseline_duration_s"] >= 0.0
        assert comparison["latent_duration_s"] >= 0.0
        assert comparison["latent_run_duration_s"] >= 0.0
        assert comparison["latent_encode_duration_s"] >= 0.0
        assert comparison["latent_evolution_duration_s"] >= 0.0
        assert comparison["latent_decode_duration_s"] >= 0.0
        assert comparison["latent_non_evolution_duration_s"] >= 0.0
        assert comparison["total_compare_duration_s"] >= 0.0

    def test_orchestrator_budget_does_not_leak_across_queries(self, tmp_path):
        """Per-query budget should reset when reusing an orchestrator."""
        latent_dim = 64

        encoder = MockEncoder(dim=latent_dim)
        panel = JudgePanel(scorers=[ConstantScorerJudge(score=0.0, dim=latent_dim)])

        config = Config(
            encoder=EncoderConfig(model="mock"),
            evolution=EvolutionConfig(
                chains=4,
                generations=2,
                convergence=ConvergenceConfig(threshold=0.99, patience=20),
            ),
            budget=BudgetConfig(max_evaluations=20),
            output=OutputConfig(save_history=False),
        )

        orchestrator = Orchestrator(config=config, encoder=encoder, judge_panel=panel)
        first = orchestrator.compare("First query")
        second = orchestrator.compare("Second query")

        assert first["evaluations"] > 0
        assert second["evaluations"] > 0
        # Budget must not accumulate across queries (would be ~2x if leaked).
        assert second["evaluations"] <= first["evaluations"] + 2
        # Budget state should reflect only the latest run, not cumulative usage.
        assert orchestrator.budget.evaluations_used == second["evaluations"]
        assert orchestrator.budget.evaluations_used <= config.budget.max_evaluations

    def test_engine_preserves_config_verbosity_when_not_overridden(self):
        """Engine should not overwrite config verbosity unless explicitly requested."""
        latent_dim = 64
        cfg = Config(
            encoder=EncoderConfig(model="mock"),
            output=OutputConfig(verbosity="silent", save_history=False),
        )

        engine = Engine(
            config=cfg,
            encoder=MockEncoder(dim=latent_dim),
            scorers=[MockScorerJudge(dim=latent_dim)],
            modifiers=[],
        )
        assert engine.config.output.verbosity == "silent"

        override_engine = Engine(
            config=cfg,
            encoder=MockEncoder(dim=latent_dim),
            scorers=[MockScorerJudge(dim=latent_dim)],
            modifiers=[],
            verbosity="verbose",
        )
        assert override_engine.config.output.verbosity == "verbose"


class TestEndToEnd:
    """Full end-to-end integration tests."""

    def test_complete_reasoning_flow(self):
        """Test complete reasoning flow from query to output."""
        latent_dim = 64
        torch.manual_seed(42)

        # Setup components
        encoder = MockEncoder(dim=latent_dim)
        panel = JudgePanel(
            scorers=[
                MockScorerJudge(bias=0.0, dim=latent_dim),
                MockScorerJudge(bias=0.3, dim=latent_dim),
            ],
            modifiers=[MockModifierJudge(strength=0.1, dim=latent_dim)],
        )

        config = EvolutionConfig(
            chains=8,
            generations=10,
            selection=SelectionConfig(strategy="elitist", elite=4, survivors=4),
            mutation=MutationConfig(strategy="adaptive", trust=0.1),
            crossover=CrossoverConfig(strategy="interpolation", threshold=0.5),
            convergence=ConvergenceConfig(threshold=0.95, patience=5),
        )

        loop = EvolutionLoop(judge_panel=panel, config=config)

        # Encode query
        query = "What steps should I take to learn machine learning?"
        seed = encoder.encode(query)
        initial_score = panel.score(seed)

        # Run evolution
        result = loop.run(seed, max_evaluations=200)

        # Decode result
        output = encoder.decode(result.best_latent)

        assert output is not None
        assert len(output) > 0
        assert result.best_score is not None
        assert result.generations >= 1

    def test_multiple_queries_sequential(self):
        """Test processing multiple queries in sequence."""
        latent_dim = 64
        encoder = MockEncoder(dim=latent_dim)
        panel = JudgePanel(scorers=[MockScorerJudge(dim=latent_dim)])

        queries = [
            "How do I debug a memory leak?",
            "What is the best database for my use case?",
            "How should I structure this API?",
        ]

        results = []
        for query in queries:
            latent = encoder.encode(query)
            score = panel.score(latent)
            output = encoder.decode(latent)
            results.append({
                "query": query,
                "score": score,
                "output": output,
            })

        assert len(results) == 3
        assert all(r["output"] is not None for r in results)
        assert all(-1 <= r["score"] <= 1 for r in results)

    def test_chain_tracking_integration(self):
        """Test chain tracking through evolution."""
        latent_dim = 64

        tracker = ChainTracker()

        # Simulate evolution steps
        for i in range(5):
            latent = torch.randn(latent_dim)
            score = 0.1 * (i + 1)
            tracker.record(latent, score)

        # Check progress should work
        new_latent = torch.randn(latent_dim)
        progress = tracker.check_progress(new_latent)

        assert "ok" in progress
        assert tracker.compute_momentum() is not None
        assert tracker.compute_stuck_signal() is not None

    def test_diversity_preservation(self):
        """Test that evolution maintains diversity."""
        latent_dim = 64
        torch.manual_seed(42)

        # Create diverse initial population
        population = [torch.randn(latent_dim) for _ in range(8)]
        initial_diversity = population_diversity(population)

        # Create panel that rewards diversity implicitly
        panel = JudgePanel(scorers=[MockScorerJudge(dim=latent_dim)])

        config = EvolutionConfig(
            chains=8,
            generations=5,
            crossover=CrossoverConfig(threshold=0.3),  # Low threshold to encourage crossover
            merge=MergeConfig(threshold=0.99),  # High threshold to avoid merging
        )

        loop = EvolutionLoop(judge_panel=panel, config=config)

        # Run evolution
        seed = population[0]
        result = loop.run(seed, max_evaluations=100)

        # Final population should have some diversity
        final_population = [s.latent for s in result.survivors]
        if len(final_population) > 1:
            final_diversity = population_diversity(final_population)
            # Some diversity should remain
            assert final_diversity > 0

    def test_adaptive_survivor_budget_decays_on_plateau(self):
        """Adaptive survivor budget should shrink when quality plateaus."""
        latent_dim = 64
        seed = torch.randn(latent_dim)

        panel = JudgePanel(scorers=[ConstantScorerJudge(score=0.0, dim=latent_dim)])

        config = EvolutionConfig(
            chains=6,
            generations=5,
            temperature=0.2,
            selection=SelectionConfig(
                strategy="elitist",
                survivors=5,
                elite=2,
                adaptive_survivors=True,
                min_survivors=2,
                survivor_decay=0.5,
                survivor_decay_patience=1,
            ),
            mutation=MutationConfig(strategy="gaussian", noise_scale=0.05),
            crossover=CrossoverConfig(strategy="mean", threshold=0.0),
            convergence=ConvergenceConfig(threshold=0.99, patience=50),
        )

        loop = EvolutionLoop(judge_panel=panel, config=config)
        result = loop.run(seed, max_evaluations=200)

        survivor_budgets = [entry["survivor_budget"] for entry in result.history]
        assert survivor_budgets, "History should include survivor budget metrics"
        assert survivor_budgets[0] == 5
        assert min(survivor_budgets) >= 2
        assert any(budget < 5 for budget in survivor_budgets[1:])

    def test_evolution_run_state_does_not_leak_across_queries(self):
        """Evaluation counters and temperature should reset each run."""
        latent_dim = 64
        panel = JudgePanel(scorers=[ConstantScorerJudge(score=0.0, dim=latent_dim)])

        config = EvolutionConfig(
            chains=1,
            generations=1,
            temperature=0.6,
            convergence=ConvergenceConfig(threshold=0.99, patience=10),
        )

        loop = EvolutionLoop(judge_panel=panel, config=config)
        seed_a = torch.randn(latent_dim)
        seed_b = torch.randn(latent_dim)

        result_a = loop.run(seed_a, max_evaluations=100)
        temperature_after_a = loop.current_temperature
        result_b = loop.run(seed_b, max_evaluations=100)
        temperature_after_b = loop.current_temperature

        assert result_a.total_evaluations > 0
        assert result_a.total_evaluations <= 3
        assert result_b.total_evaluations > 0
        assert result_b.total_evaluations <= 3
        assert result_b.total_evaluations == result_a.total_evaluations
        # Temperatures should match if each run starts from the same initial value.
        assert temperature_after_a == pytest.approx(temperature_after_b)

    def test_score_cache_reduces_evaluations_for_duplicate_latents(self):
        """Score cache should cut scorer calls when latents repeat."""
        latent_dim = 64
        seed = torch.randn(latent_dim)
        panel = JudgePanel(scorers=[ConstantScorerJudge(score=0.5, dim=latent_dim)])

        base_config = EvolutionConfig(
            chains=5,
            generations=2,
            temperature=0.0,          # Keep population identical to seed
            temperature_decay=1.0,
            merge=MergeConfig(threshold=1.0),  # Avoid merging identical vectors
            crossover=CrossoverConfig(strategy="mean", threshold=1.0),  # Disable crossover
            mutation=MutationConfig(strategy="gaussian", noise_scale=0.0),
            convergence=ConvergenceConfig(threshold=0.99, patience=10),
        )

        cfg_no_cache = base_config.model_copy(deep=True)
        cfg_no_cache.score_cache = False
        result_no_cache = EvolutionLoop(judge_panel=panel, config=cfg_no_cache).run(seed, max_evaluations=200)

        cfg_cache = base_config.model_copy(deep=True)
        cfg_cache.score_cache = True
        cfg_cache.score_cache_precision = 4
        result_cache = EvolutionLoop(judge_panel=panel, config=cfg_cache).run(seed, max_evaluations=200)

        assert result_cache.total_evaluations < result_no_cache.total_evaluations
