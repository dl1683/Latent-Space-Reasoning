"""Tests for evolution operators."""

import pytest
import torch

from latent_reasoning.evolution.selection import (
    ElitistSelection,
    TournamentSelection,
    RankSelection,
    RouletteSelection,
    get_selection_strategy,
)
from latent_reasoning.evolution.mutation import (
    GaussianMutation,
    DirectedMutation,
    AdaptiveMutation,
    get_mutation_strategy,
)
from latent_reasoning.evolution.crossover import (
    MeanCrossover,
    WeightedCrossover,
    InterpolationCrossover,
    get_crossover_strategy,
    population_diversity,
    select_crossover_pairs,
)


class TestSelection:
    @pytest.fixture
    def population(self):
        return [torch.randn(768) for _ in range(10)]

    @pytest.fixture
    def scores(self):
        return [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.95]

    def test_elitist_selection(self, population, scores):
        strategy = ElitistSelection(elite_k=2)
        selected, selected_scores = strategy.select(population, scores, n_survivors=5)

        assert len(selected) == 5
        assert len(selected_scores) == 5
        # Top 2 scores should always be included
        assert 0.95 in selected_scores
        assert 0.9 in selected_scores

    def test_rank_selection(self, population, scores):
        strategy = RankSelection()
        selected, selected_scores = strategy.select(population, scores, n_survivors=3)

        assert len(selected) == 3
        # Should be top 3
        assert sorted(selected_scores, reverse=True) == selected_scores

    def test_tournament_selection(self, population, scores):
        strategy = TournamentSelection(tournament_size=3)
        selected, selected_scores = strategy.select(population, scores, n_survivors=5)

        assert len(selected) == 5

    def test_roulette_selection(self, population, scores):
        strategy = RouletteSelection()
        selected, selected_scores = strategy.select(population, scores, n_survivors=5)

        assert len(selected) == 5

    def test_get_selection_strategy(self):
        strategy = get_selection_strategy("elitist", elite_k=3)
        assert isinstance(strategy, ElitistSelection)

        strategy = get_selection_strategy("tournament", tournament_size=4)
        assert isinstance(strategy, TournamentSelection)

        with pytest.raises(ValueError):
            get_selection_strategy("invalid")


class TestMutation:
    @pytest.fixture
    def candidate(self):
        return torch.randn(768)

    @pytest.fixture
    def hint(self):
        return torch.randn(768)

    def test_gaussian_mutation(self, candidate):
        strategy = GaussianMutation(noise_scale=1.0)
        mutated = strategy.mutate(candidate, None, temperature=0.5)

        assert mutated.shape == candidate.shape
        assert not torch.allclose(mutated, candidate)

    def test_directed_mutation_no_hint(self, candidate):
        strategy = DirectedMutation(trust=0.7)
        mutated = strategy.mutate(candidate, None, temperature=0.5)

        assert mutated.shape == candidate.shape

    def test_directed_mutation_with_hint(self, candidate, hint):
        strategy = DirectedMutation(trust=0.7)
        mutated = strategy.mutate(candidate, hint, temperature=0.5)

        assert mutated.shape == candidate.shape

    def test_adaptive_mutation(self, candidate, hint):
        strategy = AdaptiveMutation()

        # Initial mutation
        mutated1 = strategy.mutate(candidate, hint, temperature=0.5)
        assert mutated1.shape == candidate.shape

        # Update with improving score
        strategy.update_adaptation(0.5)
        strategy.update_adaptation(0.6)

        # Trust should have increased
        assert strategy.current_trust > strategy.base_trust

        # Reset
        strategy.reset()
        assert strategy.current_trust == strategy.base_trust

    def test_get_mutation_strategy(self):
        strategy = get_mutation_strategy("gaussian", noise_scale=0.5)
        assert isinstance(strategy, GaussianMutation)

        strategy = get_mutation_strategy("directed", trust=0.8)
        assert isinstance(strategy, DirectedMutation)

        with pytest.raises(ValueError):
            get_mutation_strategy("invalid")


class TestCrossover:
    @pytest.fixture
    def parent_a(self):
        return torch.randn(768)

    @pytest.fixture
    def parent_b(self):
        return torch.randn(768)

    def test_mean_crossover(self, parent_a, parent_b):
        strategy = MeanCrossover()
        child = strategy.crossover(parent_a, parent_b)

        assert child.shape == parent_a.shape
        # Child should be between parents
        expected = (parent_a + parent_b) / 2
        assert torch.allclose(child, expected)

    def test_weighted_crossover(self, parent_a, parent_b):
        strategy = WeightedCrossover()
        child = strategy.crossover(parent_a, parent_b, score_a=0.8, score_b=0.2)

        assert child.shape == parent_a.shape
        # Child should be closer to parent_a (higher score)

    def test_interpolation_crossover(self, parent_a, parent_b):
        strategy = InterpolationCrossover()
        child = strategy.crossover(parent_a, parent_b)

        assert child.shape == parent_a.shape

    def test_get_crossover_strategy(self):
        strategy = get_crossover_strategy("mean")
        assert isinstance(strategy, MeanCrossover)

        strategy = get_crossover_strategy("weighted")
        assert isinstance(strategy, WeightedCrossover)

        with pytest.raises(ValueError):
            get_crossover_strategy("invalid")


class TestDiversity:
    def test_identical_population(self):
        vec = torch.randn(768)
        population = [vec.clone() for _ in range(5)]

        diversity = population_diversity(population)
        assert diversity < 0.01  # Nearly zero

    def test_diverse_population(self):
        population = [torch.randn(768) for _ in range(5)]

        diversity = population_diversity(population)
        assert diversity > 0.5  # Should be high for random vectors

    def test_single_vector(self):
        population = [torch.randn(768)]
        diversity = population_diversity(population)
        assert diversity == 0.0

    def test_select_crossover_pairs(self):
        population = [torch.randn(768) for _ in range(10)]
        scores = list(range(10))

        pairs = select_crossover_pairs(population, scores, n_pairs=3)

        assert len(pairs) <= 3
        for a, b in pairs:
            assert a != b
