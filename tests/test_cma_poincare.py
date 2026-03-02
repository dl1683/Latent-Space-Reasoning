"""Tests for Poincare CMA-ES (low-rank)."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.evolution.cma_poincare import PoincareCMA
from latent_reasoning.utils import hyperbolic as hyp


class TestPoincareCMA:
    def test_all_samples_inside_ball(self):
        """All sampled candidates must be inside the Poincare ball."""
        cma = PoincareCMA(dim=64, population_size=8, curvature=0.5)
        rng = torch.Generator().manual_seed(42)
        candidates = cma.sample(rng)
        ball_radius = 1.0 / math.sqrt(0.5)
        for c in candidates:
            assert c.norm().item() < ball_radius

    def test_correct_population_size(self):
        """sample() should return exactly lambda candidates."""
        cma = PoincareCMA(dim=32, population_size=12)
        candidates = cma.sample()
        assert len(candidates) == 12

    def test_candidate_shapes(self):
        """Each candidate should have shape (dim,)."""
        cma = PoincareCMA(dim=64, population_size=4)
        for c in cma.sample():
            assert c.shape == (64,)

    def test_mean_moves_toward_high_fitness(self):
        """Mean should move toward the high-fitness region over many gens."""
        dim = 16
        cma = PoincareCMA(dim=dim, population_size=12, curvature=0.5, sigma=0.1)

        # Target: point at (0.3, 0, 0, ..., 0) in Poincare ball
        target = torch.zeros(dim)
        target[0] = 0.3

        initial_dist = hyp.hyperbolic_distance(
            cma.mean, target, 0.5,
        ).item()

        # Run enough generations for CMA to converge
        for _ in range(20):
            candidates = cma.sample()
            fitnesses = [
                -hyp.hyperbolic_distance(c, target, 0.5).item()
                for c in candidates
            ]
            cma.update(candidates, fitnesses)

        final_dist = hyp.hyperbolic_distance(
            cma.mean, target, 0.5,
        ).item()
        assert final_dist < initial_dist

    def test_low_rank_u_shape(self):
        """Low-rank factor U should have shape (dim, rank)."""
        cma = PoincareCMA(dim=64, population_size=8, rank=10)
        assert cma.U.shape == (64, 10)

    def test_sigma_adapts(self):
        """Step size should change after update."""
        cma = PoincareCMA(dim=32, population_size=6)
        initial_sigma = cma.sigma
        candidates = cma.sample()
        fitnesses = [float(i) for i in range(6)]
        cma.update(candidates, fitnesses)
        # Sigma should have changed (not be exactly the same)
        assert cma.sigma != initial_sigma

    def test_mean_stays_in_ball_after_updates(self):
        """Mean should stay inside the Poincare ball after multiple updates."""
        cma = PoincareCMA(dim=32, population_size=6, curvature=1.0)
        ball_radius = 1.0 / math.sqrt(1.0)
        for _ in range(10):
            candidates = cma.sample()
            fitnesses = [torch.randn(1).item() for _ in candidates]
            cma.update(candidates, fitnesses)
            assert cma.mean.norm().item() < ball_radius

    def test_rank_clamped_to_dim(self):
        """Rank should not exceed dim."""
        cma = PoincareCMA(dim=8, rank=100)
        assert cma.rank == 8
        assert cma.U.shape == (8, 8)

    def test_best_point_is_mean(self):
        """best_point should return a copy of the mean."""
        cma = PoincareCMA(dim=16)
        bp = cma.best_point
        assert torch.allclose(bp, cma.mean)
        # Should be a copy, not a reference
        bp[0] = 999.0
        assert cma.mean[0] != 999.0
