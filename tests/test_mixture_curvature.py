"""Tests for mixture-of-curvature mutation."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.evolution.mutation import MixtureCurvatureMutation, get_mutation_strategy
from latent_reasoning.utils import hyperbolic as hyp


class TestMixtureCurvatureMutation:
    def test_factory_registration(self):
        """mixture_curvature should be available from get_mutation_strategy."""
        strat = get_mutation_strategy("mixture_curvature", noise_scale=0.2)
        assert isinstance(strat, MixtureCurvatureMutation)

    def test_curvature_stays_in_range(self):
        """Curvature should stay within [c_min, c_max] after many mutations."""
        strat = MixtureCurvatureMutation(
            noise_scale=0.3, curvature_sigma=0.5,
            c_min=0.1, c_max=5.0,
        )
        c = 1.0
        candidate = torch.zeros(64)
        for _ in range(100):
            strat.mutate(candidate, None, temperature=1.0, curvature=c)
            c = strat.new_curvature
            assert 0.1 <= c <= 5.0

    def test_latent_valid_in_new_ball(self):
        """Mutated latent should be inside the Poincare ball at new curvature."""
        strat = MixtureCurvatureMutation(noise_scale=0.2, max_norm=0.95)
        candidate = hyp.expmap0(torch.randn(64) * 0.1, 0.5)
        for _ in range(50):
            mutated = strat.mutate(candidate, None, temperature=0.5, curvature=0.5)
            new_c = strat.new_curvature
            ball_radius = 1.0 / math.sqrt(new_c)
            assert mutated.norm().item() < ball_radius
            candidate = mutated

    def test_curvature_actually_changes(self):
        """Curvature should change from the parent value (stochastic)."""
        strat = MixtureCurvatureMutation(curvature_sigma=0.3)
        candidate = torch.zeros(64)
        curvatures = set()
        for _ in range(20):
            strat.mutate(candidate, None, temperature=1.0, curvature=1.0)
            curvatures.add(round(strat.new_curvature, 4))
        # With sigma=0.3 and 20 samples, we should see multiple values
        assert len(curvatures) > 5

    def test_output_shape_matches_input(self):
        """Mutated tensor should have same shape as input."""
        strat = MixtureCurvatureMutation()
        candidate = torch.randn(128) * 0.1
        mutated = strat.mutate(candidate, None, temperature=0.5, curvature=0.5)
        assert mutated.shape == candidate.shape
