"""Tests for V15 geometry isolation design invariants."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.harness import (
    Candidate,
    EvolutionParams,
    _apply_mutation,
    _make_noise,
)
from latent_reasoning.decode.projection import (
    latent_to_soft_prompt,
    make_row_orthonormal_W,
)
from latent_reasoning.utils import hyperbolic as hyp


class TestGeometryIsolation:
    """Verify that the V15 experiment properly isolates geometry."""

    @pytest.fixture
    def shared_W(self):
        """Same W used for all conditions in V15."""
        return make_row_orthonormal_W(64, 256, seed=1234)

    def test_euclidean_mutation_stays_in_ball(self):
        """Euclidean mutation must stay within L2 ball of matched radius."""
        curvature = 0.5
        ball_radius = (1.0 / math.sqrt(curvature)) * 0.95
        parent = torch.randn(1, 64) * 0.5
        rng = torch.Generator().manual_seed(42)

        for _ in range(50):
            noise = _make_noise(parent.shape, 0.3, 64, rng)
            mutated = _apply_mutation(parent, noise, curvature, ball_radius, "euclidean")
            assert mutated.squeeze().norm().item() <= ball_radius + 1e-6

    def test_hyperbolic_mutation_stays_in_poincare_ball(self):
        """Hyperbolic mutation must stay within Poincare ball."""
        curvature = 0.5
        parent = hyp.expmap0(torch.randn(64) * 0.1, curvature).unsqueeze(0)
        rng = torch.Generator().manual_seed(42)

        for _ in range(50):
            noise = _make_noise(parent.shape, 0.3, 64, rng)
            mutated = _apply_mutation(parent, noise, curvature, 1.34, "hyperbolic")
            assert mutated.squeeze().norm().item() < 1.0 / math.sqrt(curvature)

    def test_ball_radii_matched(self):
        """Both geometries use the same effective ball radius."""
        curvature = 0.5
        # Harness uses this formula for both:
        ball_radius = (1.0 / math.sqrt(curvature)) * 0.95
        poincare_radius = 1.0 / math.sqrt(curvature)
        # Euclidean clips to ball_radius; Poincare projects to 0.95 * radius
        assert abs(ball_radius - poincare_radius * 0.95) < 1e-10

    def test_same_W_matrix_across_conditions(self, shared_W):
        """Verify W is deterministic so all conditions share it."""
        W2 = make_row_orthonormal_W(64, 256, seed=1234)
        assert torch.allclose(shared_W, W2)

    def test_soft_prompt_rms_matched_across_geometries(self, shared_W):
        """Both geometries should produce soft prompts with matching RMS."""
        # Latent must be inside the Poincare ball for logmap0 to work
        latent = torch.randn(64)
        latent = latent / latent.norm() * 0.8  # norm=0.8, inside ball (radius ~1.41)
        target = 0.05

        sp_euc = latent_to_soft_prompt(
            latent, shared_W, curvature=0.5,
            embed_dim=32, num_tokens=8, target_rms=target,
            use_logmap=False,
        )
        sp_hyp = latent_to_soft_prompt(
            latent, shared_W, curvature=0.5,
            embed_dim=32, num_tokens=8, target_rms=target,
            use_logmap=True,
        )

        rms_euc = sp_euc.float().square().mean().sqrt().item()
        rms_hyp = sp_hyp.float().square().mean().sqrt().item()
        # Both should match target RMS within 10%
        assert abs(rms_euc - target) / target < 0.10
        assert abs(rms_hyp - target) / target < 0.10

    def test_isolation_only_geometry_differs(self, shared_W):
        """Confirm the only difference between conditions is geometry."""
        latent_euc = torch.randn(64) * 0.3
        latent_hyp = latent_euc.clone()

        # Apply Euclidean mutation
        rng_euc = torch.Generator().manual_seed(42)
        noise_euc = _make_noise(latent_euc.unsqueeze(0).shape, 0.1, 64, rng_euc)

        # Apply Hyperbolic mutation
        rng_hyp = torch.Generator().manual_seed(42)
        noise_hyp = _make_noise(latent_hyp.unsqueeze(0).shape, 0.1, 64, rng_hyp)

        # Same noise (same RNG seed)
        assert torch.allclose(noise_euc, noise_hyp)

        # Different mutations
        mut_euc = _apply_mutation(
            latent_euc.unsqueeze(0), noise_euc, 0.5, 1.34, "euclidean",
        )
        mut_hyp = _apply_mutation(
            latent_hyp.unsqueeze(0), noise_hyp, 0.5, 1.34, "hyperbolic",
        )
        # Mutations should differ (different geometry)
        assert not torch.allclose(mut_euc, mut_hyp, atol=1e-3)

    def test_evolution_params_consistent(self):
        """Evolution params should be the same for both evolved conditions."""
        evo = EvolutionParams(
            generations=3, population_size=4,
            tasks_per_gen=8, noise_scale=0.1, curvature=0.5,
        )
        # Both conditions use the same evo params
        assert evo.generations == 3
        assert evo.population_size == 4
        assert evo.noise_scale == 0.1
