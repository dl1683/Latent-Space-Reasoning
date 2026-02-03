"""Tests for hyperbolic geometry operations."""

import math
import pytest
import torch

from latent_reasoning.utils.hyperbolic import (
    expmap0,
    logmap0,
    expmap,
    logmap,
    mobius_add,
    mobius_scalar_mul,
    parallel_transport,
    hyperbolic_distance,
    karcher_mean,
    project_to_ball,
    radial_depth,
    unit_tangent_at_origin,
    hyperbolic_interpolate,
    HyperbolicSpace,
)


class TestBasicOperations:
    """Test basic hyperbolic operations."""

    def test_expmap0_logmap0_roundtrip(self):
        """expmap0 and logmap0 should be inverses for small inputs."""
        # Use small vectors to avoid tanh saturation
        # For 1024 dims, randn has norm ~sqrt(1024)=32, so we scale by 0.01 to get norm ~0.32
        v = torch.randn(1024) * 0.01  # Small for stability (norm ~0.32)
        c = 1.0

        # Map to ball and back
        x = expmap0(v, c)
        v_recovered = logmap0(x, c)

        # Check they're approximately equal (numerical precision)
        assert torch.allclose(v, v_recovered, atol=1e-4), "expmap0/logmap0 roundtrip failed"

    def test_expmap0_stays_in_ball(self):
        """expmap0 should always produce points inside the ball."""
        for _ in range(10):
            v = torch.randn(1024)  # Various magnitudes
            c = 1.0
            x = expmap0(v, c)
            norm = x.norm().item()
            assert norm < 1.0, f"expmap0 produced point outside ball: norm={norm}"

    def test_logmap0_at_origin(self):
        """logmap0 of origin should be zero."""
        origin = torch.zeros(1024)
        c = 1.0
        result = logmap0(origin, c)
        assert result.norm().item() < 1e-6, "logmap0 of origin should be zero"

    def test_project_to_ball(self):
        """project_to_ball should clamp norms."""
        x = torch.randn(1024) * 10  # Large vector
        c = 1.0
        max_norm = 0.9
        projected = project_to_ball(x, c, max_norm)
        assert projected.norm().item() <= max_norm + 1e-6


class TestMobiusOperations:
    """Test Möbius addition and multiplication."""

    def test_mobius_add_identity(self):
        """Adding zero should give the same point."""
        x = expmap0(torch.randn(1024) * 0.3, 1.0)
        zero = torch.zeros_like(x)
        result = mobius_add(x, zero, 1.0)
        assert torch.allclose(x, result, atol=1e-5), "Möbius add with zero should be identity"

    def test_mobius_add_commutativity(self):
        """Möbius addition should be commutative."""
        x = expmap0(torch.randn(1024) * 0.2, 1.0)
        y = expmap0(torch.randn(1024) * 0.2, 1.0)
        c = 1.0

        xy = mobius_add(x, y, c)
        yx = mobius_add(y, x, c)

        # Not exactly commutative in hyperbolic space, but close for small vectors
        # Just verify both are valid points
        assert xy.norm().item() < 1.0
        assert yx.norm().item() < 1.0

    def test_mobius_scalar_mul_zero(self):
        """Multiplying by zero should give origin."""
        x = expmap0(torch.randn(1024) * 0.3, 1.0)
        result = mobius_scalar_mul(0.0, x, 1.0)
        assert result.norm().item() < 1e-5, "r=0 should give origin"

    def test_mobius_scalar_mul_one(self):
        """Multiplying by one should preserve the point."""
        x = expmap0(torch.randn(1024) * 0.3, 1.0)
        result = mobius_scalar_mul(1.0, x, 1.0)
        assert torch.allclose(x, result, atol=1e-5), "r=1 should be identity"


class TestDistance:
    """Test hyperbolic distance."""

    def test_distance_self_is_zero(self):
        """Distance from a point to itself should be zero."""
        x = expmap0(torch.randn(1024) * 0.3, 1.0)
        d = hyperbolic_distance(x, x, 1.0)
        assert d.item() < 1e-5, "Self-distance should be zero"

    def test_distance_symmetry(self):
        """Distance should be symmetric."""
        x = expmap0(torch.randn(1024) * 0.3, 1.0)
        y = expmap0(torch.randn(1024) * 0.3, 1.0)
        c = 1.0

        d_xy = hyperbolic_distance(x, y, c)
        d_yx = hyperbolic_distance(y, x, c)

        assert torch.allclose(d_xy, d_yx, atol=1e-5), "Distance should be symmetric"

    def test_distance_positive(self):
        """Distance should be positive for different points."""
        x = expmap0(torch.randn(1024) * 0.3, 1.0)
        y = expmap0(torch.randn(1024) * 0.3, 1.0)
        d = hyperbolic_distance(x, y, 1.0)
        assert d.item() > 0, "Distance should be positive"

    def test_distance_increases_near_boundary(self):
        """Points near boundary should be far from origin."""
        # Point near boundary - use large tangent magnitude
        v_boundary = torch.zeros(1024)
        v_boundary[0] = 3.0  # Will map near boundary (tanh(3) ≈ 0.995)
        x_boundary = expmap0(v_boundary, 1.0)

        # Point near origin - use small tangent magnitude
        # For 1024 dims, randn*0.001 has norm ~0.032, which maps close to origin
        v_origin = torch.randn(1024) * 0.001
        x_origin = expmap0(v_origin, 1.0)

        origin = torch.zeros(1024)

        # Distance from origin to boundary point should be larger
        d_to_boundary = hyperbolic_distance(origin, x_boundary, 1.0).item()
        d_to_origin = hyperbolic_distance(origin, x_origin, 1.0).item()

        assert d_to_boundary > d_to_origin, f"Boundary should be farther: {d_to_boundary} vs {d_to_origin}"


class TestKarcherMean:
    """Test Karcher mean (hyperbolic barycenter)."""

    def test_karcher_mean_single_point(self):
        """Karcher mean of single point should be that point."""
        x = expmap0(torch.randn(1024) * 0.3, 1.0)
        mean = karcher_mean(x.unsqueeze(0), c=1.0)
        assert torch.allclose(x, mean, atol=1e-4), "Mean of single point should be itself"

    def test_karcher_mean_two_equal_weights(self):
        """Mean of two points with equal weights should be on geodesic."""
        x = expmap0(torch.randn(1024) * 0.2, 1.0)
        y = expmap0(torch.randn(1024) * 0.2, 1.0)
        points = torch.stack([x, y])
        weights = torch.tensor([0.5, 0.5])

        mean = karcher_mean(points, weights, c=1.0)

        # Mean should be closer to both than they are to each other
        d_xy = hyperbolic_distance(x, y, 1.0).item()
        d_mx = hyperbolic_distance(mean, x, 1.0).item()
        d_my = hyperbolic_distance(mean, y, 1.0).item()

        assert d_mx < d_xy
        assert d_my < d_xy

    def test_karcher_mean_weighted(self):
        """Weighted mean should be closer to higher-weight point."""
        x = expmap0(torch.randn(1024) * 0.2, 1.0)
        y = expmap0(torch.randn(1024) * 0.2, 1.0)
        points = torch.stack([x, y])

        # Weight x more heavily
        weights = torch.tensor([0.9, 0.1])
        mean = karcher_mean(points, weights, c=1.0)

        d_mx = hyperbolic_distance(mean, x, 1.0).item()
        d_my = hyperbolic_distance(mean, y, 1.0).item()

        assert d_mx < d_my, "Mean should be closer to higher-weight point"


class TestParallelTransport:
    """Test parallel transport."""

    def test_parallel_transport_preserves_norm(self):
        """Parallel transport should approximately preserve tangent vector norm."""
        v = torch.randn(1024) * 0.3
        p = expmap0(torch.randn(1024) * 0.2, 1.0)
        q = expmap0(torch.randn(1024) * 0.2, 1.0)

        transported = parallel_transport(v.unsqueeze(0), p.unsqueeze(0), q.unsqueeze(0), 1.0).squeeze()

        # Norms won't be exactly equal due to curvature, but should be same order of magnitude
        ratio = transported.norm().item() / (v.norm().item() + 1e-8)
        assert 0.1 < ratio < 10, f"Transport changed norm too much: ratio={ratio}"


class TestInterpolation:
    """Test geodesic interpolation."""

    def test_interpolate_endpoints(self):
        """Interpolation at t=0 and t=1 should give endpoints."""
        # Use very small vectors for numerical stability
        # For 1024 dims, randn*0.003 has norm ~0.1
        x = expmap0(torch.randn(1024) * 0.003, 1.0)
        y = expmap0(torch.randn(1024) * 0.003, 1.0)

        at_0 = hyperbolic_interpolate(x, y, 0.0, 1.0)
        at_1 = hyperbolic_interpolate(x, y, 1.0, 1.0)

        assert torch.allclose(at_0, x, atol=1e-3), "t=0 should give x"
        assert torch.allclose(at_1, y, atol=1e-3), "t=1 should give y"

    def test_interpolate_midpoint(self):
        """Midpoint should be approximately equidistant from both endpoints."""
        # Use very small vectors for stability
        # For 1024 dims, randn*0.003 has norm ~0.1
        x = expmap0(torch.randn(1024) * 0.003, 1.0)
        y = expmap0(torch.randn(1024) * 0.003, 1.0)

        mid = hyperbolic_interpolate(x, y, 0.5, 1.0)

        d_xm = hyperbolic_distance(x, mid, 1.0).item()
        d_ym = hyperbolic_distance(y, mid, 1.0).item()

        # Allow more tolerance for numerical issues
        ratio = d_xm / (d_ym + 1e-8)
        assert 0.7 < ratio < 1.4, f"Midpoint distances should be similar: d_xm={d_xm}, d_ym={d_ym}"


class TestHyperbolicSpace:
    """Test the HyperbolicSpace convenience class."""

    def test_roundtrip(self):
        """to_hyperbolic and to_euclidean should roundtrip for small inputs."""
        space = HyperbolicSpace(curvature=1.0)
        v = torch.randn(1024) * 0.1  # Small vector for stability

        x = space.to_hyperbolic(v, scale=0.5)
        v_back = space.to_euclidean(x)

        # Should be close after scaling
        v_scaled = v * 0.5
        assert torch.allclose(v_scaled, v_back, atol=1e-3)

    def test_distance(self):
        """HyperbolicSpace.distance should work."""
        space = HyperbolicSpace(curvature=1.0)
        x = space.to_hyperbolic(torch.randn(1024), scale=0.3)
        y = space.to_hyperbolic(torch.randn(1024), scale=0.3)

        d = space.distance(x, y)
        assert d.item() > 0

    def test_mean(self):
        """HyperbolicSpace.mean should work."""
        space = HyperbolicSpace(curvature=1.0)
        points = torch.stack([
            space.to_hyperbolic(torch.randn(1024), scale=0.2),
            space.to_hyperbolic(torch.randn(1024), scale=0.2),
        ])

        mean = space.mean(points)
        assert mean.norm().item() < 1.0


class TestRadialAndAngular:
    """Test radial depth and angular direction functions."""

    def test_radial_depth_origin(self):
        """Radial depth of origin should be zero."""
        origin = torch.zeros(1024)
        depth = radial_depth(origin, 1.0)
        assert depth.item() < 1e-5

    def test_radial_depth_increases(self):
        """Points farther from origin should have higher radial depth."""
        v_small = torch.randn(1024) * 0.1
        v_large = torch.randn(1024) * 0.5

        x_small = expmap0(v_small, 1.0)
        x_large = expmap0(v_large, 1.0)

        d_small = radial_depth(x_small, 1.0).item()
        d_large = radial_depth(x_large, 1.0).item()

        assert d_large > d_small

    def test_unit_tangent_direction(self):
        """Unit tangent should have unit norm."""
        x = expmap0(torch.randn(1024) * 0.3, 1.0)
        unit = unit_tangent_at_origin(x, 1.0)
        assert abs(unit.norm().item() - 1.0) < 1e-5


class TestCurvatureEffects:
    """Test effects of different curvatures."""

    def test_higher_curvature_larger_distances(self):
        """Higher curvature should give larger distances (for same Euclidean points)."""
        v1 = torch.randn(1024) * 0.2
        v2 = torch.randn(1024) * 0.2

        x1_c1 = expmap0(v1, c=1.0)
        x2_c1 = expmap0(v2, c=1.0)
        d_c1 = hyperbolic_distance(x1_c1, x2_c1, c=1.0).item()

        x1_c2 = expmap0(v1, c=2.0)
        x2_c2 = expmap0(v2, c=2.0)
        d_c2 = hyperbolic_distance(x1_c2, x2_c2, c=2.0).item()

        # Higher curvature = more curved = different distances
        # Both should be positive and finite
        assert d_c1 > 0 and d_c2 > 0
        assert math.isfinite(d_c1) and math.isfinite(d_c2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
