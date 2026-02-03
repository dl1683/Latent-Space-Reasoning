"""
Hyperbolic geometry operations for latent space evolution.

This module implements Poincaré ball model operations for hyperbolic latent
space evolution. Hyperbolic space naturally matches hierarchical reasoning
structures due to its exponential volume growth.

Key Operations:
- expmap0: Map from tangent space at origin to Poincaré ball
- logmap0: Map from Poincaré ball to tangent space at origin
- expmap: Map from tangent space at point p to Poincaré ball
- logmap: Map from Poincaré ball to tangent space at point p
- mobius_add: Möbius addition (hyperbolic translation)
- mobius_scalar_mul: Möbius scalar multiplication
- parallel_transport: Move tangent vectors between points
- hyperbolic_distance: Geodesic distance in hyperbolic space
- project_to_ball: Ensure points stay inside the ball
- karcher_mean: Weighted barycenter (Fréchet mean)

Mathematical Background:
The Poincaré ball model B^n = {x ∈ R^n : ||x|| < 1} with metric:
    g_x = (2/(1-||x||^2))^2 * I

has curvature -c (we parameterize by c > 0).

Reference: "Hyperbolic Neural Networks" (Ganea et al., 2018)
"""

from __future__ import annotations

import torch
from torch import Tensor
import math


# Small epsilon for numerical stability
EPS = 1e-15
MIN_NORM = 1e-15


def _clamp_norm(x: Tensor, max_norm: float = 0.98) -> Tensor:
    """Clamp tensor norm to stay inside the Poincaré ball."""
    norm = x.norm(dim=-1, keepdim=True).clamp(min=MIN_NORM)
    desired = torch.clamp(norm, max=max_norm)
    return x * (desired / norm)


def project_to_ball(x: Tensor, c: float = 1.0, max_norm: float = 0.98) -> Tensor:
    """
    Project points to be inside the Poincaré ball of radius 1/sqrt(c).

    Args:
        x: Points to project (..., dim)
        c: Curvature (positive, ball radius = 1/sqrt(c))
        max_norm: Maximum allowed norm (< 1/sqrt(c) for stability)

    Returns:
        Projected points inside the ball
    """
    max_radius = (1.0 / math.sqrt(c)) * max_norm
    norm = x.norm(dim=-1, keepdim=True).clamp(min=MIN_NORM)
    factor = torch.clamp(max_radius / norm, max=1.0)
    return x * factor


def _lambda_x(x: Tensor, c: float = 1.0) -> Tensor:
    """Conformal factor λ_x = 2 / (1 - c||x||^2)."""
    x_sqnorm = (x * x).sum(dim=-1, keepdim=True).clamp(min=0, max=1/c - EPS)
    return 2.0 / (1.0 - c * x_sqnorm).clamp(min=EPS)


def mobius_add(x: Tensor, y: Tensor, c: float = 1.0) -> Tensor:
    """
    Möbius addition in the Poincaré ball: x ⊕_c y.

    This is the hyperbolic analog of vector addition.

    Args:
        x: First point (..., dim)
        y: Second point (..., dim)
        c: Curvature

    Returns:
        x ⊕_c y in the Poincaré ball
    """
    x_sqnorm = (x * x).sum(dim=-1, keepdim=True).clamp(min=0)
    y_sqnorm = (y * y).sum(dim=-1, keepdim=True).clamp(min=0)
    xy_inner = (x * y).sum(dim=-1, keepdim=True)

    num = (1 + 2*c*xy_inner + c*y_sqnorm) * x + (1 - c*x_sqnorm) * y
    denom = 1 + 2*c*xy_inner + c*c*x_sqnorm*y_sqnorm

    result = num / denom.clamp(min=EPS)
    return project_to_ball(result, c)


def mobius_scalar_mul(r: float | Tensor, x: Tensor, c: float = 1.0) -> Tensor:
    """
    Möbius scalar multiplication: r ⊗_c x.

    Args:
        r: Scalar (or tensor broadcastable to x)
        x: Point in Poincaré ball (..., dim)
        c: Curvature

    Returns:
        r ⊗_c x in the Poincaré ball
    """
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=MIN_NORM)
    sqrt_c = math.sqrt(c)

    # r ⊗ x = (1/sqrt(c)) * tanh(r * arctanh(sqrt(c) * ||x||)) * (x / ||x||)
    scaled_norm = sqrt_c * x_norm
    scaled_norm = scaled_norm.clamp(max=1.0 - EPS)  # Ensure valid for arctanh

    arctanh_norm = torch.arctanh(scaled_norm)
    new_norm = torch.tanh(r * arctanh_norm) / sqrt_c

    result = new_norm * (x / x_norm)
    return project_to_ball(result, c)


def expmap0(v: Tensor, c: float = 1.0) -> Tensor:
    """
    Exponential map at the origin: maps tangent vector to Poincaré ball.

    This is the core operation for mapping Euclidean latents to hyperbolic space.

    Args:
        v: Tangent vector at origin (..., dim)
        c: Curvature

    Returns:
        Point in Poincaré ball
    """
    sqrt_c = math.sqrt(c)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=MIN_NORM)

    # exp_0(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
    result = torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
    return project_to_ball(result, c)


def logmap0(y: Tensor, c: float = 1.0) -> Tensor:
    """
    Logarithmic map at the origin: maps Poincaré ball to tangent space.

    This is the inverse of expmap0 - used for decoding and scoring.

    Args:
        y: Point in Poincaré ball (..., dim)
        c: Curvature

    Returns:
        Tangent vector at origin
    """
    sqrt_c = math.sqrt(c)
    y_norm = y.norm(dim=-1, keepdim=True).clamp(min=MIN_NORM)

    # Clamp to valid range for arctanh
    scaled_norm = (sqrt_c * y_norm).clamp(max=1.0 - EPS)

    # log_0(y) = arctanh(sqrt(c) * ||y||) * y / (sqrt(c) * ||y||)
    result = torch.arctanh(scaled_norm) * y / (sqrt_c * y_norm)
    return result


def expmap(v: Tensor, p: Tensor, c: float = 1.0) -> Tensor:
    """
    Exponential map at point p: maps tangent vector at p to Poincaré ball.

    Args:
        v: Tangent vector at p (..., dim)
        p: Base point (..., dim)
        c: Curvature

    Returns:
        Point in Poincaré ball
    """
    # exp_p(v) = p ⊕ (tanh(λ_p * sqrt(c) * ||v|| / 2) * v / (sqrt(c) * ||v||))
    sqrt_c = math.sqrt(c)
    lambda_p = _lambda_x(p, c)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=MIN_NORM)

    second_term = torch.tanh(lambda_p * sqrt_c * v_norm / 2) * v / (sqrt_c * v_norm)
    result = mobius_add(p, second_term, c)
    return project_to_ball(result, c)


def logmap(y: Tensor, p: Tensor, c: float = 1.0) -> Tensor:
    """
    Logarithmic map at point p: maps Poincaré ball point y to tangent space at p.

    Args:
        y: Point in Poincaré ball (..., dim)
        p: Base point (..., dim)
        c: Curvature

    Returns:
        Tangent vector at p
    """
    # log_p(y) = (2 / (λ_p * sqrt(c))) * arctanh(sqrt(c) * ||-p ⊕ y||) * (-p ⊕ y) / ||-p ⊕ y||
    sqrt_c = math.sqrt(c)
    lambda_p = _lambda_x(p, c)

    # -p ⊕ y
    neg_p = -p
    diff = mobius_add(neg_p, y, c)
    diff_norm = diff.norm(dim=-1, keepdim=True).clamp(min=MIN_NORM)

    # Clamp for arctanh
    scaled_norm = (sqrt_c * diff_norm).clamp(max=1.0 - EPS)

    result = (2 / (lambda_p * sqrt_c)) * torch.arctanh(scaled_norm) * diff / diff_norm
    return result


def parallel_transport(v: Tensor, p: Tensor, q: Tensor, c: float = 1.0) -> Tensor:
    """
    Parallel transport tangent vector v from p to q.

    Used to move modification hints from origin to candidate point.

    Args:
        v: Tangent vector at p (..., dim)
        p: Source point (..., dim)
        q: Target point (..., dim)
        c: Curvature

    Returns:
        Tangent vector at q
    """
    # Parallel transport via λ ratio
    lambda_p = _lambda_x(p, c)
    lambda_q = _lambda_x(q, c)

    return v * (lambda_p / lambda_q)


def hyperbolic_distance(x: Tensor, y: Tensor, c: float = 1.0) -> Tensor:
    """
    Compute geodesic distance between points in the Poincaré ball.

    Args:
        x: First point (..., dim)
        y: Second point (..., dim)
        c: Curvature

    Returns:
        Distance tensor (...,)
    """
    sqrt_c = math.sqrt(c)

    # d(x, y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||-x ⊕ y||)
    diff = mobius_add(-x, y, c)
    diff_norm = diff.norm(dim=-1)

    # Clamp for numerical stability
    scaled_norm = (sqrt_c * diff_norm).clamp(max=1.0 - EPS)

    return (2 / sqrt_c) * torch.arctanh(scaled_norm)


def pairwise_hyperbolic_distance(x: Tensor, c: float = 1.0) -> Tensor:
    """
    Compute pairwise hyperbolic distances for a batch of points.

    Args:
        x: Batch of points (n, dim)
        c: Curvature

    Returns:
        Distance matrix (n, n)
    """
    n = x.shape[0]
    distances = torch.zeros(n, n, device=x.device, dtype=x.dtype)

    for i in range(n):
        for j in range(i + 1, n):
            d = hyperbolic_distance(x[i], x[j], c)
            distances[i, j] = d
            distances[j, i] = d

    return distances


def karcher_mean(
    points: Tensor,
    weights: Tensor | None = None,
    c: float = 1.0,
    max_iters: int = 10,
    tol: float = 1e-6,
) -> Tensor:
    """
    Compute weighted Fréchet/Karcher mean (hyperbolic barycenter).

    This is used for hyperbolic crossover - combining parent latents.

    Args:
        points: Points to average (n, dim)
        weights: Weights for each point (n,), normalized internally
        c: Curvature
        max_iters: Maximum iterations for gradient descent
        tol: Convergence tolerance

    Returns:
        Karcher mean point (dim,)
    """
    n, dim = points.shape

    if weights is None:
        weights = torch.ones(n, device=points.device, dtype=points.dtype) / n
    else:
        weights = weights / weights.sum()  # Normalize

    # Initialize at weighted Euclidean mean projected to ball
    mean = (weights.unsqueeze(1) * points).sum(dim=0)
    mean = project_to_ball(mean.unsqueeze(0), c, max_norm=0.5).squeeze(0)

    # Riemannian gradient descent
    for _ in range(max_iters):
        # Compute gradient: sum of weighted log maps
        grad = torch.zeros(dim, device=points.device, dtype=points.dtype)
        for i in range(n):
            log_vec = logmap(points[i], mean.unsqueeze(0), c).squeeze(0)
            grad = grad + weights[i] * log_vec

        # Check convergence
        if grad.norm() < tol:
            break

        # Step in gradient direction (Riemannian gradient descent)
        # Step size chosen for stability
        step_size = 0.5
        mean = expmap(step_size * grad.unsqueeze(0), mean.unsqueeze(0), c).squeeze(0)
        mean = project_to_ball(mean.unsqueeze(0), c).squeeze(0)

    return mean


def hyperbolic_midpoint(x: Tensor, y: Tensor, c: float = 1.0) -> Tensor:
    """
    Compute hyperbolic midpoint between two points.

    Args:
        x: First point (..., dim)
        y: Second point (..., dim)
        c: Curvature

    Returns:
        Midpoint in Poincaré ball
    """
    return karcher_mean(
        torch.stack([x.squeeze(), y.squeeze()]),
        weights=None,
        c=c,
        max_iters=5,
    )


def radial_depth(x: Tensor, c: float = 1.0) -> Tensor:
    """
    Compute radial depth (distance from origin) in hyperbolic space.

    Useful for behavioral descriptors.

    Args:
        x: Points in Poincaré ball (..., dim)
        c: Curvature

    Returns:
        Radial depth (...,)
    """
    origin = torch.zeros_like(x)
    return hyperbolic_distance(origin, x, c)


def unit_tangent_at_origin(x: Tensor, c: float = 1.0) -> Tensor:
    """
    Get unit tangent direction from origin to point.

    Useful for angular component of behavioral descriptors.

    Args:
        x: Point in Poincaré ball (..., dim)
        c: Curvature

    Returns:
        Unit tangent vector at origin (..., dim)
    """
    log_vec = logmap0(x, c)
    norm = log_vec.norm(dim=-1, keepdim=True).clamp(min=MIN_NORM)
    return log_vec / norm


def hyperbolic_interpolate(
    x: Tensor,
    y: Tensor,
    t: float,
    c: float = 1.0,
) -> Tensor:
    """
    Interpolate along geodesic from x to y.

    Args:
        x: Start point (..., dim)
        y: End point (..., dim)
        t: Interpolation parameter in [0, 1]
        c: Curvature

    Returns:
        Interpolated point
    """
    # Get direction from x to y in tangent space at x
    log_xy = logmap(y, x, c)
    # Scale by t and map back
    return expmap(t * log_xy, x, c)


class HyperbolicSpace:
    """
    Hyperbolic space manager for latent evolution.

    Provides a convenient interface for all hyperbolic operations
    with consistent curvature and numerical stability settings.

    Args:
        curvature: Negative curvature magnitude (default 1.0)
        max_norm: Maximum allowed norm (for numerical stability)
        device: Torch device
    """

    def __init__(
        self,
        curvature: float = 1.0,
        max_norm: float = 0.98,
        device: torch.device | str = "cpu",
    ):
        self.c = curvature
        self.max_norm = max_norm
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

    def to_hyperbolic(self, v: Tensor, scale: float = 1.0) -> Tensor:
        """Map Euclidean vector to Poincaré ball via expmap0."""
        v = v.to(self.device)
        scaled = v * scale
        result = expmap0(scaled, self.c)
        return project_to_ball(result, self.c, self.max_norm)

    def to_euclidean(self, x: Tensor) -> Tensor:
        """Map Poincaré ball point back to Euclidean via logmap0."""
        x = x.to(self.device)
        return logmap0(x, self.c)

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        """Möbius addition."""
        return mobius_add(x, y, self.c)

    def scale(self, r: float, x: Tensor) -> Tensor:
        """Möbius scalar multiplication."""
        return mobius_scalar_mul(r, x, self.c)

    def distance(self, x: Tensor, y: Tensor) -> Tensor:
        """Hyperbolic distance."""
        return hyperbolic_distance(x, y, self.c)

    def midpoint(self, x: Tensor, y: Tensor) -> Tensor:
        """Hyperbolic midpoint."""
        return hyperbolic_midpoint(x, y, self.c)

    def mean(self, points: Tensor, weights: Tensor | None = None) -> Tensor:
        """Karcher mean (hyperbolic barycenter)."""
        return karcher_mean(points, weights, self.c)

    def expmap(self, v: Tensor, p: Tensor) -> Tensor:
        """Exponential map at point p."""
        return expmap(v, p, self.c)

    def logmap(self, y: Tensor, p: Tensor) -> Tensor:
        """Logarithmic map at point p."""
        return logmap(y, p, self.c)

    def transport(self, v: Tensor, p: Tensor, q: Tensor) -> Tensor:
        """Parallel transport from p to q."""
        return parallel_transport(v, p, q, self.c)

    def project(self, x: Tensor) -> Tensor:
        """Project to ball interior."""
        return project_to_ball(x, self.c, self.max_norm)

    def radial_depth(self, x: Tensor) -> Tensor:
        """Distance from origin."""
        return radial_depth(x, self.c)

    def unit_direction(self, x: Tensor) -> Tensor:
        """Unit tangent direction from origin."""
        return unit_tangent_at_origin(x, self.c)

    def interpolate(self, x: Tensor, y: Tensor, t: float) -> Tensor:
        """Geodesic interpolation."""
        return hyperbolic_interpolate(x, y, t, self.c)

    def __repr__(self) -> str:
        return f"HyperbolicSpace(curvature={self.c}, max_norm={self.max_norm})"
