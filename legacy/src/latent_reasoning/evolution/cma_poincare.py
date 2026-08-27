"""Low-rank CMA-ES on the Poincare ball.

All covariance operations happen in tangent space at the distributional mean.
Sampling: tangent noise -> expmap at mean -> Poincare ball point.
Update: logmap candidates to tangent at mean -> weighted recombination ->
rank-mu covariance update.

The low-rank representation C = sigma^2 * (I + U U^T) where U is
(dim, rank) keeps memory at O(dim*rank) instead of O(dim^2), making
this tractable for d_latent = 1024.

Reference:
- CMA-ES: Hansen & Ostermeier (2001)
- Riemannian CMA: Colutto et al. (2010)
- EGGROLL: Low-rank ES at billion scale (NeurIPS 2025)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from latent_reasoning.utils import hyperbolic as hyp


class PoincareCMA:
    """Low-rank CMA-ES operating on the Poincare ball.

    Maintains a distribution N(mean, sigma^2 * (I + U U^T)) in tangent
    space at ``mean``, then maps samples to the Poincare ball via expmap.
    """

    def __init__(
        self,
        dim: int,
        population_size: int = 8,
        rank: int = 10,
        curvature: float = 0.5,
        sigma: float = 0.3,
        max_norm: float = 0.95,
    ):
        self.dim = dim
        self.lam = population_size  # lambda (population size)
        self.rank = min(rank, dim)
        self.curvature = curvature
        self.sigma = sigma
        self.max_norm = max_norm

        # Distributional mean — start at origin
        self.mean = torch.zeros(dim)

        # Low-rank factor U: (dim, rank)
        self.U = torch.zeros(dim, self.rank)

        # Recombination weights (log-linear) — must be before adaptation params
        mu = self.lam // 2
        raw_w = [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]
        total_w = sum(raw_w)
        self.weights = torch.tensor([w / total_w for w in raw_w])
        self.mu_eff = 1.0 / (self.weights ** 2).sum().item()

        # Step size adaptation
        self.p_sigma = torch.zeros(dim)  # Evolution path for sigma
        self.c_sigma = 0.3  # Learning rate for p_sigma
        self.d_sigma = 1.0  # Damping for sigma

        # Covariance adaptation
        self.p_c = torch.zeros(dim)  # Evolution path for covariance
        self.c_c = 4.0 / (dim + 4)  # Learning rate for p_c
        self.c_1 = 2.0 / ((dim + 1.3) ** 2)  # Rank-one update weight
        self.c_mu = min(
            1 - self.c_1,
            2 * (self.mu_eff - 2 + 1.0 / self.mu_eff)
            / ((dim + 2) ** 2 + self.mu_eff),
        )

        # Expected norm of N(0, I)
        self.chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        self.generation = 0

    def sample(self, rng: Optional[torch.Generator] = None) -> List[Tensor]:
        """Sample lambda candidates on the Poincare ball.

        Returns list of tensors, each shape (dim,), inside the ball.
        """
        candidates = []
        for _ in range(self.lam):
            # Sample in tangent space: z ~ N(0, I + U U^T)
            z = torch.randn(self.dim, generator=rng)
            if self.U.norm() > 1e-8:
                # Add low-rank component: z + U @ (U^T @ z_extra)
                z_extra = torch.randn(self.rank, generator=rng)
                z = z + self.U @ z_extra

            tangent = self.sigma * z

            # Map to Poincare ball via expmap at mean
            point = hyp.expmap(
                tangent.unsqueeze(0), self.mean.unsqueeze(0), self.curvature,
            ).squeeze(0)
            point = hyp.project_to_ball(point, self.curvature, self.max_norm)
            candidates.append(point)

        return candidates

    def update(
        self,
        candidates: List[Tensor],
        fitnesses: List[float],
    ) -> None:
        """Update distribution parameters from evaluated candidates.

        Args:
            candidates: List of Poincare ball points, length = lambda.
            fitnesses: Corresponding fitness values (higher = better).
        """
        assert len(candidates) == self.lam
        assert len(fitnesses) == self.lam

        self.generation += 1
        mu = len(self.weights)

        # Sort by fitness (descending)
        order = sorted(range(self.lam), key=lambda i: fitnesses[i], reverse=True)

        # Map candidates to tangent space at old mean
        tangent_vecs = []
        for idx in order[:mu]:
            tv = hyp.logmap(
                candidates[idx].unsqueeze(0),
                self.mean.unsqueeze(0),
                self.curvature,
            ).squeeze(0)
            tv = torch.nan_to_num(tv, nan=0.0, posinf=0.0, neginf=0.0)
            tangent_vecs.append(tv)

        # Weighted mean shift in tangent space
        step = torch.zeros(self.dim)
        for i, tv in enumerate(tangent_vecs):
            step = step + self.weights[i] * tv

        # Update mean: move along geodesic from old mean
        new_mean = hyp.expmap(
            step.unsqueeze(0), self.mean.unsqueeze(0), self.curvature,
        ).squeeze(0)
        self.mean = hyp.project_to_ball(new_mean, self.curvature, self.max_norm)

        # Normalized step (for adaptation)
        y_w = step / max(self.sigma, 1e-8)

        # Update evolution path for sigma (cumulative step-size adaptation)
        self.p_sigma = (
            (1 - self.c_sigma) * self.p_sigma
            + math.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * y_w
        )

        # Update sigma
        p_sigma_norm = self.p_sigma.norm().item()
        self.sigma *= math.exp(
            (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1)
        )
        self.sigma = max(1e-8, min(self.sigma, 10.0))  # Clamp

        # Update evolution path for covariance
        h_sig = 1.0 if p_sigma_norm / math.sqrt(
            1 - (1 - self.c_sigma) ** (2 * self.generation)
        ) < (1.4 + 2.0 / (self.dim + 1)) * self.chi_n else 0.0

        self.p_c = (
            (1 - self.c_c) * self.p_c
            + h_sig * math.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_w
        )

        # Rank-mu update of U (low-rank covariance factor)
        # Accumulate weighted outer products into a (dim, rank) factor
        # We use a simplified approach: keep top-rank singular vectors of
        # the weighted sample matrix
        sample_matrix = torch.stack([
            math.sqrt(self.weights[i].item()) * (tangent_vecs[i] / max(self.sigma, 1e-8))
            for i in range(mu)
        ], dim=1)  # (dim, mu)

        # Include rank-one path
        if h_sig > 0 and self.c_1 > 0:
            rank_one = math.sqrt(self.c_1) * self.p_c.unsqueeze(1)  # (dim, 1)
            combined = torch.cat([
                math.sqrt(self.c_mu) * sample_matrix,
                rank_one,
            ], dim=1)  # (dim, mu+1)
        else:
            combined = math.sqrt(self.c_mu) * sample_matrix

        # Low-rank truncation via SVD
        if combined.shape[1] > 0:
            try:
                U_new, S, _ = torch.linalg.svd(combined, full_matrices=False)
                k = min(self.rank, U_new.shape[1])
                # Exponential moving average of low-rank factor
                decay = 1 - self.c_1 - self.c_mu
                self.U = decay * self.U + U_new[:, :k] * S[:k].unsqueeze(0)
            except Exception:
                pass  # SVD can fail for degenerate matrices; keep old U

    @property
    def best_point(self) -> Tensor:
        """Current distributional mean as the best estimate."""
        return self.mean.clone()
