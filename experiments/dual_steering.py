"""
Dual Steering Decode - Logit-level Newton steering via information geometry.

Based on "The Information Geometry of Softmax: Probing and Steering"
(arXiv:2602.15293, Feb 2026). The paper proves that naive vector addition in
logit space commits a geometric type error. The correct approach is a
regularized Newton step in the dual (probability) coordinate system.

This module provides:
- DualSteeringProcessor: A HuggingFace LogitsProcessor that applies per-token
  Newton steering with KL safety cap
- make_steer_projection: Creates a fixed orthonormal projection W_steer
  from latent tangent space to model hidden space
- compute_steering_direction: Full pipeline from Poincare ball latent to
  vocabulary-sized steering direction omega_W

Codex review fixes incorporated:
1. No epsilon parameter (mathematically inert due to L2 normalization)
2. No tanh_squash in steering branch (only NaN/Inf guard + final L2-norm)
3. KL cap with automatic eta downscaling (prevents temperature interaction)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


class DualSteeringProcessor:
    """Per-token regularized Newton steering in the dual coordinate system.

    Plugs into model.generate(logits_processor=[processor]).

    The Woodbury identity solves (diag(p+alpha) - pp^T)^{-1} @ omega_W
    in O(V) per token instead of O(V^2).

    Args:
        omega_W: (V,) unit steering direction in logit space, computed ONCE
        eta: Step size for the Newton direction (default 0.05, conservative)
        alpha: Tikhonov regularization for the Fisher information (default 0.01)
        kl_cap: Maximum per-token KL divergence before auto-downscaling eta
    """

    def __init__(
        self,
        omega_W: Tensor,
        eta: float = 0.05,
        alpha: float = 0.01,
        kl_cap: float = 0.5,
    ):
        self.omega_W = omega_W
        self.eta = eta
        self.alpha = alpha
        self.kl_cap = kl_cap
        self.kl_triggered_count = 0
        self.total_tokens = 0

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        """Apply dual steering to logits. Called per generated token.

        Args:
            input_ids: (batch, seq_len) generated token IDs so far
            scores: (batch, vocab_size) raw logits for next token

        Returns:
            Steered logits (batch, vocab_size)
        """
        self.total_tokens += 1

        if self.eta == 0.0:
            return scores

        # Work in float32 for numerical stability
        logits = scores.float()
        omega = self.omega_W.to(logits.device).float()

        # p = softmax(logits)
        p = F.softmax(logits, dim=-1)  # (batch, V)

        # Woodbury solve: v = (diag(p+alpha) - pp^T)^{-1} @ omega_W
        # Using Sherman-Morrison: (D - uu^T)^{-1} b = D^{-1}b + D^{-1}u (u^T D^{-1}b) / (1 - u^T D^{-1}u)
        # where D = diag(p + alpha), u = p
        d = p + self.alpha  # (batch, V)
        d_inv_omega = omega / d  # (batch, V)
        d_inv_p = p / d  # (batch, V)

        # scalar = p^T D^{-1} omega
        scalar = (p * d_inv_omega).sum(dim=-1, keepdim=True)  # (batch, 1)
        # denom = 1 - p^T D^{-1} p
        denom = (1.0 - (p * d_inv_p).sum(dim=-1, keepdim=True)).clamp(min=1e-8)

        v = d_inv_omega + d_inv_p * (scalar / denom)  # (batch, V)

        # L2-normalize the Newton direction
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v_hat = v / v_norm

        # Apply steering
        logits_steered = logits + self.eta * v_hat

        # KL safety cap: auto-downscale eta if KL exceeds cap
        if self.kl_cap > 0:
            p_steered = F.softmax(logits_steered, dim=-1)
            # KL(p || p_steered) - using log for numerical stability
            log_p = torch.log(p.clamp(min=1e-10))
            log_p_steered = torch.log(p_steered.clamp(min=1e-10))
            kl = (p * (log_p - log_p_steered)).sum(dim=-1, keepdim=True)

            if (kl > self.kl_cap).any():
                self.kl_triggered_count += 1
                # Scale down eta to bring KL within budget
                eta_scale = (self.kl_cap / kl.clamp(min=1e-8)).sqrt().clamp(max=1.0)
                logits_steered = logits + self.eta * eta_scale * v_hat

        return logits_steered.to(scores.dtype)


def make_steer_projection(
    d_latent: int,
    d_hidden: int,
    seed: int = 5678,
) -> Tensor:
    """Create a fixed row-orthonormal projection for steering.

    Maps from latent tangent space (d_latent) to model hidden space (d_hidden).
    W has shape (d_latent, d_hidden) with orthonormal rows: W W^T = I.

    This reuses the same QR construction as V13's make_row_orthonormal_W
    but targets hidden dim instead of soft prompt flat dim.

    Args:
        d_latent: Latent dimension (e.g. 2560 for Qwen3-4B)
        d_hidden: Model hidden dimension (e.g. 2560 for Qwen3-4B)
        seed: RNG seed (must differ from soft prompt projection seed)

    Returns:
        W_steer: (d_latent, d_hidden) row-orthonormal projection
    """
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d_hidden, d_latent, generator=g, dtype=torch.float32)
    Q, _ = torch.linalg.qr(A, mode="reduced")  # Q: (d_hidden, d_latent)
    W = Q.T.contiguous()  # W: (d_latent, d_hidden), rows orthonormal
    return W


def compute_steering_direction(
    latent: Tensor,
    W_steer: Tensor,
    lm_head_weight: Tensor,
    curvature: float,
    device: torch.device,
) -> Tensor:
    """Compute the steering direction omega_W from a Poincare ball latent.

    Pipeline:
    1. logmap0(latent, c) -> tangent space
    2. NaN/Inf guard (no tanh_squash - Codex fix #3)
    3. W_steer projection: tangent -> hidden space
    4. lm_head forward: hidden -> vocab logits
    5. L2-normalize -> unit steering direction omega_W

    Args:
        latent: Point in Poincare ball (latent_dim,) or (1, latent_dim)
        W_steer: (latent_dim, d_hidden) orthonormal projection
        lm_head_weight: (vocab_size, d_hidden) the model's output projection
        curvature: Poincare ball curvature
        device: Target device

    Returns:
        omega_W: (vocab_size,) unit steering direction
    """
    from latent_reasoning.utils import hyperbolic as hyp

    lat = latent.squeeze().float()

    # 1. Map to tangent space at origin
    tangent = hyp.logmap0(lat, curvature)

    # 2. NaN/Inf guard only (no tanh_squash - normalization at end handles scale)
    tangent = torch.nan_to_num(tangent, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Project to hidden space: (d_latent,) @ (d_latent, d_hidden) -> (d_hidden,)
    W = W_steer.to(device=device, dtype=torch.float32)
    hidden_vec = tangent.to(device) @ W  # (d_hidden,)

    # 4. Route through lm_head: (d_hidden,) @ (d_hidden, vocab) -> (vocab,)
    # lm_head_weight is (vocab, d_hidden), so we need hidden @ weight^T
    lm_weight = lm_head_weight.float().to(device)
    logit_vec = hidden_vec @ lm_weight.T  # (vocab_size,)

    # 5. L2-normalize to unit steering direction
    norm = logit_vec.norm().clamp(min=1e-8)
    omega_W = logit_vec / norm

    return omega_W
