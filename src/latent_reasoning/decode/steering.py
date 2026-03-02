"""Dual steering decode -- logit-level Newton steering via information geometry.

Based on "The Information Geometry of Softmax: Probing and Steering"
(arXiv:2602.15293, Feb 2026).  Naive vector addition in logit space commits
a geometric type error; the correct approach is a regularised Newton step in
the dual (probability) coordinate system.

Provides:
- DualSteeringProcessor: HuggingFace LogitsProcessor for per-token Newton
  steering with KL safety cap.
- make_steer_projection: Fixed orthonormal projection W_steer from latent
  tangent space to model hidden space.
- compute_steering_direction: Full pipeline from Poincare ball latent to
  vocabulary-sized steering direction omega_W.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class DualSteeringProcessor:
    """Per-token regularised Newton steering in the dual coordinate system.

    Plugs into ``model.generate(logits_processor=[processor])``.

    The Woodbury identity solves ``(diag(p+alpha) - pp^T)^{-1} @ omega_W``
    in O(V) per token instead of O(V^2).
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
        self.total_tokens += 1

        if self.eta == 0.0:
            return scores

        logits = scores.float()
        omega = self.omega_W.to(logits.device).float()

        p = F.softmax(logits, dim=-1)

        # Woodbury solve via Sherman-Morrison
        d = p + self.alpha
        d_inv_omega = omega / d
        d_inv_p = p / d

        scalar = (p * d_inv_omega).sum(dim=-1, keepdim=True)
        denom = (1.0 - (p * d_inv_p).sum(dim=-1, keepdim=True)).clamp(min=1e-8)

        v = d_inv_omega + d_inv_p * (scalar / denom)

        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v_hat = v / v_norm

        logits_steered = logits + self.eta * v_hat

        if self.kl_cap > 0:
            p_steered = F.softmax(logits_steered, dim=-1)
            log_p = torch.log(p.clamp(min=1e-10))
            log_p_steered = torch.log(p_steered.clamp(min=1e-10))
            kl = (p * (log_p - log_p_steered)).sum(dim=-1, keepdim=True)

            if (kl > self.kl_cap).any():
                self.kl_triggered_count += 1
                eta_scale = (self.kl_cap / kl.clamp(min=1e-8)).sqrt().clamp(max=1.0)
                logits_steered = logits + self.eta * eta_scale * v_hat

        return logits_steered.to(scores.dtype)


def make_steer_projection(
    d_latent: int,
    d_hidden: int,
    seed: int = 5678,
) -> Tensor:
    """Create a fixed row-orthonormal projection for steering.

    Maps from latent tangent space (*d_latent*) to model hidden space
    (*d_hidden*).  Uses the same QR construction as the soft-prompt
    projection but targets hidden dim instead.
    """
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d_hidden, d_latent, generator=g, dtype=torch.float32)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    W = Q.T.contiguous()
    return W


def compute_steering_direction(
    latent: Tensor,
    W_steer: Tensor,
    lm_head_weight: Tensor,
    curvature: float,
    device: torch.device,
) -> Tensor:
    """Compute the unit steering direction omega_W from a Poincare ball latent.

    Pipeline: logmap0 -> NaN guard -> W_steer projection -> lm_head -> L2 norm.
    """
    from latent_reasoning.utils import hyperbolic as hyp

    lat = latent.squeeze().float()
    tangent = hyp.logmap0(lat, curvature)
    tangent = torch.nan_to_num(tangent, nan=0.0, posinf=0.0, neginf=0.0)

    W = W_steer.to(device=device, dtype=torch.float32)
    hidden_vec = tangent.to(device) @ W

    lm_weight = lm_head_weight.float().to(device)
    logit_vec = hidden_vec @ lm_weight.T

    norm = logit_vec.norm().clamp(min=1e-8)
    return logit_vec / norm
