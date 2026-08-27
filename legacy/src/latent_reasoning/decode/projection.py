"""Fixed orthogonal projection from latent space to soft prompt tokens.

Core V13 innovation: a fixed row-orthonormal matrix W projects from the
latent tangent space (e.g. 1024d) to soft prompt tokens (e.g. 8 x 2560 =
20,480 continuous values).  ~650x more information bandwidth than the 31-bit
RNG-seed conditioning mechanism.

The projection does NOT need training.  Row-orthonormality preserves inner
products (Johnson-Lindenstrauss), so similar latents produce similar soft
prompts and different latents produce different prompts.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from latent_reasoning.utils import hyperbolic as hyp


def make_row_orthonormal_W(
    d_latent: int,
    d_out: int,
    seed: int = 1234,
) -> Tensor:
    """Create a fixed row-orthonormal projection matrix W.

    W has shape (d_latent, d_out) with orthonormal rows: W W^T = I.
    Construction: generate random (d_out, d_latent) matrix, QR decompose,
    take Q^T as W.
    """
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d_out, d_latent, generator=g, dtype=torch.float32)
    Q, _ = torch.linalg.qr(A, mode="reduced")  # Q: (d_out, d_latent)
    W = Q.T.contiguous()  # W: (d_latent, d_out), rows orthonormal
    return W


def radial_tanh_squash(v: Tensor, r_max: float, eps: float = 1e-8) -> Tensor:
    """Smooth radial squash to prevent boundary blow-up in logmap0 output.

    Maps any norm to [0, r_max) via tanh saturation.
    Preserves direction, only affects magnitude.
    """
    r = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    r_new = r_max * torch.tanh(r / r_max)
    return v * (r_new / r)


def latent_to_soft_prompt(
    latent: Tensor,
    W: Tensor,
    curvature: float,
    embed_dim: int,
    num_tokens: int,
    target_rms: float,
    *,
    use_logmap: bool = True,
) -> Tensor:
    """Convert a latent vector to soft prompt embeddings.

    Pipeline:
    1. (optional) logmap0: Poincare ball -> tangent space at origin
    2. NaN guard
    3. Radial tanh squash (prevents blow-up, smooth saturation)
    4. Project via row-orthonormal W
    5. Reshape to (num_tokens, embed_dim)
    6. Scale to match real token embedding RMS

    Args:
        latent: Latent vector, shape (..., d_latent).
        W: Row-orthonormal projection, shape (d_latent, d_out).
        curvature: Poincare ball curvature (used only when use_logmap=True).
        embed_dim: Model embedding dimension.
        num_tokens: Number of soft prompt tokens.
        target_rms: Per-element RMS of real token embeddings.
        use_logmap: If True, apply logmap0 first (for hyperbolic latents).
            Set to False for Euclidean latents already in tangent space.

    Returns:
        Soft prompt tensor of shape (1, num_tokens, embed_dim).
    """
    d_latent = W.shape[0]
    r_max = 2.0 * math.sqrt(d_latent) * target_rms

    lat = latent.squeeze().float()

    # 1. Map to tangent space (skip for Euclidean latents)
    if use_logmap:
        tangent = hyp.logmap0(lat, curvature)
    else:
        tangent = lat

    # 2. NaN guard
    tangent = torch.nan_to_num(tangent, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Radial tanh squash
    tangent = radial_tanh_squash(tangent, r_max)

    # 4. Project (preserves inner products)
    W_dev = W
    if W_dev.device != tangent.device or W_dev.dtype != tangent.dtype:
        W_dev = W_dev.to(device=tangent.device, dtype=tangent.dtype)
    flat = tangent @ W_dev  # (d_latent,) @ (d_latent, d_out) -> (d_out,)

    # 5. Reshape to token sequence
    soft_prompt = flat.view(num_tokens, embed_dim)

    # 6. Scale: match per-element RMS to real token embeddings
    current_rms = soft_prompt.square().mean().sqrt().clamp_min(1e-8)
    soft_prompt = soft_prompt * (target_rms / current_rms)

    return soft_prompt.unsqueeze(0)  # (1, num_tokens, embed_dim)
