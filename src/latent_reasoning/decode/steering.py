"""Steering decode -- logit-level and intermediate-layer steering.

Provides two complementary steering mechanisms:

1. DualSteeringProcessor: Logit-level Newton steering via information geometry
   (arXiv:2602.15293). Modifies the final token distribution.

2. IntermediateLayerSteering: Residual stream injection at specified transformer
   layers. Based on Turner et al. 2023 (Activation Engineering / ActAdd).
   More powerful than logit-level because it changes internal computation.

Also provides:
- make_steer_projection: Fixed orthonormal projection from latent to hidden space.
- compute_steering_direction: Full pipeline from latent to vocabulary-sized direction.
- latent_to_layer_vectors: Compute per-layer steering vectors from a single latent.
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


# =====================================================================
# Intermediate layer steering (ActAdd-style residual stream injection)
# =====================================================================

class IntermediateLayerSteering:
    """Inject steering vectors into the residual stream at specific layers.

    Uses PyTorch forward hooks to add a fixed vector to each specified
    layer's output during generation. This bypasses attention and directly
    modifies the hidden states, providing stronger conditioning than
    soft prompts (which only bias attention output in a rank-n_s subspace).

    Usage::

        steering = IntermediateLayerSteering(model, {22: vec22, 25: vec25, 28: vec28})
        with steering:
            output = model.generate(...)

    Args:
        model: HuggingFace model with ``model.model.layers`` attribute.
        layer_vectors: Dict mapping layer index to steering vector (d_hidden,).
        scale: Global scaling factor for all steering vectors.
    """

    def __init__(
        self,
        model,
        layer_vectors: dict,
        scale: float = 1.0,
    ):
        self.model = model
        self.layer_vectors = layer_vectors
        self.scale = scale
        self._handles = []

    def _get_layers(self):
        """Get the list of transformer layers from the model."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        raise AttributeError(
            "Model does not have model.model.layers. "
            "IntermediateLayerSteering requires a standard HuggingFace transformer."
        )

    def attach(self):
        """Register forward hooks on specified layers."""
        layers = self._get_layers()

        for layer_idx, vec in self.layer_vectors.items():
            if layer_idx >= len(layers):
                continue

            layer = layers[layer_idx]

            def _make_hook(v, s):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        steering = v.to(device=h.device, dtype=h.dtype) * s
                        h = h + steering.unsqueeze(0).unsqueeze(0)
                        return (h,) + output[1:]
                    else:
                        steering = v.to(device=output.device, dtype=output.dtype) * s
                        return output + steering.unsqueeze(0).unsqueeze(0)
                return hook_fn

            handle = layer.register_forward_hook(_make_hook(vec, self.scale))
            self._handles.append(handle)

    def detach(self):
        """Remove all forward hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *args):
        self.detach()


def latent_to_layer_vectors(
    latent: Tensor,
    layer_projections: dict,
    curvature: float,
    target_rms: float,
    *,
    use_logmap: bool = True,
) -> dict:
    """Convert a latent vector to per-layer steering vectors.

    Each layer gets its own projection of the same latent, with different
    random seeds producing orthogonal steering directions across layers.

    Args:
        latent: Latent vector, shape (..., d_latent).
        layer_projections: Dict mapping layer_idx -> W projection (d_latent, d_hidden).
        curvature: Poincare ball curvature (used only when use_logmap=True).
        target_rms: Per-element RMS to match model's hidden state scale.
        use_logmap: If True, apply logmap0 first (for hyperbolic latents).

    Returns:
        Dict mapping layer_idx -> steering vector (d_hidden,).
    """
    from latent_reasoning.utils import hyperbolic as hyp
    from latent_reasoning.decode.projection import radial_tanh_squash
    import math

    lat = latent.squeeze().float()

    if use_logmap:
        tangent = hyp.logmap0(lat, curvature)
    else:
        tangent = lat

    tangent = torch.nan_to_num(tangent, nan=0.0, posinf=0.0, neginf=0.0)

    d_latent = tangent.shape[-1]
    r_max = 2.0 * math.sqrt(d_latent) * target_rms
    tangent = radial_tanh_squash(tangent, r_max)

    vectors = {}
    for layer_idx, W in layer_projections.items():
        W_dev = W.to(device=tangent.device, dtype=tangent.dtype)
        vec = tangent @ W_dev
        current_rms = vec.square().mean().sqrt().clamp_min(1e-8)
        vec = vec * (target_rms / current_rms)
        vectors[layer_idx] = vec

    return vectors
