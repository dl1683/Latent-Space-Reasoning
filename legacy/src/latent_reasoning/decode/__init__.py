"""Decode pathways for latent-conditioned generation."""

from latent_reasoning.decode.distribution_geometry import (
    CounterfactualMassMetrics,
    LogitGeometryMetrics,
    compare_logit_geometry,
    counterfactual_mass_metrics,
    entropy_from_logits,
    entropy_from_probs,
    js_divergence,
    kl_divergence,
    probabilities_from_logits,
    topk_overlap,
    weighted_rank_drift,
)
from latent_reasoning.decode.projection import (
    latent_to_soft_prompt,
    make_row_orthonormal_W,
    radial_tanh_squash,
)
from latent_reasoning.decode.steering import (
    DualSteeringProcessor,
    IntermediateLayerSteering,
    compute_steering_direction,
    latent_to_layer_vectors,
    make_steer_projection,
)

__all__ = [
    "CounterfactualMassMetrics",
    "DualSteeringProcessor",
    "IntermediateLayerSteering",
    "LogitGeometryMetrics",
    "compare_logit_geometry",
    "compute_steering_direction",
    "counterfactual_mass_metrics",
    "entropy_from_logits",
    "entropy_from_probs",
    "js_divergence",
    "kl_divergence",
    "latent_to_layer_vectors",
    "latent_to_soft_prompt",
    "make_row_orthonormal_W",
    "make_steer_projection",
    "probabilities_from_logits",
    "radial_tanh_squash",
    "topk_overlap",
    "weighted_rank_drift",
]
