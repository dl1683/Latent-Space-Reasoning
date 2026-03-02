"""Decode pathways for latent-conditioned generation."""

from latent_reasoning.decode.projection import (
    latent_to_soft_prompt,
    make_row_orthonormal_W,
    radial_tanh_squash,
)
from latent_reasoning.decode.steering import (
    DualSteeringProcessor,
    compute_steering_direction,
    make_steer_projection,
)

__all__ = [
    "DualSteeringProcessor",
    "compute_steering_direction",
    "latent_to_soft_prompt",
    "make_row_orthonormal_W",
    "make_steer_projection",
    "radial_tanh_squash",
]
