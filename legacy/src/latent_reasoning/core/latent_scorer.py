"""Latent-space scorer model for quality prediction.

This scorer works directly on the encoder's latent space, making it
usable during evolution without needing to decode latents to text.

This module contains only the model definition needed for inference.
Training code has been moved to a separate repository.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LatentScorer(nn.Module):
    """
    Neural network that scores latent vectors directly.

    Takes a latent from the encoder's space and predicts quality 0-1.
    Optionally conditions on a query latent for context.
    """

    def __init__(
        self,
        latent_dim: int = 1024,  # Match encoder's latent dim
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_query_conditioning: bool = True,
    ):
        """
        Initialize the latent scorer.

        Args:
            latent_dim: Dimension of input latents (must match encoder)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate
            use_query_conditioning: If True, concatenate query latent with plan latent
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.use_query_conditioning = use_query_conditioning

        # Input dim is 2x latent_dim if using query conditioning
        input_dim = latent_dim * 2 if use_query_conditioning else latent_dim

        # Build MLP layers
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else hidden_dim // 2
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = next_dim

        # Final prediction head
        layers.extend([
            nn.Linear(current_dim, 1),
            nn.Sigmoid(),
        ])

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        plan_latent: Tensor,
        query_latent: Tensor | None = None,
    ) -> Tensor:
        """
        Score a plan latent.

        Args:
            plan_latent: Latent vector of the plan (batch, latent_dim) or (latent_dim,)
            query_latent: Optional query latent for conditioning

        Returns:
            Quality score 0-1 (batch,) or scalar
        """
        # Handle single vectors
        if plan_latent.dim() == 1:
            plan_latent = plan_latent.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Concatenate with query if using conditioning
        if self.use_query_conditioning:
            if query_latent is None:
                # Use zeros if no query provided
                query_latent = torch.zeros_like(plan_latent)
            elif query_latent.dim() == 1:
                query_latent = query_latent.unsqueeze(0)

            # Expand query to match batch size if needed
            if query_latent.size(0) == 1 and plan_latent.size(0) > 1:
                query_latent = query_latent.expand(plan_latent.size(0), -1)

            x = torch.cat([plan_latent, query_latent], dim=-1)
        else:
            x = plan_latent

        # Forward through network
        score = self.network(x).squeeze(-1)

        if squeeze_output:
            score = score.squeeze(0)

        return score

    def score_batch(
        self,
        latents: list[Tensor],
        query_latent: Tensor | None = None,
    ) -> list[float]:
        """
        Score a batch of latents.

        Args:
            latents: List of latent tensors
            query_latent: Optional query latent

        Returns:
            List of scores
        """
        # Stack latents
        stacked = torch.stack([l.squeeze() for l in latents])

        # Score
        with torch.no_grad():
            scores = self.forward(stacked, query_latent)

        return scores.tolist()
