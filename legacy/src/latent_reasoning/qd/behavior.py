"""
Behavioral descriptor computation for Quality Diversity.

Behavioral descriptors (BDs) characterize HOW a solution achieves its goal,
not just how well. This enables maintaining diverse approaches even when
they have similar fitness scores.

For latent space reasoning, we define behavior via:
1. Latent cluster position (where in latent space)
2. Structural stats (reasoning complexity)
3. Latent trajectory (evolution path taken)

Key Components:
- RFFProjector: Random Fourier Features for dimensionality reduction
- BehaviorComputer: Combines multiple BD components
- BehaviorDescriptor: Container for BD vectors with metadata
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from latent_reasoning.core.chain import ChainState


@dataclass
class BehaviorDescriptor:
    """
    Container for a behavioral descriptor with metadata.

    Attributes:
        vector: The full BD vector (bd_dim,)
        latent_component: Contribution from latent position
        structural_component: Contribution from structural stats
        trajectory_component: Contribution from evolution trajectory
    """
    vector: Tensor
    latent_component: Tensor
    structural_component: Tensor
    trajectory_component: Tensor

    def to(self, device: torch.device | str) -> "BehaviorDescriptor":
        """Move all tensors to specified device."""
        return BehaviorDescriptor(
            vector=self.vector.to(device),
            latent_component=self.latent_component.to(device),
            structural_component=self.structural_component.to(device),
            trajectory_component=self.trajectory_component.to(device),
        )


class RFFProjector:
    """
    Random Fourier Feature projector for dimensionality reduction.

    Uses random Fourier features to project high-dimensional latent vectors
    into a lower-dimensional behavioral descriptor space while preserving
    relative distances (kernel approximation).

    This approach is validated by AutoQD (June 2025) and provides:
    - No training required (instant use)
    - Preserves relative distances approximately
    - Theoretically grounded BD generation

    The projection approximates a Gaussian RBF kernel:
    k(x, y) ≈ φ(x)ᵀφ(y) where φ(x) = [cos(Wx+b), sin(Wx+b)]/√(D/2)

    Args:
        input_dim: Dimension of input vectors (e.g., 1024 for latent)
        output_dim: Dimension of output BD (must be even, e.g., 16)
        gamma: Kernel bandwidth - controls sensitivity to distances
               Higher gamma = more sensitive to small differences
               Lower gamma = smoother, less sensitive
        device: Device for computations
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        gamma: float = 0.1,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ):
        if output_dim % 2 != 0:
            raise ValueError("output_dim must be even for RFF (cos + sin)")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        # Random weights and biases for RFF
        # W ~ N(0, gamma^2) for RBF kernel approximation
        self.W = torch.randn(input_dim, output_dim // 2, device=device) * gamma
        self.b = torch.rand(output_dim // 2, device=device) * 2 * np.pi

    def project(self, x: Tensor) -> Tensor:
        """
        Project input to RFF space.

        Args:
            x: Input tensor (batch_size, input_dim) or (input_dim,)

        Returns:
            RFF projection (batch_size, output_dim) or (output_dim,)
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        # Ensure float and on correct device
        x = x.float().to(self.device)

        # z = x @ W + b
        z = x @ self.W + self.b

        # RFF: [cos(z), sin(z)] / sqrt(output_dim/2)
        scale = np.sqrt(self.output_dim / 2)
        rff = torch.cat([torch.cos(z), torch.sin(z)], dim=-1) / scale

        if squeeze:
            rff = rff.squeeze(0)

        return rff

    def to(self, device: torch.device | str) -> "RFFProjector":
        """Move projector to device."""
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.W = self.W.to(device)
        self.b = self.b.to(device)
        return self

    def __repr__(self) -> str:
        return f"RFFProjector(input_dim={self.input_dim}, output_dim={self.output_dim}, gamma={self.gamma})"


class BehaviorComputer:
    """
    Computes behavioral descriptors from latent vectors and evolution context.

    Combines multiple behavior components:
    1. Latent cluster (RFF projection of latent position)
    2. Structural stats (generation, norm - normalized)
    3. Trajectory stats (delta norms, direction consistency)

    The weights control how much each component contributes to the final BD.
    Default weights: 50% latent, 25% structural, 25% trajectory.

    Args:
        latent_dim: Dimension of input latent vectors
        bd_dim: Total behavioral descriptor dimension
        rff_gamma: Gamma for RFF projector
        latent_weight: Weight for latent component
        structural_weight: Weight for structural component
        trajectory_weight: Weight for trajectory component
        device: Device for computations
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        latent_dim: int,
        bd_dim: int = 16,
        rff_gamma: float = 0.1,
        latent_weight: float = 0.5,
        structural_weight: float = 0.25,
        trajectory_weight: float = 0.25,
        device: torch.device | str = "cpu",
        seed: int | None = 42,
    ):
        self.latent_dim = latent_dim
        self.bd_dim = bd_dim

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Compute component dimensions based on weights
        total_weight = latent_weight + structural_weight + trajectory_weight
        self.latent_bd_dim = int(bd_dim * latent_weight / total_weight)
        self.structural_bd_dim = int(bd_dim * structural_weight / total_weight)
        self.trajectory_bd_dim = bd_dim - self.latent_bd_dim - self.structural_bd_dim

        # Ensure even dimensions for RFF (requires cos + sin pairs)
        if self.latent_bd_dim % 2 != 0:
            self.latent_bd_dim += 1
            self.trajectory_bd_dim -= 1
        if self.trajectory_bd_dim % 2 != 0 and self.trajectory_bd_dim > 0:
            self.trajectory_bd_dim += 1
            self.structural_bd_dim -= 1

        # Clamp to valid ranges
        self.latent_bd_dim = max(2, self.latent_bd_dim)
        self.structural_bd_dim = max(1, self.structural_bd_dim)
        self.trajectory_bd_dim = max(0, self.trajectory_bd_dim)

        # Adjust to match bd_dim exactly
        total = self.latent_bd_dim + self.structural_bd_dim + self.trajectory_bd_dim
        if total != bd_dim:
            self.structural_bd_dim += bd_dim - total

        # RFF projector for latent component
        self.latent_projector = RFFProjector(
            input_dim=latent_dim,
            output_dim=self.latent_bd_dim,
            gamma=rff_gamma,
            device=device,
            seed=seed,
        )

        # RFF projector for trajectory component (if used)
        if self.trajectory_bd_dim >= 2:
            self.trajectory_projector = RFFProjector(
                input_dim=latent_dim,
                output_dim=self.trajectory_bd_dim,
                gamma=rff_gamma,
                device=device,
                seed=seed + 1 if seed else None,
            )
        else:
            self.trajectory_projector = None

    def compute(
        self,
        latent: Tensor,
        generation: int = 0,
        history: list[Tensor] | None = None,
        decoded_length: int | None = None,
    ) -> BehaviorDescriptor:
        """
        Compute behavioral descriptor for a latent vector.

        Args:
            latent: The latent vector (latent_dim,)
            generation: Current generation number
            history: List of previous latent positions (for trajectory)
            decoded_length: Optional token length of decoded text

        Returns:
            BehaviorDescriptor with all components
        """
        latent = latent.to(self.device).float()

        # 1. Latent cluster component (RFF projection)
        latent_component = self.latent_projector.project(latent)

        # 2. Structural stats component
        structural_component = self._compute_structural(
            latent, generation, decoded_length
        )

        # 3. Trajectory component
        trajectory_component = self._compute_trajectory(latent, history)

        # Combine components
        bd_vector = torch.cat([
            latent_component,
            structural_component,
            trajectory_component,
        ], dim=-1)

        return BehaviorDescriptor(
            vector=bd_vector,
            latent_component=latent_component,
            structural_component=structural_component,
            trajectory_component=trajectory_component,
        )

    def _compute_structural(
        self,
        latent: Tensor,
        generation: int,
        decoded_length: int | None,
    ) -> Tensor:
        """Compute structural stats component."""
        stats = []

        # Generation-based stat (normalized to [0, 1])
        gen_stat = min(generation / 50.0, 1.0)
        stats.append(gen_stat)

        # Latent norm stat (normalized)
        norm_stat = min(latent.norm().item() / 100.0, 1.0)
        stats.append(norm_stat)

        # Latent mean (normalized)
        mean_stat = (latent.mean().item() + 1.0) / 2.0  # Assume roughly [-1, 1] range
        mean_stat = max(0.0, min(1.0, mean_stat))
        stats.append(mean_stat)

        # Latent std (normalized)
        std_stat = min(latent.std().item() / 2.0, 1.0)
        stats.append(std_stat)

        # Decoded length stat if available
        if decoded_length is not None:
            len_stat = min(decoded_length / 2000.0, 1.0)
        else:
            len_stat = 0.5  # Default
        stats.append(len_stat)

        # Pad or truncate to structural_bd_dim
        while len(stats) < self.structural_bd_dim:
            stats.append(0.0)
        stats = stats[:self.structural_bd_dim]

        return torch.tensor(stats, device=self.device, dtype=torch.float32)

    def _compute_trajectory(
        self,
        current: Tensor,
        history: list[Tensor] | None,
    ) -> Tensor:
        """Compute trajectory component using RFF on trajectory summary."""
        if self.trajectory_projector is None or self.trajectory_bd_dim == 0:
            return torch.zeros(0, device=self.device)

        if history is None or len(history) < 2:
            # No trajectory info - use zeros
            return torch.zeros(self.trajectory_bd_dim, device=self.device)

        # Compute trajectory summary vector
        history_tensors = [h.to(self.device).float() for h in history[-5:]]  # Last 5 positions

        # Compute deltas
        deltas = []
        for i in range(len(history_tensors) - 1):
            delta = history_tensors[i + 1] - history_tensors[i]
            deltas.append(delta)

        # Average delta direction
        if deltas:
            avg_delta = torch.stack(deltas).mean(dim=0)
        else:
            avg_delta = torch.zeros_like(current)

        # Project trajectory summary
        return self.trajectory_projector.project(avg_delta)

    def compute_from_chain(self, chain: "ChainState") -> BehaviorDescriptor:
        """
        Compute BD directly from a ChainState object.

        Args:
            chain: ChainState with latent and history

        Returns:
            BehaviorDescriptor
        """
        history = chain.history if hasattr(chain, 'history') and chain.history else None
        generation = chain.generation if hasattr(chain, 'generation') else 0
        return self.compute(chain.latent, generation=generation, history=history)

    def compute_batch(
        self,
        latents: list[Tensor],
        generations: list[int] | None = None,
        histories: list[list[Tensor] | None] | None = None,
    ) -> list[BehaviorDescriptor]:
        """
        Compute BDs for a batch of latents.

        Args:
            latents: List of latent vectors
            generations: Optional list of generation numbers
            histories: Optional list of history lists

        Returns:
            List of BehaviorDescriptors
        """
        if not latents:
            return []

        generations = generations or [0] * len(latents)
        histories = histories or [None] * len(latents)

        return [
            self.compute(lat, gen, hist)
            for lat, gen, hist in zip(latents, generations, histories)
        ]

    def to(self, device: torch.device | str) -> "BehaviorComputer":
        """Move computer to device."""
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.latent_projector.to(device)
        if self.trajectory_projector is not None:
            self.trajectory_projector.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"BehaviorComputer(latent_dim={self.latent_dim}, bd_dim={self.bd_dim}, "
            f"components=[latent:{self.latent_bd_dim}, struct:{self.structural_bd_dim}, "
            f"traj:{self.trajectory_bd_dim}])"
        )
