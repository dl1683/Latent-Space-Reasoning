"""
Grammar rules with contractive transforms for fractal latent grammars.

Each grammar rule defines a contractive mapping in latent space:
T_r(z) = P_r * activation(W_r @ z + b_r)

The contraction property (||W_r||_2 < 1) ensures that repeated application
converges to a fixed-point attractor, enabling stable and predictable
latent generation.

Key Properties:
- Contraction: Spectral norm < 1 ensures convergence
- Nonlinearity: Activation function adds expressiveness
- Projection: Optional projection matrix for dimensionality
- Attractor: Fixed point computed via iteration

Reference: Iterated Function Systems (IFS) - Barnsley (1988)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from latent_reasoning.config import GrammarConfig


class GrammarRule(nn.Module):
    """
    A single grammar rule implementing a contractive transform.

    The rule computes: T(z) = P * activation(W @ z + b)

    Where:
    - W: Weight matrix with enforced spectral norm < contraction_factor
    - b: Bias vector
    - P: Optional projection matrix
    - activation: Nonlinear activation function

    The contraction property ensures that repeated application converges
    to a unique fixed-point attractor, regardless of starting point.

    Args:
        latent_dim: Dimension of latent vectors
        hidden_dim: Hidden dimension (if different from latent_dim)
        contraction_factor: Maximum spectral norm for W (< 1)
        activation: Activation function ("tanh", "gelu", "silu")
        use_projection: Whether to use projection matrix P

    Usage:
        >>> rule = GrammarRule(latent_dim=1024, contraction_factor=0.9)
        >>> z_new = rule(z)  # Apply transform
        >>> attractor = rule.compute_attractor()  # Get fixed point
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int | None = None,
        contraction_factor: float = 0.9,
        activation: Literal["tanh", "gelu", "silu"] = "tanh",
        use_projection: bool = False,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim or latent_dim
        self.contraction_factor = contraction_factor
        self.use_projection = use_projection

        # Main weight matrix (will be normalized)
        self.W = nn.Parameter(torch.randn(self.hidden_dim, latent_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(self.hidden_dim))

        # Optional projection back to latent_dim
        if use_projection and self.hidden_dim != latent_dim:
            self.P = nn.Parameter(torch.randn(latent_dim, self.hidden_dim) * 0.1)
        else:
            self.P = None

        # Activation - store the type for cloning
        self.activation_type = activation
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "silu":
            self.activation = F.silu
        else:
            self.activation_type = "tanh"
            self.activation = torch.tanh

        # Cached attractor
        self._attractor: Tensor | None = None
        self._attractor_valid = False

        # Register buffer for deterministic spectral norm computation
        # Using a consistent u vector makes the spectral norm stable across forward passes
        self.register_buffer("_spectral_u", F.normalize(torch.randn(1, self.hidden_dim), dim=1))

        # If projection is used, we need to normalize it too for contraction guarantee
        if self.P is not None:
            self.register_buffer("_spectral_u_P", F.normalize(torch.randn(1, latent_dim), dim=1))
        else:
            self._spectral_u_P = None

    def forward(self, z: Tensor) -> Tensor:
        """
        Apply the contractive transform.

        Args:
            z: Input latent vector (batch_size, latent_dim) or (latent_dim,)

        Returns:
            Transformed latent vector
        """
        # Enforce contraction via spectral normalization
        W_normalized = self._normalize_weight()

        # Linear transform
        squeeze = z.dim() == 1
        if squeeze:
            z = z.unsqueeze(0)

        h = z @ W_normalized.t() + self.b

        # Activation
        h = self.activation(h)

        # Optional projection (also normalized for contraction)
        if self.P is not None:
            P_normalized = self._normalize_projection()
            h = h @ P_normalized.t()

        if squeeze:
            h = h.squeeze(0)

        return h

    def _normalize_weight(self) -> Tensor:
        """Normalize weight matrix to enforce contraction."""
        # Compute spectral norm using cached u vector for stability
        with torch.no_grad():
            # Power iteration for spectral norm (fast approximation)
            # Use cached u vector instead of random for deterministic behavior
            u = self._spectral_u
            for _ in range(3):
                v = F.normalize(u @ self.W, dim=1)
                u = F.normalize(v @ self.W.t(), dim=1)
            # Update cached u for next iteration (in-place to avoid graph issues)
            self._spectral_u.copy_(u)
            sigma = (u @ self.W @ v.t()).item()

        # Scale down if exceeds contraction factor
        if sigma > self.contraction_factor:
            scale = self.contraction_factor / (sigma + 1e-8)
        else:
            scale = 1.0

        return self.W * scale

    def _normalize_projection(self) -> Tensor:
        """Normalize projection matrix to enforce contraction guarantee.

        For the overall transform T(z) = P * f(W*z + b) to be contractive,
        we need ||P||_2 * ||W||_2 < 1. Since W is already normalized to
        contraction_factor, we normalize P to have ||P||_2 <= 1.
        """
        if self.P is None:
            raise ValueError("No projection matrix to normalize")

        with torch.no_grad():
            # Power iteration for spectral norm of P
            u = self._spectral_u_P
            for _ in range(3):
                v = F.normalize(u @ self.P, dim=1)
                u = F.normalize(v @ self.P.t(), dim=1)
            self._spectral_u_P.copy_(u)
            sigma = (u @ self.P @ v.t()).item()

        # Normalize P to have spectral norm <= 1
        if sigma > 1.0:
            scale = 1.0 / (sigma + 1e-8)
        else:
            scale = 1.0

        return self.P * scale

    def compute_attractor(
        self,
        initial: Tensor | None = None,
        iterations: int = 20,
        tol: float = 1e-6,
    ) -> Tensor:
        """
        Compute the fixed-point attractor via iteration.

        Due to contraction, repeated application converges to a unique
        fixed point regardless of starting point.

        Args:
            initial: Starting point (random if None)
            iterations: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Attractor latent vector
        """
        if self._attractor is not None and self._attractor_valid:
            return self._attractor

        device = self.W.device
        if initial is None:
            z = torch.randn(self.latent_dim, device=device)
        else:
            z = initial.to(device)

        for _ in range(iterations):
            z_new = self(z)
            if (z_new - z).norm() < tol:
                break
            z = z_new

        self._attractor = z.detach()
        self._attractor_valid = True
        return self._attractor

    def invalidate_attractor(self) -> None:
        """Mark attractor cache as invalid (call after parameter update)."""
        self._attractor_valid = False

    @property
    def spectral_norm(self) -> float:
        """Compute current spectral norm of W."""
        with torch.no_grad():
            u = torch.randn(1, self.W.shape[0], device=self.W.device)
            for _ in range(5):
                v = F.normalize(u @ self.W, dim=1)
                u = F.normalize(v @ self.W.t(), dim=1)
            return (u @ self.W @ v.t()).item()

    def clone(self) -> "GrammarRule":
        """Create a deep copy of this rule."""
        new_rule = GrammarRule(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            contraction_factor=self.contraction_factor,
            activation=self.activation_type,
            use_projection=self.use_projection,
        )
        new_rule.load_state_dict(self.state_dict())
        return new_rule

    def __repr__(self) -> str:
        return (
            f"GrammarRule(latent_dim={self.latent_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"contraction={self.contraction_factor:.2f})"
        )


class RuleBank(nn.Module):
    """
    Collection of grammar rules shared across a grammar.

    The rule bank enables parameter sharing and efficient representation.
    All rules in the bank have the same structure but different parameters.

    Args:
        num_rules: Number of rules in the bank
        latent_dim: Dimension of latent vectors
        hidden_dim: Hidden dimension for rules
        contraction_factor: Maximum spectral norm

    Usage:
        >>> bank = RuleBank(num_rules=8, latent_dim=1024)
        >>> z_new = bank.apply(rule_idx=3, z=z)
        >>> attractor = bank.get_attractor(rule_idx=3)
    """

    def __init__(
        self,
        num_rules: int,
        latent_dim: int,
        hidden_dim: int | None = None,
        contraction_factor: float = 0.9,
        activation: Literal["tanh", "gelu", "silu"] = "tanh",
    ):
        super().__init__()

        self.num_rules = num_rules
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim or latent_dim
        self.contraction_factor = contraction_factor
        self.activation_type = activation

        # Create rules
        # Enable projection when hidden_dim differs from latent_dim to maintain output dimension
        use_projection = self.hidden_dim != latent_dim
        self.rules = nn.ModuleList([
            GrammarRule(
                latent_dim=latent_dim,
                hidden_dim=self.hidden_dim,
                contraction_factor=contraction_factor,
                activation=activation,
                use_projection=use_projection,
            )
            for _ in range(num_rules)
        ])

    def apply(self, rule_idx: int, z: Tensor) -> Tensor:
        """Apply a specific rule."""
        return self.rules[rule_idx](z)

    def apply_batch(self, rule_indices: list[int], z: Tensor) -> list[Tensor]:
        """Apply multiple rules to the same input."""
        return [self.rules[idx](z) for idx in rule_indices]

    def get_attractor(self, rule_idx: int) -> Tensor:
        """Get the attractor for a specific rule."""
        return self.rules[rule_idx].compute_attractor()

    def get_all_attractors(self) -> list[Tensor]:
        """Get attractors for all rules."""
        return [rule.compute_attractor() for rule in self.rules]

    def invalidate_attractors(self) -> None:
        """Invalidate all cached attractors."""
        for rule in self.rules:
            rule.invalidate_attractor()

    @classmethod
    def from_config(
        cls,
        config: "GrammarConfig",
        latent_dim: int,
    ) -> "RuleBank":
        """Create rule bank from configuration."""
        return cls(
            num_rules=config.num_rules,
            latent_dim=latent_dim,
            hidden_dim=config.rule_hidden_dim,
            contraction_factor=config.contraction_factor,
        )

    def clone(self) -> "RuleBank":
        """Create a deep copy of this rule bank."""
        new_bank = RuleBank(
            num_rules=self.num_rules,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            contraction_factor=self.contraction_factor,
            activation=self.activation_type,
        )
        new_bank.load_state_dict(self.state_dict())
        return new_bank

    def __len__(self) -> int:
        return self.num_rules

    def __getitem__(self, idx: int) -> GrammarRule:
        return self.rules[idx]

    def __repr__(self) -> str:
        return f"RuleBank(num_rules={self.num_rules}, latent_dim={self.latent_dim})"
