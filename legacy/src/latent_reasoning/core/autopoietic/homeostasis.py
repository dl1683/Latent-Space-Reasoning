"""
Homeostatic temperature controller for autopoietic evolution.

The homeostasis controller automatically adjusts mutation temperature based
on population diversity, maintaining a target diversity level to prevent
both premature convergence and excessive exploration.

The control law is inspired by biological homeostasis:
T_{t+1} = T_t * exp(k * (D* - D_t))

Where:
- T_t is current temperature
- D* is target diversity
- D_t is measured diversity
- k is control gain

When diversity is below target, temperature increases to encourage exploration.
When diversity is above target, temperature decreases to focus search.

Reference: Homeostatic regulation in biological systems (Cannon, 1932)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from latent_reasoning.core.chain import ChainState


@dataclass
class HomeostasisState:
    """
    State of the homeostasis controller.

    Tracks current values and history for monitoring and analysis.
    """
    temperature: float
    diversity: float
    generation: int = 0
    history: list[dict] = field(default_factory=list)


class HomeostasisController:
    """
    Adaptive temperature controller based on diversity homeostasis.

    Maintains population diversity near a target level by automatically
    adjusting mutation temperature. This prevents both premature convergence
    (diversity too low) and wasteful random exploration (diversity too high).

    Control Law:
        T_{t+1} = clamp(T_t * exp(k * (D* - D_t)), T_min, T_max)

    Where:
        - T_t: Current temperature
        - D*: Target diversity (config.target_diversity)
        - D_t: Measured diversity
        - k: Control gain (config.homeostasis_k)
        - T_min, T_max: Temperature bounds

    The exponential form ensures:
        - Smooth, multiplicative updates
        - Bounded temperature within [T_min, T_max]
        - Stable control without oscillation

    Args:
        target_diversity: Target diversity level D*
        control_gain: Control gain k (higher = more aggressive adjustment)
        min_temperature: Minimum allowed temperature
        max_temperature: Maximum allowed temperature
        initial_temperature: Starting temperature

    Usage:
        >>> controller = HomeostasisController(target_diversity=0.4)
        >>> diversity = compute_diversity(population)
        >>> new_temp = controller.update(diversity)
        >>> mutation_strategy.temperature = new_temp
    """

    def __init__(
        self,
        target_diversity: float = 0.4,
        control_gain: float = 0.1,
        min_temperature: float = 0.1,
        max_temperature: float = 2.0,
        initial_temperature: float = 0.5,
    ):
        self.target_diversity = target_diversity
        self.control_gain = control_gain
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

        # State
        self._temperature = initial_temperature
        self._diversity = 0.0
        self._generation = 0
        self._history: list[dict] = []

    @property
    def temperature(self) -> float:
        """Current temperature value."""
        return self._temperature

    @property
    def diversity(self) -> float:
        """Last measured diversity."""
        return self._diversity

    def update(self, diversity: float, generation: int | None = None) -> float:
        """
        Update temperature based on measured diversity.

        Applies the homeostatic control law to adjust temperature
        toward the target diversity level.

        Args:
            diversity: Current population diversity (0 to 1)
            generation: Current generation number (for logging)

        Returns:
            New temperature value
        """
        if generation is not None:
            self._generation = generation

        self._diversity = diversity

        # Compute diversity error
        error = self.target_diversity - diversity

        # Apply exponential control law
        adjustment = math.exp(self.control_gain * error)
        new_temp = self._temperature * adjustment

        # Clamp to bounds
        new_temp = max(self.min_temperature, min(self.max_temperature, new_temp))

        # Record history
        self._history.append({
            "generation": self._generation,
            "diversity": diversity,
            "target": self.target_diversity,
            "error": error,
            "adjustment": adjustment,
            "temperature_before": self._temperature,
            "temperature_after": new_temp,
        })

        self._temperature = new_temp
        return new_temp

    def compute_diversity(
        self,
        latents: list[Tensor] | None = None,
        chains: list["ChainState"] | None = None,
    ) -> float:
        """
        Compute diversity metric from population.

        Uses mean pairwise cosine distance as diversity measure.
        Values range from 0 (identical) to 1 (orthogonal).

        Args:
            latents: List of latent vectors
            chains: Alternative: list of ChainState objects

        Returns:
            Diversity score (0 to 1)
        """
        if chains is not None:
            latents = [c.latent for c in chains]

        if latents is None or len(latents) < 2:
            return 0.0

        # Stack and normalize
        stacked = torch.stack([lat.flatten().float() for lat in latents])
        norms = stacked.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = stacked / norms

        # Pairwise cosine similarities
        similarities = torch.mm(normalized, normalized.t())

        # Extract upper triangle (excluding diagonal)
        n = len(latents)
        mask = torch.triu(torch.ones(n, n, device=similarities.device), diagonal=1)
        upper_sims = similarities[mask.bool()]

        if len(upper_sims) == 0:
            return 0.0

        # Diversity = 1 - mean similarity
        mean_similarity = upper_sims.mean().item()
        return 1.0 - mean_similarity

    def update_from_population(
        self,
        latents: list[Tensor] | None = None,
        chains: list["ChainState"] | None = None,
        generation: int | None = None,
    ) -> float:
        """
        Convenience method to compute diversity and update temperature.

        Args:
            latents: List of latent vectors
            chains: Alternative: list of ChainState objects
            generation: Current generation number

        Returns:
            New temperature value
        """
        diversity = self.compute_diversity(latents, chains)
        return self.update(diversity, generation)

    def get_state(self) -> HomeostasisState:
        """Get current controller state."""
        return HomeostasisState(
            temperature=self._temperature,
            diversity=self._diversity,
            generation=self._generation,
            history=self._history.copy(),
        )

    def get_statistics(self) -> dict:
        """Get controller statistics for monitoring."""
        stats = {
            "temperature": self._temperature,
            "diversity": self._diversity,
            "target_diversity": self.target_diversity,
            "diversity_error": self.target_diversity - self._diversity,
            "generation": self._generation,
            "updates": len(self._history),
        }

        if self._history:
            errors = [h["error"] for h in self._history]
            temps = [h["temperature_after"] for h in self._history]
            stats["mean_error"] = sum(errors) / len(errors)
            stats["mean_temperature"] = sum(temps) / len(temps)
            stats["temp_range"] = (min(temps), max(temps))

        return stats

    def reset(self, initial_temperature: float | None = None) -> None:
        """
        Reset controller state.

        Args:
            initial_temperature: New starting temperature (uses current if None)
        """
        if initial_temperature is not None:
            self._temperature = initial_temperature
        self._diversity = 0.0
        self._generation = 0
        self._history.clear()

    def __repr__(self) -> str:
        return (
            f"HomeostasisController("
            f"temp={self._temperature:.3f}, "
            f"diversity={self._diversity:.3f}, "
            f"target={self.target_diversity:.3f})"
        )
