"""
Autopoietic Judge module for self-updating evaluation in Latent Space Reasoning.

This module provides a self-improving judge system that:
1. Maintains an experience buffer for online learning
2. Uses homeostatic temperature control for diversity
3. Grounds internal scores against external evaluation
4. Adapts to improve correlation with external quality signals

The autopoietic approach addresses the known scorer weakness (low correlation
with external quality) by treating the judge as a living system that maintains
itself through interaction with its environment (external evaluators).

Key Components:

**ExperienceBuffer (experience_buffer.py)**:
- Ring buffer storing (latent, internal_score, external_score) tuples
- Supports sampling for training updates
- Tracks statistics for analysis

**HomeostasisController (homeostasis.py)**:
- Adaptive temperature control based on diversity
- Formula: T_{t+1} = T_t * exp(k * (D* - D_t))
- Maintains target diversity through mutation strength

**ExternalEvaluator (external_evaluator.py)**:
- Wraps external judge (e.g., Gemini) for grounding
- Handles API calls and error recovery
- Provides quality signals for scorer updates

**AutopoieticJudge (autopoietic_judge.py)**:
- Self-updating judge with two-time-scale learning
- Fast: EMA update of internal scorer
- Slow: Grounding against external evaluator

**AutopoieticPanel (autopoietic_panel.py)**:
- Panel integration for EvolutionLoop
- Coordinates all autopoietic components

Quick Start:
    >>> from latent_reasoning.core.autopoietic import create_autopoietic_panel
    >>> from latent_reasoning.config import Config
    >>>
    >>> config = Config()
    >>> config.autopoietic.enabled = True
    >>> config.autopoietic.external_model = "gemini-2.5-flash"
    >>>
    >>> panel = create_autopoietic_panel(config.autopoietic, latent_dim=1024)
    >>>
    >>> # In evolution loop:
    >>> score = panel.evaluate(latent, context)
    >>> panel.step_generation(chains, generation)

References:
- Autopoiesis concept from Maturana & Varela (1980)
- Two-time-scale learning from reinforcement learning
- Homeostatic regulation in biological systems
"""

from latent_reasoning.core.autopoietic.experience_buffer import (
    ExperienceBuffer,
    ExperienceEntry,
)
from latent_reasoning.core.autopoietic.homeostasis import (
    HomeostasisController,
)
from latent_reasoning.core.autopoietic.external_evaluator import (
    ExternalEvaluator,
)
from latent_reasoning.core.autopoietic.autopoietic_judge import (
    AutopoieticJudge,
)
from latent_reasoning.core.autopoietic.autopoietic_panel import (
    AutopoieticPanel,
    create_autopoietic_panel,
)

__all__ = [
    # Experience
    "ExperienceBuffer",
    "ExperienceEntry",
    # Homeostasis
    "HomeostasisController",
    # External
    "ExternalEvaluator",
    # Judge
    "AutopoieticJudge",
    # Panel
    "AutopoieticPanel",
    "create_autopoietic_panel",
]
