"""Tests for Karcher mean crossover auto-upgrade in hyperbolic mode."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.config import Config, EvolutionConfig, GeometryConfig, CrossoverConfig
from latent_reasoning.evolution.crossover import HyperbolicCrossover, get_crossover_strategy
from latent_reasoning.evolution.loop import EvolutionLoop


def _make_mock_panel():
    """Create a minimal mock JudgePanel for EvolutionLoop init."""
    panel = MagicMock()
    panel.evaluate.return_value = (0.5, [0.5], [None])
    return panel


class TestKarcherDefault:
    def test_hyperbolic_mode_auto_upgrades_crossover(self):
        """When geometry is hyperbolic, crossover should auto-upgrade to HyperbolicCrossover."""
        config = EvolutionConfig()
        geometry = GeometryConfig(space="hyperbolic", curvature=0.5)
        loop = EvolutionLoop(
            judge_panel=_make_mock_panel(),
            config=config,
            geometry_config=geometry,
        )
        assert isinstance(loop.crossover, HyperbolicCrossover)

    def test_euclidean_mode_unchanged(self):
        """Euclidean mode should use the config-specified crossover (not Karcher)."""
        config = EvolutionConfig()
        config.crossover = CrossoverConfig(strategy="weighted")
        geometry = GeometryConfig(space="euclidean")
        loop = EvolutionLoop(
            judge_panel=_make_mock_panel(),
            config=config,
            geometry_config=geometry,
        )
        assert not isinstance(loop.crossover, HyperbolicCrossover)

    def test_explicit_override_still_works(self):
        """Explicit crossover param should override auto-upgrade."""
        from latent_reasoning.evolution.crossover import MeanCrossover
        config = EvolutionConfig()
        geometry = GeometryConfig(space="hyperbolic")
        explicit = MeanCrossover()
        loop = EvolutionLoop(
            judge_panel=_make_mock_panel(),
            config=config,
            geometry_config=geometry,
            crossover=explicit,
        )
        assert isinstance(loop.crossover, MeanCrossover)

    def test_hyperbolic_crossover_uses_correct_curvature(self):
        """Auto-upgraded crossover should use the geometry config's curvature."""
        config = EvolutionConfig()
        geometry = GeometryConfig(space="hyperbolic", curvature=2.0)
        loop = EvolutionLoop(
            judge_panel=_make_mock_panel(),
            config=config,
            geometry_config=geometry,
        )
        assert isinstance(loop.crossover, HyperbolicCrossover)
        assert loop.crossover.curvature == 2.0
