"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).parent.parent
SRC_ROOT = REPO_ROOT / "src"

for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture
def device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def sample_latent(device):
    """Create a sample latent vector."""
    return torch.randn(768, device=device)


@pytest.fixture
def sample_latents(device):
    """Create a batch of sample latent vectors."""
    return torch.randn(10, 768, device=device)


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "What steps should be taken to assess GDPR compliance for a SaaS product?"


@pytest.fixture
def simple_config():
    """Simple configuration for testing."""
    from latent_reasoning.config import (
        Config,
        EncoderConfig,
        EvolutionConfig,
        JudgeConfig,
        ScorerConfig,
    )

    return Config(
        encoder=EncoderConfig(model="bert-base-uncased"),
        judges=JudgeConfig(
            scorers=[ScorerConfig(model="bert-base-uncased")],
            modifiers=[],  # Skip modifiers for simple tests
        ),
        evolution=EvolutionConfig(
            chains=3,
            generations=2,
        ),
    )
