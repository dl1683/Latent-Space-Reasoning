"""Tests for model-agnostic infrastructure (auto_calibrate, W dims, etc).

These tests verify the model-agnostic infrastructure works correctly
without loading actual models (uses mocks for speed).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.harness import auto_calibrate, check_soft_prompt_compatibility
from latent_reasoning.decode.projection import make_row_orthonormal_W


def _make_mock_encoder(embed_dim: int, vocab_size: int, hidden_size: int = None):
    """Create a mock encoder with realistic embedding table."""
    if hidden_size is None:
        hidden_size = embed_dim

    encoder = MagicMock()

    # Mock embedding weight
    embed_weight = torch.randn(vocab_size, embed_dim) * 0.02
    embed_layer = MagicMock()
    embed_layer.weight = embed_weight

    encoder.model.get_input_embeddings.return_value = embed_layer
    encoder.model.config.hidden_size = hidden_size

    return encoder


class TestAutoCalibrate:
    def test_structure(self):
        """auto_calibrate returns dict with expected keys."""
        encoder = _make_mock_encoder(embed_dim=2560, vocab_size=32000)
        cal = auto_calibrate(encoder)
        assert "embed_dim" in cal
        assert "hidden_dim" in cal
        assert "vocab_size" in cal
        assert "embedding_rms" in cal
        assert "mean_token_norm" in cal

    def test_correct_dimensions(self):
        """Calibration reflects actual model dimensions."""
        encoder = _make_mock_encoder(embed_dim=2560, vocab_size=32000, hidden_size=2560)
        cal = auto_calibrate(encoder)
        assert cal["embed_dim"] == 2560
        assert cal["hidden_dim"] == 2560
        assert cal["vocab_size"] == 32000

    def test_rms_reasonable(self):
        """Embedding RMS should be in a reasonable range."""
        encoder = _make_mock_encoder(embed_dim=768, vocab_size=50000)
        cal = auto_calibrate(encoder)
        # With random init * 0.02, RMS should be around 0.02
        assert 0.005 < cal["embedding_rms"] < 0.1

    def test_different_models_different_calibration(self):
        """Different model sizes produce different calibrations."""
        enc_small = _make_mock_encoder(embed_dim=768, vocab_size=32000)
        enc_large = _make_mock_encoder(embed_dim=2560, vocab_size=152000)
        cal_small = auto_calibrate(enc_small)
        cal_large = auto_calibrate(enc_large)
        assert cal_small["embed_dim"] != cal_large["embed_dim"]
        assert cal_small["vocab_size"] != cal_large["vocab_size"]


class TestWProjectionDims:
    def test_w_matches_embed_dim(self):
        """W projection output should match model embed_dim * num_tokens."""
        for embed_dim in [768, 2560, 4096]:
            num_tokens = 8
            d_latent = 1024
            d_out = num_tokens * embed_dim
            W = make_row_orthonormal_W(d_latent, d_out, seed=1234)
            assert W.shape == (d_latent, d_out)
            # Verify orthonormality
            WWT = W @ W.T
            I = torch.eye(d_latent)
            assert torch.allclose(WWT, I, atol=1e-5)

    def test_small_latent_dim(self):
        """W works with smaller latent dims (e.g. from smaller models)."""
        W = make_row_orthonormal_W(512, 8 * 768, seed=42)
        assert W.shape == (512, 6144)


class TestSoftPromptCompatibility:
    def test_compatible_model(self):
        """Model that supports inputs_embeds should return True."""
        encoder = MagicMock()
        encoder._device = torch.device("cpu")

        embed_layer = MagicMock()
        embed_layer.return_value = torch.randn(1, 1, 256)
        encoder.model.get_input_embeddings.return_value = embed_layer

        # generate succeeds
        encoder.model.generate.return_value = torch.tensor([[1, 2, 3]])

        assert check_soft_prompt_compatibility(encoder) is True

    def test_incompatible_model(self):
        """Model that raises on inputs_embeds should return False."""
        encoder = MagicMock()
        encoder._device = torch.device("cpu")

        embed_layer = MagicMock()
        embed_layer.return_value = torch.randn(1, 1, 256)
        encoder.model.get_input_embeddings.return_value = embed_layer

        # generate fails
        encoder.model.generate.side_effect = TypeError("inputs_embeds not supported")

        assert check_soft_prompt_compatibility(encoder) is False
