"""
Unit tests for dual steering module (V14).

All tests run on CPU with small vocab sizes - no GPU required.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Ensure experiments/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from latent_reasoning.decode.steering import (
    DualSteeringProcessor,
    IntermediateLayerSteering,
    compute_steering_direction,
    latent_to_layer_vectors,
    make_steer_projection,
)


class TestWoodburyMatchesDenseInverse:
    """Test that the O(V) Woodbury solve matches brute-force O(V^2) solve."""

    def test_woodbury_matches_dense_inverse(self):
        torch.manual_seed(42)
        V = 100
        alpha = 0.01

        # Random logits -> softmax probs
        logits = torch.randn(1, V)
        p = F.softmax(logits, dim=-1)  # (1, V)
        omega = F.normalize(torch.randn(V), dim=0)  # unit direction

        # --- Dense solve (ground truth) ---
        # Sigma = diag(p + alpha) - p^T p
        p_vec = p.squeeze()  # (V,)
        D = torch.diag(p_vec + alpha)  # (V, V)
        Sigma = D - p_vec.unsqueeze(1) @ p_vec.unsqueeze(0)  # (V, V)
        v_dense = torch.linalg.solve(Sigma, omega)  # (V,)

        # --- Woodbury solve (our O(V) implementation) ---
        d = p + alpha  # (1, V)
        d_inv_omega = omega / d  # (1, V)
        d_inv_p = p / d  # (1, V)
        scalar = (p * d_inv_omega).sum(dim=-1, keepdim=True)
        denom = (1.0 - (p * d_inv_p).sum(dim=-1, keepdim=True)).clamp(min=1e-8)
        v_woodbury = (d_inv_omega + d_inv_p * (scalar / denom)).squeeze()

        # They should match closely
        assert torch.allclose(v_dense, v_woodbury, atol=1e-4, rtol=1e-3), (
            f"Max diff: {(v_dense - v_woodbury).abs().max().item():.6f}"
        )


class TestWoodburyNumericalStability:
    """Test Woodbury with near-zero probabilities (one dominant token)."""

    def test_near_zero_probabilities(self):
        torch.manual_seed(123)
        V = 100
        alpha = 0.01

        # Create a very peaked distribution (one dominant token)
        logits = torch.zeros(1, V)
        logits[0, 0] = 20.0  # Very high logit for first token
        p = F.softmax(logits, dim=-1)

        omega = F.normalize(torch.randn(V), dim=0)

        # The Woodbury solve should not produce NaN/Inf
        d = p + alpha
        d_inv_omega = omega / d
        d_inv_p = p / d
        scalar = (p * d_inv_omega).sum(dim=-1, keepdim=True)
        denom = (1.0 - (p * d_inv_p).sum(dim=-1, keepdim=True)).clamp(min=1e-8)
        v = d_inv_omega + d_inv_p * (scalar / denom)

        assert not torch.isnan(v).any(), "Woodbury produced NaN"
        assert not torch.isinf(v).any(), "Woodbury produced Inf"
        assert v.norm().item() > 0, "Woodbury produced zero vector"


class TestProcessorPreservesShape:
    """Test that DualSteeringProcessor output shape matches input."""

    def test_shape_preserved(self):
        torch.manual_seed(7)
        V = 200
        batch = 1

        omega = F.normalize(torch.randn(V), dim=0)
        processor = DualSteeringProcessor(omega_W=omega, eta=0.05)

        input_ids = torch.zeros(batch, 10, dtype=torch.long)
        scores = torch.randn(batch, V)

        result = processor(input_ids, scores)
        assert result.shape == scores.shape, (
            f"Shape mismatch: {result.shape} vs {scores.shape}"
        )


class TestEtaZeroIsIdentity:
    """Test that eta=0 produces no modification to logits."""

    def test_eta_zero_identity(self):
        torch.manual_seed(99)
        V = 150

        omega = F.normalize(torch.randn(V), dim=0)
        processor = DualSteeringProcessor(omega_W=omega, eta=0.0)

        input_ids = torch.zeros(1, 5, dtype=torch.long)
        scores = torch.randn(1, V)

        result = processor(input_ids, scores)
        assert torch.equal(result, scores), "eta=0 should return unmodified logits"


class TestKLCapTriggers:
    """Test that KL cap triggers and downscales eta appropriately."""

    def test_kl_cap_triggers_on_large_eta(self):
        torch.manual_seed(55)
        V = 100

        omega = F.normalize(torch.randn(V), dim=0)
        # Very large eta to guarantee KL exceeds cap
        processor = DualSteeringProcessor(
            omega_W=omega, eta=5.0, kl_cap=0.01
        )

        input_ids = torch.zeros(1, 5, dtype=torch.long)
        scores = torch.randn(1, V)

        result = processor(input_ids, scores)

        # KL cap should have triggered
        assert processor.kl_triggered_count > 0, "KL cap should have triggered"

        # Result should differ from input (steering still applied, just capped)
        assert not torch.equal(result, scores), "Steered logits should differ"

        # Verify the resulting KL is within budget (approximately)
        p_orig = F.softmax(scores.float(), dim=-1)
        p_steered = F.softmax(result.float(), dim=-1)
        kl = (p_orig * (
            torch.log(p_orig.clamp(min=1e-10))
            - torch.log(p_steered.clamp(min=1e-10))
        )).sum().item()

        # Allow some tolerance since capping is approximate
        assert kl < 0.1, f"KL {kl:.4f} too large after cap (expected < 0.1)"


class TestSteerProjectionOrthonormality:
    """Test that W_steer has orthonormal rows."""

    def test_orthonormal_rows(self):
        d_latent = 64
        d_hidden = 128

        W = make_steer_projection(d_latent, d_hidden, seed=5678)

        assert W.shape == (d_latent, d_hidden), (
            f"Wrong shape: {W.shape}, expected ({d_latent}, {d_hidden})"
        )

        # W W^T should be identity
        WWT = W @ W.T
        I = torch.eye(d_latent)
        max_off_diag = (WWT - I).abs().max().item()

        assert max_off_diag < 1e-5, (
            f"W W^T is not identity, max deviation: {max_off_diag:.8f}"
        )


class TestSteerProjectionDifferentSeed:
    """Test that W_steer differs from W_soft when using different seeds."""

    def test_different_seeds_produce_different_W(self):
        d_latent = 64
        d_out = 128

        W_soft = make_steer_projection(d_latent, d_out, seed=1234)
        W_steer = make_steer_projection(d_latent, d_out, seed=5678)

        # They should NOT be equal
        assert not torch.equal(W_soft, W_steer), (
            "W_soft and W_steer should differ with different seeds"
        )

        # Cosine similarity of flattened matrices should be low
        cos_sim = F.cosine_similarity(
            W_soft.flatten().unsqueeze(0),
            W_steer.flatten().unsqueeze(0),
        ).item()

        # Random orthogonal matrices should have near-zero cosine similarity
        assert abs(cos_sim) < 0.2, (
            f"Cosine similarity {cos_sim:.4f} too high for different seeds"
        )


# =====================================================================
# IntermediateLayerSteering tests
# =====================================================================

class _MockLayer(torch.nn.Module):
    def forward(self, x):
        return (x,)


class _MockModel(torch.nn.Module):
    def __init__(self, n_layers=36):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [_MockLayer() for _ in range(n_layers)]
        )


class TestIntermediateLayerSteeringAttachDetach:
    """Test that hooks are properly attached and detached."""

    def test_attach_detach(self):
        model = _MockModel(36)
        vecs = {22: torch.randn(64), 25: torch.randn(64)}
        steering = IntermediateLayerSteering(model, vecs)

        steering.attach()
        assert len(steering._handles) == 2

        x = torch.randn(1, 5, 64)
        out_steered = model.model.layers[22](x)
        diff = (out_steered[0] - x).abs().mean().item()
        assert diff > 0, "Steered layer should modify output"

        steering.detach()
        out_after = model.model.layers[22](x)
        diff_after = (out_after[0] - x).abs().mean().item()
        assert diff_after == 0, "After detach, no modification"


class TestIntermediateLayerSteeringContextManager:
    """Test context manager usage."""

    def test_context_manager(self):
        model = _MockModel(36)
        vecs = {22: torch.randn(64)}
        x = torch.randn(1, 5, 64)

        with IntermediateLayerSteering(model, vecs):
            out = model.model.layers[22](x)
            assert (out[0] - x).abs().mean().item() > 0

        out_after = model.model.layers[22](x)
        assert (out_after[0] - x).abs().mean().item() == 0


class TestIntermediateLayerSteeringScale:
    """Test that scale parameter works correctly."""

    def test_scale_doubles_effect(self):
        model = _MockModel(36)
        vec = torch.randn(64) * 0.01
        x = torch.randn(1, 5, 64)

        with IntermediateLayerSteering(model, {22: vec}, scale=1.0):
            out1 = model.model.layers[22](x)
            diff1 = (out1[0] - x).abs().mean().item()

        with IntermediateLayerSteering(model, {22: vec}, scale=2.0):
            out2 = model.model.layers[22](x)
            diff2 = (out2[0] - x).abs().mean().item()

        assert abs(diff2 / diff1 - 2.0) < 0.01, (
            f"Scale 2x should double effect: {diff2/diff1:.2f}x"
        )


class TestIntermediateLayerSteeringUnsteeredUnchanged:
    """Test that layers not in layer_vectors are unmodified."""

    def test_unsteered_layers_unchanged(self):
        model = _MockModel(36)
        vecs = {22: torch.randn(64)}
        x = torch.randn(1, 5, 64)

        with IntermediateLayerSteering(model, vecs):
            for i in [0, 10, 20, 25, 35]:
                out = model.model.layers[i](x)
                diff = (out[0] - x).abs().mean().item()
                assert diff == 0, f"Layer {i} should be unchanged"


class TestLatentToLayerVectors:
    """Test latent_to_layer_vectors produces correct shapes and RMS."""

    def test_output_shape_and_rms(self):
        # d_latent == d_hidden (realistic: both = model hidden_size)
        d = 128
        target_rms = 0.02

        layer_projs = {
            22: make_steer_projection(d, d, seed=2200),
            25: make_steer_projection(d, d, seed=2500),
        }
        latent = torch.randn(1, d) * 0.5

        vecs = latent_to_layer_vectors(
            latent, layer_projs, 0.5, target_rms, use_logmap=False)

        assert set(vecs.keys()) == {22, 25}
        for idx, v in vecs.items():
            assert v.shape == (d,), f"Layer {idx}: {v.shape}"
            rms = v.square().mean().sqrt().item()
            assert abs(rms - target_rms) < 1e-4, (
                f"Layer {idx} RMS {rms:.5f} != target {target_rms}"
            )
