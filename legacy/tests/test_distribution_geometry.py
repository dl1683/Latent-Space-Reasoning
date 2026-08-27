"""Tests for distribution-geometry diagnostics."""

import torch

from latent_reasoning.decode.distribution_geometry import (
    compare_logit_geometry,
    counterfactual_mass_metrics,
    entropy_from_logits,
    js_divergence,
    kl_divergence,
    probabilities_from_logits,
    topk_overlap,
    weighted_rank_drift,
)


class TestProbabilityGeometry:
    def test_probabilities_are_normalized(self):
        logits = torch.tensor([[2.0, 1.0, -1.0], [0.0, 0.0, 0.0]])
        probs = probabilities_from_logits(logits)
        assert probs.shape == logits.shape
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2))

    def test_entropy_matches_uniform_distribution(self):
        logits = torch.zeros(1, 4)
        entropy = entropy_from_logits(logits)
        assert entropy.item() == torch.log(torch.tensor(4.0)).item()

    def test_kl_matches_manual_full_vocab(self):
        p_logits = torch.tensor([[2.0, 0.0, -1.0]])
        q_logits = torch.tensor([[1.0, 0.5, -0.5]])
        p = torch.softmax(p_logits, dim=-1)
        q = torch.softmax(q_logits, dim=-1)

        expected = (p * (p.log() - q.log())).sum(dim=-1)
        actual = kl_divergence(p, q)
        assert torch.allclose(actual, expected)

    def test_js_is_symmetric(self):
        p = torch.tensor([[0.7, 0.2, 0.1]])
        q = torch.tensor([[0.2, 0.7, 0.1]])
        assert torch.allclose(js_divergence(p, q), js_divergence(q, p))

    def test_topk_with_other_keeps_kl_finite(self):
        p = torch.tensor([[0.6, 0.2, 0.1, 0.1]])
        q = torch.tensor([[0.1, 0.2, 0.6, 0.1]])
        value = kl_divergence(p, q, topk=2)
        assert torch.isfinite(value).all()
        assert value.item() > 0.0


class TestRankDiagnostics:
    def test_topk_overlap_detects_identical_top_tokens(self):
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        assert topk_overlap(logits, logits, k=2).item() == 1.0

    def test_topk_overlap_detects_changed_top_tokens(self):
        a = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        b = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        assert topk_overlap(a, b, k=2).item() == 0.0

    def test_weighted_rank_drift_is_zero_for_identity(self):
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        assert weighted_rank_drift(logits, logits, k=4).item() == 0.0

    def test_compare_logit_geometry_identity_is_quiet(self):
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        metrics = compare_logit_geometry(logits, logits, topk=4)
        summary = metrics.mean_dict()
        assert summary["forward_kl"] == 0.0
        assert summary["reverse_kl"] == 0.0
        assert summary["js"] == 0.0
        assert summary["top1_changed"] == 0.0
        assert summary["topk_overlap"] == 1.0
        assert summary["weighted_rank_drift"] == 0.0

    def test_compare_logit_geometry_detects_top1_change(self):
        a = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        b = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        metrics = compare_logit_geometry(a, b, topk=2)
        assert metrics.top1_changed.item() == 1.0
        assert metrics.forward_kl.item() > 0.0
        assert metrics.weighted_rank_drift.item() > 0.0


class TestCounterfactualMassMetrics:
    def test_pair_mass_is_preserved_when_mass_moves_within_pair(self):
        reference = torch.full((1, 6), -8.0)
        candidate = reference.clone()
        reference[0, 0] = 5.0
        reference[0, 1] = 4.0
        candidate[0, 0] = 4.0
        candidate[0, 1] = 5.0

        metrics = counterfactual_mass_metrics(reference, candidate, [(0, 1)])
        assert abs(metrics.pair_mass_delta.item()) < 1e-3
        assert abs(metrics.neutral_mass_delta.item()) < 1e-3

    def test_pair_mass_detects_leakage_to_neutral_tokens(self):
        reference = torch.full((1, 6), -8.0)
        candidate = reference.clone()
        reference[0, 0] = 5.0
        reference[0, 1] = 4.0
        candidate[0, 5] = 5.0

        metrics = counterfactual_mass_metrics(reference, candidate, [(0, 1)])
        assert metrics.pair_mass_delta.item() < -0.5
        assert metrics.neutral_mass_delta.item() > 0.5
