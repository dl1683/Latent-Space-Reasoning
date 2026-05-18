from __future__ import annotations

"""Tests for the HeuristicScorer and DiversityBonusScorer."""

import torch
import pytest

from latent_reasoning.core.heuristic_scorer import (
    HeuristicScorer,
    DiversityBonusScorer,
    TextQualityScores,
)


class TestHeuristicScorer:
    """Test the heuristic-based text quality scoring."""

    @pytest.fixture
    def scorer(self):
        return HeuristicScorer()

    def test_score_returns_text_quality_scores(self, scorer):
        text = "Step 1: Create a plan\nStep 2: Implement\nStep 3: Test\nStep 4: Deploy\nStep 5: Monitor"
        result = scorer.score(text)
        assert isinstance(result, TextQualityScores)
        assert 0 <= result.structure_score <= 1.0
        assert 0 <= result.depth_score <= 1.0
        assert 0 <= result.action_score <= 1.0
        assert 0 <= result.coherence_score <= 1.0
        assert 0 <= result.overall_score <= 1.0

    def test_empty_text(self, scorer):
        result = scorer.score("")
        assert result.overall_score == 0.0

    def test_short_text_low_depth(self, scorer):
        result = scorer.score("hello")
        assert result.depth_score < 0.5

    def test_structured_text_gets_structure_bonus(self, scorer):
        text = """## Implementation Plan\n1. First step\n2. Second step\n3. Third step\n4. Fourth step\n5. Fifth step\n- Bullet one\n- Bullet two\n- Bullet three\n"""
        result = scorer.score(text)
        assert result.structure_score > 0.5

    def test_action_verbs_boost_score(self, scorer):
        text = "We will create, implement, design, develop, build, deploy, configure, test, validate, and analyze the system."
        result = scorer.score(text)
        assert result.action_score > 0.5

    def test_coherence_markers_boost_score(self, scorer):
        text = "first we do this then we do that next we proceed after that we finish finally we review. therefore we conclude."
        result = scorer.score(text)
        assert result.coherence_score > 0.5

    def test_long_text_gets_depth_bonus(self, scorer):
        words = "word " * 300
        result = scorer.score(words)
        assert result.depth_score >= 0.7

    def test_very_long_text_gets_slight_penalty(self, scorer):
        words = "word " * 2000
        result = scorer.score(words)
        assert result.depth_score < 1.0

    def test_custom_weights(self):
        custom = HeuristicScorer(
            structure_weight=0.5,
            depth_weight=0.3,
            action_weight=0.1,
            coherence_weight=0.1,
        )
        assert custom.weights["structure"] == 0.5
        assert custom.weights["depth"] == 0.3
        assert custom.weights["action"] == 0.1
        assert custom.weights["coherence"] == 0.1

    def test_numbered_steps_detection(self, scorer):
        text = "1. Do this\n2. Do that\n3. Do another thing"
        structure = scorer._score_structure(text)
        assert structure > 0

    def test_header_detection(self, scorer):
        text = "# Title\n## Section\n### Subsection\n**Bold**"
        structure = scorer._score_structure(text)
        assert structure > 0

    def test_bullet_detection(self, scorer):
        text = "- item 1\n- item 2\n- item 3"
        structure = scorer._score_structure(text)
        assert structure > 0

    def test_single_word_no_action(self, scorer):
        result = scorer.score("hello")
        assert result.action_score == 0.0


class TestDiversityBonusScorer:
    """Test the diversity bonus scoring."""

    @pytest.fixture
    def scorer(self):
        return DiversityBonusScorer()

    def test_no_seed_returns_neutral(self, scorer):
        latent = torch.randn(768)
        score = scorer.score_diversity(latent)
        assert score == 0.5

    def test_identical_seed_returns_low_score(self, scorer):
        seed = torch.randn(768)
        scorer.set_seed(seed)
        score = scorer.score_diversity(seed)
        assert score < 0.5

    def test_different_seed_returns_higher_score(self, scorer):
        seed = torch.randn(768)
        different = torch.randn(768)
        scorer.set_seed(seed)
        score = scorer.score_diversity(different)
        assert score > 0

    def test_combine_scores(self, scorer):
        seed = torch.randn(768)
        latent = torch.randn(768)
        scorer.set_seed(seed)
        combined = scorer.combine_scores(0.7, latent)
        assert 0 <= combined <= 1.0

    def test_set_seed_normalizes(self, scorer):
        seed = torch.randn(768)
        scorer.set_seed(seed)
        norm = torch.norm(scorer._seed.float())
        assert abs(norm - 1.0) < 0.01

    def test_diversity_weight_effect(self, scorer):
        high = DiversityBonusScorer(diversity_weight=0.9)
        low = DiversityBonusScorer(diversity_weight=0.1)
        seed = torch.randn(768)
        latent = torch.randn(768)
        scorer.set_seed(seed)
        high.set_seed(seed)
        low.set_seed(seed)
        ch = high.combine_scores(1.0, latent)
        cl = low.combine_scores(1.0, latent)
        assert ch != cl
