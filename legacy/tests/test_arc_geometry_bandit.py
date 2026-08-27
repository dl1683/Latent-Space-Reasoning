"""Tests for ARC geometry-bandit strategy plumbing."""

import pytest

from latent_reasoning.eval.arc_agi2 import ARCEvaluator, _normalize_arc_strategy


def test_geometry_bandit_strategy_is_supported():
    assert _normalize_arc_strategy("geometry_bandit") == "geometry_bandit"


def test_geometry_bandit_explores_profiles_before_scoring(tmp_path):
    evaluator = ARCEvaluator(
        output_dir=str(tmp_path),
        arc_strategy="geometry_bandit",
        lr_retries=3,
    )

    selected = [
        evaluator._select_geometry_bandit_profile(attempt)["name"]
        for attempt in range(len(evaluator._geometry_bandit_profiles))
    ]

    assert selected == [profile["name"] for profile in evaluator._geometry_bandit_profiles]


def test_geometry_bandit_records_parse_partial_and_trace_signal(tmp_path):
    evaluator = ARCEvaluator(
        output_dir=str(tmp_path),
        arc_strategy="geometry_bandit",
    )
    profile = evaluator._geometry_bandit_profiles[1]
    trace = [
        {
            "forward_kl": evaluator.geometry_feedback_target_forward_kl,
            "topk_overlap": 0.8,
            "entropy_delta": 0.05,
        }
    ]

    evaluator._record_geometry_bandit_outcome(
        profile_name=profile["name"],
        parsed=True,
        partial=0.75,
        decode_trace=trace,
    )

    stats = evaluator._geometry_bandit_profile_stats[profile["name"]]
    assert evaluator._geometry_bandit_total_attempts == 1
    assert stats["samples"] == 1
    assert stats["parse_success"] == 1
    assert stats["partial_sum"] == pytest.approx(0.75)
    assert stats["trace_signal_sum"] > 0.0
    assert 0.0 < stats["reward_sum"] <= 1.0
