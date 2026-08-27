"""Tests for phase-source threshold GPU sweep reports."""

import json

import pytest

from experiments.analyze_diffusion_phase_source_threshold_sweep import (
    build_phase_source_threshold_sweep,
    render_markdown,
)


def test_phase_source_threshold_sweep_detects_loose_history_regression(tmp_path):
    strict_scores = tmp_path / "strict_scores.json"
    strict_raw = tmp_path / "strict_raw.jsonl"
    loose_scores = tmp_path / "loose_scores.json"
    loose_raw = tmp_path / "loose_raw.jsonl"
    strict097_scores = tmp_path / "strict097_scores.json"
    strict097_raw = tmp_path / "strict097_raw.jsonl"
    phase_final_scores = tmp_path / "phase_final_scores.json"
    phase_final_raw = tmp_path / "phase_final_raw.jsonl"
    strict_scores.write_text(
        json.dumps(_scores("strict-run", score=0.531116, cost=2.625)),
        encoding="utf-8",
    )
    loose_scores.write_text(
        json.dumps(
            {
                **_scores("loose-run", score=0.524554, cost=2.625),
                "phase_source_history_char_ratio_min": 0.90,
                "phase_source_target_similarity_min": 0.90,
                "phase_source_text_similarity_min": 0.90,
            }
        ),
        encoding="utf-8",
    )
    strict_raw.write_text(
        "\n".join(
            [
                json.dumps(_repair_record("plan_001", "history", 0.528214, "history_advantage")),
                json.dumps(_repair_record("plan_003", "final", 0.538214, "final_no_advantage")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    loose_raw.write_text(
        "\n".join(
            [
                json.dumps(_repair_record("plan_001", "history", 0.528214, "history_advantage")),
                json.dumps(_repair_record("plan_003", "history", 0.485714, "history_advantage")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    strict097_scores.write_text(
        json.dumps(
            {
                **_scores("strict097-run", score=0.531116, cost=2.625),
                "phase_source_history_char_ratio_min": 0.95,
                "phase_source_target_similarity_min": 0.97,
                "phase_source_text_similarity_min": 0.97,
            }
        ),
        encoding="utf-8",
    )
    strict097_raw.write_text(
        "\n".join(
            [
                json.dumps(_repair_record("plan_001", "final", 0.528214, "final_no_advantage")),
                json.dumps(_repair_record("plan_003", "final", 0.538214, "final_no_advantage")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    phase_final_scores.write_text(
        json.dumps(_scores("phase-final-run", score=0.531116, cost=2.625)),
        encoding="utf-8",
    )
    phase_final_raw.write_text(
        "\n".join(
            [
                json.dumps(_repair_record("plan_001", "final", 0.528214, "final_named", name="phase_final")),
                json.dumps(_repair_record("plan_003", "final", 0.538214, "final_named", name="phase_final")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    audit = build_phase_source_threshold_sweep(
        strict_scores_path=strict_scores,
        strict_raw_path=strict_raw,
        loose_scores_path=loose_scores,
        loose_raw_path=loose_raw,
        strict097_scores_path=strict097_scores,
        strict097_raw_path=strict097_raw,
        phase_final_scores_path=phase_final_scores,
        phase_final_raw_path=phase_final_raw,
    )
    rendered = render_markdown(audit)

    assert audit["schema"] == "diffusion_phase_source_threshold_sweep.v1"
    assert audit["summary"]["best_policies"] == ["strict_096", "strict_097", "phase_final_named"]
    assert audit["summary"]["loose_policy_score_delta"] == pytest.approx(-0.006562)
    assert audit["summary"]["loose_policy_extra_history_switches"] == 1
    assert audit["summary"]["strict097_policy_score_delta"] == pytest.approx(0.0)
    assert audit["summary"]["strict097_history_switches_removed"] == 1
    assert audit["summary"]["phase_final_named_policy_score_delta"] == pytest.approx(0.0)
    assert audit["summary"]["phase_final_named_history_switches_removed"] == 1
    assert audit["source_change_rows"][1]["task_id"] == "plan_003"
    assert audit["source_change_rows"][1]["source_changed"] is True
    assert audit["source_change_rows"][1]["task_score_delta"] == pytest.approx(-0.0525)
    assert audit["source_change_rows"][2]["comparison_policy_id"] == "strict_097"
    assert audit["source_change_rows"][2]["task_id"] == "plan_001"
    assert audit["source_change_rows"][2]["task_score_delta"] == pytest.approx(0.0)
    assert "Diffusion Phase-Source Threshold Sweep" in rendered
    assert "loose_090" in rendered
    assert "strict_097" in rendered
    assert "phase_final_named" in rendered
    assert "n/a" in rendered


def _scores(run_id, *, score, cost):
    return {
        "by_family_arm": {
            "planning": {
                "repair_selected": {
                    "mean_generation_budget_per_task": cost,
                    "mean_task_score": score,
                }
            }
        },
        "run_id": run_id,
    }


def _repair_record(task_id, source_state, score, reason, *, name="phase_hybrid"):
    repair_name = {
        "phase_hybrid": "constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair",
        "phase_final": "constraint_gap_span_phase_final_preserve_seeded_gated_repair",
    }[name]
    return {
        "repair": {
            "anchor_selection_features": {
                "target_similarity": 0.96,
                "text_similarity": 0.98,
            },
            "anchor_selection_reason": reason,
            "name": repair_name,
            "source_state": source_state,
        },
        "task": {"task_id": task_id},
        "task_score": {"score": score},
    }
