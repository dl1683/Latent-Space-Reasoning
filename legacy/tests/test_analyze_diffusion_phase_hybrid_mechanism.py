"""Tests for strict phase-hybrid mechanism audit generation."""

import json

import pytest

from experiments.analyze_diffusion_phase_hybrid_mechanism import (
    build_phase_hybrid_mechanism_audit,
    render_markdown,
)


def test_phase_hybrid_mechanism_audit_tracks_error_correction_loop(tmp_path):
    scores = tmp_path / "scores.json"
    raw = tmp_path / "raw.jsonl"
    scores.write_text(
        json.dumps(
            {
                "by_family_arm": {
                    "planning": {
                        "repair_selected": {
                            "mean_generation_budget_per_task": 2.625,
                            "mean_task_score": 0.60,
                        }
                    }
                },
                "comparison_rows": [
                    {
                        "fixed_task_score": 0.40,
                        "random_task_score": 0.30,
                        "repair_control": "constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair",
                        "repair_source_state": "history",
                        "repair_task_score": 0.70,
                        "task_id": "plan_001",
                    },
                    {
                        "fixed_task_score": 0.50,
                        "random_task_score": 0.35,
                        "repair_control": "constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair",
                        "repair_source_state": "final",
                        "repair_task_score": 0.50,
                        "task_id": "plan_002",
                    },
                ],
                "run_id": "diffusion-test",
            }
        ),
        encoding="utf-8",
    )
    raw.write_text(
        "\n".join(
            [
                json.dumps(
                    _repair_record(
                        task_id="plan_001",
                        source_state="history",
                        source_task=0.40,
                        repair_task=0.70,
                        reason="phase_hybrid_history_source_advantage",
                        first_repairable=10,
                        first_safe=30,
                        lag=20,
                        target_similarity=0.97,
                    )
                ),
                json.dumps(
                    _repair_record(
                        task_id="plan_002",
                        source_state="final",
                        source_task=0.50,
                        repair_task=0.50,
                        reason="phase_hybrid_final_no_source_advantage",
                        first_repairable=12,
                        first_safe=12,
                        lag=0,
                        target_similarity=0.91,
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    audit = build_phase_hybrid_mechanism_audit(scores_path=scores, raw_path=raw)
    rendered = render_markdown(audit)

    assert audit["schema"] == "diffusion_phase_hybrid_mechanism_audit.v2"
    assert audit["summary"]["run_id"] == "diffusion-test"
    assert audit["summary"]["planning_repair_score"] == pytest.approx(0.60)
    assert audit["summary"]["relative_gpu_cost"] == pytest.approx(2.625)
    assert audit["summary"]["source_state_counts"] == {"history": 1, "final": 1}
    assert audit["summary"]["positive_delta_count"] == 1
    assert audit["summary"]["mean_delta_vs_source"] == pytest.approx(0.15)
    assert audit["summary"]["mean_phase_retention_safety_lag"] == pytest.approx(10.0)
    assert audit["summary"]["final_kept_with_phase_signal_count"] == 1
    assert audit["summary"]["loss_target_count"] == 2
    assert audit["summary"]["history_trust_loss_target_count"] == 1
    assert audit["summary"]["final_preserve_loss_target_count"] == 1
    assert audit["summary"]["mean_loss_weight"] == pytest.approx(0.15)
    assert audit["loss_targets"][0]["target_action"] == "trust_history_source"
    assert audit["loss_targets"][0]["label"] == 1
    assert audit["loss_targets"][0]["loss_weight"] == pytest.approx(0.30)
    assert audit["loss_targets"][1]["target_action"] == "preserve_final_source"
    assert audit["loss_targets"][1]["label"] == 0
    assert audit["loss_targets"][1]["loss_weight"] == pytest.approx(0.0)
    assert "Diffusion Phase-Hybrid Mechanism Audit" in rendered
    assert "Phase-Source Loss Targets" in rendered
    assert "phase_hybrid_history_source_advantage" in rendered


def _repair_record(
    *,
    task_id,
    source_state,
    source_task,
    repair_task,
    reason,
    first_repairable,
    first_safe,
    lag,
    target_similarity,
):
    return {
        "generation_stage": "repair_candidate",
        "repair": {
            "anchor_selection_features": {
                "phase_first_repairable_step": first_repairable,
                "phase_first_safe_repairable_step": first_safe,
                "phase_repairable_sample_count": 2,
                "phase_retention_safety_lag": lag,
                "phase_safe_repairable_sample_count": 1,
                "target_similarity": target_similarity,
                "text_similarity": 0.98,
            },
            "anchor_selection_reason": reason,
            "name": "constraint_gap_span_phase_hybrid_preserve_seeded_gated_repair",
            "source_state": source_state,
            "source_task_score": source_task,
        },
        "task": {"task_id": task_id},
        "task_score": {"score": repair_task},
    }
