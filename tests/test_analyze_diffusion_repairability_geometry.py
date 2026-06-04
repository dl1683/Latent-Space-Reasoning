"""Tests for diffusion repairability geometry audit generation."""

import json

from experiments.analyze_diffusion_repairability_geometry import (
    build_repairability_audit,
    render_markdown,
)


def _write_raw(path):
    records = [
        {
            "generation_stage": "candidate_generation",
            "planning_quality_score": 0.35,
            "prompt": (
                "Compare a baseline and intervention, record metrics, preserve "
                "rollback, and define a threshold."
            ),
            "schedule": {"name": "low_confidence_32"},
            "task": {"family": "planning", "task_id": "plan_001"},
            "task_score": {"score": 0.40},
            "text": "Compare the baseline and intervention, record metrics, and preserve rollback.",
            "trajectory_control_score": {"overall": 0.66},
            "trajectory_summary": {
                "samples": [
                    {"step": 2, "visible_chars": 18, "visible_text": "Compare baseline."},
                    {
                        "step": 4,
                        "visible_chars": 48,
                        "visible_text": "Compare baseline and intervention; record metrics.",
                    },
                ]
            },
        },
        {
            "generation_stage": "candidate_generation",
            "planning_quality_score": 0.62,
            "prompt": (
                "Compare a baseline and intervention, record metrics, preserve "
                "rollback, and define a threshold."
            ),
            "schedule": {"name": "low_confidence_32"},
            "task": {"family": "planning", "task_id": "plan_002"},
            "task_score": {"score": 0.70},
            "text": (
                "Compare the baseline and intervention, record metrics, preserve "
                "rollback, and define the threshold before launch."
            ),
            "trajectory_control_score": {"overall": 0.70},
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")


def _write_scores(path):
    path.write_text(
        json.dumps(
            {
                "comparison_rows": [
                    {
                        "fixed_schedule": "low_confidence_32",
                        "fixed_task_score": 0.40,
                        "random_schedule": "random_32",
                        "random_task_score": 0.10,
                        "repair_control": "constraint_gap_span_repair",
                        "repair_selection_reason": "max_repair_pool",
                        "repair_source_control": "low_confidence_32",
                        "repair_task_score": 0.55,
                        "task_id": "plan_001",
                    },
                    {
                        "fixed_schedule": "low_confidence_32",
                        "fixed_task_score": 0.70,
                        "random_schedule": "random_32",
                        "random_task_score": 0.20,
                        "repair_control": "low_confidence_32",
                        "repair_selection_reason": "repair_spend_gate_kept_evolved",
                        "repair_source_control": "",
                        "repair_task_score": 0.70,
                        "task_id": "plan_002",
                    },
                ],
                "repair_spend_gate_rows": [
                    {
                        "has_repairable_denoise_skeleton": True,
                        "first_repairable_denoise_skeleton_step": 4,
                        "first_repairable_denoise_skeleton_step_fraction": 0.5,
                        "peak_denoise_prompt_coverage": 0.5,
                        "in_repairable_band": True,
                        "prompt_coverage": 0.50,
                        "prompt_gap_count": 3,
                        "reason": "denoise_phase_repairable",
                        "should_run": True,
                        "task_id": "plan_001",
                    },
                    {
                        "has_repairable_denoise_skeleton": False,
                        "in_repairable_band": False,
                        "prompt_coverage": 0.25,
                        "prompt_gap_count": 10,
                        "reason": "outside_repairable_band",
                        "should_run": False,
                        "task_id": "plan_002",
                    },
                ],
                "repair_spend_trigger": "source_repairability_geometry",
            }
        ),
        encoding="utf-8",
    )


def _write_reference_scores(path):
    path.write_text(
        json.dumps(
            {
                "comparison_rows": [
                    {"repair_task_score": 0.55, "task_id": "plan_001"},
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_extra_reference_scores(path):
    path.write_text(
        json.dumps(
            {
                "comparison_rows": [
                    {"repair_task_score": 0.70, "task_id": "plan_002"},
                ]
            }
        ),
        encoding="utf-8",
    )


def test_repairability_audit_classifies_productive_and_skipped_rows(tmp_path):
    raw = tmp_path / "raw.jsonl"
    scores = tmp_path / "scores.json"
    reference = tmp_path / "reference.json"
    extra_reference = tmp_path / "extra_reference.json"
    _write_raw(raw)
    _write_scores(scores)
    _write_reference_scores(reference)
    _write_extra_reference_scores(extra_reference)

    audit = build_repairability_audit(
        scores_path=scores,
        raw_path=raw,
        reference_scores_path=reference,
        extra_reference_scores_paths=[extra_reference],
        promotion_margin=0.02,
    )
    rendered = render_markdown(audit)
    rows = {row["task_id"]: row for row in audit["rows"]}

    assert audit["schema"] == "diffusion_repairability_geometry_audit.v1"
    assert audit["extra_reference_scores_paths"] == [str(extra_reference)]
    assert rows["plan_001"]["classification"] == "productive_spend"
    assert rows["plan_001"]["gate_should_run"] is True
    assert rows["plan_001"]["gate_reason"] == "denoise_phase_repairable"
    assert rows["plan_001"]["first_repairable_denoise_skeleton_step"] == 4
    assert rows["plan_001"]["gate_first_repairable_denoise_skeleton_step"] == 4
    assert rows["plan_001"]["peak_denoise_prompt_coverage"] > 0.0
    assert rows["plan_002"]["classification"] == "skipped_no_lift"
    assert rows["plan_002"]["gate_should_run"] is False
    assert rows["plan_002"]["gate_prompt_gap_count"] == 10
    assert audit["summary"]["productive_spend_count"] == 1
    assert audit["summary"]["skipped_no_lift_count"] == 1
    assert audit["summary"]["gate_true_positive_count"] == 1
    assert audit["summary"]["gate_true_negative_count"] == 1
    assert audit["summary"]["mean_first_skeleton_step_spent"] == 4.0
    assert audit["summary"]["gate_reason_counts"] == {
        "denoise_phase_repairable": 1,
        "outside_repairable_band": 1,
    }
    assert "Diffusion Repairability Geometry Audit" in rendered
    assert "Gate TP/FP/TN/FN" in rendered
    assert "plan_001" in rendered
