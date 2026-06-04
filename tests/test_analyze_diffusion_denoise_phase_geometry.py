"""Tests for denoise-phase geometry audit generation."""

import json

from experiments.analyze_diffusion_denoise_phase_geometry import (
    build_denoise_phase_audit,
    render_markdown,
)


def test_denoise_phase_audit_tracks_constraint_skeleton_steps(tmp_path):
    raw = tmp_path / "raw.jsonl"
    repairability = tmp_path / "repairability.json"
    prompt = "Collect baseline metrics rollback threshold intervention failure."
    raw.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "generation_stage": "candidate_generation",
                    "history_steps": 4,
                    "planning_quality_score": 0.30,
                    "prompt": prompt,
                    "schedule": {"name": "low_confidence_32"},
                    "task": {"family": "planning", "task_id": "plan_001"},
                    "text": "Collect baseline metrics and preserve rollback for failure.",
                    "trajectory_summary": {
                        "first_mask_free_step": 4,
                        "mask_count_increase_count": 0,
                        "samples": [
                            {
                                "eos_count": 0,
                                "mask_count": 6,
                                "step": 1,
                                "visible_chars": 0,
                                "visible_text": "",
                            },
                            {
                                "eos_count": 1,
                                "mask_count": 4,
                                "step": 2,
                                "visible_chars": 24,
                                "visible_text": "Collect baseline metrics",
                            },
                            {
                                "eos_count": 1,
                                "mask_count": 2,
                                "step": 3,
                                "visible_chars": 42,
                                "visible_text": "Collect baseline metrics and preserve rollback",
                            },
                            {
                                "eos_count": 1,
                                "mask_count": 0,
                                "step": 4,
                                "visible_chars": 56,
                                "visible_text": "Collect baseline metrics and preserve rollback for failure.",
                            },
                        ],
                    },
                },
                {
                    "generation_stage": "candidate_generation",
                    "history_steps": 4,
                    "planning_quality_score": 0.10,
                    "prompt": prompt,
                    "schedule": {"name": "random_32"},
                    "task": {"family": "planning", "task_id": "plan_002"},
                    "text": "Do something later.",
                    "trajectory_summary": {
                        "first_mask_free_step": 4,
                        "mask_count_increase_count": 0,
                        "samples": [
                            {
                                "eos_count": 0,
                                "mask_count": 4,
                                "step": 1,
                                "visible_chars": 0,
                                "visible_text": "",
                            },
                            {
                                "eos_count": 1,
                                "mask_count": 2,
                                "step": 2,
                                "visible_chars": 12,
                                "visible_text": "Do something",
                            },
                            {
                                "eos_count": 1,
                                "mask_count": 0,
                                "step": 4,
                                "visible_chars": 19,
                                "visible_text": "Do something later.",
                            },
                        ],
                    },
                },
            ]
        ),
        encoding="utf-8",
    )
    repairability.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "classification": "productive_spend",
                        "source_control": "low_confidence_32",
                        "task_id": "plan_001",
                    },
                    {
                        "classification": "skipped_no_lift",
                        "source_control": "random_32",
                        "task_id": "plan_002",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    audit = build_denoise_phase_audit(
        raw_path=raw,
        repairability_audit_path=repairability,
        coverage_floor=0.30,
        quality_floor=0.20,
    )
    rendered = render_markdown(audit)
    rows = {row["task_id"]: row for row in audit["rows"]}

    assert audit["schema"] == "diffusion_denoise_phase_geometry.v1"
    assert rows["plan_001"]["first_skeleton_step"] == 2
    assert rows["plan_001"]["phase"] in {"repairable_skeleton", "low_quality_repairable_skeleton"}
    assert rows["plan_002"]["phase"] == "undercovered_or_overdiffuse"
    assert audit["summary"]["repairable_phase_precision"] == 1.0
    assert audit["summary"]["repairable_phase_recall"] == 1.0
    assert audit["summary"]["row_count"] == 2
    assert "Diffusion Denoise Phase Geometry" in rendered
    assert "plan_001" in rendered
