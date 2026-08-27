"""Tests for repairability geometry gate sweeps."""

import json

from experiments.sweep_diffusion_repairability_geometry import (
    build_repairability_geometry_sweep,
    render_markdown,
)


def test_repairability_geometry_sweep_scores_gate_frontier(tmp_path):
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(
        json.dumps(
            {
                "by_family_arm": {
                    "planning": {
                        "fixed": {"mean_task_score": 0.4666666667},
                        "random": {"mean_task_score": 0.3},
                    }
                },
                "repair_source_min_chars": 240,
                "repair_source_prompt_coverage_max": 1.0,
                "repair_source_prompt_coverage_min": 0.4,
                "repair_source_prompt_gap_max": 6,
                "repair_source_prompt_gap_min": 2,
                "repair_source_quality_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )
    audit_path = tmp_path / "audit.json"
    confirmation_path = tmp_path / "phase20_scores.json"
    audit_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "prompt_coverage": 0.5,
                        "prompt_gap_count": 3,
                        "first_repairable_denoise_skeleton_step": 3,
                        "reference_repair_score": 0.55,
                        "source_chars": 300,
                        "source_planning_quality": 0.35,
                        "source_task_score": 0.40,
                        "task_id": "plan_001",
                    },
                    {
                        "prompt_coverage": 0.3,
                        "prompt_gap_count": 7,
                        "reference_repair_score": 0.70,
                        "source_chars": 310,
                        "source_planning_quality": 0.62,
                        "source_task_score": 0.70,
                        "task_id": "plan_002",
                    },
                    {
                        "prompt_coverage": 0.2,
                        "prompt_gap_count": 10,
                        "first_repairable_denoise_skeleton_step": 8,
                        "reference_repair_score": 0.60,
                        "source_chars": 320,
                        "source_planning_quality": 0.25,
                        "source_task_score": 0.30,
                        "task_id": "plan_003",
                    },
                ],
                "scores_path": str(scores_path),
            }
        ),
        encoding="utf-8",
    )
    confirmation_path.write_text(
        json.dumps(
            {
                "all_generation_count": 26,
                "by_family_arm": {
                    "planning": {
                        "fixed": {"mean_task_score": 0.40},
                        "random": {"mean_task_score": 0.30},
                        "repair_selected": {
                            "mean_generation_budget_per_task": 2.5,
                            "mean_task_score": 0.55,
                        },
                    }
                },
                "repair_denoise_skeleton_max_step": 4,
                "repair_pack": "constraint_span_phase_final_preserve_seeded_gated",
                "repair_spend_gate_rows": [
                    {
                        "reason": "denoise_phase_repairable",
                        "should_run": True,
                        "task_id": "plan_001",
                    },
                    {
                        "reason": "late_repairable_denoise_skeleton",
                        "should_run": False,
                        "task_id": "plan_003",
                    },
                ],
                "run_id": "diffusion-confirmed",
            }
        ),
        encoding="utf-8",
    )

    sweep = build_repairability_geometry_sweep(
        audit_path=audit_path,
        quality_thresholds=[0.5],
        gap_mins=[2],
        gap_maxs=[6, 10],
        coverage_mins=[0.2, 0.4],
        coverage_maxs=[1.0],
        skeleton_step_maxs=[None, 4],
        phase_window_confirmation_score_paths=[confirmation_path],
        source_min_chars=240,
        promotion_margin=0.02,
    )
    rendered = render_markdown(sweep)
    current = sweep["current_gate_result"]
    best = sweep["zero_miss_best_score_rows"][0]

    assert sweep["schema"] == "diffusion_repairability_geometry_sweep.v1"
    assert current["spent_tasks"] == ["plan_001"]
    assert current["missed_repair_count"] == 1
    assert best["spent_tasks"] == ["plan_001", "plan_003"]
    assert best["skeleton_step_max"] is None
    assert sweep["summary"]["phase_window_count"] == 2
    assert [
        row["skeleton_step_max"] for row in sweep["phase_window_tradeoff_rows"]
    ] == [None, 4]
    assert sweep["phase_window_confirmation_rows"][0]["run_id"] == "diffusion-confirmed"
    assert (
        sweep["phase_window_confirmation_rows"][0]["repair_pack"]
        == "constraint_span_phase_final_preserve_seeded_gated"
    )
    assert sweep["phase_window_confirmation_rows"][0]["late_skeleton_skip_count"] == 1
    assert any(
        row["skeleton_step_max"] == 4 and row["spent_tasks"] == ["plan_001"]
        for row in sweep["sweep_rows"]
    )
    assert best["selected_score"] > current["selected_score"]
    assert "Diffusion Repairability Geometry Sweep" in rendered
    assert "Phase Window Tradeoff" in rendered
    assert "Fresh Phase-Window Confirmations" in rendered
    assert "diffusion-confirmed" in rendered
    assert "constraint_span_phase_final_preserve_seeded_gated" in rendered
    assert "Skeleton <=" in rendered
    assert "plan_003" in rendered
