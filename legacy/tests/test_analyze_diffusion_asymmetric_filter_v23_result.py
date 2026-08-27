import json

from experiments.analyze_diffusion_asymmetric_filter_v23_result import build_result_summary


def test_asymmetric_filter_result_can_validate_when_cost_offsets_waste(tmp_path):
    freeze = tmp_path / "freeze.json"
    scores = tmp_path / "scores.json"
    targets = tmp_path / "targets.json"
    freeze.write_text(json.dumps(_freeze()), encoding="utf-8")
    scores.write_text(json.dumps(_scores()), encoding="utf-8")
    targets.write_text(json.dumps(_targets()), encoding="utf-8")

    result = build_result_summary(freeze_path=freeze, scores_path=scores, targets_path=targets)

    assert result["summary"]["asymmetric_selected_positive_count"] == 1
    assert result["summary"]["asymmetric_selected_waste_count"] == 0
    assert result["decision"]["status"] == "validated"


def _freeze():
    return {
        "target_surface": {
            "final_preserve_planning_delta_min": 0.005,
            "final_preserve_span_score_min": 1.85,
            "history_prefix_planning_delta_min": 0.20,
        }
    }


def _scores():
    return {
        "comparison_rows": [
            {"repair_control": "history_prefix_25_repair", "task_id": "plan_a"},
            {"repair_control": "constraint_gap_span_phase_final_preserve_seeded_gated_repair", "task_id": "plan_b"},
        ],
        "repair_task_delta_per_extra_generation_vs_evolved": 0.01,
        "repair_task_delta_vs_evolved": 0.02,
        "run_id": "diffusion-test",
    }


def _targets():
    return {
        "rows": [
            {
                "candidate_lift_vs_trajectory": -0.01,
                "max_span_target_score": 0.0,
                "planning_quality_delta_vs_source": 0.1,
                "repair": "history_prefix_25_repair",
                "task_id": "plan_a",
            },
            {
                "candidate_lift_vs_trajectory": 0.02,
                "max_span_target_score": 2.0,
                "planning_quality_delta_vs_source": 0.01,
                "repair": "constraint_gap_span_phase_final_preserve_seeded_gated_repair",
                "task_id": "plan_b",
            },
        ],
        "summary": {"candidate_aware_promotion_error_count": 1},
    }
