import json

from experiments.analyze_diffusion_history_guard_v24_result import build_result_summary


def test_history_guard_result_revalidates_unchanged_when_it_selects_all_positives(tmp_path):
    scores = tmp_path / "scores.json"
    targets = tmp_path / "targets.json"
    scores.write_text(json.dumps(_scores()), encoding="utf-8")
    targets.write_text(json.dumps(_targets()), encoding="utf-8")

    result = build_result_summary(scores_path=scores, targets_path=targets)

    assert result["summary"]["unchanged_selected_positive_count"] == 2
    assert result["summary"]["unchanged_selected_waste_count"] == 0
    assert result["decision"]["status"] == "unchanged_baseline_revalidated"


def _scores():
    return {
        "all_generation_count": 10,
        "comparison_rows": [
            {"repair_control": "history_prefix_25_repair", "task_id": "plan_a"},
            {
                "repair_control": "constraint_gap_span_phase_final_preserve_seeded_gated_repair",
                "task_id": "plan_b",
            },
        ],
        "oracle_headroom_vs_repair": 0.0,
        "repair_task_delta_per_extra_generation_vs_evolved": 0.01,
        "repair_task_delta_vs_evolved": 0.02,
        "run_id": "diffusion-test",
    }


def _targets():
    return {
        "rows": [
            {
                "candidate_lift_vs_trajectory": 0.03,
                "max_span_target_score": 0.0,
                "planning_quality_delta_vs_source": 0.03,
                "repair": "history_prefix_25_repair",
                "task_id": "plan_a",
            },
            {
                "candidate_lift_vs_trajectory": -0.01,
                "max_span_target_score": 2.0,
                "planning_quality_delta_vs_source": 0.0,
                "repair": "constraint_gap_span_phase_final_preserve_seeded_gated_repair",
                "task_id": "plan_a",
            },
            {
                "candidate_lift_vs_trajectory": -0.02,
                "max_span_target_score": 0.0,
                "planning_quality_delta_vs_source": -0.02,
                "repair": "history_prefix_25_repair",
                "task_id": "plan_b",
            },
            {
                "candidate_lift_vs_trajectory": 0.04,
                "max_span_target_score": 2.0,
                "planning_quality_delta_vs_source": 0.04,
                "repair": "constraint_gap_span_phase_final_preserve_seeded_gated_repair",
                "task_id": "plan_b",
            },
        ],
        "summary": {"candidate_aware_promotion_error_count": 2},
    }
