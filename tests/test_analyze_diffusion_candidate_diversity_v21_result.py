import json

from experiments.analyze_diffusion_candidate_diversity_v21_result import (
    build_result_summary,
    render_markdown,
)


def test_candidate_diversity_result_marks_selector_failed_when_waste_selected(tmp_path):
    freeze = tmp_path / "freeze.json"
    scores = tmp_path / "scores.json"
    targets = tmp_path / "targets.json"
    freeze.write_text(
        json.dumps(
            {
                "conclusive_result_gates": {
                    "maximum_selected_no_lift_rows": 0,
                    "minimum_generated_positive_count": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    scores.write_text(json.dumps(_scores()), encoding="utf-8")
    targets.write_text(json.dumps(_targets()), encoding="utf-8")

    result = build_result_summary(
        freeze_path=freeze,
        scores_path=scores,
        targets_path=targets,
    )
    markdown = render_markdown(result)

    assert result["summary"]["positive_count"] == 1
    assert result["summary"]["selected_waste_count"] == 1
    assert result["decision"]["status"] == "availability_positive_selector_failed"
    by_name = {row["repair"]: row for row in result["candidate_name_summary"]}
    assert by_name["candidate_b"]["selected_waste_count"] == 1
    assert "does not validate live broadening" in markdown


def _scores():
    return {
        "all_generation_count": 10,
        "arms": {"repair_selected": {"count": 2}},
        "comparison_rows": [
            {
                "repair_delta_vs_evolved": 0.05,
                "repair_control": "candidate_a",
                "repair_selection_reason": "max_generated_repair_value_v1_score_repair_pool",
                "task_id": "plan_a",
            },
            {
                "repair_delta_vs_evolved": 0.0,
                "repair_control": "candidate_b",
                "repair_selection_reason": "max_generated_repair_value_v1_score_repair_pool",
                "task_id": "plan_b",
            },
        ],
        "repair_task_delta_per_extra_generation_vs_evolved": 0.02,
        "repair_task_delta_vs_evolved": 0.05,
        "repair_task_delta_vs_fixed": 0.04,
        "repair_task_delta_vs_random": 0.06,
        "run_id": "diffusion-test",
    }


def _targets():
    return {
        "rows": [
            {
                "candidate_lift_vs_source": 0.05,
                "candidate_lift_vs_trajectory": 0.05,
                "repair": "candidate_a",
                "task_id": "plan_a",
            },
            {
                "candidate_lift_vs_source": 0.0,
                "candidate_lift_vs_trajectory": 0.0,
                "repair": "candidate_b",
                "task_id": "plan_b",
            },
        ],
        "summary": {"candidate_aware_promotion_error_count": 1},
    }
