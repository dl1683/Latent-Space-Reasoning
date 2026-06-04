import json

from experiments.evaluate_diffusion_transfer_promotion_value import (
    evaluate_transfer_promotion_value,
    render_markdown,
)


def test_transfer_promotion_value_prefers_policy_that_closes_oracle_headroom(tmp_path):
    all_repairable = tmp_path / "all.json"
    weak_policy = tmp_path / "weak.json"
    strong_policy = tmp_path / "strong.json"
    all_repairable.write_text(
        json.dumps(
            {
                "comparison_rows": [
                    _comparison("plan_a", trajectory=0.30, oracle=0.35, selected_lift=0.05),
                    _comparison("plan_b", trajectory=0.50, oracle=0.50, selected_lift=0.0),
                ]
            }
        ),
        encoding="utf-8",
    )
    weak_policy.write_text(
        json.dumps(
            _policy(
                "weak_selector",
                run_id="weak-run",
                headroom=0.05,
                repair_task=0.30,
                rows=[
                    _comparison("plan_a", trajectory=0.30, oracle=0.35, selected_lift=0.0),
                    _comparison("plan_b", trajectory=0.50, oracle=0.50, selected_lift=0.0),
                ],
            )
        ),
        encoding="utf-8",
    )
    strong_policy.write_text(
        json.dumps(
            _policy(
                "strong_selector",
                run_id="strong-run",
                headroom=0.0,
                repair_task=0.35,
                rows=[
                    _comparison("plan_a", trajectory=0.30, oracle=0.35, selected_lift=0.05),
                    _comparison("plan_b", trajectory=0.50, oracle=0.50, selected_lift=0.0),
                ],
            )
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_transfer_promotion_value(
        all_repairable_scores_path=all_repairable,
        policy_score_paths=(weak_policy, strong_policy),
    )

    assert evaluation["summary"]["available_repair_tasks"] == ["plan_a"]
    assert evaluation["summary"]["best_policy"] == "strong_selector"
    assert evaluation["policies"][0]["missed_available_count"] == 1
    assert evaluation["policies"][1]["selected_available_count"] == 1


def test_render_markdown_includes_policy_comparison(tmp_path):
    all_repairable = tmp_path / "all.json"
    policy = tmp_path / "policy.json"
    all_repairable.write_text(
        json.dumps(
            {"comparison_rows": [_comparison("plan_a", trajectory=0.30, oracle=0.35, selected_lift=0.05)]}
        ),
        encoding="utf-8",
    )
    policy.write_text(
        json.dumps(
            _policy(
                "selector",
                run_id="run",
                headroom=0.0,
                repair_task=0.35,
                rows=[_comparison("plan_a", trajectory=0.30, oracle=0.35, selected_lift=0.05)],
            )
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_transfer_promotion_value(
        all_repairable_scores_path=all_repairable,
        policy_score_paths=(policy,),
    )
    markdown = render_markdown(evaluation)

    assert "# Diffusion Transfer Promotion Value" in markdown
    assert "Policy Comparison" in markdown
    assert "Available Repair Rows" in markdown
    assert "candidate_aware_promotion_v1" in markdown


def test_transfer_promotion_value_ignores_nonrepair_oracle_lift(tmp_path):
    all_repairable = tmp_path / "all.json"
    policy = tmp_path / "policy.json"
    all_repairable.write_text(
        json.dumps(
            {
                "comparison_rows": [
                    _comparison("plan_a", trajectory=0.30, oracle=0.40, selected_lift=0.0),
                ]
            }
        ),
        encoding="utf-8",
    )
    policy.write_text(
        json.dumps(
            _policy(
                "selector",
                run_id="run",
                headroom=0.0,
                repair_task=0.30,
                rows=[_comparison("plan_a", trajectory=0.30, oracle=0.40, selected_lift=0.0)],
            )
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_transfer_promotion_value(
        all_repairable_scores_path=all_repairable,
        policy_score_paths=(policy,),
    )

    assert evaluation["summary"]["available_repair_tasks"] == []


def _comparison(task_id, *, trajectory, oracle, selected_lift=0.0):
    return {
        "oracle_delta_vs_trajectory": oracle - trajectory,
        "oracle_task_score": oracle,
        "repair_delta_vs_evolved": selected_lift,
        "task_id": task_id,
        "trajectory_task_score": trajectory,
    }


def _policy(selector, *, run_id, headroom, repair_task, rows):
    return {
        "all_generation_count": 10,
        "arms": {
            "repair_selected": {
                "mean_generation_budget_per_task": 2.0,
                "mean_task_score": repair_task,
            }
        },
        "comparison_rows": rows,
        "content_hash": f"{run_id}-hash",
        "oracle_headroom_vs_repair": headroom,
        "repair_selector": selector,
        "repair_task_delta_vs_trajectory": repair_task - 0.30,
        "run_id": run_id,
    }
