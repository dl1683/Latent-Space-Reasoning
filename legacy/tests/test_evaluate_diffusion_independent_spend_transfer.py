import json

from experiments.evaluate_diffusion_independent_spend_transfer import (
    evaluate_independent_spend_transfer,
    render_markdown,
)


def test_independent_spend_transfer_scores_decomposed_against_single_label(tmp_path):
    scores = tmp_path / "scores.json"
    scores.write_text(
        json.dumps(
            {
                "comparison_rows": [
                    _comparison("plan_a", lift=0.20),
                    _comparison("plan_b", lift=0.00),
                    _comparison("plan_c", lift=0.15),
                ],
                "repair_spend_gate_rows": [
                    _gate("plan_a", should_run=True, quality=0.20),
                    _gate("plan_b", should_run=True, quality=0.60),
                    _gate("plan_c", should_run=True, quality=0.70),
                ],
            }
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_independent_spend_transfer(
        all_repairable_scores_path=scores,
        source_quality_max=0.301429,
    )

    assert evaluation["summary"]["single_repairability_error_count"] == 1
    assert evaluation["summary"]["decomposed_error_count"] == 1
    assert evaluation["summary"]["trajectory_relative_error_count"] == 1
    assert evaluation["summary"]["learned_availability_error_count"] == 1
    assert evaluation["summary"]["calibrated_availability_error_count"] == 1
    assert evaluation["summary"]["profitable_tasks"] == ["plan_a", "plan_c"]
    assert evaluation["summary"]["decomposed_selected_tasks"] == ["plan_a"]
    assert evaluation["summary"]["trajectory_relative_selected_tasks"] == ["plan_a"]
    assert evaluation["summary"]["learned_availability_selected_tasks"] == ["plan_a"]
    assert evaluation["summary"]["calibrated_availability_selected_tasks"] == [
        "plan_a",
        "plan_b",
        "plan_c",
    ]


def test_render_markdown_includes_transfer_summary(tmp_path):
    scores = tmp_path / "scores.json"
    scores.write_text(
        json.dumps(
            {
                "comparison_rows": [_comparison("plan_a", lift=0.20)],
                "repair_spend_gate_rows": [_gate("plan_a", should_run=True, quality=0.20)],
            }
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_independent_spend_transfer(all_repairable_scores_path=scores)
    markdown = render_markdown(evaluation)

    assert "# Diffusion Independent Spend Transfer" in markdown
    assert "Single repairability errors" in markdown
    assert "Decomposed spend-head errors" in markdown
    assert "Trajectory-relative spend-head errors" in markdown
    assert "Learned availability predictor errors" in markdown
    assert "Calibrated availability predictor errors" in markdown


def test_independent_spend_transfer_ignores_nonrepair_oracle_lift(tmp_path):
    scores = tmp_path / "scores.json"
    scores.write_text(
        json.dumps(
            {
                "comparison_rows": [
                    _comparison("plan_a", lift=0.0, oracle_lift=0.02),
                ],
                "repair_spend_gate_rows": [
                    _gate("plan_a", should_run=True, quality=0.20),
                ],
            }
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_independent_spend_transfer(all_repairable_scores_path=scores)

    assert evaluation["rows"][0]["repair_lift"] == 0.0
    assert evaluation["rows"][0]["selected_repair_lift"] == 0.0
    assert evaluation["rows"][0]["profitable"] is False
    assert evaluation["summary"]["profitable_tasks"] == []


def test_independent_spend_transfer_blocks_source_below_selected_trajectory(tmp_path):
    scores = tmp_path / "scores.json"
    scores.write_text(
        json.dumps(
            {
                "comparison_rows": [
                    _comparison(
                        "plan_a",
                        lift=0.0,
                        oracle_lift=0.0,
                        trajectory_task_score=0.32,
                    ),
                    _comparison(
                        "plan_b",
                        lift=0.07,
                        oracle_lift=0.07,
                        trajectory_task_score=0.25,
                    ),
                ],
                "repair_spend_gate_rows": [
                    _gate("plan_a", should_run=True, quality=0.20, source_task_score=0.26),
                    _gate("plan_b", should_run=True, quality=0.20, source_task_score=0.25),
                ],
            }
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_independent_spend_transfer(all_repairable_scores_path=scores)

    assert evaluation["summary"]["decomposed_error_count"] == 1
    assert evaluation["summary"]["trajectory_relative_error_count"] == 0
    assert evaluation["summary"]["trajectory_relative_selected_tasks"] == ["plan_b"]
    assert evaluation["summary"]["calibrated_availability_error_count"] == 0


def test_independent_spend_transfer_scores_learned_v3_thresholds(tmp_path):
    scores = tmp_path / "scores.json"
    scores.write_text(
        json.dumps(
            {
                "comparison_rows": [
                    _comparison("plan_a", lift=0.05, oracle_lift=0.05),
                    _comparison("plan_b", lift=0.04, oracle_lift=0.04),
                    _comparison("plan_c", lift=0.00, oracle_lift=0.00),
                ],
                "repair_spend_gate_rows": [
                    _gate(
                        "plan_a",
                        should_run=True,
                        quality=0.25,
                        prompt_gap_count=8,
                        source_task_score=0.30,
                    ),
                    _gate(
                        "plan_b",
                        should_run=True,
                        quality=0.25,
                        prompt_gap_count=9,
                        source_task_score=0.30,
                    ),
                    _gate(
                        "plan_c",
                        should_run=True,
                        quality=0.24,
                        prompt_gap_count=8,
                        source_task_score=-0.01,
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_independent_spend_transfer(all_repairable_scores_path=scores)

    assert evaluation["summary"]["trajectory_relative_selected_tasks"] == [
        "plan_a",
        "plan_b",
    ]
    assert evaluation["summary"]["learned_availability_selected_tasks"] == ["plan_a"]
    assert evaluation["summary"]["learned_availability_error_count"] == 1


def test_independent_spend_transfer_scores_calibrated_gap_boundary(tmp_path):
    scores = tmp_path / "scores.json"
    scores.write_text(
        json.dumps(
            {
                "comparison_rows": [
                    _comparison("plan_good_low", lift=0.05, oracle_lift=0.05),
                    _comparison("plan_gap_seven", lift=0.00, oracle_lift=0.00),
                    _comparison("plan_good_high", lift=0.04, oracle_lift=0.04),
                    _comparison("plan_below_traj", lift=0.00, oracle_lift=0.00),
                ],
                "repair_spend_gate_rows": [
                    _gate(
                        "plan_good_low",
                        should_run=True,
                        quality=0.24,
                        prompt_gap_count=6,
                        source_task_score=0.30,
                    ),
                    _gate(
                        "plan_gap_seven",
                        should_run=True,
                        quality=0.33,
                        prompt_gap_count=7,
                        source_task_score=0.30,
                    ),
                    _gate(
                        "plan_good_high",
                        should_run=True,
                        quality=0.43,
                        prompt_gap_count=9,
                        source_task_score=0.30,
                    ),
                    _gate(
                        "plan_below_traj",
                        should_run=True,
                        quality=0.18,
                        prompt_gap_count=6,
                        source_task_score=-0.01,
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_independent_spend_transfer(all_repairable_scores_path=scores)

    assert evaluation["summary"]["calibrated_availability_error_count"] == 0
    assert evaluation["summary"]["calibrated_availability_selected_tasks"] == [
        "plan_good_low",
        "plan_good_high",
    ]


def _comparison(task_id, *, lift, oracle_lift=None, trajectory_task_score=0.0):
    return {
        "oracle_delta_vs_trajectory": lift if oracle_lift is None else oracle_lift,
        "repair_delta_vs_evolved": lift,
        "task_id": task_id,
        "trajectory_task_score": trajectory_task_score,
    }


def _gate(task_id, *, should_run, quality, prompt_gap_count=4, source_task_score=0.0):
    return {
        "first_repairable_denoise_skeleton_step": 10 if should_run else None,
        "prompt_gap_count": prompt_gap_count,
        "reason": "denoise_phase_repairable" if should_run else "no_repairable_denoise_skeleton",
        "should_run": should_run,
        "source_quality": quality,
        "source_task_score": source_task_score,
        "task_id": task_id,
    }
