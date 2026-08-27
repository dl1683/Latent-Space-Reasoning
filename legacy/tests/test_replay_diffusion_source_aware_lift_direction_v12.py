import json

from experiments.replay_diffusion_source_aware_lift_direction_v12 import replay_v12, render_markdown


def test_v12_replay_scores_selected_and_oracle_failures(tmp_path):
    measurement = tmp_path / "measurement.json"
    labels = tmp_path / "labels.json"
    measurement.write_text(
        json.dumps(
            {
                "row_diagnostics": [
                    {
                        "measured_probe_value_prediction": 0.04,
                        "prompt_coverage": 0.8,
                        "prompt_gap_count": 2,
                        "source_task_delta_vs_trajectory": 0.0,
                        "surface_selected": True,
                        "task_id": "plan_selected_fp",
                    },
                    {
                        "measured_probe_value_prediction": 0.03,
                        "prompt_coverage": 0.4,
                        "prompt_gap_count": 6,
                        "source_task_delta_vs_trajectory": 0.0,
                        "surface_selected": False,
                        "task_id": "plan_selected_fn",
                    },
                    {
                        "measured_probe_value_prediction": 0.02,
                        "prompt_coverage": 0.2,
                        "prompt_gap_count": 12,
                        "source_task_delta_vs_trajectory": 0.0,
                        "surface_selected": False,
                        "task_id": "plan_oracle",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    labels.write_text(
        json.dumps(
            {
                "comparison_rows": [
                    {
                        "oracle_task_score": 0.4,
                        "repair_task_score": 0.4,
                        "task_id": "plan_selected_fp",
                        "trajectory_task_score": 0.4,
                    },
                    {
                        "oracle_task_score": 0.5,
                        "repair_task_score": 0.5,
                        "task_id": "plan_selected_fn",
                        "trajectory_task_score": 0.4,
                    },
                    {
                        "oracle_task_score": 0.6,
                        "repair_task_score": 0.4,
                        "task_id": "plan_oracle",
                        "trajectory_task_score": 0.4,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = replay_v12(measurement_boundary_path=measurement, label_scores_path=labels)
    markdown = render_markdown(result)
    frozen = result["selected_repair_hypotheses"]["frozen_source_aware_surface"]
    oracle = result["oracle_hypotheses"]["frozen_source_aware_surface"]

    assert frozen["false_positive_task_ids"] == ["plan_selected_fp"]
    assert frozen["false_negative_task_ids"] == ["plan_selected_fn"]
    assert oracle["false_negative_task_ids"] == ["plan_selected_fn", "plan_oracle"]
    assert result["summary"]["selected_repair_positive_task_ids"] == ["plan_selected_fn"]
    assert result["summary"]["oracle_positive_task_ids"] == ["plan_selected_fn", "plan_oracle"]
    assert "Do not promote" in markdown
    assert "oracle-positive selector miss" in markdown
