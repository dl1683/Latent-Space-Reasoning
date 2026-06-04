import json

from experiments.replay_diffusion_span_probe_value_floor_v11 import replay_value_floor_v11


def test_value_floor_replay_scores_false_positives_and_probe_cost(tmp_path):
    freeze = tmp_path / "freeze.json"
    measurement = tmp_path / "measurement.json"
    labels = tmp_path / "labels.json"
    freeze.write_text(
        json.dumps(
            {
                "controller": {
                    "feature": "measured_probe_value_prediction",
                    "operator": "ge",
                    "threshold": 0.03,
                    "rule_id": "measured_probe_value_prediction_ge_0p030000",
                }
            }
        ),
        encoding="utf-8",
    )
    measurement.write_text(
        json.dumps(
            {
                "repair_spend_gate_rows": [
                    {
                        "counterfactual_probe_cost_relative": 0.1875,
                        "measured_probe_value_prediction": 0.04,
                        "source_task_delta_vs_trajectory": -0.2,
                        "task_id": "plan_a",
                    },
                    {
                        "counterfactual_probe_cost_relative": 0.1875,
                        "measured_probe_value_prediction": 0.02,
                        "source_task_delta_vs_trajectory": -0.1,
                        "task_id": "plan_b",
                    },
                    {
                        "counterfactual_probe_cost_relative": 0.1875,
                        "measured_probe_value_prediction": 0.05,
                        "source_task_delta_vs_trajectory": 0.0,
                        "task_id": "plan_c",
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
                    {"task_id": "plan_a", "trajectory_task_score": 0.4, "repair_task_score": 0.5, "oracle_task_score": 0.5},
                    {"task_id": "plan_b", "trajectory_task_score": 0.4, "repair_task_score": 0.6, "oracle_task_score": 0.6},
                    {"task_id": "plan_c", "trajectory_task_score": 0.4, "repair_task_score": 0.4, "oracle_task_score": 0.4},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = replay_value_floor_v11(
        freeze_json_path=freeze,
        measurement_scores_path=measurement,
        label_scores_path=labels,
        selection_penalty=0.02,
    )

    assert result["summary"]["selected_count"] == 2
    assert result["summary"]["positive_count"] == 2
    assert result["summary"]["false_positive_task_ids"] == ["plan_c"]
    assert result["summary"]["false_negative_task_ids"] == ["plan_b"]
    assert round(result["summary"]["policy_utility"], 6) == 0.06
    assert round(result["summary"]["probe_cost_penalty"], 6) == 0.01125
    assert round(result["summary"]["policy_utility_with_probe_cost"], 6) == 0.04875
