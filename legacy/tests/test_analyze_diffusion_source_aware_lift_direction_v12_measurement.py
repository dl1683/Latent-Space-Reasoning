import json

from experiments.analyze_diffusion_source_aware_lift_direction_v12_measurement import (
    analyze_measurement,
    render_markdown,
)


def test_v12_measurement_analysis_scores_source_aware_surface(tmp_path):
    freeze = tmp_path / "freeze.json"
    measurement = tmp_path / "measurement.json"
    freeze.write_text(
        json.dumps(
            {
                "planning_task_ids": ["plan_a", "plan_b", "plan_c"],
                "target_surface": {
                    "prompt_coverage_min": 0.7,
                    "prompt_gap_count_max": 4.0,
                    "probe_value_feature_role": "recorded_for_diagnostics_not_positive_direction_threshold",
                    "source_task_delta_vs_trajectory_min": 0.0,
                    "surface_id": "source_nonnegative_gap_le_4_coverage_ge_0p7_frozen_for_v12",
                },
            }
        ),
        encoding="utf-8",
    )
    measurement.write_text(
        json.dumps(
            {
                "all_generation_count": 6,
                "counterfactual_probe_generation_count": 3,
                "repair_spend_gate_rows": [
                    {
                        "measured_probe_value_prediction": 0.05,
                        "prompt_coverage": 0.8,
                        "prompt_gap_count": 3,
                        "source_control": "fixed",
                        "source_task_delta_vs_trajectory": 0.0,
                        "task_id": "plan_a",
                        "would_probe": True,
                    },
                    {
                        "measured_probe_value_prediction": 0.06,
                        "prompt_coverage": 0.8,
                        "prompt_gap_count": 3,
                        "source_control": "random",
                        "source_task_delta_vs_trajectory": -0.1,
                        "task_id": "plan_b",
                        "would_probe": True,
                    },
                    {
                        "measured_probe_value_prediction": 0.04,
                        "prompt_coverage": 0.2,
                        "prompt_gap_count": 12,
                        "source_control": "random",
                        "source_task_delta_vs_trajectory": 0.0,
                        "task_id": "plan_c",
                        "would_probe": False,
                    },
                ],
                "run_id": "diffusion-test",
            }
        ),
        encoding="utf-8",
    )

    result = analyze_measurement(freeze_path=freeze, measurement_path=measurement)
    markdown = render_markdown(result)

    assert result["summary"]["source_divergence_gate_passed"] is True
    assert result["summary"]["negative_source_delta_task_ids"] == ["plan_b"]
    assert result["summary"]["surface_selected_task_ids"] == ["plan_a"]
    assert result["summary"]["high_probe_blocked_task_ids"] == ["plan_b", "plan_c"]
    assert "label pass is authorized" in markdown
    assert "Probe value role" in markdown
