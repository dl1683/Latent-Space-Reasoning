import json

from experiments.analyze_diffusion_realization_value_v14_measurement import analyze_measurement, render_markdown


def test_v14_measurement_analysis_blocks_empty_surface(tmp_path):
    freeze = tmp_path / "freeze.json"
    measurement = tmp_path / "measurement.json"
    freeze.write_text(
        json.dumps(
            {
                "planning_task_ids": ["plan_a", "plan_b"],
                "target_surface": {
                    "measured_probe_value_prediction_max": 0.032,
                    "prompt_gap_count_max": 7,
                    "prompt_gap_count_min": 4,
                    "requires_label_pass_denoise_trigger": True,
                    "source_task_delta_vs_trajectory_min": 0.0,
                    "surface_id": "realization_value_probe_banded_v14",
                },
            }
        ),
        encoding="utf-8",
    )
    measurement.write_text(
        json.dumps(
            {
                "all_generation_count": 4,
                "counterfactual_probe_generation_count": 2,
                "repair_spend_gate_rows": [
                    {
                        "measured_probe_value_prediction": 0.033,
                        "peak_denoise_prompt_coverage": 0.4,
                        "prompt_coverage": 0.6,
                        "prompt_gap_count": 5,
                        "source_control": "fixed",
                        "source_task_delta_vs_trajectory": 0.0,
                        "task_id": "plan_a",
                        "would_probe": True,
                    },
                    {
                        "measured_probe_value_prediction": 0.02,
                        "peak_denoise_prompt_coverage": 0.1,
                        "prompt_coverage": 0.4,
                        "prompt_gap_count": 8,
                        "source_control": "random",
                        "source_task_delta_vs_trajectory": -0.1,
                        "task_id": "plan_b",
                        "would_probe": True,
                    },
                ],
                "run_id": "diffusion-test",
            }
        ),
        encoding="utf-8",
    )

    result = analyze_measurement(freeze_path=freeze, measurement_path=measurement)
    markdown = render_markdown(result)

    assert result["summary"]["measurement_gate_passed"] is False
    assert result["summary"]["surface_selected_task_ids"] == []
    assert result["summary"]["near_miss_task_ids"] == ["plan_a"]
    assert "Do not run the v14 label pass" in markdown


def test_v14_measurement_analysis_authorizes_nonempty_source_divergent_surface(tmp_path):
    freeze = tmp_path / "freeze.json"
    measurement = tmp_path / "measurement.json"
    freeze.write_text(
        json.dumps(
            {
                "planning_task_ids": ["plan_a", "plan_b"],
                "target_surface": {
                    "measured_probe_value_prediction_max": 0.032,
                    "prompt_gap_count_max": 7,
                    "prompt_gap_count_min": 4,
                    "requires_label_pass_denoise_trigger": True,
                    "source_task_delta_vs_trajectory_min": 0.0,
                    "surface_id": "realization_value_probe_banded_v14",
                },
            }
        ),
        encoding="utf-8",
    )
    measurement.write_text(
        json.dumps(
            {
                "all_generation_count": 4,
                "counterfactual_probe_generation_count": 2,
                "repair_spend_gate_rows": [
                    {
                        "measured_probe_value_prediction": 0.02,
                        "peak_denoise_prompt_coverage": 0.4,
                        "prompt_coverage": 0.6,
                        "prompt_gap_count": 5,
                        "source_control": "fixed",
                        "source_task_delta_vs_trajectory": 0.0,
                        "task_id": "plan_a",
                        "would_probe": True,
                    },
                    {
                        "measured_probe_value_prediction": 0.02,
                        "peak_denoise_prompt_coverage": 0.1,
                        "prompt_coverage": 0.4,
                        "prompt_gap_count": 8,
                        "source_control": "random",
                        "source_task_delta_vs_trajectory": -0.1,
                        "task_id": "plan_b",
                        "would_probe": True,
                    },
                ],
                "run_id": "diffusion-test",
            }
        ),
        encoding="utf-8",
    )

    result = analyze_measurement(freeze_path=freeze, measurement_path=measurement)

    assert result["summary"]["measurement_gate_passed"] is True
    assert result["summary"]["surface_selected_task_ids"] == ["plan_a"]
    assert result["summary"]["negative_source_delta_task_ids"] == ["plan_b"]
