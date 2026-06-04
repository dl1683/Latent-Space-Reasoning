import json

from experiments.analyze_diffusion_denoise_phase_realization_v13_measurement import (
    analyze_measurement,
    render_markdown,
)


def test_v13_measurement_analysis_scores_denoise_realization_surface(tmp_path):
    freeze = tmp_path / "freeze.json"
    measurement = tmp_path / "measurement.json"
    freeze.write_text(
        json.dumps(
            {
                "planning_task_ids": ["plan_a", "plan_b", "plan_c"],
                "target_surface": {
                    "first_repairable_denoise_skeleton_step_fraction_max": 0.4,
                    "peak_denoise_prompt_coverage_min": 0.4,
                    "requires_repairable_denoise_skeleton": True,
                    "source_task_delta_vs_trajectory_min": 0.0,
                    "surface_id": "source_aligned_denoise_realization_v13",
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
                        "first_repairable_denoise_skeleton_step_fraction": 0.25,
                        "has_repairable_denoise_skeleton": True,
                        "measured_probe_value_prediction": 0.01,
                        "peak_denoise_prompt_coverage": 0.5,
                        "prompt_coverage": 0.8,
                        "prompt_gap_count": 3,
                        "source_control": "fixed",
                        "source_task_delta_vs_trajectory": 0.0,
                        "task_id": "plan_a",
                        "would_probe": True,
                    },
                    {
                        "first_repairable_denoise_skeleton_step_fraction": 0.25,
                        "has_repairable_denoise_skeleton": True,
                        "measured_probe_value_prediction": 0.06,
                        "peak_denoise_prompt_coverage": 0.5,
                        "prompt_coverage": 0.7,
                        "prompt_gap_count": 4,
                        "source_control": "random",
                        "source_task_delta_vs_trajectory": -0.1,
                        "task_id": "plan_b",
                        "would_probe": True,
                    },
                    {
                        "first_repairable_denoise_skeleton_step_fraction": 0.7,
                        "has_repairable_denoise_skeleton": True,
                        "measured_probe_value_prediction": 0.04,
                        "peak_denoise_prompt_coverage": 0.2,
                        "prompt_coverage": 0.3,
                        "prompt_gap_count": 8,
                        "source_control": "random",
                        "source_task_delta_vs_trajectory": 0.2,
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

    assert result["summary"]["measurement_gate_passed"] is True
    assert result["summary"]["source_divergent_task_ids"] == ["plan_b", "plan_c"]
    assert result["summary"]["negative_source_delta_task_ids"] == ["plan_b"]
    assert result["summary"]["positive_source_delta_task_ids"] == ["plan_c"]
    assert result["summary"]["surface_selected_task_ids"] == ["plan_a"]
    assert result["summary"]["high_probe_blocked_task_ids"] == ["plan_b", "plan_c"]
    assert result["summary"]["skeleton_only_task_ids"] == ["plan_a", "plan_b", "plan_c"]
    assert result["row_diagnostics"][0]["prompt_gap_count"] == 3.0
    assert result["row_diagnostics"][0]["prompt_coverage"] == 0.8
    assert "label pass is authorized" in markdown
    assert "Skeleton presence alone is too broad" in markdown
