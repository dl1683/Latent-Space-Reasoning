import json

from experiments.analyze_diffusion_realization_value_v15_measurement import analyze_measurement, render_markdown


def test_v15_measurement_analysis_authorizes_static_probe_disagreement(tmp_path):
    freeze = tmp_path / "freeze.json"
    measurement = tmp_path / "measurement.json"
    freeze.write_text(json.dumps(_freeze_payload()), encoding="utf-8")
    measurement.write_text(
        json.dumps(
            {
                "all_generation_count": 8,
                "counterfactual_probe_generation_count": 4,
                "repair_spend_gate_rows": [
                    _row("plan_static", source_delta=0.0, gap=4, coverage=0.7, probe=0.04),
                    _row("plan_probe", source_delta=0.0, gap=5, coverage=0.2, probe=0.02),
                    _row("plan_both", source_delta=0.0, gap=5, coverage=0.7, probe=0.02),
                    _row("plan_neither", source_delta=-0.1, gap=5, coverage=0.7, probe=0.02),
                ],
                "run_id": "diffusion-test",
            }
        ),
        encoding="utf-8",
    )

    result = analyze_measurement(freeze_path=freeze, measurement_path=measurement)
    markdown = render_markdown(result)

    assert result["summary"]["label_pass_authorized"] is True
    assert result["summary"]["static_selected_task_ids"] == ["plan_both", "plan_static"]
    assert result["summary"]["probe_selected_task_ids"] == ["plan_both", "plan_probe"]
    assert result["summary"]["static_only_task_ids"] == ["plan_static"]
    assert result["summary"]["probe_only_task_ids"] == ["plan_probe"]
    assert result["summary"]["disagreement_task_ids"] == ["plan_probe", "plan_static"]
    assert "frozen label pass is authorized" in markdown


def test_v15_measurement_analysis_blocks_matching_surfaces(tmp_path):
    freeze = tmp_path / "freeze.json"
    measurement = tmp_path / "measurement.json"
    freeze.write_text(json.dumps(_freeze_payload()), encoding="utf-8")
    measurement.write_text(
        json.dumps(
            {
                "all_generation_count": 4,
                "counterfactual_probe_generation_count": 2,
                "repair_spend_gate_rows": [
                    _row("plan_both", source_delta=0.0, gap=5, coverage=0.7, probe=0.02),
                    _row("plan_neither", source_delta=-0.1, gap=5, coverage=0.7, probe=0.02),
                ],
                "run_id": "diffusion-test",
            }
        ),
        encoding="utf-8",
    )

    result = analyze_measurement(freeze_path=freeze, measurement_path=measurement)
    markdown = render_markdown(result)

    assert result["summary"]["label_pass_authorized"] is False
    assert result["summary"]["disagreement_task_ids"] == []
    assert "Do not run the v15 label pass" in markdown


def _freeze_payload():
    return {
        "conclusive_result_gates": {
            "minimum_static_probe_disagreement_count": 1,
        },
        "planning_task_ids": ["plan_static", "plan_probe", "plan_both", "plan_neither"],
        "target_surfaces": [
            {
                "prompt_coverage_max": 1.0,
                "prompt_coverage_min": 0.4,
                "prompt_gap_count_max": 7,
                "prompt_gap_count_min": 4,
                "requires_label_pass_denoise_trigger": True,
                "source_task_delta_vs_trajectory_min": 0.0,
                "surface_id": "static_source_gap_coverage_v15",
                "uses_probe_measurement": False,
            },
            {
                "measured_probe_value_prediction_max": 0.033,
                "prompt_gap_count_max": 7,
                "prompt_gap_count_min": 4,
                "requires_label_pass_denoise_trigger": True,
                "source_task_delta_vs_trajectory_min": 0.0,
                "surface_id": "probe_conditioned_realization_value_v15",
                "uses_probe_measurement": True,
            },
        ],
    }


def _row(task_id, *, source_delta, gap, coverage, probe):
    return {
        "measured_probe_value_prediction": probe,
        "peak_denoise_prompt_coverage": coverage / 2,
        "prompt_coverage": coverage,
        "prompt_gap_count": gap,
        "source_control": "random",
        "source_task_delta_vs_trajectory": source_delta,
        "task_id": task_id,
        "would_probe": True,
    }
