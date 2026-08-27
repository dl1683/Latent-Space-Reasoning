import json

from experiments.analyze_diffusion_denoise_phase_realization_v13_counterexamples import (
    analyze_counterexamples,
    render_markdown,
)


def test_v13_counterexample_analysis_finds_realization_value_band(tmp_path):
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps({"row_diagnostics": _rows()}), encoding="utf-8")

    analysis = analyze_counterexamples(replay_path=replay)
    markdown = render_markdown(analysis)

    assert analysis["summary"]["counterexample_count"] == 6
    assert analysis["summary"]["selected_repair_positive_task_ids"] == ["plan_099", "plan_104"]
    assert analysis["summary"]["broad_trigger_no_lift_task_ids"] == ["plan_098", "plan_100", "plan_101"]
    best = analysis["selected_repair_hypotheses"][0]
    assert best["hypothesis_id"] == "label_trigger_source_nonnegative_gap_4_to_7_probe_le_0p032"
    assert best["false_positive_task_ids"] == []
    assert best["false_negative_task_ids"] == []
    assert "realization-value counterexample map" in markdown


def _rows():
    return [
        {
            "has_repairable_denoise_skeleton": True,
            "label": False,
            "label_pass_denoise_trigger": False,
            "measured_probe_value_prediction": 0.011,
            "oracle_label": True,
            "oracle_lift_vs_trajectory": 0.108,
            "peak_denoise_prompt_coverage": 0.125,
            "prompt_coverage": 0.375,
            "prompt_gap_count": 10,
            "repair_lift_vs_trajectory": 0.0,
            "source_task_delta_vs_trajectory": 0.108,
            "surface_selected": False,
            "task_id": "plan_097",
        },
        {
            "has_repairable_denoise_skeleton": True,
            "label": False,
            "label_pass_denoise_trigger": True,
            "measured_probe_value_prediction": 0.049,
            "oracle_label": False,
            "oracle_lift_vs_trajectory": 0.0,
            "peak_denoise_prompt_coverage": 0.307,
            "prompt_coverage": 0.538,
            "prompt_gap_count": 6,
            "repair_lift_vs_trajectory": 0.0,
            "source_task_delta_vs_trajectory": -0.175,
            "surface_selected": False,
            "task_id": "plan_098",
        },
        {
            "has_repairable_denoise_skeleton": True,
            "label": True,
            "label_pass_denoise_trigger": True,
            "measured_probe_value_prediction": 0.031,
            "oracle_label": True,
            "oracle_lift_vs_trajectory": 0.139,
            "peak_denoise_prompt_coverage": 0.5,
            "prompt_coverage": 0.786,
            "prompt_gap_count": 4,
            "repair_lift_vs_trajectory": 0.139,
            "source_task_delta_vs_trajectory": 0.0,
            "surface_selected": True,
            "task_id": "plan_099",
        },
        {
            "has_repairable_denoise_skeleton": True,
            "label": False,
            "label_pass_denoise_trigger": True,
            "measured_probe_value_prediction": 0.012,
            "oracle_label": False,
            "oracle_lift_vs_trajectory": 0.0,
            "peak_denoise_prompt_coverage": 0.25,
            "prompt_coverage": 0.875,
            "prompt_gap_count": 2,
            "repair_lift_vs_trajectory": 0.0,
            "source_task_delta_vs_trajectory": 0.0,
            "surface_selected": False,
            "task_id": "plan_100",
        },
        {
            "has_repairable_denoise_skeleton": True,
            "label": False,
            "label_pass_denoise_trigger": True,
            "measured_probe_value_prediction": 0.035,
            "oracle_label": False,
            "oracle_lift_vs_trajectory": 0.0,
            "peak_denoise_prompt_coverage": 0.059,
            "prompt_coverage": 0.765,
            "prompt_gap_count": 5,
            "repair_lift_vs_trajectory": 0.0,
            "source_task_delta_vs_trajectory": 0.0,
            "surface_selected": False,
            "task_id": "plan_101",
        },
        {
            "has_repairable_denoise_skeleton": True,
            "label": False,
            "label_pass_denoise_trigger": False,
            "measured_probe_value_prediction": 0.0,
            "oracle_label": False,
            "oracle_lift_vs_trajectory": 0.0,
            "peak_denoise_prompt_coverage": 0.5,
            "prompt_coverage": 1.0,
            "prompt_gap_count": 0,
            "repair_lift_vs_trajectory": 0.0,
            "source_task_delta_vs_trajectory": 0.0,
            "surface_selected": True,
            "task_id": "plan_102",
        },
        {
            "has_repairable_denoise_skeleton": True,
            "label": True,
            "label_pass_denoise_trigger": True,
            "measured_probe_value_prediction": 0.010,
            "oracle_label": True,
            "oracle_lift_vs_trajectory": 0.182,
            "peak_denoise_prompt_coverage": 0.333,
            "prompt_coverage": 0.533,
            "prompt_gap_count": 7,
            "repair_lift_vs_trajectory": 0.182,
            "source_task_delta_vs_trajectory": 0.0,
            "surface_selected": False,
            "task_id": "plan_104",
        },
    ]
