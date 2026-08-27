import json

from experiments.build_diffusion_span_probe_value_floor_random_source_freeze import (
    build_random_source_freeze_manifest,
    render_markdown,
)


def test_random_source_freeze_manifest_locks_v11_random_source_stress(tmp_path):
    freeze = tmp_path / "freeze.json"
    measurement = tmp_path / "measurement.md"
    freeze.write_text(
        json.dumps(
            {
                "task_ids": _task_ids(),
                "controller": {
                    "feature": "measured_probe_value_prediction",
                    "operator": "ge",
                    "threshold": 0.02891517987715706,
                    "rule_id": "measured_probe_value_prediction_ge_0p028915",
                    "probe_policy": "span_tomography_probe_v4",
                    "source_policy": "fixed",
                },
            }
        ),
        encoding="utf-8",
    )
    measurement.write_text("all rows had zero source_task_delta_vs_trajectory", encoding="utf-8")

    manifest = build_random_source_freeze_manifest(
        value_floor_freeze_json_path=freeze,
        fixed_source_measurement_path=measurement,
    )
    markdown = render_markdown(manifest)

    assert manifest["task_preset"] == "lean_gpu_mixed_transfer_v11"
    assert manifest["source_policy"] == "random"
    assert manifest["controller"]["threshold"] == 0.02891517987715706
    assert "--repair-source-policy random" in manifest["fresh_slice_protocol"]["measurement_pass"]
    assert "--repair-source-policy random" in manifest["fresh_slice_protocol"]["label_pass"]
    assert manifest["prior_boundary"]["reason"] == (
        "fixed_source_measurement_had_zero_source_task_delta_on_all_planning_rows"
    )
    assert "frozen_random_source_stress_not_live_spend_trigger" in markdown


def test_random_source_freeze_manifest_rejects_non_fixed_prior_freeze(tmp_path):
    freeze = tmp_path / "freeze.json"
    measurement = tmp_path / "measurement.md"
    freeze.write_text(
        json.dumps({"task_ids": _task_ids(), "controller": {"source_policy": "random"}}),
        encoding="utf-8",
    )
    measurement.write_text("boundary", encoding="utf-8")

    try:
        build_random_source_freeze_manifest(
            value_floor_freeze_json_path=freeze,
            fixed_source_measurement_path=measurement,
        )
    except ValueError as exc:
        assert "fixed source policy" in str(exc)
    else:
        raise AssertionError("expected non-fixed prior freeze to fail")


def _task_ids():
    return [
        "plan_081",
        "plan_082",
        "plan_083",
        "plan_084",
        "plan_085",
        "plan_086",
        "plan_087",
        "plan_088",
        "math_009",
        "sym_007",
        "sci_002",
    ]
