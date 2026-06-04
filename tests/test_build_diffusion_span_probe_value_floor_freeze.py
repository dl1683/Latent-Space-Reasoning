import json

from experiments.build_diffusion_span_probe_value_floor_freeze import (
    build_freeze_manifest,
    render_markdown,
)


def test_value_floor_freeze_manifest_locks_v11_without_fit_overlap(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    specificity = tmp_path / "specificity.json"
    tasks.write_text(
        "\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n",
        encoding="utf-8",
    )
    specificity.write_text(
        json.dumps(
            {
                "row_diagnostics": [
                    {"task_id": "plan_073", "measured_probe_value_prediction": 0.030584403876493135},
                    {"task_id": "plan_074", "measured_probe_value_prediction": 0.02891517987715706},
                    {"task_id": "plan_075", "measured_probe_value_prediction": 0.027305529300567108},
                ],
                "selected_rule": {
                    "rule_id": "measured_probe_value_prediction_ge_0p028915",
                    "policy_utility": 0.4882142857142856,
                    "false_positive_count": 0,
                    "false_negative_count": 0,
                },
                "selection_penalty": 0.02,
                "summary": {"positive_count": 5},
            }
        ),
        encoding="utf-8",
    )

    manifest = build_freeze_manifest(tasks_path=tasks, specificity_json_path=specificity)
    markdown = render_markdown(manifest)

    assert manifest["task_preset"] == "lean_gpu_mixed_transfer_v11"
    assert manifest["planning_task_ids"] == [
        "plan_081",
        "plan_082",
        "plan_083",
        "plan_084",
        "plan_085",
        "plan_086",
        "plan_087",
        "plan_088",
    ]
    assert manifest["overlap_with_v10_fit_rows"] == []
    assert manifest["controller"]["threshold"] == 0.02891517987715706
    assert manifest["controller"]["source_policy"] == "fixed"
    assert "lean_gpu_mixed_transfer_v11" in manifest["fresh_slice_protocol"]["measurement_pass"]
    assert "--repair-source-policy fixed" in manifest["fresh_slice_protocol"]["label_pass"]
    assert "not a live spend trigger" in markdown


def test_value_floor_freeze_manifest_rejects_missing_v11_tasks(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    specificity = tmp_path / "specificity.json"
    tasks.write_text(json.dumps({"task_id": "plan_081"}) + "\n", encoding="utf-8")
    specificity.write_text(json.dumps({"selected_rule": {"rule_id": "measured_probe_value_prediction_ge_0p028915"}}))

    try:
        build_freeze_manifest(tasks_path=tasks, specificity_json_path=specificity)
    except ValueError as exc:
        assert "plan_082" in str(exc)
    else:
        raise AssertionError("expected missing frozen task ids to fail")


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
