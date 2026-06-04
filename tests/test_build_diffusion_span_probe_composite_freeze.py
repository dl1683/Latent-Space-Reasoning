import json

from experiments.build_diffusion_span_probe_composite_freeze import build_freeze_manifest, render_markdown


def test_build_freeze_manifest_locks_v10_without_prior_span_overlap(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    gate = tmp_path / "gate.json"
    tasks.write_text(
        "\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n",
        encoding="utf-8",
    )
    gate.write_text(
        json.dumps(
            {
                "row_diagnostics": [{"task_id": "plan_017"}, {"task_id": "plan_024"}],
                "trajectory_relative_gate": {
                    "false_negative_count": 0,
                    "false_positive_count": 2,
                    "policy_utility": 0.8055,
                    "weak_slice_summary": {"false_positive_count": 0},
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_freeze_manifest(tasks_path=tasks, gate_json_path=gate)
    markdown = render_markdown(manifest)

    assert manifest["task_preset"] == "lean_gpu_mixed_transfer_v10"
    assert manifest["planning_task_ids"] == [
        "plan_073",
        "plan_074",
        "plan_075",
        "plan_076",
        "plan_077",
        "plan_078",
        "plan_079",
        "plan_080",
    ]
    assert manifest["overlap_with_prior_span_rows"] == []
    assert "counterfactual_micro_probe_v1" in manifest["fresh_slice_protocol"]["measurement_pass"]
    assert "denoise_phase_repairability" in manifest["fresh_slice_protocol"]["label_pass"]
    assert "not promoted as a live spend trigger" in markdown


def test_build_freeze_manifest_rejects_missing_frozen_tasks(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    gate = tmp_path / "gate.json"
    tasks.write_text(json.dumps({"task_id": "plan_073"}) + "\n", encoding="utf-8")
    gate.write_text(json.dumps({"row_diagnostics": []}), encoding="utf-8")

    try:
        build_freeze_manifest(tasks_path=tasks, gate_json_path=gate)
    except ValueError as exc:
        assert "plan_074" in str(exc)
    else:
        raise AssertionError("expected missing frozen task ids to fail")


def _task_ids():
    return [
        "plan_073",
        "plan_074",
        "plan_075",
        "plan_076",
        "plan_077",
        "plan_078",
        "plan_079",
        "plan_080",
        "math_009",
        "sym_007",
        "sci_002",
    ]
