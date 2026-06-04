import json

from experiments.build_diffusion_source_aware_selector_v22_freeze import (
    build_freeze_manifest,
    render_markdown,
)


def test_source_aware_selector_v22_freeze_locks_v21_waste_boundary(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    result = tmp_path / "result.json"
    targets = tmp_path / "targets.json"
    labels = tmp_path / "labels.json"
    tasks.write_text("\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n", encoding="utf-8")
    result.write_text(json.dumps(_result()), encoding="utf-8")
    targets.write_text(json.dumps({"rows": [{"task_id": "plan_161"}]}), encoding="utf-8")

    manifest = build_freeze_manifest(
        tasks_path=tasks,
        v21_result_path=result,
        v21_targets_path=targets,
        label_scores_path=labels,
    )
    markdown = render_markdown(manifest)

    assert manifest["task_preset"] == "lean_gpu_mixed_transfer_v22"
    assert manifest["target_surface"]["surface_id"] == "source_aware_candidate_selector_v22"
    assert manifest["target_surface"]["history_prefix_planning_delta_min"] == 0.20
    assert manifest["fit_boundary"]["v21_selected_waste_tasks"] == ["plan_161", "plan_162"]
    assert "--task-preset lean_gpu_mixed_transfer_v22" in manifest["fresh_slice_protocol"]["label_pass"]
    assert "source-specific replay surface" in markdown


def test_source_aware_selector_v22_freeze_refuses_existing_labels(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    result = tmp_path / "result.json"
    targets = tmp_path / "targets.json"
    labels = tmp_path / "labels.json"
    tasks.write_text("\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n", encoding="utf-8")
    result.write_text(json.dumps(_result()), encoding="utf-8")
    targets.write_text(json.dumps({"rows": []}), encoding="utf-8")
    labels.write_text("{}", encoding="utf-8")

    try:
        build_freeze_manifest(
            tasks_path=tasks,
            v21_result_path=result,
            v21_targets_path=targets,
            label_scores_path=labels,
        )
    except ValueError as exc:
        assert "labels exist" in str(exc)
    else:
        raise AssertionError("expected existing labels to block v22 freeze")


def _task_ids():
    return [
        "plan_169",
        "plan_170",
        "plan_171",
        "plan_172",
        "plan_173",
        "plan_174",
        "plan_175",
        "plan_176",
        "math_009",
        "sym_007",
        "sci_002",
    ]


def _result():
    return {
        "decision": {"status": "availability_positive_selector_failed"},
        "summary": {
            "positive_count": 4,
            "repair_task_delta_vs_evolved": 0.03,
            "selected_positive_tasks": ["plan_164", "plan_167", "plan_168"],
            "selected_waste_tasks": ["plan_161", "plan_162"],
        },
    }
