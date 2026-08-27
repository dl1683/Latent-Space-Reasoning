import json

from experiments.build_diffusion_asymmetric_filter_v23_freeze import (
    build_freeze_manifest,
    render_markdown,
)


def test_asymmetric_filter_v23_freeze_locks_v22_overfilter_boundary(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    result = tmp_path / "result.json"
    targets = tmp_path / "targets.json"
    labels = tmp_path / "labels.json"
    tasks.write_text("\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n", encoding="utf-8")
    result.write_text(json.dumps(_result()), encoding="utf-8")
    targets.write_text(json.dumps({"rows": [{"task_id": "plan_169"}]}), encoding="utf-8")

    manifest = build_freeze_manifest(
        tasks_path=tasks,
        v22_result_path=result,
        v22_targets_path=targets,
        label_scores_path=labels,
    )
    markdown = render_markdown(manifest)

    assert manifest["task_preset"] == "lean_gpu_mixed_transfer_v23"
    assert manifest["target_surface"]["surface_id"] == "asymmetric_repair_source_filter_v23"
    assert manifest["target_surface"]["final_preserve_planning_delta_min"] == 0.005
    assert manifest["fit_boundary"]["v22_status"] == "precision_positive_utility_failed"
    assert "--task-preset lean_gpu_mixed_transfer_v23" in manifest["fresh_slice_protocol"]["label_pass"]
    assert "relaxes final-preserve recall" in markdown


def test_asymmetric_filter_v23_freeze_refuses_existing_labels(tmp_path):
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
            v22_result_path=result,
            v22_targets_path=targets,
            label_scores_path=labels,
        )
    except ValueError as exc:
        assert "labels exist" in str(exc)
    else:
        raise AssertionError("expected existing labels to block v23 freeze")


def _task_ids():
    return [
        "plan_177",
        "plan_178",
        "plan_179",
        "plan_180",
        "plan_181",
        "plan_182",
        "plan_183",
        "plan_184",
        "math_009",
        "sym_007",
        "sci_002",
    ]


def _result():
    return {
        "decision": {"status": "precision_positive_utility_failed"},
        "summary": {
            "positive_tasks": ["plan_169", "plan_170", "plan_175"],
            "source_aware_selected_positive_count": 1,
            "source_aware_selected_waste_count": 0,
            "unchanged_selected_waste_count": 1,
        },
    }
