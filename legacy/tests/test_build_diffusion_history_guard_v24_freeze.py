import json

from experiments.build_diffusion_history_guard_v24_freeze import (
    build_freeze_manifest,
    render_markdown,
)


def test_history_guard_v24_freeze_locks_v23_selected_hook_boundary(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    result = tmp_path / "result.json"
    targets = tmp_path / "targets.json"
    labels = tmp_path / "labels.json"
    tasks.write_text("\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n", encoding="utf-8")
    result.write_text(json.dumps(_result()), encoding="utf-8")
    targets.write_text(json.dumps(_targets()), encoding="utf-8")

    manifest = build_freeze_manifest(
        tasks_path=tasks,
        v23_result_path=result,
        v23_targets_path=targets,
        label_scores_path=labels,
    )
    markdown = render_markdown(manifest)

    assert manifest["task_preset"] == "lean_gpu_mixed_transfer_v24"
    assert manifest["audit_surface"]["audit_id"] == "history_prefix_guard_audit_v24"
    assert manifest["audit_surface"]["final_preserve_filtering_allowed"] is False
    assert manifest["fit_boundary"]["v23_unchanged_selected_positive_count"] == 6
    assert "--task-preset lean_gpu_mixed_transfer_v24" in manifest["fresh_slice_protocol"]["label_pass"]
    assert "unchanged hook as the baseline" in markdown


def test_history_guard_v24_freeze_refuses_existing_labels(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    result = tmp_path / "result.json"
    targets = tmp_path / "targets.json"
    labels = tmp_path / "labels.json"
    tasks.write_text("\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n", encoding="utf-8")
    result.write_text(json.dumps(_result()), encoding="utf-8")
    targets.write_text(json.dumps(_targets()), encoding="utf-8")
    labels.write_text("{}", encoding="utf-8")

    try:
        build_freeze_manifest(
            tasks_path=tasks,
            v23_result_path=result,
            v23_targets_path=targets,
            label_scores_path=labels,
        )
    except ValueError as exc:
        assert "labels exist" in str(exc)
    else:
        raise AssertionError("expected existing labels to block v24 freeze")


def _task_ids():
    return [
        "plan_185",
        "plan_186",
        "plan_187",
        "plan_188",
        "plan_189",
        "plan_190",
        "plan_191",
        "plan_192",
        "math_009",
        "sym_007",
        "sci_002",
    ]


def _result():
    return {
        "decision": {"status": "precision_positive_utility_failed"},
        "summary": {
            "asymmetric_selected_positive_count": 3,
            "asymmetric_selected_waste_count": 0,
            "positive_count": 7,
            "positive_tasks": ["plan_177", "plan_178", "plan_179", "plan_181", "plan_182", "plan_183"],
            "unchanged_selected_positive_count": 6,
            "unchanged_selected_waste_count": 0,
        },
    }


def _targets():
    return {
        "rows": [
            {
                "candidate_lift_vs_trajectory": 0.02,
                "repair": "history_prefix_25_repair",
                "task_id": "plan_177",
            },
            {
                "candidate_lift_vs_trajectory": -0.01,
                "repair": "history_prefix_25_repair",
                "task_id": "plan_182",
            },
        ]
    }
