import json

from experiments.build_diffusion_low_margin_repair_v20_freeze import (
    build_freeze_manifest,
    render_markdown,
)


def test_low_margin_repair_v20_freeze_locks_counterexample_slice(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    targets = tmp_path / "targets.json"
    audit = tmp_path / "audit.json"
    labels = tmp_path / "labels.json"
    tasks.write_text("\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n", encoding="utf-8")
    targets.write_text(json.dumps(_targets()), encoding="utf-8")
    audit.write_text(json.dumps({"summary": {"control_only_waste_tasks": ["plan_150", "plan_151"]}}), encoding="utf-8")

    manifest = build_freeze_manifest(
        tasks_path=tasks,
        v19_targets_path=targets,
        v19_cost_audit_path=audit,
        label_scores_path=labels,
    )
    markdown = render_markdown(manifest)

    assert manifest["task_preset"] == "lean_gpu_mixed_transfer_v20"
    assert manifest["planning_task_ids"] == [
        "plan_153",
        "plan_154",
        "plan_155",
        "plan_156",
        "plan_157",
        "plan_158",
        "plan_159",
        "plan_160",
    ]
    assert manifest["target_surface"]["surface_id"] == "low_margin_source_tie_repair_v20"
    assert manifest["fit_boundary"]["plan_151_candidate_lift_vs_trajectory"] == 0.007
    assert "--task-preset lean_gpu_mixed_transfer_v20" in manifest["fresh_slice_protocol"]["label_pass"]
    assert "--repair-selector planning_quality" in manifest["fresh_slice_protocol"]["label_pass"]
    assert "V20 tests whether that fallback geometry repeats" in markdown


def test_low_margin_repair_v20_freeze_refuses_existing_labels(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    targets = tmp_path / "targets.json"
    audit = tmp_path / "audit.json"
    labels = tmp_path / "labels.json"
    tasks.write_text("\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n", encoding="utf-8")
    targets.write_text(json.dumps(_targets()), encoding="utf-8")
    audit.write_text(json.dumps({"summary": {"control_only_waste_tasks": ["plan_150", "plan_151"]}}), encoding="utf-8")
    labels.write_text("{}", encoding="utf-8")

    try:
        build_freeze_manifest(
            tasks_path=tasks,
            v19_targets_path=targets,
            v19_cost_audit_path=audit,
            label_scores_path=labels,
        )
    except ValueError as exc:
        assert "labels exist" in str(exc)
    else:
        raise AssertionError("expected existing labels to block v20 freeze")


def _task_ids():
    return [
        "plan_153",
        "plan_154",
        "plan_155",
        "plan_156",
        "plan_157",
        "plan_158",
        "plan_159",
        "plan_160",
        "math_009",
        "sym_007",
        "sci_002",
    ]


def _targets():
    return {
        "rows": [
            {
                "candidate_lift_vs_trajectory": 0.0,
                "max_span_target_score": 3.8,
                "planning_quality_delta_vs_source": 0.0,
                "task_id": "plan_150",
            },
            {
                "candidate_lift_vs_trajectory": 0.007,
                "max_span_target_score": 0.0,
                "planning_quality_delta_vs_source": 0.0,
                "task_id": "plan_151",
            },
        ]
    }
