import json

from experiments.build_diffusion_generated_repair_value_hook_v19_freeze import (
    build_freeze_manifest,
    render_markdown,
)


def test_generated_repair_value_hook_v19_freeze_locks_fresh_live_hook_slice(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    scores = tmp_path / "scores.json"
    report = tmp_path / "report.md"
    labels = tmp_path / "labels.json"
    tasks.write_text("\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n", encoding="utf-8")
    scores.write_text(json.dumps(_scores_payload()), encoding="utf-8")
    report.write_text(_report_text(), encoding="utf-8")

    manifest = build_freeze_manifest(
        tasks_path=tasks,
        v18_hook_scores_path=scores,
        v18_hook_report_path=report,
        label_scores_path=labels,
    )
    markdown = render_markdown(manifest)

    assert manifest["task_preset"] == "lean_gpu_mixed_transfer_v19"
    assert manifest["planning_task_ids"] == [
        "plan_145",
        "plan_146",
        "plan_147",
        "plan_148",
        "plan_149",
        "plan_150",
        "plan_151",
        "plan_152",
    ]
    assert manifest["target_selector"]["selector_id"] == "generated_repair_value_v1"
    assert manifest["fit_boundary"]["v18_selected_generated_repair_tasks"] == ["plan_137", "plan_139"]
    assert manifest["fit_boundary"]["v18_rejected_no_lift_tasks"] == ["plan_141", "plan_144"]
    assert "--task-preset lean_gpu_mixed_transfer_v19" in manifest["fresh_slice_protocol"]["label_pass"]
    assert "--repair-selector generated_repair_value_v1" in manifest["fresh_slice_protocol"]["label_pass"]
    assert "No promotion language exists until the fresh validation passes" in markdown


def test_generated_repair_value_hook_v19_freeze_refuses_existing_labels(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    scores = tmp_path / "scores.json"
    report = tmp_path / "report.md"
    labels = tmp_path / "labels.json"
    tasks.write_text("\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n", encoding="utf-8")
    scores.write_text(json.dumps(_scores_payload()), encoding="utf-8")
    report.write_text(_report_text(), encoding="utf-8")
    labels.write_text("{}", encoding="utf-8")

    try:
        build_freeze_manifest(
            tasks_path=tasks,
            v18_hook_scores_path=scores,
            v18_hook_report_path=report,
            label_scores_path=labels,
        )
    except ValueError as exc:
        assert "labels exist" in str(exc)
    else:
        raise AssertionError("expected existing labels to block v19 freeze")


def _task_ids():
    return [
        "plan_145",
        "plan_146",
        "plan_147",
        "plan_148",
        "plan_149",
        "plan_150",
        "plan_151",
        "plan_152",
        "math_009",
        "sym_007",
        "sci_002",
    ]


def _scores_payload():
    return {
        "repair_selector": "generated_repair_value_v1",
        "run_id": "diffusion-d4a90959bf5734b2",
        "comparison_rows": [
            {
                "repair_selection_reason": "max_generated_repair_value_v1_score_repair_pool",
                "task_id": "plan_139",
            },
            {
                "repair_selection_reason": "max_generated_repair_value_v1_score_repair_pool",
                "task_id": "plan_137",
            },
        ],
    }


def _report_text():
    return "\n".join(
        [
            "Run ID: `diffusion-d4a90959bf5734b2`",
            "| plan_137 | True |",
            "| plan_139 | True |",
            "| plan_141 | False |",
            "| plan_144 | False |",
        ]
    )
