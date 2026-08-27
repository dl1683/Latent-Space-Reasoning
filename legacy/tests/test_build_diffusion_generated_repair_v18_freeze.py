import json

from experiments.build_diffusion_generated_repair_v18_freeze import build_freeze_manifest, render_markdown


def test_generated_repair_v18_freeze_locks_fresh_generated_repair_slice(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    targets = tmp_path / "targets.json"
    report = tmp_path / "report.md"
    labels = tmp_path / "labels.json"
    tasks.write_text("\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n", encoding="utf-8")
    targets.write_text(json.dumps(_targets_payload()), encoding="utf-8")
    report.write_text("# report\n", encoding="utf-8")

    manifest = build_freeze_manifest(
        tasks_path=tasks,
        v17_targets_path=targets,
        v17_report_path=report,
        label_scores_path=labels,
    )
    markdown = render_markdown(manifest)

    assert manifest["task_preset"] == "lean_gpu_mixed_transfer_v18"
    assert manifest["planning_task_ids"] == [
        "plan_137",
        "plan_138",
        "plan_139",
        "plan_140",
        "plan_141",
        "plan_142",
        "plan_143",
        "plan_144",
    ]
    assert manifest["target_surface"]["surface_id"] == "generated_repair_value_v18"
    assert manifest["fit_boundary"]["v17_generated_repair_positive_tasks"] == [
        "plan_129",
        "plan_130",
        "plan_131",
    ]
    assert "--task-preset lean_gpu_mixed_transfer_v18" in manifest["fresh_slice_protocol"]["label_pass"]
    assert "candidate_lift_vs_trajectory > 0.000000" in markdown


def test_generated_repair_v18_freeze_refuses_existing_labels(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    targets = tmp_path / "targets.json"
    report = tmp_path / "report.md"
    labels = tmp_path / "labels.json"
    tasks.write_text("\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n", encoding="utf-8")
    targets.write_text(json.dumps(_targets_payload()), encoding="utf-8")
    report.write_text("# report\n", encoding="utf-8")
    labels.write_text("{}", encoding="utf-8")

    try:
        build_freeze_manifest(
            tasks_path=tasks,
            v17_targets_path=targets,
            v17_report_path=report,
            label_scores_path=labels,
        )
    except ValueError as exc:
        assert "labels exist" in str(exc)
    else:
        raise AssertionError("expected existing labels to block v18 freeze")


def _task_ids():
    return [
        "plan_137",
        "plan_138",
        "plan_139",
        "plan_140",
        "plan_141",
        "plan_142",
        "plan_143",
        "plan_144",
        "math_009",
        "sym_007",
        "sci_002",
    ]


def _targets_payload():
    return {
        "rows": [
            {
                "candidate_lift_vs_trajectory": 0.0375,
                "generated_repair_positive": True,
                "prompt_coverage": 0.846,
                "prompt_gap_count": 2,
                "task_id": "plan_129",
            },
            {
                "candidate_lift_vs_trajectory": 0.05,
                "generated_repair_positive": True,
                "prompt_coverage": 0.8,
                "prompt_gap_count": 3,
                "task_id": "plan_130",
            },
            {
                "candidate_lift_vs_trajectory": 0.05,
                "generated_repair_positive": True,
                "prompt_coverage": 0.647,
                "prompt_gap_count": 6,
                "task_id": "plan_131",
            },
        ],
        "summary": {
            "generated_repair_positive_tasks": ["plan_129", "plan_130", "plan_131"],
            "source_positive_repair_degradation_count": 0,
        },
    }
