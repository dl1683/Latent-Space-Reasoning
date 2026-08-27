import json

from experiments.build_diffusion_source_preservation_v17_freeze import build_freeze_manifest, render_markdown


def test_source_preservation_v17_freeze_locks_source_degradation_slice(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    targets = tmp_path / "targets.json"
    scores = tmp_path / "scores.json"
    labels = tmp_path / "labels.json"
    tasks.write_text("\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n", encoding="utf-8")
    targets.write_text(json.dumps(_targets_payload()), encoding="utf-8")
    scores.write_text(json.dumps(_scores_payload()), encoding="utf-8")

    manifest = build_freeze_manifest(
        tasks_path=tasks,
        v16_targets_path=targets,
        v16_scores_path=scores,
        label_scores_path=labels,
    )
    markdown = render_markdown(manifest)

    assert manifest["task_preset"] == "lean_gpu_mixed_transfer_v17"
    assert manifest["planning_task_ids"] == [
        "plan_129",
        "plan_130",
        "plan_131",
        "plan_132",
        "plan_133",
        "plan_134",
        "plan_135",
        "plan_136",
    ]
    assert manifest["target_surface"]["surface_id"] == "source_positive_repair_degradation_v17"
    assert manifest["fit_boundary"]["named_counterexample_task_id"] == "plan_128"
    assert "--task-preset lean_gpu_mixed_transfer_v17" in manifest["fresh_slice_protocol"]["label_pass"]
    assert "candidate_lift_vs_source < 0" in markdown


def test_source_preservation_v17_freeze_refuses_existing_labels(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    targets = tmp_path / "targets.json"
    scores = tmp_path / "scores.json"
    labels = tmp_path / "labels.json"
    tasks.write_text("\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n", encoding="utf-8")
    targets.write_text(json.dumps(_targets_payload()), encoding="utf-8")
    scores.write_text(json.dumps(_scores_payload()), encoding="utf-8")
    labels.write_text("{}", encoding="utf-8")

    try:
        build_freeze_manifest(
            tasks_path=tasks,
            v16_targets_path=targets,
            v16_scores_path=scores,
            label_scores_path=labels,
        )
    except ValueError as exc:
        assert "labels exist" in str(exc)
    else:
        raise AssertionError("expected existing labels to block v17 freeze")


def _task_ids():
    return [
        "plan_129",
        "plan_130",
        "plan_131",
        "plan_132",
        "plan_133",
        "plan_134",
        "plan_135",
        "plan_136",
        "math_009",
        "sym_007",
        "sci_002",
    ]


def _targets_payload():
    return {
        "rows": [
            {
                "candidate_lift_vs_source": -0.04,
                "candidate_lift_vs_trajectory": -0.02,
                "task_id": "plan_128",
            },
        ],
        "summary": {
            "positive_count": 0,
            "target_count": 1,
        },
    }


def _scores_payload():
    return {
        "comparison_rows": [
            {
                "oracle_delta_vs_trajectory": 0.02,
                "task_id": "plan_128",
            },
        ],
        "repair_spend_gate_rows": [
            {
                "should_run": True,
                "source_control": "random_32",
                "source_task_delta_vs_trajectory": 0.02,
                "task_id": "plan_128",
            },
        ],
    }
