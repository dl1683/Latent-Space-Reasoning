import json

from experiments.build_diffusion_promotion_margin_v16_freeze import build_freeze_manifest, render_markdown


def test_promotion_margin_v16_freeze_locks_low_margin_slice(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    targets = tmp_path / "targets.json"
    replay = tmp_path / "replay.json"
    labels = tmp_path / "labels.json"
    tasks.write_text("\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n", encoding="utf-8")
    targets.write_text(json.dumps(_targets_payload()), encoding="utf-8")
    replay.write_text(json.dumps(_replay_payload()), encoding="utf-8")

    manifest = build_freeze_manifest(
        tasks_path=tasks,
        v15_targets_path=targets,
        v15_replay_path=replay,
        label_scores_path=labels,
    )
    markdown = render_markdown(manifest)

    assert manifest["task_preset"] == "lean_gpu_mixed_transfer_v16"
    assert manifest["planning_task_ids"] == [
        "plan_121",
        "plan_122",
        "plan_123",
        "plan_124",
        "plan_125",
        "plan_126",
        "plan_127",
        "plan_128",
    ]
    assert manifest["target_surface"]["surface_id"] == "low_margin_candidate_realization_v16"
    assert "--task-preset lean_gpu_mixed_transfer_v16" in manifest["fresh_slice_protocol"]["label_pass"]
    assert "candidate_aware_promotion_v1" in markdown


def test_promotion_margin_v16_freeze_refuses_existing_labels(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    targets = tmp_path / "targets.json"
    replay = tmp_path / "replay.json"
    labels = tmp_path / "labels.json"
    tasks.write_text("\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n", encoding="utf-8")
    targets.write_text(json.dumps(_targets_payload()), encoding="utf-8")
    replay.write_text(json.dumps(_replay_payload()), encoding="utf-8")
    labels.write_text("{}", encoding="utf-8")

    try:
        build_freeze_manifest(
            tasks_path=tasks,
            v15_targets_path=targets,
            v15_replay_path=replay,
            label_scores_path=labels,
        )
    except ValueError as exc:
        assert "labels exist" in str(exc)
    else:
        raise AssertionError("expected existing labels to block v16 freeze")


def _task_ids():
    return [
        "plan_121",
        "plan_122",
        "plan_123",
        "plan_124",
        "plan_125",
        "plan_126",
        "plan_127",
        "plan_128",
        "math_009",
        "sym_007",
        "sci_002",
    ]


def _targets_payload():
    return {
        "rows": [
            {"task_id": "plan_118"},
            {"task_id": "plan_120"},
        ],
        "summary": {
            "candidate_aware_promotion_error_count": 1,
            "candidate_aware_selected_tasks": ["plan_120"],
            "positive_tasks": ["plan_118", "plan_120"],
        },
    }


def _replay_payload():
    return {
        "selected_repair_hypotheses": {
            "probe_conditioned_realization_value_v15": {
                "false_negative_task_ids": ["plan_120"],
            },
            "static_source_gap_coverage_v15": {
                "false_negative_count": 0,
            },
        },
    }
