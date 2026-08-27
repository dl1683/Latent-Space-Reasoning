import json

from experiments.build_diffusion_realization_value_v15_freeze import build_freeze_manifest, render_markdown


def test_realization_value_v15_freeze_locks_static_probe_disagreement_slice(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    replay = tmp_path / "replay.json"
    measurement = tmp_path / "measurement.json"
    labels = tmp_path / "labels.json"
    tasks.write_text(
        "\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n",
        encoding="utf-8",
    )
    replay.write_text(json.dumps(_replay_payload()), encoding="utf-8")

    manifest = build_freeze_manifest(
        tasks_path=tasks,
        v14b_replay_path=replay,
        measurement_scores_path=measurement,
        label_scores_path=labels,
    )
    markdown = render_markdown(manifest)

    assert manifest["task_preset"] == "lean_gpu_mixed_transfer_v15"
    assert manifest["planning_task_ids"] == [
        "plan_113",
        "plan_114",
        "plan_115",
        "plan_116",
        "plan_117",
        "plan_118",
        "plan_119",
        "plan_120",
    ]
    assert [surface["surface_id"] for surface in manifest["target_surfaces"]] == [
        "static_source_gap_coverage_v15",
        "probe_conditioned_realization_value_v15",
    ]
    assert manifest["conclusive_result_gates"]["minimum_static_probe_disagreement_count"] == 1
    assert "--task-preset lean_gpu_mixed_transfer_v15" in manifest["fresh_slice_protocol"]["measurement_pass"]
    assert "static-vs-probe disagreement slice" in markdown


def test_realization_value_v15_freeze_refuses_existing_measurement_or_labels(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    replay = tmp_path / "replay.json"
    measurement = tmp_path / "measurement.json"
    labels = tmp_path / "labels.json"
    tasks.write_text(
        "\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n",
        encoding="utf-8",
    )
    replay.write_text(json.dumps(_replay_payload()), encoding="utf-8")
    measurement.write_text("{}", encoding="utf-8")

    try:
        build_freeze_manifest(
            tasks_path=tasks,
            v14b_replay_path=replay,
            measurement_scores_path=measurement,
            label_scores_path=labels,
        )
    except ValueError as exc:
        assert "measurement exists" in str(exc)
    else:
        raise AssertionError("expected existing measurement to block v15 freeze")


def _task_ids():
    return [
        "plan_113",
        "plan_114",
        "plan_115",
        "plan_116",
        "plan_117",
        "plan_118",
        "plan_119",
        "plan_120",
        "math_009",
        "sym_007",
        "sci_002",
    ]


def _replay_payload():
    return {
        "row_diagnostics": [
            {"task_id": "plan_109"},
            {"task_id": "plan_112"},
        ],
        "selected_repair_hypotheses": {
            "realization_value_probe_banded_v14b": {
                "false_negative_count": 0,
                "false_positive_count": 0,
                "selected_task_ids": ["plan_109", "plan_112"],
            },
            "static_source_gap_coverage_control": {
                "false_negative_count": 0,
                "false_positive_count": 0,
                "selected_task_ids": ["plan_109", "plan_112"],
            },
        },
    }
