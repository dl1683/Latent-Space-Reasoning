import json

from experiments.build_diffusion_source_aware_lift_direction_v12_freeze import (
    build_freeze_manifest,
    render_markdown,
)


def test_source_aware_lift_direction_freeze_locks_fresh_v12_surface(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    counterexamples = tmp_path / "counterexamples.json"
    tasks.write_text(
        "\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n",
        encoding="utf-8",
    )
    counterexamples.write_text(json.dumps(_counterexample_payload()), encoding="utf-8")

    manifest = build_freeze_manifest(tasks_path=tasks, counterexamples_path=counterexamples)
    markdown = render_markdown(manifest)

    assert manifest["task_preset"] == "lean_gpu_mixed_transfer_v12"
    assert manifest["planning_task_ids"] == [
        "plan_089",
        "plan_090",
        "plan_091",
        "plan_092",
        "plan_093",
        "plan_094",
        "plan_095",
        "plan_096",
    ]
    assert manifest["overlap_with_v11_counterexample_rows"] == []
    assert manifest["target_surface"]["source_task_delta_vs_trajectory_min"] == 0.0
    assert manifest["target_surface"]["prompt_gap_count_max"] == 4.0
    assert manifest["target_surface"]["prompt_coverage_min"] == 0.7
    assert manifest["target_surface"]["probe_value_feature_role"] == (
        "recorded_for_diagnostics_not_positive_direction_threshold"
    )
    assert "--task-preset lean_gpu_mixed_transfer_v12" in manifest["fresh_slice_protocol"]["measurement_pass"]
    assert "--repair-source-policy random" in manifest["fresh_slice_protocol"]["label_pass"]
    assert manifest["replay_gates"]["no_live_spend_trigger_without_runner_implementation"] is True
    assert "not a promoted controller" in markdown
    assert "No live spend trigger exists" in markdown


def test_source_aware_lift_direction_freeze_rejects_v11_overlap(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    counterexamples = tmp_path / "counterexamples.json"
    task_ids = list(_task_ids())
    task_ids[0] = "plan_081"
    tasks.write_text(
        "\n".join(json.dumps({"task_id": task_id}) for task_id in task_ids) + "\n",
        encoding="utf-8",
    )
    counterexamples.write_text(json.dumps(_counterexample_payload()), encoding="utf-8")

    try:
        build_freeze_manifest(tasks_path=tasks, counterexamples_path=counterexamples)
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected missing frozen v12 task to fail")


def test_source_aware_lift_direction_freeze_rejects_counterexample_overlap(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    counterexamples = tmp_path / "counterexamples.json"
    tasks.write_text(
        "\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n",
        encoding="utf-8",
    )
    payload = _counterexample_payload()
    payload["counterexample_rows"].append({"task_id": "plan_089"})
    counterexamples.write_text(json.dumps(payload), encoding="utf-8")

    try:
        build_freeze_manifest(tasks_path=tasks, counterexamples_path=counterexamples)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("expected v11/v12 overlap to fail")


def _task_ids():
    return [
        "plan_089",
        "plan_090",
        "plan_091",
        "plan_092",
        "plan_093",
        "plan_094",
        "plan_095",
        "plan_096",
        "math_009",
        "sym_007",
        "sci_002",
    ]


def _counterexample_payload():
    return {
        "counterexample_rows": [
            {"task_id": "plan_081"},
            {"task_id": "plan_082"},
            {"task_id": "plan_084"},
            {"task_id": "plan_086"},
            {"task_id": "plan_088"},
        ],
        "selected_repair_hypotheses": [
            {
                "error_count": 1,
                "false_negative_count": 0,
                "false_positive_count": 1,
                "hypothesis_id": "source_nonnegative_gap_le_4_coverage_ge_0p7",
            },
            {
                "error_count": 4,
                "false_negative_count": 1,
                "false_positive_count": 3,
                "hypothesis_id": "frozen_probe_value_floor",
            },
        ],
        "oracle_hypotheses": [
            {
                "error_count": 0,
                "false_negative_count": 0,
                "false_positive_count": 0,
                "hypothesis_id": "source_nonnegative_gap_le_4_coverage_ge_0p7",
            }
        ],
    }
