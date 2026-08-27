import json

from experiments.build_diffusion_realization_value_v14_freeze import build_freeze_manifest, render_markdown


def test_realization_value_v14_freeze_locks_fresh_slice(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    counterexamples = tmp_path / "counterexamples.json"
    tasks.write_text(
        "\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n",
        encoding="utf-8",
    )
    counterexamples.write_text(json.dumps(_counterexample_payload()), encoding="utf-8")

    manifest = build_freeze_manifest(tasks_path=tasks, counterexamples_path=counterexamples)
    markdown = render_markdown(manifest)

    assert manifest["task_preset"] == "lean_gpu_mixed_transfer_v14"
    assert manifest["planning_task_ids"] == [
        "plan_105",
        "plan_106",
        "plan_107",
        "plan_108",
        "plan_109",
        "plan_110",
        "plan_111",
        "plan_112",
    ]
    assert manifest["target_surface"]["prompt_gap_count_min"] == 4.0
    assert manifest["target_surface"]["prompt_gap_count_max"] == 7.0
    assert manifest["target_surface"]["measured_probe_value_prediction_max"] == 0.032
    assert "--task-preset lean_gpu_mixed_transfer_v14" in manifest["fresh_slice_protocol"]["measurement_pass"]
    assert "--repair-spend-trigger denoise_phase_repairability" in manifest["fresh_slice_protocol"]["label_pass"]
    assert "diagnostic-only" in markdown


def test_realization_value_v14_freeze_rejects_nonzero_fit_errors(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    counterexamples = tmp_path / "counterexamples.json"
    tasks.write_text(
        "\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n",
        encoding="utf-8",
    )
    payload = _counterexample_payload()
    payload["selected_repair_hypotheses"][0]["error_count"] = 1
    counterexamples.write_text(json.dumps(payload), encoding="utf-8")

    try:
        build_freeze_manifest(tasks_path=tasks, counterexamples_path=counterexamples)
    except ValueError as exc:
        assert "zero-error" in str(exc)
    else:
        raise AssertionError("expected nonzero diagnostic fit errors to fail")


def _task_ids():
    return [
        "plan_105",
        "plan_106",
        "plan_107",
        "plan_108",
        "plan_109",
        "plan_110",
        "plan_111",
        "plan_112",
        "math_009",
        "sym_007",
        "sci_002",
    ]


def _counterexample_payload():
    return {
        "counterexample_rows": [
            {"task_id": "plan_099"},
            {"task_id": "plan_102"},
            {"task_id": "plan_104"},
        ],
        "selected_repair_hypotheses": [
            {
                "error_count": 0,
                "false_negative_count": 0,
                "false_positive_count": 0,
                "hypothesis_id": "label_trigger_source_nonnegative_gap_4_to_7_probe_le_0p032",
            },
            {
                "error_count": 2,
                "false_negative_count": 1,
                "false_positive_count": 1,
                "hypothesis_id": "frozen_denoise_realization_surface",
            },
            {
                "error_count": 3,
                "false_negative_count": 0,
                "false_positive_count": 3,
                "hypothesis_id": "label_pass_denoise_trigger",
            },
        ],
    }
