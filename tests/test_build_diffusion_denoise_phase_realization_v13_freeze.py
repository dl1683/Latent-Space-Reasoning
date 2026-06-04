import json

from experiments.build_diffusion_denoise_phase_realization_v13_freeze import (
    build_freeze_manifest,
    render_markdown,
)


def test_denoise_phase_realization_freeze_locks_fresh_v13_boundary(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    replay = tmp_path / "replay.json"
    label_scores = tmp_path / "label_scores.json"
    tasks.write_text(
        "\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n",
        encoding="utf-8",
    )
    replay.write_text(json.dumps(_replay_payload()), encoding="utf-8")
    label_scores.write_text(json.dumps(_label_scores_payload()), encoding="utf-8")

    manifest = build_freeze_manifest(tasks_path=tasks, v12_replay_path=replay, v12_label_scores_path=label_scores)
    markdown = render_markdown(manifest)

    assert manifest["task_preset"] == "lean_gpu_mixed_transfer_v13"
    assert manifest["planning_task_ids"] == [
        "plan_097",
        "plan_098",
        "plan_099",
        "plan_100",
        "plan_101",
        "plan_102",
        "plan_103",
        "plan_104",
    ]
    assert manifest["overlap_with_v12_replay_rows"] == []
    assert manifest["target_surface"]["requires_repairable_denoise_skeleton"] is True
    assert manifest["target_surface"]["first_repairable_denoise_skeleton_step_fraction_max"] == 0.40
    assert manifest["fit_boundary"]["named_counterexamples"]["static_surface_false_negative"]["task_id"] == "plan_093"
    assert "--task-preset lean_gpu_mixed_transfer_v13" in manifest["fresh_slice_protocol"]["measurement_pass"]
    assert "--repair-spend-trigger denoise_phase_repairability" in manifest["fresh_slice_protocol"]["label_pass"]
    assert manifest["replay_gates"]["reject_if_skeleton_only_matches_combined_surface"] is True
    assert "not a promoted controller" in markdown
    assert "Skeleton-only controls must not match" in markdown


def test_denoise_phase_realization_freeze_rejects_v12_overlap(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    replay = tmp_path / "replay.json"
    label_scores = tmp_path / "label_scores.json"
    task_ids = list(_task_ids())
    task_ids[0] = "plan_091"
    tasks.write_text(
        "\n".join(json.dumps({"task_id": task_id}) for task_id in task_ids) + "\n",
        encoding="utf-8",
    )
    replay.write_text(json.dumps(_replay_payload()), encoding="utf-8")
    label_scores.write_text(json.dumps(_label_scores_payload()), encoding="utf-8")

    try:
        build_freeze_manifest(tasks_path=tasks, v12_replay_path=replay, v12_label_scores_path=label_scores)
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected missing frozen v13 task to fail")


def test_denoise_phase_realization_freeze_rejects_missing_plan093_skeleton(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    replay = tmp_path / "replay.json"
    label_scores = tmp_path / "label_scores.json"
    tasks.write_text(
        "\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n",
        encoding="utf-8",
    )
    replay.write_text(json.dumps(_replay_payload()), encoding="utf-8")
    payload = _label_scores_payload()
    for row in payload["repair_spend_gate_rows"]:
        if row["task_id"] == "plan_093":
            row["has_repairable_denoise_skeleton"] = False
    label_scores.write_text(json.dumps(payload), encoding="utf-8")

    try:
        build_freeze_manifest(tasks_path=tasks, v12_replay_path=replay, v12_label_scores_path=label_scores)
    except ValueError as exc:
        assert "repairable denoise skeleton" in str(exc)
    else:
        raise AssertionError("expected missing plan_093 skeleton to fail")


def _task_ids():
    return [
        "plan_097",
        "plan_098",
        "plan_099",
        "plan_100",
        "plan_101",
        "plan_102",
        "plan_103",
        "plan_104",
        "math_009",
        "sym_007",
        "sci_002",
    ]


def _replay_payload():
    return {
        "selected_repair_hypotheses": {
            "frozen_source_aware_surface": {
                "false_negative_task_ids": ["plan_093"],
                "false_positive_task_ids": ["plan_091"],
            }
        },
        "oracle_hypotheses": {
            "frozen_source_aware_surface": {
                "false_negative_task_ids": ["plan_093", "plan_094"],
                "false_positive_task_ids": ["plan_091"],
            }
        },
        "row_diagnostics": [
            {"task_id": "plan_091", "label": False, "oracle_label": False},
            {"task_id": "plan_092", "label": False, "oracle_label": False},
            {"task_id": "plan_093", "label": True, "oracle_label": True},
            {"task_id": "plan_094", "label": False, "oracle_label": True},
        ],
    }


def _label_scores_payload():
    return {
        "repair_spend_gate_rows": [
            {
                "task_id": "plan_091",
                "source_task_delta_vs_trajectory": 0.0,
                "has_repairable_denoise_skeleton": True,
                "first_repairable_denoise_skeleton_step_fraction": 0.25,
                "peak_denoise_prompt_coverage": 0.4666666666666667,
            },
            {
                "task_id": "plan_092",
                "source_task_delta_vs_trajectory": 0.0,
                "has_repairable_denoise_skeleton": True,
                "first_repairable_denoise_skeleton_step_fraction": 0.3125,
                "peak_denoise_prompt_coverage": 0.4,
            },
            {
                "task_id": "plan_093",
                "source_task_delta_vs_trajectory": 0.0,
                "has_repairable_denoise_skeleton": True,
                "first_repairable_denoise_skeleton_step_fraction": 0.375,
                "peak_denoise_prompt_coverage": 0.45454545454545453,
            },
            {
                "task_id": "plan_094",
                "source_task_delta_vs_trajectory": 0.0,
                "has_repairable_denoise_skeleton": False,
                "first_repairable_denoise_skeleton_step_fraction": None,
                "peak_denoise_prompt_coverage": 0.25,
            },
        ]
    }
