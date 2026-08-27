import json

import experiments.build_diffusion_learned_selector_v25_freeze as freeze


def test_learned_selector_v25_freeze_locks_training_packet_and_features(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks.jsonl"
    labels = tmp_path / "labels.json"
    target_paths = {}
    result_paths = {}
    for slice_id in ("v21", "v22", "v23", "v24"):
        target = tmp_path / f"{slice_id}_targets.json"
        result = tmp_path / f"{slice_id}_result.json"
        target.write_text(json.dumps(_targets(slice_id)), encoding="utf-8")
        result.write_text(json.dumps(_result(slice_id)), encoding="utf-8")
        target_paths[slice_id] = target
        result_paths[slice_id] = result
    monkeypatch.setattr(freeze, "TRAINING_TARGETS", target_paths)
    monkeypatch.setattr(freeze, "TRAINING_RESULTS", result_paths)
    tasks.write_text("\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n", encoding="utf-8")

    manifest = freeze.build_freeze_manifest(tasks_path=tasks, label_scores_path=labels)
    markdown = freeze.render_markdown(manifest)

    assert manifest["task_preset"] == "lean_gpu_mixed_transfer_v25"
    assert manifest["training_packet"]["row_count"] == 8
    assert manifest["training_packet"]["positive_count"] == 4
    assert "candidate_lift_vs_trajectory" in manifest["learned_selector_protocol"]["forbidden_features"]
    assert "planning_quality_delta_vs_source" in manifest["learned_selector_protocol"]["label_free_features"]
    assert "--task-preset lean_gpu_mixed_transfer_v25" in manifest["fresh_slice_protocol"]["label_pass"]
    assert "held-out learned-selector proof obligation" in markdown


def test_learned_selector_v25_freeze_refuses_existing_labels(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks.jsonl"
    labels = tmp_path / "labels.json"
    monkeypatch.setattr(freeze, "TRAINING_TARGETS", {})
    monkeypatch.setattr(freeze, "TRAINING_RESULTS", {})
    tasks.write_text("\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n", encoding="utf-8")
    labels.write_text("{}", encoding="utf-8")

    try:
        freeze.build_freeze_manifest(tasks_path=tasks, label_scores_path=labels)
    except ValueError as exc:
        assert "labels exist" in str(exc)
    else:
        raise AssertionError("expected existing labels to block v25 freeze")


def _task_ids():
    return [
        "plan_193",
        "plan_194",
        "plan_195",
        "plan_196",
        "plan_197",
        "plan_198",
        "plan_199",
        "plan_200",
        "math_009",
        "sym_007",
        "sci_002",
    ]


def _targets(slice_id):
    return {
        "rows": [
            {
                "candidate_lift_vs_trajectory": 0.02,
                "repair": "history_prefix_25_repair",
                "task_id": f"{slice_id}_a",
            },
            {
                "candidate_lift_vs_trajectory": -0.01,
                "repair": "constraint_gap_span_phase_final_preserve_seeded_gated_repair",
                "task_id": f"{slice_id}_b",
            },
        ]
    }


def _result(slice_id):
    return {
        "decision": {"status": "fixture"},
        "summary": {"run_id": f"run-{slice_id}"},
    }
