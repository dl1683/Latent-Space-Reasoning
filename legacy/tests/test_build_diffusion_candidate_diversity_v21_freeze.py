import json

from experiments.build_diffusion_candidate_diversity_v21_freeze import (
    build_freeze_manifest,
    render_markdown,
)


def test_candidate_diversity_v21_freeze_locks_two_candidate_slice(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    targets = tmp_path / "targets.json"
    scores = tmp_path / "scores.json"
    labels = tmp_path / "labels.json"
    tasks.write_text("\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n", encoding="utf-8")
    targets.write_text(json.dumps(_targets()), encoding="utf-8")
    scores.write_text(json.dumps({"summary": {"repair_delta_vs_evolved": 0.0}}), encoding="utf-8")

    manifest = build_freeze_manifest(
        tasks_path=tasks,
        v20_targets_path=targets,
        v20_scores_path=scores,
        label_scores_path=labels,
    )
    markdown = render_markdown(manifest)

    assert manifest["task_preset"] == "lean_gpu_mixed_transfer_v21"
    assert manifest["planning_task_ids"] == [
        "plan_161",
        "plan_162",
        "plan_163",
        "plan_164",
        "plan_165",
        "plan_166",
        "plan_167",
        "plan_168",
    ]
    assert manifest["fit_boundary"]["v20_positive_count"] == 0
    assert manifest["candidate_pool"]["candidate_names"] == [
        "history_prefix_25_repair",
        "constraint_gap_span_phase_final_preserve_seeded_gated_repair",
    ]
    assert "--task-preset lean_gpu_mixed_transfer_v21" in manifest["fresh_slice_protocol"]["label_pass"]
    assert "--include-history-repairs" in manifest["fresh_slice_protocol"]["label_pass"]
    assert "--repair-selector generated_repair_value_v1" in manifest["fresh_slice_protocol"]["label_pass"]
    assert "v21 moves upstream" in markdown


def test_candidate_diversity_v21_freeze_refuses_existing_labels(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    targets = tmp_path / "targets.json"
    scores = tmp_path / "scores.json"
    labels = tmp_path / "labels.json"
    tasks.write_text("\n".join(json.dumps({"task_id": task_id}) for task_id in _task_ids()) + "\n", encoding="utf-8")
    targets.write_text(json.dumps(_targets()), encoding="utf-8")
    scores.write_text(json.dumps({"summary": {"repair_delta_vs_evolved": 0.0}}), encoding="utf-8")
    labels.write_text("{}", encoding="utf-8")

    try:
        build_freeze_manifest(
            tasks_path=tasks,
            v20_targets_path=targets,
            v20_scores_path=scores,
            label_scores_path=labels,
        )
    except ValueError as exc:
        assert "labels exist" in str(exc)
    else:
        raise AssertionError("expected existing labels to block v21 freeze")


def _task_ids():
    return [
        "plan_161",
        "plan_162",
        "plan_163",
        "plan_164",
        "plan_165",
        "plan_166",
        "plan_167",
        "plan_168",
        "math_009",
        "sym_007",
        "sci_002",
    ]


def _targets():
    return {
        "rows": [
            {"candidate_lift_vs_trajectory": -0.043, "task_id": "plan_154"},
            {"candidate_lift_vs_trajectory": -0.063, "task_id": "plan_155"},
            {"candidate_lift_vs_trajectory": -0.127286, "task_id": "plan_158"},
            {"candidate_lift_vs_trajectory": -0.13, "task_id": "plan_160"},
        ]
    }
