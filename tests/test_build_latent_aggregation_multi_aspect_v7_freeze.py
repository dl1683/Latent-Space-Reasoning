import json

from experiments.build_latent_aggregation_multi_aspect_v7_freeze import (
    FROZEN_TASK_IDS,
    build_freeze_manifest,
    render_markdown,
)


def test_multi_aspect_v7_freeze_locks_tasks_and_expanded_ontology(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    replay = tmp_path / "v6_replay.json"
    coverage = tmp_path / "v6_coverage.json"
    threshold = tmp_path / "v6_threshold.json"
    label_raw = tmp_path / "labels.jsonl"
    ontology_probe_raw = tmp_path / "ontology_probe.jsonl"
    cross_latent_raw = tmp_path / "cross_latent.jsonl"
    tasks.write_text(
        "\n".join(json.dumps(_task(task_id)) for task_id in FROZEN_TASK_IDS) + "\n",
        encoding="utf-8",
    )
    replay.write_text(
        json.dumps(
            {
                "gate_evaluation": {"overall_status": "failed"},
                "summary": {"complement_coverage_count": 27},
            }
        ),
        encoding="utf-8",
    )
    coverage.write_text(
        json.dumps(
            {
                "summary": {
                    "no_complement_blockers": {
                        "anchor_dominates_candidate_aspects": 19,
                        "positive_but_below_threshold": 2,
                    },
                    "tasks_without_selected_complement": 21,
                }
            }
        ),
        encoding="utf-8",
    )
    threshold.write_text(
        json.dumps(
            {
                "summary": {
                    "positive_floor_coverage_count": 29,
                    "threshold_can_explain_failure": False,
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = build_freeze_manifest(
        tasks_path=tasks,
        v6_replay_path=replay,
        v6_coverage_path=coverage,
        v6_threshold_path=threshold,
        label_raw_path=label_raw,
        ontology_probe_raw_path=ontology_probe_raw,
        cross_latent_raw_path=cross_latent_raw,
    )
    markdown = render_markdown(manifest)

    assert manifest["task_preset"] == "latent_aggregation_multi_aspect_v7_plan345_392"
    assert manifest["task_ids"][0] == "plan_345"
    assert manifest["task_ids"][-1] == "plan_392"
    assert manifest["freshness_contract"]["prior_planning_task_max"] == 344
    assert manifest["source_family_contract"]["command_status"] == "implementation_required_before_generation"
    aspect_ids = {aspect["aspect_id"] for aspect in manifest["expanded_aspect_ontology"]["aspects"]}
    assert {"owner_assignment", "timeline_or_sequence", "polarity_or_action_direction"} <= aspect_ids
    assert manifest["statistical_gates"]["minimum_complement_coverage_count"] == 36
    assert manifest["v7_specific_gates"]["must_report_label_leakage_check"] is True
    assert "does not authorize generation yet" in markdown


def test_multi_aspect_v7_freeze_requires_threshold_failure_boundary(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    replay = tmp_path / "v6_replay.json"
    coverage = tmp_path / "v6_coverage.json"
    threshold = tmp_path / "v6_threshold.json"
    tasks.write_text(
        "\n".join(json.dumps(_task(task_id)) for task_id in FROZEN_TASK_IDS) + "\n",
        encoding="utf-8",
    )
    replay.write_text(
        json.dumps({"gate_evaluation": {"overall_status": "failed"}, "summary": {"complement_coverage_count": 27}}),
        encoding="utf-8",
    )
    coverage.write_text(json.dumps({"summary": {"tasks_without_selected_complement": 21}}), encoding="utf-8")
    threshold.write_text(
        json.dumps({"summary": {"positive_floor_coverage_count": 36, "threshold_can_explain_failure": True}}),
        encoding="utf-8",
    )

    try:
        build_freeze_manifest(
            tasks_path=tasks,
            v6_replay_path=replay,
            v6_coverage_path=coverage,
            v6_threshold_path=threshold,
            label_raw_path=tmp_path / "labels.jsonl",
            ontology_probe_raw_path=tmp_path / "ontology_probe.jsonl",
            cross_latent_raw_path=tmp_path / "cross_latent.jsonl",
        )
    except ValueError as exc:
        assert "threshold sensitivity" in str(exc)
    else:
        raise AssertionError("expected threshold-explainable v6 failure to block v7 freeze")


def _task(task_id):
    return {
        "answer": None,
        "answer_type": "rubric",
        "family": "planning",
        "max_new_tokens": 64,
        "prompt": f"Prompt for {task_id}",
        "rubric_items": [f"rubric {index}" for index in range(5)],
        "scorer": "planning_rubric_v1",
        "task_id": task_id,
    }
