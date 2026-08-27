import json

import pytest

from experiments.build_latent_aggregation_multi_aspect_v10_complement_freeze import (
    FROZEN_TASK_IDS,
    build_freeze_manifest,
    render_markdown,
)


def test_v10_complement_freeze_locks_fresh_transfer_contract(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    replay = tmp_path / "v9_replay.json"
    source_scores = tmp_path / "v9_scores.json"
    tasks.write_text(
        "\n".join(json.dumps(_task(task_id)) for task_id in FROZEN_TASK_IDS) + "\n",
        encoding="utf-8",
    )
    replay.write_text(json.dumps(_v9_replay()), encoding="utf-8")
    source_scores.write_text(json.dumps({"summary": {"json_parseable_packet_count": 72}}), encoding="utf-8")

    manifest = build_freeze_manifest(
        tasks_path=tasks,
        v9_replay_path=replay,
        v9_source_scores_path=source_scores,
        label_raw_path=tmp_path / "labels.jsonl",
        label_scores_path=tmp_path / "labels.json",
        ontology_probe_raw_path=tmp_path / "ontology.jsonl",
        ontology_probe_scores_path=tmp_path / "ontology.json",
        cross_latent_raw_path=tmp_path / "cross.jsonl",
        cross_latent_scores_path=tmp_path / "cross.json",
        packet_prompts_path=tmp_path / "packet_prompts.jsonl",
        packet_raw_path=tmp_path / "packet_raw.jsonl",
        packet_scores_path=tmp_path / "packet_scores.json",
        packet_report_path=tmp_path / "packet_report.md",
        replay_output_path=tmp_path / "v10_replay.json",
        aspects_output_path=tmp_path / "aspects.jsonl",
        realized_output_path=tmp_path / "realized.jsonl",
        replay_report_path=tmp_path / "replay.md",
    )
    markdown = render_markdown(manifest)

    assert manifest["schema"] == "latent_aggregation_multi_aspect_v10_complement_freeze.v1"
    assert manifest["task_ids"][0] == "plan_393"
    assert manifest["task_ids"][-1] == "plan_440"
    assert manifest["freshness_contract"]["prior_planning_task_max"] == 392
    assert manifest["prior_evidence"]["v9_evidence_boundary_status"] == "post_failure_v9_complement_packet_replay"
    assert manifest["transfer_contract"]["policy"] == "v9_complement_packet_policy_fixed_before_v10_labels"
    assert manifest["transfer_contract"]["packet_policy"]["samples_per_task"] == 3
    assert manifest["statistical_gates"]["minimum_wilson_lower_bound"] == 0.60
    assert manifest["v10_specific_gates"]["must_report_packet_shape_metrics"] is True
    assert "--model-path external\\diffusion_models\\LLaDA-8B-Instruct" in manifest["source_family_contract"]["packet_generation_command"]
    assert "v10 is the first valid chance" in markdown


def test_v10_complement_freeze_requires_post_failure_v9_boundary(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    replay = tmp_path / "v9_replay.json"
    source_scores = tmp_path / "v9_scores.json"
    tasks.write_text(
        "\n".join(json.dumps(_task(task_id)) for task_id in FROZEN_TASK_IDS) + "\n",
        encoding="utf-8",
    )
    bad_replay = _v9_replay()
    bad_replay["evidence_boundary"]["status"] = "fresh_promotion"
    replay.write_text(json.dumps(bad_replay), encoding="utf-8")
    source_scores.write_text(json.dumps({"summary": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="post-failure diagnostic"):
        build_freeze_manifest(
            tasks_path=tasks,
            v9_replay_path=replay,
            v9_source_scores_path=source_scores,
            label_raw_path=tmp_path / "labels.jsonl",
            label_scores_path=tmp_path / "labels.json",
            ontology_probe_raw_path=tmp_path / "ontology.jsonl",
            ontology_probe_scores_path=tmp_path / "ontology.json",
            cross_latent_raw_path=tmp_path / "cross.jsonl",
            cross_latent_scores_path=tmp_path / "cross.json",
            packet_prompts_path=tmp_path / "packet_prompts.jsonl",
            packet_raw_path=tmp_path / "packet_raw.jsonl",
            packet_scores_path=tmp_path / "packet_scores.json",
            packet_report_path=tmp_path / "packet_report.md",
            replay_output_path=tmp_path / "v10_replay.json",
            aspects_output_path=tmp_path / "aspects.jsonl",
            realized_output_path=tmp_path / "realized.jsonl",
            replay_report_path=tmp_path / "replay.md",
        )


def test_v10_complement_freeze_refuses_existing_outputs(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    replay = tmp_path / "v9_replay.json"
    source_scores = tmp_path / "v9_scores.json"
    packet_raw = tmp_path / "packet_raw.jsonl"
    tasks.write_text(
        "\n".join(json.dumps(_task(task_id)) for task_id in FROZEN_TASK_IDS) + "\n",
        encoding="utf-8",
    )
    replay.write_text(json.dumps(_v9_replay()), encoding="utf-8")
    source_scores.write_text(json.dumps({"summary": {}}), encoding="utf-8")
    packet_raw.write_text("already generated\n", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing v10 freeze after output artifacts exist"):
        build_freeze_manifest(
            tasks_path=tasks,
            v9_replay_path=replay,
            v9_source_scores_path=source_scores,
            label_raw_path=tmp_path / "labels.jsonl",
            label_scores_path=tmp_path / "labels.json",
            ontology_probe_raw_path=tmp_path / "ontology.jsonl",
            ontology_probe_scores_path=tmp_path / "ontology.json",
            cross_latent_raw_path=tmp_path / "cross.jsonl",
            cross_latent_scores_path=tmp_path / "cross.json",
            packet_prompts_path=tmp_path / "packet_prompts.jsonl",
            packet_raw_path=packet_raw,
            packet_scores_path=tmp_path / "packet_scores.json",
            packet_report_path=tmp_path / "packet_report.md",
            replay_output_path=tmp_path / "v10_replay.json",
            aspects_output_path=tmp_path / "aspects.jsonl",
            realized_output_path=tmp_path / "realized.jsonl",
            replay_report_path=tmp_path / "replay.md",
        )


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


def _v9_replay():
    return {
        "evidence_boundary": {"status": "post_failure_v9_complement_packet_replay"},
        "gate_evaluation": {"overall_status": "passed"},
        "summary": {
            "complement_coverage_count": 47,
            "hard_contradiction_count": 0,
            "online_promoted_task_count": 46,
            "unsupported_addition_count": 0,
        },
    }
