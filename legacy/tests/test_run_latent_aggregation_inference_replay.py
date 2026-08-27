import json

from experiments.run_latent_aggregation_inference_replay import run_inference_replay


def test_inference_replay_scores_realized_label_free_aggregate(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    freeze = tmp_path / "freeze.json"
    raw = tmp_path / "raw.jsonl"
    task = _task("plan_a")
    tasks.write_text(json.dumps(task) + "\n", encoding="utf-8")
    freeze.write_text(
        json.dumps(
            {
                "extractor_contract": {"name": "literal_rubric_component_extractor_v1"},
                "realizer_contract": {"name": "component_provenance_template_realizer_v1"},
                "task_ids": ["plan_a"],
            }
        ),
        encoding="utf-8",
    )
    raw.write_text(
        "\n".join(
            [
                json.dumps(_record("plan_a", "a", "preserve baseline and compare intervention", [True, True, False], 0.35)),
                json.dumps(_record("plan_a", "b", "record failure metrics and rollback threshold", [False, False, True], 0.30)),
            ]
        ),
        encoding="utf-8",
    )

    result = run_inference_replay(freeze_path=freeze, raw_path=raw, tasks_path=tasks)

    assert result["summary"]["component_precision"] == 1.0
    assert result["summary"]["component_recall"] == 1.0
    assert result["tasks"][0]["realized_aggregate_score"] > result["tasks"][0]["best_single_score"]
    assert result["tasks"][0]["decision"]["status"] == "online_promoted_local"


def test_inference_replay_reports_realizer_failure_when_component_union_has_headroom(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    freeze = tmp_path / "freeze.json"
    raw = tmp_path / "raw.jsonl"
    tasks.write_text(json.dumps(_task("plan_b")) + "\n", encoding="utf-8")
    freeze.write_text(
        json.dumps(
            {
                "extractor_contract": {"name": "literal_rubric_component_extractor_v1"},
                "realizer_contract": {"name": "component_provenance_template_realizer_v1"},
                "task_ids": ["plan_b"],
            }
        ),
        encoding="utf-8",
    )
    raw.write_text(
        "\n".join(
            [
                json.dumps(_record("plan_b", "a", "preserve baseline and compare intervention", [True, True, False], 0.95)),
                json.dumps(_record("plan_b", "b", "record failure metrics and rollback threshold", [False, False, True], 0.20)),
            ]
        ),
        encoding="utf-8",
    )

    result = run_inference_replay(freeze_path=freeze, raw_path=raw, tasks_path=tasks)

    assert result["tasks"][0]["component_union_score"] == 1.0
    assert result["tasks"][0]["realized_aggregate_score"] <= result["tasks"][0]["best_single_score"]
    assert result["tasks"][0]["decision"]["status"] == "online_components_good_but_realizer_failed"


def test_inference_replay_marks_smoke_fixtures_as_non_gpu_evidence(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    freeze = tmp_path / "freeze.json"
    raw = tmp_path / "latent_aggregation_inference_smoke_raw.jsonl"
    tasks.write_text(json.dumps(_task("plan_c")) + "\n", encoding="utf-8")
    freeze.write_text(
        json.dumps(
            {
                "extractor_contract": {"name": "literal_rubric_component_extractor_v1"},
                "realizer_contract": {"name": "component_provenance_template_realizer_v1"},
                "task_ids": ["plan_c"],
            }
        ),
        encoding="utf-8",
    )
    raw.write_text(
        json.dumps(_record("plan_c", "a", "preserve baseline", [True, False, False], 0.20))
        + "\n",
        encoding="utf-8",
    )

    result = run_inference_replay(freeze_path=freeze, raw_path=raw, tasks_path=tasks)

    assert result["evidence_boundary"]["status"] == "smoke_fixture_only"
    assert "do not cite as frozen GPU evidence" in result["evidence_boundary"]["reason"]


def test_inference_replay_marks_post_hoc_thresholds_as_diagnostic(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    freeze = tmp_path / "freeze.json"
    raw = tmp_path / "raw.jsonl"
    tasks.write_text(json.dumps(_task("plan_d")) + "\n", encoding="utf-8")
    freeze.write_text(
        json.dumps(
            {
                "extractor_contract": {"name": "literal_rubric_component_extractor_v1"},
                "realizer_contract": {"name": "component_provenance_template_realizer_v1"},
                "task_ids": ["plan_d"],
            }
        ),
        encoding="utf-8",
    )
    raw.write_text(
        json.dumps(_record("plan_d", "a", "preserve baseline", [True, False, False], 0.20))
        + "\n",
        encoding="utf-8",
    )

    result = run_inference_replay(
        freeze_path=freeze,
        raw_path=raw,
        tasks_path=tasks,
        support_threshold=0.1,
        threshold_source="post_hoc_extractor_failure_v1",
    )

    assert result["support_threshold"] == 0.1
    assert result["threshold_source"] == "post_hoc_extractor_failure_v1"
    assert result["evidence_boundary"]["status"] == "post_hoc_threshold_replay"


def _task(task_id):
    return {
        "answer": None,
        "answer_type": "rubric",
        "family": "planning",
        "max_new_tokens": 64,
        "prompt": "Plan the experiment.",
        "rubric_items": [
            "preserve baseline",
            "compare intervention",
            "record failure metrics",
        ],
        "scorer": "planning_rubric_v1",
        "task_id": task_id,
    }


def _record(task_id, schedule, text, hits, score):
    return {
        "candidate_key": "model",
        "generation_stage": "candidate_generation",
        "schedule": {"name": schedule},
        "task": {"family": "planning", "task_id": task_id},
        "task_score": {
            "details": {
                "rubric_hits": [
                    {"hit": hit, "item": item}
                    for hit, item in zip(hits, _task(task_id)["rubric_items"], strict=True)
                ]
            },
            "score": score,
        },
        "text": text,
    }
