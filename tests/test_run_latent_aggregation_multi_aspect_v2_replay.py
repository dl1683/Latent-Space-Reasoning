import json

from experiments.run_latent_aggregation_multi_aspect_v2_replay import run_replay


def test_multi_aspect_replay_promotes_surviving_dimension_complement(tmp_path):
    freeze = tmp_path / "freeze.json"
    raw = tmp_path / "raw.jsonl"
    tasks = tmp_path / "tasks.jsonl"
    freeze.write_text(json.dumps(_freeze(["plan_a"])), encoding="utf-8")
    tasks.write_text(json.dumps(_task("plan_a")) + "\n", encoding="utf-8")
    raw.write_text(
        "\n".join(
            [
                json.dumps(
                    _record(
                        "plan_a",
                        "anchor",
                        score=0.30,
                        text="preserve baseline",
                        specificity=0.1,
                    )
                ),
                json.dumps(
                    _record(
                        "plan_a",
                        "candidate",
                        score=0.25,
                        text="preserve baseline measure risk threshold",
                        specificity=0.8,
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_replay(freeze_path=freeze, raw_path=raw, tasks_path=tasks)
    task = result["tasks"][0]

    assert task["selected_complement_count"] > 0
    assert task["dimension_gain_count"] > 0
    assert task["realized_score"] > task["anchor_score"]
    assert task["decision"]["status"] == "online_promoted_local"


def test_multi_aspect_replay_blocks_no_complement_material(tmp_path):
    freeze = tmp_path / "freeze.json"
    raw = tmp_path / "raw.jsonl"
    tasks = tmp_path / "tasks.jsonl"
    freeze.write_text(json.dumps(_freeze(["plan_b"])), encoding="utf-8")
    tasks.write_text(json.dumps(_task("plan_b")) + "\n", encoding="utf-8")
    raw.write_text(
        json.dumps(
            _record(
                "plan_b",
                "anchor",
                score=0.30,
                text="preserve baseline",
                specificity=0.1,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_replay(freeze_path=freeze, raw_path=raw, tasks_path=tasks)
    task = result["tasks"][0]
    realized = result["realized_rows"][0]

    assert task["selected_complement_count"] == 0
    assert realized["realized_text"] == realized["anchor_text"]
    assert task["decision"]["status"] == "blocked_no_complement_material"


def _freeze(task_ids):
    return {
        "statistical_gates": {
            "minimum_aggregate_win_count": 1,
            "minimum_aggregate_win_fraction": 0.1,
            "minimum_mean_non_rubric_lift": 0.01,
            "minimum_task_count": len(task_ids),
            "minimum_wilson_lower_bound": 0.0,
        },
        "task_ids": task_ids,
    }


def _task(task_id):
    return {
        "answer": None,
        "answer_type": "rubric",
        "family": "planning",
        "max_new_tokens": 64,
        "prompt": "Plan the experiment.",
        "rubric_items": ["preserve baseline"],
        "scorer": "planning_rubric_v1",
        "task_id": task_id,
    }


def _record(task_id, candidate_key, *, score, text, specificity):
    return {
        "candidate_key": candidate_key,
        "generation_stage": "candidate_generation",
        "schedule": {"name": "fixed"},
        "task": {"task_id": task_id},
        "task_score": {
            "details": {
                "causal_diagnosis": 0.0,
                "completion": 0.65,
                "constraint_handling": 0.0,
                "risk_awareness": 0.0,
                "rubric_coverage": 1.0,
                "rubric_hits": [{"hit": True, "item": "preserve baseline"}],
                "specificity": specificity,
            },
            "score": score,
        },
        "text": text,
    }
