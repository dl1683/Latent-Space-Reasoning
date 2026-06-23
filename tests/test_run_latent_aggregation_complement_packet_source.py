import json

from experiments.run_latent_aggregation_complement_packet_source import (
    RunnerConfig,
    render_markdown,
    run_complement_packet_source,
)


def test_complement_packet_source_runner_writes_replay_compatible_records(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    tasks = tmp_path / "tasks.jsonl"
    prompts.write_text(
        json.dumps(
            {
                "anchor_score": 0.4,
                "anchor_trajectory_id": "plan_a:anchor",
                "failure_class": "repair_not_stronger_no_new_expanded_aspect",
                "missing_anchor_aspects": ["owner_assignment"],
                "prompt": "Return owner complement.",
                "targeted_delta_vs_original_anchor": -0.1,
                "targeted_score": 0.3,
                "task_id": "plan_a",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tasks.write_text(json.dumps(_task("plan_a")) + "\n", encoding="utf-8")

    records, summary = run_complement_packet_source(
        prompts_path=prompts,
        tasks_path=tasks,
        config=RunnerConfig(candidates=("fake-model",), samples_per_task=2),
        backend_factory=lambda candidate_key: _FakeBackend(candidate_key),
    )
    markdown = render_markdown(summary)

    assert len(records) == 2
    assert summary["source_family"] == "complement_packet"
    assert summary["generated_record_count"] == 2
    assert records[0]["source_family"] == "complement_packet"
    assert records[0]["generation_stage"] == "candidate_generation"
    assert records[0]["schedule"]["name"] == "complement_packet_00"
    assert records[0]["task"]["task_id"] == "plan_a"
    assert records[0]["task_score"]["score"] > 0
    assert records[0]["complement_packet_prompt"]["missing_anchor_aspects"] == ["owner_assignment"]
    assert "replay is required" in markdown


def test_complement_packet_source_runner_rejects_missing_prompt_task(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    tasks = tmp_path / "tasks.jsonl"
    prompts.write_text(json.dumps({"prompt": "x", "task_id": "plan_missing"}) + "\n", encoding="utf-8")
    tasks.write_text(json.dumps(_task("plan_a")) + "\n", encoding="utf-8")

    try:
        run_complement_packet_source(
            prompts_path=prompts,
            tasks_path=tasks,
            config=RunnerConfig(candidates=("fake-model",), samples_per_task=1),
            backend_factory=lambda candidate_key: _FakeBackend(candidate_key),
        )
    except ValueError as exc:
        assert "missing from" in str(exc)
    else:
        raise AssertionError("expected missing task failure")


class _FakeGeneration:
    def __init__(self, candidate_key: str, prompt: str):
        self.candidate_key = candidate_key
        self.prompt = prompt

    def to_dict(self):
        return {
            "candidate_key": self.candidate_key,
            "config": {},
            "generated_token_count": 8,
            "generated_token_ids": [1, 2, 3],
            "model_id": "fake",
            "prompt": self.prompt,
            "text": "Assign an owner and measure reliability before rollback.",
        }


class _FakeBackend:
    def __init__(self, candidate_key: str):
        self.candidate_key = candidate_key

    def generate(self, prompt, config=None):
        return _FakeGeneration(self.candidate_key, prompt)


def _task(task_id):
    return {
        "answer": None,
        "answer_type": "rubric",
        "family": "planning",
        "max_new_tokens": 64,
        "prompt": "Plan a reliability decision.",
        "rubric_items": ["assign owner", "measure reliability", "define rollback"],
        "scorer": "planning_rubric_v1",
        "task_id": task_id,
    }
