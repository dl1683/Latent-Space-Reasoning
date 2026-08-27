import json

from experiments.run_latent_aggregation_complement_packet_source import (
    RunnerConfig,
    _backend_factory,
    _packet_shape,
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
    assert records[0]["complement_packet_shape"]["json_parseable"]
    assert records[0]["generation_stage"] == "candidate_generation"
    assert records[0]["schedule"]["name"] == "complement_packet_00"
    assert records[0]["task"]["task_id"] == "plan_a"
    assert records[0]["task_score"]["score"] > 0
    assert records[0]["complement_packet_prompt"]["missing_anchor_aspects"] == ["owner_assignment"]
    assert summary["json_parseable_packet_count"] == 2
    assert summary["exact_three_clause_packet_count"] == 2
    assert summary["nonempty_why_packet_count"] == 2
    assert summary["fenced_json_packet_count"] == 2
    assert "JSON-parseable packets" in markdown
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


def test_backend_factory_passes_local_model_path(monkeypatch):
    captured = {}

    class FakeHFDiffusionBackend:
        def __init__(self, candidate_key, *, device=None, dtype=None, model_path=None):
            captured.update(
                {
                    "candidate_key": candidate_key,
                    "device": device,
                    "dtype": dtype,
                    "model_path": model_path,
                }
            )

    monkeypatch.setattr(
        "experiments.run_latent_aggregation_complement_packet_source.HFDiffusionBackend",
        FakeHFDiffusionBackend,
    )
    backend = _backend_factory(
        RunnerConfig(
            device="cuda",
            dtype="bfloat16",
            model_path="external/diffusion_models/LLaDA-8B-Instruct",
        )
    )("llada-8b-instruct-hf")

    assert isinstance(backend, FakeHFDiffusionBackend)
    assert captured == {
        "candidate_key": "llada-8b-instruct-hf",
        "device": "cuda",
        "dtype": "bfloat16",
        "model_path": "external/diffusion_models/LLaDA-8B-Instruct",
    }


def test_packet_shape_accepts_fenced_json_and_counts_clause_quality():
    shape = _packet_shape(
        '```json\n{"complement_clauses":['
        '{"aspect_type":"owner_assignment","clause":"Assign an owner.","why_not_in_anchor":"No owner is named."},'
        '{"aspect_type":"timeline_or_sequence","clause":"Freeze labels before scoring.","why_not_in_anchor":"No sequence is given."},'
        '{"aspect_type":"scope_boundary","clause":"Limit the slice to uncovered tasks.","why_not_in_anchor":"No scope boundary is set."}'
        "]}\n```"
    )

    assert shape == {
        "all_clauses_have_nonempty_why": True,
        "clause_count": 3,
        "exact_three_clauses": True,
        "has_markdown_fence": True,
        "json_parseable": True,
    }


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
            "text": (
                '```json\n{"complement_clauses":['
                '{"aspect_type":"owner_assignment","clause":"Assign an owner.","why_not_in_anchor":"No owner is named."},'
                '{"aspect_type":"timeline_or_sequence","clause":"Measure reliability before rollback.","why_not_in_anchor":"No sequence is given."},'
                '{"aspect_type":"rollback_or_exit_criteria","clause":"Define rollback criteria.","why_not_in_anchor":"No rollback criteria are named."}'
                "]}\n```"
            ),
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
