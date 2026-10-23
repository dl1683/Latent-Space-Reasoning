"""Synthetic evaluator binding checks, never real golden material."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.dogfood import digest, write
from experiments.dogfood_evaluate import audit, register_canary
from experiments.dogfood_report import enrich


def test_control_collects_both_readers_without_overwriting_receipt(tmp_path, monkeypatch):
    from experiments import dogfood as d
    config = tmp_path / "config.json"
    write(config, dict(readers=[dict(id="first"), dict(id="second")]))
    monkeypatch.setattr(d, "config_at", lambda _: json.loads(config.read_text()))
    monkeypatch.setattr(d, "reader_info", lambda _: {})
    base, amendment = d.NOTICE_SOURCE_IDS
    responses = iter([dict(notice_days="30", source=base),
                      dict(notice_days=45, source=amendment),
                      dict(notice_days=30, source=base),
                      dict(notice_days=45, source=amendment)])
    monkeypatch.setattr(d, "reader_call", lambda *args: dict(done_reason="stop",
        message=dict(content=json.dumps(next(responses)))))
    output = tmp_path / "control.json"
    d.smoke(config, output)
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "FAIL"
    assert [r["passed"] for r in receipt["rows"]] == [False, True, True, True]
    assert len(Path(str(output) + ".events.jsonl").read_text().splitlines()) == 4


@pytest.mark.parametrize("notice", [False, True])
def test_reader_uses_typed_schema_without_answer_values(monkeypatch, notice):
    import io
    from experiments import dogfood as d
    captured = {}
    def open_request(request, **kwargs):
        captured.update(json.loads(request.data))
        return io.BytesIO(json.dumps(dict(done=True, done_reason="stop",
            message=dict(content="{}"))).encode() + b"\n")
    monkeypatch.setattr(d.urllib.request, "urlopen", open_request)
    monkeypatch.setattr(d, "ram_guard", lambda: 30 * 2**30)
    config = dict(seed=1729, num_ctx=4096, max_output_tokens=512)
    schema = d.NOTICE_SCHEMA if notice else None
    events = []
    response = d.reader_call(config, dict(id="fake"), [], schema, progress=events.append)
    assert captured["format"] == (d.NOTICE_SCHEMA if notice else d.ACTION_SCHEMA)
    assert captured["options"]["num_gpu"] == 0
    assert len(response["response_schema_sha256"]) == 64
    assert "45" not in json.dumps(captured["format"])
    assert [e["event"] for e in events] == ["REQUEST_PREPARED", "MODEL_STREAM_CHUNK"]


def test_final_ram_refusal_still_saves_answer_and_trace(tmp_path, monkeypatch):
    from experiments import dogfood as d
    config_path = tmp_path / "config.json"
    write(config_path, {})
    index = tmp_path / "index"
    write(index / "build.json", dict(config_sha256=digest(config_path.read_bytes())))
    write(index / "t" / "manifest.json", {})
    config = dict(tasks=[dict(id="t", questions=[dict(id="q", question="Synthetic question")])],
                  readers=[dict(id="fake", origin="test")], caps=[0])
    monkeypatch.setattr(d, "config_at", lambda _: config)
    monkeypatch.setattr(d, "reader_info", lambda _: {})
    monkeypatch.setattr(d, "run_episode", lambda *args, **kwargs: dict(
        reader="fake", question_id="q", cap=0, interactions_used=0,
        seconds=0, errors=[], status="UNJUDGED_UNAUDITED", answer="Retained",
        trace=[dict(raw_text="Retained")], terminated=True))
    def refuse():
        raise MemoryError("synthetic final floor")
    monkeypatch.setattr(d, "ram_guard", refuse)
    output = tmp_path / "record.json"
    d.run_one(config_path, index, output, "t", "q", "fake", 0)
    record = json.loads(output.read_text())
    assert record["answer"] == "Retained" and record["trace"]
    assert record["errors"] and "MemoryError" in record["errors"][0]
    assert not record["terminated"]
    events = [json.loads(line) for line in output.with_suffix(".journal.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "EPISODE_START"
    assert events[-1]["record"]["answer"] == "Retained"


def test_canary_copy_does_not_change_original(tmp_path):
    original = tmp_path / "references.json"
    write(original, [dict(id="q", task_id="t", text="Synthetic golden answer")])
    before = original.read_bytes()
    registration = tmp_path / "registration.json"
    register_canary(original, registration)
    record = json.loads(registration.read_text())
    fake = Path(record["fake_gold"])
    assert original.read_bytes() == before
    assert "DOGFOODCANARY" in fake.read_text()
    assert digest(fake.read_bytes()) == record["fake_gold_sha256"]


def test_report_rejects_unbound_global_audit_pass(tmp_path):
    generation = tmp_path / "generation"
    generation.mkdir()
    path = generation / "00000.json"
    write(path, {"answer": "Candidate"})
    judgments = tmp_path / "judgments.json"
    write(judgments, [dict(file=path.name, judge="Evaluator", score=0,
        reasoning="Synthetic mismatch", golden_answer="Reference", failure_classes=["wrong_synthesis"],
        generation_sha256=digest(path.read_bytes()))])
    receipt = tmp_path / "audit.json"
    write(receipt, {"status": "PASS", "generation_hashes": {}})
    with pytest.raises(ValueError, match="audit does not bind"):
        enrich(generation, judgments, receipt, tmp_path / "enriched.jsonl")


def test_audit_refuses_uncovered_task(tmp_path):
    generation = tmp_path / "generation"
    generation.mkdir()
    write(generation / "00000.json", dict(task_id="uncovered"))
    references = tmp_path / "references.json"
    write(references, [dict(id="q", task_id="different", text="Synthetic gold")])
    with patch("experiments.dogfood_evaluate.encoder_at", return_value=object()):
        with pytest.raises(ValueError, match="every generation task"):
            audit(tmp_path / "index", generation, references, "unused", tmp_path / "audit.json")


def test_audit_refuses_empty_generation(tmp_path):
    generation = tmp_path / "generation"
    generation.mkdir()
    references = tmp_path / "references.json"
    write(references, [dict(id="q", task_id="t", text="Synthetic gold")])
    with patch("experiments.dogfood_evaluate.encoder_at", return_value=object()):
        with pytest.raises(ValueError, match="no frozen generations"):
            audit(tmp_path / "index", generation, references, "unused", tmp_path / "audit.json")
