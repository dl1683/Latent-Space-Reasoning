"""Synthetic label validation only, not engine authentication or semantic grading."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def report():
    path = Path(__file__).with_name("dogfood_report.py")
    spec = importlib.util.spec_from_file_location("private_judge_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path, report):
    generation = tmp_path / "generation"
    generation.mkdir()
    raw = b'{"answer": "Synthetic candidate"}'
    record = generation / "00000.json"
    record.write_bytes(raw)
    judgments = tmp_path / "judgments.json"
    judgment = dict(file=record.name, judge="Evaluator", score=0,
                    reasoning="Synthetic mismatch", golden_answer="Synthetic reference",
                    failure_classes=["wrong_synthesis"],
                    generation_sha256=report.digest(raw))
    judgments.write_text(json.dumps([judgment]), encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(dict(status="PASS",
        generation_hashes={record.name: report.digest(raw)})), encoding="utf-8")
    return generation, judgments, audit, tmp_path / "enriched.jsonl"


def test_canonical_evaluator_accepted_without_rewriting_inputs(report, inputs):
    generation, judgments, audit, output = inputs
    paths = [generation / "00000.json", judgments, audit]
    before = [path.read_bytes() for path in paths]
    rows = report.enrich(*inputs)
    assert len(rows) == 1
    assert rows[0]["judge"]["judge"] == "Evaluator"
    assert rows[0]["judge"]["score"] == 0
    assert rows[0]["status"] == "JUDGED_AUDITED"
    assert json.loads(output.read_text(encoding="utf-8")) == rows[0]
    assert [path.read_bytes() for path in paths] == before


@pytest.mark.parametrize("label", [
    "OtherEngine", "otherengine", "OtherCLI", "othercli", "otheragent", "OtherAgent", "AlternateEngine",
    "arbitrary", "", " ", "evaluator", "EVALUATOR", " Evaluator", "Evaluator ", "Evaluator\n",
    "model-evaluator", "not-evaluator", "not-Evaluator", "AlternateEngine+Evaluator", "Evaluator+AlternateEngine",
    "OtherCLI/Evaluator", "Evaluator (alternate CLI)", "Evaluator-spoof", "Evaluatortic",
    "Lun\u0430", "Evaluator\u200b", None, False, True, 0, 1, 1.5, [], ["Evaluator"],
    {}, {"judge": "Evaluator"},
])
def test_rejects_noncanonical_labels_with_value_error(report, inputs, label):
    _, judgments, _, output = inputs
    rows = json.loads(judgments.read_text(encoding="utf-8"))
    rows[0]["judge"] = label
    judgments.write_text(json.dumps(rows), encoding="utf-8")
    before = judgments.read_bytes()
    with pytest.raises(ValueError, match="permitted judge label required: Evaluator"):
        report.enrich(*inputs)
    assert not output.exists()
    assert judgments.read_bytes() == before


def test_missing_label_is_value_error(report, inputs):
    _, judgments, _, output = inputs
    rows = json.loads(judgments.read_text(encoding="utf-8"))
    del rows[0]["judge"]
    judgments.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="permitted judge label required: Evaluator"):
        report.enrich(*inputs)
    assert not output.exists()


def test_canonical_label_does_not_bypass_generation_binding(report, inputs):
    _, _, audit, output = inputs
    audit.write_text(json.dumps(dict(status="PASS", generation_hashes={})),
                     encoding="utf-8")
    with pytest.raises(ValueError, match="audit does not bind"):
        report.enrich(*inputs)
    assert not output.exists()


def test_canonical_label_preserves_audit_withholding(report, inputs):
    _, _, audit, _ = inputs
    receipt = json.loads(audit.read_text(encoding="utf-8"))
    receipt["status"] = "FAIL"
    audit.write_text(json.dumps(receipt), encoding="utf-8")
    rows = report.enrich(*inputs)
    assert rows[0]["status"] == "JUDGED_WITHHELD_AUDIT"
