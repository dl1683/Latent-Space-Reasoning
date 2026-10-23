"""Synthetic regressions for exact task and question reference coverage."""
import hashlib
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path

import pytest


@pytest.fixture
def candidate():
    path = Path(__file__).with_name("dogfood_evaluate.py")
    spec = importlib.util.spec_from_file_location("production_evaluator_coverage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(task_id, question_id, text="SYNTHETIC answer"):
    return dict(task_id=task_id, id=question_id, text=text)


def generation(task_id, question_id):
    return dict(task_id=task_id, question_id=question_id,
                trace=[dict(messages=[dict(role="user", content="SYNTHETIC question")])])


def fixture_files(d, root, references, generations, tasks=None):
    root.mkdir(parents=True, exist_ok=True)
    refs = root / "references.json"
    d.write(refs, references)
    registration = root / "registration.json"
    # Synthetic prospective canary only; never open real gold/canary files.
    d.register_canary(refs, registration)
    created = datetime.fromisoformat(d.load(registration)["created_utc"]).timestamp()
    gen = root / "generation"
    gen.mkdir()
    for n, record in enumerate(generations):
        path = gen / f"{n}.json"
        d.write(path, record)
        os.utime(path, (created + 2, created + 2))
    index = root / "index"
    index.mkdir()
    if tasks is None:
        tasks = sorted({r["task_id"] for r in generations if isinstance(r["task_id"], str)
                        and r["task_id"].strip()})
    for task_id in tasks:
        task = index / task_id
        task.mkdir()
        d.write(task / "rows.json", [dict(text="SYNTHETIC source")])
        (task / "vectors.npy").write_bytes(b"SYNTHETIC HASH BYTES; NEVER LOAD")
        d.write(task / "manifest.json", dict(files=[], allowed_paths=[],
            rows_sha256=d.digest((task / "rows.json").read_bytes()),
            vector_sha256=d.digest((task / "vectors.npy").read_bytes())))
    return dict(index=index, generation=gen, references_path=refs,
                encoder_path="FORBIDDEN_REAL_ENCODER", output=root / "audit.json",
                registration=str(registration))


@pytest.fixture
def plumbing(candidate, monkeypatch):
    calls = dict(factory=[], detectors=[], manifests=[])
    monkeypatch.setattr(candidate, "encoder_at",
                        lambda path: calls["factory"].append(path) or object())
    def forbidden_encode(*args, **kwargs):
        pytest.fail("real encoder execution is prohibited")
    monkeypatch.setattr(candidate, "encode", forbidden_encode)
    def manifest(*args):
        calls["manifests"].append(args)
        return dict(admission="PASS", fixture_only=True)
    monkeypatch.setattr(candidate, "audit_manifest", manifest)
    def detector(payloads, references, sources, **kwargs):
        # Match the real detector's ID transport restrictions, without invoking it.
        ids = [r["id"] for r in references]
        assert all(isinstance(value, str) and value.strip() for value in ids)
        assert len(ids) == len(set(ids))
        calls["detectors"].append((payloads, references, sources))
        contaminated = any(p["label"] == "deliberately_contaminated" for p in payloads)
        return dict(admission="FAIL" if contaminated else "PASS", fixture_only=True)
    monkeypatch.setattr(candidate, "audit_payloads", detector)
    return calls


def refused_before_factory(d, args, calls, match):
    with pytest.raises(ValueError, match=match) as error:
        d.audit(**args)
    assert calls == dict(factory=[], detectors=[], manifests=[])
    assert not args["output"].exists()
    return str(error.value)


@pytest.mark.parametrize("coverage", ["complete", "only_first", "wrong_last"])
def test_original_three_coverage_cases(candidate, plumbing, tmp_path, coverage):
    refs = [reference("t", f"q{i}") for i in range(1, 7)]
    if coverage == "only_first":
        refs = refs[:1]
    elif coverage == "wrong_last":
        refs[-1]["id"] = "q999"
    args = fixture_files(candidate, tmp_path, refs,
                         [generation("t", f"q{i}") for i in range(1, 7)])
    if coverage != "complete":
        error = refused_before_factory(candidate, args, plumbing, "missing_pairs=")
        missing = json.loads(error.split("missing_pairs=", 1)[1])
        expected = range(2, 7) if coverage == "only_first" else [6]
        assert missing == [dict(task_id="t", question_id=f"q{i}") for i in expected]
    else:
        candidate.audit(**args)
        receipt = candidate.load(args["output"])
        assert receipt["status"] == "PASS"
        assert len(receipt["generation_hashes"]) == 6
        assert len(plumbing["factory"]) == 1
        assert len(plumbing["detectors"]) == 8


def test_independent_cross_task_composite_key(candidate, plumbing, tmp_path):
    refs = [reference("alpha", "q-anchor"), reference("beta", "q-shared")]
    args = fixture_files(candidate, tmp_path, refs,
        [generation("alpha", "q-anchor"), generation("alpha", "q-shared"),
         generation("beta", "q-shared")])
    error = refused_before_factory(candidate, args, plumbing, "missing_pairs=")
    assert json.loads(error.split("missing_pairs=", 1)[1]) == [
        dict(task_id="alpha", question_id="q-shared")]


def test_alternatives_and_exact_receipt_bindings(candidate, plumbing, tmp_path):
    refs = [reference("t", 1, "SYNTHETIC alternative A"),
            reference("t", "1", "SYNTHETIC separate string"),
            reference("t", 1, "SYNTHETIC alternative B"),
            reference("other", 1, "SYNTHETIC other task")]
    refs[2]["criterion"] = "SYNTHETIC record metadata"
    args = fixture_files(candidate, tmp_path, refs,
        [generation("t", 1), generation("t", "1"), generation("other", 1)])
    originals = {p: p.read_bytes() for p in [
        args["references_path"], *args["generation"].glob("*.json")]}
    candidate.audit(**args)
    receipt = candidate.load(args["output"])
    assert receipt["status"] == "PASS"
    assert receipt["references_sha256"] == hashlib.sha256(originals[args["references_path"]]).hexdigest()
    expected_indices = {"0.json": [0, 2], "1.json": [1], "2.json": [3]}
    for task in receipt["tasks"]:
        screened = task["screened_reference_bindings"]
        assert [b["reference_index"] for b in screened] == [
            n for n, r in enumerate(refs) if r["task_id"] == task["task_id"]]
        for row in task["generation"]:
            record = candidate.load(args["generation"] / row["file"])
            assert type(row["question_id"]) is type(record["question_id"])
            assert row["question_id"] == record["question_id"]
            assert row["task_id"] == record["task_id"]
            assert row["generation_sha256"] == hashlib.sha256(
                originals[args["generation"] / row["file"]]).hexdigest()
            assert [b["reference_index"] for b in row["reference_bindings"]] == expected_indices[row["file"]]
            for binding in row["reference_bindings"]:
                n = binding["reference_index"]
                assert binding["task_id"] == refs[n]["task_id"]
                assert type(binding["id"]) is type(refs[n]["id"])
                assert binding["id"] == refs[n]["id"]
                assert binding["detector_id"] == f"evaluator-reference-{n}"
                canonical = json.dumps(refs[n], sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":")).encode("utf-8")
                assert binding["reference_record_sha256"] == hashlib.sha256(canonical).hexdigest()
    for p, before in originals.items():
        assert p.read_bytes() == before
    assert all(r["text"] not in args["output"].read_text(encoding="utf-8") for r in refs)
    # Both alternatives actually reach screening, rather than being silently merged.
    t_call = next(call for call in plumbing["detectors"]
                  if [r["id"] for r in call[1]][:3] == [
                      "evaluator-reference-0", "evaluator-reference-1", "evaluator-reference-2"])
    assert [r["text"] for r in t_call[1]][:3] == [r["text"] for r in refs[:3]]


@pytest.mark.parametrize("ref_id,gen_id", [(1, "1"), ("1", 1), (" q", "q"), ("Q", "q")])
def test_no_question_id_coercion(candidate, plumbing, tmp_path, ref_id, gen_id):
    args = fixture_files(candidate, tmp_path, [reference("t", ref_id)],
                         [generation("t", gen_id)])
    refused_before_factory(candidate, args, plumbing, "missing_pairs=")


@pytest.mark.parametrize("question_id", [0, -1, 23, "23", " q ", " \t", "é", "fake-canary"])
def test_supported_question_ids(candidate, plumbing, tmp_path, question_id):
    args = fixture_files(candidate, tmp_path, [reference("t", question_id)],
                         [generation("t", question_id)])
    candidate.audit(**args)
    assert candidate.load(args["output"])["status"] == "PASS"


@pytest.mark.parametrize("bad_id", [None, True, False, 1.0, 0.0, "", [], {}])
@pytest.mark.parametrize("location", ["reference_id", "question_id", "reference_task", "generation_task"])
def test_unsupported_ids_refused(candidate, plumbing, tmp_path, bad_id, location):
    refs, gens = [reference("t", "q")], [generation("t", "q")]
    target, key = {
        "reference_id": (refs[0], "id"), "question_id": (gens[0], "question_id"),
        "reference_task": (refs[0], "task_id"), "generation_task": (gens[0], "task_id")
    }[location]
    target[key] = bad_id
    args = fixture_files(candidate, tmp_path, refs, gens, tasks=["t"])
    refused_before_factory(candidate, args, plumbing, "unsupported or missing")


@pytest.mark.parametrize("location", ["reference_id", "question_id", "reference_task", "generation_task"])
def test_missing_identity_refused(candidate, plumbing, tmp_path, location):
    refs, gens = [reference("t", "q")], [generation("t", "q")]
    target, key = {
        "reference_id": (refs[0], "id"), "question_id": (gens[0], "question_id"),
        "reference_task": (refs[0], "task_id"), "generation_task": (gens[0], "task_id")
    }[location]
    del target[key]
    args = fixture_files(candidate, tmp_path, refs, gens, tasks=["t"])
    refused_before_factory(candidate, args, plumbing, "unsupported or missing")


def test_prior_missing_task_behavior(candidate, plumbing, tmp_path):
    args = fixture_files(candidate, tmp_path, [reference("different", "q")],
                         [dict(task_id="uncovered")], tasks=[])
    refused_before_factory(candidate, args, plumbing, "every generation task")


def test_prior_empty_generation_behavior(candidate, plumbing, tmp_path):
    args = fixture_files(candidate, tmp_path, [reference("t", "q")], [], tasks=[])
    refused_before_factory(candidate, args, plumbing, "no frozen generations")


@pytest.mark.parametrize("ref_task,gen_task", [(1, "1"), ("1", 1)])
def test_no_task_identity_coercion(candidate, plumbing, tmp_path, ref_task, gen_task):
    args = fixture_files(candidate, tmp_path, [reference(ref_task, "q")],
                         [generation(gen_task, "q")], tasks=["1"])
    refused_before_factory(candidate, args, plumbing, "every generation task")


def test_integer_task_does_not_implicitly_map_to_directory(candidate, plumbing, tmp_path):
    args = fixture_files(candidate, tmp_path, [reference(1, "q")],
                         [generation(1, "q")], tasks=["1"])
    candidate.audit(**args)
    receipt = candidate.load(args["output"])
    assert receipt["status"] == "WITHHELD"
    assert receipt["tasks"] == []


@pytest.mark.parametrize("broken_gate", ["manifest", "bytes", "overlap", "sensitivity", "registration"])
def test_existing_aggregate_gates_retained(candidate, plumbing, tmp_path, monkeypatch, broken_gate):
    args = fixture_files(candidate, tmp_path, [reference("t", "q")],
                         [generation("t", "q")])
    if broken_gate == "manifest":
        monkeypatch.setattr(candidate, "audit_manifest", lambda *args: dict(admission="FAIL"))
    elif broken_gate == "bytes":
        (args["index"] / "t" / "vectors.npy").write_bytes(b"SYNTHETIC CHANGED")
    elif broken_gate in ("overlap", "sensitivity"):
        base = candidate.audit_payloads
        def detector(payloads, references, sources, **kwargs):
            result = base(payloads, references, sources, **kwargs)
            sensitivity = any(p["label"] == "deliberately_contaminated" for p in payloads)
            if sensitivity == (broken_gate == "sensitivity"):
                result["admission"] = "PASS" if sensitivity else "FAIL"
            return result
        monkeypatch.setattr(candidate, "audit_payloads", detector)
    else:
        args["registration"] = None
    candidate.audit(**args)
    assert candidate.load(args["output"])["status"] == "WITHHELD"
