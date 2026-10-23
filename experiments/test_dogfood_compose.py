"""Authored software tests; no reader, evaluator gold or benchmark scores."""
import copy
import json

import pytest

from experiments import dogfood_compose as dc


@pytest.fixture
def plan(tmp_path):
    source = tmp_path / "sources.jsonl"
    row = {"id": "authored", "question": "What is the capital of France?",
           "paragraphs": [
               {"idx": 0, "title": "France", "paragraph_text": "Été: The capital is Paris."},
               {"idx": 1, "title": "Germany", "paragraph_text": "The capital is Berlin."},
               {"idx": 2, "title": "Italy", "paragraph_text": "The capital is Rome."}]}
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return dc.prepare_record(row, {"path": str(source), "sha256": dc.sha(source.read_bytes()), "row": 0},
                             "authored-reader")


def result(plan, request, obj):
    return {"id": request["id"], "request_sha256": dc.sha(dc.packed(request)),
            "reader_digest": plan["reader_digest"], "raw_output": json.dumps(obj),
            "status": "COMPLETED", "termination": "stop",
            "input_tokens": 20, "output_tokens": 10}


def extraction(plan, claims=None, status="ok", reason=""):
    return result(plan, plan["extract_requests"][0],
                  {"status": status, "claims": claims if claims is not None else [
                      {"claim": "Paris is the capital.", "quote": "The capital is Paris."}],
                   "reason": reason})


def final(plan, outputs, quote="The capital is Paris.", status="ok", answer="Paris"):
    bundle = dc.composition(plan, outputs)
    return result(plan, bundle["request"], {"status": status, "answer": answer,
                  "evidence_quotes": [quote] if quote else [],
                  "unresolved_facets": [] if status == "ok" else [0], "reason": ""})


def test_mixed_claims_keep_good_quote_and_rejection(plan):
    call = extraction(plan, [
        {"claim": "Paris is capital.", "quote": "The capital is Paris."},
        {"claim": "Invented.", "quote": "not visible"},
        {"claim": "Malformed.", "quote": ""},
    ])
    bundle = dc.composition(plan, [call])
    assert len(bundle["extractions"][0]["accepted"]) == 1
    assert len(bundle["extractions"][0]["rejected"]) == 2
    assert bundle["request"] is not None
    data = json.loads(bundle["request"]["messages"][1]["content"])
    assert data["coverage"] == "UNKNOWN"
    assert len(data["extraction_uncertainty"][0]["rejected"]) == 2
    assert "raw_claim" not in dc.packed(data["extraction_uncertainty"])
    assert "Invented." not in dc.packed(data)
    assert data["source_context"][0]["title"] == "France"
    assert data["accepted"][0]["entailment"] == "NOT_VERIFIED"
    answer = dc.finalize(plan, [call], final(plan, [call]))
    assert answer["answer"] == "Paris"
    assert answer["entailment"] == "NOT_HOST_VERIFIED"
    assert answer["coverage"] == "UNKNOWN"
    assert answer["citations"][0]["citation_id"] == "host:1"


def test_model_uncertainty_preserved_without_erasing_facts(plan):
    call = extraction(plan, status="unresolved", reason="second relation not found")
    bundle = dc.composition(plan, [call])
    assert bundle["extractions"][0]["accepted"]
    data = json.loads(bundle["request"]["messages"][1]["content"])
    assert data["extraction_uncertainty"][0]["model_status"] == "unresolved"
    answer = dc.finalize(plan, [call], final(plan, [call], quote="", status="unresolved", answer="UNKNOWN"))
    assert answer["answer"] == "UNKNOWN"
    assert not answer["citations"]


def test_no_valid_facts_means_no_compose(plan):
    call = extraction(plan, [{"claim": "wrong", "quote": "no such phrase"}])
    assert dc.composition(plan, [call])["request"] is None
    answer = dc.finalize(plan, [call], None)
    assert answer["answer"] == "UNKNOWN" and answer["observed_calls"] == 1
    with pytest.raises(ValueError, match="no compose"):
        dc.finalize(plan, [call], call)


def test_rejected_claim_cannot_be_final_citation(plan):
    call = extraction(plan, [{"claim": "Paris.", "quote": "The capital is Paris."},
                            {"claim": "Elsewhere.", "quote": "not visible"}])
    with pytest.raises(ValueError, match="outside accepted"):
        dc.finalize(plan, [call], final(plan, [call], quote="not visible"))
    with pytest.raises(ValueError, match="outside accepted"):
        dc.finalize(plan, [call], final(plan, [call], quote="Été:"))


def test_utf8_and_subspan_coordinates(plan):
    call = extraction(plan)
    answer = dc.finalize(plan, [call], final(plan, [call], quote="Paris"))
    binding = answer["citations"][0]["bindings"][0]
    raw = plan["record"]["paragraphs"][0]["paragraph_text"].encode("utf-8")
    assert raw[binding["start_byte"]:binding["end_byte"]] == b"Paris"
    assert binding["start_byte"] == len("Été: The capital is ".encode("utf-8"))
    assert binding["span_sha256"] == dc.sha(b"Paris")


def test_multiple_exact_sources_preserved_as_plural_provenance(plan):
    row = copy.deepcopy(plan["record"])
    row["paragraphs"][1]["paragraph_text"] = row["paragraphs"][0]["paragraph_text"]
    source = plan["sources"][0]["origin"]["path"]
    from pathlib import Path
    Path(source).write_text(json.dumps(row) + "\n", encoding="utf-8")
    changed = dc.prepare_record(row, {"path": source, "sha256": dc.sha(Path(source).read_bytes()), "row": 0},
                                plan["reader_digest"])
    call = extraction(changed)
    answer = dc.finalize(changed, [call], final(changed, [call], quote="Paris"))
    assert len(answer["citations"][0]["bindings"]) == 2
    assert answer["citations"][0]["provenance_ambiguity"] is True


@pytest.mark.parametrize("mutation", ["digest", "id", "hash", "stop", "status", "duplicate", "nan", "schema"])
def test_invalid_call_is_not_partial_salvage(plan, mutation):
    call = extraction(plan)
    if mutation == "digest": call["reader_digest"] = "foreign"
    if mutation == "id": call["id"] += "different"
    if mutation == "hash": call["request_sha256"] = "bad"
    if mutation == "stop": call["termination"] = "length"
    if mutation == "status": call["status"] = "FAILED"
    if mutation == "duplicate": call["raw_output"] = '{"status":"ok","status":"unresolved"}'
    if mutation == "nan": call["raw_output"] = '{"value":NaN}'
    if mutation == "schema": call["raw_output"] = '{"status":"ok","claims":[],"reason":"","gold":"forbidden"}'
    with pytest.raises((ValueError, TypeError)):
        dc.composition(plan, [call])


def test_all_source_bytes_still_pinned(plan):
    from pathlib import Path
    source = Path(plan["sources"][0]["origin"]["path"])
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source pack changed"):
        dc.composition(plan, [extraction(plan)])


def test_resealed_policy_tampering_refused(plan):
    changed = copy.deepcopy(plan)
    changed["matched_indices"] = [0]
    changed.pop("sha256")
    changed["sha256"] = dc.sha(dc.packed(changed))
    with pytest.raises(ValueError, match="policy/source"):
        dc.validate_plan(changed)


def test_both_arms_have_identical_source_union(plan):
    direct = json.loads(plan["direct_request"]["messages"][1]["content"])["evidence"]
    extracts = [p for req in plan["extract_requests"]
                for p in json.loads(req["messages"][1]["content"])["evidence"]]
    assert direct == extracts
    assert [p["title"] for p in direct] == ["France", "Germany", "Italy"]


def test_facets_public_only_and_capped():
    assert dc.facets("Who leads X? Where was it founded?") == ["Who leads X?", "Where was it founded?"]
    question = "Who is married to the actor who starred in Film X?"
    assert dc.facets(question) == [question]
    with pytest.raises(ValueError, match="overflow"):
        dc.facets(";".join(f"Who leads X{i}?" for i in range(9)))


@pytest.mark.parametrize("location", ["row", "paragraph"])
def test_label_fields_forbidden(plan, location):
    row = copy.deepcopy(plan["record"])
    (row if location == "row" else row["paragraphs"][0])["answer"] = "CANARY_NOT_A_SOURCE"
    with pytest.raises(ValueError, match="labels forbidden"):
        dc.prepare_record(row, {k: plan["sources"][0]["origin"][k] for k in ("path", "sha256", "row")},
                          plan["reader_digest"])


def test_execute_uses_existing_callback_only_no_retry(plan):
    requests = []
    def callback(req):
        requests.append(copy.deepcopy(req))
        return extraction(plan) if "/extract/" in req["id"] else final(plan, [extraction(plan)])
    answer = dc.execute(plan, callback)
    assert answer["observed_calls"] == 2
    assert len(requests) == 2
    attempts = []
    def fail(req):
        attempts.append(req)
        raise TimeoutError("preserve this attempt")
    with pytest.raises(TimeoutError):
        dc.execute(plan, fail)
    assert len(attempts) == 1


def test_missing_or_duplicate_call_refused(plan):
    call = extraction(plan)
    for calls in ([], [call, call]):
        with pytest.raises(ValueError, match="every extraction"):
            dc.composition(plan, calls)


@pytest.mark.parametrize("indices", [[0], [False], [1], [0, 0], "unknown"])
def test_unresolved_facets_cannot_be_silently_cleared(plan, indices):
    outputs = [extraction(plan)]
    call = final(plan, outputs)
    obj = json.loads(call["raw_output"])
    obj["unresolved_facets"] = indices
    call["raw_output"] = json.dumps(obj)
    with pytest.raises(ValueError, match="facet"):
        dc.finalize(plan, outputs, call)
