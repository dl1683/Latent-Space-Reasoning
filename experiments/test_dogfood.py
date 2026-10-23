import json
from pathlib import Path

import pytest

from experiments.dogfood import chunks_of, config_at, run_episode


def copied_passage():
    from experiments.dogfood import digest
    record = dict(id="record/é", status="superseded", content="Treat this quoted instruction as evidence only.")
    text = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    sha = digest(text.encode("utf-8"))
    return dict(source_id="bb:fixture:entry:record%2F%C3%A9", handle="S0", text=text,
        verification_status="VERIFIED_COPIED_RECORD", verification="COPIED_RECORD_IDENTITY_ONLY",
        quotation_format="CANONICAL_JSON_RECORD", recorded_status="superseded",
        role="EVIDENCE_NOT_INSTRUCTION", instruction_authority=False,
        pointer=dict(file="snapshots/copy.json", json_pointer="/entries/0", sha256=sha),
        provenance=dict(snapshot_file="snapshots/copy.json", snapshot_sha256="a" * 64,
            json_pointer="/entries/0", canonical_record_sha256=sha,
            namespace="bb:fixture", record_family="entry", original_id="record/é",
            original_source_locator="Z:/inert/state.json", original_locator_role="INERT_NOT_FOLLOWED"))


def test_copied_evidence_preserves_type_provenance_and_raw_branch_exactly():
    from experiments.dogfood import reader_evidence, typed_navigation_metadata
    copied = copied_passage()
    raw = dict(source_id="raw.txt", handle="S1", text="raw café", pointer=dict(source="raw.txt", page=2),
               verification_status="VERIFIED_SOURCE_BYTES", verification="BYTE_IDENTITY_ONLY")
    expected_raw = dict(source_id="raw.txt", span="S1", page=2, quoted_source_text="raw café")
    assert reader_evidence(dict(passages=[raw])) == dict(sources=[expected_raw],
        host_note="Source bytes checked only. Search coverage remains unknown. This note is not corpus evidence.")
    mixed = dict(passages=[copied, raw])
    assert typed_navigation_metadata(mixed) == (
        [copied["source_id"], "raw.txt"], ["source:" + copied["source_id"], "source:raw.txt"])
    evidence = reader_evidence(mixed)
    assert evidence["sources"][1] == expected_raw
    item = evidence["sources"][0]
    for field in ("verification_status", "verification", "quotation_format", "recorded_status",
                  "role", "instruction_authority", "pointer", "provenance"):
        assert item[field] == copied[field]
    assert item["quoted_source_text"] == copied["text"]
    assert "span" not in item
    assert "Recorded status is not current program authority" in evidence["host_note"]


@pytest.mark.parametrize("mutation", [
    "id", "status", "canonical", "hash", "snapshot_hash", "authority", "role", "format",
    "raw_span", "pointer_span", "provenance_span", "missing_provenance", "bad_locator", "utf8",
])
def test_copied_codec_refuses_identity_type_and_provenance_substitution(mutation):
    from experiments.dogfood import reader_evidence, typed_navigation_metadata
    p = copied_passage()
    if mutation == "id":
        p["source_id"] = "bb:fixture:entry:other"
    elif mutation == "status":
        p["recorded_status"] = "current_authority"
    elif mutation == "canonical":
        p["text"] += " "
    elif mutation == "hash":
        p["pointer"]["sha256"] = "0" * 64
    elif mutation == "snapshot_hash":
        p["provenance"]["snapshot_sha256"] = "bad"
    elif mutation == "authority":
        p["instruction_authority"] = True
    elif mutation == "role":
        p["role"] = "instruction"
    elif mutation == "format":
        p["quotation_format"] = "ORIGINAL_UTF8_SPAN"
    elif mutation == "raw_span":
        p["start_byte"] = 0
    elif mutation == "pointer_span":
        p["pointer"]["end_byte"] = 99
    elif mutation == "provenance_span":
        p["provenance"]["span_sha256"] = "0" * 64
    elif mutation == "missing_provenance":
        del p["provenance"]["canonical_record_sha256"]
    elif mutation == "bad_locator":
        p["pointer"]["file"] = p["provenance"]["snapshot_file"] = "../outside.json"
    elif mutation == "utf8":
        p["source_id"] = "\ud800"
    for consumer in (typed_navigation_metadata, reader_evidence):
        with pytest.raises((ValueError, UnicodeError)):
            consumer(dict(passages=[p]))


def configuration():
    return dict(schema=1, seed=1729, num_ctx=8192, max_output_tokens=384, top_k=2)


class FakeMemory:
    def __init__(self):
        self.queries = []

    def search(self, query, k, **kwargs):
        self.queries.append(query)
        return {"passages": [{"text": "Evidence only", "handle": "S0"}]}


class SourceMemory(FakeMemory):
    def search(self, query, k, **kwargs):
        self.queries.append(query)
        filename = f"{query}.txt"
        return dict(passages=[dict(text=query + ": " + "Quoted evidence " * 64,
                                  handle=f"S{len(self.queries)}", source_id=filename,
                                  pointer=dict(source=filename))])


def episode(actions, cap, **kwargs):
    memory = FakeMemory()
    calls = []
    def call(messages):
        calls.append(messages)
        return dict(message={"content": json.dumps(actions[len(calls)-1])},
                    done_reason="stop", prompt_eval_count=100, eval_count=20)
    result = run_episode(configuration(), dict(id="task", benchmark="test"),
        dict(id="q1", type="multi_hop", question="Follow A to B"),
        dict(id="fake", origin="test"), cap, memory, call, **kwargs)
    return result, memory, calls


def test_two_real_followups_and_answer():
    result, memory, calls = episode([{"query": "A"}, {"query": "B"}, {"answer": "A amends B [S0]"}], 3)
    assert memory.queries == ["A", "B"]
    assert result["interactions_used"] == 2
    assert result["input_tokens"] == 300
    assert result["terminated"]
    assert result["golden_answer"] is None and result["judge"] is None
    assert len(calls) == 3


def test_cap_forces_answer_not_extra_tool():
    result, memory, calls = episode([{"query": "A"}, {"query": "B"}], 1)
    assert memory.queries == ["A"]
    assert result["errors"] == ["INVALID_ACTION_OR_CAP_VIOLATION"]
    assert "No tool calls remain" in calls[-1][-1]["content"]


def test_model_alone_has_no_evidence():
    result, memory, calls = episode([{"answer": "Unknown"}], 0)
    assert not memory.queries
    assert result["interactions_used"] == 0
    assert not any("Evidence only" in m["content"] for m in calls[0])


@pytest.mark.parametrize("arm", ["bm25", "dense", "hybrid"])
@pytest.mark.parametrize("cap", [0, 1])
def test_run_one_only_loads_encoder_for_dense_memory(tmp_path, monkeypatch, arm, cap):
    from experiments import dogfood as d
    source = tmp_path / "source.txt"
    source.write_text("Synthetic source", encoding="utf-8")
    config = dict(configuration(), encoder=str(tmp_path / "encoder"), caps=[0, 1],
        chunk_chars=128, readers=[dict(id="fake", origin="test", parameters=100)],
        tasks=[dict(id="task", benchmark="test", allowed_paths=[str(source)],
                    source_roots=[str(tmp_path)],
                    questions=[dict(id="q", type="test", question="Read the source")])])
    config_path = tmp_path / "config.json"
    d.write(config_path, config)
    index = tmp_path / "index"
    d.write(index / "build.json", dict(config_sha256=d.file_digest(config_path)))
    d.write(index / "task" / "manifest.json", {})
    encoder = object()
    encoder_loads, memory_encoders = [], []

    def encoder_at(path):
        encoder_loads.append(path)
        return encoder

    class Memory(FakeMemory):
        def __init__(self, directory, value, route_matrix="float32"):
            super().__init__()
            self.route_matrix_retained_numeric_bytes = 0
            self.route_matrix_conversion_seconds = 0.0
            memory_encoders.append(value)

    actions = iter(([dict(query="source")] if cap else []) + [dict(answer="Evidence only")])
    monkeypatch.setattr(d, "encoder_at", encoder_at)
    monkeypatch.setattr(d, "Memory", Memory)
    monkeypatch.setattr(d, "reader_info", lambda _: dict(synthetic=True))
    monkeypatch.setattr(d, "ram_guard", lambda: 30 * 2**30)
    monkeypatch.setattr(d, "reader_call", lambda *args, **kwargs:
        dict(message=dict(content=json.dumps(next(actions))), done_reason="stop"))
    output = tmp_path / "answer.json"
    d.run_one(config_path, index, output, "task", "q", "fake", cap, arm=arm)
    needs_encoder = bool(cap and arm != "bm25")
    assert encoder_loads == ([config["encoder"]] if needs_encoder else [])
    assert memory_encoders == ([encoder if needs_encoder else None] if cap else [])
    result = d.load(output)
    assert result["terminated"] and not result["errors"]
    assert result["arm"] == arm and result["interactions_used"] == cap


@pytest.mark.parametrize("protocol", ["v1", "v2"])
@pytest.mark.parametrize("cap", [0, 3])
def test_body_instruction_reaches_baseline_and_early_answer(protocol, cap):
    from experiments.dogfood import ANSWER_BODY_INSTRUCTION
    actions = ([dict(query="facts")] if cap else []) + [dict(answer="Substantive body")]
    prompts = []

    def call(messages):
        prompts.append(messages)
        return dict(message=dict(content=json.dumps(actions[len(prompts) - 1])),
                    done_reason="stop")

    result = run_episode(dict(configuration(), protocol=protocol),
        dict(id="t", benchmark="test"),
        dict(id="q", type="test", question="Create report.docx."),
        dict(id="fake", origin="test"), cap, SourceMemory(), call)
    assert result["terminated"] and result["answer"] == "Substantive body"
    assert result["interactions_used"] == bool(cap)
    assert all(ANSWER_BODY_INSTRUCTION in prompt[0]["content"] for prompt in prompts)
    assert not result["context_evictions"]


@pytest.mark.parametrize("protocol", ["v1", "v2"])
@pytest.mark.parametrize("context,evicts", [(8192, False), (1800, True)])
def test_forced_body_instruction_survives_eviction(protocol, context, evicts):
    from experiments.dogfood import ANSWER_BODY_INSTRUCTION, FORCED_ANSWER_INSTRUCTION
    actions = [dict(query="first"), dict(query="second"), dict(answer="First\nSecond")]
    prompts = []

    def call(messages):
        prompts.append(messages)
        return dict(message=dict(content=json.dumps(actions[len(prompts) - 1])),
                    done_reason="stop", prompt_eval_count=100)

    result = run_episode(dict(configuration(), protocol=protocol, num_ctx=context,
                              max_output_tokens=128),
        dict(id="t", benchmark="test"), dict(id="q", type="test", question="Two parts."),
        dict(id="fake", origin="test"), 2, SourceMemory(), call)
    assert result["terminated"] and result["answer"] == "First\nSecond"
    assert bool(result["context_evictions"]) is evicts
    assert prompts[-1][-1] == dict(role="user", content=FORCED_ANSWER_INSTRUCTION)
    assert ANSWER_BODY_INSTRUCTION in prompts[-1][-1]["content"]
    assert sum("latent_memory returned" in m["content"] for m in prompts[-1]) >= 1
    assert "second.txt" in json.dumps(prompts[-1])
    assert ("first.txt" not in json.dumps(prompts[-1])) is evicts


def test_plain_answer_is_preserved_inside_json():
    result, _, _ = episode([{"answer": "First line\nSecond line"}], 1)
    assert result["answer"] == "First line\nSecond line"


@pytest.fixture
def routing_memory(tmp_path, monkeypatch):
    import numpy as np
    from experiments import dogfood as d

    memory = d.Memory.__new__(d.Memory)
    memory.directory, memory.encoder = tmp_path, object()
    memory.vectors = np.eye(3, dtype=np.float32)
    memory.rows, memory.seen, memory.source_ids = [], set(), {}
    leaves = tmp_path / "leaves"
    leaves.mkdir()
    for i in range(3):
        data = f"Synthetic source {i}".encode()
        pin = d.digest(data)
        path = str(tmp_path / f"document{i}.txt")
        (leaves / (pin + ".txt")).write_bytes(data)
        memory.rows.append(dict(source=path, leaf_sha256=pin, start_byte=0,
                                end_byte=len(data), span_sha256=pin, text=data.decode()))
        memory.source_ids[path] = f"document{i}.txt"
    values = {"narrow": [1, .9, 0], "whole question": [0, 1, .5],
              "other question": [.5, 0, 1], "null question": [0, 0, 0]}
    encodes = []

    def encode(encoder, texts):
        encodes.extend(texts)
        return np.asarray([values.get(text, [1, .9, 0]) for text in texts],
                          dtype=np.float32)

    class Lexical:
        def score(self, text):
            return np.asarray(values.get(text, [1, .9, 0]), dtype=np.float32)

    memory.bm25 = Lexical()
    monkeypatch.setattr(d, "encode", encode)
    return memory, encodes


@pytest.mark.parametrize("arm", ["bm25", "dense", "hybrid"])
def test_public_question_prior_reranks_and_caches_without_changing_sources(routing_memory, arm):
    memory, encodes = routing_memory
    legacy = memory.search("narrow", 1, arm=arm)
    assert legacy["passages"][0]["source_id"] == "document0.txt"
    assert "routing" not in legacy
    first = memory.search("narrow", 1, arm=arm, question="whole question")
    second = memory.search("narrow", 1, arm=arm, question="whole question")
    for result in (first, second):
        assert result["passages"][0]["source_id"] == "document1.txt"
        assert result["passages"][0]["text"] == "Synthetic source 1"
        assert result["passages"][0]["verification_status"] == "VERIFIED_SOURCE_BYTES"
        assert result["routing"]["question_prior_used"]
    assert first["routing"]["question_prior_encoder_calls"] == int(arm != "bm25")
    assert second["routing"]["question_prior_cache_hit"]
    assert second["routing"]["question_prior_encoder_calls"] == 0
    assert encodes.count("whole question") == int(arm != "bm25")


def test_question_prior_preserves_explicit_source_addresses(routing_memory):
    memory, encodes = routing_memory
    result = memory.search("source:document2.txt", 1, arm="dense",
                           question="whole question")
    assert result["passages"][0]["source_id"] == "document2.txt"
    assert not result["routing"]["question_prior_used"]
    assert encodes == ["source:document2.txt"]


def test_question_prior_keeps_null_prior_and_same_query_behavior(routing_memory):
    memory, encodes = routing_memory
    for question in ("null question", "narrow"):
        result = memory.search("narrow", 1, arm="dense", question=question)
        assert result["passages"][0]["source_id"] == "document0.txt"
        assert not result["routing"]["question_prior_used"]
    assert encodes == ["narrow", "null question", "narrow"]


def test_question_prior_cache_is_keyed_by_question_and_arm(routing_memory):
    memory, encodes = routing_memory
    for question, arm in (("whole question", "dense"), ("other question", "dense"),
                          ("other question", "hybrid")):
        result = memory.search("narrow", 1, arm=arm, question=question)
        assert not result["routing"]["question_prior_cache_hit"]
    assert encodes.count("whole question") == 1
    assert encodes.count("other question") == 2


def test_question_prior_cannot_bypass_span_verification(routing_memory):
    memory, _ = routing_memory
    row = memory.rows[1]
    (memory.directory / "leaves" / (row["leaf_sha256"] + ".txt")).write_text("tampered")
    with pytest.raises(ValueError, match="source leaf/span verification failed"):
        memory.search("narrow", 1, arm="dense", question="whole question")


@pytest.mark.parametrize("enabled", [False, True])
def test_episode_only_passes_public_question_when_prior_enabled(enabled):
    options = []

    class CaptureMemory(SourceMemory):
        def search(self, query, k, **kwargs):
            options.append(kwargs)
            return super().search(query, k, **kwargs)

    actions = iter([dict(query="narrow"), dict(answer="Synthetic answer")])
    result = run_episode(dict(configuration(), protocol="v2", question_prior=enabled),
        dict(id="t", benchmark="test"),
        dict(id="q", type="test", question="The unchanged public question."),
        dict(id="fake", origin="test"), 3, CaptureMemory(),
        lambda _: dict(message=dict(content=json.dumps(next(actions))), done_reason="stop"))
    assert result["terminated"]
    assert ("question" in options[0]) is enabled
    if enabled:
        assert options[0]["question"] == "The unchanged public question."


def test_invalid_action_does_not_fake_answer():
    result, _, _ = episode([{"query": "A", "answer": "B"}], 1)
    assert not result["terminated"] and not result["answer"]


def test_utf8_spans_are_exact():
    text = "é🙂Hello\n" * 50
    chunks = list(chunks_of(text, 128))
    assert "".join(c for _, _, c in chunks) == text
    raw = text.encode()
    assert all(raw[a:b].decode() == c for a, b, c in chunks)


def test_gold_config_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(dict(schema=1, gold_standard="secret")), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluator"):
        config_at(path)


@pytest.mark.parametrize("value", [0, -1, 513, True, 128.5])
def test_prefill_batch_is_a_bounded_integer(tmp_path, value):
    config = dict(configuration(), tasks=[], readers=[], caps=[0],
                  chunk_chars=128, encoder=str(tmp_path), num_batch=value)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="prefill batch"):
        config_at(path)


@pytest.mark.parametrize("value", [0, -1, 8193, True, 2048.5])
def test_answer_bound_is_a_bounded_integer(tmp_path, value):
    config = dict(configuration(), tasks=[], readers=[], caps=[0],
                  chunk_chars=128, encoder=str(tmp_path), max_answer_chars=value)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="answer length"):
        config_at(path)


@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_question_prior_config_requires_boolean(tmp_path, value):
    config = dict(configuration(), tasks=[], readers=[], caps=[0],
                  chunk_chars=128, encoder=str(tmp_path), question_prior=value)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="question prior must be boolean"):
        config_at(path)


@pytest.mark.parametrize("phase", ["query", "choose", "answer"])
def test_answer_bound_does_not_mutate_original_schemas_or_queries(phase):
    from experiments.dogfood import PHASE_SCHEMAS, action_schema
    before = json.dumps(PHASE_SCHEMAS, sort_keys=True)
    schema = action_schema(dict(max_answer_chars=2048), phase)
    for branch in schema.get("oneOf", [schema]):
        properties = branch["properties"]
        if "answer" in properties:
            assert properties["answer"] == dict(type="string", minLength=1, maxLength=2048)
        if "query" in properties:
            assert properties["query"] == dict(type="string", minLength=1, maxLength=256)
    assert json.dumps(PHASE_SCHEMAS, sort_keys=True) == before
    assert action_schema({}, phase) is PHASE_SCHEMAS[phase]


def test_answer_bound_rejects_oversized_output_without_truncating():
    config = dict(configuration(), protocol="v2", max_answer_chars=2048)
    answer = "x" * 2049
    response = dict(message=dict(content=json.dumps(dict(answer=answer))), done_reason="stop")
    result = run_episode(config, dict(id="t", benchmark="test"),
        dict(id="q", type="test", question="A question"), dict(id="fake", origin="test"),
        0, FakeMemory(), lambda _: response)
    assert result["answer"] == "" and not result["terminated"]
    assert result["errors"] == ["INVALID_ACTION_OR_CAP_VIOLATION"]
    assert json.loads(result["trace"][0]["raw_text"])["answer"] == answer


def test_prefill_batch_is_sent_and_journaled_without_changing_context(monkeypatch):
    from experiments import dogfood as d
    config = dict(configuration(), num_batch=128, max_answer_chars=2048)
    events, payloads = [], []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return iter([b'{"message":{"content":"{}"},"done":true}'])

    def open_request(request, timeout):
        payloads.append(json.loads(request.data))
        return Stream()

    monkeypatch.setattr(d, "ram_guard", lambda: 30 * 2**30)
    monkeypatch.setattr(d.urllib.request, "urlopen", open_request)
    result = d.reader_call(config, dict(id="fake"), [], progress=events.append)
    options = payloads[0]["options"]
    assert options["num_batch"] == 128 and options["num_ctx"] == config["num_ctx"]
    assert options["num_predict"] == config["max_output_tokens"]
    assert result["request_options"] == events[0]["request_options"] == options
    assert payloads[0]["format"]["oneOf"][1]["properties"]["answer"]["maxLength"] == 2048


def test_early_stop_does_not_consume_full_cap():
    result, memory, _ = episode([{"answer": "No source supplied"}], 20)
    assert result["interactions_used"] == 0 and not memory.queries


def test_v2_requires_actual_initial_query_even_with_plain_fake_call():
    config = dict(configuration(), protocol="v2")
    response = dict(message=dict(content='{"answer":"Unsupported"}'), done_reason="stop")
    result = run_episode(config, dict(id="t", benchmark="test"),
        dict(id="q", type="test", question="A question"), dict(id="fake", origin="test"),
        3, FakeMemory(), lambda _: response)
    assert result["errors"] == ["ANSWER_BEFORE_REQUIRED_MEMORY_QUERY"]
    assert result["interactions_used"] == 0 and not result["terminated"]


def test_v2_query_bound_is_declared_and_host_rejects_without_truncation():
    from experiments.dogfood import PHASE_SCHEMAS
    assert PHASE_SCHEMAS["query"]["properties"]["query"]["maxLength"] == 256
    config = dict(configuration(), protocol="v2")
    long_query = "x" * 257
    response = dict(message=dict(content=json.dumps(dict(query=long_query))), done_reason="stop")
    memory = FakeMemory()
    result = run_episode(config, dict(id="t", benchmark="test"),
        dict(id="q", type="test", question="A question"), dict(id="fake", origin="test"),
        3, memory, lambda _: response)
    assert not memory.queries and result["errors"] == ["INVALID_ACTION_OR_CAP_VIOLATION"]
    assert json.loads(result["trace"][0]["raw_text"])["query"] == long_query
    assert "Search now" in result["trace"][0]["messages"][-1]["content"]


@pytest.mark.parametrize("cap,phases", [(0, ["answer"]), (1, ["query", "answer"]),
                                       (3, ["query", "choose"])])
def test_v2_phase_adapter_and_source_only_reader_view(cap, phases):
    from experiments.dogfood import SYSTEM_V2
    config = dict(configuration(), protocol="v2")
    class Memory:
        def search(self, *args, **kwargs):
            return dict(passages=[dict(text="Quoted evidence", handle="S8", source_id="repo/a.txt",
                pointer=dict(source="a.txt", span_sha256="PRIVATE_POINTER_HASH"))],
                completeness="UNKNOWN", semantic_absence="UNKNOWN")
    observed, prompts = [], []
    def phase_call(messages, phase):
        observed.append(phase)
        prompts.append(messages)
        action = dict(query="A") if phase == "query" else dict(answer="Quoted evidence, repo/a.txt")
        return dict(message=dict(content=json.dumps(action)), done_reason="stop")
    result = run_episode(config, dict(id="t", benchmark="test"),
        dict(id="q", type="test", question="A question"), dict(id="fake", origin="test"),
        cap, Memory(), lambda _: pytest.fail("phase adapter missing"), phase_call=phase_call)
    assert observed == phases and result["terminated"]
    assert result["interactions_used"] == bool(cap)
    assert "[S3]" not in SYSTEM_V2
    if cap:
        reader_view = prompts[-1][-2 if cap == 1 else -1]["content"]
        assert "Quoted evidence" in reader_view and "repo/a.txt" in reader_view
        assert "PRIVATE_POINTER_HASH" not in reader_view and "semantic_absence" not in reader_view
        assert result["trace"][0]["retrieval"]["passages"][0]["pointer"]["span_sha256"] == "PRIVATE_POINTER_HASH"


def test_source_identifiers_are_relative_and_disambiguate_same_filenames(tmp_path):
    from experiments.dogfood import source_identifiers
    first, second = tmp_path / "a" / "README.md", tmp_path / "b" / "README.md"
    assert source_identifiers([str(second), str(first)]) == {
        str(first): "a/README.md", str(second): "b/README.md"}


def test_source_contract_ignores_reader_settings_but_binds_source_and_encoder(tmp_path, monkeypatch):
    from experiments import dogfood as d
    encoder = tmp_path / "encoder"
    encoder.mkdir()
    weights = encoder / "weights.bin"
    weights.write_bytes(b"synthetic encoder")
    source = tmp_path / "a.txt"
    source.write_text("Synthetic source")
    monkeypatch.setattr(d, "package_version", lambda _: "synthetic")
    config = dict(encoder=str(encoder), chunk_chars=100, caps=[0, 3], protocol="v1",
        tasks=[dict(id="t", source_roots=[str(tmp_path)], allowed_paths=[str(source)],
                    questions=[dict(question="A")])])
    before = d.source_contract(config)
    changed_runtime = dict(config, caps=[0, 1, 10], protocol="v2", max_output_tokens=1024)
    assert d.source_contract(changed_runtime) == before
    source.write_text("Changed source")
    assert d.source_contract(config) != before
    source.write_text("Synthetic source")
    weights.write_bytes(b"changed encoder")
    assert d.source_contract(config) != before


def test_identical_source_corpora_are_encoded_once_not_once_per_question_task(tmp_path, monkeypatch):
    import numpy as np
    from experiments import dogfood as d
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = source_root / "a.txt"
    source.write_text("Quoted synthetic source.")
    config_path = tmp_path / "config.json"
    d.write(config_path, {})
    config = dict(encoder="synthetic", chunk_chars=128,
        tasks=[dict(id=name, source_roots=[str(source_root)], allowed_paths=[str(source)])
               for name in ("first", "second")])
    class Encoder:
        max_seq_length = 256
        def get_sentence_embedding_dimension(self):
            return 2
    encodes = []
    monkeypatch.setattr(d, "config_at", lambda _: config)
    monkeypatch.setattr(d, "source_contract", lambda _: dict(synthetic=True))
    monkeypatch.setattr(d, "encoder_at", lambda _: Encoder())
    monkeypatch.setattr(d, "encoder_chunks", lambda text, size, encoder: d.chunks_of(text, size))
    monkeypatch.setattr(d, "ram_guard", lambda: 30 * 2**30)
    def encode(encoder, texts):
        encodes.append(texts)
        return np.ones((len(texts), 2), dtype=np.float32)
    monkeypatch.setattr(d, "encode", encode)
    index = tmp_path / "index"
    d.build(config_path, index)
    assert len(encodes) == 1
    assert (index / "first" / "rows.json").read_bytes() == (index / "second" / "rows.json").read_bytes()
    assert d.load(index / "first" / "manifest.json")["corpus_encodes"] == 1
    assert d.load(index / "second" / "manifest.json")["corpus_encodes"] == 0
