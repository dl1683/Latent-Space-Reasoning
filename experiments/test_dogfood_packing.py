"""No inference: real source-span logic, isolated saved traces, mocked transport."""
import copy
import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "CLAIM_PATHS", (tmp_path / "gpu.lock", tmp_path / "owner.json"))
    monkeypatch.setattr(dp, "api", lambda *a, **k: {"models": []})
    monkeypatch.setattr(dp.time, "sleep", lambda _: None)

from experiments import dogfood_packing as dp


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path)


@pytest.fixture
def fixture(tmp_path):
    tasks, rows = [], []
    for i in range(2):
        source = tmp_path / str(i) / "source.txt"
        source.parent.mkdir()
        source.write_bytes(f"source {i}: caf\u00e9\r\nsecond line".encode())
        leaf = source.read_text(encoding="utf-8").encode()
        p = {"text": leaf.decode(), "verification_status": "VERIFIED_SOURCE_BYTES",
             "pointer": {"source": str(source), "source_sha256": dp.sha(source.read_bytes()),
                         "leaf_sha256": dp.sha(leaf), "page": None, "start_byte": 0,
                         "end_byte": len(leaf), "span_sha256": dp.sha(leaf)}}
        tasks.append({"id": str(i), "source_roots": [str(source.parent)], "allowed_paths": [str(source)]})
        rows.append({"task_id": str(i), "question_id": str(i) + ":1", "question": f"Question {i}?",
                     "control": "own", "trace": [{"retrieval": {"passages": [p, p]},
                         "messages": [{"content": "OLD_PROMPT_POISON"}], "response": "OLD_RESPONSE_POISON"}],
                     "answer": "OLD_ANSWER_POISON", "golden_answer": "GOLD_POISON",
                     "judge": "JUDGE_POISON"})
    source_config = write(tmp_path / "source_config.json", {"tasks": tasks})
    inputs = [write(tmp_path / f"row{i}.json", r) for i, r in enumerate(rows)]
    c = {"source_config": source_config, "inputs": [{"cohort": "test", "paths": inputs}],
         "model": "llama3.2:3b", "model_digest": "pinned", "cooldown_seconds": 0, "num_gpu": 0}
    config = write(tmp_path / "config.json", c)
    return tmp_path, rows, c, config


def test_prepare_union_dedupe_controls_and_no_answer_leakage(fixture):
    root, rows, c, config = fixture
    # A second episode returns the same evidence; each exact source span occurs once.
    c["inputs"][0]["paths"].append(write(root / "repeat.json", rows[0]))
    write(Path(config), c)
    m = dp.prepare(config, root / "out")
    assert len(m["cells"]) == 9
    assert [x["unique_passages"] for x in m["cells"]] == [1, 0, 1] * 3
    assert m["cells"][0]["episodes"] == 1
    assert m["cells"][0]["evidence_bytes"] == len(rows[0]["trace"][0]["retrieval"]["passages"][0]["text"].encode())
    wire = dp.packed(m)
    assert all(v not in wire for v in ["OLD_PROMPT_POISON", "OLD_RESPONSE_POISON", "OLD_ANSWER_POISON", "GOLD_POISON", "JUDGE_POISON"])
    for cell in m["cells"]:
        if cell["condition"] == "wrong_task":
            assert cell["donor_question_id"] != cell["question_id"]
    assert dp.prepare(config, root / "out")["fingerprint"] == m["fingerprint"]


@pytest.mark.parametrize("mutation", ["text", "hash", "bounds", "status", "source", "page"])
def test_reject_tampered_unverified_and_disallowed_spans(fixture, mutation):
    root, rows, c, config = fixture
    p = copy.deepcopy(rows[0]["trace"][0]["retrieval"]["passages"][0])
    if mutation == "text": p["text"] += "injected"
    if mutation == "hash": p["pointer"]["span_sha256"] = "0" * 64
    if mutation == "bounds": p["pointer"]["end_byte"] = 999999
    if mutation == "status": p["verification_status"] = "TRUST_ME"
    if mutation == "source": p["pointer"]["source"] = c["source_config"]
    if mutation == "page": p["pointer"]["page"] = 1
    allowed = {str((root / "0/source.txt").resolve())}
    with pytest.raises(ValueError): dp.verify_passage(p, allowed)


def test_changed_source_and_output_manifest_are_refused(fixture):
    root, rows, c, config = fixture
    dp.prepare(config, root / "out")
    c["num_batch"] = 32
    write(Path(config), c)
    with pytest.raises(ValueError, match="different manifest"): dp.prepare(config, root / "out")
    (root / "0/source.txt").write_text("changed")
    with pytest.raises(ValueError, match="source pin"): dp.prepare(config, root / "new")


def test_cross_corpus_wrong_control_requires_disjoint_sources(fixture):
    root, rows, c, config = fixture
    rows[1]["trace"] = rows[0]["trace"]
    rows[1]["task_id"] = rows[0]["task_id"]
    write(root / "row1.json", rows[1])
    with pytest.raises(ValueError, match="allowlist"): dp.prepare(config, root / "out")


def test_declared_wrong_corpus_rejects_overlap(fixture):
    root, _, c, config = fixture
    c["wrong_task_corpora"] = {"test": {"0:1": ["unavailable-corpus"]}}
    write(Path(config), c)
    with pytest.raises(ValueError, match="no saved spans"):
        dp.prepare(config, root / "out")


def test_config_and_runtime_permission_boundaries(fixture):
    root, _, c, config = fixture
    for field, value in [("model", "qwen3:0.6b"), ("num_gpu", -1), ("golden", "forbidden")]:
        bad = {**c, field: value}
        with pytest.raises(ValueError): dp.validate_config(bad)
    with pytest.raises(ValueError, match="caller"): dp.run(root / "does-not-exist")


def test_runtime_payload_is_one_raw_deterministic_call(fixture):
    _, _, c, _ = fixture
    p = dp.request_payload(dp.validate_config(c), "wire")
    assert p["raw"] and p["stream"] and p["keep_alive"] == 0
    assert p["options"] == {"seed": 1729, "temperature": 0, "num_ctx": 16384,
        "num_gpu": 0, "num_batch": 128, "num_predict": 2048, "num_thread": 4}


def test_termination_empty_and_token_truncation_rejected():
    r = {"raw_text": "answer", "response": {"done": True, "done_reason": "stop", "prompt_eval_count": 123, "eval_count": 5}}
    assert dp.admitted(r, 123)
    assert not dp.admitted(r, 124)
    assert not dp.admitted({**r, "raw_text": " "}, 123)
    for reason in ["length", None]:
        assert not dp.admitted({**r, "response": {**r["response"], "done_reason": reason}}, 123)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        class Encoded:
            ids = [1] * 122
        return Encoded()


def test_smoke_gate_resume_and_preserved_failure(fixture, monkeypatch):
    root, _, _, config = fixture
    out = root / "out"
    dp.prepare(config, out)
    monkeypatch.setattr(dp, "tokenizer_from_metadata", lambda info: FakeTokenizer())
    calls = []
    def call(c, prompt, emit):
        calls.append(prompt)
        emit({"event": "raw_chunk", "item": {"response": "partial"}})
        return {"raw_text": ("9143" if "9143 days of notice" in prompt else "6327") if "days of notice" in prompt else "answer",
                "response": {"done": True, "done_reason": "stop", "prompt_eval_count": 123, "eval_count": 5}}
    with pytest.raises(ValueError, match="smoke-only"):
        dp.run(out, True, call=call, info={})
    dp.run(out, True, True, call=call, info={})
    assert dp.load(out / "smoke.json")["pass"]
    dp.run(out, True, call=call, info={})
    assert len(calls) == 8
    dp.run(out, True, call=call, info={})
    assert len(calls) == 8
    assert dp.summarize(out)["responses"] == 6
    # Mutating a pinned trace cannot resume even when all cells are complete.
    (root / "row0.json").write_text("{}")
    with pytest.raises(ValueError, match="input/source changed"):
        dp.run(out, True, call=call, info={})


def test_transport_failure_saved_and_not_retried(fixture, monkeypatch):
    root, _, _, config = fixture
    out = root / "out"
    dp.prepare(config, out)
    monkeypatch.setattr(dp, "tokenizer_from_metadata", lambda info: FakeTokenizer())
    def fail(c, prompt, emit):
        emit({"event": "raw_chunk", "item": {"response": "partial"}})
        raise TimeoutError("injected timeout")
    with pytest.raises(TimeoutError): dp.run(out, True, True, call=fail, info={})
    assert "injected timeout" in (out / "failures.jsonl").read_text()
    assert "partial" in (out / "stream.jsonl").read_text()
    assert not (out / "RUN.lock").exists()
    with pytest.raises(RuntimeError, match="both source-value"):
        dp.run(out, True, True, call=lambda *args: pytest.fail("retried failure"), info={})


def test_overflow_refused_before_transport(fixture, monkeypatch):
    root, _, c, config = fixture
    c.update(num_ctx=1024, num_predict=1000)
    write(Path(config), c)
    out = root / "out"
    dp.prepare(config, out)
    monkeypatch.setattr(dp, "tokenizer_from_metadata", lambda info: FakeTokenizer())
    with pytest.raises(ValueError, match="no truncation"):
        dp.run(out, True, True, call=lambda *args: pytest.fail("inference on overflow"), info={})
    assert "overflow" in (out / "failures.jsonl").read_text()


def test_resource_stop_thresholds(monkeypatch):
    import psutil
    class RAM:
        available = 20 * 1024**3
    monkeypatch.setattr(psutil, "virtual_memory", lambda: RAM())
    with pytest.raises(RuntimeError, match="RAM"): dp.resource_guard(0)
    RAM.available += 1
    class Result:
        stdout = "66\n"
    monkeypatch.setattr(dp.subprocess, "run", lambda *a, **k: Result())
    with pytest.raises(RuntimeError, match="66C"): dp.resource_guard(1)


def test_metadata_rejects_4b_or_changed_digest(monkeypatch):
    c = {"model": "llama3.2:3b", "model_digest": "pin"}
    info = {"model_info": {"general.parameter_count": 4_000_000_000,
        "general.basename": "Llama-3.2", "llama.context_length": 131072, "tokenizer.ggml.pre": "llama-bpe"}}
    monkeypatch.setattr(dp, "api", lambda route, *a: {"models": [{"name": c["model"], "digest": "pin"}]} if route == "tags" else info)
    with pytest.raises(ValueError, match="metadata"): dp.model_metadata(c)
    info["model_info"]["general.parameter_count"] = 3212749888
    assert dp.model_metadata(c) == info
    c["model_digest"] = "changed"
    with pytest.raises(ValueError): dp.model_metadata(c)


def test_claims_atomic_foreign_and_own_bytes(tmp_path, monkeypatch):
    paths = (tmp_path / "first", tmp_path / "second")
    monkeypatch.setattr(dp, "CLAIM_PATHS", paths)
    paths[1].write_bytes(b"foreign")
    with pytest.raises(FileExistsError): dp.acquire_claims()
    assert not paths[0].exists() and paths[1].read_bytes() == b"foreign"
    paths[1].unlink()
    claims = dp.acquire_claims()
    paths[1].write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="ownership"): dp.release_claims(claims)
    assert paths[0].exists()


def test_conditions_limit_and_episode_provenance(fixture):
    root, rows, c, config = fixture
    c.update(conditions=["own"], limit=1)
    write(Path(config), c)
    m = dp.prepare(config, root / "out")
    assert len(m["cells"]) == 1 and m["cells"][0]["condition"] == "own"
    assert m["cells"][0]["source_episode"].endswith("row0.json")


def test_failed_request_retains_claim_until_real_idle(fixture, monkeypatch):
    root, _, c, config = fixture
    c["num_gpu"] = 1
    write(Path(config), c)
    out = root / "out"
    dp.prepare(config, out)
    monkeypatch.setattr(dp, "tokenizer_from_metadata", lambda _: FakeTokenizer())
    monkeypatch.setattr(dp, "wait_idle", lambda: (_ for _ in ()).throw(RuntimeError("still busy")))
    with pytest.raises(RuntimeError, match="still busy"):
        dp.run(out, True, True, call=lambda *a: (_ for _ in ()).throw(TimeoutError("request")), info={})
    assert all(p.exists() for p in dp.CLAIM_PATHS)
    assert (out / "RUN.lock").exists()


def test_original_selector_keeps_episodes_separate(fixture):
    root, rows, c, config = fixture
    c["inputs"][0]["cohort"] = "original36"
    for i, row in enumerate(rows):
        row.update(reader="qwen3:0.6b", arm="bm25", cap=3, terminated=True)
        write(root / f"row{i}.json", row)
    second = copy.deepcopy(rows[0])
    second.update(arm="hybrid", cap=5)
    c["inputs"][0]["paths"].append(write(root / "second.json", second))
    rejected = copy.deepcopy(rows[0])
    rejected.update(reader="phi3:mini")
    c["inputs"][0]["paths"].append(write(root / "rejected.json", rejected))
    write(Path(config), c)
    own = [x for x in dp.prepare(config, root / "out")["cells"] if x["condition"] == "own"]
    assert len(own) == 3
    assert {(x["source_arm"], x["source_cap"]) for x in own} == {("bm25", 3), ("hybrid", 5)}
    assert all(x["episodes"] == 1 for x in own)


class FakeClock:
    def __init__(self):
        self.now = 0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        assert 0 < seconds <= 2
        self.now += seconds


@pytest.mark.parametrize("hot_at,finish", [(None, 30), (20, 52)])
def test_cooldown_requires_continuous_30_seconds(monkeypatch, hot_at, finish):
    clock = FakeClock()
    monkeypatch.setattr(dp.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(dp.time, "sleep", clock.sleep)
    monkeypatch.setattr(dp, "resource_guard", lambda gpu: {
        "gpu_temperatures_c": [61 if clock.now == hot_at else 60]})
    dp.wait_cooldown(1)
    assert clock.now == finish


def test_hot_cooldown_times_out_at_150(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(dp.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(dp.time, "sleep", clock.sleep)
    monkeypatch.setattr(dp, "resource_guard", lambda gpu: {"gpu_temperatures_c": [61]})
    with pytest.raises(TimeoutError, match="150s"):
        dp.wait_cooldown(1)
    assert clock.now == 150


def test_cooldown_timeout_preserves_answer_and_releases_own_claims(fixture, monkeypatch):
    root, _, c, config = fixture
    c["num_gpu"] = 1
    write(Path(config), c)
    out = root / "out"
    dp.prepare(config, out)
    monkeypatch.setattr(dp, "tokenizer_from_metadata", lambda _: FakeTokenizer())
    clock = FakeClock()
    monkeypatch.setattr(dp.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(dp.time, "sleep", clock.sleep)
    monkeypatch.setattr(dp, "resource_guard", lambda gpu: {"gpu_temperatures_c": [61]})
    idle_calls = []
    monkeypatch.setattr(dp, "wait_idle", lambda: idle_calls.append(clock.now))
    calls = []
    def call(*args):
        calls.append(True)
        return {"raw_text": "6327", "response": {
            "done": True, "done_reason": "stop", "prompt_eval_count": 123, "eval_count": 5}}
    with pytest.raises(TimeoutError, match="150s"):
        dp.run(out, True, True, call=call, info={})
    records = dp.read_rows(out / "responses.jsonl")
    assert len(calls) == len(records) == 1
    assert records[0]["raw_text"] == "6327" and records[0]["admitted"]
    assert idle_calls == [0, 150]
    assert not any(p.exists() for p in dp.CLAIM_PATHS)
    assert not (out / "RUN.lock").exists()
    assert not (out / "smoke.json").exists()


def test_excluded_cross_corpus_contains_only_verified_provenance(fixture):
    root, rows, c, config = fixture
    foreign = copy.deepcopy(rows[1]["trace"][0]["retrieval"]["passages"][0])
    source_config = dp.load(c["source_config"])
    source_config["tasks"][0]["source_roots"].append(str(root / "1"))
    source_config["tasks"][0]["allowed_paths"].append(foreign["pointer"]["source"])
    write(Path(c["source_config"]), source_config)
    rows[0]["trace"][0]["retrieval"]["passages"].extend([foreign, foreign])
    write(root / "row0.json", rows[0])
    m = dp.prepare(config, root / "out")
    for cell in m["cells"]:
        if cell["question_id"] != "0:1":
            continue
        excluded = cell["excluded_cross_corpus"]
        assert len(excluded) == 2
        for record in excluded:
            assert set(record) == {"pointer", "verification_status"}
            assert record["pointer"] == foreign["pointer"]
            assert record["verification_status"] == "VERIFIED_SOURCE_BYTES"
        assert foreign["text"] not in dp.packed(excluded)
    assert m["sources"][foreign["pointer"]["source"]] == foreign["pointer"]["source_sha256"]
    assert m["input_record_counts"][str((root / "row0.json").resolve())] == 1
    assert m["input_pins"][str((root / "row0.json").resolve())] == dp.sha((root / "row0.json").read_bytes())


def test_cpu_timeout_and_fixed_safety_config(fixture):
    _, _, c, _ = fixture
    cpu = dp.validate_config({**c, "timeout_seconds": 180})
    assert cpu["num_gpu"] == 0 and cpu["num_thread"] == 4
    assert cpu["cooldown_seconds"] == 10
    assert dp.request_payload(cpu, "wire")["options"]["num_gpu"] == 0
    with pytest.raises(ValueError, match="timeout_seconds"):
        dp.validate_config({**c, "num_gpu": 1, "timeout_seconds": 180})
    with pytest.raises(ValueError, match="num_thread"):
        dp.validate_config({**c, "num_thread": 8})


@pytest.mark.parametrize("failure", [False, True])
def test_cpu_only_service_claim_leaves_foreign_gpu_untouched(fixture, monkeypatch, failure):
    root, _, c, config = fixture
    c["timeout_seconds"] = 180
    write(Path(config), c)
    out = root / "cpu"
    dp.prepare(config, out)
    gpu, service = dp.CLAIM_PATHS
    gpu.write_bytes(b"foreign ABORTED_CLEANUP_UNVERIFIED")
    before = gpu.stat().st_mtime_ns
    monkeypatch.setattr(dp, "tokenizer_from_metadata", lambda _: FakeTokenizer())
    monkeypatch.setattr(dp.subprocess, "run", lambda *a, **k: pytest.fail("GPU command"))
    monkeypatch.setattr(dp, "wait_cooldown", lambda *a: pytest.fail("GPU cooldown"))
    sleeps = []
    monkeypatch.setattr(dp.time, "sleep", sleeps.append)
    def call(config, prompt, emit):
        assert config["num_gpu"] == 0 and service.exists()
        assert gpu.read_bytes() == b"foreign ABORTED_CLEANUP_UNVERIFIED"
        if failure:
            raise TimeoutError("CPU transport failed")
        return {"raw_text": "9143" if "9143 days of notice" in prompt else "6327",
                "response": {"done": True, "done_reason": "stop", "prompt_eval_count": 123, "eval_count": 5}}
    if failure:
        with pytest.raises(TimeoutError, match="CPU transport"):
            dp.run(out, True, True, call=call, info={})
    else:
        dp.run(out, True, True, call=call, info={})
        assert sleeps == [10, 10]
    assert not service.exists()
    assert gpu.read_bytes() == b"foreign ABORTED_CLEANUP_UNVERIFIED"
    assert gpu.stat().st_mtime_ns == before


def test_cpu_claim_without_gpu_lease(tmp_path):
    claims = dp.acquire_claims(0)
    assert [p for p, _ in claims] == [dp.CLAIM_PATHS[1]]
    assert not dp.CLAIM_PATHS[0].exists()
    dp.release_claims(claims)


def test_bos_accounting_special_tokens_and_exact_counter():
    from tokenizers import AddedToken, Tokenizer, models
    specials = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, **{t: i for i, t in enumerate(specials, 1)}}, unk_token="[UNK]"))
    tok.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in specials])
    for i, token in enumerate(specials, 1):
        assert tok.encode(token, add_special_tokens=False).ids == [i]
    prompt = dp.render("Question?", [])
    assert not prompt.startswith(specials[0])
    ids = tok.encode(prompt, add_special_tokens=False).ids
    assert ids.count(2) == ids.count(3) == 3 and ids.count(4) == 2
    accounting = dp.prompt_accounting(tok, prompt)
    assert accounting["wire_prompt_tokens"] == len(ids)
    assert accounting["runtime_bos_tokens"] == 1
    assert accounting["expected_prompt_tokens"] == len(ids) + 1
    result = {"raw_text": "6327", "response": {"done": True, "done_reason": "stop", "eval_count": 2,
              "prompt_eval_count": len(ids) + 1}}
    assert dp.admitted(result, len(ids) + 1)
    assert not dp.admitted(result, len(ids))
    assert not dp.admitted(result, len(ids) + 2)
    with pytest.raises(ValueError, match="explicit BOS"):
        dp.prompt_accounting(tok, specials[0] + prompt)


def test_numeric_source_intervention_frozen_targets():
    jobs = dp.source_value_controls()
    assert list(dp.CONTROL_TARGETS.values()) == ["6327", "9143"]
    a, b = [job["prompt"] for job in jobs]
    assert a.replace("6327", "9143") == b
    assert a.count("6327") == b.count("9143") == 1
    assert "Return digits only" in a
    assert "6327" not in a.split("EVIDENCE (coverage unknown):")[0]
    assert "9143" not in b.split("EVIDENCE (coverage unknown):")[0]


def test_constant_numeric_answer_fails_second_source_value(fixture, monkeypatch):
    root, _, _, config = fixture
    out = root / "numeric"
    dp.prepare(config, out)
    monkeypatch.setattr(dp, "tokenizer_from_metadata", lambda _: FakeTokenizer())
    calls = []
    def call(*args):
        calls.append(True)
        return {"raw_text": "6327", "response": {"done": True, "done_reason": "stop",
                "prompt_eval_count": 123, "eval_count": 2}}
    with pytest.raises(RuntimeError, match="positive-control"):
        dp.run(out, True, True, call=call, info={})
    assert len(calls) == 2
    assert not (out / "smoke.json").exists()
    assert "positive-control" in (out / "failures.jsonl").read_text()


def test_metadata_tokenizer_specials_unicode_and_accounting():
    from tokenizers.pre_tokenizers import ByteLevel
    specials = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
    alphabet = sorted(ByteLevel.alphabet())
    tokens = alphabet + specials
    tok = dp.tokenizer_from_metadata({"model_info": {
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": [1] * len(alphabet) + [3] * len(specials),
        "tokenizer.ggml.merges": []}})
    for i, token in enumerate(specials, len(alphabet)):
        assert tok.encode(token, add_special_tokens=False).ids == [i]
    prompt = dp.render("café 6327?", [])
    ids = tok.encode(prompt, add_special_tokens=False).ids
    assert ids.count(len(alphabet) + 1) == 3
    assert ids.count(len(alphabet) + 2) == 3
    assert ids.count(len(alphabet) + 3) == 2
    assert len(alphabet) not in ids
    # No postprocessor silently adds BOS; UTF-8 byte handling is explicit.
    assert tok.encode("café", add_special_tokens=False).ids == tok.encode("café").ids
    assert len(tok.encode("café", add_special_tokens=False).ids) == len("café".encode())
    assert dp.prompt_accounting(tok, prompt)["expected_prompt_tokens"] == len(ids) + 1
