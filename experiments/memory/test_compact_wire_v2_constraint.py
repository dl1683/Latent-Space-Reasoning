from collections import deque
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.memory import compact_task_planner_wire_v2 as wire
from experiments.memory.compact_wire_v2_constraint import CompactWireV2Constraint
from experiments.memory.task_episode import EpisodeLimits, EpisodePolicy, PolicyCall, _decode, _packed
from experiments.memory.task_evidence import TaskBudget
from experiments.memory.test_compact_episode_constraint import _CharacterTokenizer
from experiments.memory.test_scoped_episode_constraint import _scripted_backbone


_LIMITS = EpisodeLimits(7, 6, 6, 0, 8192, 8192, 1792, 512, access=TaskBudget(candidates_per_search=1))
_NAMES = ("Zircon", "Copper", "Hazel", "Osprey", "Marble", "Willow")


def _need(name):
    return name + ";confirm;current;code;receipt;-;-;-"


def _response(frame):
    phase = frame["planning_phase"]
    if phase == "declare":
        return {"p": [{"x": _need(name)} for name in _NAMES[:3]]}
    receipts = wire.unpack_receipts(frame)
    count = len(receipts)
    if count in (0, 6):
        start = count // 2
        children = [{"p": f"n{i - 2}", "d": str(i - 4), "x": _need(_NAMES[i - 1])}
                    for i in range(4, 7)] if start else []
        return {"a": children, "t": [{"op": "search", "need_id": f"n{i}", "query": _NAMES[i - 1]}
                                     for i in range(start + 1, start + 4)]}
    if count in (3, 9):
        return {"a": [], "t": [{"op": "read", "need_id": row["request"]["need_id"],
                               "candidate": row["result"]["candidates"][0]["candidate"]}
                              for row in receipts[-3:]]}
    start = 3 if phase == "finalize" else 0
    return {"f": [{"i": f"n{i + 1}", "s": "s", "c": f"{_NAMES[i]} code is {i + 1}.", "r": "Current registry",
                   "v": "0,3" if i == 0 else str(i), "q": ""} for i in range(start, start + 3)]}


def _policy(constraint):
    return EpisodePolicy("fixture", "ordinary-fixture", "pinned-tokenizer-fixture", "compact_needs",
                         output_constraint_sha256=constraint.sha256, compact_delivery=True,
                         compact_wire_version=2, compact_wire_sha256=wire.WIRE_SHA256)


def _frame(phase, receipts, *names):
    return {"compact_wire_version": 2, "state": "", "planning_phase": phase, "compact_search_limit": 6,
            "compact_read_limit": 6, "need_registry": wire.normalize_plan([{"x": _need(name)} for name in names]),
            "receipts": receipts}


def _call(frame):
    return PolicyCall((("user", _packed(wire.project_prompt(frame))),), (ord("X"),), 512)


def test_projection_preserves_receipts_and_only_successful_search_authorizes_candidates():
    span = {"span_id": 3, "source_version": 2, "source_id": "test"}
    search = {"request": {"op": "search", "need_id": "n1", "query": "Zircon"},
              "result": {"status": "SEARCHED", "matched": 1, "candidates": [{"candidate": 4, "span": span}]}}
    read = {"request": {"op": "read", "need_id": "n1", "candidate": 4},
            "result": {"status": "VERIFIED_SOURCE_BYTES", "id": 1, "span": span, "text": 'Exact "\u03b2"\n',
                       "source_bytes_read": 11}}
    fault = {"request": {"op": "search", "need_id": "n2", "query": "Copper"},
             "result": {"status": "SOURCE_ERROR", "candidates": [{"candidate": 9, "span": span}]}}
    failed_read = {**read, "result": {"status": "SOURCE_ERROR"}}
    frame = _frame("investigate", [search, failed_read, fault, read], "Zircon", "Copper")
    projected = wire.project_prompt(frame)
    assert projected["receipt_spans"] == {"3": span}
    assert projected["need_registry"] == [{"id": f"n{i}", "p": None, "d": "", "x": _need(name)}
                                          for i, name in enumerate(_NAMES[:2], 1)]
    assert wire.unpack_receipts(projected) == frame["receipts"]
    facts = wire._receipts(projected)
    assert facts["candidates"] == {"n1": {4}} and facts["verified"] == [1]
    assert (facts["searches"], facts["reads"]) == (2, 2)
    assert not wire._receipts(frame)["candidates"]["n1"]


def test_constraint_rejects_bad_need_and_unguarded_prefixes():
    pytest.importorskip("lmformatenforcer")
    pytest.importorskip("jsonschema")
    tokenizer = _CharacterTokenizer()
    constraint = CompactWireV2Constraint(tokenizer, _LIMITS)
    call = _call(_frame("declare", [], "Zircon"))
    allow = constraint(call)
    with pytest.raises(ValueError, match="exact prepared prefix"):
        allow((ord("Y"),))
    assert tokenizer.eos_token_id not in allow(call.input_ids)
    with pytest.raises(ValueError, match="one permitted next token"):
        allow((*call.input_ids, tokenizer.eos_token_id))
    prefix = call.input_ids
    for token in tokenizer.encode(_packed({"p": [{"x": "only one slot"}]})):
        assert token in allow(prefix)
        prefix = (*prefix, token)
    with pytest.raises(ValueError, match="invalid output"):
        allow(prefix)


def test_constraint_accepts_legal_spaces_in_a_search_query():
    pytest.importorskip("lmformatenforcer")
    pytest.importorskip("jsonschema")
    tokenizer = _CharacterTokenizer()
    constraint = CompactWireV2Constraint(tokenizer, _LIMITS)
    call = _call(_frame("act_after_plan", [], "Zircon"))
    action = {"a": [], "t": [{"op": "search", "need_id": "n1", "query": "Zircon" + " " * 20 + "code"}]}
    allow = constraint(call)
    prefix = call.input_ids
    for token in tokenizer.encode(_packed(action)):
        assert token in allow(prefix)
        prefix = (*prefix, token)
    assert tokenizer.eos_token_id in allow(prefix)


def test_caller_supplied_v2_backend_keeps_its_object_prompt(tmp_path):
    from experiments.memory.source_revision_feed import write_source_delta
    from experiments.memory.task_episode import PolicyReply, read_episode, run_episode
    from experiments.memory.task_evidence import TaskWorld

    feed = tmp_path / "source"
    source = write_source_delta(feed, {"1" * 32: b"Zircon code is 1."})
    world = TaskWorld.from_source_epoch(feed, source.seal, max_feed_bytes=1024, max_source_bytes=1024, max_passages=2)
    policy = EpisodePolicy("fixture", "caller-v2", "fixture-characters", "compact_needs",
                           compact_delivery=True, compact_wire_version=2, compact_wire_sha256=wire.WIRE_SHA256)
    phases = []

    def backend(call):
        frame = _decode(dict(call.messages)["user"])
        phases.append(frame["planning_phase"])
        assert "receipt_spans" not in frame
        if phases[-1] == "act_after_plan":
            assert frame["need_registry"][0]["target_or_event"] == "Zircon"
            assert dict(call.messages)["system"] == wire.system_prompt("act_after_plan")
        reply = _packed(_response(frame)) if phases[-1] == "declare" else _packed(
            {"a": [], "t": [{"op": "search", "need_id": "n1", "query": "Zircon"}]}
        )
        return PolicyReply(reply, len(reply) + 1, True)

    artifact = run_episode("Confirm Zircon.", world, policy, replace(_LIMITS, calls=2), _CharacterTokenizer(),
                           backend, tmp_path / "caller")
    assert phases == ["declare", "act_after_plan"]
    assert read_episode(artifact)[-1]["data"]["usage"]["attempts"]["search"] == 1


def test_cached_six_source_route_runs_through_both_adapters(tmp_path, monkeypatch):
    import numpy as np

    from experiments.memory.linear_feedback import LinearFeedbackKernel
    from experiments.memory.native_latent_recurrence import NativeLatentModel
    from experiments.memory.native_publication import read_native_publication, write_native_publication
    from experiments.memory.native_task_episode import NativeTaskEpisodeBackend
    from experiments.memory.source_revision_feed import write_source_delta
    from experiments.memory.task_episode import read_episode_pair, run_episode
    from experiments.memory.task_evidence import TaskWorld
    from experiments.memory.verified_source_tool import sha256

    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    pytest.importorskip("lmformatenforcer")
    pytest.importorskip("jsonschema")
    revision = "a10cc1512eabd3dde888204e902eca88bddb4951"
    snapshot = (Path.home() / ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots"
                / revision)
    if not (snapshot / "tokenizer.json").is_file():
        pytest.skip("pinned tokenizer-only snapshot is not cached")
    tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    eos = tokenizer.eos_token_id
    constraint = CompactWireV2Constraint(tokenizer, _LIMITS)
    feed = tmp_path / "sources"
    source = write_source_delta(feed, {str(i) * 32: f"{name} code is {i}. Signed registry.".encode()
                                       for i, name in enumerate(_NAMES, 1)})
    world = TaskWorld.from_source_epoch(feed, source.seal, max_feed_bytes=8192, max_source_bytes=8192, max_passages=8)
    plain_model = _scripted_backbone(tokenizer, _LIMITS.context_tokens)
    native_model = _scripted_backbone(tokenizer, _LIMITS.context_tokens)
    axes = np.ones((1, 1))
    kernel = LinearFeedbackKernel(query=axes, carry=None, observation_key=axes, observation_value=axes,
                                  feedback_key=axes, feedback_value=axes, max_steps=1)
    reader = NativeLatentModel(native_model, tokenizer, kernel, source_chars=128, context_tokens=_LIMITS.context_tokens,
                               latent_steps=1, output_tokens=_LIMITS.max_new_tokens, representation_scale=1,
                               memory_gain=1, null_weight=1, eos_ids=(eos,), top_k=1)
    runtime = sha256(b"v2 common decoder fixture")
    location = tmp_path / "publication"
    seal = write_native_publication(location, reader, reader.publish(feed, source.seal, max_source_bytes=8192),
                                    runtime_sha256=runtime, max_bytes=8192)
    memory = read_native_publication(location, reader, seal=seal, source_seal=source.seal,
                                     runtime_sha256=runtime, max_bytes=8192)
    policy = _policy(constraint)
    native_policy = replace(policy, backend_id="native-fixture")
    native = NativeTaskEpisodeBackend(world, reader, memory, backend_id=native_policy.backend_id,
                                      tokenizer_id=native_policy.tokenizer_id, expected_runtime_sha256=runtime,
                                      output_token_allowlist=constraint, output_constraint_sha256=constraint.sha256,
                                      steps=0)
    original_allow = CompactWireV2Constraint.__call__
    scripts, traces = [], [[], []]
    arm = 0

    def scripted_allow(self, call):
        frame = _decode(dict(call.messages)["user"])
        text = _packed(_response(frame))
        index = len(traces[arm])
        if arm == 0:
            walk = original_allow(self, call)
            prefix = call.input_ids
            remaining = deque([*tokenizer.encode(text, add_special_tokens=False), eos])
            while remaining:
                token = remaining.popleft()
                if token in walk(prefix):
                    prefix = (*prefix, token)
                    continue
                fragment = tokenizer.decode([token])
                assert token != eos and len(fragment) > 1
                pieces = [piece for char in fragment for piece in tokenizer.encode(char, add_special_tokens=False)]
                assert len(pieces) > 1 and tokenizer.decode(pieces) == fragment
                remaining.extendleft(reversed(pieces))
            scripts.append(prefix[len(call.input_ids):])
        ids = scripts[index]
        assert ids[-1] == eos and len(ids) <= call.max_new_tokens
        assert tokenizer.decode(ids[:-1]) == text
        (plain_model if arm == 0 else native_model).script = ids
        trace = []
        traces[arm].append(trace)
        allowed = original_allow(self, call)

        def record(prefix):
            choices = allowed(prefix)
            assert ids[len(prefix) - len(call.input_ids)] in choices
            trace.append(choices)
            return choices

        return record

    monkeypatch.setattr(CompactWireV2Constraint, "__call__", scripted_allow)
    task = "Confirm the six registry codes and cite the current records."
    plain = run_episode(task, world, policy, _LIMITS, tokenizer, constraint.plain_backend(plain_model),
                        tmp_path / "ordinary")
    arm = 1
    attached = run_episode(task, world, native_policy, _LIMITS, tokenizer, native, tmp_path / "native",
                           native_backend=native)
    left, right = read_episode_pair(plain, attached, contrast="native_text")
    assert traces[0] == traces[1] and len(scripts) == len(traces[0]) == 7
    expected_citations = [[0, 3], [1], [2], [3], [4], [5]]
    for records in (left, right):
        finish = records[-1]["data"]
        assert finish["status"] == "completed"
        assert [item["citations"] for item in finish["submission"]["findings"]] == expected_citations
        usage = finish["usage"]
        assert usage["attempts"] == {"search": 6, "read": 6, "probe": 0}
        assert usage["source_bytes"] == 203 and usage["calls"] == 7
        assert usage["input_tokens"] <= _LIMITS.input_tokens
        assert usage["known_output_tokens"] == sum(map(len, scripts)) <= _LIMITS.output_tokens
    for row, ids in zip((row["data"] for row in right if row["kind"] == "response"), scripts, strict=True):
        assert row["backend_receipt"]["output_token_ids"] == list(ids) and row["terminated"]
    print(f"matched v2 replay: input={left[-1]['data']['usage']['input_tokens']}, output={sum(map(len, scripts))}")
