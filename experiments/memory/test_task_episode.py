from dataclasses import replace
import json

import numpy as np
import pytest

from experiments.memory.corpus_binding_journal import JournaledCorpusWorkspace
from experiments.memory.corpus_bindings import BindingMention, CorpusBindingMemory
from experiments.memory.task_episode import (
    EpisodeLimits, EpisodePolicy, PolicyReply, read_episode, read_episode_pair, run_episode,
)
from experiments.memory.task_evidence import TaskWorld
from experiments.memory.test_task_evidence import _source_tool, _write


TASK = "Can we proceed this evening?"
LIMITS = EpisodeLimits(8, 3, 3, 1, 8192, 45000, 4096, 1024)
POLICY = EpisodePolicy("controller", "fixture-backend", "fixture-tokenizer", "notes")


class Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
        assert tokenize and add_generation_prompt and not return_dict
        return list(json.dumps(messages, ensure_ascii=True).encode())


def world(root, verdict="cleared"):
    tool, spans = _source_tool(root, {
        "rules": "Release requires Cobalt approval and a confirmed duty assignment.",
        "routing": "Cobalt approval is recorded under Juniper in the separate review log.",
        "log": f"Juniper review log: approval {verdict}.",
    })
    return TaskWorld(tool, spans)


def reply(**value):
    text = json.dumps(value)
    return PolicyReply(text, len(text.encode()), True)


def investigate(kind):
    def generate(call):
        frame = json.loads(dict(call.messages)["user"])
        assert frame["task"] == TASK
        assert "pin" not in frame
        receipts = frame["receipts"]
        step = len(receipts)
        state = {"needs": "approval and duty"} if kind == "needs" else "Check approval and duty."
        if step == 0:
            return reply(state=state, tool={"op": "search", "query": "release"})
        if step == 2:
            assert "Cobalt" in receipts[-1]["result"]["text"]
            return reply(state=state, tool={"op": "search", "query": "Cobalt recorded"})
        if step == 4:
            assert "Juniper" in receipts[-1]["result"]["text"]
            return reply(state=state, tool={"op": "search", "query": "Juniper review log"})
        if step == 6:
            log = receipts[-1]["result"]["text"]
            assert "approval cleared" in log or "approval withheld" in log
            return reply(finish={"findings": [
                {"claim": "Cobalt approved the release.", "stance": "supports" if "approval cleared" in log else "defeats",
                 "citations": [0, 1, 2], "reason": "The rules require approval and the routing record identifies the log."},
                {"claim": "Duty assignment is established.", "stance": "unknown", "citations": [0],
                 "reason": "The rule specifies the requirement. The assignment record is still needed."},
            ], "open_questions": ["Who is assigned to duty?"]})
        source = {1: "rules", 3: "routing", 5: "log"}[step]
        candidate = next(row["candidate"] for row in receipts[-1]["result"]["candidates"]
                         if row["span"]["source_id"] == source)
        return reply(state=state, tool={"op": "read", "candidate": candidate})

    return generate


def test_matched_policies_follow_indirect_sources_and_charge_repeated_context(tmp_path):
    pinned = world(tmp_path)
    structured = replace(POLICY, policy_id="structured", state_kind="needs")
    notes = investigate("notes")
    left = run_episode(TASK, pinned, POLICY, LIMITS, Tokenizer(), notes, tmp_path / "left")
    right = run_episode(TASK, pinned, structured, LIMITS, Tokenizer(), investigate("needs"), tmp_path / "right")
    a, b = read_episode_pair(left, right)
    assert [event["data"] for event in a if event["kind"] == "tool"] == [
        event["data"] for event in b if event["kind"] == "tool"
    ]
    for records in (a, b):
        finish = records[-1]["data"]
        assert finish["status"] == "completed"
        assert finish["submission"]["findings"][0]["citations"] == [0, 1, 2]
        assert finish["submission"]["open_questions"] == ["Who is assigned to duty?"]
        requests = [row["data"] for row in records if row["kind"] == "request"]
        assert len(requests) == finish["usage"]["calls"] == 7
        replies = [row["data"]["text"] for row in records if row["kind"] == "response"]
        assert finish["usage"]["known_output_tokens"] == sum(len(text.encode()) for text in replies)
        assert finish["usage"]["input_tokens"] == sum(len(row["input_ids"]) for row in requests)
        assert finish["usage"]["input_tokens"] > len(requests[-1]["input_ids"])
        assert finish["usage"]["attempts"] == {"search": 3, "read": 3, "probe": 0}
    assert a[-1]["data"]["submission"]["findings"][0]["stance"] == "supports"
    corrected = run_episode(TASK, world(tmp_path, "withheld"), POLICY, LIMITS, Tokenizer(), notes,
                            tmp_path / "corrected")
    assert read_episode(corrected)[-1]["data"]["submission"]["findings"][0]["stance"] == "defeats"
    with pytest.raises(ValueError, match="pin differs"):
        read_episode_pair(left, corrected)
    with pytest.raises(ValueError, match="same source pin"):
        read_episode_pair(left, corrected, contrast="memory")


def test_attempt_limits_invalid_citations_and_probe_contrast_are_explicit(tmp_path):
    pinned = world(tmp_path)

    def actions():
        yield reply(state="", tool={"op": "read", "candidate": False})
        yield reply(state="", tool={"op": "read", "candidate": 0})
        yield reply(state="", tool={"op": "probe", "read": 0, "ordinal": 0, "outcomes": {"cleared": 0}})
        yield reply(state="", tool={"op": "probe", "read": 0, "ordinal": 0, "outcomes": ["cleared"], "hops": True})
        yield reply(state="", tool={"op": "probe", "read": 0, "ordinal": 0, "outcomes": ["cleared"]})
        yield reply(finish={"findings": [{"claim": "x", "stance": "supports", "reason": "x", "citations": [0]}],
                            "open_questions": []})
        yield reply(finish={"findings": [], "open_questions": ["Where is approval recorded?"]})

    def run(policy, name):
        stream = actions()
        return run_episode(TASK, pinned, policy, replace(LIMITS, reads=1, probes=3), Tokenizer(),
                           lambda _: next(stream), tmp_path / name)

    off = run(POLICY, "off")
    on = run(replace(POLICY, probes_enabled=True), "on")
    with pytest.raises(ValueError, match="equal probe access"):
        read_episode_pair(off, on)
    first, second = read_episode_pair(off, on, contrast="probes")
    tools = [row["data"]["result"]["status"] for row in first if row["kind"] == "tool"]
    assert tools == ["INVALID_TOOL_ARGUMENTS", "TOOL_ATTEMPT_LIMIT"] + ["PROBE_DISABLED"] * 3
    enabled = [row["data"]["result"] for row in second if row["kind"] == "tool"]
    for result in enabled[2:4]:
        assert result == {"status": "INVALID_TOOL_ARGUMENTS", "reason": "probe requires an outcome list and integer hops"}
    assert first[-1]["data"]["usage"]["attempts"] == {"search": 0, "read": 2, "probe": 3}
    assert sum(row["kind"] == "rejection" for row in first) == 1
    assert first[-1]["data"]["status"] == "completed"


def test_later_task_can_use_query_blind_program_from_recovered_world(tmp_path):
    records = {"routing": "Cobalt approval is under Juniper.", "log": "Juniper approval withheld."}
    axes = np.eye(3)

    def encode(raw):
        if raw.startswith(b"Cobalt"):
            yield BindingMention("entry", axes[0], .9)
            yield BindingMention("exit", axes[1], .9)
        else:
            yield BindingMention("withheld", axes[1], .8)

    frame = CorpusBindingMemory.prepare(("entry", "exit", "cleared", "withheld"), axes, {}, {},
                                        encode=encode, encoder_version="episode-fixture", min_similarity=.9, gap=.05)
    path = tmp_path / "world.sqlite"
    owner = JournaledCorpusWorkspace(path, frame, create=True)
    _write(owner, {source: raw.encode() for source, raw in records.items()})
    worker = JournaledCorpusWorkspace(path, frame)
    worker.publish_program(worker.prepare_role_links(worker.snapshot(), ("entry", "exit"),
                                                    min_similarity=.95, compile_work_limit=64))
    tool, spans = _source_tool(tmp_path, records)
    task = "Which record governs this approval?"
    policy = replace(POLICY, probes_enabled=True)

    def generate(call):
        receipts = json.loads(dict(call.messages)["user"])["receipts"]
        if not receipts:
            return reply(state="Find the Cobalt routing record.", tool={"op": "search", "query": "Cobalt"})
        if len(receipts) == 1:
            return reply(state="Verify the candidate.",
                         tool={"op": "read", "candidate": receipts[-1]["result"]["candidates"][0]["candidate"]})
        if len(receipts) == 2:
            result = receipts[-1]["result"]
            if not result["observation_predicates"]:
                return reply(finish={"findings": [], "open_questions": ["Where is the approval decision recorded?"]})
            return reply(state="Check the attached route.",
                         tool={"op": "probe", "read": result["id"],
                               "ordinal": result["observation_predicates"].index("entry"),
                               "outcomes": ["cleared", "withheld"]})
        if len(receipts) == 3:
            candidates = receipts[-1]["result"]["candidates"]
            if not candidates:
                return reply(finish={"findings": [], "open_questions": ["Where is the approval decision recorded?"]})
            assert candidates[0]["span"]["source_id"] == "log"
            return reply(state="Verify the proposed terminal.", tool={"op": "read", "candidate": candidates[0]["candidate"]})
        assert "Juniper approval withheld" in receipts[-1]["result"]["text"]
        return reply(finish={"findings": [{"claim": "Cobalt approved the release.", "stance": "defeats",
                                         "reason": "The routing record names Juniper. The Juniper log withholds approval.",
                                         "citations": [0, 1]}], "open_questions": []})

    def probe(pin, name):
        pinned = TaskWorld(tool, spans, bindings=pin)
        artifact = run_episode(task, pinned, policy, LIMITS, Tokenizer(), generate, tmp_path / name)
        ledger = read_episode(artifact)
        event = next(row["data"] for row in ledger if row["kind"] == "tool"
                     and row["data"]["request"]["op"] == "probe")
        finish = ledger[-1]["data"]
        assert finish["status"] == "completed" and finish["usage"]["attempts"]["search"] == 1
        return artifact, event["result"]["read"], event["usage"]["role_work"], finish["submission"]

    reopened = JournaledCorpusWorkspace(path, frame)
    retained = reopened.snapshot()
    maintained, result, work, report = probe(retained, "program")
    assert result["status"] == "supported" and result["witnesses"][0]["outcome"] == "withheld" and work > 0
    assert report["findings"][0]["citations"] == [0, 1]
    removed = reopened.publish_program(reopened.prepare_role_links(retained, None))
    assert removed.memory.source_digests == retained.memory.source_digests
    withdrawn, result, work, report = probe(removed, "removed")
    assert result["status"] == "no_role_program" and work == 0
    assert report == {"findings": [], "open_questions": ["Where is the approval decision recorded?"]}
    on, off = read_episode_pair(maintained, withdrawn, contrast="memory")
    ordinary_access = [[row["data"] for row in arm if row["kind"] == "tool"][:2] for arm in (on, off)]
    assert ordinary_access[0] == ordinary_access[1]
    omitted = run_episode(task, TaskWorld(tool, spans), policy, LIMITS, Tokenizer(), generate, tmp_path / "omitted")
    _, no_memory = read_episode_pair(maintained, omitted, contrast="memory")
    assert no_memory[-1]["data"]["submission"] == report
    assert no_memory[-1]["data"]["usage"]["attempts"]["probe"] == 0


@pytest.mark.parametrize(("generate", "status", "complete", "tokens"), [
    (lambda _: PolicyReply("partial", None, False), "usage_unknown", False, 0),
    (lambda _: PolicyReply("partial", 17, False), "output_contract_violation", True, 17),
    (lambda _: PolicyReply("partial", 10, False), "unterminated", True, 10),
])
def test_response_failures_preserve_reported_usage(tmp_path, generate, status, complete, tokens):
    artifact = run_episode(TASK, world(tmp_path), POLICY, replace(LIMITS, max_new_tokens=16), Tokenizer(),
                           generate, tmp_path / "run")
    records = read_episode(artifact)
    assert records[-2]["data"]["text"] == "partial"
    finish = records[-1]["data"]
    assert finish["submission"] is None and finish["status"] == status
    assert finish["usage"]["output_usage_complete"] is complete
    assert finish["usage"]["known_output_tokens"] == tokens


def test_predispatch_refusal_and_backend_failure_are_durable(tmp_path):
    def fail(_):
        raise RuntimeError("unknown backend usage")

    pinned = world(tmp_path)
    for name, limits in (("context", replace(LIMITS, context_tokens=80, max_new_tokens=16)),
                         ("input", replace(LIMITS, input_tokens=80))):
        path = tmp_path / name
        artifact = run_episode(TASK, pinned, POLICY, limits, Tokenizer(), fail, path)
        records = read_episode(artifact)
        assert [row["kind"] for row in records] == ["start", "refusal", "finish"]
        assert records[-1]["data"]["status"] == name + "_limit"
        assert records[-1]["data"]["usage"]["input_tokens"] == 0
        with pytest.raises(FileExistsError):
            run_episode(TASK, pinned, POLICY, limits, Tokenizer(), fail, path)
    artifact = run_episode(TASK, pinned, POLICY, LIMITS, Tokenizer(), fail, tmp_path / "error")
    records = read_episode(artifact)
    assert records[-2]["data"] == {"error_type": "RuntimeError"}
    assert records[-1]["data"]["status"] == "backend_error"
    assert records[-1]["data"]["usage"]["output_usage_complete"] is False
    raw = artifact.path.read_bytes()
    artifact.path.write_bytes(raw.replace(b'"notes"', b'"other"', 1))
    with pytest.raises(ValueError, match="record changed"):
        read_episode(artifact)
    artifact.path.write_bytes(raw[:-1])
    with pytest.raises(ValueError, match="incomplete episode record"):
        read_episode(artifact)
