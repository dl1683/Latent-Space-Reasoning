from dataclasses import replace
import json

import numpy as np
import pytest

from experiments.memory.corpus_binding_journal import JournaledCorpusWorkspace
from experiments.memory.corpus_bindings import BindingMention, CorpusBindingMemory, StaleCorpusRevision
from experiments.memory.latent_factor_field import LatentFactorField
from experiments.memory.task_episode import (
    EpisodeLimits, EpisodePolicy, PolicyReply, _Tools, read_episode, read_episode_pair, run_episode,
)
from experiments.memory.task_evidence import JointStateBudget, TaskWorld
from experiments.memory.test_task_evidence import _source_tool, _write
from experiments.memory.verified_source_tool import SourceSpan


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

def _joint_world(tmp_path):
    def encode(raw):
        start = raw.index(b":") + 2
        return (BindingMention("rule", np.array([1., 0.]), 1., byte_span=(start, len(raw))),)

    frame = CorpusBindingMemory.prepare(("rule",), np.eye(2), {}, {}, encode=encode, encoder_version="joint-fixture",
                                        min_similarity=.9, gap=.05)
    owner = JournaledCorpusWorkspace(tmp_path / "joint.sqlite", frame, create=True)
    field = LatentFactorField((2,) * 4, compiler_id="supplied-commission-rules")
    equal, opposite = np.eye(2), np.fliplr(np.eye(2))
    records = {"seed": "Commission: Audit and release modes must match.",
               "archive": "Storage: Archive and audit modes must match.",
               "limit": "Oversight: Release and archive modes must differ.",
               "unrelated": "Payroll: Payroll mode is on."}
    entries = {"seed": ((0, (1, 2), equal),), "archive": ((0, (0, 1), equal),),
               "limit": ((0, (2, 0), opposite),), "unrelated": ((0, (3,), [0., 1.]),)}
    corpus = _write(owner, {key: value.encode() for key, value in records.items()})
    batches = {key: field.compile_source(corpus, key, factors) for key, factors in entries.items()}
    raw = field.prepare(corpus, batches.values())
    compiled = field.maintain_clusters(raw, max_bag_cells=8, max_retained_cells=100)
    return field, owner, records, entries, batches, raw, compiled


def _joint_task_world(tmp_path, records, pin, *, memory=True):
    tool, spans = _source_tool(tmp_path, records)
    return TaskWorld(tool, spans, bindings=pin.corpus, factor_pin=pin if memory else None)


def _compact_joint_policy(*, assume=None, rejected=False):
    attempted = []
    states = {"inconsistent": "conflicted", "compatible_in_registered_scope": "supports",
              "conditional_supported": "hypothetical"}

    def generate(call):
        frame = json.loads(dict(call.messages)["user"])
        assert frame["joint_state_enabled"]
        phase, receipts = frame["planning_phase"], frame["receipts"]
        if phase == "declare":
            return reply(plan=[{
                "id": "rule", "parent": None, "discovered_from": [],
                "target_or_event": "commission", "relation": "joint feasibility", "time_version_scope": "current pin",
                "evidence_sought": ["operative rule", "related constraints"],
                "support_requirements": ["verified compatible assignment"],
                "defeat_requirements": ["incompatible joint constraints"],
                "alternatives": ["more than one assignment"], "unresolved_conditions": ["coverage incomplete"],
            }])

        def act(op, **fields):
            return reply(state="Check the commission rules.", need_additions=[],
                         tool={"op": op, "need_id": "rule", **fields})

        if phase == "act_after_plan":
            return act("search", query="commission")
        if len(receipts) == 1:
            found = receipts[-1]["result"]
            assert [row["span"]["source_id"] for row in found["candidates"]] == ["seed"]
            return act("read", candidate=found["candidates"][0]["candidate"])
        if len(receipts) == 2:
            seed = receipts[-1]["result"]
            assert seed["status"] == "VERIFIED_SOURCE_BYTES"
            assert seed["factor_observations"] in ([], [{"ordinal": 0, "widths": [2, 2]}])
            if assume is None:
                return act("joint_check", read=seed["id"], ordinal=0)
            return act("joint_state", read=seed["id"], ordinal=0, axis=1, assume=assume)
        joint = receipts[2]["result"]
        assert joint["status"] == "JOINT_STATE_RESULT"
        result = joint["state_status"]
        assert "probabilities" not in joint and "support" not in joint

        def finish(citations, *, stance=None, support=True):
            row = {"need_id": "rule", "stance": stance or states.get(result, "unknown"),
                   "claim": "The current rules have the reported joint status.",
                   "reason": "The reported outcome uses the pinned dependency computation.", "citations": citations}
            if support:
                row["joint_support"] = {"receipt": joint["receipt"], "status": result, "code": joint["code"]}
            questions = [{"need_id": "rule", "question": "What would resolve the remaining possibilities?"}]
            return reply(finish={"findings": [row], "open_questions": questions if row["stance"] == "unknown" else []})
        if rejected and len(attempted) < 3:
            which = len(attempted)
            attempted.append(which)
            if which == 0:
                return finish([0], support=False)
            if which == 1:
                return finish([0], stance="supports")
            return finish([0])
        if len(receipts) == 3 and result in states:
            candidate_ids = [row["candidate"] for row in joint["candidates"] if row["span"]["source_id"] != "seed"]
            assert candidate_ids
            return act("read", candidate=candidate_ids)
        if len(receipts) > 3:
            reads = receipts[-1]["result"]["reads"]
            assert all(row["status"] == "VERIFIED_SOURCE_BYTES" for row in reads)
            return finish([0, *(row["id"] for row in reads)])
        return finish([0])

    return generate


def test_compact_task_verifies_joint_conflict_withdrawal_and_restoration(tmp_path):
    field, owner, records, entries, batches, raw, opening = _joint_world(tmp_path)
    initial_world = _joint_task_world(tmp_path, records, opening)
    policy = replace(POLICY, state_kind="compact_needs")
    limits = replace(LIMITS, calls=12, context_tokens=16000, input_tokens=160000, output_tokens=8192,
                     access=replace(LIMITS.access, joint_state=JointStateBudget()))

    def run(world, name, status, **options):
        artifact = run_episode("Can the commission follow all current rules?", world, policy, limits, Tokenizer(),
                               _compact_joint_policy(**options), tmp_path / name)
        ledger = read_episode(artifact)
        assert ledger[-1]["data"]["status"] == "completed"
        assert ledger[0]["data"]["joint_state_version"] == 1
        assert ledger[0]["data"]["pin"].get("factor_sha256") == world.pin.factor_sha256
        assert ledger[-1]["data"]["usage"]["attempts"]["joint_state"] == 1
        support = ledger[-1]["data"]["submission"]["findings"][0]["joint_support"]
        assert support["status"] == status
        assert support["code"] == (1 if status == "conditional_supported" else None)
        return artifact, ledger

    _, old = run(initial_world, "conflict", "inconsistent", rejected=True)
    refusals = [row["data"]["reason"] for row in old if row["kind"] == "rejection"]
    assert len(refusals) == 3
    assert "structured receipt" in refusals[0]
    assert "receipt's stance" in refusals[1]
    assert "exact component source spans" in refusals[2]
    assert old[0]["data"]["setup_factor_bytes"] > 0
    assert old[0]["data"]["factor_compiler_id"] == "supplied-commission-rules"
    assert old[-1]["data"]["submission"]["findings"][0]["citations"] == [0, 1, 2]

    withdrawn = _write(owner, {"limit": None})
    current = field.prepare(withdrawn, (), previous=opening)
    reduced = {name: text for name, text in records.items() if name != "limit"}
    current_world = _joint_task_world(tmp_path, reduced, current)
    _, after_withdrawal = run(current_world, "withdrawn", "compatible_in_registered_scope")
    assert after_withdrawal[-1]["data"]["submission"]["findings"][0]["stance"] == "supports"
    _, conditional = run(current_world, "hypothesis", "conditional_supported", assume={"axis": 0, "code": 1})
    assert conditional[-1]["data"]["submission"]["findings"][0]["stance"] == "hypothetical"
    assert conditional[-1]["data"]["usage"]["joint_tensor_cells"] > 0

    readded = _write(owner, {"limit": records["limit"].encode()})
    with pytest.raises(StaleCorpusRevision, match="changed after compilation"):
        field.prepare(readded, (batches["limit"],), previous=current)
    reinstated = field.prepare(readded, (field.compile_source(readded, "limit", entries["limit"]),), previous=current)
    repaired = field.maintain_clusters(reinstated, max_bag_cells=8, max_retained_cells=100)
    restore_world = _joint_task_world(tmp_path, records, repaired)
    restored, new = run(restore_world, "restored", "inconsistent")
    assert new[0]["data"]["pin"]["source_snapshot"] == old[0]["data"]["pin"]["source_snapshot"]
    assert new[0]["data"]["pin"]["factor_sha256"] != old[0]["data"]["pin"]["factor_sha256"]
    _, historical = run(initial_world, "historical", "inconsistent")
    assert historical[0]["data"]["pin"] == old[0]["data"]["pin"]
    without, detached = run(_joint_task_world(tmp_path, records, repaired, memory=False),
                            "no-memory", "no_factor_memory")
    read_episode_pair(restored, without, contrast="memory")
    first_request = next(row["data"]["messages"] for row in new if row["kind"] == "request")
    no_memory_request = next(row["data"]["messages"] for row in detached if row["kind"] == "request")
    assert first_request == no_memory_request

    with pytest.raises(ValueError, match="exact journaled task revision"):
        TaskWorld(restore_world._tool, restore_world._spans, bindings=readded, factor_pin=opening)
    with pytest.raises(ValueError, match="effective-time views"):
        TaskWorld(restore_world._tool, restore_world._spans, bindings=readded, factor_pin=repaired, as_of=123)
    assert _joint_task_world(tmp_path, records, raw).pin.factor_sha256 != initial_world.pin.factor_sha256


@pytest.mark.parametrize("maximum,expected", [(0, "work_limit"), (8192, "inconsistent")])
def test_joint_tool_limits_preserve_scope_and_citation_obligations(tmp_path, maximum, expected):
    _, _, records, _, _, _, pin = _joint_world(tmp_path)
    world = _joint_task_world(tmp_path, records, pin)
    limits = replace(LIMITS, access=replace(LIMITS.access, joint_state=JointStateBudget(
        calls=4, updates=0, tensor_cells=maximum, candidate_spans=1,
    )))
    tools = _Tools(TASK, world, limits, probes_enabled=False, typed_needs=True)
    needs = ("a", "b")

    def call(op, need="a", **fields):
        return tools.call({"op": op, "need_id": need, **fields}, needs)

    found = call("search", query="commission")
    seed = call("read", candidate=found["candidates"][0]["candidate"])["id"]
    wrong = call("joint_state", need="b", read=seed, ordinal=0, axis=1)
    assert wrong["status"] == "INVALID_TOOL_ARGUMENTS"
    forged = call("joint_state", read=seed, ordinal=1, axis=0)
    assert forged["status"] == "INVALID_TOOL_ARGUMENTS"
    receipt = call("joint_state", read=seed, ordinal=0, axis=1)
    assert receipt["state_status"] == expected
    assert receipt["component_spans"] == 3 and receipt["candidate_spans_truncated"]
    assert len(receipt["candidates"]) == 1
    assert not tools.counters()["joint_tensor_cells"]
    if maximum:
        with pytest.raises(ValueError, match="exact component"):
            tools.submission({"findings": [{"need_id": "a", "claim": "Conflict", "stance": "conflicted",
                                           "reason": "Joint state", "citations": [seed],
                                           "joint_support": {"receipt": 0, "status": expected, "code": None}}],
                              "open_questions": []}, ("a",))
        unknown = {"findings": [{"need_id": "a", "claim": "Conflict evidence is incomplete.", "stance": "unknown",
                                 "reason": "Other clauses still need reopening.", "citations": [seed],
                                 "joint_support": {"receipt": 0, "status": expected, "code": None}}],
                   "open_questions": [{"need_id": "a", "question": "Where are the other clauses?"}]}
        assert tools.submission(unknown, ("a",)) == unknown

def test_joint_receipts_preserve_distinct_assumptions_and_reject_partial_seed(tmp_path):
    _, _, records, _, _, _, pin = _joint_world(tmp_path)
    world = _joint_task_world(tmp_path, records, pin)
    limits = replace(LIMITS, access=replace(LIMITS.access, joint_state=JointStateBudget(calls=3)))
    tools = _Tools(TASK, world, limits, probes_enabled=False, typed_needs=True)

    def call(op, **fields):
        return tools.call({"op": op, "need_id": "need", **fields}, ("need",))

    search = call("search", query="commission")
    seed = call("read", candidate=search["candidates"][0]["candidate"])["id"]
    fields = {"read": seed, "ordinal": 0, "axis": 1}
    conditional = call("joint_state", **fields, assume={"axis": 0, "code": 1})
    factual = call("joint_state", **fields)
    assert conditional["state_status"] == "conditional_inconsistent"
    assert factual["state_status"] == "inconsistent"

    submission = {"findings": [{"need_id": "need", "claim": "Still open", "stance": "unknown",
                                "reason": "Proofs not reopened.", "citations": [seed],
                                "joint_support": {"receipt": 1, "status": "inconsistent", "code": None}}],
                  "open_questions": [{"need_id": "need", "question": "Which clauses need to be reopened?"}]}
    with pytest.raises(ValueError, match="each distinct joint query"):
        tools.submission(submission, ("need",))
    submission["findings"].append(submission["findings"][0] | {
        "joint_support": {"receipt": 0, "status": "conditional_inconsistent", "code": None},
    })
    assert tools.submission(submission, ("need",)) == submission

    title = SourceSpan.from_characters("seed", records["seed"].encode(), 0, len("Commission"))
    partial_world = TaskWorld(world._tool, (title,), bindings=pin.corpus, factor_pin=pin)
    partial_tools = _Tools(TASK, partial_world, limits, probes_enabled=False, typed_needs=True)
    result = partial_tools.call({"op": "search", "need_id": "need", "query": "commission"}, ("need",))
    pointer = result["candidates"][0]["candidate"]
    evidence = partial_tools.call({"op": "read", "need_id": "need", "candidate": pointer}, ("need",))
    assert evidence["status"] == "VERIFIED_SOURCE_BYTES" and evidence["factor_observations"] == []
    refused = partial_tools.call({"op": "joint_state", "need_id": "need", "read": evidence["id"],
                                  "ordinal": 0, "axis": 1}, ("need",))
    assert refused["status"] == "INVALID_TOOL_ARGUMENTS"

def test_joint_candidate_pages_reopen_nonlexical_dependencies_on_the_same_need(tmp_path):
    _, _, records, _, _, _, pin = _joint_world(tmp_path)
    world = _joint_task_world(tmp_path, records, pin)
    limits = replace(LIMITS, queries=4, access=replace(
        LIMITS.access, candidates_per_search=1, joint_state=JointStateBudget(candidate_spans=3),
    ))
    tools = _Tools(TASK, world, limits, probes_enabled=False, typed_needs=True)

    def call(op, need="a", **fields):
        return tools.call({"op": op, "need_id": need, **fields}, ("a", "b"))

    result = call("search", query="commission")
    seed = call("read", candidate=result["candidates"][0]["candidate"])["id"]
    page = call("joint_state", read=seed, ordinal=0, axis=1)
    wrong = call("joint_spans", need="b", receipt=page["receipt"], offset=page["next_offset"])
    assert wrong["status"] == "INVALID_TOOL_ARGUMENTS"
    candidates = list(page["candidates"])
    while "next_offset" in page:
        page = call("joint_spans", receipt=page["receipt"], offset=page["next_offset"])
        assert page["status"] == "JOINT_SPANS"
        candidates.extend(page["candidates"])
    assert not page["candidate_spans_truncated"]
    assert {row["span"]["source_id"] for row in candidates} == {"archive", "limit"}
    opened = call("read", candidate=[row["candidate"] for row in candidates if row["span"]["source_id"] != "seed"])
    report = {"findings": [{"need_id": "a", "claim": "Conflict", "stance": "conflicted", "reason": "Complete proof",
                            "citations": [seed, *(read["id"] for read in opened["reads"])],
                            "joint_support": {"receipt": 0, "status": "inconsistent", "code": None}}],
              "open_questions": []}
    assert tools.submission(report, ("a",)) == report

def test_later_unfinished_joint_receipt_cannot_hide_a_completed_result(tmp_path):
    _, _, records, _, _, _, pin = _joint_world(tmp_path)
    world = _joint_task_world(tmp_path, records, pin)
    graph = pin._graph
    slot = graph.factors["seed", 0].axes[0]
    bank = pin._clusters[graph.components[slot]]
    bag = bank.plan.owner[slot]
    cells = int(bank.locals[bag].size) * (2 + len(bank.plan.adjacent[bag]))
    limits = replace(LIMITS, access=replace(LIMITS.access, joint_state=JointStateBudget(tensor_cells=cells)))
    host = _Tools(TASK, world, limits, probes_enabled=False, typed_needs=True)

    def call(op, **fields):
        return host.call({"op": op, "need_id": "rule", **fields}, ("rule",))

    found = call("search", query="commission")
    seed = call("read", candidate=found["candidates"][0]["candidate"])["id"]
    complete = call("joint_check", read=seed, ordinal=0)
    exhausted = call("joint_check", read=seed, ordinal=0)
    assert complete["state_status"] == "inconsistent"
    assert exhausted["state_status"] == "work_limit"
    finding = {"need_id": "rule", "claim": "Incomplete source review.", "stance": "unknown",
               "reason": "Other clauses are unopened.", "citations": [seed],
               "joint_support": {"receipt": 1, "status": "work_limit", "code": None}}
    report = {"findings": [finding],
              "open_questions": [{"need_id": "rule", "question": "Where are the other clauses?"}]}
    with pytest.raises(ValueError, match="incomplete receipt cannot replace"):
        host.submission(report, ("rule",))

    finding["joint_support"] = {"receipt": 0, "status": "inconsistent", "code": None}
    report["findings"].append({"need_id": "rule", "claim": "The source names audit and release.",
                               "stance": "supports", "reason": "The opened rule names both.",
                               "citations": [seed]})
    assert host.submission(report, ("rule",)) == report
