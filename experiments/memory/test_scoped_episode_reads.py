from dataclasses import replace
import json

import pytest

from experiments.memory.source_revision_feed import write_source_delta
from experiments.memory.task_episode import EpisodeArtifact, _Ledger, _Tools, read_episode, run_episode
from experiments.memory.task_evidence import TaskWorld
from experiments.memory.test_task_episode import LIMITS, POLICY, Tokenizer, reply


TASK = "Can the evening transfer proceed?"
NEEDS = ("approval", "staffing", "inspection")
QUERIES = ("zircon", "quartz", "basalt")
EVIDENCE = ("Zircon approval is recorded.", "Quartz duty roster is signed.", "Basalt inspection passed.")
SOURCE_IDS = ("a" * 32, "b" * 32, "c" * 32)


@pytest.fixture
def disjoint_world(tmp_path):
    feed = tmp_path / "source"
    source = write_source_delta(feed, {
        source_id: text.encode() for source_id, text in zip(SOURCE_IDS, EVIDENCE, strict=True)
    })
    return TaskWorld.from_source_epoch(feed, source.seal, max_feed_bytes=8192,
                                       max_source_bytes=8192, max_passages=8)


def _need(need_id):
    return {
        "id": need_id, "parent": None, "discovered_from": [], "target_or_event": "transfer",
        "relation": need_id, "time_version_scope": "current", "evidence_sought": [need_id],
        "support_requirements": [], "defeat_requirements": [], "alternatives": [], "unresolved_conditions": [],
    }


def _run_disjoint(world, output, *, grouped, calls):
    limits = replace(LIMITS, calls=calls, queries=3, reads=4, probes=0,
                     scoped_read_batches=grouped)
    policy = replace(POLICY, state_kind="compact_needs", compact_delivery=True)

    def generate(call):
        frame = json.loads(dict(call.messages)["user"])
        if frame["planning_phase"] == "declare":
            return reply(plan=[_need(need) for need in NEEDS[:2]])

        receipts = frame["receipts"]
        step = len(receipts)
        pending = []
        if step < 3:
            if step == 0:
                pending = [_need(NEEDS[2])]
            tool = {"op": "search", "need_id": NEEDS[step], "query": QUERIES[step]}
            return reply(state="", need_additions=pending, tool=tool)

        def wanted(i):
            matches = receipts[i]["result"]["candidates"]
            assert len(matches) == receipts[i]["result"]["matched"] == 1
            assert matches[0]["span"]["source_id"] == SOURCE_IDS[i]
            return {"need_id": NEEDS[i], "candidate": matches[0]["candidate"]}

        if step == 3 and grouped:
            assert frame["scoped_read_batches_enabled"] is True
            assert "read_scoped" in dict(call.messages)["system"]
            return reply(state="", need_additions=[], tool={"op": "read_scoped", "items": [wanted(0), wanted(1)]})

        if step < calls - 2:
            item = wanted(2 if grouped else step - 3)
            return reply(state="", need_additions=[], tool={"op": "read", **item})

        reads = []
        for receipt in receipts[3:]:
            result = receipt["result"]
            reads.extend(result["reads"] if result["status"] == "SCOPED_READ_BATCH" else [result])
        findings, unresolved = [], []
        for i, need in enumerate(NEEDS):
            if i < len(reads):
                read = reads[i]
                assert read["status"] == "VERIFIED_SOURCE_BYTES" and read["text"] == EVIDENCE[i]
                findings.append({"need_id": need, "claim": read["text"], "stance": "supports",
                                 "citations": [read["id"]], "reason": "The source confirms it."})
            else:
                findings.append({"need_id": need, "claim": "The requirement is unresolved.", "stance": "unknown",
                                 "citations": [], "reason": "Its document has not been verified."})
                unresolved.append({"need_id": need, "question": f"Is {need} confirmed?"})
        return reply(finish={"findings": findings, "open_questions": unresolved})

    return read_episode(run_episode(TASK, world, policy, limits, Tokenizer(), generate, output))


def _tool_records(ledger):
    return [row["data"] for row in ledger if row["kind"] == "tool"]


def test_grouped_reads_fit_three_disjoint_needs_in_five_actions(disjoint_world, tmp_path):
    grouped = _run_disjoint(disjoint_world, tmp_path / "grouped", grouped=True, calls=7)
    ungrouped = _run_disjoint(disjoint_world, tmp_path / "ungrouped", grouped=False, calls=7)
    extended = _run_disjoint(disjoint_world, tmp_path / "extended", grouped=False, calls=8)
    assert grouped[0]["data"]["source_epoch_seal"] == ungrouped[0]["data"]["source_epoch_seal"]
    assert grouped[0]["data"]["limits"]["scoped_read_batches"] is True
    assert "scoped_read_batches" not in ungrouped[0]["data"]["limits"]

    for ledger, calls, actions, reads in ((grouped, 7, 5, 3), (ungrouped, 7, 5, 2), (extended, 8, 6, 3)):
        finish = ledger[-1]["data"]
        assert finish["status"] == "completed" and finish["usage"]["calls"] == calls
        assert len(_tool_records(ledger)) == actions
        assert finish["usage"]["attempts"] == {"search": 3, "read": reads, "probe": 0}
        assert finish["usage"]["source_bytes"] == finish["usage"]["evidence_bytes"] == sum(
            len(value.encode()) for value in EVIDENCE[:reads]
        )
        assert [row["citations"] for row in finish["submission"]["findings"]] == [
            [i] if i < reads else [] for i in range(3)
        ]

    batched = _tool_records(grouped)[3]
    assert batched["request"]["op"] == "read_scoped"
    assert batched["result"]["status"] == "SCOPED_READ_BATCH"
    for item, ordinary in zip(batched["result"]["reads"], _tool_records(extended)[3:5], strict=True):
        expected = {"need_id": ordinary["request"]["need_id"], "candidate": ordinary["request"]["candidate"],
                    **ordinary["result"]}
        assert item == expected
    assert grouped[-1]["data"]["submission"] == extended[-1]["data"]["submission"]

    # Reissue the chain in the test to exercise receipt validation beyond its hash.
    batched["result"]["reads"].reverse()
    path = tmp_path / "reordered"
    with path.open("xb") as handle:
        rewritten = _Ledger(handle)
        for row in grouped:
            rewritten.write(row["kind"], row["data"])
    with pytest.raises(ValueError, match="in order"):
        read_episode(EpisodeArtifact(path, rewritten.head))


def _prepared_tools(world, *, reads=4, grouped=True, access=LIMITS.access):
    limits = replace(LIMITS, reads=reads, access=access, scoped_read_batches=grouped)
    tools = _Tools(TASK, world, limits, probes_enabled=False, typed_needs=True)
    items = []
    for need, query in zip(NEEDS[:2], QUERIES[:2], strict=True):
        search = tools.call({"op": "search", "need_id": need, "query": query}, NEEDS)
        candidate, = search["candidates"]
        items.append({"need_id": need, "candidate": candidate["candidate"]})
    return tools, items


@pytest.mark.parametrize("problem", ["scope", "unknown", "candidate", "duplicate", "three", "boolean", "budget", "disabled"])
def test_invalid_group_opens_no_source(disjoint_world, problem):
    tools, items = _prepared_tools(disjoint_world, reads=1 if problem == "budget" else 4,
                                   grouped=problem != "disabled")
    if problem == "scope":
        items[1]["need_id"] = NEEDS[2]
    elif problem == "unknown":
        items[1]["need_id"] = "not registered"
    elif problem == "candidate":
        items[1]["candidate"] = 99
    elif problem == "duplicate":
        items[1]["candidate"] = items[0]["candidate"]
    elif problem == "three":
        items += [items[0]]
    elif problem == "boolean":
        items[1]["candidate"] = True

    result = tools.call({"op": "read_scoped", "items": items}, NEEDS)
    expected = {"budget": "TOOL_ATTEMPT_LIMIT", "disabled": "INVALID_TOOL"}
    assert result["status"] == expected.get(problem, "INVALID_TOOL_ARGUMENTS")
    assert tools._session.reads == [] and tools.counters()["source_bytes"] == 0
    assert tools.counters()["attempts"]["read"] == (0 if problem == "disabled" else 2)


def test_group_preserves_owner_and_individual_source_limit(disjoint_world):
    budget = replace(LIMITS.access, source_bytes=len(EVIDENCE[0].encode()))
    tools, items = _prepared_tools(disjoint_world, access=budget)
    result = tools.call({"op": "read_scoped", "items": items}, NEEDS)
    rows = result["reads"]
    assert [row["status"] for row in rows] == ["VERIFIED_SOURCE_BYTES", "SOURCE_LIMIT"]
    assert [read.need for read in tools._session.reads] == list(NEEDS[:2])
    assert [row["need_id"] for row in rows] == list(NEEDS[:2])
    assert tools._session._citations([rows[0]["id"]]) == (rows[0]["id"],)
    with pytest.raises(ValueError, match="verified reads"):
        tools._session._citations([rows[1]["id"]])


@pytest.mark.parametrize("native", [False, True])
def test_existing_compact_decoder_refuses_grouped_reads(disjoint_world, tmp_path, native):
    from experiments.memory.compact_episode_constraint import CompactSearchReadConstraint, _PlainCompactBackend
    from experiments.memory.native_task_episode import NativeTaskEpisodeBackend

    decoder = object.__new__(CompactSearchReadConstraint)
    if native:
        backend = object.__new__(NativeTaskEpisodeBackend)
        backend._output_token_allowlist = decoder
    else:
        backend = object.__new__(_PlainCompactBackend)
        backend.constraint = decoder
    path = tmp_path / "unopened"
    limits = replace(LIMITS, scoped_read_batches=True)
    policy = replace(POLICY, state_kind="compact_needs", compact_delivery=True)
    with pytest.raises(ValueError, match="existing compact decoder does not support grouped reads"):
        run_episode(TASK, disjoint_world, policy, limits, Tokenizer(), backend, path,
                    native_backend=backend if native else None)
    assert not path.exists()


def test_group_records_a_source_error_and_continues(disjoint_world, monkeypatch):
    tools, items = _prepared_tools(disjoint_world)
    source_tool_type = type(disjoint_world._tool)
    verify = source_tool_type.verify

    def reject_first(self, span, **limits):
        if span.source_id == SOURCE_IDS[0]:
            raise OSError("fixture source refused")
        return verify(self, span, **limits)

    monkeypatch.setattr(source_tool_type, "verify", reject_first)
    rows = tools.call({"op": "read_scoped", "items": items}, NEEDS)["reads"]
    assert [(row["need_id"], row["status"]) for row in rows] == [
        (NEEDS[0], "SOURCE_ERROR"), (NEEDS[1], "VERIFIED_SOURCE_BYTES"),
    ]
    assert "id" not in rows[0] and rows[1]["id"] == 0
    assert tools._session.reads[0].need == NEEDS[1]
    assert tools.counters()["attempts"]["read"] == 2
    assert tools.counters()["source_bytes"] == len(EVIDENCE[1].encode())
