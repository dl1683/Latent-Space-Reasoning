from dataclasses import asdict
import json

import numpy as np
import pytest

from experiments.memory.corpus_bindings import BindingMention, CorpusBindingMemory
from experiments.memory.corpus_binding_journal import JournaledCorpusWorkspace
from experiments.memory.task_evidence import (
    EvidenceCase, EvidenceNeed, OpenEvidence, ProbeRoleLinks, ReconcileProbe, SearchEvidence, TaskBudget,
    TaskInvestigator, TaskWorld,
)
from experiments.memory.verified_source_tool import SourceSpan, VerifiedSourceTool, sha256


def _source_tool(root, records):
    leaves = root / "leaves"
    leaves.mkdir(exist_ok=True)
    versions, spans = {}, []
    for source, text in records.items():
        raw = text.encode()
        digest = sha256(raw)
        (leaves / f"{digest}.txt").write_bytes(raw)
        versions[source] = digest
        spans.append(SourceSpan.from_characters(source, raw, 0, len(text)))
    return VerifiedSourceTool(root, versions), tuple(spans)


def _write(workspace, changes):
    return workspace.publish(workspace.prepare_changes(workspace.snapshot(), changes))


def test_task_evidence_follows_an_alias_and_keeps_revision_conflicts_and_gaps(tmp_path):
    records = {
        "rules": "Release rule: a Cobalt sign-off and a confirmed duty cover are both required.",
        "routing": "Cobalt sign-off is recorded under Juniper in the separate review log.",
        "log": "Juniper review log: sign-off cleared.",
        "handover": "Duty arrangements are recorded separately. This note names no duty holder.",
    }
    axes = np.eye(3)

    def encode(raw):
        if raw.startswith(b"Cobalt"):
            yield BindingMention("entry", axes[0], .9)
            yield BindingMention("exit", axes[1], .9)
        if raw.startswith(b"Juniper"):
            outcome = "withheld" if b"withheld" in raw else "cleared"
            yield BindingMention(outcome, axes[1], .8)

    frame = CorpusBindingMemory.prepare(("entry", "exit", "cleared", "withheld"), axes, {}, {},
                                        encode=encode, encoder_version="task-fixture", min_similarity=.9, gap=.05)
    workspace = JournaledCorpusWorkspace(tmp_path / "sources.sqlite", frame, create=True)
    initial = _write(workspace, {source: text.encode() for source, text in records.items()})
    prepared = workspace.prepare_role_links(initial, ("entry", "exit"), min_similarity=.95, compile_work_limit=64)
    opening = workspace.publish_program(prepared)
    tool, spans = _source_tool(tmp_path, records)
    first = TaskWorld(tool, spans, bindings=opening)
    reopened = JournaledCorpusWorkspace(tmp_path / "sources.sqlite", frame).snapshot()
    assert TaskWorld(tool, spans, bindings=reopened).pin == first.pin
    other = JournaledCorpusWorkspace(tmp_path / "other.sqlite", frame, create=True)
    other_initial = _write(other, {source: text.encode() for source, text in records.items()})
    other_pin = other.publish_program(other.prepare_role_links(
        other_initial, ("entry", "exit"), min_similarity=.95, compile_work_limit=64,
    ))
    assert other_pin.revision == opening.revision
    assert TaskWorld(tool, spans, bindings=other_pin).pin != first.pin

    # The policy fixture chooses its next action from the live task transcript.
    # It has no access to the world object or sources it has not opened.
    task = "Can we proceed this evening?"

    def policy(view):
        assert view.task == task
        if view.steps == 0:
            return EvidenceNeed("release", "What must be true to proceed?", "The decision depends on the release rule.")
        if view.steps == 1:
            return SearchEvidence("release", "release")
        if view.steps == 2:
            return OpenEvidence("release", view.searches[-1].candidates[0])
        if view.steps == 3:
            assert "Cobalt" in view.reads[0].text
            return EvidenceNeed("check", "Has Cobalt signed off?", "The release rule requires it.", "release", (0,))
        if view.steps == 4:
            return SearchEvidence("check", "recorded Cobalt")
        if view.steps == 5:
            candidate = next(span for span in view.searches[-1].candidates if span.source_id != "rules")
            return OpenEvidence("check", candidate)
        if view.steps == 6:
            assert "Juniper" in view.reads[1].text
            return SearchEvidence("check", "Juniper review log")
        if view.steps == 7:
            return OpenEvidence("check", view.searches[-1].candidates[0])
        if view.steps == 8:
            ordinal = view.reads[1].observation_predicates.index("entry")
            return ProbeRoleLinks("check", None, ("cleared", "withheld"), source_observation=(1, ordinal))
        if view.steps == 9:
            return EvidenceCase("check", "supports" if "cleared" in view.reads[2].text else "defeats",
                                "The Juniper log decides the Cobalt sign-off.",
                                "The routing record names the matching log.", (1, 2))
        if view.steps == 10 and "supersedes" in view.reads[2].text:
            return EvidenceCase("check", "hypothetical", "An old dashboard could still show clearance.",
                                "This would matter only if no correction applied.", (2,))
        if not any(item.need.id == "cover" for item in view.needs):
            assert "duty cover" in view.reads[0].text
            return EvidenceNeed("cover", "Who has confirmed duty cover?", "Release requires a named assignment.",
                                "release", (0,))
        if view.searches[-1].need != "cover":
            return SearchEvidence("cover", "duty holder")
        if view.reads[-1].need != "cover":
            return OpenEvidence("cover", view.searches[-1].candidates[0])
        if not view.needs[-1].cases:
            return EvidenceCase("cover", "unknown", "The handover note names no duty holder.",
                                "Ask for the assignment record.", (view.reads[-1].id,))
        return None

    planner = TaskInvestigator(policy)
    before = planner.plan(task, first)
    assert [(item.need.id, item.status) for item in before.needs] == [
        ("release", "underdetermined"), ("check", "supported"), ("cover", "underdetermined"),
    ]
    assert before.role_evidence[0].read.answer == "cleared"
    assert before.role_evidence[0].request.source_observation == (1, 0)
    assert before.role_evidence[0].direction == (1., 0., 0.)
    assert before.needs[-1].cases[0].stance == "unknown"
    assert before.needs[1].cases[0].citations == (1, 2)
    assert before.needs[0].open_children == ("cover",)
    assert before.stop_reason == "policy_stopped"
    assert before.returned_evidence_bytes == sum(len(row.text.encode()) for row in before.reads)
    assert before.source_bytes_read >= before.returned_evidence_bytes
    assert before.lexical_rows_scored == len(before.searches) * len(spans)
    json.dumps(asdict(before))

    records["log"] = "Juniper review log: sign-off withheld. This supersedes the earlier clearance."
    revised = _write(workspace, {"log": records["log"].encode()})
    current_tool, current_spans = _source_tool(tmp_path, records)
    with pytest.raises(ValueError, match="same sources and versions"):
        TaskWorld(current_tool, current_spans, bindings=opening)
    after = planner.plan(task, TaskWorld(current_tool, current_spans, bindings=revised))
    assert after.pin != before.pin and after.pin.binding_revision == revised.revision
    assert after.needs[1].status == "defeated" and after.role_evidence[0].read.answer == "withheld"
    assert tuple(case.stance for case in after.needs[1].cases) == ("defeats", "hypothetical")
    assert after.needs[0].open_children == ("check", "cover")

    records["parallel"] = "Juniper copy: sign-off cleared."
    contested = _write(workspace, {"parallel": records["parallel"].encode()})
    contest_tool, contest_spans = _source_tool(tmp_path, records)
    unresolved = planner.plan(task, TaskWorld(contest_tool, contest_spans, bindings=contested))
    assert unresolved.role_evidence[0].read.status == "conflict"
    assert unresolved.needs[1].status == "underdetermined"
    assert unresolved.needs[1].cases[0].stance == "defeats"

    def settle(view):
        action = policy(view)
        if action is None and view.role_evidence[0].reconciliation is None:
            return ReconcileProbe("check", 0, "The verified log explicitly supersedes the old clearance.", (2,))
        return action

    reconciled = TaskInvestigator(settle).plan(task, TaskWorld(contest_tool, contest_spans, bindings=contested))
    assert reconciled.role_evidence[0].read.status == "conflict" and reconciled.needs[1].status == "defeated"
    assert reconciled.role_evidence[0].reconciliation.citations == (2,)
    assert planner.plan(task, first) == before


def test_conflicting_source_cases_and_unfinished_needs_stay_visible(tmp_path):
    tool, spans = _source_tool(tmp_path, {"a": "Review completed.", "b": "Review still pending."})
    world = TaskWorld(tool, spans)
    subset = TaskWorld(tool, spans[:1])
    assert subset.pin.source_snapshot == world.pin.source_snapshot
    assert subset.pin.span_catalogue_sha256 != world.pin.span_catalogue_sha256
    pending = EvidenceNeed("other", "Was duty cover confirmed?", "A separate prerequisite.")
    actions = iter((
        EvidenceNeed("review", "Is the review complete?", "Required for the decision."),
        pending,
        OpenEvidence("review", spans[0]),
        OpenEvidence("review", spans[1]),
        EvidenceCase("review", "supports", "Complete.", "First report.", (0,)),
        EvidenceCase("review", "defeats", "Pending.", "Contrary report.", (1,)),
    ))
    report = TaskInvestigator(lambda _: next(actions, None), budget=TaskBudget(steps=6)).plan("Proceed?", world)
    assert report.stop_reason == "step_limit"
    assert report.needs[0].status == "conflicted" and len(report.needs[0].cases) == 2
    assert report.needs[1].need == pending and report.needs[1].status == "not_investigated"
    actions = iter((EvidenceNeed("review", "Is the review complete?", "Required."),
                    SearchEvidence("review", "review")))
    searched = TaskInvestigator(lambda _: next(actions, None), budget=TaskBudget(candidates_per_search=1))
    search, = searched.plan("Proceed?", world).searches
    assert search.matched == 2 and len(search.candidates) == 1


def test_stale_and_budget_refusals_cannot_be_cited(tmp_path):
    old_tool, old_spans = _source_tool(tmp_path, {"source": "First version."})
    new_tool, new_spans = _source_tool(tmp_path, {"source": "Corrected version."})
    world = TaskWorld(new_tool, new_spans)
    with pytest.raises(ValueError, match="unverifiable"):
        TaskWorld(new_tool, old_spans)
    assert old_tool.versions != new_tool.versions

    need = EvidenceNeed("check", "Is the finding current?", "Check the revision.")
    actions = iter((need, OpenEvidence("check", old_spans[0])))
    stale = TaskInvestigator(lambda _: next(actions, None)).plan("Proceed?", world)
    assert stale.reads[0].status == "STALE" and stale.needs[0].status == "source_unavailable"
    assert stale.source_bytes_read == 0
    actions = iter((need, OpenEvidence("check", new_spans[0])))
    limited = TaskInvestigator(lambda _: next(actions, None), budget=TaskBudget(evidence_bytes=1))
    report = limited.plan("Proceed?", world)
    assert report.needs[0].status == "budget_exhausted" and report.reads[0].status == "EVIDENCE_LIMIT"
    assert not report.reads[0].text and report.source_bytes_read == 0
    actions = iter((need, OpenEvidence("check", SourceSpan.from_characters(
        "source", b"Corrected version.", 0, 1,
    ))))
    source_limited = TaskInvestigator(lambda _: next(actions, None), budget=TaskBudget(source_bytes=1))
    report = source_limited.plan("Proceed?", world)
    assert report.needs[0].status == "budget_exhausted" and report.reads[0].status == "SOURCE_LIMIT"
    assert report.source_bytes_read == report.returned_evidence_bytes == 0
    actions = iter((need, OpenEvidence("check", old_spans[0]),
                    EvidenceCase("check", "supports", "Current.", "Uses a stale finding.", (0,)),
                    ProbeRoleLinks("check", (10 ** 400,), ("matches",)),
                    ProbeRoleLinks("check", (1., 0.), ("matches",)),
                    OpenEvidence("check", new_spans[0]),
                    EvidenceCase("check", "supports", "Current.", "Uses the corrected source.", (1,))))
    recovered = TaskInvestigator(lambda _: next(actions, None)).plan("Proceed?", world)
    assert len(recovered.rejected_actions) == 2
    assert "verified reads" in recovered.rejected_actions[0].reason and recovered.rejected_actions[0].step == 3
    assert recovered.reads[0].status == "STALE" and recovered.needs[0].status == "supported"
    assert recovered.role_evidence[0].read.status == "no_memory"
    assert recovered.role_evidence[0].request.outcomes == ("matches",)
    assert recovered.rejected_actions[1].step == 4
    assert recovered.steps == 7

def test_text_source_world_indexes_a_complete_opaque_revision_without_rereading_other_sources(tmp_path):
    from experiments.memory.compressed_routing_map import BM25
    from experiments.memory.source_revision_feed import write_source_delta

    changing, steady, empty = (letter * 32 for letter in "abc")
    old = "Café\n\nKind 🌍\nWörter".encode("utf8")
    new = "Café\n\nOther 🌏\nWörter".encode("utf8")
    anchored = b"Anchor survives"
    opening_dir = tmp_path / "opening"
    opening = write_source_delta(opening_dir, {changing: old, steady: anchored, empty: b""})
    limits = dict(max_feed_bytes=16384, max_source_bytes=1024, max_passages=16, passage_bytes=8)
    before = TaskWorld.from_source_epoch(opening_dir, opening.seal, **limits)
    assert before.setup_source_bytes == len(old) + len(anchored)
    assert not before._spans_by_source.get(empty) and before._tool.versions.keys() == opening.manifest.keys()
    assert before._search(changing, 4) == ((), 0)

    def compare_cold(world, source, query):
        passages = world._spans_by_source[source]
        raw = world._tool._read_verified_source(source)
        assert passages[0].start_byte == 0 and passages[-1].end_byte == len(raw)
        assert all(a.end_byte == b.start_byte for a, b in zip(passages, passages[1:]))
        assert all(span.end_byte - span.start_byte <= limits["passage_bytes"] for span in passages)
        texts = [world._tool.verify(span)["text"] for span in world._spans]
        assert "".join(text for text, span in zip(texts, world._spans, strict=True) if span.source_id == source) == (
            raw.decode("utf8"))
        np.testing.assert_array_equal(world._index.score(query), BM25(texts).score(query))

    compare_cold(before, changing, "Café Kind Anchor")
    corrected_dir = tmp_path / "corrected"
    corrected = write_source_delta(corrected_dir, {changing: new}, previous=opening)
    after = TaskWorld.from_source_epoch(corrected_dir, corrected.seal, previous=before, **limits)
    assert after.setup_source_bytes == len(new)
    assert after._text_passages[steady] is before._text_passages[steady]
    assert after._tool._content[steady] is before._tool._content[steady]
    old_span, = before._search("Kind", 1)[0]
    assert before._tool.verify(old_span)["status"] == "VERIFIED_SOURCE_BYTES"
    assert after._tool.verify(old_span)["status"] == "STALE"
    compare_cold(after, changing, "Café Other Anchor")
    with pytest.raises(ValueError, match="retained source byte allowance"):
        TaskWorld.from_source_epoch(corrected_dir, corrected.seal, previous=before,
                                    **{**limits, "max_source_bytes": len(new)})
    with pytest.raises(ValueError, match="passage allowance"):
        TaskWorld.from_source_epoch(corrected_dir, corrected.seal, previous=before,
                                    **{**limits, "max_passages": 1})
    with pytest.raises(ValueError, match="projection"):
        TaskWorld.from_source_epoch(corrected_dir, corrected.seal, **limits)

    removed_dir = tmp_path / "removed"
    removed = write_source_delta(removed_dir, {changing: None}, previous=corrected)
    view = TaskWorld.from_source_epoch(removed_dir, removed.seal, previous=after, **limits)
    assert view.setup_source_bytes == 0 and view._search("Other", 2) == ((), 0)
    assert view._tool.verify(old_span)["status"] == "UNKNOWN_SOURCE"
    assert after._tool.verify(after._spans_by_source[changing][0])["status"] == "VERIFIED_SOURCE_BYTES"
