from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from fractions import Fraction
from pathlib import Path
import json
import sqlite3
import subprocess
import sys
from threading import Barrier, Event

import numpy as np
import pytest

from experiments.memory import corpus_binding_journal, corpus_bindings
from experiments.memory.binding_memory import BindingCapacityError, BindingGate, BindingProof
from experiments.memory.corpus_binding_journal import (
    CorpusProgramExpectation, CorpusProgramProbe, JournaledCorpusWorkspace, _event_digest,
)
from experiments.memory.corpus_bindings import (
    BindingMention, CorpusBindingMemory, CorpusBindingRead, CorpusBindingWorkspace, StaleCorpusRevision,
)


def _encode(raw):
    predicate, x, y, strength = raw.decode().split(",")
    return [BindingMention(predicate, np.array([float(x), float(y)]), float(strength))]


def _no_encoding(raw):
    raise AssertionError(f"replay tried to encode a source of {len(raw)} bytes")


def _frame(encode=_encode, *, version="journal-fixture", term_limit=64):
    return CorpusBindingMemory.prepare(
        ("inspection", "permit"), np.eye(2),
        {"release": BindingGate("disjoint_and", ("inspection", "permit"))},
        {}, encode=encode, encoder_version=version, min_similarity=.6, gap=.05, term_limit=term_limit,
    )


def _write(owner, changes):
    return owner.publish(owner.prepare_changes(owner.snapshot(), changes))


@pytest.mark.parametrize("durable", [False, True])
def test_rewiring_transitive_gate_changes_proof_and_keeps_live_source_edits(tmp_path, durable):
    original = {
        "match": BindingGate("and", ("inspection", "permit")),
        "release": BindingGate("or", ("match",)),
        "audit": BindingGate("or", ("permit",)),
        "candidate": BindingGate("or", ("permit",)),
        "corroborated": BindingGate("disjoint_and", ("candidate", "candidate")),
    }
    revised = {**original, "match": BindingGate("disjoint_and", ("inspection", "permit")),
               "candidate": BindingGate("or", ("permit", "inspection"))}

    def frame(encode):
        return CorpusBindingMemory.prepare(
            ("inspection", "permit"), np.eye(2), original, {}, encode=encode,
            encoder_version="program-fixture", min_similarity=.6, gap=.05,
        )

    def encode(raw):
        return [mention for line in raw.splitlines() for mention in _encode(line)]

    path = tmp_path / "program.sqlite"
    workspace = (JournaledCorpusWorkspace(path, frame(encode), create=True) if durable
                 else CorpusBindingWorkspace(frame(encode)))
    _write(workspace, {"co_report": b"permit,1,0,0.9\ninspection,1,0,0.9",
                       "uncertain": b"inspection,1,1,0.7"})
    for node in ("release", "audit", "corroborated"):
        workspace.remember(workspace.snapshot(), node, 0)
    base = workspace.snapshot()
    candidate = workspace.prepare_program(base, revised)
    assert workspace.snapshot() is base
    assert base.memory.resolve("release", 0, work_limit=1) == CorpusBindingRead(
        BindingProof(.9, frozenset({"co_report"})), .9,
    )
    assert candidate.preview.resolve("audit", 0, work_limit=1) == base.memory.resolve("audit", 0, work_limit=1)
    with pytest.raises(BindingCapacityError):
        candidate.preview.resolve("release", 0, work_limit=1)
    assert candidate.preview.resolve("release", 0) == CorpusBindingRead(None, .7)
    assert base.memory.resolve("corroborated", 0, work_limit=1) == CorpusBindingRead(None, None)
    assert candidate.preview.resolve("corroborated", 0) == CorpusBindingRead(None, .7)

    installed = workspace.publish_program(candidate)
    assert installed.revision == base.revision + 1
    assert installed.memory is candidate.preview
    assert installed.memory.source_digests is base.memory.source_digests
    for old, new in ((base.memory._certain, installed.memory._certain),
                     (base.memory._possible, installed.memory._possible)):
        assert new._rows is old._rows and new._sources is old._sources
    removed = _write(workspace, {"uncertain": None})
    assert removed.memory.resolve("release", 0) == CorpusBindingRead(None, None)
    restored = _write(workspace, {"uncertain": b"inspection,1,0,0.8"})
    assert restored.memory.resolve("release", 0) == CorpusBindingRead(
        BindingProof(.8, frozenset({"co_report", "uncertain"})), .8,
    )
    assert restored.memory.resolve("corroborated", 0) == restored.memory.resolve("release", 0)
    for origin in ("uncertain", "co_report"):
        assert restored.memory.resolve("corroborated", 0, without_sources=(origin,)) == CorpusBindingRead(None, None)
    if durable:
        fresh = JournaledCorpusWorkspace(path, frame(_no_encoding)).snapshot()
        assert fresh.revision == restored.revision
        assert fresh.memory.resolve("release", 0) == restored.memory.resolve("release", 0)
        assert fresh.memory.resolve("corroborated", 0) == restored.memory.resolve("corroborated", 0)
    rollback = workspace.publish_program(workspace.prepare_program(restored, original))
    assert rollback.memory.resolve("release", 0) == base.memory.resolve("release", 0)
    assert rollback.memory.source_digests == restored.memory.source_digests
    if durable:
        fresh = JournaledCorpusWorkspace(path, frame(_no_encoding)).snapshot()
        assert fresh.revision == rollback.revision
        assert fresh.memory.resolve("release", 0) == rollback.memory.resolve("release", 0)


def test_program_publication_fences_drafts_and_allows_source_local_rebase(tmp_path):
    path = tmp_path / "program.sqlite"
    first = JournaledCorpusWorkspace(path, _frame(), create=True)
    second = JournaledCorpusWorkspace(path, _frame())
    earlier = first.snapshot()
    earlier_graph = first.prepare_program(earlier, {})
    _write(second, {"permit": b"permit,1,0,0.9"})
    with pytest.raises(StaleCorpusRevision):
        first.publish_program(earlier_graph)

    barrier = first.snapshot()
    sensor = {"sensor": b"inspection,1,0,0.7"}
    local = first.prepare_changes(barrier, sensor, allow_program_rebase=True)
    ordinary = first.prepare_changes(barrier, sensor)
    whole = first.prepare_changes(barrier, sensor, require_tip=True)
    doomed = first.prepare_program(barrier, {})
    program = {"release": BindingGate("and", ("permit", "inspection"))}
    second_base = second.snapshot()
    installed = second.publish_program(second.prepare_program(second_base, program))
    assert installed.revision == 2
    with pytest.raises(StaleCorpusRevision):
        first.publish_program(doomed)
    with pytest.raises(StaleCorpusRevision):
        first.publish(whole)
    with pytest.raises(StaleCorpusRevision, match="program changed"):
        first.publish(ordinary)
    assert second.publish_program(second.prepare_program(installed, program)) is installed
    result = first.publish(local)
    assert result.revision == 3
    assert result.memory.resolve("release", 0).certain.strength == .7
    before_restore = first.prepare_changes(barrier, {"other": b"permit,0,1,0.8"})
    recovered = second.snapshot()
    original = {"release": BindingGate("disjoint_and", ("inspection", "permit"))}
    second.publish_program(second.prepare_program(recovered, original))
    with pytest.raises(StaleCorpusRevision, match="program changed"):
        first.publish(before_restore)


def _program_with_extension():
    return {"release": BindingGate("disjoint_and", ("inspection", "permit")),
            "extension": BindingGate("or", ("inspection",))}


def _extension_target():
    return CorpusProgramExpectation(CorpusProgramProbe((("permit", 0), ("extension", 1))),
                                    CorpusBindingRead(BindingProof(.8, frozenset({"license", "sensor"})), .8))


class _FractionSubclass(Fraction):
    pass


class _FloatAlias(type(Fraction)):
    def __eq__(cls, other):
        return other is float or cls is other


class _FractionWithFloatAlias(Fraction, metaclass=_FloatAlias):
    pass


@pytest.mark.parametrize("grade_type", [Fraction, _FractionSubclass, _FractionWithFloatAlias])
@pytest.mark.parametrize("rounded_bound", ["certain", "possible"])
def test_program_trial_requires_builtin_grade(tmp_path, rounded_bound, grade_type):
    workspace = JournaledCorpusWorkspace(tmp_path / "precise-program.sqlite", _frame(), create=True)
    rounded = grade_type(2 ** 54 + 1, 2 ** 55)
    lower = rounded if rounded_bound == "certain" else .5
    upper = rounded if rounded_bound == "possible" else .75
    target = CorpusProgramExpectation(CorpusProgramProbe((("extension", 1),)),
                                      CorpusBindingRead(BindingProof(lower, frozenset({"sensor"})), upper))
    with pytest.raises(ValueError, match="built-in floats or integers"):
        workspace.begin_program_trial(workspace.snapshot(), _program_with_extension(), targets=(target,),
                                      future_batches=1, work_limit=65536)
    assert workspace.pending_program_trials(limit=1) == ()


@pytest.mark.parametrize("role", ["target", "protected"])
def test_program_trial_requires_builtin_node_names(tmp_path, role):
    class NodeName(str):
        pass

    workspace = JournaledCorpusWorkspace(tmp_path / "named-program.sqlite", _frame(), create=True)
    probe = CorpusProgramProbe(((NodeName("permit"), 0),))
    targets = (CorpusProgramExpectation(probe, CorpusBindingRead(None, None)),) if role == "target" else (
        _extension_target(),)
    protected = (probe,) if role == "protected" else ()
    with pytest.raises(ValueError, match="built-in strings"):
        workspace.begin_program_trial(workspace.snapshot(), _program_with_extension(), targets=targets,
                                      protected=protected, future_batches=1, work_limit=65536)
    assert workspace.pending_program_trials(limit=1) == ()


def test_frozen_program_trial_installs_on_future_sources_and_recovers_receipt(tmp_path, monkeypatch):
    path = tmp_path / "prospective-program.sqlite"
    writer = JournaledCorpusWorkspace(path, _frame(), create=True)
    before = _write(writer, {"license": b"permit,1,0,1"})
    assert writer.pending_program_trials(limit=1) == ()
    program, targets = _program_with_extension(), [_extension_target()]
    preview = writer.prepare_program(before, program).preview
    assert preview.resolve_disjoint(targets[0].probe.addresses) == CorpusBindingRead(None, None)
    trial = writer.begin_program_trial(
        before, program, targets=targets, protected=(CorpusProgramProbe((("permit", 0),)),),
        future_batches=2, work_limit=65536,
    )
    assert trial.status == "opened" and trial.observed_revision == before.revision == 1
    program.clear()
    targets.clear()
    reader = JournaledCorpusWorkspace(path, _frame(_no_encoding))
    assert reader.snapshot().revision == before.revision
    assert reader.pending_program_trials(limit=1) == (trial,)
    assert reader.pending_program_trials(limit=1, after=trial.trial_id) == ()
    assert reader.finish_program_trial(trial.trial_id).status == "pending"

    _write(writer, {"unrelated": b"permit,0,1,0.6"})
    assert reader.finish_program_trial(trial.trial_id).status == "pending"
    unchanged = _write(writer, {"unrelated": b"permit,0,1,0.6"})
    assert unchanged.revision == 2
    assert reader.finish_program_trial(trial.trial_id).status == "pending"
    ready = _write(writer, {"sensor": b"inspection,0,1,0.8"})
    rival = JournaledCorpusWorkspace(path, _frame(_no_encoding))
    barrier = Barrier(2)
    evaluate = corpus_binding_journal._check_program_trial

    def concur(*args):
        result = evaluate(*args)
        barrier.wait(timeout=10)
        return result

    monkeypatch.setattr(corpus_binding_journal, "_check_program_trial", concur)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(handle.finish_program_trial, trial.trial_id) for handle in (reader, rival)]
        published, repeated = [future.result(timeout=10) for future in futures]
    assert repeated == published
    assert published.status == "published" and published.observed_revision == 3 and published.published_revision == 4
    assert reader.pending_program_trials(limit=1) == ()
    revised = reader.snapshot()
    target = _extension_target()
    assert revised.memory.resolve_disjoint(target.probe.addresses) == target.expected
    assert revised.memory.resolve_disjoint(target.probe.addresses, without_sources=("sensor",)).is_exact
    assert revised.memory.resolve_disjoint(target.probe.addresses, without_sources=("sensor",)).certain is None
    assert revised.memory.source_digests == ready.memory.source_digests
    assert "extension" not in ready.memory._certain._dependents
    assert "extension" not in before.memory._certain._dependents

    fresh = JournaledCorpusWorkspace(path, _frame(_no_encoding))
    assert fresh.finish_program_trial(trial.trial_id) == published
    assert fresh.snapshot().memory.resolve_disjoint(target.probe.addresses) == target.expected
    assert fresh.snapshot().revision == published.published_revision
    _write(writer, {"sensor": None})
    assert fresh.finish_program_trial(trial.trial_id) == published
    assert fresh.snapshot().memory.resolve_disjoint(target.probe.addresses) == CorpusBindingRead(None, None)


@pytest.mark.parametrize("decision", [
    "protected_changed", "target_mismatch", "no_target_gain", "capacity", "new_absence",
])
def test_program_trial_rejects_failed_checks_without_installation(tmp_path, decision):
    path = tmp_path / "rejected-program.sqlite"
    journal = JournaledCorpusWorkspace(path, _frame(), create=True)
    base = _write(journal, {"license": b"permit,1,0,1", "present": b"inspection,1,0,0.7"})
    gates, target, checks, limit = _program_with_extension(), _extension_target(), (), 65536
    if decision == "protected_changed":
        gates["release"] = BindingGate("or", ("inspection", "permit"))
        checks = (CorpusProgramProbe((("release", 0),)),)
    if decision in ("capacity", "no_target_gain"):
        probe = CorpusProgramProbe((("permit", 0),))
        target = CorpusProgramExpectation(probe, base.memory.resolve("permit", 0))
        if decision == "capacity":
            limit = 1
    if decision == "new_absence":
        target = CorpusProgramExpectation(CorpusProgramProbe((("extension", 1),)), CorpusBindingRead(None, None))
    trial = journal.begin_program_trial(base, gates, targets=(target,), protected=checks,
                                        future_batches=1, work_limit=limit)
    if decision == "capacity":
        cold_trial = journal.begin_program_trial(base, gates, targets=(target,), future_batches=1, work_limit=limit)
    late = b"permit,0,1,0.8" if decision == "new_absence" else b"inspection,1,1,0.8"
    ready = _write(journal, {"sensor": late})
    if decision == "capacity":
        warm = journal.remember(ready, "permit", 0)
        assert warm.memory.resolve("permit", 0, work_limit=1) == target.expected
    rejected = journal.finish_program_trial(trial.trial_id)
    expected = "no_target_gain" if decision == "new_absence" else decision
    assert rejected.status == expected and rejected.published_revision is None
    assert journal.snapshot().revision == ready.revision
    fresh = JournaledCorpusWorkspace(path, _frame(_no_encoding))
    if decision == "capacity":
        cold = fresh.finish_program_trial(cold_trial.trial_id)
        assert (cold.status, cold.observed_revision) == (rejected.status, rejected.observed_revision)
    assert fresh.finish_program_trial(trial.trial_id) == rejected
    assert "extension" not in fresh.snapshot().memory._certain._dependents


def test_program_trial_target_can_require_abstaining_from_a_stronger_incumbent(tmp_path):
    path = tmp_path / "abstention-program.sqlite"

    def encode(raw):
        return [mention for line in raw.splitlines() for mention in _encode(line)]

    original = {"release": BindingGate("and", ("inspection", "permit"))}
    frame = _frame(encode).reprogram(original)
    writer = JournaledCorpusWorkspace(path, frame, create=True)
    base = _write(writer, {"report": b"inspection,1,0,0.9\npermit,1,0,0.9",
                           "sensor": b"inspection,1,1,0.7"})
    probe = CorpusProgramProbe((("release", 0),))
    assert base.memory.resolve("release", 0) == CorpusBindingRead(BindingProof(.9, frozenset({"report"})), .9)
    target = CorpusProgramExpectation(probe, CorpusBindingRead(None, .7))
    candidate = {"release": BindingGate("disjoint_and", ("inspection", "permit"))}
    options = dict(targets=(target,), future_batches=1, work_limit=65536)
    protected = writer.begin_program_trial(base, candidate, protected=(probe,), **options)
    explicitly_changed = writer.begin_program_trial(base, candidate, **options)
    assert writer.pending_program_trials(limit=1) == (protected,)
    assert writer.pending_program_trials(limit=1, after=protected.trial_id) == (explicitly_changed,)
    _write(writer, {"other": b"permit,0,1,1"})
    assert writer.finish_program_trial(protected.trial_id).status == "protected_changed"
    accepted = writer.finish_program_trial(explicitly_changed.trial_id)
    assert accepted.status == "published"
    assert writer.snapshot().memory.resolve("release", 0) == target.expected


def test_program_trial_refuses_a_tip_change_during_the_check(tmp_path, monkeypatch):
    path = tmp_path / "racing-program.sqlite"
    writer = JournaledCorpusWorkspace(path, _frame(), create=True)
    base = _write(writer, {"license": b"permit,1,0,1"})
    trial = writer.begin_program_trial(base, _program_with_extension(), targets=(_extension_target(),),
                                       future_batches=1, work_limit=65536)
    _write(writer, {"sensor": b"inspection,0,1,0.8"})
    follower = JournaledCorpusWorkspace(path, _frame(_no_encoding))
    evaluate = corpus_binding_journal._check_program_trial

    def check_and_edit(*args):
        result = evaluate(*args)
        assert result == "published"
        _write(writer, {"newer": b"permit,0,1,0.7"})
        return result

    monkeypatch.setattr(corpus_binding_journal, "_check_program_trial", check_and_edit)
    with pytest.raises(StaleCorpusRevision, match="while checking trial"):
        follower.finish_program_trial(trial.trial_id)
    result = follower.finish_program_trial(trial.trial_id)
    assert result.status == "horizon_passed"
    assert follower.snapshot().revision == 3
    assert "extension" not in follower.snapshot().memory._certain._dependents


def test_program_trial_keeps_a_concurrent_handle_local_proof(tmp_path, monkeypatch):
    path = tmp_path / "retained-during-program.sqlite"
    journal = JournaledCorpusWorkspace(path, _frame(), create=True)
    base = _write(journal, {"license": b"permit,1,0,1"})
    trial = journal.begin_program_trial(base, _program_with_extension(), targets=(_extension_target(),),
                                        future_batches=1, work_limit=65536)
    ready = _write(journal, {"sensor": b"inspection,0,1,0.8"})
    evaluate = corpus_binding_journal._check_program_trial

    def check_and_remember(*args):
        result = evaluate(*args)
        retained = journal.remember(ready, "permit", 0)
        assert retained is not ready and retained.revision == ready.revision
        return result

    monkeypatch.setattr(corpus_binding_journal, "_check_program_trial", check_and_remember)
    result = journal.finish_program_trial(trial.trial_id)
    assert result.status == "published"
    installed = journal.snapshot()
    assert installed.memory.resolve("permit", 0, work_limit=1) == ready.memory.resolve("permit", 0)


def test_program_trial_detects_competing_wiring_even_after_rollback(tmp_path):
    path = tmp_path / "competing-program.sqlite"
    journal = JournaledCorpusWorkspace(path, _frame(), create=True)
    base = journal.snapshot()
    trial = journal.begin_program_trial(base, _program_with_extension(), targets=(_extension_target(),),
                                        future_batches=2, work_limit=65536)
    other = journal.publish_program(journal.prepare_program(base, {}))
    restored = journal.publish_program(journal.prepare_program(other, dict(base.memory._certain._gates)))
    fresh = JournaledCorpusWorkspace(path, _frame(_no_encoding))
    outcome = fresh.finish_program_trial(trial.trial_id)
    assert outcome.status == "competing_program" and outcome.published_revision is None
    assert fresh.snapshot().revision == restored.revision


def test_program_trial_append_and_receipt_rollback_together(tmp_path):
    path = tmp_path / "atomic-program.sqlite"
    journal = JournaledCorpusWorkspace(path, _frame(), create=True)
    base = _write(journal, {"license": b"permit,1,0,1"})
    trial = journal.begin_program_trial(base, _program_with_extension(), targets=(_extension_target(),),
                                        future_batches=1, work_limit=65536)
    ready = _write(journal, {"sensor": b"inspection,0,1,0.8"})
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("""CREATE TRIGGER fail_program_trial BEFORE UPDATE ON corpus_program_trials
                      BEGIN SELECT RAISE(ABORT, 'trial receipt unavailable'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="trial receipt unavailable"):
        journal.finish_program_trial(trial.trial_id)
    fresh = JournaledCorpusWorkspace(path, _frame(_no_encoding))
    assert fresh.snapshot().revision == ready.revision
    assert "extension" not in fresh.snapshot().memory._certain._dependents
    with closing(sqlite3.connect(path)) as db, db:
        assert db.execute("SELECT status FROM corpus_program_trials").fetchone() == (None,)
        db.execute("DROP TRIGGER fail_program_trial")
    receipt = fresh.finish_program_trial(trial.trial_id)
    assert receipt.status == "published"
    assert journal.finish_program_trial(trial.trial_id) == receipt


def test_unchanged_or_cyclic_graph_events_are_rejected_even_with_valid_hash(tmp_path):
    path = tmp_path / "program.sqlite"
    frame = _frame(_no_encoding)
    journal = JournaledCorpusWorkspace(path, frame, create=True)
    initial = journal.snapshot()
    for payload in (b'["program",[["release","disjoint_and",["inspection","permit"]]]]',
                    b'["program",[["loop","and",["loop"]]]]'):
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("INSERT OR REPLACE INTO corpus_events VALUES (1, ?, ?, ?)",
                       (journal._digest, _event_digest(journal._digest, 1, payload), payload))
        for read in (journal.snapshot, lambda: JournaledCorpusWorkspace(path, frame)):
            with pytest.raises(ValueError, match="unchanged graph|parents must precede"):
                read()
        assert journal._tip is initial


def test_program_preview_freezes_gate_definitions():
    workspace = CorpusBindingWorkspace(_frame())
    base = _write(workspace, {"q": b"inspection,1,0,0.8"})
    parents = ["inspection", "permit"]
    with pytest.raises(ValueError, match="unknown binding gate"):
        workspace.prepare_program(base, {"release": BindingGate(np.array("and"), parents)})
    draft = workspace.prepare_program(base, {"release": BindingGate("and", parents)})
    parents.pop()
    assert workspace.publish_program(draft).memory.resolve("release", 0) == CorpusBindingRead(None, None)


def test_json_identifier_aliases_cannot_poison_the_journal(tmp_path):
    surrogate_pair, character = "\ud83d\ude00", "\U0001f600"
    frame = _frame()
    path = tmp_path / "program.sqlite"
    with pytest.raises(ValueError, match="JSON round trip"):
        overloaded_frame = frame.reprogram({surrogate_pair: BindingGate("or", ("permit",))})
        JournaledCorpusWorkspace(path, overloaded_frame, create=True)
    assert not path.exists()
    journal = JournaledCorpusWorkspace(path, frame, create=True)
    base = journal.snapshot()
    raw = b"permit,1,0,0.8"
    with pytest.raises(ValueError, match="JSON round trip"):
        _write(journal, {surrogate_pair: raw, character: raw})
    gates = {name: BindingGate("or", ("permit",)) for name in (surrogate_pair, character)}
    with pytest.raises(ValueError, match="JSON round trip"):
        journal.publish_program(journal.prepare_program(base, gates))
    assert journal.snapshot() is base
    seeded = _write(journal, {character: raw})
    journal.publish_program(journal.prepare_program(seeded, {character: gates[character]}))
    fresh = JournaledCorpusWorkspace(path, _frame(_no_encoding)).snapshot()
    assert fresh.revision == 2
    assert fresh.memory.resolve(character, 0) == CorpusBindingRead(BindingProof(.8, frozenset({character})), .8)


def test_fresh_process_commits_and_replay_never_encodes_old_sources(tmp_path):
    path = tmp_path / "corpus.sqlite"
    parent = JournaledCorpusWorkspace(path, _frame(), create=True)
    initial = _write(parent, {"permit": b"permit,1,0,0.9"})
    delayed = parent.prepare_changes(initial, {"correction": b"inspection,1,0,0.6"})
    script = """
import os
import sys
from experiments.memory.corpus_binding_journal import JournaledCorpusWorkspace
from experiments.memory.test_corpus_binding_journal import _frame
workspace = JournaledCorpusWorkspace(sys.argv[1], _frame())
assert workspace.snapshot().revision == 1
draft = workspace.prepare_changes(workspace.snapshot(), {"sensor": b"inspection,1,1,0.8"})
connect = workspace._connect

class ExitAfterCommit:
    def __init__(self):
        self.connection = connect()

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def commit(self):
        self.connection.commit()
        os._exit(0)

workspace._connect = ExitAfterCommit
workspace.publish(draft)
raise AssertionError("child returned after commit")
"""
    child = subprocess.run([sys.executable, "-B", "-c", script, str(path)],
                           cwd=Path(__file__).resolve().parents[2], capture_output=True, timeout=20)
    assert child.returncode == 0, child.stderr.decode()
    revived = JournaledCorpusWorkspace(path, _frame(_no_encoding)).snapshot()
    assert revived.revision == 2
    assert revived.changed_sources == ("sensor",)
    mention, = revived.memory.source_mentions("sensor")
    assert mention.predicate == "inspection" and mention.strength == .8
    assert np.allclose(mention.vector, np.array([1., 1.]) / np.sqrt(2.))
    assert revived.memory.resolve("release", 0) == CorpusBindingRead(None, .8)
    combined = parent.publish(delayed)
    assert combined.revision == 3
    assert combined.memory.resolve("release", 0) == CorpusBindingRead(
        BindingProof(.6, frozenset({"permit", "correction"})), .8,
    )
    assert initial.memory.resolve("release", 0) == CorpusBindingRead(None, None)
    with closing(sqlite3.connect(path)) as db:
        events = db.execute("SELECT payload FROM corpus_events ORDER BY revision").fetchall()
    assert len(events) == 3
    assert all(b"permit,1,0,0.9" not in payload and b"inspection,1,1,0.8" not in payload for payload, in events)


def test_replay_fences_delete_restore_and_absent_source_changes(tmp_path):
    path = tmp_path / "corpus.sqlite"
    raw = b"permit,1,0,0.9"
    first = JournaledCorpusWorkspace(path, _frame(), create=True)
    base = _write(first, {"permit": raw})
    second = JournaledCorpusWorkspace(path, _frame())
    noop = first.prepare_changes(base, {"permit": raw, "missing": None})
    draft = first.prepare_changes(base, {"sensor": b"inspection,1,0,0.7"}, depends_on=("permit",))
    _write(second, {"permit": None, "missing": raw})
    restored = _write(second, {"permit": raw, "missing": None})
    for proposal in (noop, draft):
        with pytest.raises(StaleCorpusRevision, match="dependency"):
            first.publish(proposal)
    assert first.snapshot().revision == restored.revision == 3
    assert first.snapshot().memory.source_digests == base.memory.source_digests
    with pytest.raises(ValueError, match="another workspace"):
        second.publish(draft)


def test_capacity_refusal_appends_nothing_and_retry_uses_the_same_encoded_rows(tmp_path):
    path = tmp_path / "corpus.sqlite"
    first = JournaledCorpusWorkspace(path, _frame(term_limit=1), create=True)
    initial = _write(first, {"p": b"permit,1,0,0.9"})
    second = JournaledCorpusWorkspace(path, _frame(term_limit=1))
    staged = first.prepare_changes(initial, {"q": b"inspection,1,0,0.8"})
    other = _write(second, {"r": b"inspection,1,1,0.7"})
    with pytest.raises(BindingCapacityError):
        first.publish(staged)
    unchanged = JournaledCorpusWorkspace(path, _frame(_no_encoding, term_limit=1)).snapshot()
    assert unchanged.revision == other.revision == 2
    assert unchanged.memory.source_digests == other.memory.source_digests
    assert unchanged.memory.resolve("release", 0) == CorpusBindingRead(None, .7)
    _write(second, {"r": None})
    assert first.publish(staged).revision == 4
    assert second.snapshot().memory.resolve("release", 0).certain.strength == .8


def test_independent_handles_serialize_same_origin_publication(tmp_path):
    path = tmp_path / "corpus.sqlite"
    first = JournaledCorpusWorkspace(path, _frame(), create=True)
    second = JournaledCorpusWorkspace(path, _frame())
    ready = Barrier(2, timeout=5)

    def submit(owner, raw):
        proposal = owner.prepare_changes(owner.snapshot(), {"p": raw})
        ready.wait()
        try:
            return owner.publish(proposal)
        except StaleCorpusRevision:
            return None

    with ThreadPoolExecutor(max_workers=2) as workers:
        tasks = [workers.submit(submit, owner, raw) for owner, raw in (
            (first, b"permit,1,0,0.8"), (second, b"permit,0,1,0.7"),
        )]
        results = [task.result(timeout=10) for task in tasks]
    assert sum(result is not None for result in results) == 1
    assert first.snapshot().revision == second.snapshot().revision == 1
    assert first.snapshot().memory.source_digests == second.snapshot().memory.source_digests


def test_proof_retention_is_local_and_frame_and_event_mismatches_are_rejected(tmp_path):
    path = tmp_path / "corpus.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        JournaledCorpusWorkspace(path, _frame())
    assert not path.exists()
    owner = JournaledCorpusWorkspace(path, _frame(), create=True)
    initial = _write(owner, {"permit": b"permit,1,0,0.9"})
    assert JournaledCorpusWorkspace(path, _frame(_no_encoding), create=True).snapshot().revision == 1
    promoted = owner.remember(initial, "release", 0)
    assert promoted.revision == initial.revision
    assert promoted.memory.resolve("release", 0, work_limit=1) == CorpusBindingRead(None, None)
    fresh = JournaledCorpusWorkspace(path, _frame(_no_encoding)).snapshot()
    with pytest.raises(BindingCapacityError):
        fresh.memory.resolve("release", 0, work_limit=1)
    with pytest.raises(ValueError, match="frame mismatch"):
        JournaledCorpusWorkspace(path, _frame(version="other"))
    with closing(sqlite3.connect(path)) as db:
        db.execute("UPDATE corpus_events SET payload = ? WHERE revision = 1", (b"[]",))
        db.commit()
    with pytest.raises(ValueError, match="event chain mismatch"):
        JournaledCorpusWorkspace(path, _frame(_no_encoding))

    partial = tmp_path / "partial.sqlite"
    with closing(sqlite3.connect(partial)) as db:
        db.execute("CREATE TABLE corpus_events (revision INTEGER PRIMARY KEY)")
        db.commit()
    with pytest.raises(ValueError, match="schema is incomplete"):
        JournaledCorpusWorkspace(partial, _frame(), create=True)
    with closing(sqlite3.connect(partial)) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall() == [("corpus_events",)]


@pytest.mark.parametrize("program", [False, True])
def test_append_error_leaves_the_pinned_tip_and_a_retryable_proposal(tmp_path, program):
    path = tmp_path / "corpus.sqlite"
    journal = JournaledCorpusWorkspace(path, _frame(), create=True)
    base = _write(journal, {"permit": b"permit,1,0,0.9"})
    if program:
        prepared = journal.prepare_program(base, {"release": BindingGate("or", ("permit",))})
        publish, expected = journal.publish_program, .9
    else:
        prepared = journal.prepare_changes(base, {"sensor": b"inspection,1,0,0.7"})
        publish, expected = journal.publish, .7
    with closing(sqlite3.connect(path)) as db:
        db.execute("""CREATE TRIGGER refuse_insert BEFORE INSERT ON corpus_events
            BEGIN SELECT RAISE(ABORT, 'fixture append refusal'); END""")
        db.commit()
    with pytest.raises(sqlite3.IntegrityError, match="fixture append refusal"):
        publish(prepared)
    assert journal.snapshot() is base
    assert JournaledCorpusWorkspace(path, _frame(_no_encoding)).snapshot().revision == 1
    with closing(sqlite3.connect(path)) as db:
        db.execute("DROP TRIGGER refuse_insert")
        db.commit()
    assert publish(prepared).memory.resolve("release", 0).certain.strength == expected


def test_copied_read_suffix_does_not_block_another_handles_commit(tmp_path, monkeypatch):
    path = tmp_path / "corpus.sqlite"
    writer = JournaledCorpusWorkspace(path, _frame(), create=True)
    reader = JournaledCorpusWorkspace(path, _frame(_no_encoding))
    _write(writer, {"p": b"permit,1,0,0.9"})
    started, release = Event(), Event()
    replay = JournaledCorpusWorkspace._replay

    def delayed(owner, suffix):
        if owner is reader and not release.is_set():
            started.set()
            if not release.wait(5):
                raise TimeoutError("fixture did not release replay")
        return replay(owner, suffix)

    monkeypatch.setattr(JournaledCorpusWorkspace, "_replay", delayed)
    with ThreadPoolExecutor(max_workers=1) as worker:
        pending = worker.submit(reader.snapshot)
        try:
            assert started.wait(5)
            final = _write(writer, {"q": b"inspection,1,0,0.7"})
        finally:
            release.set()
        copied = pending.result(timeout=5)
    assert copied.revision == 1 and final.revision == 2
    assert copied.memory.resolve("release", 0) == CorpusBindingRead(None, None)
    assert reader.snapshot().memory.resolve("release", 0).certain.strength == .7



def test_row_only_history_replays_without_fabricating_observations(tmp_path, monkeypatch):
    path = tmp_path / "legacy.sqlite"
    old_writer = JournaledCorpusWorkspace(path, _frame(), create=True)
    new_encoder = corpus_binding_journal._encode
    with monkeypatch.context() as patched:
        patched.setattr(corpus_binding_journal, "_encode", lambda edits, **kw: new_encoder(edits, vectors=False))
        _write(old_writer, {"permit": b"permit,1,0,0.9"})

    reader = JournaledCorpusWorkspace(path, _frame(_no_encoding))
    legacy = reader.snapshot()
    assert legacy.memory.source_mentions("permit") is None
    assert legacy.memory.resolve("permit", 0) == CorpusBindingRead(BindingProof(.9, frozenset({"permit"})), .9)
    with pytest.raises(ValueError, match="legacy sources"):
        legacy.memory._observation_rows(1)
    assert _write(reader, {"permit": b"permit,1,0,0.9"}) is legacy

    writer = JournaledCorpusWorkspace(path, _frame())
    replaced = _write(writer, {"permit": b"permit,1,0,0.8"})
    assert reader.snapshot().memory.source_mentions("permit")[0].strength == .8
    assert np.array_equal(replaced.memory._observation_rows(1), [[1., 0.]])
    assert legacy.memory.source_mentions("permit") is None
    removed = _write(writer, {"permit": None})
    assert reader.snapshot().revision == removed.revision
    with pytest.raises(KeyError):
        reader.snapshot().memory.source_mentions("permit")


@pytest.mark.parametrize("corruption, error", [("nonfinite", "unit vector"), ("routing", "envelope")])
def test_recovery_checks_observations_even_with_a_matching_event_hash(tmp_path, corruption, error):
    path = tmp_path / "corpus.sqlite"
    writer = JournaledCorpusWorkspace(path, _frame(), create=True)
    _write(writer, {"permit": b"permit,1,0,0.9"})
    with closing(sqlite3.connect(path)) as db:
        revision, parent, payload = db.execute("SELECT revision, parent, payload FROM corpus_events").fetchone()
        encoded = json.loads(payload)
        mentions = encoded[1][0][5]
        if corruption == "nonfinite":
            mentions[0][2] = b64encode(np.array([np.nan, 0.], dtype="<f8").tobytes()).decode("ascii")
        else:
            encoded[1][0][3] = []
        altered = corpus_binding_journal._json(encoded)
        db.execute("UPDATE corpus_events SET digest = ?, payload = ? WHERE revision = ?",
                   (_event_digest(parent, revision, altered), altered, revision))
        db.commit()
    with pytest.raises(ValueError, match=error):
        JournaledCorpusWorkspace(path, _frame(_no_encoding))



def test_source_vector_byte_budget_is_separate_from_expanded_rows(tmp_path, monkeypatch):
    def encode(raw):
        return () if raw == b"no mentions" else _encode(raw)

    path = tmp_path / "budget.sqlite"
    owner = JournaledCorpusWorkspace(path, _frame(encode), create=True)
    monkeypatch.setattr(corpus_bindings, "_SOURCE_VECTOR_BYTES", 16)
    with pytest.raises(BindingCapacityError, match="vector batch byte limit"):
        _write(owner, {"a": b"permit,1,0,0.9", "b": b"inspection,1,0,0.8"})
    assert owner.snapshot().revision == 0
    pin = _write(owner, {"a": b"permit,1,0,0.9", "b": b"no mentions"})
    reader = JournaledCorpusWorkspace(path, _frame(_no_encoding))
    assert reader.snapshot().memory.source_mentions("b") == ()
    assert np.array_equal(reader.snapshot().memory._observation_rows(1), [[1., 0.]])
    monkeypatch.setattr(corpus_bindings, "_SOURCE_VECTOR_BYTES", 15)
    with pytest.raises(ValueError, match="vector batch byte limit"):
        JournaledCorpusWorkspace(path, _frame(_no_encoding))
    assert pin.memory.source_mentions("b") == ()
