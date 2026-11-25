from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess
import sys
from threading import Barrier, Event

import numpy as np
import pytest

from experiments.memory.binding_memory import BindingCapacityError, BindingGate, BindingProof
from experiments.memory.corpus_binding_journal import JournaledCorpusWorkspace
from experiments.memory.corpus_bindings import (
    BindingMention, CorpusBindingMemory, CorpusBindingRead, StaleCorpusRevision,
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


def test_append_error_leaves_the_pinned_tip_and_a_retryable_proposal(tmp_path):
    path = tmp_path / "corpus.sqlite"
    journal = JournaledCorpusWorkspace(path, _frame(), create=True)
    base = _write(journal, {"permit": b"permit,1,0,0.9"})
    prepared = journal.prepare_changes(base, {"sensor": b"inspection,1,0,0.7"})
    with closing(sqlite3.connect(path)) as db:
        db.execute("""CREATE TRIGGER refuse_insert BEFORE INSERT ON corpus_events
            BEGIN SELECT RAISE(ABORT, 'fixture append refusal'); END""")
        db.commit()
    with pytest.raises(sqlite3.IntegrityError, match="fixture append refusal"):
        journal.publish(prepared)
    assert journal.snapshot() is base
    assert JournaledCorpusWorkspace(path, _frame(_no_encoding)).snapshot().revision == 1
    with closing(sqlite3.connect(path)) as db:
        db.execute("DROP TRIGGER refuse_insert")
        db.commit()
    assert journal.publish(prepared).memory.resolve("release", 0).certain.strength == .7


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
