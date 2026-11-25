from contextlib import closing
from math import copysign, nextafter
import sqlite3
import subprocess
import sys

import pytest

from experiments.memory.relative_hierarchy import RelativeObservationHierarchy
from experiments.memory.relative_journal import JournaledRelativeHierarchyWorkspace, StaleRelativeRevision
from experiments.memory.relative_observations import RelativeObservation, solve_relative_direct


_ORDER = tuple(range(18))
_RIDGE = 0.35
_LATE = {"row-15": RelativeObservation(15, -1.8, source=14, weight=3)}
_LATER = {"row-15": RelativeObservation(15, 2, source=14, weight=2)}


def _root():
    hierarchy = RelativeObservationHierarchy(
        tuple(cell // 3 for cell in _ORDER), ridge=_RIDGE, elimination_order=reversed(_ORDER),
    )
    sources = {"anchor": RelativeObservation(0, 0.7, weight=2)}
    sources.update({f"row-{cell}": RelativeObservation(cell, 0.13, source=cell - 1)
                    for cell in _ORDER[1:]})
    return hierarchy, sources


def _open(path, *, create=False):
    return JournaledRelativeHierarchyWorkspace(path, *_root(), create=create)


def _assert_values(head, expected):
    reference = solve_relative_direct(18, expected, ridge=_RIDGE)
    for actual, target in zip(head.snapshot.read_many(_ORDER), reference):
        if target is None:
            assert actual is None
        else:
            assert actual == pytest.approx(target, abs=1e-12, rel=0)


def test_journal_replays_layout_sources_and_compaction_without_hypothetical_edits(tmp_path):
    path = tmp_path / "shared.sqlite"
    journal, follower = _open(path, create=True), _open(path)
    _, sources = _root()
    original, retained = journal.snapshot(), follower.snapshot()
    selected, receipt = journal.maintain_generated(
        original, ((_LATE, (16,)), (_LATER, (16, 17))),
        max_candidates=2, max_symbolic_ports=10, protected=(0, 17),
    )
    assert selected.number == 1 and selected.snapshot is receipt.snapshot
    assert selected.snapshot._sources is original.snapshot._sources
    assert selected.snapshot.elimination_order != original.snapshot.elimination_order
    caught_up = follower.snapshot()
    assert caught_up.digest == selected.digest
    assert caught_up.snapshot.elimination_order == selected.snapshot.elimination_order
    _assert_values(caught_up, sources)

    removed = journal.revise(selected, {"row-15": None})
    compacted = journal.compact(removed, max_regions=2, protected_cells=(0, 16))
    assert compacted.number == 3 and compacted.snapshot.coarse_size == 8
    assert journal.revise(compacted, {}) is compacted
    reopened = _open(path).snapshot()
    assert reopened.digest == follower.snapshot().digest == compacted.digest
    assert reopened.snapshot.elimination_order == selected.snapshot.elimination_order
    assert reopened.snapshot.coarse_size == 8 and reopened.snapshot.read(16) is None
    _assert_values(reopened, {name: row for name, row in sources.items() if name != "row-15"})
    _assert_values(retained, sources)


def test_separate_source_worker_fences_a_prepared_layout(tmp_path, monkeypatch):
    path = tmp_path / "race.sqlite"
    journal = _open(path, create=True)
    base = journal.snapshot()
    commit = journal._commit
    child = """
import sys
from experiments.memory.test_relative_journal import _open
from experiments.memory.relative_observations import RelativeObservation
journal = _open(sys.argv[1])
journal.revise(journal.snapshot(), {"anchor": RelativeObservation(0, 1.7, weight=2)})
"""

    def publish_after_source(parent, revised, kind, payload):
        assert kind == "layout" and revised.elimination_order != parent.snapshot.elimination_order
        subprocess.run([sys.executable, "-B", "-c", child, str(path)], check=True, timeout=30)
        return commit(parent, revised, kind, payload)

    monkeypatch.setattr(journal, "_commit", publish_after_source)
    with pytest.raises(StaleRelativeRevision, match="before publication"):
        journal.maintain_generated(base, ((_LATE, (16,)), (_LATER, (17,))),
                                   max_candidates=2, max_symbolic_ports=10)
    current = journal.snapshot()
    assert current.number == 1 and current.snapshot.elimination_order == base.snapshot.elimination_order
    assert current.snapshot.read(0) != base.snapshot.read(0)


def test_reopen_recovers_worker_commit_interrupted_before_acknowledgment(tmp_path):
    path = tmp_path / "recovery.sqlite"
    observer = _open(path, create=True)
    before = observer.snapshot()
    child = """
import os, sys
from experiments.memory import relative_journal
from experiments.memory.test_relative_journal import _open
from experiments.memory.relative_observations import RelativeObservation
journal = _open(sys.argv[1])
def vanish(*args):
    os._exit(73)
relative_journal.RelativeJournalRevision = vanish
journal.revise(journal.snapshot(), {"pin": RelativeObservation(1, -0.0)})
"""
    process = subprocess.run([sys.executable, "-B", "-c", child, str(path)], timeout=30)
    assert process.returncode == 73
    recovered = observer.snapshot()
    assert recovered.number == 1 and before.number == 0
    assert copysign(1, recovered.snapshot._sources["pin"].value) == -1
    assert recovered.digest == _open(path).snapshot().digest
    assert "pin" not in before.snapshot._sources


def test_journal_pins_root_and_refuses_corrupted_unseen_tail(tmp_path):
    path = tmp_path / "history.sqlite"
    journal, follower = _open(path, create=True), _open(path)
    base, waiting = journal.snapshot(), follower.snapshot()
    hierarchy, sources = _root()
    sources["anchor"] = RelativeObservation(0, nextafter(0.7, 1), weight=2)
    with pytest.raises(ValueError, match="initial workspace inputs"):
        JournaledRelativeHierarchyWorkspace(path, hierarchy, sources)
    revised = journal.revise(base, {"precise": RelativeObservation(2, nextafter(1, 2))})
    assert _open(path).snapshot().snapshot._sources["precise"].value.hex() == nextafter(1, 2).hex()
    with closing(sqlite3.connect(path)) as db:
        db.execute("UPDATE relative_events SET payload = ? WHERE number = ?", (b"[]", revised.number))
        db.commit()
    with pytest.raises(ValueError, match="invalid relative journal history"):
        follower.snapshot()
    assert follower._head is waiting


def test_a_worker_open_does_not_create_or_recreate_a_missing_journal(tmp_path):
    path = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        _open(path)
    assert not path.exists()
    journal = _open(path, create=True)
    path.rename(tmp_path / "moved.sqlite")
    with pytest.raises(sqlite3.OperationalError):
        journal.snapshot()
    assert not path.exists()
