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
    assert caught_up.snapshot.support_witness(16) == original.snapshot.support_witness(16)
    _assert_values(caught_up, sources)

    removed = journal.revise(selected, {"row-15": None})
    compacted = journal.compact(removed, max_regions=2, protected_cells=(0, 16))
    assert compacted.number == 3 and compacted.snapshot.coarse_size == 8
    assert journal.revise(compacted, {}) is compacted
    reopened = _open(path).snapshot()
    assert reopened.digest == follower.snapshot().digest == compacted.digest
    assert reopened.snapshot.elimination_order == selected.snapshot.elimination_order
    assert reopened.snapshot.coarse_size == 8 and reopened.snapshot.read(16) is None
    assert reopened.snapshot.support_witness(16) is None
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


def test_journal_fuses_then_restores_partition_around_a_new_source_tip(tmp_path):
    path = tmp_path / "fusion.sqlite"
    journal, follower = _open(path, create=True), _open(path)
    _, sources = _root()
    parent = journal.snapshot()
    fused = journal.merge_regions(parent, 4, 5, protected_cells=_ORDER)
    assert fused.number == 1 and fused.snapshot.coarse_size == parent.snapshot.coarse_size - 2
    assert fused.snapshot._sources is parent.snapshot._sources
    assert follower.snapshot().digest == fused.digest
    assert follower.snapshot().snapshot.membership == fused.snapshot.membership
    _assert_values(fused, sources)

    revised = follower.revise(follower.snapshot(), _LATE)
    assert revised.snapshot.rebuilt_regions == (4,) and "row-15" not in revised.snapshot._bridges
    assert revised.snapshot.read(17) != fused.snapshot.read(17)
    restored = journal.repartition(journal.snapshot(), parent.snapshot.membership, protected_cells=_ORDER)
    assert restored.number == 3 and restored.snapshot.elimination_order == parent.snapshot.elimination_order
    assert "row-15" in restored.snapshot._bridges
    assert journal.repartition(restored, restored.snapshot.membership, protected_cells=_ORDER) is restored
    reopened = _open(path).snapshot()
    assert reopened.digest == restored.digest == follower.snapshot().digest
    assert reopened.snapshot.membership == parent.snapshot.membership
    _assert_values(reopened, sources | _LATE)
    _assert_values(parent, sources)


def test_guarded_merge_refuses_changed_reads_and_fences_a_competing_source(tmp_path, monkeypatch):
    path = tmp_path / "fusion-fence.sqlite"
    journal, writer = _open(path, create=True), _open(path)
    parent = journal.snapshot()
    read_many = type(parent.snapshot).read_many

    def drift(snapshot, cells, **options):
        values = read_many(snapshot, cells, **options)
        if snapshot.membership != parent.snapshot.membership:
            return tuple(value + 1 for value in values)
        return values

    with monkeypatch.context() as fault:
        fault.setattr(type(parent.snapshot), "read_many", drift)
        with pytest.raises(ValueError, match="altered a protected read"):
            journal.merge_regions(parent, 4, 5, protected_cells=(17,))
    assert journal.snapshot() is parent
    commit = journal._commit

    def publish_after_source(base, prepared, kind, payload):
        assert kind == "merge_regions"
        writer.revise(writer.snapshot(), _LATE)
        return commit(base, prepared, kind, payload)

    monkeypatch.setattr(journal, "_commit", publish_after_source)
    with pytest.raises(StaleRelativeRevision, match="before publication"):
        journal.merge_regions(parent, 4, 5, protected_cells=(17,))
    current = journal.snapshot()
    assert current.number == 1 and current.snapshot.membership == parent.snapshot.membership
    assert current.snapshot._sources["row-15"] == _LATE["row-15"]


def test_retained_fusion_replays_child_edits_and_splits_current_partition(tmp_path):
    path = tmp_path / "retained-fusion.sqlite"
    journal, observer = _open(path, create=True), _open(path)
    _, sources = _root()
    parent = journal.snapshot()
    fused = journal.merge_regions(parent, 4, 5, retain_children=True, protected_cells=_ORDER)
    assert fused.snapshot._regions[4].fusion is not None

    child = """
import sys
from experiments.memory.test_relative_journal import _LATE, _open
journal = _open(sys.argv[1])
head = journal.snapshot()
assert head.snapshot._regions[4].fusion is not None
journal.revise(head, _LATE)
"""
    subprocess.run([sys.executable, "-B", "-c", child, str(path)], check=True, timeout=30)

    with pytest.raises(StaleRelativeRevision, match="before preparation"):
        journal.split_region(fused, 4, protected_cells=_ORDER)
    changed = journal.snapshot()
    assert changed.number == 2 and changed.snapshot._regions[4].fusion.bridges["row-15"] == _LATE["row-15"]
    separated = journal.split_region(changed, 4, protected_cells=_ORDER)
    assert separated.number == 3 and separated.snapshot.membership == parent.snapshot.membership
    assert separated.snapshot._bridges["row-15"] == _LATE["row-15"]
    assert observer.snapshot().digest == separated.digest == _open(path).snapshot().digest
    _assert_values(observer.snapshot(), sources | _LATE)
    _assert_values(fused, sources)

    with closing(sqlite3.connect(path)) as db:
        history = db.execute("SELECT kind FROM relative_events ORDER BY number").fetchall()
    assert history == [("merge_retained",), ("sources",), ("split_region",)]
    with pytest.raises(ValueError, match="with retained children"):
        journal.split_region(separated, 4)
    assert journal.snapshot() is separated


def test_lazy_fusion_journal_recovers_after_source_revision_and_split(tmp_path):
    path = tmp_path / "lazy-fusion.sqlite"
    journal, observer = _open(path, create=True), _open(path)
    _, sources = _root()
    parent = journal.snapshot()
    with pytest.raises(ValueError, match="requires retained children"):
        journal.merge_regions(parent, 4, 5, lazy_readout=True)
    fused = journal.merge_regions(parent, 4, 5, retain_children=True, lazy_readout=True, protected_cells=_ORDER)
    assert fused.snapshot._regions[4].lazy_readout
    assert observer.snapshot().snapshot._regions[4].lazy_readout
    updated = observer.revise(observer.snapshot(), _LATE)
    assert updated.snapshot._regions[4].lazy_readout
    reopened = _open(path)
    recovered = reopened.snapshot()
    assert recovered.digest == updated.digest and recovered.snapshot._regions[4].lazy_readout
    _assert_values(recovered, sources | _LATE)

    split = reopened.split_region(recovered, 4, protected_cells=_ORDER)
    assert not split.snapshot._regions[4].lazy_readout
    assert journal.snapshot().digest == _open(path).snapshot().digest == split.digest
    _assert_values(split, sources | _LATE)
    _assert_values(fused, sources)
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT kind FROM relative_events ORDER BY number").fetchall() == [
            ("merge_lazy",), ("sources",), ("split_region",),
        ]


def test_value_first_protects_finite_answer_and_reopens_existing_event_format(tmp_path):
    from experiments.memory.test_relative_hierarchy import _finite_cancellation_sources

    path = tmp_path / "value-guard.sqlite"
    sources = _finite_cancellation_sources()
    hierarchy = RelativeObservationHierarchy((0, 1, 1, 2), ridge=0.002, elimination_order=range(4))
    expected = solve_relative_direct(4, sources, ridge=0.002)
    journal = JournaledRelativeHierarchyWorkspace(path, hierarchy, sources, create=True)
    base = journal.snapshot()
    with pytest.raises(FloatingPointError):
        journal.merge_regions(base, 0, 1, retain_children=True, lazy_readout=True, protected_cells=(2,))
    assert journal.snapshot() is base

    options = {"protected_cells": range(4), "tolerance": 1e296, "value_first": True}
    merged = journal.merge_regions(base, 0, 1, retain_children=True, lazy_readout=True, **options)
    follower = JournaledRelativeHierarchyWorkspace(path, hierarchy, sources)
    recovered = follower.snapshot()
    assert recovered.digest == merged.digest
    assert recovered.snapshot.read_many(range(4), value_first=True) == pytest.approx(expected, rel=1e-12, abs=0)
    split = follower.split_region(recovered, 0, **options)
    assert split.number == 2 and journal.snapshot().digest == split.digest
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT kind FROM relative_events ORDER BY number").fetchall() == [
            ("merge_lazy",), ("split_region",),
        ]


def test_retry_guard_recovers_affine_refusal_without_changing_replay(tmp_path):
    from experiments.memory.test_relative_hierarchy import _finite_cancellation_sources

    path = tmp_path / "retry-guard.sqlite"
    sources = _finite_cancellation_sources()
    hierarchy = RelativeObservationHierarchy((0, 1, 1, 2), ridge=0.002, elimination_order=range(4))
    journal = JournaledRelativeHierarchyWorkspace(path, hierarchy, sources, create=True)
    parent = journal.snapshot()
    options = {"retain_children": True, "lazy_readout": True, "protected_cells": (2,), "tolerance": 1e296}
    with pytest.raises(FloatingPointError):
        journal.merge_regions(parent, 0, 1, **options)
    assert journal.snapshot() is parent
    fused = journal.merge_regions(parent, 0, 1, retry_failed=True, **options)
    recovered = JournaledRelativeHierarchyWorkspace(path, hierarchy, sources).snapshot()
    assert recovered.digest == fused.digest
    assert recovered.snapshot.read(2, retry_failed=True) == pytest.approx(parent.snapshot.read(2), rel=1e-12)
    restored = journal.split_region(fused, 0, protected_cells=(2,), tolerance=1e296, retry_failed=True)
    assert restored.number == 2
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT kind, payload FROM relative_events ORDER BY number").fetchall() == [
            ("merge_lazy", b"[0,1]"), ("split_region", b"[0]"),
        ]
