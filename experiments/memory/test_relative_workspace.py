from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import pytest

from experiments.memory.relative_hierarchy import RelativeObservationHierarchy
from experiments.memory.relative_observations import RelativeObservation, solve_relative_direct
from experiments.memory.relative_workspace import RelativeHierarchyWorkspace, RelativeReplayWork


_ORDER = tuple(range(18))
_RIDGE = 0.35
_LATE_CHANGE = {"row-15": RelativeObservation(15, -1.8, source=14, weight=3)}


def _chain():
    field = RelativeObservationHierarchy(
        tuple(cell // 3 for cell in _ORDER), ridge=_RIDGE, elimination_order=reversed(_ORDER),
    )
    sources = {"anchor": RelativeObservation(0, 0.7, weight=2)}
    sources.update({f"row-{cell}": RelativeObservation(cell, 0.13, source=cell - 1)
                    for cell in _ORDER[1:]})
    return field, field.initialize(sources), sources


def test_maintenance_installs_order_only_and_future_source_edits_use_it():
    field, base, sources = _chain()
    workspace = RelativeHierarchyWorkspace(base)
    workload = (({}, (16,)), (_LATE_CHANGE, (16, 17, 16)))
    result = workspace.maintain(base, (_ORDER,), workload, max_candidates=1, protected=(0, 17))
    assert result.considered == 1 and result.conversion_buckets == 10
    assert result.baseline_work == RelativeReplayWork(10, 20, 37)
    assert result.selected_work == RelativeReplayWork(2, 2, 37)

    layout = result.snapshot
    assert workspace.snapshot() is layout and layout.elimination_order == _ORDER
    assert layout._sources is base._sources and layout._regions == base._regions
    assert layout.read(16) == pytest.approx(base.read(16), abs=1e-12, rel=0)
    assert base.elimination_order == tuple(reversed(_ORDER))
    revised = workspace.revise(layout, _LATE_CHANGE)
    assert revised.rebuilt_buckets == (14, 15) and revised.elimination_order == _ORDER
    expected = solve_relative_direct(18, {**sources, **_LATE_CHANGE}, ridge=_RIDGE)
    assert revised.read_many((0, 16, 17)) == pytest.approx(
        tuple(expected[cell] for cell in (0, 16, 17)), abs=1e-12, rel=0,
    )
    assert abs(revised.read(16) - layout.read(16)) > 0.1

    detached = field.revise(layout, {"row-15": None})
    compacted = field.compact(detached, max_regions=2)
    assert compacted.elimination_order == _ORDER and compacted._elimination is None
    restored = field.revise(compacted, {"row-15": sources["row-15"]})
    assert restored.elimination_order == _ORDER
    assert restored.read(16) == pytest.approx(base.read(16), abs=1e-12, rel=0)
    assert field.initialize(sources).elimination_order == base.elimination_order


def test_maintenance_leaves_the_tip_when_workload_prefers_the_existing_order():
    _, base, _ = _chain()
    workspace = RelativeHierarchyWorkspace(base)
    early = {"row-3": RelativeObservation(3, -0.8, source=2, weight=3)}
    result = workspace.maintain(base, (_ORDER,), ((early, (0, 1)),),
                                max_candidates=1, protected=(0, 17))
    assert result.snapshot is workspace.snapshot() is base
    assert result.selected_work == result.baseline_work and result.conversion_buckets == 0


def test_maintenance_rejects_guard_regression_and_limits_candidate_consumption(monkeypatch):
    field, base, _ = _chain()
    workspace = RelativeHierarchyWorkspace(base)
    reorder = field.reorder

    def drift(snapshot, order):
        return replace(reorder(snapshot, order), _supported_ports=frozenset())

    def orders():
        yield _ORDER
        pytest.fail("candidate round consumed past its cap")

    monkeypatch.setattr(field, "reorder", drift)
    result = workspace.maintain(base, orders(), ((_LATE_CHANGE, (16,)),),
                                protected=(16,), max_candidates=1)
    assert result.snapshot is workspace.snapshot() is base
    assert result.considered == 1


def test_maintenance_rejects_a_result_after_a_concurrent_source_revision(monkeypatch):
    field, base, _ = _chain()
    workspace = RelativeHierarchyWorkspace(base)
    started, released = Event(), Event()
    reorder = field.reorder

    def wait_for_source(snapshot, order):
        started.set()
        assert released.wait(timeout=5)
        return reorder(snapshot, order)

    monkeypatch.setattr(field, "reorder", wait_for_source)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(
            workspace.maintain, base, (_ORDER,), ((_LATE_CHANGE, (16,)),), max_candidates=1,
        )
        try:
            assert started.wait(timeout=5)
            real_edit = {"anchor": RelativeObservation(0, 1.7, weight=2)}
            current = workspace.revise(base, real_edit)
        finally:
            released.set()
        with pytest.raises(RuntimeError, match="revision is stale"):
            pending.result(timeout=5)
    assert workspace.snapshot() is current
    assert current.elimination_order == base.elimination_order
    assert current.read(0) != base.read(0)
