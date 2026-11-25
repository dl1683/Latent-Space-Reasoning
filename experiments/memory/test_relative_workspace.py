from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import pytest

from experiments.memory import relative_workspace
from experiments.memory.relative_hierarchy import RelativeObservationHierarchy
from experiments.memory.relative_observations import RelativeObservation, solve_relative_direct
from experiments.memory.relative_ordering import _profile, propose_orders
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


def test_generated_order_requires_replay_to_cover_its_conversion():
    _, base, sources = _chain()
    workspace = RelativeHierarchyWorkspace(base)
    first = (_LATE_CHANGE, (16,))
    second = ({"row-15": RelativeObservation(15, 2, source=14, weight=2)}, (16, 17))
    short = workspace.maintain_generated(
        base, (first,), max_candidates=2, max_symbolic_ports=10, protected=(0, 17),
    )
    assert short.considered == 1 and short.snapshot is workspace.snapshot() is base
    assert short.baseline_work == RelativeReplayWork(10, 10, 37)

    selected = workspace.maintain_generated(
        base, iter((first, second)), max_candidates=2, max_symbolic_ports=10, protected=(0, 17),
    )
    assert selected.snapshot is workspace.snapshot() and selected.snapshot is not base
    assert selected.baseline_work == RelativeReplayWork(20, 20, 37)
    assert selected.selected_work == RelativeReplayWork(4, 2, 37)
    assert selected.conversion_buckets == 10
    assert selected.snapshot._sources is base._sources
    revised = workspace.revise(selected.snapshot, _LATE_CHANGE)
    expected = solve_relative_direct(18, {**sources, **_LATE_CHANGE}, ridge=_RIDGE)
    assert revised.read_many((0, 16, 17)) == pytest.approx(
        tuple(expected[cell] for cell in (0, 16, 17)), abs=1e-12, rel=0,
    )


def test_symbolic_graph_includes_promotions_and_future_local_merges():
    field = RelativeObservationHierarchy((0, 0, 0, 0, 1, 2, 3), ridge=_RIDGE,
                                         elimination_order=reversed(range(7)))
    source = {
        "anchor": RelativeObservation(0, 1),
        "first": RelativeObservation(1, 0.5, source=0),
        "bridge": RelativeObservation(4, 0.5, source=1),
        "second": RelativeObservation(3, -0.5, source=2),
        "far": RelativeObservation(5, 0.5, source=3),
    }
    base = field.initialize(source)
    edits = {
        "bridge": None, "merge": RelativeObservation(2, 0, source=1),
        "new_bridge": RelativeObservation(5, 0.5, source=0),
    }
    workload = ((edits, (5,)),)
    graph, activity = _profile(base, workload, 5)
    assert graph == {0: {1, 3, 5}, 1: {0, 3, 4}, 3: {0, 1, 5}, 4: {1}, 5: {0, 3}}
    assert activity[5] == 2 and activity[4] == 1
    proposals = tuple(propose_orders(base, workload, 5))
    assert proposals and proposals == tuple(propose_orders(base, workload, 5))
    for order in proposals:
        assert sorted(order) == list(range(7))
        assert order.index(2) == base.elimination_order.index(2)
        assert order.index(6) == base.elimination_order.index(6)

    workspace = RelativeHierarchyWorkspace(base)
    stopped = workspace.maintain_generated(base, workload, max_candidates=2, max_symbolic_ports=4)
    assert stopped.considered == 0 and stopped.snapshot is workspace.snapshot() is base


def test_generated_maintenance_fences_a_source_revision_during_symbolic_search(monkeypatch):
    _, base, _ = _chain()
    workspace = RelativeHierarchyWorkspace(base)
    generator = relative_workspace.propose_orders

    def revise_first(snapshot, workload, max_ports):
        workspace.revise(base, {"anchor": RelativeObservation(0, 1.7, weight=2)})
        yield from generator(snapshot, workload, max_ports)

    monkeypatch.setattr(relative_workspace, "propose_orders", revise_first)
    with pytest.raises(RuntimeError, match="revision is stale"):
        workspace.maintain_generated(base, ((_LATE_CHANGE, (16,)),),
                                     max_candidates=2, max_symbolic_ports=10)
    assert workspace.snapshot().elimination_order == base.elimination_order
    assert workspace.snapshot().read(0) != base.read(0)
