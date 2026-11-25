from dataclasses import replace

import numpy as np
import pytest

from experiments.memory import relative_elimination, relative_hierarchy
from experiments.memory.latent_field import _frozen
from experiments.memory.relative_hierarchy import RelativeObservationHierarchy
from experiments.memory.relative_observations import RelativeObservation, solve_relative_direct


_MEMBERSHIP = (0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 1)
_RIDGE = 0.35


def _sources():
    return {
        "anchor": RelativeObservation(0, 0.7, weight=2),
        "start": RelativeObservation(1, 1.2, source=0),
        "first": RelativeObservation(2, 0.7, source=1),
        "conflict": RelativeObservation(2, -0.7, source=0, weight=2),
        "bridge": RelativeObservation(3, -0.5, source=2),
        "center": RelativeObservation(4, 0.9, source=3),
        "continue": RelativeObservation(5, -0.4, source=4),
        "downstream": RelativeObservation(6, 0.2, source=5),
        "late": RelativeObservation(7, 0.8, source=6),
        "end": RelativeObservation(8, -0.1, source=7),
        "other_anchor": RelativeObservation(9, -0.3),
        "other": RelativeObservation(10, 0.6, source=9),
    }


def _assert_direct(snapshot, sources, cells=len(_MEMBERSHIP)):
    expected = solve_relative_direct(cells, sources, ridge=_RIDGE)
    for cell, value in enumerate(expected):
        if value is None:
            assert snapshot.read(cell) is None
        else:
            assert snapshot.read(cell) == pytest.approx(value, abs=1e-12, rel=0)


def test_bridge_correction_reuses_region_operators_for_late_interior_read():
    hierarchy = RelativeObservationHierarchy(_MEMBERSHIP, ridge=_RIDGE)
    sources = _sources()
    original = hierarchy.initialize(sources)
    bridge = RelativeObservation(3, 1.5, source=2, weight=3)
    revised = hierarchy.revise(original, {"bridge": bridge})

    assert revised.rebuilt_regions == ()
    assert revised.coarse_size == original.coarse_size == 4
    assert all(new is old for new, old in zip(revised._regions, original._regions))
    assert abs(revised.read(8) - original.read(8)) > 0.1
    assert revised.read(10) == original.read(10)
    assert revised.read(11) is None
    _assert_direct(revised, {**sources, "bridge": bridge})
    _assert_direct(original, sources)


def test_detached_region_edit_rebuilds_one_operator_and_survives_reconnection():
    hierarchy = RelativeObservationHierarchy(_MEMBERSHIP, ridge=_RIDGE)
    sources = _sources()
    original = hierarchy.initialize(sources)
    detached = hierarchy.revise(original, {"bridge": None})
    assert detached.coarse_size == 4 and detached.rebuilt_regions == ()
    assert detached.read(2) is not None and detached.read(8) is None
    without_bridge = {name: row for name, row in sources.items() if name != "bridge"}
    _assert_direct(detached, without_bridge)

    late = RelativeObservation(7, -0.8, source=6)
    corrected = hierarchy.revise(detached, {"late": late})
    assert corrected.rebuilt_regions == (2,)
    assert corrected.read(8) is None
    rejoined = hierarchy.revise(corrected, {"bridge": sources["bridge"]})
    assert rejoined.rebuilt_regions == ()
    assert abs(rejoined.read(8) - original.read(8)) > 0.1
    assert rejoined.read(10) == original.read(10)
    _assert_direct(rejoined, {**sources, "late": late})
    _assert_direct(original, sources)

    split = hierarchy.revise(original, {"continue": None})
    assert split.rebuilt_regions == (1,)
    assert split.read(4) is not None and split.read(5) is None and split.read(8) is None
    _assert_direct(split, {name: row for name, row in sources.items() if name != "continue"})


def test_moved_bridge_promotes_new_ports_and_does_not_merge_disconnected_cells():
    hierarchy = RelativeObservationHierarchy(_MEMBERSHIP, ridge=_RIDGE)
    sources = _sources()
    original = hierarchy.initialize(sources)
    moved = RelativeObservation(11, 1.5, source=1)
    revised = hierarchy.revise(original, {"bridge": moved})
    assert revised.rebuilt_regions == (0, 1)
    assert revised.coarse_size == 6
    assert revised.read(11) is not None and revised.read(4) is None and revised.read(8) is None
    _assert_direct(revised, {**sources, "bridge": moved})

    branch = hierarchy.revise(original, {"center": RelativeObservation(4, 1.2, source=3)})
    assert branch.rebuilt_regions == (1,) and branch.coarse_size == 4
    assert branch.read(11) is None and branch.read(8) is not None
    _assert_direct(branch, {**sources, "center": RelativeObservation(4, 1.2, source=3)})


def test_no_ports_or_all_ports_and_failed_update_preserves_the_base():
    hierarchy = RelativeObservationHierarchy((0, 1), ridge=1)
    sources = {"anchor": RelativeObservation(0, 2)}
    original = hierarchy.initialize(sources)
    assert original.coarse_size == 0
    assert original.read(0) == pytest.approx(1) and original.read(1) is None
    revised = hierarchy.revise(original, {"bridge": RelativeObservation(1, 0, source=0)})
    assert revised.coarse_size == 2 and revised.rebuilt_regions == (0, 1)
    assert revised.read(1) == pytest.approx(0.4)
    assert hierarchy.revise(revised, sources) is revised
    with pytest.raises(FloatingPointError, match="nonfinite weighted observation"):
        hierarchy.revise(revised, {"bridge": RelativeObservation(1, 1e308, source=0, weight=16)})
    assert revised.read(0) == pytest.approx(0.8)
    assert original.read(1) is None
    retired = hierarchy.revise(revised, {"bridge": None})
    assert retired.coarse_size == 2 and retired.rebuilt_regions == ()
    assert retired.read(0) == pytest.approx(1) and retired.read(1) is None


def test_compaction_skips_source_rebuild_and_coarse_solve_and_keeps_forks(monkeypatch):
    hierarchy = RelativeObservationHierarchy(_MEMBERSHIP, ridge=_RIDGE)
    sources = _sources()
    original = hierarchy.initialize({**sources, "parallel": sources["bridge"]})
    single = hierarchy.revise(original, {"parallel": None})
    assert hierarchy.compact(single) is single
    detached = hierarchy.revise(single, {"bridge": None})
    without_bridge = {name: row for name, row in sources.items() if name != "bridge"}

    def unexpected(*args, **kwargs):
        pytest.fail("pure compaction rebuilt from sources or reran the coarse solve")

    with monkeypatch.context() as patch:
        patch.setattr(relative_hierarchy, "_build_region", unexpected)
        patch.setattr(relative_hierarchy, "_solve_coarse", unexpected)
        first = hierarchy.compact(detached, protected_cells=(0, 2, 3, 8, 11))
        second = hierarchy.compact(first, protected_cells=(0, 2, 3, 8, 11))

    assert first.compacted_regions == (0,) and first.coarse_size == 3
    assert second.compacted_regions == (1,) and second.coarse_size == 2
    assert first.rebuilt_regions == second.rebuilt_regions == ()
    assert second._sources is detached._sources and second._bridges is detached._bridges
    assert second._regions[2] is detached._regions[2]
    assert second._regions[1]._sources is detached._regions[1]._sources
    assert hierarchy.compact(second) is second
    assert detached.coarse_size == 4 and hierarchy.compact(single) is single
    for cell, position in second._port_positions.items():
        assert second._values[position] == detached._values[detached._port_positions[cell]]
    _assert_direct(first, without_bridge)
    _assert_direct(second, without_bridge)

    restored = hierarchy.revise(second, {"bridge": sources["bridge"]})
    assert restored.rebuilt_regions == (0, 1) and restored.coarse_size == 4
    _assert_direct(restored, sources)
    _assert_direct(detached, without_bridge)


def test_compaction_composes_retired_port_reads_across_two_rounds():
    hierarchy = RelativeObservationHierarchy(_MEMBERSHIP, ridge=_RIDGE)
    sources = {**_sources(), "remote_anchor": RelativeObservation(8, 1.1, weight=2)}
    original = hierarchy.initialize(sources)
    detached = hierarchy.revise(original, {"bridge": None})
    first = hierarchy.compact(detached, max_regions=2, protected_cells=(2, 3, 4, 5))
    assert first.compacted_regions == (0, 1) and first._regions[1].ports == (5,)
    assert first.read(4) == pytest.approx(detached.read(4), abs=1e-12, rel=0)
    without_bridge = {name: row for name, row in sources.items() if name != "bridge"}
    _assert_direct(first, without_bridge)
    before, after = detached._regions[1], first._regions[1]
    assert before.factor.size + before.readout.size - after.factor.size - after.readout.size == 4

    bridge = RelativeObservation(6, -1.3, source=5, weight=2)
    changed = hierarchy.revise(first, {"downstream": bridge})
    assert changed.rebuilt_regions == () and changed._regions[1] is after
    assert abs(changed.read(4) - first.read(4)) > 0.02
    _assert_direct(changed, {**without_bridge, "downstream": bridge})

    isolated = hierarchy.revise(changed, {"downstream": None})
    second = hierarchy.compact(isolated, max_regions=2, protected_cells=range(len(_MEMBERSHIP)))
    assert second.compacted_regions == (1, 2) and second.coarse_size == 0
    assert second.read(3) is None and second.read(6) is not None
    _assert_direct(second, {name: row for name, row in sources.items()
                            if name not in {"bridge", "downstream"}})
    _assert_direct(original, sources)


def test_compaction_refuses_changed_protected_reads_without_affecting_parent(monkeypatch):
    hierarchy = RelativeObservationHierarchy(_MEMBERSHIP, ridge=_RIDGE)
    original = hierarchy.initialize(_sources())
    detached = hierarchy.revise(original, {"bridge": None})
    expected = detached.read(0)
    compact_region = relative_hierarchy._compact_region

    def drift(region, active_ports):
        proposed = compact_region(region, active_ports)
        readout = proposed.readout.copy()
        readout[0, -1] += 0.25
        return replace(proposed, readout=readout)

    monkeypatch.setattr(relative_hierarchy, "_compact_region", drift)
    with pytest.raises(RuntimeError, match="compaction changed protected read 0"):
        hierarchy.compact(detached, protected_cells=(0,))
    assert detached.read(0) == expected and detached.coarse_size == 4
    with pytest.raises(ValueError, match="positive region cap"):
        hierarchy.compact(detached, max_regions=0)


def test_coarse_rhs_edits_reuse_operators_and_unrelated_solutions(monkeypatch):
    membership = (*_MEMBERSHIP, 4)
    hierarchy = RelativeObservationHierarchy(membership, ridge=_RIDGE)
    sources = {**_sources(), "unrelated_bridge": RelativeObservation(12, 0.4, source=10)}
    base = hierarchy.initialize(sources)
    assert base.coarse_factorizations == base.coarse_rhs_solves == 2
    operator = base._coarse_blocks[(2, 3, 5, 6)].operator
    unrelated = base._coarse_blocks[(10, 12)]
    bridge = RelativeObservation(3, 1.5, source=2)
    center = RelativeObservation(4, -0.7, source=3)

    def unexpected(*args, **kwargs):
        pytest.fail("a value-only correction refactored the coarse system")

    with monkeypatch.context() as patch:
        patch.setattr(relative_hierarchy, "_factor_coarse", unexpected)
        corrected = hierarchy.revise(base, {"bridge": bridge})
        revised = hierarchy.revise(corrected, {"center": center})
    for snapshot in (corrected, revised):
        assert snapshot.coarse_factorizations == 0 and snapshot.coarse_rhs_solves == 1
        assert snapshot._coarse_blocks[(2, 3, 5, 6)].operator is operator
        assert snapshot._coarse_blocks[(10, 12)] is unrelated
        assert snapshot.read(12) == base.read(12)
    assert abs(base.read(8) - corrected.read(8)) > 0.05
    _assert_direct(corrected, {**sources, "bridge": bridge}, len(membership))
    _assert_direct(revised, {**sources, "bridge": bridge, "center": center}, len(membership))

    reweighted_bridge = RelativeObservation(3, 1.5, source=2, weight=3)
    reweighted = hierarchy.revise(revised, {"bridge": reweighted_bridge})
    assert reweighted.coarse_factorizations == reweighted.coarse_rhs_solves == 1
    assert reweighted._coarse_blocks[(10, 12)] is unrelated
    assert reweighted._coarse_blocks[(2, 3, 5, 6)].operator is not operator
    _assert_direct(reweighted, {**sources, "bridge": reweighted_bridge, "center": center}, len(membership))
    _assert_direct(base, sources, len(membership))


def test_coarse_split_projection_and_lazy_factorization_follow_the_selected_branch():
    membership = (*_MEMBERSHIP, 4)
    hierarchy = RelativeObservationHierarchy(membership, ridge=_RIDGE)
    sources = {**_sources(), "unrelated_bridge": RelativeObservation(12, 0.4, source=10)}
    base = hierarchy.initialize(sources)
    detached = hierarchy.revise(base, {"bridge": None})
    assert detached.coarse_factorizations == detached.coarse_rhs_solves == 2
    assert detached.read(8) is None

    compacted = hierarchy.compact(detached, max_regions=2)
    assert compacted.coarse_factorizations == compacted.coarse_rhs_solves == 0
    projected = compacted._coarse_blocks[(5, 6)]
    assert projected.operator is None
    assert compacted._coarse_blocks[(10, 12)] is base._coarse_blocks[(10, 12)]

    remote = RelativeObservation(12, -0.6, source=10)
    independent = hierarchy.revise(compacted, {"unrelated_bridge": remote})
    assert independent.coarse_factorizations == 0 and independent.coarse_rhs_solves == 1
    assert independent._coarse_blocks[(5, 6)] is projected

    downstream = RelativeObservation(6, 3, source=5)
    refreshed = hierarchy.revise(independent, {"downstream": downstream})
    assert refreshed.coarse_factorizations == refreshed.coarse_rhs_solves == 1
    assert refreshed._coarse_blocks[(5, 6)].operator is not None
    assert compacted._coarse_blocks[(5, 6)].operator is None

    restored = hierarchy.revise(refreshed, {"bridge": sources["bridge"]})
    assert restored.coarse_factorizations == restored.coarse_rhs_solves == 1
    _assert_direct(restored, {**sources, "unrelated_bridge": remote, "downstream": downstream},
                   len(membership))
    _assert_direct(base, sources, len(membership))


def test_coarse_scaled_transform_handles_large_finite_weighted_observations():
    hierarchy = RelativeObservationHierarchy((0, 1), ridge=1e306)
    sources = {"anchor": RelativeObservation(0, 0, weight=1e308)}
    sources.update({str(i): RelativeObservation(1, 1e154, source=0, weight=1e308) for i in range(8)})
    snapshot = hierarchy.initialize(sources)
    # Analytic 2-by-2 normal equations after dividing by 1e308 and scaling values by 1e154.
    target = (8 - 64 / 9.01) / (8.01 - 64 / 9.01)
    source = (8 * target - 8) / 9.01
    assert snapshot.read(0) / 1e154 == pytest.approx(source, abs=1e-13)
    assert snapshot.read(1) / 1e154 == pytest.approx(target, abs=1e-13)


def test_coarse_rank_threshold_is_local_to_each_independent_block():
    hierarchy = RelativeObservationHierarchy((0, 1, 2, 3), ridge=1)
    sources = {
        "large_anchor": RelativeObservation(0, 1, weight=1e80),
        "large_bridge": RelativeObservation(1, 2, source=0, weight=1e80),
        "small_anchor": RelativeObservation(2, 2),
        "small_bridge": RelativeObservation(3, 1, source=2),
    }
    snapshot = hierarchy.initialize(sources)
    assert snapshot.coarse_factorizations == 2
    assert tuple(snapshot.read(cell) for cell in range(4)) == pytest.approx(
        (1, 3, 0.6, 0.8), abs=1e-12, rel=0,
    )
    with pytest.raises(ValueError, match="full numerical rank"):
        solve_relative_direct(4, sources, ridge=1)


def test_connected_bridge_reweighting_reuses_the_unchanged_elimination_prefix():
    membership = tuple(cell // 3 for cell in range(18))
    hierarchy = RelativeObservationHierarchy(membership, ridge=_RIDGE, elimination_order=range(18))
    sources = {"anchor": RelativeObservation(0, 0.7, weight=2)}
    sources.update({f"row-{cell}": RelativeObservation(cell, 0.13, source=cell - 1)
                    for cell in range(1, 18)})
    base = hierarchy.initialize(sources)
    assert base.coarse_size == len(base.rebuilt_buckets) == 10
    change = RelativeObservation(15, -1.8, source=14, weight=3)
    revised = hierarchy.revise(base, {"row-15": change})
    assert revised.coarse_size == 10 and revised.rebuilt_buckets == (14, 15)
    assert revised.visited_buckets == (14, 15)
    assert revised.rebuilt_regions == ()
    assert revised._elimination.steps[12] is base._elimination.steps[12]
    assert abs(revised.read(12) - base.read(12)) > 0.02
    assert revised.coarse_factorizations == revised.coarse_rhs_solves == 0
    _assert_direct(revised, {**sources, "row-15": change}, len(membership))
    _assert_direct(base, sources, len(membership))

    reverse = RelativeObservationHierarchy(membership, ridge=_RIDGE,
                                           elimination_order=reversed(range(18)))
    reverse_base = reverse.initialize(sources)
    reverse_revised = reverse.revise(reverse_base, {"row-15": change})
    assert len(reverse_revised.rebuilt_buckets) == reverse_base.coarse_size == 10
    _assert_direct(reverse_revised, {**sources, "row-15": change}, len(membership))


def test_elimination_survives_changed_support_compaction_and_bridge_promotion():
    hierarchy = RelativeObservationHierarchy(_MEMBERSHIP, ridge=_RIDGE,
                                             elimination_order=reversed(range(len(_MEMBERSHIP))))
    sources = _sources()
    base = hierarchy.initialize(sources)
    detached = hierarchy.revise(base, {"bridge": None})
    remaining = {name: row for name, row in sources.items() if name != "bridge"}
    assert detached.read(8) is None
    _assert_direct(detached, remaining)
    compacted = hierarchy.compact(detached, max_regions=2, protected_cells=range(len(_MEMBERSHIP)))
    assert compacted.coarse_size == 2 and compacted.rebuilt_buckets == ()
    assert compacted._elimination is None
    _assert_direct(compacted, remaining)

    bridge = RelativeObservation(4, -0.3, source=1, weight=2)
    replaced = hierarchy.revise(compacted, {"bridge": bridge})
    assert replaced.rebuilt_regions == (0, 1)
    assert replaced.rebuilt_buckets == (6, 5, 4, 1)
    assert replaced.read(8) is not None
    _assert_direct(replaced, {**remaining, "bridge": bridge})

    historical = hierarchy.revise(base, {"anchor": RelativeObservation(0, 1.7, weight=2)})
    _assert_direct(historical, {**sources, "anchor": RelativeObservation(0, 1.7, weight=2)})
    _assert_direct(base, sources)


def test_equal_updated_remainder_keeps_downstream_elimination_step():
    first = relative_elimination.EliminationFactor((0, 1), _frozen(np.array([[2, 0, 4], [0, 3, 9]])))
    changed = relative_elimination.EliminationFactor((0, 1), _frozen(np.array([[2, 0, 8], [0, 3, 9]])))
    rank = {0: 0, 1: 1}
    steps, _ = relative_elimination.compile_elimination((0, 1), rank, [first], {})
    revised, rebuilt = relative_elimination.compile_elimination((0, 1), rank, [changed], steps)
    assert rebuilt == (0,)
    assert revised[0].remainder is steps[0].remainder and revised[1] is steps[1]
    assert relative_elimination.resolve_elimination(steps, (0, 1)) == {0: 2, 1: 3}
    assert relative_elimination.resolve_elimination(revised, (0,)) == {0: 4}


def test_elimination_failure_keeps_parent_snapshot_and_requires_explicit_permutation(monkeypatch):
    with pytest.raises(ValueError, match="must contain every cell once"):
        RelativeObservationHierarchy((0, 1), ridge=_RIDGE, elimination_order=(0, 0))
    hierarchy = RelativeObservationHierarchy(_MEMBERSHIP, ridge=_RIDGE,
                                             elimination_order=range(len(_MEMBERSHIP)))
    sources = _sources()
    base = hierarchy.initialize(sources)
    before = base.read(8)

    def refuse(*args):
        raise ValueError("unresolved test pivot")

    monkeypatch.setattr(relative_elimination, "_reduce_bucket", refuse)
    with pytest.raises(ValueError, match="unresolved test pivot"):
        hierarchy.revise(base, {"bridge": RelativeObservation(3, 1.5, source=2)})
    assert base.read(8) == before
    _assert_direct(base, sources)


def test_late_batched_reads_recover_only_the_needed_connected_ports(monkeypatch):
    membership = tuple(cell // 3 for cell in range(18))
    hierarchy = RelativeObservationHierarchy(membership, ridge=_RIDGE, elimination_order=range(18))
    sources = {"anchor": RelativeObservation(0, 0.7, weight=2)}
    sources.update({f"row-{cell}": RelativeObservation(cell, 0.13, source=cell - 1)
                    for cell in range(1, 18)})
    change = RelativeObservation(15, -1.8, source=14, weight=3)

    def unexpected(*args, **kwargs):
        pytest.fail("revision eagerly reconstructed boundary values")

    with monkeypatch.context() as patch:
        patch.setattr(relative_hierarchy, "resolve_elimination", unexpected)
        base = hierarchy.initialize(sources)
        revised = hierarchy.revise(base, {"row-15": change})
    assert base._values is revised._values is None
    assert revised.coarse_size == 10

    resolve = relative_hierarchy.resolve_elimination
    calls = []

    def track(steps, cells):
        recovered = resolve(steps, cells)
        calls.append((tuple(cells), tuple(recovered)))
        return recovered

    monkeypatch.setattr(relative_hierarchy, "resolve_elimination", track)
    request = (16, 15, 17, 16)
    expected = solve_relative_direct(18, {**sources, "row-15": change}, ridge=_RIDGE)
    assert revised.read_many(iter(request)) == pytest.approx(tuple(expected[cell] for cell in request),
                                                             abs=1e-12, rel=0)
    assert calls == [((15,), (15,))]
    assert revised.read_many(()) == ()
    assert calls == [((15,), (15,))]
    assert base.read(16) != revised.read(16)
    assert all(roots == recovered == (15,) for roots, recovered in calls)


def test_sparse_reads_skip_unanchored_components_and_zero_regional_weights(monkeypatch):
    hierarchy = RelativeObservationHierarchy((0, 0, 0, 0, 1, 2), ridge=_RIDGE,
                                             elimination_order=range(6))
    sources = {
        "anchor": RelativeObservation(0, 1),
        "first": RelativeObservation(1, 0.5, source=0),
        "supported_bridge": RelativeObservation(4, 0.5, source=1),
        "unsupported_local": RelativeObservation(3, -0.5, source=2),
        "unsupported_bridge": RelativeObservation(5, 0.5, source=3),
    }
    snapshot = hierarchy.initialize(sources)
    assert snapshot.coarse_size == 4
    resolve = relative_hierarchy.resolve_elimination
    calls = []

    def track(steps, cells):
        recovered = resolve(steps, cells)
        calls.append((tuple(cells), set(recovered)))
        return recovered

    monkeypatch.setattr(relative_hierarchy, "resolve_elimination", track)
    expected = solve_relative_direct(6, sources, ridge=_RIDGE)
    reads = snapshot.read_many((5, 0, 0, 4))
    assert reads[0] is None
    assert reads[1:] == pytest.approx((expected[0], expected[0], expected[4]), abs=1e-12, rel=0)
    assert calls == [((1, 4), {1, 4})]
    assert snapshot.read_many((5, 3)) == (None, None)
    assert len(calls) == 1


def test_unrepresentable_sparse_port_can_fail_at_read_without_mutating_snapshot():
    hierarchy = RelativeObservationHierarchy((0, 1), ridge=1e-6, elimination_order=(1, 0))
    sources = {
        "anchor": RelativeObservation(0, 1e308, weight=0.25),
        "bridge": RelativeObservation(1, 1e308, source=0, weight=0.25),
    }
    snapshot = hierarchy.initialize(sources)
    assert snapshot.coarse_size == 2 and snapshot._values is None
    before = snapshot.read(0)
    assert 0.99 < before / 1e308 < 1
    with pytest.raises(FloatingPointError):
        snapshot.read_many((0, 1))
    assert snapshot.read(0) == before
    assert snapshot.support_witness(1).cells == (1, 0)


def test_demand_recovery_handles_a_deep_dependency_chain():
    size = 2048
    readout = _frozen(np.array([1.0, 1.0]))
    constant = _frozen(np.array([1.0]))
    steps = {cell: relative_elimination.EliminationStep((), (cell + 1,), readout, None, 1.0)
             for cell in range(size - 1)}
    steps[size - 1] = relative_elimination.EliminationStep((), (), constant, None, 1.0)
    values = relative_elimination.resolve_elimination(steps, (0, size - 1, 0))
    assert len(values) == size
    assert values[0] == size and values[size - 1] == 1


def test_indexed_update_routes_to_both_old_and_new_remainder_recipients():
    def factor(cells, rows):
        return relative_elimination.EliminationFactor(cells, _frozen(np.asarray(rows, dtype=float)))

    rank = {cell: cell for cell in range(4)}
    anchors = [factor((cell,), [[2, cell]]) for cell in rank]
    first = factor((0, 1), [[1, -1, 2]])
    second = factor((0, 2), [[1, -1, 1]])
    key = (1, "link", 0)
    seeds = [((0, cell, 0), None, row) for cell, row in enumerate(anchors)]
    parent, routes, _, _ = relative_elimination.update_elimination(
        rank, {}, {}, [*seeds, (key, None, first)], rank,
    )
    result, new_routes, rebuilt, visited = relative_elimination.update_elimination(
        rank, parent, routes, [(key, first, second)], (),
    )
    assert rebuilt == visited == (0, 1, 2)
    assert result[3] is parent[3] and new_routes[3] is routes[3]
    assert parent[0].remainder.cells == (1,) and result[0].remainder.cells == (2,)
    assert (2, 0, 0) in routes[1] and (2, 0, 0) not in new_routes[1]
    reference, _ = relative_elimination.compile_elimination(rank, rank, [*anchors, second], {})
    assert relative_elimination.resolve_elimination(result, rank) == (
        relative_elimination.resolve_elimination(reference, rank)
    )


def test_indexed_revisions_match_full_sweep_on_promotions_splits_and_forks():
    hierarchy = RelativeObservationHierarchy(_MEMBERSHIP, ridge=_RIDGE,
                                             elimination_order=reversed(range(len(_MEMBERSHIP))))
    observations = _sources()
    original = hierarchy.initialize(observations)

    def check(base, sources, edits):
        revised = hierarchy.revise(base, edits)
        numeric = revised._elimination
        seeds = [factor for number in sorted(numeric.regions) for factor in numeric.regions[number][1]]
        seeds.extend(numeric.bridges[name][1] for name in sorted(numeric.bridges))
        full, rebuilt = relative_elimination.compile_elimination(
            revised._port_positions, revised._elimination_rank, seeds, base._elimination.steps,
        )
        assert revised.rebuilt_buckets == rebuilt
        for cell, step in full.items():
            indexed = numeric.steps[cell]
            assert indexed.dependencies == step.dependencies and np.array_equal(indexed.readout, step.readout)
            if step.remainder is None:
                assert indexed.remainder is None
            else:
                assert indexed.remainder.cells == step.remainder.cells
                assert np.array_equal(indexed.remainder.rows, step.remainder.rows)
        updated = {**sources, **{name: row for name, row in edits.items() if row is not None}}
        for name, row in edits.items():
            if row is None:
                del updated[name]
        _assert_direct(revised, updated)
        return revised, updated

    moved, rows = check(original, observations, {
        "bridge": RelativeObservation(4, 0.5, source=0),
        "joined": RelativeObservation(11, 0.6, source=4),
    })
    assert moved.rebuilt_regions == (0, 1) and moved.coarse_size == 6
    check(moved, rows, {"center": None, "downstream": None})
    check(original, observations, {"bridge": None})
    portless, _ = check(original, observations, {"other_anchor": RelativeObservation(9, 2)})
    assert portless._elimination is original._elimination and portless.visited_buckets == ()
    assert portless.read(10) != original.read(10)
    _assert_direct(original, observations)


def test_indexed_support_tracks_parallel_bridges_and_leaves_unrelated_groups_untouched():
    membership = (*_MEMBERSHIP, 4)
    hierarchy = RelativeObservationHierarchy(membership, ridge=_RIDGE,
                                             elimination_order=range(len(membership)))
    sources = _sources()
    sources["parallel"] = sources["bridge"]
    sources["other_bridge"] = RelativeObservation(12, 0.4, source=10)
    parent = hierarchy.initialize(sources)
    graph = parent._support_graph
    other_node = parent._regions[3].components[10]
    other = graph.groups[other_node]
    incident_node, downstream_node = graph.edges["bridge"]
    assert graph.neighbors[incident_node][downstream_node] == 2

    weight = RelativeObservation(3, 0.4, source=2, weight=2)
    reweighted = hierarchy.revise(parent, {"bridge": weight})
    assert reweighted._support_graph is graph and reweighted.visited_support_components == ()
    single = hierarchy.revise(reweighted, {"parallel": None})
    assert single.visited_support_components == ()
    assert single._support_graph.groups is graph.groups
    assert single._support_graph.neighbors[incident_node][downstream_node] == 1

    detached = hierarchy.revise(single, {"bridge": None})
    right = detached._support_graph.groups[downstream_node]
    assert set(detached.visited_support_components) == right
    assert right.isdisjoint(other) and detached._support_graph.groups[other_node] is other
    assert parent.read(8) is not None and detached.read(8) is None
    assert detached.read(12) == parent.read(12)
    remaining = {name: row for name, row in sources.items() if name not in {"bridge", "parallel"}}
    _assert_direct(detached, remaining, len(membership))
    compacted = hierarchy.compact(detached, max_regions=2)
    assert compacted._support_graph is detached._support_graph
    reordered = hierarchy.reorder(compacted, reversed(range(len(membership))))
    assert reordered._support_graph is detached._support_graph
    _assert_direct(reordered, remaining, len(membership))
    _assert_direct(parent, sources, len(membership))


def test_indexed_support_reprojects_unchanged_bridges_on_local_splits_and_anchor_changes():
    membership = (*_MEMBERSHIP, 4)
    hierarchy = RelativeObservationHierarchy(membership, ridge=_RIDGE)
    sources = {**_sources(), "other_bridge": RelativeObservation(12, 0.4, source=10)}
    parent = hierarchy.initialize(sources)
    other_node = parent._regions[3].components[10]
    other = parent._support_graph.groups[other_node]

    split = hierarchy.revise(parent, {"continue": None})
    assert split.rebuilt_regions == (1,)
    assert split._bridges == parent._bridges
    assert split._support_graph.edges != parent._support_graph.edges
    assert split.read(4) is not None and split.read(8) is None
    assert split._support_graph.groups[other_node] is other
    _assert_direct(split, {name: row for name, row in sources.items() if name != "continue"},
                   len(membership))

    no_anchor = hierarchy.revise(parent, {"anchor": None})
    assert no_anchor._support_graph.edges is parent._support_graph.edges
    assert no_anchor._support_graph.groups[other_node] is other
    assert no_anchor.visited_support_components
    assert no_anchor.read(8) is None and no_anchor.read(12) == parent.read(12)
    restored = hierarchy.revise(no_anchor, {"anchor": sources["anchor"]})
    _assert_direct(restored, sources, len(membership))
    _assert_direct(parent, sources, len(membership))


def _assert_witness(snapshot, cell):
    witness = snapshot.support_witness(cell)
    assert witness.cells[0] == cell and len(set(witness.cells)) == len(witness.cells)
    assert len(witness.relative_sources) == len(witness.cells) - 1
    for first, second, (name, row) in zip(witness.cells, witness.cells[1:], witness.relative_sources):
        assert row is snapshot._sources[name] and {row.source, row.target} == {first, second}
    name, row = witness.anchor_source
    assert row is snapshot._sources[name] and row.source is None and row.target == witness.cells[-1]
    return witness


def test_named_witness_tracks_parallel_bridge_paths_and_selected_history():
    hierarchy = RelativeObservationHierarchy(
        _MEMBERSHIP, ridge=_RIDGE, elimination_order=range(len(_MEMBERSHIP)),
    )
    sources = _sources()
    sources["parallel"] = RelativeObservation(3, 0.9, source=2, weight=2)
    base = hierarchy.initialize(sources)
    original = _assert_witness(base, 8)
    assert original.cells == (8, 7, 6, 5, 4, 3, 2, 0)
    assert tuple(name for name, _ in original.relative_sources) == (
        "end", "late", "downstream", "continue", "center", "bridge", "conflict",
    )
    _, anchor = original.anchor_source
    path_estimate = anchor.value - sum(
        row.value if row.source == first else -row.value
        for first, (_, row) in zip(original.cells, original.relative_sources)
    )
    assert base.read(8) != pytest.approx(path_estimate, abs=1e-3, rel=0)

    changed_bridge = RelativeObservation(3, -0.9, source=2, weight=3)
    revised = hierarchy.revise(base, {"bridge": changed_bridge})
    assert revised._support_graph is base._support_graph
    assert dict(_assert_witness(revised, 8).relative_sources)["bridge"] is changed_bridge
    moved = hierarchy.revise(revised, {"bridge": RelativeObservation(3, -0.9, source=1, weight=3)})
    assert moved._support_graph is revised._support_graph
    assert _assert_witness(moved, 8).cells == (8, 7, 6, 5, 4, 3, 1, 0)
    single = hierarchy.revise(moved, {"bridge": None})
    assert "parallel" in dict(_assert_witness(single, 8).relative_sources)
    disconnected = hierarchy.revise(single, {"parallel": None})
    compacted = hierarchy.compact(disconnected, max_regions=2)
    assert disconnected.support_witness(8) is compacted.support_witness(8) is None
    reversed_bridge = RelativeObservation(2, 0.7, source=3)
    restored = hierarchy.revise(compacted, {"bridge": reversed_bridge})
    assert dict(_assert_witness(restored, 8).relative_sources)["bridge"].source == 3
    reordered = hierarchy.reorder(restored, reversed(range(len(_MEMBERSHIP))))
    assert _assert_witness(reordered, 8) == _assert_witness(restored, 8)
    assert _assert_witness(base, 8) == original


def test_support_witness_can_return_to_a_different_component_in_the_same_region():
    hierarchy = RelativeObservationHierarchy((0, 0, 0, 0, 1, 1, 2), ridge=_RIDGE)
    sources = {
        "local_a": RelativeObservation(1, 0.2, source=0),
        "local_b": RelativeObservation(3, -0.3, source=2),
        "local_c": RelativeObservation(5, 0.6, source=4),
        "to_other": RelativeObservation(4, 0.3, source=1),
        "return": RelativeObservation(2, 0.4, source=5),
        "anchor": RelativeObservation(3, 1.2),
        "backup": RelativeObservation(3, 2.1),
    }
    base = hierarchy.initialize(sources)
    witness = _assert_witness(base, 0)
    assert witness.cells == (0, 1, 4, 5, 2, 3)
    assert witness.anchor_source[0] == "anchor"
    assert base.support_witness(6) is None
    assert _assert_witness(base, 3).relative_sources == ()
    revised = hierarchy.revise(base, {"anchor": None})
    assert revised._support_graph is base._support_graph
    assert _assert_witness(revised, 0).anchor_source[0] == "backup"
    assert _assert_witness(base, 0) == witness


@pytest.mark.parametrize("sparse", (False, True))
def test_named_source_gains_match_dense_counterfactuals_and_a_real_batch(sparse):
    order = range(len(_MEMBERSHIP)) if sparse else None
    hierarchy = RelativeObservationHierarchy(_MEMBERSHIP, ridge=_RIDGE, elimination_order=order)
    observations = _sources()
    base = hierarchy.initialize(observations)
    requested = ("anchor", "first", "conflict", "bridge", "late", "end", "other_anchor", "bridge")
    reference = solve_relative_direct(len(_MEMBERSHIP), observations, ridge=_RIDGE)[8]
    gains = base.source_gains(8, iter(requested))
    assert tuple(name for name, _ in gains) == requested
    assert dict(gains)["other_anchor"] == 0
    assert "first" not in dict(base.support_witness(8).relative_sources)
    assert abs(dict(gains)["first"]) > 0.001
    assert base.source_gains(8, ()) == () and base.source_gains(11, requested) is None
    with pytest.raises(KeyError):
        base.source_gains(8, ("missing",))

    for name, gain in gains:
        changed = replace(observations[name], value=observations[name].value + 2.0)
        new_reference = solve_relative_direct(
            len(_MEMBERSHIP), {**observations, name: changed}, ridge=_RIDGE,
        )[8]
        assert 2 * gain == pytest.approx(new_reference - reference, abs=1e-12, rel=0)
    reconstructed = sum(observations[name].value * gain
                        for name, gain in base.source_gains(8, observations))
    assert reconstructed == pytest.approx(reference, abs=1e-12, rel=0)

    deltas = {"first": 1.4, "late": -0.7, "bridge": 0.5}
    changes = {name: replace(observations[name], value=observations[name].value + delta)
               for name, delta in deltas.items()}
    actual = hierarchy.revise(base, changes)
    predicted = reference + sum(dict(gains)[name] * delta for name, delta in deltas.items())
    assert actual.read(8) == pytest.approx(predicted, abs=1e-12, rel=0)
    assert base.read(8) == pytest.approx(reference, abs=1e-12, rel=0)
    for (_, old_gain), (_, new_gain) in zip(gains, actual.source_gains(8, requested), strict=True):
        assert new_gain == pytest.approx(old_gain, abs=1e-12, rel=0)


@pytest.mark.parametrize("sparse", (False, True))
def test_source_gains_survive_compaction_for_a_retired_port_and_a_remaining_port(sparse):
    order = reversed(range(len(_MEMBERSHIP))) if sparse else None
    hierarchy = RelativeObservationHierarchy(_MEMBERSHIP, ridge=_RIDGE, elimination_order=order)
    observations = {**_sources(), "remote_anchor": RelativeObservation(8, 1.3)}
    original = hierarchy.initialize(observations)
    detached = hierarchy.revise(original, {"bridge": None})
    compacted = hierarchy.compact(detached, max_regions=2)
    assert 3 not in compacted._port_positions and 5 in compacted._port_positions
    requested = ("center", "downstream", "late", "remote_anchor", "anchor")
    for cell in (3, 5):
        response = compacted.source_gains(cell, requested)
        assert response == tuple((name, pytest.approx(gain, abs=1e-12, rel=0))
                                 for name, gain in detached.source_gains(cell, requested))
        assert dict(response)["anchor"] == 0
        changed = hierarchy.revise(compacted, {"center": replace(observations["center"], value=1.5)})
        assert changed.read(cell) - compacted.read(cell) == pytest.approx(
            0.6 * dict(response)["center"], abs=1e-12, rel=0,
        )
        edits = {"center": replace(observations["center"], weight=0.25), "late": None}
        predicted = compacted.predict_source_edits(cell, edits)
        actual = hierarchy.revise(compacted, edits)
        assert predicted == pytest.approx(actual.read(cell), abs=1e-12, rel=0)


@pytest.mark.parametrize("sparse", (False, True))
def test_source_gains_are_zero_across_disconnected_components_sharing_a_region(sparse):
    hierarchy = RelativeObservationHierarchy((0, 0, 0, 0, 1, 2), ridge=_RIDGE,
                                             elimination_order=range(6) if sparse else None)
    observations = {
        "anchor": RelativeObservation(0, 1),
        "first": RelativeObservation(1, 0.5, source=0),
        "bridge": RelativeObservation(4, 0.5, source=1),
        "other_link": RelativeObservation(3, -0.5, source=2),
        "other_bridge": RelativeObservation(5, 0.5, source=3),
        "other_anchor": RelativeObservation(5, 0.7),
    }
    snapshot = hierarchy.initialize(observations)
    gains = dict(snapshot.source_gains(0, observations))
    assert gains["other_link"] == gains["other_bridge"] == gains["other_anchor"] == 0
    assert gains["anchor"] > 0
    assert snapshot.read(5) is not None


@pytest.mark.parametrize("sparse", (False, True))
def test_prediction_combines_weights_values_and_removal_without_building_a_revision(sparse, monkeypatch):
    order = reversed(range(len(_MEMBERSHIP))) if sparse else None
    field = RelativeObservationHierarchy(_MEMBERSHIP, ridge=_RIDGE, elimination_order=order)
    sources = {**_sources(), "parallel": RelativeObservation(3, 1.1, source=2, weight=2)}
    baseline = field.initialize(sources)
    edits = {
        "bridge": None,
        "conflict": replace(sources["conflict"], value=1.25, weight=0.7),
        "late": replace(sources["late"], value=-0.6),
    }
    old_value = baseline.read(8)

    def unexpected(*args, **kwargs):
        pytest.fail("prediction performed a full source revision")

    with monkeypatch.context() as patch:
        patch.setattr(field, "revise", unexpected)
        predicted = baseline.predict_source_edits(8, edits)
        reversed_batch = baseline.predict_source_edits(8, dict(reversed(tuple(edits.items()))))
        unsupported = baseline.predict_source_edits(8, {"bridge": None, "parallel": None})
    assert predicted == pytest.approx(reversed_batch, abs=1e-12, rel=0)
    assert unsupported is None
    actual = field.revise(baseline, edits)
    new_rows = {**sources, **edits}
    del new_rows["bridge"]
    dense = solve_relative_direct(len(_MEMBERSHIP), new_rows, ridge=_RIDGE)[8]
    assert predicted == pytest.approx(actual.read(8), abs=1e-12, rel=0)
    assert predicted == pytest.approx(dense, abs=1e-12, rel=0)
    assert abs(predicted - old_value) > 0.1
    assert baseline.read(8) == old_value and baseline._sources["bridge"] is sources["bridge"]
    assert baseline.predict_source_edits(8, {"other_anchor": None}) == old_value
    assert baseline.predict_source_edits(11, edits) is None


def test_prediction_finds_alternate_local_support_and_rejects_endpoint_changes():
    field = RelativeObservationHierarchy((0, 0), ridge=0.5)
    sources = {
        "anchor": RelativeObservation(0, 1),
        "backup": RelativeObservation(0, 2),
        "a": RelativeObservation(1, 0.4, source=0),
        "b": RelativeObservation(1, 0.8, source=0),
    }
    base = field.initialize(sources)
    edits = {"a": None, "anchor": None}
    assert base.support_witness(1).anchor_source[0] == "anchor"
    assert base.predict_source_edits(1, edits) == pytest.approx(
        field.revise(base, edits).read(1), abs=1e-12, rel=0,
    )
    assert base.predict_source_edits(1, {"a": None, "b": None}) is None
    with pytest.raises(ValueError, match="unchanged endpoints"):
        base.predict_source_edits(1, {"a": RelativeObservation(0, 0.4, source=1)})
    with pytest.raises(KeyError):
        base.predict_source_edits(1, {"missing": None})


@pytest.mark.parametrize("ridge", (1e-15, 1e-16))
def test_prediction_refuses_unresolved_weight_correction_with_parent_preserved(ridge):
    field = RelativeObservationHierarchy((0,), ridge=ridge)
    base = field.initialize({"anchor": RelativeObservation(0, 1),
                             "backup": RelativeObservation(0, 2, weight=ridge)})
    old = base.read(0)
    with pytest.raises(FloatingPointError, match="correction system"):
        base.predict_source_edits(0, {"anchor": None})
    assert field.revise(base, {"anchor": None}).read(0) == pytest.approx(1)
    assert base.read(0) == old
    patch = base.preview_source_patch(0, {"anchor": None}, max_cells=1)
    assert patch.anchored is True and patch.boundary_error == 0
    assert patch.value == pytest.approx(1)


@pytest.mark.parametrize("sparse", (False, True))
def test_source_patch_shrinks_its_boundary_error_and_uses_no_retained_solve(sparse, monkeypatch):
    cells = 40
    ridge = 2.0
    field = RelativeObservationHierarchy(tuple(i // 4 for i in range(cells)), ridge=ridge,
                                         elimination_order=range(cells) if sparse else None)
    sources = {
        "anchor": RelativeObservation(0, 1),
        "remote": RelativeObservation(cells - 1, -1),
        "strong": RelativeObservation(1, 0.8, source=0, weight=80),
        **{f"r{i}": RelativeObservation(i, 0.2, source=i - 1) for i in range(1, cells)},
    }
    base = field.initialize(sources)
    edits = {"strong": None, "r1": RelativeObservation(1, -0.4, source=0)}
    reference = solve_relative_direct(
        cells, {**{name: row for name, row in sources.items() if name != "strong"}, "r1": edits["r1"]},
        ridge=ridge,
    )[0]
    assert base.predict_source_edits(0, edits) == pytest.approx(reference, abs=1e-12, rel=0)

    def unexpected(*args, **kwargs):
        pytest.fail("patch used the retained numerical operator")

    with monkeypatch.context() as patch:
        patch.setattr(field, "revise", unexpected)
        patch.setattr(relative_hierarchy.RelativeHierarchySnapshot, "read_many", unexpected)
        small, medium, full = (base.preview_source_patch(0, edits, max_cells=cap)
                               for cap in (1, 12, cells))
    assert small.cells == (0,) and medium.cells == tuple(range(12))
    assert small.anchored is medium.anchored is full.anchored is True
    assert small.boundary_error > medium.boundary_error > full.boundary_error == 0
    assert medium.boundary_error < 1e-5
    assert abs(medium.value - reference) < medium.boundary_error
    assert full.value == pytest.approx(reference, abs=1e-12, rel=0)
    assert base._sources["strong"] is sources["strong"]


def test_source_patch_overlays_moves_and_additions_with_three_support_outcomes():
    field = RelativeObservationHierarchy((0, 0, 1, 2), ridge=0.25)
    base = field.initialize({
        "anchor": RelativeObservation(0, 2),
        "move": RelativeObservation(1, 0.75, source=0),
        "remote": RelativeObservation(3, -4),
    })
    edits = {
        "move": RelativeObservation(2, -2, source=1),
        "add": RelativeObservation(1, 1, source=3),
        "remote": None,
        "backup": RelativeObservation(2, -1),
    }
    uncertain = base.preview_source_patch(3, edits, max_cells=2)
    anchored = base.preview_source_patch(3, edits, max_cells=3)
    reversed_batch = base.preview_source_patch(3, dict(reversed(tuple(edits.items()))), max_cells=3)
    assert uncertain.cells == (3, 1) and uncertain.anchored is None
    assert uncertain.value is None and uncertain.boundary_error is None
    assert anchored.cells == (3, 1, 2) and anchored.anchored is True
    assert anchored.boundary_error == 0
    assert anchored.value == pytest.approx(field.revise(base, edits).read(3), abs=1e-12, rel=0)
    assert reversed_batch == anchored
    unanchored = base.preview_source_patch(3, {"remote": None, "add": edits["add"], "move": None},
                                           max_cells=3)
    assert unanchored.cells == (3, 1) and unanchored.anchored is False
    assert unanchored.value is None and unanchored.boundary_error is None
    assert base.read(3) == pytest.approx(-3.2) and base.read(2) is None
    with pytest.raises(ValueError, match="unknown"):
        base.preview_source_patch(3, {"missing": None}, max_cells=1)
    with pytest.raises(ValueError, match="cell cap"):
        base.preview_source_patch(3, {}, max_cells=0)


def test_source_patch_envelope_survives_revision_compaction_and_reorder():
    field = RelativeObservationHierarchy((0, 1, 2), ridge=1, elimination_order=range(3))
    base = field.initialize({
        "anchor": RelativeObservation(0, 1),
        "edge": RelativeObservation(1, 0.25, source=0),
        "temporary": RelativeObservation(2, 0, source=1),
        "remote": RelativeObservation(1, 8),
    })
    revised = field.revise(base, {"temporary": None, "anchor": RelativeObservation(0, 0.5)})
    compacted = field.compact(revised, max_regions=2)
    reordered = field.reorder(compacted, reversed(range(3)))
    assert compacted is not revised and reordered is not compacted
    for snapshot in (base, revised, compacted, reordered):
        patch = snapshot.preview_source_patch(0, {}, max_cells=1)
        assert patch.anchored is True
        assert patch.boundary_error == pytest.approx(8)  # (3 * 8) * (1 / 3)
        assert abs(patch.value - snapshot.read(0)) <= patch.boundary_error


def test_source_patch_refuses_underflowed_open_boundary_influence():
    field = RelativeObservationHierarchy((0, 0), ridge=1e100)
    base = field.initialize({
        "anchor": RelativeObservation(0, 0),
        "edge": RelativeObservation(1, 0, source=0, weight=1e-300),
        "remote": RelativeObservation(1, 1e200, weight=1e200),
    })
    with pytest.raises(FloatingPointError, match="boundary error"):
        base.preview_source_patch(0, {}, max_cells=1)
    assert base.support_witness(0) is not None


@pytest.mark.parametrize("order", [None, tuple(reversed(range(len(_MEMBERSHIP))))])
def test_merge_transports_retained_factors_and_named_sources(order, monkeypatch):
    hierarchy = RelativeObservationHierarchy(_MEMBERSHIP, ridge=_RIDGE, elimination_order=order)
    sources = {**_sources(), "a_parallel": RelativeObservation(3, 0.2, source=2, weight=2)}
    parent = hierarchy.initialize(sources)
    shapes = []
    eliminate = relative_hierarchy._eliminate

    def track(matrix, split):
        shapes.append((matrix.shape, split))
        return eliminate(matrix, split)

    def unexpected(*args, **kwargs):
        pytest.fail("merge rebuilt fine factors")

    with monkeypatch.context() as probe:
        probe.setattr(relative_hierarchy, "_eliminate", track)
        probe.setattr(relative_hierarchy, "_build_region", unexpected)
        merged = hierarchy.merge_regions(parent, 1, 0)

    assert shapes == [((5, 4), 2)]
    assert merged.membership == (0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 0)
    assert merged.elimination_order == order
    assert merged.coarse_size == 2 and merged.rebuilt_regions == (0,)
    assert merged._regions[1] is parent._regions[2] and merged._regions[2] is parent._regions[3]
    assert merged._sources is parent._sources
    assert set(merged._bridges) == {"downstream"}
    assert {"bridge", "a_parallel"} <= merged._regions[0]._sources.keys()
    assert "a_parallel" in dict(merged.support_witness(8).relative_sources)
    assert merged.support_witness(11) is None
    _assert_direct(merged, sources)

    rebuilt = hierarchy.repartition(parent, merged.membership)
    assert merged.read_many(range(len(_MEMBERSHIP))) == pytest.approx(
        rebuilt.read_many(range(len(_MEMBERSHIP))), abs=1e-12, rel=0,
    )
    detached = merged._owner.revise(merged, {"bridge": None, "a_parallel": None})
    assert detached.rebuilt_regions == (0,) and detached.read(8) is None
    assert detached.support_witness(8) is None and parent.support_witness(8) is not None
    _assert_direct(detached, {name: row for name, row in sources.items() if name not in ("bridge", "a_parallel")})
    _assert_direct(parent, sources)


def test_merge_composes_an_idle_port_and_can_merge_portless_regions():
    hierarchy = RelativeObservationHierarchy((0, 1, 2), ridge=_RIDGE, elimination_order=(2, 1, 0))
    sources = {"anchor": RelativeObservation(0, 0.7), "edge": RelativeObservation(1, 1.2, source=0)}
    parent = hierarchy.initialize(sources)
    detached = hierarchy.revise(parent, {"edge": None})
    assert detached.coarse_size == 2
    merged = hierarchy.merge_regions(detached, 0, 1)
    assert merged.membership == (0, 0, 1) and merged.coarse_size == 0
    all_cells = merged._owner.merge_regions(merged, 0, 1)
    assert all_cells.membership == (0, 0, 0) and all_cells.coarse_size == 0
    _assert_direct(all_cells, {"anchor": sources["anchor"]}, cells=3)
    with pytest.raises(ValueError, match="distinct existing"):
        all_cells._owner.merge_regions(all_cells, 0, 0)


@pytest.mark.parametrize("order", [None, tuple(reversed(range(len(_MEMBERSHIP))))])
def test_retained_fusion_reuses_children_and_routes_edits_across_nested_boundaries(order, monkeypatch):
    hierarchy = RelativeObservationHierarchy(_MEMBERSHIP, ridge=_RIDGE, elimination_order=order)
    sources = _sources()
    parent = hierarchy.initialize(sources)
    first = hierarchy.merge_regions(parent, 0, 1, retain_children=True)
    nested = first._owner.merge_regions(first, 0, 1, retain_children=True)
    outer = nested._regions[0].fusion
    built = []
    build = relative_hierarchy._build_region

    def track(cells, ports, rows, ridge):
        built.append(tuple(cells))
        return build(cells, ports, rows, ridge)

    monkeypatch.setattr(relative_hierarchy, "_build_region", track)
    leaf_edit = RelativeObservation(1, -0.3, source=0)
    child_edit = nested._owner.revise(nested, {"start": leaf_edit})
    assert built == [(0, 1, 2)]
    child_outer = child_edit._regions[0].fusion
    assert child_outer.children[1] is outer.children[1]
    assert child_outer.children[0].fusion.children[1] is outer.children[0].fusion.children[1]
    _assert_direct(child_edit, sources | {"start": leaf_edit})

    built.clear()
    bridge_edit = RelativeObservation(3, 1.5, source=2, weight=2)
    revised = nested._owner.revise(nested, {"bridge": bridge_edit})
    refreshed = revised._regions[0].fusion
    assert not built and refreshed.children[1] is outer.children[1]
    assert refreshed.children[0].fusion.children == outer.children[0].fusion.children
    assert refreshed.children[0].fusion.bridges["bridge"] is bridge_edit
    _assert_direct(revised, sources | {"bridge": bridge_edit})

    moved = RelativeObservation(8, -0.3, source=1)
    relocated = revised._owner.revise(revised, {"bridge": moved})
    relocated_fusion = relocated._regions[0].fusion
    assert set(built) == {(0, 1, 2), (6, 7, 8)} and len(built) == 2
    assert relocated_fusion.bridges["bridge"] is moved
    assert "bridge" not in relocated_fusion.children[0].fusion.bridges
    assert 1 in relocated_fusion.children[0].ports and 8 in relocated_fusion.children[1].ports
    expected = sources | {"bridge": moved}
    _assert_direct(relocated, expected)

    built.clear()
    outside = RelativeObservation(10, 0.23, source=0)
    promoted = relocated._owner.revise(relocated, {"new_crossing": outside})
    assert set(built) == {(0, 1, 2), (9, 10)} and len(built) == 2
    assert 0 in promoted._regions[0].ports
    _assert_direct(promoted, expected | {"new_crossing": outside})
    removed = promoted._owner.revise(promoted, {"new_crossing": None})
    compacted = removed._owner.compact(removed, max_regions=2, protected_cells=(0, 8, 11))
    assert 0 not in compacted._regions[0].ports and compacted._regions[0].fusion is removed._regions[0].fusion
    after_compaction = RelativeObservation(8, 0.9, source=1)
    changed = compacted._owner.revise(compacted, {"bridge": after_compaction})
    expected["bridge"] = after_compaction
    assert 0 not in changed._regions[0].ports
    _assert_direct(changed, expected)

    built.clear()
    left, right = changed._regions[0].fusion.children
    split = changed._owner.split_region(changed, 0)
    assert split._regions[0] is left and split._regions[-1] is right
    assert split._bridges["bridge"] is after_compaction
    restored = split._owner.split_region(split, 0)
    assert restored._sources is changed._sources and "bridge" in restored._bridges
    assert not built and not restored.rebuilt_regions
    _assert_direct(restored, expected)
    _assert_direct(nested, sources)
    _assert_direct(parent, sources)


@pytest.mark.parametrize("order", [None, tuple(reversed(range(len(_MEMBERSHIP))))])
def test_lazy_fusion_uses_requested_rows_for_reads_and_adjoint_after_edits(order, monkeypatch):
    hierarchy = RelativeObservationHierarchy(_MEMBERSHIP, ridge=_RIDGE, elimination_order=order)
    sources = _sources() | {"external": RelativeObservation(9, 0.35, source=0)}
    parent = hierarchy.initialize(sources)
    eager_child = hierarchy.merge_regions(parent, 0, 1, retain_children=True)
    eager = eager_child._owner.merge_regions(eager_child, 0, 1, retain_children=True)
    with pytest.raises(ValueError, match="requires retained children"):
        hierarchy.merge_regions(parent, 0, 1, lazy_readout=True)
    lazy_child = hierarchy.merge_regions(parent, 0, 1, retain_children=True, lazy_readout=True)
    eager_over_lazy = lazy_child._owner.merge_regions(lazy_child, 0, 1, retain_children=True)
    assert not eager_over_lazy._regions[0].lazy_readout
    assert eager_over_lazy._regions[0].fusion.children[0] is lazy_child._regions[0]
    _assert_direct(eager_over_lazy, sources)

    def unexpected(*args, **kwargs):
        pytest.fail("lazy revision requested a complete child readout")

    monkeypatch.setattr(relative_hierarchy, "_full_region_readout", unexpected)
    lazy = lazy_child._owner.merge_regions(lazy_child, 0, 1, retain_children=True, lazy_readout=True)
    region = lazy._regions[0]
    assert region.ports == (0,) and region.interior_size == 9
    assert region.readout.shape == (2, 2)
    assert lazy_child._regions[0].readout.shape == (2, 3)
    assert eager._regions[0].readout.shape == (9, 2)

    calls = []
    readout_row = relative_hierarchy._region_readout_row

    def count(region, cell):
        calls.append(cell)
        return readout_row(region, cell)

    requested = (8, 8, 4, 11, 0, 9)
    expected = eager.read_many(requested)
    with monkeypatch.context() as tracked:
        tracked.setattr(relative_hierarchy, "_region_readout_row", count)
        assert lazy.read_many(requested) == pytest.approx(expected, abs=1e-12, rel=0)
        assert calls == [8, 4]
    _assert_direct(lazy, sources)
    assert lazy.support_witness(8) == eager.support_witness(8)

    changed = {"bridge": RelativeObservation(3, -0.7, source=2, weight=1.8)}
    assert dict(lazy.source_gains(8, tuple(sources))) == pytest.approx(
        dict(eager.source_gains(8, tuple(sources))), abs=1e-12, rel=0,
    )
    assert lazy.predict_source_edits(8, changed) == pytest.approx(
        eager.predict_source_edits(8, changed), abs=1e-12, rel=0,
    )
    revised = lazy._owner.revise(lazy, changed)
    assert revised._regions[0].lazy_readout and revised._regions[0].readout.shape == (2, 2)
    temporary = revised._owner.revise(revised, {"temporary": RelativeObservation(10, 0.6, source=8)})
    removed = temporary._owner.revise(temporary, {"temporary": None})
    compacted = removed._owner.compact(removed, max_regions=2, protected_cells=(4, 8, 11))
    assert compacted._regions[0].readout.shape == (3, 2) and compacted._regions[0].lazy_readout
    split = compacted._owner.split_region(compacted, 0)
    assert split._regions[0].lazy_readout
    for child in (revised, compacted, split):
        _assert_direct(child, sources | changed)
    _assert_direct(lazy, sources)


def _finite_cancellation_sources():
    return {
        "negative-anchor": RelativeObservation(3, -1.05e308, weight=0.25),
        "surviving-boundary": RelativeObservation(0, 0, source=3, weight=0.25),
        "first-rise": RelativeObservation(1, 1.02e308, source=0, weight=0.25),
        "second-rise": RelativeObservation(2, 1.02e308, source=1, weight=0.25),
    }


@pytest.mark.parametrize("order", [None, (0, 1, 2, 3)])
def test_value_first_reads_finite_answer_after_lifted_constant_overflows(order):
    sources = _finite_cancellation_sources()
    hierarchy = RelativeObservationHierarchy((0, 1, 1, 2), ridge=0.002, elimination_order=order)
    base = hierarchy.initialize(sources)
    revised = hierarchy.merge_regions(base, 0, 1, retain_children=True, lazy_readout=True)
    with pytest.raises(FloatingPointError):
        revised.read(2)
    expected = solve_relative_direct(4, sources, ridge=0.002)
    assert revised.read_many((2, 1, 0, 2, 3), value_first=True) == pytest.approx(
        tuple(expected[cell] for cell in (2, 1, 0, 2, 3)), rel=1e-12, abs=0,
    )
    assert revised.read(2, value_first=True) == pytest.approx(expected[2], rel=1e-12, abs=0)


@pytest.mark.parametrize("order", [None, tuple(reversed(range(len(_MEMBERSHIP))))])
def test_value_first_shares_stored_conditionals_after_compaction_and_split(order, monkeypatch):
    sources = _sources() | {"external": RelativeObservation(9, 0.35, source=0)}
    hierarchy = RelativeObservationHierarchy(_MEMBERSHIP, ridge=_RIDGE, elimination_order=order)
    original = hierarchy.initialize(sources)
    first = hierarchy.merge_regions(original, 0, 1, retain_children=True, lazy_readout=True)
    nested = first._owner.merge_regions(first, 0, 1, retain_children=True, lazy_readout=True)
    requested = (7, 8, 8, 4, 11, 0, 9)
    expected = solve_relative_direct(len(_MEMBERSHIP), sources, ridge=_RIDGE)
    expected_batch = tuple(expected[cell] for cell in requested)
    planned, coarse = [], []
    stored = relative_hierarchy._stored_readout
    read_ports = type(nested)._read_ports

    def watch_row(region, cell, path=None):
        planned.append(cell)
        return stored(region, cell, path)

    def watch_ports(snapshot, ports):
        coarse.append(tuple(ports))
        return read_ports(snapshot, ports)

    def unexpected(*args, **kwargs):
        pytest.fail("value-first path assembled a lifted affine row")

    with monkeypatch.context() as watch:
        watch.setattr(relative_hierarchy, "_stored_readout", watch_row)
        watch.setattr(relative_hierarchy, "_region_readout_row", unexpected)
        watch.setattr(type(nested), "_read_ports", watch_ports)
        assert nested.read_many(iter(requested), value_first=True) == pytest.approx(
            expected_batch, abs=1e-12, rel=0,
        )
        assert len(planned) == len(set(planned)) and {6, 7, 8} <= set(planned)
        assert len(coarse) == 1 and set(coarse[0]) == {0, 9}
        batched = len(planned)
        planned.clear()
        for cell in dict.fromkeys(requested):
            nested.read(cell, value_first=True)
        assert len(planned) > batched
    assert nested.read_many(iter(()), value_first=True) == ()
    with pytest.raises(ValueError, match="out of range"):
        nested.read_many((0, -1), value_first=True)

    temporary = nested._owner.revise(nested, {"temporary": RelativeObservation(10, 0.6, source=8)})
    removed = temporary._owner.revise(temporary, {"temporary": None})
    compacted = removed._owner.compact(removed, max_regions=2, protected_cells=requested, value_first=True)
    split = compacted._owner.split_region(compacted, 0)
    for state in (compacted, split):
        assert state.read_many(range(len(_MEMBERSHIP)), value_first=True) == pytest.approx(
            expected, abs=1e-12, rel=0,
        )
