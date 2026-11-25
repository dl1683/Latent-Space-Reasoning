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
    steps, old_values, _ = relative_elimination.solve_elimination((0, 1), rank, [first], {})
    revised, values, rebuilt = relative_elimination.solve_elimination((0, 1), rank, [changed], steps)
    assert rebuilt == (0,)
    assert revised[0].remainder is steps[0].remainder and revised[1] is steps[1]
    assert np.array_equal(old_values, [2, 3])
    assert np.array_equal(values, [4, 3])


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
