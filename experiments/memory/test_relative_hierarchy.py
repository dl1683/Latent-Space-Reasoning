import pytest

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


def _assert_direct(snapshot, sources):
    expected = solve_relative_direct(len(_MEMBERSHIP), sources, ridge=_RIDGE)
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
