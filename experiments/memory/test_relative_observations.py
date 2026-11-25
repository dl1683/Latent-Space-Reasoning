import numpy as np
import pytest

from experiments.memory.relative_observations import (
    RelativeObservation,
    RelativeObservationCG,
    RelativeObservationField,
    solve_relative_direct,
)


def _assert_oracle_agreement(snapshot, observations, cells, ridge):
    expected = solve_relative_direct(cells, observations, ridge=ridge)
    for cell, value in enumerate(expected):
        actual = snapshot.read(cell)
        if value is None:
            assert actual is None
        else:
            assert actual == pytest.approx(value, abs=snapshot.error_bound + 2e-14, rel=0)


def _measurements():
    return {
        "anchor": RelativeObservation(0, 0.25, weight=2),
        "first": RelativeObservation(1, 1, source=0),
        "second": RelativeObservation(2, 0.75, source=1),
        "noisy_shortcut": RelativeObservation(2, 1.2, source=0, weight=2),
        "bridge": RelativeObservation(3, 0.4, source=2),
        "late": RelativeObservation(4, 0.7, source=3),
        "unrelated_anchor": RelativeObservation(5, 0.8),
        "unrelated_difference": RelativeObservation(6, -0.25, source=5),
    }


def test_affine_relative_field_matches_a_hand_solved_orientation_control():
    observations = {"anchor": RelativeObservation(0, 2), "delta": RelativeObservation(1, 1, source=0)}
    field = RelativeObservationField(2, ridge=1, tolerance=1e-11)
    snapshot = field.initialize(observations)
    # [3 -1; -1 2] x = [1, 1].
    assert [snapshot.read(0), snapshot.read(1)] == pytest.approx([0.6, 0.8], abs=1e-11, rel=0)
    assert solve_relative_direct(2, observations, ridge=1) == pytest.approx([0.6, 0.8])


def test_inconsistent_triangle_matches_hand_solution_before_and_after_correction():
    observations = {
        "anchor": RelativeObservation(0, 0),
        "ab": RelativeObservation(1, 1, source=0),
        "bc": RelativeObservation(2, 1, source=1),
        "ac": RelativeObservation(2, 3, source=0),
    }
    field = RelativeObservationField(3, ridge=1, tolerance=1e-11)
    original = field.initialize(observations)
    before, after = [-2 / 3, 1 / 6, 7 / 6], [-5 / 6, 5 / 24, 35 / 24]
    update = {"ac": RelativeObservation(2, 4, source=0)}
    revised = field.revise(original, update)
    assert [original.read(cell) for cell in range(3)] == pytest.approx(before, abs=1e-11, rel=0)
    assert [revised.read(cell) for cell in range(3)] == pytest.approx(after, abs=1e-11, rel=0)
    assert solve_relative_direct(3, observations, ridge=1) == pytest.approx(before)
    assert solve_relative_direct(3, {**observations, **update}, ridge=1) == pytest.approx(after)


def test_noisy_revision_removal_restoration_and_old_read_match_independent_oracle():
    observations = _measurements()
    field = RelativeObservationField(7, ridge=0.5, tolerance=1e-10)
    base = field.initialize(observations)
    _assert_oracle_agreement(base, observations, 7, 0.5)
    early_answer, unaffected = base.read(4), (base.read(5), base.read(6))
    correction = RelativeObservation(2, -0.6, source=0, weight=2)
    revised = field.revise(base, {"noisy_shortcut": correction})
    corrected = {**observations, "noisy_shortcut": correction}
    _assert_oracle_agreement(revised, corrected, 7, 0.5)
    assert abs(revised.read(4) - early_answer) > 0.1
    assert revised.recompiled == (0, 2)
    assert revised.affected == (0, 1, 2, 3, 4)
    assert revised._field is base._field
    assert (revised.read(5), revised.read(6)) == unaffected

    disconnected = field.revise(revised, {"bridge": None})
    without_bridge = {name: obs for name, obs in corrected.items() if name != "bridge"}
    _assert_oracle_agreement(disconnected, without_bridge, 7, 0.5)
    assert disconnected.recompiled == (2, 3)
    assert disconnected.read(3) is None and disconnected.read(4) is None
    assert disconnected._state.values[4, 0] != 0
    assert (disconnected.read(5), disconnected.read(6)) == unaffected

    restored = field.revise(disconnected, {"bridge": observations["bridge"]})
    _assert_oracle_agreement(restored, corrected, 7, 0.5)
    assert restored.read(4) == pytest.approx(revised.read(4), abs=2e-10, rel=0)
    assert base.read(4) == early_answer


def test_permuted_identifiers_and_reversed_observation_directions_keep_late_answer():
    observations = _measurements()
    order = [3, 1, 6, 5, 0, 2, 4]
    permuted = {}
    for i, row in enumerate(reversed(list(observations.values()))):
        if row.source is None:
            rewritten = RelativeObservation(order[row.target], row.value, weight=row.weight)
        else:
            rewritten = RelativeObservation(order[row.source], -row.value,
                                            source=order[row.target], weight=row.weight)
        permuted[f"fact-{i}"] = rewritten
    field = RelativeObservationField(7, ridge=0.5, tolerance=1e-10)
    original, reordered = field.initialize(observations), field.initialize(permuted)
    _assert_oracle_agreement(reordered, permuted, 7, 0.5)
    assert [reordered.read(cell) for cell in order] == pytest.approx(
        [original.read(cell) for cell in range(7)], abs=2e-10, rel=0,
    )


def test_supported_zero_and_evidence_removal_have_distinct_results_without_numeric_rework():
    field = RelativeObservationField(2, ridge=1)
    observations = {"zero": RelativeObservation(0, 0)}
    original = field.initialize(observations)
    assert original.read(0) == 0 and original.read(1) is None
    removed = field.revise(original, {"zero": None})
    _assert_oracle_agreement(removed, {}, 2, 1)
    assert removed.read(0) is None and original.read(0) == 0
    assert removed.recompiled == (0,) and removed.affected == () and removed.messages == 0
    with pytest.raises(ValueError, match="unknown observation"):
        field.revise(original, {"new": RelativeObservation(1, 1), "missing": None})
    assert field.revise(original, observations) is original


@pytest.mark.parametrize("solver", [RelativeObservationField, RelativeObservationCG])
def test_weight_endpoint_and_parallel_source_edits_match_a_fresh_assembly(solver):
    observations = _measurements()
    observations["other_bridge"] = RelativeObservation(3, 0.8, source=2, weight=0.25)
    field = solver(7, ridge=0.5, tolerance=1e-10)
    base = field.initialize(observations)
    changes = {
        "bridge": RelativeObservation(5, 0.4, source=4, weight=4),
        "unrelated_anchor": RelativeObservation(5, 0.8, weight=0.25),
    }
    revised = field.revise(base, changes)
    _assert_oracle_agreement(revised, {**observations, **changes}, 7, 0.5)
    assert revised.recompiled == (2, 3, 4, 5)
    assert revised.affected == tuple(range(7))
    without_parallel = field.revise(revised, {"other_bridge": None})
    final = {name: row for name, row in {**observations, **changes}.items()
             if name != "other_bridge"}
    _assert_oracle_agreement(without_parallel, final, 7, 0.5)
    assert base.read(5) == pytest.approx(solve_relative_direct(7, observations, ridge=0.5)[5])


def test_sparse_cg_warm_starts_affected_component_and_keeps_old_bounds():
    observations = _measurements()
    cg = RelativeObservationCG(7, ridge=0.5, tolerance=1e-10)
    base = cg.initialize(observations)
    _assert_oracle_agreement(base, observations, 7, 0.5)
    untouched = base.read(5), base.read(6)
    correction = RelativeObservation(2, -0.6, source=0, weight=2)
    revised = cg.revise(base, {"noisy_shortcut": correction})
    corrected = {**observations, "noisy_shortcut": correction}
    _assert_oracle_agreement(revised, corrected, 7, 0.5)
    assert revised.recompiled == (0, 2) and revised.affected == (0, 1, 2, 3, 4)
    assert revised.iterations > 0 and revised.matvecs >= revised.iterations + 2
    assert revised.messages == 2 * 5 * revised.matvecs
    assert revised.error_bound >= base.error_bound
    assert (revised.read(5), revised.read(6)) == untouched
    assert abs(revised.read(4) - base.read(4)) > 0.1

    heavier_bridge = RelativeObservation(3, 0.4, source=2, weight=3)
    reweighted = cg.revise(revised, {"bridge": heavier_bridge})
    _assert_oracle_agreement(reweighted, {**corrected, "bridge": heavier_bridge}, 7, 0.5)
    assert reweighted.recompiled == (2, 3) and reweighted.affected == (0, 1, 2, 3, 4)

    detached = cg.revise(reweighted, {"bridge": None})
    without_bridge = {name: obs for name, obs in corrected.items() if name != "bridge"}
    _assert_oracle_agreement(detached, without_bridge, 7, 0.5)
    assert detached.affected == (0, 1, 2, 3, 4)
    assert detached.read(4) is None and revised.read(4) is not None
    late = RelativeObservation(4, -0.6, source=3)
    revised_detached = cg.revise(detached, {"late": late})
    assert revised_detached.affected == (3, 4)
    assert revised_detached.iterations > 0
    _assert_oracle_agreement(revised_detached, {**without_bridge, "late": late}, 7, 0.5)
    restored = cg.revise(revised_detached, {"bridge": observations["bridge"]})
    _assert_oracle_agreement(restored, {**corrected, "late": late}, 7, 0.5)
    assert (restored.read(5), restored.read(6)) == untouched
    fork = cg.revise(base, {"bridge": heavier_bridge})
    _assert_oracle_agreement(fork, {**observations, "bridge": heavier_bridge}, 7, 0.5)
    _assert_oracle_agreement(base, observations, 7, 0.5)


def test_sparse_cg_accepts_a_certified_warm_start_with_no_iterations():
    cg = RelativeObservationCG(2, ridge=1, tolerance=0.01)
    old = cg.initialize({"anchor": RelativeObservation(0, 1)})
    observations = {"anchor": RelativeObservation(0, 1.001)}
    revised = cg.revise(old, observations)
    assert revised.affected == (0,)
    assert revised.iterations == 0 and revised.matvecs == 1
    assert revised.read(0) == old.read(0) == 0.5
    assert revised.error_bound == pytest.approx(0.001)
    _assert_oracle_agreement(revised, observations, 2, 1)


def test_sparse_cg_budget_exhaustion_preserves_prior_snapshot():
    cg = RelativeObservationCG(2, ridge=0.5, tolerance=1e-10, max_iterations=1)
    observations = {
        "anchor": RelativeObservation(0, 0),
        "relative": RelativeObservation(1, 0, source=0),
    }
    original = cg.initialize(observations)
    with pytest.raises(RuntimeError, match="exhausted its iteration budget"):
        cg.revise(original, {"anchor": RelativeObservation(0, 2)})
    assert original.read(0) == original.read(1) == 0
    assert cg.revise(original, observations) is original


def test_sparse_cg_rejects_scaling_that_erases_a_nonzero_rhs():
    cg = RelativeObservationCG(2, ridge=1)
    smallest = float.fromhex("0x0.0000000000001p-1022")
    observations = {
        "low": RelativeObservation(0, 2 * smallest),
        "high": RelativeObservation(1, 0, weight=15),
    }
    with pytest.raises(FloatingPointError, match="unresolved right-hand side"):
        cg.initialize(observations)
