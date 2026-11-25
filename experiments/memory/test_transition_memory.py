from itertools import product
from math import exp, log

import numpy as np
import pytest

from experiments.memory import transition_memory
from experiments.memory.transition_memory import TransitionMemory, continue_latent


def _sequential(seed, slots, operators, start, stop):
    current = np.asarray(seed, dtype=float)
    for source in slots[start:stop]:
        operator = operators.get(source)
        if operator is None:
            return None
        current = operator @ current
    return current


def test_late_ordered_binding_requires_both_sources_and_the_right_order():
    slots = ("binding", "consequence", "independent", "identity")
    first = np.eye(3)
    first[1, 0] = 1
    second = np.eye(3)
    second[2, 1] = 1
    other = np.diag([0.5, 0.25, 0.125])
    forward = dict(zip(slots, (first, second, other, np.eye(3))))
    reverse = dict(zip(slots, (second, first, other, np.eye(3))))
    np.testing.assert_array_equal(sum(forward.values()), sum(reverse.values()))
    base = TransitionMemory.prepare(slots, 3, forward)
    swapped = base.revise(reverse)
    assert swapped._root.right is base._root.right

    # The seed, interval and target coordinate are selected after both compilations.
    seed = np.array([0.75, 0, 0])
    target = 2
    assert continue_latent(seed, memory=base, stop=2)[target] == 0.75
    assert continue_latent(seed, memory=swapped, stop=2)[target] == 0
    assert continue_latent(seed, memory=base, stop=1)[target] == 0
    assert continue_latent(seed, memory=base, start=1, stop=2)[target] == 0
    np.testing.assert_array_equal(continue_latent(seed), seed)
    for source in slots[:2]:
        removed = base.revise({source: None})
        assert continue_latent(seed, memory=removed, stop=2) is None
        restored = removed.revise({source: forward[source]})
        assert continue_latent(seed, memory=restored, stop=2)[target] == 0.75
        np.testing.assert_array_equal(continue_latent(seed, memory=removed, start=2), other @ seed)

    replaced = first.copy()
    replaced[1, 0] = -1
    revised = base.revise({"binding": replaced})
    assert continue_latent(seed, memory=revised, stop=2)[target] == -0.75
    replaced = second.copy()
    replaced[2, 1] = 2
    revised = base.revise({"consequence": replaced})
    assert continue_latent(seed, memory=revised, stop=2)[target] == 1.5
    restored = swapped.revise(forward)
    np.testing.assert_array_equal(continue_latent(seed, memory=restored), continue_latent(seed, memory=base))
    assert continue_latent(seed, memory=base, stop=2)[target] == 0.75
    renamed_slots = ("z", "y", "x", "w")
    renamed = TransitionMemory.prepare(renamed_slots, 3, dict(zip(renamed_slots, forward.values())))
    np.testing.assert_array_equal(continue_latent(seed, memory=renamed), continue_latent(seed, memory=base))


def test_subranges_and_forks_match_equal_access_sequential_continuation():
    rng = np.random.default_rng(602)
    slots = tuple(f"row-{i}" for i in range(13))
    operators = {source: rng.uniform(-0.3, 0.3, (3, 3)) for source in slots}
    original = TransitionMemory.prepare(slots, 3, operators)
    edits = {slots[1]: None, slots[6]: rng.uniform(-0.3, 0.3, (3, 3)), slots[10]: np.zeros((3, 3))}
    revised = original.revise(edits)
    restored = revised.revise({slots[1]: operators[slots[1]]})
    fork = original.revise({slots[6]: np.eye(3)})
    views = ((original, operators), (revised, operators | edits),
             (restored, operators | edits | {slots[1]: operators[slots[1]]}),
             (fork, operators | {slots[6]: np.eye(3)}))
    for snapshot, sources in views:
        for start in range(len(slots) + 1):
            for stop in range(start, len(slots) + 1):
                seed = rng.uniform(-1, 1, 3)
                actual = continue_latent(seed, memory=snapshot, start=start, stop=stop)
                expected = _sequential(seed, slots, sources, start, stop)
                if expected is None:
                    assert actual is None
                else:
                    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
    assert continue_latent([1, 2, 3], memory=revised, start=10, stop=11).tolist() == [0, 0, 0]


@pytest.mark.parametrize("algebra", ["linear", "max_plus", "log_sum_exp"])
def test_retention_counts_compilation_revision_and_late_read_separately(monkeypatch, algebra):
    slots = tuple(f"operator-{i}" for i in range(64))
    if algebra == "linear":
        identity = np.eye(2)
        first, second = np.diag([2, 1]), np.diag([0.5, 1])
        kernel = "_multiply"
    else:
        identity = np.array([[0, -np.inf], [-np.inf, 0]])
        first, second = identity.copy(), identity.copy()
        first[0, 0], second[0, 0] = 2, -2
        kernel = "_path_product"
    operators = dict.fromkeys(slots, identity)
    calls = []
    multiply = getattr(transition_memory, kernel)

    def watch(left, right, **kwargs):
        calls.append(right.ndim)
        return multiply(left, right, **kwargs)

    monkeypatch.setattr(transition_memory, kernel, watch)
    base = TransitionMemory.prepare(slots, 2, operators, algebra=algebra)
    assert calls == [2] * 63
    calls.clear()
    revised = base.revise({slots[7]: first, slots[8]: second})
    assert calls == [2] * 9 and revised._root.right is base._root.right
    calls.clear()
    assert revised.revise({slots[7]: first}) is revised and calls == []
    np.testing.assert_array_equal(continue_latent([3, 4], memory=revised), [3, 4])
    assert calls == [1]


def test_source_and_result_arrays_are_isolated_and_range_failures_are_explicit():
    supplied = np.array([[1.0]])
    base = TransitionMemory.prepare(("first", "second"), 1, {"first": supplied, "second": supplied})
    supplied[0, 0] = 0
    output = continue_latent([1], memory=base)
    output.shape = ()
    assert continue_latent([1], memory=base).tolist() == [1.0]
    with pytest.raises(FloatingPointError):
        base.revise({"first": [[1e308]], "second": [[2.0]]})
    extreme = dict.fromkeys(base.source_ids, np.array([[1e200]]))
    np.testing.assert_allclose(_sequential([1e-300], base.source_ids, extreme, 0, 2), [1e100])
    with pytest.raises(FloatingPointError):
        base.revise(extreme)
    tiny = dict.fromkeys(base.source_ids, np.array([[1e-200]]))
    np.testing.assert_allclose(_sequential([1e300], base.source_ids, tiny, 0, 2), [1e-100], atol=0)
    np.testing.assert_array_equal(continue_latent([1e300], memory=base.revise(tiny)), [0])
    with pytest.raises(ValueError, match="unknown transition source"):
        base.revise({"first": [[2]], "other": [[2]]})
    with pytest.raises(ValueError, match="square"):
        base.revise({"first": [[2, 0]]})
    with pytest.raises(ValueError, match="outside"):
        continue_latent([1], memory=base, stop=3)
    with pytest.raises(ValueError, match="requires memory"):
        continue_latent([1], stop=1)
    assert continue_latent([1], memory=base).tolist() == [1]
    large = base.revise({"first": [[1e308]], "second": None})
    assert continue_latent([2], memory=large) is None
    with pytest.raises(FloatingPointError):
        continue_latent([2], memory=large, stop=1)
    assert continue_latent([0.5], memory=large, stop=1).tolist() == [5e307]


def test_late_path_aggregation_preserves_alternatives_and_is_nonlinear_in_scores():
    slots = ("enter", "finish", "other", "tail")
    enter = np.array([[2, -np.inf], [0, 1]])
    finish = np.array([[0, 0], [-np.inf, -np.inf]])
    identity = np.array([[0, -np.inf], [-np.inf, 0]])
    sources = dict(zip(slots, (enter, finish, identity, identity)))
    best = TransitionMemory.prepare(slots, 2, sources, algebra="max_plus")
    all_paths = TransitionMemory.prepare(slots, 2, sources, algebra="log_sum_exp")
    seed = [0, -np.inf]
    result = continue_latent(seed, memory=all_paths, stop=2)
    assert result[0] == pytest.approx(log(exp(2) + 1))
    assert result[1] == -np.inf
    assert continue_latent(seed, memory=best, stop=2)[0] == 2
    tied = enter.copy()
    tied[1, 0] = 2
    assert continue_latent(seed, memory=best.revise({"enter": tied}), stop=2)[0] == 2
    assert continue_latent(seed, memory=all_paths.revise({"enter": tied}), stop=2)[0] == pytest.approx(2 + log(2))

    alternative_removed = enter.copy()
    alternative_removed[1, 0] = -np.inf
    for snapshot in (best, all_paths):
        revised = snapshot.revise({"enter": alternative_removed})
        assert revised._root.right is snapshot._root.right
        assert continue_latent(seed, memory=revised, stop=2)[0] == 2
        assert continue_latent([-np.inf, 0], memory=revised, stop=2)[0] == 1
        assert continue_latent(seed, memory=snapshot.revise({"finish": None}), stop=2) is None
        np.testing.assert_array_equal(continue_latent(seed, memory=revised, start=2), seed)
        np.testing.assert_array_equal(continue_latent(seed, memory=revised, start=2, stop=2), seed)
        restored = revised.revise({"enter": enter})
        np.testing.assert_array_equal(continue_latent(seed, memory=restored, stop=2),
                                      continue_latent(seed, memory=snapshot, stop=2))
        a, b = np.array([0, -4]), np.array([-4, 4])
        separate = (continue_latent(a, memory=snapshot)[0] + continue_latent(b, memory=snapshot)[0]) / 2
        together = continue_latent((a + b) / 2, memory=snapshot)[0]
        assert abs(together - separate) > 1
    np.testing.assert_array_equal(continue_latent(seed, memory=all_paths, stop=2), result)


def test_removing_a_weaker_path_changes_the_soft_decision_only():
    enter = np.full((3, 3), -np.inf)
    enter[:, 0] = [0, -0.5, -0.5]
    finish = np.array([[0, -np.inf, -np.inf], [-np.inf, 0, 0], [-np.inf, -np.inf, -np.inf]])
    sources = {"enter": enter, "finish": finish}
    best = TransitionMemory.prepare(sources, 3, sources, algebra="max_plus")
    all_paths = TransitionMemory.prepare(sources, 3, sources, algebra="log_sum_exp")
    seed = [0, -np.inf, -np.inf]
    entered = continue_latent(seed, memory=all_paths)
    np.testing.assert_allclose(entered, [0, log(2) - 0.5, -np.inf])
    assert np.argmax(entered) == 1
    reduced = enter.copy()
    reduced[2, 0] = -np.inf
    np.testing.assert_array_equal(continue_latent(seed, memory=best), [0, -0.5, -np.inf])
    for snapshot in (best, all_paths):
        revised = snapshot.revise({"enter": reduced})
        fallback = continue_latent(seed, memory=revised)
        np.testing.assert_array_equal(fallback, [0, -0.5, -np.inf])
        assert np.argmax(fallback) == 0


def _enumerated_paths(seed, layers, algebra):
    if any(layer is None for layer in layers):
        return None
    scores = [[] for _ in seed]
    for path in product(range(len(seed)), repeat=len(layers) + 1):
        score = seed[path[0]] + sum(layer[path[i + 1], path[i]] for i, layer in enumerate(layers))
        scores[path[-1]].append(score)
    if algebra == "max_plus":
        return [max(paths) for paths in scores]
    totals = [sum(exp(score) for score in paths) for paths in scores]
    return [log(total) if total else -np.inf for total in totals]


@pytest.mark.parametrize("algebra", ["max_plus", "log_sum_exp"])
def test_path_products_match_explicit_enumeration_on_forked_subranges(algebra):
    rng = np.random.default_rng(818)
    slots = ("a", "b", "c", "d", "e")
    bank = {}
    for source in slots:
        layer = rng.integers(-3, 4, size=(2, 2)).astype(float)
        layer[rng.random(size=(2, 2)) < 0.3] = -np.inf
        bank[source] = layer
    original = TransitionMemory.prepare(slots, 2, bank, algebra=algebra)
    edits = {"b": None, "d": np.full((2, 2), -np.inf)}
    revised = original.revise(edits)
    restored = revised.revise({"b": bank["b"]})
    fork = original.revise({"e": np.zeros((2, 2))})
    views = ((original, bank), (revised, bank | edits),
             (restored, bank | edits | {"b": bank["b"]}), (fork, bank | {"e": np.zeros((2, 2))}))
    for snapshot, sources in views:
        for start in range(len(slots) + 1):
            for stop in range(start, len(slots) + 1):
                seed = rng.integers(-3, 4, size=2).astype(float)
                seed[rng.random(size=2) < 0.3] = -np.inf
                expected = _enumerated_paths(seed, [sources[key] for key in slots[start:stop]], algebra)
                actual = continue_latent(seed, memory=snapshot, start=start, stop=stop)
                if expected is None:
                    assert actual is None
                else:
                    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)


@pytest.mark.parametrize("algebra", ["max_plus", "log_sum_exp"])
def test_path_products_distinguish_unreachable_from_invalid_and_overflow(algebra):
    slots = ("a", "b")
    snapshot = TransitionMemory.prepare(slots, 1, dict.fromkeys(slots, [[0]]), algebra=algebra)
    for invalid in (np.nan, np.inf, 1j, "-inf"):
        with pytest.raises(ValueError):
            snapshot.revise({"a": [[invalid]]})
        with pytest.raises(ValueError):
            continue_latent([invalid], memory=snapshot)
    widest = np.finfo(np.longdouble).max
    if widest > np.finfo(np.float64).max:
        with pytest.raises(FloatingPointError):
            snapshot.revise({"a": np.array([[-widest]])})
        with pytest.raises(FloatingPointError):
            continue_latent(np.array([-widest]), memory=snapshot)
    for large in (-1e308, 1e308):
        with pytest.raises(FloatingPointError):
            snapshot.revise(dict.fromkeys(slots, [[large]]))
        partial = snapshot.revise({"a": [[large]], "b": None})
        assert continue_latent([large], memory=partial) is None
        with pytest.raises(FloatingPointError):
            continue_latent([large], memory=partial, stop=1)
        # A balanced tree adds the two large same-sign stages before cancellation.
        stages = {"a": [[-large]], "b": [[large]], "c": [[large]]}
        total = 0.0
        for source in stages.values():
            total += source[0][0]
        assert total == large
        with pytest.raises(FloatingPointError):
            TransitionMemory.prepare(stages, 1, stages, algebra=algebra)
    wide = TransitionMemory.prepare(("a",), 2, {"a": [[1e308, -1e308], [-np.inf, -np.inf]]}, algebra=algebra)
    np.testing.assert_array_equal(continue_latent([0, 0], memory=wide), [1e308, -np.inf])
    np.testing.assert_array_equal(continue_latent([-np.inf, -np.inf], memory=wide), [-np.inf, -np.inf])
    np.testing.assert_array_equal(continue_latent([0], memory=snapshot), [0])
