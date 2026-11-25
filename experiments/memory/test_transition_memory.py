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


def test_retention_counts_compilation_revision_and_late_read_separately(monkeypatch):
    slots = tuple(f"operator-{i}" for i in range(64))
    operators = {source: np.eye(2) for source in slots}
    calls = []
    multiply = transition_memory._multiply

    def watch(left, right):
        calls.append(right.ndim)
        return multiply(left, right)

    monkeypatch.setattr(transition_memory, "_multiply", watch)
    base = TransitionMemory.prepare(slots, 2, operators)
    assert calls == [2] * 63
    calls.clear()
    revised = base.revise({slots[7]: np.diag([2, 1]), slots[8]: np.diag([0.5, 1])})
    assert calls == [2] * 9 and revised._root.right is base._root.right
    calls.clear()
    assert revised.revise({slots[7]: np.diag([2, 1])}) is revised and calls == []
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
