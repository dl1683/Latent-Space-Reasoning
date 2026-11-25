from collections import defaultdict

import numpy as np
import pytest

from experiments.memory import disjoint_transition
from experiments.memory.disjoint_transition import DisjointTransitionMemory, DisjointTransitionWitness, propagate_disjoint


def _joined_reference(width, sources, seed):
    left, right = defaultdict(list), defaultdict(list)
    for origin, rows in sources.items():
        for hop, start, end, grade in rows or ():
            if hop == 0:
                left[end].append((start, origin, grade))
            else:
                right[start].append((end, origin, grade))
    grades = np.full(width, -np.inf)
    witnesses = defaultdict(set)
    for middle, incoming in left.items():
        for start, left_origin, left_grade in incoming:
            for end, right_origin, right_grade in right.get(middle, ()):
                if left_origin == right_origin or seed[start] == -np.inf:
                    continue
                grade = min(seed[start], left_grade, right_grade)
                if grade > grades[end]:
                    grades[end] = grade
                    witnesses[end].clear()
                if grade == grades[end]:
                    witnesses[end].add(DisjointTransitionWitness(float(grade), start, middle, (left_origin, right_origin)))
    return grades, witnesses


def test_late_propagation_preserves_distinct_origin_alternatives_across_revisions(monkeypatch):
    sources = {
        "shared": [(0, 0, 2, .95), (1, 2, 4, .96), (0, 0, 2, .2)],
        "second": [(0, 0, 2, .75)],
        "third": [(0, 0, 2, .5)],
        "alternate-right": [(1, 2, 4, .6)],
        "unrelated-left": [(0, 1, 3, .9)],
        "unrelated-right": [(1, 3, 6, 0.)],
    }
    base = DisjointTransitionMemory.prepare(8, sources)
    seed = [.8, .9, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
    np.testing.assert_array_equal(propagate_disjoint(seed), seed)
    expected = [-np.inf, -np.inf, -np.inf, -np.inf, .75, -np.inf, 0., -np.inf]
    np.testing.assert_array_equal(propagate_disjoint(seed, memory=base), expected)
    assert base.prove(seed, 4) == DisjointTransitionWitness(.75, 0, 2, ("second", "shared"))
    assert base.prove(seed, 6).strength == 0.
    assert base.prove(seed, 5) is None

    real_compile = disjoint_transition._compile_middle
    calls = []

    def watch(assertions, order):
        calls.append(frozenset(assertions))
        return real_compile(assertions, order)

    monkeypatch.setattr(disjoint_transition, "_compile_middle", watch)
    no_second = base.revise({"second": None})
    assert no_second.prove(seed, 4) == DisjointTransitionWitness(.6, 0, 2, ("shared", "alternate-right"))
    no_alternate = no_second.revise({"alternate-right": None})
    assert no_alternate.prove(seed, 4) == DisjointTransitionWitness(.5, 0, 2, ("third", "shared"))
    only_shared = no_alternate.revise({"third": None})
    assert only_shared.prove(seed, 4) is None
    assert propagate_disjoint(seed, memory=only_shared)[4] == -np.inf
    restored = only_shared.revise({"second": sources["second"]})
    np.testing.assert_array_equal(propagate_disjoint(seed, memory=restored), expected)
    assert all(pin._middles[3] is base._middles[3] for pin in (no_second, no_alternate, only_shared, restored))
    assert len(calls) == 4 and all("unrelated-left" not in origins for origins in calls)
    assert restored.revise({"second": [(0, 0, 2, .1), (0, 0, 2, .75)]}) is restored
    assert len(calls) == 4
    np.testing.assert_array_equal(propagate_disjoint(seed, memory=base), expected)

    moved = restored.revise({"second": [(0, 0, 7, .7)], "new-finish": [(1, 7, 4, .8)]})
    assert moved.prove(seed, 4) == DisjointTransitionWitness(.7, 0, 7, ("second", "new-finish"))
    assert moved._middles[2] is not restored._middles[2]
    assert moved._middles[3] is base._middles[3]
    assert len(calls) == 6 and all("unrelated-left" not in origins for origins in calls)
    np.testing.assert_array_equal(propagate_disjoint(seed, memory=restored), expected)


def test_factorized_transfer_matches_an_exhaustive_indexed_join_across_forks():
    rng = np.random.default_rng(620)
    width = 6
    sources = {}
    memory = DisjointTransitionMemory.prepare(width, sources)

    def new_rows():
        return [(*map(int, rng.integers([2, width, width])), float(rng.choice([0., .25, .5, .75, 1.])))
                for _ in range(12)]

    snapshots = [(memory, sources)]
    for _ in range(36):
        branch, current = snapshots[int(rng.integers(len(snapshots)))]
        changes = {f"source-{int(origin)}": new_rows() if rng.random() > .2 else None
                   for origin in rng.choice(12, size=5, replace=False)}
        snapshot = branch.revise(changes)
        updated = current | changes
        snapshots.append((snapshot, updated))
        for _ in range(5):
            seed = rng.choice([-np.inf, 0., .25, .5, .75, 1.], size=width)
            expected, witnesses = _joined_reference(width, updated, seed)
            np.testing.assert_array_equal(propagate_disjoint(seed, memory=snapshot), expected)
            for end in range(width):
                witness = snapshot.prove(seed, end)
                if expected[end] == -np.inf:
                    assert witness is None
                else:
                    assert witness in witnesses[end]


def test_factorized_storage_avoids_endpoint_pairs_and_invalid_batches_preserve_the_parent():
    width = 24
    sources = {f"origin-{i:02}": [(0, start, 0, 1. - i / 100) for start in range(width)]
               for i in range(12)}
    sources["finish"] = [(1, 0, end, 1.) for end in range(width)]
    snapshot = DisjointTransitionMemory.prepare(width, sources)
    block = snapshot._middles[0]
    assert (block.incoming.grades.size, block.outgoing.grades.size, len(block.origins)) == (48, 24, 3)
    seed = np.ones(width)
    np.testing.assert_array_equal(propagate_disjoint(seed, memory=snapshot), seed)
    sources["origin-00"].clear()
    with pytest.raises(ValueError, match="in-range"):
        snapshot.revise({"finish": None, "invalid": [(0, 0, width, 1.)]})
    np.testing.assert_array_equal(propagate_disjoint(seed, memory=snapshot), seed)
    with pytest.raises(ValueError, match="one grade"):
        propagate_disjoint(np.full(width, -1.), memory=snapshot)
    for invalid in (np.nan, np.inf):
        malformed = seed.copy()
        malformed[0] = invalid
        with pytest.raises(ValueError):
            propagate_disjoint(malformed, memory=snapshot)
        with pytest.raises(ValueError):
            snapshot.prove(malformed, 0)


def test_sparse_seed_and_destination_skip_unrelated_middle_blocks(monkeypatch):
    count = 64
    width = 3 * count
    sources = {
        **{f"in-{i}": [(0, i, count + i, .75)] for i in range(count)},
        **{f"out-{i}": [(1, count + i, 2 * count + i, .5)] for i in range(count)},
    }
    bank = DisjointTransitionMemory.prepare(width, sources)
    activate = disjoint_transition._activation
    evaluated = []

    def watch(middle, seed):
        evaluated.append(middle.origins)
        return activate(middle, seed)

    monkeypatch.setattr(disjoint_transition, "_activation", watch)
    sparse = np.full(width, -np.inf)
    sparse[7] = -0.
    expected, witnesses = _joined_reference(width, sources, sparse)
    np.testing.assert_array_equal(propagate_disjoint(sparse, memory=bank), expected)
    assert evaluated == [("in-7", "out-7")]
    assert expected[2 * count + 7] == 0.

    evaluated.clear()
    assert bank.prove(sparse, 2 * count + 7) in witnesses[2 * count + 7]
    assert bank.prove(sparse, 0) is None
    assert evaluated == [("in-7", "out-7")]

    evaluated.clear()
    dense = np.ones(width)
    expected, witnesses = _joined_reference(width, sources, dense)
    np.testing.assert_array_equal(propagate_disjoint(dense, memory=bank), expected)
    assert len(evaluated) == count
    evaluated.clear()
    assert bank.prove(dense, 2 * count + 9) in witnesses[2 * count + 9]
    assert evaluated == [("in-9", "out-9")]
    evaluated.clear()
    dense[9] = -np.inf
    assert bank.prove(dense, 2 * count + 9) is None
    assert evaluated == [("in-9", "out-9")]

    hub_source = {"hub": [(0, 7, count + i, .25) for i in range(count)]}
    hub = bank.revise(hub_source)
    expected, _ = _joined_reference(width, sources | hub_source, sparse)
    evaluated.clear()
    assert hub._candidate_middles(sparse) is hub._middles
    np.testing.assert_array_equal(propagate_disjoint(sparse, memory=hub), expected)
    assert len(evaluated) == count


def test_converging_entry_postings_charge_references_before_fallback(monkeypatch):
    sources = {
        "converging": [(0, entry, middle, .8) for entry in (0, 1, 2) for middle in (6, 7)],
        "other": [(0, 3, middle, .7) for middle in (8, 9, 10, 11)],
        "finish": [(1, middle, 12, .6) for middle in range(6, 12)],
    }
    bank = DisjointTransitionMemory.prepare(13, sources)
    seed = np.full(13, -np.inf)
    seed[:3] = 1.
    assert all(bank._entry_middles[entry] == frozenset({6, 7}) for entry in range(3))
    assert bank._candidate_middles(seed) is bank._middles

    activate = disjoint_transition._activation
    evaluated = []

    def watch(middle, grades):
        evaluated.append(middle)
        return activate(middle, grades)

    monkeypatch.setattr(disjoint_transition, "_activation", watch)
    expected, witnesses = _joined_reference(13, sources, seed)
    np.testing.assert_array_equal(propagate_disjoint(seed, memory=bank), expected)
    assert len(evaluated) == 6
    evaluated.clear()
    assert bank.prove(seed, 12) in witnesses[12]
    assert len(evaluated) == 6


def test_index_revisions_share_unchanged_postings_and_follow_completion_and_withdrawal():
    bank = DisjointTransitionMemory.prepare(8, {
        "front": [(0, 0, 2, .75), (0, 0, 3, .6)],
        "last": [(1, 2, 4, .9)], "unrelated-in": [(0, 1, 5, .8)],
        "unrelated-out": [(1, 5, 6, .8)],
    })
    assert bank._entry_middles[0] == frozenset({2})
    assert bank._destination_middles[4] == frozenset({2})
    graded = bank.revise({"last": [(1, 2, 4, .5)]})
    assert graded._entry_middles is bank._entry_middles
    assert graded._destination_middles is bank._destination_middles

    moved = bank.revise({"last": [(1, 3, 8, .9)]}, width=9)
    assert moved._entry_middles[0] == frozenset({3})
    assert 4 not in moved._destination_middles
    assert moved._destination_middles[8] == frozenset({3})
    assert moved._entry_middles[1] is bank._entry_middles[1]
    assert moved._destination_middles[6] is bank._destination_middles[6]
    removed = moved.revise({"front": None})
    seed = np.full(9, -np.inf)
    seed[0] = 1.
    assert moved.prove(seed, 8) == DisjointTransitionWitness(.6, 0, 3, ("front", "last"))
    assert removed.prove(seed, 8) is None and 0 not in removed._entry_middles
    assert 8 not in removed._destination_middles
    assert bank.prove(seed[:8], 4) == DisjointTransitionWitness(.75, 0, 2, ("front", "last"))


def test_sparse_and_scan_paths_keep_the_order_of_an_earlier_incomplete_middle():
    bank = DisjointTransitionMemory.prepare(9, {
        "early": [(0, 0, 7, .75)], "late": [(0, 0, 2, .75)], "late-out": [(1, 2, 8, .9)],
        "other": [(0, 1, 5, .25)], "other-out": [(1, 5, 8, .9)],
    }).revise({"early-out": [(1, 7, 8, .9)]})
    sparse = np.full(9, -np.inf)
    sparse[0] = 1.
    dense = np.ones(9)
    assert list(bank._candidate_middles(sparse, 8)) == [7, 2]
    assert list(bank._candidate_middles(dense, 8)) == [7, 2, 5]
    assert bank.prove(sparse, 8).middle == 2
    equal_key = dict.fromkeys((2, 5, 7), 0).__getitem__
    expected = DisjointTransitionWitness(.75, 0, 7, ("early", "early-out"))
    assert bank.prove(sparse, 8, middle_key=equal_key) == expected
    assert bank.prove(dense, 8, middle_key=equal_key) == expected
