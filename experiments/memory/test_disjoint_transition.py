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

    def watch(assertions):
        calls.append(frozenset(assertions))
        return real_compile(assertions)

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
