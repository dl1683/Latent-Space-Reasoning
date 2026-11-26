"""Exact-integer control for deferred state-dependent third/fourth operations.
Standalone, no model or source encoder. Run with Python 3.10+.
"""
from dataclasses import dataclass
from collections import Counter
from random import Random


@dataclass(frozen=True)
class Cell:
    lo: tuple
    hi: tuple
    witness: tuple
    left: object = None
    right: object = None


def merge(left, right):
    if left is None and right is None:
        return None
    if left is None or right is None:
        n = left or right
        return Cell(n.lo, n.hi, n.witness, left, right)
    return Cell(tuple(map(min, zip(left.lo, right.lo))),
                tuple(map(max, zip(left.hi, right.hi))), left.witness, left, right)


def build(rows):
    if len(rows) == 1:
        return Cell(rows[0], rows[0], rows[0])
    mid = len(rows) // 2
    return merge(build(rows[:mid]), build(rows[mid:]))


def edit(node, start, stop, index, value):
    if stop - start == 1:
        return Cell(value, value, value)
    mid = (start + stop) // 2
    if index < mid:
        return merge(edit(node.left, start, mid, index, value), node.right)
    return merge(node.left, edit(node.right, mid, stop, index, value))


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def interval(a, cell):
    pairs = [(v*l, v*u) for v, l, u in zip(a, cell.lo, cell.hi)]
    return sum(min(p) for p in pairs), sum(max(p) for p in pairs)


# Operation 3: an incoming-state gate and an affine output. Operation 4:
# fixed review terminal accepts the output at or above 1. The accept path
# is independent and supplied as a separate known witness.
def eligible(h, query):
    a, low, high, b = query
    return low <= dot(a, h) <= high and dot(b, h) >= 1


def cover(root, query, budget):
    a, low, high, b = query
    stack, checks = [root], 0
    while stack and checks < budget:
        cell = stack.pop()
        checks += 1
        u, v = interval(a, cell)
        p, q = interval(b, cell)
        if v < low or u > high or q < 1:
            continue
        if low <= u and v <= high and p >= 1:
            assert eligible(cell.witness, query)
            return True, checks
        stack.extend(c for c in (cell.right, cell.left) if c is not None)
    return (None if stack else False), checks


class ExactJoin:
    """Deduplicate full hidden states, cache arbitrary late predicates.

    Update previously seen predicates only for the changed state. A new
    predicate is evaluated once per distinct state, with exact witnesses
    recoverable from the retained rows. This is not a timing benchmark.
    """
    def __init__(self, rows):
        self.states = Counter(rows)
        self.cache = {}
        self.evaluations = 0

    def read(self, query):
        if query not in self.cache:
            self.cache[query] = {h for h in self.states if eligible(h, query)}
            self.evaluations += len(self.states)
        return bool(self.cache[query])

    def replace(self, old, new):
        self.states[old] -= 1
        if self.states[old] == 0:
            del self.states[old]
            for matches in self.cache.values():
                matches.discard(old)
        if new not in self.states:
            for query, matches in self.cache.items():
                self.evaluations += 1
                if eligible(new, query):
                    matches.add(new)
        self.states[new] += 1


def outcome(review, strong=True):
    if review is None:
        return 'unresolved'
    return ('conflict' if review else 'accept') if strong else ('review' if review else 'absent')


def main():
    # 252 fixed distractors at the four corners, two fixed duplicates,
    # two weak relays whose correlation is revised. Same convex hull,
    # complete coordinate marginals and terminal catalogue on both pins.
    corners = [(-4,-4), (-4,4), (4,-4), (4,4)]
    old = [(0,-2), (1,2)] + corners*63 + corners[:2]
    new = [(0,2), (1,-2)] + old[2:]
    assert len(old) == len(new) == 256
    assert all(Counter(h[j] for h in old) == Counter(h[j] for h in new) for j in (0,1))
    assert set(corners) <= set(old) and set(corners) <= set(new)
    assert all(max(map(abs,h)) <= 4 for h in old+new)
    opening = build(old)  # no late operation supplied to construction
    corrected = edit(edit(opening,0,256,0,new[0]),0,256,1,new[1])
    restored = edit(edit(corrected,0,256,0,old[0]),0,256,1,old[1])
    assert opening.lo == corrected.lo and opening.hi == corrected.hi
    held = ((1,0),0,0,(2,1))
    join = ExactJoin(old)
    for name, root, rows, strong in [('opening',opening,old,True),
                                     ('corrected',corrected,new,True),
                                     ('strong withdrawn',corrected,new,False),
                                     ('restored',restored,old,True),
                                     ('historical',opening,old,True)]:
        if name == 'corrected':
            for a,b in zip(old[:2],new[:2]): join.replace(a,b)
        elif name == 'restored':
            for a,b in zip(new[:2],old[:2]): join.replace(a,b)
        truth = any(eligible(h,held) for h in rows)
        full, checks = cover(root,held,511)
        assert full == join.read(held) == truth
        small, _ = cover(root,held,1)
        assert small is None
        # Also give the cover the same state factoring as the join.
        # This rebuild is a diagnostic, not a charged incremental implementation.
        factored, factored_checks = cover(build(sorted(set(rows))),held,511)
        assert factored == truth
        print(name, outcome(full,strong), checks, outcome(small,strong),
              join.evaluations, factored_checks)
    # Independent held-query family. Exact full traversal must agree with
    # an exhaustive row oracle; every finite-budget conclusion must be sound.
    rng = Random(9081)
    budgets = (1, 4, 16, 64, 511)
    totals = Counter()
    for _ in range(200):
        a = tuple(rng.randint(-3,3) for _ in range(2))
        b = tuple(rng.randint(-3,3) for _ in range(2))
        lo,hi = sorted((rng.randint(-10,10),rng.randint(-10,10)))
        query = (a,lo,hi,b)
        for root,rows in ((opening,old),(corrected,new),(restored,old)):
            truth = any(eligible(h,query) for h in rows)
            for budget in budgets:
                result, work = cover(root,query,budget)
                assert work <= budget
                assert result is None or result == truth
                assert budget != 511 or result == truth
                totals[budget] += result is None
    print('unresolved of 600 per budget:', dict(totals))


if __name__ == '__main__':
    main()
