"""Counterexample to extending source-anchor reconciliation to unseen starts.

Not a native backbone simulation. Uses the native comparison rule on exact
traces from a separate, four-coordinate, piecewise-linear dynamical system.
"""
from experiments.memory.native_reconciliation import _compare, _ReplayTrace


def trajectory(start, bank):
    state = start
    states, selected = [], set()
    for _ in range(4):
        a, b, c, d = state
        scores = {"A": 3 * a, "B": sum(state)}
        eligible = sorted((scores[key], key) for key in bank if scores[key] > 0)
        if eligible:
            assert len(eligible) == 1 or eligible[-1][0] != eligible[-2][0]
            _, key = eligible[-1]
            selected.add(key)
            if key == "A":
                # A maps e0 -> e1, e2 -> e3, and fixes e1 and e3.
                state = (0, a + b, 0, c + d)
            else:
                # B maps e0/e1/e2 -> e2 and preserves e3.
                state = (0, 0, a + b + c, d)
        assert sum(state) == 100 and min(state) >= 0 and max(state) <= 127
        states.append(bytes(state))
    return _ReplayTrace(tuple(states), True, frozenset(selected), True)


def compare(starts, exclude):
    active = [trajectory(x, ("A", "B")) for x in starts]
    masked = [trajectory(x, tuple(k for k in ("A", "B") if k != exclude)) for x in starts]
    source_only = [trajectory(x, ()) for x in starts]
    return _compare(exclude, active, masked, source_only.__getitem__, (1, 2), 3)


def main():
    anchors = [(100, 0, 0, 0), (0, 0, 100, 0)]
    a = compare(anchors, "A")
    b = compare(anchors, "B")
    assert (a.status, a.improved_checks) == ("qualified", 1)
    assert b.status == "endpoint_disagreement"
    lost = 0
    for w in range(1, 100):
        # Every late start is strictly inside the convex hull of the anchors.
        x = (w, 0, 100 - w, 0)
        on = trajectory(x, ("A", "B"))
        off = trajectory(x, ("B",))
        restored = trajectory(x, ("A", "B"))
        assert restored == on
        assert off.states[1][3] == 0
        assert on.states[1][3] == (100 - w if w >= 34 else 0)
        lost += on.states[1][3] != off.states[1][3]
        if w >= 34:
            assert compare([*anchors, x], "A").status == "endpoint_disagreement"
    assert lost == 66
    print(f"A: {a.status}, {a.improved_checks} strict gain; B: {b.status}")
    print(f"Lost held readouts: {lost}/99; same-pin restoration: 99/99")
    print("Each of the 66 affected starts vetoes A if included as an anchor.")


if __name__ == "__main__":
    main()
