"""Standalone score-grid control; not integrated into the native reader."""
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from functools import lru_cache
from random import Random

GRID = 127 ** 2
PLACES = 24
SCALE = 10 ** PLACES
# Exact decimal temperature for this control, not a floating-point runtime policy.
TAU = Decimal(GRID) * Decimal("0.04")


@lru_cache(maxsize=8192)
def weight_bounds(score, peak):
    """Enclose SCALE * exp((score-peak)/TAU) by two integers."""
    assert type(score) is type(peak) is int and score <= peak
    if score == peak:
        return SCALE, SCALE
    with localcontext() as ctx:
        ctx.prec = 50
        ctx.rounding = ROUND_FLOOR
        xlo = Decimal(score - peak) / TAU
        ctx.rounding = ROUND_CEILING
        xhi = Decimal(score - peak) / TAU
        # Decimal.exp is correctly rounded (half even). Widen one Decimal ulp.
        lo = xlo.exp().next_minus().scaleb(PLACES)
        hi = xhi.exp().next_plus().scaleb(PLACES)
        return (max(0, int(lo.to_integral_value(rounding=ROUND_FLOOR))),
                int(hi.to_integral_value(rounding=ROUND_CEILING)))


def mass_bounds(scores, peak):
    lo = hi = 0
    for score in scores:
        a, b = weight_bounds(score, peak)
        lo += a
        hi += b
    return lo, hi


def revise_tail(bounds, retired, added, peak, cutoff):
    """Only call after selection-survival checks, with complete old/new deltas."""
    retired, added = tuple(retired), tuple(added)
    if any(score >= cutoff for score in retired + added):
        raise ValueError("selected frontier may change; rebuild")
    a, b = mass_bounds(retired, peak)
    c, d = mass_bounds(added, peak)
    lo, hi = bounds[0] - a + c, bounds[1] - b + d
    if not 0 <= lo <= hi:
        raise ValueError("invalid tail ledger")
    return lo, hi


def admit(selected, tail, p, q):
    """Three-valued certificate for global retained softmax mass >= p/q."""
    assert 0 < p < q and selected[0] > 0
    if (q-p)*selected[0] >= p*tail[1]:
        return True
    if (q-p)*selected[1] < p*tail[0]:
        return False
    return None


@lru_cache(maxsize=8192)
def oracle_weight(score, peak):
    with localcontext() as ctx:
        ctx.prec = 100
        return ((Decimal(score - peak) / TAU).exp() * SCALE)


def check():
    peak = 127 * 100
    cutoff = peak
    selected = mass_bounds([peak], peak)
    # One-dimensional packed query 127; keys 100, 99 and 0.
    tail = [0] * 12087
    initial = mass_bounds(tail, peak)
    changed = revise_tail(initial, [0], [127*99], peak, cutoff)
    restored = revise_tail(changed, [127*99], [0], peak, cutoff)
    assert restored == initial
    assert [admit(selected, x, 9, 10) for x in (initial, changed, restored)] == [True, False, True]
    def mass(t):
        return SCALE / (SCALE + (t[0] + t[1])/2)
    print("retained mass:", *(f"{mass(t):.9f}" for t in (initial, changed, restored)))
    rng = Random(20261126)
    edits = checks = 0
    with localcontext() as ctx:
        ctx.prec = 100
        for _ in range(128):
            # A fixed top-two inclusive frontier, with all other keys below cutoff.
            chosen = [127*100, 127*98]
            cut = chosen[-1]
            sel = mass_bounds(chosen, peak)
            exact_sel = sum(oracle_weight(s, peak) for s in chosen)
            rows = [127*rng.randrange(-127, 98) for _ in range(64)]
            ledger = mass_bounds(rows, peak)
            for _ in range(32):
                old = [rows.pop(rng.randrange(len(rows))) for _ in range(min(len(rows), rng.randrange(4)))]
                new = [127*rng.randrange(-127, 98) for _ in range(rng.randrange(4))]
                rows.extend(new)
                ledger = revise_tail(ledger, old, new, peak, cut)
                assert ledger == mass_bounds(rows, peak)
                exact = sum(oracle_weight(s, peak) for s in rows)
                assert ledger[0] <= exact <= ledger[1]
                edits += 1
                for p,q in ((1,2),(9,10),(99,100)):
                    result = admit(sel, ledger, p, q)
                    if result is not None:
                        assert result == ((q-p)*exact_sel >= p*exact)
                    checks += 1
    assert admit((100,101), (100,101), 1, 2) is None
    try:
        revise_tail(initial, [0], [peak], peak, cutoff)
    except ValueError:
        pass
    else:
        raise AssertionError("cutoff entry was not refused")
    print(f"{edits} integer delta/full-rebuild checks; {checks} threshold/oracle checks")


if __name__ == "__main__":
    check()
