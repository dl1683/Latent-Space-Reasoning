"""Visible algebraic control; no pretrained model or held task is run."""
from fractions import Fraction
from types import SimpleNamespace as NS

import numpy as np

from experiments.memory.selective_continuations import RetainedSelector


def rank(rows):
    """Exact rational elimination; row vectors are source-change probes."""
    a = [[Fraction(int(v)) for v in row] for row in rows]
    if not a:
        return 0
    k = 0
    for j in range(len(a[0])):
        pivot = next((i for i in range(k, len(a)) if a[i][j]), None)
        if pivot is None:
            continue
        a[k], a[pivot] = a[pivot], a[k]
        p = a[k][j]
        a[k] = [v / p for v in a[k]]
        for i in range(k + 1, len(a)):
            p = a[i][j]
            if p:
                a[i] = [u - p * v for u, v in zip(a[i], a[k])]
        k += 1
        if k == len(a):
            break
    return k


def run(worlds=128, d=8):
    rng = np.random.default_rng(20261126)
    selector = RetainedSelector(None, top_k=1, tie_limit=2, temperature=0.1, score_margin=0)
    good, bad, seed = (NS(source_id=name) for name in ("good", "bad", "seed"))
    # Signed dot-product winner is independent of the candidate's payload.
    bank = NS(spans=(good, bad), keys=np.array([[70, 0], [100, 0]], dtype=np.int8))
    frontier = NS(receipt=NS(status="ready"), query=np.array([100, 0], dtype=np.int8).tobytes())
    counts = dict(worlds=0, wrong_proposals=0, influence_and_removal=0,
                  nondiscriminating_checks=0, decisive_checks=0,
                  span_predictions=0, outside_span_disagreements=0,
                  stale_validation_checks=0, rank_checks=0)
    for _ in range(worlds):
        # Integer unimodular basis with mixed coordinates, not only unit vectors.
        basis = np.eye(d, dtype=np.int64)[rng.permutation(d)]
        for j in range(1, d):
            basis[j] += int(rng.choice([-1, 1])) * basis[j - 1]
        normal = np.rint(np.linalg.inv(basis)[:, -1]).astype(np.int64)
        assert np.array_equal(basis @ normal, np.eye(d, dtype=np.int64)[:, -1])
        truth = np.eye(d, dtype=np.int64)[rng.permutation(d)]
        u = np.zeros(d, dtype=np.int64)
        u[int(rng.integers(d))] = int(rng.choice([-1, 1])) * 2
        decoy = truth + np.outer(u, normal)
        proposal = selector.discover_parent(seed, (frontier,), bank)
        assert proposal is bad
        counts["wrong_proposals"] += 1
        # Both candidates change the output relative to the no-parent arm,
        # while the address and original source-only trajectory stay fixed.
        anchor = basis[0]
        assert np.any(truth @ anchor) and np.any(decoy @ anchor)
        assert np.array_equal(truth @ anchor, decoy @ anchor)
        counts["influence_and_removal"] += 1
        seen = list(basis[:-1])
        for probe in seen:
            assert np.array_equal(truth @ probe, decoy @ probe)
            counts["nondiscriminating_checks"] += 1
        # More checks are not more independent checks.  On the observed
        # span, reconstruct a consequence from independent probe outcomes.
        observed = np.stack([truth @ probe for probe in seen])
        for _ in range(16):
            c = rng.integers(-3, 4, size=d - 1)
            x = c @ basis[:-1]
            assert np.array_equal(truth @ x, c @ observed)
            assert np.array_equal(decoy @ x, c @ observed)
            counts["span_predictions"] += 1
            x = x + int(rng.choice([-1, 1])) * basis[-1]
            assert not np.array_equal(truth @ x, decoy @ x)
            counts["outside_span_disagreements"] += 1
        for k in range(d):
            assert rank(basis[:k]) == k
            counts["rank_checks"] += 1
        assert rank(basis) == d and rank(seen + [2 * seen[0]]) == d - 1
        counts["rank_checks"] += 2
        # Freeze both candidate predictions before the missing-direction
        # source outcome is disclosed by the independent finite oracle.
        probe = basis[-1]
        predictions = (truth @ probe, decoy @ probe)
        oracle = truth @ probe
        assert np.array_equal(predictions[0], oracle) and not np.array_equal(predictions[1], oracle)
        counts["decisive_checks"] += 1
        # A genuine operator-defining correction can invalidate a previously
        # accepted prediction while leaving all old calibration observations fixed.
        revised = decoy
        assert all(np.array_equal(revised @ x, truth @ x) for x in seen)
        assert not np.array_equal(revised @ probe, oracle)
        counts["stale_validation_checks"] += 1
        counts["worlds"] += 1
    return counts


if __name__ == "__main__":
    print(run())
