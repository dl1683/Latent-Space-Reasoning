"""In-process admission of sparse hierarchy order changes against a replay."""

from dataclasses import dataclass
from itertools import islice
from math import isfinite
from operator import index
from threading import Lock
from types import MappingProxyType
from typing import NamedTuple

from numpy.linalg import LinAlgError

from experiments.memory.relative_hierarchy import RelativeHierarchySnapshot
from experiments.memory.relative_ordering import propose_orders


class RelativeReplayWork(NamedTuple):
    rebuilt_buckets: int
    recovered_ports: int
    peak_factor_entries: int


@dataclass(frozen=True, eq=False)
class RelativeMaintenanceRound:
    snapshot: RelativeHierarchySnapshot
    baseline_work: RelativeReplayWork
    selected_work: RelativeReplayWork
    considered: int
    conversion_buckets: int


def _factor_entries(snapshot):
    if snapshot._elimination is None:
        return 0
    return sum(step.readout.size + (step.remainder.rows.size if step.remainder is not None else 0)
               for step in snapshot._elimination.steps.values())


def _replay(snapshot, workload):
    answers = []
    rebuilt = recovered = 0
    peak = _factor_entries(snapshot)
    for edits, queries in workload:
        revised = snapshot._owner.revise(snapshot, edits)
        if revised is not snapshot:
            rebuilt += len(revised.rebuilt_buckets)
            peak = max(peak, _factor_entries(revised))
        values, ports = revised._read_batch(queries)
        recovered += ports
        answers.extend(values)
        snapshot = revised
    return RelativeReplayWork(rebuilt, recovered, peak), tuple(answers)


def _agrees(expected, actual, tolerance):
    return len(expected) == len(actual) and all(
        before is after if before is None or after is None else abs(before - after) <= tolerance
        for before, after in zip(expected, actual)
    )


class RelativeHierarchyWorkspace:
    """Publish source edits or guarded order changes to one in-process tip."""

    def __init__(self, snapshot):
        if snapshot.elimination_order is None:
            raise ValueError("order maintenance needs a sparse hierarchy snapshot")
        self._current = snapshot
        self._lock = Lock()

    def snapshot(self):
        with self._lock:
            return self._current

    def _require_current(self, base):
        if self._current is not base:
            raise RuntimeError("relative hierarchy workspace revision is stale")

    def _publish(self, base, revised):
        with self._lock:
            self._require_current(base)
            self._current = revised

    def revise(self, base, changes):
        """Apply real source changes to the current layout."""
        with self._lock:
            self._require_current(base)
        revised = base._owner.revise(base, changes)
        self._publish(base, revised)
        return revised

    def maintain(self, base, orders, workload, *, max_candidates, protected=(), tolerance=1e-10):
        """Replay a finite candidate round and publish only a better layout."""
        return self._maintain(base, orders, workload, max_candidates, protected, tolerance)

    def maintain_generated(self, base, workload, *, max_candidates, max_symbolic_ports, protected=(),
                           tolerance=1e-10):
        """Propose bounded symbolic orders and charge conversion in the admission score."""
        max_ports = index(max_symbolic_ports)
        if max_ports < 1:
            raise ValueError("automatic order maintenance needs a positive symbolic port cap")
        return self._maintain(base, None, workload, max_candidates, protected, tolerance, max_ports=max_ports)

    def _maintain(self, base, orders, workload, max_candidates, protected, tolerance, *, max_ports=None):
        max_candidates, tolerance = index(max_candidates), float(tolerance)
        if max_candidates < 1 or not isfinite(tolerance) or tolerance < 0:
            raise ValueError("order maintenance needs a positive candidate cap and finite tolerance")
        with self._lock:
            self._require_current(base)
        protected = tuple(protected)
        workload = tuple((MappingProxyType(dict(edits)), tuple(queries)) for edits, queries in workload)
        protected_values = base.read_many(protected)
        baseline, expected = _replay(base, workload)
        if max_ports is not None:
            orders = propose_orders(base, workload, max_ports)
        best, best_work, considered = base, baseline, 0
        best_score = baseline
        seen = {base.elimination_order}
        for order in islice(orders, max_candidates):
            considered += 1
            order = tuple(order)
            if order in seen:
                continue
            seen.add(order)
            try:
                candidate = base._owner.reorder(base, order)
                if not _agrees(protected_values, candidate.read_many(protected), tolerance):
                    continue
                work, answers = _replay(candidate, workload)
            except (FloatingPointError, ValueError, LinAlgError):
                continue
            score = work
            if max_ports is not None:
                score = work._replace(rebuilt_buckets=work.rebuilt_buckets + len(candidate.rebuilt_buckets))
            if (score < best_score
                    and all(value <= limit for value, limit in zip(score, baseline))
                    and _agrees(expected, answers, tolerance)):
                best, best_work, best_score = candidate, work, score
        self._publish(base, best)
        conversion = len(best.rebuilt_buckets) if best is not base else 0
        return RelativeMaintenanceRound(best, baseline, best_work, considered, conversion)
