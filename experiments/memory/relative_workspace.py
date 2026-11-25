"""In-process admission of sparse hierarchy changes against a replay."""

from collections import Counter, deque
from collections.abc import Mapping
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


class RelativeStructureWork(NamedTuple):
    factor_units: int
    read_units: int
    peak_operator_entries: int

    @property
    def units(self):
        return self.factor_units + self.read_units


@dataclass(frozen=True, eq=False)
class RelativeStructureRound:
    snapshot: RelativeHierarchySnapshot
    change: tuple[str, tuple[int, ...]] | None
    baseline_work: RelativeStructureWork
    selected_work: RelativeStructureWork
    considered: int
    conversion_units: int


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


def _check_structure_forecast(base, proposal, workload, *, max_operator_entries, tolerance):
    baseline, expected = _structure_replay(base, workload)
    if all(value is None for value in expected):
        return "check_unsupported", baseline, None
    try:
        candidate, actual = _structure_replay(
            proposal.snapshot, workload, conversion=proposal.conversion_units, entry_cap=max_operator_entries,
        )
    except (FloatingPointError, ValueError, LinAlgError):
        return "check_refused", baseline, None
    if not _agrees(expected, actual, tolerance):
        return "check_drift", baseline, candidate
    if candidate.units >= baseline.units:
        return "check_no_gain", baseline, candidate
    return "published", baseline, candidate


def _operator_inventory(snapshot):
    operators, pending = set(), list(snapshot._regions)
    while pending:
        region = pending.pop()
        if region in operators:
            continue
        operators.add(region)
        if region.fusion is not None:
            pending.extend(region.fusion.children)
    arrays = {}

    def retain(array):
        arrays[id(array)] = array.size

    for region in operators:
        retain(region.factor)
        retain(region.readout)
    if snapshot._values is not None:
        retain(snapshot._values)
    if snapshot._elimination is not None:
        for step in snapshot._elimination.steps.values():
            retain(step.readout)
            for factor in step.inputs:
                retain(factor.rows)
            if step.remainder is not None:
                retain(step.remainder.rows)
    return operators, sum(arrays.values())


def _qr_units(rows, columns):
    return rows * columns * min(rows, columns)


def _factor_units(previous, current, snapshot):
    units = 0
    for region in current - previous:
        if region.fusion is None:
            variables = len(region.positions)
            rows = variables + len(region._sources)
            interior = region.interior_size
        else:
            children = region.fusion.children
            variables = sum(len(child.ports) for child in children)
            rows = variables + len(region.fusion.bridges)
            interior = variables - len(region.ports)
            if not region.lazy_readout:
                units += sum(child.interior_size * (len(child.ports) + 1) * (len(region.ports) + 1)
                             for child in children)
        units += _qr_units(rows, variables + 1) + interior**2 * (variables + 1)
    for port in snapshot.rebuilt_buckets:
        inputs = snapshot._elimination.steps[port].inputs
        variables = {cell for factor in inputs for cell in factor.cells}
        units += _qr_units(sum(len(factor.rows) for factor in inputs), len(variables) + 1)
    return units


def _structure_replay(snapshot, workload, *, conversion=0, entry_cap=None):
    operators, peak = _operator_inventory(snapshot)
    answers, read_work = [], [0]
    factor_work = conversion
    for edits, queries in workload:
        revised = snapshot._owner.revise(snapshot, edits)
        if revised is not snapshot:
            current, entries = _operator_inventory(revised)
            peak = max(peak, entries)
            if entry_cap is not None and peak > entry_cap:
                raise ValueError("structure replay exceeds the operator entry cap")
            factor_work += _factor_units(operators, current, revised)
            operators = current
        values, _ = revised._read_batch(queries, work=read_work)
        answers.extend(values)
        snapshot = revised
    return RelativeStructureWork(factor_work, read_work[0], peak), tuple(answers)


def _bounded_structure_inputs(workload, protected, max_batches, max_items):
    def consume(items, freeze):
        nonlocal max_items
        values = []
        for item in islice(items, max_items + 1):
            if len(values) == max_items:
                raise ValueError("structure maintenance workload exceeds the input item cap")
            values.append(freeze(item))
        max_items -= len(values)
        return tuple(values)

    guards = consume(protected, index)
    frozen = []
    for number, batch in enumerate(islice(workload, max_batches + 1)):
        if number == max_batches:
            raise ValueError("structure maintenance workload is outside the replay batch cap")
        edits, queries = batch
        rows = consume(edits.items() if isinstance(edits, Mapping) else edits, lambda pair: tuple(islice(pair, 3)))
        frozen.append((MappingProxyType(dict(rows)), consume(queries, index)))
    if not frozen:
        raise ValueError("structure maintenance workload is outside the replay batch cap")
    return tuple(frozen), guards


def _structure_proposals(base, workload, max_symbolic_ports):
    membership = base.membership
    adjacent = Counter(tuple(sorted((membership[row.source], membership[row.target])))
                       for row in base._bridges.values())
    merges = (("merge_lazy", pair) for pair in sorted(adjacent, key=lambda pair: (-adjacent[pair], pair)))
    splits = (("split_region", (label,)) for label, region in enumerate(base._regions)
              if region.fusion is not None)
    orders = (("layout", order) for order in propose_orders(base, workload, max_symbolic_ports))
    streams = deque((merges, splits, orders))
    while streams:
        stream = streams.popleft()
        proposal = next(stream, None)
        if proposal is not None:
            yield proposal
            streams.append(stream)


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

    def maintain_structure(self, base, workload, *, max_candidates, max_replay_batches, max_operator_entries,
                           max_symbolic_ports, max_input_items, protected=(), tolerance=1e-10):
        """Publish a guarded structure or ordering change if it lowers work after conversion."""
        max_candidates, max_replay_batches, max_operator_entries, max_symbolic_ports, max_input_items = map(
            index, (max_candidates, max_replay_batches, max_operator_entries, max_symbolic_ports, max_input_items),
        )
        tolerance = float(tolerance)
        if (min(max_candidates, max_replay_batches, max_operator_entries, max_symbolic_ports, max_input_items) < 1
                or not isfinite(tolerance) or tolerance < 0):
            raise ValueError("structure maintenance needs positive caps and finite nonnegative tolerance")
        with self._lock:
            self._require_current(base)
        workload, protected = _bounded_structure_inputs(workload, protected, max_replay_batches, max_input_items)
        expected_protected = base.read_many(protected)
        baseline, expected = _structure_replay(base, workload)
        if all(value is None for value in (*expected_protected, *expected)):
            self._publish(base, base)
            return RelativeStructureRound(base, None, baseline, baseline, 0, 0)
        prior_operators, _ = _operator_inventory(base)
        best, selected, best_change, best_conversion, considered = base, baseline, None, 0, 0
        proposals = _structure_proposals(base, workload, max_symbolic_ports)
        for change in islice(proposals, max_candidates):
            considered += 1
            kind, labels = change
            try:
                if kind == "merge_lazy":
                    candidate = base._owner.merge_regions(base, *labels, retain_children=True, lazy_readout=True)
                elif kind == "split_region":
                    candidate = base._owner.split_region(base, *labels)
                else:
                    candidate = base._owner.reorder(base, labels)
                if candidate is base:
                    continue
                operators, entries = _operator_inventory(candidate)
                if entries > max_operator_entries:
                    continue
                if not _agrees(expected_protected, candidate.read_many(protected), tolerance):
                    continue
                conversion = _factor_units(prior_operators, operators, candidate)
                work, answers = _structure_replay(candidate, workload, conversion=conversion,
                                                  entry_cap=max_operator_entries)
            except (FloatingPointError, ValueError, LinAlgError):
                continue
            if work.units < selected.units and _agrees(expected, answers, tolerance):
                best, selected, best_change, best_conversion = candidate, work, change, conversion
        self._publish(base, best)
        return RelativeStructureRound(best, best_change, baseline, selected, considered, best_conversion)

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
