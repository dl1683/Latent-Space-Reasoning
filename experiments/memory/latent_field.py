"""Sparse recurrent latent field with local, immutable revisions.

This is an untrained construction. Callers supply latent observations and weight
blocks. Cell IDs determine where messages can flow; the computation uses only
numerical inputs and weights. Source identity and provenance belong to the caller.
"""

from collections import deque
from dataclasses import dataclass
from math import isfinite
from operator import index

import numpy as np


def _frozen(array):
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, eq=False)
class FieldState:
    _field: object
    drives: np.ndarray
    values: np.ndarray
    affected: tuple[int, ...]
    sweeps: int
    messages: int
    residual: float

    @property
    def error_bound(self):
        """Real-arithmetic infinity-norm bound, without a roundoff allowance."""
        return self.residual / (1.0 - self._field.contraction)


class LatentField:
    """A fixed set of cells linked by directed linear maps and shared relaxation.

    Each edge is (source, target, matrix). Parallel edges add. Each cell updates
    as tanh(drive + sum of incoming messages). The topology and weights stay
    fixed within one field. Reconfiguration returns a new field and snapshot.
    """

    def __init__(self, cells, width, edges, *, tolerance=1e-8, max_sweeps=128):
        self._cells = index(cells)
        self._width = index(width)
        self._max_sweeps = index(max_sweeps)
        self._tolerance = float(tolerance)
        if min(self._cells, self._width, self._max_sweeps) <= 0:
            raise ValueError("cells, width, and max_sweeps must be positive")
        if not isfinite(self._tolerance) or self._tolerance <= 0:
            raise ValueError("tolerance must be positive and finite")

        sources, targets, weights = [], [], []
        incoming = [[] for _ in range(self._cells)]
        outgoing = [set() for _ in range(self._cells)]
        row_mass = np.zeros((self._cells, self._width))
        for source, target, weight in edges:
            source, target = self._cell(source), self._cell(target)
            block = self._array(weight, (self._width, self._width))
            incoming[target].append(len(weights))
            outgoing[source].add(target)
            sources.append(source)
            targets.append(target)
            weights.append(block.copy())
            row_mass[target] += np.abs(block).sum(axis=1)

        self._contraction = float(row_mass.max())
        if self._contraction >= 1.0:
            raise ValueError("the aggregate absolute weight row sums must be below one")
        self._sources = _frozen(np.asarray(sources, dtype=np.intp))
        self._targets = _frozen(np.asarray(targets, dtype=np.intp))
        self._weights = _frozen(np.asarray(weights).reshape(-1, self._width, self._width))
        self._incoming = tuple(tuple(edges) for edges in incoming)
        self._outgoing = tuple(tuple(sorted(targets)) for targets in outgoing)

    @property
    def contraction(self):
        return self._contraction

    def _cell(self, cell):
        return self._checked_cell(cell, self._cells)

    @staticmethod
    def _checked_cell(cell, count):
        cell = index(cell)
        if not 0 <= cell < count:
            raise ValueError("cell index out of range")
        return cell

    @staticmethod
    def _array(value, shape):
        if np.iscomplexobj(value):
            raise ValueError("expected real coordinates")
        array = np.asarray(value, dtype=np.float64)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"expected a finite real array with shape {shape}")
        return array

    def initialize(self, drives):
        initial = self._array(drives, (self._cells, self._width)).copy()
        return self._relax(initial, np.zeros_like(initial), tuple(range(self._cells)), 0.0)

    def _require_state(self, state):
        if state._field is not self:
            raise ValueError("state belongs to a different latent field")

    def revise(self, state, changes):
        """Replace observation drives; recompute only their forward closure."""
        self._require_state(state)
        return self._resume(state, changes, set())

    def reconfigure(self, state, *, append=None, edge_updates=None, changes=None):
        """Return (field, state) for one atomic batch of numerical and topology edits.

        Append rows keep existing cell IDs stable. An edge update replaces all
        parallel blocks between the named source and target. None removes them.
        Existing snapshots keep their original field, drives and values.
        """
        self._require_state(state)
        additions = None
        if append is not None:
            rows = np.asarray(append)
            if rows.ndim != 2:
                raise ValueError("appended observations must be a matrix")
            additions = self._array(rows, (len(rows), self._width))
        added = 0 if additions is None else len(additions)
        count = self._cells + added
        updates = {
            (self._checked_cell(source, count), self._checked_cell(target, count)):
                None if block is None else self._array(block, (self._width, self._width))
            for (source, target), block in (edge_updates or {}).items()
        }
        changes = {} if changes is None else changes
        if not added and not updates:
            return self, self.revise(state, changes)

        next_edges = []
        affected = set(range(self._cells, count))
        for source, target, block in zip(self._sources, self._targets, self._weights):
            if (source, target) in updates:
                affected.add(target)
            else:
                next_edges.append((source, target, block))
        for (source, target), block in updates.items():
            if block is not None:
                next_edges.append((source, target, block))
                affected.add(target)
        if not affected:
            return self, self.revise(state, changes)
        candidate = LatentField(count, self._width, next_edges, tolerance=self._tolerance,
                                max_sweeps=self._max_sweeps)
        return candidate, candidate._resume(state, changes, affected, additions)

    def _resume(self, state, changes, affected, additions=None):
        replacements = {
            self._cell(cell): self._array(value, (self._width,))
            for cell, value in changes.items()
        }
        for cell, value in replacements.items():
            if cell >= len(state.drives) or not np.array_equal(state.drives[cell], value):
                affected.add(cell)
        if not affected:
            return state

        residual = state.residual
        if residual > self._residual_threshold():
            # A larger contraction bound can make the old residual too coarse.
            affected = set(range(self._cells))
        else:
            frontier = deque(affected)
            while frontier:
                for target in self._outgoing[frontier.popleft()]:
                    if target not in affected:
                        affected.add(target)
                        frontier.append(target)
        if len(affected) == self._cells:
            residual = 0.0
        if additions is None:
            drives, values = state.drives.copy(), state.values.copy()
        else:
            drives = np.concatenate((state.drives, additions))
            values = np.concatenate((state.values, np.zeros_like(additions)))
        for cell, value in replacements.items():
            drives[cell] = value
        return self._relax(drives, values, tuple(sorted(affected)), residual)

    def _residual_threshold(self):
        return (1.0 - self._contraction) * self._tolerance

    def _relax(self, drives, values, affected, previous_residual):
        rows = np.asarray(affected, dtype=np.intp)
        edges = np.asarray([edge for cell in affected for edge in self._incoming[cell]],
                           dtype=np.intp)
        source = self._sources[edges]
        target = np.searchsorted(rows, self._targets[edges])
        matrices = self._weights[edges]
        local_drives = drives[rows]
        threshold = self._residual_threshold()

        for sweep in range(1, self._max_sweeps + 1):
            proposed = local_drives.copy()
            messages = np.einsum("eij,ej->ei", matrices, values[source])
            np.add.at(proposed, target, messages)
            np.tanh(proposed, out=proposed)
            residual = float(np.max(np.abs(proposed - values[rows])))
            if residual <= threshold:
                # Outside the forward closure, state and incoming inputs
                # are unchanged. Carry their prior residual bound forward.
                return FieldState(
                    self, _frozen(drives), _frozen(values), affected, sweep,
                    sweep * len(edges), max(residual, previous_residual),
                )
            values[rows] = proposed
        raise RuntimeError("latent field exhausted its relaxation budget")
