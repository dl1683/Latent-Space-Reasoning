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
    fixed across drive revisions. Callers may retain any earlier FieldState.
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
        cell = index(cell)
        if not 0 <= cell < self._cells:
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

    def revise(self, state, changes):
        """Replace observation drives; recompute only their forward closure."""
        if state._field is not self:
            raise ValueError("state belongs to a different latent field")
        replacements = {
            self._cell(cell): self._array(value, (self._width,))
            for cell, value in changes.items()
        }
        frontier = deque(cell for cell, value in replacements.items()
                         if not np.array_equal(state.drives[cell], value))
        if not frontier:
            return state
        affected = set(frontier)
        while frontier:
            for target in self._outgoing[frontier.popleft()]:
                if target not in affected:
                    affected.add(target)
                    frontier.append(target)
        drives = state.drives.copy()
        for cell, value in replacements.items():
            drives[cell] = value
        return self._relax(drives, state.values.copy(), tuple(sorted(affected)), state.residual)

    def _relax(self, drives, values, affected, previous_residual):
        rows = np.asarray(affected, dtype=np.intp)
        edges = np.asarray([edge for cell in affected for edge in self._incoming[cell]],
                           dtype=np.intp)
        source = self._sources[edges]
        target = np.searchsorted(rows, self._targets[edges])
        matrices = self._weights[edges]
        local_drives = drives[rows]
        threshold = (1.0 - self._contraction) * self._tolerance

        for sweep in range(1, self._max_sweeps + 1):
            proposed = local_drives.copy()
            messages = np.einsum("eij,ej->ei", matrices, values[source])
            np.add.at(proposed, target, messages)
            np.tanh(proposed, out=proposed)
            residual = float(np.max(np.abs(proposed - values[rows])))
            if residual <= threshold:
                # Outside the forward closure, both state and incoming drives
                # are unchanged. Carry their prior residual bound forward.
                return FieldState(
                    self, _frozen(drives), _frozen(values), affected, sweep,
                    sweep * len(edges), max(residual, previous_residual),
                )
            values[rows] = proposed
        raise RuntimeError("latent field exhausted its relaxation budget")
