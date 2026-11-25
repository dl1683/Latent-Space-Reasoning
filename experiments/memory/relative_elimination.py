"""Persistent square-root elimination for a sparse coarse observation system."""

from collections.abc import Mapping
from dataclasses import dataclass
from heapq import heappop, heappush
from types import MappingProxyType

import numpy as np

from experiments.memory.latent_field import _frozen


@dataclass(frozen=True, eq=False)
class EliminationFactor:
    cells: tuple[int, ...]
    rows: np.ndarray


@dataclass(frozen=True, eq=False)
class EliminationStep:
    inputs: tuple[EliminationFactor, ...]
    dependencies: tuple[int, ...]
    readout: np.ndarray
    remainder: EliminationFactor | None
    pivot: float


def _reduce_bucket(cell, inputs, rank, previous):
    cells = tuple(sorted({variable for factor in inputs for variable in factor.cells},
                         key=rank.__getitem__))
    if not cells or cells[0] != cell:
        raise ValueError(f"coarse elimination has no retained factor for port {cell}")
    columns = {variable: i for i, variable in enumerate(cells)}
    matrix = np.zeros((sum(len(factor.rows) for factor in inputs), len(cells) + 1))
    start = 0
    for factor in inputs:
        stop = start + len(factor.rows)
        indices = [columns[variable] for variable in factor.cells]
        matrix[start:stop, indices] = factor.rows[:, :-1]
        matrix[start:stop, -1] = factor.rows[:, -1]
        start = stop

    triangular = np.linalg.qr(matrix, mode="r")
    scale = np.max(np.abs(matrix[:, :-1]))
    threshold = np.finfo(matrix.dtype).eps * max(matrix.shape) * scale
    if not np.isfinite(triangular).all() or abs(triangular[0, 0]) <= threshold:
        raise ValueError(f"coarse elimination could not resolve pivot for port {cell}")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        readout = triangular[0, 1:] / triangular[0, 0]
    readout[:-1] *= -1
    needed = np.flatnonzero(readout[:-1] != 0)
    dependencies = tuple(cells[column + 1] for column in needed)
    readout = readout[[*needed, len(cells) - 1]]

    tail = triangular[1:len(cells), 1:]
    nonconstant = tail[np.any(tail[:, :-1] != 0, axis=1)]
    remainder = None
    if len(nonconstant):
        active = np.flatnonzero(np.any(nonconstant[:, :-1] != 0, axis=0))
        live_cells = tuple(cells[column + 1] for column in active)
        rows = nonconstant[:, [*active, nonconstant.shape[1] - 1]]
        old = previous.remainder if previous is not None else None
        if old is not None and old.cells == live_cells and np.array_equal(old.rows, rows):
            remainder = old
        else:
            remainder = EliminationFactor(live_cells, _frozen(rows))
    return EliminationStep(inputs, dependencies, _frozen(readout), remainder, float(triangular[0, 0]))


def compile_elimination(cells, rank, factors, previous: Mapping[int, EliminationStep]):
    """Reuse equal input buckets and retain conditionals for later reads."""
    order = sorted(cells, key=rank.__getitem__)
    buckets = {cell: [] for cell in order}
    for factor in factors:
        buckets[min(factor.cells, key=rank.__getitem__)].append(factor)

    steps, rebuilt = {}, []
    for cell in order:
        inputs = tuple(buckets.pop(cell))
        old = previous.get(cell)
        if old is not None and old.inputs == inputs:
            step = old
        else:
            step = _reduce_bucket(cell, inputs, rank, old)
            rebuilt.append(cell)
        steps[cell] = step
        if step.remainder is not None:
            buckets[step.remainder.cells[0]].append(step.remainder)
    return MappingProxyType(steps), tuple(rebuilt)


def update_elimination(rank, prior_steps, prior_intakes, changes, promoted):
    """Reroute changed inputs and reduce their affected buckets in rank order."""
    steps, intakes = dict(prior_steps), dict(prior_intakes)
    pending, queue = {}, []

    def touch(cell):
        if cell not in pending:
            pending[cell] = dict(intakes.get(cell, {}))
            heappush(queue, (rank[cell], cell))
        return pending[cell]

    def route(key, previous, replacement):
        if previous is replacement:
            return
        if previous is not None:
            old_cell = min(previous.cells, key=rank.__getitem__)
            del touch(old_cell)[key]
        if replacement is not None:
            new_cell = min(replacement.cells, key=rank.__getitem__)
            touch(new_cell)[key] = replacement

    for key, previous, replacement in changes:
        route(key, previous, replacement)
    for cell in promoted:
        touch(cell)

    rebuilt, visited = [], []
    while queue:
        position, cell = heappop(queue)
        visited.append(cell)
        intake = pending.pop(cell)
        inputs = tuple(intake[key] for key in sorted(intake))
        intakes[cell] = MappingProxyType(intake)
        previous = steps.get(cell)
        if previous is not None and previous.inputs == inputs:
            continue
        step = _reduce_bucket(cell, inputs, rank, previous)
        steps[cell] = step
        rebuilt.append(cell)
        old_remainder = previous.remainder if previous is not None else None
        route((2, position, 0), old_remainder, step.remainder)
    return MappingProxyType(steps), MappingProxyType(intakes), tuple(rebuilt), tuple(visited)


def resolve_elimination(steps: Mapping[int, EliminationStep], cells, *, offsets=None, solved=None, failed=None):
    """Recover a dependency closure. Optional memos belong to the same steps and offsets."""
    solved = {} if solved is None else solved
    isolate = failed is not None
    failed = set() if failed is None else failed
    with np.errstate(over="raise", invalid="raise"):
        for root in cells:
            if root in solved or root in failed:
                continue
            pending = [(root, iter(steps[root].dependencies))]
            while pending:
                cell, dependencies = pending[-1]
                dependency = next(dependencies, None)
                if dependency is not None:
                    if dependency not in solved and dependency not in failed:
                        pending.append((dependency, iter(steps[dependency].dependencies)))
                    continue
                pending.pop()
                step = steps[cell]
                if failed and any(dependency in failed for dependency in step.dependencies):
                    failed.add(cell)
                    continue
                later = [solved[dependency] for dependency in step.dependencies]
                offset = step.readout[-1] if offsets is None else offsets.get(cell, 0.0)
                try:
                    value = float(offset + step.readout[:-1] @ later)
                    if not np.isfinite(value):
                        raise FloatingPointError("nonfinite coarse elimination readout")
                except FloatingPointError:
                    if not isolate:
                        raise
                    failed.add(cell)
                else:
                    solved[cell] = value
    return solved


def resolve_adjoint(steps, rank, forcing, cells):
    """Apply the inverse coarse precision without constructing normal equations."""
    pending, queue, offsets = {}, [], {}

    def add(cell, value):
        if cell not in pending:
            heappush(queue, (rank[cell], cell))
        value += pending.get(cell, 0.0)
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite coarse adjoint forcing")
        pending[cell] = value

    for cell, value in forcing.items():
        if value:
            add(cell, value)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        while queue:
            _, cell = heappop(queue)
            value = pending.pop(cell)
            if not value:
                continue
            step = steps[cell]
            # R = D L. Solve L.T z = forcing, then L u = D^-2 z.
            offset = (value / step.pivot) / step.pivot
            if not np.isfinite(offset):
                raise FloatingPointError("nonfinite coarse adjoint offset")
            offsets[cell] = offset
            for dependency, coefficient in zip(step.dependencies, step.readout[:-1]):
                add(dependency, coefficient * value)
    return resolve_elimination(steps, cells, offsets=offsets)
