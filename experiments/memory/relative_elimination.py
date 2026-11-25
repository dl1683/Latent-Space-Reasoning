"""Persistent square-root elimination for a sparse coarse observation system."""

from collections.abc import Mapping
from dataclasses import dataclass
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
    return EliminationStep(inputs, cells[1:], _frozen(readout), remainder)


def solve_elimination(cells, rank, factors, previous: Mapping[int, EliminationStep]):
    """Reuse equal input buckets, then recover coarse values in reverse order."""
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

    solved = {}
    with np.errstate(over="raise", invalid="raise"):
        for cell in reversed(order):
            step = steps[cell]
            later = [solved[variable] for variable in step.dependencies]
            value = float(step.readout[-1] + step.readout[:-1] @ later)
            if not np.isfinite(value):
                raise FloatingPointError("nonfinite coarse elimination readout")
            solved[cell] = value
    values = np.fromiter((solved[cell] for cell in cells), dtype=float, count=len(cells))
    return MappingProxyType(steps), _frozen(values), tuple(rebuilt)
