"""Revision of weighted relative measurements with recurrent and conventional solvers."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from operator import index
from types import MappingProxyType

import numpy as np

from experiments.memory.latent_field import FieldState, LatentField, _frozen


@dataclass(frozen=True)
class RelativeObservation:
    """Measure x[target] - x[source], or x[target] for an absolute anchor."""

    target: int
    value: float
    source: int | None = None
    weight: float = 1.0

    def __post_init__(self):
        target = index(self.target)
        source = None if self.source is None else index(self.source)
        value, weight = float(self.value), float(self.weight)
        if source == target:
            raise ValueError("relative observation endpoints must differ")
        if not isfinite(value) or not isfinite(weight) or weight <= 0:
            raise ValueError("observation needs a finite value and positive finite weight")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "weight", weight)

    def _endpoints(self):
        return (self.target,) if self.source is None else (self.source, self.target)


def _problem(cells, ridge):
    cells, ridge = index(cells), float(ridge)
    if cells < 1 or not isfinite(ridge) or ridge <= 0:
        raise ValueError("relative field needs positive cells and a positive finite ridge")
    return cells, ridge


def _validate_observation(cells, source_id, observation):
    if type(source_id) is not str or not source_id:
        raise ValueError("observation IDs must be nonempty strings")
    if (type(observation) is not RelativeObservation
            or any(not 0 <= cell < cells for cell in observation._endpoints())):
        raise ValueError("invalid relative observation or endpoint")


@dataclass(frozen=True)
class _Row:
    drive: float
    inputs: tuple[tuple[int, float], ...]
    weights: tuple[tuple[int, float], ...]
    diagonal: float
    anchor_weight: float

    @property
    def anchored(self):
        return self.anchor_weight > 0

    def _neighbors(self):
        return tuple(neighbor for neighbor, _ in self.inputs)


@dataclass(frozen=True, eq=False)
class RelativeSnapshot:
    _owner: object
    _field: LatentField | None
    _state: FieldState | _CGState
    _observations: Mapping[str, RelativeObservation]
    _incident: tuple[frozenset[str], ...]
    _rows: tuple[_Row, ...]
    _supported: frozenset[int]
    recompiled: tuple[int, ...]
    _relaxed: bool = True

    @property
    def affected(self):
        return self._state.affected if self._relaxed else ()

    @property
    def messages(self):
        return self._state.messages if self._relaxed else 0

    @property
    def error_bound(self):
        return self._state.error_bound

    def read(self, cell):
        """Read an absolute value in an anchored component, or None for unsupported."""
        cell = LatentField._checked_cell(cell, len(self._rows))
        return float(self._state.values[cell, 0]) if cell in self._supported else None


class RelativeObservationField:
    """Compile weighted relative measurements into a revision-capable affine field."""

    _snapshot_type = RelativeSnapshot

    def __init__(self, cells, *, ridge, tolerance=1e-8, max_sweeps=512):
        self._cells, self._ridge = _problem(cells, ridge)
        self._field_options = {"tolerance": tolerance, "max_sweeps": max_sweeps,
                               "activation": "identity"}

    def _row(self, cell, incident, observations):
        measured = [observations[name] for name in sorted(incident)]
        diagonal = fsum([self._ridge, *(row.weight for row in measured)])
        inputs, weights, drive, anchor_weights = {}, {}, [], []
        for row in measured:
            coefficient = row.weight / diagonal
            if coefficient == 0:
                raise FloatingPointError("observation weight is unresolved at Float64")
            if row.source is None:
                anchor_weights.append(row.weight)
                drive.append(coefficient * row.value)
            else:
                other = row.target if row.source == cell else row.source
                inputs.setdefault(other, []).append(coefficient)
                weights.setdefault(other, []).append(row.weight)
                drive.append(coefficient * row.value if row.target == cell else -coefficient * row.value)
        return _Row(fsum(drive), tuple((i, fsum(inputs[i])) for i in sorted(inputs)),
                    tuple((i, fsum(weights[i])) for i in sorted(weights)),
                    diagonal, fsum(anchor_weights))

    @staticmethod
    def _closure(rows, seeds):
        reached = set(seeds)
        frontier = deque(sorted(reached))
        while frontier:
            for neighbor in rows[frontier.popleft()]._neighbors():
                if neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
        return reached

    @staticmethod
    def _support(rows):
        return frozenset(RelativeObservationField._closure(
            rows, (cell for cell, row in enumerate(rows) if row.anchored),
        ))

    def _initialize_rows(self, rows):
        edges = ((neighbor, cell, [[weight]]) for cell, row in enumerate(rows)
                 for neighbor, weight in row.inputs)
        field = LatentField(self._cells, 1, edges, **self._field_options)
        return field, field.initialize([[row.drive] for row in rows])

    def initialize(self, observations):
        observations = dict(observations)
        incident = [set() for _ in range(self._cells)]
        for source_id, observation in observations.items():
            _validate_observation(self._cells, source_id, observation)
            for cell in observation._endpoints():
                incident[cell].add(source_id)
        rows = tuple(self._row(cell, names, observations) for cell, names in enumerate(incident))
        field, state = self._initialize_rows(rows)
        return self._snapshot_type(self, field, state, MappingProxyType(observations),
                                   tuple(map(frozenset, incident)), rows, self._support(rows),
                                   tuple(range(self._cells)))

    def _revise_rows(self, base, rows, cells):
        drives, edge_updates = {}, {}
        for cell in cells:
            previous, revised = base._rows[cell], rows[cell]
            drives[cell] = [revised.drive]
            old_inputs, new_inputs = dict(previous.inputs), dict(revised.inputs)
            for neighbor in sorted(old_inputs.keys() | new_inputs.keys()):
                weight = new_inputs.get(neighbor)
                if old_inputs.get(neighbor) != weight:
                    edge_updates[(neighbor, cell)] = None if weight is None else [[weight]]
        return base._field.reconfigure(base._state, edge_updates=edge_updates, changes=drives)

    def revise(self, base, changes):
        """Apply replacements and named removals, preserving earlier snapshots."""
        if base._owner is not self:
            raise ValueError("relative snapshot belongs to a different field")
        staged = []
        for source_id, replacement in changes.items():
            if type(source_id) is not str or not source_id:
                raise ValueError("observation IDs must be nonempty strings")
            previous = base._observations.get(source_id)
            if replacement is None:
                if previous is None:
                    raise ValueError("cannot remove an unknown observation")
            else:
                _validate_observation(self._cells, source_id, replacement)
            if replacement != previous:
                staged.append((source_id, previous, replacement))
        if not staged:
            return base

        cells = tuple(sorted({cell for _, before, after in staged for row in (before, after)
                              if row is not None for cell in row._endpoints()}))
        observations = dict(base._observations)
        incident = list(base._incident)
        updated = {cell: set(incident[cell]) for cell in cells}
        for source_id, previous, replacement in staged:
            if previous is not None:
                for cell in previous._endpoints():
                    updated[cell].remove(source_id)
            if replacement is None:
                del observations[source_id]
            else:
                observations[source_id] = replacement
                for cell in replacement._endpoints():
                    updated[cell].add(source_id)

        rows = list(base._rows)
        support_changed = False
        for cell in cells:
            incident[cell] = frozenset(updated[cell])
            previous = rows[cell]
            revised = self._row(cell, updated[cell], observations)
            rows[cell] = revised
            support_changed |= (previous.anchored != revised.anchored or
                                previous._neighbors() != revised._neighbors())
        rows = tuple(rows)
        field, state = self._revise_rows(base, rows, cells)
        support = self._support(rows) if support_changed else base._supported
        return self._snapshot_type(self, field, state, MappingProxyType(observations), tuple(incident),
                                   rows, support, cells, state is not base._state)


@dataclass(frozen=True, eq=False)
class _CGState:
    values: np.ndarray
    affected: tuple[int, ...]
    iterations: int
    matvecs: int
    messages: int
    error_bound: float


@dataclass(frozen=True, eq=False)
class RelativeCGSnapshot(RelativeSnapshot):
    """Retain the conventional solution and its most recent solve counts."""

    @property
    def iterations(self):
        return self._state.iterations if self._relaxed else 0

    @property
    def matvecs(self):
        return self._state.matvecs if self._relaxed else 0


@dataclass(frozen=True)
class _CGSystem:
    ridge: float
    diagonal: np.ndarray
    grounding: np.ndarray
    source: np.ndarray
    target: np.ndarray
    weights: np.ndarray
    rhs: np.ndarray

    def apply(self, vector):
        differences = self.weights * (vector[self.target] - vector[self.source])
        result = self.grounding * vector
        size = len(vector)
        result += np.bincount(self.target, weights=differences, minlength=size)
        result -= np.bincount(self.source, weights=differences, minlength=size)
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite conventional matrix product")
        return result

    def bound(self, residual):
        return float(np.max(np.abs(residual))) / self.ridge


def _preconditioned_cg(system, initial, tolerance, max_iterations):
    matvecs = 0

    def apply(vector):
        nonlocal matvecs
        matvecs += 1
        return system.apply(vector)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        values = initial.copy()
        residual = system.rhs - apply(values)
        bound = system.bound(residual)
        if bound <= tolerance:
            return values, 0, matvecs, bound
        direction = residual / system.diagonal
        energy = float(residual @ direction)
        for iteration in range(1, max_iterations + 1):
            product = apply(direction)
            curvature = float(direction @ product)
            if energy <= 0 or curvature <= 0 or not isfinite(energy) or not isfinite(curvature):
                raise FloatingPointError("conventional CG could not resolve positive curvature")
            step = energy / curvature
            values += step * direction
            residual -= step * product
            refresh = system.bound(residual) <= tolerance or iteration == max_iterations
            if refresh:
                residual = system.rhs - apply(values)
                bound = system.bound(residual)
                if bound <= tolerance:
                    return values, iteration, matvecs, bound
                if iteration == max_iterations:
                    break
            preconditioned = residual / system.diagonal
            next_energy = float(residual @ preconditioned)
            direction = preconditioned if refresh else preconditioned + (next_energy / energy) * direction
            energy = next_energy
    raise RuntimeError("conventional CG exhausted its iteration budget")


class RelativeObservationCG(RelativeObservationField):
    """Warm-started sparse conventional comparator on the shared observation problem."""

    _snapshot_type = RelativeCGSnapshot

    def __init__(self, cells, *, ridge, tolerance=1e-8, max_iterations=512):
        self._cells, self._ridge = _problem(cells, ridge)
        self._tolerance = float(tolerance)
        self._max_iterations = index(max_iterations)
        if not isfinite(self._tolerance) or self._tolerance <= 0 or self._max_iterations < 1:
            raise ValueError("CG needs positive finite tolerance and a positive iteration budget")

    def _system(self, rows, affected):
        positions = {cell: local for local, cell in enumerate(affected)}
        degrees = np.asarray([rows[cell].diagonal for cell in affected])
        scale = float(degrees.max())
        ridge = self._ridge / scale
        diagonal = degrees / scale
        anchors = np.asarray([rows[cell].anchor_weight for cell in affected])
        scaled_anchors = anchors / scale
        source, target, weights = [], [], []
        for cell in affected:
            for neighbor, weight in rows[cell].weights:
                if cell < neighbor:
                    source.append(positions[cell])
                    target.append(positions[neighbor])
                    weights.append(weight / scale)
        weights = np.asarray(weights)
        if ridge == 0 or np.any(weights == 0) or np.any((anchors > 0) & (scaled_anchors == 0)):
            raise FloatingPointError("unresolved coefficient in conventional sparse system")
        drives = np.asarray([rows[cell].drive for cell in affected])
        rhs = diagonal * drives
        if np.any((drives != 0) & (rhs == 0)):
            raise FloatingPointError("unresolved right-hand side in conventional sparse system")
        return _CGSystem(ridge, diagonal, ridge + scaled_anchors, np.asarray(source, dtype=np.intp),
                         np.asarray(target, dtype=np.intp), weights, rhs)

    def _solve_rows(self, rows, affected, initial, inherited_bound):
        system = self._system(rows, affected)
        active = np.asarray(affected, dtype=np.intp)
        solved, iterations, matvecs, bound = _preconditioned_cg(
            system, initial[active], self._tolerance, self._max_iterations,
        )
        values = initial.copy()
        values[active] = solved
        return None, _CGState(_frozen(values[:, None]), affected, iterations, matvecs,
                              matvecs * 2 * len(system.weights), max(bound, inherited_bound))

    def _initialize_rows(self, rows):
        return self._solve_rows(rows, tuple(range(len(rows))), np.zeros(len(rows)), 0.0)

    def _revise_rows(self, base, rows, cells):
        changed = tuple(cell for cell in cells if rows[cell] != base._rows[cell])
        if not changed:
            return None, base._state
        affected = tuple(sorted(self._closure(rows, changed)))
        inherited = base.error_bound if len(affected) < len(rows) else 0.0
        return self._solve_rows(rows, affected, base._state.values[:, 0], inherited)


def solve_relative_direct(cells, observations, *, ridge):
    """Solve the same weighted objective by direct incidence least squares."""
    cells, ridge = _problem(cells, ridge)
    observations = dict(observations)
    system = np.zeros((len(observations) + cells, cells))
    rhs = np.zeros(len(system))
    components = list(range(cells))
    anchors = set()

    def root(cell):
        while components[cell] != cell:
            cell = components[cell]
        return cell

    for i, (source_id, observation) in enumerate(observations.items()):
        _validate_observation(cells, source_id, observation)
        scale = sqrt(observation.weight)
        system[i, observation.target] = scale
        rhs[i] = scale * observation.value
        if observation.source is None:
            anchors.add(observation.target)
        else:
            system[i, observation.source] = -scale
            components[root(observation.source)] = root(observation.target)
    columns = np.arange(cells)
    system[len(observations) + columns, columns] = sqrt(ridge)
    if not np.isfinite(rhs).all():
        raise FloatingPointError("nonfinite weighted observation in the direct system")
    solution, _, rank, _ = np.linalg.lstsq(system, rhs, rcond=None)
    if rank != cells or not np.isfinite(solution).all():
        raise ValueError("direct solve did not resolve the full numerical rank")
    anchored = {root(cell) for cell in anchors}
    return tuple(float(value) if root(cell) in anchored else None
                 for cell, value in enumerate(solution))
