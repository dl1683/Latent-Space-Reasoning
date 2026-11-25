"""Affine latent revision over weighted measurements, with an independent dense oracle."""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from operator import index
from types import MappingProxyType

import numpy as np

from experiments.memory.latent_field import FieldState, LatentField


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
    anchored: bool

    def _neighbors(self):
        return tuple(neighbor for neighbor, _ in self.inputs)


@dataclass(frozen=True, eq=False)
class RelativeSnapshot:
    _owner: object
    _field: LatentField
    _state: FieldState
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

    def __init__(self, cells, *, ridge, tolerance=1e-8, max_sweeps=512):
        self._cells, self._ridge = _problem(cells, ridge)
        self._field_options = {"tolerance": tolerance, "max_sweeps": max_sweeps,
                               "activation": "identity"}

    def _row(self, cell, incident, observations):
        measured = [observations[name] for name in sorted(incident)]
        diagonal = fsum([self._ridge, *(row.weight for row in measured)])
        inputs, drive = {}, []
        anchored = False
        for row in measured:
            coefficient = row.weight / diagonal
            if coefficient == 0:
                raise FloatingPointError("observation weight is unresolved at Float64")
            if row.source is None:
                anchored = True
                drive.append(coefficient * row.value)
            else:
                other = row.target if row.source == cell else row.source
                inputs.setdefault(other, []).append(coefficient)
                drive.append(coefficient * row.value if row.target == cell else -coefficient * row.value)
        return _Row(fsum(drive), tuple((i, fsum(inputs[i])) for i in sorted(inputs)), anchored)

    @staticmethod
    def _support(rows):
        supported = {cell for cell, row in enumerate(rows) if row.anchored}
        frontier = deque(sorted(supported))
        while frontier:
            for neighbor in rows[frontier.popleft()]._neighbors():
                if neighbor not in supported:
                    supported.add(neighbor)
                    frontier.append(neighbor)
        return frozenset(supported)

    def initialize(self, observations):
        observations = dict(observations)
        incident = [set() for _ in range(self._cells)]
        for source_id, observation in observations.items():
            _validate_observation(self._cells, source_id, observation)
            for cell in observation._endpoints():
                incident[cell].add(source_id)
        rows = tuple(self._row(cell, names, observations) for cell, names in enumerate(incident))
        edges = ((neighbor, cell, [[weight]]) for cell, row in enumerate(rows)
                 for neighbor, weight in row.inputs)
        field = LatentField(self._cells, 1, edges, **self._field_options)
        state = field.initialize([[row.drive] for row in rows])
        return RelativeSnapshot(self, field, state, MappingProxyType(observations),
                                tuple(map(frozenset, incident)), rows, self._support(rows),
                                tuple(range(self._cells)))

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

        cells = {cell for _, before, after in staged for row in (before, after)
                 if row is not None for cell in row._endpoints()}
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
        drives, edge_updates = {}, {}
        support_changed = False
        for cell in sorted(cells):
            incident[cell] = frozenset(updated[cell])
            previous = rows[cell]
            revised = self._row(cell, updated[cell], observations)
            rows[cell] = revised
            drives[cell] = [revised.drive]
            old_inputs, new_inputs = dict(previous.inputs), dict(revised.inputs)
            support_changed |= (previous.anchored != revised.anchored or
                                old_inputs.keys() != new_inputs.keys())
            for neighbor in sorted(old_inputs.keys() | new_inputs.keys()):
                weight = new_inputs.get(neighbor)
                if old_inputs.get(neighbor) != weight:
                    edge_updates[(neighbor, cell)] = None if weight is None else [[weight]]
        field, state = base._field.reconfigure(base._state, edge_updates=edge_updates, changes=drives)
        support = self._support(rows) if support_changed else base._supported
        return RelativeSnapshot(self, field, state, MappingProxyType(observations), tuple(incident),
                                tuple(rows), support, tuple(sorted(cells)), state is not base._state)


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
