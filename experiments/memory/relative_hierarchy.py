"""Retained region factors for multiscale revision of relative observations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, sqrt
from operator import index
from types import MappingProxyType

import numpy as np

from experiments.memory.latent_field import LatentField, _frozen
from experiments.memory.relative_observations import (
    RelativeObservation,
    _problem,
    _validate_observation,
)


class _Components:
    def __init__(self, cells):
        self._parent = {cell: cell for cell in cells}

    def root(self, cell):
        while self._parent[cell] != cell:
            self._parent[cell] = self._parent[self._parent[cell]]
            cell = self._parent[cell]
        return cell

    def join(self, first, second):
        self._parent[self.root(first)] = self.root(second)


def _write_observation(row, positions, observation):
    scale = sqrt(observation.weight)
    row[positions[observation.target]] = scale
    if observation.source is not None:
        row[positions[observation.source]] = -scale
    row[-1] = scale * observation.value
    if not isfinite(row[-1]):
        raise FloatingPointError("nonfinite weighted observation in region hierarchy")


@dataclass(frozen=True, eq=False)
class _Region:
    _sources: Mapping[str, RelativeObservation]
    ports: tuple[int, ...]
    positions: Mapping[int, int]
    components: Mapping[int, int]
    anchored: frozenset[int]
    factor: np.ndarray
    readout: np.ndarray


def _eliminate(matrix, split):
    size = matrix.shape[1] - 1
    triangular = np.linalg.qr(matrix, mode="r")
    if not np.isfinite(triangular).all() or np.any(triangular.diagonal()[:size] == 0):
        raise FloatingPointError("unresolved region elimination")
    # R_ii x_i + R_ip x_p = r_i. Drop any trailing constant residual.
    readout = np.linalg.solve(triangular[:split, :split], triangular[:split, split:])
    readout[:, :-1] *= -1
    if not np.isfinite(readout).all():
        raise FloatingPointError("nonfinite region read operator")
    return triangular[split:size, split:], readout


def _build_region(cells, ports, sources, ridge):
    interior = [cell for cell in cells if cell not in ports]
    ports = tuple(sorted(ports))
    order = (*interior, *ports)
    positions = {cell: column for column, cell in enumerate(order)}
    size, split = len(order), len(interior)
    matrix = np.zeros((len(sources) + size, size + 1))
    components = _Components(cells)
    anchors = set()
    for row, name in enumerate(sorted(sources)):
        observation = sources[name]
        _write_observation(matrix[row], positions, observation)
        if observation.source is None:
            anchors.add(observation.target)
        else:
            components.join(observation.source, observation.target)
    columns = np.arange(size)
    matrix[len(sources) + columns, columns] = sqrt(ridge)

    factor, readout = _eliminate(matrix, split)
    return _Region(
        MappingProxyType(sources), ports, MappingProxyType(positions),
        MappingProxyType({cell: components.root(cell) for cell in cells}),
        frozenset(components.root(cell) for cell in anchors),
        _frozen(factor), _frozen(readout),
    )


def _compact_region(region, active_ports):
    retired = [column for column, cell in enumerate(region.ports) if cell not in active_ports]
    retained = [column for column, cell in enumerate(region.ports) if cell in active_ports]
    retained_and_rhs = [*retained, len(region.ports)]
    factor, retired_readout = _eliminate(region.factor[:, retired + retained_and_rhs], len(retired))
    # Substitute the retired coordinates into each original interior read.
    readout = region.readout[:, retained_and_rhs] + region.readout[:, retired] @ retired_readout
    if not np.isfinite(readout).all():
        raise FloatingPointError("nonfinite compacted region read operator")
    readout = np.concatenate((readout, retired_readout), axis=0)
    ports = tuple(region.ports[column] for column in retained)
    order = (*tuple(region.positions)[:len(region.readout)],
             *(region.ports[column] for column in retired), *ports)
    positions = MappingProxyType({cell: column for column, cell in enumerate(order)})
    return _Region(region._sources, ports, positions, region.components, region.anchored,
                   _frozen(factor), _frozen(readout))


def _solve_coarse(ports, bridges, regions):
    size = len(ports)
    if not size:
        return np.empty(0)
    matrix = np.zeros((size + len(bridges), size + 1))
    start = 0
    for region in regions:
        stop = start + len(region.ports)
        matrix[start:stop, start:stop] = region.factor[:, :-1]
        matrix[start:stop, -1] = region.factor[:, -1]
        start = stop
    for row, name in enumerate(sorted(bridges), start=size):
        _write_observation(matrix[row], ports, bridges[name])
    values, _, rank, _ = np.linalg.lstsq(matrix[:, :-1], matrix[:, -1], rcond=None)
    if rank != size or not np.isfinite(values).all():
        raise ValueError("coarse solve did not resolve the full numerical rank")
    return values


@dataclass(frozen=True, eq=False)
class RelativeHierarchySnapshot:
    _owner: RelativeObservationHierarchy
    _sources: Mapping[str, RelativeObservation]
    _bridges: Mapping[str, RelativeObservation]
    _regions: tuple[_Region, ...]
    _port_positions: Mapping[int, int]
    _values: np.ndarray
    _supported_ports: frozenset[int]
    rebuilt_regions: tuple[int, ...]
    compacted_regions: tuple[int, ...] = ()

    @property
    def coarse_size(self):
        return len(self._port_positions)

    def read(self, cell):
        """Read an anchored estimate, recovering only the requested interior coordinate."""
        cell = LatentField._checked_cell(cell, len(self._owner._membership))
        region = self._regions[self._owner._membership[cell]]
        component = region.components[cell]
        if component not in region.anchored and component not in self._supported_ports:
            return None
        if cell in self._port_positions:
            return float(self._values[self._port_positions[cell]])
        coeffs = region.readout[region.positions[cell]]
        start = self._port_positions[region.ports[0]] if region.ports else 0
        values = self._values[start:start + len(region.ports)]
        value = float(coeffs[-1] + coeffs[:-1] @ values)
        if not isfinite(value):
            raise FloatingPointError("nonfinite reconstructed interior coordinate")
        return value


class RelativeObservationHierarchy:
    """Eliminate each region's interior and retain operators for cross-region revisions."""

    def __init__(self, membership, *, ridge):
        self._membership = tuple(map(index, membership))
        self._cells, self._ridge = _problem(len(self._membership), ridge)
        labels = sorted(set(self._membership))
        if labels != list(range(len(labels))):
            raise ValueError("region labels must be consecutive integers from zero")
        regions = [[] for _ in labels]
        for cell, region in enumerate(self._membership):
            regions[region].append(cell)
        self._regions = tuple(map(tuple, regions))

    def _region_of(self, observation):
        target = self._membership[observation.target]
        if observation.source is None or self._membership[observation.source] == target:
            return target
        return None

    def _snapshot(self, sources, bridges, regions, rebuilt, *, compaction_base=None, compacted=()):
        ports = {cell: position for position, cell in enumerate(
            cell for region in regions for cell in region.ports
        )}
        components = {cell: region.components[cell] for region in regions for cell in region.ports}
        coarse_support = _Components(components.values())
        anchors = {region.components[cell] for region in regions for cell in region.ports
                   if region.components[cell] in region.anchored}
        for bridge in bridges.values():
            coarse_support.join(components[bridge.source], components[bridge.target])
        if compaction_base is None:
            values = _solve_coarse(ports, bridges, regions)
        else:
            retained = [compaction_base._port_positions[cell] for cell in ports]
            values = compaction_base._values[retained]
        anchor_roots = {coarse_support.root(component) for component in anchors}
        supported = frozenset(component for component in components.values()
                              if coarse_support.root(component) in anchor_roots)
        return RelativeHierarchySnapshot(
            self, sources, bridges, tuple(regions), MappingProxyType(ports), _frozen(values),
            supported, tuple(rebuilt), tuple(compacted),
        )

    def initialize(self, observations):
        sources = dict(observations)
        regions = [{} for _ in self._regions]
        ports = [set() for _ in self._regions]
        bridges = {}
        for name, observation in sources.items():
            _validate_observation(self._cells, name, observation)
            region = self._region_of(observation)
            if region is None:
                bridges[name] = observation
                for cell in observation._endpoints():
                    ports[self._membership[cell]].add(cell)
            else:
                regions[region][name] = observation
        factors = [_build_region(cells, ports[region], regions[region], self._ridge)
                   for region, cells in enumerate(self._regions)]
        return self._snapshot(MappingProxyType(sources), MappingProxyType(bridges), factors,
                              range(len(factors)))

    def revise(self, base, changes):
        """Apply a named source batch. Exposed ports persist until compacted."""
        if base._owner is not self:
            raise ValueError("relative snapshot belongs to a different hierarchy")
        staged = []
        for name, replacement in changes.items():
            if type(name) is not str or not name:
                raise ValueError("observation IDs must be nonempty strings")
            previous = base._sources.get(name)
            if replacement is None:
                if previous is None:
                    raise ValueError("cannot remove an unknown observation")
            else:
                _validate_observation(self._cells, name, replacement)
            if previous != replacement:
                staged.append((name, previous, replacement))
        if not staged:
            return base

        sources = dict(base._sources)
        bridges = dict(base._bridges)
        factors = list(base._regions)
        local_edits, promotions = {}, {}
        for name, previous, replacement in staged:
            if previous is not None:
                region = self._region_of(previous)
                if region is None:
                    del bridges[name]
                else:
                    local_edits.setdefault(region, {})[name] = None
                del sources[name]
            if replacement is not None:
                region = self._region_of(replacement)
                if region is None:
                    bridges[name] = replacement
                    for cell in replacement._endpoints():
                        if cell not in base._port_positions:
                            promotions.setdefault(self._membership[cell], set()).add(cell)
                else:
                    local_edits.setdefault(region, {})[name] = replacement
                sources[name] = replacement

        rebuilt = sorted(local_edits.keys() | promotions.keys())
        for region in rebuilt:
            previous = factors[region]
            local = dict(previous._sources)
            for name, replacement in local_edits.get(region, {}).items():
                if replacement is None:
                    del local[name]
                else:
                    local[name] = replacement
            ports = set(previous.ports).union(promotions.get(region, ()))
            factors[region] = _build_region(self._regions[region], ports, local, self._ridge)
        return self._snapshot(MappingProxyType(sources), MappingProxyType(bridges), factors, rebuilt)

    def compact(self, base, *, max_regions=1, protected_cells=(), tolerance=1e-10):
        """Retire inactive ports in a bounded number of regions of the selected branch."""
        if base._owner is not self:
            raise ValueError("relative snapshot belongs to a different hierarchy")
        max_regions, tolerance = index(max_regions), float(tolerance)
        if max_regions < 1 or not isfinite(tolerance) or tolerance < 0:
            raise ValueError("compaction needs a positive region cap and finite nonnegative tolerance")
        protected = tuple(LatentField._checked_cell(cell, self._cells) for cell in protected_cells)
        active = {cell for bridge in base._bridges.values() for cell in bridge._endpoints()}
        candidates = [(sum(cell not in active for cell in region.ports), number)
                      for number, region in enumerate(base._regions)]
        chosen = tuple(number for count, number in sorted(candidates, key=lambda pair: (-pair[0], pair[1]))
                       if count)[:max_regions]
        if not chosen:
            return base
        regions = list(base._regions)
        for number in chosen:
            regions[number] = _compact_region(regions[number], active)
        compacted = self._snapshot(base._sources, base._bridges, regions, (),
                                   compaction_base=base, compacted=chosen)
        for cell in protected:
            before, after = base.read(cell), compacted.read(cell)
            if before is None or after is None:
                agrees = before is after
            else:
                agrees = abs(before - after) <= tolerance
            if not agrees:
                raise RuntimeError(f"compaction changed protected read {cell}")
        return compacted
