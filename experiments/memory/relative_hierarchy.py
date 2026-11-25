"""Retained region factors for multiscale revision of relative observations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import zip_longest
from math import isfinite, sqrt
from operator import index
from types import MappingProxyType

import numpy as np

from experiments.memory.latent_field import LatentField, _frozen
from experiments.memory.relative_elimination import (
    EliminationFactor,
    EliminationStep,
    resolve_elimination,
    update_elimination,
)
from experiments.memory.relative_observations import (
    RelativeObservation,
    _problem,
    _validate_observation,
)
from experiments.memory.relative_support import (
    BridgeSupport,
    LocalSourcePaths,
    support_witness,
    update_bridge_support,
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


def _weighted_value(observation):
    value = sqrt(observation.weight) * observation.value
    if not isfinite(value):
        raise FloatingPointError("nonfinite weighted observation in region hierarchy")
    return value


def _write_coefficients(row, positions, observation):
    scale = sqrt(observation.weight)
    row[positions[observation.target]] = scale
    if observation.source is not None:
        row[positions[observation.source]] = -scale


@dataclass(frozen=True, eq=False)
class _Region:
    _sources: Mapping[str, RelativeObservation]
    ports: tuple[int, ...]
    positions: Mapping[int, int]
    components: Mapping[int, int]
    anchored: frozenset[int]
    source_paths: LocalSourcePaths
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
    local_paths, anchor_sources = {}, {}
    for row, name in enumerate(sorted(sources)):
        observation = sources[name]
        _write_coefficients(matrix[row, :-1], positions, observation)
        matrix[row, -1] = _weighted_value(observation)
        if observation.source is None:
            anchor_sources.setdefault(observation.target, name)
        else:
            components.join(observation.source, observation.target)
            local_paths.setdefault(observation.source, {}).setdefault(observation.target, name)
            local_paths.setdefault(observation.target, {}).setdefault(observation.source, name)
    columns = np.arange(size)
    matrix[len(sources) + columns, columns] = sqrt(ridge)

    factor, readout = _eliminate(matrix, split)
    return _Region(
        MappingProxyType(sources), ports, MappingProxyType(positions),
        MappingProxyType({cell: components.root(cell) for cell in cells}),
        frozenset(components.root(cell) for cell in anchor_sources),
        LocalSourcePaths(
            MappingProxyType({cell: MappingProxyType(edges) for cell, edges in local_paths.items()}),
            MappingProxyType(anchor_sources),
        ),
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
    return replace(region, ports=ports, positions=positions, factor=_frozen(factor), readout=_frozen(readout))


@dataclass(frozen=True, eq=False)
class _CoarseOperator:
    left: np.ndarray
    singular: np.ndarray
    right: np.ndarray

    def solve(self, rhs):
        rhs_scale = np.max(np.abs(rhs))
        if not rhs_scale:
            return _frozen(np.zeros_like(self.singular))
        design_scale = self.singular[0]
        rhs_fraction, rhs_exponent = np.frexp(rhs_scale)
        design_fraction, design_exponent = np.frexp(design_scale)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            # Transform at unit input scale before restoring the solution magnitude.
            coordinates = (self.left.T @ (rhs / rhs_scale)) / (self.singular / design_scale)
            values = self.right.T @ coordinates
            values = np.ldexp(values * (rhs_fraction / design_fraction),
                              rhs_exponent - design_exponent)
        if not np.isfinite(values).all():
            raise FloatingPointError("nonfinite coarse solution")
        return _frozen(values)


@dataclass(frozen=True, eq=False)
class _CoarseBlock:
    regions: tuple[_Region, ...]
    bridges: tuple[tuple[str, RelativeObservation], ...]
    operator: _CoarseOperator | None
    values: np.ndarray

    def matches_design(self, regions, bridges):
        return (
            len(self.regions) == len(regions)
            and all(old is new or (
                old.ports == new.ports
                and np.array_equal(old.factor[:, :-1], new.factor[:, :-1])
            ) for old, new in zip(self.regions, regions))
            and _bridge_design(self.bridges) == _bridge_design(bridges)
        )


def _bridge_design(bridges):
    return tuple((name, obs.source, obs.target, obs.weight) for name, obs in bridges)


def _factor_coarse(cells, bridges, regions):
    size = len(cells)
    positions = {cell: column for column, cell in enumerate(cells)}
    matrix = np.zeros((size + len(bridges), size))
    start = 0
    for region in regions:
        stop = start + len(region.ports)
        matrix[start:stop, start:stop] = region.factor[:, :-1]
        start = stop
    for row, (_, observation) in enumerate(bridges, start=size):
        _write_coefficients(matrix[row], positions, observation)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    threshold = np.finfo(matrix.dtype).eps * max(matrix.shape) * singular[0]
    if not np.isfinite(singular).all() or singular[-1] <= threshold:
        raise ValueError("coarse block did not resolve the full numerical rank")
    return _CoarseOperator(_frozen(left), _frozen(singular), _frozen(right))


def _solve_coarse(cells, regions, bridges, previous):
    rhs = np.concatenate([*(region.factor[:, -1] for region in regions),
                          [_weighted_value(observation) for _, observation in bridges]])
    operator = None
    if previous is not None and previous.operator is not None and previous.matches_design(regions, bridges):
        operator = previous.operator
    refactored = operator is None
    if refactored:
        operator = _factor_coarse(cells, bridges, regions)
    return _CoarseBlock(regions, bridges, operator, operator.solve(rhs)), refactored


def _coarse_groups(regions, bridges):
    # A numerical block includes all ports in each member region. Support is finer.
    owners = {cell: number for number, region in enumerate(regions) for cell in region.ports}
    components = _Components(owners.values())
    for observation in bridges.values():
        components.join(owners[observation.source], owners[observation.target])
    grouped_regions = {}
    for number, region in enumerate(regions):
        if region.ports:
            grouped_regions.setdefault(components.root(number), []).append(region)
    grouped_bridges = {root: [] for root in grouped_regions}
    for name, observation in sorted(bridges.items()):
        grouped_bridges[components.root(owners[observation.source])].append((name, observation))
    return ((tuple(group), tuple(grouped_bridges[root])) for root, group in grouped_regions.items())


def _coarse_state(ports, regions, bridges, base, projecting):
    blocks = {}
    values = np.empty(len(ports))
    factorizations = solves = 0
    for members, rows in _coarse_groups(regions, bridges):
        cells = tuple(cell for region in members for cell in region.ports)
        previous = base._coarse_blocks.get(cells) if base is not None else None
        if previous is not None and previous.regions == members and previous.bridges == rows:
            block = previous
        elif projecting:
            old_columns = [base._port_positions[cell] for cell in cells]
            block = _CoarseBlock(members, rows, None, _frozen(base._values[old_columns]))
        else:
            block, refactored = _solve_coarse(cells, members, rows, previous)
            factorizations += refactored
            solves += 1
        blocks[cells] = block
        values[[ports[cell] for cell in cells]] = block.values
    return MappingProxyType(blocks), _frozen(values), factorizations, solves


@dataclass(frozen=True, eq=False)
class _EliminationCache:
    regions: Mapping[int, tuple[_Region, tuple[EliminationFactor, ...]]]
    bridges: Mapping[str, tuple[RelativeObservation, EliminationFactor]]
    steps: Mapping[int, EliminationStep]
    intakes: Mapping[int, Mapping[tuple, EliminationFactor]]


def _regional_elimination_factors(region):
    components = {}
    for column, cell in enumerate(region.ports):
        components.setdefault(region.components[cell], []).append(column)
    if len(components) == 1:
        return (EliminationFactor(region.ports, region.factor),)
    # Coefficients between disconnected regional components are structural zeros.
    rhs = len(region.ports)
    return tuple(
        EliminationFactor(tuple(region.ports[column] for column in columns),
                          _frozen(region.factor[np.ix_(columns, [*columns, rhs])]))
        for columns in components.values()
    )


def _sparse_coarse(ports, regions, bridges, rank, previous, rebuilt_regions, bridge_edits):
    if previous is None:
        rebuilt_regions, bridge_edits = range(len(regions)), bridges
        regional_factors, bridge_factors, prior_steps, prior_intakes = {}, {}, {}, {}
        promoted = list(ports)
    else:
        if not bridge_edits and all(not regions[number].ports for number in rebuilt_regions):
            return previous, (), ()
        regional_factors, bridge_factors = dict(previous.regions), dict(previous.bridges)
        prior_steps, prior_intakes = previous.steps, previous.intakes
        promoted = []

    changes = []
    for number in rebuilt_regions:
        region = regions[number]
        old = regional_factors.pop(number, None)
        old_factors = old[1] if old is not None else ()
        new_factors = ()
        if region.ports:
            new_factors = _regional_elimination_factors(region)
            regional_factors[number] = region, new_factors
            if previous is not None:
                promoted.extend(cell for cell in region.ports if cell not in prior_steps)
        changes.extend(((0, number, part), before, after)
                       for part, (before, after) in enumerate(zip_longest(old_factors, new_factors)))

    for name in bridge_edits:
        bridge = bridges.get(name)
        old = bridge_factors.pop(name, None)
        old_factor = old[1] if old is not None else None
        new_factor = None
        if bridge is not None:
            cells = bridge._endpoints()
            row = np.zeros((1, 3))
            _write_coefficients(row[0, :-1], {cell: i for i, cell in enumerate(cells)}, bridge)
            row[0, -1] = _weighted_value(bridge)
            new_factor = EliminationFactor(cells, _frozen(row))
            bridge_factors[name] = bridge, new_factor
        changes.append(((1, name, 0), old_factor, new_factor))

    steps, intakes, rebuilt, visited = update_elimination(
        rank, prior_steps, prior_intakes, changes, promoted,
    )
    cache = _EliminationCache(MappingProxyType(regional_factors), MappingProxyType(bridge_factors),
                              steps, intakes)
    return cache, rebuilt, visited


@dataclass(frozen=True, eq=False)
class RelativeHierarchySnapshot:
    _owner: RelativeObservationHierarchy
    _sources: Mapping[str, RelativeObservation]
    _bridges: Mapping[str, RelativeObservation]
    _regions: tuple[_Region, ...]
    _port_positions: Mapping[int, int]
    _values: np.ndarray | None
    _coarse_blocks: Mapping[tuple[int, ...], _CoarseBlock]
    _supported_ports: frozenset[int]
    _support_graph: BridgeSupport
    rebuilt_regions: tuple[int, ...]
    compacted_regions: tuple[int, ...] = ()
    coarse_factorizations: int = 0
    coarse_rhs_solves: int = 0
    rebuilt_buckets: tuple[int, ...] = ()
    visited_buckets: tuple[int, ...] = ()
    visited_support_components: tuple[int, ...] = ()
    _elimination: _EliminationCache | None = None
    _elimination_rank: Mapping[int, int] | None = None

    @property
    def coarse_size(self):
        return len(self._port_positions)

    @property
    def elimination_order(self):
        """Return this branch's sparse ordering, or None for the default SVD."""
        return None if self._elimination_rank is None else tuple(self._elimination_rank)

    def read(self, cell):
        """Read an anchored estimate and recover the boundary values it needs."""
        return self.read_many((cell,))[0]

    def support_witness(self, cell):
        """Return a named source path to an anchor, or None for an unsupported cell."""
        cell = LatentField._checked_cell(cell, len(self._owner._membership))
        return support_witness(self, cell)

    def _readout(self, cell):
        cell = LatentField._checked_cell(cell, len(self._owner._membership))
        region = self._regions[self._owner._membership[cell]]
        component = region.components[cell]
        if component not in region.anchored and component not in self._supported_ports:
            return None
        if cell in self._port_positions:
            return (cell,), None
        coeffs = region.readout[region.positions[cell]]
        if self._elimination is None:
            return region.ports, coeffs
        needed = [column for column, port in enumerate(region.ports)
                  if region.components[port] == component and coeffs[column] != 0]
        return tuple(region.ports[column] for column in needed), coeffs[[*needed, len(region.ports)]]

    def _read_ports(self, ports):
        if not ports:
            return {}
        if self._elimination is not None:
            return resolve_elimination(self._elimination.steps, ports)
        return {port: float(self._values[self._port_positions[port]]) for port in ports}

    def read_many(self, cells):
        """Read in request order, sharing boundary recovery within this batch."""
        return self._read_batch(cells)[0]

    def _read_batch(self, cells):
        plans = tuple(self._readout(cell) for cell in cells)
        values = None
        if self._elimination is not None:
            requested = dict.fromkeys(port for plan in plans if plan is not None for port in plan[0])
            values = self._read_ports(requested)
        reads = []
        for plan in plans:
            if plan is None:
                reads.append(None)
                continue
            ports, coeffs = plan
            if values is None:
                start = self._port_positions[ports[0]] if ports else 0
                inputs = self._values[start:start + len(ports)]
            else:
                inputs = [values[port] for port in ports]
            if coeffs is None:
                reads.append(float(inputs[0]))
                continue
            value = float(coeffs[-1] + coeffs[:-1] @ inputs)
            if not isfinite(value):
                raise FloatingPointError("nonfinite reconstructed interior coordinate")
            reads.append(value)
        return tuple(reads), len(values) if values is not None else 0


class RelativeObservationHierarchy:
    """Eliminate each region's interior and retain operators for cross-region revisions."""

    def __init__(self, membership, *, ridge, elimination_order=None):
        self._membership = tuple(map(index, membership))
        self._cells, self._ridge = _problem(len(self._membership), ridge)
        self._initial_rank = None if elimination_order is None else self._rank(elimination_order)
        labels = sorted(set(self._membership))
        if labels != list(range(len(labels))):
            raise ValueError("region labels must be consecutive integers from zero")
        regions = [[] for _ in labels]
        for cell, region in enumerate(self._membership):
            regions[region].append(cell)
        self._regions = tuple(map(tuple, regions))

    def _rank(self, elimination_order):
        order = tuple(map(index, elimination_order))
        if len(order) != self._cells or set(order) != set(range(self._cells)):
            raise ValueError("coarse elimination order must contain every cell once")
        return MappingProxyType({cell: rank for rank, cell in enumerate(order)})

    def _region_of(self, observation):
        target = self._membership[observation.target]
        if observation.source is None or self._membership[observation.source] == target:
            return target
        return None

    def _snapshot(self, sources, bridges, regions, rebuilt, *, base=None, compacted=(), new_rank=None,
                  bridge_edits=()):
        ports = {cell: position for position, cell in enumerate(
            cell for region in regions for cell in region.ports
        )}
        support, support_visited = update_bridge_support(
            self._membership, regions, bridges, base, rebuilt, bridge_edits,
        )
        cache, buckets, visited = None, (), ()
        factorizations = solves = 0
        blocks = MappingProxyType({})
        rank = self._initial_rank if base is None else base._elimination_rank
        if new_rank is not None:
            rank = new_rank
        if rank is None:
            blocks, values, factorizations, solves = _coarse_state(
                ports, regions, bridges, base, projecting=bool(compacted),
            )
        elif compacted:
            projected = base._read_ports(ports)
            values = _frozen(np.fromiter((projected[cell] for cell in ports), dtype=float, count=len(ports)))
        else:
            prior = base._elimination if base is not None and new_rank is None else None
            cache, buckets, visited = _sparse_coarse(ports, regions, bridges, rank, prior,
                                                      rebuilt, bridge_edits)
            values = None
        return RelativeHierarchySnapshot(
            self, sources, bridges, tuple(regions), MappingProxyType(ports), values, blocks,
            support.supported, support, rebuilt_regions=tuple(rebuilt), compacted_regions=tuple(compacted),
            coarse_factorizations=factorizations, coarse_rhs_solves=solves, rebuilt_buckets=buckets,
            visited_buckets=visited, visited_support_components=support_visited, _elimination=cache,
            _elimination_rank=rank,
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
        local_edits, promotions, bridge_edits = {}, {}, set()
        for name, previous, replacement in staged:
            if previous is not None:
                region = self._region_of(previous)
                if region is None:
                    del bridges[name]
                    bridge_edits.add(name)
                else:
                    local_edits.setdefault(region, {})[name] = None
                del sources[name]
            if replacement is not None:
                region = self._region_of(replacement)
                if region is None:
                    bridges[name] = replacement
                    bridge_edits.add(name)
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
        return self._snapshot(MappingProxyType(sources), MappingProxyType(bridges), factors, rebuilt,
                              base=base, bridge_edits=bridge_edits)

    def reorder(self, base, elimination_order):
        """Branch from the same sources with a different sparse elimination order."""
        if base._owner is not self or base._elimination_rank is None:
            raise ValueError("reordering needs a sparse snapshot from this hierarchy")
        rank = self._rank(elimination_order)
        if rank == base._elimination_rank:
            return base
        return self._snapshot(base._sources, base._bridges, base._regions, (), base=base, new_rank=rank)

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
        compacted = self._snapshot(base._sources, base._bridges, regions, (), base=base,
                                   compacted=chosen)
        for cell, before, after in zip(protected, base.read_many(protected), compacted.read_many(protected)):
            if before is None or after is None:
                agrees = before is after
            else:
                agrees = abs(before - after) <= tolerance
            if not agrees:
                raise RuntimeError(f"compaction changed protected read {cell}")
        return compacted
