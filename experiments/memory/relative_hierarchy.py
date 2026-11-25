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
    resolve_adjoint,
    resolve_elimination,
    update_elimination,
)
from experiments.memory.relative_observations import (
    RelativeObservation,
    _problem,
    _validate_observation,
)
from experiments.memory.relative_patch import preview_source_patch
from experiments.memory.relative_support import (
    BridgeSupport,
    LocalSourcePaths,
    supported_after_removal,
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
class _RegionFusion:
    children: tuple[_Region, _Region]
    bridges: Mapping[str, RelativeObservation]
    lazy_readout: bool


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
    fusion: _RegionFusion | None = None

    @property
    def interior_size(self):
        return len(self.positions) - len(self.ports)

    @property
    def lazy_readout(self):
        return self.fusion is not None and self.fusion.lazy_readout


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
    order = (*tuple(region.positions)[:region.interior_size],
             *(region.ports[column] for column in retired), *ports)
    positions = MappingProxyType({cell: column for column, cell in enumerate(order)})
    return replace(region, ports=ports, positions=positions, factor=_frozen(factor), readout=_frozen(readout))


def _stored_readout(region, cell, path=None):
    while region.lazy_readout:
        # Retired boundary rows form a suffix of the interior order.
        if region.positions[cell] >= region.interior_size - len(region.readout):
            break
        if path is not None:
            path.append(region)
        left, right = region.fusion.children
        region = left if cell in left.positions else right
    return region, region.readout[region.positions[cell] - region.interior_size]


def _region_readout_row(region, cell, *, work=None):
    path = []
    region, row = _stored_readout(region, cell, path)
    if not path:
        return row
    with np.errstate(over="raise", invalid="raise"):
        for parent in reversed(path):
            lifted = np.zeros(len(parent.ports) + 1)
            lifted[-1] = row[-1]
            for port, weight in zip(region.ports, row[:-1]):
                if weight == 0:
                    continue
                column = parent.positions[port] - parent.interior_size
                if column < 0:
                    lifted += weight * parent.readout[column]
                else:
                    lifted[column] += weight
                if work is not None:
                    work[0] += len(lifted) if column < 0 else 1
            row, region = lifted, parent
    return row


def _full_region_readout(region):
    if not region.lazy_readout:
        return region.readout
    cells = tuple(region.positions)[:region.interior_size]
    rows = [_region_readout_row(region, cell) for cell in cells]
    return np.asarray(rows).reshape(region.interior_size, len(region.ports) + 1)


def _merged_region(regions, internalized, active_ports, *, retain_children=False, lazy_readout=False):
    old_ports = tuple(cell for region in regions for cell in region.ports)
    retired_ports = tuple(cell for cell in old_ports if cell not in active_ports)
    ports = tuple(sorted(active_ports))
    columns = {cell: column for column, cell in enumerate((*retired_ports, *ports))}
    matrix = np.zeros((len(old_ports) + len(internalized), len(old_ports) + 1))
    row = 0
    for region in regions:
        block = slice(row, row + len(region.ports))
        matrix[block, [columns[cell] for cell in region.ports]] = region.factor[:, :-1]
        matrix[block, -1] = region.factor[:, -1]
        row = block.stop
    for row, name in enumerate(sorted(internalized), start=row):
        source = internalized[name]
        _write_coefficients(matrix[row, :-1], columns, source)
        matrix[row, -1] = _weighted_value(source)

    factor, retired_readout = _eliminate(matrix, len(retired_ports))
    readout = retired_readout
    if not lazy_readout:
        transport = np.concatenate((retired_readout, np.eye(len(ports) + 1)), axis=0)
        readouts = [_full_region_readout(region) @ transport[[*(columns[cell] for cell in region.ports), -1]]
                    for region in regions]
        readout = np.concatenate((*readouts, retired_readout), axis=0)
    interior = tuple(cell for region in regions for cell in tuple(region.positions)[:region.interior_size])
    if not np.isfinite(readout).all():
        raise FloatingPointError("nonfinite merged region read operator")
    cells = (*interior, *retired_ports, *ports)
    positions = MappingProxyType({cell: column for column, cell in enumerate(cells)})

    components = _Components(cells)
    sources, neighbors, anchors = {}, {}, {}
    for region in regions:
        for cell, component in region.components.items():
            components.join(cell, component)
        sources.update(region._sources)
        neighbors.update(region.source_paths.neighbors)
        anchors.update(region.source_paths.anchors)
    sources.update(internalized)
    joined_neighbors = {}
    for name, source in sorted(internalized.items()):
        first, second = source.source, source.target
        components.join(first, second)
        for cell, peer in ((first, second), (second, first)):
            if cell not in joined_neighbors:
                joined_neighbors[cell] = dict(neighbors.get(cell, {}))
            joined_neighbors[cell].setdefault(peer, name)
    neighbors.update({cell: MappingProxyType(edges) for cell, edges in joined_neighbors.items()})
    return _Region(
        MappingProxyType(sources), ports, positions,
        MappingProxyType({cell: components.root(cell) for cell in cells}),
        frozenset(components.root(cell) for cell in anchors),
        LocalSourcePaths(MappingProxyType(neighbors), MappingProxyType(anchors)),
        _frozen(factor), _frozen(readout),
        _RegionFusion(regions, MappingProxyType(internalized), lazy_readout) if retain_children else None,
    )


def _apply_region_edits(sources, edits):
    revised = dict(sources)
    for name, replacement in edits.items():
        if replacement is None:
            del revised[name]
        else:
            revised[name] = replacement
    return revised


def _stage_fusion_edits(region, edits, ports):
    left, right = region.fusion.children
    left_edits, right_edits, crossing_edits = {}, {}, {}

    def placement(row):
        target_on_left = row.target in left.positions
        if row.source is None or (row.source in left.positions) == target_on_left:
            return left_edits if target_on_left else right_edits
        return crossing_edits

    for name, replacement in edits.items():
        previous = region._sources.get(name)
        if previous is not None:
            placement(previous)[name] = None
        if replacement is not None:
            placement(replacement)[name] = replacement

    bridges = _apply_region_edits(region.fusion.bridges, crossing_edits)
    required = ports | {cell for row in bridges.values() for cell in row._endpoints()}
    requests = [(child, child_edits, set(child.ports) | (required & child.positions.keys()))
                for child, child_edits in ((left, left_edits), (right, right_edits))]
    return bridges, requests


def _revise_region(region, edits, ports, ridge):
    pending = [(region, edits, ports)]
    completed, composites = {}, []
    while pending:
        node, changes, exposed = pending.pop()
        if not changes and exposed == set(node.ports):
            completed[node] = node
        elif node.fusion is None:
            sources = _apply_region_edits(node._sources, changes)
            completed[node] = _build_region(sorted(node.positions), exposed, sources, ridge)
        else:
            bridges, children = _stage_fusion_edits(node, changes, exposed)
            composites.append((node, bridges, exposed))
            pending.extend(children)

    for node, bridges, exposed in reversed(composites):
        children = tuple(completed[child] for child in node.fusion.children)
        completed[node] = _merged_region(
            children, bridges, exposed, retain_children=True, lazy_readout=node.lazy_readout,
        )
    return completed[region]


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


def _local_root(region, ridge, component):
    interior = tuple(node for node, position in region.positions.items()
                     if position < region.interior_size and region.components[node] == component)
    positions = {node: column for column, node in enumerate(interior)}
    observations = [row for _, row in sorted(region._sources.items())
                    if any(node in positions for node in row._endpoints())]
    size = len(interior)
    matrix = np.zeros((len(observations) + size, size))
    for i, row in enumerate(observations):
        scale = sqrt(row.weight)
        if row.target in positions:
            matrix[i, positions[row.target]] = scale
        if row.source in positions:
            matrix[i, positions[row.source]] = -scale
    indices = np.arange(size)
    matrix[len(observations) + indices, indices] = sqrt(ridge)
    return interior, np.linalg.qr(matrix, mode="r")


def _port_weights(snapshot, cell):
    if cell in snapshot._port_positions:
        return {cell: 1.0}
    region = snapshot._regions[snapshot._owner._membership[cell]]
    component = region.components[cell]
    weights = _region_readout_row(region, cell)[:-1]
    return {port: weight for port, weight in zip(region.ports, weights)
            if weight and region.components[port] == component}


def _coarse_adjoint(snapshot, forcing, needed, cold):
    if not forcing or not needed:
        return {}
    if snapshot._elimination_rank is not None:
        cache = snapshot._elimination or cold.get("sparse")
        if cache is None:
            cache, _, _ = _sparse_coarse(
                snapshot._port_positions, snapshot._regions, snapshot._bridges,
                snapshot._elimination_rank, None, (), (),
            )
            cold["sparse"] = cache
        return resolve_adjoint(cache.steps, snapshot._elimination_rank, forcing, sorted(needed))
    values = {}
    for cells, block in snapshot._coarse_blocks.items():
        if not forcing.keys().isdisjoint(cells) and not needed.isdisjoint(cells):
            operator = block.operator or cold.get(cells)
            if operator is None:
                operator = _factor_coarse(cells, block.bridges, block.regions)
                cold[cells] = operator
            rhs = np.asarray([forcing.get(cell, 0.0) for cell in cells])
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                coordinates = (operator.right @ rhs) / operator.singular
                response = operator.right.T @ (coordinates / operator.singular)
            if not np.isfinite(response).all():
                raise FloatingPointError("nonfinite coarse adjoint")
            values.update(zip(cells, response))
    return values


def _adjoint_values(snapshot, forcing, endpoints, local_roots, coarse_roots):
    membership = snapshot._owner._membership
    regions = snapshot._regions
    ports = snapshot._port_positions
    weights = {node: _port_weights(snapshot, node) for node in endpoints}
    needed = {port for transfer in weights.values() for port in transfer}
    interior_groups = {(membership[node], regions[membership[node]].components[node])
                       for node in endpoints if node not in ports}
    local_forcing, coarse_forcing = {}, {}
    for cell, value in forcing.items():
        for port, weight in _port_weights(snapshot, cell).items():
            coarse_forcing[port] = coarse_forcing.get(port, 0.0) + weight * value
        number = membership[cell]
        group = number, regions[number].components[cell]
        if cell not in ports and group in interior_groups:
            local_forcing.setdefault(group, {})[cell] = value

    coarse = _coarse_adjoint(snapshot, coarse_forcing, needed, coarse_roots)
    local = {}
    for (number, component), values in local_forcing.items():
        group = number, component
        if group not in local_roots:
            local_roots[group] = _local_root(regions[number], snapshot._owner._ridge, component)
        interior, root = local_roots[group]
        rhs = np.asarray([values.get(node, 0.0) for node in interior])
        response = np.linalg.solve(root, np.linalg.solve(root.T, rhs))
        if not np.isfinite(response).all():
            raise FloatingPointError("nonfinite local adjoint")
        local.update(zip(interior, response))
    answer = {node: local.get(node, 0.0) + sum(weight * coarse.get(port, 0.0)
                                             for port, weight in transfer.items())
              for node, transfer in weights.items()}
    if any(not isfinite(value) for value in answer.values()):
        raise FloatingPointError("nonfinite adjoint endpoint")
    return answer


def _source_projection(row, values):
    return values.get(row.target, 0.0) - values.get(row.source, 0.0)


def _source_direction(row):
    return {row.target: 1.0} if row.source is None else {row.target: 1.0, row.source: -1.0}


def _apply_source_prediction(snapshot, cell, edits):
    rows = [old for _, old, _ in edits]
    endpoints = {node for row in rows for node in row._endpoints()}
    weights = np.asarray([0.0 if new is None else new.weight for _, _, new in edits])
    change = weights - np.asarray([row.weight for row in rows])
    weight_edits = np.flatnonzero(change)
    baseline_endpoints = {cell}
    for i in weight_edits:
        baseline_endpoints.update(rows[i]._endpoints())
    ordered = sorted(baseline_endpoints)
    current = dict(zip(ordered, snapshot.read_many(ordered)))
    local_roots, coarse_roots = {}, {}
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        u_query = _adjoint_values(snapshot, {cell: 1.0}, endpoints, local_roots, coarse_roots)
        gain = np.asarray([_source_projection(row, u_query) for row in rows])
        residual = np.asarray([
            (0.0 if new is None else new.weight * (new.value - old.value))
            + (change[i] * (old.value - _source_projection(old, current)) if change[i] else 0.0)
            for i, (_, old, new) in enumerate(edits)
        ])
        if len(weight_edits):
            response = []
            for i in weight_edits:
                values = _adjoint_values(snapshot, _source_direction(rows[i]), endpoints,
                                         local_roots, coarse_roots)
                response.append([_source_projection(row, values) for row in rows])
            leverage = change[weight_edits, None] * np.asarray(response)
            system = np.eye(len(weight_edits)) + leverage[:, weight_edits]
            unchanged = change == 0
            rhs = residual[weight_edits] - leverage[:, unchanged] @ residual[unchanged]
            singular = np.linalg.svd(system, compute_uv=False)
            # Downdates subtract nearly equal terms before the solve sees them.
            formation_floor = sqrt(np.finfo(float).eps) * len(weight_edits) * max(1, singular[0])
            if singular[-1] <= formation_floor:
                raise FloatingPointError("source edits did not resolve a correction system")
            residual[weight_edits] = np.linalg.solve(system, rhs)
        prediction = float(current[cell] + gain @ residual)
    if not isfinite(prediction):
        raise FloatingPointError("nonfinite source edit prediction")
    return prediction


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
    _source_magnitude: float
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
    def membership(self):
        """Return the cell-to-region assignment of this revision."""
        return self._owner._membership

    @property
    def elimination_order(self):
        """Return this branch's sparse ordering, or None for the default SVD."""
        return None if self._elimination_rank is None else tuple(self._elimination_rank)

    def read(self, cell, *, value_first=False, retry_failed=False):
        """Read an anchored estimate and recover the boundary values it needs."""
        return self.read_many((cell,), value_first=value_first, retry_failed=retry_failed)[0]

    def support_witness(self, cell):
        """Return a named source path to an anchor, or None for an unsupported cell."""
        cell = LatentField._checked_cell(cell, len(self._owner._membership))
        return support_witness(self, cell)

    def source_gains(self, cell, source_ids):
        """Return fixed-weight value sensitivities in request order, or None if unsupported."""
        cell = LatentField._checked_cell(cell, len(self._owner._membership))
        observations = tuple((name, self._sources[name]) for name in source_ids)
        if self._readout(cell) is None:
            return None
        if not observations:
            return ()

        group = self._component_group(cell)
        relevant = (row for _, row in observations if self._source_in_group(row, group))
        endpoints = {node for row in relevant for node in row._endpoints()}
        if not endpoints:
            return tuple((name, 0.0) for name, _ in observations)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            values = _adjoint_values(self, {cell: 1.0}, endpoints, {}, {})
            gains = tuple((name, float(row.weight * _source_projection(row, values)))
                          for name, row in observations)
        if any(not isfinite(gain) for _, gain in gains):
            raise FloatingPointError("nonfinite source gain")
        return gains

    def _component_group(self, cell):
        region = self._regions[self._owner._membership[cell]]
        component = region.components[cell]
        return self._support_graph.groups.get(component, (component,))

    def _source_in_group(self, row, group):
        region = self._regions[self._owner._membership[row.target]]
        return region.components[row.target] in group

    def predict_source_edits(self, cell, changes):
        """Predict one read after value/weight changes or deletions of existing sources."""
        cell = LatentField._checked_cell(cell, len(self._owner._membership))
        group = self._component_group(cell)
        edits = []
        for name, replacement in changes.items():
            previous = self._sources[name]
            if replacement is not None:
                _validate_observation(self._owner._cells, name, replacement)
                if replacement._endpoints() != previous._endpoints():
                    raise ValueError("source edit prediction requires unchanged endpoints")
            if previous != replacement and self._source_in_group(previous, group):
                edits.append((name, previous, replacement))
        if self._readout(cell) is None:
            return None
        removed = {name for name, _, new in edits if new is None}
        if removed and not supported_after_removal(self, cell, removed):
            return None
        return _apply_source_prediction(self, cell, edits) if edits else self.read(cell)

    def preview_source_patch(self, cell, changes, *, max_cells):
        """Preview an edited read with a bounded local Dirichlet solve."""
        cell = LatentField._checked_cell(cell, self._owner._cells)
        max_cells = index(max_cells)
        if max_cells < 1:
            raise ValueError("source patch needs a positive cell cap")
        staged = self._owner._stage_source_edits(self, changes)
        return preview_source_patch(self, cell, staged, max_cells)

    def _is_supported(self, cell):
        region = self._regions[self._owner._membership[cell]]
        component = region.components[cell]
        return component in region.anchored or component in self._supported_ports

    def _readout(self, cell, *, work=None):
        cell = LatentField._checked_cell(cell, len(self._owner._membership))
        if not self._is_supported(cell):
            return None
        region = self._regions[self._owner._membership[cell]]
        component = region.components[cell]
        if cell in self._port_positions:
            return (cell,), None
        coeffs = (_region_readout_row(region, cell) if work is None else
                  _region_readout_row(region, cell, work=work))
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

    def read_many(self, cells, *, value_first=False, retry_failed=False):
        """Read in request order, optionally retrying numerical failures in the other order."""
        if type(value_first) is not bool:
            raise ValueError("value_first must be a boolean")
        if type(retry_failed) is not bool:
            raise ValueError("retry_failed must be a boolean")
        if retry_failed:
            return self._read_retry_batch(cells, value_first=value_first)
        if value_first:
            targets = self._read_targets(cells)
            values = self._read_value_targets(targets, self._read_ports)
            return tuple(None if cell is None else values[cell] for cell in targets)
        return self._read_batch(cells)[0]

    def _read_targets(self, cells):
        requested = [LatentField._checked_cell(cell, self._owner._cells) for cell in cells]
        return [cell if self._is_supported(cell) else None for cell in requested]

    def _read_value_targets(self, targets, recover, *, failed=None):
        coarse, plans, ordered, pending = {}, {}, [], []

        def schedule(cell):
            if cell in self._port_positions:
                coarse[cell] = None
                return
            root = self._regions[self._owner._membership[cell]]
            region, row = _stored_readout(root, cell)
            component = region.components[cell]
            columns = [column for column, port in enumerate(region.ports)
                       if region.components[port] == component and row[column] != 0]
            dependencies = tuple(region.ports[column] for column in columns)
            plans[cell] = dependencies, row[[*columns, len(region.ports)]]
            pending.append((cell, iter(dependencies)))

        for target in targets:
            if target is None or target in plans or target in coarse:
                continue
            schedule(target)
            # An internal port's definition belongs to an ancestor of its consumer.
            while pending:
                cell, dependencies = pending[-1]
                dependency = next(dependencies, None)
                if dependency is None:
                    pending.pop()
                    ordered.append(cell)
                elif dependency not in plans and dependency not in coarse:
                    schedule(dependency)

        return _evaluate_regional_rows(ordered, plans, recover(coarse), failed=failed)

    def _read_affine_targets(self, targets, recover, *, failed):
        coarse, plans = {}, {}
        for cell in targets:
            try:
                ports, row = self._readout(cell)
            except FloatingPointError:
                continue
            coarse.update(dict.fromkeys(ports))
            if row is not None:
                plans[cell] = ports, row
        return _evaluate_regional_rows(plans, plans, recover(coarse), failed=failed)

    def _read_retry_batch(self, cells, *, value_first):
        targets = self._read_targets(cells)
        unique = dict.fromkeys(cell for cell in targets if cell is not None)
        coarse, failed = {}, set()

        def recover(ports):
            missing = [cell for cell in ports if cell not in coarse and cell not in failed]
            if missing:
                if self._elimination is None:
                    coarse.update(self._read_ports(missing))
                else:
                    resolve_elimination(self._elimination.steps, missing, solved=coarse, failed=failed)
            return coarse

        if value_first:
            values = self._read_value_targets(unique, recover, failed=failed)
        else:
            values = self._read_affine_targets(unique, recover, failed=failed)
        missing = [cell for cell in unique if cell not in values]
        if missing:
            if value_first:
                alternate = self._read_affine_targets(missing, recover, failed=failed)
            else:
                alternate = self._read_value_targets(missing, recover, failed=failed)
            for cell in missing:
                if cell not in alternate:
                    raise FloatingPointError(f"regional read unavailable for cell {cell} in both evaluation orders")
                values[cell] = alternate[cell]
        return tuple(None if cell is None else values[cell] for cell in targets)

    def _read_batch(self, cells, *, work=None):
        plans, by_cell = [], {}
        for cell in map(index, cells):
            if cell not in by_cell:
                by_cell[cell] = self._readout(cell) if work is None else self._readout(cell, work=work)
            plans.append(by_cell[cell])
        values = None
        if self._elimination is not None:
            requested = dict.fromkeys(port for plan in plans if plan is not None for port in plan[0])
            values = self._read_ports(requested)
            if work is not None:
                work[0] += sum(len(self._elimination.steps[port].readout) for port in values)
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
            if work is not None:
                work[0] += 1 if coeffs is None else len(coeffs)
            if coeffs is None:
                reads.append(float(inputs[0]))
                continue
            value = float(coeffs[-1] + coeffs[:-1] @ inputs)
            if not isfinite(value):
                raise FloatingPointError("nonfinite reconstructed interior coordinate")
            reads.append(value)
        return tuple(reads), len(values) if values is not None else 0


def _evaluate_regional_rows(ordered, plans, values, *, failed):
    isolate = failed is not None
    if isolate:
        values, failed = dict(values), set(failed)
    with np.errstate(over="raise", invalid="raise"):
        for cell in ordered:
            dependencies, row = plans[cell]
            if isolate and any(port in failed for port in dependencies):
                failed.add(cell)
                continue
            try:
                value = float(row[-1] + row[:-1] @ [values[port] for port in dependencies])
                if not isfinite(value):
                    raise FloatingPointError("nonfinite reconstructed interior coordinate")
            except FloatingPointError:
                if not isolate:
                    raise
                failed.add(cell)
            else:
                values[cell] = value
    return values


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
                  bridge_edits=(), source_magnitude=0.0):
        if base is not None:
            source_magnitude = max(source_magnitude, base._source_magnitude)
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
            support.supported, support, source_magnitude, rebuilt_regions=tuple(rebuilt),
            compacted_regions=tuple(compacted),
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
        magnitude = max((abs(row.value) for row in sources.values()), default=0.0)
        return self._snapshot(MappingProxyType(sources), MappingProxyType(bridges), factors,
                              range(len(factors)), source_magnitude=magnitude)

    def _stage_source_edits(self, base, changes):
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
        return staged

    def revise(self, base, changes):
        """Apply a named source batch. Exposed ports persist until compacted."""
        if base._owner is not self:
            raise ValueError("relative snapshot belongs to a different hierarchy")
        staged = self._stage_source_edits(base, changes)
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
            ports = set(previous.ports).union(promotions.get(region, ()))
            factors[region] = _revise_region(previous, local_edits.get(region, {}), ports, self._ridge)
        magnitude = max((abs(row.value) for _, _, row in staged if row is not None), default=0.0)
        return self._snapshot(MappingProxyType(sources), MappingProxyType(bridges), factors, rebuilt,
                              base=base, bridge_edits=bridge_edits, source_magnitude=magnitude)

    def merge_regions(self, base, first, second, *, retain_children=False, lazy_readout=False):
        """Fuse two regions, optionally keeping child operators and deferring their readout."""
        if type(retain_children) is not bool or type(lazy_readout) is not bool:
            raise ValueError("retain_children and lazy_readout must be booleans")
        if lazy_readout and not retain_children:
            raise ValueError("lazy readout requires retained children")
        if base._owner is not self:
            raise ValueError("relative snapshot belongs to a different hierarchy")
        first, second = sorted((index(first), index(second)))
        if not 0 <= first < second < len(base._regions):
            raise ValueError("merge needs two distinct existing region labels")

        incident = base._support_graph.incident
        left = incident.get(first, frozenset())
        right = incident.get(second, frozenset())
        internalized = {name: base._bridges[name] for name in left & right}
        active_ports = {cell for name in left ^ right for cell in base._bridges[name]._endpoints()
                        if self._membership[cell] in (first, second)}
        merged = _merged_region((base._regions[first], base._regions[second]), internalized, active_ports,
                                retain_children=retain_children, lazy_readout=lazy_readout)
        regions = list(base._regions)
        regions[first] = merged
        del regions[second]
        bridges = {name: source for name, source in base._bridges.items() if name not in internalized}
        membership = tuple(first if label == second else label - int(label > second)
                           for label in self._membership)
        owner = RelativeObservationHierarchy(membership, ridge=self._ridge, elimination_order=base.elimination_order)
        # Region IDs and incidence changed. Index the new topology from scratch.
        return owner._snapshot(base._sources, MappingProxyType(bridges), regions, (first,),
                               source_magnitude=base._source_magnitude)

    def split_region(self, base, region):
        """Expose a fusion's current children and their crossing measurements."""
        if base._owner is not self:
            raise ValueError("relative snapshot belongs to a different hierarchy")
        region = index(region)
        if not 0 <= region < len(base._regions) or base._regions[region].fusion is None:
            raise ValueError("split needs a region with retained children")
        fusion = base._regions[region].fusion
        left, right = fusion.children
        regions = list(base._regions)
        regions[region] = left
        new_label = len(regions)
        regions.append(right)
        membership = tuple(new_label if cell in right.positions else label
                           for cell, label in enumerate(self._membership))
        bridges = MappingProxyType(dict(base._bridges) | fusion.bridges)
        owner = RelativeObservationHierarchy(membership, ridge=self._ridge, elimination_order=base.elimination_order)
        return owner._snapshot(base._sources, bridges, regions, (), source_magnitude=base._source_magnitude)

    def repartition(self, base, membership):
        """Rebuild a partition from the current measurements, including after a merge."""
        if base._owner is not self:
            raise ValueError("relative snapshot belongs to a different hierarchy")
        membership = tuple(map(index, membership))
        if len(membership) != self._cells:
            raise ValueError("partition must contain every cell")
        if membership == self._membership:
            return base
        owner = RelativeObservationHierarchy(membership, ridge=self._ridge, elimination_order=base.elimination_order)
        rebuilt = owner.initialize(base._sources)
        return replace(rebuilt, _sources=base._sources, _source_magnitude=base._source_magnitude)

    def reorder(self, base, elimination_order):
        """Branch from the same sources with a different sparse elimination order."""
        if base._owner is not self or base._elimination_rank is None:
            raise ValueError("reordering needs a sparse snapshot from this hierarchy")
        rank = self._rank(elimination_order)
        if rank == base._elimination_rank:
            return base
        return self._snapshot(base._sources, base._bridges, base._regions, (), base=base, new_rank=rank)

    def compact(self, base, *, max_regions=1, protected_cells=(), tolerance=1e-10,
                value_first=False, retry_failed=False):
        """Retire inactive ports in a bounded number of regions of the selected branch."""
        if base._owner is not self:
            raise ValueError("relative snapshot belongs to a different hierarchy")
        max_regions, tolerance = index(max_regions), float(tolerance)
        if max_regions < 1 or not isfinite(tolerance) or tolerance < 0:
            raise ValueError("compaction needs a positive region cap and finite nonnegative tolerance")
        if type(value_first) is not bool:
            raise ValueError("value_first must be a boolean")
        if type(retry_failed) is not bool:
            raise ValueError("retry_failed must be a boolean")
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
        before_values = base.read_many(protected, value_first=value_first, retry_failed=retry_failed)
        after_values = compacted.read_many(protected, value_first=value_first, retry_failed=retry_failed)
        for cell, before, after in zip(protected, before_values, after_values):
            if before is None or after is None:
                agrees = before is after
            else:
                agrees = abs(before - after) <= tolerance
            if not agrees:
                raise RuntimeError(f"compaction changed protected read {cell}")
        return compacted
