"""Local Dirichlet previews using the edited source rows of a snapshot."""

from dataclasses import dataclass
from itertools import chain
from math import isfinite, sqrt

import numpy as np

from experiments.memory.relative_observations import RelativeObservation


@dataclass(frozen=True)
class RelativePatchPreview:
    """An anchored estimate or an unresolved or unsupported patch."""

    cells: tuple[int, ...]
    anchored: bool | None
    value: float | None = None
    boundary_error: float | None = None


def _patch_rows(snapshot, cell, staged, max_cells):
    membership = snapshot._owner._membership
    replaced = {name for name, _, _ in staged}
    incoming = {}
    for name, _, row in staged:
        if row is not None:
            for endpoint in row._endpoints():
                incoming.setdefault(endpoint, {})[name] = row

    regional_rows, active = {}, {}
    cells, included = [cell], {cell}
    for node in cells:
        region = membership[node]
        if region not in regional_rows:
            index = {}
            local = snapshot._regions[region]._sources
            bridges = snapshot._support_graph.incident.get(region, ())
            bridge_rows = ((name, snapshot._bridges[name]) for name in bridges)
            for name, row in chain(local.items(), bridge_rows):
                if name not in replaced:
                    for endpoint in row._endpoints():
                        if membership[endpoint] == region:
                            index.setdefault(endpoint, {})[name] = row
            regional_rows[region] = index

        incident = regional_rows[region].get(node, {}) | incoming.get(node, {})
        for name in sorted(incident):
            row = incident[name]
            active[name] = row
            for endpoint in row._endpoints():
                if endpoint not in included and len(cells) < max_cells:
                    included.add(endpoint)
                    cells.append(endpoint)
    return tuple(cells), active


def _solve_patch(cells, rows: dict[str, RelativeObservation], ridge, envelope):
    positions = {cell: i for i, cell in enumerate(cells)}
    boundary = any(node not in positions for row in rows.values() for node in row._endpoints())
    if not any(row.source is None for row in rows.values()):
        return RelativePatchPreview(cells, None if boundary else False)

    size = len(cells)
    matrix = np.zeros((len(rows) + size, size))
    rhs = np.zeros((len(matrix), 2))
    for i, name in enumerate(sorted(rows)):
        row = rows[name]
        scale = sqrt(row.weight)
        rhs[i, 0] = scale * row.value
        if row.target in positions:
            matrix[i, positions[row.target]] = scale
        else:
            rhs[i, 1] = -scale
        if row.source is not None:
            if row.source in positions:
                matrix[i, positions[row.source]] = -scale
            else:
                rhs[i, 1] = scale
    columns = np.arange(size)
    matrix[len(rows) + columns, columns] = sqrt(ridge)
    if not np.isfinite(rhs).all():
        raise FloatingPointError("nonfinite weighted patch observation")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        solution, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    if rank != size or not np.isfinite(solution).all():
        raise FloatingPointError("unresolved source patch solve")
    value, influence = map(float, solution[0])
    error = envelope * abs(influence) if boundary else 0.0
    # An open connected patch has positive influence at the query.
    if not isfinite(error) or (boundary and envelope > 0 and error == 0):
        raise FloatingPointError("unresolved source patch boundary error")
    return RelativePatchPreview(cells, True, value, error)


def preview_source_patch(snapshot, cell, staged, max_cells):
    cells, rows = _patch_rows(snapshot, cell, staged, max_cells)
    magnitude = max((abs(row.value) for _, _, row in staged if row is not None), default=0.0)
    envelope = snapshot._owner._cells * max(snapshot._source_magnitude, magnitude)
    return _solve_patch(cells, rows, snapshot._owner._ridge, envelope)
