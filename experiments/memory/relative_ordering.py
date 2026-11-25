"""Bounded symbolic proposals for sparse relative-observation layouts."""

from collections import Counter
from itertools import combinations
from operator import index

from experiments.memory.relative_hierarchy import _Components


def _profile(snapshot, workload, max_ports):
    hierarchy = snapshot._owner
    ports = set(snapshot._port_positions)
    if len(ports) > max_ports:
        return None
    bridges = {row._endpoints() for row in snapshot._bridges.values()}
    local_links = set()

    def component(cell):
        region = snapshot._regions[hierarchy._membership[cell]]
        return region.components[cell]

    for edits, _ in workload:
        for row in edits.values():
            if row is None or row.source is None:
                continue
            if hierarchy._region_of(row) is None:
                endpoints = row._endpoints()
                bridges.add(endpoints)
                ports.update(endpoints)
                if len(ports) > max_ports:
                    return None
            else:
                local_links.add((component(row.source), component(row.target)))

    labels = {component(port) for port in ports}
    labels.update(label for link in local_links for label in link)
    components = _Components(labels)
    for first, second in local_links:
        components.join(first, second)
    grouped = {}
    for port in sorted(ports):
        grouped.setdefault(components.root(component(port)), []).append(port)
    graph = {port: set() for port in ports}
    for members in grouped.values():
        for first, second in combinations(members, 2):
            graph[first].add(second)
            graph[second].add(first)
    for first, second in bridges:
        graph[first].add(second)
        graph[second].add(first)

    def scope(cell):
        label = component(cell)
        return grouped.get(components.root(label), ()) if label in labels else ()

    current, activity = {}, Counter()
    for edits, queries in workload:
        for name, row in edits.items():
            previous = current.get(name, snapshot._sources.get(name))
            if previous == row:
                continue
            current[name] = row
            touched = set()
            for observation in (previous, row):
                if observation is None:
                    continue
                endpoints = observation._endpoints()
                if hierarchy._region_of(observation) is None:
                    touched.update(endpoints)
                else:
                    for cell in endpoints:
                        touched.update(scope(cell))
            activity.update(touched)
        for cell in map(index, queries):
            activity.update((cell,) if cell in ports else scope(cell))
    return graph, activity


def _minimum_fill(graph, activity):
    remaining = {cell: set(neighbors) for cell, neighbors in graph.items()}
    order = []

    def key(cell):
        neighbors = remaining[cell]
        fills = sum(second not in remaining[first] for first, second in combinations(neighbors, 2))
        return fills, activity.get(cell, 0), len(neighbors), cell

    while remaining:
        cell = min(remaining, key=key)
        neighbors = remaining.pop(cell)
        for first in neighbors:
            remaining[first].remove(cell)
            remaining[first].update(neighbors - {first})
        order.append(cell)
    return iter(order)


def propose_orders(snapshot, workload, max_ports):
    """Offer geometry-only and workload-sensitive orders on the same union graph."""
    profile = _profile(snapshot, workload, max_ports)
    if profile is None:
        return
    graph, activity = profile
    current = snapshot.elimination_order
    seen = {current}
    for priorities in (activity, {}):
        active_order = _minimum_fill(graph, priorities)
        candidate = tuple(next(active_order) if cell in graph else cell for cell in current)
        if candidate not in seen:
            seen.add(candidate)
            yield candidate
