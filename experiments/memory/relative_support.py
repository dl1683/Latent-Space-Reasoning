"""Snapshot-owned anchor reachability over bridged observation components."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, eq=False)
class BridgeSupport:
    edges: Mapping[str, tuple[int, int]]
    incident: Mapping[int, frozenset[str]]
    neighbors: Mapping[int, Mapping[int, int]]
    groups: Mapping[int, frozenset[int]]
    supported: frozenset[int]


_EMPTY = MappingProxyType({})
_EMPTY_SUPPORT = BridgeSupport(_EMPTY, _EMPTY, _EMPTY, _EMPTY, frozenset())


def _updated_edges(previous, replacements, membership):
    if not replacements:
        return previous.edges, previous.incident, Counter()
    edges, incident = dict(previous.edges), dict(previous.incident)
    changed_incident = {}
    delta = Counter()

    def regions(edge):
        return {membership[cell] for cell in edge} if edge is not None else set()

    def region_names(region):
        if region not in changed_incident:
            changed_incident[region] = set(incident.get(region, ()))
        return changed_incident[region]

    for name, replacement in replacements.items():
        old = edges.pop(name, None)
        if old is not None:
            delta[old] -= 1
        if replacement is not None:
            edges[name] = replacement
            delta[replacement] += 1
        before, after = regions(old), regions(replacement)
        for region in before - after:
            region_names(region).remove(name)
        for region in after - before:
            region_names(region).add(name)

    for region, names in changed_incident.items():
        if names:
            incident[region] = frozenset(names)
        else:
            del incident[region]
    return MappingProxyType(edges), MappingProxyType(incident), delta


def _updated_neighbors(previous, delta, dirty):
    edited = {}
    for (first, second), change in delta.items():
        if change == 0:
            continue
        count = previous.get(first, {}).get(second, 0)
        replacement = count + change
        if bool(count) != bool(replacement):
            dirty.update((first, second))
        for node, other in ((first, second), (second, first)):
            if node not in edited:
                edited[node] = dict(previous.get(node, {}))
            if replacement:
                edited[node][other] = replacement
            else:
                del edited[node][other]
    if not edited:
        return previous
    neighbors = dict(previous)
    for node, links in edited.items():
        if links:
            neighbors[node] = MappingProxyType(links)
        else:
            del neighbors[node]
    return MappingProxyType(neighbors)


def _reconnect(previous, neighbors, seeds, anchored):
    if not seeds:
        return previous.groups, previous.supported, ()
    changed = set(seeds)
    affected_groups = {previous.groups[node] for node in seeds if node in previous.groups}
    for group in affected_groups:
        changed.update(group)
    groups = dict(previous.groups)
    for node in changed:
        groups.pop(node, None)
    supported = set(previous.supported - changed)
    visited = set()
    for start in changed:
        if start in visited or start not in neighbors:
            continue
        reached, queue = {start}, [start]
        for node in queue:
            adjacent = neighbors[node].keys() - reached
            reached.update(adjacent)
            queue.extend(adjacent)
        visited.update(reached)
        group = frozenset(reached)
        groups.update(dict.fromkeys(group, group))
        if any(anchored(node) for node in group):
            supported.update(group)
    return MappingProxyType(groups), frozenset(supported), tuple(sorted(visited))


def update_bridge_support(membership, regions, bridges, base, rebuilt, bridge_edits):
    """Reproject named bridge edges and revisit only affected connected groups."""
    previous = base._support_graph if base is not None else _EMPTY_SUPPORT
    names = set(bridge_edits if base is not None else bridges)
    if base is not None:
        for number in rebuilt:
            before, after = base._regions[number], regions[number]
            if before.components != after.components or before.anchored != after.anchored:
                names.update(previous.incident.get(number, ()))

    def component(cell):
        return regions[membership[cell]].components[cell]

    def anchored(node):
        return node in regions[membership[node]].anchored

    replacements, dirty = {}, set()
    for name in names:
        old = previous.edges.get(name)
        bridge = bridges.get(name)
        new = None if bridge is None else tuple(sorted((component(bridge.source), component(bridge.target))))
        if old != new:
            replacements[name] = new
        if base is not None:
            for node in new or ():
                if anchored(node) != (node in base._regions[membership[node]].anchored):
                    dirty.add(node)
    if not replacements and not dirty:
        return previous, ()

    edges, incident, delta = _updated_edges(previous, replacements, membership)
    neighbors = _updated_neighbors(previous.neighbors, delta, dirty)
    groups, supported, visited = _reconnect(previous, neighbors, dirty, anchored)
    return BridgeSupport(edges, incident, neighbors, groups, supported), visited
