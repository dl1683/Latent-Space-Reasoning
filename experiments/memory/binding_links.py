"""A query-independent bank of paired continuous directions from typed sources."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from operator import index
from types import MappingProxyType

import numpy as np

from experiments.memory.binding_memory import BindingCapacityError
from experiments.memory.latent_feedback import _real_array


@dataclass(frozen=True)
class BindingLinkStep:
    source: str
    entry_ordinal: int
    exit_ordinal: int
    exit_source: str | None = None
    key_ordinals: tuple[int, int] | None = None

    @property
    def sources(self):
        return (self.source,) if self.exit_source is None else (self.source, self.exit_source)


@dataclass(frozen=True)
class BindingLinkWitness:
    outcome: str
    link_source: str
    entry_ordinal: int
    exit_ordinal: int
    terminal_source: str
    terminal_ordinal: int
    strength: float
    prefix: tuple[BindingLinkStep, ...] = ()
    exit_source: str | None = None
    key_ordinals: tuple[int, int] | None = None


@dataclass(frozen=True)
class BindingLinkCondition:
    """A late interval test on the terminal latent direction."""

    axis: tuple[float, ...]
    lower: float
    upper: float
    uncertainty: float = 0.

    def __post_init__(self):
        axis = _real_array(self.axis)
        if axis.ndim != 1 or not axis.size:
            raise ValueError("role-link condition requires a nonzero vector axis")
        direction = _unit_direction(axis)
        if direction is None:
            raise ValueError("role-link condition requires a nonzero vector axis")
        lower, upper, uncertainty = map(float, (self.lower, self.upper, self.uncertainty))
        if not -1. <= lower <= upper <= 1. or not 0. <= uncertainty <= 2.:
            raise ValueError("role-link condition bounds or uncertainty are out of range")
        object.__setattr__(self, "axis", tuple(map(float, direction)))
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "uncertainty", uncertainty)


@dataclass(frozen=True)
class BindingLinkRead:
    status: str
    witnesses: tuple[BindingLinkWitness, ...] = ()
    work: int = 0
    open_sources: int = 0
    unmatched_links: int = 0
    marginal: bool = False
    terminal_dots: int = 0
    outcomes: tuple[str, ...] = ()
    outside_outcomes: tuple[str, ...] = ()
    hops: int = 1
    transports: int = 0
    condition: BindingLinkCondition | None = None
    condition_dots: int = 0
    inherited_open: bool = False
    bridge_pending: bool = False

    @property
    def answer(self):
        return self.witnesses[0].outcome if self.status == "supported" else None


def _role_hops(hops):
    hops = index(hops)
    if hops < 1:
        raise ValueError("role-link hop count must be positive")
    return hops


def _unit_direction(direction):
    peak = np.max(np.abs(direction))
    if peak == 0.:
        return None
    with np.errstate(under="ignore"):
        scaled = direction / peak
        return scaled / np.linalg.norm(scaled)


@dataclass(frozen=True, slots=True)
class _RoleLink:
    step: BindingLinkStep
    entry: bytes
    exit: bytes
    strength: float
    reflector: bytes | None = None

    @classmethod
    def from_ends(cls, step, left, right, strength, transport):
        reflector = None
        if transport != "fixed":
            start, end = np.frombuffer(left, dtype="<f8"), np.frombuffer(right, dtype="<f8")
            with np.errstate(under="ignore"):
                normal = start - end if transport == "reflection" else start + end
            if transport == "reflection" or np.max(np.abs(normal)) > 1e-12:
                axis = _unit_direction(normal)
                if axis is not None:
                    reflector = np.asarray(axis, dtype="<f8").tobytes()
        return cls(step, left, right, strength, reflector)

    def advance(self, direction, transport):
        if transport == "fixed":
            return np.frombuffer(self.exit, dtype="<f8")
        if self.reflector is None:
            return direction
        if transport == "rotation":
            entry = np.frombuffer(self.entry, dtype="<f8")
            direction = direction - 2. * (direction @ entry) * entry
        reflector = np.frombuffer(self.reflector, dtype="<f8")
        return _unit_direction(direction - 2. * (direction @ reflector) * reflector)


@dataclass(frozen=True, slots=True)
class _Path:
    direction: np.ndarray
    steps: tuple[BindingLinkStep, ...] = ()
    used: frozenset[str] = frozenset()
    strength: float = 1.


@dataclass(frozen=True, slots=True)
class _BridgeFragment:
    ordinal: int
    key_ordinal: int
    strength: float
    direction: bytes
    key: bytes


def _bridge_fragment(records, roles, key):
    anchors, ends = [], []
    for ordinal, (role, grade, raw, _) in enumerate(records):
        if role == key:
            anchors.append((ordinal, grade, raw))
        elif role in roles:
            ends.append((ordinal, role, grade, raw))
    fragment = None
    if len(anchors) == len(ends) == 1:
        ordinal, role, grade, raw = ends[0]
        key_ordinal, key_grade, anchor = anchors[0]
        fragment = role, _BridgeFragment(ordinal, key_ordinal, min(grade, key_grade), raw, anchor)
    return bool(anchors), fragment


def _key_match(entry, exit, floor, uncertainty):
    left, right = np.frombuffer(entry, dtype="<f8"), np.frombuffer(exit, dtype="<f8")
    score = float(np.clip(left @ right, -1., 1.))
    if score < floor - uncertainty:
        return None
    return score >= floor and (not uncertainty or score > floor + uncertainty)


def _cross_link(source, target, entry, exit, transport):
    step = BindingLinkStep(source, entry.ordinal, exit.ordinal, target, (entry.key_ordinal, exit.key_ordinal))
    return _RoleLink.from_ends(step, entry.direction, exit.direction, min(entry.strength, exit.strength), transport)


def _update_candidates(rows, columns, edges, entries, exits, changed_entries, changed_exits, links, limit):
    """Replace incident candidate relations and find entries touched through an exit."""
    new_entries, new_exits = changed_entries & entries.keys(), changed_exits & exits.keys()
    width = len(next(iter(entries.values()))) // 8 if entries else 0
    comparisons = len(new_entries) * len(exits) + len(new_exits) * (len(entries) - len(new_entries))
    if width * comparisons > limit:
        raise BindingCapacityError("bridge incident comparison limit exceeded")

    row_edits, col_edits = {}, {}

    def edit(source, target, match):
        nonlocal edges
        if source not in row_edits:
            row_edits[source] = dict(rows.get(source, {}))
        if target not in col_edits:
            col_edits[target] = set(columns.get(target, ()))
        row, col = row_edits[source], col_edits[target]
        if match is None:
            if target in row:
                edges -= 1
                del row[target]
                col.discard(source)
        else:
            edges += int(target not in row)
            row[target] = match
            col.add(source)

    for source in changed_entries:
        for target in rows.get(source, ()):
            edit(source, target, None)
    for target in changed_exits:
        for entry in columns.get(target, ()):
            edit(entry, target, None)
    for source in new_entries:
        for target, exit in exits.items():
            match = _key_match(entries[source], exit, links.floor, links.uncertainty)
            if match is not None:
                edit(source, target, match)
    for target in new_exits:
        for source, entry in entries.items():
            if source not in new_entries:
                match = _key_match(entry, exits[target], links.floor, links.uncertainty)
                if match is not None:
                    edit(source, target, match)

    affected = set(row_edits)
    for target, col in col_edits.items():
        affected.update(columns.get(target, ()))
        affected.update(col)
    rows, columns = dict(rows), dict(columns)
    for source, row in row_edits.items():
        if row:
            rows[source] = MappingProxyType(row)
        else:
            rows.pop(source, None)
    for target, col in col_edits.items():
        if col:
            columns[target] = frozenset(col)
        else:
            columns.pop(target, None)
    return MappingProxyType(rows), MappingProxyType(columns), edges, affected, comparisons


def _stale_directions(stale, affected, before, after):
    updated = dict(stale)
    for source in affected:
        old, new = before.get(source), after.get(source)
        directions = frozenset(fragment.direction for fragment in (old, new) if fragment is not None)
        updated[source] = updated.get(source, frozenset()) | directions
    return MappingProxyType(updated)


@dataclass(frozen=True, eq=False)
class _BridgeIndex:
    entries: Mapping[str, _BridgeFragment]
    exits: Mapping[str, _BridgeFragment]
    rows: Mapping[str | bytes, Mapping[str | bytes, bool]]
    columns: Mapping[str | bytes, frozenset[str | bytes]]
    stale: Mapping[str, frozenset[bytes]]
    edges: int = 0

    @classmethod
    def build(cls, entries, exits, links):
        empty = MappingProxyType({})
        index = cls(empty, empty, empty, empty, empty)
        return index.revise(entries, exits, frozenset(entries) | frozenset(exits), links)

    def revise(self, entries, exits, changed, links):
        rows, columns, edges, affected, _ = _update_candidates(
            self.rows, self.columns, self.edges,
            {source: fragment.key for source, fragment in entries.items()},
            {source: fragment.key for source, fragment in exits.items()},
            changed, changed, links, links.compile_limit,
        )
        if edges + len(entries) + len(exits) > links.compile_limit:
            raise BindingCapacityError("retained bridge index limit exceeded")
        affected.update(changed & (self.entries.keys() | entries.keys()))
        stale = _stale_directions(self.stale, affected, self.entries, entries)
        if edges + len(entries) + len(exits) + sum(map(len, stale.values())) > links.compile_limit:
            raise BindingCapacityError("retained bridge obligations limit exceeded")
        return _BridgeIndex(MappingProxyType(entries), MappingProxyType(exits), rows, columns, stale, edges)

    def matches(self, links):
        bridges = dict(links.bridges)
        for source in self.stale:
            target = self.match_target(source)
            if target is not None:
                bridges[source, target] = _cross_link(source, target, self.entries[source],
                                                      self.exits[target], links.transport)
        return replace(links, bridges=MappingProxyType(bridges), bridge_pending=False,
                       bridge_index=replace(self, stale=MappingProxyType({})))

    def match_target(self, source):
        row = self.rows.get(source)
        if row and len(row) == 1:
            target, definite = next(iter(row.items()))
            if definite and len(self.columns[target]) == 1:
                return target

    def openings(self, bridges):
        installed = {source for source, _ in bridges}
        openings = {source: frozenset((entry.direction,)) for source, entry in self.entries.items()
                    if source not in installed}
        for source, directions in self.stale.items():
            openings[source] = openings.get(source, frozenset()) | directions
        return tuple((source, raw, source in self.stale)
                     for source, directions in sorted(openings.items()) for raw in sorted(directions))


def _revise_key_members(groups, before, after, changed):
    edits = {}
    for source in changed:
        old, new = before.get(source), after.get(source)
        if old is not None:
            if old.key not in edits:
                edits[old.key] = set(groups[old.key])
            edits[old.key].remove(source)
        if new is not None:
            if new.key not in edits:
                edits[new.key] = set(groups.get(new.key, ()))
            edits[new.key].add(source)
    members = dict(groups)
    for key, group in edits.items():
        if group:
            members[key] = frozenset(group)
        else:
            del members[key]
    return MappingProxyType(members), edits.keys()


@dataclass(frozen=True, eq=False)
class _FactoredBridgeIndex(_BridgeIndex):
    # In this index rows and columns are keyed by exact key bytes.
    entry_members: Mapping[bytes, frozenset[str]] = field(default_factory=lambda: MappingProxyType({}))
    exit_members: Mapping[bytes, frozenset[str]] = field(default_factory=lambda: MappingProxyType({}))
    comparisons: int = 0

    @classmethod
    def build(cls, entries, exits, links, remaining):
        empty = MappingProxyType({})
        index = cls(empty, empty, empty, empty, empty)
        return index.revise(entries, exits, entries.keys() | exits.keys(), links, comparison_limit=remaining)

    def revise(self, entries, exits, changed, links, *, comparison_limit=None):
        left, left_changed = _revise_key_members(self.entry_members, self.entries, entries, changed)
        right, right_changed = _revise_key_members(self.exit_members, self.exits, exits, changed)
        entry_changes = self.entry_members.keys() ^ left.keys()
        exit_changes = self.exit_members.keys() ^ right.keys()
        limit = links.compile_limit if comparison_limit is None else comparison_limit
        rows, columns, edges, _, comparisons = _update_candidates(
            self.rows, self.columns, self.edges, {key: key for key in left}, {key: key for key in right},
            entry_changes, exit_changes, links, limit,
        )

        # An entry-group membership edit can make existing exits non-unique for another entry.
        affected_right = set(right_changed)
        for key in left_changed:
            affected_right.update(self.rows.get(key, ()))
            affected_right.update(rows.get(key, ()))
        affected_left = set(left_changed)
        for key in affected_right:
            affected_left.update(self.columns.get(key, ()))
            affected_left.update(columns.get(key, ()))
        affected_sources = set()
        for key in affected_left:
            affected_sources.update(self.entry_members.get(key, ()))
            affected_sources.update(left.get(key, ()))
        stale = _stale_directions(self.stale, affected_sources, self.entries, entries)
        retained = len(entries) + len(exits) + len(left) + len(right) + edges + sum(map(len, stale.values()))
        if retained > links.compile_limit:
            raise BindingCapacityError("retained factored bridge index and obligations limit exceeded")
        return _FactoredBridgeIndex(MappingProxyType(entries), MappingProxyType(exits), rows, columns, stale,
                                    edges, left, right, comparisons)

    def match_target(self, source):
        entry = self.entries.get(source)
        if entry is None:
            return None
        row = self.rows.get(entry.key)
        if row and len(row) == 1:
            key, definite = next(iter(row.items()))
            exits = self.exit_members[key]
            if definite and len(exits) == len(self.columns[key]) == len(self.entry_members[entry.key]) == 1:
                return next(iter(exits))


class _LinkReadWork:
    def __init__(self, limit, floor, uncertainty):
        self.limit, self.floor, self.uncertainty = limit, floor, uncertainty
        self.work = self.transports = self.condition_dots = 0
        self.unavailable = set()
        self.marginal = False
        self.bridge_open = set()
        self.bridge_stale = False

    def charge(self, amount=1):
        if self.work + amount > self.limit:
            raise BindingCapacityError("role-link read work limit exceeded")
        self.work += amount

    def qualifies(self, score):
        if self.uncertainty and self.floor - self.uncertainty <= score <= self.floor + self.uncertainty:
            self.marginal = True
            return False
        return score >= self.floor


class _RoleBranchLimit(BindingCapacityError):
    pass


@dataclass(frozen=True, eq=False)
class _RoleLinks:
    roles: tuple[str, str]
    floor: float
    uncertainty: float
    compile_limit: int
    transport: str
    pairs: Mapping[str, _RoleLink]
    unresolved: frozenset[str]
    bridge_key: str | None = None
    bridges: Mapping[tuple[str, str], _RoleLink] = field(default_factory=lambda: MappingProxyType({}))
    key_sources: frozenset[str] = frozenset()
    bridge_pending: bool = False
    legacy_sources: frozenset[str] = frozenset()
    local_bridges: bool = False
    bridge_index: _BridgeIndex | None = None
    factored_bridges: bool = False

    @classmethod
    def prepare(cls, roles, floor, uncertainty, limit, transport, digests, mentions, bridge_key=None,
                local_bridges=False, factored_bridges=False):
        remaining = limit - len(digests)
        if remaining < 0:
            raise BindingCapacityError("role-link compilation work limit exceeded")
        remaining -= sum(map(len, mentions.values()))
        if remaining < 0:
            raise BindingCapacityError("role-link compilation work limit exceeded")
        empty = cls(roles, floor, uncertainty, limit, transport, MappingProxyType({}), frozenset())
        local = empty.revise(digests, digests, mentions)
        if bridge_key is None:
            return local
        return replace(local, local_bridges=local_bridges,
                       factored_bridges=factored_bridges)._find_bridges(bridge_key, mentions, remaining)

    def _find_bridges(self, key, mentions, remaining):
        enters, exits, key_sources = {}, {}, set()
        for source, records in mentions.items():
            has_key, fragment = _bridge_fragment(records, self.roles, key)
            if has_key:
                key_sources.add(source)
            if fragment is not None:
                role, value = fragment
                table = enters if role == self.roles[0] else exits
                table[source] = value
        selected = replace(self, bridge_key=key, key_sources=frozenset(key_sources))
        if self.legacy_sources:
            return selected

        if self.factored_bridges:
            index = _FactoredBridgeIndex.build(enters, exits, self, remaining)
            return index.matches(replace(selected, unresolved=self.unresolved - enters.keys() - exits.keys()))

        width = len(next(iter(enters.values())).key) // 8 if enters else 0
        if len(enters) * len(exits) * width > remaining:
            raise BindingCapacityError("cross-source role matching work limit exceeded")
        if self.local_bridges:
            index = _BridgeIndex.build(enters, exits, self)
            return index.matches(replace(selected, unresolved=self.unresolved - enters.keys() - exits.keys()))

        entries, departures, pairs = {}, {}, {}
        for source, entry in enters.items():
            for target, exit in exits.items():
                match = _key_match(entry.key, exit.key, self.floor, self.uncertainty)
                if match is None:
                    continue
                entries[source] = entries.get(source, 0) + 1
                departures[target] = departures.get(target, 0) + 1
                if match:
                    pairs[source] = target

        bridges, resolved = {}, set()
        for source, target in pairs.items():
            if entries[source] != 1 or departures[target] != 1:
                continue
            bridges[source, target] = _cross_link(source, target, enters[source], exits[target], self.transport)
            resolved.update((source, target))
        return replace(selected, bridges=MappingProxyType(bridges), unresolved=self.unresolved - resolved)

    def revise(self, changed, digests, mentions):
        pairs, unresolved = dict(self.pairs), set(self.unresolved)
        key_sources = set(self.key_sources)
        legacy_sources = set(self.legacy_sources)
        bridge_pending = self.bridge_pending
        bridge_changes = {}
        entry_role, exit_role = self.roles
        for source in changed:
            pairs.pop(source, None)
            unresolved.discard(source)
            legacy = source in digests and source not in mentions
            if self.bridge_key is not None:
                old_key = source in key_sources
                new_key = any(role == self.bridge_key for role, _, _, _ in mentions.get(source, ()))
                if old_key or new_key or source in legacy_sources or legacy:
                    bridge_pending = True
                    bridge_changes[source] = mentions.get(source, ())
                if new_key:
                    key_sources.add(source)
                else:
                    key_sources.discard(source)
            legacy_sources.discard(source)
            if legacy:
                legacy_sources.add(source)
                unresolved.add(source)
            if source not in mentions:
                continue
            entries, exits = [], []
            for ordinal, (predicate, strength, vector, _) in enumerate(mentions[source]):
                if predicate == entry_role:
                    entries.append((ordinal, vector, strength))
                elif predicate == exit_role:
                    exits.append((ordinal, vector, strength))
            if len(entries) == len(exits) == 1:
                left_id, left, left_strength = entries[0]
                right_id, right, right_strength = exits[0]
                pairs[source] = _RoleLink.from_ends(BindingLinkStep(source, left_id, right_id),
                                                    left, right, min(left_strength, right_strength), self.transport)
            elif entries or exits:
                unresolved.add(source)
        revised = replace(self, pairs=MappingProxyType(pairs), unresolved=frozenset(unresolved),
                          key_sources=frozenset(key_sources), bridge_pending=bridge_pending,
                          legacy_sources=frozenset(legacy_sources))
        if self.bridge_index is not None:
            return revised._revise_bridge_index(bridge_changes)
        return replace(revised, bridges=MappingProxyType({})) if bridge_pending else revised

    def _revise_bridge_index(self, changed):
        if not changed:
            return self
        entries, exits = dict(self.bridge_index.entries), dict(self.bridge_index.exits)
        for source, records in changed.items():
            entries.pop(source, None)
            exits.pop(source, None)
            _, fragment = _bridge_fragment(records, self.roles, self.bridge_key)
            if fragment is not None:
                role, value = fragment
                table = entries if role == self.roles[0] else exits
                table[source] = value
        if not self.legacy_sources:
            try:
                index = self.bridge_index.revise(entries, exits, changed.keys(), self)
            except BindingCapacityError:
                pass
            else:
                bridges = {pair: link for pair, link in self.bridges.items() if pair[0] not in index.stale}
                return replace(self, bridges=MappingProxyType(bridges), bridge_index=index,
                               bridge_pending=bool(index.stale),
                               unresolved=self.unresolved - entries.keys() - exits.keys())
        return replace(self, bridges=MappingProxyType({}), bridge_index=None)

    @property
    def bridge_global_pending(self):
        return self.bridge_pending and self.bridge_index is None

    def structural(self, role):
        return role in self.roles or role == self.bridge_key

    def _seed_path(self, direction, width, condition, prior_steps=(), prior_strength=1.):
        direction = _real_array(direction)
        if direction.shape != (width,):
            raise ValueError("role-link direction must match the source frame width")
        if condition is not None and len(condition.axis) != width:
            raise ValueError("role-link condition must match the source frame width")
        direction = _unit_direction(direction)
        if direction is not None:
            return _Path(direction, prior_steps, frozenset(source for step in prior_steps for source in step.sources),
                         prior_strength)

    def _frontier(self, seed, *, hops, condition, width, effort, branch_limit=None):
        frontier = [seed]
        if len(self.pairs) + len(self.bridges) > effort.limit:
            raise BindingCapacityError("role-link catalog exceeds the read work limit")
        links = tuple(self.pairs[source] for source in sorted(self.pairs))
        links += tuple(self.bridges[pair] for pair in sorted(self.bridges))
        openings = () if self.bridge_index is None else self.bridge_index.openings(self.bridges)
        for _ in range(hops):
            advanced = []
            for path in frontier:
                for source, raw, stale in openings:
                    effort.charge(width)
                    if source not in path.used and np.frombuffer(raw, dtype="<f8") @ path.direction >= (
                        self.floor - self.uncertainty
                    ):
                        effort.bridge_open.add(source)
                        effort.bridge_stale |= stale
                effort.charge(len(links))
                for link in links:
                    sources = link.step.sources
                    if not path.used.isdisjoint(sources) or not effort.qualifies(
                        np.frombuffer(link.entry, dtype="<f8") @ path.direction
                    ):
                        continue
                    if self.transport == "rotation" and link.reflector is None:
                        effort.unavailable.update(sources)
                        continue
                    if branch_limit is not None and len(advanced) >= branch_limit:
                        raise _RoleBranchLimit("role frontier exceeds the branch limit")
                    if path.steps:
                        effort.charge(2 * len(path.steps))
                    if self.transport != "fixed":
                        effort.charge(width * (2 if self.transport == "rotation" else 1))
                        effort.transports += 1
                    advanced.append(_Path(
                        link.advance(path.direction, self.transport),
                        (*path.steps, link.step), path.used.union(sources), min(path.strength, link.strength),
                    ))
            frontier = advanced
            if not frontier:
                break

        groups = ()
        if condition is not None and frontier:
            axis = np.asarray(condition.axis)
            gate_results, retained, retained_groups = {}, [], []
            for path in frontier:
                effort.charge(width)
                key = np.asarray(path.direction, dtype="<f8").tobytes()
                if key not in gate_results:
                    effort.charge(width)
                    effort.condition_dots += 1
                    score = max(-1., min(1., axis @ path.direction))
                    boundary = condition.uncertainty and (
                        abs(score - condition.lower) <= condition.uncertainty
                        or abs(score - condition.upper) <= condition.uncertainty
                    )
                    effort.marginal |= bool(boundary)
                    accepted = not boundary and condition.lower <= score <= condition.upper
                    gate_results[key] = effort.condition_dots if accepted else None
                group = gate_results[key]
                if group is not None:
                    retained.append(path)
                    retained_groups.append(group)
            frontier, groups = retained, retained_groups

        return frontier, groups

    def read(self, direction, *, outcomes, mentions, limit, width, hops, condition,
             prior_steps=(), prior_strength=1., branch_limit=None):
        vocabulary = tuple(sorted(outcomes))
        seed = self._seed_path(direction, width, condition, prior_steps, prior_strength)
        if seed is None:
            return BindingLinkRead("no_direction", outcomes=vocabulary, hops=hops, condition=condition,
                                   bridge_pending=self.bridge_global_pending)

        effort = _LinkReadWork(limit, self.floor, self.uncertainty)
        witnesses = {}
        terminal_dots = 0
        unmatched = set()
        outside = set()
        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                frontier, groups = self._frontier(seed, hops=hops, condition=condition, width=width,
                                                 effort=effort, branch_limit=branch_limit)
                if frontier:
                    unmatched.update(range(len(frontier)))
                    effort.charge(len(mentions))
                    for terminal_source in sorted(mentions):
                        for ordinal, (outcome, strength, raw_terminal, _) in enumerate(mentions[terminal_source]):
                            effort.charge()
                            if self.structural(outcome):
                                continue
                            terminal = np.frombuffer(raw_terminal, dtype="<f8")
                            scores = {} if condition is not None else None
                            for position, path in enumerate(frontier):
                                effort.charge()
                                if terminal_source in path.used:
                                    continue
                                if scores is None:
                                    terminal_dots += 1
                                    matches = effort.qualifies(path.direction @ terminal)
                                else:
                                    group = groups[position]
                                    if group not in scores:
                                        effort.charge(width)
                                        terminal_dots += 1
                                        scores[group] = effort.qualifies(path.direction @ terminal)
                                    matches = scores[group]
                                if not matches:
                                    continue
                                if outcome not in outcomes:
                                    outside.add(outcome)
                                    continue
                                unmatched.discard(position)
                                grade = min(path.strength, strength)
                                prior = witnesses.get(outcome)
                                if prior is None or grade > prior.strength:
                                    effort.charge(len(path.steps) - 1)
                                    step = path.steps[-1]
                                    witnesses[outcome] = BindingLinkWitness(
                                        outcome, step.source, step.entry_ordinal, step.exit_ordinal,
                                        terminal_source, ordinal, grade, path.steps[:-1], step.exit_source,
                                        step.key_ordinals,
                                    )
        except BindingCapacityError as error:
            status = "branch_limit" if isinstance(error, _RoleBranchLimit) else "work_limit"
            return BindingLinkRead(status, work=effort.work,
                                   open_sources=len(self.unresolved | effort.unavailable | effort.bridge_open),
                                   terminal_dots=terminal_dots, outcomes=vocabulary, hops=hops, transports=effort.transports,
                                   condition=condition, condition_dots=effort.condition_dots,
                                   bridge_pending=self.bridge_global_pending or effort.bridge_stale)
        open_sources = len(self.unresolved | effort.unavailable | effort.bridge_open)
        pending = self.bridge_global_pending or effort.bridge_stale
        incomplete = open_sources or unmatched or effort.marginal or outside or pending
        status = ("conflict" if len(witnesses) > 1 else "unresolved" if incomplete
                  else "supported" if witnesses else "no_link")
        return BindingLinkRead(status, tuple(witnesses[key] for key in sorted(witnesses)), effort.work,
                               open_sources, len(unmatched), effort.marginal, terminal_dots,
                               vocabulary, tuple(sorted(outside)), hops, effort.transports, condition, effort.condition_dots,
                               bridge_pending=pending)
