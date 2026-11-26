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
    key: np.ndarray


class _LinkReadWork:
    def __init__(self, limit, floor, uncertainty):
        self.limit, self.floor, self.uncertainty = limit, floor, uncertainty
        self.work = self.transports = self.condition_dots = 0
        self.unavailable = set()
        self.marginal = False

    def charge(self, amount=1):
        if self.work + amount > self.limit:
            raise BindingCapacityError("role-link read work limit exceeded")
        self.work += amount

    def qualifies(self, score):
        if self.uncertainty and self.floor - self.uncertainty <= score <= self.floor + self.uncertainty:
            self.marginal = True
            return False
        return score >= self.floor


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

    @classmethod
    def prepare(cls, roles, floor, uncertainty, limit, transport, digests, mentions, bridge_key=None):
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
        return local._find_bridges(bridge_key, mentions, remaining)

    def _find_bridges(self, key, mentions, remaining):
        enters, exits, key_sources = {}, {}, set()
        for source, records in mentions.items():
            anchors = [(ordinal, grade, raw) for ordinal, (role, grade, raw, _) in enumerate(records) if role == key]
            if not anchors:
                continue
            key_sources.add(source)
            ends = [(ordinal, role, grade, raw) for ordinal, (role, grade, raw, _) in enumerate(records)
                    if role in self.roles]
            if len(anchors) == len(ends) == 1:
                ordinal, role, grade, raw = ends[0]
                anchor_ordinal, anchor_grade, anchor = anchors[0]
                table = enters if role == self.roles[0] else exits
                table[source] = _BridgeFragment(ordinal, anchor_ordinal, min(grade, anchor_grade), raw,
                                                np.frombuffer(anchor, dtype="<f8"))
        selected = replace(self, bridge_key=key, key_sources=frozenset(key_sources))
        if self.legacy_sources:
            return selected

        width = next(iter(enters.values())).key.size if enters else 0
        if len(enters) * len(exits) * width > remaining:
            raise BindingCapacityError("cross-source role matching work limit exceeded")
        entries, departures, pairs = {}, {}, {}
        for source, entry in enters.items():
            for target, exit in exits.items():
                score = float(np.clip(entry.key @ exit.key, -1., 1.))
                if score < self.floor - self.uncertainty:
                    continue
                entries[source] = entries.get(source, 0) + 1
                departures[target] = departures.get(target, 0) + 1
                if score >= self.floor and (not self.uncertainty or score > self.floor + self.uncertainty):
                    pairs[source] = target

        bridges, resolved = {}, set()
        for source, target in pairs.items():
            if entries[source] != 1 or departures[target] != 1:
                continue
            entry, exit = enters[source], exits[target]
            step = BindingLinkStep(source, entry.ordinal, exit.ordinal, target, (entry.key_ordinal, exit.key_ordinal))
            strength = min(entry.strength, exit.strength)
            bridges[source, target] = _RoleLink.from_ends(step, entry.direction, exit.direction, strength, self.transport)
            resolved.update((source, target))
        return replace(selected, bridges=MappingProxyType(bridges), unresolved=self.unresolved - resolved)

    def revise(self, changed, digests, mentions):
        pairs, unresolved = dict(self.pairs), set(self.unresolved)
        key_sources = set(self.key_sources)
        legacy_sources = set(self.legacy_sources)
        bridge_pending = self.bridge_pending
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
        return replace(self, pairs=MappingProxyType(pairs), unresolved=frozenset(unresolved),
                       bridges=MappingProxyType({}) if bridge_pending else self.bridges,
                       key_sources=frozenset(key_sources), bridge_pending=bridge_pending,
                       legacy_sources=frozenset(legacy_sources))

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

    def _frontier(self, seed, *, hops, condition, width, effort):
        frontier = [seed]
        if len(self.pairs) + len(self.bridges) > effort.limit:
            raise BindingCapacityError("role-link catalog exceeds the read work limit")
        links = tuple(self.pairs[source] for source in sorted(self.pairs))
        links += tuple(self.bridges[pair] for pair in sorted(self.bridges))
        for _ in range(hops):
            advanced = []
            for path in frontier:
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
             prior_steps=(), prior_strength=1.):
        vocabulary = tuple(sorted(outcomes))
        seed = self._seed_path(direction, width, condition, prior_steps, prior_strength)
        if seed is None:
            return BindingLinkRead("no_direction", outcomes=vocabulary, hops=hops, condition=condition,
                                   bridge_pending=self.bridge_pending)

        effort = _LinkReadWork(limit, self.floor, self.uncertainty)
        witnesses = {}
        terminal_dots = 0
        unmatched = set()
        outside = set()
        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                frontier, groups = self._frontier(seed, hops=hops, condition=condition, width=width, effort=effort)
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
        except BindingCapacityError:
            return BindingLinkRead("work_limit", work=effort.work,
                                   open_sources=len(self.unresolved) + len(effort.unavailable),
                                   terminal_dots=terminal_dots, outcomes=vocabulary, hops=hops, transports=effort.transports,
                                   condition=condition, condition_dots=effort.condition_dots,
                                   bridge_pending=self.bridge_pending)
        open_sources = len(self.unresolved) + len(effort.unavailable)
        incomplete = open_sources or unmatched or effort.marginal or outside or self.bridge_pending
        status = ("conflict" if len(witnesses) > 1 else "unresolved" if incomplete
                  else "supported" if witnesses else "no_link")
        return BindingLinkRead(status, tuple(witnesses[key] for key in sorted(witnesses)), effort.work,
                               open_sources, len(unmatched), effort.marginal, terminal_dots,
                               vocabulary, tuple(sorted(outside)), hops, effort.transports, condition, effort.condition_dots,
                               bridge_pending=self.bridge_pending)
