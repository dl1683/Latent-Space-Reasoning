"""A query-independent bank of paired continuous directions from typed sources."""

from collections.abc import Mapping
from dataclasses import dataclass
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


@dataclass(frozen=True, eq=False)
class _RoleLinks:
    roles: tuple[str, str]
    floor: float
    uncertainty: float
    compile_limit: int
    transport: str
    pairs: Mapping[str, _RoleLink]
    unresolved: frozenset[str]

    @classmethod
    def prepare(cls, roles, floor, uncertainty, limit, transport, digests, mentions):
        if len(digests) > limit or sum(map(len, mentions.values())) > limit - len(digests):
            raise BindingCapacityError("role-link compilation work limit exceeded")
        empty = cls(roles, floor, uncertainty, limit, transport, MappingProxyType({}), frozenset())
        return empty.revise(digests, digests, mentions)

    def revise(self, changed, digests, mentions):
        pairs, unresolved = dict(self.pairs), set(self.unresolved)
        entry_role, exit_role = self.roles
        for source in changed:
            pairs.pop(source, None)
            unresolved.discard(source)
            if source not in digests:
                continue
            if source not in mentions:
                unresolved.add(source)
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
                reflector = None
                if self.transport != "fixed":
                    start, end = np.frombuffer(left, dtype="<f8"), np.frombuffer(right, dtype="<f8")
                    with np.errstate(under="ignore"):
                        normal = start - end if self.transport == "reflection" else start + end
                    if self.transport == "reflection" or np.max(np.abs(normal)) > 1e-12:
                        axis = _unit_direction(normal)
                        if axis is not None:
                            reflector = np.asarray(axis, dtype="<f8").tobytes()
                pairs[source] = _RoleLink(BindingLinkStep(source, left_id, right_id),
                                          left, right, min(left_strength, right_strength), reflector)
            elif entries or exits:
                unresolved.add(source)
        return _RoleLinks(self.roles, self.floor, self.uncertainty, self.compile_limit, self.transport,
                          MappingProxyType(pairs), frozenset(unresolved))

    def read(self, direction, *, outcomes, mentions, limit, width, hops, condition):
        direction = _real_array(direction)
        if direction.shape != (width,):
            raise ValueError("role-link direction must match the source frame width")
        if condition is not None and len(condition.axis) != width:
            raise ValueError("role-link condition must match the source frame width")
        vocabulary = tuple(sorted(outcomes))
        direction = _unit_direction(direction)
        if direction is None:
            return BindingLinkRead("no_direction", outcomes=vocabulary, hops=hops, condition=condition)

        witnesses = {}
        work = terminal_dots = transports = condition_dots = 0
        marginal = False
        unmatched = set()
        outside = set()
        unavailable = set()

        def charge(amount=1):
            nonlocal work
            if work + amount > limit:
                raise BindingCapacityError("role-link read work limit exceeded")
            work += amount

        def qualifies(score):
            nonlocal marginal
            if self.uncertainty and self.floor - self.uncertainty <= score <= self.floor + self.uncertainty:
                marginal = True
                return False
            return score >= self.floor

        frontier = [_Path(direction)]
        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                if len(self.pairs) > limit:
                    raise BindingCapacityError("role-link catalog exceeds the read work limit")
                links = tuple(self.pairs[source] for source in sorted(self.pairs))
                for _ in range(hops):
                    advanced = []
                    for path in frontier:
                        charge(len(links))
                        for link in links:
                            source = link.step.source
                            if source in path.used or not qualifies(np.frombuffer(link.entry, dtype="<f8") @ path.direction):
                                continue
                            if self.transport == "rotation" and link.reflector is None:
                                unavailable.add(source)
                                continue
                            if path.steps:
                                charge(2 * len(path.steps))
                            if self.transport != "fixed":
                                charge(width * (2 if self.transport == "rotation" else 1))
                                transports += 1
                            advanced.append(_Path(
                                link.advance(path.direction, self.transport),
                                (*path.steps, link.step), path.used | {source}, min(path.strength, link.strength),
                            ))
                    frontier = advanced
                    if not frontier:
                        break

                groups = ()
                if condition is not None and frontier:
                    axis = np.asarray(condition.axis)
                    gate_results, retained, retained_groups = {}, [], []
                    for path in frontier:
                        charge(width)
                        key = np.asarray(path.direction, dtype="<f8").tobytes()
                        if key not in gate_results:
                            charge(width)
                            condition_dots += 1
                            score = max(-1., min(1., axis @ path.direction))
                            boundary = condition.uncertainty and (
                                abs(score - condition.lower) <= condition.uncertainty
                                or abs(score - condition.upper) <= condition.uncertainty
                            )
                            marginal |= bool(boundary)
                            accepted = not boundary and condition.lower <= score <= condition.upper
                            gate_results[key] = condition_dots if accepted else None
                        group = gate_results[key]
                        if group is not None:
                            retained.append(path)
                            retained_groups.append(group)
                    frontier, groups = retained, retained_groups

                if frontier:
                    unmatched.update(range(len(frontier)))
                    charge(len(mentions))
                    for terminal_source in sorted(mentions):
                        for ordinal, (outcome, strength, raw_terminal, _) in enumerate(mentions[terminal_source]):
                            charge()
                            if outcome in self.roles:
                                continue
                            terminal = np.frombuffer(raw_terminal, dtype="<f8")
                            scores = {} if condition is not None else None
                            for position, path in enumerate(frontier):
                                charge()
                                if terminal_source in path.used:
                                    continue
                                if scores is None:
                                    terminal_dots += 1
                                    matches = qualifies(path.direction @ terminal)
                                else:
                                    group = groups[position]
                                    if group not in scores:
                                        charge(width)
                                        terminal_dots += 1
                                        scores[group] = qualifies(path.direction @ terminal)
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
                                    charge(len(path.steps) - 1)
                                    step = path.steps[-1]
                                    witnesses[outcome] = BindingLinkWitness(
                                        outcome, step.source, step.entry_ordinal, step.exit_ordinal,
                                        terminal_source, ordinal, grade, path.steps[:-1],
                                    )
        except BindingCapacityError:
            return BindingLinkRead("work_limit", work=work, open_sources=len(self.unresolved) + len(unavailable),
                                   terminal_dots=terminal_dots, outcomes=vocabulary, hops=hops, transports=transports,
                                   condition=condition, condition_dots=condition_dots)
        open_sources = len(self.unresolved) + len(unavailable)
        incomplete = open_sources or unmatched or marginal or outside
        status = ("conflict" if len(witnesses) > 1 else "unresolved" if incomplete
                  else "supported" if witnesses else "no_link")
        return BindingLinkRead(status, tuple(witnesses[key] for key in sorted(witnesses)), work,
                               open_sources, len(unmatched), marginal, terminal_dots,
                               vocabulary, tuple(sorted(outside)), hops, transports, condition, condition_dots)
