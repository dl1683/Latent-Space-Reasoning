"""A query-independent bank of paired continuous directions from typed sources."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from experiments.memory.binding_memory import BindingCapacityError
from experiments.memory.latent_feedback import _real_array


@dataclass(frozen=True)
class BindingLinkWitness:
    outcome: str
    link_source: str
    entry_ordinal: int
    exit_ordinal: int
    terminal_source: str
    terminal_ordinal: int
    strength: float


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

    @property
    def answer(self):
        return self.witnesses[0].outcome if self.status == "supported" else None


@dataclass(frozen=True, eq=False)
class _RoleLinks:
    roles: tuple[str, str]
    floor: float
    uncertainty: float
    compile_limit: int
    pairs: Mapping[str, tuple[int, bytes, int, bytes, float]]
    unresolved: frozenset[str]

    @classmethod
    def prepare(cls, roles, floor, uncertainty, limit, digests, mentions):
        if len(digests) > limit or sum(map(len, mentions.values())) > limit - len(digests):
            raise BindingCapacityError("role-link compilation work limit exceeded")
        empty = cls(roles, floor, uncertainty, limit, MappingProxyType({}), frozenset())
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
                pairs[source] = (left_id, left, right_id, right, min(left_strength, right_strength))
            elif entries or exits:
                unresolved.add(source)
        return _RoleLinks(self.roles, self.floor, self.uncertainty, self.compile_limit,
                          MappingProxyType(pairs), frozenset(unresolved))

    def read(self, direction, *, outcomes, mentions, limit, width):
        direction = _real_array(direction)
        if direction.shape != (width,):
            raise ValueError("role-link direction must match the source frame width")
        peak = np.max(np.abs(direction))
        vocabulary = tuple(sorted(outcomes))
        if peak == 0.:
            return BindingLinkRead("no_direction", outcomes=vocabulary)
        # Scale the direction without making large or tiny coordinates overflow.
        with np.errstate(under="ignore"):
            direction = direction / peak
            direction /= np.linalg.norm(direction)
        witnesses = {}
        work = 0
        marginal = False
        terminal_dots = 0
        unmatched = set()
        outside = set()

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

        ready = []
        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                charge(len(self.pairs))
                for source in sorted(self.pairs):
                    entry_id, entry, exit_id, exit, strength = self.pairs[source]
                    if qualifies(np.frombuffer(entry, dtype="<f8") @ direction):
                        ready.append((source, entry_id, exit_id, np.frombuffer(exit, dtype="<f8"), strength))
                        unmatched.add(source)
                if ready:
                    charge(len(mentions))
                    for terminal_source in sorted(mentions):
                        for ordinal, (outcome, terminal_strength, terminal, _) in enumerate(mentions[terminal_source]):
                            charge()
                            if outcome in self.roles:
                                continue
                            terminal = np.frombuffer(terminal, dtype="<f8")
                            for link_source, entry_id, exit_id, exit, link_strength in ready:
                                charge()
                                if link_source == terminal_source:
                                    continue
                                terminal_dots += 1
                                if not qualifies(exit @ terminal):
                                    continue
                                if outcome not in outcomes:
                                    outside.add(outcome)
                                    continue
                                unmatched.discard(link_source)
                                candidate = BindingLinkWitness(
                                    outcome, link_source, entry_id, exit_id, terminal_source, ordinal,
                                    min(link_strength, terminal_strength),
                                )
                                prior = witnesses.get(outcome)
                                if prior is None or candidate.strength > prior.strength:
                                    witnesses[outcome] = candidate
        except BindingCapacityError:
            return BindingLinkRead("work_limit", work=work, open_sources=len(self.unresolved),
                                   terminal_dots=terminal_dots, outcomes=vocabulary)
        incomplete = self.unresolved or unmatched or marginal or outside
        status = ("conflict" if len(witnesses) > 1 else "unresolved" if incomplete
                  else "supported" if witnesses else "no_link")
        return BindingLinkRead(status, tuple(witnesses[key] for key in sorted(witnesses)), work,
                               len(self.unresolved), len(unmatched), marginal, terminal_dots,
                               vocabulary, tuple(sorted(outside)))
