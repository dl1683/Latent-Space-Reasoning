"""Explicit effective time for version-bound local role replacements."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import islice
from types import MappingProxyType

from experiments.memory.binding_memory import BindingCapacityError
from experiments.memory.verified_source_tool import SourceSpan


@dataclass(frozen=True)
class RoleSupersession:
    previous: SourceSpan
    replacement: SourceSpan
    authority: SourceSpan
    effective_at: int
    previous_valid_from: int | None = None

    def __post_init__(self):
        if type(self.effective_at) is not int:
            raise ValueError("effective time must be an integer")
        if self.previous_valid_from is not None and (
            type(self.previous_valid_from) is not int or self.previous_valid_from >= self.effective_at
        ):
            raise ValueError("predecessor start must be an integer before the cutover")
        for name in ("previous", "replacement", "authority"):
            span = getattr(self, name)
            if type(span) is not SourceSpan:
                raise TypeError("role supersession requires exact SourceSpan pointers")
            object.__setattr__(self, name, replace(span, start_character=None, end_character=None))
        if self.previous.source_id == self.replacement.source_id:
            raise ValueError("role supersession requires two distinct documents")


def _pair_address(link):
    return link.step.source, link.step.entry_ordinal


def _scope_pairs(transitions, pairs, digests, mentions):
    sources = {span.source_id for rule in transitions for span in (rule.previous, rule.replacement)}
    by_span = {}
    for source in sources:
        for link in pairs.get(source, ()):
            scope = mentions[source][link.step.entry_ordinal][4]
            by_span[SourceSpan(source, digests[source], *scope)] = link
    return by_span


def _chains(rules):
    tails = {}
    sequences = []
    for rule in rules:
        sequence = tails.pop(rule.previous, None)
        if sequence is None:
            sequence = []
            sequences.append(sequence)
        sequence.append(rule)
        tails[rule.replacement] = sequence
    return sequences


@dataclass(frozen=True)
class _RoleValidity:
    as_of: int
    transitions: tuple[RoleSupersession, ...]
    entry_vectors: tuple[bytes, ...]
    dependencies: frozenset[str]
    withheld: frozenset[tuple[str, int]]
    uncertain_entries: Mapping[bytes, tuple[RoleSupersession, ...]]
    by_step: Mapping[tuple[str, int], tuple[RoleSupersession, ...]]
    authorities: Mapping[tuple[str, int], tuple[SourceSpan, ...]]
    authority_sources: Mapping[tuple[str, int], frozenset[str]]

    @classmethod
    def prepare(cls, as_of, transitions, pairs, digests, mentions, limit, prior):
        if type(as_of) is not int:
            raise ValueError("role horizon must be an integer")
        rules = tuple(islice(transitions, limit + 1))
        if len(rules) > limit:
            raise BindingCapacityError("role supersession rule limit exceeded")
        if any(type(rule) is not RoleSupersession for rule in rules):
            raise TypeError("role validity requires RoleSupersession rules")
        if not rules:
            return None
        rules = tuple(sorted(rules, key=lambda rule: (rule.effective_at, rule.previous.source_id,
                                                      rule.previous.start_byte, rule.previous.end_byte)))
        if prior is not None and rules == prior.transitions:
            return prior if as_of == prior.as_of else replace(prior, as_of=as_of).revise(pairs, digests, mentions)

        outgoing = {rule.previous: rule for rule in rules}
        incoming = {rule.replacement: rule for rule in rules}
        if len(incoming) != len(rules) or len(outgoing) != len(rules):
            raise ValueError("role supersession cannot branch at a claim")
        for span in incoming.keys() & outgoing.keys():
            start = incoming[span].effective_at
            if start >= outgoing[span].effective_at:
                raise ValueError("role supersession times must increase along a chain")
            if outgoing[span].previous_valid_from not in (None, start):
                raise ValueError("role supersession chain disagrees about predecessor start")

        by_span = _scope_pairs(rules, pairs, digests, mentions)
        known = {} if prior is None else dict(zip(prior.transitions, prior.entry_vectors, strict=True))
        vectors = []
        for rule in rules:
            if rule in known:
                vectors.append(known[rule])
                continue
            before, after = by_span.get(rule.previous), by_span.get(rule.replacement)
            if before is None or after is None or before.entry != after.entry:
                raise ValueError("role supersession requires current scoped links with the same entry")
            if digests.get(rule.authority.source_id) != rule.authority.version_sha256:
                raise ValueError("role supersession authority is stale or unavailable")
            vectors.append(before.entry)
        empty = MappingProxyType({})
        dependencies = frozenset(span.source_id for rule in rules
                                 for span in (rule.previous, rule.replacement, rule.authority))
        drafted = cls(as_of, rules, tuple(vectors), dependencies, frozenset(), empty, empty, empty, empty)
        return drafted.revise(pairs, digests, mentions, by_span=by_span)

    def revise(self, pairs, digests, mentions, *, by_span=None):
        if by_span is None:
            by_span = _scope_pairs(self.transitions, pairs, digests, mentions)
        entries = dict(zip(self.transitions, self.entry_vectors, strict=True))
        withheld, uncertain = set(), {}
        authorities, by_step = {}, {}
        for sequence in _chains(self.transitions):
            chain = tuple(sequence)
            claims = (chain[0].previous, *(rule.replacement for rule in chain))
            links = [by_span.get(claim) for claim in claims]
            active = sum(rule.effective_at <= self.as_of for rule in chain)
            withheld.update(_pair_address(link) for position, link in enumerate(links)
                            if link is not None and position != active)
            for link in links:
                if link is not None:
                    by_step[_pair_address(link)] = chain
            proofs = tuple(dict.fromkeys(rule.authority for rule in chain))
            entry = entries[chain[0]]
            start = chain[0].previous_valid_from
            if (any(link is None or link.entry != entry for link in links)
                    or any(digests.get(span.source_id) != span.version_sha256 for span in proofs)
                    or (active == 0 and (start is None or self.as_of < start))):
                uncertain.setdefault(entry, []).extend(chain)
            else:
                authorities[_pair_address(links[active])] = proofs
        origins = {key: frozenset(span.source_id for span in spans) for key, spans in authorities.items()}
        return replace(self, withheld=frozenset(withheld),
                       uncertain_entries=MappingProxyType({key: tuple(rules) for key, rules in uncertain.items()}),
                       by_step=MappingProxyType(by_step), authorities=MappingProxyType(authorities),
                       authority_sources=MappingProxyType(origins))
