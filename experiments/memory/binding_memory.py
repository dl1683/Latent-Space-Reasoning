"""Retain same-coordinate conjunctions and their declared source sets across edits."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from numbers import Real
from operator import index
from types import MappingProxyType

import numpy as np

from experiments.memory.latent_feedback import _frozen, _real_array


_EMPTY = MappingProxyType({})


class BindingCapacityError(ValueError):
    """The declared support or join work limit would be exceeded."""


@dataclass(frozen=True)
class BindingGate:
    operation: str
    parents: tuple[str, ...]


@dataclass(frozen=True)
class _Support:
    terms: Mapping[frozenset[str], float]
    strongest: float


_UNSUPPORTED = _Support(_EMPTY, -np.inf)


def _checked_binding(binding, width):
    binding = index(binding)
    if not 0 <= binding < width:
        raise ValueError("binding coordinate is outside the declared width")
    return binding


def _source_rows(facts, predicates, width):
    rows = {}
    for predicate, binding, strength in facts:
        if predicate not in predicates:
            raise ValueError("source rows must target declared predicates")
        key = (predicate, _checked_binding(binding, width))
        if not isinstance(strength, Real) or not 0 <= strength <= 1:
            raise ValueError("binding strengths must be real numbers in [0, 1]")
        rows[key] = max(rows.get(key, 0.0), float(strength))
    return rows


def _retain(terms, sources, strength, limit):
    terms[sources] = max(terms.get(sources, 0.0), strength)
    if len(terms) > limit:
        raise BindingCapacityError("source-set limit exceeded")


def _gate_terms(gate, children, limit, pairs_left):
    if gate.operation == "or":
        result = {}
        for terms in children:
            for sources, strength in terms.items():
                _retain(result, sources, strength, limit)
        return result, pairs_left

    if not all(children):
        return {}, pairs_left
    result = children[0]
    for terms in children[1:]:
        pairs_left -= len(result) * len(terms)
        if pairs_left < 0:
            raise BindingCapacityError("source-set pair limit exceeded")
        joined = {}
        for left, left_strength in result.items():
            for right, right_strength in terms.items():
                if gate.operation == "and" or left.isdisjoint(right):
                    _retain(joined, left | right, min(left_strength, right_strength), limit)
        result = joined
    return result, pairs_left


@dataclass(frozen=True, eq=False)
class BindingMemory:
    """A fixed acyclic gate graph with immutable, locally revised support rows."""

    width: int
    term_limit: int
    pair_limit: int
    _predicates: frozenset[str]
    _gates: tuple[tuple[str, BindingGate], ...]
    _dependents: Mapping[str, tuple[int, ...]]
    _sources: Mapping[str, Mapping[tuple[str, int], float]]
    _rows: Mapping[tuple[str, int], _Support]

    @classmethod
    def prepare(cls, predicates, width, gates, sources, *, term_limit=64, pair_limit=65536):
        """Compile gates in topological order and apply the initial source batch."""
        predicates = tuple(predicates)
        width, term_limit, pair_limit = index(width), index(term_limit), index(pair_limit)
        if min(width, term_limit, pair_limit) < 1:
            raise ValueError("binding width and limits must be positive")
        if (not predicates or any(type(name) is not str or not name for name in predicates)
                or len(set(predicates)) != len(predicates)):
            raise ValueError("input predicates must have unique nonempty names")
        known = {name: [] for name in predicates}
        ordered = []
        for name, gate in gates.items():
            if type(name) is not str or not name or name in known:
                raise ValueError("gate names must be nonempty and distinct from other nodes")
            if type(gate) is not BindingGate or gate.operation not in ("or", "and", "disjoint_and"):
                raise ValueError("unknown binding gate")
            parents = tuple(gate.parents)
            if not parents or any(parent not in known for parent in parents):
                raise ValueError("gate parents must precede the gate")
            for parent in set(parents):
                known[parent].append(len(ordered))
            known[name] = []
            ordered.append((name, BindingGate(gate.operation, parents)))
        dependents = MappingProxyType({node: tuple(users) for node, users in known.items()})
        empty = cls(width, term_limit, pair_limit, frozenset(predicates), tuple(ordered), dependents, _EMPTY, _EMPTY)
        return empty.revise(sources)

    def revise(self, changes):
        """Replace all facts for each supplied source ID, or remove it with None."""
        sources = None
        inputs = {}
        for source, facts in changes.items():
            if type(source) is not str or not source:
                raise ValueError("source IDs must be nonempty strings")
            previous = self._sources.get(source, _EMPTY)
            current = {} if facts is None else _source_rows(facts, self._predicates, self.width)
            if previous == current:
                continue
            if sources is None:
                sources = dict(self._sources)
            if current:
                sources[source] = MappingProxyType(current)
            else:
                del sources[source]
            term = frozenset({source})
            for key in previous.keys() | current.keys():
                if previous.get(key) == current.get(key):
                    continue
                if key not in inputs:
                    inputs[key] = dict(self._rows.get(key, _UNSUPPORTED).terms)
                if key in current:
                    inputs[key][term] = current[key]
                else:
                    del inputs[key][term]
        if not inputs:
            return self

        rows = dict(self._rows)
        pending, scheduled = [], set()

        def install(node, binding, terms):
            key = (node, binding)
            if rows.get(key, _UNSUPPORTED).terms == terms:
                return
            if len(terms) > self.term_limit:
                raise BindingCapacityError("source-set limit exceeded")
            if terms:
                rows[key] = _Support(MappingProxyType(dict(terms)), max(terms.values()))
            else:
                rows.pop(key, None)
            for user in self._dependents[node]:
                update = (user, binding)
                if update not in scheduled:
                    heappush(pending, update)
                    scheduled.add(update)

        for (node, binding), terms in inputs.items():
            install(node, binding, terms)
        pairs_left = self.pair_limit
        while pending:
            position, binding = heappop(pending)
            node, gate = self._gates[position]
            children = [rows.get((parent, binding), _UNSUPPORTED).terms for parent in gate.parents]
            terms, pairs_left = _gate_terms(gate, children, self.term_limit, pairs_left)
            install(node, binding, terms)
        return replace(self, _sources=MappingProxyType(sources), _rows=MappingProxyType(rows))

    def supports(self, node, binding):
        """Return the retained source sets and strengths for a specific binding."""
        if node not in self._dependents:
            raise ValueError("unknown binding node")
        return self._rows.get((node, _checked_binding(binding, self.width)), _UNSUPPORTED).terms


def continue_bindings(seed, *, memory=None, output=None, without_sources=()):
    """Clamp each late seed coordinate by the strongest support at the selected node."""
    seed = _real_array(seed)
    if seed.ndim != 1 or not seed.size or np.any((seed < 0) | (seed > 1)):
        raise ValueError("binding seed must be a nonempty vector in [0, 1]")
    if isinstance(without_sources, str):
        raise ValueError("without_sources must contain source IDs")
    excluded = frozenset(without_sources)
    if any(type(source) is not str or not source for source in excluded):
        raise ValueError("without_sources must contain source IDs")
    if memory is None:
        if output is not None or excluded:
            raise ValueError("an output node or source exclusion requires binding memory")
        return _frozen(seed)
    if type(memory) is not BindingMemory or seed.shape != (memory.width,):
        raise ValueError("binding seed must match the memory width")
    if output not in memory._dependents:
        raise ValueError("unknown binding node")
    rows = (memory._rows.get((output, binding), _UNSUPPORTED) for binding in range(memory.width))
    if excluded:
        scores = (max((value for origins, value in row.terms.items() if origins.isdisjoint(excluded)),
                      default=-np.inf) for row in rows)
    else:
        scores = (row.strongest for row in rows)
    strongest = np.fromiter(scores, dtype=np.float64, count=memory.width)
    return _frozen(np.minimum(seed, strongest))
