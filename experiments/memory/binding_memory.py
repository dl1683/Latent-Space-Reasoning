"""Retain same-coordinate conjunctions and their declared source sets across edits."""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from itertools import count
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
class BindingProof:
    strength: float
    sources: frozenset[str]


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
    _gate_index: Mapping[str, int]
    _sources: Mapping[str, Mapping[tuple[str, int], float]]
    _rows: Mapping[tuple[str, int], _Support]
    materialize: bool = True

    @classmethod
    def prepare(cls, predicates, width, gates, sources, *, term_limit=64, pair_limit=65536, materialize=True):
        """Compile gates in topological order and apply the initial source batch."""
        predicates = tuple(predicates)
        width, term_limit, pair_limit = index(width), index(term_limit), index(pair_limit)
        if min(width, term_limit, pair_limit) < 1:
            raise ValueError("binding width and limits must be positive")
        if type(materialize) is not bool:
            raise ValueError("materialize must be a boolean")
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
        positions = MappingProxyType({node: position for position, (node, _) in enumerate(ordered)})
        empty = cls(width, term_limit, pair_limit, frozenset(predicates), tuple(ordered), dependents,
                    positions, _EMPTY, _EMPTY, materialize)
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
            for user in self._dependents[node] if self.materialize else ():
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
        if not self.materialize and node not in self._predicates:
            raise ValueError("factored gates expose a selected proof through resolve")
        return self._rows.get((node, _checked_binding(binding, self.width)), _UNSUPPORTED).terms

    def resolve(self, node, binding, *, without_sources=(), work_limit=65536):
        """Select a strongest proof, or return None after establishing no support."""
        binding = _checked_binding(binding, self.width)
        if node not in self._dependents:
            raise ValueError("unknown binding node")
        excluded = _exclusions(without_sources)
        if self.materialize:
            candidates = self.supports(node, binding)
            strength, origins = max(
                ((value, sources) for sources, value in candidates.items() if sources.isdisjoint(excluded)),
                key=lambda item: (item[0], tuple(sorted(item[1]))), default=(-np.inf, None),
            )
            return None if origins is None else BindingProof(strength, origins)
        search = _ProofSearch(self, node, excluded, work_limit)
        return search.resolve(binding)


def _exclusions(without_sources):
    if isinstance(without_sources, str):
        raise ValueError("without_sources must contain source IDs")
    excluded = frozenset(without_sources)
    if any(type(source) is not str or not source for source in excluded):
        raise ValueError("without_sources must contain source IDs")
    return excluded


@dataclass(frozen=True, eq=False)
class _Task:
    node: str
    guards: tuple[tuple[int, int], ...]
    following: "_Task | None" = None


class _ProofSearch:
    def __init__(self, memory, output, excluded, work_limit):
        self.remaining = index(work_limit)
        if self.remaining < 1:
            raise ValueError("proof work limit must be positive")
        self.rows, self.output, self.excluded = memory._rows, output, excluded
        pending, seen = [output], set()
        inputs, positions = [], []
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            self.charge()
            position = memory._gate_index.get(node)
            if position is None:
                inputs.append(node)
            else:
                _, gate = memory._gates[position]
                self.charge(len(gate.parents))
                positions.append(position)
                pending.extend(gate.parents)
        self.rules = {}
        for position in sorted(positions):
            node, gate = memory._gates[position]
            if gate.operation != "disjoint_and":
                # Reusing one proof for identical ordinary parents can only shrink its source set.
                gate = BindingGate(gate.operation, tuple(dict.fromkeys(gate.parents)))
            self.rules[node] = gate
        self.inputs = sorted(inputs)
        disjoint = [node for node, gate in self.rules.items() if gate.operation == "disjoint_and"]
        self.unguarded = not disjoint
        self.single_source_matching = disjoint == [output] and all(
            node == output or gate.operation == "or" for node, gate in self.rules.items()
        )

    def charge(self, amount=1):
        self.remaining -= amount
        if self.remaining < 0:
            raise BindingCapacityError("proof search work limit exceeded")

    def resolve(self, binding):
        leaves, upper, strengths = {}, {}, set()
        for node in self.inputs:
            candidates = self.rows.get((node, binding), _UNSUPPORTED).terms
            self.charge(1 + len(candidates))
            choices = [(value, next(iter(sources))) for sources, value in candidates.items()
                       if sources.isdisjoint(self.excluded)]
            leaves[node] = sorted(choices, reverse=True)
            upper[node] = max((value for value, _ in choices), default=-np.inf)
            strengths.update(value for value, _ in choices)
        for node, gate in self.rules.items():
            self.charge(1 + len(gate.parents))
            reduce = max if gate.operation == "or" else min
            upper[node] = reduce(upper[parent] for parent in gate.parents)
        # Ignoring source overlap yields an upper bound, not a feasible proof.
        ceiling = upper[self.output]
        if ceiling == -np.inf:
            return None
        if self.unguarded:
            return BindingProof(ceiling, self.unguarded_sources(upper, leaves))
        domains = self.matching_domains(leaves) if self.single_source_matching else None

        def witness(threshold):
            if domains is not None:
                return self.match(threshold, domains)
            return self.witness(threshold, upper, leaves)

        proof = witness(ceiling)
        if proof is not None:
            return BindingProof(ceiling, proof)
        levels = sorted(value for value in strengths if value < ceiling)
        lo, hi = 0, len(levels)
        best = None
        while lo < hi:
            middle = (lo + hi) // 2
            threshold = levels[middle]
            proof = witness(threshold)
            if proof is None:
                hi = middle
            else:
                best = BindingProof(threshold, proof)
                lo = middle + 1
        return best

    def unguarded_sources(self, upper, leaves):
        pending, seen, origins = [self.output], set(), set()
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            self.charge()
            gate = self.rules.get(node)
            if gate is None:
                origins.add(leaves[node][0][1])
                continue
            self.charge(len(gate.parents))
            if gate.operation == "or":
                pending.append(max(gate.parents, key=upper.__getitem__))
            else:
                pending.extend(gate.parents)
        return frozenset(origins)

    def matching_domains(self, leaves):
        domains = {node: {origin: value for value, origin in choices} for node, choices in leaves.items()}
        for node, gate in self.rules.items():
            if node == self.output:
                break
            merged = {}
            for parent in gate.parents:
                self.charge(1 + len(domains[parent]))
                for origin, value in domains[parent].items():
                    merged[origin] = max(merged.get(origin, 0.0), value)
            domains[node] = merged
        return tuple(domains[parent] for parent in self.rules[self.output].parents)

    def match(self, threshold, domains):
        choices = []
        for domain in domains:
            self.charge(1 + len(domain))
            choices.append(tuple(sorted(origin for origin, value in domain.items() if value >= threshold)))
        owners, selected = {}, {}
        for start in sorted(range(len(choices)), key=lambda branch: len(choices[branch])):
            pending, visited = deque([start]), {start}
            via = {}
            free = None
            while pending and free is None:
                branch = pending.popleft()
                for origin in choices[branch]:
                    self.charge()
                    if origin in via:
                        continue
                    via[origin] = branch
                    owner = owners.get(origin)
                    if owner is None:
                        free = origin
                        break
                    if owner not in visited:
                        pending.append(owner)
                        visited.add(owner)
            if free is None:
                return None
            # Follow the alternating path back to the unassigned starting branch.
            while free is not None:
                self.charge()
                branch = via[free]
                owners[free] = branch
                previous = selected.get(branch)
                selected[branch] = free
                free = previous
        return frozenset(owners)

    def witness(self, threshold, upper, leaves):
        scopes = count()
        stack = [(_Task(self.output, ()), {}, frozenset())]
        while stack:
            task, claims, used = stack.pop()
            while task is not None:
                self.charge()
                node, guards, following = task.node, task.guards, task.following
                if upper[node] < threshold:
                    break
                gate = self.rules.get(node)
                if gate is None:
                    for strength, source in reversed(leaves[node]):
                        self.charge(1 + len(guards))
                        if strength < threshold:
                            continue
                        if any(claims.get((scope, source), branch) != branch for scope, branch in guards):
                            continue
                        next_claims = claims | {(scope, source): branch for scope, branch in guards}
                        stack.append((following, next_claims, used | {source}))
                    break
                self.charge(len(gate.parents))
                if gate.operation == "or":
                    for parent in sorted(gate.parents, key=upper.__getitem__):
                        if upper[parent] >= threshold:
                            stack.append((_Task(parent, guards, following), claims, used))
                    break
                # Give each invocation its own sibling constraint, even for reused gates.
                scope = next(scopes) if gate.operation == "disjoint_and" else None
                for branch in range(len(gate.parents) - 1, -1, -1):
                    inherited = guards if scope is None else guards + ((scope, branch),)
                    following = _Task(gate.parents[branch], inherited, following)
                task = following
            else:
                return used
        return None


def continue_bindings(seed, *, memory=None, output=None, without_sources=(), work_limit=65536):
    """Clamp each late seed coordinate by the strongest support at the selected node."""
    seed = _real_array(seed)
    if seed.ndim != 1 or not seed.size or np.any((seed < 0) | (seed > 1)):
        raise ValueError("binding seed must be a nonempty vector in [0, 1]")
    excluded = _exclusions(without_sources)
    if memory is None:
        if output is not None or excluded:
            raise ValueError("an output node or source exclusion requires binding memory")
        return _frozen(seed)
    if type(memory) is not BindingMemory or seed.shape != (memory.width,):
        raise ValueError("binding seed must match the memory width")
    if output not in memory._dependents:
        raise ValueError("unknown binding node")
    if not memory.materialize:
        search = _ProofSearch(memory, output, excluded, work_limit)
        strengths = np.full(memory.width, -np.inf)
        for binding in range(memory.width):
            proof = search.resolve(binding)
            if proof is not None:
                strengths[binding] = proof.strength
        return _frozen(np.minimum(seed, strengths))
    rows = (memory._rows.get((output, binding), _UNSUPPORTED) for binding in range(memory.width))
    if excluded:
        scores = (max((value for origins, value in row.terms.items() if origins.isdisjoint(excluded)),
                      default=-np.inf) for row in rows)
    else:
        scores = (row.strongest for row in rows)
    strongest = np.fromiter(scores, dtype=np.float64, count=memory.width)
    return _frozen(np.minimum(seed, strongest))
