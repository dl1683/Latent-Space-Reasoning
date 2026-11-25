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
_Address = tuple[str, int]
_SearchNode = str | _Address | None


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
class _Derivation:
    strength: float
    leaves: frozenset[tuple[str | _Address, str]]

    def proof(self):
        return BindingProof(self.strength, frozenset(origin for _, origin in self.leaves))


@dataclass(frozen=True)
class _Retained:
    upper: float | None
    proof: BindingProof | None
    leaves: frozenset[tuple[str, str]]
    inputs: frozenset[str]

    @property
    def exact(self):
        return self.upper == -np.inf or (self.proof is not None and self.proof.strength == self.upper)

    def advance(self, changes, excluded):
        upper = self.upper
        lower = -np.inf if self.proof is None else self.proof.strength
        for node, origin, before, after in changes:
            if node not in self.inputs or origin in excluded:
                continue
            if upper is not None and before <= upper < after:
                upper = None
            if (node, origin) in self.leaves:
                lower = min(lower, after)
        if upper is None and lower == -np.inf:
            return None
        proof, leaves = self.proof, self.leaves
        if proof is not None and lower < proof.strength:
            if lower == -np.inf:
                proof, leaves = None, frozenset()
            else:
                proof = BindingProof(lower, proof.sources)
        if upper == self.upper and proof is self.proof:
            return self
        return _Retained(upper, proof, leaves, self.inputs)


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
    _retained: Mapping[tuple[str, int, frozenset[str]], _Retained]
    materialize: bool = True
    retained_limit: int = 64

    @classmethod
    def prepare(cls, predicates, width, gates, sources, *, term_limit=64, pair_limit=65536,
                materialize=True, retained_limit=64):
        """Compile gates in topological order and apply the initial source batch."""
        predicates = tuple(predicates)
        width, term_limit, pair_limit = index(width), index(term_limit), index(pair_limit)
        retained_limit = index(retained_limit)
        if min(width, term_limit, pair_limit, retained_limit) < 1:
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
            if (type(gate) is not BindingGate or type(gate.operation) is not str
                    or gate.operation not in ("or", "and", "disjoint_and")):
                raise ValueError("unknown binding gate")
            parents = tuple(gate.parents)
            if not parents or any(type(parent) is not str or parent not in known for parent in parents):
                raise ValueError("gate parents must precede the gate")
            for parent in set(parents):
                known[parent].append(len(ordered))
            known[name] = []
            ordered.append((name, BindingGate(gate.operation, parents)))
        dependents = MappingProxyType({node: tuple(users) for node, users in known.items()})
        positions = MappingProxyType({node: position for position, (node, _) in enumerate(ordered)})
        empty = cls(width, term_limit, pair_limit, frozenset(predicates), tuple(ordered), dependents,
                    positions, _EMPTY, _EMPTY, _EMPTY, materialize, retained_limit)
        return empty.revise(sources)

    def reprogram(self, gates):
        """Replace a factored gate graph while sharing its input rows and valid bounds."""
        if self.materialize:
            raise ValueError("reprogram requires a factored binding bank")
        layout = self.prepare(self._predicates, self.width, gates, {}, term_limit=self.term_limit,
                              pair_limit=self.pair_limit, materialize=False, retained_limit=self.retained_limit)
        if layout._gates == self._gates:
            return self

        old_gates = dict(self._gates)
        unchanged = set(self._predicates)
        for name, gate in layout._gates:
            if old_gates.get(name) == gate and unchanged.issuperset(gate.parents):
                unchanged.add(name)
        retained = {key: bound for key, bound in self._retained.items() if key[0] in unchanged}
        return replace(layout, _sources=self._sources, _rows=self._rows, _retained=MappingProxyType(retained))

    def revise(self, changes):
        """Replace all facts for each supplied source ID, or remove it with None."""
        sources = None
        inputs = {}
        deltas = {}
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
                before, after = previous.get(key, -np.inf), current.get(key, -np.inf)
                if before == after:
                    continue
                if self._retained:
                    node, binding = key
                    deltas.setdefault(binding, []).append((node, source, before, after))
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
        retained = self._retained
        if deltas:
            surviving = {}
            changed = False
            for key, result in retained.items():
                advanced = result.advance(deltas.get(key[1], ()), key[2])
                if advanced is not None:
                    surviving[key] = advanced
                changed |= advanced is not result
            if changed:
                retained = MappingProxyType(surviving)
        return replace(self, _sources=MappingProxyType(sources), _rows=MappingProxyType(rows), _retained=retained)

    def supports(self, node, binding):
        """Return the retained source sets and strengths for a specific binding."""
        if node not in self._dependents:
            raise ValueError("unknown binding node")
        if not self.materialize and node not in self._predicates:
            raise ValueError("factored gates expose a selected proof through resolve")
        return self._rows.get((node, _checked_binding(binding, self.width)), _UNSUPPORTED).terms

    def resolve(self, node, binding, *, without_sources=(), work_limit=65536):
        """Select a strongest proof, or return None after establishing no support."""
        node, binding, excluded = self._proof_key(node, binding, without_sources)
        if self.materialize:
            candidates = self.supports(node, binding)
            strength, origins = max(
                ((value, sources) for sources, value in candidates.items() if sources.isdisjoint(excluded)),
                key=lambda item: (item[0], tuple(sorted(item[1]))), default=(-np.inf, None),
            )
            return None if origins is None else BindingProof(strength, origins)
        work_limit = _proof_work_limit(work_limit)
        retained = self._retained.get((node, binding, excluded))
        if retained is not None and retained.exact:
            return retained.proof
        search = _ProofSearch(self, node, excluded, work_limit)
        return search.resolve(binding, retained)

    def resolve_disjoint(self, addresses, *, without_sources=(), work_limit=65536):
        """Find the strongest joint proof with disjoint origins across addressed roots."""
        work_limit = _proof_work_limit(work_limit)
        roots = self._joint_keys(addresses, work_limit)
        return self._resolve_disjoint(roots, _exclusions(without_sources, limit=work_limit), work_limit)

    def _joint_keys(self, addresses, work_limit):
        roots = []
        for address in addresses:
            if len(roots) == work_limit:
                raise BindingCapacityError("proof search work limit exceeded")
            node, binding = address
            if node not in self._dependents:
                raise ValueError("unknown binding node")
            roots.append((node, _checked_binding(binding, self.width)))
        if not roots:
            raise ValueError("a disjoint proof requires at least one binding address")
        return tuple(roots)

    def _resolve_disjoint(self, roots, excluded, work_limit):
        if len(roots) == 1:
            return self.resolve(*roots[0], without_sources=excluded, work_limit=work_limit)
        search = _ProofSearch(self, None, excluded, work_limit, addresses=roots)
        return search.resolve(None)

    def remember(self, node, binding, *, without_sources=(), work_limit=65536):
        """Return a factored snapshot with one completed proof read retained."""
        if self.materialize:
            raise ValueError("remember requires a factored binding bank")
        key = self._proof_key(node, binding, without_sources)
        work_limit = _proof_work_limit(work_limit)
        prior = self._retained.get(key)
        if prior is not None and prior.exact:
            return self
        node, binding, excluded = key
        search = _ProofSearch(self, node, excluded, work_limit)
        derivation = search.derive(binding, prior)
        if derivation is None:
            stored = _Retained(-np.inf, None, frozenset(), frozenset(search.inputs))
        else:
            stored = _Retained(derivation.strength, derivation.proof(), derivation.leaves, frozenset(search.inputs))
        retained = dict(self._retained)
        if key not in retained and len(retained) == self.retained_limit:
            del retained[next(iter(retained))]
        retained[key] = stored
        return replace(self, _retained=MappingProxyType(retained))

    def _proof_key(self, node, binding, without_sources):
        binding = _checked_binding(binding, self.width)
        if node not in self._dependents:
            raise ValueError("unknown binding node")
        return node, binding, _exclusions(without_sources)


def _proof_work_limit(work_limit):
    work_limit = index(work_limit)
    if work_limit < 1:
        raise ValueError("proof work limit must be positive")
    return work_limit


def _exclusions(without_sources, *, limit=None):
    if isinstance(without_sources, str):
        raise ValueError("without_sources must contain source IDs")
    if limit is not None:
        supplied = []
        for source in without_sources:
            if len(supplied) == limit:
                raise BindingCapacityError("proof search work limit exceeded")
            supplied.append(source)
        without_sources = supplied
    excluded = frozenset(without_sources)
    if any(type(source) is not str or not source for source in excluded):
        raise ValueError("without_sources must contain source IDs")
    return excluded


@dataclass(frozen=True, eq=False)
class _Task:
    node: _SearchNode
    guards: tuple[tuple[int, int], ...]
    following: "_Task | None" = None


class _ProofSearch:
    def __init__(self, memory, output, excluded, work_limit, *, addresses=()):
        self.remaining = _proof_work_limit(work_limit)
        self.rows, self.output, self.excluded = memory._rows, output, excluded
        pending, seen = list(addresses) if addresses else [output], set()
        inputs, positions = [], []
        while pending:
            key = pending.pop()
            if key in seen:
                continue
            seen.add(key)
            self.charge()
            node, coordinate = key if addresses else (key, None)
            position = memory._gate_index.get(node)
            if position is None:
                inputs.append(key)
            else:
                _, gate = memory._gates[position]
                self.charge(len(gate.parents))
                positions.append((position, coordinate))
                pending.extend(((parent, coordinate) for parent in gate.parents) if addresses else gate.parents)
        self.rules = {}
        for position, coordinate in sorted(positions):
            node, gate = memory._gates[position]
            # Reusing one proof for identical ordinary parents can only shrink its source set.
            parents = gate.parents if gate.operation == "disjoint_and" else tuple(dict.fromkeys(gate.parents))
            key = node
            if addresses:
                key = (node, coordinate)
                parents = tuple((parent, coordinate) for parent in parents)
            self.rules[key] = (gate.operation, parents)
        if addresses:
            self.charge(1 + len(addresses))
            self.rules[output] = ("disjoint_and", addresses)
        self.inputs = sorted(inputs)
        disjoint = [node for node, (operation, _) in self.rules.items() if operation == "disjoint_and"]
        self.unguarded = not disjoint
        self.single_source_matching = disjoint == [output] and all(
            node == output or operation == "or" for node, (operation, _) in self.rules.items()
        )

    def charge(self, amount=1):
        self.remaining -= amount
        if self.remaining < 0:
            raise BindingCapacityError("proof search work limit exceeded")

    def resolve(self, binding, retained=None):
        derivation = self.derive(binding, retained)
        return None if derivation is None else derivation.proof()

    def derive(self, binding, retained=None):
        leaves, upper, strengths = {}, {}, set()
        for node in self.inputs:
            address = node if binding is None else (node, binding)
            candidates = self.rows.get(address, _UNSUPPORTED).terms
            self.charge(1 + len(candidates))
            choices = [(value, next(iter(sources))) for sources, value in candidates.items()
                       if sources.isdisjoint(self.excluded)]
            leaves[node] = sorted(choices, reverse=True)
            upper[node] = max((value for value, _ in choices), default=-np.inf)
            strengths.update(value for value, _ in choices)
        for node, (operation, parents) in self.rules.items():
            self.charge(1 + len(parents))
            reduce = max if operation == "or" else min
            upper[node] = reduce(upper[parent] for parent in parents)
        # Ignoring source overlap yields an upper bound, not a feasible proof.
        ceiling = upper[self.output]
        if retained is not None and retained.upper is not None:
            ceiling = min(ceiling, retained.upper)
        if ceiling == -np.inf:
            return None
        if self.unguarded:
            return _Derivation(ceiling, self.unguarded_leaves(upper, leaves))
        best = None
        if retained is not None and retained.proof is not None:
            best = _Derivation(retained.proof.strength, retained.leaves)
            if best.strength == ceiling:
                return best
        domains = self.matching_domains(leaves) if self.single_source_matching else None

        def witness(threshold):
            if domains is not None:
                return self.match(threshold, domains)
            return self.witness(threshold, upper, leaves)

        proof = witness(ceiling)
        if proof is not None:
            return _Derivation(ceiling, proof)
        floor = -np.inf if best is None else best.strength
        levels = sorted(value for value in strengths if floor < value < ceiling)
        lo, hi = 0, len(levels)
        while lo < hi:
            middle = (lo + hi) // 2
            threshold = levels[middle]
            proof = witness(threshold)
            if proof is None:
                hi = middle
            else:
                best = _Derivation(threshold, proof)
                lo = middle + 1
        return best

    def unguarded_leaves(self, upper, leaves):
        pending, seen, selected = [self.output], set(), set()
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            self.charge()
            gate = self.rules.get(node)
            if gate is None:
                selected.add((node, leaves[node][0][1]))
                continue
            operation, parents = gate
            self.charge(len(parents))
            if operation == "or":
                pending.append(max(parents, key=upper.__getitem__))
            else:
                pending.extend(parents)
        return frozenset(selected)

    def matching_domains(self, leaves):
        domains = {node: {origin: (value, node) for value, origin in choices} for node, choices in leaves.items()}
        for node, (_, parents) in self.rules.items():
            if node == self.output:
                break
            merged = {}
            for parent in parents:
                self.charge(1 + len(domains[parent]))
                for origin, choice in domains[parent].items():
                    if origin not in merged or choice > merged[origin]:
                        merged[origin] = choice
            domains[node] = merged
        return tuple(domains[parent] for parent in self.rules[self.output][1])

    def match(self, threshold, domains):
        choices = []
        for domain in domains:
            self.charge(1 + len(domain))
            choices.append(tuple(sorted(origin for origin, (value, _) in domain.items() if value >= threshold)))
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
        return frozenset((domains[branch][origin][1], origin) for branch, origin in selected.items())

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
                        stack.append((following, next_claims, used | {(node, source)}))
                    break
                operation, parents = gate
                self.charge(len(parents))
                if operation == "or":
                    for parent in sorted(parents, key=upper.__getitem__):
                        if upper[parent] >= threshold:
                            stack.append((_Task(parent, guards, following), claims, used))
                    break
                # Give each invocation its own sibling constraint, even for reused gates.
                scope = next(scopes) if operation == "disjoint_and" else None
                for branch in range(len(parents) - 1, -1, -1):
                    inherited = guards if scope is None else guards + ((scope, branch),)
                    following = _Task(parents[branch], inherited, following)
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
        work_limit = _proof_work_limit(work_limit)
        search = None
        strengths = np.full(memory.width, -np.inf)
        for binding in range(memory.width):
            retained = memory._retained.get((output, binding, excluded))
            if retained is not None and retained.proof is not None and retained.proof.strength >= seed[binding]:
                # A feasible witness at the seed cap fixes the clamped value.
                strengths[binding] = seed[binding]
                continue
            if retained is None or not retained.exact:
                if search is None:
                    search = _ProofSearch(memory, output, excluded, work_limit)
                proof = search.resolve(binding, retained)
            else:
                proof = retained.proof
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
