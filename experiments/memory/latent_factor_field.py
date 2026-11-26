"""Bidirectional latent recurrence over source-scoped compatibility tensors.

This carrier uses log-domain sum-product messages. Forests and compiled
cluster trees admit exact component reads. Raw loopy messages are proposals.
"""

from collections import deque
from dataclasses import dataclass, replace
from operator import index
from types import MappingProxyType

import numpy as np

from experiments.memory._latent_factor_clusters import (
    _ClusterBank, _ClusterPlan, _refresh_clusters, _relative_log, _settle_clusters, _signature,
)
from experiments.memory.corpus_bindings import CorpusRevision, StaleCorpusRevision
from experiments.memory.latent_feedback import _frozen, _real_array
from experiments.memory.verified_source_tool import SourceSpan


def _limits(updates, cells):
    limits = index(updates), index(cells)
    if min(limits) < 0:
        raise ValueError("recurrence budgets must be nonnegative")
    return limits


def _weights(value, shape):
    values = _real_array(value)
    if values.shape != shape or np.any(values < 0):
        raise ValueError("expected nonnegative weights with the declared latent shape")
    with np.errstate(divide="ignore"):
        return _frozen(_relative_log(np.log(values)))


@dataclass(frozen=True, eq=False)
class _Factor:
    axes: tuple[int, ...]
    potential: np.ndarray
    source: SourceSpan


@dataclass(frozen=True, eq=False)
class LatentFactorBatch:
    source: str
    _field: object
    _corpus: CorpusRevision
    _factors: MappingProxyType


@dataclass(frozen=True, eq=False)
class _Graph:
    factors: MappingProxyType
    adjacent: tuple[tuple, ...]
    components: tuple[int, ...]
    loopy: frozenset[int]
    groups: MappingProxyType
    sources: MappingProxyType

    @classmethod
    def build(cls, widths, factors):
        adjacent = [[] for _ in widths]
        for key, factor in factors.items():
            for slot in factor.axes:
                adjacent[slot].append(key)
        components, loopy = [None] * len(widths), set()
        for root in range(len(widths)):
            if components[root] is not None:
                continue
            slots, incident = [root], set()
            components[root] = root
            edges = 0
            for slot in slots:
                edges += len(adjacent[slot])
                for key in adjacent[slot]:
                    if key in incident:
                        continue
                    incident.add(key)
                    for other in factors[key].axes:
                        if components[other] is None:
                            slots.append(other)
                            components[other] = root
            if edges >= len(slots) + len(incident):
                loopy.add(root)
        groups = {}
        for key, factor in factors.items():
            groups.setdefault(components[factor.axes[0]], []).append(key)
        sources = {root: tuple(dict.fromkeys(factors[key].source for key in keys)) for root, keys in groups.items()}
        return cls(MappingProxyType(factors), tuple(map(tuple, adjacent)), tuple(components), frozenset(loopy),
                   MappingProxyType({root: tuple(keys) for root, keys in groups.items()}), MappingProxyType(sources))


@dataclass(frozen=True, eq=False)
class _Messages:
    banks: MappingProxyType
    pending: tuple
    pending_components: frozenset[int]
    updates: int
    tensor_cells: int


@dataclass(frozen=True, eq=False)
class LatentFactorPin:
    corpus: CorpusRevision
    uncompiled_sources: frozenset[str]
    _field: object
    _versions: MappingProxyType
    _graph: _Graph
    _state: _Messages
    _clusters: MappingProxyType
    cluster_updates: int
    cluster_tensor_cells: int

    @property
    def updates(self):
        return self._state.updates + self.cluster_updates

    @property
    def tensor_cells(self):
        return self._state.tensor_cells + self.cluster_tensor_cells


    @property
    def clustered_components(self):
        return len(self._clusters)

    @property
    def retained_cluster_cells(self):
        return sum(bank.plan.retained_cells for bank in self._clusters.values())


@dataclass(frozen=True, eq=False)
class LatentFactorRead:
    status: str
    probabilities: np.ndarray | None  # Diagnostic when support is withheld.
    sources: tuple[SourceSpan, ...]
    conditioned_slots: tuple[int, ...]
    support: np.ndarray | None


@dataclass(frozen=True, eq=False)
class LatentFactorState:
    pin: LatentFactorPin | None
    _field: object
    _graph: _Graph
    _drives: MappingProxyType
    _state: _Messages
    _clusters: MappingProxyType
    cluster_updates: int
    cluster_tensor_cells: int

    @property
    def updates(self):
        return self._state.updates + self.cluster_updates

    @property
    def tensor_cells(self):
        return self._state.tensor_cells + self.cluster_tensor_cells

    def read(self, slot):
        """Read one slot and its component dependencies after recurrence."""
        slot = self._field._slot(slot)
        root = self._graph.components[slot]
        cluster = self._clusters.get(root)
        if cluster is not None:
            return self._cluster_read(root, cluster, cluster.plan.owner[slot], (slot,))
        bank = self._state.banks.get(root, {})
        log_mass = self._field._cavity(self._graph, bank, self._drives, slot)
        return self._read_mass(root, log_mass)

    def relation(self, source, ordinal):
        """Read the joint distribution on one compiled relation's latent axes."""
        key = (source, index(ordinal))
        factor = self._graph.factors[key]
        root = self._graph.components[factor.axes[0]]
        cluster = self._clusters.get(root)
        if cluster is not None:
            return self._cluster_read(root, cluster, cluster.plan.host[key], factor.axes)
        log_mass = self._field._factor_mass(self._graph, self._state.banks[root], self._drives, key)
        return self._read_mass(root, log_mass)

    def _cluster_read(self, root, cluster, bag, axes):
        log_mass = cluster.project(bag, axes, self._drives) if cluster.can_read(bag) else None
        return self._read_mass(root, log_mass, clustered=True)

    def _read_mass(self, root, log_mass, *, clustered=False):
        sources = self._graph.sources.get(root, ())
        conditioned = tuple(sorted(slot for slot in self._drives if self._graph.components[slot] == root))
        support = np.isfinite(log_mass) if log_mass is not None else None
        feasible = np.count_nonzero(support) if support is not None else 0
        probabilities = None
        if feasible:
            mass = np.exp(_relative_log(log_mass))
            probabilities = _frozen(mass / mass.sum())
        if self.pin is not None and self.pin.uncompiled_sources:
            status = "uncompiled"
        elif root in self._graph.loopy and not clustered:
            status = "loopy"
        elif log_mass is None or (not clustered and root in self._state.pending_components):
            status = "update_limit"
        else:
            status = "inconsistent" if not feasible else ("supported" if feasible == 1 else "alternatives")
            if conditioned:
                status = "conditional_" + status
            return LatentFactorRead(status, probabilities, sources, conditioned, _frozen(support) if feasible else None)
        return LatentFactorRead(status, probabilities, sources, conditioned, None)


class LatentFactorField:
    """Retain task-blind messages on source pins and revise them in either direction."""

    def __init__(self, widths):
        self._widths = tuple(map(index, widths))
        if not self._widths or min(self._widths) < 1:
            raise ValueError("latent slots require positive widths")
        self._neutral = tuple(_frozen(np.zeros(width)) for width in self._widths)
        self._empty = _Graph.build(self._widths, {})
        self._no_evidence = MappingProxyType({})
        self._no_clusters = MappingProxyType({})
        self._no_messages = _Messages(MappingProxyType({}), (), frozenset(), 0, 0)

    def _slot(self, slot):
        slot = index(slot)
        if not 0 <= slot < len(self._widths):
            raise IndexError("latent slot outside the field")
        return slot

    def compile_source(self, corpus, source, entries):
        """Prepare source-local (mention ordinal, ordered axes, likelihood tensor) entries.

        Use entries derived only from this source's bytes and a fixed field
        compiler. Factor multiplication assumes the caller's factorization.
        An empty batch records that the caller assessed the source.
        """
        if type(corpus) is not CorpusRevision or source not in corpus.memory.source_digests:
            raise ValueError("a source on the corpus revision is required")
        factors, scopes = {}, set()
        for ordinal, axes, values in entries:
            ordinal = index(ordinal)
            key = source, ordinal
            axes = tuple(map(self._slot, axes))
            if not axes or len(set(axes)) != len(axes) or key in factors:
                raise ValueError("factor axes and mention keys must be distinct and nonempty")
            span = corpus.memory.source_mention_span(source, ordinal)
            if span is None:
                raise ValueError("a factor needs a scoped source observation")
            scope = span, frozenset(axes)
            if scope in scopes:
                raise ValueError("combine repeated factors over the same source span and axes before installation")
            scopes.add(scope)
            potential = _weights(values, tuple(self._widths[slot] for slot in axes))
            factors[key] = _Factor(axes, potential, span)
        return LatentFactorBatch(source, self, corpus, MappingProxyType(factors))

    @staticmethod
    def _source_version(corpus, source):
        digest = corpus.memory.source_digests.get(source)
        return None if digest is None else (digest, corpus._source_revisions.get(source, 0))

    def prepare(self, corpus, updates, *, previous=None, max_updates=1024, max_tensor_cells=1_000_000):
        """Assemble independently compiled batches on one corpus pin and update its retained messages."""
        if type(corpus) is not CorpusRevision:
            raise ValueError("a corpus revision is required")
        self._require_pin(previous)
        if previous is not None and previous.corpus._owner is not corpus._owner:
            raise ValueError("corpus revision belongs to a different workspace")
        changes = {}
        for batch in updates:
            if type(batch) is not LatentFactorBatch or batch._field is not self:
                raise ValueError("expected a source batch from this field")
            source = batch.source
            if (batch._corpus._owner is not corpus._owner
                    or self._source_version(batch._corpus, source) != self._source_version(corpus, source)):
                raise StaleCorpusRevision("factor batch source changed after compilation")
            if source in changes:
                raise ValueError("each source requires one compiled batch")
            changes[source] = batch
        old_graph = self._empty if previous is None else previous._graph
        old_state = self._no_messages if previous is None else previous._state
        versions = {} if previous is None else {
            name: version for name, version in previous._versions.items()
            if self._source_version(corpus, name) == version
        }
        factors = {key: factor for key, factor in old_graph.factors.items()
                   if key[0] in versions and key[0] not in changes}
        for source, batch in changes.items():
            versions[source] = self._source_version(corpus, source)
            for key, factor in batch._factors.items():
                prior = old_graph.factors.get(key)
                same = (prior is not None and factor.source == prior.source and factor.axes == prior.axes
                        and np.array_equal(factor.potential, prior.potential))
                factors[key] = prior if same else factor

        if factors == old_graph.factors:
            graph, changed_slots, changed_keys = old_graph, set(), set()
        else:
            graph = _Graph.build(self._widths, factors)
            changed_keys = {key for key in old_graph.factors.keys() | factors.keys()
                            if old_graph.factors.get(key) is not factors.get(key)}
            changed_slots = {slot for bank in (old_graph.factors, factors) for key in changed_keys & bank.keys()
                             for slot in bank[key].axes}
        limits = _limits(max_updates, max_tensor_cells)
        clusters = self._no_clusters if previous is None else _refresh_clusters(previous._clusters, graph, changed_keys)
        state, clusters, updates, cells = self._advance(
            graph, old_state, clusters, self._no_evidence, changed_slots, limits,
            prior_graph=old_graph, changed_keys=changed_keys,
        )
        return LatentFactorPin(corpus, frozenset(corpus.memory.source_digests.keys() - versions.keys()), self,
                               MappingProxyType(versions), graph, state, clusters, updates, cells)

    def maintain_clusters(self, pin, *, max_bag_cells, max_retained_cells, max_component_slots=256,
                          max_updates=1024, max_tensor_cells=1_000_000):
        """Compile bounded cyclic regions before a question is chosen."""
        self._require_pin(pin)
        if pin is None:
            raise ValueError("cluster maintenance requires a source pin")
        limits = _limits(max_updates, max_tensor_cells)
        max_bag_cells, remaining, slot_limit = map(index, (max_bag_cells, max_retained_cells, max_component_slots))
        if min(max_bag_cells, remaining, slot_limit) < 1:
            raise ValueError("cluster construction limits must be positive")
        graph = pin._graph
        sizes = {}
        for root in graph.components:
            sizes[root] = sizes.get(root, 0) + 1
        selected = {}
        for root in sorted(graph.loopy):
            if sizes[root] > slot_limit or remaining < 1:
                continue
            bank = pin._clusters.get(root)
            if bank is None:
                plan = _ClusterPlan.build(self._widths, _signature(graph, root),
                                          max_bag_cells=max_bag_cells, max_retained_cells=remaining)
                if plan is None:
                    continue
                bank = _ClusterBank.initialize(plan, graph.factors)
            if bank.plan.peak_cells <= max_bag_cells and bank.plan.retained_cells <= remaining:
                selected[root] = bank
                remaining -= bank.plan.retained_cells
        raw, clusters, updates, cells = self._advance(graph, pin._state, selected, self._no_evidence, (), limits)
        return replace(pin, _clusters=clusters, cluster_updates=updates, cluster_tensor_cells=cells, _state=raw)

    def _require_pin(self, pin):
        if pin is not None and (type(pin) is not LatentFactorPin or pin._field is not self):
            raise ValueError("factor pin belongs to a different field")

    def reason(self, evidence, *, memory=None, max_updates=1024, max_tensor_cells=1_000_000):
        """Condition numeric slots with or without a retained corpus pin."""
        self._require_pin(memory)
        graph, state = (self._empty, self._no_messages) if memory is None else (memory._graph, memory._state)
        clusters = self._no_clusters if memory is None else memory._clusters
        return self._condition(memory, graph, state, clusters, self._no_evidence, evidence,
                               _limits(max_updates, max_tensor_cells))

    def revise(self, state, evidence, *, max_updates=1024, max_tensor_cells=1_000_000):
        """Replace task evidence or clear it with None, leaving the shared pin intact."""
        if type(state) is not LatentFactorState or state._field is not self:
            raise ValueError("latent state belongs to a different field")
        return self._condition(state.pin, state._graph, state._state, state._clusters, state._drives, evidence,
                               _limits(max_updates, max_tensor_cells))

    def _condition(self, pin, graph, state, clusters, drives, evidence, limits):
        revised, changed = dict(drives), set()
        for slot, values in dict(evidence).items():
            slot = self._slot(slot)
            neutral = self._neutral[slot]
            new = neutral if values is None else _weights(values, (self._widths[slot],))
            if not np.array_equal(new, drives.get(slot, neutral)):
                if np.array_equal(new, neutral):
                    revised.pop(slot, None)
                else:
                    revised[slot] = new
                changed.add(slot)
        revised = MappingProxyType(revised)
        result, clusters, updates, cells = self._advance(graph, state, clusters, revised, changed, limits)
        return LatentFactorState(pin, self, graph, revised, result, clusters, updates, cells)

    def _advance(self, graph, state, clusters, drives, changed, limits, *, prior_graph=None, changed_keys=()):
        raw = self._settle(graph, state, drives, changed, limits, prior_graph=prior_graph,
                           changed_keys=changed_keys, skip_roots=clusters, defer_loopy=bool(clusters))
        if not clusters:
            return raw, clusters, 0, 0
        remaining = limits[0] - raw.updates, limits[1] - raw.tensor_cells
        drive_changes = changed if prior_graph is None else ()
        clusters, updates, cells = _settle_clusters(clusters, drives, drive_changes, remaining)
        remaining = remaining[0] - updates, remaining[1] - cells
        if raw.pending_components & graph.loopy and min(remaining) > 0:
            resumed = self._settle(graph, raw, drives, (), remaining, skip_roots=clusters)
            raw = replace(resumed, updates=raw.updates + resumed.updates,
                          tensor_cells=raw.tensor_cells + resumed.tensor_cells)
        return raw, clusters, updates, cells

    def _cavity(self, graph, messages, drives, slot, *, omit=None):
        result = drives.get(slot, self._neutral[slot]).copy()
        for key in graph.adjacent[slot]:
            if key != omit:
                result += messages[key, slot]
        return _relative_log(result)

    def _factor_mass(self, graph, messages, drives, key, *, omit=None):
        factor = graph.factors[key]
        mass = factor.potential.copy()
        dimensions = [1] * len(factor.axes)
        for axis, slot in enumerate(factor.axes):
            if slot == omit:
                continue
            shape = dimensions.copy()
            shape[axis] = self._widths[slot]
            mass += self._cavity(graph, messages, drives, slot, omit=key).reshape(shape)
        return mass

    def _settle(self, graph, state, drives, changed_slots, limits, *, prior_graph=None, changed_keys=(),
                skip_roots=(), defer_loopy=False):
        update_limit, cell_limit = limits
        excluded = frozenset(skip_roots)
        reset = ({graph.components[slot] for slot in changed_slots} & graph.loopy) - excluded
        queued = {address: None for address in state.pending
                  if address[0] in graph.factors and address[1] in graph.factors[address[0]].axes
                  and graph.components[address[1]] not in excluded}

        def initialize(root, *, from_graph=None):
            bank = {}
            for key in graph.groups[root]:
                for target in graph.factors[key].axes:
                    address = key, target
                    if from_graph is None or key in changed_keys:
                        old = None
                    else:
                        prior_root = from_graph.components[target]
                        old = state.banks.get(prior_root, {}).get(address)
                    if old is None:
                        bank[address] = self._neutral[target]
                        queued[address] = None
                    else:
                        bank[address] = old
            return MappingProxyType(bank)

        if prior_graph is not None and prior_graph is not graph:
            banks = {}
            for root, keys in graph.groups.items():
                if root in excluded:
                    continue
                if root in state.banks and keys == prior_graph.groups.get(root) and changed_keys.isdisjoint(keys):
                    banks[root] = state.banks[root]
                else:
                    banks[root] = initialize(root, from_graph=prior_graph)
        else:
            banks = {root: bank for root, bank in state.banks.items() if root not in excluded}
            for root in sorted(graph.groups.keys() - banks.keys() - excluded):
                banks[root] = initialize(root)
        mutable = {}
        for root in sorted(reset):
            bank = dict(banks[root])
            for address in bank:
                bank[address] = self._neutral[address[1]]
                queued[address] = None
            mutable[root] = bank

        def propagate(slot, omit=None):
            if graph.components[slot] in excluded:
                return
            for key in graph.adjacent[slot]:
                if key != omit:
                    for target in graph.factors[key].axes:
                        if target != slot:
                            address = key, target
                            if address not in queued:
                                queued[address] = None
                                queues[graph.components[target] in graph.loopy].append(address)

        queues = (deque(), deque())
        for address in queued:
            queues[graph.components[address[1]] in graph.loopy].append(address)
        for slot in sorted(changed_slots):
            propagate(slot)
        updates = cells = 0
        deferred = []
        while (queues[0] or (not defer_loopy and queues[1])) and updates < update_limit:
            # Prioritize components that can deliver a supported read.
            key, target = address = (queues[0] or queues[1]).popleft()
            factor = graph.factors[key]
            size = factor.potential.size
            if size > cell_limit - cells:
                deferred.append(address)
                continue
            del queued[address]
            root = graph.components[target]
            if root not in mutable:
                mutable[root] = dict(banks[root])
            messages = mutable[root]
            mass = self._factor_mass(graph, messages, drives, key, omit=target)
            slab = np.moveaxis(mass, factor.axes.index(target), 0).reshape(self._widths[target], -1)
            result = _relative_log(np.logaddexp.reduce(slab, axis=1))
            updates += 1
            cells += size
            if not np.array_equal(messages[address], result):
                messages[address] = _frozen(result)
                propagate(target, omit=key)
        banks.update((root, MappingProxyType(bank)) for root, bank in mutable.items())
        pending = (*deferred, *queues[0], *queues[1])
        pending_components = frozenset(graph.components[target] for _, target in pending)
        return _Messages(MappingProxyType(banks), pending, pending_components, updates, cells)
