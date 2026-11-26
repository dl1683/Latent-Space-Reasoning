"""Run one latent model against optional, journal-pinned source observations."""

from dataclasses import dataclass, replace
from math import isfinite
from operator import index
from types import MappingProxyType

import numpy as np

from experiments.memory import relational_feedback as relations
from experiments.memory.corpus_bindings import CorpusRevision
from experiments.memory.latent_feedback import FeedbackKernel, ObservationMemory, _apply_value, _frozen, _real_array
from experiments.memory.source_constraints import ConstraintUse, SourceConstraintPolicy, SourceConstraintPublication
from experiments.memory.world_transport import (
    DiscoveredTransportPolicy, DiscoveryCoverage, SourceTransportPolicy, SourceTransportPublication,
    SourceTransportUse, WorldWitness, require_transport_scope, transport_accepts, transport_authorities,
    transport_coordinate_key, transport_discovery_coverage, transport_observation_ordinals, world_kernel_digest,
)


def _within_relay_tolerance(actual, expected, tolerance):
    expected_scale = np.max(np.abs(expected))
    if expected_scale == 0.:
        return False
    scale = max(expected_scale, np.max(np.abs(actual)))
    return np.max(np.abs(actual - expected)) <= tolerance * scale


@dataclass(frozen=True, eq=False)
class _SourceBank:
    version: tuple[str, int]
    observations: ObservationMemory
    witnesses: tuple[WorldWitness, ...]
    matches: tuple[int, ...]
    indices: MappingProxyType | None

    def row_index(self, witness):
        return witness.ordinal if self.indices is None else self.indices[witness.ordinal]


@dataclass(frozen=True, eq=False)
class RecurrentWorldMemory:
    """A source snapshot with reusable observation keys and exact source pointers."""

    corpus: CorpusRevision
    _bank: ObservationMemory | None
    _key_map: np.ndarray
    _sources: MappingProxyType
    projected_rows: int
    packed_rows: int
    _witnesses: tuple[WorldWitness, ...]
    unlocated_sources: tuple[str, ...]
    _transport_policy: SourceTransportPolicy | DiscoveredTransportPolicy | None
    _source_transports: SourceTransportPublication | None
    _constraint_policy: SourceConstraintPolicy | None
    _source_constraints: SourceConstraintPublication | None
    _transport_coverage: DiscoveryCoverage | None

    @property
    def spans(self):
        return tuple(witness.span for witness in self._witnesses)

    def without_source_transport(self):
        """Remove only the published transport, retaining observations and coverage."""
        if self._source_transports is None:
            raise ValueError("memory has no published source transport to ablate")
        return replace(self, _source_transports=None)

    def without_source_constraints(self):
        """Remove only the published constraints, retaining observations and policy."""
        if self._source_constraints is None:
            raise ValueError("memory has no published source constraints to ablate")
        return replace(self, _source_constraints=None)


@dataclass(frozen=True, eq=False)
class RecurrentWorldRead:
    memory: RecurrentWorldMemory | None
    history: tuple[np.ndarray, ...]
    observation_scores: int
    feedback_scores: int
    packed_rows: int = 0

    @property
    def hidden(self):
        return self.history[-1].view()


@dataclass(frozen=True, eq=False, repr=False)
class JointWorldBranch:
    """One retained transition linked to its task-private parent."""

    _hidden: np.ndarray
    _source_hidden: np.ndarray
    _parent: "JointWorldBranch | None" = None
    _witness: WorldWitness | None = None
    _key: np.ndarray | None = None
    _value: np.ndarray | None = None
    _complete: bool = True
    _transport_use: SourceTransportUse | None = None
    _pending_destination: np.ndarray | None = None
    _source_relations: tuple[relations.Expression, ...] | None = None
    _constraint_use: ConstraintUse | None = None
    _constraint_input: MappingProxyType | None = None
    _unit_context: WorldWitness | None = None
    _unit_candidates: tuple[str, ...] = ()
    _catalog_actions: MappingProxyType | None = None
    _catalog_used: bool = False

    def _ancestry(self):
        node = self
        while node is not None:
            yield node
            node = node._parent

    @property
    def history(self):
        return tuple(node._hidden for node in reversed(tuple(self._ancestry())))

    @property
    def witnesses(self):
        return tuple(node._witness for node in reversed(tuple(self._ancestry())) if node._parent is not None)

    @property
    def hidden(self):
        return self._hidden.view()

    @property
    def source_hidden(self):
        """A representative point. Use the joint read to certify signed source evidence."""
        return self._source_hidden.view()

    @property
    def transport_uses(self):
        return tuple(node._transport_use for node in reversed(tuple(self._ancestry()))
                     if node._transport_use is not None)

    @property
    def constraint_uses(self):
        return tuple(node._constraint_use for node in reversed(tuple(self._ancestry()))
                     if node._constraint_use is not None)

    @property
    def unit_candidates(self):
        return self._unit_candidates


@dataclass(frozen=True)
class JointWorldOutcome:
    status: str
    support: tuple[JointWorldBranch, ...] = ()
    opposition: tuple[JointWorldBranch, ...] = ()
    unit_premises: tuple[str, ...] = ()


@dataclass(frozen=True, eq=False)
class _JointExpansion:
    parent: JointWorldBranch
    source_query: np.ndarray
    children: tuple[JointWorldBranch, ...]
    withheld: tuple[WorldWitness, ...]


@dataclass(frozen=True)
class _JointPlan:
    parent: JointWorldBranch
    source_query: np.ndarray
    choices: tuple[WorldWitness | None, ...]
    withheld: tuple[WorldWitness, ...]
    previous: dict[WorldWitness | None, JointWorldBranch]


@dataclass(frozen=True, eq=False)
class JointWorldRead:
    memory: RecurrentWorldMemory | None
    branches: tuple[JointWorldBranch, ...]
    _reasoner: object
    _thresholds: tuple[float, float]
    uncertain_routes: tuple[WorldWitness, ...] = ()
    steps: int = 0
    overflow: bool = False
    observation_scores: int = 0
    feedback_scores: int = 0
    transition_writes: int = 0
    block_evaluations: int = 0
    priority_scores: int = 0
    reused_writes: int = 0
    unit_premises: tuple[str, ...] = ()
    _layers: tuple[tuple[_JointExpansion, ...], ...] = ()
    _transport_withheld_contexts: frozenset[WorldWitness] = frozenset()

    def read(self, direction, *, floor):
        """Report signed branch observations within the configured numerical scope."""
        direction, floor = _real_array(direction), float(floor)
        if direction.shape != self.branches[0].hidden.shape or not isfinite(floor) or floor <= 0:
            raise ValueError("readout requires a matching direction and positive finite floor")
        if self.overflow:
            return JointWorldOutcome("hypothesis_limit", unit_premises=self.unit_premises)
        if self.memory is None:
            return JointWorldOutcome("memory_omitted", unit_premises=self.unit_premises)

        complete = tuple(branch for branch in self.branches
                         if branch._complete and branch._pending_destination is None
                         and (not self.unit_premises or branch._catalog_used)
                         and (not self._transport_withheld_contexts
                              or self._transport_withheld_contexts.isdisjoint(branch.witnesses)))
        support, opposition, open_readout = [], [], False
        with np.errstate(over="raise", invalid="raise"):
            for branch in complete:
                if branch._source_relations is None:
                    low = high = branch._source_hidden @ direction
                else:
                    low, high = relations.dot(branch._source_relations, direction).score_bounds()
                if low >= floor:
                    support.append(branch)
                elif high <= -floor:
                    opposition.append(branch)
                elif high >= floor or low <= -floor:
                    open_readout = True
        unused_premises = self.unit_premises and any(not branch._catalog_used for branch in self.branches)
        if (self.uncertain_routes or self.memory.unlocated_sources or open_readout or unused_premises
                or any(branch._pending_destination is not None for branch in self.branches)):
            status = "unresolved"
        elif not complete:
            status = "missing"
        elif len(complete) != len(self.branches):
            status = "unresolved"
        elif support and opposition:
            status = "conflict"
        elif len(support) == len(complete):
            status = "supported"
        elif len(opposition) == len(complete):
            status = "opposed"
        else:
            status = "undecided"
        return JointWorldOutcome(status, tuple(support), tuple(opposition), self.unit_premises)


@dataclass(frozen=True)
class JointStateComparison:
    """Diagnostic readouts for one ordinary suffix and its counterfactual."""

    journal_sha256: str
    transport_sha256: str | None
    depth: int
    channel: str
    recipient_prefix: tuple[WorldWitness | None, ...]
    donor_prefix: tuple[WorldWitness | None, ...]
    recipient_cut_score: float
    donor_cut_score: float
    first_route_change: int | None
    ordinary_status: str
    hypothetical_status: str
    ordinary_paths: tuple[tuple[WorldWitness | None, ...], ...]
    hypothetical_paths: tuple[tuple[WorldWitness | None, ...], ...]
    ordinary_block_evaluations: int
    hypothetical_block_evaluations: int


class RecurrentWorld:
    """Reuse a fixed latent block or feed back a growing sequence of latent thoughts."""

    def __init__(self, kernel, *, encoder_version, max_observations, feedback, transport_policy=None,
                 constraint_policy=None):
        if type(kernel) is not FeedbackKernel or type(feedback) is not bool:
            raise ValueError("a world reasoner requires a feedback kernel and a transition choice")
        if type(encoder_version) is not str or not encoder_version:
            raise ValueError("an observation encoder version is required")
        max_observations = index(max_observations)
        if max_observations < 1:
            raise ValueError("observation allowance must be positive")
        self._kernel = kernel
        self._encoder_version = encoder_version
        self._max_observations = max_observations
        self._feedback = feedback
        if transport_policy is not None and type(transport_policy) not in (SourceTransportPolicy, DiscoveredTransportPolicy):
            raise ValueError("transport policy must declare scoped source roles")
        if constraint_policy is not None and (type(constraint_policy) is not SourceConstraintPolicy
                                               or transport_policy is not None):
            raise ValueError("constraint reasoning requires its own declared policy")
        self._transport_policy = transport_policy
        self._constraint_policy = constraint_policy
        self._kernel_sha256 = (None if transport_policy is None and constraint_policy is None
                               else world_kernel_digest(kernel))

    def _require_corpus(self, corpus):
        if type(corpus) is not CorpusRevision or corpus.journal_sha256 is None:
            raise ValueError("the observation world requires a journaled source revision")
        memory = corpus.memory
        if memory.encoder_version != self._encoder_version or memory._anchors.shape[1] != self._kernel._width:
            raise ValueError("corpus observations use a different encoder frame")

    def maintain_transports(self, workspace, corpus):
        """Publish scoped source-derived operators before any task is known."""
        self._require_corpus(corpus)
        if self._transport_policy is None:
            raise ValueError("source transport maintenance requires a declared policy")
        return workspace.maintain_source_transports(corpus, self._transport_policy,
                                                    kernel_sha256=self._kernel_sha256)

    def maintain_constraints(self, workspace, corpus, source_records):
        """Publish reusable relations from exact source records without a task seed."""
        self._require_corpus(corpus)
        if self._constraint_policy is None:
            raise ValueError("constraint maintenance requires a declared policy")
        return workspace.maintain_source_constraints(corpus, self._constraint_policy, source_records,
                                                      kernel_sha256=self._kernel_sha256)

    def prepare(self, corpus, *, previous=None, packed=True, operators=True):
        """Project changed source generations and load any applicable operator publication."""
        if type(packed) is not bool or type(operators) is not bool:
            raise ValueError("packed and operators must be boolean")
        self._require_corpus(corpus)
        memory = corpus.memory
        policy = self._transport_policy
        constraint_policy = self._constraint_policy
        if policy is not None:
            require_transport_scope(corpus, policy)
        if constraint_policy is not None:
            constraint_policy.require_frame(corpus)
        authorities = ()
        if policy is not None:
            authorities = transport_authorities(corpus, policy)
        elif constraint_policy is not None:
            authorities = constraint_policy.authorities
        scoped = type(policy) is DiscoveredTransportPolicy and policy.scoped_clauses
        sources_in_scope = memory.source_digests.keys() if scoped else memory.source_digests.keys() - authorities
        coverage = (transport_discovery_coverage(corpus, policy, authorities)
                    if operators and type(policy) is DiscoveredTransportPolicy else None)
        transport = None
        publication = corpus._source_transports
        if operators and policy is not None and publication is not None and publication.active_on(corpus):
            if publication.policy != policy or publication.kernel_sha256 != self._kernel_sha256:
                raise ValueError("active source transports require their exact policy and kernel")
            transport = publication
        constraint = None
        if operators and constraint_policy is not None:
            publication = corpus._source_constraints.get((constraint_policy, self._kernel_sha256))
            if publication is not None and publication.active_on(corpus):
                constraint = publication
        if previous is not None:
            self._require_memory(previous)
            if corpus._owner is not previous.corpus._owner:
                raise ValueError("previous observations belong to a different journal handle")
        if not sources_in_scope <= memory._mentions.keys():
            raise ValueError("world observations are unavailable for legacy row-only sources")
        selected = {}
        for source in sources_in_scope:
            mentions = memory._mentions[source]
            indices = transport_observation_ordinals(mentions) if scoped else range(len(mentions))
            if indices or not mentions:
                selected[source] = indices
        if sum(map(len, selected.values())) > self._max_observations:
            raise ValueError("world observation allowance exceeded")

        sources = {}
        projected = 0
        prior = {} if previous is None else previous._sources
        for source, indices in sorted(selected.items()):
            if (policy is not None and policy.alignment is not None
                    and any(row[0] in ("transport_entry", "transport_exit") for row in memory._mentions[source])):
                raise ValueError("aligned calibration predicates require a declared calibration source")
            digest = memory.source_digests[source]
            version = digest, corpus._source_revisions[source]
            bank = prior.get(source)
            if bank is None or bank.version != version:
                mentions = memory.source_mentions(source)
                spans = tuple(memory.source_mention_span(source, n) for n in indices)
                if any(span is None for span in spans):
                    raise ValueError("world observations require exact mention spans")
                rows = np.asarray([mentions[n].vector for n in indices]).reshape(-1, self._kernel._width)
                witnesses = tuple(WorldWitness(span, version[1], n) for n, span in zip(indices, spans, strict=True))
                matches = tuple(memory._mentions[source][n][3] for n in indices)
                row_indices = (MappingProxyType({n: i for i, n in enumerate(indices)}) if scoped else None)
                bank = _SourceBank(version, self._kernel.prepare(rows), witnesses, matches, row_indices)
                projected += len(indices)
            sources[source] = bank
        if previous is not None and sources == prior and (not packed or previous._bank is not None):
            return replace(previous, corpus=corpus, _bank=previous._bank if packed else None,
                           projected_rows=0, packed_rows=0, _source_transports=transport,
                           _source_constraints=constraint, _transport_coverage=coverage)
        dense = self._pack_sources(sources) if packed else None
        witnesses = tuple(witness for bank in sources.values() for witness in bank.witnesses)
        unlocated = tuple(source for source, bank in sources.items() if not bank.matches or 0 in bank.matches)
        packed_rows = len(witnesses) if packed else 0
        return RecurrentWorldMemory(corpus, dense, self._kernel._observation_key, MappingProxyType(sources),
                                    projected, packed_rows, witnesses, unlocated, policy, transport,
                                    constraint_policy, constraint, coverage)

    def _pack_sources(self, sources):
        if not sources:
            return self._kernel._empty
        banks = tuple(source.observations for source in sources.values())
        return ObservationMemory(
            self._kernel._observation_key,
            _frozen(np.concatenate([bank._keys for bank in banks])),
            _frozen(np.concatenate([bank._observations for bank in banks])),
        )

    def _require_memory(self, memory):
        if (type(memory) is not RecurrentWorldMemory
                or memory._key_map is not self._kernel._observation_key
                or memory.corpus.memory.encoder_version != self._encoder_version
                or memory._transport_policy != self._transport_policy
                or memory._constraint_policy != self._constraint_policy
                or (memory._source_transports is not None
                    and memory._source_transports.kernel_sha256 != self._kernel_sha256)
                or (memory._source_constraints is not None
                    and memory._source_constraints.kernel_sha256 != self._kernel_sha256)
                or len(memory._witnesses) > self._max_observations):
            raise ValueError("memory belongs to a different observation frame or exceeds the allowance")

    def reason(self, seed, *, steps, memory=None, replace_first=None):
        """Keep input and readout unchanged when resident world memory is attached."""
        steps = index(steps)
        if not 1 <= steps <= self._kernel._max_steps:
            raise ValueError("steps must fit the latent block horizon")
        bank = None
        packed_rows = 0
        if memory is not None:
            self._require_memory(memory)
            if memory._source_transports is not None or memory._transport_coverage is not None:
                raise ValueError("source transports require joint reasoning for scoped evidence")
            if memory._constraint_policy is not None:
                raise ValueError("source constraints require joint reasoning for relational evidence")
            bank = memory._bank
            if bank is None:
                bank = self._pack_sources(memory._sources)
                packed_rows = len(bank._keys)
        if replace_first is not None:
            replace_first = self._kernel._vector(replace_first)
        state = self._kernel.start(seed, bank)
        if self._feedback:
            for step in range(steps):
                state = self._kernel.step(state)
                if step == 0 and replace_first is not None:
                    state = self._kernel.intervene(state, 0, replace_first)
            return RecurrentWorldRead(memory, state.history, state.observation_scores, state.feedback_scores, packed_rows)

        history = []
        for step in range(steps):
            preactivation, _, _ = self._kernel._step_preactivation(state)
            hidden = _frozen(np.tanh(preactivation))
            if step == 0 and replace_first is not None:
                hidden = replace_first
            history.append(hidden)
            state = replace(state, _seed=hidden)
        return RecurrentWorldRead(memory, tuple(history), steps * len(state._memory._keys), 0, packed_rows)

    def reason_joint(self, seed, *, steps, max_hypotheses, null_logit, margin, memory=None, unit_candidates=None):
        """Advance locally distinct hypotheses instead of pooling their observations."""
        null_logit, margin = float(null_logit), float(margin)
        lower, upper = null_logit - margin, null_logit + margin
        if not isfinite(lower) or not isfinite(upper) or not lower < upper:
            raise ValueError("joint reasoning requires a finite null score and positive margin")
        if memory is not None:
            self._require_memory(memory)
        if unit_candidates is not None:
            policy = self._constraint_policy
            if (policy is None or not policy.unit_catalog or type(unit_candidates) is not tuple
                    or not unit_candidates or len(unit_candidates) > index(max_hypotheses)
                    or any(type(name) is not str or len(name) > 128 or not name.isidentifier()
                           for name in unit_candidates)
                    or len(unit_candidates) != len(set(unit_candidates))):
                raise ValueError("unit candidates must be distinct registry names within the hypothesis limit")
        initial = self._kernel._vector(seed)
        publication = None if memory is None else memory._source_constraints
        additions = None
        if publication is not None:
            points = relations.point(initial)
            additions = MappingProxyType({action: action.project(publication.policy, points)
                                          for action in dict.fromkeys(publication.actions.values())})
        openings = []
        for names, actions in self._unit_hypotheses(publication, unit_candidates):
            projected = (additions if not actions else MappingProxyType(
                {action: action.project(publication.policy, points) for action in actions.values()},
            ))
            openings.append(JointWorldBranch(initial, initial, _constraint_input=projected,
                                             _unit_candidates=names, _catalog_actions=actions))
        state = JointWorldRead(memory, tuple(openings), self, (lower, upper), unit_premises=unit_candidates or ())
        return self.continue_joint(state, steps=steps, max_hypotheses=max_hypotheses)

    @staticmethod
    def _unit_hypotheses(publication, candidates):
        if candidates is None:
            return (((), None),)
        if publication is None or publication.catalog is None and not publication.scoped_catalogs:
            return ((candidates, None),)
        groups = {}
        for name in candidates:
            actions = publication.unit_actions(name)
            key = tuple((source, action.relations, action.use) for source, action in actions.items())
            names, representative = groups.setdefault(key, ([], actions))
            names.append(name)
        return tuple((tuple(names), MappingProxyType({
            source: replace(action, use=replace(action.use, source_units=tuple(names)))
            for source, action in actions.items()
        })) for names, actions in groups.values())

    def continue_joint(self, state, *, steps, max_hypotheses):
        """Resume at a committed frontier on the same source and operator pin."""
        steps, max_hypotheses = index(steps), index(max_hypotheses)
        if type(state) is not JointWorldRead or state._reasoner is not self:
            raise ValueError("joint state belongs to a different reasoner")
        last = state.steps + steps
        if steps < 1 or last > self._kernel._max_steps or max_hypotheses < len(state.branches):
            raise ValueError("joint continuation exceeds the horizon or cannot hold the current frontier")
        return self._expand_joint(state, last, max_hypotheses)

    def compare_joint_state(self, recipient, donor, *, steps, max_hypotheses, direction, floor,
                            recipient_branch=0, donor_branch=0, channel="joint"):
        """Compare suffixes after replacing one reached numerical state at a fixed pin."""
        if type(channel) is not str or channel not in ("joint", "source", "task"):
            raise ValueError("state comparison channel must be joint, source or task")
        if any(type(state) is not JointWorldRead or state._reasoner is not self for state in (recipient, donor)):
            raise ValueError("joint prefixes must come from the same reasoner")
        if recipient.memory is None or recipient.memory is not donor.memory:
            raise ValueError("joint prefixes must share the exact prepared memory")
        if recipient.memory._constraint_policy is not None:
            raise ValueError("joint state comparison requires point states without relational constraints")
        if (not recipient.steps or recipient.steps != donor.steps or recipient.overflow or donor.overflow
                or recipient._thresholds != donor._thresholds):
            raise ValueError("joint prefixes must have the same committed depth and routing thresholds")
        selected = []
        for state, branch_index in ((recipient, recipient_branch), (donor, donor_branch)):
            branch_index = index(branch_index)
            if not 0 <= branch_index < len(state.branches):
                raise ValueError("selected branch is outside the joint frontier")
            if len(state._layers) != state.steps or state.branches != tuple(
                child for expansion in state._layers[-1] for child in expansion.children
            ):
                raise ValueError("joint prefixes must contain their committed frontier")
            branch = state.branches[branch_index]
            if not branch._complete:
                raise ValueError("state comparison requires reached source-backed prefixes")
            selected.append(branch)
        original, source = selected
        if (source._pending_destination is not None or donor.uncertain_routes or donor.memory.unlocated_sources
                or not donor._transport_withheld_contexts.isdisjoint(source.witnesses)):
            raise ValueError("donor state has unresolved source obligations")
        direction, floor = self._kernel._vector(direction), float(floor)
        if not isfinite(floor) or floor <= 0.:
            raise ValueError("state comparison requires a positive finite readout floor")

        with np.errstate(over="raise", invalid="raise"):
            recipient_cut = float(original._source_hidden @ direction)
            donor_cut = float(source._source_hidden @ direction)

        # Preserve the task seed, earlier history, pending proof obligations
        # and every sibling. The generated key and value belong to the donor row.
        task_state = original if channel == "source" else source
        source_state = original if channel == "task" else source
        midpoint = replace(original, _hidden=task_state._hidden, _source_hidden=source_state._source_hidden,
                           _key=task_state._key, _value=task_state._value)
        branches = tuple(midpoint if branch is original else branch for branch in recipient.branches)
        ordinary = self.continue_joint(recipient, steps=steps, max_hypotheses=max_hypotheses)
        hypothetical = self.continue_joint(replace(recipient, branches=branches),
                                           steps=steps, max_hypotheses=max_hypotheses)
        first_route_change = self._first_route_change(ordinary, hypothetical, after=recipient.steps)
        publication = recipient.memory._source_transports
        transport_sha = None if publication is None else next(iter(publication.operations.values())).use.publication_sha256
        # An intervened path can diagnose computation but cannot furnish evidence.
        return JointStateComparison(
            journal_sha256=recipient.memory.corpus.journal_sha256,
            transport_sha256=transport_sha,
            depth=recipient.steps,
            channel=channel,
            recipient_prefix=original.witnesses,
            donor_prefix=source.witnesses,
            recipient_cut_score=recipient_cut,
            donor_cut_score=donor_cut,
            first_route_change=first_route_change,
            ordinary_status=ordinary.read(direction, floor=floor).status,
            hypothetical_status=hypothetical.read(direction, floor=floor).status,
            ordinary_paths=tuple(branch.witnesses for branch in ordinary.branches),
            hypothetical_paths=tuple(branch.witnesses for branch in hypothetical.branches),
            ordinary_block_evaluations=ordinary.block_evaluations - recipient.block_evaluations,
            hypothetical_block_evaluations=hypothetical.block_evaluations - recipient.block_evaluations,
        )

    @staticmethod
    def _first_route_change(ordinary, hypothetical, *, after):
        def route(layer):
            return tuple(
                (expansion.withheld, tuple(
                    (child._witness, child._complete, child._transport_use,
                     None if child._pending_destination is None else child._pending_destination.tobytes())
                    for child in expansion.children
                ))
                for expansion in layer
            )

        pairs = zip(ordinary._layers[after:], hypothetical._layers[after:])
        for depth, (left, right) in enumerate(pairs, after + 1):
            if route(left) != route(right):
                return depth
        return None

    def repair_joint(self, state, memory, *, max_hypotheses):
        """Update retained transitions on a newer pin, including previously unselected rows."""
        max_hypotheses = index(max_hypotheses)
        if type(state) is not JointWorldRead or state._reasoner is not self:
            raise ValueError("joint state belongs to a different reasoner")
        if state.memory is None or state.overflow or not state.steps or max_hypotheses < 1:
            raise ValueError("repair requires a complete attached frontier and positive capacity")
        self._require_memory(memory)
        previous = state.memory
        if memory._source_constraints is not previous._source_constraints:
            raise ValueError("changed source constraints require a fresh frontier")
        if (memory.corpus._owner is not previous.corpus._owner
                or memory.corpus.revision <= previous.corpus.revision):
            raise ValueError("repair requires a newer pin on the same journal handle")

        changed = {source for source in previous._sources.keys() | memory._sources.keys()
                   if self._bank_version(previous, source) != self._bank_version(memory, source)}
        roots = tuple(expansion.parent for expansion in state._layers[0])
        if max_hypotheses < len(roots):
            raise ValueError("repair requires capacity for the original hypotheses")
        opening = JointWorldRead(memory, roots, self, state._thresholds, unit_premises=state.unit_premises)
        transport_before = previous if memory._source_transports is not previous._source_transports else None
        return self._expand_joint(opening, state.steps, max_hypotheses, prior_layers=state._layers, changed=changed,
                                  transport_before=transport_before)

    @staticmethod
    def _bank_version(memory, source):
        bank = memory._sources.get(source)
        return None if bank is None else bank.version

    def _expand_joint(self, state, last, max_hypotheses, *, prior_layers=(), changed=(), transport_before=None):
        memory = state.memory
        all_banks = () if memory is None else tuple(memory._sources.values())
        changed_banks = () if not changed else tuple(bank for source, bank in memory._sources.items()
                                                    if source in changed)
        branches = state.branches
        layers = list(state._layers)
        uncertain = dict.fromkeys(state.uncertain_routes)
        blocked_contexts = set(state._transport_withheld_contexts)
        scanned = state.observation_scores
        feedback = state.feedback_scores
        written = state.transition_writes
        evaluated = state.block_evaluations
        ranked = state.priority_scores
        reused = state.reused_writes
        null_row = np.zeros(self._kernel._width)
        rebound_parents = {}
        retained_transports = {}

        def checkpoint(overflow):
            return replace(state, branches=branches, uncertain_routes=tuple(uncertain), steps=len(layers),
                           overflow=overflow, observation_scores=scanned, feedback_scores=feedback,
                           transition_writes=written, block_evaluations=evaluated,
                           priority_scores=ranked, reused_writes=reused, _layers=tuple(layers),
                           _transport_withheld_contexts=frozenset(blocked_contexts))

        for depth in range(state.steps, last):
            prior = ({item.parent: item for item in prior_layers[depth]} if depth < len(prior_layers) else {})
            plans = []
            total = 0
            for parent in branches:
                incumbent = prior.get(rebound_parents.get(parent, parent))
                banks = all_banks if incumbent is None else changed_banks
                plan, source_scores, priority_scores = self._plan_joint(
                    parent, memory, banks, state._thresholds, incumbent, changed,
                )
                scanned += source_scores
                ranked += priority_scores
                total += len(plan.choices)
                if total > max_hypotheses:
                    return checkpoint(True)
                plans.append(plan)

            layer = []
            rebound_parents = {}
            for plan in plans:
                if transport_before is not None:
                    plan = self._rebase_joint_plan(plan, transport_before, memory, retained_transports, rebound_parents)
                expansion, writes, evaluations, feedback_scores = self._apply_joint_plan(plan, memory, null_row)
                layer.append(expansion)
                written += writes
                reused += len(expansion.children) - writes
                evaluated += evaluations
                feedback += feedback_scores
                uncertain.update(dict.fromkeys(expansion.withheld))
                if memory is not None and memory._transport_coverage is not None:
                    for child in expansion.children:
                        pending = self._pending_discovery(memory, child._witness)
                        if pending:
                            blocked_contexts.add(child._witness)
                            uncertain.update(dict.fromkeys(pending))
            layers.append(tuple(layer))
            branches = tuple(child for expansion in layer for child in expansion.children)
        return checkpoint(False)

    def _plan_joint(self, parent, memory, banks, thresholds, incumbent, changed):
        lower, upper = thresholds
        kernel = self._kernel
        previous = {} if incumbent is None else {child._witness: child for child in incumbent.children}
        if incumbent is None:
            source = parent._source_hidden
            with np.errstate(over="raise", invalid="raise"):
                source_query = _frozen(kernel._query_observations(source, source @ kernel._query))
        else:
            source_query = incumbent.source_query

        choices = [w for w in previous if w is not None and w.span.source_id not in changed]
        withheld = ([] if incumbent is None else [w for w in incumbent.withheld if w.span.source_id not in changed])
        scores = 0
        query_relations = None
        if parent._source_relations is not None:
            matrix = kernel._query if kernel._observation_query is None else kernel._observation_query
            query_relations = relations.linear(parent._source_relations, matrix)
        for bank in banks:
            # A source is the same numerical batch on a fresh scan and a delta scan.
            keys = bank.observations._keys
            if query_relations is None:
                with np.errstate(over="raise", invalid="raise"):
                    logits = ((value, value) for value in keys @ source_query)
            else:
                with np.errstate(over="raise", invalid="raise"):
                    logits = relations.scores(query_relations, keys)
            scores += len(keys)
            for witness, (low, high), matches in zip(bank.witnesses, logits, bank.matches, strict=True):
                if low > upper and matches == 1:
                    choices.append(witness)
                elif high >= lower:
                    withheld.append(witness)

        if not choices:
            admitted = (None,)
            priority_scores = 0
        elif previous.keys() == set(choices):
            admitted = tuple(previous)
            priority_scores = 0
        else:
            choices.sort(key=lambda w: (w.span.source_id, w.ordinal))
            hidden = parent._hidden
            priority_scores = 0
            if len(choices) > 1 and hidden is not parent._source_hidden:
                with np.errstate(over="raise", invalid="raise"):
                    query = kernel._query_observations(hidden, hidden @ kernel._query)
                    priority = []
                    for witness in choices:
                        bank = memory._sources[witness.span.source_id]
                        priority.append(bank.observations._keys[bank.row_index(witness)] @ query)
                choices = [choices[row] for row in np.argsort(-np.asarray(priority), kind="stable")]
                priority_scores = len(choices)
            admitted = tuple(choices)
        withheld.sort(key=lambda w: (w.span.source_id, w.ordinal))
        return _JointPlan(parent, source_query, admitted, tuple(withheld), previous), scores, priority_scores

    @staticmethod
    def _same_transport_action(before, after):
        if before is after:
            return True
        if before is None or after is None:
            return False
        return (replace(before.use, publication_sha256=after.use.publication_sha256) == after.use
                and before.axis.tobytes() == after.axis.tobytes()
                and before.destination.tobytes() == after.destination.tobytes()
                and before.output.tobytes() == after.output.tobytes())

    def _rebase_joint_plan(self, plan, before, after, checked, rebound):
        previous = {}
        for witness in plan.choices:
            child = plan.previous.get(witness)
            if child is None:
                continue
            if witness is not None and witness not in checked:
                observation = self._witness_observation(after, witness)
                earlier = self._selected_transport(before, witness, observation)
                current = self._selected_transport(after, witness, observation)
                checked[witness] = (self._same_transport_action(earlier, current),
                                    None if current is None else current.use.publication_sha256)
            keep, publication = checked.get(witness, (True, None))
            if not keep:
                continue
            use = child._transport_use
            if use is not None and publication != use.publication_sha256:
                use = replace(use, publication_sha256=publication)
            if child._parent is not plan.parent or child._transport_use is not use:
                updated = replace(child, _parent=plan.parent, _transport_use=use)
                rebound[updated] = child
                child = updated
            previous[witness] = child
        return replace(plan, previous=previous)

    def _apply_joint_plan(self, plan, memory, null_row):
        parent = plan.parent
        feedback_value = None
        feedback_scores = 0
        if parent._key is not None and any(w not in plan.previous for w in plan.choices):
            with np.errstate(over="raise", invalid="raise"):
                query = parent._hidden @ self._kernel._query
            feedback_value, feedback_scores = self._joint_feedback(parent, query)
        children = []
        written = evaluated = 0
        for witness in plan.choices:
            child = plan.previous.get(witness)
            if child is None:
                observation = null_row if witness is None else self._witness_observation(memory, witness)
                transport = self._selected_transport(memory, witness, observation)
                policy = memory._transport_policy if memory is not None else None
                constraint_policy, constraint = self._selected_constraint(memory, witness, parent)
                relay_allowed = (witness is not None and parent._pending_destination is not None
                                 and self._eligible_relay(memory, parent, witness, policy))
                child, evaluations = self._advance_joint(
                    parent, observation, witness, feedback_value, transport, policy, constraint_policy, constraint,
                    relay_allowed=relay_allowed,
                )
                written += 1
                evaluated += evaluations
            children.append(child)
        expansion = _JointExpansion(parent, plan.source_query, tuple(children), plan.withheld)
        return expansion, written, evaluated, feedback_scores

    @staticmethod
    def _eligible_relay(memory, parent, witness, policy):
        if not transport_accepts(memory.corpus, policy, witness, relay=True):
            return False
        if type(policy) is DiscoveredTransportPolicy and policy.scoped_clauses:
            return all(law.span.source_id != witness.span.source_id for law in parent._transport_use.evidence)
        return True

    @staticmethod
    def _witness_observation(memory, witness):
        bank = memory._sources[witness.span.source_id]
        return bank.observations._observations[bank.row_index(witness)]

    @staticmethod
    def _pending_discovery(memory, witness):
        policy = memory._transport_policy
        if witness is None or not transport_accepts(memory.corpus, policy, witness, relay=False):
            return ()
        observation = RecurrentWorld._witness_observation(memory, witness)
        key = transport_coordinate_key(observation, policy)
        if memory._source_transports is not None and key in memory._source_transports.operations:
            return ()
        pending = memory._transport_coverage.pending(key)
        return () if pending is None else (witness, *pending)

    @staticmethod
    def _selected_transport(memory, witness, observation):
        if witness is None or memory._source_transports is None:
            return None
        publication = memory._source_transports
        if not transport_accepts(memory.corpus, publication.policy, witness, relay=False):
            return None
        return publication.operations.get(transport_coordinate_key(observation, publication.policy))

    @staticmethod
    def _selected_constraint(memory, witness, parent):
        if witness is None or memory._constraint_policy is None:
            return None, None
        policy = memory._constraint_policy
        source = witness.span.source_id
        if source not in policy.contexts:
            return None, None
        publication = memory._source_constraints
        if policy.unit_catalog:
            if (publication is not None and parent._catalog_actions is not None
                    and witness == publication.context_witness(source)):
                return policy, parent._catalog_actions.get(source)
            return policy, None
        return policy, None if publication is None else publication.actions.get(source)

    @staticmethod
    def _joint_feedback(branch, query):
        history = tuple(node for node in reversed(tuple(branch._ancestry())) if node._key is not None)
        with np.errstate(over="raise", invalid="raise"):
            logits = np.asarray([node._key for node in history]) @ query
            with np.errstate(over="ignore", under="ignore"):
                weights = np.exp(logits - logits.max())
            return (weights / weights.sum()) @ np.asarray([node._value for node in history]), len(history)

    def _relay_key_matches(self, observation, expected, policy):
        if type(policy) is DiscoveredTransportPolicy:
            return transport_coordinate_key(observation, policy) == transport_coordinate_key(expected, policy)
        keys = self._kernel._observation_key
        actual_key, expected_key = observation @ keys, expected @ keys
        if policy.alignment is None:
            return np.array_equal(actual_key, expected_key)
        tolerance = policy.alignment.relay_tolerance
        return (_within_relay_tolerance(observation, expected, tolerance)
                and _within_relay_tolerance(actual_key, expected_key, tolerance))

    def _advance_relational_source(self, rows, observation, observed, additions):
        kernel = self._kernel
        fixed = (relations.linear(rows, kernel._carry) if kernel._carry is not None
                 else (relations.Expression(),) * len(rows))
        if observed:
            for state_axis, observation_axis, output, threshold in kernel._conditional:
                magnitude = observation @ observation_axis
                if magnitude:
                    gate = relations.hinge(relations.dot(rows, state_axis), threshold)
                    gate = relations.scale(gate, magnitude)
                    fixed = tuple(relations.add(current, relations.scale(gate, weight))
                                  for current, weight in zip(fixed, output, strict=True))
        observation_value = relations.point(_apply_value(kernel._observation_value, observation))
        return relations.activate(tuple(relations.add(relations.add(current, extra), value)
                                        for current, extra, value in zip(fixed, additions, observation_value, strict=True)))

    @staticmethod
    def _open_constraint_additions(rows, policy):
        return tuple(relations.Expression.unknown() if policy is not None and axis in policy.output_axes
                     else relations.Expression() for axis in range(len(rows)))

    def _advance_joint(self, branch, observation, witness, feedback_value, transport, policy,
                       constraint_policy, constraint, *, relay_allowed):
        kernel = self._kernel
        current, source = branch._hidden, branch._source_hidden
        observed = witness is not None
        earlier_unit = branch._unit_context
        unit_context = earlier_unit
        mixed_scope = False
        installed_additions, constraint_use = ((None, None) if constraint is None
                                               else branch._constraint_input[constraint])
        if constraint_policy is not None and constraint_policy.unit_registry is not None:
            if earlier_unit is None:
                unit_context = witness
            elif earlier_unit.span.source_id == witness.span.source_id:
                constraint_policy, constraint, constraint_use = None, None, None
            else:
                constraint = None
                mixed_scope = True
                if constraint_use is not None:
                    constraint_use = replace(constraint_use, evidence=(), open_axes=constraint_policy.output_axes,
                                             withheld_conversion=("multiple_conversions", (earlier_unit, witness)))
        with np.errstate(over="raise", invalid="raise"):
            source_extra = None if transport is None else transport.contribution(source)
            source_rows = branch._source_relations
            if source_rows is None and constraint_policy is None:
                source_input, _ = kernel._preactivation(source, observation, None, observed,
                                                       observation_extra=source_extra)
                source_next = _frozen(np.tanh(source_input))
            else:
                if source_rows is None:
                    source_rows = relations.point(source)
                additions = (installed_additions if constraint is not None
                             else self._open_constraint_additions(source_rows, constraint_policy))
                source_rows = self._advance_relational_source(source_rows, observation, observed, additions)
                source_next = _frozen(relations.representatives(source_rows))
            hidden = source_next
            task_extra = source_extra
            evaluations = 1
            if current is not source or feedback_value is not None:
                if transport is not None and current is not source:
                    task_extra = transport.contribution(current)
                if constraint is not None:
                    task_extra = relations.representatives(installed_additions)
                task_input, _ = kernel._preactivation(current, observation, feedback_value, observed,
                                                       observation_extra=task_extra)
                hidden = _frozen(np.tanh(task_input))
                evaluations = 2
        key, value = kernel._project_generated(hidden) if self._feedback else (None, None)
        complete = branch._complete and observed
        pending = branch._pending_destination
        if complete and pending is not None:
            with np.errstate(over="raise", invalid="raise"):
                complete = relay_allowed and self._relay_key_matches(observation, pending, policy)
        use = None
        if transport is not None and (source_extra is not None or task_extra is not None):
            use = replace(transport.use, source_row=source_extra is not None, task_row=task_extra is not None)
        destination = None if source_extra is None else transport.destination
        catalog_used = not mixed_scope and (branch._catalog_used or bool(
            constraint_use and constraint_use.source_units and constraint_use.withheld_conversion is None,
        ))
        successor = JointWorldBranch(hidden, source_next, branch, witness, key, value, complete,
                                     _transport_use=use, _pending_destination=destination, _source_relations=source_rows,
                                     _constraint_use=constraint_use,
                                     _constraint_input=branch._constraint_input, _unit_context=unit_context,
                                     _unit_candidates=branch._unit_candidates, _catalog_actions=branch._catalog_actions,
                                     _catalog_used=catalog_used)
        return successor, evaluations
