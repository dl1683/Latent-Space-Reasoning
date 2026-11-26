"""Run one latent model against optional, journal-pinned source observations."""

from dataclasses import dataclass, replace
from math import isfinite
from operator import index
from types import MappingProxyType

import numpy as np

from experiments.memory.corpus_bindings import CorpusRevision
from experiments.memory.latent_feedback import FeedbackKernel, ObservationMemory, _frozen, _real_array
from experiments.memory.verified_source_tool import SourceSpan


@dataclass(frozen=True)
class WorldWitness:
    span: SourceSpan
    generation: int
    ordinal: int


@dataclass(frozen=True, eq=False)
class _SourceBank:
    version: tuple[str, int]
    observations: ObservationMemory
    witnesses: tuple[WorldWitness, ...]
    matches: tuple[int, ...]


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

    @property
    def spans(self):
        return tuple(witness.span for witness in self._witnesses)


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
        return self._source_hidden.view()


@dataclass(frozen=True)
class JointWorldOutcome:
    status: str
    support: tuple[JointWorldBranch, ...] = ()
    opposition: tuple[JointWorldBranch, ...] = ()


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
    _layers: tuple[tuple[_JointExpansion, ...], ...] = ()

    def read(self, direction, *, floor):
        """Report signed branch observations within the configured numerical scope."""
        direction, floor = _real_array(direction), float(floor)
        if direction.shape != self.branches[0].hidden.shape or not isfinite(floor) or floor <= 0:
            raise ValueError("readout requires a matching direction and positive finite floor")
        if self.overflow:
            return JointWorldOutcome("hypothesis_limit")
        if self.memory is None:
            return JointWorldOutcome("memory_omitted")

        complete = tuple(branch for branch in self.branches if branch._complete)
        with np.errstate(over="raise", invalid="raise"):
            scores = np.array([branch._source_hidden @ direction for branch in complete])
        support = tuple(branch for branch, score in zip(complete, scores, strict=True) if score >= floor)
        opposition = tuple(branch for branch, score in zip(complete, scores, strict=True) if score <= -floor)
        if self.uncertain_routes or self.memory.unlocated_sources:
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
        return JointWorldOutcome(status, support, opposition)


class RecurrentWorld:
    """Reuse a fixed latent block or feed back a growing sequence of latent thoughts."""

    def __init__(self, kernel, *, encoder_version, max_observations, feedback):
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

    def prepare(self, corpus, *, previous=None, packed=True):
        """Project changed source generations and optionally build a dense bank."""
        if type(packed) is not bool:
            raise ValueError("packed must be a boolean")
        if type(corpus) is not CorpusRevision or corpus.journal_sha256 is None:
            raise ValueError("the observation world requires a journaled source revision")
        memory = corpus.memory
        if memory.encoder_version != self._encoder_version or memory._anchors.shape[1] != self._kernel._width:
            raise ValueError("corpus observations use a different encoder frame")
        if previous is not None:
            self._require_memory(previous)
            if corpus._owner is not previous.corpus._owner:
                raise ValueError("previous observations belong to a different journal handle")
        if memory.source_digests.keys() != memory._mentions.keys():
            raise ValueError("world observations are unavailable for legacy row-only sources")
        if sum(map(len, memory._mentions.values())) > self._max_observations:
            raise ValueError("world observation allowance exceeded")

        sources = {}
        projected = 0
        prior = {} if previous is None else previous._sources
        for source, digest in sorted(memory.source_digests.items()):
            version = digest, corpus._source_revisions[source]
            bank = prior.get(source)
            if bank is None or bank.version != version:
                mentions = memory.source_mentions(source)
                spans = tuple(memory.source_mention_span(source, n) for n in range(len(mentions)))
                if any(span is None for span in spans):
                    raise ValueError("world observations require exact mention spans")
                rows = np.asarray([mention.vector for mention in mentions]).reshape(-1, self._kernel._width)
                witnesses = tuple(WorldWitness(span, version[1], n) for n, span in enumerate(spans))
                matches = tuple(row[3] for row in memory._mentions[source])
                bank = _SourceBank(version, self._kernel.prepare(rows), witnesses, matches)
                projected += len(mentions)
            sources[source] = bank
        if previous is not None and sources == prior and (not packed or previous._bank is not None):
            return replace(previous, corpus=corpus, _bank=previous._bank if packed else None,
                           projected_rows=0, packed_rows=0)
        dense = self._pack_sources(sources) if packed else None
        witnesses = tuple(witness for bank in sources.values() for witness in bank.witnesses)
        unlocated = tuple(source for source, bank in sources.items() if not bank.matches or 0 in bank.matches)
        packed_rows = len(witnesses) if packed else 0
        return RecurrentWorldMemory(corpus, dense, self._kernel._observation_key, MappingProxyType(sources),
                                    projected, packed_rows, witnesses, unlocated)

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

    def reason_joint(self, seed, *, steps, max_hypotheses, null_logit, margin, memory=None):
        """Advance locally distinct hypotheses instead of pooling their observations."""
        null_logit, margin = float(null_logit), float(margin)
        lower, upper = null_logit - margin, null_logit + margin
        if not isfinite(lower) or not isfinite(upper) or not lower < upper:
            raise ValueError("joint reasoning requires a finite null score and positive margin")
        if memory is not None:
            self._require_memory(memory)
        initial = self._kernel._vector(seed)
        opening = JointWorldBranch(initial, initial)
        state = JointWorldRead(memory, (opening,), self, (lower, upper))
        return self.continue_joint(state, steps=steps, max_hypotheses=max_hypotheses)

    def continue_joint(self, state, *, steps, max_hypotheses):
        """Resume at a committed frontier on the same source and operator pin."""
        steps, max_hypotheses = index(steps), index(max_hypotheses)
        if type(state) is not JointWorldRead or state._reasoner is not self:
            raise ValueError("joint state belongs to a different reasoner")
        last = state.steps + steps
        if steps < 1 or last > self._kernel._max_steps or max_hypotheses < len(state.branches):
            raise ValueError("joint continuation exceeds the horizon or cannot hold the current frontier")
        return self._expand_joint(state, last, max_hypotheses)

    def repair_joint(self, state, memory, *, max_hypotheses):
        """Update retained transitions on a newer pin, including previously unselected rows."""
        max_hypotheses = index(max_hypotheses)
        if type(state) is not JointWorldRead or state._reasoner is not self:
            raise ValueError("joint state belongs to a different reasoner")
        if state.memory is None or state.overflow or not state.steps or max_hypotheses < 1:
            raise ValueError("repair requires a complete attached frontier and positive capacity")
        self._require_memory(memory)
        previous = state.memory
        if (memory.corpus._owner is not previous.corpus._owner
                or memory.corpus.revision <= previous.corpus.revision):
            raise ValueError("repair requires a newer pin on the same journal handle")

        changed = {source for source in previous._sources.keys() | memory._sources.keys()
                   if self._bank_version(previous, source) != self._bank_version(memory, source)}
        opening = JointWorldRead(memory, (state._layers[0][0].parent,), self, state._thresholds)
        return self._expand_joint(opening, state.steps, max_hypotheses, prior_layers=state._layers, changed=changed)

    @staticmethod
    def _bank_version(memory, source):
        bank = memory._sources.get(source)
        return None if bank is None else bank.version

    def _expand_joint(self, state, last, max_hypotheses, *, prior_layers=(), changed=()):
        memory = state.memory
        all_banks = () if memory is None else tuple(memory._sources.values())
        changed_banks = () if not changed else tuple(bank for source, bank in memory._sources.items()
                                                    if source in changed)
        branches = state.branches
        layers = list(state._layers)
        uncertain = dict.fromkeys(state.uncertain_routes)
        scanned = state.observation_scores
        feedback = state.feedback_scores
        written = state.transition_writes
        evaluated = state.block_evaluations
        ranked = state.priority_scores
        reused = state.reused_writes
        null_row = np.zeros(self._kernel._width)

        def checkpoint(overflow):
            return replace(state, branches=branches, uncertain_routes=tuple(uncertain), steps=len(layers),
                           overflow=overflow, observation_scores=scanned, feedback_scores=feedback,
                           transition_writes=written, block_evaluations=evaluated,
                           priority_scores=ranked, reused_writes=reused, _layers=tuple(layers))

        for depth in range(state.steps, last):
            prior = ({item.parent: item for item in prior_layers[depth]} if depth < len(prior_layers) else {})
            plans = []
            total = 0
            for parent in branches:
                incumbent = prior.get(parent)
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
            for plan in plans:
                expansion, writes, evaluations, feedback_scores = self._apply_joint_plan(plan, memory, null_row)
                layer.append(expansion)
                written += writes
                reused += len(expansion.children) - writes
                evaluated += evaluations
                feedback += feedback_scores
                uncertain.update(dict.fromkeys(expansion.withheld))
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
        for bank in banks:
            # A source is the same numerical batch on a fresh scan and a delta scan.
            with np.errstate(over="raise", invalid="raise"):
                logits = bank.observations._keys @ source_query
            scores += len(logits)
            for witness, logit, matches in zip(bank.witnesses, logits, bank.matches, strict=True):
                if logit > upper and matches == 1:
                    choices.append(witness)
                elif logit >= lower:
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
                    priority = [memory._sources[w.span.source_id].observations._keys[w.ordinal] @ query
                                for w in choices]
                choices = [choices[row] for row in np.argsort(-np.asarray(priority), kind="stable")]
                priority_scores = len(choices)
            admitted = tuple(choices)
        withheld.sort(key=lambda w: (w.span.source_id, w.ordinal))
        return _JointPlan(parent, source_query, admitted, tuple(withheld), previous), scores, priority_scores

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
                observation = (null_row if witness is None else
                               memory._sources[witness.span.source_id].observations._observations[witness.ordinal])
                child, evaluations = self._advance_joint(parent, observation, witness, feedback_value)
                written += 1
                evaluated += evaluations
            children.append(child)
        expansion = _JointExpansion(parent, plan.source_query, tuple(children), plan.withheld)
        return expansion, written, evaluated, feedback_scores

    @staticmethod
    def _joint_feedback(branch, query):
        history = tuple(node for node in reversed(tuple(branch._ancestry())) if node._key is not None)
        with np.errstate(over="raise", invalid="raise"):
            logits = np.asarray([node._key for node in history]) @ query
            with np.errstate(over="ignore", under="ignore"):
                weights = np.exp(logits - logits.max())
            return (weights / weights.sum()) @ np.asarray([node._value for node in history]), len(history)

    def _advance_joint(self, branch, observation, witness, feedback_value):
        kernel = self._kernel
        current, source = branch._hidden, branch._source_hidden
        observed = witness is not None
        with np.errstate(over="raise", invalid="raise"):
            source_input, _ = kernel._preactivation(source, observation, None, observed)
            source_next = _frozen(np.tanh(source_input))
            hidden = source_next
            evaluations = 1
            if current is not source or feedback_value is not None:
                task_input, _ = kernel._preactivation(current, observation, feedback_value, observed)
                hidden = _frozen(np.tanh(task_input))
                evaluations = 2
        key, value = kernel._project_generated(hidden) if self._feedback else (None, None)
        successor = JointWorldBranch(hidden, source_next, branch, witness, key, value, branch._complete and observed)
        return successor, evaluations
