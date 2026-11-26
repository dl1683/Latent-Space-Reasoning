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
    _bank: ObservationMemory
    _sources: MappingProxyType
    projected_rows: int
    packed_rows: int
    _witnesses: tuple[WorldWitness, ...]
    _unique: np.ndarray
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

    @property
    def hidden(self):
        return self.history[-1].view()


@dataclass(frozen=True, eq=False)
class JointWorldBranch:
    """One task-private latent trajectory and its ordered source uses."""

    history: tuple[np.ndarray, ...]
    _source_hidden: np.ndarray
    witnesses: tuple[WorldWitness | None, ...] = ()
    _keys: tuple[np.ndarray, ...] = ()
    _values: tuple[np.ndarray, ...] = ()

    @property
    def hidden(self):
        return self.history[-1].view()

    @property
    def source_hidden(self):
        return self._source_hidden.view()


@dataclass(frozen=True)
class JointWorldOutcome:
    status: str
    support: tuple[JointWorldBranch, ...] = ()
    opposition: tuple[JointWorldBranch, ...] = ()


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

    def read(self, direction, *, floor):
        """Report signed branch observations within the configured numerical scope."""
        direction, floor = _real_array(direction), float(floor)
        if direction.shape != self.branches[0].hidden.shape or not isfinite(floor) or floor <= 0:
            raise ValueError("readout requires a matching direction and positive finite floor")
        if self.overflow:
            return JointWorldOutcome("hypothesis_limit")
        if self.memory is None:
            return JointWorldOutcome("memory_omitted")

        complete = tuple(branch for branch in self.branches if None not in branch.witnesses)
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

    def prepare(self, corpus, *, previous=None):
        """Project only changed source generations before any question arrives."""
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
        packed_rows = 0
        if previous is not None and sources == prior:
            return replace(previous, corpus=corpus, projected_rows=0, packed_rows=0)
        if sources:
            banks = tuple(source.observations for source in sources.values())
            packed = ObservationMemory(
                self._kernel._observation_key,
                _frozen(np.concatenate([bank._keys for bank in banks])),
                _frozen(np.concatenate([bank._observations for bank in banks])),
            )
            packed_rows = len(packed._keys)
        else:
            packed = self._kernel._empty
        witnesses = tuple(witness for bank in sources.values() for witness in bank.witnesses)
        unique = _frozen(np.array([count == 1 for bank in sources.values() for count in bank.matches], dtype=bool))
        unlocated = tuple(source for source, bank in sources.items() if not bank.matches or 0 in bank.matches)
        return RecurrentWorldMemory(corpus, packed, MappingProxyType(sources), projected, packed_rows,
                                    witnesses, unique, unlocated)

    def _require_memory(self, memory):
        if (type(memory) is not RecurrentWorldMemory
                or memory._bank._key_map is not self._kernel._observation_key
                or memory.corpus.memory.encoder_version != self._encoder_version
                or len(memory._bank._keys) > self._max_observations):
            raise ValueError("memory belongs to a different observation frame or exceeds the allowance")

    def reason(self, seed, *, steps, memory=None, replace_first=None):
        """Keep input and readout unchanged when resident world memory is attached."""
        steps = index(steps)
        if not 1 <= steps <= self._kernel._max_steps:
            raise ValueError("steps must fit the latent block horizon")
        if memory is not None:
            self._require_memory(memory)
        if replace_first is not None:
            replace_first = self._kernel._vector(replace_first)
        state = self._kernel.start(seed, None if memory is None else memory._bank)
        if self._feedback:
            for step in range(steps):
                state = self._kernel.step(state)
                if step == 0 and replace_first is not None:
                    state = self._kernel.intervene(state, 0, replace_first)
            return RecurrentWorldRead(memory, state.history, state.observation_scores, state.feedback_scores)

        history = []
        for step in range(steps):
            preactivation, _, _ = self._kernel._step_preactivation(state)
            hidden = _frozen(np.tanh(preactivation))
            if step == 0 and replace_first is not None:
                hidden = replace_first
            history.append(hidden)
            state = replace(state, _seed=hidden)
        return RecurrentWorldRead(memory, tuple(history), steps * len(state._memory._keys), 0)


    def reason_joint(self, seed, *, steps, max_hypotheses, null_logit, margin, memory=None):
        """Advance locally distinct hypotheses instead of pooling their observations."""
        null_logit, margin = float(null_logit), float(margin)
        lower, upper = null_logit - margin, null_logit + margin
        if not isfinite(lower) or not isfinite(upper) or not lower < upper:
            raise ValueError("joint reasoning requires a finite null score and positive margin")
        if memory is not None:
            self._require_memory(memory)
        initial = self._kernel._vector(seed)
        opening = JointWorldBranch((initial,), initial)
        state = JointWorldRead(memory, (opening,), self, (lower, upper))
        return self.continue_joint(state, steps=steps, max_hypotheses=max_hypotheses)

    def continue_joint(self, state, *, steps, max_hypotheses):
        """Resume at a committed frontier on the same source and operator pin."""
        steps, max_hypotheses = index(steps), index(max_hypotheses)
        if type(state) is not JointWorldRead or state._reasoner is not self:
            raise ValueError("joint state belongs to a different reasoner")
        kernel = self._kernel
        last = state.steps + steps
        if steps < 1 or last > kernel._max_steps or max_hypotheses < len(state.branches):
            raise ValueError("joint continuation exceeds the horizon or cannot hold the current frontier")
        memory = state.memory
        lower, upper = state._thresholds
        bank = kernel._empty if memory is None else memory._bank
        branches = state.branches
        uncertain = dict.fromkeys(state.uncertain_routes)
        observation_scores = state.observation_scores
        feedback_scores = state.feedback_scores
        transition_writes = state.transition_writes
        block_evaluations = state.block_evaluations
        priority_scores = state.priority_scores
        null_row = np.zeros(kernel._width)
        for depth in range(state.steps, last):
            planned = []
            deferred = {}
            total = 0
            for branch in branches:
                source_hidden = branch._source_hidden
                hidden = branch.history[-1]
                with np.errstate(over="raise", invalid="raise"):
                    source_query = source_hidden @ kernel._query
                    logits = bank._keys @ kernel._query_observations(source_hidden, source_query)
                    query = source_query if hidden is source_hidden else hidden @ kernel._query
                observation_scores += len(logits)
                choices = np.flatnonzero((logits > upper) & memory._unique) if len(logits) else ()
                if len(logits):
                    withheld = (logits >= lower) & ((logits <= upper) | ~memory._unique)
                    deferred.update(dict.fromkeys(memory._witnesses[row] for row in np.flatnonzero(withheld)))
                total += max(1, len(choices))
                if total > max_hypotheses:
                    return replace(state, branches=branches, uncertain_routes=tuple(uncertain), steps=depth,
                                   overflow=True, observation_scores=observation_scores,
                                   feedback_scores=feedback_scores, transition_writes=transition_writes,
                                   block_evaluations=block_evaluations, priority_scores=priority_scores)
                if len(choices) > 1 and hidden is not source_hidden:
                    with np.errstate(over="raise", invalid="raise"):
                        priority = bank._keys[choices] @ kernel._query_observations(hidden, query)
                    choices = choices[np.argsort(-priority, kind="stable")]
                    priority_scores += len(choices)
                planned.append((branch, query, choices))

            advanced = []
            for branch, query, choices in planned:
                # Generated history and corpora have independent normalization.
                feedback_scores += len(branch._keys)
                feedback_value = self._joint_feedback(branch, query)
                for row in choices if len(choices) else (None,):
                    witness = None if row is None else memory._witnesses[row]
                    observation = null_row if row is None else bank._observations[row]
                    successor, evaluations = self._advance_joint(branch, observation, witness, feedback_value)
                    advanced.append(successor)
                    block_evaluations += evaluations
            branches = tuple(advanced)
            transition_writes += len(branches)
            uncertain.update(deferred)
        return replace(state, branches=branches, uncertain_routes=tuple(uncertain), steps=last,
                       overflow=False, observation_scores=observation_scores, feedback_scores=feedback_scores,
                       transition_writes=transition_writes, block_evaluations=block_evaluations,
                       priority_scores=priority_scores)

    @staticmethod
    def _joint_feedback(branch, query):
        if not branch._keys:
            return None
        with np.errstate(over="raise", invalid="raise"):
            logits = np.asarray(branch._keys) @ query
            with np.errstate(over="ignore", under="ignore"):
                weights = np.exp(logits - logits.max())
            return (weights / weights.sum()) @ np.asarray(branch._values)

    def _advance_joint(self, branch, observation, witness, feedback_value):
        kernel = self._kernel
        current, source = branch.history[-1], branch._source_hidden
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
        keys, values = branch._keys, branch._values
        if self._feedback:
            key, value = kernel._project_generated(hidden)
            keys, values = keys + (key,), values + (value,)
        successor = JointWorldBranch(branch.history + (hidden,), source_next, branch.witnesses + (witness,), keys, values)
        return successor, evaluations
