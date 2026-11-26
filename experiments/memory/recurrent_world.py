"""Run one latent model against optional, journal-pinned source observations."""

from dataclasses import dataclass, replace
from operator import index
from types import MappingProxyType

import numpy as np

from experiments.memory.corpus_bindings import CorpusRevision
from experiments.memory.latent_feedback import FeedbackKernel, ObservationMemory, _frozen
from experiments.memory.verified_source_tool import SourceSpan


@dataclass(frozen=True, eq=False)
class _SourceBank:
    version: tuple[str, int]
    observations: ObservationMemory
    spans: tuple[SourceSpan, ...]


@dataclass(frozen=True, eq=False)
class RecurrentWorldMemory:
    """A source snapshot with reusable observation keys and exact source pointers."""

    corpus: CorpusRevision
    _bank: ObservationMemory
    _sources: MappingProxyType
    projected_rows: int
    packed_rows: int

    @property
    def spans(self):
        return tuple(span for bank in self._sources.values() for span in bank.spans)


@dataclass(frozen=True, eq=False)
class RecurrentWorldRead:
    memory: RecurrentWorldMemory | None
    history: tuple[np.ndarray, ...]
    observation_scores: int
    feedback_scores: int

    @property
    def hidden(self):
        return self.history[-1].view()


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
                bank = _SourceBank(version, self._kernel.prepare(rows), spans)
                projected += len(mentions)
            sources[source] = bank
        packed_rows = 0
        if previous is not None and sources == prior:
            packed = previous._bank
        elif sources:
            banks = tuple(source.observations for source in sources.values())
            packed = ObservationMemory(
                self._kernel._observation_key,
                _frozen(np.concatenate([bank._keys for bank in banks])),
                _frozen(np.concatenate([bank._observations for bank in banks])),
            )
            packed_rows = len(packed._keys)
        else:
            packed = self._kernel._empty
        return RecurrentWorldMemory(corpus, packed, MappingProxyType(sources), projected, packed_rows)

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
