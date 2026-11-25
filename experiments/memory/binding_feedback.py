"""Optional pinned binding reads inside a continuous-feedback trajectory."""

from dataclasses import InitVar, dataclass
from math import isfinite
from operator import index

import numpy as np

from experiments.memory.binding_memory import BindingCapacityError
from experiments.memory.corpus_bindings import CorpusBindingMemory, CorpusBindingRead
from experiments.memory.latent_feedback import FeedbackKernel, FeedbackState, _frozen, _real_array


_STATE_CREATION_KEY = object()


@dataclass(frozen=True)
class BindingStepRead:
    status: str
    address: tuple[str, int] | None = None
    bounds: CorpusBindingRead | None = None


@dataclass(frozen=True, eq=False)
class BindingFeedbackState:
    _receiver: object
    _feedback: FeedbackState
    _memory: CorpusBindingMemory | None
    _creation_key: InitVar[object | None] = None
    _reads: tuple[BindingStepRead, ...] = ()

    def __post_init__(self, _creation_key):
        if _creation_key is not _STATE_CREATION_KEY:
            raise TypeError("construct binding feedback states through start and step")

    @property
    def feedback(self):
        return self._feedback

    @property
    def memory(self):
        return self._memory

    @property
    def reads(self):
        return self._reads

    @property
    def latest(self):
        return self.feedback.latest


def _frame_contract(memory):
    return memory.encoder_version, memory._certain._predicates, memory._min_similarity, memory._gap


class BindingFeedback:
    """Route each hidden state to at most one corpus address before generating the next."""

    def __init__(self, kernel, frame, addresses, *, query, observation,
                 route_floor=0.0, route_gap=0.0, work_limit=65536, disjoint_reads=False):
        if type(kernel) is not FeedbackKernel or type(frame) is not CorpusBindingMemory:
            raise ValueError("binding feedback requires a dense feedback kernel and a corpus frame")
        addresses = tuple((node, index(binding)) for node, binding in addresses)
        if not addresses or any(type(node) is not str or not node or not 0 <= binding < frame._certain.width
                                for node, binding in addresses) or len(set(addresses)) != len(addresses):
            raise ValueError("binding feedback addresses must be distinct and inside the frame")
        query, observation = _real_array(query), _real_array(observation)
        if query.shape != (kernel._width, len(addresses)) or observation.shape != (6, kernel._width):
            raise ValueError("binding feedback map has incorrect shape")
        work_limit, route_floor, route_gap = index(work_limit), float(route_floor), float(route_gap)
        if work_limit < 1 or not isfinite(route_floor) or not isfinite(route_gap) or route_gap < 0:
            raise ValueError("binding feedback requires a positive work limit, finite floor and nonnegative finite gap")
        if type(disjoint_reads) is not bool:
            raise ValueError("disjoint_reads must be a boolean")
        self._kernel = kernel
        self._contract, self._anchors = _frame_contract(frame), frame._anchors
        self._addresses = addresses
        self._query, self._observation = _frozen(query), _frozen(observation)
        self._route_floor, self._route_gap, self._work_limit = route_floor, route_gap, work_limit
        self._disjoint_reads = disjoint_reads

    def start(self, seed, memory=None):
        """Pin an optional compiled corpus snapshot for this trajectory."""
        if memory is not None and (
            type(memory) is not CorpusBindingMemory or _frame_contract(memory) != self._contract
            or (memory._anchors is not self._anchors and not np.array_equal(memory._anchors, self._anchors))
        ):
            raise ValueError("memory belongs to a different binding frame")
        return BindingFeedbackState(self, self._kernel.start(seed), memory, _STATE_CREATION_KEY)

    def step(self, state):
        """Read the chosen bounds, then perform the same transition with or without memory."""
        if type(state) is not BindingFeedbackState or state._receiver is not self:
            raise ValueError("state belongs to a different binding receiver")
        self._kernel._require_state(state.feedback)
        if len(state.feedback._history) >= self._kernel._max_steps:
            raise RuntimeError("feedback step budget exhausted")
        if self._disjoint_reads and state.memory is not None and state.reads:
            prior = state.reads[-1]
            if prior.status == "resolved":
                prefix = tuple(read.address for read in state.reads)
                read = self._read(state.memory, state.latest, prefix=prefix,
                                  prefix_absent=prior.bounds.possible_strength is None)
            else:
                read = BindingStepRead("incomplete")
        else:
            read = self._read(state.memory, state.latest)
        extra = None
        if state.memory is not None:
            features = np.zeros(6)
            features[0] = 1.0
            if read.bounds is not None:
                lower, upper = read.bounds.certain, read.bounds.possible_strength
                features[1] = 1.0
                if lower is not None:
                    features[2:4] = 1.0, lower.strength
                if upper is not None:
                    features[4:6] = 1.0, upper
            with np.errstate(over="raise", invalid="raise"):
                extra = features @ self._observation
        feedback, _, _ = self._kernel._advance(state.feedback, observation_extra=extra)
        return BindingFeedbackState(self, feedback, state.memory, _STATE_CREATION_KEY, state.reads + (read,))

    def _read(self, memory, hidden, *, prefix=(), prefix_absent=False):
        if memory is None:
            return BindingStepRead("detached")
        with np.errstate(over="raise", invalid="raise"):
            scores = hidden @ self._query
        best = int(np.argmax(scores))
        winning = float(scores[best])
        if winning <= self._route_floor:
            return BindingStepRead("below_floor")
        if len(scores) > 1 and winning - float(np.partition(scores, -2)[-2]) <= self._route_gap:
            return BindingStepRead("ambiguous")
        address = self._addresses[best]
        if address[0] not in memory._certain._dependents:
            return BindingStepRead("unavailable", address)
        if prefix_absent:
            # Appending a root cannot make an infeasible conjunction feasible on this pin.
            return BindingStepRead("resolved", address, CorpusBindingRead(None, None))
        try:
            if prefix:
                bounds = memory.resolve_disjoint((*prefix, address), work_limit=self._work_limit)
            else:
                bounds = memory.resolve(*address, work_limit=self._work_limit)
        except BindingCapacityError:
            return BindingStepRead("capacity", address)
        return BindingStepRead("resolved", address, bounds)
