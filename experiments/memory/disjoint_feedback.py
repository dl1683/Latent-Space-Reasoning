"""Feed a pinned source-disjoint transition back into the normal hidden-state update."""

from dataclasses import InitVar, dataclass
from math import isfinite

import numpy as np

from experiments.memory.disjoint_transition import DisjointTransitionMemory, propagate_disjoint
from experiments.memory.latent_feedback import FactoredValue, FeedbackKernel, FeedbackState, _apply_value, _frozen, _real_array


_CREATION_KEY = object()


@dataclass(frozen=True, eq=False)
class _RelationObservation:
    _receiver: "DisjointFeedback"
    _transitions: DisjointTransitionMemory
    _anchors: np.ndarray
    _creation_key: InitVar[object | None] = None

    def __post_init__(self, _creation_key):
        if _creation_key is not _CREATION_KEY:
            raise TypeError("construct relation memory through prepare")


@dataclass(frozen=True, eq=False)
class DisjointFeedbackState:
    _receiver: "DisjointFeedback"
    _feedback: FeedbackState
    _memory: _RelationObservation | None
    _grades: np.ndarray | None = None
    _creation_key: InitVar[object | None] = None

    def __post_init__(self, _creation_key):
        if _creation_key is not _CREATION_KEY:
            raise TypeError("construct disjoint feedback states through start and step")

    @property
    def feedback(self):
        return self._feedback

    @property
    def latest(self):
        return self._feedback.latest

    @property
    def grades(self):
        """Destination grades from the last operator call, with -inf for absence."""
        return None if self._grades is None else self._grades.view()


@dataclass(frozen=True, eq=False, init=False)
class DisjointFeedback:
    """A shared receiver with optional, numerically distinct relation and observation banks."""

    _kernel: FeedbackKernel
    _query: np.ndarray
    _presence_value: np.ndarray | FactoredValue
    _grade_value: np.ndarray | FactoredValue
    _floor: float

    def __init__(self, kernel, *, query, presence_value, grade_value, floor):
        if type(kernel) is not FeedbackKernel:
            raise ValueError("disjoint feedback needs a dense feedback kernel")
        floor = float(floor)
        if not isfinite(floor) or floor <= 0.:
            raise ValueError("relation query floor must be finite and positive")
        object.__setattr__(self, "_kernel", kernel)
        object.__setattr__(self, "_query", kernel._weight(query))
        object.__setattr__(self, "_presence_value", kernel._value(presence_value))
        object.__setattr__(self, "_grade_value", kernel._value(grade_value))
        object.__setattr__(self, "_floor", floor)

    def prepare(self, transitions, anchors):
        """Attach one supplied anchor row per transition coordinate to an immutable pin."""
        if type(transitions) is not DisjointTransitionMemory:
            raise ValueError("expected a disjoint transition snapshot")
        anchors = _real_array(anchors)
        if anchors.shape != (transitions.width, self._kernel._width):
            raise ValueError("relation anchors need one hidden row per transition coordinate")
        return _RelationObservation(self, transitions, _frozen(anchors), _creation_key=_CREATION_KEY)

    def start(self, seed, memory=None, *, observations=None):
        """Pin the relation snapshot and the ordinary observation memory for this trajectory."""
        if memory is not None and (type(memory) is not _RelationObservation or memory._receiver is not self):
            raise ValueError("relation memory belongs to a different receiver")
        return DisjointFeedbackState(self, self._kernel.start(seed, observations), memory,
                                     _creation_key=_CREATION_KEY)

    def step(self, state, *, use_operator=True, relation_query=None):
        """Advance normally, with an optional intervention on just the relation read."""
        if type(state) is not DisjointFeedbackState or state._receiver is not self:
            raise ValueError("state belongs to a different disjoint feedback receiver")
        if len(state._feedback._history) >= self._kernel._max_steps:
            raise RuntimeError("feedback step budget exhausted")
        grades = extra = None
        if use_operator and state._memory is not None:
            memory = state._memory
            anchors = memory._anchors
            hidden = state.latest if relation_query is None else self._kernel._vector(relation_query)
            with np.errstate(over="raise", invalid="raise"):
                scores = anchors @ (hidden @ self._query)
                seed = np.full(memory._transitions.width, -np.inf)
                eligible = scores >= self._floor
                seed[eligible] = np.minimum(scores[eligible] - self._floor, 1.)
                grades = propagate_disjoint(seed, memory=memory._transitions)
                supported = np.isfinite(grades)
                if supported.any():
                    present = supported @ anchors
                    strengths = np.where(supported, grades, 0.) @ anchors
                    extra = _apply_value(self._presence_value, present) + _apply_value(self._grade_value, strengths)
        elif relation_query is not None:
            raise ValueError("a relation query intervention needs the attached operator")
        feedback, _, _ = self._kernel._advance(state._feedback, observation_extra=extra)
        return DisjointFeedbackState(self, feedback, state._memory, grades, _creation_key=_CREATION_KEY)
