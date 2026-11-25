"""Continuous feedback over immutable, separately projected observation memory.

The caller supplies numerical observations and weights. Each computed hidden
state is available to later steps as both a query input and an attention row.
"""

from dataclasses import dataclass, replace
from operator import index

import numpy as np


def _frozen(array):
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _real_array(value):
    if np.iscomplexobj(value):
        raise ValueError("expected real coordinates")
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("coordinates and weights must be finite")
    return array


@dataclass(frozen=True, eq=False)
class ObservationMemory:
    _kernel: object
    _keys: np.ndarray
    _values: np.ndarray


@dataclass(frozen=True, eq=False)
class FeedbackState:
    _kernel: object
    _memory: ObservationMemory
    seed: np.ndarray
    history: tuple[np.ndarray, ...] = ()
    _keys: tuple[np.ndarray, ...] = ()
    _values: tuple[np.ndarray, ...] = ()
    observation_scores: int = 0
    feedback_scores: int = 0

    @property
    def latest(self):
        return self.history[-1] if self.history else self.seed


class FeedbackKernel:
    """A bounded shared transition with separate observation and feedback maps."""

    def __init__(self, *, query, carry, observation_key, observation_value,
                 feedback_key, feedback_value, max_steps):
        query = _real_array(query)
        if query.ndim != 2 or not len(query) or query.shape[0] != query.shape[1]:
            raise ValueError("query map must be a nonempty square matrix")
        self._width = len(query)
        self._max_steps = index(max_steps)
        if self._max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._query = _frozen(query)
        self._carry = self._weight(carry)
        self._observation_key = self._weight(observation_key)
        self._observation_value = self._weight(observation_value)
        self._feedback_key = self._weight(feedback_key)
        self._feedback_value = self._weight(feedback_value)
        self._empty = self.prepare(np.empty((0, self._width)))

    def _weight(self, value):
        matrix = _real_array(value)
        if matrix.shape != (self._width, self._width):
            raise ValueError("weight map has incorrect shape")
        return _frozen(matrix)

    def _vector(self, value):
        vector = _real_array(value)
        if vector.shape != (self._width,):
            raise ValueError("hidden vector has incorrect width")
        return _frozen(vector)

    def prepare(self, observations):
        """Project an observation snapshot once for reuse across queries."""
        rows = _real_array(observations)
        if rows.ndim != 2 or rows.shape[1] != self._width:
            raise ValueError("observation matrix has incorrect width")
        with np.errstate(over="raise", invalid="raise"):
            keys = rows @ self._observation_key
            values = rows @ self._observation_value
        return ObservationMemory(self, _frozen(keys), _frozen(values))

    def start(self, seed, memory=None):
        """Start a trajectory pinned to one observation snapshot, if supplied."""
        memory = self._empty if memory is None else memory
        if memory._kernel is not self:
            raise ValueError("memory belongs to a different feedback kernel")
        return FeedbackState(self, memory, self._vector(seed))

    def _require_state(self, state):
        if state._kernel is not self:
            raise ValueError("state belongs to a different feedback kernel")

    def _project_generated(self, vector):
        with np.errstate(over="raise", invalid="raise"):
            return _frozen(vector @ self._feedback_key), _frozen(vector @ self._feedback_value)

    def step(self, state):
        """Append one generated vector and tally the attention scores evaluated."""
        self._require_state(state)
        if len(state.history) >= self._max_steps:
            raise RuntimeError("feedback step budget exhausted")
        memory = state._memory
        observation_count = len(memory._keys)
        history_count = len(state.history)
        with np.errstate(over="raise", invalid="raise"):
            hidden = state.latest
            query = hidden @ self._query
            update = hidden @ self._carry
            if observation_count or history_count:
                history_keys = np.asarray(state._keys).reshape(-1, self._width)
                history_values = np.asarray(state._values).reshape(-1, self._width)
                logits = np.concatenate((memory._keys @ query, history_keys @ query))
                # Extreme negative differences may safely round down to zero mass.
                with np.errstate(over="ignore", under="ignore"):
                    weights = np.exp(logits - logits.max())
                weights /= weights.sum()
                update += weights[:observation_count] @ memory._values
                update += weights[observation_count:] @ history_values
            next_hidden = _frozen(np.tanh(update))
        next_key, next_value = self._project_generated(next_hidden)
        return replace(
            state,
            history=state.history + (next_hidden,),
            _keys=state._keys + (next_key,),
            _values=state._values + (next_value,),
            observation_scores=state.observation_scores + observation_count,
            feedback_scores=state.feedback_scores + history_count,
        )

    def intervene(self, state, position, hidden):
        """Replace a generated state in a copy for a causal control.

        Later generated rows are held fixed. This differs from replay after an
        observation update, which starts a new trajectory with the new memory.
        """
        self._require_state(state)
        position = index(position)
        if not 0 <= position < len(state.history):
            raise ValueError("generated-state position out of range")
        vector = self._vector(hidden)
        key, value = self._project_generated(vector)
        suffix = position + 1
        return replace(
            state,
            history=state.history[:position] + (vector,) + state.history[suffix:],
            _keys=state._keys[:position] + (key,) + state._keys[suffix:],
            _values=state._values[:position] + (value,) + state._values[suffix:],
        )
