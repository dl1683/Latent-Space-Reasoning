"""Continuous feedback over immutable, separately projected observation memory.

The caller supplies numerical observations and weights. Each computed hidden
state is available to later steps as both a query input and an attention row.
"""

from copy import copy
from dataclasses import dataclass, replace
from operator import index

import numpy as np

from experiments.memory.weight_program import program_weight


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
    _key_map: np.ndarray
    _keys: np.ndarray
    _observations: np.ndarray


@dataclass(frozen=True, eq=False)
class FeedbackState:
    _kernel: object
    _memory: ObservationMemory
    _seed: np.ndarray
    _history: tuple[np.ndarray, ...] = ()
    _keys: tuple[np.ndarray, ...] = ()
    _values: tuple[np.ndarray, ...] = ()
    observation_scores: int = 0
    feedback_scores: int = 0

    @property
    def seed(self):
        return self._seed.view()

    @property
    def history(self):
        return tuple(row.view() for row in self._history)

    @property
    def latest(self):
        latest = self._history[-1] if self._history else self._seed
        return latest.view()


@dataclass(frozen=True, eq=False, repr=False, init=False)
class FeedbackKernel:
    """A bounded shared transition with separate observation and feedback maps."""

    _width: int
    _max_steps: int
    _query: np.ndarray
    _carry: np.ndarray
    _observation_key: np.ndarray
    _observation_value: np.ndarray
    _feedback_key: np.ndarray
    _feedback_value: np.ndarray
    _empty: ObservationMemory

    def __init__(self, *, query, carry, observation_key, observation_value,
                 feedback_key, feedback_value, max_steps):
        query = _real_array(query)
        if query.ndim != 2 or not len(query) or query.shape[0] != query.shape[1]:
            raise ValueError("query map must be a nonempty square matrix")
        object.__setattr__(self, "_width", len(query))
        object.__setattr__(self, "_max_steps", index(max_steps))
        if self._max_steps <= 0:
            raise ValueError("max_steps must be positive")
        object.__setattr__(self, "_query", _frozen(query))
        object.__setattr__(self, "_carry", self._weight(carry))
        object.__setattr__(self, "_observation_key", self._weight(observation_key))
        object.__setattr__(self, "_observation_value", self._weight(observation_value))
        object.__setattr__(self, "_feedback_key", self._weight(feedback_key))
        object.__setattr__(self, "_feedback_value", self._weight(feedback_value))
        object.__setattr__(self, "_empty", self.prepare(np.empty((0, self._width))))

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
        """Retain observation coordinates and project their reusable keys."""
        rows = _real_array(observations)
        if rows.ndim != 2 or rows.shape[1] != self._width:
            raise ValueError("observation matrix has incorrect width")
        with np.errstate(over="raise", invalid="raise"):
            keys = rows @ self._observation_key
        return ObservationMemory(self._observation_key, _frozen(keys), _frozen(rows))

    def start(self, seed, memory=None):
        """Start a trajectory pinned to one observation snapshot, if supplied."""
        memory = self._empty if memory is None else memory
        if memory._key_map is not self._observation_key:
            raise ValueError("memory belongs to a different feedback kernel key map")
        return FeedbackState(self, memory, self._vector(seed))

    def _require_state(self, state):
        if state._kernel is not self:
            raise ValueError("state belongs to a different feedback kernel")

    def _project_generated(self, vector):
        with np.errstate(over="raise", invalid="raise"):
            return _frozen(vector @ self._feedback_key), _frozen(vector @ self._feedback_value)

    def step(self, state):
        """Append one generated vector and tally the attention scores evaluated."""
        return self._advance(state)[0]

    def _advance(self, state):
        self._require_state(state)
        if len(state._history) >= self._max_steps:
            raise RuntimeError("feedback step budget exhausted")
        memory = state._memory
        observation_count = len(memory._keys)
        history_count = len(state._history)
        with np.errstate(over="raise", invalid="raise"):
            hidden = state.latest
            query = hidden @ self._query
            fixed = hidden @ self._carry
            mixture = np.zeros(self._width)
            if observation_count or history_count:
                history_keys = np.asarray(state._keys).reshape(-1, self._width)
                history_values = np.asarray(state._values).reshape(-1, self._width)
                logits = np.concatenate((memory._keys @ query, history_keys @ query))
                # Extreme negative differences may safely round down to zero mass.
                with np.errstate(over="ignore", under="ignore"):
                    weights = np.exp(logits - logits.max())
                weights /= weights.sum()
                mixture = weights[:observation_count] @ memory._observations
                fixed += weights[observation_count:] @ history_values
            next_hidden = _frozen(np.tanh(fixed + mixture @ self._observation_value))
        next_key, next_value = self._project_generated(next_hidden)
        next_state = replace(
            state,
            _history=state._history + (next_hidden,),
            _keys=state._keys + (next_key,),
            _values=state._values + (next_value,),
            observation_scores=state.observation_scores + observation_count,
            feedback_scores=state.feedback_scores + history_count,
        )
        return next_state, mixture, fixed

    def _with_observation_value(self, weight):
        candidate = copy(self)
        object.__setattr__(candidate, "_observation_value", self._weight(weight))
        return candidate

    def _trace(self, seed, memory, steps):
        steps = index(steps)
        if not 1 <= steps <= self._max_steps:
            raise ValueError("target step is outside the feedback budget")
        traced = self.start(seed, memory)
        mixtures = []
        for _ in range(steps):
            traced, mixture, fixed = self._advance(traced)
            mixtures.append(mixture)
        return traced, mixtures, fixed

    def _checked_replay(self, seed, memory, expected_history, atol):
        replay = self.start(seed, memory)
        for expected in expected_history:
            replay = self.step(replay)
            if not np.allclose(replay.latest, expected, atol=atol, rtol=0):
                raise ValueError("edited trajectory fails replay tolerance")
        return replay

    def retarget(self, state, step, target, *, max_update_norm, atol=1e-8, protect=()):
        """Program a one-based step and check its prefix and any protected runs.

        Each protected run is a (seed, step count) pair on the same observation
        bank. Return the candidate kernel, target replay, and weight edit.
        """
        self._require_state(state)
        target = self._vector(target)
        if np.any(np.abs(target) >= 1):
            raise ValueError("target coordinates must be strictly inside tanh saturation")

        bank = state._memory
        baseline, mixtures, fixed = self._trace(state._seed, bank, step)
        protected = mixtures[:-1]
        references = []
        for seed, steps in protect:
            reference, reference_mixtures, _ = self._trace(seed, bank, steps)
            references.append(reference)
            protected.extend(reference_mixtures)
        anchors = np.asarray(protected).reshape(-1, self._width)
        with np.errstate(over="raise", invalid="raise"):
            target_values = np.arctanh(target) - fixed
        edit = program_weight(
            self._observation_value, mixtures[-1:], [target_values],
            preserve_inputs=anchors, max_update_norm=max_update_norm, atol=atol,
        )
        candidate = self._with_observation_value(edit.weight)
        replay = candidate._checked_replay(
            baseline._seed, bank, (*baseline._history[:-1], target), atol,
        )
        for reference in references:
            candidate._checked_replay(reference._seed, bank, reference._history, atol)
        return candidate, replay, edit

    def intervene(self, state, position, hidden):
        """Replace a generated state in a copy for a causal control.

        Later generated rows are held fixed. This differs from replay after an
        observation update, which starts a new trajectory with the new memory.
        """
        self._require_state(state)
        position = index(position)
        if not 0 <= position < len(state._history):
            raise ValueError("generated-state position out of range")
        vector = self._vector(hidden)
        key, value = self._project_generated(vector)
        suffix = position + 1
        return replace(
            state,
            _history=state._history[:position] + (vector,) + state._history[suffix:],
            _keys=state._keys[:position] + (key,) + state._keys[suffix:],
            _values=state._values[:position] + (value,) + state._values[suffix:],
        )
