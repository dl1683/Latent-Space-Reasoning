"""Continuous feedback over immutable, separately projected observation memory.

The caller supplies numerical observations and weights. Each computed hidden
state is available to later steps as both a query input and an attention row.
"""

from copy import copy
from dataclasses import dataclass, replace
from math import isfinite, prod
from operator import index

import numpy as np

from experiments.memory.weight_program import _frobenius, _remove_span, _row_basis, program_weight


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


@dataclass(frozen=True)
class ConditionalTerm:
    """Data for a rank-one interaction between state and observation axes."""

    state_axis: tuple[float, ...]
    observation_axis: tuple[float, ...]
    output: tuple[float, ...]

    def __post_init__(self):
        for name in ("state_axis", "observation_axis", "output"):
            vector = _real_array(getattr(self, name))
            if vector.ndim != 1 or not len(vector):
                raise ValueError("conditional factors must be nonempty vectors")
            if name != "output" and abs(_frobenius(vector) - 1.0) > 1e-12:
                raise ValueError("conditional state and observation axes must have unit norm")
            object.__setattr__(self, name, tuple(vector))

    @property
    def norm(self):
        factors = (self.state_axis, self.observation_axis, self.output)
        return prod(_frobenius(np.asarray(factor)) for factor in factors)


@dataclass(frozen=True, eq=False, repr=False, init=False)
class FactoredValue:
    """A square value map represented by two rectangular factors."""

    _left: np.ndarray
    _right: np.ndarray

    def __init__(self, left, right):
        left = _real_array(left)
        right = _real_array(right)
        if (left.ndim != 2 or right.ndim != 2 or not all(left.shape)
                or right.shape != left.shape[::-1]):
            raise ValueError("invalid factored value dimensions")
        object.__setattr__(self, "_left", _frozen(left))
        object.__setattr__(self, "_right", _frozen(right))

    @property
    def width(self):
        return self._left.shape[0]

    def apply(self, vector):
        return (vector @ self._left) @ self._right

    def dense(self):
        with np.errstate(over="raise", invalid="raise"):
            return self._left @ self._right


def _apply_value(value, vector):
    if isinstance(value, FactoredValue):
        return value.apply(vector)
    return vector @ value


@dataclass(frozen=True, eq=False, repr=False, init=False)
class FeedbackKernel:
    """A bounded shared transition with separate observation and feedback maps."""

    _width: int
    _max_steps: int
    _query: np.ndarray
    _carry: np.ndarray | None
    _observation_key: np.ndarray
    _observation_value: np.ndarray | FactoredValue
    _feedback_key: np.ndarray
    _feedback_value: np.ndarray | FactoredValue
    _empty: ObservationMemory
    _conditional: tuple[tuple[np.ndarray, ...], ...]

    def __init__(self, *, query, carry, observation_key, observation_value,
                 feedback_key, feedback_value, max_steps):
        query = _real_array(query)
        if query.ndim != 2 or not all(query.shape):
            raise ValueError("query map must be a nonempty matrix")
        object.__setattr__(self, "_width", len(query))
        object.__setattr__(self, "_max_steps", index(max_steps))
        if self._max_steps <= 0:
            raise ValueError("max_steps must be positive")
        object.__setattr__(self, "_query", _frozen(query))
        object.__setattr__(self, "_carry", None if carry is None else self._weight(carry))
        object.__setattr__(self, "_observation_key", self._weight(observation_key, query.shape))
        object.__setattr__(self, "_observation_value", self._value(observation_value))
        object.__setattr__(self, "_feedback_key", self._weight(feedback_key, query.shape))
        object.__setattr__(self, "_feedback_value", self._value(feedback_value))
        object.__setattr__(self, "_empty", self.prepare(np.empty((0, self._width))))
        object.__setattr__(self, "_conditional", ())

    def _weight(self, value, shape=None):
        matrix = _real_array(value)
        expected = (self._width, self._width) if shape is None else shape
        if matrix.shape != expected:
            raise ValueError("weight map has incorrect shape")
        return _frozen(matrix)

    def _value(self, value):
        if isinstance(value, FactoredValue):
            if value.width != self._width:
                raise ValueError("factored value map has incorrect width")
            return value
        return self._weight(value)

    def _dense_observation_value(self):
        value = self._observation_value
        return value.dense() if isinstance(value, FactoredValue) else value

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
            return _frozen(vector @ self._feedback_key), _frozen(_apply_value(self._feedback_value, vector))

    def step(self, state):
        """Append one generated vector and tally the attention scores evaluated."""
        return self._advance(state)[0]

    def _preactivation(self, hidden, mixture, history_value, has_observations):
        fixed = np.zeros(self._width) if self._carry is None else hidden @ self._carry
        if history_value is not None:
            fixed += history_value
        if has_observations:
            for state_axis, observation_axis, output in self._conditional:
                observed = mixture @ observation_axis
                if observed:
                    fixed += ((hidden @ state_axis) * observed) * output
        return fixed + _apply_value(self._observation_value, mixture), fixed

    def _advance(self, state, *, diagnostics=None):
        self._require_state(state)
        if len(state._history) >= self._max_steps:
            raise RuntimeError("feedback step budget exhausted")
        memory = state._memory
        observation_count = len(memory._keys)
        history_count = len(state._history)
        with np.errstate(over="raise", invalid="raise"):
            hidden = state.latest
            query = hidden @ self._query
            mixture = np.zeros(self._width)
            history_value = None
            if observation_count or history_count:
                history_keys = np.asarray(state._keys).reshape(-1, self._query.shape[1])
                history_values = np.asarray(state._values).reshape(-1, self._width)
                logits = np.concatenate((memory._keys @ query, history_keys @ query))
                # Extreme negative differences may safely round down to zero mass.
                with np.errstate(over="ignore", under="ignore"):
                    weights = np.exp(logits - logits.max())
                weights /= weights.sum()
                mixture = weights[:observation_count] @ memory._observations
                history_value = weights[observation_count:] @ history_values
            preactivation, fixed = self._preactivation(
                hidden, mixture, history_value, observation_count,
            )
            next_hidden = _frozen(np.tanh(preactivation))
            if diagnostics is not None:
                diagnostics.update(
                    logits=logits if observation_count or history_count else np.empty(0),
                    probabilities=weights if observation_count or history_count else np.empty(0),
                    preactivation=preactivation,
                )
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

    @property
    def conditional_terms(self):
        return len(self._conditional)

    def _with_conditional(self, term):
        factors = tuple(self._vector(vector) for vector in
                        (term.state_axis, term.observation_axis, term.output))
        candidate = copy(self)
        object.__setattr__(candidate, "_conditional", self._conditional + (factors,))
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

    def _edit_traces(self, state, step, target, protect):
        self._require_state(state)
        target = self._vector(target)
        if np.any(np.abs(target) >= 1):
            raise ValueError("target coordinates must be strictly inside tanh saturation")
        bank = state._memory
        baseline, mixtures, fixed = self._trace(state._seed, bank, step)
        references = [self._trace(seed, bank, steps) for seed, steps in protect]
        return target, baseline, mixtures, fixed, references

    def _check_edit(self, baseline, target, references, atol):
        bank = baseline._memory
        replay = self._checked_replay(
            baseline._seed, bank, (*baseline._history[:-1], target), atol,
        )
        for reference, _, _ in references:
            self._checked_replay(reference._seed, bank, reference._history, atol)
        return replay

    def retarget(self, state, step, target, *, max_update_norm, atol=1e-8, protect=()):
        """Program a one-based step and check its prefix and any protected runs.

        Each protected run is a (seed, step count) pair on the same observation
        bank. Return the candidate kernel, target replay, and weight edit.
        """
        target, baseline, mixtures, fixed, references = self._edit_traces(
            state, step, target, protect,
        )
        protected = mixtures[:-1]
        for _, reference_mixtures, _ in references:
            protected.extend(reference_mixtures)
        anchors = np.asarray(protected).reshape(-1, self._width)
        with np.errstate(over="raise", invalid="raise"):
            target_values = np.arctanh(target) - fixed
        edit = program_weight(
            self._dense_observation_value(), mixtures[-1:], [target_values],
            preserve_inputs=anchors, max_update_norm=max_update_norm, atol=atol,
        )
        candidate = self._with_observation_value(edit.weight)
        return candidate, candidate._check_edit(baseline, target, references, atol), edit

    def program_conditional(self, state, step, target, *, max_update_norm,
                            atol=1e-8, protect=()):
        """Append one interaction for a target that has a distinct prior state."""
        if not isfinite(max_update_norm) or max_update_norm < 0:
            raise ValueError("max_update_norm must be finite and nonnegative")
        if not isfinite(atol) or atol <= 0:
            raise ValueError("atol must be positive and finite")
        target, baseline, mixtures, fixed, references = self._edit_traces(
            state, step, target, protect,
        )
        inputs = (baseline._seed, *baseline._history[:-1])
        hidden, mixture = inputs[-1], mixtures[-1]
        protected = list(inputs[:-1])
        for reference, _, _ in references:
            protected.extend((reference._seed, *reference._history[:-1]))
        rcond = 1e-12
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            basis = _row_basis(np.asarray(protected).reshape(-1, self._width), rcond)
            direction = _remove_span(hidden, basis)
            length = _frobenius(direction)
            observed = _frobenius(mixture)
            if not isfinite(length) or length <= _frobenius(hidden, scale=rcond):
                raise ValueError("target has no unprotected state direction")
            if not isfinite(observed) or observed == 0:
                raise ValueError("target has no nonzero observation mixture")
            state_axis, observation_axis = direction / length, mixture / observed
            gate = float(hidden @ state_axis) * float(mixture @ observation_axis)
            if not isfinite(gate) or gate <= 0:
                raise ValueError("conditional gate is numerically unresolved")
            residual = np.arctanh(target) - (fixed + _apply_value(self._observation_value, mixture))
            term = ConditionalTerm(state_axis, observation_axis, residual / gate)
        norm = term.norm
        if not isfinite(norm) or norm > max_update_norm:
            raise ValueError("conditional term exceeds the update norm budget")
        candidate = self._with_conditional(term)
        return candidate, candidate._check_edit(baseline, target, references, atol), term

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
