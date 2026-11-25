"""Continuous feedback using reversible, quantized linear attention summaries."""

from dataclasses import dataclass, replace
from operator import index

import numpy as np

from experiments.memory.latent_feedback import (
    FeedbackKernel, FeedbackState, _apply_value, _frozen, _real_array,
)


_LEVELS = 127
_MAX_ROWS = (1 << 31) - 1


def _sign_features(values):
    return np.concatenate((np.maximum(values, 0), np.maximum(-values, 0)), axis=-1)


def _quantize(values):
    return np.rint(_LEVELS * values).astype(np.int64)


@dataclass(frozen=True, eq=False)
class MomentMemory:
    _key_map: np.ndarray
    _moments: np.ndarray
    count: int

    @property
    def nbytes(self):
        return self._moments.nbytes


@dataclass(frozen=True, eq=False)
class MomentState(FeedbackState):
    _memory: MomentMemory
    _history_moments: np.ndarray | None = None


@dataclass(frozen=True, eq=False, repr=False, init=False)
class LinearFeedbackKernel(FeedbackKernel):
    """Attention with same-sign features and no per-step scan of retained rows.

    Observation coordinates and key maps use a fixed grid on [-1, 1].
    Query projections remain continuous.
    """

    _integer_observation_key: np.ndarray
    _integer_feedback_key: np.ndarray
    _pair_left: np.ndarray
    _pair_right: np.ndarray

    def __init__(self, *, query, carry, observation_key, observation_value,
                 feedback_key, feedback_value, max_steps, feature_pairs=()):
        observation_key, feedback_key = map(_real_array, (observation_key, feedback_key))
        if observation_key.ndim != 2:
            raise ValueError("linear feedback key maps must be matrices")
        if np.any(np.abs(observation_key) > 1) or np.any(np.abs(feedback_key) > 1):
            raise ValueError("linear feedback key maps must be within [-1, 1]")
        width = observation_key.shape[1]
        pairs = tuple(tuple(sorted((index(left), index(right)))) for left, right in feature_pairs)
        if any(not 0 <= left < right < 2 * width or left % width == right % width
               for left, right in pairs) or len(set(pairs)) != len(pairs):
            raise ValueError("feature pairs need unique signed axes from distinct key coordinates")
        object.__setattr__(self, "_pair_left", _frozen(np.array([left for left, _ in pairs], dtype=np.intp)))
        object.__setattr__(self, "_pair_right", _frozen(np.array([right for _, right in pairs], dtype=np.intp)))
        object.__setattr__(self, "_integer_observation_key", _frozen(_quantize(observation_key)))
        object.__setattr__(self, "_integer_feedback_key", _frozen(_quantize(feedback_key)))
        super().__init__(query=query, carry=carry, observation_key=observation_key,
                         observation_value=observation_value, feedback_key=feedback_key,
                         feedback_value=feedback_value, max_steps=max_steps)

    def _summarize(self, rows, key_map):
        payload = _quantize(rows)
        projected = payload @ key_map
        magnitude = (np.abs(projected) + _LEVELS // 2) // _LEVELS
        keys = np.sign(projected) * np.minimum(magnitude, _LEVELS)
        features = self._add_pairs(_sign_features(keys), _LEVELS)
        # Column zero carries the unnormalized mass for each feature.
        packed = np.column_stack((np.ones(len(rows), dtype=np.int64), payload))
        return features.T @ packed

    def _add_pairs(self, unary, unary_scale):
        if not len(self._pair_left):
            return unary
        paired = unary[..., self._pair_left] * unary[..., self._pair_right]
        return np.concatenate((unary * unary_scale, paired), axis=-1)

    def prepare(self, observations):
        """Compile a bank into a feature mass and weighted coordinate summary."""
        rows = _real_array(observations)
        if rows.ndim != 2 or rows.shape[1] != self._width:
            raise ValueError("observation matrix has incorrect width")
        if max(rows.shape) > _MAX_ROWS or np.any(np.abs(rows) > 1):
            raise ValueError("observations exceed the row or [-1, 1] coordinate bound")
        return MomentMemory(
            self._observation_key, _frozen(self._summarize(rows, self._integer_observation_key)),
            len(rows),
        )

    def _check_memory(self, memory):
        if not isinstance(memory, MomentMemory) or memory._key_map is not self._observation_key:
            raise ValueError("memory belongs to a different linear feedback key map")

    def revise(self, memory, *, remove=None, append=None):
        """Apply source-ledger-verified row retractions and additions to a snapshot."""
        self._check_memory(memory)
        retired = self._empty if remove is None else self.prepare(remove)
        added = self._empty if append is None else self.prepare(append)
        count = memory.count - retired.count + added.count
        if retired.count > memory.count or count > _MAX_ROWS:
            raise ValueError("revision exceeds the retained row bounds")
        remaining = memory._moments - retired._moments
        mass = remaining[:, 0]
        max_feature = _LEVELS ** (2 if len(self._pair_left) else 1)
        if (np.any(mass < 0) or np.any(mass > (memory.count - retired.count) * max_feature)
                or np.any(np.abs(remaining[:, 1:]) > mass[:, None] * _LEVELS)):
            raise ValueError("retraction exceeds the retained feature moments")
        return MomentMemory(self._observation_key, _frozen(remaining + added._moments), count)

    def start(self, seed, memory=None):
        memory = self._empty if memory is None else memory
        self._check_memory(memory)
        return MomentState(self, memory, self._vector(seed))

    def _advance(self, state, *, diagnostics=None):
        self._require_state(state)
        if len(state._history) >= min(self._max_steps, _MAX_ROWS):
            raise RuntimeError("feedback step budget exhausted")
        obs = state._memory._moments
        generated = state._history_moments
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            hidden = state.latest
            unary = np.minimum(_sign_features(hidden @ self._query), 1)
            query = self._add_pairs(unary, 1)
            mass = obs[:, 0] if generated is None else obs[:, 0] + generated[:, 0]
            query[mass == 0] = 0
            scale = query.max()
            mixture = np.zeros(self._width)
            history_value = None
            routed_mass = 0.0
            if scale > 0:
                query /= scale
                routed_mass = float(query @ mass)
                mixture = (query @ obs[:, 1:]) / (routed_mass * _LEVELS)
                if generated is not None:
                    history_mix = (query @ generated[:, 1:]) / (routed_mass * _LEVELS)
                    history_value = _apply_value(self._feedback_value, history_mix)
            preactivation, fixed = self._preactivation(
                hidden, mixture, history_value, state._memory.count,
            )
            next_hidden = _frozen(np.tanh(preactivation))
            if diagnostics is not None:
                diagnostics.update(routed_mass=routed_mass, preactivation=preactivation)
        next_moments = self._summarize(next_hidden[None, :], self._integer_feedback_key)
        if generated is not None:
            next_moments += generated
        return replace(
            state, _history=state._history + (next_hidden,), _history_moments=_frozen(next_moments),
        ), mixture, fixed

    def intervene(self, state, position, hidden):
        """Replace one generated row's summary contribution while holding later rows fixed."""
        self._require_state(state)
        position = index(position)
        if not 0 <= position < len(state._history):
            raise ValueError("generated-state position out of range")
        vector = self._vector(hidden)
        if np.any(np.abs(vector) > 1):
            raise ValueError("intervention coordinates must be within [-1, 1]")
        retired = self._summarize(state._history[position][None, :], self._integer_feedback_key)
        added = self._summarize(vector[None, :], self._integer_feedback_key)
        history = state._history[:position] + (vector,) + state._history[position + 1:]
        return replace(state, _history=history,
                       _history_moments=_frozen(state._history_moments - retired + added))
