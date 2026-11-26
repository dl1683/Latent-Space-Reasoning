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
class AssociationMemory:
    _key_map: np.ndarray
    _moments: np.ndarray


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
    _disagreement_axes: tuple[int, ...]
    _disagreement_value: np.ndarray | None
    _payload_branch: tuple[int, int] | None

    def __init__(self, *, query, carry, observation_key, observation_value,
                 feedback_key, feedback_value, max_steps, feature_pairs=(),
                 disagreement_axes=(), disagreement_value=None, payload_branch=None):
        query = _real_array(query)
        if query.ndim != 2 or not all(query.shape):
            raise ValueError("query map must be a nonempty matrix")
        axes = tuple(map(index, disagreement_axes))
        if len(set(axes)) != len(axes) or any(not 0 <= axis < len(query) for axis in axes):
            raise ValueError("disagreement axes must be unique state coordinates")
        if disagreement_value is not None:
            disagreement_value = _real_array(disagreement_value)
            if not axes or disagreement_value.shape != (len(axes), len(query)):
                raise ValueError("disagreement value map has incorrect shape")
            disagreement_value = _frozen(disagreement_value)
        object.__setattr__(self, "_disagreement_axes", axes)
        object.__setattr__(self, "_disagreement_value", disagreement_value)
        if payload_branch is not None:
            payload_branch = tuple(map(index, payload_branch))
            if len(payload_branch) != 2 or any(not 0 <= axis < len(query) for axis in payload_branch):
                raise ValueError("payload branch requires valid payload and selector axes")
        object.__setattr__(self, "_payload_branch", payload_branch)
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

    def _summarize(self, rows, key_map, *, observation=False):
        payload = _quantize(rows)
        return self._summarize_quantized(payload, payload, key_map, observation=observation)

    def _summarize_quantized(self, payload, addresses, key_map, *, observation):
        projected = addresses @ key_map
        magnitude = (np.abs(projected) + _LEVELS // 2) // _LEVELS
        keys = np.sign(projected) * np.minimum(magnitude, _LEVELS)
        features = self._add_pairs(_sign_features(keys), _LEVELS)
        # Column zero carries the unnormalized mass for each feature.
        base = np.column_stack((np.ones(len(payload), dtype=np.int64), payload))
        columns = [base]
        if observation:
            if self._disagreement_axes:
                columns.append(_sign_features(payload[:, self._disagreement_axes]))
            if self._payload_branch is not None:
                axis, _ = self._payload_branch
                columns.extend((base * (payload[:, [axis]] > 0), base * (payload[:, [axis]] < 0)))
        packed = np.column_stack(columns) if len(columns) > 1 else base
        return features.T @ packed

    def _add_pairs(self, unary, unary_scale):
        if not len(self._pair_left):
            return unary
        paired = unary[..., self._pair_left] * unary[..., self._pair_right]
        return np.concatenate((unary * unary_scale, paired), axis=-1)

    def prepare(self, observations):
        """Compile a bank into a feature mass and weighted coordinate summary."""
        rows = self._bounded_rows(observations)
        return MomentMemory(
            self._observation_key,
            _frozen(self._summarize(rows, self._integer_observation_key, observation=True)),
            len(rows),
        )

    def _bounded_rows(self, observations):
        rows = _real_array(observations)
        if rows.ndim != 2 or rows.shape[1] != self._width:
            raise ValueError("observation matrix has incorrect width")
        if max(rows.shape) > _MAX_ROWS or np.any(np.abs(rows) > 1):
            raise ValueError("observations exceed the row or [-1, 1] coordinate bound")
        return rows

    def prepare_associations(self, addresses, values):
        """Compile separately encoded addresses and values for later association reads."""
        addresses, values = map(self._bounded_rows, (addresses, values))
        if addresses.shape != values.shape:
            raise ValueError("association addresses and values must have the same shape")
        moments = self._summarize_quantized(_quantize(values), _quantize(addresses),
                                           self._integer_observation_key, observation=False)
        return AssociationMemory(self._observation_key, _frozen(moments))

    def _check_memory(self, memory):
        if not isinstance(memory, MomentMemory) or memory._key_map is not self._observation_key:
            raise ValueError("memory belongs to a different linear feedback key map")

    def _disagreement_columns(self, moments):
        start = 1 + self._width
        return moments[:, start:start + 2 * len(self._disagreement_axes)]

    def _branch_partitions(self, moments):
        start = 1 + self._width + 2 * len(self._disagreement_axes)
        return np.split(moments[:, start:], 2, axis=1)

    def _check_branch_retraction(self, remaining):
        axis, _ = self._payload_branch
        positive, negative = self._branch_partitions(remaining)
        neutral = remaining[:, :1 + self._width] - positive - negative
        for partition in (positive, negative, neutral):
            mass = partition[:, :1]
            if np.any(mass < 0) or np.any(np.abs(partition[:, 1:]) > mass * _LEVELS):
                raise ValueError("retraction exceeds the retained payload partition")
        if (np.any(positive[:, 1 + axis] < positive[:, 0])
                or np.any(negative[:, 1 + axis] > -negative[:, 0])
                or np.any(neutral[:, 1 + axis] != 0)):
            raise ValueError("retraction violates the retained payload partition signs")

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
        if self._disagreement_axes:
            positive, negative = np.split(self._disagreement_columns(remaining), 2, axis=1)
            means = remaining[:, 1:1 + self._width][:, self._disagreement_axes]
            if (np.any(positive < 0) or np.any(negative < 0)
                    or np.any(positive + negative > mass[:, None] * _LEVELS)
                    or np.any(positive - negative != means)):
                raise ValueError("retraction exceeds the retained disagreement moments")
        if self._payload_branch is not None:
            self._check_branch_retraction(remaining)
        return MomentMemory(self._observation_key, _frozen(remaining + added._moments), count)

    def start(self, seed, memory=None):
        memory = self._empty if memory is None else memory
        self._check_memory(memory)
        return MomentState(self, memory, self._vector(seed))

    def _query_features(self, hidden):
        unary = np.minimum(_sign_features(hidden @ self._query), 1)
        return self._add_pairs(unary, 1)

    @staticmethod
    def _scaled_query(features, mass):
        query = np.where(mass > 0, features, 0)
        scale = query.max()
        if scale > 0:
            query /= scale
        return query

    def _read_disagreement(self, features, obs):
        query = self._scaled_query(features, obs[:, 0])
        mass = float(query @ obs[:, 0])
        if mass == 0:
            return np.zeros(len(self._disagreement_axes))
        sign_values = (query @ self._disagreement_columns(obs)) / (mass * _LEVELS)
        positive, negative = np.split(sign_values, 2)
        return np.minimum(2 * np.minimum(positive, negative), 1)

    def observation_disagreement(self, state):
        """Read balanced opposing payload support in the configured axis order."""
        self._require_state(state)
        if not self._disagreement_axes:
            return _frozen(np.empty(0))
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _frozen(self._read_disagreement(
                self._query_features(state.latest), state._memory._moments,
            ))

    def _selected_payloads(self, hidden, moments):
        base = moments[:, 1:1 + self._width]
        if self._payload_branch is None:
            return base
        _, selector = self._payload_branch
        control = float(np.clip(hidden[selector], -1, 1))
        if control == 0:
            return base
        positive, negative = self._branch_partitions(moments)
        partition = (positive if control > 0 else negative)[:, 1:]
        strength = abs(control)
        if strength == 1:
            return partition
        return (1 - strength) * base + strength * partition

    def _observation_mixture(self, hidden, query, moments, routed_mass):
        return (query @ self._selected_payloads(hidden, moments)) / (routed_mass * _LEVELS)

    def read_observations(self, hidden, memory):
        """Return the retained value and its unscaled routing mass in unit-row weights."""
        self._check_memory(memory)
        return self._read_bank(hidden, memory, projected=True)

    def read_associations(self, hidden, memory):
        """Read already projected values without applying the source value map again."""
        if type(memory) is not AssociationMemory or memory._key_map is not self._observation_key:
            raise ValueError("expected an association bank from the same linear key map")
        return self._read_bank(hidden, memory, projected=False)

    def _read_bank(self, hidden, memory, *, projected):
        hidden = self._vector(hidden)
        moments = memory._moments
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            features = self._query_features(hidden)
            scale = _LEVELS ** (2 if len(self._pair_left) else 1)
            support = float(features @ moments[:, 0]) / scale
            query = self._scaled_query(features, moments[:, 0])
            mass = float(query @ moments[:, 0])
            if not mass:
                return _frozen(np.zeros(self._width)), support
            if projected:
                mixture = self._observation_mixture(hidden, query, moments, mass)
                value = _apply_value(self._observation_value, mixture)
                if self._disagreement_value is not None:
                    value = value + self._read_disagreement(query, moments) @ self._disagreement_value
            else:
                value = (query @ moments[:, 1:1 + self._width]) / (mass * _LEVELS)
        if not np.isfinite(value).all():
            raise FloatingPointError("nonfinite observation read")
        return _frozen(value), support

    def _advance(self, state, *, diagnostics=None):
        self._require_state(state)
        if len(state._history) >= min(self._max_steps, _MAX_ROWS):
            raise RuntimeError("feedback step budget exhausted")
        obs = state._memory._moments
        generated = state._history_moments
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            hidden = state.latest
            features = self._query_features(hidden)
            mass = obs[:, 0] if generated is None else obs[:, 0] + generated[:, 0]
            query = self._scaled_query(features, mass)
            mixture = np.zeros(self._width)
            history_value = None
            routed_mass = float(query @ mass)
            if routed_mass > 0:
                mixture = self._observation_mixture(hidden, query, obs, routed_mass)
                if generated is not None:
                    history_mix = (query @ generated[:, 1:]) / (routed_mass * _LEVELS)
                    history_value = _apply_value(self._feedback_value, history_mix)
            extra = None
            if self._disagreement_value is not None:
                extra = self._read_disagreement(features, obs) @ self._disagreement_value
            preactivation, fixed = self._preactivation(
                hidden, mixture, history_value, state._memory.count, observation_extra=extra,
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
