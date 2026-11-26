"""Continuous feedback over immutable, separately projected observation memory.

The caller supplies numerical observations and weights. Each computed hidden
state is available to later steps as both a query input and an attention row.
"""

from copy import copy
from dataclasses import dataclass, replace
from math import isfinite, log, prod
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


def _source_mass_distribution(scores):
    center = float(np.max(scores))
    weights = np.exp(scores - center)
    total = float(weights.sum())
    return -center - log(total), weights / total


def _source_mass_shift(offsets, contrasts, log_floor, limit):
    """Find the first feasible shift along the local log-mass gradient."""
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        current, probabilities = _source_mass_distribution(offsets)
        if current >= log_floor:
            return np.zeros(contrasts.shape[1])
        direction = -(probabilities @ contrasts)
        scale = float(np.max(np.abs(direction)))
        if not scale or not limit:
            raise ValueError("source mass has no editable ascent direction")
        direction /= scale
        direction /= np.linalg.norm(direction)
        slopes = contrasts @ direction

        def point(shift):
            mass, weights = _source_mass_distribution(offsets + shift * slopes)
            return mass, -float(weights @ slopes)

        upper = limit
        peak, derivative = point(upper)
        if derivative < 0:
            lower = 0.0
            for _ in range(60):
                middle = lower + (upper - lower) * 0.5
                if point(middle)[1] > 0:
                    lower = middle
                else:
                    upper = middle
            peak, upper = max((point(shift)[0], shift) for shift in (lower, upper))
        if peak < log_floor:
            raise ValueError("source mass search found no feasible shift on the bounded direction")
        # Leave room for a different floating-point reduction on installed replay.
        search_floor = log_floor + min(1e-12, (peak - log_floor) * 0.5)
        lower = 0.0
        for _ in range(60):
            middle = lower + (upper - lower) * 0.5
            if point(middle)[0] >= search_floor:
                upper = middle
            else:
                lower = middle
        return upper * direction


def _source_mass_projection(offsets, contrasts, log_floor, limit):
    """Seek a small feasible displacement by projecting onto two halfspaces."""
    displacement = np.zeros(contrasts.shape[1])
    slack = min(1e-10, -log_floor * 0.25)
    with np.errstate(over="raise", invalid="raise", under="ignore", divide="raise"):
        for _ in range(256):
            mass, probabilities = _source_mass_distribution(offsets + contrasts @ displacement)
            if mass >= log_floor:
                return displacement
            gradient = -(probabilities @ contrasts)
            scale = float(np.max(np.abs(gradient)))
            if not scale or not limit:
                raise ValueError("source mass projection found no editable ascent direction")
            normal = gradient / scale
            length = float(np.linalg.norm(normal))
            normal /= length
            shortfall = log_floor - mass
            padded_shortfall = shortfall + min(slack, 0.5 * shortfall)
            gap = (padded_shortfall / scale) / length

            # First project zero onto the new tangent halfspace alone.
            tangent_offset = float(normal @ displacement) + gap
            proposed = max(0.0, tangent_offset) * normal
            radius = _frobenius(displacement)
            if radius:
                previous_normal = displacement / radius
                if previous_normal @ proposed < radius:
                    # Both boundaries are active. Move within the previous plane.
                    tangent = normal - (previous_normal @ normal) * previous_normal
                    tangent -= (previous_normal @ tangent) * previous_normal
                    tangent_norm = _frobenius(tangent)
                    if tangent_norm < 1e-12:
                        raise ValueError("source mass projection has unresolved parallel constraints")
                    proposed = displacement + (gap / tangent_norm) * (tangent / tangent_norm)

            proposed_radius = _frobenius(proposed)
            if not isfinite(proposed_radius):
                raise ValueError("source mass projection is outside numerical range")
            residual_tolerance = (32 * np.finfo(np.float64).eps
                                  * max(radius, proposed_radius, abs(tangent_offset)))
            if (normal @ proposed < tangent_offset - residual_tolerance
                    or (radius and previous_normal @ proposed < radius - residual_tolerance)):
                raise ValueError("source mass projection fails its halfspace residual check")
            if proposed_radius > limit:
                boundary = proposed * (limit / proposed_radius)
                boundary_mass, _ = _source_mass_distribution(offsets + contrasts @ boundary)
                if boundary_mass >= log_floor:
                    return boundary
                raise ValueError("source mass projection found no feasible shift within the norm budget")
            displacement = proposed
    raise ValueError("source mass projection exhausted its iteration budget")


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
    threshold: float | None = None

    def __post_init__(self):
        for name in ("state_axis", "observation_axis", "output"):
            vector = _real_array(getattr(self, name))
            if vector.ndim != 1 or not len(vector):
                raise ValueError("conditional factors must be nonempty vectors")
            if name != "output" and abs(_frobenius(vector) - 1.0) > 1e-12:
                raise ValueError("conditional state and observation axes must have unit norm")
            object.__setattr__(self, name, tuple(vector))
        if self.threshold is not None:
            threshold = float(self.threshold)
            if not isfinite(threshold) or threshold < 0:
                raise ValueError("hinge threshold must be finite and nonnegative")
            object.__setattr__(self, "threshold", threshold)

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


def _outward(lower, upper, scale, operations):
    limits = np.finfo(float)
    operations = max(1, operations)
    padding = (8 * limits.eps * operations) * scale + operations * limits.smallest_subnormal
    return np.nextafter(lower - padding, -np.inf), np.nextafter(upper + padding, np.inf)


def _linear_bounds(bounds, weight):
    if isinstance(weight, FactoredValue):
        return _linear_bounds(_linear_bounds(bounds, weight._left), weight._right)
    lower, upper = bounds
    positive, negative = np.maximum(weight, 0.), np.minimum(weight, 0.)
    low = lower @ positive + upper @ negative
    high = upper @ positive + lower @ negative
    scale = np.maximum(np.abs(lower), np.abs(upper)) @ np.abs(weight)
    return _outward(low, high, scale, lower.size)


def _add_bounds(left, right):
    low, high = left[0] + right[0], left[1] + right[1]
    scale = np.maximum(np.abs(left[0]), np.abs(left[1])) + np.maximum(np.abs(right[0]), np.abs(right[1]))
    return _outward(low, high, scale, 1)


def _product_bounds(left, right):
    corners = np.stack([a * b for a in left for b in right])
    return _outward(corners.min(axis=0), corners.max(axis=0), np.max(np.abs(corners), axis=0), 1)


def _exclusive_logsumexp(logits):
    left = np.concatenate(([-np.inf], np.logaddexp.accumulate(logits[:-1])))
    right = np.concatenate((np.logaddexp.accumulate(logits[:0:-1])[::-1], [-np.inf]))
    return np.logaddexp(left, right)


def _softmax_bounds(lower, upper):
    # A row's mass increases with its own logit and decreases with all others.
    # Exclusive scans avoid cancellation when removing the dominant exponential.
    scale = float(np.max(np.maximum(np.abs(lower), np.abs(upper))))
    padding = (16 * np.finfo(float).eps * len(lower)) * max(1., scale)
    with np.errstate(over="ignore", under="ignore"):
        low = np.exp(-np.logaddexp(0., (_exclusive_logsumexp(upper) - lower) + padding))
        high = np.exp(-np.logaddexp(0., (_exclusive_logsumexp(lower) - upper) - padding))
    return np.maximum(0., np.nextafter(low, -np.inf)), np.minimum(1., np.nextafter(high, np.inf))


def _weighted_bounds(probabilities, rows, last=None):
    width = rows.shape[1]
    low, high = _linear_bounds(probabilities, rows) if len(rows) else (np.zeros(width), np.zeros(width))
    if last is not None:
        low, high = _add_bounds((low, high), _product_bounds(last[0], last[1]))
    return low, high


def _hinge_bounds(gate, threshold):
    if threshold is None:
        return gate
    low, high = _add_bounds(gate, (-threshold, -threshold))
    return np.maximum(0., low), np.maximum(0., high)


def _separating_hinge(hidden, protected, rcond):
    """Search for a strict separator of hidden from protected states and zero."""
    points = np.asarray(protected).reshape(-1, len(hidden))
    scale = max(float(np.max(np.abs(hidden))), float(np.max(np.abs(points), initial=0)))
    if scale == 0:
        raise ValueError("target has no resolved hinge margin")
    target = hidden / scale
    vertices = np.vstack((np.zeros_like(target), points / scale))
    nearest = np.zeros_like(target)
    # Use a bounded Frank-Wolfe search; margin checks do not rely on convergence.
    for _ in range(32):
        residual = target - nearest
        vertex = vertices[np.argmax(vertices @ residual)]
        direction = vertex - nearest
        squared = float(direction @ direction)
        improvement = float(direction @ residual)
        if squared == 0 or improvement <= 0:
            break
        fraction = 1.0 if improvement >= squared else improvement / squared
        updated = nearest + fraction * direction
        if np.array_equal(updated, nearest):
            break
        nearest = updated
    direction = target - nearest
    length = _frobenius(direction)
    if length <= _frobenius(target, scale=rcond):
        raise ValueError("target has no resolved hinge margin")
    axis = direction / length
    threshold = max(0.0, max((float(point @ axis) for point in points), default=0.0))
    margin = float(hidden @ axis) - threshold
    if not isfinite(margin) or margin <= _frobenius(hidden, scale=rcond):
        raise ValueError("target has no resolved hinge margin")
    return axis, threshold


@dataclass(frozen=True, eq=False, repr=False, init=False)
class FeedbackKernel:
    """A bounded shared transition with separate observation and feedback maps."""

    _width: int
    _max_steps: int
    _query: np.ndarray
    _observation_query: np.ndarray | None
    _carry: np.ndarray | None
    _observation_key: np.ndarray
    _observation_value: np.ndarray | FactoredValue
    _feedback_key: np.ndarray
    _feedback_value: np.ndarray | FactoredValue
    _empty: ObservationMemory
    _conditional: tuple[tuple[np.ndarray, np.ndarray, np.ndarray, float | None], ...]

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
        object.__setattr__(self, "_observation_query", None)
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

    def _query_observations(self, hidden, query):
        return query if self._observation_query is None else hidden @ self._observation_query

    def _preactivation(self, hidden, mixture, history_value, has_observations, *, observation_extra=None):
        fixed = np.zeros(self._width) if self._carry is None else hidden @ self._carry
        if history_value is not None:
            fixed += history_value
        if has_observations:
            for state_axis, observation_axis, output, threshold in self._conditional:
                observed = mixture @ observation_axis
                if observed:
                    state_gate = hidden @ state_axis
                    if threshold is not None:
                        state_gate = max(0.0, float(state_gate) - threshold)
                    fixed += (state_gate * observed) * output
        if observation_extra is not None:
            fixed = fixed + observation_extra
        return fixed + _apply_value(self._observation_value, mixture), fixed

    def _step_preactivation(self, state, *, diagnostics=None, observation_extra=None):
        """Evaluate the exact prefix once, with optional batched additive inputs."""
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
                observation_query = self._query_observations(hidden, query) if observation_count else query
                logits = np.concatenate((memory._keys @ observation_query, history_keys @ query))
                # Extreme negative differences may safely round down to zero mass.
                with np.errstate(over="ignore", under="ignore"):
                    weights = np.exp(logits - logits.max())
                weights /= weights.sum()
                mixture = weights[:observation_count] @ memory._observations
                history_value = weights[observation_count:] @ history_values
            preactivation, fixed = self._preactivation(
                hidden, mixture, history_value, observation_count, observation_extra=observation_extra,
            )
            if diagnostics is not None:
                diagnostics.update(
                    logits=logits if observation_count or history_count else np.empty(0),
                    probabilities=weights if observation_count or history_count else np.empty(0),
                    preactivation=preactivation,
                )
        return preactivation, mixture, fixed

    def _bounded_step_work(self, prefix):
        observation_count = len(prefix._memory._keys)
        history_count = len(prefix._keys)
        width, query_width = self._query.shape
        value_cells = sum(
            value._left.size + value._right.size if isinstance(value, FactoredValue) else value.size
            for value in (self._observation_value, self._feedback_value)
        )
        return 16 * (
            4 * width * query_width + width * width + 2 * value_cells
            + (observation_count + history_count + 1) * (width + query_width + 8)
            + width * (8 + 4 * len(self._conditional))
        )

    def _step_bounded(self, prefix, hidden, *, observation_extra):
        """Enclose one step with an exact prefix and one bounded appended row."""
        self._require_state(prefix)
        if len(prefix._history) + 1 >= self._max_steps:
            raise RuntimeError("feedback step budget exhausted")
        memory = prefix._memory
        query_width = self._query.shape[1]
        history_keys = np.asarray(prefix._keys).reshape(-1, query_width)
        history_values = np.asarray(prefix._values).reshape(-1, self._width)
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            query = _linear_bounds(hidden, self._query)
            observation_query = (query if self._observation_query is None
                                 else _linear_bounds(hidden, self._observation_query))
            observed = _linear_bounds(observation_query, memory._keys.T)
            history = _linear_bounds(query, history_keys.T)
            key = _linear_bounds(hidden, self._feedback_key)
            new_score = _linear_bounds(_product_bounds(key, query), np.ones(query_width))
            logits = [np.concatenate((ob, past, [new])) for ob, past, new in zip(observed, history, new_score)]
            if not all(np.isfinite(endpoint).all() for endpoint in logits):
                raise FloatingPointError("bounded attention requires finite logits")
            low, high = _softmax_bounds(*logits)
            count = len(memory._keys)
            mixture = _weighted_bounds((low[:count], high[:count]), memory._observations)
            new_value = _linear_bounds(hidden, self._feedback_value)
            past_value = _weighted_bounds((low[count:-1], high[count:-1]), history_values,
                                          ((low[-1], high[-1]), new_value))
            fixed = _add_bounds(_linear_bounds(hidden, self._carry), past_value) if self._carry is not None else past_value
            if count:
                for state_axis, observation_axis, output, threshold in self._conditional:
                    gate = _hinge_bounds(_linear_bounds(hidden, state_axis), threshold)
                    observed = _linear_bounds(mixture, observation_axis)
                    fixed = _add_bounds(fixed, _product_bounds(_product_bounds(gate, observed), (output, output)))
            if observation_extra is not None:
                fixed = _add_bounds(fixed, observation_extra)
            lower, upper = _add_bounds(fixed, _linear_bounds(mixture, self._observation_value))
            result = np.nextafter(np.tanh(lower), -np.inf), np.nextafter(np.tanh(upper), np.inf)
            # Ordinary step appends a row, so a completion that could fail its
            # generated projections cannot support an exact-step claim.
            _linear_bounds(result, self._feedback_key)
            _linear_bounds(result, self._feedback_value)
            return result

    def _advance(self, state, *, diagnostics=None, observation_extra=None):
        preactivation, mixture, fixed = self._step_preactivation(
            state, diagnostics=diagnostics, observation_extra=observation_extra,
        )
        next_hidden = _frozen(np.tanh(preactivation))
        next_key, next_value = self._project_generated(next_hidden)
        next_state = replace(
            state,
            _history=state._history + (next_hidden,),
            _keys=state._keys + (next_key,),
            _values=state._values + (next_value,),
            observation_scores=state.observation_scores + len(state._memory._keys),
            feedback_scores=state.feedback_scores + len(state._history),
        )
        return next_state, mixture, fixed

    def _with_observation_value(self, weight):
        candidate = copy(self)
        object.__setattr__(candidate, "_observation_value", self._weight(weight))
        return candidate

    def _with_observation_query(self, weight):
        candidate = copy(self)
        object.__setattr__(candidate, "_observation_query", self._weight(weight, self._query.shape))
        return candidate

    @property
    def conditional_terms(self):
        return len(self._conditional)

    def _with_conditional(self, term):
        factors = tuple(self._vector(vector) for vector in
                        (term.state_axis, term.observation_axis, term.output))
        candidate = copy(self)
        object.__setattr__(candidate, "_conditional", self._conditional + ((*factors, term.threshold),))
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

    def _route_program_keys(self, memory, slots):
        if type(self) is not FeedbackKernel:
            raise ValueError("route programming requires a supported softmax kernel")
        if any(not 0 <= slot < len(memory._keys) for slot in slots):
            raise ValueError("route programming requires occupied observation slots")
        return tuple(memory._keys[slot] for slot in slots)

    def _route_program_step(self, state, slots, keys):
        diagnostics = {}
        stepped = self._advance(state, diagnostics=diagnostics)[0]
        first, second = slots
        logits = diagnostics["logits"]
        return stepped, float(logits[first]) - float(logits[second])

    def _source_mass_bank(self, memory, slot):
        self._route_program_keys(memory, (slot,))
        return memory._keys, slot

    def _source_mass_step(self, state, slot):
        if not 0 <= slot < len(state._memory._keys):
            return self.step(state), 0.0
        diagnostics = {}
        stepped = self._advance(state, diagnostics=diagnostics)[0]
        return stepped, float(diagnostics["probabilities"][slot])

    def _route_program_inputs(self, state, step, protect):
        bank = state._memory
        baseline, _, _ = self._trace(state._seed, bank, step)
        prefix = baseline._history[:-1]
        references = [self._trace(seed, bank, steps)[0] for seed, steps in protect]
        protected_inputs = [baseline._seed, *prefix[:-1]] if prefix else []
        for reference in references:
            protected_inputs.extend((reference._seed, *reference._history[:-1]))
        anchors = np.asarray(protected_inputs).reshape(-1, self._width)
        prior = prefix[-1] if prefix else baseline._seed
        weight = self._query if self._observation_query is None else self._observation_query
        return baseline, references, anchors, prior, weight

    def _fit_route_program(self, baseline, references, anchors, prior, weight, target, max_norm, atol):
        edit = program_weight(weight, [prior], [target], preserve_inputs=anchors,
                              max_update_norm=max_norm, atol=atol)
        candidate = self._with_observation_query(edit.weight) if edit.update_norm else self
        bank = baseline._memory
        replay = candidate._checked_replay(baseline._seed, bank, baseline._history[:-1], atol)
        for reference in references:
            candidate._checked_replay(reference._seed, bank, reference._history, atol)
        return candidate, replay, edit

    def program_observation_route(self, state, step, preferred_slot, other_slot, log_odds, *,
                                  max_update_norm, atol=1e-8, protect=()):
        """Fit the selected source log odds and replay protected trajectories.

        The edit affects only observation logits. The bank, values and query
        map for generated rows remain available unchanged. Slots identify two
        rows within this bank, and protect contains (seed, step count) pairs.
        Return the candidate kernel, target replay and weight edit.
        """
        self._require_state(state)
        slots = index(preferred_slot), index(other_slot)
        if slots[0] == slots[1] or not isfinite(log_odds):
            raise ValueError("route programming needs distinct slots and finite log odds")
        keys = self._route_program_keys(state._memory, slots)
        baseline, references, anchors, prior, weight = self._route_program_inputs(state, step, protect)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            contrast = keys[0] - keys[1]
            scale = float(np.max(np.abs(contrast)))
            if scale == 0.0:
                raise ValueError("observation slots have identical projected keys")
            scaled_contrast = contrast / scale
            old_query = prior @ weight
            gap = float(keys[0] @ old_query) - float(keys[1] @ old_query)
            shift = ((log_odds - gap) / scale) / (scaled_contrast @ scaled_contrast)
            target_query = old_query + shift * scaled_contrast
        candidate, replay, edit = self._fit_route_program(
            baseline, references, anchors, prior, weight, target_query, max_update_norm, atol,
        )
        replay, replay_gap = candidate._route_program_step(replay, slots, keys)
        if not isfinite(replay_gap) or abs(replay_gap - log_odds) > atol:
            raise ValueError("programmed observation log odds fail replay tolerance")
        return candidate, replay, edit

    def program_observation_mass(self, state, step, slot, min_mass, *,
                                 max_update_norm, atol=1e-8, protect=(), search="directional"):
        """Fit and replay a floor on one source's share of the joint attention.

        Search along the baseline projected-query gradient or choose
        search="multivariate" to use a bounded sequence of tangent projections.
        Both honor the installed map's Frobenius update cap.
        Return the candidate kernel, target replay and weight edit.
        """
        self._require_state(state)
        slot = index(slot)
        if not 0 < min_mass < 1:
            raise ValueError("source mass floor must be strictly between zero and one")
        if not isfinite(max_update_norm) or max_update_norm < 0:
            raise ValueError("max_update_norm must be finite and nonnegative")
        if search not in ("directional", "multivariate"):
            raise ValueError("source mass search must be directional or multivariate")
        keys, selected = self._source_mass_bank(state._memory, slot)
        baseline, references, anchors, prior, weight = self._route_program_inputs(state, step, protect)
        with np.errstate(over="raise", invalid="raise"):
            query = prior @ weight
            history_keys = np.asarray(baseline._keys[:-1]).reshape(-1, self._query.shape[1])
            scores = np.concatenate((keys @ query, history_keys @ (prior @ self._query)))
            offsets = scores - scores[selected]
            # Generated rows keep fixed scores when the observation query moves.
            contrasts = np.concatenate((keys - keys[selected],
                                        np.broadcast_to(-keys[selected], history_keys.shape)))
            free_input = _remove_span(prior[None, :], _row_basis(anchors, 1e-12))
            free_norm = _frobenius(free_input)
            if free_norm <= _frobenius(prior, scale=1e-12):
                free_norm = 0.0
            limit = max_update_norm * free_norm
            if not isfinite(limit):
                raise ValueError("source mass search bound is outside numerical range")
            solve = _source_mass_projection if search == "multivariate" else _source_mass_shift
            target_query = query + solve(offsets, contrasts, log(float(min_mass)), limit)
        candidate, replay, edit = self._fit_route_program(
            baseline, references, anchors, prior, weight, target_query, max_update_norm, atol,
        )
        replay, achieved = candidate._source_mass_step(replay, slot)
        if not achieved >= min_mass:
            raise ValueError("programmed source mass falls below the replay floor")
        return candidate, replay, edit

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
                            atol=1e-8, protect=(), hinge=False):
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
            threshold = None
            if hinge:
                state_axis, threshold = _separating_hinge(hidden, protected, rcond)
            else:
                basis = _row_basis(np.asarray(protected).reshape(-1, self._width), rcond)
                direction = _remove_span(hidden, basis)
                length = _frobenius(direction)
                if not isfinite(length) or length <= _frobenius(hidden, scale=rcond):
                    raise ValueError("target has no unprotected state direction")
                state_axis = direction / length
            observed = _frobenius(mixture)
            if not isfinite(observed) or observed == 0:
                raise ValueError("target has no nonzero observation mixture")
            observation_axis = mixture / observed
            state_gate = float(hidden @ state_axis)
            if threshold is not None:
                state_gate = max(0.0, state_gate - threshold)
            gate = state_gate * float(mixture @ observation_axis)
            if not isfinite(gate) or gate <= 0:
                raise ValueError("conditional gate is numerically unresolved")
            residual = np.arctanh(target) - (fixed + _apply_value(self._observation_value, mixture))
            term = ConditionalTerm(state_axis, observation_axis, residual / gate, threshold=threshold)
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
