"""Continuous softmax feedback with revisable summaries of observation slots."""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from itertools import count
from math import exp, expm1, isfinite, log, tanh
from operator import index
from types import MappingProxyType

import numpy as np

from experiments.memory.latent_feedback import (
    FactoredValue, FeedbackKernel, FeedbackState, _apply_value, _frozen, _real_array,
)


@dataclass(frozen=True, eq=False)
class _Node:
    count: int
    lower_key: np.ndarray
    upper_key: np.ndarray
    mean_row: np.ndarray
    left: _Node | None = None
    right: _Node | None = None


def _join(left, right):
    if left is None or right is None:
        child = left if right is None else right
        if child is None:
            return None
        return replace(child, left=left, right=right)
    size = left.count + right.count
    mean = _frozen((left.count / size) * left.mean_row + (right.count / size) * right.mean_row)
    return _Node(size, _frozen(np.minimum(left.lower_key, right.lower_key)),
                 _frozen(np.maximum(left.upper_key, right.upper_key)), mean, left, right)


def _set_slots(node, depth, edits):
    if not edits:
        return node
    if depth == 0:
        (slot, leaf), = edits
        if node is None and leaf is None:
            raise KeyError(slot)
        return leaf
    left = None if node is None else node.left
    right = None if node is None else node.right
    bit = 1 << (depth - 1)
    left_edits, right_edits = [], []
    for slot, leaf in edits:
        (right_edits if slot & bit else left_edits).append((slot, leaf))
    return _join(_set_slots(left, depth - 1, left_edits),
                 _set_slots(right, depth - 1, right_edits))


@dataclass(frozen=True, eq=False)
class TreeMemory:
    _key_map: np.ndarray
    _depth: int
    _root: _Node | None

    @property
    def count(self):
        return 0 if self._root is None else self._root.count


@dataclass(frozen=True)
class TreeAttention:
    """Work and exact-arithmetic softmax error model for one completed step."""

    groups: int
    scored_nodes: int
    joint_l1_bound: float
    reused_route: bool = False
    reused_node_scores: int = 0


@dataclass(frozen=True, eq=False)
class TreeFeedbackState(FeedbackState):
    _memory: TreeMemory
    attention: TreeAttention | None = None


@dataclass(frozen=True, eq=False)
class TreeFeedbackBoundedRun:
    """Tree trajectory and per-step max-coordinate error models against dense replay."""

    state: TreeFeedbackState
    error_bounds: tuple[float, ...]
    reference_memory: TreeMemory | None = None
    revision_group_scores: int = 0
    reused_opening_steps: int = 0


def _column_l1(matrix):
    return np.abs(matrix).sum(axis=0)


def _value_linf(value):
    if isinstance(value, FactoredValue):
        return (_column_l1(value._left) @ np.abs(value._right)).max()
    return _column_l1(value).max()


@dataclass(frozen=True, eq=False)
class _TreeRoute:
    mixture: np.ndarray
    history_value: np.ndarray | None
    attention: TreeAttention
    query_bytes: bytes | None = None
    node_scores: Mapping[_Node, _Score] | None = None

    def retain(self):
        history = None if self.history_value is None else _frozen(self.history_value)
        return replace(self, mixture=_frozen(self.mixture), history_value=history)


@dataclass(frozen=True, eq=False)
class _TreeReplay:
    kernel: TreeFeedbackKernel
    memory: TreeMemory
    seed: bytes
    steps: tuple[tuple[_TreeRoute, np.ndarray], ...]
    final: TreeFeedbackState
    error_policy: tuple[tuple[str, str], ...]

    def matches(self, kernel, memory, seed, steps):
        return (self.kernel is kernel and self.memory is memory and self.seed == seed
                and 0 < steps <= len(self.steps)
                and all(mode not in ("call", "log") for _, mode in self.error_policy)
                and self.error_policy == tuple(sorted(np.geterr().items())))

    def prefix(self):
        return tuple(hidden.view() for _, hidden in self.steps[:-1])


@dataclass(frozen=True, eq=False)
class _Score:
    node: _Node
    logit: float
    upper_logit: float
    lost_fraction: float
    radius: float


def _scorable(node):
    while node.count > 1 and (node.left is None or node.right is None):
        node = node.right if node.left is None else node.left
    return node


def _key_interval(node, query, positive_query):
    if node.count == 1:
        lower = upper = float(node.lower_key @ query)
    else:
        lower = float(np.where(positive_query, node.lower_key, node.upper_key) @ query)
        upper = float(np.where(positive_query, node.upper_key, node.lower_key) @ query)
    if not isfinite(lower) or not isfinite(upper) or upper < lower:
        raise FloatingPointError("unresolved tree attention key interval")
    return lower, upper


def _score(node, query, positive_query):
    lower, upper = _key_interval(node, query, positive_query)
    logit = lower if lower == upper else lower * 0.5 + upper * 0.5
    radius = upper * 0.5 - lower * 0.5
    return _Score(node, logit, upper, -expm1(lower - upper), radius)


def _revision_groups(opening, reference):
    shared, removed, added, paired = [], [], [], []
    pending = [(opening, reference)]
    while pending:
        old, new = pending.pop()
        if old is new:
            if old is not None:
                shared.append(old)
        elif old is None:
            added.append(new)
        elif new is None:
            removed.append(old)
        elif old.left is None and old.right is None:
            reference_leaf = None if np.array_equal(old.lower_key, new.lower_key) else new
            paired.append((old, reference_leaf, new.mean_row - old.mean_row))
        else:
            pending.extend(((old.left, new.left), (old.right, new.right)))
    return shared, removed, added, paired


def _shifted_mass(scores, offset):
    return sum(exp((score - offset) + log(multiplicity)) for score, multiplicity in scores)


def _revision_mass_gap(common, before, after, paired_scores):
    offset = max(
        max((score for scores in (common, before, after) for score, _ in scores), default=-np.inf),
        max((max(old, new) for old, new in paired_scores), default=-np.inf),
    )
    kept = _shifted_mass(common, offset)
    lost = _shifted_mass(before, offset)
    gained = _shifted_mass(after, offset)
    for old, new in paired_scores:
        low, high = min(old, new), max(old, new)
        kept += exp(low - offset)
        # Subtracting the two exponentials can erase a small difference.
        excess = exp(high - offset) * -expm1(low - high)
        if old > new:
            lost += excess
        else:
            gained += excess
    changed = max(lost, gained)
    overlap_gap = 2. * (changed / (kept + changed))
    if not before and not after:
        shifts = [new - old for old, new in paired_scores]
        if common:
            shifts.append(0.)
        if all(isfinite(shift) for shift in shifts):
            overlap_gap = min(overlap_gap, 2. * tanh((max(shifts) - min(shifts)) / 4.))
    return overlap_gap


def _revision_attention_change(groups, observation_query, history_logits, paired_deltas):
    positive = observation_query >= 0
    shared, removed, added, paired = groups
    common = [(float(score), 1) for score in history_logits]
    common.extend((_key_interval(node, observation_query, positive)[0], node.count) for node in shared)
    paired_scores = []
    for opening, reference, _ in paired:
        score = _key_interval(opening, observation_query, positive)[0]
        updated = score if reference is None else _key_interval(reference, observation_query, positive)[0]
        paired_scores.append((score, updated))
    before = [(_key_interval(node, observation_query, positive)[1], node.count) for node in removed]
    after = [(*_key_interval(node, observation_query, positive), node.count) for node in added]
    upper_after = [(upper, multiplicity) for _, upper, multiplicity in after]
    probability_gap = _revision_mass_gap(common, before, upper_after, paired_scores)

    delta = None
    if paired_scores:
        reference = common + [(lower, multiplicity) for lower, _, multiplicity in after]
        reference.extend((score, 1) for _, score in paired_scores)
        offset = max(score for score, _ in reference)
        weights = np.asarray([exp(score - offset) for _, score in paired_scores])
        # Every paired term uses the same lower reference normalizer.
        delta = (weights / _shifted_mass(reference, offset)) @ paired_deltas
    return probability_gap, delta


class _BoundComparison:
    def __init__(self, opening_root, reference):
        self.reference = reference
        root = reference._root
        self.groups = _revision_groups(opening_root, root) if opening_root is not root else None
        paired = () if self.groups is None else self.groups[-1]
        self.intervals = 0 if self.groups is None else sum(map(len, self.groups))
        self.intervals += sum(reference is not None for _, reference, _ in paired)
        self.paired_deltas = np.asarray([delta for _, _, delta in paired]) if paired else None
        self.bounds = []
        self.history_error = 0.
        self.observation_logit_gain = None

    def record(self, error):
        self.bounds.append(error)
        self.history_error = max(self.history_error, error)


def _frontier(root, query, limit, tolerance, previous, recorded, retain_scores):
    if root is None:
        return (), 0, 0
    positive_query = query >= 0
    pending, settled = [], []
    ordinal = count()
    scored = reused = 0

    def insert(node):
        nonlocal scored, reused
        node = _scorable(node)
        score = None if previous is None else previous.get(node)
        if score is None:
            score = _score(node, query, positive_query)
            scored += 1
        else:
            reused += 1
        if recorded is not None and len(recorded) < retain_scores:
            recorded[node] = score
        if score.radius > tolerance:
            priority = score.upper_logit + log(score.node.count) + log(score.lost_fraction)
            heappush(pending, (-priority, next(ordinal), score))
        else:
            settled.append(score)

    insert(root)
    while pending and len(pending) + len(settled) < limit:
        _, _, score = heappop(pending)
        insert(score.node.left)
        insert(score.node.right)
    return (*settled, *(score for _, _, score in pending)), scored, reused


@dataclass(frozen=True, eq=False, repr=False, init=False)
class TreeFeedbackKernel(FeedbackKernel):
    """Use a persistent binary tree and a capped softmax frontier for observations."""

    _depth: int
    _max_frontier: int
    _logit_tolerance: float

    def __init__(self, *, depth, max_frontier, logit_tolerance=0.0, **weights):
        depth, max_frontier = index(depth), index(max_frontier)
        logit_tolerance = float(logit_tolerance)
        if not 0 <= depth <= 63 or max_frontier < 1:
            raise ValueError("tree needs depth in [0, 63] and a positive frontier cap")
        if not isfinite(logit_tolerance) or logit_tolerance < 0:
            raise ValueError("tree logit tolerance must be finite and nonnegative")
        object.__setattr__(self, "_depth", depth)
        object.__setattr__(self, "_max_frontier", max_frontier)
        object.__setattr__(self, "_logit_tolerance", logit_tolerance)
        super().__init__(**weights)

    def _leaves(self, rows):
        rows = _real_array(rows)
        if rows.ndim != 2 or rows.shape[1] != self._width:
            raise ValueError("observation matrix has incorrect width")
        if len(rows) > 1 << self._depth:
            raise ValueError("observations exceed tree slot capacity")
        if np.any(np.abs(rows) > 1):
            raise ValueError("tree observation coordinates must be within [-1, 1]")
        with np.errstate(over="raise", invalid="raise"):
            keys = rows @ self._observation_key
        if not np.isfinite(keys).all():
            raise FloatingPointError("nonfinite tree observation key")
        for key, row in zip(keys, rows, strict=True):
            key = _frozen(key)
            yield _Node(1, key, key, _frozen(row))

    def prepare(self, observations):
        """Encode sequential rows into slots, leaving unused slots available."""
        leaves = dict(enumerate(self._leaves(observations)))
        for _ in range(self._depth):
            parents = {slot // 2 for slot in leaves}
            leaves = {slot: _join(leaves.get(2 * slot), leaves.get(2 * slot + 1))
                      for slot in parents}
        return TreeMemory(self._observation_key, self._depth, leaves.get(0))

    def _check_memory(self, memory):
        if (not isinstance(memory, TreeMemory) or memory._depth != self._depth
                or memory._key_map is not self._observation_key):
            raise ValueError("memory belongs to a different tree feedback layout")

    def with_frontier(self, max_frontier, *, logit_tolerance=None):
        """Change the resolution policy while sharing weights and observations."""
        max_frontier = index(max_frontier)
        tolerance = self._logit_tolerance if logit_tolerance is None else float(logit_tolerance)
        if max_frontier < 1:
            raise ValueError("tree frontier cap must be positive")
        if not isfinite(tolerance) or tolerance < 0:
            raise ValueError("tree logit tolerance must be finite and nonnegative")
        if (max_frontier, tolerance) == (self._max_frontier, self._logit_tolerance):
            return self
        kernel = copy(self)
        object.__setattr__(kernel, "_max_frontier", max_frontier)
        object.__setattr__(kernel, "_logit_tolerance", tolerance)
        return kernel

    def revise(self, memory, changes):
        """Replace, insert, or remove named slots without rebuilding other subtrees."""
        self._check_memory(memory)
        if not changes:
            return memory
        edits = {}
        for slot, row in changes.items():
            slot = index(slot)
            if not 0 <= slot < 1 << self._depth:
                raise ValueError("observation slot outside tree layout")
            if slot in edits:
                raise ValueError("duplicate normalized tree slot")
            edits[slot] = None if row is None else next(self._leaves([row]))
        root = _set_slots(memory._root, self._depth, tuple(edits.items()))
        return replace(memory, _root=root)

    def start(self, seed, memory=None):
        memory = self._empty if memory is None else memory
        self._check_memory(memory)
        return TreeFeedbackState(self, memory, self._vector(seed))

    def _revision_value_gap(self, hidden, delta):
        if delta is None or not delta.any():
            return 0.
        change = _apply_value(self._observation_value, delta)
        for axis, observed, output, threshold in self._conditional:
            projection = delta @ observed
            if projection:
                gate = hidden @ axis
                if threshold is not None:
                    gate = max(0., gate - threshold)
                change += (gate * projection) * output
        gap = float(np.abs(change).max())
        if not isfinite(gap):
            raise FloatingPointError("nonfinite paired observation value change")
        return gap

    def run_with_error_bounds(self, seed, memory=None, *, steps, reference_memory=None):
        """Bound dense-replay error, optionally for a revision of the opening bank."""
        return self._run_with_error_bounds(seed, memory, steps=steps, reference_memory=reference_memory)

    def _run_with_error_bounds(self, seed, memory, *, steps, reference_memory, opening=None):
        return self._run_many_with_error_bounds(
            seed, memory, steps=steps, references=(reference_memory,), opening=opening,
        )[0]

    def _run_many_with_error_bounds(self, seed, memory, *, steps, references, opening=None):
        if type(self) is not TreeFeedbackKernel or any(
                bank is not None and type(bank) is not TreeMemory for bank in (memory, *references)):
            raise TypeError("error bounds require the standard tree feedback transition and memory")
        if any(type(value) not in (np.ndarray, FactoredValue)
               for value in (self._observation_value, self._feedback_value)):
            raise TypeError("error bounds require standard value maps")
        steps = index(steps)
        if not 0 <= steps <= self._max_steps:
            raise ValueError("bounded run steps exceed the feedback budget")
        state = self.start(seed, memory)
        if opening is not None and not opening.matches(self, state._memory, state._seed.tobytes(), steps):
            raise ValueError("retained trajectory does not match the opening run")
        references = tuple(state._memory if bank is None else bank for bank in references)
        for bank in references:
            self._check_memory(bank)
        if not steps or not references:
            return tuple(TreeFeedbackBoundedRun(state, (), bank) for bank in references)

        opening_root = state._memory._root
        comparisons = tuple(_BoundComparison(opening_root, bank) for bank in references)
        has_edits = any(comparison.groups is not None for comparison in comparisons)
        observation_query = self._query if self._observation_query is None else self._observation_query
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            gains_ready = False
            observation_query_gain = None
            for step in range(steps):
                query = None
                if opening is not None:
                    hidden = state._seed if not step else opening.final._history[step - 1]
                    opening_gap = opening.steps[step][0].attention.joint_l1_bound
                    if has_edits:
                        query = hidden @ self._query
                        projected = self._query_observations(hidden, query)
                        history_logits = self._history_logits(opening.final._keys[:step], query)
                else:
                    hidden = state.latest
                    if has_edits:
                        diagnostics = {}
                        state, _, _ = self._advance(state, diagnostics=diagnostics)
                        projected = (hidden @ observation_query if opening_root is None
                                     else diagnostics["observation_query"])
                        history_logits = diagnostics["logits"][state.attention.groups:]
                    else:
                        state = self.step(state)
                    opening_gap = state.attention.joint_l1_bound

                for comparison in comparisons:
                    local_gap, value_edit_gap = opening_gap, 0.
                    if comparison.groups is not None:
                        revision_gap, delta = _revision_attention_change(
                            comparison.groups, projected, history_logits, comparison.paired_deltas,
                        )
                        local_gap += revision_gap
                        value_edit_gap = self._revision_value_gap(hidden, delta)
                    latest_error = comparison.bounds[-1] if step else 0.
                    history_error = comparison.history_error
                    if local_gap == latest_error == history_error == 0.:
                        comparison.record(2. * tanh(value_edit_gap / 2.))
                        continue

                    if not gains_ready:
                        observation_value_gain = _value_linf(self._observation_value)
                        carry_gain = 0. if self._carry is None else _column_l1(self._carry).max()
                        feedback_keys = _column_l1(self._feedback_key) if steps > 1 else None
                        feedback_query_gain = 0. if feedback_keys is None else feedback_keys @ _column_l1(self._query)
                        feedback_value_gain = 0. if feedback_keys is None else _value_linf(self._feedback_value)
                        conditional = tuple(
                            (axis, threshold, np.abs(observed).sum() * np.abs(output).max(), np.abs(axis).sum())
                            for axis, observed, output, threshold in self._conditional
                        )
                        gains_ready = True

                    if comparison.observation_logit_gain is None:
                        root = comparison.reference._root
                        comparison.observation_logit_gain = 0.
                        if root is not None:
                            if observation_query_gain is None:
                                observation_query_gain = _column_l1(observation_query)
                            observation_keys = np.maximum(np.abs(root.lower_key), np.abs(root.upper_key))
                            comparison.observation_logit_gain = observation_keys @ observation_query_gain
                    logit_gap = latest_error * comparison.observation_logit_gain
                    value_gain = observation_value_gain
                    value_drift = latest_error * carry_gain
                    if step:
                        query = hidden @ self._query if query is None else query
                        feedback_logit_gap = latest_error * feedback_query_gain
                        feedback_logit_gap += history_error * (feedback_keys @ np.abs(query))
                        logit_gap = max(logit_gap, feedback_logit_gap)
                        value_drift += history_error * feedback_value_gain
                    for axis, threshold, amplitude, slope in conditional:
                        gate = hidden @ axis
                        gate_size = abs(gate) if threshold is None else max(0., gate - threshold)
                        value_gain += gate_size * amplitude
                        value_drift += latest_error * slope * amplitude
                    if step:
                        value_gain = max(value_gain, feedback_value_gain)

                    probability_gap = min(2., local_gap + 2. * tanh(logit_gap / 2.))
                    preactivation_gap = probability_gap * value_gain + value_drift + value_edit_gap
                    comparison.record(2. * tanh(preactivation_gap / 2.))

        reused = 0
        if opening is not None:
            retained = opening.final
            attention = replace(opening.steps[steps - 1][0].attention, scored_nodes=0,
                                reused_route=True, reused_node_scores=0)
            history_scores = steps * (steps - 1) // 2 if has_edits else 0
            state = replace(state, _history=retained._history[:steps], _keys=retained._keys[:steps],
                            _values=retained._values[:steps], observation_scores=0,
                            feedback_scores=history_scores, attention=attention)
            reused = steps
        return tuple(TreeFeedbackBoundedRun(state, tuple(comparison.bounds), comparison.reference,
                                            steps * comparison.intervals, reused)
                     for comparison in comparisons)

    def _advance(self, state, *, diagnostics=None):
        self._require_can_step(state)
        route = self._route(state, diagnostics)
        return self._apply_route(state, route, diagnostics=diagnostics)

    def _require_can_step(self, state):
        self._require_state(state)
        if len(state._history) >= self._max_steps:
            raise RuntimeError("feedback step budget exhausted")

    def _route_program_path(self, memory, slot, *, missing_ok=False):
        if type(self) is not TreeFeedbackKernel:
            raise ValueError("route programming requires a supported softmax kernel")
        if self._logit_tolerance != 0.0 or self._max_frontier < memory.count:
            raise ValueError("route programming requires a full-resolution tree")
        if not 0 <= slot < 1 << self._depth:
            raise ValueError("route programming requires occupied observation slots")
        path = []
        node = memory._root
        for bit in range(self._depth - 1, -1, -1):
            if node is None:
                break
            path.append(node)
            node = node.right if slot & (1 << bit) else node.left
        if node is None:
            if missing_ok:
                return ()
            raise ValueError("route programming requires occupied observation slots")
        return (*path, node)

    def _route_program_keys(self, memory, slots):
        return tuple(self._route_program_path(memory, slot)[-1].lower_key for slot in slots)

    def _route_program_step(self, state, slots, keys):
        stepped = self.step(state)
        with np.errstate(over="raise", invalid="raise"):
            hidden = state.latest
            query = self._query_observations(hidden, hidden @ self._query)
            gap = float(keys[0] @ query) - float(keys[1] @ query)
        return stepped, gap

    def _source_mass_bank(self, memory, slot):
        leaf = self._route_program_path(memory, slot)[-1]
        keys = []
        pending = [memory._root]
        while pending:
            node = pending.pop()
            if node.left is None and node.right is None:
                if node is leaf:
                    selected = len(keys)
                keys.append(node.lower_key)
            else:
                pending.extend(child for child in (node.right, node.left) if child is not None)
        return np.asarray(keys), selected

    def _source_mass_step(self, state, slot):
        self._require_state(state)
        path = set(self._route_program_path(state._memory, slot, missing_ok=True))
        if not path:
            return self.step(state), 0.0
        diagnostics = {}
        stepped = self._advance(state, diagnostics=diagnostics)[0]
        for (score, probability) in zip(diagnostics["frontier"], diagnostics["probabilities"]):
            if score.node in path:
                return stepped, float(probability) / score.node.count
        raise RuntimeError("tree softmax frontier omitted the selected source")

    def _history_logits(self, keys, query):
        scores = np.asarray(keys).reshape(-1, self._query.shape[1]) @ query
        if not np.isfinite(scores).all():
            raise FloatingPointError("nonfinite tree feedback history score")
        return scores

    def _route(self, state, diagnostics=None, *, previous=None, retain_scores=0):
        history_count = len(state._history)
        memory = state._memory
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            hidden = state.latest
            query = hidden @ self._query
            observation_query = self._query_observations(hidden, query) if memory.count else query
            query_bytes = observation_query.tobytes() if retain_scores else None
            prior_scores = None
            if retain_scores and previous is not None and previous.query_bytes == query_bytes:
                prior_scores = previous.node_scores
            recorded = {} if retain_scores else None
            frontier, scored, reused = _frontier(
                memory._root, observation_query, self._max_frontier, self._logit_tolerance,
                prior_scores, recorded, retain_scores,
            )
            history_logits = self._history_logits(state._keys, query)
            groups = len(frontier)
            logits = np.concatenate(([score.logit for score in frontier], history_logits))
            mixture = np.zeros(self._width)
            history_value = None
            bound = 0.0
            if len(logits):
                multiplicity = [log(score.node.count) for score in frontier]
                with np.errstate(over="ignore"):
                    shifted = logits - logits.max()
                # Apply multiplicity after removing the common logit offset.
                shifted[:groups] += multiplicity
                probabilities = np.exp(shifted - shifted.max())
                probabilities /= probabilities.sum()
                if groups:
                    mixture = probabilities[:groups] @ np.asarray([score.node.mean_row for score in frontier])
                    upper = np.concatenate(([score.upper_logit for score in frontier], history_logits))
                    # Center upper logits separately so the width cannot erase the count.
                    with np.errstate(over="ignore"):
                        upper -= upper.max()
                    upper[:groups] += multiplicity
                    upper_weights = np.exp(upper - upper.max())
                    lost = upper_weights[:groups] @ [score.lost_fraction for score in frontier]
                    bound = float(2 * lost / upper_weights.sum())
                if history_count:
                    history_value = probabilities[groups:] @ np.asarray(state._values)
            if diagnostics is not None:
                diagnostics.update(logits=logits, joint_l1_bound=bound, frontier=frontier,
                                   observation_query=observation_query,
                                   probabilities=probabilities if len(logits) else np.empty(0))
        attention = TreeAttention(groups, scored, bound, reused_node_scores=reused)
        scores = None if recorded is None else MappingProxyType(recorded)
        return _TreeRoute(mixture, history_value, attention, query_bytes, scores)

    def _apply_route(self, state, route, *, reused=False, diagnostics=None):
        mixture = route.mixture
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            preactivation, fixed = self._preactivation(
                state.latest, mixture, route.history_value, state._memory.count,
            )
            next_hidden = _frozen(np.tanh(preactivation))
            if diagnostics is not None:
                diagnostics.update(preactivation=preactivation)
        key, value = self._project_generated(next_hidden)
        attention = (replace(route.attention, scored_nodes=0, reused_route=True, reused_node_scores=0)
                     if reused else route.attention)
        return replace(
            state, _history=state._history + (next_hidden,), _keys=state._keys + (key,),
            _values=state._values + (value,), observation_scores=state.observation_scores + attention.scored_nodes,
            feedback_scores=state.feedback_scores + (0 if reused else len(state._history)),
            attention=attention,
        ), mixture, fixed

    def _shares_routing(self, other):
        return (
            type(self) is type(other) is TreeFeedbackKernel
            and type(self._feedback_value) in (np.ndarray, FactoredValue)
            and self._query is other._query and self._observation_key is other._observation_key
            and self._observation_query is other._observation_query
            and self._feedback_key is other._feedback_key and self._feedback_value is other._feedback_value
            and (self._depth, self._max_frontier, self._logit_tolerance)
            == (other._depth, other._max_frontier, other._logit_tolerance)
        )

    def _replay_routes(self, seed, memory, steps, previous=None, *, retain_scores=0):
        state = self.start(seed, memory)
        seed_bytes = state._seed.tobytes()
        reusable = ()
        if (previous is not None and previous.memory is state._memory
                and previous.seed == seed_bytes and self._shares_routing(previous.kernel)):
            reusable = previous.steps
        prior_steps = ()
        if (retain_scores and previous is not None and type(self) is type(previous.kernel) is TreeFeedbackKernel
                and self._observation_key is previous.kernel._observation_key):
            prior_steps = previous.steps
        trace = []
        for step in range(steps):
            self._require_can_step(state)
            hit = step < len(reusable)
            if hit:
                route = reusable[step][0]
            else:
                old_route = prior_steps[step][0] if step < len(prior_steps) else None
                if retain_scores:
                    route = self._route(state, previous=old_route, retain_scores=retain_scores).retain()
                else:
                    route = self._route(state).retain()
            state, _, _ = self._apply_route(state, route, reused=hit)
            hidden = state._history[-1]
            if hit and hidden.tobytes() != reusable[step][1].tobytes():
                reusable = ()
            trace.append((route, hidden))
        return state, _TreeReplay(self, state._memory, seed_bytes, tuple(trace), state,
                                  tuple(sorted(np.geterr().items())))
