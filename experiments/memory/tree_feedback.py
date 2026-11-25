"""Continuous softmax feedback with revisable summaries of observation slots."""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from itertools import count
from math import expm1, isfinite, log
from operator import index
from types import MappingProxyType

import numpy as np

from experiments.memory.latent_feedback import (
    FactoredValue, FeedbackKernel, FeedbackState, _frozen, _real_array,
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


def _set_slot(node, depth, slot, leaf):
    if depth == 0:
        if node is None and leaf is None:
            raise KeyError(slot)
        return leaf
    left = None if node is None else node.left
    right = None if node is None else node.right
    if slot & (1 << (depth - 1)):
        right = _set_slot(right, depth - 1, slot, leaf)
    else:
        left = _set_slot(left, depth - 1, slot, leaf)
    return _join(left, right)


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


def _score(node, query, positive_query):
    if node.count == 1:
        lower = upper = float(node.lower_key @ query)
    else:
        lower = float(np.where(positive_query, node.lower_key, node.upper_key) @ query)
        upper = float(np.where(positive_query, node.upper_key, node.lower_key) @ query)
    if not isfinite(lower) or not isfinite(upper) or upper < lower:
        raise FloatingPointError("unresolved tree attention key interval")
    logit = lower if lower == upper else lower * 0.5 + upper * 0.5
    radius = upper * 0.5 - lower * 0.5
    return _Score(node, logit, upper, -expm1(lower - upper), radius)


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
        root = memory._root
        for slot, row in changes.items():
            slot = index(slot)
            if not 0 <= slot < 1 << self._depth:
                raise ValueError("observation slot outside tree layout")
            leaf = None if row is None else next(self._leaves([row]))
            root = _set_slot(root, self._depth, slot, leaf)
        return replace(memory, _root=root)

    def start(self, seed, memory=None):
        memory = self._empty if memory is None else memory
        self._check_memory(memory)
        return TreeFeedbackState(self, memory, self._vector(seed))

    def _advance(self, state, *, diagnostics=None):
        self._require_can_step(state)
        route = self._route(state, diagnostics)
        return self._apply_route(state, route, diagnostics=diagnostics)

    def _require_can_step(self, state):
        self._require_state(state)
        if len(state._history) >= self._max_steps:
            raise RuntimeError("feedback step budget exhausted")

    def _route(self, state, diagnostics=None, *, previous=None, retain_scores=0):
        history_count = len(state._history)
        memory = state._memory
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            hidden = state.latest
            query = hidden @ self._query
            query_bytes = query.tobytes() if retain_scores else None
            prior_scores = None
            if retain_scores and previous is not None and previous.query_bytes == query_bytes:
                prior_scores = previous.node_scores
            recorded = {} if retain_scores else None
            frontier, scored, reused = _frontier(
                memory._root, query, self._max_frontier, self._logit_tolerance,
                prior_scores, recorded, retain_scores,
            )
            history_keys = np.asarray(state._keys).reshape(-1, self._query.shape[1])
            history_logits = history_keys @ query
            if not np.isfinite(history_logits).all():
                raise FloatingPointError("nonfinite tree feedback history score")
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
                diagnostics.update(logits=logits, joint_l1_bound=bound)
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
        return state, _TreeReplay(self, state._memory, seed_bytes, tuple(trace))
