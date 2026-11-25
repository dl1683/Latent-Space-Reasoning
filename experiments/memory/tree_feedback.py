"""Continuous softmax feedback with revisable summaries of observation slots."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from itertools import count
from math import expm1, isfinite, log
from operator import index

import numpy as np

from experiments.memory.latent_feedback import FeedbackKernel, FeedbackState, _frozen, _real_array


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


@dataclass(frozen=True, eq=False)
class TreeFeedbackState(FeedbackState):
    _memory: TreeMemory
    attention: TreeAttention | None = None


@dataclass(frozen=True, eq=False)
class _Score:
    node: _Node
    logit: float
    upper_logit: float
    lost_fraction: float
    radius: float


def _score(node, query, positive_query):
    while node.count > 1 and (node.left is None or node.right is None):
        node = node.right if node.left is None else node.left
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


def _frontier(root, query, limit, tolerance):
    if root is None:
        return (), 0
    positive_query = query >= 0
    pending, settled = [], []
    ordinal = count()

    def insert(node):
        score = _score(node, query, positive_query)
        if score.radius > tolerance:
            priority = score.upper_logit + log(score.node.count) + log(score.lost_fraction)
            heappush(pending, (-priority, next(ordinal), score))
        else:
            settled.append(score)

    insert(root)
    scored = 1
    while pending and len(pending) + len(settled) < limit:
        _, _, score = heappop(pending)
        insert(score.node.left)
        insert(score.node.right)
        scored += 2
    return (*settled, *(score for _, _, score in pending)), scored


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

    def with_frontier(self, max_frontier):
        """Use a different query cap with the same weights and observation snapshots."""
        max_frontier = index(max_frontier)
        if max_frontier < 1:
            raise ValueError("tree frontier cap must be positive")
        if max_frontier == self._max_frontier:
            return self
        kernel = copy(self)
        object.__setattr__(kernel, "_max_frontier", max_frontier)
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
        self._require_state(state)
        history_count = len(state._history)
        if history_count >= self._max_steps:
            raise RuntimeError("feedback step budget exhausted")
        memory = state._memory
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            hidden = state.latest
            query = hidden @ self._query
            frontier, scored = _frontier(memory._root, query, self._max_frontier, self._logit_tolerance)
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
            preactivation, fixed = self._preactivation(hidden, mixture, history_value, memory.count)
            next_hidden = _frozen(np.tanh(preactivation))
            if diagnostics is not None:
                diagnostics.update(logits=logits, preactivation=preactivation, joint_l1_bound=bound)
        key, value = self._project_generated(next_hidden)
        return replace(
            state, _history=state._history + (next_hidden,), _keys=state._keys + (key,),
            _values=state._values + (value,), observation_scores=state.observation_scores + scored,
            feedback_scores=state.feedback_scores + history_count, attention=TreeAttention(groups, scored, bound),
        ), mixture, fixed
