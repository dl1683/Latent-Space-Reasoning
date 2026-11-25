"""Optional pinned binding reads inside a continuous-feedback trajectory."""

from collections import deque
from copy import copy
from dataclasses import InitVar, dataclass, replace
from math import isfinite
from operator import index

import numpy as np

from experiments.memory.binding_memory import BindingCapacityError, _ProofWork
from experiments.memory.corpus_bindings import CorpusBindingMemory, CorpusBindingRead, _DisjointBranchReads
from experiments.memory.latent_feedback import ConditionalTerm, FeedbackKernel, FeedbackState, _frozen, _real_array


_STATE_CREATION_KEY = object()


@dataclass(frozen=True)
class BindingStepRead:
    status: str
    address: tuple[str, int] | None = None
    bounds: CorpusBindingRead | None = None
    tried: tuple["BindingStepRead", ...] = ()
    proof_work: tuple[int, ...] = ()
    score: float | None = None
    expansion_work: int = 0


@dataclass(frozen=True, eq=False)
class BindingRouteExpansion:
    """Caller-authorized nodes and factorized scores for additional addresses."""

    nodes: tuple[str, ...]
    anchor_query: np.ndarray
    node_query: np.ndarray
    work_limit: int
    projection: str = "sets"

    def __post_init__(self):
        nodes = tuple(self.nodes) if not isinstance(self.nodes, str) else ()
        if not nodes or any(type(node) is not str or not node for node in nodes) or len(set(nodes)) != len(nodes):
            raise ValueError("expansion requires distinct named binding nodes")
        limit = index(self.work_limit)
        if limit < 1:
            raise ValueError("expansion work limit must be positive")
        if type(self.projection) is not str or self.projection not in ("sets", "pointwise"):
            raise ValueError("expansion projection must be sets or pointwise")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "work_limit", limit)
        object.__setattr__(self, "anchor_query", _frozen(_real_array(self.anchor_query)))
        object.__setattr__(self, "node_query", _frozen(_real_array(self.node_query)))


@dataclass(frozen=True, eq=False)
class BindingFeedbackState:
    _receiver: object
    _feedback: FeedbackState
    _memory: CorpusBindingMemory | None
    _creation_key: InitVar[object | None] = None
    _reads: tuple[BindingStepRead, ...] = ()

    def __post_init__(self, _creation_key):
        if _creation_key is not _STATE_CREATION_KEY:
            raise TypeError("construct binding feedback states through start and step")

    @property
    def feedback(self):
        return self._feedback

    @property
    def memory(self):
        return self._memory

    @property
    def reads(self):
        return self._reads

    @property
    def latest(self):
        return self.feedback.latest


def _frame_contract(memory):
    return memory.encoder_version, memory._certain._predicates, memory._min_similarity, memory._gap


@dataclass(frozen=True)
class BindingContinuation:
    prefix: BindingFeedbackState
    terminal: BindingStepRead


@dataclass(frozen=True)
class BindingSearchResult:
    best: BindingContinuation | None
    possible_strength: float | None
    unresolved_strength: float | None
    attempted_routes: int
    transition_attempts: int
    ranked_states: int
    proof_work: tuple[int, ...]


class BindingFeedback:
    """Commit at most one corpus address before generating the next hidden state."""

    def __init__(self, kernel, frame, addresses, *, query, observation,
                 route_floor=0.0, route_gap=0.0, work_limit=65536, disjoint_reads=False, route_trials=1,
                 reuse_matching_failures=False, reuse_prefix_matching=False, reuse_weighted_matching=False,
                 expansion=None, coordinate_value=None):
        if type(kernel) is not FeedbackKernel or type(frame) is not CorpusBindingMemory:
            raise ValueError("binding feedback requires a dense feedback kernel and a corpus frame")
        addresses = tuple((node, index(binding)) for node, binding in addresses)
        if not addresses or any(type(node) is not str or not node or not 0 <= binding < frame._certain.width
                                for node, binding in addresses) or len(set(addresses)) != len(addresses):
            raise ValueError("binding feedback addresses must be distinct and inside the frame")
        query, observation = _real_array(query), _real_array(observation)
        if query.shape != (kernel._width, len(addresses)) or observation.shape != (6, kernel._width):
            raise ValueError("binding feedback map has incorrect shape")
        work_limit, route_floor, route_gap = index(work_limit), float(route_floor), float(route_gap)
        if work_limit < 1 or not isfinite(route_floor) or not isfinite(route_gap) or route_gap < 0:
            raise ValueError("binding feedback requires a positive work limit, finite floor and nonnegative finite gap")
        if type(disjoint_reads) is not bool:
            raise ValueError("disjoint_reads must be a boolean")
        route_trials = index(route_trials)
        if route_trials < 1 or (route_trials > 1 and not disjoint_reads):
            raise ValueError("multiple route trials require disjoint reads")
        if type(reuse_matching_failures) is not bool or (reuse_matching_failures and route_trials == 1):
            raise ValueError("matching failure reuse requires multiple route trials")
        if type(reuse_prefix_matching) is not bool or (reuse_prefix_matching and route_trials == 1):
            raise ValueError("prefix matching reuse requires multiple route trials")
        if type(reuse_weighted_matching) is not bool or (reuse_weighted_matching and route_trials == 1):
            raise ValueError("weighted matching reuse requires multiple route trials")
        if sum((reuse_matching_failures, reuse_prefix_matching, reuse_weighted_matching)) > 1:
            raise ValueError("select one matching reuse method")
        if expansion is not None and (
            type(expansion) is not BindingRouteExpansion or route_trials == 1
            or expansion.anchor_query.shape != (kernel._width, frame._anchors.shape[1])
            or expansion.node_query.shape != (kernel._width, len(expansion.nodes))
        ):
            raise ValueError("route expansion requires multiple trials and matching score shapes")
        self._kernel = kernel
        self._contract, self._anchors = _frame_contract(frame), frame._anchors
        self._addresses = addresses
        self._query, self._observation = _frozen(query), _frozen(observation)
        self._coordinate_value = (None if coordinate_value is None else
                                  kernel._weight(coordinate_value, (frame._anchors.shape[1], kernel._width)))
        self._coordinate_terms = ()
        self._route_floor, self._route_gap, self._work_limit = route_floor, route_gap, work_limit
        self._disjoint_reads = disjoint_reads
        self._route_trials = route_trials
        self._reuse_matching_failures = reuse_matching_failures
        self._reuse_prefix_matching = reuse_prefix_matching
        self._reuse_weighted_matching = reuse_weighted_matching
        self._expansion = expansion
        self._fixed_addresses = frozenset(addresses) if expansion is not None else None

    def start(self, seed, memory=None):
        """Pin an optional compiled corpus snapshot for this trajectory."""
        if memory is not None and (
            type(memory) is not CorpusBindingMemory or _frame_contract(memory) != self._contract
            or (memory._anchors is not self._anchors and not np.array_equal(memory._anchors, self._anchors))
        ):
            raise ValueError("memory belongs to a different binding frame")
        return BindingFeedbackState(self, self._kernel.start(seed), memory, _STATE_CREATION_KEY)

    def _intervene_latest(self, state, hidden):
        """Replace the generated row and its feedback projections, leaving the proof trace pinned."""
        if type(state) is not BindingFeedbackState or state._receiver is not self or not state.reads:
            raise ValueError("a binding intervention needs a generated state from this receiver")
        self._kernel._require_state(state.feedback)
        row = self._kernel._vector(hidden)
        key, value = self._kernel._project_generated(row)
        feedback = state.feedback
        intervened = replace(feedback, _history=(*feedback._history[:-1], row),
                             _keys=(*feedback._keys[:-1], key), _values=(*feedback._values[:-1], value))
        return replace(state, _feedback=intervened, _creation_key=_STATE_CREATION_KEY)

    def _with_route_column(self, column, weights):
        candidate = copy(self)
        query = self._query.copy()
        query[:, column] = self._kernel._vector(weights)
        candidate._query = _frozen(query)
        return candidate

    def _with_coordinate_value(self, weight):
        candidate = copy(self)
        shape = (self._anchors.shape[1], self._kernel._width)
        candidate._coordinate_value = None if weight is None else self._kernel._weight(weight, shape)
        return candidate

    def _with_coordinate_term(self, term, *, position=None):
        if type(term) is not ConditionalTerm or term.threshold is None:
            raise ValueError("coordinate context needs a hinged conditional term")
        anchor = _frozen(_real_array(term.observation_axis))
        if anchor.shape != (self._anchors.shape[1],):
            raise ValueError("coordinate context anchor axis has the wrong width")
        state = self._kernel._vector(term.state_axis)
        output = self._kernel._vector(term.output)
        if position is None:
            position = len(self._coordinate_terms)
        if not 0 <= position <= len(self._coordinate_terms):
            raise ValueError("context insertion position is outside the receiver")
        candidate = copy(self)
        candidate._coordinate_terms = (*self._coordinate_terms[:position], (state, anchor, output, term.threshold),
                                        *self._coordinate_terms[position:])
        return candidate

    def _without_coordinate_term(self, position):
        if not 0 <= position < len(self._coordinate_terms):
            raise ValueError("context retirement position is outside the receiver")
        candidate = copy(self)
        candidate._coordinate_terms = self._coordinate_terms[:position] + self._coordinate_terms[position + 1:]
        return candidate

    def _read_with_route_map(self, state):
        """Evaluate an earlier route map on a ready state from the same kernel."""
        self._kernel._require_state(state.feedback)
        return self._next_read(replace(state, _receiver=self, _creation_key=_STATE_CREATION_KEY))

    def _next_read(self, state):
        if type(state) is not BindingFeedbackState or state._receiver is not self:
            raise ValueError("state belongs to a different binding receiver")
        self._kernel._require_state(state.feedback)
        if len(state.feedback._history) >= self._kernel._max_steps:
            raise RuntimeError("feedback step budget exhausted")
        if self._disjoint_reads and state.memory is not None and state.reads:
            prior = state.reads[-1]
            if prior.status == "resolved":
                prefix = tuple(read.address for read in state.reads)
                return self._read(state.memory, state.latest, prefix=prefix,
                                  prefix_absent=prior.bounds.possible_strength is None)
            if prior.status == "catalog_absent":
                return BindingStepRead("catalog_absent", bounds=prior.bounds)
            return BindingStepRead("incomplete")
        return self._read(state.memory, state.latest)

    def step(self, state):
        """Read the chosen bounds, then perform the same transition with or without memory."""
        return self._advance_read(state, self._next_read(state))

    def _advance_read(self, state, read, *, diagnostics=None):
        extra = None
        if state.memory is not None:
            features = np.zeros(6)
            features[0] = 1.0
            upper = None
            if read.bounds is not None:
                lower, upper = read.bounds.certain, read.bounds.possible_strength
                features[1] = 1.0
                if lower is not None:
                    features[2:4] = 1.0, lower.strength
                if upper is not None:
                    features[4:6] = 1.0, upper
            with np.errstate(over="raise", invalid="raise"):
                extra = features @ self._observation
                if read.status == "resolved" and upper is not None:
                    anchor = self._anchors[read.address[1]]
                    if self._coordinate_value is not None:
                        extra += anchor @ self._coordinate_value
                    for state_axis, anchor_axis, output, threshold in self._coordinate_terms:
                        amount = max(0., float(state.latest @ state_axis) - threshold)
                        if amount:
                            extra += (amount * (anchor @ anchor_axis)) * output
        feedback, _, _ = self._kernel._advance(state.feedback, observation_extra=extra, diagnostics=diagnostics)
        return BindingFeedbackState(self, feedback, state.memory, _STATE_CREATION_KEY, state.reads + (read,))

    def search_to(self, state, target, *, steps, route_limit):
        """Find a certain continuation to a fixed terminal address on this corpus pin."""
        if type(state) is not BindingFeedbackState or state._receiver is not self:
            raise ValueError("state belongs to a different binding receiver")
        memory = state.memory
        if memory is None or not self._disjoint_reads or self._expansion is not None:
            raise ValueError("continuation search requires pinned disjoint reads and a fixed catalog")
        steps, route_limit = index(steps), index(route_limit)
        if not 1 <= steps <= self._kernel._max_steps - len(state.reads) or route_limit < 1:
            raise ValueError("continuation search requires a positive route limit and steps within the horizon")
        if any(read.status != "resolved" for read in state.reads):
            raise ValueError("continuation search requires a resolved prefix")
        node, coordinate = target
        target = self._addresses.index((node, index(coordinate)))
        bound = state.reads[-1].bounds.possible_strength if state.reads else 1.0
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            return self._search_continuations(state, target, steps, route_limit, bound)

    def _search_continuations(self, state, target, steps, route_limit, bound):
        memory = state.memory
        certain = _ProofWork(self._work_limit, reuse_rows=True)
        shared_bank = memory._certain is memory._possible
        possible = certain if shared_bank else _ProofWork(self._work_limit, reuse_rows=True)
        best, best_grade, upper, unresolved = None, -1., -1., -1.
        ranked, attempted, transitions = 0, 0, 0
        stack = []

        def push(prefix, remaining, grade):
            nonlocal ranked, unresolved
            # Another disjoint premise can only lower this prefix's possible grade.
            if grade is None or grade <= best_grade:
                return
            ranked += 1
            try:
                choices = self._continuation_routes(prefix.latest, target, terminal=remaining == 1)
            except FloatingPointError:
                unresolved = max(unresolved, grade)
                return
            if choices:
                stack.append((prefix, remaining, grade, choices))

        push(state, steps, bound)
        while stack:
            prefix, remaining, grade, choices = stack[-1]
            if grade <= best_grade or not choices:
                stack.pop()
                continue
            if attempted == route_limit:
                unresolved = max(unresolved, max(entry[2] for entry in stack if entry[3]))
                break
            address, score = choices.popleft()
            attempted += 1
            if address[0] not in memory._certain._dependents:
                unresolved = max(unresolved, grade)
                continue
            try:
                addresses = tuple(read.address for read in prefix.reads) + (address,)
                roots = memory._certain._joint_keys(addresses, certain.limit)
                bounds = memory._disjoint_bounds(roots, frozenset(), certain, possible)
            except BindingCapacityError:
                unresolved = max(unresolved, grade)
                continue
            if bounds.possible_strength is None:
                continue
            read = BindingStepRead("resolved", address, bounds, score=score)
            if remaining == 1:
                upper = max(upper, bounds.possible_strength)
                if bounds.certain is not None and bounds.certain.strength > best_grade:
                    best, best_grade = BindingContinuation(prefix, read), bounds.certain.strength
            elif bounds.possible_strength > best_grade:
                transitions += 1
                try:
                    child = self._advance_read(prefix, read)
                except FloatingPointError:
                    unresolved = max(unresolved, bounds.possible_strength)
                    continue
                push(child, remaining - 1, bounds.possible_strength)

        upper = max(upper, unresolved)
        work = (certain.used,) if shared_bank else (certain.used, possible.used)
        return BindingSearchResult(best, upper if upper >= 0 else None, unresolved if unresolved >= 0 else None,
                                   attempted, transitions, ranked, work)

    def _continuation_routes(self, hidden, target, *, terminal):
        if terminal:
            score = float(hidden @ self._query[:, target])
            return deque([(self._addresses[target], score)]) if score > self._route_floor else deque()
        addresses = self._addresses[:target] + self._addresses[target + 1:]
        scores = np.concatenate((hidden @ self._query[:, :target], hidden @ self._query[:, target + 1:]))
        return deque((addresses[i], float(scores[i])) for i in np.argsort(-scores, kind="stable")
                     if scores[i] > self._route_floor)

    def _read(self, memory, hidden, *, prefix=(), prefix_absent=False):
        if memory is None:
            return BindingStepRead("detached")
        with np.errstate(over="raise", invalid="raise"):
            scores = hidden @ self._query
        best = int(np.argmax(scores))
        winning = float(scores[best])
        if prefix and not prefix_absent and self._route_trials > 1:
            if winning > self._route_floor or self._expansion is not None:
                return self._read_branches(memory, scores, prefix, hidden)
        if winning <= self._route_floor:
            return BindingStepRead("below_floor")
        if len(scores) > 1 and winning - float(np.partition(scores, -2)[-2]) <= self._route_gap:
            return BindingStepRead("ambiguous")
        address = self._addresses[best]
        if address[0] not in memory._certain._dependents:
            return BindingStepRead("unavailable", address)
        if prefix_absent:
            # Appending a root cannot make an infeasible conjunction feasible on this pin.
            return BindingStepRead("resolved", address, CorpusBindingRead(None, None))
        try:
            if prefix:
                bounds = memory.resolve_disjoint((*prefix, address), work_limit=self._work_limit)
            else:
                bounds = memory.resolve(*address, work_limit=self._work_limit)
        except BindingCapacityError:
            return BindingStepRead("capacity", address)
        return BindingStepRead("resolved", address, bounds)

    def _read_branches(self, memory, scores, prefix, hidden):
        session = _DisjointBranchReads(memory, prefix, self._work_limit,
                                       reuse_matching_failures=self._reuse_matching_failures,
                                       reuse_prefix_matching=self._reuse_prefix_matching,
                                       reuse_weighted_matching=self._reuse_weighted_matching)
        tried = []
        result = self._try_ranked(memory, session, self._addresses, scores, tried)
        expansion_work = 0
        if result is None and self._expansion is not None:
            result, expansion_work = self._expand_branches(memory, session, tried, hidden)
        if result is None:
            result = (BindingStepRead("catalog_absent", bounds=tried[0].bounds) if tried
                      else BindingStepRead("below_floor"))
        return replace(result, tried=tuple(tried), proof_work=session.proof_work, expansion_work=expansion_work)

    def _try_ranked(self, memory, session, addresses, scores, tried):
        ranked = np.argsort(-scores, kind="stable")
        for position, candidate in enumerate(ranked):
            score = float(scores[candidate])
            if score <= self._route_floor:
                break
            if position + 1 < len(ranked) and score - float(scores[ranked[position + 1]]) <= self._route_gap:
                return BindingStepRead("ambiguous")
            if len(tried) == self._route_trials:
                return BindingStepRead("candidate_limit")
            address = addresses[candidate]
            before = session.proof_work
            attempt = self._read_branch(memory, session, address)
            work = tuple(used - prior for used, prior in zip(session.proof_work, before, strict=True))
            attempt = replace(attempt, score=score, proof_work=work)
            tried.append(attempt)
            if attempt.bounds is None or attempt.bounds.possible_strength is not None:
                return attempt
        return None

    def _expand_branches(self, memory, session, tried, hidden):
        expansion = self._expansion
        spent = 0

        def charge(amount):
            nonlocal spent
            if spent + amount > expansion.work_limit:
                raise BindingCapacityError("route expansion work limit exceeded")
            spent += amount

        try:
            charge(len(expansion.nodes))
            if any(node not in memory._possible._dependents for node in expansion.nodes):
                return BindingStepRead("unavailable"), spent
            candidates, scores = self._expanded_scores(memory, hidden, charge)
        except BindingCapacityError:
            return BindingStepRead("expansion_capacity"), spent
        return self._try_ranked(memory, session, candidates, scores, tried), spent

    def _expanded_scores(self, memory, hidden, charge):
        expansion = self._expansion
        coordinates = memory._possible._possible_coordinates(
            expansion.nodes, charge, pointwise=expansion.projection == "pointwise")
        active, positions, candidates = {}, [], []
        for position, node in enumerate(expansion.nodes):
            eligible = coordinates[node]
            charge(len(eligible))
            start = len(candidates)
            for coordinate in sorted(eligible):
                address = node, coordinate
                if address not in self._fixed_addresses:
                    active.setdefault(coordinate, len(active))
                    candidates.append((address, len(positions)))
            if len(candidates) != start:
                positions.append(position)
        if not candidates:
            return (), np.empty(0)
        charge((hidden.size + len(active)) * memory._anchors.shape[1]
               + hidden.size * len(positions) + len(candidates))
        with np.errstate(over="raise", invalid="raise"):
            anchor_scores = memory._anchors[list(active)] @ (hidden @ expansion.anchor_query)
            node_scores = hidden @ expansion.node_query[:, positions]
            addresses, scores = [], []
            for address, position in candidates:
                addresses.append(address)
                scores.append(node_scores[position] + anchor_scores[active[address[1]]])
        return addresses, np.asarray(scores)

    def _read_branch(self, memory, session, address):
        if address[0] not in memory._certain._dependents:
            return BindingStepRead("unavailable", address)
        try:
            bounds = session.resolve(address)
        except BindingCapacityError:
            return BindingStepRead("capacity", address)
        return BindingStepRead("resolved", address, bounds)
