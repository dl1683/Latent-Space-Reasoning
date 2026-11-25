"""Optional pinned binding reads inside a continuous-feedback trajectory."""

from dataclasses import InitVar, dataclass, replace
from math import isfinite
from operator import index

import numpy as np

from experiments.memory.binding_memory import BindingCapacityError
from experiments.memory.corpus_bindings import CorpusBindingMemory, CorpusBindingRead, _DisjointBranchReads
from experiments.memory.latent_feedback import FeedbackKernel, FeedbackState, _frozen, _real_array


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

    def __post_init__(self):
        nodes = tuple(self.nodes) if not isinstance(self.nodes, str) else ()
        if not nodes or any(type(node) is not str or not node for node in nodes) or len(set(nodes)) != len(nodes):
            raise ValueError("expansion requires distinct named binding nodes")
        limit = index(self.work_limit)
        if limit < 1:
            raise ValueError("expansion work limit must be positive")
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


class BindingFeedback:
    """Commit at most one corpus address before generating the next hidden state."""

    def __init__(self, kernel, frame, addresses, *, query, observation,
                 route_floor=0.0, route_gap=0.0, work_limit=65536, disjoint_reads=False, route_trials=1,
                 reuse_matching_failures=False, reuse_prefix_matching=False, reuse_weighted_matching=False,
                 expansion=None):
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

    def step(self, state):
        """Read the chosen bounds, then perform the same transition with or without memory."""
        if type(state) is not BindingFeedbackState or state._receiver is not self:
            raise ValueError("state belongs to a different binding receiver")
        self._kernel._require_state(state.feedback)
        if len(state.feedback._history) >= self._kernel._max_steps:
            raise RuntimeError("feedback step budget exhausted")
        if self._disjoint_reads and state.memory is not None and state.reads:
            prior = state.reads[-1]
            if prior.status == "resolved":
                prefix = tuple(read.address for read in state.reads)
                read = self._read(state.memory, state.latest, prefix=prefix,
                                  prefix_absent=prior.bounds.possible_strength is None)
            elif prior.status == "catalog_absent":
                read = BindingStepRead("catalog_absent", bounds=prior.bounds)
            else:
                read = BindingStepRead("incomplete")
        else:
            read = self._read(state.memory, state.latest)
        extra = None
        if state.memory is not None:
            features = np.zeros(6)
            features[0] = 1.0
            if read.bounds is not None:
                lower, upper = read.bounds.certain, read.bounds.possible_strength
                features[1] = 1.0
                if lower is not None:
                    features[2:4] = 1.0, lower.strength
                if upper is not None:
                    features[4:6] = 1.0, upper
            with np.errstate(over="raise", invalid="raise"):
                extra = features @ self._observation
        feedback, _, _ = self._kernel._advance(state.feedback, observation_extra=extra)
        return BindingFeedbackState(self, feedback, state.memory, _STATE_CREATION_KEY, state.reads + (read,))

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
            candidates, expanded_scores, expansion_work = self._expanded_scores(memory, hidden)
            if candidates is None:
                result = BindingStepRead("expansion_capacity")
            else:
                result = self._try_ranked(memory, session, candidates, expanded_scores, tried)
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

    def _expanded_scores(self, memory, hidden):
        expansion = self._expansion
        rows = memory._possible._rows
        scanned = len(rows)
        if scanned > expansion.work_limit:
            return None, None, 0
        active = sorted({coordinate for _, coordinate in rows})
        if not active:
            return (), np.empty(0), scanned
        # Count row visits, scalar dot terms and node/anchor score combinations.
        total = scanned + (hidden.size + len(active)) * (memory._anchors.shape[1] + len(expansion.nodes))
        if total > expansion.work_limit:
            return None, None, scanned
        with np.errstate(over="raise", invalid="raise"):
            anchor_scores = memory._anchors[active] @ (hidden @ expansion.anchor_query)
            node_scores = hidden @ expansion.node_query
            combined = node_scores[:, None] + anchor_scores
        candidates, scores = [], []
        for node, row in zip(expansion.nodes, combined, strict=True):
            for coordinate, score in zip(active, row, strict=True):
                address = node, coordinate
                if address not in self._fixed_addresses:
                    candidates.append(address)
                    scores.append(score)
        return candidates, np.asarray(scores), total

    def _read_branch(self, memory, session, address):
        if address[0] not in memory._certain._dependents:
            return BindingStepRead("unavailable", address)
        try:
            bounds = session.resolve(address)
        except BindingCapacityError:
            return BindingStepRead("capacity", address)
        return BindingStepRead("resolved", address, bounds)
