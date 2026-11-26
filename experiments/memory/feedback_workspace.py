"""Publish independently checked numerical operator edits on a pinned observation bank."""

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from dataclasses import dataclass, replace
from math import hypot, isfinite, sqrt
from operator import index
from threading import Lock

import numpy as np

from experiments.memory.latent_feedback import (
    ConditionalTerm, FeedbackKernel, ObservationMemory, _real_array,
)
from experiments.memory.linear_feedback import MomentMemory
from experiments.memory.tree_feedback import TreeFeedbackKernel, TreeMemory, _TreeReplay
from experiments.memory.weight_program import _frobenius


def _split_candidate_kind(kind):
    if kind.startswith("joint:"):
        return "joint", int(kind.removeprefix("joint:"))
    return kind, None


def _interior_midpoint(left, right):
    middle = left + (right - left) * 0.5
    return middle if left < middle < right else None


@dataclass(frozen=True)
class FeedbackRoutingGoal:
    slot: int
    min_mass: float

    def __post_init__(self):
        slot, floor = index(self.slot), float(self.min_mass)
        if slot < 0 or not 0 < floor < 1:
            raise ValueError("routing goal needs a nonnegative slot and a mass floor in (0, 1)")
        object.__setattr__(self, "slot", slot)
        object.__setattr__(self, "min_mass", floor)


@dataclass(frozen=True)
class FeedbackProbe:
    seed: tuple[float, ...]
    steps: int
    target: tuple[float, ...]
    routing: FeedbackRoutingGoal | None = None

    def __post_init__(self):
        seed = _real_array(self.seed)
        target = _real_array(self.target)
        steps = index(self.steps)
        if seed.ndim != 1 or not len(seed) or seed.shape != target.shape:
            raise ValueError("probe must have equally sized vector seed and target")
        if steps < 1 or np.any(np.abs(target) >= 1):
            raise ValueError("probe needs positive steps and an interior tanh target")
        if self.routing is not None and type(self.routing) is not FeedbackRoutingGoal:
            raise ValueError("invalid source routing goal")
        object.__setattr__(self, "seed", tuple(seed))
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "target", tuple(target))


@dataclass(frozen=True, eq=False)
class FeedbackRevision:
    number: int
    kernel: FeedbackKernel
    memory: ObservationMemory | MomentMemory | TreeMemory
    errors: tuple[float, ...]
    _anchors: tuple[tuple[np.ndarray, ...] | None, ...]
    guard_errors: tuple[float, ...] = ()
    _probe_routes: tuple[_TreeReplay | None, ...] = ()
    _guard_routes: tuple[_TreeReplay, ...] = ()
    source_masses: tuple[float | None, ...] = ()

    def start(self, seed):
        return self.kernel.start(seed, self.memory)


@dataclass(frozen=True, eq=False)
class FeedbackProposal:
    base: FeedbackRevision
    probe: int
    _weight: np.ndarray | None = None
    term: ConditionalTerm | None = None
    _query: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "probe", index(self.probe))
        maps = self._weight is not None or self._query is not None
        if maps == (self.term is not None):
            raise ValueError("proposal needs map edits or one conditional term")
        if self.term is not None and type(self.term) is not ConditionalTerm:
            raise ValueError("invalid conditional proposal data")
        if self._weight is not None:
            object.__setattr__(self, "_weight", self.base.kernel._weight(self._weight))
        if self._query is not None:
            kernel = self.base.kernel
            object.__setattr__(self, "_query", kernel._weight(self._query, kernel._query.shape))


class StaleFeedbackRevision(ValueError):
    """The observations or operator have advanced since this work was based."""


@dataclass(frozen=True)
class FeedbackFidelityAttempt:
    resolution: str
    frontier_cap: int
    rejection_stage: str | None = None
    rejection: str | None = None


@dataclass(frozen=True)
class FeedbackRepairAttempt:
    probe: int
    conditional: bool
    rejection: str | None = None
    rank: tuple[int, float, int] | None = None
    hinge: bool = False
    fidelity_attempts: tuple[FeedbackFidelityAttempt, ...] = ()
    routing: bool = False
    joint: bool = False
    joint_trial: int | None = None


@dataclass(frozen=True)
class FeedbackMaintenanceRound:
    status: str
    base: FeedbackRevision
    attempts: tuple[FeedbackRepairAttempt, ...] = ()
    revision: FeedbackRevision | None = None
    selected_attempt: int | None = None
    next_candidate: int | None = None


@dataclass(frozen=True)
class FeedbackSlotReadReview:
    """Estimated and optionally replayed outcome for one configured read."""

    read_index: int
    is_guard: bool
    opening_error: float
    opening_mass: float | None
    source_floor: float | None
    opening_satisfied: bool
    error_interval: tuple[float, float]
    estimated_satisfied: bool | None
    replayed_error: float | None = None
    replayed_mass: float | None = None
    replayed_satisfied: bool | None = None
    replay_failure: str | None = None


@dataclass(frozen=True, eq=False)
class FeedbackSlotAlternativeReview:
    reference_memory: TreeMemory
    reads: tuple[FeedbackSlotReadReview, ...]


@dataclass(frozen=True, eq=False)
class FeedbackSlotTriage:
    base: FeedbackRevision
    alternatives: tuple[FeedbackSlotAlternativeReview, ...]
    replay_count: int
    trajectory_replay_count: int = 0


class FeedbackWorkspace:
    """A bounded in-process commit point. Numeric replays run outside its lock."""

    def __init__(self, kernel, observations, probes, *, max_update_norm, atol=1e-8,
                 max_terms=0, guards=(), allow_hinge=False, max_reference_frontier=None,
                 reuse_tree_routes=False, max_retained_tree_scores=0, allow_joint=False,
                 joint_trials=1, separate_joint_trials=False, joint_refinements=0):
        if type(reuse_tree_routes) is not bool:
            raise ValueError("tree route reuse must be a boolean")
        self._max_retained_tree_scores = index(max_retained_tree_scores)
        if self._max_retained_tree_scores < 0:
            raise ValueError("retained tree score cap must be nonnegative")
        self._reuse_tree_routes = reuse_tree_routes or self._max_retained_tree_scores > 0
        if self._reuse_tree_routes and type(kernel) is not TreeFeedbackKernel:
            raise ValueError("route reuse requires a tree feedback kernel")
        self._draft_tree_policy = None
        if max_reference_frontier is not None:
            if not isinstance(kernel, TreeFeedbackKernel):
                raise ValueError("reference frontier requires a tree feedback kernel")
            reference_cap = index(max_reference_frontier)
            if reference_cap < kernel._max_frontier:
                raise ValueError("reference frontier cannot be smaller than the draft frontier")
            self._draft_tree_policy = kernel._max_frontier, kernel._logit_tolerance
            kernel = kernel.with_frontier(reference_cap, logit_tolerance=0)
        self._probes = tuple(FeedbackProbe(p.seed, p.steps, p.target, getattr(p, "routing", None)) for p in probes)
        self._guards = tuple(FeedbackProbe(p.seed, p.steps, p.target, getattr(p, "routing", None)) for p in guards)
        if any(p.routing is not None for p in self._guards):
            raise ValueError("routing goals are supported only on working probes")
        self._has_routing = any(p.routing is not None for p in self._probes)
        if type(allow_joint) is not bool or (allow_joint and not self._has_routing):
            raise ValueError("joint edits require a boolean policy and a working routing goal")
        self._joint_trials = index(joint_trials)
        if type(joint_trials) is bool or self._joint_trials < 1 or (self._joint_trials > 1 and not allow_joint):
            raise ValueError("joint trials must be positive and multiple trials require joint edits")
        if type(separate_joint_trials) is not bool or (separate_joint_trials and self._joint_trials == 1):
            raise ValueError("separate joint trials require a boolean policy and more than one trial")
        self._joint_refinements = index(joint_refinements)
        if (type(joint_refinements) is bool or self._joint_refinements < 0
                or (self._joint_refinements and not separate_joint_trials)):
            raise ValueError("joint refinements must be nonnegative and require separate trials when enabled")
        self._allow_joint = allow_joint
        self._separate_joint_trials = separate_joint_trials
        if self._has_routing:
            if type(kernel) not in (FeedbackKernel, TreeFeedbackKernel):
                raise ValueError("routing goals require a supported softmax kernel")
            if type(kernel) is TreeFeedbackKernel and any(
                    p.routing is not None and p.routing.slot >= 1 << kernel._depth for p in self._probes):
                raise ValueError("routing goal slot is outside the tree layout")
        self._atol = float(atol)
        self._max_update_norm = float(max_update_norm)
        if not self._probes:
            raise ValueError("at least one behavioral probe is required")
        if not isfinite(self._atol) or self._atol <= 0:
            raise ValueError("atol must be positive and finite")
        if not isfinite(self._max_update_norm) or self._max_update_norm < 0:
            raise ValueError("max_update_norm must be finite and nonnegative")
        self._max_terms = index(max_terms)
        if self._max_terms < kernel.conditional_terms:
            raise ValueError("workspace term budget is below the initial architecture")
        if type(allow_hinge) is not bool:
            raise ValueError("workspace hinge permission must be a boolean")
        if not allow_hinge and any(term[-1] is not None for term in kernel._conditional):
            raise ValueError("workspace hinge terms are disabled")
        self._allow_hinge = allow_hinge
        self._lock = Lock()
        memory = kernel.prepare(observations)
        self._head = self._measure_bank(0, kernel, memory)

    def snapshot(self):
        with self._lock:
            return self._head

    def _require_current(self, base):
        if self.snapshot() is not base:
            raise StaleFeedbackRevision("feedback revision is no longer current")

    def _check_reference_capacity(self, kernel, memory):
        if self._draft_tree_policy is not None and memory.count > kernel._max_frontier:
            raise ValueError("observation bank exceeds the full-reference frontier")
        if (self._has_routing and type(kernel) is TreeFeedbackKernel
                and (kernel._logit_tolerance != 0 or memory.count > kernel._max_frontier)):
            raise ValueError("routing goals require a full-resolution tree")

    def _commit(self, base, revised, *, proposal=None):
        with self._lock:
            if self._head is not base:
                raise StaleFeedbackRevision("feedback revision changed before publication")
            self._head = revised
            return self._head

    @staticmethod
    def _run(kernel, memory, probe):
        state = kernel.start(probe.seed, memory)
        for _ in range(probe.steps):
            state = kernel.step(state)
        return state

    @staticmethod
    def _probe_error(state, probe):
        return FeedbackWorkspace._target_error(state.latest, probe.target)

    @staticmethod
    def _target_error(hidden, target):
        error = float(np.max(np.abs(hidden - target)))
        if not isfinite(error):
            raise ValueError("probe error must be finite")
        return error

    @staticmethod
    def _run_routed(kernel, memory, probe):
        state = kernel.start(probe.seed, memory)
        for _ in range(probe.steps - 1):
            state = kernel.step(state)
        return kernel._source_mass_step(state, probe.routing.slot)

    def _measure_guards(self, kernel, memory):
        return tuple(self._probe_error(self._run(kernel, memory, guard), guard)
                     for guard in self._guards)

    def _replay_tree_checks(self, kernel, memory, probes, previous):
        previous = previous or (None,) * len(probes)
        for probe, trace in zip(probes, previous, strict=True):
            state, replay = kernel._replay_routes(
                probe.seed, memory, probe.steps, trace, retain_scores=self._max_retained_tree_scores,
            )
            yield self._probe_error(state, probe), replay

    def _working_results(self, kernel, memory, previous=()):
        errors, prefixes, traces, masses = [], [], [], []
        previous = previous or (None,) * len(self._probes)
        for probe, trace in zip(self._probes, previous, strict=True):
            mass, replay = None, None
            if probe.routing is not None:
                state, mass = self._run_routed(kernel, memory, probe)
                if not 0.0 <= mass <= 1.0:
                    raise ValueError("probe source mass must be a probability")
            elif self._reuse_tree_routes:
                state, replay = kernel._replay_routes(
                    probe.seed, memory, probe.steps, trace, retain_scores=self._max_retained_tree_scores,
                )
            else:
                state = self._run(kernel, memory, probe)
            errors.append(self._probe_error(state, probe))
            prefixes.append(state.history[:-1] if replay is None else replay.prefix())
            traces.append(replay)
            masses.append(mass)
        return tuple(errors), tuple(prefixes), tuple(traces) if self._reuse_tree_routes else (), tuple(masses)

    def _guard_results(self, kernel, memory, previous=()):
        if not self._reuse_tree_routes:
            return self._measure_guards(kernel, memory), ()
        runs = tuple(self._replay_tree_checks(kernel, memory, self._guards, previous))
        errors = tuple(error for error, _ in runs)
        return errors, tuple(trace for _, trace in runs)

    def _probe_satisfied(self, probe, error, mass):
        route = self._probes[probe].routing
        return error <= self._atol and (route is None or mass >= route.min_mass)

    def _anchors(self, errors, prefixes, previous=None):
        if previous is None:
            previous = (None,) * len(errors)
        return tuple(
            anchor if anchor is not None else prefix if error <= self._atol else None
            for error, prefix, anchor in zip(errors, prefixes, previous, strict=True)
        )

    def _open_probe(self, base, probe):
        probe = index(probe)
        if not 0 <= probe < len(self._probes):
            raise ValueError("probe index out of range")
        if self._probe_satisfied(probe, base.errors[probe], base.source_masses[probe]):
            raise ValueError("probe already satisfies the workspace tolerance")
        return probe

    def _require_term_capacity(self, base, *, hinge=False):
        if hinge and not self._allow_hinge:
            raise ValueError("workspace hinge terms are disabled")
        if base.kernel.conditional_terms >= self._max_terms:
            raise ValueError("workspace conditional term budget exhausted")

    def propose(self, base, probe, *, conditional=False, hinge=False):
        """Produce a candidate using the unresolved probe and satisfied runs."""
        self._require_current(base)
        if self._draft_tree_policy is not None:
            probe = self._open_probe(base, probe)
            proposal, _, trials = self._assess_tree_repair(base, probe, conditional or hinge, hinge)
            if proposal is None:
                raise ValueError(trials[-1].rejection)
            return proposal
        return self._propose(base, probe, conditional=conditional or hinge, hinge=hinge)

    def _propose(self, base, probe, *, conditional, hinge=False):
        probe = self._open_probe(base, probe)
        return self._synthesize(base, base.kernel, probe, conditional=conditional, hinge=hinge)

    def _synthesize(self, base, kernel, probe, *, conditional, hinge):
        goal = self._probes[probe]
        protected = self._protected_runs(base, probe)
        if conditional:
            self._require_term_capacity(base, hinge=hinge)
            _, _, term = kernel.program_conditional(
                kernel.start(goal.seed, base.memory), goal.steps, goal.target, protect=protected,
                atol=self._atol, max_update_norm=self._max_update_norm, hinge=hinge,
            )
            return FeedbackProposal(base, probe, term=term)
        _, _, edit = kernel.retarget(
            kernel.start(goal.seed, base.memory), goal.steps, goal.target, protect=protected,
            atol=self._atol, max_update_norm=self._max_update_norm,
        )
        return FeedbackProposal(base, probe, edit.weight)

    def _protected_runs(self, base, selected):
        return ((check.seed, check.steps) for probe, check in enumerate(self._probes)
                if probe != selected and base.errors[probe] <= self._atol)

    def propose_routing(self, base, probe, *, joint=False, trial=None):
        """Fit a query for an unmet source floor and optionally fit its value map."""
        self._require_current(base)
        if type(joint) is not bool or (joint and not self._allow_joint):
            raise ValueError("workspace joint query and value edits are disabled")
        if trial is not None and (not joint or not self._separate_joint_trials):
            raise ValueError("workspace separate joint trials are disabled")
        probe = self._open_probe(base, probe)
        goal = self._probes[probe]
        if goal.routing is None or base.source_masses[probe] >= goal.routing.min_mass:
            raise ValueError("probe has no unmet source mass floor")
        protected = tuple(self._protected_runs(base, probe))
        if not joint:
            _, query_edit = self._fit_routing_query(base, goal, goal.routing.min_mass, protected)
            return FeedbackProposal(base, probe, _query=query_edit.weight)
        if trial is not None:
            levels = tuple(self._joint_trial_levels(goal.routing.min_mass))
            trial = index(trial)
            if trial == len(levels) and self._can_refine_joint_levels(levels):
                return self._refine_joint_trials(base, probe, protected, levels)
            if not 0 <= trial < len(levels):
                raise ValueError("joint trial is outside the available source levels")
            anchor = self._joint_trial_anchor(base, probe)
            proposal, _ = self._fit_checked_joint_trial(base, probe, levels[trial], protected, anchor)
            self._require_current(base)
            return proposal
        if self._joint_trials == 1:
            proposal, _, _ = self._fit_joint_trial(base, probe, goal.routing.min_mass, protected)
            return proposal
        return self._search_joint_trials(base, probe, protected)

    def _fit_routing_query(self, base, goal, mass, protected):
        draft, _, query_edit = base.kernel.program_observation_mass(
            base.start(goal.seed), goal.steps, goal.routing.slot, mass,
            max_update_norm=self._max_update_norm, atol=self._atol, protect=protected,
            search="multivariate",
        )
        return draft, query_edit

    def _fit_joint_trial(self, base, probe, mass, protected, *, price_pair=False):
        goal = self._probes[probe]
        draft, query_edit = self._fit_routing_query(base, goal, mass, protected)
        self._require_current(base)
        limit, used = self._max_update_norm, query_edit.update_norm
        value_budget = limit
        if not price_pair:
            # Avoid squaring the cap or either update in Float64.
            fraction = ((limit - used) / limit) * (1 + used / limit) if limit else 0.0
            value_budget = limit * sqrt(min(1.0, fraction))
        candidate, _, value_edit = draft.retarget(
            draft.start(goal.seed, base.memory), goal.steps, goal.target,
            max_update_norm=value_budget, atol=self._atol, protect=protected,
        )
        proposal = FeedbackProposal(base, probe, _weight=value_edit.weight, _query=query_edit.weight)
        return proposal, candidate, hypot(used, value_edit.update_norm)

    def _joint_trial_levels(self, floor):
        mass = floor
        for _ in range(self._joint_trials):
            yield mass
            next_mass = mass + (1 - mass) * 0.5
            if not mass < next_mass < 1.0:
                break
            mass = next_mass

    def _can_refine_joint_levels(self, levels):
        return self._joint_refinements > 0 and any(
            _interior_midpoint(left, right) is not None for left, right in zip(levels, levels[1:])
        )

    def _joint_trial_anchor(self, base, probe):
        anchor = base._anchors[probe]
        if anchor is None:
            anchor = self._run(base.kernel, base.memory, self._probes[probe]).history[:-1]
        return anchor

    def _fit_checked_joint_trial(self, base, probe, mass, protected, anchor, *, price_pair=False):
        if price_pair:
            proposal, candidate, cost = self._fit_joint_trial(base, probe, mass, protected, price_pair=True)
        else:
            proposal, candidate, cost = self._fit_joint_trial(base, probe, mass, protected)
        if not price_pair and cost > self._max_update_norm:
            raise ValueError("joint candidate exceeds the combined update budget")
        self._check_joint_replay(base, probe, candidate, anchor)
        return proposal, cost

    def _check_joint_replay(self, base, probe, candidate, anchor):
        goal = self._probes[probe]
        result, actual_mass = self._run_routed(candidate, base.memory, goal)
        if not goal.routing.min_mass <= actual_mass <= 1.0 or self._probe_error(result, goal) > self._atol:
            raise ValueError("joint candidate fails the original target on replay")
        if not np.allclose(result.history[:-1], anchor, atol=self._atol, rtol=0):
            raise ValueError("joint candidate changes the original target prefix")

    def _price_joint_anchor(self, base, probe, proposal, anchor):
        if proposal._query is None or proposal._weight is None or proposal.base is not base or proposal.probe != probe:
            raise ValueError("reused anchor must contain both maps on the pinned probe")
        kernel = base.kernel
        old_query = kernel._query if kernel._observation_query is None else kernel._observation_query
        with np.errstate(over="raise", invalid="raise"):
            query_norm = _frobenius(proposal._query - old_query)
            value_norm = _frobenius(proposal._weight - kernel._dense_observation_value())
        if max(query_norm, value_norm) > self._max_update_norm:
            raise ValueError("reused anchor exceeds an individual map budget")
        candidate = kernel._with_observation_query(proposal._query)._with_observation_value(proposal._weight)
        self._check_joint_replay(base, probe, candidate, anchor)
        return hypot(query_norm, value_norm)

    def _search_joint_trials(self, base, probe, protected):
        anchor = self._joint_trial_anchor(base, probe)
        best = None
        reason = None
        for mass in self._joint_trial_levels(self._probes[probe].routing.min_mass):
            self._require_current(base)
            try:
                proposal, cost = self._fit_checked_joint_trial(base, probe, mass, protected, anchor)
                if best is None or cost < best[0]:
                    best = cost, proposal
            except StaleFeedbackRevision:
                raise
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                reason = error
        self._require_current(base)
        if best is None:
            raise ValueError(f"joint trials found no usable pair: {reason}") from reason
        return best[1]

    def _refine_joint_trials(self, base, probe, protected, levels, *, anchors=None):
        anchor = self._joint_trial_anchor(base, probe)
        best = None
        anchors = anchors or {}

        def price(mass, ordinal=None):
            nonlocal best
            self._require_current(base)
            proposal = anchors.get(ordinal)
            if ordinal in anchors and proposal is None:
                return float("inf")
            try:
                if proposal is not None:
                    try:
                        cost = self._price_joint_anchor(base, probe, proposal, anchor)
                    except StaleFeedbackRevision:
                        raise
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                        proposal = None
                if proposal is None:
                    proposal, cost = self._fit_checked_joint_trial(
                        base, probe, mass, protected, anchor, price_pair=True,
                    )
            except StaleFeedbackRevision:
                raise
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                return float("inf")
            if cost <= self._max_update_norm and (best is None or cost < best[0]):
                best = cost, proposal
            return cost

        samples = [(mass, price(mass, ordinal)) for ordinal, mass in enumerate(levels)]
        for _ in range(self._joint_refinements):
            intervals = []
            for position, ((left, left_cost), (right, right_cost)) in enumerate(zip(samples, samples[1:])):
                middle = _interior_midpoint(left, right)
                if middle is not None:
                    intervals.append((min(left_cost, right_cost), max(left_cost, right_cost),
                                      left - right, position, middle))
            if not intervals:
                break
            _, _, _, position, middle = min(intervals)
            samples.insert(position + 1, (middle, price(middle)))
        self._require_current(base)
        if best is None:
            raise ValueError("joint refinement found no usable pair within the cap")
        return best[1]

    def publish(self, proposal):
        """Admit one newly satisfied probe with every other probe non-worsening."""
        base = proposal.base
        self._require_current(base)
        return self._commit(base, self._prepare_operator_revision(proposal), proposal=proposal)

    def _candidate_order(self, base):
        unresolved = sorted(
            (probe for probe, (error, mass) in enumerate(zip(base.errors, base.source_masses, strict=True))
             if not self._probe_satisfied(probe, error, mass)),
            key=lambda probe: (-base.errors[probe], probe),
        )
        kinds = ["weight"]
        if base.kernel.conditional_terms < self._max_terms:
            kinds.append("conditional")
            if self._allow_hinge:
                kinds.append("hinge")
        candidates = []
        for probe in unresolved:
            candidates.extend((probe, kind) for kind in kinds)
            goal = self._probes[probe].routing
            if goal is not None and base.source_masses[probe] < goal.min_mass:
                candidates.append((probe, "route"))
                if self._separate_joint_trials:
                    levels = tuple(self._joint_trial_levels(goal.min_mass))
                    count = len(levels) + self._can_refine_joint_levels(levels)
                    candidates.extend((probe, f"joint:{trial}") for trial in range(count))
                elif self._allow_joint:
                    candidates.append((probe, "joint"))
        return candidates

    def _repair_rank(self, revision):
        return (-sum(self._probe_satisfied(probe, error, mass)
                     for probe, (error, mass) in enumerate(zip(revision.errors, revision.source_masses, strict=True))),
                sum(revision.errors), revision.kernel.conditional_terms)

    def _assess_tree_repair(self, base, probe, conditional, hinge):
        draft_cap, draft_tolerance = self._draft_tree_policy
        draft = base.kernel.with_frontier(draft_cap, logit_tolerance=draft_tolerance)
        resolutions = (("reference", base.kernel),)
        if draft is not base.kernel:
            resolutions = (("coarse", draft), *resolutions)
        trials = []
        for resolution, kernel in resolutions:
            self._require_current(base)
            stage = "synthesis"
            try:
                proposal = self._synthesize(base, kernel, probe, conditional=conditional, hinge=hinge)
                self._require_current(base)
                stage = "admission"
                revised = self._prepare_operator_revision(proposal)
            except StaleFeedbackRevision:
                raise
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                trials.append(FeedbackFidelityAttempt(resolution, kernel._max_frontier, stage, str(error)))
                continue
            trials.append(FeedbackFidelityAttempt(resolution, kernel._max_frontier))
            return proposal, revised, tuple(trials)
        return None, None, tuple(trials)

    def _evaluate_candidate(self, base, ordinal, candidates):
        probe, position_kind = candidates[ordinal]
        kind, trial = _split_candidate_kind(position_kind)
        conditional, hinge = kind in ("conditional", "hinge"), kind == "hinge"
        routing, joint = kind in ("route", "joint"), kind == "joint"
        if self._draft_tree_policy is not None and not routing:
            proposal, revised, trials = self._assess_tree_repair(base, probe, conditional, hinge)
            attempt = FeedbackRepairAttempt(
                probe, conditional, rejection=trials[-1].rejection,
                rank=None if revised is None else self._repair_rank(revised),
                hinge=hinge, fidelity_attempts=trials,
            )
            return attempt, proposal, revised
        try:
            proposal = (self.propose_routing(base, probe, joint=joint, trial=trial) if routing
                        else self._propose(base, probe, conditional=conditional, hinge=hinge))
            revised = self._prepare_operator_revision(proposal)
        except StaleFeedbackRevision:
            raise
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            attempt = FeedbackRepairAttempt(probe, conditional, str(error), hinge=hinge, routing=routing,
                                            joint=joint, joint_trial=trial)
            return attempt, None, None
        rank = self._repair_rank(revised)
        attempt = FeedbackRepairAttempt(probe, conditional, rank=rank, hinge=hinge, routing=routing,
                                        joint=joint, joint_trial=trial)
        return attempt, proposal, revised

    def _maintenance_candidates(self, base, candidates, offset, stop, workers):
        error_settings = np.geterr()
        error_handler = np.geterrcall()

        def inherit_error_policy():
            np.seterr(**error_settings)
            np.seterrcall(error_handler)

        pool = (ThreadPoolExecutor(max_workers=workers, initializer=inherit_error_policy)
                if workers > 1 else nullcontext())
        with pool as executor:
            for first in range(offset, stop, workers):
                self._require_current(base)
                if executor is None:
                    yield self._evaluate_candidate(base, first, candidates)
                    continue
                futures = [
                    executor.submit(self._evaluate_candidate, base, ordinal, candidates)
                    for ordinal in range(first, min(first + workers, stop))
                ]
                failures = [error for future in futures if (error := future.exception()) is not None]
                unexpected = [error for error in failures if not isinstance(error, StaleFeedbackRevision)]
                failures = unexpected or failures[:1]
                if len(failures) == 1:
                    raise failures[0]
                if failures:
                    raise BaseExceptionGroup("multiple candidate evaluations failed", failures)
                for future in futures:
                    yield future.result()
                del futures

    def maintain(self, *, max_candidates, resume=None, max_workers=1):
        """Compare a finite candidate batch on one revision and publish at most one edit."""
        limit = index(max_candidates)
        workers = index(max_workers)
        if limit < 1:
            raise ValueError("maintenance candidate limit must be positive")
        if workers < 1:
            raise ValueError("maintenance worker limit must be positive")
        base = self.snapshot()
        offset = 0
        if resume is not None:
            if (not isinstance(resume, FeedbackMaintenanceRound)
                    or resume.status != "budget_exhausted" or resume.next_candidate is None):
                raise ValueError("maintenance can only resume an exhausted candidate budget")
            if resume.base is not base:
                return FeedbackMaintenanceRound("stale", resume.base)
            offset = index(resume.next_candidate)
        candidates = self._candidate_order(base)
        if not candidates:
            status = "guard_unsatisfied" if any(e > self._atol for e in base.guard_errors) else "satisfied"
            return FeedbackMaintenanceRound(status, base)
        total = len(candidates)
        if not 0 <= offset < total:
            raise ValueError("maintenance continuation is outside the candidate list")
        stop = min(offset + limit, total)
        workers = min(workers, stop - offset)
        attempts = []
        best = None
        try:
            search = self._maintenance_candidates(base, candidates, offset, stop, workers)
            with closing(search) as results:
                for attempt, proposal, revised in results:
                    if attempt.rank is not None and (best is None or attempt.rank < best[0]):
                        best = attempt.rank, proposal, revised, len(attempts)
                    attempts.append(attempt)
            self._require_current(base)
            if best is not None:
                _, proposal, revised, selected = best
                published = self._commit(base, revised, proposal=proposal)
                return FeedbackMaintenanceRound("committed", base, tuple(attempts), published, selected)
        except StaleFeedbackRevision:
            return FeedbackMaintenanceRound("stale", base, tuple(attempts))
        if stop < total:
            return FeedbackMaintenanceRound("budget_exhausted", base, tuple(attempts), next_candidate=stop)
        return FeedbackMaintenanceRound("no_candidate", base, tuple(attempts))

    def _regresses_endpoints(self, old_errors, errors):
        return any(new > max(old, self._atol) for old, new in zip(old_errors, errors, strict=True))

    def _require_preserved_probes(self, base, errors, masses, *, subject):
        for check, old_mass, mass in zip(self._probes, base.source_masses, masses, strict=True):
            if check.routing is not None and old_mass >= check.routing.min_mass > mass:
                raise ValueError(f"{subject} regresses a protected source mass floor")
        if self._regresses_endpoints(base.errors, errors):
            raise ValueError(f"{subject} regresses a workspace probe")

    def _prepare_operator_revision(self, proposal):
        base = proposal.base
        probe = self._open_probe(base, proposal.probe)
        goal = self._probes[probe]
        if proposal.term is not None:
            self._require_term_capacity(base, hinge=proposal.term.threshold is not None)
            norm = proposal.term.norm
            candidate = base.kernel._with_conditional(proposal.term)
        else:
            candidate = base.kernel
            query_norm = value_norm = 0.0
            with np.errstate(over="raise", invalid="raise"):
                if proposal._query is not None:
                    if goal.routing is None:
                        raise ValueError("routing edit requires a declared source mass goal")
                    if proposal._weight is not None and not self._allow_joint:
                        raise ValueError("workspace joint query and value edits are disabled")
                    old_query = base.kernel._observation_query
                    if old_query is None:
                        old_query = base.kernel._query
                    query_norm = _frobenius(proposal._query - old_query)
                    candidate = candidate._with_observation_query(proposal._query)
                if proposal._weight is not None:
                    value_norm = _frobenius(proposal._weight - base.kernel._dense_observation_value())
                    candidate = candidate._with_observation_value(proposal._weight)
            norm = hypot(query_norm, value_norm)
        if not isfinite(norm) or norm > self._max_update_norm:
            raise ValueError("proposed operator exceeds the workspace update budget")
        errors, prefixes, probe_routes, masses = self._working_results(candidate, base.memory, base._probe_routes)
        if errors[probe] > self._atol:
            raise ValueError("proposed operator does not satisfy its target probe")
        if goal.routing is not None and masses[probe] < goal.routing.min_mass:
            raise ValueError("proposed operator does not meet the target source mass floor")
        self._require_preserved_probes(base, errors, masses, subject="proposed operator")
        guard_errors, guard_routes = self._guard_results(candidate, base.memory, base._guard_routes)
        if self._regresses_endpoints(base.guard_errors, guard_errors):
            raise ValueError("proposed operator regresses a workspace guard")
        anchors = list(base._anchors)
        if anchors[probe] is None:
            route = base._probe_routes[probe] if base._probe_routes else None
            anchors[probe] = route.prefix() if route is not None else self._run(
                base.kernel, base.memory, goal,
            ).history[:-1]
        for expected, actual in zip(anchors, prefixes, strict=True):
            if expected is not None and not np.allclose(actual, expected, atol=self._atol, rtol=0):
                raise ValueError("proposed operator changes a protected trajectory prefix")
        anchors = self._anchors(errors, prefixes, tuple(anchors))
        return FeedbackRevision(base.number + 1, candidate, base.memory, errors, anchors,
                                guard_errors, probe_routes, guard_routes, masses)

    def replace_observations(self, base, observations):
        """Install a new observation bank and measure its outcomes with the current operator."""
        self._require_current(base)
        return self._commit(base, self._prepare_observation_revision(base, observations))

    def revise_observation_slots(self, base, changes, *, repair_probe=None):
        """Revise tree slots, optionally requiring a guarded repair of one working probe."""
        self._require_current(base)
        revised = self._prepare_slot_update(base, changes, repair_probe)
        return base if revised is base else self._commit(base, revised)

    def _slot_preview_kernel(self, base, published_opening):
        self._require_current(base)
        kernel = base.kernel
        if type(kernel) is not TreeFeedbackKernel:
            raise ValueError("slot impact preview requires a standard tree feedback kernel")
        if type(published_opening) is not bool:
            raise ValueError("published opening preference must be a boolean")
        return kernel

    def _slot_preview_opening(self, base, kernel, seed, steps, published_opening):
        opening = None
        if published_opening:
            steps = index(steps)
            seed = kernel._vector(seed)
            signature = seed.tobytes()
            opening = next((trace for trace in (*base._probe_routes, *base._guard_routes)
                            if trace is not None and trace.matches(kernel, base.memory, signature, steps)), None)
        elif self._draft_tree_policy is not None:
            frontier, tolerance = self._draft_tree_policy
            kernel = kernel.with_frontier(frontier, logit_tolerance=tolerance)
        return kernel, seed, steps, opening

    def preview_observation_slots(self, base, changes, *, seed, steps, published_opening=False):
        """Compare an opening trajectory with dense replay of a proposed slot edit."""
        kernel = self._slot_preview_kernel(base, published_opening)
        reference = kernel.revise(base.memory, changes)
        self._check_reference_capacity(kernel, reference)
        kernel, seed, steps, opening = self._slot_preview_opening(base, kernel, seed, steps, published_opening)
        if opening is None:
            preview = kernel.run_with_error_bounds(seed, base.memory, steps=steps, reference_memory=reference)
        else:
            preview = kernel._run_with_error_bounds(
                seed, base.memory, steps=steps, reference_memory=reference, opening=opening,
            )
        self._require_current(base)
        return preview

    def _slot_batch_references(self, base, kernel, candidates, max_candidates):
        limit = index(max_candidates)
        if limit < 1:
            raise ValueError("preview batch needs a positive candidate budget")
        references = []
        for position, candidate in enumerate(candidates):
            if position == limit:
                raise ValueError("preview batch exceeds the candidate budget")
            if not isinstance(candidate, Mapping):
                raise TypeError("each preview candidate must be a slot-edit mapping")
            reference = kernel.revise(base.memory, candidate)
            self._check_reference_capacity(kernel, reference)
            references.append(reference)
        self._require_current(base)
        return tuple(references)

    def preview_observation_slot_batch(self, base, candidates, *, seed, steps, max_candidates, published_opening=False):
        """Compare independent proposed edits with a single opening trajectory."""
        kernel = self._slot_preview_kernel(base, published_opening)
        references = self._slot_batch_references(base, kernel, candidates, max_candidates)
        kernel, seed, steps, opening = self._slot_preview_opening(base, kernel, seed, steps, published_opening)
        previews = kernel._run_many_with_error_bounds(
            seed, base.memory, steps=steps, references=references, opening=opening,
        )
        self._require_current(base)
        return previews

    def _estimate_slot_read(self, preview, probe, read_index, is_guard, opening_error, opening_mass):
        endpoint = probe.steps - 1
        error = self._target_error(preview.state._history[endpoint], probe.target)
        radius = preview.error_bounds[endpoint]
        interval = max(0., error - radius), error + radius
        decision = None
        if interval[0] > self._atol:
            decision = False
        elif interval[1] < self._atol and probe.routing is None:
            decision = True
        floor = None if probe.routing is None else probe.routing.min_mass
        opening_satisfied = opening_error <= self._atol and (floor is None or opening_mass >= floor)
        return FeedbackSlotReadReview(read_index, is_guard, opening_error, opening_mass, floor,
                                      opening_satisfied, interval, decision)

    def _record_slot_replay(self, review, probe, state, mass):
        if mass is not None and not 0. <= mass <= 1.:
            return replace(review, replay_failure="probe source mass must be a probability")
        try:
            error = self._probe_error(state, probe)
        except FloatingPointError as failure:
            return replace(review, replay_failure=str(failure))
        satisfied = error <= self._atol and (probe.routing is None or mass >= probe.routing.min_mass)
        return replace(review, replayed_error=error, replayed_mass=mass, replayed_satisfied=satisfied)

    def _replay_slot_reads(self, base, reference, memory, reads, reviews):
        if len(reads) == 1:
            position, probe = reads[0]
            try:
                if probe.routing is None:
                    state, mass = self._run(reference, memory, probe), None
                else:
                    state, mass = self._run_routed(reference, memory, probe)
                reviews[position] = self._record_slot_replay(reviews[position], probe, state, mass)
            except FloatingPointError as failure:
                reviews[position] = replace(reviews[position], replay_failure=str(failure))
            return

        horizons = {}
        for position, probe in reads:
            horizons.setdefault(probe.steps, []).append((position, probe))
        completed = 0
        try:
            state = reference.start(reads[0][1].seed, memory)
            for horizon, endpoints in sorted(horizons.items()):
                if completed:
                    self._require_current(base)
                for _ in range(completed + 1, horizon):
                    state = reference.step(state)
                slots = tuple(dict.fromkeys(probe.routing.slot for _, probe in endpoints
                                            if probe.routing is not None))
                if slots:
                    state, probabilities = reference._source_masses_step(state, slots)
                    masses = dict(zip(slots, probabilities, strict=True))
                else:
                    state, masses = reference.step(state), {}
                for position, probe in endpoints:
                    mass = None if probe.routing is None else masses[probe.routing.slot]
                    reviews[position] = self._record_slot_replay(reviews[position], probe, state, mass)
                completed = horizon
        except FloatingPointError as failure:
            for position, probe in reads:
                if probe.steps > completed:
                    reviews[position] = replace(reviews[position], replay_failure=str(failure))

    def triage_observation_slot_batch(self, base, candidates, *, max_candidates, max_replays=0):
        """Review proposed edits against configured reads while bounding reference replay work."""
        kernel = self._slot_preview_kernel(base, True)
        replay_budget = index(max_replays)
        if replay_budget < 0:
            raise ValueError("slot triage needs a nonnegative replay budget")
        references = self._slot_batch_references(base, kernel, candidates, max_candidates)
        if not references:
            return FeedbackSlotTriage(base, (), 0)

        configured = [(probe, i, False, base.errors[i], base.source_masses[i])
                      for i, probe in enumerate(self._probes)]
        configured.extend((guard, i, True, base.guard_errors[i], None)
                          for i, guard in enumerate(self._guards))
        # Preserve separate executions for underflow warnings and NumPy's callback, log, and print modes.
        error_modes = np.geterr()
        share = (error_modes["under"] != "warn"
                 and not any(mode in ("call", "log", "print") for mode in error_modes.values()))
        signatures = [np.asarray(probe.seed, dtype=np.float64).tobytes() if share else position
                      for position, (probe, *_) in enumerate(configured)]
        seed_groups = {}
        for position, signature in enumerate(signatures):
            seed_groups.setdefault(signature, []).append(position)

        reviews = [[None] * len(configured) for _ in references]
        for positions in seed_groups.values():
            probe = configured[positions[0]][0]
            longest = max(configured[position][0].steps for position in positions)
            _, seed, steps, opening = self._slot_preview_opening(base, kernel, probe.seed, longest, True)
            previews = kernel._run_many_with_error_bounds(
                seed, base.memory, steps=steps, references=references, opening=opening,
            )
            for entries, preview in zip(reviews, previews, strict=True):
                for position in positions:
                    entries[position] = self._estimate_slot_read(preview, *configured[position])

        pending = []
        for alternative, entries in enumerate(reviews):
            for position, review in enumerate(entries):
                opening_floor_met = review.source_floor is not None and review.opening_mass >= review.source_floor
                if review.estimated_satisfied is None or opening_floor_met:
                    opened_within = review.opening_error <= self._atol
                    pending.append((not (opened_within or opening_floor_met), not review.is_guard,
                                    alternative, position))
        pending.sort()
        selected = pending[:replay_budget]
        replay_groups = {}
        for _, _, alternative, position in selected:
            key = alternative, signatures[position]
            replay_groups.setdefault(key, []).append((position, configured[position][0]))
        for (alternative, _), reads in replay_groups.items():
            self._require_current(base)
            memory = references[alternative]
            reference = kernel.with_frontier(max(kernel._max_frontier, memory.count), logit_tolerance=0.)
            self._replay_slot_reads(base, reference, memory, reads, reviews[alternative])

        self._require_current(base)
        alternatives = tuple(FeedbackSlotAlternativeReview(memory, tuple(entries))
                             for memory, entries in zip(references, reviews, strict=True))
        return FeedbackSlotTriage(base, alternatives, len(selected), len(replay_groups))

    def _prepare_slot_revision(self, base, changes):
        if not isinstance(base.kernel, TreeFeedbackKernel):
            raise ValueError("slot revisions require a tree feedback kernel")
        memory = base.kernel.revise(base.memory, changes)
        return base if memory is base.memory else self._prepare_memory_revision(base, memory)

    def _prepare_slot_update(self, base, changes, repair_probe):
        if repair_probe is None:
            return self._prepare_slot_revision(base, changes)
        if isinstance(repair_probe, (bool, np.bool_)):
            raise ValueError("repair probe must be a working probe index")
        probe = self._open_probe(base, repair_probe)
        revised = self._prepare_slot_revision(base, changes)
        self._require_current(base)
        if not self._probe_satisfied(probe, revised.errors[probe], revised.source_masses[probe]):
            raise ValueError("proposed slot repair does not satisfy its target probe")
        self._require_preserved_probes(base, revised.errors, revised.source_masses, subject="proposed slot repair")
        if self._regresses_endpoints(base.guard_errors, revised.guard_errors):
            raise ValueError("proposed slot repair regresses a workspace guard")
        return revised

    def _prepare_observation_revision(self, base, observations):
        return self._prepare_memory_revision(base, base.kernel.prepare(observations))

    def _prepare_memory_revision(self, base, memory):
        return self._measure_bank(base.number + 1, base.kernel, memory, base)

    def _measure_bank(self, number, kernel, memory, previous=None):
        self._check_reference_capacity(kernel, memory)
        probe_previous = () if previous is None or not self._max_retained_tree_scores else previous._probe_routes
        guard_previous = () if previous is None or not self._max_retained_tree_scores else previous._guard_routes
        errors, prefixes, probe_routes, masses = self._working_results(kernel, memory, probe_previous)
        guard_errors, guard_routes = self._guard_results(kernel, memory, guard_previous)
        return FeedbackRevision(number, kernel, memory, errors, self._anchors(errors, prefixes),
                                guard_errors, probe_routes, guard_routes, masses)
