"""Publish independently checked numerical operator edits on a pinned observation bank."""

from dataclasses import dataclass
from math import isfinite
from operator import index
from threading import Lock

import numpy as np

from experiments.memory.latent_feedback import (
    ConditionalTerm, FeedbackKernel, ObservationMemory, _real_array,
)
from experiments.memory.linear_feedback import MomentMemory
from experiments.memory.weight_program import _frobenius


@dataclass(frozen=True)
class FeedbackProbe:
    seed: tuple[float, ...]
    steps: int
    target: tuple[float, ...]

    def __post_init__(self):
        seed = _real_array(self.seed)
        target = _real_array(self.target)
        steps = index(self.steps)
        if seed.ndim != 1 or not len(seed) or seed.shape != target.shape:
            raise ValueError("probe must have equally sized vector seed and target")
        if steps < 1 or np.any(np.abs(target) >= 1):
            raise ValueError("probe needs positive steps and an interior tanh target")
        object.__setattr__(self, "seed", tuple(seed))
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "target", tuple(target))


@dataclass(frozen=True, eq=False)
class FeedbackRevision:
    number: int
    kernel: FeedbackKernel
    memory: ObservationMemory | MomentMemory
    errors: tuple[float, ...]
    _anchors: tuple[tuple[np.ndarray, ...] | None, ...]
    guard_errors: tuple[float, ...] = ()

    def start(self, seed):
        return self.kernel.start(seed, self.memory)


@dataclass(frozen=True, eq=False)
class FeedbackProposal:
    base: FeedbackRevision
    probe: int
    _weight: np.ndarray | None = None
    term: ConditionalTerm | None = None

    def __post_init__(self):
        object.__setattr__(self, "probe", index(self.probe))
        if (self._weight is None) == (self.term is None):
            raise ValueError("proposal must contain one value map or one conditional term")
        if self.term is not None and type(self.term) is not ConditionalTerm:
            raise ValueError("invalid conditional proposal data")
        if self._weight is not None:
            object.__setattr__(self, "_weight", self.base.kernel._weight(self._weight))


class StaleFeedbackRevision(ValueError):
    """The observations or operator have advanced since this work was based."""


@dataclass(frozen=True)
class FeedbackRepairAttempt:
    probe: int
    conditional: bool
    rejection: str | None = None
    rank: tuple[int, float, int] | None = None
    hinge: bool = False


@dataclass(frozen=True)
class FeedbackMaintenanceRound:
    status: str
    base: FeedbackRevision
    attempts: tuple[FeedbackRepairAttempt, ...] = ()
    revision: FeedbackRevision | None = None
    selected_attempt: int | None = None
    next_candidate: int | None = None


class FeedbackWorkspace:
    """A bounded in-process commit point. Numeric replays run outside its lock."""

    def __init__(self, kernel, observations, probes, *, max_update_norm, atol=1e-8,
                 max_terms=0, guards=(), allow_hinge=False):
        self._probes = tuple(FeedbackProbe(p.seed, p.steps, p.target) for p in probes)
        self._guards = tuple(FeedbackProbe(p.seed, p.steps, p.target) for p in guards)
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
        errors, prefixes = self._measure(kernel, memory)
        guard_errors = self._measure_guards(kernel, memory)
        self._head = FeedbackRevision(0, kernel, memory, errors,
                                      self._anchors(errors, prefixes), guard_errors)

    def snapshot(self):
        with self._lock:
            return self._head

    def _require_current(self, base):
        if self.snapshot() is not base:
            raise StaleFeedbackRevision("feedback revision is no longer current")

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
        error = float(np.max(np.abs(state.latest - probe.target)))
        if not isfinite(error):
            raise ValueError("probe error must be finite")
        return error

    def _measure(self, kernel, memory):
        errors, prefixes = [], []
        for probe in self._probes:
            state = self._run(kernel, memory, probe)
            errors.append(self._probe_error(state, probe))
            prefixes.append(state.history[:-1])
        return tuple(errors), tuple(prefixes)

    def _measure_guards(self, kernel, memory):
        return tuple(self._probe_error(self._run(kernel, memory, guard), guard)
                     for guard in self._guards)

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
        if base.errors[probe] <= self._atol:
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
        return self._propose(base, probe, conditional=conditional or hinge, hinge=hinge)

    def _propose(self, base, probe, *, conditional, hinge=False):
        probe = self._open_probe(base, probe)
        goal = self._probes[probe]
        protected = (
            (check.seed, check.steps)
            for check, error in zip(self._probes, base.errors, strict=True)
            if error <= self._atol
        )
        if conditional:
            self._require_term_capacity(base, hinge=hinge)
            _, _, term = base.kernel.program_conditional(
                base.start(goal.seed), goal.steps, goal.target, protect=protected,
                atol=self._atol, max_update_norm=self._max_update_norm, hinge=hinge,
            )
            return FeedbackProposal(base, probe, term=term)
        _, _, edit = base.kernel.retarget(
            base.start(goal.seed), goal.steps, goal.target, protect=protected,
            atol=self._atol, max_update_norm=self._max_update_norm,
        )
        return FeedbackProposal(base, probe, edit.weight)

    def publish(self, proposal):
        """Admit one newly satisfied probe with every other probe non-worsening."""
        base = proposal.base
        self._require_current(base)
        return self._commit(base, self._prepare_operator_revision(proposal), proposal=proposal)

    def maintain(self, *, max_candidates, resume=None):
        """Compare a finite candidate batch on one revision and publish at most one edit."""
        limit = index(max_candidates)
        if limit < 1:
            raise ValueError("maintenance candidate limit must be positive")
        base = self.snapshot()
        offset = 0
        if resume is not None:
            if (not isinstance(resume, FeedbackMaintenanceRound)
                    or resume.status != "budget_exhausted" or resume.next_candidate is None):
                raise ValueError("maintenance can only resume an exhausted candidate budget")
            if resume.base is not base:
                return FeedbackMaintenanceRound("stale", resume.base)
            offset = index(resume.next_candidate)
        unresolved = sorted(
            (probe for probe, error in enumerate(base.errors) if error > self._atol),
            key=lambda probe: (-base.errors[probe], probe),
        )
        if not unresolved:
            status = "guard_unsatisfied" if any(e > self._atol for e in base.guard_errors) else "satisfied"
            return FeedbackMaintenanceRound(status, base)
        modes = [(False, False)]
        if base.kernel.conditional_terms < self._max_terms:
            modes.append((True, False))
            if self._allow_hinge:
                modes.append((True, True))
        total = len(unresolved) * len(modes)
        if not 0 <= offset < total:
            raise ValueError("maintenance continuation is outside the candidate list")
        stop = min(offset + limit, total)
        attempts = []
        best = None
        try:
            for candidate in range(offset, stop):
                probe_offset, mode_offset = divmod(candidate, len(modes))
                probe = unresolved[probe_offset]
                conditional, hinge = modes[mode_offset]
                self._require_current(base)
                try:
                    proposal = self._propose(base, probe, conditional=conditional, hinge=hinge)
                    revised = self._prepare_operator_revision(proposal)
                except StaleFeedbackRevision:
                    raise
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    attempts.append(FeedbackRepairAttempt(probe, conditional, str(error), hinge=hinge))
                    continue
                rank = (-sum(error <= self._atol for error in revised.errors),
                        sum(revised.errors), revised.kernel.conditional_terms)
                if best is None or rank < best[0]:
                    best = rank, proposal, revised, len(attempts)
                attempts.append(FeedbackRepairAttempt(probe, conditional, rank=rank, hinge=hinge))
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

    def _prepare_operator_revision(self, proposal):
        base = proposal.base
        probe = self._open_probe(base, proposal.probe)
        if proposal.term is None:
            with np.errstate(over="raise", invalid="raise"):
                norm = _frobenius(proposal._weight - base.kernel._dense_observation_value())
            candidate = base.kernel._with_observation_value(proposal._weight)
        else:
            self._require_term_capacity(base, hinge=proposal.term.threshold is not None)
            norm = proposal.term.norm
            candidate = base.kernel._with_conditional(proposal.term)
        if not isfinite(norm) or norm > self._max_update_norm:
            raise ValueError("proposed operator exceeds the workspace update budget")
        errors, prefixes = self._measure(candidate, base.memory)
        if errors[probe] > self._atol:
            raise ValueError("proposed operator does not satisfy its target probe")
        if any(new > max(old, self._atol)
               for old, new in zip(base.errors, errors, strict=True)):
            raise ValueError("proposed operator regresses a workspace probe")
        guard_errors = self._measure_guards(candidate, base.memory)
        if any(new > max(old, self._atol)
               for old, new in zip(base.guard_errors, guard_errors, strict=True)):
            raise ValueError("proposed operator regresses a workspace guard")
        anchors = list(base._anchors)
        goal = self._probes[probe]
        anchors[probe] = self._run(base.kernel, base.memory, goal).history[:-1]
        for expected, actual in zip(anchors, prefixes, strict=True):
            if expected is not None and not np.allclose(actual, expected, atol=self._atol, rtol=0):
                raise ValueError("proposed operator changes a protected trajectory prefix")
        anchors = self._anchors(errors, prefixes, tuple(anchors))
        return FeedbackRevision(base.number + 1, candidate, base.memory, errors, anchors, guard_errors)

    def replace_observations(self, base, observations):
        """Install a new observation bank and measure its outcomes with the current operator."""
        self._require_current(base)
        return self._commit(base, self._prepare_observation_revision(base, observations))

    def _prepare_observation_revision(self, base, observations):
        return self._prepare_memory_revision(base, base.kernel.prepare(observations))

    def _prepare_memory_revision(self, base, memory):
        errors, prefixes = self._measure(base.kernel, memory)
        guard_errors = self._measure_guards(base.kernel, memory)
        return FeedbackRevision(base.number + 1, base.kernel, memory, errors,
                                self._anchors(errors, prefixes), guard_errors)
