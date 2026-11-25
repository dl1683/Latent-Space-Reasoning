"""Publish independently checked numerical operator edits on a pinned observation bank."""

from dataclasses import dataclass
from math import isfinite
from operator import index
from threading import Lock

import numpy as np

from experiments.memory.latent_feedback import FeedbackKernel, ObservationMemory, _real_array
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
    memory: ObservationMemory
    errors: tuple[float, ...]

    def start(self, seed):
        return self.kernel.start(seed, self.memory)


@dataclass(frozen=True, eq=False)
class FeedbackProposal:
    base: FeedbackRevision
    probe: int
    _weight: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "_weight", self.base.kernel._weight(self._weight))


class StaleFeedbackRevision(ValueError):
    """The observations or operator have advanced since this work was based."""


class FeedbackWorkspace:
    """A bounded in-process commit point. Numeric replays run outside its lock."""

    def __init__(self, kernel, observations, probes, *, max_update_norm, atol=1e-8):
        self._probes = tuple(FeedbackProbe(p.seed, p.steps, p.target) for p in probes)
        self._atol = float(atol)
        self._max_update_norm = float(max_update_norm)
        if not self._probes:
            raise ValueError("at least one behavioral probe is required")
        if not isfinite(self._atol) or self._atol <= 0:
            raise ValueError("atol must be positive and finite")
        if not isfinite(self._max_update_norm) or self._max_update_norm < 0:
            raise ValueError("max_update_norm must be finite and nonnegative")
        self._lock = Lock()
        memory = kernel.prepare(observations)
        self._head = FeedbackRevision(0, kernel, memory, self._measure(kernel, memory))

    def snapshot(self):
        with self._lock:
            return self._head

    def _require_current(self, base):
        if self.snapshot() is not base:
            raise StaleFeedbackRevision("feedback revision is no longer current")

    def _commit(self, base, kernel, memory, errors):
        with self._lock:
            if self._head is not base:
                raise StaleFeedbackRevision("feedback revision changed before publication")
            self._head = FeedbackRevision(base.number + 1, kernel, memory, errors)
            return self._head

    def _measure(self, kernel, memory):
        errors = []
        for probe in self._probes:
            state = kernel.start(probe.seed, memory)
            for _ in range(probe.steps):
                state = kernel.step(state)
            error = float(np.max(np.abs(state.latest - probe.target)))
            if not isfinite(error):
                raise ValueError("probe error must be finite")
            errors.append(error)
        return tuple(errors)

    def _open_probe(self, base, probe):
        probe = index(probe)
        if not 0 <= probe < len(self._probes):
            raise ValueError("probe index out of range")
        if base.errors[probe] <= self._atol:
            raise ValueError("probe already satisfies the workspace tolerance")
        return probe

    def propose(self, base, probe):
        """Produce a candidate using the unresolved probe and satisfied runs."""
        self._require_current(base)
        probe = self._open_probe(base, probe)
        goal = self._probes[probe]
        protected = (
            (check.seed, check.steps)
            for check, error in zip(self._probes, base.errors, strict=True)
            if error <= self._atol
        )
        _, _, edit = base.kernel.retarget(
            base.start(goal.seed), goal.steps, goal.target, protect=protected,
            atol=self._atol, max_update_norm=self._max_update_norm,
        )
        return FeedbackProposal(base, probe, edit.weight)

    def publish(self, proposal):
        """Admit one newly satisfied probe with every other probe non-worsening."""
        base = proposal.base
        self._require_current(base)
        probe = self._open_probe(base, proposal.probe)
        with np.errstate(over="raise", invalid="raise"):
            norm = _frobenius(proposal._weight - base.kernel._observation_value)
        if not isfinite(norm) or norm > self._max_update_norm:
            raise ValueError("proposed operator exceeds the workspace update budget")
        candidate = base.kernel._with_observation_value(proposal._weight)
        errors = self._measure(candidate, base.memory)
        if errors[probe] > self._atol:
            raise ValueError("proposed operator does not satisfy its target probe")
        if any(new > max(old, self._atol)
               for old, new in zip(base.errors, errors, strict=True)):
            raise ValueError("proposed operator regresses a workspace probe")
        return self._commit(base, candidate, base.memory, errors)

    def replace_observations(self, base, observations):
        """Install a new observation bank and measure its outcomes with the current operator."""
        self._require_current(base)
        memory = base.kernel.prepare(observations)
        errors = self._measure(base.kernel, memory)
        return self._commit(base, base.kernel, memory, errors)
