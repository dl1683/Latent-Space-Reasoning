"""Propose one bounded graph rewrite from a failed pinned feedback read."""

from dataclasses import dataclass
from operator import index

import numpy as np

from experiments.memory.binding_feedback import BindingFeedback, BindingFeedbackState
from experiments.memory.binding_memory import BindingGate, BindingProof, _proof_work_limit
from experiments.memory.corpus_binding_journal import (
    CorpusProgramExpectation, CorpusProgramProbe, CorpusProgramTrialResult, JournaledCorpusWorkspace,
    _trial_expectation,
)
from experiments.memory.corpus_bindings import CorpusBindingRead


_INCOMPLETE_REPLAY = frozenset({
    "ambiguous", "below_floor", "unavailable", "capacity", "expansion_capacity", "candidate_limit", "incomplete",
})


@dataclass(frozen=True)
class BindingRepairAttempt:
    slot: int
    replacement: str
    status: str
    replayed_steps: int


@dataclass(frozen=True)
class BindingRepairResult:
    status: str
    attempts: tuple[BindingRepairAttempt, ...] = ()
    trial: CorpusProgramTrialResult | None = None


def _closed_absence(read):
    return read.status == "resolved" and read.bounds.possible_strength is None


def _prefix_probes(trajectory, addresses):
    prefix, last = trajectory.reads[:-1], trajectory.reads[-1]
    if any(read.status != "resolved" or read.bounds.possible_strength is None for read in prefix):
        return None
    address = addresses[-1]
    if not ((_closed_absence(last) and last.address == address)
            or (last.status == "catalog_absent" and any(
                _closed_absence(read) and read.address == address for read in last.tried))):
        return None
    earlier = tuple(read.address for read in prefix)
    if trajectory._receiver._disjoint_reads:
        if addresses[:-1] != earlier:
            return None
        return tuple(CorpusProgramProbe(earlier[:i]) for i in range(1, len(earlier) + 1))
    if len(addresses) != 1:
        return None
    return tuple(CorpusProgramProbe((address,)) for address in earlier)


def _same_read(first, second):
    return (first.status, first.address, first.bounds) == (second.status, second.address, second.bounds)


def _replay_repair(trajectory, memory, address):
    receiver = trajectory._receiver
    replay = receiver.start(trajectory.feedback.seed, memory)
    for step, original in enumerate(trajectory.reads[:-1], start=1):
        replay = receiver.step(replay)
        read = replay.reads[-1]
        if read.status in _INCOMPLETE_REPLAY:
            return "replay_incomplete", step
        if not _same_read(read, original):
            return "prefix_changed", step
    replay = receiver.step(replay)
    read = replay.reads[-1]
    steps = len(trajectory.reads)
    if read.status in _INCOMPLETE_REPLAY:
        return "replay_incomplete", steps
    if read.status != "resolved" or read.address != address or read.bounds.certain is None:
        return "target_unrepaired", steps
    if np.array_equal(replay.latest, trajectory.latest):
        return "state_unchanged", steps
    return "repaired", steps


def _allowed_replacements(substitutions, predicates):
    allowed = {}
    for source, replacement in substitutions:
        if (type(source) is not str or type(replacement) is not str
                or source not in predicates or replacement not in predicates or source == replacement):
            raise ValueError("binding repair substitutions require distinct declared leaf predicates")
        allowed.setdefault(source, {})[replacement] = None
    return allowed


def propose_binding_program(workspace, base, trajectory, *, substitutions, future_target,
                            future_batches, work_limit, max_candidates, protected=()):
    """Replay authorized single-parent edits and register the first warranted future trial."""
    if type(workspace) is not JournaledCorpusWorkspace:
        raise ValueError("binding repair needs a shared corpus journal")
    workspace._check_owner(base)
    if (type(trajectory) is not BindingFeedbackState or type(trajectory._receiver) is not BindingFeedback
            or trajectory.memory is not base.memory or not trajectory.reads):
        raise ValueError("binding repair needs a trajectory on the exact corpus pin")
    max_candidates, future_batches = index(max_candidates), index(future_batches)
    if min(max_candidates, future_batches) < 1:
        raise ValueError("binding repair needs positive candidate and future batch limits")
    work_limit = _proof_work_limit(work_limit)
    (addresses, excluded), witness, upper = _trial_expectation(base.memory, future_target, work_limit)
    if excluded or witness is None:
        raise ValueError("binding repair needs a future certain proof for the same unexcluded read")
    probe = CorpusProgramProbe(addresses)
    expectation = CorpusProgramExpectation(probe, CorpusBindingRead(BindingProof(witness[0], frozenset(witness[1])),
                                                                   upper))
    prefix = _prefix_probes(trajectory, addresses)
    address = addresses[-1]
    node, _ = address
    gates = dict(base.memory._certain._gates)
    if prefix is None or node not in gates:
        return BindingRepairResult("ineligible")
    allowed = _allowed_replacements(substitutions, base.memory._certain._predicates)
    gate, attempts = gates[node], []
    for slot, parent in enumerate(gate.parents):
        for replacement in allowed.get(parent, ()):
            if len(attempts) == max_candidates:
                return BindingRepairResult("candidate_limit", tuple(attempts))
            parents = gate.parents[:slot] + (replacement,) + gate.parents[slot + 1:]
            gates[node] = BindingGate(gate.operation, parents)
            proposal = workspace.prepare_program(base, gates)
            status, steps = _replay_repair(trajectory, proposal.preview, address)
            attempts.append(BindingRepairAttempt(slot, replacement, status, steps))
            if status == "repaired":
                trial = workspace.begin_program_trial(
                    base, gates, targets=(expectation,), protected=(*prefix, *protected),
                    future_batches=future_batches, work_limit=work_limit,
                )
                return BindingRepairResult("opened", tuple(attempts), trial)
    status = "replay_incomplete" if any(attempt.status == "replay_incomplete" for attempt in attempts) else "no_candidate"
    return BindingRepairResult(status, tuple(attempts))
