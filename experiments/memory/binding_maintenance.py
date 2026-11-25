"""Propose one bounded graph rewrite from a failed pinned feedback read."""

from dataclasses import dataclass
from math import isfinite
from operator import index

from experiments.memory.binding_feedback import BindingFeedback, BindingFeedbackState
from experiments.memory.binding_memory import BindingGate, BindingProof, _proof_work_limit
from experiments.memory.binding_replay import (
    RouteFitRefusal, failed_binding_prefix, program_binding_route, replay_binding_repair, route_norm_cap,
)
from experiments.memory.corpus_binding_journal import (
    CorpusProgramExpectation, CorpusProgramProbe, CorpusProgramTrialResult, JournaledCorpusWorkspace,
    _trial_expectation,
)
from experiments.memory.corpus_bindings import CorpusBindingRead


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


def _prefix_probes(trajectory, earlier):
    if trajectory._receiver._disjoint_reads:
        return tuple(CorpusProgramProbe(earlier[:i]) for i in range(1, len(earlier) + 1))
    return tuple(CorpusProgramProbe((address,)) for address in earlier)


def _allowed_replacements(substitutions, predicates):
    allowed = {}
    for source, replacement in substitutions:
        if (type(source) is not str or type(replacement) is not str
                or source not in predicates or replacement not in predicates or source == replacement):
            raise ValueError("binding repair substitutions require distinct declared leaf predicates")
        allowed.setdefault(source, {})[replacement] = None
    return allowed


def propose_binding_program(workspace, base, trajectory, *, substitutions, future_target,
                            future_batches, work_limit, max_candidates, protected=(), downstream_target=None,
                            max_route_update=None, route_margin=.1):
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
    if max_route_update is not None:
        max_route_update, route_margin = route_norm_cap(max_route_update), float(route_margin)
        if downstream_target is None or not isfinite(route_margin) or route_margin <= 0:
            raise ValueError("route fitting needs a downstream target and finite positive limits")
        followup, _, _ = _trial_expectation(base.memory, downstream_target, work_limit)
        next_address = followup[0][-1]
        if next_address not in trajectory._receiver._addresses:
            raise ValueError("route fitting needs a fixed downstream address")
    (addresses, excluded), witness, upper = _trial_expectation(base.memory, future_target, work_limit)
    if excluded or witness is None:
        raise ValueError("binding repair needs a future certain proof for the same unexcluded read")
    probe = CorpusProgramProbe(addresses)
    expectation = CorpusProgramExpectation(probe, CorpusBindingRead(BindingProof(witness[0], frozenset(witness[1])),
                                                                   upper))
    prefix = failed_binding_prefix(trajectory, addresses)
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
            status, steps, repaired = replay_binding_repair(trajectory, proposal.preview, address)
            patch = None
            if status == "repaired" and max_route_update is not None:
                try:
                    patch = program_binding_route(trajectory, repaired, next_address,
                                                  margin=route_margin, max_update_norm=max_route_update)
                except ValueError:
                    status = "route_fit_refused"
                except FloatingPointError:
                    status = "replay_numerical"
            if status == "repaired":
                try:
                    trial = workspace.begin_program_trial(
                        base, gates, targets=(expectation,), protected=(*_prefix_probes(trajectory, prefix), *protected),
                        future_batches=future_batches, work_limit=work_limit, replay=trajectory, downstream=downstream_target,
                        route_edit=patch, route_update_limit=max_route_update,
                    )
                except RouteFitRefusal:
                    status = "route_fit_refused"
                else:
                    attempts.append(BindingRepairAttempt(slot, replacement, status, steps))
                    return BindingRepairResult("opened", tuple(attempts), trial)
            attempts.append(BindingRepairAttempt(slot, replacement, status, steps))
    statuses = {attempt.status for attempt in attempts}
    for status in ("replay_numerical", "replay_incomplete", "route_fit_refused"):
        if status in statuses:
            return BindingRepairResult(status, tuple(attempts))
    return BindingRepairResult("no_candidate", tuple(attempts))
