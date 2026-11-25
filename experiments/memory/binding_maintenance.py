"""Propose bounded graph or transition edits from pinned feedback reads."""

from dataclasses import dataclass
from math import isfinite, nextafter
from operator import index

import numpy as np

from experiments.memory.binding_feedback import BindingFeedback, BindingFeedbackState
from experiments.memory.binding_memory import BindingCapacityError, BindingGate, BindingProof, _proof_work_limit
from experiments.memory.binding_replay import (
    RouteFitRefusal, _REPLAY_ERRORS, _clears_opening_rank, _receiver_digest, _replay_coordinate_prefix,
    check_coordinate_protection, check_coordinate_replay, failed_binding_prefix, program_binding_context,
    program_binding_route, replay_binding_repair, route_norm_cap,
)
from experiments.memory.corpus_binding_journal import (
    CorpusProgramExpectation, CorpusProgramProbe, CorpusProgramTrialResult, JournaledCorpusWorkspace,
    _cold_trial_memory, _trial_expectation,
)
from experiments.memory.corpus_bindings import CorpusBindingRead, StaleCorpusRevision


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


@dataclass(frozen=True)
class BindingContextProposal:
    status: str
    sweeps: int = 0
    target: CorpusProgramExpectation | None = None
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


def _project_context_route(receiver, hidden, column, padding, max_sweeps, *, preactivation_weighted):
    """Try a padded score cell in the open tanh box, then check the real scores."""
    query = receiver._query
    wanted = query[:, column]
    rivals = np.delete(query, column, axis=1)
    normals = np.vstack((wanted, (wanted[:, None] - rivals).T))
    levels = np.full(len(normals), receiver._route_gap + padding)
    levels[0] = receiver._route_floor + padding
    if not np.isfinite(levels).all() or levels[0] <= receiver._route_floor:
        return None, 0

    scales = np.max(np.abs(normals), axis=1)
    if np.any((scales == 0.) & (levels > 0.)):
        return None, 0
    keep = scales > 0.
    axes, levels = normals[keep] / scales[keep, None], levels[keep] / scales[keep]
    # One predecessor of 1.0 keeps inversion defined even when clipping.
    edge = nextafter(1., 0.)
    point = np.clip(hidden, -edge, edge)
    # The inverse local metric favors coordinates with larger tanh derivatives.
    directions = axes * (1. - point ** 2) ** 2 if preactivation_weighted else axes
    squared_norms = np.einsum("ij,ij->i", axes, directions)
    directions = directions / squared_norms[:, None]
    for sweep in range(1, max_sweeps + 1):
        previous = point.copy()
        for axis, level, direction in zip(axes, levels, directions, strict=True):
            violation = level - point @ axis
            if violation > 0.:
                point += violation * direction
        np.clip(point, -edge, edge, out=point)
        if _clears_opening_rank(receiver, point, column):
            return point, sweep
        if np.array_equal(previous, point):
            return None, sweep
    return None, max_sweeps


def propose_binding_context(workspace, base, ready, terminal, *, future_batches, max_update_norm, max_sweeps,
                            padding=.1, protect=()):
    """Derive a context edit from an opening disjoint proof and freeze a future trial.

    Ready is the incumbent state before the transition to edit. Each protected
    trajectory has already performed its declared terminal read on this pin.
    """
    if type(workspace) is not JournaledCorpusWorkspace:
        raise ValueError("context proposal needs a shared corpus journal")
    workspace._check_owner(base)
    if workspace.snapshot() is not base:
        raise StaleCorpusRevision("corpus workspace tip changed")
    if type(ready) is not BindingFeedbackState or type(ready._receiver) is not BindingFeedback:
        raise ValueError("context proposal needs a binding feedback state")
    receiver = ready._receiver
    if not receiver._disjoint_reads or receiver._expansion is not None:
        raise ValueError("context proposal needs a fixed disjoint receiver")
    if base._receiver_programs and _receiver_digest(receiver) != base._receiver_programs[-1][1]:
        raise ValueError("context proposal does not use the installed receiver")
    protect = tuple(protect)
    if any(type(state) is not BindingFeedbackState or state._receiver is not receiver or state.memory is not base.memory
           for state in (ready, *protect)):
        raise ValueError("context proposal states must come from the incumbent on the exact opening pin")
    future_batches, max_sweeps = index(future_batches), index(max_sweeps)
    padding, cap = float(padding), route_norm_cap(max_update_norm)
    if min(future_batches, max_sweeps) < 1 or not isfinite(padding) or padding <= 0.:
        raise ValueError("context proposal needs positive horizons, sweep limit and route padding")
    name, coord = terminal
    terminal = name, index(coord)
    column = receiver._addresses.index(terminal)
    if len(ready.reads) + 2 > receiver._kernel._max_steps:
        raise ValueError("context proposal needs one transition and a terminal read remaining")
    if any(len(run.reads) < 2 for run in protect):
        raise ValueError("a protected continuation needs an observed terminal and an earlier read")
    if any(read.status != "resolved" or not read.bounds.is_exact or read.bounds.certain is None
           for state in (ready, *protect) for read in state.reads):
        return BindingContextProposal("opening_incomplete")
    memory = _cold_trial_memory(base.memory)
    candidate, sweeps = None, 0
    try:
        with np.errstate(**_REPLAY_ERRORS):
            try:
                ready = _replay_coordinate_prefix(receiver, ready, 1e-8, memory=memory)
                protect = tuple(_replay_coordinate_prefix(receiver, run, 1e-8, memory=memory) for run in protect)
            except ValueError:
                return BindingContextProposal("opening_trace_changed")
            next_read = receiver._next_read(ready)
            if next_read.status != "resolved" or not next_read.bounds.is_exact or next_read.bounds.certain is None:
                return BindingContextProposal("opening_incomplete")
            incumbent = receiver._advance_read(ready, next_read)
            addresses = tuple(read.address for read in incumbent.reads) + (terminal,)
            proof = memory.resolve_disjoint(addresses, work_limit=receiver._work_limit)
            if proof.certain is None or not proof.is_exact:
                return BindingContextProposal("target_not_exact")
            previous = receiver._next_read(incumbent)
            if previous.status == "resolved" and previous.address == terminal and previous.bounds == proof:
                return BindingContextProposal("already_reached")
            refusal = "route_projection_refused"
            for preactivation_weighted in (True, False):
                if sweeps == max_sweeps:
                    break
                hidden, used = _project_context_route(receiver, incumbent.latest, column, padding,
                                                      max_sweeps - sweeps,
                                                      preactivation_weighted=preactivation_weighted)
                sweeps += used
                if hidden is None:
                    continue
                try:
                    candidate, _, _ = program_binding_context(receiver, ready, hidden, max_update_norm=cap,
                                                              protect=protect)
                    break
                except ValueError:
                    refusal = "context_fit_refused"
    except BindingCapacityError:
        return BindingContextProposal("opening_incomplete", sweeps)
    except FloatingPointError:
        return BindingContextProposal("coordinate_numerical", sweeps)

    if candidate is None:
        return BindingContextProposal(refusal, sweeps)
    protected = tuple((run.feedback.seed, CorpusProgramExpectation(
        CorpusProgramProbe(tuple(read.address for read in run.reads)), run.reads[-1].bounds,
    )) for run in protect)
    seed = ready.feedback.seed
    status = check_coordinate_replay(receiver, candidate, memory, seed, addresses, proof)
    if status != "published":
        return BindingContextProposal(status, sweeps)
    for start, expectation in protected:
        if not check_coordinate_protection(receiver, candidate, memory, start, expectation.probe.addresses,
                                           expectation.expected):
            return BindingContextProposal("coordinate_protected_changed", sweeps)
    target = CorpusProgramExpectation(CorpusProgramProbe(addresses), proof)
    trial = workspace.begin_coordinate_trial(base, receiver, candidate, seed=seed, target=target,
                                             future_batches=future_batches, max_update_norm=cap, protect=protected)
    return BindingContextProposal("opened", sweeps, target, trial)
