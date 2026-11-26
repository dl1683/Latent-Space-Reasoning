"""Freeze and check a binding repair's feedback trajectory across source revisions."""

from bisect import bisect_left
from fractions import Fraction
from hashlib import sha256
import json
from math import nextafter, prod

import numpy as np

from experiments.memory.binding_feedback import BindingFeedback, BindingFeedbackState
from experiments.memory.corpus_bindings import _SourceObservationRefusal
from experiments.memory.latent_feedback import ConditionalTerm, FactoredValue, _separating_hinge
from experiments.memory.weight_program import _frobenius, _remove_span, _row_basis, program_weight


_INCOMPLETE_REPLAY = frozenset({
    "ambiguous", "below_floor", "unavailable", "capacity", "expansion_capacity", "candidate_limit", "incomplete",
})
_REPLAY_ERRORS = dict(over="raise", invalid="raise", divide="raise", under="ignore")
_ROUTE_ATOL = 1e-8
_CONTEXT_RCOND = 1e-12


class RouteFitRefusal(ValueError):
    """The proposed route column failed its opening bounds or replay."""


def _replay_step(receiver, state):
    # Workers must settle the same trial under the same error policy.
    with np.errstate(**_REPLAY_ERRORS):
        return receiver.step(state)


def failed_binding_prefix(trajectory, addresses):
    """Return the ordered route prefix only for a closed absence at the target."""
    if trajectory.role_steps:
        return None
    prefix, last = trajectory.reads[:-1], trajectory.reads[-1]
    if any(read.status != "resolved" or read.bounds.possible_strength is None for read in prefix):
        return None
    address = addresses[-1]
    attempts = last.tried if last.status == "catalog_absent" else (last,)
    if not any(read.status == "resolved" and read.address == address and read.bounds.possible_strength is None
               for read in attempts):
        return None
    earlier = tuple(read.address for read in prefix)
    expected = (*earlier, address) if trajectory._receiver._disjoint_reads else (address,)
    return earlier if addresses == expected else None


def _matches_read(read, address, expected):
    if read.status != "resolved" or read.address != address or read.bounds.certain is None:
        return False
    return expected is None or (read.bounds.certain.strength == expected.certain.strength
                                and read.bounds.possible_strength == expected.possible_strength)


def replay_binding_repair(trajectory, memory, address, *, expected=None, receiver=None):
    """Compare with an incumbent replay on the same source rows."""
    if trajectory.role_steps:
        return "replay_incomplete", 0, None
    if receiver is None:
        receiver = trajectory._receiver
    replay = receiver.start(trajectory.feedback.seed, memory)
    steps = len(trajectory.reads)
    try:
        for step, original in enumerate(trajectory.reads, start=1):
            replay = _replay_step(receiver, replay)
            read = replay.reads[-1]
            if read.status in _INCOMPLETE_REPLAY:
                return "replay_incomplete", step, None
            if step < steps and (read.status, read.address, read.bounds) != (
                original.status, original.address, original.bounds
            ):
                return "prefix_changed", step, None
    except FloatingPointError:
        return "replay_numerical", step, None
    if not _matches_read(read, address, expected):
        return "target_unrepaired", steps, None
    if np.array_equal(replay.latest, trajectory.latest):
        return "state_unchanged", steps, None
    return "repaired", steps, replay


def _array_signature(array):
    if array is None:
        return None
    values = np.asarray(array, dtype="<f8", order="C")
    return values.shape, sha256(memoryview(values).cast("B")).hexdigest()


def _value_signature(value):
    if type(value) is FactoredValue:
        return "factored", _array_signature(value._left), _array_signature(value._right)
    if type(value) is not np.ndarray:
        raise ValueError("feedback trials require built-in dense or factored value maps")
    return "dense", _array_signature(value)


def _receiver_digest(receiver):
    if type(receiver) is not BindingFeedback:
        raise ValueError("feedback trials require a binding feedback receiver")
    kernel, expansion = receiver._kernel, receiver._expansion
    encoder, predicates, minimum, gap = receiver._contract
    scope = dict(
        version=1, frame=(encoder, sorted(predicates), minimum, gap, _array_signature(receiver._anchors)),
        routes=(receiver._addresses, _array_signature(receiver._query), _array_signature(receiver._observation),
                receiver._route_floor, receiver._route_gap, receiver._work_limit, receiver._disjoint_reads,
                receiver._route_trials, receiver._reuse_matching_failures, receiver._reuse_prefix_matching,
                receiver._reuse_weighted_matching),
        expansion=None if expansion is None else (
            expansion.nodes, _array_signature(expansion.anchor_query), _array_signature(expansion.node_query),
            expansion.work_limit, expansion.projection,
        ),
        kernel=(kernel._max_steps, _array_signature(kernel._query), _array_signature(kernel._carry),
                _array_signature(kernel._observation_query), _array_signature(kernel._observation_key),
                _value_signature(kernel._observation_value), _array_signature(kernel._feedback_key),
                _value_signature(kernel._feedback_value),
                tuple((*map(_array_signature, terms[:3]), terms[3]) for terms in kernel._conditional)),
    )
    if receiver._source_observation_limit is not None:
        scope["source_observation_limit"] = receiver._source_observation_limit
    if receiver._coordinate_value is not None:
        scope["coordinate_value"] = _array_signature(receiver._coordinate_value)
    if receiver._coordinate_terms:
        scope["coordinate_terms"] = tuple((*map(_array_signature, term[:3]), term[3])
                                          for term in receiver._coordinate_terms)
    # ASCII JSON aliases an astral code point with two explicit surrogate code units.
    payload = json.dumps(scope, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False)
    payload = payload.encode("utf-8", errors="surrogatepass")
    return sha256(payload).hexdigest()


def program_binding_route(incumbent, repaired, address, *, margin, max_update_norm):
    """Fit one fixed route column to the repaired opening state."""
    receiver = incumbent._receiver
    column = receiver._addresses.index(address)
    with np.errstate(**_REPLAY_ERRORS):
        current = repaired.latest @ receiver._query
        competitors = np.delete(current, column)
        threshold = max(receiver._route_floor, float(competitors.max(initial=-np.inf)) + receiver._route_gap)
        target = threshold + margin
        if not np.isfinite(target) or target <= threshold:
            raise ValueError("the requested route margin cannot be represented")
        if current[column] >= target:
            raise ValueError("the opening readout already has the requested margin")
        protected = (incumbent.feedback.seed, *incumbent.feedback.history)
        edit = program_weight(
            receiver._query[:, column:column + 1], repaired.latest[None, :], [[target]],
            preserve_inputs=protected, max_update_norm=max_update_norm, atol=_ROUTE_ATOL,
        )
        weights = tuple(edit.weight[:, 0].tolist())
        changed = receiver._with_route_column(column, weights)
        _require_opening_rank(changed, repaired, column)
        before, after = _receiver_digest(receiver), _receiver_digest(changed)
    return before, after, column, weights


def apply_receiver_program(receiver, patch, *, programs=()):
    before, after, column, weights = patch
    if _receiver_digest(receiver) != before:
        raise ValueError("receiver program needs the matching incumbent receiver")
    if column == "context":
        candidate = receiver._with_coordinate_term(ConditionalTerm(*weights))
    elif column == "retire_context":
        candidate = receiver._without_coordinate_term(weights[0])
    elif column == "restore_context":
        archive = context_archive(receiver, programs, weights)
        candidate = receiver._with_coordinate_term(ConditionalTerm(*archive[3:]), position=archive[2])
    elif column is None:
        candidate = receiver._with_coordinate_value(weights)
    elif 0 <= column < len(receiver._addresses):
        candidate = receiver._with_route_column(column, weights)
    else:
        raise ValueError("route program column is outside the receiver")
    if _receiver_digest(candidate) != after or before == after:
        raise ValueError("programmed receiver fingerprint mismatch")
    return candidate


def archived_context_terms(receiver, programs):
    """Return eligible retirements with their chronological insertion positions and factors."""
    initial = len(receiver._coordinate_terms)
    for _, _, kind, _ in programs:
        initial += (kind == "retire_context") - (kind in ("context", "restore_context"))
    if initial < 0:
        raise ValueError("context history is inconsistent with the installed receiver")
    live, factors, retired = list(range(-initial, 0)), {}, {}
    for ordinal, (_, _, kind, weights) in enumerate(programs):
        if kind == "context":
            live.append(ordinal)
            factors[ordinal] = weights
        elif kind == "retire_context":
            retired[ordinal] = live.pop(weights[0])
        elif kind == "restore_context":
            retirement, expected_origin = weights
            origin = retired.pop(retirement, None)
            if origin != expected_origin or origin not in factors:
                raise ValueError("restored term does not match its journal ancestry")
            live.insert(bisect_left(live, origin), origin)
    return tuple((retirement, origin, bisect_left(live, origin), *factors[origin])
                 for retirement, origin in reversed(retired.items()) if origin in factors)


def context_archive(receiver, programs, reference):
    archive = next((item for item in archived_context_terms(receiver, programs) if item[:2] == reference), None)
    if archive is None:
        raise ValueError("context restoration does not reference a dormant published term")
    if context_term_is_live(receiver, archive[3:]):
        raise ValueError("context restoration would duplicate an identical live term")
    return archive


def context_term_is_live(receiver, factors):
    return any(all(np.array_equal(expected, actual) for expected, actual in zip(factors, live, strict=True))
               for live in receiver._coordinate_terms)


def _clears_opening_rank(receiver, hidden, column):
    scores = hidden @ receiver._query
    winning = float(scores[column])
    competitor = float(np.delete(scores, column).max(initial=-np.inf))
    return winning > receiver._route_floor and winning - competitor > receiver._route_gap


def _require_opening_rank(receiver, state, column):
    if not _clears_opening_rank(receiver, state.latest, column):
        raise RouteFitRefusal("route program does not clear the opening score floor and gap")


def route_norm_cap(value):
    message = "weight program needs a finite positive weight norm cap"
    if type(value) is not float and type(value) is not int:
        raise ValueError(message)
    try:
        cap = float(value)
    except OverflowError as error:
        raise ValueError(message) from error
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError(message)
    return nextafter(cap, 0.) if cap > value else cap


def _within_exact_norm(old, new, cap):
    # Compare exact squared distances between stored Float64 weights at admission.
    squared_update = sum((Fraction(after) - Fraction(before)) ** 2
                         for before, after in zip(old.ravel().tolist(), new.ravel().tolist(), strict=True))
    return squared_update <= Fraction(cap) ** 2


def _coordinate_weights(receiver):
    weight = receiver._coordinate_value
    return np.zeros((receiver._anchors.shape[1], receiver._kernel._width)) if weight is None else weight


def freeze_coordinate_program(receiver, candidate, cap):
    before, after = _receiver_digest(receiver), _receiver_digest(candidate)
    if len(candidate._coordinate_terms) == len(receiver._coordinate_terms) + 1:
        state_axis, anchor_axis, output, threshold = candidate._coordinate_terms[-1]
        term = ConditionalTerm(state_axis, anchor_axis, output, threshold)
        if not any(term.output) or not _within_context_norm(term, cap):
            raise ValueError("coordinate context has no change or exceeds its exact update norm cap")
        factors = tuple(tuple(vector.tolist()) for vector in (state_axis, anchor_axis, output))
        patch = before, after, "context", (*factors, threshold)
        apply_receiver_program(receiver, patch)
        return patch
    weights = _coordinate_weights(candidate)
    if np.array_equal(_coordinate_weights(receiver), weights):
        raise ValueError("coordinate program needs a changed projection")
    patch = before, after, None, tuple(tuple(row) for row in weights.tolist())
    apply_receiver_program(receiver, patch)
    if not _within_exact_norm(_coordinate_weights(receiver), weights, cap):
        raise ValueError("coordinate program exceeds its exact weight norm cap")
    return patch


def _same_binding_read(left, right):
    return (left.status, left.address, left.bounds) == (right.status, right.address, right.bounds)


def _replay_coordinate_prefix(candidate, state, atol, *, memory=None):
    if state.role_steps:
        raise ValueError("guarded replay requires a trajectory without role continuations")
    replay = candidate.start(state.feedback.seed, state.memory if memory is None else memory)
    for read, hidden in zip(state.reads, state.feedback.history, strict=True):
        replay = candidate.step(replay)
        if not _same_binding_read(replay.reads[-1], read) or np.max(np.abs(replay.latest - hidden)) > atol:
            raise ValueError("coordinate program changes a protected trajectory")
    return replay


def _coordinate_fit_step(receiver, state, desired):
    desired = receiver._kernel._vector(desired)
    if np.any(np.abs(desired) >= 1.):
        raise ValueError("target coordinates must be strictly inside tanh saturation")
    if len(state.reads) >= receiver._kernel._max_steps:
        raise ValueError("coordinate programming example has no step remaining")
    read = receiver._next_read(state)
    if read.status != "resolved" or read.bounds.certain is None or not read.bounds.is_exact:
        raise ValueError("coordinate programming needs an exact supported next read")
    return read, desired


def _check_coordinate_fit(candidate, state, read, desired, atol):
    replay = candidate.step(_replay_coordinate_prefix(candidate, state, atol))
    if not _same_binding_read(replay.reads[-1], read) or np.max(np.abs(replay.latest - desired)) > atol:
        raise ValueError("coordinate program misses its next read or hidden target in replay")
    return replay


def _within_context_norm(term, cap):
    factors = (term.state_axis, term.observation_axis, term.output)
    squared_norm = prod(sum(Fraction(float(value)) ** 2 for value in factor) for factor in factors)
    return squared_norm <= Fraction(cap) ** 2


def _fit_context_term(state, read, desired, residual, anchor_axis, protected_rows, protect, cap, atol):
    receiver = state._receiver
    anchor = receiver._anchors[read.address[1]]
    axis, threshold = _separating_hinge(state.latest, protected_rows, _CONTEXT_RCOND)
    activation = (float(state.latest @ axis) - threshold) * float(anchor @ anchor_axis)
    term = ConditionalTerm(axis, anchor_axis, residual / activation, threshold)
    if not any(term.output) or not _within_context_norm(term, cap):
        raise ValueError("coordinate context has no change or exceeds its exact update norm cap")
    candidate = receiver._with_coordinate_term(term)
    replay = _check_coordinate_fit(candidate, state, read, desired, atol)
    for run in protect:
        _replay_coordinate_prefix(candidate, run, atol)
    return candidate, replay, term


def program_binding_context(receiver, state, desired, *, max_update_norm, protect=(), atol=1e-8):
    """Fit a hinged coordinate interaction and replay the target and protected runs."""
    protect = tuple(protect)
    if type(receiver) is not BindingFeedback or any(type(run) is not BindingFeedbackState
            or run._receiver is not receiver for run in (state, *protect)):
        raise ValueError("coordinate programming states must come from the incumbent receiver")
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("coordinate programming needs a finite positive tolerance")
    cap = route_norm_cap(max_update_norm)
    protected = []
    for run in (state, *protect):
        predecessors = (run.feedback.seed, *run.feedback.history)[:-1]
        for hidden, read in zip(predecessors, run.reads, strict=True):
            if read.status == "resolved" and read.bounds.possible_strength is not None:
                protected.append((hidden, read.address[1]))
    with np.errstate(**_REPLAY_ERRORS):
        read, desired = _coordinate_fit_step(receiver, state, desired)
        diagnostics = {}
        receiver._advance_read(state, read, diagnostics=diagnostics)
        coordinate = read.address[1]
        anchor = receiver._anchors[coordinate]
        residual = np.arctanh(desired) - diagnostics["preactivation"]
        try:
            return _fit_context_term(state, read, desired, residual, anchor,
                                     [hidden for hidden, _ in protected], protect, cap, atol)
        except (ValueError, FloatingPointError):
            other_coordinates = sorted({index for _, index in protected if index != coordinate})
            if not other_coordinates:
                raise
            basis = _row_basis(receiver._anchors[other_coordinates], _CONTEXT_RCOND)
            projected = _remove_span(anchor, basis)
            length = _frobenius(projected)
            if length <= _CONTEXT_RCOND:
                raise
            anchor_axis = projected / length
            return _fit_context_term(state, read, desired, residual, anchor_axis,
                                     [hidden for hidden, index in protected if index == coordinate], protect, cap, atol)


def program_binding_coordinates(receiver, examples, *, max_update_norm, protect=(), atol=1e-8):
    """Fit shared anchor-to-state weights and replay the normal decisions on their pins.

    Examples are (ready state, desired next hidden row) pairs. Protected states
    retain their complete generated prefixes. Return the candidate, example
    replays, and weight edit for inspection.
    """
    if type(receiver) is not BindingFeedback:
        raise ValueError("coordinate programming needs a binding feedback receiver")
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("coordinate programming needs a finite positive tolerance")
    cap = route_norm_cap(max_update_norm)
    examples, protect = tuple(examples), tuple(protect)
    if not examples:
        raise ValueError("coordinate programming needs at least one example")
    states = tuple(state for state, _ in examples) + protect
    if any(type(state) is not BindingFeedbackState or state._receiver is not receiver for state in states):
        raise ValueError("coordinate programming states must come from the incumbent receiver")
    protected = sorted({read.address[1] for state in states for read in state.reads
                        if read.status == "resolved" and read.bounds.possible_strength is not None})
    old = _coordinate_weights(receiver)
    unprojected = receiver._with_coordinate_value(None)
    inputs, values, steps = [], [], []
    with np.errstate(**_REPLAY_ERRORS):
        for state, desired in examples:
            read, desired = _coordinate_fit_step(receiver, state, desired)
            diagnostics = {}
            unprojected._advance_read(state, read, diagnostics=diagnostics)
            inputs.append(receiver._anchors[read.address[1]])
            values.append(np.arctanh(desired) - diagnostics["preactivation"])
            steps.append((state, read, desired))

        edit = program_weight(old, inputs, values, preserve_inputs=receiver._anchors[protected],
                              max_update_norm=cap, atol=atol)
        if not _within_exact_norm(old, edit.weight, cap):
            raise ValueError("coordinate program exceeds its exact weight norm cap")
        candidate = receiver._with_coordinate_value(edit.weight)
        replays = []
        for state, read, desired in steps:
            replays.append(_check_coordinate_fit(candidate, state, read, desired, atol))
        for state in protect:
            _replay_coordinate_prefix(candidate, state, atol)
    return candidate, tuple(replays), edit


def validate_route_program(trajectory, patch, max_update_norm, memory, repair_address, later_address):
    max_update_norm = route_norm_cap(max_update_norm)
    receiver = trajectory._receiver
    candidate = apply_receiver_program(receiver, patch)
    column = patch[2]
    if candidate._addresses[column] != later_address:
        raise ValueError("route program must edit the downstream route column")
    old, new = receiver._query[:, column], candidate._query[:, column]
    if not _within_exact_norm(old, new, max_update_norm):
        raise RouteFitRefusal("route program exceeds its weight norm cap")
    protected = np.asarray((trajectory.feedback.seed, *trajectory.feedback.history))
    with np.errstate(**_REPLAY_ERRORS):
        if np.max(np.abs(protected @ new - protected @ old)) > _ROUTE_ATOL:
            raise RouteFitRefusal("route program changes a protected opening score")
        status, _, repaired = replay_binding_repair(trajectory, memory, repair_address, receiver=candidate)
        if status != "repaired":
            raise RouteFitRefusal("route program does not preserve the opening repair")
        _require_opening_rank(candidate, repaired, column)
    return max_update_norm


def coordinate_proof_gain(incumbent, terminal_node, expected, minimum_gain):
    """Compare a certain target grade to the incumbent's possible grade in its family."""
    possible = 0.
    if incumbent.status == "resolved" and incumbent.address[0] == terminal_node:
        possible = incumbent.bounds.possible_strength or 0.
    return Fraction(float(expected.certain.strength)) - Fraction(float(possible)) >= Fraction(minimum_gain)


def supported_coordinate_continuation(receiver, memory, seed, steps):
    """Read a complete ordinary continuation through the terminal proof."""
    state = receiver.start(seed, memory)
    addresses = []
    try:
        with np.errstate(**_REPLAY_ERRORS):
            for step in range(steps):
                if step + 1 == steps:
                    read = receiver._next_read(state)
                else:
                    state = receiver.step(state)
                    read = state.reads[-1]
                if read.status != "resolved" or read.bounds.certain is None or not read.bounds.is_exact:
                    return None
                addresses.append(read.address)
    except FloatingPointError:
        return None
    return tuple(addresses), read.bounds


def check_coordinate_replay(receiver, candidate, memory, seed, addresses, expected, *, minimum_gain=None):
    """Judge the changed transition and its state intervention on the same source pin."""
    try:
        prior, revised = receiver.start(seed, memory), candidate.start(seed, memory)
        with np.errstate(**_REPLAY_ERRORS):
            for address in addresses[:-1]:
                prior, revised = receiver.step(prior), candidate.step(revised)
                old, new = prior.reads[-1], revised.reads[-1]
                if old.status in _INCOMPLETE_REPLAY or new.status in _INCOMPLETE_REPLAY:
                    return "coordinate_incomplete"
                if (new.status != "resolved" or new.address != address or new.bounds.certain is None
                        or not new.bounds.is_exact or not _same_binding_read(old, new)):
                    return "coordinate_prefix_changed"
            terminal = candidate._next_read(revised)
            incumbent = receiver._next_read(prior)
            substitute = candidate._intervene_latest(revised, prior.latest)
            control = candidate._next_read(substitute)
    except _SourceObservationRefusal as refusal:
        return refusal.status
    except FloatingPointError:
        return "coordinate_numerical"
    if any(read.status in _INCOMPLETE_REPLAY for read in (terminal, incumbent, control)):
        return "coordinate_incomplete"
    if terminal.status != "resolved" or terminal.address != addresses[-1] or terminal.bounds != expected:
        return "coordinate_target_mismatch"
    if _same_binding_read(terminal, incumbent):
        return "coordinate_incumbent_matches"
    if _same_binding_read(terminal, control):
        return "coordinate_state_unused"
    if minimum_gain is not None:
        if not _same_binding_read(control, incumbent):
            return "coordinate_rollback_changed"
        if not coordinate_proof_gain(incumbent, terminal.address[0], expected, minimum_gain):
            return "coordinate_gain_lost"
    return "published"


def check_coordinate_protection(receiver, candidate, memory, seed, addresses, expected):
    """Require the already useful ordinary path on both receivers at adjudication."""
    try:
        with np.errstate(**_REPLAY_ERRORS):
            for model in (receiver, candidate):
                state = model.start(seed, memory)
                for address in addresses[:-1]:
                    state = model.step(state)
                    read = state.reads[-1]
                    if (read.status != "resolved" or read.address != address or read.bounds.certain is None
                            or not read.bounds.is_exact):
                        return False
                terminal = model._next_read(state)
                if terminal.status != "resolved" or terminal.address != addresses[-1] or terminal.bounds != expected:
                    return False
    except FloatingPointError:
        return False
    return True


def check_context_retirement(receiver, candidate, memory, probes, atol):
    """Require equal supported ordinary traces on the same current source pin."""
    try:
        with np.errstate(**_REPLAY_ERRORS):
            for seed, steps in probes:
                prior, revised = receiver.start(seed, memory), candidate.start(seed, memory)
                for _ in range(steps):
                    prior, revised = receiver.step(prior), candidate.step(revised)
                    old, new = prior.reads[-1], revised.reads[-1]
                    if old.status != "resolved" or old.bounds.certain is None or not old.bounds.is_exact:
                        return "retirement_incomplete"
                    if not _same_binding_read(old, new):
                        return "retirement_read_changed"
                    if np.max(np.abs(prior.latest - revised.latest)) > atol:
                        return "retirement_state_changed"
    except _SourceObservationRefusal as refusal:
        return refusal.status
    except FloatingPointError:
        return "retirement_numerical"
    return "published"


def freeze_binding_replay(trajectory, memory, targets, downstream=None):
    if type(trajectory) is not BindingFeedbackState or trajectory.memory is not memory or not trajectory.reads:
        raise ValueError("feedback trials require a trajectory on the opening corpus pin")
    if len(targets) != 1:
        raise ValueError("feedback trials require one unexcluded certain target")
    (addresses, excluded), witness, _ = targets[0]
    if excluded or witness is None:
        raise ValueError("feedback trials require one unexcluded certain target")
    prefix = failed_binding_prefix(trajectory, addresses)
    if prefix is None:
        raise ValueError("feedback trial target must identify the failed read in the opening trajectory")
    if downstream is not None:
        (later, excluded), witness, _ = downstream
        receiver = trajectory._receiver
        earlier = addresses if receiver._disjoint_reads else ()
        if excluded or witness is None or later[:-1] != earlier:
            raise ValueError("the downstream target must identify the next unexcluded certain read")
        if len(trajectory.reads) >= receiver._kernel._max_steps:
            raise ValueError("the downstream target needs a remaining feedback step")
    return dict(receiver=_receiver_digest(trajectory._receiver), seed=trajectory.feedback.seed.tolist(), prefix=prefix)


def require_replay_receiver(spec, receiver):
    if _receiver_digest(receiver) != spec["receiver"]:
        raise ValueError("feedback trial receiver differs from the opening receiver")


def _check_downstream(receiver, incumbent, repaired, address, expected):
    candidate_receiver = repaired._receiver
    try:
        with np.errstate(**_REPLAY_ERRORS):
            read = candidate_receiver._next_read(repaired)
            if read.status in _INCOMPLETE_REPLAY:
                return "downstream_incomplete"
            if not _matches_read(read, address, expected):
                return "downstream_mismatch"
            prior = receiver._next_read(incumbent)
            if prior.status in _INCOMPLETE_REPLAY:
                return "downstream_incomplete"
            if _matches_read(prior, address, expected):
                return "downstream_incumbent_matches"
            control = candidate_receiver._intervene_latest(repaired, incumbent.latest)
            substitute = candidate_receiver._next_read(control)
            if substitute.status in _INCOMPLETE_REPLAY:
                return "downstream_incomplete"
            if _matches_read(substitute, address, expected):
                return "downstream_state_unused"
            if candidate_receiver is not receiver:
                restored = receiver._read_with_route_map(repaired)
                if restored.status in _INCOMPLETE_REPLAY:
                    return "downstream_incomplete"
                if _matches_read(restored, address, expected):
                    return "downstream_weights_unused"
    except FloatingPointError:
        return "downstream_numerical"
    return "published"


def check_binding_replay(spec, receiver, incumbent, candidate, addresses, expected, *, downstream=None,
                         candidate_receiver=None):
    """Require the same failed route and a causal repair on the checked future tip."""
    prefix = tuple(tuple(address) for address in spec["prefix"])
    try:
        baseline = receiver.start(spec["seed"], incumbent)
        for address in prefix:
            baseline = _replay_step(receiver, baseline)
            read = baseline.reads[-1]
            if read.status in _INCOMPLETE_REPLAY:
                return "replay_incomplete"
            if read.address != address or read.status != "resolved" or read.bounds.possible_strength is None:
                return "replay_route_changed"
        baseline = _replay_step(receiver, baseline)
    except _SourceObservationRefusal as refusal:
        return refusal.status
    except FloatingPointError:
        return "replay_numerical"
    if baseline.reads[-1].status in _INCOMPLETE_REPLAY:
        return "replay_incomplete"
    if failed_binding_prefix(baseline, addresses) != prefix:
        return "replay_route_changed"
    status, _, repaired = replay_binding_repair(baseline, candidate, addresses[-1], expected=expected,
                                                receiver=candidate_receiver)
    if status != "repaired":
        return status
    if downstream is None:
        return "published"
    return _check_downstream(receiver, baseline, repaired, *downstream)
