"""Freeze and check a binding repair's feedback trajectory across source revisions."""

from fractions import Fraction
from hashlib import sha256
import json
from math import nextafter

import numpy as np

from experiments.memory.binding_feedback import BindingFeedback, BindingFeedbackState
from experiments.memory.latent_feedback import FactoredValue
from experiments.memory.weight_program import program_weight


_INCOMPLETE_REPLAY = frozenset({
    "ambiguous", "below_floor", "unavailable", "capacity", "expansion_capacity", "candidate_limit", "incomplete",
})
_REPLAY_ERRORS = dict(over="raise", invalid="raise", divide="raise", under="ignore")
_ROUTE_ATOL = 1e-8


class RouteFitRefusal(ValueError):
    """The proposed route column failed its opening bounds or replay."""


def _replay_step(receiver, state):
    # Workers must settle the same trial under the same error policy.
    with np.errstate(**_REPLAY_ERRORS):
        return receiver.step(state)


def failed_binding_prefix(trajectory, addresses):
    """Return the ordered route prefix only for a closed absence at the target."""
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
    if receiver._coordinate_value is not None:
        scope["coordinate_value"] = _array_signature(receiver._coordinate_value)
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


def apply_route_program(receiver, patch):
    before, after, column, weights = patch
    if _receiver_digest(receiver) != before or not 0 <= column < len(receiver._addresses):
        raise ValueError("route program needs the matching incumbent receiver")
    candidate = receiver._with_route_column(column, weights)
    if _receiver_digest(candidate) != after or before == after:
        raise ValueError("route program receiver fingerprint mismatch")
    return candidate


def _require_opening_rank(receiver, state, column):
    scores = state.latest @ receiver._query
    winning = float(scores[column])
    competitor = float(np.delete(scores, column).max(initial=-np.inf))
    if winning <= receiver._route_floor or winning - competitor <= receiver._route_gap:
        raise RouteFitRefusal("route program does not clear the opening score floor and gap")


def route_norm_cap(value):
    message = "route program needs a finite positive weight norm cap"
    if type(value) is not float and type(value) is not int:
        raise ValueError(message)
    try:
        cap = float(value)
    except OverflowError as error:
        raise ValueError(message) from error
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError(message)
    return nextafter(cap, 0.) if cap > value else cap


def validate_route_program(trajectory, patch, max_update_norm, memory, repair_address, later_address):
    max_update_norm = route_norm_cap(max_update_norm)
    receiver = trajectory._receiver
    candidate = apply_route_program(receiver, patch)
    column = patch[2]
    if candidate._addresses[column] != later_address:
        raise ValueError("route program must edit the downstream route column")
    old, new = receiver._query[:, column], candidate._query[:, column]
    # Compare exact squared distances between stored Float64 weights at admission.
    squared_update = sum((Fraction(after) - Fraction(before)) ** 2
                         for before, after in zip(old.tolist(), new.tolist(), strict=True))
    if squared_update > Fraction(max_update_norm) ** 2:
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
    baseline = receiver.start(spec["seed"], incumbent)
    prefix = tuple(tuple(address) for address in spec["prefix"])
    try:
        for address in prefix:
            baseline = _replay_step(receiver, baseline)
            read = baseline.reads[-1]
            if read.status in _INCOMPLETE_REPLAY:
                return "replay_incomplete"
            if read.address != address or read.status != "resolved" or read.bounds.possible_strength is None:
                return "replay_route_changed"
        baseline = _replay_step(receiver, baseline)
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
