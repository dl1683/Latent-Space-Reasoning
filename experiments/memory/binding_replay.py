"""Freeze and check a binding repair's feedback trajectory across source revisions."""

from hashlib import sha256
import json

import numpy as np

from experiments.memory.binding_feedback import BindingFeedback, BindingFeedbackState
from experiments.memory.latent_feedback import FactoredValue


_INCOMPLETE_REPLAY = frozenset({
    "ambiguous", "below_floor", "unavailable", "capacity", "expansion_capacity", "candidate_limit", "incomplete",
})


def _replay_step(receiver, state):
    # Workers must settle the same trial under the same error policy.
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
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


def replay_binding_repair(trajectory, memory, address, *, expected=None):
    """Compare with an incumbent replay on the same source rows."""
    receiver = trajectory._receiver
    replay = receiver.start(trajectory.feedback.seed, memory)
    steps = len(trajectory.reads)
    try:
        for step, original in enumerate(trajectory.reads, start=1):
            replay = _replay_step(receiver, replay)
            read = replay.reads[-1]
            if read.status in _INCOMPLETE_REPLAY:
                return "replay_incomplete", step
            if step < steps and (read.status, read.address, read.bounds) != (
                original.status, original.address, original.bounds
            ):
                return "prefix_changed", step
    except FloatingPointError:
        return "replay_numerical", step
    if read.status != "resolved" or read.address != address or read.bounds.certain is None:
        return "target_unrepaired", steps
    if expected is not None and (
        read.bounds.certain.strength != expected.certain.strength
        or read.bounds.possible_strength != expected.possible_strength
    ):
        return "target_unrepaired", steps
    if np.array_equal(replay.latest, trajectory.latest):
        return "state_unchanged", steps
    return "repaired", steps


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
    # ASCII JSON aliases an astral code point with two explicit surrogate code units.
    payload = json.dumps(scope, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False)
    payload = payload.encode("utf-8", errors="surrogatepass")
    return sha256(payload).hexdigest()


def freeze_binding_replay(trajectory, memory, targets):
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
    return dict(receiver=_receiver_digest(trajectory._receiver), seed=trajectory.feedback.seed.tolist(), prefix=prefix)


def require_replay_receiver(spec, receiver):
    if _receiver_digest(receiver) != spec["receiver"]:
        raise ValueError("feedback trial receiver differs from the opening receiver")


def check_binding_replay(spec, receiver, incumbent, candidate, addresses, expected):
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
    status, _ = replay_binding_repair(baseline, candidate, addresses[-1], expected=expected)
    return "published" if status == "repaired" else status
