"""Bounded source-only reconciliation of a sealed native continuation bank."""

from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from experiments.memory.linear_feedback import _quantize
from experiments.memory.native_latent_recurrence import NativeLatentModel, NativeSourceMemory, _require_cpu_eval
from experiments.memory.native_publication import _reader_digest
from experiments.memory.source_revision_feed import _encode
from experiments.memory.verified_source_tool import SourceSpan, require_digest, sha256


@dataclass(frozen=True, slots=True)
class ContinuationReplay:
    seed: SourceSpan
    status: str
    improved_checks: int = 0


@dataclass(frozen=True, slots=True)
class _SelectionFrontier:
    query: bytes
    cutoff: int
    selected: tuple[SourceSpan, ...]


@dataclass(frozen=True, slots=True)
class _ReplayTrace:
    states: tuple[bytes, ...]
    clean: bool
    selected: frozenset[SourceSpan]
    complete: bool
    frontiers: tuple[tuple[_SelectionFrontier, _SelectionFrontier], ...] = ()

    def unaffected_by(self, seed):
        return self.complete and seed not in self.selected


@dataclass(frozen=True, slots=True, eq=False)
class _IntactReplay:
    opening: NativeSourceMemory
    confirmation_steps: int
    implementation_sha256: str
    traces: MappingProxyType


@dataclass(frozen=True, slots=True)
class NativeReconciliation:
    memory: NativeSourceMemory
    status: str
    opening_publication_sha256: str
    anchor_count: int
    backbone_calls: int = 0
    row_scores: int = 0
    comparisons: tuple[ContinuationReplay, ...] = ()
    retired: SourceSpan | None = None
    reused_pairs: int = 0
    reused_anchors: int = 0
    frontier_scores: int = 0
    intact_replay: _IntactReplay | None = None


class _ReplayBudgetExceeded(Exception):
    pass


def _implementation_sha256():
    modules = (
        "latent_feedback.py", "linear_feedback.py", "native_latent_recurrence.py",
        "native_publication.py", "native_reconciliation.py", "native_replay_checkpoint.py",
        "selective_continuations.py", "source_revision_feed.py", "verified_source_tool.py",
    )
    return sha256(_encode([(name, sha256(Path(__file__).with_name(name).read_bytes())) for name in modules]))


def _frontier(steps, row_count, source_seal):
    if len(steps) != 1:
        return None
    step, = steps
    receipt = step.receipt
    if (receipt.status not in ("ready", "empty", "competing")
            or receipt.eligible != len(receipt.selected) or receipt.rows_scored != row_count
            or receipt.observed_source_seal != source_seal):
        return None
    return _SelectionFrontier(step.query, step.cutoff, tuple(hit.span for hit in receipt.selected))


def _trace(reader, item, memory, confirmation_steps, *, capture=False):
    lookup = reader._model.get_input_embeddings().weight
    hidden, cache, _ = reader._seed_state(item, lookup)
    states, frontiers, selected = [], [], set()
    derived_count = sum(len(shard.spans) for shard in memory._selective.derived)
    source_count = memory.rows if capture else 0
    clean = complete = True
    for _ in range(confirmation_steps):
        run = reader._continue(hidden, cache, memory, 1)
        hidden, cache = run.hidden, run.cache
        states.append(_quantize(reader._encode_hidden(hidden)).astype(np.int8).tobytes())
        clean &= all(step.receipt.status in ("ready", "empty") for step in (*run.source_steps, *run.derived_steps))
        receipts = run.derived_reads
        derived = _frontier(run.derived_steps, derived_count, memory.seal)
        consistent = derived is not None and receipts == (run.derived_steps[0].receipt,)
        complete &= consistent
        selected.update(hit.span for receipt in receipts for hit in receipt.selected)
        if capture and consistent:
            source = _frontier(run.source_steps, source_count, memory.seal)
            if source is not None:
                frontiers.append((source, derived))
    return _ReplayTrace(tuple(states), clean, frozenset(selected), complete, tuple(frontiers))


def _bank_rows(shards):
    return {span: (key, value) for shard in shards
            for span, key, value in zip(shard.spans, shard.keys, shard.values, strict=True)}


def _bank_delta(before, after):
    before, after = _bank_rows(before), _bank_rows(after)
    shared = before.keys() & after.keys()
    changed = {span for span in shared if any(not np.array_equal(a, b)
               for a, b in zip(before[span], after[span], strict=True))}
    surviving = shared - changed
    if tuple(span for span in before if span in surviving) != tuple(span for span in after if span in surviving):
        return None
    removed = before.keys() - surviving
    arrivals = [key for span, (key, _) in after.items() if span not in surviving]
    return removed, np.stack(arrivals) if arrivals else None


def _arrivals_reach_cutoff(keys, frontier):
    query = np.frombuffer(frontier.query, dtype=np.int8).astype(np.int64)
    return bool(np.any(keys @ query >= frontier.cutoff))


def _can_reuse(trace, deltas, required_steps, score_arrivals):
    if len(trace.frontiers) != required_steps:
        return False
    for pair in trace.frontiers:
        for frontier, (removed, arrivals) in zip(pair, deltas, strict=True):
            if not removed.isdisjoint(frontier.selected):
                return False
            if arrivals is not None and score_arrivals(arrivals, frontier):
                return False
    return True


def _previous_opening(replay, memory, confirmation_steps, implementation):
    if (replay is None or replay.confirmation_steps != confirmation_steps
            or replay.implementation_sha256 != implementation):
        return None
    opening = replay.opening
    if (opening._owner is not memory._owner
            or opening._reader_fingerprint_sha256 != memory._reader_fingerprint_sha256
            or opening._publication_runtime_sha256 != memory._publication_runtime_sha256
            or opening._scope.epoch + 1 != memory._scope.epoch
            or json.loads(memory._scope.descriptor)["parent"] != opening.seal):
        return None
    return opening


def _distance(state, reference):
    left = np.frombuffer(state, dtype=np.int8).astype(np.int16)
    right = np.frombuffer(reference, dtype=np.int8)
    return int(np.abs(left - right).sum(dtype=np.int64))


def _compare(seed, intact, masked, source_only, short_steps, reference_step):
    improved = 0
    reference_index = reference_step - 1
    for index, (active, excluded) in enumerate(zip(intact, masked, strict=True)):
        before, after = active.states, excluded.states
        if active is excluded:
            continue
        if before == after:
            if active.clean != excluded.clean:
                return ContinuationReplay(seed, "unqualified_trace")
            continue
        reference = before[reference_index]
        if not all(reference == state for state in
                   (before[-1], after[reference_index], after[-1])):
            return ContinuationReplay(seed, "endpoint_disagreement")
        if not active.clean or not excluded.clean or not any(reference):
            return ContinuationReplay(seed, "unqualified_trace")
        gains = [_distance(before[step - 1], reference) - _distance(after[step - 1], reference)
                 for step in short_steps]
        if min(gains) < 0:
            return ContinuationReplay(seed, "short_regression")
        if max(gains) == 0:
            continue
        local = source_only(index).states
        if reference in (local[reference_index], local[-1]):
            return ContinuationReplay(seed, "source_only_endpoint")
        improved += sum(gain > 0 for gain in gains)
    return ContinuationReplay(seed, "qualified" if improved else "no_improvement", improved)


def _retire_leaf(reader, memory, seed):
    selective = memory._selective
    masked = reader._without_continuation(memory, seed)
    requeued = {child for child, parent in selective.parents.items() if parent == seed}
    requeued.add(seed)
    attempted = {item: steps for item, steps in selective.attempted.items() if item not in requeued}
    acquired = {item: row for item, row in selective.acquired.items() if item != seed}
    source_order = dict.fromkeys((*selective.priority, *sorted(item.source_id for item in requeued)))
    updated = replace(
        selective, acquired=MappingProxyType(acquired), attempted=MappingProxyType(attempted),
        parents=MappingProxyType({item: parent for item, parent in selective.parents.items() if item not in requeued}),
        forecasts=MappingProxyType({item: check for item, check in selective.forecasts.items() if item != seed}),
        forecast_sets=MappingProxyType({item: check for item, check in selective.forecast_sets.items() if item != seed}),
        priority=reader._pending_groups(source_order, memory._sources, attempted),
        derived=masked._selective.derived,
    )
    return replace(memory, _selective=updated, derived_seeds=tuple(acquired), _publication_seal=None)


def reconcile_native_continuations(reader, memory, *, expected_publication_sha256, short_steps, reference_step,
                                   max_anchors, max_leaves, max_backbone_calls, max_row_scores,
                                   use_frontiers=False, previous_replay=None):
    """Propose one same-pin retirement. Frontier reuse requires deterministic replay."""
    if type(reader) is not NativeLatentModel:
        raise TypeError("native reconciliation requires a native latent reader")
    reader._check_memory(memory)
    require_digest(expected_publication_sha256)
    if (memory._selective is None or memory.omitted_sources or memory._derived_off
            or memory._publication_seal != expected_publication_sha256):
        raise ValueError("native reconciliation requires the expected intact sealed publication")
    if (type(reference_step) is not int or not 2 <= reference_step <= reader._steps
            or type(short_steps) is not tuple or not short_steps
            or any(type(step) is not int or not 1 <= step < reference_step for step in short_steps)
            or short_steps != tuple(sorted(set(short_steps)))):
        raise ValueError("reconciliation requires increasing short steps before the bounded reference step")
    if type(use_frontiers) is not bool:
        raise ValueError("native reconciliation frontier selection must be boolean")
    if previous_replay is not None and (not use_frontiers or type(previous_replay) is not _IntactReplay):
        raise ValueError("previous replay requires a worker checkpoint and frontier reuse")
    limits = max_anchors, max_leaves, max_backbone_calls, max_row_scores
    if any(type(value) is not int or value < 1 for value in limits):
        raise ValueError("native reconciliation requires positive work and coverage bounds")
    _require_cpu_eval(reader._model)

    anchor_count = memory.rows
    if anchor_count > max_anchors:
        return NativeReconciliation(memory, "budget_exceeded", expected_publication_sha256, anchor_count)
    selective = memory._selective
    protected = {parent for child, parent in selective.parents.items() if child in selective.acquired}
    protected.update(parent for check in selective.forecast_sets.values() for parent in check.forecast.parents)
    leaves = tuple(seed for seed in memory.derived_seeds if seed not in protected)
    if not leaves:
        return NativeReconciliation(memory, "no_eligible_leaf", expected_publication_sha256, anchor_count)
    confirmation_steps = reference_step + 1
    implementation = _implementation_sha256() if use_frontiers else None
    opening = _previous_opening(previous_replay, memory, confirmation_steps, implementation)
    if opening is not None and opening.rows > max_anchors:
        opening = None
    derived_rows = len(selective.acquired)
    # Frontier mode reserves a complete intact pass before examining its
    # selections. Exhaustive mode can reserve all mandatory work up front.
    variants = 1 if use_frontiers else len(leaves) + 2
    masked_variants = 0 if use_frontiers else len(leaves)
    opening_calls = anchor_count * variants * (confirmation_steps + 1)
    opening_scores = anchor_count * confirmation_steps * (
        anchor_count * variants + derived_rows + masked_variants * (derived_rows - 1)
    )
    if len(leaves) > max_leaves or (opening is None and (
            opening_calls > max_backbone_calls or opening_scores > max_row_scores)):
        return NativeReconciliation(memory, "budget_exceeded", expected_publication_sha256, anchor_count)
    if _reader_digest(reader) != memory._reader_fingerprint_sha256:
        raise ValueError("native reconciliation requires the unchanged opening reader")

    import torch

    anchors = tuple(item for source in sorted(memory._sources) for item in memory._sources[source])
    calls = scores = reused = reused_anchors = frontier_scores = 0

    def replay(item, view, derived_count, *, capture=False):
        nonlocal calls, scores
        next_calls = calls + confirmation_steps + 1
        next_scores = scores + confirmation_steps * (anchor_count + derived_count)
        if next_calls > max_backbone_calls or next_scores > max_row_scores:
            raise _ReplayBudgetExceeded
        trace = _trace(reader, item, view, confirmation_steps, capture=capture)
        calls, scores = next_calls, next_scores
        return trace

    def score_arrivals(keys, frontier):
        nonlocal scores, frontier_scores
        if scores + len(keys) > max_row_scores:
            raise _ReplayBudgetExceeded
        scores += len(keys)
        frontier_scores += len(keys)
        return _arrivals_reach_cutoff(keys, frontier)

    with reader._forecast_lock, torch.inference_mode():
        if reader._pending_forecast is not None:
            return NativeReconciliation(memory, "pending_forecast", expected_publication_sha256, anchor_count)
        source_bank = reader.without_derived(memory)
        local_traces = {}

        def source_only(index):
            if index not in local_traces:
                local_traces[index] = replay(anchors[index], source_bank, 0)
            return local_traces[index]

        deltas = ()
        prior_rows = {}
        if opening is not None:
            deltas = (_bank_delta(opening._selective.shards.values(), selective.shards.values()),
                      _bank_delta(opening._selective.derived, selective.derived))
            if all(delta is not None for delta in deltas):
                prior_rows = {item.span: item.row for rows in opening._sources.values() for item in rows}
        try:
            intact = []
            for item in anchors:
                prior_row = prior_rows.get(item.span)
                if (prior_row is not None and prior_row.shape == item.row.shape
                        and prior_row.dtype == item.row.dtype and prior_row.tobytes() == item.row.tobytes()):
                    prior_trace = previous_replay.traces.get(item.span)
                    if prior_trace is not None and _can_reuse(prior_trace, deltas, confirmation_steps, score_arrivals):
                        intact.append(prior_trace)
                        reused_anchors += 1
                        continue
                intact.append(replay(item, memory, derived_rows, capture=use_frontiers))
            if not use_frontiers:
                for index in range(anchor_count):
                    source_only(index)
            comparisons = []
            for seed in leaves:
                affected = tuple(index for index, trace in enumerate(intact)
                                 if not use_frontiers or not trace.unaffected_by(seed))
                reused += anchor_count - len(affected)
                masked = list(intact)
                if affected:
                    view = reader._without_continuation(memory, seed)
                    for index in affected:
                        masked[index] = replay(anchors[index], view, derived_rows - 1)
                comparisons.append(_compare(seed, intact, masked, source_only, short_steps, reference_step))
        except _ReplayBudgetExceeded:
            return NativeReconciliation(memory, "budget_exceeded", expected_publication_sha256, anchor_count,
                                        calls, scores, frontier_scores=frontier_scores)
        checkpoint = (_IntactReplay(memory, confirmation_steps, implementation, MappingProxyType(
            {item.span: trace for item, trace in zip(anchors, intact, strict=True)})) if use_frontiers else None)
        evidence = dict(backbone_calls=calls, row_scores=scores, comparisons=tuple(comparisons),
                        reused_pairs=reused, reused_anchors=reused_anchors, frontier_scores=frontier_scores,
                        intact_replay=checkpoint)
        qualified = [row.seed for row in comparisons if row.status == "qualified"]
        if len(qualified) == 1:
            seed, = qualified
            revised = _retire_leaf(reader, memory, seed)
            return NativeReconciliation(revised, "retired", expected_publication_sha256, anchor_count,
                                        retired=seed, **evidence)
        return NativeReconciliation(memory, "ambiguous" if qualified else "keep", expected_publication_sha256,
                                    anchor_count, **evidence)
