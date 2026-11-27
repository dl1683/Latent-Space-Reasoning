"""Bounded source-only reconciliation of a sealed native continuation bank."""

from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from experiments.memory.linear_feedback import _quantize
from experiments.memory.native_latent_recurrence import NativeLatentModel, NativeSourceMemory, _require_cpu_eval
from experiments.memory.native_publication import _reader_digest
from experiments.memory.verified_source_tool import SourceSpan, require_digest


@dataclass(frozen=True, slots=True)
class ContinuationReplay:
    seed: SourceSpan
    status: str
    improved_checks: int = 0


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


@dataclass(frozen=True, slots=True)
class _ReplayTrace:
    states: tuple[np.ndarray, ...]
    clean: bool
    selected: frozenset[SourceSpan]
    complete: bool

    def unaffected_by(self, seed):
        return self.complete and seed not in self.selected


class _ReplayBudgetExceeded(Exception):
    pass


def _trace(reader, item, memory, confirmation_steps):
    import torch

    lookup = reader._model.get_input_embeddings().weight
    encoded = torch.tensor(item.row, dtype=lookup.dtype, device=lookup.device)[None, None, :]
    hidden, cache = reader._backbone(encoded * reader._relay_gain, retain_cache=True)
    states, selected = [], set()
    derived_count = sum(len(shard.spans) for shard in memory._selective.derived)
    clean = complete = True
    for _ in range(confirmation_steps):
        run = reader._continue(hidden, cache, memory, 1)
        hidden, cache = run.hidden, run.cache
        states.append(_quantize(reader._encode_hidden(hidden)).astype(np.int8))
        clean &= all(step.receipt.status in ("ready", "empty") for step in (*run.source_steps, *run.derived_steps))
        receipts = run.derived_reads
        complete &= len(receipts) == 1 and all(
            receipt.status in ("ready", "empty", "competing")
            and receipt.eligible == len(receipt.selected)
            and receipt.rows_scored == derived_count and receipt.observed_source_seal == memory.seal
            for receipt in receipts
        )
        selected.update(hit.span for receipt in receipts for hit in receipt.selected)
    return _ReplayTrace(tuple(states), clean, frozenset(selected), complete)


def _distance(state, reference):
    return int(np.abs(state.astype(np.int16) - reference).sum(dtype=np.int64))


def _compare(seed, intact, masked, source_only, short_steps, reference_step):
    improved = 0
    reference_index = reference_step - 1
    for index, (active, excluded) in enumerate(zip(intact, masked, strict=True)):
        before, after = active.states, excluded.states
        if active is excluded:
            continue
        if all(np.array_equal(left, right) for left, right in zip(before, after, strict=True)):
            if active.clean != excluded.clean:
                return ContinuationReplay(seed, "unqualified_trace")
            continue
        reference = before[reference_index]
        if not all(np.array_equal(reference, state) for state in
                   (before[-1], after[reference_index], after[-1])):
            return ContinuationReplay(seed, "endpoint_disagreement")
        if not active.clean or not excluded.clean or not reference.any():
            return ContinuationReplay(seed, "unqualified_trace")
        gains = [_distance(before[step - 1], reference) - _distance(after[step - 1], reference)
                 for step in short_steps]
        if min(gains) < 0:
            return ContinuationReplay(seed, "short_regression")
        if max(gains) == 0:
            continue
        local = source_only(index).states
        if any(np.array_equal(reference, state) for state in (local[reference_index], local[-1])):
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
                                   use_frontiers=False):
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
    derived_rows = len(selective.acquired)
    # Frontier mode reserves a complete intact pass before examining its
    # selections. Exhaustive mode can reserve all mandatory work up front.
    variants = 1 if use_frontiers else len(leaves) + 2
    masked_variants = 0 if use_frontiers else len(leaves)
    opening_calls = anchor_count * variants * (confirmation_steps + 1)
    opening_scores = anchor_count * confirmation_steps * (
        anchor_count * variants + derived_rows + masked_variants * (derived_rows - 1)
    )
    if len(leaves) > max_leaves or opening_calls > max_backbone_calls or opening_scores > max_row_scores:
        return NativeReconciliation(memory, "budget_exceeded", expected_publication_sha256, anchor_count)
    if _reader_digest(reader) != memory._reader_fingerprint_sha256:
        raise ValueError("native reconciliation requires the unchanged opening reader")

    import torch

    anchors = tuple(item for source in sorted(memory._sources) for item in memory._sources[source])
    calls = scores = reused = 0

    def replay(item, view, derived_count):
        nonlocal calls, scores
        next_calls = calls + confirmation_steps + 1
        next_scores = scores + confirmation_steps * (anchor_count + derived_count)
        if next_calls > max_backbone_calls or next_scores > max_row_scores:
            raise _ReplayBudgetExceeded
        trace = _trace(reader, item, view, confirmation_steps)
        calls, scores = next_calls, next_scores
        return trace

    with reader._forecast_lock, torch.inference_mode():
        if reader._pending_forecast is not None:
            return NativeReconciliation(memory, "pending_forecast", expected_publication_sha256, anchor_count)
        source_bank = reader.without_derived(memory)
        local_traces = {}

        def source_only(index):
            if index not in local_traces:
                local_traces[index] = replay(anchors[index], source_bank, 0)
            return local_traces[index]

        try:
            intact = tuple(replay(item, memory, derived_rows) for item in anchors)
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
                                        calls, scores)
        qualified = [row.seed for row in comparisons if row.status == "qualified"]
        if len(qualified) == 1:
            seed, = qualified
            revised = _retire_leaf(reader, memory, seed)
            return NativeReconciliation(revised, "retired", expected_publication_sha256, anchor_count, calls,
                                        scores, tuple(comparisons), seed, reused)
        return NativeReconciliation(memory, "ambiguous" if qualified else "keep", expected_publication_sha256,
                                    anchor_count, calls, scores, tuple(comparisons), reused_pairs=reused)
