"""Bounded source-only retries for native continuations that only read themselves."""

from copy import copy
from dataclasses import dataclass, replace
from functools import partial
from itertools import islice
from math import isfinite
from types import MappingProxyType

import numpy as np

from experiments.memory.latent_feedback import _frozen
from experiments.memory.linear_feedback import _quantize
from experiments.memory.native_latent_recurrence import NativeLatentModel, _Continuation, _require_cpu_eval
from experiments.memory.native_publication import SourceExclusionReceipt
from experiments.memory.native_workspace import NativeWorkspacePin
from experiments.memory.verified_source_tool import SourceSpan


@dataclass(frozen=True, slots=True)
class NativeCrossSourceAcquisition:
    pin: NativeWorkspacePin
    tried: int
    acquired: tuple[SourceSpan, ...]


class _SteeringRefused(Exception):
    pass


@dataclass(frozen=True, slots=True)
class _DeflatedSourceReads:
    selector: object
    source: str
    basis: np.ndarray
    max_relative_shift: float
    withhold: bool = False

    def read(self, latent, shards):
        if not shards:
            return self.selector.read(latent, shards)

        # Pulling the source key span back through Q leaves downstream
        # publication full-bank receipts in the existing Q8 key coordinates.
        direction = latent - self.basis @ (self.basis.T @ latent)
        projected = direction @ self.selector._kernel._query
        scale = max(1.0, float(np.max(np.abs(direction))), float(np.max(np.abs(projected))))
        direction = direction / scale
        if np.linalg.norm(direction - latent) > self.max_relative_shift * np.linalg.norm(latent):
            raise _SteeringRefused
        value, mass, step = self.selector.read(direction, shards)
        if any(hit.span.source_id == self.source for hit in step.receipt.selected):
            raise _SteeringRefused
        if self.withhold and value is not None:
            value = np.zeros_like(value)
        return value, mass, step


@dataclass(frozen=True, slots=True)
class _ExcludedSourceReads:
    selector: object
    source: str
    withhold: bool = False

    def read(self, latent, shards):
        if not shards:
            return self.selector.read(latent, shards)
        outside = tuple(shard for shard in shards if shard.spans[0].source_id != self.source)
        value, mass, step = self.selector.read(latent, outside)
        if self.withhold and value is not None:
            value = np.zeros_like(value)
        receipt = step.receipt
        scope = SourceExclusionReceipt(receipt.status, receipt.selected, receipt.eligible, receipt.rows_scored,
                                       receipt.observed_source_seal, self.source)
        return value, mass, replace(step, receipt=scope)


def _deflated_policy(selector, source, own_keys, *, max_relative_shift):
    left, singular, _ = np.linalg.svd(selector._kernel._query @ own_keys.T, full_matrices=False)
    keep = singular > singular[0] / 1024
    return (_DeflatedSourceReads(selector, source, left[:, keep], max_relative_shift) if np.any(keep) else None)


def _exclusion_policy(selector, source, own_keys):
    return _ExcludedSourceReads(selector, source)


def _self_only(steps, source):
    return (any(step.applied for step in steps)
            and all(step.receipt.status in ("ready", "empty") for step in steps)
            and all(hit.span.source_id == source for step in steps for hit in step.receipt.selected))


def retry_cross_source_acquisition(workspace, reader, pin, *, max_seeds, max_source_rows,
                                   max_relative_shift):
    """Reopen self-only misses with a source-deflated query and publish actual changes."""
    if (type(max_relative_shift) not in (int, float) or not isfinite(max_relative_shift)
            or not 0 < max_relative_shift <= 1):
        raise ValueError("cross-source acquisition requires a positive relative shift limit at most one")
    policy = partial(_deflated_policy, max_relative_shift=max_relative_shift)
    return _retry(workspace, reader, pin, max_seeds, max_source_rows, policy)


def retry_excluding_seed_source(workspace, reader, pin, *, max_seeds, max_source_rows):
    """Reopen self-only misses without their own source in the selection bank."""
    return _retry(workspace, reader, pin, max_seeds, max_source_rows, _exclusion_policy)


def _retry(workspace, reader, pin, max_seeds, max_source_rows, make_policy):
    import torch

    if (not isinstance(reader, NativeLatentModel) or type(max_seeds) is not int or max_seeds <= 0
            or type(max_source_rows) is not int or max_source_rows <= 0):
        raise ValueError("cross-source acquisition requires finite positive limits")
    _require_cpu_eval(reader._model)
    memory = workspace._opening(reader, pin)
    selective = memory._selective
    if selective is None:
        raise ValueError("cross-source acquisition needs a selective publication")

    candidates = (
        item for source, items in memory._sources.items() if len(items) <= max_source_rows
        for item in items if item.span not in selective.acquired and item.span not in selective.parents
        and _self_only(selective.attempted.get(item.span, ()), source)
    )
    source_only = replace(memory, _selective=replace(selective, derived=()))
    worker = copy(reader)
    lookup = reader._model.get_input_embeddings().weight
    acquired, attempted = dict(selective.acquired), dict(selective.attempted)
    installed = []
    tried = 0
    with torch.inference_mode():
        for item in islice(candidates, max_seeds):
            tried += 1
            source = item.span.source_id
            address, baseline = reader._run_seed(item, source_only, lookup)
            if not _self_only(baseline.source_steps, source):
                continue
            policy = make_policy(reader._selector, source, selective.shards[source].keys)
            if policy is None:
                continue
            worker._selector = policy
            try:
                changed_address, run = worker._run_seed(item, source_only, lookup)
            except _SteeringRefused:
                continue
            value = reader._encode_hidden(run.hidden)
            if (not reader._installable(run.source_steps, source)
                    or not np.array_equal(address, changed_address)
                    or np.array_equal(_quantize(value), _quantize(reader._encode_hidden(baseline.hidden)))):
                continue
            # Keep score mass and selection fixed until the first changed
            # source input. The recurrence after that intervention stays live.
            worker._selector = replace(policy, withhold=True)
            try:
                held_address, held = worker._run_seed(item, source_only, lookup)
            except _SteeringRefused:
                continue
            first = next(index for index, step in enumerate(run.source_steps) if step.applied)
            if (not np.array_equal(address, held_address)
                    or run.source_steps[:first] != held.source_steps[:first]
                    or replace(run.source_steps[first], applied=False) != held.source_steps[first]
                    or np.array_equal(_quantize(value), _quantize(reader._encode_hidden(held.hidden)))):
                continue
            acquired[item.span] = _Continuation(_frozen(address), _frozen(value))
            attempted[item.span] = run.source_steps
            installed.append(item.span)

    if not installed:
        workspace._expect(pin, workspace.head())
        return NativeCrossSourceAcquisition(pin, tried, ())
    world = replace(selective, acquired=MappingProxyType(acquired), attempted=MappingProxyType(attempted),
                    derived=reader._derived_rows(acquired))
    updated = replace(memory, _selective=world, derived_seeds=tuple(acquired), _publication_seal=None)
    successor = workspace._commit(reader, updated, pin, operation="maintain",
                                  opening_reader_sha256=memory._reader_fingerprint_sha256)
    return NativeCrossSourceAcquisition(successor, tried, tuple(installed))


def archived_self_route_probe():
    """Score frozen Q8 worker steps with no backbone or evaluator access."""
    from fractions import Fraction
    from json import loads
    from struct import unpack_from

    from experiments.memory.archived_native_span_control import (
        _ARCHIVE, _INDEX_DIGEST, _LEVELS, _WIDTH, _dot, _pinned_bytes,
    )

    index = loads(_pinned_bytes(_ARCHIVE / "publication_index.json", _INDEX_DIGEST))
    if index["runtime"]["kernel"] != "identity-query-key-value-feedback":
        raise ValueError("archived source-routing probe expects the original identity grid")
    results = {}
    for pin in index["pins"]:
        directory = _ARCHIVE / "publications" / pin["pin"]
        header = loads(_pinned_bytes(directory / "manifest.json", pin["publication_seal"]))
        payload = _pinned_bytes(directory / "rows.bin", header["rows_sha256"])
        keys, offset = [], 0
        for _, spans in header["documents"]:
            if len(spans) != 1:
                raise ValueError("archived probe expects one key per source")
            values = unpack_from(f"<{_WIDTH}d", payload, offset)
            key = unpack_from(f"{_WIDTH}b", payload, offset + 8 * _WIDTH)
            if key != tuple(round(x * _LEVELS) for x in values):
                raise ValueError("archived source key and row disagree")
            keys.append(key)
            offset += 9 * _WIDTH

        counts = dict(attempts=len(header["attempted"]), pending=len(header["queue"]),
                      acquired=len(header["derived"]), self_only_steps=0, mask_unique_other=0,
                      unscaled_out_of_grid=0, deflated_unique_other=0)
        for seed, steps in header["attempted"]:
            if seed in header["derived"]:
                continue
            for status, _, scored, hits, query_hex, _, applied, _ in steps:
                query = unpack_from(f"{_WIDTH}b", bytes.fromhex(query_hex))
                if (status != "ready" or not applied or scored != len(keys) or not hits
                        or any(source != seed or _dot(query, keys[source]) != score for source, score in hits)):
                    raise ValueError("archived rejection is not a verified self-only step")
                counts["self_only_steps"] += 1
                other = sorted(_dot(query, key) for j, key in enumerate(keys) if j != seed)
                counts["mask_unique_other"] += int(other[-1] > max(0, other[-2]))
                own = keys[seed]
                component = Fraction(_dot(query, own), _dot(own, own))
                residual = tuple(x - component * y for x, y in zip(query, own, strict=True))
                scale = max(1, max(map(abs, residual)) / _LEVELS)
                counts["unscaled_out_of_grid"] += int(any(abs(round(x)) > _LEVELS for x in residual))
                grid = tuple(round(x / scale) for x in residual)
                ranked = sorted((_dot(grid, key), j) for j, key in enumerate(keys))
                counts["deflated_unique_other"] += int(
                    ranked[-1][0] > max(0, ranked[-2][0]) and ranked[-1][1] != seed)
        results[pin["pin"]] = counts
    return results


if __name__ == "__main__":
    from json import dumps

    print(dumps(archived_self_route_probe(), indent=2))
