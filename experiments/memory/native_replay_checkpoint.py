"""Data-only transport for an intact native reconciliation checkpoint."""

from dataclasses import dataclass
import json
from os import fsync
from pathlib import Path
from types import MappingProxyType

from experiments.memory.native_latent_recurrence import NativeLatentModel
from experiments.memory.native_publication import _between, _enforce_limit, _reader_digest, _shape
from experiments.memory.native_reconciliation import (
    _IntactReplay, _ReplayTrace, _SelectionFrontier, _implementation_sha256, _trace,
)
from experiments.memory.source_revision_feed import _encode
from experiments.memory.verified_source_tool import require_digest, sha256


_SCHEMA = "lsr-native-replay-checkpoint-v1"


@dataclass(frozen=True, slots=True)
class NativeReplayHandoff:
    replay: _IntactReplay | None
    status: str
    bytes_read: int
    backbone_calls: int = 0
    row_scores: int = 0


def _binding(reader, opening):
    if type(reader) is not NativeLatentModel:
        raise TypeError("native replay checkpoint requires a native reader")
    reader._check_memory(opening)
    if opening._selective is None or opening.omitted_sources or opening._derived_off:
        raise ValueError("native replay checkpoint requires an intact opening publication")
    digests = (opening._publication_seal, opening.seal, opening._publication_runtime_sha256,
               opening._reader_fingerprint_sha256)
    for digest in digests:
        require_digest(digest)
    if _reader_digest(reader) != opening._reader_fingerprint_sha256:
        raise ValueError("native replay checkpoint requires the unchanged opening reader")
    publication, source, runtime, fingerprint = digests
    return {"schema": _SCHEMA, "publication": publication, "source": source,
            "runtime": runtime, "reader": fingerprint, "implementation": _implementation_sha256()}


def _inventories(opening):
    sources = tuple(item.span for source in sorted(opening._sources) for item in opening._sources[source])
    derived = tuple(span for shard in opening._selective.derived for span in shard.spans)
    return sources, derived


def _frontier_record(frontier, indices):
    return [frontier.query.hex(), frontier.cutoff, [indices[span] for span in frontier.selected]]


def write_native_replay_checkpoint(path, reader, replay, *, max_bytes):
    """Export complete intact replay rows and return the digest for the controller."""
    _enforce_limit(max_bytes)
    if type(replay) is not _IntactReplay:
        raise TypeError("a completed native replay checkpoint is required")
    header = _binding(reader, replay.opening)
    if replay.implementation_sha256 != header["implementation"]:
        raise ValueError("native replay checkpoint was captured by a different implementation")
    steps = replay.confirmation_steps
    if not _between(steps, 3, reader._steps + 1):
        raise ValueError("native replay checkpoint depth is invalid")
    sources, derived = _inventories(replay.opening)
    indices = tuple({span: position for position, span in enumerate(bank)} for bank in (sources, derived))
    payload = bytearray()

    def append(record):
        line = _encode(record)
        if len(payload) + len(line) + 1 > max_bytes:
            raise ValueError("native replay checkpoint exceeds its byte allowance")
        payload.extend(line)
        payload.append(10)

    append({**header, "steps": steps})
    for position, span in enumerate(sources):
        trace = replay.traces.get(span)
        if trace is None or not trace.complete or len(trace.frontiers) != steps:
            continue
        frontiers = [[_frontier_record(frontier, bank) for frontier, bank in zip(pair, indices, strict=True)]
                     for pair in trace.frontiers]
        append([position, [state.hex() for state in trace.states], trace.clean, frontiers])
    with Path(path).open("xb") as handle:
        handle.write(payload)
        handle.flush()
        fsync(handle.fileno())
    return sha256(payload)


def _read_line(line):
    record = json.loads(line)
    if _encode(record) != line:
        raise ValueError("native replay checkpoint record is not canonical")
    return record


def _q8(value, width):
    if type(value) is not str or len(value) != width * 2:
        raise ValueError("native replay checkpoint Q8 width differs")
    raw = bytes.fromhex(value)
    if raw.hex() != value or 128 in raw:
        raise ValueError("native replay checkpoint Q8 coordinates are invalid")
    return raw


def _read_frontier(record, bank, key_width, tie_limit):
    if type(record) is not list or len(record) != 3:
        raise ValueError("native replay checkpoint frontier is malformed")
    query, cutoff, selected = record
    if (not _between(cutoff, 1, key_width * 127 ** 2) or type(selected) is not list or len(selected) > tie_limit
            or any(not _between(index, 0, len(bank) - 1) for index in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("native replay checkpoint frontier indices or cutoff are invalid")
    return _SelectionFrontier(_q8(query, key_width), cutoff, tuple(bank[index] for index in selected))


def read_native_replay_checkpoint(path, reader, opening, *, seal, max_bytes, max_backbone_calls, max_row_scores):
    """Import a controller-authenticated checkpoint and check one anchor on this worker."""
    _enforce_limit(max_bytes)
    require_digest(seal)
    if any(type(value) is not int or value < 0 for value in (max_backbone_calls, max_row_scores)):
        raise ValueError("native replay handoff requires nonnegative work bounds")
    with Path(path).open("rb") as handle:
        payload = handle.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError("native replay checkpoint exceeds its byte allowance")
    if sha256(payload) != seal:
        raise ValueError("native replay checkpoint seal differs")
    if not payload.endswith(b"\n"):
        raise ValueError("native replay checkpoint lacks a complete final record")
    binding = _binding(reader, opening)
    lines = iter(payload[:-1].split(b"\n"))
    header = _read_line(next(lines))
    steps = header.get("steps") if type(header) is dict else None
    if not _between(steps, 3, reader._steps + 1) or header != {**binding, "steps": steps}:
        raise ValueError("native replay checkpoint opening, runtime or depth differs")
    sources, derived = _inventories(opening)
    width, key_width = _shape(reader)
    tie_limit = reader._selector._tie_limit
    traces = {}
    previous = -1
    for line in lines:
        record = _read_line(line)
        if type(record) is not list or len(record) != 4:
            raise ValueError("native replay checkpoint anchor record is malformed")
        position, states, clean, frontiers = record
        if (not _between(position, previous + 1, len(sources) - 1)
                or type(states) is not list or len(states) != steps or type(clean) is not bool
                or type(frontiers) is not list or len(frontiers) != steps
                or any(type(pair) is not list or len(pair) != 2 for pair in frontiers)):
            raise ValueError("native replay checkpoint anchor index or depth differs")
        parsed = tuple((_read_frontier(left, sources, key_width, tie_limit),
                        _read_frontier(right, derived, key_width, tie_limit)) for left, right in frontiers)
        selected = frozenset(span for _, frontier in parsed for span in frontier.selected)
        traces[sources[position]] = _ReplayTrace(tuple(_q8(state, width) for state in states),
                                                clean, selected, True, parsed)
        previous = position
    if not traces:
        return NativeReplayHandoff(None, "empty", len(payload))
    calls, scores = steps + 1, steps * (len(sources) + len(derived))
    if calls > max_backbone_calls or scores > max_row_scores:
        return NativeReplayHandoff(None, "budget_exceeded", len(payload))
    # Ordered records make ties deterministic. Prefer a trace that actually
    # exercised the derived bank over one that only saw an empty selection.
    sample = max(traces, key=lambda span: len(traces[span].selected))
    item = next(item for item in opening._sources[sample.source_id] if item.span == sample)

    import torch

    with reader._forecast_lock, torch.inference_mode():
        actual = _trace(reader, item, opening, steps, capture=True)
    if actual != traces[sample]:
        return NativeReplayHandoff(None, "sample_mismatch", len(payload), calls, scores)
    replay = _IntactReplay(opening, steps, binding["implementation"], MappingProxyType(traces))
    return NativeReplayHandoff(replay, "attached", len(payload), calls, scores)
