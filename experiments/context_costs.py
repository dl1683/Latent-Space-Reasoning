"""Request-local logical payload accounting, shared by every context strategy.

Not physical disk I/O, FLOPs, allocator traffic, or a reader's private tokenizer.
Nested setup scopes keep construction separate from warm query work.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import time


CURRENT = ContextVar("context_access", default=None)
TIMING = ContextVar("context_timing", default=None)


def stage_seconds():
    return dict.fromkeys(("navigation_seconds", "verification_reads_seconds",
                          "packing_seconds", "serialization_seconds"), 0.0)


def _flush_stage(timer):
    now = time.perf_counter()
    timer["seconds"][timer["stage"] + "_seconds"] += now - timer["since"]
    timer["since"] = now


@contextmanager
def timing():
    timer = dict(seconds=stage_seconds(), stage="packing", since=time.perf_counter())
    token = TIMING.set(timer)
    try:
        yield timer["seconds"]
    finally:
        _flush_stage(timer)
        TIMING.reset(token)


@contextmanager
def stage(name):
    """Exclusive request-local wall time; nested reads/serialization leave navigation."""
    timer = TIMING.get()
    if timer is None:
        yield
        return
    previous = timer["stage"]
    _flush_stage(timer)
    timer["stage"] = name
    try:
        yield
    finally:
        _flush_stage(timer)
        timer["stage"] = previous


def timing_snapshot():
    timer = TIMING.get()
    if timer is None:
        return stage_seconds()
    _flush_stage(timer)
    return dict(timer["seconds"])


def counters():
    return dict.fromkeys((
        "map_scans", "map_vector_rows_read", "map_bytes_read",
        "exact_vector_rows_read", "exact_vector_bytes_read",
        "lexical_row_scans", "lexical_text_bytes_read",
        "source_row_records_read", "row_iteration_visits",
        "edge_reads", "edge_record_bytes_read", "reference_path_metadata_checks",
        "source_open_attempts", "source_opens", "source_bytes_read",
        "leaf_open_attempts", "leaf_opens", "leaf_bytes_read",
        "metadata_open_attempts", "metadata_opens", "metadata_bytes_read",
        "verification_bytes_read", "verification_hash_checks", "verification_hash_bytes",
        "query_encoder_calls", "query_encoder_seconds",
    ), 0)


@contextmanager
def accounting(values=None):
    values = counters() if values is None else values
    token = CURRENT.set(values)
    try:
        yield values
    finally:
        CURRENT.reset(token)


def charge(**amounts):
    values = CURRENT.get()
    if values is not None:
        for key, amount in amounts.items():
            values[key] += amount


def read_payload(path, kind):
    # ExplicitReferenceMemory's existing leaf-accounting adapter delegates here.
    if not isinstance(path, (str, Path)) and hasattr(path, "read_bytes"):
        return path.read_bytes()
    charge(**{kind + "_open_attempts": 1})
    raw = Path(path).read_bytes()
    charge(**{kind + "_opens": 1, kind + "_bytes_read": len(raw)})
    if kind in ("source", "leaf"):
        charge(verification_bytes_read=len(raw))
    return raw


def visit_edge(edge):
    charge(edge_reads=1, edge_record_bytes_read=len(repr(edge).encode("utf-8")))
    return edge


class MeteredRows(list):
    """Count actual fetched retained records, including repeated/full-list visits."""

    def __iter__(self):
        for row in super().__iter__():
            charge(source_row_records_read=1, row_iteration_visits=1)
            yield row

    def __getitem__(self, key):
        value = super().__getitem__(key)
        charge(source_row_records_read=len(value) if isinstance(key, slice) else 1)
        return value
