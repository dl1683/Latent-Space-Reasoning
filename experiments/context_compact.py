"""Reversible evidence view. Diagnostics stay in the bounded same-pack cache."""
from collections import OrderedDict
from copy import deepcopy
import time

from experiments.latent_mcp import serialized


SCHEMA = "lsr.context-compact.v1"
DIAGNOSTICS = frozenset(("costs", "trace", "seconds"))
SUMMARY_COSTS = (
    "reader_model_calls", "reader_model_tokens", "corpus_encodes",
    "query_encoder_calls", "context_calls", "continuation_calls",
    "omission_recovery_calls", "verification_bytes_read", "query_seconds",
    "stages", "client", "output_limits",
)


def compact(pack, meter):
    """Keep every nondiagnostic field; factor identical span fields, not facts."""
    view = {k: deepcopy(v) for k, v in pack.items()
            if k not in DIAGNOSTICS | {"schema", "evidence_groups"}}
    spans, by_id = [], {}
    for group in pack["evidence_groups"]:
        for span in group["spans"]:
            if span["id"] in by_id:
                if serialized(by_id[span["id"]]) != serialized(span):
                    raise ValueError("different evidence occurrences share a span id")
            else:
                by_id[span["id"]] = span
                spans.append(span)
    keys = list(dict.fromkeys(k for span in spans for k in span))
    defaults, aliases, columns = {}, {}, []
    for key in keys:
        if key not in ("id", "text") and all(
                key in s and serialized(s[key]) == serialized(spans[0][key]) for s in spans):
            defaults[key] = spans[0][key]
        else:
            previous = next((other for other in columns if all(
                (key in s) == (other in s) and
                serialized(s.get(key)) == serialized(s.get(other)) for s in spans)), None)
            if previous is not None and key not in ("id", "text"):
                aliases[key] = previous
            else:
                columns.append(key)
    view.update(
        schema=SCHEMA, pack_schema=pack["schema"],
        span_defaults=deepcopy(defaults), span_aliases=aliases,
        spans=[{k: deepcopy(s[k]) for k in columns if k in s} for s in spans],
        evidence_groups=[{**{k: deepcopy(v) for k, v in g.items() if k != "spans"},
                          "span_ids": [s["id"] for s in g["spans"]]}
                         for g in pack["evidence_groups"]],
        diagnostics=dict(tool="context_details",
                         arguments=dict(pack_id=pack["pack_id"], corpus=pack["corpus"]),
                         scope="Same full JSON pack; 32 packs / one idle hour / this process. "
                               "Saved verification receipt, not fresh verification; use open_span."),
        costs={k: deepcopy(pack["costs"][k]) for k in SUMMARY_COSTS if k in pack["costs"]})
    view["costs"]["reader_visible"] = dict(
        serialized_bytes=0, local_tokens_estimate=0, metadata_bytes=0,
        metadata_tokens_estimate=0, quotation_bytes=0, quotation_tokens_estimate=0)
    view["costs"]["timing_scope"] = (
        "query_seconds is navigation/packing only; context_details.costs.serving includes "
        "formatting/validation. Client wall time includes receipt/cache/transport.")
    measure_compact(view, meter)
    return view


def expand_compact(view):
    """Reconstruct ALL reader fields exactly, excluding costs/trace/seconds."""
    if view.get("schema") != SCHEMA:
        raise ValueError("unsupported compact schema")
    by_id = {}
    for row in view["spans"]:
        span = dict(deepcopy(view["span_defaults"]), **deepcopy(row))
        for alias, original in view["span_aliases"].items():
            if original in span:
                span[alias] = deepcopy(span[original])
        if span["id"] in by_id:
            raise ValueError("duplicate compact span id")
        by_id[span["id"]] = span
    omitted = DIAGNOSTICS | {
        "schema", "pack_schema", "span_defaults", "span_aliases", "spans",
        "evidence_groups", "diagnostics",
    }
    result = {k: deepcopy(v) for k, v in view.items() if k not in omitted}
    result.update(schema=view["pack_schema"], evidence_groups=[
        {**{k: deepcopy(v) for k, v in g.items() if k != "span_ids"},
         "spans": [deepcopy(by_id[i]) for i in g["span_ids"]]}
        for g in view["evidence_groups"]])
    return result


def restore_compact(view, details):
    """Full JSON round trip with the same-pack context_details response."""
    reader = expand_compact(view)
    expected = {k: v for k, v in details.items() if k not in DIAGNOSTICS}
    if reader != expected:
        raise ValueError("compact evidence and diagnostic pack mismatch")
    return deepcopy(details)


def measure_compact(view, meter):
    receipt = view["costs"]["reader_visible"]
    quotes = [s["text"] for s in view["spans"]]
    receipt.update(quotation_bytes=sum(len(q.encode("utf8")) for q in quotes),
                   quotation_tokens_estimate=sum(meter.tokens(q) for q in quotes))
    for _ in range(16):
        metadata = deepcopy(view)
        for span in metadata["spans"]:
            span["text"] = ""
        raw, meta = serialized(view), serialized(metadata)
        values = dict(serialized_bytes=len(raw.encode("utf8")),
                      local_tokens_estimate=max(receipt["local_tokens_estimate"], meter.tokens(raw) + 8),
                      metadata_bytes=len(meta.encode("utf8")),
                      metadata_tokens_estimate=max(receipt["metadata_tokens_estimate"], meter.tokens(meta) + 8))
        if all(receipt[k] == v for k, v in values.items()):
            return
        receipt.update(values)
    raise ValueError("compact cost receipt did not stabilize")


class PackDetails:
    """Bounded diagnostics, never an unbounded corpus or durable verification store."""

    def __init__(self):
        self.entries = OrderedDict()

    def retain(self, pack):
        now = time.monotonic()
        self.entries = OrderedDict((k, v) for k, v in self.entries.items() if v[0] > now)
        while len(self.entries) >= 32:
            self.entries.popitem(last=False)
        self.entries[pack["pack_id"]] = (now + 3600, deepcopy(pack))

    def fetch(self, pack_id, corpus):
        value = self.entries.get(pack_id) if isinstance(pack_id, str) else None
        if value is None or value[0] <= time.monotonic() or value[1]["corpus"] != corpus:
            raise ValueError("unknown, expired or corpus-mismatched pack_id; request new context")
        self.entries[pack_id] = (time.monotonic() + 3600, value[1])
        self.entries.move_to_end(pack_id)
        return deepcopy(value[1])
