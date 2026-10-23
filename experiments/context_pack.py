"""One bounded source-unit packer shared by all context routing SLICES."""
from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import tempfile
import time
from typing import Annotated
import uuid
from typing_extensions import TypedDict

from pydantic import Field

from experiments.latent_mcp import serialized
from experiments.context_spans import leaf_span, reopen_unit, origin_id, omission
from experiments.dogfood import ram_guard
from experiments.context_candidates import prepare_units, proposal_id
from experiments.context_costs import CURRENT, stage_seconds


MAX_PACK_BYTES = 48000
MAX_PACK_TOKENS = 6000


class ContextBudget(TypedDict, total=False):
    tokens: Annotated[int, Field(strict=True, ge=512, le=MAX_PACK_TOKENS)]
    bytes: Annotated[int, Field(strict=True, ge=2048, le=MAX_PACK_BYTES)]
    max_spans: Annotated[int, Field(strict=True, ge=1, le=16)]


def budget_for(budget):
    budget = {} if budget is None else budget
    if not isinstance(budget, dict) or budget.keys() - {"tokens", "bytes", "max_spans"}:
        raise ValueError("budget accepts tokens, bytes and max_spans only")
    result = dict(tokens=MAX_PACK_TOKENS, bytes=MAX_PACK_BYTES, max_spans=8)
    result.update(budget)
    for key, low, high in (("tokens", 512, MAX_PACK_TOKENS),
                           ("bytes", 2048, MAX_PACK_BYTES), ("max_spans", 1, 16)):
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f"budget.{key} must be an integer in {low}..{high}")
    return result


class OutputMeter:
    """Count local o200k tokens only when already cached. Never download a tokenizer."""

    def __init__(self):
        self.encoding = None
        url = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
        cache = Path(os.environ.get("TIKTOKEN_CACHE_DIR", os.environ.get(
            "DATA_GYM_CACHE_DIR", str(Path(tempfile.gettempdir()) / "data-gym-cache"))))
        cached = cache / hashlib.sha1(url.encode()).hexdigest()
        expected = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
        if cached.is_file() and hashlib.sha256(cached.read_bytes()).hexdigest() == expected:
            try:
                import tiktoken
                self.encoding = tiktoken.get_encoding("o200k_base")
            except (ImportError, OSError, ValueError):
                pass
        self.name = "o200k_base" if self.encoding else "utf8_bytes_upper_bound"

    def tokens(self, text):
        if self.encoding:
            return len(self.encoding.encode(text, disallowed_special=()))
        return len(text.encode("utf-8"))

    def fits(self, pack, budget):
        from experiments.latent_context import render
        raw = serialized(pack)
        # Human CLI output is bounded too, including room for its latency footer.
        human = render(pack) + "\nTotal including connection/startup: 999999.999s"
        return (len(raw.encode("utf-8")) <= budget["bytes"]
                and self.tokens(raw) <= budget["tokens"]
                and len(human.encode("utf-8")) <= budget["bytes"]
                and self.tokens(human) <= budget["tokens"])

    def blocked(self, pack, budget):
        from experiments.latent_context import render
        texts = [serialized(pack), render(pack) + "\nTotal including connection/startup: 999999.999s"]
        return ([name for name in ("bytes", "tokens") if any(
            (len(text.encode("utf-8")) if name == "bytes" else self.tokens(text)) > budget[name]
            for text in texts)])


class Continuations:
    """Bounded, expiring, process-local capabilities. Never accept a caller's row queue."""

    def __init__(self):
        self.entries = OrderedDict()
        # Retain chains separately from page cursors: draining >32 pages must not
        # evict the stable omission-recovery capability of that same chain.
        self.ledgers = OrderedDict()

    def start_ledger(self, binding):
        now = time.monotonic()
        self.ledgers = OrderedDict((k, v) for k, v in self.ledgers.items() if v[0] > now)
        while len(self.ledgers) >= 32:
            self.ledgers.popitem(last=False)
        token = "ctxl_" + uuid.uuid4().hex
        self.ledgers[token] = (now + 3600, binding, dict(records=[], keys=set()))
        return token

    def ledger(self, token, binding):
        value = self.ledgers.get(token)
        self._validate(value, binding)
        self.ledgers[token] = (time.monotonic() + 3600, binding, value[2])
        self.ledgers.move_to_end(token)
        return value[2]

    @staticmethod
    def _validate(value, binding):
        if value is None or value[0] <= time.monotonic():
            raise ValueError("unknown or expired continuation; restart with continuation=null")
        if value[1] != binding:
            raise ValueError("continuation question/corpus/snapshot/intent/readings mismatch")

    def issue(self, binding, pending):
        now = time.monotonic()
        self.entries = OrderedDict((k, v) for k, v in self.entries.items() if v[0] > now)
        while len(self.entries) >= 32:
            self.entries.popitem(last=False)
        token = "ctx1_" + uuid.uuid4().hex
        self.entries[token] = (now + 3600, binding, pending)
        return token

    def resolve(self, token, binding):
        if isinstance(token, str) and token.startswith("ctxl_"):
            ledger = self.ledger(token, binding)
            return dict(kind="omissions", ledger=token, offset=0, end=len(ledger["records"]))
        value = self.entries.get(token) if isinstance(token, str) else None
        self._validate(value, binding)
        self.ledger(value[2]["ledger"], binding)
        return value[2]


def build_context(memory, question, corpus, snapshot, intent, budget, readings,
                  continuation, strategy, options, progress, format="json"):
    """Pack atomic proposals; continuations retain groups, never reroute."""
    from experiments.latent_context import facets_for, words

    if (not isinstance(question, str) or not question.strip() or len(question) > 4096
            or intent not in ("answer", "explore", "find_all") or strategy not in ("one_shot", "ppr", "expand")):
        raise ValueError("question <=4096 chars, intent=answer|explore|find_all, strategy=one_shot|ppr|expand")
    tool = memory.tool
    corpus = tool.corpus_id if corpus is None else corpus
    if not isinstance(corpus, str) or not corpus or len(corpus) > 1024:
        raise ValueError("bounded nonempty corpus name required")
    if snapshot is not None and snapshot != tool.snapshot:
        raise ValueError("snapshot mismatch; use the snapshot returned by warm/context")
    snapshot = tool.snapshot
    readings = [] if readings is None else readings
    if (not isinstance(readings, list) or len(readings) > 4 or
            any(not isinstance(r, str) or not r.strip() or len(r) > 512 for r in readings)):
        raise ValueError("readings must contain at most four nonempty strings <=512 chars")
    budget = budget_for(budget)
    options = {} if options is None else options
    binding = serialized([question, corpus, snapshot, intent, readings, strategy])
    ram_guard()
    started = time.perf_counter()
    before = dict(tool.memory.reference_costs)
    state = memory.continuations.resolve(continuation, binding) if continuation is not None else None
    ledger_token = state["ledger"] if state else memory.continuations.start_ledger(binding)
    ledger = memory.continuations.ledger(ledger_token, binding)
    recovering = state is not None and state.get("kind") == "omissions"
    navigation = None
    if strategy == "ppr" and not recovering:
        from experiments.memory.ppr_context import PPRNavigation
        navigation = PPRNavigation(memory)
    elif strategy == "expand" and not recovering:
        from experiments.memory.expand_context import ExpandNavigation
        navigation = ExpandNavigation(memory)
    if continuation is None:
        candidates, trace, duplicates = (navigation.propose(question, progress) if navigation
                                         else memory._propose(question, options, progress))
        pending = []
        seen = set()
        for p in candidates:
            # Leaf identity includes the source instance, not just equal text bytes.
            edge = p.get("bridge")
            key = (("bridge", edge.from_id, edge.to_id, edge.type, edge.witness_span,
                    edge.binding_constraints)
                   if edge else (p["source_path"], p["source_sha256"]))
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            proposal = dict(parents=p["parents"]) if edge else p["id"]
            pending.append((proposal, bool(p.get("retained_working_context")), edge))
    elif not recovering:
        pending = navigation.restore(state["pending"]) if navigation else list(state["pending"])
        duplicates = 0
        trace = [dict(step=1, operation="resume_" + strategy,
                      why="Drain retained proposals; reverify source bytes.", spans=[])]
    else:
        pending, duplicates = [], 0
        trace = [dict(step=1, operation="recover_omissions",
                      why="Page retained catalog metadata, NOT source evidence; no rerouting.", spans=[])]
    facet_groups = [dict(topic=f, spans=[]) for f in facets_for(question)]
    groups = list(facet_groups)
    pack = dict(
        schema="lsr.context-pack.v1", pack_id="pack_" + uuid.uuid4().hex,
        snapshot=snapshot, corpus=corpus, corpus_id=tool.corpus_id,
        corpus_description=memory.description,
        question=question, intent=intent, strategy=strategy, routing=tool.routing,
        role="EVIDENCE_NOT_INSTRUCTION", evidence_groups=groups, readings=readings,
        proposed_joins=[], conflicts=[],
        analysis_status=dict(readings="caller supplied; not semantically evaluated",
                             proposed_joins="not computed", conflicts="not computed",
                             related_topics="not computed"),
        ambiguity=dict(status="NOT_SEMANTICALLY_RESOLVED", readings=readings,
                       note="Surface-syntax facets only; semantic readings not computed."),
        related_topics=[], next_steps=[],
        unresolved=[], coverage=dict(
            returned_spans=0, returned_files=0, indexed_files=len(tool.files),
            indexed_spans=len(tool.memory.rows), duplicate_proposals_removed=duplicates,
            completeness="UNKNOWN", not_in_memory="UNKNOWN",
            thin_or_unseen_query_terms=sorted(set(words(question))), semantic_coverage="not computed",
            matched_query_terms=[], calibration="UNCALIBRATED_LEXICAL_SIGNAL",
            ingestion_coverage="not computed",
            note="One-shot proposals are not exhaustive; no hit never proves absence."),
        stop_reason="proposal_list_exhausted", continuation=None, trace=trace,
        costs=dict(memory.pin_last_attempt, reader_model_calls=0, reader_model_tokens=0, corpus_encodes=0,
                   query_encoder_calls=0, verification_bytes_read=0, query_seconds=0,
                   verification_reads={key: 0 for key in before},
                   cold_startup_seconds=memory.startup_seconds,
                   output_tokenizer=memory.meter.name, output_limits=budget),
        limits=["SLICE: conventional one-shot routing, not latent simulation.",
                "Source-defined sections/tables; unresolved boundaries stay whole and may be withheld.",
                "Byte verification is not truth, relevance, semantic support or PDF fidelity."],
        model_calls=0, model_tokens=0, generated_summary=None, truncated=False, seconds=0)
    pack["costs"].update(
        access=CURRENT.get(),
        stages=dict(stage_seconds(), scope="Exclusive wall stages through format/cap validation; "
                    "verification includes structural opening and hashes. Excludes final receipt/cache/transport."),
        client=dict(wall_seconds=None, connection_seconds=None, transport_seconds=None,
                    waiting_seconds=memory.context_last_attempt["waiting_seconds"],
                    scope="Client-local through response decode; excludes receipt self-metering and host wrapping. "
                          "Transport is wall minus measured server time, not socket-only. Null means unmeasured."),
        accounting_scope="Logical payloads and actual retained-row visits; not physical I/O or reader usage.",
        setup_accounting="Backend startup attributed once to the first context attempt; may precede that call.",
        context_calls=1, continuation_calls=int(continuation is not None),
        omission_recovery_calls=int(recovering),
        setup_this_call=memory.context_last_attempt["setup_this_call"],
        retained_setup=dict(startup=memory.startup_access),
        startup_this_call_seconds=memory.startup_seconds
            if memory.context_last_attempt["startup_this_call"] else 0)
    pack["strategy_selection"] = dict(default_strategy=memory.default_strategy,
                                     effective_strategy=strategy)
    pack["format_selection"] = dict(default_format=memory.default_format, effective_format=format)
    pack["costs"]["reader_visible"] = dict(
        serialized_bytes=0, local_tokens_estimate=0, metadata_bytes=0,
        metadata_tokens_estimate=0, quotation_bytes=0, quotation_tokens_estimate=0)
    if strategy == "expand":
        pack["unresolved_obligations"] = [
            dict(reason="required_premises_and_semantic_support_not_established",
                 status="READER_MUST_ASSESS",
                 note="Navigation and byte checks do not determine whether all required premises are present.")]
    if strategy == "one_shot":
        pack["costs"]["routing_access"] = dict(memory.routing_access,
            scope="Legacy seed scans only; costs.access includes all routing, opens, hashes and row visits.")
    if navigation:
        pack["coverage"]["ingestion_coverage"] = navigation.map.coverage()
        pack["coverage"]["note"] = "Bounded graph proposals are not exhaustive; exhaustion never proves absence."
        pack["limits"][0] = ("SLICE: INTERNAL PPR ablation, not latent reasoning." if strategy == "ppr"
                               else "SLICE: conventional source-bound ONE-hop expansion, not latent reasoning.")
        pack["costs"][strategy] = navigation.metadata()
    arguments = dict(question=question, corpus=corpus, snapshot=snapshot, intent=intent,
                     budget=budget, readings=readings, strategy=strategy, format=format,
                     continuation="ctx1_" + "0" * 32)
    pack["omission_ledger"] = dict(
        total=len(ledger["records"]), returned=0,
        scope="CATALOG_METADATA_NOT_EVIDENCE; omissions recorded so far, not exhaustive",
        recovery=dict(tool="context", arguments=dict(arguments, continuation=ledger_token)))
    # Reserve the actual cursor envelope even on the last page. This keeps the
    # fit decision independent of whether another atomic leaf fits.
    pack["continuation"] = dict(tool="context", arguments=arguments)
    reserve = dict(budget, bytes=budget["bytes"] - 1024, tokens=budget["tokens"] - 384)
    if not memory.meter.fits(pack, reserve):
        raise ValueError("budget too small for context metadata; increase tokens/bytes or shorten question")
    delivered, remaining, cached = [], [], {}
    position = 0
    if recovering:
        position = state["offset"]
        while position < state["end"]:
            pack["unresolved"].append(ledger["records"][position])
            if not memory.meter.fits(pack, reserve):
                pack["unresolved"].pop()
                if not pack["unresolved"]:
                    raise ValueError("budget too small to report an omission; increase tokens/bytes")
                break
            position += 1
        if position < state["end"]:
            arguments["continuation"] = memory.continuations.issue(
                binding, dict(kind="omissions", ledger=ledger_token, offset=position, end=state["end"]))
            pack["stop_reason"] = "omission_ledger_page"
        else:
            pack["continuation"] = None
            pack["stop_reason"] = "omission_ledger_exhausted"
        pack["omission_ledger"].update(offset=state["offset"], end=state["end"])
        # The proposal loop below is empty on ledger-only recovery pages.
        position = 0

    deferred_omissions = []
    deferred_pending = []

    def cached_seed(item):
        proposal, _, candidate_edge = item
        if candidate_edge or not isinstance(proposal, str):
            return False
        return navigation.has_verified_source(proposal)

    def add_omission(item):
        if strategy == "expand" and edge:
            from dataclasses import asdict
            item["bridge_identity"] = asdict(edge)
        # Rejected bridges must not crowd verified standalone evidence off the
        # page. Keep their full recovery records, then spend only spare wire
        # budget on inline metadata after trying the seeds.
        if strategy == "expand":
            deferred_omissions.append(item)
            # Try ranked seeds before spending more access budget on bridges.
            # A standalone oversized seed must NOT reorder an affordable bridge
            # behind enough unrelated seeds to fill the page.
            if edge and not delivered:
                tail = pending[position + 1:]
                pending[position + 1:] = sorted(tail, key=lambda p:
                    (bool(p[2]), not cached_seed(p)))
            return True
        pack["unresolved"].append(item)
        if memory.meter.fits(pack, reserve):
            return True
        pack["unresolved"].pop()
        if not pack["unresolved"] and not delivered:
            raise ValueError("budget too small to report an omission; increase tokens/bytes")
        return False

    while position < len(pending):
        proposal, retained, edge = pending[position]
        span_id = proposal_id(proposal)
        if len(delivered) >= budget["max_spans"]:
            remaining = pending[position:]
            break
        if isinstance(proposal, (str, dict)):
            try:
                units = prepare_units(memory, navigation, proposal, question, edge)
            except ValueError as error:
                if not edge or "missing bridge parent" not in str(error):
                    raise
                if not add_omission(omission(memory, span_id, str(error), [],
                                             parents=proposal["parents"], semantic_support="NOT_ESTABLISHED")):
                    remaining = pending[position:]
                    break
                position += 1
                continue
            if units is None:
                if not add_omission(omission(memory, span_id, navigation.access.stop_reason,
                    ["verification_bytes"], **(dict(parents=proposal["parents"]) if edge else {}))):
                    remaining = pending[position:]
                    break
                if navigation.access.stop_reason == "atomic_source_exceeds_verification_budget":
                    position += 1
                    continue
                if (strategy == "expand" and edge and not delivered
                        and any(cached_seed(p) for p in pending[position + 1:])):
                    # The first bridge parent may already be verified. Drain
                    # cached seeds without new reads, retaining the whole bridge
                    # for a fresh access budget on a later page.
                    deferred_pending.append(pending[position])
                    position += 1
                    continue
                remaining = pending[position:]
                break
            replacements = []
            for unit in units:
                ids = [s["id"] for s in unit]
                cached.update((s["id"], s) for s in unit)
                replacements.append((ids, retained, edge))
            pending[position:position + 1] = replacements
            proposal, retained, edge = pending[position]
            span_id = proposal_id(proposal)
        leaf_ids = proposal
        # If raw leaf bytes alone exceed the wire cap, this source cannot fit.
        # Stat is a rejection shortcut, never verification or returned evidence.
        leaf_bytes = sum(len(cached[i]["text"].encode()) if i in cached else
                             int(i.rsplit("_", 1)[1]) - int(i.rsplit("_", 2)[1]) if ":U" in i else
                             (tool.memory.directory / "leaves" / (
                                 tool.memory.rows[tool._index(origin_id(tool, i))]["leaf_sha256"] + ".txt")).stat().st_size
                             for i in leaf_ids)
        source_spans = []
        if leaf_bytes <= budget["bytes"] and len(leaf_ids) <= budget["max_spans"]:
            source_spans = ([cached[i] for i in leaf_ids] if all(i in cached for i in leaf_ids)
                            else navigation.reopen_units(leaf_ids) if navigation
                            else [cached[i] if i in cached else reopen_unit(tool, i) if ":U" in i
                                  else leaf_span(tool, origin_id(tool, i)) for i in leaf_ids])
        if source_spans is None:
            if navigation.access.stop_reason == "atomic_source_exceeds_verification_budget":
                if not add_omission(omission(memory, span_id, navigation.access.stop_reason,
                                             ["verification_bytes"])):
                    remaining = pending[position:]
                    break
                position += 1
                continue
            remaining = pending[position:]
            add_omission(omission(memory, span_id, navigation.access.stop_reason,
                                  [navigation.access.stop_reason]))
            break
        if retained:
            for span in source_spans:
                span["retained_working_context"] = True
        existing_ids = {s["id"] for s in delivered}
        verified_group = bool(source_spans)
        bridge_added = False
        if edge and source_spans:
            from experiments.memory.bridge_groups import bridge_group
            try:
                group = bridge_group(navigation, edge, delivered + source_spans)
            except ValueError as error:
                if not add_omission(omission(memory, span_id, str(error), [],
                    parents=[edge.from_id, edge.to_id], semantic_support="NOT_ESTABLISHED")):
                    remaining = pending[position:]
                    break
                position += 1
                continue
            # Adjacency inside an already complete unit is not a two-parent
            # bridge. Repeated chunk edges must not fill the pack with self-pairs.
            if len(set(group["parents"])) > 1:
                groups.append(group)
                bridge_added = True
        if not bridge_added:
            text = " ".join(p["text"] for p in source_spans)
            matches = [len(set(words(g["topic"])) & set(words(text))) for g in facet_groups]
            group = facet_groups[matches.index(max(matches))]
        source_spans = [s for s in source_spans if s["id"] not in existing_ids]
        group["spans"].extend(source_spans)
        if ((source_spans or (verified_group and edge)
             or (not edge and all(i in existing_ids for i in leaf_ids)))
                and len(delivered) + len(source_spans) <= budget["max_spans"]
                and memory.meter.fits(pack, reserve)):
            delivered.extend(source_spans)
            position += 1
            continue
        blocked = memory.meter.blocked(pack, reserve)
        if len(delivered) + len(source_spans) > budget["max_spans"] or len(leaf_ids) > budget["max_spans"]:
            blocked.append("max_spans")
        if leaf_bytes > budget["bytes"]:
            blocked.append("bytes")
        if source_spans:
            del group["spans"][-len(source_spans):]
        if bridge_added:
            groups.remove(group)
        if delivered or (source_spans and pack["unresolved"]):
            remaining = pending[position:]
            break
        # A leaf cannot fit on an otherwise empty page. Never split a table to
        # manufacture progress. Skip it explicitly so continuation cannot loop.
        if not add_omission(omission(memory, span_id,
            "atomic_leaf_exceeds_output_budget", sorted(set(blocked)),
            leaf_bytes=leaf_bytes, source_leaves=len(leaf_ids),
            **(dict(parents=[edge.from_id, edge.to_id], semantic_support="NOT_ESTABLISHED") if edge else {}))):
            remaining = pending[position:]
            break
        position += 1
    # Full rejected-bridge records remain recoverable even when the evidence
    # leaves no room for them inline. Do not drop or truncate their obligations.
    for item in deferred_omissions:
        key = serialized(item)
        if key not in ledger["keys"]:
            ledger["keys"].add(key)
            ledger["records"].append(item)
        pack["unresolved"].append(item)
        if not memory.meter.fits(pack, reserve):
            pack["unresolved"].pop()
    remaining = deferred_pending + remaining
    if remaining:
        state = dict(pending=navigation.cursor_state(remaining) if navigation else remaining,
                     ledger=ledger_token)
        arguments["continuation"] = memory.continuations.issue(binding, state)
        pack["stop_reason"] = (navigation.access.stop_reason if navigation and
                               navigation.access.stop_reason in ("navigation_time_cap", "verification_byte_cap")
                               else "output_budget")
    elif not recovering:
        pack["continuation"] = None
        if pack["unresolved"] or ledger["records"]:
            pack["stop_reason"] = "atomic_leaf_exceeds_output_budget"
    present = set(words(" ".join(p["text"] for p in delivered)))
    pack["coverage"].update(
        returned_spans=len(delivered), returned_files=len({p["file"] for p in delivered}),
        remaining_proposals=len(remaining),
        matched_query_terms=sorted(set(words(question)) & present),
        thin_or_unseen_query_terms=sorted(set(words(question)) - present))
    delta = {k: v - before[k] for k, v in tool.memory.reference_costs.items()}
    seconds = time.perf_counter() - started
    pack["costs"].update(query_encoder_calls=CURRENT.get()["query_encoder_calls"],
                         verification_bytes_read=delta["source_version_bytes_read"] + delta["leaf_bytes_read"],
                         verification_reads=delta, query_seconds=seconds)
    if navigation:
        pack["costs"][strategy] = navigation.metadata()
        if not remaining and navigation.access.stop_reason:
            pack["stop_reason"] = navigation.access.stop_reason
        elif not remaining and pack["stop_reason"] == "proposal_list_exhausted":
            pack["stop_reason"] = strategy + "_" + navigation.metadata()["stop_reason"]
    if not recovering:
        for item in pack["unresolved"]:
            key = serialized(item)
            if key not in ledger["keys"]:
                ledger["keys"].add(key)
                ledger["records"].append(item)
    pack["omission_ledger"].update(total=len(ledger["records"]), returned=len(pack["unresolved"]))
    pack.update(seconds=seconds, truncated=bool(remaining or ledger["records"]))
    measure_output(pack, memory.meter)
    if not memory.meter.fits(pack, budget):
        raise ValueError("context metadata exceeds output budget; increase tokens/bytes")
    ram_guard()
    return pack


def measure_output(pack, meter):
    """Include the receipt itself. Token estimates separate metadata from quotes."""
    import copy
    receipt = pack["costs"]["reader_visible"]
    group_key = "evidence_groups" if "evidence_groups" in pack else "facets"
    quotes = [s["text"] for g in pack[group_key] for s in g["spans"]]
    receipt.update(quotation_bytes=sum(len(q.encode("utf-8")) for q in quotes),
                   quotation_tokens_estimate=sum(meter.tokens(q) for q in quotes))
    # Integer widths and tokenizer splits can change when their own counts change.
    # Keep token estimates conservatively above any self-referential oscillation.
    for _ in range(16):
        metadata = copy.deepcopy(pack)
        for group in metadata[group_key]:
            for span in group["spans"]:
                span["text"] = ""
        raw, meta = serialized(pack), serialized(metadata)
        values = dict(serialized_bytes=len(raw.encode("utf-8")),
                      local_tokens_estimate=max(receipt["local_tokens_estimate"], meter.tokens(raw) + 8),
                      metadata_bytes=len(meta.encode("utf-8")),
                      metadata_tokens_estimate=max(receipt["metadata_tokens_estimate"], meter.tokens(meta) + 8))
        if all(receipt[k] == v for k, v in values.items()):
            return
        receipt.update(values)
    raise ValueError("reader-visible cost receipt did not stabilize")
