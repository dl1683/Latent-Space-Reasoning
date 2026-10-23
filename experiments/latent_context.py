"""Memory-only context packs over the existing source-verified dogfood index.

No reader, generation, training, or corpus encoding occurs here. Dense/BM25
routing and literal links are conventional components, not latent reasoning.
"""
import json
from pathlib import Path
import re
import time

import numpy as np

from experiments.latent_mcp import LatentMemoryTool, serialized
from experiments.context_pack import MAX_PACK_BYTES, Continuations, OutputMeter, build_context, measure_output
from experiments.context_spans import leaf_span, origin_id, reopen_unit
from experiments.dogfood import digest, file_digest
from experiments.context_costs import accounting, visit_edge, timing, stage, timing_snapshot
from experiments.context_compact import PackDetails, compact, measure_compact, SCHEMA as COMPACT_SCHEMA


STOP = frozenset("""
a an and are as at be been by can compare describe did do does explain for from
give how i in is it its me of on or our that the their them these they this to
us using was were what when where which who why will with would across about
between documents document sources source evidence find show different
""".split())


def words(text):
    return [w for w in re.findall(r"[^\W_][\w-]*", text.casefold())
            if len(w) > 2 and w not in STOP and not w.isnumeric()]


def facets_for(question):
    """Surface syntax only, not an LLM decomposition or ambiguity detector."""
    parts = re.split(r"\s*;\s*|\?\s+(?=\w)|\s+(?:and also|versus|vs\.?)\s+",
                     question, flags=re.I)
    # A single 'and' can join an entity name. Keep the original query as a facet.
    if len(parts) == 1:
        parts = re.split(r"\s+and\s+", question, maxsplit=2, flags=re.I)
    facets = list(dict.fromkeys(p.strip(" ?;") for p in parts if p.strip(" ?;")))[:3]
    return facets or [question.strip()]


class ContextMemory:
    """One warm corpus backend; serialize access because navigation is mutable."""

    def __init__(self, config, *, encoder_cache=None, description="Pinned source-only corpus."):
        started = time.perf_counter()
        try:
            with accounting() as startup_access:
                self.tool = LatentMemoryTool(config, encoder_cache=encoder_cache)
                self.config_pin = file_digest(config)
        except Exception as error:
            error.attempt_costs = dict(status="REFUSED", phase="startup", access=startup_access,
                                      seconds=time.perf_counter() - started)
            raise
        self.startup_access = startup_access
        self.startup_reported = False
        self.default_strategy = self.tool.default_strategy
        self.default_format = self.tool.default_format
        self.pack_details = PackDetails()
        self.startup_seconds = time.perf_counter() - started
        self.config = str(config)
        self.continuations = Continuations()
        self.meter = OutputMeter()
        self.encoder_calls = 0
        self.routing_access = {}
        self.description = description
        self.context_last_attempt = None

    def _search(self, question, k):
        self.encoder_calls += int(self.tool.routing != "bm25")
        for key, amount in (
            ("exact_vector_rows_read", len(self.tool.memory.rows) if self.tool.routing != "bm25" else 0),
            ("exact_vector_bytes_read", self.tool.memory.vectors.nbytes if self.tool.routing != "bm25" else 0),
            ("lexical_row_scans", len(self.tool.memory.rows) if self.tool.routing != "dense" else 0),
        ):
            self.routing_access[key] = self.routing_access.get(key, 0) + int(amount)
        value = json.loads(self.tool.latent_memory(question, min(k, self.tool.max_results)))
        return value["passages"]

    def open_span(self, id):
        original = origin_id(self.tool, id)
        if original != id:
            result = dict(passages=[reopen_unit(self.tool, id) if ":U" in id else
                                    leaf_span(self.tool, original)], returned=1,
                          snapshot=self.tool.snapshot, corpus_id=self.tool.corpus_id)
            if len(serialized(result).encode("utf-8")) > MAX_PACK_BYTES:
                raise ValueError("complete leaf exceeds open_span output budget; it was not clipped")
            return result
        return json.loads(self.tool.open_span(id))

    def follow_reference(self, span_id):
        return json.loads(self.tool.follow_reference(origin_id(self.tool, span_id)))

    def neighbors(self, span_id, k=4, kind="both"):
        if type(k) is not int or not 1 <= k <= 8 or kind not in ("both", "source", "map"):
            raise ValueError("neighbors needs k=1..8 and kind=both|source|map")
        started = time.perf_counter()
        span_id = origin_id(self.tool, span_id)
        origin = self.tool._open(span_id)[0]
        memory = self.tool.memory
        index = self.tool._index(span_id)
        row = memory.rows[index]
        order = memory._source_order(row["source"], row["source_sha256"])
        pos = order.index(index)
        candidates = []
        if kind != "map":
            for target in (pos - 1, pos + 1):
                if 0 <= target < len(order):
                    visit_edge((index, order[target], "source_adjacency"))
                    candidates.append((order[target], "source_adjacency"))
        if kind != "source":
            vector = memory.vectors[index]
            scores = memory.vectors @ vector
            for target in np.argsort(-scores, kind="stable")[:k + 3]:
                if int(target) != index and int(target) not in [x[0] for x in candidates]:
                    candidates.append((int(target), "vector_similarity"))
        passages = []
        for target, relation in candidates[:k]:
            fetched = memory.search("reopen verified neighbor", 1, "bm25",
                                    allowed_rows=[target], source_navigation=True)
            for passage in fetched["passages"]:
                projected = self.tool._project(passage)
                projected["relation"] = relation
                projected["from_span"] = self.tool.corpus_id + ":" + origin["handle"]
                passages.append(projected)
        result = dict(passages=passages, returned=len(passages),
                      note="Adjacency and vector similarity propose context, not logical support.",
                      seconds=time.perf_counter() - started, truncated=False)
        while len(serialized(result).encode("utf-8")) > MAX_PACK_BYTES:
            result["passages"].pop()
            result.update(returned=len(result["passages"]), truncated=True)
        return result

    def ask(self, question, options=None, *, progress=None, corpus=None):
        options = {} if options is None else dict(options)
        allowed = {"k", "hops", "working_ids", "previous_question", "brief"}
        if options.keys() - allowed:
            raise ValueError("unknown context-pack option")
        k, hops = options.get("k", 8), options.get("hops", 2)
        working = options.get("working_ids", [])
        previous = options.get("previous_question", "")
        if (not isinstance(question, str) or not question.strip() or len(question) > 4096
                or type(k) is not int or not 1 <= k <= 16
                or type(hops) is not int or not 0 <= hops <= 2
                or not isinstance(working, list) or len(working) > 16
                or any(not isinstance(item, str) for item in working)
                or not isinstance(previous, str) or len(previous) > 4096
                or type(options.get("brief", False)) is not bool):
            raise ValueError("question <=4096 chars, k=1..16, hops=0..2, working_ids<=16 required")
        if options.get("brief"):
            raise ValueError("--brief is not installed: packs are extractive; no reader is loaded")
        pack = self.context(question, corpus=corpus, budget=dict(max_spans=k), _options=options,
                            progress=progress, strategy="one_shot", format="json")
        # Legacy shape, without duplicating every quotation in the wire pack.
        pack["facets"] = pack.pop("evidence_groups")
        measure_output(pack, self.meter)
        return pack

    def context(self, question, corpus=None, snapshot=None, intent="answer", budget=None,
                readings=None, continuation=None, *, strategy=None, _options=None,
                progress=None, format=None, _waiting_seconds=0.0):
        """One bounded context call. Source identity is verified, semantic support is not."""
        selected = self.default_strategy if strategy is None else strategy
        selected_format = self.default_format if format is None else format
        with accounting() as access, timing() as stages:
            self.context_last_attempt = dict(status="ATTEMPTED", strategy=selected, access=access,
                waiting_seconds=_waiting_seconds, stages=stages,
                continuation_calls=int(continuation is not None),
                startup_this_call=not self.startup_reported,
                setup_this_call=dict(
                    startup=self.startup_access if not self.startup_reported else None,
                    adapter=None))
            self.startup_reported = True
            started = time.perf_counter()
            try:
                self.pin_last_attempt = dict(pin_bytes_hashed=0, pin_hash_checks=0)
                if selected_format not in ("json", "compact"):
                    raise ValueError("format must be json or compact")
                result = self._context(question, corpus, snapshot, intent, budget, readings,
                    continuation, strategy=selected, _options=_options, progress=progress,
                    format=selected_format)
                built = time.perf_counter()
                delivered = compact(result, self.meter) if selected_format == "compact" else result
                formatted = time.perf_counter()
                if not self.meter.fits(delivered, result["costs"]["output_limits"]):
                    raise ValueError("context metadata exceeds output budget; increase tokens/bytes")
                validated = time.perf_counter()
                result["costs"]["serving"] = dict(
                    pack_build_seconds=built - started,
                    format_seconds=formatted - built,
                    validation_seconds=validated - formatted,
                    total_seconds=validated - started,
                    scope="Wall time through formatting/cap validation. Excludes final receipt "
                          "self-metering, cache retention and transport; use client wall time for total.")
                result["costs"]["stages"].update(timing_snapshot())
                if delivered is not result:
                    delivered["costs"]["stages"] = dict(result["costs"]["stages"])
                    measure_compact(delivered, self.meter)
                    if not self.meter.fits(delivered, result["costs"]["output_limits"]):
                        raise ValueError("context metadata exceeds output budget; increase tokens/bytes")
                measure_output(result, self.meter)
                if not self.meter.fits(result, result["costs"]["output_limits"]):
                    raise ValueError("context metadata exceeds output budget; increase tokens/bytes")
                from experiments.request_control import checkpoint
                checkpoint()
                self.pack_details.retain(result)
                self.context_last_attempt["status"] = "RETURNED"
                return delivered
            except Exception:
                self.context_last_attempt["status"] = "REFUSED"
                raise
            finally:
                self.context_last_attempt["query_seconds"] = time.perf_counter() - started
                self.context_last_attempt["pins"] = dict(self.pin_last_attempt)

    def _context(self, question, corpus, snapshot, intent, budget, readings,
                 continuation, *, strategy, _options, progress, format):
        self.pin_last_attempt = dict(pin_bytes_hashed=0, pin_hash_checks=0)
        self.routing_access = {}
        # Pin hashes are a separate cost category, never source-verification hashes.
        with accounting():
            for path, expected, label in (
                    (Path(self.config), self.config_pin, "corpus configuration"),
                    (self.tool.memory.directory / "manifest.json", self.tool.snapshot, "snapshot manifest")):
                raw = path.read_bytes()
                self.pin_last_attempt["pin_bytes_hashed"] += len(raw)
                self.pin_last_attempt["pin_hash_checks"] += 1
                if digest(raw) != expected:
                    raise ValueError(label + " changed; restart the memory server")
        return build_context(self, question, corpus, snapshot, intent, budget, readings,
                             continuation, strategy, _options, progress, format)

    def context_details(self, pack_id, corpus=None):
        return self.pack_details.fetch(pack_id, self.tool.corpus_id if corpus is None else corpus)

    @stage("navigation")
    def _propose(self, question, options, progress=None):
        k, hops = options.get("k", 8), options.get("hops", 2)
        working = options.get("working_ids", [])
        previous = options.get("previous_question", "")
        trace, candidates = [], []
        facets = facets_for(question)

        def record(operation, why, spans, **extra):
            trace.append(dict(step=len(trace) + 1, operation=operation, why=why,
                              spans=[p["id"] for p in spans],
                              regions=list(dict.fromkeys(p["file"] for p in spans)), **extra))
            if progress:
                progress(trace[-1])

        for span_id in dict.fromkeys(working):
            spans = self.open_span(origin_id(self.tool, span_id))["passages"]
            for span in spans:
                span["retained_working_context"] = True
            candidates.extend(spans)
        if working:
            record("reverify_working_context", "Reopen retained pointers; never trust old quotes.",
                   candidates)
        spans = self._search(question, min(k, 8))
        candidates.extend(spans)
        record("map_route", "Existing dense/BM25 route over the encoded-once corpus.",
               spans, query=question, routing=self.tool.routing)
        if len(facets) > 1:
            for facet in facets:
                spans = self._search(facet, min(3, k))
                candidates.extend(spans)
                record("facet_route", "Give a separately worded facet its own evidence proposal.",
                       spans, query=facet)
        elif previous:
            query = (previous[:1800] + " " + question)[:4096]
            spans = self._search(query, min(3, k))
            candidates.extend(spans)
            record("followup_route", "Retain the previous question's scope for this refinement.",
                   spans, query=query)

        # Walk actual source/reference edges, bounded independently of result k.
        # No generated entities, query rewriting or frontier model is involved.
        seeds = list(dict((p["id"], p) for p in candidates).values())
        frontier = seeds[:1]
        visited = {p["id"] for p in seeds}
        for hop in range(hops):
            if not frontier:
                break
            origin = frontier[0]
            if any(r["resolution"] == "MANIFEST_RESOLVED_UNFETCHED"
                   for r in map(visit_edge, origin.get("references", []))):
                found = self.follow_reference(origin["id"])["passages"]
                operation, reason = "reference_hop", "Follow a literal, allow-listed source link."
            else:
                found = self.neighbors(origin["id"], 2, "source")["passages"]
                operation, reason = "source_hop", "Open adjacent context for a possibly clipped passage."
            for p in found:
                p["exploration_hop"] = hop + 1
            candidates.extend(found)
            record(operation, reason, found, from_span=origin["id"], hop=hop + 1)
            frontier = [p for p in found if p["id"] not in visited]
            visited.update(p["id"] for p in found)

        # Deduplicate proposals by instance, never merge different sources or revisions.
        by_id = {}
        for candidate in candidates:
            prior = by_id.setdefault(candidate["id"], candidate)
            if candidate.get("retained_working_context"):
                prior["retained_working_context"] = True
        unique = list(by_id.values())
        # Scope proposals using the pinned catalog, not a generated date claim.
        # Embedding similarity alone does not enforce requested years or "latest".
        years = set(re.findall(r"\b20\d{2}\b", question))
        annual = "annual report" in question.casefold()
        financial_table = bool(years and re.search(r"\bfinancial tables?\b", question, re.I))
        preferred = []
        if annual or financial_table:
            filings = sorted(name for name in self.tool.files if "/10-K/" in name)
            if filings and (re.search(r"\blatest\b", question, re.I) or financial_table):
                if financial_table:
                    year = max(map(int, years))
                    filings = [f for f in filings if re.search(fr"/(?:{year}|{year + 1})-", f)]
                if filings:
                    preferred = [filings[-1]]
        elif years:
            terms = set(words(question))
            names = [name for name in self.tool.files
                     if years & set(re.findall(r"\b20\d{2}\b", name))]
            if names:
                preferred = sorted(names, key=lambda f: -len(terms & set(words(
                    f.replace("-", " ").replace("/", " ")))))[:1]
        for name in reversed(preferred):
            if not any(p["file"] == name for p in unique):
                source = self.tool.files[name]
                index = next(i for i, r in enumerate(self.tool.memory.rows) if r["source"] == source)
                found = [self.tool._project(p) for p in self.tool._open(self.tool.corpus_id + f":S{index}")]
                unique.extend(found)
                record("catalog_scope", "Pinned path/date scope proposal; not latest-authority evidence.", found)
        buckets = []
        for facet in facets:
            terms = set(words(facet))
            buckets.append(sorted(unique, key=lambda p: -len(terms & set(words(p["text"])))))
        selected = [p for p in unique if p.get("retained_working_context")]
        while len(selected) < len(unique):
            for bucket in buckets:
                candidate = next((p for p in bucket if p not in selected), None)
                if candidate is not None:
                    selected.append(candidate)
        if preferred:
            selected.sort(key=lambda p: (p["file"] not in preferred,))
        return selected, trace, max(0, len(candidates) - len(unique))


def context(question, corpus="datadog", snapshot=None, intent="answer", budget=None,
            readings=None, continuation=None, *, strategy=None, format=None):
    """Python API using the same long-lived CPU backend as the CLI and MCP."""
    from experiments.latent_context_app import call, ensure_server
    endpoint = ensure_server(corpus=corpus)
    return call(endpoint, "context", question=question, corpus=corpus, snapshot=snapshot,
                intent=intent, budget=budget, readings=readings, continuation=continuation,
                strategy=strategy, format=format)


def context_details(pack_id, corpus="datadog"):
    """Fetch the same full JSON receipt, without routing or claiming fresh verification."""
    from experiments.latent_context_app import call, ensure_server
    return call(ensure_server(corpus=corpus), "context_details", corpus=corpus, pack_id=pack_id)


def ask(question, corpus="datadog", options=None):
    """Python agent API. Uses the same warm local service as CLI/UI/MCP."""
    from experiments.latent_context_app import call, ensure_server
    endpoint = ensure_server(corpus=corpus)
    return call(endpoint, "ask", corpus=corpus, question=question, options=options or {})


def render(pack):
    if pack.get("schema") == COMPACT_SCHEMA:
        return serialized(pack)
    lines = [f"CONTEXT PACK | {pack.get('corpus', pack['corpus_id'])} | {pack.get('strategy', 'one_shot')}",
             pack["question"], "Exact source excerpts, not a generated answer.",
             "Snapshot " + pack["snapshot"],
             "Byte verified only. Semantic support must be judged by the reader.", ""]
    for group in pack.get("evidence_groups", pack.get("facets", [])):
        lines.append("## " + group["topic"])
        if group.get("kind") == "PROVISIONAL_SOURCE_BRIDGE":
            lines += ["PROVISIONAL navigation bridge, not a computed semantic join.",
                      "Both parents: " + ", ".join(group["parents"]),
                      "Witness: " + json.dumps(group["witness"], ensure_ascii=False),
                      "Unresolved: " + "; ".join(group["obligations"])]
        for span in group["spans"]:
            page = "" if span["page"] is None else f" page/leaf {span['page']}"
            lines.append(f"[{span['id']}] {span['file']}{page} "
                         f"bytes {span['start_byte']}:{span['end_byte']} | VERIFIED BYTE IDENTITY")
            lines.append("SHA-256 " + span["sha256"])
            lines.extend("> " + line for line in span["text"].splitlines())
            lines.append("")
    coverage = pack["coverage"]
    lines += ["## Coverage",
              f"{coverage['returned_spans']} spans from {coverage['returned_files']} files; "
              f"index: {coverage['indexed_files']} files / {coverage['indexed_spans']} spans.",
              "Completeness UNKNOWN. Not in memory UNKNOWN. Lexical hints are uncalibrated.",
              "Thin/unseen terms: " + ", ".join(coverage["thin_or_unseen_query_terms"]),
              "## How memory got here"]
    lines += [f"{step['step']}. {step['operation']}: {step['why']} "
              f"({len(step['spans'])} spans)" for step in pack["trace"]]
    lines += ["## Stop", pack["stop_reason"]]
    for item in pack["unresolved"]:
        lines.append("Unresolved: " + item["reason"])
        if "catalog_identity" in item:
            meta = item["catalog_identity"]
            lines += ["Catalog metadata, NOT evidence: " + meta["file"],
                      "Source version " + meta["source_version"],
                      "Blocked budget: " + ", ".join(item["blocked_dimensions"]),
                      "Exact reader (raw bytes, not a complete table): " + item["exact_reader"]]
    ledger = pack.get("omission_ledger")
    if ledger and ledger["total"]:
        lines += [f"## Omission ledger | {ledger['total']} recorded; {ledger['returned']} on this page",
                  "Catalog metadata, NOT evidence. Recover all recorded omissions with these context arguments:",
                  serialized(ledger["recovery"]["arguments"])]
    if pack["continuation"]:
        lines += ["## Continue (executable Python arguments)",
                  serialized(pack["continuation"]["arguments"])]
    lines.append(f"\nMemory query: {pack['seconds']:.3f}s | model calls 0 | generated tokens 0")
    if pack["truncated"]:
        lines.append("Evidence units remain omitted; see unresolved/continuation, not an absence verdict.")
    return "\n".join(lines)
