"""Source-bound adapter over retained generic vectors, not a corpus-trained map.

Rows are inherited encoder chunks, not newly encoded whole leaves. Identities
include source occurrence and snapshot even when leaf payload hashes coincide.
All edges propose navigation; none certifies semantic support.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from experiments.dogfood import digest, ram_guard
from experiments.memory.explicit_reference_memory import inline_links
from experiments.memory.latent_frontend import title_graph
from experiments.memory.refined_routing_map import RefinedMap
from experiments.context_costs import charge, visit_edge


EDGE_TYPES = frozenset(("containment", "adjacency", "explicit_reference",
                        "scoped_association", "revision", "geometric_neighbor"))


class DeferredOutgoing(dict):
    """Materialize large-scope title edges only for nodes actually visited."""

    def __init__(self, values, pending, build):
        super().__init__(values)
        self.pending, self.build = pending, build

    def __getitem__(self, key):
        if key in self.pending:
            scope, position = self.pending[key]
            edges = self.build(scope, position)
            self[key] = super().get(key, ()) + tuple(edges)
            del self.pending[key]
        return super().__getitem__(key)

    def get(self, key, default=None):
        return self[key] if key in self or key in self.pending else default


@dataclass(frozen=True)
class SourceVersion:
    id: str
    original_hash: str
    extraction_revision: str


@dataclass(frozen=True)
class Span:
    instance_id: str
    source_version: str
    leaf_hash: str
    coordinate_space: str
    start: int
    end: int
    span_hash: str
    vector_row: int | None
    scope: tuple[str, ...]


@dataclass(frozen=True)
class Edge:
    from_id: str
    to_id: str
    type: str
    witness_span: Span | None
    binding_constraints: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if self.type not in EDGE_TYPES:
            raise ValueError("unknown evidence edge type")
        if self.type in ("explicit_reference", "scoped_association") and self.witness_span is None:
            raise ValueError("literal relationships require a source-bound witness")
        if self.type == "geometric_neighbor" and self.witness_span is not None:
            raise ValueError("geometry has no source witness")

    @property
    def semantic_support(self):
        return "PROPOSAL_ONLY"


class EvidenceMap:
    """Warm snapshot metadata plus an 8-bit view of the existing FP32 bank.

    Reuses literal-link resolution and the existing scoped whole-title matcher.
    Construction reads retained row text, never reopens/re-encodes the corpus.
    Revision/geometry types are supported but no revision order or all-pairs
    neighbor graph is invented when the inherited snapshot supplies neither.
    """

    def __init__(self, tool):
        ram_guard()
        started = time.perf_counter()
        self.hash_bytes = self.hash_checks = 0
        self.tool, self.snapshot = tool, tool.snapshot
        memory = tool.memory
        bank = memory.vectors
        if not isinstance(bank, np.ndarray) or bank.dtype != np.float32:
            raise ValueError("retained FP32 frozen-encoder bank required")
        self.bank = bank
        self.spans, self.sources, self.source_rows, self.leaves = {}, {}, {}, {}
        self.omissions = []
        outgoing = defaultdict(list)
        manifest_files = {item["path"]: item for item in memory.manifest["files"]}
        by_source = defaultdict(list)
        for i, row in enumerate(memory.rows):
            item = manifest_files.get(row["source"])
            if item is None or item["sha256"] != row["source_sha256"]:
                raise ValueError("row source/version outside snapshot")
            raw = row["text"].encode("utf-8")
            if (type(row["start_byte"]) is not int or type(row["end_byte"]) is not int
                    or not 0 <= row["start_byte"] < row["end_byte"]
                    or row["end_byte"] - row["start_byte"] != len(raw)
                    or self._digest(raw) != row["span_sha256"]):
                raise ValueError("invalid retained source span")
            by_source[row["source"]].append(i)
        for path, item in manifest_files.items():
            source_id = self.node_id("source", path, item["sha256"])
            version = SourceVersion(source_id, item["sha256"],
                str(item.get("extraction_revision", "UNKNOWN:" + item.get("extraction", "UNKNOWN"))))
            self.sources[path] = version
            order = sorted(by_source[path], key=lambda i: (
                type(memory.rows[i]["page"]).__name__, memory.rows[i]["page"],
                memory.rows[i]["start_byte"], i))
            self.source_rows[path] = tuple(order)
            if not order:
                self.omissions.append(dict(source=path, reason="source_has_no_indexed_rows"))
            leaf_rows = defaultdict(list)
            for i in order:
                row = memory.rows[i]
                span = Span(tool.corpus_id + f":S{i}", source_id, row["leaf_sha256"],
                            "EXTRACTED_LEAF_UTF8", row["start_byte"], row["end_byte"],
                            row["span_sha256"], i, (path, item["sha256"], str(row["page"])))
                self.spans[span.instance_id] = span
                leaf_rows[(row["page"], row["leaf_sha256"])].append(i)
            for (page, leaf_hash), ids in leaf_rows.items():
                leaf_id = self.node_id("leaf", path, item["sha256"], str(page), leaf_hash)
                leaf_path = memory.directory / "leaves" / (leaf_hash + ".txt")
                size = leaf_path.stat().st_size
                self.leaves[leaf_id] = tuple(ids)
                constraints = (("source_version", source_id), ("page", str(page)))
                outgoing[source_id].append(Edge(source_id, leaf_id, "containment", None, constraints))
                outgoing[leaf_id].append(Edge(leaf_id, source_id, "containment", None, constraints))
                end = 0
                for i in ids:
                    span = self.spans[tool.corpus_id + f":S{i}"]
                    if span.end > size:
                        raise ValueError("retained span exceeds leaf size")
                    if span.start > end:
                        self.omissions.append(dict(source=path, page=page, start=end,
                            end=span.start, reason="unencoded_leaf_interval"))
                    end = max(end, span.end)
                    outgoing[leaf_id].append(Edge(leaf_id, span.instance_id, "containment", None, constraints))
                    outgoing[span.instance_id].append(Edge(span.instance_id, leaf_id, "containment", None, constraints))
                if end < size:
                    self.omissions.append(dict(source=path, page=page, start=end, end=size,
                                               reason="unencoded_leaf_interval"))
            for left, right in zip(order, order[1:]):
                a, b = tool.corpus_id + f":S{left}", tool.corpus_id + f":S{right}"
                binding = (("source_version", source_id),)
                outgoing[a].append(Edge(a, b, "adjacency", None, binding))
                outgoing[b].append(Edge(b, a, "adjacency", None, binding))
        for item in memory.manifest.get("empty_pages", []):
            self.omissions.append(dict(item, reason="empty_extracted_leaf"))
        for span in self.spans.values():
            row = memory.rows[span.vector_row]
            if Path(row["source"]).suffix.casefold() not in (".md", ".markdown"):
                continue
            for a, b, literal, destination in inline_links(row["text"]):
                resolved = memory._resolve(destination, row["source"])
                if resolved["resolution"] != "MANIFEST_RESOLVED_UNFETCHED":
                    self.omissions.append(dict(span=span.instance_id, destination=destination,
                                               reason="unresolved_literal_reference"))
                    continue
                target = next(path for path in self.sources
                              if "source:" + memory.source_ids[path] == resolved["address"])
                start = span.start + len(row["text"][:a].encode("utf-8"))
                witness = Span(span.instance_id + f":ref{a}", span.source_version,
                    span.leaf_hash, span.coordinate_space, start,
                    start + len(literal.encode("utf-8")), self._digest(literal.encode("utf-8")),
                    None, span.scope)
                outgoing[span.instance_id].append(Edge(span.instance_id,
                    tool.corpus_id + f":S{self.source_rows[target][0]}",
                    "explicit_reference", witness,
                    (("origin_version", span.source_version),
                     ("target_version", self.sources[target].id),
                     ("anchor", "NOT_RESOLVED"))))
        # The retained index has no entity extractor. Reuse literal title edges
        # only among source-first chunks in the same directory, and disclose scope.
        scopes = defaultdict(list)
        for path, ids in self.source_rows.items():
            if ids:
                scopes[str(Path(path).parent)].append(ids[0])
        deferred = {}

        def title_edges(scope_data, position, edges):
            scope, ids, paragraphs = scope_data
            origin = self.spans[tool.corpus_id + f":S{ids[position]}"]
            text = paragraphs[position]["paragraph_text"].encode("utf-8")
            result = []
            for edge in edges:
                a, b = edge["start_byte"], edge["end_byte"]
                witness = Span(origin.instance_id + f":title{a}", origin.source_version,
                    origin.leaf_hash, origin.coordinate_space, origin.start + a,
                    origin.start + b, self._digest(text[a:b]), None, origin.scope)
                target = tool.corpus_id + f":S{ids[edge['target']]}"
                result.append(Edge(origin.instance_id, target,
                    "scoped_association", witness, (("directory", scope),
                     ("origin_version", origin.source_version),
                     ("target_version", self.spans[target].source_version))))
            return result

        def deferred_edges(scope_data, position):
            edges = title_graph(scope_data[2], origins=[position])[0]
            result = title_edges(scope_data, position, edges)
            for edge in result:
                visit_edge(edge)
            # Earlier pack metadata may still reference the previous counts.
            self.edge_counts = dict(self.edge_counts, scoped_association=(
                self.edge_counts.get("scoped_association", 0) + len(result)))
            return result

        for scope, ids in scopes.items():
            paragraphs = [dict(idx=i, title=memory.rows[i]["text"].splitlines()[0].strip("# ")[:240],
                               paragraph_text=memory.rows[i]["text"]) for i in ids]
            scope_data = (scope, ids, paragraphs)
            # Avoid an eager quadratic graph for large flat source directories.
            # The same ordered edges remain available to expand and PPR on access.
            if len(ids) > 512:
                for position, i in enumerate(ids):
                    deferred[tool.corpus_id + f":S{i}"] = (scope_data, position)
            else:
                for position, edges in enumerate(title_graph(paragraphs)):
                    outgoing[tool.corpus_id + f":S{ids[position]}"].extend(
                        title_edges(scope_data, position, edges))
        self.outgoing = DeferredOutgoing(
            {key: tuple(edges) for key, edges in outgoing.items()}, deferred, deferred_edges)
        self.edge_counts = dict(Counter(visit_edge(e).type for edges in self.outgoing.values() for e in edges))
        charge(exact_vector_rows_read=len(bank), exact_vector_bytes_read=int(bank.nbytes))
        self.routing = RefinedMap.build(bank, 8)
        self.setup_costs = dict(seconds=time.perf_counter() - started,
            exact_vector_rows_read=len(bank), exact_vector_bytes_read=int(bank.nbytes),
            retained_exact_vector_bytes=int(bank.nbytes),
            compressed_map_bytes=int(self.routing.payload_bytes),
            retained_row_text_bytes_read=sum(len(r["text"].encode("utf-8")) for r in memory.rows),
            hash_bytes=self.hash_bytes, hash_checks=self.hash_checks,
            source_opens=0, leaf_opens=0, leaf_size_stats=len(self.leaves),
            edge_records_built=sum(self.edge_counts.values()),
            **(dict(deferred_title_origins=len(deferred),
                    edge_counts_scope="materialized edges only; title edges generated on node access")
               if deferred else {}),
            physical_io="UNKNOWN", numeric_accounting="logical payload, not allocator traffic")
        ram_guard()

    def node_id(self, kind, *parts):
        return kind + ":" + self._digest((self.snapshot + "\0" + "\0".join(parts)).encode("utf-8"))

    def _digest(self, raw):
        self.hash_bytes += len(raw)
        self.hash_checks += 1
        return digest(raw)

    def coverage(self):
        return dict(indexed_vector_rows=len(self.spans), accounted_vector_rows=len(self.spans),
                    indexed_leaf_occurrences=len(self.leaves), source_occurrences=len(self.sources),
                    known_omissions=len(self.omissions),
                    omission_counts=dict(Counter(o["reason"] for o in self.omissions)),
                    extraction_universe="UNKNOWN: inherited manifest has no full leaf inventory",
                    representation="retained chunk vectors; no new whole-leaf embeddings",
                    association_scope="literal first-chunk titles within directory only",
                    **(dict(deferred_title_origins=len(self.outgoing.pending))
                       if self.outgoing.pending else {}),
                    revision_edges="not supplied; no chronological authority inferred",
                    geometric_edges="not precomputed; no hidden per-hop vector sweeps")
