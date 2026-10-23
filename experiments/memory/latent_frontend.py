"""Encoded-once compressed routing -> title references -> verified evidence.

C4/C7/C8 supporting SLICE, not latent-core reasoning. Reuses RefinedMap's
quantized candidate filter and exact-vector refinement. The exact bank is retained
and charged, so the small map is NOT the full memory footprint. No model loads,
training, network access, evaluator fields or answer generation occur here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from experiments.memory.compressed_routing_map import BM25, reciprocal_rank_fusion
from experiments.memory.refined_routing_map import RefinedMap, feature_scores, rank


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def source_rows(paragraphs):
    """Reject rather than silently retain answer/support/evaluator metadata."""
    result = []
    for p in paragraphs:
        if (set(p) != {"idx", "title", "paragraph_text"} or type(p["idx"]) is not int
                or not all(isinstance(p[k], str) for k in ("title", "paragraph_text"))):
            raise ValueError("source-only idx/title/paragraph_text required")
        result.append(dict(p))
    if not result or len({p["idx"] for p in result}) != len(result):
        raise ValueError("nonempty unique source indices required")
    return result


def title_graph(paragraphs, *, origins=None):
    """Literal whole-title mentions, not semantic dependencies or entailment.

    All matching destinations survive ambiguous titles. No support labels or
    question enters construction. Edges cite the origin's exact body byte span.
    Optional origins select rows without changing destination or match order.
    """
    # Repeated extraction headers share a pattern and one scan per origin.
    # Keep every destination, including ambiguous duplicate titles.
    compiled = {}
    for p in paragraphs:
        title = p["title"].strip()
        if title not in compiled:
            compiled[title] = (re.compile(r"(?<!\w)" + re.escape(title) + r"(?!\w)", re.I)
                               if len(title) >= 3 else None)
    patterns = [compiled[p["title"].strip()] for p in paragraphs]
    graph = []
    for i in range(len(paragraphs)) if origins is None else origins:
        p = paragraphs[i]
        edges = []
        matches = {}
        for j, pattern in enumerate(patterns):
            if i == j or pattern is None:
                continue
            if pattern not in matches:
                matches[pattern] = [
                    (len(p["paragraph_text"][:match.start()].encode("utf8")),
                     len(p["paragraph_text"][:match.end()].encode("utf8")))
                    for match in pattern.finditer(p["paragraph_text"])]
            for start, end in matches[pattern]:
                edges.append(dict(target=j, start_byte=start, end_byte=end))
        graph.append(edges)
    return graph


def expand(order, graph, k, enabled=True, destination_scores=None):
    """Same seed/edge/fill policy for every routing arm, <=one graph hop."""
    order = list(map(int, order))
    if not enabled:
        return order[:k], 0
    seeds = order[:max(1, (k + 1)//2)]
    selected, visits = list(seeds), 0
    position = {j: n for n, j in enumerate(order)}
    destinations = set()
    for i in seeds:
        for edge in graph[i]:
            visits += 1
            destinations.add(edge["target"])
    destination_order = sorted(destinations, key=(
        (lambda j: (-destination_scores[j], j)) if destination_scores is not None
        else (lambda j: position[j])))
    for j in destination_order + order:
        if len(selected) >= k:
            break
        if j not in selected:
            selected.append(j)
    return selected, visits


class LatentFrontend:
    """One allow-listed task corpus; a caller supplies its generic query encoder."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        manifest = json.loads((self.directory / "manifest.json").read_bytes())
        self.manifest = manifest
        for name, expected in manifest["files"].items():
            if sha((self.directory / name).read_bytes()) != expected:
                raise ValueError("frontend artifact changed: " + name)
        self.rows = source_rows(json.loads((self.directory / "sources.json").read_bytes()))
        self.bank = np.load(self.directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
        self.map = RefinedMap.load(self.directory / "map.npz")
        self.graph = json.loads((self.directory / "graph.json").read_bytes())
        if len(self.rows) != len(self.bank) or len(self.graph) != len(self.rows):
            raise ValueError("source/vector/graph mismatch")
        self.lexical = BM25([p["title"] + "\n" + p["paragraph_text"] for p in self.rows])

    @classmethod
    def build(cls, directory, paragraphs, vectors, *, encoder_identity, bits=8):
        """Only task source documents enter the memory; question/labels do not."""
        rows = source_rows(paragraphs)
        x = np.asarray(vectors)
        if x.dtype != np.float32 or x.ndim != 2 or len(x) != len(rows):
            raise ValueError("aligned FP32 pretrained vectors required")
        memory = RefinedMap.build(x, bits)
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        def save(name, value):
            with (directory / name).open("x", encoding="utf8") as stream:
                json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        save("sources.json", rows)
        save("graph.json", title_graph(rows))
        leaves = directory / "leaves"
        leaves.mkdir()
        pointers = []
        for p in rows:
            raw = (p["title"] + "\n" + p["paragraph_text"]).encode("utf8")
            h = sha(raw)
            path = leaves / (h + ".txt")
            if not path.exists():
                path.write_bytes(raw)
            pointers.append(dict(source_sha256=h, span_sha256=h, start_byte=0,
                                 end_byte=len(raw), idx=p["idx"]))
        save("pointers.json", pointers)
        with (directory / "vectors.npy").open("xb") as stream:
            np.save(stream, x, allow_pickle=False)
        memory.save(directory / "map.npz")
        files = {name: sha((directory / name).read_bytes()) for name in (
            "sources.json", "graph.json", "vectors.npy", "map.npz", "pointers.json")}
        save("manifest.json", dict(schema=1, files=files, encoder_identity=encoder_identity,
            corpus_encodes=1, bits=bits, documents=len(rows), dimensions=x.shape[1],
            map_numeric_bytes=memory.payload_bytes, exact_bank_numeric_bytes=x.nbytes,
            source_leaf_bytes=sum(p.stat().st_size for p in leaves.iterdir()),
            source_only=True, completeness="UNKNOWN", semantic_absence="UNKNOWN"))
        return cls(directory)

    def retrieve(self, query, query_vector=None, *, k=6, arm="latent", graph=True):
        """ONE shared front-end: query -> routes -> references -> exact evidence.

        `packing` keeps the original source order. `latent` uses quantized bounds
        plus retained exact vectors; `dense`/`hybrid` use the full exact bank.
        Logical bytes exclude Python object overhead/OS cache effects. BM25 byte
        accounting explicitly remains unknown rather than calling it zero.
        """
        if (not isinstance(query, str) or not query.strip() or type(k) is not int
                or not 1 <= k <= len(self.rows) or type(graph) is not bool
                or arm not in {"latent", "bm25", "hybrid", "dense", "packing"}):
            raise ValueError("valid query, arm and bounded k required")
        started = time.perf_counter()
        numeric_bytes = exact_bytes = 0
        candidates = 0
        destination_scores = None
        if arm in {"latent", "dense", "hybrid"}:
            q = np.asarray(query_vector)
            if q.dtype != np.float32 or q.shape != (self.bank.shape[1],) or not np.isfinite(q).all():
                raise ValueError("aligned finite FP32 query required")
        if arm == "latent":
            # Full order is unnecessary: seed plus fill can require at most k.
            routed = self.map.search(q, lambda ids: self.bank[ids], k=k)
            order = routed["indices"].tolist()
            # Graph destinations outside the short list are ordered deterministically.
            order += [i for i in range(len(self.rows)) if i not in order]
            numeric_bytes = self.map.payload_bytes
            exact_bytes = routed["logical_exact_vector_bytes"]
            candidates = len(routed["candidate_indices"])
            if graph:
                targets = sorted({e["target"] for i in order[:max(1, (k+1)//2)]
                                  for e in self.graph[i]})
                if targets:
                    vectors = np.asarray(self.bank[targets])
                    destination_scores = dict(zip(targets, feature_scores(vectors, q)))
                    exact_bytes += int(vectors.nbytes)
        elif arm == "packing":
            order = list(range(len(self.rows)))
        else:
            lexical = self.lexical.score(query) if arm in {"bm25", "hybrid"} else None
            dense = feature_scores(self.bank, q) if arm in {"dense", "hybrid"} else None
            if arm == "hybrid":
                values = reciprocal_rank_fusion(dense, lexical) if np.any(lexical) else dense
            else:
                values = dense if arm == "dense" else lexical
            order = rank(values).tolist()
            exact_bytes = int(self.bank.nbytes) if dense is not None else 0
        selected, edge_visits = expand(order, self.graph, k, graph and arm != "packing",
                                      destination_scores=destination_scores)
        pointer_raw = (self.directory / "pointers.json").read_bytes()
        if sha(pointer_raw) != self.manifest["files"]["pointers.json"]:
            raise ValueError("source pointers changed")
        pointers = json.loads(pointer_raw)
        if len(pointers) != len(self.rows) or any(
                p.get("idx") != row["idx"] for p, row in zip(pointers, self.rows)):
            raise ValueError("source pointer row mismatch")
        evidence, source_bytes = [], 0
        for i in selected:
            p = pointers[i]
            h = p["source_sha256"]
            if not re.fullmatch("[0-9a-f]{64}", h):
                raise ValueError("invalid source identity")
            path = self.directory / "leaves" / (h + ".txt")
            if path.is_symlink() or path.resolve().parent != (self.directory / "leaves").resolve():
                raise ValueError("source escapes allow-list")
            raw = path.read_bytes()
            source_bytes += len(raw)
            a, b = p["start_byte"], p["end_byte"]
            if (type(a) is not int or type(b) is not int or not 0 <= a < b <= len(raw)
                    or sha(raw) != h or sha(raw[a:b]) != p["span_sha256"]):
                raise ValueError("source version/span changed")
            text = raw[a:b].decode("utf8")
            if text != self.rows[i]["title"] + "\n" + self.rows[i]["paragraph_text"]:
                raise ValueError("source projection differs from leaf")
            evidence.append(dict(idx=p["idx"], title=self.rows[i]["title"],
                paragraph_text=self.rows[i]["paragraph_text"], text=text, pointer=p,
                verification="BYTE_IDENTITY_ONLY", role="EVIDENCE_NOT_INSTRUCTION"))
        graph_bytes = sum(len(self.graph[i])*12 for i in order[:max(1, (k+1)//2)]) if edge_visits else 0
        return dict(evidence=evidence, selected_indices=selected, arm=arm,
            seconds=time.perf_counter()-started, exact_vector_candidates=candidates,
            map_bytes_touched=numeric_bytes, exact_vector_bytes_touched=exact_bytes,
            graph_edge_bytes_touched=graph_bytes, source_bytes_touched=source_bytes,
            lexical_bytes_touched=None if arm in {"bm25", "hybrid"} else 0,
            pointer_file_bytes_touched=len(pointer_raw),
            physical_io="UNKNOWN", reference_edges_visited=edge_visits,
            reader_calls=0, corpus_reencodes=0, completeness="UNKNOWN",
            semantic_absence="UNKNOWN")


def retrieve_evidence(frontend, query, query_vector=None, **options):
    """Stable function for the shared harness; see LatentFrontend.build."""
    return frontend.retrieve(query, query_vector, **options)
