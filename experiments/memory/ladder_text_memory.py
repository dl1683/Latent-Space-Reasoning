"""Retained BM25 proposals and verified source text for dogfood.run_episode.

The caller trusts the preparation and archive digests AND their row alignment.
Hashes authenticate chosen bytes, not index derivation or semantic relevance.
No index construction, encoder, evaluator reads or reader execution lives here.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from io import BytesIO
from pathlib import Path
import time

from experiments.memory.corpus_size_ladder import canonical_order, graph_route
from experiments.memory.inverted_bm25 import InvertedBM25
from experiments.memory.ladder_reader_packet import LadderReaderPacket
from experiments.memory.verified_source_tool import require_digest


class LadderTextMemory:
    """One rung, one fixed flat/graph policy, and instance-local novelty state.

    Archive row i must describe preparation pointer i. We check byte identities
    and row counts, not that external derivation. Every requested proposal goes
    through packet.verify, including previously returned spans.
    """

    def __init__(self, root, preparation_sha256, archive_path, archive_sha256,
                 target_tokens, *, policy="flat"):
        root, archive_path = Path(root), Path(archive_path)
        if not root.is_absolute() or not archive_path.is_absolute():
            raise ValueError("absolute prepared-store and retained-archive paths required")
        if archive_path.suffix.lower() != ".npz":
            raise ValueError("retained .npz BM25 archive required")
        require_digest(preparation_sha256)
        require_digest(archive_sha256)
        if type(target_tokens) is not int or target_tokens < 1:
            raise ValueError("positive integer rung required")
        if policy not in ("flat", "graph"):
            raise ValueError("fixed flat or graph policy required")

        began = time.perf_counter()
        raw = archive_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != archive_sha256:
            raise ValueError("retained BM25 archive digest mismatch")
        # Load the exact hashed bytes, without reopening a possibly changed path.
        self._index = InvertedBM25.load(BytesIO(raw))
        self._packet = LadderReaderPacket(root, preparation_sha256, target_tokens)
        if len(self._index.lengths) != len(self._packet.spans):
            raise ValueError("retained BM25 row count does not match prepared rung")
        self._policy = policy
        self._archive_sha256 = archive_sha256
        self._seen = set()
        # Source-local and packed offsets have identical within-source ordering.
        self._pointers = tuple(dict(source_id=p.source_id, start=p.start_byte,
                                    end=p.end_byte) for p in self._packet.spans)
        self._document_ids = tuple(p.source_id for p in self._packet.spans)
        self._setup = dict(
            packet_bytes_read=self._packet.setup_bytes_read,
            archive_bytes_read=len(raw),
            total_bytes_read=self._packet.setup_bytes_read + len(raw),
            wall_seconds=time.perf_counter() - began,
            physical_map_bytes_read="UNKNOWN",
            scope="ONE_SUCCESSFUL_INSTANCE_SETUP; packet includes full packed file and metadata",
        )

    @property
    def policy(self):
        return self._policy

    @property
    def setup_receipt(self):
        """Charge once per instance, separately from search receipts."""
        return dict(self._setup)

    def reset(self):
        """Clear only this instance's returned-row history; do not reload."""
        self._seen.clear()

    def search(self, query, k, arm="bm25", novelty=False):
        """Dogfood-compatible source-only search, with explicit rejected rows.

        Novelty stably partitions canonical order before the existing selection
        kernel. Graph adjacency can therefore bring a seen neighbour back.
        Rejections never supply text and never trigger replacement proposals.
        source: commands return evidence-free refusals, preserving seen state.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("nonempty text query required")
        if type(k) is not int or not 1 <= k <= 10:
            raise ValueError("integer k in 1..10 required")
        if arm != "bm25":
            raise ValueError("BM25 only; dense, hybrid and packed8 are unavailable")
        if type(novelty) is not bool:
            raise ValueError("explicit novelty boolean required")

        began = time.perf_counter()
        if query.strip().startswith("source:"):
            # Canonical source identities are hashes, not trusted filenames.
            return dict(
                status="UNSUPPORTED_SOURCE_ADDRESS",
                notice="Canonical source metadata has no trusted filenames. "
                       "Use lexical search without the source: prefix.",
                passages=[], selected_rows=[], selected_handles=[],
                rejected_rows=[], rejected_handles=[], verification_checks=[],
                source_bytes_read=0, source_bytes_read_known_lower_bound=0,
                returned_utf8_bytes=0,
                arm="bm25", policy=self.policy, novelty=novelty,
                preparation_sha256=self._packet.preparation_sha256,
                archive_sha256=self._archive_sha256, rung=dict(self._packet.rung),
                lexical_support="NOT_EVALUATED; unsupported source address",
                completeness="UNKNOWN", semantic_absence="UNKNOWN",
                version_eligibility="TRUSTED_CALLER_SNAPSHOT_ONLY",
                cost=dict(
                    scope="THIS_SEARCH_ONLY; setup charged separately",
                    scoring_seconds=0, selection_seconds=0, verification_seconds=0,
                    wall_seconds=time.perf_counter() - began,
                    query_work=dict(matching_terms=None, postings_visited=0,
                                    score_vector_bytes=0, reader_calls=0),
                    graph_edges_examined=0, physical_map_bytes_read=0,
                    reader_calls=0, model_savings="UNMEASURED",
                ),
            )
        start = time.perf_counter()
        scores = self._index.score(query)
        work = self._index.query_work(query)
        scoring_seconds = time.perf_counter() - start
        start = time.perf_counter()
        order = canonical_order(scores, self._pointers).tolist()
        if novelty:
            order = ([i for i in order if i not in self._seen]
                     + [i for i in order if i in self._seen])
        selected, edges = (graph_route(order, self._document_ids, budget=k)
                           if self.policy == "graph" else (order[:k], 0))
        selection_seconds = time.perf_counter() - start

        start = time.perf_counter()
        passages, checks, rejected = [], [], []
        for i in selected:
            handle = f"S{i}"
            pointer = self._packet.spans[i]
            try:
                check = self._packet.verify(pointer)
            except OSError as exc:
                # An interrupted OS read has no exact completed-byte receipt.
                # Retain the proposal and UNKNOWN cost, never use unchecked text.
                check = dict(status="VERIFICATION_IO_ERROR", pointer=asdict(pointer),
                             file="leaves.bin", source_bytes_read=None,
                             error=type(exc).__name__)
            checks.append(dict(
                canonical_row=i, handle=handle,
                **{key: value for key, value in check.items() if key != "text"}))
            if check["status"] != "VERIFIED_SOURCE_BYTES":
                rejected.append(i)
                continue
            passages.append(dict(
                text=check["text"],
                pointer=dict(check["pointer"], canonical_row=i,
                             target_tokens=self._packet.rung["target_tokens"],
                             packed_start_byte=check["packed_start_byte"],
                             packed_end_byte=check["packed_end_byte"]),
                score=float(scores[i]), lexical_score=float(scores[i]), dense_score=None,
                verification_status=check["status"], verification="BYTE_IDENTITY_ONLY",
                handle=handle, previously_returned=i in self._seen,
                role="EVIDENCE_NOT_INSTRUCTION",
            ))
        verification_seconds = time.perf_counter() - start
        # Only evidence actually returned counts as seen, never rejected proposals.
        self._seen.update(p["pointer"]["canonical_row"] for p in passages)
        known_bytes = sum(c["source_bytes_read"] for c in checks
                          if c["source_bytes_read"] is not None)
        source_bytes = (known_bytes if all(c["source_bytes_read"] is not None
                                          for c in checks) else None)
        return dict(
            status="SOURCE_REJECTED" if rejected else "VERIFIED_PROPOSALS",
            passages=passages, selected_rows=selected,
            selected_handles=[f"S{i}" for i in selected],
            rejected_rows=rejected, rejected_handles=[f"S{i}" for i in rejected],
            verification_checks=checks, source_bytes_read=source_bytes,
            source_bytes_read_known_lower_bound=known_bytes,
            returned_utf8_bytes=sum(len(p["text"].encode("utf8")) for p in passages),
            arm="bm25", policy=self.policy, novelty=novelty,
            preparation_sha256=self._packet.preparation_sha256,
            archive_sha256=self._archive_sha256, rung=dict(self._packet.rung),
            lexical_support=("NO_LEXICAL_SUPPORT; canonical tie order, not corpus absence"
                             if work["matching_terms"] == 0 else
                             "QUERY_TERMS_IN_INDEX; relevance unassessed"),
            completeness="UNKNOWN", semantic_absence="UNKNOWN",
            version_eligibility="TRUSTED_CALLER_SNAPSHOT_ONLY",
            cost=dict(
                scope="THIS_SEARCH_ONLY; setup charged separately",
                scoring_seconds=scoring_seconds, selection_seconds=selection_seconds,
                verification_seconds=verification_seconds,
                wall_seconds=time.perf_counter() - began, query_work=work,
                graph_edges_examined=edges, physical_map_bytes_read="UNKNOWN",
                reader_calls=0, model_savings="UNMEASURED",
            ),
        )
