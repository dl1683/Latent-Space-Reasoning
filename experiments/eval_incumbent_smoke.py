"""Finite GOLDEN retrieval SLICE; citation coverage is NOT answer quality.

Default: BM25 only, no neural model loaded or encoding performed. Dense/hybrid
require an exactly compatible saved embedding cache or explicit --encode-cpu.
CPU encoding is explicit opt-in and requires a locally cached model, offline
operation and more than 20 GiB available RAM before each batch.
All writes are confined to a NEW directory below the configured baseline root.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "workspace/eval_corpora"
WRITE_ROOT = BASE / "baseline/incumbent_smoke_20261017"
PACK_REL = "drafts/repo_gold/smoke_cross_domain_v2.json"
PACK_SHA = "c8d34c760ef15b6e096f1b4d17dc58771ad7c88a58901da94872fdb4ab103d0e"
MANIFEST_SHA = "cef38360e875e02cf797b5fda2cc2f09d483950bbafa43d0d671034c5c647067"
SNAPSHOT = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_FILES = (
    "config.json", "config_sentence_transformers.json", "model.safetensors",
    "modules.json", "sentence_bert_config.json", "special_tokens_map.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.txt", "1_Pooling/config.json",
)
SOURCE_IDS = ("agent-tooling", "mapu", "latent-space")
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
    "CUDA_VISIBLE_DEVICES": "", "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def save(path: Path, value) -> int:
    data = json_bytes(value)
    with path.open("xb") as stream:
        stream.write(data)
    return len(data)


def pin(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": sha(data)}


def checked_json(path: Path, expected: str | None = None):
    data = path.read_bytes()
    if expected is not None and sha(data) != expected:
        raise ValueError(f"Pinned input changed: {path}")
    return json.loads(data.decode("utf-8-sig"))


def memory_check(samples: list, stage: str) -> float:
    import psutil

    available = psutil.virtual_memory().available / 2**30
    samples.append({"stage": stage, "free_ram_gib": available,
                    "process_rss_bytes": psutil.Process().memory_info().rss})
    if available <= 20:
        raise RuntimeError(f"RAM gate refused {stage}: {available:.3f} GiB <= 20 GiB")
    return available


def load_sources(manifest: dict, pack: dict) -> list[dict]:
    """Only original source_documents are eligible; no gold/answer text enters."""
    documents = []
    # Inputs are authenticated by the caller before this adapter runs. Preserve
    # their original identifiers; neutral public aliases never rewrite receipts.
    source_ids = list(SOURCE_IDS)
    known = set(source_ids[1:])
    historical = set(pack["eligible_source_documents"]) - known
    if source_ids[0] not in historical:
        if len(historical) != 1:
            raise ValueError("Ambiguous historical source identity")
        source_ids[0] = historical.pop()
    for source_id in source_ids:
        source = next(s for s in manifest["sources"] if s["id"] == source_id)
        eligible = pack["eligible_source_documents"][source_id]
        if set(eligible) != {d["id"] for d in source["documents"]}:
            raise ValueError(f"Eligible-document set changed: {source_id}")
        for item in sorted(source["documents"], key=lambda d: d["id"]):
            path = Path(item["path"])
            raw = path.read_bytes()
            if sha(raw) != item["sha256"] or len(raw) != item["bytes"]:
                raise ValueError(f"Original source mismatch: {path}")
            text = raw.decode("utf-8")  # strict; no replacement or newline rewriting
            lines = text.splitlines(keepends=True)
            starts, offset = [], 0
            for line in lines:
                starts.append(offset)
                offset += len(line.encode("utf-8"))
            if offset != len(raw):
                raise ValueError(f"Line reconstruction mismatch: {path}")
            documents.append({"source": source_id, "document": item["id"],
                              "path": str(path), "sha256": item["sha256"],
                              "raw": raw, "text": text, "lines": lines,
                              "line_byte_starts": starts})
    if len(documents) != 29 or sum(len(d["raw"]) for d in documents) != 588630:
        raise ValueError("Expected exactly 29 original documents / 588630 bytes")
    inventory = {(str(Path(d["path"]).resolve()), d["sha256"]) for d in pack["source_inventory"]}
    if inventory != {(str(Path(d["path"]).resolve()), d["sha256"]) for d in documents}:
        raise ValueError("Pack inventory and original manifest disagree")
    return documents


def split_text(text: str, tokenizer, token_limit: int, byte_limit: int,
               line_limit: int) -> list[str]:
    """Greedy whole-line packing; split a long line losslessly if necessary.

    Each exact fragment is re-tokenized without truncation and fits both limits.
    Binary search need not find the longest possible prefix: only a valid one.
    No whitespace, Unicode, SOURCE_PATH header or newline is silently discarded.
    """
    def fits(value):
        return (len(value.encode("utf-8")) <= byte_limit
                and len(tokenizer.encode(value).ids) <= token_limit)

    result, pending = [], ""
    for line in text.splitlines(keepends=True):
        if pending and (not fits(pending + line)
                        or len((pending + line).splitlines()) > line_limit):
            result.append(pending)
            pending = ""
        while not fits(line):
            lo, hi = 0, len(line)
            while lo < hi:
                middle = (lo + hi + 1) // 2
                if fits(line[:middle]):
                    lo = middle
                else:
                    hi = middle - 1
            if lo == 0:
                raise ValueError("A single character cannot fit the chunk budget")
            result.append(line[:lo])
            line = line[lo:]
        pending += line
    if pending:
        result.append(pending)
    if "".join(result) != text or any(not fits(part) for part in result):
        raise ValueError("Lossless nontruncating text partition failed")
    return result


def make_chunks(documents: list, tokenizer, args) -> list[dict]:
    from bisect import bisect_right

    chunks = []
    for doc in documents:
        offset = 0
        for part in split_text(doc["text"], tokenizer, args.chunk_tokens,
                               args.passage_bytes, args.chunk_lines):
            raw = part.encode("utf-8")
            end = offset + len(raw)
            start_line = bisect_right(doc["line_byte_starts"], offset)
            end_line = bisect_right(doc["line_byte_starts"], end - 1)
            if doc["raw"][offset:end] != raw:
                raise ValueError("Chunk byte-pointer mismatch")
            chunks.append({"id": f"{doc['source']}/{doc['document']}@{offset}:{end}",
                           "source": doc["source"], "document": doc["document"],
                           "source_sha256": doc["sha256"], "source_path": doc["path"],
                           "line_start": start_line, "line_end": end_line,
                           "byte_start": offset, "byte_end": end, "bytes": len(raw),
                           "sha256": sha(raw), "encoder_tokens": len(tokenizer.encode(part).ids),
                           "text": part})
            offset = end
        if offset != len(doc["raw"]):
            raise ValueError("Incomplete ingestion")
    return chunks


def lexical_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def bm25_index(chunks: list) -> dict:
    counts = [dict(sorted(Counter(lexical_tokens(c["text"])).items())) for c in chunks]
    lengths = [sum(row.values()) for row in counts]
    frequency = Counter(term for row in counts for term in row)
    count = len(chunks)
    return {"formula": "sum_unique_query_terms log(1+(N-df+0.5)/(df+0.5))*tf*(k1+1)/(tf+k1*(1-b+b*dl/avgdl))",
            "k1": 1.2, "b": 0.75, "tokenization": "Unicode casefold then regex \\w+; no stopword/stemmer; query terms once",
            "average_length": sum(lengths) / count, "lengths": lengths,
            "term_counts": counts,
            "idf": {t: math.log1p((count - df + 0.5) / (df + 0.5))
                    for t, df in sorted(frequency.items())}}


def bm25_scores(index: dict, query: str) -> list[float]:
    terms = sorted(set(lexical_tokens(query)))
    k1, b, avg = index["k1"], index["b"], index["average_length"]
    return [sum(index["idf"].get(t, 0.0) * tf.get(t, 0) * (k1 + 1)
                / (tf.get(t, 0) + k1 * (1 - b + b * length / avg)) for t in terms)
            for tf, length in zip(index["term_counts"], index["lengths"])]


def rank(scores: list[float]) -> list[int]:
    # Input chunk order is stable source/document/byte order; ties use that order.
    if not all(math.isfinite(float(value)) for value in scores):
        raise ValueError("Non-finite retrieval score")
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))


def select_passages(order: list[int], scores, chunks: list, args) -> list[dict]:
    selected, used = [], 0
    for corpus_rank, i in enumerate(order, 1):
        if len(selected) == args.top_k:
            break
        chunk = chunks[i]
        if used + chunk["bytes"] > args.byte_budget:
            continue  # deterministic whole-passage skip; never truncate outputs
        selected.append({"rank": len(selected) + 1, "corpus_rank": corpus_rank,
                         "score": float(scores[i]), **chunk})
        used += chunk["bytes"]
    return selected


def model_pins(snapshot: Path) -> dict:
    if snapshot.name != SNAPSHOT:
        raise ValueError(f"Only pinned cached snapshot {SNAPSHOT} is admitted")
    files = {name: pin(snapshot / name) for name in MODEL_FILES}
    return {"model_id": MODEL_ID, "snapshot": SNAPSHOT, "files": files,
            "snapshot_bytes": sum(f["bytes"] for f in files.values()),
            "content_sha256": sha(json_bytes({n: f["sha256"] for n, f in files.items()}))}


def embeddings(args, snapshot: Path, chunks: list, queries: list,
               tokenizer, compatibility: dict, output: Path, report: dict):
    """Return unit document/query vectors. Never called at import time."""
    if not args.encode_cpu and args.embedding_cache is None:
        report["encoding"] = {"status": "UNRUN", "reason": "No compatible embeddings supplied; provide --embedding-cache or explicitly opt in with --encode-cpu",
                              "corpus_seconds": None, "query_seconds": None,
                              "corpus_encoder_calls": None, "query_encoder_calls": None}
        return None
    import numpy as np

    memory_check(report["ram_samples"], "before embedding load/optional CPU encoding")
    cache_start = time.perf_counter()
    if args.embedding_cache is not None:
        cache = args.embedding_cache.resolve()
        meta = checked_json(cache / "embedding_cache.json")
        if meta["compatibility"] != compatibility:
            raise ValueError("Embedding cache is not EXACT-compatible (code/model/software/chunks/queries)")
        arrays = []
        for name in ("document_embeddings.npy", "query_embeddings.npy"):
            if sha((cache / name).read_bytes()) != meta["array_sha256"][name]:
                raise ValueError(f"Embedding array hash mismatch: {name}")
            arrays.append(np.load(cache / name, allow_pickle=False))
        doc_vectors, query_vectors = arrays
        report["encoding"] = {"status": "REUSED_EXACT_COMPATIBLE_CACHE", "cache": str(cache),
                              "cache_metadata": pin(cache / "embedding_cache.json"),
                              "load_seconds": time.perf_counter() - cache_start,
                              "corpus_seconds": None, "query_seconds": None,
                              "corpus_encoder_calls_this_run": 0, "query_encoder_calls_this_run": 0,
                              "historical_encoding": meta["encoding_receipt"]}
    else:
        # Offline environment and CPU visibility were established before imports.
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        memory_check(report["ram_samples"], "before CPU model allocation")
        start = time.perf_counter()
        model = SentenceTransformer(str(snapshot), device="cpu", local_files_only=True,
                                    trust_remote_code=False)
        model.eval()
        if str(model.device) != "cpu" or model.get_max_seq_length() != 256:
            raise ValueError("Expected CPU MiniLM max_seq_length=256")
        if model.tokenizer.get_vocab() != tokenizer.get_vocab():
            raise ValueError("Tokenizer vocabulary differs from chunking tokenizer")
        parameter_count = sum(p.numel() for p in model.parameters())
        if parameter_count >= 4_000_000_000:
            raise ValueError("Reader/encoder parameter gate failed")
        load_seconds = time.perf_counter() - start

        def encode(texts, phase):
            result, batch_calls = [], 0
            start = time.perf_counter()
            for at in range(0, len(texts), args.batch_size):
                memory_check(report["ram_samples"], f"{phase} batch {batch_calls}")
                batch = texts[at:at + args.batch_size]
                # Compare FULL untruncated token IDs to the actual model's
                # tokenization before encode can apply its internal truncation.
                actual = model.tokenizer(batch, truncation=False, padding=False)["input_ids"]
                if any(ids != tokenizer.encode(text).ids or len(ids) > model.max_seq_length
                       for text, ids in zip(batch, actual)):
                    raise ValueError("Actual encoder token IDs mismatch or would truncate")
                tokenized = model.tokenize(batch)
                for i, ids in enumerate(actual):
                    visible = tokenized["input_ids"][i][tokenized["attention_mask"][i].bool()].tolist()
                    if visible != ids:
                        raise ValueError("Actual model.tokenize silently changed/truncated tokens")
                with torch.inference_mode():
                    result.append(model.encode(batch, batch_size=len(batch), device="cpu",
                                               show_progress_bar=False, convert_to_numpy=True,
                                               normalize_embeddings=True, prompt=""))
                batch_calls += 1
            return np.concatenate(result).astype(np.float32), time.perf_counter() - start, batch_calls

        doc_vectors, corpus_seconds, corpus_calls = encode([c["text"] for c in chunks], "corpus")
        windows = [w for q in queries for w in q["encoder_windows"]]
        window_vectors, query_seconds, query_calls = encode(windows, "queries")
        query_vectors, at = [], 0
        for q in queries:
            size = len(q["encoder_windows"])
            vector = window_vectors[at:at + size].mean(axis=0)
            norm = np.linalg.norm(vector)
            if not np.isfinite(norm) or norm == 0:
                raise ValueError("Invalid pooled query vector")
            query_vectors.append(vector / norm)
            at += size
        query_vectors = np.asarray(query_vectors, dtype=np.float32)
        del model
        report["encoding"] = {"status": "MEASURED_CPU_OPT_IN", "model_load_seconds": load_seconds,
                              "parameter_count": parameter_count, "corpus_seconds": corpus_seconds,
                              "query_seconds": query_seconds, "corpus_encoder_calls": corpus_calls,
                              "query_encoder_calls": query_calls,
                              "corpus_tokens_including_special": sum(c["encoder_tokens"] for c in chunks),
                              "query_tokens_including_special": sum(len(tokenizer.encode(w).ids) for w in windows),
                              "query_windows": len(windows), "device": "cpu"}
    for array, expected in ((doc_vectors, len(chunks)), (query_vectors, len(queries))):
        if (array.shape != (expected, 384) or array.dtype != np.float32
                or not np.isfinite(array).all()
                or not np.allclose(np.linalg.norm(array, axis=1), 1, atol=2e-5, rtol=0)):
            raise ValueError("Embedding shape/dtype/finiteness/unit-norm gate failed")
    for name, array in (("document_embeddings.npy", doc_vectors), ("query_embeddings.npy", query_vectors)):
        with (output / name).open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
    save(output / "embedding_cache.json", {"schema": 1, "compatibility": compatibility,
         "encoding_receipt": report["encoding"],
         "array_sha256": {n: sha((output / n).read_bytes()) for n in
                          ("document_embeddings.npy", "query_embeddings.npy")}})
    report["encoding"]["vector_payload_bytes"] = int(doc_vectors.nbytes + query_vectors.nbytes)
    report["encoding"]["total_load_or_encode_seconds"] = time.perf_counter() - cache_start
    return doc_vectors, query_vectors


def interval_size(intervals: list[tuple[int, int]]) -> int:
    total, right = 0, -1
    for start, end in sorted(intervals):
        total += max(0, end - max(start, right))
        right = max(right, end)
    return total


def citation_coverage(citations: list, selected: list, documents: dict) -> list[dict]:
    """Evaluator ONLY, after retrieval is frozen. Exact byte/line overlap, no judge."""
    rows = []
    for citation in citations:
        key = (citation["source"], citation["document"])
        doc = documents[key]
        if citation["source_sha256"] != doc["sha256"]:
            raise ValueError("Admitted citation source hash differs from corpus")
        first, last = citation["line_start"], citation["line_end"]
        if not 1 <= first <= last <= len(doc["lines"]):
            raise ValueError("Admitted citation line range is invalid")
        local = [p for p in selected if (p["source"], p["document"]) == key]
        full_lines, overlaps, total_bytes, covered_bytes = 0, [], 0, 0
        for line in range(first - 1, last):
            begin = doc["line_byte_starts"][line]
            end = begin + len(doc["lines"][line].encode("utf-8"))
            found = interval_size([(max(begin, p["byte_start"]), min(end, p["byte_end"]))
                                   for p in local if p["byte_start"] < end and p["byte_end"] > begin])
            total_bytes += end - begin
            covered_bytes += found
            full_lines += int(found == end - begin)
            if found:
                overlaps.append(line + 1)
        rows.append({"evidence_id": citation["evidence_id"], "source": key[0], "document": key[1],
                     "line_start": first, "line_end": last, "document_hit": bool(local),
                     "any_exact_overlap": covered_bytes > 0, "fully_covered": covered_bytes == total_bytes,
                     "fully_covered_lines": full_lines, "cited_lines": last - first + 1,
                     "overlapping_lines": overlaps, "covered_bytes": covered_bytes, "cited_bytes": total_bytes,
                     "line_coverage": full_lines / (last - first + 1),
                     "byte_coverage": covered_bytes / total_bytes,
                     "semantic_entailment": "UNRUN"})
    return rows


def summary(rows: list) -> dict:
    citations = [c for q in rows for c in q["citation_coverage"]]
    return {"status": "MEASURED_MECHANICAL_ONLY", "questions": len(rows),
            "citations": len(citations),
            "document_hits": sum(c["document_hit"] for c in citations),
            "citations_with_any_exact_overlap": sum(c["any_exact_overlap"] for c in citations),
            "citations_fully_covered": sum(c["fully_covered"] for c in citations),
            "fully_covered_cited_line_occurrences": sum(c["fully_covered_lines"] for c in citations),
            "cited_line_occurrences": sum(c["cited_lines"] for c in citations),
            "covered_cited_byte_occurrences": sum(c["covered_bytes"] for c in citations),
            "cited_byte_occurrences": sum(c["cited_bytes"] for c in citations),
            "questions_all_citations_overlap": sum(all(c["any_exact_overlap"] for c in q["citation_coverage"]) for q in rows),
            "questions_all_citations_fully_covered": sum(all(c["fully_covered"] for c in q["citation_coverage"]) for q in rows),
            "semantic_entailment": "UNRUN", "answer_quality": "UNRUN",
            "lm_calls": None, "lm_input_tokens": None, "lm_output_tokens": None}


def run(args, output: Path, report: dict):
    start = time.perf_counter()
    os.environ.update(OFFLINE_ENV)
    memory_check(report["ram_samples"], "before setup")
    report["software"] = {name: importlib.metadata.version(name) for name in
                          ("numpy", "tokenizers", "psutil", "sentence-transformers", "transformers", "torch")}
    report["host"] = {"python": sys.version, "platform": platform.platform(),
                      "processor": platform.processor(), "logical_cpus": os.cpu_count()}
    report["code"] = pin(Path(__file__))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    report["git_head"] = head.stdout.strip() if head.returncode == 0 else None
    report["offline_env"] = OFFLINE_ENV
    manifest = checked_json(BASE / "manifest.json", MANIFEST_SHA)
    pack = checked_json(BASE / PACK_REL, PACK_SHA)
    admission = checked_json(BASE / "admission.json")
    admitted = [q for q in admission["admitted_questions"] if q["pack"] == PACK_REL]
    if (len(admitted) != 12 or {q["id"] for q in admitted} != {q["id"] for q in pack["questions"]}
            or any(q["pack_sha256"] != PACK_SHA for q in admitted)
            or admission["manifest_sha256"] != MANIFEST_SHA):
        raise ValueError("Exact 12-question smoke admission gate failed")
    report["inputs"] = {name: pin(BASE / name) for name in ("manifest.json", "admission.json", PACK_REL)}
    source_start = time.perf_counter()
    documents = load_sources(manifest, pack)
    report["source_read_seconds"] = time.perf_counter() - source_start
    report["corpus"] = {"documents": len(documents), "raw_bytes": sum(len(d["raw"]) for d in documents),
                        "pins": [{k: d[k] for k in ("source", "document", "path", "sha256")} for d in documents]}
    snapshot = args.model_snapshot.resolve()
    stage_start = time.perf_counter()
    report["model"] = model_pins(snapshot)
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    config = checked_json(snapshot / "sentence_bert_config.json")
    if config["max_seq_length"] != 256:
        raise ValueError("Pinned model max length changed")
    report["tokenizer_and_model_pin_seconds"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    chunks = make_chunks(documents, tokenizer, args)
    # These are the only per-question fields the retrievers receive.
    queries = [{"id": q["id"], "question": q["question"],
                "encoder_windows": split_text(q["question"], tokenizer, args.chunk_tokens,
                                               args.passage_bytes, args.chunk_lines)} for q in pack["questions"]]
    report["chunking_seconds"] = time.perf_counter() - stage_start
    report["chunking"] = {"chunks": len(chunks), "overlap": 0, "lossless_partition": True,
                          "max_tokens_including_special": max(c["encoder_tokens"] for c in chunks),
                          "max_bytes": max(c["bytes"] for c in chunks),
                          "query_windows": sum(len(q["encoder_windows"]) for q in queries),
                          "query_pooling": "mean of normalized window embeddings, then L2 normalize; full text; no truncation",
                          "encoder_input": "source text only; SOURCE_PATH headers preserved; no gold, answers, citations, tags or required-source metadata"}
    save(output / "chunks.json", chunks)
    save(output / "queries.json", queries)
    stage_start = time.perf_counter()
    index = bm25_index(chunks)
    report["bm25_index_seconds"] = time.perf_counter() - stage_start
    report["bm25_serialized_index_bytes"] = save(output / "bm25_index.json", index)
    compatibility = {"schema": "exact-embedding-cache-v1", "code_sha256": report["code"]["sha256"],
                     "model_content_sha256": report["model"]["content_sha256"],
                     "software": report["software"], "python_version": platform.python_version(),
                     "chunks_sha256": sha(json_bytes(chunks)), "queries_sha256": sha(json_bytes(queries)),
                     "batch_size": args.batch_size, "threads": 1, "dtype": "float32",
                     "query_pooling": report["chunking"]["query_pooling"]}
    vectors = embeddings(args, snapshot, chunks, queries, tokenizer, compatibility, output, report)
    report["setup_seconds_before_retrieval"] = time.perf_counter() - start
    memory_check(report["ram_samples"], "before retrieval")
    methods = {"bm25": []}
    if vectors is not None:
        methods.update({"dense_cosine": [], "hybrid_rrf": []})
        import numpy as np

        dense_setup_start = time.perf_counter()
        # FP64 dot product of saved unit FP32 vectors; exact exhaustive scan.
        doc_vectors, query_vectors = (v.astype(np.float64) for v in vectors)
        doc_vectors /= np.linalg.norm(doc_vectors, axis=1, keepdims=True)
        query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)
        report["dense_index_seconds"] = time.perf_counter() - dense_setup_start
        report["dense_scan_working_vector_bytes"] = int(doc_vectors.nbytes + query_vectors.nbytes)
    docs_by_key = {(d["source"], d["document"]): d for d in documents}
    for qi, query in enumerate(queries):
        memory_check(report["ram_samples"], f"query {qi + 1}")
        bm_start = time.perf_counter()
        lexical = bm25_scores(index, query["question"])
        lexical_order = rank(lexical)
        scores_and_orders = {"bm25": (lexical, lexical_order, time.perf_counter() - bm_start)}
        if vectors is not None:
            dense_start = time.perf_counter()
            dense = (doc_vectors @ query_vectors[qi]).tolist()
            dense_order = rank(dense)
            dense_seconds = time.perf_counter() - dense_start
            fusion_start = time.perf_counter()
            fusion = [0.0] * len(chunks)
            for order in (lexical_order, dense_order):
                for position, i in enumerate(order, 1):
                    fusion[i] += 1 / (args.rrf_k + position)
            fusion_order = rank(fusion)
            fusion_seconds = time.perf_counter() - fusion_start
            scores_and_orders.update({"dense_cosine": (dense, dense_order, dense_seconds),
                                      "hybrid_rrf": (fusion, fusion_order, scores_and_orders["bm25"][2] + dense_seconds + fusion_seconds)})
        # Freeze all methods' selections before the evaluator sees citations.
        selections = {name: select_passages(order, scores, chunks, args)
                      for name, (scores, order, _) in scores_and_orders.items()}
        gold = pack["questions"][qi]
        for name, selected in selections.items():
            evaluation_start = time.perf_counter()
            coverage = citation_coverage(gold["citations"], selected, docs_by_key)
            methods[name].append({"id": query["id"], "question": query["question"],
                                  "retrieval_seconds_including_ranking": scores_and_orders[name][2],
                                  "retrieval_seconds_excludes": "encoding, whole-passage budget selection, evaluation, serialization",
                                  "evaluator_seconds": time.perf_counter() - evaluation_start,
                                  "returned_passages": len(selected), "returned_bytes": sum(p["bytes"] for p in selected),
                                  "passages": selected, "citation_coverage": coverage,
                                  "lm_calls": None, "lm_input_tokens": None, "lm_output_tokens": None})
    report["methods"] = {}
    failures = []
    for name in ("bm25", "dense_cosine", "hybrid_rrf"):
        if name not in methods:
            report["methods"][name] = {"status": "UNRUN", "reason": report["encoding"]["reason"],
                                      "semantic_entailment": "UNRUN", "answer_quality": "UNRUN"}
            continue
        rows = methods[name]
        save(output / f"{name}.json", rows)
        report["methods"][name] = summary(rows)
        for question, row in zip(pack["questions"], rows):
            missing = [c for c in row["citation_coverage"] if not c["fully_covered"]]
            if missing:
                failures.append({"method": name, "id": row["id"], "missing_support": missing,
                                 "admitted_quotes_evaluator_only": [c for c in question["citations"]
                                    if c["evidence_id"] in {m["evidence_id"] for m in missing}],
                                 "exact_retrieval_output_excerpts": [{"id": p["id"], "line_start": p["line_start"],
                                      "line_end": p["line_end"], "text": p["text"]} for p in row["passages"]],
                                 "interpretation": "Missing admitted citation span within budget, NOT a semantic or product failure verdict"})
    save(output / "missing_support.json", failures)
    memory_check(report["ram_samples"], "after evaluation")
    report["min_sampled_free_ram_gib"] = min(r["free_ram_gib"] for r in report["ram_samples"])
    report["total_seconds_before_report_serialization"] = time.perf_counter() - start
    report["artifacts_before_report"] = {p.name: pin(p) for p in sorted(output.iterdir()) if p.is_file()}
    report["persisted_artifact_bytes_before_report"] = sum(p["bytes"] for p in report["artifacts_before_report"].values())
    report["status"] = "COMPLETE" if vectors is not None else "PARTIAL_BM25_MEASURED_DENSE_HYBRID_UNRUN"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument("--output", type=Path, required=True,
                        help="New unique directory strictly under workspace/eval_corpora/baseline/incumbent_smoke_20261017; refuses existing paths")
    result.add_argument("--model-snapshot", type=Path,
                        default=Path.home() / f".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/{SNAPSHOT}",
                        help="Existing pinned local cache; never downloads")
    result.add_argument("--top-k", type=int, default=8)
    result.add_argument("--passage-bytes", type=int, default=1536, help="Per-passage UTF-8 source-text byte cap")
    result.add_argument("--byte-budget", type=int, default=12288, help="Total returned UTF-8 source-text bytes per method/query; pointer metadata accounted separately")
    result.add_argument("--chunk-tokens", type=int, default=224, help="Max encoder tokens INCLUDING special tokens; must be <=256")
    result.add_argument("--chunk-lines", type=int, default=12)
    result.add_argument("--rrf-k", type=int, default=60)
    result.add_argument("--batch-size", type=int, default=16)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--embedding-cache", type=Path, help="Prior run with EXACT-compatible embedding_cache.json and two .npy arrays")
    mode.add_argument("--encode-cpu", action="store_true",
                      help="Explicit opt-in; cached model, CPU only, offline, RAM >20GiB before each batch")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if (not 4 <= args.chunk_tokens <= 256 or not 1 <= args.top_k <= 64
            or min(args.passage_bytes, args.byte_budget, args.chunk_lines, args.rrf_k, args.batch_size) < 1
            or args.batch_size > 64):
        parser().error("Invalid budget; require tokens 4..256, top-k 1..64, batch 1..64 and positive other limits")
    output = args.output.resolve()
    # Fixed non-symlink authority root prevents a caller from widening writes.
    expected_root = WRITE_ROOT.absolute()
    if WRITE_ROOT.resolve() != expected_root or not output.is_relative_to(expected_root) or output == expected_root:
        parser().error(f"Output must resolve strictly below {expected_root}, without a redirected authority root")
    if output.exists():
        parser().error(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "incumbent-smoke-retrieval-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "command_argv": [sys.executable, *sys.argv],
              "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "status": "STARTED", "ram_samples": [],
              "program_claim": "C5/C7/C4 supporting retrieval SLICE; no optional-memory reasoner constructed",
              "failure_criterion": "One or more admitted citation ranges not fully returned under matched passage/byte budgets",
              "confound": "Long citation ranges, lexical query wording, shared anchors and finite development-pack dependence",
              "null_and_control": "BM25 is the cheap lexical incumbent; exact lossless source/line/token pointer checks are mechanical controls",
              "boundary": "Citation overlap is not entailment, exhaustive evidence, absence, answer parity or a product weakness map",
              "reader_quality": "UNRUN", "semantic_entailment": "UNRUN", "product_level_weakness_testing": "UNRUN",
              "lm_calls": None, "lm_input_tokens": None, "lm_output_tokens": None,
              "effort": {"measurement_minutes": None, "object_minutes": None, "ratio": None,
                         "basis": "Measurement and construction effort were not measured"},
              "cost_scope": "Local setup/index/query timings and serialized bytes; no economic cost or model-call saving claimed",
              "budget_scope": "Same top-k, whole passage bytes and total source-text bytes for all arms; JSON pointer/wire bytes reported in artifacts",
              "timing_scope": "Single serial run, no warmup/repetition; hybrid includes both retrieval scans plus fusion; encode/setup separate",
              "coverage_denominator": "49 admitted citation occurrences; line/byte occurrences include repeated/overlapping gold ranges; not independent facts",
              "ram_scope": "Stage/batch samples, not a continuous OS memory ceiling"}
    code = 0
    try:
        run(args, output, report)
    except Exception as error:
        report.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        code = 1
    save(output / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(output / "report.json"),
                      "methods": report.get("methods"), "error": report.get("error")}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
