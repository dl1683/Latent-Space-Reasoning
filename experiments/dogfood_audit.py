"""Evaluator-only DOGFOOD leakage/custody audit; no model loading or file discovery.

SLICE: protects evaluation isolation, not a latent-memory capability claim. Only
the audit process may receive references (gold answers AND evaluator criteria).
Never import this module with real references in a reader/builder process. Receipts
contain span hashes/word offsets, not reference text; keep receipts evaluator-side.

Matching uses NFKC/casefolded Unicode words. Exact source-contained spans are
legitimate; a partial source quotation NEVER exempts a whole reference. Uncovered
gold runs are checked separately, including runs shorter than ngram_size. Fuzzy
and semantic checks use sentences/overlapping <=64-word windows, not exhaustive
gold-length/subsentence fuzzy coverage or a completeness proof.
An embedding callable must be supplied explicitly; this module downloads nothing.
Conservative literal source attribution can flag source paraphrases for review.

Limits fail admission closed instead of silently sampling/truncating. Memory is
bounded by input/unit limits, a capped embedding cache and one similarity batch,
never an all-pairs similarity matrix. Exact allowed-source witnesses bypass
redundant heavy screening, never partially exempt a reference.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import re
import secrets
import stat
import unicodedata
from difflib import SequenceMatcher
from bisect import bisect_left
from typing import Callable


MAX_INPUT_CHARS = 2_000_000
# Counts supplied source AND payload copies; a ~100k-word source supplied twice
# must not fail before its exact provenance witness can discharge redundant work.
MAX_WORDS = 400_000
MAX_RECORDS = 10_000
MAX_UNITS = 20_000
MAX_COMPARISONS = 250_000
MAX_FINDINGS = 2_000
MAX_SUPPORTED_EXAMPLES = 32
MAX_WINDOW_WORDS = 64
EMBEDDING_BATCH_SIZE = 32
MAX_EMBEDDING_DIMENSIONS = 8_192
MAX_EMBEDDING_CACHE_BYTES = 64 * 1024 * 1024
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_CANARY = re.compile(r"dogfoodcanary[0-9a-f]{32}", re.IGNORECASE)


class _Limit(Exception):
    pass


def _words(text: str) -> list[str]:
    return _WORDS.findall(unicodedata.normalize("NFKC", text).casefold())


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_canary_reference(
    fake_text: str, *, reference_id: str = "fake-canary", nonce: str | None = None
) -> dict:
    """Return a NEW fake reference; call only on synthetic copies, never originals.

    The input is not modified and no file is read/written. Nonce injection is for
    deterministic software tests; ordinary controls should use the random default.
    Any matching canary in a payload rejects admission, even if it is in a source.
    """
    nonce = secrets.token_hex(16) if nonce is None else nonce
    if not isinstance(fake_text, str) or not isinstance(reference_id, str):
        raise ValueError("fake_text and reference_id must be strings")
    if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-fA-F]{32}", nonce) is None:
        raise ValueError("nonce must be exactly 32 hexadecimal characters")
    return {"id": reference_id, "text": fake_text + "\nDOGFOODCANARY" + nonce.lower()}


def _records(records: list[dict], key: str) -> list[dict]:
    if not isinstance(records, list):
        raise ValueError("records must be lists")
    if len(records) > MAX_RECORDS:
        raise _Limit("record count limit")
    seen = set()
    result = []
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("record must be a dictionary")
        name, text = item.get(key), item.get("text")
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError("record identifiers must be nonempty unique strings")
        if not isinstance(text, str):
            raise ValueError("record text must be a string")
        seen.add(name)
        result.append({key: name, "text": text})
    return result


def _source_support(text: str, sources: list[dict]) -> list[str]:
    needle = " " + text + " "
    return [s["label"] for s in sources if needle in s["padded"]]


def _screening_units(record: dict, source_words: set[str] | None = None):
    """Full text (exact only if >64 words), sentences, stride-32 windows, novel runs.

    Source-word membership ONLY proposes extra reference runs. It never waives
    a full/sentence/window or n-gram comparison, including source-word mosaics.
    """
    words = record["words"]
    spans = set()

    def add(start, end):
        if start < end:
            spans.add((start, end))
            if len(spans) > MAX_UNITS:
                raise _Limit("record unit limit")

    def windows(start, end):
        for offset in range(start, end, MAX_WINDOW_WORDS // 2):
            add(offset, min(offset + MAX_WINDOW_WORDS, end))
            if offset + MAX_WINDOW_WORDS >= end:
                break

    add(0, len(words))
    position = 0
    for sentence in re.split(r"[.!?\n]+", record["text"]):
        count = len(_words(sentence))
        if count <= MAX_WINDOW_WORDS:
            add(position, position + count)
        position += count
    windows(0, len(words))
    if source_words is not None:
        start = 0
        while start < len(words):
            if words[start] in source_words:
                start += 1
                continue
            end = start + 1
            while end < len(words) and words[end] not in source_words:
                end += 1
            windows(start, end)
            start = end
    for start, end in sorted(spans):
        yield {"reference_id": record.get("id", ""), "start": start, "end": end,
               "text": " ".join(words[start:end]), "length": end - start}


def audit_payloads(
    payloads: list[dict],
    references: list[dict],
    source_payloads: list[dict],
    embedder: Callable | None = None,
    ngram_size: int = 12,
    fuzzy_threshold: float = 0.92,
    embedding_threshold: float = 0.93,
) -> dict:
    """Audit already-captured payloads; no files, subprocesses or models are opened.

    Records are {label,text}; references are {id,text}. Source texts must be the
    trusted, independently pinned permitted corpus, NOT builder-supplied claims.
    Caller must include every reader/builder prompt, configuration and side input.
    Exact reference phrases and ALL normalized n-grams are checked. Fuzzy/semantic
    coverage is only full <=64-word texts, <=64-word sentences, stride-32/width-64
    windows, and reference runs of source-absent words. NOT exhaustive word-start
    fuzzy coverage. Entire source-contained payloads/windows get exact provenance
    witnesses, not expensive redundant comparisons; canaries are always checked.
    A PASS is bounded heuristic screening, not proof of noninterference/isolation.

    embedder(list[str]) -> finite nonzero numpy-compatible (batch, dimension).
    No embedder means embedding=UNRUN and admission=NOT_PASS. Malformed inputs,
    resource exhaustion, canary hits and encoder errors fail closed in the receipt.
    """
    checks = {name: "UNRUN" for name in ("exact", "ngram", "fuzzy", "embedding", "canary")}
    receipt = {
        "schema": "dogfood-payload-audit-v1",
        "admission": "NOT_PASS",
        "checks": checks,
        "leakage": [],
        "source_supported": [],
        "source_supported_summary": {
            "count": 0, "by_method": {}, "examples_truncated": False,
            "count_basis": "overlap match events plus explicitly labeled source witnesses",
        },
        "errors": [],
        "parameters": {},
        "comparisons": 0,
        "work": {"ngram_windows": 0, "fuzzy_pairs": 0, "embedding_pairs": 0,
                 "embedding_unique_texts": 0, "embedding_batches": 0,
                 "embedding_cache_bytes": 0, "source_pure_payloads": 0,
                 "source_pure_windows": 0},
    }
    seen_findings = set()
    source_cache = {}

    def tick():
        receipt["comparisons"] += 1
        if receipt["comparisons"] > MAX_COMPARISONS:
            raise _Limit("comparison limit")

    def support(text):
        if text not in source_cache:
            if len(source_cache) >= MAX_UNITS:
                raise _Limit("source attribution cache limit")
            source_cache[text] = _source_support(text, sources) if text else []
        return source_cache[text]

    def finding(method, payload, unit, start, end, text, score=1.0, count=1):
        tick()
        supported_by = [] if method == "canary" else support(text)
        if supported_by:
            summary = receipt["source_supported_summary"]
            summary["count"] += count
            summary["by_method"][method] = summary["by_method"].get(method, 0) + count
            if len(receipt["source_supported"]) >= MAX_SUPPORTED_EXAMPLES:
                summary["examples_truncated"] = True
                return
        else:
            key = (method, payload["label"], unit["reference_id"], unit["start"],
                   unit["end"], start, end)
            if key in seen_findings:
                return
            if len(seen_findings) >= MAX_FINDINGS:
                raise _Limit("leakage finding limit; audit incomplete")
            seen_findings.add(key)
        row = {
            "method": method,
            "payload_label": payload["label"],
            "reference_id": unit["reference_id"],
            "payload_word_span": [start, end],
            "reference_word_span": [unit["start"], unit["end"]],
            "span_sha256": _digest(text),
            "score": float(score),
        }
        if supported_by:
            row["source_labels"] = supported_by
            if count > 1:
                row["summarized_matches"] = count
            receipt["source_supported"].append(row)
        else:
            checks[method] = "FAIL"
            receipt["leakage"].append(row)

    def matches(method, payload_rows, reference_rows, score):
        for payload, window in payload_rows:
            for unit in reference_rows:
                finding(method, payload, unit, window["start"], window["end"],
                        window["text"], score)

    def exact_matches(method, payload, reference_rows, start, end, text):
        if not reference_rows:
            return
        if support(text):
            # Identical allowed-source matches need counts plus an example, not
            # a loop over every duplicate reference span. Never aggregate leakage.
            finding(method, payload, reference_rows[0], start, end, text,
                    count=len(reference_rows))
        else:
            for unit in reference_rows:
                finding(method, payload, unit, start, end, text)

    try:
        if type(ngram_size) is not int or not 1 <= ngram_size <= MAX_WINDOW_WORDS:
            raise ValueError("ngram_size must be an integer in [1,64]")
        for threshold in (fuzzy_threshold, embedding_threshold):
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                raise ValueError("thresholds must be finite numbers")
            if not math.isfinite(threshold) or not 0 < threshold <= 1:
                raise ValueError("thresholds must be in (0,1]")
        if embedder is not None and not callable(embedder):
            raise ValueError("embedder must be callable")
        receipt["parameters"] = {
            "ngram_size": ngram_size,
            "fuzzy_threshold": float(fuzzy_threshold),
            "embedding_threshold": float(embedding_threshold),
            "max_window_words": MAX_WINDOW_WORDS,
            "window_stride_words": MAX_WINDOW_WORDS // 2,
            "fuzzy_embedding_scope": "short_full_text,sentences,overlapping_windows,gold_novel_runs",
            "exhaustive_subsentence_fuzzy": False,
            "max_comparisons": MAX_COMPARISONS,
            "max_input_chars": MAX_INPUT_CHARS,
            "max_input_words": MAX_WORDS,
            "max_supported_examples": MAX_SUPPORTED_EXAMPLES,
            "max_embedding_cache_bytes": MAX_EMBEDDING_CACHE_BYTES,
        }
        payloads = _records(payloads, "label")
        references = _records(references, "id")
        sources = _records(source_payloads, "label")
        if not payloads or not references:
            raise ValueError("payloads and references must be nonempty")
        all_records = payloads + references + sources
        if sum(len(row["text"]) + len(row.get("id", row.get("label", "")))
               for row in all_records) > MAX_INPUT_CHARS:
            raise _Limit("input character limit")
        for row in all_records:
            row["words"] = _words(row["text"])
            row["normalized"] = " ".join(row["words"])
            row["padded"] = " " + row["normalized"] + " "
        if sum(len(row["words"]) for row in all_records) > MAX_WORDS:
            raise _Limit("input word limit")
        if any(not r["words"] for r in references):
            raise ValueError("empty normalized reference")
        source_words = {word for source in sources for word in source["words"]}
        units = []
        ngrams = {}
        ngram_count = 0
        for ref in references:
            for unit in _screening_units(ref, source_words):
                units.append(unit)
                if len(units) > MAX_UNITS:
                    raise _Limit("total reference unit limit")
            for start in range(max(0, len(ref["words"]) - ngram_size + 1)):
                ngram_count += 1
                if ngram_count > MAX_UNITS:
                    raise _Limit("reference ngram limit")
                words = tuple(ref["words"][start : start + ngram_size])
                unit = {"reference_id": ref["id"], "start": start,
                        "end": start + ngram_size, "text": " ".join(words)}
                ngrams.setdefault(words, []).append(unit)
        receipt["counts"] = {
            "payloads": len(payloads), "references": len(references),
            "sources": len(sources), "reference_units": len(units),
            "reference_ngrams": ngram_count,
        }
        reference_texts = {}
        for unit in units:
            reference_texts.setdefault(unit["text"], []).append(unit)
        payload_texts = {}
        window_count = 0
        for name in ("exact", "ngram", "fuzzy", "canary"):
            checks[name] = "PASS"
        for payload in payloads:
            for index, word in enumerate(payload["words"]):
                if _CANARY.search(word):
                    unit = {"reference_id": "canary-pattern", "start": 0, "end": 0}
                    finding("canary", payload, unit, index, index + 1, word)
            if support(payload["normalized"]):
                receipt["work"]["source_pure_payloads"] += 1
                unit = {"reference_id": "allowed-source-witness", "start": 0, "end": 0}
                finding("source_witness", payload, unit, 0, len(payload["words"]),
                        payload["normalized"])
                continue
            # Character positions of preceding word-boundary spaces give O(log W)
            # offsets for exact phrase matches, without repeatedly slicing prefixes.
            offsets, position = [], 0
            for word in payload["words"]:
                offsets.append(position)
                position += len(word) + 1
            for text, reference_rows in reference_texts.items():
                tick()
                needle, haystack = " " + text + " ", payload["padded"]
                offset = 0
                while True:
                    offset = haystack.find(needle, offset)
                    if offset < 0:
                        break
                    tick()
                    start = bisect_left(offsets, offset)
                    exact_matches("exact", payload, reference_rows, start,
                                  start + reference_rows[0]["length"], text)
                    offset += 1
            # One indexed pass over payload n-grams, NOT a reference x payload sweep.
            if ngrams:
                for start in range(max(0, len(payload["words"]) - ngram_size + 1)):
                    tick()
                    receipt["work"]["ngram_windows"] += 1
                    key = tuple(payload["words"][start : start + ngram_size])
                    rows = ngrams.get(key)
                    if rows:
                        exact_matches("exact", payload, rows, start, start + ngram_size,
                                      rows[0]["text"])
                        exact_matches("ngram", payload, rows, start, start + ngram_size,
                                      rows[0]["text"])
            for window in _screening_units(payload):
                if window["length"] > MAX_WINDOW_WORDS:
                    continue
                window_count += 1
                if window_count + len(units) > MAX_UNITS:
                    raise _Limit("total screening unit limit")
                if support(window["text"]):
                    receipt["work"]["source_pure_windows"] += 1
                    witness = {"reference_id": "allowed-source-witness", "start": 0, "end": 0}
                    finding("source_witness", payload, witness, window["start"], window["end"],
                            window["text"])
                else:
                    payload_texts.setdefault(window["text"], []).append((payload, window))
        heavy_references = {text: rows for text, rows in reference_texts.items()
                            if rows[0]["length"] <= MAX_WINDOW_WORDS}
        receipt["counts"]["payload_screening_units"] = window_count
        receipt["counts"]["unique_payload_windows"] = len(payload_texts)
        receipt["counts"]["unique_reference_windows"] = len(heavy_references)
        for reference_text, reference_rows in heavy_references.items():
            for text, payload_rows in payload_texts.items():
                tick()
                receipt["work"]["fuzzy_pairs"] += 1
                if text == reference_text:
                    continue  # exact already covered
                if 2 * min(len(text), len(reference_text)) < (
                    fuzzy_threshold * (len(text) + len(reference_text))
                ):
                    continue
                score = SequenceMatcher(None, reference_text, text, autojunk=False).ratio()
                if score >= fuzzy_threshold:
                    matches("fuzzy", payload_rows, reference_rows, score)

        if embedder is not None:
            import numpy as np  # optional: no imports from model/reader code

            dimensions = None
            vectors = {}

            def encode(texts):
                nonlocal dimensions
                array = np.asarray(embedder(texts), dtype=np.float64)
                if array.ndim != 2 or array.shape[0] != len(texts):
                    raise ValueError("encoder output must have shape (batch,dimension)")
                if not 1 <= array.shape[1] <= MAX_EMBEDDING_DIMENSIONS:
                    raise ValueError("invalid encoder dimension")
                if dimensions is not None and array.shape[1] != dimensions:
                    raise ValueError("encoder dimension changed")
                dimensions = array.shape[1]
                if not np.isfinite(array).all():
                    raise ValueError("encoder output must be finite")
                # Scale before the norm to avoid overflow on otherwise finite inputs.
                scale = np.max(np.abs(array), axis=1, keepdims=True)
                if np.any(scale == 0):
                    raise ValueError("encoder output contains a zero vector")
                array = array / scale
                return array / np.linalg.norm(array, axis=1, keepdims=True)

            checks["embedding"] = "PASS"
            # Global de-duplication: each distinct reference/payload text is encoded
            # ONCE. Source-pure payloads/windows need no embedding at all.
            texts = list(dict.fromkeys([*heavy_references, *payload_texts])) if payload_texts else []
            for offset in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                batch = texts[offset : offset + EMBEDDING_BATCH_SIZE]
                array = encode(batch)
                used = receipt["work"]["embedding_cache_bytes"] + array.nbytes
                if used > MAX_EMBEDDING_CACHE_BYTES:
                    raise _Limit("embedding cache byte limit")
                receipt["work"]["embedding_cache_bytes"] = used
                receipt["work"]["embedding_batches"] += 1
                for text, vector in zip(batch, array):
                    vectors[text] = vector
                receipt["work"]["embedding_unique_texts"] += len(batch)
            payload_keys = list(payload_texts)
            for reference_text, reference_rows in heavy_references.items():
                for offset in range(0, len(payload_keys), EMBEDDING_BATCH_SIZE):
                    batch = payload_keys[offset : offset + EMBEDDING_BATCH_SIZE]
                    for _ in batch:
                        tick()
                        receipt["work"]["embedding_pairs"] += 1
                    scores = np.stack([vectors[text] for text in batch]) @ vectors[reference_text]
                    for text, score in zip(batch, scores):
                        if float(score) >= embedding_threshold:
                            matches("embedding", payload_texts[text], reference_rows,
                                    min(1.0, max(-1.0, float(score))))
        if all(status == "PASS" for status in checks.values()):
            receipt["admission"] = "PASS"
    except _Limit as exc:
        receipt["errors"].append({"kind": "resource_limit", "detail": str(exc)})
        for name, status in checks.items():
            if status == "PASS":
                checks[name] = "INCOMPLETE"
    except Exception as exc:
        # Do not echo arbitrary encoder exception messages: they may contain gold.
        receipt["errors"].append({"kind": "audit_error", "exception_type": type(exc).__name__})
        if embedder is not None and checks["embedding"] == "PASS":
            checks["embedding"] = "FAIL"
    if receipt["errors"] or receipt["leakage"]:
        receipt["admission"] = "FAIL"
    return receipt


def _absolute_path(value: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("paths must be nonempty absolute strings")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("absolute paths without parent traversal required")
    # Reject Windows alternate data streams; drive colon is not in parts[1:].
    if os.name == "nt" and any(":" in part for part in path.parts[1:]):
        raise ValueError("alternate data streams are forbidden")
    return path


def _key(path: Path) -> str:
    return os.path.normcase(str(path))


def _checked_file(path: Path):
    """lstat named components only; never enumerate a directory or follow links."""
    signatures = []
    for component in reversed((path, *path.parents)):
        info = component.lstat()
        is_reparse = (
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
        tag = getattr(info, "st_reparse_tag", 0)
        cloud_placeholder = (tag & 0xFFFF0FFF) == 0x9000001A
        if stat.S_ISLNK(info.st_mode) or (is_reparse and not cloud_placeholder):
            raise ValueError("symlink/junction/reparse components forbidden")
        signatures.append((str(component), info.st_dev, info.st_ino))
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("allowlisted path must be a regular file")
    resolved = path.resolve(strict=True)
    if _key(resolved) != _key(path):
        raise ValueError("path must be its canonical resolved spelling")
    return resolved, info, signatures


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def audit_manifest(manifest: list[dict], allowed_paths: list[str]) -> dict:
    """Require the EXACT nonempty file allowlist, unique paths/identities and hashes.

    No outside-manifest path is opened; no directories are scanned. Unallowlisted
    entries are rejected lexically before filesystem resolution. Symlinks,
    junctions and unknown reparse tags fail closed; the known non-name-surrogate
    Windows cloud-placeholder tag family is allowed. SHA256 reads use 1MiB blocks
    and verify descriptor/path identities before and after hashing.

    This is a custody check, NOT an OS sandbox: concurrent hostile parent-directory
    swaps cannot be excluded portably. Run against quiescent, access-controlled
    files, then consume the same pinned bytes; receipt does not authorize later I/O.
    """
    receipt = {"schema": "dogfood-manifest-audit-v1", "admission": "FAIL",
               "files": [], "errors": []}
    try:
        if not isinstance(manifest, list) or not isinstance(allowed_paths, list):
            raise ValueError("manifest and allowed_paths must be lists")
        if not manifest or not allowed_paths:
            raise ValueError("manifest and allowlist must be nonempty")
        allowed = {}
        for raw in allowed_paths:
            path = _absolute_path(raw)
            key = _key(path)
            if key in allowed:
                raise ValueError("duplicate allowlisted path")
            allowed[key] = path
        entries = {}
        for entry in manifest:
            if not isinstance(entry, dict):
                raise ValueError("manifest entry must be a dictionary")
            path = _absolute_path(entry.get("path"))
            key = _key(path)
            if key not in allowed:
                raise ValueError("manifest path is outside the exact allowlist")
            if key in entries:
                raise ValueError("duplicate manifest path")
            digest = entry.get("sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
                raise ValueError("invalid SHA256")
            entries[key] = digest.lower()
        if set(entries) != set(allowed):
            raise ValueError("manifest does not cover the exact allowlist")
        identities = set()
        for key, path in allowed.items():
            row = {"path": str(path), "status": "FAIL"}
            receipt["files"].append(row)
            resolved, before, components = _checked_file(path)
            identity = (before.st_dev, before.st_ino)
            if identity in identities:
                raise ValueError("duplicate file identity")
            identities.add(identity)
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            flags |= getattr(os, "O_NONBLOCK", 0)
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as stream:
                opened = os.fstat(stream.fileno())
                # Compare identity across APIs, timestamps within the same API.
                # Windows path stat and descriptor stat can differ in precision.
                if ((opened.st_dev, opened.st_ino, opened.st_size)
                        != (before.st_dev, before.st_ino, before.st_size)):
                    raise ValueError("file changed before hashing")
                hasher = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
                if _identity(os.fstat(stream.fileno())) != _identity(opened):
                    raise ValueError("file changed during hashing")
            _, after, after_components = _checked_file(path)
            if _identity(before) != _identity(after) or components != after_components:
                raise ValueError("path/components changed during hashing")
            actual = hasher.hexdigest()
            row.update(path=str(resolved), sha256=actual, size_bytes=before.st_size,
                       status="PASS" if actual == entries[key] else "FAIL")
            if row["status"] != "PASS":
                raise ValueError("SHA256 mismatch")
        receipt["admission"] = "PASS"
    except Exception as exc:
        receipt["errors"].append({"kind": "manifest_error", "exception_type": type(exc).__name__,
                                  "detail": str(exc)})
    return receipt