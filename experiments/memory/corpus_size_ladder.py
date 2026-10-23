"""Nested real-source scale ladder; retrieval is not answer quality.

Preparation and retrieval are CPU-only. Optional generic MiniLM encoding must
run through the existing external GPU resource_guard (this module owns no controller).
Raw corpora, pointers, vectors, questions and per-query receipts remain private.
No training, downloads, cloud operations or import-time model initialization.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from experiments.memory.compressed_routing_map import (
    BM25, normalize, pack_rows, ranked, unpack_rows,
)

SIZES = (10_000, 100_000, 1_000_000, 10_000_000)
ARMS = ("bm25", "dense", "hybrid", "packed8")
POLICIES = ("flat", "graph")
SENSITIVE = re.compile(
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{30,})"
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save(path, value):
    with Path(path).open("x", encoding="utf8") as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")


class Limits:
    """Cooperative memory/deadline checks, not a continuous OS guarantee."""

    def __init__(self, seconds=1800):
        import psutil
        self.psutil = psutil
        self.start = time.monotonic()
        self.seconds = seconds
        self.minimum_free = None
        self.maximum_rss = 0
        self.check()

    def check(self):
        free = self.psutil.virtual_memory().available
        rss = self.psutil.Process().memory_info().rss
        self.minimum_free = min(free, self.minimum_free or free)
        self.maximum_rss = max(rss, self.maximum_rss)
        if free < 20 * 2**30 or rss > 3 * 2**30:
            raise RuntimeError("20GiB free / 3GiB RSS limit")
        if time.monotonic() - self.start > self.seconds:
            raise TimeoutError("finite CPU phase deadline")

    def receipt(self):
        self.check()
        return dict(seconds=time.monotonic() - self.start,
                    sampled_min_free_bytes=self.minimum_free,
                    sampled_peak_rss_bytes=self.maximum_rss,
                    enforcement="cooperative sampled checks, not continuous")


def byte_chunks(text, tokenizer, width=160):
    """Nonoverlapping exact UTF8 spans; no silent truncation or text decoding."""
    if type(width) is not int or width < 1:
        raise ValueError("positive integer chunk width required")
    offsets = tokenizer(text, add_special_tokens=False,
                        return_offsets_mapping=True)["offset_mapping"]
    if not offsets:
        return [(0, len(text.encode("utf8")))] if text else []
    # Fast tokenizer offsets are character offsets, including non-ASCII input.
    boundaries = [0]
    for i in range(width, len(offsets), width):
        point = int(offsets[i][0])
        if point > boundaries[-1]:
            boundaries.append(point)
    boundaries.append(len(text))
    byte_offsets = [len(text[a:b].encode("utf8"))
                    for a, b in zip(boundaries, boundaries[1:])]
    cumulative = np.cumsum([0, *byte_offsets]).tolist()
    return list(zip(cumulative, cumulative[1:]))


def rung_counts(token_counts, sizes=SIZES):
    """First whole-chunk prefix reaching each requested token count."""
    counts = np.asarray(token_counts, dtype=np.int64)
    if counts.ndim != 1 or (counts <= 0).any():
        raise ValueError("positive one-dimensional token counts required")
    cumulative = np.cumsum(counts)
    out = []
    for size in sizes:
        if type(size) is not int or size < 1:
            raise ValueError("positive integer target required")
        index = int(np.searchsorted(cumulative, size))
        available = index < len(counts)
        n = index + 1 if available else len(counts)
        out.append(dict(target_tokens=size, chunks=n,
                        actual_tokens=int(cumulative[n - 1]) if n else 0,
                        status="AVAILABLE" if available else "INSUFFICIENT_SOURCE"))
    return out


def graph_route(order, document_ids, budget=10):
    """Equal-exposure document adjacency: seed then adjacent source spans.

    Not a semantic reference graph. Give this identical procedure to every arm.
    """
    if type(budget) is not int or budget < 1:
        raise ValueError("positive budget required")
    order = list(dict.fromkeys(map(int, order)))
    if any(i < 0 or i >= len(document_ids) for i in order):
        raise ValueError("unknown chunk")
    selected = order[: (budget + 1) // 2]
    edges = 0
    for i in tuple(selected):
        for j in (i - 1, i + 1):
            if 0 <= j < len(document_ids) and document_ids[j] == document_ids[i]:
                edges += 1
                if j not in selected and len(selected) < budget:
                    selected.append(j)
    for i in order:
        if len(selected) >= budget:
            break
        if i not in selected:
            selected.append(i)
    return selected, edges


def canonical_order(scores, pointers):
    """Break score ties by source hash/span, never support-enriched row order."""
    scores = np.asarray(scores)
    if scores.shape != (len(pointers),) or not np.isfinite(scores).all():
        raise ValueError("finite aligned score vector required")
    permutation = np.asarray(sorted(
        range(len(pointers)), key=lambda i: (
            pointers[i]["source_id"], pointers[i]["start"], pointers[i]["end"])))
    return permutation[ranked(scores[permutation])]


def hybrid_scores(dense, lexical, pointers):
    """Canonical RRF; a wholly empty lexical route contributes no fake ranks."""
    dense, lexical = np.asarray(dense), np.asarray(lexical)
    if dense.shape != lexical.shape:
        raise ValueError("aligned score vectors required")
    out = np.zeros(len(pointers), dtype=np.float32)
    if not np.isfinite(dense).all() or not np.isfinite(lexical).all():
        raise ValueError("finite scores required")
    if np.all(lexical == 0):
        return dense.copy()
    for values in (dense, lexical):
        out[canonical_order(values, pointers)] += 1 / (60 + np.arange(1, len(pointers) + 1))
    return out


def verify_span(root, pointer):
    """Hash-check the fetched exact span against the trusted ingestion pointer."""
    start, end = pointer["start"], pointer["end"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError("invalid byte span")
    with (Path(root) / "leaves.bin").open("rb") as source:
        source.seek(start)
        raw = source.read(end - start)
    if len(raw) != end - start or sha(raw) != pointer["span_sha256"]:
        raise ValueError("source span changed or missing")
    return raw.decode("utf8"), len(raw)


def prepare(root, config):
    """Source selection is a declared development benchmark construction.

    The same six fixed development questions survive every rung. Their complete source
    paragraphs are the fixed enriched core; distractors are real, hash-ordered
    evaluation documents. This is not natural prevalence or held-out evaluation.
    """
    from transformers import AutoTokenizer
    root = Path(root)
    guard = Limits()
    cfg = load(config)
    encoder_tok = AutoTokenizer.from_pretrained(cfg["encoder"], local_files_only=True)
    reader_tok = AutoTokenizer.from_pretrained(cfg["reader_tokenizer"],
                                               local_files_only=True)
    source_rows, queries = {}, []
    with Path(cfg["musique"]).open(encoding="utf8") as stream:
        for i, line in enumerate(stream):
            if i < 200:
                continue
            if i == 206:
                break
            row = json.loads(line)
            if not row["answerable"]:
                raise ValueError("answerable development source required")
            gold = []
            for p in row["paragraphs"]:
                if p["is_supporting"]:
                    text = p["title"] + "\n" + p["paragraph_text"]
                    if SENSITIVE.search(text):
                        raise ValueError("sensitive core source, stop before copying")
                    key = sha(text.encode("utf8"))
                    source_rows.setdefault(key, text)
                    gold.append(key)
            queries.append(dict(id=row["id"], question=row["question"],
                                gold=sorted(set(gold)), type="multi_hop"))
    if len(queries) != 6:
        raise ValueError("six fixed development questions required")
    # Labels are not passed to the encoder or navigation. Lookup-only diagnostic
    # questions are source titles, not end-answer tasks.
    for key, text in list(source_rows.items())[:6]:
        queries.append(dict(id="title_" + key[:12], question=text.partition("\n")[0],
                            gold=[key], type="title_lookup"))
    pointers, counts, sources = [], [], []
    cumulative, rejected, duplicate = 0, 0, 0
    encoding = time.perf_counter()
    with (root / "leaves.bin").open("xb") as leaves:
        def append(text, key, origin):
            nonlocal cumulative
            raw = text.encode("utf8")
            start = leaves.tell()
            leaves.write(raw)
            sources.append(dict(source_id=key, start=start, end=start + len(raw),
                                sha256=sha(raw), origin=origin))
            for a, b in byte_chunks(text, encoder_tok):
                piece = raw[a:b]
                n = len(reader_tok.encode(piece.decode("utf8"),
                                          add_special_tokens=False))
                if not n:
                    continue
                pointers.append(dict(source_id=key, start=start + a, end=start + b,
                                     span_sha256=sha(piece), tokens=n))
                counts.append(n)
                cumulative += n
                if cumulative >= max(SIZES):
                    return True
            return False
        for key, text in source_rows.items():
            append(text, key, dict(dataset="MuSiQue", content_sha256=key))
        core_tokens = cumulative
        if core_tokens > min(SIZES):
            raise ValueError("declared core does not fit smallest rung")
        seen = set(source_rows)
        # Only existing private copied text, never originals or live stores.
        paths = sorted(Path(cfg["eval_text"]).glob("*/*.txt"),
                       key=lambda p: (p.stem, p.parent.name))
        for path in paths:
            guard.check()
            raw = path.read_bytes()
            text = raw.decode("utf8")
            if SENSITIVE.search(text):
                rejected += 1
                continue
            key = sha(raw)
            if key in seen:
                duplicate += 1
                continue
            seen.add(key)
            if append(text, key, dict(dataset="eval_corpora",
                                     relative_file=str(path.relative_to(cfg["eval_text"])),
                                     input_sha256=key)):
                break
    rungs = rung_counts(counts)
    save(root / "pointers.json", pointers)
    save(root / "sources.json", sources)
    save(root / "questions.json", [{k: q[k] for k in ("id", "question", "type")}
                                    for q in queries])
    save(root / "evaluation_only.json", queries)
    save(root / "rungs.json", rungs)
    pins = {name: file_sha(root / name) for name in (
        "leaves.bin", "pointers.json", "sources.json", "questions.json",
        "evaluation_only.json", "rungs.json")}
    pins["musique_input"] = file_sha(cfg["musique"])
    pins["worker"] = file_sha(__file__)
    pins["config"] = file_sha(config)
    save(root / "prepare.json", dict(
        status="PREPARED", rungs=rungs, core_tokens=core_tokens,
        sources=len(sources), chunks=len(pointers), source_bytes=(root / "leaves.bin").stat().st_size,
        token_count="sum of unframed reader-tokenizer chunk lengths; not generation input",
        no_duplicate_padding=True, duplicate_sources_skipped=duplicate,
        sensitive_sources_skipped=rejected, core_selection="MuSiQue rows200..205 gold supports",
        source_distribution="support-enriched fixed core plus hash-ordered real document distractors",
        cpu_prepare_seconds=time.perf_counter() - encoding,
        reader_window=cfg.get("reader_window", 12288), pins=pins, resources=guard.receipt(),
        latent_reasoning="UNRUN; A reused-block and B continuous-feedback remain distinct"))
    return rungs


def encoding_worker(root, phase):
    """Generic frozen encoder under unchanged existing resource resource_guard."""
    root = Path(root)
    cfg = load(root / "config.json")
    if os.environ.get("LSR_GPU_GUARDED") != "1":
        raise RuntimeError("existing external resource_guard required")
    from experiments.memory.verified_two_source_run import checked_thermal
    import psutil
    import torch
    from transformers import AutoModel, AutoTokenizer
    execution = os.environ["LSR_EXECUTION_ID"]
    pins_sha = os.environ["LSR_PIN_DIGEST"]
    if file_sha(root / "pins.json") != pins_sha:
        raise ValueError("changed pins")
    for path, expected in load(root / "pins.json").items():
        if file_sha(path) != expected:
            raise ValueError("actual input pin changed")
    guard = Limits(1750)
    thermal_path = root / (phase + "_external_thermal.json")
    burst_started = time.monotonic()

    def thermal():
        return checked_thermal(thermal_path, phase, execution,
                               os.environ["LSR_GPU_UUID"],
                               rejection_path=root / (phase + "_thermal_rejections.jsonl"))

    def checkpoint():
        nonlocal burst_started
        guard.check()
        row = thermal()
        if row["temperature"] >= 66 or time.monotonic() - burst_started >= 20:
            torch.cuda.synchronize()
            since = None
            until = time.monotonic() + 90
            while time.monotonic() < until:
                guard.check()
                row = thermal()
                if row["temperature"] <= 60:
                    since = since or time.monotonic()
                    if time.monotonic() - since >= 30:
                        burst_started = time.monotonic()
                        return
                else:
                    since = None
                time.sleep(.025)
            raise TimeoutError("unchanged burst cooldown not established")

    thermal()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.8)
    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder"], local_files_only=True)
    model = AutoModel.from_pretrained(cfg["encoder"], local_files_only=True).eval()
    count = sum(p.numel() for p in model.parameters())
    if count >= 4_000_000_000:
        raise ValueError("model must be under4B")
    model.cuda()
    checkpoint()

    def encode(texts):
        batch = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        if batch["input_ids"].shape[1] > 256:
            raise ValueError("silent embedding truncation refused")
        checkpoint()
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"][:, :, None]
            value = torch.nn.functional.normalize((hidden * mask).sum(1) /
                                                   mask.sum(1).clamp_min(1), dim=-1)
        torch.cuda.synchronize()
        value = value.cpu().numpy()
        checkpoint()
        return value

    started = time.perf_counter()
    if phase == "smoke":
        texts = ["A cobalt observatory studies stellar motion.",
                 "An orchard cultivates apples and pears."]
        batch = encode(texts)
        singles = np.concatenate([encode([text]) for text in texts])
        error = float(np.max(np.abs(batch - singles)))
        if error > 1e-4 or np.allclose(batch[0], batch[1]):
            raise ValueError("encoder repeat/batch/effect control failed")
        codes, scales = pack_rows(batch, 8)
        if (normalize(unpack_rows(codes, scales, 384, 8)) @ batch.T).argmax(1).tolist() != [0, 1]:
            raise ValueError("packed encoder control failed")
        details = dict(passed=True, max_batch_error=error, rows=2)
    elif phase == "panel":
        verdict = load(root / "smoke_measurements.json")
        resource = load(root / "smoke_resources.json")
        if (not verdict["passed"] or verdict["pins_sha256"] != pins_sha
                or resource["status"] != "COMPLETED" or not resource["cleanup_verified"]):
            raise ValueError("separate matching operation smoke required")
        pointers = load(root / "pointers.json")
        questions = load(root / "questions.json")
        bank = np.lib.format.open_memmap(root / "bank.npy", mode="w+",
                                         dtype=np.float32, shape=(len(pointers), 384))
        for start in range(0, len(pointers), 8):
            texts = [verify_span(root, p)[0] for p in pointers[start:start + 8]]
            bank[start:start + len(texts)] = encode(texts)
            if start % 256 == 0:
                bank.flush()
                print(json.dumps(dict(encoded_chunks=start + len(texts))), flush=True)
        bank.flush()
        qbank = np.concatenate([encode([q["question"]]) for q in questions])
        np.save(root / "query_vectors.npy", qbank)
        details = dict(passed=True, encoded_corpus_chunks=len(pointers),
                       encoded_queries=len(questions), corpus_encoding_passes=1,
                       no_per_query_corpus_reencoding=True,
                       bank_sha256=file_sha(root / "bank.npy"),
                       query_vectors_sha256=file_sha(root / "query_vectors.npy"))
    else:
        raise ValueError("smoke or panel only")
    thermal()
    save(root / (phase + "_measurements.json"), dict(
        **details, phase=phase, execution_id=execution, pins_sha256=pins_sha,
        encoder_parameters=count, seconds=time.perf_counter() - started,
        torch_version=torch.__version__, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        resources=guard.receipt(), reader_calls=0, reader_tokens=0))


def support_score(selected, pointers, gold):
    expected = set(gold)
    if not expected:
        raise ValueError("support recall is undefined for absence")
    found = {pointers[i]["source_id"] for i in selected}
    if not expected <= {p["source_id"] for p in pointers}:
        raise ValueError("gold support source missing from this rung")
    return dict(recall=len(found & expected) / len(expected),
                all_sources_found=expected <= found)


def support_chunk_score(selected, pointers, gold):
    expected = set(gold)
    if not expected or not expected <= {p["source_id"] for p in pointers}:
        raise ValueError("nonempty present gold support sources required")
    expected_chunks = {i for i, p in enumerate(pointers) if p["source_id"] in expected}
    return dict(support_chunk_recall=len(set(selected) & expected_chunks) / len(expected_chunks),
                all_support_chunks_found=expected_chunks <= set(selected))


def evaluate(root, out):
    """Measure support access at actual sizes, NOT a decoded reader benchmark."""
    from threadpoolctl import threadpool_limits
    from experiments.memory.inverted_bm25 import InvertedBM25
    threadpool_limits(1)
    root, out = Path(root), Path(out)
    out.mkdir(exist_ok=False)
    guard = Limits()
    preparation = load(root / "prepare.json")
    for name in ("leaves.bin", "pointers.json", "sources.json", "questions.json",
                 "evaluation_only.json", "rungs.json"):
        if file_sha(root / name) != preparation["pins"][name]:
            raise ValueError("prepared input changed")
    pointers = load(root / "pointers.json")
    questions = load(root / "questions.json")
    golds = load(root / "evaluation_only.json")
    if [(q["id"], q["question"]) for q in questions] != [
            (q["id"], q["question"]) for q in golds]:
        raise ValueError("question identity mismatch")
    # An incomplete/aborted GPU phase never admits any dense result.
    dense_ready = False
    if (root / "panel_resources.json").exists():
        resource = load(root / "panel_resources.json")
        operation = load(root / "operation.json") if (root / "operation.json").exists() else {}
        dense_ready = (
            resource["status"] == "COMPLETED" and resource["cleanup_verified"]
            and operation.get("status") == "COMPLETED"
            and operation.get("safe_cleanup") is True
            and operation.get("own_lease_released") is True)
    if dense_ready:
        measurements = load(root / "panel_measurements.json")
        if (file_sha(root / "panel_measurements.json") !=
                resource["artifact_sha256"]["panel_measurements.json"]):
            raise ValueError("encoder measurements changed")
        if (measurements["execution_id"] != resource["execution_id"]
                or measurements["pins_sha256"] != resource["pins_sha256"]):
            raise ValueError("encoder receipt binding changed")
        for name, field in (("bank.npy", "bank_sha256"),
                            ("query_vectors.npy", "query_vectors_sha256")):
            if file_sha(root / name) != measurements[field]:
                raise ValueError("encoder vector artifact changed")
        bank = np.load(root / "bank.npy", mmap_mode="r", allow_pickle=False)
        qbank = np.load(root / "query_vectors.npy", allow_pickle=False)
        if bank.shape != (len(pointers), 384) or qbank.shape != (len(questions), 384):
            raise ValueError("dense alignment mismatch")
    rows, summaries = [], []
    text_read_seconds = time.perf_counter()
    texts = []
    for p in pointers:
        guard.check()
        texts.append(verify_span(root, p)[0])
    text_read_seconds = time.perf_counter() - text_read_seconds
    for rung in load(root / "rungs.json"):
        guard.check()
        if rung["status"] != "AVAILABLE":
            summaries.append(dict(**rung, arm_status="UNRUN_INSUFFICIENT_SOURCE"))
            continue
        n = rung["chunks"]
        current = pointers[:n]
        document_ids = [p["source_id"] for p in current]
        began = time.perf_counter()
        bm25 = InvertedBM25(texts[:n])
        bm25_setup = time.perf_counter() - began
        bm25_path = out / ("bm25_" + str(rung["target_tokens"]) + ".npz")
        bm25.save(bm25_path)
        compressed = None
        packed_file = None
        if dense_ready:
            began = time.perf_counter()
            codes, scale = pack_rows(bank[:n], 8)
            packed_file = out / ("packed8_" + str(rung["target_tokens"]) + ".npz")
            np.savez(packed_file, codes=codes, scale=scale)
            compressed = normalize(unpack_rows(codes, scale, 384, 8))
            pack_setup = time.perf_counter() - began
        pointer_path = out / ("pointers_" + str(rung["target_tokens"]) + ".json")
        save(pointer_path, current)
        for qi, question in enumerate(questions):
            guard.check()
            start = time.perf_counter()
            lexical = bm25.score(question["question"])
            scores = {"bm25": lexical}
            times = {"bm25": time.perf_counter() - start}
            if dense_ready:
                for arm, matrix in (("dense", bank[:n]), ("packed8", compressed)):
                    start = time.perf_counter()
                    scores[arm] = matrix @ qbank[qi]
                    times[arm] = time.perf_counter() - start
                start = time.perf_counter()
                scores["hybrid"] = hybrid_scores(scores["dense"], lexical, current)
                times["hybrid"] = (time.perf_counter() - start
                                   + times["dense"] + times["bm25"])
            for arm, values in scores.items():
                start = time.perf_counter()
                order = canonical_order(values, current).tolist()
                sorting_seconds = time.perf_counter() - start
                for policy in POLICIES:
                    start = time.perf_counter()
                    selected, edges = (graph_route(order, document_ids)
                                       if policy == "graph" else (order[:10], 0))
                    navigation_seconds = time.perf_counter() - start
                    verified_bytes = 0
                    verified_tokens = 0
                    start = time.perf_counter()
                    for i in selected:
                        _, cost = verify_span(root, current[i])
                        verified_bytes += cost
                        verified_tokens += current[i]["tokens"]
                    verification_seconds = time.perf_counter() - start
                    rows.append(dict(
                        target_tokens=rung["target_tokens"], actual_tokens=rung["actual_tokens"],
                        question_id=question["id"], question_type=question["type"],
                        arm=arm, policy=policy, selected=selected,
                        **support_score(selected, current, golds[qi]["gold"]),
                        **support_chunk_score(selected, current, golds[qi]["gold"]),
                        verified_proposals=len(selected), proposals=len(selected),
                        verified_source_bytes=verified_bytes, prepared_passage_tokens=verified_tokens,
                        reader_calls=0, reader_input_tokens=0, reader_output_tokens=0,
                        reader_quality="UNRUN", semantic_absence="UNKNOWN",
                        graph_edges_examined=edges, graph_hops=1 if policy == "graph" else 0,
                        score_seconds=times[arm], sort_seconds=sorting_seconds,
                        navigation_seconds=navigation_seconds,
                        verification_seconds=verification_seconds,
                        wall_seconds=times[arm] + sorting_seconds + navigation_seconds + verification_seconds))
        fit = rung["actual_tokens"] + 512 + 96 <= preparation["reader_window"]
        summaries.append(dict(
            **rung, full_context="ESTIMATED_FIT_READER_UNRUN" if fit else "NOT_FIT",
            reader_window=preparation["reader_window"], reserved_query_output_tokens=608,
            source_prefix_bytes=current[-1]["end"], pointer_file_bytes=pointer_path.stat().st_size,
            bm25_file_bytes=bm25_path.stat().st_size, bm25_setup_seconds=bm25_setup,
            bm25_numeric_payload_bytes=bm25.nbytes,
            bm25_backend="retained inverted postings; same lexical formula",
            dense_status="MEASURED_SUPPORT_ONLY" if dense_ready else "UNRUN_ENCODER_NOT_ADMITTED",
            dense_payload_bytes=n * 384 * 4 if dense_ready else None,
            packed_file_bytes=packed_file.stat().st_size if packed_file else None,
            packed_setup_seconds=pack_setup if dense_ready else None,
            strong_text_harness="FIRST_STAGE_ONLY; reranker/iterative reader/cache UNRUN",
            tool="compressed routing + same document adjacency + exact-span verification" if dense_ready else "UNRUN",
            answer_accuracy=None, actual_reader_calls=0, actual_reader_tokens=0,
            equal_quality_crossover=None, amortization_questions=None))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["target_tokens"], row["question_type"], row["arm"], row["policy"])].append(row)
    aggregates = []
    for (size, kind, arm, policy), group in grouped.items():
        aggregates.append(dict(
            target_tokens=size, question_type=kind, arm=arm, policy=policy,
            questions=len(group), mean_support_recall=float(np.mean([r["recall"] for r in group])),
            all_sources_found=sum(r["all_sources_found"] for r in group),
            mean_support_chunk_recall=float(np.mean([r["support_chunk_recall"] for r in group])),
            all_support_chunks_found=sum(r["all_support_chunks_found"] for r in group),
            verified_provenance_rate=sum(r["verified_proposals"] for r in group) /
                sum(r["proposals"] for r in group),
            mean_verified_bytes=float(np.mean([r["verified_source_bytes"] for r in group])),
            median_wall_seconds=float(np.median([r["wall_seconds"] for r in group])),
            reader_answer_accuracy=None, measured_reader_calls=0, measured_reader_tokens=0))
    save(out / "rows.json", rows)
    result = dict(
        status="SUPPORT_ACCESS_SLICE_NOT_READER_PARITY", rungs=summaries, aggregate=aggregates,
        scope="C5/C7/C8 supporting development source-access SLICE; not whole-corpus completeness/C9",
        questions="same six fixed MuSiQue rows200..205 multi-hop questions plus six source-title lookups",
        selection="gold-support-enriched small core; real hash-ordered distractors; no duplicate padding",
        verification="exact UTF8-span byte identity, not entailment/truth/current authority",
        bytes_touched="verified leaf bytes measured; index/map scoring work timed; not OS physical I/O",
        prepared_tokens="not consumed model tokens; tokenizer counts differ from framing",
        text_loading_seconds=text_read_seconds, corpus_encoding_admitted=dense_ready,
        reader_calls_saved=None, reader_tokens_saved=None, equal_quality_crossover=None,
        setup_amortization=None, completeness="UNMEASURED", absence="UNKNOWN",
        ambiguity_exploration="UNRUN", revision="UNRUN",
        costs="local wall/setup/file bytes only; energy/money/actor/inherited model costs UNKNOWN",
        prepare_sha256=file_sha(root / "prepare.json"), rows_sha256=file_sha(out / "rows.json"),
        resources=guard.receipt())
    save(out / "SUMMARY.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "evaluate"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--phase", choices=("smoke", "panel"))
    parser.add_argument("--out")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.action == "prepare":
        print(json.dumps(prepare(root, root / "config.json")))
    elif args.action == "worker":
        encoding_worker(root, args.phase)
    else:
        result = evaluate(root, Path(args.out).resolve())
        print(json.dumps({"status": result["status"], "rungs": result["rungs"]}))


if __name__ == "__main__":
    main()
