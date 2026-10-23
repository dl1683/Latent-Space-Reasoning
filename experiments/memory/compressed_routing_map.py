"""Compressed routing only: geometry proposes, hash-checked source leaves verify.

No answer generation, semantic absence certificate, model downloads or cloud use.
CUDA encoding is separate from CPU map/query operation and needs a resource_guard.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, obj):
    with Path(path).open("x", encoding="utf8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def pack_rows(x, bits):
    """Per-row symmetric scalar quantization; actually store packed bit codes."""
    x = np.asarray(x, dtype=np.float32)
    if bits not in (1, 2, 4, 8) or x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("finite matrix and 1/2/4/8 bits required")
    scale = np.maximum(np.max(np.abs(x), axis=1), 1e-12).astype(np.float32)
    levels = 2**bits - 1
    codes = np.rint((x / scale[:, None] + 1) * levels / 2)
    codes = np.clip(codes, 0, levels).astype(np.uint8)
    binary = ((codes[:, :, None] >> np.arange(bits)) & 1).reshape(len(x), -1)
    packed = np.packbits(binary, axis=1, bitorder="little")
    return packed, scale


def unpack_rows(packed, scale, dim, bits):
    binary = np.unpackbits(packed, axis=1, count=dim * bits,
                           bitorder="little").reshape(len(packed), dim, bits)
    codes = (binary * (1 << np.arange(bits))).sum(-1)
    return ((2 * codes.astype(np.float32) / (2**bits - 1) - 1)
            * np.asarray(scale, dtype=np.float32)[:, None])


def projection(dim):
    if dim == 384:
        return None
    return (np.random.default_rng(1735 + dim).standard_normal((384, dim))
            / math.sqrt(dim)).astype(np.float32)


def ranked(scores):
    """Stable ties, including null/constant maps."""
    return np.argsort(-np.asarray(scores), kind="stable")


def unique_documents(order, chunks, k):
    seen, out = set(), []
    for index in order:
        doc = chunks[int(index)]["document"]
        if doc not in seen:
            seen.add(doc)
            out.append(int(index))
            if len(out) == k:
                break
    return out


def support_metrics(selected, chunks, gold):
    found = {chunks[i]["document"] for i in selected}
    return dict(recall=len(found & set(gold)) / len(gold),
                complete=set(gold) <= found)


def fetch_leaf(root, chunk):
    """Exact source is authoritative; a vector cannot validate an answer."""
    data = (Path(root) / chunk["file"]).read_bytes()
    if digest(data) != chunk["source_sha256"]:
        raise ValueError("source changed")
    text = data.decode("utf8")
    return text[chunk["start"]:chunk["end"]], len(data)


def save_pointers(path, chunks):
    """Shared 32-byte source hashes plus 12-byte chunk address triples."""
    sources = list(dict.fromkeys(c["document"] for c in chunks))
    lookup = {key: i for i, key in enumerate(sources)}
    hashes = np.asarray([list(bytes.fromhex(key)) for key in sources], dtype=np.uint8)
    spans = np.asarray([[lookup[c["document"]], c["start"], c["end"]]
                        for c in chunks], dtype=np.uint32)
    np.savez(path, source_hashes=hashes, spans=spans)


def load_pointers(path):
    with np.load(path, allow_pickle=False) as state:
        hashes, spans = state["source_hashes"], state["spans"]
    sources = [row.tobytes().hex() for row in hashes]
    return [dict(document=sources[int(i)], source_sha256=sources[int(i)],
                 file=f"leaves/{sources[int(i)]}.txt", start=int(a), end=int(b))
            for i, a, b in spans]


class PackedMap:
    """Usable source-linked CPU routing component, not a semantic answerer."""
    def __init__(self, directory, name, chunks):
        self.directory, self.name, self.chunks = Path(directory), name, chunks
        match = re.fullmatch(r"d(32|64|128|256|384)b(1|2|4|8)", name)
        if not match:
            raise ValueError("unsupported packed map")
        self.dim, self.bits = map(int, match.groups())
        with np.load(self.directory / (name + ".npz"), allow_pickle=False) as state:
            self.codes, self.scale = state["codes"], state["scale"]
        self.proj = (np.load(self.directory / f"projection{self.dim}.npy",
                             allow_pickle=False) if self.dim != 384 else None)
        if len(self.codes) != len(chunks):
            raise ValueError("pointer/map mismatch")

    def search(self, query_vector, k=10):
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, 384)
        if self.proj is not None:
            query = normalize(query @ self.proj)
        values = normalize(unpack_rows(self.codes, self.scale, self.dim, self.bits))
        scores = values @ query[0]
        selected = unique_documents(ranked(scores), self.chunks, k)
        return dict(status="UNKNOWN_UNTIL_SOURCE_VERIFICATION",
                    selected_chunks=selected, scores=scores[selected].tolist())

    def fetch(self, root, query_vector, k=10):
        result = self.search(query_vector, k)
        leaves, read_bytes = [], 0
        for i in result["selected_chunks"]:
            text, cost = fetch_leaf(root, self.chunks[i])
            leaves.append(dict(pointer=self.chunks[i], text=text))
            read_bytes += cost
        return dict(**result, leaves=leaves, source_bytes_read=read_bytes,
                    semantic_absence="UNKNOWN", completeness="UNKNOWN")


def lexical_tokens(text):
    return re.findall(r"\w+", text.lower())


class BM25:
    def __init__(self, texts):
        self.rows = [Counter(lexical_tokens(x)) for x in texts]
        self.lengths = np.asarray([sum(x.values()) for x in self.rows])
        self.average = max(float(self.lengths.mean()), 1)
        self.df = Counter()
        for row in self.rows:
            self.df.update(row.keys())

    def score(self, query):
        out = np.zeros(len(self.rows), dtype=np.float32)
        norm = 1.5 * (.25 + .75 * self.lengths / self.average)
        for word in set(lexical_tokens(query)):
            freq = np.asarray([x[word] for x in self.rows], dtype=np.float32)
            df = self.df[word]
            idf = math.log(1 + (len(self.rows) - df + .5) / (df + .5))
            out += idf * freq * 2.5 / (freq + norm)
        return out


def reciprocal_rank_fusion(a, b):
    out = np.zeros(len(a), dtype=np.float32)
    for scores in (a, b):
        out[ranked(scores)] += 1 / (60 + np.arange(1, len(a) + 1))
    return out


def chunk_text(text, tokenizer, width=160, stride=128):
    offsets = tokenizer(text, add_special_tokens=False,
                        return_offsets_mapping=True)["offset_mapping"]
    if not offsets:
        return [(0, len(text))]
    out = []
    for i in range(0, len(offsets), stride):
        end = min(i + width, len(offsets))
        start_char = 0 if i == 0 else offsets[i][0]
        end_char = len(text) if end == len(offsets) else offsets[end][0]
        out.append((start_char, end_char))
        if end == len(offsets):
            break
    if out[0][0] != 0 or out[-1][1] != len(text):
        raise AssertionError("incomplete coverage")
    if any(b[0] > a[1] for a, b in zip(out, out[1:])):
        raise AssertionError("chunk gap")
    return out


def encode(model, tokenizer, texts, batch_size=16):
    import torch
    result = []
    for start in range(0, len(texts), batch_size):
        check_ram()
        batch = tokenizer(texts[start:start + batch_size], padding=True,
                          truncation=False, return_tensors="pt")
        if batch["input_ids"].shape[1] > 256:
            raise ValueError("silent encoder truncation refused")
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"][:, :, None]
            vectors = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
        result.append(vectors.cpu().numpy())
    return np.concatenate(result)


def check_ram():
    import psutil
    if psutil.virtual_memory().available < 20.5 * 2**30:
        raise RuntimeError("RAM floor")


def gpu_model(path):
    import torch
    from transformers import AutoModel, AutoTokenizer
    if os.environ.get("LSR_MAP_SUPERVISED") != "1":
        raise RuntimeError("exclusive resource resource_guard required")
    check_ram()
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(path, local_files_only=True).cuda().eval()
    if model.config.hidden_size != 384:
        raise ValueError("registered model dimension mismatch")
    return model, tokenizer


def construction_binding(root, cfg):
    model = Path(cfg["model"])
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json",
                       "special_tokens_map.json", "vocab.txt")
    return dict(config_sha256=digest((root / "config.json").read_bytes()),
                worker_sha256=digest(Path(__file__).read_bytes()),
                model_sha256=digest((model / "model.safetensors").read_bytes()),
                tokenizer_sha256={name: digest((model / name).read_bytes())
                                  for name in tokenizer_files})


def require_smoke_binding(root, cfg):
    receipt = json.loads((root / "smoke.json").read_text())
    if not receipt.get("passed") or receipt.get("binding") != construction_binding(root, cfg):
        raise RuntimeError("separate smoke does not qualify these construction bytes")


def smoke(root, cfg):
    model, tok = gpu_model(cfg["model"])
    import torch
    texts = ["A cobalt observatory measures stellar motion.",
             "An orchard cultivates apples and pears."]
    batch = encode(model, tok, texts)
    single = np.concatenate([encode(model, tok, [x]) for x in texts])
    error = float(np.max(np.abs(batch - single)))
    np.save(root / "smoke_vectors.npy", batch)
    assert np.array_equal(np.load(root / "smoke_vectors.npy"), batch)
    assert error <= 1e-4
    assert (batch @ batch.T).argmax(1).tolist() == [0, 1]
    assert not np.allclose(batch[0], batch[1])
    for bits in (1, 2, 4, 8):
        packed, scales = pack_rows(batch, bits)
        restored = unpack_rows(packed, scales, 384, bits)
        assert np.isfinite(restored).all()
        assert (normalize(restored) @ batch.T).argmax(1).tolist() == [0, 1]
    long_text = " ".join(["The orchard grows apples."] * 150)
    spans = chunk_text(long_text, tok)
    assert spans[0][0] == 0 and spans[-1][1] == len(long_text)
    dump(root / "smoke.json", dict(passed=True, batch_error=error,
         cuda=torch.cuda.get_device_name(), torch=torch.__version__,
         model_sha256=digest((Path(cfg["model"]) / "model.safetensors").read_bytes()),
         binding=construction_binding(root, cfg)))


def prepare(root, cfg):
    """Copy corpus and labels separately; benchmark labels never enter encoding."""
    from itertools import islice
    documents, queries = {}, []
    with Path(cfg["dataset"]).open(encoding="utf8") as handle:
        for number, line in enumerate(islice(handle, 100)):
            obj = json.loads(line)
            if not obj["answerable"]:
                raise ValueError("registered answerable subset required")
            supports = []
            for item in obj["paragraphs"]:
                text = item["title"] + "\n" + item["paragraph_text"]
                raw = text.encode("utf8")
                key = digest(raw)
                documents.setdefault(key, text)
                if item["is_supporting"]:
                    supports.append(key)
            queries.append(dict(id=obj["id"], question=obj["question"],
                                split="DEV" if number < 20 else "EVAL",
                                gold=sorted(set(supports))))
    if len(queries) != 100:
        raise ValueError("100 examples required")
    leaves = root / "leaves"
    leaves.mkdir()
    for key, text in documents.items():
        (leaves / (key + ".txt")).write_bytes(text.encode("utf8"))
    dump(root / "sources.json", [dict(document=key, file=f"leaves/{key}.txt",
         source_sha256=key) for key in documents])
    dump(root / "queries.json", [dict(id=q["id"], question=q["question"])
                                for q in queries])
    dump(root / "evaluation_only.json", queries)
    dump(root / "source_pins.json", dict(dataset_sha256=digest(
         Path(cfg["dataset"]).read_bytes()), documents=len(documents),
         queries=len(queries), text_bytes=sum(len(t.encode("utf8"))
                                             for t in documents.values())))


def build(root, cfg):
    require_smoke_binding(root, cfg)
    model, tok = gpu_model(cfg["model"])
    chunks, texts = [], []
    sources = json.loads((root / "sources.json").read_text())
    for source in sources:
        raw = (root / source["file"]).read_bytes()
        if digest(raw) != source["source_sha256"]:
            raise ValueError("source pin mismatch")
        text = raw.decode("utf8")
        for start, end in chunk_text(text, tok):
            chunks.append(dict(**source, start=start, end=end))
            texts.append(text[start:end])
    dump(root / "chunks.json", chunks)
    save_pointers(root / "pointers.npz", chunks)
    assert load_pointers(root / "pointers.npz") == chunks
    questions = json.loads((root / "queries.json").read_text())
    begin = time.perf_counter()
    bank = encode(model, tok, texts)
    qbank = encode(model, tok, [q["question"] for q in questions])
    import torch
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    np.save(root / "bank.npy", bank)
    np.save(root / "query_vectors.npy", qbank)
    dump(root / "build.json", dict(chunks=len(chunks), documents=len(sources),
         encoding_seconds=seconds, encoded_corpus_chunks=len(texts),
         encoded_queries=len(qbank), gpu_peak_bytes=torch.cuda.max_memory_allocated(),
         corpus_sha256=digest((root / "sources.json").read_bytes()),
         evaluator_access=False, model=cfg["model"]))


def evaluate(root):
    """Model-free routing. Labels are opened only here, after immutable encoding."""
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    check_ram()
    bank = np.load(root / "bank.npy")
    queries = np.load(root / "query_vectors.npy")
    chunks = json.loads((root / "chunks.json").read_text())
    # Canonical hashes remove benchmark grouping from ties without re-encoding.
    permutation = sorted(range(len(chunks)), key=lambda i: (
        chunks[i]["document"], chunks[i]["start"], chunks[i]["end"]))
    bank = bank[permutation]
    chunks = [chunks[i] for i in permutation]
    save_pointers(root / "routing_pointers.npz", chunks)
    dump(root / "routing_chunks.json", chunks)
    golds = json.loads((root / "evaluation_only.json").read_text())
    texts = [fetch_leaf(root, c)[0] for c in chunks]
    lexical = BM25(texts)
    dump(root / "bm25_index.json", [dict(row) for row in lexical.rows])
    lexical_scores = np.stack([lexical.score(q["question"]) for q in golds])
    dense_scores = queries @ bank.T
    arms = dict(dense384=dense_scores, bm25=lexical_scores,
                hybrid=np.stack([reciprocal_rank_fusion(d, b)
                                 for d, b in zip(dense_scores, lexical_scores)]),
                fixed=np.zeros_like(dense_scores))
    rates = {name: dict(map_payload_bytes=bank.nbytes if name == "dense384" else None)
             for name in arms}
    maps = root / "maps"
    maps.mkdir()
    pointer_bytes = (root / "routing_pointers.npz").stat().st_size
    lexical_bytes = (root / "bm25_index.json").stat().st_size
    rates["bm25"].update(serialized_bytes=lexical_bytes,
                        total_serving_bytes=lexical_bytes + pointer_bytes)
    rates["dense384"].update(total_serving_bytes=(root / "bank.npy").stat().st_size + pointer_bytes)
    rates["hybrid"].update(total_serving_bytes=(root / "bank.npy").stat().st_size
                           + pointer_bytes + lexical_bytes)
    for dim in (32, 64, 128, 256, 384):
        proj = projection(dim)
        x = normalize(bank @ proj) if proj is not None else bank
        q = normalize(queries @ proj) if proj is not None else queries
        if proj is not None:
            np.save(maps / f"projection{dim}.npy", proj)
        arms[f"d{dim}float"] = q @ x.T
        rates[f"d{dim}float"] = dict(map_payload_bytes=x.nbytes,
            projection_bytes=0 if proj is None else proj.nbytes)
        for bits in (1, 2, 4, 8):
            name = f"d{dim}b{bits}"
            packed, scales = pack_rows(x, bits)
            np.savez(maps / (name + ".npz"), codes=packed, scale=scales)
            # Charge real query-time unpack/normalize/matmul/rank, not offline decode.
            times, scores = [], []
            for query in q:
                begin = time.perf_counter()
                restored = normalize(unpack_rows(packed, scales, dim, bits))
                score = restored @ query
                ranked(score)
                times.append(time.perf_counter() - begin)
                scores.append(score)
            arms[name] = np.stack(scores)
            rates[name] = dict(bits=bits, dimension=dim,
                map_payload_bytes=packed.nbytes + scales.nbytes,
                code_bytes=packed.nbytes, scale_bytes=scales.nbytes,
                projection_bytes=0 if proj is None else proj.nbytes,
                serialized_bytes=(maps / (name + ".npz")).stat().st_size,
                bytes_per_chunk=(packed.nbytes + scales.nbytes) / len(bank),
                median_scoring_kernel_seconds=float(np.median(times)),
                timing_boundary="unpack+normalize+dot+sort; excludes projection/document aggregation")
            projection_file_bytes = ((maps / f"projection{dim}.npy").stat().st_size
                                     if proj is not None else 0)
            rates[name]["total_serving_bytes"] = (
                rates[name]["serialized_bytes"] + projection_file_bytes + pointer_bytes)
            # Conventional compressed hybrid is an equal-access incumbent.
            hybrid_name = name + "hybrid"
            arms[hybrid_name] = np.stack([reciprocal_rank_fusion(d, b)
                for d, b in zip(arms[name], lexical_scores)])
            rates[hybrid_name] = dict(
                total_serving_bytes=rates[name]["total_serving_bytes"] + lexical_bytes,
                map_payload_bytes=packed.nbytes + scales.nbytes,
                lexical_index_bytes=lexical_bytes,
                projection_bytes=0 if proj is None else proj.nbytes)
    rows, aggregate = [], []
    for name, scores in arms.items():
        for k in (3, 5, 10, 20):
            for i, query in enumerate(golds):
                order = ranked(scores[i])
                order = order[np.isfinite(scores[i, order])]
                selected = unique_documents(order, chunks, k)
                metrics = support_metrics(selected, chunks, query["gold"])
                row = dict(arm=name, k=k, id=query["id"], split=query["split"],
                           selected_chunks=selected, **metrics)
                rows.append(row)
            for split in ("DEV", "EVAL"):
                subset = [r for r in rows if r["arm"] == name
                          and r["k"] == k and r["split"] == split]
                aggregate.append(dict(arm=name, k=k, split=split, n=len(subset),
                    mean_support_recall=float(np.mean([r["recall"] for r in subset])),
                    all_support_hits=sum(r["complete"] for r in subset)))
    eval10 = {r["arm"]: r for r in aggregate if r["split"] == "EVAL" and r["k"] == 10}
    ref = {r["id"]: r for r in rows if r["arm"] == "dense384"
           and r["k"] == 10 and r["split"] == "EVAL"}
    for name, rate in rates.items():
        comparable = [r for r in rows if r["arm"] == name and r["k"] == 10
                      and r["split"] == "EVAL"]
        rate["per_question_recall_parity"] = all(
            r["recall"] == ref[r["id"]]["recall"] for r in comparable)
        def retrieved_support(row):
            gold = next(q["gold"] for q in golds if q["id"] == row["id"])
            return {chunks[i]["document"] for i in row["selected_chunks"]} & set(gold)
        rate["identical_support_sets_vs_dense"] = all(
            retrieved_support(r) == retrieved_support(ref[r["id"]]) for r in comparable)
        rate["no_lost_dense_support"] = all(
            retrieved_support(ref[r["id"]]) <= retrieved_support(r) for r in comparable)
        rate["lost_complete_vs_dense"] = sum(
            ref[r["id"]]["complete"] and not r["complete"] for r in comparable)
        rate["mean_recall_retention_vs_dense"] = (
            eval10[name]["mean_support_recall"] / eval10["dense384"]["mean_support_recall"]
            if eval10["dense384"]["mean_support_recall"] else None)
    dump(root / "routing_rows.json", rows)
    dump(root / "routing_summary.json", dict(aggregate=aggregate, rates=rates,
         chunks=len(bank), documents=len({c["document"] for c in chunks}),
         pointer_json_bytes=(root / "chunks.json").stat().st_size,
         shared_pointer_bank_bytes=pointer_bytes,
         bm25_index_bytes=lexical_bytes,
         encoder_bytes=(Path(json.loads((root / "config.json").read_text())["model"])
                        / "model.safetensors").stat().st_size,
         corpus_reencodes_per_query=0, answer_parity=None, semantic_absence=None,
         reader_tokens_saved=None, avoided_corpus_encoding_vs_cached_baseline=0))


def main():
    expected = os.environ.get("LSR_MAP_EXPECTED_SHA256")
    if expected and digest(Path(__file__).read_bytes()) != expected:
        raise RuntimeError("worker changed between admission and execution")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "smoke", "build", "evaluate"),
                        required=True)
    args = parser.parse_args()
    cfg = json.loads((args.root / "config.json").read_text())
    if args.phase == "evaluate":
        evaluate(args.root)
    else:
        globals()[args.phase](args.root, cfg)


if __name__ == "__main__":
    main()
