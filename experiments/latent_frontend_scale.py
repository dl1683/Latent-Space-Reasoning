"""MuSiQue-only source size ladder, CPU construction/routing, no reader scores.

Only title/body fields are projected from explicitly named cached dataset files.
No question, answer, alias, support or decomposition enters the memory. Whole
source chunks reach each rung; token units are the named cached Phi tokenizer.
This separate global-search corpus is NOT the provided-context dev benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from experiments.memory.compressed_routing_map import lexical_tokens, reciprocal_rank_fusion
from experiments.memory.inverted_bm25 import InvertedBM25
from experiments.memory.latent_frontend import expand, sha
from experiments.memory.refined_routing_map import RefinedMap, feature_scores, rank
from experiments.retrieval_benchmark import encoder_at, generation_at, guard, read, save


def build(config_path, out):
    from tokenizers import Tokenizer
    c = read(config_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    reader_tok = Tokenizer.from_file(c["scale_tokenizer"])
    reader_tok.no_truncation()
    model = encoder_at(c["encoder"])
    encoder_tok = model.tokenizer
    unique = {}
    source_pins = {}
    for name in c["scale_sources"]:
        path = Path(name)
        source_pins[name] = sha(path.read_bytes())
        with path.open(encoding="utf8") as stream:
            for line in stream:
                row = json.loads(line)
                for p in row["paragraphs"]:
                    raw = (p["title"]+"\n"+p["paragraph_text"]).encode("utf8")
                    unique.setdefault(sha(raw), raw)
    # Document order is query/label independent, no support enrichment or padding.
    chunks, counts, total, documents = [], [], 0, []
    with (out/"leaves.bin").open("xb") as stream:
        for h in sorted(unique):
            raw = unique[h]
            text = raw.decode("utf8")
            offsets = encoder_tok(text, add_special_tokens=False, truncation=False,
                                  return_offsets_mapping=True)["offset_mapping"]
            cuts = [0] + [offsets[j][0] for j in range(160, len(offsets), 160)] + [len(text)]
            for a, b in zip(cuts, cuts[1:]):
                piece = text[a:b]
                if not piece:
                    continue
                data = piece.encode("utf8")
                at = stream.tell()
                stream.write(data)
                count = len(reader_tok.encode(piece, add_special_tokens=False).ids)
                if count < 1:
                    continue
                chunks.append(dict(source_sha256=h, start=at, end=at+len(data),
                                   span_sha256=sha(data)))
                counts.append(count)
                documents.append(h)
                total += count
                if total >= 10_000_000:
                    break
            if total >= 10_000_000:
                break
    del unique
    save(out/"pointers.json", chunks)
    save(out/"token_counts.json", counts)
    bank = np.lib.format.open_memmap(out/"vectors.npy", mode="w+", dtype=np.float32,
                                    shape=(len(chunks), model.get_sentence_embedding_dimension()))
    with (out/"leaves.bin").open("rb") as stream:
        for at in range(0, len(chunks), 32):
            guard()
            texts = []
            for p in chunks[at:at+32]:
                stream.seek(p["start"])
                text = stream.read(p["end"]-p["start"]).decode("utf8")
                if len(encoder_tok(text, add_special_tokens=True)["input_ids"]) > model.max_seq_length:
                    raise ValueError("encoder truncation refused")
                texts.append(text)
            bank[at:at+len(texts)] = model.encode(texts, batch_size=16,
                normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            if at % 3200 == 0:
                bank.flush()
                print(json.dumps(dict(phase="scale_encode", chunks=at, total=len(chunks))), flush=True)
    bank.flush()
    del bank
    save(out/"BUILD.json", dict(source_files=source_pins, chunks=len(chunks),
         documents=len(set(documents)), actual_phi_chunk_tokens=total,
         source_only=True, ram_floor_gib=10, seconds=time.perf_counter()-start,
         source_bytes=(out/"leaves.bin").stat().st_size,
         encoder_identity=c["encoder"], tokenizer=c["scale_tokenizer"],
         input_fields=["title","paragraph_text"], question_conditioned_selection=False,
         source_policy="hash-ordered unique MuSiQue train/dev source paragraphs",
         completeness="UNKNOWN"))


def measure(config_path, root, out):
    c = read(config_path)
    root = Path(root)
    construction = read(root/"BUILD.json")
    vector_path = root/construction.get("vector_file", "vectors.npy")
    if (construction.get("vector_sha256") is not None
            and sha(vector_path.read_bytes()) != construction["vector_sha256"]):
        raise ValueError("completed vector bank changed")
    bank = np.load(vector_path, mmap_mode="r", allow_pickle=False)
    counts = np.cumsum(read(root/"token_counts.json"))
    pointers = read(root/"pointers.json")
    # Same already encoded public questions; no per-query corpus encoding.
    queries = np.load(Path(c["work"])/"queries.npy", allow_pickle=False)
    rows = generation_at(c["generation"])
    output = []
    with (root/"leaves.bin").open("rb") as stream:
        for target in (10_000, 100_000, 1_000_000, 10_000_000):
            guard()
            n = min(int(np.searchsorted(counts, target))+1, len(bank))
            ps = pointers[:n]
            texts = []
            for p in ps:
                stream.seek(p["start"])
                data = stream.read(p["end"]-p["start"])
                if sha(data) != p["span_sha256"]:
                    raise ValueError("source span changed")
                texts.append(data.decode("utf8"))
            setup_start = time.perf_counter()
            memory = RefinedMap.build(np.asarray(bank[:n]), 8)
            packed_seconds = time.perf_counter()-setup_start
            before = time.perf_counter()
            lexical = InvertedBM25(texts)
            bm25_seconds = time.perf_counter()-before
            graph = [[] for _ in range(n)]
            for i in range(n):
                for j in (i-1, i+1):
                    if 0 <= j < n and ps[i]["source_sha256"] == ps[j]["source_sha256"]:
                        graph[i].append(dict(target=j))
            costs = {arm:[] for arm in ("bm25","hybrid","latent")}
            for row, q in zip(rows, queries):
                guard()
                k = min(6, n)
                for arm in costs:
                    begin = time.perf_counter()
                    if arm == "latent":
                        result = memory.search(q, lambda ids: bank[ids], k=k)
                        order = result["indices"].tolist()
                        order += [i for i in range(n) if i not in order]
                        numeric = memory.payload_bytes + result["logical_exact_vector_bytes"]
                        posting = 0
                    else:
                        lex = lexical.score(row["question"])
                        posting = lexical.query_work(row["question"])["postings_visited"]
                        if arm == "hybrid":
                            dense = feature_scores(bank[:n], q)
                            scores = reciprocal_rank_fusion(dense, lex) if np.any(lex) else dense
                            numeric = n*bank.shape[1]*4
                        else:
                            scores = lex
                            numeric = 0
                        order = rank(scores).tolist()
                    selected, visits = expand(order, graph, k)
                    source_bytes = 0
                    for i in selected:
                        p = ps[i]
                        stream.seek(p["start"])
                        data = stream.read(p["end"]-p["start"])
                        if sha(data) != p["span_sha256"]:
                            raise ValueError("source span changed")
                        source_bytes += len(data)
                    costs[arm].append(dict(seconds=time.perf_counter()-begin,
                        numeric_bytes_touched=int(numeric), source_bytes_touched=source_bytes,
                        postings_visited=posting, graph_edge_visits=visits))
            output.append(dict(target_tokens=target, actual_tokens=int(counts[n-1]), chunks=n,
                status="AVAILABLE" if counts[-1] >= target else "INSUFFICIENT_SOURCE",
                flat_text_fits_16k=bool(counts[n-1]+640 <= 16384),
                fit_scope="unframed Phi tokenizer chunk sum; not a measured reader request",
                map_payload_bytes=memory.payload_bytes, retained_exact_bank_bytes=int(bank[:n].nbytes),
                bm25_numeric_bytes=lexical.nbytes, map_build_seconds=packed_seconds,
                bm25_build_seconds=bm25_seconds,
                arms={arm:dict(queries=len(v), median_seconds=float(np.median([r["seconds"] for r in v])),
                        median_numeric_bytes_touched=float(np.median([r["numeric_bytes_touched"] for r in v])),
                        median_verified_source_bytes=float(np.median([r["source_bytes_touched"] for r in v])),
                        median_postings_visited=float(np.median([r["postings_visited"] for r in v])))
                      for arm,v in costs.items()},
                reader_calls=0, reader_tokens=0, answer_quality=None,
                graph="equal same-document chunk adjacency, not semantic reference edges"))
            print(json.dumps(dict(phase="scale_measure", target=target, chunks=n)), flush=True)
    save(out, dict(rungs=output, scope="routing cost only; no answer parity or full-cost crossover",
                   corpus="MuSiQue-only global corpus, distinct from per-question supplied contexts",
                   construction=read(root/"BUILD.json"), lexical_physical_bytes="UNKNOWN",
                   physical_io="UNKNOWN", reader_quality="UNRUN", corpus_reencodes_per_query=0))


def resume_build(config_path, out):
    """Checkpoint encoding of existing owned source chunks; keep old partial bank."""
    c, out = read(config_path), Path(out)
    if (out / "BUILD.json").exists():
        raise FileExistsError("completed construction already exists")
    chunks = read(out / "pointers.json")
    counts = read(out / "token_counts.json")
    if not chunks or len(chunks) != len(counts):
        raise ValueError("source/chunk roster mismatch")
    pins = {name: sha((out / name).read_bytes()) for name in (
        "pointers.json", "token_counts.json", "leaves.bin")}
    pins["config"] = sha(Path(config_path).read_bytes())
    pins["code"] = sha(Path(__file__).read_bytes())
    pins["sources"] = {name: sha(Path(name).read_bytes()) for name in c["scale_sources"]}
    state_path = out / "ENCODING.json"
    vector_file = "vectors_checkpointed.npy"
    started = time.perf_counter()
    model = encoder_at(c["encoder"])
    shape = (len(chunks), model.get_sentence_embedding_dimension())
    if state_path.exists():
        state = read(state_path)
        if state["pins"] != pins:
            raise ValueError("checkpoint input/code binding changed")
        at = state["completed"]
        if (type(at) is not int or not 0 <= at <= len(chunks)
                or (at != len(chunks) and at % 32)
                or len(state.get("batch_sha256", [])) != (at + 31) // 32):
            raise ValueError("invalid checkpoint completion cursor")
        bank = np.lib.format.open_memmap(out / vector_file, mode="r+")
        if bank.shape != shape or bank.dtype != np.float32:
            raise ValueError("checkpoint bank shape/dtype changed")
        for index, offset in enumerate(range(0, at, 32)):
            batch = bank[offset:min(offset+32, at)]
            if (not np.isfinite(batch).all()
                    or sha(batch.tobytes()) != state["batch_sha256"][index]):
                raise ValueError("completed checkpoint vectors changed")
    else:
        if (out / vector_file).exists():
            raise FileExistsError("unbound bank exists; preserve it")
        state = dict(pins=pins, completed=0, seconds=0, batch_sha256=[],
                     prior_uncheckpointed_encoding_cost="UNKNOWN, partial bank retained")
        bank = np.lib.format.open_memmap(out / vector_file, mode="w+",
                                        dtype=np.float32, shape=shape)
        save(state_path, state)
        at = 0
    previous_seconds = state["seconds"]
    with (out / "leaves.bin").open("rb") as stream:
        for offset in range(at, len(chunks), 32):
            guard()
            texts = []
            for pointer in chunks[offset:offset+32]:
                stream.seek(pointer["start"])
                raw = stream.read(pointer["end"] - pointer["start"])
                if sha(raw) != pointer["span_sha256"]:
                    raise ValueError("chunk/source hash mismatch")
                text = raw.decode("utf8")
                if len(model.tokenizer(text, add_special_tokens=True)["input_ids"]) > model.max_seq_length:
                    raise ValueError("encoder truncation refused")
                texts.append(text)
            bank[offset:offset+len(texts)] = model.encode(
                texts, batch_size=16, normalize_embeddings=True,
                show_progress_bar=False, convert_to_numpy=True)
            batch = bank[offset:offset+len(texts)]
            if not np.isfinite(batch).all():
                raise ValueError("nonfinite encoder output")
            bank.flush()
            state["batch_sha256"].append(sha(batch.tobytes()))
            state.update(completed=offset+len(texts),
                         seconds=previous_seconds + time.perf_counter()-started)
            temporary = out / "ENCODING.next.json"
            save(temporary, state)
            os.replace(temporary, state_path)
            if offset % 3200 == 0:
                print(json.dumps(dict(phase="scale_resume", completed=state["completed"],
                                      chunks=len(chunks))), flush=True)
    del bank
    if {name: sha(Path(name).read_bytes()) for name in c["scale_sources"]} != pins["sources"]:
        raise ValueError("source files changed during encoding")
    save(out / "BUILD.json", dict(
        source_files=pins["sources"],
        chunks=len(chunks), documents=len({p["source_sha256"] for p in chunks}),
        actual_phi_chunk_tokens=sum(counts), source_only=True, ram_floor_gib=10,
        seconds=state["seconds"], prior_construction_seconds="UNKNOWN",
        source_bytes=(out / "leaves.bin").stat().st_size,
        encoder_identity=c["encoder"], tokenizer=c["scale_tokenizer"],
        vector_file=vector_file, vector_sha256=sha((out / vector_file).read_bytes()),
        input_fields=["title", "paragraph_text"],
        question_conditioned_selection=False, completeness="UNKNOWN",
        source_policy="hash-ordered unique MuSiQue train/dev source paragraphs",
        checkpoint_pins=pins))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("build","resume","measure"))
    p.add_argument("--config", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out")
    a = p.parse_args()
    if a.phase == "build":
        build(a.config, a.root)
    elif a.phase == "resume":
        resume_build(a.config, a.root)
    else:
        measure(a.config, a.root, a.out)
