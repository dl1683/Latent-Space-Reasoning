"""Small CPU/offline context-backend DEV check. Not the frozen reader pilot.

Reuse the reference pinned source-only MiniLM bank. Labels enter only AFTER all
80 real backend calls (40 questions x two strategies) have produced their packs.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np

from experiments.dogfood import digest, file_digest, ram_guard
from experiments.latent_context import ContextMemory
from experiments.latent_mcp import serialized


def save(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def prepared_inputs(directory):
    receipt = json.loads((directory / "round3_bridge_receipt.json").read_bytes())
    names = ("sources", "vectors", "preparation", "questions", "gold")
    paths = {name: directory / ("round3_bridge_" + name + (".npz" if name == "vectors" else ".json"))
             for name in names}
    for path in paths.values():
        if file_digest(path) != receipt["files"][path.name]:
            raise ValueError("reference input pin mismatch: " + path.name)
    return paths


def prepare(directory, output):
    started = time.perf_counter()
    ram_guard()
    inputs = prepared_inputs(directory)
    prep = json.loads(inputs["preparation"].read_bytes())
    dataset = Path(prep["dataset"])
    if file_digest(dataset) != prep["dataset_sha256"]:
        raise ValueError("MuSiQue dataset changed")
    # Verify source identities against the cached real dataset, never evaluator fields.
    universe = set()
    for number, line in enumerate(dataset.read_text(encoding="utf8").splitlines()):
        if prep["rows"][0] <= number < prep["rows"][1]:
            universe.update(p["title"] + "\n" + p["paragraph_text"]
                            for p in json.loads(line)["paragraphs"])
    sources = json.loads(inputs["sources"].read_bytes())
    texts = [p["title"] + "\n" + p["paragraph_text"] for p in sources]
    if set(texts) != universe or any(p["idx"] != i for i, p in enumerate(sources)):
        raise ValueError("real source universe/alignment differs")
    with np.load(inputs["vectors"], allow_pickle=False) as data:
        bank = data["bank"]
    if bank.dtype != np.float32 or bank.shape != (len(sources), prep["dimension"]):
        raise ValueError("frozen vector alignment differs")
    encoder = Path(prep["encoder"])
    if file_digest(encoder / "model.safetensors") != prep["encoder_model_sha256"]:
        raise ValueError("retained encoder differs")
    output.mkdir(parents=True, exist_ok=False)
    source, index = output / "sources", output / "index"
    source.mkdir()
    index.mkdir()
    leaves = index / "leaves"
    leaves.mkdir()
    rows, files = [], []
    for i, text in enumerate(texts):
        raw = text.encode()
        path = source / f"p{i:04}.txt"
        path.write_bytes(raw)
        (leaves / (digest(raw) + ".txt")).write_bytes(raw)
        files.append(dict(path=str(path), sha256=digest(raw), bytes=len(raw), extraction="utf8_text"))
        rows.append(dict(source=str(path), source_sha256=digest(raw), page=None,
                         leaf_sha256=digest(raw), start_byte=0, end_byte=len(raw),
                         span_sha256=digest(raw), text=text))
    save(index / "rows.json", rows)
    np.save(index / "vectors.npy", bank, allow_pickle=False)
    allowed = [f["path"] for f in files]
    save(index / "manifest.json", dict(files=files, allowed_paths=allowed,
        source_roots=[str(source)], encoder=str(encoder), rows_sha256=file_digest(index / "rows.json"),
        vector_sha256=file_digest(index / "vectors.npy")))
    save(output / "latent-memory.json", dict(index_dir=str(index), source_root=str(source),
        allowed_paths=allowed, manifest_sha256=file_digest(index / "manifest.json"),
        encoder=str(encoder), routing="hybrid", max_results=8, max_output_bytes=48000))
    save(output / "preparation.json", dict(scope="DEV source access SLICE; not reader pilot",
        selection="first 40 of reference DEV rows 400:480, no retrieval-outcome selection",
        source_count=len(rows), vector_bytes=int(bank.nbytes), corpus_reencodes=0,
        corpus_encoding="reused frozen reference MiniLM bank; original preparation cost retained below",
        inherited_preparation=prep, seconds=time.perf_counter()-started,
        pins={p.name: file_digest(p) for p in inputs.values()},
        config_sha256=file_digest(output / "latent-memory.json")))
    print(serialized(dict(operation="prepare", sources=len(rows), output=str(output))), flush=True)


def evaluate(directory, output, run_name):
    import psutil
    inputs = prepared_inputs(directory)
    questions = json.loads(inputs["questions"].read_bytes())[:40]
    if len(questions) != 40:
        raise ValueError("exactly 40 DEV questions required")
    prep = json.loads((output / "preparation.json").read_bytes())
    if any(file_digest(path) != prep["pins"][path.name] for path in inputs.values()):
        raise ValueError("prepared inputs changed")
    config = output / "latent-memory.json"
    if file_digest(config) != prep["config_sha256"]:
        raise ValueError("prepared corpus config changed")
    samples = []
    def guard():
        ram_guard()
        samples.append(psutil.virtual_memory().available / 2**30)
    guard()
    memory = ContextMemory(config, description="MuSiQue DEV paragraph union, frozen reference vectors.")
    if next(memory.tool.memory.encoder.parameters()).device.type != "cpu":
        raise ValueError("CPU encoder required")
    raw_rows = []
    budget = dict(bytes=48000, tokens=6000, max_spans=8)
    with (output / (run_name + ".packs.jsonl")).open("x", encoding="utf8") as stream:
        for number, question in enumerate(questions):
            for strategy in ("one_shot", "expand"):
                guard()
                pack = memory.context(question["question"], corpus=str(config), strategy=strategy, budget=budget)
                spans = [s for g in pack["evidence_groups"] for s in g["spans"]]
                assert memory.meter.fits(pack, budget)
                ids = {s["id"] for s in spans}
                for group in pack["evidence_groups"]:
                    if group.get("kind") == "PROVISIONAL_SOURCE_BRIDGE":
                        assert set(group["parents"]) <= ids
                for span in spans:
                    raw = Path(span["source_path"]).read_bytes()
                    assert digest(raw) == span["source_sha256"]
                    assert digest(raw[span["start_byte"]:span["end_byte"]]) == span["sha256"]
                stream.write(serialized(dict(id=question["id"], pack=pack)) + "\n")
                stream.flush()
                raw_rows.append(dict(id=question["id"], strategy=strategy,
                    delivered=[int(Path(s["file"]).stem[1:]) for s in spans],
                    delivered_bytes=sum(len(s["text"].encode()) for s in spans),
                    wire_bytes=len(serialized(pack).encode()), costs=pack["costs"],
                    stop_reason=pack["stop_reason"], continuation=pack["continuation"] is not None))
            print(serialized(dict(completed=number+1, questions=40)), flush=True)
    # Gold is not provided to memory, seeds, graph, packing or routing.
    gold = {g["id"]: g for g in json.loads(inputs["gold"].read_bytes())}
    for row in raw_rows:
        parents = gold[row["id"]]["parents"]
        row.update(supports=sum(i in row["delivered"] for i in parents),
                   complete=all(i in row["delivered"] for i in parents))
    table = {}
    for strategy in ("one_shot", "expand"):
        selected = [r for r in raw_rows if r["strategy"] == strategy]
        accesses = [r["costs"]["access"] for r in selected]
        table[strategy] = dict(questions=40, complete=sum(r["complete"] for r in selected),
            supports=sum(r["supports"] for r in selected), support_total=80,
            delivered_bytes=sum(r["delivered_bytes"] for r in selected),
            wire_bytes=sum(r["wire_bytes"] for r in selected),
            verification_bytes=sum(r["costs"]["verification_bytes_read"] for r in selected),
            query_encodes=sum(r["costs"]["query_encoder_calls"] for r in selected),
            query_seconds=sum(r["costs"]["query_seconds"] for r in selected),
            charged_access={key: (sum(a.get(key, 0) for a in accesses)
                                 if any(key in a for a in accesses) else None) for key in (
                "map_bytes_read", "exact_vector_rows_read", "exact_vector_bytes_read",
                "edge_reads", "edge_record_bytes_read", "lexical_row_scans",
                "source_opens", "leaf_opens", "verification_hash_checks", "verification_hash_bytes")},
            stop_reasons=dict(Counter(r["stop_reason"] for r in selected)),
            continuation_pages_not_scored=sum(r["continuation"] for r in selected))
    paired = {q["id"]: {r["strategy"]: r["complete"] for r in raw_rows if r["id"] == q["id"]}
              for q in questions}
    result = dict(scope="40-question DEV check of FIRST packs, not answer accuracy or pilot",
        table=table, paired_expand_wins=sum(v["expand"] and not v["one_shot"] for v in paired.values()),
        paired_expand_losses=sum(v["one_shot"] and not v["expand"] for v in paired.values()),
        budget=budget, rows=raw_rows, min_sampled_free_gib=min(samples),
        cold_startup_seconds=memory.startup_seconds, map_setup=memory._evidence_map.setup_costs,
        source_reencodes_per_query=0, reader_calls=0, cpu_only=True,
        original_preparation=prep, script_sha256=file_digest(__file__),
        packs_sha256=file_digest(output / (run_name + ".packs.jsonl")))
    save(output / (run_name + ".results.json"), result)
    print(serialized(dict(table=table, paired_expand_wins=result["paired_expand_wins"],
                          paired_expand_losses=result["paired_expand_losses"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "evaluate"))
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="dev40")
    args = parser.parse_args()
    if not args.prepared_dir.is_absolute() or not args.output_dir.is_absolute():
        parser.error("absolute paths required")
    if not args.run_name.isidentifier():
        parser.error("run-name must be a simple identifier")
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      LSR_CPU_RAM_FLOOR_GIB="10", TOKENIZERS_PARALLELISM="false")
    if args.operation == "prepare":
        prepare(args.prepared_dir, args.output_dir)
    else:
        evaluate(args.prepared_dir, args.output_dir, args.run_name)


if __name__ == "__main__":
    main()
