"""Evaluator-only preparation, leakage checks, EM/F1 and paired bootstrap.

Gold never enters musique_dev generation or routing. Oracle evidence selection
is privileged and explicitly separate. Public benchmark pretraining exposure
cannot be ruled out by the input canary/overlap checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import string

from experiments.musique_dev import (ARMS, LexicalIndex, attach_quotes, encoded,
    latest_attempts, load, read_rows, sha, spans, validate_row, verify_spans, write)

DATASET_SHA = "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"


def normalize_answer(s):
    s = "".join(ch for ch in s.lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def score(prediction, answers):
    p = normalize_answer(prediction).split()
    em, f1 = 0, 0.
    for answer in answers:
        g = normalize_answer(answer).split()
        em = max(em, int(normalize_answer(prediction) == normalize_answer(answer)))
        common = sum((Counter(p) & Counter(g)).values())
        value = 2 * common / (len(p) + len(g)) if p and g else float(p == g)
        f1 = max(f1, value)
    return dict(em=em, f1=f1)


def bootstrap(values, *, seed=1729, iterations=10000):
    import numpy as np
    v = np.asarray(values, dtype=float)
    if not len(v):
        return None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (iterations, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return dict(mean=float(v.mean()), ci95=[float(lo), float(hi)],
                n=len(v), iterations=iterations, unit="question",
                limitation="IID question bootstrap; shared source/decomposition dependence possible")


def prepare(dataset, directory, encoder, count=150):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    raw = Path(dataset).read_bytes()
    if sha(raw) != DATASET_SHA:
        raise ValueError("wrong MuSiQue dev snapshot")
    selected = [json.loads(line) for line in raw.splitlines()[:count]]
    if len(selected) != count or count < 150:
        raise ValueError("need at least 150 source rows")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    generation, gold, oracle = [], [], []
    for row in selected:
        gen = dict(id=row["id"], question=row["question"], paragraphs=[
            {k: p[k] for k in ("idx", "title", "paragraph_text")}
            for p in row["paragraphs"]])
        generation.append(validate_row(gen))
        gold.append(dict(id=row["id"], answer=row["answer"],
                         answer_aliases=row.get("answer_aliases", []),
                         supporting=[p["idx"] for p in row["paragraphs"] if p["is_supporting"]]))
        oracle.append(dict(id=row["id"], indices=gold[-1]["supporting"]))
    for name, rows in (("generation", generation), ("gold", gold), ("oracle_indices", oracle)):
        (directory / f"{name}.jsonl").write_bytes(b"".join(encoded(r) + b"\n" for r in rows))
    # Encode each unique source only once, never questions, labels or answers into memory.
    texts = sorted({p["title"] + "\n" + p["paragraph_text"]
                    for row in generation for p in row["paragraphs"]})
    torch.set_num_threads(2)
    model = SentenceTransformer(str(encoder), device="cpu", local_files_only=True)
    embeddings = model.encode(texts, batch_size=16, normalize_embeddings=True,
                              show_progress_bar=False)
    query_vectors = model.encode([r["question"] for r in generation], batch_size=16,
                                  normalize_embeddings=True, show_progress_bar=False)
    index = {text: i for i, text in enumerate(texts)}
    routes = []
    for row, q in zip(generation, query_vectors):
        ids = [index[p["title"] + "\n" + p["paragraph_text"]] for p in row["paragraphs"]]
        scores = embeddings[ids] @ q
        order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i))
        routes.append(dict(id=row["id"], order=order))
    (directory / "routes.jsonl").write_bytes(b"".join(encoded(r) + b"\n" for r in routes))
    np.savez_compressed(directory / "dense_map.npz", vectors=embeddings)
    write(directory / "dense_sources.json", [sha(t.encode("utf8")) for t in texts])
    write(directory / "preparation.json", dict(
        dataset_sha256=sha(raw), selection="first N in pinned dev file order",
        questions=count, unique_sources=len(texts), encoder=str(encoder),
        encoder_scope="generic dense retrieval, NOT the explorer latent/attention component",
        encoder_source_tokens_truncated_at=model.max_seq_length,
        oracle="gold support IDs only; privileged diagnostic, never normal routing",
        pretraining_contamination="UNKNOWN, public benchmark may occur in reader pretraining"))


def audit(config_path, output):
    config = load(config_path)
    rows = [validate_row(r) for r in read_rows(config["generation"])]
    by_id = {r["id"]: r for r in rows}
    gold = read_rows(Path(config["generation"]).parent / "gold.jsonl")
    # Random evaluator-only canary tests the scanner, not the model.
    canary = "LSR_EVAL_ONLY_" + os.urandom(24).hex()
    scanner = lambda text: canary in text
    source_text = "\n".join(p["paragraph_text"] for r in rows for p in r["paragraphs"])
    forbidden_keys = re.compile(r'"(?:answer|answer_aliases|is_supporting|question_decomposition)"\s*:')
    raw = Path(config["generation"]).read_text(encoding="utf8")
    errors, overlaps = [], []
    if scanner(source_text) or not scanner(source_text + canary):
        errors.append("canary sensitivity/isolation failed")
    if forbidden_keys.search(raw):
        errors.append("evaluator field in generation projection")
    # Answers naturally occur in valid source passages. Report, do not mislabel as leakage.
    for g in gold:
        row = by_id[g["id"]]
        answers = [g["answer"], *g["answer_aliases"]]
        overlaps.append(dict(id=g["id"], answer_occurs_in_own_sources=any(
            a.casefold() in p["paragraph_text"].casefold()
            for a in answers if a for p in row["paragraphs"])))
    pred_path = Path(output) / "predictions.jsonl"
    if pred_path.exists():
        for p in read_rows(pred_path):
            if p["status"] != "COMPLETED":
                continue
            try:
                verify_spans(by_id[p["id"]], p["evidence"])
                accepted, rejected = attach_quotes(
                    p["evidence"], [c["quote"] for c in p["citations"]])
                if accepted != p["citations"] or rejected:
                    errors.append("citation mismatch: " + p["id"])
            except (ValueError, KeyError) as e:
                errors.append(str(e))
    for key, expected in config["input_sha256"].items():
        if sha(Path(config[key]).read_bytes()) != expected:
            errors.append("input hash mismatch: " + key)
    report = dict(status="PASS" if not errors else "FAIL", errors=errors,
                  canary_positive_control=True, canary_in_sources=False,
                  source_projection_exact=True, answer_source_overlap=overlaps,
                  scope="schema/canary/exact own-source span checks, not semantic contamination proof",
                  reader_pretraining_contamination="UNKNOWN")
    write(Path(output) / "leakage_audit.json", report)
    return report


def evaluate(config_path, output, gold_path):
    config = load(config_path)
    gold = {r["id"]: r for r in read_rows(gold_path)}
    generation = {r["id"]: r for r in read_rows(config["generation"])}
    planned = list(generation)
    if set(gold) != set(planned):
        raise ValueError("gold roster mismatch")
    output = Path(output)
    predictions = read_rows(output / "predictions.jsonl") if (output / "predictions.jsonl").exists() else []
    by_key = latest_attempts(predictions)
    unexpected = [list(key) for key in by_key if key[2] not in gold
                  or key[0] not in {r["id"] for r in config["readers"]}
                  or key[1] not in config["arms"]]
    scored, summary, paired = [], [], []
    for reader in config["readers"]:
        metrics = {}
        for arm in config["arms"]:
            completed, failures = [], Counter()
            for ident in planned:
                p = by_key.get((reader["id"], arm, ident))
                g = gold[ident]
                result = dict(reader=reader["id"], arm=arm, id=ident,
                              status=p["status"] if p else "NOT_RUN", em=None, f1=None)
                if p and p["status"] == "COMPLETED":
                    result.update(score(p["answer"], [g["answer"], *g["answer_aliases"]]))
                    verify_spans(generation[ident], p["evidence"])
                    full = set()
                    for paragraph in generation[ident]["paragraphs"]:
                        intervals = sorted((s["start_byte"], s["end_byte"])
                                           for s in p["evidence"] if s["idx"] == paragraph["idx"])
                        end = 0
                        for a, b in intervals:
                            if a > end:
                                break
                            end = max(end, b)
                        if end == len(paragraph["paragraph_text"].encode("utf8")):
                            full.add(paragraph["idx"])
                    result["complete_annotated_support"] = set(g["supporting"]) <= full
                    result["answer"] = p["answer"]
                    completed.append(result)
                    if result["em"] < 1:
                        if p["answer"].strip() == "UNKNOWN":
                            reason = "abstained_with_all_support" if result["complete_annotated_support"] else "abstained_missing_support"
                        else:
                            reason = "wrong_with_all_support" if result["complete_annotated_support"] else "wrong_missing_support"
                        failures[reason] += 1
                elif p:
                    failures["runtime_or_protocol_failure"] += 1
                scored.append(result)
            full = len(completed) == len(planned)
            metrics[arm] = {r["id"]: r for r in completed}
            n = len(completed)
            entry = dict(reader=reader["id"], arm=arm, completed=n, planned=len(planned),
                         status="COMPLETE" if full else "PARTIAL",
                         em=bootstrap([r["em"] for r in completed]) if full else None,
                         f1=bootstrap([r["f1"] for r in completed]) if full else None,
                         conditional_em=sum(r["em"] for r in completed) / n if n else None,
                         conditional_f1=sum(r["f1"] for r in completed) / n if n else None,
                         distinct_answers=len({r["answer"] for r in completed}),
                         unknown_answers=sum(r["answer"] == "UNKNOWN" for r in completed),
                         empty_answers=sum(not r["answer"].strip() for r in completed),
                         reader_calls=sum(p["calls"] for p in predictions
                                          if p["reader"] == reader["id"] and p["arm"] == arm
                                          and p["id"] in gold),
                         failure_indicators=dict(failures))
            summary.append(entry)
        for arm in config["arms"]:
            if arm == "bm25":
                continue
            common = [i for i in planned if i in metrics["bm25"] and i in metrics[arm]]
            paired.append(dict(reader=reader["id"], arm=arm, baseline="bm25",
                paired=len(common), planned=len(planned),
                delta_f1=bootstrap([metrics[arm][i]["f1"] - metrics["bm25"][i]["f1"]
                                    for i in common]) if len(common) == len(planned) else None))
    constants = []
    modal = Counter(g["answer"] for g in gold.values()).most_common(1)[0][0]
    for value in ("", "UNKNOWN", modal):
        scores = [score(value, [g["answer"], *g["answer_aliases"]]) for g in gold.values()]
        constants.append(dict(answer=value, em=sum(s["em"] for s in scores) / len(scores),
                              f1=sum(s["f1"] for s in scores) / len(scores),
                              type="analytic oracle-modal" if value == modal else "analytic fixed"))
    run_metadata = load(output / "run.json") if (output / "run.json").exists() else {}
    write(output / "scores.json", dict(
        metric="MuSiQue normalized answer EM/token F1, max over gold aliases; no LLM judge",
        latent_component=run_metadata.get("latent_component", False),
        summary=summary, paired=paired, constants=constants,
        unexpected_prediction_keys=unexpected,
        per_question=scored, pretraining_contamination="UNKNOWN",
        sealed_six="NOT_OPENED_OR_SCORED_THIS_RUN",
        failure_scope="lexical/exposure indicators, not a causal semantic taxonomy"))
    return summary


def main():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for key in ("dataset", "directory", "encoder"):
        prep.add_argument("--" + key, required=True)
    prep.add_argument("--count", type=int, default=150)
    for name in ("audit", "score"):
        s = sub.add_parser(name)
        s.add_argument("--config", required=True)
        s.add_argument("--output", required=True)
        if name == "score":
            s.add_argument("--gold", required=True)
    a = p.parse_args()
    if a.command == "prepare":
        prepare(a.dataset, a.directory, a.encoder, a.count)
    elif a.command == "audit":
        result = audit(a.config, a.output)
        print(json.dumps({k: v for k, v in result.items() if k != "answer_source_overlap"}))
        if result["status"] != "PASS":
            raise SystemExit(1)
    else:
        print(json.dumps(evaluate(a.config, a.output, a.gold), indent=2))


if __name__ == "__main__":
    main()
