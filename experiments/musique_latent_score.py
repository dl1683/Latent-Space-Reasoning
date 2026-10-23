"""Shared scorer plus clustered intervals and all-attempt retrieval/reader costs."""
import argparse
from collections import Counter
from pathlib import Path

from experiments import musique_dev as shared
from experiments.musique_dev_eval import audit, evaluate
from experiments.retrieval_benchmark import bootstrap, source_clusters


def report(config_path, output, gold_path):
    config, output = shared.load(config_path), Path(output)
    # The reference exact metric and quote/span audit stay authoritative.
    summary = evaluate(config_path, output, gold_path)
    leakage = audit(config_path, output)
    if leakage["status"] != "PASS":
        raise ValueError("source/canary audit failed")
    scores = shared.load(output / "scores.json")
    generation = shared.read_rows(config["generation"])
    gold = {g["id"]: dict(g, support_indices=g["supporting"])
            for g in shared.read_rows(gold_path)}
    clusters = source_clusters(generation, gold)
    path = output / "predictions.jsonl"
    attempts = shared.read_rows(path) if path.exists() else []
    for entry in summary:
        values = [r for r in scores["per_question"]
                  if r["reader"] == entry["reader"] and r["arm"] == entry["arm"]
                  and r["status"] == "COMPLETED"]
        charged = [a for a in attempts if a["reader"] == entry["reader"]
                   and a["arm"] == entry["arm"]]
        for metric in ("em", "f1"):
            entry[metric + "_cluster_ci95"] = (
                bootstrap([r[metric] for r in values],
                          clusters=[clusters[r["id"]] for r in values])
                if len(values) == len(generation) else None)
        usage = [a.get("reader_usage", {}) for a in charged if a["calls"]]
        retrievals = [a.get("retrieval", {}) for a in charged]
        entry.update(
            prompt_eval_tokens=sum(u.get("prompt_tokens") or 0 for u in usage),
            output_tokens=sum(u.get("output_tokens") or 0 for u in usage),
            unknown_usage_attempts=sum(u.get("prompt_tokens") is None or
                                      u.get("output_tokens") is None for u in usage),
            statuses=dict(Counter(a["status"] for a in charged)),
            termination_rate=sum(u.get("done_reason") == "stop" for u in usage) / len(usage)
                if usage else None,
            capability_interpretation="DEV_ONLY" if (
                len(values) == len(generation) and usage and
                sum(u.get("done_reason") == "stop" for u in usage) / len(usage) >= .95)
                else "WITHHELD",
            complete_support=sum(r["complete_annotated_support"] for r in values),
            retrieval_bytes={name: sum(r.get(name, 0) for r in retrievals) for name in (
                "source_bytes_touched", "map_bytes_touched", "exact_vector_bytes_touched",
                "graph_edge_bytes_touched", "pointer_file_bytes_touched")},
            lexical_bytes_touched="UNKNOWN" if entry["arm"] in ("bm25", "hybrid") else 0)
    overhead = {}
    for reader in config["readers"]:
        calls_path = output / reader["id"] / "calls.jsonl"
        calls = shared.read_rows(calls_path) if calls_path.exists() else []
        associated = {a.get("reader_usage", {}).get("request_id") for a in attempts
                      if a["reader"] == reader["id"]}
        extra = [c for c in calls if c["request_id"] not in associated]
        overhead[reader["id"]] = dict(
            calls=len(extra),
            prompt_eval_tokens=sum((c.get("terminal") or {}).get("prompt_eval_count", 0)
                                   for c in extra),
            output_tokens=sum((c.get("terminal") or {}).get("eval_count", 0) for c in extra),
            scope="smoke and calls interrupted before prediction append")
    result = dict(summary=summary, paired=scores["paired"], overhead=overhead,
                  constants=scores["constants"], leakage_status=leakage["status"],
                  preparation=shared.load(Path(config["work"]) / "PREPARED.json"),
                  confidence_unit="connected annotated-source clusters, 4000 draws",
                  judge="programmatic shared MuSiQue normalized EM/token F1",
                  scope="provided-context development, not whole-corpus answer quality",
                  accounting="native evaluated tokens, not certified total logical context",
                  full_physical_io_and_lifetime_costs="UNKNOWN")
    shared.write(output / "LATENT_RESULTS.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gold", required=True)
    args = parser.parse_args()
    report(args.config, args.output, args.gold)
