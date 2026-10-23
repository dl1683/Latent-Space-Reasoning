"""Compressed-map adapter for the reference source-only MuSiQue harness.

Same reader, answer schema, prompts, budgets, roster and scorer as musique_dev.
All ranked arms receive the same literal reference graph. Packing reads every
provided paragraph. This is a retrieval SLICE, not latent-core reasoning.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments import musique_dev as shared
from experiments.memory import compressed_routing_map, latent_frontend, refined_routing_map
from experiments.retrieval_benchmark import generation_at, read, sha


class SharedFrontend:
    arms = ("bm25", "hybrid", "packed", "latent")

    def __init__(self, config):
        self.root = Path(config["work"])
        prepared_raw = (self.root / "PREPARED.json").read_bytes()
        prepared = read(self.root / "PREPARED.json")
        generation_raw = Path(config["generation"]).read_bytes()
        query_raw = (self.root / "queries.npy").read_bytes()
        if (sha(generation_raw) != prepared["generation_sha256"]
                or sha(query_raw) != prepared["query_sha256"]):
            raise ValueError("frontend generation/query binding changed")
        rows = generation_at(config["generation"])
        self.rows = {r["id"]: (n, r) for n, r in enumerate(rows)}
        self.queries = np.load(self.root / "queries.npy", allow_pickle=False)
        if (self.queries.dtype != np.float32 or self.queries.ndim != 2
                or len(self.queries) != len(rows) or not np.isfinite(self.queries).all()):
            raise ValueError("aligned finite FP32 query roster required")
        self.top_k, self.graph = config.get("top_k", 4), config.get("graph", True)
        if type(self.top_k) is not int or self.top_k < 1 or type(self.graph) is not bool:
            raise ValueError("invalid frontend budget or graph setting")
        manifests = [sha((self.root / f"{n:03d}" / "manifest.json").read_bytes())
                     for n in range(len(rows))]
        self.identity = dict(
            prepared_sha256=sha(prepared_raw), manifest_sha256=manifests,
            generation_sha256=sha(generation_raw), query_sha256=sha(query_raw),
            code_sha256={Path(m.__file__).name: sha(Path(m.__file__).read_bytes())
                         for m in (latent_frontend, refined_routing_map, compressed_routing_map)},
            adapter_sha256=sha(Path(__file__).read_bytes()),
            top_k=self.top_k, graph=self.graph)
        self.memories = {}

    def retrieve(self, row, arm):
        shared.validate_row(row)
        if arm not in self.arms or row["id"] not in self.rows:
            raise ValueError("unknown frontend arm or question")
        n, expected = self.rows[row["id"]]
        if row != expected:
            raise ValueError("query or task source differs from prepared allow-list")
        if n not in self.memories:
            memory = latent_frontend.LatentFrontend(self.root / f"{n:03d}")
            if memory.rows != row["paragraphs"]:
                raise ValueError("memory differs from task sources")
            self.memories[n] = memory
        memory = self.memories[n]
        result = memory.retrieve(
            row["question"], self.queries[n],
            arm="packing" if arm == "packed" else arm,
            k=len(memory.rows) if arm == "packed" else min(self.top_k, len(memory.rows)),
            graph=self.graph)
        evidence = shared.spans(row["paragraphs"], result["selected_indices"])
        shared.verify_spans(row, evidence)
        return dict(evidence=evidence, costs={k: v for k, v in result.items()
                                              if k != "evidence"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reader")
    parser.add_argument("--arm", choices=SharedFrontend.arms)
    args = parser.parse_args()
    config = read(args.config)
    shared.run(args.config, args.output, resume=args.resume,
               only_reader=args.reader, only_arm=args.arm,
               retriever=SharedFrontend(config))


if __name__ == "__main__":
    main()
