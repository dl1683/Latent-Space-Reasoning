"""Deterministic one-hop literal expansion, a conventional evidence-access SLICE."""
import time

from experiments.memory.latent_ppr import query_seeds
from experiments.memory.ppr_context import PPRNavigation
from experiments.context_costs import visit_edge, stage


class ExpandNavigation(PPRNavigation):
    def __init__(self, memory, *, limits=None):
        super().__init__(memory, limits=limits)
        memory.expand_last_attempt = self.access.costs
        self.seed_ids = []

    @stage("navigation")
    def propose(self, question, progress=None):
        seeds = query_seeds(self.map, question, self.access, hybrid=True)
        self.memory.encoder_calls += self.access.costs["query_encoder_calls"]
        self.seed_ids = list(seeds)
        candidates = []
        # Only the first four hybrid seeds expand. Destinations never expand.
        for node in self.seed_ids[:4]:
            for edge in self.map.outgoing.get(node, ()):
                if not self.access.reserve("expansions", 1, self.access.limits.expansions,
                                           "expansion_cap"):
                    break
                self.access.costs["edge_reads"] += 1
                self.access.costs["edge_record_bytes_read"] += len(repr(edge).encode())
                visit_edge(edge)
                if (edge.type not in ("explicit_reference", "scoped_association", "adjacency")
                        or edge.to_id not in self.map.spans or edge.to_id == node):
                    continue
                candidates.append(self.candidate(node, parents=[node, edge.to_id], bridge=edge))
            if self.access.stop_reason:
                break
        candidates.extend(self.candidate(node) for node in self.seed_ids)
        self.result = dict(stop_reason=self.access.stop_reason or "one_hop_exhausted",
                           residual_mass=None, touched_nodes=len(self.seed_ids), restart=None)
        trace = [dict(step=1, operation="source_bound_expand",
            why="Hybrid seeds, ONE typed source-bound hop; literal bindings checked at delivery, not entailment.",
            spans=self.seed_ids, navigation_stop=self.result["stop_reason"])]
        if progress:
            progress(trace[0])
        return candidates, trace, 0

    def candidate(self, node, **extra):
        row = self.tool.memory.rows[self.map.spans[node].vector_row]
        return dict(id=node, source_path=row["source"], source_sha256=row["source_sha256"], **extra)

    def metadata(self):
        result = super().metadata()
        result.update(internal_ablation=False, restart=None, hops=1,
                      seed_count=len(self.seed_ids), expanding_seeds=min(4, len(self.seed_ids)),
                      seed_ranking="existing hybrid RRF with capped dense refinement",
                      navigation_seconds=time.perf_counter() - self.access.started)
        return result
