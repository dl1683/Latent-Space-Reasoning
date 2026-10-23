"""Capped query-seeded local PPR, an INTERNAL conventional graph ablation.

One charged compressed seed scan is allowed; residual pushes never scan vectors.
Restart is 0.15. Types receive fixed weights, then each type's outgoing degree
normalizes its weight. Dangling mass stays at its node (a self-loop).
"""
from collections import defaultdict, deque
from dataclasses import dataclass
import math
import time

import numpy as np

from experiments.dogfood import encode, ram_guard
from experiments.memory.refined_routing_map import RefinedMap, feature_scores, rank
from experiments.context_costs import charge


TYPE_WEIGHTS = dict(containment=0.5, adjacency=1.0, explicit_reference=2.0,
                    scoped_association=0.5, revision=0.25, geometric_neighbor=0.25)


@dataclass(frozen=True)
class PPRLimits:
    expansions: int = 2048
    exact_vectors: int = 512
    map_bytes: int = 64 * 2**20
    verification_bytes: int = 64 * 2**20
    seconds: float = 5.0

    def __post_init__(self):
        for key, maximum in (("expansions", 2048), ("exact_vectors", 512),
                             ("map_bytes", 64 * 2**20),
                             ("verification_bytes", 64 * 2**20)):
            value = getattr(self, key)
            if type(value) is not int or not 0 <= value <= maximum:
                raise ValueError("invalid PPR " + key + " cap")
        if (isinstance(self.seconds, bool) or not isinstance(self.seconds, (int, float))
                or not math.isfinite(self.seconds) or not 0 < self.seconds <= 5):
            raise ValueError("PPR seconds must be in (0,5]")


class AccessBudget:
    def __init__(self, limits=None, *, clock=time.perf_counter):
        self.limits = limits or PPRLimits()
        self.clock, self.started = clock, clock()
        self.costs = dict(map_scans=0, map_vector_rows_read=0, map_bytes_read=0,
                          exact_vector_rows_read=0, exact_vector_bytes_read=0,
                          expansions=0, edge_reads=0, node_pushes=0,
                          edge_record_bytes_read=0,
                          query_encoder_calls=0, query_encoder_seconds=0.0,
                          lexical_row_scans=0, verification_bytes_read=0,
                          lexical_text_bytes_read=0,
                          verification_hash_bytes=0, verification_hash_checks=0,
                          source_opens=0, leaf_opens=0,
                          source_row_records_read=0, leaf_size_stats=0,
                          reference_path_metadata_checks=0)
        self.stop_reason = None

    def check(self):
        ram_guard()
        if self.clock() - self.started >= self.limits.seconds:
            self.stop_reason = "navigation_time_cap"
            return False
        return True

    def reserve(self, key, amount, maximum, reason):
        if not self.check():
            return False
        if self.costs[key] + amount > maximum:
            self.stop_reason = reason
            return False
        self.costs[key] += amount
        return True


def local_ppr(graph, seeds, access, *, tolerance=1e-5):
    """Sparse residual push. No full-graph iteration or truncated hub-degree walk.

    Return unpushed residual explicitly. Scores are a lower approximation, not
    an exact stationary solve or a semantic-completeness stopping criterion.
    """
    if (not seeds or any(not math.isfinite(v) or v < 0 for v in seeds.values())
            or sum(seeds.values()) <= 0 or not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("nonnegative finite seeds and positive tolerance required")
    total = sum(seeds.values())
    residual = defaultdict(float, {k: v / total for k, v in seeds.items()})
    scores = defaultdict(float)
    queue, queued, touched = deque(seeds), set(seeds), set()
    stop = "residual_tolerance"
    while queue:
        if not access.check():
            stop = access.stop_reason
            break
        node = queue.popleft()
        queued.remove(node)
        mass = residual[node]
        if mass <= tolerance:
            continue
        edges = graph.get(node, ())
        # Reading the length is metadata. Never inspect an unaffordable prefix
        # then silently normalize against a smaller degree.
        if not access.reserve("expansions", 1 + len(edges),
                              access.limits.expansions, "expansion_cap"):
            stop = access.stop_reason
            break
        access.costs["edge_reads"] += len(edges)
        access.costs["edge_record_bytes_read"] += sum(len(repr(e).encode("utf-8")) for e in edges)
        charge(edge_reads=len(edges),
               edge_record_bytes_read=sum(len(repr(e).encode("utf-8")) for e in edges))
        access.costs["node_pushes"] += 1
        touched.add(node)
        residual[node] = 0.0
        if not edges:
            scores[node] += mass
            continue
        groups = defaultdict(list)
        for edge in edges:
            groups[edge.type].append(edge.to_id)
        weight = sum(TYPE_WEIGHTS[k] for k in groups)
        scores[node] += 0.15 * mass
        for kind, targets in groups.items():
            share = 0.85 * mass * TYPE_WEIGHTS[kind] / (weight * len(targets))
            for target in targets:
                residual[target] += share
                if residual[target] > tolerance and target not in queued:
                    queue.append(target)
                    queued.add(target)
    return dict(scores=dict(scores), residual=dict(residual), stop_reason=stop,
                residual_mass=sum(residual.values()), touched_nodes=len(touched),
                restart=0.15, semantic_absence="UNKNOWN")


def query_seeds(evidence_map, question, access, *, seed_count=8, hybrid=False):
    """One compressed scan in bounded blocks, <=512 selective exact reads.

    Capping refinement makes this approximate routing, not RefinedMap's exact
    top-k certificate. A BM25-only index uses its retained lexical seed scorer,
    with its full row scan disclosed, and reads no vectors.
    """
    memory = evidence_map.tool.memory
    if not access.check():
        return {}
    if memory.encoder is None:
        access.costs["lexical_row_scans"] += len(memory.rows)
        access.costs["lexical_text_bytes_read"] += sum(len(r["text"].encode()) for r in memory.rows)
        charge(lexical_row_scans=len(memory.rows), lexical_text_bytes_read=memory.lexical_text_bytes)
        values = memory.bm25.score(question)
        if not access.check():
            return {}
        selected = rank(values)[:seed_count]
        return {evidence_map.tool.corpus_id + f":S{int(i)}": float(max(values[i], 0) + 1e-6)
                for i in selected}
    access.costs["query_encoder_calls"] += 1
    started = time.perf_counter()
    query = encode(memory.encoder, [question])[0]
    access.costs["query_encoder_seconds"] += time.perf_counter() - started
    if not access.check():
        return {}
    packed = evidence_map.routing
    scores = []
    access.costs["map_scans"] += 1
    charge(map_scans=1)
    for start in range(0, len(packed.codes), 128):
        end = min(start + 128, len(packed.codes))
        size = packed.codes[start:end].nbytes + packed.scales[start:end].nbytes
        if not access.reserve("map_bytes_read", int(size), access.limits.map_bytes, "map_byte_cap"):
            break
        access.costs["map_vector_rows_read"] += end - start
        charge(map_vector_rows_read=end - start, map_bytes_read=int(size))
        block = RefinedMap(packed.codes[start:end], packed.scales[start:end],
                           packed.errors[start:end], packed.dimensions, packed.bits)
        scores.extend(feature_scores(block.restore(), query).tolist())
    if not scores or not access.check():
        return {}
    proposed = rank(scores)[:access.limits.exact_vectors]
    if not len(proposed):
        access.stop_reason = "exact_vector_cap"
        return {}
    # A single indexed FP32 gather. Count every row, including duplicate uses.
    access.costs["exact_vector_rows_read"] += len(proposed)
    exact = evidence_map.bank[proposed]
    access.costs["exact_vector_bytes_read"] += int(exact.nbytes)
    charge(exact_vector_rows_read=len(proposed), exact_vector_bytes_read=int(exact.nbytes))
    refined = feature_scores(exact, query)
    if hybrid:
        # The existing hybrid's reciprocal-rank fusion, with capped dense refinement.
        from experiments.memory.compressed_routing_map import reciprocal_rank_fusion
        access.costs["lexical_row_scans"] += len(memory.rows)
        access.costs["lexical_text_bytes_read"] += sum(len(r["text"].encode()) for r in memory.rows)
        charge(lexical_row_scans=len(memory.rows), lexical_text_bytes_read=memory.lexical_text_bytes)
        lexical = memory.bm25.score(question)
        if not access.check():
            return {}
        # Compressed scores give unrefined destinations a deterministic proposal rank.
        dense_order = list(map(int, proposed[rank(refined)]))
        refined_ids = set(map(int, proposed))
        dense_order += [int(i) for i in rank(scores) if i not in refined_ids]
        proxy = np.empty(len(memory.rows))
        proxy.fill(-np.inf)
        for position, i in enumerate(dense_order):
            proxy[i] = -position
        values = reciprocal_rank_fusion(proxy, lexical) if np.any(lexical) else proxy
        chosen = rank(values)[:seed_count]
        return {evidence_map.tool.corpus_id + f":S{int(i)}": float(seed_count - j)
                for j, i in enumerate(chosen)}
    chosen = rank(refined)[:seed_count]
    best = float(np.max(refined))
    return {evidence_map.tool.corpus_id + f":S{int(proposed[j])}":
            float(np.exp(8 * (refined[j] - best))) for j in chosen}
