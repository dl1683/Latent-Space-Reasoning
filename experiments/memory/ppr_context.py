"""PPR proposal/verification hooks for the existing coherent context packer."""
import time

from experiments.dogfood import digest, plain_file
from experiments.context_spans import units_from_leaves, origin_id
from experiments.memory.evidence_map import EvidenceMap
from experiments.memory.latent_ppr import AccessBudget, local_ppr, query_seeds
from experiments.context_costs import CURRENT, accounting, charge, stage


class PPRNavigation:
    def __init__(self, memory, *, limits=None):
        self.memory, self.tool = memory, memory.tool
        cold = not hasattr(memory, "_evidence_map")
        if cold:
            with accounting() as setup_access:
                if memory.context_last_attempt is not None:
                    memory.context_last_attempt["setup_this_call"]["adapter"] = setup_access
                memory._evidence_map = EvidenceMap(self.tool)
            memory._evidence_map.setup_access = setup_access
        self.map = memory._evidence_map
        self.setup_this_call = dict(self.map.setup_costs) if cold else None
        self.access = AccessBudget(limits)
        # Preserve live attempted-work charges even when byte verification raises.
        memory.ppr_last_attempt = self.access.costs
        self.result = None
        self.verified_source_opens = 0
        self.verified_leaf_opens = 0
        self.refused_reads = 0
        self._units_cache = {}
        # Request-local verified bytes, shared by anchored and standalone views.
        # Never carry them across a continuation or a fresh navigation attempt.
        self._leaves_cache = {}

    @stage("navigation")
    def propose(self, question, progress=None):
        seeds = query_seeds(self.map, question, self.access)
        self.memory.encoder_calls += self.access.costs["query_encoder_calls"]
        if seeds:
            self.result = local_ppr(self.map.outgoing, seeds, self.access)
            # Rank only visited/seed chunk nodes, never scan every map node.
            scores = self.result["scores"]
            nodes = {node for node in scores if node in self.map.spans} | set(seeds)
            order = sorted(nodes, key=lambda node: (-scores.get(node, 0), -seeds.get(node, 0), node))
        else:
            order = []
            self.result = dict(stop_reason=self.access.stop_reason or "no_seeds",
                               residual_mass=1.0, touched_nodes=0, restart=0.15)
        candidates = []
        for node in order:
            row = self.tool.memory.rows[self.map.spans[node].vector_row]
            candidates.append(dict(id=node, source_path=row["source"],
                                   source_sha256=row["source_sha256"]))
        trace = [dict(step=1, operation="local_ppr",
            why="INTERNAL ablation: one charged seed route, then typed residual pushes; proposals only.",
            spans=list(seeds), seed_count=len(seeds),
            navigation_stop=self.access.stop_reason or self.result["stop_reason"])]
        if progress:
            progress(trace[0])
        return candidates, trace, 0

    def _read(self, path, kind):
        if not self.access.check():
            return None
        path = plain_file(path)
        size = path.stat().st_size
        remaining = self.access.limits.verification_bytes - self.access.costs["verification_bytes_read"]
        if size > remaining:
            self.access.stop_reason = "verification_byte_cap"
            self.refused_reads += 1
            return None
        # Bound the actual read too, including concurrent source growth.
        kind_name = "source" if kind == "source_version_bytes_read" else "leaf"
        charge(**{kind_name + "_open_attempts": 1})
        with path.open("rb") as stream:
            charge(**{kind_name + "_opens": 1})
            self.access.costs["source_opens" if kind == "source_version_bytes_read" else "leaf_opens"] += 1
            raw = stream.read(min(size + 1, remaining))
        self.access.costs["verification_bytes_read"] += len(raw)
        charge(verification_bytes_read=len(raw), **{kind_name + "_bytes_read": len(raw)})
        self.tool.memory.reference_costs[kind] += len(raw)
        if len(raw) != size or path.stat().st_size != size:
            raise ValueError("source size changed during verification")
        if not self.access.check():
            return None
        return raw

    @stage("verification_reads")
    def source_spans(self, leaf_ids):
        """Atomic complete source: direct verified reads, no hidden lexical scans."""
        if not self.can_read_source():
            return None
        tool, memory = self.tool, self.tool.memory
        rows = [memory.rows[tool._index(i)] for i in leaf_ids]
        source = rows[0]["source"]
        if any(r["source"] != source or r["source_sha256"] != rows[0]["source_sha256"] for r in rows):
            raise ValueError("mixed source/version in atomic pack unit")
        paths = [memory.directory / "leaves" / (r["leaf_sha256"] + ".txt") for r in rows]
        needed = plain_file(source).stat().st_size + sum(plain_file(p).stat().st_size for p in paths)
        if needed > self.access.limits.verification_bytes:
            self.access.stop_reason = "atomic_source_exceeds_verification_budget"
            self.refused_reads += 1
            return None
        if needed + self.access.costs["verification_bytes_read"] > self.access.limits.verification_bytes:
            self.access.stop_reason = "verification_byte_cap"
            self.refused_reads += 1
            return None
        raw = self._read(source, "source_version_bytes_read")
        if raw is None:
            return None
        self.verified_source_opens += 1
        if not self._hash_matches(raw, rows[0]["source_sha256"]):
            raise ValueError("changed source version")
        passages = []
        for span_id, row, path in zip(leaf_ids, rows, paths):
            raw = self._read(path, "leaf_bytes_read")
            if raw is None:
                return None
            self.verified_leaf_opens += 1
            a, b = row["start_byte"], row["end_byte"]
            if (not self._hash_matches(raw, row["leaf_sha256"])
                    or not self._hash_matches(raw[a:b], row["span_sha256"])):
                raise ValueError("changed source leaf/span")
            projected = tool._project(dict(pointer=row, source_id=memory.source_ids[source],
                                          handle=span_id.split(":")[-1], text=raw.decode("utf-8")))
            projected.update(id=tool.corpus_id + ":L" + span_id.rsplit(":S", 1)[1],
                origin_id=span_id, start_byte=0, end_byte=len(raw), sha256=row["leaf_sha256"],
                unit="COMPLETE_EXTRACTED_LEAF", navigation=None)
            passages.append(projected)
        return passages

    def source_leaf_ids(self, span_id):
        row = self.tool.memory.rows[self.tool._index(span_id)]
        ids = self.map.source_rows[row["source"]]
        self.access.costs["source_row_records_read"] += len(ids)
        selected = {}
        for i in ids:
            other = self.tool.memory.rows[i]
            selected.setdefault((other["page"], other["leaf_sha256"]), self.tool.corpus_id + f":S{i}")
        self.access.costs["leaf_size_stats"] += len(selected)
        return list(selected.values())

    def _hash_matches(self, raw, expected):
        self.access.costs["verification_hash_bytes"] += len(raw)
        self.access.costs["verification_hash_checks"] += 1
        return digest(raw) == expected

    def can_read_source(self):
        return (self.access.stop_reason not in ("navigation_time_cap", "verification_byte_cap")
                and self.access.check())

    def has_verified_source(self, span_id):
        row = self.tool.memory.rows[self.tool._index(span_id)]
        return (row["source"], row["source_sha256"]) in self._leaves_cache

    def source_units(self, span_id, question, *, anchors=()):
        row = self.tool.memory.rows[self.tool._index(span_id)]
        source_key = (row["source"], row["source_sha256"])
        key = (source_key, question, anchors)
        if key in self._units_cache:
            return self._units_cache[key]
        if source_key not in self._leaves_cache:
            if not self.can_read_source():
                return None
            leaves = self.source_spans(self.source_leaf_ids(span_id))
            if leaves is None:
                return None
            self._leaves_cache[source_key] = leaves
        units = units_from_leaves(self.tool, span_id, question, self._leaves_cache[source_key],
                                 anchors=anchors)
        for unit in units:
            for span in unit:
                if ":U" in span["id"]:
                    self.access.costs["verification_hash_bytes"] += len(span["text"].encode())
                    self.access.costs["verification_hash_checks"] += 1
        self._units_cache[key] = units
        return units

    def source_read_size(self, span_id):
        ids = self.source_leaf_ids(span_id)
        rows = [self.tool.memory.rows[self.tool._index(i)] for i in ids]
        return (plain_file(rows[0]["source"]).stat().st_size + sum(
            plain_file(self.tool.memory.directory / "leaves" / (r["leaf_sha256"] + ".txt")).stat().st_size
            for r in rows))

    @stage("verification_reads")
    def reopen_units(self, ids):
        """Reverify retained units using the same structural boundaries on every arm."""
        from experiments.context_spans import section_ranges
        result = []
        by_source = {}
        for span_id in ids:
            original = origin_id(self.tool, span_id)
            row = self.tool.memory.rows[self.tool._index(original)]
            key = row["source"]
            if key not in by_source:
                source_key = (key, row["source_sha256"])
                if source_key in self._leaves_cache:
                    by_source[key] = self._leaves_cache[source_key]
                else:
                    if not self.can_read_source():
                        return None
                    by_source[key] = self.source_spans(self.source_leaf_ids(original))
                    if by_source[key] is not None:
                        self._leaves_cache[source_key] = by_source[key]
            leaves = by_source[key]
            if leaves is None:
                return None
            leaf = next(p for p in leaves if p["leaf_sha256"] == row["leaf_sha256"]
                        and p["page"] == row["page"])
            if ":U" in span_id:
                _, a, b = span_id.rsplit(":U", 1)[1].split("_")
                a, b = int(a), int(b)
                boundary = next((r for r in section_ranges(leaf["text"]) if r[:2] == (a, b)), None)
                if boundary is None:
                    raise ValueError("unknown structural unit boundary")
                raw = leaf["text"].encode()[a:b]
                leaf = dict(leaf, id=span_id, origin_id=original, start_byte=a, end_byte=b,
                            text=raw.decode(), sha256=digest(raw), unit=boundary[2],
                            section_title=boundary[3])
                self.access.costs["verification_hash_bytes"] += len(raw)
                self.access.costs["verification_hash_checks"] += 1
            result.append(leaf)
        return result

    def metadata(self):
        result = self.result or dict(stop_reason="retained_proposals_only", residual_mass=None,
                                     touched_nodes=0, restart=0.15)
        return dict(internal_ablation=True, restart=0.15,
                    stop_reason=self.access.stop_reason or result["stop_reason"],
                    residual_mass=result["residual_mass"], touched_nodes=result["touched_nodes"],
                    access=dict(self.access.costs, **(CURRENT.get() or {})),
                    navigation_seconds=time.perf_counter() - self.access.started,
                    limits=vars(self.access.limits), setup_this_call=self.setup_this_call,
                    retained_setup=self.map.setup_costs,
                    verified_source_opens=self.verified_source_opens,
                    verified_leaf_opens=self.verified_leaf_opens, refused_reads=self.refused_reads,
                    edge_types=self.map.edge_counts, physical_io="UNKNOWN",
                    seed_scan_complete=(self.access.costs["map_vector_rows_read"] == len(self.map.bank)
                                        if self.tool.memory.encoder is not None else None),
                    seed_ranking="approximate capped refinement; not exact top-k",
                    time_enforcement="cooperative checks around encoder/read and <=128-row numeric blocks")

    def cursor_state(self, remaining):
        return dict(pending=remaining)

    def restore(self, state):
        self.result = dict(stop_reason="retained_proposals_only", residual_mass=None,
                           touched_nodes=0, restart=0.15)
        return list(state["pending"])
