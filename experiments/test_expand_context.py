"""Authored STEP 3 operation smokes, not a latent-reasoning/answer benchmark."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.latent_context import ContextMemory, render
from experiments.latent_context_app import MemoryHost, mcp_server
from experiments.memory.bridge_groups import compatible_bindings
from experiments.memory.evidence_map import Edge, EvidenceMap
from experiments.memory.latent_ppr import PPRLimits
from experiments.memory.ppr_context import PPRNavigation
from experiments.test_latent_mcp import put, sha


@pytest.fixture
def bridge_corpus(tmp_path, monkeypatch, request):
    monkeypatch.setenv("LSR_CPU_RAM_FLOOR_GIB", "10")
    source, index = tmp_path / "sources", tmp_path / "index"
    source.mkdir()
    index.mkdir()
    (index / "leaves").mkdir()
    texts = ["Departure Ledger\nBeacon registry beacon registry beacon registry sends vessels to Nebula Port.\n"]
    texts += [f"Distractor {i}\nBeacon registry catalog unrelated entry {i}.\n" for i in range(1, 11)]
    texts += ["Nebula Port\nHarbor authority is Mira.\n"]
    markdown = getattr(request, "param", False)
    if markdown:
        texts[0] = texts[0].replace("Nebula Port.", "[Nebula Port](11.md).")
    rows, files = [], []
    for i, text in enumerate(texts):
        raw = text.encode()
        path = source / (f"{i:02}.md" if markdown else f"{i:02}.txt")
        path.write_bytes(raw)
        (index / "leaves" / (sha(raw) + ".txt")).write_bytes(raw)
        files.append(dict(path=str(path), sha256=sha(raw), bytes=len(raw), extraction="utf8_text"))
        rows.append(dict(source=str(path), source_sha256=sha(raw), leaf_sha256=sha(raw),
                         page=None, start_byte=0, end_byte=len(raw), span_sha256=sha(raw), text=text))
    put(index / "rows.json", rows)
    np.save(index / "vectors.npy", np.ones((len(rows), 3), dtype=np.float32))
    put(index / "manifest.json", dict(files=files, allowed_paths=[f["path"] for f in files],
        source_roots=[str(source)], encoder="unused-bm25",
        rows_sha256=sha((index / "rows.json").read_bytes()),
        vector_sha256=sha((index / "vectors.npy").read_bytes())))
    config = tmp_path / "config.json"
    put(config, dict(index_dir=str(index), source_root=str(source),
        manifest_sha256=sha((index / "manifest.json").read_bytes()),
        allowed_paths=[f["path"] for f in files], routing="bm25", max_results=8,
        max_output_bytes=28000))
    memory = ContextMemory(config)
    memory._evidence_map = EvidenceMap(memory.tool)
    return memory, source, index


def spans(pack):
    return [s for g in pack["evidence_groups"] for s in g["spans"]]


def bridges(pack):
    return [g for g in pack["evidence_groups"] if g.get("kind") == "PROVISIONAL_SOURCE_BRIDGE"]


def literal_edge(memory):
    return next(e for e in memory._evidence_map.outgoing[memory.tool.corpus_id + ":S0"]
                if e.type == "scoped_association")


def set_edge(memory, edge):
    graph = memory._evidence_map.outgoing
    node = memory.tool.corpus_id + ":S0"
    graph[node] = tuple(e for e in graph[node] if e.type != "scoped_association") + ((edge,) if edge else ())


def invoke(memory, **kwargs):
    return memory.context("Beacon registry", strategy="expand",
                          budget={"max_spans": 4}, **kwargs)


def test_bridge_disable_title_checked_swap_restore_changes_delivered_support(bridge_corpus):
    memory, _, _ = bridge_corpus
    edge = literal_edge(memory)
    normal = invoke(memory)
    assert {"00.txt", "11.txt"} <= {s["file"] for s in spans(normal)}
    assert len(bridges(normal)) == 1
    group = bridges(normal)[0]
    assert set(group["parents"]) <= {s["id"] for s in spans(normal)}
    assert group["witness"]["text"] == "Nebula Port" and group["witness"]["byte_verified"]
    assert group["semantic_support"] == "READER_MUST_ASSESS" and group["join_status"] == "not computed"
    set_edge(memory, None)
    disabled = invoke(memory)
    assert "11.txt" not in {s["file"] for s in spans(disabled)} and not bridges(disabled)
    wrong = memory.tool.corpus_id + ":S10"
    binding = tuple((k, memory._evidence_map.spans[wrong].source_version if k == "target_version" else v)
                    for k, v in edge.binding_constraints)
    set_edge(memory, replace(edge, to_id=wrong, binding_constraints=binding))
    swapped = invoke(memory)
    assert "11.txt" not in {s["file"] for s in spans(swapped)} and not bridges(swapped)
    assert any("binding" in u["reason"] for u in swapped["unresolved"])
    set_edge(memory, edge)
    restored = invoke(memory)
    assert [s["id"] for s in spans(restored)] == [s["id"] for s in spans(normal)]
    assert bridges(restored)[0]["witness"] == group["witness"]
    assert restored["costs"]["expand"]["hops"] == 1
    assert restored["costs"]["expand"]["access"]["edge_reads"] > 0
    assert restored["costs"]["expand"]["access"]["exact_vector_rows_read"] == 0
    assert "PROVISIONAL" in render(restored)


def test_incompatible_identity_join_is_rejected_not_semantically_accepted(bridge_corpus):
    memory, _, _ = bridge_corpus
    edge = literal_edge(memory)
    with pytest.raises(ValueError, match="incompatible"):
        compatible_bindings((("entity", "A"),), (("entity", "B"),))
    set_edge(memory, replace(edge, binding_constraints=edge.binding_constraints +
                            (("target_version", "wrong-version"),)))
    pack = invoke(memory)
    assert not bridges(pack)
    assert any("incompatible" in item["reason"] for item in pack["unresolved"])
    assert pack["analysis_status"]["proposed_joins"] == pack["analysis_status"]["conflicts"] == "not computed"
    assert pack["proposed_joins"] == pack["conflicts"] == []


def test_all_rejected_bridges_leave_room_for_verified_seeds(bridge_corpus, monkeypatch):
    memory, _, _ = bridge_corpus
    import experiments.memory.expand_context as module
    from experiments.context_candidates import prepare_units
    import experiments.context_pack as pack_module
    method = module.ExpandNavigation.propose

    def many_bridges(nav, question, progress=None):
        candidates, trace, duplicates = method(nav, question, progress)
        seed = next(p for p in candidates if not p.get("bridge"))
        edge = literal_edge(memory)
        # Distinct rejected proposals exceed the inline metadata budget.
        rejected = [nav.candidate(edge.from_id, parents=[edge.from_id, edge.to_id],
                    bridge=replace(edge, witness_span=replace(edge.witness_span,
                                   start=i, end=i + 1))) for i in range(40)]
        return rejected + [seed], trace, duplicates

    def reject_bridge(mem, nav, proposal, question, edge=None):
        if edge:
            raise ValueError("missing bridge parent or witness")
        return prepare_units(mem, nav, proposal, question, edge)

    monkeypatch.setattr(module.ExpandNavigation, "propose", many_bridges)
    monkeypatch.setattr(pack_module, "prepare_units", reject_bridge)
    pack = invoke(memory)
    assert spans(pack) and spans(pack)[0]["file"] == "00.txt"
    assert not bridges(pack)
    assert pack["unresolved"] and pack["omission_ledger"]["total"] > 0
    recovered = memory.context(**pack["omission_ledger"]["recovery"]["arguments"])
    assert recovered["unresolved"]  # All rejected proposals remain recoverable.
    assert pack["omission_ledger"]["total"] == 40
    identities = [json.dumps(u["bridge_identity"], sort_keys=True) for u in pack["unresolved"]]
    assert len(set(identities)) == len(identities)
    assert pack["coverage"]["completeness"] == "UNKNOWN"
    assert memory.meter.fits(pack, pack["costs"]["output_limits"])


def test_unaffordable_bridge_still_delivers_seed(bridge_corpus):
    memory, _, _ = bridge_corpus
    pack = memory.context("Beacon registry", strategy="expand", budget={"max_spans": 1})
    assert spans(pack) and spans(pack)[0]["file"] == "00.txt"
    assert not bridges(pack)
    assert any(u.get("parents") for u in pack["unresolved"])


def test_access_stopped_bridge_delivers_cached_seed_and_retains_pair(bridge_corpus, monkeypatch):
    memory, _, _ = bridge_corpus
    original = PPRNavigation.source_units
    reads = PPRNavigation.source_spans

    def stop_second(nav, node, question, **kwargs):
        if node == literal_edge(memory).to_id:
            nav.access.stop_reason = "navigation_time_cap"
            nav.access.clock = lambda: nav.access.started + nav.access.limits.seconds
            return None
        return original(nav, node, question, **kwargs)

    def no_reads_after_stop(nav, ids):
        return reads(nav, ids)

    monkeypatch.setattr(PPRNavigation, "source_units", stop_second)
    monkeypatch.setattr(PPRNavigation, "source_spans", no_reads_after_stop)
    pack = invoke(memory)
    assert spans(pack) and spans(pack)[0]["file"] == "00.txt"
    assert not bridges(pack) and pack["continuation"]
    state = memory.continuations.resolve(pack["continuation"]["arguments"]["continuation"],
                                         serialized_binding(pack))
    assert any(p[2] for p in state["pending"]["pending"])
    assert any(u.get("bridge_identity") for u in pack["unresolved"])
    assert pack["costs"]["expand"]["access"]["source_opens"] == 1


def test_oversized_seed_metadata_does_not_consume_affordable_bridge(bridge_corpus, monkeypatch):
    memory, _, _ = bridge_corpus
    import experiments.context_pack as module
    import experiments.memory.expand_context as expand
    original = module.prepare_units
    propose = expand.ExpandNavigation.propose
    node = memory.tool.corpus_id + ":S1"

    def oversized_first(nav, question, progress=None):
        candidates, trace, duplicates = propose(nav, question, progress)
        return [nav.candidate(node), *candidates], trace, duplicates

    def oversized(mem, nav, proposal, question, edge=None):
        units = original(mem, nav, proposal, question, edge)
        if proposal == node and not edge:
            units = [[dict(s, text=s["text"] * 10000) for s in units[0]]]
        return units

    monkeypatch.setattr(expand.ExpandNavigation, "propose", oversized_first)
    monkeypatch.setattr(module, "prepare_units", oversized)
    pack = invoke(memory)
    assert pack["omission_ledger"]["total"] > 0
    for _ in range(8):
        if bridges(pack):
            break
        assert pack["continuation"]
        pack = memory.context(**pack["continuation"]["arguments"])
    assert bridges(pack) and {"00.txt", "11.txt"} <= {s["file"] for s in spans(pack)}


def test_corrupt_span_refused_and_attempts_charged(bridge_corpus):
    memory, _, index = bridge_corpus
    row = memory.tool.memory.rows[11]
    leaf = index / "leaves" / (row["leaf_sha256"] + ".txt")
    leaf.write_bytes(b"!" * leaf.stat().st_size)
    with pytest.raises(ValueError, match="changed"):
        invoke(memory)
    assert memory.expand_last_attempt["verification_hash_checks"] > 0
    assert memory.expand_last_attempt["leaf_opens"] > 0


def test_missing_premise_remains_unresolved(bridge_corpus):
    memory, _, _ = bridge_corpus
    pack = memory.context("Beacon registry harbor authority and the absent insurance policy",
                          strategy="expand")
    assert pack["unresolved_obligations"][0]["status"] == "READER_MUST_ASSESS"
    assert all(g["join_status"] == "not computed" for g in bridges(pack))
    assert pack["coverage"]["completeness"] == pack["coverage"]["not_in_memory"] == "UNKNOWN"
    assert not any(s["semantic_support"] != "READER_MUST_ASSESS" for s in spans(pack))


def test_atomic_pair_survives_continuation_and_never_splits(bridge_corpus, monkeypatch):
    memory, _, _ = bridge_corpus
    first = memory.context("Beacon registry", strategy="expand", budget={"max_spans": 1})
    assert not bridges(first)
    omitted = next(u for u in first["unresolved"] if u.get("parents"))
    assert "max_spans" in omitted["blocked_dimensions"] and len(omitted["parents"]) == 2
    assert first["omission_ledger"]["total"] > 0
    node = memory.tool.corpus_id + ":S0"
    target = memory.tool.corpus_id + ":S11"
    edge = literal_edge(memory)
    # An unrelated seed first defers the pair under the two-span limit.
    import experiments.memory.expand_context as module
    method = module.ExpandNavigation.propose
    def seed_first(nav, question, progress=None):
        candidates, trace, duplicates = method(nav, question, progress)
        return [nav.candidate(memory.tool.corpus_id + ":S1"), *candidates], trace, duplicates
    monkeypatch.setattr(module.ExpandNavigation, "propose", seed_first)
    page = memory.context("Beacon registry", strategy="expand", budget={"max_spans": 2})
    assert page["continuation"] and not bridges(page)
    args = page["continuation"]["arguments"]
    monkeypatch.setattr(module, "query_seeds", lambda *a, **k: pytest.fail("resume rerouted"))
    next_page = memory.context(**args)
    assert bridges(next_page)
    assert len(bridges(next_page)[0]["parents"]) == 2
    assert set(bridges(next_page)[0]["parents"]) <= {s["id"] for s in spans(next_page)}
    assert next_page["costs"]["query_encoder_calls"] == 0
    assert next_page["costs"]["expand"]["access"]["edge_reads"] == 0
    assert edge.from_id == node and edge.to_id == target


def test_duplicate_occurrences_keep_identity(bridge_corpus):
    memory, source, index = bridge_corpus
    rows = json.loads((index / "rows.json").read_text())
    manifest = json.loads((index / "manifest.json").read_text())
    copy = source / "12.txt"
    copy.write_bytes((source / "11.txt").read_bytes())
    rows.append(dict(rows[11], source=str(copy)))
    manifest["files"].append(dict(manifest["files"][11], path=str(copy)))
    manifest["allowed_paths"].append(str(copy))
    put(index / "rows.json", rows)
    np.save(index / "vectors.npy", np.ones((len(rows), 3), dtype=np.float32))
    manifest.update(rows_sha256=sha((index / "rows.json").read_bytes()),
                    vector_sha256=sha((index / "vectors.npy").read_bytes()))
    put(index / "manifest.json", manifest)
    config = json.loads(Path(memory.config).read_text(encoding="utf8"))
    config.update(allowed_paths=manifest["allowed_paths"],
                  manifest_sha256=sha((index / "manifest.json").read_bytes()))
    put(Path(memory.config), config)
    pack = ContextMemory(memory.config).context("Beacon registry", strategy="expand")
    copies = [s for s in spans(pack) if s["file"] in ("11.txt", "12.txt")]
    assert len(copies) == 2 and copies[0]["sha256"] == copies[1]["sha256"]
    assert copies[0]["id"] != copies[1]["id"] and copies[0]["file"] != copies[1]["file"]


def test_all_strategies_share_structural_unit_policy(bridge_corpus):
    from experiments.context_spans import source_units
    memory, _, _ = bridge_corpus
    node = memory.tool.corpus_id + ":S0"
    one_shot = source_units(memory.tool, node, "Beacon registry")
    ppr = PPRNavigation(memory).source_units(node, "Beacon registry")
    assert one_shot[0][0]["text"] == ppr[0][0]["text"]
    assert one_shot[0][0]["unit"] == ppr[0][0]["unit"]


def test_expansion_host_mcp_and_access_caps(bridge_corpus, tmp_path, monkeypatch):
    memory, source, _ = bridge_corpus
    registry = tmp_path / "registry.json"
    put(registry, dict(test=dict(config=memory.config, source_root=str(source), description="authored smoke")))
    host = MemoryHost(registry)
    pack = host.invoke(dict(method="context", corpus="test", question="Beacon registry", strategy="expand"))
    assert bridges(pack)
    server = mcp_server({}, "test")
    primary = next(t for t in asyncio.run(server.list_tools()) if t.name == "context")
    assert "expand" in primary.input_schema["properties"]["strategy"]["anyOf"][0]["enum"]
    import experiments.latent_context_app as app
    monkeypatch.setattr(app, "call", lambda endpoint, method, **args: host.invoke(dict(method=method, **args)))
    result = asyncio.run(server.call_tool("context", {"question": "Beacon registry", "strategy": "expand"}))
    assert not result.is_error and bridges(json.loads(result.content[0].text))
    from experiments.memory.expand_context import ExpandNavigation
    nav = ExpandNavigation(memory, limits=PPRLimits(expansions=0))
    nav.propose("Beacon registry")
    assert nav.access.costs["edge_reads"] == nav.access.costs["expansions"] == 0
    assert nav.access.stop_reason == "expansion_cap"


def test_geometry_is_never_a_bridge_or_support(bridge_corpus):
    memory, _, _ = bridge_corpus
    edge = literal_edge(memory)
    set_edge(memory, None)
    graph = memory._evidence_map.outgoing
    graph[edge.from_id] += (Edge(edge.from_id, edge.to_id, "geometric_neighbor", None, ()),)
    pack = invoke(memory)
    assert not bridges(pack) and "11.txt" not in {s["file"] for s in spans(pack)}
    assert pack["costs"]["expand"]["access"]["edge_reads"] > 0


def test_pair_exceeding_total_access_cap_is_not_an_endless_cursor(bridge_corpus, monkeypatch):
    memory, _, _ = bridge_corpus
    import experiments.memory.expand_context as module
    original = module.ExpandNavigation
    # Each parent fits individually; their sum cannot. Do not reopen the first
    # parent on each page while indefinitely deferring the second.
    size = 2 * len(memory.tool.memory.rows[0]["text"].encode())
    monkeypatch.setattr(module, "ExpandNavigation",
                        lambda m: original(m, limits=PPRLimits(verification_bytes=size)))
    pack = invoke(memory)
    assert any(u.get("parents") and u["reason"] == "atomic_source_exceeds_verification_budget"
               for u in pack["unresolved"])
    assert not bridges(pack)
    if pack["continuation"]:
        pending = memory.continuations.resolve(
            pack["continuation"]["arguments"]["continuation"],
            serialized_binding(pack))
        assert not any(item[2] for item in pending["pending"]["pending"])


def serialized_binding(pack):
    from experiments.latent_mcp import serialized
    return serialized([pack["question"], pack["corpus"], pack["snapshot"], pack["intent"],
                       pack["readings"], pack["strategy"]])


def test_source_opens_hashes_and_failed_work_match_independent_counters(bridge_corpus, monkeypatch):
    memory, source, index = bridge_corpus
    import experiments.memory.ppr_context as module
    opens, hashes = [], []
    original_open, original_digest = Path.open, module.digest
    def counted_open(path, *args, **kwargs):
        if path.parent in (source, index / "leaves"):
            opens.append(path.parent)
        return original_open(path, *args, **kwargs)
    def counted_digest(raw):
        hashes.append(len(raw))
        return original_digest(raw)
    monkeypatch.setattr(Path, "open", counted_open)
    monkeypatch.setattr(module, "digest", counted_digest)
    pack = invoke(memory)
    access = pack["costs"]["expand"]["access"]
    assert access["source_opens"] == opens.count(source)
    assert access["leaf_opens"] == opens.count(index / "leaves")
    assert access["verification_hash_checks"] == len(hashes)
    assert access["verification_hash_bytes"] == sum(hashes)
    assert access["verification_bytes_read"] == pack["costs"]["verification_bytes_read"]


def test_chunk_adjacency_inside_complete_leaf_does_not_emit_self_bridges(bridge_corpus):
    memory, _, index = bridge_corpus
    rows = json.loads((index / "rows.json").read_bytes())
    original = rows[0]
    raw = original["text"].encode()
    split = raw.index(b" sends")
    rows[0] = dict(original, text=raw[:split].decode(), end_byte=split,
                   span_sha256=sha(raw[:split]))
    rows.append(dict(original, text=raw[split:].decode(), start_byte=split,
                     span_sha256=sha(raw[split:])))
    put(index / "rows.json", rows)
    np.save(index / "vectors.npy", np.ones((len(rows), 3), dtype=np.float32))
    manifest = json.loads((index / "manifest.json").read_bytes())
    manifest.update(rows_sha256=sha((index / "rows.json").read_bytes()),
                    vector_sha256=sha((index / "vectors.npy").read_bytes()))
    put(index / "manifest.json", manifest)
    config = json.loads(Path(memory.config).read_bytes())
    config["manifest_sha256"] = sha((index / "manifest.json").read_bytes())
    put(Path(memory.config), config)
    pack = invoke(ContextMemory(memory.config))
    assert not bridges(pack)
    assert pack["coverage"]["returned_files"] == 4
    assert sum(s["file"] == "00.txt" for s in spans(pack)) == 1


@pytest.mark.parametrize("bridge_corpus", [True], indirect=True)
def test_markdown_reference_resolution_visits_only_charged_target_rows(bridge_corpus, monkeypatch):
    memory, _, _ = bridge_corpus
    backend = memory.tool.memory
    from experiments.context_costs import MeteredRows
    class Visits(MeteredRows):
        iterations = 0
        indexed = 0

        def __iter__(self):
            for row in super().__iter__():
                self.iterations += 1
                yield row

        def __getitem__(self, key):
            self.indexed += 1
            return super().__getitem__(key)

    backend.rows = Visits(backend.rows)
    original = backend._resolve
    resolutions = []
    def measured(*args, **kwargs):
        costs = kwargs["costs"]
        before = costs["source_row_records_read"]
        indexed, iterations = backend.rows.indexed, backend.rows.iterations
        result = original(*args, **kwargs)
        actual = backend.rows.indexed - indexed
        assert backend.rows.iterations == iterations  # Never iterate the corpus.
        assert costs["source_row_records_read"] - before == actual == 1
        resolutions.append(result)
        return result
    monkeypatch.setattr(backend, "_resolve", measured)
    pack = invoke(memory)
    assert {g["edge_type"] for g in bridges(pack)} >= {"explicit_reference", "scoped_association"}
    assert resolutions and all(r["resolution"] == "MANIFEST_RESOLVED_UNFETCHED" for r in resolutions)
    assert backend.rows.iterations == pack["costs"]["expand"]["access"]["lexical_row_scans"] == 12
    assert pack["costs"]["access"]["source_row_records_read"] == backend.rows.indexed + backend.rows.iterations
    assert pack["costs"]["access"]["row_iteration_visits"] == backend.rows.iterations
    assert pack["costs"]["expand"]["access"]["reference_path_metadata_checks"] > 0


@pytest.mark.parametrize("bridge_corpus", [True], indirect=True)
def test_reference_resolution_checks_target_identity_and_deadline(bridge_corpus):
    memory, _, _ = bridge_corpus
    backend = memory.tool.memory
    backend.rows[11] = dict(backend.rows[11], source_sha256="changed")
    assert backend._resolve("11.md", backend.rows[0]["source"])["resolution"] == "UNRESOLVED"
    assert backend._resolve("11.md", backend.rows[0]["source"], check=lambda: False)["reason"] == "navigation_time_cap"
