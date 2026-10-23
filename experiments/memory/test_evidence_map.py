"""Step-two operation checks, not a source-answer evaluation."""
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.latent_context import ContextMemory, render
from experiments.latent_context_app import MemoryHost, mcp_server
from experiments.latent_mcp import serialized
from experiments.memory.evidence_map import Edge, EvidenceMap, Span
from experiments.memory.latent_ppr import AccessBudget, PPRLimits, local_ppr, query_seeds
from experiments.memory.ppr_context import PPRNavigation
from experiments.test_latent_mcp import corpus, put, sha


@pytest.fixture(autouse=True)
def cpu_floor(monkeypatch):
    monkeypatch.setenv("LSR_CPU_RAM_FLOOR_GIB", "10")


def passages(pack):
    return [p for group in pack["evidence_groups"] for p in group["spans"]]


def edge(a, b, kind="adjacency"):
    return Edge(a, b, kind, None, ())


def repin(config, index):
    manifest = json.loads((index / "manifest.json").read_text())
    manifest.update(rows_sha256=sha((index / "rows.json").read_bytes()),
                    vector_sha256=sha((index / "vectors.npy").read_bytes()))
    put(index / "manifest.json", manifest)
    cfg = json.loads(config.read_text())
    cfg["manifest_sha256"] = sha((index / "manifest.json").read_bytes())
    put(config, cfg)


def test_map_accounts_rows_and_source_bound_link_witness(corpus):
    tool = ContextMemory(corpus[0]).tool
    mapped = EvidenceMap(tool)
    assert mapped.bank is tool.memory.vectors
    assert mapped.coverage()["accounted_vector_rows"] == 2
    assert mapped.coverage()["indexed_leaf_occurrences"] == 2
    assert mapped.setup_costs["exact_vector_bytes_read"] == tool.memory.vectors.nbytes
    assert "UNKNOWN" in mapped.coverage()["extraction_universe"]
    edges = [e for values in mapped.outgoing.values() for e in values]
    ref = next(e for e in edges if e.type == "explicit_reference")
    assert ref.to_id == tool.corpus_id + ":S1"
    witness = ref.witness_span
    raw = (corpus[1] / "origin.md").read_bytes()
    assert sha(raw[witness.start:witness.end]) == witness.span_hash
    assert witness.scope[0].endswith("origin.md")
    assert ref.semantic_support == "PROPOSAL_ONLY"
    with pytest.raises(FrozenInstanceError):
        witness.start = 99
    with pytest.raises(ValueError, match="witness"):
        Edge("a", "b", "geometric_neighbor", witness, ())


def test_duplicate_bytes_keep_distinct_occurrences_and_known_omissions(corpus):
    config, source, index = corpus
    rows = json.loads((index / "rows.json").read_text())
    files = json.loads((index / "manifest.json").read_text())["files"]
    duplicate = source / "copy.md"
    duplicate.write_bytes((source / "origin.md").read_bytes())
    rows.append(dict(rows[0], source=str(duplicate)))
    files.append(dict(files[0], path=str(duplicate)))
    # This is a known empty extraction, not an invented vector.
    empty = source / "empty.md"
    empty.write_bytes(b"")
    files.append(dict(path=str(empty), sha256=sha(b""), bytes=0, extraction="utf8_text"))
    manifest = json.loads((index / "manifest.json").read_text())
    manifest.update(files=files, allowed_paths=[f["path"] for f in files],
                    empty_pages=[dict(source=str(empty), page=None)])
    put(index / "manifest.json", manifest)
    put(index / "rows.json", rows)
    np.save(index / "vectors.npy", np.ones((3, 3), dtype=np.float32))
    cfg = json.loads(config.read_text())
    cfg["allowed_paths"] = manifest["allowed_paths"]
    put(config, cfg)
    repin(config, index)
    memory = ContextMemory(config)
    mapped = EvidenceMap(memory.tool)
    a, b = mapped.spans[memory.tool.corpus_id + ":S0"], mapped.spans[memory.tool.corpus_id + ":S2"]
    assert a.leaf_hash == b.leaf_hash and a.instance_id != b.instance_id
    assert a.source_version != b.source_version and a.scope != b.scope
    assert mapped.coverage()["indexed_leaf_occurrences"] == 3
    assert mapped.coverage()["omission_counts"]["empty_extracted_leaf"] == 1
    pack = memory.context("Alpha", strategy="ppr")
    assert {"origin.md", "copy.md"} <= {p["file"] for p in passages(pack)}


def test_unencoded_intervals_are_explicit(corpus):
    config, _, index = corpus
    rows = json.loads((index / "rows.json").read_text())
    text = rows[0]["text"][6:]
    rows[0].update(start_byte=6, text=text, span_sha256=sha(text.encode()))
    put(index / "rows.json", rows)
    repin(config, index)
    mapped = EvidenceMap(ContextMemory(config).tool)
    assert any(o.get("start") == 0 and o.get("end") == 6 for o in mapped.omissions)


def test_scoped_title_association_keeps_literal_witness(corpus):
    config, source, index = corpus
    rows = json.loads((index / "rows.json").read_text())
    manifest = json.loads((index / "manifest.json").read_text())
    for i, text in enumerate(("# Alpha\nSee Beta for details.\n", "# Beta\nTarget detail.\n")):
        raw = text.encode()
        (source / ("origin.md" if i == 0 else "target.md")).write_bytes(raw)
        (index / "leaves" / (sha(raw) + ".txt")).write_bytes(raw)
        rows[i].update(source_sha256=sha(raw), leaf_sha256=sha(raw),
                       span_sha256=sha(raw), text=text, start_byte=0, end_byte=len(raw))
        manifest["files"][i].update(sha256=sha(raw), bytes=len(raw))
    put(index / "rows.json", rows)
    put(index / "manifest.json", manifest)
    repin(config, index)
    tool = ContextMemory(config).tool
    mapped = EvidenceMap(tool)
    association = next(e for e in mapped.outgoing[tool.corpus_id + ":S0"]
                       if e.type == "scoped_association")
    assert association.to_id == tool.corpus_id + ":S1"
    witness = association.witness_span
    assert rows[0]["text"].encode()[witness.start:witness.end] == b"Beta"
    assert dict(association.binding_constraints)["directory"] == str(source)


def test_ppr_chain_and_dangling_mass_match_stationary_solution():
    access = AccessBudget()
    result = local_ppr({"a": (edge("a", "b"),)}, {"a": 1}, access, tolerance=1e-10)
    assert result["scores"] == pytest.approx({"a": .15, "b": .85})
    assert sum(result["scores"].values()) + result["residual_mass"] == pytest.approx(1)
    assert result["touched_nodes"] == 2
    assert access.costs["edge_reads"] == 1 and access.costs["expansions"] == 3
    assert access.costs["edge_record_bytes_read"] > 0
    assert access.costs["map_scans"] == access.costs["exact_vector_rows_read"] == 0


def test_ppr_type_degree_normalization_and_hub_cap():
    graph = {"a": tuple(edge("a", str(i)) for i in range(10)) +
             (edge("a", "other", "containment"),)}
    result = local_ppr(graph, {"a": 1}, AccessBudget(), tolerance=1e-10)
    assert result["scores"]["other"] == pytest.approx(.85 / 3)
    assert result["scores"]["0"] == pytest.approx(.85 * 2 / 3 / 10)
    capped = AccessBudget(PPRLimits(expansions=5))
    result = local_ppr(graph, {"a": 1}, capped)
    assert result["stop_reason"] == "expansion_cap"
    assert capped.costs["edge_reads"] == 0 and result["residual_mass"] == 1
    assert result["scores"] == {}


def test_local_push_never_visits_disconnected_graph():
    graph = {"a": (edge("a", "b"),)}
    graph.update({str(i): (edge(str(i), str(i + 1)),) for i in range(10000)})
    result = local_ppr(graph, {"a": 1}, AccessBudget())
    assert result["touched_nodes"] == 2


def test_deadline_and_invalid_limits():
    clock = iter([0.0, 6.0])
    access = AccessBudget(clock=lambda: next(clock))
    assert not access.check() and access.stop_reason == "navigation_time_cap"
    for limits in (dict(expansions=2049), dict(seconds=float("nan")), dict(exact_vectors=True)):
        with pytest.raises(ValueError):
            PPRLimits(**limits)


def test_dense_seeding_charges_one_map_scan_and_capped_exact_rows(corpus, monkeypatch):
    import experiments.memory.latent_ppr as ppr
    memory = ContextMemory(corpus[0])
    mapped = EvidenceMap(memory.tool)
    memory.tool.memory.encoder = object()
    monkeypatch.setattr(ppr, "encode", lambda *a: np.ones((1, 3), dtype=np.float32))
    access = AccessBudget(PPRLimits(exact_vectors=1))
    seeds = query_seeds(mapped, "Alpha", access)
    assert len(seeds) == 1
    assert access.costs["map_scans"] == 1
    assert access.costs["map_vector_rows_read"] == 2
    assert access.costs["map_bytes_read"] == mapped.routing.codes.nbytes + mapped.routing.scales.nbytes
    assert access.costs["exact_vector_rows_read"] == 1
    assert access.costs["exact_vector_bytes_read"] == 12
    capped = AccessBudget(PPRLimits(map_bytes=0))
    assert query_seeds(mapped, "Alpha", capped) == {}
    assert capped.costs["map_bytes_read"] == capped.costs["exact_vector_rows_read"] == 0
    assert capped.stop_reason == "map_byte_cap"


def test_ppr_context_reuses_packer_cursor_and_default(corpus, monkeypatch):
    import experiments.memory.ppr_context as module
    memory = ContextMemory(corpus[0])
    assert memory.context("Alpha")["strategy"] == "one_shot"
    first = memory.context("Alpha", strategy="ppr", budget=dict(max_spans=1))
    assert len(passages(first)) == 1 and first["continuation"]
    args = first["continuation"]["arguments"]
    assert args["strategy"] == "ppr"
    monkeypatch.setattr(module, "query_seeds", lambda *a: pytest.fail("continuation rerouted"))
    second = memory.context(**args)
    assert passages(second)[0]["id"] != passages(first)[0]["id"]
    assert second["costs"]["ppr"]["access"]["map_scans"] == 0
    assert second["costs"]["ppr"]["setup_this_call"] is None
    assert second["coverage"]["completeness"] == "UNKNOWN"
    assert "| ppr" in render(second)
    assert len(serialized(second).encode()) <= 48000
    for pack in (first, second):
        costs = pack["costs"]
        assert costs["corpus_encodes"] == costs["reader_model_calls"] == 0
        assert costs["verification_bytes_read"] == costs["ppr"]["access"]["verification_bytes_read"]
        for p in passages(pack):
            assert p["text"].encode() == (corpus[1] / p["file"]).read_bytes()
            assert p["byte_verified"] and sha(p["text"].encode()) == p["sha256"]
    with pytest.raises(ValueError, match="mismatch"):
        memory.context(**dict(args, strategy="one_shot"))


def test_verification_cap_and_changed_bytes_remain_charged(corpus):
    memory = ContextMemory(corpus[0])
    nav = PPRNavigation(memory, limits=PPRLimits(verification_bytes=1))
    assert nav.source_spans([memory.tool.corpus_id + ":S0"]) is None
    assert nav.access.stop_reason == "atomic_source_exceeds_verification_budget"
    assert nav.access.costs["verification_bytes_read"] == 0 and nav.refused_reads == 1
    nav = PPRNavigation(memory)
    source = corpus[1] / "origin.md"
    source.write_bytes(b"!" * source.stat().st_size)
    with pytest.raises(ValueError, match="changed"):
        nav.source_spans([memory.tool.corpus_id + ":S0"])
    assert nav.access.costs["verification_bytes_read"] == source.stat().st_size
    assert memory.ppr_last_attempt["verification_hash_bytes"] == source.stat().st_size
    assert memory.ppr_last_attempt["verification_hash_checks"] == 1


def test_corrupt_leaf_read_and_hash_attempts_are_charged(corpus):
    memory = ContextMemory(corpus[0])
    nav = PPRNavigation(memory)
    row = memory.tool.memory.rows[0]
    path = corpus[2] / "leaves" / (row["leaf_sha256"] + ".txt")
    path.write_bytes(b"!" * path.stat().st_size)
    with pytest.raises(ValueError, match="changed"):
        nav.source_spans([memory.tool.corpus_id + ":S0"])
    assert nav.access.costs["verification_bytes_read"] == 2 * path.stat().st_size
    assert nav.access.costs["verification_hash_checks"] == 2


def test_permanently_unaffordable_source_has_no_nonprogressing_cursor(corpus, monkeypatch):
    import experiments.memory.ppr_context as module
    original = module.PPRNavigation
    monkeypatch.setattr(module, "PPRNavigation",
                        lambda memory: original(memory, limits=PPRLimits(verification_bytes=1)))
    memory = ContextMemory(corpus[0])
    monkeypatch.setattr(memory.tool.memory, "_source_order",
                        lambda *a: pytest.fail("PPR packing scanned global rows"))
    pack = memory.context("Alpha", strategy="ppr")
    assert not passages(pack) and pack["continuation"] is None
    assert {u["reason"] for u in pack["unresolved"]} == {"atomic_source_exceeds_verification_budget"}
    assert pack["costs"]["verification_bytes_read"] == 0


def test_ppr_host_and_mcp_strategy_schema(corpus, tmp_path):
    import asyncio
    registry = tmp_path / "registry.json"
    put(registry, dict(test=dict(config=str(corpus[0]), source_root=str(corpus[1]),
                                description="operation fixture")))
    host = MemoryHost(registry)
    pack = host.invoke(dict(method="context", corpus="test", question="Alpha", strategy="ppr"))
    assert pack["strategy"] == "ppr" and passages(pack)
    server = mcp_server({}, "test")
    tool = next(t for t in asyncio.run(server.list_tools()) if t.name == "context")
    assert tool.input_schema["properties"]["strategy"]["anyOf"][0]["enum"] == ["one_shot", "ppr", "expand"]


@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize("strategy", ["one_shot", "ppr"])
def test_pin_receipt_matches_independent_reads_and_hashes(corpus, tmp_path, monkeypatch,
                                                         hosted, strategy):
    registry = tmp_path / "registry.json"
    put(registry, dict(test=dict(config=str(corpus[0]), source_root=str(corpus[1]),
                                description="pin operation fixture")))
    host = MemoryHost(registry)
    memory, _ = host.backend("test")
    # Construction is setup, not request work. Warm the PPR adapter too.
    memory.context("Alpha", strategy=strategy)
    pins = {corpus[0].resolve(), (corpus[2] / "manifest.json").resolve()}
    contents = {path.read_bytes() for path in pins}
    reads, hashes = [], []
    read_bytes, sha256 = Path.read_bytes, hashlib.sha256

    def observe_read(path):
        raw = read_bytes(path)
        if path.resolve() in pins:
            reads.append(len(raw))
        return raw

    def observe_hash(raw=b"", **kwargs):
        if raw in contents:
            hashes.append(len(raw))
        return sha256(raw, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", observe_read)
    monkeypatch.setattr(hashlib, "sha256", observe_hash)

    def invoke(arguments):
        if hosted:
            return host.invoke(dict(method="context", **arguments))
        return memory.context(**arguments)

    arguments = dict(question="Alpha", corpus="test", strategy=strategy, budget={"max_spans": 1})
    for page in range(2):
        reads.clear()
        hashes.clear()
        pack = invoke(arguments)
        assert sorted(reads) == sorted(hashes) == sorted(map(len, contents))
        assert pack["costs"]["pin_bytes_hashed"] == sum(hashes) == sum(reads)
        assert pack["costs"]["pin_hash_checks"] == len(hashes) == 2
        if page == 0:
            assert pack["continuation"]
            arguments = pack["continuation"]["arguments"]


@pytest.mark.parametrize("pin", ["config", "manifest"])
def test_hosted_context_still_invalidates_changed_pins(corpus, tmp_path, pin):
    registry = tmp_path / "registry.json"
    put(registry, dict(test=dict(config=str(corpus[0]), source_root=str(corpus[1]),
                                description="pin operation fixture")))
    host = MemoryHost(registry)
    payload = dict(method="context", corpus="test", question="Alpha", strategy="ppr",
                   budget={"max_spans": 1})
    first = host.invoke(payload)
    changed = corpus[0] if pin == "config" else corpus[2] / "manifest.json"
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed"):
        host.invoke(dict(method="context", **first["continuation"]["arguments"]))
    memory = next(iter(host.memories.values()))
    assert memory.pin_last_attempt["pin_hash_checks"] == (1 if pin == "config" else 2)

def test_large_title_scope_defers_only_unvisited_edges(corpus):
    from experiments.memory.latent_frontend import title_graph
    config, source, index = corpus
    text = 'Shared title\né → Shared title.\n'
    raw = text.encode()
    (index / 'leaves' / (sha(raw) + '.txt')).write_bytes(raw)
    files, rows = [], []
    for i in range(513):
        path = source / f'flat{i}.txt'
        path.write_bytes(raw)
        files.append(dict(path=str(path), sha256=sha(raw), bytes=len(raw), extraction='utf8_text'))
        rows.append(dict(source=str(path), source_sha256=sha(raw), leaf_sha256=sha(raw),
                         page=None, start_byte=0, end_byte=len(raw), span_sha256=sha(raw), text=text))
    manifest = json.loads((index / 'manifest.json').read_text())
    manifest.update(files=files, allowed_paths=[f['path'] for f in files])
    put(index / 'manifest.json', manifest)
    put(index / 'rows.json', rows)
    np.save(index / 'vectors.npy', np.ones((len(rows), 3), dtype=np.float32))
    cfg = json.loads(config.read_text())
    cfg['allowed_paths'] = manifest['allowed_paths']
    put(config, cfg)
    repin(config, index)
    mapped = EvidenceMap(ContextMemory(config).tool)
    assert len(mapped.outgoing.pending) == 513
    assert mapped.edge_counts.get('scoped_association', 0) == 0
    earlier_counts = mapped.edge_counts
    node = mapped.tool.corpus_id + ':S0'
    edges = mapped.outgoing.get(node)
    associations = [e for e in edges if e.type == 'scoped_association']
    eager = title_graph([dict(title='Shared title', paragraph_text=text)] * 513, origins=[0])[0]
    assert [(e.to_id, e.witness_span.start, e.witness_span.end) for e in associations] == [
        (mapped.tool.corpus_id + f":S{e['target']}", e['start_byte'], e['end_byte']) for e in eager]
    assert mapped.outgoing[node] is edges
    assert len(mapped.outgoing.pending) == 512
    assert mapped.edge_counts['scoped_association'] == len(eager)
    assert earlier_counts.get('scoped_association', 0) == 0
    assert mapped.outgoing.get('missing', ()) == ()
