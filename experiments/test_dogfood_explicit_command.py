"""Stock run_one software delivery checks; scripted BM25, no external assets."""
from copy import deepcopy
import inspect
import json
from pathlib import Path

import pytest

from experiments import dogfood as d
from experiments.memory import explicit_reference_memory as ref
from experiments.memory.test_explicit_reference_memory import fixture_index

REAL_ENCODER_AT, REAL_ENCODE = d.encoder_at, d.encode


def forbidden(*args, **kwargs):
    raise AssertionError("External/model/encoder work is forbidden")


@pytest.fixture(autouse=True)
def no_native(monkeypatch):
    for name in ("encoder_at", "encode", "ollama_json"):
        monkeypatch.setattr(d, name, forbidden)
    monkeypatch.setattr(d, "reader_info", lambda _: dict(scripted=True))
    monkeypatch.setattr(d, "ram_guard", lambda: 30 * 2**30)
    monkeypatch.setattr(d, "reader_call", forbidden)


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def ordinary_manifest(directory, manifest, config, task):
    """Promote a constructor-only fixture to the complete ordinary build shape."""
    manifest.update(schema=1, task_id=task["id"], allowed_paths=task["allowed_paths"],
        source_roots=task["source_roots"], encoder=config["encoder"], chunk_chars=config["chunk_chars"],
        empty_pages=[], extraction="PER_FILE", corpus_encodes=1,
        chunks=len(d.load(directory / "rows.json")), encoder_max_seq_length=128,
        pooling=2, normalization="l2", truncation="NONE_TOKENIZER_LENGTH_CHECKED",
        semantic_completeness="UNKNOWN")
    save(directory / "manifest.json", manifest)


def setup(tmp_path, **changes):
    directory, root, manifest = fixture_index(tmp_path)
    config = dict(schema=1, seed=1729, encoder=str(tmp_path / "NO_ENCODER"),
        readers=[dict(id="SCRIPTED", origin="software fixture", parameters=1)],
        caps=[0, 3], num_ctx=8192, max_output_tokens=128, chunk_chars=128, top_k=1,
        protocol="v2", source_navigation=True, source_navigation_protocol="typed_v2",
        explicit_references=True, tasks=[dict(id=directory.name, benchmark="PLUMBING_ONLY",
            allowed_paths=[x["path"] for x in manifest["files"]],
            source_roots=[str(root.resolve())], questions=[dict(id="q", type="scripted",
                question="What model and memory interface?")])])
    config.update(changes)
    task = config["tasks"][0]
    ordinary_manifest(directory, manifest, config, task)
    config_path = tmp_path / "config.json"
    save(config_path, config)
    save(tmp_path / "build.json", dict(config_sha256=d.file_digest(config_path)))
    return config_path, config, directory, root, manifest


def run(tmp_path, monkeypatch, state, actions, cap=3, wrong_task=None, callback=None):
    config_path, config, directory, _, _ = state
    actions = iter(actions)
    calls, memories = [], []
    real_episode = d.run_episode
    def episode(*args, **kwargs):
        memories.append(args[5])
        return real_episode(*args, **kwargs)
    def scripted(config, reader, messages, schema=None, **kwargs):
        calls.append((messages, schema))
        if callback:
            callback(len(calls))
        return dict(message=dict(content=json.dumps(next(actions))), done_reason="stop",
                    prompt_eval_count=0, eval_count=0)
    monkeypatch.setattr(d, "run_episode", episode)
    monkeypatch.setattr(d, "reader_call", scripted)
    output = tmp_path / "answer.json"
    d.run_one(config_path, directory.parent, output, config["tasks"][0]["id"],
              "q", "SCRIPTED", cap, arm="bm25", wrong_task=wrong_task)
    result = d.load(output)
    journal = [json.loads(line) for line in output.with_suffix(".journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert journal[0]["memory_backend"] == result["memory_backend"]
    assert journal[0]["memory_backend_sha256"] == result["memory_backend_sha256"]
    assert journal[-1]["record"] == result
    return result, calls, memories[0]


@pytest.mark.parametrize("profile", ["typed_v2", "typed_v3"])
@pytest.mark.parametrize("matrix", ["float32", "affine8"])
def test_explicit_follow_actual_module_pin_and_existing_counters(tmp_path, monkeypatch, profile, matrix):
    state = setup(tmp_path, source_navigation_protocol=profile, route_matrix=matrix)
    result, calls, memory = run(tmp_path, monkeypatch, state, [dict(query="Opening"),
        dict(navigate="source:docs/target name.md"),
        dict(answer="One model, memory optional.", citations=["docs/target name.md"])])
    assert type(memory) is ref.ExplicitReferenceMemory
    assert result["terminated"] and result["errors"] == [] and len(calls) == 3
    assert result["interactions_used"] == 2 and result["arm"] == "bm25"
    assert result["route_matrix"] == matrix and memory.route_matrix == matrix
    assert result["trace"][1]["final_visible_source_ids"] == ["README.md"]
    assert "source:docs/target name.md" in result["trace"][1]["final_visible_navigation_addresses"]
    assert result["trace"][-1]["final_visible_source_ids"] == ["README.md", "docs/target name.md"]
    follow = result["trace"][1]["retrieval"]
    assert follow["reference_follow"]["fetched"]
    assert follow["reference_costs"]["reference_hops_attempted"] == 1
    assert follow["reference_costs"]["reference_hops"] == 1
    assert follow["source_bytes_read"] == (follow["reference_costs"]["source_version_bytes_read"]
                                            + follow["reference_costs"]["leaf_bytes_read"])
    assert result["memory_backend"] == "explicit_reference"
    assert result["memory_backend_sha256"] == d.file_digest(ref.__file__)
    assert result["model_calls_attempted"] == result["model_calls_completed"] == 3
    assert result["memory_calls_attempted"] == 2 and result["judge"] is None


@pytest.mark.parametrize("value", [None, False])
def test_default_ordinary(tmp_path, monkeypatch, value):
    state = setup(tmp_path, explicit_references=value)
    config_path, config, *_ = state
    if value is None:
        del config["explicit_references"]
        save(config_path, config)
        save(tmp_path / "build.json", dict(config_sha256=d.file_digest(config_path)))
    result, _, memory = run(tmp_path, monkeypatch, state, [dict(query="Opening"),
        dict(answer="Opening evidence.", citations=["README.md"])])
    assert type(memory) is d.Memory
    assert result["terminated"] and result["memory_backend"] == "ordinary"
    assert result["memory_backend_sha256"] == d.file_digest(d.__file__)
    assert "explicit_references" not in result["trace"][0]["retrieval"]["passages"][0]


def test_cap0_constructs_neither_backend(tmp_path, monkeypatch):
    state = setup(tmp_path)
    monkeypatch.setattr(d, "Memory", forbidden)
    monkeypatch.setattr(ref, "ExplicitReferenceMemory", forbidden)
    result, calls, memory = run(tmp_path, monkeypatch, state, [dict(answer="UNRESOLVED", citations=[])], cap=0)
    assert memory is None and len(calls) == 1 and result["interactions_used"] == 0
    assert result["memory_backend"] == "none" and result["memory_backend_sha256"] is None
    assert result["terminated"]


@pytest.mark.parametrize("changes", [dict(explicit_references=x) for x in (1, 0, "true", None)] + [
    dict(protocol="v1"), dict(source_navigation=False), dict(source_navigation=1),
    dict(source_navigation_protocol="tag_v1"), dict(dependency_evidence=True)])
def test_rejectbool_and_conflictingprofile_before_reader(tmp_path, monkeypatch, changes):
    state = setup(tmp_path, **changes)
    monkeypatch.setattr(d, "reader_info", forbidden)
    with pytest.raises(ValueError):
        d.run_one(state[0], tmp_path, tmp_path / "answer.json", "index", "q", "SCRIPTED", 3, arm="bm25")


@pytest.mark.parametrize("roots", [[], ["ROOT", "ROOT"], ["RELATIVE"]])
def test_ambiguousroot_before_reader(tmp_path, monkeypatch, roots):
    config_path, config, _, root, _ = setup(tmp_path)
    config["tasks"][0]["source_roots"] = [str(root) if p == "ROOT" else p for p in roots]
    save(config_path, config)
    monkeypatch.setattr(d, "reader_info", forbidden)
    with pytest.raises(ValueError):
        d.run_one(config_path, tmp_path, tmp_path / "answer.json", "index", "q", "SCRIPTED", 3, arm="bm25")


@pytest.mark.parametrize("mutation", ["subset", "extra", "duplicate", "roots", "allowed_paths", "task_id", "rows", "stale", "missing"])
def test_full_manifest_mismatch_before_reader(tmp_path, monkeypatch, mutation):
    config_path, config, directory, root, manifest = setup(tmp_path)
    if mutation == "subset":
        manifest["files"].pop()
    elif mutation == "extra":
        extra = root / "unlisted.md"
        extra.write_text("outside allowlist", encoding="utf-8")
        manifest["files"].append(dict(path=str(extra), sha256=d.file_digest(extra), bytes=extra.stat().st_size))
    elif mutation == "duplicate":
        manifest["files"].append(manifest["files"][0])
    elif mutation == "roots":
        manifest["source_roots"] = [str(root.parent)]
    elif mutation == "allowed_paths":
        manifest["allowed_paths"] = manifest["allowed_paths"][:-1]
    elif mutation == "task_id":
        manifest["task_id"] = "foreign"
    elif mutation == "rows":
        (directory / "rows.json").write_text("[]", encoding="utf-8")
    elif mutation == "stale":
        (root / "docs/target name.md").write_text("changed", encoding="utf-8")
    elif mutation == "missing":
        (root / "docs/target name.md").unlink()
    save(directory / "manifest.json", manifest)
    monkeypatch.setattr(d, "reader_info", forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        d.run_one(config_path, tmp_path, tmp_path / "answer.json", "index", "q", "SCRIPTED", 3, arm="bm25")


@pytest.mark.parametrize("field", ["schema", "task_id", "allowed_paths", "source_roots", "encoder", "chunk_chars"])
def test_required_manifest_field_omission_before_reader(tmp_path, monkeypatch, field):
    config_path, _, directory, _, manifest = setup(tmp_path)
    del manifest[field]
    save(directory / "manifest.json", manifest)
    monkeypatch.setattr(d, "reader_info", forbidden)
    with pytest.raises(ValueError, match=f"missing required field: {field}"):
        d.run_one(config_path, tmp_path, tmp_path / "answer.json", "index", "q", "SCRIPTED", 3, arm="bm25")
    assert not (tmp_path / "answer.journal.jsonl").exists()


@pytest.mark.parametrize("field,value", [
    ("schema", 2), ("schema", True), ("schema", 1.0),
    ("encoder", "other-encoder"), ("chunk_chars", 129), ("chunk_chars", 128.0)])
def test_manifest_schema_encoder_chunk_exact_before_reader(tmp_path, monkeypatch, field, value):
    config_path, _, directory, _, manifest = setup(tmp_path)
    manifest[field] = value
    save(directory / "manifest.json", manifest)
    monkeypatch.setattr(d, "reader_info", forbidden)
    with pytest.raises(ValueError, match=f"reference manifest {field} differs"):
        d.run_one(config_path, tmp_path, tmp_path / "answer.json", "index", "q", "SCRIPTED", 3, arm="bm25")


def test_current_stock_complete_build_compatible_without_model(tmp_path, monkeypatch):
    from types import SimpleNamespace
    state = setup(tmp_path)
    config_path, config, _, _, _ = state
    encoder_dir = Path(config["encoder"])
    encoder_dir.mkdir()
    (encoder_dir / "fixture.txt").write_text("software identity only", encoding="utf-8")
    encoder = SimpleNamespace(max_seq_length=128,
        tokenizer=lambda text, **kwargs: {"input_ids": [0] * len(text)},
        get_sentence_embedding_dimension=lambda: 2)
    monkeypatch.setattr(d, "package_version", lambda _: "fixture")
    monkeypatch.setattr(d, "encoder_at", lambda _: encoder)
    monkeypatch.setattr(d, "encode", lambda _, texts: d.np.tile(
        d.np.array([[1.0, 0.0]], dtype=d.np.float32), (len(texts), 1)))
    index = tmp_path / "stock_build"
    # Real stock builder and manifest writer; no model is loaded.
    d.build(config_path, index)
    # Keep the same software stubs so the build source-contract codec pins match.
    directory = index / config["tasks"][0]["id"]
    manifest = d.load(directory / "manifest.json")
    assert set(manifest) == set(state[4])
    assert d.reference_manifest_at(config, config["tasks"][0], directory) == manifest
    built_state = (config_path, config, directory, state[3], manifest)
    result, calls, memory = run(tmp_path, monkeypatch, built_state,
        [dict(query="Opening"), dict(answer="Build compatible.", citations=["README.md"])])
    assert result["terminated"] and result["errors"] == [] and len(calls) == 2
    assert type(memory) is ref.ExplicitReferenceMemory and memory.manifest == manifest


def test_citationgate_before_verified_follow(tmp_path, monkeypatch):
    result, _, _ = run(tmp_path, monkeypatch, setup(tmp_path), [dict(query="Opening"),
        dict(answer="Cannot cite target yet.", citations=["docs/target name.md"])])
    assert not result["terminated"] and result["errors"]
    assert result["trace"][-1]["final_visible_source_ids"] == ["README.md"]


@pytest.mark.parametrize("missing", [False, True])
def test_stale_target_after_query_never_published(tmp_path, monkeypatch, missing):
    state = setup(tmp_path)
    def mutate(call):
        if call == 2:
            target = state[3] / "docs/target name.md"
            if missing:
                target.unlink()
            else:
                target.write_text("changed", encoding="utf-8")
    result, _, memory = run(tmp_path, monkeypatch, state, [dict(query="Opening"),
        dict(navigate="source:docs/target name.md")], callback=mutate)
    assert not result["terminated"] and result["errors"]
    assert "docs/target name.md" not in memory._returned
    assert memory.last_reference_attempt["status"] == "REFUSED"
    assert memory.reference_costs["reference_hops_attempted"] == 1
    assert memory.reference_costs["reference_hops"] == 0


def test_wrongtask_uses_selected_root_exact_manifest(tmp_path, monkeypatch):
    state = setup(tmp_path)
    peer = tmp_path / "peer"
    peer.mkdir()
    other_dir, other_root, other_manifest = fixture_index(peer)
    destination = tmp_path / "wrong"
    other_dir.rename(destination)
    wrong = deepcopy(state[1]["tasks"][0])
    wrong.update(id="wrong", allowed_paths=[x["path"] for x in other_manifest["files"]],
                 source_roots=[str(other_root)])
    ordinary_manifest(destination, other_manifest, state[1], wrong)
    state[1]["tasks"].append(wrong)
    save(state[0], state[1])
    save(tmp_path / "build.json", dict(config_sha256=d.file_digest(state[0])))
    result, _, memory = run(tmp_path, monkeypatch, state, [dict(query="Opening"),
        dict(navigate="source:docs/target name.md"),
        dict(answer="Wrong-task fixture.", citations=["docs/target name.md"])], wrong_task="wrong")
    assert result["control"] == "wrong_task" and result["memory_task_id"] == "wrong"
    assert memory.source_root == other_root.resolve() and memory.manifest == other_manifest
    assert result["index_manifest_sha256"] == d.file_digest(destination / "manifest.json")
    assert result["terminated"]


def test_wrongtask_overlap_rejected_before_reader(tmp_path, monkeypatch):
    state = setup(tmp_path)
    wrong = deepcopy(state[1]["tasks"][0])
    wrong["id"] = "wrong"
    state[1]["tasks"].append(wrong)
    save(state[0], state[1])
    save(tmp_path / "build.json", dict(config_sha256=d.file_digest(state[0])))
    monkeypatch.setattr(d, "reader_info", forbidden)
    with pytest.raises(ValueError, match="disjoint"):
        d.run_one(state[0], tmp_path, tmp_path / "answer.json", "index", "q", "SCRIPTED", 3,
                  arm="bm25", wrong_task="wrong")


def test_actual_source_contract_toggle_preserves_helper_hashes_and_identity(tmp_path, monkeypatch):
    state = setup(tmp_path)
    monkeypatch.setattr(d, "encoder_at", REAL_ENCODER_AT)
    monkeypatch.setattr(d, "encode", REAL_ENCODE)
    config_path, config, _, _, _ = state
    encoder = Path(config["encoder"])
    encoder.mkdir()
    (encoder / "fixture.txt").write_text("identity only, never loaded", encoding="utf-8")
    monkeypatch.setattr(d, "package_version", lambda _: "fixture")
    ordinary = deepcopy(config)
    del ordinary["explicit_references"]
    contract = d.source_contract(ordinary)
    assert contract == d.source_contract(config)
    for helper in (d.source_pages, d.encoder_at, d.encoder_chunks, d.encode):
        assert contract["codecs"][helper.__name__] == d.digest(inspect.getsource(helper).encode())
    save(tmp_path / "build.json", dict(config_sha256="different-original-config", source_contract=contract))
    result, _, _ = run(tmp_path, monkeypatch, state, [dict(query="Opening"),
        dict(answer="Source contract compatible.", citations=["README.md"])])
    assert result["terminated"]
    broader = deepcopy(config)
    broader["tasks"][0]["source_roots"] = [str(tmp_path)]
    assert d.source_contract(broader) != contract


def test_batch_forwards_same_opt_in_config_without_launch(tmp_path, monkeypatch):
    from types import SimpleNamespace
    state = setup(tmp_path)
    seen = []
    monkeypatch.setattr(d, "validate_smoke", lambda *args: None)
    def scripted_process(args, **kwargs):
        seen.append(args)
        forwarded = Path(args[args.index("--config") + 1])
        assert forwarded == state[0].resolve()
        assert d.config_at(forwarded)["explicit_references"] is True
        save(Path(args[args.index("--output") + 1]), dict(errors=[]))
        return SimpleNamespace(returncode=0, stdout="scripted only", stderr="")
    monkeypatch.setattr(d.subprocess, "run", scripted_process)
    d.batch(state[0], tmp_path, tmp_path / "batch", limit=1, arm="bm25", smoke_receipt="SCRIPTED")
    assert len(seen) == 1
    assert seen[0][1:5] == ["-B", "-m", "experiments.dogfood", "run"]


@pytest.mark.parametrize("field", ["source_roots", "allowed_paths"])
def test_opt_in_cannot_bypass_pinned_ordinary_source_contract(tmp_path, monkeypatch, field):
    state = setup(tmp_path)
    config_path, config, _, _, _ = state
    monkeypatch.setattr(d, "encoder_at", REAL_ENCODER_AT)
    monkeypatch.setattr(d, "encode", REAL_ENCODE)
    monkeypatch.setattr(d, "package_version", lambda _: "fixture")
    encoder = Path(config["encoder"])
    encoder.mkdir()
    (encoder / "fixture.txt").write_text("identity only", encoding="utf-8")
    ordinary = deepcopy(config)
    ordinary["explicit_references"] = False
    save(tmp_path / "build.json", dict(config_sha256="ordinary-config", source_contract=d.source_contract(ordinary)))
    if field == "source_roots":
        config["tasks"][0][field] = [str(tmp_path)]
    else:
        config["tasks"][0][field] = config["tasks"][0][field][:-1]
    save(config_path, config)
    monkeypatch.setattr(d, "reader_info", forbidden)
    with pytest.raises(ValueError, match="pinned source/encoder contract"):
        d.run_one(config_path, tmp_path, tmp_path / "answer.json", "index", "q", "SCRIPTED", 3, arm="bm25")
