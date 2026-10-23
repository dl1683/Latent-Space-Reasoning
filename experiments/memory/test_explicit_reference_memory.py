"""Structural CPU fixtures; no models, training, service, or quality judgment."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import dogfood as d
from experiments.memory.explicit_reference_memory import ExplicitReferenceMemory, inline_links


def fixture_index(tmp_path, texts=None, split_origin=False):
    source_root, directory = tmp_path / "repo", tmp_path / "index"
    source_root.mkdir()
    (directory / "leaves").mkdir(parents=True)
    if texts is None:
        texts = {
            "README.md": "Préface é🙂\nOpening row.\n\nCafé\n[the\ndefinition](docs/target%20name.md#Meaning)\n",
            "docs/target name.md": "# Meaning\nOne model, memory optional.\nSecond row.\n",
            "other/target name.md": "Duplicate basename is unrelated.\n",
        }
    rows, files = [], []
    for name, text in texts.items():
        source = source_root / name
        source.parent.mkdir(parents=True, exist_ok=True)
        raw = text.encode("utf-8")
        source.write_bytes(raw)
        version = d.digest(raw)
        files.append(dict(path=str(source.resolve()), sha256=version, bytes=len(raw), extraction="utf8_text"))
        (directory / "leaves" / (version + ".txt")).write_bytes(raw)
        cuts = [0, len(raw)]
        if split_origin and name == "README.md":
            cuts.insert(1, raw.index(b"Caf"))
        for a, b in zip(cuts, cuts[1:]):
            span = raw[a:b]
            rows.append(dict(source=str(source.resolve()), source_sha256=version, page=None,
                leaf_sha256=version, start_byte=a, end_byte=b, span_sha256=d.digest(span),
                text=span.decode("utf-8")))
    d.write(directory / "rows.json", rows)
    # Constructor-only structural fixture. No neural meaning is assigned to zeros.
    np.save(directory / "vectors.npy", np.zeros((len(rows), 2), dtype=np.float32), allow_pickle=False)
    manifest = dict(files=files, vector_sha256=d.file_digest(directory / "vectors.npy"),
                    rows_sha256=d.file_digest(directory / "rows.json"))
    d.write(directory / "manifest.json", manifest)
    return directory, source_root, manifest


def backend(tmp_path, **kwargs):
    directory, root, manifest = fixture_index(tmp_path, **kwargs)
    memory = ExplicitReferenceMemory(directory, source_root=root, manifest=manifest)
    return memory


def open_readme(memory, k=1):
    index = next(i for i, r in enumerate(memory.rows) if Path(r["source"]).name == "README.md")
    return memory.search("Opening", k, arm="bm25", source_navigation=True, allowed_rows=[index])


def resolved(result):
    return [r for p in result["passages"] for r in p["explicit_references"]
            if r["resolution"] == "MANIFEST_RESOLVED_UNFETCHED"]


def state(memory):
    return (set(memory.seen), dict(getattr(memory, "navigation_anchors", {})),
            deepcopy(memory._references), set(memory._returned), deepcopy(memory._lineages))


def test_multiline_utf8_nonzero_leaf_offsets_and_no_full_leaf_parsing(tmp_path):
    memory = backend(tmp_path, split_origin=True)
    first = open_readme(memory)
    assert first["passages"][0]["explicit_references"] == []
    assert not memory._references  # target link exists in same leaf, but is unreturned
    result = memory.search("next:S0", 1, arm="dense", source_navigation=True)
    edge = resolved(result)[0]
    row = memory.rows[1]
    raw = Path(row["source"]).read_bytes()
    span = raw[edge["link_start_byte"]:edge["link_end_byte"]]
    assert span == b"[the\ndefinition](docs/target%20name.md#Meaning)"
    assert edge["link_sha256"] == d.digest(span)
    assert edge["link_start_byte"] > row["start_byte"] > len("Préface é🙂")
    assert edge["origin_pointer"] == result["passages"][0]["pointer"]
    assert edge["fragment"] == "Meaning"
    assert edge["address"] == "source:docs/target name.md"
    assert edge["target_verification"] == "REQUIRED_ON_FOLLOW"
    assert d.reader_evidence(result)["sources"][0]["explicit_references"] == [edge]


def test_ordinary_memory_crlf_extraction_keeps_leaf_coordinates_explicit(tmp_path):
    directory, root, manifest = fixture_index(tmp_path, texts={
        "README.md": "Opening é🙂\r\n[x](docs/a.md)\r\n", "docs/a.md": "Target.\r\n"})
    # Use the actual ordinary source_pages UTF-8 extraction (newline translation).
    rows = []
    for item in manifest["files"]:
        page, text = next(d.source_pages(Path(item["path"])))
        raw = text.encode()
        leaf = d.digest(raw)
        (directory / "leaves" / (leaf + ".txt")).write_bytes(raw)
        rows.append(dict(source=item["path"], source_sha256=item["sha256"], page=page,
            leaf_sha256=leaf, start_byte=0, end_byte=len(raw), span_sha256=leaf, text=text))
    (directory / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    manifest["rows_sha256"] = d.file_digest(directory / "rows.json")
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    memory = ExplicitReferenceMemory(directory, source_root=root, manifest=manifest)
    edge = resolved(open_readme(memory))[0]
    assert edge["link_coordinate_space"] == "VERIFIED_SOURCE_LEAF_UTF8"
    assert edge["origin_pointer"]["source_sha256"] != edge["origin_pointer"]["leaf_sha256"]
    leaf = (directory / "leaves" / (edge["origin_pointer"]["leaf_sha256"] + ".txt")).read_bytes()
    assert leaf[edge["link_start_byte"]:edge["link_end_byte"]] == edge["link_text"].encode()
    assert memory.search(edge["address"], 1, arm="bm25", source_navigation=True)["passages"][0]["source_id"] == "docs/a.md"


@pytest.mark.parametrize("arm", ["bm25", "dense", "hybrid"])
def test_follow_no_encoder_vectors_unchanged_and_exact_cost(tmp_path, monkeypatch, arm):
    memory = backend(tmp_path)
    initial = open_readme(memory)
    edge = resolved(initial)[0]
    ids, addresses = d.typed_navigation_metadata(initial)
    assert ids == ["README.md"]
    assert edge["address"] in addresses
    assert "docs/target name.md" not in ids
    with pytest.raises(ValueError, match="citation"):
        d.accept_typed_navigation_answer(dict(answer="unfetched", citations=["docs/target name.md"]), ids)
    def forbidden(*args, **kwargs):
        pytest.fail("addressed follow encoded a query")
    monkeypatch.setattr(d, "encode", forbidden)
    vectors = memory.vectors.tobytes()
    result = memory.search(edge["address"], 1, arm=arm, novelty=True,
                           question="public question", source_navigation=True, allowed_rows=None)
    p = result["passages"][0]
    assert p["source_id"] == "docs/target name.md"
    assert "unrelated" not in p["text"]
    assert p["reference_lineage"] == [edge]
    assert result["reference_follow"]["fragment_resolved"] is False
    assert result["reference_follow"]["operation"] == "PARTIAL_SOURCE_ROWS"
    assert memory.vectors.tobytes() == vectors and memory.question_prior is None
    source_len = Path(edge["origin_pointer"]["source"]).stat().st_size
    target_len = Path(p["pointer"]["source"]).stat().st_size
    assert result["source_version_bytes_read"] == source_len + target_len
    assert result["navigation_verification_bytes_read"] == source_len + target_len
    assert result["source_bytes_read"] == 2 * (source_len + target_len)
    assert result["reference_costs"]["reference_bytes_checked"] == len(edge["link_text"].encode())
    assert result["reference_costs"]["reference_hops"] == 1
    assert d.typed_navigation_metadata(result)[0] == ["docs/target name.md"]


@pytest.mark.parametrize("destination", [
    "missing.md", "target%20name.md", "https://example.org/docs/target.md", "file:///tmp/target.md",
    "/docs/target.md", "C:/docs/target.md", "//server/docs/target.md", "../escape.md",
    "%2e%2e/escape.md", "%2Fdocs/target.md", "docs/%ZZ.md", "docs/%FF.md",
    "docs/target%20name.md?query", "docs/%00target.md", "#Meaning", "docs%5ctarget.md",
])
def test_unresolved_no_basename_external_absolute_escape_missing(tmp_path, destination):
    memory = backend(tmp_path, texts={"README.md": "Opening [x](" + destination + ")\n",
                                     "docs/target name.md": "Target."})
    result = open_readme(memory)
    refs = result["passages"][0]["explicit_references"]
    assert len(refs) == 1 and refs[0]["resolution"] == "UNRESOLVED"
    assert "address" not in refs[0] and not memory._references
    assert d.typed_navigation_metadata(result) == (["README.md"], ["source:README.md"])


def test_exact_relative_parent_resolution_and_encoded_hash(tmp_path):
    memory = backend(tmp_path, texts={"docs/README.md": "Opening [x](../other/a%23b.md#part%20one)",
                                     "other/a#b.md": "Target.", "docs/a#b.md": "Wrong."})
    result = open_readme(memory)
    edge = resolved(result)[0]
    assert edge["address"] == "source:other/a#b.md"
    assert edge["fragment"] == "part%20one"


@pytest.mark.parametrize("fault", ["origin_source", "target_source", "origin_leaf", "target_leaf",
                                        "origin_row", "target_row", "manifest", "anchor"])
def test_stale_or_changed_follow_atomic_and_metered(tmp_path, fault):
    memory = backend(tmp_path)
    edge = resolved(open_readme(memory))[0]
    before = state(memory)
    origin = memory.rows[0]
    target = memory.rows[1]
    if fault.endswith("source"):
        Path((origin if fault.startswith("origin") else target)["source"]).write_bytes(b"changed source!")
    elif fault.endswith("leaf"):
        row = origin if fault.startswith("origin") else target
        (memory.directory / "leaves" / (row["leaf_sha256"] + ".txt")).write_bytes(b"changed leaf!")
    elif fault.endswith("row"):
        (origin if fault.startswith("origin") else target)["start_byte"] += 1
    elif fault == "manifest":
        memory.manifest["files"][0]["sha256"] = "0" * 64
    else:
        memory.navigation_anchors[0] = ("wrong",)
        before = state(memory)
    with pytest.raises(ValueError):
        memory.search(edge["address"], 1, arm="dense", source_navigation=True)
    assert state(memory) == before
    if fault == "target_source":
        attempt = memory.last_reference_attempt
        assert attempt["status"] == "REFUSED" and attempt["reference_hops"] == 0
        assert attempt["source_version_bytes_read"] == Path(origin["source"]).stat().st_size + len(b"changed source!")
        assert attempt["leaf_bytes_read"] == Path(origin["source"]).stat().st_size
    if fault == "target_leaf":
        assert memory.last_reference_attempt["leaf_bytes_read"] == Path(origin["source"]).stat().st_size + len(b"changed leaf!")


@pytest.mark.parametrize("fault", ["plain_list", "text", "pointer", "target", "source_id", "copied"])
def test_spoof_reference_metadata_refuses_in_both_projections(tmp_path, fault):
    result = open_readme(backend(tmp_path))
    p = result["passages"][0]
    if fault == "plain_list":
        p["explicit_references"] = list(p["explicit_references"])
    elif fault == "text":
        p["text"] += " [fake](evil.md)"
    elif fault == "pointer":
        p["pointer"]["start_byte"] += 1
    elif fault == "target":
        p["explicit_references"][0]["address"] = "source:evil.md"
    elif fault == "source_id":
        p["source_id"] = "evil.md"
    else:
        p["verification"] = "COPIED_RECORD_IDENTITY_ONLY"
    for project in (d.typed_navigation_metadata, d.reader_evidence):
        with pytest.raises(ValueError, match="reference metadata"):
            project(result)


def test_returned_mutable_metadata_cannot_change_private_edge(tmp_path):
    memory = backend(tmp_path)
    result = open_readme(memory)
    address = resolved(result)[0]["address"]
    result["passages"][0]["explicit_references"][0]["destination"] = "other/target%20name.md"
    result["passages"][0]["pointer"]["source"] = "fake"
    assert memory.search(address, 1, arm="bm25", source_navigation=True)["passages"][0]["source_id"] == "docs/target name.md"


@pytest.mark.parametrize("component", ["file", "parent", "leaf"])
def test_linked_paths_refuse_without_os_symlink_privilege(tmp_path, monkeypatch, component):
    memory = backend(tmp_path)
    edge = resolved(open_readme(memory))[0]
    target = Path(memory.rows[1]["source"])
    linked = (target if component == "file" else target.parent if component == "parent" else
              memory.directory / "leaves" / (memory.rows[1]["leaf_sha256"] + ".txt"))
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == linked or original(path))
    before = state(memory)
    with pytest.raises(ValueError, match="linked|identity"):
        memory.search(edge["address"], 1, source_navigation=True)
    assert state(memory) == before


def test_origin_lineage_survives_next_window_and_second_reference(tmp_path):
    memory = backend(tmp_path, texts={"README.md": "Opening [x](docs/a.md)",
        "docs/a.md": "row one\nrow two [x](../b.md)\n", "b.md": "Final target."})
    # Split target into two original-byte rows, then pin a fresh genuine index.
    original = memory.rows[1]
    raw = original["text"].encode()
    cut = raw.index(b"row two")
    pieces = [dict(original, start_byte=a, end_byte=b, span_sha256=d.digest(raw[a:b]),
                   text=raw[a:b].decode()) for a, b in ((0, cut), (cut, len(raw)))]
    rows = memory.rows[:1] + pieces + memory.rows[2:]
    (memory.directory / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    np.save(memory.directory / "vectors.npy", np.zeros((len(rows), 2), np.float32), allow_pickle=False)
    manifest = dict(memory.manifest, rows_sha256=d.file_digest(memory.directory / "rows.json"),
                    vector_sha256=d.file_digest(memory.directory / "vectors.npy"))
    (memory.directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    memory = ExplicitReferenceMemory(memory.directory, source_root=memory.source_root, manifest=manifest)
    first_edge = resolved(open_readme(memory))[0]
    first_target = memory.search(first_edge["address"], 1, arm="bm25", source_navigation=True)
    second_window = memory.search(first_target["passages"][0]["navigation"]["next"],
                                  1, arm="bm25", source_navigation=True)
    assert second_window["passages"][0]["reference_lineage"] == [first_edge]
    second_edge = resolved(second_window)[0]
    final = memory.search(second_edge["address"], 1, arm="bm25", source_navigation=True)
    assert final["passages"][0]["reference_lineage"] == [first_edge, second_edge]
    assert final["passages"][0]["pointer"]["source"] == memory.rows[-1]["source"]


def test_target_failure_on_second_row_publishes_no_first_row(tmp_path, monkeypatch):
    memory = backend(tmp_path)
    edge = resolved(open_readme(memory))[0]
    # A duplicate row is pinned into the host catalogue to exercise Memory's
    # all-rows-before-publication behavior, then source changes between checks.
    memory.rows.append(dict(memory.rows[1]))
    memory._rows_pin = tuple(memory._row_identity(row) for row in memory.rows)
    before = state(memory)
    verify = memory._verify_current_source
    checks = []
    def checked(row):
        checks.append(row)
        if len(checks) == 3:  # origin, first target row, second target row
            Path(row["source"]).write_bytes(b"changed between target rows")
        return verify(row)
    monkeypatch.setattr(memory, "_verify_current_source", checked)
    with pytest.raises(ValueError, match="source"):
        memory.search(edge["address"], 2, arm="bm25", source_navigation=True)
    assert state(memory) == before


def test_unreturned_target_direct_address_refuses(tmp_path):
    memory = backend(tmp_path)
    before = state(memory)
    with pytest.raises(ValueError, match="unreturned"):
        memory.search("source:docs/target name.md", 1, source_navigation=True)
    assert state(memory) == before


def test_symlink_target_rejected_and_swapped_target_atomic(tmp_path):
    memory = backend(tmp_path)
    result = open_readme(memory)
    target = Path(memory.rows[1]["source"])
    outside = tmp_path / "outside.md"
    outside.write_text("Outside", encoding="utf-8")
    backup = target.read_bytes()
    target.unlink()
    try:
        target.symlink_to(outside)
    except OSError as exc:
        target.write_bytes(backup)
        pytest.skip("OS does not grant symlink creation: " + str(exc))
    before = state(memory)
    with pytest.raises(ValueError):
        memory.search(resolved(result)[0]["address"], 1, source_navigation=True)
    assert state(memory) == before
    again = open_readme(memory)
    assert again["passages"][0]["explicit_references"][0]["resolution"] == "UNRESOLVED"


@pytest.mark.parametrize("text", [
    "![image](docs/a.md)", "[label][ref]", "[label](docs/a.md", "[label](docs/a.md 'title')",
    "[nested [label]](docs/a.md)", "[label](docs/(a).md)", "[label](<docs/a.md>)",
    "`[code](docs/a.md)`", "```md\n[code](docs/a.md)\n```", "\\[escaped](docs/a.md)",
    "<!-- [comment](docs/a.md) -->", "[blank\n\nparagraph](docs/a.md)",
])
def test_narrow_parser_limits(text):
    assert list(inline_links(text)) == []


def test_link_split_across_rows_never_joined(tmp_path):
    memory = backend(tmp_path, texts={"README.md": "Opening [target](docs/target.md)", "docs/target.md": "target"})
    # Recreate pinned on-disk rows as a genuine split fixture before construction.
    row = memory.rows[0]
    cuts = [0, len("Opening [target](docs/"), len(row["text"])]
    pieces = []
    for a, b in zip(cuts, cuts[1:]):
        part = dict(row, start_byte=a, end_byte=b, text=row["text"][a:b], span_sha256=d.digest(row["text"][a:b].encode()))
        pieces.append(part)
    rows = pieces + memory.rows[1:]
    (memory.directory / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    np.save(memory.directory / "vectors.npy", np.zeros((len(rows), 2), np.float32), allow_pickle=False)
    manifest = dict(memory.manifest, rows_sha256=d.file_digest(memory.directory / "rows.json"),
                    vector_sha256=d.file_digest(memory.directory / "vectors.npy"))
    (memory.directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    split = ExplicitReferenceMemory(memory.directory, source_root=memory.source_root, manifest=manifest)
    result = open_readme(split, k=2)
    assert not resolved(result)


@pytest.mark.parametrize("arm", ["bm25", "dense", "hybrid"])
def test_normal_search_delegates_unchanged(tmp_path, monkeypatch, arm):
    memory = backend(tmp_path)
    base = d.Memory(memory.directory, object())
    def encode(encoder, texts):
        return np.ones((len(texts), 2), dtype=np.float32)
    monkeypatch.setattr(d, "encode", encode)
    for query in ("Opening", "source:target name.md", "next:S0"):
        assert memory.search(query, 2, arm=arm, novelty=True, question="public question") == base.search(
            query, 2, arm=arm, novelty=True, question="public question")
    assert not memory._references and not memory.reference_costs["reference_hops"]


def test_default_memory_and_old_projection_have_no_new_metadata(tmp_path):
    directory, _, _ = fixture_index(tmp_path)
    result = d.Memory(directory, None).search("Opening", 1, arm="bm25", source_navigation=True)
    p = result["passages"][0]
    assert "explicit_references" not in p
    assert d.typed_navigation_metadata(result)[0] == ["README.md"]
    assert set(d.reader_evidence(result)["sources"][0]) == {
        "source_id", "span", "page", "quoted_source_text", "navigation"}


def test_constructor_rejects_manifest_mismatch_and_duplicate(tmp_path):
    directory, root, manifest = fixture_index(tmp_path)
    with pytest.raises(ValueError, match="differs"):
        ExplicitReferenceMemory(directory, source_root=root, manifest=dict(files=[]))
    manifest["files"].append(manifest["files"][0])
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        ExplicitReferenceMemory(directory, source_root=root, manifest=manifest)


def test_existing_run_episode_uses_reference_address_then_target_citation(tmp_path):
    memory = backend(tmp_path)
    actions = iter([dict(query="Opening"), dict(navigate="source:docs/target name.md"),
                    dict(answer="One model, memory optional.", citations=["docs/target name.md"])])
    schemas = []
    def scripted(messages, phase, schema):
        schemas.append(schema)
        return dict(message=dict(content=json.dumps(next(actions))), done_reason="stop")
    result = d.run_episode(dict(protocol="v2", source_navigation=True,
        source_navigation_protocol="typed_v2", top_k=1, num_ctx=65536,
        max_output_tokens=128, seed=1729, question_prior=True),
        dict(id="structural", benchmark="PLUMBING_ONLY"),
        dict(id="q", type="scripted", question="What model and memory interface?"),
        dict(id="SCRIPTED_NO_MODEL", origin="software fixture"), 2, memory, None,
        schema_call=scripted, arm="bm25")
    assert result["terminated"] and result["errors"] == []
    assert result["trace"][1]["final_visible_source_ids"] == ["README.md"]
    assert "source:docs/target name.md" in result["trace"][1]["final_visible_navigation_addresses"]
    assert result["trace"][-1]["final_visible_source_ids"] == ["README.md", "docs/target name.md"]
    assert result["interactions_used"] == 2 and result["judge"] is None


def test_nested_label_substring_is_not_a_reference_capability(tmp_path):
    nested = "[outer [inner](t.md)](other.md)"
    assert list(inline_links(nested)) == []
    # A separate complete simple link still works after the unsupported construct.
    simple = "[valid](t.md)"
    combined = nested + " " + simple
    assert list(inline_links(combined)) == [(len(nested) + 1, len(combined), simple, "t.md")]
    memory = backend(tmp_path, texts={"README.md": "Opening " + nested,
                                     "t.md": "Inner target.", "other.md": "Outer target."})
    result = open_readme(memory)
    assert result["passages"][0]["explicit_references"] == []
    assert d.typed_navigation_metadata(result) == (["README.md"], ["source:README.md"])
    assert not memory._references


def test_empty_reference_follow_is_attempted_unfetched_and_does_not_publish(tmp_path, monkeypatch):
    memory = backend(tmp_path)
    edge = resolved(open_readme(memory))[0]
    before, directory = state(memory), memory.directory
    costs_before = dict(memory.reference_costs)
    prior_attempt = memory.last_reference_attempt
    reads = []
    read_bytes = Path.read_bytes
    def record_read(path):
        reads.append(path)
        return read_bytes(path)
    def forbidden(*args, **kwargs):
        pytest.fail("empty addressed follow encoded a query")
    monkeypatch.setattr(Path, "read_bytes", record_read)
    monkeypatch.setattr(d, "encode", forbidden)
    result = memory.search(edge["address"], 1, arm="hybrid", source_navigation=True, allowed_rows=[])
    origin = Path(edge["origin_pointer"]["source"])
    leaf = directory / "leaves" / (edge["origin_pointer"]["leaf_sha256"] + ".txt")
    origin_size = origin.stat().st_size
    assert reads == [origin, leaf]  # no target-source or target-leaf read
    assert result["passages"] == [] and state(memory) == before
    assert memory.directory is directory
    assert result["completeness"] == result["semantic_absence"] == "UNKNOWN"
    assert d.typed_navigation_metadata(result) == ([], [])
    assert result["reference_follow"]["status"] == "ATTEMPTED"
    assert result["reference_follow"]["fetched"] is False
    assert result["reference_follow"]["lineage"] == []
    expected = dict(source_version_bytes_read=origin_size, leaf_bytes_read=origin_size,
        reference_bytes_checked=len(edge["link_text"].encode()), reference_hops_attempted=1,
        reference_hops=0)
    assert result["reference_costs"] == expected
    assert result["source_version_bytes_read"] == result["navigation_verification_bytes_read"] == origin_size
    assert result["source_bytes_read"] == 2 * origin_size
    assert memory.last_reference_attempt is not prior_attempt
    assert memory.last_reference_attempt == dict(status="ATTEMPTED", **expected)
    assert memory.reference_costs == {key: costs_before[key] + value for key, value in expected.items()}
    # The retained reference remains usable; only the actual fetch completes a hop.
    fetched = memory.search(edge["address"], 1, arm="hybrid", source_navigation=True)
    assert fetched["reference_follow"]["status"] == "COMPLETED"
    assert fetched["reference_follow"]["fetched"] is True
    assert fetched["reference_costs"]["reference_hops_attempted"] == 1
    assert fetched["reference_costs"]["reference_hops"] == 1


def test_reference_next_prev_reference_lineage_lengths(tmp_path):
    directory, root, manifest = fixture_index(tmp_path, texts={
        "README.md": "Opening [first](docs/a.md)",
        "docs/a.md": "row one [second](../b.md)\nrow two\n", "b.md": "Final target."})
    rows = d.load(directory / "rows.json")
    target = rows[1]
    raw = target["text"].encode()
    cut = raw.index(b"row two")
    rows[1:2] = [dict(target, start_byte=a, end_byte=b, span_sha256=d.digest(raw[a:b]),
                      text=raw[a:b].decode()) for a, b in ((0, cut), (cut, len(raw)))]
    (directory / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    np.save(directory / "vectors.npy", np.zeros((len(rows), 2), np.float32), allow_pickle=False)
    manifest.update(rows_sha256=d.file_digest(directory / "rows.json"),
                    vector_sha256=d.file_digest(directory / "vectors.npy"))
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    memory = ExplicitReferenceMemory(directory, source_root=root, manifest=manifest)
    first_edge = resolved(open_readme(memory))[0]
    first = memory.search(first_edge["address"], 1, arm="bm25", source_navigation=True)
    next_row = memory.search(first["passages"][0]["navigation"]["next"], 1,
                             arm="bm25", source_navigation=True)
    previous = memory.search(next_row["passages"][0]["navigation"]["prev"], 1,
                             arm="bm25", source_navigation=True)
    second_edge = resolved(previous)[0]
    final = memory.search(second_edge["address"], 1, arm="bm25", source_navigation=True)
    assert [len(r["passages"][0]["reference_lineage"]) for r in (first, next_row, previous, final)] == [1, 1, 1, 2]
    for result in (first, next_row, previous):
        assert result["passages"][0]["reference_lineage"] == [first_edge]
        d.typed_navigation_metadata(result)
        d.reader_evidence(result)
    assert final["passages"][0]["reference_lineage"] == [first_edge, second_edge]
    assert final["passages"][0]["source_id"] == "b.md"
    d.typed_navigation_metadata(final)
    d.reader_evidence(final)


def test_catalogue_refusal_replaces_previous_returned_receipt(tmp_path):
    memory = backend(tmp_path)
    edge = resolved(open_readme(memory))[0]
    previous = memory.last_reference_attempt
    assert previous["status"] == "RETURNED"
    memory.rows[1]["start_byte"] += 1
    before, costs_before, directory = state(memory), dict(memory.reference_costs), memory.directory
    with pytest.raises(ValueError, match="catalogue"):
        memory.search(edge["address"], 1, source_navigation=True)
    assert memory.last_reference_attempt is not previous
    assert memory.last_reference_attempt == dict(status="REFUSED", **{key: 0 for key in costs_before})
    assert state(memory) == before and memory.reference_costs == costs_before
    assert memory.directory is directory


@pytest.mark.parametrize("text, literals", [
    (r"[escaped\](t.md) [ok](t.md)", ["[ok](t.md)"]),
    ("[unclosed opening [ok](t.md)", ["[ok](t.md)"]),
    ("[outer [inner](t.md)](other.md) [ok](t.md)", ["[ok](t.md)"]),
    (r"[before](t.md) [escaped\](t.md) [ok](t.md)", ["[before](t.md)", "[ok](t.md)"]),
    (r"\[escaped](t.md) [ok](t.md)", ["[ok](t.md)"]),
    (r"\\[escaped](t.md) [ok](t.md)", ["[ok](t.md)"]),
    (r"\[[escaped](t.md)] [ok](t.md)", ["[ok](t.md)"]),
    ("é🙂 [多行\n標籤](t.md) [ok](t.md)", ["[多行\n標籤](t.md)", "[ok](t.md)"]),
    ("![image](t.md) [title](t.md 'title') [angle](<t.md>) [ok](t.md)", ["[ok](t.md)"]),
    ("[ref][style] [blank\n\nlabel](t.md) [ok](t.md)", ["[ok](t.md)"]),
    ("`[code](t.md)` <!-- [comment](t.md) --> [ok](t.md)", ["[ok](t.md)"]),
    ("```md\n[code](t.md)\n```\n[ok](t.md)", ["[ok](t.md)"]),
    ("[dest](literal[part].md) [ok](t.md)", ["[dest](literal[part].md)", "[ok](t.md)"]),
    ("[dest](literal[.md) [ok](t.md) trailing]", ["[dest](literal[.md)", "[ok](t.md)"]),
    ("[dest](literal].md) [ok](t.md)", ["[dest](literal].md)", "[ok](t.md)"]),
    ("[bad](path([fake](t.md))) [ok](t.md)", ["[ok](t.md)"]),
    ("[bad](path [fake](t.md)) [ok](t.md)", ["[ok](t.md)"]),
    ("[bad](<path[fake](t.md)>) [ok](t.md)", ["[ok](t.md)"]),
    ("[bad](t.md '[fake](t.md)') [ok](t.md)", ["[ok](t.md)"]),
    ("(ordinary prose [ok](t.md))", ["[ok](t.md)"]),
    ("[bad](unterminated [fake](t.md)", []),
])
def test_recovery_matched_enclosures_and_token_boundaries(text, literals):
    found = list(inline_links(text))
    assert [link[2] for link in found] == literals
    for start, end, literal, destination in found:
        assert text[start:end] == literal
        assert literal.endswith("(" + destination + ")")


@pytest.mark.parametrize("destination", ["t.md", "literal[part].md", "literal[.md", "literal].md"])
@pytest.mark.parametrize("unicode_case", [False, True])
def test_recovery_backend_exact_bytes_address_then_citation(tmp_path, monkeypatch, destination, unicode_case):
    # Exactly ONE backslash before the first closing bracket, including UTF-8
    # before the supported multiline link. No target content is read on issue.
    literal = ("[多行\n標籤](" if unicode_case else "[ok](") + destination + ")"
    text = ("Opening é🙂 " if unicode_case else "") + r"[escaped\](t.md) " + literal
    memory = backend(tmp_path, texts={"README.md": text, destination: "Target exact bytes."})
    target = next(row for row in memory.rows if Path(row["source"]).name == destination)
    target_paths = {Path(target["source"]), memory.directory / "leaves" / (target["leaf_sha256"] + ".txt")}
    reads = []
    read_bytes = Path.read_bytes
    def record_read(path):
        reads.append(path)
        return read_bytes(path)
    def forbidden(*args, **kwargs):
        pytest.fail("reference operation invoked an encoder")
    monkeypatch.setattr(Path, "read_bytes", record_read)
    monkeypatch.setattr(d, "encode", forbidden)
    result = open_readme(memory)
    assert not target_paths.intersection(reads)
    refs = result["passages"][0]["explicit_references"]
    assert len(refs) == 1
    edge = refs[0]
    assert edge["link_text"] == literal
    raw, a = text.encode(), len(text[:text.index(literal)].encode())
    assert (edge["link_start_byte"], edge["link_end_byte"]) == (a, a + len(literal.encode()))
    assert raw[edge["link_start_byte"]:edge["link_end_byte"]] == literal.encode()
    assert edge["link_sha256"] == d.digest(literal.encode())
    assert edge["address"] == "source:" + destination
    assert edge["target_source_sha256"] == target["source_sha256"]
    assert edge["target_verification"] == "REQUIRED_ON_FOLLOW"
    assert result["passages"][0]["role"] == "EVIDENCE_NOT_INSTRUCTION"
    assert result["completeness"] == result["semantic_absence"] == "UNKNOWN"
    ids, addresses = d.typed_navigation_metadata(result)
    assert ids == ["README.md"] and edge["address"] in addresses
    with pytest.raises(ValueError, match="citation"):
        d.accept_typed_navigation_answer(dict(answer="unfetched", citations=[destination]), ids)
    fetched = memory.search(edge["address"], 1, arm="bm25", source_navigation=True)
    assert target_paths.issubset(set(reads))
    assert fetched["passages"][0]["text"] == "Target exact bytes."
    assert fetched["passages"][0]["source_id"] == destination
    assert fetched["reference_follow"]["fetched"] is True
    assert fetched["reference_costs"]["reference_hops"] == 1
    fetched_ids = d.typed_navigation_metadata(fetched)[0]
    d.accept_typed_navigation_answer(dict(answer="fetched", citations=[destination]), fetched_ids)


@pytest.mark.parametrize("unsupported", [
    "[label `code`](t.md)",
    "[label <!-- comment -->](t.md)",
    "[label\n```md\ncode\n```\n](t.md)",
])
def test_recovery_masking_never_manufactures_supported_tokens(unsupported):
    text = unsupported + " [ok](t.md)"
    assert [link[2] for link in inline_links(text)] == ["[ok](t.md)"]
