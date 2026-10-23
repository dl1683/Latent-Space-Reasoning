"""Small, offline checks. No original corpus or model calls."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("eval_corpora", Path(__file__).with_name("eval_corpora.py"))
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.mark.parametrize("supplied", [False, True])
def test_inventory_explicit_provenance_preserves_input_bytes(tmp_path, monkeypatch, supplied):
    root = tmp_path / "sources"
    root.mkdir()
    (root / "a.txt").write_text("source", encoding="utf-8")
    questions = tmp_path / "questions.json"
    module.write_json(questions, [])
    smoke = tmp_path / "smoke"
    module.write_json(smoke / "manifest.json", {"tasks": []})
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(b'{"historical":true}\n')
    original = receipt.read_bytes()
    pins = tmp_path / "pins.json"
    module.write_json(pins, {"one": [{"path": receipt.name, "sha256": module.digest(receipt)}]})
    monkeypatch.setattr(module, "SPECS", [("one", root, questions)])
    monkeypatch.setattr(module, "SMOKE", smoke)
    monkeypatch.setattr(module, "WORK", tmp_path / "output")
    monkeypatch.setattr(module, "headroom", lambda: 25)
    result = module.inventory(pins if supplied else None)
    assert result["sources"][0]["provenance"] == (
        [{"path": str(receipt), "sha256": module.digest(receipt)}] if supplied else [])
    assert receipt.read_bytes() == original
    if supplied:
        receipt.write_bytes(b"changed")
        saved = (module.WORK / "manifest.json").read_bytes()
        with pytest.raises(ValueError, match="provenance hash mismatch"):
            module.inventory(pins)
        assert (module.WORK / "manifest.json").read_bytes() == saved


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "WORK", tmp_path)
    original = tmp_path / "a.md"
    original.write_text("First claim.\nSecond claim.", encoding="utf-8")
    doc = {"id": "a.md", "sha256": "abc", "path": str(original), "bytes": original.stat().st_size}
    monkeypatch.setattr(module, "digest", lambda path: "abc")
    monkeypatch.setattr(module, "headroom", lambda: 25)
    source = {"id": "one", "documents": [doc]}
    module.write_json(tmp_path / "text/one/abc.json", {
        "source_id": "one", "document_id": "a.md", "source_sha256": "abc",
        "pages": [{"page": 1, "text": "First claim.\nSecond claim."}]})
    return tmp_path, {"sources": [source]}


def make_pack(root, questions):
    path = root / "pack.json"
    module.write_json(path, {"source": "one", "questions": questions})
    return path


def question(quote="First claim."):
    return {"id": 1, "question": "What?", "citations": [
        {"document": "a.md", "page": 1, "quote": quote}]}


def test_exact_quote_and_pin(fixture):
    root, manifest = fixture
    result = module.validate(make_pack(root, [question()]), manifest)
    assert result["all_exact"]
    assert result["pack_sha256"] == module.digest(root / "pack.json")


@pytest.mark.parametrize("quote", ["invented", "", " ", "First  claim."])
def test_bad_quotes_fail(fixture, quote):
    root, manifest = fixture
    assert not module.validate(make_pack(root, [question(quote)]), manifest)["all_exact"]


def test_missing_evidence_is_not_vacuous_pass(fixture):
    root, manifest = fixture
    assert not module.validate(make_pack(root, [{"id": 1}]), manifest)["all_exact"]
    assert not module.validate(make_pack(root, []), manifest)["all_exact"]


def test_duplicate_ids_rejected(fixture):
    root, manifest = fixture
    with pytest.raises(ValueError, match="duplicate"):
        module.validate(make_pack(root, [question(), question()]), manifest)


def test_cache_identity_rejected(fixture):
    root, manifest = fixture
    cache = root / "text/one/abc.json"
    content = module.load(cache)
    content["document_id"] = "other.md"
    module.write_json(cache, content)
    with pytest.raises(ValueError, match="identity"):
        module.validate(make_pack(root, [question()]), manifest)


def test_identical_content_alias_keeps_both_identities(fixture):
    _, manifest = fixture
    source = manifest["sources"][0]
    alias = {**source["documents"][0], "id": "duplicate.md"}
    source["documents"].append(alias)
    assert module.cached_document(source, alias)["document_id"] == "a.md"


def test_tampered_cache_cannot_certify_invented_quote(fixture):
    root, manifest = fixture
    cache = root / "text/one/abc.json"
    content = module.load(cache)
    content["pages"][0]["text"] = "Invented claim."
    module.write_json(cache, content)
    assert not module.validate(make_pack(root, [question("Invented claim.")]), manifest)["all_exact"]


def test_unknown_source_or_page_rejected(fixture):
    root, manifest = fixture
    q = question()
    q["citations"][0]["source"] = "outside"
    with pytest.raises(KeyError):
        module.validate(make_pack(root, [q]), manifest)
    q["citations"][0]["source"] = "one"
    q["citations"][0]["page"] = 2
    with pytest.raises(StopIteration):
        module.validate(make_pack(root, [q]), manifest)


def test_usage_preserves_separate_fields(tmp_path):
    import json
    usage = {"input_tokens": 12, "output_tokens": 567,
             "cache_read_input_tokens": 20910, "cache_creation_input_tokens": 9299}
    events = [{"type": "tool_call", "toolName": "Read"},
              {"type": "tool_result", "value": "é"},
              {"type": "completion", "usage": usage}]
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events), encoding="utf-8")
    result = module.summarize_events(tmp_path)
    assert result["reported_usage"] == [usage]
    assert result["tool_calls"] == 1
    assert result["returned_utf8_bytes"] == 2


def test_extraction_coverage_and_errors_are_explicit():
    source = {"id": "one", "documents": [{"id": "a"}, {"id": "b"}]}
    assert not module.extraction_status(source, [{"id": "a", "status": "ok"}])["complete"]
    rows = [{"id": "a", "status": "ok"}, {"id": "b", "status": "error"}]
    assert not module.extraction_status(source, rows)["complete"]
    assert module.extraction_status(source, rows)["error_count"] == 1
    rows[1]["status"] = "cached"
    assert module.extraction_status(source, rows)["complete"]


def test_mixed_pack_uses_declared_membership(fixture, monkeypatch):
    root, manifest = fixture
    pack = root / "mixed.json"
    module.write_json(pack, {"source": "mixed", "source_memberships": ["one"],
                             "questions": [{"id": 1, "question": "claim"}]})
    result = module.baseline(pack, manifest, 1)
    assert result["questions"][0]["retrieved"][0]["document"] == "one/a.md"


def test_stale_smoke_manifest_reconciles_by_identity(tmp_path):
    inherited = tmp_path / "manifest.json"
    module.write_json(inherited, {"tasks": [{"task_id": "one/q1"}, {"task_id": "one/q2"}]})
    present = tmp_path / "one/q1/output/answer.md"
    present.parent.mkdir(parents=True)
    present.write_text("answer", encoding="utf-8")
    unlisted = tmp_path / "one/q3/output/answer.md"
    manifest = {"inherited_smoke_manifest": {"path": str(inherited)}, "sources": [
        {"id": "one", "prior_answers": [{"path": str(present)}, {"path": str(unlisted)}]}]}
    result = module.reconcile_smoke(manifest)
    assert [r["status"] for r in result["scheduled"]] == ["present", "missing"]
    assert result["observed_unlisted"][0]["task_id"] == "one/q3"


def test_reader_export_does_not_include_gold(fixture, monkeypatch):
    root, manifest = fixture
    monkeypatch.setattr(module, "SPECS", [("one", None, None)])
    module.write_json(root / "manifest.json", manifest)
    module.write_json(root / "packs/smoke.json", {"source": "one", "questions": [
        {"id": "one:1", "question": "claim?", "answer": "SECRET GOLD", "tags": ["provenance"]}]})
    result = module.package_smoke(manifest)
    questions = module.load(Path(result["directory"]) / "questions.json")
    assert questions == [{"id": "one:1", "question": "claim?"}]
    assert len(result["documents"]) == 1


def test_explicit_smoke_pack_exports_only_questions(fixture, monkeypatch):
    root, manifest = fixture
    monkeypatch.setattr(module, "SPECS", [("one", None, None)])
    module.write_json(root / "manifest.json", manifest)
    pack = root / "drafts/cross_domain.json"
    module.write_json(pack, {"source": "one", "questions": [
        {"id": "one:1", "question": "New question?", "answer": "SECRET GOLD",
         "domains": ["one", "two"], "domain_dependency": "EVALUATOR ONLY"}]})
    result = module.package_smoke(manifest, pack)
    assert module.load(Path(result["directory"]) / "questions.json") == [
        {"id": "one:1", "question": "New question?"}]
    assert len(result["documents"]) == 1


def test_out_of_universe_citation_rejected(fixture):
    root, manifest = fixture
    manifest["sources"].append({"id": "excluded", "documents": manifest["sources"][0]["documents"]})
    q = question()
    q["citations"][0]["source"] = "excluded"
    with pytest.raises(KeyError):
        module.validate(make_pack(root, [q]), manifest)


def test_unknown_membership_is_not_silently_dropped(fixture):
    _, manifest = fixture
    with pytest.raises(ValueError, match="unknown eligible"):
        module.eligible_sources({"source": "mixed", "sources": ["one", "unknown"]}, manifest)


def test_document_subset_is_shared_between_paths(fixture):
    _, manifest = fixture
    selected = module.eligible_sources({"source": "one", "eligible_source_documents": {"one": []}},
                                       manifest)
    assert selected[0]["documents"] == []


def test_single_source_document_list_supported(fixture):
    root, manifest = fixture
    p = make_pack(root, [question()])
    pack = module.load(p)
    pack["eligible_source_documents"] = ["a.md"]
    module.write_json(p, pack)
    assert module.validate(p, manifest)["all_exact"]


def bank_pack():
    return {"source": "one", "evidence_bank": {
        "first": {"document": "a.md", "page": 1, "quote": "First claim."},
        "second": {"document": "a.md", "page": 1, "quote": "Second claim."}},
        "questions": [{"id": 1, "answer": [
            {"claim": "First fact", "evidence_refs": ["first"]},
            {"claim": "Both facts", "evidence_refs": ["first", "second"]}]}]}


def test_claim_bank_resolves_all_references_without_duplicate_checks(fixture):
    root, manifest = fixture
    p = root / "bank.json"
    module.write_json(p, bank_pack())
    result = module.validate(p, manifest)
    assert result["all_exact"]
    assert [c["evidence_ref"] for c in result["checks"]] == ["first", "second"]


@pytest.mark.parametrize("refs", [[], "first", ["missing"], [None]])
def test_claim_bank_rejects_missing_or_malformed_references(refs):
    pack = bank_pack()
    pack["questions"][0]["answer"][0]["evidence_refs"] = refs
    with pytest.raises(ValueError, match="known evidence"):
        module.citations_for_question(pack, pack["questions"][0])


def test_claim_bank_rejects_uncited_claim_and_mixed_inline_evidence():
    pack = bank_pack()
    pack["questions"][0]["answer"].append({"claim": "Unsupported"})
    with pytest.raises(ValueError, match="known evidence"):
        module.citations_for_question(pack, pack["questions"][0])
    pack["questions"][0]["citations"] = question()["citations"]
    with pytest.raises(ValueError, match="mixed"):
        module.citations_for_question(pack, pack["questions"][0])


def test_same_named_packs_do_not_overwrite_each_others_receipts(fixture):
    root, manifest = fixture
    paths = [root / name / "hard_candidates.json" for name in ["a", "b"]]
    results = []
    for p in paths:
        module.write_json(p, {"source": "one", "questions": [question()]})
        results.append(module.validate(p, manifest))
    receipts = [Path(result["receipt_path"]) for result in results]
    assert receipts[0] != receipts[1]
    assert all(p.exists() for p in receipts)


@pytest.fixture
def smoke_fixture(tmp_path, monkeypatch):
    """Real hashes: refusal checks must not depend on the shared mock digest."""
    root = tmp_path / "work"
    monkeypatch.setattr(module, "WORK", root)
    monkeypatch.setattr(module, "SPECS", [("one", None, None)])
    monkeypatch.setattr(module, "headroom", lambda: 25)
    sources = tmp_path / "originals"
    sources.mkdir()
    docs = []
    for name in ["a.md", "nested/b.md"]:
        path = sources / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Authorized source " + name, encoding="utf-8")
        docs.append({"id": name, "path": str(path), "sha256": module.digest(path),
                     "bytes": path.stat().st_size})
    manifest = {"sources": [{"id": "one", "root": str(sources), "documents": docs}]}
    module.write_json(root / "manifest.json", manifest)
    pack = root / "packs/smoke.json"
    module.write_json(pack, {"source": "one", "questions": [
        {"id": "legacy", "question": "What?", "answer": "PRIVATE GOLD"}]})
    out = root / "readers/smoke" / module.digest(pack)[:16]
    return tmp_path, manifest, out


def tree_snapshot(root):
    return {p.relative_to(root).as_posix():
            (p.read_bytes(), p.stat().st_mtime_ns) if p.is_file() else None
            for p in root.rglob("*")}


def directory_link(link, target):
    import os
    import subprocess
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        # Junctions exercise the Windows reparse gate without symlink privilege.
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                       check=True, capture_output=True)


def test_clean_smoke_repeat_is_unchanged_and_legacy(smoke_fixture):
    root, manifest, out = smoke_fixture
    first = module.package_smoke(manifest)
    before = tree_snapshot(root)
    assert module.package_smoke(manifest) == first
    assert tree_snapshot(root) == before
    assert module.load(out / "questions.json") == [{"id": "legacy", "question": "What?"}]


@pytest.mark.parametrize("contamination", ["answers.json", "nested-extra", "source", "questions", "receipt"])
def test_smoke_contamination_refuses_before_any_mutation(smoke_fixture, contamination):
    root, manifest, out = smoke_fixture
    module.package_smoke(manifest)
    target = {"answers.json": out / "answers.json",
              "nested-extra": out / "sources/one/nested/gold.json",
              "source": out / "sources/one/nested/b.md",
              "questions": out / "questions.json", "receipt": out / "receipt.json"}[contamination]
    target.write_text("PRIVATE GOLD", encoding="utf-8")
    # A missing early copy must not be restored before discovering a late error.
    (out / "sources/one/a.md").unlink()
    before = tree_snapshot(root)
    with pytest.raises(ValueError, match="unexpected or contaminated"):
        module.package_smoke(manifest)
    assert tree_snapshot(root) == before


@pytest.mark.parametrize("location", ["source", "export-source", "export-root"])
def test_smoke_directory_link_refuses_before_any_mutation(smoke_fixture, location):
    root, manifest, out = smoke_fixture
    if location == "source":
        directory_link(root / "linked-originals", root / "originals")
        manifest["sources"][0]["documents"][1]["path"] = str(root / "linked-originals/nested/b.md")
    elif location == "export-source":
        (out / "sources/one").mkdir(parents=True)
        directory_link(out / "sources/one/nested", root / "originals/nested")
    else:
        out.parent.mkdir(parents=True)
        directory_link(out, root / "originals")
    before = tree_snapshot(root)
    with pytest.raises(ValueError, match="unsafe"):
        module.package_smoke(manifest)
    assert tree_snapshot(root) == before


def test_smoke_late_changed_original_refuses_before_copy(smoke_fixture):
    root, manifest, _ = smoke_fixture
    (root / "originals/nested/b.md").write_text("Changed source", encoding="utf-8")
    before = tree_snapshot(root)
    with pytest.raises(ValueError, match="source changed"):
        module.package_smoke(manifest)
    assert tree_snapshot(root) == before


@pytest.mark.parametrize("name", ["../escape.md", "C:/escape.md", "nested/../../escape.md", "a.md:gold"])
def test_smoke_unsafe_names_refuse_before_copy(smoke_fixture, name):
    root, manifest, _ = smoke_fixture
    manifest["sources"][0]["documents"][1]["id"] = name
    before = tree_snapshot(root)
    with pytest.raises(ValueError, match="unsafe reader name"):
        module.package_smoke(manifest)
    assert tree_snapshot(root) == before


@pytest.mark.parametrize("name", [
    "a.md.", "a.md ", "nested./b.md", "nested /b.md",
    "nested./deeper/b.md", "nested /deeper/b.md",
    "nested\\b.md.", "nested\\b.md ",
])
@pytest.mark.parametrize("location", ["document", "source"])
def test_smoke_trailing_dot_or_space_refuses_without_writes(
        smoke_fixture, monkeypatch, name, location):
    root, manifest, out = smoke_fixture
    if location == "document":
        # The valid first document is planned before the alias is encountered.
        manifest["sources"][0]["documents"][1]["id"] = name
    else:
        manifest["sources"][0]["id"] = name
        monkeypatch.setattr(module, "SPECS", [(name, None, None)])
        module.write_json(root / "work/packs/smoke.json", {
            "source": name, "questions": [{"id": "legacy", "question": "What?"}]})
    before = tree_snapshot(root)

    def unexpected_write(*args, **kwargs):
        pytest.fail("name refusal must precede every mkdir/copy/write")

    monkeypatch.setattr(Path, "mkdir", unexpected_write)
    monkeypatch.setattr(module.shutil, "copyfile", unexpected_write)
    monkeypatch.setattr(module, "write_json", unexpected_write)
    with pytest.raises(ValueError, match="unsafe reader name"):
        module.package_smoke(manifest)
    assert tree_snapshot(root) == before
    assert not out.exists()


@pytest.mark.parametrize("name", [
    "nested/report.v2.md", "nested folder/report name.md", "nested/rapport-é.md",
])
def test_smoke_valid_component_names_remain_idempotent(smoke_fixture, name):
    root, manifest, out = smoke_fixture
    doc = manifest["sources"][0]["documents"][1]
    doc["id"] = name
    first = module.package_smoke(manifest)
    assert (out / "sources/one" / name).read_bytes() == Path(doc["path"]).read_bytes()
    before = tree_snapshot(root)
    assert module.package_smoke(manifest) == first
    assert tree_snapshot(root) == before


@pytest.mark.parametrize("kind", ["hardlink", "directory", "parent-file"])
def test_smoke_unsafe_existing_target_refuses_before_copy(smoke_fixture, kind):
    root, manifest, out = smoke_fixture
    parent = out / "sources/one/nested"
    parent.parent.mkdir(parents=True)
    if kind == "parent-file":
        parent.write_text("Must survive", encoding="utf-8")
    else:
        parent.mkdir()
        target = parent / "b.md"
        if kind == "hardlink":
            target.hardlink_to(root / "originals/nested/b.md")
        else:
            target.mkdir()
    before = tree_snapshot(root)
    with pytest.raises(ValueError):
        module.package_smoke(manifest)
    assert tree_snapshot(root) == before


@pytest.mark.parametrize("suffix", [".", " "])
@pytest.mark.parametrize("location", ["source-file", "source-parent", "export-parent"])
def test_smoke_physical_alias_refuses_before_any_write(
        smoke_fixture, monkeypatch, suffix, location, record_property):
    root, manifest, out = smoke_fixture
    pack = module.WORK / "packs/smoke.json"
    doc = manifest["sources"][0]["documents"][1]
    original = Path(doc["path"])
    if location == "source-file":
        alias = original.with_name(original.name + suffix)
        doc["path"] = str(alias)
    elif location == "source-parent":
        alias = original.parent.with_name(original.parent.name + suffix) / original.name
        doc["path"] = str(alias)
    else:
        original = module.WORK
        alias = original.with_name(original.name + suffix)
        monkeypatch.setattr(module, "WORK", alias)
    # Real Windows path spellings: some parent-space aliases do not resolve.
    # Those must still fail lexically, rather than with FileNotFoundError.
    if module.os.name == "nt":
        resolves = alias.exists()
        record_property("windows_alias_resolves", resolves)
        if resolves:
            assert alias.samefile(original)
    before = tree_snapshot(root)

    def unexpected_write(*args, **kwargs):
        pytest.fail("physical name refusal must precede every mkdir/copy/write")

    monkeypatch.setattr(Path, "mkdir", unexpected_write)
    monkeypatch.setattr(Path, "write_text", unexpected_write)
    monkeypatch.setattr(Path, "write_bytes", unexpected_write)
    monkeypatch.setattr(module.shutil, "copyfile", unexpected_write)
    monkeypatch.setattr(module, "write_json", unexpected_write)
    with pytest.raises(ValueError, match="unsafe reader/source path"):
        module.package_smoke(manifest, pack)
    assert tree_snapshot(root) == before
    assert not out.exists()


@pytest.mark.parametrize("suffix", [".", " "])
@pytest.mark.parametrize("location", ["source-file", "source-parent", "export-parent"])
def test_smoke_physical_alias_guard_precedes_lstat(
        smoke_fixture, monkeypatch, suffix, location):
    _, manifest, _ = smoke_fixture
    pack = module.WORK / "packs/smoke.json"
    doc = manifest["sources"][0]["documents"][1]
    original = Path(doc["path"])
    if location == "export-parent":
        monkeypatch.setattr(module, "WORK", module.WORK.with_name(module.WORK.name + suffix))
    else:
        alias = (original.with_name(original.name + suffix) if location == "source-file"
                 else original.parent.with_name(original.parent.name + suffix) / original.name)
        doc["path"] = str(alias)
    original_lstat = Path.lstat

    def refuse_alias_stat(path, *args, **kwargs):
        if any(part.endswith((".", " ")) for part in module.PureWindowsPath(path).parts):
            pytest.fail("physical alias must be refused before lstat")
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", refuse_alias_stat)
    with pytest.raises(ValueError, match="unsafe reader/source path"):
        module.package_smoke(manifest, pack)


@pytest.mark.parametrize("name", ["folder.v2/report.v3.md", "folder name/report name.md", "données/rapport-é.md"])
def test_smoke_valid_physical_paths_remain_idempotent(smoke_fixture, name):
    root, manifest, out = smoke_fixture
    doc = manifest["sources"][0]["documents"][1]
    original = Path(doc["path"])
    physical = root / "originals" / name
    physical.parent.mkdir(parents=True, exist_ok=True)
    physical.write_bytes(original.read_bytes())
    doc["path"] = str(physical)
    first = module.package_smoke(manifest)
    assert (out / "sources/one/nested/b.md").read_bytes() == physical.read_bytes()
    before = tree_snapshot(root)
    assert module.package_smoke(manifest) == first
    assert tree_snapshot(root) == before


@pytest.fixture
def current_delivery_fixture(tmp_path, monkeypatch):
    """Tiny real-hash admission graph; no private corpus, model or external CLI."""
    monkeypatch.setattr(module, "REPO", tmp_path)
    work = tmp_path / "workspace/eval_corpora"
    monkeypatch.setattr(module, "WORK", work)
    monkeypatch.setattr(module, "headroom", lambda: 25)
    original = tmp_path / "original.md"
    original.write_text("Source evidence only.", encoding="utf-8")
    doc = {"id": "original.md", "path": str(original), "bytes": original.stat().st_size,
           "sha256": module.digest(original)}
    manifest = {"sources": [{"id": "one", "documents": [doc],
                             "prior_answers": [{"answer": "PRIVATE PRIOR"}]}]}
    module.write_json(work / "manifest.json", manifest)

    def pin(path):
        return {"path": path.relative_to(tmp_path).as_posix(), "sha256": module.digest(path),
                "bytes": path.stat().st_size}

    evidence = {"e": {"source": "one", "document": "original.md", "page": 1,
                      "quote": "Source evidence only.", "source_sha256": doc["sha256"]}}
    old = "Illustrative number (and unsupported absence)."
    new = "Illustrative number."
    q = {"id": "ddog-q014", "question": "Check the authored condition?", "answer": [
        {"claim": "PRIVATE GOLD one", "evidence_refs": ["e"]},
        {"claim": "PRIVATE GOLD two", "evidence_refs": ["e"]},
        {"claim": old, "evidence_refs": ["e"]}], "tags": ["PRIVATE TYPE"]}
    base = {"source": "one", "status": "DRAFT_HISTORICAL", "evidence_bank": evidence,
            "questions": [q, {**q, "id": "diagnostic-only", "question": "Do not admit me?"}]}
    paths = {"basepack": work / "drafts/base.json", "support": work / "reviews/support.json",
             "domain": work / "reviews/domain.json", "supplemental_review": work / "reviews/sup.json"}
    module.write_json(paths["basepack"], base)
    for key in ("support", "domain", "supplemental_review"):
        module.write_json(paths[key], {"review": "PRIVATE VERDICT"})
    primary = {"status": "CROSS_DOMAIN_BENCHMARK_ADMITTED_1",
        "manifest_sha256": module.digest(work / "manifest.json"), "admitted_questions": [{
        "source": "one", "id": q["id"], "pack": "drafts/base.json",
        "pack_sha256": module.digest(paths["basepack"]), "domain_conformance": "met",
        "support_review": "reviews/support.json", "support_review_sha256": module.digest(paths["support"]),
        "domain_review": "reviews/domain.json", "domain_review_sha256": module.digest(paths["domain"])}],
        "documentary_supported_questions": [{"id": "diagnostic-only"}]}
    module.write_json(work / "admission.json", primary)
    sq = {"id": "plural-1", "question": "Give defensible readings?", "answer": [
        {"claim": "PRIVATE SUPPLEMENTAL GOLD", "evidence_refs": ["e"]}]}
    supplemental_pack = work / "drafts/plural.json"
    module.write_json(supplemental_pack, {"source": "one", "evidence_bank": evidence,
                                         "questions": [sq], "rubrics": "PRIVATE RUBRIC"})
    supplemental = {"status": "ELIGIBLE_SUPPLEMENTAL_DOCUMENTARY_GUIDANCE",
        "primary_count": 1, "primary_admission": pin(work / "admission.json"),
        "candidate_pack": pin(supplemental_pack), "fresh_review": pin(paths["supplemental_review"]),
        "eligible_supplemental_questions": [{"id": sq["id"], "verdict": "supported",
                                             "question_object_sha256": module.value_digest(sq)}]}
    module.write_json(work / "supplemental_admission.json", supplemental)
    erratum = {"basepackpath": pin(paths["basepack"])["path"],
        "basepack_sha256": module.digest(paths["basepack"]), "questionID": "ddog-q014",
        "answerindex": 2, "answerindex_basis": "zero-based", "exactoldclaim": old, "exactnewclaim": new,
        "exactoldclaim_canonical_utf8_sha256": module.hashlib.sha256(old.encode()).hexdigest(),
        "exactnewclaim_canonical_utf8_sha256": module.hashlib.sha256(new.encode()).hexdigest(),
        "replacement": {"exact_old_substring": " (and unsupported absence)", "exact_new_substring": "",
                        "occurrences": 1}, "evidence_refs": ["e"], "mandatory_application": {"required": True}}
    module.write_json(work / "correction.json", erratum)
    review = {"verdict": "PASS_SCOPED", "checks": [{"pass": True}], "pins": {
        "basepack": pin(paths["basepack"]), "erratum": pin(work / "correction.json")}}
    module.write_json(work / "erratum_review.json", review)
    module.write_json(work / "current_review.json", {"input_sha256": module.digest(paths["basepack"])})
    module.write_json(work / "readiness.json", {"historical": "MEASURED COST MUST NOT TRANSFER"})
    lock = {"schema": "lsr.current_eval_delivery_inputs.v1", **{key: pin(work / file) for key, file in {
        "manifest": "manifest.json", "admission": "admission.json", "supplemental": "supplemental_admission.json",
        "erratum": "correction.json", "erratum_review": "erratum_review.json",
        "current_datadog_review": "current_review.json", "historical_readiness": "readiness.json"}.items()}}
    lock_path = work / "delivery/inputs.lock.json"
    module.write_json(lock_path, lock)
    output = work / "delivery/current"
    return {"root": tmp_path, "work": work, "lock": lock_path, "output": output,
            "original": original, "paths": paths, "pin": pin, "old": old, "new": new}


def build_current(f):
    return module.package_current(f["lock"], module.digest(f["lock"]), f["output"])


def repin_delivery_input(f, field, value):
    lock = module.load(f["lock"])
    path = f["root"] / lock[field]["path"]
    module.write_json(path, value)
    lock[field] = f["pin"](path)
    module.write_json(f["lock"], lock)


def test_current_delivery_admission_correction_costs_and_reader_boundary(current_delivery_fixture):
    f = current_delivery_fixture
    protected = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                 for p in f["root"].rglob("*") if p.is_file()}
    result = build_current(f)
    out = f["output"]
    assert (result["primary_questions"], result["supplemental_questions"]) == (1, 1)
    assert module.load(out / "reader/questions.json") == [
        {"pack": "primary:one", "id": "ddog-q014", "question": "Check the authored condition?"},
        {"pack": "supplemental", "id": "plural-1", "question": "Give defensible readings?"}]
    reader_text = "".join(p.read_text(encoding="utf-8") for p in (out / "reader").rglob("*.json"))
    assert "PRIVATE" not in reader_text and "unsupported absence" not in reader_text
    assert set(p.name for p in (out / "reader").iterdir()) == {"questions.json", "source_manifest.json"}
    packs = module.load(out / "evaluator/package.json")["packs"]
    effective = packs[0]["pack"]
    assert effective["status"] == "DRAFT_HISTORICAL"
    assert len(effective["questions"]) == 1
    assert effective["questions"][0]["answer"][2]["claim"] == f["new"]
    expected = module.load(f["paths"]["basepack"])
    expected["questions"] = expected["questions"][:1]
    expected["questions"][0]["answer"][2]["claim"] = f["new"]
    assert effective == expected
    assert module.load(f["paths"]["basepack"])["questions"][0]["answer"][2]["claim"] == f["old"]
    assert packs[1]["current_eligibility"] == "supplemental_documentary_guidance_only"
    contracts = module.load(out / "evaluator/cost_contracts.json")["questions"]
    assert len(contracts) == 2
    assert all(c["status"] == "UNRUN" and set(c["metrics"].values()) == {None} for c in contracts)
    assert contracts[0]["effective_question_sha256"] == module.value_digest(effective["questions"][0])
    assert len(contracts[0]["correction_pins"]) == 3
    assert not contracts[1]["correction_pins"]
    for name, expected in result["artifacts"].items():
        assert module.digest(out / name) == expected
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == before for p, before in protected.items())
    before = tree_snapshot(f["root"])
    assert build_current(f) == result
    assert tree_snapshot(f["root"]) == before


@pytest.mark.parametrize("target", [
    "lock", "manifest", "admission", "supplemental", "erratum", "erratum_review",
    "current_datadog_review", "historical_readiness", "basepack", "support", "domain",
    "supplemental_review", "original",
])
def test_current_delivery_hash_mismatch_refuses_before_writes(current_delivery_fixture, target):
    f = current_delivery_fixture
    expected_lock = module.digest(f["lock"])
    lock = module.load(f["lock"])
    path = (f[target] if target in {"lock", "original"} else f["paths"][target]
            if target in f["paths"] else f["root"] / lock[target]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    before = tree_snapshot(f["root"])
    with pytest.raises(ValueError, match="hash mismatch"):
        module.package_current(f["lock"], expected_lock, f["output"])
    assert tree_snapshot(f["root"]) == before
    assert not f["output"].exists()


@pytest.mark.parametrize("change", ["unsupported", "bad-question-pin", "missing-id", "duplicate-id"])
def test_current_delivery_supplemental_membership_fails_closed(current_delivery_fixture, change):
    f = current_delivery_fixture
    registry = module.load(f["work"] / "supplemental_admission.json")
    row = registry["eligible_supplemental_questions"][0]
    if change == "unsupported":
        row["verdict"] = "needs_revision"
    elif change == "bad-question-pin":
        row["question_object_sha256"] = "0" * 64
    elif change == "missing-id":
        row["id"] = "not-admitted"
    else:
        registry["eligible_supplemental_questions"].append(dict(row))
    repin_delivery_input(f, "supplemental", registry)
    before = tree_snapshot(f["root"])
    with pytest.raises(ValueError):
        build_current(f)
    assert tree_snapshot(f["root"]) == before


@pytest.mark.parametrize("change", ["missing-correction", "not-required", "old-claim", "evidence-refs", "review-binding", "failed-review"])
def test_current_delivery_erratum_is_mandatory_and_review_bound(current_delivery_fixture, change):
    f = current_delivery_fixture
    lock = module.load(f["lock"])
    if change == "missing-correction":
        del lock["erratum"]
        module.write_json(f["lock"], lock)
    elif change in {"review-binding", "failed-review"}:
        review = module.load(f["root"] / lock["erratum_review"]["path"])
        if change == "review-binding":
            review["pins"]["erratum"]["sha256"] = "0" * 64
        else:
            review["checks"][0]["pass"] = False
        repin_delivery_input(f, "erratum_review", review)
    else:
        erratum = module.load(f["root"] / lock["erratum"]["path"])
        if change == "not-required":
            erratum["mandatory_application"]["required"] = False
        elif change == "old-claim":
            erratum["exactoldclaim"] = "Already corrected or wrong version."
        else:
            erratum["evidence_refs"] = ["changed"]
        repin_delivery_input(f, "erratum", erratum)
        # Keep review pinned to the newly authored erratum, so exact checks run.
        review = module.load(f["root"] / lock["erratum_review"]["path"])
        review["pins"]["erratum"] = f["pin"](f["root"] / lock["erratum"]["path"])
        repin_delivery_input(f, "erratum_review", review)
    before = tree_snapshot(f["root"])
    with pytest.raises((KeyError, ValueError)):
        build_current(f)
    assert tree_snapshot(f["root"]) == before


@pytest.mark.parametrize("target", ["extra-gold", "reader-question", "evaluator", "receipt"])
def test_current_delivery_contaminated_repeat_preserves_all_files(current_delivery_fixture, target):
    f = current_delivery_fixture
    build_current(f)
    path = f["output"] / {"extra-gold": "reader/gold.json", "reader-question": "reader/questions.json",
                          "evaluator": "evaluator/package.json", "receipt": "receipt.json"}[target]
    path.write_text("Contamination", encoding="utf-8")
    (f["output"] / "evaluator/cost_contracts.json").unlink()
    before = tree_snapshot(f["root"])
    with pytest.raises(ValueError, match="unexpected or contaminated"):
        build_current(f)
    assert tree_snapshot(f["root"]) == before


def test_current_delivery_output_escape_and_link_refused(current_delivery_fixture):
    f = current_delivery_fixture
    before = tree_snapshot(f["root"])
    for output in (f["root"] / "escape", f["work"] / "../escape", f["work"] / "alias./current"):
        with pytest.raises(ValueError):
            module.package_current(f["lock"], module.digest(f["lock"]), output)
    assert tree_snapshot(f["root"]) == before
    target = f["root"] / "external"
    target.mkdir()
    directory_link(f["output"], target)
    with pytest.raises(ValueError, match="unsafe delivery link"):
        build_current(f)
    assert list(target.iterdir()) == []


def test_current_delivery_portable_storage_preserves_logical_input_pins(current_delivery_fixture, monkeypatch):
    f = current_delivery_fixture
    portable = f["root"] / "portable-inputs"
    f["work"].rename(portable)
    monkeypatch.setattr(module, "WORK", portable)
    lock = portable / f["lock"].relative_to(f["work"])
    output = portable / "delivery/current"
    before = {p: p.read_bytes() for p in portable.rglob("*") if p.is_file()}
    result = module.package_current(lock, module.digest(lock), output)
    assert result["primary_questions"] == 1
    assert result["supplemental_questions"] == 1
    assert all(p.read_bytes() == raw for p, raw in before.items())
