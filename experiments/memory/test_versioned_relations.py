"""Regression checks for source-versioned exact reads and committed PEP ingestion."""

from dataclasses import replace
import json

import pytest

from experiments.memory.git_source import git_read
from experiments.memory.pep_relation_control import PepRelationCorpus, PepSourceCustody, export_source_feed, run
from experiments.memory.source_revision_feed import read_source_delta, write_source_delta
from experiments.memory.verified_source_tool import SourceSpan, sha256
from experiments.memory.versioned_relations import ExactRelationReader, Relation, RelationIndex, RelationStep


def _assertion(source, subject, role, target, raw=None):
    content = raw if raw is not None else f"{subject} {role} {target}".encode()
    digest = sha256(content)
    return Relation(subject, role, target, SourceSpan(source, digest, 0, len(content), digest))


def test_warm_composition_revisits_empty_postings_and_preserves_provenance():
    lead = _assertion("p", "a", "requires", "b")
    occupied = _assertion("p", "c", "replaces", "b", raw=b"a requires b")
    untouched = _assertion("x", "x", "status", "final")
    first = RelationIndex.empty().revise({"p": (lead, occupied), "x": (untouched,)})
    query = (RelationStep("requires"), RelationStep("replaces", reverse=True))
    reader = ExactRelationReader()

    def read(snapshot, max_edges=10):
        return reader.read("a", query, memory=snapshot, distinct_sources=True, max_edges=max_edges)

    # The old second posting contains a same-source route that cannot prove independence.
    assert read(first).status == "no_path"
    assert read(first).cached
    assert reader.read("x", (RelationStep("status"),), memory=first).paths[0].endpoint == "final"

    independent = _assertion("q", "c", "replaces", "b")
    second = first.revise({"q": (independent,)})
    assert second.updated_postings == 2
    found = read(second)
    assert not found.cached and found.cache_checks == 2
    assert len(found.paths) == 1 and found.paths[0].evidence == (lead, independent)
    assert read(second, max_edges=2).status == "work_limit"
    assert not read(second, max_edges=2).paths
    warm = reader.read("x", (RelationStep("status"),), memory=second)
    assert warm.cached and warm.edge_visits == 0 and warm.cache_checks == 1

    rephrased = _assertion("q", "c", "replaces", "b", raw=b"c now replaces b")
    third = second.revise({"q": (rephrased,)})
    assert read(third).paths[0].evidence[-1] == rephrased
    assert read(second).paths[0].evidence[-1] == independent
    assert read(third.revise({"q": None})).status == "no_path"

    # An exactly empty reverse posting must also invalidate on a later insertion.
    empty_reverse = RelationIndex.empty().revise({"p": (lead,)})
    assert read(empty_reverse).status == "no_path"
    assert read(empty_reverse).cached
    filled = empty_reverse.revise({"q": (independent,)})
    assert read(filled).paths[0].endpoint == "c"
    assert reader.read("a", query).status == "memory_omitted"


def _commit(repo, files):
    for name, raw in files.items():
        (repo / "peps" / name).write_bytes(raw)
    git_read(repo, "add", "--", "peps")
    git_read(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
             "-c", "commit.gpgsign=false", "commit", "-qm", "Source revision")
    return git_read(repo, "rev-parse", "HEAD").decode().strip()


def _init_pep_repo(tmp_path):
    git_read(tmp_path, "init", "-q")
    git_read(tmp_path, "config", "core.autocrlf", "false")
    empty_hooks = tmp_path / "empty-hooks"
    empty_hooks.mkdir()
    git_read(tmp_path, "config", "core.hooksPath", str(empty_hooks))
    (tmp_path / "peps").mkdir()


def test_pep_control_reads_committed_headers_and_revises_without_query_input(tmp_path, monkeypatch):
    _init_pep_repo(tmp_path)
    original = ("PEP: 749\r\nTitle: Résumé\r\nStatus: Draft\r\nRequires: 649,\r\n          650\r\n"
                "\r\nDraft body.\r\n\r\nRequires: 999\r\n").encode()
    first = _commit(tmp_path, {
        "pep-0749.rst": original,
        "pep-0649.rst": b"PEP: 649\nTitle: Foundation\nStatus: Accepted\n\nBase text.\n",
        "pep-0563.rst": b"PEP: 563\nTitle: Earlier\nStatus: Accepted\nSuperseded-By: 649\n\nEarlier proposal.\n",
    })
    changed = original.replace(b"Draft body.", b"The \x60\x60SOURCE\x60\x60 format is renamed to \x60\x60STRING\x60\x60.")
    second = _commit(tmp_path, {"pep-0749.rst": changed})
    third = _commit(tmp_path, {"pep-0749.rst": original, "pep-0649.rst":
                              b"PEP: 649\nTitle: Foundation\nStatus: Final\n\nBase text.\n"})

    (tmp_path / "peps" / "pep-0749.rst").write_bytes(b"Uncommitted bytes")
    opening = PepRelationCorpus.prepare(tmp_path, first)
    revised = opening.revise(second)
    rolled = revised.revise(third)
    reader = ExactRelationReader()

    requires = reader.read("pep:749", (RelationStep("requires"),), memory=opening.memory)
    assert {path.endpoint for path in requires.paths} == {"pep:649", "pep:650"}
    assert all(original[path.evidence[0].span.start_byte:path.evidence[0].span.end_byte].startswith(b"Requires:")
               for path in requires.paths)
    steps = (RelationStep("renamed_to"),)
    assert reader.read("symbol:749:SOURCE", steps, memory=opening.memory).status == "no_path"
    answer = reader.read("symbol:749:SOURCE", steps, memory=revised.memory)
    path, = answer.paths
    assert path.endpoint == "symbol:749:STRING"
    span = path.evidence[0].span
    assert span == SourceSpan(span.source_id, sha256(changed), span.start_byte, span.end_byte,
                              sha256(changed[span.start_byte:span.end_byte]))
    assert revised.verify((span,))["checked_spans"] == 1
    with pytest.raises(ValueError, match="pinned source"):
        revised.verify((replace(span, span_sha256="0" * 64),))
    with pytest.raises(ValueError, match="pinned source"):
        rolled.verify((span,))
    assert reader.read("symbol:749:SOURCE", steps, memory=rolled.memory).status == "no_path"
    assert reader.read("pep:649", (RelationStep("status"),), memory=rolled.memory).paths[0].endpoint == "status:Final"
    assert reader.read("symbol:749:SOURCE", steps, memory=revised.memory).paths == answer.paths

    route = ("pep:749", (RelationStep("requires"), RelationStep("superseded_by", True)))
    reports = run(tmp_path, (first, second, third), (route,), distinct_sources=True)
    assert [row["source_records_read"] for row in reports] == [3, 1, 2]
    direct = run(tmp_path, (first, second, third), (route,), distinct_sources=True, cache_size=0)
    assert [row["scope_sha256"] for row in reports] == [row["scope_sha256"] for row in direct]
    for report, cold in zip(reports, direct, strict=True):
        assert report["reads"][0]["paths"] == cold["reads"][0]["paths"]
        assert [item["endpoint"] for item in report["reads"][0]["paths"]] == ["pep:563"]
        assert report["verified"]["checked_spans"] == 2


    feed_root = tmp_path / "feed"
    receipts = export_source_feed(tmp_path, (first, second, third), feed_root)
    custody = json.loads((feed_root / "custody.json").read_bytes())
    source = next(key for key, locator in custody[0]["sources"].items() if locator["path"] == "peps/pep-0749.rst")
    previous = None
    verifiers = []
    for epoch, raw in enumerate((original, changed, original), 1):
        directory = feed_root / "epochs" / f"{epoch:04d}"
        assert sorted(path.name for path in directory.iterdir()) == ["manifest.json", "updates.json"]
        scope = json.loads((directory / "manifest.json").read_bytes())
        assert set(scope) == {"schema", "epoch", "parent", "changes_sha256", "documents"}
        assert set(scope["documents"]) == custody[epoch - 1]["sources"].keys()
        assert all(len(name) == 32 for name in scope["documents"])
        seal = receipts[epoch - 1]["seal"]
        state = read_source_delta(directory, seal, previous=previous)
        assert state.epoch == epoch and state.changes[source] == raw
        assert state.manifest[source].sha256 == sha256(raw) and state.manifest[source].generation == epoch
        verifiers.append(PepSourceCustody.open(tmp_path, (first, second, third)[epoch - 1],
                                              custody[epoch - 1], state))
        if epoch == 1:
            with pytest.raises(ValueError, match="committed source inventory"):
                PepSourceCustody.open(tmp_path, second, custody[0], state)
        previous = state

    old, current, restored = verifiers
    old_text = "Title: Résumé"
    start = original.decode("utf8").index(old_text)
    proof = SourceSpan.from_characters(source, original, start, start + len(old_text))
    proof_bytes = len(old_text.encode("utf8"))
    report = old.reopen(old.seal, (proof, proof), max_source_bytes=len(original), max_returned_bytes=proof_bytes)
    assert report["source_bytes_read"] == len(original)
    assert report["returned_utf8_bytes"] == proof_bytes
    assert [row["text"] for row in report["passages"]] == [old_text]
    with pytest.raises(ValueError, match="another epoch"):
        current.reopen(old.seal, (proof,), max_source_bytes=len(original), max_returned_bytes=proof_bytes)
    with pytest.raises(ValueError, match="another source version"):
        current.reopen(current.seal, (proof,), max_source_bytes=len(original), max_returned_bytes=proof_bytes)
    assert restored.reopen(restored.seal, (proof,), max_source_bytes=len(original),
                           max_returned_bytes=proof_bytes)["passages"] == report["passages"]
    with pytest.raises(ValueError, match="returned byte budget"):
        old.reopen(old.seal, (proof,), max_source_bytes=len(original), max_returned_bytes=proof_bytes - 1)

    from experiments.memory import git_source
    requested = []

    def record_git(repo, *args, **kwargs):
        requested.append(args)
        return git_read(repo, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(git_source, "git_read", record_git)
        with pytest.raises(ValueError, match="source byte budget"):
            old.reopen(old.seal, (proof,), max_source_bytes=len(original) - 1, max_returned_bytes=proof_bytes)
    assert requested == [("cat-file", "--batch-check")]
    with pytest.raises(ValueError, match="pinned source bytes"):
        old.reopen(old.seal, (replace(proof, span_sha256="0" * 64),),
                   max_source_bytes=len(original), max_returned_bytes=proof_bytes)

    with pytest.raises(ValueError, match="retained scope and parent"):
        read_source_delta(feed_root / "epochs" / "0002", receipts[1]["seal"])
    with pytest.raises(ValueError, match="manifest seal"):
        read_source_delta(feed_root / "epochs" / "0001", "0" * 64)
    first_update = feed_root / "epochs" / "0001" / "updates.json"
    first_update.write_bytes(first_update.read_bytes() + b" ")
    with pytest.raises(ValueError, match="source update bytes"):
        read_source_delta(feed_root / "epochs" / "0001", receipts[0]["seal"])


def test_source_feed_withdrawal_and_restore_require_new_scope(tmp_path):
    source = "a" * 32
    original = b"Retained manuscript."
    opening = write_source_delta(tmp_path / "first", {source: original})
    with pytest.raises(ValueError, match="repeats the current bytes"):
        write_source_delta(tmp_path / "duplicate", {source: original}, previous=opening)

    removed = write_source_delta(tmp_path / "removed", {source: None}, previous=opening)
    assert not removed.manifest
    old = read_source_delta(tmp_path / "first", opening.seal)
    new = read_source_delta(tmp_path / "removed", removed.seal, previous=old)
    assert new.changes[source] is None and not new.manifest

    restored = write_source_delta(tmp_path / "restored", {source: original}, previous=removed)
    recovered = read_source_delta(tmp_path / "restored", restored.seal, previous=new)
    assert recovered.manifest[source].sha256 == old.manifest[source].sha256
    assert (old.manifest[source].generation, recovered.manifest[source].generation) == (1, 3)


def test_git_controls_replay_withdrawal_of_entire_scope(tmp_path):
    _init_pep_repo(tmp_path)
    name = "pep-0749.rst"
    original = b"PEP: 749\nStatus: Draft\n\nDraft text.\n"
    first = _commit(tmp_path, {name: original})
    (tmp_path / "peps" / name).unlink()
    withdrawn = _commit(tmp_path, {})
    restored = _commit(tmp_path, {name: original})
    revisions = first, withdrawn, restored

    query = "pep:749", (RelationStep("status"),)
    reports = run(tmp_path, revisions, (query,))
    assert [row["sources"] for row in reports] == [1, 0, 1]
    assert [row["source_records_read"] for row in reports] == [1, 0, 1]
    assert [row["reads"][0]["status"] for row in reports] == ["matched", "no_path", "matched"]
    assert reports[2]["reads"][0]["paths"] == reports[0]["reads"][0]["paths"]

    folder = tmp_path / "withdrawal-feed"
    receipts = export_source_feed(tmp_path, revisions, folder)
    assert [row["sources"] for row in receipts] == [1, 0, 1]
    previous = None
    for epoch, receipt in enumerate(receipts, 1):
        previous = read_source_delta(folder / "epochs" / f"{epoch:04d}", receipt["seal"], previous=previous)
        if epoch == 1:
            source, = previous.manifest
        if epoch == 2:
            assert previous.changes == {source: None}
    assert previous.changes == {source: original}
    assert previous.manifest[source].generation == 3
