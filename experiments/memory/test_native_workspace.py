from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
import json
import sqlite3
import subprocess
import sys
from threading import Barrier, Event

import pytest

from experiments.memory.native_workspace import NativeWorkspace, StaleNativeWorkspace
from experiments.memory.source_revision_feed import write_source_delta
from experiments.memory.test_native_publication import _LIMIT, _RUNTIME, _new_reader


@pytest.fixture
def starting_world(tmp_path, request):
    cue, middle, outcome = (letter * 32 for letter in "abc")
    feed_path = tmp_path / "feed"
    source = write_source_delta(feed_path, {cue: "\u00e9cho".encode(), middle: b"blue", outcome: b"side"})
    reader, _ = _new_reader()
    memory = reader.publish(feed_path, source.seal)
    directory = tmp_path / "workspace"
    workspace = NativeWorkspace.create(directory, reader, memory, runtime_sha256=_RUNTIME, max_bytes=_LIMIT,
                                       share_rows=getattr(request, "param", False))
    return workspace, directory, source, outcome, reader, memory


@pytest.mark.parametrize("starting_world", [False, True], indirect=True)
def test_workers_share_history_without_overwriting_a_source_correction(starting_world, tmp_path, monkeypatch):
    left, directory, source, outcome, _, _ = starting_world
    right = NativeWorkspace(directory, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    first_reader, _ = _new_reader()
    next_reader, next_backbone = _new_reader()
    opening = left.head()
    maintained = left.maintain(first_reader, opening, max_seeds=1)
    assert maintained.revision == 1 and maintained.source_seal == opening.source_seal == source.seal
    assert maintained.publication_seal != opening.publication_seal
    assert right.head() == maintained
    with pytest.raises(StaleNativeWorkspace, match="head changed"):
        right.maintain(next_reader, opening, max_seeds=1)
    assert next_backbone.calls == []

    correction_path = tmp_path / "withdrawal"
    correction = write_source_delta(correction_path, {outcome: None}, previous=source)
    computed = Event()
    release = Event()
    normal_maintain = first_reader.maintain

    def delay_after_build(memory, **options):
        proposal = normal_maintain(memory, **options)
        assert proposal is not memory
        computed.set()
        assert release.wait(10)
        return proposal

    monkeypatch.setattr(first_reader, "maintain", delay_after_build)
    with ThreadPoolExecutor(max_workers=1) as pool:
        losing_work = pool.submit(left.maintain, first_reader, maintained, max_seeds=1)
        try:
            assert computed.wait(10)
            revised = right.revise(next_reader, maintained, correction_path, correction.seal, max_source_bytes=_LIMIT)
        finally:
            release.set()
        with pytest.raises(StaleNativeWorkspace, match="head changed"):
            losing_work.result(timeout=10)

    assert revised.revision == 2 and revised.source_seal == correction.seal
    assert left.head() == revised
    assert {item.name for item in (directory / "objects").iterdir()} == {
        opening.publication_seal, maintained.publication_seal, revised.publication_seal,
    }
    with pytest.raises(ValueError, match="not committed"):
        right.attach(next_reader, pin=replace(maintained, source_seal=correction.seal))

    late_reader, late_backbone = _new_reader()
    historical = right.attach(late_reader, pin=maintained)
    current = right.attach(late_reader)
    base = right.attach(late_reader, pin=opening)
    assert late_backbone.calls == []
    task = "What follows?"
    historical_read = late_reader(task, memory=historical)
    assert historical_read.text == "side"
    assert late_reader(task, memory=base).text == ""
    assert late_reader(task, memory=late_reader.without_derived(historical)).text == ""
    assert late_reader(task).text == ""
    assert late_reader(task, memory=current).text == ""
    assert late_reader(task, memory=right.attach(late_reader, pin=maintained)) == historical_read

    script = """
import json
import sys
from experiments.memory.native_workspace import NativeWorkspace
from experiments.memory.test_native_publication import _LIMIT, _RUNTIME, _new_reader

workspace = NativeWorkspace(sys.argv[1], runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
reader, backbone = _new_reader()
pin = workspace.head()
memory = workspace.attach(reader, pin=pin)
historical = workspace.attach(reader, pin=workspace.pin_at(1))
assert not backbone.calls
print(json.dumps([pin.revision, memory.seal, reader("What follows?", memory=historical).text,
                  reader("What follows?", memory=memory).text]))
"""
    fresh = subprocess.check_output([sys.executable, "-B", "-c", script, str(directory)], timeout=45)
    assert json.loads(fresh) == [2, correction.seal, "side", ""]


def test_install_refusal_keeps_history_and_reverifies_orphan(starting_world, tmp_path):
    workspace, directory, _, _, reader, memory = starting_world
    opening = workspace.head()
    foreign = NativeWorkspace.create(tmp_path / "foreign", reader, memory, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    assert foreign.head().publication_seal == opening.publication_seal
    with pytest.raises(ValueError, match="not committed"):
        foreign.attach(reader, pin=opening)

    with closing(sqlite3.connect(directory / "publications.sqlite3")) as db, db:
        db.execute("""CREATE TRIGGER refuse BEFORE INSERT ON native_publications
                      BEGIN SELECT RAISE(ABORT, 'injected install refusal'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="injected install refusal"):
        workspace.maintain(reader, opening, max_seeds=1)
    assert workspace.head() == opening
    orphan, = (path for path in (directory / "objects").iterdir() if path.name != opening.publication_seal)
    with pytest.raises(ValueError, match="not committed"):
        workspace.attach(reader, pin=replace(opening, revision=1, publication_seal=orphan.name))
    payload = orphan / "rows.bin"
    preserved = payload.read_bytes()
    payload.write_bytes(b"corrupted")
    with closing(sqlite3.connect(directory / "publications.sqlite3")) as db, db:
        db.execute("DROP TRIGGER refuse")
    with pytest.raises(ValueError, match="numeric bytes differ"):
        workspace.maintain(reader, opening, max_seeds=1)
    assert workspace.head() == opening

    payload.write_bytes(preserved)
    successor = workspace.maintain(reader, opening, max_seeds=1)
    assert successor.revision == 1 and successor.publication_seal == orphan.name
    assert workspace.attach(reader)._publication_seal == successor.publication_seal

def test_successor_preserves_opening_reader(starting_world, monkeypatch):
    import torch

    workspace, directory, _, _, _, _ = starting_world
    opening = workspace.head()
    changed_reader, model = _new_reader()
    maintain = changed_reader.maintain

    def drift(memory, **options):
        candidate = maintain(memory, **options)
        with torch.no_grad():
            model.get_input_embeddings().weight[0, 0] += 0.125
        return candidate

    monkeypatch.setattr(changed_reader, "maintain", drift)
    with pytest.raises(ValueError, match="changed the opening reader"):
        workspace.maintain(changed_reader, opening, max_seeds=1)
    assert workspace.head() == opening
    assert [item.name for item in (directory / "objects").iterdir()] == [opening.publication_seal]
    compatible, _ = _new_reader()
    assert workspace.attach(compatible, pin=opening)._publication_seal == opening.publication_seal


def test_first_fixed_source_keeps_fast_tokenizer_reader_compatible(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    def make_reader():
        reader, _ = _new_reader()
        backend = tokenizers.Tokenizer(
            tokenizers.models.WordLevel({"[PAD]": 0, "[UNK]": 1, "blue": 2}, unk_token="[UNK]"),
        )
        tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
        tokenizer.backend_tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=4)
        tokenizer.backend_tokenizer.enable_truncation(max_length=5)
        reader._tokenizer = tokenizer
        return reader

    initial_dir = tmp_path / "empty"
    initial = write_source_delta(initial_dir, {})
    reader = make_reader()
    backend = reader._tokenizer.backend_tokenizer
    settings = backend.padding, backend.truncation
    memory = reader.publish(initial_dir, initial.seal)
    root = tmp_path / "workspace"
    workspace = NativeWorkspace.create(root, reader, memory, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    opening = workspace.head()
    revision_dir = tmp_path / "first"
    feed = write_source_delta(revision_dir, {"a" * 32: b"blue"}, previous=initial)
    revised = workspace.revise(reader, opening, revision_dir, feed.seal, max_source_bytes=_LIMIT)
    assert (backend.padding, backend.truncation) == settings
    fresh = make_reader()
    assert workspace.attach(fresh, pin=opening).rows == 0
    assert workspace.attach(fresh, pin=revised).rows == 1


@pytest.mark.parametrize("starting_world", [False, True], indirect=True)
def test_same_source_workers_cannot_both_install(starting_world, monkeypatch):
    left, directory, _, _, _, _ = starting_world
    right = NativeWorkspace(directory, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    opening = left.head()
    readers = [_new_reader()[0] for _ in range(2)]
    barrier = Barrier(2)
    release = Event()

    def synchronize(maintain):
        def wrapper(memory, **options):
            successor = maintain(memory, **options)
            barrier.wait(timeout=10)
            if options["max_seeds"] == 2:
                assert release.wait(10)
            return successor
        return wrapper

    for reader in readers:
        monkeypatch.setattr(reader, "maintain", synchronize(reader.maintain))
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(left.maintain, readers[0], opening, max_seeds=1)
        losing = pool.submit(right.maintain, readers[1], opening, max_seeds=2)
        try:
            winner = first.result(timeout=10)
        finally:
            release.set()
        with pytest.raises(StaleNativeWorkspace, match="head changed"):
            losing.result(timeout=10)
    assert winner == left.head() == right.head()
    assert winner.revision == 1 and winner.source_seal == opening.source_seal
    assert len(list((directory / "objects").iterdir())) == 2
    blocks = directory / "numeric-blocks"
    if blocks.is_dir():
        assert sorted(item.stat().st_size for item in blocks.iterdir()) == [51, 51, 51, 96]


@pytest.mark.parametrize("starting_world", [False, True], indirect=True)
def test_process_exit_before_and_after_commit_recovers_exact_head(starting_world, tmp_path):
    workspace, directory, _, _, reader, memory = starting_world
    opening = workspace.head()
    crash = """
import os
from pathlib import Path
import sqlite3
import sys
from experiments.memory.native_workspace import NativeWorkspace
from experiments.memory.test_native_publication import _LIMIT, _RUNTIME, _new_reader

workspace = NativeWorkspace(sys.argv[1], runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
phase = sys.argv[2]
connect = sqlite3.connect
rename = Path.rename

def rename_for_test(self, destination):
    result = rename(self, destination)
    if phase == "block" and result.parent.name == "numeric-blocks":
        os._exit(39)
    return result

Path.rename = rename_for_test

class Interrupt(sqlite3.Connection):
    def execute(self, sql, parameters=()):
        if phase == "before" and sql.startswith("INSERT INTO native_publications"):
            os._exit(37)
        result = super().execute(sql, parameters)
        if phase == "after" and sql == "COMMIT":
            os._exit(38)
        return result

def connect_for_test(*args, **kwargs):
    return connect(*args, factory=Interrupt, **kwargs)

sqlite3.connect = connect_for_test
reader, _ = _new_reader()
workspace.maintain(reader, workspace.head(), max_seeds=1)
"""
    def exit_during(root, phase, expected_code):
        result = subprocess.run([sys.executable, "-B", "-c", crash, str(root), phase],
                                capture_output=True, timeout=45)
        assert result.returncode == expected_code, result.stderr.decode(errors="replace")
        return NativeWorkspace(root, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)

    before = exit_during(directory, "before", 37)
    assert before.head() == opening
    with pytest.raises(ValueError, match="not committed"):
        before.pin_at(1)
    after = exit_during(directory, "after", 38)
    committed = after.pin_at(1)
    assert committed == after.head()
    assert after.attach(reader, pin=committed)._publication_seal == committed.publication_seal
    assert reader("What follows?", memory=after.attach(reader, pin=opening)).text == ""
    assert reader("What follows?", memory=after.attach(reader, pin=committed)).text == "side"
    with closing(sqlite3.connect(directory / "publications.sqlite3")) as db:
        assert db.execute("SELECT parent_seal, operation FROM native_publications ORDER BY revision").fetchall() == [
            (None, "open"), (opening.publication_seal, "maintain"),
        ]

    if (directory / "numeric-blocks").is_dir():
        new_root = tmp_path / "fresh-shared"
        NativeWorkspace.create(new_root, reader, memory, runtime_sha256=_RUNTIME,
                               max_bytes=_LIMIT, share_rows=True)
        fresh_blocks = new_root / "numeric-blocks"
        assert len(list(fresh_blocks.iterdir())) == 3
        committed_without_orphan = exit_during(new_root, "after", 38)
        assert committed_without_orphan.head().publication_seal == committed.publication_seal
        assert len(list(fresh_blocks.iterdir())) == 4
        assert reader("What follows?", memory=committed_without_orphan.attach(reader)).text == "side"

        block_root = tmp_path / "block-orphan"
        empty_orphan_case = NativeWorkspace.create(block_root, reader, memory, runtime_sha256=_RUNTIME,
                                                   max_bytes=_LIMIT, share_rows=True)
        current = empty_orphan_case.head()
        interrupted = exit_during(block_root, "block", 39)
        assert interrupted.head() == current
        assert {path.name for path in (block_root / "objects").iterdir() if not path.name.startswith(".")} == {
            current.publication_seal,
        }
        assert len(list((block_root / "numeric-blocks").iterdir())) == 4
        recovered = interrupted.maintain(reader, current, max_seeds=1)
        assert recovered.publication_seal == committed.publication_seal


def test_shared_rows_keep_exact_publications_and_record_retained_bytes(tmp_path, monkeypatch):
    from zlib import compress

    from experiments.memory.native_isolated_stage import _CausalControl, _reader
    from experiments.memory.native_workspace_rows import SharedNativeRows

    source_ids = [f"{number:032x}" for number in range(1, 9)]
    bodies = (b"link L03 L04", b"link L08 L10", b"link L02 L03", b"link L01 L02",
              b"link L04 L05", b"name L05 L06", b"name L10 L11", b"name L10 L11")
    feed_path = tmp_path / "feed"
    initial = write_source_delta(feed_path, dict(zip(source_ids, bodies, strict=True)))
    opening_id = "0" * 32
    insertion = tmp_path / "insert"
    inserted = write_source_delta(insertion, {opening_id: b"name L11 L09"}, previous=initial)
    corrected = tmp_path / "corrected"
    named_outcome = source_ids[5]
    withdrawn = write_source_delta(corrected, {named_outcome: None}, previous=inserted)
    recovery = tmp_path / "restored"
    restored = write_source_delta(recovery, {named_outcome: bodies[5]}, previous=withdrawn)

    staged_numeric = []
    original_write = SharedNativeRows._write_bytes

    def count_write(handle, raw):
        staged_numeric.append(len(raw))
        original_write(handle, raw)

    monkeypatch.setattr(SharedNativeRows, "_write_bytes", staticmethod(count_write))
    directories = [tmp_path / name for name in ("flat", "shared")]
    readers = [_reader(_CausalControl().eval()) for _ in directories]
    workspaces = []
    for directory, reader, share in zip(directories, readers, (False, True), strict=True):
        memory = reader.publish(feed_path, initial.seal)
        workspaces.append(NativeWorkspace.create(directory, reader, memory, runtime_sha256=_RUNTIME,
                                                 max_bytes=_LIMIT, share_rows=share))
    for seeds, scores in ((2, 1), (2, 4), (1, 6)):
        for workspace, reader in zip(workspaces, readers, strict=True):
            workspace.maintain(reader, workspace.head(), max_seeds=seeds, max_parent_scores=scores)
    for directory, revision in ((insertion, inserted), (corrected, withdrawn), (recovery, restored)):
        before_write = sum(staged_numeric)
        for workspace, reader in zip(workspaces, readers, strict=True):
            workspace.revise(reader, workspace.head(), directory, revision.seal, max_source_bytes=_LIMIT)
        if revision == restored:
            assert sum(staged_numeric) == before_write
    for workspace, reader in zip(workspaces, readers, strict=True):
        workspace.maintain(reader, workspace.head(), max_seeds=8, max_parent_scores=40)

    flat, shared = workspaces
    flat_dir, shared_dir = directories
    shared = NativeWorkspace(shared_dir, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    shared_rows = SharedNativeRows(shared_dir / "numeric-blocks", _LIMIT)
    assert shared.head().revision == flat.head().revision == 7
    logical_bytes = index_bytes = compressed_bytes = manifest_bytes = 0
    historical_outputs = []
    for revision in range(8):
        left, right = flat.pin_at(revision), shared.pin_at(revision)
        assert left.source_seal == right.source_seal and left.publication_seal == right.publication_seal
        left_path = flat_dir / "objects" / left.publication_seal
        right_path = shared_dir / "objects" / right.publication_seal
        manifest = (left_path / "manifest.json").read_bytes()
        raw = (left_path / "rows.bin").read_bytes()
        assert (right_path / "manifest.json").read_bytes() == manifest
        assert shared_rows.read_rows(right_path, len(raw)) == raw
        manifest_bytes += len(manifest)
        logical_bytes += len(raw)
        compressed_bytes += len(compress(raw))
        index_bytes += (right_path / "rows.refs").stat().st_size
        records = []
        for workspace, pin in ((flat, left), (shared, right)):
            reader = _reader(_CausalControl().eval())
            memory = workspace.attach(reader, pin=pin)
            assert set(memory._sources) >= set(source_ids) - {named_outcome}
            assert source_ids[6] in memory._sources and source_ids[7] in memory._sources
            records.append(reader("follow L01", memory=memory).json_record())
        assert records[0] == records[1]
        historical_outputs.append(records[0]["text"])
    assert historical_outputs[:4] == ["", "", "", "L06"]
    assert historical_outputs[4] == "L06" and historical_outputs[5] == ""
    assert historical_outputs[-1] == "L06"

    stored_bytes = sum(path.stat().st_size for path in (shared_dir / "numeric-blocks").iterdir())
    assert sum(staged_numeric) == stored_bytes
    assert stored_bytes + index_bytes < logical_bytes
    print(f"native-history bytes: logical={logical_bytes}, shared={stored_bytes}, refs={index_bytes}, "
          f"zlib={compressed_bytes}, common_manifests={manifest_bytes}")


@pytest.mark.parametrize("starting_world", [True], indirect=True)
def test_shared_rows_detect_corruption_and_reordered_references(starting_world):
    from experiments.memory.native_workspace_rows import _HEADER, _REFERENCE

    workspace, directory, _, _, reader, _ = starting_world
    opening = workspace.head()
    index = directory / "objects" / opening.publication_seal / "rows.refs"
    valid = index.read_bytes()
    entries = list(_REFERENCE.iter_unpack(valid[len(_HEADER):]))
    first_digest, first_size = entries[0]
    assert all(length == first_size for _, length in entries)
    assert entries[0] != entries[1]
    manifest_size = (index.parent / "manifest.json").stat().st_size
    budget = manifest_size + sum(length for _, length in entries) - 1
    too_small = NativeWorkspace(directory, runtime_sha256=_RUNTIME, max_bytes=budget)
    with pytest.raises(ValueError, match="exceeds its byte allowance"):
        too_small.attach(reader)
    swapped = [entries[1], entries[0], *entries[2:]]
    index.write_bytes(_HEADER + b"".join(_REFERENCE.pack(*entry) for entry in swapped))
    with pytest.raises(ValueError, match="numeric bytes differ"):
        workspace.attach(reader)
    index.write_bytes(_HEADER + _REFERENCE.pack(first_digest, first_size + 1) + valid[len(_HEADER) + _REFERENCE.size:])
    with pytest.raises(ValueError, match="reference sizes differ"):
        workspace.attach(reader)
    index.write_bytes(valid)

    block = directory / "numeric-blocks" / first_digest.hex()
    original = block.read_bytes()
    block.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="shared numeric block differs"):
        workspace.attach(reader)
    block.rename(block.with_suffix(".fixture"))
    with pytest.raises(FileNotFoundError):
        workspace.attach(reader)


def test_empty_shared_workspace_cannot_fall_back_to_flat(tmp_path):
    feed = tmp_path / "empty"
    opening = write_source_delta(feed, {})
    reader, _ = _new_reader()
    memory = reader.publish(feed, opening.seal)
    root = tmp_path / "workspace"
    workspace = NativeWorkspace.create(root, reader, memory, runtime_sha256=_RUNTIME,
                                       max_bytes=_LIMIT, share_rows=True)
    assert workspace.attach(reader).rows == 0
    (root / "numeric-blocks").rename(root / "fixture-hidden-blocks")
    with pytest.raises(ValueError, match="shared numeric store is missing"):
        NativeWorkspace(root, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    with pytest.raises(ValueError, match="shared numeric store is missing"):
        workspace.attach(reader)
