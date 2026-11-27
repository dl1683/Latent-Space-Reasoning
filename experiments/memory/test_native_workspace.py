from concurrent.futures import ThreadPoolExecutor, wait
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
def starting_world(tmp_path):
    cue, middle, outcome = (letter * 32 for letter in "abc")
    feed_path = tmp_path / "feed"
    source = write_source_delta(feed_path, {cue: "\u00e9cho".encode(), middle: b"blue", outcome: b"side"})
    reader, _ = _new_reader()
    memory = reader.publish(feed_path, source.seal)
    directory = tmp_path / "workspace"
    workspace = NativeWorkspace.create(directory, reader, memory, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    return workspace, directory, source, outcome, reader, memory


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


def test_same_source_workers_cannot_both_install(starting_world, monkeypatch):
    left, directory, _, _, _, _ = starting_world
    right = NativeWorkspace(directory, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    opening = left.head()
    readers = [_new_reader()[0] for _ in range(2)]
    barrier = Barrier(2)

    def synchronize(maintain):
        def wrapper(memory, **options):
            successor = maintain(memory, **options)
            barrier.wait(timeout=10)
            return successor
        return wrapper

    for reader in readers:
        monkeypatch.setattr(reader, "maintain", synchronize(reader.maintain))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(workspace.maintain, reader, opening, max_seeds=1)
                   for workspace, reader in zip((left, right), readers, strict=True)]
        wait(futures, timeout=10)
    assert sum(isinstance(future.exception(), StaleNativeWorkspace) for future in futures) == 1
    winner, = (future.result() for future in futures if future.exception() is None)
    assert winner == left.head() == right.head()
    assert winner.revision == 1 and winner.source_seal == opening.source_seal
    assert len(list((directory / "objects").iterdir())) == 2


def test_process_exit_before_and_after_commit_recovers_exact_head(starting_world):
    workspace, directory, _, _, reader, _ = starting_world
    opening = workspace.head()
    crash = """
import os
import sqlite3
import sys
from experiments.memory.native_workspace import NativeWorkspace
from experiments.memory.test_native_publication import _LIMIT, _RUNTIME, _new_reader

workspace = NativeWorkspace(sys.argv[1], runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
phase = sys.argv[2]
connect = sqlite3.connect

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
    def exit_during(phase, expected_code):
        result = subprocess.run([sys.executable, "-B", "-c", crash, str(directory), phase],
                                capture_output=True, timeout=45)
        assert result.returncode == expected_code, result.stderr.decode(errors="replace")
        return NativeWorkspace(directory, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)

    before = exit_during("before", 37)
    assert before.head() == opening
    with pytest.raises(ValueError, match="not committed"):
        before.pin_at(1)
    after = exit_during("after", 38)
    committed = after.pin_at(1)
    assert committed == after.head()
    assert after.attach(reader, pin=committed)._publication_seal == committed.publication_seal
    assert reader("What follows?", memory=after.attach(reader, pin=opening)).text == ""
    assert reader("What follows?", memory=after.attach(reader, pin=committed)).text == "side"
    with closing(sqlite3.connect(directory / "publications.sqlite3")) as db:
        assert db.execute("SELECT parent_seal, operation FROM native_publications ORDER BY revision").fetchall() == [
            (None, "open"), (opening.publication_seal, "maintain"),
        ]
