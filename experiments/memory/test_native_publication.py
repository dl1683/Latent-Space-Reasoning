from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.memory.native_publication import read_native_publication, write_native_publication
from experiments.memory.source_revision_feed import _encode, write_source_delta
from experiments.memory.test_native_latent_recurrence import _reader, _StandIn
from experiments.memory.verified_source_tool import sha256


_RUNTIME = sha256(b"controlled fixture runtime")
_LIMIT = 64 * 1024


def _new_reader(*, top_k=1, clear_previous=True, tie_limit=16):
    backbone = _StandIn(clear_previous=clear_previous).eval()
    return _reader(backbone, terminal_axis=2, top_k=top_k, tie_limit=tie_limit), backbone


def _world(root):
    cue, middle, outcome, inactive, empty = (letter * 32 for letter in "abcde")
    old = write_source_delta(root / "feed", {
        cue: "écho".encode(), middle: b"blue", outcome: b"side", inactive: b".", empty: b"",
    })
    reader, _ = _new_reader()
    warmed = reader.maintain(reader.publish(root / "feed", old.seal), max_seeds=1)
    assert warmed.derived_seeds == warmed.source_spans(cue)
    seal = write_native_publication(root / "publication", reader, warmed, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    return reader, warmed, old, seal, cue, inactive


def _read(directory, reader, seal, source):
    return read_native_publication(directory, reader, seal=seal, source_seal=source,
                                   runtime_sha256=_RUNTIME, max_bytes=_LIMIT)


def _packet(read):
    packet = asdict(read)
    packet["omitted_sources"] = sorted(read.omitted_sources)
    return packet


def _publish_child(directory, source, destination):
    reader, _ = _new_reader()
    memory = reader.maintain(reader.publish(directory, source), max_seeds=1)
    return write_native_publication(destination, reader, memory, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)


def _child(directory, seal, source, question_file):
    reader, backbone = _new_reader()
    memory = _read(directory, reader, seal, source)
    assert not backbone.calls
    question = Path(question_file).read_text(encoding="utf8")
    return [_packet(reader(question, memory=memory)),
            _packet(reader(question, memory=reader.without_derived(memory))), _packet(reader(question))]


def test_fresh_process_attaches_maintained_and_source_only_arms(tmp_path):
    reader, warmed, old, seal, _, _ = _world(tmp_path)
    replica = tmp_path / "replica"
    publisher = subprocess.check_output([
        sys.executable, "-m", "experiments.memory.test_native_publication",
        "publish", str(tmp_path / "feed"), old.seal, str(replica),
    ], timeout=60)
    assert json.loads(publisher) == seal
    assert (tmp_path / "publication" / "rows.bin").read_bytes() == (tmp_path / "replica" / "rows.bin").read_bytes()

    question = "What follows?"
    path = tmp_path / "late-question.txt"
    path.write_text(question, encoding="utf8")
    result = subprocess.check_output([
        sys.executable, "-m", "experiments.memory.test_native_publication",
        "read", str(replica), seal, old.seal, str(path),
    ], timeout=60)
    expected = [_packet(reader(question, memory=warmed)),
                _packet(reader(question, memory=reader.without_derived(warmed))), _packet(reader(question))]
    expected[0]["publication_seal"] = expected[1]["publication_seal"] = seal
    assert json.loads(result) == json.loads(json.dumps(expected))
    assert [packet["text"] for packet in expected] == ["side", "", ""]
    assert [packet["derived_omitted"] for packet in expected] == [False, True, False]


def test_reopened_worker_reuses_its_attempts_on_a_quiet_revision(tmp_path):
    reader, warmed, old, seal, cue, inactive = _world(tmp_path)
    fresh, backbone = _new_reader()
    reopened = _read(tmp_path / "publication", fresh, seal, old.seal)
    assert backbone.calls == []
    assert reopened.derived_seeds == warmed.derived_seeds
    assert reopened.pending_seeds == warmed.pending_seeds
    assert write_native_publication(tmp_path / "round-trip", fresh, reopened, runtime_sha256=_RUNTIME,
                                    max_bytes=_LIMIT) == seal
    with pytest.raises(ValueError, match="intact selective world"):
        write_native_publication(tmp_path / "ablation", fresh, fresh.without_derived(reopened),
                                 runtime_sha256=_RUNTIME, max_bytes=_LIMIT)

    revision = write_source_delta(tmp_path / "revision", {inactive: b" "}, previous=old)
    revised_original = reader.publish(tmp_path / "revision", revision.seal, previous=warmed)
    revised_reopened = fresh.publish(tmp_path / "revision", revision.seal, previous=reopened)
    assert backbone.calls == [("backbone", False, 1)]
    seed = warmed.source_spans(cue)[0]
    assert revised_reopened.continuation_reads(seed) == revised_original.continuation_reads(seed)
    assert fresh("What follows?", memory=revised_reopened) == reader("What follows?", memory=revised_original)

    maintained = fresh.maintain(revised_reopened, max_seeds=1)
    assert maintained.pending_seeds == revised_reopened.pending_seeds - 1
    updated = tmp_path / "updated"
    new_seal = write_native_publication(updated, fresh, maintained, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    later, later_backbone = _new_reader()
    reattached = _read(updated, later, new_seal, revision.seal)
    assert not later_backbone.calls
    assert reattached.continuation_reads(seed)[0].observed_source_seal == old.seal
    assert reattached.pending_seeds == maintained.pending_seeds


def test_publication_checks_scope_runtime_tensor_and_payload(tmp_path):
    _, _, old, seal, _, _ = _world(tmp_path)
    root = tmp_path / "publication"
    reader, backbone = _new_reader()
    with pytest.raises(ValueError, match="source or runtime differs"):
        _read(root, reader, seal, "1" * 64)
    with pytest.raises(ValueError, match="source or runtime differs"):
        read_native_publication(root, reader, seal=seal, source_seal=old.seal,
                                runtime_sha256="1" * 64, max_bytes=_LIMIT)
    backbone.get_input_embeddings().weight[0, 0] += 0.25
    with pytest.raises(ValueError, match="reader differs"):
        _read(root, reader, seal, old.seal)
    changed_policy, changed_backbone = _new_reader(clear_previous=False)
    with pytest.raises(ValueError, match="reader differs"):
        _read(root, changed_policy, seal, old.seal)
    assert not changed_backbone.calls
    reader, backbone = _new_reader()
    with pytest.raises(ValueError, match="byte allowance"):
        read_native_publication(root, reader, seal=seal, source_seal=old.seal,
                                runtime_sha256=_RUNTIME, max_bytes=1)
    binary = root / "rows.bin"
    damaged = bytearray(binary.read_bytes())
    damaged[-1] ^= 1
    binary.write_bytes(damaged)
    with pytest.raises(ValueError, match="numeric bytes differ"):
        _read(root, reader, seal, old.seal)
    assert not backbone.calls


def test_custody_digest_does_not_replace_source_descriptor_validation(tmp_path):
    _, _, old, _, _, _ = _world(tmp_path)
    root = tmp_path / "publication"
    file = root / "manifest.json"
    header = json.loads(file.read_bytes())
    header["source_manifest"] = "{}"
    changed = _encode(header)
    file.write_bytes(changed)
    reader, backbone = _new_reader()
    with pytest.raises(ValueError, match="source descriptor differs"):
        _read(root, reader, sha256(changed), old.seal)
    assert not backbone.calls


@pytest.mark.parametrize(("damage", "message"), [
    ("score", "publication witness is invalid"), ("cutoff", "frontier cutoff or scope differs"),
    ("scope", "attempt source scope differs"), ("competition", "competition status differs"),
])
def test_reopened_frontiers_are_checked_against_packed_keys(tmp_path, damage, message):
    _, _, old, _, _, _ = _world(tmp_path)
    root = tmp_path / "publication"
    file = root / "manifest.json"
    header = json.loads(file.read_bytes())
    step = header["attempted"][0][1][0]
    if damage == "score":
        step[3][0][1] -= 1
    elif damage == "scope":
        step[7] = "1" * 64
        step[2] = 1
    elif damage == "competition":
        step[0] = "competing"
        step[6] = False
    if damage in ("score", "cutoff"):
        step[5] = 1
    changed = _encode(header)
    file.write_bytes(changed)
    reader, backbone = _new_reader()
    with pytest.raises(ValueError, match=message):
        _read(root, reader, sha256(changed), old.seal)
    assert not backbone.calls


def test_an_overflow_attempt_cannot_be_restored_across_an_epoch(tmp_path):
    cue, middle, result, tie, inactive = (letter * 32 for letter in "abcde")
    old = write_source_delta(tmp_path / "feed", {
        cue: "écho".encode(), middle: b"blue", result: b"side", tie: b"gold", inactive: b".",
    })
    reader, _ = _new_reader(tie_limit=1)
    initial = reader.maintain(reader.publish(tmp_path / "feed", old.seal), max_seeds=1)
    original = tmp_path / "publication"
    original_seal = write_native_publication(original, reader, initial, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    first_header = json.loads((original / "manifest.json").read_bytes())
    assert first_header["attempted"][0][1][-1][0] == "overflow"
    assert _read(original, reader, original_seal, old.seal).pending_seeds == 4

    delta = write_source_delta(tmp_path / "revision", {inactive: b" "}, previous=old)
    revised = reader.publish(tmp_path / "revision", delta.seal, previous=initial)
    assert revised.pending_seeds == 5
    updated = tmp_path / "updated"
    write_native_publication(updated, reader, revised, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    file = updated / "manifest.json"
    header = json.loads(file.read_bytes())
    header["attempted"] = first_header["attempted"]
    header["queue"].remove(cue)
    changed = _encode(header)
    file.write_bytes(changed)
    with pytest.raises(ValueError, match="carries an overflow from an older epoch"):
        _read(updated, reader, sha256(changed), delta.seal)


def test_reopened_workers_build_on_each_other_and_withdraw_dependents(tmp_path):
    from experiments.memory.native_isolated_stage import _CausalControl, _reader as make_reader

    # The first two source steps cannot reach the answer from the late cue.
    root, relay, cue, bridge, outcome, idle = sources = tuple(letter * 32 for letter in "abcdef")
    feed = write_source_delta(tmp_path / "opening", dict(zip(sources, (
        b"link L03 L04", b"link L02 L03", b"link L01 L02", b"link L04 L05",
        b"name L05 L06", b"name L08 L09",
    ), strict=True)))
    reader = make_reader(_CausalControl().eval())
    base = reader.publish(tmp_path / "opening", feed.seal)
    root_seed, = base.source_spans(root)
    relay_seed, = base.source_spans(relay)
    cue_seed, = base.source_spans(cue)

    first = reader.maintain(base, max_seeds=1)
    with pytest.raises(ValueError, match="parent must be an installed continuation"):
        reader.maintain(first, max_seeds=1, parent=cue_seed)

    opened = tmp_path / "one-worker"
    seal = write_native_publication(opened, reader, first, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    model = _CausalControl().eval()
    worker = make_reader(model)
    recovered = _read(opened, worker, seal, feed.seal)
    assert model.calls == 0
    second = worker.maintain(recovered, max_seeds=1, parent=root_seed)
    composed = worker.maintain(second, max_seeds=1, parent=relay_seed)
    assert model.calls == 6
    assert composed.derived_seeds == (root_seed, relay_seed, cue_seed)
    assert [composed.continuation_parent(seed) for seed in composed.derived_seeds] == [None, root_seed, relay_seed]
    assert bridge not in {hit.span.source_id for step in composed.continuation_reads(cue_seed)
                          for hit in step.selected}

    calls = model.calls
    source_only = worker.maintain(recovered, max_seeds=2)
    assert model.calls - calls == 6
    active_shard, = composed._selective.derived
    control_shard, = source_only._selective.derived
    active_payload = active_shard.values[active_shard.spans.index(cue_seed)].tobytes()
    control_payload = control_shard.values[control_shard.spans.index(cue_seed)].tobytes()
    assert active_payload != control_payload
    assert worker("follow L01", memory=source_only).text == ""

    published = tmp_path / "three-workers"
    seal = write_native_publication(published, worker, composed, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    later_model = _CausalControl().eval()
    later = make_reader(later_model)
    restored = _read(published, later, seal, feed.seal)
    assert later_model.calls == 0
    assert restored.continuation_parent(cue_seed) == relay_seed
    assert write_native_publication(tmp_path / "replay", later, restored, runtime_sha256=_RUNTIME,
                                    max_bytes=_LIMIT) == seal
    read = later("follow L01", memory=restored)
    assert read.text == "L06" and read.terminated and read.applied_derived_steps == 1
    proof = restored.evidence_spans(read)
    assert {span.source_id for span in proof} == {root, relay, cue, bridge, outcome}
    assert len(proof) == 5
    ablated = later.without_derived(restored)
    removed = later("follow L01", memory=ablated)
    assert removed.text == ""
    assert {span.source_id for span in ablated.evidence_spans(removed)} == {cue, relay}
    with pytest.raises(ValueError, match="matching reopened publication"):
        restored.evidence_spans(removed)
    with pytest.raises(ValueError, match="matching reopened publication"):
        restored.evidence_spans(replace(read, publication_seal="0" * 64))
    assert later("follow L01", memory=restored) == read

    quiet = write_source_delta(tmp_path / "quiet", {idle: b"name L08 L10"}, previous=feed)
    stable = later.publish(tmp_path / "quiet", quiet.seal, previous=restored)
    assert stable.continuation_parent(cue_seed) == relay_seed
    assert later("follow L01", memory=stable).text == "L06"
    withdrawal = write_source_delta(tmp_path / "withdrawal", {bridge: None}, previous=quiet)
    revoked = later.publish(tmp_path / "withdrawal", withdrawal.seal, previous=stable)
    assert revoked.pending_seeds == 5 and revoked.derived_seeds == ()
    assert later("follow L01", memory=revoked).text == ""
    assert later("follow L01", memory=restored) == read

    manifest = published / "manifest.json"
    original = json.loads(manifest.read_bytes())
    for damage, message in (("payload", "parent payload differs"),
                            ("self", "parent must precede"),
                            ("descendant", "parent must precede"),
                            ("unreachable", "parent is unreachable")):
        header = json.loads(json.dumps(original))
        edge = header["parents"][0]
        if damage == "payload":
            edge[2] = "0" * 64
        elif damage == "self":
            edge[1] = edge[0]
        elif damage == "descendant":
            edge[1] = header["parents"][1][0]
        else:
            header["parents"][1][1:] = edge[1:]
        altered = _encode(header)
        manifest.write_bytes(altered)
        with pytest.raises(ValueError, match=message):
            _read(published, later, sha256(altered), feed.seal)


if __name__ == "__main__":
    run = {"publish": _publish_child, "read": _child}[sys.argv[1]]
    print(json.dumps(run(*sys.argv[2:])))
