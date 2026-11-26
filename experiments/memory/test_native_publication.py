from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.memory.native_publication import read_native_publication, write_native_publication
from experiments.memory.source_revision_feed import _encode, read_source_delta, write_source_delta
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


def _publish_child(directory, source, destination):
    reader, _ = _new_reader()
    memory = reader.maintain(reader.publish(directory, source), max_seeds=1)
    return write_native_publication(destination, reader, memory, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)


def _child(directory, seal, source, question_file):
    reader, backbone = _new_reader()
    memory = _read(directory, reader, seal, source)
    assert not backbone.calls
    question = Path(question_file).read_text(encoding="utf8")
    return [reader(question, memory=memory).json_record(),
            reader(question, memory=reader.without_derived(memory)).json_record(),
            reader(question).json_record()]


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
    expected = [reader(question, memory=warmed).json_record(),
                reader(question, memory=reader.without_derived(warmed)).json_record(),
                reader(question).json_record()]
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


def test_binary_media_supplies_a_necessary_bridge_to_the_same_reader(tmp_path):
    from experiments.memory.native_isolated_stage import LABELS, _CausalControl, _reader as make_reader

    cue, image, red_name, blue_name = (letter * 32 for letter in "abcd")
    header = b"P6\n1 1\n255\n"
    red = header + bytes((255, 0, 0))
    blue = header + bytes((0, 0, 100))
    blue.decode("utf8")
    media_calls = []

    def projector(red_edge, blue_edge):
        spec = dict(format="P6-single-pixel", address=1, red=red_edge, blue=blue_edge)

        def project(source, raw):
            if not raw.startswith(header) or len(raw) != len(header) + 3:
                return None
            rgb = tuple(raw[len(header):])
            media_calls.append((source, rgb))
            width = len(LABELS)
            row = [0.0] * (3 * width + 1)
            row[spec["address"]] = 1
            row[width + spec["red" if rgb[0] > rgb[2] else "blue"]] = 1
            return row

        return project, sha256(_encode(spec))

    project, digest = projector(2, 3)

    text = write_source_delta(tmp_path / "text", {
        cue: b"link L00 L01", red_name: b"name L02 L07", blue_name: b"name L03 L11",
    })
    opening = write_source_delta(tmp_path / "opening", {image: red}, previous=text,
                                 binary_sources=(image,), max_bytes=_LIMIT)
    edited = write_source_delta(tmp_path / "edited", {image: blue}, previous=opening,
                                binary_sources=(image,), max_bytes=_LIMIT)
    removed = write_source_delta(tmp_path / "removed", {image: None}, previous=edited, max_bytes=_LIMIT)
    put_back = write_source_delta(tmp_path / "restored", {image: red}, previous=removed,
                                  binary_sources=(image,), max_bytes=_LIMIT)
    reader = make_reader(_CausalControl().eval(), source_ingress=project, source_ingress_sha256=digest)
    text_only = reader.publish(tmp_path / "text", text.seal, max_source_bytes=_LIMIT)
    first = reader.publish(tmp_path / "opening", opening.seal, previous=text_only, max_source_bytes=_LIMIT)
    first = reader.maintain(first, max_seeds=1)
    before_dir = tmp_path / "before"
    before_seal = write_native_publication(before_dir, reader, first, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    assert media_calls == [(image, (255, 0, 0))]
    assert first.continuation_reads(first.derived_seeds[0])[1].selected[0].span.source_id == image

    offline = make_reader(_CausalControl().eval(), source_ingress_sha256=digest)
    old = _read(before_dir, offline, before_seal, opening.seal)
    answer = offline("follow L00", memory=old)
    assert answer.text == "L07" and answer.terminated and answer.applied_derived_steps == 1
    assert offline("follow L00", memory=offline.without_derived(old)).text == ""
    assert offline("follow L00").text == ""
    span, = [span for span in old.evidence_spans(answer) if span.source_id == image]
    assert span.start_byte == 0 and span.end_byte == len(red)
    assert span.version_sha256 == span.span_sha256 == sha256(red)
    reopened_bytes = read_source_delta(tmp_path / "opening", opening.seal, previous=text,
                                       allow_binary=True, max_bytes=_LIMIT).changes[span.source_id]
    assert reopened_bytes == red
    assert sha256(reopened_bytes[span.start_byte:span.end_byte]) == span.span_sha256

    swapped, swapped_hash = projector(3, 2)
    wrong = make_reader(_CausalControl().eval(), source_ingress=swapped, source_ingress_sha256=swapped_hash)
    with pytest.raises(ValueError, match="reader differs"):
        _read(before_dir, wrong, before_seal, opening.seal)
    with pytest.raises(ValueError, match="configured source ingress"):
        offline.publish(tmp_path / "edited", edited.seal, previous=old, max_source_bytes=_LIMIT)
    text_reader = make_reader(_CausalControl().eval())
    text_memory = text_reader.publish(tmp_path / "text", text.seal)
    with pytest.raises(ValueError, match="binary source"):
        text_reader.publish(tmp_path / "opening", opening.seal, previous=text_memory, max_source_bytes=_LIMIT)

    writer = make_reader(_CausalControl().eval(), source_ingress=project, source_ingress_sha256=digest)
    previous = _read(before_dir, writer, before_seal, opening.seal)
    revised = writer.publish(tmp_path / "edited", edited.seal, previous=previous, max_source_bytes=_LIMIT)
    assert revised.derived_seeds == ()
    assert writer("follow L00", memory=revised).text == ""
    revised = writer.maintain(revised, max_seeds=4)
    assert writer("follow L00", memory=revised).text == "L11"
    withdrawn = writer.publish(tmp_path / "removed", removed.seal, previous=revised, max_source_bytes=_LIMIT)
    withdrawn = writer.maintain(withdrawn, max_seeds=4)
    assert writer("follow L00", memory=withdrawn).text == ""
    restored = writer.publish(tmp_path / "restored", put_back.seal, previous=withdrawn, max_source_bytes=_LIMIT)
    restored = writer.maintain(restored, max_seeds=4)
    restored_read = writer("follow L00", memory=restored)
    assert restored_read.text == answer.text and restored_read.source_seal != answer.source_seal
    assert media_calls == [(image, (255, 0, 0)), (image, (0, 0, 100)), (image, (255, 0, 0))]
    assert offline("follow L00", memory=old) == answer

    cold_pin = write_source_delta(tmp_path / "cold", {**text.changes, image: blue},
                                  binary_sources=(image,), max_bytes=_LIMIT)
    cold = writer.maintain(writer.publish(tmp_path / "cold", cold_pin.seal, max_source_bytes=_LIMIT), max_seeds=4)
    assert writer("follow L00", memory=cold).text == "L11"
    for source in text.changes.keys() | {image}:
        cold_rows = tuple(item.row.tobytes() for item in cold._sources[source])
        assert cold_rows == tuple(item.row.tobytes() for item in revised._sources[source])

    wrong_start = wrong.publish(tmp_path / "text", text.seal, max_source_bytes=_LIMIT)
    wrong_memory = wrong.publish(tmp_path / "opening", opening.seal, previous=wrong_start, max_source_bytes=_LIMIT)
    assert wrong("follow L00", memory=wrong.maintain(wrong_memory, max_seeds=4)).text == "L11"


def test_workers_discover_and_replay_parents_without_a_late_task(tmp_path):
    from experiments.memory.native_isolated_stage import _CausalControl, _reader as make_reader

    sources = tuple(f"{number:032x}" for number in range(1, 8))
    root, decoy, relay, cue, bridge, outcome, _ = sources
    source_dir = tmp_path / "source"
    feed = write_source_delta(source_dir, dict(zip(sources, (
        b"link L03 L04", b"link L08 L10", b"link L02 L03", b"link L01 L02",
        b"link L04 L05", b"name L05 L06", b"name L10 L11",
    ), strict=True)))
    model = _CausalControl().eval()
    reader = make_reader(model)
    base = reader.publish(source_dir, feed.seal)
    root_seed, decoy_seed, relay_seed, cue_seed = (base.source_spans(source)[0] for source in sources[:4])
    opening = reader.maintain(base, max_seeds=2, max_parent_scores=1)
    assert opening.derived_seeds == (root_seed, decoy_seed)
    opening_world = tmp_path / "opening-world"
    opening_seal = write_native_publication(opening_world, reader, opening, runtime_sha256=_RUNTIME,
                                           max_bytes=_LIMIT)

    second_model = _CausalControl().eval()
    second = make_reader(second_model)
    reopened = _read(opening_world, second, opening_seal, feed.seal)
    with pytest.raises(ValueError, match="cannot cover one pending seed"):
        second.maintain(reopened, max_seeds=2, max_parent_scores=3)
    assert second_model.calls == 0 and reopened.pending_seeds == 5
    relay_world = second.maintain(reopened, max_seeds=2, max_parent_scores=4)
    assert second_model.calls == 6 and relay_world.pending_seeds == 4
    assert relay_world.continuation_parent(relay_seed) == root_seed
    following_world = tmp_path / "relay-world"
    following_seal = write_native_publication(following_world, second, relay_world, runtime_sha256=_RUNTIME,
                                             max_bytes=_LIMIT)

    third_model = _CausalControl().eval()
    third = make_reader(third_model)
    before_cue = _read(following_world, third, following_seal, feed.seal)
    composed = third.maintain(before_cue, max_seeds=1, max_parent_scores=6)
    assert third_model.calls == 6 and composed.pending_seeds == 3
    assert composed.continuation_parent(cue_seed) == relay_seed
    assert composed.continuation_parent(decoy_seed) is None
    composed_path = tmp_path / "composed-world"
    composed_seal = write_native_publication(composed_path, third, composed, runtime_sha256=_RUNTIME,
                                            max_bytes=_LIMIT)
    assert _read(composed_path, third, composed_seal, feed.seal).derived_seeds == composed.derived_seeds

    prompt = "follow L01"
    late_model = _CausalControl().eval()
    late = make_reader(late_model)
    attached = _read(composed_path, late, composed_seal, feed.seal)
    original = late(prompt, memory=attached)
    assert original.text == "L06" and original.applied_derived_steps == 1
    assert {span.source_id for span in attached.evidence_spans(original)} == {root, relay, cue, bridge, outcome}
    before = _read(following_world, late, following_seal, feed.seal)
    assert late(prompt, memory=before).text == ""
    calls = late_model.calls
    baseline = late.maintain(before, max_seeds=1)
    assert late_model.calls - calls == 3
    assert late(prompt, memory=baseline).text == ""
    derived_off = late.without_derived(attached)
    assert late(prompt, memory=derived_off).text == ""
    with pytest.raises(ValueError, match="needs an intact selective world"):
        late.maintain(derived_off, max_seeds=1, max_parent_scores=8)
    assert late(prompt).text == ""

    def resume(memory):
        for _ in range(memory.pending_seeds):
            memory = late.maintain(memory, max_seeds=1, max_parent_scores=16)
        assert memory.pending_seeds == 0 and memory.continuation_parent(cue_seed) == relay_seed
        return memory

    settled = late.maintain(attached, max_seeds=attached.pending_seeds)
    assert settled.pending_seeds == 0 and late(prompt, memory=settled).text == "L06"
    recoded_path = tmp_path / "recoded"
    recoded = write_source_delta(recoded_path, {bridge: b"link L04 L05 "}, previous=feed)
    next_pin = late.publish(recoded_path, recoded.seal, previous=settled)
    assert next_pin.derived_seeds == (decoy_seed,)
    assert late(prompt, memory=next_pin).text == ""
    rebuilt = resume(next_pin)
    old_parent = attached._selective.acquired[root_seed]
    rebuilt_parent = rebuilt._selective.acquired[root_seed]
    assert old_parent.address.tobytes() == rebuilt_parent.address.tobytes()
    assert old_parent.value.tobytes() == rebuilt_parent.value.tobytes()
    assert rebuilt.continuation_reads(root_seed) != attached.continuation_reads(root_seed)
    assert late(prompt, memory=rebuilt).text == "L06"

    changed_path = tmp_path / "changed-value"
    changed = write_source_delta(changed_path, {bridge: b"link L04 L07", outcome: b"name L07 L07"},
                                 previous=recoded)
    next_pin = late.publish(changed_path, changed.seal, previous=rebuilt)
    assert cue_seed not in next_pin.derived_seeds
    assert late(prompt, memory=next_pin).text == ""
    corrected = resume(next_pin)
    corrected_parent = corrected._selective.acquired[root_seed]
    assert corrected_parent.address.tobytes() == old_parent.address.tobytes()
    assert corrected_parent.value.tobytes() != old_parent.value.tobytes()
    assert late(prompt, memory=corrected).text == "L07"

    removed_path = tmp_path / "removed"
    removed = write_source_delta(removed_path, {bridge: None}, previous=changed)
    withdrawn = late.publish(removed_path, removed.seal, previous=corrected)
    assert decoy_seed in withdrawn.derived_seeds and cue_seed not in withdrawn.derived_seeds
    assert late(prompt, memory=withdrawn).text == ""
    restored_path = tmp_path / "restored"
    restored = write_source_delta(restored_path, {bridge: b"link L04 L07"}, previous=removed)
    current = resume(late.publish(restored_path, restored.seal, previous=withdrawn))
    assert late(prompt, memory=current).text == "L07"
    assert late(prompt, memory=attached) == original


def test_parent_discovery_refuses_a_route_through_its_own_seed_source(tmp_path):
    from experiments.memory.native_isolated_stage import _CausalControl, _reader as make_reader

    earlier, later = (letter * 32 for letter in "ab")
    source = write_source_delta(tmp_path / "source", {earlier: b"link L01 L02", later: b"link L02 L01"})
    reader = make_reader(_CausalControl().eval())
    opening = reader.maintain(reader.publish(tmp_path / "source", source.seal), max_seeds=1, max_parent_scores=1)
    parent, = opening.derived_seeds
    child, = opening.source_spans(later)
    assert later in {hit.span.source_id for step in opening.continuation_reads(parent) for hit in step.selected}
    manual = reader.maintain(opening, max_seeds=1, parent=parent)
    assert manual.continuation_parent(child) == parent
    discovered = reader.maintain(opening, max_seeds=1, max_parent_scores=2)
    assert discovered.continuation_parent(child) is None
    plain_seal = write_native_publication(tmp_path / "plain", reader, reader.maintain(opening, max_seeds=1),
                                         runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    assert write_native_publication(tmp_path / "found", reader, discovered, runtime_sha256=_RUNTIME,
                                    max_bytes=_LIMIT) == plain_seal


def test_parent_discovery_checks_inherited_source_reads(tmp_path):
    from experiments.memory.native_isolated_stage import _CausalControl, _reader as make_reader

    first, middle, last = (letter * 32 for letter in "abc")
    source_dir = tmp_path / "source"
    source = write_source_delta(source_dir, {
        first: b"link L03 L04", middle: b"link L02 L03", last: b"link L04 L02",
    })
    reader = make_reader(_CausalControl().eval())
    base = reader.publish(source_dir, source.seal)
    root, parent, seed = (base.source_spans(source_id)[0] for source_id in (first, middle, last))
    initial = reader.maintain(base, max_seeds=1, max_parent_scores=1)
    before = reader.maintain(initial, max_seeds=1, max_parent_scores=2)
    assert before.continuation_parent(parent) == root
    assert last in {hit.span.source_id for step in before.continuation_reads(root) for hit in step.selected}
    assert last not in {hit.span.source_id for step in before.continuation_reads(parent) for hit in step.selected}
    assert reader.maintain(before, max_seeds=1, parent=parent).continuation_parent(seed) == parent
    assert reader.maintain(before, max_seeds=1, max_parent_scores=4).continuation_parent(seed) is None


@pytest.mark.parametrize(("pixel", "status"), [
    (4, "admitted"),
    (3, "baseline_confirmed"),
    (5, "disconfirmed"),
    (None, "witness_absent"),
    (4, "dependencies_changed"),
])
def test_forecast_admits_only_a_later_better_observation(tmp_path, pixel, status):
    import numpy as np
    from experiments.memory.native_isolated_stage import LABELS, _CausalControl, _reader

    root, cue, bridge, outcome, witness, unrelated = (letter * 32 for letter in "abcdef")
    p6 = b"P6\n1 1\n255\n"
    ingress_digest = sha256(b"controlled forecast pixel coordinates")

    def media_row(source, raw):
        if source != witness or not raw.startswith(p6) or len(raw) != len(p6) + 3 or raw[-3] not in (3, 4, 5):
            return None
        row = np.zeros(3 * len(LABELS) + 1)
        row[raw[-3]] = 0.75
        return row

    model = _CausalControl().eval()
    reader = _reader(model, source_ingress=media_row, source_ingress_sha256=ingress_digest)
    directory = tmp_path / "sources"
    opening = write_source_delta(directory, {
        root: b"link L02 L03", cue: b"link L01 L02", bridge: b"link L03 L04", outcome: b"name L04 L06",
    }, max_bytes=_LIMIT)
    initial = reader.publish(directory, opening.seal, max_source_bytes=_LIMIT)
    first = reader.maintain(initial, max_seeds=1)
    opening_path = tmp_path / "opening"
    opening_seal = write_native_publication(opening_path, reader, first, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    base = _read(opening_path, reader, opening_seal, opening.seal)
    options = dict(witness_source=witness, features=(3, 4, 5), tolerance=40, max_parent_scores=2)
    calls = model.calls
    forecast = reader.forecast_next(base, **options)
    assert model.calls - calls == 6
    assert forecast.seed == base.source_spans(cue)[0] and forecast.parent == base.source_spans(root)[0]
    assert forecast.baseline == (127, 0, 0) and forecast.prediction == (0, 127, 0)
    assert forecast.seed not in base.derived_seeds
    with pytest.raises(ValueError, match="already holds a pending forecast"):
        reader.forecast_next(base, **options)
    with pytest.raises(ValueError, match="different opening world"):
        reader.resolve_forecast(first, forecast, tmp_path / "no-such-feed", opening.seal, max_source_bytes=_LIMIT)

    changes = {unrelated: b"name L09 L11"} if pixel is None else {witness: p6 + bytes((pixel, 0, 0))}
    if status == "dependencies_changed":
        changes[bridge] = b"link L03 L05"
    future = tmp_path / "observation"
    observed = write_source_delta(future, changes, previous=opening,
                                  binary_sources=() if pixel is None else (witness,), max_bytes=_LIMIT)
    if status == "admitted":
        restarted = _reader(_CausalControl().eval(), source_ingress=media_row, source_ingress_sha256=ingress_digest)
        reopened_base = _read(opening_path, restarted, opening_seal, opening.seal)
        with pytest.raises(ValueError, match="no longer pending"):
            restarted.resolve_forecast(reopened_base, forecast, future, observed.seal, max_source_bytes=_LIMIT)
        reader._selector._margin = 1
        with pytest.raises(ValueError, match="unchanged reader"):
            reader.resolve_forecast(base, forecast, future, observed.seal, max_source_bytes=_LIMIT)
        reader._selector._margin = 0
    calls = model.calls
    resolution = reader.resolve_forecast(base, forecast, future, observed.seal, max_source_bytes=_LIMIT)
    learned = resolution.memory
    assert model.calls - calls == int(status in ("witness_absent", "dependencies_changed"))
    assert resolution.status == status and learned.seal == observed.seal
    assert resolution.observation == (None if pixel is None else tuple(95 * (index == pixel) for index in (3, 4, 5)))
    assert (forecast.seed in learned.derived_seeds) == (status == "admitted")

    if pixel is None:
        later_path = tmp_path / "late-observation"
        later = write_source_delta(later_path, {witness: p6 + bytes((4, 0, 0))}, previous=observed,
                                   binary_sources=(witness,), max_bytes=_LIMIT)
        with pytest.raises(ValueError, match="no longer pending"):
            reader.resolve_forecast(learned, forecast, later_path, later.seal, max_source_bytes=_LIMIT)
        assert forecast.seed not in reader.publish(later_path, later.seal, previous=learned,
                                                  max_source_bytes=_LIMIT).derived_seeds
    if status != "admitted":
        return

    prompt = "follow L01"
    assert reader(prompt, memory=base).text == ""
    assert learned.continuation_witness(forecast.seed) == learned.source_spans(witness)[0]
    published = tmp_path / "admitted"
    admitted_seal = write_native_publication(published, reader, learned, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    offline = _reader(_CausalControl().eval(), source_ingress_sha256=ingress_digest)
    attached = _read(published, offline, admitted_seal, observed.seal)
    answer = offline(prompt, memory=attached)
    assert answer.text == "L06" and answer.applied_derived_steps == 1
    assert offline(prompt, memory=offline.without_derived(attached)).text == ""
    assert offline(prompt).text == ""
    assert attached.continuation_witness(forecast.seed) == learned.source_spans(witness)[0]
    assert witness not in {span.source_id for span in attached.evidence_spans(answer)}

    quiet_path = tmp_path / "quiet"
    quiet = write_source_delta(quiet_path, {unrelated: b"link L00 L01"}, previous=observed, max_bytes=_LIMIT)
    still_valid = reader.publish(quiet_path, quiet.seal, previous=learned, max_source_bytes=_LIMIT)
    assert still_valid.continuation_witness(forecast.seed) == learned.source_spans(witness)[0]
    quiet_world = tmp_path / "quiet-world"
    quiet_seal = write_native_publication(quiet_world, reader, still_valid, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    quiet_read = _read(quiet_world, offline, quiet_seal, quiet.seal)
    assert offline(prompt, memory=quiet_read).text == "L06"

    quiet_manifest = quiet_world / "manifest.json"
    original = quiet_manifest.read_bytes()
    header = json.loads(original)
    old_witness_index = next(index for index, document in enumerate(header["documents"]) if document[0] == outcome)
    for field, substitute, message in (
        (1, old_witness_index, "witness generation differs"),
        (2, "1" * 64, "opening source differs from its execution pin"),
    ):
        forged = json.loads(original)
        forged["forecasts"][0][field] = substitute
        altered = _encode(forged)
        quiet_manifest.write_bytes(altered)
        with pytest.raises(ValueError, match=message):
            _read(quiet_world, offline, sha256(altered), quiet.seal)
    quiet_manifest.write_bytes(original)
    restored_valid = _read(quiet_world, reader, quiet_seal, quiet.seal)
    expanded = reader.maintain(restored_valid, max_seeds=restored_valid.pending_seeds, parent=forecast.seed)
    child, = expanded.source_spans(unrelated)
    assert expanded.continuation_parent(child) == forecast.seed
    altered_path = tmp_path / "replacement-observation"
    altered_bytes = write_source_delta(altered_path, {witness: p6 + bytes((5, 0, 0))}, previous=quiet,
                                       binary_sources=(witness,), max_bytes=_LIMIT)
    replaced = reader.publish(altered_path, altered_bytes.seal, previous=expanded, max_source_bytes=_LIMIT)
    assert forecast.seed not in replaced.derived_seeds and child not in replaced.derived_seeds

    removed_path = tmp_path / "withdrawn"
    removed = write_source_delta(removed_path, {witness: None}, previous=quiet, max_bytes=_LIMIT)
    revised = reader.publish(removed_path, removed.seal, previous=expanded, max_source_bytes=_LIMIT)
    assert forecast.seed not in revised.derived_seeds and child not in revised.derived_seeds
    assert reader(prompt, memory=revised).text == ""
    assert offline(prompt, memory=attached) == answer
    restored_path = tmp_path / "restored-observation"
    restored = write_source_delta(restored_path, {witness: changes[witness]}, previous=removed,
                                  binary_sources=(witness,), max_bytes=_LIMIT)
    restored_memory = reader.publish(restored_path, restored.seal, previous=revised, max_source_bytes=_LIMIT)
    assert forecast.seed not in restored_memory.derived_seeds

    manifest = published / "manifest.json"
    header = json.loads(manifest.read_bytes())
    assert header["schema"] == "lsr-native-selective-v3"
    header["forecasts"][0][-1] = list(resolution.observation)
    altered = _encode(header)
    manifest.write_bytes(altered)
    with pytest.raises(ValueError, match="comparison does not support admission"):
        _read(published, offline, sha256(altered), observed.seal)


def test_same_byte_modality_revision_changes_a_native_continuation(tmp_path):
    import numpy as np
    from experiments.memory.native_isolated_stage import LABELS, _CausalControl, _reader as make_reader

    cue, bridge, first_name, second_name = (letter * 32 for letter in "abcd")
    raw_bridge = b"link L01 L02"
    ingress_digest = sha256(b"interpret the same bridge bytes as L01 to L03")

    def binary_bridge(source_id, raw):
        if source_id != bridge or raw != raw_bridge:
            return None
        row = np.zeros(3 * len(LABELS) + 1)
        row[1] = row[len(LABELS) + 3] = 0.8
        return row

    writer = make_reader(_CausalControl().eval(), source_ingress=binary_bridge,
                         source_ingress_sha256=ingress_digest)
    offline = make_reader(_CausalControl().eval(), source_ingress_sha256=ingress_digest)
    source_dir = tmp_path / "text"
    text_pin = write_source_delta(source_dir, {
        cue: b"link L00 L01", bridge: raw_bridge,
        first_name: b"name L02 L07", second_name: b"name L03 L11",
    }, max_bytes=_LIMIT)
    original = writer.maintain(writer.publish(source_dir, text_pin.seal, max_source_bytes=_LIMIT), max_seeds=4)
    text_world = tmp_path / "text-world"
    text_seal = write_native_publication(text_world, writer, original, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    original_read = _read(text_world, offline, text_seal, text_pin.seal)
    prompt = "follow L00"
    old_answer = offline(prompt, memory=original_read)
    assert old_answer.text == "L07" and old_answer.applied_derived_steps == 1

    binary_dir = tmp_path / "binary"
    binary_pin = write_source_delta(binary_dir, {bridge: raw_bridge}, previous=text_pin,
                                    binary_sources=(bridge,), max_bytes=_LIMIT)
    revised = writer.publish(binary_dir, binary_pin.seal, previous=original, max_source_bytes=_LIMIT)
    assert original.source_spans(bridge) == revised.source_spans(bridge)
    assert revised.seal != original.seal
    assert writer(prompt, memory=revised).text == ""
    maintained = writer.maintain(revised, max_seeds=4)
    binary_world = tmp_path / "binary-world"
    binary_seal = write_native_publication(binary_world, writer, maintained, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    revised_read = _read(binary_world, offline, binary_seal, binary_pin.seal)
    new_answer = offline(prompt, memory=revised_read)
    assert new_answer.text == "L11" and new_answer.applied_derived_steps == 1
    assert offline(prompt, memory=offline.without_derived(revised_read)).text == ""
    assert set(revised_read.source_spans(bridge)).issubset(revised_read.evidence_spans(new_answer))

    restored_dir = tmp_path / "restored"
    restored_pin = write_source_delta(restored_dir, {bridge: raw_bridge}, previous=binary_pin, max_bytes=_LIMIT)
    restored = writer.publish(restored_dir, restored_pin.seal, previous=maintained, max_source_bytes=_LIMIT)
    assert writer(prompt, memory=restored).text == ""
    assert writer(prompt, memory=writer.maintain(restored, max_seeds=4)).text == "L07"
    assert offline(prompt, memory=original_read) == old_answer
    assert offline(prompt, memory=revised_read) == new_answer


def test_calibrated_binary_projection_changes_the_retained_read(tmp_path):
    import numpy as np
    from experiments.memory.native_isolated_stage import LABELS, _CausalControl, _reader as make_reader
    from experiments.memory.paired_native_ingress import PairedNativeIngress
    from experiments.memory.weight_program import PairedRoleCalibration

    cue, other_cue, image, other_image, red_name, blue_name = (letter * 32 for letter in "abcdef")
    p6 = b"P6\n2 1\n255\n"
    n = 3 * len(LABELS) + 1
    producer = sha256(b"two-pixel spatial feature control v1")

    def donor_row(left, right, *, green=0):
        row = np.zeros(n)
        row[[0, 2, 3, 4]] = (0.2 * right / 255, -0.6 * left / 255, 0.4, green / 255)
        return row

    def extract(_source, raw):
        if raw.startswith(p6) and len(raw) == len(p6) + 6:
            left, green, _, right, _, _ = raw[len(p6):]
            return donor_row(left, right, green=green)
        return None

    def target_row(left, right):
        row = np.zeros(n)
        row[[1, 5]] = (0.8 * left / 255, 0.8 * (255 - left) / 255)
        row[len(LABELS) + 2:len(LABELS) + 4] = (0.7 * right / 255, 0.7 * (255 - right) / 255)
        return row

    fit = ((50, 177), (79, 42), (178, 200), (201, 23), (113, 112))
    validation = ((200, 10), (8, 194), (110, 124), (125, 33))
    first, mirrored, corrected_pixels = (255, 0), (0, 255), (255, 255)
    assert not set(fit) & set(validation)
    assert not {first, mirrored, corrected_pixels} & (set(fit) | set(validation))

    def calibrate(targets):
        source_fit = [donor_row(*pixel) for pixel in fit]
        source_check = [donor_row(*pixel) for pixel in validation]
        return PairedRoleCalibration.fit(
            source_fit, [targets(*pixel) for pixel in fit],
            source_check, [targets(*pixel) for pixel in validation],
            rank=3, min_correlation=0.99, max_disagreement=1e-8,
            max_span_error=1e-8, max_prediction_error=1e-8,
        )

    calibration = calibrate(target_row)
    adapter = PairedNativeIngress(extract, producer, calibration)
    assert PairedNativeIngress(extract, producer, calibration).sha256 == adapter.sha256
    assert PairedNativeIngress(extract, "0" * 64, calibration).sha256 != adapter.sha256

    def asset(layout, *, green=0):
        left, right = layout
        return p6 + bytes((left, green, 255 - left, right, 0, 255 - right))

    # The assets have the same color counts and opposite spatial layouts.
    assert asset(first)[len(p6):len(p6) + 3] == asset(mirrored)[-3:]
    assert asset(first)[-3:] == asset(mirrored)[len(p6):len(p6) + 3]
    directory = tmp_path / "media"
    pinned = write_source_delta(directory, {
        cue: b"link L00 L01", other_cue: b"link L04 L05",
        image: asset(first), other_image: asset(mirrored),
        red_name: b"name L02 L07", blue_name: b"name L03 L11",
    }, binary_sources=(image, other_image), max_bytes=_LIMIT)
    reader = make_reader(_CausalControl().eval(), source_ingress=adapter, source_ingress_sha256=adapter.sha256)
    memory = reader.maintain(reader.publish(directory, pinned.seal, max_source_bytes=_LIMIT), max_seeds=4)
    world = tmp_path / "world"
    publication = write_native_publication(world, reader, memory, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    offline = make_reader(_CausalControl().eval(), source_ingress_sha256=adapter.sha256)
    attached = _read(world, offline, publication, pinned.seal)
    prompt = "follow L00"
    read = offline(prompt, memory=attached)
    assert read.text == "L11" and read.applied_derived_steps == 1
    assert offline("follow L04", memory=attached).text == "L07"
    assert offline(prompt, memory=offline.without_derived(attached)).text == ""
    assert offline(prompt).text == ""
    assert offline(prompt, memory=attached) == read

    raw = make_reader(_CausalControl().eval(), source_ingress=extract, source_ingress_sha256=producer)
    raw_memory = raw.maintain(raw.publish(directory, pinned.seal, max_source_bytes=_LIMIT), max_seeds=4)
    assert raw("follow L04", memory=raw_memory).text == ""
    raw_first = raw(prompt, memory=raw_memory)

    corrected_dir = tmp_path / "corrected"
    corrected = write_source_delta(corrected_dir, {image: asset(corrected_pixels)}, previous=pinned,
                                   binary_sources=(image,), max_bytes=_LIMIT)
    updated = reader.publish(corrected_dir, corrected.seal, previous=memory, max_source_bytes=_LIMIT)
    updated = reader.maintain(updated, max_seeds=6)
    corrected_world = tmp_path / "corrected-world"
    corrected_publication = write_native_publication(corrected_world, reader, updated, runtime_sha256=_RUNTIME,
                                                     max_bytes=_LIMIT)
    reopened = _read(corrected_world, offline, corrected_publication, corrected.seal)
    updated_answer = offline(prompt, memory=reopened)
    assert updated_answer.text == "L07"
    assert offline(prompt, memory=offline.without_derived(reopened)).text == ""
    media_spans = {span.source_id for span in reopened.evidence_spans(updated_answer)} & {image, other_image}
    assert media_spans == {image}
    raw_updated = raw.publish(corrected_dir, corrected.seal, previous=raw_memory, max_source_bytes=_LIMIT)
    raw_updated = raw.maintain(raw_updated, max_seeds=6)
    assert raw_first.text != "L11" or raw(prompt, memory=raw_updated).text != "L07"
    assert reader("follow L04", memory=updated).text == "L07"
    assert offline(prompt, memory=attached) == read

    outside_dir = tmp_path / "outside"
    outside = write_source_delta(outside_dir, {image: asset(first, green=255)}, previous=pinned,
                                 binary_sources=(image,), max_bytes=_LIMIT)
    with pytest.raises(ValueError, match="outside the calibrated native span"):
        reader.publish(outside_dir, outside.seal, previous=memory, max_source_bytes=_LIMIT)
    assert reader(prompt, memory=memory).text == "L11"

    shifted = PairedNativeIngress(extract, producer, calibrate(
        lambda left, right: target_row(left, 255 - right),
    ))
    wrong = make_reader(_CausalControl().eval(), source_ingress=shifted, source_ingress_sha256=shifted.sha256)
    with pytest.raises(ValueError, match="reader differs"):
        _read(world, wrong, publication, pinned.seal)
    wrong_memory = wrong.maintain(wrong.publish(directory, pinned.seal, max_source_bytes=_LIMIT), max_seeds=4)
    wrong_updated = wrong.publish(corrected_dir, corrected.seal, previous=wrong_memory, max_source_bytes=_LIMIT)
    wrong_updated = wrong.maintain(wrong_updated, max_seeds=6)
    wrong_world = tmp_path / "wrong-world"
    wrong_publication = write_native_publication(wrong_world, wrong, wrong_updated, runtime_sha256=_RUNTIME,
                                                 max_bytes=_LIMIT)
    wrong_reader = make_reader(_CausalControl().eval(), source_ingress_sha256=shifted.sha256)
    wrong_reopened = _read(wrong_world, wrong_reader, wrong_publication, corrected.seal)
    wrong_answer = wrong_reader(prompt, memory=wrong_reopened)
    assert wrong_answer.text == "L11"
    assert wrong_answer.source_seal == updated_answer.source_seal
    assert wrong_answer.publication_seal != updated_answer.publication_seal
    with pytest.raises(ValueError, match="matching reopened publication"):
        reopened.evidence_spans(wrong_answer)
    cue_span, = updated.source_spans(cue)
    assert updated.continuation_reads(cue_span)[1].selected == wrong_updated.continuation_reads(cue_span)[1].selected
    assert updated.continuation_reads(cue_span)[1].selected[0].span.source_id == image


if __name__ == "__main__":
    run = {"publish": _publish_child, "read": _child}[sys.argv[1]]
    print(json.dumps(run(*sys.argv[2:])))
