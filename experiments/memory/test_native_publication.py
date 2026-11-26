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
    assert not offline._parent_conflicts(attached._selective, forecast.seed, unrelated)
    assert offline._parent_conflicts(attached._selective, forecast.seed, unrelated, require_complete_lineage=True)
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


@pytest.mark.parametrize("outcome", ["admitted", "ambiguous", "witness_out_of_order", "dependencies_changed"])
def test_native_forecast_set_carries_rivals_across_source_epochs(tmp_path, outcome):
    import numpy as np
    from experiments.memory.native_forecast_set import NativeForecastWitness
    from experiments.memory.native_isolated_stage import LABELS, _CausalControl, _reader

    high, low, cue, bridge, answer, first, second = (letter * 32 for letter in "abcdef1")
    support = "fedcba98" * 4
    width = len(LABELS)
    marker = width + 10
    ingress_digest = sha256(b"two-witness native forecast set control")

    class RivalControl(_CausalControl):
        def base_model(self, *, inputs_embeds, past_key_values, use_cache, return_dict):
            result = super().base_model(inputs_embeds=inputs_embeds, past_key_values=past_key_values,
                                        use_cache=use_cache, return_dict=return_dict)
            hidden = result.last_hidden_state[0, 0]
            if past_key_values is not None and hidden[4] > 0:
                hidden[marker] = -1
                if past_key_values.address[0, 0, 2] > 0:
                    hidden[4], hidden[5] = 0, 1
            return result

    def ingress(source, raw):
        row = np.zeros(3 * width + 1)
        if source == low:
            row[1], row[7], row[width + 8] = 0.5, 0.9, 0.9
        elif source == first:
            row[marker] = -0.75 if raw == b"first" else 0
        elif source == second:
            row[marker if outcome == "ambiguous" else 4] = -0.75 if outcome == "ambiguous" else 0.75
        return row

    model = RivalControl().eval()
    worker = _reader(model, source_ingress=ingress, source_ingress_sha256=ingress_digest)
    source_path, opening_path = tmp_path / "source", tmp_path / "opening"
    source = write_source_delta(source_path, {
        high: b"link L02 L03", low: b"alternate", cue: b"link L01 L02", bridge: b"link L03 L04",
        answer: b"name L04 L06", support: b"link L08 L03",
    }, binary_sources=(low,), max_bytes=_LIMIT)
    memory = worker.maintain(worker.publish(source_path, source.seal, max_source_bytes=_LIMIT), max_seeds=2)
    opening = write_native_publication(opening_path, worker, memory, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    memory = _read(opening_path, worker, opening, source.seal)
    seed, = memory.source_spans(cue)
    specs = (NativeForecastWitness(first, (marker,), 40),
             NativeForecastWitness(second, (marker,) if outcome == "ambiguous" else (4, 5), 40))
    with pytest.raises(ValueError, match="bounded complete parent scan"):
        worker.forecast_set(memory, witnesses=specs, max_candidates=2, max_parent_scores=3)
    calls = model.calls
    forecast = worker.forecast_set(memory, witnesses=specs, max_candidates=2, max_parent_scores=4)
    assert model.calls - calls == 9
    commitment = forecast.commitment
    highest, lower = forecast.parents
    assert (highest.source_id, lower.source_id) == (high, low)
    assert forecast.seed == seed and seed not in memory.derived_seeds
    assert forecast.baseline[0] == (0,)
    assert tuple(arm.predictions[0] for arm in forecast.alternatives) == ((-127,), (-127,))
    if outcome != "ambiguous":
        assert tuple(arm.predictions[1] for arm in forecast.alternatives) == ((0, 127), (127, 0))

    changes = {first: b"first"}
    if outcome == "witness_out_of_order":
        changes[second] = b"early"
    first_path = tmp_path / "first"
    early = write_source_delta(first_path, changes, previous=source, binary_sources=changes, max_bytes=_LIMIT)
    with pytest.raises(ValueError, match="opening or commitment differs"):
        worker.advance_forecast_set(memory, forecast, first_path, early.seal, commitment="0" * 64,
                                    max_source_bytes=_LIMIT)
    pending = worker.advance_forecast_set(memory, forecast, first_path, early.seal, commitment=commitment,
                                          max_source_bytes=_LIMIT)
    assert seed not in pending.memory.derived_seeds
    if outcome == "witness_out_of_order":
        assert pending.status == outcome
        return
    assert pending.status == "pending" and pending.survivors == (highest, lower)
    assert pending.observations == ((-95,),)

    changes = {second: b"second"}
    if outcome == "dependencies_changed":
        changes[high] = b"link L02 L05"
    second_path = tmp_path / "second"
    later = write_source_delta(second_path, changes, previous=early, binary_sources=(second,), max_bytes=_LIMIT)
    with pytest.raises(ValueError, match="opening or commitment differs"):
        worker.advance_forecast_set(memory, forecast, second_path, later.seal, commitment=commitment,
                                    max_source_bytes=_LIMIT)
    resolution = worker.advance_forecast_set(pending.memory, forecast, second_path, later.seal, commitment=commitment,
                                             max_source_bytes=_LIMIT)
    assert resolution.status == outcome
    assert resolution.memory.seal == later.seal
    if outcome != "admitted":
        assert seed not in resolution.memory.derived_seeds
        if outcome == "ambiguous":
            assert resolution.survivors == (highest, lower)
        return
    assert resolution.survivors == (lower,) and resolution.observations == ((-95,), (95, 0))
    destination = tmp_path / "admitted"
    seal = write_native_publication(destination, worker, resolution.memory,
                                    runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    reader = _reader(RivalControl().eval(), source_ingress_sha256=ingress_digest)
    loaded = _read(destination, reader, seal, later.seal)
    assert loaded.continuation_parent(seed) == lower
    assert [span.source_id for span in loaded.continuation_witnesses(seed)] == [first, second]
    question = "follow L01"
    assert reader(question, memory=loaded).text == "L06"
    assert reader(question, memory=reader.without_derived(loaded)).text == reader(question).text == ""
    assert reader(question, memory=loaded).text == "L06"

    manifest = destination / "manifest.json"
    original = manifest.read_bytes()
    tampered = json.loads(original)
    tampered["forecast_sets"][0][-1][0][3][1] = [1, 126]
    revised_header = _encode(tampered)
    manifest.write_bytes(revised_header)
    with pytest.raises(ValueError, match="forecast set commitment differs"):
        _read(destination, reader, sha256(revised_header), later.seal)
    manifest.write_bytes(original)

    for label, line, failure in (
        ("arm", b"link L03 L04", "later competing source row"),
        ("ancestor", b"link L08 L03", "conflicting or incomplete parent lineage"),
    ):
        competing, changes = "9" * 32, tmp_path / f"competing-{label}"
        update = write_source_delta(changes, {competing: line}, previous=later, max_bytes=_LIMIT)
        revised = worker.publish(changes, update.seal, previous=resolution.memory, max_source_bytes=_LIMIT)
        assert seed not in revised.derived_seeds
        prior = resolution.memory._selective
        stale = replace(prior, shards=revised._selective.shards, priority=(*prior.priority, competing))
        forged = replace(revised, _selective=stale, derived_seeds=resolution.memory.derived_seeds)
        forged_path = tmp_path / f"stale-admission-{label}"
        forged_seal = write_native_publication(forged_path, worker, forged,
                                              runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
        with pytest.raises(ValueError, match=failure):
            _read(forged_path, reader, forged_seal, update.seal)

    for source_id, contents in ((first, b"corrected"), (high, None)):
        path = tmp_path / source_id
        correction = write_source_delta(path, {source_id: contents}, previous=later,
                                        binary_sources=(first,) if source_id == first else (), max_bytes=_LIMIT)
        updated = worker.publish(path, correction.seal, previous=resolution.memory, max_source_bytes=_LIMIT)
        assert seed not in updated.derived_seeds
        assert reader(question, memory=_read(destination, reader, seal, later.seal)).text == "L06"

@pytest.mark.parametrize("real_backbone", [False, True])
def test_content_defined_reuse_matches_cold_source_rows_after_reopen(tmp_path, real_backbone):
    import numpy as np
    import torch
    from experiments.memory.native_latent_recurrence import NativeLatentModel, _content_chunks
    from experiments.memory.test_native_latent_recurrence import _kernel

    class ChunkTokenizer:
        def encode(self, text, *, add_special_tokens, truncation, padding=False):
            assert add_special_tokens and not truncation and not padding
            digest = bytes.fromhex(sha256(text.encode("utf8")))
            return [1 + byte % 9 for byte in digest[:3]]

        def decode(self, ids, *, skip_special_tokens):
            return " ".join(map(str, ids))

    def make_reader(mode="content", tokenizer_version=b"chunk-tokenizer-v1"):
        tokenizer = ChunkTokenizer()
        digest = sha256(tokenizer_version) if mode == "content" and tokenizer_version is not None else None
        if real_backbone:
            transformers = pytest.importorskip("transformers")
            tokenizers = pytest.importorskip("tokenizers")
            words = ("[PAD]", "[UNK]", "Heading", "Neighboring", "text", "New", "clause.", "café", "🌍.", "steady")
            backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(dict(zip(words, range(len(words)))),
                                                                       unk_token="[UNK]"))
            backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
            tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                                             pad_token="[PAD]")
            tokenizer.backend_tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            tokenizer.backend_tokenizer.enable_truncation(max_length=5)
            config = transformers.LlamaConfig(vocab_size=10, hidden_size=6, intermediate_size=12, head_dim=6,
                                              num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
                                              max_position_embeddings=32, eos_token_id=0)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(17)
                model = transformers.LlamaForCausalLM(config).eval()
            model.calls = []
            model.base_model.register_forward_pre_hook(lambda _module, _inputs: model.calls.append("backbone"))
        else:
            model = _StandIn().eval()
        return model, NativeLatentModel(
            model, tokenizer, _kernel(), source_chars=128, context_tokens=32 if real_backbone else 12, latent_steps=2,
            output_tokens=2, representation_scale=0.25, memory_gain=1, null_weight=0.1, eos_ids=(0,),
            top_k=1, source_chunking=mode, source_tokenizer_sha256=digest,
        )

    source, steady = (letter * 32 for letter in "ab")
    original = "".join(f"Heading {index}: café 🌍. Neighboring text {sha256(str(index).encode())}\n"
                       for index in range(48))
    edited = original[:154] + "New clause. " + original[154:]
    raw, revised_raw = original.encode("utf8"), edited.encode("utf8")
    initial_dir, revision_dir = tmp_path / "source", tmp_path / "correction"
    opening = write_source_delta(initial_dir, {source: raw, steady: b"steady"})
    correction = write_source_delta(revision_dir, {source: revised_raw}, previous=opening)
    author_model, author = make_reader()
    encoded = author.publish(initial_dir, opening.seal)
    assert "".join(_content_chunks(edited, 128)) == edited
    opening_dir = tmp_path / "opening-publication"
    seal = write_native_publication(opening_dir, author, encoded, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)

    model, reader = make_reader()
    loaded = _read(opening_dir, reader, seal, opening.seal)
    assert not model.calls
    assert tuple(item.input_sha256 for item in loaded._sources[source]) == tuple(
        item.input_sha256 for item in encoded._sources[source]
    )
    assert len(encoded._sources[source]) == len(list(_content_chunks(original, 128)))
    with pytest.raises(ValueError, match="explicit tokenizer identity"):
        make_reader(tokenizer_version=None)
    _, different_tokenizer = make_reader(tokenizer_version=b"different configuration")
    with pytest.raises(ValueError, match="reader differs"):
        _read(opening_dir, different_tokenizer, seal, opening.seal)
    legacy, fixed_reader = make_reader("fixed")
    with pytest.raises(ValueError, match="reader differs"):
        _read(opening_dir, fixed_reader, seal, opening.seal)

    revised = reader.publish(revision_dir, correction.seal, previous=loaded)
    warm_rows = revised._sources[source]
    old_inputs = {(item.span.span_sha256, item.input_sha256): item for item in loaded._sources[source]}
    current_inputs = {(item.span.span_sha256, item.input_sha256) for item in warm_rows}
    assert 0 < len(current_inputs - old_inputs.keys()) < len(current_inputs) // 2
    assert len(model.calls) == len(current_inputs - old_inputs.keys())
    assert revised._sources[steady] is loaded._sources[steady]
    assert any(item.span.start_byte != old_inputs[item.span.span_sha256, item.input_sha256].span.start_byte
               for item in warm_rows if (item.span.span_sha256, item.input_sha256) in old_inputs)
    assert all(item.span.version_sha256 == sha256(revised_raw) for item in warm_rows)
    assert all(sha256(revised_raw[item.span.start_byte:item.span.end_byte]) == item.span.span_sha256
               for item in warm_rows)

    published_dir = tmp_path / "revised-publication"
    revised_seal = write_native_publication(published_dir, reader, revised, runtime_sha256=_RUNTIME,
                                            max_bytes=_LIMIT)
    reopened_model, reopened_reader = make_reader()
    reopened = _read(published_dir, reopened_reader, revised_seal, correction.seal)
    assert not reopened_model.calls
    cold_model, cold_reader = make_reader()
    cold_dir = tmp_path / "cold-source"
    cold_feed = write_source_delta(cold_dir, {source: revised_raw, steady: b"steady"})
    cold = cold_reader.publish(cold_dir, cold_feed.seal)
    for warm, fresh in zip(reopened._sources[source], cold._sources[source], strict=True):
        assert warm.span == fresh.span and warm.input_sha256 == fresh.input_sha256
        assert warm.row.tobytes() == fresh.row.tobytes()
    np.testing.assert_array_equal(reopened._selective.shards[source].keys, cold._selective.shards[source].keys)
    warm_read = reopened_reader("Heading", memory=reopened)
    cold_read = cold_reader("Heading", memory=cold)
    assert warm_read.token_ids == cold_read.token_ids
    assert author_model.calls and cold_model.calls and not legacy.calls

    with torch.no_grad():
        model.get_input_embeddings().weight[1, 0].add_(1)
    calls = list(model.calls)
    with pytest.raises(ValueError, match="unchanged opening reader"):
        reader.publish(revision_dir, correction.seal, previous=loaded)
    with pytest.raises(ValueError, match="unchanged source encoder"):
        write_native_publication(tmp_path / "drift", reader, revised, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    assert model.calls == calls and not (tmp_path / "drift").exists()

@pytest.mark.parametrize("outcome", [
    "admitted", "persistence_confirmed", "disconfirmed", "witness_change_limit",
    "deletion_only", "witness_unchanged", "witness_withdrawn", "witness_out_of_order",
    "weighted_admitted", "weighted_persistence", "weighted_cap", "same_edit_all_rows",
])
def test_forecast_set_qualifies_an_edit_to_an_existing_text_document(tmp_path, outcome):
    import numpy as np
    from experiments.memory.native_forecast_set import (
        NativeForecastWitness, NativeRevisionWitness, NativeWeightedRevisionWitness,
    )
    from experiments.memory.native_isolated_stage import LABELS, _CausalControl, _reader
    from experiments.memory.native_task_episode import NativeTaskWorldPin
    from experiments.memory.task_evidence import TaskWorld

    high, low, cue, bridge, answer, first = (letter * 32 for letter in "abcdef")
    second, support = "face" * 8, "fedcba98" * 4
    width, marker = len(LABELS), len(LABELS) + 10

    class RevisionControl(_CausalControl):
        def base_model(self, *, inputs_embeds, past_key_values, use_cache, return_dict):
            result = super().base_model(inputs_embeds=inputs_embeds, past_key_values=past_key_values,
                                        use_cache=use_cache, return_dict=return_dict)
            hidden = result.last_hidden_state[0, 0]
            if past_key_values is None and not use_cache and int(inputs_embeds[0, 0, -1]) == 4:
                if hidden[7] > 0:
                    hidden[1] = 0.25 * np.arctanh(0.5)
                    hidden[7] = hidden[width + 8] = 0.25 * np.arctanh(0.9)
                elif hidden[6] > 0:
                    hidden[6] = 0
                    hidden[marker] = -0.25 * np.arctanh(0.75)
                else:
                    hidden[:width] *= 0.25 * np.arctanh(0.75)
            if past_key_values is not None and hidden[4] > 0:
                hidden[marker] = -1
                if past_key_values.address[0, 0, 2] > 0:
                    hidden[4], hidden[5] = 0, 1
            return result

    def document(*lines):
        return "".join(line.ljust(32) for line in lines).encode()

    old_head = "follow  L04" if outcome == "persistence_confirmed" else "follow L09"
    old_text = document(old_head, "follow L10", "follow L11")
    new_tail = "follow L05" if outcome == "disconfirmed" else "follow   L04"
    new_text = document("follow L04", new_tail, "follow L11")
    if outcome == "deletion_only":
        new_text = document(old_head, "follow L11")
    weighted = outcome.startswith("weighted_")
    single_edit = weighted or outcome == "same_edit_all_rows"
    if single_edit:
        new_text = document("follow L04") + b"follow L05"
    if outcome == "weighted_persistence":
        old_text = document("follow  L04") + b"follow  L05"

    model = RevisionControl().eval()
    worker = _reader(model)
    source_path = tmp_path / "source"
    source = write_source_delta(source_path, {
        high: b"link L02 L03", low: b"follow L07", cue: b"link L01 L02", bridge: b"link L03 L04",
        answer: b"name L04 L06", support: b"link L08 L03", second: old_text,
    }, max_bytes=_LIMIT)
    opened = worker.publish(source_path, source.seal, max_source_bytes=_LIMIT)
    warmed = worker.maintain(opened, max_seeds=2)
    opening_path = tmp_path / "opening"
    opening_seal = write_native_publication(opening_path, worker, warmed, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    opening = _read(opening_path, worker, opening_seal, source.seal)
    seed, = opening.source_spans(cue)
    high_parent, = opening.source_spans(high)
    low_parent, = opening.source_spans(low)
    spec_type = NativeWeightedRevisionWitness if weighted else NativeRevisionWitness
    witness = spec_type(second, (4, 5), 60 if single_edit else 40,
                        3, 1 if outcome in ("witness_change_limit", "weighted_cap") else 2)
    fresh = () if single_edit else (NativeForecastWitness(first, (marker,), 40),)
    budgets = dict(max_candidates=2, max_parent_scores=4)
    too_small = replace(witness, max_opening_rows=len(opening.source_spans(second)) - 1)
    with pytest.raises(ValueError, match="opening row allowance"):
        worker.forecast_set(opening, witnesses=(*fresh, too_small), **budgets)
    draft = worker.forecast_set(opening, witnesses=(*fresh, witness), **budgets)
    commitment = draft.commitment
    assert draft.parents == (high_parent, low_parent)
    assert draft._opening_at(len(fresh)).version_sha256 == sha256(old_text)
    assert draft._opening_at(len(fresh)).generation == 1
    if outcome == "weighted_persistence":
        assert draft._opening_at(0).persistence == (71, 24)

    previous_memory, epoch = opening, source
    if fresh:
        changes = {first: b"follow L06"}
        if outcome == "witness_out_of_order":
            changes[second] = new_text
        first_dir = tmp_path / "first"
        epoch = write_source_delta(first_dir, changes, previous=source, max_bytes=_LIMIT)
        pending = worker.advance_forecast_set(opening, draft, first_dir, epoch.seal, commitment=commitment,
                                             max_source_bytes=_LIMIT)
        if outcome == "witness_out_of_order":
            assert pending.status == outcome and seed not in pending.memory.derived_seeds
            return
        assert pending.status == "pending" and pending.survivors == (high_parent, low_parent)
        previous_memory = pending.memory

    changes = {second: new_text}
    if outcome == "witness_unchanged":
        changes = {"0" * 32: b"follow L11"}
    elif outcome == "witness_withdrawn":
        changes[second] = None
    revision_path = tmp_path / "revision"
    revision = write_source_delta(revision_path, changes, previous=epoch, max_bytes=_LIMIT)
    if outcome == "weighted_admitted":
        with pytest.raises(ValueError, match="source feed exceeds the byte budget"):
            worker.advance_forecast_set(previous_memory, draft, revision_path, revision.seal,
                                        commitment=commitment, max_source_bytes=1)
    resolution = worker.advance_forecast_set(previous_memory, draft, revision_path, revision.seal,
                                             commitment=commitment, max_source_bytes=_LIMIT)
    expected = {
        "deletion_only": "unscorable_witness", "weighted_admitted": "admitted",
        "weighted_persistence": "persistence_confirmed", "same_edit_all_rows": "disconfirmed",
        "weighted_cap": "witness_change_limit",
    }.get(outcome, outcome)
    assert resolution.status == expected and resolution.memory.seal == revision.seal
    assert (seed in resolution.memory.derived_seeds) == (expected == "admitted")
    if weighted and outcome != "weighted_cap":
        assert resolution.observations == ((73, 23),)
    if outcome == "same_edit_all_rows":
        assert resolution.observations == (((95, 0), (0, 95)),)
    if expected != "admitted":
        return
    if not weighted:
        assert resolution.observations == ((-95,), ((95, 0), (95, 0)))
    assert resolution.survivors == (low_parent,)

    destination = tmp_path / "admitted"
    seal = write_native_publication(destination, worker, resolution.memory, runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    reader = _reader(RevisionControl().eval())
    restored = _read(destination, reader, seal, revision.seal)
    first_spans = restored.source_spans(first) if fresh else ()
    assert restored.continuation_witnesses(seed) == (*first_spans, *restored.source_spans(second)[:2])
    seals = (epoch.seal, revision.seal) if fresh else (revision.seal,)
    assert restored._selective.forecast_sets[seed].observation_seals == seals
    unadmitted = worker.publish(revision_path, revision.seal, previous=previous_memory, max_source_bytes=_LIMIT)
    assert set(resolution.memory.derived_seeds) - {seed} == set(unadmitted.derived_seeds)
    assert all(resolution.memory._selective.acquired[span] is unadmitted._selective.acquired[span]
               for span in unadmitted.derived_seeds)
    baseline_path = tmp_path / "ordinary-revision"
    baseline_seal = write_native_publication(baseline_path, worker, unadmitted,
                                             runtime_sha256=_RUNTIME, max_bytes=_LIMIT)
    baseline = _read(baseline_path, reader, baseline_seal, revision.seal)
    assert restored._sources.keys() == baseline._sources.keys()
    source_caps = dict(max_feed_bytes=_LIMIT, max_source_bytes=_LIMIT, max_passages=64)
    world = TaskWorld.from_source_epoch(source_path, source.seal, **source_caps)
    if fresh:
        world = TaskWorld.from_source_epoch(first_dir, epoch.seal, previous=world, **source_caps)
    world = TaskWorld.from_source_epoch(revision_path, revision.seal, previous=world, **source_caps)
    pin_options = dict(memory_view="active", latent_steps=2, expected_runtime_sha256=_RUNTIME,
                       max_source_verification_bytes=_LIMIT)
    full_pin = NativeTaskWorldPin.bind(world, reader, restored, **pin_options).to_record()
    baseline_pin = NativeTaskWorldPin.bind(world, reader, baseline, **pin_options).to_record()
    assert full_pin["native_span_catalogue_sha256"] == baseline_pin["native_span_catalogue_sha256"]
    assert full_pin["native_derived_span_count"] == baseline_pin["native_derived_span_count"] + 1
    for source_id in restored._sources:
        for admitted_row, baseline_row in zip(restored._sources[source_id], baseline._sources[source_id], strict=True):
            assert admitted_row.span == baseline_row.span
            assert admitted_row.row.tobytes() == baseline_row.row.tobytes()
    question = "follow L01"
    assert reader(question, memory=restored).text == "L06"
    assert reader(question, memory=baseline).text == reader(question).text == ""
    assert reader(question, memory=restored).text == "L06"

    manifest = destination / "manifest.json"
    original = manifest.read_bytes()
    version = 6 if weighted else 5
    assert json.loads(original)["schema"] == f"lsr-native-selective-v{version}"
    attacks = ("drop_row", "change_predecessor", "wrong_schema")
    if weighted:
        attacks += ("change_persistence",)
    for attack in attacks:
        forged = json.loads(original)
        record, = forged["forecast_sets"]
        if attack == "drop_row":
            record[7][len(fresh)][0].pop()
            reason = "cover all introduced rows"
        elif attack == "wrong_schema":
            forged["schema"] = "lsr-native-selective-v5" if weighted else "lsr-native-selective-v6"
            reason = "witness inventory is malformed" if weighted else "has no weighted witnesses"
        elif attack == "change_persistence":
            record[7][0][3][-1][0] = 1
            reason = "commitment differs"
        else:
            record[-1][0] = "1" * 64
            reason = "observation chain differs"
        changed = _encode(forged)
        manifest.write_bytes(changed)
        with pytest.raises(ValueError, match=reason):
            _read(destination, reader, sha256(changed), revision.seal)
    manifest.write_bytes(original)
    reverted_path = tmp_path / "reverted"
    reverted = write_source_delta(reverted_path, {second: old_text}, previous=revision, max_bytes=_LIMIT)
    retained = worker.publish(reverted_path, reverted.seal, previous=resolution.memory, max_source_bytes=_LIMIT)
    assert seed not in retained.derived_seeds
    assert reader(question, memory=_read(destination, reader, seal, revision.seal)).text == "L06"


def test_weighted_witness_keeps_new_duplicate_locations_and_rounds_once():
    from types import SimpleNamespace

    import numpy as np

    from experiments.memory.native_forecast_set import (
        NativeWeightedRevisionWitness, _WeightedWitnessOpening, _forecast_values,
        _measure_witness, _witness_prediction,
    )
    from experiments.memory.verified_source_tool import SourceSpan

    current = b"aabbcd"
    source = "a" * 32
    version = sha256(current)

    def row(start, end, value):
        return SimpleNamespace(
            span=SourceSpan(source, version, start, end, sha256(current[start:end])),
            row=(value,),
        )

    old_digest = sha256(b"a")
    opening = _WeightedWitnessOpening(old_digest, 1, (old_digest,), (127,))
    rows = [row(0, 1, 1.0), row(1, 2, 1.0), row(2, 3, 1.0), row(3, 4, 1.0), row(4, 6, 0.0)]
    witness = NativeWeightedRevisionWitness(source, (0,), 4, 1, 3)
    assert opening.introduced_rows(rows, 3) == tuple(rows[2:])
    assert opening.introduced_rows(rows, 2) is None
    assert _measure_witness(witness, tuple(rows[2:])) == (64,)

    assert _measure_witness(witness, (row(0, 1, -1.0), row(1, 2, 0.0))) == (-64,)
    for sign in (-1, 1):
        nearly_half = (sign * 1.5 / 127,)
        assert _forecast_values(np.asarray(nearly_half), (0,)) == (sign * 2,)
        assert _witness_prediction(witness, nearly_half) == (sign,)
    unequal = (row(0, 3, 0.49 / 127), row(3, 4, 1.49 / 127))
    assert _measure_witness(witness, unequal) == (1,)
