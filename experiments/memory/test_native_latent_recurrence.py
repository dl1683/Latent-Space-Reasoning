from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.memory.linear_feedback import LinearFeedbackKernel
from experiments.memory.native_latent_recurrence import NativeLatentModel
from experiments.memory.source_revision_feed import write_source_delta
from experiments.memory.verified_source_tool import sha256


torch = pytest.importorskip("torch")


class _Tokenizer:
    _ids = {"What follows?": 1, "What else?": 2, "écho": 3, ".": 0, "blue": 4, "gold": 5, "side": 6,
            " ": 0}
    _readout = {7: "blue", 8: "gold", 9: "side"}

    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens and not truncation
        return [self._ids[text]]

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens
        return " ".join(self._readout.get(token, str(token)) for token in ids if token != 0)


class _Cache:
    def __init__(self, length):
        self._length = length

    def get_seq_length(self):
        return self._length


class _StandIn(torch.nn.Module):
    """Deterministic CPU state machine with the expected backbone and head interfaces."""

    def __init__(self, *, clear_previous=False):
        super().__init__()
        self.config = SimpleNamespace(max_position_embeddings=16, clear_previous=clear_previous)
        rows = torch.zeros((10, 6))
        rows[1, 0] = rows[2, 2] = 1
        rows[3, 0] = rows[3, 3] = 1
        rows[4, 1] = rows[5, 1] = 1
        rows[4, 4], rows[5, 4] = 1, -1
        rows[6, 2] = rows[6, 5] = 1
        self._lookup = torch.nn.Embedding.from_pretrained(rows)
        self.calls = []

    def get_input_embeddings(self):
        return self._lookup

    def get_output_embeddings(self):
        return self._scores

    def base_model(self, *, inputs_embeds, past_key_values, use_cache, return_dict):
        assert return_dict
        self.calls.append(("backbone", use_cache, inputs_embeds.shape[1]))
        hidden = inputs_embeds.clone()
        prefix = 0
        if past_key_values is not None:
            prefix = past_key_values.get_seq_length()
            hidden[..., 0] = 0
            if self.config.clear_previous:
                hidden[..., 1] = torch.where(hidden[..., 2] != 0, 0, hidden[..., 1])
        cache = _Cache(prefix + hidden.shape[1]) if use_cache else None
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=cache)

    def _scores(self, hidden):
        self.calls.append(("head",))
        logits = torch.full((1, 1, 10), -1.0)
        logits[..., 0] = 0.125
        logits[..., 7] = hidden[..., 4]
        logits[..., 8] = -hidden[..., 4]
        logits[..., 9] = hidden[..., 5]
        return logits


def _kernel(*, terminal_axis=4):
    query = np.eye(6, 3)
    value = np.zeros((6, 6))
    value[3, 1] = value[4, terminal_axis] = value[5, 5] = 1
    return LinearFeedbackKernel(query=query, carry=None, observation_key=query, observation_value=value,
                                feedback_key=query, feedback_value=np.zeros((6, 6)), max_steps=2)


def _reader(backbone, *, terminal_axis=4, latent_steps=2, top_k=None, tie_limit=16, tokenizer=None, **options):
    return NativeLatentModel(backbone, _Tokenizer() if tokenizer is None else tokenizer,
                             _kernel(terminal_axis=terminal_axis),
                             source_chars=4, context_tokens=12, latent_steps=latent_steps, top_k=top_k,
                             tie_limit=tie_limit,
                             output_tokens=2, representation_scale=0.25, memory_gain=1, null_weight=0.1,
                             eos_ids=(0,), **options)


def test_two_native_steps_use_only_the_current_source_moments(tmp_path):
    a, b, side = "a" * 32, "b" * 32, "c" * 32
    opening_feed = write_source_delta(tmp_path / "p0", {a: "écho.".encode(), b: b"blue", side: b"side"})
    changed_feed = write_source_delta(tmp_path / "p1", {b: b"gold"}, previous=opening_feed)
    restored_feed = write_source_delta(tmp_path / "p2", {b: b"blue"}, previous=changed_feed)
    withdrawn_feed = write_source_delta(tmp_path / "p3", {b: None}, previous=restored_feed)

    backbone = _StandIn().eval()
    reader = _reader(backbone)
    opening = reader.publish(tmp_path / "p0", opening_feed.seal)
    correction = reader.publish(tmp_path / "p1", changed_feed.seal, previous=opening)
    restoration = reader.publish(tmp_path / "p2", restored_feed.seal, previous=correction)
    withdrawn = reader.publish(tmp_path / "p3", withdrawn_feed.seal, previous=restoration)
    assert backbone.calls == [("backbone", False, 1)] * 6
    assert (opening.source_count, opening.rows) == (3, 4)
    assert (withdrawn.source_count, withdrawn.rows) == (2, 3)
    spans = opening.source_spans(a)
    assert len(spans) == 2 and [(s.start_byte, s.end_byte) for s in spans] == [(0, 5), (5, 6)]
    assert spans[0].span_sha256 == sha256("écho".encode())
    assert correction.source_spans(a) == spans
    assert correction.source_spans(b)[0].version_sha256 == sha256(b"gold")

    question = "What follows?"
    backbone.calls.clear()
    corrected = reader(question, memory=correction)
    assert corrected.text == "gold" and corrected.terminated
    assert corrected.routed_memory_steps == corrected.applied_memory_steps == 2
    assert corrected.source_seal == changed_feed.seal
    assert backbone.calls == [("backbone", True, 1)] * 3 + [
        ("head",), ("backbone", True, 1), ("head",),
    ]
    without_b = reader.without_contribution(correction, b)
    uncertain = reader(question, memory=without_b)
    assert uncertain.text == "" and uncertain.routed_memory_steps == 1
    assert uncertain.source_seal == corrected.source_seal and uncertain.omitted_sources == {b}
    assert reader(question, memory=correction) == corrected
    assert reader(question, memory=opening).text == reader(question, memory=restoration).text == "blue"
    assert reader(question, memory=withdrawn).text == ""
    assert reader("What else?", memory=opening).text == reader("What else?", memory=without_b).text == "side"
    backbone.calls.clear()
    bare = reader(question)
    assert bare.source_seal is None and bare.routed_memory_steps == bare.applied_memory_steps == 0
    assert bare.latent_steps == 2
    assert bare.text == "" and backbone.calls == [("backbone", True, 1)] * 3 + [("head",)]
    with pytest.raises(ValueError, match="ablated"):
        reader.publish(tmp_path / "p2", restored_feed.seal, previous=without_b)
    with pytest.raises(ValueError, match="different native reader"):
        _reader(backbone)(question, memory=opening)


def test_source_only_maintenance_installs_a_useful_continuation_at_fixed_pin(tmp_path):
    cue, downstream, outcome = "a" * 32, "b" * 32, "c" * 32
    opening_feed = write_source_delta(tmp_path / "p0", {cue: "écho".encode(), downstream: b"blue", outcome: b"side"})
    revised_feed = write_source_delta(tmp_path / "p1", {downstream: b"gold"}, previous=opening_feed)
    backbone = _StandIn().eval()
    reader = _reader(backbone, terminal_axis=2)
    opening = reader.publish(tmp_path / "p0", opening_feed.seal)
    backbone.calls.clear()
    maintained = reader.maintain(opening, max_seeds=1)
    assert backbone.calls == [("backbone", True, 1)] * 3
    assert maintained.derived_seeds == opening.source_spans(cue)

    question = "What follows?"
    enabled = reader(question, memory=maintained)
    removed = reader(question, memory=reader.without_derived(maintained))
    assert enabled.source_seal == removed.source_seal == opening.seal
    assert enabled.text == "side" and enabled.routed_derived_steps == 1
    assert removed.derived_omitted and replace(removed, derived_omitted=False) == reader(question, memory=opening)
    assert removed.text == "" and removed.routed_derived_steps == 0
    assert reader(question, memory=maintained) == enabled
    assert reader("What else?", memory=maintained).text == reader("What else?", memory=opening).text == "side"
    assert reader.maintain(maintained, max_seeds=1) is maintained

    revised = reader.publish(tmp_path / "p1", revised_feed.seal, previous=maintained)
    assert revised.derived_seeds == ()
    corrected = reader.maintain(revised, max_seeds=1)
    assert corrected.derived_seeds == revised.source_spans(cue)
    assert reader(question, memory=corrected).text == ""
    assert reader(question, memory=maintained) == enabled


def test_operand_group_preserves_duplicate_origins_and_refuses_conflicting_values():
    from experiments.memory.native_operand_model import _OperandSelector
    from experiments.memory.selective_continuations import RetainedSelector, RowShard

    identity = np.eye(4)
    kernel = LinearFeedbackKernel(query=identity, carry=None, observation_key=identity, observation_value=identity,
                                  feedback_key=identity, feedback_value=identity, max_steps=1)
    selector = _OperandSelector(RetainedSelector(kernel, top_k=2, tie_limit=4, temperature=1),
                                address_axes=(2, 3), transport_l1=2)
    spans = ("a", "b", "duplicate")
    keys = np.array([[64, 0, 32, 0], [0, 64, 32, 0], [64, 0, 32, 0]], dtype=np.int8)
    values = np.array([[64, 0, 0, 0], [0, 64, 0, 0], [64, 0, 0, 0]], dtype=np.int8)
    operand = np.array([16, -32, 8, 8]) / 127
    clean = selector.prepare((RowShard(spans[:2], keys[:2], values[:2]),))
    duplicated = selector.prepare((RowShard(spans, keys, values),))
    base, _, _ = selector.read(operand, clean)
    reused, _, step = selector.read(operand, duplicated)
    np.testing.assert_allclose(reused, np.array([16, -32, 0, 0]) / 127)
    np.testing.assert_array_equal(base, reused)
    assert [hit.span for hit in step.receipt.selected] == list(spans)
    with pytest.raises(ValueError, match="coherent operand basis row"):
        selector.read_withheld_value(operand, duplicated, "duplicate")

    wrong = values.copy()
    wrong[-1] *= -1
    conflict = selector.prepare((RowShard(spans, keys, wrong),))
    assert selector.read(operand, conflict)[2].receipt.status == "competing"
    competing_address = np.array([[64, 0, 0, 32]], dtype=np.int8)
    rival = selector.prepare((RowShard(spans, keys, values),
                              RowShard(("rival",), competing_address, values[:1])))
    assert selector.read(operand, rival)[2].receipt.status == "competing"


@pytest.mark.parametrize("mode", ["legacy", "operand", "admission"])
def test_source_only_rows_compute_a_new_late_input_across_an_image_revision(tmp_path, mode):
    from experiments.memory.native_operand_admission import (
        NativeAdmittedOperandModel, draft_operand_admission, resolve_operand_admission,
    )
    from experiments.memory.native_operand_model import NativeOperandModel
    from experiments.memory.native_task_episode import NativeTaskWorldPin
    from experiments.memory.native_transport_model import NativeTransportModel
    from experiments.memory.native_workspace import NativeWorkspace, StaleNativeWorkspace
    from experiments.memory.task_evidence import TaskWorld

    separate_address = mode != "legacy"
    admitted = mode == "admission"

    class Tokenizer:
        def encode(self, text, **_):
            ids = {"cue-a": 1, "cue-b": 2, "witness-a": 3, "new late input": 4,
                   "unsupported input": 5, "other late input": 6, "cue-c": 9}
            return [ids[text]]

        def decode(self, ids, **_):
            return " ".join({7: "blue", 8: "gold"}[token] for token in ids if token != 0)

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(max_position_embeddings=12)
            rows = torch.zeros(10 if admitted else 9, 8)
            rows[1, :2] = torch.atanh(torch.tensor([100, 0]) / 127)
            rows[2, :2] = torch.atanh(torch.tensor([40, 50]) / 127)
            rows[3, :2] = torch.atanh(torch.tensor([80, -20]) / 127)
            rows[1:3, 7] = rows[3, 3] = 1
            rows[3, 7] = 0.5
            rows[4:6, :2] = torch.atanh(torch.tensor([55, -25]) / 127)
            rows[5, 2] = torch.atanh(torch.tensor(1 / 127))
            rows[6, :2] = torch.atanh(torch.tensor([75, 25]) / 127)
            if separate_address:
                rows[2, :2] = torch.atanh(torch.tensor([0, 50]) / 127)
                rows[1:3, 5] = torch.atanh(torch.tensor(64 / 127))
                rows[4:7, 5] = torch.atanh(torch.tensor(8 / 127))
                rows[4:6, :2] = torch.atanh(torch.tensor([40, -48]) / 127)
                rows[6, :2] = torch.atanh(torch.tensor([-48, 40]) / 127)
            if admitted:
                rows[1:3, 5] = torch.atanh(torch.tensor(8 / 127))
                rows[9, 1] = torch.atanh(torch.tensor(-50 / 127))
                rows[9, 5], rows[9, 7] = rows[1, 5], 1
            self.lookup = torch.nn.Embedding.from_pretrained(rows)

        def get_input_embeddings(self):
            return self.lookup

        def get_output_embeddings(self):
            def scores(hidden):
                logits = hidden.new_full((1, 1, 9), -1)
                logits[..., 0] = 0.08
                logits[..., 7] = -hidden[..., 6]
                logits[..., 8] = hidden[..., 6]
                return logits
            return scores

        def base_model(self, *, inputs_embeds, past_key_values, use_cache, return_dict):
            prefix = 0 if past_key_values is None else past_key_values.get_seq_length()
            state = inputs_embeds.clone()
            worker = bool(state[..., 7].any()) if prefix == 0 else past_key_values.worker
            if prefix == 0 and use_cache and worker:
                # Cues re-enter the same Q8 coordinates that were published.
                state[..., :3] = (torch.atanh(state[..., :3])
                                  if bool((state[..., 7] > 0.6).any()) else 0)
            if prefix:
                if worker:
                    state[..., 5:7] = torch.sign(state[..., 3:5])
                elif prefix == 1:
                    # The next ordinary source can change with this answer,
                    # while its observation channels remain separate.
                    later = [40, 50, 1] if float(state[..., 6]) < 0 else [55, -25, 1]
                    state[..., :3] = torch.atanh(state.new_tensor(later) / 127)
                state[..., 3:5] = 0
            cache = _Cache(prefix + inputs_embeds.shape[1]) if use_cache else None
            if cache is not None:
                cache.worker = worker
            return SimpleNamespace(last_hidden_state=state, past_key_values=cache)

    cue_a, cue_b, text, image, outcome_source, cue_c, shadow_image, neutral = (
        character * 32 for character in "abcdef78"
    )
    pixmap = b"P6\n1 1\n255\n"
    opening_pixel, corrected_pixel = pixmap + bytes((0, 0, 255)), pixmap + bytes((255, 192, 0))

    def ingress(source, raw):
        if source == image and raw in (opening_pixel, corrected_pixel):
            sign = 1 if raw == opening_pixel else -1
            return (0, 100 / 127, 0, 0, sign, 0, 0, 0.5)
        if admitted and source == shadow_image and raw == opening_pixel:
            return (0, -100 / 127, 0, 0, 1, 0, 0, 0.5)
        if separate_address and source == outcome_source and raw[:-3] == pixmap and raw[-3] <= 254:
            return (0, 0, 0, 0, 0, 0, (raw[-3] - 127) / 127, 0)
        return None

    query = np.eye(8)[:, [0, 1, 2, 5]] if separate_address else np.eye(8, 3)
    value = np.diag((0, 0, 0, 1, 1, 0, 0, 0))
    kernel = LinearFeedbackKernel(query=query, carry=None, observation_key=query, observation_value=value,
                                  feedback_key=query, feedback_value=np.zeros((8, 8)), max_steps=2)

    def reader(*, l1, legacy=False):
        model = NativeLatentModel if l1 is None else NativeTransportModel
        transport = {} if l1 is None else {"derived_transport_l1": l1}
        if separate_address and l1 is not None and not legacy:
            model = NativeAdmittedOperandModel if admitted else NativeOperandModel
            transport["address_axes"] = (3,)
        return model(Backbone().eval(), Tokenizer(), kernel, source_chars=12, context_tokens=12,
                     latent_steps=1 if separate_address else 2, output_tokens=2, representation_scale=1,
                     memory_gain=1, null_weight=0.001 if separate_address else 0.1,
                     derived_gain=1, eos_ids=(0,), top_k=2, tie_limit=2,
                     temperature=0.5, source_ingress=ingress,
                     source_ingress_sha256=sha256(b"test PPM plus text row mapping"), **transport)

    limit, runtime = 65_536, sha256(b"isolated local transport fixture")
    p0, p1 = tmp_path / "p0", tmp_path / "p1"
    opening_sources = {cue_a: b"cue-a", cue_b: b"cue-b", text: b"witness-a", image: opening_pixel}
    binaries = (image,)
    if admitted:
        opening_sources.update({cue_c: b"cue-c", shadow_image: opening_pixel})
        binaries = (*binaries, shadow_image)
    original_feed = write_source_delta(p0, opening_sources, binary_sources=binaries, max_bytes=limit)
    correction = write_source_delta(p1, {image: corrected_pixel}, previous=original_feed,
                                    binary_sources=(image,), max_bytes=limit)
    worker = reader(l1=2)
    source = worker.publish(p0, original_feed.seal, max_source_bytes=limit)
    directory = tmp_path / "workspace"
    workspace = NativeWorkspace.create(directory, worker, source, runtime_sha256=runtime, max_bytes=limit)
    opening = workspace.maintain(worker, workspace.head(), max_seeds=6 if admitted else 4)
    if admitted:
        reopened = NativeWorkspace(directory, runtime_sha256=runtime, max_bytes=limit)
        fresh = reader(l1=2)
        original = reopened.attach(fresh, pin=opening)
        a, b, c = (original.source_spans(source)[0] for source in (cue_a, cue_b, cue_c))
        cohorts = ((a, b), (a, c))
        assert set(original.derived_seeds) == {a, b, c}
        questions = ("new late input", "other late input")
        assert all(fresh(question, memory=original).text == "" for question in questions)
        events = ((neutral, (32, 0, 0, 0, 0, 8, 0, 0)), (outcome_source, (16, 32, 0, 0, 0, 8, 0, 0)))
        draft = draft_operand_admission(reopened, worker, opening, cohorts=cohorts, events=events,
                                        coordinates=(6,), tolerance=0)
        no_contrast = draft_operand_admission(reopened, worker, opening, cohorts=cohorts, events=events[:1],
                                              coordinates=(6,), tolerance=0)
        assert no_contrast.source is None and reopened.head() == opening
        assert draft.source == outcome_source
        assert draft.options == ((neutral, 0, ((0,), (0,))), (outcome_source, 0, ((62,), (-62,))))

        negative = NativeWorkspace.create(tmp_path / "disconfirmed", fresh, original,
                                          runtime_sha256=runtime, max_bytes=limit)
        negative_draft = draft_operand_admission(negative, worker, negative.head(), cohorts=cohorts,
                                                 events=events, coordinates=(6,), tolerance=0)
        neutral_path = tmp_path / "disconfirming-image"
        false_witness = write_source_delta(neutral_path, {outcome_source: pixmap + bytes((127, 0, 0))},
                                           previous=original_feed, binary_sources=(outcome_source,),
                                           max_bytes=limit)
        disconfirmed = resolve_operand_admission(
            negative, worker, negative_draft, neutral_path, false_witness.seal,
            commitment=negative_draft.commitment, max_source_bytes=limit)
        assert disconfirmed.status == "disconfirmed" and not disconfirmed.survivors
        denied = negative.attach(fresh, pin=disconfirmed.pin)
        assert all(fresh(question, memory=denied).text == "" for question in questions)

        commitment_path = tmp_path / "operand-draft-commitment"
        commitment_path.write_text(draft.commitment, encoding="ascii")
        outcome_path = tmp_path / "admitting-image"
        assert not outcome_path.exists()
        outcome = write_source_delta(outcome_path, {outcome_source: pixmap + bytes((65, 0, 0))},
                                     previous=original_feed, binary_sources=(outcome_source,), max_bytes=limit)
        racing = NativeWorkspace.create(tmp_path / "racing-admission", fresh, original,
                                        runtime_sha256=runtime, max_bytes=limit)
        racing_draft = draft_operand_admission(racing, worker, racing.head(), cohorts=cohorts, events=events,
                                               coordinates=(6,), tolerance=0)
        intervening_path = tmp_path / "intervening-source"
        intervening = write_source_delta(intervening_path, {neutral: b""}, previous=original_feed, max_bytes=limit)
        racing_resolver = reader(l1=2)
        apply_revision = racing_resolver._apply_source_revision

        def advance_other_writer(delta, before):
            candidate = apply_revision(delta, before)
            racing.revise(worker, racing_draft.pin, intervening_path, intervening.seal, max_source_bytes=limit)
            return candidate

        racing_resolver._apply_source_revision = advance_other_writer
        with pytest.raises(StaleNativeWorkspace, match="head changed"):
            resolve_operand_admission(racing, racing_resolver, racing_draft, outcome_path, outcome.seal,
                                      commitment=racing_draft.commitment, max_source_bytes=limit)
        assert racing.head().source_seal == intervening.seal
        assert all(fresh(question, memory=racing.attach(fresh)).text == "" for question in questions)

        forged = replace(draft, payload_digests=(draft.payload_digests[1], draft.payload_digests[0]))
        with pytest.raises(ValueError, match="pinned cohorts"):
            resolve_operand_admission(reopened, worker, forged, outcome_path, outcome.seal,
                                      commitment=forged.commitment, max_source_bytes=limit)
        commitment = commitment_path.read_text(encoding="ascii")
        resolver = reader(l1=2)
        resolution = resolve_operand_admission(reopened, resolver, draft, outcome_path, outcome.seal,
                                               commitment=commitment, max_source_bytes=limit)
        assert resolution.status == "admitted" and resolution.survivors == (1,)
        assert reopened.head() == resolution.pin
        with pytest.raises(StaleNativeWorkspace, match="head changed"):
            resolve_operand_admission(reopened, worker, draft, outcome_path, outcome.seal,
                                      commitment=commitment, max_source_bytes=limit)

        live_reader = reader(l1=2)
        live = reopened.attach(live_reader, pin=resolution.pin)
        with pytest.raises(ValueError, match="opening is not in this workspace history"):
            NativeWorkspace.create(tmp_path / "foreign-grant", live_reader, live,
                                   runtime_sha256=runtime, max_bytes=limit)
        assert set(live.derived_seeds) == {a, b, c}
        reads = tuple(live_reader(question, memory=live) for question in questions)
        assert tuple(read.text for read in reads) == ("gold", "blue")
        for read in reads:
            assert {hit.span for hit in read.derived_reads[0].selected} == {a, c}
            assert resolution.witness in live_reader.operation_evidence(live, read)
        off = live_reader.without_derived(live)
        cut = live_reader._without_continuation(live, c)
        assert all(live_reader(question, memory=off).text == live_reader(question, memory=cut).text
                   == live_reader(question).text == "" for question in questions)
        bad_address = np.array([16, 32, 0, 0, 0, 9, 0, 0]) / 127
        value, receipt = live_reader.predict_operation(live, bad_address)
        assert value is None and receipt.status == "empty"
        other_cohort = replace(live._selective.operand_admission, winner=0)
        wrong = replace(live, _publication_seal=None,
                        _selective=replace(live._selective, operand_admission=other_cohort))
        wrong_reads = tuple(live_reader(question, memory=wrong) for question in questions)
        assert tuple(read.text for read in wrong_reads) == ("blue", "gold")
        with pytest.raises(ValueError, match="matching reopened publication"):
            live_reader.operation_evidence(live, wrong_reads[0])
        assert tuple(live_reader(question, memory=live) for question in questions) == reads

        withdrawn_path = tmp_path / "withdraw-witness"
        withdrawal = write_source_delta(withdrawn_path, {outcome_source: None}, previous=outcome, max_bytes=limit)
        retired = reopened.revise(worker, resolution.pin, withdrawn_path, withdrawal.seal, max_source_bytes=limit)
        inactive = reopened.attach(live_reader, pin=retired)
        assert set(inactive.derived_seeds) == {a, b, c}
        assert all(live_reader(question, memory=inactive).text == "" for question in questions)
        historical = reopened.attach(live_reader, pin=resolution.pin)
        assert tuple(live_reader(question, memory=historical) for question in questions) == reads

        retry = draft_operand_admission(reopened, worker, retired, cohorts=cohorts, events=events,
                                        coordinates=(6,), tolerance=0)
        reissue_path = tmp_path / "reissue-witness"
        reissue = write_source_delta(reissue_path, {outcome_source: pixmap + bytes((65, 0, 0))},
                                     previous=withdrawal, binary_sources=(outcome_source,), max_bytes=limit)
        revived = resolve_operand_admission(reopened, worker, retry, reissue_path, reissue.seal,
                                            commitment=retry.commitment, max_source_bytes=limit)
        assert revived.status == "admitted" and revived.survivors == (1,)
        renewed = reopened.attach(live_reader, pin=revived.pin)
        assert tuple(live_reader(question, memory=renewed).text for question in questions) == ("gold", "blue")
        assert renewed.seal == revived.pin.source_seal != live.seal

        extra_path = tmp_path / "unrelated-source"
        extra = write_source_delta(extra_path, {neutral: b""}, previous=reissue, max_bytes=limit)
        carried = reopened.revise(worker, revived.pin, extra_path, extra.seal, max_source_bytes=limit)
        assert tuple(live_reader(question, memory=reopened.attach(live_reader, pin=carried)).text
                     for question in questions) == ("gold", "blue")

        losing_support_path = tmp_path / "withdraw-losing-support"
        lost_rival = write_source_delta(losing_support_path, {image: None}, previous=extra, max_bytes=limit)
        invalidated = reopened.revise(worker, carried, losing_support_path, lost_rival.seal,
                                      max_source_bytes=limit)
        no_loser = reopened.attach(live_reader, pin=invalidated)
        assert set(no_loser.derived_seeds) == {a, c}
        assert all(live_reader(question, memory=no_loser).text == "" for question in questions)
        assert tuple(live_reader(question, memory=renewed).text for question in questions) == ("gold", "blue")
        return

    revised = workspace.revise(worker, opening, p1, correction.seal, max_source_bytes=limit)
    updated = workspace.maintain(worker, revised, max_seeds=1)

    fresh = reader(l1=2)
    reopened = NativeWorkspace(directory, runtime_sha256=runtime, max_bytes=limit)
    original, corrected = (reopened.attach(fresh, pin=x) for x in (opening, updated))
    assert [span.source_id for span in original.derived_seeds] == [cue_a, cue_b]
    assert [span.source_id for span in corrected.derived_seeds] == [cue_a, cue_b]
    assert len(reopened.attach(fresh, pin=revised).derived_seeds) == 1

    if separate_address:
        from fractions import Fraction

        from experiments.memory.native_witness_selection import choose_forecast_witness
        from experiments.memory.source_revision_feed import _encode

        # Both candidate pins precede the controller's outcome release.
        events = (("agree", (32, 0)), (outcome_source, (16, 32)))
        options = []
        for event, (left, right) in events:
            latent = np.array([left, right, 0, 0, 0, 8, 0, 0]) / 127
            predictions = []
            for memory, polarity in ((original, 1), (corrected, -1)):
                consequence, receipt = fresh.predict_operation(memory, latent)
                assert receipt.status == "ready" and receipt.observed_source_seal == memory.seal
                # Exact oracle for the image-controlled role, independent of the fitted solve.
                assert consequence[6] == pytest.approx(float(Fraction(polarity * right * 97, 50 * 127)))
                predictions.append((round(127 * consequence[6]),))
            options.append((event, 0, tuple(predictions)))
        assert choose_forecast_witness(options[:1]) is None
        selected = choose_forecast_witness(options)
        assert selected == outcome_source
        expected_options = (("agree", 0, ((0,), (0,))), (outcome_source, 0, ((62,), (-62,))))
        assert tuple(options) == expected_options
        preimage = ["q8-rint-v1", asdict(opening), asdict(updated), events, 8, 6, selected, tuple(options)]
        frozen = sha256(_encode(preimage))

        outcome_path = tmp_path / "chosen-event"
        released_pixel = pixmap + bytes((65, 0, 0))
        event = write_source_delta(outcome_path, {outcome_source: released_pixel}, previous=correction,
                                   binary_sources=(outcome_source,), max_bytes=limit)
        observed_pin = reopened.revise(worker, updated, outcome_path, event.seal, max_source_bytes=limit)
        observed = reopened.attach(fresh, pin=observed_pin)
        outcome, = observed._sources[outcome_source]
        live = (round(127 * outcome.row[6]),)
        assert frozen == sha256(_encode(preimage))
        assert live == options[1][2][1] != options[1][2][0]
        assert fresh("new late input", memory=observed).text == "gold"
        assert fresh("other late input", memory=observed).text == "blue"

    # The exact control solves 50*c_b = -25 then 100*c_a + 40*c_b = 55.
    # The PPM correction negates the second source consequence.
    task = "new late input"
    read = fresh(task, memory=original)
    fixed = fresh(task, memory=corrected)
    assert (read.text, fixed.text) == ("blue", "gold")
    assert read.terminated and fixed.terminated and read.applied_derived_steps == fixed.applied_derived_steps == 1
    expected_query = [40, -48, 0, 8] if separate_address else [55, -25, 0]
    assert np.frombuffer(read.derived_steps[0].query, dtype=np.int8).tolist() == expected_query
    assert {span.source_id for span in original.evidence_spans(read)} == {cue_a, cue_b, text, image}
    assert fresh(task, memory=reopened.attach(fresh, pin=revised)).text == ""
    assert fresh(task, memory=fresh.without_derived(original)).text == ""
    assert fresh(task).text == ""
    unsupported = fresh("unsupported input", memory=original)
    assert unsupported.text == "" and unsupported.derived_reads[0].status == "unsupported"
    assert fresh(task, memory=reopened.attach(fresh, pin=opening)) == read
    # The same two published rows give coefficients 0.55 and 0.5 here.
    second = "other late input"
    second_read = fresh(second, memory=original)
    assert second_read.text == "gold"
    assert fresh(second, memory=corrected).text == "blue"
    assert fresh(second, memory=reopened.attach(fresh, pin=revised)).text == ""

    if separate_address:
        assert second_read.derived_reads[0].selected == read.derived_reads[0].selected
        assert np.frombuffer(second_read.derived_steps[0].query, dtype=np.int8).tolist() == [-48, 40, 0, 8]
        baseline = reader(l1=2, legacy=True)
        baseline_opening = baseline.publish(p0, original_feed.seal, max_source_bytes=limit)
        baseline_opening = baseline.maintain(baseline_opening, max_seeds=4)
        for question in (task, second):
            plain = baseline(question, memory=baseline_opening)
            assert plain.text == "" and plain.derived_reads[0].status == "unsupported"
            assert len(plain.derived_reads[0].selected) == 1
        with pytest.raises(ValueError, match="reader"):
            reopened.attach(baseline, pin=opening)
        donor = fresh.compare_continuation_value(fresh.task_token_ids(task), original,
                                                 original.derived_seeds[1], corrected, corrected.derived_seeds[1])
        assert donor.source_clamped.text == "gold"
        assert donor.source_clamped.source_steps == read.source_steps
        assert donor.source_clamped.value_substitution.input_max_abs_delta > 0
        return

    baseline = reader(l1=None)
    plain_source = baseline.publish(p0, original_feed.seal, max_source_bytes=limit)
    plain_memory = baseline.maintain(plain_source, max_seeds=4)
    plain = baseline(task, memory=plain_memory)
    assert [span.source_id for span in plain_memory.derived_seeds] == [cue_a, cue_b]
    assert plain.source_steps[0] == read.source_steps[0]
    assert plain.derived_steps[0] == read.derived_steps[0] and plain.text == "gold"
    donor = fresh.compare_continuation_value(fresh.task_token_ids(task), original, original.derived_seeds[1],
                                             corrected, corrected.derived_seeds[1])
    assert donor.rerouted.text == donor.source_clamped.text == "gold"
    assert donor.rerouted.source_steps[1] != read.source_steps[1]
    assert donor.source_clamped.source_steps == read.source_steps
    assert donor.source_clamped.value_substitution.input_max_abs_delta > 0
    assert donor.original_q8_squared_norm == donor.donor_q8_squared_norm > 0
    assert all(arm.derived_reads[1].status == "unsupported"
               for arm in (donor.intact, donor.rerouted, donor.source_clamped))
    with pytest.raises(ValueError, match="reader"):
        reopened.attach(baseline, pin=opening)

    # Episode records have no representation for transport coefficients.
    text_dir = tmp_path / "episode-text"
    text_feed = write_source_delta(text_dir, {cue_a: b"cue-a", cue_b: b"cue-b"})
    text_source = worker.publish(text_dir, text_feed.seal, max_source_bytes=limit)
    text_store = NativeWorkspace.create(tmp_path / "episode-workspace", worker, text_source,
                                        runtime_sha256=runtime, max_bytes=limit)
    text_memory = text_store.attach(fresh)
    world = TaskWorld.from_source_epoch(text_dir, text_feed.seal, max_feed_bytes=limit,
                                       max_source_bytes=limit, max_passages=8)
    with pytest.raises(TypeError, match="native latent reader"):
        NativeTaskWorldPin.bind(world, fresh, text_memory, memory_view="active", latent_steps=2,
                                expected_runtime_sha256=runtime, max_source_verification_bytes=limit)


def test_small_alignment_has_proportionally_small_observation_support():
    kernel = _kernel()
    row = np.array([[1, 0, 0, 1, 0, 0]])
    bank = kernel.prepare(row)
    value, mass = kernel.read_observations(np.array([1.0, 0, 0, 0, 0, 0]), bank)
    weaker, small = kernel.read_observations(np.array([0.001, 0, 0, 0, 0, 0]), bank)
    np.testing.assert_array_equal(value, weaker)
    assert mass == 1 and small == pytest.approx(0.001)


def test_selective_worker_keeps_a_valid_continuation_and_invalidates_for_new_tie(tmp_path):
    cue, middle, result, unrelated, tie = (letter * 32 for letter in "abcde")
    documents = {cue: "écho".encode(), middle: b"blue", result: b"side", unrelated: b"."}
    opening_feed = write_source_delta(tmp_path / "p0", documents)
    quiet_feed = write_source_delta(tmp_path / "p1", {unrelated: b" "}, previous=opening_feed)
    tie_feed = write_source_delta(tmp_path / "p2", {tie: b"gold"}, previous=quiet_feed)
    restored_feed = write_source_delta(tmp_path / "p3", {tie: None}, previous=tie_feed)
    backbone = _StandIn(clear_previous=True).eval()
    reader = _reader(backbone, terminal_axis=2, top_k=1)
    opening = reader.publish(tmp_path / "p0", opening_feed.seal)
    maintained = reader.maintain(opening, max_seeds=1)
    seed, = maintained.derived_seeds
    assert seed == opening.source_spans(cue)[0]
    dependencies = maintained.continuation_reads(seed)
    assert [step.selected[0].span.source_id for step in dependencies] == [cue, middle]
    assert all(step.rows_scored == 4 for step in dependencies)
    question = "What follows?"
    supported = reader(question, memory=maintained)
    source_only = reader(question, memory=reader.without_derived(maintained))
    assert supported.text == "side" and supported.applied_derived_steps == 1
    assert supported.derived_reads[0].selected[0].span == seed
    assert not source_only.text and source_only.source_seal == supported.source_seal
    ablated = reader.without_contribution(reader.without_derived(maintained), cue)
    omitted = reader(question, memory=ablated)
    assert omitted.source_seal == supported.source_seal and omitted.omitted_sources == {cue}
    assert omitted.source_reads[0].status == "empty" and not omitted.text

    backbone.calls.clear()
    quiet = reader.publish(tmp_path / "p1", quiet_feed.seal, previous=maintained)
    assert backbone.calls == [("backbone", False, 1)]
    assert quiet.derived_seeds == maintained.derived_seeds
    assert quiet.continuation_reads(seed) == dependencies
    quiet_answer = reader(question, memory=quiet)
    assert quiet_answer.text == supported.text
    rebuilt = reader.maintain(reader.without_derived(quiet), max_seeds=1)
    assert rebuilt.derived_seeds == quiet.derived_seeds
    np.testing.assert_array_equal(rebuilt._selective.derived[0].keys, quiet._selective.derived[0].keys)
    np.testing.assert_array_equal(rebuilt._selective.derived[0].values, quiet._selective.derived[0].values)
    assert reader(question, memory=rebuilt) == quiet_answer

    contested = reader.publish(tmp_path / "p2", tie_feed.seal, previous=quiet)
    assert not contested.derived_seeds
    retried = reader.maintain(contested, max_seeds=1)
    assert not retried.derived_seeds
    read = reader(question, memory=retried)
    ambiguous = read.source_reads[1]
    assert ambiguous.status == "competing" and ambiguous.eligible == 2
    assert {hit.span.source_id for hit in ambiguous.selected} == {middle, tie}
    assert not read.text

    restored = reader.publish(tmp_path / "p3", restored_feed.seal, previous=retried)
    # Fresh workers can continue after the changed-source budget is used.
    while seed not in restored.derived_seeds:
        following = reader.maintain(restored, max_seeds=1)
        assert following is not restored
        restored = following
    assert reader(question, memory=restored).text == supported.text
    assert reader(question, memory=maintained) == supported


def test_selective_queue_preserves_rejected_work_and_progress_across_edit(tmp_path):
    idle, cue, middle, result, edited = (letter * 32 for letter in "abcde")
    opening = write_source_delta(tmp_path / "p0", {
        idle: b".", cue: "écho".encode(), middle: b"blue", result: b"side", edited: b".",
    })
    following = write_source_delta(tmp_path / "p1", {edited: b" "}, previous=opening)
    reader = _reader(_StandIn(clear_previous=True).eval(), terminal_axis=2, top_k=1)
    world = reader.publish(tmp_path / "p0", opening.seal)
    assert world.pending_seeds == 5
    first = reader.maintain(world, max_seeds=1)
    assert first.pending_seeds == 4 and not first.derived_seeds
    revised = reader.publish(tmp_path / "p1", following.seal, previous=first)
    assert revised.pending_seeds == 4
    second = reader.maintain(revised, max_seeds=1)
    assert second.derived_seeds == world.source_spans(cue)
    assert second.pending_seeds == 3


def test_source_only_third_runtime_step_matches_the_shorter_maintained_route(tmp_path):
    documents = {"a" * 32: "écho".encode(), "b" * 32: b"blue", "c" * 32: b"side"}
    folder = tmp_path / "p0"
    feed = write_source_delta(folder, documents)
    backbone = _StandIn(clear_previous=True).eval()
    shallow = _reader(backbone, terminal_axis=2, top_k=1)
    deeper = _reader(backbone, terminal_axis=2, latent_steps=3, top_k=1)
    prepared = shallow.maintain(shallow.publish(folder, feed.seal), max_seeds=1)
    source_only = deeper.publish(folder, feed.seal)
    question = "What follows?"
    offline = shallow(question, memory=prepared)
    online = deeper(question, memory=source_only)
    assert offline.text == online.text == "side"
    assert offline.applied_derived_steps > 0
    assert offline.latent_steps == 2 and online.latent_steps == 3
    assert deeper(question, memory=source_only, steps=2).text == ""
    assert deeper(question, steps=0).latent_steps == 0


def test_native_reader_rejects_a_non_cpu_shard_before_forward():
    backbone = _StandIn().eval()
    reader = _reader(backbone)
    backbone.register_buffer("non_cpu_shard", torch.empty(1, device="meta"))
    with pytest.raises(ValueError, match="entire native backbone"):
        reader("What follows?")
    assert backbone.calls == []


@pytest.mark.parametrize("top_k", [None, 1])
def test_transformers_llama_backbone_accepts_continuous_cached_inputs(tmp_path, top_k):
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(vocab_size=10, hidden_size=6, intermediate_size=12, head_dim=6,
                                      num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
                                      max_position_embeddings=16, eos_token_id=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        backbone = transformers.LlamaForCausalLM(config).eval()
    reader = _reader(backbone, top_k=top_k)
    source = "b" * 32
    feed = write_source_delta(tmp_path / "p0", {source: b"blue"})
    memory = reader.publish(tmp_path / "p0", feed.seal)
    with_memory = reader("What follows?", memory=memory)
    no_memory = reader("What follows?")
    assert memory.rows == 1
    assert with_memory.source_seal == feed.seal and no_memory.source_seal is None
    assert with_memory.latent_steps == no_memory.latent_steps == 2
    assert len(with_memory.source_reads) == (0 if top_k is None else 2)
    for result in (with_memory, no_memory):
        assert 1 <= len(result.token_ids) <= 2
        assert result.terminated or len(result.token_ids) == 2
    if top_k is not None:
        from experiments.memory.native_publication import read_native_publication, write_native_publication

        directory = tmp_path / "publication"
        runtime = sha256(b"random Llama software interface fixture")
        seal = write_native_publication(directory, reader, memory, runtime_sha256=runtime, max_bytes=65_536)
        fresh = _reader(backbone, top_k=top_k)
        reopened = read_native_publication(directory, fresh, seal=seal, source_seal=feed.seal,
                                           runtime_sha256=runtime, max_bytes=65_536)
        assert fresh("What follows?", memory=reopened) == replace(with_memory, publication_seal=seal)



def test_residual_recurrence_carries_worker_computation_across_source_revisions(tmp_path):
    from experiments.memory.native_workspace import NativeWorkspace

    class ResidualStep(torch.nn.Module):
        def forward(self, hidden_states, *, cached):
            result = hidden_states.clone()
            if cached:
                result[..., 0] = 0
            return result, None

    class ReadoutLayer(torch.nn.Module):
        def forward(self, hidden_states):
            result = hidden_states.clone()
            result[..., 0] *= torch.where(result[..., 3] != 0, 1, -1)
            return result

    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity(), ResidualStep(), ReadoutLayer()])

        def forward(self, *, inputs_embeds, past_key_values, use_cache, return_dict):
            assert return_dict
            prefix = 0 if past_key_values is None else past_key_values.get_seq_length()
            lower = self.layers[0](inputs_embeds)
            state, _ = self.layers[1](hidden_states=lower, cached=prefix > 0)
            final = self.layers[2](state)
            cache = _Cache(prefix + inputs_embeds.shape[1]) if use_cache else None
            return SimpleNamespace(last_hidden_state=final, past_key_values=cache)

    class LayeredBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fixture = _StandIn()
            self.base_model = Body()
            self.config = self.fixture.config

        def get_input_embeddings(self):
            return self.fixture.get_input_embeddings()

        def get_output_embeddings(self):
            return self.fixture.get_output_embeddings()

    cue, bridge, distractor = (letter * 32 for letter in "abc")
    pixmap = b"P6\n1 1\n255\n"
    blue, gold = pixmap + b"\x00\x00\xff", pixmap + b"\xff\xc0\x00"

    def ingress(source, raw):
        if source == bridge and raw in (blue, gold):
            return (0, 1, 0, 0, 1 if raw == blue else -1, 0)
        return None

    tokenizer_identity = sha256(b"supplied fixture token mapping")

    def make_reader(layer=1, source_tokenizer=tokenizer_identity):
        return _reader(LayeredBackbone().eval(), top_k=1, residual_layer=layer, source_ingress=ingress,
                       source_ingress_sha256=sha256(b"blue and gold pixel fixture"),
                       source_tokenizer_sha256=source_tokenizer)

    runtime, limit = sha256(b"layer boundary software fixture"), 65_536
    opening_dir = tmp_path / "opening"
    opening = write_source_delta(opening_dir, {cue: "\u00e9cho".encode(), bridge: blue, distractor: b"side"},
                                 binary_sources=(bridge,), max_bytes=limit)
    worker = make_reader()
    initial = worker.publish(opening_dir, opening.seal, max_source_bytes=limit)
    directory = tmp_path / "workspace"
    workspace = NativeWorkspace.create(directory, worker, initial, runtime_sha256=runtime, max_bytes=limit)
    first = workspace.maintain(worker, workspace.head(), max_seeds=1)

    fresh = make_reader()
    reopened = NativeWorkspace(directory, runtime_sha256=runtime, max_bytes=limit)
    active = reopened.attach(fresh, pin=first)
    seed, = active.derived_seeds
    question = "What follows?"
    answer = fresh(question, memory=active, steps=1)
    assert answer.text == "blue" and answer.terminated and answer.applied_derived_steps == 1
    assert fresh(question, memory=fresh._without_continuation(active, seed), steps=1).text == ""
    withheld = fresh.read_token_ids(fresh.task_token_ids(question, steps=1), memory=active, steps=1,
                                    withhold_continuation_value=seed)
    assert withheld.text == "" and withheld.value_withholding.changed_input
    assert fresh(question, steps=1).text == ""
    assert active.evidence_spans(answer) == (*active.source_spans(cue), *active.source_spans(bridge))

    bank, = active._selective.derived
    flipped = replace(bank, keys=-bank.keys)
    address_only = replace(active, _selective=replace(active._selective, derived=(flipped,)))
    missed = fresh(question, memory=address_only, steps=1)
    assert missed.text == "" and missed.derived_reads[0].status == "empty"
    assert missed.source_steps == answer.source_steps
    assert missed.derived_steps[0].query == answer.derived_steps[0].query

    legacy = make_reader(None)
    legacy_memory = legacy.maintain(legacy.publish(opening_dir, opening.seal, max_source_bytes=limit), max_seeds=1)
    assert legacy_memory.derived_seeds == (seed,)
    ordinary = legacy(question, memory=legacy_memory, steps=1)
    assert ordinary.derived_reads[0].status == "empty" and ordinary.text == ""
    q8_query = np.frombuffer(ordinary.derived_steps[0].query, dtype=np.int8).astype(np.int64)
    legacy_shard, = legacy_memory._selective.derived
    assert int(legacy_shard.keys[0] @ q8_query) < 0
    with pytest.raises(ValueError, match="reader differs"):
        reopened.attach(legacy, pin=first)
    with pytest.raises(ValueError, match="reader differs"):
        reopened.attach(make_reader(source_tokenizer=sha256(b"different mapping")), pin=first)
    with pytest.raises(ValueError, match="custom source tokenizer identity"):
        make_reader(source_tokenizer=None)

    correction_dir = tmp_path / "correction"
    correction = write_source_delta(correction_dir, {bridge: gold}, previous=opening,
                                    binary_sources=(bridge,), max_bytes=limit)
    from experiments.memory.native_publication import write_native_publication

    worker._scale *= 2
    with pytest.raises(ValueError, match="unchanged opening reader"):
        worker.publish(correction_dir, correction.seal, previous=initial, max_source_bytes=limit)
    with pytest.raises(ValueError, match="unchanged source encoder"):
        write_native_publication(tmp_path / "drift", worker, initial, runtime_sha256=runtime, max_bytes=limit)
    worker._scale /= 2

    corrected = workspace.revise(worker, first, correction_dir, correction.seal, max_source_bytes=limit)
    assert reopened.attach(fresh, pin=corrected).derived_seeds == ()
    assert fresh(question, memory=reopened.attach(fresh), steps=1).text == ""
    second = workspace.maintain(worker, corrected, max_seeds=3)
    updated = reopened.attach(fresh, pin=second)
    assert fresh(question, memory=updated, steps=1).text == "gold"
    assert fresh(question, memory=reopened.attach(fresh, pin=first), steps=1) == answer



@pytest.mark.parametrize("stores_terminal", [False, True])
def test_native_value_substitution_distinguishes_late_observation_from_terminal(tmp_path, stores_terminal):
    from experiments.memory.native_workspace import NativeWorkspace

    class Interior(torch.nn.Module):
        def forward(self, hidden_states, *, prefix, worker):
            if prefix == 0:
                # The image is a premise. Only a cue can initiate its own route.
                if worker and torch.count_nonzero(hidden_states[..., 3:5]):
                    return torch.zeros_like(hidden_states)
                return hidden_states
            output = torch.zeros_like(hidden_states)
            routes = hidden_states[0, -1, 3:5]
            if float(routes.max()) > 0:
                route = int(routes.argmax())
                output[..., 1 + route] = output[..., 3 + route] = 1
            if worker and prefix == 2:
                output[..., :3] = 0
                if stores_terminal:
                    output[..., 6] = torch.sign(hidden_states[..., 5])
            elif not worker:
                output[..., 5:7] = hidden_states[..., 5:7]
            return output

    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity(), Interior(), torch.nn.Identity()])

        def forward(self, *, inputs_embeds, past_key_values, use_cache, return_dict):
            assert return_dict
            prefix = 0 if past_key_values is None else past_key_values.get_seq_length()
            worker = not torch.count_nonzero(inputs_embeds) if past_key_values is None else past_key_values.worker
            state = self.layers[0](inputs_embeds)
            state = self.layers[1](state, prefix=prefix, worker=worker)
            state = self.layers[2](state)
            cache = _Cache(prefix + inputs_embeds.shape[1]) if use_cache else None
            if cache is not None:
                cache.worker = worker
            return SimpleNamespace(last_hidden_state=state, past_key_values=cache)

    class MixedModalBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(max_position_embeddings=12, stores_terminal=stores_terminal)
            self.base_model = Body()
            rows = torch.zeros(10, 8)
            rows[1, 0] = rows[2, 1] = 1
            rows[3, 0] = 0.2
            rows[4, 1] = rows[4, 5] = rows[5, 2] = 1
            rows[5, 5] = -1
            self.lookup = torch.nn.Embedding.from_pretrained(rows)

        def get_input_embeddings(self):
            return self.lookup

        def get_output_embeddings(self):
            def score(hidden):
                chosen = torch.where(hidden[..., 6].abs() > 0.125, hidden[..., 6], hidden[..., 5])
                logits = hidden.new_full((1, 1, 10), -1)
                logits[..., 0] = 0.125
                logits[..., 7], logits[..., 8] = chosen, -chosen
                return logits
            return score

    cue, picture, left, right = (letter * 32 for letter in "abcd")
    pixmap = b"P6\n1 1\n255\n"
    blue, gold = pixmap + b"\x00\x00\xff", pixmap + b"\xff\xc0\x00"

    def ingress(source, raw):
        if source != picture or raw not in (blue, gold):
            return None
        return (1, 0, 0, int(raw == blue), int(raw == gold), 0, 0, 0)

    query = np.eye(8, 3)
    values = np.diag((0, 0, 0, 1, 1, 1, 0, 0))
    kernel = LinearFeedbackKernel(query=query, carry=None, observation_key=query, observation_value=values,
                                  feedback_key=query, feedback_value=np.zeros((8, 8)), max_steps=2)

    def make_reader():
        return NativeLatentModel(MixedModalBackbone().eval(), _Tokenizer(), kernel, source_chars=4,
                                 context_tokens=12, latent_steps=2, output_tokens=2, representation_scale=0.25,
                                 memory_gain=1, null_weight=0.1, derived_gain=3, eos_ids=(0,), top_k=1,
                                 source_ingress=ingress, source_ingress_sha256=sha256(b"two genuine PPM rows"),
                                 residual_layer=1, source_tokenizer_sha256=sha256(b"fixture tokenizer mapping"))

    limit, runtime = 65_536, sha256(b"mixed modal source mediation fixture")
    opening_dir, edited_dir = tmp_path / "opening", tmp_path / "edited"
    opening = write_source_delta(opening_dir, {cue: "\u00e9cho".encode(), picture: blue, left: b"blue", right: b"gold"},
                                 binary_sources=(picture,), max_bytes=limit)
    edited = write_source_delta(edited_dir, {picture: gold}, previous=opening,
                                binary_sources=(picture,), max_bytes=limit)
    worker = make_reader()
    source_world = worker.publish(opening_dir, opening.seal, max_source_bytes=limit)
    directory = tmp_path / "workspace"
    workspace = NativeWorkspace.create(directory, worker, source_world, runtime_sha256=runtime, max_bytes=limit)
    initial = workspace.maintain(worker, workspace.head(), max_seeds=1)
    corrected = workspace.revise(worker, initial, edited_dir, edited.seal, max_source_bytes=limit)
    updated = workspace.maintain(worker, corrected, max_seeds=4)

    fresh = make_reader()
    store = NativeWorkspace(directory, runtime_sha256=runtime, max_bytes=limit)
    original, donor = (store.attach(fresh, pin=pin) for pin in (initial, updated))
    seed, = original.derived_seeds
    assert seed == original.source_spans(cue)[0]
    assert donor.derived_seeds == (seed,)
    question = "What follows?"
    ids = fresh.task_token_ids(question)
    assert fresh(question, memory=donor).text == "gold"
    paired = fresh.compare_continuation_value(ids, original, seed, donor, seed)
    assert paired.intact == fresh(question, memory=original)
    assert paired.original_q8_squared_norm == paired.donor_q8_squared_norm > 0
    assert paired.intact.text == "blue"
    assert paired.rerouted.text == "gold"
    assert paired.source_clamped.text == ("gold" if stores_terminal else "blue")
    assert paired.intact.source_reads[1].selected[0].span.source_id == left
    assert paired.rerouted.source_reads[1].selected[0].span.source_id == right
    assert paired.source_clamped.source_reads[1] == paired.intact.source_reads[1]
    assert paired.source_clamped.value_substitution.live_suffix_reads == paired.rerouted.source_reads[1:]
    assert paired.intact.derived_steps[0] == paired.rerouted.derived_steps[0]
    receipt = paired.rerouted.value_substitution
    assert receipt.step == 1 and receipt.input_max_abs_delta > 0
    assert receipt.donor_publication_seal == updated.publication_seal and receipt.donor_source_seal == edited.seal
    assert paired.source_clamped.value_substitution.input_max_abs_delta == receipt.input_max_abs_delta
    assert "value_substitution" not in paired.intact.json_record()
    for invalid in (paired.rerouted, paired.source_clamped):
        assert invalid.json_record()["value_substitution"]["step"] == 1
        with pytest.raises(ValueError, match="evidence spans require"):
            original.evidence_spans(invalid)

    sham = fresh.compare_continuation_value(ids, original, seed, original, seed)
    assert sham.rerouted.value_substitution.input_max_abs_delta == 0
    assert replace(sham.source_clamped, value_substitution=None) == sham.intact
    other_ids = fresh.task_token_ids("What else?")
    untouched = fresh.compare_continuation_value(other_ids, original, seed, donor, seed)
    assert untouched.source_clamped.value_substitution.step is None
    assert untouched.source_clamped.value_substitution.live_suffix_reads == ()
    assert replace(untouched.source_clamped, value_substitution=None) == untouched.intact
    last = fresh.compare_continuation_value(ids, original, seed, donor, seed, steps=1)
    assert last.source_clamped.value_substitution.step == 1
    assert last.source_clamped.value_substitution.live_suffix_reads == ()
    assert replace(last.source_clamped, value_substitution=last.rerouted.value_substitution) == last.rerouted

    with pytest.raises(ValueError, match="intact compatible"):
        fresh.compare_continuation_value(ids, original, seed, fresh.without_derived(donor), seed)
    foreign = store.attach(make_reader(), pin=updated)
    with pytest.raises(ValueError, match="different native reader"):
        fresh.compare_continuation_value(ids, original, seed, foreign, seed)
    other_runtime = replace(donor, _publication_runtime_sha256=sha256(b"different runtime"))
    with pytest.raises(ValueError, match="one publication runtime"):
        fresh.compare_continuation_value(ids, original, seed, other_runtime, seed)

def test_random_llama_boundary_injects_the_exact_row_and_releases_hooks(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from experiments.memory.native_latent_recurrence import _RESIDUAL_CALL

    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(vocab_size=10, hidden_size=6, intermediate_size=12, head_dim=6,
                                      num_hidden_layers=3, num_attention_heads=1, num_key_value_heads=1,
                                      max_position_embeddings=16, eos_token_id=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = transformers.LlamaForCausalLM(config).eval()
    tokenizer_identity = sha256(b"supplied fixture token mapping")
    residual = dict(residual_layer=1, source_tokenizer_sha256=tokenizer_identity)
    reader = _reader(model, top_k=1, **residual)
    for invalid in (True, 0, 2):
        with pytest.raises(ValueError, match="eager backbone layer index"):
            _reader(model, top_k=1, residual_layer=invalid)
    assert reader.compare_greedy("What follows?").matches

    layer = model.base_model.layers[1]
    seen = []

    def observe(_layer, args, kwargs, output):
        hidden = args[0] if args else kwargs["hidden_states"]
        if isinstance(output, tuple):
            output = output[0]
        seen.append((hidden.clone(), output[:, -1:, :].clone()))

    with torch.inference_mode():
        embeddings = reader._embed_ids((1, 2))
        state, cache, final = reader._latent_backbone(embeddings, retain_cache=True)
        ordinary, alternative_cache = reader._backbone(embeddings, retain_cache=True)
        torch.testing.assert_close(final, ordinary, atol=0, rtol=0)
        assert not torch.equal(final, state)
        thought = state * 0.5
        with layer.register_forward_hook(observe, with_kwargs=True):
            next_state, advanced, _ = reader._latent_backbone(thought, cache, retain_cache=True, inject=True)
        assert advanced.get_seq_length() == 3
        supplied, captured = seen.pop()
        torch.testing.assert_close(supplied, thought, atol=0, rtol=0)
        torch.testing.assert_close(captured, next_state, atol=0, rtol=0)
        _, alternative_cache, _ = reader._latent_backbone(-thought, alternative_cache,
                                                           retain_cache=True, inject=True)
        for attr in ("keys", "values"):
            early = getattr(advanced.layers[0], attr)
            torch.testing.assert_close(early, getattr(alternative_cache.layers[0], attr), atol=0, rtol=0)
            assert not torch.equal(getattr(advanced.layers[1], attr), getattr(alternative_cache.layers[1], attr))

        prefix_keys = [slot.keys.clone() for slot in advanced.layers]
        _, advanced, _ = reader._latent_backbone(next_state * 0.5, advanced, retain_cache=True, inject=True)
        assert [slot.keys.shape[-2] for slot in advanced.layers] == [4, 4, 4]
        _, advanced = reader._backbone(reader._embed_ids((0,)), advanced, retain_cache=True)
        for position, prefix in enumerate(prefix_keys):
            assert advanced.layers[position].keys.shape[-2] == 5
            torch.testing.assert_close(advanced.layers[position].keys[..., :3, :], prefix, atol=0, rtol=0)

        with layer.register_forward_hook(lambda *_: torch.full_like(embeddings, float("nan"))):
            with pytest.raises(FloatingPointError, match="native residual state"):
                reader._latent_backbone(embeddings, retain_cache=True, inject=True)
        assert not layer._forward_hooks and not layer._forward_pre_hooks

    alternate_reader = _reader(model, top_k=1, **residual)
    assert alternate_reader._residual_lock is reader._residual_lock
    native_entered, release_native = Event(), Event()
    ordinary_snapshotted, release_ordinary = Event(), Event()

    def pause_native_call(_layer, _args, _kwargs):
        if _RESIDUAL_CALL.get() is not None:
            native_entered.set()
            assert release_native.wait(10)

    def pause_ordinary_call(_layer, _args, _kwargs):
        if _RESIDUAL_CALL.get() is None:
            ordinary_snapshotted.set()
            assert release_ordinary.wait(10)

    def advance_in_thread():
        with torch.inference_mode():
            return alternate_reader._latent_backbone(thought, retain_cache=True, inject=True)

    def decode_in_thread():
        with torch.inference_mode():
            return reader._backbone(embeddings, retain_cache=True)

    with (model.base_model.layers[0].register_forward_pre_hook(pause_native_call, with_kwargs=True),
          layer.register_forward_pre_hook(pause_ordinary_call, with_kwargs=True),
          ThreadPoolExecutor(max_workers=2) as pool):
        native = pool.submit(advance_in_thread)
        try:
            assert native_entered.wait(10)
            raw = pool.submit(decode_in_thread)
            assert ordinary_snapshotted.wait(10)
            release_native.set()
            assert native.result(timeout=10)[1].get_seq_length() == 1
        finally:
            release_native.set()
            release_ordinary.set()
        concurrent, _ = raw.result(timeout=10)
        torch.testing.assert_close(concurrent, ordinary, atol=0, rtol=0)
    assert not layer._forward_hooks and not layer._forward_pre_hooks

    moments = _reader(model, **residual)
    source_path = tmp_path / "source"
    delta = write_source_delta(source_path, {"b" * 32: b"blue"})
    live = moments.publish(source_path, delta.seal)
    assert live.rows == 1 and live._reader_fingerprint_sha256 is not None
    assert moments("What follows?", memory=live).source_seal == delta.seal

    from experiments.memory.native_publication import read_native_publication, write_native_publication

    tokenizers = pytest.importorskip("tokenizers")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[PAD]": 0, "[UNK]": 1, "blue": 4},
                                                               unk_token="[UNK]"))
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]")

    def set_transient_options():
        fast.backend_tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=3)
        fast.backend_tokenizer.enable_truncation(max_length=4)

    set_transient_options()
    fast_reader = _reader(model, top_k=1, residual_layer=1, tokenizer=fast)
    fast_rows = fast_reader.publish(source_path, delta.seal)
    directory = tmp_path / "fast-publication"
    runtime = sha256(b"fixed-source residual fixture")
    seal = write_native_publication(directory, fast_reader, fast_rows, runtime_sha256=runtime, max_bytes=65_536)
    set_transient_options()
    assert read_native_publication(directory, fast_reader, seal=seal, source_seal=delta.seal,
                                   runtime_sha256=runtime, max_bytes=65_536).rows == 1


def test_chat_task_matches_plain_greedy_and_pins_its_template(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from experiments.memory.native_publication import read_native_publication, write_native_publication

    words = ("<pad>", "<s>", "</s>", "<user>", "<assistant>", "Which", "route", "blue", "ends", "<unk>")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(dict(zip(words, range(len(words)))),
                                                                unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    template = ("{{ bos_token }} <user> {{ messages[0].content }}"
                "{% if add_generation_prompt %} <assistant>{% endif %}")
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, chat_template=template,
                                                     bos_token="<s>", eos_token="</s>", unk_token="<unk>")
    config = transformers.LlamaConfig(vocab_size=len(words), hidden_size=6, intermediate_size=12, head_dim=6,
                                      num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
                                      max_position_embeddings=16, eos_token_id=2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = transformers.LlamaForCausalLM(config).eval()

    def make_reader(task_template, eos_ids=(2,)):
        return NativeLatentModel(model, tokenizer, _kernel(), source_chars=32, context_tokens=16,
                                 latent_steps=2, output_tokens=3, representation_scale=0.25,
                                 memory_gain=1, null_weight=0.1, top_k=1, eos_ids=eos_ids,
                                 task_template=task_template)

    question = "Which route"
    reader = make_reader(template)
    pair = reader.compare_greedy(question)
    assert pair.task_ids == (1, 3, 5, 6, 4)
    assert pair.matches and pair.native == reader(question, steps=0)
    assert pair.native.source_seal is None and pair.native.latent_steps == 0

    # Use the actual first greedy token as an additional EOS to exercise early termination.
    stop_id = pair.reference_ids[0]
    stopped = make_reader(template, eos_ids=tuple({2, stop_id})).compare_greedy(question)
    assert stopped.matches and stopped.reference_ids == (stop_id,) and stopped.native.terminated

    model.generation_config.min_new_tokens = 2
    reference_config = transformers.GenerationConfig(max_new_tokens=3, do_sample=False, use_cache=True,
                                                     eos_token_id=[2, stop_id], pad_token_id=0)
    input_ids = torch.tensor([pair.task_ids])
    inherited = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                               generation_config=reference_config)
    assert int(inherited[0, len(pair.task_ids)]) != stop_id
    assert make_reader(template, eos_ids=tuple({2, stop_id})).compare_greedy(question) == stopped

    ordinary_forward = model.forward

    def different_reference(**kwargs):
        state = ordinary_forward(**kwargs)
        changed = torch.zeros_like(state.logits)
        changed[..., (stop_id + 1) % len(words)] = 1
        state.logits = changed
        return state

    with monkeypatch.context() as patch:
        patch.setattr(model, "forward", different_reference)
        difference = reader.compare_greedy(question)
        assert not difference.matches and difference.native == pair.native

    long_question = " ".join(["route"] * 9)
    assert len(reader.task_token_ids(long_question, steps=0)) == 12
    with pytest.raises(ValueError, match="context bound"):
        reader(long_question, steps=2)

    # Source rows stay in the same plain encoding when task framing is enabled.
    source = "a" * 32
    feed = write_source_delta(tmp_path / "feed", {source: b"blue"})
    world = reader.publish(tmp_path / "feed", feed.seal)
    plain = make_reader(None)
    plain_world = plain.publish(tmp_path / "feed", feed.seal)
    np.testing.assert_array_equal(world._sources[source][0].row, plain_world._sources[source][0].row)
    board, runtime = tmp_path / "sealed", sha256(b"greedy control test runtime")
    seal = write_native_publication(board, reader, world, runtime_sha256=runtime, max_bytes=65_536)
    changed_template = template.replace("<user>", "")
    tokenizer.chat_template = {"default": template}
    fresh = make_reader("default")
    tokenizer.chat_template = {"default": changed_template}
    assert reader.task_token_ids(question, steps=0) == pair.task_ids
    assert fresh.task_token_ids(question, steps=0) == pair.task_ids
    reopened = read_native_publication(board, fresh, seal=seal, source_seal=feed.seal,
                                      runtime_sha256=runtime, max_bytes=65_536)
    assert fresh(question, memory=reopened) == replace(reader(question, memory=world), publication_seal=seal)
    for changed in (plain, make_reader("default")):
        with pytest.raises(ValueError, match="reader differs"):
            read_native_publication(board, changed, seal=seal, source_seal=feed.seal,
                                    runtime_sha256=runtime, max_bytes=65_536)
