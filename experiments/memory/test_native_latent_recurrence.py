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
        self.config = SimpleNamespace(max_position_embeddings=16)
        rows = torch.zeros((10, 6))
        rows[1, 0] = rows[2, 2] = 1
        rows[3, 0] = rows[3, 3] = 1
        rows[4, 1] = rows[5, 1] = 1
        rows[4, 4], rows[5, 4] = 1, -1
        rows[6, 2] = rows[6, 5] = 1
        self._lookup = torch.nn.Embedding.from_pretrained(rows)
        self.calls = []
        self._clear_previous = clear_previous

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
            if self._clear_previous:
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


def _reader(backbone, *, terminal_axis=4, latent_steps=2, top_k=None):
    return NativeLatentModel(backbone, _Tokenizer(), _kernel(terminal_axis=terminal_axis),
                             source_chars=4, context_tokens=12, latent_steps=latent_steps, top_k=top_k,
                             output_tokens=2, representation_scale=0.25, memory_gain=1, null_weight=0.1,
                             eos_ids=(0,))


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
    assert removed == reader(question, memory=opening)
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
