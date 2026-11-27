from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.memory.compact_episode_constraint import CompactSearchReadConstraint
from experiments.memory.linear_feedback import LinearFeedbackKernel
from experiments.memory.native_latent_recurrence import NativeLatentModel
from experiments.memory.native_publication import read_native_publication, write_native_publication
from experiments.memory.native_task_episode import NativeTaskEpisodeBackend
from experiments.memory.source_revision_feed import write_source_delta
from experiments.memory.task_episode import EpisodeLimits, EpisodePolicy, _packed, read_episode_pair, run_episode
from experiments.memory.task_evidence import TaskWorld
from experiments.memory.verified_source_tool import sha256


_RULES, _ROUTING, _LOG = (symbol * 32 for symbol in "abc")
_REVIEW = 'review-"log"'


class _CharacterTokenizer:
    eos_token_id = 0
    all_special_ids = [0]
    backend_tokenizer = SimpleNamespace(to_str=lambda: "fixture ASCII characters v1")

    def __len__(self):
        return 128

    def encode(self, text, **_):
        return list(text.encode("ascii"))

    def decode(self, ids, **_):
        return bytes(token for token in ids if token != self.eos_token_id).decode("ascii")

    def apply_chat_template(self, messages, **_):
        return self.encode(_packed(messages))


def _need(need_id, target, *, parent=None, discovered=()):
    return {
        "id": need_id, "parent": parent, "discovered_from": list(discovered), "target_or_event": target,
        "relation": "requires", "time_version_scope": "current pin", "evidence_sought": [target],
        "support_requirements": [], "defeat_requirements": [], "alternatives": [], "unresolved_conditions": [],
    }


def _action(tool, *additions):
    return {"state": "", "need_additions": list(additions), "tool": tool}


def _search(need_id, query):
    return {"op": "search", "need_id": need_id, "query": query}


def _read(need_id, candidate):
    return {"op": "read", "need_id": need_id, "candidate": candidate}


def _finding(need_id, stance, claim, citations):
    return {"need_id": need_id, "stance": stance, "claim": claim, "citations": citations, "reason": claim}


def _responses(frame):
    receipts = frame["receipts"]
    if frame["planning_phase"] == "declare":
        return {"plan": [_need("go-ahead", "release approval"), _need("staff", "duty assignment")]}

    match len(receipts):
        case 0:
            return _action(_search("go-ahead", "Cobalt"))
        case 1:
            candidates = [r["candidate"] for r in receipts[0]["result"]["candidates"]]
            assert len(candidates) == 2
            return _action(_read("go-ahead", candidates))
        case 2:
            return _action(_search(_REVIEW, "Juniper log"),
                           _need(_REVIEW, "review record", parent="go-ahead", discovered=(1,)))
        case 3:
            candidates = receipts[2]["result"]["candidates"]
            log = next(row["candidate"] for row in candidates if row["span"]["source_id"] == _LOG)
            wrong = next(row["candidate"] for row in receipts[0]["result"]["candidates"]
                         if row["candidate"] not in {item["candidate"] for item in candidates})
            return _action(_read(_REVIEW, log)), _action(_read(_REVIEW, wrong))
        case 4:
            return _action(_search("staff", "assigned duty"))
        case 5:
            return {"finish": {
                "findings": [
                    _finding("go-ahead", "supports", "The matching review log clears approval.", [0, 1, 2]),
                    _finding("staff", "unknown", "A confirmed duty assignment is missing.", [0]),
                    _finding(_REVIEW, "supports", "The Juniper log is the linked record.", [1, 2]),
                ],
                "open_questions": [{"need_id": "staff", "question": "Who is assigned to duty?"}],
            }}
    raise AssertionError("unexpected scripted frame")


def test_compact_constraint_runs_the_same_prefix_on_both_adapters(tmp_path):
    pytest.importorskip("lmformatenforcer")
    pytest.importorskip("jsonschema")
    torch = pytest.importorskip("torch")
    limits = EpisodeLimits(7, 3, 3, 0, 16384, 45000, 3000, 1024, batch_reads=True)
    tokenizer = _CharacterTokenizer()
    constraint = CompactSearchReadConstraint(tokenizer, limits)
    assert constraint.sha256 == CompactSearchReadConstraint(tokenizer, limits).sha256
    assert constraint.sha256 != CompactSearchReadConstraint(tokenizer, replace(limits, batch_reads=False)).sha256
    policy = EpisodePolicy("matched", "ordinary-script", "fixture-characters", "compact_needs",
                           output_constraint_sha256=constraint.sha256, compact_delivery=True)

    class Cache(SimpleNamespace):
        def get_seq_length(self):
            return self.length

    class ScriptedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(max_position_embeddings=limits.context_tokens)
            self.lookup = torch.nn.Embedding.from_pretrained(torch.zeros(len(tokenizer), 1))
            self.script = ()

        def get_input_embeddings(self):
            return self.lookup

        def get_output_embeddings(self):
            return self._head

        def _head(self, hidden):
            logits = hidden.new_zeros((1, 1, len(tokenizer)))
            logits[..., self.script[int(hidden.item())]] = 1
            return logits

        def base_model(self, *, inputs_embeds, past_key_values, use_cache, return_dict):
            cursor = 0 if past_key_values is None else past_key_values.cursor + 1
            length = inputs_embeds.shape[1] + (0 if past_key_values is None else past_key_values.length)
            cache = Cache(cursor=cursor, length=length) if use_cache else None
            return SimpleNamespace(last_hidden_state=inputs_embeds.new_full((1, 1, 1), cursor), past_key_values=cache)

        def forward(self, *, input_ids, past_key_values, use_cache, return_dict):
            state = self.base_model(inputs_embeds=self.lookup(input_ids), past_key_values=past_key_values,
                                    use_cache=use_cache, return_dict=return_dict)
            return SimpleNamespace(logits=self._head(state.last_hidden_state), past_key_values=state.past_key_values)

    def prepare(call):
        frame = json.loads(dict(call.messages)["user"])
        assert tuple(tokenizer.encode(_packed([{"role": k, "content": v} for k, v in call.messages]))) == call.input_ids
        response = _responses(frame)
        illegal = None
        if frame["planning_phase"] == "declare":
            illegal = {"plan": [_need("invented", "unknown", discovered=(2,))]}
        if type(response) is tuple:
            response, illegal = response
        if frame["planning_phase"] == "act_after_plan":
            findings = [_finding(need["id"], "hypothetical", "Pending evidence", [])
                        for need in frame["need_registry"]]
            illegal = {"finish": {"findings": findings, "open_questions": []}}
        if illegal is not None:
            allow = constraint(call)
            prefix = call.input_ids
            for token in (*tokenizer.encode(_packed(illegal)), tokenizer.eos_token_id):
                try:
                    allows_token = token in allow(prefix)
                except ValueError:
                    # The token parser can accept a structurally invalid
                    # completion. The independent output gate must refuse it.
                    break
                if not allows_token:
                    break
                prefix = (*prefix, token)
            else:
                raise AssertionError("decoder accepted an illegal phase or wrong-need candidate")
        ids = (*tokenizer.encode(_packed(response)), tokenizer.eos_token_id)
        assert len(ids) <= call.max_new_tokens
        return ids

    ordinary_model, native_model = ScriptedModel().eval(), ScriptedModel().eval()
    issued = []

    class ScriptedPlainConstraint(CompactSearchReadConstraint):
        def __call__(self, call):
            ordinary_model.script = prepare(call)
            issued.append(ordinary_model.script)
            return super().__call__(call)

    class ScriptedNativeConstraint(CompactSearchReadConstraint):
        def __call__(self, call):
            native_model.script = prepare(call)
            return super().__call__(call)

    plain_constraint = ScriptedPlainConstraint(tokenizer, limits)
    native_constraint = ScriptedNativeConstraint(tokenizer, limits)
    assert plain_constraint.sha256 == native_constraint.sha256 == constraint.sha256
    generate = plain_constraint.plain_backend(ordinary_model)

    feed = tmp_path / "feed"
    source = write_source_delta(feed, {
        _RULES: b"Release requires Cobalt approval and a confirmed duty assignment.",
        _ROUTING: b"Cobalt approval is recorded under Juniper in the separate review log.",
        _LOG: b"Juniper review log: approval cleared.",
    })
    world = TaskWorld.from_source_epoch(feed, source.seal, max_feed_bytes=25000,
                                        max_source_bytes=25000, max_passages=32)
    axes = np.ones((1, 1))
    kernel = LinearFeedbackKernel(query=axes, carry=None, observation_key=axes, observation_value=axes,
                                  feedback_key=axes, feedback_value=axes, max_steps=1)
    reader = NativeLatentModel(
        native_model, tokenizer, kernel, source_chars=128, context_tokens=limits.context_tokens,
        latent_steps=1, output_tokens=limits.max_new_tokens, representation_scale=1, memory_gain=1,
        null_weight=1, eos_ids=(0,), top_k=1,
    )
    runtime = sha256(b"scripted common grammar control")
    publication = tmp_path / "publication"
    memory = reader.publish(feed, source.seal, max_source_bytes=25000)
    seal = write_native_publication(publication, reader, memory, runtime_sha256=runtime, max_bytes=25000)
    memory = read_native_publication(publication, reader, seal=seal, source_seal=source.seal,
                                     runtime_sha256=runtime, max_bytes=25000)
    native_policy = replace(policy, backend_id="native-script")
    backend = NativeTaskEpisodeBackend(
        world, reader, memory, backend_id=native_policy.backend_id, tokenizer_id=policy.tokenizer_id,
        expected_runtime_sha256=runtime, output_token_allowlist=native_constraint,
        output_constraint_sha256=constraint.sha256,
        steps=0,
    )
    task = "Can we proceed and who is on duty?"
    reader._eos = frozenset({1})
    with pytest.raises(ValueError, match="stop tokens"):
        backend.opening_record(world, native_policy, limits, tokenizer)
    reader._eos = frozenset({0})
    with pytest.raises(ValueError, match="must match the compact decoder"):
        run_episode(task, world, replace(policy, output_constraint_sha256=None), limits, tokenizer, generate,
                    tmp_path / "unbound.jsonl")
    assert not (tmp_path / "unbound.jsonl").exists()
    plain = run_episode(task, world, policy, limits, tokenizer, generate, tmp_path / "plain.jsonl")
    native = run_episode(task, world, native_policy, limits, tokenizer, backend, tmp_path / "native.jsonl",
                         native_backend=backend)
    left, right = read_episode_pair(plain, native, contrast="native_text")
    for records in (left, right):
        last = records[-1]["data"]
        assert last["status"] == "completed", last
        assert sum(map(len, issued)) == last["usage"]["known_output_tokens"]
        assert {finding["need_id"] for finding in last["submission"]["findings"]} == {"go-ahead", "staff", _REVIEW}
    assert len(issued) == 7 and left[-1]["data"]["submission"] == right[-1]["data"]["submission"]
    left_replies = [row["data"] for row in left if row["kind"] == "response"]
    right_replies = [row["data"] for row in right if row["kind"] == "response"]
    assert [(r["text"], r["terminated"], r["output_tokens"]) for r in left_replies] == [
        (r["text"], r["terminated"], r["output_tokens"]) for r in right_replies
    ]
    assert [(*tokenizer.encode(row["text"]), tokenizer.eos_token_id) for row in left_replies] == issued
    assert [row["data"] for row in left if row["kind"] == "request"] == [
        row["data"] for row in right if row["kind"] == "request"
    ]
    assert [list(ids) for ids in issued] == [
        row["backend_receipt"]["output_token_ids"] for row in right_replies
    ]
