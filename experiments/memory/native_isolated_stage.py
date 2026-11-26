"""Bounded CPU stand-in for the native publication software control."""

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.memory.linear_feedback import LinearFeedbackKernel
from experiments.memory.native_latent_recurrence import NativeLatentModel
from experiments.memory.native_publication import read_native_publication, write_native_publication


MAX_BYTES = 64 * 1024
LABELS = tuple(f"L{index:02d}" for index in range(12))
_WORDS = ("<stop>", "link", "name", "follow", *LABELS)


class _Tokenizer:
    def encode(self, text, *, add_special_tokens, truncation):
        role, *labels = text.split()
        if (role not in ("link", "name", "follow") or len(labels) != (1 if role == "follow" else 2)
                or any(label not in LABELS for label in labels)):
            raise ValueError("expected a link, name or follow clause over the fixture vocabulary")
        return [_WORDS.index(word) for word in (role, *labels)]

    def decode(self, ids, *, skip_special_tokens):
        return " ".join(_WORDS[token] for token in ids if token)


class _Cache:
    def __init__(self, length, address):
        self.length = length
        self.address = address

    def get_seq_length(self):
        return self.length


class _CausalControl(torch.nn.Module):
    """Known role grammar with a reusable vocabulary and a continuous relay."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(max_position_embeddings=12, vocabulary=_WORDS)
        size = len(LABELS)
        self.width = 3 * size + 1
        tokens = torch.zeros((len(_WORDS), self.width))
        tokens[:, -1] = torch.arange(1, len(_WORDS) + 1)
        tokens[4:, :size] = torch.eye(size)
        self.lookup = torch.nn.Embedding.from_pretrained(tokens)
        self.calls = 0

    def get_input_embeddings(self):
        return self.lookup

    def get_output_embeddings(self):
        return self._scores

    def base_model(self, *, inputs_embeds, past_key_values, use_cache, return_dict):
        self.calls += 1
        size = len(LABELS)
        hidden = torch.zeros_like(inputs_embeds[:, -1:, :])
        length = inputs_embeds.shape[1]
        address = hidden[..., :size]
        if past_key_values is None:
            role = int(inputs_embeds[0, 0, -1])
            if role == 0:
                # A worker starts from the stored numeric row, with no token marker.
                address.copy_(inputs_embeds[..., :size])
            else:
                address.copy_(inputs_embeds[:, 1:2, :size])
                if role == 2:
                    hidden[..., size:2 * size] = inputs_embeds[:, 2:3, :size]
                elif role == 3:
                    hidden[..., 2 * size:3 * size] = inputs_embeds[:, 2:3, :size]
        else:
            length += past_key_values.length
            if not inputs_embeds[0, 0, -1]:
                novel = inputs_embeds[..., :size] - past_key_values.address
                route = inputs_embeds[..., size:2 * size]
                direction = novel if novel.max() > 1e-6 else route
                if direction.max() > 1e-6:
                    address[..., int(direction.argmax())] = 1
                hidden[..., 2 * size:3 * size] = inputs_embeds[..., 2 * size:3 * size]
        cache = _Cache(length, address) if use_cache else None
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=cache)

    def _scores(self, hidden):
        logits = hidden.new_full((1, 1, len(_WORDS)), -1)
        logits[..., 0] = 0.125
        logits[..., 4:] = hidden[..., 2 * len(LABELS):3 * len(LABELS)]
        return logits


def _reader(model, *, tokenizer=None, **source_options):
    size = len(LABELS)
    query = np.eye(model.width, size)
    value = np.zeros((model.width, model.width))
    value[size:3 * size, size:3 * size] = np.eye(2 * size)
    kernel = LinearFeedbackKernel(
        query=query, carry=None, observation_key=query, observation_value=value,
        feedback_key=query, feedback_value=np.zeros_like(value), max_steps=2,
    )
    return NativeLatentModel(
        model, _Tokenizer() if tokenizer is None else tokenizer, kernel, source_chars=32, context_tokens=12,
        latent_steps=2,
        top_k=1, output_tokens=2, representation_scale=0.25, memory_gain=1, null_weight=0.1,
        eos_ids=(0,), **source_options,
    )


def main(kind, args):
    model = _CausalControl().eval()
    reader = _reader(model)
    runtime, source, mode, *parent = args

    def attach(seal, source_seal):
        return read_native_publication(
            "/publication", reader, seal=seal, source_seal=source_seal,
            runtime_sha256=runtime, max_bytes=MAX_BYTES,
        )

    if kind in ("publish", "revise"):
        previous = attach(*parent) if kind == "revise" else None
        if model.calls:
            raise ValueError("attachment called the backbone")
        memory = reader.publish(Path("/source"), source, previous=previous)
        source_calls = model.calls
        memory = reader.maintain(memory, max_seeds=2) if previous is None else memory
        seal = write_native_publication(
            "/output/publication", reader, memory, runtime_sha256=runtime, max_bytes=MAX_BYTES,
        )
        receipt = dict(seal=seal, source_seal=memory.seal, rows=memory.rows,
                       source_backbone_calls=source_calls, worker_backbone_calls=model.calls - source_calls,
                       seeds=[dict(span=asdict(seed), reads=[asdict(step) for step in memory.continuation_reads(seed)])
                              for seed in memory.derived_seeds])
    else:
        memory = None if mode == "none" else attach(*parent)
        if model.calls:
            raise ValueError("attachment called the backbone")
        if mode == "source-only":
            memory = reader.without_derived(memory)
        read = reader(Path("/task.txt").read_text(encoding="utf8"), memory=memory)
        receipt = {**read.json_record(), "backbone_calls": model.calls}
    print(json.dumps(receipt, sort_keys=True))
