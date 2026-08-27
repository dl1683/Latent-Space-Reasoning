"""Token accounting on the soft-prompt (perturbation) decode path.

`generate(inputs_embeds=...)` returns ONLY the newly generated tokens -- there
are no input_ids for it to prepend. `decode_with_raw_soft_prompt` relies on that
when it decodes the whole returned tensor as the completion, but it used to also
subtract the prompt length when counting tokens. That under-counted every
perturbation generation by the prompt length and made
`generated_tokens >= max_new_tokens` unsatisfiable, so truncation in this arm was
invisible in the recorded diagnostics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from experiments.run_latent_sensitivity import decode_with_raw_soft_prompt

N_PROMPT_TOKENS = 3
N_SOFT_TOKENS = 2
N_GENERATED = 5
EMBED_DIM = 4


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 99
    unk_token_id = None

    def apply_chat_template(self, messages, **kwargs):
        return "formatted prompt"

    def __call__(self, text, return_tensors):
        return {
            "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
            "attention_mask": torch.ones(1, N_PROMPT_TOKENS, dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        return "the answer is 42"

    def convert_tokens_to_ids(self, token):
        return None


class _Embeddings:
    def __call__(self, input_ids):
        return torch.zeros(1, input_ids.shape[1], EMBED_DIM)


class _Model:
    dtype = torch.float32

    def __init__(self):
        self.generation_config = type("cfg", (), {"eos_token_id": [99]})()
        self.last_generate_kwargs = None

    def get_input_embeddings(self):
        return _Embeddings()

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        # Only the new tokens, as generate() does for inputs_embeds.
        return torch.tensor([[7, 7, 7, 7, 42]], dtype=torch.long)


class _Encoder:
    def __init__(self):
        self._device = "cpu"
        self.tokenizer = _Tokenizer()
        self.model = _Model()


def _decode():
    encoder = _Encoder()
    soft_prompt = torch.zeros(1, N_SOFT_TOKENS, EMBED_DIM)
    text, meta = decode_with_raw_soft_prompt(
        encoder, soft_prompt, "What is 40 + 2?", max_new_tokens=8,
    )
    return encoder, text, meta


def test_generated_tokens_counts_only_new_tokens():
    _, _, meta = _decode()
    assert meta["generated_tokens"] == N_GENERATED


def test_generated_tokens_is_not_reduced_by_prompt_length():
    """The old bug: n_generated = output.shape[1] - n_input."""
    _, _, meta = _decode()
    assert meta["generated_tokens"] != N_GENERATED - meta["prompt_tokens"]
    assert meta["generated_tokens"] > 0


def test_prompt_tokens_includes_the_soft_prompt():
    _, _, meta = _decode()
    assert meta["prompt_tokens"] == N_PROMPT_TOKENS + N_SOFT_TOKENS


def test_truncation_is_detectable():
    """With correct counting, hitting the cap is representable at all."""
    _, _, meta = _decode()
    assert meta["generated_tokens"] <= 8            # never exceeds max_new_tokens
    assert meta["generated_tokens"] >= meta["prompt_tokens"] - N_PROMPT_TOKENS


def test_stop_ids_come_from_the_generation_config():
    """Gemma's turn-end token lives here, not on tokenizer.eos_token_id."""
    encoder, _, _ = _decode()
    assert 99 in encoder.model.last_generate_kwargs["eos_token_id"]


def test_whole_returned_tensor_is_decoded_as_the_completion():
    _, text, _ = _decode()
    assert text == "the answer is 42"
