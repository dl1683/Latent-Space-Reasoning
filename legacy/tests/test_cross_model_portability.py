"""Regression tests for the assumptions that broke on Gemma 4.

Each test here corresponds to a defect found while porting the perturbation
experiments off Qwen3-on-Windows. They use synthetic modules and stub configs so
they run without a GPU or any model download.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from latent_reasoning.utils.architecture import (
    decoder_layers,
    hidden_size,
    num_decoder_layers,
    text_config,
)
from latent_reasoning.utils.quantization import resolve_load_dtype


class _Config:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _CompositeConfig(_Config):
    """Mimics a multimodal config: text hyperparameters live one level down."""

    def get_text_config(self):
        return self.text_config


def _stack(n):
    return nn.ModuleList(nn.Linear(2, 2) for _ in range(n))


# --------------------------------------------------------------------------
# hidden_size / text_config
# --------------------------------------------------------------------------

def test_hidden_size_from_flat_config():
    model = _Config(config=_Config(hidden_size=2560))
    assert hidden_size(model) == 2560


def test_hidden_size_from_composite_config():
    """Gemma 4 has no config.hidden_size; it is under config.text_config."""
    cfg = _CompositeConfig(text_config=_Config(hidden_size=5376))
    model = _Config(config=cfg)
    assert hidden_size(model) == 5376
    assert text_config(model).hidden_size == 5376


def test_hidden_size_missing_raises():
    model = _Config(config=_Config())
    with pytest.raises(ValueError, match="hidden_size"):
        hidden_size(model)


# --------------------------------------------------------------------------
# decoder_layers
# --------------------------------------------------------------------------

class _PlainCausalLM(nn.Module):
    def __init__(self, n=36):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = _stack(n)


class _MultimodalLM(nn.Module):
    """Gemma 4 layout: text decoder beside a vision tower, both named `layers`."""

    def __init__(self, n_text=60, n_vision=27):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = _stack(n_text)
        self.model.vision_tower = nn.Module()
        self.model.vision_tower.encoder = nn.Module()
        self.model.vision_tower.encoder.layers = _stack(n_vision)


def test_decoder_layers_plain():
    assert num_decoder_layers(_PlainCausalLM(36)) == 36


def test_decoder_layers_prefers_text_tower_over_vision():
    """The vision tower must never be mistaken for the decoder."""
    assert num_decoder_layers(_MultimodalLM(n_text=60, n_vision=27)) == 60


def test_decoder_layers_absent_returns_none():
    assert decoder_layers(nn.Linear(2, 2)) is None
    assert num_decoder_layers(nn.Linear(2, 2)) == 0


# --------------------------------------------------------------------------
# resolve_load_dtype
# --------------------------------------------------------------------------

CPU = torch.device("cpu")


def test_dtype_honours_declared_checkpoint_dtype():
    """A bfloat16-native checkpoint must not be silently loaded as float16."""
    assert resolve_load_dtype(CPU, config=_Config(dtype=torch.bfloat16)) is torch.bfloat16


def test_dtype_accepts_string_declaration():
    assert resolve_load_dtype(CPU, config=_Config(dtype="bfloat16")) is torch.bfloat16


def test_dtype_reads_legacy_torch_dtype_field():
    assert resolve_load_dtype(CPU, config=_Config(torch_dtype="float16")) is torch.float16


def test_dtype_override_wins():
    cfg = _Config(dtype=torch.bfloat16)
    assert resolve_load_dtype(CPU, config=cfg, override="float32") is torch.float32


def test_dtype_falls_back_when_config_silent():
    assert resolve_load_dtype(CPU, config=_Config()) is torch.float32


def test_dtype_rejects_nonsense_override():
    with pytest.raises(ValueError):
        resolve_load_dtype(CPU, override="quadruple")


# --------------------------------------------------------------------------
# stop_token_ids / split_after_reasoning
# --------------------------------------------------------------------------

class _Tokenizer:
    def __init__(self, eos_id, vocab, unk_id=None):
        self.eos_token_id = eos_id
        self.unk_token_id = unk_id
        self._vocab = vocab

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, self.unk_token_id)


def _model_with_stop_ids(ids):
    m = nn.Module()
    m.generation_config = _Config(eos_token_id=ids)
    return m


def test_stop_ids_include_turn_end_not_just_eos():
    """Gemma ends a turn on <turn|> (106), while its eos_token is <eos> (1).

    Passing only the tokenizer's eos to generate() overrides the checkpoint's
    configured stop set, so the model runs past the end of its answer.
    """
    from harness import stop_token_ids

    tok = _Tokenizer(eos_id=1, vocab={"<turn|>": 106})
    ids = stop_token_ids(_model_with_stop_ids([1, 106, 50]), tok)
    assert set(ids) == {1, 106, 50}


def test_stop_ids_handle_scalar_config():
    from harness import stop_token_ids

    tok = _Tokenizer(eos_id=151645, vocab={})
    assert stop_token_ids(_model_with_stop_ids(151645), tok) == [151645]


def test_stop_ids_union_when_tokenizer_disagrees():
    from harness import stop_token_ids

    tok = _Tokenizer(eos_id=7, vocab={})
    assert set(stop_token_ids(_model_with_stop_ids([1, 2]), tok)) == {1, 2, 7}


# --------------------------------------------------------------------------
# Decimal-aware answer extraction
# --------------------------------------------------------------------------

def test_decimal_final_answer_is_not_split():
    """"412.5" must not parse as [412, 5] and extract as 5."""
    from harness import extract_answer, verify_answer

    resp = "Total: 550. Given away: 137.5. Remaining: 412.5"
    assert extract_answer(resp) is None       # not an integer answer
    assert verify_answer(resp, 413) is False  # and it is wrong
    assert verify_answer(resp, 5) is False    # emphatically not "5"


def test_decimal_answer_does_not_fall_back_to_earlier_integer():
    """Dropping the decimal must not promote an intermediate step to 'the answer'."""
    from harness import extract_answer

    assert extract_answer("Step 1: 550. Final: 412.5") is None


def test_integral_answers_unaffected():
    from harness import extract_answer, verify_answer

    assert extract_answer("The answer is 1140") == 1140
    assert extract_answer("Final Answer:\n6193") == 6193
    assert verify_answer("...so 159", 159) is True


def test_comma_grouping_still_handled():
    from harness import extract_answer

    assert extract_answer("57 * 20 = 1,140") == 1140


def test_trailing_decimal_zero_counts_as_integer():
    from harness import extract_answer, verify_answer

    assert extract_answer("The result is 413.0") == 413
    assert verify_answer("The result is 413.0", 413) is True


def test_negative_answers_preserved():
    from harness import extract_answer

    assert extract_answer("which gives -42") == -42


def test_dense_score_uses_whole_decimal():
    """dense_score carried the same split; 412.5 is near 413, not near 5."""
    from harness import dense_score

    assert dense_score("Remaining: 412.5", 413) > dense_score("Remaining: 5", 413)


def test_split_after_reasoning_qwen_style():
    from harness import split_after_reasoning

    tok = _Tokenizer(eos_id=0, vocab={"</think>": 99})
    assert split_after_reasoning([5, 6, 99, 7, 8], tok) == [7, 8]


def test_split_after_reasoning_gemma_style():
    from harness import split_after_reasoning

    tok = _Tokenizer(eos_id=0, vocab={"<channel|>": 105})
    assert split_after_reasoning([5, 105, 7], tok) == [7]


def test_split_after_reasoning_uses_last_closer():
    """Multiple channels: the answer follows the final close, not the first."""
    from harness import split_after_reasoning

    tok = _Tokenizer(eos_id=0, vocab={"<channel|>": 105})
    assert split_after_reasoning([105, 1, 105, 2, 3], tok) == [2, 3]


def test_split_after_reasoning_unterminated_is_none():
    """An unterminated trace is a truncation, not an answer."""
    from harness import split_after_reasoning

    tok = _Tokenizer(eos_id=0, vocab={"</think>": 99})
    assert split_after_reasoning([5, 6, 7], tok) is None


def test_split_after_reasoning_absent_marker_is_none():
    from harness import split_after_reasoning

    tok = _Tokenizer(eos_id=0, vocab={}, unk_id=3)
    assert split_after_reasoning([1, 2, 3], tok) is None
