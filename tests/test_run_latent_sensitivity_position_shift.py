import torch

from experiments.run_latent_sensitivity import run_zero_shot


def test_run_zero_shot_applies_position_offset_without_extra_prompt_tokens():
    encoder = _DummyEncoder()

    response, raw, meta = run_zero_shot(
        encoder,
        "What is 40 + 2?",
        max_new_tokens=8,
        enable_thinking=False,
        position_offset=2,
    )

    kwargs = encoder.model.last_generate_kwargs
    assert kwargs["input_ids"].tolist() == [[10, 11, 12]]
    assert kwargs["attention_mask"].tolist() == [[1, 1, 1]]
    assert kwargs["position_ids"].tolist() == [[2, 3, 4]]
    assert meta["prompt_tokens"] == 3
    assert meta["generated_tokens"] == 2
    assert meta["terminated_by_eos"] is True
    assert response == "42"
    assert raw == "42"


def test_run_zero_shot_omits_position_ids_when_offset_is_zero():
    encoder = _DummyEncoder()

    run_zero_shot(
        encoder,
        "What is 40 + 2?",
        max_new_tokens=8,
        enable_thinking=False,
    )

    assert "position_ids" not in encoder.model.last_generate_kwargs


class _DummyEncoder:
    def __init__(self):
        self._device = "cpu"
        self.tokenizer = _DummyTokenizer()
        self.model = _DummyModel()


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert messages[-1]["content"] == "What is 40 + 2?"
        return "formatted prompt"

    def __call__(self, text, return_tensors):
        assert text == "formatted prompt"
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "42"


class _DummyModel:
    def __init__(self):
        self.last_generate_kwargs = None

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        return torch.tensor([[10, 11, 12, 42, 99]], dtype=torch.long)
