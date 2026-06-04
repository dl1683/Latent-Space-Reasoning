"""CPU-safe tests for language-diffusion experiment scaffolding."""

import torch

from latent_reasoning.diffusion import DiffusionGenerationConfig, HFDiffusionBackend
from latent_reasoning.diffusion.backends import _llada_apply_revision_remask, _llada_mask_token_id
from latent_reasoning.diffusion.candidates import (
    available_candidates,
    candidate_keys,
    get_candidate,
    is_llada_family,
)


def test_candidate_registry_has_default_gpu_targets():
    keys = candidate_keys()
    assert keys[0] == "dream-7b-instruct-hf"
    assert "llada-8b-instruct-hf" in keys
    assert "llada-moe-7b-a1b-instruct-hf" in keys
    assert "llada-moe-7b-a1b-instruct-gguf-q4" in keys

    dream = get_candidate("dream-7b-instruct-hf")
    assert dream.backend == "hf_custom"
    assert dream.supports_history is True
    assert dream.min_vram_gb == 20.0

    moe = get_candidate("llada-moe-7b-a1b-instruct-hf")
    assert moe.backend == "hf_custom"
    assert moe.family == "LLaDA MoE 7B-A1B"
    assert moe.priority > get_candidate("llada-8b-instruct-hf").priority
    assert moe.default_algorithm == "low_confidence"
    assert moe.min_vram_gb == 12.0


def test_candidate_records_are_json_friendly():
    records = [candidate.to_dict() for candidate in available_candidates()]
    assert records
    assert all(isinstance(record["sources"], list) for record in records)
    assert all("model_id" in record for record in records)


def test_generation_config_uses_llada_algorithm_default():
    llada = get_candidate("llada-8b-instruct-hf")
    config = DiffusionGenerationConfig().with_candidate_defaults(llada)
    assert config.algorithm == "low_confidence"
    assert config.history_sample_count == 5
    assert config.revision_remask_fraction is None
    assert config.revision_steps == 0
    assert config.system_prompt is not None

    moe = get_candidate("llada-moe-7b-a1b-instruct-hf")
    moe_config = DiffusionGenerationConfig().with_candidate_defaults(moe)
    assert moe_config.algorithm == "low_confidence"


def test_llada_family_helper_includes_sparse_moe_family():
    assert is_llada_family("LLaDA 8B")
    assert is_llada_family("LLaDA MoE 7B-A1B")
    assert not is_llada_family("Dream 7B")


def test_llada_mask_id_prefers_tokenizer_value_for_sparse_moe():
    class Tokenizer:
        mask_token_id = 156895

        def convert_tokens_to_ids(self, token):
            return {"[gMASK]": 156894}[token]

    class Model:
        config = object()

    assert _llada_mask_token_id(Tokenizer(), Model()) == 156895


def test_llada_mask_id_can_fall_back_to_named_tokens():
    class Tokenizer:
        mask_token_id = None
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            return {"<|mask|>": 0, "[MASK]": 156895, "[gMASK]": 156894}[token]

    class Model:
        config = object()

    assert _llada_mask_token_id(Tokenizer(), Model()) == 156895


def test_llada_revision_remask_masks_low_confidence_committed_tokens():
    x = torch.tensor([[101, 11, 12, 13, 14]])
    confidences = torch.tensor([[float("nan"), 0.90, 0.10, 0.40, 0.80]])

    _llada_apply_revision_remask(
        x,
        confidences,
        prompt_length=1,
        gen_length=4,
        mask_id=99,
        remask_fraction=0.50,
    )

    assert x.tolist() == [[101, 11, 99, 99, 14]]
    assert torch.isnan(confidences[0, 2])
    assert torch.isnan(confidences[0, 3])


def test_hf_backend_rejects_gguf_candidate_without_loading():
    try:
        HFDiffusionBackend("dream-7b-instruct-gguf-q4")
    except ValueError as exc:
        assert "hf_custom" in str(exc)
    else:
        raise AssertionError("Expected GGUF candidate to be rejected by HF backend")

    try:
        HFDiffusionBackend("llada-moe-7b-a1b-instruct-gguf-q4")
    except ValueError as exc:
        assert "hf_custom" in str(exc)
    else:
        raise AssertionError("Expected MoE GGUF candidate to be rejected by HF backend")


def test_hf_backend_accepts_local_model_path_without_loading():
    backend = HFDiffusionBackend(
        "dream-7b-instruct-hf",
        model_path="external/diffusion_models/Dream-v0-Instruct-7B",
    )
    assert backend.model_path.endswith("Dream-v0-Instruct-7B")
    assert backend.model is None
