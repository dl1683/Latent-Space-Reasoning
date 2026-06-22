"""CPU-safe tests for language-diffusion experiment scaffolding."""

import torch

from latent_reasoning.diffusion import DiffusionGenerationConfig, HFDiffusionBackend
from latent_reasoning.diffusion.backends import (
    _ensure_all_tied_weights_keys_compat,
    _ensure_generation_config_validate_compat,
    _ensure_transformers_default_rope,
    _fill_generation_special_token_ids,
    _llada_apply_revision_remask,
    _llada_mask_token_id,
    _wrap_legacy_tie_weights,
)
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


def test_default_rope_compatibility_patch_supports_dream_remote_code():
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    original = ROPE_INIT_FUNCTIONS.pop("default", None)
    try:
        _ensure_transformers_default_rope()

        class Config:
            rope_theta = 10000.0
            hidden_size = 64
            num_attention_heads = 4

        inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS["default"](Config(), device="cpu")

        assert inv_freq.shape == (8,)
        assert attention_scaling == 1.0
    finally:
        ROPE_INIT_FUNCTIONS.pop("default", None)
        if original is not None:
            ROPE_INIT_FUNCTIONS["default"] = original


def test_generation_config_update_accepts_legacy_validate_signature():
    from transformers.generation.configuration_utils import GenerationConfig

    _ensure_generation_config_validate_compat()

    class LegacyGenerationConfig(GenerationConfig):
        def validate(self):
            self.validated = True

    config = object.__new__(LegacyGenerationConfig)
    config.flag = False
    config.validated = False
    unused = config.update(flag=True, unknown_value=3)

    assert config.flag is True
    assert config.validated is True
    assert unused == {"unknown_value": 3}


def test_generation_special_token_backfill_uses_tokenizer_and_model_config():
    class GenerationConfig:
        mask_token_id = None
        pad_token_id = None
        bos_token_id = None
        eos_token_id = 9

    class Tokenizer:
        mask_token_id = 7
        pad_token_id = None
        bos_token_id = 5
        eos_token_id = 8

    class ModelConfig:
        pad_token_id = 6

    class Model:
        config = ModelConfig()

    generation_config = GenerationConfig()

    _fill_generation_special_token_ids(generation_config, Tokenizer(), Model())

    assert generation_config.mask_token_id == 7
    assert generation_config.pad_token_id == 6
    assert generation_config.bos_token_id == 5
    assert generation_config.eos_token_id == 9


def test_all_tied_weights_keys_compatibility_property_exposes_legacy_name():
    from transformers.modeling_utils import PreTrainedModel

    original = getattr(PreTrainedModel, "all_tied_weights_keys", None)
    if original is not None:
        delattr(PreTrainedModel, "all_tied_weights_keys")
    try:
        _ensure_all_tied_weights_keys_compat()

        class Model(PreTrainedModel):
            config_class = None
            _tied_weights_keys = ["lm_head.weight"]

            def __init__(self):
                torch.nn.Module.__init__(self)

        assert Model().all_tied_weights_keys == {"lm_head.weight": "lm_head.weight"}
        model = Model()
        model.all_tied_weights_keys = ["custom.weight"]
        assert model.all_tied_weights_keys == {"custom.weight": "custom.weight"}
    finally:
        if original is not None:
            PreTrainedModel.all_tied_weights_keys = original
        elif hasattr(PreTrainedModel, "all_tied_weights_keys"):
            delattr(PreTrainedModel, "all_tied_weights_keys")


def test_legacy_tie_weights_wrapper_accepts_new_finalize_kwargs():
    class Model:
        def __init__(self):
            self.calls = 0

        def tie_weights(self):
            self.calls += 1
            return "tied"

    model = Model()

    _wrap_legacy_tie_weights(model)

    assert model.tie_weights(missing_keys=["lm_head.weight"], recompute_mapping=False) == "tied"
    assert model.calls == 1


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
