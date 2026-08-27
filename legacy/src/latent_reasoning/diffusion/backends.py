"""Lazy language-diffusion backends.

The classes here are import-safe: no heavyweight model code is imported until
``load()`` or ``generate()`` is called. That keeps unit tests and benchmark
manifest generation cheap while still giving the repo a real execution path.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, replace
from math import ceil, isnan
from typing import Any

from latent_reasoning.diffusion.candidates import (
    DiffusionModelCandidate,
    get_candidate,
    is_llada_family,
)
from latent_reasoning.diffusion.trajectory import summarize_history_samples


@dataclass(frozen=True)
class DiffusionGenerationConfig:
    """Generation settings shared by supported language-diffusion backends."""

    max_new_tokens: int = 128
    steps: int = 128
    temperature: float = 0.0
    top_p: float | None = None
    algorithm: str = "entropy"
    alg_temp: float = 0.0
    block_length: int = 32
    remasking: str = "low_confidence"
    output_history: bool = False
    history_sample_count: int = 5
    revision_remask_fraction: float | None = None
    revision_steps: int = 0
    system_prompt: str | None = (
        "You are completing safe benchmark tasks for research. Answer directly, "
        "concisely, and operationally. Do not refuse harmless planning, math, "
        "logic, or science questions."
    )
    initial_suffix_token_ids: tuple[int | None, ...] | None = None
    device: str | None = None
    dtype: str | None = None

    def with_candidate_defaults(self, candidate: DiffusionModelCandidate) -> DiffusionGenerationConfig:
        """Fill unset-like values from a candidate without mutating callers."""
        algorithm = self.algorithm
        if algorithm == "entropy" and is_llada_family(candidate.family):
            algorithm = candidate.default_algorithm
        return replace(
            self,
            max_new_tokens=self.max_new_tokens or candidate.default_max_new_tokens,
            steps=self.steps or candidate.default_steps,
            algorithm=algorithm,
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly config."""
        return asdict(self)


@dataclass(frozen=True)
class DiffusionGenerationResult:
    """Generated text plus minimal execution metadata."""

    text: str
    prompt: str
    candidate_key: str
    model_id: str
    config: dict[str, object]
    generated_token_ids: list[int]
    generated_token_count: int
    generated_token_confidences: list[float | None] | None = None
    model_path: str | None = None
    history_steps: int | None = None
    history_samples: list[dict[str, object]] | None = None
    trajectory_summary: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly output."""
        return asdict(self)


class HFDiffusionBackend:
    """Hugging Face custom-code backend for Dream and LLaDA."""

    def __init__(
        self,
        candidate_key: str = "dream-7b-instruct-hf",
        *,
        device: str | None = None,
        dtype: str | None = None,
        model_path: str | None = None,
    ) -> None:
        self.candidate = get_candidate(candidate_key)
        if self.candidate.backend != "hf_custom":
            raise ValueError(
                f"{candidate_key!r} uses backend {self.candidate.backend!r}; "
                "HFDiffusionBackend only supports hf_custom candidates"
            )
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self.model: Any | None = None
        self.tokenizer: Any | None = None

    def load(self) -> None:
        """Load model and tokenizer if needed."""
        if self.model is not None and self.tokenizer is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = _resolve_torch_dtype(torch, self.dtype, device)
        model_ref = self.model_path or self.candidate.model_id
        tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        _ensure_torchvision_optional_import_compat()
        _ensure_transformers_default_rope()
        _ensure_generation_config_validate_compat()
        _ensure_all_tied_weights_keys_compat()
        _ensure_tie_weights_signature_compat()
        model = AutoModel.from_pretrained(
            model_ref,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        _fill_model_config_defaults(model)
        if is_llada_family(self.candidate.family) and getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    def generate(
        self,
        prompt: str,
        config: DiffusionGenerationConfig | None = None,
    ) -> DiffusionGenerationResult:
        """Generate one completion with the selected diffusion model."""
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        config = (config or DiffusionGenerationConfig()).with_candidate_defaults(self.candidate)
        if self.candidate.family == "Dream 7B":
            text, token_ids, token_confidences, history_steps, history_samples = self._generate_dream(
                prompt,
                config,
            )
        elif is_llada_family(self.candidate.family):
            text, token_ids, token_confidences, history_steps, history_samples = self._generate_llada(
                prompt,
                config,
            )
        else:
            raise NotImplementedError(f"Unsupported diffusion family: {self.candidate.family}")

        mask_token_id = _first_not_none(
            getattr(self.tokenizer, "mask_token_id", None),
            getattr(getattr(self.model, "config", None), "mask_token_id", None),
        )
        eos_token_id = _first_not_none(
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(getattr(self.model, "config", None), "eos_token_id", None),
        )
        trajectory_summary = summarize_history_samples(
            history_samples,
            final_text=text,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            mask_token_text=_mask_token_text(self.tokenizer, mask_token_id),
        )
        return DiffusionGenerationResult(
            text=text,
            prompt=prompt,
            candidate_key=self.candidate.key,
            model_id=self.candidate.model_id,
            config=config.to_dict(),
            generated_token_ids=token_ids,
            generated_token_count=len(token_ids),
            generated_token_confidences=token_confidences,
            model_path=self.model_path,
            history_steps=history_steps,
            history_samples=history_samples,
            trajectory_summary=trajectory_summary,
        )

    def _generate_dream(
        self,
        prompt: str,
        config: DiffusionGenerationConfig,
    ) -> tuple[str, list[int], list[float | None] | None, int | None, list[dict[str, object]] | None]:
        import torch

        if config.initial_suffix_token_ids is not None:
            raise ValueError("Dream backend does not support suffix inpainting yet.")

        messages = _chat_messages(config.system_prompt, prompt)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        input_ids = inputs.input_ids.to(device=self.device)
        attention_mask = inputs.attention_mask.to(device=self.device)
        kwargs: dict[str, object] = {
            "attention_mask": attention_mask,
            "max_new_tokens": config.max_new_tokens,
            "output_history": config.output_history,
            "return_dict_in_generate": True,
            "steps": config.steps,
            "temperature": config.temperature,
            "alg": config.algorithm,
            "alg_temp": config.alg_temp,
        }
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            _fill_generation_special_token_ids(generation_config, self.tokenizer, self.model)
            kwargs["generation_config"] = generation_config

        with torch.no_grad():
            output = self.model.diffusion_generate(input_ids, **kwargs)
        generated_ids = output.sequences[0][input_ids.shape[1] :].tolist()
        decoded = self.tokenizer.decode(generated_ids)
        text = _strip_after_eos(decoded, getattr(self.tokenizer, "eos_token", None))
        history = getattr(output, "history", None)
        history_steps = len(history) if history is not None else None
        history_samples = _decode_history_samples(
            self.tokenizer,
            history,
            prompt_length=input_ids.shape[1],
            sample_count=config.history_sample_count,
        )
        return text, generated_ids, None, history_steps, history_samples

    def _generate_llada(
        self,
        prompt: str,
        config: DiffusionGenerationConfig,
    ) -> tuple[str, list[int], list[float | None] | None, int | None, list[dict[str, object]] | None]:
        import torch

        messages = _chat_messages(config.system_prompt, prompt)
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        encoded = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.no_grad():
            output, history, token_confidences = _llada_generate(
                self.model,
                input_ids,
                attention_mask=attention_mask,
                steps=config.steps,
                gen_length=config.max_new_tokens,
                block_length=config.block_length,
                temperature=config.temperature,
                remasking=config.remasking,
                mask_id=_llada_mask_token_id(self.tokenizer, self.model),
                output_history=config.output_history,
                initial_suffix_token_ids=config.initial_suffix_token_ids,
                revision_remask_fraction=config.revision_remask_fraction,
                revision_steps=config.revision_steps,
            )
        generated_ids = output[0, input_ids.shape[1] :].tolist()
        generated_confidences = _slice_confidences(
            token_confidences,
            prompt_length=input_ids.shape[1],
            gen_length=config.max_new_tokens,
        )
        decoded = self.tokenizer.batch_decode(
            output[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )[0]
        history_steps = len(history) if history is not None else None
        history_samples = _decode_history_samples(
            self.tokenizer,
            history,
            prompt_length=input_ids.shape[1],
            sample_count=config.history_sample_count,
        )
        return decoded, generated_ids, generated_confidences, history_steps, history_samples


def _resolve_torch_dtype(torch_module: Any, dtype: str | None, device: str) -> Any:
    if dtype is None:
        return torch_module.bfloat16 if device == "cuda" else torch_module.float32
    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch_module.float16
    if normalized in {"fp32", "float32"}:
        return torch_module.float32
    raise ValueError(f"Unsupported dtype {dtype!r}")


def _ensure_transformers_default_rope() -> None:
    """Restore the legacy Dream remote-code RoPE key on newer Transformers."""
    try:
        import torch
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except Exception:
        return
    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _compute_default_rope_parameters(
        config: Any | None = None,
        device: Any | None = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple[Any, float]:
        del seq_len, layer_type
        if config is None:
            raise ValueError("Dream default RoPE compatibility requires a model config")
        base = getattr(config, "rope_theta", getattr(config, "default_theta", 10000.0))
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = getattr(config, "hidden_size") // getattr(config, "num_attention_heads")
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _ensure_torchvision_optional_import_compat() -> None:
    """Treat broken optional torchvision installs as unavailable for text models."""
    try:
        import torchvision  # noqa: F401

        return
    except Exception:
        pass
    for module_name in list(sys.modules):
        if module_name == "torchvision" or module_name.startswith("torchvision."):
            sys.modules.pop(module_name, None)
    sys.modules.pop("transformers.image_utils", None)
    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as import_utils
    except Exception:
        return

    def _torchvision_unavailable() -> bool:
        return False

    import_utils.is_torchvision_available = _torchvision_unavailable
    transformers_utils.is_torchvision_available = _torchvision_unavailable


def _ensure_generation_config_validate_compat() -> None:
    """Let older remote GenerationConfig subclasses survive newer update calls."""
    try:
        from transformers.generation.configuration_utils import GenerationConfig
    except Exception:
        return
    if getattr(GenerationConfig.update, "_latent_reasoning_validate_compat", False):
        return

    original_update = GenerationConfig.update

    def _compatible_update(self: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return original_update(self, **kwargs)
        except TypeError as exc:
            if "user_set_attributes" not in str(exc):
                raise
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)
        self.validate()
        return {key: value for key, value in kwargs.items() if key not in to_remove}

    _compatible_update._latent_reasoning_validate_compat = True  # type: ignore[attr-defined]
    _compatible_update._latent_reasoning_original_update = original_update  # type: ignore[attr-defined]
    GenerationConfig.update = _compatible_update


def _ensure_all_tied_weights_keys_compat() -> None:
    """Expose old remote-code `_tied_weights_keys` under the newer HF name."""
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return
    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        return

    def _get_all_tied_weights_keys(self: Any) -> dict[str, str]:
        explicit_keys = getattr(self, "_latent_reasoning_all_tied_weights_keys", None)
        tied_keys = explicit_keys if explicit_keys is not None else getattr(self, "_tied_weights_keys", None)
        if tied_keys is None:
            return {}
        if isinstance(tied_keys, dict):
            return {str(key): str(value) for key, value in tied_keys.items()}
        return {str(key): str(key) for key in tied_keys}

    def _set_all_tied_weights_keys(self: Any, value: Any) -> None:
        self._latent_reasoning_all_tied_weights_keys = value

    PreTrainedModel.all_tied_weights_keys = property(
        _get_all_tied_weights_keys,
        _set_all_tied_weights_keys,
    )


def _ensure_tie_weights_signature_compat() -> None:
    """Let old remote-code tie_weights methods ignore newer HF kwargs."""
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return
    original_finalize = PreTrainedModel._finalize_model_loading
    if getattr(original_finalize, "_latent_reasoning_tie_weights_compat", False):
        return

    def _compatible_finalize(model: Any, load_config: Any, loading_info: Any) -> Any:
        _wrap_legacy_tie_weights(model)
        return original_finalize(model, load_config, loading_info)

    _compatible_finalize._latent_reasoning_tie_weights_compat = True  # type: ignore[attr-defined]
    _compatible_finalize._latent_reasoning_original_finalize = original_finalize  # type: ignore[attr-defined]
    PreTrainedModel._finalize_model_loading = staticmethod(_compatible_finalize)


def _wrap_legacy_tie_weights(model: Any) -> None:
    tie_weights = getattr(model, "tie_weights", None)
    if not callable(tie_weights) or getattr(tie_weights, "_latent_reasoning_tie_weights_compat", False):
        return

    def _compatible_tie_weights(*args: Any, **kwargs: Any) -> Any:
        try:
            return tie_weights(*args, **kwargs)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            if not any(key in str(exc) for key in ("missing_keys", "recompute_mapping")):
                raise
            if args:
                return tie_weights(*args)
            return tie_weights()

    _compatible_tie_weights._latent_reasoning_tie_weights_compat = True  # type: ignore[attr-defined]
    model.tie_weights = _compatible_tie_weights


def _fill_model_config_defaults(model: Any) -> None:
    """Backfill conservative config defaults required by older remote code."""
    config = getattr(model, "config", None)
    if config is None:
        return
    if not hasattr(config, "use_cache"):
        setattr(config, "use_cache", False)


def _fill_generation_special_token_ids(generation_config: Any, tokenizer: Any, model: Any) -> None:
    """Fill missing remote generation token ids from tokenizer/model config."""
    model_config = getattr(model, "config", None)
    for attr in ("mask_token_id", "pad_token_id", "bos_token_id", "eos_token_id"):
        if getattr(generation_config, attr, None) is not None:
            continue
        value = _first_not_none(
            getattr(tokenizer, attr, None),
            getattr(model_config, attr, None),
        )
        if value is not None:
            setattr(generation_config, attr, value)


def _chat_messages(system_prompt: str | None, prompt: str) -> list[dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _strip_after_eos(text: str, eos_token: str | None) -> str:
    if eos_token and eos_token in text:
        return text.split(eos_token, 1)[0]
    return text


def _first_not_none(*values: object) -> object | None:
    for value in values:
        if value is not None:
            return value
    return None


def _llada_mask_token_id(tokenizer: Any, model: Any) -> int:
    """Resolve the mask id for dense LLaDA and sparse LLaDA-MoE variants."""
    for value in (
        getattr(tokenizer, "mask_token_id", None),
        getattr(getattr(model, "config", None), "mask_token_id", None),
    ):
        if isinstance(value, int):
            return value

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if callable(convert):
        for token in ("<|mask|>", "[MASK]", "[gMASK]"):
            try:
                token_id = convert(token)
            except Exception:
                continue
            if isinstance(token_id, int) and token_id >= 0 and token_id != unk_token_id:
                return token_id

    return 126336


def _mask_token_text(tokenizer: Any, mask_token_id: object | None) -> str:
    mask_token = getattr(tokenizer, "mask_token", None)
    if mask_token:
        return str(mask_token)
    if isinstance(mask_token_id, int):
        try:
            return str(tokenizer.decode([mask_token_id]))
        except Exception:
            return "<|mask|>"
    return "<|mask|>"


def _decode_history_samples(
    tokenizer: Any,
    history: Any | None,
    *,
    prompt_length: int,
    sample_count: int,
) -> list[dict[str, object]] | None:
    if history is None:
        return None
    if sample_count <= 0:
        return []
    total = len(history)
    if total == 0:
        return []

    eos_token = getattr(tokenizer, "eos_token", None)
    if total <= sample_count:
        indices = list(range(total))
    else:
        indices = sorted(
            {
                round(position * (total - 1) / (sample_count - 1))
                for position in range(sample_count)
            }
        )

    samples: list[dict[str, object]] = []
    for index in indices:
        generated_ids = history[index][0, prompt_length:].tolist()
        raw_text = tokenizer.decode(generated_ids)
        samples.append(
            {
                "step": index + 1,
                "generated_token_ids": generated_ids,
                "text": _strip_after_eos(raw_text, eos_token),
            }
        )
    return samples


def _llada_add_gumbel_noise(logits: Any, temperature: float) -> Any:
    if temperature == 0:
        return logits
    import torch

    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _llada_transfer_counts(mask_index: Any, steps: int) -> Any:
    import torch

    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    counts = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for row in range(mask_num.size(0)):
        counts[row, : remainder[row]] += 1
    return counts


def _llada_generate(
    model: Any,
    prompt: Any,
    *,
    attention_mask: Any | None,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    remasking: str,
    mask_id: int = 126336,
    output_history: bool = False,
    initial_suffix_token_ids: tuple[int | None, ...] | list[int | None] | None = None,
    revision_remask_fraction: float | None = None,
    revision_steps: int = 0,
) -> tuple[Any, list[Any] | None, Any]:
    import torch

    if gen_length <= 0:
        raise ValueError("gen_length must be positive")
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if revision_steps < 0:
        raise ValueError("revision_steps must be non-negative")
    if revision_remask_fraction is not None and not 0.0 < revision_remask_fraction <= 1.0:
        raise ValueError("revision_remask_fraction must be greater than 0 and at most 1")
    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length for LLaDA")

    device = _model_device(model)
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    _apply_initial_suffix_tokens(
        x,
        prompt_length=prompt.shape[1],
        gen_length=gen_length,
        initial_suffix_token_ids=initial_suffix_token_ids,
    )
    if attention_mask is not None:
        suffix_mask = torch.ones(
            (prompt.shape[0], gen_length),
            dtype=attention_mask.dtype,
            device=device,
        )
        attention_mask = torch.cat([attention_mask, suffix_mask], dim=-1)

    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError("steps must be divisible by gen_length / block_length for LLaDA")
    block_steps = steps // num_blocks
    history = [] if output_history else None
    token_confidences = torch.full(x.shape, torch.nan, dtype=torch.float32, device=device)
    _apply_seed_confidences(
        token_confidences,
        prompt_length=prompt.shape[1],
        initial_suffix_token_ids=initial_suffix_token_ids,
    )

    for block_idx in range(num_blocks):
        start = prompt.shape[1] + block_idx * block_length
        end = prompt.shape[1] + (block_idx + 1) * block_length
        _llada_denoise_masked_span(
            model,
            x,
            attention_mask=attention_mask,
            steps=block_steps,
            start=start,
            end=end,
            temperature=temperature,
            remasking=remasking,
            mask_id=mask_id,
            token_confidences=token_confidences,
            history=history,
        )
    if revision_steps and revision_remask_fraction is not None:
        _llada_apply_revision_remask(
            x,
            token_confidences,
            prompt_length=prompt.shape[1],
            gen_length=gen_length,
            mask_id=mask_id,
            remask_fraction=revision_remask_fraction,
        )
        if history is not None:
            history.append(x.clone())
        _llada_denoise_masked_span(
            model,
            x,
            attention_mask=attention_mask,
            steps=revision_steps,
            start=prompt.shape[1],
            end=prompt.shape[1] + gen_length,
            temperature=temperature,
            remasking=remasking,
            mask_id=mask_id,
            token_confidences=token_confidences,
            history=history,
        )
    return x, history, token_confidences


def _llada_denoise_masked_span(
    model: Any,
    x: Any,
    *,
    attention_mask: Any | None,
    steps: int,
    start: int,
    end: int,
    temperature: float,
    remasking: str,
    mask_id: int,
    token_confidences: Any,
    history: list[Any] | None,
) -> None:
    import torch
    import torch.nn.functional as functional

    if steps <= 0:
        return
    block_mask_index = x[:, start:end] == mask_id
    transfer_counts = _llada_transfer_counts(block_mask_index, steps)

    for step_idx in range(steps):
        mask_index = x == mask_id
        logits = model(x, attention_mask=attention_mask).logits
        logits_with_noise = _llada_add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        probs = functional.softmax(logits, dim=-1)
        token_probs = torch.squeeze(torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)), -1)

        if remasking == "low_confidence":
            x0_p = token_probs
        elif remasking == "random":
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise NotImplementedError(remasking)

        x0_p[:, :start] = -torch.inf
        x0_p[:, end:] = -torch.inf
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -torch.inf)
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        for row in range(confidence.shape[0]):
            k = int(transfer_counts[row, step_idx].item())
            if k > 0:
                _, select_index = torch.topk(confidence[row], k=k)
                transfer_index[row, select_index] = True
        x[transfer_index] = x0[transfer_index]
        token_confidences[transfer_index] = token_probs[transfer_index].to(torch.float32)
        if history is not None:
            history.append(x.clone())


def _llada_apply_revision_remask(
    x: Any,
    token_confidences: Any,
    *,
    prompt_length: int,
    gen_length: int,
    mask_id: int,
    remask_fraction: float,
) -> None:
    import torch

    if remask_fraction <= 0.0 or remask_fraction > 1.0:
        raise ValueError("remask_fraction must be greater than 0 and at most 1")
    start = prompt_length
    end = prompt_length + gen_length
    for row in range(x.shape[0]):
        suffix = x[row, start:end]
        visible_positions = torch.nonzero(suffix != mask_id, as_tuple=False).flatten()
        if visible_positions.numel() == 0:
            continue
        remask_count = max(1, min(int(visible_positions.numel()), ceil(int(visible_positions.numel()) * remask_fraction)))
        confidences = token_confidences[row, start:end][visible_positions]
        confidences = torch.where(torch.isnan(confidences), torch.ones_like(confidences), confidences)
        selected_offsets = visible_positions[torch.argsort(confidences)[:remask_count]]
        selected_indices = start + selected_offsets
        x[row, selected_indices] = mask_id
        token_confidences[row, selected_indices] = torch.nan


def _apply_initial_suffix_tokens(
    x: Any,
    *,
    prompt_length: int,
    gen_length: int,
    initial_suffix_token_ids: tuple[int | None, ...] | list[int | None] | None,
) -> None:
    """Seed the generated suffix before denoising.

    Non-``None`` suffix ids are treated as already denoised and remain fixed
    because the LLaDA loop only updates mask-token positions. ``None`` leaves a
    position masked.
    """
    if initial_suffix_token_ids is None:
        return
    if len(initial_suffix_token_ids) > gen_length:
        raise ValueError("initial_suffix_token_ids cannot exceed gen_length")
    for offset, token_id in enumerate(initial_suffix_token_ids):
        if token_id is None:
            continue
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError("initial_suffix_token_ids must contain int or None values")
        x[:, prompt_length + offset] = token_id


def _apply_seed_confidences(
    token_confidences: Any,
    *,
    prompt_length: int,
    initial_suffix_token_ids: tuple[int | None, ...] | list[int | None] | None,
) -> None:
    if initial_suffix_token_ids is None:
        return
    for offset, token_id in enumerate(initial_suffix_token_ids):
        if token_id is not None:
            token_confidences[:, prompt_length + offset] = 1.0


def _slice_confidences(
    token_confidences: Any | None,
    *,
    prompt_length: int,
    gen_length: int,
) -> list[float | None] | None:
    if token_confidences is None:
        return None
    suffix = token_confidences[0, prompt_length : prompt_length + gen_length].detach().cpu()
    values: list[float | None] = []
    for value in suffix.tolist():
        if isnan(value):
            values.append(None)
        else:
            values.append(float(value))
    return values


def _model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device
