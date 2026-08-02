from __future__ import annotations

from typing import Any

import torch


def get_default_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def resolve_load_dtype(
    device: torch.device,
    config: Any = None,
    override: str | None = None,
) -> torch.dtype:
    """Choose the dtype for an *unquantized* load.

    Precedence:

    1. an explicit ``override`` ("float16", "bfloat16", "float32");
    2. the dtype the checkpoint declares for itself;
    3. the historical device default.

    Step 2 matters for correctness, not just memory. Gemma and Qwen3 checkpoints
    are trained and released in bfloat16; forcing them into float16 narrows the
    exponent range by 5 bits, and Gemma in particular overflows to inf in the
    attention logits and the final ``logit_softcapping`` when loaded that way.
    Honouring the declared dtype is also what ``transformers>=5`` does by
    default, so this keeps behaviour stable across versions rather than relying
    on the library's default.
    """
    if override:
        named = getattr(torch, override, None)
        if not isinstance(named, torch.dtype):
            raise ValueError(f"unsupported dtype override: {override}")
        return named

    if config is not None:
        # transformers>=5 renamed ``torch_dtype`` to ``dtype`` on configs.
        for attr in ("dtype", "torch_dtype"):
            declared = getattr(config, attr, None)
            if isinstance(declared, str):
                declared = getattr(torch, declared, None)
            if isinstance(declared, torch.dtype):
                if declared == torch.bfloat16 and device.type == "cuda" and (
                    not torch.cuda.is_bf16_supported()
                ):
                    return torch.float16
                return declared

    return get_default_dtype(device)


def get_quantization_kwargs(
    mode: str,
    device: torch.device,
) -> tuple[dict[str, Any], bool, str | None]:
    if not mode:
        return {}, False, None

    normalized = mode.lower()
    if normalized == "none":
        return {}, False, None
    if normalized == "auto":
        normalized = "4bit"

    if normalized not in ("4bit", "8bit"):
        return {}, False, f"unsupported quantization mode: {mode}"
    if device.type != "cuda":
        return {}, False, "quantization requires CUDA"

    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig
    except Exception:
        return {}, False, "bitsandbytes not available"

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if normalized == "8bit":
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    return {"quantization_config": quant_config, "device_map": "auto"}, True, None
