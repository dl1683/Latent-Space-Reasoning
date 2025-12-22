from __future__ import annotations

from typing import Any

import torch


def get_default_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


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

    if normalized != "4bit":
        return {}, False, f"unsupported quantization mode: {mode}"
    if device.type != "cuda":
        return {}, False, "quantization requires CUDA"

    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig
    except Exception:
        return {}, False, "bitsandbytes not available"

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    return {"quantization_config": quant_config, "device_map": "auto"}, True, None
