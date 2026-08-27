"""
Device management utilities for GPU acceleration and compute optimization.

This module provides intelligent device selection and management for the
latent space reasoning system. It handles GPU detection, memory management,
and fallback strategies to ensure optimal performance across different
hardware configurations.

Key Features:
- Automatic GPU detection with fallback to CPU
- Memory usage monitoring and optimization
- Device compatibility checking
- Multi-GPU support for distributed processing
- Robust error handling for hardware issues

The system automatically selects the best available device but allows
manual override for specific use cases or hardware configurations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DeviceInfo:
    """
    Information about a compute device and its capabilities.

    This class encapsulates all relevant information about a compute device,
    including hardware specifications, memory availability, and compatibility
    details. Used for device selection and resource planning.

    Attributes:
        device: PyTorch device object for tensor operations
        name: Human-readable device name (e.g., "NVIDIA RTX 4090")
        is_cuda: Whether this is a CUDA-capable GPU device
        total_memory: Total device memory in GB (None for CPU)
        cuda_capability: CUDA compute capability version (major, minor)

    Example:
        >>> info = get_device_info("cuda:0")
        >>> print(f"Using {info.name} with {info.total_memory:.1f}GB memory")
        >>> if info.cuda_capability and info.cuda_capability >= (8, 0):
        ...     print("Supports modern CUDA features")
    """
    device: torch.device
    name: str
    is_cuda: bool
    total_memory: float | None = None  # GB
    cuda_capability: tuple[int, int] | None = None


def _cuda_is_usable() -> bool:
    """Check if CUDA is actually usable (not just available)."""
    if not torch.cuda.is_available():
        return False
    try:
        # Try to actually use CUDA - this will fail on incompatible GPUs
        test_tensor = torch.zeros(1, device="cuda")
        del test_tensor
        return True
    except Exception:
        return False


def get_device(preference: str = "auto") -> torch.device:
    """
    Get the best available device.

    Args:
        preference: One of "auto", "cuda", "cpu", or specific like "cuda:0"

    Returns:
        torch.device for the selected device
    """
    if preference == "auto":
        if _cuda_is_usable():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    elif preference.startswith("cuda"):
        if _cuda_is_usable():
            return torch.device(preference)
        else:
            print(f"Warning: CUDA not usable (GPU may be incompatible), falling back to CPU")
            return torch.device("cpu")
    else:
        return torch.device(preference)


def get_device_info(device: torch.device | str | None = None) -> DeviceInfo:
    """Get detailed information about a device."""
    if device is None:
        device = get_device("auto")
    elif isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        return DeviceInfo(
            device=device,
            name=props.name,
            is_cuda=True,
            total_memory=props.total_memory / 1e9,
            cuda_capability=(props.major, props.minor),
        )
    else:
        return DeviceInfo(
            device=device,
            name="CPU",
            is_cuda=False,
        )


def clear_cuda_cache() -> None:
    """Clear CUDA memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_usage() -> dict[str, float]:
    """Get current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "total": 0}

    return {
        "allocated": torch.cuda.memory_allocated() / 1e9,
        "reserved": torch.cuda.memory_reserved() / 1e9,
        "total": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }


def ensure_tensor_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Ensure a tensor is on the specified device."""
    if tensor.device != device:
        return tensor.to(device)
    return tensor
