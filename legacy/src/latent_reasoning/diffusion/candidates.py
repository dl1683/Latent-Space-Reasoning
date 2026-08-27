"""Candidate language-diffusion models for local reasoning experiments.

This registry is deliberately lightweight. Importing it must not download or
load model weights, because benchmark and unit-test code should be able to
inspect the plan on CPU-only machines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DiffusionModelCandidate:
    """Static metadata for a runnable or near-runnable diffusion LM target."""

    key: str
    model_id: str
    family: str
    backend: str
    priority: int
    min_vram_gb: float | None
    precision: str
    license: str
    generation_method: str
    default_max_new_tokens: int
    default_steps: int
    default_algorithm: str
    supports_history: bool
    local_role: str
    notes: str
    sources: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly metadata."""
        data = asdict(self)
        data["sources"] = list(self.sources)
        return data


_CANDIDATES: tuple[DiffusionModelCandidate, ...] = (
    DiffusionModelCandidate(
        key="dream-7b-instruct-hf",
        model_id="Dream-org/Dream-v0-Instruct-7B",
        family="Dream 7B",
        backend="hf_custom",
        priority=1,
        min_vram_gb=20.0,
        precision="bfloat16",
        license="apache-2.0",
        generation_method="model.diffusion_generate",
        default_max_new_tokens=128,
        default_steps=128,
        default_algorithm="entropy",
        supports_history=True,
        local_role="first GPU target for reasoning, planning, and trajectory-history probes",
        notes=(
            "Official code exposes diffusion_generate(), supports entropy remasking, "
            "and can return intermediate denoising history."
        ),
        sources=(
            "https://huggingface.co/Dream-org/Dream-v0-Instruct-7B",
            "https://github.com/DreamLM/Dream",
            "https://arxiv.org/abs/2508.15487",
        ),
    ),
    DiffusionModelCandidate(
        key="llada-8b-instruct-hf",
        model_id="GSAI-ML/LLaDA-8B-Instruct",
        family="LLaDA 8B",
        backend="hf_custom",
        priority=2,
        min_vram_gb=20.0,
        precision="bfloat16",
        license="mit",
        generation_method="repo-style masked diffusion loop",
        default_max_new_tokens=128,
        default_steps=128,
        default_algorithm="low_confidence",
        supports_history=False,
        local_role="second GPU target and architecture-diversity check",
        notes=(
            "Official repo provides a masked-diffusion generate loop; useful as a "
            "cross-check against Dream's Qwen-derived path."
        ),
        sources=(
            "https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct",
            "https://github.com/ML-GSAI/LLaDA",
            "https://arxiv.org/abs/2502.09992",
        ),
    ),
    DiffusionModelCandidate(
        key="llada-moe-7b-a1b-instruct-hf",
        model_id="inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        family="LLaDA MoE 7B-A1B",
        backend="hf_custom",
        priority=3,
        min_vram_gb=12.0,
        precision="bfloat16",
        license="apache-2.0",
        generation_method="repo-style masked diffusion loop",
        default_max_new_tokens=128,
        default_steps=128,
        default_algorithm="low_confidence",
        supports_history=False,
        local_role="cheap active-parameter LLaDA-family target after dense LLaDA",
        notes=(
            "Sparse MoE diffusion LM with 7B total parameters and roughly 1B-1.4B "
            "active parameters at inference; should be tested as the cheap LLaDA "
            "successor once custom-code loading and history instrumentation are verified."
        ),
        sources=(
            "https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
            "https://arxiv.org/abs/2509.24389",
        ),
    ),
    DiffusionModelCandidate(
        key="dream-7b-instruct-gguf-q4",
        model_id="diffuse-cpp/Dream-v0-Instruct-7B-GGUF:dream-7b-q4km.gguf",
        family="Dream 7B",
        backend="diffuse_cpp_or_llama_cpp",
        priority=4,
        min_vram_gb=None,
        precision="Q4_K_M GGUF",
        license="apache-2.0",
        generation_method="diffuse-cpp or llama.cpp server",
        default_max_new_tokens=128,
        default_steps=16,
        default_algorithm="entropy_exit",
        supports_history=False,
        local_role="fallback when BF16 Python inference is too memory-heavy",
        notes=(
            "Quantized route keeps the benchmark alive on CPU or lower-memory GPU, "
            "but gives less direct access to denoising internals."
        ),
        sources=(
            "https://huggingface.co/diffuse-cpp/Dream-v0-Instruct-7B-GGUF",
            "https://github.com/iafiscal1212/diffuse-cpp",
        ),
    ),
    DiffusionModelCandidate(
        key="llada-8b-instruct-gguf-q4",
        model_id="diffuse-cpp/LLaDA-8B-Instruct-GGUF:llada-8b-q4km.gguf",
        family="LLaDA 8B",
        backend="diffuse_cpp_or_llama_cpp",
        priority=5,
        min_vram_gb=None,
        precision="Q4_K_M GGUF",
        license="mit",
        generation_method="diffuse-cpp or llama.cpp server",
        default_max_new_tokens=128,
        default_steps=16,
        default_algorithm="entropy_exit",
        supports_history=False,
        local_role="fallback when BF16 Python inference is too memory-heavy",
        notes=(
            "Quantized route is operationally cheap, especially for smoke tests, "
            "but not the first choice for trajectory instrumentation."
        ),
        sources=(
            "https://huggingface.co/diffuse-cpp/LLaDA-8B-Instruct-GGUF",
            "https://github.com/iafiscal1212/diffuse-cpp",
        ),
    ),
    DiffusionModelCandidate(
        key="llada-moe-7b-a1b-instruct-gguf-q4",
        model_id="mradermacher/LLaDA-MoE-7B-A1B-Instruct-i1-GGUF",
        family="LLaDA MoE 7B-A1B",
        backend="diffuse_cpp_or_llama_cpp",
        priority=6,
        min_vram_gb=None,
        precision="Q4_K_M GGUF",
        license="apache-2.0",
        generation_method="llama.cpp-compatible GGUF",
        default_max_new_tokens=128,
        default_steps=16,
        default_algorithm="entropy_exit",
        supports_history=False,
        local_role="quantized cheap fallback for the LLaDA-MoE target",
        notes=(
            "Community GGUF quantization route for cheap smoke tests; useful if "
            "BF16 custom-code loading is too slow or memory-heavy, but expected "
            "to expose fewer denoising internals."
        ),
        sources=(
            "https://huggingface.co/mradermacher/LLaDA-MoE-7B-A1B-Instruct-i1-GGUF",
            "https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        ),
    ),
)


def available_candidates() -> tuple[DiffusionModelCandidate, ...]:
    """Return candidates ordered by practical priority."""
    return tuple(sorted(_CANDIDATES, key=lambda item: item.priority))


def candidate_keys() -> tuple[str, ...]:
    """Return known candidate keys."""
    return tuple(candidate.key for candidate in available_candidates())


def get_candidate(key: str) -> DiffusionModelCandidate:
    """Look up a candidate by stable key."""
    for candidate in _CANDIDATES:
        if candidate.key == key:
            return candidate
    valid = ", ".join(candidate_keys())
    raise KeyError(f"Unknown diffusion candidate {key!r}. Valid keys: {valid}")


def is_llada_family(family: str) -> bool:
    """Return whether a candidate family uses the LLaDA masked-denoise loop."""
    return family.lower().startswith("llada")


def default_gpu_candidate() -> DiffusionModelCandidate:
    """Return the current first-choice GPU target."""
    return get_candidate("dream-7b-instruct-hf")
