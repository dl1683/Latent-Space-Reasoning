"""Distribution-geometry diagnostics for steering and soft prompts.

These utilities measure perturbations in the coordinate system that matters
for generation: the token probability distribution induced by logits.  They
are intentionally model-agnostic and CPU-testable, so experiment scripts can
audit whether a perturbation is clean steering, broad distribution drift, or
mostly harmless noise.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as functional
from torch import Tensor


def _as_batch(logits: Tensor) -> Tensor:
    if logits.dim() == 1:
        return logits.unsqueeze(0)
    if logits.dim() != 2:
        raise ValueError(
            f"Expected logits with shape (vocab,) or (batch, vocab), got {tuple(logits.shape)}"
        )
    return logits


def _normalize_distribution(probs: Tensor, eps: float) -> Tensor:
    probs = probs.clamp_min(eps)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


def probabilities_from_logits(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """Convert logits to probabilities with a positive temperature."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batched = _as_batch(logits).float()
    return functional.softmax(batched / temperature, dim=-1)


def entropy_from_probs(probs: Tensor, eps: float = 1e-10) -> Tensor:
    """Shannon entropy for a batch of probability distributions."""
    p = _normalize_distribution(_as_batch(probs).float(), eps)
    return -(p * p.clamp_min(eps).log()).sum(dim=-1)


def entropy_from_logits(logits: Tensor, eps: float = 1e-10) -> Tensor:
    """Shannon entropy of the softmax distribution induced by logits."""
    return entropy_from_probs(probabilities_from_logits(logits), eps=eps)


def _reference_topk_with_other(
    reference_probs: Tensor,
    candidate_probs: Tensor,
    topk: int | None,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Project distributions onto reference top-k tokens plus an ``other`` bucket."""
    reference_probs = _as_batch(reference_probs).float()
    candidate_probs = _as_batch(candidate_probs).float()
    if reference_probs.shape != candidate_probs.shape:
        raise ValueError(
            "reference_probs and candidate_probs must have the same shape, got "
            f"{tuple(reference_probs.shape)} and {tuple(candidate_probs.shape)}"
        )

    vocab = reference_probs.shape[-1]
    if topk is None or topk >= vocab:
        return (
            _normalize_distribution(reference_probs, eps),
            _normalize_distribution(candidate_probs, eps),
        )
    if topk <= 0:
        raise ValueError("topk must be positive when provided")

    idx = torch.topk(reference_probs, k=topk, dim=-1).indices
    ref_top = torch.gather(reference_probs, dim=-1, index=idx)
    cand_top = torch.gather(candidate_probs, dim=-1, index=idx)
    ref_other = (1.0 - ref_top.sum(dim=-1, keepdim=True)).clamp_min(0.0)
    cand_other = (1.0 - cand_top.sum(dim=-1, keepdim=True)).clamp_min(0.0)
    return (
        _normalize_distribution(torch.cat([ref_top, ref_other], dim=-1), eps),
        _normalize_distribution(torch.cat([cand_top, cand_other], dim=-1), eps),
    )


def kl_divergence(
    reference_probs: Tensor,
    candidate_probs: Tensor,
    *,
    topk: int | None = None,
    eps: float = 1e-10,
) -> Tensor:
    """Compute ``KL(reference || candidate)``.

    If ``topk`` is set, distributions are projected onto reference top-k tokens
    plus an ``other`` bucket.  That keeps the metric cheap while preserving
    total probability mass.
    """
    ref, cand = _reference_topk_with_other(reference_probs, candidate_probs, topk, eps)
    return (ref * (ref.clamp_min(eps).log() - cand.clamp_min(eps).log())).sum(dim=-1)


def js_divergence(
    reference_probs: Tensor,
    candidate_probs: Tensor,
    *,
    topk: int | None = None,
    eps: float = 1e-10,
) -> Tensor:
    """Compute Jensen-Shannon divergence between two distributions."""
    ref, cand = _reference_topk_with_other(reference_probs, candidate_probs, topk, eps)
    mid = 0.5 * (ref + cand)
    return 0.5 * kl_divergence(ref, mid, eps=eps) + 0.5 * kl_divergence(cand, mid, eps=eps)


def topk_overlap(reference_logits: Tensor, candidate_logits: Tensor, k: int = 20) -> Tensor:
    """Fractional overlap between reference and candidate top-k token sets."""
    if k <= 0:
        raise ValueError("k must be positive")
    ref = _as_batch(reference_logits)
    cand = _as_batch(candidate_logits)
    if ref.shape != cand.shape:
        raise ValueError(
            f"reference_logits and candidate_logits must match, got {ref.shape} and {cand.shape}"
        )

    k_eff = min(k, ref.shape[-1])
    ref_idx = torch.topk(ref, k=k_eff, dim=-1).indices
    cand_idx = torch.topk(cand, k=k_eff, dim=-1).indices

    overlaps = []
    for ref_row, cand_row in zip(ref_idx, cand_idx, strict=False):
        ref_set = set(ref_row.detach().cpu().tolist())
        cand_set = set(cand_row.detach().cpu().tolist())
        overlaps.append(len(ref_set & cand_set) / k_eff)
    return torch.tensor(overlaps, device=ref.device, dtype=torch.float32)


def weighted_rank_drift(reference_logits: Tensor, candidate_logits: Tensor, k: int = 50) -> Tensor:
    """Weighted reciprocal-rank drift of reference top-k tokens.

    A value near zero means the important reference tokens kept similar ranks.
    Larger values mean high-probability reference tokens moved substantially.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    ref = _as_batch(reference_logits).float()
    cand = _as_batch(candidate_logits).float()
    if ref.shape != cand.shape:
        raise ValueError(
            f"reference_logits and candidate_logits must match, got {ref.shape} and {cand.shape}"
        )

    vocab = ref.shape[-1]
    k_eff = min(k, vocab)
    ref_probs = probabilities_from_logits(ref)
    ref_top_probs, ref_top_idx = torch.topk(ref_probs, k=k_eff, dim=-1)
    ref_weights = ref_top_probs / ref_top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-10)

    cand_order = torch.argsort(cand, dim=-1, descending=True)
    cand_ranks = torch.empty_like(cand_order, dtype=torch.float32)
    rank_values = torch.arange(1, vocab + 1, device=cand.device, dtype=torch.float32)
    cand_ranks.scatter_(dim=-1, index=cand_order, src=rank_values.expand_as(cand_ranks))

    candidate_top_ranks = torch.gather(cand_ranks, dim=-1, index=ref_top_idx)
    reference_ranks = torch.arange(1, k_eff + 1, device=ref.device, dtype=torch.float32)
    reference_ranks = reference_ranks.unsqueeze(0).expand_as(candidate_top_ranks)

    reciprocal_delta = (1.0 / candidate_top_ranks - 1.0 / reference_ranks).abs()
    return (ref_weights * reciprocal_delta).sum(dim=-1)


@dataclass(frozen=True)
class LogitGeometryMetrics:
    """Summary of how one logit distribution moved relative to another."""

    forward_kl: Tensor
    reverse_kl: Tensor
    js: Tensor
    reference_entropy: Tensor
    candidate_entropy: Tensor
    entropy_delta: Tensor
    top1_changed: Tensor
    topk_overlap: Tensor
    weighted_rank_drift: Tensor

    def mean_dict(self) -> dict[str, float]:
        """Return scalar means for logging JSON-friendly experiment summaries."""
        out: dict[str, float] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            out[field.name] = float(value.float().mean().detach().cpu().item())
        return out


def compare_logit_geometry(
    reference_logits: Tensor,
    candidate_logits: Tensor,
    *,
    topk: int = 50,
    eps: float = 1e-10,
) -> LogitGeometryMetrics:
    """Compare two batches of logits using distribution-space diagnostics."""
    ref_logits = _as_batch(reference_logits).float()
    cand_logits = _as_batch(candidate_logits).float()
    if ref_logits.shape != cand_logits.shape:
        raise ValueError(
            f"reference_logits and candidate_logits must match, got {ref_logits.shape} and "
            f"{cand_logits.shape}"
        )

    ref_probs = probabilities_from_logits(ref_logits)
    cand_probs = probabilities_from_logits(cand_logits)
    ref_entropy = entropy_from_probs(ref_probs, eps=eps)
    cand_entropy = entropy_from_probs(cand_probs, eps=eps)

    return LogitGeometryMetrics(
        forward_kl=kl_divergence(ref_probs, cand_probs, topk=topk, eps=eps),
        reverse_kl=kl_divergence(cand_probs, ref_probs, topk=topk, eps=eps),
        js=js_divergence(ref_probs, cand_probs, topk=topk, eps=eps),
        reference_entropy=ref_entropy,
        candidate_entropy=cand_entropy,
        entropy_delta=cand_entropy - ref_entropy,
        top1_changed=(ref_logits.argmax(dim=-1) != cand_logits.argmax(dim=-1)).float(),
        topk_overlap=topk_overlap(ref_logits, cand_logits, k=topk),
        weighted_rank_drift=weighted_rank_drift(ref_logits, cand_logits, k=topk),
    )


@dataclass(frozen=True)
class CounterfactualMassMetrics:
    """Distribution movement over explicit counterfactual token pairs."""

    reference_pair_mass: Tensor
    candidate_pair_mass: Tensor
    pair_mass_delta: Tensor
    pair_mass_retention: Tensor
    neutral_mass_delta: Tensor
    pair_distribution_kl: Tensor

    def mean_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            out[field.name] = float(value.float().mean().detach().cpu().item())
        return out


def counterfactual_mass_metrics(
    reference_logits: Tensor,
    candidate_logits: Tensor,
    pairs: Tensor | list[tuple[int, int]],
    *,
    eps: float = 1e-10,
) -> CounterfactualMassMetrics:
    """Measure preservation of explicit counterfactual-pair probability mass.

    ``pairs`` should contain token-index pairs such as ``(father, mother)`` or
    ``(maintain, maintains)``.  This mirrors the paper's off-target diagnostic:
    a clean targeted intervention should move mass within relevant pairs while
    preserving total pair mass and neutral mass.
    """
    ref_logits = _as_batch(reference_logits).float()
    cand_logits = _as_batch(candidate_logits).float()
    if ref_logits.shape != cand_logits.shape:
        raise ValueError(
            f"reference_logits and candidate_logits must match, got {ref_logits.shape} and "
            f"{cand_logits.shape}"
        )

    if not torch.is_tensor(pairs):
        pairs = torch.tensor(pairs, dtype=torch.long, device=ref_logits.device)
    else:
        pairs = pairs.to(device=ref_logits.device, dtype=torch.long)
    if pairs.dim() != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (n_pairs, 2)")
    if pairs.numel() == 0:
        raise ValueError("pairs must contain at least one pair")
    if pairs.min().item() < 0 or pairs.max().item() >= ref_logits.shape[-1]:
        raise ValueError("pair indices must be within vocabulary range")

    ref_probs = probabilities_from_logits(ref_logits)
    cand_probs = probabilities_from_logits(cand_logits)

    flat_pair_indices = pairs.reshape(-1)
    ref_pair_token_probs = ref_probs[:, flat_pair_indices].view(ref_probs.shape[0], -1, 2)
    cand_pair_token_probs = cand_probs[:, flat_pair_indices].view(cand_probs.shape[0], -1, 2)

    ref_pair_masses = ref_pair_token_probs.sum(dim=-1)
    cand_pair_masses = cand_pair_token_probs.sum(dim=-1)
    ref_pair_mass = ref_pair_masses.sum(dim=-1)
    cand_pair_mass = cand_pair_masses.sum(dim=-1)

    ref_pair_distribution = _normalize_distribution(ref_pair_masses, eps)
    cand_pair_distribution = _normalize_distribution(cand_pair_masses, eps)

    return CounterfactualMassMetrics(
        reference_pair_mass=ref_pair_mass,
        candidate_pair_mass=cand_pair_mass,
        pair_mass_delta=cand_pair_mass - ref_pair_mass,
        pair_mass_retention=cand_pair_mass / ref_pair_mass.clamp_min(eps),
        neutral_mass_delta=(1.0 - cand_pair_mass) - (1.0 - ref_pair_mass),
        pair_distribution_kl=kl_divergence(ref_pair_distribution, cand_pair_distribution, eps=eps),
    )
