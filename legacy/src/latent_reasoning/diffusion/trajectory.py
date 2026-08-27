"""Trajectory summaries for language-diffusion denoising histories."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+?\|>")


@dataclass(frozen=True)
class HistorySampleSummary:
    """Compact per-step trajectory metrics."""

    step: int
    mask_count: int
    eos_count: int
    visible_chars: int
    visible_text: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_history_samples(
    history_samples: list[dict[str, object]] | None,
    *,
    final_text: str,
    mask_token_id: int | None = None,
    eos_token_id: int | None = None,
    mask_token_text: str = "<|mask|>",
) -> dict[str, object] | None:
    """Summarize sampled denoising states for judge/evolution hooks."""
    if history_samples is None:
        return None

    sample_summaries = [
        _summarize_sample(
            sample,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            mask_token_text=mask_token_text,
        )
        for sample in history_samples
    ]
    final_norm = _normalize_visible_text(final_text)
    first_visible_step = next(
        (sample.step for sample in sample_summaries if sample.visible_chars > 0),
        None,
    )
    first_final_text_step = next(
        (
            sample.step
            for sample in sample_summaries
            if final_norm and _normalize_visible_text(sample.visible_text) == final_norm
        ),
        None,
    )
    first_mask_free_step = next(
        (sample.step for sample in sample_summaries if sample.mask_count == 0),
        None,
    )
    transition_metrics = _history_token_transition_metrics(
        history_samples,
        mask_token_id=mask_token_id,
    )

    return {
        "sample_count": len(sample_summaries),
        "first_visible_step": first_visible_step,
        "first_final_text_step": first_final_text_step,
        "first_mask_free_step": first_mask_free_step,
        "final_visible_chars": len(final_norm),
        "final_has_visible_text": bool(final_norm),
        **transition_metrics,
        "samples": [sample.to_dict() for sample in sample_summaries],
    }


def _summarize_sample(
    sample: dict[str, object],
    *,
    mask_token_id: int | None,
    eos_token_id: int | None,
    mask_token_text: str,
) -> HistorySampleSummary:
    token_ids = _coerce_token_ids(sample.get("generated_token_ids"))
    text = str(sample.get("text", ""))
    visible_text = _normalize_visible_text(_SPECIAL_TOKEN_RE.sub("", text.replace(mask_token_text, "")))
    mask_count = token_ids.count(mask_token_id) if mask_token_id is not None else text.count(mask_token_text)
    eos_count = token_ids.count(eos_token_id) if eos_token_id is not None else 0
    return HistorySampleSummary(
        step=int(sample.get("step", 0)),
        mask_count=mask_count,
        eos_count=eos_count,
        visible_chars=len(visible_text),
        visible_text=visible_text,
    )


def _coerce_token_ids(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value if isinstance(item, int)]


def _history_token_transition_metrics(
    history_samples: list[dict[str, object]],
    *,
    mask_token_id: int | None,
) -> dict[str, object]:
    newly_visible = 0
    committed_changes = 0
    committed_remasks = 0
    remasked_rewrites = 0
    mask_count_increases = 0
    previous_ids: list[int] | None = None
    previous_mask_count: int | None = None
    pending_remasks: dict[int, int] = {}

    for sample in history_samples:
        token_ids = _coerce_token_ids(sample.get("generated_token_ids"))
        if not token_ids:
            continue
        mask_count = token_ids.count(mask_token_id) if mask_token_id is not None else 0
        if previous_mask_count is not None and mask_count > previous_mask_count:
            mask_count_increases += mask_count - previous_mask_count
        if previous_ids is not None:
            for index, (previous_token, current_token) in enumerate(zip(previous_ids, token_ids, strict=False)):
                previous_masked = mask_token_id is not None and previous_token == mask_token_id
                current_masked = mask_token_id is not None and current_token == mask_token_id
                if previous_masked and not current_masked:
                    newly_visible += 1
                    remasked_token = pending_remasks.pop(index, None)
                    if remasked_token is not None and current_token != remasked_token:
                        remasked_rewrites += 1
                elif not previous_masked and current_masked:
                    committed_remasks += 1
                    pending_remasks[index] = previous_token
                elif not previous_masked and not current_masked and previous_token != current_token:
                    committed_changes += 1
        previous_ids = token_ids
        previous_mask_count = mask_count

    return {
        "newly_visible_token_count": newly_visible,
        "committed_token_change_count": committed_changes,
        "committed_token_remask_count": committed_remasks,
        "remasked_token_rewrite_count": remasked_rewrites,
        "mask_count_increase_count": mask_count_increases,
        "sampled_history_is_monotonic_fill": (
            committed_changes == 0
            and committed_remasks == 0
            and remasked_rewrites == 0
            and mask_count_increases == 0
        ),
    }


def _normalize_visible_text(text: str) -> str:
    return " ".join(text.split())
