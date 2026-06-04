"""Diffusion-native trajectory control primitives.

This is the replacement surface for AR-first latent reasoning. A candidate is
not just a soft prefix; it can be a denoising schedule, remasking policy, or
future logits/token hook. The immediate implementation scores sampled
trajectory summaries so schedule search has a cheap objective before external
judges are added.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from latent_reasoning.diffusion.backends import DiffusionGenerationConfig


@dataclass(frozen=True)
class DiffusionScheduleCandidate:
    """A denoising schedule that can be searched, evolved, or judge-selected."""

    name: str
    steps: int
    max_new_tokens: int
    algorithm: str
    temperature: float = 0.0
    top_p: float | None = None
    alg_temp: float = 0.0
    block_length: int = 32
    remasking: str = "low_confidence"
    output_history: bool = True
    history_sample_count: int = 6
    revision_remask_fraction: float | None = None
    revision_steps: int = 0

    def to_config(self) -> DiffusionGenerationConfig:
        """Convert to backend generation config."""
        return DiffusionGenerationConfig(
            max_new_tokens=self.max_new_tokens,
            steps=self.steps,
            temperature=self.temperature,
            top_p=self.top_p,
            algorithm=self.algorithm,
            alg_temp=self.alg_temp,
            block_length=self.block_length,
            remasking=self.remasking,
            output_history=self.output_history,
            history_sample_count=self.history_sample_count,
            revision_remask_fraction=self.revision_remask_fraction,
            revision_steps=self.revision_steps,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryControlScore:
    """A cheap score for selecting denoising trajectories before judge calls."""

    overall: float
    final_text_score: float
    early_stability_score: float
    mask_resolution_score: float
    eos_pressure_penalty: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def default_dream_schedules(max_new_tokens: int = 64) -> tuple[DiffusionScheduleCandidate, ...]:
    """Small schedule pack for Dream before expensive benchmark runs."""
    return (
        DiffusionScheduleCandidate(
            name="entropy_32",
            steps=32,
            max_new_tokens=max_new_tokens,
            algorithm="entropy",
            temperature=0.2,
            top_p=0.95,
        ),
        DiffusionScheduleCandidate(
            name="entropy_64",
            steps=64,
            max_new_tokens=max_new_tokens,
            algorithm="entropy",
            temperature=0.2,
            top_p=0.95,
        ),
        DiffusionScheduleCandidate(
            name="origin_64",
            steps=64,
            max_new_tokens=max_new_tokens,
            algorithm="origin",
            temperature=0.2,
            top_p=0.95,
        ),
    )


def default_llada_schedules(max_new_tokens: int = 32) -> tuple[DiffusionScheduleCandidate, ...]:
    """Small schedule pack for LLaDA architecture checks."""
    return (
        DiffusionScheduleCandidate(
            name="low_confidence_32",
            steps=32,
            max_new_tokens=max_new_tokens,
            algorithm="low_confidence",
            temperature=0.0,
            block_length=max_new_tokens,
            remasking="low_confidence",
            output_history=True,
            history_sample_count=6,
        ),
        DiffusionScheduleCandidate(
            name="random_32",
            steps=32,
            max_new_tokens=max_new_tokens,
            algorithm="low_confidence",
            temperature=0.0,
            block_length=max_new_tokens,
            remasking="random",
            output_history=True,
            history_sample_count=6,
        ),
    )


def score_trajectory_summary(
    trajectory_summary: dict[str, object] | None,
    *,
    history_steps: int | None,
    final_text: str,
) -> TrajectoryControlScore:
    """Score a sampled denoising trajectory without calling an external judge."""
    normalized_text = " ".join(final_text.split())
    final_chars = len(normalized_text)
    final_text_score = min(1.0, final_chars / 180.0)
    if final_chars < 20:
        final_text_score *= 0.25

    if trajectory_summary is None or history_steps is None or history_steps <= 0:
        early_stability_score = 0.0
        mask_resolution_score = 1.0 if final_chars > 0 else 0.0
        eos_pressure_penalty = 0.0
    else:
        first_final = _optional_int(trajectory_summary.get("first_final_text_step"))
        first_mask_free = _optional_int(trajectory_summary.get("first_mask_free_step"))
        early_stability_score = 0.0
        if first_final is not None:
            early_stability_score = max(0.0, 1.0 - ((first_final - 1) / history_steps))
        mask_resolution_score = 0.0
        if first_mask_free is not None:
            mask_resolution_score = max(0.0, 1.0 - ((first_mask_free - 1) / history_steps) * 0.25)
        eos_pressure_penalty = _eos_pressure_penalty(trajectory_summary, history_steps)

    overall = (
        0.55 * final_text_score
        + 0.25 * early_stability_score
        + 0.20 * mask_resolution_score
        - eos_pressure_penalty
    )
    return TrajectoryControlScore(
        overall=max(0.0, min(1.0, overall)),
        final_text_score=final_text_score,
        early_stability_score=early_stability_score,
        mask_resolution_score=mask_resolution_score,
        eos_pressure_penalty=eos_pressure_penalty,
    )


def attach_control_score(record: dict[str, object]) -> dict[str, object]:
    """Return a copy of a generation record with trajectory control score."""
    scored = dict(record)
    score = score_trajectory_summary(
        _dict_or_none(record.get("trajectory_summary")),
        history_steps=_optional_int(record.get("history_steps")),
        final_text=str(record.get("text", "")),
    )
    scored["trajectory_control_score"] = score.to_dict()
    return scored


def _eos_pressure_penalty(trajectory_summary: dict[str, object], history_steps: int) -> float:
    samples = trajectory_summary.get("samples")
    if not isinstance(samples, list) or not samples:
        return 0.0
    penalties = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        step = _optional_int(sample.get("step"))
        eos_count = _optional_int(sample.get("eos_count")) or 0
        mask_count = _optional_int(sample.get("mask_count")) or 0
        if step is None or step >= history_steps:
            continue
        denom = max(1, eos_count + mask_count)
        penalties.append((eos_count / denom) * 0.15)
    return max(penalties, default=0.0)


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _dict_or_none(value: object) -> dict[str, object] | None:
    return value if isinstance(value, dict) else None
