"""Label-free expanded planning-aspect extraction for v7 aggregation.

The extractor uses prompt and candidate text only. It deliberately does not read
task scores, rubric hits, or realized labels, so it can be used before v7 labels
exist.
"""

from __future__ import annotations

import re

EXPANDED_PLANNING_ASPECTS = (
    "owner_assignment",
    "timeline_or_sequence",
    "rollback_or_exit_criteria",
    "evidence_or_measurement",
    "scope_boundary",
    "polarity_or_action_direction",
)

_OWNER_RE = re.compile(
    r"\b(owner|responsib(?:le|ility)|assign(?:ed)?|accountable|lead|team|reviewer|maintainer|stakeholder|operator|on[- ]call)\b",
    re.IGNORECASE,
)
_TIMELINE_RE = re.compile(
    r"\b(first|then|next|before|after|during|phase|step|sequence|timeline|week|day|milestone|prerequisite|dependency)\b",
    re.IGNORECASE,
)
_ROLLBACK_RE = re.compile(
    r"\b(rollback|roll back|revert|back out|fallback|fall back|exit criteria|stop if|abort|escalate if|kill switch)\b",
    re.IGNORECASE,
)
_MEASUREMENT_RE = re.compile(
    r"\b(measure|metric|metrics|monitor|telemetry|log|logs|audit|verify|validate|evidence|threshold|kpi|dashboard|test result|success criteria)\b",
    re.IGNORECASE,
)
_SCOPE_RE = re.compile(
    r"\b(scope|boundary|only|except|unless|within|outside|out of scope|limit|constraint|applies to|does not apply)\b",
    re.IGNORECASE,
)
_POLARITY_RE = re.compile(
    r"\b(do not|don't|avoid|block|defer|delay|escalate|proceed|continue|stop|rollback|roll back|revert|must not|should not|require|allow|deny)\b",
    re.IGNORECASE,
)

_GENERIC_PROCESS_RE = re.compile(
    r"\b(plan|process|approach|strategy|system|workflow|framework|ensure|improve|consider)\b",
    re.IGNORECASE,
)

_ASPECT_PATTERNS = {
    "owner_assignment": _OWNER_RE,
    "timeline_or_sequence": _TIMELINE_RE,
    "rollback_or_exit_criteria": _ROLLBACK_RE,
    "evidence_or_measurement": _MEASUREMENT_RE,
    "scope_boundary": _SCOPE_RE,
    "polarity_or_action_direction": _POLARITY_RE,
}


def expanded_aspect_scores(text: str, *, prompt: str = "") -> dict[str, dict[str, object]]:
    """Return binary expanded-aspect support scores from text-only evidence."""

    normalized_text = _normalize_space(text)
    prompt_terms = _content_terms(prompt)
    scores: dict[str, dict[str, object]] = {}
    for aspect_id in EXPANDED_PLANNING_ASPECTS:
        spans = _support_spans(normalized_text, _ASPECT_PATTERNS[aspect_id])
        supported_spans = [
            span for span in spans if _span_has_task_content(span, prompt_terms) or aspect_id == "polarity_or_action_direction"
        ]
        scores[f"expanded::{aspect_id}"] = {
            "aspect_class": "expanded",
            "aspect_type": aspect_id,
            "source_spans": supported_spans,
            "support_score": 1.0 if supported_spans else 0.0,
        }
    return scores


def expanded_complement_aspects(
    *,
    anchor_text: str,
    candidate_text: str,
    prompt: str,
    trajectory_id: str,
) -> list[dict[str, object]]:
    """Return expanded aspects present in candidate text but absent in anchor text."""

    anchor_scores = expanded_aspect_scores(anchor_text, prompt=prompt)
    candidate_scores = expanded_aspect_scores(candidate_text, prompt=prompt)
    complements = []
    for aspect_id, candidate in candidate_scores.items():
        anchor_score = float(anchor_scores.get(aspect_id, {}).get("support_score", 0.0))
        candidate_score = float(candidate.get("support_score", 0.0))
        delta = candidate_score - anchor_score
        if delta <= 0:
            continue
        complements.append(
            {
                "aspect_class": "expanded",
                "aspect_id": aspect_id,
                "aspect_type": str(candidate.get("aspect_type", "")),
                "delta": delta,
                "source_spans": list(candidate.get("source_spans", [])),
                "support_score": candidate_score,
                "trajectory_id": trajectory_id,
            }
        )
    return complements


def _support_spans(text: str, pattern: re.Pattern[str]) -> list[str]:
    spans = []
    for sentence in _sentences(text):
        if pattern.search(sentence):
            spans.append(sentence)
    return spans[:3]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;])\s+|\n+", text)
    return [part.strip(" -\t\r\n") for part in parts if part.strip()]


def _span_has_task_content(span: str, prompt_terms: set[str]) -> bool:
    if not span:
        return False
    words = _content_terms(span)
    if words & prompt_terms:
        return True
    return len(words - _generic_terms()) >= 3


def _content_terms(text: str) -> set[str]:
    words = {word.lower() for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text)}
    return words - _generic_terms()


def _generic_terms() -> set[str]:
    return {
        "about",
        "action",
        "also",
        "answer",
        "approach",
        "create",
        "define",
        "ensure",
        "from",
        "have",
        "improve",
        "into",
        "make",
        "need",
        "plan",
        "process",
        "report",
        "should",
        "step",
        "system",
        "task",
        "that",
        "their",
        "then",
        "this",
        "with",
        "work",
    }


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
