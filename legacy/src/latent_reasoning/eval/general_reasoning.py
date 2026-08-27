"""General-purpose reasoning task scoring for scout benchmarks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GeneralReasoningTask:
    """A locked benchmark task from the scout/pilot manifests."""

    task_id: str
    family: str
    prompt: str
    answer_type: str
    scorer: str
    max_new_tokens: int
    answer: object | None = None
    choices: dict[str, str] | None = None
    rubric_items: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskScore:
    """Task score normalized to 0-1 with scorer-specific details."""

    score: float
    extracted_answer: object | None
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "extracted_answer": self.extracted_answer,
            "details": self.details,
        }


def load_tasks(path: str | Path) -> list[GeneralReasoningTask]:
    """Load JSONL task manifest."""
    tasks: list[GeneralReasoningTask] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            tasks.append(
                GeneralReasoningTask(
                    task_id=str(record["task_id"]),
                    family=str(record["family"]),
                    prompt=str(record["prompt"]),
                    answer_type=str(record["answer_type"]),
                    scorer=str(record["scorer"]),
                    max_new_tokens=int(record.get("max_new_tokens", 64)),
                    answer=record.get("answer"),
                    choices=record.get("choices"),
                    rubric_items=tuple(record.get("rubric_items", ())),
                )
            )
    return tasks


def score_task_output(task: GeneralReasoningTask, text: str) -> TaskScore:
    """Score generated text for one task."""
    if task.answer_type == "rubric":
        return score_planning_output(task, text)
    if task.answer_type == "integer":
        return score_integer_output(task, text)
    if task.answer_type == "multiple_choice":
        return score_multiple_choice_output(task, text)
    if task.answer_type == "short_text":
        return score_short_text_output(task, text)
    raise ValueError(f"Unsupported answer_type: {task.answer_type}")


def score_planning_output(task: GeneralReasoningTask, text: str) -> TaskScore:
    """Fixed cheap planning rubric used before external judges."""
    normalized = _normalize(text)
    words = normalized.split()
    completion = _score_completion(normalized, len(words))
    causal = _score_keyword_hits(normalized, _causal_terms(task))
    specificity = _score_specificity(normalized)
    constraints = _score_keyword_hits(normalized, _constraint_terms(task))
    risk = _score_keyword_hits(normalized, _risk_terms(task))

    rubric_hits = []
    for item in task.rubric_items:
        hit = _rubric_item_hit(normalized, item)
        rubric_hits.append({"item": item, "hit": hit})
    rubric_coverage = (
        sum(1 for item in rubric_hits if item["hit"]) / len(rubric_hits)
        if rubric_hits
        else 0.0
    )

    raw = {
        "completion": completion,
        "causal_diagnosis": causal,
        "specificity": specificity,
        "constraint_handling": constraints,
        "risk_awareness": risk,
        "rubric_coverage": rubric_coverage,
    }
    score = (
        0.18 * completion
        + 0.20 * causal
        + 0.20 * specificity
        + 0.17 * constraints
        + 0.15 * risk
        + 0.10 * rubric_coverage
    )
    return TaskScore(
        score=max(0.0, min(1.0, score)),
        extracted_answer=None,
        details={**raw, "rubric_hits": rubric_hits, "word_count": len(words)},
    )


def score_integer_output(task: GeneralReasoningTask, text: str) -> TaskScore:
    expected = int(task.answer)
    numbers = [int(match) for match in re.findall(r"(?<![\w.])-?\d+(?![\w.])", text.replace(",", ""))]
    extracted = numbers[-1] if numbers else None
    answer_anywhere = expected in numbers
    final_correct = extracted == expected
    return TaskScore(
        score=1.0 if final_correct else 0.0,
        extracted_answer=extracted,
        details={
            "expected": expected,
            "answer_anywhere": answer_anywhere,
            "parse_failure": extracted is None,
            "numbers": numbers[-5:],
        },
    )


def score_multiple_choice_output(task: GeneralReasoningTask, text: str) -> TaskScore:
    expected = str(task.answer).upper()
    choices = task.choices or {}
    extracted = extract_choice(text, choices)
    return TaskScore(
        score=1.0 if extracted == expected else 0.0,
        extracted_answer=extracted,
        details={
            "expected": expected,
            "parse_failure": extracted is None,
        },
    )


def score_short_text_output(task: GeneralReasoningTask, text: str) -> TaskScore:
    expected = _normalize(str(task.answer))
    normalized = _normalize(text)
    correct = _contains_expected_token_sequence(normalized, expected)
    extracted = expected if correct else _extract_short_text_candidate(normalized, expected)
    return TaskScore(
        score=1.0 if correct else 0.0,
        extracted_answer=extracted,
        details={"expected": expected, "parse_failure": extracted is None},
    )


def extract_choice(text: str, choices: dict[str, str]) -> str | None:
    """Extract a multiple-choice letter from model text."""
    upper = text.upper()
    explicit = re.findall(r"(?:^|\b)(?:ANSWER\s*[:\-]?\s*)?\(?([A-D])\)?(?:\.|\b)", upper)
    if explicit:
        return explicit[-1]
    normalized = _normalize(text)
    for letter, choice_text in choices.items():
        if _normalize(choice_text) in normalized:
            return letter.upper()
    return None


def _score_completion(normalized: str, word_count: int) -> float:
    if not normalized:
        return 0.0
    if word_count < 12:
        return 0.25
    if word_count < 30:
        return 0.65
    if normalized.endswith((",", ";", "and", "or", "then", "because")):
        return 0.6
    return 1.0


def _score_specificity(normalized: str) -> float:
    concrete_markers = (
        "measure",
        "record",
        "compare",
        "isolate",
        "rollback",
        "validate",
        "check",
        "test",
        "threshold",
        "sample",
        "log",
        "metric",
        "baseline",
        "failure",
        "decision",
        "owner",
        "time",
    )
    hits = sum(1 for marker in concrete_markers if marker in normalized)
    step_markers = len(re.findall(r"\b(first|second|third|then|next|before|after|finally)\b", normalized))
    return min(1.0, (hits / 7.0) * 0.75 + min(1.0, step_markers / 4.0) * 0.25)


def _score_keyword_hits(normalized: str, terms: tuple[str, ...]) -> float:
    if not terms:
        return 0.0
    hits = sum(1 for term in terms if term in normalized)
    return min(1.0, hits / min(4, len(terms)))


def _rubric_item_hit(normalized: str, item: str) -> bool:
    words = [word for word in re.findall(r"[a-z0-9]+", item.lower()) if len(word) > 3]
    if not words:
        return False
    hits = sum(1 for word in words if word in normalized)
    return hits >= max(1, min(3, len(words)) // 2)


def _causal_terms(task: GeneralReasoningTask) -> tuple[str, ...]:
    return (
        "because",
        "cause",
        "root",
        "isolate",
        "confound",
        "tradeoff",
        "mechanism",
        "why",
        *tuple(_keywords_from_rubric(task, ("cause", "diagnos", "tradeoff", "isolate"))),
    )


def _constraint_terms(task: GeneralReasoningTask) -> tuple[str, ...]:
    return (
        "only",
        "budget",
        "constraint",
        "limit",
        "overnight",
        "rollback",
        "preserve",
        "before",
        "after",
        *tuple(_keywords_from_rubric(task, ("constraint", "budget", "preserve", "limit"))),
    )


def _risk_terms(task: GeneralReasoningTask) -> tuple[str, ...]:
    return (
        "risk",
        "fail",
        "failure",
        "rollback",
        "regression",
        "monitor",
        "validate",
        "guard",
        "fallback",
        "publishable",
        *tuple(_keywords_from_rubric(task, ("risk", "fail", "rollback", "monitor"))),
    )


def _keywords_from_rubric(task: GeneralReasoningTask, stems: tuple[str, ...]) -> tuple[str, ...]:
    terms: set[str] = set()
    for item in task.rubric_items:
        for word in re.findall(r"[a-z0-9]+", item.lower()):
            if any(word.startswith(stem) for stem in stems):
                terms.add(word)
    return tuple(sorted(terms))


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _contains_expected_token_sequence(normalized_text: str, expected: str) -> bool:
    text_tokens = re.findall(r"[a-z0-9]+", normalized_text)
    expected_tokens = re.findall(r"[a-z0-9]+", expected)
    if not expected_tokens:
        return False
    if len(expected_tokens) == 1:
        return expected_tokens[0] in text_tokens
    window = len(expected_tokens)
    return any(text_tokens[index : index + window] == expected_tokens for index in range(len(text_tokens)))


def _extract_short_text_candidate(normalized_text: str, expected: str) -> str | None:
    text_tokens = re.findall(r"[a-z0-9]+", normalized_text)
    expected_tokens = re.findall(r"[a-z0-9]+", expected)
    if not text_tokens:
        return None
    if len(expected_tokens) <= 1:
        return text_tokens[-1]
    window = min(len(expected_tokens), len(text_tokens))
    return " ".join(text_tokens[-window:])
