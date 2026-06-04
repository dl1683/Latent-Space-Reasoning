"""Counterfactual answer proposals for exact-answer repair loops."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import floor

from latent_reasoning.eval.general_reasoning import GeneralReasoningTask


@dataclass(frozen=True)
class AnswerProposal:
    """A proposed exact answer and the source that produced it."""

    value: str
    source: str


def counterfactual_answer_proposals(
    task: GeneralReasoningTask,
    extracted_answer: object | None,
    *,
    limit: int | None = None,
) -> list[AnswerProposal]:
    """Generate answer alternatives without reading the hidden expected answer."""
    proposals: list[AnswerProposal] = []
    if task.answer_type == "multiple_choice" and task.choices:
        proposals.extend(
            AnswerProposal(value=letter.upper(), source="multiple_choice_options")
            for letter in sorted(task.choices)
        )
    elif task.answer_type == "short_text":
        proposals.extend(
            AnswerProposal(value=value, source="prompt_answer_options")
            for value in _short_text_candidates_from_prompt(task.prompt)
        )
        symbolic = symbolic_short_text_candidate_from_prompt(task.prompt)
        if symbolic is not None:
            proposals.append(AnswerProposal(value=symbolic, source="symbolic_prompt_solver"))
    elif task.answer_type == "integer":
        integer = _integer_candidate_from_prompt(task.prompt)
        if integer is not None:
            proposals.append(AnswerProposal(value=str(integer), source="arithmetic_prompt_solver"))

    filtered = [
        proposal
        for proposal in _dedupe_proposals(proposals)
        if not _matches_extracted_answer(proposal.value, extracted_answer)
    ]
    return filtered if limit is None else filtered[:limit]


def counterfactual_answer_candidates(
    task: GeneralReasoningTask,
    extracted_answer: object | None,
    *,
    limit: int | None = None,
) -> list[str]:
    """Return only proposal values for tests and simple callers."""
    return [
        proposal.value
        for proposal in counterfactual_answer_proposals(task, extracted_answer, limit=limit)
    ]


def _short_text_candidates_from_prompt(prompt: str) -> list[str]:
    lower = prompt.lower()
    match = re.search(r"answer\s+(?:only\s+)?([a-z0-9 ,/]+?)(?:\.|$)", lower)
    if match is None:
        return []
    option_text = match.group(1)
    option_text = option_text.replace("/", " or ").replace(",", " or ")
    raw_options = [item.strip() for item in re.split(r"\bor\b", option_text) if item.strip()]
    candidates = []
    for option in raw_options:
        words = re.findall(r"[a-z0-9]+", option)
        if 0 < len(words) <= 3:
            candidates.append(" ".join(words))
    return _dedupe_values(candidates)


def symbolic_short_text_candidate_from_prompt(prompt: str) -> str | None:
    """Return a label-free symbolic short-text answer when the prompt is mechanically solvable."""

    return (
        _before_chain_candidate(prompt)
        or _list_swap_candidate(prompt)
        or _letter_code_transform_candidate(prompt)
        or _toggle_candidate(prompt)
        or _syllogism_candidate(prompt)
    )


def _before_chain_candidate(prompt: str) -> str | None:
    if "full order" not in prompt.lower():
        return None
    pairs = [
        (left.upper(), right.upper())
        for left, right in re.findall(r"\b([a-z])\s+is\s+before\s+([a-z])\b", prompt, flags=re.IGNORECASE)
    ]
    if not pairs:
        return None

    nodes = _dedupe_values([item for pair in pairs for item in pair])
    outgoing = {node: [] for node in nodes}
    incoming = dict.fromkeys(nodes, 0)
    for left, right in pairs:
        if right not in outgoing[left]:
            outgoing[left].append(right)
            incoming[right] += 1

    ready = [node for node in nodes if incoming[node] == 0]
    order = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in outgoing[node]:
            incoming[child] -= 1
            if incoming[child] == 0:
                ready.append(child)
    if len(order) != len(nodes):
        return None
    return " ".join(order)


def _list_swap_candidate(prompt: str) -> str | None:
    lower = prompt.lower()
    match = re.search(r"start with the list\s+(.+?)\.", lower)
    if match is None:
        return None
    items = [item.strip() for item in re.split(r",|\band\b", match.group(1)) if item.strip()]
    if not items:
        return None

    swaps = re.findall(
        r"swap the\s+([a-z0-9]+)\s+and\s+([a-z0-9]+)\s+items",
        lower,
    )
    for left_name, right_name in swaps:
        left = _ordinal_index(left_name)
        right = _ordinal_index(right_name)
        if left is None or right is None or left >= len(items) or right >= len(items):
            return None
        items[left], items[right] = items[right], items[left]
    return " ".join(items) if swaps else None


def _letter_code_transform_candidate(prompt: str) -> str | None:
    code_match = re.search(
        r"\bstarts\s+with\s+the\s+code\s+((?:[A-Z]\s+){1,9}[A-Z])\b",
        prompt,
    )
    if code_match is None:
        return None
    items = re.findall(r"\b[A-Z]\b", code_match.group(1))
    if len(items) < 2:
        return None

    lower = prompt.lower()
    operations: list[tuple[int, str, tuple[str, ...]]] = []
    for match in re.finditer(r"\brotate(?:\s+the\s+code)?\s+one\s+step\s+(left|right)\b", lower):
        operations.append((match.start(), "rotate", (match.group(1),)))
    for match in re.finditer(r"\bswap\s+the\s+(first|final|last)\s+two\s+letters\b", lower):
        operations.append((match.start(), "swap_pair", (match.group(1),)))
    for match in re.finditer(r"\bswap\s+the\s+([a-z0-9]+)\s+and\s+([a-z0-9]+)\s+letters\b", lower):
        operations.append((match.start(), "swap_ordinal", (match.group(1), match.group(2))))
    if not operations:
        return None

    for _, kind, args in sorted(operations):
        if kind == "rotate":
            direction = args[0]
            if direction == "left":
                items = [*items[1:], items[0]]
            elif direction == "right":
                items = [items[-1], *items[:-1]]
        elif kind == "swap_pair":
            if len(items) < 2:
                return None
            if args[0] == "first":
                left, right = 0, 1
            else:
                left, right = len(items) - 2, len(items) - 1
            items[left], items[right] = items[right], items[left]
        elif kind == "swap_ordinal":
            left = _ordinal_index(args[0])
            right = _ordinal_index(args[1])
            if left is None or right is None or left >= len(items) or right >= len(items):
                return None
            items[left], items[right] = items[right], items[left]
    return " ".join(items)


def _toggle_candidate(prompt: str) -> str | None:
    lower = prompt.lower()
    match = re.search(r"starts\s+(on|off).*?toggled\s+(\d+)\s+times", lower)
    if match is None:
        return None
    state = match.group(1)
    toggles = int(match.group(2))
    if toggles % 2 == 1:
        return "off" if state == "on" else "on"
    return state


def _syllogism_candidate(prompt: str) -> str | None:
    lower = prompt.lower()
    if "answer yes or no" not in lower:
        return None
    question = re.search(r"\bcan\s+(?:a|an)\s+([a-z][a-z-]*)\s+be\s+(?:a|an)\s+([a-z][a-z-]*)\b", lower)
    if question is None:
        return None
    subject = _symbolic_category(question.group(1))
    target = _symbolic_category(question.group(2))
    if not subject or not target:
        return None

    inheritance: dict[str, set[str]] = {}
    for child, parent in re.findall(r"\ball\s+([a-z][a-z-]*)\s+are\s+([a-z][a-z-]*)\b", lower):
        inheritance.setdefault(_symbolic_category(child), set()).add(_symbolic_category(parent))

    exclusions: set[frozenset[str]] = set()
    for left, right in re.findall(r"\bno\s+([a-z][a-z-]*)\s+are\s+([a-z][a-z-]*)\b", lower):
        exclusions.add(frozenset((_symbolic_category(left), _symbolic_category(right))))

    subject_closure = _category_closure(subject, inheritance)
    target_closure = _category_closure(target, inheritance)
    if target in subject_closure:
        return "yes"
    for left in subject_closure:
        for right in target_closure:
            if frozenset((left, right)) in exclusions:
                return "no"
    return None


def _category_closure(root: str, inheritance: dict[str, set[str]]) -> set[str]:
    closure = {root}
    frontier = [root]
    while frontier:
        current = frontier.pop()
        for parent in inheritance.get(current, set()):
            if parent in closure:
                continue
            closure.add(parent)
            frontier.append(parent)
    return closure


def _symbolic_category(term: str) -> str:
    normalized = term.lower().replace("-", " ").strip()
    if normalized.endswith("ies") and len(normalized) > 3:
        return f"{normalized[:-3]}y"
    if normalized.endswith("s") and len(normalized) > 3:
        return normalized[:-1]
    return normalized


def _integer_candidate_from_prompt(prompt: str) -> int | None:
    compact = prompt.replace(",", "").replace("$", "")
    lower = _normalize_number_words(compact.lower())
    for solver in (
        _warehouse_candidate,
        _train_minutes_candidate,
        _budget_candidate,
        _operation_sequence_candidate,
        _class_absence_candidate,
        _machine_rate_candidate,
        _grid_candidate,
        _notebook_candidate,
        _letter_code_candidate,
        _linear_function_candidate,
    ):
        candidate = solver(lower)
        if candidate is not None:
            return candidate
    return None


def _warehouse_candidate(text: str) -> int | None:
    match = re.search(
        r"(\d+)\s+shelves\s+with\s+(\d+)\s+boxes\s+each.*?remove\w*\s+(\d+)\s+carts\s+with\s+(\d+)\s+boxes",
        text,
    )
    if match is None:
        return None
    shelves, boxes_per_shelf, carts, boxes_per_cart = _ints(match)
    return shelves * boxes_per_shelf - carts * boxes_per_cart


def _train_minutes_candidate(text: str) -> int | None:
    match = re.search(
        r"travels\s+(\d+)\s+miles\s+at\s+(\d+)\s+miles per hour.*?stops?\s+for\s+(\d+)\s+minutes",
        text,
    )
    if match is None:
        return None
    miles, mph, stop_minutes = _ints(match)
    if mph == 0:
        return None
    return int((miles / mph) * 60 + stop_minutes)


def _budget_candidate(text: str) -> int | None:
    match = re.search(
        r"has\s+(\d+).*?each\s+\w+\s+costs\s+(\d+).*?shipping\s+is.*?(\d+)",
        text,
    )
    if match is None:
        return None
    budget, unit_cost, shipping = _ints(match)
    if unit_cost <= 0 or budget < shipping:
        return None
    return floor((budget - shipping) / unit_cost)


def _operation_sequence_candidate(text: str) -> int | None:
    match = re.search(
        r"start with\s+(-?\d+).*?double.*?add\s+(-?\d+).*?multiply by\s+(-?\d+)",
        text,
    )
    if match is None:
        return None
    start, addend, multiplier = _ints(match)
    return (start * 2 + addend) * multiplier


def _class_absence_candidate(text: str) -> int | None:
    match = re.search(r"classes\s+have\s+(.+?)\s+students.*?(\d+)\s+students\s+are\s+absent", text)
    if match is None:
        return None
    class_sizes = [int(value) for value in re.findall(r"\d+", match.group(1))]
    if not class_sizes:
        return None
    absent = int(match.group(2))
    return sum(class_sizes) - absent


def _machine_rate_candidate(text: str) -> int | None:
    match = re.search(
        r"(\d+)\s+identical machines make\s+(\d+)\s+parts in\s+(\d+)\s+hours.*?(\d+)\s+machines make in\s+(\d+)\s+hours",
        text,
    )
    if match is None:
        return None
    source_machines, source_parts, source_hours, target_machines, target_hours = _ints(match)
    denominator = source_machines * source_hours
    if denominator == 0:
        return None
    return int(source_parts * target_machines * target_hours / denominator)


def _grid_candidate(text: str) -> int | None:
    match = re.search(r"(\d+)\s+by\s+(\d+)\s+grid\s+has\s+(\d+)\s+cells\s+blocked", text)
    if match is None:
        return None
    width, height, blocked = _ints(match)
    return width * height - blocked


def _notebook_candidate(text: str) -> int | None:
    match = re.search(r"starts\s+with\s+(\d+)\s+pages.*?sections\s+use\s+(.+?)\s+pages", text)
    if match is None:
        return None
    start = int(match.group(1))
    used = [int(value) for value in re.findall(r"\d+", match.group(2))]
    if not used:
        return None
    return start - sum(used)


def _letter_code_candidate(text: str) -> int | None:
    match = re.search(r"maps\s+a\s+to\s+1.*?what number does\s+([a-z])\s+map to", text)
    if match is None:
        return None
    return ord(match.group(1)) - ord("a") + 1


def _linear_function_candidate(text: str) -> int | None:
    f_match = re.search(r"f\(x\)\s*=\s*([+-]?\d*)x\s*([+-]\s*\d+)?", text)
    g_match = re.search(r"g\(x\)\s*=\s*([+-]?\d*)x\s*([+-]\s*\d+)?", text)
    arg_match = re.search(r"what is\s+f\(g\((-?\d+)\)\)", text)
    if f_match is None or g_match is None or arg_match is None:
        return None
    f_coefficient, f_bias = _linear_terms(f_match)
    g_coefficient, g_bias = _linear_terms(g_match)
    argument = int(arg_match.group(1))
    return f_coefficient * (g_coefficient * argument + g_bias) + f_bias


def _linear_terms(match: re.Match[str]) -> tuple[int, int]:
    raw_coefficient = match.group(1)
    if raw_coefficient in {"", "+"}:
        coefficient = 1
    elif raw_coefficient == "-":
        coefficient = -1
    else:
        coefficient = int(raw_coefficient)
    raw_bias = match.group(2)
    bias = int(raw_bias.replace(" ", "")) if raw_bias else 0
    return coefficient, bias


def _ordinal_index(value: str) -> int | None:
    ordinals = {
        "first": 0,
        "1st": 0,
        "second": 1,
        "2nd": 1,
        "third": 2,
        "3rd": 2,
        "fourth": 3,
        "4th": 3,
        "fifth": 4,
        "5th": 4,
    }
    return ordinals.get(value)


def _ints(match: re.Match[str]) -> tuple[int, ...]:
    return tuple(int(item) for item in match.groups())


def _matches_extracted_answer(value: str, extracted_answer: object | None) -> bool:
    if extracted_answer is None:
        return False
    return _normalize(value) == _normalize(str(extracted_answer))


def _dedupe_proposals(proposals: list[AnswerProposal]) -> list[AnswerProposal]:
    seen = set()
    deduped = []
    for proposal in proposals:
        key = _normalize(proposal.value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(proposal)
    return deduped


def _dedupe_values(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        key = _normalize(value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _normalize(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _normalize_number_words(value: str) -> str:
    replacements = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    pattern = re.compile(r"\b(" + "|".join(replacements) + r")\b")
    return pattern.sub(lambda match: replacements[match.group(1)], value)
