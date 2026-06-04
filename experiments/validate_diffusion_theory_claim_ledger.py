"""Validate the diffusion theory claim ledger."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_LEDGER_PATH = Path("docs/DIFFUSION_THEORY_CLAIM_LEDGER.md")
DEFAULT_REQUIRED_BACKLINK_DOCS = (
    Path("README.md"),
    Path("docs/DIFFUSION_READER_GUIDE.md"),
    Path("docs/DIFFUSION_REASONING_GEOMETRY_THEORY.md"),
)
ALLOWED_STATUSES = {
    "boundary",
    "hypothesis",
    "supported-conditional",
    "validated-local",
}
LEDGER_REF = "DIFFUSION_THEORY_CLAIM_LEDGER.md"
MARKDOWN_REF_RE = re.compile(r"`([^`]+\.md)`|\[([^\]]+\.md)\]\(([^)]+)\)")


@dataclass(frozen=True)
class TheoryClaimRow:
    row_number: int
    claim_id: str
    status: str
    assertion: str
    current_evidence: str
    assumptions: str
    falsifier: str
    next_proof_obligation: str


@dataclass(frozen=True)
class TheoryLedgerIssue:
    line: int
    claim_id: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument(
        "--require-backlink-doc",
        action="append",
        dest="backlink_docs",
        help="Document that must reference DIFFUSION_THEORY_CLAIM_LEDGER.md.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backlink_docs = (
        tuple(Path(path) for path in args.backlink_docs)
        if args.backlink_docs
        else DEFAULT_REQUIRED_BACKLINK_DOCS
    )
    issues = validate_theory_claim_ledger(
        ledger_path=args.ledger,
        required_backlink_docs=backlink_docs,
    )
    if args.json:
        print(
            json.dumps(
                {"issue_count": len(issues), "issues": [asdict(issue) for issue in issues]},
                indent=2,
                sort_keys=True,
            )
        )
    elif issues:
        for issue in issues:
            location = f"{args.ledger}:{issue.line}" if issue.line else str(args.ledger)
            print(f"ERROR: {location}: {issue.claim_id}: {issue.reason}", file=sys.stderr)
    else:
        print("Diffusion theory claim ledger is valid.")
    return 1 if issues else 0


def validate_theory_claim_ledger(
    *,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    required_backlink_docs: tuple[Path, ...] = DEFAULT_REQUIRED_BACKLINK_DOCS,
) -> list[TheoryLedgerIssue]:
    issues: list[TheoryLedgerIssue] = []
    if not ledger_path.exists():
        return [TheoryLedgerIssue(line=0, claim_id="", reason="ledger file is missing")]
    rows, parse_issues = parse_theory_claim_rows(ledger_path)
    issues.extend(parse_issues)
    if not rows:
        issues.append(TheoryLedgerIssue(line=0, claim_id="", reason="ledger has no claim rows"))
        return issues
    issues.extend(_validate_rows(rows, ledger_path=ledger_path))
    issues.extend(_validate_backlinks(required_backlink_docs))
    return issues


def parse_theory_claim_rows(path: Path) -> tuple[list[TheoryClaimRow], list[TheoryLedgerIssue]]:
    rows: list[TheoryClaimRow] = []
    issues: list[TheoryLedgerIssue] = []
    in_table = False
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("| ID | Status | Assertion |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not stripped:
            break
        if stripped.startswith("| ---"):
            continue
        if not stripped.startswith("|"):
            break
        cells = _markdown_cells(stripped)
        if len(cells) != 7:
            issues.append(
                TheoryLedgerIssue(
                    line=line_number,
                    claim_id=cells[0] if cells else "",
                    reason=f"expected 7 table cells, found {len(cells)}",
                )
            )
            continue
        rows.append(
            TheoryClaimRow(
                row_number=line_number,
                claim_id=cells[0],
                status=cells[1],
                assertion=cells[2],
                current_evidence=cells[3],
                assumptions=cells[4],
                falsifier=cells[5],
                next_proof_obligation=cells[6],
            )
        )
    return rows, issues


def _validate_rows(rows: list[TheoryClaimRow], *, ledger_path: Path) -> list[TheoryLedgerIssue]:
    issues: list[TheoryLedgerIssue] = []
    seen_ids: set[str] = set()
    expected_index = 1
    for row in rows:
        if row.claim_id in seen_ids:
            issues.append(_issue(row, f"duplicate claim id {row.claim_id}"))
        seen_ids.add(row.claim_id)
        expected_id = f"T{expected_index}"
        if row.claim_id != expected_id:
            issues.append(_issue(row, f"expected ordered claim id {expected_id}"))
        expected_index += 1
        if row.status not in ALLOWED_STATUSES:
            issues.append(_issue(row, f"unsupported status {row.status!r}"))
        required_fields = {
            "assertion": row.assertion,
            "current evidence": row.current_evidence,
            "assumptions": row.assumptions,
            "falsifier": row.falsifier,
            "next proof obligation": row.next_proof_obligation,
        }
        for name, value in required_fields.items():
            if not value.strip():
                issues.append(_issue(row, f"missing {name}"))
        if not _looks_like_falsifier(row.falsifier):
            issues.append(_issue(row, "falsifier should name a disconfirming condition"))
        if not _looks_like_action(row.next_proof_obligation):
            issues.append(_issue(row, "next proof obligation should name an action"))
        issues.extend(_validate_markdown_refs(row, ledger_path=ledger_path))
    return issues


def _validate_markdown_refs(
    row: TheoryClaimRow,
    *,
    ledger_path: Path,
) -> list[TheoryLedgerIssue]:
    issues: list[TheoryLedgerIssue] = []
    for ref in _markdown_refs(row.current_evidence):
        if ref.startswith("http://") or ref.startswith("https://"):
            continue
        candidate = _resolve_markdown_ref(ref, ledger_path=ledger_path)
        if not candidate.exists():
            issues.append(_issue(row, f"referenced evidence file does not exist: {ref}"))
    return issues


def _validate_backlinks(required_docs: tuple[Path, ...]) -> list[TheoryLedgerIssue]:
    issues: list[TheoryLedgerIssue] = []
    for path in required_docs:
        if not path.exists():
            issues.append(
                TheoryLedgerIssue(
                    line=0,
                    claim_id="",
                    reason=f"required backlink doc is missing: {path.as_posix()}",
                )
            )
            continue
        if LEDGER_REF not in path.read_text(encoding="utf-8"):
            issues.append(
                TheoryLedgerIssue(
                    line=0,
                    claim_id="",
                    reason=f"required backlink doc does not mention {LEDGER_REF}: {path.as_posix()}",
                )
            )
    return issues


def _markdown_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _markdown_refs(text: str) -> list[str]:
    refs = []
    for match in MARKDOWN_REF_RE.finditer(text):
        refs.append(match.group(1) or match.group(3) or "")
    return [ref for ref in refs if ref]


def _resolve_markdown_ref(ref: str, *, ledger_path: Path) -> Path:
    normalized = ref.replace("\\", "/")
    if normalized.startswith("../"):
        return (ledger_path.parent / normalized).resolve()
    if "/" in normalized:
        return Path(normalized)
    root_candidate = Path(normalized)
    if root_candidate.exists():
        return root_candidate
    return ledger_path.parent / normalized


def _looks_like_falsifier(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "fails",
            "failure",
            "matches",
            "beats",
            "dominates",
            "misclassified",
            "regresses",
            "without",
        )
    )


def _looks_like_action(text: str) -> bool:
    lowered = text.lower()
    return any(
        lowered.startswith(verb)
        for verb in (
            "add ",
            "build ",
            "compare ",
            "re-run ",
            "replace ",
            "report ",
            "run ",
            "test ",
            "train",
        )
    )


def _issue(row: TheoryClaimRow, reason: str) -> TheoryLedgerIssue:
    return TheoryLedgerIssue(line=row.row_number, claim_id=row.claim_id, reason=reason)


if __name__ == "__main__":
    raise SystemExit(main())
