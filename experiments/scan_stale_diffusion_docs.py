"""Scan public docs for stale diffusion evidence artifact references.

The diffusion benchmark history intentionally keeps many diagnostic artifacts.
This scanner is narrower: it guards public-facing docs against calling an old
score/report/raw file current, canonical, promoted, or public after the ground
truth index has moved on.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_INDEX_PATH = Path("eval_results/diffusion_language/diffusion_ground_truth_index.json")
DEFAULT_DOC_PATHS = (
    Path("README.md"),
    Path("RESEARCH_BRIEF.md"),
    Path("ARTICLE_UPDATE.md"),
    Path("DIFFUSION_PUBLIC_BENCHMARK.md"),
    Path("EXPERIMENTS.md"),
)

ARTIFACT_RE = re.compile(
    r"eval_results[\\/]+diffusion_language[\\/]+[A-Za-z0-9_.-]+"
    r"(?:_scores\.json|_report\.md|_raw\.jsonl)"
)
PUBLIC_CLAIM_RE = re.compile(
    r"\b("
    r"best|budget|canonical|claim|current|evidence|ground\s+truth|headline|latest|"
    r"promoted|public|supported|strongest|top[-\s]?score"
    r")\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class StaleDiffusionDocIssue:
    path: str
    line: int
    artifact: str
    reason: str
    context: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH))
    parser.add_argument(
        "--doc",
        action="append",
        dest="docs",
        help="Document to scan. Defaults to public top-level docs.",
    )
    parser.add_argument(
        "--strict-all-artifacts",
        action="store_true",
        help="Flag every non-canonical diffusion artifact reference, not just public-claim contexts.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    docs = tuple(Path(path) for path in args.docs) if args.docs else DEFAULT_DOC_PATHS
    issues = scan_stale_diffusion_docs(
        index_path=Path(args.index),
        doc_paths=docs,
        strict_all_artifacts=args.strict_all_artifacts,
    )
    if args.json:
        print(json.dumps({"issue_count": len(issues), "issues": [asdict(issue) for issue in issues]}, indent=2))
    elif issues:
        for issue in issues:
            print(
                f"ERROR: {issue.path}:{issue.line}: {issue.reason}: {issue.artifact}",
                file=sys.stderr,
            )
    else:
        print("No stale diffusion public-doc artifact references found.")
    return 1 if issues else 0


def scan_stale_diffusion_docs(
    *,
    index_path: Path = DEFAULT_INDEX_PATH,
    doc_paths: tuple[Path, ...] = DEFAULT_DOC_PATHS,
    strict_all_artifacts: bool = False,
) -> list[StaleDiffusionDocIssue]:
    canonical_artifacts = load_canonical_diffusion_artifacts(index_path)
    issues: list[StaleDiffusionDocIssue] = []
    for doc_path in doc_paths:
        if not doc_path.exists():
            continue
        issues.extend(
            _scan_doc(
                doc_path,
                canonical_artifacts=canonical_artifacts,
                strict_all_artifacts=strict_all_artifacts,
            )
        )
    return issues


def load_canonical_diffusion_artifacts(index_path: Path = DEFAULT_INDEX_PATH) -> set[str]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    artifacts: set[str] = set()
    for claim in _iter_claim_records(index):
        files = claim.get("canonical_files", {})
        if not isinstance(files, dict):
            continue
        for value in files.values():
            if isinstance(value, str) and value:
                artifacts.add(_normalize_artifact_path(value))
    return artifacts


def _iter_claim_records(index: dict[str, object]) -> list[dict[str, object]]:
    claims = index.get("claims", [])
    if not isinstance(claims, list):
        return []
    return [claim for claim in claims if isinstance(claim, dict)]


def _scan_doc(
    doc_path: Path,
    *,
    canonical_artifacts: set[str],
    strict_all_artifacts: bool,
) -> list[StaleDiffusionDocIssue]:
    lines = doc_path.read_text(encoding="utf-8").splitlines()
    issues: list[StaleDiffusionDocIssue] = []
    for index, line in enumerate(lines):
        for match in ARTIFACT_RE.finditer(line):
            artifact = _normalize_artifact_path(match.group(0))
            if artifact in canonical_artifacts:
                continue
            context = _context_window(lines, index)
            if not strict_all_artifacts and not _looks_like_public_claim_context(context):
                continue
            reason = (
                "non-canonical diffusion artifact in public-claim context"
                if not strict_all_artifacts
                else "non-canonical diffusion artifact"
            )
            issues.append(
                StaleDiffusionDocIssue(
                    path=doc_path.as_posix(),
                    line=index + 1,
                    artifact=artifact,
                    reason=reason,
                    context=context,
                )
            )
    return issues


def _context_window(lines: list[str], index: int, *, radius: int = 2) -> str:
    start = max(0, index - radius)
    end = min(len(lines), index + radius + 1)
    return "\n".join(lines[start:end])


def _looks_like_public_claim_context(context: str) -> bool:
    return bool(PUBLIC_CLAIM_RE.search(context))


def _normalize_artifact_path(path: str) -> str:
    return path.replace("\\", "/")


if __name__ == "__main__":
    raise SystemExit(main())
