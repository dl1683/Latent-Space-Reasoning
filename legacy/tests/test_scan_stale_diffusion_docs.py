"""Tests for stale public diffusion evidence scans."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.scan_stale_diffusion_docs import (
    load_canonical_diffusion_artifacts,
    scan_stale_diffusion_docs,
)


def _write_index(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "canonical_files": {
                            "scores": "eval_results/diffusion_language/current_scores.json",
                            "report": "eval_results/diffusion_language/current_report.md",
                            "raw": "eval_results/diffusion_language/current_raw.jsonl",
                        },
                        "claim_id": "current",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_load_canonical_diffusion_artifacts_normalizes_paths(tmp_path):
    index_path = tmp_path / "index.json"
    _write_index(index_path)

    artifacts = load_canonical_diffusion_artifacts(index_path)

    assert artifacts == {
        "eval_results/diffusion_language/current_scores.json",
        "eval_results/diffusion_language/current_report.md",
        "eval_results/diffusion_language/current_raw.jsonl",
    }


def test_scan_accepts_current_public_doc_artifacts(tmp_path):
    index_path = tmp_path / "index.json"
    doc_path = tmp_path / "README.md"
    _write_index(index_path)
    doc_path.write_text(
        "Current public evidence: "
        "eval_results/diffusion_language/current_report.md\n",
        encoding="utf-8",
    )

    issues = scan_stale_diffusion_docs(index_path=index_path, doc_paths=(doc_path,))

    assert issues == []


def test_scan_rejects_stale_artifact_in_public_claim_context(tmp_path):
    index_path = tmp_path / "index.json"
    doc_path = tmp_path / "README.md"
    _write_index(index_path)
    doc_path.write_text(
        "\n".join(
            [
                "Current public evidence:",
                "eval_results/diffusion_language/old_report.md",
            ]
        ),
        encoding="utf-8",
    )

    issues = scan_stale_diffusion_docs(index_path=index_path, doc_paths=(doc_path,))

    assert len(issues) == 1
    assert issues[0].artifact == "eval_results/diffusion_language/old_report.md"
    assert issues[0].line == 2


def test_scan_allows_historical_diagnostic_mentions_by_default(tmp_path):
    index_path = tmp_path / "index.json"
    doc_path = tmp_path / "RESEARCH_BRIEF.md"
    _write_index(index_path)
    doc_path.write_text(
        "Historical diagnostic: eval_results/diffusion_language/old_report.md\n",
        encoding="utf-8",
    )

    issues = scan_stale_diffusion_docs(index_path=index_path, doc_paths=(doc_path,))

    assert issues == []


def test_strict_scan_rejects_any_noncanonical_artifact(tmp_path):
    index_path = tmp_path / "index.json"
    doc_path = tmp_path / "ARTICLE_UPDATE.md"
    _write_index(index_path)
    doc_path.write_text(
        "Historical diagnostic: eval_results/diffusion_language/old_scores.json\n",
        encoding="utf-8",
    )

    issues = scan_stale_diffusion_docs(
        index_path=index_path,
        doc_paths=(doc_path,),
        strict_all_artifacts=True,
    )

    assert len(issues) == 1
    assert issues[0].reason == "non-canonical diffusion artifact"
