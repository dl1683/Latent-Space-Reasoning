"""Tests for ARC-AGI-3 run diagnostics."""

import json

from experiments.analyze_arc3_run import _extract_scorecard, summarize_run


def test_extract_scorecard_from_harness_stdout():
    stdout = """
noise
--- FINAL SCORECARD REPORT ---
{
  "score": 0.25,
  "total_levels_completed": 1,
  "total_levels": 7,
  "total_actions": 42
}
after
"""

    scorecard = _extract_scorecard(stdout)

    assert scorecard["score"] == 0.25
    assert scorecard["total_levels_completed"] == 1


def test_summarize_run_combines_smoke_harness_and_trace(tmp_path):
    harness = tmp_path / "harness.json"
    trace = tmp_path / "trace.jsonl"
    smoke = tmp_path / "smoke.json"
    harness.write_text(
        json.dumps(
            {
                "stdout": """
--- FINAL SCORECARD REPORT ---
{
  "score": 0.0,
  "total_levels_completed": 0,
  "total_levels": 7,
  "total_actions": 66,
  "environments": [
    {
      "id": "ls20-test",
      "completed": false,
      "score": 0.0,
      "actions": 66,
      "levels_completed": 0
    }
  ]
}
"""
            }
        ),
        encoding="utf-8",
    )
    trace.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "normalized_action": "ACTION1",
                        "raw_transcript_chars": 1000,
                        "compact_transcript_chars": 250,
                        "latent_action": "ACTION1",
                        "mechanistic_action": "ACTION1",
                        "mechanistic_guard": "scripted_plan",
                    }
                ),
                json.dumps(
                    {
                        "normalized_action": "ACTION2",
                        "raw_transcript_chars": 800,
                        "compact_transcript_chars": 200,
                        "error": "cuda_out_of_memory",
                        "fallback_reason": "cuda_out_of_memory",
                    }
                ),
                json.dumps(
                    {
                        "normalized_action": "ACTION3",
                        "raw_plan": "MAX_LATENT_CALLS_FALLBACK: ACTION3",
                        "fallback_reason": "max_latent_calls",
                        "mechanistic_guard": "scripted_plan",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    smoke.write_text(
        json.dumps(
            {
                "server_ready": True,
                "harness_output": str(harness),
                "trace_jsonl": str(trace),
                "harness": {"returncode": 2, "completed": True},
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_run(smoke)

    assert summary["protocol"]["harness_completed"] is True
    assert summary["scorecard"]["environment_id"] == "ls20-test"
    assert summary["trace"]["records"] == 3
    assert summary["trace"]["action_counts"] == {"ACTION1": 1, "ACTION2": 1, "ACTION3": 1}
    assert summary["trace"]["error_counts"] == {"cuda_out_of_memory": 1}
    assert summary["trace"]["fallback_counts"] == {"cuda_out_of_memory": 1, "max_latent_calls": 1}
    assert summary["trace"]["compact_to_raw_ratio_mean"] == 0.25
    assert summary["attribution"]["official_score"] == 0.0
    assert summary["attribution"]["model_actions"] == 1
    assert summary["attribution"]["model_legal_actions"] == 1
    assert summary["attribution"]["model_aligned_with_mechanics"] == 1
    assert summary["attribution"]["fallback_actions"] == 2
    assert summary["attribution"]["scripted_actions"] == 2
