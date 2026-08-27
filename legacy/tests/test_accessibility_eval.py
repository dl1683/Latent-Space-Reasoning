"""Tests for accessibility-first evaluation helpers."""

from __future__ import annotations

import json

import pytest

from latent_reasoning.eval import load_compare_results, summarize_compare_results


def test_load_compare_results_from_dict_and_list(tmp_path):
    single_path = tmp_path / "single.json"
    list_path = tmp_path / "list.json"

    single_payload = {"query": "q1", "latent_score": 0.2}
    list_payload = [
        {"query": "q2", "latent_score": 0.4},
        {"query": "q3", "latent_score": 0.6},
    ]

    single_path.write_text(json.dumps(single_payload), encoding="utf-8")
    list_path.write_text(json.dumps(list_payload), encoding="utf-8")

    loaded = load_compare_results([single_path, list_path])
    assert len(loaded) == 3
    assert loaded[0]["query"] == "q1"
    assert loaded[1]["query"] == "q2"
    assert loaded[2]["query"] == "q3"


def test_load_compare_results_handles_utf8_bom(tmp_path):
    bom_path = tmp_path / "bom.json"
    payload = {"query": "bom", "latent_score": 0.5}

    # Simulate PowerShell/Windows BOM-authored JSON.
    bom_path.write_text(json.dumps(payload), encoding="utf-8-sig")

    loaded = load_compare_results([bom_path])
    assert len(loaded) == 1
    assert loaded[0]["query"] == "bom"


def test_summarize_compare_results_computes_tradeoff_metrics():
    results = [
        {
            "query": "q1",
            "baseline": "abc",
            "latent_reasoning": "abcdef",
            "latent_score": 0.4,
            "baseline_duration_s": 1.0,
            "latent_duration_s": 2.0,
            "latent_run_duration_s": 2.2,
            "latent_evolution_duration_s": 1.6,
            "latent_non_evolution_duration_s": 0.6,
            "latency_overhead_ratio": 2.0,
            "evaluations": 20,
            "generations": 4,
        },
        {
            "query": "q2",
            "baseline": "ab",
            "latent_reasoning": "abcd",
            "latent_score": 0.0,
            "baseline_duration_s": 1.5,
            "latent_duration_s": 2.5,
            "latent_run_duration_s": 2.7,
            "latent_evolution_duration_s": 2.0,
            "latent_non_evolution_duration_s": 0.7,
            "latency_overhead_ratio": 1.6666667,
            "evaluations": 10,
            "generations": 3,
        },
    ]

    summary = summarize_compare_results(results)

    assert summary["num_runs"] == 2
    assert summary["avg_latent_score"] == pytest.approx(0.2)
    assert summary["median_latent_score"] == pytest.approx(0.2)
    assert summary["avg_baseline_duration_s"] == pytest.approx(1.25)
    assert summary["avg_latent_duration_s"] == pytest.approx(2.25)
    assert summary["avg_latent_run_duration_s"] == pytest.approx(2.45)
    assert summary["avg_latent_evolution_duration_s"] == pytest.approx(1.8)
    assert summary["avg_latent_non_evolution_duration_s"] == pytest.approx(0.65)
    assert summary["avg_latency_overhead_ratio"] == pytest.approx(1.83333335)
    assert summary["avg_evaluations"] == pytest.approx(15.0)
    assert summary["avg_generations"] == pytest.approx(3.5)
    assert summary["avg_baseline_length_chars"] == pytest.approx(2.5)
    assert summary["avg_latent_length_chars"] == pytest.approx(5.0)
    assert summary["avg_evaluations_per_quality"] == pytest.approx(24.2857142857)
