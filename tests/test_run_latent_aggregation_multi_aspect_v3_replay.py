import json

from experiments.run_latent_aggregation_multi_aspect_v3_replay import render_markdown, run_replay


def test_v3_replay_reports_coverage_conditional_and_global_gates(tmp_path):
    freeze = tmp_path / "freeze.json"
    raw = tmp_path / "raw.jsonl"
    tasks = tmp_path / "tasks.jsonl"
    probe = tmp_path / "probe.json"
    freeze.write_text(json.dumps(_freeze(["plan_a"])), encoding="utf-8")
    tasks.write_text(json.dumps(_task("plan_a")) + "\n", encoding="utf-8")
    probe.write_text(
        json.dumps({"summary": {"mean_probe_cost_relative": 0.1875, "measured_probe_count": 1}}),
        encoding="utf-8",
    )
    raw.write_text(
        "\n".join(
            [
                json.dumps(
                    _record(
                        "plan_a",
                        "anchor",
                        score=0.30,
                        text="preserve baseline",
                        specificity=0.1,
                    )
                ),
                json.dumps(
                    _record(
                        "plan_a",
                        "candidate",
                        score=0.25,
                        text="preserve baseline measure risk threshold",
                        specificity=0.8,
                    )
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_replay(freeze_path=freeze, raw_path=raw, tasks_path=tasks, probe_analysis_path=probe)
    summary = result["summary"]
    gates = {row["name"]: row for row in result["gate_evaluation"]["gates"]}

    assert summary["complement_coverage_count"] == 1
    assert summary["conditional_promoted_fraction"] == 1
    assert summary["all_task_mean_non_rubric_lift"] > 0
    assert summary["probe_cost_reported"] is True
    assert summary["equal_budget_best_of_control_reported"] is True
    assert gates["minimum_complement_coverage_count"]["status"] == "pass"
    assert gates["minimum_conditional_promoted_fraction"]["status"] == "pass"
    assert gates["must_report_probe_cost"]["status"] == "pass"


def test_v3_replay_fails_probe_cost_gate_when_probe_analysis_missing(tmp_path):
    freeze = tmp_path / "freeze.json"
    raw = tmp_path / "raw.jsonl"
    tasks = tmp_path / "tasks.jsonl"
    freeze.write_text(json.dumps(_freeze(["plan_b"])), encoding="utf-8")
    tasks.write_text(json.dumps(_task("plan_b")) + "\n", encoding="utf-8")
    raw.write_text(
        json.dumps(
            _record(
                "plan_b",
                "anchor",
                score=0.30,
                text="preserve baseline",
                specificity=0.1,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_replay(
        freeze_path=freeze,
        raw_path=raw,
        tasks_path=tasks,
        probe_analysis_path=tmp_path / "missing_probe.json",
    )
    gates = {row["name"]: row for row in result["gate_evaluation"]["gates"]}

    assert result["summary"]["probe_cost_reported"] is False
    assert gates["must_report_probe_cost"]["status"] == "fail"


def test_v3_replay_can_use_extra_raw_complement_source(tmp_path):
    freeze = tmp_path / "freeze.json"
    raw = tmp_path / "raw.jsonl"
    extra_raw = tmp_path / "extra_raw.jsonl"
    tasks = tmp_path / "tasks.jsonl"
    probe = tmp_path / "probe.json"
    freeze.write_text(json.dumps(_freeze(["plan_c"])), encoding="utf-8")
    tasks.write_text(json.dumps(_task("plan_c")) + "\n", encoding="utf-8")
    probe.write_text(
        json.dumps({"summary": {"mean_probe_cost_relative": 0.1875, "measured_probe_count": 1}}),
        encoding="utf-8",
    )
    raw.write_text(
        json.dumps(
            _record(
                "plan_c",
                "anchor",
                score=0.30,
                text="preserve baseline",
                specificity=0.1,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    extra_raw.write_text(
        json.dumps(
            _record(
                "plan_c",
                "probe",
                score=0.20,
                text="preserve baseline measure risk threshold",
                specificity=0.8,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_replay(
        freeze_path=freeze,
        raw_path=raw,
        extra_raw_paths=[extra_raw],
        tasks_path=tasks,
        probe_analysis_path=probe,
    )

    assert result["summary"]["complement_coverage_count"] == 1
    assert result["tasks"][0]["decision"]["status"] == "online_promoted_local"
    assert result["inputs"]["raw_paths"] == [str(raw), str(extra_raw)]


def test_v4_replay_marks_predeclared_multi_source_and_reports_diversity_cost(tmp_path):
    freeze = tmp_path / "freeze.json"
    raw = tmp_path / "raw.jsonl"
    probe_raw = tmp_path / "probe_raw.jsonl"
    diversity_raw = tmp_path / "diversity_raw.jsonl"
    tasks = tmp_path / "tasks.jsonl"
    probe = tmp_path / "probe.json"
    freeze_data = _freeze(["plan_d"])
    freeze_data["schema"] = "latent_aggregation_multi_aspect_v4_freeze.v1"
    freeze_data["trajectory_generation_contract"] = {
        "diversity_raw_output": str(diversity_raw).replace("/", "\\")
    }
    freeze_data["statistical_gates"]["must_report_diversity_generation_cost"] = True
    freeze.write_text(json.dumps(freeze_data), encoding="utf-8")
    tasks.write_text(json.dumps(_task("plan_d")) + "\n", encoding="utf-8")
    probe.write_text(
        json.dumps({"summary": {"mean_probe_cost_relative": 0.1875, "measured_probe_count": 1}}),
        encoding="utf-8",
    )
    raw.write_text(
        json.dumps(
            _record(
                "plan_d",
                "anchor",
                score=0.30,
                text="preserve baseline",
                specificity=0.1,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    probe_raw.write_text("", encoding="utf-8")
    diversity_raw.write_text(
        json.dumps(
            _record(
                "plan_d",
                "diversity",
                score=0.20,
                text="preserve baseline measure risk threshold",
                specificity=0.8,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_replay(
        freeze_path=freeze,
        raw_path=raw,
        extra_raw_paths=[probe_raw, diversity_raw],
        tasks_path=tasks,
        probe_analysis_path=probe,
    )
    gates = {row["name"]: row for row in result["gate_evaluation"]["gates"]}
    markdown = render_markdown(
        {key: value for key, value in result.items() if key not in {"aspect_rows", "realized_rows"}}
    )

    assert result["evidence_boundary"]["status"] == "fresh_predeclared_multi_source_v4_replay"
    assert result["summary"]["diversity_generation_cost_reported"] is True
    assert result["summary"]["diversity_raw_record_count"] == 1
    assert gates["must_report_diversity_generation_cost"]["status"] == "pass"
    assert "# Latent Aggregation Multi-Aspect V4 Replay" in markdown
    assert "Diversity generation cost reported: `True`" in markdown


def test_v5_replay_reports_robustness_and_source_family_ablations(tmp_path):
    freeze = tmp_path / "freeze.json"
    raw = tmp_path / "raw.jsonl"
    diversity_raw = tmp_path / "diversity_raw.jsonl"
    tasks = tmp_path / "tasks.jsonl"
    probe = tmp_path / "probe.json"
    freeze_data = _freeze(["plan_e", "plan_f"])
    freeze_data["schema"] = "latent_aggregation_multi_aspect_v5_freeze.v1"
    freeze_data["trajectory_generation_contract"] = {
        "raw_output": str(raw).replace("/", "\\"),
        "diversity_raw_output": str(diversity_raw).replace("/", "\\"),
    }
    freeze_data["statistical_gates"].update(
        {
            "minimum_aggregate_win_count": 2,
            "minimum_complement_coverage_count": 2,
            "minimum_complement_coverage_fraction": 1.0,
            "minimum_wilson_lower_bound": 0.0,
            "must_report_diversity_generation_cost": True,
            "must_report_theme_bucket_results": True,
        }
    )
    freeze_data["robustness_gates"] = {
        "maximum_single_task_share_of_total_lift": 0.75,
        "must_report_complement_yield_per_raw_row": True,
        "must_report_cost_normalized_lift": True,
        "must_report_high_leverage_task_ids": True,
        "must_report_leave_one_out_mean_lift_range": True,
        "must_report_median_non_rubric_lift": True,
        "must_report_median_score_lift": True,
        "must_report_source_family_ablation": True,
        "must_report_wins_ties_losses": True,
    }
    freeze_data["task_mix_contract"] = {
        "task_theme_by_id": {
            "plan_e": "statistical_validation",
            "plan_f": "systems_reliability",
        }
    }
    freeze.write_text(json.dumps(freeze_data), encoding="utf-8")
    tasks.write_text(
        json.dumps(_task("plan_e")) + "\n" + json.dumps(_task("plan_f")) + "\n",
        encoding="utf-8",
    )
    probe.write_text(
        json.dumps({"summary": {"mean_probe_cost_relative": 0.1875, "measured_probe_count": 2}}),
        encoding="utf-8",
    )
    raw.write_text(
        "\n".join(
            json.dumps(
                _record(
                    task_id,
                    "anchor",
                    score=0.30,
                    text="preserve baseline",
                    specificity=0.1,
                )
            )
            for task_id in ["plan_e", "plan_f"]
        )
        + "\n",
        encoding="utf-8",
    )
    diversity_raw.write_text(
        "\n".join(
            json.dumps(
                _record(
                    task_id,
                    "diversity",
                    score=0.20,
                    text="preserve baseline measure risk threshold",
                    specificity=0.8,
                )
            )
            for task_id in ["plan_e", "plan_f"]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_replay(
        freeze_path=freeze,
        raw_path=raw,
        extra_raw_paths=[diversity_raw],
        tasks_path=tasks,
        probe_analysis_path=probe,
    )
    summary = result["summary"]
    gates = {row["name"]: row for row in result["gate_evaluation"]["gates"]}
    markdown = render_markdown(
        {key: value for key, value in result.items() if key not in {"aspect_rows", "realized_rows"}}
    )

    assert result["evidence_boundary"]["status"] == "fresh_predeclared_multi_source_v5_replay"
    assert summary["wins_ties_losses"]["wins"] == 2
    assert summary["median_score_lift"] > 0
    assert summary["leave_one_out_mean_score_lift_range"][0] > 0
    assert summary["maximum_single_task_share_of_total_lift"] <= 0.75
    assert summary["selected_complement_source_family_counts"] == {"diversity": 2}
    assert set(summary["source_family_ablation"]) == {"diversity", "label"}
    assert set(summary["theme_bucket_results"]) == {"statistical_validation", "systems_reliability"}
    assert gates["must_report_source_family_ablation"]["status"] == "pass"
    assert gates["must_report_theme_bucket_results"]["status"] == "pass"
    assert gates["maximum_single_task_share_of_total_lift"]["status"] == "pass"
    assert "# Latent Aggregation Multi-Aspect V5 Replay" in markdown
    assert "Source-Family Ablation" in markdown


def _freeze(task_ids):
    return {
        "statistical_gates": {
            "maximum_hard_contradiction_count": 0,
            "maximum_unsupported_addition_count": 0,
            "minimum_aggregate_win_count": 1,
            "minimum_all_task_mean_non_rubric_lift": 0.01,
            "minimum_complement_coverage_count": 1,
            "minimum_complement_coverage_fraction": 0.1,
            "minimum_conditional_non_rubric_lift": 0.01,
            "minimum_conditional_promoted_fraction": 0.5,
            "minimum_task_count": len(task_ids),
            "minimum_wilson_lower_bound": 0.0,
            "must_report_equal_budget_best_of_control": True,
            "must_report_probe_cost": True,
            "must_report_rubric_and_dimension_gain_separately": True,
        },
        "task_ids": task_ids,
    }


def _task(task_id):
    return {
        "answer": None,
        "answer_type": "rubric",
        "family": "planning",
        "max_new_tokens": 64,
        "prompt": "Plan the experiment.",
        "rubric_items": ["preserve baseline"],
        "scorer": "planning_rubric_v1",
        "task_id": task_id,
    }


def _record(task_id, candidate_key, *, score, text, specificity):
    return {
        "candidate_key": candidate_key,
        "generation_stage": "candidate_generation",
        "schedule": {"name": "fixed"},
        "task": {"task_id": task_id},
        "task_score": {
            "details": {
                "causal_diagnosis": 0.0,
                "completion": 0.65,
                "constraint_handling": 0.0,
                "risk_awareness": 0.0,
                "rubric_coverage": 1.0,
                "rubric_hits": [{"hit": True, "item": "preserve baseline"}],
                "specificity": specificity,
            },
            "score": score,
        },
        "text": text,
    }
