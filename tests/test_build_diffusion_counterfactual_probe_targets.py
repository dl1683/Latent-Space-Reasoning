import json

from experiments.build_diffusion_counterfactual_probe_targets import (
    build_counterfactual_probe_targets,
    render_markdown,
)


def test_counterfactual_probe_targets_join_counterexamples_and_edge_rows(tmp_path):
    workbench = tmp_path / "workbench.json"
    edge_proxy = tmp_path / "edge_proxy.json"
    workbench.write_text(
        json.dumps(
            {
                "counterexamples": [
                    _counterexample("plan_a", error_type="false_negative", lift=0.2, gap=4),
                    _counterexample("plan_b", error_type="false_positive", lift=0.0, gap=12),
                ]
            }
        ),
        encoding="utf-8",
    )
    edge_proxy.write_text(
        json.dumps(
            {
                "rows": [
                    _edge("plan_a", label=True, lift=0.2, source_quality=0.2, gap=4),
                    _edge("plan_b", label=False, lift=0.0, source_quality=0.4, gap=12),
                ]
            }
        ),
        encoding="utf-8",
    )

    targets = build_counterfactual_probe_targets(
        edge_proxy_path=edge_proxy,
        probe_cost_relative=0.125,
        probe_policy="probe_test",
        workbench_path=workbench,
    )
    markdown = render_markdown(targets)

    assert targets["schema"] == "diffusion_counterfactual_probe_targets.v1"
    assert targets["summary"]["probe_row_count"] == 2
    assert targets["summary"]["positive_label_count"] == 1
    assert targets["summary"]["gate_decision"] == "diagnostic_only"
    assert targets["rows"][0]["pre_probe_features"]["prompt_gap_count"] == 4.0
    assert "Counterfactual Probe Targets" in markdown


def _counterexample(task_id, *, error_type, lift, gap):
    return {
        "error_type": error_type,
        "prompt_gap_count": gap,
        "repair_lift": lift,
        "repair_selector_edge": lift,
        "source_quality": 0.25,
        "source_task_delta_vs_trajectory": 0.0,
        "task_id": task_id,
    }


def _edge(task_id, *, label, lift, source_quality, gap):
    return {
        "candidate_lift_vs_trajectory": lift,
        "degeneracy_score": 0.2,
        "first_repairable_step": 4,
        "label": label,
        "max_span_target_score": 2.0,
        "min_span_source_relative_preservation": 0.8,
        "prompt_gap_count": gap,
        "source_quality": source_quality,
        "source_task_delta_vs_trajectory": 0.0,
        "task_id": task_id,
    }
