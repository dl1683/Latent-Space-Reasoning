import json

from experiments.analyze_latent_trajectory_aggregation import (
    build_aggregation_scout,
    render_markdown,
)


def test_aggregation_scout_promotes_composed_non_overlapping_components(tmp_path):
    components = tmp_path / "components.jsonl"
    components.write_text(
        "\n".join(
            [
                json.dumps(_component("task_a", "greedy", "greedy", "cause")),
                json.dumps(_component("task_a", "greedy", "greedy", "mitigation")),
                json.dumps(_component("task_a", "prefix", "prefix_perturbation", "edge_case")),
                json.dumps(_component("task_a", "repair", "diffusion_repair", "measurement")),
            ]
        ),
        encoding="utf-8",
    )

    result = build_aggregation_scout(component_path=components)
    task = result["tasks"][0]
    markdown = render_markdown(result)

    assert task["best_single_score"] == 0.5
    assert task["aggregate_score"] == 1.0
    assert task["component_gain"] == 2
    assert task["decision"]["status"] == "promoted_local_scout"
    assert result["summary"]["promoted_task_count"] == 1
    assert "not a promoted model result" in markdown


def test_aggregation_scout_blocks_unresolved_contradictions(tmp_path):
    components = tmp_path / "components.jsonl"
    components.write_text(
        "\n".join(
            [
                json.dumps(
                    _component(
                        "task_b",
                        "greedy",
                        "greedy",
                        "route_manager",
                        contradiction_group="route",
                        contradiction_label="manager",
                    )
                ),
                json.dumps(
                    _component(
                        "task_b",
                        "prefix",
                        "prefix_perturbation",
                        "route_skip",
                        contradiction_group="route",
                        contradiction_label="skip",
                    )
                ),
                json.dumps(_component("task_b", "repair", "diffusion_repair", "audit")),
            ]
        ),
        encoding="utf-8",
    )

    result = build_aggregation_scout(component_path=components)
    task = result["tasks"][0]

    assert task["aggregate_score"] > task["best_single_score"]
    assert task["contradiction_count"] == 1
    assert task["decision"]["status"] == "blocked_contradiction"
    assert result["summary"]["promoted_task_count"] == 0


def test_aggregation_scout_blocks_unsupported_additions(tmp_path):
    components = tmp_path / "components.jsonl"
    components.write_text(
        "\n".join(
            [
                json.dumps(_component("task_c", "greedy", "greedy", "snapshot")),
                json.dumps(_component("task_c", "greedy", "greedy", "idempotent")),
                json.dumps(
                    _component(
                        "task_c",
                        "repair",
                        "diffusion_repair",
                        "skip_validation",
                        supported=False,
                        unsupported_addition=True,
                    )
                ),
            ]
        ),
        encoding="utf-8",
    )

    result = build_aggregation_scout(component_path=components)
    task = result["tasks"][0]

    assert task["unsupported_addition_count"] == 1
    assert task["decision"]["status"] == "blocked_unsupported"
    assert result["summary"]["blocked_task_count"] == 1


def _component(
    task_id,
    trajectory_id,
    family,
    component_id,
    *,
    supported=True,
    unsupported_addition=False,
    contradiction_group="",
    contradiction_label="",
):
    return {
        "component_id": component_id,
        "component_type": "planning",
        "component_weight": 1.0,
        "contradiction_group": contradiction_group,
        "contradiction_label": contradiction_label,
        "source_span": f"{trajectory_id}:{component_id}",
        "support_score": 0.9,
        "supported": supported,
        "task_id": task_id,
        "trajectory_family": family,
        "trajectory_id": trajectory_id,
        "unsupported_addition": unsupported_addition,
    }
