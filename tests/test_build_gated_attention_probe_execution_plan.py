import json

from experiments.build_gated_attention_probe_execution_plan import (
    PRIMARY_MODEL,
    build_execution_plan,
    render_markdown,
)


def test_execution_plan_blocks_when_primary_model_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(_freeze()), encoding="utf-8")
    hf_cache = tmp_path / "hf" / "hub"
    hf_cache.mkdir(parents=True)

    plan = build_execution_plan(freeze_path=freeze, hf_cache=hf_cache)
    markdown = render_markdown(plan)

    assert plan["ready_for_primary_gpu_run"] is False
    assert "not cached locally" in plan["blocking_reasons"][0]
    assert plan["ordered_runs"][1]["id"] == "primary_position_shift_control"
    assert plan["ordered_runs"][2]["id"] == "primary_zero_prefix_control"
    assert "--control-mode zero_embedding" in plan["ordered_runs"][2]["command"]
    assert "mean last-integer accuracy" in plan["reporting_order"][0]
    assert "Ready for primary GPU run: `False`" in markdown


def test_execution_plan_ready_when_primary_cached_and_no_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(_freeze()), encoding="utf-8")
    hf_cache = tmp_path / "hf" / "hub"
    primary_cache = hf_cache / ("models--" + PRIMARY_MODEL.replace("/", "--")) / "snapshots" / "abc123"
    primary_cache.mkdir(parents=True)

    plan = build_execution_plan(freeze_path=freeze, hf_cache=hf_cache)

    assert plan["ready_for_primary_gpu_run"] is True
    assert plan["blocking_reasons"] == []
    assert plan["model_cache"]["primary_gated"]["cached"] is True
    assert plan["model_cache"]["primary_gated"]["snapshot_count"] == 1


def test_execution_plan_blocks_existing_primary_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(_freeze()), encoding="utf-8")
    hf_cache = tmp_path / "hf" / "hub"
    primary_cache = hf_cache / ("models--" + PRIMARY_MODEL.replace("/", "--")) / "snapshots" / "abc123"
    primary_cache.mkdir(parents=True)
    result_dir = tmp_path / "eval_results" / "gated_attention"
    result_dir.mkdir(parents=True)
    (result_dir / "qwen3_next_zero_prefix_result.json").write_text("{}", encoding="utf-8")

    plan = build_execution_plan(freeze_path=freeze, hf_cache=hf_cache)

    assert plan["ready_for_primary_gpu_run"] is False
    assert plan["existing_primary_results"] == [
        "eval_results\\gated_attention\\qwen3_next_zero_prefix_result.json"
    ]
    assert "already exist" in plan["blocking_reasons"][0]


def _freeze():
    return {
        "probe_id": "gated_attention_null_probe_v1",
        "task_preset": "lsr_25_arithmetic_plus_cache_debug",
        "commands": {
            "gated_position_shift": "python position",
            "gated_zero_prefix": "python zero --control-mode zero_embedding",
            "gated_primary_random_prefix": "python random",
        },
    }
