import json

import experiments.build_gated_attention_probe_execution_plan as execution_plan
from experiments.build_gated_attention_probe_execution_plan import (
    PRIMARY_MODEL,
    build_execution_plan,
    _exception_chain_summary,
    render_markdown,
)


def test_execution_plan_blocks_when_primary_weights_are_not_cached(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _patch_runtime_supported(monkeypatch)
    _patch_modeling_supported(monkeypatch)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(_freeze()), encoding="utf-8")
    hf_cache = tmp_path / "hf" / "hub"
    hf_cache.mkdir(parents=True)

    plan = build_execution_plan(freeze_path=freeze, hf_cache=hf_cache)
    markdown = render_markdown(plan)

    assert plan["ready_for_primary_gpu_run"] is False
    assert "weights are not cached locally" in plan["blocking_reasons"][0]
    assert plan["ordered_runs"][1]["id"] == "primary_position_shift_control"
    assert plan["ordered_runs"][2]["id"] == "primary_zero_prefix_control"
    assert "--control-mode zero_embedding" in plan["ordered_runs"][2]["command"]
    assert "mean last-integer accuracy" in plan["reporting_order"][0]
    assert "Ready for primary GPU run: `False`" in markdown


def test_execution_plan_ready_when_primary_cached_and_no_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _patch_runtime_supported(monkeypatch)
    _patch_modeling_supported(monkeypatch)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(_freeze()), encoding="utf-8")
    hf_cache = tmp_path / "hf" / "hub"
    primary_cache = hf_cache / ("models--" + PRIMARY_MODEL.replace("/", "--")) / "snapshots" / "abc123"
    primary_cache.mkdir(parents=True)
    (primary_cache / "model.safetensors").write_text("fake", encoding="utf-8")

    plan = build_execution_plan(freeze_path=freeze, hf_cache=hf_cache)

    assert plan["ready_for_primary_gpu_run"] is True
    assert plan["blocking_reasons"] == []
    assert plan["model_cache"]["primary_gated"]["cache_dir_exists"] is True
    assert plan["model_cache"]["primary_gated"]["has_weight_files"] is True
    assert plan["model_cache"]["primary_gated"]["weight_file_count"] == 1
    assert plan["model_cache"]["primary_gated"]["snapshot_count"] == 1


def test_execution_plan_blocks_existing_primary_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _patch_runtime_supported(monkeypatch)
    _patch_modeling_supported(monkeypatch)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(_freeze()), encoding="utf-8")
    hf_cache = tmp_path / "hf" / "hub"
    primary_cache = hf_cache / ("models--" + PRIMARY_MODEL.replace("/", "--")) / "snapshots" / "abc123"
    primary_cache.mkdir(parents=True)
    (primary_cache / "model.safetensors").write_text("fake", encoding="utf-8")
    result_dir = tmp_path / "eval_results" / "gated_attention"
    result_dir.mkdir(parents=True)
    (result_dir / "qwen3_next_zero_prefix_result.json").write_text("{}", encoding="utf-8")

    plan = build_execution_plan(freeze_path=freeze, hf_cache=hf_cache)

    assert plan["ready_for_primary_gpu_run"] is False
    assert plan["existing_primary_results"] == [
        "eval_results\\gated_attention\\qwen3_next_zero_prefix_result.json"
    ]
    assert "already exist" in plan["blocking_reasons"][0]


def test_execution_plan_blocks_when_transformers_lacks_qwen3_next(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        execution_plan,
        "_runtime_compatibility_state",
        lambda: {
            "transformers_version": "4.test",
            "transformers_supports_qwen3_next": False,
            "current_runner_requires_transformers_inputs_embeds": True,
            "gguf_openai_compatible_servers_do_not_expose_soft_prefix_inputs_embeds": True,
        },
    )
    _patch_modeling_supported(monkeypatch)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(_freeze()), encoding="utf-8")
    hf_cache = tmp_path / "hf" / "hub"
    primary_cache = hf_cache / ("models--" + PRIMARY_MODEL.replace("/", "--")) / "snapshots" / "abc123"
    primary_cache.mkdir(parents=True)
    (primary_cache / "model.safetensors").write_text("fake", encoding="utf-8")

    plan = build_execution_plan(freeze_path=freeze, hf_cache=hf_cache)
    markdown = render_markdown(plan)

    assert plan["ready_for_primary_gpu_run"] is False
    assert "does not support model_type=qwen3_next" in plan["blocking_reasons"][0]
    assert plan["runtime_compatibility"]["transformers_supports_qwen3_next"] is False
    assert "Supports `qwen3_next`: `False`" in markdown


def test_execution_plan_blocks_when_qwen3_next_modeling_dependencies_fail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _patch_runtime_supported(monkeypatch)
    monkeypatch.setattr(
        execution_plan,
        "_modeling_dependency_state",
        lambda: {
            "fla_available": True,
            "triton_available": False,
            "causal_conv1d_available": True,
            "flash_linear_attention_available": False,
            "empty_model_constructible": False,
            "failure_type": "ModuleNotFoundError",
            "failure_summary": "No module named 'triton'",
        },
    )
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(_freeze()), encoding="utf-8")
    hf_cache = tmp_path / "hf" / "hub"
    primary_cache = hf_cache / ("models--" + PRIMARY_MODEL.replace("/", "--")) / "snapshots" / "abc123"
    primary_cache.mkdir(parents=True)
    (primary_cache / "model.safetensors").write_text("fake", encoding="utf-8")

    plan = build_execution_plan(freeze_path=freeze, hf_cache=hf_cache)
    markdown = render_markdown(plan)

    assert plan["ready_for_primary_gpu_run"] is False
    assert "model construction fails before weights load" in plan["blocking_reasons"][0]
    assert plan["modeling_dependency_state"]["triton_available"] is False
    assert "Empty Qwen3-Next model constructible: `False`" in markdown


def test_exception_chain_summary_preserves_root_cause():
    try:
        try:
            raise ModuleNotFoundError("No module named 'triton'")
        except ModuleNotFoundError as exc:
            raise RuntimeError("wrapper") from exc
    except RuntimeError as exc:
        summary = _exception_chain_summary(exc)

    assert "RuntimeError: wrapper" in summary
    assert "ModuleNotFoundError: No module named 'triton'" in summary


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


def _patch_runtime_supported(monkeypatch):
    monkeypatch.setattr(
        execution_plan,
        "_runtime_compatibility_state",
        lambda: {
            "transformers_version": "4.test",
            "transformers_supports_qwen3_next": True,
            "current_runner_requires_transformers_inputs_embeds": True,
            "gguf_openai_compatible_servers_do_not_expose_soft_prefix_inputs_embeds": True,
        },
    )


def _patch_modeling_supported(monkeypatch):
    monkeypatch.setattr(
        execution_plan,
        "_modeling_dependency_state",
        lambda: {
            "fla_available": True,
            "triton_available": True,
            "causal_conv1d_available": True,
            "flash_linear_attention_available": False,
            "empty_model_constructible": True,
            "failure_type": "none",
            "failure_summary": "none",
        },
    )
