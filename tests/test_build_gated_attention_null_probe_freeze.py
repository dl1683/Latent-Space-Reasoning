from experiments.build_gated_attention_null_probe_freeze import (
    FROZEN_RANDOM_PREFIX_SEEDS,
    build_freeze_manifest,
    render_markdown,
)


def test_gated_attention_freeze_locks_pre_result_boundary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    manifest = build_freeze_manifest(result_glob="eval_results/gated_attention/*result*.json")
    markdown = render_markdown(manifest)

    assert manifest["probe_id"] == "gated_attention_null_probe_v1"
    assert manifest["pre_result_boundary"]["no_gated_attention_outputs_seen"] is True
    assert manifest["conditions"][2]["seeds"] == list(FROZEN_RANDOM_PREFIX_SEEDS)
    assert manifest["infrastructure_gates"]["inputs_embeds_token_count_bug_fixed"] is True
    assert manifest["interpretation_gates"]["sink_dependent"]["oracle_coverage_lt"] == 0.60
    assert "--control-mode position_shift" in manifest["commands"]["gated_position_shift"]
    assert "Report mean metrics before oracle metrics" in markdown
    assert "Position-shift control" in markdown


def test_gated_attention_freeze_refuses_existing_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result_dir = tmp_path / "eval_results" / "gated_attention"
    result_dir.mkdir(parents=True)
    (result_dir / "qwen3_next_random_prefix_n10_result.json").write_text("{}", encoding="utf-8")

    try:
        build_freeze_manifest(result_glob="eval_results/gated_attention/*result*.json")
    except ValueError as exc:
        assert "result artifacts exist" in str(exc)
    else:
        raise AssertionError("expected existing gated-attention results to block freeze")


def test_gated_attention_freeze_can_be_rebuilt_for_audit_after_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result_dir = tmp_path / "eval_results" / "gated_attention"
    result_dir.mkdir(parents=True)
    (result_dir / "qwen3_next_random_prefix_n10_result.json").write_text("{}", encoding="utf-8")

    manifest = build_freeze_manifest(
        result_glob="eval_results/gated_attention/*result*.json",
        allow_existing_results=True,
    )

    assert manifest["pre_result_boundary"]["allow_existing_results"] is True
    assert manifest["pre_result_boundary"]["no_gated_attention_outputs_seen"] is False
