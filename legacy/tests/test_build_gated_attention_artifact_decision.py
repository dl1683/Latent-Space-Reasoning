from experiments.build_gated_attention_artifact_decision import (
    GGUF_Q4_FILE,
    build_artifact_decision,
    render_markdown,
)


def test_artifact_decision_blocks_current_soft_prefix_runner_when_qwen3_next_unsupported():
    decision = build_artifact_decision(
        inventory=_inventory(),
        transformers_version="4.56.2",
        transformers_supports_qwen3_next=False,
        llama_cpp_available=False,
    )
    markdown = render_markdown(decision)

    assert decision["selected_immediate_primary_artifact"] is None
    assert "model_type=qwen3_next" in decision["blockers"][0]
    assert "direct 24GB single-GPU run" in decision["blockers"][1]
    assert decision["selected_download_candidate"]["file"] == GGUF_Q4_FILE
    assert "No immediate primary Qwen3-Next soft-prefix run" in markdown
    assert "Transformers supports `qwen3_next`: `False`" in markdown


def test_artifact_decision_can_select_full_artifact_when_runtime_and_size_are_ok():
    inventory = _inventory(full_safetensor_size=10 * 1024**3)

    decision = build_artifact_decision(
        inventory=inventory,
        transformers_version="4.future",
        transformers_supports_qwen3_next=True,
        llama_cpp_available=True,
    )

    assert decision["selected_immediate_primary_artifact"]["format"] == "safetensors_transformers"
    assert decision["blockers"] == []


def test_artifact_decision_moves_to_memory_gate_when_transformers_support_exists():
    decision = build_artifact_decision(
        inventory=_inventory(),
        transformers_version="5.dev",
        transformers_supports_qwen3_next=True,
        llama_cpp_available=False,
    )

    assert decision["selected_immediate_primary_artifact"] is None
    assert decision["blockers"] == [
        "full safetensors artifact is too large for a direct 24GB single-GPU run",
        "llama.cpp Python bindings are not installed for local GGUF execution",
    ]
    assert "Choose between a full-weights Transformers/offload path" in decision["next_engineering_gate"]


def _inventory(full_safetensor_size: int = 162_000_000_000):
    return {
        "full": {
            "sha": "full-sha",
            "total_bytes": full_safetensor_size + 1000,
            "files": [
                {"name": "model-00001-of-00001.safetensors", "size": full_safetensor_size},
                {"name": "config.json", "size": 1000},
            ],
        },
        "gguf": {
            "sha": "gguf-sha",
            "total_bytes": 48_410_988_384,
            "files": [
                {"name": GGUF_Q4_FILE, "size": 48_410_988_384},
                {"name": "README.md", "size": 1000},
            ],
        },
    }
