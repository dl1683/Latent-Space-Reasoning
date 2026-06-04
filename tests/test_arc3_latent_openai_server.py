"""CPU-only tests for the ARC-AGI-3 local OpenAI-compatible bridge."""

import json
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import latent_reasoning

from experiments.compute_guard import enforce_gpu_guard
from experiments.arc3_latent_openai_server import (
    FirstLegalOpenAIServer,
    FrontierProbeOpenAIServer,
    GraphProbeOpenAIServer,
    LatentOpenAIServer,
    LearnedVisualOpenAIServer,
    ScriptedPlanOpenAIServer,
    StateProbeOpenAIServer,
    _compact_arc3_transcript,
    _first_legal_action,
    _levels_completed,
    _grid_signature,
    _grid_delta_summary,
    _extract_available_actions,
    _extract_grid_blocks,
    _extract_grid_rows,
    _extract_visual_state,
    _legal_action_names,
    _message_text,
    _normalize_action_output,
)


def test_message_text_formats_chat_transcript():
    messages = [
        {"role": "system", "content": "Choose one action."},
        {"role": "user", "content": "Available actions:\n- UP\n- DOWN"},
    ]

    text = _message_text(messages)

    assert "SYSTEM:\nChoose one action." in text
    assert "USER:\nAvailable actions:" in text


def test_levels_completed_extracts_latest_value():
    transcript = "Levels completed: 0\nFrame 0\n\nLevels completed: 2\nFrame 1"

    assert _levels_completed(transcript) == 2


def test_extract_available_actions_from_latest_frame_text():
    transcript = "Frame 0:\n  [0, 1]\nAvailable actions:\n- RESET\n- MOVE x y\n- LEFT\n\nNext:"

    assert _extract_available_actions(transcript) == [
        {"name": "RESET", "is_complex": False},
        {"name": "MOVE", "is_complex": True},
        {"name": "LEFT", "is_complex": False},
    ]


def test_normalize_action_output_returns_last_mentioned_legal_action():
    transcript = "Available actions:\n- UP\n- DOWN\n- LEFT\n- RIGHT"
    output = "I first considered LEFT, but the final move is RIGHT"

    assert _normalize_action_output(output, transcript) == "RIGHT"


def test_normalize_action_output_preserves_complex_action_coordinates():
    transcript = "Available actions:\n- RESET\n- MOVE x y\n- RIGHT"
    output = "The target square is useful.\nMOVE 12 34"

    assert _normalize_action_output(output, transcript) == "MOVE 12 34"


def test_normalize_action_output_accepts_official_complex_syntax_variants():
    transcript = "Available actions:\n- RESET\n- MOVE x y  (where x and y are integers 0-63)\n- RIGHT"

    assert _normalize_action_output("MOVE: 12, 34", transcript) == "MOVE 12 34"
    assert _normalize_action_output("Final action: MOVE(5, 6)", transcript) == "MOVE 5 6"


def test_normalize_action_output_rejects_out_of_range_complex_coordinates():
    transcript = "Available actions:\n- RESET\n- MOVE x y\n- RIGHT"

    assert _normalize_action_output("MOVE 64 2", transcript) == "RIGHT"


def test_normalize_action_output_falls_back_to_non_reset_action():
    transcript = "Available actions:\n- RESET\n- LEFT\n- RIGHT"
    output = "I am unsure."

    assert _normalize_action_output(output, transcript) == "LEFT"


def test_normalize_action_output_completes_bare_complex_action():
    transcript = "\n".join(
        [
            "Frame:",
            "  [0, 0, 0, 0]",
            "  [0, 9, 9, 0]",
            "  [0, 9, 9, 0]",
            "  [0, 0, 0, 0]",
            "Available actions:",
            "- RESET",
            "- MOVE x y",
        ]
    )

    assert _normalize_action_output("MOVE", transcript) == "MOVE 2 2"


def test_first_legal_action_prefers_simple_non_reset_action():
    transcript = "Available actions:\n- RESET\n- MOVE x y\n- RIGHT"

    assert _first_legal_action(transcript) == "RIGHT"


def test_first_legal_action_completes_complex_action_before_reset():
    transcript = "\n".join(
        [
            "Frame:",
            "  [0, 0, 0, 0]",
            "  [0, 9, 9, 0]",
            "  [0, 9, 9, 0]",
            "  [0, 0, 0, 0]",
            "Available actions:",
            "- RESET",
            "- MOVE x y",
        ]
    )

    assert _first_legal_action(transcript) == "MOVE 2 2"


def test_legal_action_names_prefers_simple_non_reset_actions():
    transcript = "Available actions:\n- RESET\n- MOVE x y\n- RIGHT\n- LEFT"

    assert _legal_action_names(transcript) == ["RIGHT", "LEFT"]


def test_compact_arc3_transcript_summarizes_grid_and_keeps_actions():
    transcript = "\n".join(
        [
            "Frame 0:",
            "  [4, 4, 4, 4, 4, 4]",
            "  [4, 4, 9, 9, 4, 4]",
            "  [4, 4, 9, 5, 4, 4]",
            "Available actions:",
            "- RESET",
            "- ACTION1",
        ]
    )

    compact = _compact_arc3_transcript(transcript)

    assert "grid_size: 3x6" in compact
    assert "background_color: 4" in compact
    assert "non_background_bbox: y=1..2, x=2..3" in compact
    assert "y1: 9x2" in compact
    assert "Available actions:" in compact
    assert "- ACTION1" in compact


def test_compact_arc3_transcript_includes_recent_deltas():
    transcript = "\n".join(
        [
            "Frame 0:",
            "  [4, 4]",
            "  [4, 9]",
            "Action taken: ACTION1",
            "Frame 1:",
            "  [4, 3]",
            "  [4, 9]",
            "Available actions:",
            "- ACTION1",
        ]
    )

    compact = _compact_arc3_transcript(transcript)

    assert "recent_frame_changes:" in compact
    assert "changed_cells=1" in compact


def test_extract_grid_rows_and_signature_from_transcript():
    transcript = "Frame 0:\n  [4, 4]\n  [4, 9]\nAvailable actions:\n- ACTION1"

    assert _extract_grid_rows(transcript) == [[4, 4], [4, 9]]
    assert _grid_signature(transcript) == "[[4,4],[4,9]]"


def test_extract_visual_state_summarizes_latest_grid():
    transcript = "Levels completed: 2\nFrame 0:\n  [4, 4, 4]\n  [4, 9, 5]\nAvailable actions:\n- ACTION1"

    state = _extract_visual_state(transcript)

    assert state["levels_completed"] == 2
    assert state["grid_height"] == 2
    assert state["grid_width"] == 3
    assert state["bbox_y0"] == 1
    assert state["bbox_x1"] == 2
    assert state["foreground_counts"] == {5: 1, 9: 1}
    assert state["foreground_components"][0]["size"] == 2


def test_extract_visual_state_tracks_delta_components():
    transcript = "\n".join(
        [
            "Frame 0:",
            "  [4, 4, 4, 4]",
            "  [4, 9, 9, 4]",
            "  [4, 4, 4, 4]",
            "Action taken: ACTION1",
            "Frame 1:",
            "  [4, 4, 4, 4]",
            "  [4, 4, 9, 9]",
            "  [4, 4, 4, 4]",
            "Available actions:",
            "- ACTION1",
        ]
    )

    state = _extract_visual_state(transcript)

    assert state["delta_cells"] == 2
    assert len(state["delta_components"]) == 2
    assert {component["x0"] for component in state["delta_components"]} == {1, 3}
    assert all(component["size"] == 1 for component in state["delta_components"])


def test_extract_grid_rows_uses_latest_frame_only():
    transcript = "\n".join(
        [
            "Frame 0:",
            "  [1, 1]",
            "  [1, 2]",
            "Action taken: ACTION1",
            "Frame 1:",
            "  [4, 4]",
            "  [4, 9]",
            "Available actions:",
            "- ACTION1",
        ]
    )

    assert _extract_grid_rows(transcript) == [[4, 4], [4, 9]]


def test_extract_grid_blocks_preserves_recent_frame_history():
    transcript = "\n".join(
        [
            "Frame 0:",
            "  [1, 1]",
            "Action taken: ACTION1",
            "Frame 1:",
            "  [1, 2]",
            "Available actions:",
            "- ACTION1",
        ]
    )

    assert _extract_grid_blocks(transcript) == [[[1, 1]], [[1, 2]]]


def test_grid_delta_summary_counts_changed_cells():
    summary = _grid_delta_summary([[4, 4], [4, 9]], [[4, 3], [4, 9]])

    assert "changed_cells=1" in summary
    assert "bbox=y=0..0,x=1..1" in summary
    assert "4->3:1" in summary


def test_compact_arc3_transcript_reduces_large_sparse_grid():
    rows = []
    for y in range(64):
        row = [4] * 64
        if 20 <= y <= 24:
            row[20:24] = [9, 9, 5, 5]
        rows.append("  " + repr(row))
    transcript = "\n".join(["Frame 0:", *rows, "Available actions:", "- RESET", "- ACTION1"])

    compact = _compact_arc3_transcript(transcript)

    assert len(compact) < len(transcript) // 4
    assert "grid_size: 64x64" in compact
    assert "non_background_bbox: y=20..24, x=20..23" in compact


def test_server_writes_action_trace_jsonl(tmp_path, monkeypatch):
    class FakeEngine:
        def __init__(self, **kwargs):
            self.config = SimpleNamespace(
                synthesis=SimpleNamespace(),
                evolution=SimpleNamespace(),
            )

        def run(self, prompt):
            return SimpleNamespace(plan="I will move there.\nMOVE: 12, 34")

    args = Namespace(
        encoder="fake",
        model_name="local-latent-reasoning",
        decode_mode="geometry_feedback",
        reasoning_mode="hybrid",
        max_tokens=128,
        chains=1,
        generations=1,
        geometry_feedback_target_forward_kl=0.06,
        geometry_feedback_steering_eta=0.05,
        geometry_feedback_controller="pid",
        max_latent_calls=-1,
        fallback_policy="state_probe",
        trace_jsonl=str(tmp_path / "trace.jsonl"),
    )
    monkeypatch.setattr(latent_reasoning, "Engine", FakeEngine)
    monkeypatch.setattr("experiments.arc3_latent_openai_server.enforce_gpu_guard", lambda args: None)

    server = LatentOpenAIServer(args)
    response = server.complete(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Available actions:\n- RESET\n- MOVE x y  (where x and y are integers 0-63)\n- RIGHT",
                }
            ]
        }
    )

    trace = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8"))
    assert response["choices"][0]["message"]["content"] == "MOVE 12 34"
    assert trace["normalized_action"] == "MOVE 12 34"
    assert trace["available_actions"][1] == {"name": "MOVE", "is_complex": True}
    assert trace["compact_transcript_chars"] <= trace["raw_transcript_chars"]


def test_server_returns_legal_fallback_on_cuda_oom(tmp_path, monkeypatch):
    class OomEngine:
        def __init__(self, **kwargs):
            self.config = SimpleNamespace(
                synthesis=SimpleNamespace(),
                evolution=SimpleNamespace(),
            )

        def run(self, prompt):
            raise RuntimeError("CUDA error: out of memory")

    args = Namespace(
        encoder="fake",
        model_name="local-latent-reasoning",
        decode_mode="geometry_feedback",
        reasoning_mode="hybrid",
        max_tokens=128,
        chains=1,
        generations=1,
        geometry_feedback_target_forward_kl=0.06,
        geometry_feedback_steering_eta=0.05,
        geometry_feedback_controller="pid",
        max_latent_calls=-1,
        fallback_policy="state_probe",
        trace_jsonl=str(tmp_path / "trace.jsonl"),
    )
    monkeypatch.setattr(latent_reasoning, "Engine", OomEngine)
    monkeypatch.setattr("experiments.arc3_latent_openai_server.enforce_gpu_guard", lambda args: None)
    monkeypatch.setattr("experiments.arc3_latent_openai_server._clear_cuda_cache", lambda: None)

    server = LatentOpenAIServer(args)
    response = server.complete(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Available actions:\n- RESET\n- MOVE x y\n- RIGHT",
                }
            ]
        }
    )

    trace = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8"))
    assert response["choices"][0]["message"]["content"] == "RIGHT"
    assert trace["error"] == "cuda_out_of_memory"
    assert trace["normalized_action"] == "RIGHT"


def test_server_uses_fallback_after_max_latent_calls(tmp_path, monkeypatch):
    class CountingEngine:
        calls = 0

        def __init__(self, **kwargs):
            self.config = SimpleNamespace(
                synthesis=SimpleNamespace(),
                evolution=SimpleNamespace(),
            )

        def run(self, prompt):
            CountingEngine.calls += 1
            return SimpleNamespace(plan="ACTION2")

    args = Namespace(
        encoder="fake",
        model_name="local-latent-reasoning",
        decode_mode="geometry_feedback",
        reasoning_mode="hybrid",
        max_tokens=128,
        chains=1,
        generations=1,
        geometry_feedback_target_forward_kl=0.06,
        geometry_feedback_steering_eta=0.05,
        geometry_feedback_controller="pid",
        max_latent_calls=1,
        fallback_policy="round_robin",
        trace_jsonl=str(tmp_path / "trace.jsonl"),
    )
    monkeypatch.setattr(latent_reasoning, "Engine", CountingEngine)
    monkeypatch.setattr("experiments.arc3_latent_openai_server.enforce_gpu_guard", lambda args: None)
    monkeypatch.setattr("experiments.arc3_latent_openai_server._clear_cuda_cache", lambda: None)

    server = LatentOpenAIServer(args)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Available actions:\n- RESET\n- ACTION1\n- ACTION2",
            }
        ]
    }

    first = server.complete(payload)
    second = server.complete(payload)
    third = server.complete(payload)

    traces = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert first["choices"][0]["message"]["content"] == "ACTION2"
    assert second["choices"][0]["message"]["content"] == "ACTION1"
    assert third["choices"][0]["message"]["content"] == "ACTION2"
    assert CountingEngine.calls == 1
    assert traces[1]["fallback_reason"] == "max_latent_calls"
    assert traces[2]["fallback_reason"] == "max_latent_calls"


def test_server_uses_fallback_when_latent_output_mentions_no_legal_action(tmp_path, monkeypatch):
    class NoActionEngine:
        def __init__(self, **kwargs):
            self.config = SimpleNamespace(
                synthesis=SimpleNamespace(),
                evolution=SimpleNamespace(),
            )

        def run(self, prompt):
            return SimpleNamespace(plan="<think>I need to inspect the board first.")

    args = Namespace(
        encoder="fake",
        model_name="local-latent-reasoning",
        decode_mode="geometry_feedback",
        reasoning_mode="hybrid",
        max_tokens=128,
        chains=1,
        generations=1,
        geometry_feedback_target_forward_kl=0.06,
        geometry_feedback_steering_eta=0.05,
        geometry_feedback_controller="pid",
        max_latent_calls=-1,
        fallback_policy="round_robin",
        trace_jsonl=str(tmp_path / "trace.jsonl"),
    )
    monkeypatch.setattr(latent_reasoning, "Engine", NoActionEngine)
    monkeypatch.setattr("experiments.arc3_latent_openai_server.enforce_gpu_guard", lambda args: None)
    monkeypatch.setattr("experiments.arc3_latent_openai_server._clear_cuda_cache", lambda: None)

    server = LatentOpenAIServer(args)
    response = server.complete(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Available actions:\n- RESET\n- ACTION1\n- ACTION2",
                }
            ]
        }
    )

    trace = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8"))
    assert response["choices"][0]["message"]["content"] == "ACTION1"
    assert trace["fallback_reason"] == "no_legal_action_in_latent_output"


def test_latent_server_can_fallback_to_scripted_plan(tmp_path, monkeypatch):
    class NoActionEngine:
        def __init__(self, **kwargs):
            self.config = SimpleNamespace(
                synthesis=SimpleNamespace(),
                evolution=SimpleNamespace(),
            )

        def run(self, prompt):
            return SimpleNamespace(plan="<think>I need more time.")

    plan = tmp_path / "plans.json"
    plan.write_text(json.dumps({"1": ["ACTION4"]}), encoding="utf-8")
    args = Namespace(
        encoder="fake",
        model_name="local-latent-reasoning",
        decode_mode="geometry_feedback",
        reasoning_mode="hybrid",
        max_tokens=128,
        chains=1,
        generations=1,
        geometry_feedback_target_forward_kl=0.06,
        geometry_feedback_steering_eta=0.05,
        geometry_feedback_controller="pid",
        max_latent_calls=-1,
        scripted_plan=str(plan),
        fallback_policy="scripted_plan",
        state_probe_repeat_cap=8,
        trace_jsonl=str(tmp_path / "trace.jsonl"),
    )
    monkeypatch.setattr(latent_reasoning, "Engine", NoActionEngine)
    monkeypatch.setattr("experiments.arc3_latent_openai_server.enforce_gpu_guard", lambda args: None)
    monkeypatch.setattr("experiments.arc3_latent_openai_server._clear_cuda_cache", lambda: None)

    server = LatentOpenAIServer(args)
    response = server.complete(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Levels completed: 0\nAvailable actions:\n- ACTION1\n- ACTION4",
                }
            ]
        }
    )

    trace = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8"))
    assert response["choices"][0]["message"]["content"] == "ACTION4"
    assert trace["fallback_reason"] == "no_legal_action_in_latent_output"


def test_mechanistic_guard_overrides_wrong_legal_latent_action(tmp_path, monkeypatch):
    class WrongActionEngine:
        def __init__(self, **kwargs):
            self.config = SimpleNamespace(
                synthesis=SimpleNamespace(),
                evolution=SimpleNamespace(),
            )

        def run(self, prompt):
            return SimpleNamespace(plan="ACTION1")

    plan = tmp_path / "plans.json"
    plan.write_text(json.dumps({"1": ["ACTION4"]}), encoding="utf-8")
    args = Namespace(
        encoder="fake",
        model_name="local-latent-reasoning",
        decode_mode="geometry_feedback",
        reasoning_mode="hybrid",
        max_tokens=128,
        chains=1,
        generations=1,
        geometry_feedback_target_forward_kl=0.06,
        geometry_feedback_steering_eta=0.05,
        geometry_feedback_controller="pid",
        max_latent_calls=-1,
        scripted_plan=str(plan),
        fallback_policy="scripted_plan",
        mechanistic_guard="scripted_plan",
        state_probe_repeat_cap=8,
        trace_jsonl=str(tmp_path / "trace.jsonl"),
    )
    monkeypatch.setattr(latent_reasoning, "Engine", WrongActionEngine)
    monkeypatch.setattr("experiments.arc3_latent_openai_server.enforce_gpu_guard", lambda args: None)
    monkeypatch.setattr("experiments.arc3_latent_openai_server._clear_cuda_cache", lambda: None)

    server = LatentOpenAIServer(args)
    response = server.complete(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Levels completed: 0\nAvailable actions:\n- ACTION1\n- ACTION4",
                }
            ]
        }
    )

    trace = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8"))
    assert response["choices"][0]["message"]["content"] == "ACTION4"
    assert trace["fallback_reason"] == "mechanistic_guard_override"
    assert trace["latent_action"] == "ACTION1"
    assert trace["mechanistic_action"] == "ACTION4"


def test_state_probe_backend_explores_untried_actions_per_state(tmp_path):
    args = Namespace(
        model_name="local-latent-reasoning",
        state_probe_repeat_cap=8,
        trace_jsonl=str(tmp_path / "trace.jsonl"),
    )
    server = StateProbeOpenAIServer(args)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Frame 0:\n  [4, 4]\n  [4, 9]\nAvailable actions:\n- RESET\n- ACTION1\n- ACTION2",
            }
        ]
    }

    first = server.complete(payload)
    second = server.complete(payload)

    traces = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert first["choices"][0]["message"]["content"] == "ACTION1"
    assert second["choices"][0]["message"]["content"] == "ACTION2"
    assert traces[0]["backend"] == "state_probe"
    assert traces[0]["policy_metadata"]["policy"] == "state_probe"


def test_state_probe_repeat_cap_forces_new_action(tmp_path):
    args = Namespace(
        model_name="local-latent-reasoning",
        state_probe_repeat_cap=2,
        trace_jsonl=str(tmp_path / "trace.jsonl"),
    )
    server = StateProbeOpenAIServer(args)

    responses = []
    for step in range(3):
        responses.append(
            server.complete(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"Frame {step}:\n"
                                f"  [4, {step}]\n"
                                "  [4, 9]\n"
                                "Available actions:\n"
                                "- RESET\n"
                                "- ACTION1\n"
                                "- ACTION2"
                            ),
                        }
                    ]
                }
            )["choices"][0]["message"]["content"]
        )

    assert responses == ["ACTION1", "ACTION1", "ACTION2"]


def test_state_probe_repeat_cap_reaches_all_actions(tmp_path):
    args = Namespace(
        model_name="local-latent-reasoning",
        state_probe_repeat_cap=1,
        trace_jsonl=str(tmp_path / "trace.jsonl"),
    )
    server = StateProbeOpenAIServer(args)

    responses = []
    for step in range(4):
        responses.append(
            server.complete(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"Frame {step}:\n"
                                f"  [4, {step}]\n"
                                "  [4, 9]\n"
                                "Available actions:\n"
                                "- RESET\n"
                                "- ACTION1\n"
                                "- ACTION2\n"
                                "- ACTION3\n"
                                "- ACTION4"
                            ),
                        }
                    ]
                }
            )["choices"][0]["message"]["content"]
        )

    assert responses == ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]


def test_frontier_probe_tries_unseen_actions_per_state(tmp_path):
    args = Namespace(
        model_name="local-latent-reasoning",
        state_probe_repeat_cap=2,
        trace_jsonl=str(tmp_path / "trace.jsonl"),
    )
    server = FrontierProbeOpenAIServer(args)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Frame 0:\n  [0, 1]\nAvailable actions:\n- ACTION1\n- ACTION2\n- ACTION3",
            }
        ]
    }

    responses = [server.complete(payload)["choices"][0]["message"]["content"] for _ in range(3)]
    events = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert set(responses) == {"ACTION1", "ACTION2", "ACTION3"}
    assert events[-1]["backend"] == "frontier_probe"
    assert events[-1]["policy_metadata"]["policy"] == "frontier_probe"


def test_graph_probe_records_transition_metadata(tmp_path):
    args = Namespace(
        model_name="local-latent-reasoning",
        state_probe_repeat_cap=2,
        trace_jsonl=str(tmp_path / "trace.jsonl"),
    )
    server = GraphProbeOpenAIServer(args)
    first = {
        "messages": [
            {
                "role": "user",
                "content": "Frame 0:\n  [0, 1]\nAvailable actions:\n- ACTION1\n- ACTION2",
            }
        ]
    }
    second = {
        "messages": [
            {
                "role": "user",
                "content": "Frame 1:\n  [1, 0]\nAvailable actions:\n- ACTION1\n- ACTION2",
            }
        ]
    }

    assert server.complete(first)["choices"][0]["message"]["content"] == "ACTION1"
    assert server.complete(second)["choices"][0]["message"]["content"] == "ACTION2"
    events = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert events[-1]["backend"] == "graph_probe"
    assert events[-1]["policy_metadata"]["known_edges"] == 1


def test_state_probe_fallback_tracks_changed_observations(tmp_path, monkeypatch):
    class OomEngine:
        def __init__(self, **kwargs):
            self.config = SimpleNamespace(
                synthesis=SimpleNamespace(),
                evolution=SimpleNamespace(),
            )

        def run(self, prompt):
            raise RuntimeError("CUDA error: out of memory")

    args = Namespace(
        encoder="fake",
        model_name="local-latent-reasoning",
        decode_mode="geometry_feedback",
        reasoning_mode="hybrid",
        max_tokens=128,
        chains=1,
        generations=1,
        geometry_feedback_target_forward_kl=0.06,
        geometry_feedback_steering_eta=0.05,
        geometry_feedback_controller="pid",
        max_latent_calls=-1,
        fallback_policy="state_probe",
        trace_jsonl=str(tmp_path / "trace.jsonl"),
    )
    monkeypatch.setattr(latent_reasoning, "Engine", OomEngine)
    monkeypatch.setattr("experiments.arc3_latent_openai_server.enforce_gpu_guard", lambda args: None)
    monkeypatch.setattr("experiments.arc3_latent_openai_server._clear_cuda_cache", lambda: None)

    server = LatentOpenAIServer(args)
    first = server.complete(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Frame 0:\n  [4, 4]\n  [4, 9]\nAvailable actions:\n- RESET\n- ACTION1\n- ACTION2",
                }
            ]
        }
    )
    second = server.complete(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Frame 1:\n  [4, 9]\n  [4, 4]\nAvailable actions:\n- RESET\n- ACTION1\n- ACTION2",
                }
            ]
        }
    )

    assert first["choices"][0]["message"]["content"] == "ACTION1"
    assert second["choices"][0]["message"]["content"] == "ACTION1"


def test_scripted_plan_backend_steps_plan_then_falls_back(tmp_path):
    plan = tmp_path / "plans.json"
    plan.write_text(json.dumps({"1": ["ACTION4", "ACTION4"]}), encoding="utf-8")
    args = Namespace(
        model_name="local-latent-reasoning",
        scripted_plan=str(plan),
        state_probe_repeat_cap=8,
        trace_jsonl=str(tmp_path / "trace.jsonl"),
    )
    server = ScriptedPlanOpenAIServer(args)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Levels completed: 0\nFrame 0:\n  [4]\nAvailable actions:\n- ACTION1\n- ACTION4",
            }
        ]
    }

    first = server.complete(payload)
    second = server.complete(payload)
    third = server.complete(payload)

    traces = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert first["choices"][0]["message"]["content"] == "ACTION4"
    assert second["choices"][0]["message"]["content"] == "ACTION4"
    assert third["choices"][0]["message"]["content"] == "ACTION1"
    assert traces[2]["source"] == "state_probe_fallback"


def test_learned_visual_backend_chooses_by_nearest_state(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            [
                {
                    "trace": [
                        {
                            "action": "ACTION4",
                            "state_before": {
                                "levels_completed": 0,
                                "grid_height": 2,
                                "grid_width": 3,
                                "bbox_y0": 1,
                                "bbox_y1": 1,
                                "bbox_x0": 1,
                                "bbox_x1": 2,
                                "background": 4,
                                "foreground_counts": {5: 1, 9: 1},
                            },
                        },
                        {
                            "action": "ACTION1",
                            "state_before": {
                                "levels_completed": 0,
                                "grid_height": 2,
                                "grid_width": 3,
                                "bbox_y0": 0,
                                "bbox_y1": 0,
                                "bbox_x0": 0,
                                "bbox_x1": 1,
                                "background": 4,
                                "foreground_counts": {3: 2},
                            },
                        },
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    args = Namespace(
        model_name="local-latent-reasoning",
        learned_trace=str(trace),
        learned_policy_k=1,
        state_probe_repeat_cap=8,
        trace_jsonl=str(tmp_path / "server_trace.jsonl"),
    )
    server = LearnedVisualOpenAIServer(args)

    response = server.complete(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Levels completed: 0\nFrame 0:\n  [4, 4, 4]\n  [4, 9, 5]\nAvailable actions:\n- ACTION1\n- ACTION4",
                }
            ]
        }
    )

    event = json.loads((tmp_path / "server_trace.jsonl").read_text(encoding="utf-8"))
    assert response["choices"][0]["message"]["content"] == "ACTION4"
    assert event["backend"] == "learned_visual"
    assert event["policy_metadata"]["policy"] == "learned_visual"


def test_learned_visual_backend_can_filter_training_levels(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            [
                {
                    "trace": [
                        {
                            "action": "ACTION4",
                            "state_before": {
                                "levels_completed": 0,
                                "grid_height": 1,
                                "grid_width": 2,
                                "bbox_y0": 0,
                                "bbox_y1": 0,
                                "bbox_x0": 1,
                                "bbox_x1": 1,
                                "background": 4,
                                "foreground_counts": {9: 1},
                            },
                        },
                        {
                            "action": "ACTION1",
                            "state_before": {
                                "levels_completed": 1,
                                "grid_height": 1,
                                "grid_width": 2,
                                "bbox_y0": 0,
                                "bbox_y1": 0,
                                "bbox_x0": 1,
                                "bbox_x1": 1,
                                "background": 4,
                                "foreground_counts": {9: 1},
                            },
                        },
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    args = Namespace(
        model_name="local-latent-reasoning",
        learned_trace=str(trace),
        learned_policy_k=1,
        learned_max_train_level=0,
        learned_sequence_backoff=False,
        learned_phase_switch=True,
        learned_goal_seek=False,
        state_probe_repeat_cap=8,
        trace_jsonl=str(tmp_path / "server_trace.jsonl"),
    )
    server = LearnedVisualOpenAIServer(args)

    response = server.complete(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Levels completed: 1\nFrame 0:\n  [4, 9]\nAvailable actions:\n- ACTION1\n- ACTION4",
                }
            ]
        }
    )

    event = json.loads((tmp_path / "server_trace.jsonl").read_text(encoding="utf-8"))
    assert response["choices"][0]["message"]["content"] == "ACTION4"
    assert event["policy_metadata"]["training_examples"] == 1
    assert event["policy_metadata"]["max_train_level"] == 0


def test_learned_visual_backend_uses_sequence_backoff_for_unseen_level(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            [
                {
                    "trace": [
                        {
                            "action": "ACTION1",
                            "state_before": {"levels_completed": 0, "grid_height": 1, "grid_width": 1},
                        },
                        {
                            "action": "ACTION2",
                            "state_before": {"levels_completed": 0, "grid_height": 1, "grid_width": 1},
                        },
                        {
                            "action": "ACTION3",
                            "state_before": {"levels_completed": 0, "grid_height": 1, "grid_width": 1},
                        },
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    args = Namespace(
        model_name="local-latent-reasoning",
        learned_trace=str(trace),
        learned_policy_k=1,
        learned_max_train_level=0,
        learned_sequence_backoff=True,
        state_probe_repeat_cap=8,
        trace_jsonl=str(tmp_path / "server_trace.jsonl"),
    )
    server = LearnedVisualOpenAIServer(args)

    response = server.complete(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Levels completed: 1\n"
                        "Action taken: ACTION1\n"
                        "Action taken: ACTION2\n"
                        "Frame 0:\n  [4]\n"
                        "Available actions:\n- ACTION1\n- ACTION2\n- ACTION3"
                    ),
                }
            ]
        }
    )

    event = json.loads((tmp_path / "server_trace.jsonl").read_text(encoding="utf-8"))
    assert response["choices"][0]["message"]["content"] == "ACTION3"
    assert event["policy_metadata"]["policy"] == "learned_visual_sequence_backoff"


def test_learned_visual_ood_backoff_does_not_repeat_one_action_forever(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            [
                {
                    "trace": [
                        {
                            "action": "ACTION1",
                            "state_before": {"levels_completed": 0, "grid_height": 1, "grid_width": 1},
                        },
                        {
                            "action": "ACTION3",
                            "state_before": {"levels_completed": 0, "grid_height": 1, "grid_width": 1},
                        },
                        {
                            "action": "ACTION3",
                            "state_before": {"levels_completed": 0, "grid_height": 1, "grid_width": 1},
                        },
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    args = Namespace(
        model_name="local-latent-reasoning",
        learned_trace=str(trace),
        learned_policy_k=1,
        learned_max_train_level=0,
        learned_sequence_backoff=True,
        state_probe_repeat_cap=2,
        trace_jsonl=str(tmp_path / "server_trace.jsonl"),
    )
    server = LearnedVisualOpenAIServer(args)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Levels completed: 1\n"
                    "Action taken: ACTION1\n"
                    "Frame 0:\n  [4]\n"
                    "Available actions:\n- ACTION2\n- ACTION3"
                ),
            }
        ]
    }

    first = server.complete(payload)
    second = server.complete(payload)
    third = server.complete(payload)

    assert first["choices"][0]["message"]["content"] == "ACTION3"
    assert second["choices"][0]["message"]["content"] == "ACTION3"
    assert third["choices"][0]["message"]["content"] == "ACTION2"


def test_learned_visual_ood_backoff_blocks_ineffective_last_action(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            [
                {
                    "trace": [
                        {
                            "action": "ACTION1",
                            "state_before": {"levels_completed": 0, "grid_height": 1, "grid_width": 1},
                        },
                        {
                            "action": "ACTION3",
                            "state_before": {"levels_completed": 0, "grid_height": 1, "grid_width": 1},
                        },
                        {
                            "action": "ACTION3",
                            "state_before": {"levels_completed": 0, "grid_height": 1, "grid_width": 1},
                        },
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    args = Namespace(
        model_name="local-latent-reasoning",
        learned_trace=str(trace),
        learned_policy_k=1,
        learned_max_train_level=0,
        learned_sequence_backoff=True,
        state_probe_repeat_cap=8,
        trace_jsonl=str(tmp_path / "server_trace.jsonl"),
    )
    server = LearnedVisualOpenAIServer(args)
    first_payload = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Levels completed: 1\n"
                    "Action taken: ACTION1\n"
                    "Frame 0:\n  [4]\n"
                    "Available actions:\n- ACTION2\n- ACTION3"
                ),
            }
        ]
    }
    ineffective_payload = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Levels completed: 1\n"
                    "Frame 0:\n  [4, 4]\n"
                    "Action taken: ACTION3\n"
                    "Frame 1:\n  [4, 9]\n"
                    "Available actions:\n- ACTION2\n- ACTION3"
                ),
            }
        ]
    }

    first = server.complete(first_payload)
    second = server.complete(ineffective_payload)
    events = [
        json.loads(line)
        for line in (tmp_path / "server_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert first["choices"][0]["message"]["content"] == "ACTION3"
    assert second["choices"][0]["message"]["content"] == "ACTION2"
    assert events[-1]["policy_metadata"]["ood_block_reason"] == "ineffective_last_action"


def test_learned_visual_phase_switches_axis_at_component_boundary(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            [
                {
                    "trace": [
                        {
                            "action": "ACTION3",
                            "state_before": {
                                "levels_completed": 0,
                                "grid_height": 64,
                                "grid_width": 64,
                                "foreground_components": [
                                    {"size": 100, "x0": 4, "x1": 30, "y0": 0, "y1": 36}
                                ],
                            },
                        }
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    args = Namespace(
        model_name="local-latent-reasoning",
        learned_trace=str(trace),
        learned_policy_k=1,
        learned_max_train_level=0,
        learned_sequence_backoff=False,
        learned_phase_switch=True,
        learned_goal_seek=False,
        state_probe_repeat_cap=8,
        trace_jsonl=str(tmp_path / "server_trace.jsonl"),
    )
    server = LearnedVisualOpenAIServer(args)

    response = server.complete(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Levels completed: 1\n"
                        "Frame 0:\n"
                        "  [5, 5, 5, 5, 5, 5]\n"
                        "  [5, 3, 3, 3, 5, 5]\n"
                        "Available actions:\n- ACTION1\n- ACTION2\n- ACTION3\n- ACTION4"
                    ),
                }
            ]
        }
    )

    event = json.loads((tmp_path / "server_trace.jsonl").read_text(encoding="utf-8"))
    assert response["choices"][0]["message"]["content"] == "ACTION2"
    assert event["policy_metadata"]["policy"] == "learned_visual_phase_switch"
    assert event["policy_metadata"]["phase_reason"] == "horizontal_boundary_to_vertical_phase"


def test_learned_visual_goal_seek_aligns_to_small_target_component(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            [
                {
                    "trace": [
                        {
                            "action": "ACTION3",
                            "state_before": {
                                "levels_completed": 0,
                                "grid_height": 64,
                                "grid_width": 64,
                                "foreground_components": [
                                    {"size": 800, "x0": 4, "x1": 30, "y0": 0, "y1": 36},
                                    {"size": 20, "x0": 3, "x1": 8, "y0": 55, "y1": 60},
                                ],
                            },
                        }
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    args = Namespace(
        model_name="local-latent-reasoning",
        learned_trace=str(trace),
        learned_policy_k=1,
        learned_max_train_level=0,
        learned_sequence_backoff=False,
        learned_phase_switch=False,
        learned_goal_seek=True,
        state_probe_repeat_cap=8,
        trace_jsonl=str(tmp_path / "server_trace.jsonl"),
    )
    server = LearnedVisualOpenAIServer(args)

    response = server.complete(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Levels completed: 1\n"
                        "Frame 0:\n"
                        "  [5, 5, 5, 5, 5, 5]\n"
                        "  [5, 3, 3, 3, 5, 5]\n"
                        "  [5, 5, 5, 5, 5, 5]\n"
                        "  [12, 12, 5, 5, 5, 5]\n"
                        "Available actions:\n- ACTION1\n- ACTION2\n- ACTION3\n- ACTION4"
                    ),
                }
            ]
        }
    )

    event = json.loads((tmp_path / "server_trace.jsonl").read_text(encoding="utf-8"))
    assert response["choices"][0]["message"]["content"] == "ACTION2"
    assert event["policy_metadata"]["policy"] == "learned_visual_goal_seek"
    assert event["policy_metadata"]["goal_reason"] == "align_y_to_target_component"


def test_first_legal_server_responds_without_engine_or_gpu(tmp_path):
    args = Namespace(
        model_name="local-latent-reasoning",
        trace_jsonl=str(tmp_path / "trace.jsonl"),
    )

    server = FirstLegalOpenAIServer(args)
    response = server.complete(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Available actions:\n- RESET\n- MOVE x y\n- RIGHT",
                }
            ]
        }
    )

    trace = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8"))
    assert response["choices"][0]["message"]["content"] == "RIGHT"
    assert trace["backend"] == "first_legal"


def test_gpu_guard_rejects_busy_gpu():
    args = Namespace(
        max_gpu_utilization=35.0,
        max_gpu_memory_used_mb=12000.0,
        wait_for_gpu=False,
    )

    with patch("experiments.compute_guard.query_gpu_load", return_value=(61.0, 5340.0)):
        with pytest.raises(RuntimeError, match="GPU utilization"):
            enforce_gpu_guard(args)


def test_gpu_guard_allows_disabled_limits():
    args = Namespace(
        max_gpu_utilization=-1.0,
        max_gpu_memory_used_mb=-1.0,
        wait_for_gpu=False,
    )

    with patch("experiments.compute_guard.query_gpu_load", return_value=(99.0, 24000.0)):
        enforce_gpu_guard(args)


def test_gpu_guard_waits_until_gpu_is_available():
    args = Namespace(
        max_gpu_utilization=35.0,
        max_gpu_memory_used_mb=12000.0,
        wait_for_gpu=True,
        gpu_wait_timeout_s=30.0,
        gpu_wait_poll_s=0.1,
    )

    with patch(
        "experiments.compute_guard.query_gpu_load",
        side_effect=[(80.0, 5000.0), (10.0, 5000.0)],
    ) as query:
        with patch("experiments.compute_guard.time.sleep") as sleep:
            enforce_gpu_guard(args)

    assert query.call_count == 2
    sleep.assert_called_once_with(0.1)


def test_gpu_guard_wait_mode_times_out():
    args = Namespace(
        max_gpu_utilization=35.0,
        max_gpu_memory_used_mb=12000.0,
        wait_for_gpu=True,
        gpu_wait_timeout_s=0.0,
        gpu_wait_poll_s=0.1,
    )

    with patch("experiments.compute_guard.query_gpu_load", return_value=(80.0, 5000.0)):
        with pytest.raises(RuntimeError, match="Timed out waiting"):
            enforce_gpu_guard(args)
