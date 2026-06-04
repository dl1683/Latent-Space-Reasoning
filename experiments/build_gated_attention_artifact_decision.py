"""Build the Qwen3-Next artifact decision for the gated-attention probe."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import transformers
from huggingface_hub import HfApi
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

DEFAULT_JSON_OUTPUT = Path("eval_results/gated_attention/gated_attention_artifact_decision.json")
DEFAULT_REPORT_OUTPUT = Path("GATED_ATTENTION_ARTIFACT_DECISION.md")

FULL_REPO = "Qwen/Qwen3-Next-80B-A3B-Instruct"
GGUF_REPO = "Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF"
GGUF_Q4_FILE = "Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory = fetch_live_inventory()
    decision = build_artifact_decision(
        inventory=inventory,
        transformers_version=transformers.__version__,
        transformers_supports_qwen3_next="qwen3_next" in CONFIG_MAPPING,
        llama_cpp_available=importlib.util.find_spec("llama_cpp") is not None,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(decision), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
                "selected_immediate_primary_artifact": decision["selected_immediate_primary_artifact"],
                "next_engineering_gate": decision["next_engineering_gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def fetch_live_inventory() -> dict[str, object]:
    api = HfApi()
    full = api.model_info(FULL_REPO, files_metadata=True)
    gguf = api.model_info(GGUF_REPO, files_metadata=True)
    return {
        "full": _repo_summary(full),
        "gguf": _repo_summary(gguf),
    }


def build_artifact_decision(
    *,
    inventory: dict[str, object],
    transformers_version: str,
    transformers_supports_qwen3_next: bool,
    llama_cpp_available: bool,
) -> dict[str, object]:
    full = dict(inventory["full"])
    gguf = dict(inventory["gguf"])
    gguf_q4 = _file_record(gguf, GGUF_Q4_FILE)
    full_safetensor_total = sum(
        int(file["size"]) for file in full["files"] if str(file["name"]).endswith(".safetensors")
    )
    full_safetensor_shards = sum(
        1 for file in full["files"] if str(file["name"]).endswith(".safetensors")
    )

    blockers = []
    if not transformers_supports_qwen3_next:
        blockers.append("local Transformers does not recognize model_type=qwen3_next")
    if full_safetensor_total > 120 * 1024**3:
        blockers.append("full safetensors artifact is too large for a direct 24GB single-GPU run")
    if not llama_cpp_available:
        blockers.append("llama.cpp Python bindings are not installed for local GGUF execution")

    selected_immediate_primary_artifact = None
    if transformers_supports_qwen3_next and full_safetensor_total <= 120 * 1024**3:
        selected_immediate_primary_artifact = {
            "repo": FULL_REPO,
            "sha": full["sha"],
            "format": "safetensors_transformers",
        }
    if transformers_supports_qwen3_next:
        next_engineering_gate = (
            "Choose between a full-weights Transformers/offload path for the frozen soft-prefix claim "
            "and a separate GGUF/server adapter for a non-soft-prefix serving smoke; do not start the "
            "full download until the memory/offload plan is explicit."
        )
    else:
        next_engineering_gate = (
            "Upgrade/install a qwen3_next-capable Transformers runtime for the soft-prefix runner, "
            "or build a separate GGUF/server adapter and explicitly mark it as non-soft-prefix."
        )

    return {
        "schema": "gated_attention_artifact_decision.v1",
        "generated_by": "experiments/build_gated_attention_artifact_decision.py",
        "full_repo": {
            "repo": FULL_REPO,
            "sha": full["sha"],
            "total_bytes": full["total_bytes"],
            "safetensor_bytes": full_safetensor_total,
            "safetensor_shards": full_safetensor_shards,
            "source_url": f"https://huggingface.co/{FULL_REPO}",
        },
        "gguf_repo": {
            "repo": GGUF_REPO,
            "sha": gguf["sha"],
            "total_bytes": gguf["total_bytes"],
            "q4_k_m_file": gguf_q4,
            "source_url": f"https://huggingface.co/{GGUF_REPO}",
        },
        "local_runtime": {
            "transformers_version": transformers_version,
            "transformers_supports_qwen3_next": transformers_supports_qwen3_next,
            "llama_cpp_available": llama_cpp_available,
            "gpu_vram_bytes": 24463 * 1024 * 1024,
        },
        "selected_immediate_primary_artifact": selected_immediate_primary_artifact,
        "selected_download_candidate": {
            "repo": GGUF_REPO,
            "sha": gguf["sha"],
            "file": gguf_q4["name"],
            "bytes": gguf_q4["size"],
            "purpose": "local serving/runtime feasibility smoke, not soft-prefix claim run",
        },
        "blockers": blockers,
        "next_engineering_gate": next_engineering_gate,
        "claim_boundary": [
            "Do not use the GGUF OpenAI-compatible path for the frozen soft-prefix claim unless it exposes inputs_embeds or an equivalent embedding-prefix hook.",
            "Do not download full safetensors until Transformers qwen3_next support is verified locally.",
            "Do not treat artifact selection as a gated-attention result.",
        ],
    }


def render_markdown(decision: dict[str, object]) -> str:
    full = decision["full_repo"]
    gguf = decision["gguf_repo"]
    runtime = decision["local_runtime"]
    q4 = gguf["q4_k_m_file"]
    lines = [
        "# Gated Attention Artifact Decision",
        "",
        "This file is generated by `experiments/build_gated_attention_artifact_decision.py`.",
        "",
        "## Decision",
        "",
        "No immediate primary Qwen3-Next soft-prefix run is selected for the current runner.",
        "",
        _decision_summary(runtime),
        "",
        "## Full Transformers Artifact",
        "",
        f"- Repo: `{full['repo']}`",
        f"- SHA: `{full['sha']}`",
        f"- Safetensor shards: `{full['safetensor_shards']}`",
        f"- Safetensor bytes: `{full['safetensor_bytes']}`",
        f"- Source: {full['source_url']}",
        "",
        "## GGUF Candidate",
        "",
        f"- Repo: `{gguf['repo']}`",
        f"- SHA: `{gguf['sha']}`",
        f"- Q4_K_M file: `{q4['name']}`",
        f"- Q4_K_M bytes: `{q4['size']}`",
        f"- Source: {gguf['source_url']}",
        "",
        "## Local Runtime",
        "",
        f"- Transformers version: `{runtime['transformers_version']}`",
        f"- Transformers supports `qwen3_next`: `{runtime['transformers_supports_qwen3_next']}`",
        f"- `llama_cpp` available: `{runtime['llama_cpp_available']}`",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- {blocker}" for blocker in decision["blockers"])
    lines.extend(["", "## Next Engineering Gate", "", decision["next_engineering_gate"], ""])
    lines.extend(["## Claim Boundary", ""])
    lines.extend(f"- {item}" for item in decision["claim_boundary"])
    lines.append("")
    return "\n".join(lines)


def _repo_summary(info) -> dict[str, object]:
    files = [
        {"name": sibling.rfilename, "size": int(getattr(sibling, "size", 0) or 0)}
        for sibling in info.siblings
    ]
    return {
        "sha": info.sha,
        "total_bytes": sum(file["size"] for file in files),
        "files": files,
    }


def _file_record(repo: dict[str, object], filename: str) -> dict[str, object]:
    matches = [file for file in repo["files"] if file["name"] == filename]
    if not matches:
        raise ValueError(f"missing required file in repo inventory: {filename}")
    return matches[0]


def _decision_summary(runtime: dict[str, object]) -> str:
    if runtime["transformers_supports_qwen3_next"]:
        return (
            "The full Transformers artifact is now architecture-compatible with the current "
            "`inputs_embeds` soft-prefix runner, but the full safetensors payload is larger "
            "than a direct 24GB single-GPU run. The GGUF Q4_K_M artifact remains the best "
            "download candidate for a local serving smoke, but it does not by itself satisfy "
            "the frozen soft-prefix claim surface."
        )
    return (
        "The full Transformers artifact is the only path that matches the current "
        "`inputs_embeds` soft-prefix runner, but the local Transformers install does not "
        "recognize `qwen3_next` yet and the full safetensors payload is larger than a direct "
        "24GB single-GPU run. The GGUF Q4_K_M artifact is the best download candidate for a "
        "local serving smoke, but it does not by itself satisfy the frozen soft-prefix claim surface."
    )


if __name__ == "__main__":
    raise SystemExit(main())
