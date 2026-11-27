"""Re-tokenize archived planner traces with the current compact delivery frame.

This is a deterministic prompt-cost reconstruction. It does not load model
weights, generate responses, open calibration gold, or claim that the original
action traces would recur under the changed prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

from experiments.memory.task_episode import _compact_delivery_system, _receipt_handles
from experiments.memory.verified_reader_bridge import encode_messages, frame_evidence


MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
HF_CACHE_ROOT = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
MODEL_PATH = Path(os.environ.get(
    "LSR_QWEN3_0_6B_SNAPSHOT",
    HF_CACHE_ROOT / "hub" / "models--Qwen--Qwen3-0.6B" / "snapshots" / MODEL_REVISION,
))
RUNS = {
    "02": "8d545eec1e3dd9095b0c4c9a76453ecc15e2161f1bae5ef4f67c6e304324bfad",
    "03": "80f39bc734c75a1c24c11aeafb4bcadebab6a405fc588130cfd565d16640910f",
    "04": "436fef04319a7719dcaebaad460f8ac5bab957a9be39592e913e76cc8b63d566",
    "05": "40485ca02a3de311d72c9ea121285e653026470731db642aa37a6a279bb95928",
    "06": "effe82ac4a9f7c61633589c5efabb593bb8ea5425496e905cc482017155ad7c2",
}
INPUT_BUDGET = 8192
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = REPOSITORY_ROOT / "experiments" / "results" / "task_episode_calibration_2026-11-26"
DEFAULT_OUTPUT = ARCHIVE_ROOT / "compact_delivery_counterfactual_20261126" / "summary.json"


class NoThinkingTokenizer:
    def __init__(self, inner):
        self.inner = inner

    def apply_chat_template(self, messages, **kwargs):
        kwargs["enable_thinking"] = False
        return self.inner.apply_chat_template(messages, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf8")


def _counterfactual_frame(old_frame, phase):
    """Rebuild the current run_episode compact-delivery payload in field order."""
    frame = {
        "task": old_frame["task"],
        "state": "",
        "receipts": _receipt_handles(old_frame.get("receipts", [])),
        "probes_enabled": old_frame["probes_enabled"],
        "last_rejection": old_frame.get("last_rejection"),
        "need_registry": old_frame.get("need_registry", []),
        "planning_phase": phase,
    }
    if old_frame.get("as_of") is not None:
        frame["as_of"] = old_frame["as_of"]
    if old_frame.get("joint_roles_enabled") is not None:
        frame["joint_roles_enabled"] = old_frame["joint_roles_enabled"]
    if old_frame.get("joint_state_enabled"):
        frame["joint_state_enabled"] = True
    return frame


def _phase_for_request(index):
    return "declare" if index == 0 else "act_after_plan" if index == 1 else "investigate"


def analyze(tokenizer):
    from transformers import __version__ as transformers_version

    details = []
    for run, expected_digest in RUNS.items():
        relative_path = Path("experiments/results/task_episode_calibration_2026-11-26") / (
            f"compact_task_planner_calibration_20261126_{run}/counterfactual-alternative.jsonl"
        )
        path = REPOSITORY_ROOT / relative_path
        raw = path.read_bytes()
        digest = _sha256(raw)
        if digest != expected_digest:
            raise ValueError(f"archived run {run} digest changed: {digest}")
        rows = [json.loads(line) for line in raw.decode("utf8").splitlines()]
        requests = [row["data"] for row in rows if row["kind"] == "request"]
        if not requests:
            raise ValueError(f"archived run {run} has no model requests")

        original_counts, rebuilt_counts, phase_hashes = [], [], {}
        for index, request in enumerate(requests):
            message_rows = request["messages"]
            original_messages = [{"role": role, "content": content} for role, content in message_rows]
            original_ids = encode_messages(tokenizer, original_messages)
            if original_ids != request["input_ids"]:
                raise ValueError(f"archived run {run} request {index} no longer matches its pinned tokenizer")
            original_counts.append(len(original_ids))

            if len(message_rows) != 2 or message_rows[1][0] != "user":
                raise ValueError(f"archived run {run} request {index} has an unexpected frame")
            old_frame = json.loads(message_rows[1][1])
            phase = _phase_for_request(index)
            frame = _counterfactual_frame(old_frame, phase)
            system = _compact_delivery_system(phase)
            if phase not in ("declare", "act_after_plan"):
                # Match the current task_episode system suffix for these traces:
                # batching is enabled, and no joint/recurrent tools are present.
                if old_frame.get("batch_reads_enabled"):
                    system += " Pass a list of candidate numbers to read to open those candidates together."
                system += " Every candidate in a read list counts against the read allowance."
            phase_hashes.setdefault(phase, _sha256(system.encode("utf8")))
            messages = frame_evidence(system, frame)
            rebuilt_counts.append(len(encode_messages(tokenizer, messages)))

        original_total = sum(original_counts)
        rebuilt_total = sum(rebuilt_counts)
        details.append({
            "run": run,
            "source_path": relative_path.as_posix(),
            "source_sha256": digest,
            "calls": len(requests),
            "recorded_input_tokens_by_call": original_counts,
            "recorded_input_tokens_total": original_total,
            "compact_delivery_reconstructed_tokens_by_call": rebuilt_counts,
            "compact_delivery_reconstructed_tokens_total": rebuilt_total,
            "tokens_saved_on_reconstruction": original_total - rebuilt_total,
            "input_budget_margin_after_reconstructed_calls": INPUT_BUDGET - rebuilt_total,
            "compact_delivery_system_sha256_by_phase": phase_hashes,
        })

    return {
        "schema": "lsr-compact-delivery-counterfactual-v1",
        "experiment": "re-tokenize saved planner request frames with current compact_delivery formatting",
        "model_revision": MODEL_REVISION,
        "tokenizer_path": str(MODEL_PATH),
        "tokenizer_chat_template_sha256": _sha256(tokenizer.chat_template.encode("utf8")),
        "transformers_version": transformers_version,
        "python_version": platform.python_version(),
        "input_budget_per_episode": INPUT_BUDGET,
        "generation_performed": False,
        "model_weights_loaded": False,
        "gold_opened": False,
        "source_traces": details,
        "aggregate": {
            "episodes": len(details),
            "calls": sum(row["calls"] for row in details),
            "recorded_input_tokens": sum(row["recorded_input_tokens_total"] for row in details),
            "reconstructed_input_tokens": sum(
                row["compact_delivery_reconstructed_tokens_total"] for row in details
            ),
            "tokens_saved_on_reconstruction": sum(
                row["tokens_saved_on_reconstruction"] for row in details
            ),
            "minimum_episode_margin": min(
                row["input_budget_margin_after_reconstructed_calls"] for row in details
            ),
        },
        "interpretation": (
            "The current framing would have reduced input use on these exact recorded traces. "
            "This is not a replay of model behavior: changed prompts can change actions, receipts, "
            "need registries, call counts, and answer quality. It does not qualify compact delivery "
            "or a task planner."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"pinned local tokenizer snapshot is unavailable: {MODEL_PATH}")

    from transformers import AutoTokenizer

    base = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=False)
    tokenizer = NoThinkingTokenizer(base)
    record = analyze(tokenizer)
    output = args.output if args.output.is_absolute() else REPOSITORY_ROOT / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + ".pending")
    if pending.exists():
        raise FileExistsError(f"refusing to overwrite pending diagnostic artifact: {pending}")
    pending.write_bytes(_canonical(record))
    pending.replace(output)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
