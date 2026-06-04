"""Build a WSL bootstrap preflight for the Qwen3-Next soft-prefix path."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

DEFAULT_JSON_OUTPUT = Path("eval_results/gated_attention/gated_attention_wsl_bootstrap.json")
DEFAULT_REPORT_OUTPUT = Path("GATED_ATTENTION_WSL_BOOTSTRAP.md")
DEFAULT_DISTRO = "Ubuntu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distro", default=DEFAULT_DISTRO)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_wsl_bootstrap_plan(distro=args.distro)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(render_markdown(plan), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_output": str(args.json_output),
                "report_output": str(args.report_output),
                "ready_for_wsl_runtime_bootstrap": plan["ready_for_wsl_runtime_bootstrap"],
                "blocking_reasons": plan["blocking_reasons"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_wsl_bootstrap_plan(*, distro: str = DEFAULT_DISTRO) -> dict[str, object]:
    wsl_path = shutil.which("wsl.exe") or shutil.which("wsl")
    checks = {
        "wsl_list": _run(["wsl.exe", "-l", "-v"]) if wsl_path else _missing("wsl.exe"),
        "linux_kernel": _wsl(distro, "uname -a") if wsl_path else _missing("wsl.exe"),
        "gpu_visible": _wsl(
            distro,
            "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu "
            "--format=csv,noheader,nounits",
        )
        if wsl_path
        else _missing("wsl.exe"),
        "python": _wsl(distro, "command -v python3 && python3 --version") if wsl_path else _missing("wsl.exe"),
        "pip": _wsl(distro, "python3 -m pip --version") if wsl_path else _missing("wsl.exe"),
        "ensurepip": _wsl(distro, "python3 -c 'import ensurepip; print(\"ensurepip ok\")'")
        if wsl_path
        else _missing("wsl.exe"),
        "sudo_noninteractive": _wsl(distro, "sudo -n true") if wsl_path else _missing("wsl.exe"),
        "triton_package_index": _wsl(distro, "python3 -m pip index versions triton")
        if wsl_path
        else _missing("wsl.exe"),
    }

    blocking_reasons = []
    if not wsl_path:
        blocking_reasons.append("wsl.exe is not available")
    if checks["gpu_visible"]["returncode"] != 0:
        blocking_reasons.append("WSL cannot see the NVIDIA GPU")
    if checks["pip"]["returncode"] != 0:
        blocking_reasons.append("WSL Python has no pip")
    if checks["ensurepip"]["returncode"] != 0:
        blocking_reasons.append("WSL Python lacks ensurepip/python3-venv")
    if checks["sudo_noninteractive"]["returncode"] != 0:
        blocking_reasons.append("sudo requires a password; cannot install python3.12-venv non-interactively")

    install_commands = [
        "sudo apt-get update",
        "sudo apt-get install -y python3.12-venv python3-pip",
        "python3 -m venv ~/.venvs/lsr-qwen-next",
        "source ~/.venvs/lsr-qwen-next/bin/activate",
        "python -m pip install --upgrade pip",
        "python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch",
        (
            "python -m pip install "
            "'transformers @ git+https://github.com/huggingface/transformers.git@032db9c8d6c3c3cb89e71cc414bfb5a469b1a6da' "
            "accelerate safetensors huggingface_hub bitsandbytes triton flash-linear-attention causal-conv1d"
        ),
    ]

    validation_command = "\n".join(
        [
            "source ~/.venvs/lsr-qwen-next/bin/activate",
            "python - <<'PY'",
            "from transformers import AutoConfig, AutoModelForCausalLM",
            "from accelerate import init_empty_weights",
            "cfg = AutoConfig.from_pretrained('Qwen/Qwen3-Next-80B-A3B-Instruct', trust_remote_code=True)",
            "print(cfg.model_type)",
            "with init_empty_weights():",
            "    AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)",
            "print('empty model ok')",
            "PY",
        ]
    )

    return {
        "schema": "gated_attention_wsl_bootstrap.v1",
        "generated_by": "experiments/build_gated_attention_wsl_bootstrap.py",
        "distro": distro,
        "wsl_path": wsl_path,
        "checks": checks,
        "ready_for_wsl_runtime_bootstrap": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "manual_install_commands": install_commands,
        "post_install_validation_command": validation_command,
        "claim_boundary": [
            "This bootstrap only prepares a Triton-capable runtime.",
            "It does not download Qwen3-Next weights.",
            "It does not produce a gated-attention result.",
        ],
    }


def render_markdown(plan: dict[str, object]) -> str:
    lines = [
        "# Gated Attention WSL Bootstrap",
        "",
        "This file is generated by `experiments/build_gated_attention_wsl_bootstrap.py`.",
        "",
        "## Status",
        "",
        f"- Distro: `{plan['distro']}`",
        f"- WSL path: `{plan['wsl_path']}`",
        f"- Ready for WSL runtime bootstrap: `{plan['ready_for_wsl_runtime_bootstrap']}`",
        "",
    ]
    if plan["blocking_reasons"]:
        lines.extend(["Blocking reasons:", ""])
        lines.extend(f"- {reason}" for reason in plan["blocking_reasons"])
        lines.append("")

    lines.extend(["## Checks", "", "| Check | Return | Output |", "|---|---:|---|"])
    for name, check in plan["checks"].items():
        output = _markdown_cell((check.get("stdout") or check.get("stderr") or "").strip())
        lines.append(f"| `{name}` | `{check['returncode']}` | {output} |")

    lines.extend(["", "## Manual Bootstrap Commands", "", "```bash"])
    lines.extend(plan["manual_install_commands"])
    lines.extend(["```", "", "## Post-Install Validation", "", "```bash", plan["post_install_validation_command"], "```", ""])
    lines.extend(["## Claim Boundary", ""])
    lines.extend(f"- {item}" for item in plan["claim_boundary"])
    lines.append("")
    return "\n".join(lines)


def _wsl(distro: str, command: str) -> dict[str, object]:
    return _run(["wsl.exe", "-d", distro, "--", "bash", "-lc", command])


def _run(cmd: list[str]) -> dict[str, object]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception as exc:  # pragma: no cover - environment specific
        return {"returncode": -1, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}", "command": cmd}
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "command": cmd}


def _missing(name: str) -> dict[str, object]:
    return {"returncode": -1, "stdout": "", "stderr": f"{name} not found", "command": [name]}


def _markdown_cell(text: str) -> str:
    compact = " ".join(text.replace("\x00", "").split())
    compact = compact.replace("|", "\\|")
    if not compact:
        return "`none`"
    if len(compact) > 220:
        compact = compact[:217] + "..."
    return f"`{compact}`"


if __name__ == "__main__":
    raise SystemExit(main())
