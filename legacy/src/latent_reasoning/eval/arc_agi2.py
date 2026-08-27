"""
ARC Evaluation for Latent Space Reasoning.

This module provides comprehensive evaluation of the Latent Space Reasoning Engine
against static ARC-style grid benchmarks.

Note: the current official ARC-AGI-3 benchmark is interactive and agentic, not a
static grid dataset. Use experiments/run_arc3_official_harness.py for the true
ARC-AGI-3 harness. This evaluator remains useful for ARC-AGI-2 and static proxy
experiments while interactive-agent integration is developed.

ARC-AGI Overview:
The Abstraction and Reasoning Corpus (ARC) consists of visual reasoning puzzles
where models must identify patterns in grid transformations and apply them to
new test cases. These tasks test core cognitive abilities like:
- Pattern recognition and abstraction
- Logical reasoning and rule inference
- Spatial and visual understanding
- Generalization from few examples

Evaluation Framework:
- Systematic Comparison: Side-by-side evaluation of baseline vs latent reasoning
- Detailed Metrics: Accuracy, parsing success, response quality analysis
- Performance Tracking: Historical comparison and improvement measurement
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import traceback

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

console = Console()
SUPPORTED_ARC_VERSIONS = ("2", "3")
SUPPORTED_ARC_STRATEGIES = (
    "single",
    "adaptive",
    "repair",
    "consensus",
    "geometry_bandit",
    "self_improving",
)


def _normalize_arc_version(arc_version: str) -> str:
    """Normalize ARC version input into a stable '2' or '3' string."""
    normalized = str(arc_version).strip().lower()
    normalized = normalized.replace("arc", "").replace("-", "").replace("_", "")
    if normalized.startswith("agi"):
        normalized = normalized[3:]
    if normalized.startswith("v"):
        normalized = normalized[1:]
    if normalized not in SUPPORTED_ARC_VERSIONS:
        raise ValueError(f"Unsupported ARC version '{arc_version}'. Supported: 2 or 3")
    return normalized


def _normalize_arc_strategy(strategy: str) -> str:
    """Normalize ARC-AGI strategy input."""
    normalized = str(strategy).strip().lower()
    if normalized not in SUPPORTED_ARC_STRATEGIES:
        raise ValueError(
            "Unsupported ARC strategy. Supported: single, adaptive, repair, consensus, geometry_bandit, self_improving"
        )
    return normalized


def _normalize_reasoning_mode(reasoning_mode: str) -> str:
    """Normalize reasoning mode input."""
    normalized = str(reasoning_mode).strip().lower()
    if normalized not in {"evolution", "trajectory", "hybrid"}:
        raise ValueError(
            "Unsupported reasoning mode. Supported: evolution, trajectory, hybrid"
        )
    return normalized


def _arc_dataset_dir(data_dir: Path, arc_version: str) -> Path:
    """Get the dataset directory path for the selected ARC version."""
    return Path(data_dir) / f"arc-agi-{_normalize_arc_version(arc_version)}"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ARCTask:
    """A single ARC-AGI task."""
    task_id: str
    train_examples: List[Dict[str, List[List[int]]]]  # [{"input": [[...]], "output": [[...]]}]
    test_inputs: List[List[List[int]]]  # List of test input grids
    test_outputs: List[List[List[int]]]  # List of expected output grids (ground truth)


@dataclass
class TaskResult:
    """Result of evaluating a single task."""
    task_id: str = ""
    test_index: int = 0

    # Ground truth
    expected_output: List[List[int]] = field(default_factory=list)

    # Baseline results
    baseline_output: str = ""  # Raw text output
    baseline_parsed: Optional[List[List[int]]] = None  # Parsed grid (None if parsing failed)
    baseline_correct: bool = False
    baseline_parse_error: Optional[str] = None

    # Latent Reasoning results
    lr_output: str = ""  # Raw text output
    lr_parsed: Optional[List[List[int]]] = None  # Parsed grid
    lr_correct: bool = False
    lr_accepted_attempt: int = 0
    lr_score: float = 0.0
    lr_generations: int = 0
    lr_retries: int = 0
    lr_parse_attempts: int = 0
    lr_best_partial: float = 0.0
    lr_decode_trace: List[Dict[str, Any]] = field(default_factory=list)
    lr_parse_error: Optional[str] = None
    lr_strategy_trace: List[Dict[str, Any]] = field(default_factory=list)

    # Timing
    baseline_time: float = 0.0
    lr_time: float = 0.0

    # Error info
    error: Optional[str] = None


@dataclass
class EvaluationResults:
    """Aggregate results from evaluation run."""
    encoder_model: str
    timestamp: str
    arc_version: str
    reasoning_mode: str
    trajectory_steps: int
    trajectory_decode_interval: int
    trajectory_step_scale: float
    geometry_feedback_target_forward_kl: float
    geometry_feedback_kl_tolerance: float
    geometry_feedback_steering_eta: float
    geometry_feedback_alpha: float
    geometry_feedback_kl_cap: float
    geometry_feedback_topk: int
    geometry_feedback_eta_min: float
    geometry_feedback_eta_max: float
    geometry_feedback_eta_growth: float
    geometry_feedback_eta_decay: float
    geometry_feedback_controller: str
    geometry_feedback_controller_kp: float
    geometry_feedback_controller_ki: float
    geometry_feedback_controller_kd: float
    geometry_feedback_controller_error_ema: float
    total_tasks: int
    total_tests: int

    # Accuracy
    baseline_correct: int = 0
    lr_correct: int = 0
    baseline_parsed: int = 0  # Successfully parsed outputs
    lr_parsed: int = 0

    # Per-task results
    task_results: List[TaskResult] = field(default_factory=list)

    # Timing
    total_time: float = 0.0

    # Errors
    errors: List[str] = field(default_factory=list)


# =============================================================================
# Prompt Engineering - CRITICAL for getting solutions not plans
# =============================================================================

SOLUTION_INJECTION = """
CRITICAL INSTRUCTION: You must output the ACTUAL SOLUTION GRID, not a plan or explanation.
DO NOT describe steps. DO NOT explain your reasoning. DO NOT give a plan.
ONLY output the final answer grid as a JSON array of arrays.

Your response must be ONLY the output grid in this exact format:
[[row1], [row2], [row3], ...]

Example of CORRECT response:
[[0, 1, 2], [3, 4, 5], [6, 7, 8]]

Example of WRONG response (do NOT do this):
"Step 1: First, I would analyze the pattern..."
"The solution involves rotating the grid..."
"Here's my plan for solving this..."

JUST OUTPUT THE GRID. NOTHING ELSE.
"""


def format_grid_for_prompt(grid: List[List[int]]) -> str:
    """Format a grid for display in prompt."""
    return json.dumps(grid)


def format_arc_prompt(task: ARCTask, test_index: int = 0) -> str:
    """
    Format an ARC task as a prompt for the model.

    Includes strong injection to get direct solution, not a plan.
    """
    prompt_parts = []

    # Header with strong solution instruction
    prompt_parts.append(SOLUTION_INJECTION)
    prompt_parts.append("\n" + "="*60 + "\n")
    prompt_parts.append("ARC PUZZLE - Find the pattern and give the output grid\n")
    prompt_parts.append("="*60 + "\n\n")

    # Training examples
    prompt_parts.append("TRAINING EXAMPLES (learn the pattern from these):\n\n")

    for i, example in enumerate(task.train_examples, 1):
        prompt_parts.append(f"Example {i}:\n")
        prompt_parts.append(f"Input:  {format_grid_for_prompt(example['input'])}\n")
        prompt_parts.append(f"Output: {format_grid_for_prompt(example['output'])}\n\n")

    # Test input
    prompt_parts.append("="*60 + "\n")
    prompt_parts.append("TEST - Apply the pattern you learned:\n\n")
    prompt_parts.append(f"Test Input: {format_grid_for_prompt(task.test_inputs[test_index])}\n\n")

    # Final instruction with strong emphasis
    prompt_parts.append("="*60 + "\n")
    prompt_parts.append("YOUR ANSWER (output grid ONLY as JSON array, no text):\n")
    prompt_parts.append("Output: ")

    return "".join(prompt_parts)


def format_arc_retry_prompt(
    task: ARCTask,
    test_index: int = 0,
    previous_output: str = "",
    failure_hint: str = "",
) -> str:
    """Build a strict retry prompt after a non-grid or malformed previous attempt."""
    prompt_parts = []
    prompt_parts.append(SOLUTION_INJECTION)
    prompt_parts.append("\n" + "="*60 + "\n")
    prompt_parts.append("ARC PUZZLE - Find the pattern and give the output grid\n")
    prompt_parts.append("="*60 + "\n\n")

    prompt_parts.append("Previous attempt was malformed or wrong. Re-issue the answer.\n")
    if failure_hint:
        prompt_parts.append(f"Failure hint: {failure_hint}\n")
    if previous_output:
        prompt_parts.append("Previous answer:\n")
        prompt_parts.append(f"{previous_output}\n\n")

    prompt_parts.append(
        "Output requirements:\n"
        "- ONLY the final grid as a JSON 2D array\n"
        "- No explanation, no prose, no markdown\n"
        "- Example: [[0,1,2],[3,4,5]]\n\n"
    )

    prompt_parts.append("TRAINING EXAMPLES (learn the pattern from these):\n\n")

    for i, example in enumerate(task.train_examples, 1):
        prompt_parts.append(f"Example {i}:\n")
        prompt_parts.append(f"Input:  {format_grid_for_prompt(example['input'])}\n")
        prompt_parts.append(f"Output: {format_grid_for_prompt(example['output'])}\n\n")

    prompt_parts.append("="*60 + "\n")
    prompt_parts.append("TEST - Apply the pattern you learned:\n\n")
    prompt_parts.append(f"Test Input: {format_grid_for_prompt(task.test_inputs[test_index])}\n\n")

    prompt_parts.append("="*60 + "\n")
    prompt_parts.append("YOUR ANSWER (output grid ONLY as JSON array, no text):\n")
    prompt_parts.append("Output: ")

    return "".join(prompt_parts)


def format_arc_consensus_prompt(
    task: ARCTask,
    test_index: int = 0,
    prior_outputs: List[str] = None,
) -> str:
    """Build a strict prompt after multiple failed attempts using candidate outputs."""
    prompt_parts = []
    prompt_parts.append(SOLUTION_INJECTION)
    prompt_parts.append("\n" + "="*60 + "\n")
    prompt_parts.append("ARC PUZZLE - Find the pattern and give the output grid\n")
    prompt_parts.append("="*60 + "\n\n")

    prior_outputs = prior_outputs or []
    prompt_parts.append("You have multiple prior outputs. Choose the most plausible one:\n\n")
    for idx, candidate in enumerate(prior_outputs[-3:], start=1):
        prompt_parts.append(f"{idx}. {candidate if candidate else '[empty]'}\n")

    prompt_parts.append(
        "\nUse the candidate that best matches the training examples. If all candidates are wrong, "
        "return a correct grid anyway.\n"
        "Your final response must be ONLY a JSON grid.\n\n"
    )

    # Training examples
    prompt_parts.append("TRAINING EXAMPLES (learn the pattern from these):\n\n")
    for i, example in enumerate(task.train_examples, 1):
        prompt_parts.append(f"Example {i}:\n")
        prompt_parts.append(f"Input:  {format_grid_for_prompt(example['input'])}\n")
        prompt_parts.append(f"Output: {format_grid_for_prompt(example['output'])}\n\n")

    # Test input
    prompt_parts.append("="*60 + "\n")
    prompt_parts.append("TEST - Apply the pattern you learned:\n\n")
    prompt_parts.append(f"Test Input: {format_grid_for_prompt(task.test_inputs[test_index])}\n\n")

    # Final instruction with strictness
    prompt_parts.append("="*60 + "\n")
    prompt_parts.append("YOUR ANSWER (output grid ONLY as JSON array, no text):\n")
    prompt_parts.append("Output: ")

    return "".join(prompt_parts)


# =============================================================================
# Output Parsing
# =============================================================================


def _coerce_grid(grid: List[Any]) -> Optional[List[List[int]]]:
    """Validate and coerce a parsed object into ARC grid shape."""
    if not isinstance(grid, list) or not grid:
        return None

    expected_width: Optional[int] = None
    parsed_rows: List[List[int]] = []
    for row in grid:
        if not isinstance(row, list) or not row:
            return None
        if expected_width is None:
            expected_width = len(row)
        elif len(row) != expected_width:
            return None
        parsed_row = []
        for cell in row:
            if isinstance(cell, bool):
                return None
            try:
                value = int(cell)
            except (TypeError, ValueError):
                return None
            if value < 0 or value > 9:
                return None
            parsed_row.append(value)
        parsed_rows.append(parsed_row)
    return parsed_rows


def parse_grid_from_output(text: str) -> Tuple[Optional[List[List[int]]], Optional[str]]:
    """
    Parse a grid from model output.

    Returns:
        (grid, error_message) - grid is None if parsing failed
    """
    if not text or not text.strip():
        return None, "Empty output"

    # Try to find JSON array in the output
    # Look for patterns like [[...], [...], ...]

    # Method 1: Direct JSON parse of whole output
    try:
        cleaned = text.strip()
        # Remove common prefixes
        for prefix in ["Output:", "Answer:", "Result:", "```json", "```"]:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        grid = json.loads(cleaned)
        grid = _coerce_grid(grid)
        if grid is not None:
            return grid, None
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Method 2: Find JSON array pattern with regex
    # Match nested arrays: [[0,1],[2,3]]
    pattern = r'\[\s*\[[\d\s,\[\]]+\]\s*\]'
    matches = re.findall(pattern, text)

    for match in matches:
        try:
            grid = json.loads(match)
            grid = _coerce_grid(grid)
            if grid is not None:
                return grid, None
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    # Method 3: Try to extract line by line
    lines = text.strip().split('\n')
    grid = []
    for line in lines:
        # Look for array pattern in line
        match = re.search(r'\[[\d\s,]+\]', line)
        if match:
            try:
                candidate = _coerce_grid([json.loads(match.group())])
                if candidate:
                    grid = candidate
                    return grid, None
            except (json.JSONDecodeError, ValueError):
                continue

    return None, f"Could not parse grid from output: {text[:200]}..."


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert mixed-type metric values into a float."""
    try:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return float(value)
    except (TypeError, ValueError):
        return default


def grids_match(grid1: Optional[List[List[int]]], grid2: Optional[List[List[int]]]) -> bool:
    """Check if two grids are exactly equal."""
    if grid1 is None or grid2 is None:
        return False

    if len(grid1) != len(grid2):
        return False

    for row1, row2 in zip(grid1, grid2):
        if len(row1) != len(row2):
            return False
        if row1 != row2:
            return False

    return True


def partial_match_score(predicted: Optional[List[List[int]]], expected: Optional[List[List[int]]]) -> float:
    """Calculate partial match score between predicted and expected grids.

    Returns a score from 0.0 to 1.0:
    - 1.0 = exact match
    - 0.0 = no match or no output
    - Partial scores for partial matches
    """
    if predicted is None or expected is None:
        return 0.0

    if not predicted or not expected:
        return 0.0

    # Count matching cells in overlapping region
    total_expected_cells = sum(len(row) for row in expected)
    matching_cells = 0

    for i, expected_row in enumerate(expected):
        if i < len(predicted):
            pred_row = predicted[i]
            for j, expected_val in enumerate(expected_row):
                if j < len(pred_row) and pred_row[j] == expected_val:
                    matching_cells += 1

    return matching_cells / total_expected_cells if total_expected_cells > 0 else 0.0


def rows_match_prefix(predicted: Optional[List[List[int]]], expected: Optional[List[List[int]]]) -> bool:
    """Check if predicted rows match the beginning of expected (for truncated outputs)."""
    if predicted is None or expected is None:
        return False

    if not predicted:
        return False

    # Check if all predicted rows match the corresponding expected rows
    for i, pred_row in enumerate(predicted):
        if i >= len(expected):
            return False
        if pred_row != expected[i]:
            return False

    return True


# =============================================================================
# Data Loading
# =============================================================================

def _find_existing_task_dir(arc_dir: Path, split: str) -> Optional[Path]:
    """Find where task JSON files are stored inside an ARC checkout."""
    possible_paths = [
        arc_dir / "data" / split,
        arc_dir / split,
        arc_dir / "data" / "evaluation",
    ]

    for path in possible_paths:
        if path.exists():
            return path
    return None


def download_arc_dataset(data_dir: Path, arc_version: str = "3") -> Path:
    """Download ARC dataset (version 2 or 3) if not present."""
    arc_version = _normalize_arc_version(arc_version)
    arc_dir = _arc_dataset_dir(data_dir, arc_version)
    data_dir = Path(data_dir)
    label = f"ARC-AGI-{arc_version}"

    for eval_path in [
        arc_dir / "data" / "evaluation",
        arc_dir / "evaluation",
        arc_dir / "data" / "eval",
    ]:
        if eval_path.exists() and any(eval_path.glob("*.json")):
            console.print(f"[green]{label} dataset found at {arc_dir}[/green]")
            return arc_dir

    if (arc_dir.exists() and any(arc_dir.rglob("*.json"))):
        return arc_dir

    console.print(f"[yellow]Downloading {label} dataset...[/yellow]")

    # Clone from GitHub
    import subprocess
    import shutil

    if arc_dir.exists():
        console.print(f"[yellow]Removing incomplete data directory...[/yellow]")
        shutil.rmtree(arc_dir)

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/fchollet/ARC-AGI.git", str(arc_dir)],
            check=True,
            capture_output=True,
        )
        console.print(f"[green]Downloaded {label} to {arc_dir}[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to download: {e.stderr.decode()}[/red]")
        raise

    return arc_dir


def download_arc_agi2(data_dir: Path) -> Path:
    """Backward-compatible ARC-AGI-2 download helper."""
    return download_arc_dataset(data_dir=data_dir, arc_version="2")


def load_arc_tasks(arc_dir: Path, split: str = "evaluation", arc_version: str = "3") -> List[ARCTask]:
    """Load ARC tasks from the dataset directory."""
    arc_dir = Path(arc_dir)
    arc_version = _normalize_arc_version(arc_version)

    tasks_dir = _find_existing_task_dir(arc_dir, split)

    if tasks_dir is None:
        # Fallback: if version structure shifted, find any directory containing json tasks.
        candidate_dirs = [
            p for p in arc_dir.glob("**/")
            if p.is_dir() and any(file.suffix == ".json" for file in p.iterdir())
        ]
        if candidate_dirs:
            tasks_dir = sorted(candidate_dirs)[0]

    if tasks_dir is None:
        raise FileNotFoundError(
            f"Could not find ARC-{arc_version} tasks under {arc_dir}"
        )

    tasks = []
    task_files = list(tasks_dir.glob("*.json"))

    console.print(f"[blue]Loading {len(task_files)} tasks from {tasks_dir}[/blue]")

    for task_file in sorted(task_files):
        try:
            with open(task_file, 'r') as f:
                data = json.load(f)

            task = ARCTask(
                task_id=task_file.stem,
                train_examples=data.get("train", []),
                test_inputs=[t["input"] for t in data.get("test", [])],
                test_outputs=[t.get("output", []) for t in data.get("test", [])],
            )
            tasks.append(task)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load {task_file}: {e}[/yellow]")

    return tasks


# =============================================================================
# Main Evaluator
# =============================================================================

class ARCEvaluator:
    """Evaluator for ARC-AGI benchmark variants."""

    def __init__(
        self,
        encoder_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        chains: int = 4,
        generations: int = 5,
        max_tokens: int = 512,
        decode_mode: str = "soft_prompt",
        data_dir: str = "./data",
        output_dir: str = "./eval_results",
        arc_version: str = "3",
        lr_retries: int = 1,
        arc_strategy: str = "adaptive",
        reasoning_mode: str = "evolution",
        trajectory_steps: int = 6,
        trajectory_decode_interval: int = 0,
        trajectory_step_scale: float = 0.2,
        geometry_feedback_target_forward_kl: float = 0.06,
        geometry_feedback_kl_tolerance: float = 0.5,
        geometry_feedback_steering_eta: float = 0.05,
        geometry_feedback_alpha: float = 0.01,
        geometry_feedback_kl_cap: float = 0.5,
        geometry_feedback_topk: int = 50,
        geometry_feedback_eta_min: float = 0.01,
        geometry_feedback_eta_max: float = 0.5,
        geometry_feedback_eta_growth: float = 1.06,
        geometry_feedback_eta_decay: float = 0.85,
        geometry_feedback_controller: str = "legacy",
        geometry_feedback_controller_kp: float = 0.0,
        geometry_feedback_controller_ki: float = 0.0,
        geometry_feedback_controller_kd: float = 0.0,
        geometry_feedback_controller_error_ema: float = 0.2,
    ):
        self.encoder_model = encoder_model
        self.chains = chains
        self.generations = generations
        self.max_tokens = max_tokens
        self.decode_mode = decode_mode
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.arc_version = _normalize_arc_version(arc_version)
        self.lr_retries = max(0, lr_retries)
        self.arc_strategy = _normalize_arc_strategy(arc_strategy)
        self.reasoning_mode = _normalize_reasoning_mode(reasoning_mode)
        self.trajectory_steps = trajectory_steps
        self.trajectory_decode_interval = trajectory_decode_interval
        self.trajectory_step_scale = trajectory_step_scale
        self.geometry_feedback_target_forward_kl = geometry_feedback_target_forward_kl
        self.geometry_feedback_kl_tolerance = geometry_feedback_kl_tolerance
        self.geometry_feedback_steering_eta = geometry_feedback_steering_eta
        self.geometry_feedback_alpha = geometry_feedback_alpha
        self.geometry_feedback_kl_cap = geometry_feedback_kl_cap
        self.geometry_feedback_topk = geometry_feedback_topk
        self.geometry_feedback_eta_min = geometry_feedback_eta_min
        self.geometry_feedback_eta_max = geometry_feedback_eta_max
        self.geometry_feedback_eta_growth = geometry_feedback_eta_growth
        self.geometry_feedback_eta_decay = geometry_feedback_eta_decay
        self.geometry_feedback_controller = geometry_feedback_controller
        self.geometry_feedback_controller_kp = geometry_feedback_controller_kp
        self.geometry_feedback_controller_ki = geometry_feedback_controller_ki
        self.geometry_feedback_controller_kd = geometry_feedback_controller_kd
        self.geometry_feedback_controller_error_ema = geometry_feedback_controller_error_ema

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.decode_mode not in {
            "seed",
            "soft_prompt",
            "dual_steering",
            "intermediate_steering",
            "geometry_feedback",
        }:
            raise ValueError(
                "Unsupported decode_mode. Valid options: seed, soft_prompt, "
                "dual_steering, intermediate_steering, geometry_feedback"
            )

        # Lazy load engine
        self._engine = None
        self._self_improving_profiles = self._build_self_improving_profiles()
        self._repair_profiles = self._build_repair_profiles()
        self._geometry_bandit_profiles = self._build_geometry_bandit_profiles()
        self._self_improving_profile_stats = {
            profile["name"]: {
                "samples": 0,
                "exact": 0,
                "partial_sum": 0.0,
            }
            for profile in self._self_improving_profiles
        }
        self._geometry_bandit_profile_stats = {
            profile["name"]: {
                "samples": 0,
                "exact": 0,
                "parse_success": 0,
                "partial_sum": 0.0,
                "trace_signal_sum": 0.0,
                "reward_sum": 0.0,
            }
            for profile in self._geometry_bandit_profiles
        }
        self._geometry_bandit_total_attempts = 0
        self._geometry_bandit_exploration = 0.35

    def _default_decode_profile(self) -> dict[str, Any]:
        """Return the evaluator's base decode profile."""
        return {
            "name": "default",
            "decode_mode": self.decode_mode,
            "geometry_controller": self.geometry_feedback_controller,
            "geometry_controller_kp": self.geometry_feedback_controller_kp,
            "geometry_controller_ki": self.geometry_feedback_controller_ki,
            "geometry_controller_kd": self.geometry_feedback_controller_kd,
            "geometry_controller_error_ema": self.geometry_feedback_controller_error_ema,
        }

    def _build_self_improving_profiles(self) -> list[dict[str, Any]]:
        """Build the candidate decode profiles used by the self-improving strategy."""
        profiles = [
            {
                **self._default_decode_profile(),
                "name": "geometry_feedback_legacy",
            },
            {
                "name": "geometry_feedback_pid_balanced",
                "decode_mode": "geometry_feedback",
                "geometry_controller": "pid",
                "geometry_controller_kp": 0.20,
                "geometry_controller_ki": 0.02,
                "geometry_controller_kd": 0.01,
                "geometry_controller_error_ema": 0.30,
            },
            {
                "name": "geometry_feedback_pid_aggressive",
                "decode_mode": "geometry_feedback",
                "geometry_controller": "pid",
                "geometry_controller_kp": 0.35,
                "geometry_controller_ki": 0.04,
                "geometry_controller_kd": 0.02,
                "geometry_controller_error_ema": 0.25,
            },
            {
                "name": "dual_steering_fallback",
                "decode_mode": "dual_steering",
                "geometry_controller": self.geometry_feedback_controller,
                "geometry_controller_kp": self.geometry_feedback_controller_kp,
                "geometry_controller_ki": self.geometry_feedback_controller_ki,
                "geometry_controller_kd": self.geometry_feedback_controller_kd,
                "geometry_controller_error_ema": self.geometry_feedback_controller_error_ema,
            },
        ]
        if self.decode_mode != "geometry_feedback":
            # Keep the configured non-geometry mode as a separate candidate.
            profiles = [profiles[0]] + profiles[1:]
        return profiles

    def _build_repair_profiles(self) -> list[dict[str, Any]]:
        """Build the candidate decode profiles used by the repair strategy."""
        return [
            {
                **self._default_decode_profile(),
            },
            {
                "name": "repair_geometry_balanced",
                "decode_mode": "geometry_feedback",
                "geometry_controller": "pid",
                "geometry_controller_kp": 0.22,
                "geometry_controller_ki": 0.03,
                "geometry_controller_kd": 0.02,
                "geometry_controller_error_ema": 0.30,
            },
            {
                "name": "repair_geometry_aggressive",
                "decode_mode": "geometry_feedback",
                "geometry_controller": "pid",
                "geometry_controller_kp": 0.40,
                "geometry_controller_ki": 0.08,
                "geometry_controller_kd": 0.03,
                "geometry_controller_error_ema": 0.20,
            },
            {
                "name": "repair_dual_steering_fallback",
                "decode_mode": "dual_steering",
                "geometry_controller": self.geometry_feedback_controller,
                "geometry_controller_kp": self.geometry_feedback_controller_kp,
                "geometry_controller_ki": self.geometry_feedback_controller_ki,
                "geometry_controller_kd": self.geometry_feedback_controller_kd,
                "geometry_controller_error_ema": self.geometry_feedback_controller_error_ema,
            },
        ]

    def _build_geometry_bandit_profiles(self) -> list[dict[str, Any]]:
        """Build the candidate decode profiles used by the geometry-bandit strategy."""
        return [
            {
                **self._default_decode_profile(),
            },
            {
                "name": "bandit_geometry_balanced",
                "decode_mode": "geometry_feedback",
                "geometry_controller": "pid",
                "geometry_controller_kp": 0.24,
                "geometry_controller_ki": 0.03,
                "geometry_controller_kd": 0.015,
                "geometry_controller_error_ema": 0.30,
            },
            {
                "name": "bandit_geometry_aggressive",
                "decode_mode": "geometry_feedback",
                "geometry_controller": "pid",
                "geometry_controller_kp": 0.42,
                "geometry_controller_ki": 0.06,
                "geometry_controller_kd": 0.03,
                "geometry_controller_error_ema": 0.22,
            },
            {
                "name": "bandit_dual_steering_fallback",
                "decode_mode": "dual_steering",
                "geometry_controller": self.geometry_feedback_controller,
                "geometry_controller_kp": self.geometry_feedback_controller_kp,
                "geometry_controller_ki": self.geometry_feedback_controller_ki,
                "geometry_controller_kd": self.geometry_feedback_controller_kd,
                "geometry_controller_error_ema": self.geometry_feedback_controller_error_ema,
            },
        ]

    def _repair_profile_signal(self, trace: List[Dict[str, Any]]) -> float:
        """Compute a compact quality signal from geometry trace.

        Higher means better geometry control and lower output instability.
        """
        if not trace:
            return 0.0

        forward_kl = [_safe_float(item.get("forward_kl")) for item in trace]
        if not forward_kl:
            return 0.0

        forward_kl_tolerance = max(1e-8, self.geometry_feedback_kl_tolerance)
        target_kl = self.geometry_feedback_target_forward_kl
        if target_kl > 0.0:
            low = max(0.0, target_kl * (1.0 - forward_kl_tolerance))
            high = target_kl * (1.0 + forward_kl_tolerance)
            within_band = sum(1 for value in forward_kl if low <= value <= high) / len(forward_kl)
        else:
            within_band = sum(1 for value in forward_kl if 0 <= value <= forward_kl_tolerance) / len(forward_kl)

        topk_overlap = [_safe_float(item.get("topk_overlap")) for item in trace]
        topk_overlap_mean = sum(topk_overlap) / len(topk_overlap) if topk_overlap else 0.0

        entropy_delta = [_safe_float(item.get("entropy_delta")) for item in trace]
        entropy_drift = sum(abs(value) for value in entropy_delta) / len(entropy_delta) if entropy_delta else 0.0
        entropy_signal = max(0.0, 1.0 - min(1.0, entropy_drift))

        return max(
            0.0,
            min(
                1.0,
                0.55 * within_band + 0.25 * topk_overlap_mean + 0.20 * entropy_signal,
            ),
        )

    def _repair_hint(
        self,
        parsed: bool,
        partial: float,
        parse_error: Optional[str],
    ) -> str:
        if not parsed:
            if parse_error:
                return f"Previous output was not parseable as JSON grid. Parse error: {parse_error}"
            return "Previous output was not parseable as a JSON grid."
        if partial >= 1.0:
            return "Previous output was exactly correct."
        if partial >= 0.75:
            return "Very close. Fix only remaining grid cells and keep JSON shape exactly."
        if partial >= 0.40:
            return "Partially close. Re-check full dimensions and every expected cell value."
        return "Output is far from target; preserve JSON shape and recompute the full rule."

    def _select_repair_profile(
        self,
        attempt: int,
        last_attempt_parsed: Optional[bool],
        last_attempt_partial: float,
        best_partial: float,
        last_profile_name: str,
        last_parse_error: Optional[str],
        last_decode_trace: List[Dict[str, Any]],
    ) -> dict[str, Any]:
        """Select a repair profile from prior attempt outcome and geometry signal."""
        profiles_by_name = {profile["name"]: profile for profile in self._repair_profiles}
        if attempt == 0:
            return self._default_decode_profile()

        if last_attempt_parsed is None:
            return profiles_by_name.get("repair_geometry_balanced", self._default_decode_profile())

        signal = self._repair_profile_signal(last_decode_trace)

        if not last_attempt_parsed:
            if last_profile_name == "repair_dual_steering_fallback":
                return profiles_by_name.get("repair_geometry_balanced", self._default_decode_profile())
            if signal > 0.55:
                return profiles_by_name.get("repair_geometry_balanced", self._default_decode_profile())
            return profiles_by_name.get("repair_dual_steering_fallback", self._default_decode_profile())

        partial_improved = last_attempt_partial - best_partial

        if last_attempt_partial >= 1.0:
            return profiles_by_name.get("repair_geometry_balanced", self._default_decode_profile())

        if partial_improved > 0.10:
            if signal >= 0.40:
                return profiles_by_name.get("repair_geometry_balanced", self._default_decode_profile())
            return profiles_by_name.get("repair_geometry_aggressive", self._default_decode_profile())

        if partial_improved < -0.10:
            if signal < 0.25:
                return profiles_by_name.get("repair_dual_steering_fallback", self._default_decode_profile())
            return profiles_by_name.get("repair_geometry_aggressive", self._default_decode_profile())

        if best_partial >= 0.40:
            return profiles_by_name.get("repair_geometry_balanced", self._default_decode_profile())

        if last_parse_error:
            return profiles_by_name.get("repair_dual_steering_fallback", self._default_decode_profile())

        if signal < 0.35:
            return profiles_by_name.get("repair_dual_steering_fallback", self._default_decode_profile())

        return profiles_by_name.get("repair_geometry_balanced", self._default_decode_profile())

    def _score_self_improving_profile(self, profile_name: str) -> float:
        """Score a profile with lightweight exploration regularization."""
        stats = self._self_improving_profile_stats[profile_name]
        samples = max(1, stats["samples"])
        exact_rate = stats["exact"] / samples
        partial_rate = stats["partial_sum"] / samples
        # Encourage early exploration while converging toward measured success.
        exploration = 0.20 / samples**0.5
        return 0.70 * exact_rate + 0.30 * partial_rate + exploration

    def _select_self_improving_profile(self, attempt: int) -> dict[str, Any]:
        """Pick a decode profile for a given retry attempt."""
        if attempt == 0:
            return self._self_improving_profiles[0]
        scored = [
            (self._score_self_improving_profile(profile["name"]), profile)
            for profile in self._self_improving_profiles
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _select_geometry_bandit_profile(self, attempt: int) -> dict[str, Any]:
        """Pick a decode profile for one attempt using UCB-style exploration."""
        if not self._geometry_bandit_profiles:
            return self._default_decode_profile()

        if attempt < len(self._geometry_bandit_profiles):
            return self._geometry_bandit_profiles[attempt]

        total = max(1, self._geometry_bandit_total_attempts)
        scored = []
        for profile in self._geometry_bandit_profiles:
            stats = self._geometry_bandit_profile_stats[profile["name"]]
            samples = max(1, stats["samples"])
            mean_reward = stats["reward_sum"] / samples
            exploration = self._geometry_bandit_exploration * math.sqrt(math.log(total) / samples)
            scored.append((mean_reward + exploration, profile))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _apply_decode_profile(self, profile: dict[str, Any]) -> None:
        """Apply profile knobs to the active engine config."""
        if self._engine is None:
            _ = self.engine
        self._engine.config.synthesis.decode_mode = profile["decode_mode"]
        self._engine.config.synthesis.geometry_feedback_controller = profile["geometry_controller"]
        self._engine.config.synthesis.geometry_feedback_controller_kp = float(profile["geometry_controller_kp"])
        self._engine.config.synthesis.geometry_feedback_controller_ki = float(profile["geometry_controller_ki"])
        self._engine.config.synthesis.geometry_feedback_controller_kd = float(profile["geometry_controller_kd"])
        self._engine.config.synthesis.geometry_feedback_controller_error_ema = float(profile["geometry_controller_error_ema"])

    def _record_self_improving_outcome(
        self,
        profile_name: str,
        exact: bool,
        partial: float,
    ) -> None:
        """Track performance for the profile to improve future task attempts."""
        stats = self._self_improving_profile_stats[profile_name]
        stats["samples"] += 1
        if exact:
            stats["exact"] += 1
        stats["partial_sum"] += max(0.0, float(partial))

        # Keep only stable numeric ranges.
        if stats["samples"] > 0:
            max_partial = 1.0 * stats["samples"]
            if stats["partial_sum"] > max_partial:
                stats["partial_sum"] = max_partial

    def _record_geometry_bandit_outcome(
        self,
        profile_name: str,
        parsed: bool,
        partial: float,
        decode_trace: List[Dict[str, Any]],
    ) -> None:
        """Track bandit profile performance from one decode attempt."""
        if profile_name not in self._geometry_bandit_profile_stats:
            return

        stats = self._geometry_bandit_profile_stats[profile_name]
        self._geometry_bandit_total_attempts += 1
        stats["samples"] += 1

        partial_score = _safe_float(partial, default=0.0)
        partial_score = max(0.0, min(1.0, partial_score))
        parse_ok = 1.0 if parsed else 0.0
        exact_score = 1.0 if (parsed and partial >= 1.0) else 0.0
        signal = self._repair_profile_signal(decode_trace)

        stats["parse_success"] += int(parse_ok > 0.5)
        stats["exact"] += int(exact_score > 0.5)
        stats["partial_sum"] += partial_score
        stats["trace_signal_sum"] += signal

        reward = 0.55 * partial_score + 0.30 * signal + 0.10 * parse_ok + 0.10 * exact_score
        if reward > 1.0:
            reward = 1.0
        stats["reward_sum"] += reward

        max_partial = 1.0 * stats["samples"]
        max_signal = 1.0 * stats["samples"]
        if stats["partial_sum"] > max_partial:
            stats["partial_sum"] = max_partial
        if stats["trace_signal_sum"] > max_signal:
            stats["trace_signal_sum"] = max_signal

    @property
    def engine(self):
        """Lazy-load the engine."""
        if self._engine is None:
            from latent_reasoning import Engine

            console.print(f"[blue]Initializing engine with {self.encoder_model}...[/blue]")
            self._engine = Engine(
                encoder=self.encoder_model,
                verbosity="silent",
            )
            # Configure evolution parameters
            self._engine.config.evolution.chains = self.chains
            self._engine.config.evolution.generations = self.generations
            self._engine.config.synthesis.max_tokens = self.max_tokens
            self._engine.config.synthesis.decode_mode = self.decode_mode
            self._engine.config.synthesis.reasoning_mode = self.reasoning_mode
            self._engine.config.synthesis.trajectory_steps = self.trajectory_steps
            self._engine.config.synthesis.trajectory_decode_interval = self.trajectory_decode_interval
            self._engine.config.synthesis.trajectory_step_scale = self.trajectory_step_scale
            self._engine.config.synthesis.geometry_feedback_target_forward_kl = (
                self.geometry_feedback_target_forward_kl
            )
            self._engine.config.synthesis.geometry_feedback_kl_tolerance = (
                self.geometry_feedback_kl_tolerance
            )
            self._engine.config.synthesis.geometry_feedback_steering_eta = (
                self.geometry_feedback_steering_eta
            )
            self._engine.config.synthesis.geometry_feedback_alpha = (
                self.geometry_feedback_alpha
            )
            self._engine.config.synthesis.geometry_feedback_kl_cap = (
                self.geometry_feedback_kl_cap
            )
            self._engine.config.synthesis.geometry_feedback_topk = (
                self.geometry_feedback_topk
            )
            self._engine.config.synthesis.geometry_feedback_eta_min = (
                self.geometry_feedback_eta_min
            )
            self._engine.config.synthesis.geometry_feedback_eta_max = (
                self.geometry_feedback_eta_max
            )
            self._engine.config.synthesis.geometry_feedback_eta_growth = (
                self.geometry_feedback_eta_growth
            )
            self._engine.config.synthesis.geometry_feedback_eta_decay = (
                self.geometry_feedback_eta_decay
            )
            self._engine.config.synthesis.geometry_feedback_controller = (
                self.geometry_feedback_controller
            )
            self._engine.config.synthesis.geometry_feedback_controller_kp = (
                self.geometry_feedback_controller_kp
            )
            self._engine.config.synthesis.geometry_feedback_controller_ki = (
                self.geometry_feedback_controller_ki
            )
            self._engine.config.synthesis.geometry_feedback_controller_kd = (
                self.geometry_feedback_controller_kd
            )
            self._engine.config.synthesis.geometry_feedback_controller_error_ema = (
                self.geometry_feedback_controller_error_ema
            )

        return self._engine

    def evaluate_task(
        self,
        task: ARCTask,
        test_index: int = 0
    ) -> TaskResult:
        """Evaluate a single task with both baseline and LR."""

        prompt = format_arc_prompt(task, test_index)
        expected = task.test_outputs[test_index] if task.test_outputs else None

        result = TaskResult(
            task_id=task.task_id,
            test_index=test_index,
            expected_output=expected,
            baseline_output="",
            baseline_parsed=None,
            baseline_correct=False,
            lr_output="",
            lr_parsed=None,
            lr_correct=False,
            lr_retries=self.lr_retries,
        )

        try:
            # Run baseline
            start = time.time()
            baseline_output = self.engine.run_baseline(prompt)
            result.baseline_time = time.time() - start
            result.baseline_output = baseline_output

            # Parse baseline
            parsed, error = parse_grid_from_output(baseline_output)
            result.baseline_parsed = parsed
            result.baseline_parse_error = error
            result.baseline_correct = grids_match(parsed, expected)

        except Exception as e:
            result.error = f"Baseline error: {str(e)}"
            result.baseline_output = str(e)

        try:
            # Run Latent Reasoning
            selected_profile_names = []
            attempts = 1 if self.arc_strategy == "single" else 1 + self.lr_retries
            best_partial = -1.0
            best_plan = ""
            best_grid = None
            best_confidence = 0.0
            best_generations = 0
            best_parse_error = None
            best_attempt = 0
            best_decode_trace: List[Dict[str, Any]] = []
            last_decode_trace: List[Dict[str, Any]] = []
            attempt_outputs: List[str] = []
            best_profile_name = "default"
            last_attempt_parsed: Optional[bool] = None
            last_attempt_partial = -1.0
            last_parse_error: Optional[str] = None
            last_profile_name = "default"

            for attempt in range(attempts):
                if attempt == 0:
                    attempt_prompt = format_arc_prompt(task, test_index)
                elif self.arc_strategy == "consensus":
                    attempt_prompt = format_arc_consensus_prompt(
                        task=task,
                        test_index=test_index,
                        prior_outputs=attempt_outputs,
                    )
                else:
                    failure_hint = ""
                    if self.arc_strategy == "repair":
                        failure_hint = self._repair_hint(
                            parsed=bool(last_attempt_parsed),
                            partial=last_attempt_partial,
                            parse_error=last_parse_error,
                        )
                    attempt_prompt = format_arc_retry_prompt(
                        task=task,
                        test_index=test_index,
                        previous_output=result.lr_output,
                        failure_hint=failure_hint,
                    )

                if self.arc_strategy == "self_improving":
                    profile = self._select_self_improving_profile(attempt)
                    selected_profile_names.append(profile["name"])
                    self._apply_decode_profile(profile)
                elif self.arc_strategy == "repair":
                    profile = self._select_repair_profile(
                        attempt=attempt,
                        last_attempt_parsed=last_attempt_parsed,
                        last_attempt_partial=last_attempt_partial,
                        best_partial=best_partial,
                        last_profile_name=last_profile_name,
                        last_parse_error=last_parse_error,
                        last_decode_trace=last_decode_trace,
                    )
                    selected_profile_names.append(profile["name"])
                    self._apply_decode_profile(profile)
                elif self.arc_strategy == "geometry_bandit":
                    profile = self._select_geometry_bandit_profile(attempt)
                    selected_profile_names.append(profile["name"])
                    self._apply_decode_profile(profile)
                else:
                    profile = self._default_decode_profile()
                    selected_profile_names.append(profile["name"])

                start = time.time()
                lr_result = self.engine.run(attempt_prompt)
                elapsed = time.time() - start

                parsed, parse_error = parse_grid_from_output(lr_result.plan)
                partial = partial_match_score(parsed, expected) if parsed is not None else 0.0
                decode_trace = list(getattr(lr_result, "decode_trace", []) or [])
                last_decode_trace = decode_trace

                result.lr_time += elapsed
                result.lr_parse_attempts += 1
                attempt_outputs.append(lr_result.plan)

                result.lr_strategy_trace.append({
                    "attempt": attempt + 1,
                    "profile": profile["name"],
                    "decode_mode": profile["decode_mode"],
                    "geometry_controller": profile["geometry_controller"],
                    "geometry_controller_kp": profile["geometry_controller_kp"],
                    "geometry_controller_ki": profile["geometry_controller_ki"],
                    "geometry_controller_kd": profile["geometry_controller_kd"],
                    "geometry_controller_error_ema": profile["geometry_controller_error_ema"],
                    "time_s": elapsed,
                    "parse_ok": parsed is not None,
                    "parse_error": parse_error,
                    "partial_score": partial,
                    "confidence": lr_result.confidence,
                    "generations": lr_result.generations,
                    "decode_trace": decode_trace,
                })

                # Track best structured candidate so far.
                if parsed is not None and (
                    partial > best_partial or (
                        partial == best_partial and lr_result.confidence > best_confidence
                    )
                ):
                    best_partial = partial
                    best_plan = lr_result.plan
                    best_grid = parsed
                    best_confidence = lr_result.confidence
                    best_generations = lr_result.generations
                    best_parse_error = parse_error
                    best_attempt = attempt + 1
                    best_decode_trace = decode_trace
                    best_profile_name = profile["name"]

                if parsed is not None and expected is not None and grids_match(parsed, expected):
                    result.lr_output = lr_result.plan
                    result.lr_parsed = parsed
                    result.lr_score = lr_result.confidence
                    result.lr_generations = lr_result.generations
                    result.lr_parse_error = None
                    result.lr_accepted_attempt = attempt + 1
                    best_partial = 1.0
                    if self.arc_strategy == "geometry_bandit":
                        self._record_geometry_bandit_outcome(
                            profile_name=profile["name"],
                            parsed=True,
                            partial=1.0,
                            decode_trace=decode_trace,
                        )
                    break

                if self.arc_strategy == "self_improving":
                    self._record_self_improving_outcome(
                        profile_name=profile["name"],
                        exact=False,
                        partial=partial,
                    )
                if self.arc_strategy == "geometry_bandit":
                    self._record_geometry_bandit_outcome(
                        profile_name=profile["name"],
                        parsed=parsed is not None,
                        partial=partial,
                        decode_trace=decode_trace,
                    )

                # Keep latest attempt for debugging and frontier diagnostics.
                result.lr_output = lr_result.plan
                result.lr_score = lr_result.confidence
                result.lr_generations = lr_result.generations
                result.lr_parse_error = parse_error

                should_retry = attempt < attempts - 1
                if self.arc_strategy == "adaptive":
                    if parsed is None:
                        should_retry = True
                    elif last_attempt_parsed is not None:
                        should_retry = partial > last_attempt_partial
                    else:
                        should_retry = False
                elif self.arc_strategy == "repair":
                    should_retry = attempt < attempts - 1
                elif self.arc_strategy == "self_improving":
                    should_retry = attempt < attempts - 1
                elif self.arc_strategy == "consensus":
                    should_retry = attempt < attempts - 1
                elif self.arc_strategy == "geometry_bandit":
                    should_retry = attempt < attempts - 1

                last_attempt_parsed = parsed is not None
                last_attempt_partial = partial
                last_parse_error = parse_error
                last_profile_name = profile["name"]
                last_decode_trace = decode_trace

                if self.arc_strategy == "single":
                    break
                if not should_retry:
                    break

            if self.arc_strategy in {"self_improving", "repair", "geometry_bandit"}:
                self._apply_decode_profile(self._default_decode_profile())

        except Exception as e:
            if result.error:
                result.error += f"; LR error: {str(e)}"
            else:
                result.error = f"LR error: {str(e)}"
            result.lr_output = str(e)
            result.lr_parse_error = str(e)

        if best_grid is not None:
            result.lr_parsed = best_grid
            result.lr_output = best_plan
            result.lr_correct = expected is not None and grids_match(best_grid, expected)
            result.lr_score = best_confidence
            result.lr_generations = best_generations
            result.lr_parse_error = best_parse_error
            result.lr_best_partial = max(best_partial, 0.0)
            result.lr_decode_trace = best_decode_trace
            result.lr_accepted_attempt = best_attempt
            if self.arc_strategy == "self_improving":
                if result.lr_correct:
                    exact = True
                else:
                    exact = False
                self._record_self_improving_outcome(
                    profile_name=best_profile_name,
                    exact=exact,
                    partial=result.lr_best_partial,
                )
        else:
            result.lr_decode_trace = last_decode_trace
            result.lr_best_partial = 0.0

        return result

    def run_evaluation(
        self,
        max_tasks: Optional[int] = None,
        task_ids: Optional[List[str]] = None,
    ) -> EvaluationResults:
        """
        Run full evaluation on ARC-AGI dataset.

        Args:
            max_tasks: Limit number of tasks (None = all)
            task_ids: Specific task IDs to evaluate (None = all)
        """
        arc_version = self.arc_version
        arc_dir = download_arc_dataset(self.data_dir, arc_version=arc_version)
        tasks = load_arc_tasks(arc_dir, arc_version=arc_version)

        # Filter tasks if specified
        if task_ids:
            tasks = [t for t in tasks if t.task_id in task_ids]
        if max_tasks:
            tasks = tasks[:max_tasks]

        # Count total tests
        total_tests = sum(len(t.test_inputs) for t in tasks)

        results = EvaluationResults(
            encoder_model=self.encoder_model,
            timestamp=datetime.now().isoformat(),
            arc_version=self.arc_version,
            reasoning_mode=self.reasoning_mode,
            trajectory_steps=self.trajectory_steps,
            trajectory_decode_interval=self.trajectory_decode_interval,
            trajectory_step_scale=self.trajectory_step_scale,
            geometry_feedback_target_forward_kl=self.geometry_feedback_target_forward_kl,
            geometry_feedback_kl_tolerance=self.geometry_feedback_kl_tolerance,
            geometry_feedback_steering_eta=self.geometry_feedback_steering_eta,
            geometry_feedback_alpha=self.geometry_feedback_alpha,
            geometry_feedback_kl_cap=self.geometry_feedback_kl_cap,
            geometry_feedback_topk=self.geometry_feedback_topk,
            geometry_feedback_eta_min=self.geometry_feedback_eta_min,
            geometry_feedback_eta_max=self.geometry_feedback_eta_max,
            geometry_feedback_eta_growth=self.geometry_feedback_eta_growth,
            geometry_feedback_eta_decay=self.geometry_feedback_eta_decay,
            geometry_feedback_controller=self.geometry_feedback_controller,
            geometry_feedback_controller_kp=self.geometry_feedback_controller_kp,
            geometry_feedback_controller_ki=self.geometry_feedback_controller_ki,
            geometry_feedback_controller_kd=self.geometry_feedback_controller_kd,
            geometry_feedback_controller_error_ema=self.geometry_feedback_controller_error_ema,
            total_tasks=len(tasks),
            total_tests=total_tests,
        )

        console.print(f"\n[bold]Starting ARC-AGI-{arc_version} Evaluation[/bold]")
        console.print(f"  Model: {self.encoder_model}")
        console.print(f"  ARC Version: {arc_version}")
        console.print(f"  Tasks: {len(tasks)}")
        console.print(f"  Total tests: {total_tests}")
        console.print(f"  Chains: {self.chains}, Generations: {self.generations}")
        console.print()

        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:

            task_progress = progress.add_task(
                "[cyan]Evaluating tasks...",
                total=total_tests
            )

            for task in tasks:
                for test_idx in range(len(task.test_inputs)):
                    progress.update(
                        task_progress,
                        description=f"[cyan]Task {task.task_id} (test {test_idx+1})"
                    )

                    try:
                        result = self.evaluate_task(task, test_idx)
                        results.task_results.append(result)

                        # Update counts
                        if result.baseline_parsed is not None:
                            results.baseline_parsed += 1
                        if result.lr_parsed is not None:
                            results.lr_parsed += 1
                        if result.baseline_correct:
                            results.baseline_correct += 1
                        if result.lr_correct:
                            results.lr_correct += 1

                        if result.error:
                            results.errors.append(f"{task.task_id}: {result.error}")

                    except Exception as e:
                        results.errors.append(f"{task.task_id}: {traceback.format_exc()}")

                    progress.advance(task_progress)

                    # Reset engine to free memory periodically
                    if len(results.task_results) % 10 == 0:
                        self.engine.reset()

        results.total_time = time.time() - start_time

        # Save results
        self._save_results(results)
        self._print_summary(results)

        return results

    def _save_results(self, results: EvaluationResults):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.encoder_model.replace("/", "_")

        output_file = self.output_dir / f"arc_{self.arc_version}_eval_{model_name}_{timestamp}.json"

        # Convert to dict for JSON serialization
        results_dict = {
            "arc_version": self.arc_version,
            "arc_strategy": self.arc_strategy,
            "reasoning_mode": self.reasoning_mode,
            "trajectory_steps": self.trajectory_steps,
            "trajectory_decode_interval": self.trajectory_decode_interval,
            "trajectory_step_scale": self.trajectory_step_scale,
            "geometry_feedback_target_forward_kl": self.geometry_feedback_target_forward_kl,
            "geometry_feedback_kl_tolerance": self.geometry_feedback_kl_tolerance,
            "geometry_feedback_steering_eta": self.geometry_feedback_steering_eta,
            "geometry_feedback_alpha": self.geometry_feedback_alpha,
            "geometry_feedback_kl_cap": self.geometry_feedback_kl_cap,
            "geometry_feedback_topk": self.geometry_feedback_topk,
            "geometry_feedback_eta_min": self.geometry_feedback_eta_min,
            "geometry_feedback_eta_max": self.geometry_feedback_eta_max,
            "geometry_feedback_eta_growth": self.geometry_feedback_eta_growth,
            "geometry_feedback_eta_decay": self.geometry_feedback_eta_decay,
            "geometry_feedback_controller": self.geometry_feedback_controller,
            "geometry_feedback_controller_kp": self.geometry_feedback_controller_kp,
            "geometry_feedback_controller_ki": self.geometry_feedback_controller_ki,
            "geometry_feedback_controller_kd": self.geometry_feedback_controller_kd,
            "geometry_feedback_controller_error_ema": self.geometry_feedback_controller_error_ema,
            "geometry_bandit_profile_stats": self._geometry_bandit_profile_stats,
            "geometry_bandit_total_attempts": self._geometry_bandit_total_attempts,
            "encoder_model": results.encoder_model,
            "timestamp": results.timestamp,
            "total_tasks": results.total_tasks,
            "total_tests": results.total_tests,
            "baseline_correct": results.baseline_correct,
            "lr_correct": results.lr_correct,
            "baseline_parsed": results.baseline_parsed,
            "lr_parsed": results.lr_parsed,
            "baseline_accuracy": results.baseline_correct / results.total_tests if results.total_tests > 0 else 0,
            "lr_accuracy": results.lr_correct / results.total_tests if results.total_tests > 0 else 0,
            "total_time_seconds": results.total_time,
            "errors": results.errors,
            "task_results": [
                {
                    "task_id": r.task_id,
                    "test_index": r.test_index,
                    "expected_output": r.expected_output,
                    "baseline_output": r.baseline_output,
                    "baseline_parsed": r.baseline_parsed,
                    "baseline_correct": r.baseline_correct,
                    "baseline_parse_error": r.baseline_parse_error,
                    "baseline_time": r.baseline_time,
                    "lr_output": r.lr_output,
                    "lr_retries": r.lr_retries,
                    "lr_parse_attempts": r.lr_parse_attempts,
                    "lr_accepted_attempt": r.lr_accepted_attempt,
                    "lr_parsed": r.lr_parsed,
                    "lr_best_partial": r.lr_best_partial,
                    "lr_strategy_trace": r.lr_strategy_trace,
                    "lr_correct": r.lr_correct,
                    "lr_score": r.lr_score,
                    "lr_generations": r.lr_generations,
                    "lr_decode_trace": r.lr_decode_trace,
                    "lr_parse_error": r.lr_parse_error,
                    "lr_time": r.lr_time,
                    "error": r.error,
                }
                for r in results.task_results
            ],
        }

        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        console.print(f"\n[green]Results saved to: {output_file}[/green]")

    def _print_summary(self, results: EvaluationResults):
        """Print evaluation summary."""
        console.print("\n" + "="*60)
        console.print(f"[bold]ARC-AGI-{results.arc_version} EVALUATION RESULTS[/bold]")
        console.print("="*60 + "\n")
        console.print(
            f"[dim]Reasoning mode: {results.reasoning_mode} | "
            f"trajectory_steps: {results.trajectory_steps} | "
            f"trajectory_decode_interval: {results.trajectory_decode_interval} | "
            f"trajectory_step_scale: {results.trajectory_step_scale} | "
            f"geometry_feedback_controller: {results.geometry_feedback_controller}[/dim]"
        )

        table = Table(title="Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Baseline", justify="right")
        table.add_column("Latent Reasoning", justify="right")

        total = results.total_tests

        table.add_row(
            "Correct",
            f"{results.baseline_correct}/{total}",
            f"{results.lr_correct}/{total}"
        )
        table.add_row(
            "Accuracy",
            f"{100*results.baseline_correct/total:.1f}%" if total > 0 else "N/A",
            f"{100*results.lr_correct/total:.1f}%" if total > 0 else "N/A"
        )
        table.add_row(
            "Parsed Successfully",
            f"{results.baseline_parsed}/{total}",
            f"{results.lr_parsed}/{total}"
        )
        table.add_row(
            "Parse Success",
            f"{100*results.baseline_parsed/total:.1f}%" if total > 0 else "N/A",
            f"{100*results.lr_parsed/total:.1f}%" if total > 0 else "N/A",
        )
        avg_attempts = (
            sum(getattr(r, "lr_parse_attempts", 0) for r in results.task_results) / total
            if total > 0 else 0.0
        )
        table.add_row(
            "LR Attempts / Test",
            "n/a",
            f"{avg_attempts:.2f}"
        )
        avg_best_partial = (
            sum(getattr(r, "lr_best_partial", 0.0) for r in results.task_results) / total
            if total > 0 else 0.0
        )
        table.add_row(
            "LR Avg Best Partial",
            "n/a",
            f"{avg_best_partial:.3f}"
        )

        console.print(table)

        console.print(f"\n[dim]Total time: {results.total_time:.1f}s[/dim]")
        console.print(f"[dim]Errors: {len(results.errors)}[/dim]")

        # Show winner
        if results.lr_correct > results.baseline_correct:
            console.print("\n[bold green]WINNER: Latent Space Reasoning[/bold green]")
        elif results.baseline_correct > results.lr_correct:
            console.print("\n[bold yellow]WINNER: Baseline[/bold yellow]")
        else:
            console.print("\n[bold blue]TIE[/bold blue]")


# =============================================================================
# CLI Entry Point
# =============================================================================

def run_arc_evaluation(
    encoder: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    max_tasks: Optional[int] = None,
    chains: int = 4,
    generations: int = 5,
    max_tokens: int = 512,
    decode_mode: str = "soft_prompt",
    data_dir: str = "./data",
    output_dir: str = "./eval_results",
    arc_version: str = "3",
    lr_retries: int = 1,
    arc_strategy: str = "adaptive",
    reasoning_mode: str = "evolution",
    trajectory_steps: int = 6,
    trajectory_decode_interval: int = 0,
    trajectory_step_scale: float = 0.2,
    geometry_feedback_target_forward_kl: float = 0.06,
    geometry_feedback_kl_tolerance: float = 0.5,
    geometry_feedback_steering_eta: float = 0.05,
    geometry_feedback_alpha: float = 0.01,
    geometry_feedback_kl_cap: float = 0.5,
    geometry_feedback_topk: int = 50,
    geometry_feedback_eta_min: float = 0.01,
    geometry_feedback_eta_max: float = 0.5,
    geometry_feedback_eta_growth: float = 1.06,
    geometry_feedback_eta_decay: float = 0.85,
    geometry_feedback_controller: str = "legacy",
    geometry_feedback_controller_kp: float = 0.0,
    geometry_feedback_controller_ki: float = 0.0,
    geometry_feedback_controller_kd: float = 0.0,
    geometry_feedback_controller_error_ema: float = 0.2,
) -> EvaluationResults:
    """
    Run ARC-AGI evaluation.

    Args:
        encoder: HuggingFace model ID for encoder
        max_tasks: Limit number of tasks (None = all ~400)
        chains: Number of evolution chains
        generations: Max generations per task
        max_tokens: Max output tokens
        decode_mode: Decode mode used during latent reasoning
        data_dir: Directory for dataset
        output_dir: Directory for results
        arc_version: ARC dataset version. Use '2' for ARC-AGI-2, '3' for ARC-AGI-3.
        lr_retries: Number of additional latent-reasoning attempts per task when output is malformed
        arc_strategy: ARC reasoning strategy for LR attempts:
            single, adaptive, repair, consensus, geometry_bandit, self_improving
        reasoning_mode: Reasoning mode for latent pipeline.
            One of evolution, trajectory, hybrid
        trajectory_steps: Number of trajectory refinement steps for trajectory mode.
        trajectory_decode_interval: Steps between trajectory preview decodes (0 disables).
        trajectory_step_scale: Scale for trajectory movement updates.
        geometry_feedback_target_forward_kl: Geometry-feedback target forward KL.
        geometry_feedback_kl_tolerance: Geometry-feedback KL tolerance band.
        geometry_feedback_steering_eta: Geometry-feedback steering learning-rate.
        geometry_feedback_alpha: Geometry-feedback EMA momentum.
        geometry_feedback_kl_cap: Geometry-feedback KL cap.
        geometry_feedback_topk: Geometry-feedback token top-k cutoff.
        geometry_feedback_eta_min: Geometry-feedback minimum step size.
        geometry_feedback_eta_max: Geometry-feedback maximum step size.
        geometry_feedback_eta_growth: Geometry-feedback step-size growth factor.
        geometry_feedback_eta_decay: Geometry-feedback step-size decay factor.
        geometry_feedback_controller: Geometry-feedback controller mode.
        geometry_feedback_controller_kp: Controller proportional gain.
        geometry_feedback_controller_ki: Controller integral gain.
        geometry_feedback_controller_kd: Controller derivative gain.
        geometry_feedback_controller_error_ema: Controller error smoothing alpha.

    Returns:
        EvaluationResults with full details
    """
    if decode_mode not in {
        "seed",
        "soft_prompt",
        "dual_steering",
        "intermediate_steering",
        "geometry_feedback",
    }:
        raise ValueError(
            "Unsupported decode_mode. Valid options: seed, soft_prompt, "
            "dual_steering, intermediate_steering, geometry_feedback"
        )

    evaluator = ARCEvaluator(
        encoder_model=encoder,
        chains=chains,
        generations=generations,
        max_tokens=max_tokens,
        decode_mode=decode_mode,
        data_dir=data_dir,
        output_dir=output_dir,
        arc_version=arc_version,
        lr_retries=lr_retries,
        arc_strategy=arc_strategy,
        reasoning_mode=reasoning_mode,
        trajectory_steps=trajectory_steps,
        trajectory_decode_interval=trajectory_decode_interval,
        trajectory_step_scale=trajectory_step_scale,
        geometry_feedback_target_forward_kl=geometry_feedback_target_forward_kl,
        geometry_feedback_kl_tolerance=geometry_feedback_kl_tolerance,
        geometry_feedback_steering_eta=geometry_feedback_steering_eta,
        geometry_feedback_alpha=geometry_feedback_alpha,
        geometry_feedback_kl_cap=geometry_feedback_kl_cap,
        geometry_feedback_topk=geometry_feedback_topk,
        geometry_feedback_eta_min=geometry_feedback_eta_min,
        geometry_feedback_eta_max=geometry_feedback_eta_max,
        geometry_feedback_eta_growth=geometry_feedback_eta_growth,
        geometry_feedback_eta_decay=geometry_feedback_eta_decay,
        geometry_feedback_controller=geometry_feedback_controller,
        geometry_feedback_controller_kp=geometry_feedback_controller_kp,
        geometry_feedback_controller_ki=geometry_feedback_controller_ki,
        geometry_feedback_controller_kd=geometry_feedback_controller_kd,
        geometry_feedback_controller_error_ema=geometry_feedback_controller_error_ema,
        )

    return evaluator.run_evaluation(max_tasks=max_tasks)


if __name__ == "__main__":
    # Quick test with 5 tasks
    run_arc_evaluation(max_tasks=5)
