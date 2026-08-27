"""
Main CLI application for latent space reasoning.

This module provides the command-line interface for the Latent Space Reasoning Engine.
It includes commands for running reasoning, comparing methods, checking system status,
and more. All commands include helpful error messages and progress indicators.

Commands:
- run: Run latent space reasoning on a query
- compare: Compare baseline vs latent reasoning side-by-side
- baseline: Run only baseline generation
- models: List recommended models
- check-gpu: Check GPU availability and system info
- version: Show version information
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

app = typer.Typer(
    name="latent-reason",
    help="""
Latent Space Reasoning Engine

Evolutionary optimization in latent space for higher-quality LLM responses.

Instead of generating text directly, this engine:
• Encodes your query into latent space
• Evolves the representation through selection, mutation, crossover
• Scores candidates using trained neural networks
• Decodes the best result to structured, specific text

Quick start:
  latent-reason compare "How to implement authentication?"
  latent-reason run "Design a REST API" --encoder Qwen/Qwen3-4B

For help with any command: latent-reason COMMAND --help
""",
    add_completion=False,
    no_args_is_help=True,
)

# Use safe box characters for Windows
console = Console(force_terminal=True, legacy_windows=True)


def _sanitize_text(text: str) -> str:
    """Sanitize text for Windows console compatibility."""
    if sys.platform != "win32":
        return text  # Linux/Mac handle Unicode fine
    # On Windows, always sanitize - even Windows Terminal can have issues
    # with Rich's legacy_windows mode and certain console configurations

    # Build output character by character, replacing non-ASCII
    result = []
    for char in text:
        code = ord(char)
        if code < 128:
            # ASCII is safe
            result.append(char)
        elif code < 256:
            # Extended ASCII - try to keep if encodable
            try:
                char.encode("cp1252")
                result.append(char)
            except UnicodeEncodeError:
                result.append("?")
        else:
            # Unicode - replace common ones, otherwise use ?
            replacements = {
                0x2705: "[OK]",   # check mark
                0x274c: "[X]",    # cross mark
                0x2713: "[OK]",   # checkmark
                0x2717: "[X]",    # X mark
                0x2022: "-",      # bullet
                0x2728: "*",      # sparkles
                0x26a0: "[!]",    # warning
                0x2b50: "*",      # star
                0x2699: "[*]",    # gear
                0x2757: "[!]",    # exclamation
                0x2753: "[?]",    # question
                0x27a1: "->",     # arrow
                0x2714: "[OK]",   # check
                0x2716: "[X]",    # X
                0x1f310: "[*]",   # globe
                0x1f4e6: "[*]",   # package
                0x1f527: "[*]",   # wrench
                0x1f4a1: "[i]",   # lightbulb
                0x1f6e0: "[*]",   # hammer/wrench
                0x1f4cc: "[>]",   # pushpin
                0x1f50d: "[?]",   # magnifying glass
                0x1f389: "[!]",   # party popper
                0x1f60a: ":)",    # smile
                0x1f600: ":)",    # grinning
                0x1f4dd: "[>]",   # memo
                0x1f680: "[>]",   # rocket
                0x1f512: "[>]",   # lock
                0x1f511: "[>]",   # key
                0x1f4bb: "[PC]",  # laptop
                0x1f4f1: "[>]",   # mobile
                0x1f517: "[>]",   # link
                0x1f4ca: "[>]",   # chart
                0x1f3af: "[>]",   # target
                0x1f9e9: "[>]",   # puzzle
                0x1f4e1: "[>]",   # antenna
                0x2194: "<->",    # left-right arrow
                0x2195: "^v",     # up-down arrow
                0x2b06: "^",      # up arrow
                0x2b07: "v",      # down arrow
                0x25b6: ">",      # play
                0x25c0: "<",      # rewind
                0x2139: "[i]",    # info
                0x1f4c1: "[>]",   # folder
                0x1f4c2: "[>]",   # folder open
                0x1f4c4: "[>]",   # document
                0x1f5a5: "[PC]",  # desktop
                0x1f578: "[*]",   # spider web
                0x1f504: "[*]",   # refresh
            }
            result.append(replacements.get(code, ""))

    return "".join(result)


def safe_print(text: str) -> None:
    """Print text safely on all platforms."""
    try:
        console.print(_sanitize_text(text))
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


# ============================================================================
# Main Commands
# ============================================================================


@app.command()
def run(
    query: str = typer.Argument(..., help="The question or problem to reason about"),

    # Model selection
    encoder: str = typer.Option(
        "Qwen/Qwen3-0.6B",
        "--encoder", "-e",
        help="Model for encoding/decoding. Options: Qwen/Qwen3-4B (best quality, ~8GB), Qwen/Qwen3-1.7B (balanced, ~4GB), Qwen/Qwen3-0.6B (fast, ~2GB)",
    ),
    quantization: str = typer.Option(
        "4bit",
        "--quantization",
        help="Quantization mode: auto, 4bit, none",
    ),
    decode_strategy: str = typer.Option(
        "best",
        "--decode-strategy",
        help="Decode strategy: best, combined",
    ),
    decode_mode: str = typer.Option(
        "soft_prompt",
        "--decode-mode",
        help="Decode control mode: seed, soft_prompt, dual_steering, intermediate_steering, geometry_feedback",
    ),
    reasoning_mode: str = typer.Option(
        "evolution",
        "--reasoning-mode",
        help="Reasoning strategy: evolution, trajectory, hybrid",
    ),
    trajectory_steps: int = typer.Option(
        6,
        "--trajectory-steps",
        help="Number of trajectory refinement steps when trajectory mode is enabled",
        min=1,
        max=200,
    ),
    trajectory_decode_interval: int = typer.Option(
        0,
        "--trajectory-decode-interval",
        help="Decode preview interval during trajectory steps (0 disables previews)",
        min=0,
        max=20,
    ),
    trajectory_step_scale: float = typer.Option(
        0.2,
        "--trajectory-step-scale",
        help="Step scale for trajectory movement updates",
        min=0.0,
        max=5.0,
    ),
    geometry_feedback_target_forward_kl: float = typer.Option(
        0.06,
        "--geometry-feedback-target-forward-kl",
        help="Target forward-KL for geometry feedback decoding.",
        min=0.0,
    ),
    geometry_feedback_kl_tolerance: float = typer.Option(
        0.5,
        "--geometry-feedback-kl-tolerance",
        help="KL tolerance band for geometry feedback decoder steering.",
        min=0.0,
    ),
    geometry_feedback_steering_eta: float = typer.Option(
        0.05,
        "--geometry-feedback-steering-eta",
        help="Learning-rate-like step size for geometry feedback steering.",
        min=0.0,
        max=1.0,
    ),
    geometry_feedback_alpha: float = typer.Option(
        0.01,
        "--geometry-feedback-alpha",
        help="EMA momentum for geometry feedback steering updates.",
        min=0.0,
        max=1.0,
    ),
    geometry_feedback_kl_cap: float = typer.Option(
        0.5,
        "--geometry-feedback-kl-cap",
        help="KL cap for geometry feedback updates.",
        min=0.0,
    ),
    geometry_feedback_topk: int = typer.Option(
        50,
        "--geometry-feedback-topk",
        help="Top-k logits to consider for geometry feedback.",
        min=1,
    ),
    geometry_feedback_eta_min: float = typer.Option(
        0.01,
        "--geometry-feedback-eta-min",
        help="Minimum geometry steering step size.",
        min=0.0,
        max=1.0,
    ),
    geometry_feedback_eta_max: float = typer.Option(
        0.5,
        "--geometry-feedback-eta-max",
        help="Maximum geometry steering step size.",
        min=0.0,
        max=1.0,
    ),
    geometry_feedback_eta_growth: float = typer.Option(
        1.06,
        "--geometry-feedback-eta-growth",
        help="Growth factor when steering error is low.",
        min=1.0,
    ),
    geometry_feedback_eta_decay: float = typer.Option(
        0.85,
        "--geometry-feedback-eta-decay",
        help="Decay factor when steering error is high.",
        min=0.0,
        max=1.0,
    ),
    geometry_feedback_controller: str = typer.Option(
        "legacy",
        "--geometry-feedback-controller",
        help="Geometry controller type: legacy or pid",
    ),
    geometry_feedback_controller_kp: float = typer.Option(
        0.0,
        "--geometry-feedback-controller-kp",
        help="Proportional gain for PID controller.",
        min=0.0,
        max=2.0,
    ),
    geometry_feedback_controller_ki: float = typer.Option(
        0.0,
        "--geometry-feedback-controller-ki",
        help="Integral gain for PID controller.",
        min=0.0,
        max=2.0,
    ),
    geometry_feedback_controller_kd: float = typer.Option(
        0.0,
        "--geometry-feedback-controller-kd",
        help="Derivative gain for PID controller.",
        min=0.0,
        max=2.0,
    ),
    geometry_feedback_controller_error_ema: float = typer.Option(
        0.2,
        "--geometry-feedback-controller-error-ema",
        help="EMA smoothing for controller error signal.",
        min=0.0,
        max=1.0,
    ),
    scorer: Optional[List[str]] = typer.Option(
        None,
        "--scorer", "-s",
        help="Additional scorer models for quality evaluation (advanced usage)",
    ),
    modifier: Optional[List[str]] = typer.Option(
        None,
        "--modifier", "-m",
        help="Modifier models for suggesting improvements (advanced usage)",
    ),

    # Text generation settings
    max_tokens: int = typer.Option(
        2048,
        "--max-tokens", "-t",
        help="Maximum output length in tokens (1-32768). Higher = longer responses but slower",
        min=1,
        max=32768,
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        help="Sampling randomness (0.0-2.0). Lower = more focused, higher = more creative",
        min=0.0,
        max=2.0,
    ),
    # Evolution parameters (advanced)
    chains: int = typer.Option(
        5,
        "--chains", "-c",
        help="Parallel evolution chains (more = better exploration, slower). Recommended: 5-8",
        min=1,
        max=50,
    ),
    generations: int = typer.Option(
        10,
        "--generations", "-g",
        help="Maximum evolution generations (more = better refinement, slower). Most queries converge in 10-20",
        min=1,
        max=100,
    ),
    mutation_temp: float = typer.Option(
        0.5,
        "--mutation-temp",
        help="Mutation strength (0.3=focused, 0.5=balanced, 0.8=exploratory)",
        min=0.0,
        max=2.0,
    ),
    survivors: int = typer.Option(
        5,
        "--survivors",
        help="Chains kept each generation (should be <= chains)",
        min=1,
    ),

    # Configuration and output
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="YAML config file (see config.example.yaml). Overrides command-line options",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save detailed results to JSON file for analysis",
    ),
    format: str = typer.Option(
        "text",
        "--format", "-f",
        help="Output format: text (human-readable), json (structured), markdown (formatted)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug output (very detailed)",
    ),
    silent: bool = typer.Option(
        False,
        "--silent",
        help="Suppress all output except final result",
    ),
):
    """
    Run latent space reasoning on a query.

    This command uses evolutionary optimization in latent space to generate
    higher-quality, more specific responses than standard LLM generation.

    The process:
    1. Encodes your query into the model's latent space
    2. Evolves the representation through multiple generations
    3. Scores candidates using trained neural networks
    4. Decodes the best result to structured text

    Tips:
    • Start with default settings, then experiment with larger models
    • Use --chains 8 --generations 15 for complex queries
    • Try --encoder Qwen/Qwen3-4B for best quality (needs ~8GB VRAM)
    • Use --verbose to see evolution progress

    Examples:
        # Basic usage
        latent-reason run "How to implement user authentication?"

        # High quality with larger model
        latent-reason run "Design a REST API" --encoder Qwen/Qwen3-4B --chains 8

        # Quick iteration with small model
        latent-reason run "Debug memory leak" --encoder Qwen/Qwen3-0.6B --temperature 0.5

        # Save results for analysis
        latent-reason run "Create test plan" --output results.json --format json
    """
    from latent_reasoning.config import Config, ScorerConfig, ModifierConfig
    from latent_reasoning.engine import Engine

    # Determine verbosity
    verbosity = "normal"
    if silent:
        verbosity = "silent"
    elif debug:
        verbosity = "debug"
    elif verbose:
        verbosity = "verbose"

    # Load or create config
    if config:
        cfg = Config.from_yaml(config)
    else:
        cfg = Config()

    # Override config with CLI options
    cfg.encoder.model = encoder
    cfg.encoder.quantization = quantization
    if decode_strategy not in {"best", "combined"}:
        console.print("[red]Error: --decode-strategy must be 'best' or 'combined'[/red]")
        raise typer.Exit(1)
    cfg.synthesis.decode_strategy = decode_strategy
    if decode_mode not in {"seed", "soft_prompt", "dual_steering", "intermediate_steering", "geometry_feedback"}:
        console.print(
            "[red]Error: --decode-mode must be one of: "
            "seed, soft_prompt, dual_steering, intermediate_steering, geometry_feedback[/red]"
        )
        raise typer.Exit(1)
    cfg.synthesis.decode_mode = decode_mode
    if reasoning_mode not in {"evolution", "trajectory", "hybrid"}:
        console.print(
            "[red]Error: --reasoning-mode must be one of: "
            "evolution, trajectory, hybrid[/red]"
        )
        raise typer.Exit(1)
    cfg.synthesis.reasoning_mode = reasoning_mode
    cfg.synthesis.trajectory_steps = trajectory_steps
    cfg.synthesis.trajectory_decode_interval = trajectory_decode_interval
    cfg.synthesis.trajectory_step_scale = trajectory_step_scale
    cfg.synthesis.geometry_feedback_target_forward_kl = geometry_feedback_target_forward_kl
    cfg.synthesis.geometry_feedback_kl_tolerance = geometry_feedback_kl_tolerance
    cfg.synthesis.geometry_feedback_steering_eta = geometry_feedback_steering_eta
    cfg.synthesis.geometry_feedback_alpha = geometry_feedback_alpha
    cfg.synthesis.geometry_feedback_kl_cap = geometry_feedback_kl_cap
    cfg.synthesis.geometry_feedback_topk = geometry_feedback_topk
    cfg.synthesis.geometry_feedback_eta_min = geometry_feedback_eta_min
    cfg.synthesis.geometry_feedback_eta_max = geometry_feedback_eta_max
    cfg.synthesis.geometry_feedback_eta_growth = geometry_feedback_eta_growth
    cfg.synthesis.geometry_feedback_eta_decay = geometry_feedback_eta_decay
    cfg.synthesis.geometry_feedback_controller = geometry_feedback_controller
    cfg.synthesis.geometry_feedback_controller_kp = geometry_feedback_controller_kp
    cfg.synthesis.geometry_feedback_controller_ki = geometry_feedback_controller_ki
    cfg.synthesis.geometry_feedback_controller_kd = geometry_feedback_controller_kd
    cfg.synthesis.geometry_feedback_controller_error_ema = geometry_feedback_controller_error_ema
    cfg.evolution.chains = chains
    cfg.evolution.generations = generations
    cfg.evolution.temperature = mutation_temp
    cfg.evolution.selection.survivors = survivors
    cfg.synthesis.max_tokens = max_tokens
    cfg.synthesis.temperature = temperature
    cfg.output.verbosity = verbosity
    cfg.output.format = format

    if scorer:
        cfg.judges.scorers = [ScorerConfig(type="semantic", model=s) for s in scorer]
    if modifier:
        cfg.judges.modifiers = [ModifierConfig(model=m) for m in modifier]

    # Validate parameters
    if chains < survivors:
        console.print("[red]Error: --survivors cannot be greater than --chains[/red]")
        console.print(f"[dim]You specified {survivors} survivors but only {chains} chains[/dim]")
        raise typer.Exit(1)

    # Run reasoning with helpful error messages
    try:
        if not silent:
            console.print(f"[dim]Initializing Latent Space Reasoning Engine...[/dim]")
            console.print(f"[dim]   Model: {encoder}[/dim]")
            console.print(f"[dim]   Chains: {chains}, Generations: {generations}[/dim]")

        engine = Engine(config=cfg)
        result = engine.run(query)

        # Format output based on format option
        if format == "json":
            output_data = {
                "query": query,
                "plan": _sanitize_text(result.plan),
                "confidence": result.confidence,
                "generations": result.generations,
                "evaluations": result.evaluations,
                "stop_reason": result.stop_reason,
                "reasoning_mode": result.reasoning_mode,
                "reasoning_trace": result.reasoning_trace,
                "all_plans": [_sanitize_text(p) for p in result.all_plans],
                "decode_trace": result.decode_trace,
            }
            console.print_json(data=output_data)
        elif format == "markdown":
            safe_print(f"# Result\n\n{result.plan}")
        # text format is handled by the engine's print_result

        # Save to file if requested
        if output:
            output_data = {
                "query": query,
                "plan": result.plan,  # Keep original for file (supports UTF-8)
                "confidence": result.confidence,
                "generations": result.generations,
                "evaluations": result.evaluations,
                "stop_reason": result.stop_reason,
                "reasoning_mode": result.reasoning_mode,
                "reasoning_trace": result.reasoning_trace,
                "all_plans": result.all_plans,
                "decode_trace": result.decode_trace,
            }
            with open(output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            console.print(f"\n[dim]Results saved to {output}[/dim]")

        # Provide helpful feedback
        if not silent and format == "text":
            if result.confidence > 0.8:
                console.print(f"\n[green]High quality result! (confidence: {result.confidence:.3f})[/green]")
            elif result.confidence > 0.6:
                console.print(f"\n[blue]Good result (confidence: {result.confidence:.3f})[/blue]")
            else:
                console.print(f"\n[yellow]Lower quality result (confidence: {result.confidence:.3f})[/yellow]")
                console.print("[dim]   Try: --encoder Qwen/Qwen3-4B --chains 8 --generations 15[/dim]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(130)

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("[dim]Try: pip install -e .[/dim]")
        raise typer.Exit(1)

    except FileNotFoundError as e:
        if "checkpoint" in str(e).lower():
            console.print(f"[red]Missing trained model checkpoint[/red]")
            console.print(f"[dim]   Expected: {e}[/dim]")
            console.print("[dim]Make sure you have the complete repository with checkpoints/[/dim]")
        else:
            console.print(f"[red]File not found: {e}[/red]")
        raise typer.Exit(1)

    except RuntimeError as e:
        error_msg = str(e)
        if "cuda" in error_msg.lower() or "gpu" in error_msg.lower():
            console.print(f"[red]GPU/CUDA error: {error_msg}[/red]")
            console.print("[dim]Try: --encoder Qwen/Qwen3-0.6B (smaller model) or use CPU[/dim]")
            console.print("[dim]   Check GPU with: latent-reason check-gpu[/dim]")
        elif "memory" in error_msg.lower() or "out of memory" in error_msg.lower():
            console.print(f"[red]Out of memory: {error_msg}[/red]")
            console.print("[dim]Try: --encoder Qwen/Qwen3-0.6B --chains 3 --max-tokens 1024[/dim]")
        else:
            console.print(f"[red]Runtime error: {_sanitize_text(error_msg)}[/red]")
        raise typer.Exit(1)

    except UnicodeEncodeError:
        # Output was shown but console had encoding issue - not a real error
        console.print("\n[dim]Note: Some special characters could not be displayed[/dim]")

    except Exception as e:
        error_msg = str(e)
        if "charmap" in error_msg or "encode" in error_msg.lower():
            # Unicode encoding issue in console - output was likely shown
            console.print("\n[dim]Note: Some special characters could not be displayed[/dim]")
        else:
            console.print(f"[red]Unexpected error: {_sanitize_text(error_msg)}[/red]")
            console.print("[dim]For help: latent-reason --help[/dim]")
            if not silent:
                console.print(f"[dim]   Query was: {query[:100]}{'...' if len(query) > 100 else ''}[/dim]")
            raise typer.Exit(1)
            if debug:
                console.print_exception()
            raise typer.Exit(1)


@app.command()
def compare(
    query: str = typer.Argument(..., help="The query to compare on"),
    # Model options
    encoder: Optional[str] = typer.Option(
        None,
        "--encoder", "-e",
        help="Encoder model (HuggingFace ID or path). Overrides config if set.",
    ),
    quantization: Optional[str] = typer.Option(
        None,
        "--quantization",
        help="Quantization mode: auto, 4bit, none. Overrides config if set.",
    ),
    decode_strategy: Optional[str] = typer.Option(
        None,
        "--decode-strategy",
        help="Decode strategy: best, combined. Overrides config if set.",
    ),
    decode_mode: Optional[str] = typer.Option(
        None,
        "--decode-mode",
        help="Decode control mode: seed, soft_prompt, dual_steering, intermediate_steering, geometry_feedback. "
             "Overrides config if set.",
    ),
    reasoning_mode: Optional[str] = typer.Option(
        None,
        "--reasoning-mode",
        help="Reasoning strategy: evolution, trajectory, hybrid. Overrides config if set.",
    ),
    trajectory_steps: Optional[int] = typer.Option(
        None,
        "--trajectory-steps",
        help="Number of trajectory refinement steps when trajectory reasoning is enabled.",
        min=1,
        max=200,
    ),
    trajectory_decode_interval: Optional[int] = typer.Option(
        None,
        "--trajectory-decode-interval",
        help="Decode preview interval during trajectory steps (0 disables previews).",
        min=0,
        max=20,
    ),
    trajectory_step_scale: Optional[float] = typer.Option(
        None,
        "--trajectory-step-scale",
        help="Step scale for trajectory movement updates.",
        min=0.0,
        max=5.0,
    ),
    # Generation options
    max_tokens: Optional[int] = typer.Option(
        None,
        "--max-tokens", "-t",
        help="Maximum tokens to generate. Overrides config if set.",
        min=1,
        max=32768,
    ),
    temperature: Optional[float] = typer.Option(
        None,
        "--temperature",
        help="Sampling temperature. Overrides config if set.",
        min=0.0,
        max=2.0,
    ),
    # Evolution options
    chains: Optional[int] = typer.Option(
        None,
        "--chains", "-c",
        help="Number of parallel reasoning chains. Overrides config if set.",
    ),
    generations: Optional[int] = typer.Option(
        None,
        "--generations", "-g",
        help="Maximum evolution generations. Overrides config if set.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="YAML config profile for reproducible runs (e.g., low-resource profile)",
        exists=True,
    ),
    # Output options
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save results to JSON file",
    ),
    full: bool = typer.Option(
        True,
        "--full/--brief",
        help="Show full outputs (default) or brief summary",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output",
    ),
):
    """
    Compare baseline generation vs latent space reasoning.

    Runs both methods on the same query and displays a side-by-side comparison
    showing the full output from each approach.

    Examples:
        latent-reason compare "How to optimize a database query?"
        latent-reason compare "Design an API" --encoder Qwen/Qwen3-4B
        latent-reason compare "Design an API" --config configs/aim_v1_low_resource.yaml
        latent-reason compare "Debug this issue" --full --output results.json
    """
    from latent_reasoning.config import Config
    from latent_reasoning.engine import Engine

    cfg = Config.from_yaml(config) if config else Config()

    # Sensible compare defaults when no config profile is provided.
    if not config:
        cfg.encoder.model = "Qwen/Qwen3-0.6B"
        cfg.encoder.quantization = "4bit"
        cfg.synthesis.decode_strategy = "best"
        cfg.evolution.chains = 5
        cfg.evolution.generations = 10
        cfg.synthesis.max_tokens = 2048
        cfg.synthesis.temperature = 0.7
        cfg.output.verbosity = "minimal"

    if encoder is not None:
        cfg.encoder.model = encoder
    if quantization is not None:
        cfg.encoder.quantization = quantization
    if decode_strategy is not None:
        cfg.synthesis.decode_strategy = decode_strategy
    if decode_mode is not None:
        if decode_mode not in {"seed", "soft_prompt", "dual_steering", "intermediate_steering", "geometry_feedback"}:
            console.print(
                "[red]Error: --decode-mode must be one of: "
                "seed, soft_prompt, dual_steering, intermediate_steering, geometry_feedback[/red]"
            )
            raise typer.Exit(1)
        cfg.synthesis.decode_mode = decode_mode
    if reasoning_mode is not None:
        if reasoning_mode not in {"evolution", "trajectory", "hybrid"}:
            console.print(
                "[red]Error: --reasoning-mode must be one of: "
                "evolution, trajectory, hybrid[/red]"
            )
            raise typer.Exit(1)
        cfg.synthesis.reasoning_mode = reasoning_mode
    if trajectory_steps is not None:
        cfg.synthesis.trajectory_steps = trajectory_steps
    if trajectory_decode_interval is not None:
        cfg.synthesis.trajectory_decode_interval = trajectory_decode_interval
    if trajectory_step_scale is not None:
        cfg.synthesis.trajectory_step_scale = trajectory_step_scale
    if chains is not None:
        cfg.evolution.chains = chains
    if generations is not None:
        cfg.evolution.generations = generations
    if max_tokens is not None:
        cfg.synthesis.max_tokens = max_tokens
    if temperature is not None:
        cfg.synthesis.temperature = temperature

    if cfg.synthesis.decode_strategy not in {"best", "combined"}:
        console.print("[red]Error: --decode-strategy must be 'best' or 'combined'[/red]")
        raise typer.Exit(1)

    if verbose:
        cfg.output.verbosity = "verbose"

    try:
        console.print(f"\n[bold cyan]Comparing reasoning methods...[/bold cyan]")
        console.print(f"[dim]Query: {query}[/dim]\n")

        engine = Engine(config=cfg)
        result = engine.compare(query)

        # Display comparison
        console.print("\n[bold]=== COMPARISON RESULTS ===[/bold]\n")

        # Query panel
        console.print(Panel(query, title="[bold]Query[/bold]", border_style="blue"))

        # Baseline output (sanitize for Windows console)
        baseline_text = _sanitize_text(result["baseline"])
        baseline_display = baseline_text if full else (baseline_text[:500] + "..." if len(baseline_text) > 500 else baseline_text)
        console.print(Panel(
            baseline_display,
            title="[bold yellow]Baseline (Direct Generation)[/bold yellow]",
            border_style="yellow",
        ))

        # Latent reasoning output (sanitize for Windows console)
        latent_text = _sanitize_text(result["latent_reasoning"])
        latent_display = latent_text if full else (latent_text[:500] + "..." if len(latent_text) > 500 else latent_text)
        console.print(Panel(
            latent_display,
            title="[bold green]Latent Space Reasoning[/bold green]",
            border_style="green",
        ))

        # Statistics
        stats_table = Table(show_header=False, box=None)
        stats_table.add_column("Metric", style="dim")
        stats_table.add_column("Value", style="bold")
        stats_table.add_row("Final Score", f"{result['latent_score']:.3f}")
        stats_table.add_row("Generations", str(result['generations']))
        stats_table.add_row("Evaluations", str(result['evaluations']))
        stats_table.add_row("Baseline Length", f"{len(result['baseline'])} chars")
        stats_table.add_row("Latent Length", f"{len(result['latent_reasoning'])} chars")
        if "baseline_duration_s" in result:
            stats_table.add_row("Baseline Time", f"{result['baseline_duration_s']:.2f}s")
        if "latent_duration_s" in result:
            stats_table.add_row("Latent Time", f"{result['latent_duration_s']:.2f}s")
        if "latent_evolution_duration_s" in result:
            stats_table.add_row("Evolution Time", f"{result['latent_evolution_duration_s']:.2f}s")
        if "latent_non_evolution_duration_s" in result:
            stats_table.add_row("Non-Evolution Time", f"{result['latent_non_evolution_duration_s']:.2f}s")
        if "latency_overhead_ratio" in result and result["latency_overhead_ratio"] is not None:
            stats_table.add_row("Latency Overhead", f"{result['latency_overhead_ratio']:.2f}x")

        console.print(Panel(stats_table, title="[bold]Statistics[/bold]", border_style="dim"))

        # Save if requested
        if output:
            with open(output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            console.print(f"\n[dim]Results saved to {output}[/dim]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(130)
    except UnicodeEncodeError:
        # Output was shown but console had encoding issue
        console.print("\n[dim]Note: Some special characters could not be displayed[/dim]")
    except Exception as e:
        error_msg = str(e)
        if "charmap" in error_msg or "encode" in error_msg.lower():
            console.print("\n[dim]Note: Some special characters could not be displayed[/dim]")
        else:
            console.print(f"[red]Error: {_sanitize_text(error_msg)}[/red]")
            raise typer.Exit(1)


@app.command()
def baseline(
    query: str = typer.Argument(..., help="The query to process"),
    encoder: str = typer.Option(
        "Qwen/Qwen3-0.6B",
        "--encoder", "-e",
        help="Encoder model",
    ),
    quantization: str = typer.Option(
        "4bit",
        "--quantization",
        help="Quantization mode: auto, 4bit, none",
    ),
    max_tokens: int = typer.Option(
        2048,
        "--max-tokens", "-t",
        help="Maximum tokens to generate",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        help="Sampling temperature",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save result to file",
    ),
):
    """
    Run baseline generation without latent space evolution.

    Useful for quick generation or comparing against the full pipeline.

    Examples:
        latent-reason baseline "How to implement caching?"
        latent-reason baseline "Design pattern for X" --max-tokens 1024
    """
    from latent_reasoning.config import Config
    from latent_reasoning.engine import Engine

    cfg = Config()
    cfg.encoder.model = encoder
    cfg.encoder.quantization = quantization
    cfg.synthesis.max_tokens = max_tokens
    cfg.synthesis.temperature = temperature
    cfg.output.verbosity = "silent"

    try:
        engine = Engine(config=cfg)
        result = engine.run_baseline(query)

        # Sanitize for Windows console
        safe_result = _sanitize_text(result)

        try:
            console.print(Panel(safe_result, title="[bold]Baseline Result[/bold]"))
        except UnicodeEncodeError:
            # Fallback to simple print if Panel fails
            print("\n=== Baseline Result ===")
            print(safe_result.encode('ascii', errors='replace').decode('ascii'))
            print("=" * 24)

        if output:
            output_data = {"query": query, "result": result}
            with open(output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            console.print(f"\n[dim]Saved to {output}[/dim]")

    except UnicodeEncodeError:
        # Just note that there were encoding issues but result was shown
        console.print("\n[dim]Note: Some characters could not be displayed[/dim]")
    except Exception as e:
        error_msg = str(e)
        if "charmap" in error_msg or "encode" in error_msg.lower():
            console.print("\n[dim]Note: Some characters could not be displayed[/dim]")
        else:
            console.print(f"[red]Error: {_sanitize_text(error_msg)}[/red]")
            raise typer.Exit(1)


# ============================================================================
# Utility Commands
# ============================================================================


@app.command("check-gpu")
def check_gpu():
    """Check GPU availability and display system info."""
    import torch

    console.print("\n[bold]System Information[/bold]\n")

    # Python info
    import sys
    console.print(f"Python: {sys.version.split()[0]}")
    console.print(f"PyTorch: {torch.__version__}")

    # GPU info
    if torch.cuda.is_available():
        console.print("\n[green][+] CUDA Available[/green]")
        for i in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(i)
            memory_gb = device.total_memory / (1024**3)
            console.print(f"  GPU {i}: {device.name}")
            console.print(f"    Memory: {memory_gb:.1f} GB")
            console.print(f"    Compute: sm_{device.major}{device.minor}")
    else:
        console.print("\n[yellow][-] CUDA Not Available[/yellow]")
        console.print("  Running on CPU (will be slower)")

    # MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        console.print("\n[green][+] MPS Available (Apple Silicon)[/green]")


@app.command()
def models():
    """List recommended models for different use cases."""
    console.print("\n[bold]Recommended Encoder Models[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Size", style="yellow")
    table.add_column("VRAM", style="green")
    table.add_column("Quality", style="magenta")
    table.add_column("Notes")

    table.add_row(
        "Qwen/Qwen3-4B",
        "4B",
        "~8 GB",
        "Excellent",
        "Best quality, rivals 72B models",
    )
    table.add_row(
        "Qwen/Qwen3-1.7B",
        "1.7B",
        "~4 GB",
        "Very Good",
        "Great balance of speed/quality",
    )
    table.add_row(
        "Qwen/Qwen3-0.6B",
        "0.6B",
        "~1.5 GB",
        "Good",
        "Fast, works on CPU",
    )
    table.add_row(
        "microsoft/phi-2",
        "2.7B",
        "~6 GB",
        "Good",
        "Alternative, older model",
    )
    table.add_row(
        "ibm-granite/granite-3.1-1b-a400m-instruct",
        "1B",
        "~2 GB",
        "Good",
        "IBM's compact model",
    )

    console.print(table)

    console.print("\n[bold]Recommended Scorer Models[/bold]\n")
    console.print("  - sentence-transformers/all-MiniLM-L6-v2 (default, fast)")
    console.print("  - sentence-transformers/all-mpnet-base-v2 (higher quality)")
    console.print("  - Or use --scorer with the same model as encoder")


@app.command()
def version():
    """Show version and dependency information."""
    from latent_reasoning import __version__

    console.print(f"\n[bold]latent-reasoning[/bold] v{__version__}\n")

    try:
        import torch
        console.print(f"  torch: {torch.__version__}")
    except ImportError:
        console.print("  torch: [red]not installed[/red]")

    try:
        import transformers
        console.print(f"  transformers: {transformers.__version__}")
    except ImportError:
        console.print("  transformers: [red]not installed[/red]")

    try:
        import pydantic
        console.print(f"  pydantic: {pydantic.__version__}")
    except ImportError:
        console.print("  pydantic: [red]not installed[/red]")


# ============================================================================
# Placeholder Commands (TODO)
# ============================================================================


@app.command("arc-eval")
def arc_eval(
    encoder: str = typer.Option(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "--encoder", "-e",
        help="Encoder model (HuggingFace ID)",
    ),
    max_tasks: Optional[int] = typer.Option(
        None,
        "--max-tasks", "-n",
        help="Limit number of tasks (default: all ~400)",
    ),
    chains: int = typer.Option(
        4,
        "--chains", "-c",
        help="Number of evolution chains",
    ),
    generations: int = typer.Option(
        5,
        "--generations", "-g",
        help="Max generations per task",
    ),
    max_tokens: int = typer.Option(
        512,
        "--max-tokens", "-t",
        help="Max output tokens",
    ),
    decode_mode: str = typer.Option(
        "soft_prompt",
        "--decode-mode",
        help="Decode control mode: seed, soft_prompt, dual_steering, intermediate_steering, geometry_feedback",
    ),
    reasoning_mode: str = typer.Option(
        "evolution",
        "--reasoning-mode",
        help="Reasoning strategy: evolution, trajectory, hybrid",
    ),
    trajectory_steps: int = typer.Option(
        6,
        "--trajectory-steps",
        help="Number of trajectory refinement steps when trajectory reasoning is enabled.",
        min=1,
        max=200,
    ),
    trajectory_decode_interval: int = typer.Option(
        0,
        "--trajectory-decode-interval",
        help="Decode preview interval during trajectory steps (0 disables previews).",
        min=0,
        max=20,
    ),
    trajectory_step_scale: float = typer.Option(
        0.2,
        "--trajectory-step-scale",
        help="Step scale for trajectory movement updates.",
        min=0.0,
        max=5.0,
    ),
    geometry_feedback_target_forward_kl: float = typer.Option(
        0.06,
        "--geometry-feedback-target-forward-kl",
        help="Target forward-KL for geometry feedback decoding.",
        min=0.0,
    ),
    geometry_feedback_kl_tolerance: float = typer.Option(
        0.5,
        "--geometry-feedback-kl-tolerance",
        help="KL tolerance band for geometry feedback decoder steering.",
        min=0.0,
    ),
    geometry_feedback_steering_eta: float = typer.Option(
        0.05,
        "--geometry-feedback-steering-eta",
        help="Learning-rate-like step size for geometry feedback steering.",
        min=0.0,
        max=1.0,
    ),
    geometry_feedback_alpha: float = typer.Option(
        0.01,
        "--geometry-feedback-alpha",
        help="EMA momentum for geometry feedback steering updates.",
        min=0.0,
        max=1.0,
    ),
    geometry_feedback_kl_cap: float = typer.Option(
        0.5,
        "--geometry-feedback-kl-cap",
        help="KL cap for geometry feedback updates.",
        min=0.0,
    ),
    geometry_feedback_topk: int = typer.Option(
        50,
        "--geometry-feedback-topk",
        help="Top-k logits to consider for geometry feedback.",
        min=1,
    ),
    geometry_feedback_eta_min: float = typer.Option(
        0.01,
        "--geometry-feedback-eta-min",
        help="Minimum geometry steering step size.",
        min=0.0,
        max=1.0,
    ),
    geometry_feedback_eta_max: float = typer.Option(
        0.5,
        "--geometry-feedback-eta-max",
        help="Maximum geometry steering step size.",
        min=0.0,
        max=1.0,
    ),
    geometry_feedback_eta_growth: float = typer.Option(
        1.06,
        "--geometry-feedback-eta-growth",
        help="Growth factor when steering error is low.",
        min=1.0,
    ),
    geometry_feedback_eta_decay: float = typer.Option(
        0.85,
        "--geometry-feedback-eta-decay",
        help="Decay factor when steering error is high.",
        min=0.0,
        max=1.0,
    ),
    geometry_feedback_controller: str = typer.Option(
        "legacy",
        "--geometry-feedback-controller",
        help="Geometry controller type: legacy or pid",
    ),
    geometry_feedback_controller_kp: float = typer.Option(
        0.0,
        "--geometry-feedback-controller-kp",
        help="Proportional gain for PID controller.",
        min=0.0,
        max=2.0,
    ),
    geometry_feedback_controller_ki: float = typer.Option(
        0.0,
        "--geometry-feedback-controller-ki",
        help="Integral gain for PID controller.",
        min=0.0,
        max=2.0,
    ),
    geometry_feedback_controller_kd: float = typer.Option(
        0.0,
        "--geometry-feedback-controller-kd",
        help="Derivative gain for PID controller.",
        min=0.0,
        max=2.0,
    ),
    geometry_feedback_controller_error_ema: float = typer.Option(
        0.2,
        "--geometry-feedback-controller-error-ema",
        help="EMA smoothing for controller error signal.",
        min=0.0,
        max=1.0,
    ),
    arc_version: str = typer.Option(
        "3",
        "--arc-version",
        help="ARC benchmark version: 2 (legacy) or 3 (default)",
    ),
    lr_retries: int = typer.Option(
        1,
        "--lr-retries",
        help="Additional ARC LR attempts per test when parsing fails",
    ),
    arc_strategy: str = typer.Option(
        "adaptive",
        "--arc-strategy",
        help="ARC LR strategy: single, adaptive, repair, consensus, geometry_bandit, self_improving",
    ),
    data_dir: Path = typer.Option(
        Path("./data"),
        "--data-dir",
        help="Directory for ARC dataset files",
    ),
    output_dir: Path = typer.Option(
        Path("./eval_results"),
        "--output-dir", "-o",
        help="Directory for evaluation results",
    ),
):
    """
    Run static ARC-style grid evaluation comparing Baseline vs Latent Reasoning.

    Note: official ARC-AGI-3 is now an interactive agent benchmark. Use
    experiments/run_arc3_official_harness.py for the real ARC-AGI-3 harness.
    This command is for static grid tasks and cheap proxy experiments.

    ARC tasks are grid-based puzzles where you must find the pattern from
    training examples and apply it to test inputs.

    Examples:
        latent-reason arc-eval --max-tasks 10
        latent-reason arc-eval --encoder Qwen/Qwen3-0.6B --chains 6
        latent-reason arc-eval --arc-strategy consensus --lr-retries 2 --arc-version 3
        latent-reason arc-eval --output-dir ./my_results
    """
    from latent_reasoning.eval.arc_agi2 import run_arc_evaluation

    if decode_mode not in {"seed", "soft_prompt", "dual_steering", "intermediate_steering", "geometry_feedback"}:
        console.print(
            "[red]Error: --decode-mode must be one of: "
            "seed, soft_prompt, dual_steering, intermediate_steering, geometry_feedback[/red]"
        )
        raise typer.Exit(1)
    if reasoning_mode not in {"evolution", "trajectory", "hybrid"}:
        console.print(
            "[red]Error: --reasoning-mode must be one of: "
            "evolution, trajectory, hybrid[/red]"
        )
        raise typer.Exit(1)

    try:
        results = run_arc_evaluation(
            encoder=encoder,
            max_tasks=max_tasks,
            chains=chains,
            generations=generations,
            max_tokens=max_tokens,
            decode_mode=decode_mode,
            arc_version=arc_version,
            lr_retries=lr_retries,
            data_dir=str(data_dir),
            output_dir=str(output_dir),
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

        # Summary already printed by run_arc_evaluation
        console.print(f"\n[bold green]Evaluation complete![/bold green]")
        console.print(f"Results saved to: {output_dir}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"[red]Error: {_sanitize_text(str(e))}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def benchmark(
    corpus: str = typer.Option(
        "general",
        "--corpus", "-c",
        help="Corpus to benchmark: general, legal, coding, adversarial",
    ),
    output: Path = typer.Option(
        Path("benchmark_results"),
        "--output", "-o",
        help="Output directory for results",
    ),
    limit: int = typer.Option(
        10,
        "--limit", "-n",
        help="Maximum questions to evaluate",
    ),
    encoder: str = typer.Option(
        "Qwen/Qwen3-0.6B",
        "--encoder", "-e",
        help="Encoder model",
    ),
):
    """
    Run benchmark evaluation (coming soon).

    Compares baseline vs latent reasoning on standard test sets.
    """
    console.print(f"[bold]Benchmark: {corpus}[/bold]")
    console.print(f"Limit: {limit} questions, Output: {output}")
    console.print("\n[yellow]Note: Full benchmark requires Gemini API key[/yellow]")
    console.print("[dim]Benchmark implementation in progress...[/dim]")


@app.command()
def train(
    model_type: str = typer.Argument(
        ...,
        help="What to train: scorer, modifier, projector",
    ),
    data: Path = typer.Option(
        ...,
        "--data", "-d",
        help="Training data path (JSONL)",
    ),
    output: Path = typer.Option(
        Path("models"),
        "--output", "-o",
        help="Output directory",
    ),
    epochs: int = typer.Option(
        3,
        "--epochs", "-e",
        help="Training epochs",
    ),
    base_model: str = typer.Option(
        "Qwen/Qwen3-0.6B",
        "--model", "-m",
        help="Base model to build on",
    ),
):
    """
    Train custom components (coming soon).

    Train scorers, modifiers, or projection layers on your data.
    """
    console.print(f"[bold]Training {model_type}[/bold]")
    console.print(f"Data: {data}, Output: {output}, Epochs: {epochs}")
    console.print("[dim]Training implementation in progress...[/dim]")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
