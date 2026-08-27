"""Logging utilities for Latent Space Reasoning Engine."""

from __future__ import annotations

import logging
import os
import sys
from enum import IntEnum
from typing import Any

from rich.console import Console
from rich.logging import RichHandler


def _is_windows_console() -> bool:
    """Check if running in Windows console that may have encoding issues."""
    return sys.platform == "win32"  # Always sanitize on Windows


def _sanitize_for_console(text: str) -> str:
    """Remove problematic Unicode characters for Windows console."""
    if sys.platform != "win32":
        return text  # Linux/Mac handle Unicode fine

    # Build output character by character, replacing non-ASCII
    replacements = {
        0x2705: "[OK]",   # check mark button
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
        0x2b06: "^",      # up arrow
        0x2b07: "v",      # down arrow
        0x1f527: "[*]",   # wrench
        0x1f4a1: "[i]",   # lightbulb
        0x1f6e0: "[*]",   # hammer/wrench
        0x1f4cc: "[>]",   # pushpin
        0x1f50d: "[?]",   # magnifying glass
        0x1f389: "[!]",   # party popper
        0x1f60a: ":)",    # smile
        0x1f600: ":)",    # grinning
        0x1f4dd: "[>]",   # memo
        0x1f517: "[>]",   # link
        0x1f4e6: "[>]",   # package
        0x1f680: "[>]",   # rocket
        0x1f3af: "[>]",   # target
        0x1f4ca: "[>]",   # chart
        0x1f512: "[>]",   # lock
        0x1f511: "[>]",   # key
    }

    result = []
    for char in text:
        code = ord(char)
        if code < 128:
            result.append(char)
        elif code < 256:
            try:
                char.encode("cp1252")
                result.append(char)
            except UnicodeEncodeError:
                result.append("?")
        else:
            result.append(replacements.get(code, "?"))

    return "".join(result)


# Use ASCII-safe console for Windows compatibility
_console = Console(force_terminal=True, legacy_windows=True, safe_box=True)


class LogLevel(IntEnum):
    """Log verbosity levels."""
    SILENT = 0
    MINIMAL = 1
    NORMAL = 2
    VERBOSE = 3
    DEBUG = 4


# Global verbosity setting
_verbosity = LogLevel.NORMAL
_logger: logging.Logger | None = None


def set_verbosity(level: LogLevel | str | int) -> None:
    """Set the global verbosity level."""
    global _verbosity

    if isinstance(level, str):
        level = LogLevel[level.upper()]
    elif isinstance(level, int):
        level = LogLevel(level)

    _verbosity = level

    # Update logger level
    if _logger:
        if level == LogLevel.SILENT:
            _logger.setLevel(logging.CRITICAL + 1)
        elif level == LogLevel.MINIMAL:
            _logger.setLevel(logging.WARNING)
        elif level == LogLevel.NORMAL:
            _logger.setLevel(logging.INFO)
        elif level == LogLevel.VERBOSE:
            _logger.setLevel(logging.DEBUG)
        else:  # DEBUG
            _logger.setLevel(logging.DEBUG)


def get_verbosity() -> LogLevel:
    """Get the current verbosity level."""
    return _verbosity


def get_logger(name: str = "latent_reasoning") -> logging.Logger:
    """Get a configured logger instance."""
    global _logger

    if _logger is None:
        _logger = logging.getLogger(name)
        _logger.handlers.clear()

        # Rich handler for nice console output
        handler = RichHandler(
            console=_console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        _logger.addHandler(handler)

        # Set initial level
        set_verbosity(_verbosity)

    return _logger


def log_event(
    event: str,
    level: LogLevel = LogLevel.NORMAL,
    **kwargs: Any,
) -> None:
    """Log an event with optional structured data."""
    if _verbosity < level:
        return

    logger = get_logger()

    # Format message
    if kwargs:
        details = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        message = f"[{event}] {details}"
    else:
        message = f"[{event}]"

    # Map level to logging level
    if level <= LogLevel.MINIMAL:
        logger.warning(message)
    elif level == LogLevel.NORMAL:
        logger.info(message)
    else:
        logger.debug(message)


def log_generation(
    gen: int,
    chains: int,
    best_score: float,
    mean_score: float,
    **extra: Any,
) -> None:
    """Log generation progress."""
    log_event(
        f"GEN {gen:02d}",
        level=LogLevel.NORMAL,
        chains=chains,
        best=f"{best_score:.3f}",
        mean=f"{mean_score:.3f}",
        **extra,
    )


def log_chain(
    chain_id: int,
    score: float,
    modified: bool = False,
    **extra: Any,
) -> None:
    """Log chain details (verbose level)."""
    log_event(
        f"Chain {chain_id}",
        level=LogLevel.VERBOSE,
        score=f"{score:.3f}",
        modified=modified,
        **extra,
    )


def print_header(title: str) -> None:
    """Print a styled header."""
    if _verbosity >= LogLevel.MINIMAL:
        _console.print()
        _console.print(f"[bold blue]{'-' * 60}[/bold blue]")
        _console.print(f"[bold blue]  {title}[/bold blue]")
        _console.print(f"[bold blue]{'-' * 60}[/bold blue]")
        _console.print()


def print_result(result: str, score: float, **stats: Any) -> None:
    """Print the final result."""
    if _verbosity >= LogLevel.MINIMAL:
        _console.print()
        _console.print("[bold green]Result[/bold green]")
        _console.print(f"[dim]{'-' * 60}[/dim]")
        # Sanitize for Windows console compatibility
        safe_result = _sanitize_for_console(result)
        try:
            _console.print(safe_result)
        except UnicodeEncodeError:
            # Final fallback: use sys.stdout with errors='replace'
            sys.stdout.buffer.write(result.encode('utf-8', errors='replace'))
            sys.stdout.write('\n')
        _console.print(f"[dim]{'-' * 60}[/dim]")

        stat_str = f"Score: {score:.3f}"
        if stats:
            stat_str += " | " + " | ".join(f"{k}: {v}" for k, v in stats.items())
        _console.print(f"[dim]{stat_str}[/dim]")
        _console.print()


def print_progress(current: int, total: int, prefix: str = "") -> None:
    """Print a progress indicator."""
    if _verbosity >= LogLevel.NORMAL:
        pct = current / total
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "#" * filled + "." * (bar_len - filled)
        _console.print(f"\r{prefix}[{bar}] {current}/{total}", end="")
        if current == total:
            _console.print()
