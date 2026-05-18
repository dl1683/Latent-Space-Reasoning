"""Logging utilities for Latent Space Reasoning Engine."""

from __future__ import annotations

import logging
import sys
from enum import IntEnum
from typing import Any

from rich.console import Console
from rich.logging import RichHandler


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


remaining file unchanged from here...