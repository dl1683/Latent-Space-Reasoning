"""Post-process legal v2 outputs: strip <think>...</think> tokens for fair comparison.

The perturbation condition uses decode_with_raw_soft_prompt() which keeps thinking
tokens visible, while baseline strips them and evolution goes through encoder.decode()
which also strips them. This script normalizes all outputs by stripping thinking tokens.

Usage:
    python experiments/strip_thinking_tokens.py experiments/legal_v2_full.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def strip_thinking(text: str) -> tuple[str, int]:
    """Strip <think>...</think> blocks, return (cleaned_text, thinking_word_count)."""
    think_blocks = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
    thinking_words = sum(len(b.split()) for b in think_blocks)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    # Handle unclosed thinking blocks (model ran out of tokens mid-think)
    if cleaned.startswith("<think>"):
        cleaned = ""
        thinking_words = len(text.split())
    return cleaned, thinking_words


def process_file(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_stripped = 0
    for o in data["outputs"]:
        original = o["response"]
        cleaned, think_words = strip_thinking(original)
        if think_words > 0:
            o["response_raw"] = original
            o["response"] = cleaned
            o["response_length"] = len(cleaned)
            o["word_count"] = len(cleaned.split())
            o["thinking_words_stripped"] = think_words
            n_stripped += 1

    out_path = path.replace(".json", "_clean.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(data['outputs'])} outputs, stripped thinking from {n_stripped}")
    print(f"Written to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.json>")
        sys.exit(1)
    process_file(sys.argv[1])
