"""
Symbolic normalizer and Cayley-table generator for the suffix action algebra.

Given the left-regular band (LRB) identities:
  a^2 = a        (idempotence)
  aba = ab        (first-occurrence absorption)

the normal form of any word is the ordered subsequence of first occurrences
of each letter. This script generates the full Cayley table and all
predictions through length 3 for a given alphabet, BEFORE any new model
output is inspected.

Usage:
  python theory/suffix_algebra.py              # default {C, P}
  python theory/suffix_algebra.py C P U V      # four generators
"""

from __future__ import annotations
import sys
from itertools import product


def normalize(word: tuple[str, ...]) -> tuple[str, ...]:
    """Reduce a word to its LRB normal form: ordered first occurrences."""
    seen: set[str] = set()
    result: list[str] = []
    for letter in word:
        if letter not in seen:
            seen.add(letter)
            result.append(letter)
    return tuple(result)


def enumerate_normal_forms(alphabet: list[str]) -> list[tuple[str, ...]]:
    """All LRB normal forms over the alphabet (ordered subsets, including empty)."""
    forms = [()]
    n = len(alphabet)
    for length in range(1, n + 1):
        for perm in _ordered_subsets(alphabet, length):
            forms.append(tuple(perm))
    return forms


def _ordered_subsets(alphabet: list[str], k: int) -> list[list[str]]:
    """All ordered subsets of size k from alphabet."""
    if k == 0:
        return [[]]
    result = []
    for combo in _combinations(alphabet, k):
        for perm in _permutations(combo):
            result.append(perm)
    return result


def _combinations(lst: list[str], k: int) -> list[list[str]]:
    if k == 0:
        return [[]]
    if not lst:
        return []
    first, rest = lst[0], lst[1:]
    with_first = [[first] + c for c in _combinations(rest, k - 1)]
    without_first = _combinations(rest, k)
    return with_first + without_first


def _permutations(lst: list[str]) -> list[list[str]]:
    if len(lst) <= 1:
        return [lst[:]]
    result = []
    for i, item in enumerate(lst):
        rest = lst[:i] + lst[i + 1:]
        for perm in _permutations(rest):
            result.append([item] + perm)
    return result


def cayley_table(alphabet: list[str]) -> dict[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Build the right-multiplication Cayley table for all normal forms."""
    forms = enumerate_normal_forms(alphabet)
    table: dict[tuple[str, ...], dict[str, tuple[str, ...]]] = {}
    for form in forms:
        table[form] = {}
        for gen in alphabet:
            table[form][gen] = normalize(form + (gen,))
    return table


def format_word(word: tuple[str, ...]) -> str:
    if not word:
        return "e"
    return "".join(word)


def predictions_through_length(alphabet: list[str], max_len: int = 3) -> list[dict]:
    """Generate all reduction predictions through the given word length."""
    preds = []
    for length in range(2, max_len + 1):
        for word in product(alphabet, repeat=length):
            nf = normalize(word)
            if nf != word:
                preds.append({
                    "word": format_word(word),
                    "normal_form": format_word(nf),
                    "reduction": f"{format_word(word)} ~= {format_word(nf)}",
                    "type": _classify_reduction(word, nf),
                })
    seen = set()
    unique_preds = []
    for p in preds:
        key = (p["word"], p["normal_form"])
        if key not in seen:
            seen.add(key)
            unique_preds.append(p)
    return unique_preds


def _classify_reduction(word: tuple[str, ...], nf: tuple[str, ...]) -> str:
    if len(set(word)) == 1:
        return "idempotence"
    if len(word) == 3 and word[0] == word[2] and word[0] != word[1]:
        return "first-occurrence (aba->ab)"
    return "composite"


def main():
    alphabet = sys.argv[1:] if len(sys.argv) > 1 else ["C", "P"]
    print(f"Suffix Action Algebra — LRB Normalizer")
    print(f"Alphabet: {{{', '.join(alphabet)}}}")
    print(f"{'=' * 60}")

    forms = enumerate_normal_forms(alphabet)
    print(f"\nNormal forms ({len(forms)} elements including identity):")
    for f in forms:
        print(f"  {format_word(f)}")

    table = cayley_table(alphabet)
    print(f"\nCayley table (right multiplication):")
    header = "  " + "·".rjust(max(len(format_word(f)) for f in forms) + 2)
    for g in alphabet:
        header += f"  {g:>4}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for form in forms:
        row = f"  {format_word(form):>{max(len(format_word(f)) for f in forms) + 2}}"
        for g in alphabet:
            result = table[form][g]
            row += f"  {format_word(result):>4}"
        absorbing = all(table[form][g] == form for g in alphabet)
        row += "  <- absorbing" if absorbing and form else ""
        print(row)

    preds = predictions_through_length(alphabet, max_len=3)
    print(f"\nPredictions through length 3 ({len(preds)} reductions):")
    print(f"{'Word':<8} {'Normal form':<12} {'Type':<25}")
    print("-" * 45)
    for p in preds:
        print(f"{p['word']:<8} {p['normal_form']:<12} {p['type']:<25}")

    decisive = [p for p in preds if p["type"] == "first-occurrence (aba->ab)"]
    print(f"\nDecisive first-occurrence predictions ({len(decisive)}):")
    for p in decisive:
        print(f"  {p['reduction']}")
    print(f"\nThese are the UNTESTED predictions that distinguish an LRB from")
    print(f"a merely idempotent noncommutative monoid.")

    print(f"\nAbsorbing elements (right zeros — place is frozen after these):")
    for form in forms:
        if form and all(table[form][g] == form for g in alphabet):
            print(f"  {format_word(form)} · a = {format_word(form)} for all a")


if __name__ == "__main__":
    main()
