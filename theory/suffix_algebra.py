"""
Symbolic normalizer and Cayley-table generator for the suffix action algebra.

Three modes reflecting the current state of knowledge:

  lrb       — Left-regular band: a²=a, aba=ab. FALSIFIED (766a1cf).
              Retained for provenance: it generated the frozen predictions
              that the LRB decisive test refuted.

  band      — Free band on generators: a²=a for ALL elements, but NO
              first-occurrence absorption. The free band on {C,P} has
              exactly 6 nonidentity elements: C, P, CP, PC, CPC, PCP.
              Generates predictions for the H-BAND2 test.

  gen_idem  — Generator-idempotent only: CC~C, PP~P, but alternating
              words may grow indefinitely. No reductions beyond adjacent
              repeats. Represents H-GEN-IDEM.

Usage:
  python theory/suffix_algebra.py                     # default band mode
  python theory/suffix_algebra.py --mode lrb
  python theory/suffix_algebra.py --mode band C P U V
"""

from __future__ import annotations
import sys
from itertools import product


def normalize_lrb(word: tuple[str, ...]) -> tuple[str, ...]:
    """LRB normal form: ordered first occurrences. FALSIFIED."""
    seen: set[str] = set()
    result: list[str] = []
    for letter in word:
        if letter not in seen:
            seen.add(letter)
            result.append(letter)
    return tuple(result)


def normalize_band(word: tuple[str, ...]) -> tuple[str, ...]:
    """Free band normal form on 2 generators.

    The free band on {a,b} has 6 elements: a, b, ab, ba, aba, bab.
    The reduction rules are x²=x for ALL elements (not just generators).
    For 2 generators, this means any word reduces to one of these 6 forms.
    Key: aba·b·a = aba (since (aba)²=aba), but aba ≠ ab (unlike LRB).
    """
    if len(word) <= 1:
        return word
    result = list(word)
    changed = True
    while changed:
        changed = False
        new = _reduce_adjacent_repeats(tuple(result))
        if new != tuple(result):
            result = list(new)
            changed = True
        new = _reduce_band_squares(tuple(result))
        if new != tuple(result):
            result = list(new)
            changed = True
    return tuple(result)


def _reduce_adjacent_repeats(word: tuple[str, ...]) -> tuple[str, ...]:
    """Remove adjacent identical letters: aa → a."""
    if len(word) <= 1:
        return word
    result = [word[0]]
    for c in word[1:]:
        if c != result[-1]:
            result.append(c)
    return tuple(result)


def _reduce_band_squares(word: tuple[str, ...]) -> tuple[str, ...]:
    """Reduce x²=x for subwords of length > 1.

    For 2 generators with alternating structure, the maximal non-repeating
    subwords are of length ≤ 3 (e.g., aba). If the word contains a
    repeated alternating block, reduce it.
    """
    w = list(word)
    for block_len in range(2, len(w) // 2 + 1):
        for start in range(len(w) - 2 * block_len + 1):
            block = w[start:start + block_len]
            if w[start + block_len:start + 2 * block_len] == block:
                w = w[:start + block_len] + w[start + 2 * block_len:]
                return _reduce_adjacent_repeats(tuple(w))
    return tuple(w)


def normalize_gen_idem(word: tuple[str, ...]) -> tuple[str, ...]:
    """Generator-idempotent: only reduce adjacent repeats of same letter."""
    return _reduce_adjacent_repeats(word)


NORMALIZERS = {
    "lrb": normalize_lrb,
    "band": normalize_band,
    "gen_idem": normalize_gen_idem,
}

FREE_BAND_2GEN = [
    (),
    ("C",), ("P",),
    ("C", "P"), ("P", "C"),
    ("C", "P", "C"), ("P", "C", "P"),
]


def enumerate_normal_forms(alphabet: list[str], mode: str = "band") -> list[tuple[str, ...]]:
    """All normal forms under the given mode."""
    if mode == "lrb":
        return _enumerate_lrb_forms(alphabet)
    elif mode == "band" and len(alphabet) == 2:
        return FREE_BAND_2GEN
    elif mode == "gen_idem":
        return _enumerate_gen_idem_forms(alphabet, max_len=4)
    else:
        return _enumerate_lrb_forms(alphabet)


def _enumerate_lrb_forms(alphabet: list[str]) -> list[tuple[str, ...]]:
    forms = [()]
    n = len(alphabet)
    for length in range(1, n + 1):
        for combo in _combinations(alphabet, length):
            for perm in _permutations(combo):
                forms.append(tuple(perm))
    return forms


def _enumerate_gen_idem_forms(alphabet: list[str], max_len: int = 4) -> list[tuple[str, ...]]:
    """Alternating words up to max_len (no adjacent repeats)."""
    forms = [()]
    for length in range(1, max_len + 1):
        for word in product(alphabet, repeat=length):
            nf = normalize_gen_idem(word)
            if nf == word and nf not in forms:
                forms.append(nf)
    return forms


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


def cayley_table(alphabet: list[str], mode: str = "band"):
    """Build the right-multiplication Cayley table."""
    normalize = NORMALIZERS[mode]
    forms = enumerate_normal_forms(alphabet, mode)
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


def predictions_through_length(alphabet: list[str], mode: str = "band",
                                max_len: int = 4) -> list[dict]:
    """Generate all reduction predictions through the given word length."""
    normalize = NORMALIZERS[mode]
    preds = []
    for length in range(2, max_len + 1):
        for word in product(alphabet, repeat=length):
            nf = normalize(word)
            if nf != word:
                preds.append({
                    "word": format_word(word),
                    "normal_form": format_word(nf),
                    "reduction": f"{format_word(word)} ~= {format_word(nf)}",
                    "type": _classify_reduction(word, nf, mode),
                })
    seen = set()
    unique_preds = []
    for p in preds:
        key = (p["word"], p["normal_form"])
        if key not in seen:
            seen.add(key)
            unique_preds.append(p)
    return unique_preds


def _classify_reduction(word: tuple[str, ...], nf: tuple[str, ...],
                         mode: str) -> str:
    if len(set(word)) == 1:
        return "idempotence"
    if mode == "lrb" and len(word) == 3 and word[0] == word[2] and word[0] != word[1]:
        return "first-occurrence (aba->ab) [FALSIFIED]"
    if len(word) > len(nf) and len(set(word)) > 1:
        return "band-square (x²->x)" if mode == "band" else "adjacent-repeat"
    return "composite"


def main():
    mode = "band"
    args = sys.argv[1:]
    if "--mode" in args:
        idx = args.index("--mode")
        mode = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    alphabet = args if args else ["C", "P"]
    normalize = NORMALIZERS[mode]

    status = {
        "lrb": "FALSIFIED (766a1cf) — retained for provenance",
        "band": "H-BAND2 — under test",
        "gen_idem": "H-GEN-IDEM — competing hypothesis",
    }

    print(f"Suffix Action Algebra — {mode.upper()} Normalizer")
    print(f"Status: {status.get(mode, 'unknown')}")
    print(f"Alphabet: {{{', '.join(alphabet)}}}")
    print(f"{'=' * 60}")

    forms = enumerate_normal_forms(alphabet, mode)
    print(f"\nNormal forms ({len(forms)} elements including identity):")
    for f in forms:
        print(f"  {format_word(f)}")

    table = cayley_table(alphabet, mode)
    print(f"\nCayley table (right multiplication):")
    max_w = max(len(format_word(f)) for f in forms)
    header = "  " + "·".rjust(max_w + 2)
    for g in alphabet:
        header += f"  {g:>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for form in forms:
        row = f"  {format_word(form):>{max_w + 2}}"
        for g in alphabet:
            result = table[form][g]
            row += f"  {format_word(result):>6}"
        absorbing = all(table[form][g] == form for g in alphabet)
        row += "  <- absorbing" if absorbing and form else ""
        print(row)

    preds = predictions_through_length(alphabet, mode, max_len=4)
    print(f"\nPredictions through length 4 ({len(preds)} reductions):")
    print(f"{'Word':<10} {'Normal form':<12} {'Type':<30}")
    print("-" * 52)
    for p in preds:
        print(f"{p['word']:<10} {p['normal_form']:<12} {p['type']:<30}")

    if mode == "band" and len(alphabet) == 2:
        print(f"\n=== DECISIVE LENGTH-4 PREDICTIONS (H-BAND2) ===")
        print(f"  CPCP ~= CP   (composite idempotence: (CP)² = CP)")
        print(f"  PCPC ~= PC   (composite idempotence: (PC)² = PC)")
        print(f"  CPCP ~/= CPC  (if CPCP~CPC, that's H-SAT3, not a band)")
        print(f"  PCPC ~/= PCP  (if PCPC~PCP, that's H-SAT3, not a band)")


if __name__ == "__main__":
    main()
