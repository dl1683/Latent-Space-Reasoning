# Qwen3-0.6B Baseline vs LR (4-bit) Output Differences

This note summarizes the qualitative differences observed in the 0.6B runs on tests 34-43.

## Setup

- Model: Qwen/Qwen3-0.6B
- Baseline: direct generation, no quantization
- LR: latent reasoning, 4-bit quantization, best decode
- Evolution: chains=3, generations=3, survivors=2
- Max tokens: 2048
- Tests: 34-43 from `examples/test_results`

## Key Observations

- Output style diverges clearly.
  - Baseline often meanders or stops without a final answer.
  - LR tends to produce a structured, decisive answer, even when incorrect.
- Correctness (10 tests):
  - LR wins: 1
  - Baseline wins: 3
  - Ties (both fail): 6
- The shift in structure suggests the judge is steering response shape and confidence, not correctness.

## Examples

- test37 constrained shortest path:
  - Baseline selects a path that violates the constraint (missing exactly one of B/D).
  - LR selects A->B->E->G, which is valid and correct.
- test38 stack machine:
  - Baseline does not converge to a final stack.
  - LR outputs a clean final stack, but it is incorrect.
- test41 bag probability:
  - Baseline oscillates between answers.
  - LR gives a confident but incorrect fraction.
- test34, test36, test39, test40, test42, test43:
  - Both fail to provide a correct final answer.

## Implications

This is preliminary evidence: search is changing output behavior, but the current judge is not aligned to correctness. Better judges/checkers are still required before making quality claims.

## Next Steps

- Add deterministic checkers for tasks with known solutions.
- Score or filter candidates by correctness or output format constraints.
- Train or calibrate judges on correctness labels from these tests.
