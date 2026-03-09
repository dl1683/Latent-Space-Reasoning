# Critical Self-Reflective Analysis: Reasoning Quality vs Verbosity
**Date**: 2026-03-08 (UPDATED with response_raw fix)

## Executive Summary

The initial reasoning quality analysis appeared to show perturbation inflating output length.
**This was ENTIRELY an artifact of the response/response_raw field ordering bug** (fixed in
commit 6c69284). With the fix, perturbation produces essentially NO change in verbosity (-3% words).
The defensible claim remains trajectory diversification and convergence improvement.

## CORRECTION: Verbosity Inflation was a Bug (2026-03-08)

### The Bug
`analyze_reasoning_quality.py` used `response` before `response_raw`. For correct baseline
tasks, `response` was the short polished post-`</think>` answer (~135 words). For wrong
tasks, `response` was the raw thinking trace (~237 words). This manufactured the
"wrong answers are more verbose" pattern.

### Before vs After Fix (Qwen3-4B)
| Metric | Before Fix (buggy) | After Fix (correct) |
|--------|-------------------|---------------------|
| Baseline correct words | 135 | 300 |
| Baseline wrong words | 237 | 376 |
| Wrong/correct ratio | 1.76x | 1.25x |
| Perturbation delta words | +67% | **-3%** |
| Perturbation delta steps | +79% | -0% |

**The perturbation verbosity effect was entirely an artifact.**

### Corrected Quality Deltas (Qwen3-4B)
| Outcome | n | Delta words | Delta steps | Delta computations |
|---------|---|-------------|-------------|--------------------|
| FIXED (wrong->right) | 67 | **-32** | -1.7 | +1.8 |
| MAINTAINED (same) | 165 | -3 | +0.6 | -0.7 |
| BROKEN (right->wrong) | 18 | -1 | -0.1 | 0.0 |

FIXED tasks actually get SHORTER under perturbation. The reasoning chains are
more efficient, not more verbose.

### DeepSeek: BROKEN > FIXED Verbosity (Red Flag)
| Outcome | n | Delta words | Delta steps |
|---------|---|-------------|-------------|
| FIXED | 37 | +14.3 | -0.5 |
| BROKEN | 41 | +44.8 | +3.4 |

Tasks that get WORSE under perturbation show 3x MORE additional computation. This directly contradicts the reasoning quality narrative.

## Heuristic Scorer: Domain Mismatch

The heuristic scorer was designed for software engineering PLANS:
- **Depth score**: Maximum at 300-1000 words (rewards verbosity)
- **Action score**: Rewards "create", "implement", "deploy" (irrelevant to arithmetic)
- **Coherence**: Partially relevant but tracks sequential words that increase with length

Results:
| Model | Overall delta | Interpretation |
|-------|---------------|----------------|
| Qwen3-4B | +0.121 | Entirely explained by word count increase |
| DeepSeek | +0.005 | Noise (honest signal from mismatched scorer) |
| phi-2 | -0.009 | Noise |

**The heuristic scorer should not appear in the paper.**

## GRADING AUDIT: Last-Integer-Wins Confound Analysis (2026-03-08)

### Qwen3-4B: The Model Can Already Compute — It Can't Converge

| Grading Rule | Baseline | Perturbed | Delta |
|-------------|----------|-----------|-------|
| Last integer (used) | 32% | 43% | +11pp |
| Answer anywhere | 80% | 82% | +2pp |

**The model already computes the correct answer 80% of the time.** It just fails to put it as the final output. Perturbation barely changes answer-anywhere (82%) but significantly improves convergence (32% -> 43%).

This means perturbation's effect is on **answer convergence/selection**, not **computation**.

Integer statistics:
| Condition | Mean integers | Mean unique | Correct answer appearances (when correct) |
|-----------|--------------|-------------|------------------------------------------|
| Baseline | 71.9 | 17.9 | 3.0 times |
| Perturbed | 111.4 | 24.5 | 5.2 times |

52% of baseline wrong answers contain the correct answer somewhere (just not last). Under perturbation, this drops to 39% — suggesting perturbation helps the model converge on the right final answer rather than wandering past it.

### DeepSeek: Perturbation Hurts Computation

| Grading Rule | Baseline | Perturbed | Delta |
|-------------|----------|-----------|-------|
| Last integer | 76% | 69% | -7pp |
| Answer anywhere | 84% | 78% | -6pp |

For DeepSeek, perturbation degrades BOTH computation and convergence. The oracle effect (100%) is purely about trajectory diversity — some directions happen to work for some tasks.

### phi-2: Too Small for Meaningful Signal

Baseline only has 9.2 integers per response; answer-anywhere difference negligible (16% -> 20%).

### Implication: Perturbation is a Convergence Aid, Not a Computation Aid

The correct reframing: perturbation doesn't help the model FIND the answer — it helps the model STOP at the answer. This is consistent with trajectory diversification: different starting conditions lead to different reasoning chains that terminate at different points, and some chains happen to terminate after computing the correct answer.

This is still interesting (and publishable), but it's a much more modest claim than "improved reasoning quality."

## Codex Review Confirmation (2026-03-08, gpt-5.4)

Codex independently confirmed all findings and identified an additional systematic bias:

**Response field ordering bug**: `analyze_reasoning_quality.py` uses `task.get('response', '') or task.get('response_raw', '')`. For successes, `response` is often the short polished post-`</think>` answer. For failures/truncations, it is the raw unfinished thinking trace. This alone can manufacture the "wrong answers are more verbose" pattern.

Codex verdict: *"Current evidence does not justify a claim that perturbation improves reasoning quality or planning quality. It supports a narrower claim: on Qwen, 2-token perturbation changes rollout mode and sometimes improves arithmetic success, but the quality analysis mostly measures verbosity, formatting, and whether the model cleanly exits its think trace."*

Codex also flagged `nest_006`: baseline derives correct answer (6) in thinking but continues generating, runs out of tokens, and last integer is not 6. Under perturbation, it produces clean `</think>` + `\boxed{6}`. This is a convergence fix, not a reasoning fix.

## Three Alternative Interpretations

### 1. Lottery-Ticket Trajectory Sampling
Perturbation diversifies starting conditions under greedy decoding. Different trajectories reach different answers. Accuracy gain = selection bias. Oracle = exploration of trajectory space. No reasoning quality improvement needed.

### 2. Token-Budget Exploitation
Perturbation activates think mode (16% -> 100%), giving more tokens. More tokens = more chances for `last-integer-wins` grading to find correct match. Non-monotonic dose-response: 2-tok activates think mode without destroying coherence.

### 3. Computational Shortcut Activation
Perturbation selects among pre-existing strategies of roughly equal quality. Some strategies work for some tasks. Explains task selectivity and force-think gap (+11.6pp beyond think-mode alone). More charitable than #1 but still ≠ "improved reasoning."

## What Would Confirm Genuine Reasoning Improvement

1. **Length-controlled comparison**: Force-think at matched token budget with n=10 directions
2. **Answer extraction audit**: Does `last-integer-wins` grading create length confound?
3. **LLM-as-judge chain validation**: Are FIXED chains actually valid derivations?
4. **Token-budget sweep**: Does accuracy scale with max_tokens (artifact) or not (genuine)?
5. **Unique-integer count**: Do perturbed responses generate more distinct numbers?

## What IS Defensible

These findings ARE genuine and do not depend on the quality question:
1. **Oracle coverage structure**: Different directions solve different tasks (permutation-validated)
2. **Non-monotonic dose-response**: 2-tok optimum (1-tok hurts, 3-tok bifurcated)
3. **Deterministic chaos under greedy decoding**: Invariance length measurements
4. **Quantization x noise interaction**: Clean within-model control (4-bit null, 8-bit +16pp)
5. **Force-think decomposition**: Perturbation contributes +11.6pp beyond think-mode activation

## Correct Framing

**Wrong**: "Perturbation improves reasoning quality"
**Wrong**: "Perturbation helps the model compute better"
**Right**: "Perturbation modulates reasoning trajectories, improving answer convergence for some tasks while degrading it for others. The model can already compute correct answers 80% of the time — the bottleneck is convergence, not computation. Different perturbation directions succeed on different tasks, enabling oracle-style coverage through trajectory diversification."

## Action Items
- [ ] Remove heuristic scorer from paper (or clearly label as exploratory)
- [ ] Frame paper around trajectory diversification, not quality
- [ ] Run length-controlled baselines
- [x] Audit last-integer-wins grading for length confound (DONE: convergence, not computation)
- [ ] Test on PLANNING tasks (the domain the scorer was designed for)
- [ ] Run LLM-as-judge evaluation on FIXED chains
