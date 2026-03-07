# Cross-Model Task-Level Analysis (2026-03-06)

## Expression Type Mapping (seed=42, sweet_spot difficulty)

| Task ID | Expression | Type | Answer |
|---------|-----------|------|--------|
| nest_000 | (34 + 23) * (45 - 25) | paren | 1140 |
| nest_001 | (24 * 23) // 4 + 21 | div_add | 159 |
| nest_002 | 89 * 73 - 38 * 8 | sub | 6193 |
| nest_003 | 55 * 39 - 37 * 2 | sub | 2071 |
| nest_004 | 97 * 53 - 15 * 6 | sub | 5051 |
| nest_005 | 99 * 66 + 83 * 22 | add | 8360 |
| nest_006 | (49 * 32 + 58) % 27 | mod | 6 |
| nest_007 | 54 * 29 - 87 * 6 | sub | 1044 |
| nest_008 | 91 * 48 + 97 * 30 | add | 7278 |
| nest_009 | (47 * 92 + 50) % 28 | mod | 6 |
| nest_010 | 11 * 21 * 4 | triple | 924 |
| nest_011 | (83 + 31) * (16 - 17) | paren | -114 |
| nest_012 | 68 * 96 - 69 * 6 | sub | 6114 |
| nest_013 | (63 * 34 + 47) % 18 | mod | 11 |
| nest_014 | (84 * 33 + 90) % 29 | mod | 20 |
| nest_015 | 89 * 87 - 10 * 5 | sub | 7693 |
| nest_016 | (50 * 27 + 40) % 15 | mod | 10 |
| nest_017 | (36 * 80) // 7 + 80 | div_add | 491 |
| nest_018 | (45 * 59 + 61) % 24 | mod | 4 |
| nest_019 | (28 * 63) // 8 + 12 | div_add | 232 |
| nest_020 | (27 + 49) * (18 - 12) | paren | 456 |
| nest_021 | 47 * 89 + 26 * 56 | add | 5639 |
| nest_022 | 75 * 65 + 64 * 36 | add | 7179 |
| nest_023 | (71 * 63 + 23) % 19 | mod | 12 |
| nest_024 | 13 * 13 * 3 | triple | 507 |

## Cross-Model Difficulty Gradient

Tasks sorted by how many models solve at baseline -> perturbation benefit:

| Baseline difficulty | Tasks | Avg noise gain | Unlock rate |
|---|---|---|---|
| 0/4 (hardest) | 5 | +0.60 | 5/5 (100%) |
| 1/4 | 9 | +0.36 | 9/9 (100%) |
| 2/4 | 6 | +0.21 | 4/6 (67%) |
| 3/4 | 4 | +0.06 | 2/4 (50%) |
| 4/4 (easiest) | 1 | +0.00 | 0/1 (0%) |

**Monotonic relationship**: harder tasks benefit MORE from perturbation.

## Universal Perturbation-Sensitive Task: nest_009

`(47*92 + 50) % 28 = 6` — unlocked by ALL 4 models (including 1.7B null model).

Codex insight: "Perturbation seems to help most when there is a low-state alternative
trajectory the model can fall into." Modular reduction creates a cheap compressed path.

## Headroom and Rescue Efficiency (Codex 2026-03-06)

| Model | Base | Oracle | Headroom used | Per-direction rescue rate |
|-------|------|--------|---------------|-------------------------|
| Qwen3-4B | 8/25 | 25/25 | 100% (17/17) | 39.4% (67/170) |
| DeepSeek | 19/25 | 25/25 | 100% (6/6) | 77.8% (14/18) |
| phi-2 | 3/25 | 7/25 | 18% (4/22) | 6.1% (4/66) |
| Qwen3-1.7B | 7/25 | 11/25 | 22% (4/18) | — (net: +4 rescue, -2 regress) |

DeepSeek directions are 2x more efficient at rescuing misses than Qwen3-4B.

## Expression Type x Model Interaction (Qwen3-4B anchor)

| Pattern | n | 4B delta | DS delta | phi-2 delta | 1.7B delta |
|---------|---|----------|----------|-------------|------------|
| triple | 2 | +50pp | +0pp | +0pp | +50pp |
| div_add | 3 | +37pp | +22pp | +11pp | +11pp |
| mod | 7 | +24pp | +10pp | +0pp | +0pp |
| add | 4 | +20pp | -17pp | +0pp | +8pp |
| sub | 6 | +5pp | -6pp | +0pp | -17pp |
| paren | 3 | +0pp | +33pp | +44pp | -11pp |

**div_add is the only pattern consistently positive across all 4 models.**

Caveat: n=2-7 tasks per pattern, n=3 latents for non-4B. Exploratory only.

## Statistical Tests (Codex-recommended)

- Per-model: Exact McNemar on paired (baseline vs oracle) task outcomes
  - Qwen3-4B: 17 gains, 0 losses, p ~ 1.5e-5
  - DeepSeek: 6 gains, 0 losses, p ~ 0.031
  - phi-2: 4 gains, 0 losses, p ~ 0.125 (underpowered)
- Cross-model: Compare rescued-failure proportions, NOT raw +pp (different headroom)
- Combined: Fisher on per-model McNemar p-values (exploratory, n=3 models)

## 1.7B Regression Mechanism (Codex 2026-03-06)

nest_003 and nest_023: correct at baseline, wrong under ALL 3 latents.
Not destroyed arithmetic — "verbose, unstable trajectories that exceed
the model's reliable completion budget." Execution stability, not capability loss.

## Data Integrity

All mismatches are at len=2000 truncation boundary. Stored `correct` values
(computed on full text at experiment time) are ground truth. No scoring bugs.
- phi-2: 0 mismatches (clean)
- DeepSeek: 6 mismatches (all truncation)
- Qwen3-1.7B: 22 mismatches (all truncation)
- Qwen3-4B: 77 mismatches (all truncation, n=10 latents)
