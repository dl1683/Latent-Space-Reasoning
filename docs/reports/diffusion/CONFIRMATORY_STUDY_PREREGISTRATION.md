# Confirmatory Study Preregistration: Clause-Append Aggregation

**Status**: FROZEN (pre-judging)
**Manifest hash**: `406b5a9076a7ad7c`
**Random seed**: `2026062650`
**Date**: 2026-06-26

## 1. Study Objective

Test whether task-specific complement-packet clauses extracted via latent aggregation provide measurable quality improvement over generic operational boilerplate when appended to anchor planning responses.

### Hypotheses

- **H1 (primary)**: `true_clause > fixed_generic` in pairwise LLM-as-judge evaluation
- **H2 (secondary)**: `true_clause > deranged_clause` (task-specificity control)
- **H3 (secondary)**: `true_clause > anchor` (any-addition effect)
- **H4 (secondary)**: `fixed_generic > anchor` (boilerplate effect)

## 2. Task Sampling Frame

### Source Populations

| Version | Total Tasks | Pilot Excluded | Available | Sampled |
|---------|-------------|----------------|-----------|---------|
| v11     | 96          | 10             | 86        | 33      |
| v10     | 48          | 0              | 48        | 17      |
| **Total** | **144**   | **10**         | **134**   | **50**  |

### Exclusions

- 10 pilot tasks excluded a priori: plan_441, plan_446, plan_463, plan_465, plan_478, plan_481, plan_488, plan_494, plan_515, plan_516
- Tasks with no extractable clauses replaced by next random draw from same version pool
- Replaced: plan_449 -> plan_534, plan_501 -> plan_443, plan_530 -> plan_514

### Sampling Method

Stratified random sampling proportional to version availability (~2:1 v11:v10). Selection NOT based on lift, score, or any outcome variable. Random seed `2026062650` applied to `random.Random` for reproducibility.

### Final Task IDs (50)

**v11 (33)**: plan_443, plan_445, plan_450, plan_451, plan_455, plan_456, plan_459, plan_462, plan_464, plan_467, plan_468, plan_470, plan_472, plan_473, plan_474, plan_475, plan_477, plan_482, plan_486, plan_487, plan_489, plan_490, plan_497, plan_498, plan_507, plan_508, plan_510, plan_514, plan_526, plan_527, plan_528, plan_532, plan_534

**v10 (17)**: plan_393, plan_396, plan_397, plan_398, plan_402, plan_408, plan_412, plan_413, plan_416, plan_420, plan_421, plan_422, plan_423, plan_425, plan_435, plan_439, plan_440

## 3. Arm Definitions

Each task produces 4 arms:

| Arm | Definition |
|-----|-----------|
| **anchor** | Best non-packet record text (highest task_score) |
| **true_clause** | Anchor + real extracted clauses from complement packets via `_realize_clause_append_v1` |
| **deranged_clause** | Anchor + clauses from a different task (within-version derangement) |
| **fixed_generic** | Anchor + count-matched generic operational sentences |

### Fixed Generic Pool (used across all tasks)

1. "Define rollback criteria for the plan."
2. "Define the scope boundary for the plan."
3. "Collect metrics to measure success."
4. "Establish monitoring for implementation progress."
5. "Document the process and measure outcomes."
6. "Define clear success criteria for each phase."
7. "Establish communication protocols for stakeholders."

Clauses are selected from the pool in order, count-matched to the number of true clauses for each task.

### Derangement Algorithm

Within-version random permutation (no fixed-point derangement). Seed `2026062650`. v10 tasks deranged among v10; v11 among v11. Deranged clauses are truncated to match the true clause count for that task when the source has more clauses. When the deranged source has fewer clauses than the true count (3 tasks), the shorter deranged arm is used as-is — this is a conservative bias favoring true in the sanity gate. Full mapping locked in manifest.

### Clause Statistics

- True clause counts: min=1, max=3, mean=2.0
- Deranged clause counts: min=1, max=3, mean=2.0
- Generic clause counts: min=1, max=3, mean=2.0

## 4. Judge Protocol

### Judge Model

Claude Sonnet 4.6 (claude-sonnet-4-6-20250514)

### Temperature and Settings

- Temperature: 0 (deterministic)
- Max tokens: 4096
- No system prompt beyond the judge prompt

### Judges Per Task

3 independent judge calls per task. Each judge sees the same prompt with all 4 arms in randomized anonymous labels (W, X, Y, Z).

### Label Randomization

Each of 3 judges per task receives an independently randomized arm-to-label mapping (seed `2026062650`, consumed sequentially). Judges for the same task see different label assignments to ensure independence. All mappings are stored in the manifest.

### Judge Prompt

The judge prompt asks for:
1. Full ranking of all 4 candidates
2. All 6 pairwise comparisons with winner/tie, confidence (1-5), and reason
3. Best and worst answer
4. Serious errors per candidate
5. One-sentence summary

Evaluation criteria (in order): correctness, constraint respect, actionable sequencing, concrete decision criteria, risk handling, absence of unsupported assumptions, clarity.

### Response Parsing

- JSON response parsed; malformed JSON retried once
- If both attempts fail, the judge vote counts as "tie" for all pairwise comparisons (not excluded)

### Majority Vote Procedure

For each pairwise comparison (e.g., true_clause vs fixed_generic):
1. Each of 3 judges declares a winner or tie
2. Majority vote determines the task-level outcome
3. If all 3 judges give different answers (one A-win, one B-win, one tie), count as tie

### Tie Handling

- Ties count as ties (not wins for either side)
- Primary endpoint counts only strict wins for true_clause

## 5. Analysis Plan

### Primary Endpoint

**true_clause > fixed_generic**: count of tasks where true_clause wins by majority vote (ties excluded from win count).

### Go / No-Go Thresholds

| Outcome | Verdict | Rationale |
|---------|---------|-----------|
| >= 35/50 wins | **Robust GO** | ~70% practical effect size |
| 32-34/50 wins | **Statistical GO** | One-sided binomial rejects 50% null at alpha ~0.05 |
| < 32/50 wins | **NO-GO** | Cannot reject chance |

### Secondary Sanity Gates

- `true > deranged >= 45/50` (task-specificity must be overwhelming)
- `true > anchor >= 32/50` (any-addition effect)
- Unique serious true-clause error flags <= 5/50
- No post-hoc task exclusions

### Confidence Intervals

Wilson 95% confidence intervals reported for all pairwise win rates.

### Statistical Test

One-sided exact binomial test: H0: P(true_clause wins) = 0.5, H1: P(true_clause wins) > 0.5.

## 6. Integrity Rules

1. **No peeking**: No judge results examined until all 150 calls (50 tasks x 3 judges) complete
2. **No mid-study changes**: Realizer, generic pool, derangement, and judge prompt are frozen
3. **No post-hoc exclusions**: All 50 tasks included in analysis regardless of outcome
4. **Single run**: Study executed exactly once; no reruns or parameter tuning

## 7. Artifacts

| Artifact | Path |
|----------|------|
| Study manifest | `eval_results/diffusion_language/confirmatory_study_manifest.json` |
| Judge results | `eval_results/diffusion_language/confirmatory_study_results.json` |
| This preregistration | `docs/reports/diffusion/CONFIRMATORY_STUDY_PREREGISTRATION.md` |
| Study builder script | `experiments/build_confirmatory_study.py` |
| Pilot v2 results | `eval_results/diffusion_language/blinded_pairwise_pilot_v2_results.json` |
| Placebo diagnostic | `eval_results/diffusion_language/placebo_diagnostic_results.json` |
