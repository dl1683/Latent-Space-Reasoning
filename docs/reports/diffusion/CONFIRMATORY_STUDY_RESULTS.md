# Confirmatory Study Results: Clause-Append Aggregation

**Status**: COMPLETE
**Date**: 2026-06-26
**Manifest hash**: `406b5a9076a7ad7c`
**Preregistration**: [CONFIRMATORY_STUDY_PREREGISTRATION.md](CONFIRMATORY_STUDY_PREREGISTRATION.md)

## Verdict: STATISTICAL_GO

## 1. Primary Endpoint

| Metric | Value |
|--------|-------|
| Comparison | true_clause vs fixed_generic |
| Wins | 33/50 (66.0%) |
| Ties | 0/50 |
| Losses | 17/50 |
| Wilson 95% CI | [0.522, 0.776] |
| Binomial p-value (one-sided) | 0.016 |
| Threshold | >= 32 STATISTICAL_GO, >= 35 ROBUST_GO |
| **Verdict** | **STATISTICAL_GO** |

The one-sided exact binomial test rejects H0 (P=0.5) at alpha=0.05. The lower bound of the Wilson CI (52.2%) exceeds 50%, confirming a statistically significant preference for task-specific clauses over generic operational boilerplate.

## 2. Secondary Endpoints

| Comparison | Wins | Ties | Losses | Win Rate | Wilson 95% CI |
|------------|------|------|--------|----------|---------------|
| true > deranged | 47/50 | 0 | 3 | 94.0% | [0.838, 0.979] |
| true > anchor | 44/50 | 1 | 5 | 88.0% | [0.762, 0.944] |
| generic > anchor | 46/50 | 1 | 3 | 92.0% | [0.812, 0.968] |
| generic > deranged | 45/50 | 0 | 5 | 90.0% | [0.786, 0.957] |
| deranged > anchor | 18/50 | 0 | 32 | 36.0% | [0.241, 0.499] |

### Sanity Gates

| Gate | Result | Threshold | Status |
|------|--------|-----------|--------|
| true > deranged | 47/50 | >= 45/50 | **PASS** |
| true > anchor | 44/50 | >= 32/50 | **PASS** |
| Unique true-clause errors | 6/50 | <= 5/50 | **FAIL** (by 1, adjudicator-sensitive) |

**Error gate adjudication**: Automated string-matching found 31/50 tasks with errors flagged for the true_clause arm, but most are shared-anchor errors rephrased per arm. Manual review identified **6 clear unique true-clause errors** where appended clauses introduced genuinely harmful content:

| Task | Failure Mode | Error |
|------|-------------|-------|
| plan_398 | Contamination | Appends "timeline for implementing the new role" to a detection task |
| plan_422 | Temporal confusion | "Establish timeline for replay test" — test already ran |
| plan_470 | Meta-instruction leak | "Specify polarity of the action" — context-free meta artifact |
| plan_489 | Presupposition | Assigns DRI to "C++ rewrite step" before plan decides between rewrite vs transpiler |
| plan_490 | Contradiction | "Prioritize driver comfort while minimizing fuel cost" contradicts task goal |
| plan_526 | Tautology | Circular restatement adding zero executable guidance |

At 6/50, this **FAILS** the preregistered threshold of ≤5 by one. Primary endpoint still passes; task-specificity gates pass; unique-error gate fails narrowly. The failure is adjudicator-sensitive — plan_472 (redundant scope) and plan_527 (partitioning timing) were judged as shared-anchor errors, not unique true-clause errors.

## 3. Judge Agreement

- **0/150** parse failures (all judges returned valid JSON)
- **38/50** tasks had unanimous 3-0 votes on the primary endpoint (27 true, 11 generic)
- **12/50** tasks had 2-1 split votes (6 true, 5 generic via split, 1 with a tie vote)
- Zero tasks had all-different votes (no 3-way splits)

**Caveat**: All 3 judges are the same model (Claude Sonnet 4.6) at temperature 0. These are correlated samples, not independent evaluators. The high unanimity reflects model consistency, not necessarily independent signal confirmation. Multi-model judging would be needed to claim evaluator independence.

## 4. Arm Hierarchy

Based on best/worst answer counts across 150 judge calls:

| Arm | Best Answer | Worst Answer |
|-----|-------------|--------------|
| true_clause | 92 (61.3%) | 2 (1.3%) |
| fixed_generic | 45 (30.0%) | 2 (1.3%) |
| anchor | 7 (4.7%) | 47 (31.3%) |
| deranged_clause | 6 (4.0%) | 99 (66.0%) |

**Hierarchy**: true_clause > fixed_generic >> anchor > deranged_clause

(The true-vs-generic margin is real but moderate at 33/50; the generic-vs-anchor gap is large at 46/50, indicating much of the lift comes from operational scaffolding.)

This hierarchy is consistent with the placebo diagnostic and confirms:
1. **Task-specific clauses are preferred** over generic boilerplate by this judge model (primary endpoint, 33/50)
2. **Any operational content helps** weak anchors (generic > anchor at 92%) — much of the lift is this scaffolding effect
3. **Wrong-task clauses usually hurt** — deranged is worst answer 66% of the time and loses to anchor 64% of the time

## 5. Version Breakdown

| Version | N | True Wins | True Losses | Win Rate |
|---------|---|-----------|-------------|----------|
| v10 | 17 | 12 | 5 | 70.6% |
| v11 | 33 | 21 | 12 | 63.6% |
| **Total** | **50** | **33** | **17** | **66.0%** |

v10 tasks show slightly higher win rates, but the difference is not statistically significant given the small sample sizes.

## 6. Per-Task Results

### Wins (33 tasks — true_clause preferred)

Unanimous (3-0): plan_393, plan_412, plan_421, plan_423, plan_425, plan_439, plan_440, plan_443, plan_451, plan_455, plan_459, plan_467, plan_468, plan_474, plan_475, plan_477, plan_482, plan_487, plan_497, plan_498, plan_507, plan_508, plan_510, plan_514, plan_528, plan_532, plan_534

Split (2-1): plan_396, plan_408, plan_413, plan_422, plan_445, plan_473

### Losses (17 tasks — fixed_generic preferred)

Unanimous (3-0): plan_397, plan_398, plan_420, plan_435, plan_456, plan_470, plan_472, plan_486, plan_490, plan_526, plan_527

Split (2-1): plan_402, plan_416, plan_450, plan_462, plan_464, plan_489

## 7. Interpretation

### What the study shows

1. **Task-specific complement-packet clauses received a statistically significant judge preference** over generic operational boilerplate when appended to anchor planning responses (p=0.016, 66% win rate). This is judge-rated quality, not yet human-validated.

2. **The improvement is genuinely task-specific**, not just an artifact of adding any text. The true > deranged result (47/50) demonstrates that wrong-task clauses are detected and penalized. The deranged arm is rated worst 66% of the time, confirming judges can distinguish task-relevant from task-irrelevant content.

3. **In this task sample, Claude Sonnet 4.6 preferred task-specific appended clauses** over a fixed generic boilerplate pool. This demonstrates judge preference, not human-validated planning quality or external task generality.

### Caveats and Threats to Validity

1. **Not ROBUST_GO**: 33/50 falls in the STATISTICAL_GO band (32-34), not the ROBUST_GO band (>=35). The CI lower bound at 52.2% is barely above chance.

2. **Single-model judge bias**: All judges are Claude Sonnet 4.6 at temperature 0. This model may have a systematic preference for operational-sounding additions. Cross-model validation is needed.

3. **Weak anchors inflate the effect**: generic > anchor at 92% shows that ANY operational content substantially improves weak anchors. The primary endpoint measures only the *marginal* value of task-specificity beyond this boilerplate effect.

4. **Fixed generic pool may be weak**: The 7-sentence generic pool is narrow. A stronger task-aware boilerplate baseline (e.g., LLM-generated generic planning advice) might close the gap.

5. **Error gate fails narrowly**: 6 clear unique true-clause errors, failing the preregistered threshold by one. Adjudicator-sensitive.

6. **Same-model deterministic judging inflates agreement**: 3 judges at temp 0 are correlated samples. The 76% unanimity reflects model consistency, not evaluator independence.

7. **Internal task distribution**: These planning tasks may not generalize to other domains or task formats.

8. **Clause extraction failures**: Some true clauses contain meta-instruction artifacts, tautologies, or contradictions. The pipeline needs filtering to suppress these failure modes.

## 8. Artifacts

| Artifact | Path |
|----------|------|
| Full results JSON | `eval_results/diffusion_language/confirmatory_study_results.json` |
| Study manifest | `eval_results/diffusion_language/confirmatory_study_manifest.json` |
| Preregistration | `docs/reports/diffusion/CONFIRMATORY_STUDY_PREREGISTRATION.md` |
| Analysis script | scratchpad (session-local) |
