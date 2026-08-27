# Placebo Diagnostic: True vs Deranged vs Generic Clauses

**Purpose:** Test whether task-specific complement extraction beats generic operational boilerplate.
**Judge model:** Claude Sonnet 4.6 (blinded, randomized labels W/X/Y/Z)
**Tasks:** Same 10 tasks as pilot v2

## Arms

| Arm | Description |
|-----|-------------|
| **anchor** | Original model output (same as v2 pilot) |
| **true_clause** | Anchor + real extracted clauses from complement packets |
| **deranged_clause** | Anchor + clauses from a DIFFERENT task (rotation by 1) |
| **fixed_generic** | Anchor + count-matched generic sentences ("Define rollback criteria for the plan." etc.) |

## Results

| Comparison | Wins | Losses | Interpretation |
|------------|:---:|:---:|----------------|
| True vs Deranged | **10/10** | 0 | Task-specificity matters — definitively |
| True vs Anchor | **8/10** | 2 | Consistent with v2 pilot (8/10) |
| True vs Generic | **8/10** | 2 | True exceeds generic boilerplate |
| Generic vs Anchor | **10/10** | 0 | ANY operational boilerplate helps weak anchors |
| Deranged vs Anchor | 2/10 | **8/10** | Wrong-task content usually hurts |
| Generic vs Deranged | **9/10** | 1 | Generic > wrong-task content |

### Hierarchy: True >> Generic >> Anchor >> Deranged

## Per-Task Detail

| Task | True vs Anc | True vs Der | True vs Gen | Gen vs Anc | Der vs Anc | Notes |
|------|:---:|:---:|:---:|:---:|:---:|-------|
| plan_441 | WIN | WIN | WIN | WIN | LOSS | Deranged: "irrelevant database reconciliation" |
| plan_478 | WIN | WIN | WIN | WIN | LOSS | Deranged: "off-topic session rotation" |
| plan_488 | WIN | WIN | WIN | WIN | LOSS | Deranged: "unrelated ETL decimal-precision" |
| plan_516 | LOSS | WIN | LOSS | WIN | LOSS | True contaminated with prompt echo |
| plan_463 | WIN | WIN | WIN | WIN | LOSS | Deranged: "WAF/SQL injection off-topic" |
| plan_465 | WIN | WIN | WIN | WIN | LOSS | Deranged: "off-topic API migration plan" |
| plan_494 | WIN | WIN | WIN | WIN | LOSS | Deranged: "safety filter irrelevant" |
| plan_481 | WIN | WIN | WIN | WIN | win | Deranged generic enough to apply |
| plan_515 | WIN | WIN | WIN | WIN | win | Deranged generic enough to apply |
| plan_446 | LOSS | WIN | LOSS | WIN | LOSS | True clause tautological |

## Key Findings

1. **Task-specificity is confirmed.** True vs deranged 10/10 wins. Judges independently flag wrong-task content as "irrelevant," "off-topic," "contamination from a different problem" in 8/10 deranged arms. This is definitive: complement extraction finds task-relevant content, not generic boilerplate.

2. **True exceeds generic boilerplate.** True vs generic 8/10 wins. The 2 losses have identifiable quality issues in the true clauses (prompt echo contamination in plan_516, tautological clause in plan_446). When true clauses are clean, they consistently beat generic.

3. **Generic boilerplate also helps.** Generic vs anchor 10/10. Part of the v2 pilot win IS a baseline boilerplate effect — weak model outputs benefit from ANY operational structure. This is a confound that must be acknowledged.

4. **Wrong-task content hurts.** Deranged vs anchor 2/10 wins. The 2 wins occur when deranged source clauses are generic enough to apply cross-task (plan_481/515 — both migration-related). Confirms that specificity matters.

5. **Clause quality matters.** Tautological and contaminated true clauses lose to clean generic boilerplate. Extraction quality directly impacts preference.

## Interpretation for Paper

The honest claim is now three-part:

1. **Complement extraction discovers task-specific gaps** — proven by true vs deranged 10/10 separation
2. **Task-specific clauses add value beyond generic boilerplate** — proven by true vs generic 8/10
3. **Weak model outputs benefit from any operational structure** — proven by generic vs anchor 10/10

The v2 pilot's 8/10 agg-vs-anchor win is REAL but partially confounded by the generic boilerplate effect. The extraction mechanism's unique contribution is the task-specificity gradient: true >> generic >> anchor >> deranged.

## Disposition

Placebo diagnostic PASSES. Task-specificity confirmed. Ready for Codex R8 review and then preregistered 50-task study with all 4 arms.
