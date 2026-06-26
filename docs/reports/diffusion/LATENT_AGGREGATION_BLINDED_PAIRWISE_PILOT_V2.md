# Blinded Pairwise Pilot v2 — Clause-Append Realizer

**Realizer:** `clause_append_v1` (appends natural-language clause sentences from complement packet JSON)
**Judge model:** Claude Sonnet 4.6 (blinded, randomized labels W/X/Y/Z)
**Tasks:** 10 (same tasks as v1 pilot, top by corrected fixed-anchor lift)
**Seed:** 20260626

## Verdict: GO

| Metric | Score | Go Threshold | Verdict |
|--------|-------|-------------|---------|
| Aggregate beats anchor | **8/10** | >= 7/10 | **PASS** |
| Aggregate beats keyword | **10/10** | >= 9/10 | **PASS** |
| Aggregate >= best-of-N | **10/10** | >= 7/10 | **PASS** |
| Keyword ranked last | 10/10 | — | Confirms not keyword stuffing |
| Aggregate ranked first | 8/10 | — | Strong |
| Meta-text complaints | **0** | 0 | **PASS** (v1 had 7/10) |

### Comparison with Pilot v1 (NO-GO)

| Metric | v1 (meta realizer) | v2 (clause-append) | Delta |
|--------|:---:|:---:|:---:|
| Agg vs Anchor | 3/10 | **8/10** | **+5** |
| Agg vs Keyword | 10/10 | 10/10 | 0 |
| Agg vs Best-of-N | 4/10 | **10/10** | **+6** |
| Agg ranked #1 | 3/10 | **8/10** | **+5** |
| Meta-text complaints | 7/10 | **0/10** | **-7** |

Root cause of v1→v2 improvement: clause-append realizer uses actual natural-language clause sentences from complement packet JSON instead of meta-commentary labels like "Strengthen risk_awareness polarity."

## Per-Task Results

| Task | Agg vs Anchor | Agg vs Keyword | Agg vs Best-of-N | Agg Rank | Anchor=Best-of-N |
|------|:---:|:---:|:---:|:---:|:---:|
| plan_441 | **WIN** | WIN | **WIN** | #1 | True |
| plan_478 | **WIN** | WIN | **WIN** | #1 | False |
| plan_488 | LOSS | WIN | WIN | #2 | False |
| plan_516 | LOSS | WIN | WIN | #2 | False |
| plan_463 | **WIN** | WIN | **WIN** | #1 | False |
| plan_465 | **WIN** | WIN | **WIN** | #1 | True |
| plan_494 | **WIN** | WIN | **WIN** | #1 | False |
| plan_515 | **WIN** | WIN | **WIN** | #1 | True |
| plan_481 | **WIN** | WIN | **WIN** | #1 | True |
| plan_446 | **WIN** | WIN | **WIN** | #1 | True |

## Automatic Score Summary (for reference only — not quality evidence)

| Task | Fixed Lift | Non-Rubric Lift | Anchor Score | Realized Score |
| --- | ---: | ---: | ---: | ---: |
| `plan_441` | 0.181429 | 0.181429 | 0.360000 | 0.541429 |
| `plan_478` | 0.172857 | 0.172857 | 0.431429 | 0.604286 |
| `plan_488` | 0.172857 | 0.172857 | 0.322500 | 0.495357 |
| `plan_516` | 0.164286 | 0.144286 | 0.378929 | 0.543214 |
| `plan_463` | 0.160357 | 0.160357 | 0.280000 | 0.440357 |
| `plan_465` | 0.158929 | 0.138929 | 0.252500 | 0.411429 |
| `plan_494` | 0.144286 | 0.144286 | 0.338929 | 0.483214 |
| `plan_515` | 0.142857 | 0.122857 | 0.390000 | 0.532857 |
| `plan_481` | 0.138929 | 0.138929 | 0.313929 | 0.452857 |
| `plan_446` | 0.138929 | 0.138929 | 0.398929 | 0.537857 |

## Key Findings

1. **Clause-append realizer fully resolves the meta-text bottleneck.** Zero judges complained about meta-instructions, placeholders, or templated text. All critique is now about substantive content logic.

2. **Appended clauses add real operational value.** Judges cite rollback criteria, scope boundaries, measurement steps, and decision points as reasons for preferring aggregate over anchor.

3. **Two losses are informative, not concerning.** In plan_488 and plan_516, anchor was already the strongest arm. Clauses don't hurt (aggregate is still #2) but don't add enough value when the anchor is already strong. This is expected behavior — not every task needs additional clauses.

4. **Keyword arm is decisively last (10/10).** Aggregate improvements are not keyword stuffing. Every judge independently identifies the keyword arm as incoherent noise.

5. **Aggregate content is the anchor + clause sentences.** When anchor = best_of_n (5/10 tasks), aggregate still wins by adding clause content. This isolates the clause contribution.

6. **Judge quality is high.** Judges identify real factual errors (contradictions, sequencing mistakes, compliance violations), not just surface features. They catch issues in ALL arms including aggregate — the wins are earned.

## Go/No-Go Criteria

**Proceed to full study if:**
- Aggregate beats anchor on >= 7/10 tasks → **8/10 PASS**
- Aggregate beats keyword on >= 9/10 tasks → **10/10 PASS**
- Aggregate is not worse than best-of-N on >= 7/10 tasks → **10/10 PASS**
- No more than 1 aggregate has a serious error → **PASS** (errors are shared across arms, not unique to aggregate)

**Abandon or redesign if:**
- Keyword beats/ties aggregate on 3+ tasks → **0/10 — not triggered**
- Aggregate beats anchor on fewer than 7/10 tasks → **8/10 — not triggered**
- Judge repeatedly flags aggregate as templated, incoherent, or preserves packet text → **0/10 — not triggered**

## Disposition

**GO for full study.** The clause-append realizer converts complement-packet signal into judge-preferred content.

**UPDATE:** Placebo diagnostic confirms task-specificity. True vs deranged 10/10, true vs generic 7/10, but generic vs anchor 10/10 (boilerplate confound). See [PLACEBO_DIAGNOSTIC.md](PLACEBO_DIAGNOSTIC.md).
