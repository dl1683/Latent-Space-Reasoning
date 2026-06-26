# Blinded Pairwise Pilot

**Tasks:** 10 (top by corrected fixed-anchor lift)
**Seed:** 20260626

| Task | Fixed Lift | Non-Rubric Lift | Anchor Score | Realized Score | Anchor=Best-of-N |
| --- | ---: | ---: | ---: | ---: | --- |
| `plan_478` | 0.250714 | 0.250714 | 0.431429 | 0.682143 | False |
| `plan_481` | 0.245357 | 0.245357 | 0.313929 | 0.559286 | True |
| `plan_465` | 0.222857 | 0.202857 | 0.252500 | 0.475357 | True |
| `plan_460` | 0.208214 | 0.208214 | 0.338929 | 0.547143 | False |
| `plan_488` | 0.208214 | 0.208214 | 0.322500 | 0.530714 | False |
| `plan_494` | 0.208214 | 0.208214 | 0.338929 | 0.547143 | False |
| `plan_508` | 0.207857 | 0.207857 | 0.377857 | 0.585714 | False |
| `plan_454` | 0.206786 | 0.186786 | 0.402500 | 0.609286 | False |
| `plan_458` | 0.206786 | 0.186786 | 0.443929 | 0.650714 | False |
| `plan_441` | 0.202857 | 0.202857 | 0.360000 | 0.562857 | True |

## Pilot Results: NO-GO

**Judge model:** Claude Sonnet 4.6 (blinded, randomized labels W/X/Y/Z)

| Metric | Score | Go Threshold | Verdict |
|--------|-------|-------------|---------|
| Aggregate beats anchor | 3/10 | >= 8/10 | **FAIL** |
| Aggregate beats keyword | 10/10 | >= 9/10 | PASS |
| Aggregate >= best-of-N | 4/10 | >= 7/10 | **FAIL** |

### Per-Task Results

| Task | Agg vs Anchor | Agg vs Keyword | Agg vs Best-of-N | Best Arm |
|------|:---:|:---:|:---:|---|
| plan_478 | LOSS | WIN | LOSS | anchor |
| plan_481 | LOSS | WIN | LOSS | best_of_n |
| plan_465 | **WIN** | WIN | **WIN** | aggregate |
| plan_460 | LOSS | WIN | LOSS | anchor |
| plan_488 | LOSS | WIN | LOSS | anchor |
| plan_494 | LOSS | WIN | LOSS | anchor |
| plan_508 | **WIN*** | WIN | **WIN*** | mixed |
| plan_454 | LOSS | WIN | TIE | anchor |
| plan_458 | LOSS | WIN | **WIN** | anchor |
| plan_441 | **WIN** | WIN | **WIN** | aggregate |

### Key Findings

1. **Keyword arm ranked LAST 10/10** — aggregate is definitively NOT keyword stuffing
2. **Root cause of failure**: the `_realize()` function (v2_replay.py:279-294) outputs meta-commentary like "Strengthen risk_awareness polarity" and "Add missing rubric requirement: X" — judges flag these as noise, not plan content
3. **Extraction mechanism works**: complement detection finds real gaps that judges recognize in the 3 winning tasks
4. **Realizer is the bottleneck**: complement packet JSON already contains natural-language clauses, but the realizer ignores them and writes aspect labels

### Disposition

Automatic score framing (+0.093 mean lift) is invalidated as a quality metric by this pilot.
Next step: clause-based realizer (clause_append_v1) that uses actual clause text from complement packet JSON.

## Original Go/No-Go Criteria

**Proceed to full study if:**
- Aggregate beats anchor on >= 8/10 tasks
- Aggregate beats keyword on >= 9/10 tasks
- Aggregate is not worse than best-of-N on >= 7/10 tasks
- No more than 1 aggregate has a serious error

**Abandon or redesign if:**
- Keyword beats/ties aggregate on 3+ tasks
- Aggregate beats anchor on fewer than 7/10 tasks
- Judge repeatedly flags aggregate as templated, incoherent, or preserves packet text

