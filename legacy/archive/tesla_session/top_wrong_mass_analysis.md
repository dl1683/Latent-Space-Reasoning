# Top Wrong Mass Analysis: Per-Task max(q_i) for Both Models

## Status: COMPUTED — Pending Codex Review

---

## 1. Summary Statistics

| Metric | Qwen3-4B (4-bit) | Qwen3-8B (8-bit) |
|--------|-------------------|-------------------|
| N (seeds) | 10 | 10 |
| Sensitive tasks | 19 | 19 |
| Frozen tasks (p=0) | 0 | 5 |
| Unanimous tasks (p=1) | 6 | 1 |
| Mean p (sensitive) | 0.363 | 0.326 |
| Mean max(q_i) (sensitive) | 0.200 | 0.195 |
| Max max(q_i) | 0.30 | 0.40 |
| Mean DM13 | 0.840 | 0.854 |
| Mean margin (p - max_qi) | +0.163 | +0.132 |
| Plurality accuracy | 18/25 = 72% | 14/25 = 56% |
| Theory prediction | 23/25 | 23/25 |

---

## 2. Key Finding: Margin Drives Plurality, Not Diversity

**Both models have nearly identical DM13** (0.840 vs 0.854 — answer diversity is equally high for both). The difference in plurality accuracy comes entirely from the **margin** between p and max(q_i):

- 4B: Mean margin = +0.163, positive margins on 10/19
- 8B: Mean margin = +0.132, positive margins on 11/19

Wait — 8B has MORE positive margins (11 vs 10) but WORSE plurality accuracy. Why?

### Resolution: Frozen Tasks

The 8B has **5 frozen tasks** (p=0, no seed gets correct) vs **0 frozen tasks** for 4B. These are automatic plurality failures. On sensitive tasks only:

| | 4B sensitive | 8B sensitive |
|---|---|---|
| Plurality correct | 12/19 | 14/19 |
| Unanimous correct | 6/6 | 0/1 |
| Frozen wrong | 0/0 | 0/5 |
| **Total** | **18/25** | **14/25** |

The 4B's plurality advantage is driven by (a) zero frozen tasks and (b) 6 unanimous tasks (always correct). On sensitive tasks, the 8B's plurality is actually BETTER (14/19 = 73.7% vs 12/19 = 63.2%).

### This is a critical reframe for the paper

The plurality voting advantage is NOT about the voting mechanism being better on 4B — it's about **oracle coverage**. 4B has 100% oracle (every task has at least one correct seed), while 8B has 80% (5 tasks have zero correct seeds). Plurality can only work where at least one correct answer exists.

---

## 3. Margin Structure

### Qwen3-4B Margins (sorted)
| Task | p | max(q_i) | Margin | Plurality |
|------|---|----------|--------|-----------|
| nest_016 | 0.80 | 0.10 | +0.70 | OK |
| nest_019 | 0.80 | 0.10 | +0.70 | OK |
| nest_015 | 0.70 | 0.10 | +0.60 | OK |
| nest_003 | 0.60 | 0.20 | +0.40 | OK |
| nest_006 | 0.50 | 0.10 | +0.40 | OK |
| nest_018 | 0.50 | 0.30 | +0.20 | OK |
| nest_023 | 0.40 | 0.20 | +0.20 | OK |
| nest_002 | 0.30 | 0.20 | +0.10 | OK |
| nest_007 | 0.30 | 0.20 | +0.10 | OK |
| nest_021 | 0.30 | 0.20 | +0.10 | OK |
| nest_004 | 0.20 | 0.20 | 0.00 | FAIL (tie) |
| nest_005 | 0.20 | 0.20 | 0.00 | FAIL (tie) |
| nest_009 | 0.10 | 0.10 | 0.00 | FAIL (tie) |
| nest_014 | 0.30 | 0.30 | 0.00 | OK (tie-won) |
| nest_017 | 0.30 | 0.30 | 0.00 | OK (tie-won) |
| nest_008 | 0.10 | 0.20 | -0.10 | FAIL |
| nest_013 | 0.10 | 0.20 | -0.10 | FAIL |
| nest_012 | 0.20 | 0.30 | -0.10 | FAIL |
| nest_022 | 0.20 | 0.30 | -0.10 | FAIL |

### Pattern: The boundary zone (margin = 0.00) is CRITICAL

5 tasks sit exactly at the boundary. Of these:
- 2 fail (correct and top-wrong are tied, tiebreak favors wrong)
- 2 succeed (correct and top-wrong are tied, tiebreak favors correct)
- 1 fails (maximum entropy — 9 distinct answers including correct, all count=1)

**At N=10, 26% of sensitive tasks (5/19) are at the boundary.** This is the core motivation for N=16 in Phase 1A — more seeds should push boundary tasks to resolvable.

---

## 4. The Tie Problem: Quantified

### Tie rates at N=10
| Model | Tasks with ties | Of which plurality correct | Of which plurality wrong |
|-------|----------------|--------------------------|------------------------|
| 4B | 6/19 (32%) | 2 | 4 |
| 8B | 3/19 (16%) | 2 | 1 |

### Expected tie rates at N=16 (prediction)
With N=16 (6 more seeds), the probability of exact ties drops substantially. For a task with p=0.2 and max(q_i)=0.2:
- At N=10: P(tie) is high because both are 2 counts out of 10
- At N=16: correct gets ~3.2 expected, top wrong gets ~3.2 expected, but wrong is split across multiple answers so concentration drops
- **Prediction: tie rate drops from ~25% to ~10% at N=16**

This is testable in Phase 1A.

---

## 5. DM13 vs max(q_i): Which Metric Matters?

### Codex R4 said: DM13 is insufficient. Use max(q_i).

The data confirms this:

| Task | DM13 | max(q_i) | Plurality |
|------|------|----------|-----------|
| nest_009 (4B) | 1.00 | 0.10 | FAIL |
| nest_006 (4B) | 1.00 | 0.10 | OK |
| nest_008 (4B) | 0.78 | 0.20 | FAIL |
| nest_018 (4B) | 0.60 | 0.30 | OK |

DM13 = 1.00 (perfect diversity) but plurality still fails on nest_009 because p = 0.10 (only 1 correct seed out of 10). DM13 = 0.60 (moderate diversity) but plurality succeeds on nest_018 because p = 0.50.

**DM13 measures error diversity (how spread out wrong answers are) but NOT the absolute position of the correct answer vs the top wrong answer.** The correct metric is margin = p - max(q_i).

### DM13 is a COMPONENT of the margin
- High DM13 → max(q_i) is lower (errors spread out) → margin improves
- But margin also depends on p (individual accuracy)
- **DM13 is necessary but not sufficient for plurality success**

---

## 6. Attractive Wrong Answers: Which Values Cluster?

### 4B: Wrong answers that appear 3+ times
| Task | Wrong answer | Count | Correct answer | Notes |
|------|-------------|-------|----------------|-------|
| nest_012 | 6528 | 3 | 6114 | Off by ~7% |
| nest_014 | 29 | 3 | 20 | Off by ~50% |
| nest_017 | 411 | 3 | 491 | Off by ~16% |
| nest_022 | 1 | 3 | 7179 | Catastrophic (trivial answer) |

### 8B: Wrong answers that appear 3+ times
| Task | Wrong answer | Count | Correct answer | Notes |
|------|-------------|-------|----------------|-------|
| nest_002 | 6497 | 3 | 6193 | Off by ~5% |
| nest_009 | 28 | 3 | 6 | Off by ~4.7x |

**Observation**: Wrong answer clustering falls into two categories:
1. **Near-miss errors** (6528 vs 6114, 411 vs 491): intermediate computation error, close to correct
2. **Trivial attractors** (1, 2): model collapses to a simple number, ignoring the problem

Category 2 is more dangerous for plurality voting because these trivial values may appear across DIFFERENT tasks, creating a systematic bias.

---

## 7. Cross-Model Comparison: Same Task Behavior

| Task | 4B p | 8B p | 4B plur | 8B plur | Notes |
|------|------|------|---------|---------|-------|
| nest_000 | 1.00 | 0.30 | OK | OK | 4B easy, 8B hard but diverse |
| nest_002 | 0.30 | 0.10 | OK | FAIL | 8B has concentrated wrong (6497×3) |
| nest_006 | 0.50 | 0.40 | OK | OK | Both robust |
| nest_007 | 0.30 | 0.20 | OK | FAIL | 8B tie at boundary |
| nest_008 | 0.10 | 0.20 | FAIL | OK | Reversed! 8B better on this task |
| nest_013 | 0.10 | 0.30 | FAIL | OK | Reversed! 8B better on this task |
| nest_014 | 0.30 | 0.10 | OK | FAIL | 4B better, higher p |
| nest_022 | 0.20 | 0.30 | FAIL | OK | Reversed! 8B better on this task |

**3 tasks where 8B plurality succeeds but 4B fails** (nest_008, nest_013, nest_022). These are tasks where the 8B model's perturbation reaches the correct answer more reliably despite being a harder model overall. This suggests **task-specific perturbation sensitivity** — the perturbation activates different latent knowledge in different models.

---

## 8. Implications for Phase 1A

### What we now know (before Phase 1A):
1. **max(q_i) is the binding constraint**, not DM13 — Codex was right
2. **N=10 produces 26% boundary tasks** — too many ties for clean results
3. **Oracle coverage matters more than plurality mechanism** — 4B's advantage is 100% oracle vs 80%
4. **Trivial attractors (1, 2) are systematic** — need extraction validation in Phase 1A
5. **Task-specific perturbation sensitivity** means 25 tasks gives reasonable coverage

### What Phase 1A temperature comparison must reveal:
1. **Does temperature change max(q_i)?** — If temperature produces HIGHER max(q_i) (more concentrated wrong answers), prefix wins
2. **Does temperature change oracle coverage?** — If temperature has lower oracle (fewer tasks with any correct seed), prefix wins
3. **Are trivial attractors worse under temperature?** — Temperature might increase probability of collapsing to "1" or "2"

### Pre-registered prediction update:
- Original H1: DM13_prefix > DM13_temperature — may be WRONG (DM13 is the wrong metric)
- **Revised H1**: max(q_i)_prefix < max(q_i)_temperature — top wrong concentration is lower under prefix
- **New H1b**: margin_prefix > margin_temperature — the gap between p and max(q_i) is larger under prefix

This should be flagged in Codex review of the pre-registration.

---

## 9. The "Tautology" Problem: Honest Assessment

Codex R4 correctly flagged that computing p and max(q_i) from the same data as plurality → algebraic tautology. Let me be precise about what IS and ISN'T tautological here:

### Tautological (do NOT claim as predictions):
- "Theory predicts 23/25 tasks" — yes, because p > max(q_i) is just restating which answer has the highest count

### NOT tautological (these ARE genuine findings):
- **max(q_i) ≈ 0.20 for both models** — this is an empirical observation about error structure under prefix perturbation. Different perturbation methods could produce different max(q_i).
- **DM13 ≈ 0.84-0.85** — this is an empirical observation about error diversity. Could be 0.5 or 0.95 under different methods.
- **Oracle coverage = 100% (4B) / 80% (8B) at N=10** — this constrains the ceiling for any voting method
- **Tie rate = 25-30% at N=10** — this constrains the statistical power of the comparison

### The genuine Phase 1A test:
Temperature changes p AND max(q_i) AND DM13 simultaneously. The question is whether the STRUCTURE of the answer distribution differs, not whether the tautological condition holds.

---

## 10. Data Appendix: Full Histograms

### Qwen3-4B sensitive tasks (19 tasks, n=10 seeds each)
- Median K_wrong = 5 (range: 2-9)
- Median DM13 = 0.83 (range: 0.60-1.00)
- Median max(q_i) = 0.20 (range: 0.10-0.30)
- Median p = 0.30 (range: 0.10-0.80)

### Qwen3-8B sensitive tasks (19 tasks, n=10 seeds each)
- Median K_wrong = 6 (range: 1-8)
- Median DM13 = 0.86 (range: 0.50-1.00)
- Median max(q_i) = 0.20 (range: 0.10-0.40)
- Median p = 0.30 (range: 0.10-0.90)

### Distribution of max(q_i) values:
| max(q_i) | 4B count | 8B count |
|----------|----------|----------|
| 0.10 | 5 | 4 |
| 0.20 | 9 | 9 |
| 0.30 | 5 | 4 |
| 0.40 | 0 | 2 |

**max(q_i) = 0.20 dominates** in both models (9/19 tasks each). This means the typical top-wrong answer appears 2 out of 10 times. At N=16, this would be ~3.2 expected — still manageable if p is 0.30+ (expected ~4.8).
