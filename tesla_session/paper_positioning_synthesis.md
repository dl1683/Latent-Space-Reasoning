# Paper Positioning Synthesis: Honest Assessment After Full Audit

## Status: CURRENT — Synthesizes all Codex reviews (R1-R5) and offline analyses

---

## 1. Where We Stand

### What we have (empirical results)
| Domain | Result | Strength |
|--------|--------|----------|
| Arithmetic (4B Q4, n=10) | Greedy 32% → Plurality 72%, Oracle 100% | Strong (same dataset) |
| Arithmetic (8B 8-bit, n=10) | Greedy 16% → Plurality 56%, Oracle 80% | Strong (same dataset) |
| Legal (4B Q4, n=5) | Oracle perturbation beats baseline 11/12 tasks | Strong (Codex blind review) |
| Planning (4B Q4, n=5) | Perturbation overcomes attention sinks | Moderate (5 tasks only) |
| DeepSeek (1.5B, n=10) | Perturbation HURTS (-1.6pp mean) | Important negative result |

### What we've discovered (theoretical/analytical)
| Finding | Novelty | Status |
|---------|---------|--------|
| p > max(q_i) condition for plurality | Known (List & Goodin 2001) | Cite, don't claim |
| max(q_i) ≈ 0.20 under prefix perturbation | Empirical (our data) | Needs comparison |
| DM13 ≈ 0.84-0.85 (high answer diversity) | Empirical (our data) | Needs comparison |
| Margin drives plurality, not raw diversity | Analytical insight | Novel framing |
| Oracle coverage bounds plurality ceiling | Analytical insight | Novel framing |
| Legal: best responses are atypical (anti-consensus) | Empirical finding | Novel |
| Legal: no verifier-free selector works | Negative result | Honest |

---

## 2. The Soft Reasoning Problem

### What Soft Reasoning (ICML 2025) already claims:
- Embedding perturbation → diverse reasoning paths ✓
- Greedy decoding from perturbed state ✓
- Improvement on math reasoning benchmarks ✓
- Placement analysis (which token to perturb) ✓

### What we'd claim that overlaps:
- "Embedding perturbation helps reasoning" → **CANNOT CLAIM** (they got there first)
- "Random perturbation produces diverse reasoning" → **CANNOT CLAIM** (they show this too)
- "Greedy decoding + noise = deterministic diversity" → **CANNOT CLAIM** (their method)

### What's genuinely ours:
1. **Random noise sufficiency**: No BO/optimization needed → but needs Phase 1A proof
2. **Verifier-free plurality voting**: No verifier/reward model → but only works for extractable answers
3. **Minority-correct regime analysis**: p > max(q_i) theory → known, but applied empirically
4. **Small quantized models** (4-bit 4B, 8-bit 8B) → regime difference, not mechanism
5. **Non-monotonic dose-response** → they report similar degradation patterns
6. **Legal/planning cross-domain** → they don't cover these, but we only have oracle-gap

---

## 3. Honest Venue Assessment (Codex R5)

| State | Venue Tier | Notes |
|-------|-----------|-------|
| Current data only | arXiv note / internal report | Not peer-reviewable |
| + Phase 1A (same 25 tasks) | Workshop submission | Discovery set, not held-out |
| + Phase 1A on held-out tasks | TMLR / ACL Findings | If prefix clearly beats temperature |
| + Soft Reasoning-style comparison | ICLR workshop→main borderline | If prefix mechanism is genuinely different |
| + Controlled quantization + multi-model | ICML/NeurIPS main | Surprising robust phenomenon |

### Most devastating review critique:
> "This is Soft Reasoning without Bayesian optimization, evaluated on 25 arithmetic problems, with a post-hoc plurality selector."

### Our best defense:
> "Random noise sufficiency + verifier-free selection + small quantized regime = different contribution. We show embedding perturbation creates answer-level error dispersion that enables verifier-free plurality voting, something Soft Reasoning's BO+verifier pipeline explicitly requires."

---

## 4. Minimum Viable Paper (MVP)

### Must-have experiments:
1. **Phase 1A (revised)**: Greedy vs prefix vs temp=0.6 vs temp=1.0, 25 tasks, 2 models, N=17
   - Primary: margin comparison (prefix vs temperature)
   - Required: answer histograms, max(q_i), p, oracle, trivial attractors
   - GPU: ~6 hours

2. **Held-out arithmetic tasks**: Generate new task set (30-50 tasks), run same protocol
   - Purpose: Move from exploratory to confirmatory
   - GPU: ~4-6 hours

3. **First-output-token perturbation** (Soft Reasoning proxy): Add as 5th operator
   - Purpose: Direct comparison to Soft Reasoning's mechanism
   - GPU: ~2 hours additional

### Should-have (strengthens to main-track):
4. **Controlled quantization sweep**: Qwen3-4B at FP16/8-bit/4-bit
   - Purpose: Isolate quantization effect from model size
   - GPU: ~6 hours

5. **Non-Qwen model**: Phi-2 or Mistral-small with same protocol
   - Purpose: Generalization beyond Qwen architecture
   - GPU: ~4 hours

### Nice-to-have:
6. Standard benchmark (GSM8K subset) for direct Soft Reasoning comparison
7. Legal conclusion extraction + voting experiment
8. Attention-sink analysis (mechanistic)

### Total GPU budget for MVP: ~14 hours
### Total GPU budget for main-track: ~30 hours

---

## 5. Paper Story (Given All Constraints)

### Title (working):
"Verifier-Free Selection from Random Embedding Perturbations in Small Quantized Language Models"

### One-paragraph abstract:
We show that random soft-prefix perturbation with greedy decoding produces diverse reasoning trajectories in small quantized language models (Qwen3-4B Q4, Qwen3-8B 8-bit), where each perturbation reaches the correct answer via qualitatively different intermediate computations. Unlike prior work requiring Bayesian optimization and verifiers (Soft Reasoning, ICML 2025), we demonstrate that simple plurality voting on extracted answers — without any verifier or optimization — can recover correct answers even when individual accuracy is below 50%. The mechanism relies on error dispersion: wrong answers are distributed across many distinct values while the correct answer clusters, satisfying the known List & Goodin (2001) condition p > max(q_i). On 25 arithmetic tasks, plurality voting improves from 32% (greedy) to 72% (N=10, Qwen3-4B) and from 16% to 56% (Qwen3-8B 8-bit). In legal reasoning, the oracle gap demonstrates that models have latent knowledge inaccessible via greedy decoding, though verifier-free selection in free-text domains remains an open challenge.

### Key claims (calibrated to evidence):
1. Random prefix perturbation + greedy decoding produces high answer-level diversity (DM13 ≈ 0.84) with low top-wrong concentration (max(q_i) ≈ 0.20) → **Empirical, needs temperature comparison**
2. Plurality voting exploits this diversity, working when p > max(q_i) (much weaker than p > 0.5) → **Theory known, application novel**
3. No verifier or optimization needed for extractable-answer domains → **Our main differentiator vs Soft Reasoning**
4. Oracle coverage demonstrates latent knowledge in legal/planning domains → **Strong**
5. The method fails on DeepSeek-1.5B and in free-text domains without extractable answers → **Honest boundary conditions**

---

## 6. What Phase 1A Must Show

### For the paper to work (minimum):
- Prefix margin > temperature margin (H1) — at least directionally with CI excluding 0
- OR: prefix and temperature are equivalent, but plurality mechanism is the contribution

### For the paper to be strong:
- Prefix produces lower max(q_i) than temperature (H1b) — mechanism claim
- Temperature dose-response exists (H4) — prefix position produces different diversity than temperature
- Out-of-sample margin prediction works (H3) — the theory is predictive, not just post-hoc

### If Phase 1A fails completely:
- If temperature = prefix on all metrics → paper is about plurality voting itself, not prefix perturbation
- Still publishable if held-out replication confirms plurality mechanism

---

## 7. Critical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Temperature produces equal diversity to prefix | Medium | High — paper loses mechanism claim | Pivot to "any diversity works" story |
| Extraction failures >5% under temperature | Medium | Medium — suspends confirmatory claims | Pre-validate extraction on temp outputs |
| Held-out tasks show different pattern | Low-Medium | High — discovery set is biased | Use diverse task difficulty distribution |
| Reviewer says "just Soft Reasoning lite" | High | High — desk rejection | Clear differentiation in intro + direct comparison |
| N=17 ties still frequent | Low | Low — fractional tie-breaking handles it | Report all 4 tiebreak policies |

---

## 8. Summary: The Honest Bottom Line

We have a **genuine scientific finding** (random embedding perturbation + plurality voting works in minority-correct regime without a verifier) that is **closely related to but distinct from** Soft Reasoning (ICML 2025). The distinction is:
- Theirs: optimization + verifier → our work doesn't need either
- Ours: random noise + plurality → works for extractable answers only

This is a **viable workshop-to-TMLR paper** with current data + Phase 1A. It becomes **main-track competitive** if we add:
- Held-out replication
- Direct Soft Reasoning comparison (output-token perturbation)
- Controlled quantization sweep
- Standard benchmark results

The **binding constraint** is Phase 1A: if prefix perturbation doesn't produce measurably different answer distributions than temperature sampling, the paper's mechanism claim collapses to "any diversity source + plurality voting works." That's still publishable but less interesting and harder to differentiate from Wang et al. (2502.11027) who already show diversity helps Best-of-N with discrete prompt rewording.
