# Prefix vs Temperature: Mechanism Analysis and Phase 1A Predictions

## Status: THEORETICAL — Predictions to be tested in Phase 1A

---

## 1. The Critical Question

Phase 1A compares prefix perturbation vs temperature sampling. The core question:

**Does prefix perturbation produce qualitatively different answer diversity than temperature sampling?**

If YES → prefix perturbation is a genuinely new operator class, and CDE's operator portfolio has value.
If NO → any diversity-generating method works equally well, and prefix perturbation adds no unique value.

---

## 2. Mechanism Analysis

### Temperature Sampling (temp=0.6)
- **Input**: Fixed prompt (identical across all seeds)
- **Randomness source**: Stochastic token selection during generation
- **Distribution**: Samples from softmax(logits / 0.6) at each step
- **Trajectory divergence pattern**: GRADUAL
  - First few tokens: highly similar (top-1 token still dominates at temp=0.6)
  - Middle tokens: progressive divergence as small differences compound
  - End tokens: potentially very different, but through gradual drift
- **Error structure prediction**: Wrong answers arise from stochastic EXECUTION of the same reasoning strategy. Different seeds may make the same mistake at the same step (e.g., wrong multiplication result) because the reasoning PLAN is identical.
- **Expected answer diversity**: MODERATE — some diversity from sampling noise, but errors cluster around similar wrong intermediate values

### Prefix Perturbation (2-token random soft prefix, temp=0)
- **Input**: Modified prompt (different context per seed via 2 random embedding tokens)
- **Randomness source**: Random prefix embedding before generation begins
- **Distribution**: Deterministic greedy from shifted initial state
- **Trajectory divergence pattern**: IMMEDIATE
  - First token: already different (different context → different conditional distribution)
  - All subsequent tokens: entirely different trajectory (greedy from different starting state)
  - No gradual drift — each seed is a completely different deterministic path
- **Error structure prediction**: Wrong answers arise from genuinely different reasoning STRATEGIES. Each seed may decompose the problem differently (different operation order, different intermediate values).
- **Expected answer diversity**: HIGH — each seed follows a qualitatively different path, producing genuinely different wrong answers

### Key Mechanistic Difference
Temperature: same plan, stochastic execution → correlated errors
Prefix: different starting state, deterministic execution → uncorrelated errors

This is analogous to the difference in ensemble learning between:
- Bagging (temperature): same algorithm, different random subsets → moderate diversity
- Random subspace method (prefix): different feature views → higher diversity

---

## 3. Formal Predictions for Phase 1A

### Prediction P1: Answer diversity (PRIMARY)
- DM13_prefix > DM13_temperature
- Specifically: K_prefix (unique wrong answers) > K_temperature on matched tasks
- Expected magnitude: ~30% more unique wrong answers per task under prefix

### Prediction P2: Top wrong mass
- max(q_i)_prefix < max(q_i)_temperature
- Prefix spreads wrong answers more evenly; temperature concentrates on a few wrong values
- This directly determines plurality voting success

### Prediction P3: Plurality accuracy
- Plurality_prefix > Plurality_temperature (both > random pick, both > majority vote)
- If P2 holds, this follows from the theoretical condition p > max(q_i)

### Prediction P4: Oracle accuracy
- Oracle_prefix ≈ Oracle_temperature
- Both methods should achieve high oracle coverage with N=16
- If one operator reaches 100% oracle and the other doesn't, the difference is in TRAJECTORY DIVERSITY, not just ANSWER DIVERSITY

### Prediction P5: Response structure
- Temperature outputs share more common intermediate computations
- Prefix outputs use different solution strategies (different operation orderings)
- Testable: compute edit distance between response pairs within each operator

### Prediction P6: DeepSeek behavior
- On DeepSeek-1.5B: prefix perturbation HURTS (known from existing data)
- Temperature sampling on DeepSeek: unclear, but likely less harmful
- If temperature works on DeepSeek while prefix hurts: this reveals that prefix disrupts model-specific representations, while temperature merely varies execution

---

## 4. Counter-Predictions (What Would Falsify Our Hypothesis?)

### F1: Temperature produces same or higher answer diversity
- Would mean: diversity comes from ARITHMETIC structure (many possible intermediate errors), not from prefix perturbation specifically
- Impact: prefix perturbation is not special; the paper shifts from "prefix method" to "answer diversity mechanism"

### F2: Temperature plurality accuracy equals prefix plurality accuracy  
- Would mean: any diversity-generating method works equally well
- Impact: CDE's operator portfolio concept loses value; just use temperature + plurality vote

### F3: Temperature produces higher oracle accuracy than prefix
- Would mean: temperature accesses MORE correct paths (perhaps by smoothly exploring nearby regions)
- Impact: prefix perturbation is actually WORSE as an operator, despite higher answer diversity

### F4: DeepSeek shows same pattern as Qwen
- Would mean: model dependence is not about prefix specifically
- Impact: the Qwen-specific claim is weakened

---

## 5. EOS-REVISED PREDICTIONS (Post-Adversarial-Audit, supersedes P1-P6)

### CRITICAL CONTEXT: The EOS-Completion Mechanism

The adversarial audit revealed P(correct|EOS) = 1.000 across all 100 EOS responses (94 perturbation + 6 baseline). ALL wrong answers come from truncated responses. This fundamentally changes what Phase 1A is testing.

**The old question**: Does prefix produce more diverse wrong answers than temperature?
**The new question**: Does prefix produce higher completion (EOS) rate than temperature?

### Revised Prediction RP1: EOS Rate Hierarchy (PRIMARY)
- EOS_rate_prefix > EOS_rate_temp0.6 > EOS_rate_temp1.0 > EOS_rate_greedy
- **Reasoning**: Prefix shifts the initial computation state, potentially routing around reasoning loops. Temperature adds noise at every step, which may break SOME loops but also introduce token-level inefficiencies (taking less probable tokens → longer paths → more truncation).
- **Alternative**: EOS_rate_temp0.6 > EOS_rate_prefix (temperature is actually a better loop-breaker because it acts at every decision point, not just the start)

### Revised Prediction RP2: P(correct|EOS) Per Operator
- P(correct|EOS)_greedy ≈ 1.0 (observed: 6/6)
- P(correct|EOS)_prefix ≈ 1.0 (observed: 94/94)
- P(correct|EOS)_temp0.6 ≈ 0.90-0.95 (LOWER than prefix)
- P(correct|EOS)_temp1.0 ≈ 0.80-0.90 (LOWEST)
- **Key mechanism**: Greedy decoding from a shifted state preserves deterministic convergence to the correct answer. Temperature sampling may produce COMPLETED but WRONG responses — the model finishes but sampling noise corrupted intermediate calculations.
- **This is the critical differentiator**: If temperature maintains P(correct|EOS)≈1.0, then EOS rate is ALL that matters and whichever operator produces more EOS wins. If temperature drops P(correct|EOS), then prefix has a qualitative advantage: its completed responses are more reliable.

### Revised Prediction RP3: Plurality Accuracy
- Plurality_prefix ≈ EOS_rate_prefix (for tasks with any EOS seeds, plurality always succeeds; observed 21/21)
- Plurality_temperature ≈ f(EOS_rate_temp, P(correct|EOS)_temp)
- If RP2 holds (P(correct|EOS)_temp < 1.0): Plurality_prefix > Plurality_temperature even at equal EOS rates
- If RP2 fails (P(correct|EOS)_temp ≈ 1.0): Plurality ranking determined entirely by EOS rate

### Revised Prediction RP4: Token Length Distribution
- EOS responses under prefix: concentrated in 345-952 range (observed, mean 661)
- EOS responses under temp=0.6: expected slightly longer (mean ~700-750) due to sampling noise adding occasional detours
- EOS responses under temp=1.0: expected even longer (mean ~800+) OR bimodal (some very short from aggressive sampling shortcuts, some long from noise-extended reasoning)
- Truncated responses: all ~968 tokens regardless of operator (hitting the wall)

### Revised Prediction RP5: Answer Diversity Source
- OLD: prefix produces more diverse wrong answers than temperature (P1)
- NEW: answer diversity is a DOWNSTREAM consequence of truncation, not the primary mechanism
- DM13_prefix ≈ DM13_temperature (both high, because truncated responses produce random last-integers)
- max(q_i)_prefix ≈ max(q_i)_temperature (both ~0.20, same truncation noise)
- **If this holds**: diversity metrics are NOT the differentiator. EOS rate IS.
- **If this fails** (prefix DM13 > temperature DM13): there's additional diversity from initial-state divergence beyond just truncation noise

### Revised Prediction RP6: Where Temperature Beats Prefix
- On the 11 "no-EOS" tasks (where no perturbation seed completes):
  - Temperature may break SOME of these loops that prefix cannot
  - Because temperature acts at EVERY token, it has more opportunities to escape
  - Prefix only shifts the start; if the loop is deep in the reasoning chain, prefix may not reach it
- **Prediction**: temp=0.6 achieves EOS on 2-4 of the 11 tasks where prefix fails entirely
- **Counter-prediction**: temp=0.6 achieves EOS on 0-1 of these tasks (the loops are structural, not breakable by sampling)

### Revised Prediction RP7: Temperature + Prefix Interaction (Future Work)
- The most interesting unexplored operator: prefix perturbation + temperature sampling
- This combines initial-state shift (reaching different regions) with per-step noise (escaping local loops)
- **Prediction**: prefix + temp=0.3 achieves highest EOS rate of any operator
- NOT in Phase 1A (too many operators), but flagged for future work

---

## 6. The Three Scenarios for Phase 1A Under EOS Paradigm

### Scenario A: Prefix wins on EOS rate AND P(correct|EOS)
- EOS_rate_prefix > EOS_rate_temp (significantly)
- P(correct|EOS)_prefix > P(correct|EOS)_temp
- **Paper story**: "Prefix perturbation produces more efficient and more reliable reasoning paths. Temperature sampling is noisier at the token level, producing some completed-but-wrong answers. The deterministic nature of prefix generation preserves computational reliability while achieving path diversity."
- **Strength**: Clear mechanism, clean story, strong differentiator

### Scenario B: Prefix wins on EOS rate, P(correct|EOS) is equal
- EOS_rate_prefix > EOS_rate_temp
- P(correct|EOS)_prefix ≈ P(correct|EOS)_temp ≈ 1.0
- **Paper story**: "The fundamental mechanism is reasoning-path completion, not error diversity. Prefix perturbation is the most effective method for increasing completion rate because it changes the initial computational trajectory rather than adding noise along the way."
- **Strength**: Clean mechanism but weaker differentiator — temperature might catch up at different temperatures

### Scenario C: Temperature matches or beats prefix on EOS rate
- EOS_rate_temp ≥ EOS_rate_prefix
- **Paper story**: "Any diversity-generating method that increases reasoning-path completion rate improves plurality voting. Prefix perturbation and temperature sampling are equivalent operators. The contribution is the completion mechanism discovery itself, not the specific operator."
- **Strength**: More general finding, but harder to differentiate from Soft Reasoning

### Scenario D (Devastating): Temperature at 2048 tokens beats everything
- This is the "just raise max_new_tokens" scenario from Section 12 of the EOS document
- If greedy at 2048 + single-shot achieves >70% accuracy → the whole perturbation story narrows
- **Paper story**: Extremely difficult to position
- **Mitigation**: H6 token budget sweep will reveal this BEFORE spending 6 hours on full Phase 1A

---

## 7. Soft Reasoning Differentiation Under EOS Paradigm

### How the EOS finding changes the Soft Reasoning comparison

**Old framing**: "We use random noise, they use Bayesian optimization. We use plurality voting, they use a verifier."

**New framing**: "Both methods increase the probability that a model completes its reasoning within a fixed budget. Our method achieves this through initial-state diversity (multiple cheap random paths), theirs through directed search (optimized perturbation toward high-reward completions). Our approach trades per-sample quality for number of samples — random noise produces fewer completing paths per trial, but plurality voting over many trials recovers the correct answer verifier-free."

### The critical empirical test (Phase 1A + 5th operator)
- Soft Reasoning perturbs the FIRST OUTPUT TOKEN embedding
- We perturb the PREFIX (before input)
- Phase 1A's eventual 5th operator should be: first-output-token perturbation with random noise (not BO)
- This directly isolates: does perturbation POSITION matter for EOS rate?
- If prefix position >> output-token position on EOS rate → architectural contribution
- If equal → position doesn't matter, just the perturbation itself

### What Soft Reasoning probably also does (hypothesis)
- Their Bayesian optimization likely converges on perturbation directions that maximize EOS rate
- Their verifier catches the completed-but-wrong responses that their GP-selected perturbations might produce
- Our approach: throw many random perturbations, rely on P(correct|EOS)≈1.0 to make the verifier unnecessary
- **The question**: Is P(correct|EOS)≈1.0 a property of the TASK (arithmetic has unique correct answer) or the MODEL (greedy decoding from any good initial state converges)?
- If task-specific → our verifier-free claim is limited to extractable-answer domains (acknowledged)
- If model-general → our verifier-free claim is broadly applicable

---

## 8. What This Analysis Cannot Resolve (Needs GPU)

1. **EOS rate under temperature**: ZERO data. All predictions above are theoretical.
2. **P(correct|EOS) under temperature**: The single most important unknown for Phase 1A.
3. **Whether temperature breaks the 11 no-EOS loops**: determines if temperature accesses regions prefix cannot.
4. **Soft Reasoning position comparison**: needs 5th operator (future work after Phase 1A).
5. **Whether P(correct|EOS)≈1.0 is arithmetic-specific**: needs non-arithmetic tasks with extractable answers.

---

## 9. Why This Matters for the Paper

### The binding hypothesis:
If P(correct|EOS) ≈ 1.0 for ALL operators (greedy, prefix, temp=0.6, temp=1.0), then:
- EOS rate is the COMPLETE explanation of plurality accuracy
- The paper becomes about WHICH operator maximizes EOS rate, not about error diversity
- DM13, max(q_i), and margin become derived quantities, not primary metrics

### If P(correct|EOS) < 1.0 for temperature operators:
- Prefix has a qualitative advantage: deterministic convergence
- Error diversity matters again — temperature produces diverse wrong answers EVEN AMONG completed responses
- The paper has BOTH an EOS rate story AND an error structure story

### Either way, EOS rate is now the primary mechanism metric (per Codex R7).

---

## 10. Experimental Design Implications (Updated)

Phase 1A MUST be designed to distinguish the EOS scenarios:

1. **Matched compute**: Same N (17) for all stochastic operators
2. **Same tasks**: Identical 25 arithmetic tasks
3. **Same selector**: Plurality vote with pre-registered extraction and tie-breaking
4. **PRIMARY metric**: Per-task EOS rate per operator
5. **CRITICAL metric**: P(correct|EOS) per operator (any non-1.0 value is newsworthy)
6. **Key metric**: Per-task answer histogram comparison
7. **Statistical test**: Wilcoxon signed-rank on per-task EOS rate and margin
8. **Response-level**: Full token count, stop reason, 5-bucket classification (per Codex R7)
9. **Store full raw output**: Essential for post-hoc analysis of truncated responses

This is pre-registered in `phase1a_preregistration.md` v2.1.
