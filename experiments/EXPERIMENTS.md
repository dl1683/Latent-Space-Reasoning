# Experiments Log

Reverse chronological. Each entry links artifacts and summarizes findings.
Only Codex-validated conclusions are stated as "confirmed."

> Note: Pre-paradigm artifacts (V1-V15, conditioning comparison, etc.) were removed in
> commit 5055d58 (2026-03-05). Historical entries below reference these for context but
> the files no longer exist. Current experiment data lives in `experiments/*.json`.

---

## Word Problem Cross-Task (2026-03-08) — WEAKLY EXPLOITABLE, NOT REASONING IMPROVEMENT

**Purpose:** First non-arithmetic test. Does perturbation help on word problems?
**Config:** Qwen3-4B Q4, 25 word problems (medium/2step), 3 latents, 2 soft tokens
**Script:** `python -u experiments/run_latent_sensitivity.py --task-type word_problem --n-latents 3 --n-tasks 25 --control-mode random_noise --num-soft-tokens 2`
**Artifacts:** `sensitivity_random_noise_t2_results.json`, `word_problem_scout_log.txt`

### Results

| Condition | Accuracy | Delta |
|-----------|----------|-------|
| Baseline | 56% (14/25) | — |
| L0 | 56% | +0pp |
| L1 | 64% | +8pp |
| L2 | 56% | +0pp |
| **Mean** | **58.7%** | **+2.7pp** |
| Oracle | 64% | — |

McNemar: 2 gains, 0 losses, p=0.5 (not significant).

### Critical Finding: Token-Cap Truncation, Not Reasoning

**100% correlation between token cap and failure:**
- ALL 11 wrong answers hit the 1024 token cap
- ALL 14 correct answers used NO think mode (31-45 words direct answer)
- Perturbed responses: 0% hit cap (max=962 tokens)

The 2 rescued tasks (wp_013, wp_021) were truncation fixes: baseline ran out of tokens
before outputting the final answer; perturbation kept output under the cap.

**This is token budget management, not reasoning improvement.**

### Critical Analysis: Convergence vs Computation (Grading Audit)

Separate grading confound audit on nested arithmetic revealed:
- **Qwen3-4B answer-anywhere accuracy: 80% baseline, 82% perturbed** (negligible)
- **Last-integer accuracy: 32% -> 43%** (where the gain comes from)
- The model can already COMPUTE correct answers 80% of the time
- Perturbation helps CONVERGENCE (ending on the right answer), not computation
- See: `experiments/CRITICAL_ANALYSIS.md` for full analysis

---

## DeepSeek Dose-Response (2026-03-07) — COMPLETE (NON-MONOTONIC WINDOW CONFIRMED)

**Purpose:** Test whether non-monotonic 2-tok optimum generalizes beyond Qwen3-4B.
**Config:** DeepSeek-R1-Distill-Qwen-1.5B Q4, 25 sweet-spot tasks, 3 latents per condition, 1/2/3 soft tokens
**Scripts:**
- 1-tok: `python -u experiments/run_latent_sensitivity.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --task-type nested --difficulty sweet_spot --n-latents 3 --n-tasks 25 --control-mode random_noise --num-soft-tokens 1 --reuse-baseline ...t2...results.json`
- 3-tok: same with `--num-soft-tokens 3`
**Artifacts:** `sensitivity_sweet_spot_random_noise_t{1,3}_deepseekr1distillqwen1.5b_results.json`

### Results — Non-Monotonic Constructive Window (n=3 scout)

| Tokens | Baseline | Mean (n=3) | Delta | SD | Oracle | Rescued | Total Regress |
|--------|----------|------------|-------|-----|--------|---------|---------------|
| 1 | 76% | 64% | -12pp | 0.040 | 96% | 5 | 19 |
| 2 | 76% | 81.3% | +5.3pp | 0.046 | 100% | 6 | 10 |
| 3 | 76% | 80% | +4pp | 0.174 | 100% | 6 | 10 |

**NOTE**: The 2-tok +5.3pp was at n=3 only. At n=10, mean drops to 74.4% (-1.6pp).
See "DeepSeek 2-tok n=10" entry below for the paper-grade result.

Per-latent at 3-tok: [60%, 88%, 92%] — massive variance (Cochran's Q=9.5, p≈0.009)
Per-latent at 2-tok (n=3): [84%, 76%, 84%] — tight clustering (Cochran's Q=0.89, NS)
Per-latent at 1-tok: [64%, 68%, 60%] — all below baseline, tight clustering

### What We Learned (Codex-validated)
- **Non-monotonic constructive window confirmed on second model**: 1-tok HURTS, 2-tok best at n=3, 3-tok enters bifurcated regime
- 2-tok vs 3-tok difference is small in mean (+1.3pp) but massive in stability (SD: 0.046 vs 0.174)
- Cochran's Q significant ONLY at 3-tok: direction sensitivity emerges at higher token counts
- 1-tok: high-churn net-negative. Capability accessible (oracle rescues 5) but operating point mistuned
- **UPDATED**: 2-tok "stable optimum" only held at n=3; at n=10 mean drops below baseline while oracle remains 100%
- Pattern matches Qwen3-4B shape: 1-tok < baseline < 2-tok (peak at small n) > 3-tok
- McNemar at oracle: 1-tok p=0.0625, 2-tok p=0.031, 3-tok p=0.031

---

## Cross-Model: Qwen3-8B 8-bit (2026-03-07) — COMPLETE (STRONGLY POSITIVE)

**Purpose:** Quantization adjudication. Same architecture as 4-bit null, only quantization changed.
**Config:** 3 random noise vectors, 2 soft tokens each, 25 sweet-spot tasks, Qwen3-8B Q8
**Script:** `python -u experiments/run_latent_sensitivity.py --model Qwen/Qwen3-8B --task-type nested --difficulty sweet_spot --n-tasks 25 --n-latents 3 --control-mode random_noise --num-soft-tokens 2 --quantization 8bit`
**Total time:** 241.4 min
**Artifacts:** `sensitivity_sweet_spot_random_noise_t2_qwen38b_8bit_results.json`

### Results — STRONGLY POSITIVE (+16pp, reverses 4-bit null)
- Baseline: 16% (4/25) — LOWER than 4-bit's 24% (different task subset)
- Noise 1: 32% (8/25) = +16pp
- Noise 2: 24% (6/25) = +8pp
- Noise 3: 40% (10/25) = +24pp
- Mean conditioned: 32% (+16pp)
- Oracle (base|any noise): 15/25 = 60% (11 rescues, 0 oracle regressions)
- Per-latent regressions: nest_009 in L0/L1 (rescued in L2), nest_020 unstable (2/3)
- McNemar: gains=11, losses=0, p ≈ 9.8e-4 (HIGHLY significant)
- VRAM: 15.6 GB (vs ~9 GB at 4-bit)

### Quantization × Noise Interaction (Within-Model Control)
| Quant | Base | Mean Noise | Delta | Oracle | Rescued | Regress |
|-------|------|-----------|-------|--------|---------|---------|
| 4-bit | 24% | 25.3% | +1.3pp | 44% | 7/19 | 2 |
| 8-bit | 16% | 32% | +16pp | 60% | 11/21 | 0 |

### What We Learned
- **Quantization is a FIRST-CLASS modulator, not just a confound**
- 4-bit regularizes dynamics: helps default path but washes out perturbation
- 8-bit preserves richer trajectory landscape: lower baseline but much higher steerability
- Codex framing: "quantization fidelity modulates access to perturbation-sensitive reasoning trajectories"
- Within-model control is unusually clean: same architecture, tokenizer, tasks, perturbation budget
- Only 2/25 baseline tasks shared between 4-bit and 8-bit (nest_020, nest_024)
- Oracle sets overlap on only 9/25 — quantization changes WHICH trajectories are accessible

---

## Cross-Model: Qwen3-8B 4-bit (2026-03-06) — COMPLETE (NULL, NOW EXPLAINED BY QUANTIZATION)

**Purpose:** Cross-model validation. Larger same-family model. Tests scale dependence.
**Config:** 3 random noise vectors, 2 soft tokens each, 25 sweet-spot tasks, Qwen3-8B Q4
**Script:** `python -u experiments/run_latent_sensitivity.py --model Qwen/Qwen3-8B --task-type nested --difficulty sweet_spot --n-tasks 25 --n-latents 3 --control-mode random_noise --num-soft-tokens 2`
**Total time:** 124.5 min
**Artifacts:** `sensitivity_sweet_spot_random_noise_t2_qwen38b_results.json`

### Results — NULL (EXPLAINED: 4-bit too aggressive for 8B)
- Baseline: 24% (6/25) — BELOW 4B (32%), 4-bit quantization too aggressive
- Noise 1: 16% (4/25) = -8pp
- Noise 2: 24% (6/25) = +0pp
- Noise 3: 36% (9/25) = +12pp
- Mean conditioned: 25.3% (+1.3pp)
- Oracle (k=3): 13/25 = 52% (base|noise)
- Regressions: nest_016, nest_018 (both mod tasks correct at baseline)
- McNemar: gains=7, losses=2, p=0.18 (not significant)

### What We Learned
- Null result now EXPLAINED by 8-bit adjudication
- Serves as within-model quantization control (same arch, different quant = different result)
- 4-bit flattens trajectory landscape, reducing perturbation sensitivity

---

## Cross-Model: phi-2 (2026-03-06) — COMPLETE (POSITIVE, OUT-OF-FAMILY)

**Purpose:** Cross-model validation. Out-of-family (Microsoft, 2.7B). Critical for reviewer objection.
**Config:** 3 random noise vectors, 2 soft tokens each, 25 sweet-spot tasks, phi-2 (no quantization)
**Script:** `python -u experiments/run_latent_sensitivity.py --model microsoft/phi-2 --task-type nested --difficulty sweet_spot --n-tasks 25 --n-latents 3 --control-mode random_noise --num-soft-tokens 2`
**Total time:** 4.5 min
**Artifacts:** `sensitivity_sweet_spot_random_noise_t2_phi2_results.json`

### Results — POSITIVE (OUT-OF-FAMILY REPLICATION)
- Baseline: 12% (3/25) — very low, no CoT/thinking mode
- Noise 1: 16% (4/25) = +4pp
- Noise 2: 24% (6/25) = +12pp
- Noise 3: 16% (4/25) = +4pp
- Mean conditioned: 18.7% (+6.7pp, relative improvement = 56%)
- Noise oracle (k=3): 7/25 = 28% (4 tasks unique to noise: nest_000, nest_001, nest_009, nest_011)
- No tasks lost by noise

### What We Learned
- **Effect EXISTS in a completely different model family** (Microsoft phi-2, not Qwen)
- phi-2 has NO thinking mode — effect is NOT dependent on CoT scaffolding
- Low baseline limits absolute delta but relative improvement is substantial
- Task selectivity present: different directions solve different tasks
- The perturbation effect is NOT a Qwen-specific artifact

---

## DeepSeek 2-tok n=10 (2026-03-08) — COMPLETE (ORACLE/TASK-SELECTIVE, NOT MEAN-EFFECT)

**Purpose:** Paper-grade n=10 replication of DeepSeek 2-tok. Tests whether n=3 mean gain (+5.3pp) holds at scale.
**Config:** 10 random noise vectors, 2 soft tokens each, 25 sweet-spot tasks, DeepSeek-R1-Distill-Qwen-1.5B Q4
**Script:** `python -u experiments/run_latent_sensitivity.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --task-type nested --difficulty sweet_spot --n-tasks 25 --n-latents 10 --control-mode random_noise --num-soft-tokens 2 --reuse-baseline experiments/sensitivity_sweet_spot_random_noise_t2_deepseekr1distillqwen1.5b_n3_results.json`
**Artifacts:** `sensitivity_sweet_spot_random_noise_t2_deepseekr1distillqwen1.5b_results.json` (n=10), `..._n3_results.json` (n=3 preserved)

### Results — MEAN BELOW BASELINE, ORACLE 100%
- Baseline: 76% (19/25)
- Latent accuracies: [84, 76, 84, 76, 84, 68, 60, 64, 88, 60]
- Mean conditioned: 74.4% (-1.6pp) — BELOW baseline
- SD: 0.107 (high heterogeneity)
- **Oracle (k=10): 25/25 = 100%** — every baseline miss is reachable
- **McNemar: 6 gains, 0 losses, p=0.031** — oracle is statistically significant
- Cochran's Q: 19.07, p=0.025 (significant heterogeneity across directions)
- First 5 latents (from n=3 scout): avg 80.8%; latents 6-10: avg 68.0%

### What We Learned (Codex-validated)
- **n=3 scout was upward-biased** by sampling good directions; mean does NOT hold at n=10
- **DeepSeek is NOT a positive mean replication** — reframe as oracle/task-selective evidence
- **Oracle 100% with McNemar 6/0 IS significant** — every baseline failure is recoverable
- **Cochran Q significant** — DeepSeek enters heterogeneity regime at n=10 (like 3-tok Qwen3-4B)
- Qwen3-4B remains the only powered positive mean-effect model
- DeepSeek contributes: (1) 100% oracle in high-baseline model, (2) dose-response confirmation, (3) task-selective recoverability

---

## Cross-Model: DeepSeek-R1-Distill-Qwen-1.5B n=3 (2026-03-06) — COMPLETE (SUPERSEDED BY n=10)

**Purpose:** Cross-model validation. Different training regime (reasoning distillation), Qwen architecture.
**Config:** 3 random noise vectors, 2 soft tokens each, 25 sweet-spot tasks, DeepSeek-R1-Distill-Qwen-1.5B Q4
**Script:** `python -u experiments/run_latent_sensitivity.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --task-type nested --difficulty sweet_spot --n-tasks 25 --n-latents 3 --control-mode random_noise --num-soft-tokens 2`
**Total time:** 39.3 min
**Artifacts:** `sensitivity_sweet_spot_random_noise_t2_deepseekr1distillqwen1.5b_n3_results.json` (preserved from overwrite)

### Results — POSITIVE at n=3 (but did not hold at n=10)
- Baseline: 76% (19/25) — much higher than Qwen3-4B (32%)
- Noise 1: 84% (21/25) = +8pp
- Noise 2: 76% (19/25) = +0pp
- Noise 3: 84% (21/25) = +8pp
- Mean conditioned: 81.3% (+5.3pp)
- **Noise oracle (k=3): 25/25 = 100%** — solved ALL 6 tasks baseline missed
- No tasks lost by noise (noise-only oracle = 25/25)
- **NOTE**: n=3 was upward-biased. See n=10 entry above for paper-grade result.

### What We Learned
- Scout-level positive result that motivated n=10 scale-up
- Oracle coverage (100%) DID hold at n=10 — the oracle finding is robust
- Mean gain (+5.3pp) did NOT hold at n=10 — sampling artifact from good directions

---

## Cross-Model: Qwen3-1.7B (2026-03-06) — COMPLETE (NEGATIVE)

**Purpose:** Cross-model validation. Same family, smaller scale.
**Config:** 3 random noise vectors, 2 soft tokens each, 25 sweet-spot tasks, Qwen3-1.7B Q4
**Script:** `python -u experiments/run_latent_sensitivity.py --model Qwen/Qwen3-1.7B --task-type nested --difficulty sweet_spot --n-tasks 25 --n-latents 3 --control-mode random_noise --num-soft-tokens 2`
**Total time:** 95.9 min
**Artifacts:** `sensitivity_sweet_spot_random_noise_t2_qwen17b_results.json`

### Results — NULL
- Baseline: 28% (7/25)
- Noise 1: 28% (7/25) = +0pp
- Noise 2: 28% (7/25) = +0pp
- Noise 3: 32% (8/25) = +4pp
- Mean conditioned: 29.3% (+1.3pp) — within noise
- Prior evidence: 1.7B hits capacity boundaries on other tasks (V4 tree traversal: tail=0.000)
- Codex analysis: genuine null for large effect, but n=3 cannot rule out small (+4-8pp) effect

### What We Learned
- The +19.6pp effect does NOT simply transfer to a smaller same-family model
- Perturbation sensitivity appears model/scale contingent
- 1.7B calibrated RMS (0.0345) is 57% higher than 4B (0.0220) — may need different dose
- Paper framing: boundary condition, not contradiction

---

## 2-Token n=10 Replication (2026-03-06) — COMPLETE

**Purpose:** EXISTENTIAL test of solve-count equalization. The n=3 result [15,15,15] (p=0.031)
was the strongest equalization evidence. Needed n=10 to confirm or refute.
**Config:** 10 random noise vectors, 2 soft tokens each, 25 sweet-spot tasks, Qwen3-4B Q4
**Script:** `python -u experiments/run_latent_sensitivity.py --task-type nested --difficulty sweet_spot --n-latents 10 --n-tasks 25 --control-mode random_noise --num-soft-tokens 2`
**Total time:** 284.7 min
**Artifacts:** `sensitivity_sweet_spot_random_noise_t2_results.json` (n=3 backup: `_t2_n3_results.json`)

### Results
- Solve counts: [15, 15, 15, 13, 14, 13, 11, 12, 9, 12], mean=12.9/25=51.6%, SD=1.87
- **Equalization DEAD**: p=0.659 vs heterogeneous iid null (100k MC sims). Small-sample noise.
- **Oracle**: 25/25 = 100% at k=10. Zero frozen tasks.
- At k=3: 19.2/25 = 76.8% (avg over C(10,3) subsets)
- 3-tok oracle (20/25) is strict subset of 2-tok oracle (25/25)
- 5 tasks unique to 2-tok vs 3-tok: {nest_002, nest_005, nest_008, nest_012, nest_021}
- nest_008 is the ONLY task uniquely solvable by a single condition (2-tok only)
- Categories at k=10: 6 unanimous, 0 frozen, 19 sensitive
- Paper fully updated: oracle leads, equalization reported as negative result

### What We Learned
- Equalization was small-sample noise (n=3). Does not replicate at n=10.
- Oracle efficiency is the real story: 2-tok covers ALL tasks with 10 directions.
- 2-tok still has highest mean accuracy (51.6% > 44% for 3/8-tok).
- Non-monotonic dose-response holds at n=10. Paper restructured accordingly (Codex review).

---

## Shi et al. Discrete Token Control — 2 Tokens (2026-03-06) — COMPLETE

**Purpose:** Head-to-head comparison of Shi et al. (2025) discrete punctuation tokens vs our
continuous random noise at 2 tokens. Key positioning experiment.
**Config:** "/" and "?" tokens each repeated 2 times, 25 sweet-spot tasks, Qwen3-4B Q4, baseline reused
**Script:** `python -u experiments/run_latent_sensitivity.py --task-type nested --difficulty sweet_spot --n-latents 2 --n-tasks 25 --control-mode discrete_tokens --discrete-token "/,?" --num-soft-tokens 2 --reuse-baseline experiments/sensitivity_sweet_spot_random_noise_t2_results.json`
**Total time:** 54.2 min
**Artifacts:** `sensitivity_sweet_spot_discrete_tokens_t2_results.json`

### Results
- "/" (2 tokens): 9/25 = 36% (+4pp vs baseline)
- "?" (2 tokens): 12/25 = 48% (+16pp vs baseline)
- Discrete mean: 42% (+10pp)
- **Random continuous 2-tok mean: 51.6% (+19.6pp)** — 9.6pp advantage
- Discrete oracle (2 tokens): 12/25 = 48%
- Combined oracle (discrete + baseline): 15/25 = 60%
- For comparison: random continuous oracle at k=2 (best pair) would be higher

### Token-Level Observations
- Native RMS: "/" = 0.02027, "?" = 0.02025 (vs target 0.02195) — close to embedding scale
- "/" barely exceeds baseline (+4pp), "?" substantially better (+16pp)
- "?" outperforms 3-tok random (44%) but trails 2-tok random (51.6%)
- Both discrete tokens solve a SUBSET of what random noise solves

### What We Learned
- Continuous embedding-space perturbation produces larger effects than discrete tokens at matched count
- The 9.6pp gap (42% vs 51.6%) supports the paper's positioning: continuous space has richer structure
- "?" > "/" suggests even within discrete tokens, token identity matters (semantic content?)
- Discrete tokens DO improve over baseline (+10pp mean), consistent with Shi et al.'s observation
- But continuous random noise adds an additional ~10pp on top of discrete

---

## Think-Gate Probe (2026-03-06) — COMPLETE

**Purpose:** Test whether perturbation activates <think> mode via first-token logit probe.
**Config:** 25 sweet-spot tasks, single forward pass per condition, 9 conditions
**Script:** `python -u experiments/run_think_gate_probe.py --n-tasks 25`
**Artifacts:** `think_gate_probe_results.json`

### Results — CRITICAL NEGATIVE FINDING
- <think> is rank=1 with probability >99.99% for ALL conditions including unperturbed baseline
- No meaningful variation: baseline=99.994%, noise_2tok=99.997%, discrete_/=99.998%
- **Think-mode gating hypothesis FALSIFIED**: model always generates <think> first
- The 16% baseline think rate was a visibility artifact (tag stripping in stored responses)
- Paper updated: removed "think-mode gating" claim, reframed as trajectory modulation

### What We Learned
- The mechanism is NOT mode activation — it's trajectory modulation within already-active think mode
- Force-think decomposition reframed: explicit <think> prefix = trajectory priming, not mode switch
- Perturbation operates on the reasoning chain content, not on whether the model reasons

---

## 3-Token Dose-Response (2026-03-05) — COMPLETE

**Purpose:** Fill critical gap in dose-response curve. Confirm non-monotonic peak at 2 tokens.
**Config:** 10 random noise vectors, 3 soft tokens each, 25 sweet-spot tasks, Qwen3-4B Q4
**Script:** `python -u experiments/run_latent_sensitivity.py --task-type nested --difficulty sweet_spot --n-latents 10 --n-tasks 25 --control-mode random_noise --num-soft-tokens 3`
**Total time:** 348 min
**Artifacts:** `sensitivity_sweet_spot_random_noise_t3_results.json`

### FINAL Results (N1-N10)
- Solve counts: [11, 11, 11, 10, 13, 12, 9, 9, 12, 12], mean=11.0/25=44.0%, SD=1.33
- **Equalization**: p=0.335 vs heterogeneous iid (NS, below median — slightly suppressed but not significant)
- Categories: 5 unanimous, 5 frozen, 15 sensitive
- Frozen: {nest_002, nest_005, nest_008, nest_012, nest_021}
- **Oracle (k=10): 20/25 = 80%** — adds NO tasks beyond 2-tok oracle (22/25=88%)
- **2-tok oracle (3 dirs) = 88% >> 3-tok oracle (10 dirs) = 80%**
- At matched n=3: 2-tok freezes 3 tasks, 3-tok freezes 9.9 (mean over C(10,3)=120 combos)
- 2 tasks frozen at 3-tok but solvable at 2-tok (nest_012, nest_021→actually also frozen at 2tok)
- Over-perturbation contracts reachable set (Codex 2026-03-05f)
- N10 broke nest_014 out of frozen (was 0/9, now 1/10)

---

## Deep Data Analysis + Paper Figures (2026-03-05) — CODEX VALIDATED

**Purpose:** Comprehensive statistical analysis of all existing data. 7 paper figures generated.
**Artifacts:** experiments/figures/fig[1-7]_*.png, experiments/analysis_summary.md
**Script:** experiments/create_figures.py

### Key Findings (Codex-Validated)
1. Strict categorization: 2 unanimous, 1 frozen, 22 sensitive (cross-condition)
2. Solve-count equalization at 2-tok: all 3 latents solve exactly 13/22 sensitive (std=0.00)
3. Full oracle: 24/25 = 96% (only nest_008 unsolvable)
4. Task-specific resonance: nest_005 (1-tok only), nest_021 (8-tok only)
5. McNemar: Latent 0 purely additive (7 gains, 0 losses, p≈0.023)
6. Cohen's h = 0.570 at 2-tok (medium-large effect)
7. Timing confound resolved: Latent 0 achieves 60% at baseline timing (73.6s)
8. Headroom analysis: 41.2% error reduction vs Shi's 2.2-16.7%

---

## Force-Think Baseline (2026-03-05) — CONFIRMED

**Purpose:** Decompose perturbation effect into think-mode gating vs noise-specific contribution.
**Config:** Prepend `<think>\n` to force think mode without noise prefix.
**Results:** 40% (10/25) — think mode alone gives +8pp, noise gives additional +20pp.
**Codex validated:** Noise contributes 2.5x more than think mode at 2-tok optimum.

---

## 2-Token Dose-Response (2026-03-05) — NON-MONOTONIC PEAK

**Purpose:** Test 2-token random noise prefix as part of dose-response sweep.
**Config:** 3 random noise vectors, 2 soft tokens each, RMS=0.022, 25 sweet-spot tasks, Qwen3-4B Q4, thinking mode
**Script:** `python experiments/run_latent_sensitivity.py --task-type nested --difficulty sweet_spot --n-latents 3 --control-mode random_noise --num-soft-tokens 2`

### Results

| Tokens | Accuracy | Change | Std |
|--------|----------|--------|-----|
| 0 (baseline) | 32.0% | -- | -- |
| 1 | 42.7% | +10.7pp | 2.3% |
| **2** | **60.0%** | **+28pp** | **0.0%** |
| 8 | 44.4% | +12.4pp | 7.4% |

**Zero variance**: all 3 latent vectors produced exactly 15/25 correct (60%).
**7 tasks fixed** (wrong→right), **2 tasks regressed** (right→wrong) vs baseline.

### What We Learned
1. **Non-monotonic optimum** — 2 tokens is the best condition tested (+28pp vs baseline)
2. **Contradicts threshold/saturation story** — more tokens is NOT better, and 1 token is NOT 89% of peak
3. **Zero variance is remarkable** — 3 independent random vectors at 2 tokens all produce identical accuracy
4. **Overshoot at 8 tokens** — too many random tokens likely causes excessive exploratory behavior
5. **Needs replication** — n=3 latents and n=25 tasks is fragile; need 4-token and 16-token data points

### Artifacts
- `experiments/sensitivity_sweet_spot_random_noise_t2_results.json`
- Ledger entry: 2-token-dose-response

---

## Error Taxonomy Analysis (2026-03-04) — REDISTRIBUTION, NOT CLEAN IMPROVEMENT

**Purpose:** Classify per-task failure patterns across all conditions to understand what "improvement" means.
**Script:** `experiments/analyze_error_taxonomy.py`

### Results (Codex CLI Reviewed)

| Metric | Value |
|--------|-------|
| Baseline correct | 8/25 (32%) |
| Baseline failures FIXED by 8 tokens (>50% recovery) | 3/17 (nest_006=70%, nest_010=100%, nest_015=70%) |
| Baseline successes REGRESSED by 8 tokens | 6/8 (nest_003=30%, nest_007=30%, nest_014=10%) |
| Always broken regardless of condition | 2 (nest_005, nest_008) |

**Codex CLI Verdict:** The +12pp mean improvement is REDISTRIBUTION — random tokens destabilize AND re-stabilize behavior across different tasks. The net effect is positive on average, but individual task reliability drops. At n=25, the signal is "promising but fragile."

### Mechanism Implications
- Effect is NOT additive computation: tasks that work get broken
- Consistent with attention perturbation/stochastic resonance
- 1-token captures 89% → threshold/trigger effect, not cumulative

### Qualitative Output Analysis (Codex CLI Reviewed)
**Recovery pattern** (baseline wrong → 8-token correct):
- Baseline gets stuck in "formal presentation mode" (LaTeX, structured steps, truncates before computing)
- Random tokens shift model into "informal stream-of-consciousness" that actually computes

**Regression pattern** (baseline correct → 8-token wrong):
- Baseline uses efficient structured computation, completes within token budget
- Random tokens cause rambling exploration, runs out of max_new_tokens (1024)

**Generation time correlation:**
- Correct answers: ~60s (completes before token budget)
- Wrong answers: ~81s (hits max_new_tokens = 1024)
- Random tokens shift output POLICY, not reasoning quality

**Codex CLI interpretation:**
> "Not 'reasoning improved magically,' but 'output policy changed.' Stochastic resonance / attractor switching is a better core mechanism than attention sink."

### Response Style Analysis
- Baseline correct: 19.4 numbers/resp, 13.0 operations/resp (efficient)
- Baseline wrong: 28.8 numbers/resp, 7.8 operations/resp (verbose, not computing)
- Random tokens increase operation density on recovered tasks

---

## NEXT: Priority Experiments (Updated 2026-03-08)

See `experiments/RUN_QUEUE.md` for full details and commands.

| # | Experiment | Purpose | Status |
|---|-----------|---------|--------|
| 1 | 8B 8-bit n=10 | Firm up within-model quant control | Queued (needs restart with checkpointing) |
| 2 | Word problem cross-task | External validity / task domain breadth | Queued |

---

## Nested-Easy Noise Control (2026-03-04) — WARM-START CONFIRMED ON EASY TASKS

**Purpose:** Replicate the sweet-spot noise control on easy_nested tasks (92% baseline) to confirm that latent direction is irrelevant across difficulty levels.
**Config:** 5 noise vectors (4 completed), 25 easy_nested tasks, torch.randn at target_rms=0.022 (NOT through W), same tasks as original latent-projected sensitivity, Qwen3-4B Q4, thinking mode
**Script:** `python experiments/run_latent_sensitivity.py --task-type nested --difficulty easy_nested --n-latents 5 --n-tasks 25 --control-mode random_noise`
**Note:** Noise 5 killed at task 18/25 (VRAM). 4 complete vectors sufficient.

### Results

| Condition | Mean Acc | Std | Range | Cochran's Q | p |
|-----------|---------|-----|-------|-------------|---|
| Baseline (no prompt) | **92%** | - | - | - | - |
| Random noise (4 vectors) | **85.0%** | 8.9% | [76%, 96%] | 6.10 | 0.107 |
| Latent-projected (10 vectors, prior exp) | **84.0%** | 8.4% | [64%, 96%] | 23.2 | 0.006 |

**Mann-Whitney U test (noise vs latent): U=19.5, p=1.000** — indistinguishable

### What We Learned (Codex-Validated)
1. **Direction irrelevant on easy tasks too** — noise (85.0%) matches latent-projected (84.0%), p=1.0
2. **Consistent with sweet-spot result** — same pattern at 32% and 92% baseline
3. **Cochran's Q non-significance is a power artifact** — k=4 vs k=10 degrees of freedom
4. **Direction-agnostic mechanism confirmed cross-difficulty** — no remaining difficulty regime where direction differentiates
5. **Both conditions hurt on easy tasks** — mean below 92% baseline in both noise and latent-projected
6. **Warm-start mechanism is confirmed as direction-agnostic** — improvement is from token presence, not content

### Artifacts
- `experiments/sensitivity_easy_nested_random_noise_results.json`
- `experiments/sensitivity_nested_easy_random_noise_log.txt`
- Ledger entry: nested-easy-noise-control

---

## Warm-Start Control (2026-03-04) — WARM-START CONFIRMED, MECHANISM IS DIRECTION-AGNOSTIC

**Purpose:** Determine if the +12pp improvement from soft prompt conditioning is due to latent direction (through W projection) or a generic warm-start from any embedding tokens at the correct RMS scale.
**Config:** 4 random noise soft prompts (torch.randn scaled to target_rms=0.022, NOT through W), same 25 sweet_spot tasks, Qwen3-4B Q4, thinking mode
**Script:** `python experiments/run_latent_sensitivity.py --task-type nested --difficulty sweet_spot --n-latents 10 --n-tasks 25 --control-mode random_noise`
**Note:** 4/10 noise vectors completed before VRAM degradation killed experiment. 4 vectors sufficient for conclusion.

### Results

| Condition | Mean Acc | Std | Range | Sig vs Baseline |
|-----------|---------|-----|-------|-----------------|
| Baseline (no prompt) | **32%** | - | - | - |
| Random noise (4 vectors) | **44.0%** | 3.3% | [40%, 48%] | 0/4 |
| Latent-projected (10 vectors) | **44.4%** | 7.4% | [36%, 56%] | 3/10 |

**Mann-Whitney U test: p = 1.000** (noise and latent-projected are statistically indistinguishable)

### What We Learned (Codex-Validated)
1. **WARM-START CONFIRMED** — random noise matches latent-projected (44.0% vs 44.4%, p=1.0)
2. **Latent direction carries NO detectable signal** — the W projection is irrelevant
3. **Improvement comes from PRESENCE of 8 embedding tokens** at correct RMS, not their content
4. **The entire latent→W→soft_prompt pipeline adds no value** over random noise
5. **Directional search doesn't add benefit** — the improvement mechanism is direction-agnostic, so optimization in latent space doesn't outperform random prefix tokens
6. **Noise has LESS variance** (std 3.3% vs 7.4%) — W-projection adds noise-like variance, not signal
7. **New finding: warm-start is real and reproducible** — +12pp free improvement from random tokens

### Implications
- Directional search in latent space doesn't add benefit over random prefix tokens (mechanism is direction-agnostic)
- Orthonormal projection, Poincare ball, curvature, etc. add no value
- **Focus shifts to characterizing the warm-start mechanism** — understanding why prefix tokens help is the key question
- Need to test: multi-model generality, task diversity, token count dose-response

### Artifacts
- `experiments/sensitivity_sweet_spot_random_noise_results.json`
- `experiments/sensitivity_sweet_spot_random_noise_log.txt`
- Ledger entry: warm-start-control

---

## Sweet-Spot Sensitivity (2026-03-03) — POSITIVE SIGNAL, WARM-START CONFOUND UNRESOLVED

**Purpose:** Test landscape exploitability on harder tasks where baseline is low (~60% target), giving room for improvement.
**Config:** 10 random Euclidean latents, 25 sweet_spot tasks (2-3 ops, 2-digit x 2-digit), thinking mode, Qwen3-4B Q4
**Script:** `python experiments/run_latent_sensitivity.py --task-type nested --difficulty sweet_spot --n-latents 10 --n-tasks 25`
**Runtime:** 5.7 hours (thinking mode)

### Results

| Condition | Accuracy |
|-----------|----------|
| Zero-shot baseline | **32%** |
| L6 (best) | **56%** (p=0.011) |
| L3, L8 | **52%** (p=0.030) |
| L9 | 48% (p=0.070) |
| L2, L4 | 44% |
| L7 | 40% |
| L0, L1, L5 | 36% |
| Mean conditioned | **44.4%** (+12.4%) |

**Cochran's Q = 8.3, p = 0.504** (NOT significant — latents don't differ from each other)
**3/10 latents individually significant** vs baseline (binomial test, p < 0.05)

### What We Learned
1. **All 10 latents beat baseline** — conditioning consistently improves accuracy (+4% to +24%)
2. **3/10 individually significant** — L6 (56%, p=0.011), L3 (52%, p=0.030), L8 (52%, p=0.030)
3. **Cochran's Q NOT significant** — latents don't differ enough from each other (p=0.504)
4. **CRITICAL CONFOUND: warm-start effect** — since all latents improve and they don't differ from each other, the improvement may be from ANY soft prompt tokens at correct RMS, not from latent direction
5. **Must run random-noise control** to distinguish direction signal from generic warm-start

### Artifacts
- `experiments/sensitivity_sweet_spot_results.json`
- Ledger entry: sweet-spot-sensitivity

---

## No-Think Sensitivity (2026-03-03) — LANDSCAPE IS FLAT WITHOUT THINKING

**Purpose:** Test whether the exploitable landscape persists when Qwen3 thinking mode is disabled (--no-think). This determines whether chain-of-thought is the mechanism through which soft prompts influence accuracy.
**Config:** 20 latents (9 completed before CUDA error), 25 easy_nested tasks, --no-think --max-new-tokens 128, Qwen3-4B Q4
**Script:** `python experiments/run_latent_sensitivity.py --task-type nested --difficulty easy_nested --n-latents 20 --no-think --max-new-tokens 128`

### Results

| Metric | With Thinking | Without Thinking |
|--------|--------------|-----------------|
| Baseline | 92% | 72% |
| Best latent | 96% | 72% |
| Worst latent | 64% | 68% |
| Range | **32%** | **4%** |
| Time per call | ~60-120s | ~8-12s |

### What We Learned
1. **Chain-of-thought IS the steering mechanism** — without it, all latents produce nearly identical accuracy
2. **No-think landscape is flat** — 4% range (not exploitable) vs 32% with thinking
3. **Soft prompts influence reasoning chains, not direct computation** — the <think> block is where conditioning has its effect
4. **No-think is ~10x faster** but provides no exploitable landscape for evolution
5. **Must use thinking mode for evolution experiments** — there's no shortcut

### Implication
All evolution experiments must use thinking mode (max_new_tokens=1024) despite being ~10x slower per call. Optimize via fewer calls (smaller population, fewer tasks per gen), not by disabling thinking.

### Artifacts
- Partial results (CUDA error after 9/20 latents): not saved as JSON
- Ledger entry: sensitivity-nothink-flat

---

## V15b: Accuracy Fitness Geometry Isolation (2026-03-03) — LOCAL EVOLUTION FAILS

**Purpose:** Re-run V15 geometry isolation with accuracy-based fitness (fixing Goodhart's Law from V15a where dense_score was used). Tests whether hyperbolic vs Euclidean mutation geometry matters when fitness correctly tracks task accuracy.
**Config:** 3 conditions, 1 seed, 80 train / 25 test easy_nested tasks, 3 gens x 4 pop, Qwen3-4B Q4, thinking disabled (--no-think --max-new-tokens 128)
**Script:** `python experiments/run_v15_geometry_isolation.py --diagnostic --task-type nested --difficulty easy_nested --fitness-mode accuracy --no-think --max-new-tokens 128`

### Results

| Condition | Test Accuracy | Fitness Curve (best per gen) |
|-----------|--------------|-----|
| No evolution (baseline) | **72.0%** | N/A |
| Euclidean evolved | **68.0%** | 62.5% -> 37.5% -> 62.5% |
| Hyperbolic evolved | **68.0%** | 62.5% -> 37.5% -> 50.0% |

### What We Learned
1. **Evolution still hurts (-4%)** even with correct accuracy-based fitness
2. **Geometry doesn't matter** — Euclidean and Hyperbolic produce identical test accuracy (68% = 68%)
3. **Fitness curves collapse in gen 2** — elitist selection converges to suboptimal region
4. **Local mutations can't exploit global landscape** — sensitivity showed 32% range across RANDOM latents, but local perturbations (noise_scale=0.1) around the seed degrade performance
5. **The gap is LOCAL vs GLOBAL search** — good latents exist (sensitivity proved it) but they're FAR from each other in latent space
6. **No-think mode reduces baseline from 88% to 72%** — chain-of-thought contributes ~16% accuracy on easy_nested tasks, but 72% baseline gives evolution more room to improve

### Implication: Need Global Search
The sensitivity analysis (32% range, p=0.006) proves good latents exist globally. V15b proves local evolution can't find them. Next steps:
- Random search baseline (sample N random latents, pick best)
- CMA-ES (learns landscape covariance, can make large adaptive jumps)
- QD with large noise (diversity pressure + global exploration)
- Increase noise_scale from 0.1 to 1.0+ (current mutations too conservative)

### Artifacts
- `experiments/v15b_accuracy_diagnostic.json`
- `experiments/run_v15_geometry_isolation.py`

---

## Nested Expression Sensitivity (2026-03-03) — INTER-LATENT VARIANCE CONFIRMED

**Purpose:** Test whether random soft prompt latents produce different accuracy on nested expression tasks (no step-by-step scaffolding).
**Config:** 10 random Euclidean latents, 25 easy_nested tasks (2-digit arithmetic expressions), greedy decoding, Qwen3-4B Q4
**Script:** `experiments/run_latent_sensitivity.py --task-type nested --difficulty easy_nested --n-latents 10`

### Results

| Condition | Accuracy |
|-----------|----------|
| Zero-shot baseline | **92%** |
| Latent 1 (best) | **96%** |
| Latent 4, 7, 9 | 92% |
| Latent 5 | 88% |
| Latent 3, 6, 8 | 84% |
| Latent 2 | 80% |
| Latent 10 (worst) | **64%** |
| Mean conditioned | 85.6% |

**Cochran's Q = 23.2, p = 0.0058** (significant — latents differ from each other)
**0/10 latents individually significant vs baseline** (binomial test, all p > 0.05)
**Mean conditioned BELOW baseline** (85.6% < 92%)

### What We Learned
1. **Latents produce different accuracy profiles** — Cochran's Q proves they're not all the same (p < 0.01)
2. **Variance is predominantly negative** — mean conditioned 85.6% < 92% baseline. Conditioning mostly hurts on easy tasks
3. **Best latent (96%) is NOT individually significant** — 24/25 vs 23/25 baseline = +1 correct answer (p=0.39)
4. **Catastrophic latents exist** — L10 drops to 64% (9 additional failures)
5. **Direction matters for harm** — some directions strongly disrupt reasoning (L10 at 64%)
6. **High-baseline tasks don't show positive signal** — need harder tasks where there's room for improvement

### Per-Task Sensitivity (most variable)
- nest_006: 60% (baseline wrong, conditioning fixes 6/10 times)
- nest_002: 20% (baseline right, conditioning breaks 8/10 times)
- nest_004: 20% (baseline wrong, conditioning fixes 2/10 times)
- nest_009, 014, 015: 70% (baseline right, conditioning breaks 3/10 times)

### Artifacts
- `experiments/sensitivity_nested_easy_results.json`
- `experiments/calibration_nested_results.json` (calibration run)
- `experiments/run_latent_sensitivity.py` (unified script)

---

## Calibration: Nested Expression Difficulty (2026-03-03) — COMPLETE

**Purpose:** Find task difficulty level producing 50-70% baseline accuracy for sensitivity testing.
**Config:** 40 nested tasks across 4 difficulty levels, zero-shot baseline, Qwen3-4B Q4

### Results

| Difficulty | Accuracy | Operations | Status |
|-----------|----------|-----------|--------|
| easy_nested | 75% (8 tasks) / 92% (25 tasks) | 2 ops, 2-digit | Sweet spot |
| medium_nested | 42% | 3-4 ops, nesting | Below target |
| hard_nested | 8.3% | 5-6 ops, branches | Too hard |
| brutal_nested | 12.5% | 7-8 ops, deep nest | Too hard |

### Artifacts
- `experiments/calibration_nested_results.json`

---

## Latent Sensitivity Diagnostic (2026-03-03) — VALIDATED

**Purpose:** Test whether different random soft prompt latents produce meaningfully different accuracy on step-by-step arithmetic tasks.
**Config:** 5 random Euclidean latents, 13 tasks (3 easy/5 medium/5 hard), greedy decoding, Qwen3-4B Q4
**Script:** `experiments/run_latent_sensitivity.py --diagnostic`

### Results

| Condition | Accuracy |
|-----------|----------|
| Zero-shot baseline | **100%** |
| Latent 1 | 100% |
| Latent 2 | 92.3% |
| Latent 3 | 92.3% |
| Latent 4 | 92.3% |
| Latent 5 | 92.3% |
| Mean conditioned | 93.8% |

### What We Learned
1. **Landscape has structure** — different latents produce different accuracy (not flat)
2. **Sensitivity is task-specific** — only 1 of 13 tasks (sens_008: 6-step chain, answer=9338) shows any latent sensitivity
3. **Soft prompts CAN disrupt reasoning** — 4 of 5 random latents fail on the boundary task; failure mode is chain management disruption (model outputs intermediate values, not final answer)
4. **Direction matters** — Latent 1 preserves accuracy while Latents 2-5 fail, despite all being random
5. **Conditioning hurts on average** — -6.2% vs baseline, consistent with V15 finding
6. **Baseline too high for positive signal** — 100% baseline leaves no room for improvement, only degradation
7. **Next required**: tasks at 50-70% baseline to test if latents can IMPROVE accuracy

### Artifacts
- `experiments/latent_sensitivity_results.json`
- `experiments/run_latent_sensitivity.py`

---

## Conditioning Comparison (2026-03-02) — VALIDATED

**Purpose:** Head-to-head comparison of 3 conditioning methods across 4 model sizes.
**Conditions:** Pure model (no conditioning), Soft prompt (orthogonal W projection, 8 tokens), RNG seed (latent stats -> torch.manual_seed)
**Models:** Qwen3-0.6B (FP16), Qwen3-4B (Q4), Qwen3-8B (Q4), Qwen3-14B (Q4)
**Questions:** 20 diverse questions (arithmetic, multi-step, word problems, logic, knowledge)
**Evaluation:** LLM-as-judge on decoded text outputs (correctness, focus, reasoning, completeness, conciseness)

### Results

| Model | Pure Model | Soft Prompt | RNG Seed | Best Method |
|-------|-----------|-------------|----------|-------------|
| 0.6B  | 2.3/5     | 2.6/5       | **3.8/5**| RNG Seed    |
| 4B    | 2.8/5     | **4.0/5**   | 3.5/5    | Soft Prompt |
| 8B    | 3.8/5     | 3.9/5       | **4.6/5**| RNG Seed    |
| 14B   | 3.4/5     | **3.7/5**   | 3.3/5    | Soft Prompt |

### What We Learned
1. **Both conditioning methods eliminate phantom question hallucination** — pure model invents extra questions ~45% of the time at all sizes
2. **Soft prompt vs RNG seed is non-monotonic** — no single method dominates across all scales
3. **Minimum model capacity for soft prompt benefit** — 0.6B is too weak (gets confused), 4B+ benefits
4. **RNG seed's conciseness advantage** — produces cleaner, more structured output (LaTeX, boxed answers)
5. **Soft prompt's reliability advantage** — never produces empty/near-empty responses unlike RNG seed (~10-15% failure rate)

### Artifacts
- `experiments/conditioning_comparison_qwen3_0.6b.json`
- `experiments/conditioning_comparison_qwen3_4b.json`
- `experiments/conditioning_comparison_qwen3_8b.json`
- `experiments/conditioning_comparison_qwen3_14b.json`
- `experiments/run_conditioning_comparison.py` (per-model runner)
- `experiments/run_multi_model_comparison.py` (multi-model orchestrator)

### Commits
- `9cc38b8` Add multi-model conditioning comparison
- `81fded7` Fix Unicode crash on Windows, add skip-existing and multi-model results
- `9b7f276` Add Qwen3-8B conditioning comparison results

---

## V15: Geometry Isolation Diagnostic (2026-03-02) — VALIDATED

**Purpose:** Test whether hyperbolic vs Euclidean mutation geometry matters when both use identical soft prompt conditioning (same W matrix, same projection pipeline).
**Commit:** `b2fe7ce`
**Config:** 3 conditions (no_evo, euclidean_softprompt, hyperbolic_softprompt), 1 seed, 20 test tasks

### Results (Hard Difficulty — 5-step chained arithmetic)
| Condition | Test Accuracy |
|-----------|--------------|
| No evolution (baseline) | **90%** |
| Euclidean evolved | 60% |
| Hyperbolic evolved | 60% |

### What We Learned
1. **Evolution HURTS with dense_score fitness** — Goodhart's Law. Evolution pushes latent away from the seed's "good mode"
2. **Geometry doesn't matter when fitness is broken** — both geometries degrade equally
3. **Baseline is already near-optimal** — the fixed random projection at seed latent already produces good soft prompts
4. **Fitness function is the bottleneck**, not geometry or conditioning channel

### Artifacts
- `experiments/v15_geometry_isolation_diagnostic.json`
- `experiments/v15_sample_outputs.json`
- `experiments/run_v15_geometry_isolation.py`

---

## V12: Mobius Mutations + Operator Ablation (2026-02-21) — DESIGNED, NOT RUN

**Purpose:** Test 6 mutation operators to isolate geometry effect from conditioning.
**Codex Grade:** A-/92
**Conditions:** no_evo, euc_constrained, hyp_origin_roundtrip, hyp_mobius, hyp_local_expmap, euc_unconstrained
**Status:** Superseded by V15 (which found the deeper issue: broken fitness)

---

## V11: Codex V10 Issue Fixes (2026-02-20) — IMPLEMENTED

**Purpose:** Fix all 10 issues identified by Codex in V10 review.
**Codex Grade:** A-

### Fixes Applied
1. Ball radius mismatch (hyp=1.34 vs euc=0.95) -> matched
2. RNG contamination -> torch.Generator + same seed all conditions
3. Train/test leakage (tiny task space) -> branching=15, unique enum
4. McNemar pseudo-replication -> per-seed only
5. No global best tracking -> track across all gens
6. dense_score depth-biased -> absolute distance 1/(1+d)
7. Loose answer parsing -> last number only
8. Missing no-evolution baseline -> added + norm-matched
9. Noise not dim-normalized -> scale/sqrt(dim)
10. Fragile statistics -> Bonferroni + pre-registered primary

---

## V10: Dense Reward + Constrained Euclidean (2026-02-19) — INVALIDATED

**Purpose:** Fair comparison with matched conditioning and dense reward signal.
**Status:** Codex found 10 issues (4 critical, 4 high, 2 medium). Results methodologically inflated.

### Diagnostic Results (Before Codex Review)
| Condition | Accuracy |
|-----------|----------|
| Hyperbolic | 90% |
| Euclidean constrained | 67.5% |
| Euclidean unconstrained | 75% |

### Why Invalidated
- Loose verifier (any number in output counted as correct)
- Magnitude-normalized scoring biased toward certain geometries
- RNG seed contamination across conditions
- Ball radius mismatch made comparison unfair

---

## V9: Rigorous 5-Seed Run (2026-02-17) — NO SIGNAL

**Purpose:** Statistically rigorous test with 5 seeds and McNemar.
**Result:** p=0.18 (not significant), evolution fitness=0.000 everywhere.
**What We Learned:** Evolution with the latent scorer was completely blind. Led to V10+ shift to verifiable tasks.

---

## V7-V8: Curvature Sweep + Model Capacity (2026-02-03) — PROMISING BUT FRAGILE

**Purpose:** Identify optimal curvature, test model size requirements.

### Key Findings
- **c=0.5 optimal for depth 2-3** (68% vs 40% Euclidean at depth 3 with Qwen3-4B)
- **Model capacity matters:** 1.7B -> Euclidean wins; 4B -> Hyperbolic wins
- **Codex verdict:** "Plausible early signal, not a legitimate breakthrough yet"
- Selection bias from curvature sweep on validation set

### Artifacts
- See `experiments/HYPERBOLIC_EVAL_FINDINGS.md` for full V7-V8 details

---

## V5-V6: First Hyperbolic Signal (2026-02-01) — MIXED

**Purpose:** Test hyperbolic vs Euclidean with verifiable arithmetic tasks on Qwen3-4B.

### V5 Results
- Prompt 1: Hyperbolic 31.2% vs Euclidean 6.2% (+25%)
- Prompt 2: Euclidean 31.2% vs Hyperbolic 6.2% (-25%)
- Average: TIE (high variance)

### V6 Results (5 seeds)
- Hyperbolic wins 3/5 seeds, +8% average margin
- Depth 3 strongest signal (+33.3%)
- No individual seed reached p < 0.05

### What We Learned
- High prompt-dependent variance in single-seed runs
- Multiple seeds essential for any claims
- Depth 3 is the sweet spot for this task structure

---

## V1-V4: Foundation Experiments (2026-01 to 2026-02) — EARLY EXPLORATION

**V1-V3:** Basic evolution + latent scorer. Discovered scorer has ~0 correlation (-0.031).
**V4:** Tree traversal tasks on Qwen3-1.7B. Model too small (tail=0.000 for both geometries).

### What We Learned
- Latent scorer is useless for quality assessment (trains on style, not correctness)
- Verifiable tasks (arithmetic, logic) are the path forward
- Model capacity >= 4B needed for hyperbolic benefits

---

## Codex Full Repo Review (2026-02-20) — GRADE: C+

**Key Verdict:** "Evidence that conditioning bandwidth matters, not that hyperbolic geometry improves reasoning"

### Critical Findings
1. V10 90% was methodologically inflated
2. RNG-seed conditioning is the bottleneck, not evolution itself
3. Soft prompt works because it's a high-bandwidth continuous channel
4. Geometry effect is entangled with conditioning method

### Recommendations (All Addressed)
1. Unify experiment harness -> Done (Task 1: `experiments/harness.py`)
2. Test geometry under same conditioning channel -> Done (V15)
3. Abandon RNG-seed conditioning -> Done (soft prompt is default)
4. Multi-model validation -> Done (conditioning comparison)
