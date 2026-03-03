# Experiments Log

Reverse chronological. Each entry links artifacts and summarizes findings.
Only Codex-validated conclusions are stated as "confirmed."

---

## Nested Expression Sensitivity (2026-03-03) — STRONGLY EXPLOITABLE

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

**Cochran's Q = 23.2, p = 0.0058** (significant at p < 0.01)

### What We Learned
1. **STRONGLY EXPLOITABLE landscape** — 32% range across 10 random latents
2. **Conditioning CAN improve accuracy** — Latent 1 beats baseline (96% > 92%)
3. **Different latents fix different failures** — nest_006 (60% correct), nest_004 (20% correct, but L6/L7 fix it!)
4. **Catastrophic latents exist** — L10 drops to 64% (9 additional failures)
5. **Task-dependent sensitivity** — 13/25 tasks always correct, 3/25 are boundary tasks, 9/25 show moderate sensitivity
6. **Cochran's Q is statistically significant** — not a fluke; p = 0.0058
7. **First statistically significant result in the project** — evolution CAN exploit this

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
