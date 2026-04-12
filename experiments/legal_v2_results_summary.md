# Legal Reasoning v2 — Codex Blind Review Results

## Experiment Design
- **Model**: Qwen3-4B (4-bit NF4 quantization)
- **Tasks**: 12 legal reasoning scenarios across 5 categories
- **Conditions**: Greedy baseline (A), Random perturbation ×5 seeds (B1-B5), Evolution ×5 seeds (C1-C5)
- **Evaluation**: Codex CLI blind reviews on 5 dimensions (1-10 scale each)
- **Scoring dimensions**: Legal Accuracy, Analytical Depth, Practical Utility, Structural Quality, Completeness

## Results Table (Codex Group Averages)

| Task | Category | Baseline (A) | Perturbation (B avg) | Evolution (C avg) | Winner | Evo Status |
|------|----------|:---:|:---:|:---:|--------|:---:|
| 01 FTC Unfairness | Framework | 5.2 | 5.8 | **6.2** | Evolution | Working |
| 02 GDPR Controller | Framework | **3.0** | 1.8 | 1.6 | Baseline | Broken (-inf) |
| 03 Disparate Impact | Framework | **6.0** | 4.6 | 5.4 | Baseline | Working |
| 04 SaaS Contract | Transactional | 4.0 | **4.3** | 4.2 | Perturbation | Broken (-inf) |
| 07 IP Risk Portfolio | IP | 3.6 | **5.0** | 3.2 | Perturbation | Broken (-inf) |
| 08 Negotiation | Strategic | 2.0 | **4.5** | 2.4 | Perturbation | Broken (-inf) |
| 09 Regulatory Response | Regulatory | **5.0** | 4.4 | 3.8 | Baseline | Working |
| 05 Startup Acquisition | Issue Spotting | **5.2** | 4.2 | 4.4 | Baseline | Broken (-inf) |
| 10 Contractor Misclass | Scenario | 2.2 | **4.2** | 1.4 | Perturbation | Broken (-inf) |
| 06 Data Breach | Risk | **4.6** | 3.0 | 2.0 | Baseline | Broken (-inf) |
| 11 Corporate Veil | Scenario | **5.4** | 5.0 | 2.4 | Baseline | Broken (-inf) |
| 12 Whistleblower | Scenario | 3.2 | **5.0** | 4.6 | Perturbation | Broken (-inf) |

## Oracle Analysis (Best-of-5 per condition)

| Task | Baseline | Best Perturbation | Best Evolution | Overall Best | Lift vs Baseline |
|------|:---:|:---:|:---:|:---:|:---:|
| 01 | 5.2 | 7.2 | **7.2** | Tied B1/C4 | +2.0 |
| 02 | **3.0** | 1.8 | 1.6 | A (baseline) | 0.0 |
| 03 | 6.0 | **6.8** | 6.2 | B4 | +0.8 |
| 04 | 4.0 | **5.6** | 4.2 | Tied B3/B4 | +1.6 |
| 07 | 3.6 | **6.4** | 3.2 | B1 | +2.8 |
| 08 | 2.0 | **5.4** | 2.4 | B2 | +3.4 |
| 09 | 5.0 | **5.6** | 4.8 | B5 | +0.6 |
| 05 | 5.2 | **5.8** | 4.4 | B3 | +0.6 |
| 10 | 2.2 | **5.6** | 1.4 | B3 | +3.4 |
| 06 | 4.6 | **5.2** | 2.0 | B1 | +0.6 |
| 11 | 5.4 | **6.6** | 2.4 | Tied B4/B5 | +1.2 |
| 12 | 3.2 | **5.6** | 4.6 | B3 | +2.4 |

**In 11/12 tasks, the best perturbation/evolution output beats baseline (lifts +0.6 to +3.4). Oracle win rate: 92%.**

## Key Findings

### 1. Latent-space perturbation proves the model has untapped knowledge
- In 11/12 tasks (92%), at least one perturbation seed produces better legal analysis than greedy decoding
- Oracle lifts range from +0.6 to +3.4 on a 10-point scale
- This is "free" improvement — same model, same hardware, different latent trajectory

### 2. Evolution adds consistency (when scorer works)
- Task 01: 4/5 evolution seeds beat baseline vs 3/5 perturbation seeds
- Evolution avg (6.2) > Perturbation avg (5.8) > Baseline (5.2) on task 01
- Tasks with broken scorer (-inf) produce degenerate identical outputs

### 3. Task difficulty mediates the effect
- Tasks where model is "stuck" on bad greedy path (07, 08): 5/5 perturbation seeds beat baseline
- Tasks with decent baseline (03): perturbation is noisier, 1/5 seeds beat baseline
- Tasks where model is overwhelmed (02): nothing helps — model lacks domain knowledge

### 4. Scorer reliability is the bottleneck
- Evolution works on 3/12 tasks (01, 03, 09), broken on 9/12
- Root cause: non-deterministic random projection layer (dimension mismatch 2560→1024)
- **Fix applied**: Deterministic projection with fixed seed (42_000)
- **Clean re-run needed** with fixed code for all 12 tasks

## Technical Notes
- Evolution scorer uses trained MLP (LatentScorer) with 2560→1024 random projection
- Scorer broken due to non-deterministic projection initialization (depended on global RNG)
- Perturbation thinking tokens stripped for fair comparison (model generates <think> blocks with noise)
- Task 02 perturbation: seeds 43-46 consumed entire 2048-token budget in thinking loops
- Encoding artifacts (mojibake) present in all outputs due to Qwen3 tokenization edge cases

### 5. Task 11 (Corporate Veil): B3 flags "real business" defense
- B3 is the only output that clearly identifies "SubCorp appears to be a real operating business, not just a sham shell" — a critical Delaware veil-piercing defense
- Most outputs overstate plaintiff's probability of success
- B4 and B5 have best overall coverage but still miss key doctrinal nuance
- Codex anchored scoring to actual Delaware Chancery opinions

### 6. Task 06 (Data Breach): Codex notes all 11 outputs are "not briefing-safe"
- All outputs fabricate statutes, agencies, or deadlines to varying degrees
- B1 best organized but still invents "HIPAA Form 100" and California rules
- Codex verified against actual current law (CA 30-day rule effective Jan 2026, TX 60-day, GDPR Art 33-34)
- Task too complex for 4B model — but oracle perturbation still beats baseline

## Status
- **ALL 12/12 TASKS REVIEWED** — complete experiment
- **Evolution broken on 9/12 tasks** due to non-deterministic projection (C1-C5 identical)
- **Clean re-run** planned after current batches complete
- Last updated: 2026-04-11
