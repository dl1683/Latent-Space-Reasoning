> **Note (2026-08-27):** this internal dialogue predates the correction of the
> nested-arithmetic perturbation claims. Figures such as "32%→72%" and
> "perturbation beats scaling/temperature" quoted below are withdrawn — see
> [docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md](../docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md).

# Dark Knowledge Census — Build Specification

**The make-or-break experiment. Build it so a negative result is clean, a yellow result is useful, and a positive result is undeniable.**

## Scientific Target

Show that frozen models contain recoverable computations whose accessibility has measurable geometry, low-dimensional structure, and answer-blind navigability.

## Six Arms

| Arm | What It Tests |
|---|---|
| 1. Greedy baseline | What the model says by default |
| 2. Temperature sampling (T=0.2,0.5,0.8,1.0,1.3) | What token-level stochasticity can reach |
| 3. Embedding perturbation (K=128 random) | What continuous-space perturbation can reach |
| 4. Perturbation + answer-blind selector | Whether landscape is navigable without labels |
| 5. Paraphrase / token-prefix control | Whether discrete-interface variation reaches the same things |
| 6. Null / zero / random-position perturbation | Whether the effect is position/template artifact |

## Tasks

1,000 verifiable tasks with EXACT answers. Binary correct/incorrect. No rubric judging.

| Domain | Count | Examples |
|---|---|---|
| Arithmetic / symbolic calculation | 400 | 1-4 op, carries, negatives, fractions, order of operations |
| Symbolic logic | 300 | Syllogisms, constraint satisfaction, boolean evaluation |
| Algorithmic / program-trace | 200 | String manipulation, sorting, counting, pattern matching |
| GSM8K-style word problems | 100 | For comparability with literature |

Pre-split into train (60%) / validation (20%) / test (20%) BY TASK before any run.

## Models × Precisions

| Model | fp16 | 8-bit | 4-bit |
|---|---|---|---|
| Qwen3-4B | yes | yes | yes |
| Qwen3-8B | yes | yes | if fits |
| Non-Qwen (Llama-3.2-3B or Phi-3-mini) | yes | yes | yes |

## What to Log Per Generation

- Task ID, model, precision, arm, sample index
- Full generation trace (thinking + answer)
- Extracted answer, correctness (binary)
- Perturbation vector (for perturbation arms) or temperature value
- Perturbation norm, injection position, prefix length
- Baseline logprob of the correct answer sequence
- Token-level entropy at key decision points
- Rank of correct answer token at first divergence point
- Generation length, EOS status, extraction success/failure
- For mechanism subset (100 tasks): first divergent token position, hidden-state trajectory checkpoints

## Mechanism Subset (100 Tasks, Deeper Probes)

On a curated 100-task subset across all domains:
1. **First divergent token** — where does perturbation change the trace?
2. **Correct-answer rank at branch points** — margin rescue or deep routing?
3. **Prefix transplant** — take correct perturbation prefix, continue under baseline
4. **Constrained final-answer decoder** — force answer grammar
5. **Trace parse** — does scratchpad contain correct intermediate values?

This distinguishes routing failure (70-85% expected) from expression failure (15-30%).

## Selector Design

**Train AFTER census data exists. Evaluate on frozen held-out.**

### Answer-Blind Features
1. Generation completion: EOS, length, malformed answer rate
2. Answer stability: same answer across nearby perturbations
3. Trace coherence: contradictions, restarts, impossible intermediate claims
4. Arithmetic/logic invariants: intermediate quantities conserve constraints
5. Token confidence shape: entropy drops after decisive steps
6. Margin dynamics: branch tokens confident after evidence
7. Hidden trajectory: distance from failed baseline trajectory
8. Locality: nearby perturbations produce similar traces
9. Self-consistency under paraphrase
10. Extraction quality: valid parse, no hedging, no multiple answers

### Training Protocol
- Split by TASK (not by perturbation). No task leaks across train/test.
- Start with logistic regression / gradient-boosted trees. If boring models can't beat random, neural selectors are learning leakage.
- Freeze everything before held-out evaluation.
- Report top-1 and top-5 lift vs random.

## Decision Thresholds

### PUSH (all must hold)
- Perturbation finds ≥15% correct answers temperature doesn't
- Perturbation oracle beats best temperature oracle by ≥5 pts (fp16 AND quantized)
- Perturbation-exclusive answers have baseline probability ≤1e-8
- fp16 unlock rate ≥8% and ≥40% of 4-bit rate
- Cross-precision landscape correlation r≥0.35
- Locality lift ≥2x
- Answer-blind selector top-5 beats random top-5 by ≥2x (held-out)
- Low-rank (≤32) predictor AUC ≥0.65

### PIVOT (any triggers)
- Temperature recovers ≥80% of perturbation successes
- Perturbation advantage ≤2 pts over temperature
- Perturbation-success baseline probability ≥1e-5
- fp16 unlock rate <2% or <15% of 4-bit rate
- Cross-precision correlation r<0.10
- Locality lift <1.25x
- Selector lift <1.25x
- Low-rank AUC <0.56

### YELLOW (interesting but narrow)
Perturbation beats temperature, but only in 4-bit and without transfer. → "Quantization accessibility artifact." Worth one focused study, not months.

## Compute Budget

~50-80 GPU-hours total on RTX 5090.

| Phase | GPU-Hours | Calendar |
|---|---|---|
| Task generation | 0 (CPU only) | 2-3 days |
| Census infrastructure | 0 (coding) | 3-4 days |
| Census run (all models/precisions/arms) | 50-80 | 1-2 weeks |
| Analysis | 0-5 (GPU for hidden states) | 1 week |
| Selector training + eval | 2-5 | 2-3 days |
| **Total** | **~60-90** | **~4-5 weeks** |

## Reporting

Report BOTH sample-normalized oracle AND compute-normalized oracle. Temperature, perturbation, and selector scoring have different overhead. The claim is strongest when both views agree.

## Plan B (If Census Kills Strong Thesis)

1. **Quantization accessibility atlas** — when does quantization destroy vs reveal computations?
2. **Interface adequacy framework** — how much competence is reachable through ordinary decoding?
3. **Routing fragility meter** — predict when sampling helps vs when failures are real capability gaps
4. **The data factory is the asset.** It can pivot to any of these.

## Our Blind Spots (Self-Audit)

1. Don't overvalue positive unlocks. A correct answer feels magical but the unit of science is the full distribution.
2. Don't underweight task generation. Bad tasks make everything fake.
3. Don't confuse navigation with selection. Sampling 128 and picking the lucky one is NOT navigation.
4. Don't chase universality too early. First prove one clean object exists.
5. No mythology before the held-out selector works.
