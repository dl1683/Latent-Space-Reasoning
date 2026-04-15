# Temperature/Prompt-Perturbation Head-to-Head — Experimental Design

## Status: DESIGN PHASE (Pre-Codex Approval)

## Purpose
Determine whether our continuous embedding prefix perturbation provides meaningfully different trajectory diversity compared to standard temperature sampling and prompt rephrasing. This is the FASTEST existential novelty check — if temperature sampling matches our oracle coverage, our contribution narrows to mechanism insight and potential efficiency gains.

## Why This Is Priority #1
Codex Round 2: "Head-to-head vs temperature / prompt perturbation on existing stack. This is the fastest existential novelty check."

The inference-time scaling literature (arXiv:2502.11027) already shows that diversified prompt perturbations improve best-of-N by 10.8%. If standard temperature sampling achieves similar oracle coverage to our prefix perturbation on the same tasks, we cannot claim novelty for "perturbation improves best-of-N." Our novelty would then be limited to:
- Sub-token continuous perturbation accesses different trajectory classes
- Deterministic greedy diversity (reproducible, no stochastic decoding)
- Potential efficiency via early routing (if Phase B works)

## Experimental Design

### Model + Tasks
- **Model**: Qwen3-4B Q4 (same as all existing experiments)
- **Tasks**: Same 25 arithmetic tasks
- **Max tokens**: 1024 (same as existing)
- **Budget**: N=10 candidates per condition per task

### Conditions (5 total):

**Condition A: Prefix Perturbation (our method, existing data)**
- Greedy decoding (do_sample=False, temp not applicable)
- 2-token random soft prefix, RMS matched to embedding RMS
- Seeds 0-9
- Data: ALREADY COLLECTED (reuse existing N=10 results from Qwen3-4B Q4)

**Condition B: Greedy Baseline (no perturbation)**
- do_sample=False, no prefix
- Single deterministic output per task
- MUST be in the same run table for fair comparison (Codex mandate)

**Condition C: Zero Soft-Prefix Control**
- do_sample=False, zero-valued 2-token prefix via inputs_embeds
- Tests whether position shift alone (without embedding noise) accounts for any effect

**Condition D: Temperature T=0.3**
- do_sample=True, temperature=0.3, top_k=0, top_p=1.0
- No prefix. Seeds 0-9.
- **3 independent N=10 panels** (seed sets 0-9, 10-19, 20-29) to assess variance (Codex mandate)

**Condition E: Temperature T=0.6**
- do_sample=True, temperature=0.6, top_k=0, top_p=1.0
- No prefix. 3 × N=10 panels.

**Condition F: Temperature T=1.0**
- do_sample=True, temperature=1.0, top_k=0, top_p=1.0
- No prefix. 3 × N=10 panels.

**Condition G: Nucleus Sampling T=0.6 + top_p=0.9** (Codex-added)
- do_sample=True, temperature=0.6, top_p=0.9
- No prefix. 3 × N=10 panels.
- Codex: "In practice, many sampling baselines use temperature plus nucleus. If pure temperature loses but nucleus matches prefix, reviewers will call the baseline incomplete."

**Condition H: Template-Based Prompt Rephrasing**
- Greedy decoding (do_sample=False), no prefix
- 10 pre-registered, manually audited templates per task
- Codex: "Do not reorder operands for noncommutative operations. Pre-register templates and inspect all 250 prompts."
- Template transforms: operation word swaps (multiply→times→×), framing changes (What is→Calculate→Compute), format changes (7×8→7 multiplied by 8)

### Metrics (per Pre-Implementation Measurement Contract)
For each condition:
1. **Individual accuracy**: mean last-integer accuracy across N=10 × 25 tasks
2. **Oracle accuracy**: best-of-N last-integer accuracy per task, averaged over 25 tasks
3. **Answer-anywhere accuracy**: mean + oracle
4. **Error correlation**: pairwise 10×10 correlation matrix (does seed i fail on same tasks as seed j?)
5. **Trajectory diversity**: cluster outputs by first 16/32 tokens, count distinct trajectory classes
6. **Output length distribution**: mean, std, min, max per task
7. **Truncation rate**: fraction of outputs hitting max_new_tokens
8. **Bootstrap CIs**: over tasks, 1000 iterations

### Comparison Outcomes (Pre-Registered — Codex-Revised)
With 25 tasks, 1 task = 4pp oracle. Thresholds defined in task counts (more interpretable):
- **"Matches"**: Best temperature/nucleus oracle differs from prefix oracle by ≤ 1 task AND mean accuracy CIs overlap materially. Report across all 3 panels — if only 1/3 panels matches, that's variance, not matching.
- **"Prefix beats"**: Prefix oracle wins by ≥ 3 tasks, OR ≥ 2 tasks plus clearly better mean accuracy/truncation profile. Must hold across all 3 panels.
- **"Mechanistically interesting"**: Distinct early-trajectory clusters (first-32-token overlap < 0.5) AND Jaccard similarity of correct-task sets < 0.7 AND different trajectory class distribution. (Codex: correct-set Jaccard alone is too coarse.)

### Variance Assessment (Codex Mandate)
- Each stochastic condition (D-G) runs 3 independent N=10 panels
- Report mean ± SD of oracle and mean accuracy across panels
- If panel SD > 2 tasks for oracle, the comparison is underpowered and needs more tasks
- Also compare **effective usable candidates**: exclude truncated/incoherent outputs from oracle denominator

### Expected Results by Hypothesis

**If trajectory diversity is the only mechanism:**
- Temperature at T=0.6-1.0 should match or exceed our oracle coverage (since temperature creates more diversity per sample)
- Our mean accuracy should be higher (greedy prefix is more controlled than temperature noise)
- Error correlation should be similar

**If our prefix accesses different trajectory classes than temperature:**
- Oracle coverage may be similar, BUT the set of correctly-solved tasks differs
- Jaccard similarity between "tasks solved by prefix oracle" and "tasks solved by temp oracle" < 0.7
- The two methods are complementary, not competing

**If our mechanism is purely attention-sink disruption:**
- Temperature sampling should match oracle (since temperature also disrupts greedy patterns)
- But prompt rephrasing (Condition E) should NOT match (since it doesn't add OOD embeddings)
- If prompt rephrasing matches, our mechanism is not about attention sinks at all

## Implementation Notes

### Condition B-D (Temperature)
```python
output = model.generate(
    input_ids=input_ids,
    max_new_tokens=1024,
    do_sample=True,
    temperature=T,
    top_k=0,        # no top-k filtering
    top_p=1.0,      # no nucleus filtering
    # Set different seeds for each of N=10 samples
)
```

### Condition E (Prompt Rephrasing)
Two approaches:
1. **Template-based**: Swap operation words ("multiply" → "times" → "×"), reorder operands ("7 × 8" → "8 × 7"), change framing ("What is 7 × 8?" → "Calculate: 7 times 8" → "7 multiplied by 8 equals?")
2. **LLM-based**: Use a separate model to generate 10 paraphrases per task
Prefer template-based for arithmetic (more controlled, fewer confounds).

### Seed Management
For temperature conditions, set `torch.manual_seed(seed)` before each generate call. This ensures reproducibility while creating the N=10 diverse samples.

## Timeline
This can run on existing infrastructure with minimal new code:
- Condition A: existing data
- Conditions B-D: one new generation config parameter (temperature), 3 runs × 25 tasks × 10 seeds = 750 generations
- Condition E: template-based rephrasing (can be hardcoded for arithmetic), 25 × 10 = 250 generations

Estimated: ~2-4 hours GPU time total.

## Dependencies
- Same Qwen3-4B Q4 model as existing experiments
- experiments/harness.py for grading
- Must use same extract_answer grading as existing data
