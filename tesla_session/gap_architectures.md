# Gap Architectures — Systematically Avoided Paradigms

## Status: DESIGN PHASE — Codex Review Required

## Origin
Self-audit after 27 architectures: what paradigms have been systematically avoided?

1. Multi-model ensembles (all 27 use a single model)
2. Auxiliary trained models (project constraint limits this but doesn't eliminate it)
3. Compression-based reasoning (none compress model output)
4. Persona/role-based diversity (none use structured role prompting)
5. Cross-problem memory (none share information across problems)

---

## Architecture 28: Multi-Model Natural Ensemble (MMNE)

### The Systematic Avoidance
All 27 architectures use ONE model with different intervention operators. But the most natural source of decorrelation is DIFFERENT MODELS. Two models trained on different data with different architectures will have fundamentally different failure modes.

### Why This Was Avoided
The project constraint: "Models: 1B-8B parameters, quantized" on a 24GB GPU. Running two models simultaneously may exceed VRAM.

### Why It Shouldn't Be Avoided
We don't need SIMULTANEOUS models. We can:
1. Load Model A, generate N/2 candidates, unload
2. Load Model B, generate N/2 candidates, unload
3. Select from the combined pool

Model switching takes ~30 seconds. For N=10 total candidates (5 per model), the overhead is 30 seconds on top of ~60 seconds generation time. ~33% overhead for potentially massive decorrelation gain.

### System Design

**Component 1: Model Pool**
Available models that fit in 24GB individually:
- Qwen3-4B Q4 (our primary)
- Phi-3-mini-4k (3.8B, Q4)
- Gemma-2-2B (Q8 or full precision)
- StableLM-2-1.6B (full precision)
- Llama-3.2-3B (Q4)
- Qwen3-1.7B (Q4)
- DeepSeek-R1-Distill-Qwen-1.5B (Q4, already tested)

**Component 2: Model Characterizer**
- Run each model on the 25 calibration tasks (greedy, no perturbation)
- Measure: individual accuracy, error pattern, output style
- Compute cross-model Jaccard: which model pairs solve different tasks?
- Select the most complementary pair (lowest Jaccard)

**Component 3: Multi-Model Generator**
```python
class MultiModelEnsemble:
    def generate(self, question, models, N_per_model=5):
        all_candidates = []
        for model_config in models:
            model = load_model(model_config)  # ~30 sec
            for seed in range(N_per_model):
                prefix = random_soft_prefix(seed, model.embed_dim)
                output = model.generate(question, prefix=prefix)
                all_candidates.append({
                    'model': model_config.name,
                    'seed': seed,
                    'output': output,
                })
            unload_model(model)  # free VRAM
        return all_candidates
```

**Component 4: Cross-Model Selector**
- For arithmetic: majority vote across all candidates (model-agnostic)
- For legal: weight by model quality (better models' candidates count more)
- Or: nested selection — pick best within each model, then pick between models

### Why This Might Work
- **Maximum decorrelation**: Different training data + different architectures = fundamentally different error patterns. No single-model intervention can match this diversity.
- **Prior art**: Model ensembles are THE standard approach for reducing error correlation in ML. The ensemble literature says: decorrelation is proportional to model diversity.
- **Synergy with prefix perturbation**: Apply prefix perturbation within EACH model. Cross-model + within-model diversity = two independent sources of decorrelation.
- **CDE natural fit**: Each model is just another OPERATOR in the CDE framework. The measurement contract applies directly.

### Failure Modes
1. **Model switching overhead**: 30 seconds per switch × 2 switches = 60 seconds. If the total generation time is 60 seconds, this doubles runtime. Acceptable if decorrelation gain is sufficient.
2. **Weak models in the pool**: If Model B has 10% accuracy while Model A has 45%, Model B's candidates are mostly noise. The ensemble may be WORSE because Model B wastes half the budget.
3. **Different output formats**: Models may format answers differently, confusing the selector. "56" vs "The answer is fifty-six" vs "7×8=56".
4. **Different tokenizers**: Can't share prefix embeddings across models. Each model needs its own RMS-matched prefix.

### Adversarial Audit
- **Model ensembles are not novel**: This is the most standard ML technique. Zero novelty.
- **The novelty is in the COMBINATION**: Multi-model + prefix perturbation within each model + CDE allocation. This specific combination is not studied.
- **Practical concern**: Model switching on consumer GPU is not instantaneous. With quantized models, loading from disk takes 10-30 seconds. This must be measured.
- **The best ensemble is often the best single model + more N**: If Qwen3-4B with N=10 beats (Qwen3-4B N=5 + Phi-3 N=5), the overhead of model switching is wasted. CDE measurement answers this.

### Verdict
OBVIOUS CANDIDATE that should have been in Wave 1. The CDE framework naturally accommodates it — each model is an operator. Include in Phase 2 measurement. Test the most complementary 2-model pair against single-model N=10.

---

## Architecture 29: Compression-Refinement Reasoning (CRR)

### The Systematic Avoidance
All architectures generate verbose outputs (500-1024 tokens of chain-of-thought). None COMPRESS the reasoning to extract the essential insight.

### Core Insight
Chain-of-thought is verbose by design — it includes backtracking, self-correction, intermediate calculations. Most of this is noise. The ANSWER is usually derivable from a small fraction of the reasoning. What if we:
1. Generate verbose reasoning (1024 tokens)
2. Ask the model to COMPRESS it to just the key steps and answer (128 tokens)
3. Use the compressed version as the final output

The compression step acts as a FILTER: it forces the model to identify what actually matters in its own reasoning and discard the noise.

### System Design

**Component 1: Verbose Generator**
- Standard generation with max_tokens=1024
- Produces full chain-of-thought

**Component 2: Compressor**
```
Prompt: "Here is a detailed solution to a problem. Extract ONLY the final answer and the critical reasoning steps. Be concise.

Problem: [original question]
Full solution: [verbose output]

Compressed solution (max 3 sentences):"
```
- Apply different perturbation seed for compression than for generation
- The compressor may "fix" errors by seeing the full chain and extracting only the correct parts

**Component 3: Multi-Compression**
- Generate ONE verbose output
- Compress it N times with different perturbation seeds
- Each compression may extract DIFFERENT key insights
- Majority vote on the compressed answers
- This is cheaper than generating N full outputs: 1 × 1024 tokens + N × 128 tokens vs N × 1024 tokens

**Component 4: Verify Compression**
- Compare compressed answer with verbose answer
- If they disagree: the compression found an inconsistency in the reasoning
- Use the disagreement as a signal for re-generation

### Why This Might Work
- **Compression as error correction**: The verbose output may contain correct intermediate steps AND a wrong final answer (error propagation). The compressor, seeing the FULL chain, may extract the correct intermediate and compute the right final answer.
- **Information-theoretic argument**: Compression removes noise and preserves signal. If the signal (correct reasoning) is present in the verbose output, compression should find it.
- **Cheap diversity**: N compressions of 1 output costs ~N/8 of N full generations (128 vs 1024 tokens). Use the savings for more diverse initial generations.

### Failure Modes
1. **Compression loses information**: The compressor may discard crucial reasoning steps, producing a confident but wrong summary.
2. **The model can't reliably extract the right answer from its own verbose reasoning**: This requires meta-cognitive ability that 4B models may lack.
3. **If the verbose output is wrong throughout**: Compression of a wrong chain produces a compressed wrong answer. No improvement.

### Adversarial Audit
- **This is just post-hoc summarization**: Not novel. The only twist is multi-compression with different perturbation seeds.
- **For arithmetic**: The compressor would need to re-derive the final answer from intermediate steps. If it can do that, it could have just answered correctly in the first place.
- **For legal**: More promising. Legal verbose outputs contain multiple arguments. Compression forces prioritization — which may surface the strongest argument.
- **The real test**: Does compressed accuracy exceed verbose accuracy? If yes, compression acts as error correction. If no, it's just information loss.

### Verdict
LOW priority but cheap to test. Worth a quick experiment: generate verbose outputs for 25 tasks, compress each, compare accuracy. If compression improves accuracy even once, the mechanism is interesting. If never, compression is pure loss.

---

## Architecture 30: Persona-Diversified Reasoning (PDR)

### The Systematic Avoidance
All architectures create diversity through MECHANISM (perturbation, temperature, sampling). None create diversity through FRAMING (role, perspective, persona).

### Core Insight
The same model, given different persona instructions, produces measurably different reasoning patterns. "Think step-by-step as a mathematician" vs "Think step-by-step as a programmer" vs "Think step-by-step as a teacher" may activate different knowledge pathways and produce different error patterns.

### System Design

**Component 1: Persona Registry**
```python
personas = [
    "You are a careful mathematician. Show each step clearly.",
    "You are a Python programmer. Think about this computationally.",
    "You are a teacher explaining this to a student. Be precise.",
    "You are a fact-checker verifying a calculation. Double-check each step.",
    "You are solving this problem for the first time. Think through it carefully.",
    "Solve this step by step, showing your work.",
    "Give just the answer, no explanation.",
    "First estimate the answer, then calculate precisely.",
    "Break this into smaller problems and solve each one.",
    "Work backwards from what the answer should look like.",
]
```

**Component 2: Persona + Perturbation Grid**
- For each persona P and perturbation seed S:
  - Prompt = [persona instruction] + [question]
  - Prefix = random_soft_prefix(seed=S)
  - Generate with greedy decoding
- This creates a 2D grid of diversity: persona × perturbation
- Total candidates: |personas| × |seeds| (e.g., 10 × 1 = 10, or 5 × 2 = 10)

**Component 3: Persona Quality Characterization**
- Run each persona on the calibration set
- Measure which personas produce highest individual accuracy
- Measure cross-persona Jaccard (which persona pairs solve different tasks)
- The best CDE allocation may be: 3 prefix seeds with "mathematician" + 3 with "programmer" + 4 with "step-by-step"

### Why This Might Work
- **Personas activate different knowledge**: The "programmer" persona may trigger the model to think about arithmetic as code (7*8), while the "mathematician" persona triggers formal calculation. Different approaches → different error modes.
- **Linguistic diversity without perturbation**: Each persona changes the prompt tokens (cheap, no embedding manipulation). This is like prompt rephrasing but structured.
- **Combines with perturbation for 2D diversity**: Perturbation changes the embedding space; personas change the token space. Two independent sources of diversity.
- **Dirt cheap**: No special infrastructure. Just prepend a different system prompt.

### Failure Modes
1. **Small models may ignore persona instructions**: 4B quantized models may not be sophisticated enough to modulate their reasoning based on persona instructions. The persona may just add noise to the prompt.
2. **Personas are correlated**: All personas share the model's underlying knowledge. If the model simply doesn't know 7×8, no persona helps.
3. **The "programmer" persona may generate code instead of answers**: Format divergence confuses the selector.

### Adversarial Audit
- **This is prompt engineering for diversity**: Not novel. Self-consistency papers already explore diverse prompts.
- **However**: The systematic integration of persona × perturbation as a 2D diversity grid within CDE is not studied. The measurement of persona-persona Jaccard similarity is informative.
- **For legal reasoning**: HIGHLY relevant. "Defense attorney" vs "prosecutor" vs "judge" personas produce genuinely different legal analyses. This is where PDR shines.

### Verdict
CHEAP TO TEST, potentially high value for open-ended tasks. Include as a Tier 2 operator (O_persona) in CDE. The measurement contract will determine if personas add decorrelation beyond what prompt rephrasing already provides.

---

## Architecture 31: Cross-Problem Memory Ensemble (CPME)

### The Systematic Avoidance
All architectures solve each problem INDEPENDENTLY. None use information from solving one problem to help solve the next.

### Core Insight
In a test set of 25 problems, the model solves some correctly and some incorrectly. What if correct solutions provide CONTEXT for subsequent problems?

Example:
- Problem 1: "7 × 8 = ?" → Model correctly answers "56"
- Problem 2: "7 × 8 + 12 × 3 = ?" → Instead of cold-starting, prepend: "Earlier, you correctly computed 7 × 8 = 56."
- This provides grounded context: the model doesn't need to re-derive 7 × 8.

### System Design

**Component 1: Problem Ordering**
- Sort problems from EASIEST to HARDEST (by baseline accuracy)
- Solve easy problems first, building up a "memory bank" of verified results
- Use verified results as context for harder problems

**Component 2: Memory Bank**
```python
memory_bank = {}  # {sub_expression: verified_result}

for problem in sorted_problems(easiest_first):
    # Inject relevant memories
    context = ""
    for sub_expr, result in memory_bank.items():
        if sub_expr in problem.text:
            context += f"Known fact: {sub_expr} = {result}\n"
    
    # Generate with context
    output = generate(context + problem.text, prefix=random_prefix())
    answer = extract_answer(output)
    
    # If correct (verified), add to memory
    if verify(answer, problem.ground_truth):
        memory_bank[problem.expression] = answer
```

**Component 3: Memory Selection**
- Not all memories are relevant. Only inject memories that contain SUB-EXPRESSIONS of the current problem.
- For "7 × 8 + 12 × 3": inject "7 × 8 = 56" if available, but NOT "5 × 9 = 45" (irrelevant).
- For legal tasks: inject relevant legal principles from previously solved problems.

**Component 4: Memory Confidence Weighting**
- Memories from high-confidence solutions (majority vote agreement) get full injection
- Memories from uncertain solutions (narrow majority) get qualified injection: "Likely: 7 × 8 = 56"
- Wrong memories are TOXIC: if the memory bank contains "7 × 8 = 54" (wrong), all downstream problems are poisoned

### Why This Might Work
- **Divide and conquer over the test set**: Easy problems are "base cases" that support harder problems.
- **Sub-expression reuse**: Many arithmetic problems share sub-expressions. Pre-verified results eliminate redundant computation.
- **Knowledge accumulation**: Each solved problem adds to the model's effective context. The 25th problem benefits from 24 prior solutions.
- **Natural for legal reasoning**: Legal principles established in one analysis apply to others.

### Failure Modes
1. **Memory contamination**: Wrong memories propagate errors to all downstream problems. If problem 1 is solved wrong but passes verification (false positive), everything downstream is contaminated.
2. **Problem ordering dependency**: The order of problems affects results. Optimal ordering is a combinatorial problem.
3. **Context window saturation**: With 24 prior solutions × 50 tokens each = 1200 tokens of memory. This consumes 30% of the 4K context window, leaving less room for actual reasoning.
4. **Not applicable to independent problems**: If the 25 arithmetic tasks share no sub-expressions, the memory bank is useless.

### Adversarial Audit
- **This is in-context learning**: Well-studied. Few-shot examples from solved problems as context. Not novel.
- **The novelty is in VERIFIED memory injection**: Using formal verification (Architecture 24) to ensure memory correctness before injection. Wrong in-context examples are known to HURT performance. Verified-correct examples should help.
- **For our 25 arithmetic tasks**: Many share operations (multiplication, addition). Sub-expression reuse is plausible.
- **For legal tasks**: Establishing legal principles in early problems helps later ones — this mirrors how legal analysis actually works (precedent).

### Verdict
NICHE but potentially powerful for structured problem sets where problems share sub-components. Requires formal verification (Architecture 24) to prevent memory contamination. Include as a Tier 3 enhancement, tested after CDE Phase 1.

---

## Architecture 32: Auxiliary Micro-Router (AMR)

### The Systematic Avoidance
The project constraint says "no training of the TARGET model." But it doesn't prohibit training a TINY auxiliary model for routing decisions.

### Core Insight
The biggest gap in CDE is the selector/router. All selectors either require ground truth (exact match) or are unreliable (self-certainty, length). What if we train a tiny model (1-10M params) specifically to predict which operator will produce the best candidate for a given input?

### System Design

**Component 1: Router Model**
- Input: question embedding (from the target model's encoder or sentence-transformer)
- Output: probability distribution over operators {prefix, temperature, rephrase, ...}
- Architecture: 2-layer MLP, ~1M parameters
- Training data: Phase 1 traces (question → which operator produced the correct answer)

**Component 2: Router Training**
```python
class MicroRouter(nn.Module):
    def __init__(self, embed_dim=384, n_operators=8):
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_operators),
        )
    
    def forward(self, question_embedding):
        return F.softmax(self.net(question_embedding), dim=-1)

# Training data from Phase 1 traces
# Label = operator that produced a correct answer (if multiple, weight equally)
router = MicroRouter()
optimizer = Adam(router.parameters(), lr=1e-3)
for question_emb, best_operator_id in training_data:
    pred = router(question_emb)
    loss = F.cross_entropy(pred, best_operator_id)
    loss.backward()
    optimizer.step()
```

**Component 3: Router-Guided Allocation**
- For a NEW question: compute question embedding, run through router
- Router output: {prefix: 0.6, temp: 0.2, rephrase: 0.2}
- Allocate N=10 budget proportionally: 6 prefix + 2 temp + 2 rephrase
- This replaces the static allocation from CDE with a LEARNED, input-dependent allocation

**Component 4: Router Confidence Calibration**
- If router is uncertain (max prob < 0.3): use uniform allocation (safe default)
- If router is confident (max prob > 0.7): use its allocation (exploit learned pattern)
- This provides a graceful fallback

### Why This Might Work
- **Some problems benefit from specific operators**: Multiplication problems might benefit from prefix perturbation (shifts computation path) while word problems benefit from prompt rephrasing (changes framing). The router learns which operator suits which problem.
- **Tiny model, no VRAM impact**: 1M params = ~4MB. Negligible compared to the 4B target model.
- **Training data exists**: Phase 1 traces provide all needed training data. No additional generations required.
- **Fits CDE perfectly**: The router IS the "Compute Allocation" layer that Codex identified as missing.

### Failure Modes
1. **25 tasks is not enough training data**: With 25 tasks × 8 operators = 200 data points. A 1M-param model may not learn anything useful from 200 examples.
2. **Overfitting**: The router may memorize "all multiplication problems should use prefix" without understanding why. On a new task type, it may fail.
3. **Philosophical tension**: Training even a tiny auxiliary model moves away from the "no training" ethos. However, the target model stays frozen.
4. **Router accuracy matters less than you think**: If the router is only 60% accurate, and the fallback (uniform allocation) works 50% of the time, the improvement is marginal (60% vs 50%).

### Adversarial Audit
- **Routing is exactly what's needed**: Codex identified the missing layer as "measurement, allocation, calibration, and routing." AMR provides the routing component.
- **Training data concern is real**: 25 tasks is tiny. Solutions: (a) augment with synthetic tasks, (b) use leave-one-out cross-validation, (c) use the router only for high-confidence predictions.
- **Comparison with static allocation**: If the greedy allocator from CDE (which uses measured rho and A values) is good enough, the learned router adds complexity without benefit. Test: does input-dependent allocation beat input-independent allocation?
- **EBRL (Architecture 15) is a selector; AMR is a router**: Different roles. EBRL picks the best candidate from a set. AMR picks which operator should GENERATE the candidates. Both are auxiliary trained models, both fit CDE.

### Verdict
DEFERRED until Phase 1 data exists. Training data comes from Phase 1 traces. Can be built and tested in < 1 hour after Phase 1 completes. Include as a Phase 2 enhancement.

---

## Summary: Complete Architecture Catalog

### Architectures 1-7 (Wave 1: alternative_architectures.md)
| # | Name | Codex Status |
|---|---|---|
| 1 | ASM (Activation State Machine) | Reviewed ✓ |
| 2 | Checkpoint-and-Branch | Reviewed ✓ |
| 3 | Spectral Reasoning Amplification | Reviewed ✓ |
| 4 | Inverse Speculative Reasoning | Reviewed ✓ |
| 5 | Multi-Surface Ensemble → CDE | Reviewed ✓ → Evolved to CDE |
| 6 | Token-Free Continuous Reasoning | Reviewed ✓ |
| 7 | Adversarial Self-Debate | Reviewed ✓ |

### Architectures 8-17 (Wave 2: radical_architectures.md)
| # | Name | Codex Status |
|---|---|---|
| 8 | GGIO (Gradient-Guided Input Optimization) | Reviewed ✓ |
| 9 | DDC (Decompose-Dispatch-Compose) | Reviewed ✓ |
| 10 | RARS (Retrieval-Augmented Reasoning from Self) | Reviewed ✓ |
| 11 | CIR (Causal Intervention Reasoning) | Reviewed ✓ |
| 12 | IDR (Iterated Distillation Reasoning) | Reviewed ✓ |
| 13 | ALM (Attractor Landscape Mapping) | Reviewed ✓ |
| 14 | Neuro-Symbolic Hybrid | Reviewed ✓ |
| 15 | EBRL (Energy-Based Reranking) | Reviewed ✓ |
| 16 | Attention Surgery | Reviewed ✓ |
| 17 | TER (Temporal Ensemble Reasoning) | Reviewed ✓ |

### Architectures 18-27 (Wave 3: missing_architectures.md)
| # | Name | Codex Status |
|---|---|---|
| 18 | DO-BoN (Diversity-Optimized Best-of-N) | Needs review |
| 19 | ETR (Early-Trajectory Router) | Needs review |
| 20 | VFR (Verifier-First Regeneration) | Needs review |
| 21 | DRCG (Diversity-Regularized Candidate Gen) | Needs review |
| 22 | Gated Attention Probe | Already designed |
| 23 | DBTR (Diffusion-Based Text Refinement) | Needs review |
| 24 | FVI (Formal Verification Integration) | Needs review |
| 25 | IPF (Inverse Problem Formulation) | Needs review |
| 26 | CSLA (Consciousness Simulation Loop) | Needs review |
| 27 | CPH (Constraint Programming Hybrid) | Needs review |

### Architectures 28-32 (Wave 4: gap_architectures.md)
| # | Name | Codex Status |
|---|---|---|
| 28 | MMNE (Multi-Model Natural Ensemble) | Needs review |
| 29 | CRR (Compression-Refinement Reasoning) | Needs review |
| 30 | PDR (Persona-Diversified Reasoning) | Needs review |
| 31 | CPME (Cross-Problem Memory Ensemble) | Needs review |
| 32 | AMR (Auxiliary Micro-Router) | Needs review |

### CDE Framework Documents
| Document | Codex Status |
|---|---|
| controlled_decorrelation_ensemble.md | Needs review |
| cde_measurement_protocol.md | Needs review |

### Codex-Approved Top 5 (from Wave 2 review)
1. CDE (Architecture 5 redesigned)
2. ALM (#13)
3. DDC (#9)
4. Neuro-Symbolic (#14)
5. CIR (#11)

### Implementation-Ready Tier (No new infrastructure needed)
1. Temperature comparison (= CDE Phase 1)
2. DO-BoN (#18) — add fingerprinting to existing pipeline
3. DRCG (#21) — add duplicate detection to existing pipeline
4. PDR (#30) — just prompt engineering
5. MMNE (#28) — swap models between runs
