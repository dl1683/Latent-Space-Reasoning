# Radical Reasoning Architectures — Wave 2

## Status: DESIGN PHASE — Codex Review Required

## Premise

Wave 1 (alternative_architectures.md) explored 7 architectures that still operate within the standard autoregressive generation framework. This document breaks that frame entirely. These architectures question whether autoregressive token-by-token generation is even the right paradigm for inference-time reasoning improvement.

**Same constraints**: No training. No fine-tuning. Frozen weights. RTX 5090 24GB. Models 1B-8B quantized.

---

## Architecture 8: Gradient-Guided Input Optimization (GGIO)

### Core Insight
Our current prefix perturbation is RANDOM. But we have access to the model's gradients (even with frozen weights). Instead of random perturbation, OPTIMIZE the input embedding to maximize a quality signal using gradient descent on the input space.

This is inference-time prompt optimization — not prompt tuning (which requires training data), but direct optimization of the input to produce better output for THIS SPECIFIC QUERY.

### How It Differs From Learned Soft Prompts
- Soft prompt tuning (Lester et al.): learns prefix across a DATASET. Requires training loop over many examples.
- Our approach: optimizes prefix for ONE QUERY at inference time. No training data. No generalization needed.
- Related: GCG (adversarial suffix optimization) does this for jailbreaks. We do it for quality.

### System Design

**Component 1: Quality Signal**
The optimization needs a differentiable objective. Options:
- **Log-probability of known answer** (for tasks with verifiable answers): maximize P(correct_answer | prefix + question)
- **Self-certainty**: maximize KL divergence between output distribution and uniform (model confidence)
- **Coherence proxy**: minimize entropy of output token distribution (sharper = more certain)
- **Answer-consistency**: generate answer, then maximize P(answer | prefix + question) — circular but effective
- For open-ended tasks: maximize P(high-quality template tokens | prefix + question)

**Component 2: Prefix Optimizer**
```
prefix_embeds = torch.randn(1, K, embed_dim, requires_grad=True)
# RMS-match to embedding scale
prefix_embeds.data *= (model_embed_rms / prefix_embeds.data.norm(dim=-1, keepdim=True))

optimizer = torch.optim.Adam([prefix_embeds], lr=0.01)

for step in range(T_steps):
    # Forward pass with prefix
    full_embeds = torch.cat([prefix_embeds, question_embeds], dim=1)
    logits = model(inputs_embeds=full_embeds).logits
    
    # Compute quality signal
    loss = -quality_signal(logits, target)
    
    # Backward through model (frozen weights, gradient flows to prefix only)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Project back to embedding manifold (RMS constraint)
    with torch.no_grad():
        prefix_embeds.data *= (model_embed_rms / prefix_embeds.data.norm(dim=-1, keepdim=True))

# Generate with optimized prefix
output = model.generate(inputs_embeds=torch.cat([prefix_embeds.detach(), question_embeds], dim=1))
```

**Component 3: Multi-Start Optimization**
- Run the optimizer from M different random initializations (M=5)
- Each converges to a different local optimum (different trajectory class)
- Select the best output via oracle
- This gives us DIRECTED diversity: each starting point finds a different quality peak

**Component 4: Gradient Analysis (Bonus)**
- The gradient of the quality signal w.r.t. prefix embeddings tells us WHICH dimensions matter
- If only a few dimensions have large gradients, the effective perturbation space is low-dimensional
- This provides mechanistic insight for free

### Interface Contracts
- Quality signal: input = logits (seq_len, vocab_size), output = scalar loss (differentiable)
- Prefix optimizer: input = (question_embeds, quality_signal, K, T_steps), output = optimized_prefix_embeds
- Multi-start: input = (question, M), output = list of (prefix_embeds, quality_score) tuples

### Why This Might Work
- **GCG proves gradient-guided input optimization works**: Adversarial attacks optimize suffixes to elicit specific behaviors. Same math, different objective.
- **Our random perturbation already works**: GGIO replaces random search (our current method) with gradient-guided search. Strictly more efficient at finding good perturbations.
- **No training data needed**: The quality signal can be self-supervised (model confidence, coherence) or task-specific (known answer for arithmetic).
- **Combines naturally with best-of-N**: M starting points × 1 optimized output each = M candidates for oracle selection.

### Failure Modes
1. **Gradient vanishing through autoregressive generation**: The quality signal depends on the FULL generated output, but gradient must flow through many autoregressive steps. Vanishing gradients may make optimization ineffective. Mitigation: use short target sequences (first 32 tokens) or teacher-forcing (not truly autoregressive gradient).
2. **Optimization finds adversarial examples, not good reasoning**: The quality signal may be gameable. Model confidence (self-certainty) doesn't equal correctness. The optimizer may find inputs that make the model confidently wrong.
3. **Computational cost**: T_steps × forward+backward passes PER QUERY. With T=20, K=2, that's 20 forward+backward passes per query. Our random method needs only 1 forward pass per seed. GGIO is 20x more expensive per optimized candidate. But if M=5 optimized candidates beat N=100 random ones, it's 2x cheaper.
4. **Memory**: Backward pass through the full model requires storing activations. With Qwen3-4B Q4 at seq_len=1024, peak memory ~8GB per backward pass. Fits in 24GB but limits batch size.

### Adversarial Audit
- **This is NOT novel**: Gradient-guided prompt optimization is well-studied. AutoPrompt (Shin et al. 2020), GCG (Zou et al. 2023), and many others. What's novel is using it for REASONING IMPROVEMENT rather than adversarial attacks or prompt search.
- **The quality signal is the hard problem**: If we have a differentiable quality signal that correlates with reasoning correctness, many architectures benefit (not just GGIO). The architecture is only as good as its objective function.
- **For arithmetic**: We can compute the gradient of P(correct_answer | prefix + question). This is the STRONGEST possible signal — direct supervision on the answer. But this requires knowing the answer, which defeats the purpose. Mitigation: use P(any_valid_number | ...) or consistency-based objectives.
- **For open-ended tasks**: No ground truth → quality signal must be self-supervised. Self-certainty is the most principled option, but it's a weak signal for legal/planning tasks.

### Verdict
Theoretically sound, practically challenging. The quality signal problem is fundamental — it's the same bottleneck as Architecture 2 (quality monitor) and Architecture 4 (quality oracle). But gradient-guided search is strictly more efficient than random search IF the gradient is informative. Worth testing on arithmetic (where the ground-truth quality signal exists) to establish whether gradient-guided prefix optimization outperforms random prefix perturbation.

---

## Architecture 9: Decompose-Dispatch-Compose (DDC)

### Core Insight
Hard problems are hard because they require MULTIPLE reasoning steps. Small models fail not because they lack knowledge, but because they can't maintain coherent multi-step reasoning in a single autoregressive pass. What if we DECOMPOSE the problem into sub-problems, solve each one independently (short, easy generation), then COMPOSE the results?

### How It Differs From Chain-of-Thought
- CoT: the model generates ALL steps sequentially in ONE pass. If any step goes wrong, everything downstream is corrupted.
- DDC: each step is generated INDEPENDENTLY in a fresh context. Errors don't propagate. Each sub-problem gets full attention capacity.

### System Design

**Component 1: Problem Decomposer**
- Input: the original problem
- Output: a sequence of sub-problems, each solvable in isolation
- Implementation: the LLM itself, with a decomposition prompt
```
"Break this problem into independent sub-steps. For each step, write a self-contained question that can be answered without seeing the other steps.

Problem: What is (7 × 8) + (12 × 3)?

Sub-problems:
1. What is 7 × 8?
2. What is 12 × 3?
3. Given that A = [answer to 1] and B = [answer to 2], what is A + B?"
```

**Component 2: Sub-Problem Solver**
- For EACH sub-problem independently:
  - Generate N=5 candidate answers (using prefix perturbation or temperature)
  - Select the best via majority voting (arithmetic) or oracle
  - Confidence score = agreement rate among N candidates
- This is our current approach, applied to EASIER sub-problems

**Component 3: Answer Composer**
- Inject sub-problem answers into the final composition step
- Generate the final answer with sub-answers provided as FACTS in the prompt
```
"Given the following facts:
- 7 × 8 = 56
- 12 × 3 = 36
What is 56 + 36?"
```
- This final step is MUCH easier than the original problem

**Component 4: Confidence Router**
- If the decomposer's confidence is low (it can't break down the problem), fall back to direct generation
- If a sub-problem's confidence is low (N candidates disagree), mark it as uncertain and try alternative decompositions

### Interface Contracts
- Decomposer: input = problem_text, output = list[(sub_problem_text, dependency_list)]
- Sub-solver: input = sub_problem_text, output = (answer, confidence, candidates)
- Composer: input = (problem_text, sub_answers_dict), output = final_answer
- Router: input = confidence_scores, output = strategy (decompose|direct|retry)

### Why This Might Work
- **Divide and conquer is fundamental**: Hard problems are easier when broken down. This is true in computation theory and in human cognition.
- **Small models are GOOD at easy problems**: Qwen3-4B can compute 7×8 reliably. It fails on (7×8)+(12×3) because the multi-step reasoning corrupts intermediate results. By isolating each step, we eliminate error propagation.
- **Self-consistency becomes powerful**: Majority voting on easy sub-problems has high accuracy (the model mostly agrees with itself on easy tasks). Majority voting on the hard combined problem is less reliable.
- **Composability**: Sub-answers can be verified independently. If 7×8=56 and 12×3=36 are both verified, the only remaining risk is the trivial addition step.

### Failure Modes
1. **Decomposition fails**: The model may not decompose well. "What is the legal liability for..." doesn't naturally decompose into independent sub-problems. Decomposition itself is a hard reasoning task.
2. **Dependencies between sub-problems**: Real reasoning is rarely fully parallelizable. Sub-problems may depend on each other's answers, requiring sequential solving. This reduces the benefit.
3. **Composition overhead**: The compose step adds a full generation pass. If the sub-problems are each cheap but the composition is hard, we've just moved the problem.
4. **Token overhead**: The decomposition prompt + N sub-problem prompts + composition prompt = much more total token generation than a single direct attempt. May be 5-10x more expensive.

### Adversarial Audit
- **This is well-studied**: Decomposed Prompting (Khot et al. 2022), Least-to-Most Prompting (Zhou et al. 2022). Not novel as a paradigm.
- **The novelty is in COMBINING with perturbation**: Apply prefix perturbation to each sub-problem independently. This gives us diversity at the sub-problem level, where it's most effective (easy problems, high agreement rate). The combination of decomposition + perturbation + voting is not well-studied.
- **For arithmetic**: This is extremely promising. Multi-digit arithmetic DECOMPOSES naturally. Each digit-level operation is easy. The model's failure mode is exactly "error propagation across steps."
- **For legal reasoning**: Decomposition is harder but possible. Legal analysis naturally has sub-questions: "Does duty of care apply?" "Was there a breach?" "Was there causation?" Each can be analyzed independently.
- **Key test**: Does decomposition + N=3 per sub-problem beat direct N=10? If yes, decomposition is strictly better use of the compute budget.

### Verdict
HIGH PRACTICAL VALUE. Not novel as a paradigm, but the combination with perturbation is underexplored. The key insight: perturbation is most effective on EASY problems (where the model has the knowledge but may take a wrong greedy path). Decomposition makes hard problems into easy problems. The synergy is clear. Priority: test on arithmetic first (natural decomposition), then legal (harder decomposition).

---

## Architecture 10: Retrieval-Augmented Reasoning from Self (RARS)

### Core Insight
Large language models store enormous knowledge in their weights. Small quantized models have the same knowledge, but accessing it is harder — the quantization noise and limited capacity mean the model's "memory lookup" is less reliable. What if we use the model ITSELF as a knowledge base, querying it for relevant facts BEFORE asking it to reason?

### How This Works
Instead of: "What is the legal liability for X?" → model reasons from scratch

Do: 
1. "List 5 relevant legal principles for analyzing X" → model retrieves knowledge
2. "List any relevant case law for X" → model retrieves more knowledge
3. "Given these principles and cases, analyze the legal liability for X" → model reasons with explicit context

The model does TWO SEPARATE things: first RETRIEVAL (easy, pattern matching), then REASONING (hard, but now grounded in retrieved facts).

### System Design

**Component 1: Knowledge Elicitor**
- Given the problem, generate M different knowledge-elicitation queries:
  - "What are the key concepts related to [topic]?"
  - "What are the common mistakes when solving [type of problem]?"
  - "What is the standard procedure for [task]?"
  - "List relevant formulas/rules for [domain]"
- Run each query with N=3 perturbation seeds, take the union of unique facts
- This is like RAG, but the "database" is the model's own weights

**Component 2: Knowledge Deduplication and Verification**
- Merge all elicited facts
- Remove duplicates (exact or semantic)
- Cross-verify: facts mentioned by multiple seeds are more reliable
- Optionally: self-verify each fact ("Is it true that [fact]? Answer yes or no.")

**Component 3: Grounded Reasoner**
- Inject the verified facts as context:
```
"The following facts are relevant to this problem:
1. [fact 1]
2. [fact 2]
...
Given these facts, solve: [original problem]"
```
- This gives the model explicit access to knowledge it already has but might not retrieve during a single reasoning pass

**Component 4: Multi-Path with Different Knowledge Sets**
- Different perturbation seeds elicit different facts
- Generate multiple knowledge sets, each leading to a different grounded reasoning path
- Oracle selects the best final answer
- This is COMPOSITIONAL DIVERSITY: diversity in both knowledge retrieval AND reasoning

### Why This Might Work
- **Human analogy**: Humans don't reason from scratch. They first recall relevant knowledge (rules, precedents, formulas), then apply it. We're giving the model the same two-phase process.
- **Small model limitation**: Quantized models often "know" the right answer but fail to access it in a single pass. Explicit knowledge elicitation is a second chance to surface relevant information.
- **Works with our perturbation**: Different perturbation seeds elicit different knowledge subsets. The diversity is now at the KNOWLEDGE level, not just the GENERATION level. A perturbation that helps the model recall a critical fact is more valuable than one that just changes the phrasing.
- **Testable**: Compare "direct reasoning" vs "elicit-then-reason" on legal tasks, where the model's knowledge access is the bottleneck.

### Failure Modes
1. **Elicited facts are wrong**: The model may confabulate facts. Self-verification doesn't help if the model is consistently wrong about the same fact.
2. **Token overhead**: Elicitation queries + deduplication + grounded reasoning = potentially 5-10x more tokens than direct generation.
3. **Irrelevant facts dilute context**: If the elicited facts are mostly irrelevant, they may HURT reasoning by consuming context window space.
4. **Arithmetic doesn't benefit**: For arithmetic, there are no "relevant facts" to elicit beyond the operation rules (which the model already knows). This architecture is domain-specific.

### Adversarial Audit
- **This is just prompt engineering**: Yes — it's structured multi-turn prompting. The novelty claim is not the technique but the combination with perturbation-based diversity at the knowledge elicitation stage.
- **Chain-of-Thought already includes knowledge**: A good CoT prompt implicitly causes the model to recall relevant knowledge. RARS makes this explicit and parallelizable.
- **For legal reasoning**: HIGHLY promising. Legal analysis depends heavily on correctly identifying the applicable legal principles. The model's legal knowledge may be extensive but poorly accessed via a single generation.
- **For arithmetic**: NOT useful. Skip this architecture for arithmetic tasks.
- **Cost-effectiveness**: If M=3 elicitation queries × N=3 seeds = 9 elicitation passes, plus the grounded reasoning, that's ~12 generation passes total vs N=10 direct generation. Similar cost, potentially better accuracy if knowledge access is the bottleneck.

### Verdict
DOMAIN-SPECIFIC but potentially very powerful for knowledge-intensive tasks (legal, planning, factual reasoning). Not useful for computational tasks (arithmetic). The key question: does the model produce DIFFERENT and MORE RELEVANT knowledge under perturbation than under vanilla prompting? If yes, this is a high-value architecture for our legal reasoning evaluation.

---

## Architecture 11: Causal Intervention Reasoning (CIR)

### Core Insight
The model's reasoning process has CAUSAL STRUCTURE — some hidden-state features cause others, and ultimately cause the output tokens. Current interventions (prefix perturbation, activation steering) treat the model as a black box and perturb randomly. What if we identify the CAUSAL BOTTLENECKS in reasoning and intervene specifically at those points?

### Theoretical Foundation
Inference-time causal intervention requires identifying:
1. **Which hidden features encode reasoning-relevant information** (localization)
2. **Which features causally influence the output** (not just correlate) (intervention)
3. **What interventions at those features improve reasoning** (optimization)

This is the mechanistic interpretability approach applied as an inference-time intervention tool, not a post-hoc analysis tool.

### System Design

**Component 1: Causal Feature Discovery (Offline)**
- Run the model on N problems with known correct/incorrect outcomes
- For each layer L and feature dimension D:
  - Patch: replace feature (L, D) from a correct trajectory into an incorrect one
  - Measure: does the output change toward the correct answer?
  - If yes: feature (L, D) is causally relevant for reasoning
- Build a "causal map": which (layer, dimension) pairs are reasoning-critical

**Component 2: Bottleneck Identifier**
- From the causal map, find the MINIMAL set of features that, when patched, flip incorrect → correct
- These are the "causal bottlenecks" — the points where reasoning succeeds or fails
- Typically: a few dozen dimensions at 2-3 specific layers

**Component 3: Targeted Perturbation**
- Instead of random prefix perturbation (perturbs ALL dimensions equally):
  - Perturb ONLY the causal bottleneck dimensions
  - Use DIRECTED perturbation: move in the direction that the causal map says helps
  - This is like activation steering but informed by causal analysis
```
for each generation:
    # Identify bottleneck features for this problem type
    bottleneck_dims = causal_map[problem_type]
    
    # Create targeted perturbation
    perturbation = torch.zeros(embed_dim)
    for (layer, dim, direction) in bottleneck_dims:
        perturbation[dim] = alpha * direction  # direction from causal analysis
    
    # Apply at the identified layer during forward pass
    model.register_hook(layer, lambda h: h + perturbation)
    output = model.generate(...)
```

**Component 4: Diverse Causal Interventions**
- Each seed uses a different SUBSET of bottleneck features to perturb
- This creates structured diversity: each candidate explores a different causal pathway
- Unlike random perturbation (which may waste effort on non-causal dimensions), every perturbation is guaranteed to hit a causal lever

### Interface Contracts
- Causal discovery: input = (model, problems, answers), output = causal_map[(layer, dim, direction, effect_size)]
- Bottleneck identifier: input = causal_map, output = minimal_set[(layer, dim, direction)]
- Targeted perturbation: input = (bottleneck_set, alpha, seed), output = model_hooks
- Diverse interventions: input = (bottleneck_set, N_candidates), output = N different hook configurations

### Why This Might Work
- **Causal interventions are more efficient than correlational ones**: Random perturbation hits all dimensions; only a few are causally relevant. Targeted perturbation has higher signal-to-noise ratio.
- **Activation patching is established**: The mech interp community has demonstrated that small sets of features causally control model behaviors. This applies that finding to inference-time improvement.
- **Explains our current results**: If our random prefix perturbation works because it ACCIDENTALLY perturbs causal bottlenecks in some seeds, then targeted perturbation should work MORE CONSISTENTLY and with SMALLER perturbation magnitude.
- **Provides mechanistic understanding**: The causal map directly answers "what is our prefix perturbation doing?" — it perturbs feature X at layer Y, which causes the model to take a different reasoning path.

### Failure Modes
1. **Causal discovery is expensive**: Activation patching requires O(layers × dimensions × problems) forward passes. With 32 layers, 3072 dimensions, 25 problems = 2.4M forward passes. Must be done smartly — use coarse-to-fine: test by layer first, then by dimension within promising layers.
2. **Causal structure is problem-specific**: The causal bottlenecks for arithmetic may differ from legal reasoning. Need separate causal maps per domain.
3. **Causal does not mean controllable**: Knowing that feature (L=12, D=1547) causally affects reasoning doesn't tell us WHICH VALUE of that feature produces better reasoning for a NEW problem.
4. **The model is quantized**: Q4 quantization may distort causal relationships. Features that are causal in the full-precision model may not be causal after quantization.

### Adversarial Audit
- **Activation patching already exists**: This is not novel — it's a well-known technique from the mech interp literature. What's novel is using it as an INFERENCE-TIME INTERVENTION rather than a POST-HOC analysis tool.
- **Computational feasibility**: Full activation patching on Qwen3-4B Q4 with 25 problems: even with coarse-to-fine, this is thousands of forward passes. ~2 hours of GPU time for the offline analysis. The per-query overhead is minimal (just targeted hooks).
- **Comparison with random perturbation**: If targeted causal perturbation beats random prefix perturbation on the same task set, that PROVES our mechanism is about hitting causal bottlenecks, not just adding noise. This is the STRONGEST possible evidence for our mechanistic claims.
- **The causal map IS the paper's contribution**: If we can show the causal map — "these specific features at these specific layers are the reasoning bottlenecks, and perturbing them is what our random prefix does accidentally" — that's a complete mechanistic story.

### Verdict
EXTREMELY high value for understanding our mechanism. The offline cost is significant but one-time. The per-query benefit is potentially transformative — replacing random search with targeted intervention. This architecture should be implemented AFTER the temperature comparison (to establish that our method works) but BEFORE the paper submission (to explain WHY it works). If the causal map shows that only 5-10 dimensions matter, that's a publishable result by itself.

---

## Architecture 12: Iterated Distillation Reasoning (IDR)

### Core Insight
Don't generate the answer in one pass. Generate a DRAFT, then use the model to CRITICIZE the draft, then generate a REVISION that addresses the criticism. Repeat until convergence. This is "iterative self-refinement" — but with a twist: each iteration uses a DIFFERENT perturbation, so the critic sees the draft from a different "perspective."

### How It Differs From Architecture 7 (Self-Debate)
- Self-Debate: generates TWO candidates, judges between them, synthesizes
- IDR: generates ONE candidate, generates criticism from a PERTURBED perspective, revises iteratively
- The key difference: IDR uses perturbation to CREATE DIVERSE CRITICISM, not diverse candidates

### System Design

**Component 1: Draft Generator**
- Standard generation with optional prefix perturbation
- Produces initial draft D_0

**Component 2: Diverse Critic**
- For each iteration i:
  - Apply a DIFFERENT prefix perturbation (seed i)
  - Prompt: "Here is a response to [question]. Identify any errors, gaps, or weaknesses:\n\n[D_{i-1}]"
  - The perturbation means each critic "sees" the draft from a different angle
  - Output: criticism C_i

**Component 3: Revision Generator**
- Prompt: "Here is a response with identified issues:\n\nResponse: [D_{i-1}]\n\nIssues found: [C_i]\n\nGenerate an improved response addressing these issues."
- Apply yet another perturbation for the revision
- Output: revised draft D_i

**Component 4: Convergence Detector**
- Compare D_i with D_{i-1}
- If substantially similar (high overlap in key terms/numbers): converged, stop
- If the critic finds no new issues: converged, stop
- Maximum iterations: 5 (to bound compute)

```
draft = generate(question, prefix=random_prefix(seed=0))

for iteration in range(max_iterations):
    # Diverse criticism
    criticism = generate(
        f"Identify errors in: {draft}",
        prefix=random_prefix(seed=iteration*2 + 1)
    )
    
    if criticism contains "no errors found" or "response is correct":
        break
    
    # Revision
    draft = generate(
        f"Improve this response addressing: {criticism}\n\nOriginal: {draft}",
        prefix=random_prefix(seed=iteration*2 + 2)
    )

return draft
```

### Why This Might Work
- **Self-refinement literature**: SELF-REFINE (Madaan et al. 2023) shows iterative refinement improves output quality on many tasks. Our twist: perturbation-diverse critics provide DIFFERENT criticisms each round.
- **Error accumulation is the enemy**: In single-pass generation, early errors compound. IDR catches and corrects errors iteratively.
- **Perturbation serves a NEW role**: In our current approach, perturbation creates diverse candidates for selection. In IDR, perturbation creates diverse CRITICS. A perturbation that makes the model notice an error it normally misses is extremely valuable.
- **Converges naturally**: Good outputs get "no errors" quickly and stop. Bad outputs get iteratively improved. Compute is allocated adaptively.

### Failure Modes
1. **Self-refinement doesn't work for small models**: SELF-REFINE's original paper found that small models (< 7B) often DEGRADE quality during refinement. They can't reliably critique their own outputs.
2. **Perturbation-diverse criticism may just be noise**: If the perturbation doesn't make the model a BETTER critic but just a DIFFERENT one, the criticism may be random — sometimes helpful, sometimes harmful.
3. **Token cost**: Each iteration requires criticism (500+ tokens) + revision (500+ tokens). 5 iterations = 5000+ tokens of overhead per problem. That's 5x the cost of a single N=10 best-of-N run.
4. **Convergence to wrong answer**: If the model is consistently wrong about a fact, no amount of self-criticism will find the error. All perturbation seeds may share the same knowledge gap.

### Adversarial Audit
- **Small model self-refinement is known to fail**: Multiple papers show < 7B models make quality WORSE during refinement. This is a near-fatal flaw for our 4B quantized model.
- **However**: The PERTURBATION may change this. Standard self-refinement uses the same model perspective for both generation and criticism. Our diverse critics may find errors the generator perspective misses. This is testable.
- **For arithmetic**: The critic can verify intermediate calculations. If the draft says "7 × 8 = 54," the critic (from a different perturbation seed that gets the multiplication right) catches the error. This is EXACTLY the scenario where perturbation + self-refinement synergizes.
- **For legal reasoning**: The critic can identify missing considerations. Different perturbation seeds may cause the model to weight different legal principles, providing complementary criticism.
- **Key experiment**: Compare single-pass perturbation (our current N=10 oracle) vs 5-iteration IDR (same compute budget: ~10 forward passes). If IDR converges to a BETTER answer than the best of 10 random candidates, iterative refinement beats parallel search.

### Verdict
Theoretically appealing but empirically risky for small models. The perturbation-diverse criticism is the genuinely novel element — it's a different use of perturbation than anyone has explored. Worth testing, but expect failure on the first attempt. If it works, it's a paper on its own: "Perturbation-Diverse Self-Refinement for Small Language Models."

---

## Architecture 13: Attractor Landscape Mapping (ALM)

### Core Insight
From our cross-domain research: the model's generation is a dynamical system with ATTRACTOR BASINS. Each basin is a class of outputs (correct answer, wrong answer, degenerate loop, etc.). Our prefix perturbation works by knocking the system into different basins.

ALM makes this explicit: MAP the attractor landscape, then NAVIGATE it deliberately.

### Theoretical Connection
- Our cross-domain research established the Kramers escape analogy: perturbation provides energy to escape a local attractor
- ALM goes further: instead of random escape attempts, map the landscape and find the OPTIMAL escape route

### System Design

**Component 1: Attractor Basin Sampler**
- Generate N=100 outputs from random perturbation seeds (cheap, fast)
- Cluster outputs by their first-32-token embeddings (trajectory class)
- Each cluster = one attractor basin
- Record: cluster center, radius, frequency (how many seeds land here), and outcomes (correct/incorrect)

**Component 2: Basin Quality Map**
- For each basin: what fraction of outputs are correct?
- Some basins are "correct-answer basins" (most outputs in this cluster are correct)
- Some are "trap basins" (degenerate, repetitive, always wrong)
- The map: {basin_id: (center, radius, frequency, quality_score)}

**Component 3: Basin Navigator**
- Given the basin map, the goal is to STEER toward high-quality basins and AWAY from trap basins
- Use the prefix perturbation, but constrained:
  - Compute the direction from the default trajectory (no perturbation) to the nearest high-quality basin center
  - Perturb in THAT direction specifically
  - This is gradient-free: just use the basin center as a target in embedding space

```
# Map the landscape (one-time per task type)
for seed in range(100):
    prefix = random_soft_prefix(seed)
    output = generate(question, prefix=prefix)
    embedding = embed_first_32_tokens(output)
    quality = evaluate(output)
    data.append((embedding, quality, seed))

clusters = cluster(data.embeddings, method='DBSCAN')
basin_map = {c: (centroid(c), quality_mean(c)) for c in clusters}
best_basin = max(basin_map, key=lambda c: basin_map[c].quality)

# Navigate toward best basin (per-query)
default_trajectory = embed_first_32_tokens(generate(question))
direction = basin_map[best_basin].centroid - default_trajectory
# Project direction back to prefix embedding space (approximately)
optimized_prefix = direction_to_prefix(direction)
output = generate(question, prefix=optimized_prefix)
```

**Component 4: Multi-Basin Exploration**
- Don't just target ONE basin — target the top-K quality basins
- Generate one candidate steered toward each
- Oracle selects the best
- This is STRUCTURED best-of-N: each candidate targets a different quality peak

### Why This Might Work
- **Makes our implicit mechanism explicit**: Our current approach IMPLICITLY searches over basins by random prefix perturbation. ALM does this EXPLICITLY with a map.
- **Efficiency**: Random search with N=10 may land in the same basin 7 times (wasted diversity). ALM with N=5 targets 5 DIFFERENT high-quality basins (no waste).
- **Diagnostic value**: The basin map directly shows: how many attractor basins exist? What's the quality distribution? How reachable is the correct-answer basin? This answers fundamental questions about our mechanism.
- **Generalizable**: Once the basin map is computed for a task type (e.g., "multiplication"), it applies to all multiplication problems. The mapping cost is amortized.

### Failure Modes
1. **Basin structure is problem-specific**: The attractor basins for "7×8" may differ from "12×3." If every problem has its own basin structure, mapping is not amortizable and must be done per-query (too expensive).
2. **First-32-token embedding is too coarse**: Output diversity may be in later tokens, not the first 32. Clustering on first-32 tokens may miss important basin structure.
3. **Direction-to-prefix projection is lossy**: The mapping from "desired trajectory direction" to "prefix embedding" is highly nonlinear. A prefix that points the trajectory embedding in the right direction may not actually land in the target basin.
4. **100 samples may be insufficient**: Complex landscapes may have many basins. N=100 may not cover the space adequately.

### Adversarial Audit
- **This is essentially surrogate-model optimization**: Build a surrogate model of the perturbation-to-quality landscape, then optimize. Well-studied in black-box optimization literature.
- **The "basin" metaphor may not hold**: LLM generation may not have discrete attractor basins. The landscape may be smooth and high-dimensional with no clear basin structure. If DBSCAN finds only 1-2 clusters, the metaphor fails.
- **However**: Our existing data shows distinct trajectory classes (from the cross-domain research). The basin metaphor has empirical support. The question is whether the structure is CONSISTENT across problems.
- **Combines perfectly with our current approach**: ALM doesn't replace our method — it OPTIMIZES it. Use the basin map to select better random seeds, not to replace random perturbation entirely.
- **Paper value**: A published attractor basin map of Qwen3-4B's reasoning space would be a novel contribution regardless of whether it improves accuracy. It's interpretability + intervention in one architecture.

### Verdict
HIGH VALUE for both practical improvement AND mechanistic understanding. The basin map is valuable regardless of whether navigation works. Start with mapping (N=100 random perturbations, cluster, analyze), then attempt navigation. If the map shows clear basin structure with quality variation, navigation is justified. If the landscape is flat or unimodal, the architecture provides negative but scientifically valuable results.

---

## Architecture 14: Neuro-Symbolic Hybrid Reasoning

### Core Insight
Small LLMs are good at pattern recognition and language understanding but BAD at precise computation. Symbolic engines (Python, Prolog, Z3) are perfect at computation but can't understand natural language. Combine them: LLM translates the problem into a symbolic representation, the symbolic engine solves it, LLM translates back.

### How It Differs From Program-of-Thought (PoT)
- PoT: model generates a Python program, executes it, returns the result. Well-studied.
- Our twist: the LLM generates MULTIPLE symbolic representations under different perturbation seeds. Different perturbations may produce different formalizations — some correct, some wrong. The symbolic engine identifies which formalizations are self-consistent (constraint satisfaction), and the LLM translates the consistent solutions back.

### System Design

**Component 1: Symbolic Translator**
- Input: natural language problem
- Output: symbolic representation (Python expression, logical formula, constraint set)
- Uses the LLM with perturbation:
```
for seed in range(N):
    prefix = random_soft_prefix(seed)
    symbolic = generate(
        "Translate this problem to a Python expression that computes the answer:\n" + problem,
        prefix=prefix
    )
    formalizations.append(symbolic)
```

**Component 2: Symbolic Executor**
- Execute each formalization in a sandboxed Python environment
- Record: result, execution success/failure, runtime
- Formalizations that crash → discard
- Formalizations that produce results → keep

**Component 3: Consistency Checker**
- Compare results across formalizations
- If majority agree → high confidence (the correct formalization is likely correct)
- If all disagree → the translation is the bottleneck
- For constraint problems: check if the solution satisfies all constraints

**Component 4: Back-Translator**
- Take the verified symbolic result
- Generate natural language explanation using the LLM
- This step is easy (the model just needs to describe a known-correct answer)

### Why This Might Work
- **Separates understanding from computation**: The LLM handles language understanding (its strength). The symbolic engine handles computation (its strength). Neither does what it's bad at.
- **Perturbation diversifies FORMALIZATIONS**: Different perturbation seeds may produce "7*8 + 12*3" vs "sum([7*8, 12*3])" vs "reduce(operator.add, [7*8, 12*3])." All correct, and their agreement confirms correctness. If one seed produces "7*8 + 12*4" (wrong operand), the majority voting catches it.
- **For legal reasoning**: Translate to logical rules. "If duty_of_care AND breach AND causation AND damages THEN liability." The symbolic engine can check logical consistency.
- **Verification is EXACT**: Unlike LLM-based oracle selection, the symbolic engine provides PROVABLY correct answers for computable problems.

### Failure Modes
1. **Translation is the hard part**: If the LLM can't correctly translate to symbolic form, the entire pipeline fails. Translation requires the same understanding that direct reasoning does.
2. **Not all problems are symbolically representable**: Legal reasoning, planning, and creative tasks don't have clean symbolic representations. This limits the architecture's scope.
3. **Execution environment setup**: Sandboxed Python execution on the GPU machine adds complexity. Security concerns with arbitrary code execution.
4. **Round-trip token cost**: Translation + execution + back-translation = more tokens than direct generation, plus the symbolic execution overhead.

### Adversarial Audit
- **This is Program-of-Thought with voting**: PAL (Gao et al. 2023) and PoT (Chen et al. 2022) already do this without the perturbation. What's new is using perturbation to diversify the symbolic translations for majority-vote verification.
- **For arithmetic**: EXTREMELY promising. Python can compute 7*8 perfectly. The only failure mode is translation error, and perturbation+voting addresses that.
- **For legal reasoning**: The symbolic representation would need to be something like Prolog or first-order logic. Small models are TERRIBLE at generating formal logic. This is unlikely to work without significant prompt engineering.
- **Compute cost**: Each seed requires generating symbolic code (short, ~50 tokens) + executing (instant) + voting (trivial). Total cost per problem: N=10 × ~50 tokens generation + N symbolic executions. MUCH cheaper than N=10 × 1024 tokens for direct reasoning. The cost advantage alone justifies testing.

### Verdict
VERY HIGH VALUE for computational tasks (arithmetic, logic, programming). The perturbation-diversified symbolic translation with majority voting is a clean, efficient architecture that leverages both the model's language understanding and exact symbolic computation. For arithmetic specifically, this may outperform all other architectures by a large margin. For legal/planning, less applicable but the logical-rules formalization is worth exploring.

---

## Architecture 15: Energy-Based Reranking with Learned Landscape (EBRL)

### Core Insight
Our current oracle selection is binary: pick the output that matches the expected answer format best. What if we train a tiny energy-based model that assigns CONTINUOUS quality scores to outputs, then use it to rerank candidates AND to guide prefix optimization?

### Theoretical Connection
Energy-based models (EBMs) define a scalar energy function E(x) over the output space. Low energy = high quality. The EBM learns the landscape of output quality, and we can:
- **Rerank**: sort candidates by energy (lower = better)
- **Navigate**: use gradient of energy to optimize the prefix (Architecture 8 + EBM)
- **Reject**: set an energy threshold below which outputs are discarded

### System Design

**Component 1: Energy Function (Tiny Model)**
- A small MLP or transformer (1-10M params) that maps output text → scalar energy
- Trained on (output, quality_label) pairs from existing experiments
- We already have data: N=10 outputs per task × 25 tasks × multiple models = hundreds of labeled examples
- Training target: energy should be LOW for correct outputs, HIGH for incorrect
```python
class EnergyModel(nn.Module):
    def __init__(self, embed_dim=384, hidden=256):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # frozen
        self.energy_head = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, text):
        embedding = self.encoder.encode(text)
        return self.energy_head(embedding)  # scalar energy
```

**Component 2: Contrastive Training**
- For each task: we have correct and incorrect outputs (from existing data)
- Training: contrastive loss — energy(correct) should be lower than energy(incorrect) by margin M
```
loss = max(0, energy(correct) - energy(incorrect) + margin)
```
- Train on existing data (no new generation needed)
- Small model, small dataset → trains in minutes

**Component 3: Energy-Based Reranking**
- Generate N candidates (using any method: prefix perturbation, temperature, etc.)
- Score each with the energy model
- Select the lowest-energy candidate
- This replaces our current extract_answer + exact match oracle with a LEARNED quality assessor

**Component 4: Energy-Guided Prefix Optimization (combines with Architecture 8)**
- Instead of using log-prob or self-certainty as the quality signal for GGIO:
- Use the energy model's score as the optimization objective
- Gradient flows: prefix → LLM → output → energy model → score
- Since the energy model is differentiable, this provides a training-free way to optimize the prefix

### Why This Might Work
- **We already have the training data**: Our existing experiments provide hundreds of (output, correct/incorrect) pairs. No new data collection needed.
- **Small energy model ≠ our broken scorer**: Our latent-space scorer was untrained and unreliable. A properly trained contrastive energy model on real output text should be much better.
- **Energy landscapes are interpretable**: The energy function reveals which output features the model associates with quality. This provides interpretability for free.
- **Combines with everything**: The energy model is a PLUG-IN quality assessor. It works with prefix perturbation, temperature sampling, self-debate, decomposition — any architecture that produces candidates.

### Failure Modes
1. **Insufficient training data**: Hundreds of examples may not be enough to learn a robust quality function. The energy model may overfit to surface features (length, keyword presence).
2. **Distribution shift**: The energy model is trained on outputs from specific conditions (Qwen3-4B Q4, arithmetic/legal). It may not generalize to new tasks, models, or generation strategies.
3. **Technically requires training**: This architecture requires training the energy model. It doesn't train the LLM, but it trains an auxiliary model. This may violate the "no training" constraint.
4. **The energy gradient may not flow usefully through the LLM**: Architecture 8's gradient-guided optimization requires gradients from the energy model to flow through the LLM's forward pass. With a frozen quantized model, these gradients may be noisy or zero.

### Adversarial Audit
- **This IS training**: The energy model is trained. This bends the "no training, no fine-tuning" constraint. However, it's training a TINY model (1-10M params) on EXISTING data, not training the LLM.
- **Reward model reranking is well-studied**: RLHF uses reward models to select outputs. Our energy model is just a small reward model. Not novel.
- **The contrastive aspect is useful**: We already have the contrastive pairs (same task, different perturbation seeds → different outcomes). This is free data for contrastive learning.
- **BUT**: CLAUDE.md says "The automated scorer scores are IRRELEVANT" and warns against trusting numeric scores. An energy model IS a numeric scorer. The project's philosophy conflicts with this architecture.
- **Resolution**: The difference is that our original scorer was UNTRAINED (random projection). This energy model is TRAINED on real correct/incorrect labels with a proper contrastive loss. The project critique applies to the specific scorer, not to the concept of quality assessment.

### Verdict
Promising as a TOOL (reranking) but philosophically in tension with the project's anti-scorer stance. Worth building and TESTING — if the energy model reranking beats extract_answer + exact match, it validates the concept. If it doesn't beat the simple baseline, we have evidence that learned quality assessment is hard. Either way, it produces useful information.

---

## Architecture 16: Attention-Pattern Surgery

### Core Insight
Our cross-domain research identified attention sinks as a potential mechanism: tokens 0-1 absorb disproportionate attention, acting as "garbage collectors." Our prefix perturbation may work by disrupting this pattern. What if we DIRECTLY modify the attention patterns instead of indirectly affecting them via embedding perturbation?

### Direct vs Indirect Intervention
- **Indirect** (our current method): Perturb input embeddings → changes what positions 0-1 represent → changes how attention sinks form → changes reasoning
- **Direct** (this architecture): Modify the attention weights/mask during generation → directly control where attention goes → changes reasoning

### System Design

**Component 1: Attention Pattern Analyzer (Offline)**
- Run the model on N problems, extract attention matrices at each layer
- For correct vs incorrect outputs:
  - Where does attention go? (positions, layers, heads)
  - What's the attention entropy? (concentrated vs distributed)
  - Which heads attend to attention sinks? Which attend to semantically relevant positions?
- Build a profile: {head_id: (sink_attention_frac, semantic_attention_frac, reasoning_relevance)}

**Component 2: Sink-Reduction Mask**
- For heads with high sink attention fraction AND low reasoning relevance:
  - Add a negative bias to attention on positions 0-1 (reduce sink attention)
  - This forces the head to attend to other positions
```python
# During generation, at specific layers/heads:
attention_scores[sink_heads, :, :, :2] -= bias_value  # reduce attention to positions 0-1
```
- Bias value: tuned to REDUCE but not ELIMINATE sink attention (model still needs some degree of attention dumping)

**Component 3: Semantic-Enhancement Mask**
- For heads with high reasoning relevance:
  - Add a positive bias to attention on positions that contain the PROBLEM STATEMENT
  - This forces reasoning heads to attend more to the actual question
```python
# Identify question positions
question_positions = range(prefix_len, prefix_len + question_len)
attention_scores[reasoning_heads, :, :, question_positions] += enhance_value
```

**Component 4: Diverse Attention Configurations**
- Create N different attention modification configurations:
  - Config 1: reduce sink attention by 10%
  - Config 2: reduce sink attention by 30%
  - Config 3: enhance question attention by 20%
  - Config 4: reduce sinks + enhance question
  - Config 5-N: different combinations and magnitudes
- Each configuration → different generation → oracle selects best

### Interface Contracts
- Attention analyzer: input = (model, problems), output = head_profiles
- Sink mask: input = (head_profiles, bias_value), output = attention_bias_tensor
- Semantic mask: input = (head_profiles, enhance_value, question_positions), output = attention_bias_tensor
- Configuration generator: input = (head_profiles, N), output = N attention_bias_configurations

### Why This Might Work
- **Attention sinks are well-documented**: StreamingLLM, gated attention papers, and our own cross-domain research all identify attention sinks as a real phenomenon.
- **Direct intervention is more controllable than indirect**: Our prefix perturbation INDIRECTLY affects attention patterns (through embedding perturbation). Direct attention modification is more precise and predictable.
- **The gated attention probe (Qwen3.5) tests this hypothesis**: If Qwen3.5 (gated attention, no sinks) shows LESS benefit from our perturbation, it confirms that sink disruption is a key mechanism. Attention surgery would then be the principled version of what our prefix does accidentally.
- **HuggingFace supports this**: The `attention_mask` parameter in generate() can be extended with additive biases via custom attention implementations.

### Failure Modes
1. **Hook compatibility**: Flash attention and fused attention kernels don't support per-head attention biases. May need to disable optimized attention (2-3x slower).
2. **Which heads matter?**: With 32 layers × 32 heads = 1024 attention heads, the search space is large. Not all heads are about sinks or reasoning.
3. **Attention patterns may not be the mechanism**: If our perturbation works through MLP pathway effects (not attention), attention surgery misses the point entirely.
4. **Model fragility**: Small changes to attention patterns may cause catastrophic failures. The model's layer-to-layer communication depends on specific attention patterns. Disrupting them may corrupt all downstream computation.

### Adversarial Audit
- **The gated attention probe must come FIRST**: If Qwen3.5 shows the same perturbation benefit as Qwen3 (no change in effect despite gated attention), then attention sinks are NOT the mechanism and this architecture is useless.
- **Activation steering at attention level already exists**: Inference-time attention manipulation is studied (attention head transplantation, attention knockout). Not novel in isolation.
- **However**: Combining attention surgery with prefix perturbation for diverse candidate generation is novel. The question is whether the combination is more than the sum.
- **Feasibility check**: Must verify that HuggingFace's generate() with inputs_embeds supports custom attention biases on quantized models. This is a technical blocker that could kill the architecture.

### Verdict
CONTINGENT on the gated attention probe results. If attention sinks are confirmed as the mechanism, this is the most principled architecture — it directly intervenes at the identified mechanism. If sinks are not the mechanism, skip entirely. Design the probe first, implement this architecture only if the probe confirms.

---

## Architecture 17: Temporal Ensemble Reasoning (TER)

### Core Insight
Instead of generating N candidates in PARALLEL (same prompt, different perturbations), generate them SEQUENTIALLY where each candidate can SEE the previous candidates' outputs. This is "reasoning over time" — the model's context grows with each iteration, incorporating information from prior attempts.

### How It Differs
- **Parallel best-of-N** (current): N independent generations, oracle selects. No information sharing.
- **TER**: Sequential generations where each iteration's context includes prior outputs. Information accumulates.

### System Design

**Component 1: Initial Generation**
- Generate first candidate C_0 with standard greedy decoding (no perturbation)
- This is the baseline

**Component 2: Informed Iteration**
- For each subsequent candidate i=1..N-1:
```
Prompt: "Previous attempts to solve this problem:

Attempt 1: [C_0]
Attempt 2: [C_1]
...

Each attempt may have errors. Generate a new, improved attempt that avoids the mistakes in the previous ones."
```
- Apply prefix perturbation (seed i) to ensure each iteration approaches differently
- The model can LEARN from prior failures without explicit error identification

**Component 3: Progressive Quality Assessment**
- After each iteration, check if quality is improving
- If C_i is worse than C_{i-1} for 2 consecutive iterations: stop (diminishing returns)
- If C_i is identical to C_{i-1}: stop (converged)

**Component 4: Context Window Management**
- Each prior attempt adds ~500-1000 tokens to the context
- With max_context=4096 and question=100 tokens, we can fit ~4-6 prior attempts
- Beyond that: summarize prior attempts or drop the oldest
- Alternatively: only include the BEST prior attempt, not all of them

### Why This Might Work
- **Humans reason temporally**: When solving a hard problem, humans don't solve it N times independently. They try, learn from failure, and try again with accumulated insight.
- **Information sharing**: If C_0 gets the first step right but the second step wrong, and C_1 gets the second step right but the first wrong, C_2 can potentially combine both correct steps.
- **Natural error correction**: Showing the model its own failures is a form of few-shot learning. "Here's what went wrong before — avoid this."
- **Perturbation prevents convergence to same error**: Without perturbation, iterative generation converges to the same output (the model repeats itself). Perturbation ensures each iteration takes a genuinely different approach.

### Failure Modes
1. **Context pollution**: Prior incorrect attempts may MISLEAD the model. Instead of learning from failures, the model gets anchored to incorrect patterns.
2. **Context window consumption**: Each prior attempt uses context capacity. With small models (4K context), only 4-6 iterations fit. May not be enough for convergence.
3. **Cost**: Sequential generation means latency grows linearly. N=5 sequential iterations take 5x longer than N=5 parallel generations (can't parallelize).
4. **Small model inability**: "Learn from prior attempts" requires meta-cognitive ability that 4B models may lack. The model may just paraphrase the last attempt rather than genuinely improving.

### Adversarial Audit
- **This is iterative self-refinement (again)**: Architecture 12 (IDR) does something similar. The difference is that TER provides the raw prior outputs as context rather than explicit criticism. Which is better? TER is simpler but provides less guidance.
- **Evidence suggests this works for large models**: GPT-4 and Claude can iteratively improve when shown prior attempts. Small models? Unknown.
- **Token cost comparison**: 5 iterations × (question + 3 prior attempts × 500 tokens + new generation) = ~10K tokens total. vs N=10 parallel × 1K tokens = 10K tokens. Same total token budget, different structure. The question is whether SEQUENTIAL information sharing beats PARALLEL independence.
- **The perturbation is critical**: Without perturbation, this degrades to repeated generation (same output each time). The perturbation ensures genuine diversity. This is a NOVEL combination of perturbation + temporal ensemble.

### Verdict
MODERATE priority. Simple to implement, builds directly on our infrastructure, but uncertain whether small models can learn from their own prior outputs. The perturbation-ensured diversity is the key differentiator from standard iterative refinement. Test with a simple version: show the model its last 2 attempts + current question, apply perturbation, measure whether accuracy improves vs parallel best-of-N with the same total compute budget.

---

## Wave 2 Comparison Matrix

| Architecture | Novel? | Training? | Cost | Best Domain | Key Risk |
|---|---|---|---|---|---|
| 8. GGIO | Medium | No | High (T backward) | Math (ground truth) | Gradient vanishing |
| 9. DDC | Low (exists) | No | Medium (decompose) | Math, Legal | Decomposition quality |
| 10. RARS | Medium | No | Medium (elicit) | Legal, Planning | Confabulated facts |
| 11. CIR | High | No* | High (offline map) | All | Causal discovery cost |
| 12. IDR | High | No | Medium (iterations) | Legal, Math | Small model can't self-critique |
| 13. ALM | High | No | Medium (100 samples) | All | Basin structure may not exist |
| 14. Neuro-Symbolic | Medium | No | Low (symbolic) | Math, Logic | Translation quality |
| 15. EBRL | Medium | Yes (tiny) | Low (rerank) | All | Overfitting, project philosophy |
| 16. Attention Surgery | Medium | No | Low (hooks) | All | Mechanism must be attention |
| 17. TER | Medium | No | Medium (sequential) | All | Small model meta-cognition |

*CIR: the causal discovery is offline analysis, not training. The model weights are never updated.

### Top Architectures by Category

**Highest Novelty + Deepest Insight:**
1. Architecture 11: Causal Intervention Reasoning — the causal map is the paper
2. Architecture 13: Attractor Landscape Mapping — makes our mechanism explicit and navigable

**Most Practical (Easiest to Test):**
1. Architecture 14: Neuro-Symbolic — Python execution is trivial, perturbation adds voting
2. Architecture 9: Decompose-Dispatch-Compose — just prompt engineering + our existing pipeline

**Best Synergy with Current Method:**
1. Architecture 8: GGIO — replaces random perturbation with directed optimization
2. Architecture 16: Attention Surgery — directly implements the hypothesized mechanism

**Highest Risk / Highest Reward:**
1. Architecture 12: IDR — if small models can self-critique with diverse perturbation, it's transformative
2. Architecture 11: CIR — if the causal map reveals a clean bottleneck, it rewrites the paper narrative

---

## Cross-Wave Synthesis: Emergent Meta-Architecture

Looking across all 17 architectures (Wave 1: 1-7, Wave 2: 8-17), a META-ARCHITECTURE emerges:

### The Three Pillars of Inference-Time Reasoning

**Pillar 1: DIVERSIFICATION** — How to generate diverse candidates
- Random: prefix perturbation (current), temperature
- Informed: GGIO (gradient-guided), CIR (causal), ALM (basin navigation)
- Structural: DDC (decomposition), RARS (knowledge elicitation)

**Pillar 2: QUALITY ASSESSMENT** — How to evaluate candidates
- Binary: exact match, format checking
- Continuous: energy model (EBRL), self-certainty, log-probability
- Symbolic: program execution (neuro-symbolic), constraint satisfaction

**Pillar 3: ITERATIVE REFINEMENT** — How to improve candidates
- Single-pass: generate + select (current)
- Multi-pass: self-critique (IDR), temporal ensemble (TER), checkpoint-branch
- Closed-loop: ASM (hidden-state controller), inverse speculative (oracle-guided)

**The Optimal System** picks ONE technique from each pillar and combines them:
- Diversify with CIR (targeted causal perturbation)
- Assess with neuro-symbolic verification (exact for computable tasks) or energy model (for open-ended)
- Refine with IDR (perturbation-diverse self-critique)

This meta-architecture unifies all 17 designs under a common framework. Each individual architecture explores ONE pillar deeply; the meta-architecture combines the best of each.

### Why This Matters for the Paper
The paper can present:
1. The meta-framework (three pillars)
2. Our current method as a specific instantiation (random prefix diversification + exact-match assessment + single-pass)
3. The design space map showing where each architecture sits
4. Evidence that our specific instantiation is a good starting point (it works with zero training)
5. The roadmap for moving to more sophisticated instantiations (from random to causal, from exact-match to energy, from single-pass to iterative)

This reframes the paper from "we found a trick" to "we mapped the design space and this trick is the simplest working point in it."
