# Alternative Reasoning Architectures — System Design Document

## Status: DESIGN PHASE — Codex Review Required

## Premise

The current Latent Space Reasoning approach (random soft prefix → trajectory diversification → oracle selection) is ONE point in a vast design space. This document maps that space and develops deep system designs for fundamentally different approaches to inference-time reasoning improvement for small, frozen LLMs.

**Constraint**: No training. No fine-tuning. The model weights are frozen. All improvement must come from inference-time interventions.

**Hardware**: RTX 5090 Laptop, ~24GB VRAM. Models: 1B-8B parameters, quantized.

---

## The Design Space Map

Every inference-time reasoning intervention operates on one or more of these surfaces:

### Surface 1: Input Space
What the model sees before generation starts.
- **Token-level**: prompt rephrasing, few-shot examples, system prompts
- **Embedding-level**: soft prefix (our method), learned prompt tuning, embedding perturbation
- **Position-level**: position ID manipulation, context window shifting

### Surface 2: Internal Activations (during forward pass)
Manipulating the model's computation mid-flight.
- **Residual stream**: activation steering (add vectors at specific layers)
- **Attention patterns**: attention masking, head ablation, attention transplant
- **Hidden states**: layer-specific intervention, representation engineering
- **MLP outputs**: selective amplification/suppression of MLP contributions

### Surface 3: Generation Control (during autoregressive loop)
Controlling how tokens are selected and sequenced.
- **Sampling strategy**: temperature, top-p, top-k, typical sampling
- **Search**: beam search, tree search, MCTS, best-of-N
- **Backtracking**: detect errors, rewind KV cache, regenerate from checkpoint
- **Constrained generation**: force-guide structure (step-by-step, format)

### Surface 4: Post-Generation
What happens after generation completes.
- **Selection**: oracle, self-consistency, majority voting, reward model, self-certainty
- **Refinement**: self-critique, self-edit, iterative improvement
- **Verification**: symbolic checker, code execution, proof assistant

### Surface 5: Multi-Agent
Multiple model instances interacting.
- **Debate**: adversarial verification between instances
- **Ensemble**: diverse instances, aggregation
- **Delegation**: small model drafts, larger model verifies/guides
- **Collaboration**: different instances handle different reasoning steps

---

## Current Approach: Where We Sit

Our method operates on **Surface 1 (embedding-level)** + **Surface 4 (oracle selection)**. We perturb the input embedding, generate deterministically, then select the best output.

This is a narrow slice of the design space. The architectures below explore fundamentally different surfaces and combinations.

---

## Architecture 1: Activation State Machine (ASM) for Reasoning

### Source
"Steering LLMs' Reasoning With Activation State Machines" (ICLR 2025 Workshop). This is already published — we'd be building on it, not inventing it.

### Core Idea
Instead of perturbing the input (one-shot intervention at t=0), construct a **closed-loop controller** that monitors the model's hidden state at each generation step and applies corrective steering vectors when the reasoning trajectory deviates from "good" dynamics.

### System Design

**Component 1: Reasoning Trajectory Collector**
- Run the model on N problems with known correct answers
- For each correct trajectory: extract hidden states at every layer at every token position
- For each incorrect trajectory: extract the same
- Build a dataset of (hidden_state_t, layer, is_correct_trajectory) tuples

**Component 2: Trajectory Dynamics Learner**
- Learn the typical "dynamics" of good reasoning: how hidden states evolve step-by-step
- Use a lightweight recurrent model (GRU/LSTM, ~1M params) trained to predict h_{t+1} from h_t for correct trajectories
- This model captures the "ideal reasoning flow" in hidden-state space

**Component 3: Deviation Detector**
- At each generation step, compare the actual hidden state h_t with the dynamics model's prediction
- If ||h_t - predicted_h_t|| > threshold, the model is deviating from good reasoning dynamics
- The deviation vector (h_t - predicted_h_t) tells us HOW the model is deviating

**Component 4: Corrective Steering**
- When deviation detected, add a steering vector to the residual stream at the NEXT step
- Steering vector = -α * (h_t - predicted_h_t) projected onto the residual stream
- α is a small gain (0.01-0.1) — enough to nudge, not enough to corrupt
- This is literally PID control applied to transformer hidden states

**Component 5: Generation Loop**
```
for each token step t:
    h_t = model.forward(input_t)  # actual hidden state
    predicted_h_t = dynamics_model(h_{t-1})  # expected hidden state
    deviation = h_t - predicted_h_t
    if ||deviation|| > threshold:
        steering_vector = -alpha * project(deviation, residual_basis)
        h_t = h_t + steering_vector  # correct the deviation
    token_t = decode(h_t)
```

### Interface Contracts
- Dynamics model: input = hidden_state (dim=embed_dim), output = predicted next hidden_state
- Deviation detector: input = (actual, predicted), output = (deviation_magnitude, deviation_vector)
- Steering applicator: input = (deviation_vector, alpha), output = corrected hidden_state
- All operate on the residual stream at a SINGLE layer (the "control layer", determined empirically)

### Failure Modes
1. **Dynamics model overfits**: Learns the specific hidden-state sequence for training problems, not general reasoning dynamics. Mitigation: use diverse task types, regularize heavily.
2. **Steering corrupts generation**: The correction vector disrupts the model's other computations (attention, MLP). Mitigation: project steering onto the "reasoning subspace" only, small alpha.
3. **Latency**: per-step dynamics model inference adds latency. Mitigation: tiny model (~1M params), quantized, on same GPU.
4. **No clear "good dynamics"**: If correct trajectories are as diverse as incorrect ones (no shared dynamics), the dynamics model learns nothing. This is the existential risk.

### Adversarial Audit
- **Training data required**: The dynamics learner needs correct trajectories. This requires either (a) problems with known answers (arithmetic, logic) or (b) human-judged quality labels. For open-ended reasoning, this is a severe limitation.
- **Single control layer assumption**: The design picks one layer to intervene. But reasoning may not be localized — different layers handle different aspects. Multi-layer control is exponentially harder.
- **KV cache invalidation**: Modifying hidden states mid-generation may invalidate the KV cache, requiring full recomputation. This kills the latency benefit.
- **Circular dependency**: To learn what "good reasoning" looks like in hidden-state space, you need the model to already reason well on some problems. This bootstraps from existing capability, not from nothing.

### vs Our Current Approach
- **Advantage**: Closed-loop (detects and corrects during generation, not one-shot at t=0)
- **Advantage**: Targeted (corrects specific deviations, not random perturbation)
- **Disadvantage**: Requires training data (correct trajectories)
- **Disadvantage**: Per-step overhead (dynamics model inference at every token)
- **Disadvantage**: KV cache management complexity

### Verdict
Promising for domains with abundant correct-trajectory data (math, code, logic). Not viable for open-ended reasoning without quality labels. The "reasoning dynamics" assumption is strong and unverified.

---

## Architecture 2: Checkpoint-and-Branch (Trajectory Surgery)

### Core Idea
Instead of generating once and hoping for the best, generate in SEGMENTS. After each segment, evaluate quality. If quality is declining, rewind to a checkpoint and branch with a different continuation strategy.

This is **backtracking search** in generation space, with quality monitoring.

### System Design

**Component 1: Segment Generator**
- Generate in fixed-length segments (e.g., 64 tokens per segment)
- After each segment, save a KV cache checkpoint
- Multiple segments form a tree of partial generations

**Component 2: Quality Monitor**
- After each segment, evaluate generation quality using lightweight signals:
  - Mean log-probability of generated tokens (declining = losing coherence)
  - Repetition rate (increasing = stuck in loop)
  - Self-certainty score (token distribution peakedness)
  - Content signals: did the segment introduce new entities/concepts or just repeat?
- Combine into a scalar "health score"

**Component 3: Branch Decision Engine**
- If health score > threshold: continue generation from current position
- If health score < threshold: BACKTRACK to previous checkpoint
- After backtracking, BRANCH with a perturbation:
  - Option A: Add a soft prefix to the remaining input (our current method applied locally)
  - Option B: Temperature perturbation for the next segment
  - Option C: Inject a "rethink" token or separator to signal the model to change approach
  - Option D: Modify attention mask to reduce attention to the failed segment

**Component 4: Tree Manager**
- Maintains a tree of (checkpoint, segment, health_score) nodes
- Exploration strategy: best-first search (expand the node with highest health score)
- Budget constraint: maximum N total segments across all branches
- Final output: the complete path from root to highest-scoring leaf

```
tree = Tree(root=initial_kv_cache)
budget = N_segments

while budget > 0:
    node = tree.best_unexpanded_node()
    segment = generate_segment(node.kv_cache, 64 tokens)
    health = monitor.evaluate(segment)
    budget -= 1
    
    if health > threshold:
        tree.add_child(node, segment, health)
    else:
        # Branch: try K different perturbations from parent
        for perturbation in [prefix, temperature, rethink_token]:
            alt_segment = generate_segment(node.parent.kv_cache, 64, perturbation)
            alt_health = monitor.evaluate(alt_segment)
            tree.add_child(node.parent, alt_segment, alt_health)
            budget -= 1

return tree.best_complete_path()
```

### Interface Contracts
- Segment generator: input = (kv_cache, max_tokens, optional_perturbation), output = (token_ids, kv_cache_new, logprobs)
- Quality monitor: input = (token_ids, logprobs), output = health_score (0-1)
- Branch decision: input = health_score, output = continue|backtrack
- Tree manager: input = (node, segment, health), output = next_node_to_expand

### Failure Modes
1. **KV cache memory explosion**: Each checkpoint stores the full KV cache. With K branches × D depth, memory grows as O(K*D*seq_len*embed_dim). Mitigation: limit tree depth, prune low-scoring branches.
2. **Quality monitor fooled**: Model produces fluent but wrong reasoning that gets high health scores. The monitor only detects syntactic problems (repetition, incoherence), not semantic errors.
3. **Backtracking doesn't help**: If the model's failure mode is consistent (always makes the same mistake), backtracking to the same checkpoint and trying again just wastes compute.
4. **Budget inefficiency**: Most segments may be fine; budget wasted on monitoring healthy generation.

### Adversarial Audit
- **VRAM constraint**: Qwen3-4B Q4 KV cache at seq_len=1024 is ~2GB per checkpoint. With 10 checkpoints, that's 20GB — nearly fills the 24GB GPU. Must be very aggressive with pruning.
- **The real bottleneck is semantic quality**: All the backtracking in the world can't help if the model simply doesn't know the answer. Backtracking helps with format/structure problems, not knowledge gaps.
- **Comparison with best-of-N**: Checkpoint-and-branch is strictly more complex than best-of-N full generation. It's only better if early detection saves compute (cheaper than generating N full outputs). With N=10 and 64-token segments, that's 10×(1024/64) = 160 segment evaluations vs 10 full generations. The overhead may not pay for itself.
- **Tree search already exists**: MCTS for LLMs (rStar, SpecReason) does exactly this but with more sophisticated node evaluation. What makes checkpoint-and-branch novel? Only the perturbation-based branching. But perturbation at mid-generation is untested — our prefix perturbation works at t=0.

### Verdict
Valuable if the quality monitor is good. But the quality monitor IS the hard problem (same as Phase B observer-router). If we solve the monitor, both this architecture and our current one benefit. The tree search adds complexity without solving the core challenge.

---

## Architecture 3: Spectral Reasoning Amplification

### Core Idea
The model's hidden states live in a high-dimensional space. Not all dimensions are equally important for reasoning. Hypothesis: there exists a low-dimensional "reasoning subspace" that encodes the computation-relevant features. By projecting hidden states onto this subspace and AMPLIFYING the projection, we can boost reasoning without corrupting other aspects (fluency, formatting, knowledge).

### System Design

**Component 1: Reasoning Subspace Identifier**
- Collect hidden states from correct vs incorrect reasoning trajectories (same data as ASM)
- Compute the PCA/SVD of the difference: Δh = h_correct - h_incorrect
- The top-k principal components of Δh define the "reasoning subspace"
- This is done ONCE per model, offline

**Component 2: Spectral Amplifier**
- At each generation step (or at specific layers):
  - Project hidden state h onto the reasoning subspace: h_reasoning = P_k * h
  - Amplify: h_amplified = h + α * h_reasoning
  - This boosts the reasoning-relevant components without changing the orthogonal complement

**Component 3: Layer Selection**
- Not all layers benefit from amplification
- Run ablation: amplify at each layer independently, measure effect on accuracy
- Select the optimal layer (or small set of layers)

```
# One-time offline:
reasoning_basis = compute_reasoning_pca(correct_hidden_states, incorrect_hidden_states, k=16)

# At inference:
for each token step t:
    for layer in selected_layers:
        h = model.hidden_states[layer]
        h_reasoning = reasoning_basis @ (reasoning_basis.T @ h)
        h_amplified = h + alpha * h_reasoning
        model.hidden_states[layer] = h_amplified
    token_t = decode(model.hidden_states[-1])
```

### Interface Contracts
- Reasoning subspace: Tensor of shape (k, embed_dim) — the basis vectors
- Amplifier: input = (h, basis, alpha), output = h_amplified
- Layer selector: empirically determined, stored as config

### Why This Might Work
- **Prototype-Based Dynamic Steering (PDS)** already does something similar: clusters activation differences between CoT and neutral prompts, uses the cluster centroids as "reasoning prototypes" to steer. Our approach is more principled (PCA gives optimal linear separation) but the same family.
- **Activation steering literature** confirms that LLM behaviors are roughly linearly encoded in activation space. If "reasoning quality" is linearly encoded, amplification should work.
- **It's complementary to our prefix perturbation**: The prefix diversifies trajectory class; spectral amplification boosts reasoning quality within each trajectory. The two could be combined.

### Failure Modes
1. **Reasoning is not linearly separable in activation space**: If the difference between correct and incorrect trajectories is nonlinear, PCA misses it. The reasoning "direction" may be a manifold, not a line.
2. **Amplification breaks coherence**: Boosting reasoning components may suppress fluency/formatting components if they're not perfectly orthogonal.
3. **Task-specific subspace**: The reasoning subspace for arithmetic may be different from legal reasoning. A universal "reasoning direction" may not exist.
4. **Same data dependency as ASM**: Needs correct vs incorrect trajectories to compute the subspace.

### Adversarial Audit
- **The linearity assumption**: Evidence is mixed. RepE papers show linearity for simple concepts (truthfulness, sentiment). Complex reasoning may not be linear. Without testing, this is a strong assumption.
- **PDS already exists**: Prototype-Based Dynamic Steering is published and does essentially this with cluster prototypes. Our PCA approach may be cleaner but is not novel.
- **Alpha sensitivity**: Too small = no effect. Too large = corruption. The optimal alpha is unknown and may vary per task, layer, and token position. This introduces a hyperparameter search problem.
- **KV cache issue (again)**: Modifying hidden states at intermediate layers during generation requires hooks into the forward pass. If the model uses fused kernels or flash attention, hooks may not be compatible.

### Verdict
Worth trying because it's complementary to our current approach. But the linearity assumption is strong, and PDS already occupies this niche. Novel contribution would be: combine spectral amplification with prefix perturbation to get both diversity AND quality.

---

## Architecture 4: Inverse Speculative Reasoning

### Core Idea
Speculative decoding uses a SMALL model to draft tokens and a LARGE model to verify/correct them. **Inverse Speculative Reasoning** flips this: use a LARGE model (or our oracle/judge) to provide GUIDANCE at key decision points, and the SMALL model to execute the detailed reasoning.

In practice: the small model generates, but at configurable checkpoints (every K tokens), we extract the partial generation, evaluate it with a stronger signal (LLM judge, symbolic checker, or heuristic), and inject a guidance signal back into the small model's generation.

### Why This Is Different From Our Current Approach
- Current: perturb once (at t=0), hope for the best
- This: monitor continuously, intervene at decision points with informed (not random) signals

### System Design

**Component 1: Small Reasoner** (Qwen3-4B Q4, our current model)
- Generates the actual output, token by token
- Provides KV cache snapshots at checkpoints

**Component 2: Quality Oracle** (LLM judge or heuristic)
- Evaluates partial generation at each checkpoint
- Returns: quality score, identified issues, suggested direction
- For arithmetic: "intermediate calculation appears wrong at step 3"
- For legal: "missing consideration of X defense"
- Can be as simple as self-certainty score, or as complex as an LLM judge call

**Component 3: Guidance Injector**
- Translates oracle feedback into an intervention on the small model:
  - Option A: Append the feedback as additional tokens ("Wait, reconsider: ...")
  - Option B: Inject a steering vector derived from the feedback
  - Option C: Modify attention weights to focus on specific earlier tokens
  - Option D: Replace the last K tokens and regenerate with a perturbation

**Component 4: Checkpoint Manager**
- Saves KV cache at each checkpoint
- If guidance is "continue" → proceed from current state
- If guidance is "backtrack" → restore previous KV cache + inject guidance
- If guidance is "abandon" → start fresh with a different prefix (our current method)

```
kv_cache = None
tokens = []

for checkpoint_idx in range(max_checkpoints):
    # Generate K tokens
    new_tokens, kv_cache = small_model.generate(
        K_tokens, kv_cache, greedy=True
    )
    tokens.extend(new_tokens)
    
    # Evaluate
    quality = oracle.evaluate(tokens)
    
    if quality.action == "continue":
        continue
    elif quality.action == "backtrack":
        kv_cache = checkpoints[quality.backtrack_to]
        tokens = tokens[:quality.backtrack_position]
        # Inject guidance
        guidance_tokens = oracle.get_guidance(tokens, quality.issues)
        tokens.extend(guidance_tokens)
        kv_cache = update_kv_cache(kv_cache, guidance_tokens)
    elif quality.action == "abandon":
        # Start fresh with perturbation (our current method)
        kv_cache = None
        tokens = []
        prefix = random_soft_prefix()
```

### Interface Contracts
- Small reasoner: standard generate() with KV cache management
- Quality oracle: input = partial_text, output = {score, action, issues, guidance}
- Guidance injector: input = (kv_cache, guidance), output = modified_kv_cache
- Checkpoint manager: FIFO queue of (position, kv_cache) tuples

### Why This Might Work
- **Speculative Thinking** (2025) shows this approach works: "larger reasoning models guide smaller ones through selective delegation at structurally meaningful points." The key insight: you don't need the large model to generate everything — just to course-correct at key moments.
- **Our oracle already exists**: Our LLM judge (for legal/planning) or extract_answer (for arithmetic) can serve as the quality oracle. We're just moving the evaluation from post-generation to mid-generation.
- **Combines our strengths**: prefix perturbation (for diversity when abandoning) + quality monitoring (for mid-course correction) + oracle selection (for final output)

### Failure Modes
1. **Oracle latency**: If the oracle is an LLM judge, each checkpoint evaluation takes seconds. With 16 checkpoints × K evaluation time, this may be slower than just generating N full outputs and selecting.
2. **Guidance injection corrupts KV cache**: Appending guidance tokens changes the attention context for all subsequent tokens. The model wasn't trained with mid-generation interventions.
3. **Oracle is wrong**: The oracle may "correct" correct reasoning, or fail to detect real errors. False guidance is worse than no guidance.
4. **Checkpoint memory**: Same KV cache memory issue as Architecture 2.

### Adversarial Audit
- **Latency comparison**: With a cheap oracle (self-certainty + logprob), checkpoint evaluation adds ~1ms per checkpoint. Affordable. With LLM judge, adds ~5s per checkpoint. Too slow for most use cases.
- **Is this just MCTS with extra steps?**: Tree search (MCTS) also evaluates partial generations and branches. The difference here is the GUIDANCE injection — not just branching, but providing information to the model about what went wrong. This is genuinely novel if the guidance mechanism works.
- **Our current prefix perturbation is the "abandon" fallback**: This architecture subsumes our current approach as the degenerate case where the oracle always says "abandon."
- **The real innovation is the guidance injector**: How to translate "this reasoning step is wrong" into a signal the model can use. This is unsolved. Appending text works (it's just prompting) but requires re-running the model from that point.

### Verdict
The most practically promising architecture if the guidance injector works. The key experiment: does mid-generation textual feedback ("reconsider step 3") actually improve the subsequent generation? If yes, this is strictly better than our current one-shot approach. If no, we fall back to our prefix perturbation.

---

## Architecture 5: Multi-Surface Resonance Ensemble

### Core Idea
Don't pick ONE intervention surface. Use ALL OF THEM simultaneously, at different scales, and let them RESONATE.

Combine:
- Surface 1: Soft prefix (embedding perturbation at t=0)
- Surface 2: Activation steering (residual stream at optimal layer)
- Surface 3: Attention bias (modify attention to early positions)
- Surface 4: Temperature perturbation (stochastic sampling)
- Surface 5: Post-selection (oracle/routing)

Each surface independently contributes some diversity and quality. The question: does multi-surface intervention produce SUPER-ADDITIVE improvement (resonance) or SUB-ADDITIVE (interference)?

### System Design

**Component 1: Surface Registry**
- Each intervention surface is a plugin with a common interface:
  - `prepare(model, config) → modified_model`
  - `intervene(hidden_state, step, layer) → modified_hidden_state`
  - `post_select(candidates) → selected_output`
- Surfaces can be independently enabled/disabled

**Component 2: Intervention Scheduler**
- Defines WHEN each surface activates:
  - Prefix: at generation start (once)
  - Activation steering: at layers 8-12, every step
  - Attention bias: first 64 tokens only
  - Temperature: tokens 65+ (after prefix effects lock in)
  - Oracle: after generation completes

**Component 3: Interaction Detector**
- After running all combinations on a small pilot set:
  - Measure: does A+B > A + B? (super-additive)
  - Measure: does A+B < min(A, B)? (destructive interference)
  - Build an interaction matrix of all surface pairs

**Component 4: Optimal Combination Selector**
- Based on the interaction matrix, select the combination of surfaces that maximizes oracle coverage
- This is a discrete optimization problem (2^5 = 32 combinations to test)
- Can be solved exhaustively on the pilot set

```
surfaces = [
    PrefixPerturbation(token_count=2, rms=1.0),
    ActivationSteering(layer=10, direction="reasoning", alpha=0.05),
    AttentionBias(positions=[0,1], bias=-2.0),  # reduce sink mass
    TemperatureSampling(temp=0.3),
]

# Generate with all surfaces active
for candidate_idx in range(N):
    model_copy = apply_surfaces(model, surfaces, seed=candidate_idx)
    output = model_copy.generate(input_ids, max_new_tokens=1024)
    candidates.append(output)

# Select best
best = oracle.select(candidates)
```

### Why This Might Work
- **Different surfaces affect different failure modes**: Prefix perturbation helps with trajectory diversity. Activation steering helps with reasoning quality. Attention bias helps with attention sink avoidance. Temperature helps with exploration. No single intervention solves all problems.
- **Cross-domain analogy**: In music, harmony (multiple frequencies resonating) produces richer sound than any single frequency. In our system, multiple interventions at different "frequencies" (layers, positions, embedding dimensions) may produce richer reasoning than any single intervention.
- **Testable**: The 32-combination matrix is small enough to test exhaustively on 25 tasks.

### Failure Modes
1. **Interference dominates**: Multiple interventions may fight each other. Prefix says "go left," steering says "go right," temperature randomizes everything.
2. **Combinatorial explosion for tuning**: Each surface has its own hyperparameters. Combined, the search space is enormous.
3. **Attribution nightmare**: If the combination works, which surface deserves credit? Hard to publish without clean ablations.

### Adversarial Audit
- **Is this just "throw everything at the wall"?**: Without the interaction matrix and principled analysis, yes. The novelty is in the systematic characterization of surface interactions.
- **The simplest version should be tested first**: Just combine prefix perturbation + temperature (two surfaces). If this doesn't beat either alone, adding more surfaces won't help.
- **KV cache and hook compatibility**: Activation steering requires model hooks. Temperature requires do_sample=True. Prefix requires inputs_embeds. These three may not be compatible in HuggingFace generate() simultaneously. Must verify before designing further.

### Verdict
A pragmatic architecture that builds on our existing work. Start with 2-surface combinations (prefix + temperature, prefix + steering, steering + temperature), measure interactions, then decide whether to expand.

---

## Architecture 6: Token-Free Continuous Reasoning (Radical)

### Core Idea
The tokenizer is a bottleneck. Every intermediate reasoning step is forced through a discrete vocabulary of ~150K tokens. What if intermediate reasoning stays in continuous space, and only the FINAL answer is decoded into tokens?

This is what COCONUT does — but COCONUT requires training. Can we achieve something similar WITHOUT training, by exploiting the model's existing continuous representations?

### Hypothesis
The model's hidden state at the last layer already contains a rich continuous representation of "what should come next." Instead of decoding this into a token and re-embedding it, what if we:
1. Take the last-layer hidden state
2. Project it back to input embedding space (via the transposed embedding matrix)
3. Feed it directly as the next input embedding (no tokenization)
4. Repeat for K steps (continuous "thinking")
5. Only then decode to tokens

### System Design

**Component 1: Continuous Step**
```python
for thinking_step in range(K):
    # Forward pass
    hidden = model.forward(inputs_embeds=current_embeds)
    last_hidden = hidden.last_hidden_state[:, -1, :]  # (1, embed_dim)
    
    # Project to embedding space (soft token)
    # Using the transpose of the embedding matrix (tie weights)
    embed_matrix = model.get_input_embeddings().weight  # (vocab_size, embed_dim)
    # Soft token: weighted average of vocabulary embeddings
    logits = last_hidden @ embed_matrix.T  # (1, vocab_size)
    soft_probs = F.softmax(logits / temperature, dim=-1)  # (1, vocab_size)
    soft_embed = soft_probs @ embed_matrix  # (1, embed_dim)
    
    # Append soft embed to input sequence
    current_embeds = torch.cat([current_embeds, soft_embed.unsqueeze(1)], dim=1)
```

**Component 2: Token Decode Phase**
After K continuous steps, switch to standard autoregressive decoding:
```python
# Normal token generation from the enriched context
output = model.generate(
    inputs_embeds=current_embeds,
    max_new_tokens=max_tokens,
    do_sample=False
)
```

### Why This Might Work
- **COCONUT proves the concept**: With training, continuous reasoning outperforms CoT on logical tasks. Without training, we're betting that the model's latent space ALREADY has useful continuous reasoning paths — it just needs to be allowed to traverse them.
- **The softmax → embedding → forward loop is a RECURRENCE**: The model effectively gets K extra "recurrent" processing steps. This is free compute: no new parameters, just repeated forward passes.
- **Information preservation**: Continuous soft tokens carry more information per step than discrete tokens. The soft token can encode multiple possible next tokens simultaneously (like COCONUT's "breadth-first search" property).
- **Compatible with our prefix perturbation**: Apply soft prefix → continuous thinking → token generation. The prefix diversifies the starting point; continuous thinking refines it.

### Failure Modes
1. **The model wasn't trained for this**: COCONUT works because it's trained with continuous inputs. Our model expects discrete token embeddings. Feeding it softmax-weighted averages is out-of-distribution — it may produce garbage.
2. **Degenerate convergence**: The continuous loop may converge to a fixed point (the same soft token repeated). Without the training signal that COCONUT uses, there's no guarantee the recurrence is productive.
3. **Exponential KV cache growth**: Each continuous step adds a position to the KV cache. K=32 thinking steps adds 32 positions, reducing the available context for actual generation.
4. **Computational cost**: Each continuous step is a full forward pass. K=32 steps = 32 forward passes before any tokens are generated.

### Adversarial Audit
- **This has been tried (sort of)**: "Latent Reasoning as Vocabulary-Space Superposition" (arXiv:2510.15522) analyzes how LLMs reason in vocabulary space. The soft token approach is related but different — we're not analyzing, we're RUNNING the recurrence.
- **Without training, this is likely to fail**: The critical insight from COCONUT is that the model needs to be TRAINED for continuous reasoning. Without training, the soft token recurrence has no reason to converge to useful representations.
- **However**: Even if the continuous loop doesn't "reason," it provides additional processing of the input — similar to how test-time training uses gradient steps to adapt. The K forward passes through the model may implicitly compute useful representations even without explicit training for this.
- **Minimum viable test**: Run 25 arithmetic tasks with K=0,1,2,4,8,16 continuous steps. If accuracy monotonically increases (even slightly), the recurrence is doing something useful. If it degrades or plateaus at K=0, the model can't do token-free reasoning without training.

### Verdict
HIGH RISK, HIGH REWARD. If it works without training, it's a breakthrough — continuous reasoning from frozen models. If it doesn't work (likely), we learn that COCONUT-style training is necessary, which still has scientific value. The minimum viable test is cheap (a few hours of GPU time).

---

## Architecture 7: Adversarial Self-Debate

### Core Idea
Generate TWO responses to the same question with different perturbations. Then ask the model (same model, fresh context) to judge which is better and WHY. Use the model's own criticism to either select the better response OR to generate a THIRD response that addresses the criticisms.

### System Design

**Round 1: Generate Candidates**
- Response A: greedy decoding (baseline)
- Response B: 2-token random prefix (our method)
- Response C: temperature=0.6

**Round 2: Self-Debate**
```
Prompt: "Here are two responses to the question '{question}':

Response A: {response_a}
Response B: {response_b}

Which response is better? Identify specific strengths and weaknesses of each.
Then generate an improved response that combines the strengths of both."
```
- Run this with greedy decoding
- The model acts as its own judge AND synthesizer

**Round 3: Quality Verification**
- Compare the Round 2 synthesis against the original Round 1 candidates
- If synthesis is better (by self-certainty or length/coherence metrics): use it
- If not: fall back to oracle selection among Round 1 candidates

### Why This Might Work
- **Self-consistency literature shows models can evaluate their own outputs**: The model has meta-knowledge about what constitutes good reasoning.
- **Synthesis > selection**: Instead of picking the best of N, CREATE a new response that combines insights from multiple candidates. This is a higher-value use of the judge.
- **Self-critique is cheap**: It's just another generation, no special infrastructure.

### Failure Modes
1. **The model can't judge its own outputs reliably**: Self-evaluation may be as unreliable as self-generation. The same reasoning failures that produced bad outputs may also produce bad judgments.
2. **Synthesis degrades quality**: The Round 2 prompt is complex (compare, judge, synthesize). The model may handle each step poorly, producing a worse response than either input.
3. **Cost**: Round 2 requires another full generation pass (1024+ tokens) that's longer than either original response. 3x the compute of a single generation.
4. **For small models, this may not work at all**: Self-debate requires strong meta-cognitive capability. 4B quantized models may not have it.

### Adversarial Audit
- **This is just self-consistency with extra steps**: At its core, this generates multiple candidates and tries to combine them. Self-consistency (majority voting) is simpler and has strong theoretical backing. What makes debate better?
- **The synthesis step is the key differentiator**: If the model can genuinely synthesize insights from multiple candidates into a superior response, that's more powerful than selection. But "synthesis" may just be "paraphrase the longer response."
- **For arithmetic**: The model either gets the answer right or wrong. Synthesis doesn't help — you can't average two wrong answers and get a right one. This architecture is only relevant for open-ended tasks.
- **For legal/planning**: More promising. Different candidates may emphasize different legal arguments or planning considerations. A synthesis that includes all of them could be genuinely better.

### Verdict
Potentially useful for open-ended tasks (legal, planning) but unlikely to help with arithmetic. The synthesis capability of small models is unverified and possibly absent. Start with a simple test: does the model produce better legal responses when shown two alternatives vs. generating from scratch?

---

## Comparison Matrix

| Architecture | Novel? | Requires Training? | Compute Cost | Best Domain | Key Risk |
|---|---|---|---|---|---|
| 1. ASM | Low (published) | Yes (dynamics model) | High (per-step) | Math/Logic | Dynamics assumption |
| 2. Checkpoint-Branch | Medium | No | Medium (KV cache) | All | Quality monitor |
| 3. Spectral Amplify | Low (PDS exists) | Yes (subspace) | Low (per-step) | Math/Logic | Linearity assumption |
| 4. Inverse Speculative | High | No | Medium (oracle) | All | Guidance injection |
| 5. Multi-Surface | Medium | No | Variable | All | Interference |
| 6. Token-Free | High | No | High (K passes) | Unknown | Training needed? |
| 7. Self-Debate | Low (published) | No | 3x generation | Open-ended | Small model capability |

### Top 3 for Immediate Testing (No Training Required)

1. **Architecture 5: Multi-Surface Resonance** — Start with prefix+temperature, measure interaction. Cheapest to test, builds on existing infrastructure.
2. **Architecture 6: Token-Free Continuous Reasoning** — High risk, high reward. The minimum viable test (K=0..16 continuous steps on arithmetic) takes hours, not days.
3. **Architecture 4: Inverse Speculative** — Most practically promising if guidance works. Test with cheap oracle (self-certainty) first, then LLM judge.

### Top 2 for Maximum Novelty (Requires Some Setup)

1. **Architecture 6: Token-Free Continuous Reasoning** — If it works without training, it's a genuine breakthrough. No one has shown this.
2. **Architecture 4: Inverse Speculative** — The guidance injection mechanism (translating "this step is wrong" into a model-usable signal) is unsolved and important.
