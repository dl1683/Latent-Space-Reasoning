# Missing Architectures — Codex-Identified + Paradigm Breaks

## Status: DESIGN PHASE — Codex Review Required

## Origin
Codex Wave 1 review identified 5 architectures that were MISSING from the original 7. This document designs those plus additional paradigm-breaking architectures that challenge assumptions shared by ALL prior designs.

---

## PART A: Codex-Identified Missing Architectures

---

## Architecture 18: Diversity-Optimized Best-of-N (DO-BoN)

### Codex Description
"Choose prefix/temperature/prompt/random-vocab operators by measured pairwise error correlation, not intuition."

### Core Insight
Standard best-of-N picks N random seeds and generates. But some seeds produce DUPLICATE trajectories (same attractor basin). This wastes budget. DO-BoN ensures every candidate explores a UNIQUE region of output space.

### System Design

**Component 1: Diversity Fingerprinter**
- For each candidate, compute a fast fingerprint during generation:
  - First 16 generated token IDs (sequence fingerprint)
  - Mean logprob of first 16 tokens (confidence fingerprint)
  - Attention entropy at layer 16 on first token (internal state fingerprint)
- Combined fingerprint: (token_hash, logprob_bucket, attn_entropy_bucket)

**Component 2: Duplicate Detector**
- After generating each candidate, compare its fingerprint to all existing candidates
- If fingerprint matches any existing candidate within threshold: REJECT and regenerate with a new seed
- Threshold: token_hash overlap > 75% AND logprob within 10% AND entropy within 10%

**Component 3: Adaptive Budget**
- If candidate is rejected: try a STRONGER perturbation
  - Increase RMS scale for prefix perturbation
  - Increase temperature for sampling
  - Switch to a different operator type
- Keep trying until a genuinely diverse candidate is found, up to 3 retries

**Component 4: Diversity-Maximizing Seed Selection**
- Instead of random seeds 0-9: precompute seeds that maximize inter-candidate diversity
- Offline: generate 100 candidates with seeds 0-99, compute pairwise fingerprint distance
- Select the 10 seeds with maximum minimum pairwise distance (maximin criterion)
- These 10 seeds are the "diversity-optimal" seeds for this task type

```python
class DiversityOptimizedBoN:
    def generate(self, question, N=10, max_retries=3):
        candidates = []
        fingerprints = []
        seed = 0
        
        while len(candidates) < N:
            candidate = self.operator.generate(question, seed=seed)
            fp = self.fingerprint(candidate)
            
            if self.is_duplicate(fp, fingerprints):
                # Retry with stronger perturbation
                for retry in range(max_retries):
                    candidate = self.operator.generate(
                        question, seed=seed, 
                        strength=1.0 + 0.5 * retry  # escalating strength
                    )
                    fp = self.fingerprint(candidate)
                    if not self.is_duplicate(fp, fingerprints):
                        break
                else:
                    seed += 1
                    continue  # skip after max retries
            
            candidates.append(candidate)
            fingerprints.append(fp)
            seed += 1
        
        return candidates
```

### Why This Matters
- Directly addresses the core CDE thesis: candidates are only valuable if they're diverse
- Cheap: fingerprinting costs ~0 compared to generation
- Compatible with ANY operator: prefix, temperature, prompt rephrase
- Provably better than random-seed BoN: every candidate is guaranteed to be unique

### Failure Modes
1. **Fingerprint is too coarse**: Token hash may match even when outputs differ substantially later
2. **All seeds produce similar outputs**: If the model has a very strong attractor, even aggressive perturbation can't escape. Every seed is a "duplicate."
3. **Diversity doesn't equal quality**: Maximum diversity may include more degenerate outputs. Diversity is only useful if some diverse candidates are correct.

### Verdict
MUST-BUILD. This is the cheapest, most direct improvement to our current pipeline. It's not a new architecture — it's a FIX for the existing one. Every other architecture benefits from diversity-optimized candidate generation.

---

## Architecture 19: Early-Trajectory Router (ETR)

### Codex Description
"Generate 16-64 tokens, cluster trajectory class, stop duplicate basins, expand only novel/high-health branches."

### Core Insight
Full generation (1024 tokens) is expensive. By generating only the first 32-64 tokens, we can classify which attractor basin each candidate is heading toward. If two candidates are in the same basin, one is redundant. Stop the duplicate, start a new candidate.

This is SPECULATIVE GENERATION for diversity: generate cheaply, classify, then invest full compute only in diverse candidates.

### System Design

**Component 1: Short-Generation Phase**
- For each of N_initial=20 seeds, generate only first 32 tokens
- Cost: 20 × 32 tokens = 640 tokens (vs 20 × 1024 = 20,480 for full generation)
- Save KV cache for each short generation

**Component 2: Trajectory Classifier**
- Embed the first-32-token outputs (using model's own embedding, or external sentence encoder)
- Cluster into K classes (DBSCAN, K-means, or hierarchical clustering)
- Each class = one trajectory type (correct approach, wrong approach A, wrong approach B, degenerate loop, etc.)

**Component 3: Basin Selector**
- From each unique basin, select the highest-health candidate:
  - Health = mean logprob × (1 - repetition_rate)
  - This filters out degenerate candidates within each basin
- If K basins found and budget allows N=10 full generations:
  - Allocate ceil(10/K) candidates per basin (balanced exploration)
  - OR allocate proportional to basin health (exploitation)

**Component 4: Full-Generation Phase**
- For selected candidates, RESUME generation from the saved KV cache
- Generate remaining 1024-32 = 992 tokens from the checkpoint
- No wasted compute: the first 32 tokens are reused, not regenerated

```python
class EarlyTrajectoryRouter:
    def generate(self, question, N_full=10, N_probe=20, probe_len=32):
        # Phase 1: Short probes
        probes = []
        for seed in range(N_probe):
            short_output, kv_cache = self.generate_short(question, seed, probe_len)
            embedding = self.embed(short_output)
            health = self.health_score(short_output)
            probes.append((seed, short_output, kv_cache, embedding, health))
        
        # Phase 2: Cluster
        embeddings = [p[3] for p in probes]
        clusters = DBSCAN(eps=0.3).fit_predict(embeddings)
        
        # Phase 3: Select diverse, healthy candidates
        selected = []
        for cluster_id in set(clusters):
            cluster_probes = [p for p, c in zip(probes, clusters) if c == cluster_id]
            # Best health in each cluster
            best = max(cluster_probes, key=lambda p: p[4])
            selected.append(best)
        
        # Phase 4: Full generation from checkpoints
        candidates = []
        for seed, short, kv_cache, _, _ in selected[:N_full]:
            full_output = self.generate_full(question, kv_cache, max_tokens=1024-probe_len)
            candidates.append(short + full_output)
        
        return candidates
```

### Why This Matters
- **Compute efficiency**: 20 probes × 32 tokens + 10 full generations × 992 tokens = 10,560 tokens. vs 20 full generations × 1024 tokens = 20,480 tokens. ~50% compute savings for the same diversity.
- **Guaranteed diversity**: Every full-generation candidate is from a different trajectory class
- **Combines with CDE**: The trajectory classifier provides the "K(O,N)" measurement for free

### Failure Modes
1. **32 tokens is not enough to distinguish basins**: Outputs may diverge later. Two candidates starting similarly may end differently.
2. **KV cache management**: Storing 20 KV caches simultaneously requires significant memory. Qwen3-4B Q4 at seq_len=32: ~0.1GB per cache × 20 = 2GB. Manageable on 24GB.
3. **DBSCAN parameter sensitivity**: The clustering eps threshold determines how many basins are found. Too small = every candidate is its own basin (no deduplication). Too large = everything in one basin (no diversity).
4. **Health score correlation with final quality**: A candidate with good first-32-token health may degrade later. Health is a weak predictor of final output quality.

### Adversarial Audit
- **This is our Phase B observer-router**: ETR is exactly what the converged blueprint calls "Component 5: Observer-Router" in Phase B. It's not new — it's already in the roadmap.
- **However**: The blueprint's observer-router uses a trained MI-based routing score. ETR uses a simpler, training-free approach (clustering + health). This is a SIMPLER version that can be tested immediately.
- **The probe length is a hyperparameter**: 32 tokens? 64? 16? Must be tuned per task type. Too short = not enough information. Too long = not enough savings.

### Verdict
HIGH PRACTICAL VALUE. This is the training-free version of our Phase B observer-router. It should be implemented as a stepping stone: if clustering-based routing works, the MI-based routing is justified. If clustering-based routing fails (all candidates cluster together), MI-based routing won't help either.

---

## Architecture 20: Verifier-First Regeneration (VFR)

### Codex Description
"For arithmetic/code, use exact intermediate checks and regenerate only failing spans."

### Core Insight
For problems with verifiable intermediate steps (arithmetic: each operation is checkable; code: each function is testable), we can verify AS WE GENERATE and regenerate only the failing parts. This is surgical error correction rather than full-output regeneration.

### System Design

**Component 1: Step-Level Verifier**
- For arithmetic: parse the chain-of-thought to identify individual operations
  - "7 × 8 = 56" → verify: 7 × 8 = 56? YES
  - "56 + 36 = 82" → verify: 56 + 36 = 82? NO (should be 92)
- For code: execute each function/line and check for errors
- Returns: list of (step, result, verified: bool)

**Component 2: Error Localizer**
- Identifies the FIRST incorrect step in the chain
- Everything before this step is correct and can be kept
- Everything after (including the incorrect step) must be regenerated

**Component 3: Targeted Regeneration**
- Keep the correct prefix (question + correct steps)
- Regenerate from the point of failure with a different perturbation
- The model sees its own correct intermediate work as context
```
Prompt: "What is (7 × 8) + (12 × 3)?

Step 1: 7 × 8 = 56 ✓
Step 2: 12 × 3 = 36 ✓  
Step 3: 56 + 36 = "
[regenerate from here with perturbation]
```

**Component 4: Iterative Verification Loop**
```python
class VerifierFirstRegeneration:
    def solve(self, question, max_retries=5):
        output = self.generate(question)  # initial generation
        
        for retry in range(max_retries):
            steps = self.parse_steps(output)
            verification = self.verify_steps(steps)
            
            first_error = self.find_first_error(verification)
            if first_error is None:
                return output  # all steps verified ✓
            
            # Keep correct prefix, regenerate from error
            correct_prefix = self.text_up_to_step(output, first_error)
            output = self.generate(
                question, 
                prefix_text=correct_prefix,
                perturbation_seed=retry
            )
        
        return output  # best effort after max retries
```

### Why This Matters
- **Surgical precision**: Instead of generating 10 full outputs hoping one is correct, fix the specific error. Compute goes exactly where it's needed.
- **Correct work is preserved**: The model's correct intermediate steps are kept. Only the error is regenerated. This is MUCH more efficient than starting over.
- **Verification is exact**: For arithmetic, Python can verify each step perfectly. No selector uncertainty.
- **Naturally integrates with CDE**: VFR provides the exact verifier that CDE's selector stack needs for Pillar 2 (Quality Assessment).

### Failure Modes
1. **Parsing is hard**: Chain-of-thought outputs aren't always structured. The model may not produce verifiable intermediate steps.
2. **Error propagation backward**: Sometimes the "error" at step N is caused by a wrong APPROACH (not a wrong calculation). The approach was decided at step 1, and regenerating from step N doesn't fix the root cause.
3. **Not all tasks are verifiable**: Legal reasoning, planning, creative tasks have no intermediate verifiers. This architecture is domain-restricted.
4. **Regeneration from mid-point**: The model wasn't trained to continue from arbitrary mid-generation points with different perturbation. The continuation may be incoherent with the prefix.

### Adversarial Audit
- **Step-by-step verification is well-studied**: Process reward models (PRM), verifier-guided decoding, and math verification tools all do this. Not novel.
- **Our twist**: Using PREFIX PERTURBATION to diversify the regeneration from the error point. This is novel — it combines our intervention with targeted error correction.
- **For arithmetic specifically**: This architecture is OPTIMAL. It reduces the problem from "generate a full correct chain" to "generate each step correctly." Since each step is simple (single operation), the model's accuracy per step is high (~90%), and perturbation + retry makes it near-certain.
- **Combine with Architecture 9 (DDC)**: Decompose the problem, verify each sub-answer, regenerate failures. This is the most powerful arithmetic-specific pipeline.

### Verdict
EXTREMELY HIGH VALUE for arithmetic and code. This should be the default architecture for any task with verifiable intermediate steps. Not applicable to open-ended tasks, but for our arithmetic evaluation, it would likely push accuracy near 100%. The question is whether this is "too good" — if it trivially solves arithmetic, we can't use arithmetic as a meaningful benchmark anymore.

---

## Architecture 21: Diversity-Regularized Candidate Generation (DRCG)

### Codex Description
"Reject candidates whose first-32-token or hidden-state signature duplicates prior candidates."

### Core Insight
This is the online version of Architecture 18 (DO-BoN). Instead of precomputing diversity-optimal seeds, ENFORCE diversity during generation by rejecting duplicates in real-time.

### System Design

**Component 1: Rolling Signature Database**
- Maintain a database of signatures for all generated candidates so far
- Signature = embedding of first 32 tokens (fast to compute from KV cache)

**Component 2: Diversity Gate**
- After generating first 32 tokens of a new candidate:
  - Compute its signature
  - Compare against all signatures in the database
  - If max_cosine_similarity > threshold: ABORT generation (don't waste 992 more tokens)
  - If diverse: continue generation to completion
  - Add completed candidate's signature to the database

**Component 3: Escalating Perturbation**
- If a candidate is rejected:
  - Increment perturbation strength: rms *= 1.5
  - Try a different operator type (switch from prefix to temperature)
  - Try a completely different seed range
- Continue until a diverse candidate is found or max_attempts exceeded

```python
class DiversityRegularizedGeneration:
    def generate(self, question, N=10, similarity_threshold=0.8):
        candidates = []
        signatures = []
        seed = 0
        abort_count = 0
        
        while len(candidates) < N:
            # Generate first 32 tokens
            partial, kv_cache = self.generate_partial(question, seed, tokens=32)
            sig = self.compute_signature(partial)
            
            # Diversity check
            if signatures and max(cosine_sim(sig, s) for s in signatures) > similarity_threshold:
                abort_count += 1
                seed += 1
                # Escalate: after 3 aborts, switch operator
                if abort_count > 3:
                    self.switch_operator()
                    abort_count = 0
                continue
            
            # Diverse! Complete generation
            full_output = self.complete_generation(kv_cache, max_tokens=992)
            candidates.append(partial + full_output)
            signatures.append(sig)
            abort_count = 0
            seed += 1
        
        return candidates
```

### Why This Matters
- **Zero wasted compute on duplicates**: Every full generation is guaranteed diverse
- **Adaptive**: Automatically escalates perturbation when the model resists diversity
- **Combines DO-BoN (Architecture 18) with ETR (Architecture 19)**: Uses early abort from ETR + diversity maximization from DO-BoN
- **The 32-token abort is cheap**: Only 32/1024 = 3% of a full generation. Aborting 5 candidates costs 5 × 3% = 15% overhead.

### Failure Modes
1. **The model is so deterministic that escalation fails**: Even with maximum perturbation, all candidates converge to the same trajectory. In this case, diversity-regularization correctly identifies that the model can't produce diverse outputs for this input.
2. **Threshold tuning**: Too low threshold = too many aborts, slow generation. Too high = accepts near-duplicates, wastes budget.
3. **32-token signatures miss late-diverging outputs**: Candidates that start similarly but end differently are incorrectly flagged as duplicates.

### Verdict
ESSENTIAL COMPONENT, not a standalone architecture. This should be INTEGRATED into the CDE framework as a generation-time optimization. Every CDE operator should use diversity-regularized generation.

---

## Architecture 22: Gated-Attention Transfer Battery

### Codex Description
"Not an architecture, but mandatory. If gated attention kills the effect, attention-sink rescue is not a durable mechanism."

This is already designed in `gated_attention_probe_design.md`. Codex is reiterating its importance as a GATE for the entire research direction. Not designing further here — it's ready for implementation.

---

## PART B: Paradigm-Breaking Architectures

The following architectures challenge assumptions shared by ALL 22 designs above.

---

## Architecture 23: Diffusion-Based Text Refinement (DBTR)

### Shared Assumption Being Challenged
All previous architectures assume AUTOREGRESSIVE generation: tokens produced left-to-right, one at a time, irreversibly. What if we treat text as an IMAGE and apply DIFFUSION?

### Core Insight
Diffusion models don't generate left-to-right. They start with noise and iteratively denoise to a clean output. Applied to text: start with a RANDOM token sequence and iteratively "denoise" it toward a coherent, correct answer.

### Why This Is Different
- Autoregressive: token i determines token i+1. Early errors are permanent.
- Diffusion: ALL tokens are refined simultaneously. Early "errors" (noisy tokens) are corrected in later denoising steps.

### System Design

**Component 1: Text-to-Embedding Noiser**
- Take the model's embedding of the question
- Append N random embedding vectors (noise tokens) as the "noisy answer"
- These noise tokens have the same dimension as real token embeddings

**Component 2: Iterative Denoising Loop**
- For T denoising steps:
  1. Forward pass: feed [question_embeddings, current_noisy_answer] to the model
  2. The model produces logits for each position
  3. At each position: BLEND the current noisy embedding with the model's predicted embedding
     - denoised_embed[i] = (1-α_t) × current[i] + α_t × model_prediction[i]
     - α_t increases from 0 to 1 over T steps (anneal from pure noise to pure model prediction)
  4. current = denoised_embed

**Component 3: Final Decode**
- After T steps, the "answer" embeddings are close to real token embeddings
- Decode each position by finding the nearest real token embedding
- Output: the decoded token sequence

```python
class DiffusionTextRefinement:
    def generate(self, question_embeds, answer_length=128, T_steps=50):
        # Initialize with noise
        noise = torch.randn(1, answer_length, embed_dim) * embed_rms
        current = noise
        
        for t in range(T_steps):
            alpha = t / T_steps  # 0 → 1
            
            # Forward pass with current noisy answer
            full_input = torch.cat([question_embeds, current], dim=1)
            logits = model(inputs_embeds=full_input).logits
            
            # Model's prediction for answer positions
            answer_logits = logits[:, question_embeds.shape[1]:, :]
            predicted_probs = F.softmax(answer_logits, dim=-1)
            predicted_embeds = predicted_probs @ model.embed_tokens.weight  # soft embedding
            
            # Blend: noise → model prediction
            current = (1 - alpha) * current + alpha * predicted_embeds
        
        # Decode final embeddings to tokens
        token_ids = (current @ model.embed_tokens.weight.T).argmax(dim=-1)
        return token_ids
```

### Why This Might Work
- **Parallel refinement**: All positions are refined simultaneously. The model can "see" the rough shape of the answer and refine globally. This allows non-local corrections that autoregressive generation can't make.
- **No error propagation**: Unlike autoregressive generation where token 10's error corrupts tokens 11-1024, diffusion allows the model to correct ANY position at ANY step.
- **COCONUT analogy**: COCONUT feeds continuous embeddings back as input. DBTR does the same but in a denoising (noise→signal) direction rather than autoregressive (left→right).
- **Perturbation is BUILT IN**: The initial noise IS the perturbation. Different random noise = different denoising path = different final answer. Diversity comes for free.

### Failure Modes
1. **The model wasn't trained for this**: Transformers are trained for autoregressive prediction. A position that "sees" noisy future tokens is completely OOD. The model's attention and MLP computations may produce garbage.
2. **No causal mask**: The diffusion approach requires BIDIRECTIONAL attention (each position needs to see all others, including future positions). Decoder-only models use CAUSAL attention. Must either (a) use a special attention mask or (b) accept that future positions can't influence earlier ones.
3. **Answer length must be pre-specified**: Unlike autoregressive generation (variable length), diffusion needs a fixed output length. Must either pad or use a length predictor.
4. **Computational cost**: T×forward_pass for the denoising loop. With T=50 and answer_length=128, that's 50 forward passes with seq_len=(question + 128). Expensive but bounded.

### Adversarial Audit
- **Diffusion for text exists**: MDLM, SEDD, Plaid, and other discrete diffusion models are active research. But they require TRAINING a diffusion model. We're trying to use a PRETRAINED autoregressive model as a denoiser. This is fundamentally different and likely to fail.
- **The causal mask problem is severe**: Decoder-only models CANNOT attend to future positions. Each "answer" position can only see earlier answer positions. This means the "parallel refinement" benefit is limited to left-to-right influence, which is almost autoregressive again.
- **However**: If we use the model's ENCODER (for encoder-decoder models) or modify the attention mask, bidirectional attention is possible. With decoder-only models, a workaround: process each position independently (each sees the question + noise at its position only, not other answer positions). This is PARALLEL but loses inter-position coherence.
- **Minimum viable test**: Initialize with random embeddings, run 10 denoising steps, decode. If the output is more coherent than random text (even slightly), the denoising does SOMETHING. If it's pure garbage, the approach fails immediately.

### Verdict
HIGH RISK, MEDIUM REWARD. The causal mask issue severely limits the approach for decoder-only models. Worth a quick test (2-3 hours) to see if iterative denoising produces anything coherent. If it does, iterate. If not, move on. This is a SCIENTIFIC PROBE, not a main-line architecture.

---

## Architecture 24: Formal Verification Integration (FVI)

### Shared Assumption Being Challenged
All previous architectures evaluate outputs using HEURISTICS (string matching, confidence, clustering). None use PROVABLY CORRECT verification.

### Core Insight
For many problems, the ANSWER can be verified formally even if the REASONING can't. 
- Arithmetic: compute the answer and check
- Logic: evaluate the truth table
- Code: execute and check output
- Constraints: check all constraints are satisfied
- Math proofs: verify each step with a proof checker

### How FVI Differs From VFR (Architecture 20)
- VFR: Verifies intermediate steps and REGENERATES failures
- FVI: Verifies the FINAL answer formally and uses verification as the SELECTOR

### System Design

**Component 1: Problem Type Classifier**
- Categorize the problem: arithmetic, logical, computational, constraint, open-ended
- For each type, select the appropriate formal verifier

**Component 2: Formal Verifier Registry**
```python
verifiers = {
    'arithmetic': ArithmeticVerifier(),   # Python eval
    'logical': LogicalVerifier(),          # Truth table or SAT solver
    'code': CodeVerifier(),                # Execute and test
    'constraint': ConstraintVerifier(),    # Z3 or OR-tools
    'math': ProofVerifier(),               # Lean4 or Isabelle
    'open_ended': None,                    # No formal verifier available
}
```

**Component 3: Extract-and-Verify Pipeline**
- Generate N candidates
- For each candidate:
  1. Extract the answer (parse from text)
  2. Formalize the answer (convert to verifiable representation)
  3. Run formal verifier
  4. Label: verified_correct, verified_incorrect, unverifiable
- Select from verified_correct candidates (or fall back to heuristic selector if none verified)

**Component 4: Verification-Guided Generation**
- If no candidate passes formal verification after N attempts:
  - Extract the closest-to-correct candidate
  - Identify which verification conditions it fails
  - Feed back to the model: "Your answer fails condition X. Try again."
  - This is VFR (Architecture 20) but with formal verification instead of heuristic checking

### Why This Matters for CDE
- FVI provides the PERFECT selector for Pillar 2 (Quality Assessment)
- Eliminates the selector ceiling problem for verifiable tasks
- Selector accuracy = 1.0 for correctly formalized problems
- This makes CDE's diversity optimization the ONLY remaining bottleneck

### Failure Modes
1. **Formalization gap**: The model's text output may not be parseable into a formal representation. "The answer is about fifty-six" can't be formally verified.
2. **Partial verification**: Many problems are only partially verifiable. You can verify the arithmetic but not the problem setup.
3. **Not applicable to open-ended tasks**: Legal reasoning, planning, creative tasks have no formal verifiers.
4. **Over-fitting to verifiable tasks**: If we benchmark on verifiable tasks (arithmetic) + formal verifier, we get 100% accuracy trivially. The interesting question is: does CDE + FVI help with HARDER problems that are still verifiable but the model rarely solves?

### Verdict
NOT an architecture — it's a SELECTOR. It belongs in the CDE selector stack as the highest-priority selector for verifiable tasks. Don't design it as a separate architecture; integrate it into CDE's Component 3 (SelectorStack).

---

## Architecture 25: Inverse Problem Formulation (IPF)

### Shared Assumption Being Challenged
All architectures START with a question and GENERATE an answer. IPF inverts this: START with the properties of a good answer and SEARCH for inputs that produce it.

### Core Insight
We know properties of a good answer:
- For arithmetic: the answer equals the correct computation
- For legal: the answer addresses all relevant legal elements
- For planning: the answer covers all constraints

Instead of asking "what answer does the model produce?", ask "what INPUT makes the model produce an answer with these properties?"

### System Design

**Component 1: Property Specifier**
- Define the desired output properties formally:
  - Arithmetic: output contains "= 56" (the correct answer)
  - Legal: output contains keywords {"duty of care", "breach", "causation", "damages"}
  - Planning: output mentions all constraint categories

**Component 2: Input Optimizer**
- Search over input space (prefix embeddings) to maximize P(output has desired properties)
- This is Architecture 8 (GGIO) but with a PROPERTY-BASED objective rather than a single-target objective

```python
# Define property checker
def has_property(output_text):
    return "56" in output_text  # for 7×8 = 56

# Optimize prefix to maximize probability of property
prefix = optimize_prefix(
    model, question,
    objective=lambda logits: probability_of_generating_text_containing("56", logits)
)
```

**Component 3: Constraint Satisfaction Search**
- For problems with MULTIPLE required properties:
  - Search for prefixes that satisfy ALL properties simultaneously
  - This is a constraint satisfaction problem in prefix embedding space
  - Use multi-objective optimization or Lagrangian relaxation

### Why This Might Work
- **The model knows the answer**: Often the model HAS the knowledge but fails to access it via the default greedy path. IPF searches for an input path that accesses the right knowledge.
- **Combines with our mechanism insight**: Our prefix perturbation works because different prefixes access different knowledge. IPF searches for the prefix that accesses the CORRECT knowledge.
- **Principled version of random search**: Our current approach randomly perturbs and hopes to find a good prefix. IPF optimizes for it directly.

### Failure Modes
1. **Circular**: To know the desired properties, you often need to know the answer. For arithmetic, you need to know "56" to search for it. This defeats the purpose.
2. **Probability estimation is hard**: Computing P(output contains "56") requires marginalizing over all possible output sequences. Intractable without approximation.
3. **For open-ended tasks**: Desired properties are vague and numerous. Can't formalize "good legal reasoning" into a searchable objective.

### Adversarial Audit
- **This is literally cheating for arithmetic**: If you know the answer is 56, you don't need the model. The architecture only works for tasks where you DON'T know the answer but DO know properties of the answer. This is a narrow niche.
- **For legal reasoning**: Properties like "mentions all legal elements" are verifiable without knowing the correct analysis. This is a legitimate use case.
- **For planning**: Properties like "addresses all constraints" are checkable. Another legitimate use case.

### Verdict
NICHE but interesting for property-verifiable tasks. Combines with Architecture 8 (GGIO) and Architecture 24 (FVI). Not a standalone architecture — it's a SEARCH STRATEGY within the CDE framework.

---

## Architecture 26: Consciousness Simulation via Looped Attention (CSLA)

### Shared Assumption Being Challenged
All architectures treat generation as a SINGLE PASS through the model. CSLA challenges this: what if the model needs to "think" about its own thoughts, creating a LOOP where output attention patterns feed back into input?

### Core Insight
Human reasoning involves recurrent processing — the prefrontal cortex repeatedly re-processes information, with each pass deepening understanding. Transformers are feed-forward — each token gets ONE pass through the layers. What if we create RECURRENCE by routing the model's attention patterns back as input features?

### This Is Different From Architecture 6 (Token-Free)
- Architecture 6: feeds hidden states back as soft token embeddings
- CSLA: feeds ATTENTION PATTERNS (which positions attended to which) back as structured input

### System Design

**Component 1: Attention Pattern Extractor**
- After a forward pass, extract the attention matrices at key layers
- Compute: which positions received the most aggregate attention (attention sinks, key positions)
- Compute: attention entropy per head (focused vs distributed)

**Component 2: Attention-to-Embedding Translator**
- Convert the attention pattern summary into an embedding that encodes "where the model was looking"
- Simple version: average the embeddings of the top-K most-attended-to positions
- Complex version: learned projection from attention matrix to embedding space (requires training)

**Component 3: Recurrent Forward Pass**
- Prepend the attention-derived embedding as a "reflection token"
- Run forward pass again: model now processes (reflection_token + original_question + previous_output)
- The reflection token tells the model "you were paying attention to these positions"
- This may cause the model to shift attention to DIFFERENT positions on the second pass

```python
class ConsciousnessLoop:
    def generate_with_reflection(self, question, K_loops=3):
        current_input = question_embeds
        
        for loop in range(K_loops):
            # Forward pass
            outputs = model(inputs_embeds=current_input, output_attentions=True)
            attention = outputs.attentions  # list of (batch, heads, seq, seq) per layer
            
            # Extract attention summary
            avg_attention = mean_over_heads_and_layers(attention)  # (seq, seq)
            top_positions = avg_attention.sum(dim=0).topk(k=5).indices
            
            # Create reflection embedding
            reflection = model.embed_tokens.weight[0:1] * 0  # zero init
            for pos in top_positions:
                reflection += current_input[:, pos, :]
            reflection /= len(top_positions)
            
            # Prepend reflection and re-run
            current_input = torch.cat([reflection.unsqueeze(1), current_input], dim=1)
        
        # Final generation from enriched context
        output = model.generate(inputs_embeds=current_input, max_new_tokens=1024)
        return output
```

### Why This Might Work
- **Global Workspace Theory**: The dominant theory of consciousness proposes a "global workspace" where information is broadcast and integrated. CSLA simulates this: the reflection token broadcasts what the model attended to, making it available for re-processing.
- **Recurrence helps reasoning**: RNNs can solve problems that feed-forward networks can't (because of recurrence). Adding loops to a transformer may enable similar benefits.
- **Compatible with our prefix perturbation**: Each loop iteration adds a new token to the context. Different initial perturbations + K loops = K different "reflection" paths.

### Failure Modes
1. **Attention extraction is expensive**: Getting full attention matrices requires disabling flash attention. ~2-3x slower.
2. **Reflection token may be meaningless**: The model wasn't trained with "reflection tokens." It may ignore or misinterpret them.
3. **Context grows each loop**: K=3 loops adds 3 tokens. Manageable. But each loop is a full forward pass + attention extraction.
4. **No gradient signal**: Unlike Architecture 8 (GGIO), there's no optimization. The reflection is heuristic. It may not converge to useful representations.

### Adversarial Audit
- **This is just prepending extra context**: The "reflection token" is just an average of highly-attended embeddings. It's additional context, not true recurrence. The model processes it as just another prefix token.
- **No evidence this helps**: Recurrent transformers (RWKV, Mamba) exist but are TRAINED for recurrence. Adding loop-like behavior to a non-recurrent model is architecturally unsound.
- **The attention extraction overhead may dominate**: If 3 loops cost 3× and the benefit is marginal, the compute is better spent on 3× more parallel candidates.

### Verdict
SPECULATIVE SCIENCE. Not expected to work in practice, but the experiment would be informative about whether frozen transformers can benefit from attention-derived recurrence. Low priority — only worth testing if Architecture 6 (Token-Free) shows positive results first (since both involve feeding internal representations back as input).

---

## Architecture 27: Constraint Programming Hybrid (CPH)

### Shared Assumption Being Challenged
All architectures let the LLM drive the generation process. CPH inverts control: a CONSTRAINT SOLVER drives the process, and the LLM is a COMPONENT that provides soft constraints.

### Core Insight
Many reasoning tasks are constraint satisfaction problems in disguise:
- Arithmetic: "What is 7×8+12×3?" → Constraints: result = 7×8+12×3, each operation follows arithmetic rules
- Legal: "Analyze liability" → Constraints: consider all legal elements, apply relevant law, address each party
- Planning: "Schedule these tasks" → Constraints: resource limits, deadlines, dependencies

Instead of asking the LLM to satisfy all constraints simultaneously (hard), extract constraints and let a solver handle the logic while the LLM handles the language.

### System Design

**Component 1: Constraint Extractor**
- Use the LLM to identify constraints from the problem:
```
"List all constraints that the answer to this problem must satisfy:
Problem: What is the maximum weight a shelf can hold if..."

Constraints:
1. Weight must be a positive number
2. Must account for material strength
3. Must include safety factor
4. Must use consistent units
..."
```
- Apply perturbation: different seeds extract different constraints (diversity in constraint identification)

**Component 2: Constraint Formalizer**
- Convert natural-language constraints to formal constraints:
  - Arithmetic: Python expressions
  - Logic: propositional/first-order logic (Z3 format)
  - Scheduling: OR-Tools constraints
- Use the LLM for this translation (with perturbation for diversity)

**Component 3: Constraint Solver**
- Feed formalized constraints to an appropriate solver:
  - Z3 for logical constraints
  - OR-Tools for optimization constraints
  - SymPy for mathematical constraints
- Solver returns: solution (if satisfiable) or UNSAT (if constraints are contradictory)

**Component 4: Solution Verbalizer**
- Use the LLM to convert the formal solution back to natural language
- The LLM only needs to DESCRIBE a known-correct solution, not DERIVE it

**Component 5: Constraint Consistency Checker**
- If different perturbation seeds extract different constraints:
  - Take the UNION of all constraints
  - Check consistency (Z3 SAT check)
  - If inconsistent: identify which constraints conflict (using UNSAT core)
  - The conflicting constraints come from LLM confabulation — remove them

### Why This Might Work
- **Separates language understanding from reasoning**: LLM does what it's good at (understanding the problem, extracting constraints, verbalizing solutions). Solver does what IT'S good at (satisfying constraints exactly).
- **Perturbation diversifies CONSTRAINT EXTRACTION**: Different perturbation seeds may cause the LLM to identify different constraints. The union provides more complete coverage.
- **Formal verification built in**: The solver GUARANTEES the solution satisfies all constraints. No selector uncertainty for the formal components.
- **Scales with problem complexity**: Adding more constraints makes the solver work harder but doesn't confuse it (unlike the LLM, which degrades with problem complexity).

### Failure Modes
1. **Constraint extraction is the hard part**: If the LLM can't correctly identify constraints, the solver gets wrong inputs and produces wrong outputs.
2. **Formalization gap**: Natural language constraints don't always map to formal representations cleanly. "Should be reasonable" can't be formalized.
3. **Solver limitations**: SAT/SMT solvers handle propositional/first-order logic well. They don't handle probabilistic reasoning, analogical reasoning, or common-sense reasoning.
4. **Not all problems are constraint problems**: Creative tasks, narrative generation, explanation — these don't have clean constraint formulations.

### Adversarial Audit
- **This is a well-studied paradigm**: Neuro-symbolic AI, program synthesis, constraint-guided generation. Not novel as a concept.
- **The perturbation-diverse constraint extraction IS novel**: Using our prefix perturbation to extract diverse constraint sets, then taking the union, is not studied. This is a legitimate new combination.
- **For arithmetic**: Over-engineered. Just compute the answer directly (Architecture 14: Neuro-Symbolic). No need for a constraint solver.
- **For planning/scheduling**: HIGHLY relevant. Planning IS constraint satisfaction. The LLM identifies constraints; OR-Tools solves.
- **For legal reasoning**: Partially applicable. Legal analysis has identifiable elements (duty, breach, causation, damages) that can be formalized as requirements. But the ANALYSIS of each element is qualitative, not formally solvable.

### Verdict
DOMAIN-APPROPRIATE for planning and scheduling. Over-engineered for arithmetic (just use Python execution). Partially applicable to legal (constraint extraction, not constraint solving). The novel contribution is perturbation-diverse constraint extraction — worth testing.

---

## PART C: Meta-Synthesis Across All 27 Architectures

### The Full Catalog

| # | Name | Wave | Surface | Novel? | Training? | Best Domain |
|---|---|---|---|---|---|---|
| 1 | ASM | 1 | Hidden states | Low | Yes | Math |
| 2 | Checkpoint-Branch | 1 | Generation | Med | No | All |
| 3 | Spectral Amplification | 1 | Hidden states | Low | Yes | Math |
| 4 | Inverse Speculative | 1 | Generation | High | No | All |
| 5 | Multi-Surface Ensemble | 1 | All | Med | No | All |
| 6 | Token-Free Continuous | 1 | Input/Hidden | High | No | Unknown |
| 7 | Self-Debate | 1 | Multi-agent | Low | No | Open-ended |
| 8 | GGIO | 2 | Input embedding | Med | No | Math |
| 9 | DDC | 2 | Prompt | Low | No | Math |
| 10 | RARS | 2 | Prompt | Med | No | Legal |
| 11 | CIR | 2 | Hidden states | High | No* | All |
| 12 | IDR | 2 | Multi-agent | High | No | All |
| 13 | ALM | 2 | Input embedding | High | No | All |
| 14 | Neuro-Symbolic | 2 | Symbolic | Med | No | Math |
| 15 | EBRL | 2 | Selection | Med | Yes | All |
| 16 | Attention Surgery | 2 | Attention | Med | No | All |
| 17 | TER | 2 | Multi-agent | Med | No | All |
| 18 | DO-BoN | 3 | Generation | Low | No | All |
| 19 | ETR | 3 | Generation | Med | No | All |
| 20 | VFR | 3 | Symbolic | Med | No | Math |
| 21 | DRCG | 3 | Generation | Low | No | All |
| 22 | Gated Attn Probe | 3 | (Experiment) | N/A | No | N/A |
| 23 | DBTR | 3 | All | High | No | Unknown |
| 24 | FVI | 3 | Selection | Low | No | Math |
| 25 | IPF | 3 | Input | Med | No | Legal |
| 26 | CSLA | 3 | Attention/Hidden | Med | No | Unknown |
| 27 | CPH | 3 | Symbolic | Med | No | Planning |

### What I've Systematically Avoided

Looking at the full catalog, I notice these gaps:

1. **No MULTI-MODEL architectures**: All 27 use a SINGLE model. What about using 2-3 DIFFERENT small models (Qwen3-4B + Phi-2 + Gemma-2B) as an ensemble? Different models have different failure modes → naturally decorrelated.

2. **No TRAINING-TIME architectures**: The constraint is "no training of the TARGET model." But what about training a tiny auxiliary model (1M params) specifically for routing/selection? Architecture 15 (EBRL) touches this but deserves deeper exploration.

3. **No COMPRESSION architectures**: None of the designs compress or distill the model's reasoning. What if we first generate a LONG reasoning chain, then ask the model to COMPRESS it to just the answer? The compression step may correct errors.

4. **No SOCIAL/CULTURAL architectures**: None model multi-perspective reasoning. What if different perturbation seeds are given different "personas" or "roles"? Perturbation + role-prompting for structured diversity.

5. **No MEMORY architectures**: None give the model memory across problems. What if solving problem 1 correctly provides context that helps solve problem 2? This is in-context learning over a problem set.

These gaps suggest 5 more architectures (28-32) which I'll design in the next wave.
