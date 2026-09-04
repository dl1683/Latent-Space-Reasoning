# Gate 3: Causal Relay-State Composition

Status: DRAFT — pending Gate 2 results and Codex design gate review.
Requires: GPU access (activation extraction + transplantation).

## What this proves that Gate 2 cannot

Gate 2 (noncommutativity) proves an obstruction: the monoid action is not
commutative. But noncommutativity is cheap — any sequential transducer has
a noncommutative action. Gate 3 proves the NATIVE claim: the model discovers
algebra that is representative-independent, compositional, and causally
realized in its internal computation.

## Core idea

If the model compiles suffix operations into internal relay states at
specific layer-position cuts, then:

1. These relay states form the objects of a causal process category
2. Composition of relay states predicts behavior of unseen suffix sequences
3. Transplanting relay states between contexts causally controls output
4. The relay states are presentation-invariant (different surfaces for the
   same role produce equivalent relay states)

## Experimental design (6 steps from Codex Architecture Theorist)

### Step 1: Nonce-coded surfaces

Assign multiple surfaces per role to random nonce codes so the physical
tokens cannot predict role identity. Example: "# banana = 7.\n" might
encode ASSERT while "# turnip = 3.\n" encodes MISLEADING. The model must
learn the role from the code, not from familiar syntax.

**Problem**: Our model (Qwen3-1.7B-Base) is a pretrained base model used
in zero-shot. It cannot learn nonce codes. Workaround: use semantically
diverse surfaces (already validated in Gate 1b) and verify relay-state
equivalence across surfaces within a role (Step 2).

### Step 2: Define relay-state equivalence at a layer cut

Pick a layer position (e.g., layer 14 of 28, the midpoint) and extract
the hidden state h_{l,p} at the last token position of the suffix.

For two surfaces u₁, u₂ in the same role, compare h_{l,p}(u₁) and
h_{l,p}(u₂) using cosine similarity and projection onto the (C,L,R)
prediction subspace. If they cluster by role (not by surface), the model
has compiled the semantic role into a relay state.

**Measurement**: For each (context, role), collect h_{l,p} across surfaces.
Compute within-role vs between-role distance. If within-role << between-role,
relay-state equivalence holds.

### Step 3: Mask and test compiled state

After the suffix, replace subsequent tokens with neutral padding (e.g.,
newlines or spaces) up to the query position. Compare:
- Normal: prefix + suffix + query → distribution
- Masked: prefix + suffix + [padding] + query → distribution

If the suffix's effect persists through padding, the model has compiled
the operation into a persistent relay state, not just a lexical cue that
the query must re-read.

### Step 4: Compositional prediction from atomic table

Build a transition table T from single-suffix measurements:
For each suffix u, measure the relay state r_u and the output effect K_u.

For composition (u then v), predict:
- Relay state: r_{uv} ≈ compose(r_u, r_v) (linear? affine? nonlinear?)
- Output: K_{uv} ≈ K_u ∘ K_v

Test on UNSEEN compositions (held-out suffix pairs not in the training set
of the table). If the atomic table predicts held-out compositions, the
relay states form a genuine compositional algebra.

### Step 5: Causal transplantation

Extract relay state r_u from context A. Transplant it into context B at
the same layer-position cut (replacing B's relay state). Run B's query.

If the output follows r_u's class identity (not B's original suffix),
the relay state causally controls the computation. This is the strongest
test: it proves the relay state is not just correlated with output but
causes it.

Implementation: use PyTorch hooks to extract and inject activations at
the target layer. Requires forward-pass access (GPU).

### Step 6: Practical control — degraded scope correction

Final boss: take a prompt where the model gets the scope answer WRONG
(e.g., predicts the inner variable value when it should predict the outer).
Extract the relay state from a correct-answer context. Transplant it.
If the answer improves, the relay state carries actionable mathematical
information.

## Resource requirements

- GPU: Required for Steps 2-6 (activation extraction/injection)
- Memory: Qwen3-1.7B-Base fits in ~3.5GB VRAM (float16)
- Time: Steps 2-3 ~20 min GPU. Steps 4-5 ~30 min GPU. Step 6 ~15 min.
- Checkpointing: Save activations to disk between steps
- Short GPU bursts: Each step can run independently (checkpointed)

## Pre-registered predictions

- **H_native**: Relay states cluster by role (Step 2), persist through
  masking (Step 3), compose predictably (Step 4), causally control output
  (Step 5), and correct errors (Step 6). This establishes genuine native
  latent-space mathematics.

- **H_surface**: Relay states cluster by surface, not role. Masking
  destroys the effect. Composition is unpredictable. Transplantation
  follows surface identity. The model uses lexical cues, not compiled
  mathematical structure.

- **H_partial**: Some steps pass, others fail. Most likely outcome.
  Specifies exactly where the model's mathematical structure ends and
  surface dependence begins.

## Kill conditions

- Step 2 fails (relay states don't cluster by role): The model doesn't
  compile operations into role-invariant states. Gate 3 is dead.
- Step 4 fails (composition unpredictable): Even if relay states exist,
  they don't compose algebraically. Native math claim weakened.
- Step 5 fails (transplantation doesn't control): Relay states are
  epiphenomenal, not causal. The strongest version of the claim fails.

## Dependencies

- Gate 2 results (noncommutativity establishes the obstruction)
- GPU approval for activation extraction
- Layer selection heuristic (information-theoretic analysis of where
  role information is maximally concentrated)
