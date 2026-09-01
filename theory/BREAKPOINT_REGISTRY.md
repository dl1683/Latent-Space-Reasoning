# Philosophy Breakpoint Registry

What existing math assumed, where it broke in latent space, and what
each break tells us about native structure. These are clues, not
conclusions. Each breakpoint is a constraint on what native math must
look like.

Source: Phase 1 (50 audited experiments, 2026-08-27 → 2026-08-31).

---

## BP-1: Linear separability ≠ causal structure

**Assumption:** If a concept is linearly decodable from a representation,
the representation "has" that concept in a causally meaningful sense.

**Where it broke:** `register_bridge_preflight_v1` achieved 0.815 accuracy
decoding explicit-legend state from frozen Qwen3-1.7B — but this was a
noncausal lookup signal. The information was present but not causally
active. The subsequent causal interventions (`coordinate_v1-v3`,
`interchange_v1-v2`, `state_bus_v1r1`) all failed to turn decoded
information into causal control.

**What this tells us:** Native math must distinguish PRESENCE of
information from ADDRESSABILITY (can you read it?) from COMPOSABILITY
(can you use it to do things?). The three-gate model: present →
addressable → composable. Linear probes only test gate 1.

---

## BP-2: Single-site interventions ≠ distributed state

**Assumption:** If a model stores a fact, there's a specific location
(layer, position, direction) where that fact lives, and patching that
location transfers the fact.

**Where it broke:** Every single-site causal intervention failed.
`coordinate_v3` found a narrow late lexical-control effect at block 20
but it worked through direct logit projection, not state. `state_bus`
achieved repeated categorical control but it was pair-specific
lexical/semantic steering, not state-general.

**Phase 2 confirmation (v1d):** Component-level transplant proves this
directly. Transplanting only the attention output or only the MLP output
from an A-different donor shifts the B-answer by <1 logit (0/17 layers
flip). Full residual-stream transplant shifts by +4.1 logits (all layers
flip). Fusion is a WHOLE-STATE property — not localizable to any single
component. The distributed nature isn't just about position; even within
a single position, the state is a holistic object that can't be
decomposed into independent component contributions.

**What this tells us:** Native "place" may not be a point in activation
space. It may be distributed across positions and layers in a way that
R^n patching can't capture. A native "move" might need to be a
transformation of the computation itself, not a vector edit.

---

## BP-3: Vector distance ≠ semantic distance

**Assumption:** Points close in cosine/Euclidean distance are
semantically similar; the metric structure of R^n reflects meaning.

**Where it broke:** NLM-001 through NLM-006b. Cosine similarity was
the best predictor of task-effective relationships in DINOv2 — but
this was exactly the R^n trap. The metric worked because the encoder
was trained to make it work, not because it reflected native structure.
When we looked for structure BEYOND what the metric captured, we found
nothing — because we were still measuring with the metric.

**What this tells us:** The native distance may not be a distance at
all in the R^n sense. It might be asymmetric (A→B costs differ from
B→A), context-dependent (distance between A and B depends on what else
is present), or categorical (either same-place or different-place, with
no meaningful gradient).

---

## BP-4: Fixed-dimensional representation ≠ fixed structure

**Assumption:** Because the residual stream has dimension d, the
representation lives in R^d and has the structure of a d-dimensional
vector space.

**Where it broke:** `reachability_v1` found effective rank ~2.3 at the
probe site, but this was inseparable from baseline logit gaps,
cross-name gradient averaging, position effects, and prompt geometry.
The "dimensionality" of the representation wasn't an intrinsic property
— it was an artifact of the measurement.

**What this tells us:** Native dimensionality may be
context-dependent, task-dependent, and measurement-dependent. There may
not be a fixed "d" for the representation — the effective structure
may change depending on what the model is doing. Superposition
(Anthropic's work) suggests models use far more features than dimensions,
which means the vector-space structure is a carrier, not the content.

---

## BP-5: Composition in representation ≠ composition in vector space

**Assumption:** If the model represents A and represents B, then the
representation of "A and B together" is some function of repr(A) and
repr(B) in vector space (e.g., addition, concatenation, some learned
function).

**Where it broke:** The toy quotient program (Rounds 36-37) tried to
build exact compositions and found that no learned artifact passed the
complete exact reducer — only the oracle fixture did. Composition that
worked perfectly in the symbolic domain (affine bijections over GF(5))
didn't transfer to learned representations.

**What this tells us:** Composition in latent space may operate through
the MODEL'S FORWARD PASS, not through vector operations. The "algebra"
might be: feed A through the model to get repr(A), then feed
"A and B" through the model to get repr(A∧B) — and the composition
rule is the forward pass itself, not any vector operation we can
observe from outside.

---

## BP-6: Observation ≠ state (the instrument problem)

**Assumption:** We can observe the model's internal state by probing
activations at specific layers and positions.

**Where it broke:** Repeatedly. `onewrite_state_v1` — the base model
couldn't even apply a stated rule to visible tags. `onewrite_recall_v1`
— a valid direct-copy instrument failed held-out recall.
`oracle_actuator_rung0` — even with oracle codes, balanced eight-way
control failed.

**What this tells us:** Observation changes what you see. The act of
choosing a probe site, a readout function, and a set of concepts to
look for constrains what you can find. Native math needs to account
for the instrument — not as a nuisance but as a fundamental part of
the theory. This is where D2 (registration-relative observation) in
the axioms was heading.

---

## BP-7: Static snapshot ≠ dynamic computation

**Assumption:** A model's representation at time t can be understood
by examining activations at time t.

**Where it broke:** The "lexical-semantic steering" pattern in
`state_bus_v1r1` — the model's response depended on the specific
donor-verbalizer pairing, not on an abstract state. The model wasn't
storing state; it was computing a response path through the activation
space, and that path depended on the full context.

**What this tells us:** Native math may need to describe TRAJECTORIES
through activation space (the computation), not POINTS in it (the
state at one moment). The "object" isn't a vector — it's a
computational path.

---

## BP-8: R^n tools find R^n structure (circularity)

**Assumption:** PCA finds the "true" low-dimensional structure. Cosine
similarity finds the "true" similarity. Linear probes find the "true"
features.

**Where it broke:** PSQ-3α. PCA was transductive over all 32 states;
the quotient-holdout design shared all x-places between calibration and
held-out sets. The measurement tool imposed its own structure on the
answer. 69.14% accuracy (gate ≥95%) — but was this a failure of the
model or a failure of PCA to capture what the model was actually doing?

**What this tells us:** Every R^n tool carries its own axioms. Using
it gives you R^n answers. The native math must be discovered by tools
that DON'T presuppose R^n structure — or by carefully analyzing where
R^n tools FAIL and asking what structure would explain the failure.

---

## BP-9: R^n distance is blind to compositional structure (Phase 2, fusion-fission v1)

**Assumption:** If two representations are cosine-similar, they are
functionally similar — similar inputs to downstream computation.

**Where it broke:** Fusion-fission v1. Four worlds (2x2 fact combinations)
have cosine similarity ~1.0000 at layers 4-24. Yet transplanting one
world's hidden state into another produces dramatically different behavioral
outcomes depending on the layer: at some layers facts are independently
controllable (SEPARATE), at others changing one fact inevitably changes
both (FUSED). R^n's distance metric sees "same" where the computation
sees "different."

**What this tells us:** The behaviorally relevant structure lives in a
subspace or nonlinear manifold that global distance metrics can't see.
"Close in cosine" does not mean "functionally equivalent." Native
distance (whatever it is) must be sensitive to the compositional
structure that cosine misses. This may be the clearest evidence yet
that R^n distance ≠ native distance.

**PCA result (v1b):** PCA CAN separate worlds at all layers — so the
structure is linearly accessible. But PCA separation doesn't predict
behavioral fusion/fission: facts are linearly decodable even when
behaviorally inseparable. PRESENCE (PCA) ≠ ADDRESSABILITY (transplant).
The native math must describe causal accessibility, not storage geometry.

**Attention result (v1c):** Cross-fact attention is LOWER at fused layers
(0.275) than separate (0.336). The mechanism causing fusion is not
attention cross-talk — ruling out the most obvious pathway.

**Component transplant (v1d):** Neither attention-only nor MLP-only
transplant can flip fact B. Only full residual-stream transplant does.
Fusion is a WHOLE-STATE emergent property of the layer transformation.
(Connects to BP-2.)

**Trajectory result (v1e):** Worlds ALWAYS diverge through all layers.
FUSED layers diverge slowly (-1.2), SEPARATE layers diverge fast (-5.7).
Fusion = world-insensitive computation (model treats all A-worlds the
same). Fission = world-sensitive computation (model discriminates).
The behavioral transplant amplifies tiny directional differences through
remaining layers — static trajectory metrics are too coarse.

---

## Phase 2–5 breakpoints (2026-08-31 → 2026-09-01)

Additional structural holes discovered across 60+ experiments on frozen
Qwen3-0.6B-Base and learned architectures.

---

## BP-10: Composition is logit addition (terminal diagnostic, Gate I)

**Assumption:** If two interventions compose (the combined effect produces
the correct output), the model is performing some nontrivial nonlinear
computation to integrate them.

**Where it broke:** Terminal diagnostic Gate I. The logit-additive null
l_a + l_b - l_0 explains 99.1% of the composition effect measured in
RAC-0/RAC-1 (sqrt(JSD) = 0.0001 between predicted and actual composed
distributions, 12/12 top-1 match). The "composition" is ordinary logit
addition through the linear unembedding matrix.

**What this tells us:** The unembedding is linear, so perturbations at
hidden-state level produce approximately additive logit effects. The
post-B20 nonlinear layers do NOT significantly mix the two steering
directions. "Composition" through vector addition is just decoder
competition — the two perturbations independently shift logits, and
softmax picks the winner. Native composition (if it exists) must
operate through a mechanism that is NOT reducible to logit addition.

---

## BP-11: No template-invariant response fibers (terminal diagnostic, Gate F)

**Assumption:** States presenting the same fact in different wordings
produce sufficiently similar response distributions that they can be
treated as the same "place" in response space.

**Where it broke:** Terminal diagnostic Gate F. No template pair has
baseline sqrt(JSD) < 0.15 across 36 tested pairs (smallest gap: 0.19).
The model's response law is highly presentation-dependent — different
wordings of the same fact produce different distributions.

**What this tells us:** In Qwen3-0.6B, there are no template-invariant
response fibers at this task granularity. The model does not have a
"canonical response" for a given fact — the response depends on HOW
the fact was stated, not just WHAT was stated. Native math must either
work with presentation-dependent response laws or find/build a model
where presentation invariance holds.

---

## BP-12: Circuit selection in learned models (LAC-0)

**Assumption:** A model trained to compose actions will learn a single
compositional mechanism.

**Where it broke:** LAC-0 (learned action carrier, 739K params, 3 seeds).
Different weight initializations learn fundamentally different
capabilities: default init → endpoint composition (96%) but 0%
sequential; Xavier init → sequential execution (95%) but 34% composed
carriers. These are different optimization basins. Neither passes the
complete composition gate.

**What this tells us:** Composition may have multiple incompatible
realizations in neural parameter space. Training doesn't converge to
"the" compositional solution — it finds one of several incomplete
solutions depending on initialization. Native math must account for
this multiplicity: the "algebra" of the learned system depends on
which basin the optimizer fell into.

---

## BP-13: Capability floors (PSQ-1, permutation eligibility)

**Assumption:** A language model can track state through prompts if
the prompts clearly describe state transitions.

**Where it broke:** Every tested model size fails the two-dial world
(Z8×Z8, 64 states, Python-completion) capability gate (≥95%): 0.6B
(48%), 1.7B (50-54%), 4B (56%), 8B (55.5%), 8B-Instruct (50-64%).
Permutation eligibility: 48% overall, with only "reverse" working (97%);
cyclic rotations fail (13-34%).

**What this tells us:** Small-to-medium base models cannot track
multi-step state evolution through prompts. There is a CAPABILITY
FLOOR below which native mathematical structure cannot be tested —
the model doesn't have the behavioral interface. Any program seeking
native math in frozen models must first confirm the model can
reliably perform the underlying task.

---

## BP-14: Selectivity is verbalizer-sufficient (OSQ-1)

**Assumption:** Query-selective processing in late layers reflects
structured computation (a behavioral algebra, attention routing, or
state-dependent processing).

**Where it broke:** OSQ-1 (observational selectivity quotient). The
62× late-layer amplification is real (S(25)=0.633, G(25)=0.633,
R(25)=0.785) but coarse-graining to 3-bin (v0, v1, rest) accounts
for 100% of the selectivity signal (V=1.01). The selectivity is
entirely ordinary answer-token routing through the verbalizer.

**What this tells us:** What looks like "structured computation" in
the logit lens may be entirely explained by token-level routing to
the answer vocabulary. The verbalizer null (does coarse answer-token
routing explain everything?) must be tested before claiming behavioral
algebra.

---

## BP-15: Bypass is destructive (ERQ-1)

**Assumption:** Individual transformer blocks make independent
contributions that can be removed (bypassed) to study their function.

**Where it broke:** ERQ-1 (endogenous response quotient). Identity
bypass at block 25 produces viable states for only 7/48 cells. The
bypassed state is off-manifold — it's too destructive to serve as a
scientific instrument.

**What this tells us:** Transformer blocks are deeply coupled. The
output of block N is precisely on the manifold that block N+1 expects.
Removing a block doesn't give you "the representation without block
N's contribution" — it gives you garbage. Native math must account
for this coupling: layers are not independent operators that can be
composed/decomposed at will.

---

## Open questions from the breakpoints

1. If composition happens through the forward pass (BP-5), can we
   characterize the algebra of the forward pass itself? What are its
   operations, identities, inverses?

2. If state is distributed (BP-2), what is the right unit of analysis?
   Not a vector at a site, but... what? A circuit? A subcomputation?
   A pattern across positions?

3. If dimensionality is context-dependent (BP-4), what determines the
   effective structure at any given moment? Is there a native notion
   of "what matters right now"?

4. If observation is registration-relative (BP-6), can we build
   instruments that are transparent about their own axioms? Instruments
   that measure relative to themselves rather than relative to R^n?

5. The three-gate model (present → addressable → composable) from
   BP-1 — can we build tests for each gate separately? Can we find
   things that pass gate 2 (addressable) but fail gate 3 (composable)?
   That gap would be the most informative breakpoint of all.
