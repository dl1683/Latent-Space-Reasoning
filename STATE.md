# STATE

Canonical current state. Updated every session; history in NOTEBOOK.md.

## Direction (set 2026-08-27)

**Native mathematics of latent spaces.** Build the mathematics of latent/embedding
spaces from axioms upward — what a latent space *is*, which relations and
operations are actually meaningful — and derive definitions and propositions from
there, Euclid-style. Not porting ℝⁿ mathematics (vector arithmetic, cosine, Bayes
over coordinates) onto embedding vectors. Classical goals carry over (measure
closeness, update on evidence, compose, infer, prove); the constructs may be
unrecognizable. Scope: existing embedding spaces, constructed latent spaces under
our own axioms, and maps between latent spaces. Axioms may be discovered from
measurement or posited and tested.

**Mode.** Codex (real CLI) drives theorems and thoughts; Claude challenges,
curates, runs cleanup, keeps the loop going. Autonomous, continuous.

## Guiding question (Devansh, 2026-08-27)

Mathematics was invented by inhabitants of a world to navigate it — counting,
measuring, mapping, predicting — and its laws were shaped by what that world
made necessary. Invert the dynamic: **take the latent space as the world.** Ask
what a denizen of that world would have to invent to find its way — what counts
as the same place, what a move is, what effort a move costs, what a map is, what
regularities make prediction possible — and let that need decide which
primitives and laws we build. Every Codex round, audit, and sub-agent brief
starts from this question, not from a theorem we already know.

## Live question

What is the smallest native axiom set that makes navigation well-defined:
identity (when I have returned to the same place), admissible moves, move cost,
maps that predict unmade moves, and regularity laws across regions — and which
native instantiation predicts held-out behavior that a fixed ℝⁿ import cannot?

## Constraints

- No GPU runs without explicit user approval (laptop hard-crashes under sustained
  load). Theory and CPU work unconstrained. Qwen3-0.6B embeddings for local probes.
- Evidence gates in the local process file apply to any empirical claim.

## Prior program (closed 2026-08-27)

LLM perturbation line: nested-arithmetic claims withdrawn
(`legacy/docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md`). Standing results and
hypotheses are in `legacy/README.md`. Residue carried forward: the hardware-dependent
decoding-determinism finding; the termination/direct-control/null-model gates.

## NLM-001 verdict (2026-08-27)

**Bounded negative falsifier of the lexical next-token-KL instrument.** Kill
condition 8 makes the run void for confirmatory claims because required runtime
metadata were reconstructed post hoc. Under the locked analysis, native
calibration KL did not beat a learned diagonal metric on contextual hidden
states for held-out closeness orderings (Qwen3-0.6B
\(\Delta=-0.058\;[-0.22,+0.03]\)); kill conditions 3 and 6 therefore apply.
Qwen's H2 diagnostic crossed its numerical gate
(\(Q=2.12\;[1.70,2.56]\)), but may recover the authored probe taxonomy rather
than an invariant of the latent space. Registered directedness was not observed
and broader directedness was not adjudicated. Details: `theory/EXPERIMENTS.md`.

## NLM-003 verdict (Round 8, 2026-08-27)

The original true-fine table appeared to pass the narrow directional gate:
`R` profile accuracy was 0.7343 versus 0.6303 for `F` Fisher,
`Delta_{F-R} = -0.104 [-0.148, -0.058]`, over 6,199 scored pairs. The required
sensitivity rerun removes `PB_coarse` and gives `R=0.586` versus `F=0.667`,
with `Delta_{F-R} = -0.095 [-0.142, -0.049]`. The R win was a taxonomy leak,
so the R-over-F directional claim is withdrawn. Cosine and Euclidean still
dominate both leak-free native candidates on the same thin supported subset.
Verdict: corrected narrow instrument comparison; no native-map result; the
DINOv2 chart remains the best task-effective map measured for this artifact.

Under the guiding question, this may reflect DINOv2 pretraining making chart
proximity useful for visual regularities, but it does not establish intrinsic
geometry. The next measurement is a predeclared nonlinear re-chart plus held-out
composition and out-of-distribution moves, with the true fine-label endpoint.

## Audit #2 (2026-08-27)

NLM-003 reclassified as a narrow instrument comparison; "native constructs are
dominated" withdrawn as a general claim. Required before any further verdict:
R without the coarse head, tie accounting, kNN k-sensitivity, the random-init
null (NLM-004), a cheap-baseline ladder, re-charting, composed moves.

## NLM-004 verdict (Round 9, 2026-08-27)

The random-init null supports the registered point-estimate thresholds: cosine
fine-label consequence accuracy is 0.575 versus 0.946 trained (gap 0.371), and
embedding-kNN is 0.069 versus 0.761 trained. Null pixel-statistic heads remain
strong (RGB 0.8335; luma 0.8205), while the null coarse head is 0.2075, so the
result separates trained semantic chart structure from cheap image-statistic
signal. Null same-class fine-kNN chart-path flicker is 0.953 versus 0.127
trained, showing that training makes both chart nearness predictive and
chart-straight paths coherent.

Classification: supported exploratory null-world evidence, not a native-geometry
claim. The ledger preregistration was written before scoring, but the result
artifact does not report its required anchor-bootstrap CIs for the main trained
versus null comparisons; the stronger confirmatory clause is therefore not
auditable. Under the guiding question, DINOv2 has learned a useful chart and
path regularity for this world. Composition and transport must still decide
whether those regularities survive beyond the imported chart operation.

## NLM-005 lock (Round 9)

Next gate: held-out composed substitution/transport in the DINOv2 world,
comparing `ST: x -> y -> T_e(y)` with `TS: x -> T_e(x) -> y`. Transport is
two fixed image edits—horizontal reflection and one-pixel right translation
with declared padding—re-encoded by the frozen encoder on CPU at about 35
ms/image. Keep the true fine-label endpoint and common-support layout fixed;
compare cosine, Euclidean, `F`, and `R` without the coarse head, with exact tie
accounting and paired anchor-bootstrap CIs. The
`nlm003_v2_diagnostics` rerun is sensitivity only, not new evidence; it covers
R-without-coarse, ties, and kNN-fine M1 at k={8,32,128}.

The native-rescue prediction requires an order-sensitive composed family and a
native lead of at least 0.05 over the best chart metric; a chart lead of at
least 0.05 on both orders with at least 80% support kills that rescue. An
order gap below 0.02 or support below 80% is non-diagnostic. Budget: about 140
seconds of re-encoding for 2,000 held-out images under each of two edits, plus
at most ten minutes scoring, all CPU-only.

## Round 10 adjudication and direction

NLM-005 is **void and non-diagnostic**: support is 129/400 (`32.25%`), below
the locked 80% requirement. Hflip has ST−TS gaps no larger than about 0.006;
shift1px has near-zero chart gaps but `R_no_coarse=0.027` on its sensitivity
row. Cosine leads the best native candidate by about 0.32 on every scored
order. Because hflip and one-pixel translation are trained-invariant
augmentations, these were near-identity moves in the encoder's world; the
result cannot adjudicate general transport. The 40-random-candidate pool over
100 classes also made the support target implausible.

Close the frozen-encoder closeness/map competition as a program. Residue:
training supplies a task-effective chart metric and coherent chart-straight
routes; those collapse in a random-init chart; and no tested native candidate
competes after removing the coarse taxonomy leak. A denizen inherits this
navigation equipment from training; it is an operational map, not an
established intrinsic law.

The replacement, NLM-006, is a CPU-only, stratified audit of transports outside the
measured invariance class: large crops, color inversion, image mixing, and
occlusion. Use 20 same-fine-class and 20 cross-class hard-negative candidates
per anchor, freeze candidates before scoring, verify non-near-identity
displacement, and report support by stratum. The hypothesis is supported if at
least two of four families break chart consequence ranking or yield a
transport-aware predictor with a >=0.05 lead; it is killed if chart retains a
>=0.05 lead on every valid family. Invalid edits, endpoint leakage, or support
below 80% void the run. Estimated cost: about 280 seconds re-encoding plus at
most ten minutes scoring, no GPU.

## Next

**Replacement design:** stratified transports outside the encoder's measured
invariance class. No further frozen-encoder closeness/map competition is
planned. No native-map promotion follows from the current residue alone.

Keep `NLM-001` verdict and all gates unchanged.
