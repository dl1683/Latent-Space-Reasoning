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
affine paths smooth under the trained representation. Affine interpolation is
an imported chart path, so this is not evidence of intrinsic straight routes
or native geometry.

Classification: supported exploratory null-world evidence, not a native-geometry
claim. The ledger preregistration was written before scoring, but the result
artifact does not report its required anchor-bootstrap CIs for the main trained
versus null comparisons; the stronger confirmatory clause is therefore not
auditable. Under the guiding question, DINOv2 has learned a useful chart and
chart smoothness under a trained representation for this world. Composition
and transport must still decide whether those regularities survive beyond the
imported chart operation.

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

## Audit #3 (2026-08-27)

Residue narrowed to: training creates a task-effective chart and affine-path
smoothness in this encoder/dataset. NLM-006 v1 relabeled exploratory (cosine-
selected negatives; undefined native predictor; unverified label preservation).
NLM-006b to be locked with independent candidates, defined transport-aware
predictors, label-preservation and calibrated displacement gates.

## Round 11 — NLM-006b lock (2026-08-27)

NLM-006b is locked before scoring. It uses independent candidates: up to 20
random same-fine-label and 20 random cross-fine-label candidates per anchor,
frozen before metric evaluation, with the true fine-label endpoint and no
fine-label head. For `\tilde{x}=T_e x` and `\tilde{y}=T_e y`, the transport-aware
native predictors are `F_T=-(\tilde{y}-\tilde{x})^T G(\tilde{y}-\tilde{x})` and
`R_T`, the coarse-excluded profile agreement on the transported pair. Matched
chart controls are `cosine_T` and `euclid_T` on that same pair; direct, ST, and
TS remain controls.

The label-preservation gate is embedding-kNN fine-label preservation
`p_e >= 0.80` over the held-out split. Below that, a family is OOD and is
reported separately, never treated as native evidence. The displacement gate
uses `d=1-cos(E x,E T_e x)` on the fixed 200-image calibration subset: at least
80% of family images must exceed hflip's calibration `q95` displacement, with
`q10(d)`, control `q95`, and the fraction reported. Each valid family also
requires `>=320/400` supported anchors.

Exact predictions are registered: `crop50`, `invert`, and `occlude50` are
expected to pass the label/displacement/support gates and each is predicted to
give a best-native `TT` lead of `>=0.05` over the best matched chart control
with paired-bootstrap lower bound `>0`; `mix50` is predicted to have label
preservation `<0.80` and is OOD, with no verdict. Reopen the native-transport
line only if at least two valid families meet the native `+0.05` criterion (or
show a gated chart lead collapse to `<=0.02` with stable controls). Finally
close this frozen-encoder transport envelope if every valid family keeps a
gated chart lead `>=0.05`; fewer than two valid families is non-diagnostic.
No scoring or experiment was run in Round 11.

## Audit #4 (2026-08-28)

Frozen-encoder program PAUSED, not killed (NLM-006b non-diagnostic; gate
proxy poorly calibrated). NLM-007 blocked from scoring until the analyzer
repairs listed in theory/EXPERIMENTS.md are in; unseen-word split and a second
model family required before any scope claim.

## Next

**Replacement design:** stratified transports outside the encoder's measured
invariance class. No further frozen-encoder closeness/map competition is
planned. No native-map promotion follows from the current residue alone.

Keep `NLM-001` verdict and all gates unchanged.

## Round 12 — NLM-006b adjudication and next program (2026-08-27)

NLM-006b is **non-diagnostic under its lock**. The predeclared
label-preservation rule was `p_e >= 0.80`, and the measured rates were
`crop50=0.458`, `invert=0.317`, `mix50=0.185`, and `occlude50=0.416`; all four
are OOD. The near-identity controls were only `0.772` and `0.761`, a warning
that this kNN readout is a weak calibration proxy, but the threshold is not
changed post hoc. All four families passed the calibrated displacement gate
and all had 400/400 support.

The TT chart-survival pattern is descriptive only: cosine led the best native
predictor by `0.208`, `0.227`, `0.090`, and `0.222`, with paired intervals away
from zero. Because zero families are valid, this is not a lock-valid gated
chart closure, and it cannot support a native-transport claim. The displaced
families do show a small `ST>TS` chart order effect of about `0.035` with CIs
excluding zero; hflip does not. This is recorded as outside-class
non-commutation, not as a native law.

Close the frozen-encoder program as scope management. Residue: a trained
encoder supplies a task-effective chart metric, affine-path smoothness under
that trained representation, and graceful relative chart degradation under
identity-destroying moves; no tested native construct competes. Do not lower
the identity gate or run another frozen-encoder score comparison.

## Next program — worlds with dynamics

Use LM residual streams, where the forward pass is the lawful transport and
the map must predict where a state goes. The hypothesis, first measurement,
decisive result, kill conditions, and CPU cap are registered in
`theory/dialogue/003.md`. Reuse `experiments/substitution_probe.py`; no run is
authorized by this documentation-only round.

## Round 13 — NLM-007 lock and C1–C5 adjudication (2026-08-27)

Round 13 is documentation-only: no NLM-007 scoring or generation was run.
Claude's C1 is conceded: ridge versus kNN is chart-versus-chart and cannot
establish a native map. C2 and C3 are adopted: NLM-007 is a law-complexity
ladder with mean, kNN `k={1,5,20}`, ridge, low-rank affine, and kernel ridge,
evaluated primarily by four carrier-block hold-out folds (three blocks in,
one block out). A per-carrier word-cross-fitted oracle is reported only as a
ceiling, and a carrier-shuffled null breaks carrier pairing while preserving
target marginals.

C4 is adopted with an implementation amendment. Successor prediction is only
coordinate forecasting; the decisive endpoint inserts the predicted slot
successor into the actual full hidden sequence, runs the remaining layers plus
final norm/LM head, and compares the completed law to truth by KL and ordering
preservation. The current capture artifact stores slot states, so missing
full-sequence completion context voids the completed-law claim. C5 is adopted
with exact Qwen3-0.6B/28-layer and runtime metadata requirements.

NLM-007 is locked in `theory/EXPERIMENTS.md` before scoring. The six fixed
pairs are `L0→L1`, `L4→L5`, `L8→L9`, `L12→L13`, `L20→L21`, and `L27→L28`,
grouped into early, middle, and late regions. The preregistered minimal-class
prediction is low-rank affine early, low-rank affine middle, and kernel ridge
late. A gated dynamics-map claim requires a paired `+0.05` lead over the best
static chart on successor cosine and both completed-law readouts in at least
two pairs, with carrier-shuffled loss, 95% cell support, and word/carrier
clustered bootstrap. The run is CPU-only, one process, with a hard 20-minute
cap and no generation.

## Next (after Round 13)

The analyzer repair is present, but do not score until the amended lock below
is the active preregistration and the completion-context path, calibration-only
selection, null, metadata, support accounting, precision check, and clustered
bootstrap are auditable.

## Round 14 — NLM-007 amendment before scoring (2026-08-27)

Round 14 is documentation-only: no NLM-007 score or generation was run. The
lock is amended and re-locked before scoring. The word-conditioned mean
successor is now a separate lexical-persistence moot-maker: average each
word's 12 calibration-carrier successors, apply that vector to held-out
carriers, and report successor plus completed-law endpoints. It is not a
ladder member. A transport reading requires a candidate to beat it by at least
`0.02`, with a paired clustered 95% lower bound above zero, on successor cosine
and both completed-law readouts in at least two pairs; a result within `0.02`
on all three endpoints is lexical persistence, not transport, and an unresolved
comparison earns no transport claim.

Static controls now include inner-selected kNN regression with `k={5,20}`
alongside raw 1-NN cosine/Euclidean lookup. Both model and tokenizer revisions
must match the capture manifest. Zero-denominator, non-finite, and undefined
cells are excluded and counted, never repaired by epsilon. Completed-law
ordering differences receive clustered CIs; minimal class is reported
separately for successor and completed-law endpoints among ladder members only;
stored float16 laws must pass a reload comparison against fresh float32 laws.

The carrier-shuffled null is interpreted by depth. If a block's slot action is
carrier-independent, within-word targets are exchangeable and shuffling cannot
break the field: the smoke's `shuffled = 0.955 = field` is evidence of
context-free transport at that depth, not a kill. If the action is
carrier-dependent, a shuffled result within `0.02` of the field is the
marginal-state/presentation failure. Transfer at a carrier-independent depth
means a carrier-invariant law learned on some carriers predicts the same
word's successor on held-out carriers; it does not mean predicting
carrier-specific variation and still must clear the word-mean gate.

The full CPU plan remains six pairs, 100 shuffles, 2,000 clustered bootstrap
replicates, one process, and a hard 20-minute cap. If six pairs are projected
to exceed the cap, decide before scoring: retain one representative pair per
region (`L0->L1`, `L8->L9`, `L27->L28`), then reduce shuffles to 20, then
bootstrap replicates to 500 if necessary. Reduced null/interval budgets are
exploratory and cannot earn the full six-pair gated verdict. Never reduce the
held-out split, controls, completion endpoint, support accounting, or metadata
checks, and never reduce after seeing outcomes.

Exact predictions remain low-rank affine as the minimal class early and
middle, kernel ridge late, with late transfer weakening at the final block.
Each is conditional on clearing the word-mean separation gate and on the
depth-specific null interpretation. The unseen-word class-stratified split
and second model family are required follow-ups; until both are run, NLM-007
is a single-model interpretability result. The 16-word smoke is pipeline
validation only, not a result.

## Next

Do not score until the Round 14 amendment is recorded as the active lock and
all its validity checks are auditable.

## Round 15 — NLM-007 fallback adjudication (2026-08-28)

Round 15 is documentation-only; no experiment was run. The predeclared CPU
fallback was scored as `L0->L1`, `L8->L9`, and `L27->L28`, with 20 shuffles and
500 bootstrap replicates, so it is incomplete for the full six-pair lock and
overran the 20-minute cap by 19%.

`L0->L1` is lexical persistence: word-mean = field at `0.949`, shuffled null
`0.95`, support `1.0`. `L8->L9` is the first qualifying pair: ridge reaches
about `0.941` successor cosine versus chart `0.86` and word mean `0.861`,
clears the `>=0.05` chart and `>=0.02` word-mean gates on successor plus both
completed-law readouts with positive clustered lower bounds, support `1.0`,
and a shuffled null about `0.75–0.84` versus field about `0.94`. Full ridge, not
rank-`<=128` low-rank affine, is the supported middle-depth complexity; the
low-rank miss is about `0.05`. The within-carrier oracle is below the
cross-carrier field.

`L27->L28` is void for the completed-law claim: reading the law at the last
token disconnects it from the slot after layer 27, yielding degenerate KL,
undefined skill, ordering `1.0`, and support about `0.42–0.56`. The corrected
endpoint reads the next-token law at the substituted slot position itself
through the final norm and head.

Decision: run the remaining three pairs `L4->L5`, `L12->L13`, and `L20->L21`
next, with the corrected endpoint and a predeclared 30-minute CPU budget for
the estimated 24-minute extension. Predictions are: `L4->L5` low-rank affine
but lexical-persistence dominated and gate-failing; `L12->L13` full ridge,
with the second qualifying pair predicted; `L20->L21` kernel ridge, with
successor improvement possible but complete late three-endpoint passage
unlikely.

Under the guiding question, this is the first measured instance of a law in
Navigation requirement (e): a denizen can learn transport at middle depth on
some contexts and reuse it on others, then predict the world's downstream law.
It is not yet native. Required follow-ups are a class-stratified unseen-word
split with disjoint calibration/test words and the same gates, and replication
on a second model family (SmolLM2 or Gemma) with pinned revisions and the same
controls. Until both pass, the claim is limited to one model and shared words.

## Next

Run only the three predeclared remaining pairs after the corrected endpoint and
30-minute extension budget are recorded before scoring. Do not call the
two-pair gate until a second pair clears all three endpoints; do not generalize
the law beyond this model until the unseen-word and second-family follow-ups
are run.
