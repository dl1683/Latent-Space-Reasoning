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
`0.95`, support `1.0`. `L8->L9` is successor-only exploratory evidence: ridge
reaches about `0.941` successor cosine versus chart `0.86` and word mean
`0.861`, with the reported successor lead and shuffled drop. The prior
completed-law numbers are void because they were read at the last token rather
than the locked slot. Full ridge, not rank-`<=128` low-rank affine, is the
observed successor-side complexity at this depth; the low-rank miss is about
`0.05`. The within-carrier oracle comparison remains descriptive, not a ceiling
argument.

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

Under the guiding question, this is a successor-side sign that a denizen may
learn transport at middle depth on some contexts and reuse it on others. It is
not yet a completed-world law or native mathematics: the corrected slot
endpoint is still unscored. Required follow-ups are a class-stratified
unseen-word split with disjoint calibration/test words and the same gates, and
replication on a second model family (SmolLM2 or Gemma) with pinned revisions
and the same controls. Until both pass, the claim is limited to one model and
shared words.

## Round 16 — corrected endpoint adjudication and next order (2026-08-27)

Round 16 is documentation-only; no experiment was run. Tier-3 audit #5 found
that the completed-law endpoint read the sequence's last token rather than the
locked substituted slot. This invalidates every completed-law number from the
fallback and the three-pair extension for lock purposes, not only `L27->L28`.
The successor endpoint is independent of that readout and remains valid as
scored exploratory coordinate-forecast evidence. The extension outcomes are:
`L12->L13` matches its successor prediction (ridge `0.977` vs chart `0.898`
and word mean `0.888`, with the registered foldwise lead and shuffled drop);
`L4->L5` rejects lexical-persistence dominance but does not clear the chart
lead in all folds; and `L20->L21` rejects the kernel-minimal prediction while
ridge remains within `0.02` of kernel. No completed-law number is lock-valid.

The addendum establishes that stored hidden index 28 is post-final-norm. The
correct `L27->L28` completed law is `head(Yhat)` at the substituted slot, with
no remaining transformer layer; the old last-token readout is undefined. The
repaired completer's identity tests pass at `L8->L9` and `L27->L28`. The final
pair successor predicts a normed vector, so its cosine needs separate
qualification from raw-residual pair cosines.

### Corrected rerun preregistration

Predeclare a corrected rerun over all six pairs with the slot-position endpoint
primary throughout, 20 carrier shuffles, 500 word/carrier-clustered bootstrap
replicates, seed `13007`, one CPU process, and a **55-minute hard wall-clock
budget**. The budget follows the observed approximately 24 minutes per three
pairs, or about 48 minutes for six, plus fixed margin. It is exploratory
relative to the original 100-shuffle/2,000-bootstrap lock and cannot earn the
original full-budget label. No post-score pair reduction or control removal is
allowed; an over-budget run is incomplete and earns no two-pair claim.

Fixed slot-endpoint predictions: `L0->L1` remains word-mean dominated;
`L4->L5` has a successor advantage but fails at least one completed-law gate;
`L8->L9` remains a full-ridge qualifying pair; `L12->L13` is the predicted
second qualifying full-ridge pair; `L20->L21` has ridge within `0.02` of
kernel but fails the complete late gate; and `L27->L28` is finite through the
direct head but is not predicted to clear the full gate. Slot ordering follows
the same predictions: no qualifying ordering lead at `L0`, `L4`, `L20`, or
`L27`, and positive `+0.05`-scale ordering with clustered lower bounds at
`L8` and `L12`. The word mean may be strong at the slot because the response
law depends on prefix plus word; if it is within `0.02` of a field on all three
endpoints, the result is lexical/marginal persistence, not state-conditioned
transport.

### Alternative order

After the corrected rerun, run the audit alternatives in this order: (1) the
cheap identity-plus-residual baseline `Yhat = X + mean_cal(Y-X)` and a
per-carrier affine diagnostic; (2) forward-time append-token/next-position
transport; (3) the class-stratified unseen-word split; and (4) one second
model family. For the first step, use the existing artifact and outer folds;
fit the per-carrier affine model separately on each of the 12 calibration
carriers with fixed five-way class-stratified word cross-fitting (64 train,
16 test words), the same ridge selection discipline, and the same successor
and corrected slot-law endpoints. Treat it as a within-carrier diagnostic,
not a held-out-carrier competitor. A closed ridge lead is a cheap residual or
carrier-local explanation.

The forward-time test appends one fixed manifest-pinned sentinel to each
carrier-plus-word sequence and predicts the sentinel's next-position hidden
state from the preceding word-position state at selected layers, using the
same carrier-block holdout and completion law. The unseen-word split keeps
calibration and held-out word identities disjoint and class-stratified. The
second-family replication pins model/tokenizer revisions and repeats the
amended controls. Until both unseen-word and second-family tests pass, this
remains a one-model, shared-word interpretability result.

Under the guiding question, word identity is the field at `L0`, but by `L4`
the early blocks have manufactured carrier dependence: full-dimensional
affine prediction beats word mean and chart, and the growing shuffle penalty
shows that context matters more with depth. A denizen would need an identity
test, a context-conditioned transport law, and a completion rule in the
world's own response currency. It would have to discover when transport is
lexical, affine, or no longer regular, then test those distinctions on new
words, forward-time moves, and other realizations.

## Round 17 — corrected slot rerun adjudication and cheap moot-makers

Round 17 is documentation-only; no experiment was run in this round. The
existing corrected artifact was checked directly from
`experiments/results/lm_dyn_v1/analysis_slot.json`, including its raw pooled
values and per-fold clustered endpoint records. The six-pair run completed in
`2145.1 s` inside the predeclared `3300 s` budget, with 20 shuffles, 500
clustered bootstrap replicates, support `1.0` everywhere, matching revision
and config pins, and a passing float16 reload check.

The corrected slot endpoint qualifies `L8->L9`, `L12->L13`, and `L27->L28`.
Each clears the `>=0.05` chart lead on successor cosine, slot skill, and slot
ordering with positive clustered lower bounds; the `>=0.02` word-mean lead on
all three; the `>0.02` carrier-shuffle drop; and support `>=0.95`. `L0->L1`
is word-mean lexical persistence and clears only support. `L4->L5` and
`L20->L21` clear both corrected slot readouts, word-mean separation, shuffle,
and support, but fail the all-fold successor-cosine chart gate; `L20->L21`
has an association fold at `0.044`.

With `word_mean` excluded from the ladder, the minimal successor classes are
`knn5` at `L0` and `ridge` at every later pair. The minimal corrected
completed-law classes are `lowrank` at `L0` and `ridge` at every later pair.
The Round 16 prediction scorecard is 5/6 held: the only failure is the
prediction that final-block attenuation would prevent `L27->L28` from clearing
the full gate. Its successor cosine is on normed vectors and is not directly
scale-comparable to raw-residual pairs.

NLM-007 therefore meets the numerical two-pair dynamics-map criterion on the
corrected endpoint, with three qualifying pairs, but only as exploratory
corrected evidence at reduced `20/500` budget. It does not earn the original
full-budget confirmatory label. The exact permitted wording is: **a
full-dimensional, regularized affine predictor wins within this finite
ladder**. Correct slot completion upgrades the evidence from successor-only
forecasting with a void completed endpoint to a bounded corrected
completed-law result; it does not establish a native affine law, intrinsic
geometry, or a law of language-model dynamics generally.
*(Superseded for `L8->L9` and `L12->L13` by Round 18 and audit #7 below;
the wording is withdrawn at those pairs.)*

The depth profile is the key residue: word-mean slot skill falls
`0.95, 0.84, 0.78, 0.70, 0.43, 0.40`, while ridge stays near `0.92–0.98`.
The static chart falls to about `0.50` and `0.51` at the last two pairs while
ridge remains about `0.93` and `0.95`. The world appears to manufacture
carrier-dependent distinctions through its early blocks and make them more
necessary with depth. A denizen needs an identity test, a context-conditioned
transport rule, and a completion map, not merely a static chart. This remains
a one-model, shared-word observation.

### Next measurement

Run the first Round 16 alternative behind `--baselines` over all six pairs on
the existing artifact: identity-plus-residual
`Yhat = X + mean_cal(Y-X)`, and a separate per-carrier ridge diagnostic on
each of the 12 calibration carriers. Use five-way class-stratified word
cross-fitting per carrier (`64` train and `16` test words), calibration-only
regularization selection, the same corrected slot endpoint, 20 shuffles, 500
clustered bootstrap replicates, one CPU process, and a hard 55-minute budget.
The diagnostic is within-carrier and is not a held-out-carrier competitor.

Pre-run predictions for the not-yet-scored full run: identity-plus-residual
may be competitive at `L0` and `L4` but is not expected to close the ridge
lead at `L12`, `L20`, or `L27`; per-carrier affine will be strong within
carrier but will not explain a cross-carrier win unless it closes the same
three-endpoint lead under the same scoring. An existing one-pair `L8->L9`
smoke artifact already points against the first prediction: identity-plus-
residual exceeds ridge on successor cosine and slot skill and closes the ridge
lead in its foldwise endpoint comparisons. It is pipeline validation, not the
full six-pair result, but it is a withdrawal flag for `L8` pending full
accounting. If either baseline closes the corrected ridge lead on all three
endpoints for a pair in the full run, withdraw the finite-ladder wording for
that pair and remove any native-law interpretation.

## Next

Run the cheap moot-makers only under this pre-registration. Keep the
class-stratified unseen-word split and second-family replication as mandatory
follow-ups before generalizing beyond one model and shared words.

## Round 18 — moot-maker adjudication and displacement next

The six-pair NLM-007 baseline run took `4540.8 s` against its predeclared
`3300 s` hard budget. Per Round 16, it is budget-incomplete and contributes no
new gate claim. The predeclared null-making withdrawal still applies:
identity-plus-residual `Yhat = X + mean_cal(Y-X)` closes the ridge lead on
the three recorded comparison metrics (successor cosine, slot skill, slot
ordering; only the latter two are completed-law slot metrics) at `L8->L9` and
`L12->L13` (pooled differences approximately `-0.008/-0.021/-0.020` and
`-0.007/-0.009/-0.013`). Audit #7: the `<=0.02` closure rule is a
conservative post-hoc one-sided null-making policy, not a preregistered
equivalence test; the intervals support "no demonstrated positive ridge
advantage under this margin", not "no lead" or "equivalence".

Those are the two pairs that carried the Round 17 two-pair criterion, so that
criterion does not survive as a claim. The finite-ladder affine wording and
native-law interpretation are withdrawn for those pairs. `L4->L5` and
`L20->L21` retain small `0.02–0.03` state-dependent remainders across the
endpoints but remain non-qualifying and live; `L27->L28` remains a separate
post-norm/direct-head family, and its raw-state versus normed-target identity
comparison is not meaningful.

NLM-007 now establishes a bounded null result: at middle depth the measured
relation is consistent with identity plus a calibration-mean displacement
under this shared-word, held-out-carrier design; the experiment does not
determine whether the displacement is carrier-, state-, or word-dependent
(audit #7 wording). The world's move should be decomposed as
`T(X)=X+Δbar_cal+ε(X,w,c)`; “same place” is observational equivalence under
declared probes and downstream consequences within a fixed tolerance, not
equality of stored coordinates. Early blocks leave a small residual after this
decomposition, while the final block changes coordinate family through
post-normalization and a direct head.

### Next

Choose the displacement ladder before forward-time transport. Predict
`Δ=Y-X` from `X` on `L0->L1`, `L4->L5`, `L8->L9`, `L12->L13`, and `L20->L21`,
using the constant mean displacement as zero-order baseline and the
word-conditioned mean displacement as lexical moot-maker; `L27->L28` stays
out of the primary raw-residual family. Use kNN `{1,5,20}`, ridge, rank-128
low-rank, and kernel ridge; reconstruct `Yhat=X+Δhat` and score displacement
cosine plus corrected slot skill and ordering. A state-dependent result must
beat the word-conditioned mean by `>=0.02` with a positive clustered 95%
lower bound on all three endpoints, finite cells, and support `>=0.95`; report
shuffle and within-word spread diagnostically, with no native-law inference.

Keep the same pins, seed `13007`, 20 shuffles, 500 clustered bootstrap
replicates, one CPU process, no generation, and no GPU. Set a `95-minute` hard
wall, based on the observed `75.7-minute` six-pair run including roughly
`36 minutes` of per-carrier diagnostics plus approximately twenty percent
margin; an overrun earns no displacement gate claim. Predictions: no
qualifying residual at `L8/L12`, small live residuals at `L4/L20`, lexical
persistence at `L0`, and no raw-residual interpretation at `L27`.

Standing process rule for Claude to place in `AGENTS.md`: before fitting or
interpreting a transition law on a residual stream, run the
identity-plus-shared-displacement null on the same held-out splits, endpoints,
support accounting, and clustered gates; if the target is `Δ=Y-X`, make its
mean displacement the primary decomposition before adding model capacity, and
block a state-dependent transport claim until it beats that null and the
lexical moot-maker.

## Audit #7 — the identity-baseline withdrawal (2026-08-28)

Adopted verbatim (full text: `theory/EXPERIMENTS.md`, Tier-3 audit #7). (1) The
closure rule "pooled ridge − identres ≤ 0.02 on all three comparison metrics"
was chosen after seeing the scores; it is a **conservative post-hoc one-sided
null-making policy**, not a preregistered equivalence test, and is labelled so
wherever the withdrawal is cited. (2) The clustered intervals support "no
demonstrated positive ridge advantage under this margin", not "no lead" or
"equivalence". (3) "Persistence plus a shared displacement" is replaced by
"consistent with identity plus a calibration-mean displacement under this
shared-word, held-out-carrier design; whether the displacement is carrier-,
state-, or word-dependent is unresolved". (4) The three comparisons are
successor cosine, slot skill, slot ordering — only the latter two are
completed-law slot metrics. (5) "Exact" completion → "routing validated to
measured precision (per-pair max KL 1.9e-6 to 6.2e-6 over 16 × 80 cells)".
(6) `L4→L5` and `L20→L21` retain small live point-estimate remainders; not
killed, not promoted. (7) `L27→L28` identres is not a persistence test.

Durable residue (verbatim): identity is the null for residual-stream
transport. The present data support persistence plus a calibration-average
displacement as a competitive finite-design description at `L8` and `L12`,
retain small unresolved remainders at `L4` and `L20`, and do not yet establish
a native or generally reusable affine law.

## Round 19 — displacement adjudication and forward-time next move

Round 18's JSON was checked directly. The run finished in `1750.3 s` of the
recorded `5700 s` wall with support `1.0`; the ledger's mechanical reading is
correct that only `L20->L21` passes the predeclared three-endpoint displacement
gate. `L0` remains lexical persistence. `L4` retains a small live remainder
but fails the gate. `L8/L12` show strong displacement-coordinate separation
from the word-conditioned displacement mean and a collapsed shuffled null,
with kernel minimal, but their slot-ordering leads are only `0.003–0.022` and
slot-skill lower bounds are mixed, so the complete gate fails. `L20` clears all
three endpoints with kernel. `L27` remains outside the raw-residual family.

`L20` is one bounded finite-design qualifying pair: a kernel-class predictor
beats the word-conditioned displacement mean on displacement cosine, slot
skill, and slot ordering under held-out carriers. “Kernel minimal” is only a
finite ladder label; the result does not establish an ontologically nonlinear
or native law. The preferred wording is “state-dependent displacement beyond
the word-conditioned mean, with a kernel as the minimal tested predictor.”
*(Narrowed by audit #8 below: “displacement variation beyond the
word-conditioned mean; carrier/template vs state dependence unresolved.”)*
The middle-depth result is not only a readout artifact because displacement
cosine is a direct target, but its downstream consequence is readout-dependent
and identity-saturated.

Under the guiding question, consequential motion is a required derived
navigation predicate, not a sixth primitive: it says whether a move changes a
declared downstream response law beyond tolerance. Same place is observational
equivalence under those probes and laws, not equal coordinates.

Next, in fixed order, is forward-time append-token transport. Use period as a
fixed one-token sentinel and comma at the same appended position as the token
identity control. Let `X` be the final original-position state before append
and `Y` the sentinel next-position state after append, at layers
`{0,4,8,12,20}`. Run identity, shared-mean displacement,
word-conditioned-mean displacement, kNN, ridge, low-rank, and kernel on the
same held-out carrier folds. Complete at the sentinel position and read its
next-token law. The three gates remain displacement cosine, law skill relative
to shared-mean completion, and law ordering: at least `+0.02` over the
word-conditioned mean, positive clustered lower bounds, support `>=0.95`,
finite/reload cells, and a null original-terminal-position control. Capture is
about two CPU minutes; analysis is about 30 CPU minutes per five layer points
at 20 shuffles and 500 bootstraps. The unseen-word split stays separate.

## Round 20 ruling — forward locality tolerance (2026-08-28)

The absolute `1e-4` state tolerance was mis-scaled for residual coordinates of
magnitude up to about `378`. Corrected clause (settled before any forward score
was opened): state `<= max(1e-6 * M_q, epsilon_state_floor)` and log-law
`<= max(1e-4, epsilon_loglaw_floor)`, floors being the measured
batched-vs-single numerical floors at `q`. Here `3.624e-4 <= 3.78e-4` and
`6.58e-5 <= 1e-4`: both pass; the forward endpoint stays eligible. Read as
"no detectable causal nonlocality beyond measured numerical/kernel-path
variation under this run's corrected tolerance" (audit #8: the margin is
narrow, ~1.8e-5). Ledger `nlm007_forward_locality_control`,
`nlm007_forward_locality_ruling`; full text `theory/EXPERIMENTS.md`.

## Audit #8 — displacement claims and forward-time implementation (2026-08-28)

Adopted verbatim (full text: `theory/EXPERIMENTS.md`, Tier-3 audit #8).
Displacement result retained as: held-out-carrier evidence for predictable
displacement variation beyond a word-conditioned mean, with a kernel as the
minimal tested predictor; carrier/template versus state dependence remains
unresolved. `L20->L21` retained as one bounded qualifying pair under the
registered displacement-and-slot-law gate. The carrier shuffle is a
carrier-alignment diagnostic, not a state-independence null (shuffled field
reported for ridge/low-rank only). "The slot law barely registers it" is a
readout fact. "Consequential motion" is a derived predicate relative to a
declared law and tolerance. Strongest alternative: a carrier/template-
conditioned nuisance law encoded in the residual state. Forward-time
implementation verified; the missing A/B unappended-state equality check
passes bit-exactly (ledger `nlm007_forward_AB_equality`). Style-balancing /
within-template null / style-held-out split / Y−X decomposition / float32
precision reports are queued ahead of any "state-dependent" claim.

## Status and next (2026-08-28)

- Displacement ladder (`analysis_delta.json`, ledger `nlm007_delta_v1`):
  adjudicated Round 19; wording per audit #8 above.
- Forward-time move, sentinel A = '.' (`analysis_fwdA.json`, ledger
  `nlm007_forward_fwdA`; predeclared `nlm007_forward_predeclared`):
  adjudicated **not met** for the primary same-sentinel rule. Support is
  `1.0`, locality passes under the corrected Round 20 clause, `F0` is
  token-identity dominated, `F4/F8/F12` have large cosine and law-skill
  leads but fail the ordering gate, and only `F20` qualifies mechanically
  (ridge).
- Sentinel B = ',' (`analysis_fwdB.json`, ledger `nlm007_forward_fwdB`) is a
  secondary replication: `F12` and `F20` qualify (ridge); `F8` misses one
  ordering lower bound by `-0.002`, and `F4` misses one skill lower bound.
  This cannot rescue the period arm because the preregistration requires two
  layers for the same sentinel. The A/B unappended states and laws are
  bit-identical (`nlm007_forward_AB_equality`, all recorded maxima `0.0`).
- Round 20 ruling, audit #9 wording (adopted verbatim): "The period sentinel
  did not meet the preregistered two-layer, three-endpoint qualification
  criterion: only F20 qualified. This is a **nonpass under the historical
  contract, not a kill of forward transport**. In the shared-word,
  held-out-carrier design, sentinel displacement is predictably improved over
  the word-conditioned mean from F4 onward, and the response law registers
  that variation in cosine and skill. The ordering endpoint was later
  diagnosed as insensitive/saturated, so the qualification failure is not a
  substantive null result." The comma arm falsifies "token identity or
  position prevents any qualifying layer". Audit #8 leaves carrier/template
  presentation versus state dependence unresolved; no native, unseen-word,
  second-family, or general dynamics claim stands.
- The across-word within-carrier ordering endpoint is ruled
  insensitive/saturated for this question, without retroactively passing any
  run. Future runs replace it with a fixed candidate-predictor KL-to-truth
  rank endpoint: normalized rank lead `>=0.02` with a positive
  word/carrier-clustered lower bound, calibration-only selection, and the
  existing support/finite/reload/locality gates. Preregistered candidate set:
  `{identity, shared mean, word mean, kNN-1/5/20, ridge, low-rank, kernel,
  chart}` (K = 10).
- Within-style-family target null (ledger `nlm007_stylenull_predeclared`;
  smoke `nlm007_stylenull_smoke_F8A`; sentinel A `analysis_styleA.json`,
  ledger `nlm007_stylenull_styleA`; sentinel B `analysis_styleB.json`,
  ledger `nlm007_stylenull_styleB`, `F8/F12/F20` mechanical). Audit #9: (a)
  the null is an **alignment-destruction diagnostic,
  not a clean style null** — a field refit on a broken carrier pairing
  predicts the wrong carrier's displacement and falls below even the shared
  mean, so "beats the within-style null by 0.02" is not informative evidence
  for a state-linked component; the style-A "style-robust" reading is
  withdrawn as a claim and stands only as that diagnostic. (b) The KL-rank
  endpoint ranked K = 7 candidates (kNN-1/5/20 omitted), not the
  preregistered 10; repaired in the analyzer (`269e46c`); the style-A and
  style-B runs are labelled K = 7 and are not contract-valid on that
  endpoint.
- Audit #9 order: (1) within-family leave-one-carrier-out control
  (`--loco`, `3a8b859`) — predeclared and run under Round 21 below;
  (2) cross-fitted style residualization or a genuinely style-preserving
  conditional permutation; (3) the disjoint, class-stratified unseen-word
  split; (4) only then the second model family. Current order after audit
  #10 is in "Status after Round 21 and audit #10" below.

## Round 21 — LOCO ruling and pre-registration (2026-08-28)

Round 21 is documentation-only; no experiment was run. The live style-A and
style-B JSONs confirm the ledger's mechanical readings: `.` passes `F4/F8/F20`
and `,` passes `F8/F12/F20` under the historical style gate, with support 1.0.
Across `F4–F20`, the pooled JSONs put the null cosine at approximately
`0.16–0.54`, versus `0.45–0.66` for shared/word means and `0.68–0.82` for
ridge/kernel.
The observed pattern is mechanically the Round 20 state-linked branch—the
within-style null collapses below the shared mean from `F4` onward while the
original field remains strong—but Audit #9 rules that branch uninformative.
The target permutation breaks the exact carrier/state pairing, so a flexible
field is expected to predict the wrong carrier. “Style-robust” is withdrawn;
the runs establish only an alignment-destruction diagnostic. Their KL-rank
endpoint is labelled `K=7`, not contract-valid `K=10`; `269e46c` repairs the
candidate universe prospectively.

The implemented `--loco` is the fair cheapest within-family diagnostic: one
carrier held out within each style block, 240 training cells from the other
three, inner leave-one-carrier-out lambda selection, and comparisons against
identity, shared mean, per-word/block mean displacement, and ridge. It tests
state information conditional on observed word identities and within-family
training data, not cross-family transfer, style independence, unseen-word
generalization, or a native law. The word/block baseline intentionally uses
the held-out word identities; ridge can still use carrier/style nuisance
encoded in `X`; standardization is training-only; the three-carrier inner
selection is noisy; and the pooled 16-carrier bootstrap is secondary because
the rows are nested in four style blocks.

The LOCO run is predeclared for both sentinels and `F0/F4/F8/F12/F20`, with
500 word-clustered bootstrap replicates, an expected runtime near 60 minutes,
and a 75-minute hard CPU wall. A layer requires ridge minus block-word mean
of at least `0.02` with positive clustered lower bounds on displacement
cosine, law skill, and four-candidate KL-rank, plus `8/16` held-out carriers
passing all three; the diagnostic requires at least two layers per sentinel.
The state-linked prediction is a pass at `F4/F8/F12/F20` and no pass at `F0`;
the carrier/template-nuisance prediction is no pass, with block-word mean
closing ridge. Any result remains conditional within-family evidence. Next:
cross-fitted style residualization or conditional permutation, unseen words,
then a second model family.

## Status after Round 21 and audit #10 (2026-08-28)

- LOCO, sentinel A = '.' (`analysis_locoA.json`, ledger `nlm007_loco_locoA`;
  predeclared `nlm007_loco_predeclared`; 2902 s of the 4500 s wall; support
  1.0): `F4/F8/F12/F20` pass the Round 21 rule (pooled ridge − block-word
  mean: cosine +0.09–0.13, law skill +0.23–0.31, KL-rank +0.29–0.40, lower
  bounds > 0.08; 11–15 of 16 held-out carriers pass all three); `F0` no
  pass. **Adjudicated in Round 22.** Audit #10 wording (adopted): "On
  already-seen words, within a style family, X predicts a held-out carrier's
  displacement and response-law consequence better than the three-carrier
  per-word family mean at F4–F20" — not a presentation-independent state or
  a native law. The block-word baseline is variance-disadvantaged; before
  interpretation ridge must be compared against equalized X-free lexical
  baselines (word-only ridge; shrunk word mean). LOCO does not distinguish
  latent state from a smooth carrier/style code; the pooled 16-carrier
  bootstrap is secondary (block-first resampling for any cross-family
  statement). `F0` = "no detected conditional gain at F0".
- LOCO, sentinel B = ',' (`analysis_locoB.json`, ledger `nlm007_loco_locoB`;
  3091 s; support 1.0): **adjudicated in Round 22**; `F12/F20` pass; `F4`
  misses skill and KL-rank; `F8` misses skill only (KL-rank LB +0.021; audit
  #11 precision); `F0` fails. Weaker in breadth (2/5 vs 4/5) — a
  sentinel-specific instrument result, not evidence that B carries less
  state information.
- Oracle defect (ledger `nlm007_oracle_defect_forward`): the per-carrier
  "oracle" values in `analysis_fwdA/B`, `analysis_styleA/B`,
  `analysis_locoA/B` are meaningless (forward/delta mode predicted X from X);
  never cite them. Fixed prospectively; the oracle is a diagnostic, not a
  gate, so no result changes.
- Unseen-word split: `--unseen-words K` implemented with the audit #10
  X-free lexical nulls (`class_mean`, `wordonly_knn`), fixed K = 11 rank
  universe, fail-fast asserts, block-first pooled bootstrap. Smokes only
  (`analysis_unseensmoke.json`, ledger `nlm007_unseen_smoke_F8A`,
  `nlm007_unseen_smoke2_F8A`) — not a result; predeclared in Round 22, gates
  per audit #10 (lead ≥0.02 with positive
  clustered lower bound over the strongest X-free lexical baseline on
  displacement cosine, law skill, and fixed-universe KL-rank; block-first
  bootstrap; all eight fold keys valid).
- Second lens (Devansh; `AGENTS.md`): structural holes that make current
  latent spaces hostile to structured reasoning are first-class findings.
  Audit #10 table (adopted): proven — identity-dominated input transition
  (locally); ordering-saturated readout (for this endpoint). Unproven —
  presentation entangled with state (strong unresolved concern);
  family-only laws; motion invisible to the response law (readout-specific).
  The serious hole: no stable quotient separating lexical content,
  presentation, operational state, and consequential motion.
- Next order and scope: see "Status after Round 22 and audit #11" below.

## Round 22 — current state (2026-08-28)

LOCO A and B are adjudicated as positive within-family diagnostics, not native
law results. On already-seen words, within a style family, `X` predicts a
held-out carrier's displacement and response-law consequence better than the
three-carrier per-word family mean at `F4–F20`: A passes `F4/F8/F12/F20`, B
passes `F12/F20`, and both fail `F0`, with support `1.0`. The state-linked
prediction is partially held mechanically; the nuisance prediction of no
layer pass is mechanically missed but scientifically unresolved.

Before interpreting this as conditional state information, run the equalized
LOCO addendum: word-only one-hot ridge with inner calibration selection and a
calibration-selected shrunk word mean. The unseen-word split does not replace
this addendum; it removes word-conditioned lookup for a separate claim.

Round 22 predeclares the unseen-word run: `--unseen-words 2`, both sentinels,
`F0/F4/F8/F12/F20`, 20 shuffles, 500 bootstraps, seed `13007`, eight fold keys,
the `class_mean` and frozen-input-embedding `wordonly_knn` nulls, fixed `K=11`
rank universe, fail-fast split/class checks, and block-first class-preserving
bootstrap. The smoke (`634.8 s`) is pipeline validation only; the full scale
budget is a `150-minute` CPU wall.

Second-lens ruling: identity-dominated input transition and the
ordering-saturated readout are proven locally; presentation/state entanglement
and family-only laws remain unresolved. The central hostile property is the
absence of a stable predictive quotient separating lexical content,
presentation, operational state, and consequential motion. The next latent
space must define identity by interchangeability of declared moves and
downstream laws, expose presentation coordinates, use consequence-sensitive
divergence, support multi-step closure, and generalize across unseen words,
styles, and model families. No new axiom is warranted this round.

**Next:** see "Status after Round 22 and audit #11" below.

## Status after Round 22 and audit #11 (2026-08-29)

- Equalized LOCO addendum, sentinel A = '.' (`analysis_locoeqA.json`, ledger
  `nlm007_loco_addendum_predeclared`, `nlm007_loco_locoeqA`; 2911 s; support
  1.0): **defect-affected (ledger `nlm007_locoeq_defect_inner_centre`)** —
  the inner leave-one-carrier-out selection centred the word-only ridge and
  the shrunk word mean on the outer three-carrier shared mean, which includes
  the validation carrier's targets (direct pressure toward maximal
  shrinkage), and the `strongest_equalized` comparator was chosen on held-out
  outcomes. Outer margins are not leaked and stand as **descriptive numbers
  only** (F4–F20 ridge − equalized baseline: cosine +0.09–0.13, skill
  +0.23–0.30, KL-rank +0.26–0.34; F0 negative); "the data selected maximal
  shrinkage" is invalid as implemented. Sentinel B (`analysis_locoeqB.json`)
  carries the same defect. KL-rank universes differ across runs (LOCO K = 4,
  equalized K = 6): breadth comparable, KL-rank effect sizes not.
- Analyzer fixed prospectively (`d10fc66`: inner two-carrier centre;
  comparator frozen by calibration score); corrected rerun `locoeq2A/B`
  queued behind the unseen-word runs.
- Withdrawn as over-claims (audit #11): "no per-word lexical signal", "the
  variance objection is answered", "the state-conditioned component is
  large", "the forward law is about context rather than content". Adopted
  wording: the word-conditioned component captured by the tested estimators
  is negligible for the measured forward displacement in this design; the
  positive object is **X-conditioned residual predictability**, not
  state-conditioned structure. Strongest alternative: `X` carries a smooth
  presentation/template coordinate along which displacement and consequence
  vary; the unresolved distinction is lexical content vs presentation vs
  contextual operational state.
- Second lens (audit #11): the narrower true statement is that lexical
  content is not a sufficient predictor of the later forward step —
  context-bearing `X` contains predictable variation that word-conditioned
  means do not capture. Local holes: F0 identity/token dominance; ordering
  saturation (a readout hole); the missing quotient. The next latent space
  must define "same place" by interchangeability of declared moves and
  downstream response laws.
- **Next, in order:** superseded — see "Status after Round 23 and audit
  #12" below. Scope until the pending controls pass: one model; no native,
  second-family, presentation-independent, or general dynamics claim.

## Round 23 — current state after unseen words (2026-08-28)

The Round 22 unseen-word gate is met by both sentinels at `F4/F8/F12/F20` and
fails at `F0`. Calibration and held-out word identities are disjoint,
support is `1.0` for every key, the class-mean and frozen-input-embedding
`wordonly_knn` nulls sit at the shared mean, and the block-first pooled
contrasts are positive on displacement cosine, response-law skill, and fixed
`K=11` KL-rank. A has `7/8, 7/8, 8/8, 8/8` full-gate keys across those layers;
B has `5/8, 6/8, 8/8, 8/8`; both have 7–8/8 positive-all-three keys at the
passing layers. A's F0 continuation block collapses; B's F0 cosine lead is
below the `0.02` point gate.

Adjudication: the state-linked prediction is held only as **X-conditioned
residual predictability, generalizing across unseen word identities**. The
tested lexical-interpolation prediction fails; the presentation/style
nuisance prediction remains live because unseen words do not remove
presentation coordinates. This earns no presentation-independent,
model-general, native-law, or multi-step navigation claim.

The equalized LOCO addendum is still required for a fair interpretation of
the seen-word LOCO gap and its audit #11 defect. It is no longer a prerequisite
for the unseen-word result; the corrected `locoeq2A/B` runs are a diagnostic
and repair of the seen-word comparison.

**Next, in order:** after `locoeq2A/B`, run the predeclared cross-fitted
presentation residualization on existing captures, using primary centered
block/template coordinates and a mandatory augmented carrier-mean/rank-4
calibration-carrier subspace sensitivity. Fit nuisance maps calibration-only,
residualize both `X` and `Delta`, and retain the same three endpoints, K=11
unseen-word null comparison, block-first bootstrap, and a paired comparison
with the un-residualized field. Then repeat the protocol in a pinned second
model family. No new axiom is warranted in Round 23.

## Status after Round 23 and audit #12 (2026-08-29)

- Unseen-word runs, sentinel A = '.' (`analysis_unseenA.json`, ledger
  `nlm007_unseen_unseenA`; 2239 s) and B = ',' (`analysis_unseenB.json`,
  ledger `nlm007_unseen_unseenB`; 2256 s; predeclared
  `nlm007_unseen_predeclared`): `F4/F8/F12/F20` pass the Round 22 gate
  mechanically, `F0` fails, support 1.0. Audit #12 status (adopted):
  **"mechanical pass under the recorded reduction; formal gate pending a
  contract-correct bootstrap"** — the predeclared class-preserving word
  bootstrap was not implemented (words resampled without class strata and
  nested within blocks although crossed with them; only four block
  clusters, so intervals are sensitivity summaries), repaired prospectively
  in the analyzer; the lexical null family (four class means; k = 5
  frozen-embedding kNN over 40 words) is weak, and nested frozen-embedding→Δ
  ridge, nested embedding-conditioned kernel, and a predeclared k ladder are
  required before any "not lexical" reading.
- Wording (audit #12): "not exact held-out-word lookup and not the tested
  lexical interpolator" — never "not word lookup" unqualified; "the tested
  lexical nulls fail", not "lexical content is absent"; the ~0.06
  seen→unseen drop is a point comparison at F8 only; `F0` = "non-qualifying,
  with the continuation held-out block providing the strongest local failure
  pattern" (no formal collapse statistic exists). The positive object remains
  **X-conditioned residual predictability, transferring across the held-out
  word fold and held-out block**. Rejected formulations and the strongest
  alternative (smooth lexical and presentation coordinates in `X` along which
  the later displacement varies; the coarse nulls collapse for coarseness):
  `theory/EXPERIMENTS.md`, Tier-3 audit #12.
- Equalized LOCO addendum, sentinel B (`analysis_locoeqB.json`, ledger
  `nlm007_loco_locoeqB`; 2977 s): defect-affected like A (audit #11);
  descriptive only; `F12/F20` mechanical. Corrected reruns
  `analysis_locoeq2A/B.json` (analyzer `d10fc66`) are executing — per Round
  23 a repair and diagnostic of the seen-word comparison, not a veto on the
  unseen-word result.
- Residualization predeclared (ledger `nlm007_residualization_predeclared`;
  Round 23 design in `theory/EXPERIMENTS.md`); analyzer `--residualize
  static|aug`, the class-preserving crossed bootstrap, and the stronger
  unseen-word lexical nulls are under smoke test (`analysis_residsmoke.json`,
  not a result).
- **Next, in order:** superseded — see "Status after Round 25 and audit
  #13" below. Scope until the pending controls pass: one model; unseen-word
  status mechanical only; no native, presentation-independent,
  second-family, or general dynamics claim.

## Round 24 — audit #12 repairs and residualization ruling (2026-08-28)

No experiment was run. `analysis_locoeq2A/B.json` remains unopened.

- Audit #12 repair: the four stronger X-free lexical nulls are fair and nested
  on calibration words (class mean, frozen-input-embedding kNN, embedding-to-
  displacement ridge, and embedding-conditioned kernel). Standardization and
  k/lambda/gamma selection use only a two-fold class-stratified calibration-
  word split. The unseen-word gate is amended to beat the strongest of all
  four nulls with the fixed K=13 KL-rank universe and the existing clustered
  gates.
- The old A/B artifacts remain mechanical-only/descriptive under K=11, two
  nulls, and the old bootstrap. They need not be discarded or rerun as a
  separate historical repair; the amended gate is carried into the
  residualization runs, where a raw shadow margin must be retained in the same
  folds.
- The pre-amendment pooled bootstrap was class-preserving but only
  approximately crossed because it drew words independently per block. The
  analyzer is prospectively tightened to share one class-stratified draw per
  word-fold key across blocks, with carriers resampled within block.
- Residualization is accepted as the intended decomposition, with binding
  repairs: calibration-word-only augmented carrier coordinates and inner
  training-carrier subspaces, residual-space ladder selection, and a same-fold
  un-residualized ridge arm plus paired block-first contrast. Round 22 fold
  values alone cannot supply the retention marker. The prospective patch adds
  the raw arm, but the raw four-null shadow margin still must be emitted
  before that marker is complete and before launch.
- The F8 sentinel-`.` static smoke is a pipeline preview: presentation-only
  `P -> Delta` cosine `0.42`, residual ridge about `0.60` versus nulls about
  `0.06–0.07`, skill about `0.36` versus `0.015`, and KL-rank lead about
  `+0.44–+0.50`. It supports survival after this static removal but does not
  establish the formal gate, causal state structure, or a native law.
- Confirmed readout order for the four CPU-only runs: A-static, A-augmented,
  B-static, B-augmented; five layers `F0/F4/F8/F12/F20`, two unseen-word
  folds, 20 shuffles, 500 bootstraps, and a 60-minute wall per
  sentinel/design. Read validity, then F0, then F4–F20; within each layer
  read residual margins/gates, reassembled law endpoints, presentation-only
  diagnostic, and paired raw-field retention. No GPU or new capture.

**Current answer after Round 24:** presentation explains a substantial
measured part of the raw displacement, while residual X-conditioned
predictability survives the static smoke. This sharpens the missing quotient
between lexical content, presentation, operational state, and consequential
motion; it does not resolve it. No new axiom, presentation-independent claim,
native-law claim, or second-family claim is warranted.

## Round 25 — launch ruling and budget amendment (2026-08-29)

No experiment was run. The raw four-null shadow prerequisite is met at
pipeline level by `analysis_residsmoke.json` / ledger
`nlm007_resid_shadow_smoke_F8A`: same folds as the residual field, raw
`unres_*` predictors scored against raw held-out `Delta`, raw skill referenced
to `unres_mean`, K=13 KL-rank by substitution into the ridge slot, four
per-null raw margins, and positive block-first pooled `unres_ridge` contrasts
over four blocks and eight word-fold keys. The smoke is not a formal result.

Retention is now frozen as a conservative minimum over the fixed four nulls,
per endpoint and identically on raw and residual sides. If `N4` is
`{class_mean, wordonly_knn, wordonly_ridge_emb, wordonly_kernel_emb}`, then
`m_raw[e] = min_n(raw_ridge[e] - raw_null_n[e])` and
`m_res[e] = min_n(ridge[e] - residual_null_n[e])`; endpoint retention passes
iff `m_raw[e] > 0` and `m_res[e] >= 0.5*m_raw[e]`. The strongest null is the
one attaining that smallest point margin, with declared-order tie breaking.
All per-null margins and the same-fold paired block-first `ridge -
unres_ridge` contrast are reported; the denominator is never reconstructed
from old Round 22 files.

Measured one-layer times are 756 s before raw ridge, 998 s with raw ridge,
and 1294 s with all five raw-shadow arms under contention. The old 60-minute
wall is amended before formal scoring to 120 minutes per run and 8 hours for
the four serial runs. The fixed K=13 ladder, including `knn1/knn5/knn20`, is
retained. After corrected equalized rerun B finishes, the conditional launch
order is A-static, A-augmented, B-static, B-augmented, one process at a time.
No launch occurs in Round 25 itself.

Corrected equalized A is a valid mechanical positive for the bounded
sentinel-A seen-word within-family diagnostic: its equalized baselines sit
0.003–0.01 above the shared mean, and F4/F8/F12/F20 pass while F0 fails.
Audit #11's inner-centre defect is repaired; “no per-word lexical signal,”
“context rather than content,” “large state-conditioned component,” and any
presentation-independent/native-law reading remain withdrawn. Current scope
remains one decoder, unresolved state-versus-presentation, and no new axiom.

## Status after Round 25 and audit #13 (2026-08-29)

- Corrected equalized LOCO addendum, sentinel A = '.'
  (`analysis_locoeq2A.json`, ledger `nlm007_loco_locoeq2A`; 3753 s of the
  4500 s wall; support 1.0): **contract-correct; adjudicated Round 25** as a
  valid mechanical positive for the bounded sentinel-A seen-word
  within-family diagnostic — the calibration-selected equalized comparator
  sits roughly 0.002–0.009 above the shared mean; `F4/F8/F12/F20` pass,
  `F0` fails. Audit #13 wording: audit #11's inner-centre *defect* concern
  is resolved by the corrected sentinel-A data (not "audit #11 is
  resolved"); the pooled equalized interval is secondary. Maximum wording:
  "On already-seen words, within sentinel A's style-family design, the
  context-bearing X field predicts the held-out carrier's forward
  displacement and response-law consequence beyond the properly nested,
  calibration-selected X-free lexical comparator at F4–F20."
- Corrected equalized LOCO addendum, sentinel B = ','
  (`analysis_locoeq2B.json`, ledger `nlm007_loco_locoeq2B`; 4196 s; support
  1.0): **contract-correct**; equalized baselines 0.002–0.007 above the
  shared mean; `F12/F20` pass, `F4/F8` miss on skill/KL-rank lower bounds
  (cosine leads hold), `F0` fails; run-level positive (2/5). Codex
  adjudication pending; Round 25 requires it before any combined A/B
  equalized reading. Both arms agree with the defect-affected runs' numbers
  (baselines moved by ≤0.01, no verdict changed).
- Residualization (Round 23 design; Round 24 contract; Round 25 launch
  ruling, ledger `nlm007_residualization_budget_amended`: 120-minute wall
  per run, 8 hours for four serial runs, K=13, retention = minimum-margin
  rule over the same-run raw four-null shadow): the raw-shadow prerequisite
  is met at pipeline level by `analysis_residsmoke.json` (ledger
  `nlm007_resid_shadow_smoke_F8A`; not a result). The chain runs in the
  locked order A-static (`analysis_resSA.json`) → A-augmented (`resAA`) →
  B-static (`resSB`) → B-augmented (`resAB`), one process at a time.
  *Superseded for A-static: see "Status after Round 26 and audit #14"
  below.*
- Retention marker (audit #13, ledger `nlm007_retention_marker_defect`):
  raw and residual margins are not commensurate (cosine on different
  targets; skill against different references; KL-rank by ridge-slot
  substitution). Until the common-scale repair is scored, the residualization
  runs may say only **"the predeclared robustness marker is mechanically
  met"** — never "half of the signal survives". The common-scale marker
  (residual arms reassembled to full Δ and scored against raw Δ with a common
  skill reference; strongest-null minimum recomputed inside each replicate)
  is predeclared (ledger `nlm007_retention_common_scale_predeclared`) and
  implemented for `resAA/resSB/resAB`; `resSA` started on the pre-patch
  analyzer and reports retention only as "robustness marker mechanically
  met". The raw four-null shadow remains valid for the amended raw
  unseen-word comparison; the residual-vs-null gate, law reassembly, and
  presentation-only arm are coherent; `P_static→Δ` = 0.427 is a held-out
  cosine, never "explains 42%".
- Public demo (audit #13): the published "Content vs Context" page had
  violated audits #10–#12 ("context state", "context takes over",
  "manufactures", "presentation explains 0.42"); every replacement was
  adopted verbatim and republished at the same URL.
- Reverse tunnel (audit #13): the X-conditioned advantage can no longer be
  dismissed as lookup, noisy-mean artifact, class mean, embedding
  interpolation, or pipeline accident; presentation may itself be part of
  operational state — the target is an operational equivalence relation
  defined by moves and consequences, not presentation invariance at all
  costs. The current observational residualization cannot decide between
  the smooth-coordinate and the context-conditioned-regularity accounts.
- **Next, in order:** residualization A-static → A-augmented → B-static →
  B-augmented (read validity, then F0, then F4–F20) → Codex adjudication of
  the four runs and of corrected equalized B → second model family (pinned
  revisions, same controls). Scope until then: one decoder; unseen-word
  status mechanical only; state-versus-presentation unresolved; no native,
  presentation-independent, second-family, or general dynamics claim; no
  new axiom.

## Round 26 — A-static residualization adjudication (2026-08-29)

`analysis_resSA.json` is contract-valid for the primary residual-vs-null
question: sentinel A, `P_static`, two unseen-word folds, K=13, crossed
class-preserving bootstrap, 20 shuffles, 500 bootstraps, support `1.0`, and
`4405.7 s` of the `7200 s` wall. F4/F8/F12/F20 pass; F0 fails. At the passing
layers `X_perp` ridge residual cosine is `0.56–0.62` versus `0.06–0.07` for
the strongest residualized X-free null. Block-first margins are cosine
`+0.50–+0.56`, skill `+0.31–+0.48`, and K=13 KL-rank `+0.40–+0.61`, with
positive lower bounds, 6–8/8 full keys, support `1.0`, and no block collapse.
F0 has negative pooled skill and an association-block collapse.

Prediction ruling: `P_static` takes the non-collapse/state-linked-side branch
of the primary Round 23 gate, but it does not establish state. The maximum
statement is residual predictability of `X_perp` beyond residualized X-free
lexical nulls after removal of the **registered static presentation
coordinates**, across held-out words and blocks. `P_static` pass plus
`P_aug` collapse remains the predeclared “static coordinates incomplete, not
state” branch. No presentation-independent, native-law, second-family, or
general dynamics claim follows.

The presentation-only `P_static -> Delta` arm has held-out cosine
`0.43–0.63` by layer. This is a large predictable presentation component, not
a percentage of displacement explained. It sharpens the earlier unseen-word
reading: much of the raw X-conditioned lead may have been presentation-
mediated. The residual result adds only that the registered static coordinates
do not account for the whole X-linked advantage; it does not identify the
remainder as operational state.

*Audit #14 (Tier-3, `theory/EXPERIMENTS.md`, ledger `nlm007_audit14_adopted`) withdrew the sentence "much of the raw lead may have been presentation-mediated" as an over-read and replaced the ruling; read this paragraph as corrected there.*

Retention remains bounded by Audit #13. `resSA` predates
`retention_common_scale` / `retention_common_scale_block_first` and contains
neither field. For A-static, say only **“the predeclared robustness marker is
mechanically met.”** A patched A-static rerun is required before any
common-scale A-static or symmetric four-cell retention claim; `resAA` and the
B runs cannot substitute for that cell. The rerun is not required for the
primary residual-vs-null verdict and follows the existing chain.

Second-lens ruling: presentation sensitivity of the measured move is proven
locally; presentation/operational-state inseparability is not. The current
space lacks a demonstrated native quotient and required analyst-known block,
length, and position coordinates for this decomposition. The next latent
space must expose lexical, presentation, and operational coordinates; define
sameness by interchangeability under declared moves and response laws; treat
presentation as state only when controlled changes alter those laws; support
consequence-sensitive multi-step closure; and transfer across unseen words,
styles, and model families. No new axiom is earned.

**Next, in order:** read A-augmented (`resAA`), B-static (`resSB`), then
B-augmented (`resAB`), without outcome selection. For each, read budget and
manifest/reload/locality/support/common-scale validity first, then F0, then
F4/F8/F12/F20; within each layer read residual gates, reassembled consequence
endpoints, key/block accounting, presentation-only arm, raw shadow, and
repaired common-scale retention. A broad `P_aug` collapse toward the strongest
residualized X-free null is the style-nuisance outcome and, after the A-static
pass, the “static coordinates incomplete, not state” branch. After the chain,
run the patched A-static common-scale repair before cross-run retention
synthesis or the second-family protocol. No experiment was run in Round 26.

## Status after Round 26 and audit #14 (2026-08-29)

- Residualization A-static (`analysis_resSA.json`, ledger
  `nlm007_resid_resSA`; 4405.7 s of the 7200 s wall; support 1.0):
  **contract-valid for the primary residual-vs-null question; adjudicated
  Round 26 as corrected by Tier-3 audit #14** (ledger
  `nlm007_audit14_adopted`). `F4/F8/F12/F20` pass (full-gate keys 7/8, 7/8,
  6/8, 8/8; misses family-localized in gloss/association; the four
  checkpoints are correlated measurements, not replications); `F0` = "no
  qualifying conditional gain at F0 under this instrument" — a genuine
  negative control. Not a cosine-geometry mirage: residualization lowers the
  ridge cosine (raw 0.65–0.76 → residual 0.56–0.62) while the lexical nulls
  fall to ~0.06; shuffled q95 ≤ 0.13; residual normalized error 0.78–0.83.
- Adopted ruling (audit #14, replacing Round 26's): "`P_static` took the
  non-collapse branch. Locally, the result establishes
  registered-presentation sensitivity and survival of X-linked residual
  predictability after cross-fitted removal of those registered coordinates.
  It identifies neither the surviving field as operational state nor the
  result as presentation-independent." Joint license: registered static
  template coordinates predict held-out raw displacement; after cross-fitted
  removal of those registered coordinates from both `X` and `Δ`, `X⊥` still
  predicts `Δ⊥` and its reassembled response-law consequence beyond the
  registered residual X-free lexical nulls at F4–F20; these facts establish
  presentation sensitivity and residual X-linked predictability and do not
  identify how much of the raw ridge advantage is attributable to
  presentation or whether the remainder is operational state. **Withdrawn**
  (over-read): "much of the raw lead may have been presentation-mediated";
  a `P_static→Δ` cosine of 0.43–0.63 gives no variance share, fraction,
  mediation, decomposition, overlap, causal effect, or proof of pure
  presentation.
- Retention for `resSA`: **"the predeclared robustness marker is
  mechanically met"** only (pre-patch analyzer; audit #13). The patched
  A-static rerun `analysis_resSA2.json` is predeclared (ledger
  `nlm007_resid_resSA2_predeclared`; identical design, common-scale
  retention block) and queued after B-augmented; required for any A-static
  or symmetric four-cell common-scale retention claim.
- The gate is too easy for a *state* claim, not for the registered narrow
  claim. To be preregistered by Codex before any state reading: a fully
  refitted Freedman–Lane residual-geometry null preserving nuisance
  geometry, and a flexible calibration-only `P_aug`/lexical interaction
  comparator without cell-level `X⊥` (a P-only zero residual is not the fair
  comparator); the common-scale decomposition must be completed before the
  raw lead is attributed.
- Strongest live alternative (audit #14): `X⊥` still contains
  high-dimensional nonlinear template/carrier geometry and smooth lexical
  coordinates shared across held-out blocks and words, along which `Δ⊥`
  varies; ridge/kernel recover it and the decoder registers it. Competing
  positive: a genuine context-conditioned transition regularity with
  presentation partly legitimate operational state. A-static observational
  residualization cannot choose between them.
- Public demo corrected again per audit #14 (nine verbatim replacements)
  and republished at the same URL.
- **Chain order:** `resAA` (A-augmented, **running**) → `resSB` → `resAB` →
  `resSA2` (patched A-static) → Codex adjudication of the four runs, the
  patched cell, and corrected equalized B → second model family (pinned
  revisions, same controls). Read validity, then F0, then F4–F20, without
  sentinel/layer/design selection after outcomes. Scope until then: one
  decoder; unseen-word status mechanical only; state-versus-presentation
  unresolved; no native, presentation-independent, second-family, or general
  dynamics claim; no new axiom.

## Round 27 — A-augmented adjudication and next comparator locks (2026-08-29)

`analysis_resAA.json` is contract-valid for the registered A-augmented
residual-vs-null question: sentinel A (`.`), `P_aug`, two unseen-word folds,
K=13, crossed class-preserving bootstrap, 20 shuffles, 500 bootstraps, support
`1.0`, and `4737.8 s` of the `7200 s` wall. The live implementation fits the
rank-at-most-4 carrier basis only on calibration carriers and calibration
words and scores a leave-current-word-out carrier mean in that basis; it
rebuilds the basis inside inner carrier folds.

All five layers pass. At F0, `X_perp` ridge residual cosine is `0.335` versus
`-0.006` for the strongest residualized X-free null; crossed block-first
margins are cosine `+0.341 [LB 0.246]`, skill `+0.156 [0.022]`, and K=13
KL-rank `+0.303 [0.122]`. Only `2/8` keys clear the full per-key gate, although
`7/8` are positive, and no block collapses. At F4/F8/F12/F20, ridge residual
cosine is `0.617/0.595/0.555/0.612` versus strongest-null
`0.062/0.074/0.060/0.071`; cosine margins are `+0.555/+0.521/+0.495/+0.541`,
skill margins `+0.458/+0.346/+0.369/+0.457`, and KL-rank margins
`+0.485/+0.432/+0.425/+0.557`, all with positive lower bounds. Full-gate keys
are `8/8`, `7/8`, `6/8`, and `8/8`; every key is positive and no block
collapses. The presentation-only `P_aug -> Delta` cosine is
`0.639/0.498/0.446/0.475/0.608` at F0/F4/F8/F12/F20.

Round 23 branch: A-static and A-augmented jointly take the **both-design
non-collapse branch**. The four registered X-free lexical nulls and the
registered static-plus-augmented presentation collapse prediction both miss.
The positive object is X-linked residual predictability after removal of the
registered coordinates, not operational state. Audit #14's joint license is
extended accordingly: registered static and augmented coordinates predict
held-out raw displacement, and after cross-fitted removal of either design
from both `X` and `Delta`, `X_perp` still predicts `Delta_perp` and its
reassembled response-law consequence beyond the residual X-free nulls at
F4–F20. The pair establishes presentation sensitivity and residual X-linked
predictability; it identifies neither overlap/fraction/mediation nor state or
presentation independence.

F0's augmented pass is a real cross-fitted **conditional residual** gain, not
held-out-target leakage, but it does not reverse raw identity dominance. The
same-run raw ridge cosine is `0.687` versus a raw null near `0.669`, and
`P_aug -> Delta` is `0.639`: the carrier-mean/subspace coordinates remove an
identity/carrier-dominated component and expose a different residual problem.
The `2/8` full-key count blocks a broad robust-F0 reading.

The repaired common-scale retention block is present. Bootstrap-median ratios
`residual margin / raw margin` [95% CI] for cosine, skill, and continuous KL
are:

- F0: `2.513 [1.337,5.098]`, `1.460 [0.871,4.559]`,
  `1.255 [0.612,5.393]`;
- F4: `1.105 [0.968,1.323]`, `1.062 [0.811,1.758]`,
  `0.778 [0.495,1.195]`;
- F8: `1.166 [1.044,1.326]`, `1.137 [0.792,1.618]`,
  `0.942 [0.652,1.249]`;
- F12: `1.231 [1.102,1.428]`, `1.160 [0.957,1.353]`,
  `1.127 [0.996,1.243]`; and
- F20: `1.105 [1.020,1.248]`, `0.951 [0.787,1.073]`,
  `0.974 [0.784,1.098]`.

Thus A-augmented retains at least half of the same-run raw
ridge-versus-strongest-null **predictive margin on the common raw-Delta scale
at the bootstrap median** for every layer and endpoint. A uniform 95%-interval
claim is not earned because F4 continuous KL dips to `0.495`. This is not a
fraction of latent signal, variance, state, or mediation, and it cannot fill
the missing A-static common-scale cell.

Two existing-capture comparators are now preregistered before any state
reading:

1. a calibration-only residual-space X-free field with `P_static`, the same
   rank-4 carrier summaries, 16 calibration-word PCA scores of the frozen
   lexical embedding, and fixed 4-by-16 interactions, matched inner tuning
   and a degrees-of-freedom sensitivity; the cell-level field must beat it on
   cosine, normalized error, skill, and continuous KL with crossed
   block-first lower bounds, `6/8` positive keys, no collapse, and at least
   two F4–F20 layers per cell; estimated `4.9–5.3 CPU h` for four cells,
   `8 h` hard wall; and
2. a fully refitted Freedman–Lane residual-geometry null with 20 null refits
   per outer fold-key, permuting calibration `Delta_perp` across carriers
   within template family and word and rerunning inner selection, ridge/kernel
   refit, and held-out scoring; observed cosine, normalized error, skill, and
   continuous KL must beat all 20 nulls (`p <= 1/21`) and the same crossed
   key/block gates; estimated `24.4–26.3 h` per cell,
   `97.8–105.3 CPU h` for four, `120 h` hard wall.

**Next, in order:** finish and adjudicate `resSB` → `resAB` → `resSA2` without
opening a running/queued artifact early. Then run the X-free field on all four
cells as the cheapest direct moot-maker, followed by the Freedman–Lane null on
all four cells. Both precede the pinned second-model-family protocol; no arm or
layer is selected after outcomes.

Second lens: presentation sensitivity is proven locally under both registered
designs; presentation entangled with operational state remains unproven. The
current representation still lacks a demonstrated native quotient. The next
latent space must expose or controllably factor lexical, presentation, and
operational coordinates, define sameness by interchangeability under declared
moves and response laws, and support consequence-sensitive multi-step closure
with transfer across unseen words, fresh styles, and model families. The
single most sharpening immediate measurement is the fair residual-space
X-free presentation/lexical interaction field. No new axiom is earned, so
`theory/AXIOMS.md` is unchanged. No experiment was run in Round 27.

## Round 28 — B-static adjudication and two-sentinel static result (2026-08-29)

`analysis_resSB.json` is contract-valid for the registered B-static
residual-vs-null question: sentinel B (`,`), `P_static`, two unseen-word folds,
K=13, crossed class-preserving bootstrap, 20 shuffles, 500 bootstraps, support
`1.0`, and `4598.4 s` of the `7200 s` wall. F4/F8/F12/F20 pass; F0 fails. At
the passing layers residual ridge cosine is `0.558/0.564/0.517/0.578` versus
strongest-null `0.061/0.075/0.064/0.089`. Block-first cosine margins are
`+0.497/+0.489/+0.453/+0.489`, skill margins
`+0.349/+0.361/+0.405/+0.415`, and KL-rank margins
`+0.396/+0.419/+0.492/+0.577`, all with positive lower bounds. Full/positive
keys are `4/8 / 8/8`, `7/8 / 8/8`, `8/8 / 8/8`, and `8/8 / 8/8`; no block
collapses at F4–F20. Presentation-only `P_static -> Delta` cosine is
`0.628/0.508/0.413/0.444/0.622` at F0/F4/F8/F12/F20.

Round 23 ruling: A-static and B-static take the same registered `P_static`
non-collapse branch. Across both sentinels, static block/length/position
coordinates predict held-out raw displacement direction; after cross-fitted
removal from both `X` and `Delta`, `X_perp` still predicts `Delta_perp` and its
reassembled response-law consequence beyond the four registered residual
X-free lexical nulls at F4–F20. This establishes two-sentinel registered-
presentation sensitivity and residual X-linked predictability within one
decoder/template population. It identifies neither a presentation fraction
or mediation nor operational state or presentation independence, and it is
not independent replication, composition, a native law, or cross-family
generality.

F0 is heterogeneous and non-qualifying. Its block-first cosine margin is
`+0.267 [LB 0.050]`, but skill is `-7.465 [-26.697]` and KL-rank lower bound
is `-0.286`; only `4/8` keys are full or point-positive, and gloss plus
association collapse. The pooled skill is driven by association fold skills
near `-30`, while the other six folds are modest. Fold-mean reference KL is
finite, so this is local cellwise denominator ill-conditioning in normalized
skill, not a globally zero reference or a uniform `-7.5` effect. Do not
compare that magnitude across residualization cells. The joint consequence
gate still fails; the correct wording is **no qualifying conditional gain at
B-static F0 under this instrument**.

The repaired common-scale field is present for B-static. At F4–F20 every
cosine, skill, and continuous-KL residual/raw predictive-margin ratio median
exceeds `0.5`. Eleven of twelve 95% lower bounds exceed `0.5`; F4 continuous
KL is the exception at `0.426`. B-static therefore retains at least half of
the same-run raw ridge-versus-strongest-null predictive margin at the
bootstrap median on all three endpoints at F4–F20, without a uniform interval
claim. This is not a fraction of signal, variance, state, or mediation. No
joint A-static/B-static common-scale claim is licensed until `resSA2` fills
the missing A-static field.

Sentinel specificity: B-static's F4 strict count is `4/8`, weaker than
A-static's `7/8`, despite `8/8` B point-positive keys and no collapse. B and A
both have `7/8` full keys at F8; B has `8/8` at F12 versus A's `6/8`; both
have `8/8` at F20. B residual ridge cosine is about `0.03–0.06` lower across
F4–F20. These differences are recorded without changing any layer verdict or
selecting a favorable sentinel.

Second lens: the second sentinel makes A-specific idiosyncrasy less plausible
and reinforces local F0 identity/token dominance plus the absence of a
demonstrated native quotient. It proves no new hostile structural hole:
presentation/state inseparability, a presentation-free residual, family-only
law, non-composition, and inability of structured reasoning to live here
remain unproven. The current architectural/epistemic deficit is that analyst-
known coordinates are still required to propose equivalence; that is a
constructive target, not proof no quotient exists. No new axiom is earned.

**Next, unchanged:** finish and adjudicate `resAB` -> `resSA2`, then run the
fair residual-space X-free field on all four cells, then the fully refitted
Freedman–Lane null on all four cells, then the pinned second-model-family
protocol. The running/queued artifacts were not opened, the analyzer diff was
not modified, and no experiment was run in Round 28.
