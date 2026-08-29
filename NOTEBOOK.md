# NOTEBOOK

Reverse-chronological running log. Newest first. Each entry: what was done, what
was learned, what's next. Canonical state lives in STATE.md.

---

## 2026-08-29 — Re-contextualization #25 (2-hour step-back; audit #25 fired on the positive-control FAIL)

Project: the native mathematics of latent spaces. Live question (Round
36): can a latent world built from behaviour alone carry a well-defined
operational quotient and composable action table — and, after today, the
prior question: is the certification regime itself passable by any learned
artifact?

Whole picture. The constructive program has now produced four registered
outcomes in one day: v1 FAIL (confidence-confounded), 36b all-ineligible
with a cross-seed-stable one-step skeleton in W64, and a positive control
that fails every exact gate — worse than behaviour alone. Read together,
the honest picture is that nothing learned has passed the exact reducer,
and the one artifact designed to pass it made things worse. That is where
tunnel vision would say "add another control"; the audit is asked instead
whether the program is measuring its reducer rather than building a latent
space. Alternatives held live: (1) the control objective is defective
(moving-target supervision in the encoder's own coordinates; loss
interference) — then the gates are still unvalidated, not unreachable;
(2) the exact 12-cell 0.10/0.90 signature reducer is too strict for any
learned artifact — a certification-regime problem; (3) supervise the
signature/quotient directly, or freeze a behaviour-trained encoder and fit
a transition head, to obtain a learned artifact that passes; (4) change the
world (nontrivial quotient) so that identity actually compresses and the
test is about structure, not exactness. Foundational thread advanced: the
positive-control discipline itself — every gate now has a registered
reachability test, and the first one failed; that is a fact about the
gates the program did not have this morning. Audit #25's corrections and
ranked alternatives are appended verbatim when it lands.

## 2026-08-29 — Round 36c positive control (w32): FAIL on every exact gate

The learned, explicitly quotient-trained control (behavioural BCE + MSE of
the transition output to the stop-gradient encoding of the true successor
over all 176 canonical transitions; same carrier, seeds, reducer) finished
in 820 s and FAILS every exact gate in every seed — including action-table
truth (0/5 seeds) and cross-seed table identity, which the behaviour-only
36b W64 cell had passed informationally. result_scope = POSITIVE-CONTROL:
this is not a behaviour-only result and not a quotient-from-behaviour
claim. Registered meaning of a control FAIL: the exact gates are not
reachable by this carrier even with direct transition supervision — a
certification-regime problem, not a latent-organisation result. The
surprise is the direction: direct supervision made things worse than
behaviour alone, which points at the auxiliary objective (target chasing
through the stop-gradient successor encoding, or signature collapse) rather
than at the gates — that is with Codex as an adjudication. The registered
conditional w64 cell runs now (its precondition, a valid complete w32
exact-gate FAIL, is met). A fresh audit follows both cells.

## 2026-08-29 — Re-contextualization #24 (2-hour step-back; audit #24 already in flight on the only new claim)

Audit: the only new capability result since audit #23 is the Round 36b
ladder outcome; a fresh, unprimed auditor (#24) was fired on it the moment
the reducers finished and is still running — no second auditor is fired on
the same claim. Its corrections and alternatives are appended verbatim
when it lands.

Audit #24 replacement paragraph (verbatim; supersedes my interpretive paragraph, which prematurely called the line 'closed or near-closed' and equated exact-truth eligibility with the audit-#23 confidence defect):

Audit #24 upholds every Round 36b reducer status and finds no PASS, but it
does not close behavior-only quotient construction. W64 is exact on training
and all H2 held-out terminal rows; its `1–24` remaining errors per seed are
all H3, seed-variable, and share no single row across all five seeds. The
exact held-out gate was therefore not reached, but is not shown
unsatisfiable. Informationally only, W64 recovers all 16 canonical identities
and the complete truthful `16 x 11` action table identically across seeds.
It still fails exact rolled-representative descent, involution, closure, and
interchangeability even in the `p>0.5` diagnostic, so the result is a stable
canonical one-step skeleton rather than a certified quotient algebra. The
next registered increment is the learned, explicitly quotient-trained
positive control scored by the unchanged reducer; a separate prospective
approximate-inquiry branch may use exact training plus a fixed held-out
tolerance, while the original exact PASS remains unchanged.

The existing “Round 36b result” entry can remain after this audit is appended;
its informational/diagnostic labels are disciplined.

#### Audit #24 — W64 section, ranked next increments, and final verdict (verbatim)

## 2. What W64 does and does not mean

W64's ineligible primary flags are unusually informative. At the registered
support threshold, every seed passes:

- quotient availability: all 16 encoder signatures are supported and truthful;
- action-table truth: `176/176` canonical state/action cells;
- cross-seed action table: the five complete tables are identical and truthful.

This is stronger than “approximately 99% behavior.” It says that behavior-only
training found the same canonical one-step operational table across five
random initializations. The result is still informational rather than a
verdict because the frozen eligibility tree says so.

The opposite reading is equally important. At `p>0.5`, W64's ranges are:

| Gate | Per-seed range | Exact seeds |
|---|---:|---:|
| Quotient availability | `17/17` | 5/5 |
| Quotient well-definedness | `7540–9965 / 10560` (71.4–94.4%) | 0/5 |
| Toggle involution | `1724–3178 / 3776` (45.7–84.2%) | 0/5 |
| Swap/toggle table | `372–384 / 384` (96.9–100%) | 1/5 |
| H2 signature closure | `1167–1183 / 1184` (98.6–99.9%) | 0/5 |
| H3 signature closure | `643–972 / 1056` (60.9–92.0%) | 0/5 |
| Interchangeability | `50928–101706 / 132160` (38.5–77.0%) | 0/5 |
| Canonical action-table truth | `176/176` | 5/5 |
| Whole-table cross-seed identity and truth | `176/176` | PASS |

Thus the correct structural description is:

> **A cross-seed-stable canonical one-step skeleton emerged, but action on the
> full representative population did not become an exact well-defined,
> involutive, closed, interchangeable quotient action.**

“The latent is organized” is licensed only with that local/canonical scope.
“The exact structural gates are unreachable” is not licensed until a learned
positive control tests reachability. “The latent is unorganized” is contradicted
by the canonical table.

## 7. Ranked next registered increments

| Rank | Increment | Cost | Why / decision value |
|---:|---|---|---|
| 1 | **Explicit learned quotient-trained positive control** | Low–moderate implementation; roughly one five-seed CPU cell, likely `<15 min` after review | Use the same 8-D carrier, same width (run 32 first; 64 only if prospectively conditional), same seeds, same representatives, and unchanged exact reducer. Add direct state-transition or quotient-consistency supervision. PASS shows the learned architecture/optimizer can reach the certificate and localizes the behavior-only gap to the objective. FAIL says the architecture or gate regime is itself the immediate problem. The affine fixture is not this control. |
| 2 | **Separate exact certification from approximate-structure eligibility** | Near-zero compute for a reducer design/replay; medium governance; a new prospective behavior run costs about 10–15 CPU min | Keep the original exact PASS unchanged. Add a distinct inquiry branch requiring exact `21184/21184` training in every seed, exact H2 terminal behavior, and at least `1046/1056` (99.0%) H3 terminal behavior in every seed. It may report structural rates but can never emit the exact PASS. This threshold is a transparent post-36b successor rule and must not retroactively reclassify W64; W64 would still miss it in seeds 11 and 71. |
| 3 | **Learned lookup baseline** | Very low; minutes | Fit handle × observed-word behavior with a frozen default for unseen spellings. Exact train plus poor H2/H3 establishes the memorization floor; comparing it with W64 quantifies what the shared transition gained. Run it alongside rank 2 if convenient. |
| 4 | **Genuinely nontrivial quotient world** | Medium design and implementation; approximately 30–60 CPU min after controls | Add nuisance bits or duplicate hidden states with identical response futures and require independent representatives to collapse. This is the first world that tests quotient formation rather than recovery of a singleton 16-state identity table. It becomes interpretable only after rank 1 establishes gate reachability. |
| 5 | **Longer/wider behavior-only cell** | Low code cost but another 15–30+ CPU min; low information gain | W64 already saturates training and canonical action truth while rolled structural errors remain large. More scale would again conflate optimizer luck, capacity, and gate reachability. Register only as a later sensitivity after ranks 1–3, not as the next move. |

The concrete next registration should therefore be the learned positive
control. The approximate eligibility branch is the next **certification-rule**
increment, but it is not a rescue and does not replace exact PASS.

## Final verdict

- **Mechanical four-cell status:** **UPHELD.**
- **No eligible cell / no PASS:** **UPHELD.**
- **“Behavior underfit” as the registered status:** **UPHELD, but for W64 say
  held-out exactness missed after exact training.**
- **“Exact-held-out eligibility is unsatisfiable by construction”:**
  **REJECTED AS UNPROVEN.** Gate reachability is unvalidated.
- **W64 canonical organization:** **FOUND, INFORMATIONAL ONLY.** Exact encoder
  identity and exact truthful cross-seed canonical action table.
- **W64 exact quotient/action algebra:** **NOT FOUND.** Rolled descent,
  involution, closure, and interchangeability remain non-exact.
- **Over-claimed WIN in result notebook/ledger wording:** **NOT FOUND.**
- **Premature closure / stale public state:** **FOUND.** Re-contextualization
  #24 needs narrowing; README, STATE, and project memory need propagation.
- **Next registered increment:** **explicit learned quotient-trained positive
  control, before further scaling.**

## 2026-08-29 — Round 36b result: every cell BEHAVIOR UNDERFIT; QUOTIENT INELIGIBLE

All four cells completed inside their walls (174 / 606 / 618 / 696 s) and
every cell returns the registered primary status "FAIL — BEHAVIOR UNDERFIT;
QUOTIENT INELIGIBLE". Behavioural fit (train correct / 21,184; held-out /
2,240; five seeds): S16 train 20,894–21,184 with one exact seed, held-out
2,179–2,218; S64 train 21,078–21,184 with two exact, held-out 2,184–2,226;
LR64 train 21,088–21,184 with four exact, held-out 2,198–2,225; W64 train
exact on all five seeds, held-out 2,216–2,239 (98.9–99.96%) — none exact.
Under the registered rule no cell is eligible for a quotient
interpretation, so no PASS, no FIT-BUT-NON-CONGRUENT, and no reading of the
DIAGNOSTIC tables as verdicts. Informational only: W64's ineligible gate
flags show action-table truth, cross-seed action-table identity and
quotient availability passing while well-definedness, involution, the
swap/toggle table, held-out closure and interchangeability fail. Plain
reading: more budget and width move behaviour toward exact fit (W64 is
exact on training) but held-out spellings still miss by 1–24 rows per seed,
so the ladder never reached the point where the quotient question could be
asked; what to make of that — exact held-out fit as a precondition may
simply be unreachable for this recipe on unseen spellings — is with the
fresh auditor (corrected by audit #24: eligibility not reached under this
ladder; reachability of the exact learned certificate unvalidated, not
unsatisfiable). Row-level evidence (≈170 MB per cell) and weights are
retained locally and hash-pinned; only config/manifest/verdict are
committed (`abef6cf`).

## 2026-08-29 — Round 36b launched under lock V3 (review #2 RUN-READY)

The behaviour-fit ladder runs now: four cells (S16 16k steps; S64 64k; LR64
64k at lr .001; W64 64k at width 64), five seeds each, sequential on one
CPU process, then four separate reducers. Before any outcome existed: the
audit-#23 amendment (three-stage primary status; DIAGNOSTIC-only p>0.5
table; cellwise cross-seed accounting; depth traces) was registered
(`9edb892`) and implemented; the lock-review defect (eligibility from
producer aggregates) was closed by row-level logit replay; lock V3 recorded
(`ff8eaa7`); review #2 returned RUN-READY with dynamic probes of every
status branch and a byte-identical v1 fixture; runner and configs
committed (`61e2430`). Outcomes are not inspected until all four producers
finish. Status of the design, verbatim from audit #23: a prospectively
locked, post-outcome, outcome-informed successor — exploratory, not
confirmatory; a PASS would show operational recovery and congruent action
maps in a finite world, not compression into a nontrivial quotient.

## 2026-08-29 — Re-contextualization #23 (2-hour step-back; audit #23 fired on the Round 36 FAIL)

Project and live question: the native mathematics of latent spaces; the
constructive question is now whether behaviour alone can make a latent
world's places and moves well-defined (an operational quotient with a
composable action table) — Round 36 — with NLM-007 closed behind its
closing statement.

Whole-picture check: the day converted a stalled instrument program into
(i) a closed, honestly bounded line and (ii) a runnable distance-0
artifact that ran in under a minute and FAILED. That FAIL is the first
result of the constructive program and it is where tunnel vision would be
most dangerous: the adjudication reads it as under-fitting, and the
registered successor (36b) adds training budget with an exact-fit
eligibility rule. Alternatives held live and put to the fresh auditor:
(1) the 12-cell all-supported signature rule turns a 98%-calibrated model
into a support failure by arithmetic (≈21% of rows fail support even with a
perfect latent) — the gate may be measuring confidence, not structure;
(2) the exact-fit eligibility rule could be a post-hoc rescue, and exact
fit could let a lookup-table-like fit pass; (3) the opposite under-read:
0/176 cross-seed action-table agreement may mean there is no composable
structure at all even where behaviour is right (corrected by audit #23: the
stored 0/176 is an all-or-none whole-table gate, not a cell count — 11/176
cells identical at the registered thresholds, 112/176 at p>0.5, 175/176 by
majority). Foundational thread
advanced: the constructive program now has a real falsifier loop
(artifact → FAIL → adjudicated cause → preregistered successor), which is
what the constitution demanded and NLM-007 never had. Audit #23 (fired, unprimed) returned: valid registered FAIL; behaviour-, calibration- and exactness-confounded; substantial but imperfect operational structure remains (a confidence-free replay at p>0.5 recovers 84.1-98.9% of the one-step action table while every exact gate still fails); the licensed sentence is REPLACED, the '0/176' phrase is withdrawn as misleading bookkeeping, and Round 36b's status logic is NOT READY AS WORDED (three-stage decision required). Replacement paragraph, sections 4-6 and the final verdict follow verbatim.

> Audit #23 upholds the v1 artifact and its all-gate FAIL but narrows the
> meaning. Under the exact 4,000-step recipe, the artifact failed the registered
> confidence-qualified certificate for a fully supported operational identity
> and exact composable action algebra. It did not reach exact behavioral fit,
> and the 12-cell `0.10/0.90` support conjunction strongly amplifies marginal
> confidence defects. A read-only `p>0.5` diagnostic nevertheless recovers
> `84.1–98.9%` of the one-step action table per seed; `112/176` cells are
> identical and truthful across all five seeds, while every exact structural
> gate still fails. Therefore neither “pure calibration failure” nor “no
> composable structure” is licensed. The v1 result is a permanent,
> recipe-specific nonpass with substantial but imperfect structure. Round 36b
> is a transparent post-outcome successor and cannot rescue or overturn it.

#### Audit #23 — sections 4-6 and final verdict (verbatim)

## 4. Strongest alternative explanation

The strongest single explanation is not “the latent has no algebra.” It is:

> The BCE objective, finite sampling distribution, and 4,000-step stop learned
> a mostly correct, partially compositional response system, but left a small
> number of low-margin or wrong response cells. The 12-way confidence
> conjunction and exact all-cell/all-seed reducer magnified those local defects
> into universal gate failures. The remaining seed-dependent errors, especially
> at depth 3 and rolled representatives, show that the transition law itself is
> also incomplete.

This explains every row more economically than either pure-calibration or
no-structure narratives. The population also heavily weights length-3 words:
training contains 1 empty, 11 one-step, 47 two-step, and 1,265 three-step words.
Only `176/21,184` training rows directly supervise the empty/one-step action
signature. More optimization may help, but the ladder does not distinguish
budget from depth weighting or objective geometry.

## 5. Tunnel-vision ruling

The constructive program is scientifically tunnel-visioned despite being much
closer to its claim than NLM-007:

- one 16-state toy;
- one binary response sensor;
- one learned handle table and one residual transition architecture;
- one optimizer family;
- one action algebra;
- one horizon (`<=3` for behavioral rows, with registered rolled probes);
- a singleton oracle quotient, because depth-1 signatures distinguish all 16
  hidden states.

That last point is decisive. A PASS would show bounded state recovery and a
congruent action table, not nontrivial quotient formation. There are no two
different hidden simulator states that the denizen must identify as one place,
and no nuisance state that the quotient must discard.

## 6. What should run alongside Round 36b

### Register before any 36b outcome

1. **Confidence-free diagnostic reducer.** Keep the `0.10/0.90` primary gate
   frozen, but prospectively report the complete `p>0.5` gate table, component
   error counts, and margins. It is diagnostic only and cannot rescue a primary
   FAIL.
2. **Literal cellwise cross-seed accounting.** Report (a) identical cells,
   (b) identical supported cells, (c) all-five truthful cells, and (d) bitwise
   majority truth, beside the existing whole-table exact gate.
3. **Three-stage decision status.** Separate behavior underfit, signature
   underconfidence, and supported non-congruence as above.
4. **Depth-balanced diagnostic.** Either add a prospectively frozen
   depth-balanced sampling arm or, minimally, report loss/accuracy/support by
   word depth throughout training. The current four-cell ladder changes budget,
   learning rate, and width but never tests the severe depth imbalance.

### Orthogonal controls

5. **Learned lookup baseline.** A handle-by-observed-word memorizer should fit
   train and fail held-out spellings; this calibrates how much closure comes
   from composition rather than finite lookup.
6. **Explicit finite-state/quotient-trained positive control.** Train the same
   carrier with direct state-transition or quotient-consistency supervision,
   scored by the unchanged reducer. The existing fixture is oracle-authored,
   not a learned representability/control arm. If the explicit control passes
   and behavior-only training fails, the gap belongs to the learning objective,
   not representability.
7. **A genuinely nontrivial quotient world.** Add nuisance hidden bits or
   duplicate simulator states with identical response futures, require several
   hidden states to collapse into each operational place, and demand action
   descent across those independently generated representatives.
8. **Longer, algebraically novel continuations.** Hold out depth 4–6 and word
   families selected by algebraic relation, not only spelling hashes, to attack
   finite-horizon lookup and behavioral redundancy.
9. **A second transition architecture.** A linear/affine action model or a
   small recurrent alternative should be fixed before outcome. One
   architecture cannot distinguish a world property from an inductive-bias
   accident.

A 36b PASS should trigger a fresh preregistration on the nontrivial-quotient
world, not immediate activation of Round 35.

## Final verdict

- **Claim (a), mechanical FAIL:** **UPHELD.**
- **Claim (a), “incomplete behavioral fit”:** **UPHELD, with optimization not
  proven as the sole cause.**
- **Licensed sentence:** **REPLACE** with the confidence-qualified
  non-certification wording above.
- **Over-claimed KILL:** **FOUND.** The primary reducer materially conflates
  confidence/support with structure; the current prose suppresses strong
  approximate action-table recovery.
- **Under-read FAIL:** **ALSO FOUND.** Confidence-free exact composition and
  cross-seed invariance still fail; the artifact is not merely underconfident.
- **Claim (b), successor legitimacy:** **UPHELD only as a transparent,
  exploratory post-outcome successor.** It is not a v1 repair and not
  confirmatory.
- **Round 36b exact-fit/non-congruence rule:** **NOT READY AS WORDED.** Split
  calibration from supported non-congruence before any 36b interpretation.
- **Tunnel vision:** **FOUND.** A singleton quotient in one toy and one
  architecture cannot carry the constructive program alone.

## 2026-08-29 — Round 36 first run: the constructive artifact exists and FAILS every gate

The first distance-0 artifact ran end to end: `produce` (five registered
seeds, CPU only, one process) completed non-claiming in 52.6 s (train 41.7 s,
evidence 11.0 s; wall 900 s); the separate `reduce` returned FAIL on every
gate — quotient availability, quotient well-definedness, toggle involution,
swap/toggle table, held-out depth-2 and depth-3 closure, interchangeability,
action-table truth (0/5 seeds; 14–56% of 176 cells), cross-seed action table
(0/176 identical — corrected by audit #23: an all-or-none whole-table gate,
not a cell count). Signatures carry many unsupported ("?") responses.
Registered meaning: this training recipe did not produce a well-defined
operational quotient in this latent space — a constructive hole here, not a
hostile hole in general. The obvious alternative reading is the boring one:
the recipe underfit (a 1,041-parameter model trained ~8 s per seed may not
have fit the behavioural data at all), in which case the quotient gates were
never really eligible. That question — underfit vs fit-but-non-congruent vs
gate construction — is with Codex as an evidence/design ruling, together
with what the registration permits next WITHOUT outcome-contingent tuning
(a preregistered budget ladder with a behaviour-fit eligibility criterion
frozen before any 36b outcome). No tuning has been done. Artifacts committed
(`073037f`).

Adjudicated (Codex, `.codex_round36_adjudication1.md`): classification (a)
— incomplete behavioural fit under the frozen recipe (train accuracy
96.6–98.5%, held-out 97.0–98.3%, depth-3 93.8–96.3%, loss still falling at
step 4,000; the 12-cell support requirement amplifies residual errors into
1–58% support). The v1 FAIL stands permanently; licensed claim, verbatim:
"Under the exact 4,000-step v1 recipe, the produced latent artifact did not
supply its denizen with a fully supported operational identity or
composable action algebra." (Corrected by audit #23: this sentence is
REPLACED by the audit #23 paragraph in the entry above — say: failed to
certify the exact confidence-qualified algebra.) Round 36b is preregistered (`f9dea33`) as a
successor design, not a repair: a four-cell behaviour-fit ladder (S16,
S64, LR64, W64), every cell run and visible, quotient gates eligible only
at exact behavioural fit (21,184/21,184 train, 2,240/2,240 held-out),
otherwise "FAIL — BEHAVIOR UNDERFIT; QUOTIENT INELIGIBLE"; exact fit
followed by a quotient failure would be the first legitimate
FIT-BUT-NON-CONGRUENT result. Configs and runner revision are hash-locked
before any 36b outcome.

## 2026-08-29 — NLM-007 closed: audit #22 upholds the terminal stop; closing statement adopted verbatim

NLM-007 is closed under the program’s terminal allocation rule, not by a scientific null.

Within one pinned decoder and authored population, the correlated A/B punctuation sentinels established bounded F4–F20 condition robustness: the qualified four-cell ridge table retained X-linked predictive separation across held-out blocks and words. The raw token-context comparator was highly non-robust to P_static residualization and therefore P_static-aligned in this fitted design. Round 34a found a small but systematic raw separation surviving the registered EDF match (+0.04–0.08 cosine), while the larger static separation was not eliminated by that match within the fixed feature classes. Round 34b did not resolve the interpretation: P+C improved on P by roughly +0.02–0.04, defeating the redundancy STOP, but C_perp→Δ_perp failed the joint retention gate; every eligible layer and the joint reducer were INCONCLUSIVE. Under the pre-adopted ruling, that is an allocation stop, so Round 34c does not run.

This line did not identify operational state, a denizen-usable or native law, composition, a representation-level hostile hole, or independent replication. It leaves frozen captures; raw/static, matched-EDF, and partial-overlap analyzer modes; hash-bound cell sidecars; block-first held-out evidence; and fail-closed producer/reducer discipline.

Round 36 now asks the constructive question directly: can behavior alone support a well-defined operational quotient and composable action table in the minimal 16-state world?

#### Audit #22 — Round 34b wording corrections, EDF-correction ruling, and Round 36 handoff (verbatim)

### Round 34b interpretation

The numerical shorthand needs slight tightening:

- `P+C − P` cosine is A `+0.0178–+0.0373` and B `+0.0238–+0.0355`. A/F4 falls just below the point threshold, but its upper interval exceeds `0.02`; the other means exceed `0.02`. Thus the redundancy STOP fails.
- Residual-context cosine is approximately `+0.019–+0.089`, not quite `+0.03–+0.10`.
- Residual normalized-error gain is negative for every ridge/kernel, sentinel, and F4–F20 cell. Clustered key/block requirements also fail. Thus retention fails.

The positive `P+C − P` increments are evidence **against the registered strict-redundancy account**, but not evidence for operational state: `P+C` has greater capacity, while the residual partial relation fails its joint gate.

### EDF correction

The correction is mathematically justified. The producer sums all nonnegative eigenvalues for EDF but defines rank only above tolerance; therefore EDF can exceed numerical rank by a small sub-tolerance tail. The old fit bound is violated in eight selected-state fits, all at excluded F0, by only `2.93–2.94×10⁻⁵`. No eligible F4–F20 fit violates it.

Producer JSON, sidecars, reductions, and gate functions were unchanged. Strictly, the old joint reducer had no verdict—it was `INCOMPLETE`. The repair changed reducer status to `COMPLETE`, while preserving the already-recorded sentinel decisions and recomputing the same joint `INCONCLUSIVE`.

Do not call the repair literally outcome-blind: it was triggered after seeing the artifacts. The defensible wording is **post-outcome but not outcome-selective**. Its formula follows the pre-existing producer definition, applies symmetrically, and affects only diagnostic F0 telemetry.

## Overclaim and underclaim audit

- Round 34a raw is a genuine registered survival, but small: capacity matching removed most of the unmatched gap. Its lower bounds clear zero, not uniformly `0.02`.
- Round 34a static supports only “not eliminated by the registered EDF match within these fixed feature classes.” It does not prove capacity independence or feature adequacy.
- `ctxS` supports high non-robustness of the raw context comparator to `P_static` residualization—hence `P_static` alignment in this fitted design—not presentation share, mediation, or causal explanation.
- The four-cell table supports qualified within-decoder condition robustness. It remains correlated, same-population evidence; B-score4’s KL-rank/low-rank qualification and F0’s model-class sensitivity remain.
- Closing now is correct under the prospectively adopted allocation constitution. Scientifically, it deliberately leaves the item-by-carrier fingerprint/local-Jacobian explanation unresolved. Round 34c might have clarified that account, but the constitution explicitly forbids escalating an `INCONCLUSIVE` rung.

One durability defect remains: `.codex_audit21.md` is only a 327-byte self-referential completion stub. Audit #21’s substantive text survives in `NOTEBOOK.md`, `STATE.md`, and the ledger, but the named output itself should not be treated as evidence.

## Round 36 handoff

**Tunnel vision:** NLM-007 remained scientifically tunnel-visioned around increasingly refined readers of one punctuation relation. Closing it is correct.

**Strongest alternative:** Round 36 could fit the finite response table without organizing a composable latent world. Also, its depth-1 signatures distinguish all 16 simulator states, so the oracle quotient has singleton hidden-state classes. A PASS demonstrates operational recovery and congruent action maps—not compression into a nontrivial quotient.

**What should run:** first close Round 36 review #1’s CLI, wall-metadata, and fixture-isolation blockers. Then run the registered reducer fixture. After it returns `PASS / INVALID / INVALID / FAIL` on the exact and three mutation cases, run the exact five-seed CPU producer and separate reducer—no pilot or seed replacement.

The first scientific falsifier is action descent on rolled representatives: if any two supported points with the same `Σ₁` signature reach different or unsupported quotient classes under one primitive action, the quotient action is not well-defined. Held-out H2/H3 interchangeability should follow immediately to attack the response-memorization alternative.

Blackboard entries e673–e683 were recorded; convergence reached 100%, and synthesis was read. This audit edited no project source/result file and made no commit.

## 2026-08-29 — Round 34b result: INCONCLUSIVE in both sentinels — the terminal rung

`analysis_ctxoverlap_A.json` (444 s) and `analysis_ctxoverlap_B.json`
(595 s), static estimand, producers run-ready. Every F4–F20 layer is
INCONCLUSIVE in both sentinels. Sentinel A (block-first means, F4/F8/F12/
F20): P_static→Δ alone reaches cosine 0.49/0.43/0.46/0.58; the token-context
ridge alone 0.51/0.47/0.50/0.62; the nested P+C 0.51/0.46/0.49/0.61 — so
P+C − P ≈ +0.02 to +0.04, which fails the redundancy STOP (needs ≤ 0.02 with
crossed UB < 0.02) — while C⊥→Δ⊥ keeps only ≈ +0.03 to +0.10 cosine by
block, which fails the retention rule. Reading under the registered rules:
the registered raw context field is neither P_static-redundant nor clearly
retaining residual signal in this design; neither the "by construction"
nor the "fitting-artefact" account is licensed. Under the continuation
ruling an INCONCLUSIVE rung is an allocation stop: the NLM-007 terminal
ladder ends here (pending the joint artifact, whose reducer currently
rejects the valid producer artifacts on a rank/EDF telemetry bound — a
bounded reducer repair is with Codex; producers untouched). Round 34c does
not run. NLM-007's closing statement is drafted after the joint and the
next fresh audit.

Joint (reducer repaired — the EDF≤rank bound was producer-inconsistent by
~3×10⁻⁵ at F0's state_selected fit; producers correct, no rerun):
`analysis_ctxoverlap_joint.json` COMPLETE/SCREEN-ONLY, decision
INCONCLUSIVE, no common retaining layer (ridge or kernel), no common stop
layer. The terminal ladder therefore ends at Round 34b. Audit #22 fired on
the terminal outcome and on the draft closing statement.

## 2026-08-29 — Audit #21 adversarial correction: both 34a verdicts upheld, claim boundaries tightened

The four float32 evidence sidecars replay exactly through the registered reduction and decision code. RAW and STATIC both return `CONTINUE` at F4/F8/F12/F20 in both sentinels, with 8/8 jointly positive keys at every eligible sentinel-layer. F0 is correctly `INCONCLUSIVE` and diagnostic because at least one required context-EDF target exceeds the selected F0 state EDF, making a downward match undefined; F0 is excluded from the ladder gate.

RAW is a valid registered `CONTINUE`, not a numerical boundary artefact. The `0.02` threshold applies to the point margin; the lower-bound criterion is `>0`. The smallest raw point over cosine/nerr is 0.0397, the smallest lower bound is 0.0146, and float32 resolution is immaterial at that distance. The replicate-wise minimum over the four predeclared candidates is conservative for survival, not multiplicity inflation. Correct wording: capacity matching removed most of the unmatched raw gap but did not exhaust it; the +0.04 to +0.08 cosine separation is small in magnitude but systematic within this locked design, with both endpoints positive in all eight keys at every F4–F20 layer in both correlated sentinels. It is not a state claim.

STATIC also mechanically survives, but withdraw the provisional sentence “the residual separation is not a capacity artefact.” The selected contextual ridge target is approximately 47 EDF throughout; the selected kernel is approximately 48 EDF at F8–F20 but falls to approximately 4.36 in 4/8 A and 2/8 B F4 keys. The selected state ridge ranges from approximately 202 to 384 EDF and is therefore heavily shrunk for the comparison. Honest wording: “the residual predictor separation was not eliminated by the registered EDF match within these fixed feature classes.” This rejects a simple unmatched-slope-EDF explanation, but the fixed context arm is near-null on `Delta_perp`, equal EDF does not equal feature adequacy, and the item-by-carrier fingerprint/local-Jacobian account remains live. No operational state, native law, or representation-level hostile hole is identified.

Both 34a estimands returned `CONTINUE`, so the terminal ladder may proceed to 34b only after its final bounded `RUN-READY`; 34c remains conditional on a 34b `CONTINUE`. These results make neither control moot and reopen none of the cut queue. Round 36 remains the higher-leverage constructive line.

#### Audit #21 — section 6 (tunnel vision, strongest alternative, run order) and final ruling, verbatim

## 6. Tunnel vision, strongest alternative, and what should run

### Tunnel-vision ruling

The program remains tunnel-visioned at the scientific level even though the
continuation ruling has now contained the allocation error. These outcomes
refine one predictor comparison in one decoder, one authored micro-world, two
punctuation tokens, one append move, and correlated depths. The static margin
does not justify another control family beyond the already terminal 34b/34c
ladder.

The strongest live scientific alternative explanation remains:

> `P_static` removes a coarse block/length/position response, while `X_perp`
> retains a dense item-by-carrier activation fingerprint and a local
> punctuation Jacobian. A low-EDF ridge can extract a few high-signal directions
> from that learned nonlinear feature map. The registered token-context field,
> even at its rank ceiling, omits the item token and dense interactions and is
> near-null on `Delta_perp`. No denizen-usable state or operation is required.

The strongest program-level alternative is already registered: Round 36's
minimal operational quotient. It directly tests whether identity and actions
descend to a behaviorally available quotient in a runnable world, instead of
perfecting another reader of the punctuation relation.

### Concrete, cheap run order

1. **34b first, if and only if RUN-READY.** Expectation: determine whether raw
   context is `P_static`-redundant and whether residual context retains signal
   under the fully nested projection. A redundant result stops the ladder; a
   retained-signal result continues but revises the static interpretation.
   The simplest fatal confound is nuisance/vocabulary reuse across an inner or
   outer held-out boundary.
2. **34c only after 34b CONTINUE and RUN-READY.** Expectation: test the
   item-by-context fingerprint account with the registered richer X-free field.
   Closure means feature sensitivity, not causal context; survival still does
   not identify state. The simplest fatal confound is PCA or vocabulary leakage
   from held-out words.
3. **Keep Round 36 moving as the higher-leverage constructive line.** Run its
   reducer fixture before learned evidence, then the CPU producer and separate
   reducer, one process at a time. No NLM-007 result changes Round 36.
4. **Run no new NLM-007 arm.** The five-seed bootstrap replay above is enough
   to answer the immediate numerical-boundary worry as an audit diagnostic.
   A 10,000-bootstrap polish, random-weight decoder, second decoder, or richer
   context family would be more measurement infrastructure and violate the
   terminal allocation ruling.

## Final ruling

- **Raw mechanical verdict:** upheld.
- **Raw claim:** small but systematic within-design survival; not a state claim.
- **Raw correction:** lower bounds clear zero, not necessarily 0.02.
- **Candidate multiplicity:** conservative for `CONTINUE`; synthetic-oracle
  wording boundary remains mandatory.
- **Static mechanical verdict:** upheld.
- **Static claim:** not eliminated by this registered EDF match; do not call it
  generally capacity-free.
- **Static telemetry correction:** kernel selected EDF has material low-EDF F4
  exceptions; selected state EDF varies from about 202 to 384.
- **F0:** correctly undefined/diagnostic and excluded from the ladder.
- **Queue:** 34b if finally `RUN-READY`, then 34c only on 34b `CONTINUE`; no
  cut item reopens.
- **Tunnel ruling:** finish the bounded closeout without adding arms; prioritize
  Round 36 as the distance-0 constructive program.

Blackboard findings e649–e655 were recorded with provenance. `bb_convergence`
returned 100% with no open signals, disputes, unread documents, or partial
documents, and `bb_synthesis` was read before this verdict. No project source
or tracked file was edited, and no commit was made.

## 2026-08-29 — Round 34a STATIC result: CONTINUE at F4–F20 with large matched margins; the "match" sits at the context rank ceiling

`analysis_ctxcapA_static.json` / `analysis_ctxcapB_static.json` (frozen
analyzer copy; 314 s / 304 s) and the static joint (`analysis_ctxcap_static_joint.json`,
re-run on the main analyzer after the NaN-replay fix: COMPLETE/SCREEN-ONLY,
CONTINUE, common layers F4/F8/F12/F20). On the P_static-residualized
relation the strongest matched margin is: A cosine +0.306 / +0.383 / +0.373
/ +0.435 at F4/F8/F12/F20 (LBs 0.227 / 0.315 / 0.305 / 0.353), nerr +0.047
/ +0.089 / +0.084 / +0.115; B cosine +0.329 / +0.352 / +0.337 / +0.367 (LBs
0.262 / 0.278 / 0.275 / 0.264), nerr +0.065 / +0.082 / +0.077 / +0.100; 8/8
keys; F0 INCONCLUSIVE (diagnostic).

What the match telemetry says: on this relation the context arms saturate
at their rank ceiling (target EDF 47 for the ridge, 48 for the kernel) while
the selected residual state ridge has EDF ≈ 267 — so the "matched" state
arm is the residual ridge shrunk to 47–48 df, and it still keeps ≈ 0.35
held-out cosine that the token-context arms cannot supply at any attainable
capacity. Provisional reading (audit #21 fired on both forms, wording
pending): the residual separation is not a capacity artefact; the raw
separation mostly was (raw matched margins +0.04–0.08). Neither is a state
claim — feature adequacy (item-by-context) is untested until 34c. Both
estimands returned CONTINUE, so under the continuation ruling the ladder
proceeds to 34b (conditional on its final RUN-READY), then 34c.

## 2026-08-29 — Round 34a RAW result: CONTINUE at F4–F20, but the matched margin is small

`analysis_ctxcapA_raw.json` / `analysis_ctxcapB_raw.json` (frozen analyzer
copy 6b93ff1; 291 s and 254 s; tokenizer only, no model forward) and the
raw joint reduction (`analysis_ctxcap_raw_joint.json`: COMPLETE/SCREEN-ONLY,
CONTINUE, common layers F4/F8/F12/F20). With the state ridge bisected down
to the contextual arm's effective df, the strongest matched margin is:
sentinel A cosine +0.072 / +0.057 / +0.045 / +0.042 at F4/F8/F12/F20 (crossed
LBs 0.034 / 0.024 / 0.019 / 0.024), normalized error +0.073 / +0.047 / +0.040
/ +0.054; sentinel B cosine +0.082 / +0.064 / +0.047 / +0.043 (LBs 0.049 /
0.034 / 0.023 / 0.022), nerr +0.088 / +0.054 / +0.042 / +0.067; 8/8 keys
jointly positive at every F4–F20 layer; F0 INCONCLUSIVE (the
selected-context-EDF match is undefined there; diagnostic only). The
strongest arm is the token-id kernel at most layers.

Plain reading: capacity matching removed most of the unmatched raw gap
(ctx_A/ctx_B cosine margins were +0.11 to +0.20); what survives is a
+0.04 to +0.08 cosine separation with lower bounds just above the 0.02
threshold at F12/F20. CONTINUE by the registered rule — a narrow survival,
not a strong one, and not a state claim. Interpretation waits for the
static form (running now) and the fresh auditor. One reducer defect
surfaced and was fixed on the main analyzer without touching the producer
(a stored NaN at the F0 diagnostic compared unequal to its replay); the
joint was re-run in seconds.

## 2026-08-29 — Parity verdict: refactored analyzer reproduces HEAD; Round 34a runs begin

The HEAD-vs-refactor CPU parity check (contextual-prefix static screens,
sentinels A and B, committed analyzer copy vs the parked branch's
refactored analyzer, decision JSON scrubbed of timing/SVD/shadow fields)
returned IDENTICAL for both sentinels. The parked consequence branch's
legacy-parity question is therefore answered in its favour; the gate itself
is cut by the continuation ruling and the result is kept as evidence only.
The Round 34a closeout ladder started immediately on the frozen analyzer
copy: ctxcapA_raw is running (raw B, raw joint, static A/B, static joint
follow; minutes each).

## 2026-08-29 — Re-contextualization #22 (2-hour step-back; audit skipped — still no new claim)

No Round 34a outcome exists yet (the parity screens are on their last
layer), Round 34b/34c are in repair round 2 of 3, and no new result has
been claimed since audit #20; the fresh auditor is held for the first 34a
outcome. Live question unchanged; direction unchanged (cheap capacity /
feature-adequacy ladder first, expensive instruments held or parked).

What this pause is used for: the one review the constitution requires per
cycle that has NOT been fired explicitly during this long instrument phase
— "should this program continue at all, and is this the highest-leverage
thing to be doing?" (global CLAUDE.md §2.7 rule 5). A fresh Codex session is
asked exactly that, portfolio-level, with Round 35 (typed truth-evaluable
world) on the table as the constructive alternative and the audit #19/#20
alternatives (random-weight architecture null; item-by-context null) as the
cheap moot-makers. Its answer is recorded verbatim below when it lands.

Its answer landed and is adopted as the Codex program ruling (verbatim
below): STOP NLM-007 as an open-ended program — infrastructure drift by the
constitution's own tripwire (measurement:artifact ≥ 6:1) — and run only a
terminal closeout ladder: Round 34a raw and static once; 34b then 34c only
on CONTINUE and only if their final bounded repair is RUN-READY; full Round
34, Round 33 (branch archived), the parity gate, the random-weight null and
a second decoder are cut. Any INCONCLUSIVE rung is an allocation stop. Round
35 becomes a requirements envelope; the next artifact is a minimal
operational-quotient / bisimulation world on the 16 four-bit states (design
gate opened). Governance: mandatory producer/reducer separation.

#### Program continuation ruling (Codex, verbatim)

# Program continuation ruling — NLM-007

## Executive ruling

**STOP NLM-007 as an open-ended research program.** It is in infrastructure drift under the project constitution.

Authorize only a tightly bounded **terminal adjudication** using the already-built cheap screens. Regardless of whether those screens return STOP, INCONCLUSIVE, or CONTINUE, NLM-007 then closes. A CONTINUE result may earn a narrow measurement claim; it does not justify full Round 34, Round 33, another decoder, or further work on this punctuation relation.

The decisive issue is not that NLM-007 failed. It is that it has already yielded its transferable result—bounded within-decoder condition robustness plus several instrument boundaries—while further work increasingly measures the measurement rather than building the README’s stated artifact: a native mathematics of latent spaces ([README.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/README.md:3)).

## 1. Distance-from-claim accounting

The constitution defines distance 0 as the central artifact, 1 as something it directly needs, 2 as a measurement, and 3+ as measurement infrastructure. It requires a raise above 2:1 and a halt above 5:1 ([CLAUDE.md](/C:/Users/devan/.claude/CLAUDE.md:143)).

| Workstream | Distance | Reason |
|---|---:|---|
| SVD telemetry gate | 3+ | Numerical telemetry and backend validation for one candidate inside an already secondary measurement endpoint. |
| Round 33 consequence instrument | 3+ | Implementation, compatibility, provenance, parity, wall, and reducer machinery for an unrun measurement. The eventual run would be distance 2. |
| Full Round 34 | 3+ | Six-arm measurement apparatus and custom claiming reducer; no outcome exists. |
| Round 34a | 3+ so far | Design, implementation, evidence sidecars, reducer, fixtures, and four reviews. Its queued run would be distance 2. |
| Round 34b | 3+ so far | Custom partial-overlap measurement apparatus and reviews. |
| Round 34c | 3+ so far | Custom item/context comparator, PCA provenance, EDF telemetry, reducer, and reviews. |
| Round 35 docs-only design | 1, generously | Directly specifies a possible constructive artifact, but nothing runnable exists and no population has been authored. |
| Central runnable mathematics artifact | 0 units | No native law, operational quotient, composition law, new axiom, or representation-level hostile hole was produced. |

Conservative workstream ratio:

- Measurement/infrastructure units: **6**
- Artifact-facing units, counting the docs-only Round 35 design generously: **1**
- Ratio: **6:1**

If “artifact” means the constitution’s runnable central artifact, the denominator is zero and the ratio is unbounded. If 34b/34c are combined as one unit, the parity instrument or completed contextual measurements immediately restore a ratio above 5:1. This is not sensitive to reasonable unitization.

Therefore the program is **in infrastructure drift by definition**. The ledger’s assertion that “the artifact here IS the measured relation” is constitutionally invalid: rule 6 says the heartbeat must anchor on the README’s central bet, not the current cycle’s internal frame.

## 2. Should NLM-007 continue?

### Strongest STOP case

NLM-007 has already produced:

- A bounded result: within one decoder and authored population, a residual ridge remains predictive under several conditions.
- Withdrawal of the stronger affine-law reading once identity plus shared displacement was tested.
- Context comparisons still confounded by capacity and feature adequacy.
- No operational state, denizen-usable quotient, composition, native law, new axiom, or representation-level hostile hole.
- Proven instrument problems: an insensitive ordering readout, SVD fragility, reducer/provenance complexity, and construct ambiguity.
- Repeated review cycles increasingly concerned with hashes, schema mirrors, telemetry binding, evidence packing, wall semantics, and custom reducers.

Audits #19 and #20 independently call the program tunnel-visioned ([audit #19](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/.codex_audit19.md:228), [audit #20](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/.codex_audit20.md:379)). The evidence discipline prevented overclaims, but further reducer perfection does not build the denizen’s mathematics.

### Strongest CONTINUE case

The 34a/34b/34c sequence is cheap relative to prior work, uses existing captures, and targets the strongest live alternatives:

- 34a: capacity sensitivity.
- 34b: `P_static`/context redundancy or projection artefact.
- 34c: omitted item-by-context features.

A terminal MOOT or REDUNDANT verdict would close the relation cleanly. Survival would justify the narrow statement that the predictor separation survived these registered controls.

### Rule

**NLM-007 does not continue as a program. It receives one terminal closeout ladder.** Even an all-CONTINUE ladder ends with a bounded measurement claim and closure; it does not reopen the broader queue.

An INCONCLUSIVE result remains scientifically inconclusive, but it is an allocation stop. It must not trigger a more elaborate instrument.

## 3. Exact queue ruling

| Item | Ruling | Reason |
|---|---|---|
| Round 34a raw | **RUN once** | Already RUN-READY; cheapest direct adjudication of the historical raw comparison. |
| Round 34a static | **RUN once, separately** | Settles the distinct residualized relation; no cross-estimand pooling or rescue. |
| Round 34b | **CONDITIONAL RUN** | Run only if both 34a estimands return CONTINUE and the current final bounded repair receives RUN-READY without scope expansion. Otherwise cut. |
| Round 34c | **CONDITIONAL RUN** | Run only after a 34b CONTINUE and the same final readiness condition; it tests the strongest omitted-feature account. |
| Full Round 34 | **CUT** | Over-bundled, farther from the central artifact, and cannot upgrade survival into operational state. |
| Round 33 consequence | **CUT / archive parked branch** | Four-review instrument debt; even a pass licenses only frozen-tail predictive persistence. |
| Parity check | **CUT as a gate** | Preserve any completed output, but do not restart, repair, or delay 34a for it; it served the now-cut consequence branch. |
| Random-weight architecture null | **CUT from NLM-007** | Another diagnostic of the same ambiguous relation; it cannot produce a native construct. Reuse the idea inside a future constructive world if needed. |
| Second decoder | **CUT** | Replicating an unresolved construct does not resolve the construct. Reconsider only after a behaviorally valid native-world artifact exists. |

After the first STOP/MOOT/REDUNDANT or INCONCLUSIVE rung, stop the ladder. If all rungs return CONTINUE, record the narrow result and close NLM-007 anyway.

## 4. Round 35 and better constructive programs

Round 35 is the **right direction but the wrong first artifact**. It supplies known state, moves, consequences, and composition, but the registered design already combines linguistic authoring, adversarial approval, tokenization parity, two surface systems, two query families, matched-EDF ladders, causal patches, transfer, composition, and a 20-hour CPU envelope ([Round 35](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/theory/EXPERIMENTS.md:7427)). That risks reproducing infrastructure-first drift before the smallest runnable world exists.

Use the Round 35 document as a requirements envelope, but build a reduced first artifact. Three concrete alternatives are:

1. **Minimal operational quotient / bisimulation world.**  
   Train a tiny latent transition system on the 16 four-bit states and fixed toggle/swap/no-op actions. Define identity solely by equality of future response signatures under allowed actions.  
   **Falsifier:** quotient-equivalent states cease to be interchangeable on held-out action sequences, or actions do not descend to well-defined maps on the quotient.

2. **Cross-seed gauge-invariant action algebra.**  
   Train several independent latent realizations of the same finite world and recover the transition semigroup without coordinate alignment.  
   **Falsifier:** the purported identity classes or operation/composition table changes with seed or chart despite identical behavioral truth tables. That would show the “law” is representation-specific, not native.

3. **Denizen-available controllability and closure graph.**  
   Give the model a small declared intervention set, construct the reachable-state graph from behavioral response signatures, and test held-out two- and three-step closure.  
   **Falsifier:** single-step moves cannot be composed into stable equivalence classes, or predicted reachable states cannot causally enact the registered consequences. That would be a genuine local composition/controllability hole.

My recommendation is alternative 1 first. It is the smallest runnable object that can falsify the central bet. Add natural-language transfer, elaborate X-free ladders, and multiple query families only after the quotient and action table work at all.

## 5. Governance ruling

**Yes, review has become a bottleneck—but the deeper cause is the coupling of scientific producers to bespoke claiming reducers.** Reviewers were finding real defects, so simply reviewing less would weaken the gates.

The single most useful change is:

> **Mandatory producer/reducer separation.** A frozen, non-claiming producer receives execution readiness independently. Claim readiness belongs to a separate declarative, fail-closed reducer. Reducer defects may block interpretation, but they do not repeatedly rewrite or block an otherwise sound producer.

Audit #19 already demonstrated the value of this split ([audit #19](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/.codex_audit19.md:192)). It preserves every evidence gate—no claim is issued before reducer validation—while eliminating the dominant producer/reducer review-loop coupling.

Blackboard findings were recorded; convergence returned 100% with no open signals or disputes, and synthesis was read before this ruling. No project source or tracked file was edited, and no commit was made.
Alternatives held live otherwise unchanged. Foundational thread advanced:
program-level continuation review as a standing artifact, not an implicit
assumption.

## 2026-08-29 — Re-contextualization #21 (2-hour step-back; audit skipped — no new claim since audit #20)

Audit: the only new artifact since audit #20 is ctxS_B, a replicate of
ctxS_A already worded under audit #20's correction; the instruments (Round
34a run-ready and queued; 34b/34c under Tier-1 review) carry no outcome. No
fresh auditor fired this cycle; the next fires when the first Round 34a
outcome exists.

Live question unchanged. Whole-picture check: the program has spent this
day converting one descriptive separation (state ridge vs token-context
field, raw and residualized) into a ladder of cheap, preregistered
capacity/feature controls — 34a (matched EDF), 34b (P/C partial overlap),
34c (item-by-context) — with the expensive instruments (six-arm Round 34,
Round 33 consequence) held or parked (superseded by the continuation ruling:
full Round 34 and Round 33 are cut; 34b/34c conditional rungs of a terminal
ladder). That is the right shape: the cheapest
moot-makers run first. Reframing: every result so far is a statement about
readers of one residual relation, not about a latent-space law; the second
lens (holes hostile to structured reasoning) has produced an instrument
boundary, not a representation-level hole.

Alternatives held live: item-by-carrier fingerprint + local Jacobian
(strongest); pure capacity; decoder specificity; architecture-matched
random-weight null (superseded by the continuation ruling: the random-weight
null and a second decoder are cut from NLM-007). Foundational thread advanced this cycle: opening the
design gate for the typed truth-evaluable world (audit #19 alternative 3 /
audit #20 tunnel ruling) as a docs-only Round 35 preregistration (superseded
by the continuation ruling: Round 35 is a requirements envelope; the first
constructive artifact is Round 36) — a
four-bit finite-state world with toggle/swap/no-op, held-out predicates and
templates, frozen forced-choice yes/no log-odds, wrapper and same-length
controls, causal patching, involution and one non-commuting two-step
composition — so that when the capacity ladder resolves, the next
population is a world with known state, move, consequence and composition
laws rather than another mentioned-string micro-world. No texts or config
authored; no GPU.

## 2026-08-29 — ctxS_B complete: contextual-prefix comparator on the P_static-residualized relation, sentinel B (audit #20 wording)

`analysis_ctxS_B.json` (committed analyzer, `--residualize static`, 20
shuffles, 500 bootstraps, 4784 s): sentinel B mirrors sentinel A. On the
residualized X⊥→Δ⊥ relation the registered `token_ids_v1` context arms fall
to held-out cosine 0.04–0.08 and normalized error ≈ 1.00 at F4–F20, while the
residual state ridge keeps cosine 0.52–0.58 and normalized error 0.82–0.86;
block-first margins vs the strongest context arm: cosine +0.46 to +0.51 (LB
≥ 0.42), nerr +0.14 to +0.19, skill +0.34 to +0.45 (LB ≥ 0.19), continuous KL
+0.24 to +0.41 (LB ≥ 0.08); 8/8 keys, support 1.0. F0: cosine +0.26 but nerr,
skill and KL margins negative. Licensed reading (audit #20, verbatim
discipline): raw context performance is highly non-robust to the registered
P_static residualization and therefore P_static-aligned in this fitted
design; not identified as presentation, not by construction, not a state
contribution; the residual predictor separation is descriptive and
unmatched in capacity. Both static-form comparators are now complete; the
parity check runs next, then Round 34a (raw and static forms).

## 2026-08-29 — Re-contextualization #20 (2-hour step-back; audit #20 fired, unprimed)

Live question unchanged: is the surviving X⊥→Δ⊥ predictability in one small
decoder an operational-state relation, a generic contextual-response
relation, a capacity artefact, or an instrument artefact — and what does the
answer say about holes hostile to structured reasoning.

What holds: the adjudicated four-cell table; ctx_A/ctx_B (descriptive
higher-EDF predictor comparisons, audit #19 wording); ctxS_A. What ctxS_A
reframes: on the residualized relation the token-context arms have nothing
left (cos ≈ 0.05) while the residual ridge keeps cos ≈ 0.6. Two readings
are live and audit #20 is asked to choose (corrected by audit #20: (i) withdrawn as an underclaim, (ii)'s "re-measuring presentation" withdrawn as a variance-share reading; the ruling is "highly non-robust to P_static residualization, P_static-aligned in this fitted design"): (i) "by construction" — P_static
and the token-id field encode the same template metadata, so the collapse
is expected and says little; (ii) the collapse is informative — it shows
the raw-relation contextual comparators were largely re-measuring
presentation, which would make the static form the right estimand for
Round 34a and the raw form a secondary check. A third reading: P_static
(~10 columns) is a much smaller nuisance design than the token field (~220
columns), so residualization could leave token-level signal for the ridge
to exploit — then the collapse of the ctx arm is a df/feature-space
artefact of the comparison, not a fact about the state. Alternatives held
live otherwise unchanged (capacity; Jacobian account; decoder specificity;
typed truth-evaluable world; random-weight architecture null). Instrument
governance status: Round 34a in repair round 2 of 3 (superseded: RUN-READY at 6b93ff1 after round 4); consequence parked;
full Round 34 held (superseded by the continuation ruling: full Round 34 cut,
Round 33 archived, random-weight null cut). Foundational thread advanced: the estimand question
(raw vs residualized) is now explicit rather than implicit in tag names.
Audit #20 (fired, unprimed) returned CONDITIONAL (one overclaim, one underclaim); its correction block, its execution priorities / strongest alternative / tunnel ruling, and its final ruling follow verbatim.

## 2026-08-29 — Audit #20 adversarial correction

(Queue items in this entry — full Round 34 held, Round 33 parked, the
random-weight null, Round 35 as the constructive program — are superseded by
the continuation ruling; the wording rules stand.)

`ctx_B` mirrors `ctx_A` only as a bounded raw predictor comparison: at F4–F20 the higher-EDF cell-state ridge retains positive outer-held-out cosine, normalized-error, skill, and continuous-KL differences from the registered `token_ids_v1` ridge/kernel pair; F0 remains non-qualifying. This does not identify state or reject the contextual/Jacobian account.

`ctxS_A` is not “largely by construction.” `P_static` and `token_ids_v1` are different feature spaces, and the residual contextual ridge is already at its approximately 47-EDF ceiling at F4–F20 while its held-out cosine collapses to approximately 0.04–0.07. The empirical finding is that the registered raw context signal is highly non-robust to `P_static` residualization. The maximum positive wording is: “a higher-capacity predictor from `X_perp` retains held-out predictive information beyond the registered `P_static` projection and this fixed context field.” Withdraw “beyond template metadata”: `X_perp` may still carry item-token, nonlinear template/carrier, activation-geometry, and interaction signal omitted by both controls.

The specific same-template leakage worry is not supported: every outer key holds out an entire carrier block and a disjoint word fold, and residual ridge cosine does not rise after residualization; the margin expands because the context arm collapses. Related-template authorship and changed target geometry remain limitations. Raw Round 34a is still required for `ctx_A`/`ctx_B`; static Round 34a is separately required for `ctxS`; neither is the universal “right” estimand. Alongside them, run a no-completion `P`/`C`/`P+C`/`C_perp` partial-overlap screen and a frozen-item-embedding-by-`P_static` comparator before full Round 34 or Round 33.

Round 34a remains unrun and Tier-1 re-review #3 is NOT-READY on one exact telemetry-binding invariant. Use one narrow final repair; do not expand the reducer or launch before RUN-READY. The strongest alternative is now an item-by-carrier activation fingerprint plus local punctuation Jacobian, not capacity alone. No native law or representation-level hostile hole is established.

#### Audit #20 — sections 6-8 verbatim

## 6. What should run instead of or alongside Round 34a

### Priority 1 — `P/C` partial-overlap screen on existing captures

This is the cheapest missing scientific control and directly adjudicates the
“by construction” interpretation. Use the identical A/B outer block-by-word
folds, training-only transformations, 500 crossed bootstraps, cosine and nerr
only, no completion, no shuffle, and no model forward.

For each layer and outer key, fit:

1. `P`: `P_static -> Delta`;
2. `C`: registered `token_ids_v1` ridge/kernel `-> Delta`;
3. `P+C`: a nested combined field `-> Delta`;
4. `C_perp -> Delta_perp`, where both `C` and `Delta` are residualized on
   `P_static` using maps fit only on the relevant training rows; and
5. the same-EDF `X_perp` ridge as a reference, not as the claim target.

Also report, on held-out rows, the alignment between the raw context
prediction and the `P_static` prediction. Refit every target-dependent
residualizer inside the downstream inner folds for this diagnostic.

Interpretation:

- If `P+C` does not improve over `P` and `C_perp` is null in both sentinels,
  the correct conclusion is that the registered raw context field is
  `P_static`-redundant in this design.
- If `C_perp` retains signal, the current `ctxS_A` collapse is a fitting or
  feature-projection artifact; “P_static-aligned context” is too strong.
- In neither case does the result identify presentation causally.

This screen is more directly diagnostic of the estimand than adding another
completion reducer.

### Priority 2 — cheap item-by-context X-free comparator

The registered context field omits the item token, while `X_perp` necessarily
contains its activation consequences. Run a no-completion ridge comparator on
existing captures with:

- `P_static`;
- 16 training-only PCs of the frozen item embedding;
- fixed `P_static x item-PC` interactions; and
- optionally the boundary-token/POS floor from `token_ids_v1`.

Fit on calibration words only, transfer to held-out words through the frozen
embedding, match state EDF downward, and score cosine/nerr on the same outer
keys. This is the cheapest direct test of the hypothesis that the residual
ridge is exploiting lexical/item-by-template structure omitted by the context
field. It is narrower and cheaper than the full six-arm Round 34 and avoids
the parked K=13/SVD path.

If this arm closes the static state margin, classify the current result as
**item/context-feature-sensitive** and stop the consequence queue. If it does
not, the result is still not operational state; it has only survived a much
fairer X-free feature test.

### Priority 3 — architecture-matched random-weight depth screen

Only if the residual margin survives Priorities 1–2, run the already-proposed
CPU random-weight null: same architecture, tokenizer, templates, sentinel,
folds, identity-plus-shared-displacement null, and matched-EDF state/context
predictors; score F0/F4/F8/F12/F20 cosine and nerr only. No completion and no
generation.

A similar middle/deep residual profile in a random decoder would strongly
support architecture/local-smoothness and fingerprint propagation. A trained-
only profile would keep learned structure live but would still not identify
operational state.

### Do not run next

- Do not run the full six-arm Round 34 before the core and partial-overlap
  screens.
- Do not reopen Round 33 merely because `ctxS_A` has a large unmatched
  margin. A smoother high-dimensional reconstruction is expected to remain
  closer under a deterministic tail.
- Do not spend another long review loop on K=13 or low-rank telemetry for this
  question; cosine/nerr and raw continuous KL are sufficient.

## 7. Strongest alternative explanation now

The strongest alternative is a sharpened **item-by-carrier fingerprint plus
local Jacobian** account:

> `P_static` removes a coarse block/length/position response. The remaining
> `X_perp` retains a dense continuous fingerprint of the item token, the
> held-out carrier, lexical class, activation scale, and their interactions.
> Appending a fixed punctuation token produces a deterministic local response
> `Delta_perp = J(X_perp, context) + noise`. A high-EDF ridge can learn a
> transferable linear readout of that already nonlinear activation feature
> map. The fixed `token_ids_v1` field has only carrier-by-POS rows, omits the
> item token and dense interactions, and therefore collapses on the residual
> target. No denizen-usable state, quotient, operation, or composition law is
> required.

This account explains all three current observations at once:

1. raw context predicts a moderate component;
2. that component disappears after coarse nuisance residualization; and
3. the rich activation still predicts the local residual response.

It is stronger now than the generic “capacity alone” objection. Matched EDF is
necessary, but feature adequacy — especially item-by-context information — is
the more important remaining confound.

## 8. Tunnel-vision and second-lens ruling

**The program remains tunnel-visioned.** It has spent many rounds on one
relation in one small decoder, one authored 80-word population, sixteen
related templates, two punctuation tokens, one append operation, one readout
site, and one completion path. Audit/reducer loops now consume a material
fraction of the research effort. The review discipline has prevented invalid
claims, but increasingly perfect reducers for this one relation do not build a
denizen's mathematics.

The remaining Round 34a defect is worth one exact repair because the screen is
cheap and already registered. Beyond that, the next scientific increment
should be orthogonal: the item-by-context comparator, the random-weight null,
or a typed truth-evaluable finite-state world with forced-choice consequences,
causal patching, and two-step composition.

Under the second lens, `ctxS_A` proves no representation-level hostile hole.
It exposes an **instrument boundary**: the registered context reader spans the
raw coarse response but not the residual response, while the activation reader
does. Whether that is a missing quotient, useful operational state, or merely
a richer fingerprint remains unresolved. A next latent space should make the
factorization denizen-available — lexical/item coordinates, presentation
coordinates, and operation-bearing state with behavioral consequences — but
the current decoder has not been shown incapable of such a factorization.

#### Audit #20 — final ruling (verbatim)

## Final ruling

- **Upheld:** `ctx_B` mirrors `ctx_A` as a descriptive higher-EDF raw
  predictor comparison; `ctxS_A` has a real F4–F20 residual predictor
  separation; F0 is non-qualifying.
- **Withdrawn as overclaim:** “beyond template metadata,” “presentation has
  been removed,” any state contribution, and any causal or variance-share
  interpretation.
- **Withdrawn as underclaim:** “largely by construction” and any implication
  that `P_static` and `token_ids_v1` are the same feature space.
- **Reframed:** raw context performance is highly non-robust to the registered
  `P_static` residualization and therefore `P_static`-aligned in this fitted
  design; it is not thereby identified as presentation.
- **Leakage ruling:** no exact template or word identity is shared across the
  outer fit/test boundary; absolute ridge cosine does not inflate. Related
  authored structure, changed target geometry, and non-fully-nested downstream
  preprocessing remain qualifications.
- **Estimand ruling:** run both raw and static Round 34a; neither substitutes
  for the other. Add the cheaper partial-overlap and item-by-context controls
  before full Round 34.
- **Implementation ruling:** Round 34a remains unrun and NOT-READY until the
  single review-#3 telemetry-binding invariant is closed.
- **Tunnel ruling:** one final narrow repair and the cheap screens are
  justified; another broad reducer loop on the same punctuation relation is
  not. Pivot the next substantive work toward the item/context null, the
  architecture null, or a typed truth-evaluable world.

Blackboard findings were recorded. `bb_convergence` returned 100% with no open
signals, disputes, unread documents, or partial documents, and `bb_synthesis`
was read before this verdict.

## 2026-08-29 — ctxS_A complete: contextual-prefix comparator on the P_static-residualized relation, sentinel A

`analysis_ctxS_A.json` (committed analyzer, `--residualize static`, 20
shuffles, 500 bootstraps, 5655 s): on the residualized X⊥→Δ⊥ relation the
token_ids_v1 contextual arms retain almost nothing (held-out cosine 0.04–0.07
at F4–F20, normalized error ≈ 1.00), while the residual state ridge keeps
cosine 0.56–0.62 and normalized error 0.78–0.83; block-first margins vs the
strongest contextual arm: cosine +0.51 to +0.58 (LB ≥ 0.46), nerr +0.17 to
+0.23, skill +0.32 to +0.49 (LB ≥ 0.16), continuous KL +0.26 to +0.48 (LB ≥
0.13); 8/8 keys, support 1.0. F0: cosine +0.26 but nerr/skill/KL margins
negative (structural regime, as before). Reading (audit #19 discipline): the
collapse of the contextual arms is largely by construction — P_static is
built from the same template metadata the token-id field encodes (corrected
by audit #20: withdrawn as an underclaim; the two are distinct feature
spaces and the collapse is an empirical non-robustness to P_static
residualization) — so this is a descriptive comparison showing the residual
X⊥ carries held-out predictive information beyond template metadata
(corrected by audit #20: withdrawn as an overclaim; say "beyond the
registered P_static projection and this fixed token_ids_v1 context field");
it is not an identified
state contribution and not capacity-matched (the residual ridge still has
far more effective df than a near-null context arm). It is the static-form
input the parked consequence loader and the Round 34a static screen were
registered to use. ctxS_B is running next, then the parity check.

## 2026-08-28 — ctx_B complete: contextual-prefix completion comparator, sentinel B (audit #19 wording)

`analysis_ctx_B.json` (committed analyzer, unresidualized form, 20 shuffles,
500 bootstraps, 4476 s): on sentinel B's outer-held-out keys the higher-EDF
cell-state ridge retained a positive held-out score difference from the
registered `token_ids_v1` context-only pair at F4–F20 on displacement
cosine (+0.11 to +0.18, LB ≥ 0.09), normalized error (+0.11 to +0.16),
completion skill (+0.33 to +0.41, LB ≥ 0.12) and continuous KL (+0.24 to
+0.40, LB ≥ 0.13); 8/8 keys point-positive, no family collapse, support
1.0. F0: cosine +0.018 (LB 0.010), continuous-KL LB below zero. Per audit
#19 this is a descriptive predictor comparison between arms of very
different effective df and feature class — not an identified state
contribution, not a rejection of the contextual-response account, and not a
live gate. Together with ctx_A it fixes the two-sentinel picture at
unmatched capacity; Round 34a's matched-EDF core screen is the registered
next step. Chain now running: ctxS_A/B (static form), then the parity
check.

## 2026-08-28 — Re-contextualization #19 (2-hour step-back; audit #19 fired, unprimed)

Live question unchanged: is the surviving X⊥→Δ⊥ predictability in one small
decoder an operational-state relation, a generic contextual-response
(Jacobian) relation, a capacity artefact, or an instrument artefact — and
what does the answer say about holes hostile to structured reasoning.

What holds: the adjudicated four-cell table; both contextual screens; ctx_A
(ridge beats the strongest contextual-prefix arm on every endpoint at F4–F20
with crossed LBs > 0, at unmatched capacity). What is reframed: the whole
line now hinges on ONE identified confound — capacity (state ridge ~5–10×
the contextual arm's effective df) (corrected by audit #19: capacity is not
the sole confound — the difference is compatible with state information,
unmatched capacity, missing contextual features, or a mixture). Round 34 is
the registered answer and runs before the consequence test (corrected by
audit #19: the Round 34a matched-EDF core screen runs first; the full Round
34 is held; Round 34 is `P_static`-residualized and cannot retroactively
capacity-match `ctx_A`); the consequence instrument is parked on a
branch after four NOT-READY rounds, which I read as partly a
reviewer-escalation artefact (each round raised a new bar) and partly real
(legacy-base pins, exact-fit reuse). Instrument reviews are now the main
consumer of the program's time; the repair cap is doing its job.

Alternatives held live: (1) capacity explains the gap (Round 34 MOOT;
corrected by audit #19: the decisive first check is Round 34a) — then
the line collapses to "context vectors predict context-vector displacements"
and the constructive program moves to a typed use-frame task; (2) capacity
does not explain it (KEEP) — then the consequence question returns, but
audit #18's construct-validity limit stands (persistence ≠ state); (3) the
cheapest decisive check may be smaller than Round 34: df-match the state
ridge alone against the EXISTING ctx artifacts (one arm, one solve) — audit
#19 is asked whether that should run first; (4) the skill margins may
partly inherit the skill-denominator pathology flagged at F0; (5) decoder
specificity remains untested. Foundational thread advanced: instrument
governance itself — split producer/joint verdicts so a read-only reducer
cannot block a producer run, and the repair cap as a standing rule.
Audit #19 (fired, unprimed) returned CONDITIONAL; its correction block and its staging ruling / alternatives follow verbatim.

## 2026-08-28 — Audit #19 adversarial correction

`ctx_A` contains a real outer-held-out score difference at F4–F20, but only between a higher-EDF state ridge and this fixed lower-EDF context-only pair. Replace “the contextual arm did not close the gap” with: “the higher-EDF state predictor retained a positive held-out score difference from the registered context-only predictors.” The result is descriptive and does not identify state, reject the contextual/Jacobian account, or make a state-reading gate live. Inner selection used calibration-only displacement cosine and the completion readouts were scored on outer keys, so the proposed same-test-fold tuning objection does not apply. The endpoints are correlated consequences of the same prediction. F0 remains non-qualifying: skill and continuous-KL lower bounds cross zero and one family collapses.

Round 34 is over-bundled for the first capacity question. Its primary relation is `P_static`-residualized, not the raw `ctx_A` estimand; its six arms combine a matched-capacity test with a context-feature-family search; and its confirmatory KL-rank reimports the parked K=13/SVD qualification. Put a matched-EDF core screen first: existing token ridge/kernel only, state matched to their selected EDF and 47/48 ceiling, same A/B outer folds, cosine and normalized error, no completion. A state-only solve against stored aggregate JSON is a screen, not a crossed gate, unless the contextual cell predictions are recomputed. Run narrow completion only if that screen survives; run the embedding/edit arms only after that.

Parking the Round 33 consequence instrument is upheld as allocation, not as a kill. Review scope escalated, but the final blockers included a real legacy-manifest crash and unproved fit reuse, so it was not run-ready. The strongest alternative remains a generic contextual-response/Jacobian relation in which a continuous residual fingerprint predicts the local punctuation response and propagates smoothly. The next orthogonal measurements are a CPU architecture-matched random-weight depth screen and a typed truth-evaluable finite-state task with forced-choice behavior, causal patching, and two-step composition. No representation-level hostile hole is proven.

#### Audit #19 — staging ruling, Round 33 parking assessment, tunnel-vision verdict, and alternatives (verbatim, sections 3-6)

## 3. Run the cheaper decisive check first

Do not discard the registered Round 34 design. Put a preregistered
short-circuit screen in front of it.

### Round 34a — matched-EDF core screen

1. Use the exact existing A/B outer carrier-by-word folds and training-only
   standardization.
2. Recompute only the registered `token_ids_v1` ridge and kernel predictions.
   Fit state ridges by continuous bisection to (a) the selected contextual EDF
   and (b) the honest 47/48 context rank ceiling.
3. Score only displacement cosine and normalized error with paired,
   block-first crossed intervals. No completion, K=13 universe, new context
   feature family, model forward, or joint claiming reducer is needed.
4. If matched margins shrink to at most 0.02 with crossed upper bounds below
   0.02 in two common F4–F20 layers for both sentinels, report
   **capacity-sensitive screen; stop**. Do not run the full six-arm audit or
   Round 33.
5. If the margins retain positive crossed lower bounds, run a completion pass
   for only the selected token ridge/kernel pairs, using raw continuous KL and
   treating skill as a diagnostic. Then decide whether the richer context
   feature audit is worth the remaining compute.

A literal “state-only one solve against the existing JSON” is acceptable only
as a point screen. `analysis_ctx_A.json` stores fold summaries and intervals,
not reusable per-cell contextual predictions, so it cannot support a new exact
paired crossed gate without recomputing the context predictions. Recomputing
those cheap context fits is still far smaller than the six-arm completion run.

If the scientific target is specifically the primary `P_static` residual
relation rather than the raw `ctx_A` sentence, use the same staged design under
`--residualize static` after the protected contextual residual artifacts are
complete. Do not claim that one answers the other.

### Round 34b — feature-adequacy audit, conditional

Only if Round 34a survives should the input-embedding sequence and
template-edit kernels run. Label this a fixed context-family adequacy audit,
not “capacity matching.” Keep the sentinel/position field as a cheap floor.
The forced low-lambda `token_ids_v1_ceiling` is useful telemetry but is not an
inner-selected fair predictor.

The current producer/joint split requested for Tier-1 review #4 is good
software governance: a read-only reducer should not block a safe producer.
It does not answer the scientific staging question. No producer run is
authorized until it separately receives RUN-READY.

## 4. Round 33 parking: justified, not a kill

There is some reviewer escalation. Later rounds increasingly audited schema
mirrors, hashes, fail-closed reducers, and hard-wall semantics rather than the
core consequence estimand. The repair process was consuming the program.

But the parking decision was not arbitrary. Review #4 still found:

- a deterministic analyzer crash on the real legacy manifests;
- only hyperparameter-selection equality, not exact contextual-fit reuse;
- incomplete two-base preflight and legacy compatibility binding;
- hard-wall paths that could emit claiming artifacts after overruns; and
- no real HEAD-versus-refactor CPU parity result.

The legacy crash and fit-reuse failure alone make the instrument not run-ready.
Parking after four rounds was therefore a defensible allocation stop. It did
not falsify the consequence hypothesis, invalidate the design idea, or justify
deleting the branch.

If a later matched-capacity result earns reopening, salvage the smallest path:
preflight both bases before model load, rerun/serialize the exact contextual
fits with fingerprints, keep the consequence producer separate from the
joint reducer, and perform one real CPU parity comparison. Do not resume the
entire review-grown diff by default.

Even a repaired PASS would license only persistence of predictive accuracy
under frozen tails. It would not distinguish operational state from a smoother
reconstruction propagated through deterministic decoder layers.

## 5. Tunnel-vision and strongest alternative

**Yes, the program is tunnel-visioned.** It has spent many rounds on one local
relation in one small decoder, one authored 80-word population, sixteen related
templates, two punctuation tokens, one append move, one readout position, and
one completion mechanism. The recent history is now dominated by instrument
and reducer reviews. That is a governance success compared with running broken
claims, but it is not progress toward a denizen's mathematics.

No representation-level hostile hole has been proven. The current holes are
primarily in measurement: raw identity dominance, presentation/context
entanglement, low-rank numerical fragility, and inability to distinguish a
useful state variable from a high-dimensional fingerprint.

The strongest alternative remains the **generic contextual-response/Jacobian
account**:

> The residual vector contains a rich continuous fingerprint of template,
> token, position, lexical class, and local activation geometry. Appending a
> fixed punctuation token induces a deterministic local response. A
> high-capacity ridge reconstructs that response better than a low-rank
> hand-built context map, and the better reconstruction remains closer after
> smooth downstream transformations. No operational quotient or denizen-usable
> state is required.

`ctx_A` strengthens this account in one respect: context alone already reaches
cosine 0.46–0.62 at F4–F20. Its normalized error remains about 1.00, so the
current state advantage may be continuous activation/magnitude information,
but that information can still be generic local geometry rather than
structured reasoning.

## 6. What should run instead of or alongside full Round 34

Priority order:

1. **Run Round 34a, not the full six-arm completion, first.** This is the
   cheapest direct capacity moot-maker and can terminate the line cleanly.
2. **Run an architecture-matched random-weight depth-profile screen on CPU.**
   Use the same tokenizer, templates, sentinel moves, outer folds, identity +
   shared-displacement null, and matched-EDF state/context predictors at
   F0/F4/F8/F12/F20. Score only displacement cosine and normalized error. A
   similar middle/deep-layer profile or matched state surplus in a random
   decoder would strongly support architecture/local-smoothness rather than
   learned operational structure. No completion or generation claim is
   needed; any GPU version still requires explicit approval.
3. **Design a typed truth-evaluable world instead of another mentioned-string
   population.** A concrete CPU-scale successor is a four-bit finite-state
   world with operations `toggle(i)`, `swap(i,j)`, and no-op. Hold out predicate
   names and surface templates. Measure frozen forced-choice yes/no log-odds
   for all four bits, not full-vocabulary KL. Require irrelevant-wrapper and
   same-length token controls, causal patching of predicted versus true moves,
   the involution `toggle(i) o toggle(i) = identity`, one noncommuting
   two-step composition, and transfer across two disjoint query-tail families.
   This gives the denizen a known state, move, consequence, and composition
   law and directly exposes where the decoder's latent world fails them.
4. **Use causal consequence, not only predictive persistence.** Patch the
   predicted post-move state into the frozen decoder and compare behavioral
   log-odds against the true post-move state, shared-displacement prediction,
   context-only prediction, and same-norm random patch. This distinguishes a
   state estimate that can enact the move from one that merely reconstructs
   nearby activations.
5. **Only then use a second trained decoder.** It tests model specificity but
   does not solve the current construct ambiguity.

The architecture null can run alongside the docs-only typed-world design. Do
not spend the next increment on another increasingly elaborate reducer for the
same punctuation relation.

## 2026-08-28 — ctx_A complete: contextual-prefix completion comparator, sentinel A (unmatched capacity)

`analysis_ctx_A.json` (committed analyzer, unresidualized form, 20 shuffles,
500 bootstraps, 5783 s): the X-conditioned ridge beats the strongest
contextual-prefix arm (token_ids_v1 ridge / kernel) on every endpoint at
F4–F20 with crossed 95% lower bounds above zero — cosine margin +0.15 to
+0.20 (LB ≥ 0.13), normalized error +0.14 to +0.20, skill +0.34 to +0.46
(LB ≥ 0.25), continuous KL +0.27 to +0.45 (LB ≥ 0.17); support 1.0. At F0
the cosine margin is +0.019 (LB 0.011) while the skill and KL lower bounds
fall below zero. Audit #18 wording governs: this completed comparator did not
close the ridge-versus-context gap at the registered (unmatched) capacity
(corrected by audit #19: the phrase "did not close the gap" is withdrawn;
say "the higher-EDF state predictor retained a positive held-out score
difference from the registered context-only pair" — a descriptive predictor
comparison, not evidence that context failed or that capacity is the sole
confound); the state ridge still carries ~5–10× the contextual arm's
effective df, so the gap remains unidentified until Round 34's
capacity-matched comparison (corrected by audit #19: Round 34 is
`P_static`-residualized and cannot retroactively capacity-match `ctx_A`;
the Round 34a matched-EDF core screen runs first).
Not a "live gate"; not a state-reading result. ctx_B is running next, then
the static-residualized forms ctxS_A/ctxS_B, then the parity check.

## 2026-08-28 — Round 34 registered: capacity-matched state-versus-context (audit #18's first control)

Codex design gate (`.codex_dfmatch_design.md`, registered in
theory/EXPERIMENTS.md, `d493cf2`): the ridge-versus-context gap is
unidentified because the state ridge carries ~210–406 effective df against
~42 for the token-id contextual ridge. Round 34 matches capacity foldwise —
for each of six fixed contextual candidates (sentinel/position only; the
Round 31 token-id ridge at its selected lambda and at a lowered "ceiling"
lambda with capacity-shortfall telemetry; the contextual RBF kernel; a frozen
input-embedding sequence RBF arm; a template-edit Levenshtein kernel) a
separately standardized state ridge is solved by bisection to the same
training EDF. The context-only rows repeat within POS (≤48 distinct rows per
fold), so the contextual ladder cannot reach the state EDF; the state is
matched downward, never the context inflated. KEEP needs matched margins
≥ 0.02 with crossed LB > 0 on cos/skill/KL-rank, ≥ 6/8 jointly positive keys,
no block collapse, support ≥ 0.95, two common F4–F20 layers in both
sentinels; MOOT needs the strongest matched margin ≤ 0.02 with crossed UB
< 0.02 under the same key rules; otherwise INCONCLUSIVE/CAPACITY-SENSITIVE.
Ruling: Round 34 runs BEFORE Round 33 (the consequence test cannot identify
state while the predictor advantage is capacity-confounded); only a KEEP
verdict returns Round 33 to the queue. (Corrected by audit #19: the full
six-arm Round 34 is HELD; a preregistered Round 34a matched-EDF core screen
runs first, K=13 KL-rank is diagnostic in favor of raw continuous KL, and
Round 33 stays parked as an allocation decision, not a kill.) Cost 2–3.5 h CPU per sentinel, four-hour
wall. The consequence instrument is parked on branch `conseq-instrument`
after four NOT-READY Tier-1 rounds (decision raised to the user). Round 34
implementation is being written against the committed main analyzer; Tier-1
review before any run. (Later the same day: implemented as
`--context-capacity-audit round34_v1`, commit `9eb1301`; producer path
RUN-READY, joint reducer flagged for one more review; run held by audit #19.)

## 2026-08-28 — Re-contextualization #18 (2-hour step-back; audit #18 fired, unprimed)

Live question unchanged: is the surviving X⊥→Δ⊥ predictability in one small
decoder an operational-state relation, a smooth presentation/lexical
relation, a generic prefix-edit response, or an instrument artefact — and
what does the answer say about holes hostile to structured reasoning.

What holds: the four-cell common-scale table (adjudicated wording in
STATE.md); the contextual-prefix state screens in both sentinels (token-id
field cos 0.45–0.65 vs ridge 0.62–0.76 at F4–F20; ctx norm-error ≥ 1.0 vs
ridge 0.81–0.89). What is reframed by the screens: a token-id-only field
already carries half or more of the raw displacement cosine, so the
X-conditioned surplus is a gap of ~0.1–0.2 cosine and, more sharply, the
ctx field cannot beat identity on norm-error at all — the state-reading
claim now rests on that norm/scale margin as much as on direction. F0 is
nearly closed by prefix ids on displacement direction only (0.65 vs 0.69;
normalized error ~1.00 vs 0.97), which is the token-identity regime reading,
not a new fact — (corrected by audit #18: a screen-level directional
near-closure, not proof that "prefix IDs explain F0" or the full
transition; F0 is a model-class-sensitive diagnostic, since a post hoc
kernel-field reduction passes the analogous rule at F0 in all four cells).
The screens do not establish that the state-reading gate is live: the
state ridge has ~210–406 effective df at F4–F20 versus ~42 for the
contextual ridge, and the completion endpoints, crossed intervals, joint
key count, and collapse checks remain unscored (corrected by audit #18).

Alternatives held live (not run): (1) the ridge–ctx gap is a
capacity/standardization artefact (df-matched ridge vs sparse token field)
— the completed ctx_A/ctx_B completion scores and the df-matched X-free
field are the direct checks; (2) the gap is a smooth lexical relation the
four word-only nulls under-fit (kernel/knn already in the K=13 universe say
no, but only at their tuned capacity); (3) the gap is real but a
one-position readout artefact — Round 33's consequence test is exactly this
falsifier; (4) decoder-specific — a second pinned decoder remains the
cheapest replication axis and is deliberately behind the consequence test;
(5) the whole line is a well-measured triviality (context vectors predict
context-vector displacements) — the hostile-hole program only earns
anything if the consequence currency survives AND a typed use-frame task
shows a move with multi-position consequences that the bridge ladder cannot
absorb. Foundational thread advanced this cycle: the licensed-wording
discipline (adjudication → STATE/memory verbatim) and the repair-round cap
stop on the SVD gate — instruments are not allowed to consume the program.
Audit #18 (fired, unprimed) returned CONDITIONAL / wording corrections required; its correction block and its alternatives follow verbatim.

### 2026-08-28 — Audit #18 adversarial correction

The contextual-prefix results are screens only. At F4–F20 they do not triage the X-conditioned hypothesis out, but they do not make a state-reading gate evidentially live: completion endpoints, crossed gates, joint key support, collapse checks, and capacity matching are missing. The state ridge uses approximately 5–10 times the contextual arm's effective degrees of freedom, so the ridge-versus-context gap is not yet identified as state information.

At F0, contextual token-sequence metadata nearly closes ridge direction but not magnitude; “prefix IDs explain F0” is too broad. The designated F0 ridge field fails three four-cell conditions, but the stored kernel field passes an analogous post hoc reduction in all four. F0 is a model-class-sensitive diagnostic, not an all-field kill.

SVD telemetry is parked by allocation choice, not by an AGENTS.md repair-round rule, and its gate remains unpassed. The Round 33 consequence instrument exists but is unrun and NOT-READY: the joint-positive-key rule is not implemented correctly, and Tier-1 provenance/parity blockers require closure. A future consequence pass would show persistence of predictive accuracy, not by itself operational or semantic state.

#### Audit #18 — tunnel-vision verdict, strongest alternative, and recommended execution order (verbatim)

## Tunnel-vision verdict

Yes. The program is currently concentrated on one residual \(X_\perp \rightarrow \Delta_\perp\) relationship in:

- one 0.6B decoder;
- one 80-word inventory;
- sixteen closely related templates;
- two punctuation sentinels;
- one append operation;
- one readout position;
- one-step local response;
- one heavily repaired analysis path.

The strongest alternative explanation is a **generic contextual-response/Jacobian account**:

> The high-dimensional residual state encodes template, token, position, and lexical context. Appending a fixed punctuation token induces a locally predictable architectural response. A high-capacity ridge learns that deterministic response. Accurate reconstruction then remains closer under later smooth decoder dynamics, without any quotient, operational state, or latent-world law being present.

The contextual screen’s cosine of approximately 0.45–0.65 strengthens this alternative rather than weakening it.

## Recommended execution order

1. **Capacity-match before interpreting Round 33.**  
   On every existing outer fold, constrain the state ridge to the contextual arm’s effective degrees of freedom—approximately 42—either by solving for a state-ridge lambda satisfying  
   \(\mathrm{tr}[X(X^\top X+\lambda I)^{-1}X^\top]\approx df_{\text{ctx}}\),  
   or by a training-only rank/PCA constraint. Preserve the same held-out splits, endpoints, and crossed gates. Add a contextual-capacity ladder as the symmetric control.

2. **Run cheaper X-free moot-makers.**

   - Sentinel/terminal-token plus absolute and relative position only.
   - Template/edit-kernel baseline.
   - Frozen input-embedding sequence baseline over the last-eight prefix and first-four suffix tokens.
   - Contextual nonlinear capacity ladder.

   If these close the state arm after capacity matching, the current interpretation becomes moot.

3. **Use an architecture null.**  
   Repeat the capture and screen in an architecture-matched randomly initialized decoder. A similar depth profile would show that the effect arises from residual architecture and local smoothness rather than learned operational structure. Any GPU execution still requires explicit approval.

4. **Change the measurement.**  
   Replace full-vocabulary KL under one artificial tail with a behavior-bearing readout:

   - frozen yes/no log-odds;
   - a typed operation-specific target;
   - causal patch/ablation effects;
   - at least two disjoint frozen tail families.

5. **Change the task family.**  
   A stronger operational-state test would use a world with known transitions, such as a controlled finite-state machine, modular arithmetic, or truth-evaluable propositions. Require:

   - held-out predicates and templates;
   - matched wrapper-edit and irrelevant-token controls;
   - bidirectional transfer;
   - involution where appropriate;
   - two-step composition;
   - prediction of an externally scored behavior.

6. **Only then test a second trained decoder.**  
   A second decoder checks model specificity, but it does not fix the present construct-validity ambiguity.

The completed contextual commands should be run only after the current provenance/parity review blockers are closed:

```powershell
.venv\Scripts\python.exe experiments\analyze_lm_dynamics.py --run lm_dyn_v1 --config experiments/config/lexical_probe_v1.json --source forward --sentinel-tag A --target delta --unseen-words 2 --residualize static --contextual-prefix-xfree --pairs 0 1 2 3 4 --n-shuffle 20 --n-boot 500 --tag ctx_A
```

```powershell
.venv\Scripts\python.exe experiments\analyze_lm_dynamics.py --run lm_dyn_v1 --config experiments/config/lexical_probe_v1.json --source forward --sentinel-tag B --target delta --unseen-words 2 --residualize static --contextual-prefix-xfree --pairs 0 1 2 3 4 --n-shuffle 20 --n-boot 500 --tag ctx_B
```

Do not run the current consequence command until the joint-key defect and Tier-1 blockers are closed.

## 2026-08-28 — resSA2 complete: the common-scale sentinel × nuisance table is filled

`analysis_resSA2.json` (sentinel A, P_static, amended K=13 candidate universe,
four word-only nulls, crossed block-first bootstrap; 5825 s, committed
analyzer) passes the residual-versus-strongest-null gate at F4, F8, F12 and
F20 (block-first lower bounds: cos ≥ 0.46, skill ≥ 0.18, KL-rank ≥ 0.20;
keys jointly positive 7–8/8 at every passing layer; retention marker held on
all three endpoints) and fails F0 (skill and KL-rank margins negative, 2/8
full-gate keys). Read with resAA, resSB and resAB this completes the
{A,B} × {P_static, P_aug-score4} table on ONE common scale: F4–F20 pass in
all four correlated same-population cells, F0 fails in every cell except the
weak A-score4 association. Audit #17 wording stands unchanged: consistent
within-decoder, within-population condition robustness, not replication,
operational state, or presentation independence; B-score4 stays
amended-implementation and SVD-telemetry-incomplete. The residual F0 failure
in all four cells was read as the one structural regularity of the table
(corrected by audit #18: the designated ridge field is non-qualifying in
three cells with a weak pooled A-score4 exception, but a post hoc kernel
reduction passes the analogous rule at F0 in all four cells — F0 is a
model-class-sensitive diagnostic, not an all-field dead end or a
structural identity law). The
Evidence-gate adjudication of the four-cell synthesis is launched; the
contextual-prefix chain (`run_ctx.cmd`, committed analyzer copy) starts
automatically now that resSA2 has written.

Adjudicated (Evidence gate, `.codex_fourcell_adjudication.md`): PASS,
qualified. Licensed wording, verbatim: "The sentinel {A,B} x
{P_static,P_aug-score4} table is complete on a common
K=13/four-word-only-null/crossed-bootstrap scale for the
residual-versus-null mechanical gate. F4-F20 pass in all four correlated
cells; F0 is non-qualifying in three cells and yields only a weak pooled
A-score4 association with 2/8 full-gate keys. This is consistent
within-decoder, within-population condition robustness. It is not
replication and does not identify operational state, presentation
independence, a presentation decomposition, composition, a native law, or a
representation-level hostile hole. B-score4's ridge cosine and skill results
are mechanically reportable; its K=13 KL-rank endpoint and every low-rank
interpretation remain amended-implementation and SVD-telemetry-incomplete."
Additionally licensed: all 48 F4-F20 layer x endpoint bootstrap-median
common-scale ratios exceed 0.5 (estimator/null competition ratios, not
retained signal; each cell has one F4 continuous-KL interval LB below 0.5).
The phrase "uniform F0 failure" is withdrawn from the entry above: F0 is a
bounded diagnostic, not a structural law (A-score4 clears the pooled gate;
four live readings: token-identity endpoint regime, local emergence boundary,
readout/normalization pathology, score4 instrument specificity). The
EXPERIMENTS.md "7-8/8 keys" reads as jointly positive keys; strict full-gate
counts are 7/8, 7/8, 6/8, 8/8. Ledger erratum appended (resSA2 row wrote
"sentinel 2"). Round 33 order unchanged.

## 2026-08-28 — SVD telemetry gate: repair-round cap tripped; Round 33 consequence test implemented

SVD telemetry re-review #4 (`.codex_svd_review4.md`) returned NOT-READY with
six open items (mixed `primary_shadow` pooling, support-mask and missing-map
fail-closed behaviour, completion-off telemetry, a crash-safe record
validator, unexpected context groups, oracle `fit_id` collisions). That is
the fourth consecutive repair round without an admissible result, so the
repair-round cap (global CLAUDE.md §2.7; corrected by audit #18: no such
rule exists in AGENTS.md, and the parking is a discretionary allocation
decision, not a rule-triggered closure) applies: the low-rank telemetry
gate is parked and unpassed, not repaired again, and the question of
whether to continue it is raised to the user. The gate only blocks low-rank (`--aug-rank`/probe-1
screen) claims; the analyzer's SVD diff stays uncommitted and
`run_r31.cmd` stays disarmed.

Round 33's consequence test is implemented as registered: runner stage
`capture_forward_consequence` (frozen `fixed_tail_v1` eight-token tails
after each sentinel, readout-equality check against the base capture,
compact per-position true-law summaries, repeat-law noise) and analyzer
`--source forward_consequence` (multi-position teacher-forced KL, uniform
mean over positions 1..k, k ∈ {4, 8}, G_k against the strongest of the four
word-only nulls and the contextual-prefix fields inside each block-first
replicate; a layer passes only at both k). Full per-position laws are not
stored (≈3 GB); the analyzer recomputes the truth per fold. Tier-1
implementation review #1 is running; nothing has been run. resSA2 is at
F20.

## 2026-08-29 — Re-contextualization #17 (2-hour step-back; audit #17 fired and adopted in Round 33)

Live question unchanged: is the surviving X⊥→Δ⊥ predictability in one small
decoder an operational-state relation, a smooth presentation/lexical
relation, or an implementation artefact — and what does the answer say about
holes hostile to structured reasoning.

What holds: the sentinel {A,B} × {P_static,P_aug-score4} table is complete only for the residual-versus-four-word-only-null mechanical gate: F4–F20 pass in all four correlated cells, while F0 fails except for a weak pooled A-score4 association with only 2/8 full-gate keys. This is consistent within-decoder, within-population condition robustness, not replication; B-score4's ridge-only cosine and skill margins are mechanically reportable, while its K=13 KL-rank endpoint and every low-rank interpretation remain amended-implementation and SVD-telemetry-incomplete. (Audit #17 wording.)

What this cycle reframed (audit #17 wording): the v1–v3 loop showed that an all-inventory ordinary-use presentation contract had not been achieved; v4 instead obtains grammatical core-operation equivalence by placing every item in the same autonymic `the word <X>` frame. Its 48/48 approval therefore licenses a bounded mentioned-string instruction micro-world, not presentation inertness across ordinary noun, verb, adjective, and function-word uses. The operation-verb update is a declared-operation-verb context intervention, not yet a denizen-enacted operational move (source and recipient are separate prefix encodings; no execution consequence is measured). The numerical-instrument finding (torch SVD non-convergence) forced a telemetry gate in front of every low-rank result. Audit #17's allocation ruling: run the contextual-prefix baseline, then one bounded multi-position consequence test BEFORE v4 bridge/interchangeability and before a second decoder; do not author a v5 — the next population should be a typed use-frame task.

Alternatives held live and NOT yet run: a second pinned decoder (cheap
replication axis; Round 31 order 11); a multi-position consequence law
(next k∈{4,8} tokens, teacher-forced) as a different notion of consequence
(audit #16's alternative); two-step writeback composition; a full-prefix
contextual baseline is now implemented and queued (order 4). Audit #17 fired
after this note was written: the multi-position consequence law is promoted
to the one bounded test that precedes v4 bridge/interchangeability and the
second decoder (Round 33 registers it); the v4 frame is a bounded
mentioned-string instruction micro-world; no v5 — the next population, when
one is authored, is a typed use-frame task, not another all-inventory
search.

Nothing starts before the two Tier-1 gates (op-update/bridge; SVD telemetry)
and resSA2 finish.

## 2026-08-29 — Population v4 approved and frozen (after v2 and v3 were voided)

- v3 (`a8b14a8`) VOID: all 32 presentation cells passed (Please/Kindly;
  ASCII vs typographic apostrophe, rated near-degenerate), but the controls
  under the orthographic wrapper failed clause 6 (a whole-word operation swap
  vs a one-glyph presentation edit). Adversary principle: co-design pairs and
  controls under a predeclared surface-distance metric.
- v4 (`afd6fcc`, fresh outcome-blind author): metalinguistic frames `{Please|Kindly} plan to OP the word <X>` and `{Hello,|Hi,} please plan to OP the word <X>`, OP ∈ {repeat, omit, capitalize, reverse}; declared metric = aligned surface-word edit distance = 1 for every pair and control; frozen `operation_updates` block. A separate fresh Codex session approved 48/48 checklist cells for grammaticality, preservation of the explicit string-edit instruction, and matched surface-word distance under the common mention frame; this is outcome-blind procedural approval, not 48 independent linguistic observations or proof that Please/Kindly and Hello,/Hi, are pragmatically or latently inert (audit #17). Tokenization PASS; approval block written; raw sha256 `f813f9b2…`, git blob `8845f75c…` in the ledger (`nlm007_fresh_v4_frozen`). The config's top-level 'not approved for capture' note is historical authoring-time text superseded by the structured approval/hash fields.
- Next on this population (Round 31 order 5–8, after the order-4 baseline;
  reordered by audit #17 / Round 33: one bounded multi-position consequence
  test comes first): captures A / B / OP_UPDATE → bridge screen →
  interchangeability → fresh analyses A/B → operation-update analysis; chain
  `run_v4.cmd` written, armed only when the operation-update and bridge code
  pass Tier-1 review.

## 2026-08-29 — Residualization B P_aug-score4 completes the 2×2 table; third launch with the SVD fallback

- Sentinel ',' with the implemented score-4 augmented design; 5074 s of the
  7200 s wall on the third launch (the first two died in the F8 grammar
  block; only the second is directly localized to torch SVD non-convergence
  on the fitted low-rank coefficient matrix at grammar_w1 — audit #17
  erratum; the committed analyzer now falls back to a float64 LAPACK SVD —
  this cell is an amended-implementation cell and is reported as such).
- **F4, F8, F12, F20 pass** the residual-vs-null gate (X⊥ ridge 0.52–0.57 vs
  strongest residual null 0.06–0.09; block-first leads cos +0.46–0.51,
  skill +0.40–0.44, KL-rank +0.45–0.54; 6–8/8 full keys; no collapse).
  **F0 fails** (cos +0.33 but skill LB −0.04; 4/8 full keys).
- Registered-static-metadata + carrier-summary nuisance arm (P_aug → Δ)
  0.42–0.64 by layer (not a presentation-only component).
- Same-run common-scale ratios exceed 0.5 at the median at F4–F20 (F0 wide).
- Reading (audit #16 discipline): within one decoder and one authored
  population, under both sentinels and both registered nuisance designs,
  X⊥ retains predictive association with Δ⊥ beyond the four X-free lexical
  nulls at F4–F20; F0 passes only for the A score-4 cell (sparse keys). These
  are four correlated same-population sensitivities, not replications; they
  identify neither operational state nor presentation independence.
- The chain continues automatically: resSA2 (patched A-static common-scale
  cell), then run_r31.cmd (probe-1 screens, P_aug-full cell A,
  contextual-prefix screens and completions). Codex round 32 adjudicated the
  cell as amended-implementation / SVD-telemetry-incomplete and forbade
  further low-rank output before an SVD telemetry gate; run_r31.cmd was
  disarmed (ledger `nlm007_r31_chain_disarmed_pending_svd_gate`).

## 2026-08-29 — Round 31 adopts audit #16; v2 population authored, then voided by the independent adversary

- Round 31 (Codex, `71b5ce3`): audit #16 adopted verbatim; fresh v1 voided for
  confirmatory probes 2–4 (ledger `nlm007_fresh_v1_voided`); the ` not`
  insertion withdrawn as the second move, replaced by the operation-verb
  update in a metalinguistic micro-world (repeat→omit, capitalize→reverse
  under matched wrappers); contextual-prefix X-free baseline (token_ids_v1)
  and calibration-only bridge ladder registered as analyzer modes; corrected
  order 0–11 (baseline before any fresh capture; bridge before
  interchangeability; hostile lower bound must exceed τ; move-norm floor).
- v2 (`lexical_probe_fresh_v2.json`, Codex as outcome-blind author):
  `Please/Kindly | For reference,/For clarity, … plan to {repeat|omit|
  capitalize|reverse} the word <X>`; tokenization pre-check passed (slot
  template-final, ` not` clean). The independent linguistic adversary
  (fresh session, no model access) VOIDED it: all 16 pair-2 cells fail — “For
  reference” vs “For clarity” introduce distinguishable discourse purposes
  that can scope over the operation; pair-1 (Please/Kindly) and all controls
  pass. Design principle for v3 (verbatim): vary only a scope-fixed form whose
  interpretation cannot supply a reason, goal, condition, or other content for
  the requested operation; semantic inertness must hold independently in every
  POS cell. v3 was authored from scratch by a fresh session (later voided
  on control edit-magnitude; see the v4 entry above).
- Implementation: the reviewed analyzer (probe-1 options, insertion source,
  interchangeability, SVD fallback) is committed (`0c774c0`); the
  contextual-prefix baseline is implemented and under Tier-1 review; the
  operation-update move is at its design gate. B-aug's third launch passed
  the fold that failed twice (fallback held); resSA2 follows automatically.

## 2026-08-29 — Re-contextualization #16 (2-hour step-back; audit #16 fired and adopted in Round 31)

Live question unchanged: is the surviving X⊥→Δ⊥ predictability in one small
decoder an operational-state relation, a smooth presentation/lexical relation
the registered designs miss, or an implementation artefact — and what does
either answer say about holes hostile to structured reasoning.

What still holds: A-static, P_aug-score4, and B-static are bounded, correlated same-population sensitivities in one decoder; registered static metadata predict raw displacement across both sentinels, X-linked residual predictability survives the tested nuisance fits at F4–F20, operational state is not identified, the inherited ordering statistic is a local measurement hole, and raw F0 remains identity/token dominated. (Audit #16 wording.)

What reframed this cycle: (1) audit #15 moved the program off the
observational residualization axis onto external axes (fresh population,
second move, interchangeability) — the queue is now about whether the
relation is a property of the authored manifold or of the space; (2) the B-aug analysis failed twice in the F8 grammar block; the preserved traceback localizes the second failure to torch SVD of the fitted low-rank coefficient matrix at grammar_w1, while the later ledger row attributes the first loss to the same defect. This is repeatable numerical-instrument non-robustness, not evidence of ill-conditioned X⊥ until finite-input and spectral diagnostics localize the cause (audit #16 wording); (3) the frozen fresh population fails its pre-capture linguistic design gate: none of the eight pairs establishes coherent presentation-only equivalence across all four word classes, and several change syntactic licensing, modality, definiteness, degree, or quantification. The population is void for confirmatory probes 2–4 and may be retained unchanged only as an exploratory mixed-frame stress set; no noun-only or pair-only post hoc rescue is confirmatory (audit #16 ruling).

Alternatives held live (not yet run; CPU-only): a second pinned decoder as a
cheap replication axis; a full-prefix contextual X-free baseline (audit #15);
a two-step writeback composition test; a wholly new population authored under a predeclared all-POS linguistic contract, reviewed by an independent linguistic adversary before hashing and capture;
and the direct "consequence-sensitive divergence" question — whether the KL
readout at one position is the right notion of consequence for a denizen at
all, or whether a multi-position law (next k tokens) is the honest one.

Nothing in probes 2–4 starts on fresh v1: finish the protected running chain, audit the B-aug numerical amendment, repair and re-review probe 1, then register and freeze a linguistically valid replacement population before capture. (Audit #16; adopted in Round 31.)

## 2026-08-29 — Round 29 reorders the queue; fresh matched population frozen

- Codex Round 29 (`4907a85`), adopting audit #15: Round 23's literal `P_aug`
  meant full carrier mean + rank-4 scores → the observed run is
  `P_aug-score4` (outcome-clean, transductive, contract-validity qualified);
  `P_aug-full` is unrun. New fixed order: (0) finish resAB → resSA2;
  (1) carrier-summary rank ladder {1,2,4,8,full} + nonlinear carrier kernel
  as a cosine screen, plus one preselected full-law cell (sentinel A,
  `P_aug-full`); (2) fresh frozen population + ` not`-insertion capture;
  (3) matched presentation interchangeability (stable vs hostile-hole gates);
  (4) fresh-population analysis; (5) different-move analysis; (6) registered
  X-free field ×4; (7) Freedman–Lane on A-static only, conditionally; (8)
  second pinned decoder. The armed X-free chain was killed.
- Probe 2 population (`experiments/config/lexical_probe_fresh_v1.json`;
  families question / instruction / comparison / enumeration, 8 matched
  presentation pairs, 4 operational control pairs, same 80 words; ` not`
  (id 537) appends as exactly one token to every prefix) was prospectively authored and committed before any new capture or score, but not independently blind to prior results. Its declared digest `c6edaa92…` is not the raw file SHA-256 (`12c72401…`), and audit #16 voids its eight-pair presentation-equivalence claim before capture. The file stays unchanged as an exploratory stress set only; a v2 population under a predeclared all-POS contract replaces it (Round 31).
- Round 30 completed its review and ruled probe 1 NOT-READY; its six repairs remain a prerequisite, while probes 2–4 are additionally paused by audit #16's population-validity failure.

## 2026-08-29 — Residualization B-static: a correlated second-sentinel check takes the same bounded P_static branch

- Sentinel ',' with P_static; 4598 s of the 7200 s wall; unseen-word folds,
  K = 13, class-preserving crossed bootstrap, same-run raw shadow and
  common-scale retention block.
- **F4, F8, F12, F20 pass** the residual-vs-null gate (X⊥ ridge 0.52–0.58 vs
  strongest residual null 0.06–0.09; block-first leads cos +0.45–0.50,
  skill +0.35–0.42, KL-rank +0.40–0.58; 8/8 positive keys at every passing
  layer; no collapse). **F0 fails** (cosine lead +0.27 but skill negative and
  KL-rank LB < 0) — as under A-static.
- Registered-static-metadata arm (`P_static → Delta`) cosine is 0.41–0.63 by layer; this is not a pure presentation component or variance share.
- All twelve F4–F20 residual/raw predictive-margin ratio medians exceed 0.5; eleven lower bounds do so, with F4 continuous KL at 0.426. These are robustness ratios, not retained signal, state, or mediation.
- Reading (audit #16 wording): across the correlated A/B static runs, registered block/length/position metadata predict raw displacement, and X⊥ retains predictive association with Delta⊥ beyond four X-free lexical nulls at F4–F20. This is a two-sentinel robustness result within one decoder and authored population, not independent replication, state, or presentation independence.

## 2026-08-29 — Re-contextualization #15 (2-hour step-back; audit #15 running)

Live question (one project): is the surviving X⊥→Δ⊥ predictability in one small
decoder an operational-state relation, a smooth presentation/lexical relation
the registered designs miss, or an implementation artefact of residual
geometry — and what does either answer say about holes hostile to structured
reasoning.

Current bounded result: in the same sentinel-A cells and folds, X-linked residual predictability survives the registered `P_static` fit and the implemented rank-4-score `P_aug` fit at F4–F20. The raw F0 transition remains identity/token dominated, and the specific across-word within-carrier pairwise-KL ordering statistic is insensitive in this probe. These are correlated sensitivity results, not replications, and they identify neither operational state nor a native law. (Audit #15 wording.)

What is reframed by A-aug (audit #15 wording): A-aug shows only that one registered P_static fit and one implemented, contract-qualified P_aug-score4 sensitivity on the same sentinel-A cells do not absorb the `X⊥–Δ⊥` association. It does not show that every finite presentation design will leave a predictive residual or that the residual is operational state. The two Round 27 comparators are the next within-dataset controls: the registered X-free interaction field tests whether a fixed low-rank presentation/lexical family can reproduce the association without cell-level `X⊥`, and the refitted permutation null tests whether the observed alignment exceeds residual-geometry null refits. Neither is decisive for operational state, because an aligned cell-level prefix/carrier fingerprint can beat both.

Tunnel-vision check — honest: everything queued is one decoder, one template
population, one move, one sentinel pair. Live alternatives held open:
(a) A second pinned decoder is a relatively cheap replication check for decoder specificity; one additional decoder cannot decide whether the relation is generic or identify its mechanism.
(b) the relation is template-population-specific → a fresh authored style
   family, held out entirely, is a cheaper test than another comparator;
(c) A two-step writeback test requires a new intervention capture rather than existing captures alone, but current timings suggest roughly 10–20 minutes of CPU capture plus about one hour of targeted scoring.
(d) The present evidence does not yet prove a structural quotient hole. Failure of two nested nuisance fits shows that the chosen coordinates are incomplete; it does not show that lexical, presentation, and operational coordinates are entangled by construction. The cheapest sharpening is a linguistically validated interchangeability test with matched controls and a predeclared calibration-only bridge ladder; raw scalar swap failure alone cannot establish a hostile quotient hole.

Audit #15 (verbatim in .codex_audit15.md; adopted into theory/EXPERIMENTS.md):
the queue is "strongly tunnel-visioned" — one decoder, one authored template
population, one punctuation-append move, one self-readout; the strongest
alternative both queued comparators miss is a high-dimensional prefix/carrier
fingerprint (aligned, cell-level, compatible with unseen-word transfer and law
improvement); CPU-only alternatives it ranks ahead of the ~100 CPU-h
Freedman–Lane expansion: full carrier-summary rank ladder {1,2,4,8,full} +
nonlinear carrier kernel; contextual X-free baseline from full tokenized
prefix features; a fresh frozen template population (16×80); a different
move (content-bearing append, negation/operator insertion, binding update);
a matched presentation-interchangeability test; two-step writeback; second
pinned decoder. The order change is a Codex decision (round 29).

## 2026-08-29 — Residualization A P_aug-score4: residual predictability survives at F4–F20; F0 remains sparse and raw-identity dominated

- 4738 s of the 7200 s wall; sentinel '.'; `P_aug` uses `P_static` plus at most four scores obtained by
  projecting a leave-calibration-word-pool carrier mean of `X` into a basis
  learned from calibration carriers; the full carrier-mean vector is not
  appended (audit #15); cross-fitted out of both X and Δ; unseen-word folds; K = 13; class-preserving
  crossed bootstrap; same-run raw four-null shadow and common-scale
  retention block present.
- All five correlated checkpoints meet the registered aggregate residual-vs-null gate. F0 is qualitatively weaker — only 2/8 keys clear the full per-key gate — and is not an independent confirmation of the F4–F20 profile. F0 numbers
  (residual cosine 0.34 vs −0.01; block-first skill +0.16 [LB 0.02],
  KL-rank +0.30 [0.12]) — the score-only nuisance fit changes the residual target and reference geometry and exposes a positive pooled F0 association, but only 2/8 keys clear the full gate, so it does not repair the raw identity-dominated transition. F4–F20: X⊥-ridge 0.56–0.62 vs 0.06–0.07;
  block-first leads cos +0.50–0.56, skill +0.35–0.46, KL-rank +0.43–0.56;
  6–8/8 keys; no block collapse.
- `P_aug` nuisance-only carrier-summary arm (P_aug → Δ) 0.45–0.64 by layer; because its scores are derived from carrier-level X, this is not a presentation-only estimate or a variance share.
- The implemented P_aug-score4 run is internally valid for its score-only sensitivity but does not instantiate Round 23's literal full-mean-plus-score P_aug contract; P_aug-full remains unrun. Under Round 23's predeclared readings this is the non-collapse branch for the implemented design: the registered static and rank-4-score nuisance fits do not absorb the association; broader presentation, carrier-geometry, and prefix-fingerprint explanations remain fully live
  (unmeasured presentation remains possible; audit #14's Freedman–Lane
  residual-geometry null and calibration-only presentation/lexical
  comparator are the next preregistered tests). Wording per audits
  #13/#14: residual predictability of X⊥ beyond residualized X-free
  lexical nulls after removal of the registered static AND augmented
  coordinates; not presentation-independence; not state.
- B-static running; then B-aug; then the patched A-static.

## 2026-08-29 — Audit #14 adopted: A-static upheld; Round 26's mediation sentence withdrawn

- Upheld: F4–F20 pass; not a residual-geometry mirage (ridge cosine falls
  under residualization while the nulls collapse; shuffle q95 ≤ 0.13;
  residual normalized error 0.78–0.83).
- Withdrawn (over-read in the kill direction): "much of the raw lead may
  have been presentation-mediated". Licensed joint statement: registered
  static coordinates predict held-out raw displacement; after their
  cross-fitted removal X⊥ still predicts Δ⊥ and its reassembled response-law
  consequence beyond the residual X-free nulls at F4–F20; the overlap
  between presentation and the raw ridge lead is not identified.
- Gate is too easy for a *state* claim: next comparators to preregister are
  a fully refitted Freedman–Lane residual-geometry null and a flexible
  calibration-only P_aug/lexical interaction field without cell-level X⊥.
- Demo copy corrected again (nine verbatim replacements) and republished.
- The two NOTEBOOK entries carrying the withdrawn phrase (Round 26 note;
  re-contextualization #14) are superseded by this entry.

## 2026-08-29 — Re-contextualization #14 (A-static in; P_aug running; audit #14 fired)

*Audit #14 (Tier-3, `theory/EXPERIMENTS.md`, ledger `nlm007_audit14_adopted`) withdrew the sentence "much of the raw lead may have been presentation-mediated" as an over-read and replaced the ruling; read this paragraph as corrected there.*

- **Central bet + second lens:** native mathematics from what a denizen must
  invent; the holes that make this space hostile to structured reasoning
  and what the next latent space must change.
- **Live question:** after the registered template coordinates are removed
  from both state and displacement, X⊥ still predicts Δ⊥ far beyond every
  residual content null (F4–F20). Does that survive the augmented
  presentation design (carrier mean + carrier subspace) and the ',' arm?
  And what, jointly, do the presentation-only arm (0.43–0.63) and the
  residual lead license — "presentation is a large part of the raw move,
  and what remains is still X-predictable" — without either side
  over-reading?
- **What reframes:** the pooled story has quietly changed shape. The
  earlier framing "content vs context" is now "content vs presentation vs
  the residual of X after presentation" — three layers, of which content
  is the smallest, presentation is large, and the X⊥ residual is what a
  denizen would actually need a map of. The demo's 42%-style intuition was
  wrong (cosine ≠ variance), and Round 26's "much of the raw lead may have
  been presentation" may itself be an over-read in the other direction —
  audit #14 is asked to fix the joint statement.
- **Alternatives held live:** (a) residual-space cosines are geometrically
  easy (nulls at ~0.06 because residual targets are near-zero-mean) — the
  fair residual comparator may be a residual-space shared mean or a
  P-only predictor scored in residual space; (b) unmeasured presentation
  remains in X⊥ (P_aug tests part of this); (c) presentation is part of
  operational state and quotienting it removes physics — the operational-
  equivalence target (same moves, same consequences) is the honest
  definition; (d) second family; (e) multi-step composition.
- **Ecosystem deposit:** "cosine of a presentation-only predictor is not a
  variance share; state 'presentation predicts the move at c' and 'the
  residual is X-predictable at r' separately" → `_meta/INDEX.md`.

## 2026-08-29 — Round 26: A-static adjudicated; the presentation-only arm revises the earlier reading

*Audit #14 (Tier-3, `theory/EXPERIMENTS.md`, ledger `nlm007_audit14_adopted`) withdrew the sentence "much of the raw lead may have been presentation-mediated" as an over-read and replaced the ruling; read this paragraph as corrected there.*

- P_static took the non-collapse branch of the primary gate at F4–F20; this
  proves neither operational state nor presentation-independence.
- The presentation-only arm (0.43–0.63 cosine) materially revises the
  earlier unseen-word interpretation: much of the raw X-conditioned lead
  may have been presentation-mediated; residualization shows only that the
  registered static coordinates do not explain all of it.
- For resSA only "the predeclared robustness marker is mechanically met" is
  admissible; a patched A-static rerun (common-scale retention) is required
  for any A-static or four-cell retention claim — queued after B-aug.
- Presentation sensitivity is proven locally; presentation/state
  inseparability remains unproven. Read order: A-aug → B-static → B-aug →
  patched A-static.

## 2026-08-29 — Residualization A-static: the X⊥ lead survives removal of the registered template coordinates

- 4406 s of the 7200 s wall; sentinel '.'; P_static (block one-hot, lengths,
  positions) cross-fitted out of both X and Δ; unseen-word folds; K = 13
  universe; class-preserving crossed bootstrap.
- **F4, F8, F12, F20 pass** the residual-vs-null gate: X⊥-ridge 0.56–0.62
  residual cosine vs 0.06–0.07 for the strongest residual X-free null;
  block-first leads cos +0.50–0.56, skill +0.31–0.48, KL-rank +0.40–0.61
  (lower bounds > 0.17); 6–8/8 keys positive; no block collapse. F0 fails
  (skill negative).
- Presentation-only arm (P_static → Δ) held-out cosine 0.43–0.63 by layer:
  the registered template coordinates are a large part of the raw
  displacement; what remains after their removal is still predicted from
  X⊥ far beyond any content null.
- Retention: "the predeclared robustness marker is mechanically met" on all
  three endpoints at F4–F20 (audit #13: not a fraction of signal; this run
  predates the common-scale block, which A-aug and the B runs carry).
- Remaining: P_aug (adds the leave-word-out carrier mean and a rank-4
  carrier subspace), both B arms; Codex round 26 adjudicates A-static now.

## 2026-08-29 — Audit #13 adopted: demo corrected; retention marker not commensurate

- The published demo over-claimed ("context state", "context takes over",
  "manufactures", "presentation explains 0.42"); every replacement adopted
  verbatim and republished at the same URL; the nearest-state predictor is
  now coloured as X-conditioned; named-word rows labelled as selected.
- Retention marker: raw and residual margins live on different scales;
  until the common-scale repair is in place the residualization runs may
  say only "the predeclared robustness marker is mechanically met". The
  raw shadow remains valid for the amended unseen-word comparison; the
  residual-vs-null gate and the law reassembly are coherent.
- Equalized A wording tightened (defect concern resolved; calibration-
  selected comparator; 0.002–0.009 above the mean).
- Reverse-tunnel note adopted: the X-conditioned advantage can no longer be
  dismissed as lookup or artifact; presentation may be part of state.

## 2026-08-29 — Corrected equalized LOCO addendum, sentinel ',': F12/F20 pass; baselines just above the shared mean

- 4196 s of the 4500 s wall. Contract-correct equalized baselines sit
  0.002–0.007 above the shared mean; ridge's lead is unchanged: **F12 and
  F20 pass** against the stronger equalized baseline, F4/F8 miss on
  skill/KL-rank lower bounds (cosine leads hold), F0 fails. Run-level
  positive (2/5). Both arms of the addendum are now contract-correct and
  agree with the defect-affected runs' numbers — the defect changed the
  baselines by ≤0.01 and no verdict.
- The residualization chain (A-static → A-aug → B-static → B-aug, 120-min
  wall each) has started.

## 2026-08-29 — Re-contextualization #13 (residualization launching; demo audited)

*Wording per audit #13 (see the 2026-08-29 audit #13 entry): "the context predictor" reads "the X-conditioned predictor"; the retention marker is not commensurate, so residualization runs may say only "the predeclared robustness marker is mechanically met".*

- **Central bet + second lens:** native mathematics from what a denizen must
  invent; holes hostile to structured reasoning; the next latent space.
- **Live question:** the forward step's regularity transfers across
  carriers, families and unseen words, and every content null sits at the
  mean — is what X carries operational state or a smooth presentation
  coordinate? The four residualization runs (static/aug × two sentinels)
  are the first direct test; they start automatically after the corrected
  equalized rerun B.
- **What reframes:** the single-template walkthroughs for the demo showed
  something the pooled numbers hide — in a gloss template the context
  predictor wins on the state but the next-token law barely moves; in
  continuation and grammar templates the law moves a lot. "Consequential
  motion" varies by template family, not just by layer. That is a
  template-level version of the middle-depth finding, and it is exactly
  the kind of structure a next-generation latent space would need to make
  explicit: when does a move matter to the world's response?
- **Alternatives held live:** (a) the residual field recovers a smooth
  presentation coordinate P_static/P_aug miss (unmeasured presentation);
  (b) the gloss/continuation difference is a readout-sensitivity artifact
  rather than a world property; (c) multi-step composition may fail even
  if one-step prediction holds; (d) a second family may reorder everything;
  (e) response-space geometry ("same place" = same law) as the native
  metric — the demo's third panel is a first look at exactly that object.
- **Ecosystem deposit:** "whether a move is consequential varies by
  context family, not only by depth — measure consequence per family" →
  `_meta/INDEX.md`.

## 2026-08-29 — Corrected equalized LOCO addendum, sentinel '.': ridge lead unchanged under contract-correct baselines

*Tightened by audit #13 (see the 2026-08-29 audit #13 entry): "audit #11's inner-centre defect concern is resolved by the corrected sentinel-A data", the comparator is the calibration-selected equalized comparator, baselines roughly 0.002–0.009 above the shared mean.*

- 3753 s of the 4500 s wall. With the audit #11 fix (inner centre = the
  inner training carriers' own mean; comparator frozen by calibration
  score), the equalized baselines (word-only one-hot ridge; shrunk word
  mean) no longer collapse exactly onto the shared mean — they land
  0.003–0.01 above it (e.g. F8: shared 0.499, word-only ridge 0.506, shrunk
  0.508, ridge 0.620). The per-word lexical component captured by these
  estimators is small but not identically zero; audit #11's "forced
  maximal shrinkage" concern is resolved by data rather than by wording.
- Gated against the stronger equalized baseline: **F4, F8, F12, F20 pass**
  (cos +0.09–0.13, skill +0.23–0.31, KL-rank +0.30–0.43, LBs > 0.08;
  11–14/16 carriers); F0 fails. Run-level positive. The ',' arm follows.

## 2026-08-29 — Audit #12 adopted: unseen-word gate is mechanical-only until the bootstrap is contract-correct and the lexical nulls are stronger

- Status of both unseen-word runs: mechanical pass under the recorded
  reduction; formal gate pending a class-preserving, crossed word bootstrap
  (being implemented) and stronger X-free lexical nulls (frozen-embedding→Δ
  ridge; embedding-conditioned kernel; k ladder — being implemented).
- Wording: "not exact held-out-word lookup and not the tested lexical
  interpolator"; the positive object is X-conditioned residual
  predictability transferring across held-out words and blocks; F0
  "non-qualifying, continuation the strongest local failure pattern".
- Strongest rival (verbatim in EXPERIMENTS.md): X contains smooth lexical
  and presentation coordinates along which the later displacement varies;
  ridge/kernel recover that geometry; the coarse nulls collapse for
  coarseness, not because the variation is operational state.

## 2026-08-29 — Unseen-word run, sentinel ',': F4/F8/F12/F20 pass — both arms clear the criterion

- 2256 s. Same structure as the '.' arm: on disjoint held-out words the
  stronger X-free lexical null sits at the shared mean at every layer; ridge
  leads it by cos +0.11–0.17, skill +0.31–0.41, KL-rank +0.31–0.52 (block-
  first lower bounds > 0.09); 5–8/8 keys pass the full per-key gate, 8/8
  positive at F12/F20; no block collapse; F0 fails (cos lead 0.018).
- Both sentinels meet the Round 22 two-of-five criterion with four layers.
  The forward-step regularity of this decoder generalizes across carriers,
  across style families, and across word identities it never saw; every
  content null sits at the mean. Adopted wording remains: X-conditioned
  residual predictability, generalizing across unseen lexical identities;
  not yet separated from a smooth presentation coordinate; one decoder.
- Corrected equalized reruns (locoeq2A/B) now executing; Codex round 23
  adjudicates the unseen pair and predeclares residualization and the
  second model family.

## 2026-08-29 — Re-contextualization #12 (unseen words in; audit #12 fired)

- **Central bet + second lens:** native mathematics from what a denizen must
  invent; holes hostile to structured reasoning and what the next space must
  change.
- **Live question:** the forward-step regularity survives unseen words
  (sentinel '.'; ',' finishing). What remains between it and a "law": is it
  a property of the contextual state or of a smooth presentation coordinate
  (residualization, next), and is it a property of this decoder or of
  decoders (second family, after).
- **What reframes:** the sequence of nulls this program has run — identity,
  shared mean, word-mean, class mean, word-only embedding kNN, word-only
  ridge, shrunk word mean, alignment-destroying permutation, three-carrier
  block mean — all sit at the shared mean on the forward move except the
  identity (which is catastrophic there). Only X-conditioned predictors
  move. The honest statement is narrow (audit #11): X-conditioned residual
  predictability, generalizing across carriers, families, and now words.
  The old "native law" ambition has become a concrete object with three
  remaining tests, which is progress of the right kind.
- **Alternatives held live:** (a) embedding-neighbourhood interpolation —
  an unseen word is near seen words in embedding space; audit #12 asked;
  (b) a stronger X-free lexical model (embedding→displacement ridge) may
  close part of the gap; (c) the sentinel pair may still share style
  ('.' and ',' are both punctuation) — a non-punctuation sentinel would
  test it; (d) the response-space geometry ("same place = same law") as a
  native metric; (e) multi-step closure — a one-step law is not yet
  navigation; the denizen needs composition (F4→F8→F12 along the token
  clock), never tested.
- **Ecosystem:** "when every content null sits at the mean, the object is
  X-conditioned predictability; name it that, not a law" → `_meta`.

## 2026-08-29 — Unseen-word run, sentinel '.': F4/F8/F12/F20 pass the full gate on words never seen

*Qualified by audit #12 (see the 2026-08-29 audit #12 entry): the pass is mechanical under the recorded reduction, formal gate pending a contract-correct bootstrap; "not word lookup and not class lookup" reads "not exact held-out-word lookup and not the tested lexical interpolator".*

- 2239 s; eight block × word-fold keys; support 1.0; calibration and
  held-out word identities disjoint; lexical nulls = class mean and
  frozen-input-embedding kNN (both ≈ shared mean at every layer).
- Under the Round 22 gate (≥0.02 over the stronger X-free lexical null with
  positive lower bounds on cosine, law skill, K = 11 KL-rank; block-first
  pooled contrast positive; ≥6/8 keys; no block collapse): **F4, F8, F12, F20
  pass** — block-first pooled leads cos +0.14–0.19, skill +0.33–0.47, KL-rank
  +0.35–0.57 (lower bounds > 0.12); 7–8/8 keys positive; F0 fails (cos lead
  0.019; the continuation block collapses). Two of five met with four.
- Reading (bounded per audit #10/#11): the forward-step regularity survives
  lexical novelty — it is not word lookup and not class lookup. What X
  carries about the next step generalizes across words it never saw, with a
  ~0.06 drop from the seen-word runs. Still one decoder, one style-family
  set; state vs smooth style code unresolved (residualization next).
- Sentinel ',' arm running; corrected equalized reruns follow; Codex round
  23 adjudicates all.

## 2026-08-29 — Equalized LOCO addendum, sentinel ',' (defect-affected; descriptive only)

- 2977 s. Same pattern as the '.' arm under the audit #11 defect (inner
  centre included the validation carrier): equalized baselines equal the
  shared mean at every layer; F12/F20 pass the mechanical gate, F4/F8 miss
  on skill/KL-rank lower bounds, F0 fails. Outer margins are descriptive
  only; the corrected rerun (locoeq2A/B) is queued behind the unseen-word
  runs, which are now executing.

## 2026-08-29 — Audit #11 adopted: equalized addendum has an inner-centre bug; wording withdrawn

- The equalized baselines' inner selection centred on the outer three-carrier
  mean (contains the validation carrier) → maximal shrinkage forced by
  construction; comparator chosen on held-out outcomes. Fixed in the
  analyzer; both arms rerun behind the running chain. The '.' addendum's
  outer margins stand as descriptive numbers only.
- Withdrawn from my previous two entries: "no per-word lexical signal",
  "variance objection answered", "the forward step is about context, not
  content", "the state-conditioned component is large". Adopted wording:
  the word-conditioned component captured by the tested estimators is
  negligible in this design; the positive object is X-conditioned residual
  predictability. The `_meta` deposit is corrected to match.
- LOCO B precision: F8 misses skill only (KL-rank LB +0.021).
- Second lens (auditor): the narrower true statement — lexical content is not
  a sufficient predictor of the later forward step; context-bearing X
  contains predictable variation that word-conditioned means do not capture;
  the next latent space must define "same place" by interchangeability of
  moves and response laws, not lexical identity or representational
  similarity.

## 2026-08-29 — Re-contextualization #11 (equalized addendum in; audit #11 fired)

*Superseded in part by audit #11 (see the 2026-08-29 audit #11 entry): the equalized-addendum inner centre included the validation carrier, so "context, not content" and "content nulls collapse to the shared mean" are withdrawn; the addendum's margins are descriptive only. Audit #13 also withdraws "governed by context state"; the object is X-conditioned residual predictability.*

- **Central bet + second lens:** native mathematics from what a denizen must
  invent; holes that make this space hostile to structured reasoning.
- **Live question:** the forward step's within-family regularity is not
  lexical (equalized lexical baselines collapse to the shared mean) — so it
  is either the contextual state or a smooth style coordinate. The
  unseen-word runs (in the chain) remove lexical lookup entirely; the
  residualization control after them is the only thing that can separate
  state from style, and audit #10 already warned the separation may be
  ill-posed here.
- **What reframes earlier work:** every lexical null in this program has
  come out at the shared mean on the forward move (word-conditioned mean,
  class mean, word-only kNN, word-only ridge, shrunk word mean). The
  forward step of this world is about context, not content: what the next
  position does depends on the state the context has built, not on which
  word was inserted. Under the second lens this is a candidate structural
  fact — and possibly a hole: a denizen cannot navigate by content alone,
  because content barely moves the next step; only context does.
- **Alternatives held live:** (a) maximal shrinkage is forced by selecting on
  two-carrier means (audit #11 asked); (b) the only surviving competitor is
  the shared mean, so the LOCO gate may be trivially passable — a fair
  competitor might be a carrier-code-only or style-coordinate predictor;
  (c) a second family may show a different content/context balance — the
  first cross-model native quantity would be "how much of the next step is
  content"; (d) the response-space geometry idea (same place = same law)
  would make this balance a metric property.
- **Ecosystem deposit:** "on the forward move, content nulls collapse to the
  shared mean; the next step is governed by context state" → `_meta`.

## 2026-08-29 — Equalized LOCO addendum, sentinel '.': the lexical baselines collapse to the shared mean; ridge's lead is unchanged

*Superseded in part by audit #11 (see the 2026-08-29 audit #11 entry): inner-centre defect; "no per-word signal" and "variance objection answered" are withdrawn; outer margins descriptive only; corrected rerun queued.*

- 2911 s of the 4500 s wall. Both equalized X-free baselines (word-only
  one-hot ridge with inner-selected λ; shrunk word mean with inner-selected
  α) select maximal shrinkage at every layer and equal the shared mean to
  three decimals: within a style family, three carriers carry no per-word
  signal about the forward displacement beyond the family's shared shift.
- Gated against the stronger equalized baseline: **F4, F8, F12, F20 pass**
  (pooled ridge − baseline: cos +0.09–0.13, skill +0.23–0.30, KL-rank
  +0.26–0.34, all lower bounds > 0.08; 11–14/16 carriers pass all three);
  F0 fails. Run-level positive.
- Reading: audit #10's variance objection is answered — the block-word mean
  was not losing to noise; there was nothing lexical to estimate. What X
  predicts within a family is not word identity; it is something carried by
  the contextual state (state or smooth style code — still unresolved;
  residualization remains the next control after unseen words).
- Sentinel ',' addendum and the two unseen-word runs follow in the chain.

## 2026-08-28 — LOCO control, sentinel ',': F12/F20 pass; weaker than the '.' arm

- 3091 s of the 4500 s wall; support 1.0. **F12 and F20 pass** the Round 21
  rule (pooled ridge − per-word block mean: cos +0.07–0.10, skill +0.15–0.20,
  KL-rank +0.20–0.26, lower bounds > 0; 12–13/16 carriers pass all three).
  F4 and F8 keep cosine leads (+0.08–0.11, LB > 0.06) but miss on skill /
  KL-rank lower bounds; F0 fails. Run-level positive (2/5).
- Both arms positive; the '.' arm at four layers, the ',' arm at two. Under
  audit #10 the wording stays: on seen words, within a style family, X
  predicts a held-out carrier's forward step better than the three-carrier
  per-word family mean — a variance-disadvantaged baseline; equalized
  word-only baselines are owed before interpretation, and LOCO cannot
  separate state from a smooth style code.
- Codex round 22 adjudicates both arms and predeclares the unseen-word run
  (lexical nulls, K = 11 universe, block-first bootstrap all implemented).

## 2026-08-28 — Audit #10 adopted: LOCO bounded; unseen-word branch needs a lexical null; second-lens table

- LOCO A = "X predicts a held-out carrier's displacement and consequence
  better than the three-carrier per-word family mean at F4–F20" — a
  variance-disadvantaged baseline; equalized X-free lexical baselines
  (word-only ridge; shrunk word mean) required before interpretation;
  block-first bootstrap for any cross-family statement; LOCO cannot separate
  state from a smooth style code — residualization is the next control.
- Unseen-word branch: correct mechanics, no lexical null once the word-mean
  is dropped → class-mean displacement null and a word-only input-embedding
  predictor added as the primary X-free baselines; fixed rank universe;
  fail-fast asserts; block-first pooled bootstrap.
- Second lens (auditor's table, adopted): proven — identity-dominated input
  transition; ordering-saturated readout (for our endpoint). Unproven —
  presentation entangled with state (strong concern); family-only laws (not
  shown; whole-block transfer works); motion invisible to the response law
  (readout-specific). The serious hole: no stable quotient separating
  lexical content, presentation, operational state, and consequential
  motion — "we may have incorrectly declared differently presented states
  to be the same place." Next-generation requirements recorded verbatim.

## 2026-08-28 — Re-contextualization #10 (LOCO A in; second lens active)

- **Central bet:** native mathematics of latent spaces from what a denizen
  must invent; **second lens:** holes that make this space hostile to
  structured reasoning, and what the next latent space must change.
- **Live question:** is the forward step's regularity a property of the
  state or of its presentation — and, under the second lens, is
  "presentation entangled with state" a hole or simply what state *is* in
  a context-conditioned world?
- **What still holds:** forward displacement predictable beyond word and
  token identity from F4 in both arms; within-family LOCO positive at
  F4–F20 (sentinel '.'); exact completion routing; nonpass (not kill) under
  the historical ordering gate.
- **What reframes:** the LOCO result narrows the nuisance rival to "carrier
  identity encoded in X predicts carrier-specific displacement" — which is
  hard to distinguish from state-dependence by construction. The
  unseen-word split changes the axis: if the regularity survives words the
  field never saw, it is not a lexical lookup either. The real reframing
  is that the distinction state-vs-presentation may be ill-posed here: a
  denizen's "place" includes the context it is in.
- **Candidate holes (for audit #10 to test):** (1) motion invisible to the
  response law at middle depth — likely a readout property; (2) identity-
  dominated layer transitions — real, but a property of residual streams,
  not a hole per se; (3) presentation entangled with state — real, status
  unclear; (4) laws holding only within template families — not shown
  (whole-block transfer works); (5) ordering-saturated readouts — proven
  for our endpoint, not for the world.
- **Alternatives held live:** the LOCO baseline is a 3-carrier mean (noisy;
  an equalized baseline may close the gap); a response-space geometry
  where "same place" = same law; a second family may have a different
  persistence/consequence profile — the first cross-model native quantity.
- **Ecosystem deposit:** "state vs presentation may be ill-posed for
  context-conditioned representations; test it via unseen identities, not
  via nulls that destroy alignment" → `_meta/INDEX.md`.

## 2026-08-28 — LOCO control, sentinel '.': within-family state information at F4–F20

- 2902 s of the 4500 s wall; support 1.0. Per the Round 21 rule, **F4, F8,
  F12, F20 pass** (pooled ridge − per-word block mean: cosine +0.09–0.13,
  law skill +0.23–0.31, KL-rank +0.29–0.40, all lower bounds > 0.08; 11–15
  of 16 held-out carriers pass all three). F0 fails as predicted (block mean
  ≥ ridge). Run-level within-family diagnostic: positive.
- Reading: inside a style family, with one carrier held out, the state
  predicts that carrier's forward step better than the family's own per-word
  mean displacement — on the displacement, on the law, and on rank. Together
  with the whole-block hold-out (transfer to an unseen family), the
  "presentation-only" rival now has to explain both a cross-family transfer
  and a within-family carrier-specific gain. What remains of it: carrier-
  specific presentation encoded in X that predicts carrier-specific
  displacement — which is close to saying the state knows its context, i.e.
  is state. Codex round 22 rules, with the second lens: is "style entangled
  with state" a hole, or the definition of state in this world?
- Sentinel ',' arm running.

## 2026-08-28 — Second lens added (Devansh): holes, and the next latent space

- Standing instruction: structural properties that make current latent
  spaces hostile to structured reasoning are first-class findings; if
  proven, the constructive program is a next-generation latent space in which
  they are closed. Candidate holes already on the table from NLM-007: motion
  the world's response cannot register (middle-depth displacement invisible to
  the slot law); identity-dominated transitions; presentation entangled with
  state; laws that may hold only within a template family. Each is now a
  question for every Codex round and audit.

## 2026-08-28 — Within-style null, sentinel ',': F8/F12/F20 mechanically pass (diagnostic only)

- 2238 s, support 1.0, K = 7 KL-rank label. Same shape as the '.' arm: the
  within-style null collapses below the shared mean from F4 on (0.21–0.54 vs
  0.45–0.65) while ridge/kernel hold 0.68–0.80; F8/F12/F20 clear the
  mechanical gate, F4 misses, F0 fails.
- Per audit #9 this is an alignment-destruction diagnostic, not a style
  control; no "style-robust" claim. Both arms recorded; Codex round 21
  adjudicates and predeclares the leave-one-carrier-out control (`--loco`,
  implemented, smoke pending).

## 2026-08-28 — Audit #9 adopted: nonpass ≠ kill; KL-rank set defect; style null is a diagnostic only

- "Not met" is a nonpass under the historical contract, not a kill; the
  comma arm falsifies "token/position prevents any qualifying layer".
- KL-rank ranked K = 7 candidates instead of the preregistered 10 (kNN-1/5/20
  omitted): fixed in the analyzer; style-A/B runs labelled K = 7, not
  contract-valid on that endpoint.
- The within-style null is an alignment-destruction diagnostic; "style-robust"
  is withdrawn as a claim. Next fair control: within-family
  leave-one-carrier-out vs per-word/per-block mean displacement (to be
  predeclared by Codex), then residualization, then unseen words, then a
  second family.

## 2026-08-28 — Within-style null, sentinel '.': F4/F8/F20 style-robust mechanically; the null itself is suspect

- 2213 s, support 1.0. Under the Round 20 gate (≥0.02, LB > 0 over the
  word-conditioned mean AND over the within-style-family null, on cosine,
  skill, KL-rank): **F4, F8, F20 pass**; F12 misses one fold's KL-rank LB
  (−0.053); F0 fails (style null = shared mean = word-mean = field).
- KL-rank (new endpoint) separates cleanly where ordering never did: ridge
  0.82–0.90 vs word-mean 0.31–0.41 at F4/F8/F20, LBs > 0.16.
- The within-style null collapses below the shared mean (0.16–0.50 vs
  0.47–0.62): a field refit on a broken pairing predicts the wrong
  carrier's displacement. That makes "beats the null" easy — audit #9 is
  asked whether the null is a straw man and what a fair style control is.
  Note for that ruling: the outer fold already holds out a whole style
  family (the four config blocks), so transfer to a held-out block cannot
  use that block's style code; the residual confound is style shared
  across families.
- Sentinel ',' arm running; Codex round 21 adjudicates both with audit #9.

## 2026-08-28 — Re-contextualization #9 (style null running; audit #9 fired)

- **Central bet:** native mathematics of latent spaces from what a denizen
  must invent. **Live question:** is the world's forward step (last context
  state → next position) governed by a regularity that belongs to the state
  rather than to the presentation (carrier/template style)? The style null
  is the first test; unseen words and a second family follow.
- **What still holds:** forward displacement predictable beyond word and
  token identity from F4 in both sentinel arms; law skill at the sentinel
  registers it; the preregistered two-layer criterion not met for the
  primary arm (Round 20), ordering ruled saturated and replaced
  prospectively by KL-rank.
- **What reframes earlier work:** every gate failure so far has come from
  the ordering endpoint, in every program; the world may have been "saying
  yes" through cosine and skill all along while our consequence endpoint
  could not hear it. The lesson is about endpoints, again: a consequence
  measure must be able to fail for the null and pass for the truth — a
  calibration we never ran for ordering.
- **Alternatives held live:** (a) the within-style permutation null is a
  straw man — a refit field on a broken pairing must predict the wrong
  carrier's displacement and fall below even the mean, so "beats the null"
  is uninformative; a fair style control holds out style *families* or
  residualizes a block code from X (audit #9 asked); (b) style explains the
  lead — testable by a per-block-mean displacement baseline (cheapest);
  (c) the KL-rank endpoint may be biased by including the compared field
  in the ranked set; (d) the whole "law" is one decoder's habit — second
  family; (e) the denizen's map might be over responses, not states.
- **Ecosystem deposit:** "a permutation null that a flexible model trivially
  beats is not a control; calibrate every null by checking it can pass for
  the truth and fail for the confound" → `_meta/INDEX.md`.

## 2026-08-28 — Forward-time move, sentinel ',': F12 and F20 clear the gate; the two arms disagree only at the ordering margin

- 1823 s, support 1.0. Same shape as the '.' arm: F0 token-identity
  dominated; F4–F20 displacement cosine ridge/kernel 0.68–0.80 vs
  word-conditioned mean 0.46–0.66 (LBs > 0.1), law skill at the sentinel
  0.46–0.57 vs 0.01–0.02, shuffle collapses.
- Gate (mechanical): **F12 and F20 pass** for ridge (ordering +0.022–0.074,
  LBs 0.004–0.037 at F12); F8 misses on one fold's ordering LB (−0.002); F4
  on one fold's skill LB. The '.' arm passed F20 only. Two of five layers
  for the same sentinel is met by the control arm, not the primary.
- Reading: the forward step is predictable from the state beyond word and
  token identity at every layer from F4 in both arms; whether a layer
  "qualifies" is decided by ordering lower bounds within ±0.02 of zero.
  Ordering is the binding endpoint in every program so far (layer
  displacement, forward A, forward B). Codex round 20 must rule on whether
  the primary/control asymmetry is a failure of the primary arm or a
  property of the ordering endpoint, before anything is claimed.

## 2026-08-28 — Audit #8 adopted (displacement wording; forward implementation verified)

- Forward-time implementation verified line by line by a fresh auditor
  before its scores were interpreted; the one missing check (A/B unappended
  states identical) passes bit-exactly.
- Displacement wording narrowed and adopted verbatim: kernel captures
  held-out-carrier displacement variation beyond the word-conditioned mean;
  carrier/template vs state dependence unresolved; the carrier shuffle is a
  carrier-alignment diagnostic, not a state-independence null; "the slot law
  barely registers it" is a readout fact; L20 = one bounded qualifying pair.
- Its cheaper controls (style balancing / residualization, within-template
  null, style-held-out split, Y−X decomposition into word/carrier/shared/
  residual, per-layer float32 precision reports) are recorded verbatim in
  EXPERIMENTS.md and enter the queue ahead of any "state-dependent" claim.

## 2026-08-28 — Forward-time move, sentinel '.': state-dependent everywhere, gated only at F20

- Five layers, 2220 s, support 1.0, locality passes under the Round 20
  clause. **F0** token-identity dominated (shared mean = word-conditioned
  mean = 0.67 ≈ field 0.69), as predicted.
- **F4 / F8 / F12:** displacement cosine ridge/kernel 0.71–0.78 vs
  word-conditioned mean 0.48–0.53 (leads +0.17–0.27, clustered LBs > 0.15
  every fold); law skill at the sentinel position 0.39–0.57 vs 0.01–0.02;
  carrier-shuffled null 0.12–0.32 vs field 0.67–0.81; within-carrier oracle
  ~0.98. The world's forward step is strongly state-dependent beyond word
  identity and beyond token identity. But ordering leads are 0.00–0.08 with
  LBs ≤ 0 in half the folds → the three-endpoint gate fails.
- **F20 qualifies** (ridge: +0.16–0.23 / +0.50–0.61 / +0.020–0.058, all
  LBs > 0). One layer; two required for the same sentinel.
- Token-identity control: the '.'-fitted predictor applied to the ','
  target scores 0.43–0.54 vs 0.26–0.30 for the shared mean — the learned
  displacement carries a sentinel-independent, state-dependent component.
- Structural reading: identical to the layer-clock displacement ladder —
  cosine and skill say "large state-dependent motion, registered by the
  law"; ordering says almost nothing until late. Ordering (per-anchor
  concordance of KL orderings across words) is dominated by word identity,
  which every predictor preserves; it is the binding gate in both clocks
  and may be an insensitive endpoint rather than evidence of
  inconsequential motion. Codex round 20 must rule on the endpoint before
  the gate is read as a world fact. Sentinel ',' arm running.

## 2026-08-28 — Re-contextualization #8 (forward-time run in progress)

- **Central bet:** native mathematics of latent spaces from what a denizen
  must invent. **Live question:** what is the denizen's actual step — the
  forward-time move from the last context state to the next position — and
  does it obey a reusable, state-dependent law that the world's response
  registers? The layer clock was the analyst's; the token clock is the
  world's.
- **What still holds:** exact routing of the completion (~1e-5); lexical
  persistence at L0 on every endpoint; identity + calibration-mean
  displacement as a competitive description at L8/L12 (post-hoc rule,
  labelled); state-dependent, nonlinear displacement on its own coordinates
  from L4 on; one gated pair (L20) where the law feels the displacement.
- **What reframes earlier work:** the question "is there a law of motion"
  split into "is there motion" (yes, everywhere from L4) and "does the
  world's response register it" (only late, under the slot readout). The
  forward-time preview at F4/F8 (ridge 0.72–0.78 vs word-mean ~0.5, skill
  ~0.45) suggests the token clock's move is both larger and more
  consequential than the layer clock's — if it holds under the gates, the
  native object is the forward step, and the layer-pair program was
  measuring the wrong move.
- **Alternatives held live:** (a) carrier/template style, not state,
  explains displacement leads (style-balancing control pending); (b) the
  sentinel choice ('.' vs ',') may dominate — the comma arm decides;
  (c) 'consequential motion' may be a readout artifact (ordering is
  saturated) rather than a world property; (d) all of this is one small
  decoder — second family untested; (e) unseen words untested; (f) maybe
  the denizen's map should be over *responses* (laws) rather than states —
  a law-space geometry where 'same place' = same response, which would
  make inconsequential motion literally zero distance.
- **Ecosystem deposit:** "a move can be large in coordinates and invisible
  to the world's response — always measure both, and name which one a claim
  is about" recorded in `_meta/INDEX.md`.

## 2026-08-28 — Displacement ladder: Δ is state-dependent at every depth ≥ L4, but the slot law only feels it late

- Five raw-residual pairs, Δ = Y − X predicted from X, 1750 s of a 5700 s
  wall, support 1.0.
- **L0→L1:** Δ is lexical persistence — the word-conditioned displacement
  mean equals every field (0.948) and the carrier shuffle changes nothing.
- **L8→L9 / L12→L13:** on the displacement's own coordinates the field beats
  the word-conditioned displacement mean by +0.07–0.22 (lower bounds >0.05
  in every fold), the carrier-shuffled null collapses (0.35–0.52 vs
  0.60–0.85), and the minimal class is **kernel** — the displacement is
  state-dependent beyond word identity and not affine. But the slot law
  barely registers it: ordering leads 0.003–0.022, slot-skill lower bounds
  mixed. Gate fails as predicted, for a reason the prediction did not name:
  the identity component saturates the law at middle depth, so the
  denizen's law of motion is invisible to the world's own next-token
  response there.
- **L20→L21 qualifies** (kernel: +0.025–0.051 / +0.13–0.32 / +0.023–0.038,
  all lower bounds >0) — falsifying "small residuals, no complete result".
  Late in the stack the displacement changes the law.
- **L4→L5:** kernel leads on cosine (+0.02–0.03) with tiny ordering
  differences; gate fails as predicted.
- Reading under the guiding question: the world moves its states in a
  state-dependent, nonlinear way from L4 on, but at middle depth those
  moves are along directions the readout is nearly blind to; only late
  moves are "consequential". A denizen would need two notions: motion
  (which exists everywhere) and consequential motion (which is manufactured
  late). Codex round 19 adjudicates; next in the fixed order is the
  forward-time move under a stricter contract.

## 2026-08-28 — Audit #7 adopted; displacement run launched

- The ≤0.02 closure rule was post-hoc: the withdrawal at L8/L12 stands as a
  conservative one-sided policy, not a preregistered equivalence. Wording
  corrected throughout: "no demonstrated positive ridge advantage under this
  margin"; "consistent with identity plus a calibration-mean displacement;
  the structure of Δ is unresolved"; completion "validated to measured
  precision". L4/L20 live remainders; L27 not a persistence test.
- Audit #7 endorses Round 18's order: displacement ladder first, then
  forward-time transport under a stricter contract (sentinel, token/position
  baselines, endpoint definition), then unseen-word/style controls, then a
  second family.
- Δ-mode smoke at L8→L9 (2 shuffles/10 boot): displacement cosine — shared
  shift 0.40, word-conditioned displacement mean 0.58, chart 0.64, ridge
  0.71, kernel 0.76 — but slot ordering moves by only 0.003–0.02 over the
  word-conditioned mean: the slot law is nearly saturated by the identity.
  The predeclared five-pair run (95-minute wall) is executing.

## 2026-08-28 — Re-contextualization #7 (after the identity baseline)

- **Central bet:** native mathematics of latent spaces from what a denizen
  must invent. **Live question now:** in a world whose middle blocks mostly
  leave a state where it is and add a shared shift, what is the law of
  motion — and is the residual motion (Δ beyond the shared shift) the object
  a denizen would care about, or is the real move forward-time?
- **What still holds:** exact slot completion (identity KL ~1e-5); lexical
  persistence at L0; the ridge/chart/word-mean ordering as a descriptive
  fact; L0 and L27 as transforming blocks in their own coordinate families.
- **What is reframed:** the "affine transport law" was the residual stream's
  identity plus a constant. Every earlier NLM-007 reading of "law" at middle
  depth is re-read as "persistence". The frozen-encoder residue (chart
  smoothness, affine-path robustness) and this one now rhyme: in both worlds
  the cheapest map (identity / straight line) explains most of what the
  fancier map explained. The recurring lesson is about our ladders, not the
  worlds: the null must be the cheapest thing the world could be doing.
- **Alternatives held live:** (a) Δ-ladder — is any state-dependent motion
  present beyond the shared shift, per depth; (b) forward-time move — the
  denizen's actual step; (c) the displacement may be word-dependent rather
  than state-dependent (word-conditioned mean displacement decides); (d) the
  whole "layer = time" framing may be the wrong clock for a denizen; (e) a
  second family may show a different persistence profile — if persistence
  depth-profiles differ across families, "where the world moves" becomes the
  first cross-model native quantity.
- **Ecosystem deposit:** "identity is the null for any residual-stream
  measurement" recorded in `_meta/INDEX.md` as a portfolio-wide rule.

## 2026-08-28 — Six-pair moot-maker run: persistence plus a shared displacement is the middle-depth "law"

- `Yhat = X + mean_cal(Y−X)` vs ridge on the corrected slot endpoint, pooled
  ridge − identres (cos / slot skill / slot ordering): L0 +0.46/+0.96/+0.38;
  L4 +0.033/+0.019/+0.022; **L8 −0.008/−0.021/−0.020; L12 −0.007/−0.009/−0.013**;
  L20 +0.018/+0.034/+0.032; L27 +0.20/large/+0.17 (post-norm target — not a
  persistence family). The per-carrier affine diagnostic is far below both
  everywhere (0.63–0.87 / 0.42–0.55).
- Withdrawal condition met at L8→L9 and L12→L13 — the two pairs that carried
  the corrected two-pair criterion. The "full-dimensional affine predictor"
  residue there was the residual-stream identity plus a constant shift. At
  L4 and L20 a state-dependent remainder of 0.02–0.03 survives the identity
  baseline but sits under every gate. L0 and L27 are transforming blocks in
  different coordinate families.
- Round 17's prediction that identity-plus-residual would not close the lead
  at L12/L20/L27 failed at L12. Run overran the 55-minute budget (4541 s):
  budget-incomplete, no gate claim drawn; the withdrawal is null-making and
  stands.
- What the question becomes: the transport content of a middle block is the
  displacement Y − X; the ladder must be rerun on Δ (mean displacement as the
  zero-order law) to ask whether any state-dependent motion exists at all
  beyond persistence. And the move a denizen actually makes is forward-time,
  not layer-to-layer — untested. Codex round 18 adjudicates and re-orders.

## 2026-08-28 — Tier-3 audit #6 adopted

- Code repair confirmed; L8/L12 qualification stands as bounded exploratory
  evidence at the reduced budget; L27 kept in its own post-norm family.
- Corrections adopted verbatim: L4/L20 non-qualifying but live (not killed);
  the all-fold +0.05 rule is a stricter convention than the original lock and
  is labelled so; "wins" = minimal within 0.02 (kernel numerically best on
  some endpoints); no manufactured-context causal language from the shuffle
  profile without spread/style controls; identity test to be extended across
  probes and pairs and stored.
- Its ordered next actions match the Round 16/17 order; the baselines run is
  executing now.

## 2026-08-28 — Moot-maker #1 smoke: identity plus a shared displacement explains L8→L9

- `Yhat = X + mean_cal(Y−X)` at L8→L9: successor 0.949 pooled vs ridge 0.941;
  slot skill 0.958–0.975 vs ridge 0.905–0.980; ridge − identres ≤ +0.013 on
  every endpoint in every fold, negative in three of four. The per-carrier
  affine diagnostic (64 training words per carrier) sits at 0.80 / 0.48.
- Round 16 withdrawal condition met at this pair: the "affine law" wording is
  withdrawn. The move at middle depth is persistence plus a shared
  displacement — the residual stream behaving as a residual stream. This is
  why rank ≤ 128 lost by 0.05 (it cannot express the identity) and why the
  static chart lost (it never sees the held-out state's own coordinates).
- Lesson logged against think-before-you-run: identity was the obvious null
  for a residual stream and should have been in the ladder from Round 13.
- What the question becomes: the transport content is the displacement
  Y − X. Is it state-dependent beyond a constant? The ladder must be rerun on
  Δ with the mean displacement as the zero-order baseline, and the depth
  profile re-read: the word-mean's late collapse (0.40) now says the
  displacement, not the state, is where context enters.
- Full six-pair baselines run and the Δ-ladder await a fresh Codex round that
  has this result in hand (round 17 was launched before it).

## 2026-08-28 — NLM-007 corrected rerun (slot endpoint): 3 pairs clear every locked gate

- Six pairs, slot-position completed law, 2145 s of a 3300 s budget; support
  1.0 everywhere; reload check unchanged.
- Mechanical gate reading (Codex adjudication pending): qualifying pairs
  ['L8->L9', 'L12->L13', 'L27->L28']. L0→L1 lexical persistence (word-mean = field on all three
  endpoints). L4→L5 and L20→L21 clear both slot readouts by wide margins but
  miss the +0.05 successor-cosine lead in some folds.
- The word-mean's slot skill decays monotonically with depth (0.95, 0.84,
  0.78, 0.70, 0.43, 0.40) while the affine field holds 0.92–0.98 and the
  static chart collapses late (0.50, 0.51): the share of the move that
  depends on context rather than word identity grows with depth, and by the
  late blocks a chart is nearly useless while an affine law is nearly exact.
- Prediction scorecard: five of six Round 16 readings held; the sixth
  (final-block attenuation at L27→L28) failed — the final pair clears every
  gate, with the qualification that its successor cosine is on normed
  vectors.
- Bounded as before: one model, shared words, reduced 20/500 budget. Next in
  the fixed order: cheap moot-makers (identity-plus-residual, per-carrier
  affine; code written, smoke running), forward-time transport, unseen-word
  split, second family.

## 2026-08-28 — Re-contextualization #6 (step-back, before the corrected rerun lands)

- **Central bet (README):** a native mathematics of latent spaces, built from
  what a denizen must invent to navigate. **Live question today:** does this
  LM world have a reusable, context-conditioned law of motion at any depth,
  or only lexical persistence plus a chart that smooth regression interpolates
  well?
- **What still holds:** successor-endpoint lead of a full-dimensional affine
  field over word-mean and chart from L4 on (one model, shared words), with
  the carrier-shuffle penalty growing with depth; L0 = lexical persistence.
- **What reframes earlier work:** the frozen-encoder program (NLM-002…006b)
  had no move at all — it was a static world, so "law" had no referent. The
  LM world supplies moves; the question sharpened from "is there a native
  metric" to "is there a native law", which is the question a denizen would
  ask first. The audit-#5 defect (right measurement, wrong position) is the
  same shape as the Igor episode; the cadence caught it before any statement.
- **Alternatives held live (not one thread):** (a) smooth implementation-
  specific conditional regression — the strongest rival, testable by the
  cheap moot-makers now written (identity-plus-residual, per-carrier affine);
  (b) the slot law structurally favours the word-mean (it depends only on
  prefix + word) — if so, the last-token readout is the navigation-relevant
  one and the lock should carry both; (c) the real move is forward-time
  (append-token / next-position), untested; (d) the law may be a property of
  this family's training, not of latent worlds — second family pending;
  (e) the whole layer-pair framing may be the wrong unit — a denizen moves
  across the full stack, and a composed multi-block law (L4→L12) would be the
  first genuinely non-local object.
- **Ecosystem thread:** the "measured law vs interpolated chart" distinction
  and the endpoint-position lesson are deposited in `_meta/INDEX.md`; they
  transfer to any project that reads a probe at a position other than the one
  it perturbs.

## 2026-08-27 — Round 16: corrected slot endpoint and next order

- Tier-3 audit #5 applies to every pair: the fallback and extension completed
  laws were read at the sequence's last token, not the substituted slot named
  by the lock. All such completed-law numbers are void for lock purposes.
  Successor scores remain valid exploratory coordinate forecasts under the
  reduced extension budget; no completed-law number is lock-valid.
- The addendum shows hidden index 28 is post-final-norm. The final pair's valid
  completion is `head(Yhat)` at the substituted slot; identity tests pass at
  `L8->L9` and `L27->L28`. Its successor is a normed-vector prediction and
  needs separate comparison from raw-residual pairs.
- Predeclared the corrected full six-pair rerun: slot endpoint, 20 shuffles,
  500 clustered bootstrap replicates, one CPU process, 55-minute hard budget
  (about 48 minutes projected from 24 minutes per three pairs plus margin).
  Fixed predictions and the slot word-mean interpretation are in
  `theory/EXPERIMENTS.md`.
- Alternative order: cheap identity-plus-residual and per-carrier-affine
  diagnostics; forward-time append-token/next-position transport; disjoint
  class-stratified unseen-word split; second model family. The first baseline
  step is specified against the existing artifact with 64/16 word
  cross-fitting per calibration carrier.
- Under the guiding question, word identity is already a field at L0, while
  early blocks manufacture increasing carrier dependence. A denizen needs an
  identity test, context-conditioned transport, and downstream completion,
  validated across new words, time, and realizations.

## 2026-08-28 — Round 15 extension: successor endpoint across depth

- L4→L5, L12→L13, L20→L21 in 1100 s (budget 1800). Successor endpoint valid;
  completed-law numbers are the last-token secondary readout only.
- **L12→L13** matched its prediction on the successor endpoint: ridge 0.977 vs
  chart 0.898 / word-mean 0.888; ≥0.05 over the chart in all four folds with
  clustered lower bounds above zero; low-rank misses by 0.05; shuffled ridge
  null 0.67–0.78. Qualifying status waits on the slot-position endpoint.
- **L4→L5** falsified the prediction: not lexical-persistence dominated
  (ridge beats the word-mean by 0.03–0.06, LB > 0), yet the chart lead reaches
  0.05 in one fold only. **L20→L21**: ridge within 0.02 of kernel (prediction
  "kernel minimal" wrong); chart lead ≥0.05 in two of four folds.
- Depth pattern (one model, shared words): the word-mean equals the field
  only at L0; from L4 on a full-dimensional affine field beats both the
  word-mean and the chart at every depth; the carrier-shuffle penalty grows
  with depth. Reading under the guiding question: carrier-dependence of the
  move is not present at the input and is built up by the world's early
  blocks — the state's dependence on its context is a manufactured quantity,
  not a given.
- Two of three Round 15 predictions failed on the class/minimality side; the
  successor-ordering prediction held. Recorded as such.

## 2026-08-28 — Tier-3 audit #5 and re-contextualization #5

- Audit #5 (fresh Codex) on the L8→L9 claim: the completed-law endpoint read
  the last token, not the slot the lock names — invalid at every pair, not
  only the degenerate late one. Adopted verbatim; analyzer repaired (slot
  primary, last-token secondary). Status is now: "L8→L9 provides exploratory
  evidence that a full-dimensional ridge field predicts stored successor
  states across held-out carrier templates on shared words; the lock-valid
  completed-law endpoint was not implemented, the fallback lacks the required
  second pair, and the result is bounded to one model."
- What still holds: the successor-endpoint lead at L8→L9 over the chart and
  the word-mean, with clustered lower bounds above zero, and the shuffled
  drop; L0→L1 dominated by word-conditioned lexical persistence.
- What is reframed: "first measured law" was premature by one readout
  position. The same mistake shape as the Igor episode — the endpoint
  measured a different question from the one claimed — caught this time
  before any public statement, by the audit cadence.
- Alternatives now live (audit's list adopted into EXPERIMENTS.md): the
  strongest rival is a smooth implementation-specific conditional regression
  (word code + carrier style denoised by a high-dimensional field) that says
  nothing about a native law; the tests that separate the two are the
  unseen-word split, a second model family, and forward-time (append-token /
  next-position) transport — the denizen's actual move, which no NLM-007
  variant yet measures. Cheaper moot-makers to run first: identity-plus-
  residual and per-carrier affine baselines at equal training budget.
- Next: finish the Round 15 extension (successor endpoints valid), smoke the
  corrected endpoint, then a fresh Codex round to predeclare the corrected
  full re-run and the forward-time move.

## 2026-08-28 — NLM-007 (fallback run): an affine transport law at middle depth

- Ran under the declared fallback (L0→L1, L8→L9, L27→L28; 20 shuffles; 500
  bootstrap); 1427 s, 19% over the cap. Float16 reload check passed
  (KL-ordering agreement 0.9998).
- **L0→L1:** word-mean = ridge = kernel = 0.949; shuffled null 0.95. The first
  block's slot action is carrier-independent: lexical persistence, no law
  beyond word identity. Minimal class on both endpoints: word_mean.
- **L8→L9:** ridge/kernel 0.94 versus best static chart (kNN-5) 0.86 and
  word-mean 0.86 on successor cosine; the prior world-completed skill and
  ordering values are void because they used the last-token endpoint. The
  successor lead and shuffled null 0.75–0.84 remain exploratory evidence of a
  carrier-transferring regression field. Low-rank (rank ≤ 128) trails full
  ridge by 0.05 — affine predictor yes, completed-world law not yet shown.
  The within-carrier comparison is descriptive, not a ceiling argument.
- **L27→L28:** successor lead +0.07–0.12, but the completed-law endpoint is
  degenerate by construction — the law is read at the last token and no
  remaining layer connects the slot to it (KL = 0, skill undefined, support
  0.42–0.56). The lock's "only norm and head remain" missed this.
- Verdict within the lock: one successor-only pair supports exploratory
  regression evidence; two corrected completed-law pairs are needed, and the
  fallback is incomplete for the gated verdict. Under the guiding question,
  middle depth may contain a reusable state-transport regularity, but the
  corrected slot endpoint must show that it cashes out in the world's response
  law. Bounded to one model and shared words.

## 2026-08-27 — Round 12: NLM-006b is non-diagnostic; pivot to dynamics

- Adjudicated NLM-006b against the locked `p_e >= 0.80` identity gate. The
  four displaced families measured 0.458, 0.317, 0.185, and 0.416, so all are
  OOD; calibrated displacement and 400/400 support passed for all four.
- The TT chart lead of 0.09–0.29 is therefore descriptive OOD evidence, not a
  gated chart-survival closure. The previous ledger wording is corrected by
  an append-only entry. The small outside-class chart `ST>TS` effect is about
  0.035 with CIs excluding zero.
- Frozen-encoder work closes as a scope decision. The residue is a trained
  task-effective chart, affine-path smoothness, and graceful relative chart
  degradation under identity-destroying moves; no native construct competes.
  Next program: LM residual-stream dynamics, specified in
  `theory/dialogue/003.md`.

## 2026-08-28 — Tier-3 re-contextualization #4 (Claude, before auditor #4)

**Live question.** In a world with its own dynamics (a causal LM's residual
stream), what is the minimal law class that predicts transport across unseen
contexts — and does prediction cash out in the world's completed response?

**Tunnel check.** The frozen-encoder program ended where every instrument
pointed: the trained chart is the operational map, and no probe-built
construct competes. NLM-007 is a different shape (laws, not closeness) but the
same substrate habit — one small LM, 80 words, 16 carriers. Alternatives held
live: (1) a second LM family (SmolLM2/gemma) in the same design, since a law
that holds in one decoder is a fact about that decoder; (2) transport of
*sequence* states (append a token) rather than layer transport — the move the
denizen actually makes in time; (3) the denotational primitive on diffusion
latents, still untouched; (4) the moot-maker: if a low-rank affine field
explains every layer pair at ceiling, the world's dynamics are locally linear
in its chart and the native program reduces to "find the chart in which
dynamics are linear" — a Koopman question, with a literature.

**What reframes earlier work.** The auditor's narrowed residue (task-effective
chart + affine-path smoothness) and NLM-007's question are the same object
seen twice: smoothness of the chart *along paths* is exactly what a linear
transport law would produce. If NLM-007 finds low-rank affine transport at
early/middle depth, it explains NLM-002's within-class monotone paths.

## 2026-08-28 — NLM-006b: the chart survives every displaced transport

- Uncontaminated run (independent candidates, 400/400 support, calibrated
  displacement gate passed by all four families): on the transported pair the
  chart leads the transport-aware natives by 0.09–0.29 with CIs far from zero;
  natives never compete. Label preservation (readout proxy) is 0.19–0.46 for
  the four families vs 0.77 for the near-identity controls, so most chart
  degradation is identity loss, not transport law.
- One real effect: order sensitivity ST > TS ≈ 0.035 (CIs exclude 0) only
  outside the invariance class — substituting then transporting the candidate
  beats transporting the anchor first. Small, but it is the first measured
  non-commutation in this world.
- Per the lock's chart-survival branch, the frozen-encoder transport line
  closes: the trained chart is the operational map for this measured envelope.
  Round 12 decides what replaces the frozen-encoder program.

## 2026-08-27 — Tier-3 re-contextualization #3 (Claude, before auditor #3)
- Infra: the flaky link could not upload the 17 MB transport embeddings (HTTP 408); unpushed history was rewritten to drop them, they are git-ignored, and provenance is the sha256 in the lock. Remote verified in sync via ls-remote (never trust 'Everything up-to-date').


**Where the program stands.** Five measurements in one vision world plus one
in the LM world converge: a trained encoder hands its denizen a chart metric
and straight routes that already serve as the one-step map; nothing we built
from probes competes; an untrained chart has neither. Round 10 closed the
frozen-encoder closeness line and opened NLM-006 (moves outside the trained
invariance class).

**Honest tunnel check.** Since the pivot, every measurement has been
"rank candidates by closeness to an anchor, score against a label". That is
one instrument shape in two worlds. Alternatives I hold live:
1. NLM-006 as designed — the last test inside this shape; if the chart
   survives crops/inversion/mixing/occlusion, closeness-on-frozen-encoders is
   done for good.
2. Worlds with dynamics: LM residual streams, where transport *is* the forward
   pass and the map must predict where the world takes a state — the only
   setting where "move" is not something we impose.
3. The denotational primitive on diffusion latents (evidence-update, not
   closeness) — the legacy program's own repair results are its instrument.
4. The moot-maker: if "inherited from training" fully explains every result,
   the honest program is the study of what training objectives install in a
   chart — encoder-invariance science, not native mathematics. The auditor is
   asked to say whether we have already become that.

**What reframes earlier work.** The legacy separatrix "islands" and NLM-004's
95% null-world flicker are the same phenomenon: chart-straight lines are
routes only where training laid them. The guiding question's "map" is, so
far, not invented by the denizen — it is issued to it.

## 2026-08-27 — Round 10: frozen-encoder closeness/map line closed; NLM-006 opens

- Codex round 10 (`fbe7bee`, blackboard-backed): NLM-005 void (support 32%,
  chart-metric ST−TS ≤ 0.006; R_no_coarse gap 0.027 on shift1px noted);
  NLM-003's R win withdrawn as a coarse-taxonomy leak (leak-free R 0.586 < F
  0.667). The frozen-encoder closeness/map competition is **closed as a
  program**. Residue: training supplies an operational chart and routes — a
  denizen inherits navigation equipment — not a proven intrinsic geometry.
- NLM-006 opens: transports outside the trained invariance class (large crop,
  color inversion, image mixing, occlusion), each verified non-near-identity by
  embedding displacement; stratified candidates (20 same-class + 20 hard
  negatives, frozen); support ≥80%; decisive if ≥2 of 4 families break the
  chart lead or expose a transport-aware native predictor. Building the
  artifact now.
- Infrastructure: blackboard MCP works for Codex via the installed binary
  (npx cold start exceeded its startup timeout); TOML paths need forward
  slashes.

## 2026-08-27 — NLM-005 composition: void on support, non-diagnostic, and a design lesson

- Composed moves with hflip / 1-px-shift transports re-encoded by the frozen
  encoder: ST−TS gaps ≤ 0.006 for every predictor (kill 2), support 129/400
  (kill 3). Cosine leads both native constructs by 0.32 on every order.
- Lesson: these transports are exactly the augmentations DINOv2 was trained
  to be invariant to — near-identity moves in its world — so composition with
  them cannot reveal a law. Transports must lie outside the trained invariance
  class (large crops, color inversion, image mixing, occlusion, or a different
  encoder's edits). And support needs stratified candidates (same-class
  candidates by construction), not 40 random draws over 100 classes.
- Standing picture after five measurements: in trained worlds the chart
  metric is the map for one-step consequences and survives trained-invariant
  transports; it collapses in an untrained chart (NLM-004). No native
  construct built so far competes with it. The program's next honest question
  is whether *any* move outside the invariance class breaks the chart.

## 2026-08-27 — NLM-003 diagnostics: R's win was the coarse head

- Rerun with audit-#2 diagnostics (same lock, new anchor sample): R without
  the coarse head = 0.586, below F (0.667). R's advantage was taxonomy leak —
  fine labels nest inside coarse classes — exactly as the fresh auditor
  predicted. Δ_F−R = −0.095 [−0.142, −0.049] marginally fails the strict gate
  on this resample. R ties on 22–33% of comparisons.
- Cheap ladder: PCA-32 cosine 0.941 ≈ cosine 0.934; pixel baselines 0.62.
  k-sensitivity: same-class fine-kNN flicker 0.10–0.18, cross-class 0.37–0.41
  for k = 8/32/128 — the world-path contrast is robust.
- Net for the primitives: neither F nor R (leak-free) is a competitive map;
  the trained chart's metric is. The program's live positive result is
  NLM-004's: that metric and its straight routes are products of training.

## 2026-08-27 — NLM-004 null world: the chart's map and its world-paths are inherited from training

- Preregistered in the ledger (before scoring) and supported: in a random-init
  DINOv2 chart the cosine map for fine-label consequences collapses from 0.946
  to 0.575 (gap 0.37 ≥ 0.20); embedding-kNN fine accuracy 0.761 → 0.069; F and
  R collapse too (0.58 / 0.57); raw-pixel and pixel-statistic baselines (0.62)
  now beat cosine. Pixel-statistic heads stay strong (rgb 0.83, luma 0.82):
  the null chart preserves pixel structure, not semantics.
- Sharper: null-world M1 — same-class fine-kNN flicker along chart-straight
  lines is 95% (trained: 12.7%), cross-class 99% (trained: 38%). So "straight
  lines are world-paths within a class" is itself a product of training. In a
  trained world the denizen inherits both a metric and a set of straight
  routes; in an untrained chart neither exists.
- Ties: R's profile statistic ties on 33–36% of comparisons (5-valued), as the
  audit warned; trained-world tie fractions, R-without-coarse, k-sensitivity
  and the cheap-baseline ladder are being rerun as nlm003_v2_diagnostics.
- Duplicate-run lesson: three NLM-004 launches overlapped and starved each
  other; the process check with a quoted tasklist filter was wrong. Runs are
  now single, detached, file-logged, with a completion watcher.

## 2026-08-27 — Tier-3 re-contextualization #2 (Claude, before the auditor)

**Live question now:** in every world tested so far (LM input rows, LM residual
states, DINOv2), the model's own chart metric is the best one-step map for
consequences. Is that a law of trained latent worlds, or an artifact of only
testing one-step moves in encoders trained to make one chart metric meaningful?

**Tunnel check.** Two days of work have stayed on "closeness / one-step map of
a frozen embedding" — three instruments, one shape. Same-shape follow-ups are
now forbidden by our own rule. Live alternatives, each with a decisive result:
1. Null world (random-init encoder, building now): if cosine still predicts
   fine-label consequence there, cosine tracks pixel similarity, not training;
   if it collapses, the chart metric's dominance is a product of training and
   the denizen's map is *inherited*, not native.
2. Two-step moves (composition): substitution∘transport vs transport∘substitution
   — a one-step metric cannot express non-commutation; if it exists, that is
   the first law no chart metric captures.
3. Cross-class world-paths (M1: 38% detours): the geometry of *routes*, not
   distances — a routing map the metric does not give.
4. Worlds with dynamics: LM residual states across layers (transport is the
   physics); the denizen's map must predict where transport takes a state.
5. The moot-maker: if a fixed contrastive-training explanation ("cosine is
   meaningful because the loss made it so") accounts for every result, the
   native program on trained encoders reduces to studying training objectives.

**What reframes earlier work.** NLM-001's loss to contextual cosine and
NLM-003's loss to chart cosine are the same finding: trained charts carry
their own metric. The program's object should shift from *closeness* to
*moves and laws* — where the chart has nothing to say.

## 2026-08-27 — NLM-003: R beats F, both dominated by the chart metric; blackboard live

- NLM-003 (locked `e2a1fb2`, true fine-label endpoint, same artifact): R
  (substitution-profile agreement) beats F (Fisher pullback) — Δ_F−R = −0.104
  [−0.148, −0.058], gate met (R 0.734 vs F 0.630). Decisive context: plain cosine
  in the DINOv2 chart scores 0.946, Euclidean 0.935 — both native constructs
  lose by 20–30 points to the imported metric on the informative endpoint.
  Support thin: 130/400 anchors had a same-class candidate among 40 draws.
- Reading before Codex round 8: DINOv2 is trained so that one chart metric is
  meaningful; in such a world the denizen's best one-step map *is* that metric.
  NLM-001 found the same in the LM world (contextual cosine at L14 = 1.000). So
  far every world tested has a chart metric that already is the map for
  one-step consequences. Where native structure could still differ from the
  chart: (i) two-step moves — does substitution-then-transport equal
  transport-then-substitution (composition / laws), which a one-step metric
  cannot express; (ii) worlds whose chart was not trained to be metric (raw
  residual states of a non-contrastive model, or a randomly-initialized
  encoder as a null world); (iii) cross-class world-paths (M1: 38% detours).
- Blackboard: `@iqidis/blackboard-mcp` installed globally, registered for
  Claude Code (user scope) and Codex; mandated in global CLAUDE.md and the
  setup skill; Codex verified `bb_list`/`bb_create`/`bb_add_entries` and seeded
  the project board (`.blackboard/5df235ea`, git-ignored). This session cannot
  call `bb_*` until restart; Codex rounds now use it.

## 2026-08-27 — NLM-002 non-LM branch run: endpoint killed, chart-path structure found

- Artifact frozen (CIFAR-100 → DINOv2-small, 6000/2000, sha256 8de4f0b0…);
  locked with two recorded implementation decisions; run in 133 s on CPU.
- M2: the raw-pixel k=32 kNN fine-label endpoint is nearly uninformative
  (0.115 accuracy; 0.12 agreement with embedding kNN) → preregistered endpoint
  kill condition met. M3 (F vs R) is therefore a tie on noise (Δ = −0.004
  [−0.034, +0.026]); no primitive verdict. Lesson: independence is necessary,
  informativeness is not optional — the true fine label (no head trained on it)
  is both and should have been the endpoint.
- M1 (chart-path closure), the informative result: along chart-straight lines
  between same-class embeddings the coarse-semantic readout is monotone in 98%
  of paths (flicker 2% [0.3, 3.7]); between classes, fine-label kNN flickers on
  38% [32, 44] of lines and any-readout on 78% [73, 83] — straight lines between
  classes pass through third classes. Within-class chart lines are near
  world-paths for semantics; cross-class lines are not. Pixel-statistic heads
  are weak (52–59% test accuracy) and their flicker is partly head noise.
- Process: Codex sessions are now always fresh with terse file-pointing
  prompts (Devansh); `_meta/INDEX.md` row for this project updated for sister
  agents; no blackboard MCP is configured on this machine.

## 2026-08-27 — NLM-001 closed negative; NLM-002 designed as a primitive competition

- NLM-001 verdict (Codex round 3, `1584514`): instrument-void for confirmation
  (runtime metadata reconstructed post hoc), bounded negative falsifier of the
  lexical-KL instrument. Native calibration-KL lost to a symmetric metric on
  contextual hidden states (Qwen Δ = −0.058 [−0.22, +0.03]; unlearned centered
  cosine at layer 14 reached 1.000 on held-out orderings vs native 0.954);
  context reversals exceeded the paraphrase null in 2/3 systems (Qwen Q = 2.12
  [1.70, 2.56]); directedness absent. T2/T3 demoted to bookkeeping; T1's
  conjunction-closure premise fixed. Fresh Tier-3 audit adopted.
- NLM-002 (`theory/dialogue/002.md`, skeleton, not locked): mutual-kill
  competition between F (one fixed Fisher response-law geometry pulled back
  through frozen decoders/heads) and R (probe-indexed substitutability tested
  outside LMs, on DINOv2 image embeddings), each with an independent behavioral
  endpoint and a common-support Q estimator.
- Artifact prep for arm R: no image data existed locally. Building
  `experiments/results/vision_cifar100_dinov2s/` — CIFAR-100 (fine + coarse
  labels) → DINOv2-small CLS embeddings on CPU (35 ms/image), plus label-free
  pixel statistics (mean RGB, luminance, edge density) so probe blocks can ask
  questions not derived from the class taxonomy. Manifest carries dataset and
  encoder revisions, split indices, seed, sha256.
- F-arm implementation note for the lock: the LM Fisher pullback can be
  estimated as G = mean over sampled tokens t~K_c(z) of the outer product of
  ∇_z log K_c(z)(t) (VJPs through the frozen decoder; ~64 samples per state and
  calibration probe, D = 1024, CPU-feasible in ~30 min); the DINOv2 arm's Fisher
  is exact for a linear head (G = mean_z W^T F(p) W).

## 2026-08-27 — Tier-3 re-contextualization (Claude, before the auditor's answer)

**Live question.** Is closeness in a latent space context-indexed and directed
in a way no symmetric contextual representation reproduces — and can a native
invariant (context rank, Q = B/W) say how many orderings are actually required?

**Tunnel check — honest answer: partially tunneled.** In one day the program
narrowed from "native mathematics of latent spaces" to "next-token-KL
substitution probes on 80 lexical tokens of three tiny causal LMs". That is a
fine first instrument; it is not the object. Risks: (i) everything measured is a
property of one decoder family; (ii) 'latent space' has so far meant 'input
embedding row', the least latent thing in the model; (iii) the primitive
(substitutability under probes) was chosen in round 1 and never competed.

**Alternatives now live, each with its decisive result:**
1. Non-LM latent spaces with downstream-head probes (DINOv2, ESM-2, CLIP,
   wav2vec are cached). If the same axioms/invariants organize a vision or
   protein space, this is about latent spaces; if not, it is about LMs.
2. The probabilistic/denotational axiomatization (dialogue 001 §1B) on
   diffusion latents — the legacy program left instrumented diffusion stacks
   that already denote laws. Decisive: evidence-update is definable there and
   predicts something the relational axioms cannot.
3. Probe family as the object: states × probes as a formal context (Galois
   connection / formal concept analysis) — existing mathematics for exactly
   the (X, C, N) structure. Decisive: the concept lattice of a real system has
   nontrivial structure that context rank flattens.
4. Intermediate residual states as latent states (not embedding rows), probes
   = continuation from that depth. Decisive: context rank changes with depth.
5. The moot-maker: contextual cosine at the right layer matches native transfer
   (H3 kill 3/6). Then "native" = "the model's contextual representation" and
   the program must change primitive or object.

**What reframes earlier work.** The legacy perturbation program was, in these
terms, a substitution probe with the whole prompt as state and free generation
as the readout; its withdrawn results were the readout's insensitivity, not the
state's. Its one surviving residue — measure the stack's numerical noise floor
before interpreting any latent-space difference — is now a standing gate here
(η) and belongs in every project that measures representations.

**Direction still makes sense?** Yes as a first falsifiable instrument, with the
explicit condition that NLM-001's verdict (either way) must be followed by at
least one of alternatives 1–4, not by NLM-002 on more words.

## 2026-08-27 — Round 2b instrument calibration amendment

- Read the disclosed eight-item full-pipeline calibration. Within-block KL
  scale varied up to 16-fold, making the preregistered instance-specific MAD
  threshold invalid and yielding zero passes.
- Locked per-paraphrase scale normalization, four-of-four sign agreement, a
  block-pooled magnitude scale, and the explicit 12.5% random-sign null.
- Demoted directed asymmetry to exploratory after 0–18% sign agreement at
  chance. Revised post-calibration predictions to \(Q=2.5\), \(R=0.20\), and
  \(\Delta_{\rm rev}=+0.07\).
- The eight inspected words are excluded from primary confirmation; the
  remaining 72 are primary and all 80 are sensitivity. No confirmatory run was
  performed.

## 2026-08-27 — Round 1 revised after Claude attack

- Appended `## Codex — revision` to `theory/dialogue/001.md`, answering A1–A7
  point by point. Withdrew existential non-collapse as an axiom; adopted finite
  context rank with anchor-dependent radii.
- Proved the finite representation theorem: context rank is 1 exactly when
  every anchor's context neighborhoods form an inclusion chain. Derived the
  incompatibility-graph coloring characterization.
- Created `theory/AXIOMS.md` as the living formal surface. Local refinement is
  now conditional on a completed probe family; cross-realization agreement is a
  measurement of transportability, not an identity axiom.
- Created `theory/EXPERIMENTS.md` with the post-smoke, pre-measurement NLM-001
  preregistration: primary directed asymmetry, graded context rank, held-out
  transfer against contextual-cosine and learned-metric baselines, paraphrase
  nulls, cross-system checks, exact predictions, and kill conditions.
- No experiment was run in this revision; the confirmatory measurement remains
  unrun. A concurrent Claude turn added the CPU measurement runner and raw
  hidden-state capture. Next: Claude audits the revision and adds the
  preregistered analysis without changing the frozen slice or thresholds.

## 2026-08-27 — Repository restarted

- Entire prior program moved unmodified to `legacy/` (its README, docs,
  experiments, results, and correction record intact and internally linked).
  Root now holds only the new program: README, STATE, NOTEBOOK, `theory/`,
  fresh `experiments/` ledger. Local-only ignore patterns mirrored for `legacy/`.
- Four watchdogs live (20-min liveness, hourly ops, 2-hour Codex audit +
  anti-tunnel, 2-hour entropy sweep). Codex round 1 launched: axiom candidates,
  first target construct, falsifier on a real embedding space, prior art.

## 2026-08-27 — Native latent mathematics, dialogue round 1

- Wrote `theory/dialogue/001.md`: two candidate foundations, with a committed
  start from contextual substitutability neighborhoods rather than coordinates.
- Derived a presentation-invariant T0 topology from the first four relational
  axioms; explicitly left metric, origin, addition, and ambient dimension
  unearned.
- Pre-registered a CPU-only Qwen3-0.6B probe: held-out next-token
  substitutability versus raw/repaired coordinate metrics, including a direct
  falsifier for contextual non-collapse and controls for norm/tied-unembedding
  confounds.
- Next: Claude attacks probe circularity, the status of the non-collapse axiom,
  and whether the topology result has any content beyond a renamed basis theorem.

## 2026-08-27 — Direction set: native mathematics of latent spaces

- Prior LLM-perturbation program closed; arithmetic claims withdrawn after Igor
  Rivin's PRs #4/#5 (merged) and reanalysis of stored data. Correction record in
  `docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md`; README rewritten to a findings
  table; three doc indexes consolidated to `docs/NAVIGATION.md`; orphaned docs
  archived.
- Residue: decoding determinism is hardware-dependent (GH200 non-deterministic,
  RTX 5090 deterministic); perturbation is a causal diversity source only on
  deterministic stacks. Process gates added (termination, direct control first,
  null model, clustered stats, propagation).
- New direction opened (see STATE.md). Starting the Codex dialogue on axiom
  candidates and the first target construct.
