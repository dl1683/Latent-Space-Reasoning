# STATE

## Current direction: Selective Write-Boundary Construction (2026-09-04)

**Pivot (Codex direction dialogue, 2026-09-04):** Close the pretrained-Qwen
path-dependence line. Skip CEG-1R — its outcomes produce no different program
decisions (both paths lead to construction). Measurement-to-artifact ratio was
infinite (506:0); governance required halt above 5:1.

**Central question:** If a model is given an internal "replace this fact"
operation (an endogenous keyed write with hard-masked overwrite), does
compositional reasoning become possible because dead history is truly dead?

**Anti-tautology constraint (Codex):** The write boundary alone is NOT the
artifact. Boundary satisfaction is by construction. The claim-bearing results
are: (1) learned use of writes for compositional reasoning, (2) predictive
compression — |Q_H| bounded by store cardinality not history length, (3)
transfer across held-out presentations, (4) causal substitution of store-
equivalent states, (5) advantage over matched append-only ablation.

**Theory:** theory/CONTINUATION_REFINEMENT.md, CR-10 through CR-14. Laws L1-L4
(overwrite absorption, cross-register commutativity, preservation, write
fidelity). The last-write normal form bounds the write orbit from one initial
place by ∏_r(|V_r|+1). With faithful native reads, an actual complete
initialization gives exactly ∏_r|V_r| places. These statements do not
bound the entire predictive carrier when preserved context varies. Five
pre-declared falsifiers.

**CR-14 revised (2026-09-04):** Original hard-register design was EAC
tautological (Codex correction). Replaced with commit-seal carrier: hard seal
(information bottleneck) + learned update U(m, address, proposal) → m+. The
model CAN fail — the learned update is the falsifiable object, not the
hard-wired register semantics. WB-1 validates the normal form math but is
architecturally tautological. Positive-control staircase: 1 address, 2 values,
2 commits first.

**Narrative gate:** "Give a model an internal act that genuinely means 'replace
this fact,' then test whether compositional reasoning becomes possible because
dead history is truly dead."

**Status relative to the central claim:** CR-15 supplies conditional
behavioral structure and sufficient certificates. It establishes no learned
realization, transfer, causal-substitution result in a learned model, or
advantage over an append-only alternative. Those remain the claim-bearing
obligations.

**Mode since 2026-09-05: analytical (compute-constrained).** Work is
derivation and proof; runs only for tiny models when a derivation needs a
concrete check. Rung 1 of the commit-seal staircase ran 2/5 seeds (both arms
passed every mechanism gate; stale partial deleted, weights not saved) and is
paused, not closed.

**CR-15 adopted (2026-09-05, theory/CONTINUATION_REFINEMENT.md):**
- Under A1-A3, each T-saturated subset is canonically B_T x S_T as a set
  equipped with its T-write action. The general write-orbit bound from one
  initial place is ∏_r(|V_r|+1). An actual complete initialization gives
  exactly ∏_r|V_r| places. Numerical initialization alone does not imply
  saturation, and preserved context B_T may be infinite.
- Transition-closed realization sets with η-accurate registered response
  laws give a history-independent 2η bound whenever the compared histories
  have the same abstract endpoint. Preserved abstract distinctions require
  margins exceeding 2η. Small local absorption errors alone do not provide
  a vanishing history-uniform error bound (cyclic-counter counterexample in
  CR-15.2). A greatest tolerance-compatible relation need not be an
  equivalence relation, and a coarsest tolerance-bounded congruence need
  not exist (three-state counterexample in CR-15.2).
- Contraction supplies invariant sets (surgeon certificate): per-value
  contraction with verified invariant domain and initialization coverage
  gives radius kappa D/(1-kappa) balls; selective-tube version for multiple
  registers requires exact preservation of unaddressed coordinates and
  contraction uniform over them. Fibered contraction with consistent fixed
  points does NOT imply commutation (explicit counterexample).
- Selective memory requires selective forgetting: a global behavioral
  contraction under every write is incompatible with indefinite preservation
  of an unaddressed register.
- A constant, address-independent sigmoid gate CAN realize exact selective
  overwrite (address selection in the candidate); keyed gating is not a
  theorem-level requirement.
- Withdrawn: "positive attention mass on dead tokens forces a nonzero
  absorption defect". Surviving restricted obstruction: with one bounded-
  logit softmax route the LIVE value's influence dilutes like e^B/(m+e^B)
  (fidelity, not absorption, fails with history).

**Status relative to the central claim:** CR-15 supplies conditional
behavioral structure and sufficient certificates. It establishes no learned
realization, transfer, causal-substitution result in a learned model, or
advantage over an append-only alternative. Those remain the claim-bearing
obligations.

**Accounting:** This is an analytical cycle with no runs. Zero runs does not
determine the apparatus-to-artifact code-line or round ratio; those
classifications must be reported separately.

**Audit #39 (2026-09-05): REVISE.** Mathematical core of CR-15 retained
(15.0, 15.2, 15.3, 15.5, 15.7 ADOPT; 15.1, 15.4, 15.6 REVISE, corrections
applied verbatim). Withdrawn from every surface: "distance from claim 0"
and "pretrained transformers lack an endogenous overwrite boundary".
Licensed sentence: under a fixed registered deterministic interface, exact
selective-write laws yield a context/store factorization on write-saturated
states, and a transition-closed, η-accurate realization of an abstract write
system bounds equal-endpoint histories by 2η under every registered
continuation; CR-15 supplies sufficient contraction certificates and an
exact constructed gated-cell example, without establishing learned
realization, transfer, or architectural advantage.

**CR-15.8 — audit #40: mathematical core ADOPT; parameter and attribution clarifications applied.** Fix one initial behavioral place and the registered deterministic interface. If its entire write orbit contains at most N exact behavioral places and absorption and commutation defects are uniformly at most ε in the full behavioral metric, equal-normal-form histories differ by at most B_N = min{1, (N−1)ε} under every common registered continuation. No additional factor two is required. With native own-write calibration error β, a sufficient live-value separation bound is Δ − 2(B_N + β). The worst-case error has order min{1, Nε}; the binary example preserves its live value exactly while accumulating nuisance-response error. The exact envelope is attained when singleton-valued registers are permitted, with zero cases handled separately. A full-behavioral packing bound gives min{1, Nτ + (N−1)ε}, and Nτ can be attained for arbitrarily small positive defects. Finite state, packing, contraction certificates, invariant relations, and ultrametric behavioral distance provide sufficient routes to uniform bounds; no exhaustive or disjoint classification is claimed. These results establish no learned realization, transfer, or architectural advantage. Corrected en route: \"identity cost c_τ = 0\" does not characterize overwrite, and append-only storage does not universally violate uniform reader accuracy.

**Next obligation

**Certificate check (2026-09-05; evidence gate REVISE):** the registered
single-cell attempt remains CERTIFICATE_INCONCLUSIVE. Separately, the sound
post-hoc confirmation procedure reports strict expansion for both writes on
D=[-1,1]^16 in the infinity norm. This is a scoped contraction-premise
failure, not a behavioral overwrite failure. Verifier defects (fixed after the
run, not rerun) prevented accepting the implementation as a general PASS
verifier. The eight-cell training-panel accuracy is 1.000, with maximum
immediate-read paired TV approximately 1.4e-8. Jacobian norms 0.987 and
0.539 are numerical diagnostics at 64-step candidates, not certified
contracting regions. No history-uniform behavioral bound is established.
Details: experiments/EXPERIMENTS.md.

**CR-15.9 adopted (2026-09-05, pending audit; register-free CR-15.8):** for
any deterministic action alphabet on a behavioral metric space, an orbit of at
most N exact places, nonexpansive actions, and a finite family of word
equations each holding to within ε on the orbit, two histories equal in the
least congruence generated by the equations are within min{1,(N−1)ε} under
every registered continuation. No registers, stores, or normal forms are
assumed. Finite orbit is sufficient, not necessary (an infinite-orbit
contracting world has arbitrarily small lifetime error).

**Re-contextualization ruling (Codex, 2026-09-05):** the program continues;
the selective-reset/state-budget certification line is NOT currently the
highest-leverage route to a real-model artifact ("not working yet" as a
route; the mathematics is retained as transferable constraint). The
"von Neumann trap" (imported addressed-register semantics certified as
discovery) is substantially real; native vocabulary = actions, response
distinctions and the continuations that reveal them, distinctions an action
erases/preserves/exposes, contextual equations with explicit error and
observation scope. Next cycle (direction dialogue, round 2 of 3): one
explicit rule for when an erased distinction stays irrelevant under
subsequent actions, anchored to the real-model CR-9 examples, with a
counterexample and an intervention consequence. Discrete-carrier testbed
deferred; no run.

**Locked direction (direction dialogue rounds 1-3 complete, 2026-09-05):
CR-15.10.6, ANALYTICAL_ONLY.** Interface-relative interchangeability with the
modulus ω_a^W(t) as organizing quantity. The next cycle is one bounded
analytical construction: (1) moduli for the contrast worlds (countdown
ω = 1 at every finite depth; doubling ω = min{1, 2t}; CR-15.5 selective
write θt on the addressed fiber; CR-15.9 decaying nuisance ½min{t, δ});
(2) the finite-panel bounding proposition; (3) the finite-observation
obstruction for unrestricted deterministic worlds; (4) one explicit
sufficient premise that lets finitely many questions yield an UPPER bound
on ω (the substantive gap: CR-15.2 bounds pairs in the same realization set,
not every W-close pair; CR-15.3 needs an observability premise in the other
direction, small d_W ⇒ small carrier separation). Real-model panel DEFERRED:
the historical CRC-0 panel has zero eligible W_H-close same-variable pairs
at τ = 0.01, P2's presentation is malformed (`func` vs `f`), and no model in
play is "tiny" under the directive; any acquisition needs explicit approval
and the locked criteria in CR-15.10.6 (τ = 0.01, numerical guard m, five
robust-close pairs over five task blocks, affine and length-only rivals
tested separately by interval-aware infeasibility, verdicts WITNESSED /
TOLERANCE_PRESERVED_ON_PANEL / INCONCLUSIVE).

**CR-15.10.7 (audit #41: REVISE, corrections applied):** finite-panel
enclosures, corrected declared-domain moduli, and the refinement identity
are retained. Response-only finite transcripts cannot establish a
nontrivial global bound over unrestricted extensions; independently
justified completeness or structural premises can. The contraction route
requires Lipschitz control of full question profiles and a uniform inverse
bound on the declared carrier domain. Saturation can degrade conditioning
without making inversion impossible. Local rank does not establish a global
inverse bound or PSR dimension. Neither the saved cell's retained-history
mechanism nor W_H's rank has been established. No new learned realization,
generalization result, or native mathematical novelty follows. Withdrawn:
the original obstruction theorem, the saturation impossibility, the global
rank bound and both dimension extrapolations, the cyclic parity claim.

**Question rotation (Devansh, 2026-09-06: never idle; rotate to a different
analytical question when a line closes).** Block 1 = CR-16, architectural
restrictions of the transformer world. Audit #42: RESTATEMENT; CR-16 requires corrections. Fixed-token causal site invariance and, with consecutive positions, the appended-token offset identity are retained. The exact two-route decomposition, content-independent prefix re-weighting, filler-effect subtraction and identification with CEG-1 RELAY-CUT are withdrawn. CR-15.10.6 already requires matched positions, valid tokenization boundaries and the registered H-versus-F rival. Close Q1 and rotate to one bounded Q4 reading of existing audited response tables under their original interface and claim limits, with no runs or new acquisition.

**Block 2 (Q4) reading done (2026-09-06, evidence gate pending):** on the
archived svb_qwen3_congruence panel (Qwen3-1.7B-Base, 24 cells, 11-bin
reduced channel, τ = 0.01), the neutral comment preserved the tolerance on
all 32 W_0-close assert-type presentation pairs (24 + 8, over all three
variables and eight values; max DN 0.0064 and 0.0100), while the heavy
comment split 29 of 32 (full-law witnesses, DH up to 0.116); the one close
misleading pair split under both. Reduced channel: N's preservation is not
a full-law statement; H's witnesses are. Candidate program decision: use
assert-type presentation pairs as the deferred panel's root family (CRC-0
supplied zero eligible pairs; this family supplies ≥ 24), with N as the
plausible TOLERANCE_PRESERVED action and H as the witnessing action; the
full-law rerun would decide whether N's preservation survives resolution.
Details: experiments/EXPERIMENTS.md.

**Program ruling (audit #41):** the analytical cycle CR-15.10.6 STOPS after
this consolidation pass; wait for approval of an empirical step capable of
changing a program decision. Further solo theory expansion is not the
highest-leverage work; "this is not working as a route to a real-model
artifact." The valid mathematical residue is retained; the wider program
remains open.

**Withdrawn en route:** the "P vs R" mechanism dichotomy (not mutually
exclusive); the CV-of-ratio rival rule (b/d = g + c/d makes spread
uninformative); "immediate-read family closed under N".

**Standing check (Codex):** this is established behavioral mathematics, not
discovered native vocabulary; value must come from effective predictions
about a real model. A proven observation hole identifies an inadequate
interface, not an obstacle to structured reasoning in the model.

**So what:** find which differences between an AI's histories it can safely
ignore, and which later question makes those differences matter again.

**Closed lanes this cycle:** single-cell contraction certificate (evidence
gate: no further run; apparatus ratio 5.71:1); commit-seal rung 1 remains
paused (2/5 seeds, no saved weights); discrete-carrier testbed deferred.

**So what:** if a model has a proved limit on its distinct behavioral
situations, small update mistakes cannot accumulate beyond that limit times
the local error, even after a million updates.


---

## Qwen path-dependence line: CLOSED (2026-09-04)

CEG-1R remains a deferred diagnostic. The tested interfaces did not
establish a useful overwrite law. CEG-1 supports bounded attention-mediated
context dependence after a failed competence preflight; it does not prove
that pretrained transformers lack endogenous behavioral overwrite. The
construction program investigates whether an explicit write boundary
provides a useful alternative.

---

## CEG-1 result: REVISE / formal NO_INTERFACE — bounded context dependence (2026-09-04)

**CEG-1 (Codex strategic design):** Custom 4D attention masks on Qwen3-1.7B-Base
(eager attention, full forward passes). 16 fixed-slot histories × 4 identifier
sets, 6 intervention arms. 914 forwards, 259s CPU.

**Verdict (Codex Evidence Gate, audit #38): REVISE.** The original RELAY headline
does not pass. Competence preflight failed (83.6% < 95% threshold); the binding
design required NO_INTERFACE. RELAY-CUT Δ=0.000 is tautological (all downstream
positions see only pair-identical tokens once the cut disconnects the graph).

**Licensed sentence (Codex, verbatim):** In this fixed-slot Qwen3-1.7B-Base
panel, changing overwritten assignments altered normalized binary response
profiles, and after direct query-to-early-span attention was cut an indirect,
identifier-consistent excess over matched dummy cuts remained (mean max-channel
TV excess 0.101); however, the residual was channel-nonspecific and observed
after a failed competence preflight, so CEG-1 identifies bounded
attention-mediated context dependence, not semantic relay of dead information
in competent computation.

**Key numbers (post-audit):**
- OPEN Δ=0.406 (accuracy 83.6%; r0=71.9%, r1=95.3%)
- QUERY-CUT Δ=0.194 (accuracy 64.1%; channel specificity disappears)
- RELAY-CUT Δ=0.000 (accuracy 100%; tautological by mask design)
- LIVE-CUT Δ=0.934 (positive control)
- SYNCHRONIZER Δ=0.267 (accuracy 89.1%; one string rejected, not synchronization)
- Dummy OPEN Δ=0.105 (accuracy 96.1%)
- Paired excess (QUERY-CUT minus DUMMY-CUT): 0.101, bootstrap CI [0.050, 0.152]
- Target vs orthogonal excess under QUERY-CUT: 0.062 vs 0.061 (nonspecific)

**Audit corrections adopted verbatim:**
1. RELAY-CUT=0 is a tautology of the intervention, not an empirical finding.
2. The "52/48 pathway split" used arm-specific maxima across channels and cannot
   support an additive decomposition. Dummy-adjusted ratio is 33/67.
3. Under QUERY-CUT, register-specific dead-value effect disappears: target and
   orthogonal channels show equal excess over dummy. The residual is broadcast
   context sensitivity, not semantic relay.
4. Four "renderings" are one syntax with four identifier pairs — name robustness,
   not held-out template generalization.
5. SYNCHRONIZER rejects one string, not synchronization or irreversibility.
6. Design omitted target/orthogonal effects, empirical CDFs, clustered bootstrap
   intervals, full-vocabulary accuracy, replay noise, and model revision binding.

**Never say:** "Dead information persists through attention relay into live-write
representations; roughly 48% is relayed, and the failed synchronizer shows that
this relay is irreversible."

**Measurement-to-artifact ratio:** 506 apparatus lines to 0 construction lines
(infinite). One-more-test cap is binding.

**Line closed.** CEG-1R skipped (Codex direction dialogue: decision value nil).
Qwen path-dependence line closed. Transferable finding: the tested interface did not
establish behavioral overwrite; the broader architectural question remains
unresolved. Construction program opened (CR-10+).

---

## CRC-1 result: INVALID / bounded negative (2026-09-04)

**CRC-1 (Codex Architecture Theorist design):** Two binary registers, 32 roots
(8 per store), 4 literal actions, task-aligned {0,1} observer. 1508 forwards,
357s CPU. **INVALID** (competence 87.5% < 95% threshold). Identity partition
(Q_1=32), zero compression at all tolerances (min D_1=0.099), algebraic
relations 25.9% (idempotence 37.9%, absorption 20.7%, commutation 19.1%).

**Licensed sentence (Codex-corrected):** In the tested Qwen3-1.7B-Base
interface, semantically cofinal assignment histories induced distinct
conditional binary response profiles; because baseline competence was 87.5%
and the principal law comparisons were not position-matched, this does not
yet prove that correctly executed overwritten facts remain causally active.

**CRC verdict (combined CRC-0 + CRC-1, Codex strategic review):** Two
experiments, two task families, two observers. Both identity partitions.
CRC-1 removed CRC-0's strongest design excuses (fine observer, root-dependent
actions, no guaranteed collisions) and still landed far from compression.
CRC on Qwen3-1.7B-Base is **closed** as a discovery engine. Retained only as
a downstream assay if a future method discovers a plausible synchronizer.

**Codex adversarial corrections (adopted verbatim):**
1. "Competence failure makes it more decisive" is scientifically wrong —
   invalid data cannot strengthen claims about competent computation.
2. Normalized {0,1} is not "the coarsest possible observer."
3. Q_1=Q_2=32 is trivial, not meaningful tower stabilization.
4. Algebraic relation rates are confounded by length, position, competence.

**Cross-framework convergence:** CRC-1 (Qwen3-1.7B, Python) partly replicates
predictive_setter_algebra_v1 (Qwen3-0.6B, NL) primacy findings. Extends that
failure class to 1.7B and a probabilistic binary observer.

**Competence position analysis:** 100% correct when queried register's last
write is the final line, 75% when it's not. All 8 failures predict the
overwritten value. Positional structure, not yet causally localized.

**Next: CEG-1 (Causal Erasure Graph v1).** Characterize the causal support of
path dependence via attention mask interventions. Where does dead information
live: direct KV access, relay into later states, or no legal synchronizer?
16 histories × 6 intervention arms × 2 channels. ~800-1000 CPU forwards.
Codex design: `.codex_post_crc1.txt`.

---

## CRC-0 result (2026-09-04; interpretation revised by audit #41, 2026-09-05)

The historical CRC-0 panel (Qwen3-1.7B-Base, 36 roots, words ≤ 2 over
{N, H, R}; numbers in experiments/EXPERIMENTS.md and
docs/STATE_ARCHIVE_closed_lines.md) had no distinct same-variable
W_H-close pairs at τ = 0.01, giving no positive preservation evidence. The
recorded closure score had zero coverage and did not test successor W_H
profiles. Two retained same-variable witnesses violate immediate-tolerance
preservation under H. These finite numerical summaries establish neither
exact quotient stabilization nor absence of compact structure elsewhere.
Earlier wording ("no closure", "tower stabilizes", "apparatus works",
"almost compressible", task family exhausted) is withdrawn.

## Archived claims

Every section below this point in the original STATE.md was moved verbatim to
docs/STATE_ARCHIVE_closed_lines.md (2026-09-05). Index (name, dates, verdict
word as written in each section header):

- Prior campaign: Continuation-Refinement Calculus (2026-09-04)
- Prior campaign: Intensional Descent Criterion (2026-09-03 — 2026-09-04, CLOSED)
- Frozen-residual measurement lane (2026-09-01, CLOSED)
- Real-model line 2026-08-29 (historical record; all constructions closed under audits #27–#31)
- Direction (set 2026-08-27); Guiding question (Devansh, 2026-08-27); Live question; Constraints
- Prior program (closed 2026-08-27)
- Closed toy program — licensed wording (Rounds 36–37; ENDED 2026-08-29)
- NLM-007 — CLOSED (2026-08-29; audit #22 closing statement, verbatim)
- NLM-007 — adoption-order record (Rounds 1–34; superseded orders kept verbatim), including the NLM-001 through NLM-006b verdict/lock sections, Rounds 10–33, audits #2–#20, and the four-cell evidence gate (2026-08-27 — 2026-08-29)
