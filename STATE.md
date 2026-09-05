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
fidelity). Last-write normal form theorem (quotient bounded by ∏(|V_r|+1)
in general, ∏|V_r| for initialized registers; not history length). Five
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

**Distance from central claim: 0.** This IS the native math artifact.

**Next step:** Codex WB-2 design gate for commit-seal carrier experiment (rung 1
of staircase). Then implement and run on CPU.

---

## Qwen path-dependence line: CLOSED (2026-09-04)

CEG-1R preserved as a deferred diagnostic only. The line produced one
transferable finding: pretrained transformers have no endogenous overwrite
boundary — dead information persists in the append-only carrier, and the
residual under attention cuts is broadcast context sensitivity, not semantic
relay. This finding directly motivates the construction program above.

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
Qwen path-dependence line closed. Transferable finding: no endogenous overwrite
boundary in pretrained transformers. Construction program opened (CR-10+).

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

## CRC-0 result: FAIL — no compression (2026-09-04)

**CRC-0 (Codex-designed, decisive):** 36 roots, 3 typed actions, exhaustive
words through length 2, full 151K-token distribution. Q_1 = 36 classes (one
per root state). No compression, no closure, no transfer. The model maintains
36 behaviorally distinct states — every (variable, value, presentation)
triple is unique after one action. Tower stabilizes trivially at the identity
partition. Verdict stable across tolerances 0.005–0.02. Clean negative:
apparatus works, model doesn't compress on this task family. Min D_1 = 0.0165,
just 65% above primary tolerance — the model is ALMOST compressible but not
quite. Q_0 had 33 classes (some presentation equivalence) but all split at Q_1.

**Implication:** The CRC framework on the SVB variable-scoping task family
produces a clean but trivial result. The refinement tower exists but is the
identity. This closes the SVB task family as a substrate for finding compact
native structure in Qwen3-1.7B-Base.

**What remains open:** Whether a different task family (one where the model
genuinely compresses — i.e., distinct histories produce equivalent futures)
would yield nontrivial CRC structure. The framework itself is not falsified;
the task family is exhausted.

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
