# Terminal Adversarial Gate 2 — incremental final-gate review

**Date:** 2026-07-16  
**Reviewer posture:** fresh terminal adversary; no private project memory used as
evidence; CPU only; no GPU; no subagents; no commit.  
**Review target:** the post-`TERMINAL_GATE_1.md` close-out at commits `c6f77d0`
and `cb021b8`, plus the whole current archive and its git provenance.  
**Controlling verdict:** **NO.**

## 0. Rigor convention and review boundary

- **PROVED**: checked directly from definitions, byte rules, or finite algebra in
  the repository.
- **UPHELD-AT-SCOPE**: a proof survives only at the explicitly stated scope.
- **CONDITIONAL**: the displayed deduction is valid if named component theorems
  are valid, but this review does not grant those components.
- **NOT ESTABLISHED**: the archive does not contain the proof needed for its
  label.
- **OPEN**: no positive or negative theorem is claimed.
- Defect severities are **FATAL**, **MAJOR**, and **MINOR** relative to this
  terminal archive gate, not relative to the long-run research mission.

I traversed all 195 tracked paths and read the complete governing, current-state,
claim, review, steering, worklog, synthesis, handoff, calibration, graveyard, and
question-loop synthesis surfaces named in the mandate. The committed source PDFs
were audited through their paired extracted texts, source indices, and the exact
claims that import them; this gate does not re-review every source theorem already
outside the three close-out defects. I also read the full git log and recovered
the missing synthesis proof sections from `d327cd1`/`390c1b8` where the current
file required provenance archaeology.

The gate asks a narrower question than mission completion: does the archive
establish what it currently claims at its stated scopes? It does not.

## 1. Recomputed hashes

### 1.1 G2 packet

**PROVED / MATCH.** SHA-256 of the current bytes of
`research/WL16_G2_PACKET.md` is:

`B70C0EEED1FBE8332497AD6717E97C2CF7D139A2D218B5FFDB79E0DA1C1938AE`

This exactly matches `SWEEP_RESULT.md`.

### 1.2 G3 manifest pair

**PROVED / MATCH TO CORRECTED SPEC.** Applying the explicit rule

`UTF8(F_payload_v1) || 0x0A || UTF8(P_payload_v1)`

to the two exact JSON payloads in `WL17_G3_PACKET.md` gives:

`03c4da60b820cb5258055a33143496e1ced3cc529dce761d87397366c1653fe5`

This matches the append-only corrected digest in that packet. The older
`914d4aad...` value remains in the original manifest and in the sweep table as
provenance, but the packet explicitly marks it nonconforming. As in Gate 1, this
is an honest correction, not a defect.

## 2. Gate-1 defect discharge findings

### Defect 1 — universal impossibility inferred from enumerated failures

**Discharge verdict: PARTIAL; the universal overclaim is retracted, but the
narrowed claim still does not match an impossibility proof.**

What the close-out fixed correctly:

1. `SYNTHESIS_GRAPH.md` 27.2.1 defines `AttackEnum(U_fixed)` as exactly six
   entries: Q, S, R, U, the WL27 bridge form, and the conjunction form.
2. It explicitly says the enumeration is incomplete and that an unenumerated
   algorithm, invariant, encoding, or direct map is not excluded.
3. Universal impossibility is marked **OPEN** in section 27, the dated ledger
   corrections, and the late v8 Paradigm correction.

That genuinely discharges the old universal quantifier. But the exact surviving
display at `SYNTHESIS_GRAPH.md` 1065–1069 is only:

> for every strategy belonging to one of the six classes, the recorded
> construction does not establish the target law.

**PROVED.** This is a statement about what the recorded construction failed to
establish, not a mathematical statement that no member of a formally defined
strategy class can establish it. Q/S/R/U are prose route descriptions, not sets
with membership predicates or normal-form theorems. The WL27 and conjunction
entries are two concrete proof forms, not classes of all algorithms. No universal
proof even over each named prose class is supplied.

The archive nevertheless calls this “impossibility established against every
enumerated attack class” in section 27.2.3, labels it
`PROVED-AT-ENUMERATED-STRATEGY-CLASS-SCOPE` in the ledger, and repeats that
impossibility wording in `PARADIGM.md` section 16. **NOT ESTABLISHED.** The exact
supported label is closer to **PROVED / SIX RECORDED ROUTES FAIL TO SUPPLY A
PROOF**, or **EMPIRICAL/HEURISTIC FAILED-ROUTE LEDGER**. It is not a CANNOT
theorem at any strategy-class scope yet.

Propagation also fails in `HANDOFF.md`, which still says the paired CAN/CANNOT
claims are “currency-stable” and the terminal checkpoint is READY. Thus the
Gate-1 universal defect is honestly retracted in the late correction blocks, but
the requested exact-scope and archive-wide discharge is incomplete.

### Defect 2 — proven-minimal priced extension

**Discharge verdict: DISCHARGED BY HONEST RE-SCOPING.**

**PROVED.** A three-field bundle has exactly seven proper field subsets, and
section 27.2.2 lists all seven:

`{}`, `{repCanon}`, `{DeltaW}`, `{DeltaF}`, `{repCanon,DeltaW}`,
`{repCanon,DeltaF}`, and `{DeltaW,DeltaF}`.

The current claim is explicitly restricted to the recorded subset table under the
recorded construction template. It excludes other encodings, algorithms,
invariants, and interface decompositions. It also reclassifies `(2,1,7)` as a
construction price and leaves all coefficient lower bounds **OPEN**. The ledger
and Paradigm propagate both restrictions.

This is not a representation-independent minimality theorem, but it no longer
claims to be one. At its exact record-audit scope—every proper subset in this one
three-field table has a named failed obligation—the statement matches the evidence.

### Defect 3 — missing typed composition proof

**Discharge verdict: NOT DISCHARGED.**

Section 28 materially improves the record. It supplies explicit intermediate
types, exposes the `11.R -> 19.F` premise mismatch, restricts the domain to a
chosen bridge-ready witness, defines the outer composite, separates the unused
trace-reconstruction side certificate, and derives the outer cost algebra. Those
are real repairs. They do not prove the load-bearing component the composite
imports.

The first failing executable stage is Step 4,
`SYN24-G19-G09-IDENTITY-GOV-32`.

The historical source recovered from `d327cd1` shows:

1. `B_32` is specified as a tuple of primed fields called “fixed-grammar
   extractions.” The extraction algorithms and their semantic relation to the
   source and target strategies are not defined.
2. `WF1`–`WF4`, `Qeq`, `DelComp`, and the `(2,1,7)` cost law are initially listed
   as “core obligations (all to be reviewed independently).”
3. The later “proof completion” says the map is total and computable, says the
   transfer was proved, and says the cost follows by “applying fixed-overhead
   bounds from `Bridge_32`.” It never states those bounds, counts the operations,
   or proves the semantic preservation obligations.
4. `SYN_REVIEW_8.md` accepts the construction because it is “compositional by
   declaration” and calls the cost arithmetic internally consistent. That review
   checks the assertion, not the missing proof. `SYN_REVIEW_10` later promotes
   exact-price minimality from use of terms, a conclusion Gate 1 correctly
   rejected.

Section 28 Step 4 imports exactly the unproved outputs: `Qeq`, `DelComp`, the full
`ObsBundle`, and

`Cost09 <= Cost19 + 2*n + B + 7`.

Calling a row “registered” cannot turn those obligations into theorem outputs.
Consequently `(24.q)`, `(24.d)`, `(24.o)`, and `(24.c)` are **NOT ESTABLISHED**,
and the composed CAN stops there.

Step 5 has a related weakness. Canonical reindexing can give a valid wrapper
theorem if every observable and cost is *defined on the canonical quotient* or is
separately proved reindex-extensional. The archive asserts transport of every
`ObsBundle` field and equality of costs; it does not prove extensionality for the
pre-existing semantics. This stage therefore cannot independently repair Step 4.

The trace-reconstruction stage also remains semantically underproved: finite
deterministic parsing proves only that `Reconstruct_tr` is total and deterministic.
It does not prove that its reconstructed certificates, answers, and deletion
obligations agree with the source `G19` semantics. In any event, section 28 admits
that Step 3 is not on the executable branch and is not used to derive the final
`Qeq`; it cannot rescue the composed map.

### 2.4 What is correct in section 28

**PROVED / CONDITIONAL.** Assuming `(C11)`, `(C19)`, `(C24)`, and `(C14)`, the
cost substitutions are correct:

`Cost34 <= Cost22 + (h_F+h_T+2)n + (k_F+k_T)D + B + (a_F+a_T+7)`.

For nonnegative `D`, `B`, and `k_11=k_F+k_T`,

`k_11 D + B <= (k_11+1)(D+B)`

because the difference is `D+k_11 B >= 0`. Hence the displayed
`(h_11+2,k_11+1,a_11+7)` outer shape is algebraically sound. The type arrows also
compose on `ChainAdm_34` if every registered component has its claimed type and
conclusions.

The downgrade from “every `11.R` witness composes” to “a chosen `11.R` witness
that already satisfies the extra `19.F` premises composes” is exact and honest.
It also substantially deflates the theorem: `ChainAdm_34` takes the bridge-ready
intermediate witness as part of the input. It is an adapter theorem on a
pre-certified chain, not an automatic constructor from an arbitrary admissible
`G22` strategy.

Conditional typechecking and correct outer algebra do not validate false or
unproved component premises. `CONFIRM-TYPED-CAN` therefore does not fire in this
review.

## 3. Proof step-check ledger

This review step-checked more than the required four proofs.

| Proof/claim | Step-check result | Exact disposition |
|---|---|---|
| `SYN24-SG7.2-B20` (`11.R`) | The simulator construction is coherent only because `WellForm`, `NC`, witness lookup, deletion safety, and observational soundness are strong premises. The prefix-cost sum and equality induction have the advertised form. | **UPHELD-AT-SCOPE**, as a strongly contracted/partly definitional transfer; it is not independent evidence of a new capability. |
| `SYN24-SG7.2-TRACE-RECON-v3` | Total deterministic parsing follows from finite recursion. The missing invariant is that parsed state equals the actual source semantics. `Qeq`, `DelComp`, and the cost theorem do not follow from parser totality/determinism. | **NOT ESTABLISHED** for the semantic theorem. |
| `SYN24-WG24.2-REINDEX-v3` | Canonical numbering is plausible and can be prefix-functorial. Transport of arbitrary observation, deletion, boundary, and cost fields needs equivariance/extensionality proofs; defining new canonicalized predicates only proves a wrapper theorem. | **PARTIAL / NOT ESTABLISHED** at the broad `ObsBundle` reading used by section 28. |
| `SYN24-G19-G09-IDENTITY-GOV-32` | Map fields, semantic preservation, and fixed overheads are asserted. The review chain never supplies the missing construction invariant or operation count. | **NOT ESTABLISHED**; first fatal executable failure in `Bridge_34`. |
| Section 28 `Bridge_34` | Domain/codomain wiring and outer cost algebra are correct conditionally. The global “induction” mostly re-instantiates imported all-prefix conclusions; it cannot prove an invalid component. | **NOT ESTABLISHED** as a composed CAN theorem. |
| Section 27 field-subset/price correction | Seven subsets are complete; general field and quantitative cost minimality are explicitly open. | **UPHELD-AT-ENUMERATED-TABLE-SCOPE**. |
| Section 27 enumerated CANNOT correction | Universal scope is properly retracted, but the surviving predicate is “the record does not establish,” not “no strategy in the class can.” | **UPHELD as a failed-route ledger; NOT ESTABLISHED as impossibility.** |

## 4. New and persistent defects

### NEW-MAJOR-1 — the narrowed “impossibility” has the wrong logical type

**PROVED.** The close-out changes the quantifier but not the proof kind. Failure of
six recorded constructions is epistemic evidence about this archive. It is not a
semantic CANNOT theorem over six strategy classes. This is a new defect in the
round-2 repair formulation, distinct from the old universal overclaim.

### NEW-MAJOR-2 — the close-out is not propagated to the current handoff surface

**PROVED.** `HANDOFF.md` ends at the pre-Gate-1 v7 state and says the pair is
currency-stable and the terminal readout READY. Neither close-out commit updates
it. `PARADIGM.md` has a late v8 correction, but its composed-CAN status predates
section 28; the ledger alone records the pending-review narrower theorem. A
stranger following the repository’s own handoff file receives a materially false
current state.

### PERSISTENT-MAJOR-1 — the canonical proof spine is still absent/corrupted

**PROVED.** The current `SYNTHESIS_GRAPH.md` does not contain the historical
sections 14–25 on which section 28 relies. Its section 13.4 is spliced directly
into later Batch-35 text. Section 27 restored only the historical 25.3 block, not
the component definitions and proofs. A reviewer must use git archaeology to
recover `14.R`, `24`, `24.R`, `24.R2`, and `25.1`. That fails the repository’s
standalone archive requirement and leaves current citations dangling or
misleading.

### PERSISTENT-MAJOR-2 — the claim ledger is not a clean current register

**PROVED.** `CLAIMS_LEDGER.md` lines 55–58 remain malformed fragments. The old
universal CANNOT and composed-CAN rows remain marked `live` and `PROVED (after
review)`; later correction rows say they supersede those rows only on conflict.
An append-only provenance policy is defensible, but a machine or newcomer reading
the `standing` column sees mutually incompatible live claims. The register needs a
canonical current-view field or explicit superseded standing on the old rows.

### MINOR-1 — the section-28 prefix “induction” is not load-bearing

The component claims already quantify over every prefix, so the proof obtains each
`I_j(rho.e)` by re-instantiating those claims, not by using `Inv_34(rho)` to prove
preservation. This is harmless if the components are valid, but it should be called
pointwise theorem composition rather than evidence that the missing component
semantics were proved inductively.

No new hash, infinity-leak, GPU-use, or git-integrity defect was found.

## 5. Propagation audit requested by the mandate

| Surface | Universal CANNOT downgrade | Minimality/price downgrade | Composed-CAN downgrade/current status | Verdict |
|---|---|---|---|---|
| `SYNTHESIS_GRAPH.md` | Universal OPEN; six routes enumerated, but mislabeled impossibility | Correctly table-scoped; price lower bounds OPEN | Broad reading retracted; bridge-ready claim pending review | **PARTIAL** |
| `CLAIMS_LEDGER.md` | Corrective row present; old universal row still live | Corrective rows accurate | Pending-review narrow row present; old PROVED row still live | **PARTIAL** |
| `PARADIGM.md` | Late v8 note corrects early universal wording | Late v8 note correct | Late note says OPEN after restoration and is stale relative to section 28; early PROVED/pair wording remains under supersession rule | **PARTIAL** |
| `HANDOFF.md` | Not propagated | No current statement | Still says currency-stable pair and terminal READY | **FAIL** |

The mandated “propagated consistently everywhere” condition fails.

## 6. The plain verdict

### Are you genuinely won over that the archive establishes what it claims at its stated scopes?

**NO.**

My three strongest attacks and how the record fared:

1. **Attack the first load-bearing component instead of the outer composition.**
   The outer types and algebra survived conditionally. The identity bridge did
   not: its map semantics, full observation bundle, and `(2,1,7)` operation count
   are assertions accepted by reviews, not proved results. The composed CAN fails
   at Step 4.
2. **Force the repaired CANNOT to state a mathematical predicate.** The archive
   honestly opened the universal claim, but its narrowed theorem says only that
   six recorded routes do not establish the law. The record then relabels that
   fact as impossibility against strategy classes. The attack succeeds: a failed
   proof search is not a CANNOT theorem.
3. **Enter through the archive’s advertised current surfaces as an uninvested
   stranger.** The handoff still announces a stable pair and terminal readiness;
   the canonical graph omits the component proof spine; the ledger contains
   incompatible live rows. The correction can be reconstructed by a motivated
   archaeologist, but the repository does not yet carry one trustworthy current
   story on its face.

The archive does establish substantial scoped negative and methodological work:
the three-geometry sweep is reproducible under its frozen contracts; its corrected
hash provenance holds; multiple failed routes and field-subset failures are
recorded honestly; and section 28 contains a useful conditional type-and-cost
wiring audit. It does **not** presently establish the flagship composed CAN, a
strategy-class CANNOT, a completed theorem pair, a representation-robust minimal
extension, or the project’s middle/ceiling mission.

## 7. Exact remaining holes

The terminal verdict can change only after all of the following are closed:

1. **Prove or retract the governed identity bridge.** Define `CanonRep`,
   `LiftRep_32`, and `B_32` as actual algorithms on the declared types. State a
   source/target semantic relation. Prove `WellForm`, `Qeq`, `DelComp`, every
   `ObsBundle` field, family compatibility, and totality. Derive `(2,1,7)` from an
   explicit cost model and operation count. Do not import the current register row
   as its own proof.
2. **Prove reindex extensionality at the used scope.** For every pre-existing
   observable and cost field consumed by Step 5, prove invariance under
   `r_34 ~= r_09`, or narrow the theorem to newly defined canonical-wrapper
   semantics and say exactly what that loses.
3. **Repair or remove the trace side certificate.** If `14.R.1` remains a claimed
   theorem, prove a simulation invariant connecting `Reconstruct_tr` to the actual
   `G19` behavior. If it is unnecessary to `Bridge_34`, remove it from the composed
   proof and from any narrative that treats it as semantic support.
4. **Choose an honest CANNOT status.** Either relabel Q/S/R/U/WL27/conjunction as
   six failed recorded routes, or define each strategy class formally and prove a
   universal no-go inside each class. Universal impossibility over all admissible
   strategies may remain OPEN.
5. **Restore one self-contained current proof spine.** Put the exact active
   component statements and proofs in the current canonical graph (or a clearly
   linked canonical theorem file), repair the section-13 splice, and remove the
   need for git-history reconstruction.
6. **Make current-state propagation unambiguous.** Update `HANDOFF.md`, the late
   Paradigm status, and the ledger’s standing fields; repair malformed ledger rows.
   Historical overclaims may remain in provenance, but they must not remain
   machine-readably live.
7. **Run a fresh independent proof review after the mathematical repair.** It must
   attack the definitions and invariants, not accept registration, declaration,
   or prior reviewer approval as proof.

Until then, Gate 2 remains **NO** and Invariant 2 remains active.
