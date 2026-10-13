# Calibration Protocol (compact machine-checkable consolidation)

**Date:** 2026-07-15

This file consolidates existing calibration law for the `PROGRAM_REVIEW_2` repair unit.
It is a predicate index, not a new law: every condition below cites the source artifact(s)
it already records.

## 1) Forcing signatures (R1-R7)

### 1.1 R1-R6 from forcing-first antecedent screening

From `research/QUESTIONS.md`:

- `R1` (**Dated recurrent constituency**): more than one finite instance appears in
  recorded practice before class naming.
- `R2` (**Operational pivot**): practice changes an admitted act/observation/composition
  rule in response to recurring bottlenecks.
- `R3` (**Erasure witness**): deleting the pivot with task and ordinary costs preserved
  changes capability or can make a failure statement inexpressible.
- `R4` (**Non-isomorphic convergence**): at least two independent non-isomorphic
  presentations expose the same tension.
- `R5` (**Open-family scale pressure**): a growing family or repetition with explicit
  nontrivial cost/scale is part of the packet.
- `R6` (**Prefomal CAN/limit tension**): reproducible technique and internal
  limitation/counterpractice are both present at the declared scope.

Predicate:
`FORCING_SIGNATURE := R1 ∧ R2 ∧ R3 ∧ R4 ∧ R5 ∧ R6`.

### 1.2 R7 seed fields (strict-delta handoff)

From `research/QUESTIONS.md` and `research/WL12_ADJUDICATION.md`:

- `R7.1` finite natural family
- `R7.2` intrinsic finite parameter
- `R7.3` same-contract paired seed
- `R7.4` irreducible obligation
- `R7.5` separating fingerprint

Predicate:
`R7 := R7.1 ∧ R7.2 ∧ R7.3 ∧ R7.4 ∧ R7.5`.

Discrepancy note: several `QUESTIONS.md` sections use `R7-STRICT-DELTA-` wrappers around this
seed; `WL12_ADJUDICATION.md` uses `I4-R7` naming for the same five logical fields.
This protocol records the five-field requirement as a single `R7` predicate and flags this
naming split as unresolved editorial drift.

## 2) Noun-erasure and typed preservation lane

### 2.1 Typed noun erasure predicate

From `research/CLASS_GATE.md` C1 + `research/WL12_ADJUDICATION.md` I6:

`NOUN_ERASURE(packet)` holds iff:

- a uniform erasure map `E` is applied,
- observation and composition remain typed and extensional structure is preserved only up
  to the erasure,
- and the result is evaluated under this protocol's same observer/task contract.

Disruption condition:
`¬NOUN_ERASURE` means the candidate fails if noun erasure alone resolves a claimed
invariance/tension into a free relabelling artefact.

## 3) CLASS_GATE v2 predicates (C0-C6)

From `research/CLASS_GATE.md` (Version 2):

- `C0`: finite, uniform, infinity-leak audit; fail on infinity leaks, unbounded advice,
  uncomputable quotient, or ORACLE-FIAT in finite contract.
- `C1`: relabelling check; fail if erasure is only candidate-authored wrappers and
  preserves all computational data.
- `C2`: resource-reindexing check; fail if one monotone computable map `f` exactly
  transports full packet including theorems.
- `C3`: antecedent forcing check; fail if no pre-existing two-presentation antecedent,
  no uniform observation, or no preserved distinction.
- `C4`: occupation check; fail on `OCCUPIED-BOUNDARY` or `OCCUPIED-PRODUCT`.
- `C5`: theorem-readiness check; fail on missing CAN/CANNOT, nontriviality mismatch,
  or missing exact architecture.
- `C6`: positive suffix; pass only if no earlier token fired and one forced
  class-change witness with route-specific nontransport clause is identified.

Predicate:
`CLASS_GATE_OK := C0 ∧ C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 ∧ C6`.

## 4) FORMATION_ROUTER v2 predicates

From `research/FORMATION_ROUTER.md` (Version 2):

### 4.1 Carrier and route selection

- `carrier_kind :=` `RESTRICTION` if `CARRIER` enforces access bound/removal; else
  `EXTENSION` if it adds explicit primitive/operation; else `UNDERDETERMINED`.

- `F0.1` CLASS-OPERATOR: class-level finite presentations + class theorem scope.
- `F0.2` PROOF-NATIVE: derivations/certificates + proof transformations.
- `F0.3` INTERACTION: two quantified roles/strategies with adaptive exchange.
- `F0.4` RESTRICTION/ACCESS: explicit bounded/forbidden access operations.
- `F0.5` ADDED-EXTENSION: explicit added primitive/operation/state transition.

### 4.2 Differential checks

- `D-EXT`: reject if generated product already has exact closure/architecture.
- `D-RES`: reject if unrestricted comparator reproduces the full restricted theorem contract.
- `D-INT`: reject if adaptive-role quantifiers erase under transfer.
- `D-PROOF`: reject if ordinary normalization/sum products already carry full proof contract.
- `D-OP`: reject if standard class operator closure already supplies exact theorem contract.

`ROUTER_OK :=` one of F0 routes chosen by frozen `carrier_kind` plus matching `D-*` non-rejection.

## 5) Anti-reduction route protocol

### 5.1 WL16 pre-gate route matrix (class-operator-aware)

From `research/WL16_G2_PACKET.md` and `research/WL17_G3_PACKET.md`:

- `Route A`: direct derivation-map route.
- `Route B`: derivability-predicate route.
- `Route C`: family-compression route.

For each route `r`, define `R7_r` as: route witness is context-independent and produces
(1) location invariance and (2) explicit nontransport witness across all required normalizers
and carriers.

`ANTI_REDUCTION_PASS(g3/g2 unit) := RouteA ∧ RouteB ∧ RouteC ∧ R7_r`.

### 5.2 Discrepancy note for anti-reduction

- `WL16_G2_PACKET.md` records route-matrix failure as context dependence and lack of
  certified context-free nontransport witness.
- `WL17_G3_PACKET.md` records route matrix fail under fixed carriers `{Lean/AFP, Dedukti}` and
  fixed normalizers `{N_A, N_B}`.

`ANTI_REDUCTION_HARD_KILL` is required if any route either fails `R7_r` or leaves unresolved
obligation (`G3-OPERATOR-LEAK-REINDEXING`, `G3-OPERATOR-METATHEORY-ABSORPTION`,
`G2-COOK-RECKHOW-SUBSUMED`) outside the strict route pass.

## 6) Machine-checkable composite

`CALIBRATION_OK(packet) := FORCING_SIGNATURE ∧ R7 ∧ NOUN_ERASURE ∧ CLASS_GATE_OK ∧ ROUTER_OK ∧ ¬ANTI_REDUCTION_HARD_KILL`

## 7) Verified divergence register

The following are explicit discrepancies preserved without resolution:

1. `R1-R6` vs `R7` live source text is split between `QUESTIONS.md` table blocks and
   `WL12_ADJUDICATION.md` naming; this file keeps the predicate semantics aligned and
   records the wording split as non-resolved metadata drift.
2. `ROUTER_OK` and `ANTI_REDUCTION_PASS` may diverge for non-class-operator geometries
   if `CARRIER` is re-interpreted; this protocol requires frozen `CARRIER` input to avoid
   route underdetermination.
