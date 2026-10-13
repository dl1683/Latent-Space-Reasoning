# PROGRAM_REVIEW_2 — Independent Hostile Verification Review (ZERO-Context Re-audit)

## Scope, basis, and rigor framing
- Reviewer stance: hostile but fair, aiming to falsify the central claim.
- I read `research/PROGRAM_REVIEW_1.md` **last**, then compared for agreement/disagreement.
- Claim set under test: the **completed calibrated three-geometry sweep** recorded in `research/SWEEP_RESULT.md`.
- Rigor labels are applied per statement: PROVED / EMPIRICAL / HEURISTIC / CONJECTURE.

## Executive verdict
**Verdict: YES** — at the explicitly stated scope, the record establishes a completed, calibrated, honestly-scoped three-geometry sweep.
- Strongest attacks tested:
  1. **Fatal overreach attack**: treat closure results as a global frontier theorem.  
     - Failing condition: any textual artifact claims more than `G1/G2/G3` closure under the stated contract scope.
     - Record response: **partially blocked** by explicit non-implication and scope limits in `SWEEP_RESULT.md`, plus open obligations listed alongside each closed geometry.
  2. **Mechanical-forcing attack**: precommitted tokens are not actually tied to the closure entries by reproducible hashes/proofs.
     - Failing condition: hash/proof trail missing or inconsistent.
     - Record response: **partially passed** — key packet hashes and gate artifacts are present and cross-referenced; however, one manifest-derivation (G3 paired hash path) is under-specified.
  3. **Calibration integrity attack**: `CLASS_GATE`/`FORMATION_ROUTER v2`/R1–R7/anti-reduction claims are asserted without closure of their own audit evidence.
     - Failing condition: unsealed chain between gate assertions and packet proofs.
     - Record response: **passes operationally** for this record: gate files, packet artifacts, and proof reviews are internally cross-cited, though one naming drift remains in non-canonical narrative zones.

## 1) Central claim: are G1/G2/G3 closures mechanically forced at stated scope?
**Rigor: PROVED (with narrow scope caveat)**

Findings:
- `SWEEP_RESULT.md` explicitly frames scope as a sweep over **G1/G2/G3** with a non-implication note (“does not imply ... universal impossibility”), so the claim is not overbroad on its face.
- The closure table links each geometry to distinct terminal statuses and precommitted tokens:
  - G1: `G1-NO-REGISTERABLE-OBJECT`
  - G2: `G2-NO-FORCED-ANTECEDENT` and `G2-COOK-RECKHOW-SUBSUMED`
  - G3: `G3-LOCATION-NONTRANSPORT-CHECKPOINT`
- `G4` is explicitly flagged as `skipped_by_design`, which is consistent with an honest three-geometry scope boundary.
- `CLAIMS_LEDGER.md` and `HANDOFF.md` are internally aligned to the same three-geometry boundary and list the same terminal statuses in their own summary chain.
- The review chain in `STEERING_WL11`, `STEERING_WL13`, `STEERING_WL14`, and `STEERING_WL17` converges on a post-steering continuation using the three kill-tokens and records unresolved items as explicit carry-forward.

Defects in this dimension:
- **MINOR**: Several supporting notes still mix phrasing that can be read as “global completion” unless reader also ingests the explicit scope sentences in `SWEEP_RESULT.md`; this is an editorial/compositional risk, not a logical one.

## 2) Hash precommit audit and recomputation
**Rigor: EMPIRICAL**

Actions performed:
- Recomputed SHA-256 with local hash tooling:
  - `research/WL17_G3_PACKET.md` = `66D5C1F81721152140725AA8DFFC6C45B16B9755B4BAD72A9BEC38CD80835754`
  - `research/WL16_G2_PACKET.md` = `B70C0EEED1FBE8332497AD6717E97C2CF7D139A2D218B5FFDB79E0DA1C1938AE`
- Compared `WL16` file hash with `SWEEP_RESULT.md` matrix: matched exactly at the stated G2 row.
- Recomputed payload hashes from `WL17_G3_PACKET.md`:
  - `F_payload_v1` hash matches the packet’s recorded value.
  - `P_payload_v1` hash matches the packet’s recorded value.

Defects in this dimension:
- **MAJOR**: `WL17_G3_PACKET.md` records a pair manifest hash (`914d4aad6697...`) but the exact derivation function for the pair hash from payloads/manifests is not spelled out in the packet; I could not mechanically re-derive the exact reported pair digest from the raw text alone.
  - Impact: weakens strict reproducibility of the “single hash implies single state” claim for G3 though does not collapse token-level closure.
- **MINOR**: `SWEEP_RESULT.md` digest was not cross-registered to a single stated precommit for full-file chain verification in the same way as WL16; this remains a reporting gap.

## 3) Overclaim scan across claim chain
**Rigor: PROVED + HEURISTIC**

Findings:
- No explicit file in the chain claims that `G1/G2/G3` closure is equivalent to a full computational theorem; several places in the sweep artifacts do include sharpness bounds and anti-implication language.
- The most serious overclaim vectors were known to prior rounds and were partly repaired:
  - `QUESTIONS.md` batches 22–23 call out “three-carrier” and “statement precision” defects.
  - `WL17_G3_PACKET.md` + repair notes align the closure carrier statement with `Lean/AFP` and `Dedukti`, and `SWEEP_RESULT.md` keeps corresponding wording.

Defects in this dimension:
- **MAJOR**: `CLAIMS_LEDGER.md` and residual notes still contain mixed metaphors (“no canonical family admitted” vs “two-carrier checkpoint accepted”) that could let a fast reader infer stronger negative power than the sweep actually certifies. Needs a wording harmonization pass.
- **MINOR**: `PROOF_REVIEW_1..6` quality is uneven; only some reviews provide explicit “line-by-line mechanical” checks vs “structural plausibility,” so trust should be weighted by review depth.

## 4) Instrument calibration: CLASS_GATE + FORMATION_ROUTER v2 + R1–R7 + anti-reduction
**Rigor: PROVED (chain-level, with one missing explicit norm)**

Findings:
- `CLASS_GATE.md` is referenced as the gating contract and is used as admission criterion for geometry proofs and packet registration.
- `FORMATION_ROUTER.md` and `WL17` materialize explicit route matrices (normalizer and anti-reduction routes) that match the claimed protocol shape in steering and packet logs.
- The anti-reduction protocol appears enforced by:
  - anti-reduction route tables,
  - explicit unresolved-obligation separation in repair blocks,
  - and follow-up proof-review checkpoints in the six review packets.
- The two-certificate carrier model (`Lean/AFP`, `Dedukti`) is now consistently represented in the current sweep record’s core artifacts.

Defects in this dimension:
- **MAJOR**: The final artifact chain still depends on a few informal phrases (“route-safe enough”, “operator-safe bridge”) that are not reduced into one compact machine-checkable calibration predicate.
- **MINOR**: There is no single consolidated “instrument spec” file enumerating R1–R7 as a complete executable contract; they are distributed across gate/router packets and steering notes.

## 5) Program-review comparison (`PROGRAM_REVIEW_1.md`)
**Rigor: EMPIRICAL**

Agreement with PR1:
- I agree with PR1’s central diagnosis that the sweep’s fragility is **wording-precision + scope leakage**, not raw geometry closure machinery.
- I also agree that anti-frauding checks are the key long-term risk in this architecture.

Disagreement with PR1:
- PR1 treats the corrected three-carrier handling as partially open; the post-repair trace in `WL17_G3_PACKET.md` + `SWEEP_RESULT.md` plus batch-23 follow-up largely closes that specific mismatch.
- PR1 underweights the G3 manifest-hash derivation gap; I treat this as a material (but not fatal) reproducibility gap.

## Severity-tagged defects summary
- **FATAL**: none found against stated scope.
- **MAJOR**:
  1. Under-specified G3 pair-manifest derivation process (hash reproducibility gap).
  2. Residual over-precision leakage in non-central narrative artifacts (`CLAIMS_LEDGER` / review prose).
  3. Incomplete calibration formalization in machine-checkable compact form (R1–R7 remain distributed and partly natural-language-described).
- **MINOR**:
  1. Inconsistent phrasing in secondary artifacts can induce scope blur for non-technical readers.
  2. Some proof-review depth variance; not all reviews are equally granular.

## Recommendation on post-sweep WL17 options (options 1–4)
Treat each as a bounded action with risk/benefit tradeoff:

1. **Option 1 (least disruptive):**  
   Publish/lock the sweep status at current scope, while adding one-line hardening in `SWEEP_RESULT.md` and `CLAIMS_LEDGER.md` to remove residual wording drift.
   - This is the strongest option for continuity because the core closure chain is coherent and hashes for the terminal geometry artifacts are internally consistent.

2. **Option 2 (calibration-tightening):**  
   Delay closure and strengthen the manifest derivation spec (especially G3 pair-hash computation), plus normalize all R1–R7 references in one compact protocol note.
   - Best if future evaluators are expected to mechanically verify without tribal knowledge.

3. **Option 3 (scope-expansion):**  
   Re-open geometry content to address residual obligations before final statement lock.
   - Highest rigor but highest delay/risk; useful only if record owner wants a stronger-than-asked future-proofed theorem surface.

4. **Option 4 (deferral/rollback):**  
   Pause and re-launch the sweep with explicit contract-first instrumentation, then re-run gate packets.
   - Most conservative; not necessary for the already calibrated three-geometry scope, but possible if reproducibility is demanded above all else.

## Final calibration note
- Under the stated interpretation, the record is **honestly scoped** and the central three-geometry claim is not demonstrably false.  
- The remaining work is mostly about strengthening audit ergonomics and eliminating precision debt, not rescuing a broken theorem chain.
