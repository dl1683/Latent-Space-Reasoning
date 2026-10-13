# Terminal Adversarial Gate 1

**Date:** 2026-07-15  
**Reviewer posture:** fresh, zero-context, terminal and adversarial  
**Review status:** COMPLETE — TERMINAL VERDICT `NO`  
**Requested decision:** whether the current archive, at `HEAD`, establishes the four
claims named in the terminal-gate request at their stated scopes  
**Rigor convention:** `PROVED`, `EMPIRICAL`, `HEURISTIC`, and `CONJECTURE` describe
the evidential status of findings. Defects are separately tagged `FATAL`, `MAJOR`,
or `MINOR`.

This review does not ask whether the original Turing-1936-scale mission succeeded;
the archive explicitly says that it did not. It asks whether the current archive
successfully establishes its narrower negative-search, synthesis, method, and
governance claims. A historical Git object may establish provenance, but it does
not silently repair a broken current archive: the constitution requires the
repository to carry its own proof for a stranger.

## 1. Review protocol and evidence boundary

**EMPIRICAL.** I read the constitution and dual-loop termination rule first, then
audited the current authored research record, the named review files, the steering
records, the question-loop syntheses, `WORKLOG.md`, and the Git history. I treated
assertions in review files as claims to be checked, not as proof merely because the
filename says `REVIEW`.

**EMPIRICAL / coverage.** The direct terminal set included `AGENTS.md`, `PROGRAM.md`,
`HANDOFF.md`, `PARADIGM.md`, `SWEEP_RESULT.md`, `CLAIMS_LEDGER.md`, the current and
historical `SYNTHESIS_GRAPH.md`, all `SYN_REVIEW_1`–`12`, all
`PROOF_REVIEW_1`–`6`, both program reviews, the method review,
`CALIBRATION_PROTOCOL.md`, `GRAVEYARD_ANALYSIS.md`, all five steering records, the
batch-synthesis portions of `QUESTIONS.md`, `WORKLOG.md`, and the complete Git log
relevant to their construction and later rewrite.

**PROVED / audit rule.** I distinguish three different propositions throughout:

1. an artifact or review file exists;
2. its exact statement is true at its frozen scope;
3. later summary documents cite no stronger proposition than item 2.

Only item 1 follows from inventory or review counts. Items 2 and 3 require direct
mathematical and cross-artifact checks.

## 2. Claim-by-claim findings

### 2.1 Claim 1 — original-mission search and honest close

**Claim under review.** The original mission was searched across three geometries
and roughly forty-five candidates, then honestly closed with certified instruments,
as stated in `research/SWEEP_RESULT.md`.

**PROVED / UPHELD AT THE FILE'S EXPRESS SCOPE.** `SWEEP_RESULT.md` says the sweep
covered exactly the frozen geometries `G1`, `G2`, and `G3`; it says `G4` was not
executed, forbids global-exhaustion and no-new-class readings, and denies both a new
class and mission success. `research/PROGRAM.md` and `research/HANDOFF.md` preserve
the same transition: the original mission did not succeed and the owner explicitly
pivoted the program to synthesis. The census contains 41 homes; the phrase “roughly
45 candidates” is acceptable only as loose archive-level shorthand, not as 45
independently executed theorem programs.

**PROVED / scope qualification.** The archive certifies a searched and instrumented
negative result over the registered candidate census and three frozen search
geometries. It does not certify exhaustion of possible computation models, of the
three geometries beyond their registered packets, or of the skipped fourth
geometry. The main sweep document states these limits plainly, so this is not
overclaiming inside the canonical sweep result.

**EMPIRICAL / supporting evidence.** The kill packets, steering records, program
reviews, graveyard, question-loop checks, and Git history show repeated precommit,
repair, and negative-result logging rather than deletion of failed routes. The
record is substantially more candid than a normal exploratory archive.

### 2.2 Claim 2 — registered synthesis suite and composition-level CAN/CANNOT pair

**Claim under review.** Under the owner's pivot, the synthesis phase produced six
theorems, two bridges, two governed extensions, and a completed composition-level
CAN/CANNOT pair whose priced extension is proven minimal and whose unextended form
is impossible.

**EMPIRICAL / inventory only.** `research/CLAIMS_LEDGER.md`, `research/PARADIGM.md`
v7, and the surviving portions of `research/SYNTHESIS_GRAPH.md` do register named
objects in all of those roles. The role counts are not counts of disjoint results:
the same composed CAN object is counted as both a theorem and a bridge, and governed
extensions also appear in the theorem list. The honest reading is a suite of
overlapping roles, not twelve distinct mathematical results.

**PROVED / current-archive integrity failure.** The current
`research/SYNTHESIS_GRAPH.md` jumps from section 13 into section 26 and lacks the
entire section 14–25 proof spine, including the cited section 25.F. Its surviving
section 13.4 is textually overwritten by material belonging to the later CANNOT
claim. `PARADIGM.md`, `CLAIMS_LEDGER.md`, reviews, and section 26.F still cite the
missing sections as if they were present. Git object `d327cd1` shows that sections
14–25 once existed; commit `fdecb4b` then rewrote the graph with 2,269 deletions.
Historical recoverability establishes provenance, not a self-contained current
archive.

**PROVED / NOT UPHELD.** Direct step-level replay does not establish the claimed
suite at its promoted rigor. In particular, the reverse `TraceC` theorem does not
derive its semantic conclusions, the exact-price minimality argument supplies no
cost lower bound, the composed CAN is an asserted composition rather than a proved
map, and the universal CANNOT infers nonexistence from failure of enumerated routes.
These are defects in the exact certified statements, not complaints that the scope
is narrow.

### 2.3 Claim 3 — self-contained refute-bridge-recompose method

**EMPIRICAL / registered.** `PARADIGM.md` v7 contains a self-contained tuple schema,
algorithmic procedure, replay condition, failure modes, output contract, and worked
application for `SYN24-RBMR-REGISTER`. `research/METHOD_REVIEW_1.md` found the earlier
version incomplete and required the replay certificate and tuple-level repair; the
current v7 text contains those repairs.

**HEURISTIC / correctly labeled.** The archive presents the method as a governed
research methodology, not as a theorem that is optimal, complete, or guaranteed to
discover seams. At that methodological scope the contribution is real and
substantially self-contained. Its value is procedural and reproducibility-oriented;
it is not evidence for a new computation class.

### 2.4 Claim 4 — independent review, kill certification, and bounded debt

**EMPIRICAL / file-count component verified.** The named files exist: 12 synthesis
reviews, 6 proof reviews, 2 program reviews, and 1 method review. Many reviews are
genuinely adversarial: they identify false statements, circular definitions,
missing variables, hidden representation dependence, and necessary repairs. The
proof reviews in particular contain several credible step-level attacks rather than
mere approval prose.

**PROVED / count is not validity.** File count does not establish that every
promoted claim is sound or that every repair received a fresh independent review.
The terminal audit therefore does not accept “independently reviewed” as a proxy
for “proved.” The synthesis reviews must stand or fall with the arguments they
certified.

**EMPIRICAL / debt disclosure is strong but not complete closure.** `PARADIGM.md`
v7 names portability, identity, alternative-carrier, boundary, replay-gap, and
composition-glue debts. `SWEEP_RESULT.md` preserves geometry limits and the G3 hash
repair. These are candid disclosures. They do not bound away a defect in the
logical core of a registered CANNOT or minimality theorem, and they do not excuse a
missing canonical proof spine.

## 3. Hash recomputation record

All recomputations were CPU-only.

1. **PROVED / MATCH.** SHA-256 of the current
   `research/WL16_G2_PACKET.md` is
   `b70c0eeed1fbe8332497ad6717e97c2cf7d139a2d218b5ffdb79e0da1c1938ae`,
   matching the G2 certificate in `SWEEP_RESULT.md` (case-insensitive rendering).
2. **PROVED / MATCH.** SHA-256 of the exact serialized `F_payload_v1` is
   `0343b7db01385b808afd1f138b694618375bce34b8592adff496f46026175fcc`.
3. **PROVED / MATCH.** SHA-256 of the exact serialized `P_payload_v1` is
   `429ba46ee2904013eb08cf5993d4a3143fa70db7753ce64823120f2839875a8d`.
4. **PROVED / MATCH TO REPAIR; MISMATCH TO ORIGINAL.** Under the packet's explicit
   `UTF8(F) || 0x0A || UTF8(P)` rule, the pair digest is
   `03c4da60b820cb5258055a33143496e1ced3cc529dce761d87397366c1653fe5`.
   This matches the appended repair and does not match the preserved original value
   `914d4aad...`. The archive discloses the mismatch and gives a reproducible repair.

## 4. Step-level proof samples

### 4.1 Sample A — forward freshness-to-deletion transfer

**Artifact:** `SYN24-SG7.2-B20`, current `SYNTHESIS_GRAPH.md` sections 11–11.R;
`SYN_REVIEW_1.md`.

**PROVED / UPHELD ONLY AT A STRONG, NEAR-DEFINITIONAL SCOPE.** The simulator stores
the complete deterministic `M22` state, replays the same visible event, obtains a
total witness from the fixed witness family, and applies the interface-supplied
dependency-closure deletion. Induction then preserves answers. The additive cost
claim follows if the declared witness/replay operations really have the fixed
bounded costs absorbed into `h_F,k_F,a_F`. Those assumptions are explicitly inside
`WellForm`, `NC`, and the fixed-family contract. Thus the construction is valid as a
transport lemma under those assumptions. It is not evidence that deletion witnesses
can be recovered from a weaker interface, and the file correctly excludes the
reverse direction here.

**HEURISTIC / significance.** This is a legitimate but very small simulator lemma:
the target interface already supplies total deterministic behavior, correct
freshness semantics, sound queries, total witnesses, and deletion closure. It can
count in a local registered suite, but it does not support the stronger later claims
without independently valid bridge theorems.

### 4.2 Sample B — reverse `TraceC` reconstruction

**Artifact:** `SYN24-SG7.2-TRACE-RECON-v3`, historical
`SYNTHESIS_GRAPH.md` sections 14.2–14.R at Git object `d327cd1`;
`SYN_REVIEW_4.md`.

**PROVED / DEFECT.** Sections 14.2.1–14.2.3 establish only that a recursive parser
over the declared trace grammar is total and deterministic and that its output is a
function of `(evseq, depLog)`. The promoted theorem in section 14.2.4 additionally
concludes `Qeq`, `DelComp`, and a specific cost inequality. No induction or semantic
simulation invariant connects the parser's reconstructed state to `Ans19`,
`Ans22`, deletion correctness, or the cost model. Total deterministic parsing does
not imply semantic correctness.

**PROVED / anti-smuggling gap.** `WF3` says `trLog` is a fixed encoding of event and
dependency history, but `G_tr` includes an `x : DataAtom` payload. The proof shows
only “no information beyond the declared history”; it does not prove that the
declared history itself excludes target witnesses or answer-bearing data. A proper
anti-smuggling theorem needs a typed provenance relation and a semantic simulation
invariant, neither of which appears in the proof.

**Verdict:** `REFUTED AS PROVED`. The theorem may be repairable, but the registered
proof does not establish its conclusion. `SYN_REVIEW_4` audits grammar closure and
canonicalization, then accepts the semantic conclusion without proving the missing
link.

### 4.3 Sample C — reindex transport

**Artifact:** `SYN24-WG24.2-REINDEX-v3`, historical
`SYNTHESIS_GRAPH.md` sections 14.3–14.R at `d327cd1`; `SYN_REVIEW_4.md`.

**PROVED / TAUTOLOGICAL CORE ONLY.** The relation `r1 ~= r2` is defined by equality
of their canonicalized states. `ObsBundle_r` is then defined on that canonicalized
state, and the proof says its fields are extensional under canonical equality. That
does prove representation invariance for predicates explicitly defined on the
canonical quotient.

**PROVED / unsupported promoted breadth.** It does not separately prove that the
rollback, query, deletion, boundary, and cost predicates of the original carrier
are all extensional with respect to that quotient. Lemma 14.3.1.2 asserts this
field-by-field; Lemma 14.3.1.3 asserts cost constants are invariant; neither supplies
a definition-level derivation or counterexample-excluding argument. “PASS by
definition” establishes a quotient wrapper, not a substantive rollback-transport
theorem for pre-existing semantics.

**Verdict:** `UPHELD` for canonical-representative equality; `NOT ESTABLISHED` for
the later class-wide semantic reading used in composition.

### 4.4 Sample D — governed identity extension and exact minimal price

**Artifact:** `SYN24-G19-G09-IDENTITY-GOV-32`, historical
`SYNTHESIS_GRAPH.md` sections 24.1, 24.R, 24.R2, and 25.1 at `d327cd1`;
`SYN_REVIEW_8.md`–`SYN_REVIEW_10.md`.

**PROVED / DEFECT IN CONSTRUCTION.** The construction is stated as
`Bridge_33 := B_32 ° LiftRep_32`, followed by assertions that `LiftRep_32` and
`Bridge_33` are total and that `repCanon` is deterministically extracted. The cited
section does not define `B_32` and `CanonRep` with enough operational and semantic
content to recompute `Qeq`, `DelComp`, or the target map. The cost inequality with
constants `(2,1,7)` is likewise declared as a substitution of “fixed-overhead
bounds,” without those component bounds being proved in the section.

**PROVED / exact-minimality failure.** Sections 24.R/24.R2 enumerate subsets of
the chosen fields and state why each named route fails. Even granting those field
necessity arguments, the “lower-cost triples” case merely says each coefficient is
required by a corresponding operation. It supplies no adversarial family, counting
argument, or operational lower bound proving that every valid implementation must
pay at least 2, 1, and 7. Necessity of an operation is not an exact cost lower bound.
Nor does enumerating subsets of one proposed representation exclude a different
encoding or algorithm on the same admitted interface.

**Verdict:** `REFUTED AS “PROVEN-MINIMAL PRICED EXTENSION.”` At most the archive
records a proposed sufficient extension and an empirical/heuristic minimality check
against its enumerated field subsets. `SYN_REVIEW_10` repeats the field-enumeration
argument and calls exact-cost minimality closed without supplying the missing lower
bound.

### 4.5 Sample E — composed CAN

**Artifact:** `SYN24-G22x34-TRACEC-REINDEX-GOV-34`, historical
`SYNTHESIS_GRAPH.md` section 25.3 at `d327cd1`; `SYN_REVIEW_11.md`.

**PROVED / DEFECT.** The proof text states “there exists a composed bridge
`Bridge_34`,” declares the sum of cost constants, reports that Q/S/R blockers pass
after extension, and lists four vector checks. It never defines `Bridge_34` as a
composition on typed states, proves that the codomain of each component is the
domain of the next, establishes the semantic invariant by induction, or derives the
cost inequality from displayed component inequalities. The assertion that the
extension “resolves conjunction” is the conclusion needing proof.

**Verdict:** `NOT ESTABLISHED`. `SYN_REVIEW_11` verifies that all named dependencies
are registered and that the intended scopes line up narratively. It does not fill
the missing typed composition and semantic proof, and several dependencies are
independently defective in Samples B–D.

### 4.6 Sample F — universal unextended CANNOT

**Artifact:** `SYN24-G22x34-UNEXTENDED-ONE-SHOT-IMP`, current corrupted section
13.4/26 material and section 26.F; `SYN_REVIEW_12.md`.

**PROVED / FATAL LOGICAL GAP.** The theorem quantifies over every admissible
strategy on `U_fixed` and denies the existence of any one-shot G22-to-G34 law. Its
argument establishes only:

1. the previously attempted conjunction did not produce such a law;
2. routes Q, S, R, and U fail under the recorded constructions; and
3. strict subsets of the selected extension package fail the archive's named
   bridge obligations.

None of those statements excludes an unenumerated algorithm, encoding, invariant,
or direct map on `U_fixed`. No indistinguishable-pair counterexample, diagonal
argument, information lower bound, invariant obstruction, or exhaustive normal-form
theorem reduces every admissible strategy to Q/S/R/U. `SYN_REVIEW_12` makes the
invalid inference explicit: after acknowledging that section 18 does not prove a
global nonexistence law, it concludes universal nonexistence because “no full
candidate was found.” Failure to find a construction is not a CANNOT proof.

**Verdict:** `REFUTED`. The extension and unextended statements are not proved
logical complements. Consequently the archive has no completed composition-level
CAN/CANNOT pair at its claimed rigor.

## 5. Severity-tagged defect ledger

Severity means: `FATAL` defeats at least one exact terminal claim or the requested
YES verdict; `MAJOR` requires substantive repair before the archive is
self-contained or reliable; `MINOR` does not by itself defeat a surviving claim.

### FATAL-1 — universal CANNOT is not proved

**PROVED.** Failure of Q/S/R/U and selected strict sub-extensions is promoted to a
universal nonexistence theorem without a completeness theorem for those routes.
`SYN_REVIEW_12.md` substitutes search failure for impossibility. This defeats the
claimed composition-level CANNOT and therefore the completed CAN/CANNOT pair.

### FATAL-2 — “proven-minimal priced extension” has neither a complete map proof nor an exact cost lower bound

**PROVED.** The archive declares `Bridge_33`, `(2,1,7)`, and subset failures, but
does not prove the bridge's full semantics or any lower bound making those constants
minimal. This directly defeats the exact wording of Claim 2.

### FATAL-3 — the composed CAN and a load-bearing reverse theorem lack their semantic proof steps

**PROVED.** `TRACE-RECON-v3` proves parser totality/determinism but not `Qeq` or
`DelComp`; the flagship CAN declares a composed bridge without defining and proving
the typed composition. Because the composed CAN depends on these steps, the CAN side
is not independently established even before considering the CANNOT failure.

### MAJOR-1 — canonical synthesis proof spine absent from current archive

**PROVED.** The current `SYNTHESIS_GRAPH.md` lacks sections 14–25, including its
cited sections 14.R, 19.F, 24.R, 24.R2, 25.3, and 25.F. The material is recoverable
from Git object `d327cd1`, but current canonical citations are dangling after rewrite
commit `fdecb4b`. A terminal archive cannot require a stranger to discover the
right historical blob in order to reconstruct its promoted proofs.

### MAJOR-2 — synthesis review gate exhibits review theater at the decisive promotions

**PROVED.** The review series is not uniformly superficial—early reviews catch real
errors—but `SYN_REVIEW_10` accepts exact-cost minimality without a cost lower bound,
`SYN_REVIEW_11` accepts a declared composition without a typed proof, and
`SYN_REVIEW_12` accepts universal nonexistence from an unsuccessful finite route
search. These are precisely the conclusions the reviews were meant to attack.
Thus the file count is true while the claimed independent certification is not.

### MAJOR-3 — the canonical ledger is not normalized or syntactically clean

**EMPIRICAL.** `CLAIMS_LEDGER.md` lines 55–58 contain a duplicated malformed fragment
inside the table. More importantly, live rows cite absent current sections 14.R,
19.F, 24.R2, and 25.3. This does not create the mathematical defects, but it defeats
the archive's claim to be a clean canonical index.

### MINOR-1 — suite arithmetic counts roles, not distinct results

**EMPIRICAL.** “Six theorems + two bridges + two governed extensions + pair” is a
role inventory with overlap: the flagship CAN is both theorem and bridge, and the
governed replay object is both theorem and extension. `PARADIGM.md` is mostly candid
about this, but a stranger can reasonably read the headline as a count of distinct
results. State “six theorem-role entries, with overlapping bridge/extension roles.”

### MINOR-2 — “roughly 45 candidates” needs the census/execution distinction

**EMPIRICAL.** The corrected graveyard census contains 41 killed candidate rows,
while only three frozen geometry packets were executed and G4 was skipped.
`SWEEP_RESULT.md` states this correctly. The shorthand is acceptable only if it is
not read as roughly 45 equally deep candidate executions.

### NON-DEFECT — G3 hash mismatch was repaired honestly

**PROVED.** The original pair digest is wrong under the later explicit byte rule,
but the packet preserves the bad value, supplies the exact derivation, records the
correct digest, and the recomputation above matches. This is good provenance, not
debt laundering.

## 6. THE PLAIN VERDICT: NO

**PROVED / terminal disposition.** Trying hard to tear the archive down, I am **not
genuinely won over that the current archive establishes everything it claims at its
stated scopes**. Claim 1—the scoped three-geometry negative sweep—is honestly
established as a calibrated research record. Claim 3—the registered
refute-bridge-recompose procedure—is a credible, self-contained **HEURISTIC** method.
Claim 2, however, is false at its promoted rigor, and the absolute review component
of Claim 4 therefore also fails. Since the requested verdict is conjunctive, the
terminal answer is `NO`.

The three strongest attacks and how the record fared:

1. **Attack: demand an actual universal obstruction for the CANNOT.** The record
   failed. It enumerates failed routes and omitted fields, then converts “no
   candidate found” into “no admissible strategy exists.” That is quantifier
   inflation, not an impossibility proof.
2. **Attack: recompute the claimed minimal extension and its exact price.** The
   record failed. It supplies a proposed field bundle and asserted constants, but no
   complete bridge semantics and no lower bound proving the constants minimal.
3. **Attack: follow every current canonical citation and replay the composed CAN.**
   The record failed twice: its proof sections were deleted from the current graph,
   and the historical proof text declares rather than proves key semantic
   composition steps. The later reviews check registry and scope bookkeeping more
   carefully than the missing mathematics.

## 7. Exact holes that must close before a new terminal gate

1. Restore one canonical current `SYNTHESIS_GRAPH.md` containing the complete,
   non-overwritten proof spine and repair every ledger/Paradigm/review pointer. Do
   not make Git archaeology part of theorem verification.
2. For `TRACE-RECON`, define a typed semantic relation between reconstructed state
   and both source/target machines; prove it by induction; derive `Qeq`, `DelComp`,
   and the cost inequality; and prove provenance excludes answer-bearing history,
   not merely arbitrary byte tokens.
3. For the G19-to-G09 extension, define every component map and cost model. Prove
   sufficiency end to end. If minimality is retained, prove a representation-robust
   field lower bound and quantitative lower bounds for each claimed minimal cost
   coefficient; otherwise downgrade to “one sufficient priced extension.”
4. Define `Bridge_34` as a typed composition, show all domain/codomain obligations,
   prove the semantic invariant and cost inequality step by step, and send the
   repaired proof to a fresh reviewer who is not instructed to treat registry
   membership as proof validity.
5. Replace the unextended CANNOT with a real universal argument: either an explicit
   indistinguishable pair/counterfamily for every admissible strategy, a preserved
   invariant that the target violates, or a proved normal-form/completeness theorem
   reducing all strategies to an exhaustively refuted route family. If that cannot
   be done, retract the CANNOT and the pair-completion claim.
6. Re-run independent reviews of all repaired load-bearing results, then correct the
   suite headline and canonical ledger so role overlap, review status, proof status,
   and remaining debts cannot be confused.

**HEURISTIC / honest surviving accomplishment.** Even with a `NO`, this archive has
real value: it records an unusually disciplined, scoped negative search; preserves
failed candidates and corrections; supplies reproducible calibration and hash
instruments; and develops a useful synthesis-governance method. What it does not yet
possess is the promoted composition theorem pair or proven-minimal extension that
would justify closing the synthesis program.
