# Everything we tried, and why each direction ended

A chronological record of every direction the project took, what it was, and what
killed or stalled it. Results that survived are in [RESULTS.md](RESULTS.md). The
decision log is [DECISIONS.md](DECISIONS.md). Source material for each item is under
`research/<line>/` and `archive/history/`. Many July files were bulk-deleted in commit
`b1684be` (2026-10-01) and are recoverable with `git show b1684be^:research/<file>`.

## The mission, in one paragraph

Do what Turing did in 1936: define a model of computation, prove what it can and
cannot do, and show it is better in some useful way. The working bet is that Turing's
three axioms (finite symbols, finite states, local steps) are dials, and that models
that stuck (BPP, IP, streaming, quantum) each made a free resource first-class and
came with a "cannot" theorem. See [../AGENTS.md](../AGENTS.md).

## Phase 1: July 14-16, 2026. Pick a dial, search for a model

All of this happened in three days (about 165 commits).

| Direction | What it was | Why it ended |
|---|---|---|
| Dial menu D1-D5 | Mind-changes, precision, energy, interaction, physical relaxation as candidate "atoms to turn". | Each dial manufactured advantages by moving costs across undefined boundaries ("boundary laundering"). Nothing proved. |
| EFRS | One cost vector for any analog/physical step, with a digital-simulation impossibility target. | Killed by a novelty check: only an accounting wrapper around three occupied results (Bournez-Graca-Pouly, Blanc-Bournez, Braverman-Grigo-Rojas). |
| Lambda / CCLT | A composable "condition length" for noisy stochastic transducers. | "Dalalyan plus bookkeeping"; the end-to-end statement was never proved. |
| Relaxation, Gibbs, tracking, nonlocal cards | Physical stochastic dynamics as a computing step. | Absorbed: the baseline may run the same device. |
| OFFT / FSSA | Optical Fourier transform; tile self-assembly. | The VLSI Fourier literature already occupies the sandwich; self-assembly collapses into known tile-assembly theory. |
| PFPWO | Post-fabrication programmable wave operator. | Absorbed by OFDM and programmable photonics. |
| ICSM / EIT | Causal commitment under incomplete information; information transport across cuts. | Occupied (linear types, quantum instruments etc.). |
| FSTT-2C | One stochastic-transducer theorem spanning two constructors. | The proof reduced to circuit unrolling and a telescoping sum. |
| Class-first cards CPT, QNC, RRC, OCCC, CIC, FrCC | Change what a "problem" is. | Each is a known object: full abstraction, choiceless computation, unlearning, Type-2 computability, quantum instruments, decision-tree complexity. The finite pairs for CIC and FrCC were reviewed as sound and still occupied. |
| Forcing funnel WL11/WL12 | Require an independent antecedent; two blinded constructors. | The blinded experiment gave incompatible mathematics from the same phenomenon (`PRESENTATION-FORK`). Probes died by linear-type and transaction absorption. |
| Mission narrowing and ACDFT | Narrow the mission to "honest theory of physical computation claims". | Devansh ruled the narrowing wrong. The ACDFT lower bound survives as the one reviewed result of the era, demoted to a side track. |
| Graveyard analysis | Study why 41 killed ideas were killed. | No dominant blocker and no vacancy: 19 of 41 had all four winner properties in an existing home. The one attempt (CfAC) failed source-binding. |
| Three-geometry sweep G1-G3 | Restriction-defined families, proof-object-first, class-operator-first. | Hard pre-gate stops. G4 never run. |
| Synthesis paradigm | Combine the 41 homes: a "freshness to deletion" seam, bridge, composed CAN and a CANNOT. | Two terminal gates returned NO. The "CANNOT" was only six failed proof routes. A later analysis showed the extension data is derivable, so the semantic CANNOT is probably false and the honest redirect is a priced-retention bound. |
| ORRC moonshot (2026-10-01) | A routing-based "heat wall" model. | A zero-context review declared it destroyed: O(1) routing hid an infinity leak, the CANNOT used a strawman, the vocabulary relabelled linear types. Purged the same day. |

**End-of-era statement, in the repo's own words:** "no new computation model, no
completed CAN/CANNOT pair, and no Turing-1936-scale result."

## Phase 2: October 3-4, 2026. Erasure and statistical contexts

After ORRC, a restart kept only the scoped entropy inequality and asked a sharper
question: with a cap on peak per-step entropy loss, which clean computations are
feasible at all? This produced the large body of scoped theorems in
[RESULTS.md](RESULTS.md) section A: exact thresholds, robust-prior and returned-context
laws, three cost meters, masked cleanup. It has the best definitions and the best
evidence hygiene in the repo. Its weaknesses, all from the record:
- the meter is a logical ensemble quantity, and the "ensemble time-sharing" objection
  is unrefuted;
- headline gaps compare different statistical promises;
- the compilers are exponential;
- the operational attempt to tie the meter to a real reset device returned NO-BINDING.

## Phase 3: October 3-4. Access dials

The failed operational binding triggered a question loop that proposed three new
dials: adaptive audit access, recallable input mutation, and observer privacy. They
were tested in one bounded unit. Outcomes are in [RESULTS.md](RESULTS.md) section B:
ADAPTIVE (model already in the literature), RECALL (sorting construction killed, single
repair failed), CAT (component pass, recursion killed). Observer privacy produced a
reviewed rigidity theorem described by its own candidate file as Shannon perfect secrecy
applied to a reconstructible transcript.

## Phase 4: October 4. Endurance-limited storage

A further question cycle ranked three hypotheses (endurance-limited persistence,
bounded-buffer network computation, hard-deadline exact convolution). Endurance won.
In about eight hours (80 commits) the project defined three machine models and ran
three construction units. Details and the current blocker are in
[RESULTS.md](RESULTS.md) section C and [STATUS.md](STATUS.md). Unit scoreboard:

| Unit | Outcome |
|---|---|
| Two-journal fixed-trie design | REPAIR-FAILURE. One repair spent; the elementary controller could not be compiled. |
| Indexed serial-head RAM compiler | RAM-FAIL. One repair spent; no legal uniform generation controller. |
| N=4 exhaustive solver (two attempts) | TINY-VOID twice (launcher defect; then an unchecked UNSAT). |
| Checked-certificate rerun | Succeeded: H <= 4 at N=4, k=2, L=2 (CANNOT only). |
| Compact crit-bit one-box design | COMPACT-REPAIR. Persistent algorithm survives; one named repair (local-state lowering and return) still unused. |

Paused by Devansh on 2026-10-04.

## What the project tried in every phase and what it never did

Every phase ended in the same shape: a well-scoped negative or conditional record, with
the candidate model proving to be a known object in disguise, or a calibration, or a
component. The pattern recorded in the repo itself, and in the review process notes in
[LESSONS.md](LESSONS.md), is that apparatus (reviews, controls, gates) accumulated
faster than the central artifact. No phase in the record built a runnable model that
could fail the central bet.

## Phase 5: October 7-8, 2026. Roundtable exploration (three engines) and four more lines closed

After the repo reorganization, a three-engine exploration loop (Codex, Droid, Antigravity, orchestrated by Claude; method in the
`roundtable` skill; all records in `ROUNDTABLE/`) looked for a new model. Rules: every disagreement is settled by an experiment; every
citation is fetched and checked; engines are weighted by measured reliability. Results, each closed by an equal-resource translation:

| Line | What it was | How it ended |
|---|---|---|
| Endurance storage (units 1-3b) | A runner with an independent monitor compared the compact crit-bit construction with direct storage and flat-log incumbents | A workspace-matched cold flat log used fewer reads and writes at N=8192 and 16384; a ~100-bit retained cache cut queries to one read. The repair gate failed; demoted to OPEN. An early "flip-read advantage" was an artifact of an O(log)-workspace restriction on the incumbent, caught by Codex at the gate. |
| MacTM / ETM (round-1 ideas) | Noisy aggregate probe machine; decaying memory | A sampling RAM reproduces MacTM's output law at equal reads; ETM's bounds were never well defined. Withdrawn. |
| Disturbance lease | Reversible circuit with one borrowed bit that must be restored within a window of ell steps | Computable functions have F2-degree at most ell+1 at any gate count (exactly degree <= 3 at ell = 2, 3; conjecture with a checked proof sketch, enumeration of 1,978,827 maps reproduced independently). The increment span equals Reed-Muller spaces; capability is the known Reed-Muller-to-Toffoli synthesis plus dirty-ancilla commutators. Brittle under contract changes; nonabelian, quantum and probabilistic variants gave no deadline-dependent non-degree impossibility. |
| Restoration-window catalytic logspace | Catalytic tape restored at least every ell steps | Equals ordinary space O(s + ell) (sparse zero-tape simulation, found independently by two engines); equals L for windows up to about log^2 n / log^2 log n (uses Williams 2025). Strict hierarchies among polynomial windows would imply L != P. |
| Shared KV-prefix scheduling under deadlines | Young systems problem with no mature cost model, chosen by both engines independently | Ordinary RAM matched the exact optimum on all 4,196 instances; the only growing separations are from LRU or EDF baselines that a standard cache-first policy closes. Prior art already has NP-hardness (Dexter et al. 2025) and competitive-ratio lower bounds (Jaillet et al. 2025). |
| Generators (Codex, Droid) | Enumerate tiny cost models; SMT synthesis of separations with certified replay | Calibrated against known results; returned honest negatives (132 apparent borrowed-memory advantages absorbed by ordinary reversible circuits; cheap parity-mask probes return to the serial cost once charged). |

Pattern, confirmed on all four lines: the gadget side is already occupied by an incumbent, and the impossibility side is an elementary algebraic or counting fact.

Addendum (2026-10-08): a fifth line, proof-complexity "lemma-interface throughput" on grid Tseitin formulas (definition-first scoping candidate #1), was killed at its pre-registered scope.
Three proof variants (exposed parity lemmas, resolution, Gaussian-style certificate) were validated line by line by a separate checker for grids 2x2..5x5; an ordinary interpreter reproduced every
exposed-lemma trace byte for byte with identical charged costs. Gaussian-style certificates won on size, traffic and replay work for 3x3 and larger; resolution won on formula depth; exposed lemmas won
only on inference-chain length. The one CANNOT stateable for that language (extension-free exposure of parity on w variables costs 2^(w-1) clauses) reduces to counting; the Hastad switching-lemma bounds
do not transfer to the mixed XOR certificate language. Evidence: `ROUNDTABLE/r8_codex.md`, `ROUNDTABLE/work/codex/r8_*`.

Addendum (2026-10-08, after the three-hour chained rounds): two self-directed chains ran for three hours each. Droid (round 10) logged 93 pre-registered units, all killed at the tested contract or only partly
useful as ordinary facts, with 93 receipts replayed. Codex (round 11) logged 67 units (53 killed, 14 partly, 0 open) and ran large exact computations: 1,973,760 cost entries over 30 model variants for every 3- and 4-input
Boolean function, a fully certified 5-input signed-cube spectrum (all 4,294,967,296 vertices) and a six-language 5-input overhead census. No unit produced a new CAN/CANNOT pair with an advantage over an equal-access
ordinary controller. A scaling check by the grunt engine re-ran 10 matched kills at 39 larger configurations (no gap appeared) and found that 37 of 54 early kill tests were hard-coded toy instances with no scalable form.
Structure that did survive as SIDE RESULTS (not new models; ordinary machines run every witness verbatim): the lease-degree theorem and its group-valued generalization. For output group G the lease-2 class is
constrained only on the abelianization G/[G,G] (degree <= 3 for the sign of S3 and S4, affine for C3 in A4, unrestricted for perfect groups such as A5), because commutators reach the derived subgroup freely. These are CONJECTURES with
author-side proofs; independent proof review is in progress. Pattern, now confirmed over roughly 160 units across seven families of ideas: the gadget side is occupied by an incumbent and the impossibility side is an elementary fact.
Evidence: ROUNDTABLE/r10_droid.md, ROUNDTABLE/r11_codex.md, ROUNDTABLE/work/droid/r10_*, r12_*, ROUNDTABLE/work/codex/r11_*.
