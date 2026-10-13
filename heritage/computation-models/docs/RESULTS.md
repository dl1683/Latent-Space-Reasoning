# Results register

Every result this project has established, with its precise scope. Labels are the
repo's own (PROVED AT SCOPE / EMPIRICAL / CONJECTURE / HEURISTIC / OPEN) and none has
been upgraded here.

**Read this first.** None of these results is a new model of computation, and none is
a verified CAN/CANNOT pair that beats the standard model. They are scoped theorems
inside cost models the project defined itself. Novelty against the literature is
unresolved for nearly all of them, and the repo says so in the files cited. "Reviewed"
below means a fresh Codex CLI instance attacked the proof and failed. It does not mean
human or peer review. The worker and the reviewer were the same tool family.

Paths below are relative to `research/<line>/theory` or `.../reviews` (see
[MAP.md](MAP.md)). The older claim index is [CLAIMS_LEDGER.md](CLAIMS_LEDGER.md).

---

## A. Erasure / statistical-context line (October 2026)

**The model.** An ordinary Boolean circuit over NOT, CNOT, Toffoli and a one-bit RESET.
Computation must be *clean*: every bit except the output ends at 0 on every input. RESET
is the only gate that destroys information. The model charges a "peak loss" meter: the
largest entropy drop at any single step, maximised over a promised class of input
priors. Three meters appear: J (full joint entropy), L (local target-bit entropy) and V
(worst positive-probability branch). Definition: `erasure/theory/STATISTICAL_CONTEXT_MODEL.md`.

**Standing caveats, repeated in the source files.**
- It is a *logical ensemble meter*, not a calibrated heat, energy or wall-clock cost.
- Comparisons across different statistical promises (uniform vs unknown prior vs
  returned context) are explicitly "not a same-promise simulation separation".
- The general compilers are exponential in n. No efficiency separation is claimed.
- Primary-literature novelty is open: Meila, Ebrahimi-Chen-Khisti, Frank, Demaine et al.,
  Yadav et al. and the catalytic-computing literature already cover major ingredients.

| Result | Plain statement | Label | Evidence |
|---|---|---|---|
| Atomic exact lower | Uniform input, largest output fibre q: some step loses at least phi(q)/M bits at any depth. | PROVED AT SCOPE | `ATOMIC_ERASURE_PROOF`, `ATOMIC_COMPILER_REVIEW_2` |
| Atomic charged upper | A compiler attains exactly phi(q)/M. Feasible iff K >= phi(q)/M. Gate count is exponential in n. | PROVED AT SCOPE | same |
| Same-entropy separation | Two tasks with equal output entropy and equal total loss, one feasible at cap 1/4, one infeasible at every depth. | PROVED AT SCOPE | same |
| Weighted exact frontier | Exact minimal worst-step loss for non-uniform rational priors (heaviest-prefix formula). | PROVED AT SCOPE | `WEIGHTED_ERASURE_REVIEW` |
| Robust prior frontier | One fixed circuit valid for every prior with max atom <= eta exists iff K >= R(q,eta). | PROVED AT SCOPE | `ROBUST_ERASURE_REVIEW` |
| Succinct AND frontier | Clean AND with a returned uniform r-bit Z: optimal work Theta(n + r + n/K). | PROVED AT SCOPE | `SUCCINCT_ERASURE_REVIEW_1/2` |
| Prior-uncertainty work | Not knowing the prior forces Theta(2^n) work for AND at joint cap 3/4. Known uniform needs Theta(n). | PROVED AT SCOPE | `PRIOR_UNCERTAINTY_REVIEW` |
| Prior-uncertainty cliff | At joint cap one bit, robust work is O(n). | PROVED AT SCOPE | same |
| Returned fair context | One returned fair bit removes the exponential gap: AND in 8n+11 gates, peak loss <= 1/2 for every data law. | PROVED AT SCOPE | `RETURNED_FAIR_BIT_REVIEW_1` |
| Returned-context capacity, threshold, feasibility | With r returned uniform bits the feasibility threshold scales by 2^-r. Work is Theta(2^(n+r)) below K = beta, Theta(n 2^r + r) at it. | PROVED AT SCOPE | `RETURNED_CONTEXT_THRESHOLD_REVIEW` |
| Correlated-context obstruction | If only the marginal of R is uniform, K > 3/8 is forced. Product-prior feasibility does not transfer. | PROVED AT SCOPE | same |
| Local-meter frontier | Under meter L the floor is h2(2^-r/8). Below theta_r the work is exponential, at it Theta(n) for fixed r. | PROVED AT SCOPE | `METER_REVIEW` |
| Branch-meter obstruction | Under meter V no finite circuit has a finite universal cap over arbitrary small atoms. | PROVED AT SCOPE | `METER_REVIEW` |
| Uniform-branch exact | Known uniform X: exact V floor is log2(3), work Theta(n + r). Returned context gives no asymptotic V advantage. | PROVED AT SCOPE | `UNIFORM_BRANCH_REVIEW` |
| Min-atom branch law | If atoms are zero or in [delta, 1/8]: b-3 < K* <= b for delta = 2^-b. | PROVED AT SCOPE | `MIN_ATOM_BRANCH_REVIEW`, `..._ERRATUM_REVIEW` |
| Masked-block critical work | Returned XOR masks: optimal work Theta(n + r + 2^r max(1, n/r)) at local cap h2(2^-r). Independence of R from X is essential. | PROVED AT SCOPE | `MASKED_BLOCK_REVIEW` |
| General masked cleanup | For any noninjective f the universal local floor is exactly h2(beta). | PROVED AT SCOPE | `GENERAL_MASKED_REVIEW` |
| Correlated local work | Arbitrary X/R correlation: critical local work Theta(n 2^r + r), a Theta(r) gap vs the product case. | PROVED AT SCOPE | `CORRELATION_LOCAL_REVIEW` |
| Total local tradeoff | One circuit simultaneously attains optimal work and total local cost Theta(n + r). | PROVED AT SCOPE | `TOTAL_LOCAL_REVIEW` |
| Mask-source frontier | With an unknown mask law of max atom tau: floor h2(tau); masked order work persists for tau < 1/2. | PROVED AT SCOPE | `MASK_SOURCE_REVIEW` |
| Window-design tradeoff | With fixed rational reset designs and a two-gate averaging window, product and correlated classes separate by Theta(r) in total cost. | PROVED AT SCOPE | `WINDOW_DESIGN_REVIEW` |
| Erasure-observer rigidity | At the rare cap, two-probe privacy of erased bits forces whole-transcript privacy and a fibre-sized private key. | PROVED AT SCOPE | `ERASURE_OBSERVER_REVIEW` |
| Entropy inequality (the surviving residue of the first "EBC" claim) | Clean computation with per-layer drop cap K needs depth >= ceil((N - H(f(X)))/K). | UPHELD AT SCOPE | `RESTART_EBC_REVIEW` |

**Finite CPU controls.** Registered finite control suites for most of the results above
carry independent raw-data admission (EMPIRICAL). The largest swept 116 circuits and 429.6 billion gate
transitions. Details and file mapping: [`experiments/README.md`](../experiments/README.md).
They test registered cases only and prove nothing in general.

**Refuted or withdrawn.** The first EBC claims "AND needs D >= N/K", "conventional
log-depth AND is a fiction", "cracks the Extended Church-Turing thesis" and a physical
latency of Omega(N^(1/3)) were refuted or not established
(`erasure/reviews/RESTART_EBC_REVIEW.md`). The min-atom shortcut circuit violated the
clean-output contract and was corrected
(`MIN_ATOM_BRANCH_ERRATUM_REVIEW`, found by `OCTOBER_MILESTONE_REVIEW_1`).

**The operational attempt failed.** WINDOW-DESIGN-1 tried to connect the logical meter
to a real finite-time reset device. Result: **NO-BINDING at the registered scope**. It
built a reset reference construction but could not supply the averaging kernel,
reversible-gate overhead, source preparation, bath capacity or controller cleanup
(`erasure/theory/OPERATING_BINDING_CONSTRUCTION.md`).

---

## B. Access-dial units (October 2026)

After the operational attempt failed, a question loop proposed new "dials". Three were
tested in one bounded unit (`DELTA_TESTS_EXPERIMENT`).

| Unit | Outcome | What is established |
|---|---|---|
| ADAPTIVE (audit an array while an adversary erases cells) | OPEN / tactical | An exact rational minimax benchmark for the failure probability, with a charged finite-seed version (PROVED AT SCOPE). The model and main randomness lower bound are already in the literature (Kelman-Linder-Raskhodnikova). The matching efficient seed generator was not found; a proof route costs an extra Q log Q. |
| RECALL translation lemma | PASS, calibration only | Restoring C owner bits needs exactly m max(C - S, 0) queries (PROVED AT SCOPE). A calibration with no sorting consequence. |
| RECALL kernel (interruptible sorting) | H2-OPEN. Construction RESTORE-KILL, single repair failed | A concrete witness where the sparse scanner must read about 5.5 x 10^11 words against a 4.3 x 10^11 budget. No universal sorting lower bound. |
| CAT (catalytic matrix powering) | CAT-OPEN | Component identities admitted at scope on a charged bit-addressed RAM. The recursive powered-input decoder was ROUTE-KILLed by an explicit state collision. A component is not a matrix-powering CAN. |
| OPERATING_BINDING | NO-BINDING | See above. |

---

## C. Endurance line (2026-10-04, paused)

**Problem.** Store k = N/2 logical bits in N binary cells, each of which can change
value at most L times. Commands are FLIP(i) and QUERY(i). The controller is reset after
every command, so all history lives in the cells. How many flips H can be guaranteed?

All results hold for deterministic cold machines with strict exhaustion, paid
initialization and arbitrary nonlinear encodings.

| Result | Statement | Label |
|---|---|---|
| Endpoint relaxation | Any machine induces a static decoder, initializer and endpoint maps. A ceiling on the relaxed graph covers every real machine. | Design-level, VALID |
| Ceiling A | H <= L min(N, 2^(N-k+1) - 1) - 1. Weak at N = 2k. | PASS at scope |
| N = k optimum | With no redundancy H = L - 1 exactly (direct storage). | PASS at scope |
| Ceiling B | H <= max(0, NL - t0 - beta + 1). | PASS at scope |
| Ceiling C | H <= (N-k+1)L + floor((k-1)L/2). A transfer of a published theorem (Jiang-Bruck). Not a novelty claim. | PASS at scope |
| Query-depth law | H + 1 <= L min(N, 2^d - 1) for each logical bit's worst query depth d. A standard decision-tree argument. | PROBE-PROOF-PASS |
| Depth-two obstruction | N=3, k=1, L=2, H=5 needs query depth >= 3. | PROVED at this scope |
| Finite certificate | For N=4, k=2, L=2 no policy achieves H = 5 or 6, so H <= 4. Checked by an independent proof replay. No H = 4 witness is known. | PROVED by checked certificates (CANNOT only) |
| Flat log | Theta(NL / log N) lifetime, but only at unpriced endpoints. Index logging and parity folding already appear in Yaakobi et al. 2014. | PROOF-PASS |
| Wear-envelope lemma | Bookkeeping identity along trajectories. | PASS |
| Conditional adaptivity | IF the compact construction is admitted, no fixed-address query scheme can match it. The antecedent is not established. | CONDITIONAL |

**Not established.** A positive lifetime on an ordinary, fully charged machine. Three
construction units were attempted. The journal design and the RAM-controller compiler
both failed on the same kind of gap (the controller has to be a legal charged program)
with their one repair spent. The compact crit-bit unit survived component attack but
its ordinary-machine realization is not established, and its one named repair has not
been performed. See [STATUS.md](STATUS.md).

---

## D. July program (2026-07-14 to 07-16)

One substantive result:

**ACDFT, area-time lower bound for an approximate coherent DFT.** For a fixed-function
device with a finite interactive cut capacity, an area-time bound AT^2 >= p^2([Q_res]^+)^2/(4q^2 cbar^2),
unconditional only on the "routed Thompson subclass". PROVED AT SCOPE after three
independent reviews, one of which caught a real sign error. The repo records its
practical teeth as zero: it binds no audited real-world claim, and it is physical-device
mathematics, not a new model class. Files: `july-program/theory/FLAGSHIP_RESULT.md`,
`ACDFT_*`.

Smaller reviewed items, all occupied by existing fields: BLSD-D (local stochastic
dynamics unroll to a local randomized circuit, folklore-adjacent), finite CIC and FrCC
pairs (sound but absorbed by quantum instruments and decision trees), QNC and OCCC
counterexamples (elementary), an audit of Extropic's 10,000x energy claim
("UNVERIFIABLE END TO END"), and reusable admission instruments `CLASS_GATE` and
`FORMATION_ROUTER` (heuristic infrastructure). Everything else from this era was
killed. See [ATTEMPTS.md](ATTEMPTS.md).

---

## E. Lease theorems (side results, 2026-10-08; not a new model)

Setting: reversible circuit with immutable inputs, a few borrowed bits that may hold arbitrary values and must be restored, and a writable output; a *lease* is the maximum number of consecutive gate boundaries a
borrowed bit may stay un-restored. Full statements and the independent attack: [`research/lease/LEASE_THEOREMS.md`](../research/lease/LEASE_THEOREMS.md). Evidence index: `ROUNDTABLE/LEASE_LINE_CLOSEOUT.md`.

| Result | Plain statement | Label | Evidence |
|---|---|---|---|
| T1 lease-degree | One borrowed bit, Boolean output: every computable function has F2-degree <= lease + 1 at any gate count; lease 1 only constants; leases 2 and 3 exactly degree <= 3 | PROVED AT SCOPE (independent LLM reviewer; the short-lease lemma is a code-assisted finite check of words up to 4 gates) | Codex proof, Droid attack, 1,978,827-map enumeration reproduced by two implementations and a third engine |
| T2 all-borrower lease-2 | Any number of borrowers on the single star: lease 1 only constants, lease 2 exactly degree <= 3 | PROVED AT SCOPE | Droid review; searches with 2 and 3 borrowers |
| T3 group-valued output, upper bound | For ANY finite group G at lease 2 the abelianized output has Mobius degree <= 3 and twice it is affine | VERIFIED by a fresh derivation (Droid) and attacked independently by Codex: PROVED AT SCOPE for the named groups | Droid r13; Codex r14 (research: ROUNDTABLE/work/codex/r14_REVIEW.md) |
| T3 matching construction | For C2, S3, A4, S4, A5 the commutator-subgroup components are unrestricted (S3 sign degree <= 3; A4/V4 = C3 affine; A5 everything); leases 3 and 4 equal lease 2; lease 1 only constants | PROVED AT SCOPE for the five named groups after repair (independent LLM review by Codex) | Droid r12/r13 (replay of 24 witnesses on 16,128 initial states), Codex r14 |
| T3 perfect groups | For EVERY finite perfect group at lease >= 2 all group-valued functions are computable (perfectness suffices; simplicity not needed) | PROVED AT SCOPE (Codex's own result, reviewed by its author only) | Codex r14 |
| "Only the abelianization matters" for every finite group | FALSE: a central 7-input AND in D8 has trivial abelian projection and is not computable in this alphabet | REFUTED (counterexample) | Droid r13 |

| Perfect groups, lease 2 | A finite group admits EVERY group-valued table at lease 2 exactly when it is perfect (the stable-derived-subgroup criterion coincides with perfectness; no finite group separates them) | PROVED AT SCOPE (Codex proof; independent attack by Droid) | Droid r17, Codex r14 |
| Nilpotent-quotient reduction, stable-derived-subgroup criterion, abelian classification (Codex R13, R8, R11); unitriangular (fixed dimension, binary) and odd-order growing lease (R12, R18) | Reductions that classify lease classes through the nilpotent residual and the abelianization; closed forms for abelian, odd-order and binary unitriangular outputs | PROVED AT SCOPE (independent LLM attack by Droid; finite censuses; one 8-input run timed out, a separate exact calculation completed) | Droid r17 (`ROUNDTABLE/work/droid/r17_REVIEW.md`) |
| Unified lease classification for EVERY finite group (Codex r16): F computable iff each Boolean-face coefficient lies in a filtered subgroup K_d built from normal closures of twisted powers and commutators in G x G; leases 2s and 2s+1 equal | One criterion for all finite groups; exact function counts (e.g. D8 and Q8 at n = 3: 16,777,216; SL2(F3): 1,358,954,496; UT4(F2): 281,474,976,710,656) | PROVED AT SCOPE (proof audit by Droid r19, independent LLM review, no GAP/WRONG; 135/135 predicted-vs-enumerated class counts on 9 unlisted groups; formula breaks outside the single-borrower alphabet); not a novelty or advantage claim | Codex r16 (`ROUNDTABLE/work/codex/r16_THEORY.md`) |
| Order-16 higher-lease obstruction in C8 x| C2 and order-32 construction | New obstruction at lease four | CONJECTURE (not promoted) | Droid r17 |

| Lease as safety guarantee (Codex r18) | Observer-guarantee framing: uniform-time corruption fraction is not ell/T (sharp bound floor(ell*T/(ell+1))); pointwise window collapses (NOT b; NOT b padding, 3x gates); paid active-cleanup incumbent matches. Exact min-gate tables n=3..6 vs lease and clean-bit incumbent | FRAMING KILLED (EMPIRICAL finite searches, independent review pending); paid-cleanup metering OPEN (CONJECTURE) | Codex r18 (`ROUNDTABLE/r18_codex.md`) |
| Register-program frontier (Codex r22/r24; Droid r25-r27) | Passive dirty-work register programs: literal 3-register 5-call x^7 word and 4-call x^5 word verified independently (exact composition over Q, F_101, F_103); degree-5 ceiling for 4 calls and D(r,c)=2r+2c-9 (r in {3,4}, c in {4,5,6}) | Literal words PROVED AT SCOPE; ceiling and formula CONJECTURE/OPEN; ADVANTAGE KILLED (ordinary machine runs dirty words verbatim, initialised register does x^7 in 2 calls); twelve table entries are Vinciguerra 2026's results | `ROUNDTABLE/r24_codex.md`, `ROUNDTABLE/r27_droid.md` |
**So what / limits.** The lease bounds the algebraic degree of what a borrowed bit can touch, regardless of how many gates are spent; commutators escape it inside non-abelian output groups. Every witness is an ordinary
reversible circuit under identical resources, the capability side is the known Reed-Muller-to-Toffoli synthesis with Barenco/Gidney dirty-ancilla gadgets, and the machinery is Barrington-adjacent. The exact
"maximum consecutive disturbance" meter has no verified prior treatment, but absence of search results is not novelty. No new model of computation and no advantage is claimed.
