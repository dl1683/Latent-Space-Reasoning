# The translation wall, version 1 (droid, 2026-10-08)

**CONJECTURE (author-side proof, independently finite-checked; NOT PROVED
under this repository's independent-review rule).** This is a containment lemma,
not an impossibility theorem for all new models.

Fix an input length n, resource budget S and program-description budget
p. Let M be a machine with
configurations (input, finite control, mutable registers, oracle state).
Let U be an unrestricted ordinary machine. The task includes the entire
input-output (or interactive transcript) distribution, and cost is an
explicit vector (time, live space, reads, communication, error, etc.).
Assume:

H1. Equal initial information: same input, initial mutable contents,
advice, **joint** randomness law (including any correlations with the
adversary and oracle) and external-oracle access; no hidden table or
preinstalled x-dependent state on M's side. The encoded source
configuration (including oracle history and reusable scratch) fits in
the charged h-space representation; any random sampling/precision cost
is included. Correlated future random bits cannot silently enter at
source instruction boundaries.

H2. **Uniform step lowering.** For each source program of description
size <=p and source instruction a, a compiler generates **one** ordinary
macro C(P,a), independent of the unknown runtime configuration, which
works for *every* reachable source state. It takes at most f(n,S,p)
charged target steps on every such state, operating on the very same
logical registers or an explicitly charged representation, with no more
than h(n,S,p) simultaneous live memory. The macro's endpoint and externally
visible trace have the same law conditional on the pre-state. It cannot
call a stronger oracle or privately read an address the target cannot.
Program generation and initialization together take at most g(n,S,p)
steps, uniformly over this program class. Equivalently, an online
interpreter can pay bounded lookup inside f rather than expand P first.
In either implementation, runtime instruction fetch, branch selection
and dispatch from the encoded program counter are included in f;
the next instruction may depend on the runtime state or public
transcript. Pre-expanding only one guessed execution path does not
meet this hypothesis.
The weaker quantifier `for every state, there exists a short macro`
is insufficient: choosing that macro may require the hidden answer.
The endpoint must reestablish one invariant representation relation
for all later instructions; an implementation that leaves untracked
garbage cannot be composed.

H3. An interface-preserving schedule composes these macros under the
same observations and future inputs, including an adaptive adversary;
if source randomness is used, the target draws from the *same allowed*
distribution. Atomic source steps cannot interleave with additional
adversarial interventions in the target macros, or the contract explicitly
allows that stuttering. If wall-clock timestamp/deadline is itself
observable, hiding f-1 substeps is NOT legal; either preserve the timed
observable exactly or state only a time-rescaled, different contract.

H4. Uniform and honest accounting: both include all setup, message
directions, precision, oracle calls and state restoration. Target's
feasible class includes the simulator with space h; cost comparison is
against the *optimum unrestricted equal-access target*, not a particular
baseline algorithm. If a source instruction is literally a target
instruction, f=1 and g=0 (modulo identical loop/controller accounting).

**Simulation lemma.** For each admitted source program P with
worst-case time T and peak space S,
there is a target program Q on the identical task with time at most
g(n,S,p)+f(n,S,p)T, space at most h(n,S,p), and the same complete output or
interaction distribution. Consequently
OPT_U(n,h) <= g+f OPT_M(n,S,p) on any task where the respective optima
exist for the fixed p class.
Equivalently, with the same input promise and error contract,
`TASKS(M,T,S,p) ⊆ TASKS(U,g+f*T,h,compiled_description_budget)`;
the compiled program length itself must be paid if program bits are
part of the task's resource vector.
If T is a stopping time rather than a worst-case bound, the same
pathwise inequality yields `E[T_U] <= g+f E[T_M]` when expectations
exist; it does not turn an expected bound into a worst-case bound.
For each cost component a separate component-wise simulation inequality
is needed; a bound on wall-clock time alone says nothing about probes or
communication. If f=polylog(n), g=polylog(n), and h<=S under the target
budget, M cannot have a super-polylog-factor **asymptotic speed advantage**
over unrestricted equal-access U when the setup is negligible relative
to source work. If f=1,g=0,h<=S, M never wins in the time metric.
The inverse inequality is NOT implied. A target *time* lower bound
`OPT_U>=L` on the same task and resource budget does transfer
backward to `OPT_M>=max(0,(L-g)/f)` for f>0; a source-only
impossibility need not constrain the stronger target. This yields
no new-model advantage by itself. Also, if g dominates T, the
claimed multiplicative overhead is false: retain the additive term.
More generally, for nonnegative source cost vectors c(step) and
target macro vectors d(step), if a fixed nonnegative matrix A
satisfies `d(step) <= A*c(step)` componentwise on every reachable
state, summation gives `cost_U(Q) <= setup_vector + A*cost_M(P)`.
An off-diagonal term, for example source time charged into target
communication, is *not* a same-coordinate simulation bound. Space
uses a peak condition, not the sum of per-step space vectors.

**Approximate-kernel extension (CONJECTURE, author-side proof).** If source and uniform
macro conditional public/state transition kernels differ by at most
epsilon in total variation at *every* common reachable history, and
their **joint machine/environment** initial laws differ by delta,
then the length-T joint transcript
laws differ by at most delta+T*epsilon (capped at 1). Couple
stepwise while histories match; a first mismatch occurs initially
with probability <=delta or at any subsequent step with conditional
probability <=epsilon, and union-bound. Hence a success probability
falls by at most that amount. This does NOT preserve exact/zero-error
computation. Accuracy epsilon=O(1/T) may make the macro overhead
f(n,S,p,epsilon) grow and its precision/state-preparation bill must
be included. `r23_approx.py` checks small rational examples; it is
not an independent reviewer or a new inequality.

**Strong dominance corollary (CONJECTURE, author-side proof).** If target may execute each
source step *literally* with equal cost and the complete task, timeline,
visible information and score are the same, inclusion of strategy
classes gives `OPT_U <= OPT_M` for **any** minimization objective,
including communication, online regret, streaming memory and proof
traffic. Changing the score alone does *not* escape inclusion. Only
an unequal interface, an unequal measured trace, or a nonconstant
simulation overhead relative to the newly chosen objective can reopen
the question. Consequently the "escape routes" are necessary
departures from a particular set of hypotheses, not sufficient routes
to a useful advantage.

**Proof sketch.** Construct Q by generating the source description and
replacing each instruction with its endpoint-and-trace preserving macro.
Induct on the **joint** law of represented machine state, any private
adversary/oracle state and the complete public interaction prefix.
H1 supplies the base coupling, H2 and H3 the induction, H4 the summation
and feasibility. For stochastic steps use the equality of conditional
transition kernels and the chain rule; for interactive steps use
the same adversary transition law given its private state and the
public prefix. Add setup g once. This is simulation, not
a lower bound for M and not a Turing-versus-quantum theorem.

**Independent proof check, distinct derivation.** For a finite stochastic
machine write the kernel of each source operation as a row-stochastic
matrix K_i on full states augmented by its *public* emitted symbol.
Uniform lowering asserts its ordinary macro has the same K_i after
hidden substates are marginalized. The kernel for a fixed word
i_1,...,i_T is the product K_{i_1}...K_{i_T}, so equal factor kernels
give equal product kernels and equal public transcripts. Conditional
selection of the next i from the common public transcript preserves
this equality by induction on transcript length. The sum of macro
lengths is at most fT; prep adds g; the live-space condition is checked
at *intermediate* target states, not only endpoints. This checks the
coupling argument without using the same pathwise proof, but is still
my own reasoning, **not an independent reviewer**. In addition,
`r23_independent.py` enumerates two-bit
configurations, two (separately chosen) source step families and all words
of length up to 5, exhaustively compares endpoint maps and mixed trace
lengths against a separate macro-composition implementation, including
empty programs and unequal macro lengths. It does not constitute a fresh
human/agent proof review; the claim remains CONJECTURE.

**Exact failure modes / map.**
1. H1 fails if an indexed data page is free only for M. An ordinary
   controller with identical access can call it; without it, x_i needs
   communication or encoded storage (separate model contract).
2. H2 fails for a truly stronger oracle or charged bulk/quantum operation
   with no polylog-sized ordinary macro. Literal unbounded-precision
   arithmetic at unit cost is an **infinity leak** without a representation
   and preparation bill. A claim that such a macro requires super-polylog
   work needs a *proved lower bound* on the target lacking it:
   a bulk SETALL-1 of n independent writable bits requires at least
   n single-bit writes from 0^n, by Hamming distance. That does not
   make SETALL-1 a new model: n output drivers, area and energy may
   be hiding inside its "unit" price, while an equal-bulk-access
   ordinary target uses the same step.
3. H3 fails if an adversary can act in the middle of the simulation or if
   the source atom hides a serial causal-order schedule. Even without
   intervention, an external deadline may distinguish a source
   single-step event from the macro's delayed event.
4. H4 fails for online regret, proof size, communication rounds, sample
   complexity or memory traffic when the compared target is optimized for
   another objective. A reweighted cost can be new theory but not by
   itself a win against U under U's original cost. A physical energy
   claim needs an operational meter and calibrated substrate on both
   sides; a logical entropy drop is not automatically heat.
5. H2/H4 fail when target space is capped at S but h>S, or when advice,
   precomputation, precision, error rate, reversible garbage, cleanup
   and randomness are not charged identically.
6. "Same register contents" is not "same access": a random-address
   word probe against a serial tape head can require moving across
   n cells. The simulator then has f=Omega(n) for some layouts,
   unless both sides receive the same indexed-memory primitive
   and pay its width/address/preparation costs.

**Back-test against all sixteen r20/r21 fresh dials (HEURISTIC
classification, not sixteen newly proved statements):**

| Prior cases | Where apparent separation came from | Equal-contract resolution |
|---|---|---|
| r21 D1-D5 pair XOR/popcount/AND/equality, triple majority | H2 sensor missing from the bit-only comparator; or H4 its input footprint unpriced | An ordinary controller given the very same sensor copies the adaptive tree step for step; bit-only footprint emulation costs 2 or 3 per sensor. |
| r21 D6 phase oracle and D7 quantum random-access bit | H2 changes coherent query and quantum message alphabet; H4 changes bounded-error vs exact | The fair comparator is an equal-quantum-access machine; Bernstein–Vazirani and quantum random-access codes already express the results. |
| r21 D8 positive OR witness | H4 changes the quantifier from exact two-sided output to a positive certificate | A certificate verifier is not a worst-case decider; same-success ordinary verifier exists. |
| r21 D9 parallel parity | H2/H4 omit processors, wires and area in serial baseline | Equal-resource circuits execute the same XOR tree at the same depth. |
| r21 D10 indexed advice | H1/H4 omit program and preparation/table bits | An ordinary indexed table with same data performs the same read. |
| r20 D1 symmetry marker | H1 hides a chosen reference frame | Count the marker and ordinary relative indexing matches. |
| r20 D2 coherent oracle order and D3 nonlocal boxes | H2/H3 genuinely alters oracle/causal access | Existing quantum-switch and nonlocal-box models, rather than a false central-local comparison. |
| r20 D4 postselection | H4 ignores success probability | Retry mass 1/p is paid, unbounded postselection is not a free atom. |
| r20 D5 causal consistency | H1/H2 hides global state size and stationary-cycle computation | Ordinary finite graph computation incurs work exponential in register bits. |
| r20 D6 contextuality | H4 demands a joint global assignment only of one comparator | Context-aware ordinary sampler matches marginal law; joint and marginal tasks differ. |

Every line is scoped to the tested contract; for example the quantum
switch can have a real oracle-call separation under an opaque
higher-order oracle, but the r20 contract is already occupied. The
classification does not assert that all future quantum or physical
models are efficiently classically simulable.

**So what:** This tells us precisely which separations cannot come from
restricting old instructions, while leaving real open territory only where
the *task, resource or access* differs under a paid, symmetric interface.
**Triviality attack:** this is the usual compiler/simulation argument,
not a new theorem. The repo's `ROUNDTABLE/work/droid/r15_TRANSLATION.md`
already states the equal-access distributed trace simulation; its
explicit warning about centralized privacy/timing is consistent with H3.
Its use here is as a falsifiable filter, not a research win. An
algorithm remaining useful after a competitor copies it is perfectly
possible; the lemma excludes only a claimed *class advantage* against
the unrestricted, equally informed target, not adoption or practical
engineering benefit.
