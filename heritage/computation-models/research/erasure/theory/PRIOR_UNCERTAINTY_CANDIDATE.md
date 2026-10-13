# Prior uncertainty can force exponential clean-computation work

Date: 2026-10-03. Rigor: PROVED AT SCOPE, independent lower/upper/code PASS
in PRIOR_UNCERTAINTY_REVIEW.md. FULL raw admission remains open.
So what: precise ignorance of the input law can change the elementary work of
the SAME clean Boolean task from linear to exponential at the same entropy cap.
This extends the live robust-prior frontier; no steering pivot or mission pass.

## 1. Contract and infinity-leak design audit

The NOT/CNOT/Toffoli/one-bit-RESET basis, fixed schedule, complete finite state,
initialized-zero distinct output/workspace, exact output and input cleanup are
the reviewed atomic contract. No stochastic transitions, source oracle, advice,
variable timing, uncharged predicates or wide gates. Every elementary gate is
one work unit; charge width and program-generation/storage bit work separately.

Task: n>=3 inputs X, output AND(X), every other bit zero. Prior-known version
uses uniform X. Prior-robust version uses ONE fixed circuit for every prior on
all 2^n input encodings with max atom <=1/8. Zero masses are permitted, but the
circuit must be correct on every input. In BOTH versions the complete-joint
entropy drop at every gate is <=K=3/4. The distinction is statistical promise,
not a faster-than-Turing claim or a measurement/physical-energy statement.

All constants are fixed rationals. Constructor is succinct, uniform and does
not read or discover an unknown prior. Infinite precision and unbounded advice
are absent. Comparison of different statistical promises is explicit: this is
a resource gap caused by uncertainty, not a same-promise simulation separation.

## 2. Candidate CANNOT: collision capacity under prior uncertainty

Consider complete-state classes of the FULL set of 2^n original inputs just
before a RESET. Reversible gates never change their number. A one-bit RESET
maps at most two distinct complete states into one, so every reduction is a
disjoint pair collision.

If a RESET has four such pairs, pick one original input from each of their eight
predecessor classes, give those inputs mass 1/8 each and every other input mass
zero. This is an admissible prior. Its eight distinct pre-reset states become
four distinct equal-probability successor states. Entropy changes from 3 to 2
bits: one full bit, violating K=3/4. This argument uses the circuit's actual
full state, including all flags, and is valid even after earlier erasures.

Therefore EVERY RESET in EVERY admitted robust circuit eliminates at most
three full-support classes. Initially there are 2^n; exact clean AND leaves
exactly two. Thus

    G >= number of RESETs >= ceil((2^n-2)/3).

Arbitrary extra initialized workspace, reversible recoding and longer depth
cannot evade the count. The prior is allowed to depend on the proposed circuit
and attacked gate: the universal-prior quantifier makes this a valid lower bound.

## 3. Candidate CAN and separation

The reviewed succinct boundary Gray constructor, with r=0, computes AND,
clears its tree, and absorbs each AND-zero singleton into one moving class.
At every actual merge the incoming singleton mass b<=1/8 and accumulated mass
a satisfies a+b<=1. Hence

    L(a,b) <= L(1-b,b) = h2(b) <= h2(1/8) <3/4.

The exact final inequality follows from 8^8/(7^7) <2^6 (equivalently 2^18<7^7).
The final zero-mass encoding move loses zero entropy; reversible gates lose
zero. The circuit is fixed and correct for every input and prior, not tuned to
that prior. Gray predicate amortization gives O(2^n+n) gates and width 2n+4;
program generation/storage bit work is additionally O(G*poly(n)).

Consequently the robust task has optimal elementary gate work Theta(2^n), with
linear width. In contrast, the independently approved succinct rational-cap
constructor gives Theta(n) gates at K=3/4 for uniform X. The exact uniform
feasibility threshold is below 3/4 for every n>=3. Thus the same function,
gate basis, full cleanup and numerical cap have an exponential uncertainty gap.

More generally if max atom<=eta, eta>=1/M, and t=ceil(1/(2eta)), a gate that
eliminates t pairs admits the uniform prior on 2t selected original atoms and
loses one bit. For K<1 this forces at most t-1 collisions, when 2t<=M.
The fixed eta=1/8 theorem needs no rounding or exceptional feasibility case.
At K=1 the upper construction computes AND and selectively resets each input
bit in O(n) charged gates for every prior. Thus robust work may have a sharp
cliff at the full-bit cap. This generalized/cliff claim also requires fresh review.

## 4. Registered CPU falsification PRIOR-UNCERTAINTY-1

Registered before implementation or execution. Code uses the fixed reviewed
succinct circuits; the experiment itself does not establish the universal theorem.

FULL: n=3..10, robust boundary constructor and uniform K=3/4 constructor.
Enumerate every initial input through each charged gate; reconstruct complete
state classes, RESET pair-collision counts, terminal state and total eliminated
classes. For each gate with >=4 collisions save eight original-input witnesses
and independently simulate their exact 1/8 prior, requiring entropy-drop ratio
2^(8*DeltaH)=256. The robust Gray constructor must have <=1 collision per RESET,
correct clean output and linear width, with G<=64*(2^n+n+1). Uniform constructor
must have correct clean output, cap compliance under uniform input, and
G<=512*(n+1+4n/3). Its robust admissibility is not presumed: record every witness.

For n=3,4 exhaust ALL eight-input uniform-subset priors (1+12,870 distributions)
through the robust constructor, checking exact peak ratios <=2^6=64, pointwise
correctness and cleanup. Smoke n=3,4; exhaustive prior sweep only n=3 and the
first 32 lexicographic eight-subsets for n=4. No random seeds or sampling claim.

CONFIRM iff all frozen checks pass, witnesses verify and all n>=4 uniform
constructors exhibit a forbidden four-pair RESET; KILL on a false witness,
robust counterexample, wrong output, dirty flag, cap/count/width violation;
VOID on changed scope/protocol/arithmetic. Persist first failure and current
attempt; never admit stale positives. Run FULL only after fresh code/proof
gate, then independent raw evidence gate. Source dependencies/hashes frozen in
pre-run metadata, both modules guarded against optimized assertion execution.

## 5. Narrative and occupancy attack

HEURISTIC narrative: knowing the input law may save exponentially much clean
computation under a strict information-loss cap. The strongest triviality attack
is that universal support counting is standard and the cap is a chosen logical
resource, not a physical throughput law. This is more than the telescoping
entropy lower bound, but novelty and useful-world value remain open. Compare
energy-aware algorithms, distribution-mismatch computing, support-sensitive
conditional reversibility and robust irreversibility bounds in original sources.
An incomplete search is not priority. No completed new paradigm is asserted.

Implementation note, 2026-10-03: smoke passed four complete circuits, 33 bounded
priors and an explicit eight-input witness against the n=4 uniform fast circuit.
This remains EMPIRICAL smoke, not universal proof or FULL admission. Publication
uses the shared snapshot/IO repair from the succinct code gate. Admitted command
is `python -B experiments/run_snapshot.py prior_uncertainty --output ...`; a
separate observed-successful-exit receipt is required. The fresh proof/code gate
PASSED; FULL returned CONFIRM for 16 circuits, 12,871 bounded-prior controls
and 55 adverse witnesses in 6.96 CPU wall seconds. Exact pre-run metadata and
successful-exit receipt are saved; independent raw admission is still required.
This launch repair changes no frozen range,
predicate, gate, statistical contract or numerical criterion.

New hardest objection from genuine steering: a trusted independent fair bit,
returned pointwise, may permit linear robust cleanup by conditioning each reset
on one of its two values. That changes the statistical resource/contract and
does not refute the no-seed lower bound. It may severely limit the gap's practical
interpretation; no physical bottleneck or same-promise speedup follows. This
observation is a CANDIDATE and is not yet a newly admitted theorem here.
