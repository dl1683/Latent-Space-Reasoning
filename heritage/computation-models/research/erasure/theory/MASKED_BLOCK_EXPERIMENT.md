# MASKED-BLOCK-1: registered larger CPU validation

2026-10-03. REGISTERED BEFORE IMPLEMENTATION/SMOKE/FULL. Binding mathematical
design/leak audit: MASKED_BLOCK_CANDIDATE.md; fresh proof review running.
No older source, registered range or meter is changed by this experiment.

## Machine, resources and fixed inventory

Clean AND n>=3; X max atom1/8 unknown to circuit; independent uniform pointwise-
returned r-bit R. Fixed NOT/CNOT/Toffoli/one-bit RESET and logical LOCAL meter
H(RESET target), caph2(2^-r). All gates, width, program generation and supplied
source/preparation charged. No probability oracle, fresh randomness, fractional
atoms or physical energy claim. Original X/R labels are retained only by the
EXTERNAL evidence evaluator, not the tested machine. Evaluator CPU resources
are explicit; they do not count as registers the circuit can read.

FULL all n3..16,r1..8:112circuits, plus n3,r9..12:fourhigh-r controls. Exactly
116circuits. A finite uint64 NumPy CPU interpreter is allowed to evaluate the
same explicit elementary words on all original labels in parallel. No GPU,
approximate probabilities or sampling. Maximum logical width60 (main ladder),
46 (high-r controls), fitting uint64 without truncation. Maximum simultaneous
original labels2^24, explicitly finite. State arrays and work/peak RAM/runtime
fingerprints are external-validation resources; source preparation remains
separate. Run sequential circuits; do not alter ranges after a failed case.

For each circuit evaluate EVERY original input through EVERY primitive gate.
Require legal arities/distinct bit operands, exact width2n+3r+4, clean final
state (AND(X),originalR), original X/R registerszero and allscratchzero.
Source word follows the reviewed chunk design; clear Yone data before masking,
mask chunk with returned lowR, incoming Gray conditional on highR and Yzero,
uncompute prefix BEFORE representative clearance, restore Yone using Y/R,
uncompute highcontext and sentinel. No hidden predicate or erased-bit copy.

At EVERY RESET count the original-R labels with flagone separately for EACH
original X. Require q_x=one for all Xexceptoriginalallones; q_allones=zero.
This is a stronger finite construction prediction than the proved cap bound,
and is not assumed as a free theorem. It implies at mostone originally marked
R per X, local probability<=2^-r for every unknown original p_X. Save its
lossless representation: qzero label[2^n-1], defaultqone; observed q histogram,
flag original-label total and hash of the complete q vector. Each exact local
supremum uses eight-smallest/largest capped-simplex LP before clamped binary
entropy, with an explicit attaining prior. For n3 the unique uniform prior
gives probability7/(8*2^r); for n>=4 a prior on eight ANDzero labels attains
2^-r. Store exact factored binary-entropy certificates, never decimal expansion
of huge integer powers or disabled conversion limits.

Require RESET count sumoverchunks[2^(r-d)*(2^d-1)], d=min(r,remaining chunk).
Gate bound G<=128*(n+r+2^r*ceil(n/r)); ceil(n/r)=one whenr>n. Record actual
gate kinds/gates/width/chunks and input-gate transition count (exact G*2^(n+r)).
The upperconstant is a sanity ceiling, not an optimality claim. Primitive
one-bit RESET has algebraic fan-in<=two; final all-label check certifies total
class decline2^r*(2^n-2). No enumeration of all intermediate class partitions
or enormous entropy-power products is required for this LOCAL protocol.

Compare registered old context-linear gate counts from its actual source word,
without changing that circuit or claiming every finite n/r case is faster.
For n=128,r8 generate both WORDS ONLY (no input execution). Require masked
word strictly smaller and record both G/width; this is an exact gate-count
comparison under the same local cap/source/output promises, not a measured
hardware speed or end-to-end bit-work comparison. Word-only width may exceed
uint64; it must not be executed by the fixed-width interpreter.

SMOKE pairs (n,r)=(3,1),(3,4),(4,2),(5,3),(6,2), fivecircuits covering r>n and
short last chunk. Full word-only comparison also included; SAME acceptance.
Independently cross-check every smoke input and every gate using a separate
Python integer scalar interpreter, exact state agreement at all boundaries.
No FULL until fresh actual proof/code gates PASS. A proof survives finite
failure only if the defect is demonstrably implementation/overstrong-control.

## Evidence and decision tokens

Capture source bytes once, execute those bytes, fingerprint launcher/helpers/
NumPy version/runtime/protocol. Record pre-run environment for smoke and FULL.
RUNNING invalidates previous positive output; firstfailure KILL/VOID with
witness and nonzero exit. Separate root tool-observed exit receipt binds
attempt/artifact/inventory/source/protocol/environment. Fresh actual independent
raw auditor reconstructs all116words and allinput gate/RESET flag/final checks,
LP certificate identities and n128r8 word-only comparison. External trusted
Python/NumPy arithmetic/observer, no malicious-host guarantee.

MB-CONFIRM: frozenfinitecontrols independently admitted; reviewed matching
theorems remain separately scoped. MB-PARTIAL: implementation/control fails
while surviving theorem clauses remain. MB-KILL: actual mathematical
counterexample to claimed lower/chargedupper. MB-VOID: changed promise/meter/
basis/range/source/inventory/completion; a cap or output failure stops FIRST.
No empirical proof of universality, practical device cooling, primary novelty
or completed mission follows. GPU staysunused.

So what: test a same-contract charged capability gain and the proposed
matching growing-context lower with a CPU sweep of billions of original-label
gate transitions; preserve exact finite resource/cost/source accounting.
