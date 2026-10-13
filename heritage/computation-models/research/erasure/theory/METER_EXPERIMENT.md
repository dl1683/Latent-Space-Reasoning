# METER-1: registered finite CPU falsification

2026-10-03. REGISTERED BEFORE IMPLEMENTATION, SMOKE OR FULL. Theory authority:
METER_REVIEW.md passes L1/L2/V1/V2 at scope. Steering priority is
STEERING_RESTART_DECISION.md. This protocol changes no older experiment.

## Finite contract and inventory

Same clean AND, n>=3, fixed NOT/CNOT/Toffoli/one-bit RESET, all gates and
registers charged. Original X has unknown max atom1/8; R is independently
uniform and returned pointwise. Local meter L=H(target bit), joint J=entropy
drop of COMPLETE state, branch V=log2(postclass mass/preclass mass), counting
only positive predecessors. These are logical cost meters, not device heat.
Finite exact integers/Fractions; CPU only; no prior oracle, free predicates,
infinite precision, fresh entropy or GPU. Source preparation and program size
remain separately charged; the dyadic family has explicit finite description.

FULL n3..8,r0..3 and THREE charged programs at each point: existing accumulated
singleton Gray; new incoming-singleton Gray; existing context-linear cleanup.
Exactly72 circuits. Evaluate EVERY original input through EVERY gate, validate
width2n+3r+4, gate arity/distinct operands, clean output/pointwise R, full class
decline 2^r*(2^n-2), each RESET fan-in<=2 and one RESET per singleton edge.
Gray gates <=64*(2^(n+r)+n+r+1); context-linear gates <=64*(n*2^r+n+r+2^r+1).
Report full elementary gate counts; no generation/time optimality assertion.

At each RESET independently group ORIGINAL labels by X and count how many
original R labels have target bit one: q_x in {0,...,2^r}. For the max-atom1/8
simplex, min/max target probability are the sums of the eight smallest/largest
q_x divided by 8*2^r. This is an exact linear optimization, not vertex sampling
of a concave entropy. Worst entropy is at mu*=clamp(1/2,[mu_min,mu_max]). Save
the two eight-atom endpoint priors and their rational mixture attaining mu*.
Verify its normalization, max atom, probability and exact binary-entropy ratio.
If an interval straddles1/2, its interior mixture is load-bearing.

Incoming Gray must have exact universal L peak h2(2^-r/8), with each genuine
flag marking ONE ORIGINAL (X,R) label; final empty-vertex flag marks none.
Existing Gray must attain L=one for n3,r0, while its final empty-vertex RESET
has J=zero and positive L: this rejects interpreting L as J. Under uniform
joint input, old/incoming programs must have identical ordered J RESET ratios
and identical post-macro states despite different flag orientations.

Context-linear must satisfy L<=h2(2^-r) for r>=1 and <=one for r0, and J<=2^-r.
For n>=4 require its exact universal L peak h2(2^-r) (r0:one). Do NOT require
this equality at n3: its capped prior is uniquely uniform, so finite-instance
peaks can be lower than the asymptotic work cliff. Exact finite controls cannot
prove universal optimality or an asymptotic lower bound.

At EVERY RESET record uniform J exact factored entropy-ratio certificate and
maximum positive-branch ratio. Neither repeated atom expansion nor entropy
products with astronomical exponents are used for small-atom weighted tests.

## Finite dyadic branch attacks

For all n4..8,r0..3 and all three programs: N in {4,8,16,32,64,128}, W=2^N,
p0=1/W, p1..7=1/8, p8=1/8-1/W, otherszero. Exactly360 circuit/prior traces.
Use integer masses [1,W/8 seven times,W/8-1], multiplied by uniform R by using
joint denominator W*2^r. Cache deterministic RESET label maps from complete
gate evaluation; calculate exact Counter class masses, branch ratios and the
X0 trajectory telescope. Require final clean (0,R), telescope W, and maximum
branch ratio>=W/8. The last inequality follows because the first positive
merge of the epsilon class meets mass at least1/8-epsilon; it is a finite
registered prediction, not yet an independently approved additional theorem.
Save all maxima/telescopes and explicit finite preparation bit-length N.
These six source precisions illustrate V1; they do NOT prove its unbounded
universal quantifier. For n3 verify old Gray admits cap log2(7) and fails capone.

Exact arithmetic controls: h2(1/32)>3/16, h2(1/4)>3/4, h2(1/8)<3/4, and
one interior-mixture entropy maximum exceeding both endpoint entropies.
Smoke n3,4 r0,1 allthree programs (12 circuits), N4,8 where n4 (12 traces),
same logical controls. No full-width or family range changes after registration.

## Gate, artifacts and decision tokens

Smoke before fresh actual zero-context code gate. FULL only after code PASS.
Exact-byte snapshot execution; RUNNING invalidates old success; first failure
KILL/VOID with witness, nonzero exit. Source/runtime fingerprints and pre-run
environment recorded; separate root tool-observed exit receipt mandatory.
Fresh actual independent raw audit must reconstruct circuit/LP/branch outcomes.

METER-CONFIRM: registered finite controls pass and are admitted; reviewed
theorems retain their separate scopes. METER-PARTIAL: a construction/control
fails but surviving theorem clauses remain. METER-KILL: an actual mathematical
counterexample refutes a reviewed clause; an implementation bug alone does not.
METER-VOID: promise/meter/basis/ranges/source/inventory differs or completion
receipt fails. No continuous joint-supremum, physical or novelty admission.

So what: put exact feasibility, charged work and failure witnesses under
three cost meters, exposing which statistical-context advantages survive a
change of accounting. Physical calibration and useful-world advantage remain
open research debts, not experiment acceptance conditions silently omitted.
