# METER-1: what survives changing the loss meter?

2026-10-03. PROVED AT SCOPE: L1, L2, V1 and V2 passed the fresh independent
actual Codex attack in METER_REVIEW.md. The original candidate derivations below
are retained as reviewed. No registered meter experiment has yet run; reviewer
probes are bounded EMPIRICAL evidence, not FULL admission. Registered finite
FULL is now admitted in METER_FULL_EVIDENCE_REVIEW.md. The growing-r critical
local work debt below is subsequently closed by MASKED_BLOCK_REVIEW.md;
physical calibration and new-paradigm standing remain OPEN.

## Same finite contract, three different resources

Clean AND, n>=3, original data prior max atom eta=1/8, independent uniform
pointwise-returned r-bit R, beta=2^-r. All registers/gates/program work charged,
fixed NOT/CNOT/Toffoli/one-bit-RESET schedule. Statistical preparation remains
an explicit supplied input resource, not a distribution oracle. Define:

J: full-joint entropy drop (the already reviewed meter).
L: entropy H(B_g) of the current TARGET BIT immediately before each RESET.
V: on a positive-probability predecessor class s, cost
log2(P(S_g=g(s))/P(S_(g-1)=s)); budget every branch and every admitted prior.
Reversible gates cost zero under each information meter; their gate work remains
charged. These are logical resources, not calibrated thermodynamic primitives.

## L: candidate exact feasibility and fixed-r efficiency cliff

Claim L1: for every n>=3,r>=0, the all-depth feasibility floor is h2(beta/8).
Claim L2: every feasible K<theta_r has optimal gate work Theta(2^(n+r)),
where theta_0=1 and theta_r=h2(beta) for r>=1. At K>=theta_r, work is Theta(n)
for fixed r, with a charged upper O(n*2^r+n+r). No growing-r optimal constant or
optimal Theta(n*2^r) critical law is claimed: the entropy lower only gives
Omega(n/theta_r+n+r), leaving a possible growing-r factor gap.

Lower proof candidate: for K<1, let a in [0,1/2) solve h2(a)=K. At a fixed
RESET the probability mu_p of target bit one is linear in p_X. Its possible
values form an interval because the capped simplex is convex. H(mu_p)<=K for
ALL priors means the entire interval lies in [0,a] or [1-a,1]. Choose that
gate's globally rare bit value accordingly. Its probability is always <=a.

If any original X value has that rare bit in any original R slice, assign it
probability eta (normalization is possible since M*eta>=1). The rare probability
is >=beta*eta. Hence K<h2(beta*eta) makes EVERY nontrivial RESET impossible;
exact noninjective AND cleanup cannot finish.

For K<theta_r, a<beta. Let d be the number of distinct original X labels having
the rare bit in at least one R slice. Assign as much admissible mass as possible
to those labels: rare probability >=beta*min(d*eta,1). Thus d*eta<1. Now assign
eta to ALL d marked X labels at once and fill the remaining mass outside them.
If N_rare counts marked original (X,R) labels, rare probability is at least
eta*beta*N_rare. Therefore N_rare<=floor(a/(beta*eta))=:b, with b<8.
Each RESET collision has a distinct rare predecessor class and hence at least
one distinct original rare label. No RESET eliminates more than b complete
classes. This gives G>=ceil(2^r*(2^n-2)/b), or impossibility when b=0.
The rare orientation can change from gate to gate; it never depends on the
realized prior. Previous coalescence and physical recoding of R cannot evade
the count. At r0 the same condition holds for every K<1.

Charged upper candidate: change each Gray merge to mark the INCOMING singleton,
merge it into the old accumulated representative, then reversibly move the
joined representative forward using the common predicate. Concretely, first
uncompute the old prefix suffix. Compute its unchanged lower-bit predicate;
and_match(predicate,changed_bit,want_new,flag); CNOT(flag,changed_bit);
RESET(flag); CNOT(predicate,changed_bit); uncompute lower predicate; rebuild
the new prefix. All dependent predicates are cleared before a changed control
is altered. The final CNOT advances both merged states; all other states match
neither representative. No predicate depending on the erased flag is retained.

Before RESET, flag one represents ONLY the new original singleton in one R
context. Its probability is <=beta*eta, so L<=h2(beta*eta), since beta*eta<=1/8.
This is finite constant overhead per existing Gray merge, not a free binary
aggregation primitive. Final zero-mass AND-one vertex and Y-one cleanup need
the same special handling as the reviewed compiler. Check that the incoming
singleton claim holds at EVERY merge after earlier coalescence. Width remains
O(n+r), work O(2^(n+r)+n+r). Actual elementary code/proof review is mandatory.

For K>=theta_r, the already reviewed full-context cleanup flags satisfy
P(flag=1)<=beta. For r>=1, h2 is increasing up to beta<=1/2, so L<=h2(beta).
For r0, every bit entropy is <=1. Input-targeting gives Omega(n+r).

Predictions at K3/4: one returned fair bit suffices for J-linear work, but L
requires exponential work at r1 and r2; r3 permits L-linear work. At the old
ladder point r2,K3/16, L is INFEASIBLE because h2(1/32)>3/16, whereas J is
feasible with exponential work. These comparisons require exact certificates.

## V: finite arbitrary-prior budgets can be impossible

Claim V1: n>=4, ANY finite r, no finite circuit has a finite universal V cap
over all max-atom1/8 priors, even with returned product-uniform R.

Proof candidate: for each finite rational epsilon in (0,1/8), assign X0 mass
epsilon, X1..7 mass1/8 each and X8 mass1/8-epsilon, all others zero. All nine
labels have AND zero for n>=4. For any exact circuit and any original R label,
initial class (X0,R) has mass beta*epsilon and final output (0,R) has mass beta.
Along this input's trajectory, the logarithmic branch ratios telescope to
log2(1/epsilon). If a fixed G-gate circuit had every branch cost <=K, then this
total would be <=G*K. Choose a finite dyadic epsilon smaller than 2^(-G*K).
This is a finite rational counterexample to that circuit, not an infinite
precision instruction or one distribution with infinitely many outcomes.

The quantifier is load-bearing: the source class has no positive lower atom
bound or prior-description/preparation budget. Every chosen prior still has
finite encoding and that preparation must be charged. Restricting that resource
or adding min-positive-atom changes this impossibility; do not transfer it to a
physical source with a fixed accuracy budget. This diagnoses omitted source
constraints, not hypercomputation or infinite power.

Claim V2, narrower uniform exception: n3's capped class is uniquely uniform.
At V cap one, every binary merge must have equal child masses. Complete-state
classes therefore retain power-of-two original leaf counts, while AND-zero
fibres have seven leaves. Exact cleanup is impossible at cap one, independently
of r. The singleton Gray schedule does admit cap log2(7); no exact optimal
uniform V threshold is claimed here.

## Gate and meaning

Fresh proof attack must independently validate the convex interval/rare-side
argument, simultaneous eta allocation, labelled-state counting, incoming macro
cleanup, exact numerical comparisons, telescoping with zero-probability states
excluded, and finite rational epsilon/preparation scope. No FULL computation is
authorized without separate frozen registration and code gate.

So what: the same gate program and statistical context can have substantially
different feasibility and work under joint, local marginal and worst-branch
logical meters. The basic context advantage may survive one meter and vanish
under another. This is a candidate audit result; it does not choose which meter
a physical system realizes or earn a new-paradigm/priority/completion claim.
