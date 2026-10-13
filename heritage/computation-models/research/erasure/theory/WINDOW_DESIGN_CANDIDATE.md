# Finite-window work and total cost under fixed rational reset designs

2026-10-03 PROVED AT SCOPE: fresh actual LOWER/charged UPPER/tradeoff/window
PASS in WINDOW_DESIGN_REVIEW.md after the notation and padding-scope repairs
in WINDOW_DESIGN_REVIEW_1.md. Original candidate text below is provenance.
Direction converged through actual THREE rounds, STEERING_OCTOBER_2_DECISION.
No new implementation or FULL yet; operational binding remains a separate gate.
No novelty, physical heat, realized-capacity or mission-completion claim.

## Finite contract and design/leak audit

Clean AND n>=r>=8, pointwise-returned r-bit R, capped X marginal max atom1/8.
P contains every capped p_X independent of uniform R. C contains every joint
law with those SAME marginals, allowing arbitrary correlation. Every finite
exact circuit uses NOT/CNOT/Toffoli/one-bit RESET with charged width, gate work,
generation/serialization, source preparation and rational description lengths.
Only source promises differ. No real oracle, fractional atom, free random gate,
unbounded advice, hidden inverse or infinite running time.

Choose a rational design q_g in (0,1) BEFORE the unknown workload for each
RESET; its target-one probability under actual input law is mu_g. Logical
fixed-design cost is

    c_g=C(mu_g,q_g)=-mu_g log2(q_g)-(1-mu_g)log2(1-q_g).

This equals h2(mu_g)+D_Ber(mu_g||q_g), so c_g>=h2(mu_g)>=J_g. Reversible gates
have zero cost under THIS scalar meter and remain charged as work. That zero
does not describe an actual finite-time physical implementation. Designs are
finite rational data; logarithms are mathematical costs, not infinite-precision
machine primitives. Scalar positivity excludes free q=0/1 singularities.

Set beta=2^-r, theta=h2(beta). Extend c_g=0 outside the finite schedule1..G.
Require for every admitted law and every integer t:

    c_t+c_(t+1) <=2theta.

Thus first/last/one-gate schedules are measured. Any boundary waiting needed
by a device must be explicitly charged in that different physical contract.
Define E_q=sum_g c_g. No gate can escape by shortening the schedule. Costs
are ensemble expected scalar costs, not per-realization peaks or thermal power.

## Candidate product CAN and matching elementary-work lower

Use existing masked AND compiler with fixed design q_g=beta at EVERY RESET.
Each mu_g<=beta for every P law; since beta<1/2, C(mu,beta) is increasing in
mu, hence c_g<=h2(beta). The stronger single-gate bound implies window2cap.
Same finite word and linear width attain

    G=O(n+r+n*2^r/r),
    E_q<=ceil(n/r)*2^r*h2(beta)<=ceil(n/r)*(r+2)=O(n+r).

No workload-adaptive q is used. All q descriptions have r+O(1) bits each,
or one shared r-bit parameter plus finite references; serialize and charge
the actual choice. Program generation and any shared-parameter representation
are additional finite resources, not an end-to-end speed theorem.

For every admissible circuit/design, uniform joint input has entropy loss
D=n-h2(2^-n). Summing disjoint two-gate windows (using zero extension for the
last gate) yields E_q<=2theta*ceil(G/2)<=theta*(G+1). Since E_q>=D,
G>=D/theta-1=Omega(n*2^r/r). Targeting/return work gives Omega(n+r).
Because n>=r, the upper matches uniformly:

    G_P^*=Theta(n+r+n*2^r/r).

Uniform input also gives E_q>=D=Omega(n+r) in this regime. Thus product total
Theta(n+r) and optimal elementary work are achieved by ONE fixed-design word.

## Candidate correlated CANNOT and matching upper

Each gate lies in a measured window, giving h2(mu_g)<=c_g<=2theta. Forr>=8,

    h2(3beta)>2h2(beta).

One elementary certificate uses h2(3beta)>=3beta(r-log2(3)),
h2(beta)<=beta(r+log2(e)) and log2(3)<8/5,log2(e)<3/2 to obtain
r-3log2(3)-2log2(e)>r-39/5>=1/5 at r8. All are analytic inequalities;
their strict rational comparisons can be certified independently.

The convex C source polytope and separated entropy tails therefore force one
globally rare target value with probability strictly below3beta. If THREE
distinct original R columns contain any rare labels, assign beta to selected
cells. Even a single X row containing all three respects3beta<=1/8. Fill
remaining columns from row capacities: totalcapacity2^n/8>=1, finite dyadic
units suffice. An admitted law then has rare probability>=3beta, contradiction.
At most TWO original R columns are active at each RESET.

Pointwise returned R prevents intermediate classes mixing original R labels.
On uniform joint input, at most two slices lose at most one bit each, so
J_g<=2beta. Telescoping gives G>=D/(2beta)=Omega(n*2^r).
Existing context-linear circuit with q_g=beta has mu_g<=beta for all C laws,
so c_g<=theta and fits the window. Work O(n*2^r+n+r), matching this lower.

## Candidate correlated total cost at fixed work allowance

For any C-admissible circuit with G<=C0*n*2^r, fixed C0>=1, let T be RESET
count and a_(g,1),a_(g,2) conditional rare probabilities in the two active
slices under uniform input, padding missing slices with zero. Then

    lambda_g=beta*(a_(g,1)+a_(g,2)) = Pr(globally rare target value),
    mu_g in {lambda_g,1-lambda_g},
    J_g<=beta*[h2(a_(g,1))+h2(a_(g,2))],
    sum_(g,j) h2(a_(g,j))>=D/beta,
    T>=D/(2beta), T<=C0*n/beta.

By Jensen, the mean of the 2T values a is at least
alpha_C0=h2^{-1}(1/(4C0))>zero on the lower half. Since lambda_g<=2beta,
h2(mu_g)=h2(lambda_g)>=(r-1)*lambda_g. Therefore

    E_q>=sum h2(mu_g)
       >=(r-1)*beta*sum_(g,j)a_(g,j)
       >=(r-1)*beta*2T*alpha_C0
       >=(r-1)*alpha_C0*D=Omega_C0(n*r).

Context-linear upper has n*2^r RESETs and E_q<=n(r+2), so this order is
attained with a sufficiently large fixed work allowance. No existence under
every prescribed C0 is implied, and arbitrarily slow programs are outside
this total-cost lower. Design precision/implementation errors must be charged
in any approximation extension; T times a per-gate error cannot be ignored.

## Candidate general window reduction

For the stated fixed-design crossentropy meter, zero-extended w-gate windows
give W1(wK;b)<=Ww(K;b), where b is the width allowance in this paragraph.
Padding must preserve the costs of original gates; crossentropy satisfies this
because probabilities and fixed designs do not change. Nonnegativity and zero-
cost identities alone do not suffice for an arbitrary history-dependent meter.
Conversely
insert w-1 reachable-state identity gates between every original gate of a
W1(wK) word. A CNOT between two additional initialized-zero scratch bits is
an explicit identity on its reachable states; it is still charged as work.
Every window contains at most one original gate. Thus Ww(K;b+2)<=w*W1(wK;b),
with two extra clean scratch bits and finite generation/description charged. The
unindexed sandwich only applies when extra scratch is allowed separately. Device
boundary idle slots, duration or positive residual costs change this contract.
The unit-reset crossover wK>=1 permits ordinary RESET padding, but says nothing
about other unmasked circuits below that crossover: Gray cleanup may remain
feasible above its own minimum. No universal unmasked exclusion is asserted.

So what: the combined work/accumulated-cost distinction is proposed to survive
a two-gate averaging window and workload-independent rational reset designs.
This tests a specific operational objection, without equating scalar log cost
to a finite-time heat experiment. Statistical source authentication, residual
reversible costs, actual gate duration and a useful primary delta remain OPEN.

Fresh actual independent attack must check both source contracts, entropy
inequality including r8, boundary windows, design quantifiers, per-gate entropy
tail, three-column transportation including same-X, original-R recovery after
RESET, Jensen/zero cases/work-allowance constants and simultaneous upper.
Return product LOWER/charged UPPER, correlated LOWER/charged UPPER and total
tradeoff PASS/REPAIR/KILL separately; no edits/agents/GPU or registered FULL.
