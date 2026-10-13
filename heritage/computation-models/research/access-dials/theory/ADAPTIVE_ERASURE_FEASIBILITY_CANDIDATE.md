# Exact feasibility control for adaptive erasure testing

2026-10-03 CANDIDATE-PROOF, tactical design control, not chosen mission direction.
Primary priority OPEN; exact private-seed sampler upper remains separate.

Integers N>=1,1<=M<=N,t>=1,q>=0; a fixed original Boolean input has exactly M
ones or, for the monotone extension, at least M. READ(i) returns the original bit
or bottom if that address was erased. Erased locations are otherwise HIDDEN.
After EVERY answered READ, including a revisit or bottom, an omniscient adversary
may permanently erase up to t further cells. Local computation does not earn
new erasure opportunities. The first READ is intact. Worst-case READ count<=q.
Public transcript includes addresses, replies and stopping; private local timing
is excluded. Independent initial uniform r-bit seed and private state are hidden
from the adversary, which knows the original input, program and public history.
One-sided acceptance of all-zero under every admitted adversary. Seed preparation,
generation, storage, addresses, local bit work and input loading all charged.
Loading supplies the oracle without revealing contents to the tester. No GPU,
input-dependent advice, unlimited source scan or published-new-paradigm claim.

For j=0,...,q-1 define a_j=max(0,M-t*j), stop when a_j=0. Proposed maximum
success against worst-case input and eraser, for an ideal rational-randomness
benchmark, is

    V(N,M,t,q)=1-product_j (1-a_j/(N-j)),

with q capped at N and factors only while a_j>0. This is a finite rational,
not a gate oracle. No unbounded/random-real seed is supplied to actual machines.

CAN route: sample a fresh uniformly chosen unqueried address each read. Conditional
on every past, at least a_j original ones remain un-erased; denominator N-j.
Adversary erasing t still-unseen original ones after each miss attains equality
for this sampler until no live defects remain. Erasing zeros or fewer ones cannot
improve this eraser's miss probability. Stop immediately on an intact one.

CANNOT: choose a uniform original M-subset and independent uniform ordering of
its elements. After each miss erase the next t live elements. This is a finite
mixture of FIXED original inputs and deterministic legal erasure schedules.
Fix the tester's entire seed. After s misses and d_s<=s distinct queries, exactly
a_s=max(0,M-t*s) live ones remain. Conditional on the actual history, this set
is uniform among N-d_s unseen addresses: permutations fixing queried addresses
preserve input distribution, deletion rule and that complete history. Zero and
bottom histories give different posterior constraints, both symmetric on unseen
addresses. A revisit cannot hit a live one. A fresh READ has hazard
a_s/(N-d_s)<=a_s/(N-s) until all live ones disappear by ceil(M/t)<=N misses.
Multiply miss inequalities; early stopping and READs beyond N cannot improve.
Mirror each no-one transcript on all-zero with identical erased addresses/timing,
a legal schedule. One-sidedness forbids rejection of that transcript. Averaging
the hard mixture gives a fixed original input and deterministic eraser witnessing
failure>=F=1-V after averaging the tester seed. This covers all adaptive testers.
At least M extension: V increases with the original defect count, and exact-M
inputs remain admitted, so the same benchmark is optimal for that promise.

Exact uniform without-replacement sampling may require non-dyadic coins. For a
fixed input/deterministic eraser, any uniform-r-bit tester's failure is a multiple
of 2^-r. Thus every such machine has worst failure >=ceil(2^r*F)/2^r. A non-dyadic
F cannot be attained at ANY finite r; dyadic F alone does not guarantee attainment.

Charged finite approximation: encoded rational0<eta<1; h=min(q,N,ceil(M/t)).
If h=0 accept without randomness. Otherwise L=ceil(log2(h/eta)), computed by
finite integer comparisons. At round j<h reserve a disjoint block of L*b_j bits,
b_j=ceil(log2(N-j)). Interpret up to L consecutive b_j-bit proposals; take first
below N-j. If none valid, ACCEPT and halt. A successful rank is exactly uniform;
abort probability rho_j=(1-(N-j)/2^b_j)^L<=2^-L. N-j=1 uses zero bits/rank0.
Couple with ideal sampler against the same causal eraser until first abort.
Disjoint private seed blocks preserve future independence: r=L*sum b_j,
abort<=h*2^-L<=eta, worst failure<=F+eta. Reject only on an intact one.
Use lazy Fisher-Yates with a linear list of <=h overrides to select the ranked
unqueried address, avoiding any uncharged N-entry setup. Let b=ceil(log2(N+1)),
B=total encoded parameter length including eta. Conservative multitape bit work
O(B^2+r+h*L*(b+log2(L+1))+h^2*b), storage O(B+r+h*b+log2(L+1)). Input-oracle
storage/loading additional shared charges. No seed-compression frontier follows.

Controls: N16,M4,t1,q>=4 has failure216/455>1/3, hence error1/3 impossible
even with more randomness. N8,M4,t1,q4 has F=16/105; eta1/12 gives r72 and
failure<=33/140<1/3. N16,M8,t1,q4 has F=128/1365; eta1/12 gives r96 and
failure<=967/5460<1/3. These controls are outside some asymptotic
N>=(tQ)^4 domains; they test a finite feasibility contract, not that sampler claim.

Decision tokens for fresh design-first/proof reviewer: DESIGN PASS/REPAIR/KILL;
EXACT BENCHMARK PASS/REPAIR/KILL; FINITE-SEED transfer PASS/REPAIR/KILL. Inspect
primary online testing results so this control is not advertised as new. Small
exact DP/enumeration can attack the formula after coherent design; EMPIRICAL
probes, no registered FULL or code modification. A failed search is not proof.
