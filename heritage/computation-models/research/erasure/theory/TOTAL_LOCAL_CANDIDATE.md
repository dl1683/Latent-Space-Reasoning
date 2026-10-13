# Small peak without hidden total local-entropy overhead

2026-10-03. PROVED AT SCOPE: fresh actual product LOWER/UPPER and conditional
correlated tradeoff PASS in TOTAL_LOCAL_REVIEW.md; its correlation prerequisite
also independently passes CORRELATION_LOCAL_REVIEW.md. Original candidate
text below is provenance. Existing registered
experiment and source remain unchanged. This is logical L, not calibrated heat.

## Contract and finite resources

Same clean AND n>=3,r>=1, X max atom1/8 unknown, independent uniform returnedR,
fixed finite NOT/CNOT/Toffoli/one-bit RESET, local peakcaptheta=h2(2^-r).
Define totallocalcost E_L=sum_RESET H(target bit) on an admitted source law.
Reversible gates costzero under this logical information meter while their
work/width/program cost is charged. Prior preparation/encoding is finite and
charged. All classes refer to original labels; no precision/advice/oracle leak.

## Candidate matched product-source total cost Theta(n+r)

For any universally peak-admissible circuit, uniformX and independent uniformR
give totaljointloss n-h2(2^-n), so E_L>=n-h2(2^-n), since J<=L gatewise.

For r>=2, peaktheta<one forces a globally rare orientation at each RESET over
the convex capped-prior class. Under the uniform source, write N_g for number
of ORIGINAL(X,R) labels taking that rare value. Its probability mu_g=N_g/2^(n+r)
is <=beta=2^-r. Since h2(mu)>=mu log2(1/mu)>=r*mu (including muzero),

    E_L >= r*sum_g N_g/2^(n+r).

Each binary collision uses a distinct rare predecessor containing at least
one original rare label. Thus N_g>=number of full-support class collisions at
this RESET, even after previous coalescence. Summing required class decline
2^r*(2^n-2) gives

    E_L >= r*(1-2^(1-n)).

Together these lower bounds give Omega(n+r). For r1 the jointloss lower alone
supplies this order. Extra running time or scratch cannot evade it. The same
lower applies to unrestricted unknownp_X because that class contains the cap.

MASKED_BLOCK_REVIEW's existing charged AND upper has
T=sum_chunks[2^(r-d)*(2^d-1)]<=ceil(n/r)*2^r RESETs. Every admitted product
law has flag probability<=beta, hence E_L<=T*h2(beta). Using
h2(beta)<=beta*(r+log2(e))<=beta*(r+2) gives

    E_L <=ceil(n/r)*(r+2)=O(n+r).

It simultaneously attains the reviewed optimal critical work order
Theta(n+r+2^r*max(1,n/r)). Thus low peak does not hide an extra factor-r total
local cost in this positive construction; the unavoidable extra-r scale is
visible. Generic exact-marginal compiler has the same upper; an AND-specific
matching lower must not be transferred to arbitrary injectivef.

## Candidate work/total-cost constraint for correlated sources

Conditional on CORRELATION_LOCAL_CANDIDATE's fresh proof PASS: for r>=4 the
larger source class (same cappedX and uniformR marginals, arbitrary correlation)
forces each RESET's globally rare value to occur in at mostONEoriginalR slice.
Consider a circuit valid for this class with G<=C*n*2^r, fixed C>=1.
On uniformjointinput let a_g be conditional rare probability in that activeR
slice (zero when no active slice), T its RESETcount, and beta=2^-r.
Then mu_g=beta*a_g, J_g<=beta*h2(a_g), and

    sum_g h2(a_g) >=2^r*(n-h2(2^-n)),
    T<=G<=C*n*2^r.

Entropyconcavity gives h2((sum a_g)/T)>=1/(2C), since n-h2(2^-n)>=n/2.
Therefore (sum a_g)/T>=alpha_C>zero, where alpha_C is the lower inverse
binaryentropy at1/(2C). Also T>=2^r*(n-h2(2^-n)) since each h2<=one.
Because h2(beta*a)>=r*beta*a,

    E_L >=r*beta*alpha_C*T >=alpha_C*r*(n-h2(2^-n))=Omega_C(n*r).

The context-linear upper uses n*2^rRESETs with mu<=beta, and totalE_L<=n(r+2),
so the lower applies to every fixed constantC work allowance. The matching
Theta(nr) upper uses a sufficiently large fixed allowance to fit the actual
context-linear word; existence under every prescribed C>=1 is not implied.
Uniformjoint already forces the lower. Its constant may depend on C.

This is NOT a lower for every arbitrarily slow correlated compiler. Singleton
Gray can spend exponentially more gates and reduce the totalcost; the time
quantifier is load-bearing. It is NOT a physical energy comparison. Product
and correlated robustness impose different supplied source promises explicitly.

## Fresh independent attack and meaning

Attack rare-side label counting, entropy inequality/zero masses, class decline,
upperceil(n/r) uniformly includingr>n, correlated one-active-R lemma, Jensen
direction, inverseentropy/constantC/TvsG, admitted uniformsource and exact
conditional scopes. Read-only actual freshCLI, CPU probes optional, noedit,
agents, GPU orFULL. Return product LOWER/UPPER and conditional correlated
tradeoff PASS/REPAIR/KILL separately. No theorem promotion before verdict.

So what: account for BOTH peak and accumulated cost. A mask-based compiler
can attain the small peak and optimal work without multiplying total local
cost by r; source correlation can force that multiplication at optimal work.
Operational calibration and primary novelty remain separate open questions.
