# General returned-mask cleanup without workload learning

2026-10-03. PROVED AT SCOPE: lower and charged upper passed fresh independent
actual attack in GENERAL_MASKED_REVIEW.md. Original candidate wording below
is provenance. Physical interpretation is conditional inference only. No new
registered general-function FULL exists. Do NOT change the running registered
116-circuit AND experiment, its q exception or primitive/source identities.

## Contract and leak audit

Finite exact f:{0,1}^n->{0,1}^m, any original data prior p_X, independently
uniform r-bit R returned pointwise, r>=1. Fixed finite NOT/CNOT/Toffoli/one-bit
RESET machine, full clean output (f(X),R), scratch/data zero. Local cost at
each RESET is target-bit entropy. Source and program/preparation/encoding/gates
charged; supplied reversible compute/copy/uncompute implementation has charged
gate cost C_f. No distribution oracle, fresh random gates, real precision or
infinite running time. Output may have arbitrary m and n,r are finite.

## Candidate generic CAN

After copying R to output and reversibly computing f with scratch cleaned,
partition X into chunks d<=r as in MASKED_BLOCK_CANDIDATE. XOR returned lowR
into the chunk, enumerate highR contexts by charged Gray predicates, but use
that context itself as ROOT for incoming chunk Gray merges. There is NO Yzero
predicate, no initial CNOT(Y,X), and no Yone special cleanup. The whole output
register Y is never touched during data erasure.

At every macro, (a,b) maps to b within EACH fixed combination of untouched
registers, including Y. Even when one predecessor pattern is absent in a
particular output fibre, the representative advances reversibly. Distinct Y
states cannot coalesce since RESET removes only its flag bit. At the end every
live chunk value is allones; uncompute its prefix and CNOT(context,chunk bits)
to zero. Undo highcontext predicates. The chunk is fullyzero for ALLoutput
fibres, without ever inspecting Y; R and Y remain unchanged.

Before every incoming RESET, every ORIGINAL X has EXACTLYone original R label
whose untouched masked chunk is the incoming pattern and whose highR matches
the selected context. Visited patterns have moved to the old representative,
so this incoming pattern is unvisited. Previous chunks may have coalesced
original labels, but did not touch the current chunk or original R. Therefore
q_x=one for EVERYX, and the target flag probability is EXACTLY2^-r under EVERY
original data law, not merely bounded above. Its local entropy is h2(2^-r).

Charged cleanup G=O(n+r+2^r*max(1,n/r)), widthO(n+m+r) in addition to the
supplied clean reversible f implementation. TotalG=O(C_f+n+r+2^r*max(1,n/r)).
No n,m truth-table complexity is hidden in C_f. If f is given only as a truth
table, constructing that clean implementation may be exponentially expensive.

## Candidate exact universal feasibility for unrestricted unknown priors

For ANY noninjective f, no universally correct clean finite circuit can obey
K<h2(beta), beta=2^-r, over ALLp_X. At a fixed RESET, target probabilities are
an interval because all p_X form a simplex. If K<one and K<h2(beta), the whole
interval must be in one entropy tail with a globally rare target probability
<=a<beta. For each original X the pure prior p_X=delta_x is admitted; rare
probability is q_x*beta, integer q_x. Thus q_x=zero for EVERYX: no original
label takes the rare value. Such a RESET cannot merge any two live complete
states. All RESETs are reversible on full original support, so final clean
noninjective f is impossible. At r1, K<h2(beta)=one gives the same interval
argument. Injective f has a separate reversible zero-loss cleanup exception.

The generic upper attains h2(beta), so this is an EXACT universal local floor
for noninjective f with completely unknown workload. It does not replace
eta1/8's lower feasible floor h2(beta/8), which is a DIFFERENTprior contract.
For AND n>=3 the masked critical work lower already reviewed on the smaller
capped class extends to allp_X; thus the same criticalTheta work holds here.

## Fixed design implication and limits

CANDIDATE INFERENCE: every RESET now has the SAME known marginalbeta even
though workload p_X is unknown. An ideal local reset protocol tuned once to
q=beta would have no local workload-design mismatch at these gates. This is
stronger than merely choosing a minimax design for an interval. It does not
eliminate modularity costs, supply finite-time physics, make reversible gate
residual costsfree, or calibrate V actual branch surprise to device heat.
Finite control precision, time, source preparation/authentication and high-
energy rare events remain explicit operational questions. No novelty asserted.

## Fresh proof attack

Read-only actual zero-context CLI should attack generic output fibres,
empty pattern classes, prior chunk coalescence, exact q_x=one after each
macro, scratch/returnedR, charged basis/work, unrestricted-prior lower and
injective exception. Bounded independent CPU prototypes on constant/parity/
majority/injective/multioutput and AND welcome; no edits, agents, GPU or FULL.
Return LOWER/UPPER PASS/REPAIR/KILL, identifying physical implication separately
as conditional inference. This is NOT the existing116-circuit FULL admission.

So what: one finite returned mask can make each erasure primitive's marginal
law exactly known without learning an arbitrary workload, while the clean
machine still has a matching universal feasibility obstruction. If verified,
this is a generic compiler result rather than an AND-only accounting curiosity.
