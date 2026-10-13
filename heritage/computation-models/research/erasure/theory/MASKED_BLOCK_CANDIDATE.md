# Masked-block cleanup: closing growing-r local work

2026-10-03. PROVED AT SCOPE: lower and charged upper passed fresh independent
actual attack in MASKED_BLOCK_REVIEW.md. Submitted candidate wording below is
retained as provenance. This closes METER_REVIEW's growing-r local critical
work gap for the stated finite contract. Registered MASKED_BLOCK_EXPERIMENT
controls remain unexecuted; physical calibration and primary priority are OPEN.

## Contract and infinity audit

Clean AND n>=3, X prior unknown with max atom1/8, independently uniform r-bit
original R returned pointwise, r>=1. Local RESET cost L=H(target bit), fixed
NOT/CNOT/Toffoli/one-bit RESET schedule, all initialized scratch finallyzero.
Cap K=theta_r=h2(2^-r). Every bit/gate/program/source-preparation charged.
No random gates, distribution queries, fractional data atoms or unbounded
precision. Use returned R as reversible XOR masks; R is never consumed.
Finite exact mass statements concern ORIGINAL (X,R) labels even after earlier
data coalescence. Supplied product-uniform law is essential; a fair marginal
with arbitrary correlation does not provide the masking bound.

Proposed critical optimal elementary gate work:

    Theta(n+r+2^r*max(1,n/r)).

Constants uniform in n,r, width O(n+r). Program-generation/encoding bit work
separate; no physical heat or novelty conclusion from this candidate.

## Candidate charged CAN

Transfer original R to output with2r CNOTs. Reversibly compute/copy/uncompute
AND in2n-1 gates. Apply CNOT(Y,Xbit) to every data bit: ANDone input is nowzero
data; ANDzero inputs retain their original data. Set clean sentinelone.

Divide X into contiguous chunks of size d<=r (d=min(r,remaining)). For each:
XOR returned R's d LOW bits into chunk data. Enumerate returned R's other r-d
bits with the charged Gray prefix maintenance. For each high-R context, form
condition=(context AND NOT Y), with clean scratch and explicit reversible
literal/predicate gates. Within the chunk, run IncomingCircuit's charged
incoming-singleton Gray merge sequence over ALL2^d patterns, simultaneously
across the low-R contexts. A prefix for the current chunk representative is
maintained, and each incoming merge clears all dependent predicates before
changing data. The pattern order ends at allones.

At every incoming RESET, each original X label can mark at MOSTone original
R label: its originally untouched masked chunk equals the incoming pattern
only when R_low=X_original_chunk XOR pattern, and R_high is the selected fixed
context. Previously visited masked patterns have moved to the old accumulated
representative, which differs from the incoming pattern; they cannot be marked.
Previous chunk cleanup changes neither untouched chunk bits nor original R.
Yone is excluded. Thus target probability<=2^-r for EVERY original p_X (even
without its max-atom promise), and L<=h2(2^-r). Empty incoming classes cause
no problem; the final original ANDone exclusion may occur at a DIFFERENT
masked pattern in each R slice, and is excluded by Yzero throughout.

Uncompute the final chunk prefix, then CNOT(condition,chunk_bit) on all chunk
bits to turn the accumulated allones representative intozero. Uncompute
condition and advance/uncompute high-R context predicates. Once ALLhigh-R
contexts are processed, chunk data arezero for Yzero; for Yone they still equal
R_low (because original data werezero before masking). TOFFOLI(Y,R_low,chunk)
clears those Yone data, without changing R or Y. The chunk is clean. Repeat;
uncompute sentinel. All scratch/initial R/data registers endzero.

Charged work per chunk: O(r+d+2^(r-d)*(2^d+d)+2^(r-d)). Gray suffix reuse costs
O(2^d) per context; reversible d-bit representative cleanup costs O(d) each,
and d<=2^d. High-context maintenance O(2^(r-d)); mask/Yone cleanup O(d).
Across chunks this is O(n+r+2^r*ceil(n/r)) when r<=n and O(n+r+2^r) otherwise.
No uncomputed erased predicate or arbitrary aggregate merge is free.

## Candidate CANNOT, including the missing class-count lower

Total joint loss for uniform X is n-h2(2^-n). J<=L at every RESET, so
G>=Omega(n/theta_r)=Omega(n*2^r/(r+1)), using
theta_r=Theta((r+1)*2^-r). Input-targeting separately gives Omega(n+r).

For r>=2, theta_r<one. At EACHRESET target-one probabilities over the capped
simplex form an interval. Requiring entropy<=h2(beta), beta=2^-r<1/2, forces
the entire interval into [0,beta] or[1-beta,1]. Choose its globally rare side.
For each original X let q_x count original R labels on that side. Its worst
probability is beta/8 times the sum of the eight largest integer q_x.
Consequently that sum is <=8. The eighth-largest is <=one; remaining q_x
are also<=one. Total rare ORIGINAL labels<=8+(2^n-8)=2^n. Every RESET collision
uses a distinct rare predecessor class, containing a distinct rare original
label. Thus each RESET eliminates at most2^n complete-state classes even
after earlier coalescence/recoding. Total decline is2^r*(2^n-2), giving
G>=2^r*(1-2^(1-n))=Omega(2^r).

For r1 the class-count lower follows simply because one-bit RESET fan-in2
eliminates at most half of the initial full-support class count, and the
needed Omega(2^r) is constant. Combining the three lower bounds matches the
charged upper for all n>=3,r>=1. Exact critical feasibility was already
approved in METER_REVIEW; the present candidate closes its work gap only.

## Proof gate and meaning

Fresh zero-context actual CLI must independently attack original-label marking
after prior chunk coalescence, ANDone exclusion at variable masked positions,
R preservation, scratch/prefix lifetimes, charged Gray work in r>n/shortlast
chunk, rare-side interval quantifier, eight-largest bound and all parameter
limits. Independent finite elementary prototypes welcome CPU only; no edits,
agents, GPU or FULL. Give LOWER/UPPER PASS/REPAIR/KILL, exact scope and defects.

So what: an explicitly returned finite mask can cut a factor growing with r
from local-budget cleanup work by processing r data bits per Gray sweep. The
same-contract matching impossibility would make the improvement a resource
law rather than a sampled speedup. Physical calibration, source authentication,
priority relative to reversible/masking literature and mission remain OPEN.
