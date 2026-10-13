# Universal closed-endurance bounds: candidate proofs

2026-10-04 CANDIDATE PROOFS / not PROVED. Model is the independently admitted
ENDURANCE_BOUNDARY_MODEL_CANDIDATE, SHA
f0ab692cf629fd1f3c4f35d7e6600c3a09f5a6505b98f9ff776649b3c8d7ec97.
All persistent history is in N binary cells; canonical volatile boundaries,
terminating all promised commands, arbitrary raw x, all update histories,
read-only queries and strict no-feedback exhaustion. All actual changes,
including initialization/intermediates, consume wear. No positive journal
construction or controller compiler is assumed. N>=k>=1,L>=1,H>=0.

Honest intended so-what: quantify how little redundant persistent state can
prevent hostile updates from concentrating finite wear, for ALL nonlinear
representations and deterministic controllers. Priority and tightness outside
the stated simple case remain open. Existing floating-code limits are stronger
comparators, not a novelty certificate.

## A. Fibre projection bound, including charged initialization

Candidate statement, r=N-k:

    H <= L*min(N,2^(r+1)-1)-1.

Use the admitted static decoder d on all reachable boundary states, extending
it arbitrarily to all 2^N strings. Every logical vector has a nonempty fibre
F_x because initialization terminates correctly for every x. Hence these
fibres partition the 2^N strings, with no linearity assumption.

For each edge {x,y} of the logical k-cube sum |F_x|+|F_y|. Each fibre occurs
on exactly k edges, so the average union size is

    k*2^N/(k*2^(k-1)) = 2^(r+1).

Choose one adjacent pair with s=|F_x union F_y|<=2^(r+1). Any set of s
distinct N-bit strings admits a distinguishing coordinate projection P with
|P|<=min(N,s-1): start with no coordinates, repeatedly select a coordinate
separating a pair in a current unresolved class. Each selection increases
the number of classes; after at most s-1 selections every class is singleton.

Initialize either x or y and repeat their differing logical FLIP H times.
Every completed state lies in F_x union F_y; each update changes the decoded
value, so its endpoints differ on P. Actual intermediate execution therefore
changes at least one P cell. Endpoint jumps need not be one-bit; intermediate
states need not stay inside the two fibres.

The two actual initializer endpoints I(x),I(y) have different projections.
At least one of them has a one on P. Choose that raw input. From unworn zeros
its initialization consumes at least one P wear unit. Thus initialization
plus H updates consumes at least H+1 units on P, while those cells supply
at most L|P|. This proves the displayed candidate inequality if the decoder/
extension/trace arguments survive independent attack. No algorithm observes
the mathematical projection or the checker's wear records.

Corollary candidate: when N=k, the optimal guaranteed lifetime is L-1.
Lower above gives H<=L-1. Direct base storage initializes x and toggles the
selected base bit on each FLIP; H=L-1 keeps each cell within L, even for an
all-one input and a single hot bit. QUERY reads that cell, canonical entry/
scrubbing and a-bit addressed routing use O(q^c) finite work and less than
S256q^2. This is an ordinary overwrite baseline, not a novel CAN mechanism.

## B. Initialization plus final-branch budget

Define t0 as the least integer t with

    sum_{j=0}^t binom(N,j) >= 2^k,

and b=ceil(log2(k+1)). Candidate statement:

    H <= max(0,N*L-t0-b+1).

Correct static decoding makes I injective. Therefore some initializer
endpoint has Hamming weight at least t0, so its actual initialization wear
is at least t0. If H>=1, start from that x and follow any H-1 FLIPs. Each
must make at least one physical change, since its decoded value changes;
total spent wear at that boundary is at least t0+H-1.

Let u be the number of cells with positive remaining wear. Every one of the
k possible final FLIPs must terminate correctly. Their endpoints represent
k distinct logical neighbors, different from the current logical vector.
Exhausted cells cannot change on any valid branch. The other u cells permit
at most 2^u-1 distinct endpoints different from the current physical state,
even allowing arbitrary multi-bit/intermediate operations. Thus k<=2^u-1,
u>=b, and total remaining wear is at least b. Consequently

    N*L >= t0+(H-1)+b.

If the right-hand bound for a positive H is negative, no positive H is
possible; H0 remains a valid possibility, giving the max(0,...) formulation.
This candidate bounds all algorithms regardless of their local work or S.
For N4,k2,L2 it gives H<=6 (t0=1,b=2). That is a prospective calibration
ceiling, not an executed solver result or asymptotic tight-loss theorem.

## C. Strong primary comparator: monotone wear lift

SOURCE-TRANSFER CANDIDATE, not PROVED. Primary Jiang-Bruck Floatingcodes.pdf
Theorem2, binary logical variables, n=N and Q=L+1 would yield

    H <= (N-k+1)*L + floor((k-1)*L/2).

Source physical levels w_j are cumulative actual bit changes; their parity
is the original persistent state. On every required lifted update, run the
original canonical program on parity(w). Its actual changes c_j give lifted
endpoint w+c, monotone and bounded by L. Decoder on required nonzero levels
is original d(parity(w)); finite reachable levels are enough for all promised
histories, and irrelevant states can be extended arbitrarily.

The original program need not terminate on an initially unreachable binary
zero. To bridge the primary zero-level starting condition, choose one raw x0,
declare lifted decoder(0)=x0, and define the FIRST lifted update at w0 as
the original initializer for x0 followed by the requested FLIP. Charge both
in the level increments. Thereafter cumulative wear is nonzero, so every
required call runs a genuine boundary state. The first FLIP necessarily
changes something; if initialization ends at zero, correctness ensures
its original decoder also yields x0. No source update index/controller may
remain outside the lifted state; the rule w=0 detects only the lifted initial
state, not an external step counter. The lifted encoder may see extra wear
information because it is a relaxation, not an implementation in the original
model. It is deterministic and stationary on valid histories.

Fresh attack must check that Theorem2 covers this initialization wrapper,
source encoder/decoder quantifiers, partial required domains, possible
repeated parity states and multi-level endpoint increases. If transfer fails,
report its precise scope independently of A/B; do not discard valid A/B or
claim a new lower theorem from a citation. No monotone construction is
transferred in the opposite direction by substituting Q=L+1.

## Admission and limits

No numerical solver/implementation/FULL is run here. Fresh actual Codex
proof attack is required before any statement becomes PROVED. Machine design
is independently admitted; the separate positive journal compiler gate remains
REPAIR, and no lower theorem would bypass it. Useful-world strict primary
delta, general tightness, local-work versus lifetime loss and low-redundancy
achievability remain OPEN. All root/source comparisons preserve rigor labels.
