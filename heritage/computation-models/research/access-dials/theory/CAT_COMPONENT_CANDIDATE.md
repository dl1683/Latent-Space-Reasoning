# Finite algebraic recall components after CAT-OPEN

2026-10-04 REPAIRED CANDIDATE / fresh regate pending. Tactical continuation allowed
by STEERING_OCTOBER_3_DECISION and actual CAT_PEBBLE_PRETEST. Not an automatic
matrix-powering CAN or fast recursive inverse. CPU only, no implementation/FULL.

Two finite components from the actual pre-test need independent admission before
investment: (A) elementary dirty-matrix power expansion; (B) isolated degree-five
triangular block with fast prefix recall. The pre-test's new claims are not PROVED.
Its characteristic-zero/passive degree results are not finite-field function lower.

A. Fix finite field F_p, dimension m, integer k>=1. Matrix U is immutable for the
call, even if its values are arbitrary owner data. All coefficients/arithmetic/
addresses/recursive controllers/program generation are charged. Four additional
arbitrary dirty matrices O,W,V,Z are explicit, not initialized. The proposed clean
operation POW(U,k;O) returns O+U^k and every other dirty matrix unchanged.

Base k1: add U coordinatewise. For k>=2, a=floor(k/2),b=k-a, recursively use
the rectangle with X=U^a,Y=U^b:

    O+=W*V
    POW(U,a;W)
    O-=W*V
    POW(U,b;V)
    O+=W*V
    POW(U,a;W) inverse
    O-=W*V
    POW(U,b;V) inverse

WV-(W+X)V+(W+X)(V+Y)-W(V+Y)=XY without commutativity assumptions. Recursive
targets are passive WITHIN that call; the other three mutable matrices may be
temporarily used as dirty auxiliaries, restoring them before the call returns.
Thus a child's use of parent O as scratch changes no parent state at call end,
but does NOT preserve global passive-output syntactic restrictions. Our finite
machine admits these temporary accesses. U stays disjoint/read-only throughout.
An inverse call reverses all expanded scalar shears; no recursion-output oracle.

Balanced recurrence F(k)<=2F(a)+2F(b)+O(m^3) gives candidate O(m^3*k^2) field
operations, with four matrices TOTAL for this backend. Include pointer/phase/
depth registers and schoolbook finite-field bit arithmetic (O(log p) temporaries)
in a finite O(log p*polylog(m+k)) clean budget; derive an explicit conservative
bound before promotion. Each scalar target update must exclude its own target
from the evaluated product; simultaneous endpoint matrix identity alone does
not certify the expanded word or aliases.

Embedding ONE passive scalar output update R*=R*+(U^k)_ij needs a distinct
four-matrix backend, not treating scalar R* as a full scratch matrix: save the
original O_ij in one clean field word, execute POW, add O_ij-original to R*,
execute inverse POW to restore O. Outer R* is disjoint from U and all four
matrices. The pre-test's three-additional-matrix statement is NOT promoted for
this scalar-output embedding. With a separate owner output word, the total is
C=C0+4m²+1; original O_ij and original R* occupy TWO persistent clean field words.
Save original R* separately if abort must restore it rather than merely suppress
future output. Generic backend prefix reversal may cost its whole expanded trace;
this component supplies polynomial elementary work, NOT fast recursive recall.

B. The pre-test selects finite p>5 and linear polynomial Y1(z), quadratic Y2(z),
immutable input-linear L(z)=sum A_i*z_i (b=m² input coordinates). Every coefficient
is a separate finite field word in the arbitrary owner bank. The conjugation

    Y2 -= Y1²/6
    Y1 += 3L
    Y2 += Y1²/6

equals Y1'=Y1+3L, Y2'=Y2+Y1*L+(3/2)L². Generate coefficient updates explicitly;
each quadratic coefficient is a constant-size expression in the two relevant
linear coefficients, not an arbitrary-degree one-step oracle. Other fixed
degree-five work transformations use analogous 1/4 and1/8 conjugations; read
the primary exact formulas before claiming the complete four-transform block.

At every field-word-write prefix, save that word's old value and write cursor
in clean storage until commitment. Record phase and coordinate loop indices.
On RECALL patch an unfinished write, reverse completed coefficient updates in
the current phase, then preceding phases. A committed Y1 input translation is
undone from immutable A; a Y2 shear from the correct corresponding Y1 phase.
Repeated recall continues abort mode. Partial arithmetic temporaries/controller
bits count; no full original coefficient snapshot. An isolated block has O(C0)
scalar coefficient updates and candidate C0*polylog(m+p) expanded recall.

Only global passive output R* may be abandoned by direct restoration from its
saved original field word; skipping output writes without restoring owner R*
would violate the contract. Recursive input L=A^(5^j) is NOT supplied free:
generating those coefficients is exactly the still-open recursive recall problem.
No base-block inverse becomes a full matrix-powering theorem by naming recursion.

### Frozen finite access meter and bit-prefix controller repair

The component uses a deterministic finite bit-addressed RAM, not a physical
flat tape. All owner field words are valid beta-bit encodings of F_p residues;
arbitrary unused binary encodings are excluded. Immutable source data and owner
bank regions are disjoint. Let q cover every bit address, beta, m, k and scalar
frontier index. Fetch or write of one addressed bit costs q+1 bit steps: consume
the q-bit address and one data bit. Boolean bit operations, copied controller
bits and finite branches each cost one. Address generation and routing are
charged. Thus one field-word access costs O(beta*(q+1)), not a free word fetch.
The claimed costs are within this explicit RAM contract; locality/physical wire
costs and a flat-tape translation are not claimed.

The normal coefficient controller stores the next-shear frontier f, phase,
coordinate indices, beta-bit arithmetic scratch, and a transaction containing
old value, new value, target address, saved predecessor frontier, write cursor,
and one pending bit. Each controller field is explicitly allocated; no host
stack or history list is an allowed model resource. Frontier generation uses
nested coefficient loops, not an initialized list of all operations. Modular
arithmetic uses finite schoolbook operations and charged generation of inverses
of 4/6/8, with p>5. The RAM PC and all branch/controller/address bits count.

One normal transaction follows this order, with each copy expanded bitwise:

1. With pending=0, prepare old/new/target and predecessor f while owner storage
   remains unchanged. Arithmetic/control interruptions here discard scratch.
2. Set pending=1 by one bit instruction only after all records are complete.
3. Write the new owner value bitwise, then copy successor f bitwise. Pending
   stays set during BOTH writes and every cursor update.
4. Clear pending by one bit instruction only after owner and frontier complete.
   This is the single commitment boundary. The next transaction may then reuse
   its scratch; completed f identifies every committed scalar shear.

An external RECALL sets a request bit between elementary instructions. Before
each normal bit instruction a charged branch checks that bit. The first request
latches abort mode before owner mutation or metadata reuse; subsequent requests
do not change the restoration PC/cursors. Dispatch uses a constant number of
bounded fields and bit copies. No atomic arbitrary-width checkpoint or swap.

On first dispatch, pending=1 implies a forward transaction may be incomplete.
Restore its saved old owner value, then its saved predecessor f, bitwise while
retaining pending. Clear pending only after BOTH are complete. Interruptions
during this patch continue the current abort PC and cursors, never restart it.
With pending=0 discard unfinished arithmetic; f already names a committed
prefix. Then generate inverse shears in reverse order from f. Undo each shear
using a new transaction, decrementing f only while pending remains set. During
this inverse phase every further recall continues the same inverse operation;
it does not apply the first-forward-transaction patch rule a second time. After
f reaches the initial frontier, clear private journal and arithmetic fields.
Coefficient data equal their entry values; passive owner R* is restored from
its saved original before return. Interruptions during cleanup continue cleanup.

For B, pending journals and arithmetic/indices use O(beta+q) clean bits. A uses
O(beta+(d+1)*q), d=ceil(log2 k), for its explicit depth-bounded frames and bit
scratch. Conservative streamed bit-work bounds, including generated constants,
addresses, arithmetic and transaction copies, are:

    H_A <= 10*m^3*k^2 scalar shears;
    T_A = O(m^3*k^2*(beta+(d+1)*q)^3);
    H_B = 4*b^2+8*b <= 8*C0 scalar shears;
    T_B and D_B = O(C0*(beta+q)^3).

Only B's fixed input-linear work maps have the fast prefix guarantee. A's
generic prefix inverse can cost its entire expanded computation. The four-map
sequence (c,h) is (3,1/6),(-2,1/4),(3,1/6),(-4,1/8). Each map performs
Y2-=h*Y1^2, Y1+=c*L, Y2+=h*Y1^2, giving respectively the primary transforms
listed in CAT_COMPONENT_REVIEW_1. Their full composition restores Y1,Y2.
No full degree-five output compiler or recursive powered-input generation is
supplied by this work-map claim. Complete output compilation remains separate.

Fresh reviewer first finite design/alias/controller/field-bit/initialization leak
audit, then separately A: clean endpoint/elementary work; B: isolated prefix
inverse and exact scope. PASS/REPAIR/KILL by component. Small CPU prototypes
welcome after coherent design as EMPIRICAL controls, no code edits/FULL. Positive
construction plus universal task limit and strict useful prior delta remain OPEN.
