# Finite atomic erasure: exact feasibility candidate

Date: 2026-10-03. Rigor: PROVED AT STATED DETERMINISTIC UNIFORM CLEAN SCOPE.
Independent fresh actual CLI review: ATOMIC_COMPILER_REVIEW_2.md, lower PASS,
charged upper PASS. This label covers the mathematical theorem pair below;
finite experiment evidence, novelty, and mission completion are separate gates.
This extends ATOMIC_ERASURE_FRONTIER's abstraction; no mission or priority claim.

## 1. Exact machine and task contract

Let n,m>=1, M=2^n, and f:{0,1}^n->{0,1}^m be supplied by its explicit M-entry
truth table. The input distribution is exactly uniform; zeros initialize every
other bit. A circuit has a finite fixed width w and a finite schedule. Each
elementary step is NOT, CNOT, Toffoli, or one-bit RESET. All gates have constant
arity, every gate counts, and all input, output, workspace, and controller state
is included in the entropy. Fixed schedule time is public and input-independent.
Finite parallel layers of disjoint gates may also be allowed: the lower bound
holds for them, and the construction uses sequential elementary gates.

Resources: gate count, depth, width, truth-table/program/construction bit work,
and peak per-step joint Shannon entropy loss. No inference to physical energy,
clock duration, noise robustness, or gate implementation is made. K is an
architectural resource parameter, not a hidden real-valued runtime oracle.

Clean task: final output bits equal f(x), and every other bit is zero, for every
x. This is the SAME contract for both the upper and lower sides. Truth-table
acquisition, storage, and reading are charged; no efficient succinct-f compiler
or speed advantage is asserted.

Let q=max_y |f^-1(y)|. Put phi(1)=0 and
phi(k)=k log2 k-(k-1)log2(k-1) for k>=2.

## 2. Candidate theorem pair

CANNOT: every clean deterministic realization has some step with entropy loss
at least phi(q)/M. If K<phi(q)/M, no finite-depth realization is admitted.

CAN: there is a uniform truth-table-to-circuit compiler over the fixed basis
whose maximum entropy loss is exactly phi(q)/M. Width can be
n+m+max(n,m)+1. Gate count is O(M*n*m); construction reads/encodes the
whole table and the generated circuit. Consequently clean realizability at any
finite depth is equivalent to K>=phi(q)/M, under this exact contract.

For q=1 the construction is entirely reversible, with peak zero. The theorem
does not characterize optimal depth/size under K, and the upper cost is generally
exponential in n. An exponential table compiler is not a useful efficiency gain.

## 3. Lower bound: indivisible full-state fibres

For each time t, define x~_t x' iff their full machine states agree. Determinism
implies ~_t is contained in ~_(t+1); equal states cannot later separate. Exact
clean output implies ~_D is precisely equality of f. Thus no intermediate class
can cross two final f-fibres. A class of a atoms has probability a/M, and

    H(S_t)=log2 M-(1/M)*sum_{A in partition_t}|A|log2|A|.

If q>=2, choose a largest final fibre F. Its q singleton initial classes become
one final class. At its FIRST complete formation step, the predecessor classes
inside F have integer sizes a_1,...,a_r, r>=2, sum a_i=q. Their contribution to
the entropy loss is (q log2 q-sum_i a_i log2 a_i)/M.

For positive integer parts with at least two parts, the sum is at most
(q-1)log2(q-1). One proof: merging two parts increases the sum because
(a+b)log2(a+b)>=a log2 a+b log2 b. Reduce to two parts. Convexity of
t log2 t makes a log2 a+(q-a)log2(q-a) maximal on integer 1<=a<=q-1 at
the endpoints 1,q-1. Other simultaneous coalescences contribute nonnegative
loss. Therefore the complete step loses at least phi(q)/M. Fixed extra zero
bits, reversible encodings, and arbitrarily many prior reversible steps do not
change this partition argument. Timing, fresh randomness, unretained output,
and external information channels are excluded by the contract, not suppressed
to rescue an already broader theorem.

## 4. Upper bound at the partition level

Within each f-fibre, choose its first lexicographic input a as representative.
In separate stages, merge each other singleton b into a. When the accumulated
class size rises from k-1 to k, the entropy drop is phi(k)/M. The derivative of
phi(t) for real t>1 is log2(t/(t-1))>0, so the largest drop is phi(q)/M.
There are exactly M-|image(f)| selective merge stages. The following compiler
realizes each stage without a free arbitrary-merge primitive.

## 5. Gate compiler for one selective merge b->a

All scratch starts and ends zero. Reserve a disposable flag z. Reversibly
compute z XOR [x=b], with negative controls implemented by explicit NOTs and
an AND ladder built from Toffoli gates. Uncompute the ladder BEFORE changing x.
This costs O(n) fixed-arity gates and O(n) reusable zero scratch. For n=1 and
n=2, CNOT and Toffoli give the direct controlled flips. Then:

    for each bit where a and b differ: CNOT(z,input_bit)
    RESET(z)

Immediately before RESET, the b state is (a,1), the prior a state is (a,0),
and every other x is (x,0). RESET therefore merges only a and b. Every XOR
change is reversible and explicitly charged; no many-bit overwrite is a primitive.
Net effect: b maps to a, every other input value is unchanged, all scratch is zero.

All non-RESET gates are permutations on the complete state. The RESET merges
exactly the accumulated a-class and the singleton b-class, losing phi(k)/M.
One O(n)-sized workspace is reused across all stages. No input-dependent
schedule, free fan-in, hidden lookup, or initialized infinite workspace enters.

## 6. Clean terminal conversion

After the merge stages, the input register contains rep(f(x)), with one distinct
representative per output label. Start an m-bit output register at zero. Compile
the full truth table into reversible XOR evaluation:

    (x,y) -> (x, y XOR f(x)).

For each table row, multi-controlled NOTs on x flip each one-bit of that row's
output value. This is a finite explicitly generated circuit. Then compile

    (x,y) -> (x XOR rep(y), y),

using controls on the output label, with rep(y)=0 for labels outside image(f).
On the reachable support, x was rep(y), so x becomes zero. Both maps are
reversible on the whole Boolean state space, so this conversion loses no entropy.
All scratch is uncomputed; only f(x) remains. Multi-controlled gates use the same
pool, with capacity max(n,m). An additional scratch bit serves the selective
erasure. Gate counts are O(M*n*m) for conversion and O(M*n) for merging.

Implementation note (2026-10-03): the initial compiler smoke used a longer
Gray-path/controlled-swap construction. The independent first reviewer derived
the simpler flag-reset construction above. The worker adopted it before full
execution, retaining the exact registered task/range/acceptance criteria and
requiring a fresh current-source code review. No outcome threshold changed.

## 7. Separation invisible to total output entropy

On M=16 uniform atoms, choose two bucket functions, both with four output bits.
Function A has fibre sizes (4,4,4,1,1,1,1). Function B has fibre sizes
(8,1,1,1,1,1,1,1,1). In both cases sum q_y log2 q_y=24, so H(f)=4-24/16=2.5
and total entropy loss is 1.5 bits. Yet A's threshold is phi(4)/16 and B's is
phi(8)/16. At K=1/4, A should be realizable and B impossible at every depth:
phi(4)<4 follows from 4^4/3^3=256/27<16; phi(8)>4 follows from
8^8/7^7>16 (equivalently 2^20>7^7=823543). Both comparisons are exact.

The original total-loss bound gives D>=6 for both and cannot distinguish them.
This proposed strict delta is mathematical relative to that bound, not a claim
of literature-final novelty or mission-level usefulness. Primary-source searches
must examine entropy-constrained quantization, information partition lattices,
conditional reversibility, and bottleneck aggregation before any novelty claim.

## 8. Adversarial limits and next gate

Nonuniform atom masses require a different optimization. Fresh random bits change
the atomic distribution and can weaken the threshold; their entropy and eventual
cleanup must be charged. Deterministic injective recoding preserves the atoms and
the bound. Free randomized padding is not such a recoding. Allowing a dirty final
state or information-bearing timing changes the task. The exact feasibility result
alone does not supply a practical efficient family or a new-paradigm verdict.

Review status: the fresh proof and code gate passed. The full registered controls
returned CONFIRM (78 clean AND, 12 dirty, 16 partitions, 308 compiler cases),
pending independent raw-output evidence admission. Source closure and efficient
family/robustness usefulness remain open. Honest narrative: two
tasks with identical total information loss can have different all-depth erasure
feasibility because finite uncertainty cannot be divided arbitrarily.
