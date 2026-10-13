# Closed endurance boundary machine: repaired design candidate

2026-10-04 DEFINITION CANDIDATE; fresh design gate pending. This applies every
explicit SOURCE/DESIGN/RELAXATION repair in ENDURANCE_SOURCE_DESIGN_PRETEST.
Genuine advisor/architect dialogue remains active; no theorem, code or solver
run is admitted by writing this definition. CS+math, CPU-only, no papers.

Parameters are finite integers N>=k>=1, L>=1 and proposed finite horizon H>=0.
Let a=ceil(log2(N+1)), q=ceil(log2(N*L+2)). Private volatile allowance is exactly
S=256*q^2 bits, including every PC, pointer, return address, register, arithmetic
temporary, branch/control bit and stack. Fixed parameter-only code is generated
uniformly; its bit length, generation, installation and execution access costs
are reported. No input-dependent code, external clock, invocation counter or
cache. Constants cannot replace charged table generation/evaluation.

Persistent state comprises exactly N binary cells, initially supplied as known
zero and unworn. Allocation/provisioning and all source/init work are charged.
Each successful bit value change consumes one of that cell's L wear units.
Same-value WRITE consumes instruction/routing work and zero wear. Every actual
intermediate change counts, including later-undone changes and initialization.
An opposite-value WRITE after L changes is INVALID execution: no failure flag,
saturation, timing signal or read-back wear oracle is furnished. Admitted
algorithms guarantee that this never occurs on their guaranteed histories.
READ observes the current bit only, not wear. The checker alone retains wear.
This is a strict finite transition-budget model; ELM's saturation semantics
require an explicit comparator transfer, not substitution.

Initialize an arbitrary raw k-bit vector x. Source bits are available only
during this phase, all reads/copies/writes charged. After completion raw x is
unavailable to the algorithm. No additional persistent cells store originals,
metadata, future commands, helper buffers or previous outputs.

Commands are FLIP(i) and QUERY(i), 0<=i<k. During an invocation only its opcode
and index are readable command input. Every invocation starts in the same
parameter-dependent canonical volatile entry state. At completion all
input-dependent volatile information is cleared with charged finite work;
PC returns to a fixed parameter-only entry state. Persistent metadata must be
in the N cells. QUERY is read-only on those cells and emits the exact logical
bit; outputs/acknowledgements are unreadable afterwards. FLIP toggles that bit.

Correctness/lifetime quantifiers: initialization terminates for EVERY initial
x. At every boundary reachable by a valid prefix containing at most H FLIPs,
EVERY QUERY terminates correctly, including after flipH. At every such boundary
reached with fewer than H FLIPs, EVERY FLIP terminates correctly. Arbitrarily
many QUERYs may be interleaved. Every promised invocation respects wearL and S
and finishes in the canonical volatile boundary state. No termination promise
applies to irrelevant states or unpromised invocations. Operations are
sequential; there is no overlapping query during an unfinished FLIP, no
mid-command crash or revocation guarantee. Such extensions need separate
contracts; completed-boundary reset is the current task.

Primitive cost: finite-fan-in Boolean bit operations, bit copies, bounded
branches, controller/PC/return-address manipulation and emitted bits each cost
one elementary step. One addressed persistent bit READ/WRITE additionally
consumes its a-bit routing address, for a+1 steps; address formation is charged
separately. All multi-bit arithmetic is expanded and all intermediates fit S.
No free random access, table lookup, parallel sweep or arbitrary-width atomic
word. This declared indexed-bit model is not a physical wire-delay theorem.

Prospective finite endpoint-policy relaxation (DESIGN CANDIDATE): arbitrary
decoder d:{0,1}^N->{0,1}^k, initialization I, stationary maps T_i(s). Begin
checker records (I(x),wear=I(x),depth0) for ALL x, with d(I(x))=x. For EACH
reachable record with depth<H and EACH i, require the correct decoded flip,
all coordinate wear+ (s xor T_i(s)) <=L, and retain the successor. At depthH
only query correctness is required. Maps are tied across histories/depths;
every reachable wear vector is kept. Init and endpoint wear undercount actual
intermediate changes, making this a relaxation. Decoder/init/updates can be
nonlinear and multi-bit; no unproved symmetry or fixed I(0) is allowed.

UNSAT needs a complete formula plus independently reviewed reduction covering
all admitted boundary machines and a checked solver result/certificate; an
incomplete search is VOID. SAT requires compiled finite bit execution and
independent raw exhaustive replay, including init/intermediate wear, query
outputs, private/controller budget, code costs and boundary clearing. A finite
policy is not a scalable efficient uniform construction. No solver is run here.

Possible general targets remain CONJECTURE: H>=c*N*L/log^2N at N=2k and
ceil(log2^2N)<=L<=N with polylog worst-case command work; universal nonlinear
bound H<=L*min(N,2^(N-k+1)-1). At N=2k the latter is only total wear. Strong
ELM/floating/local-code/wear-leveling/persistent-controller comparisons and
ENDURANCE_JOURNAL_DESIGN_CANDIDATE must survive actual source/design review.
No strict priority, matched new-model pair or physical lifetime claim here.
