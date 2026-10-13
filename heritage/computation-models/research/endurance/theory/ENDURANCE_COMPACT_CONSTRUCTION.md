# Cold one-box crit-bit construction: single registered attempt

Date: 2026-10-04. Unit: ENDURANCE-COMPACT-PROBE-CAN-1.
**Rigor for every mathematical assertion below: DERIVED / PENDING INDEPENDENT
PROOF ATTACK. Nothing in this delivery is PROVED or an admitted CAN.**
Submission token: **COMPACT-CAN-PENDING-REVIEW**. Repair consumption: zero at
entry, zero proposed/consumed in this construction. The fresh exact first gate
is ENDURANCE_COMPACT_PROBE_DESIGN_REVIEW: PASS, transfers VALID, SOURCE OPEN.
This is one construction of the frozen family, not another model or compiler.

The source receipt accompanying this file gives byte sizes and SHA256s of the
complete construction, fixed reference, frozen contract, review, decision,
pretests and actual outcome dialogues. The receipt is a read/syntax check;
no algorithm/workload/solver execution is evidence in this unit.

## 1. Exact proposed CAN, authority and exclusions

For every integer N=2k, k>=1, ceil((log2 N)^2)<=L<=N, set
q=ceil(log2(NL+2)). The program below proposes lifetime

```
H = L-1                 if no arena fits;
H = ML                  otherwise.
H >= max(L-1, floor(NL/(1024 log2 N))).
R_Q <= 32q^2;
R_F <= 192q^2; W_F <= 64q^2;
R_F + W_F <= 256q^2 <= 2^20 q^3.
S_used <= 36q^2 + 284q + 304 <= 256q^2.
prep + INIT ordinary bit-work <= 2^20 Nq^6.
INIT: R=0, W=k, current raw-input reads=k.
```

Maxima for reads and writes are bounded separately. Every DATA acquisition,
reread, write of zero and same-value write is counted. Every actual change,
including an intermediate change, belongs to the individual-cell ledger.
Correctness quantifies over every raw input, every history through H FLIPs,
arbitrary finite QUERY interleavings, and unlimited terminal read-only queries.

The canonical mutable state contains only the parameter table, zero scratch,
homed heads and READY control. It never contains an epoch count, prior index,
allocator, retained raw input, output cursor, root or random seed. Wear is not
readable. ACK follows tree update, maintenance, exchange and actual cleanup.
Only completed-command durability is claimed. No physical energy, command
runtime efficiency, old indexed-controller CAN, novelty, DELTA or mission
completion follows.

Read in this session: full AGENTS/full dual-loop skill; README; HANDOFF section
17; TERMINAL_GATE_2; SYNTHESIS_GRAPH section 29; STEERING_ENDGAME; CURRENT_STATE;
full compact MODEL/PROTOCOL/DESIGN_REVIEW/decision/construction packet; all six
actual final outcome rounds; compact/probe source pretests and fat-node sketch.
CURRENT_STATE's first-gate RUNNING text is stale relative to the supplied fresh
PASS. Historical bridge debts and terminal NO remain unchanged. No frozen
artifact has been edited. No agents, git/.git mutation, GPU, papers, solver,
H4/FULL/grid or reportable probe were used. Commands use cmd.exe/login=false;
Python source reading uses -B.

## 2. Integer layout, nulls and the direct branch

Use exactly the frozen integers, computed without floating point:

```
a = bitlength(N); d = bitlength(k-1); e = bitlength(d);
v = d + e + 3a + 6; C = 4; b = Cv + a + 1;
if k<=2: M=0;                         # BEFORE subtracting k-2
else if 8b>k-2: M=0;
else: M=4; while 4Mb<=k-2: M=2M;
A=CM; T=Mb.
```

bitlength(0)=0. q=bitlength(NL+1), exactly ceil(log2(NL+2)), even when NL+2
is a power of two. Public parameters do not need to satisfy an executable
real-log test: the real-log inequality is a theorem premise. The supplied
program checks the integer premises N=2k, k>=1, 1<=L<=N and uses the same
branch rule on all such parameters.

M=0 means **only base[0..k-1] is addressed**. All other cells remain zero.
There are no physical ROLE/STARTED or arena accesses in this branch. In
particular k=1,N=2 does not try to reserve k+2 cells. Zero-width key/crit fields
are never accessed, and a negative k-2 is never turned into an address/budget.
INIT writes raw x into base once. QUERY_i reads base[i]. FLIP_i explicitly
reads and writes its complement. Wear is at most 1+(L-1)=L, even for the
history repeatedly flipping the initially-one coordinate. No command ordinal
is needed to execute the promised L-1 updates.

For M>0, physical addresses are:

```
base[i] = i                            0<=i<k
ROLE = k; STARTED = k+1
arena(bank) = k+2+bank*T               bank in {0,1}
node(bank,u) = arena(bank)+(u-1)*v      1<=u<=A
root(bank,t) = arena(bank)+A*v+t*(a+1)  0<=t<M
```

ROLE selects NEW; 1-ROLE selects OLD. All cells at addresses >=k+2+2T remain
zero. The inequality 2T<=k-2 proves the last allocated address is <=N-1.
IDs are bank-local, with null=0 and real IDs 1..A. Since 2Av<=k-2 and v>=1,
A<k<=N, and all IDs/null fit a bits. Every actual root/child ID is allocated
in its own bank. A root record holds a pointer followed by a flag. It is a
publication record, not a snapshot. Published null is supported by the format
(although every nonempty epoch's published tree is nonnull).

Node words are little endian; key bit position p denotes its pth bit from the
most significant end, 0<=p<d. The complete v-bit offsets are:

| Field | Offset | Width |
|---|---:|---:|
| type (0 leaf, 1 internal) | 0 | 1 |
| immutable key | 1 | d |
| immutable critical position | 1+d | e |
| base left | 1+d+e | a |
| base right | 1+d+e+a | a |
| base parity | 1+d+e+2a | 1 |
| node publication | 2+d+e+2a | 1 |
| box field ID | 3+d+e+2a | 2 |
| box value | 5+d+e+2a | a |
| box publication | 5+d+e+3a | 1 |

The last offset+1 is v. Internal keys/parity and leaf critical position/children
are zero. Box fields 0/1/2 name left/right/parity; 3 is never published. A
parity value has all high bits zero. Leaf keys are integers <k (not arbitrary
unused d-bit patterns). Critical positions are <d, fitting e bits because
2^e>=d+1. The feasible branch has d>0 and e>0. No timestamp, color, rank,
parent pointer or separate validity bitmap exists.

**Lifetime sizing, including boundaries.** Put h=log2 N>=1. For all k,
d<=h; e<=d<=h (also true at d=0); a<=h+1. Therefore

```
b = 4d+4e+13a+25 <= 21h+38 <= 59h.
```

If no M fits, k-2<8b, including the guarded k<=2 cases. Hence
N<16b+4<=944h+4<=948h<1024h. Thus NL/(1024h)<L and its floor is <=L-1.
The direct branch meets H0, rather than merely an asymptotic surrogate.

If M fits, doubling fails: 4Mb>k-2, so M>(N-4)/(8b). Feasibility forces N>=8
(N<=6 has k-2<=1 and b>=25). For N>=8,
1024(N-4)>472N. Consequently

```
M > (N-4)/(472h) > N/(1024h),
ML > NL/(1024h), and ML>=L-1 because M>=4.
```

These strict real inequalities imply the required integer floor bound. No
evaluation of h or H0 is part of operation. As a manual arithmetic example,
N=16384,k=8192 gives a=15,d=13,e=4,v=68,b=288,M=8, as in architect round 3.
This is dimension arithmetic, not a measurement or universal verification.

## 3. PROGRAM: typed finite bit-level source and local operations

The ordinary program P is the following typed source, including the routines
in sections 3-5 and the actual return tail in section 8. The executable
model-level spelling is experiments/endurance_compact_reference.py: prepare,
ColdCompact's command methods, and their finite helpers. The separate
source_receipt function is outside P and unreachable from its INIT/QUERY/FLIP
call graph; it is a read-only worker syntax/hash helper, not an extra machine
input port. The receipt hashes the complete reference file, including that
helper, so identity checking never hides omitted source. It is
fixed source, parameter-independent, has no generator, installed table,
interpreter, external algorithm service, recursion or history input. Python
is a reference for the logical operations and explicit persistent ports;
neither its heap, object fields nor automatic function-stack disposal is the
ordinary-machine S certificate. The following register assignment and bit
loops are the ordinary realization. No claim depends on Python heap fitting S.

Types: Bit; fixed local vectors Word[w]; local NodeBuffer[v]; Path[q+1][4q].
All bounds are public parameter functions. Words have zero padding. For every
routine all live variables occupy the bank in section 7, including argument,
return and arithmetic temporaries. Return addresses use the fixed control bank;
local results are placed in prescribed result slots, not in a growing stack.
There is no implicit local list/string/transcript of variable length.

Primitive bit actions are: read/write one private tape bit; read one current
immutable parameter/input bit; move a private/input head by one cell; Boolean
operation on inventoried bit latches; conditional branch; READ(address) to
one latch; WRITE(address,latch); emit one bit/ACK on the write-only owner port.
Address/head/PC bits are mutable space. Private tape access can be realized
by moving its counted head to the addressed bit, not by treating word indexing
or arithmetic as a unit-cost ordinary instruction. Persistent READ/WRITE
remain exactly the frozen indexed one-bit probes.

The word notation below is a finite abbreviation, with these complete local
bit-loop realizations; every parameter width is finite:

```
ZERO(z,w): for t=0..w-1: z[t]=0.
COPY(z,x,w): for t=0..w-1: z[t]=x[t].
EQ(x,y,w): equal=1; for t=0..w-1: equal &= NOT(x[t] XOR y[t]).
CMP(x,y,w): scan t=w-1 down to 0; at first differing bit return x[t]<y[t];
             return false if none.  Counter decreases on every visit.
ADD(z,x,y,w): carry=0; for t=0..w-1:
  u=x[t]; h=y[t]; z[t]=u XOR h XOR carry;
  carry=(u AND h) OR ((u XOR h) AND carry).
SUB(z,x,y,w): borrow=0; for t=0..w-1:
  u=x[t]; h=y[t]; z[t]=u XOR h XOR borrow;
  borrow=(NOT(u) AND (h OR borrow)) OR (h AND borrow).
INC/DEC: ADD/SUB the local constant-one vector, with explicit carry/borrow.
HALF(z,x,w): for t=0..w-2: z[t]=x[t+1]; z[w-1]=0.
SHIFTLEFT(z,x,w): scan t=w-1 down to 1: z[t]=x[t-1]; z[0]=0.
MUL(z,x,y,w): ZERO(z,w); COPY(temp,x,w);
  for t=0..w-1: if y[t]: ADD(z,z,temp,w); SHIFTLEFT(temp,temp,w).
BITLENGTH(x,w): scan downwards; return first set position+1, or zero.
READWORD(z,address,w): ZERO(z,w);
  for t=0..w-1: latch=READ(address+t); z[t]=latch.
WRITEBUFFER(address,z,w): for t=0..w-1: WRITE(address+t,z[t]).
```

No ADD/SUB operation is allowed to underflow/overflow at a used logical
address; widths chosen below include the comparison trial values. Aliasing
z=x is safe for the displayed carry/borrow direction; CMP/bitlength never
modify their operands. General multiplication uses distinct product/temp
operands and width 6q, so even a trial dimension <=136qN and NL+2 fits without
truncation. Integer half computes floor, not a real-valued operation.

Every vector addressing expression is evaluated using those local routines.
For loops have inventoried counters; they are not source unrolling. All
source parameter/input heads are bounded by the finite current encoding.
Parsing the usual canonical binary public numeral encodings (no leading-zero
padding, fixed finite delimiters) counts bits and builds a word by shifts;
first determine their lengths by scans of their finite delimited encodings.
Store N,k,L, compute their product plus one and its bitlength q, then prepare
the integer layout in section 2. The fixed source has a tiny public-parameter
case N=2,k=1,L=1: it directly writes that case's fixed derived parameter image
(q=2,a=2,d=e=M=A=T=0,v=12,b=51, fallback control offsets zero), rather than
executing generic multiplication. This is a constant finite branch in the same
program, with exactly the same logical direct-base behavior. It is neither a
parameter-growing policy table nor a second construction. During bootstrap,
t=max observed public
encoding lengths<=q suffices for parameter/head storage and 6t product words;
its workspace is reused when q is known. After installing the derived table,
explicitly initialize/zero the reserved nonparameter banks and erase bootstrap
temporaries with one sequential private sweep; this work is included below.
The same tape is then reused, with actual RETURN sweeps after commands.
The encoding delimiters carry no
history. No logarithm, real precision or search through programs is used.

The fixed source has fewer than 4096 control clauses even after substituting
these fixed bit-loop bodies (loops remain loops); inline branch selectors and
calls are finite ordinary control, not parameter-growing CODE. A binary PC of
32 bits is ample. All immutable source plus fixed labels can be provisioned in
less than 2^20 bytes: the exact reference byte length is in the receipt, and
the normative source is sections 2-5, 7-8 of this finite document with the
above bit loops substituted once per fixed operation type. No parameter-sized
object code is stored. This is a finite source/code bound, not a paid runtime
compiler or a new microcoded execution kernel. A direct finite control with
local tape heads implements the displayed loops. Its immutable program-head
representation, if used instead of wired finite control, is <=32 bits and is
included below; the canonical program head is at its fixed home.

## 4. Tree routines: full bodies and literal persistence

In the code below, a retained path contains only current-tree ancestor IDs
and the child field (0 or 1) taken, in root-to-parent order. ROOT and NEXT are
private result words. NODE is a reusable v-bit buffer; decoding/copying it
has no further DATA acquisition. An effective node read acquires **all v
bits** using READWORD, including padding and flags. When box.valid=1, exactly
its named base field is replaced by box.value. Other fields remain base
fields. Publication asserts valid on promised paths, not a free DATA read.

```
PREFIX(bank,roots):
  lo=0; hi=(M if roots else A)
  while lo<hi:
    m=floor((lo+hi+1)/2)
    address=(root(bank,m-1)+a if roots else node(bank,m)+nodevalid_offset)
    flag=READ(address)
    if flag: lo=m
    else: hi=m-1
  return lo

ROOT_AT(bank,t):
  if t==0: return null
  READWORD(ROOT,root(bank,t-1),a); return ROOT

GET(bank,u):
  READWORD(NODE,node(bank,u),v)
  decode type/key/crit/base fields; if boxflag: apply its single override
  return effective fields and boxflag in assigned buffer/result slots

FRESH(bank,next,final_effective_node):
  u=next+1
  build FINAL[v] entirely in private buffer:
    final type/key/crit/left/right/parity, nodevalid=1,
    boxID=0, boxvalue=0, boxvalid=0
  for t=0..v-1 except nodevalid: WRITE(node(bank,u)+t,FINAL[t])
  WRITE(node(bank,u)+nodevalid,1)
  return (u,u)                           # result ID and new frontier

CHANGE(bank,next,u,field,value):
  effective,full=GET(bank,u)
  if NOT full:
    stage BOX[2+a+1] = (field,value,valid=1), completely final first
    WRITEBUFFER(node(bank,u)+boxID_offset,BOX,a+3)   # flag last
    return (u,next,false)
  replace requested field in private effective fields
  (replacement,next)=FRESH(bank,next,effective_with_change)
  return (replacement,next,true)          # fresh box is EMPTY

LOOKUP(bank,root,key,retain):
  u=root; depth=0
  while u!=0:
    n=GET(bank,u)
    if n.type==leaf: return (u,n.key,n.parity,depth)
    side=key[d-1-n.crit]                  # local key bit, not DATA
    if retain: path[depth]=(u,side)
    depth=depth+1; u=(n.right if side else n.left)
  return (0,0,0,depth)

SPLICE(bank,root,key,t):
  u=root; depth=0
  while true:
    n=GET(bank,u)
    if n.type==leaf OR n.crit>=t: return (u,depth)
    side=key[d-1-n.crit]; path[depth]=(u,side)
    depth=depth+1; u=(n.right if side else n.left)

ATTACH(bank,next,root,replacement,depth):
  while depth>0:
    depth=depth-1; (parent,side)=path[depth]
    (replacement,next,copied)=CHANGE(bank,next,parent,side,replacement)
    if NOT copied: return (root,next)
  return (replacement,next)

UPDATE(bank,root,next,key):
  if root==0: return FRESH(bank,next,leaf(key,parity=1))
  (u,candidate,parity,depth)=LOOKUP(bank,root,key,true)
  if candidate==key:
    (replacement,next,copied)=CHANGE(bank,next,u,2,1-parity)
    if NOT copied: return (root,next)
    return ATTACH(bank,next,root,replacement,depth)
  t=first position from MSB where candidate and key differ  # t<d
  (subtree,depth)=SPLICE(bank,root,key,t) # overwrite path with splice ancestors
  (leafID,next)=FRESH(bank,next,leaf(key,parity=1))
  side=key[d-1-t]
  (branchID,next)=FRESH(bank,next,internal(t,
    left=(subtree if side else leafID),right=(leafID if side else subtree)))
  return ATTACH(bank,next,root,branchID,depth)
```

Fresh leaf/branch irrelevant fields are zero. Every allocation, including
temporary detached replacements and garbage copies, advances next exactly
once and publishes nodevalid. A later CHANGE on an empty box writes only that
box. No base field is overwritten; no previously filled box is refilled.
Fresh zero box writes count traffic; their later final writes change each bit
at most once. Fresh final buffers, including root and box buffers, are complete
before any DATA write belonging to that publication. The only read-modify-write
outside node-copying is the explicit base read before a fold/direct flip.

## 5. Command bodies and the complete M-transition schedule

INIT receives only current x. For i=0..k-1 read input bit x[i] into a one-bit
latch and WRITE(i,latch), once each. All other DATA was initially zero. No
arena/control read or write occurs during INIT, even on the trie branch.
Run RETURN(ACK). The parameter table was computed without x and remains
parameter-only. An initialization call is promised once, not repeatedly.

```
QUERY(i):
  answer=READ(i)
  if M==0: emit(answer); RETURN(noACK)
  bank=READ(ROLE)
  j=PREFIX(bank,roots=true)
  r=ROOT_AT(bank,j)
  (u,key,p,_)=LOOKUP(bank,r,i,false)
  if u!=0 AND key==i: answer=answer XOR p
  if j<M/2:
    started=READ(STARTED)
    if started:
      old=1-bank; oldr=ROOT_AT(old,M)
      (u,key,p,_)=LOOKUP(old,oldr,i,false)
      if u!=0 AND key==i AND u>2Cj: answer=answer XOR p
  emit(answer); RETURN(noACK)

FLIP(i):
  if M==0:
    latch=READ(i); WRITE(i,1-latch); RETURN(ACK)
  bank=READ(ROLE); old=1-bank
  j=PREFIX(bank,roots=true); r=ROOT_AT(bank,j)
  next=PREFIX(bank,roots=false)            # NEW flags only
  (r,next)=UPDATE(bank,r,next,i)
  stage ROOTBUF=(r,flag=1) entirely in private bits
  WRITEBUFFER(root(bank,j),ROOTBUF,a+1)     # pointer first, flag last
  n=j+1
  if n<=M/2:
    started=READ(STARTED)
    if started:
      oldr=ROOT_AT(old,M)                  # no OLD prefix/allocator search
      for u=2C(n-1)+1 .. 2Cn:              # exactly 8 physical positions
        if READ(node(old,u)+nodevalid)==0: continue
        if READ(node(old,u)+type)==internal: continue
        READWORD(key,node(old,u)+key_offset,d)
        (active,candidate,p,_)=LOOKUP(old,oldr,key,false)
        if active==u AND candidate==key:
          value=READ(base[key]); WRITE(base[key],value XOR p)
  else:
    offset=(n-M/2-1)*2b
    for t=0..2b-1: WRITE(arena(old)+offset+t,0)
  if n==M:
    WRITE(ROLE,old)                        # including end of global epoch L
    if READ(STARTED)==0: WRITE(STARTED,1)   # actual write ONCE overall
  RETURN(ACK)
```

M>=4 is a power of two, so M/2 is an integer. First-half scan chunks cover
1..CM exactly: (M/2)*2C=CM. Second-half chunks cover exactly the entire OLD
arena: (M/2)*2b=Mb=T. They are disjoint contiguous ranges. Clearing may destroy
any OLD flag/root prefix; no routine uses it then. In the clearing half QUERY
does not even read OLD's final root, publication, node or allocator flags.
FLIP's second-half body performs zero OLD reads as well. STARTED/ROLE are
global control cells, not OLD arena cells. No command reads a readable wear
counter or catches an exhaustion exception.

At n=M the cleared bank becomes NEW with j=0,next=0; the just-completed bank
becomes OLD with final root in slot M-1. Exchange occurs only after all T
clearing writes. STARTED is set if this is the first exchange. There is no
final-epoch special case and no saved epoch ordinal. The next command recovers
only the NEW prefix. Every path/node/buffer and local root is then cleared
before ACK. The first epoch scans an empty OLD: STARTED=0 skips its acquisitions
and folds; its scheduled second-half zero writes still count traffic.

## 6. Proof certificates: latest tree, paths, allocation, folds and correctness

All lemmas in this section are DERIVED / PENDING REVIEW; these are proofs for
attack rather than imported DSST theorem labels.

### 6.1 Latest-root and critical-position induction

At a completed boundary the latest NEW root reaches an external binary tree:
each touched key has exactly one active leaf, all leaves are distinct keys,
internal nodes have two nonnull children, effective child pointers form a
tree with unique current parents, and leaf parity equals that key's occurrence
parity in NEW's j commands. A zero parity leaf remains present. Internal
critical positions strictly increase along a path. All descendant keys of a
node with position t agree on positions <t, and its left/right descendants
have bit t respectively zero/one. Consequently lookup of an existing key
reaches exactly its active physical leaf, in at most d internal visits plus
one leaf. Nodes not in the latest tree may remain published as garbage.

The empty tree starts the induction. First insertion adds one leaf with parity
one. An existing-key update changes exactly its one parity field through
CHANGE. A box leaves its physical ID and its parent's child unchanged. A full
box is replaced by an empty-box copy with the requested parity already in
its base; its ancestors are handled by ATTACH. No child/key/position changes
except the single requested child replacement at each propagated ancestor.

For a missing key i, let z be the candidate leaf reached by lookup and t its
first differing position. Traversed ancestor positions <t agree with z and
i. A traversed position equal to t would send lookup to the bit-i side, so
its returned leaf could not first differ at t: thus equality is impossible
at the splice. SPLICE stops at a leaf or an internal position >t. The stopped
subtree's keys share z's bits through t (by the compressed-prefix invariant),
and therefore all have bit t opposite i. Its retained ancestor positions are
<t. The new branch at t places that whole subtree on its old bit side and
the fresh i leaf on the other. Its ancestors' prefix/branch clauses persist;
the old child's position, when internal, is >t. This proves the strict
positions and lookup invariant after the splice, including a root splice.
No rebalancing, rank update or inverse parent metadata is hidden in this step.

ATTACH processes retained ancestors bottom-up. At each step the original
unprocessed ancestor is still the current ancestor of the old subtree: nothing
above it has yet been written. Its unchanged sibling is disjoint from the
replacement subtree. A child box reattaches the replacement and ends the
propagation; all higher ancestors retain their physical IDs. A copied parent
contains that replacement and its unchanged sibling, then itself becomes the
pending replacement for the next original ancestor. Unique current parents,
acyclicity and the routing prefix relation survive. A root-level copy changes
only the private root until publication. Original path IDs above the processing
frontier remain valid; no traversal of a stale copied parent is needed.

Historical NEW roots may cease to encode historical trees when boxes are filled.
They are never traversed. The last root at exchange reaches the current correct
tree; no OLD node/box/root is modified until every active OLD leaf is folded.
Thus OLD's final root and effective fields are frozen throughout its first half.

### 6.2 All-prefix allocation, including transient peaks

Let D_j be total allocations in NEW through its jth update, counting detached
copies, all leaves/branches and garbage. Let Phi_j be the number of filled
boxes on the current latest live tree. Initially both are zero. New structural
nodes have empty boxes. A chain copying c full live nodes removes exactly c
filled live boxes and creates c empty replacements. If it ends at an empty
live box, that adds one filled box; if it reaches a private-root replacement,
there is no added box. Therefore, for this **one external field modification**,

```
copies + Delta Phi is 1 (empty-box termination) or 0 (private-root termination).
```

An existing-key update has no structural allocations and one such modification.
A nonempty-tree insertion has two structural allocations and at most one such
modification (its parent child); root insertion has none. First insertion has
one structural allocation. Hence in every update

```
Delta D + Delta Phi <= 3,
D_j + Phi_j <= 3j, and D_j <= 3j.
```

The chain calculation refers to the uniquely determined final reattachment
tree even while intermediate copies are detached. Construction has no rollback,
extra speculative allocation or reclamation: D only increases during this
command, so **every intermediate allocation prefix is <= its final D_j <=3j**.
This is not using a boundary bound as a promise of low transient occupancy;
the monotonicity and exact complete chain account for each transient node.
The potential proof can first be carried out with an unbounded conceptual
allocation tape. It proves that no actual allocation ever requests ID>A,
since j<=M gives 3j<=3M<4M=A; thus the finite run and proof are not circular.
Worst-case allocations within one update can be d+3, exceeding C. C=4 is
total arena capacity, never a per-command allocation limit.

### 6.3 Cold prefix and publication certificate

Fresh node IDs are consecutive, and each nodevalid is set last, once. No
NEW clearing occurs within the epoch. Thus its node flags are exactly a prefix
of ones of length D_j. Command root flags are a prefix of ones of length j,
with the current root in record j-1. PREFIX's lo/hi invariant is
0<=lo<=true prefix<=hi<=limit. Each branch shrinks the range strictly and
returns that prefix, with <=ceil(log2(limit+1))<=q+1 reads. No outside stage
number or hidden allocator is supplied. Allocated but garbage nodes remain
inside the prefix and are included by recovery and in D_j.

The root pointer/flag is published only after UPDATE has reattached all copies.
Maintenance cannot change NEW. A completed-boundary observer therefore gets
its correct latest tree and derives the correct post-maintenance j. No mid-
command crash/query claim relies on a temporarily old root/partial box.

### 6.4 Exactly-once physical folding and cutoff

For each key in frozen OLD, let u(key) be its unique active physical leaf ID
and p(key) its frozen parity. Garbage leaf copies may share its immutable key.
Scanning garbage performs lookup but fails active==scanned. Internal/unpublished
slots are skipped. The sole scan of slot u(key) succeeds and folds exactly
p(key), including an explicit same-value write when p=0. No key is deleted,
so no key to fold is discoverable only through an unrepresented dead copy.
Each physical ID is visited once by the disjoint scan chunks. Therefore after
the jth first-half command exactly the keys with u(key)<=2Cj have been folded;
all other OLD contributions are still pending. Every active leaf has ID<=A,
so by j=M/2 all contributions are in base. OLD acquires no new active ID.

Zero-parity leaf retention is essential for this enumeration invariant, and
it is unconditional in UPDATE. No sorting, subtree count, next-key oracle or
timestamp is used. OLD lookup always uses ROOT_AT(old,M), never an earlier root.

### 6.5 Logical induction through all M transitions and L epochs

Let X_i be raw x_i XOR all completed FLIP_i occurrences, B_i the actual base
bit, Pnew_i the latest NEW parity (zero if key absent), and PendingOld_i be
p_i exactly for an OLD active ID above the current scan cutoff in the first
half, zero otherwise. The boundary invariant is

```
X_i = B_i XOR Pnew_i XOR PendingOld_i.
```

INIT makes B=x, both trees empty, STARTED=0. UPDATE toggles precisely Pnew_i
for the requested key, so the logical value changes precisely at that key.
Each successful fold toggles B_key by p_key and removes the identical pending
term; the XOR identity is unchanged. Unsuccessful scans change nothing. At
j=M/2 no pending terms remain. Every second-half clearing write touches only
obsolete OLD, so the identity is preserved with PendingOld=0 despite damage
to OLD's flags/tree. At j=M, OLD is zero; exchange reinterprets NEW's final
parities as the wholly pending next OLD contribution at cutoff zero and the
zero cleared arena as empty NEW. B does not change at exchange. STARTED=1
allows that contribution to be read. The same identity now holds at j=0 of
the next epoch. This proves all M transitions, not just a steady-state interior.

Induct over global epochs externally in the proof, for epochs 1..L. The
program stores no such ordinal. At the final exchange its new current arena
is empty and its old final root contains the last epoch parity. QUERY at j=0
includes that parity and base; it halts correctly for arbitrarily many terminal
queries without any write. At other prefixes QUERY's key equality test avoids
using a candidate leaf for an absent key, and the exact ID cutoff supplies the
pending term. This proves the all-raw/all-history logical contract.

### 6.6 Every-cell wear, including all undone changes

| Physical cells | Changes in one global epoch | Changes through L epochs |
|---|---|---|
| Node immutable/base fields, node publication in NEW | Initially zero, final base write once; at most one 0->1 | <=L |
| Node box ID/value/publication in NEW | Initially written zero by FRESH; final box written once; at most one 0->1 | <=L |
| Root pointer/publication in NEW | Final staged record written once; at most one 0->1 | <=L |
| Entire obsolete OLD arena | One scheduled zero write per cell in second half; at most one 1->0 | <=L |
| base[i] | Epoch 1 no fold; later at most one fold for key i | INIT <=1 plus <=L-1 folds, so <=L |
| ROLE | One complement at each exchange, including epoch L | exactly L for complete ML history |
| STARTED | Written one only when its read value is zero | <=1 |
| Surplus unused cells | Never written | 0 |

A given arena is NEW in one global epoch and OLD in the next. It is never
both filled and cleared in the same global epoch. Its clearing starts only
after its whole frozen contribution is folded. The initial empty arena still
receives same-value clearing writes. No algorithm instruction temporarily
sets and undoes a base/flag/box bit outside this ledger. Full-node copies leave
the source untouched, and empty-box overrides are single-use. Thus the table
counts actual intermediate writes, not just endpoint Hamming distance. INIT
does not change control/arena cells. ROLE's L, rather than L-1, is allowed by
the initially-zero wear convention. Every promised write therefore stays
within individual wear L; legality never depends on readable wear feedback.

## 7. Exact simultaneous private-space certificate

This is a concrete allocation of the entire ordinary mutable machine, not
merely a bound on semantic keys. The model's earlier suggested envelope was
a capacity sketch; this program uses the following allocation inside the
unchanged S. Its widened words handle small-parameter dimensions without
pretending v or b always fits q bits.

| Bank | Exact capacity | Typed assignment |
|---|---:|---|
| Parameter-only canonical table | 96q | 32 words of 3q bits |
| Scalar work/results | 64q | 16 words of 4q bits |
| Heads and private/probe addresses | 64q | 10 slots of 6q bits plus 4q bits |
| Finite control | 256 | PC[32], program head[32], 8 return labels[16], 64 condition/latch bits |
| Retained path | 4q(q+1) | q+1 frames of 4q; ID in q bits, direction in one bit, remaining zero |
| Reused node/publication buffers | 8v | eight vectors, each exactly v bits |
| Arithmetic work | 32q^2 | five banks of 6q*floor(q) bits plus 2q^2 slack |
| Cleanup tail counters | 16q | two 6q-bit sweep/home words plus 4q fixed tail bits |

The parameter table contains N,k,L,q,a,d,e,v,b,M,A,T, arena0/arena1 offsets,
control offsets (zero sentinel in fallback), the nine node offsets, frame
stride, fixed buffer offsets, scratch extent, and branch indicator: <=32 words.
If a packed table would exceed that list, offsets are recomputed from the
displayed identities rather than added as extra variables. Concretely store
N,k,L,q,a,d,e,v,b,M,A,T,arena0,arena1,ROLEaddr,STARTEDaddr,
keyOff,critOff,leftOff,rightOff,parityOff,nodevalidOff,boxIDOff,boxvalueOff,
boxvalidOff,pathStride,pathExtent,bufferStart,mutableExtentLo,mutableExtentHi,
branch,padZero. Exactly 32 entries. The private extent occupies TWO 3q words:
its low/high halves are assembled in a counted 6q head/arithmetic word when
needed. Zero/one constants come from the fixed source, without table slots.
ROOT offsets are computed, never stored as a third table. Each single entry
fits 3q: q=2 forces N=2,L=1 and b=51<2^6; for q>=3,
34q<2^(3q), v<=8q, and actual arena extents/addresses are <N<=2^q.
The abstract total private extent <=256q^2 fits 6q head bits since
256q^2+1<=2^(6q) for q>=2. Parameter table slot indexing also fits those heads.

The sixteen scalar words are assigned, with reuse only after last use, to:
request i; bank; OLD selector; j/n; NEW root; frontier; OLD root; scanned ID;
scanned key; returned active ID; returned parity/query answer; path depth;
pending replacement; first differing position; lookup candidate key; general
loop/width index. Node type/critical position/children/key/parity/box state
remain in node buffers, not simultaneous extra scalar objects. Boolean
flags/latches occupy the 64 control bits. Routine result tuples are assigned
to these same words/buffers at return; they are not separately allocated.

Heads/addresses: private tape source/destination, current input, public
parameters, DATA probe, path frame, node buffer, arithmetic bit, parameter
table, cleanup/homing. There are at most ten, each stored in its 6q slot;
every physical head position and effective bit address is included. The
program head and PC are in the control bank. Direct finite control need not
use a program tape head; its reserved word stays canonical zero. The raw input
tape belongs to the current immutable interface; its head/index and its one-bit
latch are counted, and its content is never copied into an unbounded local tape.

Buffer assignment: B0 acquired full node; B1 staged final fresh node/box/root;
B2 saved effective pending node; B3 saved candidate leaf; B4 saved splice node;
B5 field/probe staging; B6,B7 node/field return staging. Wide arithmetic words
remain in the arithmetic bank, never in a possibly narrower v-bit node buffer.
No routine needs more
than these eight simultaneously. CALL arguments are views of these buffers,
not deep copies. ATTACH is iterative. Its pending child is a scalar; its
original ancestors are in the fixed path bank. QUERY/fold lookups use no path
stack and may reuse B0/B3; after UPDATE the retained path can remain dirty
until RETURN but is never extended for maintenance.

Arithmetic bank has five 6q-by-q work rows (30q^2 bits) and 2q^2 slack.
MUL uses operand copies, accumulated product, shifted multiplicand and loop/
carry lane within these five rows. ADD/SUB/CMP/offset computation use a subset.
For the serial-head realization, physical head coordinates in the 64q bank
are charged in addition to controller-maintained position words. Those mirrors
occupy the arithmetic bank, not uncounted hardware: reserve the first 6q word
of each of the five rows for the five main operands; use ten of the remaining
6q words for head mirrors when q>=3. There are 5(q-1)>=10 such words, so this
fits before using the 2q^2 slack. At q=2 the tiny fallback activates at most
five serial heads (public input/parameters, two private operand heads and
cleanup); the five remaining 6q words hold these five mirrors. DATA's indexed
probe-address word needs no extra physical DATA head. Inactive path/buffer/
table head selectors are zero; their existing coordinate capacity is still
counted. Mirror words are cleared/homed too. Thus neither an implicit position
oracle nor a stored position duplicate escapes the simultaneous allocation.
No multiprecision library state, Python long, host stack or previous probe
transcript is part of P. The widest result used is NL+2 or a dimension trial;
6q is enough. Offset/path calculations are nested fixed calls using these
rows and the heads. Result operands that must survive GET reside in a named
scalar/buffer, not a sixth hidden work row.

The fixed call graph has maximum depth eight. Its deepest route is command ->
UPDATE -> ATTACH -> CHANGE -> FRESH -> stage/word -> address arithmetic -> bit
loop; there is no recursive call. Return labels are 16 bits and fit 128 fixed
bits. PC/program-head bit lengths <=32; branch/select/carry/borrow/probe/output
and tail latches together fit the remaining 64 bits. Conditional flags can
be reused after their branch, and no call carries an entire host activation.

Even reserving every bank at once (including the dirty path during a fold,
all eight buffers and all arithmetic/tail lanes) gives

```
B = 96q+64q+64q+256+4q(q+1)+8v+32q^2+16q
  <= 36q^2+284q+304 <= 256q^2.
```

The last inequality is equivalent to 220q^2-284q-304>=0, which is 8 at q=2
and increases strictly for integer q>=2. This includes the small fallback;
no removal of inconvenient heads/control at small q was used. Preparation
does not add its bootstrap storage to B: it reuses table/arithmetic/head banks
and has no path/node/live input stack. Expanding initial public word lengths
as they are scanned never exceeds their final capacities (lengths<=q).

## 8. Actual clearing, homing, halting and ordinary preparation work

RETURN is part of P, not a reset primitive and not newly allocated zero scratch.
Reserve the table bank unchanged, and a fixed tail pocket comprising its
cleanup words/head words and control latches. The reference _wipe loop actually
overwrites its persistent reference scratch; P additionally executes this
ordinary tail to clear all mapped scalar/argument/control information:

```
RETURN(kind):
  # QUERY has emitted its one-bit answer already. kind is one counted latch.
  # INIT/FLIP have finished all persistent maintenance; no DATA access follows.
  set probe/output/carry/borrow/argument latches to zero (except kind)
  HOME current input head to position 0, one move and decrement per position
  HOME public-parameter head likewise
  set sweep=0; private tape head=0
  while sweep<B:
    if this position is neither parameter table nor tail pocket:
      overwrite its bit with zero
    move sweep/private head one position right; increment counted sweep
  HOME private tape head; HOME every separate frame/buffer/arithmetic/table
    head and DATA-address register to zero using their counted positions
  clear tail operand/sweep words bit by bit with the fixed tail control;
    head/operand values not needed again are zeroed, never just abandoned
  HOME program head to fixed home if it is used; erase return-label slots
  # All variable-width head/address/pocket words are already zero by loops.
  final fixed-control transition: clear kind and remaining fixed tail latches,
    set PC=READY; emit ACK iff kind was ACK; return to caller
```

HOME of a physical tape head is a loop: test its counted nonnegative position,
if positive move one cell toward its fixed origin, subtract one, repeat. The
private tape head can sequentially visit banks; a private index is moved using
the same address/head lanes, so no extra implicit head position appears.
The tail pocket consists of **counted local registers**; a bit is zeroed after
its last use, with the final fixed control transition erasing only the fixed
finite remaining control/latch bits. No q-dependent vector is erased by that
last transition. This is standard irreversible local bit
work, which the contract permits. Return labels may be cleared after the last
call has completed; the tail uses fixed jumps, not a return address requiring
restoration after erase. PC=READY is the same encoding on all returns. The
parameter table is identical to its initial parameter-only image, not rebuilt
from DATA. Repeated commands reuse the same bounded workspace. The emitted
query answer or ACK supplies no readable output position/state to P.

One can realize the sweep with a single serial private head and counted 6q
position word. The head walks forward once and then homes; individual tail
registers are a fixed finite set of bit lanes. If these are serialized on the
private tape instead, preserve the two counted 6q tail words until their last
use and zero them by fixed bit loops; there is no self-erasure of the only
loop counter followed by assuming a free return. The final finite transition
homes the finitely many register-lane selectors and READY control together.
No DATA is written during any clearing/homing operation.

**Halting.** Prefix search reduces a finite interval. LOOKUP/SPLICE follow
strictly increasing positions and visit at most d+1 nodes. A first difference
exists between distinct d-bit keys, so its scan stops at t<d. ATTACH decreases
depth on every visit. FRESH loops over v bits; publication never waits for
feedback. Maintenance visits eight positions or exactly 2b bits. Preparation
doubles M at most q times and uses bounded word scans/multiplication. RETURN
has finite B/head-distance/word-width loops followed by a fixed tail. Wear
legality was established in section 6, not discovered through failure handling.

**Ordinary bit-work accounting for prep+INIT.** This coordinate includes local
head moves, bit reads/writes, latches/control transitions, input/parameter reads
and DATA probes. No ordinary command-time theorem is asserted. A conservative
implementation uses serial private tape movements of at most B<=256q^2
between bit accesses and at most two such moves per bit action. Fixed finite
control and carry/cursor bookkeeping add <=128 bit actions per scan position.
For a word width <=6q, the displayed primitive loops can be scheduled with:

| Preparation operation | Ordinary bit-work upper bound / count |
|---|---|
| Copy/zero/compare/add/sub/shift/bitlength of a <=6q word, including loop/address/control work | <=2^16 q^3 per whole word routine |
| Parameter parsing, q computation, private extent, offsets, and power-of-two sizing | <=128q such word routines, plus <=32q public bit reads |
| INIT one raw coordinate, increment/address/staging/probe and control | <=2^20 q^3 per coordinate |
| Initial private initialization/erasure plus INIT's volatile sweep/homing/final tail | <=2^20 q^5 + 8Nq |

Schedule each word routine as serial scans: carry/borrow/comparison latches
are fixed lanes; operands occupy fixed lanes of the same private tape block.
At a position there are at most four operand/latch transfers of distance <=B.
Maintain the charged mirror by ripple increment/decrement during each
monotone transfer. Across r moves its binary bit flips/scans cost <=2r+12q;
including the physical moves gives <=3r+12q, not a free position counter.
Hence those transfers and <=64 other local/control operations cost at most
12B+48q+64<=3120q^2 for q>=2. A pass has <=6q positions; at most three full
passes (setup/operation/homing) cost <=56160q^3. Fixed setup/homing of selectors
costs <4096q^3, so the whole routine is <2^16q^3.
The counter/carry bit scan is contiguous: it does not pay a B-distance movement
for each counter bit. Cursors are moved incrementally;
there is no multiplication/address recomputation per tape bit. A routine call
has fixed argument-location selectors, so it does not recursively invoke word
arithmetic merely to find its next operand. Each MUL makes <=6q ADD calls and
<=6q SHIFT calls. Preparation uses two such multiplications: NL and q^2.
The path extent follows from q^2+q, and the scratch extent from its displayed
polynomial in q and v. Sizing maintains the trial capacity by doubling the
previous one, so it never multiplies M*b on each fit test: start capacity=8b
at M=4, trial=2*capacity; after acceptance double M and capacity; at exit
T=capacity/2. Thus at most q fit iterations cost <=16 routines each, the two
multiplications cost <=24q, parsing/bitlength/constant offsets <=32q, and
table installation/setup <=32q: in total <=104q<128q routines.

The bound covers writing the canonical 32-entry table as well: its constants
and offsets are staged in named arithmetic rows then copied with these scans.
Public scans require at most 3q bits (and fixed delimiters); 32q bounds them.
Thus actual preparation <=2^23q^4+32q. A sequential INIT iteration requires
only local input acquisition, WRITE at current incremented i, and <=16 scans
of <=6q counters/latches: <=2^20q^3 per coordinate suffices. A sequential
cleanup iteration needs one zero and increment of its 6q pocket word, with
at most two B-distance transfers (including mirror work) and <=32q further
bit operations, costing <=1536q^2+56q+1<=1024q^3 for q>=2.
At most B<=256q^2 iterations plus homing and fixed-tail scans yield
<=2^19q^5+8Nq for RETURN. The initial private initialization/erasure costs
another <=2^19q^5. Consequently a valid crude complete bound is

```
prep+INIT <= 2^23q^4 + 2^20 kq^3 + 2^20q^5 + 40Nq.
```

For N>=4, q>=3. The ratio of this bound to 2^20Nq^6 is

```
8/(Nq^2) + 1/(2q^3) + 1/(Nq) + 40/(2^20q^5)
 <= 2/9 + 1/54 + 1/12 + 40/(2^20*3^5) < 1.
```

For N=2,L=2, q=3; the same ratio is at most
4/9+1/54+1/6+40/(2^20*3^5)<1. Both feasible and fallback branches are covered
by the same bound, including a full-envelope sweep. No favorable large-q
assumption is needed on a fallback history.

For the remaining case N=2,k=1,L=1,q=2, the finite tiny parameter branch in
section 3 makes generic multiplication/sizing unnecessary. Parse its four
public value bits and delimiters; write the 32 fixed six-bit table entries
(192 bits) from constant source; allocate the same at-most-1016-bit private
envelope; read the one raw bit and WRITE(0,bit); execute RETURN. Table staging
and serial copying/homing cost <=2^20 steps. Initial private initialization
plus RETURN cost <=2^20*2^5+8*2*2=2^25+32. Fixed parsing/INIT/control overhead
is <2^20. Their sum is <2^26<2^20*2*2^6=2^27. Every zero-width field/control address is
inert in this case. It has H=0 but arbitrary terminal queries still execute
and clean correctly. This explicit constant-source case removes the sole
small-q gap in the generic arithmetic bound.

The bit-work bounds are proposed serial schedules and mathematical operation
counts, not measured Python time or a claim that Python implements these head
movements. The finite serial schedule and tiny public-parameter branch are
the acceptance certificate to attack; no asymptotic big-O substitutes for it.

## 9. Literal numerical probe certificate

On all primary parameters q>=2 and a,d,e<=q, so v<=5q+6<=8q,
b=4v+a+1<=34q. One complete lookup costs at most
K=(d+1)v<=8q(q+1)<=12q^2, including all rereads/padding. Prefix recovery costs
<=q+1 per prefix, each root acquisition <=a<=q, and no DATA snapshot is free.

QUERY: one base bit, one ROLE bit, <=q+1 NEW root flags, <=q NEW root bits,
<=K NEW lookup, at most one STARTED bit, <=q OLD root bits and <=K OLD lookup.
Thus R_Q<=2K+3q+4<=32q^2. The direct branch costs exactly one read.

FLIP reads: entry ROLE/NEW root prefix/node prefix/root and global flags/root
words cost <=4q+6; UPDATE has at most two lookup/splice paths plus one change
chain, totaling <=3K; the eight scanned slots cost <=8(d+3+K), including
separate scanned nodevalid/type/key and the later lookup acquisition, and each
successful fold's base read. Including the OLD root once gives the conservative

```
R_F <= 11K+12q+30 <=192q^2.
```

Its writes: at most d+3 full fresh allocations (including structural nodes
and every copy), at most one ending empty box, one root record, <=8 base fold
writes, one 2b clearing chunk, one ROLE and one STARTED write. Folding and
clearing do not coexist, but bounding both gives a valid separate worst maximum:

```
W_F <= (q+3)8q+(q+3)+(q+1)+8+68q+2
     =8q^2+94q+14 <=64q^2.
```

For q>=2 the final inequality follows at q=2 and its slack grows. No traffic
is assigned to local copying from an already charged buffer. Every acquisition
of a node/base/root/flag uses READ; every persistent bit write uses WRITE.
Prefix searches do not acquire any unpriced tree directory. The direct branch
has R_F=1,W_F=1, so the same numerical maxima cover it. Adding the independently
bounded worst reads and writes gives <=256q^2, far below 2^20q^3.

## 10. Manual traces, falsification ledger and validation scope

These are handwritten finite examples for proof review, not executed probes,
measurements, theorem evidence or a workload grid. Node allocation IDs below
are relative to a sufficiently large feasible NEW arena.

1. Empty -> FLIP key 0: fresh leaf ID1 parity1, root1, D1=1,Phi1=0.
   FLIP key 0 again: its empty box records parity0, root1, D2=1,Phi2=1.
   FLIP key 0 again: ID2 copies leaf with base parity1, empty box, root2,
   D3=2,Phi3=0. Scanning garbage ID1 in frozen OLD fails activeID==1;
   scanning ID2 folds its parity exactly once.
2. d=3 keys 000 then 100: ID1 leaf000; ID2 leaf100 and ID3 branch at
   position0, children1/2, private root3. Insert010: splice at left leaf,
   ID4 leaf010 and ID5 branch at position1 with children1/4; box of node3
   changes left to5. Effective root3 has path positions0,1 strictly increasing.
   D3=5,Phi3=1. No root history promise is used for record2.
3. Existing leaf with a full parity box, a parent with a full child box, and
   a root with an empty child box: copy the leaf with requested parity already
   in its base; copy its parent with the new leaf ID already in its base;
   fill root's empty box and stop. Two detached allocations become reattached;
   Delta Phi=-2+1=-1, copies+Delta Phi=1. Intermediate allocator increases
   monotonically; the next original ancestor is never a copied stale ID.
4. At j=M/2, cutoff=A. All active leaves have been folded, including zero
   parity ones. The next update writes OLD chunk [0,2b), possibly clearing
   node flags immediately. QUERY's branch j>=M/2 performs no OLD read and
   remains correct. At update M its final chunk ends at T-1 before ROLE flips.
   The next boundary has j=0 in zero NEW and full frozen OLD, so terminal QUERY
   includes the last epoch parity. ROLE's last change is charged even at L.

| Potential failure | Disposition of this attempt |
|---|---|
| Treat C as allocations per command | Rejected; all-prefix D+Phi bound and monotone transient count used |
| Read historical NEW roots as snapshots | Rejected; only latest NEW/final OLD traversed |
| Refill a box or change a copied base after publication | Rejected; requested change staged in final fresh base |
| Fold every physical copy / delete parity-zero key | Rejected; unique active-ID test and no deletion |
| Read OLD flags while clearing invalidates the prefix | Rejected; second-half code has zero OLD acquisition |
| Claim ROLE uses only L-1 changes | Rejected; exactly L exchanges allowed and counted |
| Count only semantic q-bit keys or discard locals as cleanup | Rejected; complete typed banks and explicit RETURN tail supplied |
| Ignore small-q preparation or assume dimensions fit q bits | Rejected; widened words, split extent and explicit tiny-case counts supplied |
| Python reference heap equals S / automatic return equals cold homing | Explicitly unclaimed; reference and ordinary program certificates distinguished |

Read/syntax checks only are authorized and performed for this delivery. The
--receipt path reads and AST-parses fixed source, never calls prepare/ColdCompact
or any command. There is no default simulation CLI. Should a root reviewer
need a logical check, first separately register COMPACT-CONTROL-1: exact source
hashes, deterministic named key sequences (e.g. <=64 commands each on two
feasible layouts), independent oracle and per-cell change/traffic recording,
CPU1/RAM256MiB/wall30s, at most 16 sequences, no sweeps/search; then smoke/code/
raw gates as appropriate. That is a proposal only; **none has been executed**.

## 11. Narrative, source standing and next gate

**Honest scoped sentence, HEURISTIC:** a single-use persistent trie can spread
binary storage wear while recovering every command cold and paying for every
bit read, write, root, fold and private controller location.

The strongest triviality objection remains that node-copying, compressed tries,
logs, folds and second-copy scheduling are occupied. This construction does
not claim to invent them or defeat equally accounted conventional adaptive
incumbents. Its proposed useful consequence is a deterministic joint lifetime/
probe capability under the frozen strict hidden-wear/cold contract, pending
attack of the supplied algorithm and numerical certificates. SOURCE OPEN is
not DELTA; novelty/useful significance and the model-level mission remain OPEN.
A general audience would probably call this a storage component, not a change
to computation on Turing's scale. No completed new-paradigm CAN/CANNOT pair or
terminal whole-repository win is claimed.

Next required gate: a fresh independent **actual** proof attack on this exact
source/receipt, including the full ordinary-program resource and return-tail
certificates, before any CAN/PROVED promotion. One named repair remains available
across this entire unit if that review finds a substantive defect. Root owns
review and commits. This bounded submission is not a stop of the research mission.
