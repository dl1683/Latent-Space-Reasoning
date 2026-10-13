# Root positive hypothesis: single-use modification boxes and physical-ID folding

Date: 2026-10-04. Rigor: HEURISTIC DESIGN SKETCH / CONJECTURE.
For genuine third-round steering only. New bit-probe model not yet registered
or design-admitted. No construction theorem, numeric run or CAN promotion.

## Source anchor and difference from the existing copied trie

DSST1989 sections2.3 and4.1 provide node copying with constant amortized space
per field modification and partially persistent red-black trees. A single tree
access path avoids stored inverse pointers. Section6 notes a write-once variant
except access pointers. Its usual red-black simplification overwrites colors;
that permission does not protect strict per-cell wear. These are occupied
techniques, not a new invention. Targeted reading: PDF pages10–12,23–26,37.

[Primary PDF](https://www.cs.cmu.edu/~sleator/papers/making-data-structures-persistent.pdf).

The proposed adaptation stores ALL mutable tree fields, including color and
occurrence parity, through a single-use modification box or a fresh node copy.
An append-only command-root log pays for the access-pointer exception.
Critically, fold by PHYSICAL NODE ID rather than a subtree rank count. Updating
rank counts along every insertion path would reintroduce logarithmic overhead.

## Candidate finite layout

N=2k. Preserve base[k], ROLE and STARTED. Let a=ceil(log2(N+1)),
d=ceil(log2 k), and candidate C=24. Choose the largest even M>=2 satisfying
2M(Cv+a+1)<=k-2, where v=d+3a+6. Two arenas each contain:

- A=CM node slots of v bits;
- M command-root slots, each a-bit pointer plus publication flag.

Each node has immutable key[d], base left/right pointers[a each], base color
and parity bits, a node-publication flag, and one box: field ID[2], value[a],
box-publication flag. Field ID selects left, right, color or parity; bit-valued
fields use a zero-padded value. Null pointer is zero; node IDs start at one.
Every fresh-node field is written directly from a private final buffer; base
fields are never overwritten. Each box is filled once, flag last. Copying reads
effective fields and installs their final values into a fresh zero node.

CONSTRUCTION DEBT: C=24 is a proposed budget, not a proved allocation bound.
Need exact arbitrary-k sizing, primary-domain/small-branch inequalities and
ordinary preparation/code/workspace bounds. No chosen constant is admission.

## Candidate tree updates and amortized allocation

During one arena epoch insert a key only on first occurrence; never delete it.
Repeated occurrences toggle its parity. Keep a deterministic red-black tree.
Use one-box copy-on-overflow for pointer, color and parity modifications.
The current root is private DURING a command. Publish its final value to the
next command-root slot and set that slot's flag only after the tree update.

Recover current command count by the root flags' valid prefix and fresh-node
frontier by node flags' valid prefix. Both are binary-searchable cold. No warm
allocator, retained parent pointers, global epoch counter or writable root word.

Intended allocation argument, still TO REVIEW: count filled boxes on live
nodes as a potential. Filling an empty box raises it once. Copying a full live
node replaces it with an empty-box live node and lowers it once; propagating
the changed parent pointer repeats this rule. New leaves have empty boxes.
This suggests constant amortized copies per external logical field change.
Insertion-only red-black recoloring has a constant-amortized potential argument;
at most two rotations occur per insertion. Existing-key parity toggles change
one logical field. Together these suggest O(t) allocated nodes after t updates,
rather than a full path of fresh nodes every time.

Do not import this as a black box. Exact rotation ordering, detached subtrees,
live-parent uniqueness, stale private node IDs, propagation and all prefix
allocation constants must be specified. Colors must NOT be overwritten as in
the textbook partial-persistence shortcut. Parent paths can be recovered/stored
privately in O(q^2) bits; no uncharged persistent inverse-pointer lists.

Only the latest root of NEW and the frozen final root of OLD are queried.
The proposed box semantics therefore need no version timestamp: latest-root
paths use effective fields, and OLD remains frozen until every contribution
has been folded. This omission requires an invariant proof, not an assertion.

## Physical-ID fold: avoid rank/count updates

Freeze OLD's final root from its Mth root slot. During the first M/2 NEW
commands, scan 2C OLD node slots per command. For each published slot r:

1. Read its immutable key i.
2. Look up i in the frozen OLD tree, obtaining active physical ID u and parity p.
3. Fold p into base[i] ONLY IF u=r.

Copied garbage nodes can share a key but do not pass the active-ID test. Each
live node is scanned exactly once; zero-parity keys remain in the tree. Padding
slots have publication zero. No sorted enumeration or subtree key count.

At NEW prefix j, an early-phase QUERY_i looks up OLD and includes its parity
only when its active ID exceeds cutoff 2Cj. This cutoff is physical-slot order,
not key order. NEW parity is always included. Once j>=M/2, every OLD node has
been scanned: QUERY must perform NO OLD acquisition, even for its root/flags.

During the second M/2 commands, clear OLD's entire T=M(Cv+a+1)-bit arena in
chunks ceil(2T/M)=2(Cv+a+1) bits. At M, OLD is zero: exchange ROLE, set STARTED
on the first exchange, and recover the next empty NEW frontier from its cells.
For initial empty OLD, STARTED skips its contribution. Acknowledgement follows
maintenance and role exchange; there is no mid-command crash promise.

## Intended joint accounting, not a theorem

If all preceding obligations hold, every arena bit changes at most once per
global epoch: fill in its NEW epoch, clear in its next OLD epoch, then alternate.
Base initialization changes each base bit at most once, and epoch1 folds an
empty OLD. Later epochs fold each logical key at most once. ROLE changes once
per epoch; STARTED once overall. These suggest L epochs and H=ML under wear L.
This is stronger than charging fill AND clear to the same global epoch.

Candidate probe costs: a lookup follows O(q) nodes and reads O(q) bits per node.
A field modification may copy/propagate O(q) nodes. O(q) worst recolorings then
suggest O(q^3) FLIP probes. Fold does 2C fixed lookups; clearing costs O(q).
The candidate allocation bound makes M=Theta(N/logN), suggesting the fixed
NL/(1024logN) lifetime target could hold with explicit small branches.

All arithmetic/local work is free ONLY under the proposed new probe contract;
its actual space, uniform source, halting, preparation bound and cleanup remain
obligations. This sketch supplies no compiler for the closed indexed model.

## Narrative attack

Honest present story: an occupied persistence technique might remove the
extra logarithm without a warm controller or writable root pointer. The
strongest dismissal is that this merely adapts write-once persistent trees.
Only a complete every-cell/cold-recovery construction and fair incumbent
comparison can resolve that. It is a concrete alternative to an unconstructed
routing directory, not a new-paradigm claim or permission to skip design gates.
