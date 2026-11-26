# Coverage certificates for a persistent latent world (research response)

Response to [owner direction #6](https://github.com/dl1683/Latent-Space-Reasoning/issues/6), [separate-relay review](https://github.com/dl1683/Latent-Space-Reasoning/commit/c79f894e15ff7c745390f366ede3e0d315415d27#commitcomment-211913067), and [compatibility review](https://github.com/dl1683/Latent-Space-Reasoning/commit/148f3005654e76b3c905693a0f5cab6a26bbd3cf#commitcomment-211909146). Inspected live pin c79f894e. This is a mathematical proposal and a standalone synthetic CPU check, not learned competence or an authorization to resume the paused model experiment.

## The architectural decision

A fixed-width batch of continuous hypotheses by itself does not solve coverage. A missed relay may yield the only conflicting consequence. Refusal bands are useful heuristics but cannot certify what is below them. This is already the finite-beam failure in FAILURES F-014–016; do not repeat a wider beam sweep.

Instead retain a small active batch plus a *recoverable, versioned cover* of inactive possibilities. A cover cell stores a summary of its member states, the admissible transitions it bounds, and pointers to children and source dependencies. The cover is persistent, not a token prompt. Its summaries are part of the numerical world and evolve when sources change. An unresolved-support signal accompanies the active states through the computation. A maintenance swarm can refine cells with the largest unresolved effect, propose links or new partitions, and publish after independent checks. It must not interpret a lack of proposed links as evidence of absence.

This is potentially a primitive for whole-corpus self-audit on constrained devices: charge active state separately from retained summaries and backing; a cap on active states does not imply bounded total storage or full recall. Initial ingestion, representation validity, novel relations and multimodal coverage remain unsolved.

## A sound local rule for a declared finite transition family

Let H be the finite candidate relay set at a pinned version and let each retained cell enclose its states in a coordinate box [l,u]. For a fixed terminal observation t, a transition of the form tanh(A_t h + b_t) includes linear carry and any bilinear interaction with fixed t (absorb t into A_t). The routing logits are likewise affine when the key is fixed. For any affine map M h + b, a componentwise enclosure is

    lower = M_positive l + M_negative u + b
    upper = M_positive u + M_negative l + b

where the negative part contains the signed negative entries. Apply tanh componentwise to both endpoints, then bound the answer margins with the same affine rule. This covers the declared operator only, with outward rounding or a justified error allowance required in a floating-point implementation. Extra nonlinear terms need their own enclosures. Route eligibility, phase eligibility, answer margins, source independence and admission limits must all be accounted for; score bounds alone do not prove semantic validity.

Each unresolved cell/terminal pair produces an overapproximation U of possible outcomes. Exact admissible witness paths give a lower set L. Keep undecidable/refusal outcomes distinct from answer labels. On a complete declared candidate cover, L is a subset of the exact outcome set, which is a subset of U. A singleton L=U (with no potentially undecidable paths) permits a supported singleton; two distinct witnessed labels prove a conflict even if other cells remain unresolved. An empty U proves absence only for the declared finite grammar and coverage assumptions, not absence from the real corpus. Otherwise refine or return unresolved, never a false singleton. Stale summaries or a missing cover invalidate the certificate.

This rule turns a worker's gap queue into measurable uncertainty about a particular consequence, rather than another similarity ranking. Its cost can still be linear or worse if bounds are loose; general long trajectories can amplify uncertainty. Do not promise sublinear worst-case cost.

## One decisive same-terminal revision

Supplied scalar relay states (after the first continuous step) are 64 each at -0.8, -0.6, -0.4, -0.2. Both terminal observations are fixed throughout:
* A: route score 1, output tanh(h+2), always positive on [-1,1].
* B: route score h, output tanh(h-2), always negative on [-1,1].
* Route floor 0.5; answer margin 0.1. Both terminals use the same declared phase.

Opening: A only. A source-conditioned correction changes one generated relay from -0.8 to +0.8, with the terminal observations unchanged. The exact result is now conflict. Restore the original generated row as a same-pin causal intervention: the conflict disappears. A source restoration and historical reread also yield A. This models a correction of the continuous row, not an actual encoded-source revision test. The opening read and coordinate frame are identical between the compared arms. A one-state cap excluding the corrected row can falsely report A. Our cover either finds the conflicting witness or refuses. It does not invent a stronger score floor.

Executed a standalone immutable balanced min/max tree on 256 rows. Building it creates 511 nodes; a single leaf replacement copies 9 nodes (including leaf and root). A one-node query budget excludes B on the opening and restored pins, but must return unresolved on the correction. A left-first traversal witnesses B after nine bound-node checks on the corrected pin; the original root is unchanged. An exhaustive vector join agrees on all four reads. These are unit counts, not elapsed time or complete storage/cost measurements.

| Pin | Exact endpoint status | Cover (adequate budget) | Bound-node checks | Cover (budget 1) |
|---|---|---|---:|---|
| Opening | A | A | 1 | A |
| Correction | conflict | conflict | 9 | unresolved |
| Restoration | A | A | 1 | A |
| Historical opening | A | A | 1 | A |

**Strong negative result:** the equally informed indexed/vector comparator can simply maintain a maximum relay score (and its witness) and decide all four statuses in one root lookup after the same logarithmic update. The cover has no intrinsic advantage on this fixture. An exhaustive scan is not the strongest comparator. Similarly, if the comparator evaluates precisely the same generated rows, route eligibility and state-dependent per-pair maps, it must produce the same outcome as the unpruned branch reader. Replacing one generated row changes both. A causal effect of the row is not an advantage over this comparator. The proposed fixed-width successor should earn any advantage on *repeated composition, revision and reuse*, not by restricting the comparison arm.

Reproduce the synthetic check (standard library, no model, no repository dependency):

```python
from dataclasses import dataclass
from math import tanh

@dataclass(frozen=True)
class Node:
    lo: float
    hi: float
    left: object = None
    right: object = None

def build(a):
    if len(a) == 1:
        return Node(a[0], a[0])
    k = len(a)//2
    l, r = build(a[:k]), build(a[k:])
    return Node(min(l.lo,r.lo), max(l.hi,r.hi), l, r)

def edit(n, path, x):
    if not path:
        return Node(x,x), 1
    l,r = n.left,n.right
    if path[0] == 0:
        l,c = edit(l,path[1:],x)
    else:
        r,c = edit(r,path[1:],x)
    return Node(min(l.lo,r.lo),max(l.hi,r.hi),l,r),c+1

def certify(root,budget):
    stack,seen = [root],0
    while stack and seen < budget:
        n=stack.pop(); seen+=1
        if n.hi < .5:
            continue
        if n.lo >= .5:
            return 'conflict',seen
        stack.extend([n.right,n.left])
    return ('unresolved' if stack else 'A'),seen

a=[-.8]*64+[-.6]*64+[-.4]*64+[-.2]*64
r=build(a)
c,cost=edit(r,[0]*8,.8)
s,cost2=edit(c,[0]*8,-.8)
assert tanh(-1+2)>.1 and tanh(1-2)<-.1
for name,n,raw in [('opening',r,a),('correction',c,[.8]+a[1:]),
                   ('restoration',s,a),('historical',r,a)]:
    exact='conflict' if max(raw)>=.5 else 'A'
    z=certify(n,1000)
    assert z[0]==exact
    print(name,z,certify(n,1))
assert certify(c,1)[0]=='unresolved' and cost==cost2==9
```

## Next discriminating test, not another hand-designed receiver benchmark

Construct independently held revision sequences with changing first-hop routes, fixed terminal catalogs, state-dependent carry/interactions and late *previously unspecified* third/fourth operations. Compare (i) native pooled state, (ii) bounded hypotheses with unresolved cover, (iii) exact inferred-role vector join with the same continuous maps and the same opportunity to cache, factor, and incrementally update, and (iv) exact source-stream comparator. First qualify native encoding and the exact comparator on independently held data. Intervene on one generated row on the same source pin and replay each arm. Freeze budgets and outcome labels before running. Include unanswerable, ambiguity, weak omitted routes, remove-either, restore, and unrelated-edit controls. Use independently generated source changes rather than fitting from held answers.

Falsification: any certified false singleton/absence rejects the coverage mechanism. If coverage is sound but repeated refusals or full refinement eliminate any quality-cost gain against the optimized comparator, reject the efficiency claim for this mechanism. If the vector comparator matches or wins, do not relabel its failure as a latent victory. Report the entire denominator, exact endpoint/witness recall, false support, missed support, abstention, calibration, encoder calls, stored source/latent/index bytes, summary construction and repair, rejected branches, every per-pair transition, worker scheduling and validation, cold and repeated wall time and peak memory, plus the cost of source reopening. Edge feasibility needs actual consumer hardware measurements. The larger objective is a self-maintaining corpus world, not a two-hop benchmark score.

## Response to compatibility review

The gate in LATENT_COMPATIBILITY.md applies to a transplant claiming to preserve the *same native operation*. It must not categorically stop a mismatch-correcting edit: that residual is the hypothesized target. Measure before/after commutation on independent probes, preserve opening and protected native behaviors, check actual later nonlinear two-hop consequences under removal/restoration and compare a direct destination refit given the same intervention information. A passing local linear residual is neither a causal proof nor global compatibility. Across frozen bank pairs test incremental predictive value beyond state error and span coverage; learned-bank results remain NOT_RUN.
