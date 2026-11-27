# A signed operation needs a role domain, not a positive-key neighborhood

**27 November 2026. Classification: necessary mathematical foundation for
persistent operation reuse.** Reviewed head `0b80f065ff8ed87a8457f2056be64bcb4fd3397e`. This is an exact
finite arithmetic control and static code audit; no model execution, no
production change, no new acquisition or held task result.

Responds to the [operand-encoding request](https://github.com/dl1683/Latent-Space-Reasoning/issues/6#issuecomment-7210562506)
and the [source-only discriminating-event request](https://github.com/dl1683/Latent-Space-Reasoning/issues/6#issuecomment-7210744222).
The signed reader is a useful distinct candidate. The archived all-key span
failure and the public ARC failure remain intact. The former shows that the
retired task state is mostly outside the retained key span; neither proves
a general failure of operand-space computation. The latter does not justify
growing the visual grammar.

## Two admission constraints that do not follow from operator identifiability

The current opt-in [selector](../experiments/memory/selective_continuations.py)
first selects positive-dot keys (inclusive cutoff at least 1), then solves
the bounded signed fit on that selected neighborhood. It refuses a fit
unless the selected columns have full column rank. These are conservative
implementation choices. They need not coincide with the existence or
uniqueness of a reusable operation.

**1. The source event that distinguishes two rival laws can be excluded by
positive-dot selection.** Let two orthogonal Q8 role keys be
(k_1=64e_1, k_2=64e_2). A late or source-event operand
(q=a e_1-b e_2), for positive integers (a,b\le64), has exact signed
coefficients ((a/64,-b/64)), with L1 norm at most 2. The second key scores
strictly negative. Even granting a shortlist of two and an exact numerical
solver, the selector sees only the first key and cannot fit q. This is not
the prior archived all-bank span failure: here the complete bank spans q
exactly.

Consider two supplied candidate laws in these operand coordinates:
identity and role swap. Both agree on ((a,a)), but predict
((a,-b)) versus ((-b,a)) on the contrast. These consequences have
equal norm. A source-only worker that has obtained these rival hypotheses
and the operands of those two candidate source events can choose the
contrast before observing the event outcome or the late task. Yet the
current signed selector cannot apply either law to that contrast under the
specified keys. The worker should distinguish operation identity, binding
of a role-conditioned operand, and computation of the consequence.
A higher similarity score or a longer list of positive keys does not fix
this example.

**2. New coherent evidence can make a defined operation unavailable.**
Take the positive operand ((a,b)) and the same two keys, with values
(v_1,v_2). A third source with the same (k_1,v_1) leaves the linear law
unchanged. With top-k=3, margin=0, tie limit=3, the original two columns
pass full-column-rank admission; adding the coherent row makes the three
columns rank deficient, so the current fitter refuses. The ambiguity is
in the coefficients, not the consequence. This matters in a persistent
world, where multiple independently pinned documents may express the same
relation.

For any selected operation group, define the feasible set
[
 C(q)=\{c:Kc=q,\ \|c\|_1\le L\}.
]
When nonempty, a unique predicted value exists exactly when V annihilates
all differences between elements of C(q). A sufficient global condition
is `ker K subset ker V`: for two feasible solutions their difference is
in ker K, hence their outputs are equal. If a feasible solution has strict
L1 slack, this condition is also necessary for uniqueness at that q:
a sufficiently small step in any null direction stays feasible. On a
boundary, use the actual feasible set; rank alone does not decide it.
For example, identical first and third keys and values satisfy the
condition. Replacing only the third value with its negative gives an
equal-norm conflicting row, and two feasible coefficient allocations now
give distinct outputs. Never merge those contradictory observations.

This does not authorize relaxing the existing numerical or ambiguity gate.
An implementation may simply use a consistent independent basis for a
validated group, retaining each source identity and provenance separately.
Inconsistent or insufficiently verified groups remain rival hypotheses.
An exact solver needs the same grouping, operand encoder, admissible
domain, source rights and revision semantics. A basis change after
withdrawal must not silently remove the only independent witness; and
equal projected values do not establish independent semantic support.

## Exact replay

The verifier below enumerates eight signed axis permutations and sixteen
positive magnitude pairs. Coefficients are represented as integer
numerators over 64. It independently implements the relevant score cutoff
and exact rank/span consequences; it does not import or execute the native
Python reader. The source file at the reviewed head was checked for the
actual positive cutoff, full-column-rank and L1 clauses. All checks passed.

| Constructed check | Count |
|---|---:|
| Full-bank signed contrast within L1=2 | 128 |
| Positive cutoff excludes a necessary key | 128 |
| Source-only contrast distinguishes identity from swap | 128 |
| Coherent duplicate makes full-column-rank gate refuse | 128 |
| Equal-scale conflicting duplicate gives two feasible values | 128 |

The counts are not a distribution over learned states or real corpora.
The two candidate laws and role coordinates are supplied, not learned.
The example establishes a necessary design discriminator, not a reason
to build a wider deterministic grammar or to bypass the fresh ordinary
competence prerequisite.

Run the following JavaScript (Node 18+ or an equivalent ECMAScript runtime):

```js
function verify() {
  const check=(ok,msg)=>{if(!ok)throw Error(msg);};
  const dot=(x,y)=>x.reduce((s,v,i)=>s+v*y[i],0);
  const eq=(x,y)=>x.length===y.length&&x.every((v,i)=>v===y[i]);
  const det=(x,y)=>x[0]*y[1]-x[1]*y[0];
  const scale=(x,n)=>x.map(v=>v*n);
  const add=(x,y)=>x.map((v,i)=>v+y[i]);
  // Coefficients have denominator 64; this is exact integer arithmetic.
  const apply=(cols,nums)=>[0,1].map(j=>{
    const n=cols.reduce((s,col,i)=>s+col[j]*nums[i],0);
    check(n%64===0,"integer image"); return n/64;
  });
  const select=(keys,q,k)=> {
    const scores=keys.map(x=>dot(x,q)), positive=scores.filter(x=>x>0).sort((a,b)=>b-a);
    const cutoff=positive.length>=k?positive[k-1]:1;
    return scores.flatMap((s,i)=>s>=cutoff?[i]:[]);
  };
  const stats={signed_contrasts:0,positive_gate_loses_necessary_row:0,
    source_witness_distinguishes_rivals:0,coherent_redundancy_refused:0,
    conflicting_redundancy_nonunique:0};
  const values=[[64,0],[0,64]], rival=[values[1],values[0]];
  for(const swap of [false,true]) for(const s of [-1,1]) for(const t of [-1,1]) {
    const k1=swap?[0,s*64]:[s*64,0], k2=swap?[t*64,0]:[0,t*64];
    const keys=[k1,k2];
    check(det(k1,k2)!==0&&dot(k1,k2)===0,"orthogonal role coordinates");
    for(const a of [16,32,48,64]) for(const b of [16,32,48,64]) {
      const nums=[a,-b], q=apply(keys,nums);
      check(Math.abs(a)+Math.abs(b)<=128,"bounded exact full fit");
      stats.signed_contrasts++;
      const hit=select(keys,q,2);
      check(eq(hit,[0])&&det(k1,q)!==0,"positive gate loses exact span");
      stats.positive_gate_loses_necessary_row++;
      // A source-only candidate event roster; no late-task input or outcome
      // is needed to decide which event can distinguish the two supplied laws.
      const neutral=[a,a], contrast=nums;
      const predicts=c=>[apply(values,c),apply(rival,c)];
      const [n0,n1]=predicts(neutral),[c0,c1]=predicts(contrast);
      check(eq(n0,n1)&&!eq(c0,c1),"only contrast distinguishes");
      check(dot(c0,c0)===dot(c1,c1),"matched consequence norm");
      stats.source_witness_distinguishes_rivals++;
      // Redundant evidence with unchanged operator at a positive query.
      const plus=[a,b], qp=apply(keys,plus), dupKeys=[k1,k2,k1];
      check(eq(select(keys,qp,3),[0,1]),"opening neighborhood");
      check(eq(select(dupKeys,qp,3),[0,1,2]),"expanded neighborhood");
      const cA=[a,b,0],cB=[0,b,a];
      check(eq(apply(dupKeys,cA),qp)&&eq(apply(dupKeys,cB),qp),"same operand");
      const coherent=[values[0],values[1],values[0]];
      check(eq(apply(coherent,cA),apply(coherent,cB)),"coherent duplicate");
      // Three columns in R^2 cannot pass full-column-rank admission.
      // Every feasible split of the first coefficient gives the same output:
      // the first and third keys AND values are identical.
      stats.coherent_redundancy_refused++;
      const conflicting=[values[0],values[1],scale(values[0],-1)];
      check(!eq(apply(conflicting,cA),apply(conflicting,cB)),"conflicting duplicate");
      stats.conflicting_redundancy_nonunique++;
    }
  }
  check(Object.values(stats).every(x=>x===128),"enumeration counts");
  return stats;
}
console.log(verify());
```

## Decision for the next authorized learned screen

Freeze a role-sensitive operand encoder on disjoint source-only material.
For a proposed group of rival laws, a worker should search for a source
event whose operand gives separated predictions under those laws, commit
the predictions before reading its verified outcome, and preserve unresolved
alternatives when none separates them within budget. Do not choose the
event using the late tasks or outcome. Measure candidate coverage and
conditional predictive accuracy separately. An event finder and a law
fit are separate stages; choosing the former by a source-only gap does not
qualify either stage semantically.

Compare the present positive-key fit against operation-address selection
followed by signed operand-domain application. Give the exact and ordinary
text comparators equivalent access, candidate hypotheses and cache rights.
At fixed operation address, include opposite-role operands and a coherent
duplicate source arrival, plus a conflicting witness and its withdrawal.
Record signed eligibility, full-group span, bounded feasibility and
predicted-value ambiguity separately; the all-group fit is a diagnostic,
not an uncharged online comparator. Keep the actual task independent of
the worker's event search, test two new bindings, hold suffix observation
access fixed in the separate causal diagnostic, and count all source
search, fit, verification, maintenance, storage and readout work.

A competent reader and genuinely acquired, independently graded
consequences are the decisive milestone. Do not delay that screen to
implement a general solver. If an independently frozen operand encoder
does not produce valid role contrasts on new material, stop this transport
candidate rather than normalize its way through the archive.
