# Uniform positive-branch cleanup: proposed exact cap log2(3)

2026-10-03. PROVED AT SCOPE: lower and charged upper passed the fresh actual
zero-context attack in UNIFORM_BRANCH_REVIEW.md. Original submitted candidate
wording below is provenance. That review additionally removes context
enumeration for V: exact5n+2r+1 gates, hence optimal Theta(n+r) even growing r.
These are elementary gate counts; program/preparation bit work is separate.
This strengthens METER_REVIEW's narrow uniform exception, whose original
verdict did not approve an exact floor. No new registered FULL is asserted.

## Contract and audit

Finite n>=2 clean AND, original uniform X, independent uniform finite r-bit R
returned pointwise, fixed NOT/CNOT/Toffoli/one-bit RESET program, scratch zero
and finally clean. Every bit/gate/program/preparation charged. At each RESET
and each positive predecessor class, V=log2(postclass mass/preclass mass).
Reversible steps Vzero but gates charged. Common atom mass2^-(n+r). No real
precision instruction, unbounded advice or free predicate; exact gate upper
is the already code-reviewed finite context-linear constructor. This is a
logical cost cap, not device heat or priority claim.

## Proposed all-depth CANNOT

If K<log2(3), every live complete-state class has a power-of-two atom count.
Proof by induction: initially each contains one original atom. Reversible
gates preserve counts. A one-bit RESET merges at most TWO complete states.
If their power-of-two counts differ, their ratio is at least2, so the smaller
child has branch ratio (a+b)/min(a,b)>=3, violating K. Thus only equal-power
children can merge, leaving another power of two. A singleton transition also
preserves its count. Zero probability states are not live classes.

Pointwise R return forbids merging original labels with different R, since
once complete states coincide their deterministic futures coincide. Final
AND-zero/R class has 2^n-1 original atoms, not a power of two for n>=2.
Therefore no finite width or depth can realize exact cleanup below log2(3).
Logical/physical recoding of R before its return changes no argument.

## Proposed charged CAN

Use context_threshold.fast_circuit(n,r): transfer R to output, compute/copy/
uncompute AND, build each full-context predicate by charged Gray prefix gates,
erase X bits in order via flag=predicate AND X_bit, CNOT(flag,X_bit), RESET(flag),
finally uncompute context/sentinel. Width2n+3r+4; work O(n*2^r+n+r), hence
Theta(n) for fixed r, with input-target lower Omega(n+r). For r0 exactG5n+1.

After j data bits have been erased within one context, ordinary Yzero classes
have count2^j. The one suffix consisting of allones has Yzero count2^j-1:
the excluded original allones belongs instead to the distinct Yone class.
At the next selected erasure, ordinary pairs have equal count2^j and both
branch ratios2. The exceptional Yzero pair has child counts 2^j and2^j-1.
At j0 the latter is zero, so no positive branch collision occurs there. At
j>=1 the worst positive ratio is

    (2^(j+1)-1)/(2^j-1) = 2+1/(2^j-1) <=3,

with equality j1 (n>=2). The Yone class never collides with Yzero and carries
one atom; its flag erasure is reversible on live support. Other R contexts
have predicatezero, and no context is mixed. Scratch resets all obey this
analysis because the three-gate macro's flag exactly stores the erased bit
on the selected slice and CNOT restores the target to zero BEFORE flag RESET.
All remaining primitives are reversible and preserve branch masses.

Thus the proposed exact V feasibility floor is log2(3), independent of n,r.
The elementary constructor witnesses the bound with no free aggregate merge.

## Independent gate request and meaning

Attack the power-of-two invariant at arbitrary depth/width, previous coalescence,
zero states, recoded R and correlated flag encodings. Independently check the
actual source word and exceptional child-count induction, including n2/r0,
n3/r1, n>=4/r>=2, scratch cleanup and exact gate counts. Probe finite circuits
on CPU only; no FULL authorization from this proof packet. Return lower/upper
PASS/REPAIR/KILL at precise scope. No exact unknown-prior V frontier follows.

So what: an unknown small-atom workload admits no finite universal branch cap,
yet a known uniform workload has a dimension-independent exact floor and
linear fixed-context work. This isolates the role of source promises under a
worst-branch meter. Useful-world calibration and novelty remain unresolved.
