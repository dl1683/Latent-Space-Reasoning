# Round 19 independent proof review, droid (2026-10-08)

## Verdict and exact scope

**PROVED-AT-SCOPE after independent proof attack:** the universal
classification of `ROUNDTABLE/work/codex/r16_THEORY.md` §§1–3 survives
the attacks below. This is an independent LLM review, not human peer
review, priority, or a computational advantage. The general theorem
does **not** follow from the finite censuses; the finite censuses are
**EMPIRICAL** checks of separately stated instances. The machine is an
ordinary, nonuniform finite reversible-word library, and an
equal-access ordinary controller runs every constructed word verbatim.
The exact novelty/usefulness question remains **OPEN**.

Let finite `G`, `n>=0`, `ell>=0` be fixed. Immutable Boolean inputs
`x`, one arbitrary initial borrowed bit `b0`, and one arbitrary initial
group register `z0` are the only state. A public finite nonadaptive
word uses `A_p: b ^= p(x)` for **positive** monomials `deg p<=2`
(including 1), `B_(q,g): z <- g z` when `b q(x)=1` for positive
`deg q<=1`, and `O_g: z <- g z`; every gate costs one. Borrower
restoration at a boundary is an **identity of functions of every input
and both initial owners**, not a property of one run. Its consecutive
non-restored boundary count is at most `ell`; the final transformation
must be `(b0,z0)->(b0,F(x)z0)` for **all** initial states. There is no
output feedback, reset, free work tape or unpriced long predicate.
Descriptions/group labels and compilation are finite and must be
charged when comparing algorithms.

Put `P=G×G`, `s=floor(ell/2)`, commutator
`[u,v]=u^-1 v^-1 u v`. For `1<=r<=s`, put
`A_r=<(g^(2^(r-1)),g^(-2^(r-1))):g in G>^P` (normal closure).
Let `N_0=N_1=P` and, for `d>=2`,
`N_d=<A_r:2r+1>=d; [N_i,N_(d-i)]:1<=i<d>^P`.
Let `K_d={g:(g,g) in N_d}`. Sort Boolean faces by increasing
cardinality and then support mask; uniquely write
`F(x)=prod_S c_S^(prod_(i in S)x_i)` in this order.
Then, for `ell>=2`, the result is exactly
`F accepted iff c_S in K_|S| for all S`, count
`prod_(d=0..n)|K_d|^binom(n,d)`. For `ell=0,1` only
constants occur. Leases `2s` and `2s+1` have the same final class,
but not necessarily the same optimal gate count.

## Step-by-step attack of the proposed proof

| Step in r16_THEORY.md | Audit | Independent check, repair or counterexample attempt |
|---|---|---|
| §1 lease and edge cases | VERIFIED | A lease-one opened owner must close on the next gate; with `b0=0`, restored B gates do nothing. `O_g` realizes constants. Do **not** test restoration on each input separately. |
| Lemma 1, finite excursions | VERIFIED | Let `j` be the number of distinct surviving degree-≤2 ANF monomials in the offset at a B. At least `j` flips preceded it and `j` flips must follow before universal restoration, so `2j<=ell`. Isolating each B with the same `j` flips open and closed retains its two owner-slice factors in chronological order; removing original A gates does not change any z output. Every replacement peaks at `2j`. This also proves odd/even stutter for all n and G. |
| Lemma 2, integer XOR polarization | VERIFIED | Pointwise over integers, `XOR_i p_i=(1-prod_i(1-2p_i))/2`. Hence coefficient of a nonempty r-fold product is `(-2)^(r-1)`, without an illicit F2-to-odd-field conversion. In each block both output components are powers of **the same** g, so they commute there even if G does not. Möbius inversion over the subcollections of p slots plus a restored-boundary B cancels the owner-one affine term. Repeated and constant slots still obey the pointwise identity. Reversing g handles the sign. |
| Lemma 2, low faces and normal conjugates | VERIFIED | `B_(q,g)` gives `(1,g)^q`; `A_1 B_(q,g) A_1 B_(q,g)` gives `(g,g)^q` with peak two, including arbitrary non-involutions. Multiplying/inverting gives `(g,1)^q`. A constant returned pair conjugates a supported word and leaves off-face identity, so it constructs the **ambient P-normal closure** of each atom, not just its cyclic generators. |
| Lemma 3, `N_d` and centrality | VERIFIED | Inductively `N_(d+1)<=N_d`; the atom sets are nested and reducing either positive commutator index supplies the earlier defining term. Normal closure is preserved by commutator identities. `N_i,N_j` commute into `N_(i+j)`, even for large i,j, using their defining recursion and descendingness. This is the **least**, not merely a containing, central filtration of the weighted primitives. |
| Lemma 4, finite face collection | VERIFIED | Moving a face S left across T contributes a commutator on S union T. Its label lies in `N_(|S|+|T|) <= N_|S union T|`, including overlapping faces. In increasing-size/inclusion-compatible order the union is later unless S=T; equal-face factors combine in their subgroup. Processing each finite face only once terminates because corrections never re-enter earlier faces. Evaluation at `1_T` recovers its coefficient after all proper subsets, even for nonabelian G. No infinite nilpotence assumption. |
| Lemma 5, all-label localization | VERIFIED | At level d, an atom with weight at least d can use constants/duplicate slots to occupy any d-face. For a commutator `[N_i,N_(d-i)]`, partition the d *distinct* names into i and d-i and commutate the corresponding returned supported words. Their support intersection is precisely that d-face. Product, inverse and constant conjugation generate every label in the defining normal closure. Reverse chronological spelling for left multiplication is essential. |
| Completion, diagonal restriction | VERIFIED | The actual returned pair is `(F0,F1)`. The universal final contract requires `F0=F1=F`, not merely equality modulo a quotient or for one initial z. Uniqueness of ordered coefficients coordinatewise gives `(c_S,c_S) in N_|S|`. Conversely each such pair is independently constructible; reverse word order realizes the target ordered face product and all intermediate words return b. |
| §4.1 stable nilpotent residual | VERIFIED | `gamma_d(P)<=N_d`; hence the terminal `R×R` lies in all levels. Projection through `G/R` maps atoms/normal closures/commutators onto the quotient recursion. As the kernel is already contained, each `N_d` is the full preimage. At nilpotency class c, every weighted leaf has weight at most `t=2s+1`, so `d>ct` forces quotient identity. This is compatible with the earlier reviewed R13 correction of **both** owner slices. |
| §4.2 direct products and abelian formula | VERIFIED | Pure-factor atom labels separate, and normal/commutator closures separate. For additive A, `N_d` (d>=2) is the anti-diagonal of `2^(floor(d/2)-1) A` until `d=2s+1`, then trivial. Its diagonal intersection is `A[2] intersection 2^(floor(d/2)-1)A`; this matches previously independently reviewed R11, including non-cyclic torsion. |
| §4.3 elementary-2 abelianization | VERIFIED | `A_1` contains `[G,G]²` by the displayed three-factor product and component exchange, so it is exactly the anti-diagonal preimage. When the abelianization has exponent two this is also the diagonal preimage. In `gamma_j/gamma_(j+1)`, simple j-fold commutators are multilinear and squares of generators vanish because `g² in gamma_2`; thus all layers have exponent two and `g^(2^(r-1)) in gamma_r`. This gives upper `N_d<=gamma_ceil(d/3)(P)`; supported commutators of diagonal three-faces supply the matching diagonal `K_d=gamma_ceil(d/3)(G)`. The sharper whole-pair `Q_r^+` identity uses the component-kernel argument below, not just a quotient sign. |
| §4.4 odd-order component kernels | VERIFIED | `A=A_1=Q_1^-`: anti-pairs surject onto the negative abelianized diagonal and the three-factor identity generates `gamma_2(G)×1`, with exchange generating the other component. Induct on r: the projection of `[gamma_r(A),A]` to `gamma_(r+1)/gamma_(r+2)` has sign `(-1)^(r+1)` and is generated by simple iterated commutators. The **independent kernel** `gamma_(r+2)²` comes from commuting `gamma_(r+1)×1` and `1×gamma_(r+1)` already in `gamma_r(A)` with A, whose projections onto G are surjective. Thus `gamma_r(A)=Q_r^((-1)^r)` as a full preimage, not only modulo the next term; likewise `[P,Q_r]=gamma_(r+1)²` by separate component P conjugations. Weighted all-A and mixed-P commutators have maxima `rt` and `(r-1)t+1`; intersect their resulting full pair groups with the diagonal. Odd-order quotients have bijective squaring, so the minus sign contributes exactly the next lower-central term. This establishes §4.4's staircase. |
| §§4.5–4.6 coordinate/count and spectrum | VERIFIED | For UT_m, products of distance-a and distance-b upper-diagonal entries have distance at least a+b, and the displayed thresholds are superadditive. Entrywise Möbius stripping hence agrees with the filtered group coefficients. Two-group powers become trivial after their exponent threshold; nonabelian odd Sylow in the nilpotent quotient has nonzero `gamma_2/gamma_3` and an unbounded odd staircase. The class claims, **not** efficiency/novelty, follow. |

**Potential proof pitfalls attacked but not found:** replacing integer XOR with
F2 addition in odd G, neglecting the independent component kernel,
conjugating only in diagonal G instead of P, treating the borrowed bit
as restored pointwise, mistaking `F0 F1^-1=1` for an identity of
noncommuting words, and keeping chronological products in the wrong
order. The particular steps in the table have no remaining GAP or
WRONG at the stated alphabet. Code-assisted samples are not used to
upgrade any all-n proof.

**Even/odd reachability stutter does not imply
equal optimal gate cost.** Independent raw
BFS in C2/C3/C4 at n=2 finds identical
lease-two/three table sets but 2/16,
12/27 and 84/128 tables with strictly
shorter lease-three words
(`r19_COST_STUTTER.json`). In C3
the table `(0,1,1,2)` takes eight
gates at lease two but six at
lease three: `A_1; B_(x1,1);
B_(x2,1); A_1; B_(x1,1); B_(x2,1)`.
The global peak is three. The
6-versus-8 optimum is an exact
finite raw-state BFS finding,
not a general complexity claim.

## Counterexample searches and finite receipts

The new `r19_check.py` constructs groups without importing the r16
implementation. It computes `N_d` by ambient normal closure and
generator commutators, and **separately** closes every normalized B
activation pair by a faithful permutation action. Stabilizing the
identity point of the relative `y -> F0(x)yF1(x)^-1` action on every
vertex imposes exactly `F0=F1`. If the center is nontrivial, an
additional left-regular slice makes the representation faithful.
All independently generated group tables passed exhaustive
associativity checks.
`r19_NAIVE_FILTRATION.json` independently evaluates
the **literal** recursion with every subgroup-element
commutator and every ambient conjugate, rather than the
optimized generator implementation: 56/56 full N_d
sets match on C9, Dic3, M16 and the exponent-nine
extraspecial group at two leases through depth six.
S3, D32 and F39 add 42 more full-set
checks against the same naive recursion,
for **98/98** total (`r19_NAIVE_*.json`).
For F39 the pair N4/N5 genuinely
grow with lease, while their diagonal
K4/K5 do not; raw-pair growth does
not imply a new final class.
As an independent attack on *normalization itself*,
`r19_raw_bfs.py` explores raw interleaved A/B/O
gate-word graphs at n=2 without normalizing
excursions. All four C3/C4 cases at leases
2/3 or 2/4 close (largest: 720,896 states);
the accepted table **sets**, not just orders,
agree exactly with K: 27/27 C3 and 128/128
C4 (`r19_RAW_BFS.json`). These are finite
checks, not an all-word proof.
`r19_PAIR_FACES.json` additionally tests
all 142 **full pair**, not just diagonal,
supported N_d subgroup generators for
C9 and S3 at n=3 and Dic3 at n=2,
leases 2/4; every generator belongs
to its direct raw returned-gate closure.
Nonnilpotent S4 at n=2 adds 84/84
supported full-pair generators
(`r19_PAIR_FACES_S4.json`), totaling
226/226. A separate
`r19_FULL_PAIR_NONNIL.json`
tests **660/660 more
non-diagonal permitted
N face generators**
at n=3 in C3×S3,
A4×C2, Dic3, D12,
F21 and F39
at three leases,
against *direct*
full-G raw gate
collectors. Total
constructive pair-face
checks: **886/886**.
An edge follow-up enumerated 12 further
raw graphs on n=0,1, lease=0,1,2 in C3/C4.
All *sets* match the separate constant-only
clause at lease <2 and the K formula at
lease two (`r19_RAW_EDGES.json`). The first
checker pass wrongly applied K at lease
zero/one, recorded and fixed in the ledger.
Unnormalized n=1 raw BFS
also closes all S3 and Dic3
gate states at leases
zero/one/two. Their
accepted *sets* have
sizes 6/6/36 and
12/12/144, respectively,
exactly the constant-only
edge and lease-two
one-input class
(`r19_RAW_EDGE_NONABELIAN.json`).

**EMPIRICAL: 135/135 exact n=1,2,3 by lease=2,3,4,5,6 comparisons
match** definition-derived K (`r19_SUMMARY.json`, cross-checked against
`r19_filtration.jsonl`). In the final
aggregate, all **135/135 direct
full-group accepted orders**
and **135/135 raw pair
orders** match independently
defined K/N, with **60/60**
additional R13-assisted
nonnilpotent accepted-order
cross-checks. No group disagrees.
C9/C4×C4 are direct
permutation-group censuses. M16 and both extraspecial order-27 groups
are directly enumerated by independent central-chief-series
collectors `r19_class2_2.py` and `r19_class2_3.py`; these also test
**all** permitted diagonal face labels and raw pair orders.
The original runs for C3×S3, A4×C2, Dic3 and D12
are **reduction-assisted** censuses:
`r19_reduced.py` explicitly computes their lower-central residuals,
checks quotient multiplication-table isomorphisms, enumerates actual
gate closures in the nilpotent quotients (abelian integer lattice
or D8 central chief series), and multiplies by `|R|^(2^n)` using
the earlier independently reviewed R13 theorem. Those
original runs are **not** direct full-G enumeration.
Several attempted direct
large permutation jobs remain resource-limited; no uncompleted
job establishes a negative.
**Additional direct full-G
collectors recheck C3×S3,
A4×C2, Dic3 and D12
across all 60 of their
required n/lease
settings.** Each
verifies a complete
group-table isomorphism
to a semidirect coordinate
model, closes *actual* gate
words with quotient bases
and normal residual kernels,
then computes point-orbit
stabilizers and all
predicted diagonal face
memberships. All 60 raw
pair and accepted orders
match N/K
(`r19_DIRECT_C3xS3_COLLECTOR.jsonl`,
`r19_DIRECT_A4xC2_COLLECTOR.jsonl`,
`r19_DIRECT_Dic3_COLLECTOR.jsonl`,
`r19_DIRECT_D12_COLLECTOR.jsonl`).
The direct runs do **not**
invoke R13 or a
residual-fibre count formula.
Thus **all 135 required
cases have direct full-G
returned-gate closure
counts** as well as
independently computed
N/K predictions. The
R13-assisted runs give
an additional method
for the four
nonnilpotent families.
Eight n=1 group/lease
cases independently
match the custom
collectors' raw and
accepted orders under
generic faithful
permutation closure
on the *original*
group tables
(`r19_SMALL_CROSSCHECK.json`).
At n=4, direct
D12 and A4×C2
collectors also
match N/K at
leases two and
six with 1,140
diagonal face
membership checks
(`r19_NONNIL_n4.json`).
D12's first
restricted K4
has size six,
against |G|=24.
At the same n=4
frontier, direct
C3×S3/Dic3 counts
also match N/K
(`r19_NONNIL_n4_EXTRA.json`).
**Dic3 separates
lease two from
four by exactly
a factor of two**:
its K4 grows
from 3 to 6.
A 12-gate,
peak-four literal
word realizes
the central
`x²*AND4`,
excluded at
lease two
(`r19_dic3_lease.py`).
This is a new
positive contrast,
not a theorem
counterexample.
At n=5 the
same direct
Dic3 collector
finds a
**64-fold**
lease-four
gain, exactly
as predicted
by the six
new degree-four/
five face
coefficients;
all 666
diagonal face
checks pass
(`r19_Dic3_n5.json`).
A second
12-gate,
peak-four
literal word
realizes
`x²*AND5`
on all 64
input/borrower
pairs; lease
two excludes
that coefficient
too
(`r19_dic3_lease.py`).
The direct
A4×C2 collector
also matches
N/K at n=5,
leases two/six
with 656 passing
face memberships
(`r19_A4xC2_n5.json`).
Its raw pair
group grows by
a factor of
729 between
leases two/six
while the
accepted class
stutters.
Direct D12 n=5
timed out twice
(180 and 120
seconds); mark
that extension
**OPEN for
resources**,
not a negative.
An internal check
of six semidirect
collectors confirms
4,095/4,095 original
raw gate generators
remain members of
their constructed
subgroups across
54 cases
(`r19_COLLECTOR_INCLUSION.json`).
As a cross-check independent of
the R13 quotient lift, direct
full-group S3 n=2 and S4 n=1
closures match K in four
lease-two/four cases, with
28/28 diagonal face basis
checks (`r19_DIRECT_S3_n2.json`,
`r19_DIRECT_S4_n1.json`).
Direct S4 n=2 stabilizer
timed out at 180 seconds
without a result.
For a *genuine nonlinear
residual restriction*, a
separate F21=C7⋊C3
full-group collector closes
the actual gates through
F3 quotient lifts, F7
translation kernels and
explicit orbit stabilizers,
without R13 reduction.
All 15 n<=3, lease 2..6
accepted orders and 15
raw pair orders match K
and N, respectively; all
predicted diagonal faces
are members
(`r19_DIRECT_F21_COLLECTOR.jsonl`).
Its n=2 accepted count
is 64,827 rather than
the unconstrained
`21^4`. Generic
Schreier-Sims timed out
on the same instance.

| G | All five exact accepted n=3 counts |
|---|---:|
| C3×S3 | 136,048,896 |
| C9 | 6,561 |
| C4×C4 | 16,777,216 |
| A4×C2 | 1,358,954,496 |
| D12 (rotation order 12, group order 24) | 110,075,314,176 |
| Dic3 (order 12) | 26,873,856 |
| M16 (`C8 ⋊_5 C2`) | 268,435,456 |
| each extraspecial order-27 type (exponent 3 or 9) | 43,046,721 |

The equality at n<=3 does **not** mean higher leases are redundant.
At M16, n=4, the raw chief-series census changes from `2^51`
(lease two/three) to `2^52` (lease four/five/six), with every
permitted diagonal face generator checked (`r19_M16_n4.jsonl`).
An independent 22-gate peak-four word realizes `a²*AND4`.
For each of the two order-27 groups a 48-gate, peak-four
literal word realizes central AND10. The `r19_witness.py`
replay covers every input and initial owner; since gates
left-multiply and never read z, it also covers every original
regular output. The two extraspecial types have equal **raw
pair-group orders and accepted orders** at each n<=3/lease case;
their `N_d` and `K_d` **sizes** match through depth 16 at
leases 2,4,6, not an identification of subgroups in a common G.
At n=7, the exponent-nine order-27 group separates lease two
and four by an exact factor of 3, from direct chief-series
raw-gate closure (`r19_E27exp9_n7.jsonl`). Every predicted
diagonal face label was checked for membership, 574 at lease
two and 576 at lease four. The exponent-three type
independently has the **same full raw-pair and accepted
orders** in both n=7 cases, and the same membership
test counts (`r19_E27exp3_n7.jsonl`).
At the second higher-lease boundary,
`r19_lease6_literal.py` constructs a
128-gate peak-six returned word for
`[a^4,b^4]*AND14` in **each** of the
two nonisomorphic order-27 groups.
Every one of 32,768 input/borrower
pairs passes in each group, hence
all 884,736 original output
states per group by left
action. Independently defined K14
is trivial at lease four but has
order three at lease six and
contains this coefficient. This
is a finite literal witness,
not a general complexity advantage.
`r19_KERNELS.json` separately
compares **full sets**, not merely sizes, of
`gamma_r(A1)=Q_r^((-1)^r)` and
`[P,Q_r]=gamma_(r+1)(G)^2` through r=4 in both order-27
types, odd nonnilpotent F21, C9 and binary D16. All 40
equalities pass. A first failure was a consumed generator
expression in this reviewer's closure checker; the checker
was repaired and all tests rerun.
An additional `r19_PAIR_FORMULAS.json` all-element comparison
checks the **entire N_d subgroup**, not just its diagonal,
against the closed odd/binary piecewise formulas through
d=16 at leases 2,4,6 in D16, D12, both order-27 groups,
C9 and odd nonnilpotent F21: **288/288** set equalities.
An additional 144/144 full-pair set equalities
match in larger binary D32, nonsplit Q16,
and nonnilpotent S4 at three leases
(`r19_PAIR_D32.json`, `r19_PAIR_Q16.json`,
`r19_PAIR_S4.json`). S4's lower-central
residual is A4, not V4, independently
computed by the checker.
Even nonsolvable S5,
whose stable residual
is perfect A5, adds
48/48 full set
equalities across the
same three leases
(`r19_PAIR_S5.json`).
On A5 itself,
the explicit recursion
also returns N_d=P
and K_d=A5 through
16 levels at each
of the three leases
(`r19_PERFECT_A5.json`).
The larger odd nonnilpotent F39 adds
48/48 full-set equalities against the
odd staircase (`r19_PAIR_F39.json`);
the lower-central C13 residual persists.
An additional parameterized
F13 semidirect collector
directly closes *actual*
F39 gate words at
n=1,2,3 and leases
2..6: **15/15** accepted
and raw pair orders
match N/K, with all
diagonal face labels
checked
(`r19_DIRECT_F39_COLLECTOR.jsonl`).
At n=4, direct
F21 and F39
gate collectors
also match N/K
at leases two
and six, with
1,040 passing
diagonal face
memberships
(`r19_RESIDUAL_n4.json`);
the accepted
counts stutter
despite raw pair
groups growing
by a factor
of three.
These finite cases back the independently attacked kernel
proof but cannot substitute for its arbitrary-group step.
For §4.2's splitting
identity, `r19_DIRECT_PRODUCT.json`
compares every pair element
in 102/102 subgroup
sets of C3×S3 and
A4×C2 at three
leases, through
depth 16. Full
direct-product
factorization holds
in every case.
The separate literal `r19_RANDOM_FACES.json` stress generated
20,000 legal returned programs on five inputs, including
3,828 B gates with two or more live offset monomials.
Normalization preserved both borrower slices and legal
global peak, and all 640,000 independently extracted
ordered pair-face coefficients belonged to their N_d
levels. This is **EMPIRICAL** only, not an all-word proof.
An n=6 F21/F39/S4
follow-up adds 2,250
legal literal words
and 144,000 passing
pair-face coefficients
(`r19_RESIDUAL_RANDOM.json`).
Both face stress
sets together cover
22,250 programs
and 784,000 coefficients;
no mismatch was found.
A final literal
S5 test at n=4
adds 1,500
nonsolvable words
and 24,000
passing face
coefficients
(`r19_S5_RANDOM.json`).
Grand total:
23,750 legal
programs and
808,000
pair-face checks,
all **EMPIRICAL**.

M16 and D12 were **already listed** in r16 (D12 appears as
D24 under an order-based naming convention); exponent-three
order-27 also appeared as UT3(F3). They are rechecks, not
new group families. Dic3, C9, C4×C4, C3×S3, A4×C2 and the
exponent-nine order-27 group give distinct added examples.
For a high-degree nonnilpotent construction, D12 has
`[r^4,s]=r^4` with `s` a reflection. Recursively
commutating owner-independent literal echoes gives
`r^4*AND8` in 1,528 gates at lease two, checked on
all 512 input/borrower pairs at global peak two
(`r19_localize.py`). All 12,288 regular-output
states follow from the output-independent left
action. This is **EMPIRICAL** replay of a particular
word, not an efficient compiler.
The same
1,528-gate,
peak-two
commutator
construction in
nonsolvable S5
realizes a
5-cycle*AND8
inside its
perfect A5
residual on
all 512
input/borrower
pairs
(`r19_s5_localize.py`).
No useful-world
advantage is
implied by
this long
finite word.
At the odd class-three ceiling,
`r19_ut4_literal.py` constructs a
72-gate, peak-two literal word for
`E14*AND7` in UT4(F3), checked for
all 256 input/borrower pairs.
All 186,624 initial-output cases follow
from the left action. This is the last
allowed central degree predicted by §4.4,
not a test of the degree-eight exclusion.

**REPAIRED / PROVED-AT-SCOPE as a corollary:** the round-17
M16 no-lift warning is now decided by the reviewed pair
filtration. At lease four `K5(M16)=<a^4>` (order two,
the derived subgroup), which projects trivially to
`M16/[M16,M16]=C4×C2`. The latter accepts
`(a²[G,G])*AND5` since its coefficient is an order-two
element of `2(C4×C2)`. **No lift** of that table can
be accepted over M16: its degree-five coefficient would
have to lie in `<a^4>`. The independent full-group
chief-series census at n=5 yields `2^90` accepted
tables at lease two and `2^95` at lease four
(`r19_M16_n5.jsonl`); the abelian quotient permits
`2^64` tables at lease four. Every degree d<=5
M16 K_d contains `<a^4>`, so all `2^32` central
corrections to each accepted projection are available.
Consequently the image has exactly `2^63` tables:
**half** of the quotient's otherwise allowed class.
This is a no-lift counterexample to the *quotient
surjectivity shortcut*, not to R16.

**REPAIRED / PROVED-AT-SCOPE at the fixed order-32 group:**
round-17's Q8*C8 no-lift warning likewise follows from
the reviewed K test. The independently generated
group table and pair recursion give K4 of order four
containing `c²` but K5 of order two, disjoint from
the coset `c²[G,G]`; the quotient C2×C2×C4
accepts `c²[G,G]*AND5` at lease four.
No pointwise derived correction repairs that
forbidden degree-five quotient table at **any
finite lease**: `K5=[G,G]` at leases 2,4,6
by explicit pair recursion, odd leases stutter,
and `G` has exponent eight, so all further
atoms vanish after r=3. A distinct 22-gate, peak-four
word for `c²*AND4` passed literal replay
(`r19_CENTRAL_PRODUCT.json`). At central products
Q8*C_(2^k), k=3,4,5, the same disjointness
persists in all 15 lease 2/4/6/8/10 checks
(`r19_FAMILY.json`, **EMPIRICAL**). The
all-k formula K5=`[G,G]` has a new simple
class-two proof sketch in `r19_LEDGER.md`,
but its uniform extension is **CONJECTURE**
pending an independent attack, not promoted
by the finite sweep.

**REPAIRED / PROVED-AT-SCOPE, round-17 E4:** for every finite
G at **lease two**, projection from accepted G-valued tables
onto the accepted `(G/[G,G])`-valued tables is **surjective**
at every n. Let `A=G/[G,G]`. The anti-pair subgroup `A_1`
is the full preimage in `G×G` of the anti-diagonal in `A×A`.
Thus if `a in A[2]`, **every** `(g,g)` over `a` is in
`A_1=N_2=N_3`; in degree four and above the projected
`N_d` is trivial. The abelian quotient's only nonlinear
coefficients at lease two are its order-two degree-two/three
ones, and arbitrary chosen lifts of each coefficient are
constructible by the reviewed classification. This does
**not** make all pointwise `[G,G]`-valued corrections possible:
D8 central AND7 is still excluded. At higher lease the
criterion is exact but less simple: for the quotient map
`pi:G->G/[G,G]`, projection is surjective onto the whole
quotient accepted class for all input dimensions iff
`pi(K_d(G))=K_d(G/[G,G])` for **every** d, by coefficient
uniqueness and independent constructibility. M16 fails
this test at d=5, lease four.

The separate `r19_abelian.py` takes the actual two-slice gate
activations into an integer Hermite-normal-form lattice and divides
the pair-module order by the owner-difference image order. At
n=5, cyclic C4 returns `2^32` accepted functions at lease two
and `2^38` at lease four; hence direct-product C4×C4 returns
`2^64` and `2^76`. This probes the degree-five staircase that
n<=3 misses. A redundant lease-six HNF job was stopped:
C4 has exponent four, hence all atoms past r=2
are trivial and its class stabilizes at lease four.
**No completed own-statement case has disagreed.**

## Alphabet sensitivity and limitations

**WRONG outside the exact alphabet:** allowing degree-three A flips
lets C2 compute AND4 at lease two in four gates:
`A_(x1x2x3), B_(x4,1), A_(x1x2x3), B_(x4,1)`.
R16 has `K4(C2)=1` at every lease. The separate literal
`r19_alphabet.py` replays every x, initial owner and initial
output; peak two. Independently, keeping quadratic A
but allowing a degree-two B input control also gives
C2 AND4 at lease two:
`A_(x1x2), B_(x3x4,1), A_(x1x2), B_(x3x4,1)`.
The same checker replays all 64 original states.
Allowing a second borrower **and** a
borrower-to-borrower gate `b2 ^= b1*x3` lets C2 compute
AND4 at lease four in 12 gates by the four returned
subsets of `T: b1 ^= x1x2` and `S: b2 ^= b1*x3`
around `Y: z ^= b2*x4`. It replays all 128 initial
states at peak four. These are changes to the gate alphabet,
**not** counterexamples to R16. For the narrower addition of a
second independently flipped borrower with only one-owner
star-shaped B controls, the class **appears unchanged**:
normalize each B around its own borrower's offset and replace
it by a returned block, then identify both initial borrowers
with the one original b0. The resulting sequence is a legal
one-borrower word with the same output on both slices; the
opposite inclusion simply ignores the new borrower. This is
a **CONJECTURE with a proof sketch by this reviewer**, not yet
independently reviewed. `r19_two_star.jsonl` compares 12/12
C2/C3/C4 n=2,3 lease-two/four integer-module counts and finds no
extra function. With three independent
star-only borrowers, another 12/12
abelian integer-module comparisons
also match the one-borrower class
(`r19_THREE_STAR.json`), still
finite evidence only.
`r19_TWO_STAR_NONABELIAN.json`
adds 6,000 literal M16/D12/F21/
E27 exponent-nine programs at leases
2/4/6, including 1,488 B gates while both borrowers
are disturbed; normalization and collapsing the
diagonal owner slices preserved every outcome and
global lease. This is finite evidence, not a
fresh independent review of this new variant.
The argument fails to apply when there are joint
owner predicates or owner-to-owner gates.
An even simpler **joint-read** two-borrower variant needs
no owner-to-owner update: `T:b1 ^= x1x2`,
`U:b2 ^= x3x4`, `Y:z ^= b1*b2`. The four returned
blocks `Y; T Y T; U Y U; T U Y U T` compute
C2 AND4 at lease four in 12 gates for every original
borrower pair. `r19_alphabet.py` checks all 128
input/borrower/output states and a peak of four.
Deleting `O_g` changes the lease-zero/one constant class
(only the identity multiplier can then be accepted),
but for lease at least two it changes reachability
**not at all**: the returned word
`A_1 B_(1,g) A_1 B_(1,g)` applies `(g,g)`
unconditionally at peak two and replaces every `O_g`.
It does change gate count.

## Recommended `docs/RESULTS.md` label (not edited)

Replace the round-16 row's global **CONJECTURE** with
**PROVED AT SCOPE, independently attacked by an LLM reviewer**
for the exact single-borrower machine and unrestricted **finite**
gate words, including the pair recursion, count and even/odd
lease stutter. Mark the separately completed enumerations
**EMPIRICAL at their dimensions only**, not proof of the
any-n theorem. §4's closed formulas are **PROVED AT SCOPE**
at their explicit group-family assumptions; old R13/R8/R11
labels remain unchanged. Do not upgrade novelty, useful-world
advantage, physical meaning, or any two-borrower variant.

**Bibliographic checks only**, not novelty claims
(each title/authors/year was sought in web search):
Barrington and Thérien,
*Non-uniform automata over groups* (ICALP, 1987);
Barrington and Thérien,
*Finite monoids and the fine structure of NC¹*
(STOC, 1987; JACM, 1988);
Barrington, Straubing and Thérien,
*Non-Uniform Automata Over Groups*
(Information and Computation, 1990);
Péladeau and Thérien,
*Sur les langages reconnus par des groupes
nilpotents* (1988; English ECCC translation
*On the Languages Recognized by Nilpotent
Groups*, 2001);
Candela, *Notes on nilspaces:
algebraic aspects* (Discrete Analysis,
2017:15);
Host and Kra, *Nonconventional ergodic
averages and nilmanifolds* (Annals of
Mathematics, 2005). The needed finite
face-collection and component-kernel
arguments above are re-derived and
do not rely on quoting these sources.
