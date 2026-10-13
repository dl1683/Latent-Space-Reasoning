# CRC approximation by finite vertical range

2026-09-08. UNREVIEWED proof draft under CRC_APPROXIMATION_V1_REGISTRATION.json.
This proposed construction is for the registered a_c<=1/2 family only.
No numerical code, implementation, novelty or speedup claim.

Let C_L equal C with entries at vertical distances greater than L set to zero,
where L>=1 is an integer. Absolute row sums give

    ||C-C_L||_2 <= 2^(1-L),  ||C_L||_2 <= 3.

The first follows from the two geometric tails; the matrix is symmetric,
so its spectral norm is bounded by its maximum absolute row sum. Using the
reviewed ||R||,||C||<=3 and ||CRC||>=1/27,

    ||CRC-C_L R C_L||_2 <= 18 ||C-C_L||_2 <= 36*2^(-L),
    relative error <= 972*2^(-L).

This is a whole-operator telescoping bound. It neither adds independent
cross-block errors nor assumes compatible low-rank approximations.

## Finite-offset expansion

For each pair p,q in [-L,L], put delta=p+q. The corresponding contribution
to output (c,y), with t=y-delta and r=y-q, is

    1_(0<=t,r<s) a_c^|q| sum_d b^|c-d| a_d^|p| xi_(d,t).

This follows by expanding the intermediate row of C_L R C_L. Sum over all
p,q. It includes same-row, earlier-row and later-row inputs without using
the other flow's states. Let j=|p| and define z_j(c,t) as the inner row sum.

## Candidate legal states, per j=0,...,L in each flow

For increasing flow, let u be the neighbor above and hprev the earlier
horizontal neighbor. Missing-neighbor states are zero. Use

    X_j(v) = a_c^j xi_v,
    P_j(v) = a_c^j xi_v + b P_j(hprev),
    S_j(v) = X_j(u) + b S_j(hprev),
    Z_j,1(v) = P_j(u) + b S_j(hprev),
    Z_j,k(v) = Z_j,k-1(u), 2<=k<=2L-1.

The reviewed alternating-row prefix identity implies
Z_j,k(c,y)=z_j(c,y-k). Thus for delta=1 the output obtains z_j(c,y-1)
from P_j(u)+bS_j(hprev), and for 2<=delta<=2L it obtains it from
Z_j,delta-1(u). These are neighboring-state reads, not own-state outputs.
The decreasing flow mirrors these formulas with below/later horizontal
neighbors and delivers negative delta. For delta=0, use

    z_j(c,y) = a_c^j xi_v + bP_j(hprev) + bP'_j(hlater).

Multiply each legal output expression by the known coefficient a_c^|q|
and the intermediate-row boundary mask, then sum the p,q terms. The
input-row mask may also be included explicitly; zero boundary propagation
already enforces it. No divisions or cross-flow state recurrence occurs.

The proposed width is (L+1)(2L+2) per flow: X,P,S and 2L-1 delay states.
The bound may overallocate when L exceeds s, which does not invalidate an
upper bound. Sparse recurrence and output term counts are O(N L^2), as is
the arithmetic evaluation count; dense transition storage is unnecessary.
For the registered rational parameters, individual power coefficients have
O(L log s) bits, and expanded sparse descriptors cost at most
O(N L^2 (L log s + log L)) bits including aggregated output coefficients
and indices. Exact intermediate bit runtime and floating-point stability
are not included in these arithmetic counts.

Choosing L>=ceil(log2(972/eps)) would give width O(log^2(1/eps)) and
O(N log^2(1/eps)) arithmetic uniformly in s. These claims require independent
review. No optimality or linear-log width is asserted. The retained CRC word
already applies in O(N) arithmetic, so this is an approximation construction
and scope check, not a useful speedup or a completed mission tier.

Required attack cases before reliance: L=1, s=2; s=6 with reversed parameter
copy; offsets whose intermediate row lies outside the grid while endpoints
lie inside; both signs of delta; the same-row delta=0 terms; absent horizontal
neighbors; L>=s. A near-one parameter changes the tail bound and is outside
the dimension-uniform registered-family statement.

## Complete output notation and boundary audit

Define E_j,delta(v) by the following legal output expressions. For delta=0,
use a_c^j xi_v+bP_j(hprev)+bPprime_j(hlater). For delta=1, use
P_j(u)+bS_j(hprev). For 2<=delta<=2L, use Z_j,delta-1(u).
For negative delta, use the mirrored decreasing expressions with the neighbor
below and later horizontal neighbor. Put

    eta_(c,y) = sum_(p,q=-L)^L 1_(0<=y-p-q<s) 1_(0<=y-q<s)
                   * a_c^|q| E_|p|,p+q(c,y).

Both masks are known local output coefficients. Multiplying a legal output
expression by them and summing preserves Definition 3. Terms involving the
local input collect into its one scalar diagonal coefficient. The remaining
terms collect into linear maps on allowed neighboring states only. This
formula contains no read of the output vertex own state.

For each j, P_j accumulates the current weighted row in the current horizontal
direction, and S_j accumulates the previous weighted row in that direction.
P_j on the neighbor above uses the opposite direction. Thus P_j(u)+S_j(v)
counts the whole previous weighted row plus its local input X_j(u).
Substituting S_j(v)=X_j(u)+bS_j(hprev) removes the overlap and proves the
Z_j,1 identity without an own-state dependence. Induction in k proves every
delay identity. On row zero S and all Z vanish by horizontal and vertical
induction. Reversing order gives the identical future-row proof for g.

At L=1,s=2, C_L=C and the offset sum enumerates exactly the two possible
intermediate rows for each input/output pair. Delta=2 or -2 has no valid
endpoints and is masked out. For delta=0 at y=0, p=1,q=-1 has intermediate
row one and is retained; p=-1,q=1 has intermediate row minus one and is
removed. At y=1 these boundary roles reverse. Thus valid endpoints alone
would be insufficient, and the intermediate-row mask is essential.

At s=6 with the declared reversed-copy parameters, every j simply weights
its own column input. The proof never assumes alignment or distinctness
of those parameters. At y=0 the same nonzero p=-q check applies, and at
y=5 the opposite boundary excludes the corresponding outside intermediate
row. For any L, masks enumerate exactly the in-grid intermediate rows.

If hprev is absent at the first visited horizontal vertex, P_j(u) already
contains the whole previous weighted row, since that column was the last
visited in the previous row. The mirrored assertion covers g. There is no
wraparound or hidden boundary state. If L>=s, C_L=C exactly; redundant
offsets vanish by the masks and oversized delay chains carry zeros outside
the grid. Small coefficients do not prevent exact intermediate bit growth.

## Source comparison attempted within the registered gate

Roesser (1975), A discrete state-space model for linear image processing,
DOI 10.1109/TAC.1975.1100844: two targeted title/author PDF searches returned
secondary summaries and later works citing the original. Direct DOI opening
returned a non-retryable safe-open error. The original model/statement was
not retrieved in this attempt. Delay lines and linear filter state realizations
are classical ingredients; no novelty or LITERATURE_SETTLES token follows.
The exact alternating-row two-flow comparison remains an explicit source gap.
Inherited Kung and DFZ obligations also remain unresolved. This proof imports
no historical realization theorem; its recurrences and error bound are given
in full. The actual G-SS model and prior CR identity are in the reviewed packet.
