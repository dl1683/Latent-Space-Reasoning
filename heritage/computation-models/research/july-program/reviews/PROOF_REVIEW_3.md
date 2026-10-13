# Independent Proof Review 3 - ACDFT Residual Packing and Thompson Geometry

**Review status:** COMPLETE (2026-07-14). The file was written incrementally during
the attack. The final verdict table and promotion answers control.

**Reviewer posture:** fresh, zero-context, adversarial mathematical review limited
to proof-review packet 3. I had no prior involvement, used no private project memory
as evidence, did not use a GPU, and made no repository edit other than this file.

**Citation convention:** `PACKET:Lx-Ly`, `RESULT:Lx-Ly`, `ACDFT:Lx-Ly`,
`REVIEW1:Lx-Ly`, `REVIEW2:Lx-Ly`, `REGISTER:Lx-Ly`, `GATE0:Lx-Ly`, and
`FGATE:Lx-Ly` refer to the current line numbers of
`research/PROOF_REVIEW_PACKET_3.md`,
`research/ACDFT_REGIME_WIDENING_RESULT.md`, `research/ACDFT_LEMMA.md`,
`research/PROOF_REVIEW_1.md`, `research/PROOF_REVIEW_2.md`,
`research/ACDFT_REGIME_WIDENING.md`,
`research/ACDFT_REGIME_WIDENING_GATE0.md`, and
`research/FLAGSHIP_GATE0.md`. `THOMPSON:Lx-Ly` refers to
`research/sources/extracted/Thompson-1980-PhD-thesis-A-Complexity-Theory-for-VLSI.txt`;
page references are to the scanned thesis itself.

## Frozen review boundary

The new work is not covered by either prior certification. Review 1 upheld the
stable-direction count, the old `24 rho` packing, and the Fano/capacity statement,
but only at their exact scopes (`REVIEW1:L19-L35`, `REVIEW1:L57-L63`). Review 2
upheld only the repaired nonnegative-part area-time inference conditional on a
separately valid `(G)` (`REVIEW2:L221-L267`). Packet 3 expressly forbids either
registered wide token because the precommitted `rho <= 1/16` target was missed
(`PACKET:L17-L24`). I therefore rederive the new packing and geometry statements;
I import only the previously certified stable-subspace and Fano/algebra inputs,
and only under their certified quantifiers.

## Findings - increment 1: Target A, residual finite-grid projective packing

1. **SOUND - Vaaler's theorem is the right source statement and its scaling uses
   intrinsic dimension `d`, not ambient dimension `2|I|` (`RESULT:L47-L60`,
   `RESULT:L130-L147`; Vaaler 1979, p. 543).** The primary statement is that for
   the volume-one cube `[-1/2,1/2]^M` and every `d`-dimensional linear subspace
   `E`, the induced `d`-volume of the central section is at least one. Scaling the
   ambient cube by `2a`, with `a=1/sqrt(2N)`, scales induced volume on `E` by
   `(2a)^d`, hence
   `vol_d(E intersect [-a,a]^(2|I|)) >= (2a)^d`. Realifying the certified
   `k`-complex-dimensional singular subspace gives exactly `d=2k`; no factor
   involving `2|I|` is licensed or needed. This agrees with the source use already
   certified for the old lemma (`ACDFT:L139-L145`, `REVIEW1:L23-L25`), while the
   new separation constant is independently checked below.

2. **SOUND - the maximal-packing covering argument has the correct direction and
   constants (`RESULT:L130-L151`).** Compactness of `K` gives a finite maximal
   `tau`-separated set. Maximality covers `K` by intrinsic `d`-balls of radius
   `tau=alpha rho`. Thus

   \[
   |\mathcal P|v_d\tau^d\ge \operatorname{vol}_d(K)\ge(2a)^d,
   \]

   not the reverse. Using `v_d^(1/d) <= sqrt(2 pi e/d)` gives

   \[
   |\mathcal P|^{1/d}
   \ge \frac{\sqrt{d/(\pi eN)}}{\alpha\rho}
   \ge \frac1{\alpha\sqrt{3\pi e}\,\rho},
   \]

   where the last direction is correct because `d/N >= 1/3`. With
   `alpha=42/5`, this is exactly `(c_alpha/rho)^d`. The ambient dimension never
   enters the ball volume.

3. **SOUND - coordinate rounding lands in the unchanged grid, includes endpoints,
   and cannot push the rounded input norm above one (`RESULT:L153-L184`;
   `ACDFT:L221-L255`).** The real-coordinate spacing is
   `h=1/(sqrt(2N) Q_b)=a/Q_b`, and `+/-a` are grid points. Nearest rounding of at
   most `2|I| <= 2N` real coordinates therefore costs

   \[
   \|\widetilde w-w\|_2
   \le \sqrt{2N}\,h/2=1/(2Q_b)\le2^{-b}=r,
   \]

   including equality at `b=1`; padding every non-remote coordinate by zero is
   legal. Moreover `||w_tilde||_2^2 <= 2|I|a^2=|I|/N <= 1`, so
   `||A w_tilde||_2 <= 1`. The result relies on this certified cube calculation
   rather than assuming that rounding preserves membership in `E` (it generally
   does not).

4. **SOUND - the post-rounding separation arithmetic and every inequality
   direction are correct (`RESULT:L162-L184`).** Stable gain `1/2`, two rounding
   errors, and `tau=alpha rho` give

   \[
   s=\alpha\rho/2-2r
    =(21/5)\epsilon+(11/5)r.
   \]

   Consequently
   `s-4 epsilon=(epsilon+11r)/5 >= (epsilon+r)/5=rho/5>0`.
   Hence rounded inputs and outputs are distinct, and the strict margin needed by
   the later collision test is present for every allowed finite `b` and
   `epsilon>0`.

5. **SOUND - the phase-orbit cover works at zero output and at the angular endpoint
   (`RESULT:L186-L205`).** For `z != 0`, the orbit is a closed rectifiable curve of
   length `2 pi ||z|| <= 2 pi`; arclength sampling covers it by at most
   `ceil(2 pi ||z||/eta) <= 1+2 pi/eta` Euclidean `eta`-balls. The identical
   endpoints at angles `0` and `2 pi` create no extra center. For `z=0`, the orbit
   is a singleton and one ball suffices. Although `d_coh` is written using the
   half-open interval `[0,2 pi)`, its continuous periodic objective attains a
   minimum on `[0,2 pi]`, and a minimizer at `2 pi` is also the point `0`; every
   conflict neighbor is therefore within `2 epsilon+eta` of a cover center.

6. **SOUND - the same-center contradiction, maximum degree, off-by-one slack, and
   strict independent-set separation are valid (`RESULT:L192-L217`).** With
   `eta=(s-4 epsilon)/4`, two distinct neighbors assigned to one center would have
   Euclidean distance at most

   \[
   4\epsilon+2\eta=(s+4\epsilon)/2<s.
   \]

   Thus neighbors inject into orbit-cover centers. Since
   `eta >= rho/20`, the degree is at most `1+40 pi/rho`. Greedy independence
   needs a denominator at most `Delta+1 <= 2+40 pi/rho`; the displayed
   `3+40 pi/rho` is one further safe unit of slack, not a missing off-by-one.
   Because the conflict graph includes all pairs with
   `d_coh <= 2 epsilon`, nonedges have the required strict separation
   `d_coh > 2 epsilon`.

7. **SOUND - the singleton/positive-part passage never treats a fractional lower
   bound as a codeword (`RESULT:L214-L217`).** If

   \[
   H=d\log_2(c_\alpha/\rho)-\log_2(3+40\pi/\rho)>0,
   \]

   the construction gives `|C_gamma| >= 2^H > 1` and hence
   `log_2|C_gamma| >= H`. If `H<=0`, any one rounded grid point supplies the
   singleton with logarithm zero. At `H=0` the singleton proves equality. This is
   exactly `[H]^+`; no logarithm of a subunit cardinality estimate is used.

8. **SOUND - the smaller-interval corollary `(R-W-PACK)` is exact, including the
   equality endpoint (`RESULT:L219-L236`).** At
   `rho=rho_star=c_alpha 2^(-1/32)`, and hence throughout the stated interval,
   `log_2(c_alpha/rho) >= 1/32`. Also
   `3+40 pi/rho <= 128/rho` is equivalent to `3rho+40 pi <=128`; the stronger
   estimate at `rho<1/16`, using `pi<22/7`, is
   `3rho+40pi < 3/16+880/7 <128`. Replacing the first term by `d/32` and the
   subtracted logarithm by the larger `log_2(128/rho)` lowers the inner expression,
   so monotonicity of the positive part gives the claimed direction.

9. **SOUND - the residual is a genuine widening of the old certified positive
   interval, but not the registered widening (`RESULT:L15-L25`, `RESULT:L102-L126`,
   `RESULT:L462-L473`; `REGISTER:L193-L205`).** Exact arithmetic gives
   `rho_star/c_V=(120/42)2^(-1/32)=2.795920...>1`, so it is not a restatement of
   the old `rho<c_V` regime. The fixed finite witness `b=7`,
   `epsilon=1/128` has `rho=1/64`, which lies above `c_V` and below `rho_star`.
   But `rho_star=0.023016...<1/16`; the frozen six-bit boundary has
   `rho=1/16` and receives no theorem. The artifact consistently calls this a
   residual and fires the target-miss token.

10. **SOUND - Target A's degenerate and equality cases do not break the statement.**
    At `b=1`, `rho>1/2>c_alpha`, so the quantitative right side is zero and the
    singleton branch applies. As `epsilon` decreases to zero through positive
    values, `s-4epsilon >= rho/5` stays positive. The formal weakest dimension
    ratio `d/N=1/3` is already the equality case used in Finding 2. At
    `rho=rho_star`, Finding 8 includes equality. A zero output is covered by
    Finding 5 and cannot create excess degree. No cut-dependent real vector,
    singular basis, or selected subcode reaches the architecture: all continuous
    objects remain analyst-side witnesses, while actual inputs belong to the
    pre-existing finite grid (`RESULT:L62-L65`; `ACDFT:L173-L176`,
    `REVIEW1:L33-L35`).

**Target A disposition: UPHELD.** The exact candidate statement `(R-PACK)` is
supported for every
`N=2^m`, `m>=1`, `b>=1`, `epsilon>0`, every better directional DFT cross block
certified by ACDFT Lemma 1, its stable real dimension `d=2k>=N/3`, the unchanged
grid, one global-phase quotient, and analyst-only cut-dependent subcode selection.
`(R-W-PACK)` is supported only when `rho=epsilon+2^(-b)<=rho_star`. Nothing here
supports `rho<=1/16`.

## Findings - increment 2: Target B, residual cut and conditional area-time assembly

1. **SOUND - the reviewed hard-distribution/Fano argument imports without changing
   a quantifier (`RESULT:L238-L263`; `ACDFT:L297-L386`).** Fix an admitted
   architecture and a balanced named-output cut. Target A supplies a finite packing
   on the better remote-input side for that same cut. Put a uniform hard index on
   this analyst-selected subcode and fix the opposite-side input to zero. The
   architecture's per-input error bound implies average decoding error at most
   `delta`; restriction to the remote output coordinates preserves the same
   execution-dependent global phase. Strict `>2 epsilon` packing separation makes
   nearest-projective decoding valid on every success event. Conditional Fano,
   data processing, and the certified directed cut-capacity inequality then apply
   exactly as in the old proof. Feedback, noise, local randomness, coding, and
   input-independent shared randomness remain inside the complete view and
   operational capacity certificate (`ACDFT:L348-L397`; `REVIEW1:L31-L35`). The
   subcode is selected only after the fixed machine and cut and is not advice.

2. **SOUND - the residual cut lower bound has the correct constant and
   positive-part direction (`RESULT:L240-L251`).** Target A gives

   \[
   \log_2|\mathcal C_\gamma|
   \ge [d/32-\log_2(128/\rho)]^+.
   \]

   Since `d>=N/3`, `d/32>=N/96`; the map `x -> [x]^+` is monotone. Substitution
   into Fano therefore yields exactly

   \[
   TC_{\rm cut}\ge(1-\delta)
   [N/96-\log_2(128/\rho)]^+-h_2(\delta)=Q_{\rm res}
   \]

   for `rho<=rho_star`, with no claim outside that interval.

3. **SOUND - `(R-AT|G)` never squares a negative lower bound and imports Review
   2 only within its certified algebraic scope (`RESULT:L254-L263`;
   `REVIEW2:L39-L60`, `REVIEW2:L221-L267`).** Under `A>=0`, `T>=0`, a finite
   `kappa>0`, and a separately valid balanced-cut law
   `C_cut(gamma)<=kappa sqrt(A)`, the left side `T kappa sqrt(A)` is
   nonnegative and at least `Q_res`; hence it is at least `[Q_res]^+`. Squaring
   only then gives

   \[
   AT^2\ge([Q_{\rm res}]^+)^2/\kappa^2.
   \]

   No premise in Target A or the Fano proof is used to manufacture `(G)`.

4. **SOUND - all sign and zero-resource boundaries return the right result.** If
   `Q_res<0` or `Q_res=0`, the area-time display is only `AT^2>=0`. If
   `Q_res>0`, the combined premises exclude `A=0` and `T=0`. At `delta=0`,
   `h_2(delta)=0`; as `delta` approaches `1/2` from below, Fano remains valid but
   the bound weakens. The endpoint `delta=1/2` is not certified. A singleton
   packing makes the inner positive part zero and hence `Q_res=-h_2(delta)<=0`,
   so the repaired outer positive part again prevents the Review-1 sign defect.

5. **SOUND - the registered boundary is explicitly outside the theorem
   (`RESULT:L462-L467`; `PACKET:L195-L197`).** At
   `(b,epsilon,delta)=(6,3/64,1/3)`, `rho=1/16>rho_star`. Neither `(R-RCUT)` nor
   `(R-AT|G)` applies there. Independently, the conjectural target arithmetic is
   `N>1188.234600...`, whose first allowed power of two is `2048`; this confirms
   only that the missed target was nonempty, not that its packing exists.

**Target B disposition: UPHELD.** `(R-RCUT)` is certifiable only for
the exact admitted-architecture, worst-case coherent-output, finite operational
interactive cut-capacity, balanced named-output-cut, and failure scopes certified
for ACDFT-RCUT, with the new additional premise `rho<=rho_star` and the displayed
residual constants. `(R-AT|G)` is certifiable only conditionally on a separately
valid law for the same cut with `A,T>=0` and `0<kappa<infinity`.

## Findings - increment 3: Target C, delimited proof-template obstruction

1. **SOUND - `s>4 epsilon` is necessary for the explicitly delimited collision
   injection, not for projective packing in general (`RESULT:L267-L297`).** In
   that injection, two `2 epsilon`-neighbors assigned to one orbit-cover center of
   radius `eta>0` can be only bounded by `4 epsilon+2 eta`. A strict contradiction
   with the available Euclidean minimum separation exists for some positive
   `eta` exactly when `s>4 epsilon`. Equality supplies no positive cover radius.
   This necessity belongs only to the stated triangle-inequality certificate.

2. **SOUND - the limiting quantifier is legal and gives the stated threshold.**
   From the template's only separation certificate

   \[
   s\ge \alpha\rho/2-2r,
   \]

   requiring `s>4 epsilon=4(rho-r)` gives
   `alpha>8-4r/rho`. At fixed `rho=1/16`, take any sequence of finite integers
   `b -> infinity` with `r=2^(-b)` and
   `epsilon=1/16-r>0`. Every architecture in the sequence has finite precision,
   while `r/rho -> 0`; uniform certification over all allowed finite `b` thus
   requires asymptotically `alpha>=8`. No single infinite-precision architecture
   is introduced.

3. **SOUND - the Vaaler base at the limiting threshold is below one.** At
   `alpha=8` and `rho=1/16`, the continuous-volume base is exactly

   \[
   \frac1{8(1/16)\sqrt{3\pi e}}=
   \frac2{\sqrt{3\pi e}}=0.3951365\ldots<1.
   \]

   Thus Vaaler's lower bound supplies less than one before the polynomial phase
   loss and cannot certify positive exponential rate at that boundary.

4. **SOUND - the artifact does not upgrade a route failure into a packing upper
   bound (`RESULT:L294-L313`).** The conclusion is expressly limited to the
   volume + per-vector rounding + minimum-separation collision template. Discrete
   small-ball or structure-sensitive proofs remain open, no certified DFT
   block/grid counterexample family is supplied, and
   `ACDFT-WIDE-CONSTANT-BARRIER` is withheld.

**Target C disposition: UPHELD** at exactly the delimited proof-template
scope above. It is not a theorem about the optimal packing number.

## Findings - increment 4: Target D, Thompson-subclass geometry discharge

1. **SOUND - Thompson's bisection definition gives exact named-output balance for
   the allowed even `N` (Thompson thesis p. 53; `THOMPSON:L1057-L1076`).** For an
   arbitrary distinguished set `S`, Thompson partitions it as
   `S=S_1 union S_2`, with `S_i` lying on vertex side `V_i` and
   `|S_1| <= |S_2| <= |S_1|+1`; every cross-side path meets the removed edge set.
   Here `S` is the set of `N=2^m`, `m>=1`, distinct named-output vertices. Since
   `N` is even, the inequalities force `|S_1|=|S_2|=N/2`, not merely approximate
   balance.

2. **MINOR - Theorem 2 does not apply verbatim to outputs, but its proof has a
   valid distinguished-set relabeling (`RESULT:L383-L393`;
   `THOMPSON:L1063-L1076`, `THOMPSON:L1087-L1097`).** Thompson states Theorem 2
   for source nodes. But bisection width is first defined for any vertex subset
   `S`, and the geometric proof uses only membership in `S` and its balance, not
   source semantics. Re-running it with the named-output set gives occupied area
   at least `w^2/4` with the same constant. `RESULT:L383-L385` should attribute
   this to the proof of Theorem 2 applied to the output set, rather than suggesting
   the published theorem statement is verbatim. This changes no mathematics or
   constant.

3. **SOUND - the `q^2` multilayer conversion is the correct route and charges
   nodes, vias, fanout, and crossovers under the embedding admission
   (`RESULT:L342-L352`, `RESULT:L383-L393`; `THOMPSON:L721-L733`).** Thompson maps
   each physical unit of a `q`-layer chip into `q^2` one-layer unit squares,
   representing layers, interlayer connectivity, and fanout nodes. After scaling
   pitch `p` to unit pitch, the explicit admission gives at most `q^2 A/p^2`
   occupied one-layer squares. Combining this upper representation count with
   Theorem 2 gives

   \[
   w^2/4\le q^2A/p^2,
   \qquad w\le(2q/p)\sqrt A.
   \]

   Any layout with extra boundary, via, node, or crossover overhead beyond that
   mapping count is outside the exact constant. The occupied-area Theorem 2, not
   bounding-rectangle Theorem 1, is the right source. Theorem 3 is unnecessary and
   as stated concerns all vertices under different overlap and degree rules
   (`THOMPSON:L1230-L1246`).

4. **SOUND - a minimum distinguished-set edge bisection supplies an admissible
   ACDFT vertex cut (`RESULT:L396-L403`; `FGATE:L78-L101`).** Thompson's
   definition supplies a partition `V=L union R` after removing a cutset. Every
   graph edge crossing that partition lies in the cutset, so
    `|\partial\gamma|<=w` even if the minimum cutset contains redundant same-side
   edges. ACDFT expressly defines its operational object for a vertex cut
   `gamma=(L,R)`, and Finding 1 gives the required output balance. Cut fidelity
   ensures that removing the represented cross edges removes every input-dependent
   physical communication path between the two views.

5. **SOUND - the operational cutset premise is stronger than per-edge finiteness
   and does not re-assume `(G)` (`RESULT:L346-L364`, `RESULT:L396-L407`).** The
   admission law

   \[
   C_{\rm cut}(\gamma)\le\sum_{e\in\partial\gamma}c_e,
   \qquad c_e\le\bar c
   \]

   applies to full joint interactive two-way cut capacity. It must be separately
   certified with correlation or shared media; it is not inferred from isolated
   edge rates. But it contains neither balanced-cut existence nor a `sqrt(A)`
   bound. Findings 1-3 supply those, giving
   `C_cut(gamma)<=bar c w<=(2q bar c/p)sqrt(A)`. Broadcast, buses, capacitive or
   wireless coupling, shared fields, and hyperedges fail admission unless they are
   represented and the joint cutset inequality is proved.

6. **SOUND - units and the one-way/two-way convention are consistent
   (`RESULT:L329-L332`, `RESULT:L342-L364`, `RESULT:L429-L442`).** `A` has units
   length squared, `p` length, `q` is dimensionless, and `bar c` is declared total
   two-way reliable bits per unit time per represented edge. Hence
   `(2q bar c/p)sqrt(A)` has the units of `C_cut`, with no extra factor two. If
   `q`, `p`, or `bar c` scales with `N`, that dependence remains in
   `kappa_N=2q_N bar c_N/p_N`; no constant-`kappa` asymptotic follows without
   uniform control.

7. **MINOR - the zero-rate edge case needs a downstream `kappa>0` clarification,
   but does not falsify `(G-T)` (`RESULT:L376-L380`; `ACDFT:L457-L495`).** If the
   declared `bar c>0`, then `kappa=2q bar c/p` is finite and positive as required
   by the certified area-time division. If `bar c=0`, `(G-T)` is still true and
   forces `C_cut=0`, but the displayed `kappa` is zero and cannot be inserted into
   `(R-AT|G)`. The same device satisfies `(G)` with any positive slack rate bound,
   so promotion wording must either use `0<bar c<infinity` for the displayed
   `kappa` or state the zero-rate branch separately. No positive-case constant or
   architecture changes.

8. **SOUND - the real-device scope is formal and narrow, not a matched-device
   theorem (`RESULT:L409-L427`).** The certifiable class consists of finite routed
   electronic networks with distinct named-output vertices, a legal finite
   `q`-layer pitch-`p` Thompson embedding satisfying the exact mapping count, no
   unrepresented input-dependent coupling, and a proved total two-way joint
   cutset-capacity certificate bounded by finite per-edge `bar c`. Conventional
   point-to-point wired VLSI can qualify only after those facts are proved for its
   actual channel and layout. No fabricated device or product is verified here.
   Free-space or shared-field photonics, wireless or capacitive global coupling,
   free broadcast hyperedges, unbounded multiplexing, and nonadditive cut media do
   not inherit the result.

**Target D disposition: UPHELD**, with the two nonmaterial precision
qualifications in Findings 2 and 7. `(G-T)` is certifiable for the formal subclass
in Finding 8, with `q` a finite positive integer, `p>0`, total two-way
`0<bar c<infinity` for the displayed positive `kappa`, and every dependence on `N`
retained. This discharges `(G)` on that formal subclass; it does not establish that
an unexamined real device belongs to it.

## Findings - increment 5: Target E, source scope, tokens, and honest standing

1. **SOUND - all mandated independent arithmetic checks agree with the exact
   displays (`PACKET:L281-L298`; `RESULT:L449-L467`).** Recalculation gives
   `c_V=0.00823201121893689`, `c_alpha=0.0235200320541054`,
   `rho_star=0.0230160462675547`,
   `rho_star/c_V=(120/42)2^(-1/32)=2.79592017739343`,
   `2/sqrt(3 pi e)=0.395136538508970`, and
   `3/16+40pi=125.851206143592<128`. At the conjectural wide boundary,
   `h_2(1/3)` gives `(W-POS) N>1188.23460010385`; the first allowed power of two
   is `2048`. These values match, but do not elevate, the empirical ledger.

2. **SOUND - the Gate-0 occupation claims remain conservatively typed
   (`GATE0:L34-L66`, `GATE0:L91-L111`).** Vaaler supplies only continuous
   section volume. Ball 1986 is an upper hyperplane-section bound, hence the wrong
   direction. Rudelson-Vershynin has an unspecified absolute constant and no
   one-phase finite-grid pruning. Rankin and Kabatiansky-Levenshtein concern free
   spherical codes; Conway-Hardin-Sloane and Barg-Nogin concern freely chosen
   Grassmannian points; Love-Heath-Strohmer designs a predetermined channel
   codebook shared with the transmitter. None gives the frozen grid under every
   stable block with target constants and no machine advice. Tyagi-Reif and Yao
   remain `SOURCE-INCOMPLETE`; Lipton-Sedgewick remains exact crossing/counting,
   and Paturi-Simon remains Boolean sign geometry (`GATE0:L93-L99`). No novelty
   theorem follows from the checked record.

3. **SOUND - every token disposition follows the proof status
   (`RESULT:L27-L43`, `RESULT:L478-L487`).** The precommitted wide target was
   missed, so `ACDFT-WIDE-TARGET-MISS` fires and neither wide PROVED token can
   issue. No matching subexponential block/grid family was proved, so
   `ACDFT-WIDE-CONSTANT-BARRIER` does not fire. The old reviewed narrow theorem is
   untouched. The broad class still assumes `(G)`; only Target D's exact formal
   subclass discharges it. A fixed finite witness outside the old interval prevents
   `WORD-PRECISION-ONLY`, while real subspace objects remain analyst witnesses, so
   no infinity leak enters at the stated scope.

4. **SOUND - the artifact does not claim more than the proofs show
   (`RESULT:L489-L512`).** Almost three times wider means the exact `2.7959...`
   ratio relative to the old radius interval, not success at `1/16`. Geometry is
   limited to a routed-edge subclass and excludes broad photonic or analog
   inheritance. Milestone D remains partial; there is no matched device
   adjudication, optical advantage, positive CAN theorem, new computation model,
   unconditional broad area-time theorem, or mission completion.

**Target E disposition: UPHELD.** No target-miss, barrier, inherited
theorem, broad-geometry, or mission token requires correction.

## Three strongest attacks mounted

1. **Dimension-loss / volume-direction attack - failed to overturn Target A.** I
   recomputed Vaaler scaling and the covering inequality in intrinsic `d`-volume,
   then forced the weakest `d/N=1/3` ratio. No ambient-dimension factor or reversed
   inequality appears; the exact base is `1/(alpha sqrt(3 pi e) rho)`.
2. **Rounding / phase-collision / off-by-one attack - failed to overturn Target A.**
   I used maximum coordinate support, endpoint rounding, zero output, an optimizing
   phase at the half-open endpoint, and the worst pair of conflict neighbors. The
   strict `s>4 epsilon` margin survives, and the greedy denominator contains one
   extra safe unit rather than missing one.
3. **Thompson scope and capacity-laundering attack - failed to overturn Target D,
   with two MINOR qualifications.** The published theorem statement names source
   nodes, but its set-generic geometric proof re-runs for the even named-output set.
   The operational cutset law is a real extra admission, not `(G)` renamed; geometry
   supplies the missing `sqrt(A)` step. Promotion must state the output-set
   adaptation and keep `kappa>0` (or branch separately at zero edge rate). No broad
   photonic, analog, shared-field, or matched-real-device conclusion survives.

## Final verdict table

| Target | Verdict (`UPHELD` / `REPAIR` / `OVERTURNED`) | First failing line or invariant | Exact certified scope, if any |
|---|---|---|---|
| A - `(R-PACK)` and `(R-W-PACK)` | **UPHELD** | None | `(R-PACK)`: every allowed power-of-two `N`, finite `b>=1`, `epsilon>0`, certified better DFT cross block with `d=2k>=N/3`, unchanged grid, one global phase, analyst-only cut-dependent subcode. `(R-W-PACK)` additionally requires `rho<=rho_star`. |
| B - `(R-RCUT)` and `(R-AT\|G)` | **UPHELD** | None | Exact inherited worst-case/Fano/interactive-capacity scope, `0<=delta<1/2`, and `rho<=rho_star`; area-time only for the same balanced cut under separate `(G)`, `A,T>=0`, and finite `kappa>0`. |
| C - delimited proof-template obstruction | **UPHELD** | None | Only the Vaaler-volume + error-`r` rounding + stated minimum-separation collision certificate; no upper bound on actual projective packing. |
| D - `(G-T)` | **UPHELD** | No mathematical failure; MINOR source-attribution precision at `RESULT:L383-L385` and zero-rate downstream precision at `RESULT:L376-L380`. | Exact finite routed graph subclass with distinct named outputs, legal finite positive-layer pitch-`p` Thompson mapping, cut fidelity, and proved joint total-two-way additive cutset certificate. For the displayed positive `kappa`, require `0<bar c<infinity`; retain all `N`-dependence. |
| E - tokens and mission standing | **UPHELD** | None | Target miss fires; constant barrier and both wide proof tokens remain withheld; old theorem survives; broad `(G)` and mission remain open. |

## Explicit promotion answers

1. **Yes.** The repository may label Target A
   `ACDFT-RESIDUAL-PACK-PROVED-AT-SCOPE`, but only with all quantifiers in the
   Target-A table row and with `(R-W-PACK)` restricted to
   `rho<=rho_star=5/(42 2^(1/32) sqrt(3 pi e))`. This is not a wide token.
2. **Yes.** It may label Target B
   `ACDFT-RESIDUAL-RCUT-PROVED-AT-SCOPE` only at the inherited admitted-machine,
   per-input failure, every-balanced-cut operational-capacity scope, with
   `rho<=rho_star`. The area-time consequence remains conditional on a separately
   valid same-cut `(G)` and uses the outer nonnegative part.
3. **Yes.** It may label Target D
   `ACDFT-THOMPSON-G-PROVED-AT-SCOPE` for the exact formal graph subclass in the
   Target-D row. The promotion sentence must identify the output-set adaptation of
   Thompson's proof, total two-way edge rates, `q`, `p`, and positive finite
   `bar c`, preserve any `N`-dependence in `kappa_N`, and list the excluded
   shared-field and nonadditive devices.
4. **No finding alters `ACDFT-WIDE-TARGET-MISS`.** It remains mandatory because
   `rho_star<1/16` and the registered boundary receives no packing theorem.
5. **No finding justifies `ACDFT-WIDE-CONSTANT-BARRIER`.** Target C blocks only one
   proof template and provides no matching packing upper bound or certified bad
   block/grid family.
6. **Exact inheriting device class:** finite routed graph-embeddable electronics
   satisfying every admission clause in Target D, including a proved joint
   operational cutset inequality. Point-to-point wired VLSI may qualify after a
   device-specific certificate. No actual fabricated device is certified in this
   packet. Free-space or shared-field photonics, wireless or capacitive global
   coupling, free broadcast hyperedges, unbounded multiplexing, and nonadditive
   cut media are excluded absent another theorem or charged uniform compiler.

## Final standing

All five packet targets are upheld at the exact scopes above. The two MINOR Target-D
precision findings change neither `(G-T)` nor its positive-rate constant and do not
constitute a material repair. The residual packing and residual cut theorem may be
promoted only under their new scoped tokens; the registered wide tokens remain
forbidden. This is a packet-3 proof review, not a terminal repository review under
Invariant 2.
