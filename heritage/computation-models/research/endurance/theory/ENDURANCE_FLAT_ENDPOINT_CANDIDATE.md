# Flat-log endpoint capability and wear-envelope bookkeeping

2026-10-04 CANDIDATE PROOFS, ENDURANCE-FLAT-ENDPOINT-1.
This concerns the already admitted UNPRICED ENDPOINT RELAXATION only. It is not
an elementary indexed-machine CAN: S/CODE/controller/generation/cleanup and
worst-command certification are all owed. No extra repair of the running
journal compiler, numeric experiment, novelty or mission claim follows.

## Finite contract and design/infinity audit

N=2k, integer k>=2,L>=2. Set d=ceil(log2 k), M=floor(k/(d+1)), E=floor(L/2),
H=ME. Since d+1<=k, M>=1. Persistent binary cells start zero/unworn. Each
actual change, including init and intermediate writes, costs one wear unit;
exceeding L is INVALID. Every raw x and FLIP word of length<=H is promised;
QUERY interleavings and arbitrarily many terminal queries are read-only.

Reserve k base cells and M records of d index bits plus one valid flag,
total k+M(d+1)<=N. Unused cells remain zero. Finite integers, finite loops,
parameter-only layout and current operation/index; no stage/wear/clock/original
input oracle. A uniform semantic algorithm defines I,d_i,T_i; no arbitrary
advice or hidden table is invoked to claim a machine resource bound. Proof
uses the historical epoch number only for accounting; the algorithm never
stores or obtains it. This is an endpoint/design audit, not a compiler gate.

## Claim A: endpoint lifetime H=ME

Initializer I(x) writes the k base bits directly to x, leaving all records zero.
At every completed boundary, valid flags are a prefix of length j<M; every
invalid record is entirely zero. Query scans all M records and returns base[i]
XOR the parity of valid records whose index equals i. Index comparisons use
ordinary finite integer equality; access to raw x ends after initialization.

FLIP(i) obtains j from the valid-prefix flags, writes the d-bit index i directly
into record j, then sets its flag. If j+1<M, return. If j+1=M, for each logical
coordinate h=0,...,k-1, scan all M records, compute occurrence parity p_h,
read base[h] and write base[h] XOR p_h EXACTLY ONCE. Keep records unchanged
until every base coordinate has been processed. Then write every record bit,
including its valid flag, to zero, and return only after all clearing is done.
No mid-command correctness/crash/overlap promise is made. The full-log state
is transient, never a completed boundary. The decoder/maps can be extended
arbitrarily outside the promised valid boundary states.

Induction: partial filling represents base XOR the parity vector of appended
flips. A complete fold transfers precisely that vector to base; final clearing
removes its log representation, restoring the empty-log boundary invariant.
Thus every raw input, repeated hot-coordinate update, cancellation, and query
interleaving has exact logical semantics. The updater is stationary: j is
recovered from visible flags, and the same empty log repeats the same routine
without a surviving epoch counter.

In each completed epoch every record bit changes at most once during filling
and at most once during clearing. Each base cell changes at most once in the
fold. After E epochs, record wear<=2E<=L; base wear<=1+E<=L including init.
At a partial final epoch r<E, record wear<=2r+1<=2E-1<=L and base wear<=1+r.
Unused-cell wear is zero. This accounts for ALL intermediate writes, not only
endpoint XORs. At H the E-th clear/fold is complete. Subsequent QUERY consumes
no wear and remains exact indefinitely. All loops terminate finitely.

Consequently this endpoint family has H=ME. For growing k and L>=2,
M=Theta(k/log k), E=Theta(L), hence H=Theta(NL/log N). It therefore refutes a
universal O(NL/log^2 N) ceiling for this UNPRICED endpoint relaxation. It does
not refute such a ceiling with polylogarithmic elementary command work or
an unconstructed fixed-controller/CODE/preparation contract.

Visible persistent-bit access upper counts for the literal scanning routine:
init k writes; QUERY at most 1+M(d+1) reads; a nonfold FLIP at most M flag reads
plus d+1 writes. A fold FLIP additionally uses at most kM(d+1) record reads,
2k base accesses, and M(d+1) clearing writes. These are only persistent-port
counts. Private comparison, counters, code fetch, generation and self-erasure
are NOT certified by these numbers.

Index logs/parity folding already occur in Flash2014 Section VI. This candidate
is a strict binary reuse/cold-stationarity statement, with PRIORITY OPEN;
its coarse scans supply no efficient primary journal compiler.

## Claim B: exact wear-envelope recurrence, no logarithmic growth theorem

In any admitted stationary endpoint relaxation with guaranteed horizon H,
write c(s,i)=s XOR T_i(s), coordinatewise in {0,1}^N. For any state reachable
through <=t flips, let W_j(s,t) be the maximum ENDPOINT wear in coordinate j
over ALL initial raw inputs and endpoint histories of length<=t reaching s.
Endpoint initialization wear is I(x), and each update adds c. These wear
vectors undercount actual changes, including undone intermediates. All
reachable endpoint wears are <=L, so their coordinatewise maxima are <=L.
The maxima are mathematical, finite, and unavailable to the controller.

For t<H, stationarity requires the same T_i(s) for every such history.
Extending each history by i shows

    W(s,t)+c(s,i)<=L*1,
    W(T_i(s),t+1)>=W(s,t)+c(s,i).

The inequalities are COORDINATEWISE: each maximum has its own witnessing
history, and the conclusion does not assume simultaneous attainability of
all maxima. Promised execution/strict INVALID ensure all extensions exist.
Let G(s,t)=sum_j W_j(s,t) and

    J(s,i,t)=G(T_i(s),t+1)-G(s,t)-sum_j c_j(s,i)>=0.

Along any promised length-H trajectory, telescope these defining equalities:

    sum_t sum_j c_j(s_t,i_t)+sum_t J(s_t,i_t,t)
       =G(s_H,H)-G(s_0,0)<=NL-G(s_0,0).

Exact logical flips imply s_{t+1}!=s_t, hence sum_j c_j>=1, giving

    H+sum_t J(s_t,i_t,t)<=NL-G(s_0,0).

This is a bookkeeping lemma, not a new asymptotic CANNOT. The proposed
logarithmic lower bound on sum J from the question report is UNPROVED.
Commuting logical flips need not commute physically; compatible histories may
merge with zero extra J. No word-count/fibre-size argument closes that gap.
