# Independent Proof Review 5 - OCCC Affine Modulus/Latency Counterexample

**Review status:** COMPLETE (2026-07-14). The file was written incrementally during
the attack; section 4 contains the controlling verdicts.

**Reviewer posture:** fresh, zero-context, adversarial mathematical review limited
to every `PROVED-PENDING-REVIEW` assertion in `research/OCCC_GATE0.md`. No private
project memory or earlier project conclusion is evidence. The occupation, novelty,
and strategy judgments are outside scope. No GPU is used, and no repository file
other than this review is edited.

**Rigor convention:** each mathematical finding is marked `PROVED / SOUND`,
`PROVED / DEFECT`, or `PROVED / COUNTEREXAMPLE`; a choice needed to make ambiguous
source language precise is marked `HEURISTIC`. Final claim verdicts use only
`UPHELD`, `UPHELD-WITH-REPAIR`, or `REFUTED`.

**Citation convention:** `OCCC:Lx-Ly` refers to the current line numbers of
`research/OCCC_GATE0.md`.

## 1. Exact claim statements and definitions under review

### 1.1 Frozen process family, observer, and access contract

For each integer `$m\ge 4$`, define

\[
 A=\{a\},\qquad B=\{0,1\},\qquad S=\{s\},
 \qquad \theta_m=\frac12+2^{-m},
\]

and the rational one-state kernel `$K_m$` by

\[
 K_m(s,a)(1,s)=\theta_m,
 \qquad K_m(s,a)(0,s)=1-\theta_m.
\]

Let `$F_m$` be its causal stream semantics. There is exactly one input stream,
`$a^\omega$`. Let `$q_m$` be the finite nonadaptive observer whose only requested
quantity is the probability of the first-output cylinder `$[1]$`, and put
`$p_m=m+2$`, so a correct rational answer `$r$` must satisfy

\[
 |r-\theta_m|\le 2^{-p_m}=2^{-(m+2)}.
\]

Both registered process presentations are finite: `$P_{\rm ker}$` is the displayed
one-row rational kernel table and `$P_{\rm mat}$` is the equivalent pair of rational
transition/output matrices. No feedback operation is used.

**HEURISTIC - strongest coherent frozen access reading.** The input stream is
available only through finite prefix queries. A server receives a finite process
description in either charged presentation, the finite observer/query and accuracy,
and whatever finite input prefix it requests. It is a single finite uniform
ordinary bit algorithm valid over the admitted finite class; process-specific
infinite advice and precomputed extensional tables are forbidden. Description
decoding, input reads, rational arithmetic, retained work, and output writing are
charged (`OCCC:L98-L103`, `OCCC:L140-L166`).

The rational answer is read under one fixed self-contained finite-alphabet encoding:
there is a finite alphabet `$\Sigma$`, a set `$D\subseteq\Sigma^*$` of valid answer
strings, and a fixed surjective decoding map
`$\operatorname{dec}:D\to\mathbb Q$`. Thus a
string has one rational meaning without consulting the process description,
observer, or an external uncharged decoder. The proof's phrase "fixed
finite-alphabet, self-contained encoding" (`OCCC:L228-L242`) is the minimal precise
ordinary-bit interpretation of the frozen requirements that rational answers be
written and output length be charged (`OCCC:L149-L166`).

**PROVED / DEFECT - two minor formal conventions are unstated.** First,
`OCCC:L155-L159` does not specify whether prefix lengths range over
`$\mathbb N_0$` or only positive integers. The submitted value zero requires
`$0$` to be admissible. Second, the frozen definition does not explicitly name
`$\Sigma,D,\operatorname{dec}$`, even though the lower-bound proof later invokes a
fixed finite-alphabet self-contained encoding. Both conventions are standard and
are forced by the claimed zero boundary and ordinary bit-output accounting, but
they must be made explicit in any promoted theorem statement.

**PROVED / DEFECT - accuracy is placed redundantly in the observer and modulus
syntax.** `OCCC:L149-L152` says an observer node names an accuracy `$2^{-p}$`, while
`OCCC:L155-L159` also supplies `$p$` as a separate argument to `$\mu_F(q,p)$`.
This review takes `$q_m$` to be the structural one-cylinder query and `$p_m=m+2$`
to be its accuracy parameter. If the accuracy is instead stored inside `$q_m$`,
the separate argument is set to the same value. Either precise typing yields the
same true probability, interval `$I_m$`, modulus, and latency contradiction; the
API should nevertheless choose one convention.

**HEURISTIC - exact latency formalization reviewed.** A uniform server is one finite
algorithm correct for every admitted process description, presentation, observer,
and finite accuracy request. For the fixed instance `$(F_m,q_m,p_m)$`, its
worst-case runtime includes both accepted presentations and every compatible input
stream; here the latter set is a singleton. `$L_{F_m}(q_m,p_m)$` is the infimum of
those runtimes over all such uniform servers. Whether "accepting either charged
presentation" means a maximum over both presentations or a separately chosen one
does not affect the submitted lower bound, because that bound uses only the answer
string that every presentation must ultimately produce.

### 1.2 Claim 1 - exact zero-modulus statement reviewed

**CONJECTURE - exact pre-review restatement.** With prefix lengths in
`$\mathbb N_0$`,

\[
 \mu_{F_m}(q_m,p_m)=0
 \qquad\text{for every integer }m\ge4.
\]

Equivalently, for every two input streams agreeing on the empty prefix, the induced
`$q_m$`-transcript distributions have total-variation distance at most
`$2^{-p_m}$`, and no smaller nonnegative prefix length exists.

### 1.3 Claim 2 - exact unbounded-latency statement reviewed

**CONJECTURE - exact pre-review restatement.** Fix any finite alphabet `$\Sigma$`
and any surjective self-contained rational decoder
`$\operatorname{dec}:D\subseteq\Sigma^*\to\mathbb Q$` used by all instances. Then

\[
 \sup_{m\ge4} L_{F_m}(q_m,p_m)=\infty.
\]

More explicitly, for every finite time bound `$T$`, some `$m\ge4$` forces every
correct uniform server to take more than `$T$` charged ordinary bit steps on the
registered instance. This is only an unboundedness assertion. The earlier stronger
`$\Omega(m)$` rate was expressly retracted (`OCCC:L221-L226`) and is not under
review as a pending proved claim.

### 1.4 Claim 3 - exact affine-characterization falsification reviewed

**HEURISTIC - strongest precise reading of the frozen target.** The phrase "the
least semantic modulus and least operational latency determine one another within
`$a x+b$`" (`OCCC:L168-L174`), together with the frozen initial statement that the
two are equal up to one affine distortion (`OCCC:L17-L26`) and the target-integrity
rule (`OCCC:L176-L179`), is read as including at least the universal upper direction

\[
 L_F(q,p)\le a\,\mu_F(q,p)+b
 \tag{A}
\]

for universal constants `$a\ge1,b\ge0$` independent of the process, presentation,
observer, and accuracy. A two-sided equivalence would additionally contain the
reverse inequality, but only `(A)` is needed for refutation. Reading the target as
only a lower bound on latency would contradict the file's actual frozen language
and is expressly declared a different target at `OCCC:L176-L179`.

**CONJECTURE - exact pre-review restatement.** No universal finite constants
`$a\ge1,b\ge0$` satisfy `(A)` over the frozen class. Hence the registered affine
equivalence, and therefore the conjunctive exact-characterization target containing
it, is false. The witness family must remain the exact one-state rational
finite-presentation family above and must use the frozen charged-description,
charged-arithmetic, charged-output access mode.

## 2. Attack log

### Increment 1: Claim 1, admissibility and the zero-modulus boundary

1. **PROVED / SOUND - every member of the witness family is inside the frozen
   process class.** For `$m\ge4$`, `$1/2<\theta_m\le9/16<1$`, so the two
   probabilities are valid dyadic rationals with finite encodings. The alphabets
   and state set are finite and nonempty, the initial state is named, the kernel is
   causal, and both `$P_{\rm mat}$` and `$P_{\rm ker}$` are ordinary finite
   presentations. A one-node nonadaptive cylinder query is a special case of the
   finite adaptive observer basis. The registration nowhere requires at least two
   input symbols, at least two states, feedback use, or genuine adaptivity
   (`OCCC:L98-L103`, `OCCC:L140-L153`).

2. **PROVED / SOUND - every input pair has identical induced semantics.** Since
   `$A=\{a\}$`, the set `$A^\omega$` contains exactly `$a^\omega$`. Therefore every
   quantified pair `$x,y\in A^\omega$` is the same pair. Its two induced output laws,
   and hence its two `$q_m$`-transcript distributions, are identical. Their total-
   variation distance is exactly zero, not merely at most `$2^{-p_m}$`.

3. **PROVED / SOUND - the submitted zero value follows if the empty prefix is an
   admitted length.** Finding 2 proves that `$n=0$` satisfies the condition defining
   `$\mu_{F_m}(q_m,p_m)$`. No element of `$\mathbb N_0$` is smaller, so the least
   value is zero. This is independent of `$m$`, accuracy, and observer content.

4. **PROVED / DEFECT - the literal value depends on the unstated prefix-length
   convention, but the affine counterexample does not.** If minimization is only
   over positive lengths, the least value is `$1$`, not `$0$`. Claim 1 as written
   is then false, but the modulus remains constant: an affine upper bound gives
   `$L_{F_m}(q_m,p_m)\le a+b$`, still contradicted by Claim 2. Declaring
   `$n\in\mathbb N_0$` is therefore a minimal repair to Claim 1, not a repair of the
   counterexample's mechanism.

5. **PROVED / SOUND - the two presentations introduce no hidden variation.** The
   modulus is semantic and ignores whether `$K_m$` is supplied as matrices or a
   kernel table. The unique input stream and its law are unchanged. This is the
   representation-invariant side of the frozen definition, not an access-mode
   switch.

### Increment 2: Claim 2, disjoint answers and uniform latency

6. **PROVED / SOUND - the admissible-answer intervals are pairwise disjoint, with
   a strict exact gap.** Put

   \[
     I_m=[\theta_m-2^{-(m+2)},\theta_m+2^{-(m+2)}].
   \]

   If `$n>m$`, then

   \[
     \theta_m-\theta_n=2^{-m}-2^{-n}\ge2^{-(m+1)},
   \]

   whereas the radii sum to at most
   `$2^{-(m+2)}+2^{-(m+3)}<2^{-(m+1)}$`. Hence even the closed intervals do not
   meet. For the closest case `$n=m+1$`, direct common-denominator calculation
   gives the exact separation

   \[
     \min I_m-\max I_{m+1}=2^{-(m+3)}>0.
   \]

   This independently recomputes and slightly sharpens `OCCC:L232-L235`.

7. **EMPIRICAL - CPU arithmetic cross-check.** Direct finite evaluation for
   `$4\le m\le10$` returned adjacent gaps
   `$2^{-7},2^{-8},\ldots,2^{-13}$`, exactly matching Finding 6. This check is not
   needed for the proof, which is the exact algebra above.

8. **PROVED / SOUND - self-contained valid answer languages for different `$m$`
   are disjoint.** Define

   \[
     C_m=\{w\in D:\operatorname{dec}(w)\in I_m\}.
   \]

   If `$w\in C_m\cap C_n$`, its single decoded rational lies in both `$I_m$` and
   `$I_n$`, contradicting Finding 6. Thus `$C_m\cap C_n=\varnothing$` for
   `$m\ne n$`. The decoder need not be injective: different strings may denote the
   same rational, but one string cannot have two process-dependent meanings.

9. **PROVED / SOUND - shortest valid answer lengths are unbounded for every fixed
   finite output alphabet.** Let
   `$\ell_m=\min\{|w|:w\in C_m\}$`, with `$\ell_m=\infty$` if `$C_m$` is empty.
   If all `$\ell_m\le N$`, shortest choices `$w_m$` would give infinitely many
   distinct strings in the finite set `$\Sigma^{\le N}$`, contradicting Finding 8.
   Therefore `$\sup_m\ell_m=\infty$`. If some `$C_m$` is empty, no correct server
   can answer that request and unboundedness holds a fortiori.

10. **PROVED / SOUND - the output-length bound survives the infimum over uniform
    servers.** Every correct server on instance `$m$` emits some `$w\in C_m$`, so
    it writes at least `$\ell_m$` symbols. An ordinary bit step writes at most a
    fixed constant number of symbols. Thus every correct uniform server, and hence
    their runtime infimum, is bounded below by a positive constant multiple of
    `$\ell_m$`. Finding 9 proves
    `$\sup_m L_{F_m}(q_m,p_m)=\infty$`.

11. **PROVED / SOUND - each individual latency is finite under an ordinary rational
    encoding.** A uniform finite algorithm can decode either rational presentation,
    evaluate any finite observer tree by finite rational matrix/kernel operations,
    and round to the requested finite accuracy. For this observer it need only read
    the one-step probability `$\theta_m$` and emit a rational in `$I_m$`. Thus the
    witness is a sequence of finite computations with unbounded cost, not a
    noncomputable process or infinite-time answer.

12. **PROVED / SOUND - compressed and noncanonical rational syntax cannot make the
    family bounded.** An encoding may give short symbolic names such as notation
    for `$1/2+2^{-m}$`, may be non-prefix-free, and may reserve the empty string for
    one rational. A finite alphabet still has only finitely many bounded-length
    strings, while the disjoint intervals require a distinct decoded string for
    every `$m$`. Such choices can defeat the retracted `$\Omega(m)$` rate, but not
    unboundedness.

13. **PROVED / SOUND - a constant approximate answer does not evade disjointness.**
    Returning `$1/2$`, for example, fails every request because
    `$|\theta_m-1/2|=2^{-m}=4\cdot2^{-(m+2)}$`. More generally, Findings 6 and 8
    show that no self-contained rational answer string is valid for two different
    family members.

14. **PROVED / DEFECT - a process-relative decoder would invalidate the proof but
    violate the stated access contract.** If one constant symbol meant *read the
    requested rational from the process description*, its interpretation would
    depend on `$m$`; the sets `$C_m$` would no longer use one decoder. That is not a
    self-contained rational answer and does not complete charged answer production.
    It is exactly the uncharged process/observer-relative decoding maneuver rejected
    at `OCCC:L239-L242`. The lower bound therefore depends materially on making the
    fixed decoder convention from section 1.1 explicit.

### Increment 3: Claim 3, the actual frozen affine statement

15. **PROVED / COUNTEREXAMPLE - the family contradicts the universal affine upper
    direction.** Suppose finite universal `$a\ge1,b\ge0$` satisfied `(A)`. Claim 1
    gives `$\mu_{F_m}(q_m,p_m)=0$`, so for every `$m\ge4$`,

    \[
      L_{F_m}(q_m,p_m)\le a\cdot0+b=b.
    \]

    Claim 2 says these latencies are unbounded, so some `$m$` has latency greater
    than `$b$`, a contradiction. Therefore no such universal constants exist.

16. **PROVED / SOUND - the contradiction attacks the registration's frozen
    statement, not a weakened paraphrase.** The initial cell asks for equality of
    least modulus and least server latency up to one affine distortion
    (`OCCC:L17-L26`); the exact target makes the constants independent of process,
    presentation, observer, and accuracy (`OCCC:L168-L174`); and the integrity rule
    expressly calls a one-way lower bound a different target (`OCCC:L176-L179`).
    Thus the upper direction `(A)` used in Finding 15 is a necessary part of the
    registered equivalence. Refuting it falsifies the conjunction regardless of
    whether a reverse inequality, feedback law, or lower-bound family survives.

17. **PROVED / SOUND - varying `$m$` respects the universal quantifiers.** The
    witness varies the finite rational process, finite query accuracy, and finite
    answer, all of which are universally quantified while `$a,b$` are required to
    remain fixed. It does not vary the cost model or answer decoder. Allowing
    process- or accuracy-dependent affine constants would evade the contradiction
    but is explicitly outside the frozen target.

18. **PROVED / SOUND - the refutation is robust to the positive-prefix convention.**
    Under that alternative, Finding 4 gives modulus `$1$` for every family member,
    and `(A)` gives the uniform finite bound `$a+b$`. Claim 2 again contradicts it.
    Thus the exact zero in Claim 1 needs a convention repair, but the registered
    affine characterization is false under either standard convention.

19. **PROVED / SOUND - no feedback or adaptivity theorem can repair a false base
    universal.** A nonadaptive one-node observer is an admitted finite adaptive
    observer, and a no-feedback process is an admitted base instance. A statement
    universal over the frozen class must hold there before any delay-guarded
    feedback composition clause is applied. Additional structure on other instances
    cannot remove this counterexample from the existing universal quantifier.

## 3. Access-contract and boundary audit

### 3.1 Frozen-access audit

1. **PROVED / SOUND - the witness uses the registered input access mode.** The
   process input is still an infinite stream exposed only by finite prefix queries.
   Because the alphabet is singleton and the kernel is input-independent, the
   server can make zero input-symbol queries. Using fewer allowed queries is not
   replacing the stream by random access, an extensional infinite object, or an
   oracle (`OCCC:L128-L136`, `OCCC:L145-L147`).

2. **PROVED / SOUND - the varying probability is supplied through the charged
   process description, exactly as frozen.** Each `$\theta_m$` is finite rational
   transition/output data in both presentations. The observer asks for a first-bit
   probability; no exact probability oracle or precomputed answer table is added.
   A server may read and calculate from the finite table, and those operations are
   charged. The proof lower-bounds answer writing even if description lookup and
   arithmetic were otherwise free, so it does not hide work in them.

3. **PROVED / SOUND - the output argument uses, rather than evades, full latency.**
   The frozen definition charges writing the requested rational answer and its
   output length (`OCCC:L161-L166`). Findings 8-10 apply precisely to that charged
   action. Replacing the rational answer by a process-relative token, or stopping
   the clock before its self-contained representation is written, changes the
   access/cost contract and does not counter the proof.

4. **PROVED / SOUND - neither finite presentation is privileged.** The same
   `$\theta_m$` is present in `$P_{\rm mat}$` and `$P_{\rm ker}$`, their translations
   are finite, and every correct answer under either presentation lies in the same
   `$I_m$`. The counting lower bound is therefore independent of presentation size,
   layout, and translation time. No matrix-only or kernel-only shortcut is used.

5. **PROVED / DEFECT - the server correctness mode is implicit.** The frozen text
   says the server returns rational approximations but does not separately declare
   deterministic versus randomized server semantics or a failure probability. The
   certified reading requires every returned answer to meet the requested error.
   If bounded-error randomized servers with worst-case runtime were intended, the
   argument still survives for any fixed positive success probability: under a
   common runtime bound only finitely many strings can be emitted, while the success
   sets `$C_m$` are pairwise disjoint. An expectation-only notion in which individual
   returned rationals may be inaccurate would be a different answer contract and
   needs a new definition.

### 3.2 Deliberate boundary attacks

6. **PROVED / SOUND - singleton input and one-state dynamics are allowed degenerate
   cases, not illicit model changes.** The cell assumes only nonempty finite
   alphabets and a finite state set (`OCCC:L98-L103`, `OCCC:L140-L143`). The witness
   has one input symbol and one state and therefore lies at an admitted boundary.
   The initial state exists, and `$B=\{0,1\}$` remains nontrivial.

7. **PROVED / SOUND - the singleton degeneracy is not load-bearing.** If one added
   an unregistered requirement `$|A|\ge2$`, the same kernel can be defined for every
   symbol of `$A=\{0,1\}$`, independent of the symbol. Then all input streams induce
   the same observer distribution, so the modulus is again zero under the empty-
   prefix convention, while the answer intervals and latency proof are unchanged.
   This is a robustness check, not a change to the submitted certified witness.

8. **PROVED / SOUND - zero modulus does not assume zero latency.** The semantic
   definition asks which input prefix controls the observer law; it does not account
   for query decoding, process-description access, arithmetic, or answer writing.
   Those are expressly included in `$L$`. The witness makes this type distinction
   exact: semantic dependence is zero while charged latency is finite for each `$m$`
   and unbounded across `$m$`.

9. **PROVED / SOUND - a possible zero-length code for one rational cannot defeat
   unboundedness.** A fixed decoder may assign the empty string to at most one
   rational value. At most one disjoint `$C_m$` can contain that string, and only
   finitely many `$C_m$` can contain strings below any fixed length. The proof does
   not require every individual latency to be positive.

10. **PROVED / SOUND - all accuracy and probability boundaries are finite.** For
    `$m\ge4$`, `$p_m=m+2\ge6$`; the requested tolerance is positive and rational,
    and `$\theta_m\in(0,1)$`. The proof never uses zero tolerance, an exact-real
    request, `$m=\infty$`, or a limiting process as a completed input. Taking an
    unbounded sequence of finite instances is the ordinary quantifier needed to
    refute universal finite constants, not an infinity leak.

11. **PROVED / SOUND - feedback, long horizons, and adaptive branching are absent
    without leaving scope.** The queried cylinder has horizon one and the observer
    has one node. The universal target includes these base instances. Neither a
    zero-delay feedback edge nor an unbounded observer tree is used, so feedback
    unfolding and guardedness cannot be hidden sources of the latency lower bound.

12. **PROVED / SOUND - tying `$p_m=m+2$` to the process index is legitimate.** The
    target quantifies uniformly over both process and finite accuracy. A witness may
    choose a different legal accuracy for each legal process. Even a server that
    infers `$m$` from `$p_m$` rather than reading the probability table must still
    emit a self-contained rational in `$I_m$`, so the output-language lower bound
    remains unchanged.

## 4. Final verdicts and certified scopes

### 4.1 Claim 1 - zero semantic modulus

**Final verdict: `UPHELD-WITH-REPAIR`.**

**PROVED / EXACT CERTIFIED CORE.** If input-prefix lengths are the nonnegative
integers, then for every `$m\ge4$`,

\[
  \mu_{F_m}(q_m,m+2)=0.
\]

The family has one input stream, so every pair of streams induces identical
observer-transcript distributions and total-variation distance zero. The conclusion
uses the exact submitted one-state, singleton-input, rational kernel and one-cylinder
observer.

**PROVED / MINIMAL REPAIR.** State explicitly that prefix lengths range over
`$\mathbb N_0$` and that `$n=0$` means agreement on the empty prefix. Also remove
the observer/accuracy syntax duplication by specifying whether `$p$` is external to
`$q$` or stored in it. No process, observer, access mode, or proof step changes.

**PROVED / BOUNDARY SCOPE.** If the repository instead chooses positive prefix
lengths, the exact value is `$1$` and the literal zero claim must not be promoted.
The affine falsification remains valid with this constant-one value.

### 4.2 Claim 2 - unbounded full latency

**Final verdict: `UPHELD`.**

**PROVED / EXACT CERTIFIED CORE.** For every globally fixed finite alphabet
`$\Sigma$` and surjective self-contained rational decoder
`$\operatorname{dec}:D\to\mathbb Q$`, the valid answer sets `$C_m$` are pairwise
disjoint. A finite alphabet has only finitely many bounded-length strings, so their
shortest lengths are unbounded. Charged ordinary output writing lower-bounds every
uniform server before the infimum over servers is taken. Therefore

\[
  \sup_{m\ge4}L_{F_m}(q_m,m+2)=\infty.
\]

This holds for either charged process presentation and deterministic total
correctness. The same count covers bounded-error randomized servers with any fixed
positive success probability and worst-case runtime.

**PROVED / EXACT ENCODING SCOPE.** The pending claim's condition *under any fixed
finite-alphabet, self-contained encoding* is essential and already states the needed
restriction. The decoder and per-step output convention must be fixed before
defining `$L$` and held fixed across all `$m$` and all servers. A process-relative
decoder, a unit-cost unbounded output symbol, or a token interpreted using the
query/process is outside the submitted claim and ordinary charged production.

**PROVED / RATE LIMIT.** The proof certifies only unboundedness. It does not recover
the retracted `$\Omega(m)$` bound: a legitimate compressed rational syntax may make
the shortest answer length grow much more slowly than `$m$`.

### 4.3 Claim 3 - falsification of the universal affine equivalence

**Final verdict: `UPHELD`.**

**PROVED / EXACT CERTIFIED CORE.** Under the frozen ordinary-bit access contract,
with any one fixed finite-alphabet self-contained rational encoding, no finite
process-independent constants `$a\ge1,b\ge0$` satisfy

\[
  L_F(q,p)\le a\mu_F(q,p)+b
\]

for every admitted process, observer, and accuracy. On the submitted family the
right side is `$b$` under the nonnegative-prefix convention, while the left side is
unbounded. Under the positive-prefix convention the right side is `$a+b$`, so the
same contradiction holds. Consequently the registered two-way affine equivalence,
which necessarily includes this upper direction, is false.

**PROVED / TARGET-INTEGRITY SCOPE.** This is the actual frozen full-latency target:
the constants are process/presentation/observer/accuracy independent, output writing
is charged, and weakening to a one-way lower bound is declared a different target.
The counterexample does not redefine latency as lookahead, change presentation,
add advice, or replace a finite stream query by another access mode.

**PROVED / COUNTEREXAMPLE SCOPE.** The witness uses exactly one finite state,
singleton finite input alphabet, binary finite output alphabet, dyadic rational
transition data, a named initial state, either registered finite presentation, one
nonadaptive horizon-one observer, finite accuracy `$p_m=m+2$`, no feedback, and no
infinite-precision object. Every instance is finite and has finite latency; only the
family supremum is infinite.

**PROVED / SCOPE LIMIT.** The counterexample proves the affine upper equivalence
false and therefore falsifies the conjunctive registered characterization. It does
not independently refute a purely semantic fixed-point law for `$\mu$`, a one-way
lower bound from modulus to latency, or any differently typed multi-resource
statement. Those are not pending proved claims in this review.

### 4.4 Pending-claim inventory

**EMPIRICAL - COMPLETE FILE AUDIT.** The only substantive
`PROVED-PENDING-REVIEW` assertions in `OCCC_GATE0.md` are the semantic-modulus value
at `OCCC:L217-L219`, the uniform-latency lower bound and its proof at
`OCCC:L228-L242`, and the universal affine falsification at `OCCC:L246-L255`.
Sections 4.1-4.3 dispose of all three. The other occurrences at `OCCC:L6-L8`,
`OCCC:L30-L35`, and `OCCC:L302-L306` state aggregate status or the review policy,
not additional mathematical derivations.

## 5. One-paragraph summary

**PROVED / SUMMARY.** The counterexample is mathematically sound at the frozen
ordinary-bit access type. With nonnegative prefix lengths, its one-state singleton-
input process has semantic modulus exactly zero; that claim is
`UPHELD-WITH-REPAIR` only because the source must explicitly admit the empty prefix
and settle the duplicated `$q/p$` accuracy syntax. For any fixed finite-alphabet
self-contained rational encoding, the requested answer intervals are pairwise
disjoint (adjacent gap exactly `$2^{-(m+3)}$`), so bounded output length could serve
only finitely many instances; charged output writing therefore makes the least
uniform latency unbounded, and Claim 2 is `UPHELD` without restoring the retracted
`$\Omega(m)$` rate. The family stays within both finite process presentations,
finite-prefix stream access, finite rational data, finite accuracy, and charged
answer production; it uses neither feedback nor an oracle. Hence no fixed affine
upper bound `$L\le a\mu+b$` can hold—nor can it under the alternative convention
`$\mu=1$`—so the falsification of the registration's actual two-way affine
modulus/latency equivalence is `UPHELD`. The exact certified lesson is the stated
resource type error: input dependence can determine which prefix matters without
paying the independently unbounded finite cost of producing the answer.
