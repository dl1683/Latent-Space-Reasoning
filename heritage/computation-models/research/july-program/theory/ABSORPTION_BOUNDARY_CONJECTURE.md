# Absorption-Boundary Conjecture — within-frame equivalent competition is symmetrically absorbable

**Unit type:** research CONJECTURE (not a proof, not a CANNOT theorem). Registered so the
first proof obligation is explicit.

**Date:** 2026-08-21

**Provenance:** distilled from the sibling project *Sovereign AI*, whose search for a
discontinuous cheap-local advantage repeatedly failed in the same way (NOTEBOOK
Entries 156, 183–195): CCPI (cheaper-but-equivalent inference), SWARM/LWPI (a different
learning structure that was near break-even), VCCG (a certified capability-graft absorbed
by provable neural repair). The connecting pattern (Sovereign Entry 194): every attempt
produced the *same* deliverable (an equivalent function/decision/capability) by a cheaper
or different route, and each was defeated by absorption/equivalence/break-even. This unit
tries to turn that empirical pattern into a theorem in the project whose mission is
defining models of computation and proving their limits.

**Scope guard:** FRESH unit, OUTSIDE the frozen Endgame Batch 1/2 surface. Not in the
canonical claims table. A conjecture with an open first obligation — nothing here is
claimed proved.

**Status:** `CONJECTURE / FIRST-OBLIGATION-OPEN`.

---

## 1. Informal statement

If a method's entire deliverable is a function that a fixed *editor/toolkit class* `E`
can also produce when applied to the incumbent's own artifact `f`, then a competitor
granted `E` reproduces the deliverable — the method has no exclusive advantage after
disclosure. Only a deliverable *outside* `{E(f) : E ∈ E}` (frame-transcendence: a new
object, a new capability the incumbent+`E` cannot cheaply reach, or a new class with its
own limits) escapes.

This is the computation-model reading of Sovereign's "equivalence-substitution trap"
and the symmetric-closure absorption note (see `analysis/CROSS_PROJECT_ABSORPTION_PARALLEL.md`
in Sovereign): within-frame competition is self-defeating; the escape is defining a new
frame and its limits — the Turing move this project is built on.

## 2. First proof obligation (Codex-set)

> Define an **editor class** `E` (a set of admissible, cost-bounded transformations an
> incumbent may apply to its own trained artifact `f` — e.g. provable-repair edits,
> fine-tunes under a compute budget `b`, prompt/adapter wrappers). Prove a
> **closure/simulation** result:
>
> **(Absorption).** For any method `M` whose deliverable is only some `g = E_M(f)` with
> `E_M ∈ E`, an incumbent granted access to `E` produces `g` at cost within a fixed factor
> of `M`'s. Hence `M` confers no post-disclosure exclusive advantage over the frozen
> incumbent-plus-`E`.

The theorem's force is entirely in how `E` and "cost within a fixed factor" are pinned.
Required before it is more than a conjecture:

1. A precise, non-trivial `E` (broad enough to be interesting, bounded enough that the
   simulation is not vacuous).
2. A precise cost model (the same one used to say a deliverable is "cheaper").
3. A precise notion of "equivalent deliverable" (exact function equality on a domain, or
   a task-metric tolerance).
4. The matching **escape half**: exhibit a deliverable NOT in `{E(f)}` and show it is not
   absorbable — otherwise the boundary is one-sided and uninformative.

## 3. Relation to the project's live machinery

The escape half is the same shape as this project's projection-kernel / same-observable-fibre
strategy (`STEERING_ENDGAME.md` §2) and its precommitment that an internal simulator kills a
claimed separation (line 38): if the incumbent+`E` can *simulate* the method's deliverable,
there is no separation; the surviving separations are quantitative (a charged budget the
incumbent+`E` provably cannot meet). So this conjecture is a candidate *organizing frame* for
the CAN/CANNOT pair, not a competitor to it.

## 4. Ledger

Row appended to `CLAIMS_LEDGER.md` append-only section:

`ABSORPTION-BOUNDARY | CONJECTURE: a method whose deliverable is only g=E(f) for some editor E in a class E, applied to the incumbent artifact f, is symmetrically absorbable by an incumbent granted E (no post-disclosure exclusive advantage). Escape requires a deliverable outside {E(f)}. | CONJECTURE / FIRST-OBLIGATION-OPEN (define E + cost model + equivalence + the escape half) | fresh unit OUTSIDE frozen Batch-1/2 | research/ABSORPTION_BOUNDARY_CONJECTURE.md | conjecture | Sovereign NOTEBOOK Entry 194-195; CROSS_PROJECT_ABSORPTION_PARALLEL.md`


## 5. Caveat (2026-08-21, adopted from sibling correction)
Post-disclosure absorption is DISSEMINATION, not a proof of research failure or a resource lower bound: a method that is later adopted by an incumbent still SUCCEEDED. This conjecture is about post-disclosure EXCLUSIVITY (no permanent private moat), NOT a claim that equivalent-capability improvements are impossible or worthless. The escape half + a charged cost model are what would give it teeth; without them it neither forbids progress nor proves impossibility.

## 6. Cross-project note (2026-08-21) — the escape half is the shared load-bearing question

A three-project synthesis (recorded as hypothesis, not theorem):
- The **absorption half** (§1–2) is near-tautological once E is pinned: if a method's deliverable is g = E_M(f) with E_M in E, an incumbent holding f and granted E just runs E_M. The force is entirely in the definitions; on its own it forbids nothing.
- The **escape half** — characterizing a deliverable provably OUTSIDE {E(f): E in E} for a genuinely broad E — is the hard, open, load-bearing part. And it appears to be the SAME object across the sibling projects:
  - *matrix-native-math* is "sealed" precisely because a structured-GEMM primitive (B389, 16–30x exact) is an E(f)-form deliverable — a software transformation any competitor can apply — hence absorbable; its stated reopen bar ("a new mechanism") is exactly "a deliverable outside the current edit-closure."
  - *Sovereign AI* closed because each direction it tried produced an E(f)-form deliverable (cheaper-equivalent inference/learning), so none was outside the closure.
  - *This project's* Turing-style mission — define a class and prove its limits — IS the escape half: exhibit or characterize a deliverable/capability no reasonable editor class can reproduce, and prove the boundary.
- **Implication for prioritization:** do NOT spend effort proving the (trivial) absorption half. The valuable obligation is the escape half, and it should be pinned in the same language as this project's live CANNOT strategy (projection-kernel / same-observable-fibre, STEERING_ENDGAME §2): a separation survives only when the incumbent-plus-E provably CANNOT simulate the deliverable within a charged budget. That reframes "un-absorbable" as "outside the closure of E under a charged cost model" — a quantitative separation, which is exactly the kind of CANNOT this project already favors.

## 7. Escape-half taxonomy (2026-08-21) — three structurally-distinct routes outside the closure

The escape half asks: what deliverable is outside {E(f) : E in E}? Three candidate routes, in increasing strength as a *computation-model* claim (hypothesis/organizing map, not a theorem):

1. **Information-asymmetric.** The deliverable requires information not derivable from f — external data, a private coupling, a live interaction. Formally it is E(f, x) with x not a function of f, so it is not in {E(f)}. This is where Sovereign's "asymmetric off-graph witness", private-coupling, and interaction directions lived. The moat is *access*, which Sovereign Entry 167 flagged as data-possession rather than a new intelligence class — so this route tends to be a **deployment/product asymmetry**, the weakest as a computation-model separation.

2. **Cost-asymmetric.** The deliverable is in the UNBOUNDED closure {E(f)} but outside the COST-BOUNDED closure {E(f) : cost(E) <= b} — the incumbent+E could in principle produce it but not within the charged budget. This is the discovery/search route (a verified artifact whose *verification* is cheap but whose *generation* is search-hard), and it is the same object as this project's priced-retention / charged-budget CANNOT (STEERING_ENDGAME §2). It carries a genuine, **quantitative** computation-model separation — the strongest tractable route.

3. **Frame-transcendent.** The deliverable is not a function-edit of f at all — it is a new class/model plus its limit theorems (the Turing move). Not of the form E(f) because E operates within a fixed frame. This is this project's stated mission endpoint.

**Cross-project reading:** matrix-native-math sealed because its primitive sits INSIDE even the *cost-bounded* closure (a cheap software transform any competitor applies); Sovereign closed because its directions were route-1/route-2 attempts that did not reach a *proven* cost-separation on the substrates tried; this project targets routes 2+3, with route 2 (a charged-budget separation) as the load-bearing, provable backbone and route 3 as the mission's endpoint. This sharpens the first proof obligation: prioritize a route-2 charged-budget separation over the near-tautological absorption half.
