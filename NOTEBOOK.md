# NOTEBOOK

Reverse-chronological running log. Newest first. Each entry: what was done, what
was learned, what's next. Canonical state lives in STATE.md.

---

## 2026-08-29 — Audit #12 adopted: unseen-word gate is mechanical-only until the bootstrap is contract-correct and the lexical nulls are stronger

- Status of both unseen-word runs: mechanical pass under the recorded
  reduction; formal gate pending a class-preserving, crossed word bootstrap
  (being implemented) and stronger X-free lexical nulls (frozen-embedding→Δ
  ridge; embedding-conditioned kernel; k ladder — being implemented).
- Wording: "not exact held-out-word lookup and not the tested lexical
  interpolator"; the positive object is X-conditioned residual
  predictability transferring across held-out words and blocks; F0
  "non-qualifying, continuation the strongest local failure pattern".
- Strongest rival (verbatim in EXPERIMENTS.md): X contains smooth lexical
  and presentation coordinates along which the later displacement varies;
  ridge/kernel recover that geometry; the coarse nulls collapse for
  coarseness, not because the variation is operational state.

## 2026-08-29 — Unseen-word run, sentinel ',': F4/F8/F12/F20 pass — both arms clear the criterion

- 2256 s. Same structure as the '.' arm: on disjoint held-out words the
  stronger X-free lexical null sits at the shared mean at every layer; ridge
  leads it by cos +0.11–0.17, skill +0.31–0.41, KL-rank +0.31–0.52 (block-
  first lower bounds > 0.09); 5–8/8 keys pass the full per-key gate, 8/8
  positive at F12/F20; no block collapse; F0 fails (cos lead 0.018).
- Both sentinels meet the Round 22 two-of-five criterion with four layers.
  The forward-step regularity of this decoder generalizes across carriers,
  across style families, and across word identities it never saw; every
  content null sits at the mean. Adopted wording remains: X-conditioned
  residual predictability, generalizing across unseen lexical identities;
  not yet separated from a smooth presentation coordinate; one decoder.
- Corrected equalized reruns (locoeq2A/B) now executing; Codex round 23
  adjudicates the unseen pair and predeclares residualization and the
  second model family.

## 2026-08-29 — Re-contextualization #12 (unseen words in; audit #12 fired)

- **Central bet + second lens:** native mathematics from what a denizen must
  invent; holes hostile to structured reasoning and what the next space must
  change.
- **Live question:** the forward-step regularity survives unseen words
  (sentinel '.'; ',' finishing). What remains between it and a "law": is it
  a property of the contextual state or of a smooth presentation coordinate
  (residualization, next), and is it a property of this decoder or of
  decoders (second family, after).
- **What reframes:** the sequence of nulls this program has run — identity,
  shared mean, word-mean, class mean, word-only embedding kNN, word-only
  ridge, shrunk word mean, alignment-destroying permutation, three-carrier
  block mean — all sit at the shared mean on the forward move except the
  identity (which is catastrophic there). Only X-conditioned predictors
  move. The honest statement is narrow (audit #11): X-conditioned residual
  predictability, generalizing across carriers, families, and now words.
  The old "native law" ambition has become a concrete object with three
  remaining tests, which is progress of the right kind.
- **Alternatives held live:** (a) embedding-neighbourhood interpolation —
  an unseen word is near seen words in embedding space; audit #12 asked;
  (b) a stronger X-free lexical model (embedding→displacement ridge) may
  close part of the gap; (c) the sentinel pair may still share style
  ('.' and ',' are both punctuation) — a non-punctuation sentinel would
  test it; (d) the response-space geometry ("same place = same law") as a
  native metric; (e) multi-step closure — a one-step law is not yet
  navigation; the denizen needs composition (F4→F8→F12 along the token
  clock), never tested.
- **Ecosystem:** "when every content null sits at the mean, the object is
  X-conditioned predictability; name it that, not a law" → `_meta`.

## 2026-08-29 — Unseen-word run, sentinel '.': F4/F8/F12/F20 pass the full gate on words never seen

*Qualified by audit #12 (see the 2026-08-29 audit #12 entry): the pass is mechanical under the recorded reduction, formal gate pending a contract-correct bootstrap; "not word lookup and not class lookup" reads "not exact held-out-word lookup and not the tested lexical interpolator".*

- 2239 s; eight block × word-fold keys; support 1.0; calibration and
  held-out word identities disjoint; lexical nulls = class mean and
  frozen-input-embedding kNN (both ≈ shared mean at every layer).
- Under the Round 22 gate (≥0.02 over the stronger X-free lexical null with
  positive lower bounds on cosine, law skill, K = 11 KL-rank; block-first
  pooled contrast positive; ≥6/8 keys; no block collapse): **F4, F8, F12, F20
  pass** — block-first pooled leads cos +0.14–0.19, skill +0.33–0.47, KL-rank
  +0.35–0.57 (lower bounds > 0.12); 7–8/8 keys positive; F0 fails (cos lead
  0.019; the continuation block collapses). Two of five met with four.
- Reading (bounded per audit #10/#11): the forward-step regularity survives
  lexical novelty — it is not word lookup and not class lookup. What X
  carries about the next step generalizes across words it never saw, with a
  ~0.06 drop from the seen-word runs. Still one decoder, one style-family
  set; state vs smooth style code unresolved (residualization next).
- Sentinel ',' arm running; corrected equalized reruns follow; Codex round
  23 adjudicates all.

## 2026-08-29 — Equalized LOCO addendum, sentinel ',' (defect-affected; descriptive only)

- 2977 s. Same pattern as the '.' arm under the audit #11 defect (inner
  centre included the validation carrier): equalized baselines equal the
  shared mean at every layer; F12/F20 pass the mechanical gate, F4/F8 miss
  on skill/KL-rank lower bounds, F0 fails. Outer margins are descriptive
  only; the corrected rerun (locoeq2A/B) is queued behind the unseen-word
  runs, which are now executing.

## 2026-08-29 — Audit #11 adopted: equalized addendum has an inner-centre bug; wording withdrawn

- The equalized baselines' inner selection centred on the outer three-carrier
  mean (contains the validation carrier) → maximal shrinkage forced by
  construction; comparator chosen on held-out outcomes. Fixed in the
  analyzer; both arms rerun behind the running chain. The '.' addendum's
  outer margins stand as descriptive numbers only.
- Withdrawn from my previous two entries: "no per-word lexical signal",
  "variance objection answered", "the forward step is about context, not
  content", "the state-conditioned component is large". Adopted wording:
  the word-conditioned component captured by the tested estimators is
  negligible in this design; the positive object is X-conditioned residual
  predictability. The `_meta` deposit is corrected to match.
- LOCO B precision: F8 misses skill only (KL-rank LB +0.021).
- Second lens (auditor): the narrower true statement — lexical content is not
  a sufficient predictor of the later forward step; context-bearing X
  contains predictable variation that word-conditioned means do not capture;
  the next latent space must define "same place" by interchangeability of
  moves and response laws, not lexical identity or representational
  similarity.

## 2026-08-29 — Re-contextualization #11 (equalized addendum in; audit #11 fired)

*Superseded in part by audit #11 (see the 2026-08-29 audit #11 entry): the equalized-addendum inner centre included the validation carrier, so "context, not content" and "content nulls collapse to the shared mean" are withdrawn; the addendum's margins are descriptive only.*

- **Central bet + second lens:** native mathematics from what a denizen must
  invent; holes that make this space hostile to structured reasoning.
- **Live question:** the forward step's within-family regularity is not
  lexical (equalized lexical baselines collapse to the shared mean) — so it
  is either the contextual state or a smooth style coordinate. The
  unseen-word runs (in the chain) remove lexical lookup entirely; the
  residualization control after them is the only thing that can separate
  state from style, and audit #10 already warned the separation may be
  ill-posed here.
- **What reframes earlier work:** every lexical null in this program has
  come out at the shared mean on the forward move (word-conditioned mean,
  class mean, word-only kNN, word-only ridge, shrunk word mean). The
  forward step of this world is about context, not content: what the next
  position does depends on the state the context has built, not on which
  word was inserted. Under the second lens this is a candidate structural
  fact — and possibly a hole: a denizen cannot navigate by content alone,
  because content barely moves the next step; only context does.
- **Alternatives held live:** (a) maximal shrinkage is forced by selecting on
  two-carrier means (audit #11 asked); (b) the only surviving competitor is
  the shared mean, so the LOCO gate may be trivially passable — a fair
  competitor might be a carrier-code-only or style-coordinate predictor;
  (c) a second family may show a different content/context balance — the
  first cross-model native quantity would be "how much of the next step is
  content"; (d) the response-space geometry idea (same place = same law)
  would make this balance a metric property.
- **Ecosystem deposit:** "on the forward move, content nulls collapse to the
  shared mean; the next step is governed by context state" → `_meta`.

## 2026-08-29 — Equalized LOCO addendum, sentinel '.': the lexical baselines collapse to the shared mean; ridge's lead is unchanged

*Superseded in part by audit #11 (see the 2026-08-29 audit #11 entry): inner-centre defect; "no per-word signal" and "variance objection answered" are withdrawn; outer margins descriptive only; corrected rerun queued.*

- 2911 s of the 4500 s wall. Both equalized X-free baselines (word-only
  one-hot ridge with inner-selected λ; shrunk word mean with inner-selected
  α) select maximal shrinkage at every layer and equal the shared mean to
  three decimals: within a style family, three carriers carry no per-word
  signal about the forward displacement beyond the family's shared shift.
- Gated against the stronger equalized baseline: **F4, F8, F12, F20 pass**
  (pooled ridge − baseline: cos +0.09–0.13, skill +0.23–0.30, KL-rank
  +0.26–0.34, all lower bounds > 0.08; 11–14/16 carriers pass all three);
  F0 fails. Run-level positive.
- Reading: audit #10's variance objection is answered — the block-word mean
  was not losing to noise; there was nothing lexical to estimate. What X
  predicts within a family is not word identity; it is something carried by
  the contextual state (state or smooth style code — still unresolved;
  residualization remains the next control after unseen words).
- Sentinel ',' addendum and the two unseen-word runs follow in the chain.

## 2026-08-28 — LOCO control, sentinel ',': F12/F20 pass; weaker than the '.' arm

- 3091 s of the 4500 s wall; support 1.0. **F12 and F20 pass** the Round 21
  rule (pooled ridge − per-word block mean: cos +0.07–0.10, skill +0.15–0.20,
  KL-rank +0.20–0.26, lower bounds > 0; 12–13/16 carriers pass all three).
  F4 and F8 keep cosine leads (+0.08–0.11, LB > 0.06) but miss on skill /
  KL-rank lower bounds; F0 fails. Run-level positive (2/5).
- Both arms positive; the '.' arm at four layers, the ',' arm at two. Under
  audit #10 the wording stays: on seen words, within a style family, X
  predicts a held-out carrier's forward step better than the three-carrier
  per-word family mean — a variance-disadvantaged baseline; equalized
  word-only baselines are owed before interpretation, and LOCO cannot
  separate state from a smooth style code.
- Codex round 22 adjudicates both arms and predeclares the unseen-word run
  (lexical nulls, K = 11 universe, block-first bootstrap all implemented).

## 2026-08-28 — Audit #10 adopted: LOCO bounded; unseen-word branch needs a lexical null; second-lens table

- LOCO A = "X predicts a held-out carrier's displacement and consequence
  better than the three-carrier per-word family mean at F4–F20" — a
  variance-disadvantaged baseline; equalized X-free lexical baselines
  (word-only ridge; shrunk word mean) required before interpretation;
  block-first bootstrap for any cross-family statement; LOCO cannot separate
  state from a smooth style code — residualization is the next control.
- Unseen-word branch: correct mechanics, no lexical null once the word-mean
  is dropped → class-mean displacement null and a word-only input-embedding
  predictor added as the primary X-free baselines; fixed rank universe;
  fail-fast asserts; block-first pooled bootstrap.
- Second lens (auditor's table, adopted): proven — identity-dominated input
  transition; ordering-saturated readout (for our endpoint). Unproven —
  presentation entangled with state (strong concern); family-only laws (not
  shown; whole-block transfer works); motion invisible to the response law
  (readout-specific). The serious hole: no stable quotient separating
  lexical content, presentation, operational state, and consequential
  motion — "we may have incorrectly declared differently presented states
  to be the same place." Next-generation requirements recorded verbatim.

## 2026-08-28 — Re-contextualization #10 (LOCO A in; second lens active)

- **Central bet:** native mathematics of latent spaces from what a denizen
  must invent; **second lens:** holes that make this space hostile to
  structured reasoning, and what the next latent space must change.
- **Live question:** is the forward step's regularity a property of the
  state or of its presentation — and, under the second lens, is
  "presentation entangled with state" a hole or simply what state *is* in
  a context-conditioned world?
- **What still holds:** forward displacement predictable beyond word and
  token identity from F4 in both arms; within-family LOCO positive at
  F4–F20 (sentinel '.'); exact completion routing; nonpass (not kill) under
  the historical ordering gate.
- **What reframes:** the LOCO result narrows the nuisance rival to "carrier
  identity encoded in X predicts carrier-specific displacement" — which is
  hard to distinguish from state-dependence by construction. The
  unseen-word split changes the axis: if the regularity survives words the
  field never saw, it is not a lexical lookup either. The real reframing
  is that the distinction state-vs-presentation may be ill-posed here: a
  denizen's "place" includes the context it is in.
- **Candidate holes (for audit #10 to test):** (1) motion invisible to the
  response law at middle depth — likely a readout property; (2) identity-
  dominated layer transitions — real, but a property of residual streams,
  not a hole per se; (3) presentation entangled with state — real, status
  unclear; (4) laws holding only within template families — not shown
  (whole-block transfer works); (5) ordering-saturated readouts — proven
  for our endpoint, not for the world.
- **Alternatives held live:** the LOCO baseline is a 3-carrier mean (noisy;
  an equalized baseline may close the gap); a response-space geometry
  where "same place" = same law; a second family may have a different
  persistence/consequence profile — the first cross-model native quantity.
- **Ecosystem deposit:** "state vs presentation may be ill-posed for
  context-conditioned representations; test it via unseen identities, not
  via nulls that destroy alignment" → `_meta/INDEX.md`.

## 2026-08-28 — LOCO control, sentinel '.': within-family state information at F4–F20

- 2902 s of the 4500 s wall; support 1.0. Per the Round 21 rule, **F4, F8,
  F12, F20 pass** (pooled ridge − per-word block mean: cosine +0.09–0.13,
  law skill +0.23–0.31, KL-rank +0.29–0.40, all lower bounds > 0.08; 11–15
  of 16 held-out carriers pass all three). F0 fails as predicted (block mean
  ≥ ridge). Run-level within-family diagnostic: positive.
- Reading: inside a style family, with one carrier held out, the state
  predicts that carrier's forward step better than the family's own per-word
  mean displacement — on the displacement, on the law, and on rank. Together
  with the whole-block hold-out (transfer to an unseen family), the
  "presentation-only" rival now has to explain both a cross-family transfer
  and a within-family carrier-specific gain. What remains of it: carrier-
  specific presentation encoded in X that predicts carrier-specific
  displacement — which is close to saying the state knows its context, i.e.
  is state. Codex round 22 rules, with the second lens: is "style entangled
  with state" a hole, or the definition of state in this world?
- Sentinel ',' arm running.

## 2026-08-28 — Second lens added (Devansh): holes, and the next latent space

- Standing instruction: structural properties that make current latent
  spaces hostile to structured reasoning are first-class findings; if
  proven, the constructive program is a next-generation latent space in which
  they are closed. Candidate holes already on the table from NLM-007: motion
  the world's response cannot register (middle-depth displacement invisible to
  the slot law); identity-dominated transitions; presentation entangled with
  state; laws that may hold only within a template family. Each is now a
  question for every Codex round and audit.

## 2026-08-28 — Within-style null, sentinel ',': F8/F12/F20 mechanically pass (diagnostic only)

- 2238 s, support 1.0, K = 7 KL-rank label. Same shape as the '.' arm: the
  within-style null collapses below the shared mean from F4 on (0.21–0.54 vs
  0.45–0.65) while ridge/kernel hold 0.68–0.80; F8/F12/F20 clear the
  mechanical gate, F4 misses, F0 fails.
- Per audit #9 this is an alignment-destruction diagnostic, not a style
  control; no "style-robust" claim. Both arms recorded; Codex round 21
  adjudicates and predeclares the leave-one-carrier-out control (`--loco`,
  implemented, smoke pending).

## 2026-08-28 — Audit #9 adopted: nonpass ≠ kill; KL-rank set defect; style null is a diagnostic only

- "Not met" is a nonpass under the historical contract, not a kill; the
  comma arm falsifies "token/position prevents any qualifying layer".
- KL-rank ranked K = 7 candidates instead of the preregistered 10 (kNN-1/5/20
  omitted): fixed in the analyzer; style-A/B runs labelled K = 7, not
  contract-valid on that endpoint.
- The within-style null is an alignment-destruction diagnostic; "style-robust"
  is withdrawn as a claim. Next fair control: within-family
  leave-one-carrier-out vs per-word/per-block mean displacement (to be
  predeclared by Codex), then residualization, then unseen words, then a
  second family.

## 2026-08-28 — Within-style null, sentinel '.': F4/F8/F20 style-robust mechanically; the null itself is suspect

- 2213 s, support 1.0. Under the Round 20 gate (≥0.02, LB > 0 over the
  word-conditioned mean AND over the within-style-family null, on cosine,
  skill, KL-rank): **F4, F8, F20 pass**; F12 misses one fold's KL-rank LB
  (−0.053); F0 fails (style null = shared mean = word-mean = field).
- KL-rank (new endpoint) separates cleanly where ordering never did: ridge
  0.82–0.90 vs word-mean 0.31–0.41 at F4/F8/F20, LBs > 0.16.
- The within-style null collapses below the shared mean (0.16–0.50 vs
  0.47–0.62): a field refit on a broken pairing predicts the wrong
  carrier's displacement. That makes "beats the null" easy — audit #9 is
  asked whether the null is a straw man and what a fair style control is.
  Note for that ruling: the outer fold already holds out a whole style
  family (the four config blocks), so transfer to a held-out block cannot
  use that block's style code; the residual confound is style shared
  across families.
- Sentinel ',' arm running; Codex round 21 adjudicates both with audit #9.

## 2026-08-28 — Re-contextualization #9 (style null running; audit #9 fired)

- **Central bet:** native mathematics of latent spaces from what a denizen
  must invent. **Live question:** is the world's forward step (last context
  state → next position) governed by a regularity that belongs to the state
  rather than to the presentation (carrier/template style)? The style null
  is the first test; unseen words and a second family follow.
- **What still holds:** forward displacement predictable beyond word and
  token identity from F4 in both sentinel arms; law skill at the sentinel
  registers it; the preregistered two-layer criterion not met for the
  primary arm (Round 20), ordering ruled saturated and replaced
  prospectively by KL-rank.
- **What reframes earlier work:** every gate failure so far has come from
  the ordering endpoint, in every program; the world may have been "saying
  yes" through cosine and skill all along while our consequence endpoint
  could not hear it. The lesson is about endpoints, again: a consequence
  measure must be able to fail for the null and pass for the truth — a
  calibration we never ran for ordering.
- **Alternatives held live:** (a) the within-style permutation null is a
  straw man — a refit field on a broken pairing must predict the wrong
  carrier's displacement and fall below even the mean, so "beats the null"
  is uninformative; a fair style control holds out style *families* or
  residualizes a block code from X (audit #9 asked); (b) style explains the
  lead — testable by a per-block-mean displacement baseline (cheapest);
  (c) the KL-rank endpoint may be biased by including the compared field
  in the ranked set; (d) the whole "law" is one decoder's habit — second
  family; (e) the denizen's map might be over responses, not states.
- **Ecosystem deposit:** "a permutation null that a flexible model trivially
  beats is not a control; calibrate every null by checking it can pass for
  the truth and fail for the confound" → `_meta/INDEX.md`.

## 2026-08-28 — Forward-time move, sentinel ',': F12 and F20 clear the gate; the two arms disagree only at the ordering margin

- 1823 s, support 1.0. Same shape as the '.' arm: F0 token-identity
  dominated; F4–F20 displacement cosine ridge/kernel 0.68–0.80 vs
  word-conditioned mean 0.46–0.66 (LBs > 0.1), law skill at the sentinel
  0.46–0.57 vs 0.01–0.02, shuffle collapses.
- Gate (mechanical): **F12 and F20 pass** for ridge (ordering +0.022–0.074,
  LBs 0.004–0.037 at F12); F8 misses on one fold's ordering LB (−0.002); F4
  on one fold's skill LB. The '.' arm passed F20 only. Two of five layers
  for the same sentinel is met by the control arm, not the primary.
- Reading: the forward step is predictable from the state beyond word and
  token identity at every layer from F4 in both arms; whether a layer
  "qualifies" is decided by ordering lower bounds within ±0.02 of zero.
  Ordering is the binding endpoint in every program so far (layer
  displacement, forward A, forward B). Codex round 20 must rule on whether
  the primary/control asymmetry is a failure of the primary arm or a
  property of the ordering endpoint, before anything is claimed.

## 2026-08-28 — Audit #8 adopted (displacement wording; forward implementation verified)

- Forward-time implementation verified line by line by a fresh auditor
  before its scores were interpreted; the one missing check (A/B unappended
  states identical) passes bit-exactly.
- Displacement wording narrowed and adopted verbatim: kernel captures
  held-out-carrier displacement variation beyond the word-conditioned mean;
  carrier/template vs state dependence unresolved; the carrier shuffle is a
  carrier-alignment diagnostic, not a state-independence null; "the slot law
  barely registers it" is a readout fact; L20 = one bounded qualifying pair.
- Its cheaper controls (style balancing / residualization, within-template
  null, style-held-out split, Y−X decomposition into word/carrier/shared/
  residual, per-layer float32 precision reports) are recorded verbatim in
  EXPERIMENTS.md and enter the queue ahead of any "state-dependent" claim.

## 2026-08-28 — Forward-time move, sentinel '.': state-dependent everywhere, gated only at F20

- Five layers, 2220 s, support 1.0, locality passes under the Round 20
  clause. **F0** token-identity dominated (shared mean = word-conditioned
  mean = 0.67 ≈ field 0.69), as predicted.
- **F4 / F8 / F12:** displacement cosine ridge/kernel 0.71–0.78 vs
  word-conditioned mean 0.48–0.53 (leads +0.17–0.27, clustered LBs > 0.15
  every fold); law skill at the sentinel position 0.39–0.57 vs 0.01–0.02;
  carrier-shuffled null 0.12–0.32 vs field 0.67–0.81; within-carrier oracle
  ~0.98. The world's forward step is strongly state-dependent beyond word
  identity and beyond token identity. But ordering leads are 0.00–0.08 with
  LBs ≤ 0 in half the folds → the three-endpoint gate fails.
- **F20 qualifies** (ridge: +0.16–0.23 / +0.50–0.61 / +0.020–0.058, all
  LBs > 0). One layer; two required for the same sentinel.
- Token-identity control: the '.'-fitted predictor applied to the ','
  target scores 0.43–0.54 vs 0.26–0.30 for the shared mean — the learned
  displacement carries a sentinel-independent, state-dependent component.
- Structural reading: identical to the layer-clock displacement ladder —
  cosine and skill say "large state-dependent motion, registered by the
  law"; ordering says almost nothing until late. Ordering (per-anchor
  concordance of KL orderings across words) is dominated by word identity,
  which every predictor preserves; it is the binding gate in both clocks
  and may be an insensitive endpoint rather than evidence of
  inconsequential motion. Codex round 20 must rule on the endpoint before
  the gate is read as a world fact. Sentinel ',' arm running.

## 2026-08-28 — Re-contextualization #8 (forward-time run in progress)

- **Central bet:** native mathematics of latent spaces from what a denizen
  must invent. **Live question:** what is the denizen's actual step — the
  forward-time move from the last context state to the next position — and
  does it obey a reusable, state-dependent law that the world's response
  registers? The layer clock was the analyst's; the token clock is the
  world's.
- **What still holds:** exact routing of the completion (~1e-5); lexical
  persistence at L0 on every endpoint; identity + calibration-mean
  displacement as a competitive description at L8/L12 (post-hoc rule,
  labelled); state-dependent, nonlinear displacement on its own coordinates
  from L4 on; one gated pair (L20) where the law feels the displacement.
- **What reframes earlier work:** the question "is there a law of motion"
  split into "is there motion" (yes, everywhere from L4) and "does the
  world's response register it" (only late, under the slot readout). The
  forward-time preview at F4/F8 (ridge 0.72–0.78 vs word-mean ~0.5, skill
  ~0.45) suggests the token clock's move is both larger and more
  consequential than the layer clock's — if it holds under the gates, the
  native object is the forward step, and the layer-pair program was
  measuring the wrong move.
- **Alternatives held live:** (a) carrier/template style, not state,
  explains displacement leads (style-balancing control pending); (b) the
  sentinel choice ('.' vs ',') may dominate — the comma arm decides;
  (c) 'consequential motion' may be a readout artifact (ordering is
  saturated) rather than a world property; (d) all of this is one small
  decoder — second family untested; (e) unseen words untested; (f) maybe
  the denizen's map should be over *responses* (laws) rather than states —
  a law-space geometry where 'same place' = same response, which would
  make inconsequential motion literally zero distance.
- **Ecosystem deposit:** "a move can be large in coordinates and invisible
  to the world's response — always measure both, and name which one a claim
  is about" recorded in `_meta/INDEX.md`.

## 2026-08-28 — Displacement ladder: Δ is state-dependent at every depth ≥ L4, but the slot law only feels it late

- Five raw-residual pairs, Δ = Y − X predicted from X, 1750 s of a 5700 s
  wall, support 1.0.
- **L0→L1:** Δ is lexical persistence — the word-conditioned displacement
  mean equals every field (0.948) and the carrier shuffle changes nothing.
- **L8→L9 / L12→L13:** on the displacement's own coordinates the field beats
  the word-conditioned displacement mean by +0.07–0.22 (lower bounds >0.05
  in every fold), the carrier-shuffled null collapses (0.35–0.52 vs
  0.60–0.85), and the minimal class is **kernel** — the displacement is
  state-dependent beyond word identity and not affine. But the slot law
  barely registers it: ordering leads 0.003–0.022, slot-skill lower bounds
  mixed. Gate fails as predicted, for a reason the prediction did not name:
  the identity component saturates the law at middle depth, so the
  denizen's law of motion is invisible to the world's own next-token
  response there.
- **L20→L21 qualifies** (kernel: +0.025–0.051 / +0.13–0.32 / +0.023–0.038,
  all lower bounds >0) — falsifying "small residuals, no complete result".
  Late in the stack the displacement changes the law.
- **L4→L5:** kernel leads on cosine (+0.02–0.03) with tiny ordering
  differences; gate fails as predicted.
- Reading under the guiding question: the world moves its states in a
  state-dependent, nonlinear way from L4 on, but at middle depth those
  moves are along directions the readout is nearly blind to; only late
  moves are "consequential". A denizen would need two notions: motion
  (which exists everywhere) and consequential motion (which is manufactured
  late). Codex round 19 adjudicates; next in the fixed order is the
  forward-time move under a stricter contract.

## 2026-08-28 — Audit #7 adopted; displacement run launched

- The ≤0.02 closure rule was post-hoc: the withdrawal at L8/L12 stands as a
  conservative one-sided policy, not a preregistered equivalence. Wording
  corrected throughout: "no demonstrated positive ridge advantage under this
  margin"; "consistent with identity plus a calibration-mean displacement;
  the structure of Δ is unresolved"; completion "validated to measured
  precision". L4/L20 live remainders; L27 not a persistence test.
- Audit #7 endorses Round 18's order: displacement ladder first, then
  forward-time transport under a stricter contract (sentinel, token/position
  baselines, endpoint definition), then unseen-word/style controls, then a
  second family.
- Δ-mode smoke at L8→L9 (2 shuffles/10 boot): displacement cosine — shared
  shift 0.40, word-conditioned displacement mean 0.58, chart 0.64, ridge
  0.71, kernel 0.76 — but slot ordering moves by only 0.003–0.02 over the
  word-conditioned mean: the slot law is nearly saturated by the identity.
  The predeclared five-pair run (95-minute wall) is executing.

## 2026-08-28 — Re-contextualization #7 (after the identity baseline)

- **Central bet:** native mathematics of latent spaces from what a denizen
  must invent. **Live question now:** in a world whose middle blocks mostly
  leave a state where it is and add a shared shift, what is the law of
  motion — and is the residual motion (Δ beyond the shared shift) the object
  a denizen would care about, or is the real move forward-time?
- **What still holds:** exact slot completion (identity KL ~1e-5); lexical
  persistence at L0; the ridge/chart/word-mean ordering as a descriptive
  fact; L0 and L27 as transforming blocks in their own coordinate families.
- **What is reframed:** the "affine transport law" was the residual stream's
  identity plus a constant. Every earlier NLM-007 reading of "law" at middle
  depth is re-read as "persistence". The frozen-encoder residue (chart
  smoothness, affine-path robustness) and this one now rhyme: in both worlds
  the cheapest map (identity / straight line) explains most of what the
  fancier map explained. The recurring lesson is about our ladders, not the
  worlds: the null must be the cheapest thing the world could be doing.
- **Alternatives held live:** (a) Δ-ladder — is any state-dependent motion
  present beyond the shared shift, per depth; (b) forward-time move — the
  denizen's actual step; (c) the displacement may be word-dependent rather
  than state-dependent (word-conditioned mean displacement decides); (d) the
  whole "layer = time" framing may be the wrong clock for a denizen; (e) a
  second family may show a different persistence profile — if persistence
  depth-profiles differ across families, "where the world moves" becomes the
  first cross-model native quantity.
- **Ecosystem deposit:** "identity is the null for any residual-stream
  measurement" recorded in `_meta/INDEX.md` as a portfolio-wide rule.

## 2026-08-28 — Six-pair moot-maker run: persistence plus a shared displacement is the middle-depth "law"

- `Yhat = X + mean_cal(Y−X)` vs ridge on the corrected slot endpoint, pooled
  ridge − identres (cos / slot skill / slot ordering): L0 +0.46/+0.96/+0.38;
  L4 +0.033/+0.019/+0.022; **L8 −0.008/−0.021/−0.020; L12 −0.007/−0.009/−0.013**;
  L20 +0.018/+0.034/+0.032; L27 +0.20/large/+0.17 (post-norm target — not a
  persistence family). The per-carrier affine diagnostic is far below both
  everywhere (0.63–0.87 / 0.42–0.55).
- Withdrawal condition met at L8→L9 and L12→L13 — the two pairs that carried
  the corrected two-pair criterion. The "full-dimensional affine predictor"
  residue there was the residual-stream identity plus a constant shift. At
  L4 and L20 a state-dependent remainder of 0.02–0.03 survives the identity
  baseline but sits under every gate. L0 and L27 are transforming blocks in
  different coordinate families.
- Round 17's prediction that identity-plus-residual would not close the lead
  at L12/L20/L27 failed at L12. Run overran the 55-minute budget (4541 s):
  budget-incomplete, no gate claim drawn; the withdrawal is null-making and
  stands.
- What the question becomes: the transport content of a middle block is the
  displacement Y − X; the ladder must be rerun on Δ (mean displacement as the
  zero-order law) to ask whether any state-dependent motion exists at all
  beyond persistence. And the move a denizen actually makes is forward-time,
  not layer-to-layer — untested. Codex round 18 adjudicates and re-orders.

## 2026-08-28 — Tier-3 audit #6 adopted

- Code repair confirmed; L8/L12 qualification stands as bounded exploratory
  evidence at the reduced budget; L27 kept in its own post-norm family.
- Corrections adopted verbatim: L4/L20 non-qualifying but live (not killed);
  the all-fold +0.05 rule is a stricter convention than the original lock and
  is labelled so; "wins" = minimal within 0.02 (kernel numerically best on
  some endpoints); no manufactured-context causal language from the shuffle
  profile without spread/style controls; identity test to be extended across
  probes and pairs and stored.
- Its ordered next actions match the Round 16/17 order; the baselines run is
  executing now.

## 2026-08-28 — Moot-maker #1 smoke: identity plus a shared displacement explains L8→L9

- `Yhat = X + mean_cal(Y−X)` at L8→L9: successor 0.949 pooled vs ridge 0.941;
  slot skill 0.958–0.975 vs ridge 0.905–0.980; ridge − identres ≤ +0.013 on
  every endpoint in every fold, negative in three of four. The per-carrier
  affine diagnostic (64 training words per carrier) sits at 0.80 / 0.48.
- Round 16 withdrawal condition met at this pair: the "affine law" wording is
  withdrawn. The move at middle depth is persistence plus a shared
  displacement — the residual stream behaving as a residual stream. This is
  why rank ≤ 128 lost by 0.05 (it cannot express the identity) and why the
  static chart lost (it never sees the held-out state's own coordinates).
- Lesson logged against think-before-you-run: identity was the obvious null
  for a residual stream and should have been in the ladder from Round 13.
- What the question becomes: the transport content is the displacement
  Y − X. Is it state-dependent beyond a constant? The ladder must be rerun on
  Δ with the mean displacement as the zero-order baseline, and the depth
  profile re-read: the word-mean's late collapse (0.40) now says the
  displacement, not the state, is where context enters.
- Full six-pair baselines run and the Δ-ladder await a fresh Codex round that
  has this result in hand (round 17 was launched before it).

## 2026-08-28 — NLM-007 corrected rerun (slot endpoint): 3 pairs clear every locked gate

- Six pairs, slot-position completed law, 2145 s of a 3300 s budget; support
  1.0 everywhere; reload check unchanged.
- Mechanical gate reading (Codex adjudication pending): qualifying pairs
  ['L8->L9', 'L12->L13', 'L27->L28']. L0→L1 lexical persistence (word-mean = field on all three
  endpoints). L4→L5 and L20→L21 clear both slot readouts by wide margins but
  miss the +0.05 successor-cosine lead in some folds.
- The word-mean's slot skill decays monotonically with depth (0.95, 0.84,
  0.78, 0.70, 0.43, 0.40) while the affine field holds 0.92–0.98 and the
  static chart collapses late (0.50, 0.51): the share of the move that
  depends on context rather than word identity grows with depth, and by the
  late blocks a chart is nearly useless while an affine law is nearly exact.
- Prediction scorecard: five of six Round 16 readings held; the sixth
  (final-block attenuation at L27→L28) failed — the final pair clears every
  gate, with the qualification that its successor cosine is on normed
  vectors.
- Bounded as before: one model, shared words, reduced 20/500 budget. Next in
  the fixed order: cheap moot-makers (identity-plus-residual, per-carrier
  affine; code written, smoke running), forward-time transport, unseen-word
  split, second family.

## 2026-08-28 — Re-contextualization #6 (step-back, before the corrected rerun lands)

- **Central bet (README):** a native mathematics of latent spaces, built from
  what a denizen must invent to navigate. **Live question today:** does this
  LM world have a reusable, context-conditioned law of motion at any depth,
  or only lexical persistence plus a chart that smooth regression interpolates
  well?
- **What still holds:** successor-endpoint lead of a full-dimensional affine
  field over word-mean and chart from L4 on (one model, shared words), with
  the carrier-shuffle penalty growing with depth; L0 = lexical persistence.
- **What reframes earlier work:** the frozen-encoder program (NLM-002…006b)
  had no move at all — it was a static world, so "law" had no referent. The
  LM world supplies moves; the question sharpened from "is there a native
  metric" to "is there a native law", which is the question a denizen would
  ask first. The audit-#5 defect (right measurement, wrong position) is the
  same shape as the Igor episode; the cadence caught it before any statement.
- **Alternatives held live (not one thread):** (a) smooth implementation-
  specific conditional regression — the strongest rival, testable by the
  cheap moot-makers now written (identity-plus-residual, per-carrier affine);
  (b) the slot law structurally favours the word-mean (it depends only on
  prefix + word) — if so, the last-token readout is the navigation-relevant
  one and the lock should carry both; (c) the real move is forward-time
  (append-token / next-position), untested; (d) the law may be a property of
  this family's training, not of latent worlds — second family pending;
  (e) the whole layer-pair framing may be the wrong unit — a denizen moves
  across the full stack, and a composed multi-block law (L4→L12) would be the
  first genuinely non-local object.
- **Ecosystem thread:** the "measured law vs interpolated chart" distinction
  and the endpoint-position lesson are deposited in `_meta/INDEX.md`; they
  transfer to any project that reads a probe at a position other than the one
  it perturbs.

## 2026-08-27 — Round 16: corrected slot endpoint and next order

- Tier-3 audit #5 applies to every pair: the fallback and extension completed
  laws were read at the sequence's last token, not the substituted slot named
  by the lock. All such completed-law numbers are void for lock purposes.
  Successor scores remain valid exploratory coordinate forecasts under the
  reduced extension budget; no completed-law number is lock-valid.
- The addendum shows hidden index 28 is post-final-norm. The final pair's valid
  completion is `head(Yhat)` at the substituted slot; identity tests pass at
  `L8->L9` and `L27->L28`. Its successor is a normed-vector prediction and
  needs separate comparison from raw-residual pairs.
- Predeclared the corrected full six-pair rerun: slot endpoint, 20 shuffles,
  500 clustered bootstrap replicates, one CPU process, 55-minute hard budget
  (about 48 minutes projected from 24 minutes per three pairs plus margin).
  Fixed predictions and the slot word-mean interpretation are in
  `theory/EXPERIMENTS.md`.
- Alternative order: cheap identity-plus-residual and per-carrier-affine
  diagnostics; forward-time append-token/next-position transport; disjoint
  class-stratified unseen-word split; second model family. The first baseline
  step is specified against the existing artifact with 64/16 word
  cross-fitting per calibration carrier.
- Under the guiding question, word identity is already a field at L0, while
  early blocks manufacture increasing carrier dependence. A denizen needs an
  identity test, context-conditioned transport, and downstream completion,
  validated across new words, time, and realizations.

## 2026-08-28 — Round 15 extension: successor endpoint across depth

- L4→L5, L12→L13, L20→L21 in 1100 s (budget 1800). Successor endpoint valid;
  completed-law numbers are the last-token secondary readout only.
- **L12→L13** matched its prediction on the successor endpoint: ridge 0.977 vs
  chart 0.898 / word-mean 0.888; ≥0.05 over the chart in all four folds with
  clustered lower bounds above zero; low-rank misses by 0.05; shuffled ridge
  null 0.67–0.78. Qualifying status waits on the slot-position endpoint.
- **L4→L5** falsified the prediction: not lexical-persistence dominated
  (ridge beats the word-mean by 0.03–0.06, LB > 0), yet the chart lead reaches
  0.05 in one fold only. **L20→L21**: ridge within 0.02 of kernel (prediction
  "kernel minimal" wrong); chart lead ≥0.05 in two of four folds.
- Depth pattern (one model, shared words): the word-mean equals the field
  only at L0; from L4 on a full-dimensional affine field beats both the
  word-mean and the chart at every depth; the carrier-shuffle penalty grows
  with depth. Reading under the guiding question: carrier-dependence of the
  move is not present at the input and is built up by the world's early
  blocks — the state's dependence on its context is a manufactured quantity,
  not a given.
- Two of three Round 15 predictions failed on the class/minimality side; the
  successor-ordering prediction held. Recorded as such.

## 2026-08-28 — Tier-3 audit #5 and re-contextualization #5

- Audit #5 (fresh Codex) on the L8→L9 claim: the completed-law endpoint read
  the last token, not the slot the lock names — invalid at every pair, not
  only the degenerate late one. Adopted verbatim; analyzer repaired (slot
  primary, last-token secondary). Status is now: "L8→L9 provides exploratory
  evidence that a full-dimensional ridge field predicts stored successor
  states across held-out carrier templates on shared words; the lock-valid
  completed-law endpoint was not implemented, the fallback lacks the required
  second pair, and the result is bounded to one model."
- What still holds: the successor-endpoint lead at L8→L9 over the chart and
  the word-mean, with clustered lower bounds above zero, and the shuffled
  drop; L0→L1 dominated by word-conditioned lexical persistence.
- What is reframed: "first measured law" was premature by one readout
  position. The same mistake shape as the Igor episode — the endpoint
  measured a different question from the one claimed — caught this time
  before any public statement, by the audit cadence.
- Alternatives now live (audit's list adopted into EXPERIMENTS.md): the
  strongest rival is a smooth implementation-specific conditional regression
  (word code + carrier style denoised by a high-dimensional field) that says
  nothing about a native law; the tests that separate the two are the
  unseen-word split, a second model family, and forward-time (append-token /
  next-position) transport — the denizen's actual move, which no NLM-007
  variant yet measures. Cheaper moot-makers to run first: identity-plus-
  residual and per-carrier affine baselines at equal training budget.
- Next: finish the Round 15 extension (successor endpoints valid), smoke the
  corrected endpoint, then a fresh Codex round to predeclare the corrected
  full re-run and the forward-time move.

## 2026-08-28 — NLM-007 (fallback run): an affine transport law at middle depth

- Ran under the declared fallback (L0→L1, L8→L9, L27→L28; 20 shuffles; 500
  bootstrap); 1427 s, 19% over the cap. Float16 reload check passed
  (KL-ordering agreement 0.9998).
- **L0→L1:** word-mean = ridge = kernel = 0.949; shuffled null 0.95. The first
  block's slot action is carrier-independent: lexical persistence, no law
  beyond word identity. Minimal class on both endpoints: word_mean.
- **L8→L9:** ridge/kernel 0.94 versus best static chart (kNN-5) 0.86 and
  word-mean 0.86 on successor cosine; the prior world-completed skill and
  ordering values are void because they used the last-token endpoint. The
  successor lead and shuffled null 0.75–0.84 remain exploratory evidence of a
  carrier-transferring regression field. Low-rank (rank ≤ 128) trails full
  ridge by 0.05 — affine predictor yes, completed-world law not yet shown.
  The within-carrier comparison is descriptive, not a ceiling argument.
- **L27→L28:** successor lead +0.07–0.12, but the completed-law endpoint is
  degenerate by construction — the law is read at the last token and no
  remaining layer connects the slot to it (KL = 0, skill undefined, support
  0.42–0.56). The lock's "only norm and head remain" missed this.
- Verdict within the lock: one successor-only pair supports exploratory
  regression evidence; two corrected completed-law pairs are needed, and the
  fallback is incomplete for the gated verdict. Under the guiding question,
  middle depth may contain a reusable state-transport regularity, but the
  corrected slot endpoint must show that it cashes out in the world's response
  law. Bounded to one model and shared words.

## 2026-08-27 — Round 12: NLM-006b is non-diagnostic; pivot to dynamics

- Adjudicated NLM-006b against the locked `p_e >= 0.80` identity gate. The
  four displaced families measured 0.458, 0.317, 0.185, and 0.416, so all are
  OOD; calibrated displacement and 400/400 support passed for all four.
- The TT chart lead of 0.09–0.29 is therefore descriptive OOD evidence, not a
  gated chart-survival closure. The previous ledger wording is corrected by
  an append-only entry. The small outside-class chart `ST>TS` effect is about
  0.035 with CIs excluding zero.
- Frozen-encoder work closes as a scope decision. The residue is a trained
  task-effective chart, affine-path smoothness, and graceful relative chart
  degradation under identity-destroying moves; no native construct competes.
  Next program: LM residual-stream dynamics, specified in
  `theory/dialogue/003.md`.

## 2026-08-28 — Tier-3 re-contextualization #4 (Claude, before auditor #4)

**Live question.** In a world with its own dynamics (a causal LM's residual
stream), what is the minimal law class that predicts transport across unseen
contexts — and does prediction cash out in the world's completed response?

**Tunnel check.** The frozen-encoder program ended where every instrument
pointed: the trained chart is the operational map, and no probe-built
construct competes. NLM-007 is a different shape (laws, not closeness) but the
same substrate habit — one small LM, 80 words, 16 carriers. Alternatives held
live: (1) a second LM family (SmolLM2/gemma) in the same design, since a law
that holds in one decoder is a fact about that decoder; (2) transport of
*sequence* states (append a token) rather than layer transport — the move the
denizen actually makes in time; (3) the denotational primitive on diffusion
latents, still untouched; (4) the moot-maker: if a low-rank affine field
explains every layer pair at ceiling, the world's dynamics are locally linear
in its chart and the native program reduces to "find the chart in which
dynamics are linear" — a Koopman question, with a literature.

**What reframes earlier work.** The auditor's narrowed residue (task-effective
chart + affine-path smoothness) and NLM-007's question are the same object
seen twice: smoothness of the chart *along paths* is exactly what a linear
transport law would produce. If NLM-007 finds low-rank affine transport at
early/middle depth, it explains NLM-002's within-class monotone paths.

## 2026-08-28 — NLM-006b: the chart survives every displaced transport

- Uncontaminated run (independent candidates, 400/400 support, calibrated
  displacement gate passed by all four families): on the transported pair the
  chart leads the transport-aware natives by 0.09–0.29 with CIs far from zero;
  natives never compete. Label preservation (readout proxy) is 0.19–0.46 for
  the four families vs 0.77 for the near-identity controls, so most chart
  degradation is identity loss, not transport law.
- One real effect: order sensitivity ST > TS ≈ 0.035 (CIs exclude 0) only
  outside the invariance class — substituting then transporting the candidate
  beats transporting the anchor first. Small, but it is the first measured
  non-commutation in this world.
- Per the lock's chart-survival branch, the frozen-encoder transport line
  closes: the trained chart is the operational map for this measured envelope.
  Round 12 decides what replaces the frozen-encoder program.

## 2026-08-27 — Tier-3 re-contextualization #3 (Claude, before auditor #3)
- Infra: the flaky link could not upload the 17 MB transport embeddings (HTTP 408); unpushed history was rewritten to drop them, they are git-ignored, and provenance is the sha256 in the lock. Remote verified in sync via ls-remote (never trust 'Everything up-to-date').


**Where the program stands.** Five measurements in one vision world plus one
in the LM world converge: a trained encoder hands its denizen a chart metric
and straight routes that already serve as the one-step map; nothing we built
from probes competes; an untrained chart has neither. Round 10 closed the
frozen-encoder closeness line and opened NLM-006 (moves outside the trained
invariance class).

**Honest tunnel check.** Since the pivot, every measurement has been
"rank candidates by closeness to an anchor, score against a label". That is
one instrument shape in two worlds. Alternatives I hold live:
1. NLM-006 as designed — the last test inside this shape; if the chart
   survives crops/inversion/mixing/occlusion, closeness-on-frozen-encoders is
   done for good.
2. Worlds with dynamics: LM residual streams, where transport *is* the forward
   pass and the map must predict where the world takes a state — the only
   setting where "move" is not something we impose.
3. The denotational primitive on diffusion latents (evidence-update, not
   closeness) — the legacy program's own repair results are its instrument.
4. The moot-maker: if "inherited from training" fully explains every result,
   the honest program is the study of what training objectives install in a
   chart — encoder-invariance science, not native mathematics. The auditor is
   asked to say whether we have already become that.

**What reframes earlier work.** The legacy separatrix "islands" and NLM-004's
95% null-world flicker are the same phenomenon: chart-straight lines are
routes only where training laid them. The guiding question's "map" is, so
far, not invented by the denizen — it is issued to it.

## 2026-08-27 — Round 10: frozen-encoder closeness/map line closed; NLM-006 opens

- Codex round 10 (`fbe7bee`, blackboard-backed): NLM-005 void (support 32%,
  chart-metric ST−TS ≤ 0.006; R_no_coarse gap 0.027 on shift1px noted);
  NLM-003's R win withdrawn as a coarse-taxonomy leak (leak-free R 0.586 < F
  0.667). The frozen-encoder closeness/map competition is **closed as a
  program**. Residue: training supplies an operational chart and routes — a
  denizen inherits navigation equipment — not a proven intrinsic geometry.
- NLM-006 opens: transports outside the trained invariance class (large crop,
  color inversion, image mixing, occlusion), each verified non-near-identity by
  embedding displacement; stratified candidates (20 same-class + 20 hard
  negatives, frozen); support ≥80%; decisive if ≥2 of 4 families break the
  chart lead or expose a transport-aware native predictor. Building the
  artifact now.
- Infrastructure: blackboard MCP works for Codex via the installed binary
  (npx cold start exceeded its startup timeout); TOML paths need forward
  slashes.

## 2026-08-27 — NLM-005 composition: void on support, non-diagnostic, and a design lesson

- Composed moves with hflip / 1-px-shift transports re-encoded by the frozen
  encoder: ST−TS gaps ≤ 0.006 for every predictor (kill 2), support 129/400
  (kill 3). Cosine leads both native constructs by 0.32 on every order.
- Lesson: these transports are exactly the augmentations DINOv2 was trained
  to be invariant to — near-identity moves in its world — so composition with
  them cannot reveal a law. Transports must lie outside the trained invariance
  class (large crops, color inversion, image mixing, occlusion, or a different
  encoder's edits). And support needs stratified candidates (same-class
  candidates by construction), not 40 random draws over 100 classes.
- Standing picture after five measurements: in trained worlds the chart
  metric is the map for one-step consequences and survives trained-invariant
  transports; it collapses in an untrained chart (NLM-004). No native
  construct built so far competes with it. The program's next honest question
  is whether *any* move outside the invariance class breaks the chart.

## 2026-08-27 — NLM-003 diagnostics: R's win was the coarse head

- Rerun with audit-#2 diagnostics (same lock, new anchor sample): R without
  the coarse head = 0.586, below F (0.667). R's advantage was taxonomy leak —
  fine labels nest inside coarse classes — exactly as the fresh auditor
  predicted. Δ_F−R = −0.095 [−0.142, −0.049] marginally fails the strict gate
  on this resample. R ties on 22–33% of comparisons.
- Cheap ladder: PCA-32 cosine 0.941 ≈ cosine 0.934; pixel baselines 0.62.
  k-sensitivity: same-class fine-kNN flicker 0.10–0.18, cross-class 0.37–0.41
  for k = 8/32/128 — the world-path contrast is robust.
- Net for the primitives: neither F nor R (leak-free) is a competitive map;
  the trained chart's metric is. The program's live positive result is
  NLM-004's: that metric and its straight routes are products of training.

## 2026-08-27 — NLM-004 null world: the chart's map and its world-paths are inherited from training

- Preregistered in the ledger (before scoring) and supported: in a random-init
  DINOv2 chart the cosine map for fine-label consequences collapses from 0.946
  to 0.575 (gap 0.37 ≥ 0.20); embedding-kNN fine accuracy 0.761 → 0.069; F and
  R collapse too (0.58 / 0.57); raw-pixel and pixel-statistic baselines (0.62)
  now beat cosine. Pixel-statistic heads stay strong (rgb 0.83, luma 0.82):
  the null chart preserves pixel structure, not semantics.
- Sharper: null-world M1 — same-class fine-kNN flicker along chart-straight
  lines is 95% (trained: 12.7%), cross-class 99% (trained: 38%). So "straight
  lines are world-paths within a class" is itself a product of training. In a
  trained world the denizen inherits both a metric and a set of straight
  routes; in an untrained chart neither exists.
- Ties: R's profile statistic ties on 33–36% of comparisons (5-valued), as the
  audit warned; trained-world tie fractions, R-without-coarse, k-sensitivity
  and the cheap-baseline ladder are being rerun as nlm003_v2_diagnostics.
- Duplicate-run lesson: three NLM-004 launches overlapped and starved each
  other; the process check with a quoted tasklist filter was wrong. Runs are
  now single, detached, file-logged, with a completion watcher.

## 2026-08-27 — Tier-3 re-contextualization #2 (Claude, before the auditor)

**Live question now:** in every world tested so far (LM input rows, LM residual
states, DINOv2), the model's own chart metric is the best one-step map for
consequences. Is that a law of trained latent worlds, or an artifact of only
testing one-step moves in encoders trained to make one chart metric meaningful?

**Tunnel check.** Two days of work have stayed on "closeness / one-step map of
a frozen embedding" — three instruments, one shape. Same-shape follow-ups are
now forbidden by our own rule. Live alternatives, each with a decisive result:
1. Null world (random-init encoder, building now): if cosine still predicts
   fine-label consequence there, cosine tracks pixel similarity, not training;
   if it collapses, the chart metric's dominance is a product of training and
   the denizen's map is *inherited*, not native.
2. Two-step moves (composition): substitution∘transport vs transport∘substitution
   — a one-step metric cannot express non-commutation; if it exists, that is
   the first law no chart metric captures.
3. Cross-class world-paths (M1: 38% detours): the geometry of *routes*, not
   distances — a routing map the metric does not give.
4. Worlds with dynamics: LM residual states across layers (transport is the
   physics); the denizen's map must predict where transport takes a state.
5. The moot-maker: if a fixed contrastive-training explanation ("cosine is
   meaningful because the loss made it so") accounts for every result, the
   native program on trained encoders reduces to studying training objectives.

**What reframes earlier work.** NLM-001's loss to contextual cosine and
NLM-003's loss to chart cosine are the same finding: trained charts carry
their own metric. The program's object should shift from *closeness* to
*moves and laws* — where the chart has nothing to say.

## 2026-08-27 — NLM-003: R beats F, both dominated by the chart metric; blackboard live

- NLM-003 (locked `e2a1fb2`, true fine-label endpoint, same artifact): R
  (substitution-profile agreement) beats F (Fisher pullback) — Δ_F−R = −0.104
  [−0.148, −0.058], gate met (R 0.734 vs F 0.630). Decisive context: plain cosine
  in the DINOv2 chart scores 0.946, Euclidean 0.935 — both native constructs
  lose by 20–30 points to the imported metric on the informative endpoint.
  Support thin: 130/400 anchors had a same-class candidate among 40 draws.
- Reading before Codex round 8: DINOv2 is trained so that one chart metric is
  meaningful; in such a world the denizen's best one-step map *is* that metric.
  NLM-001 found the same in the LM world (contextual cosine at L14 = 1.000). So
  far every world tested has a chart metric that already is the map for
  one-step consequences. Where native structure could still differ from the
  chart: (i) two-step moves — does substitution-then-transport equal
  transport-then-substitution (composition / laws), which a one-step metric
  cannot express; (ii) worlds whose chart was not trained to be metric (raw
  residual states of a non-contrastive model, or a randomly-initialized
  encoder as a null world); (iii) cross-class world-paths (M1: 38% detours).
- Blackboard: `@iqidis/blackboard-mcp` installed globally, registered for
  Claude Code (user scope) and Codex; mandated in global CLAUDE.md and the
  setup skill; Codex verified `bb_list`/`bb_create`/`bb_add_entries` and seeded
  the project board (`.blackboard/5df235ea`, git-ignored). This session cannot
  call `bb_*` until restart; Codex rounds now use it.

## 2026-08-27 — NLM-002 non-LM branch run: endpoint killed, chart-path structure found

- Artifact frozen (CIFAR-100 → DINOv2-small, 6000/2000, sha256 8de4f0b0…);
  locked with two recorded implementation decisions; run in 133 s on CPU.
- M2: the raw-pixel k=32 kNN fine-label endpoint is nearly uninformative
  (0.115 accuracy; 0.12 agreement with embedding kNN) → preregistered endpoint
  kill condition met. M3 (F vs R) is therefore a tie on noise (Δ = −0.004
  [−0.034, +0.026]); no primitive verdict. Lesson: independence is necessary,
  informativeness is not optional — the true fine label (no head trained on it)
  is both and should have been the endpoint.
- M1 (chart-path closure), the informative result: along chart-straight lines
  between same-class embeddings the coarse-semantic readout is monotone in 98%
  of paths (flicker 2% [0.3, 3.7]); between classes, fine-label kNN flickers on
  38% [32, 44] of lines and any-readout on 78% [73, 83] — straight lines between
  classes pass through third classes. Within-class chart lines are near
  world-paths for semantics; cross-class lines are not. Pixel-statistic heads
  are weak (52–59% test accuracy) and their flicker is partly head noise.
- Process: Codex sessions are now always fresh with terse file-pointing
  prompts (Devansh); `_meta/INDEX.md` row for this project updated for sister
  agents; no blackboard MCP is configured on this machine.

## 2026-08-27 — NLM-001 closed negative; NLM-002 designed as a primitive competition

- NLM-001 verdict (Codex round 3, `1584514`): instrument-void for confirmation
  (runtime metadata reconstructed post hoc), bounded negative falsifier of the
  lexical-KL instrument. Native calibration-KL lost to a symmetric metric on
  contextual hidden states (Qwen Δ = −0.058 [−0.22, +0.03]; unlearned centered
  cosine at layer 14 reached 1.000 on held-out orderings vs native 0.954);
  context reversals exceeded the paraphrase null in 2/3 systems (Qwen Q = 2.12
  [1.70, 2.56]); directedness absent. T2/T3 demoted to bookkeeping; T1's
  conjunction-closure premise fixed. Fresh Tier-3 audit adopted.
- NLM-002 (`theory/dialogue/002.md`, skeleton, not locked): mutual-kill
  competition between F (one fixed Fisher response-law geometry pulled back
  through frozen decoders/heads) and R (probe-indexed substitutability tested
  outside LMs, on DINOv2 image embeddings), each with an independent behavioral
  endpoint and a common-support Q estimator.
- Artifact prep for arm R: no image data existed locally. Building
  `experiments/results/vision_cifar100_dinov2s/` — CIFAR-100 (fine + coarse
  labels) → DINOv2-small CLS embeddings on CPU (35 ms/image), plus label-free
  pixel statistics (mean RGB, luminance, edge density) so probe blocks can ask
  questions not derived from the class taxonomy. Manifest carries dataset and
  encoder revisions, split indices, seed, sha256.
- F-arm implementation note for the lock: the LM Fisher pullback can be
  estimated as G = mean over sampled tokens t~K_c(z) of the outer product of
  ∇_z log K_c(z)(t) (VJPs through the frozen decoder; ~64 samples per state and
  calibration probe, D = 1024, CPU-feasible in ~30 min); the DINOv2 arm's Fisher
  is exact for a linear head (G = mean_z W^T F(p) W).

## 2026-08-27 — Tier-3 re-contextualization (Claude, before the auditor's answer)

**Live question.** Is closeness in a latent space context-indexed and directed
in a way no symmetric contextual representation reproduces — and can a native
invariant (context rank, Q = B/W) say how many orderings are actually required?

**Tunnel check — honest answer: partially tunneled.** In one day the program
narrowed from "native mathematics of latent spaces" to "next-token-KL
substitution probes on 80 lexical tokens of three tiny causal LMs". That is a
fine first instrument; it is not the object. Risks: (i) everything measured is a
property of one decoder family; (ii) 'latent space' has so far meant 'input
embedding row', the least latent thing in the model; (iii) the primitive
(substitutability under probes) was chosen in round 1 and never competed.

**Alternatives now live, each with its decisive result:**
1. Non-LM latent spaces with downstream-head probes (DINOv2, ESM-2, CLIP,
   wav2vec are cached). If the same axioms/invariants organize a vision or
   protein space, this is about latent spaces; if not, it is about LMs.
2. The probabilistic/denotational axiomatization (dialogue 001 §1B) on
   diffusion latents — the legacy program left instrumented diffusion stacks
   that already denote laws. Decisive: evidence-update is definable there and
   predicts something the relational axioms cannot.
3. Probe family as the object: states × probes as a formal context (Galois
   connection / formal concept analysis) — existing mathematics for exactly
   the (X, C, N) structure. Decisive: the concept lattice of a real system has
   nontrivial structure that context rank flattens.
4. Intermediate residual states as latent states (not embedding rows), probes
   = continuation from that depth. Decisive: context rank changes with depth.
5. The moot-maker: contextual cosine at the right layer matches native transfer
   (H3 kill 3/6). Then "native" = "the model's contextual representation" and
   the program must change primitive or object.

**What reframes earlier work.** The legacy perturbation program was, in these
terms, a substitution probe with the whole prompt as state and free generation
as the readout; its withdrawn results were the readout's insensitivity, not the
state's. Its one surviving residue — measure the stack's numerical noise floor
before interpreting any latent-space difference — is now a standing gate here
(η) and belongs in every project that measures representations.

**Direction still makes sense?** Yes as a first falsifiable instrument, with the
explicit condition that NLM-001's verdict (either way) must be followed by at
least one of alternatives 1–4, not by NLM-002 on more words.

## 2026-08-27 — Round 2b instrument calibration amendment

- Read the disclosed eight-item full-pipeline calibration. Within-block KL
  scale varied up to 16-fold, making the preregistered instance-specific MAD
  threshold invalid and yielding zero passes.
- Locked per-paraphrase scale normalization, four-of-four sign agreement, a
  block-pooled magnitude scale, and the explicit 12.5% random-sign null.
- Demoted directed asymmetry to exploratory after 0–18% sign agreement at
  chance. Revised post-calibration predictions to \(Q=2.5\), \(R=0.20\), and
  \(\Delta_{\rm rev}=+0.07\).
- The eight inspected words are excluded from primary confirmation; the
  remaining 72 are primary and all 80 are sensitivity. No confirmatory run was
  performed.

## 2026-08-27 — Round 1 revised after Claude attack

- Appended `## Codex — revision` to `theory/dialogue/001.md`, answering A1–A7
  point by point. Withdrew existential non-collapse as an axiom; adopted finite
  context rank with anchor-dependent radii.
- Proved the finite representation theorem: context rank is 1 exactly when
  every anchor's context neighborhoods form an inclusion chain. Derived the
  incompatibility-graph coloring characterization.
- Created `theory/AXIOMS.md` as the living formal surface. Local refinement is
  now conditional on a completed probe family; cross-realization agreement is a
  measurement of transportability, not an identity axiom.
- Created `theory/EXPERIMENTS.md` with the post-smoke, pre-measurement NLM-001
  preregistration: primary directed asymmetry, graded context rank, held-out
  transfer against contextual-cosine and learned-metric baselines, paraphrase
  nulls, cross-system checks, exact predictions, and kill conditions.
- No experiment was run in this revision; the confirmatory measurement remains
  unrun. A concurrent Claude turn added the CPU measurement runner and raw
  hidden-state capture. Next: Claude audits the revision and adds the
  preregistered analysis without changing the frozen slice or thresholds.

## 2026-08-27 — Repository restarted

- Entire prior program moved unmodified to `legacy/` (its README, docs,
  experiments, results, and correction record intact and internally linked).
  Root now holds only the new program: README, STATE, NOTEBOOK, `theory/`,
  fresh `experiments/` ledger. Local-only ignore patterns mirrored for `legacy/`.
- Four watchdogs live (20-min liveness, hourly ops, 2-hour Codex audit +
  anti-tunnel, 2-hour entropy sweep). Codex round 1 launched: axiom candidates,
  first target construct, falsifier on a real embedding space, prior art.

## 2026-08-27 — Native latent mathematics, dialogue round 1

- Wrote `theory/dialogue/001.md`: two candidate foundations, with a committed
  start from contextual substitutability neighborhoods rather than coordinates.
- Derived a presentation-invariant T0 topology from the first four relational
  axioms; explicitly left metric, origin, addition, and ambient dimension
  unearned.
- Pre-registered a CPU-only Qwen3-0.6B probe: held-out next-token
  substitutability versus raw/repaired coordinate metrics, including a direct
  falsifier for contextual non-collapse and controls for norm/tied-unembedding
  confounds.
- Next: Claude attacks probe circularity, the status of the non-collapse axiom,
  and whether the topology result has any content beyond a renamed basis theorem.

## 2026-08-27 — Direction set: native mathematics of latent spaces

- Prior LLM-perturbation program closed; arithmetic claims withdrawn after Igor
  Rivin's PRs #4/#5 (merged) and reanalysis of stored data. Correction record in
  `docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md`; README rewritten to a findings
  table; three doc indexes consolidated to `docs/NAVIGATION.md`; orphaned docs
  archived.
- Residue: decoding determinism is hardware-dependent (GH200 non-deterministic,
  RTX 5090 deterministic); perturbation is a causal diversity source only on
  deterministic stacks. Process gates added (termination, direct control first,
  null model, clustered stats, propagation).
- New direction opened (see STATE.md). Starting the Codex dialogue on axiom
  candidates and the first target construct.
