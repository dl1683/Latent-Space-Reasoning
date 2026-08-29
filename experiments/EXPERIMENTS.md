# Experiments Log

Reverse chronological. Only gate-passed conclusions are stated as confirmed.
Program opened 2026-08-27; prior program's log is at `legacy/experiments/EXPERIMENTS.md`.

---

## Program status (2026-08-29)

- **NLM-007 — CLOSED** under the program's terminal allocation rule, not by a
  scientific null (audit #22 closing statement, verbatim in `STATE.md`; ledger
  `nlm007_closed_audit22`). Terminal ladder: 34a raw CONTINUE -> 34a static
  CONTINUE -> 34b INCONCLUSIVE (terminal rung) -> 34c not run. No operational
  state, native law, composition, representation-level hostile hole, or
  independent replication was identified. Every bullet under the NLM-007
  heading below is the closed record; its queue/order language is historical.
- **Toy quotient-world program (Rounds 36–37) — ENDED 2026-08-29** under the
  governance amendment in `AGENTS.md` (exact certificates are diagnostics
  only; one audit per result, which must answer "should this continue"; real
  models only; ratio tripwire). Rounds 36 v1 / 36b / 36c / 36d and Round 37
  are closed results with one licensed reading each (audits #23–#26 and the
  Round 37 audit; verbatim in `STATE.md` "Closed toy program — licensed
  wording"). No learned artifact passed the complete exact reducer (only the
  oracle fixture); Round 37: `NO ARCHITECTURAL WIN`. Runners and configs were
  retired from the active path (`f6dac0e`; git history); verdicts retained
  under `experiments/results/operational_quotient_*/` and
  `experiments/results/presentation_quotient_v1_*/`. Entries below.
- **Current artifact — `coordinate_v1`** (`experiments/run_coordinate.py`,
  `experiments/config/coordinate_v1.json`): demo stopped at calibration; the
  explicit baseline is invalid for polarity on Qwen3-0.6B; direction decision
  pending with Codex (entry below). No successor run is authorized.

## coordinate_v1 — two-bit causal coordinate (tense × polarity), Qwen3-0.6B residual stream (2026-08-29; registered artifact; demo stopped at calibration)

- **Registration.** NOTEBOOK re-contextualization #27; Codex direction
  dialogue rounds 1–4 (`.codex_direction_r1..r4`, not committed); runner
  `experiments/run_coordinate.py` (`7f66f55`, matched explicit baseline /
  held-out single axes / fixed random directions / hash-stamped results
  `ddb5eee`), config `experiments/config/coordinate_v1.json`. Design: two
  moves `v_T`, `v_N` estimated leave-one-family-out from single-axis
  calibration states `00/10/01` only (state `11` never used adaptively);
  causal layer rule at coefficient one on the final prompt token; free decode
  primary; chance 1/4 among canonical forms; sham + norm-matched random
  controls; explicit-instruction baseline.
- **Demo outcome** (`experiments/results/coordinate_v1/demo.log`,
  `result.json`; Qwen3-0.6B rev `c1899de2…`, CPU): calibration hidden states
  captured `(16, 28, 1024)`; **no block cleared the calibration rule** —
  `acc_T = acc_N = 0.0` with termination `1.0` at every block 0–27 — so the
  run stopped before any held-out transport; `layer: null`. Not a result
  about residual-stream coordinates generally: a bounded negative for this
  exact one-shot final-token intervention at coefficient one.
- **Baseline validity** (Codex direction round 4 numerical check, not a
  registered run): on Qwen3-0.6B the explicit-instruction baseline scored
  `14/32` (`00` 8, `10` 6, `01` 0, `11` 0) under the wording repair and the
  polarity states failed under the original wording as well, so tense ×
  polarity was **not a valid two-bit task on this model**; a 1.7B check
  reached `29/32` with `01` at `5/8`, still below a per-state gate. Codex's
  proposed headline for this artifact: `UNINTERPRETABLE — INVALID POLARITY
  BASELINE`, with "stopped at calibration" as the execution subfinding.
- **Status:** direction decision pending with Codex (model/axis change
  proposed in round 4); nothing in `coordinate_v1` is a positive or negative
  claim about a residual-stream coordinate. Ledger: `ops_heartbeat`
  2026-08-29T18:49Z records the demo stage; no result row yet.

## Round 37 — presentation-duplicated 32->16 quotient world: NO ARCHITECTURAL WIN; last toy-world round (2026-08-29; ledger `round37_lock`, `round37_result`, `round37_audit`)

- **Lock** (`round37_lock`, `eb61470`; config sha256 `5a419bea…`, module
  sha256 `6786b67b…`; registered by Codex in `theory/EXPERIMENTS.md`):
  32 presentations, true 16-class quotient; quotient-factored `z=(q,p)`
  carrier vs unrestricted carrier × 2 presentation roles × 5 seeds; rolled
  interchangeability primary; non-gating horizon/role diagnostics; Tier-1
  review passed (fixture 46.3 s, 529.5 MiB peak). Runner/config retired from
  the active path after the result (`f6dac0e`; git history).
- **Result** (`round37_result`, `b5bef1b`; artifacts
  `experiments/results/presentation_quotient_v1_{factored,unrestricted}/`
  `verdict.json` + `manifest.json` + `config.json`; evidence/weights retired,
  sha256-pinned; 35 min CPU wall): both carriers `FAIL — BEHAVIOR UNDERFIT OR
  BASE SIGNATURE UNSUPPORTED`; comparison `NO ARCHITECTURAL WIN`; both
  verdict files carry one factored-primary world verdict. Diagnostic-only:
  H2 held-out supported-truthful cells factored 867–1184/1184 vs unrestricted
  795–1184/1184; H3 factored 365–754/1056 vs unrestricted 485–992/1056; first
  divergence predominantly at step 3 of H3.
- **Licensed sentence (Round 37 audit, `round37_audit`, `d9ca753`; verbatim):**
  Under the frozen exact toy reducer, neither carrier reached
  behavior-qualified held-out presentation transfer or rolled
  interchangeability; descriptively the unrestricted carrier had higher
  transfer and interchangeability rates in 9 of 10 paired seed × role units,
  while failures were predominantly—but not exclusively—future-signature
  failures and H3 first divergence was predominantly at step 3, so the
  imposed factorization showed no benefit in this setup.
- **Never say:** every failure was future-signature / terminal responses
  never failed (128 factored and 69 unrestricted failure cells involve a
  terminal error; four cells are terminal-only) / the factorization
  constraint is harmful / unrestricted is the architectural winner / the
  Round 36d mechanism reproduced (only the horizon localization recurred) /
  the hole is a property of behaviour-supervised learning (hypothesis only) /
  anything about real residual streams based on Round 37.
- **What we learned:** the toy program ends; transferable residue = identity
  by causal interchangeability, and never use exact certificates as primary
  evidence for learned continuous systems.

## Rounds 36 v1 / 36b / 36c / 36d — minimal operational-quotient world (closed; audits #23–#26 govern)

- **Round 36 v1 — valid registered FAIL** of its frozen `0.10/0.90` exact
  reducer, every gate (`073037f`; adjudicated `e69ac72`; config
  `experiments/config/operational_quotient_v1.json` (retired; git history); artifacts
  `experiments/results/operational_quotient_v1/`). Only licensed
  reading = audit #23, verbatim in `STATE.md`: a behavior-, calibration-, and
  exactness-confounded non-certification of the registered operational
  quotient — not evidence of no approximate composable structure; the
  confidence-free replay numbers are DIAGNOSTIC only. Never say "did not
  supply ... composable action algebra", "0/176 cross-seed agreement" as a
  cell count, or "FIT BUT NON-CONGRUENT" for a support-only failure.
- **Round 36b — complete; all four cells behaviour-ineligible** (lock V3,
  review #2 RUN-READY, `61e2430`; results `abef6cf`; ledger
  `round36b_ladder`; audit #24 `round36b_audit24`, adopted `57b0961`;
  configs `experiments/config/operational_quotient_36b_{S16,S64,LR64,W64}.json` (retired; git history);
  artifacts `experiments/results/operational_quotient_36b_*/`).
  Every cell `FAIL — BEHAVIOR UNDERFIT; QUOTIENT INELIGIBLE`; no eligible
  cell, no PASS. Only licensed reading = audit #24, verbatim in `STATE.md`:
  the ladder did not reach exact held-out eligibility, while the
  reachability of that exact learned precondition remains unvalidated — not
  proven unsatisfiable; W64's cross-seed-stable canonical one-step skeleton
  is informational only, not a certified operational quotient/action
  algebra. Never say "unsatisfiable by construction", "the exact structural
  gates are unreachable", "the latent is unorganized", or "organized"
  without the local/canonical scope.
- **Round 36c — complete; both cells FAIL (POSITIVE-CONTROL scope; audit
  #25 governs).** Explicitly transition-supervised learned positive control
  (behavioural BCE + `1.0 *` MSE to the stop-gradient true-successor
  encoding; unchanged exact reducer; lock V2, review #2 RUN-READY,
  `dd699e2`; configs `experiments/config/operational_quotient_36c_{w32,w64}.json`, retired; git history).
  w32 (`3742df8`; ledger `round36c_w32_positive_control`): FAIL on every
  exact gate in every seed, action-table truth 0/5, cross-seed table FAIL.
  w64 (`d5975c1`; ledger `round36c_w64_positive_control`): FAIL — swap/toggle
  table 4/5, held-out depth-2 closure 3/5, all other gates 0/5. Only licensed
  reading = audit #25 (`round36c_audit25`, adopted `0f61280`; verbatim in
  `STATE.md`) and the w32 adjudication's licensed sentence
  (`round36c_w32_adjudication`): the registered joint moving-target
  learned-target recipe did not reach the certificate; learned gate
  reachability unresolved — not proof that the carrier or exact gates are
  unreachable; no behaviour-only interpretation; W64 stays informational
  under audit #24; only the oracle fixture has passed the reducer. Never say
  "not reachable even with direct supervision", "certification-regime
  problem" as the cause, "the auxiliary objective caused it" (leading
  hypothesis only), or "W64 beat the control". No further moving-target cells.
- **Round 36d — complete; joint FAIL — INTERCHANGEABILITY (POSITIVE-CONTROL,
  frozen chart; audit #26 governs).** The one permitted frozen-target
  head-only calibration (audit #25 rank 1): hash-pinned behaviour-only W64
  encoder/readout frozen, fresh width-64 transition head trained on the
  stationary 176-cell successor table for 16,000 steps, BCE as an evaluation
  gate, unchanged exact reducer; ONE capped cell. Registered before any
  outcome (ledger `round36d_37_registered` -> `round36d_impl_discrepancy` ->
  lock V2 `round36d_lock_v2` -> review #1 RUN-READY `round36d_run_ready`;
  runner + config `6a95c29`,
  `experiments/config/operational_quotient_36d_w64.json`, retired; git history). Run `7a5ce35`
  (ledger `round36d_frozen_chart_positive_control_produce` / `_reduce`,
  `round36d_frozen_chart_control`; `118.454 s` of a `480 s` wall; artifacts
  `experiments/results/operational_quotient_36d_w64/` — `config.json`,
  `manifest.json`, `verdict.json` committed; `evidence.json`, `weights.npz`
  git-ignored, sha256-pinned): exact behaviour in all five seeds
  (`21,184/21,184` train, `2,240/2,240` held out); exact PASS on quotient
  availability, well-definedness, toggle involution, swap/toggle table, H2
  closure, H3 closure, canonical action-table truth, and the cross-seed
  table; interchangeability misses `16/5/28/98/0` of `132,160` cells
  (`147/660,800`; all depth-2 rolled histories followed by H3; 89
  confidence-only, 58 with a future-probe truth error; no immediate
  endpoint error), so `1/5` seeds pass and the joint status is
  `FAIL — INTERCHANGEABILITY`. Only licensed reading = audit #26
  (`round36d_audit26`, adopted `cad85ef`; verbatim in `STATE.md`): narrow
  individual-gate reachability under privileged full-table supervision, not
  quotient discovery or a complete learned certificate; no complete learned
  artifact has passed the joint reducer (only the oracle fixture has); the
  adequacy ratio is diagnostic and does not identify cause. Never say
  "reached the exact certificate", "the learned-pass gap is closed", "the
  quotient/action algebra was learned", "eight independent gates", or
  "interchangeability generally fails". Round 36 calibration is closed; no
  further Round 36 cells.

## NLM-007 — LM residual-stream dynamics; middle-depth ridge lead withdrawn under the identity baseline; displacement ladder adjudicated (audit #8 wording); forward-time move adjudicated NOT MET = nonpass, not a kill (Round 20, audit #9); within-style null = diagnostic only (both arms); LOCO A/B within-family positive, bounded (audit #10 wording; adjudicated Round 22); equalized addendum defect-affected, descriptive only (audit #11); unseen-word runs A/B mechanical pass, formal gate pending (Round 23, audit #12); corrected equalized reruns contract-correct (A adjudicated Round 25; B pending); residualization A-static contract-valid, adjudicated Round 26 as corrected by audit #14 (registered-presentation sensitivity + surviving X-linked residual predictability; neither state nor presentation-independence); A-augmented (`P_aug-score4`) adjudicated Round 27 as corrected by audit #15 (nested sensitivity on the same sentinel-A cells, not a replication; outcome-clean but transductive within carrier); B-static adjudicated Round 28 as corrected by audit #16 (F4–F20 pass, F0 fails; a correlated two-sentinel check, not replication); B-augmented (`P_aug-score4`) scored on the third launch after two losses in the F8 grammar block, adjudicated Round 32 as amended-implementation and SVD-telemetry-incomplete (F4–F20 pass, F0 fails); the sentinel {A,B} × {P_static, P_aug-score4} table is complete only for the residual-versus-four-word-only-null mechanical gate — within-decoder, within-population condition robustness, not replication (audit #17); patched A-static `resSA2` complete and ADJUDICATED (Evidence gate 2026-08-28, PASS qualified: the four-cell table is complete on one common K=13/four-null/crossed-bootstrap scale; F4–F20 pass in all four correlated cells; F0 non-qualifying in three cells with a weak pooled A-score4 exception; audit #18: F0 is a model-class-sensitive diagnostic, not an all-field dead end); contextual-prefix screens `ctxscr_A/B` = point-only screens that did not triage the X-conditioned hypothesis out at F4–F20 (ctx effective df ~42.7 vs state ridge ~210–406; no state-reading gate passed; audit #18); SVD telemetry gate PARKED and unpassed by allocation (repair cap = global CLAUDE.md §2.7, not an AGENTS.md rule); Round 33 consequence instrument implemented, UNRUN, PARKED and unpassed after Tier-1 re-review #4 NOT-READY (joint-key rule closed; provenance/parity blockers open; repair cap applied as an allocation decision, ledger `nlm007_consequence_instrument_parked`; branch `conseq-instrument`, main analyzer at HEAD); contextual-prefix completions `ctx_A/ctx_B` (unresidualized) COMPLETE, scored pending adjudication, audit #19 wording upheld by audit #20: descriptive higher-EDF predictor comparisons — at F4–F20 the higher-EDF state predictor retained a positive held-out score difference from the registered `token_ids_v1` context-only pair with positive crossed lower bounds, 8/8 point-positive keys, no family collapse, support 1.0; state ridge ~5–10× the contextual ridge's effective df; F0 non-qualifying; NOT evidence that context failed, that capacity is the sole confound, or that operational state is identified (the phrase "did not close the gap" is withdrawn); Round 34 capacity-matched audit registered (`d493cf2`) and implemented (`9eb1301`; producer RUN-READY, joint reducer flagged) but HELD pending the preregistered Round 34a matched-EDF core screen (`f97a533`; K=13 KL-rank diagnostic, raw continuous KL confirmatory; audit #19 staging); `ctxS_A` and `ctxS_B` (P_static-residualized) COMPLETE, scored pending adjudication, audit #20 wording (sentinel B mirrors A: context cosine ~0.04–0.08 / nerr ~1.00 vs residual ridge cosine ~0.52–0.58 / nerr 0.82–0.86): the registered token-context ridge/kernel falls to held-out cosine ~0.04–0.07 and normalized error ~1.00 while the residual `X_perp` ridge keeps cosine ~0.56–0.62 and normalized error 0.78–0.83; raw context performance is highly non-robust to the registered `P_static` residualization and therefore `P_static`-aligned in this fitted design — NEVER "beyond template metadata", "presentation removed", "largely by construction", or "same feature space"; licensed positive wording: a higher-capacity predictor from `X_perp` carries held-out predictive information beyond the registered `P_static` nuisance projection and this fixed `token_ids_v1` context field; Round 34a matched-EDF core screen RUN-READY (`6b93ff1`, four Tier-1 rounds; six runs queued: `ctxcapA_raw`, `ctxcapB_raw`, `ctxcap_raw_joint`, `ctxcapA_static`, `ctxcapB_static`, `ctxcap_static_joint`; both estimands required, no cross-estimand verdict); Round 34b (P/C partial-overlap screen) and Round 34c (item-embedding-by-P_static X-free comparator) preregistered (`3b49321`, `ff69d82`), implemented on the main analyzer, UNRUN and under Tier-1 repair (review #1 NOT-READY on four items), to run after Round 34a and before the full Round 34 (held); Round 35 typed truth-evaluable world preregistered docs-only (`c74bfab`; authors nothing until the 34a/34b/34c ladder resolves); Round 33 parked; chain running on the frozen analyzer copy (`6b93ff1` blob): parity check (`parity_head` A/B done, `parity_ref` A running, then B and a parity verdict), then the six Round 34a runs; capture extension committed (`4137258`); contextual-prefix X-free baseline committed (`eab0a68`); populations: v1 design-void (audit #16), v2 and v3 voided by the independent linguistic adversary, v4 approved 48/48 and frozen — a bounded mentioned-string instruction micro-world (audit #17); operation-verb update = declared-operation-verb context intervention; Round 31 chain and X-free chain disarmed; Freedman–Lane conditional on one A-static cell; audit #17 allocation ruling in force (Round 33); retention marker non-commensurate (audit #13); **PROGRAM CONTINUATION RULING (2026-08-29, ledger `nlm007_program_continuation_ruling`; supersedes every queue item before it in this header): NLM-007 STOPPED as an open-ended program (infrastructure drift 6:1); terminal closeout ladder only — Round 34a raw (DONE, CONTINUE) -> Round 34a static (RUNNING) -> Round 34b (conditional on both 34a CONTINUE + final bounded repair RUN-READY) -> Round 34c (conditional on a 34b CONTINUE); first STOP/MOOT/REDUNDANT/INCONCLUSIVE rung ends it; all-CONTINUE = one narrow measurement claim then closure; CUT: full Round 34, Round 33 (branch archived as tag `archive/conseq-instrument-parked`), the parity check as a gate, the random-weight null, a second decoder; parity verdict IDENTICAL A/B (`analysis_parity_*.json`; evidence only); Round 34a RAW CONTINUE at F4–F20 in both sentinels with capacity-matched cosine margins +0.04 to +0.08 (LBs 0.02–0.05), 8/8 keys, F0 INCONCLUSIVE/diagnostic, joint COMPLETE/SCREEN-ONLY CONTINUE — capacity matching removed most of the unmatched raw gap; a narrow survival, not a strong one, not a state claim; Round 35 = requirements envelope only; Round 36 minimal operational-quotient world = the constructive program (design registered `c26eee4`; implementation in progress); producer/reducer separation mandatory** (2026-08-29); **CLOSED 2026-08-29 at the Round 34b INCONCLUSIVE rung — see the program status above and the 34a-static / 34b bullets below**

- **Lock.** Round 13, documentation-only (ledger `nlm007_round13_lock`;
  design `theory/dialogue/003.md`, `theory/EXPERIMENTS.md`); Round 14
  amendment `097e2df`; Round 16 correction (completed law read at the
  substituted slot; final pair uses `head(Yhat)` on the post-norm state,
  ledger `nlm007_round16_corrected_rerun_predeclared`). Qwen3-0.6B (28
  layers), 80 one-token words × 16 carriers, four carrier-block folds; six
  layer pairs; law ladder word-mean / kNN / ridge / low-rank affine / kernel
  ridge; per-carrier oracle; within-word carrier permutations; two-way
  cluster bootstrap. Decision: ≥0.05 lead over the best static chart with
  lower bound >0 on successor cosine and both completed-law readouts in ≥2
  layer pairs. CPU only.
- **Capture.** `experiments/run_lm_dynamics.py` →
  `experiments/results/lm_dyn_v1/manifest.json` (model revision c1899de2…,
  batch 16, batched-vs-single nulls ≤ 6.1e-5, 79 s). `states.npz` is
  git-ignored; sha256 `6ec9520845811bbd…` recorded in the manifest.
- **Artifacts (`experiments/results/lm_dyn_v1/`; all kept).**
  - `analysis.json` — fallback run, pairs L0→1 / L8→9 / L27→28, 20 shuffles,
    500 boot (ledger `nlm007_fallback_declared`, `nlm007_v1_fallback`; 1427 s,
    19% over the 20-min cap). Successor-endpoint numbers valid; completed-law
    numbers read at the last token — **secondary only, invalid for the lock**
    (Tier-3 audit #5).
  - `analysis_ext.json` — extension, pairs L4→5 / L12→13 / L20→21 (ledger
    `nlm007_ext_predeclared`, `nlm007_ext_v1`; 1100 s). Same validity split:
    successor valid, completed-law secondary/invalid for the lock.
  - `analysis_slot.json` — **canonical slot-endpoint result**: corrected rerun
    over all six pairs, 20 shuffles, 500 boot, seed 13007 (ledger
    `nlm007_slot_v1`; 2145 s of a 3300 s budget; reload check unchanged).
    Exploratory at the reduced 20/500 budget; its L8/L12 qualification is
    withdrawn below.
  - `analysis_basesmoke.json` — moot-maker smoke at L8→L9 only, 2 shuffles /
    20 boot, point estimates (ledger `nlm007_baselines_smoke_L8`; 796 s).
    Pipeline validation; superseded by `analysis_base.json`.
  - `analysis_base.json` — predeclared six-pair moot-maker run
    (identity-plus-residual and per-carrier affine; ledger
    `nlm007_baselines_v1`). Took 4540.8 s against the predeclared 3300 s
    budget: **budget-incomplete exploratory artifact** — measured values
    retained, no planned full-budget gate earned; the null-making withdrawal
    still applies (Round 18, audit #7).
  - `identity_check.json` — stored-true-successor identity test of the slot
    completion at every pair and carrier (ledger `nlm007_identity_check_v1`;
    audit #6 action 3). **Valid**: routing validated to measured precision
    (per-pair max KL 1.9e-6 to 6.2e-6 over 16 × 80 cells); no per-carrier
    error profile or fresh-float32 comparison was stored.
  - `analysis_deltasmoke.json` — `--target delta` pipeline smoke at L8→L9
    (1 shuffle / 10 boot; ledger `nlm007_delta_smoke_L8`). **Not a result.**
  - `analysis_delta.json` — **valid, adjudicated (Round 19, audit #8)**:
    five-pair displacement ladder (ledger `nlm007_delta_predeclared`,
    `nlm007_delta_v1`; 1750.3 s of the 5700 s wall; support 1.0). Reading
    below.
  - `forward_manifest_A.json` / `forward_manifest_B.json` — forward-time
    captures, sentinel A = '.' and B = ',' (ledger
    `nlm007_forward_predeclared`, `nlm007_forward_locality_control`);
    `forward_states_A/B.npz` git-ignored. Locality control passes under the
    Round 20 corrected clause (ledger `nlm007_forward_locality_ruling`);
    A/B unappended q-states and laws identical bit-exactly (ledger
    `nlm007_forward_AB_equality`).
  - `analysis_fwdsmoke.json` — `--source forward` pipeline smoke at F8, A
    (1 shuffle / 10 boot; ledger `nlm007_forward_smoke_F8A`). **Not a
    result.**
  - `analysis_fwdA.json` — forward-time move, sentinel A = '.', layers
    0/4/8/12/20, 20 shuffles / 500 boot (ledger `nlm007_forward_fwdA`;
    2220 s). **Valid; adjudicated Round 20 + audit #9**: the primary arm
    did not meet the preregistered two-layer same-sentinel criterion (only
    `F20` qualifies) — a nonpass under the historical contract, not a kill.
    Oracle field meaningless (ledger `nlm007_oracle_defect_forward`).
  - `analysis_fwdB.json` — sentinel B = ',' arm (registered as the
    control/replication arm; a correlated same-population check), same
    settings (ledger `nlm007_forward_fwdB`; 1823 s). **Valid; adjudicated
    Round 20**: `F12` and `F20` qualify (ridge); cannot rescue the period
    arm. Oracle field meaningless. Reading below.
  - `analysis_stylesmoke.json` — `--style-null` + KL-rank pipeline smoke at
    F8, A (2 shuffles / 10 boot; ledger `nlm007_stylenull_smoke_F8A`).
    **Not a result.**
  - `analysis_styleA.json` — within-style-family target null, sentinel A,
    layers 0/4/8/12/20, 20 shuffles / 500 boot (ledger
    `nlm007_stylenull_predeclared`, `nlm007_stylenull_styleA`; 2213 s;
    support 1.0). **Diagnostic only (audit #9)**: the null is an
    alignment-destruction diagnostic, not a clean style null; its KL-rank
    endpoint ranked K = 7 candidates instead of the preregistered 10 —
    labelled, **not contract-valid on that endpoint**. No claim. Oracle
    field meaningless.
  - `analysis_styleB.json` — sentinel B arm of the same control (ledger
    `nlm007_stylenull_styleB`; 2238 s; support 1.0): `F8/F12/F20` pass the
    historical style gate mechanically, `F4` misses, `F0` fails. **Diagnostic
    only**, same K = 7 label, no claim (Round 21). Oracle field meaningless.
  - `analysis_locoA.json` — within-family leave-one-carrier-out control,
    sentinel A, layers 0/4/8/12/20, 500 word-clustered boot (ledger
    `nlm007_loco_predeclared`, `nlm007_loco_locoA`; 2902 s of the 4500 s
    wall; support 1.0). **Scored under the Round 21 rule; adjudicated Round
    22.** Reading below (audit #10 wording). Oracle field meaningless.
    The LOCO smoke (`nlm007_loco_smoke_F8A`) crashed before writing a JSON;
    log numbers only.
  - `analysis_locoB.json` — sentinel B arm of the LOCO control (ledger
    `nlm007_loco_locoB`; 3091 s; support 1.0). **Scored; adjudicated Round
    22** (audit #11 precision). Reading below. Oracle field meaningless.
  - `analysis_locoeqA.json` — Round 22 equalized-baseline LOCO addendum,
    sentinel A (word-only one-hot ridge; shrunk word mean; ledger
    `nlm007_loco_addendum_predeclared`, `nlm007_loco_locoeqA`; 2911 s;
    support 1.0). **Defect-affected (audit #11; ledger
    `nlm007_locoeq_defect_inner_centre`): outer margins descriptive only**,
    inner-selection claim invalid. The equalized smoke
    (`nlm007_locoeq_smoke_F8A`) artifact was deleted; log numbers only.
  - `analysis_locoeqB.json` — sentinel B arm of the addendum (ledger
    `nlm007_loco_locoeqB`; 2977 s; support 1.0). **Defect-affected (audit
    #11): descriptive only**; `F12/F20` mechanical, `F4/F8` miss on
    skill/KL-rank lower bounds, `F0` fails.
  - `analysis_unseenA.json` / `analysis_unseenB.json` — Round 22 unseen-word
    runs, sentinel A = '.' (ledger `nlm007_unseen_unseenA`; 2239 s) and
    B = ',' (ledger `nlm007_unseen_unseenB`; 2256 s; predeclared
    `nlm007_unseen_predeclared`); support 1.0, eight block × word-fold keys.
    **Mechanical pass at `F4/F8/F12/F20`, `F0` fails; formal gate pending
    (audit #12)** — status "mechanical pass under the recorded reduction;
    formal gate pending a contract-correct bootstrap". Reading below.
  - `analysis_locoeq2A.json` — corrected equalized addendum, sentinel A
    (analyzer `d10fc66`: inner two-carrier centre; comparator frozen by
    calibration score; ledger `nlm007_loco_locoeq2A`; 3753 s of the 4500 s
    wall; support 1.0). **Contract-correct; adjudicated Round 25** (audit
    #13 wording). Reading below.
  - `analysis_locoeq2B.json` — sentinel B arm of the corrected addendum
    (ledger `nlm007_loco_locoeq2B`; 4196 s; support 1.0).
    **Contract-correct; Codex adjudication pending** (required before any
    combined A/B equalized reading). Reading below.
  - `analysis_residsmoke.json` — smoke of the audit #12 bootstrap repair,
    the stronger unseen-word lexical nulls, `--residualize static`, and the
    Round 24 raw four-null shadow arm (sentinel A, F8, 1 shuffle / 10 boot;
    ledger `nlm007_resid_smoke_F8A`, `nlm007_resid_shadow_smoke_F8A`;
    design `nlm007_residualization_predeclared`). **Not a result**; meets
    the Round 25 raw-shadow launch prerequisite at pipeline level.
  - `analysis_resSA.json` — residualization, sentinel A, `P_static`
    (Round 25 launch ruling, ledger `nlm007_residualization_budget_amended`:
    120-minute wall, K = 13, five layers, two unseen-word folds, 20 shuffles
    / 500 boot; ledger `nlm007_resid_resSA`; 4405.7 s; support 1.0).
    **Contract-valid for the primary residual-vs-null question; scored and
    adjudicated Round 26 as corrected by audit #14** (ledger
    `nlm007_audit14_adopted`). Pre-patch analyzer: retention reported only
    as "the predeclared robustness marker is mechanically met" (ledger
    `nlm007_retention_marker_defect`,
    `nlm007_retention_common_scale_predeclared`). Reading below.
  - `analysis_resAA.json` — A-augmented (`P_aug-score4`: `P_static` plus at
    most four scores obtained by projecting a leave-calibration-word-pool
    carrier mean of `X` into a basis learned from calibration carriers; the
    full carrier-mean vector is not appended — audit #15), same contract,
    common-scale retention field present (ledger `nlm007_resid_resAA`;
    4737.8 s of the 7200 s wall; support 1.0; adjudicated
    `nlm007_resid_resAA_adjudicated`, Round 27, as corrected by audit #15).
    **Outcome-clean but transductive within carrier; not unqualifiedly
    contract-valid** until the pre-result meaning of the lock's
    carrier-mean clause is resolved. Reading below.
  - `analysis_resSB.json` — B-static (`P_static`, sentinel ','), same
    contract, common-scale field present (ledger `nlm007_resid_resSB`;
    4598.4 s; support 1.0). **Contract-valid; adjudicated Round 28.**
    Reading below.
  - `analysis_resAB.json` — B-augmented (`P_aug-score4`, sentinel ','),
    same contract, common-scale field present (ledger `nlm007_resid_resAB`;
    5073.8 s of the 7200 s wall; support 1.0). Third launch: the first two
    launches were lost in the F8 grammar block (ledger
    `nlm007_resid_resAB_crash`, `nlm007_resid_resAB_crash2`; erratum
    `nlm007_erratum_resAB_crash_localization`: the first loss occurred while
    entering grammar_w0 with no traceback, only the second is localized to
    torch `linalg.svd` non-convergence on the fitted low-rank coefficient
    matrix at grammar_w1 — "the F8 grammar block", not "the same fold"); the
    third ran with a numpy float64 LAPACK SVD fallback. **Adjudicated Round
    32 (`f8a2c48`, ledger `nlm007_round32_labels`) with the labels
    amended-implementation and SVD-telemetry-incomplete**: F4–F20 pass the
    residual-vs-null gate, F0 fails; ridge-only cosine and skill margins are
    mechanically reportable, the K = 13 KL-rank endpoint and every low-rank
    interpretation are amendment-qualified until the SVD telemetry gate
    (per-fit provider/exception/shape/finite/spectrum/rank telemetry plus a
    float64 NumPy shadow-backend agreement check) passes Tier-1 numerical
    review. Reading below.
  - `analysis_resSA2.json` — patched A-static rerun (identical design to
    `resSA`, common-scale retention block; ledger
    `nlm007_resid_resSA2_predeclared`, result `nlm007_resid_resSA2`).
    **Complete (5825 s, committed analyzer): F4, F8, F12, F20 pass the
    residual-vs-strongest-null gate (block-first LBs cos >= 0.46, skill >=
    0.18, KL-rank >= 0.20; 7-8/8 keys; retention held on all three
    endpoints); F0 fails (2/8 full-gate keys, negative skill/KL-rank
    margins).** This fills the {A,B} x {P_static, P_aug-score4} table on one
    common scale. **Adjudicated (Evidence gate 2026-08-28, ledger
    `nlm007_fourcell_adjudication`; PASS, qualified):** strict full-gate
    keys 7/8, 7/8, 6/8, 8/8 (the "7-8/8 keys" above are jointly positive
    keys); minimum crossed block-first lower bounds cos 0.458, skill 0.175,
    K=13 KL-rank 0.197; all 48 F4–F20 layer × endpoint common-scale ratio
    medians exceed 0.5 (estimator/null competition ratios, not retained
    signal). Licensed wording is the audit #17 sentence in STATE.md
    (condition robustness within one decoder and population, not
    replication, state, or a native law). Provenance erratum: ledger row
    `nlm007_resid_resSA2` says sentinel `2`; the artifact records
    `sentinel_tag: A` (`nlm007_erratum_resSA2_sentinel_label`).
  - `analysis_ctxscr_A.json`, `analysis_ctxscr_B.json` — contextual-prefix
    X-free **point-only screens** (`--ctx-screen`, `token_ids_v1`, committed
    analyzer copy; ledger `nlm007_ctxprefix_ctxscr_A`,
    `nlm007_ctxprefix_ctxscr_B`). Not gate adjudications. Audit #18 wording:
    at F4–F20 the cell-state ridge exceeds the strongest `token_ids_v1`
    field by approximately 0.11–0.20 cosine and 0.11–0.20 normalized-error
    reduction in both sentinels, so the screen did not triage the
    X-conditioned hypothesis out; it does not establish that the
    state-reading gate is live (skill, continuous-KL, crossed intervals,
    joint key count, collapse checks, and a capacity-matched comparison are
    unscored). Contextual ridge effective df ~42.7 vs state ridge ~210–406
    across F4–F20. At F0 the contextual field nearly closes ridge
    direction (cosine gaps ~0.019/0.018) but not magnitude (normalized
    error ~1.00 vs 0.97) — not proof that "prefix IDs explain F0".
  - `analysis_ctx_A.json` (ledger `nlm007_ctxprefix_ctx_A`; 5783 s) and
    `analysis_ctx_B.json` (`nlm007_ctxprefix_ctx_B`; 4476 s) — unresidualized
    contextual-prefix completions (`--contextual-prefix-xfree`, 20 shuffles /
    500 boot; committed-analyzer copy): **complete, scored pending
    adjudication; audit #19 wording governs.** Each is a completed
    unresidualized, outer-held-out predictor comparison. At F4–F20 the
    higher-EDF cell-state ridge retained a positive held-out score difference
    from the registered `token_ids_v1` context-only pair on displacement
    cosine, normalized error, frozen completion skill and continuous KL — A:
    cosine +0.15 to +0.20 (LB ≥ 0.13), normalized error +0.14 to +0.20, skill
    +0.34 to +0.46 (LB ≥ 0.25), continuous KL +0.27 to +0.45 (LB ≥ 0.17); B:
    cosine +0.11 to +0.18 (LB ≥ 0.09), normalized error +0.11 to +0.16, skill
    +0.33 to +0.41, continuous KL +0.24 to +0.40 — with positive crossed lower
    bounds, all eight outer keys point-positive, no carrier family collapse,
    support 1.0. F0: cosine +0.019 (A) / +0.018 (B) while skill and
    continuous-KL lower bounds cross zero and the continuation family
    collapses — non-qualifying, model-class-sensitive. The state ridge has
    approximately 5–10 times the selected contextual ridge's effective
    degrees of freedom and a different feature class; inner tuning is
    calibration-only (no held-out-outcome double use); the endpoints are
    correlated functionals of one prediction. Licensed reading: a higher-EDF
    state predictor has a positive held-out score difference from this fixed
    context-only pair — a descriptive predictor comparison, not evidence that
    context failed, capacity is the sole confound, or operational state has
    been identified. Never "did not close the gap"; never "gate live".
    Audit #20 upholds this reading for `ctx_B`.
  - `analysis_ctxS_A.json` (ledger `nlm007_ctxprefix_ctxS_A`; 5655 s;
    `--residualize static`, 20 shuffles / 500 boot; committed-analyzer copy)
    — **complete, scored pending adjudication; audit #20 wording governs.**
    The corresponding sentinel-A comparison on the `P_static`-residualized
    relation. At F4–F20, the registered token-context ridge/kernel falls to
    held-out cosine approximately 0.04–0.07 and normalized error
    approximately 1.00, while the residual `X_perp` ridge has cosine
    approximately 0.56–0.62 and normalized error 0.78–0.83; the crossed
    cosine, normalized-error, skill, and continuous-KL margins are positive
    (cosine +0.51 to +0.58, LB ≥ 0.46; normalized error +0.17 to +0.23;
    skill +0.32 to +0.49; continuous KL +0.26 to +0.48; 8/8 keys), with
    support 1.0 and no family collapse. The contextual ridge is already at
    approximately 47 EDF in every F4–F20 key and the contextual kernel is
    approximately 48 EDF at F8–F20, so the collapse is not a
    low-selected-EDF artefact. F0: cosine margin positive, normalized-error,
    skill, and continuous-KL margins negative — model-class-sensitive
    diagnostic. `P_static` is a ten-column block/length/position nuisance
    design; `token_ids_v1` is a distinct approximately 205–222-column
    carrier/POS token-context design (at most 48 distinct training rows,
    omitting the item token and cell `X`); only `X` and `Delta` are
    residualized. Licensed reading: the registered context field's raw
    predictive signal is highly non-robust to `P_static` residualization
    and is therefore `P_static`-aligned within this fitted design; a
    higher-capacity predictor from `X_perp` carries held-out predictive
    information beyond the registered `P_static` nuisance projection and
    this fixed `token_ids_v1` context field. Never "largely by
    construction", "beyond template metadata", "presentation removed",
    "same feature space", a quantified presentation share, mediation, or
    causal attribution; not an identified state contribution, native law,
    or representation-level hostile hole.
  - `analysis_ctxS_B.json` (ledger `nlm007_ctxprefix_ctxS_B`; 4784 s;
    `--residualize static`, 20 shuffles / 500 boot; committed-analyzer copy)
    — **complete, scored pending adjudication; the same audit #20 wording
    governs.** Sentinel B mirrors sentinel A: at F4–F20 the registered
    `token_ids_v1` ridge/kernel falls to held-out cosine approximately
    0.04–0.08 and normalized error approximately 1.00, while the residual
    `X_perp` ridge keeps cosine approximately 0.52–0.58 and normalized error
    0.82–0.86; residual ridge minus the strongest contextual arm: cosine
    +0.46 to +0.51 (LB ≥ 0.42), normalized error +0.14 to +0.19, skill
    +0.34 to +0.45, continuous KL +0.24 to +0.41; 8/8 keys, support 1.0, no
    family collapse. F0: cosine margin positive, normalized-error, skill,
    and continuous-KL margins negative. Licensed reading (audit #20): raw
    context performance is highly non-robust to the registered `P_static`
    residualization and therefore `P_static`-aligned in this fitted
    design; not identified as presentation; not by construction; not a
    state contribution; the residual predictor separation is descriptive
    and unmatched in capacity. Same never-say list as `ctxS_A`.
  - **Program continuation ruling (2026-08-29; ledger
    `nlm007_program_continuation_ruling`, commit `6e74798`; supersedes the
    queue described in the bullets below):** NLM-007 is STOPPED as an
    open-ended program (infrastructure drift 6:1 by the constitution's
    tripwire) and closes via one terminal ladder — 34a raw (once) -> 34a
    static (once, separately; no cross-estimand pooling) -> 34b (only if both
    34a estimands CONTINUE and its final bounded repair is RUN-READY without
    scope expansion) -> 34c (only after a 34b CONTINUE). First
    STOP/MOOT/REDUNDANT or INCONCLUSIVE rung ends the ladder; all-CONTINUE
    records "the predictor separation survived these registered controls"
    and NLM-007 closes anyway. CUT: full six-arm Round 34; the Round 33
    consequence instrument (branch `conseq-instrument` archived as tag
    `archive/conseq-instrument-parked`, never run); the parity check as a
    gate; the random-weight architecture null; a second decoder. Round 35 is
    a requirements envelope only. Governance: mandatory producer/reducer
    separation.
  - `analysis_parity_head_A/B.json`, `analysis_parity_ref_A/B.json` —
    HEAD-vs-refactor CPU parity check (ledger `nlm007_parity_verdict`,
    commit `cbed1ee`; contextual-prefix static screens on the committed
    analyzer copy vs the parked branch's refactored analyzer; decision JSON
    scrubbed of timing/SVD/shadow fields): **IDENTICAL for A and B.** The
    gate is cut by the continuation ruling; kept as evidence only.
  - `analysis_ctxcapA_raw.json` / `analysis_ctxcapB_raw.json` /
    `analysis_ctxcap_raw_joint.json` (+ hash-bound sidecars
    `round34a_evidence_ctxcapA_raw.npz` / `round34a_evidence_ctxcapB_raw.npz`;
    frozen analyzer copy `analyze_r34a_frozen.py` = blob `6b93ff1`; 291 s /
    254 s, tokenizer only; ledger `nlm007_round34a_raw`, commit `60d06f7`)
    — **Round 34a RAW: CONTINUE at F4–F20 in both sentinels.** Strongest
    matched margin per layer (F4/F8/F12/F20): A cosine +0.072/+0.057/+0.045/
    +0.042 (crossed LBs 0.034/0.024/0.019/0.024), normalized error
    +0.073/+0.047/+0.040/+0.054; B cosine +0.082/+0.064/+0.047/+0.043 (LBs
    0.049/0.034/0.023/0.022), nerr +0.088/+0.054/+0.042/+0.067; 8/8 keys
    jointly positive at every F4–F20 layer; strongest contextual arm = the
    token-id kernel at most layers; F0 INCONCLUSIVE (matched EDF undefined;
    diagnostic only). Joint: COMPLETE/SCREEN-ONLY, CONTINUE, common layers
    F4/F8/F12/F20 (one reducer NaN-replay defect fixed on the main analyzer,
    producer untouched; ledger `nlm007_round34a_reducer_nan_fix`). Reading:
    capacity matching removed most of the unmatched raw gap (`ctx_A`/`ctx_B`
    cosine margins +0.11 to +0.20); a +0.04 to +0.08 separation survives
    (audit #21 withdrew "lower bounds just above the 0.02 threshold": the
    registered rule is point margin >= 0.02 with LB > 0; smallest raw point
    0.0397, smallest LB 0.0146) — a small-magnitude but systematic
    within-design survival, not a state claim.
  - `analysis_ctxcapA_static.json` / `analysis_ctxcapB_static.json` /
    `analysis_ctxcap_static_joint.json` (+ `round34a_evidence_*_static.npz`)
    — **Round 34a STATIC: CONTINUE at F4–F20 in both sentinels** (ledger
    `nlm007_round34a_static`, commit `850414c`; 314 s / 304 s; joint re-run
    on the main analyzer after the NaN-replay reducer fix: COMPLETE/
    SCREEN-ONLY, CONTINUE, common layers F4/F8/F12/F20). Strongest matched
    margins (F4/F8/F12/F20): A cosine +0.306/+0.383/+0.373/+0.435 (LBs
    0.227/0.315/0.305/0.353), nerr +0.047/+0.089/+0.084/+0.115; B cosine
    +0.329/+0.352/+0.337/+0.367 (LBs 0.262/0.278/0.275/0.264), nerr
    +0.065/+0.082/+0.077/+0.100; 8/8 keys; F0 INCONCLUSIVE (diagnostic).
    Selected state ridge 202–384 EDF; contextual ridge target ~47; kernel
    target ~48 at F8–F20 but ~4.36 in 4/8 A and 2/8 B F4 keys. Licensed
    wording (audit #21): the residual predictor separation was not
    eliminated by the registered EDF match within these fixed feature
    classes — never "not a capacity artefact"; not a state claim.
  - `analysis_ctxoverlap_A.json` (444 s) / `analysis_ctxoverlap_B.json`
    (595 s) / `analysis_ctxoverlap_joint.json` — **Round 34b (P/C
    partial-overlap screen, static estimand): INCONCLUSIVE in both
    sentinels and in the joint — the terminal rung** (ledger
    `nlm007_round34b_sentinels`, `nlm007_round34b_joint`; commits `b285945`,
    `21ecb3f`). `P+C − P` cosine A +0.0178..+0.0373, B +0.0238..+0.0355
    (A/F4 point just below 0.02 but its upper interval exceeds 0.02) — the
    redundancy STOP fails; residual-context (`C⊥→Δ⊥`) cosine ~+0.019..+0.089
    with residual normalized-error gain negative in every ridge/kernel,
    sentinel, F4–F20 cell — retention fails. Joint: COMPLETE/SCREEN-ONLY,
    INCONCLUSIVE, no common retaining layer, no common stop layer. The joint
    reducer's EDF<=rank bound was producer-inconsistent by ~3e-5 at excluded
    F0 fits only; repair = post-outcome but not outcome-selective (audit
    #22); producers, sidecars, and gate functions unchanged; no rerun. The
    positive `P+C − P` increments are evidence against the registered
    strict-redundancy account, not evidence for operational state. Under
    the continuation ruling an INCONCLUSIVE rung is an allocation stop:
    Round 34c (`itemctx_*`; implemented, never run) does not run and
    NLM-007 closes (audit #22).
  - **Round 36 — minimal operational-quotient world (the constructive
    program; design registered in `theory/EXPERIMENTS.md` at `c26eee4`,
    ledger `round36_design_registered`; run-ready `a383a45`):** runnable
    CPU-only latent transition system on the 16 four-bit states with
    toggle/swap/no-op, trained from behaviour only; identity = equality of
    future response signatures under allowed actions (bisimulation);
    falsifiers with registered thresholds (quotient well-definedness,
    involution, non-commutation table, held-out 2/3-step closure,
    interchangeability, cross-seed action-table invariance). One module
    `experiments/run_operational_quotient.py` (`produce` / `reduce` /
    `fixture`; retired `f6dac0e`), config `experiments/config/operational_quotient_v1.json` (retired);
    producer/reducer separated; fixture before any produce.
    **v1 first run — FAIL every gate** (commit `073037f`; ledger
    `round36_first_run`; adjudication `e69ac72`, `round36_adjudication1`;
    audit #23 `round36_audit23`). Artifacts
    `experiments/results/operational_quotient_v1/`: `config.json`,
    `manifest.json`, `evidence.json`, `verdict.json`, `weights.npz`.
    `produce` (seeds 11/23/37/53/71, CPU, one process) 52.6 s (train 41.7 s,
    evidence 11.0 s; wall 900 s); `reduce` FAIL on quotient availability,
    quotient well-definedness, toggle involution, swap/toggle table,
    held-out depth-2/3 closure, interchangeability, action-table truth
    (0/5 seeds; 14–56% of 176 cells at `0.10/0.90`), cross-seed whole-table
    gate. Behaviour: train 96.563–98.546%, held-out 97.009–98.259%, depth-3
    93.8–96.3%, loss still falling at step 4,000. Licensed reading = the
    audit #23 paragraph (verbatim in `STATE.md`): behavior-, calibration-,
    and exactness-confounded non-certification; no fit-but-non-congruent
    claim. DIAGNOSTIC only (confidence-free `p>0.5` replay, read-only): 148–
    174/176 truthful one-step action cells per seed (84.1–98.9%); cells
    identical across all five seeds 11/176 at the registered thresholds,
    112/176 at `p>0.5`; 175/176 bitwise-majority table; every exact gate
    still fails.
  - **Round 36b — behaviour-fit ladder (preregistered `f9dea33`;
    COMPLETE; audit #24):** four cells `S16` / `S64` / `LR64` / `W64`
    (16k/.003/w32; 64k/.003/32; 64k/.001/32; 64k/.003/64; configs
    `experiments/config/operational_quotient_36b_S16.json`, `_S64.json`, (retired `f6dac0e`)
    `_LR64.json`, `_W64.json`; walls 8/20/20/30 min; every cell run and
    visible; no pooling or best-cell). Locks before any outcome: V1
    `f95ff01` (`round36b_lock`) -> review #1 NOT-READY `70b58a7`
    (`round36b_review1`: eligibility from producer aggregates) -> audit #23
    amendment `9edb892` (three-stage status tree, DIAGNOSTIC `p>0.5` table,
    cellwise cross-seed accounting, depth traces) -> row-level logit replay,
    lock V3 `ff8eaa7` (`round36b_lock_v3`) -> review #2 RUN-READY
    (`round36b_run_ready`); runner + configs `61e2430`. Run (ledger
    `round36b_ladder`; `abef6cf`; walls 174/606/618/696 s): every cell
    `FAIL — BEHAVIOR UNDERFIT; QUOTIENT INELIGIBLE`. Fit (train / 21,184;
    held-out / 2,240; five seeds): S16 20,894–21,184 (1 exact) / 2,179–
    2,218; S64 21,078–21,184 (2 exact) / 2,184–2,226; LR64 21,088–21,184
    (4 exact) / 2,198–2,225; W64 21,184 on all five seeds / 2,216–2,239
    (98.9–99.96%, none exact; all misses H3, none common to all seeds).
    Artifacts `experiments/results/operational_quotient_36b_*/`:
    `config.json`, `manifest.json`, `verdict.json` committed;
    `evidence.json` (165–177 MB per cell) and `weights.npz` git-ignored
    (`.gitignore`: remote size limit), retained locally, sha256-pinned in
    manifest/verdict/ledger. Licensed reading = audit #24 (verbatim in
    `STATE.md`): eligibility not reached; reachability of the exact learned
    precondition unvalidated, not unsatisfiable. INFORMATIONAL only (W64,
    `p>0.5`): all 16 encoder identities and the truthful 176/176 canonical
    action table identical across five seeds; well-definedness 71–94%,
    involution 46–84%, H2 closure 98.6–99.9%, H3 closure 61–92%,
    interchangeability 39–77% — a cross-seed-stable canonical one-step
    skeleton, not a certified quotient/action algebra. A prospectively
    locked, post-outcome, outcome-informed successor: exploratory, not
    confirmatory; it cannot rescue or overturn v1.
  - **Round 36c — quotient-trained positive control (COMPLETE; both
    cells FAIL; audit #25):** same carrier/seeds/representatives as 36b
    with explicit transition supervision (BCE + `1.0 *` MSE to the
    stop-gradient true-successor encoding over the 176 canonical
    transitions), `result_scope = POSITIVE-CONTROL`, unchanged exact
    reducer. Locks before any outcome: `round36c_registered_locked` ->
    review #1 NOT-READY / lock V2 (`round36c_review1_lock_v2`) -> review #2
    RUN-READY (`round36c_run_ready`); runner + configs `dd699e2`. w32
    (produce 819.8 s / 1800 s wall): FAIL on every exact gate in every
    seed, action-table truth 0/5, cross-seed table FAIL. w64 (conditional
    cell, precondition met; produce 987.6 s / 2400 s wall): FAIL —
    swap/toggle 4/5, depth-2 closure 3/5, all else 0/5. Artifacts
    `experiments/results/operational_quotient_36c_w32/` and `_w64/`:
    `config.json`, `manifest.json`, `verdict.json` committed;
    `evidence.json` and `weights.npz` git-ignored, retained locally,
    sha256-pinned. Adjudication (`round36c_w32_adjudication`): the
    combined loss trace deteriorates in seeds 11/53/71 after early minima,
    plateaus in 37, near-converges only in 23; paired regression vs
    width-matched behaviour-only S64 train counts localizes the regression
    to the added joint objective. Licensed reading = audit #25 (verbatim in
    `STATE.md`); the FAIL registers as "this reachability control did not
    reach the certificate", nothing stronger. Next: Round 36d (frozen-chart
    transition control, one capped cell), then Round 37 (presentation-
    duplicated 32->16 quotient world) — both registering, neither run.
  - Round 34 capacity-matched state-versus-context audit (registered
    `d493cf2`, `theory/EXPERIMENTS.md`; implemented on the main analyzer as
    `--context-capacity-audit round34_v1`, commit `9eb1301`; ledger
    `nlm007_round34_registered`, `_impl_review1..3`,
    `nlm007_round34_producer_run_ready`): producer path RUN-READY; the joint
    claiming reducer is flagged for one further review (repair rounds on it:
    3, cap reached). **Full six-arm run CUT by the continuation ruling
    (never run)**; previously HELD (audit #19, ledger
    `nlm007_audit19`): its primary estimand is `P_static`-residualized and
    cannot retroactively capacity-match `ctx_A`/`ctx_B`; its K=13 KL-rank
    endpoint must become diagnostic (raw continuous KL confirmatory) or the
    parked SVD gate must reopen before any outcome. **Round 34a — matched-EDF
    core screen** (Codex preregistration `f97a533`; tags `ctxcapA_raw` /
    `ctxcapB_raw` for the unresidualized estimand, `ctxcapA_static` /
    `ctxcapB_static` separately; token ridge/kernel only, state ridge
    bisected to the selected contextual EDF and to the 47/48 rank ceiling,
    same outer folds, cosine + normalized error with paired block-first
    crossed intervals, no completion): **RUN-READY at `6b93ff1`** after four
    Tier-1 rounds (ledger `nlm007_round34a_registered_implemented`,
    `_review1..3`, `_fix1`, `nlm007_round34a_run_ready`; sentinel artifacts
    non-claiming, joint reducer read-only, hash-bound per-cell evidence
    sidecars `round34a_evidence_<tag>.npz`). Audit #20: raw and static are
    separate required screens — raw stages the `ctx_A`/`ctx_B` comparison,
    static stages the `ctxS` and future consequence estimand; neither
    substitutes for the other and no cross-estimand verdict is permitted.
    Six runs: `ctxcapA_raw`, `ctxcapB_raw`, `ctxcap_raw_joint` (DONE,
    CONTINUE — see above), `ctxcapA_static`, `ctxcapB_static`,
    `ctxcap_static_joint` (RUNNING). Under the continuation ruling a
    surviving margin does not reopen full Round 34, Round 33, or a completion
    comparison; it only licenses the conditional 34b rung.
  - **Round 34b / 34c (audit #20; preregistered by Codex in
    `theory/EXPERIMENTS.md`, `3b49321` / `ff69d82`, ledger
    `nlm007_round34bc_registered`):** before interpreting the static
    collapse as contextual redundancy and before the full Round 34, a cheap
    same-fold `P_static`/context partial-overlap screen (`P`, `C`, `P+C`,
    `C_perp -> Delta_perp`, same-EDF `X_perp` reference, context-to-`P_static`
    alignment; 34b, tags `ctxoverlap_A` / `ctxoverlap_B` / `ctxoverlap_joint`)
    and an item-embedding-by-`P_static` X-free comparator (`P_static` + 16
    calibration-only item-embedding PCs + 160 interactions + boundary/POS
    floor; 34c, tags `itemctx_A` / `itemctx_B` / `itemctx_joint`).
    **Implemented on the main analyzer (`round34b_overlap_analysis`,
    `round34c_itemctx_analysis`; ledger `nlm007_round34bc_implemented`),
    UNRUN, under Tier-1 repair:** review #1 NOT-READY on four items —
    leakage provenance validated by counts rather than exact fold
    identities, clamped global `cos_rows` admitting undefined-cosine cells,
    NaN able to win inner selections with feature-dimension telemetry not
    locked, walls checked only per outer key (ledger
    `nlm007_round34bc_review1`; repair round 1 of 3; fix pass 2 + re-review
    #3 = repair round 3 of 3 in flight, ledger `nlm007_round34bc_fix2`).
    Under the continuation ruling both are CONDITIONAL rungs of the terminal
    ladder: 34b only if both 34a estimands CONTINUE and this final repair is
    RUN-READY without scope expansion; 34c only after a 34b CONTINUE.
    Full Round 34 and Round 33 are cut.
  - **Round 35 — typed truth-evaluable world (docs-only design,
    `c74bfab`; ledger `nlm007_round35_design_registered`):** four-bit finite
    world (toggle/swap/no-op), population and linguistic-adversary
    contracts, frozen forced-choice yes/no log-odds, wrapper and same-length
    controls, causal patches, inherited X-free ladder, involution and one
    non-commuting composition, CPU-only budget. Authors nothing. Under the
    continuation ruling it is a **requirements envelope only** (right
    direction, wrong first artifact); the constructive program is Round 36.
  - SVD telemetry / shadow-backend gate — **parked and unpassed** after
    re-review #4 (ledger `nlm007_svd_telemetry_review4`): a discretionary
    allocation decision under the global CLAUDE.md §2.7 repair-round cap,
    not an AGENTS.md rule (audit #18). Every low-rank / K=13 KL-rank claim
    keeps its amendment qualification; the analyzer diff stays uncommitted.
  - Round 33 multi-position consequence instrument (runner
    `capture_forward_consequence`, analyzer `--source forward_consequence`;
    ledger `nlm007_consequence_impl_pending_review`, `_review2`, `_review3`):
    **implemented, unrun, PARKED and unpassed** (ledger
    `nlm007_consequence_instrument_parked`): re-review #3 found the reducer
    did not enforce six keys jointly positive across both horizons (closed);
    re-review #4 NOT-READY on base-compat schema mirror / preflight,
    serialized-fit fingerprints, wall rechecks, and a real CPU parity run —
    the fourth consecutive repair round, so the global CLAUDE.md §2.7 repair
    cap was applied as an allocation decision (instrument lives on branch
    `conseq-instrument`; main analyzer stays at HEAD; raised to the user;
    parked, not killed). No consequence run is authorized; even after repair
    a pass licenses only persistence of downstream predictive accuracy under
    frozen tails (audit #18). Audit #19 upheld the parking as allocation, not
    a kill: review #4 still found a real legacy-manifest crash, unproved
    exact-fit reuse, incomplete preflight/binding, wall gaps, and missing CPU
    parity; the branch is salvageable only if a later matched-capacity result
    makes the consequence test worth reopening. **CUT by the continuation
    ruling:** branch archived as tag `archive/conseq-instrument-parked`, not
    deleted; its legacy-parity question was answered (IDENTICAL) but no
    CONTINUE reopens it.
  - `analysis_xf{SA,SB,AA,AB}.json` — Round 27 comparator 2, the
    **registered X-free field** (`--xfree-field`: calibration-only
    residual-space field from `P_static` + the rank-4 carrier-summary scores
    + 16 frozen-embedding PCs + 64 interactions, no cell-level `X⊥`, with a
    df-matched state ridge; ledger `nlm007_xfree_field_predeclared`,
    `nlm007_xfree_comparator_implemented`, frozen at analyzer `cddcd47`
    with the literal command in `nlm007_xfree_comparator_frozen`).
    Four cells, 7200 s each; "registered", not "fair" (audit #15: the fixed
    rank-4/full-prefix omissions remain substantive). **Disarmed** (ledger
    `nlm007_xfree_chain_disarmed`): the armed chain was killed under the
    Round 29 order and sits behind the external-axis probes. No artifact.
  - `analysis_fl{SA,SB,AA,AB}.json` — Round 27 comparator 1, the fully
    refitted Freedman–Lane residual-geometry null (`--fl-null 20`:
    layer-level exact test, common cell mask, ridge-only inner grid; ledger
    `nlm007_freedman_lane_predeclared`,
    `nlm007_flnull_comparator_implemented`). **Ready, not armed**: Round 29
    (adopting audit #15) limits it to one conditional A-static cell, run
    only if the external-axis probes leave the state reading live. No
    artifact.
  - `experiments/config/lexical_probe_fresh_v1.json` — Round 29 probe-2
    population (ledger `nlm007_fresh_population_frozen`): four families
    question / instruction / comparison / enumeration, 8 matched
    presentation pairs + 4 operational control pairs, same 80 words; ` not`
    (id 537) is a single token on every prefix. Prospectively authored and
    committed before any new capture or score, not independently blind to
    prior results; the declared digest `c6edaa92…` is not the raw file
    SHA-256 (`12c72401…`). **Design-void for confirmatory probes 2–4
    (audit #16)**: no pair establishes presentation-only equivalence across
    all four word classes, and several change syntactic licensing, modality,
    definiteness, degree, or quantification. Retained unchanged as an
    exploratory mixed-frame stress set only; no post hoc subset rescue is
    confirmatory. No capture. Successors: v2 and v3 (voided), v4 (frozen)
    below.
  - `experiments/config/lexical_probe_fresh_v2.json` — Round 31 population
    (`79c8628`; Codex as outcome-blind author; `Please/Kindly |
    For reference,/For clarity, … plan to {repeat|omit|capitalize|reverse}
    the word <X>`; tokenization pre-check passed). **Voided** by the
    independent linguistic adversary (ledger `nlm007_fresh_v2_voided`,
    erratum `nlm007_erratum_v2_void_count`): all 16 pair-2 cells fail —
    "For reference" vs "For clarity" introduce distinguishable discourse
    purposes that can scope over the operation; the 16 pair-1 cells and the
    16 control cells pass. Retained unchanged as a pragmatic-purpose stress
    set (audit #17: not a dead file). No capture.
  - `experiments/config/lexical_probe_fresh_v3.json` — Round 31 population
    (`a8b14a8`, fresh session; Please/Kindly; ASCII vs typographic
    apostrophe). **Voided** on control edit-magnitude (ledger
    `nlm007_fresh_v3_voided`): all 32 presentation pair cells passed
    (apostrophe rated near-degenerate), but the 8 controls under the
    orthographic wrapper fail clause 6 (a whole-word operation swap vs a
    one-glyph presentation edit) — a control-design failure only (audit
    #17). Retained unchanged, descriptively. No capture.
  - `experiments/config/lexical_probe_fresh_v4.json` — Round 31 population
    (`afd6fcc`, fresh outcome-blind author): `{Please|Kindly} plan to OP the
    word <X>` and `{Hello,|Hi,} please plan to OP the word <X>`, OP ∈
    {repeat, omit, capitalize, reverse}; aligned surface-word edit distance
    = 1 for every pair and control; frozen `operation_updates` block.
    **Approved 48/48 by a separate fresh adversary session (outcome-blind
    procedural approval: grammaticality, preservation of the explicit
    string-edit instruction, matched surface-word distance under the common
    mention frame — not 48 independent linguistic observations); tokenization
    pass; frozen** (`3a70890`, ledger `nlm007_fresh_v4_frozen`): raw sha256
    `f813f9b2cb96546726412b55857e79324ac23b47a2cb6418f8569ce47bbc5d33`, git
    blob `8845f75c89c27d8db9c5f5cc8a11cfd109b4756b`; captures must pass
    `--expected-config-sha256`; any edit voids the approval. The config's
    top-level "not approved for capture" note is historical authoring-time
    text superseded by the structured approval/hash fields (erratum
    `nlm007_erratum_v4_config_note`). Audit #17: the approval licenses a
    bounded mentioned-string instruction micro-world (every item in the
    autonymic `the word <X>` frame), not presentation inertness across
    ordinary noun, verb, adjective, and function-word uses; v4 sentinel
    results are a fresh-population stress test of the same append
    construction, never pooled with `lm_dyn_v1`. No capture; chain
    `run_v4.cmd` not armed (requires the operation-update and bridge code to
    pass Tier-1 review).
  - Contextual-prefix X-free baseline (`eab0a68`, Round 31; analyzer
    `--contextual-prefix-xfree` / `--ctx-screen`, token_ids_v1, point-only
    screen; ledger `nlm007_ctxprefix_implemented`): screens scored
    (`ctxscr_A/B` above); completions running/queued.
    Operation-verb update capture stage (`capture_op_update`) and the
    no-model acceptance fixture `op_update_fixture.py` committed
    (`d9a6cca`); the analyzer side (`--source op_update`) and the
    bridge-ladder patch: uncommitted, under Tier-1 review. Audit #17: the operation-verb
    update is a declared-operation-verb context intervention, not yet a
    denizen-enacted operational move (source and recipient are separate
    prefix encodings; no execution consequence is measured).
  - `run_r31.cmd` (probe-1 screens, `P_aug-full` cell A, contextual-prefix
    screens and completions on `lm_dyn_v1`): **disarmed** (ledger
    `nlm007_r31_chain_disarmed_pending_svd_gate`) — Round 32 forbids further
    low-rank output before the SVD telemetry / shadow-backend gate passes
    Tier-1 numerical review. No artifact.
  - Capture extension (`4137258`, Round 30; no artifact): `run_lm_dynamics.py`
    gains a config provenance guard (`--expected-config-sha256`), the ` not`
    operator-insertion capture, repeat-noise arrays, and population-void
    hard controls. RUN-READY; nothing captured under it yet.
  - `analysis_unseensmoke.json` — `--unseen-words 2` pipeline smoke at F8, A
    (1 shuffle / 10 boot; ledger `nlm007_unseen_smoke_F8A`, overwritten by
    `nlm007_unseen_smoke2_F8A` with the audit #10 lexical nulls and the
    K = 11 rank universe). **Not a result**; the full run awaits Codex
    predeclaration.
- **Successor endpoint (valid in all runs).** L0→L1: word-mean = ridge =
  kernel = 0.949, shuffled null 0.95 — lexical persistence, no law beyond
  word identity. From L4 on, full-dimensional ridge beats word-mean and the
  best static chart at every depth (ridge/chart/word-mean: L4 0.927/0.884/
  0.886; L8 0.941/0.860/0.861; L12 0.977/0.898/0.888; L20 0.965/0.901/0.897;
  L27 0.976/0.883/0.864, the last on normed vectors). Shuffle penalty grows
  with depth.
- **Slot-endpoint gate reading (Round 17, superseded at L8/L12 by Round 18).**
  On `analysis_slot.json` the pairs L8→L9, L12→L13, L27→L28 cleared every
  locked gate mechanically (support 1.0); L4→L5 and L20→L21 cleared both slot
  readouts and the word-mean gate but missed the all-fold +0.05
  successor-cosine lead (a stricter convention than the original lock, audit
  #6); L0→L1 fails every lead gate. Word-mean slot skill decays with depth
  (0.95, 0.84, 0.78, 0.70, 0.43, 0.40) while ridge holds 0.92–0.98 and the
  chart collapses late (0.50, 0.51). Round 16 scorecard: five of six
  predictions held; the L27→L28 attenuation prediction failed.
- **Withdrawal at L8→L9 and L12→L13 (Round 18 + audit #7; ledger
  `nlm007_baselines_v1`).** Pooled ridge − identres on successor cosine /
  slot skill / slot ordering: L8 −0.008/−0.021/−0.020; L12 −0.007/−0.009/
  −0.013 (only slot skill and ordering are completed-law slot metrics). On
  shared words and held-out carrier blocks, identity-plus-shared-displacement
  is at least as good as full ridge within a post-hoc one-sided 0.02 pooled
  margin on the three recorded comparison metrics at L8→L9 and L12→L13; the
  finite-ladder ridge wording is withdrawn as a conservative policy. The
  intervals support "no demonstrated positive ridge advantage under this
  margin", not "no lead" or equivalence. The measured relation is consistent
  with identity plus a calibration-mean displacement under this design; the
  experiment does not determine whether the displacement is carrier-, state-,
  or word-dependent. The Round 17 two-pair criterion does not survive as a
  claim. Identity-plus-shared-displacement does not meet the chosen margin at
  L0 (+0.46), L4 (+0.033/+0.019/+0.022), L20 (+0.018/+0.034/+0.032), or L27;
  L4 and L20 remain non-qualifying but live, while L27 is not a valid
  raw-residual persistence comparison. Per-carrier affine is far below the
  cross-carrier field everywhere (within-carrier diagnostic only).
- **Displacement ladder (Round 19 + audit #8; `analysis_delta.json`).**
  Only `L20->L21` passes the predeclared three-endpoint gate (kernel;
  positive clustered lower bounds on displacement cosine, slot skill, slot
  ordering) — retained as one bounded qualifying pair under the registered
  displacement-and-slot-law gate. `L0` is lexical persistence. `L4` has a
  small live remainder but fails the gate. `L8/L12` separate strongly from
  the word-conditioned displacement mean on displacement coordinates, with
  kernel minimal among the tested ladder, but slot-ordering leads are only
  0.003–0.022 and slot-skill lower bounds are mixed — the gate fails. Adopted
  wording: held-out-carrier evidence for predictable displacement variation
  beyond a word-conditioned mean, with a kernel as the minimal tested
  predictor; carrier/template versus state dependence remains unresolved. The
  carrier shuffle is a carrier-alignment diagnostic, not a state-independence
  null (shuffled field reported for ridge/low-rank only). "The slot law
  barely registers it" is a readout fact, not a world fact.
- **Forward-time move (Round 20 + audit #9; `analysis_fwdA.json`,
  `analysis_fwdB.json`).** Sentinel '.': `F0` token-identity dominated
  (shared mean = word-conditioned mean = 0.67 ≈ field 0.69). `F4/F8/F12`:
  displacement cosine ridge/kernel 0.71–0.78 vs word-conditioned mean
  0.48–0.53; law skill at the sentinel position 0.39–0.57 vs 0.01–0.02;
  carrier-shuffled field 0.12–0.32 vs 0.67–0.81; but ordering leads
  0.00–0.08 with lower bounds ≤ 0 in half the folds — three-endpoint gate
  fails. `F20` qualifies (ridge: +0.16–0.23 / +0.50–0.61 / +0.020–0.058,
  all LBs > 0). Sentinel ',': same shape; `F12` and `F20` qualify (ridge),
  `F8` misses one ordering LB by −0.002, `F4` one skill LB. Token-identity
  control: the '.'-fitted predictor on the ',' target scores 0.43–0.54 vs
  0.26–0.30 for the shared mean. Adopted wording: the period sentinel did
  not meet the preregistered two-layer, three-endpoint qualification
  criterion — a nonpass under the historical contract, not a kill of forward
  transport; in the shared-word, held-out-carrier design, sentinel
  displacement is predictably improved over the word-conditioned mean from
  F4 onward and the response law registers that variation in cosine and
  skill; the ordering endpoint was later diagnosed as insensitive/saturated,
  so the qualification failure is not a substantive null result. The comma
  arm falsifies "token identity or position prevents any qualifying layer".
  Carrier/template presentation versus state dependence remains unresolved
  (audit #8). Ordering is replaced prospectively by KL-to-truth candidate
  rank (K = 10); no existing run is reclassified.
- **Within-style-family null, sentinel '.' (`analysis_styleA.json`;
  diagnostic only).** Mechanically `F4/F8/F20` beat both the word-conditioned
  mean and the null on cosine, skill, KL-rank (ridge KL-rank 0.82–0.90 vs
  word-mean 0.31–0.41); `F12` misses one fold's KL-rank LB; `F0` fails. The
  null collapses below the shared mean (0.16–0.50 vs 0.47–0.62). Audit #9:
  a field refit on a broken carrier pairing predicts the wrong carrier's
  displacement, so "beats the within-style null" is not informative evidence
  for a state-linked component; "style-robust" is withdrawn as a claim. The
  KL-rank endpoint here ranked K = 7 (kNN-1/5/20 omitted; fixed in
  `269e46c`) and is not contract-valid. Sentinel ',' (`analysis_styleB.json`)
  has the same shape: `F8/F12/F20` mechanical, same label, same verdict.
- **LOCO control, sentinel '.' (`analysis_locoA.json`; Round 21 rule;
  adjudicated Round 22).** Pooled ridge − per-word block mean: `F4`
  +0.126 / +0.313 / +0.300, `F8` +0.118 / +0.232 / +0.292 (cosine / law
  skill / K = 4 KL-rank), `F12` and `F20` in the same range, all lower
  bounds > 0.08, 11–15 of 16 held-out carriers passing all three; `F0`
  no pass (block mean ≥ ridge). Audit #10 wording: on already-seen words,
  within a style family, X predicts a held-out carrier's displacement and
  response-law consequence better than the three-carrier per-word family
  mean at F4–F20 — not a presentation-independent state or a native law.
  The baseline is variance-disadvantaged; equalized X-free lexical baselines
  (word-only ridge, shrunk word mean) are required before interpretation;
  LOCO does not separate state from a smooth carrier/style code; the pooled
  16-carrier bootstrap is secondary. `F0` = "no detected conditional gain".
- **LOCO control, sentinel ',' (`analysis_locoB.json`; Round 22, audit
  #11).** `F12/F20` pass (pooled ridge − block-word mean: cosine +0.07–0.10,
  skill +0.15–0.20, KL-rank +0.20–0.26, lower bounds > 0; 12–13 of 16
  carriers); `F4` misses skill and KL-rank; `F8` misses skill only (KL-rank
  LB +0.021); `F0` fails. Run-level positive (2/5); weaker in breadth than
  A — a sentinel-specific instrument result, not evidence that B carries
  less state information. Same audit #10 wording as A.
- **Equalized LOCO addendum, sentinel '.' (`analysis_locoeqA.json`;
  defect-affected, audit #11).** All 80 folds selected maximal shrinkage
  (`lam_wordonly = 100`, `alpha_shrunk = 1`), i.e. the equalized baselines
  equal the shared mean; ridge − equalized baseline at `F4–F20`: cosine
  +0.09–0.13, skill +0.23–0.30, KL-rank +0.26–0.34 (11–14/16 carriers);
  `F0` negative. **Descriptive only:** the inner centre included the
  validation carrier and the comparator was chosen on held-out outcomes, so
  "the data selected maximal shrinkage" is invalid as implemented; whether
  maximal shrinkage persists under the corrected centre is unknown. Withdrawn
  (audit #11): "no per-word lexical signal", "variance objection answered",
  "context not content", "the state-conditioned component is large". Adopted
  wording: the word-conditioned component captured by the tested estimators
  is negligible for the measured forward displacement in this design; the
  positive object is X-conditioned residual predictability.
- **Corrected equalized LOCO addendum (Round 25 + audit #13;
  `analysis_locoeq2A.json`, `analysis_locoeq2B.json`).** With the inner
  two-carrier centre and the comparator frozen by calibration score, the
  equalized baselines no longer collapse onto the shared mean: A's
  calibration-selected equalized comparator sits roughly 0.002–0.009 above
  it, B's 0.002–0.007. A: `F4/F8/F12/F20` pass against the equalized
  comparator (cos +0.09–0.13, skill +0.23–0.31, KL-rank +0.30–0.43, lower
  bounds > 0.08; 11–14/16 carriers), `F0` fails — Round 25: a valid
  mechanical positive for the bounded sentinel-A seen-word within-family
  diagnostic; audit #13: audit #11's inner-centre *defect* concern is
  resolved by the corrected sentinel-A data (not "audit #11 is resolved");
  the pooled equalized interval is secondary. B: `F12/F20` pass, `F4/F8`
  miss on skill/KL-rank lower bounds (cosine leads hold), `F0` fails;
  run-level positive (2/5); adjudication pending. Both arms agree with the
  defect-affected runs' numbers (baselines moved by ≤0.01, no verdict
  changed). Maximum wording (audit #13): "On already-seen words, within
  sentinel A's style-family design, the context-bearing X field predicts
  the held-out carrier's forward displacement and response-law consequence
  beyond the properly nested, calibration-selected X-free lexical
  comparator at F4–F20." Still withdrawn: "no per-word lexical signal",
  "context rather than content", "the state-conditioned component is
  large", any presentation-independent or native-law reading.
- **Unseen-word runs (Round 23 + audit #12; `analysis_unseenA.json`,
  `analysis_unseenB.json`).** Calibration and held-out word identities
  disjoint; the class-mean and frozen-input-embedding `wordonly_knn` nulls
  sit at the shared mean at every layer. Block-first pooled ridge − stronger
  X-free lexical null: A cos +0.14–0.19, skill +0.33–0.47, K = 11 KL-rank
  +0.35–0.57 (lower bounds > 0.12; full-gate keys 7/8, 7/8, 8/8, 8/8 at
  F4/F8/F12/F20); B cos +0.11–0.17, skill +0.31–0.41, KL-rank +0.31–0.52
  (lower bounds > 0.09; 5/8, 6/8, 8/8, 8/8). `F0`: A's continuation block
  collapses, B's cosine lead 0.018 is below the 0.02 point gate — audit #12
  wording "non-qualifying, with the continuation held-out block providing the
  strongest local failure pattern". Round 23 adjudication: the state-linked
  prediction is held only as X-conditioned residual predictability,
  generalizing across unseen word identities; the tested
  lexical-interpolation prediction fails; the presentation/style nuisance
  prediction remains live. Audit #12: the predeclared class-preserving word
  bootstrap was not implemented (words resampled without class strata and
  nested within blocks) and the lexical null family is weak, so the status
  is **mechanical pass under the recorded reduction; formal gate pending a
  contract-correct bootstrap** and stronger nulls (nested
  frozen-embedding→Δ ridge, nested embedding-conditioned kernel, k ladder).
  Adopted wording: "not exact held-out-word lookup and not the tested lexical
  interpolator" — never "not word lookup" or "not lexical" unqualified; "the
  tested lexical nulls fail", not "lexical content is absent"; the ~0.06
  seen→unseen drop is a point comparison at F8 only; positive object =
  X-conditioned residual predictability transferring across the held-out
  word fold and held-out block. Strongest alternative: `X` contains smooth
  lexical and presentation coordinates along which the later displacement
  varies; the coarse nulls collapse for coarseness, not because the
  variation is operational state.
- **Residualization A-static (Round 26 as corrected by audit #14;
  `analysis_resSA.json`).** Sentinel A, `P_static` (centred block one-hot,
  tokenized lengths, slot/sentinel positions) cross-fitted out of both `X`
  and `Δ`; two unseen-word folds; K = 13; crossed class-preserving
  bootstrap. `F4/F8/F12/F20` pass the residual-vs-null gate: `X⊥` ridge
  residual cosine 0.56–0.62 vs 0.06–0.07 for the strongest residualized
  X-free null; block-first margins cos +0.50–+0.56, skill +0.31–+0.48,
  KL-rank +0.40–+0.61 with positive lower bounds; full-gate keys 7/8, 7/8,
  6/8, 8/8 (misses family-localized in gloss/association; the four
  checkpoints are correlated measurements, not replications); no block
  collapse. `F0` fails (negative pooled skill; association-block collapse)
  — "no qualifying conditional gain at F0 under this instrument", a genuine
  negative control. Not a cosine-geometry mirage (audit #14): the ridge
  cosine falls from raw 0.65–0.76 to residual 0.56–0.62 while the nulls
  fall to ~0.06; shuffled q95 ≤ 0.13; residual normalized error 0.78–0.83.
  Registered-static-metadata arm (`P_static→Δ`): held-out cosine 0.43–0.63
  by layer — a cosine, never a variance share, fraction, or "explains 42%";
  not a pure presentation component (audit #16). Adopted
  ruling: "`P_static` took the non-collapse branch. Locally, the result
  establishes registered-presentation sensitivity and survival of X-linked
  residual predictability after cross-fitted removal of those registered
  coordinates. It identifies neither the surviving field as operational
  state nor the result as presentation-independent." Withdrawn (audit #14):
  Round 26's "much of the raw lead may have been presentation-mediated" —
  the overlap between presentation and the raw ridge lead is not
  identified. Retention: "the predeclared robustness marker is mechanically
  met" only (audit #13; pre-patch analyzer; patched rerun `resSA2` queued).
  The gate is too easy for a state claim: a fully refitted Freedman–Lane
  residual-geometry null and a flexible calibration-only presentation/
  lexical comparator are to be preregistered before any state reading.
  `P_static` pass plus `P_aug` collapse remains the predeclared "static
  coordinates incomplete, not state" branch.
- **Residualization A-augmented (Round 27 as corrected by audit #15;
  `analysis_resAA.json`).** Sentinel A, `P_aug-score4` cross-fitted out of
  both `X` and `Δ`; same folds, K = 13, bootstrap and raw-shadow arm as
  A-static; common-scale retention block present. All five correlated
  checkpoints meet the registered aggregate residual-vs-null gate. F4–F20:
  `X⊥` ridge residual cosine 0.56–0.62 vs 0.06–0.07 for the strongest
  residualized X-free null; block-first margins cos +0.50–+0.56, skill
  +0.35–+0.46, KL-rank +0.43–+0.56 with positive lower bounds; full-gate
  keys 8/8, 7/8, 6/8, 8/8; no block collapse. F0 is qualitatively weaker —
  residual cosine 0.34 vs −0.01, skill +0.16 [LB 0.02], KL-rank +0.30
  [0.12], only 2/8 keys clear the full per-key gate — and is not an
  independent confirmation of the F4–F20 profile; the raw F0 transition
  remains identity/token dominated (raw ridge exceeds the raw null by only
  ~0.019 cosine). `P_aug` carrier-summary nuisance arm (`P_aug → Δ`)
  0.45–0.64 cosine by layer; because its scores are derived from
  carrier-level `X`, this is not a presentation-only estimate or a variance
  share. Common-scale retention: the paired bootstrap median of the
  reassembled residual-model margin over the raw-ridge margin exceeds 0.5 in
  every layer-endpoint cell; 14 of 15 interval lower bounds exceed 0.5 (F4
  continuous KL 0.495), so no uniform 95% claim; ratios above one do not show
  strengthening. Reading (audit #15): A-static and A-aug are nested
  sensitivities on the same sentinel-A cells, not independent replications;
  the registered static and rank-4-score nuisance fits do not absorb the
  `X⊥–Δ⊥` association; broader presentation, carrier-geometry, and
  prefix-fingerprint explanations remain fully live. Neither result
  identifies operational state, presentation independence, fresh-style
  transfer, composition, or a native law. Strongest alternative missed by
  both queued comparators: a high-dimensional prefix/carrier fingerprint
  (aligned, cell-level, compatible with unseen-word transfer and law
  improvement).
- **Residualization B-static (Round 28; `analysis_resSB.json`).** Sentinel
  ',', `P_static`, same contract as A-static, common-scale field present.
  `F4/F8/F12/F20` pass the residual-vs-null gate: `X⊥` ridge residual
  cosine 0.52–0.58 vs 0.06–0.09 for the strongest residual null;
  block-first margins cos +0.45–+0.50, skill +0.35–+0.42, KL-rank
  +0.40–+0.58; 8/8 positive keys at every passing layer; no collapse. `F0`
  fails (cosine lead +0.27 but pooled skill negative — the −7.5 skill is
  driven by two association folds near −30, locally ill-conditioned
  normalization, not uniform failure — and KL-rank lower bound < 0; gloss and
  association collapse), as under A-static. Registered-static-metadata arm
  (`P_static→Δ`) 0.41–0.63 cosine by layer (a cosine, not a variance share
  or a pure presentation component).
  Common-scale retention passes at the bootstrap median for every F4–F20
  endpoint; F4 continuous-KL lower bound 0.426 blocks a uniform interval
  claim; joint static retention awaits `resSA2`; the ratios are
  robustness ratios, not retained signal, state, or mediation. Ruling
  (Round 28 as corrected by audit #16): across the correlated A/B static
  runs, registered block/length/position metadata predict raw
  displacement, and `X⊥` retains predictive association with `Δ⊥` beyond
  four X-free lexical nulls at F4–F20. This is a two-sentinel robustness
  result within one decoder and authored population — a correlated
  second-sentinel check, not independent replication, state, presentation
  independence, mediation, or a native law.
- **Residualization B-augmented (Round 32; `analysis_resAB.json`;
  amended-implementation, SVD-telemetry-incomplete).** Sentinel ',',
  `P_aug-score4`, same contract as A-augmented; third launch with the SVD
  fallback. `F4/F8/F12/F20` pass the residual-vs-null gate: `X⊥` ridge
  residual cosine 0.52–0.57 vs 0.06–0.09 for the strongest residual null;
  block-first margins cos +0.46–+0.51, skill +0.40–+0.44, KL-rank
  +0.45–+0.54; 6–8/8 full keys; no collapse. `F0` fails (cosine +0.33 but
  skill lower bound −0.04; 4/8 full keys; gloss collapse).
  Registered-static-metadata + carrier-summary nuisance arm (`P_aug→Δ`)
  0.42–0.64 by layer (not a presentation-only component). Same-run
  common-scale ratios exceed 0.5 at the median at F4–F20 (F0 wide). Only
  the ridge-only cosine and skill margins are mechanically reportable; the
  K = 13 KL-rank endpoint and every low-rank interpretation remain
  amendment-qualified (Round 32). Reading (audit #17 wording): the sentinel
  {A,B} × {P_static, P_aug-score4} table is complete only for the
  residual-versus-four-word-only-null mechanical gate: F4–F20 pass in all
  four correlated cells, while F0 fails except for a weak pooled A-score4
  association with only 2/8 full-gate keys. This is consistent
  within-decoder, within-population condition robustness, not replication.
- **Oracle defect (ledger `nlm007_oracle_defect_forward`).** The per-carrier
  oracle read the stored states directly; in forward and delta mode it
  predicted X from X, so the ~0.98 oracle values in `analysis_fwdA/B`,
  `analysis_styleA/B`, `analysis_locoA/B` are meaningless. Fixed
  prospectively; diagnostic only, no result changes.
- **What we learned.** Identity is the null for residual-stream transport.
  The present data support persistence plus a calibration-average
  displacement as a competitive finite-design description at L8 and L12,
  retain small unresolved remainders at L4 and L20, and do not yet establish
  a native or generally reusable affine law. The forward step is a bounded
  held-out-carrier displacement-forecasting result that does not yet
  distinguish a state-space regularity from a carrier/template-conditioned
  nuisance law. A permutation null that a flexible model trivially beats is
  not a control. Within one style family the state carries predictive
  variation beyond the family's per-word mean for seen words (LOCO A), which
  narrows but does not remove the carrier/template alternative; the
  positive object is X-conditioned residual predictability (audit #11). On
  words never seen in calibration the same object transfers across the
  held-out word fold and held-out block against the tested X-free lexical
  nulls — a mechanical pass whose formal gate awaits a contract-correct
  bootstrap and stronger lexical nulls (audit #12). The corrected equalized
  addendum removes the audit #11 defect without changing any verdict (Round
  25, audit #13). Residualization runs may state only that the predeclared
  robustness marker is mechanically met until the common-scale retention
  marker is scored (audit #13). After cross-fitted removal of the registered
  static presentation coordinates, `X⊥` still predicts `Δ⊥` and its
  reassembled consequence beyond the residual X-free lexical nulls at
  F4–F20 — registered-presentation sensitivity plus surviving X-linked
  residual predictability, with the presentation/raw-lead overlap
  unidentified (Round 26 as corrected by audit #14). The same survival holds
  for the second sentinel under `P_static` (Round 28) and, on the same
  sentinel-A cells, under the nested rank-4-score `P_aug` fit (Round 27 as
  corrected by audit #15) — correlated sensitivities, not replications; the
  registered nuisance fits do not absorb the association, and presentation,
  carrier-geometry, and prefix-fingerprint accounts remain live. Bounded to
  one decoder and one authored template population. The fourth cell
  (B-augmented, Round 32) completes the sentinel {A,B} × {P_static,
  P_aug-score4} table for the mechanical residual-vs-null gate only — F4–F20
  pass in all four correlated cells, F0 fails except for a weak A-score4
  association — and licenses within-decoder, within-population condition
  robustness, not replication; its low-rank content is amended-implementation
  and SVD-telemetry-incomplete until the telemetry gate passes (audit #17).
  Audit #17's allocation ruling (adopted Round 33; supersedes the Round 29
  and Round 31 orders): protected work (`resSA2`, the SVD telemetry gate) →
  the contextual-prefix X-free baseline → one bounded multi-position
  consequence test (next k ∈ {4, 8} tokens, teacher-forced) BEFORE the v4
  bridge/interchangeability probes and before a second decoder; do not
  author a v5 — the next population, when one is authored, is a typed
  use-frame task. Behind it, disarmed or conditional: the Round 31 chain
  (probe-1 screens, `P_aug-full` A cell, contextual-prefix screens), the
  registered X-free field ×4, Freedman–Lane on A-static only, the second
  pinned decoder. Further lessons: a numerical instrument that fails to
  converge is a finding about the instrument until diagnostics say
  otherwise (B-aug) — and every low-rank result now sits behind a telemetry
  gate; a template population must pass a predeclared linguistic contract
  before any capture, or the presentation axis it is meant to test is not
  defined (audit #16); the v1–v3 loop showed that an all-inventory
  ordinary-use presentation contract had not been achieved — v4 obtains
  grammatical core-operation equivalence by placing every item in the same
  autonymic `the word <X>` frame, which licenses a bounded mentioned-string
  instruction micro-world and nothing wider (audit #17); an instruction-verb
  edit without a measured execution consequence is a declared-operation-verb
  context intervention, not an operational move (audit #17).

## Round 12 closure — frozen-encoder program closed; pivot to worlds with dynamics (2026-08-27)

- Ledger `nlm006b_round12_adjudication`; dialogue `theory/dialogue/003.md`;
  commit `3294718`. NLM-006b corrected to non-diagnostic under its own
  label-preservation gate (below); frozen-encoder closeness/map work closes
  as scope management.
- **Residue (narrow, this encoder/dataset).** Training supplies a
  task-effective chart metric, affine-path smoothness, and graceful chart
  degradation under identity-destroying moves; no native construct tested
  (substitutability profiles, Fisher pullback, their transported variants)
  competes with it. Not a general claim about native constructs.
- Next program: causal-LM residual streams, where the forward pass is the
  world's own transport (NLM-007).

## NLM-006b — calibrated transport audit; chart survives, NON-DIAGNOSTIC under lock (2026-08-28)

- **Design.** Locked Round 11 (`nlm006b_prereg_transport_audit`): independent
  candidate strata (20 same-/20 cross-fine-label per anchor), transported-pair
  predictors F_T / R_T vs cosine_T / euclid_T on (T_e x, T_e y), true
  fine-label endpoint, label-preservation gate p_e ≥ 0.80, calibrated
  displacement gate. Ledger `nlm006b_v1`; artifact
  `experiments/results/nlm006b_v1/analysis.json`; transports
  `experiments/results/vision_cifar100_dinov2s_edits_v2/` (edits.npz
  git-ignored, sha256 9cc0e7c0…; displacement.json committed). 471 s, CPU.
- **Chart survives every displaced transport.** Support 400/400; displacement
  gate passes for crop50/invert/mix50/occlude50 (0.98–1.0 above control q95).
  TT chart lead over best native: crop50 +0.208, invert +0.227, occlude50
  +0.222, mix50 +0.090 (paired CIs exclude 0).
- **Non-diagnostic (Round 12).** Label preservation 0.19–0.46 for all four
  displaced families vs the 0.80 gate (controls hflip 0.77, shift 0.76): every
  family is OOD under the identity gate, so chart survival is descriptive
  only and no native/chart verdict is issued.
- **Order effect.** ST−TS cosine ≈ 0.035 for all displaced families (CIs
  exclude 0); ≈ 0 for hflip. Real, small, outside the invariance class only.

## NLM-006 v1 — transports outside the invariance class; EXPLORATORY (cosine-selected negatives) (2026-08-28)

- **Design.** Six transport families re-encoded by the frozen encoder
  (`experiments/results/vision_cifar100_dinov2s_edits_v2/edits.npz`, keyed
  `test_emb_<family>`: hflip, shift1px, crop50, invert, mix50, occlude50;
  `displacement.json` alongside), stratified candidates, true fine-label
  endpoint. Relabeled **exploratory** by Tier-3 audit #3 before results were
  read: hard negatives were cosine-selected, so the pool is adversarial to any
  chart-like ranking. Ledger `nlm006_v1_exploratory`; artifact
  `experiments/results/nlm006_v1/analysis.json`.
- **Uninterpretable for the primitive contest.** Every predictor scores below
  0.5 (cosine 0.411, Euclid 0.402, F 0.486, R_no_coarse 0.477) — cosine's
  "collapse" is manufactured by selecting negatives with the tested metric.
- **Exploratory signal.** Support 400/400 (stratification fixes NLM-005's
  support failure). Order sensitivity appears only outside the invariance
  class: ST−TS cosine 0.05–0.10 with CIs excluding 0 for crop50/invert/mix50/
  occlude50; 0.00 for hflip/shift1px. Displacement mean cos: hflip 0.96,
  shift 0.98 vs crop 0.63, invert 0.49, mix 0.43, occlude 0.66.
- **Next.** NLM-006b (ledger `nlm006b_prereg_transport_audit`): independent
  candidate strata, transported-pair predictors, label-preservation and
  calibrated displacement gates. Lesson: candidate pools must never be
  selected by the metric under test.

## Round 10 closure — frozen-encoder closeness/map line closed (2026-08-27, narrowed by audit #3)

- Ledger `round10_frozen_chart_closure`. The NLM-003 R-over-F claim is
  withdrawn (coarse-taxonomy leak, see diagnostics below); NLM-005 is void on
  support; no native construct built so far (substitutability profiles, Fisher
  pullback) competes with the trained chart metric on this artifact.
- **Residue as narrowed by Tier-3 audit #3:** training creates a
  task-effective chart and affine-path smoothness *in this encoder/dataset*
  (cosine 0.946 trained vs 0.575 random-init; same-class chart-line flicker
  12.7% vs 95%). Not a general claim that native constructs are dominated, and
  not proof of intrinsic geometry or of "straight routes inherited from
  training" beyond this encoder and dataset.
- Replacement line: NLM-006/006b — stratified transports outside the trained
  invariance class.

## NLM-005 — composed transport/substitution; VOID on support (2026-08-27)

- **Design.** Locked `a12aad4` (artifact lock `aab0f69`). hflip and 1-px-shift
  transports re-encoded by the frozen encoder, composed with random
  substitutions in both orders (ST, TS), true fine-label endpoint. Ledger
  `nlm005_v1_composition`; artifact `experiments/results/nlm005_v1/analysis.json`.
  Transport families now live in
  `experiments/results/vision_cifar100_dinov2s_edits_v2/edits.npz`
  (`test_emb_hflip`, `test_emb_shift1px`; byte-identical to the original
  NLM-005 file, which was removed as superseded).
- **Void by kill condition 3:** support 129/400 (32%) < 80%. Order gaps
  non-diagnostic: ST−TS cosine ≤ 0.006 (hflip 0.006 [−0.003, 0.017], shift
  0.004 [−0.003, 0.013]); shift1px R_no_coarse 0.027 [−0.003, 0.057] on a
  sensitivity row. Cosine leads native candidates by ≈0.32 on every order.
- **Lessons.** hflip/1-px shift are augmentations DINOv2 was trained to be
  invariant to, so they are near-identity moves in its world — transports must
  lie outside the trained invariance class. 40 random candidates over 100
  classes cannot reach 80% support — candidate sampling must be stratified.

## NLM-003 v2 diagnostics — R's win was a coarse-head leak (2026-08-27)

- **Design.** Same lock, artifact, endpoint as NLM-003; new anchor sample; audit
  #2 diagnostics (tie accounting, R without coarse head, cheap-baseline ladder,
  kNN k-sensitivity). Ledger `nlm003_v2_diagnostics` (Round 9: sensitivity
  accounting, not new evidence); artifact
  `experiments/results/nlm003_v2_diagnostics/analysis.json`.
- **Leak.** `R_no_coarse` 0.586 < `F` 0.667 (R with coarse 0.762; fine labels
  nest inside coarse classes). The NLM-003 R-over-F directional claim is
  withdrawn. Δ_{F−R} on this resample −0.095 [−0.142, −0.049]. R ties on
  22–33% of comparisons.
- **Ladder.** cosine 0.934, PCA-32 cosine 0.941, Euclid 0.933; pixel-stat
  Euclid 0.624, raw-pixel cosine 0.622. kNN same-class flicker 0.18/0.13/0.10
  vs cross-class 0.41/0.38/0.37 at k = 8/32/128 — world-path contrast robust to k.

## NLM-004 — random-init null world; SUPPORTED (2026-08-27)

- **Design.** Preregistered in ledger (`nlm004_prereg_null_world`) before
  scoring: random-init DINOv2-small chart
  (`experiments/results/vision_cifar100_randinit/`), true fine-label endpoint.
  Ledger `nlm004_v1_null_world`; adjudication `nlm004_round9_adjudication`
  (supported, exploratory — bootstrap CIs not in artifact); artifact
  `experiments/results/nlm004_v1/analysis.json`. CPU, 230 s.
- **Supported.** Cosine 0.575 in the null chart vs 0.946 trained (gap 0.371;
  gates ≤ 0.70 and ≥ 0.20). Embedding-kNN fine accuracy 0.069 vs 0.761.
  Same-class chart-line kNN flicker 95% (null) vs 12.7% (trained). Semantic
  heads collapse (coarse 0.21) while pixel-statistic heads stay strong (rgb
  0.83, luma 0.82) — cheap-baseline confound noted.
- **Reading.** The chart's task-effective metric and affine-path smoothness are
  created by training in this encoder/dataset; the null chart has neither.

## NLM-003 — R beats F on the true fine-label endpoint; cosine dominates both (2026-08-27) — R-over-F WITHDRAWN (see v2 diagnostics)

- **Design.** Locked at `e2a1fb2` (`theory/EXPERIMENTS.md`, NLM-003). Same
  frozen CIFAR-100/DINOv2-small artifact and runner as NLM-002, endpoint
  switched to the true fine label (no head is trained on it):
  `python experiments/run_nlm002_vision.py --cache experiments/results/vision_cifar100_dinov2s --out nlm003_v1 --endpoint fine_label`.
  Ledger `nlm003_v1_true_fine_endpoint`; artifact
  `experiments/results/nlm003_v1/analysis.json`.
- **Directional gate met.** Profile-continuity `R` 0.734 vs Fisher pullback `F`
  0.630, Δ_{F−R} = −0.104 [−0.148, −0.058] over 6,199 scored pairs; support
  thin (130/400 anchors had a same-fine-class candidate among 40 draws).
- **Chart metrics dominate.** Cosine 0.946 and Euclidean 0.935 on the same
  anchors beat both native constructs by 20–30pp.
- **Tier-3 audit #2 reclassification (adopted).** NLM-003 is a **narrow
  instrument comparison** — "these implementations lose to cosine on this
  endpoint" — not evidence that native geometry is generally dominated (one
  encoder, one endpoint, one-step random substitutions, one seed, 130 supported
  anchors). `R` takes five values with 0.5 tie credit and includes the coarse
  head (fine nested in coarse), so tie accounting and an `R`-without-coarse
  rerun are required. Next gate: random-init null (NLM-004,
  `nlm004_prereg_null_world`), cheap-baseline ladder, kNN k-sensitivity,
  nonlinear re-charting, composed / out-of-distribution moves.

## NLM-002 — non-LM branch (CIFAR-100/DINOv2): endpoint killed, chart-path structure found (2026-08-27)

- **Design.** CIFAR-100 → DINOv2-small CLS, 6000 train / 2000 test, raw pixels
  stored (`experiments/results/vision_cifar100_dinov2s/`, built by
  `experiments/build_vision_cache.py`). Runner `experiments/run_nlm002_vision.py`
  (default endpoint `rawpixel_knn`). Ledger `nlm002_v1_nonlm_branch`; artifact
  `experiments/results/nlm002_v1/analysis.json`. CPU, 133 s.
- **M2 kill condition met.** Raw-pixel k=32 kNN fine label is nearly
  uninformative (0.115 accuracy; 0.12 agreement with embedding kNN, which
  scores 0.761), so the locked endpoint is invalid and M3 (`F` 0.601 vs `R`
  0.605, Δ = −0.004 [−0.034, +0.026], 16,660 pairs) is a tie on noise, not a
  primitive verdict. Lesson: an endpoint must be independent of both candidates
  *and* informative — the true fine label is both (→ NLM-003).
- **M1 chart-path structure (informative, audit-qualified).** Along straight
  lines between same-class embeddings the coarse-semantic readout flickers on
  only 2% of paths; between classes the fine-label kNN flickers on 38% and
  any-readout on 78%. Audit #2: the 2% figure is weak evidence (affine argmax
  is near-monotone by construction) and kNN flicker is at k=32 only — a
  k-sensitivity analysis is required before any world-path claim. Pixel-stat
  heads are weak (52–59% test acc), so their 21–24% flicker is partly head noise.
- **Implementation decisions flagged at lock:** pixel statistics of
  interpolated points are approximated (no pixels exist off the data), and the
  fine-label head is never trained.

## NLM-001 — verdict: negative on predictive novelty (2026-08-27)

- **Design.** Analysis-preregistered at `fea3a8f` over sequestered raw
  matrices (`experiments/results/nlm001_v1/manifest.json`); three CPU systems
  (Qwen3-0.6B, gemma-3-270m, SmolLM2-360M); primary = 72 calibration-unseen
  words, all 80 as sensitivity; `--rule pooled --scale-normalize`. Command and
  metrics: ledger `nlm001_v1_primary_72`; artifacts
  `experiments/results/nlm001_v1/analysis_primary_72.json`,
  `analysis_sensitivity_80.json`.
- **Central bet fails.** Native calibration-KL closeness does not beat a learned
  diagonal Mahalanobis metric on the model's own contextual hidden states for
  held-out orderings: Qwen Δ = −0.058 [−0.222, +0.034]; gemma Δ = +0.017
  [−0.02, +0.06]. Every predictor scores 0.95–1.00 — the robust held-out labels
  are large-gap easy pairs (instrument limitation). Post-verdict reading
  (ledger `nlm001_v1_postverdict_note`): unlearned centered contextual cosine at
  layer 14 reaches 1.000 vs native 0.954, and the preregistered selection rule
  chose an overfit metric (calib 1.000, held-out 0.947), so the reported Δ
  understates the native loss.
- **Context reversals exceed the paraphrase null** in Qwen (Q = 2.12
  [1.70, 2.56], R = 0.18) and SmolLM (Q = 17.1 but W ≈ 0.005, so Q is not
  interpretable there); not in gemma (Q = 1.40 [0.90, 2.55]).
- **Directedness absent.** Robust (≥2-of-4) asymmetric pairs: 1.5% Qwen, 9.2%
  SmolLM. Cross-system transfer τ_b: 0.14 (qwen|gemma), 0.47 (qwen|smollm),
  0.14 (gemma|smollm).
- **Kill conditions 3 (predictive novelty), 6 (coordinate confound), 8
  (instrument metadata recorded post hoc) apply.** Tier-3 fresh audit adopted:
  T2 is geometrically vacuous, κ is an invariant of the probe table not the
  space, B>W may recover the hand-authored block taxonomy, and no NLM-001
  outcome could have earned "cosine is the wrong object".
- **What we learned.** The substitutability/KL primitive on lexical embedding
  rows adds nothing over a symmetric learned metric on contextual states. Do not
  run NLM-002 on more words; next is a competition among primitives (see
  `STATE.md`). Runners must record tokenizer revision, library versions, thread
  and batch settings at run time.

## NLM-001 — instrument calibration, pre-verdict (2026-08-27)

- **NLM-001 — contextual substitutability, context rank, and transfer.** Frozen
  theory contract: `theory/EXPERIMENTS.md`. One CPU entrypoint,
  `experiments/run_lexical_closeness.py` (using the existing substitution-probe
  helper); frozen slice:
  `experiments/config/lexical_probe_v1.json`. The 12-word smoke and disclosed
  eight-item full-pipeline validation are calibration only. The latter
  invalidated the MAD robustness rule and put asymmetry signs at chance; H1 is
  exploratory. Primary analysis uses the 72 calibration-unseen words, with all
  80 reported only as sensitivity. Three-system raw matrices were acquired
  concurrently before the Round-2b amendment and stayed sequestered until the
  amended contract was committed at `fea3a8f`; the verdict entry above is the
  first outcome analysis. Ledger: `substitution_probe_smoke_qwen3_0p6b`,
  `nlm001_pipeline_smoke_8`; artifacts `experiments/results/pipeline_smoke_8/`.
