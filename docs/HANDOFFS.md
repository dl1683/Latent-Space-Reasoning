# Session handoff — 2026-09-13

## Current routing

Read [README](../README.md), [STATE](../STATE.md), and the
[canonical direction and decision map](../theory/CUMULATIVE_CAPABILITY_ACQUISITION.md).
Preserve the enduring denizen philosophy—identity, moves, effort, maps and laws—
and its history. Cumulative capability acquisition is the current direction:
combine useful contributions from successive pretrained models while retaining
prior capabilities. Connected growth and assimilation after donor removal are
separate branches. No cumulative capability or cost advantage is established.

**Mode:** autonomous strategy/discovery, including definitions, source study and
small bounded CPU probes, without system-building or large-model loading.
Research began at 19:45 UTC; first periodic approach review completed around
21:42 UTC (e1027); next due around 23:42 UTC after consolidation. Challenge
approach, never mission. GPU
approval remains per run. Old locked panels and paused lines are not restart
instructions; no further model run or architecture is selected.

## Evidence and next boundary

Canonical S6 records ineligible legacy plan exports, upheld by integrity audit
e1011–e1012. S21 records four local conditional rectangles below registered
tolerances; fresh real Codex CLI audit e1021 found an additive class also fits the
primary tolerance. This does not establish useful updating or a global law.
S22's cached two-donor rank average loses to the retrospectively better donor on
all four assays. Fresh real Codex CLI audit e1022 identified exact tie handling;
the correction is in the retained result and independently validated, without
changing the negative direction. No conclusion about all combinations follows.
S23 supplies proof-only joint compatibility and identification cautions, not a
new empirical result or construction proposal.

The first task-first block closed early: the PFN1 case in S8 has a repeated
training question/answer despite distinct DNA variants. It lacks the independent
effect contrast, receiver role evidence and usable receiver needed for the
proposed joint-contribution test. This is one source example, not a benchmark
score or trained-model attribution. S24 now works through an actual table
illustration on paper, assigning text, structure, binding and later queries to
their sources. No model performance or product selection is implied. Coordinate
repair is a caller obligation; crop-independent intervention laws would largely
test our wrapper. S2.2 now specifies reuse of the acquisition procedure's recipe,
evidence permissions and budget rule, allowing new learned adapters and receiver
adaptation. Typed ports and explicit fact import remain serious alternatives.
S25 separates target editing from acquiring useful consequences; correction can
legitimately revise prior answers. Before further models, select a useful revision
and fresh-input reuse obligation, name the donor's contribution and the old uses
that should survive. Conditional contribution need not include standalone
superiority; attribute integration learning explicitly. S12's conditional tied-head
calculation supplies parameter bookkeeping, not demonstrated compression or speed.
No run, artifact, architecture or reproduction project is selected.
Details live in
[NOTEBOOK](../NOTEBOOK.md), the canonical document and the
[experiment record](../experiments/EXPERIMENTS.md).

Session totals: 421 apparatus / 0 artifact-bearing code lines (98 + 119 + 204),
three probes / zero building rounds, eight model forwards, no GPU or training.
The user's strategy-only mandate governs; construction is not required to repair
the ratio. Preserve retained results, source subsets, provenance and negatives.

Root is the sole blackboard writer after cross-session ID collisions; reviewers
read boards and return findings for root to record with provenance. Integrity
entries were restored as e1011–e1012. Skip mission synthesis under the documented
large-board hang workaround; use fresh real Codex CLI sessions for result audits.
This consolidated routing replaces repeated current-session checkpoints only;
all earlier handoffs below remain historical evidence.

## Earlier handoffs (historical)

The dated records below preserve the path to the present project. Their runtime,
quota, active-task and restart instructions describe those sessions, not this one.

# Handoff 2026-09-03f

# Session Handoff — 2026-09-03f

## What happened

Continuation of 2026-09-03e. Git at 7210e01 (pushed) on entry. EMI results
already complete and committed.

### Runtime gap localization: Mamba state continuation

Traced the EMI divergence to a HuggingFace implementation gap in Falcon-H1's
`torch_forward` method:

1. **Line 819** of `modeling_falcon_h1.py`: `previous_states = torch.zeros_like(states[:, :1])`
   — the inter-chunk SSM initial state is hardcoded to zeros for multi-token
   continuation (`seq_len > 1`).

2. **Line 705**: Convolution applied via `self.conv1d` without incorporating
   the cached conv state (uses nn.Conv1d zero-padding instead).

3. Both `mamba_chunk_scan_combined` and `causal_conv1d_fn` accept `initial_states`
   parameters — the Falcon-H1 model code never passes them for `seq_len > 1`.

4. The CUDA path (`cuda_kernels_forward`) has the same gap.

5. Single-token decode (`seq_len == 1`) correctly uses cached states (lines 686-762).

### Minimal verification

Wrote and ran `scratchpad/verify_mamba_cache_bug.py`:
- One-shot vs two-shot (multi-token): **TV = 0.118**, max logit diff = 6.527
- One-shot vs single-token decode: **TV = 0.000000** (exact match)
- Cross-prefix two-shot response convergence: TV = 0.794 (consistent with lost
  direct recurrent history; not proof of zero prefix influence through attention)

### Scope of impact

- **ALL SVB experiments on Falcon-H1** used the affected multi-token cached
  continuation pattern (`get_state_after_prefix` → `get_dist_from_state`).
  The path omits direct cached recurrent and convolutional history. Prefix
  information may still enter through attention-conditioned hidden states, so
  source inspection does not prove total prefix blindness or attribute every
  response effect to the omission.
- **ALL suffix algebra experiments** (legacy and current) used the same pattern.
- **SVB-Qwen3** does not use this Falcon/Mamba code path; its bounded result
  survives, without establishing universal cache equivalence.
- **SVB competence staircase** used full-text mode — CORRECT.
- **EMI Mode F** is the full-forward reference; all registered suffix defects
  on its nine-word panel were below TV 0.016.
- Cross-architecture comparison (Falcon-H1 vs Qwen3) is invalid.

### Corrected measurement complete

`verify_svb_decode_mode.py` completed and was independently audited. Across
four fixed 27-cell conditions, the condition-mean 11-bin response law from
tokenwise cached query processing was within `1.2107523675e-6` TV of full-text
processing. The legacy multi-token cached query path was `0.546–0.885` TV from
full text.

This is a bounded runtime diagnostic, not the complete suffix-action gate. The
runner substitutes s0/s1 into the prefetched prefix and varies only the
subsequent query chunking. The JSON retains four aggregate rows rather than the
108 per-cell distributions. It therefore verifies the query-continuation defect
but does not independently establish cellwise or full-vocabulary equivalence.

## Surviving Qwen3 finding: bounded suffix-conditioned response effect

Qwen3 is unaffected by this specific Falcon/Mamba runtime defect. Its stored
fixed-panel result remains:

| Depth | s0 (σ) | s1 (σ) | Absolute change (pp) |
|-------|--------|--------|---------------------:|
| d1    | 0.958  | 0.970  | +1.2 |
| d2    | 0.897  | 0.945  | +4.9 |
| d3    | 0.792  | 0.900  | +10.8 |
| d4    | 0.753  | 0.880  | +12.6 |

The fixed comment improves the registered correct-digit probability more at
greater depth on this one model and prompt family. The intervention changes
computation, token position, lexical/syntactic context, and attention state at
once. It does not yet identify a settling mechanism, prior answer encoding, a
continuous-vector operation, transfer, or an efficiency advantage.

## Current state

- **Reviewed starting head:** `418158080f88decbac1c4a0838c359076a855b36`.
- **SVB decode verification:** COMPLETE; result commit `826d751`, result SHA-256
  `81c3c622835c45b807bb4ee985b045a2576038edcb8a7c3de8f039578a3a88d5`.
- **Independent audit:** numeric diagnostic retained; architecture-causal
  interpretation rejected.
- **Blackboard:** e856-e867 contain the root cause and first public-surface
  audit. The append-only ledger correction and this handoff carry the terminal
  audit because the blackboard MCP was unavailable from the parent workspace.

## Restart procedure

1. Read AGENTS.md, CLAUDE.md, STATE.md, NOTEBOOK.md (top entry: MAMBA STATE BUG).
2. Read the completed result and the top correction entry in `NOTEBOOK.md`.
3. Do not rerun full SVB or enumerate more words. First freeze the original-
   segmentation F/T/L suffix-action gate with per-cell retention and complete
   provenance.
4. Treat the Qwen comment effect as a bounded observation until lexical,
   position, computation, and template controls separate its mechanism.
5. Do not use `attention consolidates`, `SSM erodes`, `architecture separation`,
   or `universal settling` without a matched intervention and causal ablation.

## What's next (priority order)

1. **Finish the suffix-action instrument:** original segmentation, F/T/L,
   per-cell laws, pinned model/runtime/cache/runner hashes, separate reduction.
2. **Run the cheapest Qwen mechanism controls:** matched-length lexical nulls,
   position controls, semantically equivalent paraphrases, and template/name
   holdouts before another model sweep.
3. **Architecture attribution only by ablation:** align intervention and prompt,
   then isolate recurrent and attention pathways rather than comparing two
   unrelated checkpoints.
4. **Native-math gate:** promote a response operation only if it survives
   presentation, composition, restoration, and future-response prediction.
5. **AGI bridge:** later acquisition remains the decisive endpoint; episode
   response improvement alone is not capability accumulation.

# Handoff 2026-09-03e

# Session Handoff — 2026-09-03e

## What happened

Continuation of 2026-09-03d. Git at fb5a3dd (pushed) on entry.

1. **Full 12x12 pairwise TV matrix analysis.** Computed all pairwise
   distances between the 12 band2 arms. Revealed structure beyond
   individual hypothesis tests: branch coalescence, overlapping tolerance
   cliques, exploratory contraction patterns.

2. **Two Codex evidence gates** (back-to-back). Both returned REVISE.
   Corrections adopted verbatim: H-SAT3 downgraded to "best fit,
   formally inconclusive"; CP not a right zero (coalescence, not
   absorption); contraction exploratory/unregistered; native object
   redefined as Moore-style response automaton; execution-mode
   invariance made top priority.

3. **Right-continuation experiment completed.** 4 arms × 27 cells,
   135 forward passes, 2654s. Critical results:
   - CPC does NOT absorb C (INCONCLUSIVE, TV=0.060)
   - PCP does NOT absorb P (REFUTE, TV=0.072)
   - CPCPC genuinely different from CPC (REFUTE, TV=0.099)
   - Sigma degrades: CPC=0.486 → CPCC=0.439 → CPCPC=0.393
   - Monoid does NOT collapse to finite quotient

4. **Commit a61a213 pushed to main.** Right-continuation results,
   both evidence gate corrections, theory doc rewrites, entropy sweep
   (9 stale files, 2 superseded handoffs, __pycache__). 11 files
   changed, 706 insertions, 213 deletions.

5. **Codex design gate for execution-mode invariance.** Verdict:
   CONDITIONAL GO after redesign. First draft NO-GO (6 critical flaws).
   Codex specified 4 modes (L/W/G/F), 15 words in 2 stages,
   replay-calibrated eps_mode, token identity gate, per-mode defect
   re-adjudication, Spearman structural invariance. Full spec in
   `scratchpad/codex_emi_design.txt`.

6. **Runner rewritten per Codex spec and launched.** Token identity
   gate PASSED (420 checks). Replay calibration: delta_replay = 0.0
   (perfectly deterministic), eps_mode = 0.01 (tightest possible).
   Mode L CPC bit-for-bit identical to band2 historical data.
   Stage 1 (core 9 words) now running.

## Current state

- **Git:** main at a61a213 (pushed). Uncommitted: EMI runner, ledger
  entries, EXPERIMENTS.md spec, theory intro corrections.
- **EMI experiment:** RUNNING (PID 27048, ~8GB RSS, CPU-only).
  Stage 1 core capture in progress (~3-4 hours estimated).
- **Crons:** 4 watchdogs live (jam/hygiene, hourly ops, 2h audit,
  2h entropy).

## Key EMI early findings

- **delta_replay = 0.0** — all 120 replay evaluations perfectly
  deterministic. The instrument is valid with maximum sensitivity.
- **eps_mode = 0.01** — any cross-mode TV above this is real signal,
  not noise. Known suffix effects (TV ≈ 0.05-0.10) are 5-10× above
  this threshold.
- **Backward compatibility** — Mode L perfectly reproduces band2 data.
  All prior experiments are confirmed as Mode L equivalents.

## Key structural findings (from prior work this session)

**The monoid does not collapse.** Words continue evolving past length 3.
H-SAT3's near-return (CPCP≈CPC) is single-step and does not persist.

**Branch coalescence, not absorption.** CP's children converge toward
each other but remain separated from CP itself.

**Native object:** truncated response-labelled continuation automaton.
All conclusions scoped to cached pathway until EMI resolves.

## Pre-registered interpretation rules (from Codex design gate)

| Outcome | Licensed conclusion |
|---|---|
| L differs from W | Query-boundary artifact in legacy runner |
| W differs from G | Generator packaging changes state; no cached action |
| L/W/G agree, F differs | Cached-state action system valid; Z = cache states |
| All four agree (core) | BAND2/SAT3/right-continuation mode-invariant |
| All four agree (all 15) | Execution-mode blocker clears |

## Zero-call analyses (completed while EMI runs)

Four analyses from existing data, zero model calls:

1. **Common-tail forgetting (Codex alt #4)**: REFUTED. Shared tail length
   has no predictive power for response distance (Spearman=0.059, p=0.55).
   Cross-tail pairs can be closer than same-tail pairs.

2. **Word-space geometry**: Pairwise mean-TV is a valid metric (0 triangle
   inequality violations / 4095 checks). 2D MDS captures 93% of variance
   (Spearman=0.975). Space is essentially 2-dimensional.

3. **C/P generator asymmetry**: C is "absorptive" (C-tail cluster TV 0.05-0.09).
   P is "disruptive" (P/PP isolated at TV 0.24-0.30 from longer P-ending words).
   Not predicted by any symmetric generator model.

4. **String baselines fail**: Shared tail (rho=0.059), edit distance (rho=0.077),
   word length diff (rho=0.311) — none predict response distances. Weakens
   simple parser-state alternative (#3).

Alternatives scorecard: #2 answer-format gate RULED OUT, #4 common-tail RULED OUT,
#5 panel superposition partially addressed, #3 parser-state weakened, #1 non-Markov
readout aliasing deferred to CPA multi-probe test.

## What's next (when EMI completes)

1. Read Stage 1 results: mode comparisons, defect re-adjudication,
   structural invariance (Spearman).
2. Run `reanalyze_emi_simultaneous.py` (simultaneous max-statistic band).
3. Apply pre-registered interpretation rules.
4. Run Codex evidence gate on EMI results.
5. Log to ledger, update STATE/NOTEBOOK/EXPERIMENTS/theory.
6. Commit EMI runner + results + all updates.
7. If EMI passes → fire Codex CPA design gate (prompt prepared).
   Per Codex strategic reflection, pivot to Causal Predictive Automaton
   test (NOT more suffix enumeration).

## Key files

- `experiments/run_execution_mode_invariance.py` — EMI runner (Codex-designed)
- `experiments/results/execution_mode_invariance/` — output directory
- `scratchpad/codex_emi_design.txt` — Codex design gate output (182 lines)
- `theory/closed/SUFFIX_ACTION_ALGEBRA.md` — central constructive artifact
- `experiments/results/band2_decisive/` — band2 distributions (backward compat confirmed)
- `experiments/results/right_continuation/` — right-continuation distributions

# Handoff 2026-09-03d

# Session Handoff — 2026-09-03d (updated with results)

## What happened

1. **Post-compaction restart.** Resumed from commit b0a9382 (suffix action
   algebra committed with Codex audit corrections).

2. **Codex design gate: LRB decisive experiment.** Two invocations —
   first jammed on off-topic analysis (HANDLE-mu data), second focused
   prompt worked. Verdict: **conditional GO**.

   Key corrections to the experiment design:
   - Primary metric: TV (total variation) on 11-bin distributions, not sqrt-JS
   - Threshold: epsilon_TV = 0.06 (calibrated from existing I_TV(C) ~ 0.050)
   - 8 arms only: C, CC, P, PP, CP, CPC, PC, PCP
   - Non-collapse guard: N(C,P) = TV(CP,PC) lower bound > 0.06
   - Pre-tokenized action IDs (compositionality verified)
   - Stratified bootstrap: 10K, seed 42, by variable
   - Three-way verdict: PASS / REFUTE / INCONCLUSIVE

3. **LRB decisive experiment: H-LRB REFUTED.** 243 forward passes,
   CPU-only, Falcon-H1-1.5B-Instruct. Committed 766a1cf.

   Results (TV metric, stratified bootstrap 10K, seed 42, 95% CI):
   - I(C) = 0.050 [0.042, 0.058] — passes registered rule
   - I(P) = 0.021 [0.018, 0.023] — passes registered rule
   - R(C,P) = 0.091 [0.081, 0.100] — REFUTES (LB > 0.06)
   - R(P,C) = 0.141 [0.126, 0.154] — REFUTES (LB > 0.06)
   - N(C,P) = 0.091 [0.075, 0.107] — non-collapse confirmed

4. **Codex evidence gate (211K tokens, xhigh).** Independent verification:
   - All 8 arrays finite 27x11, normalize exactly
   - 0/216 tokenizer boundary mismatches
   - Bonferroni-corrected LBs (~0.079, ~0.122) remain above 0.06
   - Key correction: band requires x^2=x for EVERY element, not just
     generators. Generator idempotence does not imply bandness.
   - Three competing hypotheses: H-BAND2, H-SAT3, H-GEN-IDEM
   - Warning: "forcing tolerance clusters into familiar semigroup varieties
     would be a new taxonomy trap"

5. **Theory documents updated.** SUFFIX_ACTION_ALGEBRA.md: H-LRB marked
   REFUTED, competing hypotheses added with frozen predictions.
   suffix_algebra.py: three normalizer modes (lrb/band/gen_idem).

6. **Band2 decisive experiment designed and launched.** 12 arms including
   CPCP, PCPC, CPP, PCC for right-action table. Cached-vs-full fidelity
   check included. Committed f95dfb7.

7. **Entropy sweep.** 39 stale scratchpad files deleted.

## Current state

- **Git:** main at f95dfb7 (pushed). Band2 experiment running (background).
- **Running:** experiments/run_band2_decisive.py (CPU, ~20 min expected)
- **Crons:** 4 watchdogs live (session-scoped)

## What's next

1. **Band2 experiment completes** -> read results, log to ledger, commit.
2. **Codex evidence gate** on the band2 results.
3. **If H-BAND2:** S is a free band on 2 generators. Build the Cayley
   table, right-continuation graph. Test with 4 generators (U, V).
4. **If H-SAT3:** Length-3 saturation, non-band. Characterize the
   absorbing elements. Different algebraic framework needed.
5. **If H-GEN-IDEM:** Alternating words grow. Only generator idempotence.
   Test CPCPC (length 5) to check continued growth.
6. **H-Truth experiment** remains pending (requires GPU approval).

## Key files

- `experiments/run_band2_decisive.py` — band2 test runner
- `experiments/run_lrb_decisive.py` — prior LRB test runner
- `theory/closed/SUFFIX_ACTION_ALGEBRA.md` — central constructive artifact
- `theory/suffix_algebra.py` — symbolic normalizer (3 modes)
- `theory/frozen_predictions_CP.txt` — frozen LRB predictions (falsified)
- `scratchpad/codex_lrb_refute_analysis.txt` — Codex evidence gate output
- `scratchpad/codex_lrb_design_gate.txt` — Codex design gate output

# Handoff 2026-09-03

# Handoff — 2026-09-03

## What happened this session

1. **SVB-0 completed.** Scope-Variable Binding on Falcon-H1-1.5B-Instruct.
   Python lexical scoping, DynamicCache state injection. 1854 model calls,
   73 min CPU. Verdict: INSUFFICIENT_SCOPE_BINDING (two-var 76.7% < 80% gate).
   Single-var: σ=0.68, κ=0.71, ι=0.36. First experiment 100x stronger signal
   than entity-location family. Commit eb93ac9.

2. **SVB-1 completed — SETTLING TIME LAW CONFIRMED.** Extended SVB-0 to depths
   1-4, single-var. σ decays with depth (0.68, 0.50, 0.28, 0.23) but neutral
   suffix tokens INCREASE σ at d≥2. Gain: +30% (d2), +57% (d3), +88% (d4).
   With optimal settling: σ*=(0.68, 0.65, 0.44, 0.43). Raw depth curve
   overestimates binding loss by ~2x at d4. Geometric decay prediction REJECTED
   (non-constant ratios). Commit f851203.

3. **AXIOMS.md grounded.** Connected settling time to navigation requirements
   (moves, cost, laws). Candidate native law: C(d) = d + s*(d). Commit efdeb57.

4. **SVB-2 designed but not launched.** Fine-grained suffix resolution
   [0,1,2,3,4,6,8] to resolve d4's anomalous s1 peak. Three pre-registered
   hypotheses. ~128 min CPU — not launched due to battery concern. Config at
   experiments/config/svb_2.json. Commit b00367c.

5. **Visualization artifact published.** Settling time law, interactive charts
   with hero stats, heatmap, and data table.

## Current project state

**The settling time law is the first candidate native axiom.** It has no R^n
analogue (all coordinates of a vector are equally accessible). The model stores
deep information but needs recurrent processing time to access it.

**Open questions (for Codex evidence gate, credits return 2026-09-06):**
1. Are SVB-0+SVB-1 results evidence-gate clean?
2. Does settling time qualify as a native axiom candidate?
3. Strongest counter-explanation? (memorization, tokenizer artifact, etc.)
4. Which single model most efficiently tests universality?
5. Is the 11-bin response law a confound?
6. Should we formalize C(d) = d + s*(d) or wait for SVB-2?

## Next steps (priority order)

1. **Codex evidence gate** (2026-09-06) — agenda prepared in NOTEBOOK
2. **SVB-2** — config ready, launch when battery allows (~128 min CPU)
3. **Cross-model comparison** — same task on pure transformer vs pure SSM
4. **Variable count scaling** — 1-4 vars at depth 1
5. **Binding composition** — does f(g(x)) compose from f and g?
6. **Formalize settling time** — with Codex, as native axiom

## What's preserved

- All experiment records: STATE.md, NOTEBOOK.md, experiments/EXPERIMENTS.md,
  experiments/ledger.jsonl
- All results: experiments/results/svb_0/, experiments/results/svb_1/
- Theory: theory/AXIOMS.md (empirical grounding section at line 70)
- Config for next experiment: experiments/config/svb_2.json
- Visualization: artifact at claude.ai (settling time law)
- Memory: direction_native_latent_math_2026_08.md (updated)
- Prior work: legacy/ (unmodified, do not run or cite)

## Key constraint

Codex CLI credits exhausted until 2026-09-06. Design gates and evidence gates
deferred. Theory formalization (settling time as axiom) blocked on Codex.
CPU experiments allowed but sustained load risks hard shutdown (battery).

# Handoff 2026-09-01

# Handoff — 2026-09-01

## What happened this session

1. **EAC-1 PASS → STOP.** The from-scratch carrier (359K params, 6 states, 3
   actions) passed all 7 causal gates across 3 seeds. Codex review determined
   it is architecturally tautological: the carrier IS the next-state embedding,
   host/donor share the same world, no action representation is transplanted.
   EAC/LAC line stopped. Commit 020ba04.

2. **Direction dialogue (3 Codex rounds).** Converged on testing KV command-slot
   transplant in frozen Qwen3-0.6B-Base as a bounded engineering falsification
   (not native math — Codex self-corrected in R2). Concrete task: three-item
   permutation. Budget: 4 launches, 2000 CPU forwards, 6 wall hours.

3. **Eligibility test: INELIGIBLE.** Qwen3-0.6B-Base scores 48% on three-item
   permutation (gate >=95%). Rotate left 13%, rotate right 34%, reverse 97%.
   Kill condition fires. Entire 4-launch program dead. Commit 9a27527.

4. **Codex R3: no next step.** "There is no concrete next experiment I can
   specify with confidence. The project needs user direction."

## Current project state

All experimental lines are stopped or killed:
- Phase 1: R^n tools closed (50 experiments)
- Phase 2: Observational ceiling in Qwen micro-world (18 experiments)
- Phase 3: EAC/LAC from-scratch carriers tautological
- Phase 4 attempt: Model can't do the eligibility task

The project is OPEN. Native math existence remains axiomatic. The experimental
program is paused at a genuine fork point.

## The decision needed

Codex R3 framed it as:

> Should Phase 5 pursue discovery inside a larger frozen pretrained model, or
> co-develop a modified real model whose latent space supports navigation — and
> what externally valuable task will define effectiveness against prompting,
> symbolic control, and KV caching?

Codex leans toward co-development: accumulated holes suggest current frozen
spaces may be structurally hostile to well-defined navigation. But this changes
the project from discovering native structure to constructing a next-generation
latent space.

## What's preserved

- All experiment records: STATE.md, NOTEBOOK.md, experiments/EXPERIMENTS.md,
  experiments/ledger.jsonl
- Theory deposits: theory/AXIOMS.md (D1-D9, Theorems 1/4/7/8)
- Codex direction transcripts: scratchpad/codex_direction_r1.txt through r3.txt
- EAC-1 as positive control for what tautological gates look like
- Phase 2 findings (commitment bottleneck, non-naturality, fusion-fission
  distributed mechanism, QPC-1 hole result)
- Blackboard state: 742 entries on board 5df235ea

## Restart procedure

When a direction is chosen:
1. Read AGENTS.md (operating process)
2. Read STATE.md (canonical state)
3. Read NOTEBOOK.md (latest 2-3 entries)
4. bb_get_state on board 5df235ea (mission board)
5. Run Codex design gate for the new direction (2-3 rounds)
6. Write config, runner, preregistration
7. Execute

# Handoff 2026-08-30

# Latent-Space Reasoning — Phase 1 handoff (HISTORICAL)

> **HISTORICAL DOCUMENT.** This handoff was written when the program was
> temporarily closed on 2026-08-31. **The program was REOPENED on 2026-08-31**
> because the closure was not authorized by the project owner. Native
> latent-space math existence is axiomatic. Phase 2 begins with genuinely
> non-R^n approaches. The Phase 1 record below is preserved as-is for
> provenance. Current state: `STATE.md`. Current direction: `README.md`.

This is the front door for anyone — human or coding agent — arriving with no
context. It tells you what the program was, why it closed, what it produced,
and where the residue went. Audit-licensed scientific wording is quoted or
linked to its source. The arc, repository map, and operating sections are
navigation summaries and must not be cited as findings. `STATE.md` governs
scientific claims.

## Mission and guiding question

**Program (2026-08-27 → 2026-08-31, CLOSED): the native mathematics of latent spaces.**
Embedding and latent spaces are not number systems. This program attempted to
build their mathematics from axioms upward — what a latent space is, which
relations and operations are meaningful — Euclid-style, rather than porting
ℝⁿ mathematics onto embedding vectors.

> Mathematics was invented by inhabitants of a world to navigate it — counting, measuring, mapping, predicting — and its laws were shaped by what that world made necessary. Invert the dynamic: **take the latent space as the world.** Ask what a denizen of that world would have to invent to find its way — what counts as the same place, what a move is, what effort a move costs, what a map is, what regularities make prediction possible — and let that need decide which primitives and laws we build.

## Why the program closed

Three converging findings, confirmed across 4 rounds of Codex direction
dialogue (rounds 4–7) and a final portfolio review (round 8):

1. **The R^n trap.** Every empirical approach used R^n tools (PCA, Procrustes,
   cosine similarity) to study a space whose premise is that it isn't R^n.
   The program could not escape its own measurement apparatus.

2. **No new mathematics found.** All proposed native objects (separator
   languages, executable differences, cell complexes, internal port
   sufficiency) mapped to existing established mathematics (coalgebra, PSR,
   automata, causal abstraction, higher-dimensional automata). Across 50
   audits, no construction established genuinely novel mathematical structure.

3. **Architectural impossibility (for current architectures).** The closing
   theoretical analysis (committed `dab8471`, `15ce061`) showed:
   - **Transformers:** Growing KV-cache state is non-R^n but inaccessible
     from the fixed-d residual stream. The Robust Port-Compression Conjecture
     formalizes why: no fixed layer's newest-token residual realizes the full
     append-action process once prefix complexity exceeds d.
   - **SSMs:** Fixed-dimensional state is accessible but described by standard
     spectral theory and semiseparable matrices — not new mathematics.
   - **Neither yields genuinely new latent-space math.** A substrate that
     would require new math needs fixed-dimensional, non-linear-algebraic,
     compositional, accessible state — which no current architecture has.

4. **Terminal experiment.** PSQ-3α (task-trained LoRA, single-seed, 5000
   steps) returned NO_INTERFACE (69.14% accuracy, gate ≥95%).

## Deposits and residue routing

### Theory (in this repo)

`theory/AXIOMS.md` carries an adopted foundation — future-response geometry
(D1–D9, Theorems 1/4/7/8): identity as zero future-response distance,
nonexpansive legal moves, surgeon-vs-denizen refinement, and the Robust
Port-Compression Conjecture (unaudited closing deposit). Architectural
comparison (transformer vs SSM) and speculative note on what state would
require new mathematics appended as closing deposits.

### Five transferable insights

1. A model has many operational latent spaces — indexed by (actions, observations, horizon).
2. Information ≠ state: state has three gates (present → addressable → composable).
3. The right null is the system's cheapest native mechanism, not random features.
4. A quotient must be earned by a transport — held-out presentation ≠ quotient-level generalization.
5. Absence requires a collision witness (same carrier, different future), not a failed probe.

### Cross-project deposits (Codex R8 routing)

| Deposit | Target | What |
|---|---|---|
| Three state gates, transport-earned quotients, collision witnesses, cheapest-mechanism nulls | `moonshot-llm-genome` | Strengthens its three-gate mechanism contract; HANDLE constructive pivot enters there as a co-trained substrate, not here |
| Information-vs-causal-state distinction, action-conditioned carrier preservation | `neuro-ai-lab` | Deposit into pruning/circuit work; not a reopening |
| Portfolio index update | `_meta/INDEX.md` | LSR row updated from active to closed |

### HANDLE (constructive pivot)

HANDLE (Handle-Addressable Neural Dynamics with Lawful Execution) was
identified as the constructive next step — build an architecture with explicit
state commitments, then discover its mathematics. **HANDLE belongs in
`moonshot-llm-genome`**, which already owns the contract it must satisfy
(internal signature, causal intervention, reliability). It enters as a
materially new co-trained substrate, not an inherited hidden-state license.

### The conjecture

The Robust Port-Compression Conjecture is preserved as an unaudited open
deposit. It should not become a proof project here without first passing a
bounded novelty audit that: (1) separates it from memory lower bounds and
state-realization theory; (2) defines behavioral effective dimension
non-circularly; (3) replaces exact collision with an approximate or discrete
theorem. Its broad compression thesis is already close to Haris & Onak's
linear KV-memory requirements result.

## One-screen program arc

| Line | Status | Binding provenance |
|---|---|---|
| Prior perturbation/diffusion program | Closed 2026-08-27; nested-arithmetic claims withdrawn after the benchmark was shown to measure termination under a token cap | `legacy/README.md`; `legacy/docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md` |
| NLM-007 residual-dynamics line | Closed by terminal allocation rule, not by a scientific null | Audit #22; ledger `nlm007_closed_audit22`; `STATE.md` "NLM-007 — CLOSED" |
| Toy operational-quotient program | Ended; no learned artifact passed the complete exact reducer, and Round 37 found no architectural win | Audits #23–#26 plus the Round 37 audit |
| Frozen-model coordinate/interchange/state-bus/control-cost line | Stopped as an allocation pivot, not a scientific conclusion | Audits #27–#31 |
| One-write positive-control staircase | Exact constructions failed or localized instrument, extraction, actuator, or prompt/output-geometry constraints; no general memory/capacity conclusion | Audits #32–#36 |
| Constructed register substrate | Rung 0 is a qualified synthetic oracle-code selector; rung 1 is a qualified answer-supervised eight-symbol writer; neither establishes semantic writing, persistence, geometry, or pretrained-model structure | Audits #37–#38 |
| Real-model bridge preflight | Noncausal feasibility PASS (audit #39 UPHOLD): explicit-legend state linearly decodable at the record span under held-out entities, two held-out templates and a disjoint balanced permutation bank; not a code-level or causal bridge | Audit #39 |
| Theory-first restart (2026-08-30) | D1–D9 adopted; Robust Port-Compression Conjecture formalized (unaudited); architectural comparison (transformer vs SSM) deposited | Audits #40–#50; theory/AXIOMS.md |
| PSQ-3α terminal experiment | NO_INTERFACE (69.14%, gate ≥95%); task-trained LoRA could not establish the interface | experiments/results/psq3_alpha/result.json |
| Direction rounds 4–8 | Program closed; archive + deposit recommended (1 >> 3 > 4 > 2) | Codex direction dialogue rounds |

## Do not repeat

Every exact recipe below is closed to rerun. The rows include construction
failures, a measurement, qualified synthetic calibrations, and a noncausal
preflight; those scientific statuses are not interchangeable. The right-hand
column is a pointer to the binding licensed sentence and never-say list in
`STATE.md`; it is not permission to paraphrase.

| Construction | Closure reason | Licensed/never-say authority |
|---|---|---|
| NLM-007 | Terminal allocation stop; Round 34b inconclusive | `STATE.md` NLM-007 block, audit #22 |
| Operational quotient v1/36b/36c/36d and presentation quotient | Toy program ended; exact certificates diagnostic-only; Round 37 no architectural win | `STATE.md` closed-toy blocks, audits #23–#26 / Round 37 |
| `coordinate_v1` | `UNINTERPRETABLE — INVALID POLARITY BASELINE` | `STATE.md` historical coordinate block |
| `coordinate_v2` | Killed at the predeclared baseline gate | `STATE.md` historical `coordinate_v2` block |
| `coordinate_v3` | Coordinate claim negative; narrow late lexical-control diagnostic only | Audit #27 block |
| `interchange_v1` | Locked raw-zero baseline failed; no swap arm ran | Audit #28 block |
| `interchange_v2` | Fixed single-anchor replacement construction failed | Audit #30 block |
| `state_bus_v1r1` | Fixed-construction fail; repeated supervised response controller, not autonomous state | Audit #29 block |
| `control_cost_v1` | Fixed actuator/solver/readout/budget construction failed; censored costs void broader laws | Audit #31 block |
| `onewrite_state_v1` | Killed pre-lock because the visible-fact instrument failed; no state hypothesis tested | Current statement |
| `onewrite_recall_v1` | Valid instrument, but the registered held-out channel failed | Audit #32 block |
| `onewrite_recall_rung1` | Training-item construction failed; failure initially unlocalized | Audit #33 block |
| `oracle_actuator_rung0` | Exact shared-J oracle actuator failed | Audit #34 block |
| `site_oracle_v1` | Balanced bounded eight-way control failed; exact slot/prompt recipe closed | Audit #35 block |
| `reachability_v1` | Measurement classified `NO SLOT-SPECIFIC GEOMETRY CONCLUSION` | Audit #36 block |
| Constructed register rungs 0–1 | Qualified synthetic calibrations; synthetic staircase stopped after audit #38 | Audits #37–#38 |
| `register_bridge_preflight_v1` | No repair or rerun; use audit #39 | Audit #39 |
| PSQ-3α | NO_INTERFACE; task-trained LoRA terminal experiment | experiments/results/psq3_alpha/result.json |

## Durable findings from the frozen-model line

Verbatim from `docs/STRUCTURED_NEGATIVE_2026_08_29.md` ("Findings that survive audit"):

- "Under the current apparatus budget, this sequence of frozen residual-stream constructions is not yielding a native-mathematics artifact, so open-ended layer, task, and decision-rule repair stops here. This is an allocation pivot, not evidence that frozen pretrained residual streams lack usable native structure."

- `state_bus_v1r1` licensed "repeatedly maintained supervised response control with donor-verbalizer-specific lexical/semantic steering," not autonomous persistence, abstraction, general interchangeability, or native latent mathematics.

- `onewrite_recall_v1` closes "the fixed model/layer/source-state/16-dimensional encoder/injector/norm-cap/slot/objective/400-step/prompt construction—not one-write memory, persistence, block-12 capacity, or persistent state in real models generally."

- The strongest unified explanation is local and plural: several frozen-model effects are parsimoniously explained by cue-redundant lexical-semantic response geometry rather than persistent state.

> The sequence did not show that the block-12 slot is too low-dimensional to carry state; it showed that this exact frozen-model interface never converted weak, presentation-bound source and response geometry into balanced, content-specific causal control, even though the slot has a coherent several-mode local effect on the registered output logits.

## Repository map

- `README.md`: public front door (updated to reflect closure).
- `docs/HANDOFFS.md` (this file): closure handoff and historical record.
- `STATE.md`: canonical scientific wording and never-say lists; the "Current statement" at the top governs.
- `docs/STRUCTURED_NEGATIVE_2026_08_29.md`: the audited two-day synthesis of the frozen-model line.
- `theory/AXIOMS.md`: adopted foundation (D1–D9, theorems, conjecture, architectural comparison).
- `experiments/ledger.jsonl`: append-only provenance (one JSON object per line).
- `experiments/EXPERIMENTS.md`: detailed experiment record.
- `NOTEBOOK.md`: reverse-chronological dialogue and audit record.
- `experiments/config/*.json`: immutable contracts for completed runs.
- `experiments/results/`: result JSONs and small checkpoints; the ledger row's `data_ref` is authoritative.
- `legacy/`: the closed previous program (2026-08-27), not active.
- Local-only (git-ignored; absent from a fresh clone): `AGENTS.md`, `.codex_*`, `.blackboard/`, run logs.

## Newcomer traps

- Raw status strings are not final scientific claims. Audit wording governs.
- `state_bus_v1r1`'s displayed status omits one registered failure; use audit #29.
- The oracle-actuator result's "two codes worked" interpretation was corrected: code 0 reproduced the cue prior.
- `reachability_v1` does not establish two or three reachable dimensions.
- The tracked navigator `smoke_result.json` is not the 2,000-step smoke cited in `STATE.md`.
- `legacy/` is historical; its older NEXT/queue language is not authorization.
- Hash-bearing completed configs and runners are evidence; editing them destroys reproducibility.
- ~~This program is CLOSED.~~ **REOPENED 2026-08-31.** Phase 1 experiments below are closed results; Phase 2 uses non-R^n approaches. Do not reopen Phase 1's R^n-based repair ladders or measurement sweeps.

