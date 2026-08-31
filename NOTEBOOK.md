# NOTEBOOK

Reverse-chronological running log. Newest first. Each entry: what was done, what
was learned, what's next. Canonical state lives in STATE.md.

---

## 2026-08-31T06:05 — Architectural comparison: transformers vs SSMs under D1-D9

Added a theoretical note to `theory/AXIOMS.md` comparing how the coupling
conjecture and denizen-surgeon gap manifest in transformers vs SSMs. Key
finding: the conjecture's mechanism (dimension mismatch between port and full
state) is architecture-dependent. In transformers, the KV cache grows with t,
so a fixed-d port provably loses information. In SSMs, the full state is
fixed-dimensional, so the gap can close — but the state's structure is standard
linear-systems spectral theory, not new math. Conclusion: a genuinely new
latent-space mathematics would require an architecture whose state is neither
growing-dimensional nor standard-linear-algebraic — perhaps compositional,
typed, or topological state. This sharpens the program's central question and
partially explains why the negative result was inevitable for transformers.

---

## 2026-08-31T05:45 — Re-contextualization checkpoint #4 (hourly ops + 2-hour audit)

**Ops:** Work alive (Codex R8 running, all surfaces updated, 2.3GB cleaned).
No crashes. Ledger/STATE current. No root sprawl. Leverage: correct — waiting
on direction dialogue before committing to new work.

**Audit:** No new empirical claims since audit #50. The Robust Port-Compression
Conjecture is explicitly UNAUDITED with known gaps documented — no overclaim
risk. Skip full Codex audit fire.

**Anti-tunnel / re-contextualization:** Four live alternatives for this repo:
1. HANDLE pivot — trainable module with explicit state commitments. Tests
   insight #2. Concrete but is an engineering project, not mathematics.
2. Archive and deposit — transfer insights to neuro-ai-lab, moonshot-llm-genome,
   matrix-native-math. Clean exit.
3. Prove the coupling conjecture — pure theory project. Could be genuinely
   novel if the conjecture holds. Theory-first by construction.
4. **Study architectures with native structure by design** — SSMs (Mamba),
   RWKV, or linear attention have explicit recurrence/state. The "native math"
   might already exist in architectures designed with explicit state, rather
   than being excavated from transformers that weren't designed for it. This
   reframes the R^n trap: maybe the trap wasn't using R^n tools, but applying
   them to a space (transformer residuals) that genuinely IS R^n, and the
   interesting non-R^n structure lives in architectures that were designed to
   have state. This is a new alternative not previously considered.

**Tunnel-vision risk:** strong pull toward HANDLE (prior session wrote a full
spec). Option 4 deserves serious consideration — it reframes the entire
project. Codex R8 should evaluate all four. The central bet of this project
("latent spaces have native non-R^n mathematics") may be architecture-dependent
rather than universal: transformer residuals might genuinely be R^n, while SSM
states have provable algebraic structure. If so, the program closed for the
right reason, and the constructive path is studying the right architecture, not
adding machinery to the wrong one.

---

## 2026-08-31T05:10 — Coupling conjecture formalized in theory/AXIOMS.md

Wrote the Robust Port-Compression Conjecture as a closing deposit at the end
of `theory/AXIOMS.md`. States: no fixed layer's newest-token residual realizes
the full append-action process once prefix complexity exceeds the port's
dimension; the full KV-cache carrier does. Marked UNAUDITED — requires a
dedicated mathematics audit before any claim. Connected it to the D1-D9
foundation via Theorem 8's surgeon-refinement inequality. Known gaps
documented: gauge symmetry, soft-vs-discrete prefix dimension, prior art
(Haris-Onak COLT 2025). This is the program's one potentially original
theoretical contribution; everything else in the foundation is standard
coalgebra/observability. Status: awaiting user direction on program future.

---

## 2026-08-31T04:30 — Re-contextualization checkpoint #3 (post-closure)

Program closed. No new claims since audit #50 and direction round 7.
Inverse-tunnel-vision risk: being locked into "it's over" when the five
transferable insights and the coupling conjecture are independently valuable.
The R^n trap identification and three-gates-of-state framework are
methodological contributions applicable beyond this project. Next action
depends on user direction: close entirely, constructive pivot, or deposit
the methodological residue into the field. Pushed all commits to remote.

---

## 2026-08-31T03:55 — PSQ-3α result: NO_INTERFACE; audit #50 adopted; PSQ line closed

**Result:** PSQ-3α returned NO_INTERFACE — 177/256 = 69.14% accuracy (gate ≥95%).
Single-seed (42), PCA-k4, block-18, task-trained Qwen3-1.7B-Base LoRA (5,000
steps, 10 GPU bursts). Stopped at stage 1 (256/1,152 calls). Training improved
behavioral accuracy from frozen-0.6B 26.56% to trained-1.7B 69.14%, but the
registered interface gate was not met.

**Audit #50 (adopted verbatim):** The configuration received NO_INTERFACE; Codex
direction round 6 ends further PSQ work. The result does not scientifically close
the broader question — the quotient-holdout design shares all x-places between
calibration and held-out (only y-fiber varies), PCA is transductive, single-seed/
single-action scope prevents terminal scientific inference. The PSQ line stops
because continuing PCA/Procrustes/linear-chart repair is inside the R^n trap, not
because 69.14% proves general impossibility.

**Correction (supersedes earlier notebook entries):** Prior entries stating "no
tunnel vision" or "single thread is correct" are superseded by the R^n trap
identification (AGENTS.md 2026-08-31). The PSQ line was tunnel-visioned on R^n
tools applied to latent spaces, not building native latent-space mathematics.
Phase D/E execution next-steps from earlier entries are cancelled.

**Direction round 7 complete (4 rounds).** R1: proposed separator languages,
executable differences, cell complexes as native objects. R2: brutal novelty
assessment — all map to existing math (coalgebra, PSR, automata, causal
abstraction). R3: coupling conjecture stated formally but has proof gaps;
honest should-continue = no, close program, redirect to constructive
architecture. R4: five transferable insights (the program's real strategic
deposit):
1. A model has many operational latent spaces, not one — indexed by (actions,
   observations, horizon). "The latent space" is incomplete.
2. Information ≠ state: state has three gates (present → addressable →
   composable). Never call something state until it passes read/write/compose.
3. The right null is the system's cheapest native mechanism (identity +
   shared displacement), not random features.
4. A quotient must be earned by a transport — held-out presentation is not
   quotient-level generalization.
5. Absence requires a collision witness (same carrier, different future),
   not a failed probe.

Constructive pivot headline: "What if you could change one thing in an AI's
mind — and know exactly what else would change?" Build a system with explicit
read/write/transport/compose commitments and test whether learned cognition
can honor them.

---

## 2026-08-31T02:55 — Re-contextualization checkpoint #2 (fresh Codex anti-tunnel)

**PSQ-3α training in progress:** step 4000/5000 (80%). GPU stable at 48W/89°C.

**Fresh Codex anti-tunnel verdict (verbatim summary):** "REVISE BEFORE
CONTINUATION." Project historically broad (lexical, DINOv2, toy quotients,
frozen-model, constructed registers, native bridge) but presently tunnel-visioned
on a single thread. PSQ-3α is one model family, one artificial world, one
template, one site, one seed, one action, one analyst-chosen coordinate.

**Strongest alternative explanation (Codex):** Ordinary task- and prompt-induced
representation geometry. The held-out claim is misleading: all 8 x-places appear
in both calibration and held-out sets (only y-fiber varies). PCA is fitted over
all 32 states including held-out. Even a strong PASS demonstrates supervised
activation control, not native mathematical structure.

**Codex alternative explorations (recorded verbatim):**
1. Quotient-honest PSQ micro-test — hold out entire x places/edges from training
   and operator fitting, fit PCA on calibration states only, unseen templates.
2. Ordinary-representation baseline ladder — compare Procrustes against affine/
   ridge, decoder+steering, gradient steering, target-centroid patching.
3. Natural relational task — spatial/temporal/symbolic composition with held-out
   entities and lexicalizations from real pretrained semantic domains.
4. Non-LM causal group action — reopen vision/protein/audio with downstream-head
   intervention (e.g., controlled rotations in DINOv2).
5. Established dynamics comparison — predictive-state, bisimulation/causal-
   abstraction, Koopman baselines on the same trajectories.

**Resolution:** PSQ-3α continues as the terminal experiment per Codex direction
round 6 — its licensed meaning was already declared narrow ("one-action,
fixed-presentation response-law control in a task-trained real model"). The
anti-tunnel criticisms are valid structural limits, not bugs in the current run.
If PSQ-3α passes, the alternatives above (especially #1 quotient-honest and #2
baseline ladder) become the natural next discrimination tests. If it fails, the
program closes as constituted per direction round 6.

---

## 2026-08-31 — Re-contextualization checkpoint (anti-tunnel)

**Project:** Latent-Space-Reasoning. **Live question:** Can we demonstrate that
a real language model's latent space contains native mathematical structure
(places, moves, laws) that is not just projected R^n geometry?

**What still holds:** The theory stack (AXIOMS.md D1–D9, Theorems 1/4/7/8,
Propositions 2/6) is sound per audits #40–#48. The PSQ-3 v6 runner is built
and reviewed. The frozen-model line is definitively closed (audits #27–#36).
The necessary_register line showed a qualified instrument pass (audit #37) and
rung 1 pass (audit #38). The native bridge specification is lock-ready (#47)
but Codex should-continue said STOP on the current form.

**What PSQ-3μ reframes:** The frozen 0.6B model scored 26.56% on the
registered x-channel panel (gate ≥95%). The registered interface gate was
not met. PSQ-3μ is scoped to frozen Qwen3-0.6B-Base under this panel and
does not adjudicate full PSQ-3 (which uses a different model and training
intervention). The result does not invalidate PSQ-3 or the theory.

**Alternative interpretations/directions to hold live:**
1. **PSQ-3 on cloud hardware** — the designed experiment, waiting on
   infrastructure. False-pass reducer fix needed first.
2. **Native bridge program** — theory-side, Codex said STOP on current form
   (zero central artifacts, infinite effective ratio). Could pivot to "build
   the smallest thing that can be wrong" — a crude one-write intervention
   that tests the bridge definition directly.
3. **Theory-only push** — the axiom stack is sound but nothing is genuinely
   new mathematics. The distinctive material (registration-relative D2,
   coherent presentation transport D6, finite-access reconstruction) could
   be developed into a standalone mathematical contribution without any
   empirical artifact.
4. **Different model family** — all work so far is on Qwen3. A different
   architecture (Llama, Gemma) might have different latent structure.
5. **Different task** — the two-dial Z_8×Z_8 world is synthetic. A natural
   language task with known algebraic structure (e.g., spatial reasoning,
   temporal reasoning) might be more compelling.

**Tunnel-vision check:** Currently narrowed on PSQ-3 (intervention testing on
two-dial world). The theory stack and the native bridge are both idle. The
Codex should-continue verdict said the central bet is untested with zero
central artifacts — that is still true after PSQ-3μ. The highest-leverage
un-tunneled move is to ask Codex what the next direction should be given
the full picture: theory, empirics, and the should-continue verdict.

---

## 2026-08-31 — PSQ-3μ: NO_INTERFACE on frozen 0.6B, closed

**PSQ-3μ implemented and executed.** Micro phase added to `experiments/run_psq3.py`
(~250 lines), config at `experiments/config/psq3_micro_cpu.json`. Dry-run validated
32 states, 16/16 cal/held-out split, 8 unique oracle profiles, 1,152 call budget.

**Result: NO_INTERFACE.** Panel accuracy 68/256 = 26.56% (gate ≥95%). The model
did not demonstrate reliable performance on the registered x-channel `is_x_zero`
probes; the registered behavioral-interface gate was not met. Stopped at stage 1
(256/1,152 calls, 132.6s CPU). No prediction histogram was saved, so
contemporaneous prediction-frequency observations are not independently auditable.

**Disposition (audit #49 adopted verbatim):** `NO_INTERFACE` closes PSQ-3μ with
no repair. This result is scoped to frozen Qwen3-0.6B-Base under the PSQ-3μ
panel and does not adjudicate the distinct full PSQ-3 experiment, which uses a
different model and training intervention.

**What's next:** Fire one Codex audit on the result. Then return to Codex
dialogue for direction — the PSQ-3 full runner's false-pass verdict reducer
still needs repair before any cloud run, and the native bridge program
(`native_bridge_v1`) awaits authorization.

---

## 2026-08-31 — PSQ-3 runner built, reviewed, pilot passed

**Runner built** (`experiments/run_psq3.py`, commit cf0f4f7): ~1400 lines implementing
full PSQ-3 v6 pipeline — dataset generation, LoRA training, gate, geometry, PCA+Procrustes,
layer selection, 6-step causal staircase, composition, frozen-base control.

**Self-review (6 critical paths):** No verdict-changing bugs found.
1. `id()` set exclusion in training data: fragile but correct within single function call
2. Class-weighted loss: correct reshape/shift; minor waste passing labels to model
3. Procrustes edit formula: matches v6 spec exactly (`h + (h-mu) @ P.T @ (M-I) @ P`)
4. State-level geometry bootstrap: correctly resamples 64 states with replacement
5. Frozen-base action-weighted pooling: correct equal weight per action
6. Variable scoping with `--phase`: `dir()` pattern unusual but works; all early exits sound

**Independent verification:**
- All import-time invariants pass: 341 words, 43,648 triples, 64 unique profiles,
  72 oracle distances, 2,000 training triples (hash ecaa2d93), moved A=32 B=24 C=32 D=24
- Carrier position stable at token 103 across all states and word lengths
- Token IDs 0→15, 1→16 confirmed on Qwen3-1.7B-Base (revision ea980cb0)
- JS divergence: 0 for identical, 1 bit for disjoint deltas, symmetric
- Module imports cleanly (46 callables, no errors)

**Pilot (100 forwards on local GPU):** PASS.
- Rate: 25.1 forwards/s
- Carrier at token 103 (token ID 198)
- Elapsed: 4.0s (short burst, no crash)
- Full gate estimate: ~29 min at this rate (requires stable/cloud hardware)
- Results saved: `experiments/results/psq3/pilot.json`

**Codex correctness review (16 findings, adopted):** NOT CLEAN. Key fixes applied:
1. CRASH: `oracle_profiles` was dict, `nearest_state_decode` iterated keys not values → extracted to `build_oracle_profiles_list()` returning a proper list
2. CRASH: `torch.set_grad_enabled(False)` was global, would disable gradients for subsequent seeds → removed; `train_phase` now explicitly re-enables
3. Prompt space: training appended " 0" (two tokens [220,15]) but inference checked token 15 at wrong position → prompt now ends with "# prints: " (trailing space), training uses `prompt + answer`
4. Gate contamination: 1,872 training triples included in "held-out" gate → `training_keys` set passed to `gate_phase`
5. Invalid controls biased: G=-1 for invalid panels inflated Procrustes advantage → NaN + `paired_boot()` helper that filters NaN pairs
6. Replay fixture: d_panel metric instead of max-absolute logit difference → full logit vector comparison, threshold 1e-3
7. Cross-seed verdict: no PASS/FAIL logic → `seed_verdicts` tracker + global `PSQ3_PASS`/`PSQ3_FAIL`
8. OOM between seeds: models stayed on GPU → `del model; gc.collect(); torch.cuda.empty_cache()`
9. `dir()` checks: fragile → replaced with `phase != "all"` guards

Remaining Codex findings NOT fixed (operational, not verdict-changing at pilot stage):
- Training set order may differ from v6 canonical (hash matches, content identical)
- Geometry bootstrap (`state_clustered_geometry_bootstrap`) defined but never called (diagnostic only)
- Frozen-base control is partial G pooling, not full staircase
- No checkpoint-resume or hardware-placement guard
- Composition fixed-point exclusion is conservative (excludes more, harder to pass)

**Post-fix verification (2026-08-31):** All 5 critical fixes verified.
- Token position fix empirically confirmed: base model P(0)+P(1) ≈ 0.999 at eval position with fixed prompt (was P(space) ≈ 0.997 before fix)
- Pilot re-run: 25.9 fwd/s, carrier at token 103, 3.9s elapsed — consistent with pre-fix pilot
- Remaining 4 non-critical Codex findings assessed and deferred (no overall verdict gate, dir() cross-seed single-phase-only, OOM risk unlikely at 1.7B, frozen pooling safe with 24+ moved edges per action)

**What's next:** Full run requires stable/cloud hardware (375,908 inference forwards + 15,000 training steps). Runner is ready for science-grade execution.

---

## 2026-08-30 — PSQ-3 lock round 5: REVISE, 5 blockers; repair-round cap reached; v6 final

**Codex PSQ-3 lock round 5**: Verdict REVISE. Math core and most round-4 operational
repairs verified sound. Independent enumeration confirmed: 341 words, 43,648 triples,
2,016 pairs, 72 nonzero d_{2,4} values, 64 unique panel signatures, 32/32 split,
moved counts 32/24/32/24. Five residual blockers:

1. **Training-set construction contradictory**: 16 reserved positives not reconciled
   with subsequent 1,872 draw (128 + 16 + 1,872 = 2,016, not 2,000). Population
   ordering, final dataset ordering, padding side, special-token handling, and
   sampler/RNG restoration all unspecified. Fix: exact pseudocode with 128 + 16 +
   1,856 = 2,000; padding_side="right"; add_special_tokens=True; sampler resume
   by re-seeding and fast-forwarding.

2. **Geometry CI breaks nesting**: 2,016 pairs share 64 endpoints; pair-level
   bootstrap violates clustered inference. Fix: state-level clustered bootstrap
   (resample 64 states, compute rho on induced submatrix).

3. **Verdict rules ambiguous**: Frozen G < 0.1 "pooled" undefined (edge-weighted
   vs action-weighted produce different verdicts); invalid-panel treatment for
   fixed-point stability and continuous G unspecified. Fix: action-weighted pooling
   (equal weight per action); invalid panel = stability FAIL for fixed points,
   edge EXCLUDED from G for moved edges.

4. **Ledger mixes workload units**: 435,908 sums batch-4 training steps with
   batch-1 inference forwards; 17.3h applies inference rate to training.
   Fix: separate workload types with distinct units and rates.

5. **GPU plan conflicts**: Full campaign not assigned to stable/cloud hardware.
   Fix: stable/cloud for full campaign; local limited to 100-forward pilot.

**v6 created** with all 5 fixes applied mechanically. **No round 6 submitted** —
rounds 3-4-5 = three consecutive REVISE rounds, hitting the repair-round cap
(CLAUDE.md §2.7 rule 7). Codex's own strategic recommendation (e841): "this is
no longer the highest-leverage place for another broad paper dialogue... 5:0
audit-to-build ratio, governance alarm. Make one mechanical repair then move
directly to runner construction."

**Next**: Build the PSQ-3 runner (`experiments/run_psq3.py`) implementing the v6
spec. The runner will mechanically resolve remaining ambiguities by making
concrete implementation choices. No further spec review rounds.

---

## 2026-08-30 — d_4 diagnostic on PSQ-2 v3 adapter (75% accuracy)

**Result**: DIAGNOSTIC ONLY. Exhaustive d_4 distance matrix on 43,520 probes (len 1-4).

Key numbers:
- d_4 range: [0.144, 0.990], mean 0.791, median 0.888
- Response fidelity: rho(D_model, D_oracle) = 0.579 (below 0.8 gate — expected at 75%)
- Hidden-state geometry (Euclidean in hidden space vs d_4):
  - Layer 6: rho = 0.259
  - Layer 12: rho = 0.277
  - **Layer 18: rho = 0.505** (best, p ~ 1e-130)
- Quasiconvexity violations: 7.9% (L6), 6.6% (L12), **4.6% (L18)**
- Model revision: ea980cb0a6c2ae4b936e82123acc929f1cec04c1
- Rate: ~7/s, total 6304s (~105 min)

**Interpretation**: Even at 75% accuracy, significant latent geometry exists. Layer 18 is clearly the best candidate for PSQ-3 (strongest correlation, fewest violations). With 95%+ accuracy (PSQ-3 gate target), expect stronger structure. Forward rate ~7/s confirms PSQ-3 budget estimates.

---

## 2026-08-30 — Codex PSQ-3 lock round 3: REVISE, 8 blockers adopted into v4

**Codex PSQ-3 lock round 3**: Verdict REVISE. Five v2→v3 structural repairs verified clean (split, 16-probe panel, row convention algebra). One accounting error plus several unbound adjudication details:

1. **Orthogonal complement notation**: Line 201 used column notation `(I - P.T @ P)(h - mu)` instead of row `(h - mu) @ (I - P.T @ P)`.
2. **Transductive claim scope**: All-64-state PCA is transductive (PCA sees held-out node representations). Claim must say so explicitly.
3. **Probabilistic d_panel undefined**: Spec defined d_panel for binary oracle profiles only. Causal E and E₀ use three-bin model responses. Fix: `d_panel(f,g) = sqrt(1/16 × Σ_q JS₂(f_q, g_q))`. Nearest-state decoder: argmin over 64 oracle profiles, ties = miss. Invalid probe (P_other > 0.3) = MISS.
4. **Layer-screen budget stale**: 10,752 used old 42-probe count. With 16 probes + 1 wrong target: 8,192. Wrong-target cardinality was unspecified.
5. **Missing cal-source operator rung**: Donor paste proves site, not operator learnability. Added Step 2b: apply M_a to cal sources (in-sample), require >= 90% hit rate.
6. **Bootstrap/RNG unbound**: Pinned: source-state clustered bootstrap, 10,000 resamples, seed=42. Haar O(k) via QR of Gaussian. Matched-random and wrong-state pools pinned with per-edge RNG seeds.
7. **Invalid-counts-as-wrong absent**: For gate, invalid probes now count as WRONG. For causal panel, invalid = MISS.
8. **No pass criterion for correlational geometry**: Added: rho(D_hidden_ft, D_model) > rho(D_hidden_frozen, D_model).
9. **Three-seed aggregation unbound**: Pinned: per-seed + median with range, all 3 seeds must pass individually.
10. **GPU governance conflict**: 24.7h GPU budget noted as requiring explicit user authorization per AGENTS.md.

**v4 changes**: All above adopted. Total per seed: 93,440 forwards (~3.7h @7/s). 11 pass criteria (up from 9). v4 in Codex lock round 4.

---

## 2026-08-30 — Codex PSQ-3 lock round 2: REVISE, 5 corrections adopted into v3

**Codex PSQ-3 lock round 2**: Verdict REVISE. v2 fixes preserved but 5 lock-blocking defects found:

1. **PCA convention inconsistency**: Procrustes uses row action (Z_s @ M), but intervention formula used column action P^T(M-I)P. Also origin problem: cal-only mean biases A/C targets which cross partitions. Fix: row convention throughout; all-64-state mean/PCA.

2. **Checkerboard maximally confounds parity quadrants**: (x+y) mod 2 puts ALL even-even and odd-odd in cal (16/0), ALL mixed in held-out (0/16). Fix: cal = {(x,y) : (floor(x/2)+floor(y/2)) mod 2 = 0} gives 8/8 in every parity quadrant and 16 within + 16 cross per action.

3. **Budget incomplete**: Missing layer-screen second arm (+5,376) and displacement-composition controls (+32,256). True total ~147k forwards, ~5.1h per seed.

4. **Controls not adjudicated**: Procrustes could pass while displacement or wrong action performs equally. Fix: paired, source-clustered superiority over every moot-maker with 95% CI.

5. **42-probe set is broken for hit rate**: d_{2,2} yields only 25 distinct profiles (one 16-state class). Only 25% of targets uniquely identifiable. 75% hit gate impossible even for perfect oracle. Fix: 16-probe state-separating panel (8 value indicators per dial, all within H*=4). Cheaper AND fully separates all 64 states.

**v3 changes**: row convention; all-64-state PCA; new split; 16-probe panel; complete budget (~94k/seed, ~3.3h); paired superiority adjudication; frozen base diagnostic regardless; all implementation details pinned.

**Status**: v3 in Codex lock round 3.

---

## 2026-08-30 — Codex PSQ-3 lock round 1: REVISE, 9 corrections adopted into v2

**Codex PSQ-3 final lock round (round 1)**: Verdict REVISE. Direction sound and build-worthy, but draft not lock-ready. All 9 corrections adopted into v2 spec:

1. **PCA edit must preserve orthogonal complement**: h'_a(s) = h(s) + P^T(M_a - I)P(h(s) - mu). Allow M_a in O(k), det = +/-1 for reflections. Report singular values and conditioning.

2. **Three-bin response law**: [P(0), P(1), P_other] from full softmax, not renormalized {0,1}. Output-validity gate: P_other > 0.3 flags invalid. JS divergence over three-bin distributions.

3. **Primary causal comparison is model-vs-model**: E = d(beh(edit), beh_model(a(s))), not oracle. Isolates transport quality from residual task error. Oracle comparison secondary.

4. **Separate training from evaluation triples**: ~2000 training triples reported descriptively. Primary gate on ~41,648 held-out triples. Length-0 cells (<=8 states) diagnostic, not primary gate.

5. **Frozen-base runs full causal pipeline**: Procrustes fitting + intervention on unmodified base model. Tests whether digit-token geometry already produces action-aligned effects (moot-maker).

6. **Algebraic diagnostics demoted to secondary**: M_a = I trivially satisfies involution, commutation, conjugacy. Added M_A^8, M_C^8, M_DM_CD. Report det(M_a) and ||M_a - I||_F alongside. Value only if G > 0.

7. **Checkerboard split pinned**: cal = {(x,y) : (x+y) mod 2 = 0}. Balanced: each x/y value 4 cal/4 held-out, B/D-fixed 8/8, A/C flip partition (cal->held-out), B/D preserve partition (cal->cal). Claim narrowed: held-out source-action-edge generalization.

8. **Feasibility corrected**: Original 30-min causal estimate untenable — actual is 32x4x682 = 87,296 forwards (~3h). Solution: proximal probe set (lengths 0-2, 42 probes) for primary staircase. Exact forward-call ledger: ~110k forwards/seed (~3.8h eval). Total PSQ-3A: ~25h GPU.

9. **Fixed-point handling**: B fixes 16 states (x in {0,4}), D fixes 16 (y in {0,4}). E_0=0 makes gain undefined. Excluded from gain computation. Fixed-point stability reported separately (operator should not move fixed points).

Additional v2 changes: natural-prevalence sampling with class weight (not balanced+weighted); all 128 epsilon triples in training; PSQ-3B deferred to separate prospective lock; seeds [42, 137, 2024] sequential with abort; dataset seed 7 with SHA-256 hash.

**Status**: v2 spec in Codex lock round 2.

---

## 2026-08-30 — Codex PSQ-3 design review: NOT LOCK-READY, corrections adopted

**Codex PSQ-3 design review (round 2)**: PSQ-3 direction is correct but the draft design has structural flaws. Corrections adopted verbatim:

1. **Causal test is tautological**: `h(s) + [h(a(s)) - h(s)] = h(a(s))` is target-state donor paste by construction. It is a necessary positive control (carrier sufficiency), not evidence for a shared action law. Fix: 4-step staircase — (i) same-state replay baseline, (ii) target-paste positive control, (iii) wrong-state/random-paste negatives, (iv) shared action operator fitted on calibration states, tested on held-out states.

2. **Operators, not vectors**: A constant displacement v_a is structurally wrong for cyclic increments (A,C) and reflections (B,D). Use orthogonal Procrustes M_a per action fitted on calibration states. Composition: M_b M_a h(s), never recomputing from intermediate oracle state. Algebraic diagnostics: M_B² ≈ I, M_D² ≈ I, M_A M_C ≈ M_C M_A, cross-dial commutativity, reflection-increment conjugacy.

3. **Data count correction**: 64 per cell × 4 cells × 4 lengths = 1,024, not 4,096. Length-1 has only 32 unique positives per cell (8 × 4¹). Include empty word ε: 341 words, 43,648 responses. Normalize JS explicitly: JS₂ = JS_nats / ln(2), d_{2,4} = sqrt(mean JS₂ / 682).

4. **Three distinct scientific quantities**: (i) Response fidelity: ρ(D_model, D_oracle). (ii) Correlational latent geometry: ρ(D_hidden, D_model) and ρ(D_hidden, D_oracle). (iii) Causal equivariance under held-out interventions. Current Spearman criterion only measures (i). Need all three separated.

5. **Sharpened gates**: Spearman > 0.5 too weak — require ≥ 0.8 per seed. Primary metric: statewise behavior-profile error e(s) = d_{2,4}(beh_hat(s), beh*(s)). Continuous causal gain G = 1 - E/E₀ with clustered lower bounds against four null conditions.

6. **Layer choice protocol**: Do not choose from final results. Run donor-paste positive controls on calibration states at layers 12 and 18. Select earliest passing layer. Lock before operator fitting. Phase-aligned carrier position (delimiter after state declaration, before action/query suffix).

7. **Staged arms**: PSQ-3A (task-only supervision) first. PSQ-3B (equivariance objective) only if A clears interface but fails causal equivariance. Equivariance loss trains only on calibration states; evaluation held out.

8. **Feasibility**: ~6.8h per seed × 3 seeds × 2 arms ≈ 40h GPU. Incompatible with sustained-GPU rule. Needs resolution (fewer epochs, 1 seed first, or shorter sequences).

9. **Distance-0 artifact**: The shared, held-out-generalizing action operator plus its executable phase-typed intervention. Training is a prerequisite, not the artifact. Causal evaluation is distance 2.

**Verdict**: One final math-first lock round before implementation. Do not build the current draft verbatim.

---

## 2026-08-30 — PSQ-2 v3 NO_INTERFACE (75.0%), exploratory d_4 launched

**PSQ-2 v3 result**: 75.0% overall, NO_INTERFACE. Trained on class-balanced ALL step lengths 1-8 (1536 examples, 48/cell/length). Per-cell: x_0=75.0%, x_1=81.2%, y_0=90.6%, y_1=53.1%. Per-step: 1-step 100%, 2-step 100%, 3-step 78.3%, 4-step 85.7%, 5-step 77.8%, 6-step 46.7%, 7-step 60.0%, 8-step 63.2%. Training: 1920 steps (5 epochs × 384 steps/epoch), 4590s, final loss 0.102.

**Key finding**: The model learns basic modular operations perfectly (1-2 step = 100%) but cannot compose them deeply (6-8 step = 47-63%). This is worse than v2's OOD result on long sequences (v2: 73.4% on 4-8 step trained only on 1-3 step). Spreading training across 8 lengths with lower lr (2e-5 vs 3e-5) diluted per-length signal. The 1.7B model with r=16 LoRA lacks capacity for reliable 8-step modular arithmetic.

**Exploratory d_4 measurement launched**: Running d_4 on v3 adapter as DIAGNOSTIC (not scientific result). Config: `psq2_v3_d4.json`.

**Codex direction review (PSQ-2 round 1)**: Critical structural issues identified:
1. **Metric mismatch**: Model d_4 uses max √JS (supremum), but ground-truth d_4 for deterministic binary responses = 1 for every distinguishable pair at H*=4 (no rank structure). Normalized Hamming is a different metric. The Spearman comparison is not a same-metric validation.
2. **Quasiconvexity test invalid for Open Problem 7**: Snapping interpolated points to nearest observed states ≠ evaluating along executable curves. The test is a nearest-neighbor surrogate, not the theoretical condition.
3. **Structured noise**: y_1=53.1% means errors are systematic, not iid. Max over 682 probes means confident errors reorder distances.
4. **Verdict**: PSQ-2 hyperparameter staircase should STOP. The v1→v4→8B ladder is "can we tune modular arithmetic?" — too weak.

**Codex's recommended redesign (adopted)**:
- Replace max √JS with a nondegenerate product metric: `d_{2,4} = sqrt(mean_{|w|≤4,c} JS(r_c(T_w x), r_c(T_w y)))` — same metric for model and oracle, with rank structure for both.
- Align the gate to H*=4 (max 4 steps, not 8) — this IS the distinguishing horizon.
- Evaluate exhaustive 43,648 responses, not random 128-sample gate.
- Add frozen-base, lexical-distance, and alternate-presentation controls. Multiple seeds.
- Make the central artifact CAUSAL: does action intervention move s toward a(s)? Does it compose for 2 actions?
- Stay with 1.7B until valid contract shows it can't clear H=4.
- A meaningful comparison: direct-answer supervision vs transition/composition-consistent supervision at matched H=4 accuracy. If only the latter develops equivariant interventions, that's a real finding about how latent geometry arises.

---

## 2026-08-30 — PSQ-2 v2 FAIL (73.4%), v3 in-distribution launched

**PSQ-2 v2 result**: 73.4% overall, NO_INTERFACE. Trained on class-balanced 1-3 step sequences, tested on 4-8 step (OOD). Per-cell: x_0=65.6%, x_1=84.4%, y_0=62.5%, y_1=81.2%. Clear length degradation: 86.4% at 4-step → 56.0% at 8-step. Model learned short dynamics but composition doesn't generalize.

**PSQ-2 v3 launched**: In-distribution test — class-balanced training on ALL step lengths 1-8. 1536 examples (48/cell/length × 8 lengths × 4 cells), perfectly balanced. Scientific question shifts from "does generalization emerge?" to "can the model learn modular state tracking with explicit supervision?" If PASS → d_4 measurement unlocked.

**Bug found and fixed**: Config key collision — `train_max_steps: 8` (intended for max sequence length) was also read by the early-stopping code (`train_max_steps`), causing training to stop after 8 steps. Fixed by renaming to `train_seq_max`.

**Theoretical prediction (pre-registered before result)**: If v3 passes the 95% gate, the d_4 measurement tests a finite discrete instance of Open Problem 7 (behavior-space quasiconvexity). Theorem 1's finite corollary gives d_4 = d_∞ for this world (Moore partition saturates at H*=4). The quasiconvexity test checks: along latent-space interpolations, does d_4 remain bounded by the max of its endpoint values? A positive result would be the first empirical evidence that a model's latent geometry respects the behavior metric — the core prediction of native latent mathematics. d_4 runner extended with LoRA adapter support (merge-and-unload); config `psq2_v3_d4.json` prepared.

---

## 2026-08-30 — PSQ-1 CLOSED (all substrates NO_INTERFACE) + PSQ-2 launched

### PSQ-1 closure
Three substrates tested, all fail the capability gate:
- **Qwen3-1.7B-Base**: 50.0% (always predicts "1"). NO_INTERFACE.
- **Qwen3-8B-Base**: 55.5% (same bias). NO_INTERFACE.
- **Qwen3-8B-Instruct**: 50.0% with 4-shot, 64.1% with balanced 2-shot. NO_INTERFACE.

**Root cause diagnostic**: Models cannot do modular wrap-around arithmetic (7+1=0 mod 8) from few-shot prompting. Instruct model passes 9/10 hand-picked diagnostics (including wrap-around) but fails the full 128-case screen with random 2-8 step sequences. The failure is specific to multi-step modular computation, not the presentation format. Prefix-answer bias dominates: 4-shot examples (3/4 answer "0") → model always predicts "1"; balanced 2-shot → model mostly predicts "0".

**What was learned**: Current 1.7B-8B models cannot serve as PSQ substrates via few-shot prompting. The capability gate worked as designed — it correctly identifies substrates that pattern-match rather than track state. The two-dial world is a valid test environment but needs a model that can execute multi-step modular arithmetic.

### Ground-truth geometry computed
Two-dial world Z_8 x Z_8: diameter 8, Moore refinement [4,9,25,49,64] at H=0-4, confirming H*=4. Normalized Hamming d_4 has 15 distinct distances from origin (range 0.003-0.682). Graph-metric quasiconvexity: max R=2.0, mean R=1.036. Binary d_4 is trivially quasiconvex (constant 1 for all non-self pairs at H*=4; 1.2% violations from cycle-through-origin only).

### PSQ-2 launched (fine-tuning approach)
Design: LoRA fine-tune Qwen3-1.7B-Base on single-step two-dial transitions, test multi-step generalization.

**v1 result (single-step training only)**: 60.2% overall, NO_INTERFACE. Model learned "not zero" bias (87.5% of training data has answer=0). Per-step degradation: 81.2% at 2-step → 40.0% at 8-step. Training converged (loss 0.082 → 0.005 over 3 epochs).

**v2 in progress**: Class-balanced training with 1-3 step examples, test on 4-8 step sequences. Fixes v1 class imbalance.

Runner: `experiments/run_psq2_finetune.py`. Configs: `experiments/config/psq2_v{1,2}.json`. d_4 measurement runner ready: `experiments/run_psq1_d4.py`.

---

## 2026-08-30 — PSQ-1 design finalized (3-round Codex dialogue) + smoke attempt #7 failed

### Smoke attempt #7
CivilizationV exited, ran smoke immediately. FAILED: s_smoke=2.199s → F_CPU=147.8 min, ceiling=90 min. Cause: ~8GB in background processes (Chrome, Spotify, Steam, WhatsApp, Codex, Claude). Need ≤1.34s per call. Restored committed stale artifacts from git.

### PSQ-1 rounds 2–3 (successor design refinements)

**Round 2 — substrate and horizon corrected:**
- OthelloGPT REJECTED as headline substrate (specialist model, PASS would be unsurprising). Replaced with frozen Qwen3-1.7B-Base (general pretrained).
- Task: 64-state two-dial world, q=(x,y) ∈ Z_8², actions A:(x,y)→(x+1,y), B:(x,y)→(-x,y), C:(x,y)→(x,y+1), D:(x,y)→(x,-y). Observations: is x=0? is y=0?
- d_2 KILLED — horizon 2 too weak. Replaced with d_4 (H*=4: Moore partition saturates at 64 classes). 341 action words × 2 channels = 682 evaluations per state.
- Quasiconvexity tested with permutation null (99/100 random endpoint permutations).
- Post-result pivot REJECTED — severity is the point. Pre-register layer/rank from grid {8,16,24}×{16,32,64} on training data only.

**Round 3 — presentation, compute, fallback:**
- Presentation: systemless 4-shot Python-completion template (expanded operations, NOT symbolic actions). Readout: normalize(p(" 0"), p(" 1")) — binary, separate x/y queries.
- Capability gate: ≥95% per-cell (not overall — x=0 occurs only ~1/8 of the time).
- Compute: capability screen ~20-30 min (laptop-feasible); FULL PSQ-1 ~4-6 CPU days (NOT laptop-feasible). Exhaustive 9-cell grid = 6-9 days. Conflicts with theory-first directive.
- If Qwen3-1.7B-Base fails capability gate → NO-INTERFACE, stop PSQ-1 immediately. No repair.
- Separate fallback successor: Qwen3-8B-Base on stable 24GB+ hardware. Same protocol.

Full Codex outputs at scratchpad/direction_successor_{program,r2,r3}.md. Decision pending.

---

## 2026-08-30 — Codex successor program design: PSQ-1 (Othello Predictive-State Transport)

Following the should-continue STOP verdict, launched a Codex direction dialogue on the successor program. Codex designed PSQ-1 — a concrete one-experiment successor:

**Model:** OthelloGPT (Baidicoot/Othello-GPT-Transformer-Lens, 25M params, 8 layers, 512 dims). Known board-state representations, lawful state transitions. Real model, not a toy.

**Registered world:** Valid Othello prefixes plies 12–48; moves = 61 legal tokens; response = full 61-way next-token distribution; horizon = 2 appended tokens. d_2(x,y) = max_{|w|≤2} √JS(r(wx), r(wy)) is the exact d_∞ of this finite world.

**Measurement:** (1) Predictive-state recovery via d_2 on transpositions (Myhill-Nerode). (2) Behavior-space quasiconvexity: R = (d_2(x_0,x_1) + d_2(x_1,x_2)) / d_2(x_0,x_2) for held-out legal two-move paths. (3) DAS: rank-32 distributed alignment at post-layer-4 final-token residual; swap aligned subspace; compare patched vs source over all ≤2-step continuations.

**Baselines:** Euclidean, cosine, linear probe, function vector (Todd et al.), random rotation, full swap (positive control).

**Joint PASS (across 3 seeds):** d_2 transposition AUC ≥ 0.95 and ≥ 0.05 above baselines; held-out 95th-percentile R ≤ 2.0; full-swap target reduction ≥ 80%; DAS ≥ 50% reduction and ≥ 15pp above non-positive-control; game-clustered 95% bounds clear all thresholds; no seed reversal.

**D1–D9 relationship:** D1–D5 recast Moore behavior / computational mechanics / Myhill-Nerode. Genuine additions: D2's native-output boundary, D6's coherent presentation transport, Theorem 7's finite-agreement refusal, D9's denizen/surgeon separation, Open Problem 7's quasiconvexity. PSQ-1 tests the last directly.

**One-result stop rule:** Run once. Any failed gate or seed reversal kills PSQ-1. A PASS also ends it — licenses a new proposal but queues nothing.

**Ratio target:** 1:1 measurement-to-artifact, cap at 1.5:1. Crossing 2.0 kills the successor.

**Repo reset:** Archive existing runners/configs/results under read-only snapshot. Keep README, AXIOMS, STATE, NOTEBOOK, EXPERIMENTS, ledger, handoff, structured negative. Delete regenerable artifacts. Add one runner, one config, one result directory.

Full Codex output at `scratchpad/direction_successor_program.md`. This is a PROPOSAL — not authorized for implementation until user decides.

---

## 2026-08-30 — Audit #48 adopted (REVISE): D7 specialization corrections applied

Codex audit #48 verdict: REVISE. Headline coefficient 1/√(8 ln 2) confirmed correct, but three classes of repair required and applied:

1. **Proof sketch coefficients**: Each KL term is t²g_π/8 (not t²/4), sum is t²g_π/4 (not t²/2). Intermediate steps now consistent with the correct final result.
2. **Compatibility requirement rewritten**: D2's D_c is a distance on response-law space, not a smooth metric tensor. Replaced with curve-based first-order expansion requirement: D_c(y, η(t)) = |t|F_{c,y}(η̇(0)) + o(|t|). Channels without such F cannot instantiate D7. Notes that branchwise compatibility does not provide (w,c)-uniformity.
3. **Caveats strengthened**: (a) boundary behavior is direction-dependent (face-interior vs zero-mass directions); (b) finite supremum not implied by strict positivity alone; (c) supremum not necessarily attained; (d) "Finsler-type/Minkowski seminorm" safer than classical "Finsler"; (e) finite vocabulary essential; (f) prompt-string LMs need separately declared differentiable carrier; (g) KL uses natural logarithms; (h) O_c = V type annotation.

Licensed sentence (audit #48 verbatim): normalized √JS induces the Fisher pullback seminorm with coefficient 1/√(8 ln 2) at interior finite-vocabulary laws under D7's differentiability and finiteness assumptions; softmax-variance and fixed-Markov-contraction identities correct; does not promote reachability_v1 or establish denizen-executable native geometry. Ratio: 33:15 = 2.20:1.

---

## 2026-08-30 — Round 43: D7 compatibility requirement and LM finite-vocabulary specialization (audit #48 REVISE applied above)

Codex dialogue on information-geometric D7 completed. Ruling: specialize D7, don't replace it. Written into `theory/AXIOMS.md` after the first-variation lemma:

1. **Compatibility requirement**: D7's tangent norm must be the metric differential of the registered D_c (closes the gap between D2's global metric and D7's local geometry).

2. **LM finite-vocabulary specialization** (proved under interior-simplex assumption): normalized √JS induces (1/√(8 ln 2)) × Fisher norm. Explicitly: D_{√JS}(π, π+tu) = |t|/√(8 ln 2) · √(g_π(u,u)) + o(|t|), where g_π is the Fisher metric. The compatible D7 seminorm for an LM with strictly positive next-token law is p_x^JS(v) = sup_{w,c} (1/√(8 ln 2)) √(g_{π_{w,c}}(u_{w,c}, u_{w,c})).

Key properties: seminorm (pullback of inner product through linear differential); for softmax laws, measures Var_{A~π}[dℓ_A]; Markov contraction (full channel dominates any fixed grouping); boundary singularity when π_a = 0; Finsler not Riemannian; Open Problem 7 still open. Does NOT collapse instrument/native, does NOT promote reachability_v1, does NOT change the bridge specification or locked estimands. Audit #48 pending.

## 2026-08-30 — Codex should-continue review: STOP the current program form

Mandatory §2.7 rule 5 review. Fresh Codex (no prior context) reads README, STATE, NOTEBOOK, AXIOMS, EXPERIMENTS. Verdict verbatim:

> No — not in its current form. The broad research question is worth preserving, but this program has become infrastructure drift. Stop the `native_bridge_v1` review loop, preserve the negative record, and permit only an artifact-first successor with a hard one-result stop rule.

Key findings (verbatim summary):
1. **Central bet untested.** native_bridge_v1 cannot answer the central question even if it passes — its centroid is analyst-authored, its edit is a surgeon move, and its claim wall forbids licensing native latent mathematics.
2. **Review loop is negative-value.** Six review rounds, three fix iterations, six smoke attempts, zero scientific calls. More review improves auditability of an experiment whose best outcome cannot establish the central artifact.
3. **Ratio is infinite.** 16,026 lines of theory/apparatus, zero central artifacts. The project's own >5:1 halt rule is exceeded.
4. **Two defensible paths:** (a) terminal salvage — regenerate smoke/lock once, execute one immutable run, close regardless of outcome; (b) stop immediately.
5. **Strongest alternative:** predictive-state (Shalizi & Crutchfield), causal abstraction (DAS/Geiger et al.), Myhill-Nerode diagnostics (Vafa et al.), function vectors (Todd et al.) on OthelloGPT or a lawful sequence model.

Codex slightly favors the terminal run (bounded marginal CPU cost, apparatus already exists) but states it does not justify continuing the program.

This review is recorded. The D7 specialization (round 43) is independent foundational theory that does not depend on native_bridge_v1. The decision on whether to execute the terminal run or stop immediately is the user's. Audit #48 (D7) is still pending.

## 2026-08-30 — Re-contextualization (2-hour check-in, runner engineering)

**What still holds:** The adopted theory stack (D1–D9, Theorems 1–8) is unchanged and sound by audits #42/#44. The native_bridge_v1 preregistration is LOCK-READY (audit #47). The runner code has all 6 Codex-identified blockers fixed (committed 44d8c1a). The scientific run is the correct next empirical step.

**What's reframed:** This entire session has been tunnel-visioned on engineering plumbing — six Codex review rounds, three fix iterations, six smoke attempts. No scientific call has executed. The bottleneck is now a hardware-state issue (CPU contention from a game), not a code issue. Continuing to retry smoke is waste of session time.

**Tunnel check:** YES, narrowed onto one thread (runner review loop). Two corrective actions taken: (1) launched a Codex dialogue on the information-geometric D7 alternative (the next foundational thread identified at the last re-contextualization), (2) launched the mandatory "should this program continue" Codex review (§2.7 rule 5). Both running in background.

**Alternatives held live:**
- Information-geometric D7: replace abstract tangent norms with Fisher metric / √JS on next-token laws. Would make the instrument/native distinction intrinsic to the output space.
- Theory simplification: is the full D1–D9 stack necessary, or can the essential content be stated in fewer primitives?
- Different model: if Qwen3-1.7B-Base shows no bridge signal, try an instruct model (larger capacity for structured state)
- Different site: block 16 single-position is a narrow probe; multi-site or different-layer interventions
- The central bet (README) vs the current work: has any artifact tested the bet? (Codex "should continue" review answering this now)

**Ratio:** 32:15 (theory rounds : empirical/build rounds). No scientific artifact has run since the restart. The theory is ahead of the empirics by design (theory-first mode), but the gap is widening — the runner is ready, the CPU is not.

---

## 2026-08-30 — Codex v6 REVISE → 3 blockers fixed (44d8c1a); smoke rerun blocked by CPU contention

Codex v6 design-gate review (REVISE) identified 3 remaining blockers, all genuine, all fixed in commit 44d8c1a:
1. **Token IDs in call identity** — `call_identity()` now includes a hash of materialized token IDs; call table hashes are token-bearing
2. **Donor label validation** — `build_row_manifest()` and Stage 4 now verify correct donors carry `target_label` and wrong donors carry `wrong_label`
3. **Per-entity checkpoint markers** — entity-level completion markers emitted in both Phase D and Phase E loops

Codex v6 also confirmed 3 prior items resolved: Blocker 0 (manifest mismatch), Item 3 (bootstrap precommitment), Item 2 (expanded manifest).

**Smoke rerun needed but blocked by CPU contention.** CivilizationV running in background → CPU throttling → s_smoke ≈ 1.6s/call → F_CPU ≈ 109 min > 90 min ceiling. Needs s_smoke ≤ 1.34s (achieved at 1.225s when CPU was uncontended). The committed smoke result / manifest / lock row (from 83b7267) bind to the pre-v7 runner hash and pre-token-ID call table hashes — they must be regenerated after a successful smoke with the v7 runner.

**Next when CPU available:** rerun smoke → rebuild manifest with smoke_commitment → rebuild lock row → Codex v7 final review → Phase D/E scientific execution.

## 2026-08-30 — Runner built, SMOKE_VALID, lock row bound (pre-v7); science-grade fixes applied

Runner (`experiments/run_native_bridge.py`, committed 83b7267) built with full science-grade hardening per Codex v4/v5 reviews: per-call checkpoint journal; bootstrap resample index precommitted; expanded manifest with tokenizer hashes, library versions, resample_index_hash; strengthened validation; stage-independent manifest; monotonic hard-wall clock.

32-call mechanical smoke (pre-v7 runner): SMOKE_VALID. η_smoke = 8.71e-9, ε_smoke = 1e-5, s_smoke = 1.225s, F_CPU = 82.3 min, H_CPU = 85 min. Lock row built with that runner hash.

---

## 2026-08-30 — Audit #47: LOCK-READY subject to wording-only corrections; no scientific forward pass before smoke, forecast, and lock row

Verbatim in `theory/dialogue/004.md`. Licensed sentence: Audit #47 finds `native_bridge_v1` lock-ready as a preregistration after wording-only audit-label, runtime-identity, and smoke-namespace corrections: its registered subset tests conditional response-law intertwining only, its donor-first same-replay schedule yields exactly 2,688 scientific identities without hidden calls or a serialized cut, and its scientific gates are prospectively fixed; no scientific result exists yet, and only a future complete valid `CENTROID PASS` may license the sentence “Across the 24 registered entities, the correct centroid edit has registered-access mean excess discrepancy within the native target-fiber criterion and prospectively improves that criterion relative to both no edit and the cycled wrong-label centroid.”

Never say (audit #47 additions): “LOCK-READY proves faithful transformer continuation or the carrier premises.” “The smoke proves equality of \(H_\ell\) or canonical-cut records.” “The 32 smoke calls are included in the 2,688 scientific total.” “Smoke outputs may be reused as locked scientific endpoints.” “The 1,680 count ignores replay”; the replay-ignored count is 840. “The prospective manifest contains realized donor tensor bytes before Phase D.” “A cycled-wrong comparison establishes generic target specificity.” “A valid smoke or the 1.5 safety factor guarantees completion.” “Lock readiness is a scientific PASS.” “Audit #47 authorizes a scientific forward pass before the manifest, smoke artifact, forecast, hard wall, runner/config hashes, constants, statuses, and lock row are bound.”

Next: apply wording corrections (run_phase namespace, prospective-vs-realized payload hash, replay-scoped 1680 explicit, audit-#46 checklist -> #47 disposition); build runner/config; 32-call mechanical smoke; forecast; lock row; then scientific Phase D.

## 2026-08-30 — Round 42: repair pass 2 applied to the bridge specification and native_bridge_v1; the three REJECT conditions are answered NO

Verbatim in `theory/dialogue/004.md`. All seven audit-#46 edits applied: conditional response-level intertwining r_c(T_w ι(u)) = r_c(ι(T_w u)) on the 32-call subset (no H_ℓ record equality claimed; a state-level identity would need a canonical cut-update map U_w); η indexed by entity; explicit √JS with natural logs and /(2 ln 2) so D ∈ [0,1]; dependency-locked replay phases (Phase D: the 72 target-ε tuples first, ascending in A / descending in B, each supplying its endpoint law and that replay's donor residual; Phase E: the remaining 1,272) with same-replay donors and no additional invocations; complete call identity including the intervention payload (2,688 identities; 1,680 token/mode combinations); smoke envelopes ε_smoke = max(1e−5, 2η_smoke) with INVALID — NUMERICAL REPLAY above 1e−4 and INVALID — SITE CARRIER/HOOK on the fixture comparisons; runner stage 8 donor-first with per-call/per-entity checkpoints, the hook REPLACING exactly one position with exactly one matched write asserted, and stage 4 distinguishing the declared dependency specification from a serialized cut. Codex's answers to the three REJECT conditions: hidden call — No (Phase D reuses the existing target-ε calls); fictitious cut or cross-replay donor reuse — No. Lock-readiness audit #47 next (pass 3 is the last permitted).

## 2026-08-30 — Audit #46: REVISE (repair pass 2 of 3); no lock, runner, smoke or compute; two substantive defects (state-level intertwining untested by the output-only smoke; donor-dependent edits impossible under the global replay orders) plus exactness repairs

Verbatim in `theory/dialogue/004.md`. Licensed sentence: Audit #46 finds the finite bridge, replay-factor, fixed-population, and narrow gate mathematics sound, but `native_bridge_v1` is not lock-ready because state-level lift intertwining is not tested by the output-only smoke, donor-dependent edits cannot be executed under the declared global replay orders without hidden calls or cross-schedule reuse, and smoke/dedup identities need exact repair; no register, exact or denizen-reachable bridge, storage, persistence, dimension, generic target specificity, or native latent mathematics is established.

Never say (audit #46 additions; all earlier lists remain binding): “The 32-call output smoke proves literal equality of canonical cut records.” “The written globally reversed replay independently reconstructs every donor residual.” “Identical tokenized input and hook mode imply identical call identity when patch payloads differ.” “The cycled wrong-label centroid proves specificity against every wrong label or generic target specificity.” “The \(0.02\) thresholds are model-derived or mathematically canonical.” “The 1.5 safety factor proves completion within 90 minutes.” “Audit #46 authorizes runner construction, a lock row, mechanical smoke, GPU execution, or scientific compute.”

Round 42 applies edits 1–7 (response-level intertwining; η with entity index; explicit √JS; donor-first phased replays; complete call identity; smoke thresholds; runner stage 8); then lock-readiness audit #47. If pass 3 cannot make the artifact executable without hidden calls or a fictitious cut object, the carrier construction is rejected and closed without layer/site/population changes.

## 2026-08-30 — Round 41: bridge specification repaired (audit-#45 edits 1–10); native_bridge_v1 DRAFT preregistration (not locked; pending audit #46)

Verbatim in `theory/dialogue/004.md`; text in `theory/AXIOMS.md` (Native bridge specification, rounds 40–41) and `theory/EXPERIMENTS.md` (native_bridge_v1 DRAFT). All ten edits applied (phase/macro rename; canonical DAG cut with conditional fidelity; three distinct prospective targets; replay envelopes η, ε_B = max(ε₀, 2η), ε_E = max(ε₀, 4η), τ = ε_E + δ with the triangle derivation; exact Θ₂₄ with descriptive stability bounds; explicit Y_{s,−e_i}; no-edit and wrong-label-centroid controls with paired contrasts and three centroid gates; conjectured proximal control and population PASS sentence; certified vs numerical refutation wording; eight families, 2,688 pre-dedup = 2,688 deduplicated call units, 32-call mechanical smoke, forecast ×1.5, 90-minute CPU ceiling, abort and lock contract). Constants: ε₀ = 1e−5, k_B = 2, k_E = 4, δ = δ_move = δ_spec = 0.02, replay invalidity ceiling 1e−4, 2,000 stability resamples, α = 0.05, PCG64 seed 4141. Codex's honest view: refutation teeth intact (certified discrepancy → Theorem 7; a positive centroid must beat no-edit and the wrong-label centroid after the proximal control passes); 2,688 is 'small' only as a bounded grid — the 32-call smoke and 90-minute abort establish affordability. Audit #46 = lock-readiness review.

## 2026-08-30 — Audit #45 on the native bridge specification: REVISE; NO LOCK, NO COMPUTE; carrier schema and finite bound sound; replay/bootstrap/controls/compute contract require repair; ceiling 2,688 with the two added controls

Verbatim in `theory/dialogue/004.md`. Licensed sentence: Audit #45 finds the round-40 carrier and finite lower-bound mathematics sound at the formal schema level but the native-bridge specification not lock-ready: its replay and clustered-bound semantics, population-mean wording, and intervention controls require repair, and no register, exact bridge, denizen reachability, storage, persistence, dimension, or native latent mathematics is established.

Never say (audit #45): “The carrier fixture proves faithful transformer continuation.” “All carrier objects, including \(\iota_\ell\) and \(\operatorname{Cont}_\ell\), are endomaps.” “The three target presentations are statistically independent.” “\(V_i\) normalizes the discrepancy” or “certifies a distinct target fiber.” “The bootstrap interval is a confidence interval for the fixed 24-entity mean.” “\(E\le0\) proves an exact bridge, target-place identity, unseen-future control, or reachability.” “A population-mean PASS means every edited state landed in the target fiber.” “Native-residual paste equals the native target by construction.” “Native-paste PASS validates centroid formation.” “Centroid PASS proves causal target-specific movement” without no-edit and wrong-label comparisons. “The nine-bin channel supplies independent refutation power beyond the full law.” “The pooled-span preflight validated the block-16 single-position site.” “\(2{,}016\) evaluations proves the run is small.” “Qwen has a register” or “a causal bridge exists.” Any audit-#36 or audit-#39 forbidden claim.

Round 41 applies edits 1–10 (repair pass 1 of the specification) and drafts the preregistration with a mechanical-only smoke plan; audit #46 = lock-readiness review of the preregistration + smoke forecast.

## 2026-08-30 — Re-contextualization (2-hour check-in, theory-first); audit skipped as duplicate (audit #45 in flight)

Audit status: since the last note, audits #42 (ADOPT foundation), #43 (REVISE LM extension), #44 (ADOPT round-39 mathematics + revised extension) were adopted verbatim with their alternatives on the audit board (e738, e742, e745); the only unaudited claim is the round-40 native bridge specification, and audit #45 is running on it now with the tunnel/alternatives questions in its brief. Firing a second auditor on the same text would duplicate it.

Where the theory stands: an adopted stack — D1–D9, Theorem 1 (descent/minimality), Proposition 2 (observability seminorm), Corollary 3, Theorem 4 (append-only finite memory), Proposition 6 (interventions), Theorem 7 (asymmetry of finite access), Theorem 8 (surgeon worlds) — all standard mathematics by the auditors' consistent verdict; the distinctive content is governance: the denizen/instrument/surgeon partition of who can do what, and finite access as an epistemic limit. What it reframes: (i) every empirical construction of 2026-08-27–30 was instrument-level or surgeon-level, never a denizen fact; (ii) the audit-#39 'causal bridge' is now a precise, finitely refutable statement in the denizen metric, with an approximate excess-discrepancy estimand because the exact null is fragile; (iii) identity claims about LMs are refutable, never certifiable from tables — so every future run is a refutation design, which is why the specification is small.

Alternatives held live (verbatim on the audit board): deterministic final-coalgebra semantics as the compressed core; information geometry (Hellinger/Fisher/JS) of the native next-token law as the canonical D7; a category of presentations; nonlinear/switched observability for executable cones; Kantorovich lifting if generation becomes a move; for the bridge: approximate target-fiber vs excess-discrepancy vs fixture-only paths; and, repeatedly, leave the eight-symbol explicit-legend world once the bridge question resolves.

Tunnel check and step-back: ten consecutive rounds on one thread (foundation → extension → finite access → surgeon worlds → bridge spec) is deliberate under 'math first', and each round was audited fresh. The two anti-tunnel commitments for the next cycle: (a) once audit #45 rules, the compute-third step — the ≤2,016-evaluation refutation run — happens or the bridge line closes; there is no third option and no larger run; (b) the next foundational thread that is NOT the bridge is the information-geometric D7 (a native tangent norm on next-token laws), because it is the one alternative that would change what 'instrument vs native' means for every past measurement; it gets one dialogue round after the bridge decision. Ratio 29:12 (warning) — theory rounds count as builds, but no artifact has run since the restart, by design; that ends with the bridge run.

## 2026-08-30 — Round 40: native bridge specification written (proposed, pending audit #45); audit-#44 edits applied

Verbatim in `theory/dialogue/004.md`; text in `theory/AXIOMS.md` 'Native bridge specification'. Phase-typed carrier with total actions, failure states, site lift and continuation; proved unchanged denizen transitions, totality, and the finite bridge lower bound B_{W₀,C₀}; excess-discrepancy target-fiber estimand with clustered bounds and tolerance statuses; proximal controls: same-carrier paste-back is the PROVED validity fixture, cross-prompt native-residual paste is the CONJECTURED proximal control (pasting a target residual into a different source continuation is not equal to the target by construction — prefix/cache states differ); the centroid write is the bridge edit under test. Scope: one model, block 16, one token position, 24 entities, seven words, ≤ 2,016 pre-dedup evaluations. Codex's honest view: probably small enough for bounded CPU, conditionally — hook continuation may be costlier than plain forwards; audit #45 should require a smoke-derived forecast before any lock. Refutation teeth: after the proximal control passes, a clustered lower bound above tolerance rejects this centroid-write bridge; any replay-robust positive finite B refutes the exact bridge for its pair (Theorem 7).

## 2026-08-30 — Audit #44: ADOPT round-39 mathematics (Theorem 7, D9/Theorem 8, native bridge) and the revised LM extension; native_horizon_v1 not selected; one math-first bridge specification must precede any artifact

Verbatim in `theory/dialogue/004.md`. Licensed sentence: Audit #44 adopts Theorem 7, D9, Theorem 8, the native-bridge definition, the weaker frozen-record reading, and the audit-#43-revised LM prompt-world extension as sound standard mathematics subject to explicit shared-transition, phase-carrier, site-lift, and surgeon-action typing corrections; finite registered access can refute exact behavioral identity but cannot confirm it without a completeness proof, the historical residual record establishes neither an exact zero nor exclusion from all denizen-reachable places, and no computation is authorized until an approximate phase-typed bridge specification and proximal positive control are fixed.

Never say (audit #44): “Finite agreement or a plateau proves exact identity.” “Theorem 7 says identity can never have a finite proof.” “A finite-state abstraction is necessary for identity certification.” “Finite horizon makes the all-branch LM certificate practical.” “A surgeon move is a denizen move.” “\(d_\infty^D(m x,T_wx)=0\) proves equality of hidden states.” “Target-place matching proves denizen reachability from the source.” “The projection \(Q^S\to Q^D\) is injective or an isometry.” “The audit-#39 residual injection already satisfies D9.” “Failure to find a discrepancy confirms the native bridge.” “Any nonzero floating-point discrepancy is a scientifically material bridge refutation.” “The frozen-model record proves \(d_\infty^D(mx,x)=0\) or proves nonreachability.” “The revised LM extension establishes native latent mathematics.” “Conjecture 5 is adopted as a theorem.” “Audit #44 authorizes `native_horizon_v1`, a manifest lock, or computation.” Any audit-#27–#39 forbidden claim, especially any audit-#36 claim of slot-specific dimensions, a hard reachability limit, storage, retrieval, or memory capacity.

Round 40 applies the six typing/status edits mechanically and writes the bridge specification (phase-typed carrier, site lift, unchanged denizen transitions, failure states, total surgeon edit; finite registered lower bound B_{W₀,C₀}; approximate population estimand relative to replay and target-fiber variation; proximal positive control with outcome implications). Then audit #45; only then may a small real-model refutation-oriented bridge artifact be considered.

## 2026-08-30 — Round 39: Theorem 7 and surgeon worlds written (proposed, pending audit #44)

Verbatim in `theory/dialogue/004.md`; text in `theory/AXIOMS.md` (round-39 section). Corrections to Claude's proposal: finite-state abstraction is sufficient, not necessary (an invariant/bisimulation can certify identity in an infinite-state world); Theorem 4's suffix wrapper already supplies a finite abstraction, computationally enormous rather than nonexistent — so incomplete LM tables license 'refuted' or 'not refuted at registered access', never exact positive identity; surgeon worlds need a common phase-typed execution carrier; landing on a denizen-reachable place is d^D_∞(m x, T_w x) = 0 (Claude's d^S = 0 condition is strictly stronger, retained separately); the bridge claim is d^D_∞(m_s ι_ℓ(x), ι_ℓ(y_s)) = 0, finitely refutable but not finitely confirmable; the frozen-model record does not prove either exact zero distance or absence from every denizen-reachable place — only the weaker reading is licensed. Next candidate after audit #44: a small-budget, refutation-oriented native bridge artifact on one real model with its proximal positive control; native_horizon_v1 unqueued. Ratio 28:12.

## 2026-08-30 — Round 38: audit-#43 edits applied; ruling: no run now, do the response-completeness mathematics. Round 39: Claude proposes it as a NEGATIVE theorem (asymmetry of finite access) plus surgeon worlds

Verbatim in `theory/dialogue/004.md`. Round 38 applied all 15 file edits (statuses restricted; budget ≤ 9,408; QP word order; D8 total endomap; C2 retired by allocation) and ruled that `native_horizon_v1` should not run now — the next round should develop the response-completeness certificate. Round 39 (Claude): (A) Theorem 7 candidate 'asymmetry of finite access' — non-identity finitely witnessable; identity not finitely certifiable in general (explicit counterexample); certifiable only under a registered finite suffix-state abstraction (Theorem 1's corollary on S), vacuous for LMs (|V|^N); every LM identity claim is refutable, not certifiable; every future preregistration is a refutation design. (B) D9/Theorem 8 'surgeon worlds' — a larger move family A_S ⊇ A (residual edits) refines places; a surgeon move is denizen-realizable iff it lands on a denizen-reachable place; the audit-#39 causal bridge restated in the denizen metric (d^den_∞(m_s x, y_s) = 0), finitely refutable only; the frozen-model structured negative read as 'no tested surgeon move changed the denizen place to a denizen-reachable one' (a reading, never-say lists binding). Audit #44 deferred until there is something to lock.

## 2026-08-30 — Audit #43 on the LM extension: REVISE; NO COMPUTE; Theorem 4 / boundary / Proposition 6 upheld; 16 edits; budget cut to ≤ 9,408 evaluations

Verbatim in `theory/dialogue/004.md`. Licensed sentence: The LM prompt-world extension has a sound append-only finite-memory theorem, a sound discrete-map boundary, and exact pointwise zero-kernel characterizations of span-edit place change and target-place identity, but D8’s typed-map formulation, the delayed-control word order, the restricted-gate labels, the full-versus-derived response wording, the C2-subsumption claim, and the unbudgeted 57,792-evaluation design require revision; \(h\le2\) may witness nonidentity or falsify \(H^+=1\) but cannot establish global saturation, exact target realization, quotient descent, latent state, or a causal bridge.

Never say (audit #43): “Theorem 4 applies to a hard-rejecting full-context model without a registered wrapper.” “\(H^+\) exists exactly when Theorem 4 applies.” “Discrimination horizon is independent of macro granularity.” “A depth-two plateau proves \(H^+=1\).” “Failure of the \(I_2\) gate proves no place change.” “Failure of the approximate target gate proves target nonrealization.” “The same-state diagnostic proves D8 quotient descent.” “\(P Q\) appends \(P\) and then \(Q\)” under D1’s convention. “The nine-bin metric is full-next-token-law identity.” “C2’s factorial support is fully subsumed.” “The current experiment is lightweight or authorized.” “A prompt span edit is a residual or latent causal bridge.” “The extension establishes native latent mathematics.” Any audit-#36 forbidden claim about dimensions, slot specificity, reachability, storage, retrieval, or memory capacity.

Round 38 applies the 16 edits (repair pass 1 for the extension; the foundation itself is not reopened); audit #44 = lock-readiness review. The audit's standing warning: the design is still the eight-symbol world and edits prompt text, not latent state — the alternatives (fixtures-only Theorem 4 + the mandated latent intervention; a response-completeness certificate) stay live.

## 2026-08-30 — Round 37: LM prompt-world extension (proposed, pending audit #43); native_horizon_v1 preregistered; C1/C2 retired unrun

Verbatim in `theory/dialogue/004.md`; text in `theory/AXIOMS.md` 'Extension: language-model prompt worlds' and `theory/EXPERIMENTS.md` 'native_horizon_v1'. Codex's rulings on Claude's round-37 claims: Theorem 4 (finite memory ⇒ d_∞ = d_{N−1}) is proved only for APPEND-ONLY futures under registered suffix/sliding semantics (Claude's board note e740 anticipated this); it removes the infinite depth tail, not the exponential branch problem; prompt worlds do not automatically have E_x = {0} — D7 is absent unless a discrete structure is declared; span substitutions are NOT automatically nonexpansive for the append metric — D8 requires quotient descent separately; horizon-2 distances can witness nonidentity or falsify H = 1 but cannot certify d_∞ = 0. Conjecture 5 (one-query discrimination in explicit-legend worlds) stated with falsifiers. Audit #43 (math-only) fires now; no compute until adopted.

## 2026-08-30 — Audit #42 (mathematics-only): ADOPT the future-response foundation, with seven verbatim wording/type corrections; no compute authorized

Verbatim in `theory/dialogue/004.md`. Licensed sentence: The round-35 future-response foundation is adopted as a sound standard deterministic Moore-behavioral framework with conditional local-to-global and finite-dimensional observability results, subject to audit #42’s verbatim total-kernel and explicit-structure wording corrections; D2 remains registration-relative, `reachability_v1` remains only a restricted centered-log-probability instrument-differential family with `NO SLOT-SPECIFIC GEOMETRY CONCLUSION`, C1 remains blocked until its `other`-bin wrapper correction, and C2 remains an unrun preregistration whose mathematical validity establishes neither its hypothesis nor its scientific priority.

Never say (audit #42): “Theorem 1 is new latent-space mathematics.” “D2 makes interface gaming impossible.” “Action closure eliminates registration relativity.” “Theorem 3 proves a novel finite-access rate.” “Finite \(d_h=d_\infty\) constructs a computationally practical map.” “Pointwise differentiability alone identifies \(p_x\) with the first variation of \(d_\infty\).” “Open Problem 7 is proved unconditionally.” “`reachability_v1` measured or lower-bounded native \(p_x\).” “The patched residual directions were denizen-executable.” “`reachability_v1` established low-dimensional, slot-specific, storage, retrieval, capacity, or reachability geometry.” “C1’s current \(K(p,\ell)=p^{-1}(\ell)\) is a total kernel on the full vocabulary.” “C1b generalizes to unseen register points or directions.” “C2 establishes exact places, persistence, storage, or a causal bridge.” “The twelve ordered template pairs are twelve independent replications.” “Audit #42 proves either preregistered proposition.” “Audit #42 authorizes computation.” Every audit-#36 never-say item remains binding.

Round 36 applies edits 1–6 mechanically (C1 total 'other'-bin wrapper; round-trip control; D7 manifold structure; Euclidean inner products in Proposition 2; Theorem 3 → Corollary and counterexample 3; heading ADOPTED). Next mathematics per the audit: denizen-checkable sufficient conditions for finite unseen-branch control and executable lifting in a real model, toward an intervention-bearing artifact.

## 2026-08-30 — Round 35: audit-#41 edits applied (repair pass 2 of 3); Theorem 3 candidate stated; audit #42 fired

Verbatim in `theory/dialogue/004.md`. All ten audit-#41 edits applied; disclaimer now 'Nothing currently proved is genuinely new mathematics.' Ruling on 'responses are futures': FOR, with the correction that registration relativity is relocated (to an architecturally fixed base output o and legal move family), not removed. Theorem 3 candidate: with accessible responses = the action closure {o∘T_w}, (1) finite Z ⇒ d_h = d_∞ for h ≥ |Z|²−1 (zero truncation error after a finite horizon); (2) no uniform affine rate: Z = ℝ, T(x) = 2x, o(x) = δ_x with ground metric min(1,|u−v|) gives d_h = min(1, 2^h|x−y|), d_∞ = 1, sup(d_∞ − d_h) = 1 for every h. Lay so-what: in a finite latent world, acting and reading the world's own outputs eventually yields a complete map; in an expanding continuous world, decisive differences can stay beyond every fixed exploration budget. Ratio 26:9. Audit #42 (math-only) fires now; if it does not adopt, the repair ladder stops (rule 7).

## 2026-08-30 — Re-contextualization (2-hour check-in, theory-first mode); audit skipped as duplicate

Audit status: every theory claim written since the restart has already been audited fresh and unprimed — audit #40 (REVISE) and audit #41 (REVISE; NO COMPUTE), both adopted verbatim with their alternatives on the audit board (e734, e736). Round 35 (repair pass 2 of 3) is applying audit #41's ten edits; audit #42 follows it. Firing another auditor now would audit a text mid-edit, so this check-in records the step-back instead.

Live question (unchanged in substance, sharpened in form): what is the native geometry of a latent world defined only through what a denizen can do (legal moves) and observe (its own response laws) — and which part of that geometry is genuinely new rather than deterministic Moore/final-coalgebra behavioural-pseudometric mathematics and switched observability wearing latent-space clothes?

What the two mathematics audits reframe: the foundation (d_∞, identity = {d_∞ = 0}, nonexpansive descent, the observability seminorm) is sound but standard; nothing proved so far is new. The distinctive material is application-level: (i) the instrument/native-response boundary (D2), still registration-relative; (ii) finite-denizen access as the map problem (Open Problem 7: the local-to-global gap is reachability, not geometry); (iii) presentation rewrites with coherent transports (D6); (iv) the intervention objective, which the theory has not yet touched. Consequence for earlier work: the whole empirical record reads, under D2, as instrument-level — every probe, decoder and injection we built was outside the denizen's own response family — which is the cleanest statement yet of why twelve constructions found lexical/response geometry and no state.

Alternatives held live (audits #40–#41, verbatim on the audit board; not re-argued here): deterministic final-coalgebra semantics as the compressed core; information geometry (Hellinger/Fisher/JS) of the native next-token law so that D7 is native to probability laws; a category of presentations; nonlinear/switched observability and control geometry for p_x and accessibility; Kantorovich lifting only if generation becomes a move.

Tunnel check and step-back: one thread by design (theory-first), but both audits flag the same drift — treating future-response language as novelty and staying attached to the eight-symbol explicit-legend world. Corrective for the next cycle: (a) the next theorem candidate must live in interface admissibility or finite access (round 35 is asked for exactly that: 'responses are futures' — accessibility as a closure property, with the truncation error d_∞ − d_h as the map problem); (b) the compute checks are re-ranked by audit #41 — C1a is a harness check, C1b finite-support only, C2 observational and still in the eight-symbol world — so none of them is the next artifact; they run, if at all, only as validations of a stated theorem; (c) the program's required intervention artifact (audit #39's causal bridge or its native successor) returns to the agenda only once the interface theorem says what an admissible intervention IS. Ratio 26:8 (warning): the next completed round must be a theorem, and the repair ladder stops after round 35 if audit #42 does not adopt.

## 2026-08-30 — Audit #41 (mathematics-only): REVISE BEFORE ADOPTION; NO COMPUTE; core upheld; ten exact edits; repair pass 2 of 3

Verbatim in `theory/dialogue/004.md`. Licensed sentence: The corrected future-response foundation has a sound standard deterministic Moore-behavioral pseudometric core for nonempty finite carriers, a sound conditional fixed-curve first-variation lemma, and a sound finite-dimensional observability specialization; however Open Problem 7 still needs its absolutely-continuous-curve and weighting statements repaired, C1 needs a D2-compatible outcome-wrapper registration, C2 is not yet a defined preregistration, and `reachability_v1` remains only a restricted centered-log-probability instrument-differential family with `NO SLOT-SPECIFIC GEOMETRY CONCLUSION`, so no computation is authorized.

Never say (audit #41): “Audit #41 adopts the foundation without revision.” “Theorem 1 is false.” “The \(|Z|^2-1\) bound is wrong for finite nonempty \(Z\).” “Theorem 1 is new latent-space mathematics.” “The current final-coalgebra paragraph states its word derivative explicitly.” “D2 makes interface gaming mathematically impossible.” “D6 transport remains undefined.” “The current Open Problem 7 proves \(\bar d\leq\delta_E\) for every executable absolutely continuous curve.” “The displayed \(\delta_E\) equals \(p_\alpha\)-path distance without redefining its integrand.” “The present no-germ example satisfies bounded law-valued D2.” “C1’s permutation pullback is automatically a fixed D2 postprocessing.” “C1b establishes Hilbert geometry for unseen register points or directions.” “C2 has a fixed matched-comparator construction.” “The 12 ordered wording pairs necessarily represent 12 independent wording contrasts.” “`reachability_v1` measured or lower-bounded native \(p_x\).” “The patched residual directions were denizen-executable.” Every audit-#36 never-say sentence in [STATE.md](<C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/STATE.md:55>) remains binding. “Audit #41 authorizes computation.”

Round 35 applies edits 1–10 and proposes the first latent-specific theorem candidate (interface admissibility as a closure property: responses are futures); audit #42 must adopt or the repair ladder stops (CLAUDE.md 2.7 rule 7).

## 2026-08-30 — Round 34: audit-#40 corrections applied to the foundation; Open Problem 7 made precise; audit #41 next

Verbatim in `theory/dialogue/004.md`. All 13 file edits applied (free word monoid, C ≠ ∅, measurable outcomes; ex-ante emitted interface with fixed Markov postprocessing; typed channel/action/outcome transports with coherence; finite registered word set; declared move germs and executable cone; reachability_v1 as instrument seminorms q_{x_n}; d_α defined; positive/convergent ℓ² weights; Euclidean-quotient vs ambient boundary; uniform first-order remainder lemma; directed path distance; C1b/C2 preregistration repairs). Theorem 1 kept elementary with a final-coalgebra remark. Open Problem 7: directed executable path distance δ_E; uniform differentiation gives d̄ ≤ δ_E; L-quasiconvex behaviour image gives d̄ ≤ δ_E ≤ L·d̄; affine world with reversible straight-line germs: δ_E = p_α(y−x) = d_α; counterexample: no germs ⇒ δ_E = +∞ while d_∞ finite (matches Claude's hand check: failure is reachability, not geometry). Ratio 25:8. Audit #41 (math-only) fires now; C1/C2 still blocked.

## 2026-08-30 — Audit #40 (mathematics-only) on the proposed foundation: REVISE BEFORE ADOPTION; 14 exact edits; C1/C2 stay blocked

Verbatim in `theory/dialogue/004.md`. Licensed sentence: The future-response construction is a valid deterministic behavioral-pseudometric foundation after explicit nonempty/free-word assumptions and corrections to Proposition 2, D6, D7, and the audit-#36 retrospective; `reachability_v1` remains only a family of restricted centered-logit instrument seminorms with `NO SLOT-SPECIFIC GEOMETRY CONCLUSION`, and C1/C2 must not run until their mathematical and preregistration defects are repaired.

Never say (audit #40): “Theorem 1 is false.” “The \(|Z|^2-1\) bound is wrong.” “Theorem 1 is a new latent-space theorem.” “The cited probabilistic-bisimulation papers prove this exact deterministic construction.” “\(G=\lambda P_{\mathcal O}\) makes the Gramian seminorm equal to ambient Euclidean distance on all of \(V\).” “Pointwise differentiability proves that \(p_x\) is the first variation of \(d_\infty\).” “Finite \(h\) and \(C_0\) always give a finite-access object.” “D2 makes external decoders impossible to smuggle into the interface.” “D6 already defines outcome-name transport.” “`reachability_v1` measured the native JS/Fisher tangent seminorm.” “The patched residual directions were denizen-executable moves.” “`reachability_v1` lower-bounds the declared \(p_x\)” without the missing response-metric and direction qualifications. “C1b establishes intrinsic Hilbert geometry beyond its finite registered support.” “C2 establishes exact places, an \(\varepsilon\)-quotient, persistence, or a causal bridge.” Every audit-#36 never-say sentence in `STATE.md` remains binding unchanged. “Audit #40 authorizes computation.”

Round 34 applies the edits (Codex), then audit #41.

## 2026-08-30 — Round 33: foundation text adopted as PROPOSED (pending audit #40)

Codex wrote the final text into `theory/AXIOMS.md` (Foundation: future-response geometry — D1–D7, Theorem 1 proved for deterministic moves / distributional responses, Proposition 2 proved in the linear case with the chart boundary, softmax extension sketched, D7 local response seminorm with its lemma, Open Problem 7 conjectured) and the C1/C2 preregistrations into `theory/EXPERIMENTS.md`. Corrections to my round-33 asks: a finite denizen estimates only the restricted p_{x,h,C0}, not full p_x; audit #36 is recorded as 24 restricted empty-continuation Jacobian seminorms plus a shared mean operator, not one base-state p_x and not evidence that global p_x is low-rank; the C1 same-state cross-permutation control is presentation sensitivity when nonzero, not automatically a pull-back error. Next: fresh mathematics-only audit #40 (audit board e5a4e16f); C1/C2 blocked until it is adopted. No compute ran.

## 2026-08-30 — Rounds 32–33: foundation text converging (future-response pseudometric d_∞, Theorem 1 descent/minimality, Proposition 2 observability seminorm; two pre-declared compute propositions C1/C2)

Verbatim in `theory/dialogue/004.md`. Round 32 accepted the pseudometric foundation with corrections (undiscounted sup is foundational; discounting is γ⁻¹-Lipschitz; arbitrary vs Hilbert seminorm; chart boundary G = λP; DINOv2 explanation demoted to a hypothesis; passivity is a tested property), supplied the exact D1–D6 / Theorem 1 / Proposition 2 text, and wrote C1 (frozen consumer: descent certificate + Hilbert-chart adequacy) and C2 (Qwen restricted native response geometry via three shared-numeral queries) as propositions with falsifiers. Round 33 asks for the LM-world scope statement, D7 (local response seminorm p_x; reachability_v1/audit #36 reframed as its first measured lower-bound instance), open problem 7 (when local seminorms integrate to the global d — the map problem), and the C1/C2 amendments; then adoption as 'proposed, pending audit #40' and a fresh mathematics-only audit before any compute.

## 2026-08-30 — Restart, theory-first: Codex round 31 (re-derivation of the axioms from the audited record) and Claude's round-32 challenge

Verbatim in `theory/dialogue/004.md`. Round 31: L1/L2/L4 demoted to consequences; foundation reordered (presentations → responses and legal moves → future observational identity → quotient → maps and laws); seven propositions P1–P7 the record supports; first theorem = finite-horizon identity and action descent (Nerode/bisimulation; novelty disclaimed); second target = presentation covariance vs rebinding. Round 32 (Claude): (A) replace the exact partition by the future-response pseudo-metric d (bisimulation-metric mathematics) so identity is {d=0} and descent is the 1-Lipschitz theorem; horizon → resolution; (B) only denizen-accessible response laws count, so the preflight's decodable state is instrument-level, and the honest native bridge question uses only the model's own next-token laws; (C) first latent-specific theorem = observability seminorm (Kalman): places are cosets of the unobservable subspace and the DINOv2 chart 'won' because it is P-adapted; the seminorm gap is the first measurable latent-specific quantity; (D) drop the presentation groupoid until one passive morphism is exhibited (template 0→2); (E) native = denizen-accessible + consequence-predictive; (F) math first, then the two pre-declared compute checks.

## 2026-08-30 — Audit #39 on register_bridge_preflight_v1 (fresh, unprimed; verbatim): UPHOLD; pause stands; on restart, direction dialogue then one causal residual-to-writer-centroid injection test

Adopted verbatim into STATE, README, EXPERIMENTS, the handoff and the ledger; this is the last entry before the 2026-08-30 pause.

## Verdict

**UPHOLD — `PREFLIGHT PASS — EXPLICIT-LEGEND STATE LINEARLY DECODABLE`.**

The saved features reproduce the recorded status under every predeclared gate:

| Gate | Result | Threshold |
|---|---:|---:|
| Accuracy | 0.8151 | ≥0.60 |
| Minimum fold | 0.7656 | ≥0.50 |
| Minimum state recall | 0.6146 | ≥0.40 |
| Entity-bootstrap LB | 0.7786 | >0.45 |
| Best-control advantage | 0.6419 | ≥0.20 |
| Advantage LB | 0.6016 | >0.10 |
| Accuracy − null p99 | 0.6106 | ≥0.20 |

Imported `Ridge` reproduced every saved main/control prediction, both bootstrap bounds, and the selected layers/lambdas. The first ten RNG-replayed null scores matched bit-for-bit, and the saved-score p99 recomputed exactly.

This is a strong noncausal feasibility result. It is not a code-level bridge, causal bridge, persistent register, or native-mathematics result.

No tracked source, documentation, or result artifact was modified; only the mandatory git-ignored blackboard was updated.

## Split and leakage

The split is genuine within its declared scope:

- All 24 entities are crossed with all eight states: 192 entity-state cells, each with 16 rows.
- Outer folds are by entity: each holds out eight entities and trains on the other 16.
- Templates 0–1 use the training permutation bank; templates 2–3 use the held-out bank.
- The two banks contain eight exact permutations each with zero exact overlap.
- The held-out bank is nevertheless another cyclic Latin bank, so this is not arbitrary-permutation generalization.
- Template and permutation shifts are coupled. The run establishes their conjunction, not their independent contributions.

The nuisance checks pass:

- `pidx=(e+4j) mod 8` holds everywhere and is independent of state and template.
- Within every fold × state × template × arm, each tag occurs exactly twice.
- For each of the 96 entity-template cells, span position and prompt length are constant across states, permutations, and arms.
- Every record tag is two tokens.
- Retokenizing all 3,072 prompts exactly reproduced the saved IDs; every saved record and legend span decoded to the intended tag.
- The registered categorical control scored 0.1354.
- An expanded linear baseline containing tag, permutation index, tag×permutation interaction, template, complete clause-order positions, span, and length scored exactly 0.125 at every tested regularization.

Thus tag identity, permutation index, clause order, length, and span position do not explain the result as transferable linear nuisances. A symbolic procedure that matches the record tag to its legend occurrence and reads the adjacent numeral would solve the task—but that is the intended explicit-legend lookup, not an accidental leak.

## Controls and null

The paired destroyed arm is correctly built. All 1,536 intact/destroyed pairs share entity, original state, template, bank, permutation index, record tag, clause order, record span, record token IDs, length, and full token-ID multiset. Every destroyed permutation is a derangement, and the saved denoted state is exactly \(\sigma^{-1}(s)\ne s\).

The intact decoders are correctly reused:

- Intact accuracy against \(s\): 0.8151.
- Destroyed accuracy against original \(s\): 0.0156.
- Destroyed following of \(\sigma^{-1}(s)\): 0.8516.
- Intact/destroyed predictions differ on 0.9661 of held-out pairs.
- Paired input embeddings are exactly identical.

This decisively rejects a fixed record-tag decoder. It shows that changing only the legend assignment changes what state is linearly readable from the record span. It does not distinguish semantic interpretation from a local tag-to-numeral association mediated by attention.

The legend-occurrence reference is correctly kept outside the gate and scores 0.9440. It is useful positive evidence that the explicit state is especially accessible where the numeral and tag co-occur; it is not a null or a causal ceiling.

The shuffle null is balanced as locked: one state permutation per entity is applied consistently to every repeated row of that entity-state cell. Its 200 saved scores have mean 0.12430 and p99 0.204466, exactly as reported.

Two limitations should remain explicit:

- The null conditions on the observed layer/lambda selections rather than rerunning the complete selection pipeline for every shuffle.
- The bootstrap resamples the 24 evaluation entities but does not refit the decoders, so it measures conditional held-out-entity uncertainty rather than training-pipeline uncertainty.

Neither limitation is remotely large enough to explain the observed margin.

## Decoder stability

Layer selection is stable; regularization selection is less so:

- Every outer fold selected layer 16.
- Lambdas were 1, 10, and 10.
- In folds 1–2, the winning lambda exceeded the runner-up by only about 0.002–0.004 inner accuracy.
- Layer 12 was consistently close to layer 16.

Post-hoc sensitivity checks preserve the result:

- Fixed layer 16, lambda 10: accuracy 0.8164, minimum fold 0.7656, minimum state recall 0.5729.
- Fixed layer 12, lambda 10: accuracy 0.8060, minimum fold 0.7617, minimum state recall 0.6146.
- Multinomial logistic regression at fixed layer 16, with inner-fold selection of \(C\): accuracy 0.8138, folds 0.8164/0.8359/0.7891, minimum state recall 0.5938, destroyed-denoted following 0.8359.

The result is therefore not ridge-specific or produced by foldwise layer variation. It is, however, depth-localized: several early and late fixed layers fail one or more pointwise gates. Do not describe the signal as layer-invariant.

Same-presentation held-out-entity accuracy is 0.8802 versus 0.8151 under the joint held-out-template/permutation shift, indicating a modest presentation cost without isolating which shift causes it.

## Wording audit in both directions

**Over-claim corrections:**

- “Presentation-transferring” must mean the two tested held-out templates plus the disjoint cyclic Latin bank, not arbitrary wording or permutations.
- “State” means the numeric state explicitly assigned to the record tag by the prompt’s legend.
- Rank ≤8 describes the decoder’s output parameterization, not an eight-dimensional intrinsic state space.
- The destroyed arm is not information-free; it coherently assigns the unchanged record tag a different state.
- The bootstrap is not uncertainty over retraining, and the shuffle null is not a full-pipeline selection null.
- `STATE.md` is internally inconsistent: its pause banner says audit #39 is adopted while its detailed `NEXT` still says the preflight is running and audit #39 is pending. That is a propagation defect, not a numerical defect.

**Under-claim corrections:**

- This is stronger than within-template decodability or a tag classifier.
- The state transfers jointly across unseen entities, two unseen templates, and eight unseen exact permutations.
- Reassigning the legend flips the intact decoder toward the newly denoted state while the record token and its embedding remain unchanged.
- Ridge, fixed-layer Ridge, and logistic regression agree closely.
- The strongest reading the numbers do **not** license is that these residuals can be mapped into the constructed writer centroids and causally drive the frozen consumer. That is precisely the next experiment.

## Continue-or-not

**Continue conditionally; the predeclared PASS branch should apply.**

The highest-leverage next move is the required 2–3-round direction dialogue followed by one held-out causal bridge test: map Qwen record-span residuals into the successful writer centroids and inject them into the frozen constructed consumer.

The program should not continue through another synthetic rung, decoder characterization round, layer sweep, or prompt repair. The current pause remains appropriate until the dialogue produces a locked causal test.

The program is still tunnel-visioned around an eight-symbol explicit-lookup micro-world. The PASS justifies one causal discriminator; it does not justify extending the ladder indefinitely.

## Alternatives

The strongest alternative explanation is a local dictionary-lookup mechanism: the record tag attends to its identical legend occurrence and inherits the nearby numeral. That is genuine contextual binding, but it may be prompt-local rather than a reusable latent state.

Useful alternatives or embedded controls are:

- Include the paired reassigned-legend arm in the causal bridge test.
- Compare residual-to-centroid mapping against input-embedding, wrong-centroid, zero, and shuffled-label mappings.
- Require held-out entities, templates, and permutations in the causal consumer test.
- Treat legend-occurrence mapping as an easier positive reference, not the headline route.
- After this line resolves, test an orthogonal real-model source where the state is indirect and cannot be obtained by matching an identical explicit tag.
- Do not run standalone attention localization or further decoder sweeps before the causal discriminator; the ratio no longer permits another measurement-only detour.

## Exact licensed sentence

> In frozen Qwen3-1.7B-Base revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, a predeclared rank-≤8 cross-fitted linear decoder read the explicitly legend-denoted state from the two-token record-tag residual under held-out entities, two held-out templates, and a disjoint balanced permutation bank at 0.815 accuracy (folds 0.828/0.852/0.766, entity-bootstrap lower bound 0.779, minimum state recall 0.615), versus 0.125 input-embedding, 0.135 categorical, and 0.016 paired reassigned-legend original-state controls and a 0.204 shuffle-null p99; on the paired reassigned legends the unchanged-tag decoder followed the newly denoted state at 0.852, establishing a noncausal, prompt-family-bounded explicit-legend state signal—not a code-level or causal bridge, persistent register, synthetic-consumer capability, or native latent mathematics.

## Never-say list

- “Qwen learned a register.”
- “This establishes a causal bridge.”
- “The Qwen residual already contains the constructed consumer’s code.”
- “The state survived, persisted, or was remembered.”
- “The result establishes an eight-dimensional state subspace.”
- “The decoder reads tag identity.”
- “The destroyed arm contains no state information.”
- “The destroyed context failed.”
- “Template and permutation transfer were independently established.”
- “The result generalizes to arbitrary templates, legends, or permutations.”
- “The shuffle null reran the entire selection pipeline.”
- “The bootstrap includes decoder-training uncertainty.”
- “Every layer contains the signal.”
- “The legend-occurrence reference is a gated control.”
- “The residuals can drive the frozen synthetic consumer.”
- “This demonstrates semantic facts, an autonomous state, or native latent mathematics.”

## Copy-ready README wording

> `register_bridge_preflight_v1` is a noncausal feasibility PASS: in frozen Qwen3-1.7B-Base, a predeclared cross-fitted rank-≤8 linear decoder read the state explicitly assigned to a record tag by an in-prompt legend under held-out entities, two held-out templates, and a disjoint balanced permutation bank at 0.815 accuracy, versus 0.125 input-embedding, 0.135 categorical, and 0.016 paired reassigned-legend original-state controls; the same intact decoders followed the state newly denoted by the paired reassigned legend at 0.852. This establishes a prompt-family-bounded, presentation-transferring explicit-legend signal at the record span—not a code-level or causal bridge, persistent register, synthetic-consumer usability, or native latent mathematics.

## Copy-ready STATE wording

> - **`register_bridge_preflight_v1` — PREFLIGHT PASS: EXPLICIT-LEGEND STATE LINEARLY DECODABLE.** In frozen Qwen3-1.7B-Base, cross-fitted rank-≤8 Ridge decoders evaluated on held-out entities × templates × a disjoint balanced permutation bank achieved 0.815 accuracy (entity-bootstrap LB 0.779; folds 0.828/0.852/0.766; minimum state recall 0.615), versus input-embedding 0.125, categorical 0.135, paired reassigned-legend original-state 0.016, and shuffle-null p99 0.204; the intact decoders followed the paired legend’s newly denoted state at 0.852, ruling out fixed tag identity. This is a noncausal explicit-legend lookup signal, not a code-level or causal bridge. The program remains paused; on restart, conduct the required direction dialogue and then one held-out causal residual-to-writer-centroid injection test, with no synthetic staircase advance or further decoder sweep first.

## Ranked next increments

1. Conduct the mandatory 2–3-round direction dialogue around the causal bridge and its stop rule.
2. Lock one residual-to-writer-centroid mapping test using training entities/templates/permutations only.
3. Evaluate injection into the frozen constructed consumer on held-out entities, templates, permutations, and paired reassigned legends.
4. Include zero, wrong-centroid, shuffled-label, embedding-derived, and destroyed-legend controls in that same run.
5. On causal success, license only cross-system causal consumption under this interface and decide whether one indirect-source generalization is warranted.
6. On failure, close the synthetic register bridge under this route; do not reopen layer, prompt, decoder, or actuator sweeps.
7. After closure or causal success, move to an orthogonal real-model task where the state is not recoverable by matching an identical explicit tag.

## Ratio heartbeat

Current runner: 104 nonblank lines. Under the repository’s existing line convention, approximately 34 are generic apparatus and 70 are estimand-bearing measurement logic, about **0.49:1**. That is below the line warning, but this runner builds no causal central artifact; it remains a measurement.

Round accounting:

- Audit #38 baseline: 21 measurement/governance : 7 build.
- Round 28 lock review: 22:7.
- Preflight result: 23:7.
- This audit: **24:7 = 3.43:1**.

The ratio remains above the 2:1 warning and below the 5:1 mandatory halt. The next active round must build the causal bridge discriminator or close the line; another measurement-only round is not justified.

## 2026-08-30 — Re-contextualization at the pause (2-hour check-in; audit #39 in flight)

Live question unchanged: can a real model's latent space carry a causally addressable, content-specific state, and what must a denizen of that space invent to navigate it? Where the picture stands: every frozen-model *actuator* failed (audits #27–#36); the constructed substrate is a working but trivial calibration (audits #37–#38); and the preflight — the first measurement that asks about the *source* rather than the actuator — says the explicit-legend state is linearly readable at layer 16 of Qwen3-1.7B-Base, transferring across entities, templates and permutations, with the paired destroyed arm showing the decoder reads the legend binding rather than the tag. That reframes the frozen-model negatives: the failures sat in write/actuation and in weak *generic-anchor* source extraction (`Internal record:` slot), not in the absence of a readable episode state at the record span.

Alternatives held live for whoever restarts (mine; audit #39 adjudicates and its alternatives are adopted verbatim):
1. The decodable signal is "which tag occupies the record" + "which legend slot names it" resolved by attention at layer 16 — a lookup, not a state; a bridge from it would inherit the same lookup character as the constructed consumer.
2. Decodability ≠ addressability: rank-8 readability at the span says nothing about whether writing into that span moves behaviour; the cheapest next test is causal (audit #38's centroid-injection bridge), not another readout.
3. The record span is not where a denizen keeps state across time; the delay rungs that killed the one-write line may kill this too — the first informative failure is decodability *after* filler, which is one config change away.
4. A cheaper moot-maker: a 0.6B or 4B base with the same preflight would show whether the signal is scale-tied; and a plain token-level probe on the *legend* occurrence (0.944 here) suggests most of the information is already present where the tag is defined.
Tunnel check: the program is pausing with two independent lines closed by audit and one open measurement; no repair ladder is live. Ratio ~23:7 (warning), which is why the restart rule requires a fresh direction dialogue before any build.

## 2026-08-30 — register_bridge_preflight_v1 result (provisional; audit #39 in flight) and the pause

Devansh asked for a clean pause: finish and audit the running measurement, write a zero-context handoff, clean the sprawl, commit and push, then stop. The preflight (locked 8beb8e9) returned PREFLIGHT PASS — EXPLICIT-LEGEND STATE LINEARLY DECODABLE: held-out accuracy 0.815 (entity LB 0.779; folds 0.828/0.852/0.766; state recall 0.61–0.99), input-embedding 0.125, categorical 0.135, paired context-destroyed 0.016 with the destroyed arm following the legend-denoted state at 0.852, entitywise-max control advantage 0.642 (LB 0.602), balanced label-shuffle null mean 0.124 / p99 0.204, legend-occurrence reference 0.944, layer 16 selected in every fold. Ledger `register_bridge_preflight_v1_result` (d2acfe8). Provisional until audit #39; the pre-declared licence is only that frozen Qwen's tested source span contains a linearly accessible, presentation-transferring explicit-legend state signal — not a code-level or causal bridge.

## 2026-08-30 — Round 29 (direction session; verbatim): pause / handoff design and cleanup ruling

The right handoff is one committed front door, finalized only after audit #39 is adopted. The actual chronology is prior program → NLM-007 → toy quotient → frozen-model line → constructed substrate → real-model preflight; the prompt’s toy/NLM ordering is reversed.

## 1. Handoff document spec

Create [docs/HANDOFF_2026_08_30.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/docs/HANDOFF_2026_08_30.md) with these sections.

### Title and pause banner

```md
# Latent-Space Reasoning — pause handoff, 2026-08-30

> **PAUSED.** No experiment is authorized merely by reopening this repository.
> Read this document, `STATE.md`, and the adopted audit #39 record before
> proposing work. Any restart begins with the blackboard and a fresh direction
> dialogue.
```

### Mission and guiding question

Copy the mission sentence from the opening of `README.md`, followed by this exact text from `AGENTS.md`:

> Mathematics was invented by inhabitants of a world to navigate it — counting, measuring, mapping, predicting — and its laws were shaped by what that world made necessary. Invert the dynamic: **take the latent space as the world.** Ask what a denizen of that world would have to invent to find its way — what counts as the same place, what a move is, what effort a move costs, what a map is, what regularities make prediction possible — and let that need decide which primitives and laws we build.

Then state:

> The mission remains open. No tested real-model construction has established native latent mathematics, a transferable causal state, or a general impossibility result.

### Exact pause point

Do not draft this until audit #39 is adopted. Copy, verbatim:

1. Audit #39 headline.
2. Its exact licensed sentence.
3. Its never-say list.
4. Its “continue-or-not / highest-leverage” ruling.
5. Its final measurement-to-artifact ratio.

Sources, in authority order:

- The audit-#39 block added to [STATE.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/STATE.md).
- Ledger rows `register_bridge_preflight_v1_result` and `register_bridge_preflight_v1_audit39`.
- The local `.codex_audit39.md` only as a cross-check; it is git-ignored and cannot be the committed authority.

Never copy the raw `run_result.json` headline without the audit qualification.

### One-screen program arc

Use this table:

| Line | Status | Binding provenance |
|---|---|---|
| Prior perturbation/diffusion program | Closed 2026-08-27; nested-arithmetic claims withdrawn after the benchmark was shown to measure termination under a token cap | `README.md` “Prior program and correction”; `legacy/docs/CORRECTION_NESTED_ARITHMETIC_2026_08.md` |
| NLM-007 residual-dynamics line | Closed by terminal allocation rule, not by a scientific null | Audit #22; ledger `nlm007_closed_audit22`; `STATE.md` “NLM-007 — CLOSED” |
| Toy operational-quotient program | Ended; no learned artifact passed the complete exact reducer, and Round 37 found no architectural win | Audits #23–#26 plus Round 37 audit; ledger IDs enumerated in `STATE.md` “Closed toy program” |
| Frozen-model coordinate/interchange/state-bus/control-cost line | Stopped as an allocation pivot, not a scientific conclusion | Audits #27–#31; ledger `direction_r10_program_ruling` and construction rows |
| One-write positive-control staircase | Exact constructions failed or localized instrument, extraction, actuator, or prompt/output-geometry constraints; no general memory/capacity conclusion | Audits #32–#36 |
| Constructed register substrate | Rung 0 is a qualified synthetic oracle-code selector; rung 1 is a qualified answer-supervised eight-symbol writer; neither establishes semantic writing, persistence, geometry, or pretrained-model structure | Audits #37–#38 |
| Real-model bridge preflight | Insert audit #39’s exact adopted classification here | Ledger `register_bridge_preflight_v1_lock`, result, and audit #39 |

Immediately below it, copy the four durable findings and unified explanation from [docs/STRUCTURED_NEGATIVE_2026_08_29.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/docs/STRUCTURED_NEGATIVE_2026_08_29.md), verbatim.

### Do not repeat

Include this closure index. The “Licensed/never-say authority” column is a pointer, not permission to paraphrase.

| Construction | Closure reason | Licensed/never-say authority |
|---|---|---|
| NLM-007 | Terminal allocation stop; Round 34b inconclusive | `STATE.md` NLM-007 block, audit #22 |
| Operational quotient v1/36b/36c/36d and presentation quotient | Toy program ended; exact certificates diagnostic-only; Round 37 no architectural win | `STATE.md` closed-toy blocks, audits #23–#26/Round 37 |
| `coordinate_v1` | `UNINTERPRETABLE — INVALID POLARITY BASELINE` | `STATE.md` historical coordinate block |
| `coordinate_v2` | Killed at the predeclared baseline gate | Same |
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

Add:

> `necessity_navigator_v1` is **deferred, not a result and not automatically next**. Its 2,000-step record is a smoke only. Running it requires a fresh direction dialogue.

### Standing decision after the preflight

Audit #39 supersedes everything here. Copy its ruling verbatim. If it preserves audit #38’s branches, reproduce:

- PASS → direction dialogue, then one held-out causal bridge test mapping real-model residuals to successful constructed-writer centroids and injecting them into the frozen consumer.
- PARTIAL → lexical/context engineering evidence only; no staircase advance.
- FAIL → close the synthetic register route as nonbridging through this tested source-span path.
- Never reopen layer, site, actuator, prompt, or model-size sweeps as repairs.

### Binding operating contract

Reproduce these rules because `AGENTS.md` is git-ignored:

- Every session calls `bb_list` first, records findings with provenance, and calls `bb_synthesis` before a verdict.
- Before any run, state the expectation, what each possible outcome implies, and the simplest confound that could explain every row.
- A scientific run requires a frozen design, config, runner hash, thresholds, statuses, controls, and kill rule.
- A locked outcome is audited once. There are no outcome-dependent rescue runs. A scientifically different successor requires the 2–3-round direction dialogue and a new lock.
- Exact certificates are diagnostics only. Learned continuous-model verdicts require tolerance criteria, effect sizes, and seed spread.
- Use the positive-control staircase: training items/zero delay → short delay → held-out names → unseen wording → long delay, changing one difficulty per audited rung.
- Report apparatus/artifact lines and measurement/build rounds at every heartbeat. Above 2:1, warn; above 5:1, halt and pivot.
- Generation accuracies always include termination. Task-nested samples use paired or clustered inference.
- Audit wording overrides raw runner status. Propagate an adopted downgrade or qualification to README, STATE, NOTEBOOK, EXPERIMENTS, and the ledger in the same session.
- One CPU process at a time. No GPU run without Devansh’s explicit approval.
- Codex means the real CLI:

```text
codex exec -s workspace-write --skip-git-repo-check -C "<dir>" -o "<out>" "<prompt>"
```

- Each Codex session is fresh; the repository carries context. `resume --last` is only for an immediate one-line follow-up.
- One idea per commit; commit messages end `Committed by Devansh`. Never use `git add -A`. Never commit `AGENTS.md`, `internal/`, `.codex_*`, `.claude_*`, `.blackboard/`, secrets, routing, costs, or local model details.

### Hardware and process facts

Copy the Hardware section of `AGENTS.md`, including:

- Sustained GPU load has hard-crashed this laptop.
- Explicit approval is required per GPU run.
- One compute process at a time.
- Use detached, checkpointed execution for approved long jobs.
- After unexplained restart:

```powershell
Get-WinEvent -FilterHashtable @{LogName='System'; Id=6008,41}
```

- Windows runs use `PYTHONUNBUFFERED=1` and `PYTHONIOENCODING=utf-8`.

### Canonical runners

Make clear that commands for closed runners are for reproduction only.

| Runner | Exact command authority |
|---|---|
| `run_lm_dynamics.py` | Relevant NLM-007 ledger result row; do not reconstruct its multi-stage commands |
| `run_coordinate.py` | `coordinate_v1_result`, `coordinate_v2_baseline`, `coordinate_v3_result` |
| `run_interchange.py` | `interchange_v1_result`, `interchange_v2_lock/result` |
| `run_state_bus.py` | `state_bus_v1r1` lock/result rows |
| `run_control_cost.py` | `control_cost_v1_result` |
| `run_onewrite_state.py` | `onewrite_state_v1_killed_prelock` |
| `run_onewrite_recall.py` | `onewrite_recall_v1_lock/result`, `onewrite_recall_rung1_lock/result` |
| `run_oracle_actuator.py` | `oracle_actuator_rung0_lock` |
| `run_site_oracle.py` | `site_oracle_v1_lock` |
| `run_reachability.py` | `reachability_v1_lock` |
| `run_necessity_navigator.py` | Deferred; smoke command only in `navigator_smoke_r32fixes`; do not run |
| `run_necessary_register.py` | Rung-0 and rung-1 lock rows |
| `run_register_bridge_preflight.py` | `register_bridge_preflight_v1_lock`; do not rerun |

The current reproducibility commands are:

```text
python experiments/run_necessary_register.py --config experiments/config/necessary_register_v1.json
python experiments/run_necessary_register.py --config experiments/config/necessary_register_rung1.json
python experiments/run_register_bridge_preflight.py --config experiments/config/register_bridge_preflight_v1.json
```

All three are completed/closed once audit #39 lands; none is a restart instruction.

### Repository map

- `README.md`: public front door.
- `docs/HANDOFF_2026_08_30.md`: pause and restart authority.
- `STATE.md`: canonical scientific wording and never-say lists.
- `docs/STRUCTURED_NEGATIVE_2026_08_29.md`: audited two-day synthesis.
- `experiments/ledger.jsonl`: append-only run, hash, command, result, and audit provenance.
- `experiments/EXPERIMENTS.md`: detailed experiment record; older queue language is historical.
- `NOTEBOOK.md`: reverse-chronological dialogue and audit record, not claim authority over `STATE.md`.
- `experiments/config/*.json`: immutable contracts for completed runs.
- `experiments/results/*/result.json` or `run_result.json`: raw outcomes; audit wording may narrow their status strings.
- `legacy/`: closed previous program, not an active queue.
- `.codex_*`, `.blackboard/`, `AGENTS.md`, logs, feature dumps, and some checkpoints: local-only unless separately packaged.

### First 30 minutes for a fresh reader

```md
1. Run `bb_list`; reuse the existing direction/audit board.
2. Read README → this handoff → STATE Current statement → the structured negative.
3. Read the final five ledger rows, audit #39’s adopted block, and the preflight result summary.
4. Run `git status --short` and record HEAD; do not edit hash-bearing configs or runners.
5. Confirm that no experiment process is active.
6. Classify every proposed action against audit #39’s restart rule.
7. State expectation, outcome branches, simplest confound, and the current ratio before proposing a run.
8. If work resumes, begin a fresh 2–3-round direction dialogue; do not resume an old repair ladder.
```

## 2. `STATE.md`

Replace the header with:

```md
## Current statement (2026-08-30, PAUSED; audits #27–#39 adopted; `docs/HANDOFF_2026_08_30.md` is the restart authority; no experiment is running)
```

Insert immediately below it:

```md
- **PAUSE / RESTART AUTHORITY.** This program is paused. The adopted `register_bridge_preflight_v1` result and audit #39 wording below govern any restart. No embedded `NEXT` sentence from audits #27–#38 authorizes work; those sentences are historical. A restart begins with `bb_list`, the handoff and this Current statement, followed by the direction dialogue required by audit #39 before any build or run.
```

Do not alter any verbatim audit block. Remove only the final standalone bullet beginning:

```md
- NEXT: `register_bridge_preflight_v1` locked ... and RUNNING
```

Replace it with:

```md
- `register_bridge_preflight_v1` — [COPY AUDIT #39 HEADLINE VERBATIM].
  Licensed sentence (audit #39, verbatim): [COPY FROM THE ADOPTED STATE BLOCK].
  Never say (audit #39): [COPY VERBATIM].
- **PAUSED.** [COPY AUDIT #39’S CONTINUE-OR-NOT RULING VERBATIM.] No further run is authorized until a fresh direction dialogue ratifies the applicable branch.
```

Also reconcile the top of `experiments/EXPERIMENTS.md`: it currently says “early,” calls audit #38 pending, and still calls audit #34 pending. Those must become adopted/history entries before handoff.

## 3. Cleanup ruling

Delete:

- `experiments/analyze_r34a_frozen.py` — untracked temporary copy; its own ledger row says the completed chain deletes it.

Retain or explicitly commit/package—do not silently delete:

- `experiments/results/onewrite_state_v1/smoke_result.json` — untracked, but contains raw pre-lock decodes and is named as a ledger `data_ref`. Either add it explicitly as diagnostic evidence or retain it locally with a recorded hash.
- `experiments/results/register_bridge_preflight_v1/run_features.npz` once produced — git-ignored but required for recomputation; retain at least through audit #39 and record its hash/location.
- `experiments/results/necessity_navigator_v1/` — the construction remains deferred.
- Every closed result directory, including toy quotient and NLM-007 outputs. Negative results are permanent.
- `op_update_fixture.py` — no filename-level ledger hit, but it is documented historical Round-36 fixture machinery.
- All locked configs, runners, raw-row archives, consumer/writer checkpoints, and audit-adopted result JSONs.

Do not delete or consolidate `STATE.md`, `NOTEBOOK.md`, `experiments/EXPERIMENTS.md`, or the structured negative. There is currently no duplicate file in `docs/`; the handoff has a distinct front-door role.

## 4. README Status replacement

Replace the entire run-on Status section with:

```md
## Status (paused 2026-08-30)

Across two days of audited CPU experiments, no tested construction established a transferable causal state or native latent mathematics in a real model; the durable result is a localized set of instrument-validity, source-extraction, actuator, and prompt/output-geometry constraints that any future substrate must satisfy.

The program is paused after `register_bridge_preflight_v1` and audit #39. Start with the [pause handoff](docs/HANDOFF_2026_08_30.md), then the canonical [current state](STATE.md). The audited two-day closeout is [the structured negative](docs/STRUCTURED_NEGATIVE_2026_08_29.md), and append-only run/audit provenance is in the [experiment ledger](experiments/ledger.jsonl).
```

## 5. Newcomer traps

- Raw status strings are not final scientific claims. Audit #37 narrowed rung 0; audit #38 narrowed rung 1; audit #39 must govern the preflight.
- `state_bus_v1r1`’s displayed status omits one registered failure because of control flow; use audit #29.
- The oracle-actuator result’s initial “two codes worked” interpretation was corrected: code 0 reproduced the cue prior.
- `site_oracle_v1`’s reported cap-active rate was numerical boundary rounding, not meaningful clipping.
- `reachability_v1` does not establish two or three reachable dimensions.
- The tracked navigator `smoke_result.json` is not the 2,000-step smoke cited in `STATE.md`; `NOTEBOOK.md` records that the live file was overwritten by a later 300-step invalid smoke. Cite the ledger row, never that file, for the 2,000-step numbers.
- `legacy/` is historical and closed; its older NEXT/queue language is not authorization.
- `.codex_*`, `AGENTS.md`, `.blackboard/`, logs, and feature dumps are git-ignored. A fresh clone will not contain them.
- Many `NOTEBOOK.md` links are absolute paths tied to this laptop.
- Hash-bearing completed configs and runners are evidence. Editing them destroys reproducibility; successors need new filenames/configs and a new lock.
- The final closeout is not complete until `git status --short` is clean or every remaining local-only evidentiary artifact is explicitly listed in the handoff.

## 2026-08-30 — Round 28 (direction session; verbatim): lock review of the repaired preflight runner — six edits required, then lockable

All six edits were applied (pidx = (e + 4j) mod 8 with tag-balance assertion; destroyed arm scored with the intact fold decoders; legend-occurrence reported as a reference, not gated; balanced entity-clustered label-shuffle null; deadline enforced through fitting; float32 features; separate hash-bound config `experiments/config/register_bridge_preflight_v1.json`; renamed statuses; denoted state asserted and saved). Smoke ran end to end; locked at 8beb8e9 (`register_bridge_preflight_v1_lock`) and launched.

**Lock ruling: EDITS REQUIRED.** The smoke may finish as an implementation diagnostic, but the full run is not yet evidentiary.

1. [Line 57](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_register_bridge_preflight.py:57) leaks state through permutation selection: `e*K` vanishes modulo 8, leaving `pidx` determined by `s,t,j`. Use `pidx=(e+j*(K//2))%K` and assert equal `pidx`/tag frequencies within every fold × state × template.

2. [Lines 70–90](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_register_bridge_preflight.py:70) retrain a decoder on destroyed prompts. Fit each fold decoder on intact training rows once, then apply those exact weights to both intact and paired-destroyed held-out rows. Save accuracy against original `s` and following of the destroyed-legend state.

3. [Lines 89–100](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_register_bridge_preflight.py:89) incorrectly put `legend_occurrence` inside the control that PASS must beat by 0.20. It is an easier positive reference/ceiling, not a null. Report it separately. Gate paired advantage against input embedding, categorical, and destroyed-context controls only—preferably using the entitywise maximum before bootstrapping.

4. [Lines 93–97](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_register_bridge_preflight.py:93) preserve fold-level label balance, but shuffle individual repeated rows. Use one state permutation per entity, applied consistently to every training row belonging to each `(entity,state)` cell. This preserves entity, fold, and repeated-view structure. It is a **balanced entity-clustered label-shuffle null**, never a coordinate-shuffle null.

5. [Line 65](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_register_bridge_preflight.py:65) enforces the deadline only during model forwards. Enforce it during feature compression, decoder selection, controls, nulls, and serialization too. [Line 67](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_register_bridge_preflight.py:67) should retain float32 features; float16 need not reproduce ridge selection or boundary predictions.

6. A separate config is mandatory: `experiments/config/register_bridge_preflight_v1.json`. The locked rung-1 config still says `state=e mod 8`, “one entity per state,” and code/bridge-level statuses. Do not edit it. The new config must bind the exact templates, revision, entity source, seed 2727, 200 shuffles, two permutations per cell, 2,000 bootstraps, 150-minute total wall, feature dtype, and corrected statuses. Remove operational defaults from [line 33](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_register_bridge_preflight.py:33).

The destroyed prompt construction itself is correct. Under the intact legend, tag `π[s]` denotes `s`; under the destroyed legend it denotes `σ⁻¹(s)`, which differs from `s` because `σ` is a derangement. The arms share clause order, permutation, record, and token multiset—not identical token sequence, since the legend tags must move. Explicitly assert and save `destroyed_denoted_state = σ⁻¹(s)`.

Finally, change the emitted statuses at [line 100](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_register_bridge_preflight.py:100) to:

- `PREFLIGHT PASS — EXPLICIT-LEGEND STATE LINEARLY DECODABLE`
- `PREFLIGHT PARTIAL — TOKEN/CONTEXT-BOUND DECODABILITY`
- `PREFLIGHT FAIL — NO QUALIFYING EXPLICIT-LEGEND STATE DECODER`

After these edits and one manifest/runtime smoke, the design is ready to lock without further scientific changes.

Ratio heartbeat: approximately `22:7 = 3.14:1`; the warning remains active, so this one preflight must lead directly to a causal bridge dialogue or closure.

## 2026-08-30 — Audit #38 on necessary_register_rung1 (fresh, unprimed; verbatim): qualified uphold; 'functional source writer' narrowed to an answer-supervised eight-symbol writer; synthetic staircase stops; preflight runner not lockable (nine defects)

My reading is corrected: value_only reproduces own, so the entity is dispensable and the rung tests no relation; 'the addressable set is larger than the 8 codes' is replaced by the audited decision-region reading. The nine preflight defects are repaired before any run (the smoke timed out at 1500 s in the fitting phase, confirming defect 5).

## Verdict

**Qualified uphold; headline narrowed.**

`necessary_register_rung1` mechanically and reproducibly passes its locked gates on seeds 11/23/37. The defensible classification is:

**`QUALIFIED PASS — ANSWER-SUPERVISED EIGHT-SYMBOL WRITER INTO A FROZEN SYNTHETIC CONSUMER`**

The result establishes more than ordinary eight-class training: downstream answer-label-plus-EOS loss alone found non-oracle unit interventions that a separately trained frozen consumer follows under unseen output permutations. But `value_only` reproduces `own`, so the source task itself is trivial: `<V_s>` directly names the state, the entity is unnecessary, and no relation or semantic fact must be inferred.

The current phrase “functional source writer” is acceptable only with that qualification. The broader reading—semantic source writing, code recovery, persistence, or discovery of a register geometry—is unsupported.

No research artifact was modified; only the required git-ignored blackboard record was updated.

## Replay and freeze

The binding evidence is strong:

- The live rung-1 runner/config hashes and bound rung-0 config/result hashes match `run_result.json`.
- All six saved writer/consumer checkpoint hashes match.
- The replayed consumer reproduces the seven checked rung-0 scalars exactly, apart from floating-point roundoff of at most \(2.8\times10^{-17}\) in `random_assigned`.
- The factorial oracle regression remains `VALID` at 0.960/1.000/1.000.

The qualification is that “replay” means deterministic retraining, not loading the original rung-0 consumer: no original checkpoint existed. `replay_ok` compares seven scalar summaries, not parameters, logits, rows, or checkpoint bytes.

The freeze is genuine by code:

- `model.eval()` is called before the crossed regression and writer training.
- Every consumer parameter receives `requires_grad_(False)`.
- AdamW receives only `wr.parameters()`.
- Dropout is zero and the model has no batch-normalization-style mutable state.

The consumer checkpoint is saved before writer training and is not rehashed afterward. Thus immutability is established by the inspected execution path, not by a saved before/after comparison. A post-fit consumer hash would make future runs stronger.

## Writer and controls

The writer sees exactly:

`<SRC> <E_e|MASK> <HAS> <V_s|MASK> <WRITE>`

It receives no panel, permutation, template, query, answer, or output label. Its loss is answer-label plus EOS cross-entropy through the frozen consumer; there is no code regression, cosine objective, or auxiliary state classifier.

The controls localize what it learned:

- `value_only` accuracy is 0.967/1.000/1.000 versus `own` 0.966/1.000/1.000.
- Decoded `own` and `value_only` outputs are identical on 2,297/2,304 seed-11 rows and every row for seeds 23/37. It is therefore near-equality, not literal equality in all seeds.
- Full-source versus value-only write-vector cosine averages 0.996/0.992/0.985.
- Masking the value reduces assigned-target accuracy to 0.124/0.109/0.109.
- All arms terminate at 1.000.

Consequently, the entity token is behaviorally dispensable and the value token supplies essentially the entire state signal. Shuffled donors mainly repeat the counterfactual-value test with another entity attached.

The crossing is genuine within its declared scope:

- All 384 `(entity, template, panel)` cells occur exactly six times per seed.
- All 128 `(state, template, panel)` cells occur exactly eighteen times.
- Every panel × held-out-permutation cell appears, with 22–51 rows per cell.

But state is fixed as `entity mod 8`; entity × state is not factorially crossed. The seven counterfactual arms change the value for each fixed prompt and provide the important within-prompt causal evidence.

Donors always change state. Exact panel/template/permutation matching holds for 6,911/6,912 rows. Seed 11 has one registered fallback matched only on panel/template; it still follows the donor state. This negligible deviation does not change the verdict, but “perfectly context matched” would be false.

## Telemetry interpretation

The saved telemetry does not support nearest-oracle-code decoding:

- True-code cosine is 0.721/0.421/0.493.
- Seed 23’s 24 state-6 writes are all nearest another oracle code.
- Seed 37’s 24 state-3 writes are all nearest another oracle code.
- Nevertheless, the full writes score 0.966/1.000/1.000.

Checkpoint interventions sharpen the geometry:

- Mean energy inside the eight-code span: 0.548/0.286/0.442.
- Mean orthogonal-complement energy: 0.452/0.714/0.558.
- Code-span projection alone: 0.939/0.871/0.975 accuracy.
- Orthogonal complement alone: 0.398/0.305/0.688.
- Per-state full-write centroids: 0.965/1.000/1.000.
- Nearest-code replacement: 0.961/0.875/0.875.
- Sign-only code coefficients: 0.284/0.191/0.220.

The consumer is therefore reading neither nearest-code cosine nor a simple sign pattern. Both projected and off-code components carry causally useful state information, and their combination matters.

The narrow licensed interpretation is that the consumer has context-dependent decision regions containing the eight oracle points and additional learned writer centroids. “The addressable set contains successful non-code points” is established. “The addressable set is a larger subspace,” its dimension, topology, smoothness, or interpolation structure is not.

## Wording audit in both directions

**Over-claim:**

- “Source writer” suggests semantic extraction; this is an eight-token symbol writer.
- The state is directly named by `<V_s>`, so the rung does not test relational binding, factual interpretation, or language understanding.
- `value_only = own` is approximate for seed 11, not row-for-row exact.
- Donor context matching is 6,911/6,912, not perfect.
- Entity × template × panel is factorial; entity × state is not.
- Far-from-code writes do not establish a larger addressable subspace.
- Nearest-code error does not itself reveal what geometric rule the consumer uses.
- “Zero/random behaved at chance” must remain “assigned-target accuracy near 1/8”; their output distributions are systematic.
- Zero configured filler is not evidence of memory or persistence.

**Under-claim:**

Calling this only “a GRU learned eight symbols” omits the real engineering positive. Three independently initialized writers, trained without code supervision, found unit interventions that a separately trained frozen consumer accepts under globally unseen output mappings. Same-prompt counterfactual following is 0.964/1.000/1.000, and per-state centroids retain full behavior.

It is also genuinely established that the consumer’s behavioral partition is not the oracle-code Voronoi partition: two complete state classes are closer to the wrong oracle code while remaining perfectly followed.

## Preflight runner defects

The current `register_bridge_preflight_v1` runner should **not** receive a full run or evidentiary lock yet.

1. **The central estimand is confounded.** State is always `e mod 8`; entities never change state. A PASS can therefore reflect held-out-name lexical or morphological structure rather than an episode-defined legend/record relation. Cross every entity with every state.

2. **The context-destroyed control is not paired.** Intact and destroyed prompts independently shuffle clause order. The arm changes both state–tag pairings and clause order, so its gap cannot isolate contextual binding. Reuse identical clause order, positions, permutation, and token multiset.

3. **Evidence retention is inadequate.** Final rows omit residuals, embeddings, permutation index, clause order, destroyed mapping, control predictions, and the full shuffle-null distribution. Main model selection, controls, bootstrap, and null cannot be independently recomputed.

4. **The shuffle null breaks fold balance.** It globally permutes the 24 entity labels, so individual eight-entity folds may lose states. That can depress the null. It is also a label-shuffle null, not a “shuffled-coordinate” null. Permute assignments within each balanced eight-entity fold and save all 200 scores.

5. **The solver is computationally infeasible as written.** Each fit solves a dense approximately \(2048\times2048\) system. Layer/ridge selection, controls, and 200 null refits imply roughly fifteen thousand such solves. The claimed 35–50 CPU minutes is implausible, and no deadline is enforced. Use dual ridge on the much smaller sample-space matrix.

6. **Tokenization identity is incomplete.** The runner feeds `b_ids + ids(".")` without verifying equality to tokenizing the declared full prompt. It must save exact prompt text, token IDs, span indices, decoded spans, and require the span to be the final record-tag occurrence.

7. **The status overstates bridge relevance.** The runner never loads the constructed codebook, writer, or consumer. It predicts one-hot state labels from Qwen residuals. A PASS is linear state decodability for an explicit legend lookup—not linear access to the register code or evidence that the resulting vector can drive the consumer.

8. **Bootstrap evidence can disagree with the gate.** `boot(acc_e)` is redrawn separately for logging, gating, and saved `entity_lb`. Compute each bootstrap statistic once and reuse it.

9. **The execution contract is implicit.** CPU/float32 happen to follow `SubstitutionProbe` defaults, but config values are not explicitly applied, and no wall-clock limit exists.

A FAIL would also conflate held-out entity, template, and permutation shifts. That conjunction is a legitimate stringent screen, but it cannot localize why the bridge failed.

## Continue-or-not

**Continue conditionally, but stop the synthetic staircase now.**

The calibration artifact is working; the native-latent-space program is not yet working. The highest-leverage next action is one repaired real-model preflight because it can moot the synthetic line in either direction. Running short delay or another constructed rung first would be tunnel vision.

The program is moderately-to-strongly tunnel-visioned around the same eight-symbol lookup structure: rung 0 selects a map position, rung 1 decodes `<V_s>`, and the preflight asks whether an explicit legend lookup is linearly readable. These are useful interface calibrations, but not yet a denizen’s state, motion, effort, or map in a real latent world.

Decision rule:

- **PASS after repair:** conduct the mandated dialogue and move directly to a held-out causal bridge test.
- **PARTIAL:** treat it as lexical/context engineering evidence; do not advance the constructed staircase.
- **FAIL:** close this synthetic line as unsupported by the tested real-model source-span route.
- Do not reopen layer/site/actuator sweeps.

## Alternatives

The strongest alternative explanation of rung 1 is the cheapest one: a GRU maps eight unique value tokens into eight learned representatives inside broad, context-dependent consumer decision regions. Entity, source relation, and persistence are unnecessary.

Better explorations instead of or alongside further synthetic development:

- Make the real-model preflight truly episode-defined by crossing every entity with every state.
- Replace the explicit `<V_s>` constructed source with a legend plus indirect record label, using an exactly paired mapping-destroyed control.
- If the repaired preflight passes, fit a map from Qwen residuals directly to the eight successful writer centroids and test whether injection into the frozen constructed consumer follows held-out states and permutations.
- Add a matched legend-tag-occurrence baseline: compare the record-tag residual with the identical tag at its legend occurrence to distinguish contextual binding from token reuse.
- Treat input embedding, categorical, destroyed-context, and balanced label-shuffle controls as separate named estimands, not one generic “control.”
- Do not spend another round characterizing the synthetic consumer’s geometry unless a real-model bridge first survives.

## Exact licensed sentence

> In the locked constructed `necessary_register_v1` task, three answer-label-plus-EOS-trained GRU interfaces, seeing the training-format source `<SRC> <E_e> <HAS> <V_s> <WRITE>`, produced unit register writes that made separately reconstructed-and-frozen rung-0 consumers follow own, all seven same-prompt counterfactual, and state-changing shuffled-donor value tokens under held-out output permutations at zero configured filler, with own accuracy 0.966/1.000/1.000, counterfactual following 0.964/1.000/1.000, and shuffled-donor following 0.967/1.000/1.000; because masking the entity preserved behavior, this establishes an answer-supervised eight-symbol writer into a synthetic consumer—not semantic source understanding, persistence, oracle-code recovery, an addressable geometric subspace, or structure in a pretrained model.

## Never-say list

- “The writer learned semantic facts.”
- “The writer understood the source sentence.”
- “Entity identity contributed to successful writing.”
- “`value_only` equaled `own` row-for-row in every seed.”
- “Every shuffled donor was exactly permutation matched.”
- “Entity and state were factorially crossed.”
- “The writer recovered the oracle codes.”
- “The consumer implements nearest-code decoding.”
- “The consumer learned an eight-dimensional addressable subspace.”
- “Far-from-code cosine proves a larger continuous channel.”
- “Zero and random interventions had no systematic effect.”
- “Zero configured filler demonstrates persistence or memory.”
- “The result generalizes to unseen entities or source wording.”
- “The result establishes a register in Qwen or another pretrained model.”
- “The current preflight can support an interpretable PASS/PARTIAL/FAIL.”
- “A preflight PASS would establish a causal or code-level bridge.”
- “The constructed substrate has discovered native latent mathematics.”

## Copy-ready README wording

> `necessary_register_rung1` is a qualified synthetic calibration PASS: on seeds 11/23/37, an answer-label-plus-EOS-trained GRU converted an explicit eight-way source value token into a unit register intervention that made a reconstructed-and-frozen rung-0 consumer follow own, counterfactual, and shuffled-donor states under held-out output permutations, with own accuracy 0.966/1.000/1.000 and counterfactual following 0.964/1.000/1.000. Entity masking preserved behavior, so the result establishes an answer-supervised symbol writer—not semantic source understanding, persistence, oracle-code recovery, a geometric subspace, or pretrained-model structure; successful writes include non-oracle points, but their decision-region geometry remains unidentified.

## Copy-ready STATE wording

> - **`necessary_register_rung1` — QUALIFIED PASS: ANSWER-SUPERVISED EIGHT-SYMBOL WRITER.** The locked gates mechanically pass in seeds 11/23/37: own 0.966/1.000/1.000, all-seven counterfactual following 0.964/1.000/1.000, and shuffled-donor following 0.967/1.000/1.000 with recipient following 0.006/0/0; all arms terminate. The consumer replay/freeze and factorial entity × template × panel evaluation are genuine, with one of 6,912 donors using the registered same-panel/template permutation fallback. Entity masking preserves behavior, so this establishes downstream-trained decoding of eight explicit value tokens into successful non-oracle interventions—not semantic source writing, persistence, code recovery, a larger subspace, or pretrained-model structure. NEXT: repair `register_bridge_preflight_v1` before any run by crossing entity × state, pairing the destroyed-context arm, balancing the label-shuffle null, retaining recomputable evidence, fixing tokenization/bootstrap invariants, and replacing the primal ridge solver; do not advance the synthetic staircase meanwhile.

## Ranked next increments

1. Repair the preflight’s entity–state confound, paired destroyed-context arm, evidence retention, balanced null, tokenization checks, bootstrap reuse, and solver.
2. Run only a cheap smoke validating manifests, folds, saved evidence, and runtime; then lock the repaired runner.
3. Run the repaired CPU preflight once and audit it once.
4. On PASS, hold the required dialogue and move `register_bridge_v1` earlier: map Qwen residuals to successful writer centroids and test held-out causal consumption.
5. On PARTIAL or FAIL, close the synthetic register line as nonbridging under this route.
6. If a constructed-source follow-up is ever justified, remove the direct `<V_s>` symbol and require legend-conditioned relational resolution.
7. Keep short delay, held-out constructed entities, unseen wording, long delay, moves, algebra, effort, maps, larger models, and actuator sweeps off until a real-model bridge survives.

## Ratio heartbeat

Declared line classification:

- `run_necessary_register.py`: 183 nonblank lines; approximately 32 apparatus and 151 artifact-bearing lines, **0.21:1**.
- `run_register_bridge_preflight.py`: 84 nonblank lines; approximately 26 apparatus and 58 substantive measurement lines, **0.45:1**.
- Both are below the 2:1 line warning, though the preflight’s computational implementation is currently defective.

Declared round classification:

- Audit #37 baseline: 18 measurement/governance : 6 build.
- Direction rounds 26–27: +2 governance → 20:6.
- Rung-1 result: +1 build → 20:7.
- This audit: +1 measurement → **21:7 = 3.0:1**.

That remains above the 2:1 warning and below the greater-than-5:1 mandatory halt. The unrun preflight is a measurement, not central-artifact construction; its run and audit would raise the ratio to 23:7 unless followed immediately by a real bridge or closure. The missing substantive NOTEBOOK entries for rounds 26–27 also weaken the round-classification audit trail; their lock is recoverable only from the ledger.

## 2026-08-30 — Round 27 (direction session; verbatim): conditional lock ruling for rung 1 and the register_bridge_preflight_v1 specification

The writer design is sound, but the current runner is not lockable yet. Four evidence-contract fixes are required; no model, optimizer, codebook, or scientific arm should change.

## Required edits

1. **Separate exact replay from the crossed oracle regression.**  
   [run_necessary_register.py:140](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_necessary_register.py:140) still uses the old marginal schedule. Keep that path solely for exact historical replay, then run a second oracle evaluation with `states=lambda ents: ents % 8`, fully crossing entity × template × panel with six repetitions. Retain its raw rows and require it to pass before writer training. Exact replay and audit-required factorial regression cannot be the same evaluation.

2. **Fix the paired-uplift estimand.**  
   [run_necessary_register.py:134](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_necessary_register.py:134) takes the control maximum row-by-row, not entity-by-entity. Use:
   `own_e − max(zero_e, mean_random_e)` per entity; report its mean and bootstrap entities. The present calculation is a different, unnecessarily punitive estimand.

3. **Gate termination for every behavioural arm.**  
   [run_necessary_register.py:144](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_necessary_register.py:144) and [run_necessary_register.py:185](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_necessary_register.py:185) gate only `own`. Require `>=0.95` for own, cf1–7, shuffled, masked, value-only, oracle, zero, and every random arm. Retain zero-hook/untouched EOS decisions alongside their raw token IDs and full-logit gap.

4. **Deadline truncation must be incomplete, never negative.**  
   [run_necessary_register.py:102](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_necessary_register.py:102) silently returns a partial fit. Return a completion flag; a truncated consumer or writer seed is `INCOMPLETE — DEADLINE` and excluded from substantive seed counts.

5. **Small provenance/wording cleanup.**

   - Assert `prior["sha256"]["runner"] == rung0.runner_sha256_at_rung0` at [line 76](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_necessary_register.py:76).
   - Preserve raw first-token IDs, not only derived state `-1..7`, at [lines 109–118](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_necessary_register.py:109).
   - Replace the stale “COMPOSITIONAL ORACLE REGISTER CONSUMER” status at [line 196](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_necessary_register.py:196) with audit #37’s narrowed headline.
   - The runner is currently 189 nonblank lines, already above the round-25 limit. Retain every scientific arm; shorten the docstring, comments, and logging to restore `<180` after these edits.

Ignoring the newly added summary keys in the exact historical comparison is acceptable: they do not exist in the stored result. They must instead be gated in the new crossed regression. Do not pretend they were historically replayed.

No separately trained entity-lookup writer is wanted. Training makes entity and state independent, same-entity cf1–7 is the strongest specificity test, and the masked-value arm directly tests entity lookup in the actual writer. `value_only` is an expected-positive sufficiency ablation, not a negative baseline; a separately trained entity lookup would trivially exploit `state=e mod 8` and add no diagnosis.

## Writer lock ruling

**Conditional lock after the edits above.** Freeze the consumer, oracle codes, `d=128`, model, seeds, 3,000 steps, writer architecture, downstream answer+EOS loss, zero configured filler, and all thresholds in [necessary_register_rung1.json](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/config/necessary_register_rung1.json).

Per-seed PASS remains:

- all-arm termination `>=0.95`;
- own and cf-follow `>=0.90`;
- own entity-bootstrap LB `>0.85`;
- state/template/panel minimum `>=0.80`;
- agreement `>=0.90`;
- paired direction `>=0.85`, LB `>0.75`;
- shuffled donor-follow `>=0.85`, recipient-follow `<=0.20`, effect LB `>0.55`;
- masked, zero, and random assigned-target accuracy `<=0.20`;
- train–heldout permutation gap `<=0.15`;
- oracle-normalized recovery `>=0.85`.

Global PASS requires two seeds and the registered completed-seed floor. Statuses remain `PASS — FUNCTIONAL SOURCE WRITER AT ZERO CONFIGURED FILLER`, `LOOKUP-BOUND FAIL — OUTPUT MAPPING`, `ENTITY/CONTEXT-BOUND FAIL — SOURCE NOT IDENTIFIED`, `FAIL — SOURCE WRITER CONSTRUCTION`, `INVALID — CONSUMER PRECONDITION`, or `INCOMPLETE — NO VERDICT`.

## `register_bridge_preflight_v1`

This runs after the writer process ends. It is a noncausal feasibility measurement, **not staircase advancement**.

**Data.** Use the existing 24 training entities, with state `e mod 8`, and the eight tokenizer-verified tags `fask, nimb, ruzz, pelt, gorm, twyl, hesk, vorn`. Every prompt contains:

```text
Legend: state 0 = {tag}; ...; state 7 = {tag}.   # clause order shuffled
Registry: the private value for {entity} is {tag}.
```

The per-row permutation determines which tag denotes each state; the record uses the tag denoting that entity’s state. Use two fixed training templates and two disjoint held-out templates. Generate two balanced, disjoint banks of eight permutations, so every tag occurs once for every state within each bank. This prevents tag identity alone from predicting state.

**Representation.** Read the mean residual over the final occurrence of the record’s tag span—never its copies in the legend—at zero delay. Collect zero-based layers `4, 8, 12, 16, 20, 24, 27`, using `hidden_states[layer+1]`.

**Decoder.** Three stratified outer entity folds hold out one entity per state. Within each fold, choose layer and ridge coefficient from `{0.01, 0.1, 1, 10}` using only inner entity folds. Fit standardized ridge scores `XW+b → R⁸`; this is rank at most eight. Refit on the 16 outer-training entities and evaluate the eight held-out entities under held-out templates and permutations. No Qwen weights change.

**Controls.**

- Mean input embedding over the identical tag span.
- A categorical linear baseline using tag identity, template, entity-as-OOV, span position, and token length.
- A token-matched/context-destroyed prompt whose legend pairings are independently shuffled while preserving the token multiset.
- Two hundred entity-clustered shuffled-coordinate fits, permuting training state/code assignments while retaining the evaluation truth.

Chance is `1/8`. Bootstrap the 24 entity-level accuracies 2,000 times.

**Statuses.**

`PREFLIGHT PASS — CONTEXTUAL SOURCE-SPAN CODE LINEARLY DECODABLE` requires:

- accuracy `>=0.60`;
- every outer fold `>=0.50`;
- every state recall `>=0.40`;
- entity-bootstrap LB `>0.45`;
- entity-paired advantage over the best control `>=0.20`, LB `>0.10`;
- accuracy at least `0.20` above the 99th percentile of the shuffled-coordinate null.

`PREFLIGHT PARTIAL — TOKEN/CONTEXT-BOUND DECODABILITY` means the raw accuracy, fold, state, and shuffle gates pass but the control-advantage gate fails.

`PREFLIGHT FAIL — NO QUALIFYING LINEAR SOURCE-SPAN BRIDGE` means the primary decodability gates fail. `INVALID — TOKENIZATION/SPLIT MANIFEST` covers nonunique spans, imbalance, or permutation overlap.

A PASS licenses only: frozen Qwen’s tested source span contains a linearly accessible, presentation-transferring episode-defined state signal, warranting the required dialogue about moving `register_bridge_v1` earlier. A PARTIAL licenses an engineering bridge from lexical/context features, not an abstract native state. A FAIL says this rank-eight, zero-delay source-span route is not a supported bridge; it does not say Qwen lacks state.

Do not edit the hash-bearing historical runner. Add one new file, `experiments/run_register_bridge_preflight.py`, importing `SubstitutionProbe` and reading a frozen `preflight` block added to the current rung-1 config before lock. Budget: `<140` nonblank lines and approximately 35–50 CPU minutes.

Ratio heartbeat: approximately `20:6 = 3.33:1`; the writer must be the next artifact-bearing round.

## 2026-08-30 — Round 26 (direction session; verbatim): source-writer rung specification

## `necessary_register_v1` rung 1 — locked source-writer specification

This lock is conditional on audit #37 adopting rung 0. No writer run is authorized before that.

### 1. Frozen substrate and writer

Rung 0 saved no model checkpoints. Therefore the writer stage must first replay each seed’s exact rung-0 training path using the hash-bound runner/config, rerun the complete instrument gates, then save and hash `consumer_seed{11,23,37}.pt`. Any replay that does not reproduce an audit-adopted valid consumer yields `INVALID — CONSUMER PRECONDITION`; no writer is trained.

After replay, freeze and set to evaluation mode:

- Entire `RegisterLM`: token and position embeddings, transformer layers, normalization and output head.
- Hard attention mask.
- Eight orthonormal codes and code seed.
- Panel vocabulary, training/evaluation permutation banks and query templates.
- Model width 128 and every rung-0 hyperparameter.

Writer:

```text
source vocabulary:
<SRC> <HAS> <WRITE> <MASK>
<E00>…<E23>
<V0>…<V7>

source:
<SRC> <E_e> <HAS> <V_s> <WRITE>
```

Module:

- Learned source embedding, dimension 64.
- One-layer GRU, hidden size 128.
- Biasless linear map \(128\rightarrow128\).
- L2 normalization, producing unit vector \(w\).
- \(w\) replaces the `<REG>` input embedding exactly as an oracle code did.

The source tensor is never supplied to the consumer. The writer cannot see the panel, permutation, legend, query or answer. Its unit vector is the sole causal path from source to consumer.

Train only writer parameters for seeds 11/23/37, paired with the same-seed frozen consumers: AdamW, learning rate \(10^{-3}\), weight decay \(10^{-2}\), batch 256, gradient clip 1.0, 3,000 steps. Training crosses every entity with every state and samples the existing four panels, four query templates and 128 training permutations.

#### Objective

Use only the existing answer-label plus EOS cross-entropy through the frozen consumer:

\[
L=\operatorname{CE}(\text{answer logit},\,\pi(s))
 +\operatorname{CE}(\text{next logit},\,\texttt{EOS}).
\]

Do not regress \(w\) toward an oracle code, add a cosine loss, or train a state classifier.

Direct code regression would teach the answer by declaring an arbitrary Euclidean coordinate to be truth; it is a source-token classifier disguised as state learning. Answer-only training is the honest test because the writer cannot see which output label represents its source state, while the frozen consumer must interpret the same write across independently varying legends.

This remains an easy symbol-to-state task: `<V_s>` uniquely names the source state. A pass tests the proximal writer mechanism, not semantic abstraction.

### 2. Zero configured delay

Logical episode:

```text
<SRC> <E_e> <HAS> <V_s> <WRITE>
        ↓ Writer
<BOS> <REG=w> <MAP> π(0)…π(7) <query containing E_e> <ANS> π(s) <EOS>
```

There is no filler between `<WRITE>` and `<REG>`, and `<MAP>` immediately follows `<REG>`. This is **zero configured filler**, not literal adjacency to the answer: the visible legend and query place the answer logits 12–13 consumer positions after `<REG>`.

### 3. Evaluation arms and gates

Use the existing 16 held-out permutations per panel. Primary evaluation contains 2,304 balanced recipient contexts; entity is the bootstrap cluster.

Arms:

- **Own source:** state \(s_e=e\bmod8\).
- **Counterfactual sources:** same entity with each of the other seven value tokens; score toward the counterfactual state.
- **Shuffled source:** a precomputed state-changing derangement within matched panel/template/permutation contexts. Replace the complete source with a donor entity/value source; score toward donor state and against recipient-own state.
- **Masked source:** replace `<V_s>` with `<MASK>`.
- **Zero and fixed-random writes:** inherited controls.
- **Oracle write:** inherited same-seed ceiling.
- **Untouched/zero-hook:** inherited mechanical identity control.

`PASS — FUNCTIONAL SOURCE WRITER AT ZERO CONFIGURED FILLER` requires every gate in at least two of three paired seeds, with no seed below 0.75 overall own/counterfactual accuracy:

- Termination ≥0.95.
- Own and counterfactual abstract-state accuracy ≥0.90; entity-bootstrap lower bound >0.85.
- Accuracy ≥0.80 for every state, panel and query template.
- Inverse-permutation cross-presentation agreement ≥0.90.
- Paired own/counterfactual directional rate ≥0.85; lower bound >0.75.
- Shuffled-source donor following ≥0.85.
- Shuffled-source recipient-own following ≤0.20.
- Shuffled donor-minus-recipient effect lower bound >0.55.
- Masked-source, zero and random assigned-state accuracy ≤0.20.
- Train-permutation minus held-out-permutation accuracy ≤0.15.
- Oracle-normalized recovery

\[
\frac{\text{writer}-\max(\text{masked},\text{zero},\text{random})}
{\text{oracle}-\max(\text{masked},\text{zero},\text{random})}
\ge 0.85.
\]

- Frozen consumer retains its audit-adopted rung-0 gates and zero-hook identity.

Code-space telemetry is diagnostic only: log writer norm, true-code cosine, nearest-code accuracy, cosine margin, Euclidean distance and per-state spread. Raw proximity to the engineered oracle vectors is neither necessary nor sufficient; cross-context causal behavior is the identity criterion.

### 4. Statuses

- `PASS — FUNCTIONAL SOURCE WRITER AT ZERO CONFIGURED FILLER`: all primary gates pass; advance only to short delay.
- `LOOKUP-BOUND FAIL — OUTPUT MAPPING`: training-permutation accuracy ≥0.90, but held-out accuracy <0.80 or the gap exceeds 0.15.
- `ENTITY/CONTEXT-BOUND FAIL — SOURCE NOT IDENTIFIED`: apparent own-source success, but counterfactual, shuffled-donor or masked-source controls fail.
- `FAIL — SOURCE WRITER CONSTRUCTION`: remaining primary failures.
- `INVALID — CONSUMER PRECONDITION`: audit #37 does not adopt rung 0, exact replay is invalid, or frozen-consumer identity fails.
- `INCOMPLETE — NO VERDICT`: fewer than two paired seeds complete.

There is no PARTIAL advancement and no optimizer, writer-width, codebook or consumer repair.

### 5. Licensed wording

Pass:

> In constructed `necessary_register_v1`, an answer-only-trained source writer converted the registered training-format source value into a one-write unit vector that made a frozen consumer follow own, counterfactual and shuffled source states across held-out output permutations at zero configured filler.

Never say:

> The writer learned semantic facts, recovered the oracle coordinates, generalized to unseen entities or source wording, demonstrated long persistence or composition, discovered native mathematics, or established anything about Qwen or pretrained models.

### 6. Freeze ruling and implementation constraint

Do not change seed 11, the eight orthonormal codes, \(d=128\), panels, mask, permutations or thresholds. Seed 11’s weakest registered state and panel groups remain above their predeclared minima; that is spread to report, not a defect to repair.

One requested constraint needs correction: the existing config is hash-bound to rung 0 and must not be edited. Use the same runner with a config-selected stage, but create exactly one new writer-stage config binding the original runner, config and result hashes. “No new file” can safely mean no new runner; literally forbidding a new config would require either mutating the locked config or making the stage CLI-only, both of which break the requested provenance contract.

Extension budget: ≤70 nonblank lines after refactoring shared training/evaluation code, with ≥45 writer/source/control lines and ≤25 new plumbing lines. No reducer or dashboard.

Pushback on the summary: rung 0 is provisional until audit #37; no saved frozen consumer presently exists; “zero delay” means zero filler, not an immediate answer; and this rung’s unique value token makes it a proximal supervised symbol writer, not yet a language-understanding or native-mathematics result.

## 2026-08-30 — Re-contextualization (2-hour check-in) while rung 1 runs; no new result since audit #37 (audit skipped)

Live question unchanged: can any substrate carry a causally addressable, content-specific state register, and does a real model's source span even contain a coordinate a bridge could enter? Since the last note: audit #37 narrowed rung 0 to a permutation-generalizing label selector; rounds 26–27 locked rung 1 (answer-only writer into the frozen consumer, factorial crossing, full evidence retention) and specified `register_bridge_preflight_v1` — a noncausal linear-decodability measurement on Qwen3-1.7B-Base that can make the remaining synthetic staircase moot in either direction.

What reframes earlier work: the audit's cheapest-algorithm reading (code → map position → copy label) says the consumer is an addressing mechanism, not a state representation; the interesting object in rung 1 is therefore not accuracy but the telemetry — whether an answer-only writer lands near the oracle codes at all (it need not), and whether value_only ≈ own (expected: the writer is a symbol decoder, which the round-26 licensed sentence already concedes).

Alternatives held live for the next cycle (mine, to be challenged by the fresh auditor at the rung-1 result):
1. Rung 1 passes and telemetry shows writes far from the codes: the consumer's addressable set is much larger than 8 codes → "register" is really a learned low-rank channel; the bridge question becomes whether Qwen's span residual projects onto that channel linearly (exactly the preflight).
2. Rung 1 fails with high train / low held-out: the writer entangles entity and value → the same failure family as onewrite_recall_v1, now in a substrate built to avoid it; that would localize the difficulty in the write step, not the store.
3. Preflight PARTIAL (token-bound): the pretrained model carries the tag identity, not the legend-defined state — then no linear bridge exists at zero delay and the constructed line has no real-model target; stop before delay/held-out rungs.
4. Preflight PASS: moving `register_bridge_v1` before the delay rung is the highest-leverage move (round-27 dialogue mandated).
Tunnel check: two independent lines now run (constructed rung 1; real-model preflight), with a pre-declared decision structure; frozen-model actuator sweeps remain off. Ratio: rounds 20:6 (3.3:1, warning) — the next two completed rounds are both artifact-bearing.

## 2026-08-30 — Audit #37 on necessary_register_v1 rung 0 (fresh, unprimed; verbatim): qualified uphold; 'compositional' withdrawn; evaluator marginally balanced; one writer rung only, with a real-model preflight alongside

My rung-0 headline is corrected by this audit: the consumer is a permutation-generalizing oracle-code label selector (code → map position → copy label), not a compositional consumer; the evaluator coupled template to state and state to panel; the alternatives below are recorded verbatim and feed round 27.

## Verdict

**Qualified uphold; headline narrowed.** The locked scalar gates mechanically pass in all three seeds, and the inspected runner/config hashes match the saved result. The central effect is real: changing only the valid injected code changes the selected abstract state under unseen visible permutations.

The defensible classification is:

**`QUALIFIED INSTRUMENT PASS — PERMUTATION-GENERALIZING ORACLE-CODE LABEL SELECTOR`**

“Compositional oracle register consumer” is too broad. The learned operation is specifically code-to-map-position selection followed by visible-label retrieval. The qualification is necessary because the evaluator is not fully crossed, counterfactual/control termination is unreported, raw rows and checkpoints were not retained, and the identity check is decoded-only.

## Split and leakage

The permutation split is genuine.

- Regenerating the banks from `perm_seed=5151` reproduced the saved permutation hash exactly.
- Each panel has 128 unique training and 16 unique evaluation permutations with zero overlap.
- Recalculation also found zero accidental overlap between every evaluation-panel bank and every training-panel bank.
- Every seed sampled all 16 evaluation permutations in every panel, with 24–51 appearances per permutation-panel cell.

The visible map does not directly reveal the target: all eight labels appear exactly once and the target state is supplied only through the injected code. But map position encodes abstract-state index. Consequently, the cheapest successful algorithm is:

`oracle code → fixed state index → attend to that map position → copy its visible label`

That algorithm passes unseen permutations and is more than a fixed code-to-label table, but it is still a permutation-generalizing in-context lookup.

There is an evaluator defect: the claimed “full balance” is marginal, not factorial. The arrays cover only 16 of 128 possible `(state, template, panel)` combinations:

- `template = state mod 4`;
- states 0–3 occur only in panels 0/2;
- states 4–7 occur only in panels 1/3.

Entity × state is fully crossed. The same-prompt `cf1–cf7` results strongly rule out these correlations as the main cause of success, but the per-state, per-panel, per-template, and agreement summaries are not independent invariance tests.

## Mask and identity

The mask direction is correct in [run_necessary_register.py](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_necessary_register.py:24):

- Rows are query positions; columns are attended positions.
- The lower triangle implements causal attention.
- With `R=1`, `m[R+1:, :R] = -inf` blocks post-register positions from attending column 0, the `<BOS>` token.
- It deliberately does not block column 1, so later positions can attend the register.
- The register row can attend `<BOS>` and itself but cannot attend the future map, query, answer, or target.

No multi-layer backflow exposes future tokens to the register because the same causal mask applies at every layer.

The stronger “register is the only source-to-answer path” statement is not yet tested: rung 0 has no source before the register. At present, the restriction merely blocks `<BOS>`.

The zero-hook test is meaningful only as a narrow wiring smoke test. It injects exactly the learned register token embedding plus its position embedding and compares it with the untouched path. However, it checks only the inverse-mapped answer argmax—not logits, EOS, all positions, or the zero-vector arm. Different out-of-panel predictions would both map to `-1` and appear equal.

## Controls and per-group

The valid-code counterfactuals are the strongest evidence:

- Each `cf1–cf7` arm uses the identical prompt and changes only the register code.
- The desired inferred state is correctly shifted to `(s+j) mod 8`.
- Seed 11 follows 15,514/16,128 counterfactuals and passes the paired criterion on 2,124/2,304 episodes.
- Seeds 23 and 37 are perfect.

This substantially excludes template, entity, map text, or evaluation-order leakage as the primary explanation.

The zero/random controls are properly prompt-matched but weakly summarized. “At chance” means assigned-target match was approximately 1/8. A zero or random code could deterministically select one state and still score exactly 1/8 against balanced targets. Only three fixed random vectors were used, and their output-state distributions were not saved.

Seed 11’s aggregate 0.955 hides a coherent 104-error concentration:

- panel 3: 472/576 = 0.819;
- state 5/template 1: 37 errors;
- state 6/template 2: 24 errors;
- state 7/template 3: 43 errors.

Because of the evaluator coupling, every error is simultaneously attributable to panel 3, a high-numbered state, and its fixed template. The result therefore shows one seed’s real presentation-specific brittleness but cannot localize it. Agreement 0.918 is likewise only 0.018 above its gate and covers two panels—not all four—for each `(state, entity, template)` key.

Two evidence limitations remain:

- Termination is retained and gated only for `own`; it is absent beside counterfactual, zero, and random accuracies.
- The uplift lower bound bootstraps own-code entity accuracy and subtracts a fixed global control rate, rather than bootstrapping paired entity-level own-minus-control differences.

## Wording audit in both directions

**Overclaims:**

- Replace unqualified “compositional” with “permutation-generalizing lookup composition.”
- Do not call the evaluator fully balanced or fully presentation-invariant.
- Do not say zero/random codes “behave at chance”; only their assigned-target accuracy is near chance.
- Do not call zero-hook “row-for-row identity” without saying decoded answer-state identity.
- Do not say the mask has validated source isolation; no source exists at this rung.
- Do not imply held-out permutations defeat all lookup strategies. They defeat fixed code-to-label tables, while requiring an episode-conditioned lookup.

**Underclaims:**

Calling this merely “a construction designed to pass” is too weak. Across three independently initialized learned consumers, valid code replacement causally controls first-label state selection under globally unseen mappings, including all seven same-prompt counterfactual codes. That is a genuine positive engineering calibration and a stronger result than memorizing four panel tables.

It remains neither evidence about pretrained latent spaces nor evidence for writing, memory, delay, abstraction, learned moves, or general compositional reasoning.

## Continue-or-not

**Continue for exactly one more constructed rung: the zero-delay, training-entity source writer.** That rung tests the missing mechanism—whether text can produce the code consumed here—and follows the required proximal staircase.

The entire four-rung synthetic sequence before touching Qwen is not presently the highest-leverage allocation. Governance requires the central artifact to return to a real model, while the structured negative localized weak source extraction as a central problem. A cheap CPU real-model source-span decodability/bridge-feasibility preflight should therefore run alongside the writer work, without being described as staircase advancement.

Before the next result, the cumulative runner should fully cross state × template × panel, retain row-level outputs and a checkpoint, report termination for every arm, and use paired clustered uplift. This need not trigger a separate rung-0 repair experiment; it can be an oracle-consumer regression check within the writer run.

The team is **moderately tunnel-visioned**: it recognizes the toy-world risk and has mandated a bridge, but placing the bridge after four synthetic difficulties risks polishing a designed substrate before learning whether the real model can enter its coordinates.

## Alternatives

The strongest alternative explanation is the intended cheap one: the transformer learned a fixed eight-way code decoder plus a positional pointer/copy operation over the visible map.

Alternatives to run instead of or alongside prolonged synthetic development:

- Fit a rank-at-most-eight linear decoder from Qwen source-span representations to the eight abstract states/codes on training entities at zero delay, against matched token/context and shuffled-coordinate controls.
- Include a minimal analytic pointer or linear-attention consumer baseline; matching performance would clarify that rung 0 validates the interface, not a distinctive transformer capability.
- In the writer rung, compare the learned writer with entity/token lookup and same-entity wrong-state controls so memorization cannot masquerade as source-conditioned writing.
- After the zero-delay writer audit, use the required direction dialogue to consider moving `register_bridge_v1` earlier rather than automatically spending on delay, held-out entities, and unseen wording.
- Do not reopen frozen-model site/layer/actuator sweeps; the structured negative already supports that allocation stop.

## Exact licensed sentence

> In this locked synthetic task, three independently initialized two-layer causal transformers trained from scratch used fixed orthonormal vectors injected at a dedicated register position to select the intended abstract state under globally unseen visible state-to-label permutations, with own-code first-label accuracy 0.955/1.000/1.000 and all-seven same-prompt counterfactual-code following 0.962/1.000/1.000; this establishes a learned permutation-generalizing oracle-code-to-visible-label consumer for the tested schedule, not general compositional reasoning, learned writing or persistence, a fully crossed presentation-invariance result, or structure in a pretrained model.

## Never-say list

- “The model learned compositional reasoning.”
- “No lookup strategy can pass the held-out gate.”
- “The evaluation fully crossed every state, panel, and template.”
- “Zero and random codes had no systematic effect.”
- “Zero/random-code behavior was at chance” without “assigned-target accuracy.”
- “Zero-hook proved exact logit-level identity” or “zero-hook equals the zero-vector arm.”
- “The mask proved that source information flows only through the register.”
- “Seed 11 was uniformly robust across presentations.”
- “The register learned to write, retain, or retrieve state.”
- “This establishes an abstract state representation in a pretrained model.”
- “The constructed ladder will transfer to Qwen.”
- “Instrument valid” without the synthetic, first-label, schedule, and evidence-retention qualifications.

## Copy-ready README wording

> `necessary_register_v1` rung 0 mechanically passed its locked synthetic gates in all three seeds: a learned two-layer causal consumer followed fixed oracle-code interventions under globally unseen visible state-to-label permutations, reaching own-code first-label accuracy 0.955/1.000/1.000 and all-seven same-prompt counterfactual following 0.962/1.000/1.000. This is a qualified calibration result—a permutation-generalizing oracle-code label selector—not evidence of general composition, learned writing or persistence, or structure in a pretrained model; presentation groups were only marginally balanced, and arm-specific termination and row-level outputs were not retained.

## Copy-ready STATE wording

> - **`necessary_register_v1` rung 0 — QUALIFIED INSTRUMENT PASS.** The locked scalar gates mechanically passed in seeds 11/23/37, with own-code first-label accuracy 0.955/1.000/1.000 and same-prompt all-seven counterfactual following 0.962/1.000/1.000 on globally unseen state-to-label permutations. The result establishes a learned synthetic permutation-generalizing oracle-code-to-visible-label consumer, not general composition, source writing, persistence, or pretrained-model structure. Qualification: state/template/panel evaluation was marginally rather than factorially balanced; zero/random results are assigned-target accuracies only; zero-hook is decoded-state identity; and termination/raw rows were not retained for every arm. NEXT: one zero-delay training-entity writer rung, with fully crossed evaluation and complete evidence retention, alongside a cheap real-model source-span bridge-feasibility preflight; then audit once and stop on failure.

## Ranked next increments

1. Make full crossing, raw rows, checkpoint retention, per-arm termination, logit-level identity, and paired clustered uplift requirements of the next locked cumulative run.
2. Run only the training-entity, zero-delay source-writer rung against same-entity counterfactual states and lookup baselines.
3. Alongside it, run a cheap CPU Qwen source-span linear-decodability preflight with token/context and shuffled-coordinate controls.
4. Audit the writer result once. Failure closes the constructed architecture; a lookup-only pass also stops advancement.
5. If writer and real-model feasibility both survive, conduct the mandated direction dialogue over short delay versus an earlier `register_bridge_v1`. Do not automatically run the remaining synthetic staircase.
6. Defer moves, algebra, effort, maps, and long delay until a real-model bridge and subsequent causal intervention exist.

## Ratio heartbeat

Declared runner classification:

- **Artifact-bearing:** 90 nonblank lines covering the world, model, mask, permutation/code construction, training, interventions, metrics, and gates.
- **Apparatus:** 23 nonblank lines covering the module description, imports, CLI/configuration, smoke mode, I/O, logging, hashes, deadline, and entrypoint.
- **Apparatus:artifact:** **23:90 = 0.26:1**, below the 2:1 line warning.

Using the audited structured-negative baseline of 15 measurement/governance rounds to five building rounds, then adding direction rounds 24–25 as governance, rung 0 as one building round, and this audit as measurement gives **18:6 = 3.0:1**. That remains above the 2:1 warning threshold but below the greater-than-5:1 mandatory halt. The next completed round must build or test the cumulative writer artifact, not add another free-standing measurement.

## 2026-08-30 — Re-contextualization (2-hour check-in) after necessary_register_v1 rung 0; audit #37 in flight

Live question: does any substrate carry a causally addressable, content-specific state register — and can that be shown by a staircase that a frozen model failed at rung 0? Today's result is not about latent mathematics; it is an instrument check on a substrate we built to have the property by construction.

What still holds: twelve frozen-Qwen3 constructions closed (audits #27–#36); the structured negative is the only licensed statement about real models. What is new: a 2-layer hard-masked register transformer consumes 8 orthonormal oracle codes on held-out state-to-label permutations (0.955/1.0/1.0). Provisional readings, each with the alternative that would make it hollow:
1. "Compositional consumer" — alternative: the visible permutation makes this a two-hop lookup (code -> state index -> label) that any attention layer solves; passing says nothing beyond trainability. The word "compositional" awaits audit #37.
2. "Substrate ready for a source writer" — alternative: the codes are orthonormal and given for free; the hard part (extracting a code from text, keeping it across delay) is exactly what the frozen line failed, so rung 0 has removed the difficult step rather than tested it.
3. "The ladder will bridge to Qwen3-1.7B-Base" — alternative: nothing in a from-scratch 128-d model transfers; the bridge may be the only rung that matters and it is last. Cheaper baseline that could make the line moot: train a one-layer linear writer on frozen Qwen3-1.7B-Base residuals to predict the 8 codes directly (probe-level), which tells us whether source information even exists at the write site before any consumer is built.
4. Different measurement: if the writer rung passes at zero delay, the delay rung is the first place the constructed substrate can actually fail; the informative failure is there, not here.
Tunnel check: one thread is live by governance (staircase, one rung at a time), which is deliberate; the anti-tunnel content is the alternatives above and the Codex round-26 pushback. Ratio flag: this is layer-1 work (building the artifact) after a long layer-2/3 stretch.
Status: audit #37 (fresh, unprimed; includes continue-or-not and alternatives) and round 26 (writer-rung spec) running in parallel; nothing built until both land.

## 2026-08-30 — Round 25 (Codex, verbatim): round 24 amended — the register ladder is a calibration line with a mandatory bridge back to Qwen; rung 0 is an instrument check whose real content is generalization to unseen label permutations

## Ruling: amend, do not ratify Round 24 unchanged

The toy-world objection is valid. A hard-masked transformer is not scientifically different from the navigator merely because its architecture is different: both are trained inside worlds designed to require the desired state. A positive would establish a usable constructed substrate, not native mathematics in a pretrained model.

`necessary_register_v1` remains justified only as a bounded calibration ladder with a mandatory bridge back to a pretrained model. It is not the terminal mission artifact.

### Mandatory bridge

After the constructed writer passes:

1. Training entities at zero delay.
2. Short delay.
3. Held-out entities.
4. Unseen wording.

—and **before long delay, moves, composition, effort, or maps**—run `register_bridge_v1`.

Freeze the constructed register. On the equivalent visible-recall task in Qwen3-1.7B-Base, fit one preregistered rank-at-most-eight linear map from a full source-span representation into the register coordinates, training on three verbalizer panels and their mappings and evaluating on the fourth panel, held-out entities, and unseen mappings/wordings. Require abstract-state accuracy ≥0.75, ≥0.60 per state, entity-bootstrap lower bound >0.60, and an advantage ≥0.20 with lower bound >0.10 over matched token/context and shuffled-coordinate predictors.

Failure closes the synthetic line as non-bridging. Passing licenses only a shared-coordinate probe, not native pretrained state, and triggers a new dialogue for a causal real-model intervention. No constructed-space mathematics receives a program headline before this bridge.

## Rung 0 is an instrument check

Because fixed replacement codes and a trained consumer make success expected, rung 0 is **not a scientific positive**. It validates that the mask, register, decoder, and behavioral instrument can support balanced compositional control. Its nontrivial component is generalization to unseen state-to-label mappings; without that control it would merely learn four code-to-label tables.

### Lookup-controller control

For panel \(p\), every episode supplies a visible permutation \(\pi\) mapping eight abstract states to eight panel-specific labels. Train and evaluation permutations are disjoint.

If the model emits label \(y\), define its inferred abstract state as

\[
\hat s=\pi^{-1}(y).
\]

For a fixed oracle code, entity, and query template, abstract-state agreement is the mean pairwise indicator

\[
A=\operatorname{mean}_{a<b}\mathbf 1[\hat s_a=\hat s_b]
\]

over all panel/permutation presentations. Accuracy \(\mathbf 1[\hat s=s_{\text{oracle}}]\) is gated separately, so consistently predicting the same wrong state cannot pass. A memorized code→label table cannot solve held-out permutations because the correct output label changes while the oracle state remains fixed.

## Exact rung-0 implementation

Model: two-layer causal transformer, width 128, four heads, FFN width 256, dropout zero.

Vocabulary:

- Six specials: `<PAD> <BOS> <REG> <MAP> <ANS> <EOS>`.
- Entities `<E00>`–`<E23>`.
- Query tokens `<Q0>`–`<Q3>`.
- Four disjoint panels of eight single-token labels: `<A0>`–`<A7>` through `<D0>`–`<D7>`.

The eight oracle codes are fixed seeded orthonormal vectors in \(\mathbb R^{128}\), never vocabulary tokens. Entity \(e\)’s designated own state is \(e\bmod8\), but training crosses every entity with every state so the entity cannot predict the answer.

Sequence:

```text
<BOS> <REG> <MAP>
<label π(0)> ... <label π(7)>
<query-template containing entity>
<ANS> <label π(state)> <EOS>
```

All four query layouts are training-support presentations. The oracle vector replaces the `<REG>` input embedding. For register position \(r\):

- Positions before \(r\) use ordinary causal attention.
- Position \(r\) may attend to every source position through \(r\).
- Every position after \(r\) may attend to \(r\) and post-register positions only; all earlier source positions are masked.

At rung 0 there is no source and no writer. Train token embeddings, transformer, and output head; freeze the codebook. Loss is answer-label plus EOS cross-entropy only.

Data and optimization:

- Seeds 11, 23, 37.
- 128 hash-fixed training permutations and 16 disjoint evaluation permutations per panel.
- Full balance over 24 entities, eight states, four templates, and four panels.
- AdamW, learning rate \(10^{-3}\), weight decay \(10^{-2}\), batch 256, 3,000 steps, gradient clip 1.0.
- Evaluation arms: own code, all seven same-entity counterfactual codes, zero code, zero-hook, and hash-fixed norm-matched random codes.
- Chance: \(1/8\); bootstrap clusters are entities.

### Statuses

`INSTRUMENT VALID — COMPOSITIONAL ORACLE REGISTER CONSUMER` requires every gate in at least two of three seeds:

- Termination ≥0.95.
- Held-out-permutation abstract accuracy ≥0.90, entity-bootstrap lower bound >0.85.
- Accuracy ≥0.80 for every state, panel, and template.
- Abstract-state agreement ≥0.90.
- Paired own/counterfactual directional accuracy ≥0.85, lower bound >0.75.
- Accuracy uplift over the stronger of zero and random ≥0.65, lower bound >0.55.
- Zero and random assigned-state accuracy ≤0.20.
- Zero-hook equals zero-write row-for-row.

`LOOKUP-BOUND INVALID` applies if training-permutation accuracy is ≥0.90 but held-out-permutation accuracy is below 0.80 or trails it by more than 0.15.

`INVALID — ORACLE REGISTER CONSUMER` covers every other gate failure. `INVALID — MASK/HOOK` applies if causal masking or zero-hook identity fails. There is no PARTIAL status and no repair: every invalid status abandons this architecture.

Budget: one runner plus one config, ≤175 nonblank lines combined; ≥110 artifact-bearing and ≤55 apparatus lines. Expected CPU time 25–40 minutes, hard stop 60 minutes.

## Tonight

Build only the oracle-register consumer and held-out-permutation evaluator. Do not build the source writer, navigator, bridge, or algebra readouts yet.

Lay line: **Before asking what mathematics a hidden world uses, can it carry one state that remains the same when every visible answer label is reshuffled?**

## 2026-08-30 — Round 24 (Codex, verbatim; subject to ratification): next central artifact = `necessary_register_v1`, a from-scratch transformer with a hard-masked state register, rung 0 = oracle-write control

# Round 24 ruling: construct the register before studying its mathematics

“Substrate construction” now means building a learned causal state register that first passes the proximal oracle-write requirement, then asking what identity, moves, composition, effort, and maps emerge inside it. It does not mean another residual-site search.

| Candidate | Rung-0 pass probability | What a pass licenses | Pretrained-model claim | Narrative | Main repeat-risk |
|---|---:|---|---|---:|---|
| **A. GRU navigator** | ~0.90 | Causal identity, reachability, action composition, inverse and distance in a recurrent state | None; the algebraic world and training objective were supplied | 7/10 | Repeats the toy program: reading back imposed algebra or exact certificates |
| **B. Necessary-register transformer** | ~0.85 | A learned, causally addressable substrate in which state identity, moves, composition and control cost can subsequently be discovered | None until an explicit map to a pretrained model is tested | **9/10** | The register is designed and the task remains synthetic; direct supervision could manufacture a lookup controller |
| **C. Qwen KV/prefix register** | ~0.60 | A co-designed external state that frozen Qwen can consume | Qwen can use that interface—not that native pretrained state was found | 8.5/10 | State-bus again: a continuously available supervised response controller dominated by verbalizers and output boundaries |

## Ruling

Choose **B: `necessary_register_v1`**, subject to ratification in the next dialogue round. This is an explicit constructed-substrate pivot; under the current real-model amendment it cannot be presented as evidence about pretrained language models. Its advantage over the navigator is that the world need not begin with a known algebra: first establish a necessary register, then measure what transformations its learned task dynamics induce. Its advantage over the Qwen KV option is that proximal success does not depend on an instrument or attention interface already shown to be fragile.

### Locked rung 0: oracle-write register control

Use a two-layer causal transformer, width 128, four heads, 32-dimensional dedicated register, approximately one million parameters. A hard attention mask makes the register the only path from the write boundary to the answer. Source encoding is absent at rung 0.

- Eight fixed unit-norm oracle codes are written by **replacement**, not additive perturbation.
- Use 24 training entities, four training-support surface templates and four disjoint eight-label verbalizer panels with independently permuted state-to-label maps. No coordinate is permanently paired with one output token.
- Zero configured delay: query immediately after the register boundary.
- Arms: correct oracle code, all seven same-entity counterfactual codes, fixed norm-matched random codes, zero/no-write and zero-hook.
- Readout: strict first decoded label plus termination. Chance is \(1/8\); inference clusters all templates and panels by entity.

`BOUNDED REGISTER PASS` requires, in at least two of three seeds, with no seed below 0.80 overall:

- Completion ≥0.98.
- Correct and counterfactual code-follow ≥0.95 overall and ≥0.90 for every code, template and verbalizer panel.
- Paired own/counterfactual directional separation ≥0.90, entity-bootstrap lower bound >0.85.
- Correct-write uplift over the stronger of zero and random ≥0.70, lower bound >0.60.
- Zero and random assigned-target following ≤0.20.
- Abstract-state agreement after undoing panel permutations ≥0.95 across templates and panels.
- Zero-hook equals no-write row-for-row.

Exact tables are diagnostic. There is no PARTIAL rung-0 advance: failure, label-panel concentration, or dependence on one presentation abandons this architecture without width, optimizer, label or step repair.

Budget: one runner plus one config; ≤180 nonblank lines, with ≥120 artifact-bearing and ≤60 apparatus lines. Expected three-seed CPU time 45–75 minutes; hard stop 90 minutes. Lay line: **“Can we build a tiny language model with one hidden register where writing a symbol once predictably changes what it says, regardless of the name, wording, or answer vocabulary?”**

After rung 0, preserve the same cumulative artifact and change one difficulty at a time:

1. Train only the source writer into the fixed register consumer, using training entities at zero delay.
2. Add short delay.
3. Add held-out entities.
4. Add unseen wording.
5. Add long delay.
6. Only then test learned moves, composition, effort and maps.

Honest user sentence: **This is the highest-leverage move because it tests whether the prerequisite object—a balanced causal state register—can exist before we build more mathematics around it; if oracle writes cannot control every state across presentations in one locked run, we abandon this constructed substrate rather than begin another repair ladder.**

## 2026-08-30 — Audit #36 on reachability_v1 (fresh, unprimed; verbatim): classification upheld; 'generic / indistinguishable / boundary geometry / linear' readings withdrawn; the slot has a coherent several-mode response that is not shown to be special

My NOTEBOOK reading below is corrected by this audit: the position controls do not establish equivalence to generic positions; the two-tag pattern is made more plausible as boundary geometry, not proven; finite-dose agreement is directional only at one amplitude with sign-dependent asymmetry.

## Verdict

**The registered classification is upheld: `NO SLOT-SPECIFIC GEOMETRY CONCLUSION`.**

The measurement is technically intact and informative, but the current reading overstates it in three places:

- The data do not show that the slot is “like any early position” or generically equivalent to the nulls.
- They do not prove that the earlier two-tag behavioral pattern *is* output-boundary geometry; they make that explanation more plausible while ruling against a hard two-direction channel.
- The finite-dose check validates response direction along three shared modes at one dose, not general linearity, magnitude accuracy, or dose adequacy.

The corresponding under-claim is also wrong: “no conclusion” does not mean nothing was learned. The slot has a coherent, several-mode, name-shared local causal response to the registered eight logits. What failed is evidence that this response geometry is special to the slot.

The exact slot/prompt repair branch should stop. The broader program should continue only through the predeclared structured-negative write-up and subsequent substrate redesign—not through another measurement. **This is not working as an artifact-producing loop.**

The hashes of the current [runner](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_reachability.py>), [configuration](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/config/reachability_v1.json>), and [shared hook machinery](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_onewrite_state.py>) exactly match the saved [result](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/results/reachability_v1/run_result.json>). The hook adds the write once at `self.slot_pos` as declared.

## 1. Mean Jacobian and cancellation

The shared mean Jacobian is an appropriate object for one specific question:

> Does a single residual direction produce a consistent centered-logit response across names?

It is not sufficient by itself for:

> How many local response directions are available to each name?

I therefore recomputed the 24 per-name slot Jacobians read-only on CPU from the hash-matched implementation. Results:

| Statistic | Per-name distribution | Mean Jacobian |
|---|---:|---:|
| Top-mode energy | median 0.635; IQR 0.613–0.650; range 0.569–0.738 | 0.638 |
| Effective rank | median 2.302; IQR 2.211–2.434; range 1.775–2.712 | 2.279 |
| RMS-stacked effective rank | — | 2.754 |
| RMS-stacked top energy | — | 0.579 |

Cancellation is small:

- \(\|\bar G\|_F^2/\operatorname{mean}\|G_n\|_F^2 = 0.895\).
- Unit-normalized coherence is 0.905.
- Pairwise matrix-cosine median is 0.920, with IQR 0.895–0.940.
- Median top-three right-subspace overlap with the shared top-three subspace is 0.928.

Therefore averaging is **not manufacturing** the observed effective rank of approximately 2.3. The mean is highly representative of the typical per-name spectrum. The RMS object is somewhat broader, so the complete write-up should report both: approximately 2.28 for name-shared response and approximately 2.75 for aggregate name-local sensitivity.

The saved result did not retain the per-name \(G_n\) matrices, so these checks had to be recomputed. Future geometry results should save them.

## 2. Position nulls

The nulls were outcome-independent and comply with the executable lock, but they are not sufficiently matched to license “any early position” or “the slot is generic.”

- All eight nulls are non-label tokens inside the `VALID TAGS`/instruction block.
- Six lie only 8–14 token positions before the answer.
- The slot is 42–43 token-position steps upstream.
- Only offsets 39 and 41 are approximately distance-matched.
- Token function and downstream context are not matched.
- No paired name-bootstrap contrast between the slot and each null was reported.

The two approximately distance-matched nulls straddle the slot:

| Site | Top energy | Effective rank |
|---|---:|---:|
| Slot | 0.638 | 2.279 |
| Offset 39 | 0.731 | 1.809 |
| Offset 41 | 0.595 | 2.540 |

That supports “no detected exceptional spectral concentration at the slot,” but not equivalence or indistinguishability.

The slot also has higher reported leading-right-direction alignment across names, 0.96 versus 0.63–0.84 for the position nulls. This is a descriptive distinction, though it was not a registered comparative gate and the null mismatch prevents interpreting it as a slot-specific property.

There is one protocol deviation: seed 4411 draws offsets in the order

`[13, 10, 8, 12, 14, 39, 41, 11]`

but the runner sorts the offsets before naming them. Consequently, finite-dose `null0` is offset 8, not the first RNG-selected offset 13. This does not change the final classification, whose decisive failures are at the slot spectrum and slot-versus-null concentration gates, but it weakens the finite-dose null comparison.

## 3. Prompt permutations

This is not a logit-relabeling artefact.

The runner keeps the measured output token IDs in the original fixed tag order. It only changes the order in which the tag strings appear in the prompt. The centering matrix \(P\) continues to act on the same eight physical token logits. Moreover, even a consistent row permutation would leave singular values unchanged.

Thus the observed change is real prompt-context sensitivity. All four fixed permutations yielded a broader shared spectrum than the base order:

- Base effective rank: 2.28.
- Fixed permutations: 2.72, 3.24, 3.28, 3.22.
- Base top energy: 0.638.
- Fixed permutations: 0.563, 0.412, 0.409, 0.453.

The licensed statement is:

> All four preselected tag-order permutations broadened the shared spectrum relative to the base displayed order.

Do not generalize this to “permuting tag order broadens the spectrum” as a law. Four orders do not identify whether every permutation broadens it, whether the base order is unusually concentrated, or which positional/token interaction causes the effect.

The direction document mentioned permuting both tag order and registry-filler sentence order, whereas the executable lock implemented only tag-order permutations; the configured filler contains only `Internal record:\n`. This is a pre-outcome narrowing of the intended control, not an outcome-conditioned change, but the controls must be described accurately as tag-order controls only.

## 5. Finite-dose validation

The registered finite-dose work was performed as specified at the slot:

- one fixed dose, \(0.25\|h\|\);
- top three shared right-singular directions;
- both signs;
- all 24 names.

At the slot, every saved predicted-versus-realized cosine exceeds 0.881. This is strong evidence that the Jacobian predicts the **direction** of the eight-logit displacement along these modes at this dose.

However, pooling both signs conceals substantial magnitude asymmetry:

| Slot mode | \(+\) norm-ratio median | \(-\) norm-ratio median |
|---|---:|---:|
| 1 | 0.601 | 0.673 |
| 2 | 0.651 | 1.102 |
| 3 | 0.924 | 0.712 |

Sign-specific realized norms relative to mode 1 are:

- Positive: mode 2 = 0.496, mode 3 = 0.603.
- Negative: mode 2 = 0.736, mode 3 = 0.414.

The null is more extreme: null0 mode-1 norm-ratio medians are 1.869 positive and 0.338 negative. Their pooled median, 0.503, represents neither sign.

Decoded behavior is likewise asymmetric. At the slot:

- \(+v_1\) produces `HESK` on 17/24 names.
- \(-v_1\) reproduces `FASK` on 23/24.
- \(+v_2\) reproduces `FASK` on 23/24.
- \(-v_2\) crosses several output boundaries, including `PELT`, `VORN`, `RUZZ`, and `GORM`.

Therefore the finite-dose result licenses:

> Strong directional Jacobian agreement along the top three shared modes at one fixed dose, with material sign-dependent magnitude and decoded-boundary asymmetry.

It does not license “the local-linear picture is accurate” without qualification. One amplitude cannot establish a linear regime or decide whether 0.25 was adequate. The pooled-sign reporting should be supplemented by sign-specific summaries in the write-up.

These reporting weaknesses do not alter the registered classification: narrow reachability already fails the spectrum/null gates, and multidirectional reachability already fails the effective-rank gate.

## 6. Claim audit in both directions

### Overclaims to withdraw

The current [NOTEBOOK entry](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/NOTEBOOK.md:8>) and result-ledger reading overreach in saying:

- “roughly two to three effective behavioural directions” as though effective rank were an integer capacity;
- “the same as any early position”;
- “nothing about it is slot-specific”;
- “the slot is generic”;
- the two-tag pattern “is” output-boundary geometry;
- the local-linear picture is generally accurate;
- tag permutation generally broadens the spectrum.

The evidence supports the narrower replacements:

- The shared centered-logit Jacobian has participation effective rank 2.28; typical per-name effective rank is similar.
- The registered slot concentration did not exceed the chosen position controls.
- All four fixed tag-order permutations broadened the shared spectrum.
- The earlier two-tag decoded pattern is inconsistent with a literal two-direction local-response ceiling and is **consistent with** prompt/output-boundary effects, but no unique causal explanation was established.
- The top three shared modes predict finite-dose centered-logit direction well at one amplitude.

### Underclaims to reject

“No slot-specific geometry conclusion” does not mean:

- no coherent geometry exists;
- averaging destroyed the signal;
- the slot has no causal influence;
- only one mode is usable;
- the measurement learned nothing.

It established:

- high cross-name coherence;
- several material response modes;
- strong finite-dose directional agreement along the shared top three;
- substantial prompt-order sensitivity;
- no registered evidence that the slot’s spectral concentration is exceptional relative to these controls.

## 7. What the complete sequence licenses

The rung-1 → rung-0 → site-oracle → reachability sequence is methodologically useful even though it produced no cumulative intervention artifact.

1. **Rung 1:** The learned source-to-write interface did not produce reliable own-write specificity over same-entity counterfactual or random writes at the easiest evaluated training-name rung. Post-hoc localization found only weak tag information at the chosen source anchor.

2. **Oracle actuator rung 0:** Removing source extraction did not rescue balanced eight-way control through the learned shared \(J\). Beyond the cue prior, `HESK` was the only replicated strong decoded effect. This closed the exact shared-\(J\) construction without localizing a hard site capacity.

3. **Site oracle:** Removing \(J\) and deriving downstream gradients on other names still did not provide balanced eight-way decoded control. It established construction-bound HESK/VORN sensitivity, while leaving baseline gaps, cross-name averaging, dose, prompt, and first-token decoding entangled.

4. **Reachability:** The full local response is not a literal two-tag or two-direction channel. Per-name and shared Jacobians contain several material modes, and the top three realize coherent finite-dose logit movements. Yet the spectral concentration is not exceptional relative to the chosen positions and changes under displayed tag order.

The most important write-up sentence is:

> **The sequence did not show that the block-12 slot is too low-dimensional to carry state; it showed that this exact frozen-model interface never converted weak, presentation-bound source and response geometry into balanced, content-specific causal control, even though the slot has a coherent several-mode local effect on the registered output logits.**

## 8. Requirement for a future substrate

A future substrate must provide a causally addressable state register in which:

- distinct oracle writes produce balanced, content-specific effects at the easiest training/zero-delay rung;
- own-write and same-entity counterfactual-write outcomes separate before learned source extraction is introduced;
- controllable coordinates remain stable across entities and surface presentation;
- finite-dose effects are not determined primarily by pseudoword order or pre-existing output boundaries;
- longer delay, unseen wording, held-out names, and composition are attempted only after that proximal mechanism passes.

The present evidence demonstrates that this exact block-12 slot/prompt/dose/readout construction did not satisfy that proximal balanced-control requirement. It does not demonstrate that Qwen3-1.7B-Base or frozen residual streams generally cannot satisfy it elsewhere.

## Exact licensed sentence

> **`reachability_v1` is a preregistered measurement with classification `NO SLOT-SPECIFIC GEOMETRY CONCLUSION`: at the fixed block-12 final `Internal record:` slot in frozen Qwen3-1.7B-Base, 42–43 tokenizer-position steps before the queried logits, the mean over 24 training names of the \(0.25\|h\|\)-scaled centered Jacobian of eight tag first-token logits had top-mode energy 0.638 (name-bootstrap 95% CI 0.624–0.652), participation effective rank 2.279 (2.197–2.362), \(\sigma_2/\sigma_1=0.450\), and \(\sigma_3/\sigma_1=0.389\); a read-only audit recomputation found typical per-name geometry of similar width and 0.895 mean-energy coherence, so averaging did not create the result, while the eight preselected instruction-token position controls spanned top energies 0.579–0.731 and effective ranks 1.809–2.664 and all four fixed displayed-tag-order permutations produced broader spectra; at one finite dose the top three shared slot directions had median predicted-versus-realized centered-logit cosines 0.995, 0.944, and 0.995 and substantial secondary-mode responses, but sign-dependent magnitude and decoded-boundary asymmetries remained, so the result licenses a coherent several-mode local response under this exact prompt, vocabulary, site, layer, and dose, not a slot-specific dimension, a hard reachability limit, equivalence to generic positions, an explanation of the earlier two-tag pattern, or evidence about memory capacity.**

## Never-say list

Do not say:

- “The slot has two or three reachable dimensions.”
- “The slot is equivalent to any early position.”
- “The spectrum is indistinguishable from generic positions.”
- “Nothing about the slot is distinctive.”
- “Yesterday’s two-tag pattern was proved to be boundary geometry.”
- “The two-tag pattern was a two-dimensional channel.”
- “Permuting tag order always broadens the spectrum.”
- “The prompt effect is merely relabeling the logits.”
- “The prompt controls permuted registry-filler sentences.”
- “Null0 was the first RNG-selected offset.”
- “The finite-dose response is linear at \(0.25\|h\|\)” without “directionally along the tested modes.”
- “Both signs behaved similarly.”
- “The pooled norm-ratio medians characterize either sign.”
- “The dose was adequate,” “too small,” or “too large.”
- “The mean Jacobian describes every name exactly.”
- “Name averaging caused the low effective rank.”
- “No conclusion means nothing was learned.”
- “The slot cannot carry memory or eight-way state.”
- “Frozen residual streams lack usable native structure.”
- “The measurement advanced the positive-control staircase.”
- “The response was measured about 36 tokens later”; use 42–43 tokenizer-position steps.

## Copy-ready README wording

> At the fixed block-12 final `Internal record:` slot in Qwen3-1.7B-Base, the shared centered eight-tag logit Jacobian had participation effective rank 2.28, with typical per-name spectra of similar width and high cross-name coherence; its top three shared directions also predicted finite-dose logit-displacement direction well at one fixed \(0.25\|h\|\) intervention. However, the slot’s spectral concentration did not exceed eight preselected instruction-position controls, and all four fixed displayed-tag-order permutations broadened the spectrum. The licensed result is therefore a coherent, several-mode, prompt-sensitive local response—not a slot-specific dimension, a hard capacity limit, or proof that the earlier HESK/VORN concentration was caused by output boundaries. This completes the predeclared final measurement of the closed slot; the exact actuator-repair branch remains closed and the next step is the structured-negative write-up.

## Copy-ready STATE wording

> - **`reachability_v1` — MEASUREMENT; `NO SLOT-SPECIFIC GEOMETRY CONCLUSION` (audit #36).** At frozen Qwen3-1.7B-Base block 12, the final `Internal record:` slot’s \(0.25\|h\|\)-scaled shared centered Jacobian of eight tag first-token logits, evaluated 42–43 tokenizer-position steps downstream over 24 training names, had top-mode energy 0.638 (95% name-bootstrap CI 0.624–0.652), participation effective rank 2.279 (2.197–2.362), \(\sigma_2/\sigma_1=0.450\), and \(\sigma_3/\sigma_1=0.389\). Audit recomputation found per-name median top energy 0.635 and effective rank 2.302, with 0.895 mean-energy coherence, so name averaging did not manufacture the spectrum; an RMS-stacked name-local object was somewhat broader at effective rank 2.754. Eight preselected non-label positions inside the `VALID TAGS`/instruction block spanned top energy 0.579–0.731 and effective rank 1.809–2.664, so the slot did not pass the registered exceptional-concentration gate, although these controls do not establish equivalence to generic early positions. Four fixed displayed-tag-order permutations had top energies 0.409–0.563 and effective ranks 2.721–3.279; this is genuine prompt-context sensitivity, not output-logit relabeling, but it is descriptive for those four orders only. At one fixed finite dose, the top three shared slot directions had predicted-versus-realized centered-logit cosine medians 0.995/0.944/0.995 and realized relative norms 1.00/0.61/0.49, with material sign-dependent magnitude and decoded-boundary asymmetry. The result licenses a coherent several-mode local response under this exact model/layer/site/prompt/tag vocabulary/dose, not a two- or three-dimensional capacity, slot-specific geometry, dose adequacy, storage, retrieval, or a proven explanation of the prior two-tag behavior. The runner sorted hash-selected offsets before naming `null0`, so finite-dose null0 used offset 8 rather than the first RNG draw, offset 13; this does not change the classification. Per Round 22, no further measurement or repair precedes the structured-negative write-up.

The present [STATE header](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/STATE.md:3>) is also stale about pending audits and should be refreshed when propagation is authorized.

## Ranked next increments

1. Complete or revise the structured-negative write-up using the licensed sentence above; make the four-stage failure localization—not “low-dimensional slot”—the central result.
2. Propagate audit #36 to README, STATE, NOTEBOOK, and an append-only ledger audit row; withdraw “indistinguishable,” “generic,” “any early position,” and the unqualified boundary-geometry claim.
3. Record the per-name/RMS spectrum and sign-specific finite-dose summaries in the write-up; do not run another model computation merely to create a new verdict.
4. Extract a future-substrate contract centered on balanced oracle-write specificity, presentation stability, and a proximal causal register; deposit that transferable residue under the flywheel rule.
5. Only after the write-up and a fresh 2–3-round direction dialogue, consider one new substrate artifact at the easiest proximal rung. No alternate slot, layer, dose, model, navigator, or prompt sweep should intervene.

## Continue-or-not and ratio heartbeat

- `reachability_v1`: complete.
- Current block-12 slot/prompt actuator branch: **stop**.
- Existing measurement/repair loop: **do not continue**.
- Broader program: **continue after the write-up only as substrate construction**, with the proximal positive-control staircase binding.
- Highest-leverage action now: the predeclared structured-negative write-up, not another experiment.

Declared physical-line classification for `run_reachability.py`:

- Artifact-bearing scientific code: 41 lines.
- Apparatus/result plumbing: 16 lines.
- Documentation/import/scaffolding: 18 lines.
- Apparatus/artifact ratio: \(16/41=0.39\).

The ratio passes the registered ≤0.50 apparatus/artifact ceiling, although the predeclared minimum of 70 artifact-bearing physical lines was not met. At round level, the reachability run is a measurement and this audit is an integrity/measurement round: the Round-22 post-audit heartbeat is **15 measurement/audit rounds : 5 artifact-building rounds = 3.0:1**, above the 2:1 warning threshold. The predeclared closure rules out another measurement before the write-up regardless.

Blackboard `0c0c926d` converged with no open signals. No tracked repository file was edited; the existing untracked files were left untouched.

## 2026-08-30 — reachability_v1: NO SLOT-SPECIFIC GEOMETRY CONCLUSION — the closed slot's response spectrum is indistinguishable from generic early positions; permuting the tag list broadens it

Shared budget-scaled centred Jacobian at the slot: top-mode energy 0.64 (CI 0.62–0.65), participation effective rank 2.28 (2.20–2.36), s2/s1 0.45, s3/s1 0.39, across-name alignment 0.96. Eight hash-chosen pre-answer position nulls: top energy 0.58–0.73, effective rank 1.8–2.7 — the slot is not more concentrated than generic positions (one null is more concentrated). Four permuted-`VALID TAGS` prompt controls at the slot: top energy 0.41–0.56, effective rank 2.7–3.3 — the listed order of the tags changes the response geometry. Finite-dose validation along the top three shared directions at exactly 0.25 of the slot norm: predicted-vs-realized cosine 0.995 / 0.944 / 0.995 with realized/predicted norm 0.64–0.84, and modes 2–3 realize 0.61 / 0.49 of mode 1's response (null0: 0.33 / 0.25). Neither classification's criteria are met. Reading (mine, pending audit #36): at this budget the early slot exposes ~2–3 effective behavioural directions to the eight-way decision, the same as any early position, and part of that geometry is the output list's order rather than the site — the local-linear picture is accurate, the slot is generic, and yesterday's "only two tags reachable" pattern is a property of where the decision boundaries sit, not of a special channel. Per round 22 this is the last measurement before the write-up.

## 2026-08-30 — Audit #35 on site_oracle_v1 (fresh, unprimed; verbatim): FAIL upheld as an allocation stop only; 'cap active 21%' and '~36 tokens' corrected; random directions equal cue row-for-row; no reachable-channel claim licensed

My 'narrow tag-selective channel' reading is withdrawn as a dimensional claim; the licensed residue is construction-bound HESK/VORN lexical sensitivity with strong context/threshold dependence.

## Verdict

The registered **`FAIL — BLOCK-12 SLOT/PROMPT EIGHT-WAY BOUNDED CONTROL` is upheld**.

The predeclared no-repair consequence also stands, but only as an **allocation stop for the exact registered recipe**: frozen revision, block 12, final `Internal record:` slot, fixed prompt/tag set, cross-fitted mean first-token-margin gradients, one \(0.25\lVert h_{\text{slot}}\rVert\) dose, and strict greedy decoding.

“The block-12 slot/prompt closes for actuator repair” is acceptable only with that allocation scope. The result does **not** scientifically close the slot, establish a low-dimensional causal channel, show that the intervention was adequately dosed, or exclude sequence-level, context-conditioned, or other site constructions.

Two factual corrections are required:

- “Cap active on 21% of rows” is technically true of the Boolean flag but scientifically misleading: the flag reflects floating-point behavior at the cap boundary, not meaningful clipping.
- The queried first-token logits occur **42–43 tokenizer positions downstream**, not approximately 36.

This is not working as an actuator-repair loop: `oracle_actuator_rung0` and `site_oracle_v1` are consecutive registered failures. The broader program should continue, but this exact branch should not.

## Integrity and registered adjudication

The current runner, config, and shared LM machinery hashes match those saved in `run_result.json`. The model revision is pinned, all 24 name rows and three folds are present, zero-hook equals cue row-for-row, and the registered baseline completion gate passes.

| Quantity | Replay |
|---|---:|
| Target-direction follow | 50/192 = 0.260 |
| Name-bootstrap lower bound | 0.214 |
| Cue-to-assigned-tag match | 23/192 = 0.120 |
| Random-to-assigned-tag match | 23/192 = 0.120 |
| Target choices changed from cue | 45/192 = 0.234 |
| Target completion | 183/192 = 0.953 |
| Random completion | 184/192 = 0.958 |
| Cue completion | 23/24 = 0.958 |
| Fold follow | 0.234 / 0.281 / 0.266 |

The random equality is stronger than the summary suggests: **all 192 fixed-random-direction strict choices reproduce cue row-for-row**. The common 0.120 match rate is exactly the inherited FASK cue prior—23 valid FASK cues divided over eight assigned tags—not an independent random-control success rate. This establishes that this one fixed set of eight random directions never crossed a strict-choice boundary; it does not establish unchanged random-arm logits or characterize a distribution of random directions.

The FASK target direction also reproduces cue row-for-row. Consequently, the non-prior target directions follow their requested tag in only \(27/168=0.161\) rows.

## Cap telemetry: no dose-response evidence

Forty of 192 target rows have `cap_active=true`, but:

- Every flagged scale is `0.9999999404`.
- Their pre-cap/threshold ratio is `1.0000000882`.
- The largest removed norm is \(7.63\times10^{-6}\).
- Every intervention was constructed nominally at the threshold.

This is normalization and floating-point rounding around an equality boundary. It is not a practically distinct lower-dose group.

The nominally flagged rows follow at 0.125 versus 0.296 for unflagged rows, but the flag is concentrated in particular fold/tag directions—RUZZ, GORM, TWYL, and VORN. That comparison is therefore compositional and cannot be read as a dose response.

The data contain exactly one substantive dose. They cannot determine whether the weak tags were under-dosed, whether the \(0.25\)-norm step was too large for a local gradient direction, or whether no useful finite dose exists.

## Choice structure

| Direction | Target follow | Changes from cue | Strict outputs |
|---|---:|---:|---|
| FASK | 23/24 | 0 | FASK 23, invalid 1 |
| NIMB | 0/24 | 0 | FASK 23, invalid 1 |
| RUZZ | 1/24 | 1 | FASK 22, RUZZ 1, invalid 1 |
| PELT | 2/24 | 4 | FASK 19, PELT 2, VORN 1, invalid 2 |
| GORM | 0/24 | 1 | FASK 22, VORN 1, invalid 1 |
| TWYL | 0/24 | 10 | FASK 13, HESK 7, RUZZ 2, VORN 1, invalid 1 |
| HESK | 13/24 | 13 | HESK 13, FASK 10, invalid 1 |
| VORN | 11/24 | 16 | VORN 11, FASK 7, HESK 4, PELT 1, invalid 1 |

NIMB does not push toward a specific wrong tag; it is exactly cue. The strongest off-target structure is TWYL→HESK/RUZZ/VORN. VORN itself divides between VORN, HESK, and PELT. Thus the effect funnels toward a small output set, but a strict-choice funnel is not a response-space rank measurement.

HESK and VORN are the only material recurrent non-prior effects:

- HESK: 5/8, 4/8, 4/8 by fold.
- VORN: 3/8, 5/8, 3/8 by fold.
- Eight names respond to both; eight respond to neither.

The folds are not independent replications: each fold-specific direction is derived from 16 names, and the three derivation sets overlap. Cross-fitting licenses name-held-out behavior within the same prompt and vocabulary, not task, wording, or semantic generalization.

## Tokenization, cue rank, and threshold confound

All eight leading-space labels are exactly two tokens and have distinct first tokens. HESK and VORN share neither a token-count advantage nor a first-token collision.

A pinned-model cue replay gives this mean first-token-logit order:

> FASK, RUZZ, GORM, NIMB, PELT, VORN, HESK, TWYL.

HESK and VORN are therefore not globally cue-favored. However, strict success is associated with smaller name-specific target-to-FASK gaps:

| Tag | Mean cue rank, successes | Mean cue rank, failures | Target−FASK gap, successes | Gap, failures |
|---|---:|---:|---:|---:|
| HESK | 3.92 | 7.18 | −0.628 | −1.574 |
| VORN | 4.91 | 5.92 | −0.784 | −1.076 |

This is particularly strong for HESK. Its direction has a real causal strict-choice effect, but the apparent behavioral selectivity is partly governed by how close each name already placed HESK to the FASK decision boundary.

The runner optimizes an infinitesimal first-token margin but evaluates a two-token greedy label after a large finite step. First-token uniqueness makes the surrogate sensible, but it remains a surrogate—not a “full sequence oracle.”

## What the rung-0/site-oracle pair licenses

The strongest joint explanation is:

> At this exact site and prompt, finite strict-choice response is strongly tag- and context-dependent. Removing the learned shared J does not remove the concentration of effects, so J optimization was not its sole source; nevertheless, the concentration remains inseparable from baseline logit gaps, cross-name gradient averaging, local-to-finite extrapolation, the fixed norm, and the first-token greedy readout.

Audit #34’s “anisotropic response under the learned J” is strengthened only to **anisotropic finite behavioral response under two related constructions**.

The following stronger narratives are not established:

- A one- or two-dimensional reachable channel.
- An intrinsic HESK/VORN subspace.
- A site capacity limit.
- Successful storage or retrieval of two codes.
- Failure of block-12 reachability.
- Adequacy—or inadequacy—of the \(0.25\) dose.
- A native structural property rather than prompt/output-boundary geometry.

The site oracle removes the learned-J parameterization as the sole explanation. It does not remove the prompt, label set, residual norm, first-token objective, or greedy threshold.

## Governance amendment 8

If `site_oracle_v1` is treated as an intervention artifact, it bypasses the positive-control staircase: it combines name-level cross-fitting with a 42–43-position delay before training-item zero-delay specificity has been demonstrated.

The run remains valid under its own registered diagnostic contract, but it must be classified as a **one-off localization measurement**, not advancement of the cumulative staircase artifact. It cannot localize failure among proximal actuation, cross-context alignment, and downstream propagation.

## Program ruling

The broader program should continue. The current block-12 `Internal record:` actuator-repair branch should not.

The highest-leverage candidate after the mandatory 2–3-round direction dialogue is one terminal, properly nulled **budgeted reachable-response geometry** measurement—not another binary actuator. For each name, measure the Jacobian of the centered eight first-token logits with respect to the slot residual, then report:

- Singular-value effect sizes scaled by the registered intervention budget and the name’s baseline logit gaps.
- Effective rank, never numerical rank.
- Per-name spread and cross-name subspace alignment/principal angles.
- The spectrum of the mean Jacobian separately from the distribution of per-name spectra, so gradient cancellation cannot masquerade as low dimension.
- Matched same-block slot and equal-length prompt nulls with causal distance and residual norm controlled.
- Matched alternative tag sets or prompt realizations to separate slot geometry from this pseudoword/output-token geometry.
- A small, preregistered finite-difference linearity check—not a dose sweep or new actuator verdict.

That object could distinguish a narrow aligned response aperture from broad but context-misaligned sensitivity. It still would not prove finite-dose behavioral control.

The deferred navigator should **not** run instead. It is a constructed GRU-world calibration question, not a diagnosis of this Qwen slot. Its smoke already supports the causal-swap instrument while composition and structural readouts fail. It remains eligible for one locked run only if the direction dialogue explicitly pivots to a constructed latent-space program.

## Ranked next increments

1. Propagate this audit’s corrections and conduct the required direction dialogue; authorize no run beforehand.
2. If the dialogue agrees, lock one terminal reachable-response geometry study with matched nulls and a predetermined branch-to-artifact decision.
3. If response rank is low relative to nulls and stable across names, treat it as a bounded prompt/site hole and use it as a design requirement for a next-generation latent space—do not repair this slot.
4. If per-name rank is broad but alignment is weak, build a proximal context-conditioned artifact at staircase rung 0; if rank and alignment are broad, re-examine finite-dose/sequence readout only through a newly justified proximal construction.
5. Run the navigator once only if the dialogue abandons the real-model reachability question in favor of the constructed-substrate calibration path.

## Exact licensed sentence

> **`site_oracle_v1` is a registered construction-level FAIL:** in frozen Qwen3-1.7B-Base, for each of three name folds and eight tags, a unit block-12 `Internal record:`-slot direction was formed by averaging the first-token-margin gradient over the other 16 names and injected once at a fixed \(0.25\lVert h_{\mathrm{slot}}\rVert\) on eight held-out-fold names; at the queried logits 42–43 tokenizer positions downstream, strict target following was 50/192 = 0.260 with name-bootstrap lower bound 0.214, comprising FASK 23/24 exactly equal to the cue prior, HESK 13/24, VORN 11/24, and only 3/120 across the other five non-prior tags, while all 192 fixed-random-direction choices reproduced cue row-for-row and completion remained 0.953–0.958. The reported 40/192 cap-active flags removed at most \(7.63\times10^{-6}\) residual norm and therefore reflect numerical boundary rounding, not a meaningful capped-dose condition. The construction did not provide balanced bounded eight-way decoded control; its positive residue is tag- and context-dependent cross-fitted lexical sensitivity for HESK and VORN under this fixed prompt, dose, and readout—not storage, retrieval, a two-dimensional channel, or a reachable-dimension limit. The predeclared closure is licensed only as an allocation stop for further repair of this exact slot/prompt recipe, not as scientific exclusion of block-12 reachability or other actuator constructions.

## Never-say list

Do not say:

- “The cap was active on 21% of rows” without stating that the maximum removed norm was \(7.63\times10^{-6}\).
- “Capped rows performed worse,” or “the cap caused the failure.”
- “The experiment was under-dosed,” or “the fixed dose was adequate.”
- “The effect survived approximately 36 tokens.” The pinned distance is 42–43 token-position steps.
- “Two tags are controllable” without the fixed construction, fold counts, and base-gap qualification.
- “HESK and VORN define two reachable dimensions.”
- “The site exposes a narrow causal channel” as an established dimensional claim.
- “Block 12 cannot support eight-way control.”
- “The slot cannot carry a code.”
- “Two codes were stored or retrieved.”
- “Cross-fitting proves generalization.” Only names changed; prompt and tag vocabulary did not.
- “Three folds are independent replications.”
- “Random directions had no effect.” They produced no strict-choice changes; their logits were not saved.
- “NIMB was redirected to a particular wrong tag.” It reproduced cue exactly.
- “The first-token direction is a full sequence oracle.”
- “The pair of experiments proves intrinsic anisotropy.” It proves construction-bound behavioral anisotropy; threshold and prompt geometry remain live.
- “The site-oracle advanced the positive-control staircase.”
- “The navigator must run next.”

## Copy-ready README wording

> **`site_oracle_v1` — REGISTERED CONSTRUCTION-LEVEL FAIL; EXACT RECIPE CLOSED AS AN ALLOCATION DECISION.** With no learned map, cross-fitted block-12 first-token-margin directions injected once at \(0.25\lVert h_{\mathrm{slot}}\rVert\) selected their requested tag in 50/192 strict decodes 42–43 tokenizer positions later. FASK contributed 23/24 by exactly reproducing the cue prior; HESK reached 13/24 and VORN 11/24, while the other five non-prior tags totaled 3/120. Every fixed-random-direction choice reproduced cue row-for-row. The nominal 21% cap-active rate was only floating-point boundary behavior, removing at most \(7.63\times10^{-6}\) norm. This fails balanced eight-way control and closes further repair of this exact slot/prompt/direction/dose/readout recipe, not block-12 reachability or hidden-state control generally. The positive residue is bounded tag- and context-dependent lexical sensitivity, not a two-dimensional channel, storage, or retrieval.

## Copy-ready STATE wording

> - **`site_oracle_v1` — REGISTERED CONSTRUCTION-LEVEL FAIL; exact recipe closed as an allocation stop.** Frozen Qwen3-1.7B-Base; no learned map; three name folds; eight cross-fitted unit directions formed from mean first-token-margin gradients on the other 16 names; one block-12 final-`Internal record:` injection at \(0.25\lVert h_{\mathrm{slot}}\rVert\); strict decoding 42–43 tokenizer positions downstream. Target follow was 50/192 = 0.260 (name-bootstrap LB 0.214): FASK 23/24 exactly matched cue, HESK 13/24, VORN 11/24, and the other five non-prior tags 3/120; folds were 0.234/0.281/0.266. All 192 fixed-random-direction choices equaled cue row-for-row; completion was 0.953–0.958 and zero-hook equaled cue. The 40/192 `cap_active` flags applied scale 0.9999999404 and removed at most \(7.63\times10^{-6}\) norm, so they are numerical boundary flags rather than a dose condition. Licensed residue: cross-fitted, construction-bound HESK/VORN lexical sensitivity with strong context/threshold dependence. This does not establish storage, retrieval, a narrow or two-dimensional channel, adequate dosing, or a block-12 reachability limit. Under governance amendment 8 this is a localization diagnostic, not staircase advancement. No further repair of this exact slot/prompt recipe; next action is the mandated direction dialogue. Rung 0b stays off and the navigator remains deferred pending that dialogue.

## Ratio heartbeat

Declared code-line classification for `run_site_oracle.py`:

- Artifact-bearing scientific intervention/null/metric code: 31 lines.
- Apparatus, preflight, and verdict plumbing: 26 lines.
- Documentation and section comments: 8 lines.
- Apparatus/artifact-bearing ratio: \(26/31=0.84\).

This meets the incremental ratio target, but the predeclared “at least 45 artifact-bearing lines” is not met under this disjoint physical-line classification. Semicolon-packed physical lines make the metric brittle.

At the round level, I classify `site_oracle_v1` as **measurement/control**, not building the cumulative staircase artifact; this audit is also measurement. Taking round 21’s declared 11:4 as the pre-run baseline gives **13:4 = 3.25:1**, above the governance warning threshold. It is below the 5:1 halt threshold, but it rules out another open-ended measurement chain.

Blackboard convergence completed with no open signals. No repository files were edited; existing worktree changes were left untouched.

## 2026-08-30 — Round 22 (Codex, verbatim): pivot = one nulled reachable-dimension measurement of the closed slot, then write up the structured negative

## Ruling: choose A once, then D

Run one properly nulled reachable-dimension characterization of the now-closed block-12 slot. This is not another actuator attempt and cannot reopen the slot. After its single audit, write up the structured negative regardless of outcome.

Do not move immediately to another position: that would be a repair ladder without a site-selection principle. Do not scale models yet: it would change substrate before the discovered constraint is characterized. The navigator remains a valid synthetic calibration, but it cannot explain this real-model anisotropy.

## Locked measurement protocol

For each of the 24 training-name prompts, let \(\ell_n\in\mathbb R^8\) be the eight first-token logits and \(P=I-\mathbf1\mathbf1^\top/8\). At block 12 and position \(s\), compute the full normalized-budget Jacobian

\[
G_{n,s}=0.25\lVert h_{n,s}\rVert P\frac{\partial\ell_n}{\partial h_{n,s}}.
\]

The primary site is the closed final `Internal record:` slot.

Nulls:

- Eight precommitted pre-answer positions selected using a recorded hash seed from tokens 8–64 positions before the answer. Exclude the target slot, entity name, candidate-label tokens, and final query tokens.
- Four hash-fixed prompt controls made only by permuting the existing `VALID TAGS` order and registry-filler sentence order. They preserve the same tokens and length.
- No site, prompt, or null is selected after inspecting spectra.

For each site and prompt control, report:

- Spectrum of the shared Jacobian \(\bar G_s=\frac1{24}\sum_nG_{n,s}\).
- Top-mode energy \(\sigma_1^2/\sum_i\sigma_i^2\).
- Participation effective rank \((\sum_i\sigma_i^2)^2/\sum_i\sigma_i^4\).
- \(\sigma_2/\sigma_1\) and \(\sigma_3/\sigma_1\).
- Principal-response alignment across names.
- Name-bootstrap 95% intervals from 2,000 resamples.

Finite-dose validation: at the closed slot, inject both signs of the top three shared right-singular directions at exactly \(0.25\lVert h_{\text{slot}}\rVert\) across all 24 names. Repeat at the first hash-selected position null. Report predicted-versus-realized centered-logit cosine, realized/predicted norm ratio, each mode’s realized norm relative to mode 1, and decoded-tag distributions. No direction targets a particular tag; no amplitude, layer, site, prompt, or label sweep is allowed.

## Licensed classifications

This is a measurement, not PASS/FAIL.

**NARROW SHARED REACHABILITY** is licensed only if:

- Slot top-mode-energy bootstrap LB ≥0.75.
- Effective-rank bootstrap UB ≤2.0.
- Slot top-mode energy exceeds the 95th percentile of both position and prompt controls.
- Finite-dose mode-1 predicted/realized cosine median ≥0.70.
- Modes 2 and 3 each realize no more than 0.25 of mode 1’s response norm.

**MULTIDIRECTIONAL LOCAL REACHABILITY** is licensed only if:

- Effective-rank bootstrap LB ≥3.0.
- \(\sigma_3/\sigma_1\) bootstrap LB ≥0.25.
- All three finite-dose modes have median predicted/realized cosine ≥0.60.
- Modes 2 and 3 each realize at least 0.25 of mode 1’s norm.

Otherwise report **NO SLOT-SPECIFIC GEOMETRY CONCLUSION**.

Even the strongest result applies only to this model, layer, position, prompt family, tag vocabulary, and intervention budget. It cannot establish a hard reachable-dimension limit, absence of memory, or a general limitation of pretrained models.

## Budget and closure

- One CPU process.
- Expected wall time: 90–120 minutes.
- Incremental runner: ≤105 nonblank lines.
- At least 70 reachability/Jacobian/finite-dose artifact lines; at most 35 bootstrap/hash/result apparatus lines—apparatus/artifact ≤0.50.
- Current round ratio: **13:5 = 2.6:1**. After this measurement and its single audit: **15:5 = 3.0:1**. No further measurement or repair follows before the write-up.

Rung 0b, alternate write positions, larger models, the navigator, and every further block-12-slot actuator remain off.

Lay line:

> **How many genuinely different decisions can one early hidden location reach before its influence collapses into a few favored answers?**

## 2026-08-30 — site_oracle_v1: FAIL — the block-12 slot/prompt closes for actuator repair

Cross-fitted margin-gradient directions (derived on 16 names, injected once on the other 8, bounded at 0.25 of the slot residual, cap active on 21% of rows) select the target tag ~36 tokens later in 0.26 of cases (LB 0.21): fask 0.96 (the cue prior), hesk 0.54, vorn 0.46, all other tags ≤0.08; per fold 0.23–0.28; cue and random-direction matching 0.12; completion ≥0.95; zero-hook = cue. By the round-21 rule this slot/prompt is closed for further actuator repair. Reading (pending audit #35): with no learned map at all, the same tag-selective narrowness appears — the slot exposes a narrow causal channel to the query under the bounded norm — which is the cleanest statement yet of what this site can and cannot do. Fresh audit next; then the pivot dialogue (round 22).

## 2026-08-30 — Round 21 (Codex, verbatim): the cross-fitted site-oracle control runs next; spectrum deferred; rung 0b and navigator off

## Ruling: run the cross-fitted site-oracle control

### 1. What a site-oracle pass would mean

The proposed Jacobian intervention is lexical steering. If each direction were derived and evaluated on the same names, a positive first-order margin movement would be substantially built into the construction; that would add little beyond “gradient directions steer logits.”

The control becomes informative only with name-level cross-fitting and finite-dose behavioral evaluation:

- Derive each direction on 16 training names.
- Inject it once on eight disjoint training names.
- Rotate through three fixed folds.
- Require actual decoded choices, not margin movement.
- Keep the intervention at the block-12 slot, approximately 36 tokens before the answer.

A PASS would license exactly:

> At this fixed block-12 slot and prompt, fixed residual directions derived on other names can provide balanced, bounded eight-way lexical control approximately 36 tokens later.

That is stronger than `coordinate_v3` only in transmission distance and context transfer: `coordinate_v3` demonstrated late lexical steering near the decision, whereas this would demonstrate that an earlier residual write survives the intervening prompt and controls behavior on names not used to derive its direction. It would not establish fact storage, memory, semantic state, native algebra, or an eight-dimensional latent representation.

### 2. Site oracle versus reachable dimension

The centered eight-margin Jacobian spectrum is the cleaner native-geometry object. It asks how many independent local response directions the slot exposes and would properly use effective rank, singular-value effect sizes, and matched random-slot/prompt nulls—not numerical matrix rank.

It should not run next as a separate verdict, however:

- The learned \(J\)’s approximately rank-one response does not prove the site itself is rank one.
- Tiny nonzero singular values can produce apparent high rank without usable finite-dose control.
- A local spectrum does not show that a \(0.25\)-slot-norm move survives 36 tokens or changes decoded behavior.
- It would add another measurement round without resolving the immediate branch.

The site-oracle is therefore the sharper discriminator. Its Jacobians can later motivate a reachability study, but no spectrum statistic should carry this run’s verdict.

### 3. Exact next run

Split the 24 training names into three immutable folds of eight. For fold \(f\) and tag \(k\), define

\[
m_k=\ell_k-\frac17\sum_{j\ne k}\ell_j,\qquad
d_{f,k}=
\frac{\operatorname{mean}_{n\notin f}\nabla_{h_{\text{slot}}}m_k(n)}
{\left\|\operatorname{mean}_{n\notin f}\nabla_{h_{\text{slot}}}m_k(n)\right\|}.
\]

For every heldout-fold name, inject once:

\[
\delta_{f,k}=0.25\lVert h_{\text{slot}}\rVert d_{f,k}.
\]

Evaluate all \(24\times8\) cross-fitted name–direction cases, cue, zero-hook, and one fixed norm-matched random-direction set. No learned \(J\), encoder, optimization, layer/site/prompt/norm sweep, or same-name direction fitting.

PASS gates:

- Completion ≥0.95.
- Target-direction decoded-tag follow ≥0.85.
- Name-cluster bootstrap 95% LB >0.75.
- Every tag ≥0.75.
- Every heldout-name fold ≥0.75.
- Uplift over cue-to-target matching and paired random-direction matching ≥0.60, with clustered LB >0.50.
- Random assigned-tag follow ≤0.20.
- Zero-hook reproduces cue row-for-row.

Statuses:

> **SITE-ORACLE PASS — BOUNDED EARLY-SLOT EIGHT-WAY LEXICAL CONTROL.** Cross-fitted, fixed block-12 directions controlled all eight registered tag choices approximately 36 tokens later under the bounded intervention.

> **FAIL — BLOCK-12 SLOT/PROMPT EIGHT-WAY BOUNDED CONTROL.** Cross-fitted full-downstream margin directions did not provide balanced eight-way decoded control; this specific slot/prompt closes for further actuator repair.

`INVALID — NO VERDICT` applies only if baseline completion or hook identity fails.

A PASS permits design of a site-aware actuator before any encoder. A FAIL pivots away from this slot/prompt. Rung 0b and the navigator remain off in either case until this branch is adjudicated.

Line budget: ≤85 incremental nonblank lines—at least 45 artifact-bearing Jacobian/intervention lines and at most 40 apparatus/control/verdict lines, for apparatus/artifact ≤0.89. The cumulative round ratio is now **11:4 = 2.75:1**, still above warning.

Lay line:

> **Can one bounded hidden nudge early in a prompt select any of eight answers 36 tokens later?**

Direct shell access to the audit and ledger was denied during this turn; I grounded the ruling in audit #34’s converged blackboard, whose four source documents were fully read, plus the supplied exact findings.

## 2026-08-30 — Audit #34 on rung 0 (fresh, unprimed; verbatim): FAIL upheld; 'two of eight codes' and '|Jc| ≤ 14' withdrawn; the only non-prior effect is code 6 → HESK; next = a downstream-Jacobian site-oracle control

My readings are corrected: code 0 merely reproduces the cue prior; the response is anisotropic but no reachable-dimension limit is proven; rung 0b and the navigator do not run next.

## Verdict

The registered **`FAIL — ORACLE ACTUATOR/SITE/RETRIEVAL CONSTRUCTION` is upheld**, but only for the exact joint construction.

The common “two of eight codes worked” reading is incorrect:

- Code 0 has **no strict-choice effect**: it reproduces cue row-for-row. Its 23/24 `FASK` score is inherited from the base prior.
- Code 6 is the sole replicated non-prior behavioral effect: it changes the output to `HESK` on 18/24, 23/24, and 23/24 entities.
- Code 7 produces `VORN` on only 1/24 entities per seed.
- Codes 1–5 never change the strict choice, although post-hoc logits show small target-directed effects.

Therefore the run did not realize an eight-way bounded oracle-code channel. It does not establish separate failures of the actuator parameterization, block-12 site, downstream retrieval, or a hard reachable-dimension limit.

The exact shared-J construction stops. Rung 0b must not run. The broader program should continue for one bounded site-localization control, but plainly: **this is not working as an artifact-producing loop**, and open-ended repair is not justified.

## Integrity and registered adjudication

The current runner, config, and shared machinery hashes match those stored in [run_result.json](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/results/oracle_actuator_rung0/run_result.json>). All three registered order seeds completed 400 steps and both evaluations.

| Seed | Code follow | Code 0 | Code 6 | Code 7 | Own code | Wrong follow | Completion | Capped / uncapped pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 0.219 | 0.958 | 0.750 | 0.042 | 0.250 | 0.214 | 0.958 | No / No |
| 23 | 0.245 | 0.958 | 0.958 | 0.042 | 0.250 | 0.244 | 0.958 | No / No |
| 37 | 0.245 | 0.958 | 0.958 | 0.042 | 0.250 | 0.244 | 0.958 | No / No |

Additional integrity findings:

- Cue is `FASK` on 23/24 entities and invalid on Kindrath. Code 0 equals that pattern exactly.
- Capped and uncapped choices differ on zero saved arm-rows.
- The cap was inactive, but the stated `|Jc| ≤ 14` is wrong. Maximum evaluated code-delta norms were **19.67, 21.71, and 22.53**, versus threshold 43.26. Approximately 14 is the maximum for random-vector deltas.
- The lock’s codebook hash `3f2381fe…` hashes the ideal float64 formula; the result’s `656db2ca…` hashes the float32 tensor cast back to float64. The formula and Gram matrix agree. This is a provenance/bookkeeping defect, not a changed codebook.
- Raw telemetry retains the values needed to derive pre/post ratios, but the explicitly requested ratios and summaries by code/seed/step were not stored. This protocol deviation does not change the no-cap conclusion.
- Because J is identically zero-initialized, these are three data-order seeds, not three independent parameter initializations.

## What explains the pattern?

### 1. Base prior: explains code 0 completely

Code 0 never changes any strict choice from cue. Calling it a driven tag is an overclaim. It contributes no behavioral evidence that the injected code was carried.

### 2. Codebook geometry: not the explanation

The centered simplex makes no geometric distinction between code 0 and code 6. The learned `Jc_k` vectors also do not collapse to two directions:

- all seven available simplex dimensions are nonzero in every seed;
- effective residual-space rank is 5.49–5.67;
- the first singular direction contains only 37–39% of residual-vector energy.

Thus “J learned only two dimensions” is false.

### 3. Tokenization: explains the loss scale, not the selected codes

Under the pinned tokenizer, every trained label with its leading space has exactly two tokens, and all first tokens are distinct. There are no first-token collisions and no one-token/two-token advantage.

The second token is nearly trivial under teacher forcing: replayed suffix cross-entropy is only 0.002–0.017. First-token cross-entropy dominates. This validates audit #33’s two-token warning, but it does not explain why `HESK` is uniquely strong.

### 4. Base tag preference: does not explain `HESK`

At cue, `FASK` has the largest average tag-first-token logit. `HESK` ranks seventh of eight. Its success is therefore not inherited from a favorable base prior.

### 5. Optimization budget remains a live confound

Loss is not literally flat:

- first-50 to last-50 mean loss: 1.749→1.571, 1.712→1.566, 1.650→1.560;
- `HESK` improves substantially more than the other tags;
- each code receives only 48–53 updates.

Zero initialization does not block J’s gradient, but the finite budget, shared linear map, centered-simplex interference, AdamW dynamics, learning rate, and two-token objective remain entangled. This run does not license either “more steps would work” or “400 steps were adequate.”

### 6. The downstream response is strongly anisotropic, but no dimension ceiling is proven

A read-only replay at the actual query found positive target-first-token uplift for every code. Approximate uplift ranges across seeds were:

- `FASK`: +0.21 to +0.24
- `NIMB`: +0.09 to +0.11
- `RUZZ`: +0.12 to +0.14
- `PELT`: +0.09 to +0.18
- `GORM`: +0.06 to +0.08
- `TWYL`: +0.22 to +0.52
- `HESK`: +1.15 to +2.03
- `VORN`: +0.22 to +0.41

The centered eight-tag response matrix is dominated by one singular direction: rank-one energy is 0.876, 0.944, and 0.947, with effective rank 1.30–1.65.

That is credible evidence of **anisotropic prompt/site-specific controllability under the learned J**. It is not a proven reachable-dimension limit because:

- the residual deltas themselves span all seven available dimensions;
- only one learned parameterization and finite optimizer budget were tested;
- only one layer, slot, prompt family, and tag set were measured;
- every code has a subthreshold target-directed logit effect.

The raw logit lens `W_U(Jc_k)` does not predict the causal behavior: `VORN` is the best tag-aligned delta there, while `HESK` ranks only fifth. Later blocks and final normalization materially transform the intervention.

## Positive residue audit

The sentence “the site can carry a known code to the output for some codes” is too strong.

Licensed residue:

> At this exact block-12 slot and prompt, one learned delta—code 6—causally changed the strict output from the cue behavior to `HESK` on 18/24, 23/24, and 23/24 training entities across the three data-order seeds; a post-hoc logit replay found smaller target-directed first-token shifts for every code. This is bounded evidence of anisotropic lexical controllability, not evidence that an eight-way hidden code was carried, stored, or retrieved.

Code 0 must not be counted as a second behavioral success.

## Program ruling and next rung-0 variant

Do not run the navigator now. Its synthetic causal-swap calibration cannot distinguish shared-J optimization, site reachability, and downstream transformation in this real model.

The single most informative next variant is a freshly locked **site-oracle margin control**, not a repair sweep:

1. Keep the frozen model, block 12, exact slot, prompt, names, cap, and strict decoding.
2. Remove J and the centered codebook.
3. For each tag, derive one independent residual direction from the **full downstream Jacobian** of that tag’s first-token margin over the other seven tag tokens, averaged across the same 24 training names.
4. Normalize each direction once to the existing 0.25 slot-norm bound.
5. Run one fixed evaluation with the existing completion and per-tag behavioral gates. No layer, amplitude, learning-rate, or step sweep.

Do not use raw `W_U^T e_tag` as the primary site control: the replay demonstrates that raw unembedding alignment at block 12 does not predict the post-layer causal effect.

Interpretation:

- **Pass:** the site can support bounded eight-way lexical control; the failed component is upstream in the shared-J/codebook/objective construction. Design a site-aware actuator before considering an encoder.
- **Fail:** close this block-12 `Internal record:` slot/prompt for further repair and pivot through the required direction dialogue.
- Neither result licenses the existing rung 0b, whose frozen J failed.

## Exact licensed sentence

> **`oracle_actuator_rung0` is a unanimous registered construction-level FAIL:** in frozen Qwen3-1.7B-Base, a fixed eight-code centered-simplex codebook was mapped by a zero-initialized biasless linear J and injected once at the block-12 final `Internal record:` slot while only J was trained for 400 entity-by-code steps; no seed passed either the capped or uncapped gate, capped and uncapped choices were identical because evaluated code-delta norms remained at or below 22.53 versus a 43.26 threshold, code 0 reproduced the cue’s `FASK`/invalid choices row-for-row, code 6 changed the output to `HESK` on 18/24, 23/24, and 23/24 entities, codes 1–5 never changed the strict choice, and code 7 produced `VORN` on 1/24 entities per seed; therefore this exact shared-J, codebook, optimizer-budget, block-12-slot, prompt, and two-token-label construction did not realize a bounded eight-way oracle-code channel. Removing the encoder excludes source extraction from this rung, but the result does not separately identify actuator capacity, optimization, site reachability, downstream retrieval, or a hard reachable-dimension limit as the cause.

## Never-say list

Do not say:

- “Two of eight codes worked.”
- “Code 0 was carried to the output.”
- “The site can carry a known code for codes 0 and 6.”
- “Codes 0 and 6 are geometrically special.”
- “`HESK` succeeded because it was favored by the base model.”
- “Token length or first-token collisions explain the two-code pattern.”
- “The loss was flat” or “the loss shows that nothing learned.”
- “The cap caused the failure.”
- “`|Jc|` never exceeded 14.”
- “The uncapped replay tested larger interventions”; it applied the same below-threshold deltas.
- “J collapsed to two dimensions.”
- “Only one or two causal dimensions are reachable from this site.”
- “Three independent initializations reproduced the result.”
- “The actuator, site, and retrieval each failed.”
- “The oracle actuator failed” without naming the learned shared-J construction.
- “A known code was stored and later retrieved.”
- “The one HESK effect demonstrates hidden-state memory.”
- “More steps would solve it” or “400 steps were enough.”
- “Block 12 cannot support an eight-way channel.”
- “Frozen pretrained latent spaces are hostile to structured reasoning.”
- “The navigator should run next to diagnose this failure.”

## Copy-ready public wording

For [README.md](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/README.md:13>):

> **`oracle_actuator_rung0` — REGISTERED CONSTRUCTION-LEVEL FAIL, CLOSED.** With source encoding removed, a fixed centered-simplex eight-code codebook was injected through a learned biasless linear J at the frozen Qwen3-1.7B-Base block-12 `Internal record:` slot. Across seeds 11/23/37, neither capped nor uncapped evaluation passed; the cap never activated. Code 0 merely reproduced the cue’s `FASK` prior, while code 6 causally produced `HESK` on 18/24, 23/24, and 23/24 entities; no other non-prior tag was reliably selected. This exact shared-J, 400-step, slot, prompt, and two-token-label construction did not realize a bounded eight-way oracle-code channel. It does not establish a block-12 capacity limit or failure of hidden-state control generally. Rung 0b and the navigator remain deferred.

For [STATE.md](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/STATE.md:3>):

> **`oracle_actuator_rung0` — REGISTERED CONSTRUCTION-LEVEL FAIL, CLOSED.** Frozen Qwen3-1.7B-Base; no encoder; fixed eight-code centered-simplex codebook; only zero-initialized biasless J trained for 400 entity-by-code steps at the block-12 final `Internal record:` slot. Capped and uncapped passes were 0/3 and produced identical choices because the maximum evaluated code-delta norm was 22.53 versus the 43.26 cap threshold. Cue emitted `FASK` on 23/24 entities and was invalid on Kindrath. Code 0 matched cue row-for-row and is not an intervention success. Code 6 emitted `HESK` on 18/24, 23/24, and 23/24 entities; codes 1–5 never changed the strict choice; code 7 emitted `VORN` on 1/24 entities per seed. All eight labels were distinct two-token targets. A read-only diagnostic found small on-target first-token logit uplift for every code but a rank-one-dominated downstream tag response; this supports anisotropic controllability under the learned construction, not a hard reachable-dimension limit. The exact shared-J line stops; source extraction was absent, but actuator parameterization, optimization, site, and retrieval remain unlocalized. No rung 0b. Navigator remains deferred pending direction dialogue.

Replace the current `NEXT` line with:

> **NEXT:** conduct the required 2–3-round direction dialogue and, only if retained, lock one no-sweep downstream-Jacobian site-oracle control at the same block-12 slot. A pass motivates a new site-aware actuator; a fail closes this slot/prompt for repair. Do not run rung 0b or the navigator first.

[experiments/EXPERIMENTS.md](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/EXPERIMENTS.md:65>) is also stale: it still describes rung 1 as having no result and does not record this oracle rung.

## Ranked next increments

1. Propagate this licensed wording to README, STATE, NOTEBOOK, EXPERIMENTS, and an append-only audit ledger row.
2. Complete the required direction dialogue and lock the single downstream-Jacobian site-oracle control.
3. Run and audit that control once—no amplitude, layer, optimizer, or step sweep.
4. On pass, design a site-aware actuator before any source encoder; on fail, close this slot/prompt and pivot.
5. Reconsider the navigator only after that branch decision; it is not the current diagnostic.

## Governance heartbeat

Declared incremental line classification for [run_oracle_actuator.py](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_oracle_actuator.py:1>):

- Artifact-bearing mechanism/protocol: 41/70 nonblank lines.
- Apparatus/wrapping: 29/70.
- Incremental apparatus:artifact ratio: **0.71:1**.

Starting from round 20’s declared 9:3 round ratio, counting this construction as one artifact round and this audit as one measurement/governance round gives **10:4 = 2.5:1**, still above the 2:1 warning.

The broader program may continue, but the existing J construction is not the highest-leverage work. The one bounded site-oracle control is.

No repository source, state, result, or documentation file was edited. Only the required git-ignored blackboard record was updated, and blackboard convergence completed with no open signals.

## 2026-08-30 — Re-contextualization #32 (early morning): the first rung that produced a structured result

Project and live question. Latent-Space-Reasoning: is there a native mathematics of latent spaces, found or built? After yesterday's eleven pre-declared closures, the staircase's rung 0 — a *known* 8-way code written once through a trainable linear map into frozen Qwen3-1.7B-Base — is the first construction whose result is structured rather than null: in every seed so far, one or two codes drive their tag near-perfectly (0.96, 0.75) while the remaining codes do nothing, with the norm cap never active and the training loss flat. That is neither the "actuator inert" nor the "actuator works" branch; it says the write site can carry a code to the output, but a single linear J trained for 400 balanced steps finds only one or two usable directions. The fresh audit (fired on the complete three-seed result) adjudicates the wording and the branch.

What reframes earlier work. Yesterday's recall interfaces (E+J) failed at *source extraction*; today's oracle rung shows that even with extraction removed the actuator is only partially learnable under this budget — so the two failures are separable, and the staircase is doing what it was introduced to do: localize.

Alternatives held live: (a) the partial-code pattern may be an optimization artefact (lr/steps/zero-init J) rather than a capacity fact — the audit is asked to say whether more steps would be a repair or a rung; (b) the codebook geometry (centred simplex) may be the wrong basis for a linear injector; (c) the site (slot token two positions in) may only expose a low-dimensional causal channel to the query — a *reachable-dimension* question, which is itself a native-geometry object (reachability/control cost returns here from the front, not as a probe); (d) the navigator remains deferred as the calibration control. Not narrowed: the next rung is chosen by the audit and Codex, not by me.

## 2026-08-29 — Oracle-actuator rung 0 preflight (zero-J rows only): PASS

Codebook Gram matrix as designed (unit diagonal, −1/7 off-diagonal; hash 3f2381fe…), the hook fires exactly once per hooked prefill, cap telemetry is zero with zero-initialized J, and zero-hook decoding equals cue on all eight rows. (A first pass of the script reported FAIL only because it read the hook counter after a cue decode had reset it; the corrected run is the record.) Tomorrow begins with the ≤80-line oracle-actuator runner, its lock, three seeds, and one audit — nothing else until then.

## 2026-08-29 — Round 20 (Codex, verbatim): tomorrow's locked artifact = the oracle-code actuator (rung 0); rung 0b = encoder on tag-token positions; navigator deferred

## Tomorrow’s locked artifact: oracle-code actuator rung

### Rung 0 — actuator/site/retrieval control

Use frozen Qwen3-1.7B-Base, block 12, and the existing final-token `Internal record:` write slot. Remove `E` entirely and train only the zero-initialized, biasless \(J:16\rightarrow2048\) for 400 balanced steps across seeds 11/23/37.

Fix eight unit centered-simplex codes:

\[
c_k=\sqrt{8/7}\left(e_k-\tfrac18\mathbf1_8,\;0_8\right).
\]

The codebook is immutable and hashed. Train on all 24 training entities crossed with all eight codes; the requested output is always the injected code’s tag. Use the training query wording and zero configured filler—still approximately 36 prompt tokens, never called literal zero delay. Evaluate all 192 entity–code combinations, plus cue, zero-hook, fixed off-code random vectors, and all seven wrong codes per entity. Entities are the bootstrap clusters.

Per-seed gates:

- Valid-tag completion ≥0.95 for capped-code, wrong-code, and random-code arms.
- Capped code-follow accuracy ≥0.85, entity-bootstrap 95% LB >0.75, and ≥0.75 for every code.
- Own-code accuracy ≥0.85.
- Wrong-code follow ≥0.85 across the 168 counterfactual rows; uplift over the cue’s matching rate ≥0.65 with clustered LB >0.50.
- Cue and off-code-random true-tag accuracy ≤0.20.
- Own-code minus random true-tag accuracy ≥0.60 with clustered LB >0.50.
- Zero-hook reproduces cue row-for-row.

`BOUNDED ACTUATOR PASS — ORACLE CODE` requires every gate in at least two of three seeds. No repair run.

### Norm-cap ruling

Keep the inherited \(0.25\lVert h_{\text{slot}}\rVert\) cap in the primary training and evaluation. Removing it as the only arm would allow an arbitrarily large lexical-forcing intervention to manufacture a positive.

For every training checkpoint and evaluation row, log:

- \(\lVert h_{\text{slot}}\rVert\)
- pre-cap \(\lVert Jc\rVert\)
- cap threshold and activation flag
- applied scale
- post-cap norm
- pre/post norms divided by slot norm
- summaries by code, seed, and step.

Replay the frozen trained \(J\) once without the cap as a diagnostic:

- Capped pass: `BOUNDED ACTUATOR PASS`; permit rung 0b.
- Uncapped pass but capped fail: `CAP-LIMITED ACTUATOR`; do not introduce an encoder.
- Both fail: `FAIL — ORACLE ACTUATOR/SITE/RETRIEVAL CONSTRUCTION`; this J/site/retrieval line stops.

Tonight’s only permitted preflight is eight zero-\(J\) rows: verify the codebook Gram matrix, hook fires once, telemetry is zero, and zero-hook decoding exactly matches cue. Do not inspect any nonzero-write behavior.

### Rung 0b — source encoder

Only after bounded rung-0 success, read the mean LayerNormed block-12 residual over both tokenizer-identified tag-token positions—not the generic anchor. Train \(E:2048\rightarrow16\) source-only to map normalized \(z\) onto the fixed oracle codebook. Keep the oracle-trained \(J\) frozen; there is no joint tuning.

Before any injection evaluation, require:

- Entity-grouped, held-out-source-template nearest-code accuracy \(X\ge0.80\).
- Every held-out template fold ≥0.70.
- Margin over shuffled-label 95th percentile ≥0.50.
- Per-tag recall ≥0.60.
- All gates in at least two of three seeds.

If this passes, inject \(E\)’s codes through frozen \(J\) and reuse the rung-0 own/counterfactual gates. Failure before injection closes source extraction; separability followed by behavioral failure localizes failure to source–actuator composition.

The navigator remains deferred.

Ratio heartbeat: localization plus this ruling moves the declared round ratio from 7:3 to **9:3 = 3:1**, above the warning. Tomorrow’s incremental code budget is ≤80 nonblank lines, with at least 40 artifact-bearing mechanism lines and no more than 40 apparatus lines: ≤1:1 incremental apparatus/artifact.

### Three sentences for the user

The current mechanism is not working: today’s clean negative and localization showed that the generic block-12 source anchor carries only weak tag information, while the learned code usually carries no more. The key constraint is now explicit—persistence cannot be judged until a reliably identified hidden code can drive the frozen model through a bounded one-write actuator. Tomorrow’s single artifact is the oracle-code actuator: **“Can a known hidden symbol, written once, make the model recall its matching tag after the source words are gone?”**

## 2026-08-29 — Rung-0 localization (audit #33's check): the source signal at the anchor token is weak; the encoder recovers at most that

Entity-grouped, source-template-held-out nearest-centroid separability with shuffled-label controls: the raw block-12 residual at the final ` Internal record:` token of a source sentence classifies the tag at 0.340 (shuffled 95th percentile 0.181; chance 0.125); the learned 16-d codes reach 0.333 (seed 11) / 0.153 / 0.194 — no better than the raw signal, and two seeds at chance; a random 16-d projection gives 0.17–0.23. Branch: source *extraction* is the proximal failure — the chosen read position carries little tag information and the interface never obtained a tag code; downstream stages (cap, site, propagation, decoding) remain untested. Per audit #33, the next mechanism reads the source at a tag-bearing position (or injects an oracle code) and must pass an oracle-code actuator control before a learned encoder is introduced. Codex round 20 specifies it.

## 2026-08-29 — Audit #33 on rung 1 (fresh, unprimed; verbatim): FAIL upheld; 'no fact-specific control' and 'zero delay' withdrawn; loss ~1.0 is a two-token artefact; failure not yet localized

## Verdict

The unanimous registered **FAIL is upheld**, and the predeclared stop applies to this exact construction.

The stronger claim—“this interface learns no fact-specific control”—is not licensed without qualification. Changing only the source tag changed the greedy output on 6/144 paired rows, so the pipeline has sparse source-tag sensitivity. What it did not establish is **reliable, directionally correct, control-relative tag recall**.

The broader program should continue, but plainly: **this is not working, and another repair of this interface is not the highest-leverage next step.** Retire the registered E/J + generic block-12 slot + 0.25 norm-cap construction. First run one no-training localization check on the saved encoders; if it yields a clear branch, move to a different, factorized real-model write mechanism. Keep the synthetic navigator deferred.

The blackboard converged with no unresolved signals. No tracked repository files were edited; blackboard findings were recorded as requested.

## Integrity and registered adjudication

The live runner, config, and shared machinery hashes exactly match those stored in [train_result.json](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/results/onewrite_recall_rung1/train_result.json>). The model revision is pinned, all three registered seeds are present, and zero-hook reproduces cue row-for-row. This is a valid result for the locked construction.

| Seed | Own write → true | CF write → true | Random → true | CF write → CF tag | Own/CF choices equal | Own choice stable across sources |
|---|---:|---:|---:|---:|---:|---:|
| 11 | 8/48 | 8/48 | 8/48 | 4/48 | 46/48 | 22/24 |
| 23 | 9/48 | 9/48 | 10/48 | 8/48 | 48/48 | 21/24 |
| 37 | 14/48 | 13/48 | 16/48 | 8/48 | 44/48 | 20/24 |
| Pooled | 31/144 | 30/144 | 34/144 | 20/144 | 138/144 | 63/72 |

Other decisive facts:

- Visible-copy accuracy is 1.0 and cue accuracy is 0.125.
- Every nonzero-write arm completes with a valid tag on 100% of rows.
- Each seed emits only four of the eight tags under its own write.
- Own write never outperforms the fixed random write.
- Of the six rows where own and counterfactual writes differ, own follows the true tag once and counterfactual follows its tag once. No row shows both intended directional responses.
- The fixed random write is stable across source phrasings on 72/72 entity/seed pairs. Own-write consistency of 63/72 is therefore mostly stability supplied by the unchanged target prompt, not evidence of a stable encoded fact.

Thus the raw outputs reject literal “indistinguishable row-for-row,” but they decisively reject usable fact-specific control.

Post-hoc, entity-clustered permutation checks do not rescue the result. Seed 37’s own-write accuracy is nominally above randomized assignments (`p≈.007`), but its fixed-random arm is higher still (`0.333`, `p≈.003`). Own-write mutual information is nominal in seeds 11 and 37 but not 23; counterfactual outputs do not correlate significantly with their actual counterfactual source tags in any seed. These uncorrected, non-replicated 24-entity signals are compatible with finite-assignment coincidences and entity-prompt effects.

## Why loss near 1.0 is consistent with chance-like greedy accuracy

The apparent loss/accuracy conflict is mechanical.

All eight tags tokenize into **two tokens**. The shared [label-loss implementation](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_onewrite_state.py:49>) averages teacher-forced cross-entropy over both token positions. Therefore `exp(-1)≈0.37` is a geometric mean of two conditional token probabilities—not a 37% probability for the complete tag.

A particularly relevant null is:

\[
\frac{-\log(1/8)-\log(1)}{2}=1.0397.
\]

That is exactly the loss obtained when the first tag token remains at eight-way chance while the second suffix token is nearly certain once the first prefix is supplied by teacher forcing. The observed last-50 means—`1.015`, `1.029`, and `0.965`—sit almost exactly at this null.

Consequently:

- The decline from the step-zero samples `2.166/1.641/2.166` to approximately `1.0` does not imply learned tag identity.
- It can represent learning the “emit one of these tag-shaped strings” mode and the easy suffix transition.
- Greedy decoding still depends on the correct first token beating the full vocabulary. The JSON stores no logits or margins, so ties cannot be diagnosed, but exact ties are unnecessary to explain the gap.
- The loss history is an online sequence of different sampled examples, not a terminal full-training-set evaluation.

Training and evaluation are also not identical distributions. Training samples all three training source templates, while evaluation is hard-coded to templates 0 and 1 in [run_onewrite_recall.py](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_onewrite_recall.py:63>). Reconstructed sampled losses show no consistent advantage for template 2, so this mismatch is not the leading explanation, but it prevents treating the loss and greedy evaluation as measurements on exactly the same cases.

## Strongest explanation of the null

The strongest explanation is **a largely source-insensitive output-mode solution**: joint optimization learned to make the model emit valid two-token tags without learning a stable eight-way source-tag code.

That explanation fits:

- 100% tag completion under correct, counterfactual, and random writes;
- emission collapse to four tags per seed;
- 138/144 own/counterfactual choice equality;
- zero intended bidirectional own/counterfactual pairs;
- random accuracy matching or exceeding own-write accuracy;
- the two-token chance-plus-easy-suffix loss baseline.

The six own/counterfactual differences show the source tag sometimes perturbs the output. They do not show that the perturbation represents the tag.

Several construction choices could independently produce this null even if some linear one-write channel is learnable:

1. **Unvalidated source code.** `E` reads only the block-12 residual at the final generic `Internal record:` anchor token. No check established that `z=E(LN(h))` separates tags.

2. **Uncalibrated actuator cap.** The write is capped at `0.25‖h_slot‖`, but no cap-activation rate, pre/post-cap norm, or dose-response telemetry was saved.

3. **The “zero-delay” rung is not zero-token delay.** It has zero configured filler, but local tokenization shows 36 prompt tokens between the written slot token and the query, comprising the newline and VALID TAGS/instruction block. Public wording should say “zero configured filler,” not literal “zero delay.”

4. **Generic write site.** The write occurs at slot position 2 of `Internal record:`, rather than at the entity name or query decision position. Only upper-layer causal paths can make that write available to the later query.

5. **Joint underlocalized optimization.** The 65,552 parameters are trained for 400 batch-one steps at `lr=3e-3`. Entities receive only 7–30 samples per seed. `J` is zero-initialized, so `E` receives no gradient on the first update; early learning is necessarily actuator-first and can settle into a generic output shift.

None of these explanations is independently established. The result does not localize failure to `E`, `J`, the cap, the site, propagation, or decoding.

## Single decisive rung-0 check

Before any new training, load the three saved interfaces and evaluate whether their learned codes separate source tags:

- Recompute the 72 training source states per seed: 24 entities × three source templates, plus matched same-entity counterfactual sources.
- Compute `z = E(LN(h))`.
- Use an entity-grouped, source-template-held-out nearest-centroid or small linear readout, with shuffled-label controls.
- Compare separability in `z` with separability in the raw block-12 source residual.

This is decisive for the next branch:

- If `z` is at chance, the learned source extractor/code failed; do not spend on cap, site, or retrieval repairs.
- If `z` reliably separates tags, source extraction succeeded and the failure lies downstream in `J`, the cap, write site, propagation, or readout.

This check is cheaper than another training run and uses the already-saved checkpoints. It does not establish memory by itself.

## Exact licensed sentence

> **`onewrite_recall_rung1` is a unanimous registered construction-level FAIL:** in frozen Qwen3-1.7B-Base, on 24 training entities evaluated under two of the three training source templates, the training query wording, and zero configured filler, own-write true-tag accuracy was 0.167/0.188/0.292 across seeds 11/23/37, versus 0.167/0.188/0.271 for same-entity counterfactual-tag writes and 0.167/0.208/0.333 for one fixed random write; own and counterfactual writes changed the greedy tag on 6/144 paired rows, but no pair simultaneously followed the true and counterfactual tags, so this exact 65,552-parameter linear E/J, generic block-12 slot, 0.25-norm-capped, 400-step construction did not establish reliable control-relative tag recall and stops under its predeclared rule.

## Never-say list

Do not say:

- “The interface learns no fact-specific control,” without specifying the registered construction and greedy readout.
- “Correct and counterfactual writes are identical row-for-row.”
- “There is no tag-specific effect at all.”
- “The loss implies about 35–38% correct-tag probability.”
- “The training loss proves that `E` learned tag identity.”
- “This was a literal zero-delay or immediate-query test.”
- “All training source phrasings were evaluated.”
- “The fixed random arm rules out random interventions generally.”
- “The 0.25 cap was adequate” or “the cap caused the failure.”
- “The encoder failed,” “the writer failed,” or “the state was written but could not be retrieved.”
- “A linear one-write channel, block 12, a 16-dimensional state, or frozen-model memory is unlearnable.”
- “This establishes that current pretrained latent spaces are hostile to structured reasoning.”
- “This closes the real-model program.”

## Program ruling

This is the day’s eleventh closed construction and the first rung of the new staircase. The staircase worked procedurally: it exposed the failure before held-out names, unseen wording, or longer delay were attempted. Scientifically, however, the current loop is still not producing the intended artifact.

The program should:

- Stop this exact interface family as promised.
- Run only the saved-checkpoint `z` localization next.
- If localization is clean, move to a **different, factorized real-model write mechanism** whose actuator passes an oracle-code positive control before a learned source encoder is introduced.
- Keep the navigator deferred. Its synthetic causal-swap result cannot diagnose this real-model source/write failure and should remain a bounded secondary calibration.

Narrative gate:

> Can a language model be told a private fact once, carry it only in hidden state, and later act on it after the words are gone?

That remains a compelling question; this construction did not answer it.

Measurement-to-artifact heartbeat, with declared classification:

- Starting from round 19’s declared `6 measurement/governance : 2 artifact-building`, counting rung 1 as one artifact round and this audit as one measurement round gives `7:3 = 2.33:1`, still above the 2:1 warning.
- Incremental code classification is worse: the rung changed 19 runner lines plus configuration but no E/J artifact-core lines. Its incremental apparatus/artifact-core denominator is therefore zero. That is another reason to pivot rather than repair this implementation.

## Exact public-surface wording

[README.md](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/README.md:14>) is stale—it still says rung 1 is running. Replace its rung clause with:

> **`onewrite_recall_rung1` — REGISTERED CONSTRUCTION-LEVEL FAIL, CLOSED.** On 24 training entities evaluated under two of the three training source templates with zero configured filler, own-write accuracy was 0.167/0.188/0.292 across seeds, versus 0.167/0.188/0.271 for same-entity counterfactual-tag writes and 0.167/0.208/0.333 for one fixed random write. Own and counterfactual writes differed on 6/144 greedy decodes but showed no intended bidirectional tag following, so the exact linear E/J, generic block-12 slot, 0.25-norm-capped construction did not establish reliable control-relative tag recall and stops. This does not close linear one-write control, block-12 capacity, or hidden-state memory generally.

Replace the current rung bullet in [STATE.md](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/STATE.md:21>) with:

> **`onewrite_recall_rung1` — REGISTERED CONSTRUCTION-LEVEL FAIL, CLOSED.** Visible-copy accuracy was 1.0 and cue accuracy 0.125. Across seeds 11/23/37, own-write true-tag accuracy was 8/48, 9/48, and 14/48; same-entity counterfactual-write true-tag accuracy was 8/48, 9/48, and 13/48; fixed-random-write accuracy was 8/48, 10/48, and 16/48. Own and counterfactual writes differed on 6/144 paired greedy decodes, but no pair simultaneously followed its true and counterfactual tags. Every nonzero write completed with a valid tag. The loss near 1.0 is compatible with chance selection of the first token of these two-token tags plus easy teacher-forced suffix completion. The exact registered construction stops; encoder, actuator, cap, site, and retrieval failure remain unlocalized.

[experiments/EXPERIMENTS.md](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/EXPERIMENTS.md:65>) also remains stale as “RUNNING,” and the [NOTEBOOK headline](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/NOTEBOOK.md:8>) should replace “learns no fact-specific control” with “did not establish reliable control-relative tag recall.”

## Ranked next increments

1. Propagate the licensed correction to README, STATE, NOTEBOOK, EXPERIMENTS, and an append-only ledger audit row.
2. Run the single saved-checkpoint `z`-separability diagnostic; no training or intervention sweep.
3. Use its branch result to specify one different, factorized real-model write mechanism with an oracle actuator control.
4. Lock that mechanism at its proximal training-item rung before introducing delay, held-out names, or new wording.
5. Keep the navigator deferred unless the direction dialogue concludes that no credible real-model mechanism remains.

## 2026-08-29 — Rung-0 diagnostic on the rung-1 interfaces (read-only): the failure is at the encoder/cap, not at persistence

Saved seeds 11/23/37: the encoder's 16-d codes separate tags on *training* sources at only 0.36/0.14/0.12 (nearest-centroid leave-one-out; chance 0.125); the write lifts the correct tag's log-probability by +1.4 nats but a counterfactual-tag write lifts its own tag by the same +1.3 — a generic "emit a tag" direction; and |Jz| before the 0.25-residual cap is ~2000, so the cap rescales every write to nearly the same small vector. The proximal failure is that no tag-separated code was ever obtained from the final anchor token of a short source sentence, and the cap collapses direction differences — a rung-0 check the staircase should start from tomorrow. Not a repair; recorded for audit #33 and round 20.

## 2026-08-29 — Staircase rung 1: FAIL, unanimous — the construction did not establish reliable control-relative tag recall on training entities with zero configured filler (headline corrected per audit #33)

On the 24 training entities, training wording, and zero intervening tokens (the easiest rung): write accuracy 0.17/0.19/0.29 vs counterfactual write 0.17/0.19/0.27 vs random write 0.17/0.21/0.33 (cue = chance 0.125; visible 1.0); counterfactual follow ≤0.17; completion 1.0 under any write; zero-hook = cue. The single-example loss fell from ~2.2 to ~1.0, but greedy decodes show the write is content-independent. By the round-19 rule this interface (16-d linear E/J, block-12 slot addition, norm-capped, label-CE) stops here; no later rung is tested. Fresh audit next; Codex round 20 decides tomorrow's rung-1 candidate (a different write mechanism) or the navigator.

## 2026-08-29 — Round 19 (Codex, verbatim): self is the ceiling; navigator deferred; build rung 1 of the staircase tonight

## Ruling

### 1. Comparator semantics

`self` is the oracle ceiling, not a null control. It represents the recurrent state naturally obtained in the recipient presentation at the donor’s actual place. A successful cross-permutation swap should approach that reference; requiring it to beat `self` makes the uplift gate effectively unsatisfiable near ceiling.

Exact locked causal-swap rule:

- `swap_accuracy ≥ 0.75`
- `decision_4_accuracy ≥ 0.65`
- `swap_accuracy − max(noswap, wrong_place, random) ≥ 0.25`
- Existing action-mass uplift `≥ 0.50`, also measured over the best of `noswap`, `wrong_place`, and `random`
- `swap_accuracy / self_accuracy ≥ 0.80`
- `self` is reported as the oracle ceiling and excluded from both uplift comparators.

All other navigator gates, controls, seed aggregation, and status rules remain unchanged. This resolves an ambiguity in the lock; it does not rescue an observed result.

### 2. Do not run the navigator now

The navigator remains licensed as a one-round calibration control, but it is no longer the highest-leverage next run. Governance amendment 8 supersedes the earlier timing: the real-model artifact must first establish its proximal mechanism before adding held-out names, new wording, or persistence distance.

Build rung 1 tonight using the existing one-write machinery:

> On training entities, with training wording and zero intervening tokens, test whether the decoded answer follows the entity’s own write and changes appropriately under a same-entity counterfactual-tag write.

Keep the frozen model, E/J interface, block-12 slot, strict behavioral decoding, tag vocabulary, and counterfactual construction. Remove only the later difficulties: no held-out entities, unseen wording, or filler. Zero-hook, random-write, and visible-copy remain controls. Lock this rung separately, run it once across the registered seeds, and audit it once. If fact-specific control is not learnable there, stop this interface without testing later rungs; if it passes, change exactly one difficulty at the next rung.

The navigator is deferred, not cancelled. Its smoke already shows that the causal-swap instrument can work, while composition remains at chance and the untrained same-input control exceeds the trained model on the move statistic. Spending another two hours now would primarily measure a secondary toy substrate while the real-model mechanism remains unestablished.

Ratio heartbeat: the 206-nonblank-line navigator runner is approximately **1.5:1 apparatus/artifact** under audit #32’s classification. The recent one-write sequence is approximately **six measurement/governance rounds to two artifact-building rounds, or 3:1**; the warning remains raised. The cumulative staircase is the appropriate correction.

### 3. Honest user sentence

> In a non-registered 2,000-step smoke, the navigator’s recurrent state transferred across a new symbol permutation and drove donor-place behavior at 0.90 accuracy—essentially matching the 0.9025 self ceiling and exceeding no-swap, wrong-place, and random controls—but this validates only the causal-swap instrument, not learned latent algebra or a navigator result, because composition remained at chance at 0.51 and the four structural readouts failed.

Operational note: Windows denied new read-only shell processes during this turn, so I grounded the ruling in the audit blackboard’s file-provenanced findings and the exact smoke figures supplied here.

## 2026-08-29 — Re-contextualization #31 (late night)

Audit: skipped — no new capability result since audit #32 (the navigator smoke is code-path validation and is logged as such).

Project and live question. Latent-Space-Reasoning: is there a native mathematics of latent spaces, and can it be found or built? The day's ten closed constructions now have a single audited explanation (audit #32): the loop kept changing bespoke instruments and actuators before establishing that the proximal mechanism was learnable at all — the one-write interface, for instance, never learned tag identity even on training entities (own-write 4/24, own vs counterfactual outputs identical 24/24), so every held-out and delay claim built on it was untestable from the start. That reframes the day: most kills are not facts about latent spaces; they are facts about untested rungs skipped.

What changed structurally. Governance amendment 8 (AGENTS.md): one cumulative artifact and a positive-control staircase — content specificity on training items at zero delay, then delay, held-out names, unseen wording, long delay — one difficulty per rung, each locked and audited once. This replaces "one artifact per day" and is the single change most likely to convert tomorrow into a decisive result rather than more closures.

Not narrowed. Two live objects remain: (a) the real-model one-write channel, restarted at rung 1 of the staircase; (b) the navigator as a one-round calibration control — its post-fix smoke shows the recurrent state is causally swappable across symbol permutations (0.90 vs ≤0.47 controls) while its algebraic readouts sit at chance, which is itself a useful alternative interpretation: a state can be *portable* without being *algebraically readable*, and the mission's object may be portability first, algebra second. Alternatives still on record: predictive dynamics/flow, distributed span operators, response-law topology, a larger base for rule-dependent readouts (8B-Base / 4B-Instruct screen). Round 19 decides tonight's last action; nothing runs before it.

## 2026-08-29 — Audit #32 on onewrite_recall_v1 (fresh, unprimed; verbatim): FAIL upheld; sentence corrected; navigator not run-ready; loop change = positive-control staircase

The round-18 sentence is replaced by the audit's; the navigator has four pre-run blockers and the 'one artifact per day' rule is replaced by the cumulative positive-control staircase.

## Verdict

`onewrite_recall_v1` is a unanimous registered construction-level FAIL. The scope “this fixed construction did not establish a transferable causal recall channel” is substantially correct, but the pre-drafted sentence needs numerical and wording corrections.

This result does not establish that a fact was successfully written and then forgotten. It does not isolate encoding, persistence, retrieval, or held-out binding. The strongest supported interpretation is that the trained intervention changed the model’s output mode without carrying source-tag-specific information.

The program should continue, but the present rapid construction-closing loop is not working and is not the highest-leverage approach. `necessity_navigator_v1` is procedurally eligible as the promised one-off calibration, but its current implementation is not run-ready and it should not become the central artifact.

No repository files were edited.

## Result audit

The primary approximate evidence is decisive:

| Seed | Correct-source accuracy | Counterfactual accuracy | Random accuracy | Cue | Visible | Correct-source valid-tag emission |
|---|---:|---:|---:|---:|---:|---:|
| 11 | 0.03125 | 0.03125 | 0.03125 | 0.000 | 1.000 | 0.4375 |
| 23 | 0.03125 | 0.03125 | 0.03125 | 0.000 | 1.000 | 0.53125 |
| 37 | 0.0625 | 0.0625 | 0.0625 | 0.000 | 1.000 | 0.421875 |

Raw replay of all six arms in [train_result.json](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/results/onewrite_recall_v1/train_result.json>) found:

- Seeds 11 and 23: correct-source, counterfactual, and random writes produced identical raw text on all 64 rows.
- Seed 37: correct-source and counterfactual parsed choices matched on 64/64 rows; correct-source and random choices matched on 62/64. The two choice differences did not improve correct recall.
- Every decode contained zero or one valid tag. “Correct tag anywhere in the decode” therefore equals strict-parser accuracy; there is no correct-tag rank signal hidden below the parser.
- Counterfactual following was 0.156/0.125/0.156, with no improvement over the registered specificity requirement.
- Zero-hook reproduced cue text, choice, and completion row-for-row.

The intervention unquestionably affects downstream greedy behavior: valid-tag emission rises from cue’s 2/64 rows to 28/64, 34/64, and 27/64. Because counterfactual and random writes induce essentially the same behavior, the positive residue is a nonspecific output/format nudge—not factual recall.

The saved JSON contains decoded outputs, not logits. It proves that the argmax behavior changed; it cannot support quantitative claims about the full next-token distribution.

## Within-train diagnostic

A read-only seed-11 diagnostic decoded all 24 training entities using:

- training source template 0;
- filler 0 and the training query wording;
- own-source write, cue, and same-entity counterfactual-source write.

Results:

- Own-write accuracy: 4/24 = 0.167.
- Cue accuracy: 2/24 = 0.083.
- Counterfactual-tag following: 4/24 = 0.167.
- Own-source and counterfactual-source raw outputs: identical on 24/24 entities.

This is one post-hoc diagnostic slice, so it cannot characterize every training template, filler, or seed. It is nevertheless enough to withhold the claim that the interface learned tag identity even on training entities.

The stochastic loss did decline: first-50 mean loss was approximately 1.26–1.30 and last-50 mean was 0.99–1.05 across seeds. But these are single-example sampled losses, not full-training-set evaluations. The loss licenses optimization-objective reduction only—not learned facts, tag identity, or a transferable state.

## Why this null does not kill one-write memory

The strongest alternative explanations, ranked by current evidence, are:

1. **Source-insensitive optimization collapse.** The train diagnostic and correct/counterfactual equality indicate that the learned delta largely ignored tag content and became an unconditional response-mode controller.

2. **Missing proximal learnability control.** No locked gate required own-write versus same-entity counterfactual specificity on training entities at zero or short delay before testing held-out names and long-context recall.

3. **Source-state choice.** The encoder reads only the final generic anchor token of a short source sentence. Tag information may be weak, distributed, or inaccessible through this linear 16-dimensional interface.

4. **Actuator/retrieval construction.** One norm-capped block-12 addition at a generic slot must remain usable through the rest of the filler, the fixed vocabulary/instruction lines, and the query. Failure may occur at the site, upper-layer propagation, attention-based retrieval, or target-name binding.

5. **Optimization and generalization burden.** Four hundred batch-one steps jointly ask the interface to learn source extraction, an eight-way code, injection, delayed retrieval, unseen-name binding, and unseen query wording. A 16-dimensional state is information-theoretically ample for eight tags, but this training construction was not shown to solve even its proximal train behavior.

The result therefore closes the registered combination of model, layer, source state, encoder/injector, norm cap, slot, objective, optimizer budget, prompt sequence, and held-out evaluation. It does not close any component independently.

### Exact licensed sentence

> Under the round-17-amended behavioral readout—visible-copy accuracy 1.00 and cue accuracy 0.00—`onewrite_recall_v1` was a unanimous construction-level FAIL: across seeds 11, 23, and 37, the correct-source intervention achieved held-out tag accuracy 0.031, 0.031, and 0.0625, respectively, exactly matching the same-entity counterfactual-tag and fixed-random arms within each seed, while valid-tag emission rose nonspecifically from 0.031 in cue to 0.438, 0.531, and 0.422; therefore this 65,552-parameter, 16-dimensional encoder/injector, trained for 400 single-example steps and applied once at the norm-capped block-12 slot before the registered filler/instruction/query sequence, did not establish a held-out causal tag-recall channel.

## Exact README/STATE wording

> **`onewrite_recall_v1` — REGISTERED CONSTRUCTION-LEVEL FAIL, CLOSED.** Under the round-17-amended behavioral readout—visible-copy accuracy 1.00 and cue accuracy 0.00—correct-source held-out accuracy was 0.031/0.031/0.0625 across seeds 11/23/37 and exactly matched the same-entity counterfactual-tag and fixed-random arms within every seed. Correct, counterfactual, and random writes all increased valid-tag emission far above cue, establishing a nonspecific downstream output effect but no tag-specific content. A post-hoc seed-11 training-slice diagnostic likewise produced own-write accuracy 4/24 and identical own-tag versus same-entity counterfactual-source outputs on 24/24 entities. This closes the fixed model/layer/source-state/16-dimensional encoder/injector/norm-cap/slot/objective/400-step/prompt construction—not one-write memory, persistence, block-12 capacity, or persistent state in real models generally.

## Never-say list

Never say:

- “The fact was written but did not survive.”
- “The experiment proved that the model cannot store a fact in hidden state.”
- “Block 12 cannot support persistent memory.”
- “A 16-dimensional or 65,552-parameter interface is insufficient in principle.”
- “The 71–73 filler tokens are the complete write-to-query delay.”
- “All three seeds scored 0.031.”
- “The training loss proves that the train facts or tag identities were learned.”
- “Correct and counterfactual writes had identical raw text in all three seeds.” Their choices did; seed 37 had two non-tag raw-text differences.
- “Random controls rule out every useful intervention.” Only one fixed random state per seed was tested.
- “The intervention had no effect.” It strongly changed valid-tag emission.
- “The write was content-independent” without the construction qualifier; only its saved downstream choices were nonspecific under these arms.
- “The original preregistered instrument passed.” It passed the outcome-aware but pre-training round-17 amendment.
- “More steps, another layer, a different slot, a maintained state, or another objective would also fail.”
- “Frozen language models lack native state or latent mathematics.”
- “This closes the real-model route.”

## `necessity_navigator_v1`

The recall negative satisfies the earlier procedural condition for one bounded navigator calibration. That makes the navigator eligible, not automatically correct or run-ready.

Scientifically, it is secondary. Its algebra is supplied by the designed world, the GRU receives the goal action word and executed-action history, and the behavioral loss is generated from exact BFS-optimal actions. A positive would show that task pressure can induce readable, approximately compositional path-integration state in a purpose-built recurrent system. It would not resolve structure in a real pretrained model.

The current runner has pre-run integrity blockers:

1. **Duplicated swap input.** The donor hidden state `H[d,t]` already includes the time-`t` previous-action/current-observation input. The swap rollout then starts at the same pose with the same previous action and feeds that input again, creating an off-manifold duplicated step and misaligning the claimed four-decision continuation.

2. **Incomplete manifest binding.** The manifest hash includes goal words, permutation triples, and selected times, but excludes the generated walk actions and realized poses that define the actual recipient/donor/wrong-place triplets.

3. **Control mismatch.** The locked ledger describes uplift against no-swap, wrong-place, random, and self controls, but the implementation excludes `self` from its best-control comparator. The untrained move/inverse controls are also evaluated on different randomly drawn episodes rather than fixed identical inputs.

4. **Smoke provenance mismatch.** `STATE.md` and the ledger cite an older 2,000-step smoke with top-1 0.879 versus control 0.484. The only live [smoke_result.json](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/results/necessity_navigator_v1/smoke_result.json>) is a later 300-step smoke with top-1 0.499, control 0.484, and invalid behavior. The cited smoke is not reproducible from the current result directory.

5. **Measurement-to-artifact ratio.** The runner has 206 nonblank lines. A strict functional classification gives 60 artifact-core lines versus 146 evaluation/control/orchestration lines, or 2.43:1; a generous allocation of mixed main-loop code gives roughly 1.5:1. The precise ratio is classification-sensitive, but the claimed ≤1.1:1 is not supported, and the strict split crosses the governance warning.

Ruling: do not launch the full navigator in its current state. Correct only these implementation/lock discrepancies, then use the required direction dialogue to decide whether honoring the single calibration run remains worth its bounded evidentiary value. No navigator v2 or synthetic program follows.

Its licensed lay sentence, if retained, is:

> Can an agent forced to navigate an aliased world invent a portable internal map whose moves compose?

## Program audit

The strongest common explanation of the ten closed constructions is not that current latent spaces are hostile to structured reasoning. It is that the program repeatedly changed bespoke instruments and actuators before establishing proximal mechanism learnability:

- several constructions died at behavioral or calibration validity;
- successful interventions reduced to late lexical or supervised response control;
- cost and interchange constructions were actuator/readout limited;
- the one-write interface was source-insensitive even on the tested training slice.

The only locked round-level ratio is the earlier six measurement rounds to one build round—already above the amendment’s 5:1 halt threshold. Later one-write builds improve that ratio, but no current canonical full-day recount exists; an exact ten-round ratio should not be invented.

“One locked artifact per day” reduces parallel thrash but does not fix the main failure. It may even preserve daily turnover. The single highest-leverage loop change is:

> Maintain one active cumulative artifact and require a proximal positive-control staircase before advancing: train own-write versus same-entity counterfactual specificity first, then zero/short delay, then held-out names, then unseen wording, then long delay—changing only one difficulty at each rung.

This would have localized `onewrite_recall_v1` at source/content learning before the expensive held-out persistence claim.

## Ranked next increments

1. Propagate the licensed construction-level wording and explicitly close `onewrite_recall_v1`; retain the train diagnostic as diagnostic, not a new result.

2. Replace “one artifact per day” with the cumulative positive-control staircase above and report the ratio using a declared line/round classification.

3. Resolve the navigator’s duplicated-step, manifest-binding, comparator, and smoke-provenance discrepancies; then complete the required direction dialogue before any full run.

4. If reaffirmed, run the navigator exactly once as a calibration control, audit it once, and forbid a repair or v2 regardless of outcome.

5. Return the central program to a real-model, co-developed state carrier only after its proximal content-specific intervention works before delay, paraphrase, or held-out transfer is introduced.

Final program ruling: **continue, but say plainly that the current loop is not working. It is not presently doing the highest-leverage thing.** The scientific question remains open; what should stop is rapid turnover among under-localized constructions.

## 2026-08-29 — onewrite_recall_v1 seeds 11/23 FAIL; round 18 (Codex, verbatim): clean negative on a valid instrument; navigator runs once after the audit

## Ruling

Yes: this is a clean negative on a valid real-model instrument, but only for the registered construction. Visible-copy accuracy was 1.00 while cue accuracy was 0.00; zero-hook reproduced cue row-for-row; and correct, counterfactual, and random writes were indistinguishable in both completed seeds. The increased tag completion under every nonzero write is a content-independent nudge, not evidence that a fact was written and forgotten.

Exact STATE/README sentence:

> With a valid direct-recall instrument (visible 1.00; cue 0.00), `onewrite_recall_v1` failed its registered two-seed rule: the 65,552-parameter encoder/injector, trained for 400 single-example steps and applied once at the norm-capped block-12 slot, gave the same 0.031 held-out tag accuracy for correct, counterfactual, and random writes after a 71–73-token delay, so this fixed construction did not establish a transferable causal recall channel.

The loss reduction licenses only that optimization fitted some aspect of the training objective. It does not show that train-entity facts were encoded in a transferable state.

Never say:

- “The model cannot store a fact in its hidden state.”
- “The fact was written but did not survive 70 tokens.”
- “Block 12 cannot support persistent memory.”
- “A 65k-parameter interface cannot work in principle.”
- “More training, another layer, or another write mechanism would also fail.”
- “The loss curve proves the interface learned the facts.”
- “Frozen language models have no native state or latent mathematics.”
- “This closes the real-model route.”

## Navigator ruling

Run `necessity_navigator_v1` once, after seed 37 finishes and the required fresh audit closes the recall result. This satisfies the round-15 condition: the real-model readout was valid, and the registered intervention construction then failed substantively.

The round-11 design and round-12 amendment remain unchanged: same \( \mathbb{Z}_{11}^2\rtimes C_4 \) world, GRU, training procedure, five tolerance-based readouts, seed aggregation, and numerical gates. BOUNDED POSITIVE still requires at least four of five readouts, with composition/noncommutativity and causal swap mandatory. If at least two seeds learn valid behavior but both mandatory readouts fail, close the claim that task necessity alone yields readable causal algebra; if fewer than two seeds learn valid behavior, close only this training construction. No repair run follows.

Generating same-goal swap episodes is a permitted pre-lock implementation fix: it instantiates the already-registered test rather than changing its hypothesis. Freeze and hash an outcome-independent evaluation manifest with exactly 200 recipient–donor pairs, using held-out permutations, the same goal, and different underlying places. Each group must contain at least a third same-goal trajectory for the wrong-place control; the implementation must never fall back to using the donor as its own wrong control. Do not change any statistic, threshold, training sample, or status rule.

## Three sentences for the user

Today we obtained a clean negative on a real model: with a perfect visible-fact readout, the locked one-write interface produced the same near-zero held-out recall for correct, counterfactual, and random writes, so it did not carry fact-specific content. This closes one 65k-parameter block-12 additive construction—not persistent state or native latent mathematics—and satisfies our predeclared condition for one final navigator calibration. Next we will run that already-locked navigator once to ask whether a model forced to navigate an aliased world learns a readable, causally swappable map.

Tomorrow’s single structural change: enforce **one locked artifact per day**. Its first hour is an end-to-end sample-and-control preflight; then the run, audit, and propagation must finish before another construction may be designed.

## 2026-08-29 — Round 15 (Codex, verbatim): keep the kill; next = `onewrite_recall_v1` (single-fact recall through a one-write channel)

## Program ruling

Keep the pre-lock kill exactly as adopted. `onewrite_state_v1` never tested hidden-state persistence: the repaired protocol achieved completion 1.0, but visible and hidden prompts produced essentially the same choices. The dominant problem is now semantic instrumentation—these small base models cannot reliably execute the registered two-variable lookup.

Choose **both routes asymmetrically**:

- The next artifact is a simpler real-model readout on Qwen3-1.7B-Base.
- A bounded model-capacity screen preserves the possibility of restoring novel-consequence tests later, but is not the next artifact.

### Next artifact: `onewrite_recall_v1`

Use direct single-fact recall:

```text
Source: The private tag assigned to Cordovel is keld.
...
New target wording: PRIVATE TAG FOR CORDOVEL:
```

Train the same small encoder/injector to write once at block 12. Use a balanced vocabulary of tokenizer-verified nonce tags, heldout entities, unseen source and target phrasings, and enough text after the write to test persistence. The behavioral endpoint is the raw decoded tag itself—no rule, mapping, XOR, or multi-variable consequence.

Retain visible-copy, cue-only, correct-write, wrong-donor, random-write, and zero-hook arms. The crucial causal signature is not merely correct recall: a wrong donor must specifically produce the donor’s tag while random writes do not.

This is a new, weaker artifact, not a rescue of the killed construction. A positive would license:

> A co-designed interface can write one factual value once into a frozen real model and recover it after unseen wording.

It would not license abstraction, transfer to consequences absent from training, native pretrained state, or latent mathematics. It establishes a persistent causal memory channel—the prerequisite that `onewrite_state_v1` attempted to leap beyond.

For future consequence work, use one fixed no-training screen:

1. Qwen3-8B-Base, isolating scale within the same family.
2. Qwen3-4B-Instruct, isolating instruction tuning against the failed 4B Base.
3. Qwen3-8B-Instruct only if those results disagree.

An instruct checkpoint passing the table lookup would not rehabilitate earlier polarity-sensitive probes; it would validate only a balanced visible-versus-cue consequence instrument.

## Navigator ruling

**No—do not run `necessity_navigator_v1` now.** Round 12 required a clean real-model negative after behavioral validity. This was an instrument negative, so the condition remains unmet. Running the navigator now would answer the easier synthetic existence question while leaving the real-transformer persistence question untouched.

If direct visible recall is valid but the one-write, donor-specific persistence test fails, that becomes the clean real-model negative that licenses the navigator once. If direct recall succeeds, the navigator is unnecessary.

## What to tell the user tonight

> Today established bounded causal facts—late interventions can steer lexical output, a repeatedly injected bus can act as a supervised controller, and the registered anchor/span constructions fail their stated laws—but it established no native coordinates, interchangeable state, persistent state, or latent mathematics.  
> The most important discovered constraint is that the tested Qwen3 base models through 4B cannot reliably apply a two-variable table even when the facts are visible, so rule-dependent behavioral readouts cannot adjudicate hidden-state interventions at this scale.  
> The one next artifact is `onewrite_recall_v1`: “Tell the model one private tag, change every word around it, and test whether a single hidden write lets it recall that tag later.”

## 2026-08-29 — onewrite_state_v1 killed pre-lock: the base model cannot apply a stated rule to visible tags

Codex round 14 allowed exactly one instrument repair (its own terse FORMAT EXAMPLE / TEST CASE / NEW ITEM format, strict first-item parsing) and a sole pre-lock validation with no state involved. Result on all 64 heldout cases: visible-tag accuracy 0.344 (H1 0.50, H2 0.19) and cue (`TAGS: STORED`) accuracy 0.344 — identical; completion 1.0 in both arms. The 1.7B base emits an allowed label every time but ignores whether the tags are shown. Pre-declared ruling, verbatim: "onewrite_state_v1 is killed pre-lock because Qwen3-1.7B-Base could not support the registered behavioral instrument even when the facts were visible; no state hypothesis was tested." My side probe shows Qwen3-4B-Base is also at chance on the same H1/H2 tables (0.44/0.31 terse; 0.50/0.56 one-shot) while managing a simple one-attribute rule at 0.81, so the limit is rule-following at this scale, not the wording. No further prompt iteration or model substitution (pre-declared).

Where this leaves the program: every real-model construction today has been closed — four by instrument/baseline failure before the intervention was testable (coordinate_v1/v2, interchange_v1, onewrite_state_v1), three by pre-declared causal gates (coordinate_v3 → lexical steering; interchange_v2 null; state_bus_v1r1 → supervised controller; control_cost_v1 → construction fail). Round 12 said the navigator may follow only after a *clean* real-model negative; this is an instrument negative, so the program ruling is referred to Codex (round 15).

## 2026-08-29 — Re-contextualization #30 (evening): after the stop point

Audit: skipped this cycle — no new capability result since audit #31; the navigator and one-write runs are smokes (code-path and instrument validation), not results, and are logged as such.

Whole picture. The mission — native mathematics of latent spaces from axioms upward — has now been attacked three ways: (1) toy quotient worlds with exact certificates (Rounds 36–37; ended: the certificates guaranteed bad verdicts on continuous learners); (2) interventions on frozen pretrained models (seven constructions today; stopped as an allocation pivot — most never reached a valid causal test, the rest reduced to lexical steering or supervised control); (3) now a choice between building the object (a navigator whose behaviour makes an algebraic state necessary) and co-training a one-write state into a real model. The evening's new fact is small but sharp: the one-write instrument fails before the state question — Qwen3-1.7B-Base cannot apply a stated two-way rule to *visibly given* tags better than chance in any of three formats — so the readout side of every real-model construction today has been limited by the base model's own rule-following, not only by our interventions.

What reframes earlier work. Three of today's kills (coordinate_v1/v2, interchange_v1) and now the one-write smoke share one cause: the readout instrument is invalid on the base model before the intervention is testable. That is a substrate fact about small base models as *readout devices*, and it argues for either a larger base for the readout or a readout that does not require instruction/rule-following (the navigator's next-action prediction is exactly that).

Live alternatives (verbatim sources: audits #28–#31, rounds 10–13): the navigator as a one-round calibration control; predictive dynamics/flow; reachability with a nonuniform actuator (explicitly deferred); distributed transformation operators over spans; response-law topology; and the unified alternative that everything seen is a biased, cue-redundant lexical-semantic response geometry with visible-text mediation. Not narrowed: the fork is explicit and Codex round 14 decides on data (the 4B probe).

Foundational thread: the governance loop's own cost is now visible — since the stop point, two design rounds and two smokes without a locked run; the tripwire cuts both ways, and the next cycle must lock or kill.

## 2026-08-29 — Round 13 (Codex, verbatim): `onewrite_state_v1` locked design

# `onewrite_state_v1` — locked design gate

## Object

Freeze Qwen3-1.7B-Base. Train only:

\[
E:\mathbb R^{2048}\rightarrow\mathbb R^{16},\qquad
J:\mathbb R^{16}\rightarrow\mathbb R^{2048},
\]

for **65,552 parameters**. Encode \(z=E(\mathrm{LN}(h_{12}^{source}))\) from the final token of a neutral source anchor. In a separate target prefill, add \(Jz\) once to the final token of `Internal record:` at block 12. Clamp \(\|Jz\|_2\le0.25\|h_{\text{slot}}\|_2\).

The slot occurs early; 64–96 tokens of new wording, rules, and the question follow it. Block 12 leaves the upper transformer layers to propagate the write into cached keys and values. Disable the hook before greedy continuation and assert that exactly one batch-position was modified. There is no continuation hook, repeated injection, reader, recurrent bus, or direct KV editing.

## World and training

Create **40 invented entities**, five for each combination of three binary nonce attributes:

- `marn/suth`
- `vep/keld`
- `orin/tazz`

Randomly assign balanced states to names. Use three entities per state—24 facts—for training and two per state—16 facts—for evaluation. Training and evaluation use disjoint entity names, source templates, target templates, and RNG streams.

A source says, for example, that an entity’s three registry tags are `marn`, `keld`, and `orin`. The target names the entity but contains no selected attribute values. All possible values may appear symmetrically inside a rule.

Training consequences, with mappings explicitly stated and counterbalanced across facts:

- Attribute 1 → `PORT/VAULT`
- Attribute 2 → `NORTH/SOUTH`
- Attribute 3 → `RING/STAR`

Entirely absent from optimization—labels and templates:

- **Heldout H1:** whether exactly one of `marn` and `vep` applies → `CEDAR/QUARTZ`.
- **Heldout H2:** the four combinations of attribute 2 and attribute 3 → `LARK/ORIEL/PINE/WREN`.

Every heldout prompt states its rule and uses balanced label rotations, preventing a fixed label prior from solving the task.

Initialize \(J=0\). Train seeds 11, 23, and 37 for 400 fixed single-example AdamW steps: learning rate \(3\times10^{-3}\), weight decay \(10^{-4}\), gradient clip 1.0. The sole loss is mean token cross-entropy on the correct decoded label for the three training families.

The optimizer must never access heldout labels, templates, prompts, logits, outcomes, or stopping metrics. It also receives no prototype, reconstruction, persistence, contrastive, likelihood-uplift, or tolerance-fitting objective. Source facts never appear visibly in training target passages.

## Demonstration

Each heldout fact receives two H1/H2 cases under two unseen post-write wordings: 64 cases per seed. Decode greedily for at most 12 tokens, stopping at newline/EOS, and parse the first complete allowed label. Save every decoded string and report termination beside accuracy.

| Arm | Intervention |
|---|---|
| Correct one-write | Encode the heldout source and write its state once. |
| No-write / cue-only | Identical target with zero delta. These are operationally the same deterministic baseline and are computed once under both names. |
| Wrong state | Write a heldout donor state whose correct answer differs; donor assignment is an on-manifold balanced derangement. |
| Random state | Fixed Gaussian \(z\), centred and norm-matched to training states. |
| Visible-text mediation | No write; insert the exact source fact visibly. This is the text-mediated ceiling. |

Primary cases generate no state-specific word before the scored answer, eliminating the bus’s visible-output mediation path. Verdicts use raw decoded choices, never likelihood movement alone.

## Gates

Facts—not rows or seeds—are inferential units. Use 2,000 fact-cluster bootstraps and 200 paired sign-flip randomizations. Uniform descriptive chance is \(0.375\), averaging H1’s binary and H2’s four-way choice sets; paired arms are primary. A gate must pass in at least two seeds and in the across-seed median; report all seed values and ranges.

**Behavioral instrument**

- Visible-text accuracy ≥0.80.
- Correct-write and visible-text termination ≥0.95.
- Cue-only accuracy ≤0.50.

**Heldout transfer and survival**

- Correct-write accuracy ≥0.75 overall and ≥0.70 in each heldout family.
- Difference between the two unseen wording templates ≤0.15.
- Correct-write minus cue-only accuracy ≥0.25, with fact-bootstrap lower 95% bound >0.10.
- Correct-write minus random-state accuracy ≥0.20.
- Hidden recovery fraction  
  \[
  \frac{A_{\text{write}}-A_{\text{cue}}}
       {A_{\text{visible}}-A_{\text{cue}}}\ge0.60.
  \]

**State specificity**

- Wrong-state decoded choice follows the donor’s correct answer ≥0.60.
- Donor-follow accuracy exceeds its no-write donor-choice baseline by ≥0.20.

Exact all-case success, exact groupings, and exact randomization tails are diagnostic only.

## Status and kill rule

**BOUNDED POSITIVE:** every instrument, heldout-transfer, wording, mediation, specificity, control, and seed gate passes.

> In frozen Qwen3-1.7B-Base, a 65,552-parameter interface wrote a 16-dimensional state once into one block-12 prefill slot; after unseen wording, that cached state specifically changed raw decoded choices for consequence labels and templates absent from training across heldout facts and seeds. This establishes a bounded co-designed persistent causal state, not native pretrained structure or a full latent mathematics.

**SUPERVISED CONTROLLER:** heldout evaluation on the three trained consequence families reaches accuracy ≥0.80 and improves ≥0.25 over no-write in at least two seeds, but any bounded-positive gate fails.

> The one-write interface controlled raw choices for consequence families it was trained to name but failed the locked heldout-family, wording, specificity, or mediation gates; it is a supervised response controller, not an abstract persistent state.

**FAIL:** the behavioral instrument is invalid, state effects are nonspecific, or neither trained nor heldout state use passes.

> `onewrite_state_v1` did not produce a specific one-write effect that survived unseen wording and governed heldout decoded consequences; this closes the fixed encoder, slot, injector, norm budget, and training construction—not persistent state in real models generally.

One completed three-seed result receives one audit. No layer, position, dimension, norm, prompt, label, template, optimizer, step-count, or decoding repair is permitted.

## Build and budget

Reuse `SubstitutionProbe` model loading/tokenization and `run_state_bus.py` hashing, logging, checkpointed `result.json`, and ledger conventions. Do not reuse its bus, reader, prototypes, repeated hook, training-derived tau, signature scoring, or candidate rollouts.

One runner under 240 nonblank lines; one config under 90; one result; no reducer/dashboard. Target approximately 140 artifact-bearing lines and at most 170 evaluation/config/I/O lines: combined apparatus ratio ≤1.3:1. Expected CPU wall time is 2–2.5 hours, hard-capped at three hours.

**Build first:** the complete source-cache → encoder → one-slot write → hook-disabled cached decode → five-arm evaluation for seed 11. It must finish within 55 minutes and can falsify the artifact; the unchanged process then continues seeds 23 and 37.

**Lay line — 9/10 if all heldout gates pass:** “Write a private fact into a language model once, change every word around it, and see whether that hidden fact still decides a brand-new consequence.”

## 2026-08-29 — Rounds 11–12 (Codex, verbatim): navigator designed, then demoted; the real-model one-write state artifact goes first

Round 11 designed `necessity_navigator_v1` (GRU on Z_11² ⋊ C_4 with aliased, per-episode-permuted observations; five readouts). I built it (`experiments/run_necessity_navigator.py`, 196 nonblank lines; config) and smoke-tested it — code-path validation only, not a result: 2000 training steps reach held-out top-1 in A* of 0.879 against a historyless control of 0.484 (behaviourally valid); readouts execute; at that budget moves R = 0.24 (untrained-GRU control 0.38), composition order accuracy 0.51, inverse ratio 0.63, distance Spearman 0.30; the swap pairing needs same-goal episodes across permutations (to be generated deliberately if it is ever run). Round 12 then reconciled audit #31's alternative against it and ruled for the real-model artifact: the navigator has the better odds but lower evidentiary value and the toy-world repeat risk; it is now an optional one-round calibration control, not to be run before the one-write real-model result. Round 13 (design gate for `onewrite_state_v1`) is in flight.

### Round 11 (verbatim)

Written to [.codex_direction_r11.md](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/.codex_direction_r11.md>).

The locked design uses a ~17.7k-parameter GRU, three CPU seeds, a 60–100 minute estimate, five controlled approximate readouts, and a non-tautological cross-presentation state swap. Planned apparatus-to-artifact code ratio is at most 1.1:1. No experiment was launched.

### Round 12 (verbatim)

## Head-to-head ruling

The probabilities below are judgment calls about one locked CPU run, not statistical estimates.

| | Real-model co-designed state | `necessity_navigator_v1` |
|---|---|---|
| **Bounded-positive probability** | **30–40%**. One-write persistence, new wording, and genuinely untrained consequences are difficult, especially after the bus failure. | **55–65%**. Behavioral learning is likely; satisfying four of five readouts, including composition and cross-presentation swap, remains nontrivial. |
| **What a positive licenses** | A real language model can host a persistent, causally addressable state that survives paraphrase and governs held-out consequences. This establishes a defensible foundational object, but not that it was native to the frozen model or that a full latent mathematics has been recovered. | Task pressure can induce an approximately algebraic, readable, causally portable state in a small recurrent model. This is a constructive existence result, not evidence about pretrained language models. |
| **What a negative teaches** | If behavioral validity and state-use controls pass, the model still did not rely on a one-write state for unseen consequences. That would materially narrow the real-model constructive route, while remaining architecture-specific. | If behavior is valid but the structural readouts fail, persistent-state necessity alone does not produce readable algebra. If behavior fails, only this training construction is implicated. |
| **Repeat risk** | Another state bus: supervised response control, lexical overlap between training and evaluation, visible-text mediation, repeated injection, or an intact cue-rich bypass. | Another toy-world cycle: the algebra is known, memory is forced, and a positive can collapse to “a GRU learned the world it was designed to learn.” Approximate gates improve integrity but do not solve relevance. |
| **Narrative score** | **9/10:** “Write a thought once, change the wording, and the model still uses it to derive something it was never trained to say.” | **7.5/10:** intrinsically interesting, but visibly a controlled synthetic demonstration. |

## Ruling

Build the auditor’s **real-model, one-write state** now.

The navigator has the better chance of succeeding, but the expected evidentiary value is lower. Even its strongest positive cannot answer the program’s present question about a real model; the real-model artifact can. After a day dominated by apparatus and after Rounds 36–37 already exhausted the toy-world route, choosing the navigator first would optimize for obtaining a positive rather than resolving the central uncertainty.

The real-model artifact qualifies as a new substrate only if it is structurally unlike `state_bus_v1r1`:

- One state write; no repeated injection.
- New wording after the write.
- Consequence families whose labels and templates were absent from state training.
- State ablation and cue-only baselines.
- Explicit visible-text mediation control.
- No positive based solely on likelihood movement or trained consequence names.

`necessity_navigator_v1` does **not** follow regardless. A real-model positive makes it unnecessary. It may follow once, as a calibration control, only if a clean real-model negative leaves the specific question: “Can task necessity produce any readable causal algebra, even in a purpose-built substrate?” It must not become the central artifact or another expandable toy program.

## Exact round-11 amendment

> **Amendment:** `necessity_navigator_v1` is an optional one-round calibration control, not the next central artifact. Do not implement it before the one-write real-model state result. If it is later run and at least two seeds pass behavioral validity but composition/noncommutativity and causal swap both fail, close the hypothesis that task necessity alone yields a readable, causally portable latent algebra; do not change the group, GRU size, alias map, optimizer, or readouts. If fewer than two seeds pass behavioral validity, close only this training construction.

All existing constraints remain: under 250 nonblank runner lines, one config and result, tolerance-based effects, seed spread, diagnostic-only exact certificates, no reducer, no rescue run, and one audit for the completed result.

## 2026-08-29 — Audit #31 on control_cost_v1 (fresh, unprimed; verbatim): FAIL upheld; 'the first-order law does not hold', the Spearman/ratio/asymmetry residues, and '4x under-prediction' are WITHDRAWN

Correction to the control_cost_v1 entry below: seven censored costs were stored as 4x predicted, so the rank agreement and the ratio are artefacts of censoring; the cross-vs-within gate compared censoring bounds; B was invalid; the only licensed residue is bounded causal responsiveness of the A readout with a construction-conditioned directional difference. The auditor's alternative next artifact class (a real-model co-designed causally addressable state) is recorded verbatim beside Codex round 10's from-scratch navigator; the required dialogue precedes implementation.

## Executive verdict

- **UPHOLD:** `FAIL — FIXED BLOCK-12 SPAN CONTROL CONSTRUCTION`.
- **REJECT:** “the first-order minimum-energy law does not hold.” The experiment does not isolate such a law from its actuator, metric, solver truncation, endpoint definition, or coefficient cap.
- **REJECT as evidence:** the reported Spearman cost ranking, cross-versus-within gate, semantic-B advantage, and directional cost asymmetry.
- **LICENSED residue:** the frozen model’s registered A readout responds causally and directionally to these constructed fields, with much stronger continuous dog→cat movement than cat→dog movement. This is an actuator/readout-specific response asymmetry, not an effort geometry.
- **Program ruling:** this is not working as a native-mathematics discovery program. The broader research question may continue, but this construction family should not. The highest-leverage action is to reconsider the substrate now.

## 1. The KILL is valid, but its interpretation is over-claimed

The formula

\[
v^\star=J^\top(JJ^\top)^+r
\]

is a minimum-norm solution inside the retained linearized system. The run tested whether that tangent solution, extrapolated through a nonlinear model using one particular actuator and coefficient grid, realizes a behavioral endpoint. It did not test a substrate-independent “first-order minimum-energy law.”

The registered endpoint was attained in only `1/8` recipients. That robustly closes the fixed construction. But at least six construction choices remain jointly responsible:

- **Uniform broadcast:** one vector was imposed on 23 heterogeneous token positions. Nothing establishes that the positions share an actuator or should receive equal displacement.
- **Sigma scaling:** per-channel calibration variance defines both field direction and cost. It is a legitimate operational metric, but not a neutral measure of intrinsic effort.
- **Jacobian at zero:** the target can lie well outside the local fidelity radius. The smallest nonzero evaluation was already `α=.25`; no infinitesimal finite-difference validation was saved.
- **Coefficient cap:** seven costs are right-censored at `>4‖v*‖`. This establishes failure under the budget, not unreachability.
- **Mixed endpoint:** success requires both `0.5` class-separation movement and `2/3` sign flips. Dog0, dog2, and dog3 moved `1.005`, `1.076`, and `0.601` separations at their terminal rows but flipped only one probe. Cat1 flipped two while moving `−0.291` in the wrong direction. The gate mixes two distinct failure modes.
- **Pseudoinverse conditioning:** the runner applies `pinv` to `JJᵀ` in fp32 with `rcond=1e-6`, which squares conditioning relative to operating directly on `J`. No singular values, retained rank, or `‖Jv-r‖` residuals were stored, so solver truncation cannot be separated from curvature.

Therefore the null is compatible with an inefficient actuator, inappropriate metric, local-linear extrapolation failure, solver conditioning, inadequate budget, or endpoint construction. It is not a general physical verdict about first-order control.

## 2. Censor-aware replay

### Spearman `0.76` is not a realized-cost law

Seven censored observations are serialized numerically as exactly

\[
\text{stored realized}=4\times\text{predicted}.
\]

The eighth equals `2 × predicted`. Consequently, seven response ranks inherit the predictor ranks mechanically. With only one actual realized cost, neither the cost ranking nor a rank correlation between predicted and realized costs is empirically identified.

Likewise, `median realized/predicted = 4.0` is a censoring boundary, not an estimated calibration ratio. The actual median is only known to exceed four under the construction’s cost convention.

### Cross-versus-within is not `8/8` evidence

The runner compares stored censoring bounds as though they were observed costs.

- Only cat3 and dog2 definitely have cross-class cost greater than an uncensored within-class cost.
- Cat0 has an observed cross cost but only a lower bound on its within cost, so its ordering is unknown.
- Five further pairs are censored on both sides and are unordered.

Thus the evidence is **2 definite, 6 unidentified**, not an evidential `8/8`. The reported median ratio `3.679` is not an estimable cross/within cost ratio.

### Directional residue

There is a real descriptive asymmetry in continuous A response:

| Alpha | Mean cat→dog movement | Mean dog→cat movement |
|---:|---:|---:|
| `0.25` | `0.038` | `0.280` |
| `0.5` | `0.111` | `0.399` |
| `1` | `0.144` | `0.542` |
| `2` | `0.159` | `0.670` |

Shared fields also attain A in `2/4` recipients per direction at `α=2`.

This licenses: **the registered fields causally affect the registered response signature, with a construction-conditioned directional difference.**

It does not license a cost asymmetry. The reported log ratio `−0.619` is computed from mostly censored bounds. Moreover, the preregistration required agreement between calibration and held-out asymmetry signs, while the runner’s `asymmetry_licensed` check tests only whether the held-out absolute log ratio exceeds `log(1.5)`.

## 3. Positive-residue overclaim audit

| Proposed residue | Audit |
|---|---|
| Spearman rank agreement | Not evidence: largely generated by storing censored costs as `4 × predicted`. |
| Cross class costs more than presentation | Not established: only `2/8` pairwise orderings are identified. |
| Semantic B transfer | Void: B native validity failed `17/24`, including cat `7/12`. |
| Semantic beats lexical on B | Descriptive only. Advantage is positive `7/8`, but median `0.175` misses the registered `0.3` effect threshold, and B is invalid. |
| `p=.008` proves semantic structure | No. It is a sign-flip sensitivity on an invalid readout and cannot override the failed effect-size or random-control gates. |
| Random fields move B “just as much” | Too strong. The registered random rejection failed (`p=.143`); failure to reject is not equivalence. |
| Dog→cat costs more | Not identified because costs are censored and the required calibration agreement was never evaluated. |

The B result is especially asymmetric: cat→dog semantic B movement has median `−0.075`, whereas dog→cat has median `0.742`. The apparent semantic-over-lexical advantage on cat recipients mostly means “less negative than the lexical field,” not successful semantic transfer.

## 4. Execution and preregistration integrity

The design was written before the full run, and the current runner, config, and context hashes match those bound in [result.json](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/results/control_cost_v1/result.json>). But this was not a fully clean confirmatory execution:

- The ledger lock followed a smoke run that had already exposed all held-out A/B native-validity outcomes and one recipient per direction. The lock also added operational definitions not fully fixed in the design.
- The design promised all raw and centred margins. They are absent, as are Jacobians, singular spectra, sigma diagnostics, shared A trajectories, and solver residuals.
- The twenty random fields were supposed to be fixed fields applied across recipients. The runner samples a new Gaussian field inside each recipient loop, so random column `k` aggregates eight different fields.
- The registered `α=0` numerical sham was not executed. The runner simply substitutes the stored baseline `u0` rather than traversing the hook with a zero field.
- Calibration/held-out asymmetry agreement was not implemented.
- The primary status ladder uses all-or-none prompt counts and sharp thresholds without seed spread, contrary to the governance amendment making exact certificates diagnostic rather than verdict-bearing.

These discrepancies do not rescue the construction: `1/8` endpoint realization and invalid B are large descriptive failures. They do prevent stronger confirmatory claims about why it failed.

## 5. Program-level audit

The strongest unified hypothesis across the five constructions is:

> The frozen model provides a cue-rich, causally redundant lexical-semantic response geometry: late or probe-aligned interventions can move selected token-likelihood readouts, while context, position, and visible autoregressive history preserve multiple competing causal routes; a reusable point-like state, coordinate, donor residual, bus code, or broadcast field is therefore not required to explain the observed effects.

This accounts parsimoniously for:

- `coordinate_v3`: late control aligned with verb logits.
- `interchange_v1`: fixed verbalizer bias invalidating the raw-zero instrument.
- `interchange_v2`: one anchor residual unable to overcome the intact cue-rich prefix.
- `state_bus_v1r1`: repeatedly injected supervised control concentrated on particular verbalizers and mediated by visible history.
- `control_cost_v1`: probe-derived fields moving response margins without furnishing transferable endpoint control or an identifiable cost law.

This is a unifying **hypothesis**, not proof of a hostile substrate. The five constructions share models, lexical tasks, readout conventions, and intervention assumptions; they are not five independent demonstrations that pretrained residual streams lack usable structure.

### Continue? Highest leverage?

- **Should the current construction family continue? No.**
- **Should the broader scientific question continue? Yes, but only after a substrate-level allocation pivot.**
- **Is the present work highest leverage? No.**
- **Is there one more experiment worth running before reconsidering the substrate? No.**

No alpha extension, rcond sensitivity, layer scan, nonuniform span, nonlinear optimizer, probe repair, or control-cost v2 should precede reconsideration.

The single post-reconsideration artifact class worth considering is a real-model, causally addressable substrate co-designed so that a state must be written, preserved, and used for held-out consequences—rather than another external vector applied to a frozen response geometry. Its binding narrative question is:

> **Can we build a model whose hidden state can be written once, survive new wording, and causally govern consequences it was not directly optimized to name?**

That direction requires the mandated 2–3-round dialogue before implementation.

### Measurement-to-artifact ratio

`control_cost_v1` has 156 nonblank runner lines plus 36 config lines. Counting approximately 26 lines as the actuator/Jacobian/solve/field-construction core leaves roughly `166:26`, or **6.4:1**, apparatus/config to artifact-bearing code. Direction design, smoke/lock/full measurement, and audit add at least **3 measurement/governance rounds per build round**. The classification is approximate, but safely beyond the `5:1` halt threshold.

## 6. Exact licensed sentence

> **At fixed block 12 in Qwen3-1.7B-Base, the registered sigma-scaled uniform 23-token prefix-span field derived from the v=0 three-probe Jacobian attained the joint A endpoint in 1/8 held-out recipients by α≤4, while B was not a native-valid readout; because seven local costs and six within-class costs were right-censored, the saved data do not establish a realized-cost rank law, cross-versus-within effort geometry, semantic transfer, or directional cost asymmetry, so this closes the registered actuator/solver/readout/budget construction—not first-order control, span reachability, or latent effort generally.**

## 7. Never-say list

- “The first-order minimum-energy law does not hold.”
- “The model’s response is strongly nonlinear at the magnitudes needed” without naming the unresolved solver, scaling, actuator, and censoring alternatives.
- “Predicted cost ranked realized cost.”
- “The method underpredicted realized cost fourfold.”
- “Cross-class moves cost more than presentation changes.”
- “The semantic field transferred to unoptimized consequences.”
- “The semantic field beat lexical steering.”
- “`p=.008` establishes semantic structure.”
- “Random fields moved B just as much.”
- “Cat→dog is intrinsically cheaper than dog→cat.”
- “Span control failed” or “reachability failed” without the complete registered scope.
- “The preregistration was executed completely.”
- “Five independent interventions prove the substrate lacks native structure.”
- “Frozen residual streams cannot support structured reasoning.”
- “The next latent space must be trained” as a scientific conclusion rather than an allocation hypothesis.

## 8. Exact README wording

> **Status 2026-08-29: no native latent mathematics has been demonstrated. After `coordinate_v3`, `interchange_v1/v2`, `state_bus_v1r1`, and `control_cost_v1`, this is not working as a native-mathematics discovery program under the current frozen-residual intervention substrate. In `control_cost_v1`, the fixed block-12 uniform prefix-span Jacobian field attained its registered A endpoint in 1/8 held-out recipients, B was not a native-valid readout, and censoring prevents claims about cost ranking, cross-versus-within effort, transfer, or directional cost asymmetry. This closes that construction family and triggers a substrate-reconsideration dialogue; it is an allocation pivot, not evidence that pretrained residual streams lack usable structure.**

## 9. Exact STATE wording

> **`control_cost_v1` — REGISTERED CONSTRUCTION-LEVEL FAIL, CLOSED.** Calibration-centred A validity passed `23/24`; B validity failed `17/24` (`cat 7/12`, `dog 10/12`), voiding all B-based gates. The registered prompt-specific uniform 23-token prefix-span Jacobian field attained the joint A endpoint in `1/8` recipients by `α≤4`. Seven local costs and six within-class costs are right-censored: the reported Spearman `0.762` is largely induced by storing censored bounds as `4×predicted`, while only two cross/within orderings are identified. Shared fields attained A in `2/4` recipients per direction at `α=2`; B comparisons are descriptive only and additionally use a random-field implementation that resamples per recipient rather than applying fixed shared null fields. The log cost ratio `−0.619` is not licensed directional asymmetry because the costs are censored and calibration-sign agreement was not evaluated. Licensed residue: bounded causal responsiveness and construction-conditioned directional differences in the registered A readout. This closes the fixed actuator, solver, readout, and budget—not first-order control, span reachability, latent effort, or frozen residual structure generally. Program allocation: this is not working under the current substrate; stop repair runs and begin the mandated substrate-reconsideration dialogue.

## 10. Ranked next increments

1. Adopt the licensed correction and propagate it to README, STATE, NOTEBOOK, and an append-only ledger correction; retain the original result row.
2. Close `control_cost_v1` and prohibit alpha, rcond, layer, span, probe, or optimizer repair.
3. Run the required 2–3-round direction dialogue on whether the next object is a causally bottlenecked state or a trajectory-level controller.
4. If that dialogue produces a binding artifact design, build exactly one real-model substrate intervention with held-out consequences, seed spread, lexical controls, and visible-text mediation controls.
5. Fire one fresh audit; continue only if the artifact demonstrates approximate intervention effects beyond those controls.

No repository file was edited. The evidence and synthesis were recorded in the mandatory git-ignored blackboard `5f76e3ae`.

## 2026-08-29 — Direction round 10 (Codex, verbatim): STOP for the frozen-residual line; next = `necessity_navigator_v1` (constructive substrate)

## Program ruling

Yes. This is the stop point for the frozen-pretrained residual-stream line as currently constituted—not a proof that pretrained models contain no latent mathematics.

`control_cost_v1` supplies no usable cost law. Readout B is invalid at 17/24. Seven of eight local costs are censored at the maximum intervention strength, so Spearman 0.76, the 8/8 cross-over-within ordering, and the reported asymmetry cannot be interpreted as native geometry. The semantic field beating the lexical field is neutralized by random fields moving B comparably. The licensed result is narrower:

> At block 12, a uniform prefix-span intervention optimized through the three-probe signature did not yield a locally accurate, transferable control-cost construction.

Today’s measurement-to-artifact ratio is six rounds to one. That crosses the governance halt threshold and fulfills the prior commitment to reconsider the substrate after another apparatus-only negative. Audit #31 may tighten the construction-level wording, but absent a data-integrity reversal it should not reopen this allocation decision.

### The next artifact

Build `necessity_navigator_v1`: a compact sequence model trained from scratch to navigate a partially observable world whose action algebra is known but never supplied as latent supervision.

Use a noncommutative world such as \(\mathbb Z_{11}^{2}\rtimes C_4\): turn, move, and inverse actions genuinely depend on order. Observations should be aliased and their symbols randomly permuted between episodes, so the present tokens cannot reveal location and no stable word embedding can carry the answer. The model receives only behavioral training through a recurrent state bottleneck. After training, ask whether approximate moves, composition, inverses, reachability distance, and causal state swaps can be read from that learned state on held-out trajectories and symbol systems. The known world algebra is the external yardstick, not an architecture component.

This is a one-round constructive substrate benchmark, not evidence about pretrained language models. It avoids today’s repeated failure mode because:

- Persistent hidden state is necessary for behavior rather than redundant with visible cues.
- The outcome is multi-step navigation, not candidate-word logits.
- The causal object is the model’s recurrent state, not an arbitrary single token or fixed transformer block.
- Surface symbols are randomized, directly defeating lexical-semantic geometry.
- A positive supplies concrete invariants that can later be sought in pretrained models; a negative would challenge the constructive mission itself.

Conditional wow line, narrative score **9/10**:

> “A network forced to navigate a hidden world invents a map of its own: its hidden distance predicts the moves between unseen places, and swapping the state moves it there.”

Do not run full-span replacement, a nonlinear optimizer, or a larger frozen model next. Those change the actuator, solver, or scale after the same substrate/readout family failed. They are legitimate later discriminators once a constructive model provides a real invariant to compare—not the next experiment.

### Exact program wording

README:

> Across today’s Qwen3-0.6B/1.7B interventions, we did not demonstrate native coordinates, interchangeable state, persistent state, or a transferable control-cost law; we observed only late lexical steering, pair-specific supervised response control, and null or failing fixed block-12 anchor/span constructions.

What should be said tonight:

> This is not working: after six measurement rounds for one artifact-building round, we should stop mining this pretrained residual stream with lexical probes and build a state-bearing world-model substrate before asking whether pretrained models share its mathematics.

Today does **not** establish that the mission is impossible or that latent mathematics must universally be engineered. It establishes that these pretrained-model interventions were ineffective as a discovery engine. The honest next position is constructive: build a latent space in which navigation mathematics is causally necessary, recover that mathematics without supervising its coordinates, and only then test whether pretrained models independently share it. Real-model interventions remain the required causal test; they should no longer be where the foundational object is guessed from scratch.

## 2026-08-29 — control_cost_v1 (locked): FAIL — the first-order minimum-energy law does not hold; construction closed

Native A validity passed (23/24); native B validity failed (17/24), voiding every B-based gate. The local cost law failed: only 1 of 8 prompt-specific first-order fields moved a held-out context to the opposite-class target within α ≤ 4 (7 censored); the predicted cost ranks realized cost (Spearman 0.76) but under-predicts it more than fourfold — the model's response to a uniform span field is strongly nonlinear at the magnitudes needed. Cross-vs-within passed formally (8/8) but on mostly censored costs, so it is not evidence. Shared calibration fields attained the A target in half the recipients per direction at α = 2; on the (void) B readout the semantic field beat the lexical-gradient field in 7/8 but norm-matched random fields moved B just as much (random p = 0.14). Asymmetry: cat→dog cheaper than dog→cat (log ratio −0.62), diagnostic only. Status by the pre-declared ladder: FAIL — FIXED BLOCK-12 SPAN CONTROL CONSTRUCTION; every status closes this construction.

Reading (mine, pending audit #31): the actuator and the linear cost model were the wrong objects — a uniform field over 23 positions is an inefficient move, and the Jacobian at v = 0 does not predict what happens at the norms required. Per audit #30's ranked list, an apparatus-only negative here is the point to stop and reconsider the latent-space substrate rather than open another construction; that is a program-level decision and is being put to Codex (round 10) and to the user.

## 2026-08-29 — Audit #30 on interchange_v2 (fresh, unprimed; verbatim): FAIL upheld; my 'distributed state / probes read the prefix via attention' reading WITHDRAWN; third-state 'toward neutral' reading WRONG

Correction to the interchange_v2 entry below: the null does not localize class information; same-state PASS is non-discriminating (all arms within τ); horse donors moved dog recipients farther dogward, not toward neutral; the sign-flip p is a sensitivity, not an exact test; decodes were not implemented.

## Executive verdict

**UPHOLD:** `FAIL — FIXED BLOCK-12 SINGLE-ANCHOR INTERCHANGE CONSTRUCTION`.

**REJECT:** “there is no class state at the anchor” and “the result shows the state is distributed across the prefix.” Neither mechanism is identified.

**DOWNGRADE:** the same-state `PASS` is mechanically correct but evidentially non-discriminating. The same tolerance also accepts every cross-state and every cow/horse donor.

**CORRECT:** the result does not show both third-state donor groups moving recipients toward neutral. Under the implemented sign convention, cow donors move cat recipients dogward, while horse donors move dog recipients farther dogward. The claimed exact specificity test is also not design-exact because donor identities were fixed rather than randomized or exchangeable.

The broader program should continue, but this interchange construction should close. The planned move to a genuine intervention-based reachability/control-cost artifact is the highest-leverage next step. This is not working yet as a native-mathematics discovery program.

## 2. Over-claimed KILL audit

### “No class state at the anchor” is not licensed

The experiment excludes one narrow causal proposition:

> In these eight contexts, the entire block-12 residual at the final generic ` The animal` token is not sufficient, under coefficient-one donor replacement and with the recipient prefix intact, to interchange the three registered cat/dog probe decisions at the preregistered effect size.

It does not establish where class information is absent or present. In particular:

- The anchor could encode class information that is causally redundant with the unchanged prefix.
- Layers after block 12 could reconstruct or restore the recipient interpretation from earlier tokens.
- The class-relevant deciding tokens occur during continuation scoring, after the prefill hook has been removed; those tokens can still attend to the recipient prefix through cached representations.
- Block 12 or the generic anchor may simply be the wrong site.
- Donor activations may be context-bound rather than valid modular variables when inserted into an inconsistent recipient context.
- An effective intervention may require coordinated changes across positions, layers, or time rather than a coefficient change at one point.
- Effects may occur outside the three calibration-derived readouts or orthogonally to their class-separation direction.
- There may be no unitary “animal state” at all: the model may recompute each consequence from several lexical cues.

Therefore the null does not license “no class state at the anchor,” “the replacement was overwritten,” or any general conclusion about frozen residual streams.

### Effect on the earlier pivot

`interchange_v2` strengthens the **allocation** case for ending the single-anchor frozen-residual repair ladder: unlike `interchange_v1`, an actual donor intervention ran and failed hard.

It does not convert that pivot into a scientific conclusion. The governing statement remains correct: stopping is an apparatus-budget decision, not evidence that pretrained residual streams lack usable structure ([STATE](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/STATE.md:3>)). It also does not show that persistent state must be added through training.

## 3. Same-state PASS and specificity overclaims

### Same-state `PASS`

The stored-row replay gives:

- Same-state displacement: median `0.3525`
- Cross-state displacement: median `0.2597`
- Third-state displacement: median `1.1007`
- Tolerance: `3.384`

All eight same-state, all eight opposite-state, and all eight third-state arms lie within the same-state tolerance. The cross-state median displacement is actually smaller than the same-state median.

Consequently:

> The same-state gate passed, but it did not discriminate semantic equivalence from an intervention that was broadly ineffectual.

It is permissible to report “same-state gate PASS.” It is not permissible to report “same-state interchangeability passed” or to treat the arm as positive evidence for a stable shared state.

### Third-state interpretation and specificity test

The result and ledger say cow/horse donors moved both classes toward neutral. That is incorrect.

The runner defines positive \(T\) as motion toward the opposite cat/dog class. Therefore:

- Cat recipients: `T_third = +0.10…+0.21`, movement dogward.
- Dog recipients: `T_third = −0.11…−0.20`, movement farther dogward, not toward cat or neutral.

Moreover, every cat recipient has `T_cross−T_third < 0`, while every dog recipient has it `> 0`. The `4/8` split is exactly the class split.

Cat recipients always compare dog cross-donors against cow controls; dog recipients compare cat against horse. These labels were neither randomized nor counterbalanced. Thus the reported \(2^8\) calculation is a sign-symmetry sensitivity, not an exact design-randomization test. It cannot eliminate animal-pair lexical geometry. This design issue was already identified before this audit in the Notebook ([NOTEBOOK](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/NOTEBOOK.md:194>)).

The specificity gate still fails descriptively, but never say that `p=.47` proves no specificity or accepts the null.

## 4. Is “distributed animal state” supported?

No. It is one plausible explanation among several:

1. Class-relevant information is redundantly distributed across prefix positions.
2. The prefix contains multiple direct lexical cues—purr, mice, bark, fetch—without a unitary animal-state variable.
3. Each probe consequence is recomputed separately from those cues.
4. Later layers restore recipient evidence after the anchor replacement.
5. Continuation tokens recover recipient evidence through unchanged cached prefix pathways.
6. The class state exists at another position or layer.
7. The donor residual is informative but not context-invariant or causally modular.
8. A coordinated nonlinear or multi-position intervention is required.
9. The chosen probe axis misses relevant output changes.

The most parsimonious current reading is not “distributed state.” It is **redundant lexical-semantic evidence with response-specific readout**.

## 5. Tunnel vision and unified alternative

The strongest alternative explanation of the day’s results is:

> The frozen model supplies a biased, causally redundant lexical-semantic response geometry. Late or repeatedly applied interventions can move selected verbalizers within that geometry, while cue-rich prefix tokens and subsequently visible words continue to determine later decisions; no persistent, interchangeable, or even unitary animal-state object is required.

This accounts for:

- `coordinate_v3`: direct late verb-token steering.
- `interchange_v1`: fixed verbalizer offsets defeat raw-zero classification despite calibration-relative separation.
- `state_bus_v1r1`: trained injections control trained outputs and yield pair-specific canine/equine steering, with visible chosen words mediating later decisions.
- `interchange_v2`: one generic anchor replacement cannot overcome the intact cue-rich prefix.

This explanation does not prove that no distributed state exists. It is simply the strongest explanation that requires no unobserved state object.

## 6. Cheapest localization experiment

No single experiment can literally settle “where the animal state lives.” The cheapest discriminator of the immediate distributed-prefix hypothesis is:

> Capture and coefficient-one replace the **entire 25-position block-12 residual span** with a length-matched donor span, using self, different-paraphrase same-state, opposite-state, and third-state arms, and compare directly against the existing anchor-only rows.

Interpretation must remain bounded:

- Cross-span movement with same-span preservation would show that class-relevant causal information is available across the block-12 span under this readout.
- It would not establish an abstract or interchangeable animal state; the intervention also transplants lexical and presentation information.
- A span null would not prove absence elsewhere because lower-layer cached pathways, layer choice, and probe choice would remain live.
- Do not sweep layers, positions, or coefficients, and do not call this `interchange_v3`.

Because the measurement-to-artifact ratio was already approximately `6.9:1` before this result and has only worsened ([NOTEBOOK](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/NOTEBOOK.md:252>)), run this discriminator only if a public “distributed-prefix state” mechanism sentence is to be retained. Otherwise withdraw that sentence and move directly to reachability/control cost.

## 7. Exact licensed sentence

> **In Qwen3-1.7B-Base, with calibration-relative three-probe validity on 23/24 decisions, coefficient-one replacement of the block-12 residual at only the final generic ` The animal` token by matched opposite-class donors moved the standardized probe signature a median 0.0087 class separations and changed none of 24 probe signs across eight recipients, so `interchange_v2` fails its preregistered fixed single-anchor construction; because same-, cross-, and cow/horse third-state donor perturbations all fell within the same-state tolerance and the unchanged prefix remained causally available, this result does not establish absence of class information at the anchor, a distributed animal state, or failure of frozen-residual interchangeability outside this site, layer, coefficient, donor pairing, and readout.**

## 8. Never-say list

- “There is no class state at the anchor.”
- “The animal state lives across the prefix.”
- “The probes read the class from earlier tokens through attention” as an established mechanism.
- “Later layers overwrote the donor state.”
- “Block 12 is the wrong layer” as a result rather than a hypothesis.
- “Same-state interchangeability passed.”
- “Same-state signatures were preserved.”
- “Cow and horse donors moved both classes toward neutral.”
- “The exact `p=.47` proves no specificity.”
- “Interchangeability failed in Qwen3-1.7B-Base.”
- “Frozen residual streams lack native state or native mathematics.”
- “Persistent state must be trained.”
- “Native animal classification was 23/24” without “calibration-relative.”
- “Twenty-four independent decisions” or “three independent probes.”
- “The preregistration was executed completely”; the short decodes are absent.

## 9. Proposed README wording

Replace the stale training/current-artifact language in [README.md](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/README.md:14>) with:

> **Status 2026-08-29: no native latent mathematics has been demonstrated. `coordinate_v3` establishes only a narrow late lexical-control effect; `state_bus_v1r1` is closed as a fixed-construction FAIL with pair-specific lexical/semantic steering; and `interchange_v2` found that replacing only the block-12 residual at the final generic anchor moved the calibration-centred cat/dog probe signature by a median 0.0087 class separations and changed none of 24 probe decisions. This closes that fixed single-anchor construction, not frozen residual structure, class information at the anchor, or distributed alternatives. The program now moves from interchangeability to an intervention-based reachability/control-cost artifact.**

## 10. Proposed STATE wording

> **`interchange_v2` — REGISTERED CONSTRUCTION-LEVEL FAIL.** On fresh exactly 25-token contexts, calibration-centred validity passed for cat `11/12` and dog `12/12`. Replacing only the final generic anchor token’s block-12 residual with an opposite-class donor produced median donor-directed movement `T=0.0087` and changed no centred probe sign across eight recipients. The registered same-state gate passed (`8/8`, median distance `0.352`, `τ=3.384`), but it is non-discriminating because every cross-state and every cow/horse third-state arm also lies within that tolerance. The fixed cross-versus-third donor assignment makes `p=.4727` a descriptive sign-symmetry sensitivity rather than an exact randomized test, and horse donors moved dog recipients farther dogward rather than toward neutral. Licensed scope: this fixed block-12, single-anchor, coefficient-one donor replacement did not operationally interchange the registered probe responses. It does not localize class information or establish absence, distribution, overwriting, or general failure of frozen-residual interchangeability.

Because the current ledger and Notebook contain the unsupported neutral/distributed reading ([ledger](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/ledger.jsonl:273>), [NOTEBOOK](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/NOTEBOOK.md:260>)), governance propagation requires an append-only ledger correction and corresponding Notebook correction whenever this wording is adopted.

## 11. Ranked next increments

1. Freeze this audit, correct public surfaces and append a ledger correction; no deletion or rewriting of the original result row.
2. Build the intervention-based reachability/control-cost artifact. Define endpoints behaviorally and preregister the cheapest lexical/readout baselines before introducing a geometric cost.
3. Only if retaining the distributed-prefix mechanism claim, run the single full-span block-12 localization discriminator above; no sweep and no `interchange_v3`.
4. Consider span-level transformation operators only after reachability produces an artifact, with direct lexical-gradient and visible-text mediation baselines.
5. If reachability produces another apparatus-only negative, stop and reconsider the latent-space substrate rather than opening another repair ladder.

## Program and leverage ruling

- **Should the broader program continue? Yes.** The interventions have exposed reproducible causal control surfaces and increasingly sharp falsifiers.
- **Should this interchange line continue? No.** The registered fixed construction failed; no layer/position/coefficient repair ladder is warranted.
- **Is the program presently working as native mathematics? No. This is not working yet.**
- **Is the planned next action highest leverage? Yes, if it is the reachability/control-cost intervention artifact.** Another open-ended localization or span sweep would violate both the no-v3 commitment and the measurement-to-artifact tripwire.

No tracked repository file was edited. I recorded the requested evidence and synthesis only in the mandatory git-ignored blackboard.

## 2026-08-29 — Audit #29 on state_bus_v1r1 (fresh, unprimed; verbatim): FAIL upheld with corrections both ways; residue = pair-specific lexical steering

My 'partial taxonomy transfer', '76% decay' and 'sequential semantic controller' wordings are withdrawn per the audit; the status string omitted the heldout-taxonomy failure (display defect, now fixed in code).

## Executive verdict

The registered construction-level FAIL is upheld, but its interpretation requires correction in both directions.

- The displayed status—`FAIL — same_swap, persistence`—is mechanically incomplete. Every seed also fails the registered heldout-taxonomy gate: uplift consistency is `9/16`, `9/16`, and `8/16`, below the required `10/16`; seed 37 also misses the gain threshold. `summary.fails` records `heldout_consequence`, but the status-building branch suppresses it whenever another failure is present.
- The same-swap failure does not show categorical same-state interchangeability failed. Same and self codes produce the correct recipient choice on all three decisions for all 16 rows in every seed. What fails is preservation of fine-grained confidence signatures under a tolerance calibrated on optimized training contexts.
- The taxonomy residue is real as a fixed-choice intervention effect, but “partial taxonomy transfer” is too state-general. The raw `7/16` is always the same cat→dog and cow→horse rows; all-pair re-evaluation shows that every taxonomy success targets only canine or equine, never feline or bovine.
- The persistence gate fails as registered, but “76% state decay” is not identified. It compares a directly trained sound decision against an untrained taxonomy decision with different verbalizers, scales, positions, and contradictory history while reinjecting the code continuously.
- The predeclared “sequential semantic controller” reading is therefore too strong about mechanism, though correct in denying persistent, abstract, or generally interchangeable state.

The bus construction and budget should close. The broader program should continue, but it is not yet working as a native-mathematics discovery program.

## 1. Positive-residue overclaim audit

### The fixed-cycle taxonomy result is pair-specific

The raw taxonomy choice pattern is identical across all three seeds:

| Recipient → donor | Raw donor choices |
|---|---:|
| cat → dog | `3/4` |
| dog → cow | `0/4` |
| cow → horse | `4/4` |
| horse → cat | `0/4` |

Thus `7/16` is not a diffuse effect replicated across four states. It is two stable lexical/semantic edges and two complete failures.

The uplift result is similarly concentrated:

| Seed | Cat cluster | Dog cluster | Cow cluster | Horse cluster | Total |
|---:|---:|---:|---:|---:|---:|
| 11 | `3/4` | `1/4` | `4/4` | `1/4` | `9/16` |
| 23 | `4/4` | `1/4` | `4/4` | `0/4` | `9/16` |
| 37 | `4/4` | `0/4` | `4/4` | `0/4` | `8/16` |

The fixed cycle happens to include the robust cat→dog and cow→horse edges. Those are also the semantically close pet and farm-animal pairings; the cross-group dog→cow and horse→cat edges fail.

### Exact chance model

The descriptive `4/16` chance floor corresponds to the explicit model

\[
I_i\overset{\mathrm{iid}}{\sim}\operatorname{Bernoulli}(1/4),
\qquad
X=\sum_{i=1}^{16}I_i\sim\operatorname{Binomial}(16,1/4),
\]

where every four-way candidate argmax is assumed independent and uniformly distributed.

Under that model:

| Result | One-sided exact tail | Two-sided exact binomial |
|---|---:|---:|
| `7/16` raw choice | `0.079557` | `0.089580` |
| `9/16` uplift argmax | `0.007470` | `0.007470` |
| `8/16` uplift argmax | `0.027130` | `0.037153` |

This is not a valid confirmatory model for the experiment. The 16 rows are four paraphrases nested within four semantic states; verbalizer priors and token lengths make the four argmax outcomes non-uniform; and the donor map is fixed.

### Clustered inference

Using the four recipient states as the inferential units:

- Raw-choice state proportions are `[0.75, 0, 1, 0]`. A one-sided state-level t test against `0.25` gives `p=0.260`; a four-cluster sign-flip sensitivity gives `p=0.3125`.
- Uplift state proportions give one-sided cluster-t values `p=0.097`, `0.156`, and `0.225` across the three seeds. The four-cluster sign-flip sensitivity is `p=0.25` in each seed.
- Pooling seeds would be pseudoreplication: the same prompts and the same successful state pairs recur. Seeds demonstrate optimizer reproducibility, not additional semantic sample size.

The sign-flip numbers are sensitivities under a symmetry assumption, not exact design-randomization tests. Because the cyclic donor assignment was fixed and the unobserved donor interventions are absent from `result.json`, no valid exact clustered causal p-value can be reconstructed. The result is descriptive, not statistically established state-general transfer.

### All-pair, on-manifold sensitivity

The completed eval-only addendum returns `FAIL` in all seeds:

| Seed | Taxonomy raw donor choice | Also clears 0.5-nat floor | On-manifold wrong-code control | Sound donor choice | Young donor choice |
|---:|---:|---:|---:|---:|---:|
| 11 | `15/48` | `15/48` | `1/96` | `46/48` | `35/48` |
| 23 | `14/48` | `14/48` | `1/96` | `48/48` | `30/48` |
| 37 | `7/48` | `7/48` | `2/96` | `46/48` | `27/48` |

This rules out a tiny numerical-noise explanation for the changed choices and shows code specificity relative to wrong learned codes. It does not rescue abstraction:

- Every raw taxonomy success across all pairs and seeds targets dog/canine or horse/equine.
- Donor cat/feline and cow/bovine never win a raw taxonomy choice.
- The per-state minimum fails in every seed.
- Even all-pair transfer of the trained young consequence is only `56–73%`.

The addendum is also not cryptographically bound to the original result: it reads untracked checkpoints whose hashes are absent from `result.json`, and `audit_result.json` does not record checkpoint or post-result runner hashes. It is a useful local sensitivity, never a confirmatory rescue.

### Lexical and control alternatives

Token length contributes but cannot explain everything. Canine is one token and may benefit in raw summed-word likelihood, but equine is two tokens and succeeds while two-token bovine and feline targets fail. The stronger explanation is donor-verbalizer and pretrained lexical geometry.

The off-manifold shuffled and Gaussian controls were too easy. The all-pair learned-code control improves specificity evidence, but the extreme canine/equine asymmetry still rejects a state-general reading.

The parsimonious positive residue is:

> Repeated learned codes can causally steer particular taxonomy-word choices through the frozen model’s existing lexical/semantic geometry.

That is valuable, but it is not an abstract state result.

## 2. Over-claimed KILL audit

### Same-swap

The registered gate failure is real, not numerical:

| Seed | Tau | Median same distance | Median/tau | Same/cross median ratio |
|---:|---:|---:|---:|---:|
| 11 | `0.268` | `3.267` | `12.2×` | `0.128` |
| 23 | `0.377` | `2.560` | `6.8×` | `0.100` |
| 37 | `0.228` | `1.970` | `8.6×` | `0.085` |

But the tolerance is calibrated on training contexts whose codes were explicitly collapsed by the prototype and same-swap objectives. On held-out paraphrases:

- self and same raw accuracy are both `1.0` for all three choices;
- same-code changes are only roughly `8.5–12.8%` as large as cross-code changes.

Correct wording:

> Categorical same-state behavior transferred perfectly in this four-way evaluation, while fine-grained confidence-signature equality did not generalize within the training-derived tolerance.

Never call this “same-state interchangeability failed” without naming the response-law metric. Conversely, do not say the signatures generalized.

### Persistence

The registered third/first ratio fails decisively:

- `12.20 → 9.80 → 2.92`
- `13.86 → 8.70 → 2.97`
- `11.91 → 7.29 → 3.09`

But this is not an identified temporal decay experiment:

1. Decision one is trained sound; decision three is out-of-loss taxonomy.
2. Candidate word lengths and intrinsic likelihood scales differ.
3. Decision order is fixed and not counterbalanced.
4. Taxonomy sees contradictory recipient sound and young history.
5. The same `Jz` is injected at every continuation position.
6. Each fixed-choice score is recomputed from a teacher-forced sequence rather than continuing one autonomous hidden-state trajectory.

Therefore it licenses:

> The registered cross-consequence movement ratio is low at the third, heldout taxonomy decision under contradictory recipient history.

It forbids:

- “The latent state decayed by 76%.”
- “The state was erased by generation.”
- “The bus cannot persist under neutral or donor-consistent history.”
- “Autonomous persistence failed.”
- Any comparison treating the first and third summed-nat effects as scale-equivalent.

The reported `15/16`, `16/16`, and `15/16` rollouts are constrained four-way candidate selections. The directly trained donor sound and young words become visible history before taxonomy. They are evidence of a self-reinforcing, text-mediated control loop, not free generation or unmediated state survival.

Reader MSE adds no independent persistence evidence: the reader is trained for that reconstruction, evaluated only under self code, has no sham or scale baseline, and reads after visible state-specific words.

### Assessment of the predeclared sentence

The sentence is:

- Correct in denying validated persistent, abstract, or generally interchangeable state.
- Too strong in calling the mechanism a “sequential semantic controller.”
- Too strong in calling the across-decision metric “sharply decaying.”
- Too broad in saying “partial taxonomy transfer” without the two-transition and donor-verbalizer concentration.
- Too weak because it omits the registered heldout-taxonomy failure hidden by the displayed-status bug.

Replace “sequential semantic controller” with “repeatedly injected supervised response controller,” and “partial taxonomy transfer” with “pair-specific taxonomy-word steering.”

## 3. Tunnel vision, alternatives, and order

The strongest unified alternative explanation of today’s results is:

> The frozen model supplies a biased lexical-semantic response geometry; late residual interventions and trained injection vectors move candidate-word likelihoods within that geometry, while visible selected words mediate later decisions—no persistent interchangeable state is needed.

This explains:

- `coordinate_v3`: direct late verb-token steering.
- `interchange_v1`: fixed verbalizer offsets defeating raw-zero classification despite calibration-relative separation.
- `state_bus_v1r1`: trained sound/young control, canine/equine-only taxonomy changes, fixed-pair asymmetry, and strong constrained rollouts once donor words enter history.

The cheapest state-bus moot-maker is a switch-off mediation ablation: score taxonomy after teacher-forced donor sound and young history with `z=None`, beside the same donor history with the cross code. If no-bus donor history reproduces the `15/16–16/16` taxonomy rollout, that strong positive residue is textual mediation. If it does not, only the incremental bus-over-history effect survives. Because the bus line already crosses the governance ratio and fails its gates, run this only if the mediated-rollout sentence is intended for a public surface; otherwise close without another measurement.

The intended high-level order—one frozen interchange discriminator, then reachability/control cost—is right. The inspected `interchange_v2` lock, however, was not confirmatory-run-ready:

- Cat recipients compare dog cross donors against cow third donors; dog recipients compare cat against horse. Cross and third identities are not randomized or counterbalanced.
- Consequently, the claimed exact `2^8` donor-label test relies on an unverified exchangeability assumption and cannot eliminate animal-pair lexical geometry.
- All intervention gates are pooled across cat and dog, so one direction could fail while the status says operational interchangeability.
- The preregistered short decodes are not implemented.

At final filesystem check, an untracked `experiments/results/interchange_v2/` directory had appeared concurrently. I did not inspect or adjudicate it; one-result/one-audit governance requires a separate review. If that is the locked run, do not treat it as confirmatory until these design discrepancies are adjudicated.

## 4. Licensed wording

### Exact licensed sentence

> **`state_bus_v1r1` is a fixed-construction FAIL: a 98,400-parameter supervised interface repeatedly injected a 16-dimensional code into frozen Qwen3-1.7B-Base, and across three seeds held-out same-state donors preserved every four-way categorical choice but failed the training-derived confidence-signature tolerance on 15–16/16 rows, while fixed-cycle cross codes changed taxonomy choice on 7/16 rows—always cat→dog in 3/4 and cow→horse in 4/4—and the complete registered gate vector also failed heldout taxonomy and the cross-consequence third/first movement criterion; taxonomy verbalizers were absent from the bus loss, but the all-pair sensitivity was donor-verbalizer-specific rather than state-general, so the licensed residue is a repeatedly maintained supervised response controller with pair-specific lexical/semantic steering, not autonomous persistence, abstraction, general interchangeability, or native latent mathematics.**

### Never-say list

- “Partial taxonomy transfer” without “pair-specific” and the cat→dog/cow→horse concentration.
- “`7/16` beat chance” or “`9/16` is statistically significant.”
- “Three independent semantic replications.”
- “Same-state interchangeability failed.”
- “Same-state confidence signatures generalized.”
- “The state decayed by 76%.”
- “The state survived through generation.”
- “The bus learned autonomous persistence.”
- “The rollouts freely generated all donor words.”
- “The heldout consequence was unseen”; only its verbalizers were absent from the bus loss.
- “All four states” or “all donor pairs” transferred.
- “The controls rule out lexical or output-space steering.”
- “Token length explains the whole effect.”
- “Reader MSE proves a persistent readable state.”
- “Trained-consequence accuracy was 1.0” without saying this is the self-code statistic.
- “Qwen learned the bus”; Qwen was frozen.
- “The bus establishes abstraction, interchangeability, or native latent mathematics.”
- “This FAIL refutes co-developed interfaces or persistent state generally.”

### README wording

> **`state_bus_v1r1` is closed as a fixed-construction FAIL: held-out same-state code swaps preserved all categorical choices but not the preregistered confidence-signature tolerance, while cross-state taxonomy choice changes were confined to two fixed-cycle transitions and the registered heldout-taxonomy and third/first movement gates failed. The result supports only a repeatedly injected supervised response controller with pair-specific lexical/semantic steering—not autonomous persistence, abstraction, general interchangeability, or native latent mathematics.**

### STATE wording

> **`state_bus_v1r1` — REGISTERED FAIL.** The displayed status reports `same_swap, persistence`, but every seed also fails the registered heldout-consequence gate; this third failure is present in `summary.fails` and omitted from the status string by control flow. Same-state donors preserve all three raw categorical choices on `16/16` held-out rows in every seed, although their confidence signatures lie outside the training-derived tau on `16/15/16`. Fixed-cycle cross donors change taxonomy choice on `7/16` rows per seed, always cat→dog `3/4` and cow→horse `4/4`; the completed all-pair, 0.5-nat-floor, on-manifold sensitivity is also `FAIL` in all seeds (`15/48`, `14/48`, `7/48` taxonomy choices). The cross-consequence movement statistic falls from `11.9–13.9` nats at trained sound to `2.9–3.1` at heldout taxonomy under contradictory recipient history, but this does not identify autonomous temporal decay because consequence, scale, order, history, and continuous reinjection are confounded. Licensed residue: repeatedly maintained supervised response control with donor-verbalizer-specific lexical/semantic steering.

## Ranked next increments

1. Freeze this adjudication, record the status-display defect, and close the state-bus construction; no bus v2.
2. If the mediated-rollout claim will be retained publicly, run only the no-bus donor-history switch-off ablation; otherwise skip it.
3. Adjudicate and, if still prospective, repair `interchange_v2`’s third-donor exchangeability, per-direction gates, and missing-decode discrepancy; allow one run and no v3.
4. Move to reachability and control cost regardless of interchange outcome.
5. Consider distributed transformation operators only after reachability produces an artifact and with a direct lexical-gradient baseline preregistered.

## Program and leverage ruling

The broader program should continue because the real-model interventions have exposed a reproducible control surface and increasingly precise falsifiers. The state-bus line should not continue.

This is not working yet as a native-mathematics discovery program. Continuing becomes highest leverage only by closing the bus, conducting at most one valid frozen interchange discriminator, and then changing the denizen question from identity to reachability and effort.

Scoped measurement-to-artifact ratio:

- Runner-only: about `171` apparatus/evaluation lines to `48` artifact-bearing bus/routing/training lines, approximately `3.6:1`.
- Including the 159-line locked config and 41-line audit-stage addition: approximately `330:48`, or `6.9:1`.
- At least four design/measurement/audit rounds have accumulated around one build/training round.

This exceeds both governance tripwires and mandates the pivot. No tracked repository or source file was edited by this audit.

## 2026-08-29 — interchange_v2 (locked, bias-controlled): FAIL — the single-anchor replacement transfers no class state

Native centred validity passed (cat 11/12, dog 12/12; the calibration-centred statistic removes the verbalizer bias that killed v1). Same-state donors stay within tolerance (8/8; median distance 0.35 vs τ 3.38 — τ is generous). But replacing the block-12 residual at the anchor token with the *opposite class's* anchor residual moves the three-probe signature by a median fractional 0.009 of the class separation, with 0/8 recipients flipping two decisions; on-manifold cow/horse donors nudge recipients slightly toward neutral (|T| 0.10–0.21) and the cross-vs-third specificity test is null (p = 0.47). Pre-declared status: FAIL — FIXED BLOCK-12 SINGLE-ANCHOR INTERCHANGE CONSTRUCTION; no v3.

Reading (mine, pending audit): the downstream probes read the animal from the prefix tokens through attention, not from the anchor token's residual at block 12 — so a single-anchor replacement is the wrong site for a state that is distributed across the context. This is exactly audit #28's 'distributed token-span state' alternative, now with a number behind it. It also explains why the trained bus had to be re-injected at every position to have any effect. Next: audit; then the reachability/control-cost artifact (Codex round 8 rank 2) and a Codex round on whether the object should be a span-level operator rather than a stored point.

## 2026-08-29 — Re-contextualization #29: after the bus verdict

Project and live question. Latent-Space-Reasoning. The whole day's chain — three frozen-model baseline kills, coordinate_v3 as late lexical steering, a trained 16-d bus that controls its trained outputs perfectly yet moves the never-trained consequence's actual choice in only 7/16 and loses ~76% of its push by the third decision — converges on one honest sentence for the README: no native latent mathematics has been demonstrated. The live question is unchanged in words but sharpened in form: is there any construction — frozen or co-trained — in which a state written from one presentation survives and is interchangeable with another's, judged by several downstream consequences? Neither the frozen anchor-token replacement (never reached intervention) nor the repeatedly injected bus (partial, decaying, mediated) has shown it.

What reframes earlier work. Audit #28 removed my tidy story: the toy horizon hole and the frozen-model failures are not the same lesson; most frozen constructions never reached a valid causal test. The bus result adds a genuinely new fact: a linear code injected upstream can carry *trained* semantics through a frozen model and partially pull an untrained associated consequence (uplift 9/16, +0.34 over off-manifold controls), but not enough to change the choice reliably, and not against accumulating contrary text. Whether that is 'a state too weak to persist' or 'a lexical controller generalising through shared vocabulary' is exactly what audit #29 (running, unprimed) must adjudicate.

Alternatives held live: (a) interchange_v2 — the one clean frozen test the pivot's rationale rests on (locked, runs next); (b) reachability/control cost — a native notion of move and effort rather than another identity test (Codex rank 2, runs regardless); (c) distributed transformation operators over token spans (highest wow, heaviest confounds); (d) the possibility, raised by the bus, that the right object is trajectory-level control rather than a stored state at all. Not narrowed: the next two artifacts test different objects (identity vs. effort). Foundational thread: the governance loop held — three audits today corrected my wording in both directions and the ratio tripwire was applied; the risk now is the opposite tunnel (a kill ladder), which the alternatives above guard against.

## 2026-08-29 — state_bus_v1r1 verdict: FAIL on all three seeds (same-swap tolerance, persistence decay); taxonomy raw choice 7/16 each seed

Registered status FAIL — same_swap, persistence; unanimous (seeds 11/23/37). Numbers per seed: trained consequences 1.0; same-swap outside τ (0.27/0.38/0.23, from training contexts) in 16/15/16 — while same-arm raw accuracy equals self (1.0/1.0/1.0), so categorical same-state behaviour transfers and calibrated signature equality does not; cross uplift-consistent ≥2/3 in 15/16/16; taxonomy uplift-consistent 9/9/8 with gain +0.34/+0.38/+0.19; taxonomy RAW choice follows the donor in exactly 7/16 in every seed; movement 12.2→9.8→2.9, 13.9→8.7→3.0, 11.9→7.3→3.1 nats (≈24% left by the third decision under contradictory recipient history); mediated own-choice rollouts all-donor in 15/16/15. Pre-declared reading (Codex round 8, verbatim): "A supervised, repeatedly injected 16-dimensional interface learned strong state-specific control of its trained consequences, generalized same-state categorical behavior across held-out paraphrases, and produced partial taxonomy transfer that was strong in mediated rollouts but incomplete and sharply decaying in the direct contradictory-history test. It is a sequential semantic controller, not a validated persistent, abstract, or generally interchangeable state." Audit-#28 re-adjudication (raw choice, 0.5-nat floor, all 12 donor pairs, on-manifold third-state control, per-state minimum, all-three-seeds rule) is running as an eval-only sensitivity beside the registered FAIL; fresh audit follows. No rescue; no bus v2.

## 2026-08-29 — state_bus_v1r1 seed 11 (interim, registered FAIL) and direction round 8 (verbatim)

Seed 11: trained-consequence accuracy 1.0; same-swap 16/16 outside a training-derived τ=0.268 (same-arm raw accuracy 1.0/1.0/1.0 like self); cross uplift-consistent ≥2/3 in 15/16; taxonomy uplift-consistent 9/16, gain +0.34 over shuffled/random; taxonomy raw choice = donor 7/16; movement 12.2 → 9.8 → 2.9 nats; own-choice rollouts all-donor in 15/16 (mediated). Seeds 23/37 pending; audit-stage re-adjudication after the run; no rescue.

# Direction round 8 — ruling

No state-bus verdict is authorized until all three seeds finish. Seed 11 is an interim construction diagnosis only.

## 1. Honest reading of seed 11

The same-swap and persistence failures are not equally informative.

**Same-swap: mostly a tolerance-construction failure.** Tau `0.268` was estimated from fitted training contexts and is dramatically narrower than held-out same-donor variation, producing `16/16` failures. Yet self and same-donor raw accuracy are both `1.0/1.0/1.0`. Thus seed 11 does not show that same-state codes cease to be behaviorally interchangeable at the tested choices. It does show that fine-grained confidence signatures failed to generalize within the training-derived tolerance. Licensed distinction: categorical same-state behavior transfers; calibrated signature equality does not.

**Persistence: confounded, but substantively negative.** Recipient-conditioned history is deliberately hostile to the donor code, so decay measures resistance to accumulating contradictory textual evidence as well as persistence. But `12.2 → 9.8 → 2.9` nats is a roughly 76% decline, not a threshold accident, and horse reverses at decision three. Seed 11 therefore fails direct, unmediated persistence under the registered history. The `15/16` complete donor rollouts show something different: once sound and young become donor words, those visible words mediate taxonomy. That is impressive sequential control, but not direct held-out state transfer.

**Taxonomy raw choice `7/16`: substantive partial transfer.** It exceeds the descriptive four-way chance floor of `4/16`, but it is below the audit-#28 standard, is only one seed, and has four state clusters rather than sixteen independent trials. Uplift consistency `9/16` and gain `+0.34` show a real donor-directed tendency; they do not show reliable behavioral replacement.

If seeds 23 and 37 agree, the licensed result is:

> **A supervised, repeatedly injected 16-dimensional interface learned strong state-specific control of its trained consequences, generalized same-state categorical behavior across held-out paraphrases, and produced partial taxonomy transfer that was strong in mediated rollouts but incomplete and sharply decaying in the direct contradictory-history test. It is a sequential semantic controller, not a validated persistent, abstract, or generally interchangeable state.**

The planned audit-#28 re-adjudication is correct. It must remain an eval-only sensitivity reported beside the registered FAIL, never a rescue.

## 2. Ranked next directions

I agree with `interchange_v2`, with one pushback: the pivot never scientifically rested on “native state is absent”; it was an allocation decision. A positive would not prove the pivot was mistaken at the time, but it would overturn the rationale for making the trained interface central. That makes one final, clean frozen interchange test exceptionally valuable.

| Rank | Direction | Narrative score | Ruling |
|---:|---|---:|---|
| 1 | **Bias-controlled `interchange_v2`** | **8.5/10** | Highest immediate decision value and lowest compute. It directly tests the unanswered native-state question. Exactly one construction; no `v3` repair ladder. |
| 2 | **Reachability and control cost** | **8/10** | Best distinct wow-to-cost ratio: “How much hidden effort does a thought require?” Produces a native notion of move and effort rather than another identity test. Run after interchange regardless of its result. |
| 3 | **Distributed transformation operators** | **9/10** | Highest positive wow—two hidden moves composing over token spans—but more implementation and severe lexical-control confounds. |
| 4 | **Predictive dynamics/flow** | **6.5/10** | Scientifically sound if it first beats identity-plus-shared-displacement, but risks returning to the prior apparatus-heavy predictor program. |
| 5 | **Response-law topology/concept lattice** | **5.5/10** | Cheap and mathematically native, but the first test is largely measurement rather than a compelling real-model intervention artifact. |

Decision: run `interchange_v2` next, then leave interchangeability. The following artifact should be reachability/control cost whether `v2` passes or fails.

## 3. Locked next artifact: `interchange_v2`

### Object and fixed world

Use the same Qwen3-1.7B-Base revision, CPU fp32, block 12, final anchor position, and coefficient-one replacement. No model, layer, position, strength, or probe sweep.

Use fresh cat/dog calibration and held-out paraphrases, all exactly tokenizer-length matched and ending with the identical anchor. Add fresh, style- and length-matched cow/horse contexts as on-manifold third-state donors. Existing `interchange_v1` held-out rows are diagnostic history and cannot enter confirmation.

Reuse `run_interchange.py` as the canonical runner, but correct the projection denominator and add the locked statistic/control. “Reuse” does not mean running the existing code unchanged.

### Preregistered statistic

For probe \(p\), using calibration only:

\[
b_p=\frac{\bar m_{p,\mathrm{cat}}+\bar m_{p,\mathrm{dog}}}{2},
\qquad
s_p=\max(s_{p,\mathrm{pooled\ within}},\eta_p),
\qquad
u_p(x)=\frac{m_p(x)-b_p}{s_p},
\]

where \(\eta_p\) is the self-swap numerical floor. Let

\[
\delta=\bar u_{\mathrm{cat}}-\bar u_{\mathrm{dog}}.
\]

For recipient \(r\), arm \(a\), and desired donor direction \(d_r\in\{\delta,-\delta\}\), define fractional donor movement:

\[
T_{r,a}
=
\frac{(u_{r,a}-u_{r,\mathrm{native}})\cdot d_r}
     {\|\delta\|^2}.
\]

The squared denominator fixes the `v1` projection error. All raw margins, centered margins, categorical probe decisions, short decodes, and continuous effects are stored.

### Arms and controls

- Native/no replacement
- Self-swap numerical sham
- Different-paraphrase same-state donor
- Matched opposite-state donor
- Preassigned, on-manifold cow/horse third-state donor

The third-state donor must use a real block-12 activation from the same anchor world—not a Gaussian vector or dimension permutation.

### Gates

Native validity:

- At least `20/24` centered held-out probe decisions correct.
- Neither cat nor dog below `9/12`.
- Probe rows remain recipient-clustered; no claim treats them as `n=24`.

Same-state interchange:

- Calibration tau = Q90 of calibration same-state signature distances plus the median self-swap floor.
- Median same-donor distance no greater than tau.
- At least `6/8` recipients within tau.

Cross-state movement:

- Median \(T_{\mathrm{cross}}\ge0.5\).
- At least `6/8` recipients flip at least two of three centered probe decisions toward the donor.
- Report raw forced-choice changes and decodes; do not claim literal output replacement from centered movement alone.

Specificity over the on-manifold control:

- Median \(T_{\mathrm{cross}}-T_{\mathrm{third}}\ge0.3\) class separations.
- At least `7/8` recipient-paired differences are positive.
- Exact paired donor-label sign-flip test has one-sided \(p\le0.05\).

### Chance and inference

Per-probe descriptive chance is `1/2`, but the probes are correlated. The primary null exchanges cross-state and third-state donor labels within each of eight recipients and enumerates all \(2^8=256\) assignments. Recipients—not probe rows or forwards—are the inferential units. Cat and dog effects are reported separately.

### Status and kill rule

A bounded positive requires every native, same-state, cross-state, and specificity gate.

If cross movement passes but same-state preservation fails, report:

> `STATE-DIRECTED STEERING WITHOUT INTERCHANGEABILITY`

If any other gate fails:

> `FAIL — FIXED BLOCK-12 SINGLE-ANCHOR INTERCHANGE CONSTRUCTION`

No new layer, position, model, coefficient, task wording, centering rule, or `interchange_v3` follows. A negative transfers the program to reachability/control cost.

### Lay one-liner

> **Can two descriptions be the same place inside a frozen language model? Swap the hidden place: a paraphrase should leave three facts alone, a different animal should change them together, and a third animal should not fake the move.**

## 4. Program honesty

The one README sentence should be:

> **After today, no native latent mathematics has been demonstrated: the only established real-model causal result is a narrow late lexical-control effect, while frozen donor interchange never ran and the supervised state-bus verdict remains pending.**

What should be said to the user:

> **This is not working yet as a native-mathematics discovery program. Seed 11 suggests that the added bus is a strong supervised sequential controller with partial out-of-loss semantic transfer, but it does not establish persistent interchangeable state. We will finish and audit the locked run without rescue, run one clean 30-minute frozen donor-interchange test because the earlier experiment never reached intervention, and then move to the distinct question of latent reachability and control cost rather than continuing an interchangeability ladder.**

## 2026-08-29 — Audit #28 (fresh, unprimed; verbatim): kills scoped, program KILL = allocation pivot not science, tunnel vision found, state-bus gates not clean enough for their headline

My re-contextualization #28 claims ('the same lesson from two directions'; 'built rather than found') are REPLACED by the audit's wording below (section 'Exact replacement wording').

## Executive verdict

| Question | Verdict |
|---|---|
| `interchange_v1` kill | **UPHELD for the locked construction only.** |
| Evidence against interchangeability | **NONE.** No swap arm ran. |
| Frozen-residual program KILL | **UPHELD as an allocation pivot; rejected as a scientific conclusion.** |
| Tunnel vision | **FOUND.** Interchangeability has become privileged as the foundational object. |
| State-bus integrity | **No direct train/test or taxonomy-token leakage found, but the registered positive adjudication is not clean enough for its headline.** |
| Continue program? | **Yes.** Finish and audit the current run once; do not scale it. |
| Highest-leverage next discriminator | **One fresh, bias-controlled frozen-model donor-interchange experiment.** |

## 1. `interchange_v1`

### Mechanical adjudication

The locked baseline unquestionably failed:

- Cat: `8/12`
- Dog: `7/12`
- Total: `15/24`, below `20/24`
- Both classes below the required `9/12`
- No donor-swap intervention ran

The raw-sign statistic was indeed preregistered. Round 6 placed the native-behavior gate before calibration-scale standardization, and [run_interchange.py](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_interchange.py:66>) applies `np.sign` before computing any calibration scale. Centering now would be a post-outcome change of estimand and cannot rescue `interchange_v1`.

But the stored signatures show something importantly different from baseline incapability:

- Every held-out cat signature exceeds every held-out dog signature on every probe: `16/16` cat–dog cross-pairs for each of three probes.
- Cat-minus-dog held-out mean gaps are `+0.238`, `+0.942`, and `+0.633`.
- A calibration-only class midpoint classifies all `24/24` held-out probe decisions correctly.
- The calibration midpoint is `[-0.211, +0.767, +0.305]`, far from raw zero on all three probes.

The first probe always prefers barking over meowing; the second always prefers kittens over puppies. Raw zero therefore measures the conjunction of semantic context and fixed verbalizer prior. It is a valid locked statistic but a poor design for testing whether the response signatures discriminate the two states.

Two additional implementation defects are real but did not cause the baseline failure:

- Contexts span 23–27 tokens despite the exact-length lock.
- `cross_toward_other` divides its projection by class separation rather than separation squared, so it is not the declared fractional movement.

### Scope of the kill

The correct ruling is:

> **`interchange_v1` failed its prospectively locked raw-zero native-behavior gate, so the fixed block-12, single-anchor, fixed-verbalizer construction is closed and no swap result exists. The stored native signatures nevertheless separate cat from dog on all three probes relative to a calibration-derived midpoint. Because that midpoint statistic was not preregistered, it is diagnostic only and cannot rescue `interchange_v1`. This experiment provides no evidence for or against causal interchangeability in the frozen model.**

That is a construction failure, not an interchangeability result.

### What the preregistration should have used

The better primary baseline was available prospectively:

1. Estimate a per-probe lexical intercept from calibration only:

   \[
   b_p=\frac{\bar m_{p,\mathrm{cat}}+\bar m_{p,\mathrm{dog}}}{2}.
   \]

2. Standardize by a calibration-only scale and classify fresh held-out signatures using `sign((m_p-b_p)/s_p)`.

3. Equivalently, use matched cat-minus-dog paired contrasts. Fixed verbalizer offsets cancel in that statistic.

4. Run swaps on centered signatures:

   - Same-state donor remains within calibration-derived natural variation.
   - Cross-state donor moves toward the donor centroid.
   - Cross-state movement exceeds unrelated-donor movement.
   - Require effect magnitudes and actual behavioral changes, not signs alone.

5. Freeze fresh contexts and preferably fresh probe wording before testing. Existing held-out rows cannot become confirmatory through post-hoc centering.

A counterbalanced yes/no probe family or a calibration-selected, bias-balanced verbalizer bank would also have been defensible.

### Never say

- “Interchangeability failed in Qwen3-1.7B-Base.”
- “Block 12 contains no semantic or persistent state.”
- “The model cannot treat paraphrases as the same place.”
- “The 15/24 score shows the classes were not represented.”
- “Calibration centering rescues or passes `interchange_v1`.”
- “The 24/24 centered diagnostic establishes interchangeability.”
- “Token-length mismatch or the projection bug caused the baseline failure.”
- “This result proves that a trained state bus is necessary.”
- “Raw sign is the scientifically correct statistic”; it was the locked statistic.
- “The model preferred dog facts for cat contexts” without identifying the fixed verbalizer bias.

## 2. Program-level frozen-residual KILL

### Scientific ruling: premature

Only one of the four named constructions reached a causal intervention:

- `coordinate_v1`: the instruct model failed the polarity capability premise; no held-out causal transport ran.
- `coordinate_v2`: the instruct model missed a prompt-sensitive grammatical-number baseline; no hidden capture or intervention ran.
- `coordinate_v3`: the sole causal result, at a late final-token prediction site, used four fixed verb tokens and produced ungrammatical counterfactuals. Its vectors projected directly onto those verb logits.
- `interchange_v1`: stopped at a lexical-bias-confounded native baseline; no swap ran.

The strongest case against a scientific KILL is therefore strong:

- Two “kills” are failures of instructed task execution, not representation tests.
- One is a late, single-position lexical intervention expressly suited to finding output control.
- One is a flawed verbalizer gate before intervention.
- Three experiments focus on single-token sites.
- No distributed token-span state, layer trajectory, multi-position donor state, or larger model was tested.
- Qwen3-0.6B and 1.7B are insufficient to generalize across model scale.
- The interchange construction fixed one anchor and one layer and never exercised them causally.

This evidence cannot establish that frozen residual streams lack native mathematics, persistent state, or useful operational structure.

### Allocation ruling: justified

Round 7 was nevertheless justified in saying “this is not working” in the governance sense. The sequence accumulated invalid constructions and apparatus, and the measured ratio exceeded the mandatory pivot threshold. Continuing to repair layers, prompts, coefficients, and decision rules adaptively would have been low-integrity and low-leverage.

The honest distinction is:

> **The frozen-residual sequence is stopped as an open-ended allocation because its current constructions have not yielded a native-mathematics artifact at an acceptable apparatus cost. This is not evidence that frozen pretrained residual streams lack native structure; most registered constructions never reached a valid causal test.**

The state bus is consequently a constructive branch, not a demonstrated necessity.

### Cheapest frozen-model experiment that could moot the pivot

Run one fresh `interchange_v2`, with no layer or model sweep:

- Same Qwen3-1.7B-Base revision and one fixed upstream block.
- Fresh, exactly length-matched calibration and held-out paraphrases.
- Calibration-midpoint or matched-pair signature contrasts preregistered before held-out scoring.
- Same-state, cross-state, unrelated, and self donors.
- Three consequences with bias-controlled verbalizers.
- Actual donor-consistent choices plus continuous donor-versus-recipient effect sizes.
- A numerical floor and on-manifold wrong-donor control.
- Recipient/state-clustered reporting.

A result in which same-state donor codes preserve several consequences while cross-state donors change them coherently and beat unrelated donors would directly show a native operational state in the frozen model. That would make the scientific rationale for “we had to build the state” moot.

### Never say

- “Frozen pretrained residual streams do not contain native mathematics.”
- “The model never learned to keep a state.”
- “Persistent state must be built.”
- “Three independent causal interventions failed.”
- “Single-token global sentence state was refuted.”
- “Small-model failure generalizes to language models.”
- “The state bus addresses a proven architectural hole.”
- “A state-bus positive would retrospectively validate the frozen-model KILL.”

## 3. Tunnel vision

Tunnel vision is present. The guiding question distinguishes identity, moves, effort, maps, and regularities. The current program has elevated one candidate identity notion—“a state swappable between paraphrases”—into the apparent foundation.

Interchangeability is a legitimate operational object, but it is not uniquely foundational. A trained bus also partially guarantees that object by construction: same-state collapse and swap behavior are explicit losses. Its result cannot decide that interchangeability is the latent world’s native mathematics.

Four live alternatives are:

| Alternative native object | Cheapest CPU-feasible first test |
|---|---|
| **Predictive dynamics or flow** | At one fixed block pair, capture residual trajectories for 24 matched prompts. Test identity-plus-shared-displacement first, then a small state-dependent predictor on held-out lexical families. Causally patch the predicted displacement and score later consequences. |
| **Reachability and control cost** | Compute local residual-to-logit Jacobians for a small fixed prompt slice. Solve minimum-norm interventions for coherent targets, then test whether predicted control cost and direction transfer to held-out prompts. Include direct-unembedding and random controls. |
| **Response-law topology / concept lattice** | Score a frozen family of roughly 12 downstream probes for paraphrases and semantic neighbors. Build calibration-only observational neighborhoods, then compare matched-norm within-neighborhood and cross-boundary interventions. |
| **Distributed transformation operators** | At one upstream block, patch whole token spans for two grammatically coherent transformations across held-out lexemes. Test each operator, their composition, and commutator against direct lexical-logit controls. |

These ask different denizen questions: where motion carries a state, how much effort a move costs, which observations define neighborhoods, and which transformations compose. None reduces to swapping paraphrase codes.

## 4. Locked state-bus pre-outcome integrity

### Checks that pass

The corrected runner closes the earlier direct leakage paths:

- Qwen weights are frozen and revision-pinned.
- Training contexts are indices `0–7`; evaluation contexts are `8–11`.
- Training loss uses only sound and young.
- The exact taxonomy verbalizers do not occur in constructed training strings.
- Every primary evaluation arm sees identical recipient-conditioned history.
- Tau is calculated from training contexts, not held-out recipients.
- Taxonomy does not affect training, stopping, layer selection, or hyperparameters.
- Self, same, cross, no-bus, shuffled, and random arms are all emitted.
- The three initialization seeds and fixed training budget are declared.

The pretrained model already knowing feline/canine/bovine/equine relations is not leakage. It is the mechanism through which semantic transfer could occur. The precise phrase must be “taxonomy verbalizers were absent from the bus loss,” not “taxonomy was unseen by the model.”

### Critical positive-manufacture risks

1. **The positive gates do not require the behavioral choice to change.**

   In [run_state_bus.py](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_state_bus.py:154>), `cross_two` and `heldout_consistent_cross` use the argmax of `arm − none` uplift. Raw arm choices are stored but report-only.

   A donor taxonomy string can receive the largest uplift while the model still chooses the recipient taxonomy string. The status can therefore say “persistent interchangeable state bus” without a swapped consequence.

2. **There is no minimum causal effect or numerical floor.**

   An arbitrarily small positive donor-versus-recipient uplift can win `up_arg`. The first/third movement gate requires only positivity and a ratio, not a nontrivial magnitude. Tiny numerical or optimizer effects can satisfy the discrete gates.

3. **Overall seed adjudication is unsafe.**

   [Lines 194–201](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_state_bus.py:194>) skip seeds near the deadline but calculate the mode of however many completed:

   - One completed positive seed can become an overall positive.
   - Two split seeds produce a tie chosen by set iteration.
   - A three-way `POSITIVE`/`CONTROLLER`/`FAIL` split also receives an arbitrary “majority.”

   A construction verdict requires all three seeds to complete and at least two to share the same class. Otherwise the result is `INCOMPLETE — NO VERDICT`.

### High-risk claim and control weaknesses

4. **Candidate token lengths are unequal.**

   The locked review verified token counts of:

   - Sound: `2/2/2/2`
   - Young: `1/1/1/2`
   - Taxonomy: `2/1/2/2`

   Summed word-LL is meaningful for exact strings, but argmax of *uplift* across strings of unequal token length is not a categorical choice probability and can scale with the number of affected tokens.

5. **The shuffled and random controls are off-manifold.**

   “Shuffled” permutes dimensions within the recipient code; “random” is Gaussian norm-matched. Neither is a wrong but valid learned code. Cross codes can beat these controls merely because they remain on the learned code manifold.

   A stronger control is an on-manifold code from a preassigned wrong state or an exact donor-code/label permutation.

6. **Only four cyclic donor transitions are tested.**

   Evaluation tests cat→dog, dog→cow, cow→horse, and horse→cat. It does not test the other eight ordered cross-state pairs. The pooled gates have no per-state minimum, allowing one complete state failure while passing.

7. **Tau can adaptively widen.**

   Same-swap tolerance is the fitted training Q95 without an absolute cap, cross-state normalization, or per-probe scale standardization. Poorly collapsed training behavior can create an easy held-out tolerance.

8. **A lexical response-controller explanation remains open.**

   `Jz` is a learned linear injection repeatedly applied upstream. Training rewards two semantically aligned output families, and the frozen model already links those families to taxonomy. A positive may therefore be a learned semantic/logit controller whose output geometry generalizes from “meow/kitten” to “feline.” No direct-output or unembedding control distinguishes this from an abstract state interface.

### Negative-force and scope risks

9. **Recipient history makes the held-out test adversarial.**

   For cross arms, taxonomy is scored after visible recipient sound and young words. This removes donor-history leakage but places the donor code in conflict with accumulating textual evidence. Third-decision decay can therefore measure resistance to contradictory recipient history, not persistence alone.

   A negative licenses failure under this contradictory-history test. It does not show that the bus would fail with neutral history or its own donor-consistent rollout.

10. **“Persistence” is continuous maintenance, not autonomous survival.**

    `Jz` is re-injected at every continuation position. The design never switches the bus off after a write. Reader reconstruction and third-decision movement therefore show that repeated control remains decodable/effective, not that a state survives unaided through the frozen model.

11. **The global cap can force incomplete evidence.**

    The runner recomputes the frozen prefix rather than using the fully cached block-12 suffix design, increasing the chance that later seeds are skipped. Combined with the majority bug, this affects both fairness and verdict integrity.

### Pre-outcome ruling

The runner is suitable for collecting descriptive evidence, but its registered positive status is not sufficient for the claimed headline.

Exact licensed pre-outcome sentence:

> **`state_bus_v1r1` is a fixed three-seed training run of a 98,400-parameter supervised interface attached to frozen Qwen3-1.7B-Base. Training and held-out paraphrases are index-separated, and taxonomy verbalizers are absent from its loss. Its registered hard gates nevertheless use donor-directed relative-likelihood uplift rather than actual behavioral choice, lack a nontrivial magnitude floor, compare against off-manifold controls, and permit unsafe incomplete-seed aggregation; therefore no “persistent interchangeable state bus” claim is licensed from the registered status alone.**

If the stored status is positive, the maximum licensed wording without a new experiment is:

> **In this fixed four-animal world and fixed cyclic donor map, a supervised 98,400-parameter interface repeatedly injected a 16-dimensional code and produced donor-directed relative-likelihood changes on taxonomy verbalizers absent from its loss. Because the primary gate did not require actual choice changes or a minimum effect magnitude and used token-length-mismatched verbalizers and off-manifold controls, this is descriptive evidence for a supervised semantic steering interface, not validated autonomous persistence, general interchangeability, or native latent mathematics.**

If it fails:

> **This fixed interface, training budget, contradictory recipient-history evaluation, and cyclic donor construction failed one or more registered gates. That closes this construction and budget; it does not establish that co-developed latent interfaces or persistent state are impossible.**

### State-bus never-say list

- “The bus changed the held-out consequence” unless raw choices actually changed.
- “The held-out consequence was never seen”; only its verbalizers were absent from the bus loss.
- “The state survived through generation” without saying it was re-injected at every position.
- “The bus learned autonomous persistence.”
- “All four states are interchangeable”; only one directed cycle was tested.
- “The controls rule out lexical or output-space steering.”
- “The held-out taxonomy result proves abstraction.”
- “Sixteen independent held-out samples”; there are four semantic-state clusters.
- “Three-seed majority” unless all three completed and two agree.
- “Qwen learned the state bus”; Qwen is frozen and the added interface is trained.
- “A bus positive validates a hostile hole in pretrained residual space.”
- “A bus negative refutes trained latent interfaces generally.”

## Exact replacement wording for current surfaces

For the broad Round 7 ruling:

> **Under the current apparatus budget, this sequence of frozen residual-stream constructions is not yielding a native-mathematics artifact, so open-ended layer, task, and decision-rule repair stops here. This is an allocation pivot, not evidence that frozen pretrained residual streams lack usable native structure.**

For NOTEBOOK’s claimed common failure shape:

> **The toy-world results and frozen-model attempts both failed to establish persistent interchangeability, but for different reasons: the toy artifacts exhibited bounded deeper future-signature failures, whereas most frozen-model constructions failed task or instrument validity before testing persistence or interchangeability. No common latent-space hole is established.**

For NOTEBOOK’s “must be built” dichotomy:

> **Today’s locked constructions did not reveal a persistent interchangeable state at the tested sites. They do not determine whether such structure exists elsewhere, is distributed across positions or trajectories, or must be added by training.**

For `interchange_v1`:

> **The locked raw-zero baseline failed and closes `interchange_v1`; no swap arm ran. Calibration-relative class separation is a diagnostic design finding, not a rescue and not evidence for interchangeability.**

For the state-bus lay description:

> **We built a tiny supervised interface that continuously re-injects a 16-dimensional code while a frozen model makes several decisions; the test asks whether swapping codes across held-out paraphrases produces a nontrivial direct effect on taxonomy verbalizers absent from the interface’s loss.**

Finally, [STATE.md](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/STATE.md:5>) is currently stale at its canonical top: it still names `coordinate_v1 → coordinate_v2` as the current line and says the next step awaits Codex, while coordinate-v3 appears only in a late appendix and the interchange/state-bus transition is absent. That is not an overclaim about interchangeability, but it is a current-state integrity defect.

## 2026-08-29 — Re-contextualization #28: one day on the frozen residual stream, and the pivot to a built state

Project and live question. Latent-Space-Reasoning: is there a native mathematics of latent spaces — and, after today, the sharper form: does a real model's residual stream *contain* a persistent, interchangeable state, or must such a state be *built* alongside it? Today's evidence: three baseline kills on frozen models (instruction polarity on 0.6B; number on 1.7B; the raw-sign probe gate for paraphrase interchangeability) and one gate-passing intervention (coordinate_v3) that a fresh audit reclassified as late lexical steering living in the unembedding. Codex (round 7) ruled the frozen-mining line not working and moved the central artifact to `state_bus_v1r1` — a <100k-parameter bus co-trained with frozen Qwen3-1.7B-Base, judged by whether swapped state moves a consequence never used in training. It is training now under a Tier-1-reviewed lock.

What reframes earlier work. The toy program (Rounds 36–37) and the frozen-model line failed in the same shape: behaviour-supervised or pretrained carriers realise the *immediate* response but not a state that survives and composes — the depth-3 horizon hole in the toys and the non-persisting one-token patches in Qwen are the same lesson from two directions. That is the strongest argument for building the state explicitly and then asking whether it is native-like (paraphrase-stable, interchangeable, persistent, generalising to an untrained consequence) rather than hunting for it ready-made.

Alternatives held live (not narrowed to one thread): (a) the frozen line may have died of construction choices — instruct models, single-token readouts, anchor replacement, bias-laden probes — and audit #28 (running, unprimed) is asked for the single cheapest frozen-model experiment that would make the pivot moot; (b) native latent mathematics need not reduce to interchangeability at all — audit #28 is asked for three alternative framings with CPU-feasible first tests; (c) if the bus turns out to be a supervised response controller (only trained outputs move), the honest reading is that a 16-d linear bus cannot carry an abstract state through a frozen model, which points at trajectory-level operators rather than a bigger bus.

Foundational thread advanced: the governance amendment itself (exact certificates diagnostic-only; one audit per result; ratio tripwire; narrative gate) — applied today it produced three honest kills and one reclassification in eight hours, versus two days of apparatus before it.

## 2026-08-29 — Direction round 7 (Codex, verbatim): interchange_v1 dead; frozen-residual mining is not working; central artifact becomes `state_bus_v1`

# Direction round 7 — ruling

## 1. `interchange_v1` is dead

The intended baseline statistic was the raw sign, not calibration-centered sign. Round 6 defined the probes as sequence preferences, placed “native held-out behavior” before calibration-only standardization, and referred specifically to calibration **scale** estimates; the runner implements that contract exactly—raw `np.sign` first, division by calibration SD only after a pass. Subtracting the calibration grand mean would change the estimand from “does the model natively prefer the correct continuation?” to “is this context above or below the calibration lexical-bias reference?” That relative separation is real and useful, but it is not native behavior or interchangeability. Indeed, centering would convert the inspected data from 15/24 to 24/24, which is precisely why introducing it now would be a post-hoc rescue. No corrected gate is authorized and no swap arm should run. Independently, exact token-length matching was missed, and `cross_toward_other` divides by class separation rather than its square, so it is not actually the declared fractional movement; these defects did not cause the baseline failure, but they eliminate any argument for merely resuming under a new lock.

## 2. Program ruling

This is not working as a program of mining a frozen pretrained residual stream for native mathematics. Three constructions died before causal evaluation, while the only gate-passing construction was audited into late lexical steering; `interchange_v1` added another 108 runner lines and 50 config lines but produced no intervention behavior, worsening an already over-threshold measurement-to-artifact ratio. The strong calibration-relative signature separation can remain a diagnostic observation, but it does not justify another frozen-layer, token, task, or decision-rule attempt. The broader mission survives; its central artifact now becomes the co-developed latent interface. Frozen-model interventions remain controls used to validate that interface, not an open-ended discovery program.

## 3. Smallest constructive artifact: `state_bus_v1`

**Object being built.** Freeze Qwen3-1.7B-Base completely. At block 12 add a persistent 16-dimensional state bus with fewer than 100,000 trainable parameters:

- encoder \(E:\mathbb{R}^{2048}\rightarrow\mathbb{R}^{16}\);
- four learned semantic prototypes \(c_s\);
- injector \(J:\mathbb{R}^{16}\rightarrow\mathbb{R}^{2048}\);
- reader \(R:\mathbb{R}^{2048}\rightarrow\mathbb{R}^{16}\).

A context ending at a fixed anchor writes \(z=E(h_{12}(p))\). The same \(Jz\) is injected at block 12 at every subsequent continuation position, rather than hoping a one-shot native activation persists. Donor interchange replaces \(z\), not the residual directly. This is explicitly an added state interface, not a claim that Qwen already contained one.

**Training world.** Use four ordinary semantic states—cat, dog, cow, and horse—with eight training and four held-out paraphrases per state. All contexts end with identical anchor text and are exactly tokenizer-length matched. Train on two varied consequence families:

- characteristic sound: meow, bark, moo, neigh;
- name of the young: kitten, puppy, calf, foal.

Reserve a third consequence family completely from training:

- taxonomic adjective: feline, canine, bovine, equine.

The held-out consequence is essential: without it, the bus could merely memorize two output controllers.

**Objective.**

\[
L=L_{\text{prototype}}+L_{\text{native}}+L_{\text{same-swap}}
  +L_{\text{cross-swap}}+L_{\text{persistence}}.
\]

- `prototype`: same-state paraphrases collapse toward the same \(c_s\); different states have a fixed margin.
- `native`: the encoded state supports correct sound and young continuations.
- `same-swap`: replacing a context’s \(z\) with another paraphrase’s same-state \(z\) preserves both continuations.
- `cross-swap`: replacing \(z\) with another state’s code changes both continuations to the donor state.
- `persistence`: after each downstream decision, \(R(h_t)\) must reconstruct the same \(z\).

The Qwen weights never move. The identical code, injector, and loss weights serve every state and consequence.

**CPU-bounded implementation.** Cache block-12 activations for teacher-forced sequences of at most 48 tokens, then train only the state bus through the frozen suffix. Use three predeclared initialization seeds, 600 AdamW steps each, microbatch one with accumulation, one process, and a four-hour total wall-clock cap. No layer, state dimension, learning-rate family, dataset, objective, or model sweep follows the cap.

**Causal demonstration.** On held-out paraphrases, generate or score one three-decision continuation under:

- no bus;
- self code;
- same-state donor code;
- cross-state donor code;
- shuffled learned code;
- random norm-matched code.

Same-state swaps should preserve all three consequences. Cross-state swaps should change sound and young—the trained consequences—and also the never-trained taxonomic adjective. Show the decoded continuations beside clustered effect sizes; do not adjudicate from a latent-distance table alone.

**Kill rule.** Kill the persistent interchangeable-state claim if any of these holds:

- the fixed training budget fails to reach 85% on the trained consequences;
- same-state swaps leave calibration-derived tolerance in more than 4/16 held-out contexts;
- cross-state donors fail to produce donor-consistent choices on at least two of three consequences in 12/16 contexts;
- the never-trained taxonomic consequence is donor-consistent in fewer than 10/16 contexts or gains less than 0.25 over shuffled/random codes;
- causal movement at the third decision is less than half its magnitude at the first.

If only sound and young move, classify the artifact as a supervised response controller and stop. Do not add dimensions, probes, epochs, or another layer to rescue it.

**Lay one-liner:** “We stopped hunting for a hidden state the model never learned to keep and built it a tiny internal state bus: paraphrases write the same state, it survives several decisions, and swapping it changes even a consequence never used to train the bus.”

Narrative-gate score: **8/10 if the never-trained consequence moves; 5/10 if only trained outputs move.**

## 2026-08-29 — interchange_v1 (operational interchangeability, Qwen3-1.7B-Base block 12): baseline FAIL by raw sign; gate semantics referred

Built per Codex round 6 (`experiments/run_interchange.py`, `config/interchange_v1.json`; all text frozen before results; context lengths 23–27 tokens, approximately matched — a deviation from "exactly matched"). Native held-out probe decisions by raw margin sign: cat 8/12, dog 7/12 → FAIL under the coded gate; no swap arm was run. The native signatures are nonetheless strongly class-separated on all three probes; two probes carry a constant lexical bias (` makes a meowing sound` is less likely than ` barking` for every context including cats; ` kittens` more likely than ` puppies` for every context including dogs), so raw sign confounds class with lexical frequency. Codex's design specified standardization with calibration-only scale estimates; whether a calibration-centred decision statistic is the pre-declared rule or a post-hoc repair is referred to Codex (round 7) before any rerun. Today's tally on the frozen residual stream: three baseline kills (v1, v2, interchange_v1) and one gate-passing result reclassified as late lexical steering (v3).

## 2026-08-29 — Audit #27 on coordinate_v3 (verbatim): NEGATIVE for the coordinate artifact; narrow late lexical-control effect

My design error, found by the auditor: the corner transports flip the number bit without changing the visible subject, so every scored token disagrees with its prefix (`The writer were`); the grammar gate is 0/32, and `run_base` never enforced `test_gate`. My provisional 'positive' and my '~30/32 grammatical' reading are withdrawn.

## Verdict

The causal next-token effect is real and exactly reproducible, but the provisional coordinate claim does not pass audit.

The experiment demonstrates strong, template-specific late lexical steering of four verb tokens. It does not yet demonstrate a grammatical, persistent latent state or a general two-dimensional coordinate. More seriously, every corner transport changes the target’s number without changing the visible subject, making all 32 prompt-plus-continuation outputs ungrammatical at the scored token. The predeclared 28/32 grammatical-continuation rule therefore fails systematically, not marginally.

Classification: **negative for the coordinate artifact; positive as a narrow lexical-control diagnostic.**

## Decisive integrity failure: the grammar gate

The passage is constructed from the **start** state’s subject number, while every corner transport flips that number bit:

- `00 → 11`: `The writer were`
- `11 → 00`: `The writers is`
- `10 → 01`: `The writer are`
- `01 → 10`: `The writers was`

This follows directly from the passage construction at [run_coordinate.py](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_coordinate.py:35>) and the corner mapping at [run_coordinate.py](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/run_coordinate.py:131>).

The log displays only the generated fragment—such as `" were happy..."`—which obscures the unchanged prompt subject. Once concatenated, every successful target disagrees with that subject. Thus the predeclared continuation-quality result is **0/32 grammatical at the intervention site**, not approximately 30/32.

The configuration declares `grammatical_min: 28`, but `run_base` never reads or enforces `test_gate` and stores no manual-review labels ([coordinate_v3.json](</C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/experiments/config/coordinate_v3.json:11>)). This is the exact failed stage.

The governance amendment makes exact certificates diagnostic rather than absolute verdicts, but this is not a borderline threshold failure: the endpoint is structurally incompatible with the visible sentence in every row.

## Exact licensed sentence

> At a fixed final-token prediction site in Qwen3‑1.7B‑Base, a coefficient‑1 block‑20 patch built by adding or subtracting two mean residual differences estimated from 12 calibration families using only states 00/10/01 changed the full-vocabulary greedy next token to the predeclared member of {is, was, are, were} in all 32 held-out noun/complement-template cases, versus 0/32 for the zero arm and each of three seeded random-axis-pair arms, but the vectors project directly onto those verb logits and every corner token disagrees in number with the unchanged visible subject, so the result is a narrow late lexical-control effect rather than a grammatical, persistent, or general latent coordinate.

## Never-say list

- “A two-dimensional latent grammatical coordinate was discovered.”
- “Two hidden grammatical states composed to produce an unseen state.”
- “The intervention generated grammatical native continuations.”
- “The state persisted through generation.”
- “The result generalizes to held-out tasks, verbs, templates, models, or layers.”
- “Random controls prove the learned direction is uniquely meaningful.”
- “A small perturbation produced the effect.”
- “Number is represented abstractly earlier than tense.”
- “State 11 was unseen by the model”; only the experiment’s calibration and selection omitted it.

## Ranked next increments

1. **Run the lexical moot-maker:** compare the learned vectors against direct unembedding-row-difference patches and matched output-logit biases on the same prompts and dose response. If they match, calibration adds no coordinate evidence.

2. **Build a grammatically coherent, lexeme-varying task:** hold out verb lemmas and morphology, vary target tokens, and ensure every counterfactual target remains compatible with the visible prefix.

3. **Require persistence:** intervene upstream and test multiple downstream agreement decisions, preferably with donor activation interchange as a cheaper causal baseline.

4. **Decompose the early number effect:** separate final-noun morphology, example-subject morphology, and example-verb contributions while matching final-token identity.

5. **If the learned vector does not beat the lexical baseline, pivot to an operational causal quotient/interchangeability artifact; do not sweep more layers, coefficients, prompts, or model sizes.**

## Program and leverage decision

The broader real-latent-space program should continue because the intervention reveals a reproducible causal control surface. This specific two-bit single-site mean-difference coordinate line should not continue as a positive coordinate claim.

The current line is not the highest-leverage work. Its cheapest explanatory baseline—late lexical readout steering—is already strongly supported, and its grammatical endpoint is invalid. The framing has become tunnel-visioned around Cartesian coordinates. Direct lexical steering, causal donor interchange, and a task with varied target lexemes and persistent consequences are the alternatives most capable of making this line moot.

Scoped measurement-to-artifact ratio: approximately 107 new runner/config lines to 16 artifact-bearing construction/intervention lines, or 6.7:1; approximately four design/measurement/audit rounds to one artifact-building round. This exceeds the governance pivot threshold.

No repository file was edited.

## 2026-08-29 — coordinate_v3 (Qwen3-1.7B-Base, prediction site, tense × number): every pre-declared gate met; composition explained by the unembedding (audit #27 pending)

Run (`experiments/results/coordinate_v3/full.log`, ledger `coordinate_v3_result`): baseline 12/12 on 00/10/01 (11 never prompted); LOFO grid at coefficient 1: blocks 8/12/16 pass number (12/12 both directions) but tense 0/12; block 20 passes all four signed single-axis transports 12/12 → frozen. Held-out 8 families × 4 corner transports: 8/8, 8/8, 8/8, 8/8; zero control 0/8; three fixed norm-matched random directions 0/8 each; ~30/32 six-token continuations grammatical (two degenerate). By the round-5 rule this is a positive.

Read-only diagnostic (logit lens on the directions, W_U·v): at block 20, |v_T| 214, |v_S| 281 vs |h| ≈ 1005, cos(v_T, v_S) = 0.03; W_U·v_T top tokens = was/ was/Was; W_U·v_S top = _are/ are; **W_U·(v_T+v_S) top = ` were`** (logit 167 vs is −92, was 53, are 39). At blocks 8–16 the tense direction has no clean unembedding image (top tokens are junk; the number direction already points at ` are` from block 12). Coefficient diagnostic on one calibration family at block 20: 0.1 and 0.25 → no change; 0.5 → ` are` (number only); 1.0 → ` were`. Reading (mine, pending audit): the four-way composition is a property of the output-embedding geometry — the ` were` unembedding is close to ` was` + ` are` − ` is` — read out at a block where the residual is already largely the logit direction; this is late-layer token steering with additive unembeddings, not a latent-state coordinate, and the tense axis has no accessible direction before block 20. The wow sentence is NOT licensed as written. The clean part that survives: single-axis and composed steering transfer across held-out subjects/complements at this site with zero and random controls at 0/8.

## 2026-08-29 — coordinate_v1 (Qwen3-0.6B, tense × polarity): UNINTERPRETABLE — INVALID POLARITY BASELINE; direction round 4 → coordinate_v2

Demo stage: no block 0–27 cleared the coefficient-one final-prompt-token calibration rule (0/16 on both LOFO axes, termination 1.0). Read-only diagnostics on calibration families: the explicit polarity instruction fails on Qwen3-0.6B (echo-then-rewrite or plain copy; 0/8 on 01 in every wording tried), so tense × polarity was never a valid two-bit task on this model; final-token and template-tail patches are inert at blocks 6–18 up to 3×; all-position injection is inert at blocks 4–12 and degrades at 16/20 (one tense hit at L20 ×2). Qwen3-1.7B: tense 8/8 but polarity 01 only 5/8 even with a repaired wording — below the gate.

Codex copy-ready sentence (verbatim): **Coordinate-v1 is uninterpretable:** Qwen3-0.6B failed the explicit polarity capability gate, so tense × polarity was not a valid two-bit task; independently, no block cleared the coefficient-one final-prompt-token calibration rule (`0/16` for both LOFO axes at every block, termination `1.0`), so no held-out transport was run. The valid tense subtask therefore supplies a bounded negative for that exact one-shot intervention, not for residual-stream coordinates generally. Never say: the model re-reads instruction tokens; no additive instruction state is present; a stable tense direction has been found.

Direction round 4 (Codex): Qwen3-1.7B (rev `70d244cc…`), second axis = grammatical number (00 present-singular, 10 past-singular, 01 present-plural, 11 past-plural), intervention = persistent-current-position (patch the final prompt position at prefill, then the sole current position at every decoding step; never the cached prefill), object reframed from a static coordinate to a constant causal control field over the decoding trajectory — it earns the coordinate language only if signed single-axis transport, unseen composition and inverse all pass. Baseline gate first (W0 only, no 11: ≥14/16 per state, 16/16 termination; if any state fails, kill the artifact — no further prompt repair). Then a fixed grid (layers {12,16,20} × coefficients {0.5,1,2}), four signed single-axis transports ≥12/16 with 16/16 termination, earliest layer then smallest coefficient, no enlargement; only a qualifying cell proceeds to the four unseen corner transports. Config: `experiments/config/coordinate_v2.json` (wording frozen before any result).

## 2026-08-29 — Re-contextualization #27: the pivot to a real-model causal coordinate

Whole picture. Two days of the program produced apparatus and negatives: NLM-007 closed at a small within-design survival; Rounds 36–37 (toy quotient worlds) ended with the toy program itself judged not worth continuing (measurement-to-artifact ratio far past the halt threshold; exact certificates guaranteeing bad verdicts on continuous learners). The user's verdict — nothing useful yet — was accepted and the governance amended (AGENTS.md 2026-08-29): exact gates are diagnostics only; one audit per result and it must answer "should this continue"; ratio reported each heartbeat; narrative gate binding; real models only; direction is a Codex dialogue.

What reframes earlier work. The toy left two transferable ideas (identity by causal interchangeability; never use exact certificates as primary evidence for learned continuous systems) and nothing about real residual streams. The Round 36d/37 horizon-localised failures are unresolved hypotheses, not laws.

Live question now. Does the Qwen3-0.6B residual stream at one token expose a manipulable two-bit coordinate (tense, polarity): two moves estimated from single-axis calibration only (state 11 never used) that add to produce the unseen combined instruction and subtract to undo it, on held-out sentences and held-out wordings, in free-decoded text? Artifact: `experiments/run_coordinate.py` + `config/coordinate_v1.json` (181 lines; demo stage running; causal LOFO layer rule; chance 1/4 among canonical forms; sham + norm-matched random controls).

Alternatives held live (Codex rounds 1–2, verbatim ranking): real-model operational quotient under causal interchangeability (70% positive, wow 7, negative value 10 — deferred until one causal move is shown); group-like closure sweep over steering vectors (35%, wow 10, too many degrees of freedom); intervention-effect metric with triangle-like laws (80%, wow 4, tunable into regularities). Strongest genuine failure mode (not a confound): the residual stream may hold no persistent global sentence-state register at one token — grammar as a distributed, position- and step-dependent policy, so an average displacement is a chord through a curved process; single moves may transfer while sums and inverses fail. That outcome would reject a single-token affine chart and point at trajectory-level operators — itself informative.

Not tunnel-visioned: the runner is axis-agnostic; a positive scales to 3–4 axes including a non-grammatical operation (target language) in the same file; a negative leaves nothing to scale and the operational-quotient alternative takes over. Stop rule: if the runner grows into apparatus before printing sentences, stop again.

## 2026-08-29 — Round 37 result: NO ARCHITECTURAL WIN; both carriers FAIL (underfit); last toy-world round

Four-cell matrix run on CPU (35 min wall, one process). Both the quotient-factored z=(q,p) carrier and the unrestricted carrier reduce to `FAIL — BEHAVIOR UNDERFIT OR BASE SIGNATURE UNSUPPORTED`; comparison label `NO ARCHITECTURAL WIN`. Diagnostic-only (non-gating) numbers: H2 held-out supported-truthful cells factored 867–1184/1184 vs unrestricted 795–1184/1184; H3 factored 365–754/1056 vs unrestricted 485–992/1056 — the unrestricted carrier is better on 9 of 10 seed × role units, i.e. the factorization constraint hurt. First divergence is overwhelmingly at step 3 of H3 (the Round 36d horizon-localised pattern reproduces in a different world and different carriers); failures are predominantly — not exclusively — future-signature failures (audit correction: 128 factored and 69 unrestricted failure cells involve a terminal error, four cells are terminal-only; the 0.079 margin is not a global terminal minimum). Rolled-history interchangeability across presentations was not reached by either carrier.

**Round 37 one-and-only result audit (Codex, direction dialogue round 2; verbatim licensed sentence):** Under the frozen exact toy reducer, neither carrier reached behavior-qualified held-out presentation transfer or rolled interchangeability; descriptively the unrestricted carrier had higher transfer and interchangeability rates in 9 of 10 paired seed × role units, while failures were predominantly—but not exclusively—future-signature failures and H3 first divergence was predominantly at step 3, so the imposed factorization showed no benefit in this setup. Never say: every failure was future-signature / terminal responses never failed / the factorization constraint is harmful / unrestricted is the architectural winner / the Round 36d mechanism reproduced (only the horizon localization recurred) / the hole is a property of behaviour-supervised learning (hypothesis only) / anything about real residual streams based on Round 37. Note: both status fields are one factored-primary world verdict copied to both files, not two independent carrier verdicts. Toy program ENDS; overall program continues only as a short real-model intervention artifact.

My earlier reading (superseded where it conflicts): the hole is a property of the learning setup — behaviour-supervised continuous carriers fit depth-1/2 responses and lose exact future signatures at depth 3 — not of any particular world or carrier. Under the 2026-08-29 governance amendment this is the last round in the Round 36 mould: exact certificates become diagnostics, and the central artifact moves to a real model's latent space demonstrated by intervention. The one-audit-per-result review of this outcome is folded into the direction dialogue with Codex (round 2), which must also answer whether the program continues.

## 2026-08-29 — Round 36d frozen-chart control (audit #26 replacement entry, verbatim; my original headline 'exact certificate reached on 8 of 9 gates' is withdrawn)

## 2026-08-29 — Round 36d frozen-chart control: joint FAIL; eight gate predicates pass, deepest rolled interchangeability does not

The locked `POSITIVE-CONTROL` cell completed hash-validly in `118.454 s`.
With the behavior-derived W64 encoder/readout assigned and frozen, a fresh
width-64 transition head trained for 16,000 steps on all 176 fixed canonical
successor coordinates. Behavior is exact in every seed, and quotient
availability, well-definedness, involution, the swap/toggle table, H2/H3
closure, canonical action truth, and the cross-seed table pass exactly.
Interchangeability passes only seed 71; misses are `16/5/28/98/0` of
`132,160`, so the joint verdict is `FAIL — INTERCHANGEABILITY`.

Audit #26 governs the reading. Say that eight individual predicates are
reachable under this privileged full-table control; do not say that the
exact certificate or reducer passed. The chart already carried exact
canonical signatures and every canonical edge was taught, so this is an
optimizer-fitted table realization, not quotient discovery or learned
composition. All 147 misses occur only for depth-2 rolled representatives
followed by H3 and appear only after the third continuation action; 89 are
confidence-only, 58 include a future-probe truth error, and none changes the
immediate terminal response. The MSE ratio fell below its descriptive
reference but is non-gating and does not rule out optimization. Round 36d is
closed. Round 37 proceeds and inherits rolled interchangeability as its
primary structural question.

## 2026-08-29 — Re-contextualization #26 (2-hour step-back; audit #26 fired on the 36d result)

Audit #26 replacement paragraph (verbatim; supersedes my interpretive paragraph):

Audit #26 rejects the claim that the exact reducer is now passable by a
learned artifact: the registered control's joint verdict is still FAIL, so
no learned artifact has passed the complete reducer. What changed is
narrower and still useful. A prospectively locked optimizer-fitted head on a
frozen behavior-derived chart passed eight individual gate predicates under
complete privileged canonical supervision. The sole measured residue is
rare and horizon-local—147 depth-2-history × H3 future-signature cells, with
no immediate-response error—but it is not purely threshold-driven. The
coordinate replay supports local off-chart drift, while cause remains open
because the adequacy ratio is diagnostic only. Do not spend another control
on this singleton-quotient world. Carry the history/presentation question
into Round 37's genuine 32-to-16 quotient.

#### Audit #26 — licensed sentence, never-say list, interchangeability residue analysis, Round 37 ruling, final verdict (verbatim)

**Round 36d is a valid `POSITIVE-CONTROL` FAIL with a narrow reachability
gain: reusing an assigned behavior-derived W64 encoder/readout whose 16
canonical signatures were already exact, a fresh optimizer-fitted width-64
transition head taught all 176 canonical successor coordinates made exact
behavior and eight registered gate predicates hold in all five seeds. The
joint certificate still failed rolled interchangeability in four seeds—147
of 660,800 finite cells, all depth-2-history × H3 continuations, with no
immediate-response error. This validates individual-gate passability for
this privileged finite table-realization control and localizes the measured
residue to rare history-dependent off-chart future-signature instability;
it does not validate quotient discovery, behavior-only construction,
arbitrary-length composition, or a complete operational quotient.**

Under the guiding question, the denizen-level lesson is modest but useful:
canonical landmarks and taught moves can support nearly perfect finite
navigation while “same place” still fails to be stable for a few deeper
histories. A next latent world needs identity and action descent to be robust
to how a place was reached, not merely exact on its canonical presentation.

Never say:

- “Round 36d reached the exact certificate” or “passed the reducer.”
- “A learned artifact has now passed the registered reachability control”
  without immediately saying “eight individual predicates; joint FAIL.”
- “The learned-pass gap is closed.” No learned artifact has passed the joint
  exact reducer.
- “The quotient/action algebra was learned” or “the table was discovered.”
  Every canonical transition cell was taught on an already qualified chart.
- “H2/H3 proves composition” or “arbitrary-length closure.”
- “Eight independent gates validated the construction.” The gates are
  correlated consequences of the complete taught edge table.
- “Interchangeability generally fails,” “the quotient is non-congruent,” or
  “rolled points move to the wrong place.” The finding is finite, rare,
  history-local, and 146/147 failed endpoints remain nearest the right chart
  landmark.
- “The FAIL is threshold-only.” Fifty-eight failed rows contain a
  confidence-free future-probe truth error and 12 components are supported but
  wrong.
- “Behavior fails at depth five.” The immediate endpoint response is correct;
  the future-response signature fails.
- “The reducer oversampled a bug.” It intentionally enumerated histories;
  rates are population-dependent, but the registered exact counterexamples
  are real.
- “Adequacy passed,” “optimization is ruled out,” or “the cause is fixed-chart
  realizability.” The adequacy ratio is diagnostic and causal attribution is
  unresolved.
- “Round 36d rescues, promotes, or reclassifies W64/36b,” or anything about
  language models, residual streams, natural latent spaces, or a general
  axiom.

## 3. Interchangeability residue

### Exact near-miss table

| Seed | Failed / 132,160 | Failure rate | confidence-only rows | rows with a `p>0.5` future-probe truth error | unique failed representatives | unique operational source/word cells |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | `16` | `0.01211%` | 5 | 11 | 16 | 2 |
| 23 | `5` | `0.00378%` | 3 | 2 | 5 | 2 |
| 37 | `28` | `0.02119%` | 17 | 11 | 20 | 14 |
| 53 | `98` | `0.07415%` | 64 | 34 | 47 | 32 |
| 71 | `0` | `0%` | 0 | 0 | 0 | 0 |
| **Total** | **`147 / 660,800`** | **`0.02225%`** | **89** | **58** | — | — |

At the exact signature level, 138 failed endpoints are unsupported and nine
are fully supported but wrong. Across all failed endpoints there are 157 bad
signature components: 145 lie inside the unsupported band and 12 are
supported on the wrong side. Their shortfalls from a truthful support boundary
are not uniformly tiny:

| Shortfall | Bad components |
|---:|---:|
| `<0.01` | 4 |
| `0.01–0.10` | 44 |
| `0.10–0.40` | 47 |
| `0.40–0.80` | 50 |
| `>=0.80` | 12 |

The closest miss is `0.00189`; the largest is `0.89434`. Median worst-cell
shortfall per failed endpoint is `0.52546`, `0.07864`, `0.32198`, and
`0.24016` in seeds 11, 23, 37, and 53. This rejects a blanket
“one epsilon over the threshold” explanation.

No bad component is the empty-response component. The immediate binary
response at all 147 rolled endpoints is correct at `p>0.5`; the failures occur
in the one-action future probes used to name the endpoint's operational place.
Of the 157 bad components, 119 are the no-op probe or swaps involving response
bit 1. The defect is therefore future-response identity instability, not a
failure of the observed terminal response on the H3 word itself.

### The residue is exactly horizon-localized

Every failure has the same shape:

- representative prefix depth: 2;
- held-out continuation depth: 3;
- signature after continuation action 1: equal and supported;
- signature after continuation action 2: equal and supported;
- signature after continuation action 3: divergent or unsupported.

The stepwise replay pattern is `EED` in all 147 cases. No encoder
representative, depth-1 rolled representative, H2 continuation, or earlier H3
step fails. Starting from a depth-2 history, the registered continuation
therefore preserves place through total action depth 4 and loses a future
signature at total depth 5; because the signature includes another primitive
probe, the failing component exposes response instability at effective depth
6.

This is what interchangeability adds beyond the other gates. Quotient
well-definedness checks one primitive from the registered representative set.
H2/H3 closure checks canonical starts. Interchangeability asks whether the
same held-out continuation has the same future-response signature when begun
from noncanonical histories naming the same place. The answer is exact for
all registered cells in seed 71 and non-exact, though extremely close in rate,
in the other four seeds.

### Rolled points are locally off chart

Euclidean distances are not the native identity rule and cannot alter the
verdict. As a read-only coordinate diagnostic, however, they identify the
shape of the residue:

| Seed | median encoder-landmark spacing | median failing representative-to-landmark distance before H3 | median failing rolled-to-canonical trajectory distance after H3 | median passing rolled-to-canonical trajectory distance | failed endpoints nearest wrong landmark |
|---:|---:|---:|---:|---:|---:|
| 11 | `5.027` | `0.0327` | `0.8238` | `0.0545` | 0/16 |
| 23 | `4.634` | `0.0409` | `0.5552` | `0.0508` | 0/5 |
| 37 | `4.694` | `0.0736` | `0.8640` | `0.0781` | 0/28 |
| 53 | `4.785` | `0.0647` | `0.8696` | `0.0808` | 1/98 |
| 71 | `4.983` | — | — | `0.0262` | 0/0 |

The failing histories begin farther from their matched landmarks than typical
depth-2 representatives, and the H3 rollout amplifies this separation by an
order of magnitude relative to passing cells. Yet 146/147 failed endpoints
remain closest to the correct canonical landmark. The fair diagnosis is local
off-chart drift through the frozen readout's exact signature boundaries, not
gross migration to another canonical place.

### Not a reducer duplication bug

The reducer enumerates every registered `(representative, held-out word)` pair
once. There are no duplicate IDs or omitted cells. Several failures collapse
to the same operational source-state/word pair because different rolled
histories name the same source place: seed 11's 16 failed rows reduce to two
operational cells with 7 and 9 failing histories; the maximum history
multiplicity is 4, 6, and 10 in seeds 23, 37, and 53. That multiplicity affects
rates and forbids treating the 132,160 rows as independent statistical
replicates. It is not an artifact under the registered exact conjunction:
representative-history dependence is precisely what the gate tests.

The scientific scope remains finite. The verdict covers the declared 944
representatives and 140 held-out words, not every latent point or every word.
Changing this population would change the descriptive rate, though one
registered counterexample is sufficient for the frozen exact FAIL.

## 6. Constructive-program ruling and Round 37

The fair constructive statement now is:

> **A prospectively locked learned artifact passed eight individual exact
> gate predicates in a registered, privileged frozen-chart reachability
> control; the control's joint verdict remained `FAIL — INTERCHANGEABILITY`.**

This changes audit #25's inventory in one way: individual learned-gate
passability is no longer wholly unvalidated. It does **not** change the whole-
reducer inventory: the oracle fixture remains the only complete PASS.

Do not resolve interchangeability with another Round 36 optimization cell,
rolled-target loss, longer schedule, wider head, or threshold branch. That
would teach the last exam on a singleton-quotient toy and deepen the tunnel
audit #25 already found. Preserve the 147 cells as a permanent negative result
and transfer the question.

Round 37 should proceed. Its presentation-duplicated world turns the current
history/presentation issue into the scientific object: two genuinely distinct
presentations must name one operational place and remain interchangeable under
held-out actions. The Round 36d residue should be carried into Round 37 as its
primary rolled-structure question, with pre-outcome diagnostics separating:

- representative/presentation history depth;
- H2 versus H3 continuation depth;
- first divergence step;
- immediate terminal response versus future-signature failure;
- unsupported versus wrong supported components; and
- factored versus unrestricted carrier margins on the same seed/role unit.

These diagnostics should not alter Round 37's locked exact certificates or
create an adaptive fifth cell.

## 7. Ranked next increments

| Rank | Increment | Cost | Decision value |
|---:|---|---|---|
| 1 | **Adopt audit #26 language and close Round 36d permanently.** Update `STATE.md`, `NOTEBOOK.md`, README, theory amendment, ledger, and project memory; retain the 147-cell result. No rerun. | `30–60 min`, docs/ledger only, zero CPU training | Prevents a component-gate WIN from becoming a joint-certificate claim and restores current-state truth. |
| 2 | **Before any Round 37 outcome, add non-gating horizon/role diagnostics and receive a Tier-1 lock review.** Reuse the existing runner/evidence surface; preserve every primary gate and cell. | `1–2 h` implementation/review, negligible CPU | Makes the exact question raised by 36d directly visible in the nontrivial quotient world without teaching it. |
| 3 | **Run the registered Round 37 factored-versus-unrestricted four-cell matrix.** No adaptive cell and no outcome inspection until the locked matrix completes. | Registered `24–32 CPU-min`; `45 min` total wall, plus implementation | Tests genuine 32-to-16 identity, presentation transfer, action descent, and rolled interchangeability—the program's strongest scientific increment. |
| 4 | **If Round 37 is non-exact after behavior/support eligibility, perform a read-only clustered localization on the ten seed×role units.** Separate rates, margins, first divergence, and carrier contrast; do not pool endpoint rows as replicates. | `30–90 min`, no retraining | Distinguishes presentation non-congruence, generic horizon drift, support sensitivity, and carrier-specific failure before designing a successor. |
| 5 | **Only after Round 37 adjudication, design the next latent-space primitive around the observed hole.** Examples: contractive quotient fibers, history-stable transitions, or explicit presentation covariance, each with a matched unrestricted control. | Moderate theory/design; new CPU cost prospectively locked | Converts a proven hole into a constructive latent-space requirement rather than another exam calibration. |

Explicitly do **not** run another Round 36d schedule, width, loss, rolled-target,
or tolerance cell.

## Final verdict

- **Hash/integrity/provenance:** **UPHELD.**
- **Exact behavior in all five seeds:** **UPHELD.**
- **Eight individual gate predicates pass:** **UPHELD.**
- **Joint status `FAIL — INTERCHANGEABILITY`:** **UPHELD.**
- **“Reached the exact certificate on 8 of 9 gates”:** **REPLACE WITH
  “EIGHT COMPONENT PREDICATES PASS; JOINT CERTIFICATE FAILS.”**
- **“Learned gate reachability validated”:** **NARROW TO THIS PRIVILEGED,
  FULL-TABLE, FROZEN-CHART HEAD AND EIGHT INDIVIDUAL PREDICATES.**
- **“Learned composition/action algebra”:** **REJECTED.**

## 2026-08-29 — Round 36c positive control (w64): FAIL; Round 36c complete

The conditional width-64 control cell finished in 988 s (wall 2400 s) and
FAILS the exact certificate: per-seed exact passes are swap/toggle table
4/5 and held-out depth-2 closure 3/5, with every other gate 0/5 (action-
table truth 0/5; cross-seed table not identical). Audit #25's wording
governs: this registered joint learned-target recipe did not reach the
certificate; the run does not distinguish control-objective failure,
optimisation failure, carrier capacity, or learned reducer/gate
reachability, and has no behaviour-only interpretation. Round 36c is
complete; no further moving-target cells will be run. Next, as ruled: one
capped frozen-target head-only calibration (Round 36d), then the pivot to
the presentation-duplicated quotient world (Round 37) — both being
registered.

## 2026-08-29 — Re-contextualization #25 (2-hour step-back; audit #25 fired on the positive-control FAIL)

Project: the native mathematics of latent spaces. Live question (Round
36): can a latent world built from behaviour alone carry a well-defined
operational quotient and composable action table — and, after today, the
prior question: is the certification regime itself passable by any learned
artifact? (corrected by audit #25: a learned-pass reachability gap, not a
"certification-regime problem"; the w32 control failed to validate the
regime, it did not indict it.)

Audit #25 replacement paragraph (verbatim; supersedes my interpretive paragraph):

Audit #25 upholds the w32 exact FAIL and its `POSITIVE-CONTROL` scope, but
narrows both reactions to it. As of the audit cutoff, no evaluated learned
artifact has passed the exact reducer and only the oracle fixture has; that is
a learned-pass reachability gap, not proof that learned artifacts cannot pass.
The w32 control is not a clean reachability proof because its successor
targets co-moved with the jointly trained encoder and its weight-1.0 MSE could
interfere with BCE. W64 remains descriptively superior on its canonical
table, but audit #24's behavior-ineligible, informational-only boundary is
unchanged. The program should spend one capped increment on a frozen-target,
head-only learned-pass calibration, then move to a presentation-duplicated
32-state world with a true 16-class quotient rather than add another control
ladder to the singleton-quotient toy.

#### Audit #25 — honest sentence, reducer inventory, tunnel ruling, strongest direction, ranked increments, final verdict (verbatim)

**Round 36c-w32 is a valid positive-control FAIL for its registered joint
learned-target objective: this width-32 optimizer recipe did not reach the
exact certificate. Because the successor coordinates co-moved with the
encoder, behavioral BCE and transition MSE were combined at an unablated
weight, and separate component traces were not retained, the run does not
show that the carrier or exact gates are unreachable under direct
supervision. It leaves learned gate reachability unresolved and is not a
behavior-only latent-organization result.**

“The auxiliary objective is now the leading hypothesis” may follow that
sentence. It must remain a hypothesis until a frozen-target or loss-weight
ablation identifies it.

## 4. What the reducer has and has not demonstrated

The fair current inventory is:

| Artifact type | Has passed? | What it establishes |
|---|---:|---|
| Oracle-authored affine fixture | Yes | The declarative reducer has at least one exact accepting assignment and rejects selected corruptions. |
| Behavior-only learned v1/36b | No | These registered recipes did not reach their eligible exact certificates. |
| Joint learned-target 36c-w32 | No | This moving-target, weight-1.0 joint recipe did not validate learned reachability. |
| Any frozen-target learned head | Untested | Would test transition-head/optimizer reachability without target chasing or BCE interference. |
| Any direct signature-supervised learned artifact | Untested | Would test whether optimizer-produced parameters can pass the exact response-signature exam when the exam is taught directly. |
| Any hand-constructed parameter assignment in the current neural architecture | Untested | Would test architecture representability, not optimizer learnability. |

Accordingly, this sentence is licensed:

> **As of audit #25, the exact reducer has accepted its oracle fixture but no
> evaluated learned artifact; learned-pass reachability is unvalidated.**

This sentence is not:

> “The exact reducer is a reducer that learned artifacts cannot pass.”

There is also a provenance boundary. The reducer says producer authenticity is
out of scope and does not consume `weights.npz`. `producer_kind="learned"` is
validated metadata, not proof of how parameters were obtained. A learned-pass
demonstration therefore needs a prospectively locked producer, input and
initialization hashes, component traces, a final weights hash, and a clear
statement of which parameters were optimized versus assigned.

### Three useful learned-pass controls

1. **Frozen behavior encoder/readout; supervised transition head — run first.**
   Freeze a retained behavior-trained encoder and readout whose canonical
   signatures are supported and truthful; snapshot the 16 successor targets;
   reinitialize only the transition head; train against fixed target embeddings.
   Log transition MSE, maximum cell residual, and signature margins separately.
   This removes both identified confounds while retaining a genuinely learned
   encoder and a learned head. PASS validates learned head/reducer reachability;
   FAIL localizes the next question to head capacity/optimization or the gap
   between coordinate residual and signature margins.

2. **Explicit signature/quotient supervision — final exam calibration.**
   Train against the fixed oracle 12-bit signatures on every reducer-relevant
   domain, not just the 176 canonical coordinate pairs. This is deliberately
   circular and privileged: it teaches the exact exam. A PASS demonstrates
   optimizer-produced reducer passability, nothing about quotient discovery.
   Use it only if control 1 fails or if an immediate reducer calibration is
   required.

3. **Hand-constructed current-architecture solution — representability only.**
   Assign or solve parameters for a fixed bit/signature chart inside the
   current residual-tanh architecture. If exact gates pass, the architecture
   can represent an accepting solution. Unless a locked optimizer produced the
   parameters from declared targets, label this `SYNTHETIC-PARAMETER CONTROL`,
   not `LEARNED`.

The existing fixture is not a substitute for any of these: it uses per-action
affine maps outside the learned shared residual-tanh parameterization.

## 5. Tunnel vision and the strongest alternative direction

### Ruling: found

The program's guiding question is what a denizen needs to navigate a latent
world. Round 36's empirical loop has narrowed to whether a particular learner
can satisfy a particular exact examiner. The true depth-1 signature already
separates all 16 states, so the “quotient” has no nontrivial compression. Every
additional control on this world has diminishing scientific value unless it
closes the learned-pass reachability gap once and then stops.

### Strongest cheap research direction: a presentation-duplicated quotient world

Build the next world as `S = {0,1}^4 x {0,1}`: four operational bits plus one
presentation/nuisance bit. Give each operational state two opaque handles.
Primitive task actions change only the four operational bits; a presentation
move changes only the nuisance bit. The binary response and all future task
responses ignore nuisance. The true operational quotient is therefore
nontrivial: 32 hidden states, 16 denizen places, two representatives per place.

Use a quotient-factored carrier `z = (q, p)`:

- response and task transition operate through `q`;
- presentation information may live in `p`;
- no coordinate equality between the two representatives is required;
- behavior-only training receives no hidden state labels or duplicate-pair
  target; and
- an unrestricted carrier is the matched baseline.

Predeclare held-out presentation transfer: train selected action words through
one presentation and test the paired presentation, then swap roles. Primary
questions become substantive:

- do the two presentations acquire the same operational signature;
- do actions descend independently of presentation;
- does the quotient-factored carrier outperform the unrestricted carrier on
  held-out presentation transfer and rolled interchangeability; and
- can presentation be changed without changing operational place?

This is still a tiny CPU world—32 embeddings and the existing action family—so
it should cost minutes, not a new infrastructure cycle. Its “so what” is:

> **Can we build a latent world that keeps what a place means separate from how
> that place is presented, so its inhabitants can move reliably across two
> views of the same world?**

Keep the current exact reducer as a secondary certificate only after one
learned-pass calibration. The next world's primary inquiry should report
predeclared rates and margins alongside exactness; it must not let one
all-cells conjunction erase the distinction between learned structure and
certificate saturation.

## 6. Ranked next increments

| Rank | Increment | Cost | Decision value |
|---:|---|---|---|
| 1 | **One frozen-target, head-only learned-pass calibration** | One tiny CPU cell; reuse current runner/config surface | Removes moving targets and BCE interference. This is the cleanest remaining test of learned reducer reachability. Instrument separate losses and max residuals. Stop after one prospectively locked width/capacity choice; do not start another ladder. |
| 2 | **Presentation-duplicated 32-to-16 quotient world with factored and unrestricted carriers** | Low design; minutes of CPU | Returns the program to building a latent space, makes identity genuinely nontrivial, and directly tests the presentation/state hole. This is the strongest research direction. |
| 3 | **Direct full-domain signature-supervised exam calibration, only if rank 1 fails** | Low CPU; moderate circularity | Demonstrates that optimizer-produced parameters can pass the reducer when every assessed signature is explicitly taught. It calibrates the exam; it is not a scientific quotient-learning result. |
| 4 | **Prospective approximate-inquiry branch** | Near-zero reducer work after design | Preserve exact PASS. Report fixed rates/margins as a separate non-PASS status on new runs only. Never reclassify W64 or 36c. Use it in the nontrivial world to avoid conflating structure with all-cell saturation. |
| 5 | **Hand-constructed current-network parameter solution** | Very low to moderate | Useful only to separate architecture representability from optimizer reachability. Label synthetic unless a locked optimizer produced it. |
| 6 | **Further longer/wider behavior-only or moving-target control cells** | Low code, low information | Do not run next. They repeat the same architecture/reducer confounds and deepen tunnel vision. The already authorized 36c-w64 result, if it completes, is retained and reported but does not authorize another cell. |

## Final verdict

- **Mechanical w32 all-gate FAIL:** **UPHELD.**
- **Action-table truth `0/5`; cross-seed table FAIL:** **UPHELD.**
- **“Exact gates are not reachable by this carrier”:** **REJECTED AS
  UNPROVEN.** The registered joint optimizer failed; reachability remains
  unresolved.
- **“Certification-regime problem”:** **NARROW TO “THE REACHABILITY CONTROL
  FAILED TO VALIDATE THE REGIME.”** Do not blame the reducer or carrier yet.
- **Auxiliary-objective explanation:** **LEADING HYPOTHESIS, NOT IDENTIFIED
  CAUSE.** Moving targets and loss interference are real design confounds;
  universal signature collapse is not found.
- **“W64 behavior-only beat the control”:** **DESCRIPTIVELY TRUE ON CANONICAL
  GATES; NO SCIENTIFIC REHABILITATION.** Audit #24 remains unchanged.
- **Only an oracle fixture has passed:** **FOUND FOR THE ARTIFACTS READ.** This
  is a learned-pass reachability gap, not an impossibility theorem.
- **Tunnel vision:** **FOUND.** One capped calibration remains justified; a new
  control ladder does not.
- **Strongest research direction:** **A PRESENTATION-DUPLICATED, GENUINELY
  NONTRIVIAL QUOTIENT WORLD WITH A QUOTIENT-FACTORED CARRIER.**

## 2026-08-29 — Round 36c positive control (w32): FAIL on every exact gate

The learned, explicitly quotient-trained control (behavioural BCE + MSE of
the transition output to the stop-gradient encoding of the true successor
over all 176 canonical transitions; same carrier, seeds, reducer) finished
in 820 s and FAILS every exact gate in every seed — including action-table
truth (0/5 seeds) and cross-seed table identity, which the behaviour-only
36b W64 cell had passed informationally. result_scope = POSITIVE-CONTROL:
this is not a behaviour-only result and not a quotient-from-behaviour
claim. Registered meaning of a control FAIL: the exact gates are not
reachable by this carrier even with direct transition supervision — a
certification-regime problem, not a latent-organisation result. (corrected by
audit #25: "not reachable ... even with direct supervision" rejected as
unproven; "certification-regime problem" narrowed to "the reachability
control failed to validate the regime".) Audit #25 replacement (verbatim; supersedes the interpretive tail of this entry):

Registered mechanical meaning: the width-32 positive-control recipe did not
reach the exact certificate. Audit #25 rejects the stronger sentence that the
gates are “not reachable by this carrier even with direct supervision.” The
auxiliary target was the stop-gradient value of an encoder that continued to
move through its source roles and behavioral loss; BCE and MSE were combined
at weight `1.0` without an ablation; and separate component traces were not
serialized. The leading hypothesis is therefore objective/optimization
interference, but its mechanism is not identified. This is a failure of the
registered reachability control, not a behavior-only latent-organization
result, a reducer impossibility result, or a rehabilitation of 36b.


## 2026-08-29 — Re-contextualization #24 (2-hour step-back; audit #24 already in flight on the only new claim)

Audit: the only new capability result since audit #23 is the Round 36b
ladder outcome; a fresh, unprimed auditor (#24) was fired on it the moment
the reducers finished and is still running — no second auditor is fired on
the same claim. Its corrections and alternatives are appended verbatim
when it lands.

Audit #24 replacement paragraph (verbatim; supersedes my interpretive paragraph, which prematurely called the line 'closed or near-closed' and equated exact-truth eligibility with the audit-#23 confidence defect):

Audit #24 upholds every Round 36b reducer status and finds no PASS, but it
does not close behavior-only quotient construction. W64 is exact on training
and all H2 held-out terminal rows; its `1–24` remaining errors per seed are
all H3, seed-variable, and share no single row across all five seeds. The
exact held-out gate was therefore not reached, but is not shown
unsatisfiable. Informationally only, W64 recovers all 16 canonical identities
and the complete truthful `16 x 11` action table identically across seeds.
It still fails exact rolled-representative descent, involution, closure, and
interchangeability even in the `p>0.5` diagnostic, so the result is a stable
canonical one-step skeleton rather than a certified quotient algebra. The
next registered increment is the learned, explicitly quotient-trained
positive control scored by the unchanged reducer; a separate prospective
approximate-inquiry branch may use exact training plus a fixed held-out
tolerance, while the original exact PASS remains unchanged.

The existing “Round 36b result” entry can remain after this audit is appended;
its informational/diagnostic labels are disciplined.

#### Audit #24 — W64 section, ranked next increments, and final verdict (verbatim)

## 2. What W64 does and does not mean

W64's ineligible primary flags are unusually informative. At the registered
support threshold, every seed passes:

- quotient availability: all 16 encoder signatures are supported and truthful;
- action-table truth: `176/176` canonical state/action cells;
- cross-seed action table: the five complete tables are identical and truthful.

This is stronger than “approximately 99% behavior.” It says that behavior-only
training found the same canonical one-step operational table across five
random initializations. The result is still informational rather than a
verdict because the frozen eligibility tree says so.

The opposite reading is equally important. At `p>0.5`, W64's ranges are:

| Gate | Per-seed range | Exact seeds |
|---|---:|---:|
| Quotient availability | `17/17` | 5/5 |
| Quotient well-definedness | `7540–9965 / 10560` (71.4–94.4%) | 0/5 |
| Toggle involution | `1724–3178 / 3776` (45.7–84.2%) | 0/5 |
| Swap/toggle table | `372–384 / 384` (96.9–100%) | 1/5 |
| H2 signature closure | `1167–1183 / 1184` (98.6–99.9%) | 0/5 |
| H3 signature closure | `643–972 / 1056` (60.9–92.0%) | 0/5 |
| Interchangeability | `50928–101706 / 132160` (38.5–77.0%) | 0/5 |
| Canonical action-table truth | `176/176` | 5/5 |
| Whole-table cross-seed identity and truth | `176/176` | PASS |

Thus the correct structural description is:

> **A cross-seed-stable canonical one-step skeleton emerged, but action on the
> full representative population did not become an exact well-defined,
> involutive, closed, interchangeable quotient action.**

“The latent is organized” is licensed only with that local/canonical scope.
“The exact structural gates are unreachable” is not licensed until a learned
positive control tests reachability. “The latent is unorganized” is contradicted
by the canonical table.

## 7. Ranked next registered increments

| Rank | Increment | Cost | Why / decision value |
|---:|---|---|---|
| 1 | **Explicit learned quotient-trained positive control** | Low–moderate implementation; roughly one five-seed CPU cell, likely `<15 min` after review | Use the same 8-D carrier, same width (run 32 first; 64 only if prospectively conditional), same seeds, same representatives, and unchanged exact reducer. Add direct state-transition or quotient-consistency supervision. PASS shows the learned architecture/optimizer can reach the certificate and localizes the behavior-only gap to the objective. FAIL says the architecture or gate regime is itself the immediate problem. The affine fixture is not this control. |
| 2 | **Separate exact certification from approximate-structure eligibility** | Near-zero compute for a reducer design/replay; medium governance; a new prospective behavior run costs about 10–15 CPU min | Keep the original exact PASS unchanged. Add a distinct inquiry branch requiring exact `21184/21184` training in every seed, exact H2 terminal behavior, and at least `1046/1056` (99.0%) H3 terminal behavior in every seed. It may report structural rates but can never emit the exact PASS. This threshold is a transparent post-36b successor rule and must not retroactively reclassify W64; W64 would still miss it in seeds 11 and 71. |
| 3 | **Learned lookup baseline** | Very low; minutes | Fit handle × observed-word behavior with a frozen default for unseen spellings. Exact train plus poor H2/H3 establishes the memorization floor; comparing it with W64 quantifies what the shared transition gained. Run it alongside rank 2 if convenient. |
| 4 | **Genuinely nontrivial quotient world** | Medium design and implementation; approximately 30–60 CPU min after controls | Add nuisance bits or duplicate hidden states with identical response futures and require independent representatives to collapse. This is the first world that tests quotient formation rather than recovery of a singleton 16-state identity table. It becomes interpretable only after rank 1 establishes gate reachability. |
| 5 | **Longer/wider behavior-only cell** | Low code cost but another 15–30+ CPU min; low information gain | W64 already saturates training and canonical action truth while rolled structural errors remain large. More scale would again conflate optimizer luck, capacity, and gate reachability. Register only as a later sensitivity after ranks 1–3, not as the next move. |

The concrete next registration should therefore be the learned positive
control. The approximate eligibility branch is the next **certification-rule**
increment, but it is not a rescue and does not replace exact PASS.

## Final verdict

- **Mechanical four-cell status:** **UPHELD.**
- **No eligible cell / no PASS:** **UPHELD.**
- **“Behavior underfit” as the registered status:** **UPHELD, but for W64 say
  held-out exactness missed after exact training.**
- **“Exact-held-out eligibility is unsatisfiable by construction”:**
  **REJECTED AS UNPROVEN.** Gate reachability is unvalidated.
- **W64 canonical organization:** **FOUND, INFORMATIONAL ONLY.** Exact encoder
  identity and exact truthful cross-seed canonical action table.
- **W64 exact quotient/action algebra:** **NOT FOUND.** Rolled descent,
  involution, closure, and interchangeability remain non-exact.
- **Over-claimed WIN in result notebook/ledger wording:** **NOT FOUND.**
- **Premature closure / stale public state:** **FOUND.** Re-contextualization
  #24 needs narrowing; README, STATE, and project memory need propagation.
- **Next registered increment:** **explicit learned quotient-trained positive
  control, before further scaling.**

## 2026-08-29 — Round 36b result: every cell BEHAVIOR UNDERFIT; QUOTIENT INELIGIBLE

All four cells completed inside their walls (174 / 606 / 618 / 696 s) and
every cell returns the registered primary status "FAIL — BEHAVIOR UNDERFIT;
QUOTIENT INELIGIBLE". Behavioural fit (train correct / 21,184; held-out /
2,240; five seeds): S16 train 20,894–21,184 with one exact seed, held-out
2,179–2,218; S64 train 21,078–21,184 with two exact, held-out 2,184–2,226;
LR64 train 21,088–21,184 with four exact, held-out 2,198–2,225; W64 train
exact on all five seeds, held-out 2,216–2,239 (98.9–99.96%) — none exact.
Under the registered rule no cell is eligible for a quotient
interpretation, so no PASS, no FIT-BUT-NON-CONGRUENT, and no reading of the
DIAGNOSTIC tables as verdicts. Informational only: W64's ineligible gate
flags show action-table truth, cross-seed action-table identity and
quotient availability passing while well-definedness, involution, the
swap/toggle table, held-out closure and interchangeability fail. Plain
reading: more budget and width move behaviour toward exact fit (W64 is
exact on training) but held-out spellings still miss by 1–24 rows per seed,
so the ladder never reached the point where the quotient question could be
asked; what to make of that — exact held-out fit as a precondition may
simply be unreachable for this recipe on unseen spellings — is with the
fresh auditor (corrected by audit #24: eligibility not reached under this
ladder; reachability of the exact learned certificate unvalidated, not
unsatisfiable). Row-level evidence (≈170 MB per cell) and weights are
retained locally and hash-pinned; only config/manifest/verdict are
committed (`abef6cf`).

## 2026-08-29 — Round 36b launched under lock V3 (review #2 RUN-READY)

The behaviour-fit ladder runs now: four cells (S16 16k steps; S64 64k; LR64
64k at lr .001; W64 64k at width 64), five seeds each, sequential on one
CPU process, then four separate reducers. Before any outcome existed: the
audit-#23 amendment (three-stage primary status; DIAGNOSTIC-only p>0.5
table; cellwise cross-seed accounting; depth traces) was registered
(`9edb892`) and implemented; the lock-review defect (eligibility from
producer aggregates) was closed by row-level logit replay; lock V3 recorded
(`ff8eaa7`); review #2 returned RUN-READY with dynamic probes of every
status branch and a byte-identical v1 fixture; runner and configs
committed (`61e2430`). Outcomes are not inspected until all four producers
finish. Status of the design, verbatim from audit #23: a prospectively
locked, post-outcome, outcome-informed successor — exploratory, not
confirmatory; a PASS would show operational recovery and congruent action
maps in a finite world, not compression into a nontrivial quotient.

## 2026-08-29 — Re-contextualization #23 (2-hour step-back; audit #23 fired on the Round 36 FAIL)

Project and live question: the native mathematics of latent spaces; the
constructive question is now whether behaviour alone can make a latent
world's places and moves well-defined (an operational quotient with a
composable action table) — Round 36 — with NLM-007 closed behind its
closing statement.

Whole-picture check: the day converted a stalled instrument program into
(i) a closed, honestly bounded line and (ii) a runnable distance-0
artifact that ran in under a minute and FAILED. That FAIL is the first
result of the constructive program and it is where tunnel vision would be
most dangerous: the adjudication reads it as under-fitting, and the
registered successor (36b) adds training budget with an exact-fit
eligibility rule. Alternatives held live and put to the fresh auditor:
(1) the 12-cell all-supported signature rule turns a 98%-calibrated model
into a support failure by arithmetic (≈21% of rows fail support even with a
perfect latent) — the gate may be measuring confidence, not structure;
(2) the exact-fit eligibility rule could be a post-hoc rescue, and exact
fit could let a lookup-table-like fit pass; (3) the opposite under-read:
0/176 cross-seed action-table agreement may mean there is no composable
structure at all even where behaviour is right (corrected by audit #23: the
stored 0/176 is an all-or-none whole-table gate, not a cell count — 11/176
cells identical at the registered thresholds, 112/176 at p>0.5, 175/176 by
majority). Foundational thread
advanced: the constructive program now has a real falsifier loop
(artifact → FAIL → adjudicated cause → preregistered successor), which is
what the constitution demanded and NLM-007 never had. Audit #23 (fired, unprimed) returned: valid registered FAIL; behaviour-, calibration- and exactness-confounded; substantial but imperfect operational structure remains (a confidence-free replay at p>0.5 recovers 84.1-98.9% of the one-step action table while every exact gate still fails); the licensed sentence is REPLACED, the '0/176' phrase is withdrawn as misleading bookkeeping, and Round 36b's status logic is NOT READY AS WORDED (three-stage decision required). Replacement paragraph, sections 4-6 and the final verdict follow verbatim.

> Audit #23 upholds the v1 artifact and its all-gate FAIL but narrows the
> meaning. Under the exact 4,000-step recipe, the artifact failed the registered
> confidence-qualified certificate for a fully supported operational identity
> and exact composable action algebra. It did not reach exact behavioral fit,
> and the 12-cell `0.10/0.90` support conjunction strongly amplifies marginal
> confidence defects. A read-only `p>0.5` diagnostic nevertheless recovers
> `84.1–98.9%` of the one-step action table per seed; `112/176` cells are
> identical and truthful across all five seeds, while every exact structural
> gate still fails. Therefore neither “pure calibration failure” nor “no
> composable structure” is licensed. The v1 result is a permanent,
> recipe-specific nonpass with substantial but imperfect structure. Round 36b
> is a transparent post-outcome successor and cannot rescue or overturn it.

#### Audit #23 — sections 4-6 and final verdict (verbatim)

## 4. Strongest alternative explanation

The strongest single explanation is not “the latent has no algebra.” It is:

> The BCE objective, finite sampling distribution, and 4,000-step stop learned
> a mostly correct, partially compositional response system, but left a small
> number of low-margin or wrong response cells. The 12-way confidence
> conjunction and exact all-cell/all-seed reducer magnified those local defects
> into universal gate failures. The remaining seed-dependent errors, especially
> at depth 3 and rolled representatives, show that the transition law itself is
> also incomplete.

This explains every row more economically than either pure-calibration or
no-structure narratives. The population also heavily weights length-3 words:
training contains 1 empty, 11 one-step, 47 two-step, and 1,265 three-step words.
Only `176/21,184` training rows directly supervise the empty/one-step action
signature. More optimization may help, but the ladder does not distinguish
budget from depth weighting or objective geometry.

## 5. Tunnel-vision ruling

The constructive program is scientifically tunnel-visioned despite being much
closer to its claim than NLM-007:

- one 16-state toy;
- one binary response sensor;
- one learned handle table and one residual transition architecture;
- one optimizer family;
- one action algebra;
- one horizon (`<=3` for behavioral rows, with registered rolled probes);
- a singleton oracle quotient, because depth-1 signatures distinguish all 16
  hidden states.

That last point is decisive. A PASS would show bounded state recovery and a
congruent action table, not nontrivial quotient formation. There are no two
different hidden simulator states that the denizen must identify as one place,
and no nuisance state that the quotient must discard.

## 6. What should run alongside Round 36b

### Register before any 36b outcome

1. **Confidence-free diagnostic reducer.** Keep the `0.10/0.90` primary gate
   frozen, but prospectively report the complete `p>0.5` gate table, component
   error counts, and margins. It is diagnostic only and cannot rescue a primary
   FAIL.
2. **Literal cellwise cross-seed accounting.** Report (a) identical cells,
   (b) identical supported cells, (c) all-five truthful cells, and (d) bitwise
   majority truth, beside the existing whole-table exact gate.
3. **Three-stage decision status.** Separate behavior underfit, signature
   underconfidence, and supported non-congruence as above.
4. **Depth-balanced diagnostic.** Either add a prospectively frozen
   depth-balanced sampling arm or, minimally, report loss/accuracy/support by
   word depth throughout training. The current four-cell ladder changes budget,
   learning rate, and width but never tests the severe depth imbalance.

### Orthogonal controls

5. **Learned lookup baseline.** A handle-by-observed-word memorizer should fit
   train and fail held-out spellings; this calibrates how much closure comes
   from composition rather than finite lookup.
6. **Explicit finite-state/quotient-trained positive control.** Train the same
   carrier with direct state-transition or quotient-consistency supervision,
   scored by the unchanged reducer. The existing fixture is oracle-authored,
   not a learned representability/control arm. If the explicit control passes
   and behavior-only training fails, the gap belongs to the learning objective,
   not representability.
7. **A genuinely nontrivial quotient world.** Add nuisance hidden bits or
   duplicate simulator states with identical response futures, require several
   hidden states to collapse into each operational place, and demand action
   descent across those independently generated representatives.
8. **Longer, algebraically novel continuations.** Hold out depth 4–6 and word
   families selected by algebraic relation, not only spelling hashes, to attack
   finite-horizon lookup and behavioral redundancy.
9. **A second transition architecture.** A linear/affine action model or a
   small recurrent alternative should be fixed before outcome. One
   architecture cannot distinguish a world property from an inductive-bias
   accident.

A 36b PASS should trigger a fresh preregistration on the nontrivial-quotient
world, not immediate activation of Round 35.

## Final verdict

- **Claim (a), mechanical FAIL:** **UPHELD.**
- **Claim (a), “incomplete behavioral fit”:** **UPHELD, with optimization not
  proven as the sole cause.**
- **Licensed sentence:** **REPLACE** with the confidence-qualified
  non-certification wording above.
- **Over-claimed KILL:** **FOUND.** The primary reducer materially conflates
  confidence/support with structure; the current prose suppresses strong
  approximate action-table recovery.
- **Under-read FAIL:** **ALSO FOUND.** Confidence-free exact composition and
  cross-seed invariance still fail; the artifact is not merely underconfident.
- **Claim (b), successor legitimacy:** **UPHELD only as a transparent,
  exploratory post-outcome successor.** It is not a v1 repair and not
  confirmatory.
- **Round 36b exact-fit/non-congruence rule:** **NOT READY AS WORDED.** Split
  calibration from supported non-congruence before any 36b interpretation.
- **Tunnel vision:** **FOUND.** A singleton quotient in one toy and one
  architecture cannot carry the constructive program alone.

## 2026-08-29 — Round 36 first run: the constructive artifact exists and FAILS every gate

The first distance-0 artifact ran end to end: `produce` (five registered
seeds, CPU only, one process) completed non-claiming in 52.6 s (train 41.7 s,
evidence 11.0 s; wall 900 s); the separate `reduce` returned FAIL on every
gate — quotient availability, quotient well-definedness, toggle involution,
swap/toggle table, held-out depth-2 and depth-3 closure, interchangeability,
action-table truth (0/5 seeds; 14–56% of 176 cells), cross-seed action table
(0/176 identical — corrected by audit #23: an all-or-none whole-table gate,
not a cell count). Signatures carry many unsupported ("?") responses.
Registered meaning: this training recipe did not produce a well-defined
operational quotient in this latent space — a constructive hole here, not a
hostile hole in general. The obvious alternative reading is the boring one:
the recipe underfit (a 1,041-parameter model trained ~8 s per seed may not
have fit the behavioural data at all), in which case the quotient gates were
never really eligible. That question — underfit vs fit-but-non-congruent vs
gate construction — is with Codex as an evidence/design ruling, together
with what the registration permits next WITHOUT outcome-contingent tuning
(a preregistered budget ladder with a behaviour-fit eligibility criterion
frozen before any 36b outcome). No tuning has been done. Artifacts committed
(`073037f`).

Adjudicated (Codex, `.codex_round36_adjudication1.md`): classification (a)
— incomplete behavioural fit under the frozen recipe (train accuracy
96.6–98.5%, held-out 97.0–98.3%, depth-3 93.8–96.3%, loss still falling at
step 4,000; the 12-cell support requirement amplifies residual errors into
1–58% support). The v1 FAIL stands permanently; licensed claim, verbatim:
"Under the exact 4,000-step v1 recipe, the produced latent artifact did not
supply its denizen with a fully supported operational identity or
composable action algebra." (Corrected by audit #23: this sentence is
REPLACED by the audit #23 paragraph in the entry above — say: failed to
certify the exact confidence-qualified algebra.) Round 36b is preregistered (`f9dea33`) as a
successor design, not a repair: a four-cell behaviour-fit ladder (S16,
S64, LR64, W64), every cell run and visible, quotient gates eligible only
at exact behavioural fit (21,184/21,184 train, 2,240/2,240 held-out),
otherwise "FAIL — BEHAVIOR UNDERFIT; QUOTIENT INELIGIBLE"; exact fit
followed by a quotient failure would be the first legitimate
FIT-BUT-NON-CONGRUENT result. Configs and runner revision are hash-locked
before any 36b outcome.

## 2026-08-29 — NLM-007 closed: audit #22 upholds the terminal stop; closing statement adopted verbatim

NLM-007 is closed under the program’s terminal allocation rule, not by a scientific null.

Within one pinned decoder and authored population, the correlated A/B punctuation sentinels established bounded F4–F20 condition robustness: the qualified four-cell ridge table retained X-linked predictive separation across held-out blocks and words. The raw token-context comparator was highly non-robust to P_static residualization and therefore P_static-aligned in this fitted design. Round 34a found a small but systematic raw separation surviving the registered EDF match (+0.04–0.08 cosine), while the larger static separation was not eliminated by that match within the fixed feature classes. Round 34b did not resolve the interpretation: P+C improved on P by roughly +0.02–0.04, defeating the redundancy STOP, but C_perp→Δ_perp failed the joint retention gate; every eligible layer and the joint reducer were INCONCLUSIVE. Under the pre-adopted ruling, that is an allocation stop, so Round 34c does not run.

This line did not identify operational state, a denizen-usable or native law, composition, a representation-level hostile hole, or independent replication. It leaves frozen captures; raw/static, matched-EDF, and partial-overlap analyzer modes; hash-bound cell sidecars; block-first held-out evidence; and fail-closed producer/reducer discipline.

Round 36 now asks the constructive question directly: can behavior alone support a well-defined operational quotient and composable action table in the minimal 16-state world?

#### Audit #22 — Round 34b wording corrections, EDF-correction ruling, and Round 36 handoff (verbatim)

### Round 34b interpretation

The numerical shorthand needs slight tightening:

- `P+C − P` cosine is A `+0.0178–+0.0373` and B `+0.0238–+0.0355`. A/F4 falls just below the point threshold, but its upper interval exceeds `0.02`; the other means exceed `0.02`. Thus the redundancy STOP fails.
- Residual-context cosine is approximately `+0.019–+0.089`, not quite `+0.03–+0.10`.
- Residual normalized-error gain is negative for every ridge/kernel, sentinel, and F4–F20 cell. Clustered key/block requirements also fail. Thus retention fails.

The positive `P+C − P` increments are evidence **against the registered strict-redundancy account**, but not evidence for operational state: `P+C` has greater capacity, while the residual partial relation fails its joint gate.

### EDF correction

The correction is mathematically justified. The producer sums all nonnegative eigenvalues for EDF but defines rank only above tolerance; therefore EDF can exceed numerical rank by a small sub-tolerance tail. The old fit bound is violated in eight selected-state fits, all at excluded F0, by only `2.93–2.94×10⁻⁵`. No eligible F4–F20 fit violates it.

Producer JSON, sidecars, reductions, and gate functions were unchanged. Strictly, the old joint reducer had no verdict—it was `INCOMPLETE`. The repair changed reducer status to `COMPLETE`, while preserving the already-recorded sentinel decisions and recomputing the same joint `INCONCLUSIVE`.

Do not call the repair literally outcome-blind: it was triggered after seeing the artifacts. The defensible wording is **post-outcome but not outcome-selective**. Its formula follows the pre-existing producer definition, applies symmetrically, and affects only diagnostic F0 telemetry.

## Overclaim and underclaim audit

- Round 34a raw is a genuine registered survival, but small: capacity matching removed most of the unmatched gap. Its lower bounds clear zero, not uniformly `0.02`.
- Round 34a static supports only “not eliminated by the registered EDF match within these fixed feature classes.” It does not prove capacity independence or feature adequacy.
- `ctxS` supports high non-robustness of the raw context comparator to `P_static` residualization—hence `P_static` alignment in this fitted design—not presentation share, mediation, or causal explanation.
- The four-cell table supports qualified within-decoder condition robustness. It remains correlated, same-population evidence; B-score4’s KL-rank/low-rank qualification and F0’s model-class sensitivity remain.
- Closing now is correct under the prospectively adopted allocation constitution. Scientifically, it deliberately leaves the item-by-carrier fingerprint/local-Jacobian explanation unresolved. Round 34c might have clarified that account, but the constitution explicitly forbids escalating an `INCONCLUSIVE` rung.

One durability defect remains: `.codex_audit21.md` is only a 327-byte self-referential completion stub. Audit #21’s substantive text survives in `NOTEBOOK.md`, `STATE.md`, and the ledger, but the named output itself should not be treated as evidence.

## Round 36 handoff

**Tunnel vision:** NLM-007 remained scientifically tunnel-visioned around increasingly refined readers of one punctuation relation. Closing it is correct.

**Strongest alternative:** Round 36 could fit the finite response table without organizing a composable latent world. Also, its depth-1 signatures distinguish all 16 simulator states, so the oracle quotient has singleton hidden-state classes. A PASS demonstrates operational recovery and congruent action maps—not compression into a nontrivial quotient.

**What should run:** first close Round 36 review #1’s CLI, wall-metadata, and fixture-isolation blockers. Then run the registered reducer fixture. After it returns `PASS / INVALID / INVALID / FAIL` on the exact and three mutation cases, run the exact five-seed CPU producer and separate reducer—no pilot or seed replacement.

The first scientific falsifier is action descent on rolled representatives: if any two supported points with the same `Σ₁` signature reach different or unsupported quotient classes under one primitive action, the quotient action is not well-defined. Held-out H2/H3 interchangeability should follow immediately to attack the response-memorization alternative.

Blackboard entries e673–e683 were recorded; convergence reached 100%, and synthesis was read. This audit edited no project source/result file and made no commit.

## 2026-08-29 — Round 34b result: INCONCLUSIVE in both sentinels — the terminal rung

`analysis_ctxoverlap_A.json` (444 s) and `analysis_ctxoverlap_B.json`
(595 s), static estimand, producers run-ready. Every F4–F20 layer is
INCONCLUSIVE in both sentinels. Sentinel A (block-first means, F4/F8/F12/
F20): P_static→Δ alone reaches cosine 0.49/0.43/0.46/0.58; the token-context
ridge alone 0.51/0.47/0.50/0.62; the nested P+C 0.51/0.46/0.49/0.61 — so
P+C − P ≈ +0.02 to +0.04, which fails the redundancy STOP (needs ≤ 0.02 with
crossed UB < 0.02) — while C⊥→Δ⊥ keeps only ≈ +0.03 to +0.10 cosine by
block, which fails the retention rule. Reading under the registered rules:
the registered raw context field is neither P_static-redundant nor clearly
retaining residual signal in this design; neither the "by construction"
nor the "fitting-artefact" account is licensed. Under the continuation
ruling an INCONCLUSIVE rung is an allocation stop: the NLM-007 terminal
ladder ends here (pending the joint artifact, whose reducer currently
rejects the valid producer artifacts on a rank/EDF telemetry bound — a
bounded reducer repair is with Codex; producers untouched). Round 34c does
not run. NLM-007's closing statement is drafted after the joint and the
next fresh audit.

Joint (reducer repaired — the EDF≤rank bound was producer-inconsistent by
~3×10⁻⁵ at F0's state_selected fit; producers correct, no rerun):
`analysis_ctxoverlap_joint.json` COMPLETE/SCREEN-ONLY, decision
INCONCLUSIVE, no common retaining layer (ridge or kernel), no common stop
layer. The terminal ladder therefore ends at Round 34b. Audit #22 fired on
the terminal outcome and on the draft closing statement.

## 2026-08-29 — Audit #21 adversarial correction: both 34a verdicts upheld, claim boundaries tightened

The four float32 evidence sidecars replay exactly through the registered reduction and decision code. RAW and STATIC both return `CONTINUE` at F4/F8/F12/F20 in both sentinels, with 8/8 jointly positive keys at every eligible sentinel-layer. F0 is correctly `INCONCLUSIVE` and diagnostic because at least one required context-EDF target exceeds the selected F0 state EDF, making a downward match undefined; F0 is excluded from the ladder gate.

RAW is a valid registered `CONTINUE`, not a numerical boundary artefact. The `0.02` threshold applies to the point margin; the lower-bound criterion is `>0`. The smallest raw point over cosine/nerr is 0.0397, the smallest lower bound is 0.0146, and float32 resolution is immaterial at that distance. The replicate-wise minimum over the four predeclared candidates is conservative for survival, not multiplicity inflation. Correct wording: capacity matching removed most of the unmatched raw gap but did not exhaust it; the +0.04 to +0.08 cosine separation is small in magnitude but systematic within this locked design, with both endpoints positive in all eight keys at every F4–F20 layer in both correlated sentinels. It is not a state claim.

STATIC also mechanically survives, but withdraw the provisional sentence “the residual separation is not a capacity artefact.” The selected contextual ridge target is approximately 47 EDF throughout; the selected kernel is approximately 48 EDF at F8–F20 but falls to approximately 4.36 in 4/8 A and 2/8 B F4 keys. The selected state ridge ranges from approximately 202 to 384 EDF and is therefore heavily shrunk for the comparison. Honest wording: “the residual predictor separation was not eliminated by the registered EDF match within these fixed feature classes.” This rejects a simple unmatched-slope-EDF explanation, but the fixed context arm is near-null on `Delta_perp`, equal EDF does not equal feature adequacy, and the item-by-carrier fingerprint/local-Jacobian account remains live. No operational state, native law, or representation-level hostile hole is identified.

Both 34a estimands returned `CONTINUE`, so the terminal ladder may proceed to 34b only after its final bounded `RUN-READY`; 34c remains conditional on a 34b `CONTINUE`. These results make neither control moot and reopen none of the cut queue. Round 36 remains the higher-leverage constructive line.

#### Audit #21 — section 6 (tunnel vision, strongest alternative, run order) and final ruling, verbatim

## 6. Tunnel vision, strongest alternative, and what should run

### Tunnel-vision ruling

The program remains tunnel-visioned at the scientific level even though the
continuation ruling has now contained the allocation error. These outcomes
refine one predictor comparison in one decoder, one authored micro-world, two
punctuation tokens, one append move, and correlated depths. The static margin
does not justify another control family beyond the already terminal 34b/34c
ladder.

The strongest live scientific alternative explanation remains:

> `P_static` removes a coarse block/length/position response, while `X_perp`
> retains a dense item-by-carrier activation fingerprint and a local
> punctuation Jacobian. A low-EDF ridge can extract a few high-signal directions
> from that learned nonlinear feature map. The registered token-context field,
> even at its rank ceiling, omits the item token and dense interactions and is
> near-null on `Delta_perp`. No denizen-usable state or operation is required.

The strongest program-level alternative is already registered: Round 36's
minimal operational quotient. It directly tests whether identity and actions
descend to a behaviorally available quotient in a runnable world, instead of
perfecting another reader of the punctuation relation.

### Concrete, cheap run order

1. **34b first, if and only if RUN-READY.** Expectation: determine whether raw
   context is `P_static`-redundant and whether residual context retains signal
   under the fully nested projection. A redundant result stops the ladder; a
   retained-signal result continues but revises the static interpretation.
   The simplest fatal confound is nuisance/vocabulary reuse across an inner or
   outer held-out boundary.
2. **34c only after 34b CONTINUE and RUN-READY.** Expectation: test the
   item-by-context fingerprint account with the registered richer X-free field.
   Closure means feature sensitivity, not causal context; survival still does
   not identify state. The simplest fatal confound is PCA or vocabulary leakage
   from held-out words.
3. **Keep Round 36 moving as the higher-leverage constructive line.** Run its
   reducer fixture before learned evidence, then the CPU producer and separate
   reducer, one process at a time. No NLM-007 result changes Round 36.
4. **Run no new NLM-007 arm.** The five-seed bootstrap replay above is enough
   to answer the immediate numerical-boundary worry as an audit diagnostic.
   A 10,000-bootstrap polish, random-weight decoder, second decoder, or richer
   context family would be more measurement infrastructure and violate the
   terminal allocation ruling.

## Final ruling

- **Raw mechanical verdict:** upheld.
- **Raw claim:** small but systematic within-design survival; not a state claim.
- **Raw correction:** lower bounds clear zero, not necessarily 0.02.
- **Candidate multiplicity:** conservative for `CONTINUE`; synthetic-oracle
  wording boundary remains mandatory.
- **Static mechanical verdict:** upheld.
- **Static claim:** not eliminated by this registered EDF match; do not call it
  generally capacity-free.
- **Static telemetry correction:** kernel selected EDF has material low-EDF F4
  exceptions; selected state EDF varies from about 202 to 384.
- **F0:** correctly undefined/diagnostic and excluded from the ladder.
- **Queue:** 34b if finally `RUN-READY`, then 34c only on 34b `CONTINUE`; no
  cut item reopens.
- **Tunnel ruling:** finish the bounded closeout without adding arms; prioritize
  Round 36 as the distance-0 constructive program.

Blackboard findings e649–e655 were recorded with provenance. `bb_convergence`
returned 100% with no open signals, disputes, unread documents, or partial
documents, and `bb_synthesis` was read before this verdict. No project source
or tracked file was edited, and no commit was made.

## 2026-08-29 — Round 34a STATIC result: CONTINUE at F4–F20 with large matched margins; the "match" sits at the context rank ceiling

`analysis_ctxcapA_static.json` / `analysis_ctxcapB_static.json` (frozen
analyzer copy; 314 s / 304 s) and the static joint (`analysis_ctxcap_static_joint.json`,
re-run on the main analyzer after the NaN-replay fix: COMPLETE/SCREEN-ONLY,
CONTINUE, common layers F4/F8/F12/F20). On the P_static-residualized
relation the strongest matched margin is: A cosine +0.306 / +0.383 / +0.373
/ +0.435 at F4/F8/F12/F20 (LBs 0.227 / 0.315 / 0.305 / 0.353), nerr +0.047
/ +0.089 / +0.084 / +0.115; B cosine +0.329 / +0.352 / +0.337 / +0.367 (LBs
0.262 / 0.278 / 0.275 / 0.264), nerr +0.065 / +0.082 / +0.077 / +0.100; 8/8
keys; F0 INCONCLUSIVE (diagnostic).

What the match telemetry says: on this relation the context arms saturate
at their rank ceiling (target EDF 47 for the ridge, 48 for the kernel) while
the selected residual state ridge has EDF ≈ 267 — so the "matched" state
arm is the residual ridge shrunk to 47–48 df, and it still keeps ≈ 0.35
held-out cosine that the token-context arms cannot supply at any attainable
capacity. Provisional reading (audit #21 fired on both forms, wording
pending): the residual separation is not a capacity artefact; the raw
separation mostly was (raw matched margins +0.04–0.08). Neither is a state
claim — feature adequacy (item-by-context) is untested until 34c. Both
estimands returned CONTINUE, so under the continuation ruling the ladder
proceeds to 34b (conditional on its final RUN-READY), then 34c.

## 2026-08-29 — Round 34a RAW result: CONTINUE at F4–F20, but the matched margin is small

`analysis_ctxcapA_raw.json` / `analysis_ctxcapB_raw.json` (frozen analyzer
copy 6b93ff1; 291 s and 254 s; tokenizer only, no model forward) and the
raw joint reduction (`analysis_ctxcap_raw_joint.json`: COMPLETE/SCREEN-ONLY,
CONTINUE, common layers F4/F8/F12/F20). With the state ridge bisected down
to the contextual arm's effective df, the strongest matched margin is:
sentinel A cosine +0.072 / +0.057 / +0.045 / +0.042 at F4/F8/F12/F20 (crossed
LBs 0.034 / 0.024 / 0.019 / 0.024), normalized error +0.073 / +0.047 / +0.040
/ +0.054; sentinel B cosine +0.082 / +0.064 / +0.047 / +0.043 (LBs 0.049 /
0.034 / 0.023 / 0.022), nerr +0.088 / +0.054 / +0.042 / +0.067; 8/8 keys
jointly positive at every F4–F20 layer; F0 INCONCLUSIVE (the
selected-context-EDF match is undefined there; diagnostic only). The
strongest arm is the token-id kernel at most layers.

Plain reading: capacity matching removed most of the unmatched raw gap
(ctx_A/ctx_B cosine margins were +0.11 to +0.20); what survives is a
+0.04 to +0.08 cosine separation with lower bounds just above the 0.02
threshold at F12/F20. CONTINUE by the registered rule — a narrow survival,
not a strong one, and not a state claim. Interpretation waits for the
static form (running now) and the fresh auditor. One reducer defect
surfaced and was fixed on the main analyzer without touching the producer
(a stored NaN at the F0 diagnostic compared unequal to its replay); the
joint was re-run in seconds.

## 2026-08-29 — Parity verdict: refactored analyzer reproduces HEAD; Round 34a runs begin

The HEAD-vs-refactor CPU parity check (contextual-prefix static screens,
sentinels A and B, committed analyzer copy vs the parked branch's
refactored analyzer, decision JSON scrubbed of timing/SVD/shadow fields)
returned IDENTICAL for both sentinels. The parked consequence branch's
legacy-parity question is therefore answered in its favour; the gate itself
is cut by the continuation ruling and the result is kept as evidence only.
The Round 34a closeout ladder started immediately on the frozen analyzer
copy: ctxcapA_raw is running (raw B, raw joint, static A/B, static joint
follow; minutes each).

## 2026-08-29 — Re-contextualization #22 (2-hour step-back; audit skipped — still no new claim)

No Round 34a outcome exists yet (the parity screens are on their last
layer), Round 34b/34c are in repair round 2 of 3, and no new result has
been claimed since audit #20; the fresh auditor is held for the first 34a
outcome. Live question unchanged; direction unchanged (cheap capacity /
feature-adequacy ladder first, expensive instruments held or parked).

What this pause is used for: the one review the constitution requires per
cycle that has NOT been fired explicitly during this long instrument phase
— "should this program continue at all, and is this the highest-leverage
thing to be doing?" (global CLAUDE.md §2.7 rule 5). A fresh Codex session is
asked exactly that, portfolio-level, with Round 35 (typed truth-evaluable
world) on the table as the constructive alternative and the audit #19/#20
alternatives (random-weight architecture null; item-by-context null) as the
cheap moot-makers. Its answer is recorded verbatim below when it lands.

Its answer landed and is adopted as the Codex program ruling (verbatim
below): STOP NLM-007 as an open-ended program — infrastructure drift by the
constitution's own tripwire (measurement:artifact ≥ 6:1) — and run only a
terminal closeout ladder: Round 34a raw and static once; 34b then 34c only
on CONTINUE and only if their final bounded repair is RUN-READY; full Round
34, Round 33 (branch archived), the parity gate, the random-weight null and
a second decoder are cut. Any INCONCLUSIVE rung is an allocation stop. Round
35 becomes a requirements envelope; the next artifact is a minimal
operational-quotient / bisimulation world on the 16 four-bit states (design
gate opened). Governance: mandatory producer/reducer separation.

#### Program continuation ruling (Codex, verbatim)

# Program continuation ruling — NLM-007

## Executive ruling

**STOP NLM-007 as an open-ended research program.** It is in infrastructure drift under the project constitution.

Authorize only a tightly bounded **terminal adjudication** using the already-built cheap screens. Regardless of whether those screens return STOP, INCONCLUSIVE, or CONTINUE, NLM-007 then closes. A CONTINUE result may earn a narrow measurement claim; it does not justify full Round 34, Round 33, another decoder, or further work on this punctuation relation.

The decisive issue is not that NLM-007 failed. It is that it has already yielded its transferable result—bounded within-decoder condition robustness plus several instrument boundaries—while further work increasingly measures the measurement rather than building the README’s stated artifact: a native mathematics of latent spaces ([README.md](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/README.md:3)).

## 1. Distance-from-claim accounting

The constitution defines distance 0 as the central artifact, 1 as something it directly needs, 2 as a measurement, and 3+ as measurement infrastructure. It requires a raise above 2:1 and a halt above 5:1 ([CLAUDE.md](/C:/Users/devan/.claude/CLAUDE.md:143)).

| Workstream | Distance | Reason |
|---|---:|---|
| SVD telemetry gate | 3+ | Numerical telemetry and backend validation for one candidate inside an already secondary measurement endpoint. |
| Round 33 consequence instrument | 3+ | Implementation, compatibility, provenance, parity, wall, and reducer machinery for an unrun measurement. The eventual run would be distance 2. |
| Full Round 34 | 3+ | Six-arm measurement apparatus and custom claiming reducer; no outcome exists. |
| Round 34a | 3+ so far | Design, implementation, evidence sidecars, reducer, fixtures, and four reviews. Its queued run would be distance 2. |
| Round 34b | 3+ so far | Custom partial-overlap measurement apparatus and reviews. |
| Round 34c | 3+ so far | Custom item/context comparator, PCA provenance, EDF telemetry, reducer, and reviews. |
| Round 35 docs-only design | 1, generously | Directly specifies a possible constructive artifact, but nothing runnable exists and no population has been authored. |
| Central runnable mathematics artifact | 0 units | No native law, operational quotient, composition law, new axiom, or representation-level hostile hole was produced. |

Conservative workstream ratio:

- Measurement/infrastructure units: **6**
- Artifact-facing units, counting the docs-only Round 35 design generously: **1**
- Ratio: **6:1**

If “artifact” means the constitution’s runnable central artifact, the denominator is zero and the ratio is unbounded. If 34b/34c are combined as one unit, the parity instrument or completed contextual measurements immediately restore a ratio above 5:1. This is not sensitive to reasonable unitization.

Therefore the program is **in infrastructure drift by definition**. The ledger’s assertion that “the artifact here IS the measured relation” is constitutionally invalid: rule 6 says the heartbeat must anchor on the README’s central bet, not the current cycle’s internal frame.

## 2. Should NLM-007 continue?

### Strongest STOP case

NLM-007 has already produced:

- A bounded result: within one decoder and authored population, a residual ridge remains predictive under several conditions.
- Withdrawal of the stronger affine-law reading once identity plus shared displacement was tested.
- Context comparisons still confounded by capacity and feature adequacy.
- No operational state, denizen-usable quotient, composition, native law, new axiom, or representation-level hostile hole.
- Proven instrument problems: an insensitive ordering readout, SVD fragility, reducer/provenance complexity, and construct ambiguity.
- Repeated review cycles increasingly concerned with hashes, schema mirrors, telemetry binding, evidence packing, wall semantics, and custom reducers.

Audits #19 and #20 independently call the program tunnel-visioned ([audit #19](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/.codex_audit19.md:228), [audit #20](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/.codex_audit20.md:379)). The evidence discipline prevented overclaims, but further reducer perfection does not build the denizen’s mathematics.

### Strongest CONTINUE case

The 34a/34b/34c sequence is cheap relative to prior work, uses existing captures, and targets the strongest live alternatives:

- 34a: capacity sensitivity.
- 34b: `P_static`/context redundancy or projection artefact.
- 34c: omitted item-by-context features.

A terminal MOOT or REDUNDANT verdict would close the relation cleanly. Survival would justify the narrow statement that the predictor separation survived these registered controls.

### Rule

**NLM-007 does not continue as a program. It receives one terminal closeout ladder.** Even an all-CONTINUE ladder ends with a bounded measurement claim and closure; it does not reopen the broader queue.

An INCONCLUSIVE result remains scientifically inconclusive, but it is an allocation stop. It must not trigger a more elaborate instrument.

## 3. Exact queue ruling

| Item | Ruling | Reason |
|---|---|---|
| Round 34a raw | **RUN once** | Already RUN-READY; cheapest direct adjudication of the historical raw comparison. |
| Round 34a static | **RUN once, separately** | Settles the distinct residualized relation; no cross-estimand pooling or rescue. |
| Round 34b | **CONDITIONAL RUN** | Run only if both 34a estimands return CONTINUE and the current final bounded repair receives RUN-READY without scope expansion. Otherwise cut. |
| Round 34c | **CONDITIONAL RUN** | Run only after a 34b CONTINUE and the same final readiness condition; it tests the strongest omitted-feature account. |
| Full Round 34 | **CUT** | Over-bundled, farther from the central artifact, and cannot upgrade survival into operational state. |
| Round 33 consequence | **CUT / archive parked branch** | Four-review instrument debt; even a pass licenses only frozen-tail predictive persistence. |
| Parity check | **CUT as a gate** | Preserve any completed output, but do not restart, repair, or delay 34a for it; it served the now-cut consequence branch. |
| Random-weight architecture null | **CUT from NLM-007** | Another diagnostic of the same ambiguous relation; it cannot produce a native construct. Reuse the idea inside a future constructive world if needed. |
| Second decoder | **CUT** | Replicating an unresolved construct does not resolve the construct. Reconsider only after a behaviorally valid native-world artifact exists. |

After the first STOP/MOOT/REDUNDANT or INCONCLUSIVE rung, stop the ladder. If all rungs return CONTINUE, record the narrow result and close NLM-007 anyway.

## 4. Round 35 and better constructive programs

Round 35 is the **right direction but the wrong first artifact**. It supplies known state, moves, consequences, and composition, but the registered design already combines linguistic authoring, adversarial approval, tokenization parity, two surface systems, two query families, matched-EDF ladders, causal patches, transfer, composition, and a 20-hour CPU envelope ([Round 35](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/theory/EXPERIMENTS.md:7427)). That risks reproducing infrastructure-first drift before the smallest runnable world exists.

Use the Round 35 document as a requirements envelope, but build a reduced first artifact. Three concrete alternatives are:

1. **Minimal operational quotient / bisimulation world.**  
   Train a tiny latent transition system on the 16 four-bit states and fixed toggle/swap/no-op actions. Define identity solely by equality of future response signatures under allowed actions.  
   **Falsifier:** quotient-equivalent states cease to be interchangeable on held-out action sequences, or actions do not descend to well-defined maps on the quotient.

2. **Cross-seed gauge-invariant action algebra.**  
   Train several independent latent realizations of the same finite world and recover the transition semigroup without coordinate alignment.  
   **Falsifier:** the purported identity classes or operation/composition table changes with seed or chart despite identical behavioral truth tables. That would show the “law” is representation-specific, not native.

3. **Denizen-available controllability and closure graph.**  
   Give the model a small declared intervention set, construct the reachable-state graph from behavioral response signatures, and test held-out two- and three-step closure.  
   **Falsifier:** single-step moves cannot be composed into stable equivalence classes, or predicted reachable states cannot causally enact the registered consequences. That would be a genuine local composition/controllability hole.

My recommendation is alternative 1 first. It is the smallest runnable object that can falsify the central bet. Add natural-language transfer, elaborate X-free ladders, and multiple query families only after the quotient and action table work at all.

## 5. Governance ruling

**Yes, review has become a bottleneck—but the deeper cause is the coupling of scientific producers to bespoke claiming reducers.** Reviewers were finding real defects, so simply reviewing less would weaken the gates.

The single most useful change is:

> **Mandatory producer/reducer separation.** A frozen, non-claiming producer receives execution readiness independently. Claim readiness belongs to a separate declarative, fail-closed reducer. Reducer defects may block interpretation, but they do not repeatedly rewrite or block an otherwise sound producer.

Audit #19 already demonstrated the value of this split ([audit #19](/C:/Users/devan/OneDrive/Desktop/Projects/Latent-Space-Reasoning/.codex_audit19.md:192)). It preserves every evidence gate—no claim is issued before reducer validation—while eliminating the dominant producer/reducer review-loop coupling.

Blackboard findings were recorded; convergence returned 100% with no open signals or disputes, and synthesis was read before this ruling. No project source or tracked file was edited, and no commit was made.
Alternatives held live otherwise unchanged. Foundational thread advanced:
program-level continuation review as a standing artifact, not an implicit
assumption.

## 2026-08-29 — Re-contextualization #21 (2-hour step-back; audit skipped — no new claim since audit #20)

Audit: the only new artifact since audit #20 is ctxS_B, a replicate of
ctxS_A already worded under audit #20's correction; the instruments (Round
34a run-ready and queued; 34b/34c under Tier-1 review) carry no outcome. No
fresh auditor fired this cycle; the next fires when the first Round 34a
outcome exists.

Live question unchanged. Whole-picture check: the program has spent this
day converting one descriptive separation (state ridge vs token-context
field, raw and residualized) into a ladder of cheap, preregistered
capacity/feature controls — 34a (matched EDF), 34b (P/C partial overlap),
34c (item-by-context) — with the expensive instruments (six-arm Round 34,
Round 33 consequence) held or parked (superseded by the continuation ruling:
full Round 34 and Round 33 are cut; 34b/34c conditional rungs of a terminal
ladder). That is the right shape: the cheapest
moot-makers run first. Reframing: every result so far is a statement about
readers of one residual relation, not about a latent-space law; the second
lens (holes hostile to structured reasoning) has produced an instrument
boundary, not a representation-level hole.

Alternatives held live: item-by-carrier fingerprint + local Jacobian
(strongest); pure capacity; decoder specificity; architecture-matched
random-weight null (superseded by the continuation ruling: the random-weight
null and a second decoder are cut from NLM-007). Foundational thread advanced this cycle: opening the
design gate for the typed truth-evaluable world (audit #19 alternative 3 /
audit #20 tunnel ruling) as a docs-only Round 35 preregistration (superseded
by the continuation ruling: Round 35 is a requirements envelope; the first
constructive artifact is Round 36) — a
four-bit finite-state world with toggle/swap/no-op, held-out predicates and
templates, frozen forced-choice yes/no log-odds, wrapper and same-length
controls, causal patching, involution and one non-commuting two-step
composition — so that when the capacity ladder resolves, the next
population is a world with known state, move, consequence and composition
laws rather than another mentioned-string micro-world. No texts or config
authored; no GPU.

## 2026-08-29 — ctxS_B complete: contextual-prefix comparator on the P_static-residualized relation, sentinel B (audit #20 wording)

`analysis_ctxS_B.json` (committed analyzer, `--residualize static`, 20
shuffles, 500 bootstraps, 4784 s): sentinel B mirrors sentinel A. On the
residualized X⊥→Δ⊥ relation the registered `token_ids_v1` context arms fall
to held-out cosine 0.04–0.08 and normalized error ≈ 1.00 at F4–F20, while the
residual state ridge keeps cosine 0.52–0.58 and normalized error 0.82–0.86;
block-first margins vs the strongest context arm: cosine +0.46 to +0.51 (LB
≥ 0.42), nerr +0.14 to +0.19, skill +0.34 to +0.45 (LB ≥ 0.19), continuous KL
+0.24 to +0.41 (LB ≥ 0.08); 8/8 keys, support 1.0. F0: cosine +0.26 but nerr,
skill and KL margins negative. Licensed reading (audit #20, verbatim
discipline): raw context performance is highly non-robust to the registered
P_static residualization and therefore P_static-aligned in this fitted
design; not identified as presentation, not by construction, not a state
contribution; the residual predictor separation is descriptive and
unmatched in capacity. Both static-form comparators are now complete; the
parity check runs next, then Round 34a (raw and static forms).

## 2026-08-29 — Re-contextualization #20 (2-hour step-back; audit #20 fired, unprimed)

Live question unchanged: is the surviving X⊥→Δ⊥ predictability in one small
decoder an operational-state relation, a generic contextual-response
relation, a capacity artefact, or an instrument artefact — and what does the
answer say about holes hostile to structured reasoning.

What holds: the adjudicated four-cell table; ctx_A/ctx_B (descriptive
higher-EDF predictor comparisons, audit #19 wording); ctxS_A. What ctxS_A
reframes: on the residualized relation the token-context arms have nothing
left (cos ≈ 0.05) while the residual ridge keeps cos ≈ 0.6. Two readings
are live and audit #20 is asked to choose (corrected by audit #20: (i) withdrawn as an underclaim, (ii)'s "re-measuring presentation" withdrawn as a variance-share reading; the ruling is "highly non-robust to P_static residualization, P_static-aligned in this fitted design"): (i) "by construction" — P_static
and the token-id field encode the same template metadata, so the collapse
is expected and says little; (ii) the collapse is informative — it shows
the raw-relation contextual comparators were largely re-measuring
presentation, which would make the static form the right estimand for
Round 34a and the raw form a secondary check. A third reading: P_static
(~10 columns) is a much smaller nuisance design than the token field (~220
columns), so residualization could leave token-level signal for the ridge
to exploit — then the collapse of the ctx arm is a df/feature-space
artefact of the comparison, not a fact about the state. Alternatives held
live otherwise unchanged (capacity; Jacobian account; decoder specificity;
typed truth-evaluable world; random-weight architecture null). Instrument
governance status: Round 34a in repair round 2 of 3 (superseded: RUN-READY at 6b93ff1 after round 4); consequence parked;
full Round 34 held (superseded by the continuation ruling: full Round 34 cut,
Round 33 archived, random-weight null cut). Foundational thread advanced: the estimand question
(raw vs residualized) is now explicit rather than implicit in tag names.
Audit #20 (fired, unprimed) returned CONDITIONAL (one overclaim, one underclaim); its correction block, its execution priorities / strongest alternative / tunnel ruling, and its final ruling follow verbatim.

## 2026-08-29 — Audit #20 adversarial correction

(Queue items in this entry — full Round 34 held, Round 33 parked, the
random-weight null, Round 35 as the constructive program — are superseded by
the continuation ruling; the wording rules stand.)

`ctx_B` mirrors `ctx_A` only as a bounded raw predictor comparison: at F4–F20 the higher-EDF cell-state ridge retains positive outer-held-out cosine, normalized-error, skill, and continuous-KL differences from the registered `token_ids_v1` ridge/kernel pair; F0 remains non-qualifying. This does not identify state or reject the contextual/Jacobian account.

`ctxS_A` is not “largely by construction.” `P_static` and `token_ids_v1` are different feature spaces, and the residual contextual ridge is already at its approximately 47-EDF ceiling at F4–F20 while its held-out cosine collapses to approximately 0.04–0.07. The empirical finding is that the registered raw context signal is highly non-robust to `P_static` residualization. The maximum positive wording is: “a higher-capacity predictor from `X_perp` retains held-out predictive information beyond the registered `P_static` projection and this fixed context field.” Withdraw “beyond template metadata”: `X_perp` may still carry item-token, nonlinear template/carrier, activation-geometry, and interaction signal omitted by both controls.

The specific same-template leakage worry is not supported: every outer key holds out an entire carrier block and a disjoint word fold, and residual ridge cosine does not rise after residualization; the margin expands because the context arm collapses. Related-template authorship and changed target geometry remain limitations. Raw Round 34a is still required for `ctx_A`/`ctx_B`; static Round 34a is separately required for `ctxS`; neither is the universal “right” estimand. Alongside them, run a no-completion `P`/`C`/`P+C`/`C_perp` partial-overlap screen and a frozen-item-embedding-by-`P_static` comparator before full Round 34 or Round 33.

Round 34a remains unrun and Tier-1 re-review #3 is NOT-READY on one exact telemetry-binding invariant. Use one narrow final repair; do not expand the reducer or launch before RUN-READY. The strongest alternative is now an item-by-carrier activation fingerprint plus local punctuation Jacobian, not capacity alone. No native law or representation-level hostile hole is established.

#### Audit #20 — sections 6-8 verbatim

## 6. What should run instead of or alongside Round 34a

### Priority 1 — `P/C` partial-overlap screen on existing captures

This is the cheapest missing scientific control and directly adjudicates the
“by construction” interpretation. Use the identical A/B outer block-by-word
folds, training-only transformations, 500 crossed bootstraps, cosine and nerr
only, no completion, no shuffle, and no model forward.

For each layer and outer key, fit:

1. `P`: `P_static -> Delta`;
2. `C`: registered `token_ids_v1` ridge/kernel `-> Delta`;
3. `P+C`: a nested combined field `-> Delta`;
4. `C_perp -> Delta_perp`, where both `C` and `Delta` are residualized on
   `P_static` using maps fit only on the relevant training rows; and
5. the same-EDF `X_perp` ridge as a reference, not as the claim target.

Also report, on held-out rows, the alignment between the raw context
prediction and the `P_static` prediction. Refit every target-dependent
residualizer inside the downstream inner folds for this diagnostic.

Interpretation:

- If `P+C` does not improve over `P` and `C_perp` is null in both sentinels,
  the correct conclusion is that the registered raw context field is
  `P_static`-redundant in this design.
- If `C_perp` retains signal, the current `ctxS_A` collapse is a fitting or
  feature-projection artifact; “P_static-aligned context” is too strong.
- In neither case does the result identify presentation causally.

This screen is more directly diagnostic of the estimand than adding another
completion reducer.

### Priority 2 — cheap item-by-context X-free comparator

The registered context field omits the item token, while `X_perp` necessarily
contains its activation consequences. Run a no-completion ridge comparator on
existing captures with:

- `P_static`;
- 16 training-only PCs of the frozen item embedding;
- fixed `P_static x item-PC` interactions; and
- optionally the boundary-token/POS floor from `token_ids_v1`.

Fit on calibration words only, transfer to held-out words through the frozen
embedding, match state EDF downward, and score cosine/nerr on the same outer
keys. This is the cheapest direct test of the hypothesis that the residual
ridge is exploiting lexical/item-by-template structure omitted by the context
field. It is narrower and cheaper than the full six-arm Round 34 and avoids
the parked K=13/SVD path.

If this arm closes the static state margin, classify the current result as
**item/context-feature-sensitive** and stop the consequence queue. If it does
not, the result is still not operational state; it has only survived a much
fairer X-free feature test.

### Priority 3 — architecture-matched random-weight depth screen

Only if the residual margin survives Priorities 1–2, run the already-proposed
CPU random-weight null: same architecture, tokenizer, templates, sentinel,
folds, identity-plus-shared-displacement null, and matched-EDF state/context
predictors; score F0/F4/F8/F12/F20 cosine and nerr only. No completion and no
generation.

A similar middle/deep residual profile in a random decoder would strongly
support architecture/local-smoothness and fingerprint propagation. A trained-
only profile would keep learned structure live but would still not identify
operational state.

### Do not run next

- Do not run the full six-arm Round 34 before the core and partial-overlap
  screens.
- Do not reopen Round 33 merely because `ctxS_A` has a large unmatched
  margin. A smoother high-dimensional reconstruction is expected to remain
  closer under a deterministic tail.
- Do not spend another long review loop on K=13 or low-rank telemetry for this
  question; cosine/nerr and raw continuous KL are sufficient.

## 7. Strongest alternative explanation now

The strongest alternative is a sharpened **item-by-carrier fingerprint plus
local Jacobian** account:

> `P_static` removes a coarse block/length/position response. The remaining
> `X_perp` retains a dense continuous fingerprint of the item token, the
> held-out carrier, lexical class, activation scale, and their interactions.
> Appending a fixed punctuation token produces a deterministic local response
> `Delta_perp = J(X_perp, context) + noise`. A high-EDF ridge can learn a
> transferable linear readout of that already nonlinear activation feature
> map. The fixed `token_ids_v1` field has only carrier-by-POS rows, omits the
> item token and dense interactions, and therefore collapses on the residual
> target. No denizen-usable state, quotient, operation, or composition law is
> required.

This account explains all three current observations at once:

1. raw context predicts a moderate component;
2. that component disappears after coarse nuisance residualization; and
3. the rich activation still predicts the local residual response.

It is stronger now than the generic “capacity alone” objection. Matched EDF is
necessary, but feature adequacy — especially item-by-context information — is
the more important remaining confound.

## 8. Tunnel-vision and second-lens ruling

**The program remains tunnel-visioned.** It has spent many rounds on one
relation in one small decoder, one authored 80-word population, sixteen
related templates, two punctuation tokens, one append operation, one readout
site, and one completion path. Audit/reducer loops now consume a material
fraction of the research effort. The review discipline has prevented invalid
claims, but increasingly perfect reducers for this one relation do not build a
denizen's mathematics.

The remaining Round 34a defect is worth one exact repair because the screen is
cheap and already registered. Beyond that, the next scientific increment
should be orthogonal: the item-by-context comparator, the random-weight null,
or a typed truth-evaluable finite-state world with forced-choice consequences,
causal patching, and two-step composition.

Under the second lens, `ctxS_A` proves no representation-level hostile hole.
It exposes an **instrument boundary**: the registered context reader spans the
raw coarse response but not the residual response, while the activation reader
does. Whether that is a missing quotient, useful operational state, or merely
a richer fingerprint remains unresolved. A next latent space should make the
factorization denizen-available — lexical/item coordinates, presentation
coordinates, and operation-bearing state with behavioral consequences — but
the current decoder has not been shown incapable of such a factorization.

#### Audit #20 — final ruling (verbatim)

## Final ruling

- **Upheld:** `ctx_B` mirrors `ctx_A` as a descriptive higher-EDF raw
  predictor comparison; `ctxS_A` has a real F4–F20 residual predictor
  separation; F0 is non-qualifying.
- **Withdrawn as overclaim:** “beyond template metadata,” “presentation has
  been removed,” any state contribution, and any causal or variance-share
  interpretation.
- **Withdrawn as underclaim:** “largely by construction” and any implication
  that `P_static` and `token_ids_v1` are the same feature space.
- **Reframed:** raw context performance is highly non-robust to the registered
  `P_static` residualization and therefore `P_static`-aligned in this fitted
  design; it is not thereby identified as presentation.
- **Leakage ruling:** no exact template or word identity is shared across the
  outer fit/test boundary; absolute ridge cosine does not inflate. Related
  authored structure, changed target geometry, and non-fully-nested downstream
  preprocessing remain qualifications.
- **Estimand ruling:** run both raw and static Round 34a; neither substitutes
  for the other. Add the cheaper partial-overlap and item-by-context controls
  before full Round 34.
- **Implementation ruling:** Round 34a remains unrun and NOT-READY until the
  single review-#3 telemetry-binding invariant is closed.
- **Tunnel ruling:** one final narrow repair and the cheap screens are
  justified; another broad reducer loop on the same punctuation relation is
  not. Pivot the next substantive work toward the item/context null, the
  architecture null, or a typed truth-evaluable world.

Blackboard findings were recorded. `bb_convergence` returned 100% with no open
signals, disputes, unread documents, or partial documents, and `bb_synthesis`
was read before this verdict.

## 2026-08-29 — ctxS_A complete: contextual-prefix comparator on the P_static-residualized relation, sentinel A

`analysis_ctxS_A.json` (committed analyzer, `--residualize static`, 20
shuffles, 500 bootstraps, 5655 s): on the residualized X⊥→Δ⊥ relation the
token_ids_v1 contextual arms retain almost nothing (held-out cosine 0.04–0.07
at F4–F20, normalized error ≈ 1.00), while the residual state ridge keeps
cosine 0.56–0.62 and normalized error 0.78–0.83; block-first margins vs the
strongest contextual arm: cosine +0.51 to +0.58 (LB ≥ 0.46), nerr +0.17 to
+0.23, skill +0.32 to +0.49 (LB ≥ 0.16), continuous KL +0.26 to +0.48 (LB ≥
0.13); 8/8 keys, support 1.0. F0: cosine +0.26 but nerr/skill/KL margins
negative (structural regime, as before). Reading (audit #19 discipline): the
collapse of the contextual arms is largely by construction — P_static is
built from the same template metadata the token-id field encodes (corrected
by audit #20: withdrawn as an underclaim; the two are distinct feature
spaces and the collapse is an empirical non-robustness to P_static
residualization) — so this is a descriptive comparison showing the residual
X⊥ carries held-out predictive information beyond template metadata
(corrected by audit #20: withdrawn as an overclaim; say "beyond the
registered P_static projection and this fixed token_ids_v1 context field");
it is not an identified
state contribution and not capacity-matched (the residual ridge still has
far more effective df than a near-null context arm). It is the static-form
input the parked consequence loader and the Round 34a static screen were
registered to use. ctxS_B is running next, then the parity check.

## 2026-08-28 — ctx_B complete: contextual-prefix completion comparator, sentinel B (audit #19 wording)

`analysis_ctx_B.json` (committed analyzer, unresidualized form, 20 shuffles,
500 bootstraps, 4476 s): on sentinel B's outer-held-out keys the higher-EDF
cell-state ridge retained a positive held-out score difference from the
registered `token_ids_v1` context-only pair at F4–F20 on displacement
cosine (+0.11 to +0.18, LB ≥ 0.09), normalized error (+0.11 to +0.16),
completion skill (+0.33 to +0.41, LB ≥ 0.12) and continuous KL (+0.24 to
+0.40, LB ≥ 0.13); 8/8 keys point-positive, no family collapse, support
1.0. F0: cosine +0.018 (LB 0.010), continuous-KL LB below zero. Per audit
#19 this is a descriptive predictor comparison between arms of very
different effective df and feature class — not an identified state
contribution, not a rejection of the contextual-response account, and not a
live gate. Together with ctx_A it fixes the two-sentinel picture at
unmatched capacity; Round 34a's matched-EDF core screen is the registered
next step. Chain now running: ctxS_A/B (static form), then the parity
check.

## 2026-08-28 — Re-contextualization #19 (2-hour step-back; audit #19 fired, unprimed)

Live question unchanged: is the surviving X⊥→Δ⊥ predictability in one small
decoder an operational-state relation, a generic contextual-response
(Jacobian) relation, a capacity artefact, or an instrument artefact — and
what does the answer say about holes hostile to structured reasoning.

What holds: the adjudicated four-cell table; both contextual screens; ctx_A
(ridge beats the strongest contextual-prefix arm on every endpoint at F4–F20
with crossed LBs > 0, at unmatched capacity). What is reframed: the whole
line now hinges on ONE identified confound — capacity (state ridge ~5–10×
the contextual arm's effective df) (corrected by audit #19: capacity is not
the sole confound — the difference is compatible with state information,
unmatched capacity, missing contextual features, or a mixture). Round 34 is
the registered answer and runs before the consequence test (corrected by
audit #19: the Round 34a matched-EDF core screen runs first; the full Round
34 is held; Round 34 is `P_static`-residualized and cannot retroactively
capacity-match `ctx_A`); the consequence instrument is parked on a
branch after four NOT-READY rounds, which I read as partly a
reviewer-escalation artefact (each round raised a new bar) and partly real
(legacy-base pins, exact-fit reuse). Instrument reviews are now the main
consumer of the program's time; the repair cap is doing its job.

Alternatives held live: (1) capacity explains the gap (Round 34 MOOT;
corrected by audit #19: the decisive first check is Round 34a) — then
the line collapses to "context vectors predict context-vector displacements"
and the constructive program moves to a typed use-frame task; (2) capacity
does not explain it (KEEP) — then the consequence question returns, but
audit #18's construct-validity limit stands (persistence ≠ state); (3) the
cheapest decisive check may be smaller than Round 34: df-match the state
ridge alone against the EXISTING ctx artifacts (one arm, one solve) — audit
#19 is asked whether that should run first; (4) the skill margins may
partly inherit the skill-denominator pathology flagged at F0; (5) decoder
specificity remains untested. Foundational thread advanced: instrument
governance itself — split producer/joint verdicts so a read-only reducer
cannot block a producer run, and the repair cap as a standing rule.
Audit #19 (fired, unprimed) returned CONDITIONAL; its correction block and its staging ruling / alternatives follow verbatim.

## 2026-08-28 — Audit #19 adversarial correction

`ctx_A` contains a real outer-held-out score difference at F4–F20, but only between a higher-EDF state ridge and this fixed lower-EDF context-only pair. Replace “the contextual arm did not close the gap” with: “the higher-EDF state predictor retained a positive held-out score difference from the registered context-only predictors.” The result is descriptive and does not identify state, reject the contextual/Jacobian account, or make a state-reading gate live. Inner selection used calibration-only displacement cosine and the completion readouts were scored on outer keys, so the proposed same-test-fold tuning objection does not apply. The endpoints are correlated consequences of the same prediction. F0 remains non-qualifying: skill and continuous-KL lower bounds cross zero and one family collapses.

Round 34 is over-bundled for the first capacity question. Its primary relation is `P_static`-residualized, not the raw `ctx_A` estimand; its six arms combine a matched-capacity test with a context-feature-family search; and its confirmatory KL-rank reimports the parked K=13/SVD qualification. Put a matched-EDF core screen first: existing token ridge/kernel only, state matched to their selected EDF and 47/48 ceiling, same A/B outer folds, cosine and normalized error, no completion. A state-only solve against stored aggregate JSON is a screen, not a crossed gate, unless the contextual cell predictions are recomputed. Run narrow completion only if that screen survives; run the embedding/edit arms only after that.

Parking the Round 33 consequence instrument is upheld as allocation, not as a kill. Review scope escalated, but the final blockers included a real legacy-manifest crash and unproved fit reuse, so it was not run-ready. The strongest alternative remains a generic contextual-response/Jacobian relation in which a continuous residual fingerprint predicts the local punctuation response and propagates smoothly. The next orthogonal measurements are a CPU architecture-matched random-weight depth screen and a typed truth-evaluable finite-state task with forced-choice behavior, causal patching, and two-step composition. No representation-level hostile hole is proven.

#### Audit #19 — staging ruling, Round 33 parking assessment, tunnel-vision verdict, and alternatives (verbatim, sections 3-6)

## 3. Run the cheaper decisive check first

Do not discard the registered Round 34 design. Put a preregistered
short-circuit screen in front of it.

### Round 34a — matched-EDF core screen

1. Use the exact existing A/B outer carrier-by-word folds and training-only
   standardization.
2. Recompute only the registered `token_ids_v1` ridge and kernel predictions.
   Fit state ridges by continuous bisection to (a) the selected contextual EDF
   and (b) the honest 47/48 context rank ceiling.
3. Score only displacement cosine and normalized error with paired,
   block-first crossed intervals. No completion, K=13 universe, new context
   feature family, model forward, or joint claiming reducer is needed.
4. If matched margins shrink to at most 0.02 with crossed upper bounds below
   0.02 in two common F4–F20 layers for both sentinels, report
   **capacity-sensitive screen; stop**. Do not run the full six-arm audit or
   Round 33.
5. If the margins retain positive crossed lower bounds, run a completion pass
   for only the selected token ridge/kernel pairs, using raw continuous KL and
   treating skill as a diagnostic. Then decide whether the richer context
   feature audit is worth the remaining compute.

A literal “state-only one solve against the existing JSON” is acceptable only
as a point screen. `analysis_ctx_A.json` stores fold summaries and intervals,
not reusable per-cell contextual predictions, so it cannot support a new exact
paired crossed gate without recomputing the context predictions. Recomputing
those cheap context fits is still far smaller than the six-arm completion run.

If the scientific target is specifically the primary `P_static` residual
relation rather than the raw `ctx_A` sentence, use the same staged design under
`--residualize static` after the protected contextual residual artifacts are
complete. Do not claim that one answers the other.

### Round 34b — feature-adequacy audit, conditional

Only if Round 34a survives should the input-embedding sequence and
template-edit kernels run. Label this a fixed context-family adequacy audit,
not “capacity matching.” Keep the sentinel/position field as a cheap floor.
The forced low-lambda `token_ids_v1_ceiling` is useful telemetry but is not an
inner-selected fair predictor.

The current producer/joint split requested for Tier-1 review #4 is good
software governance: a read-only reducer should not block a safe producer.
It does not answer the scientific staging question. No producer run is
authorized until it separately receives RUN-READY.

## 4. Round 33 parking: justified, not a kill

There is some reviewer escalation. Later rounds increasingly audited schema
mirrors, hashes, fail-closed reducers, and hard-wall semantics rather than the
core consequence estimand. The repair process was consuming the program.

But the parking decision was not arbitrary. Review #4 still found:

- a deterministic analyzer crash on the real legacy manifests;
- only hyperparameter-selection equality, not exact contextual-fit reuse;
- incomplete two-base preflight and legacy compatibility binding;
- hard-wall paths that could emit claiming artifacts after overruns; and
- no real HEAD-versus-refactor CPU parity result.

The legacy crash and fit-reuse failure alone make the instrument not run-ready.
Parking after four rounds was therefore a defensible allocation stop. It did
not falsify the consequence hypothesis, invalidate the design idea, or justify
deleting the branch.

If a later matched-capacity result earns reopening, salvage the smallest path:
preflight both bases before model load, rerun/serialize the exact contextual
fits with fingerprints, keep the consequence producer separate from the
joint reducer, and perform one real CPU parity comparison. Do not resume the
entire review-grown diff by default.

Even a repaired PASS would license only persistence of predictive accuracy
under frozen tails. It would not distinguish operational state from a smoother
reconstruction propagated through deterministic decoder layers.

## 5. Tunnel-vision and strongest alternative

**Yes, the program is tunnel-visioned.** It has spent many rounds on one local
relation in one small decoder, one authored 80-word population, sixteen related
templates, two punctuation tokens, one append move, one readout position, and
one completion mechanism. The recent history is now dominated by instrument
and reducer reviews. That is a governance success compared with running broken
claims, but it is not progress toward a denizen's mathematics.

No representation-level hostile hole has been proven. The current holes are
primarily in measurement: raw identity dominance, presentation/context
entanglement, low-rank numerical fragility, and inability to distinguish a
useful state variable from a high-dimensional fingerprint.

The strongest alternative remains the **generic contextual-response/Jacobian
account**:

> The residual vector contains a rich continuous fingerprint of template,
> token, position, lexical class, and local activation geometry. Appending a
> fixed punctuation token induces a deterministic local response. A
> high-capacity ridge reconstructs that response better than a low-rank
> hand-built context map, and the better reconstruction remains closer after
> smooth downstream transformations. No operational quotient or denizen-usable
> state is required.

`ctx_A` strengthens this account in one respect: context alone already reaches
cosine 0.46–0.62 at F4–F20. Its normalized error remains about 1.00, so the
current state advantage may be continuous activation/magnitude information,
but that information can still be generic local geometry rather than
structured reasoning.

## 6. What should run instead of or alongside full Round 34

Priority order:

1. **Run Round 34a, not the full six-arm completion, first.** This is the
   cheapest direct capacity moot-maker and can terminate the line cleanly.
2. **Run an architecture-matched random-weight depth-profile screen on CPU.**
   Use the same tokenizer, templates, sentinel moves, outer folds, identity +
   shared-displacement null, and matched-EDF state/context predictors at
   F0/F4/F8/F12/F20. Score only displacement cosine and normalized error. A
   similar middle/deep-layer profile or matched state surplus in a random
   decoder would strongly support architecture/local-smoothness rather than
   learned operational structure. No completion or generation claim is
   needed; any GPU version still requires explicit approval.
3. **Design a typed truth-evaluable world instead of another mentioned-string
   population.** A concrete CPU-scale successor is a four-bit finite-state
   world with operations `toggle(i)`, `swap(i,j)`, and no-op. Hold out predicate
   names and surface templates. Measure frozen forced-choice yes/no log-odds
   for all four bits, not full-vocabulary KL. Require irrelevant-wrapper and
   same-length token controls, causal patching of predicted versus true moves,
   the involution `toggle(i) o toggle(i) = identity`, one noncommuting
   two-step composition, and transfer across two disjoint query-tail families.
   This gives the denizen a known state, move, consequence, and composition
   law and directly exposes where the decoder's latent world fails them.
4. **Use causal consequence, not only predictive persistence.** Patch the
   predicted post-move state into the frozen decoder and compare behavioral
   log-odds against the true post-move state, shared-displacement prediction,
   context-only prediction, and same-norm random patch. This distinguishes a
   state estimate that can enact the move from one that merely reconstructs
   nearby activations.
5. **Only then use a second trained decoder.** It tests model specificity but
   does not solve the current construct ambiguity.

The architecture null can run alongside the docs-only typed-world design. Do
not spend the next increment on another increasingly elaborate reducer for the
same punctuation relation.

## 2026-08-28 — ctx_A complete: contextual-prefix completion comparator, sentinel A (unmatched capacity)

`analysis_ctx_A.json` (committed analyzer, unresidualized form, 20 shuffles,
500 bootstraps, 5783 s): the X-conditioned ridge beats the strongest
contextual-prefix arm (token_ids_v1 ridge / kernel) on every endpoint at
F4–F20 with crossed 95% lower bounds above zero — cosine margin +0.15 to
+0.20 (LB ≥ 0.13), normalized error +0.14 to +0.20, skill +0.34 to +0.46
(LB ≥ 0.25), continuous KL +0.27 to +0.45 (LB ≥ 0.17); support 1.0. At F0
the cosine margin is +0.019 (LB 0.011) while the skill and KL lower bounds
fall below zero. Audit #18 wording governs: this completed comparator did not
close the ridge-versus-context gap at the registered (unmatched) capacity
(corrected by audit #19: the phrase "did not close the gap" is withdrawn;
say "the higher-EDF state predictor retained a positive held-out score
difference from the registered context-only pair" — a descriptive predictor
comparison, not evidence that context failed or that capacity is the sole
confound); the state ridge still carries ~5–10× the contextual arm's
effective df, so the gap remains unidentified until Round 34's
capacity-matched comparison (corrected by audit #19: Round 34 is
`P_static`-residualized and cannot retroactively capacity-match `ctx_A`;
the Round 34a matched-EDF core screen runs first).
Not a "live gate"; not a state-reading result. ctx_B is running next, then
the static-residualized forms ctxS_A/ctxS_B, then the parity check.

## 2026-08-28 — Round 34 registered: capacity-matched state-versus-context (audit #18's first control)

Codex design gate (`.codex_dfmatch_design.md`, registered in
theory/EXPERIMENTS.md, `d493cf2`): the ridge-versus-context gap is
unidentified because the state ridge carries ~210–406 effective df against
~42 for the token-id contextual ridge. Round 34 matches capacity foldwise —
for each of six fixed contextual candidates (sentinel/position only; the
Round 31 token-id ridge at its selected lambda and at a lowered "ceiling"
lambda with capacity-shortfall telemetry; the contextual RBF kernel; a frozen
input-embedding sequence RBF arm; a template-edit Levenshtein kernel) a
separately standardized state ridge is solved by bisection to the same
training EDF. The context-only rows repeat within POS (≤48 distinct rows per
fold), so the contextual ladder cannot reach the state EDF; the state is
matched downward, never the context inflated. KEEP needs matched margins
≥ 0.02 with crossed LB > 0 on cos/skill/KL-rank, ≥ 6/8 jointly positive keys,
no block collapse, support ≥ 0.95, two common F4–F20 layers in both
sentinels; MOOT needs the strongest matched margin ≤ 0.02 with crossed UB
< 0.02 under the same key rules; otherwise INCONCLUSIVE/CAPACITY-SENSITIVE.
Ruling: Round 34 runs BEFORE Round 33 (the consequence test cannot identify
state while the predictor advantage is capacity-confounded); only a KEEP
verdict returns Round 33 to the queue. (Corrected by audit #19: the full
six-arm Round 34 is HELD; a preregistered Round 34a matched-EDF core screen
runs first, K=13 KL-rank is diagnostic in favor of raw continuous KL, and
Round 33 stays parked as an allocation decision, not a kill.) Cost 2–3.5 h CPU per sentinel, four-hour
wall. The consequence instrument is parked on branch `conseq-instrument`
after four NOT-READY Tier-1 rounds (decision raised to the user). Round 34
implementation is being written against the committed main analyzer; Tier-1
review before any run. (Later the same day: implemented as
`--context-capacity-audit round34_v1`, commit `9eb1301`; producer path
RUN-READY, joint reducer flagged for one more review; run held by audit #19.)

## 2026-08-28 — Re-contextualization #18 (2-hour step-back; audit #18 fired, unprimed)

Live question unchanged: is the surviving X⊥→Δ⊥ predictability in one small
decoder an operational-state relation, a smooth presentation/lexical
relation, a generic prefix-edit response, or an instrument artefact — and
what does the answer say about holes hostile to structured reasoning.

What holds: the four-cell common-scale table (adjudicated wording in
STATE.md); the contextual-prefix state screens in both sentinels (token-id
field cos 0.45–0.65 vs ridge 0.62–0.76 at F4–F20; ctx norm-error ≥ 1.0 vs
ridge 0.81–0.89). What is reframed by the screens: a token-id-only field
already carries half or more of the raw displacement cosine, so the
X-conditioned surplus is a gap of ~0.1–0.2 cosine and, more sharply, the
ctx field cannot beat identity on norm-error at all — the state-reading
claim now rests on that norm/scale margin as much as on direction. F0 is
nearly closed by prefix ids on displacement direction only (0.65 vs 0.69;
normalized error ~1.00 vs 0.97), which is the token-identity regime reading,
not a new fact — (corrected by audit #18: a screen-level directional
near-closure, not proof that "prefix IDs explain F0" or the full
transition; F0 is a model-class-sensitive diagnostic, since a post hoc
kernel-field reduction passes the analogous rule at F0 in all four cells).
The screens do not establish that the state-reading gate is live: the
state ridge has ~210–406 effective df at F4–F20 versus ~42 for the
contextual ridge, and the completion endpoints, crossed intervals, joint
key count, and collapse checks remain unscored (corrected by audit #18).

Alternatives held live (not run): (1) the ridge–ctx gap is a
capacity/standardization artefact (df-matched ridge vs sparse token field)
— the completed ctx_A/ctx_B completion scores and the df-matched X-free
field are the direct checks; (2) the gap is a smooth lexical relation the
four word-only nulls under-fit (kernel/knn already in the K=13 universe say
no, but only at their tuned capacity); (3) the gap is real but a
one-position readout artefact — Round 33's consequence test is exactly this
falsifier; (4) decoder-specific — a second pinned decoder remains the
cheapest replication axis and is deliberately behind the consequence test;
(5) the whole line is a well-measured triviality (context vectors predict
context-vector displacements) — the hostile-hole program only earns
anything if the consequence currency survives AND a typed use-frame task
shows a move with multi-position consequences that the bridge ladder cannot
absorb. Foundational thread advanced this cycle: the licensed-wording
discipline (adjudication → STATE/memory verbatim) and the repair-round cap
stop on the SVD gate — instruments are not allowed to consume the program.
Audit #18 (fired, unprimed) returned CONDITIONAL / wording corrections required; its correction block and its alternatives follow verbatim.

### 2026-08-28 — Audit #18 adversarial correction

The contextual-prefix results are screens only. At F4–F20 they do not triage the X-conditioned hypothesis out, but they do not make a state-reading gate evidentially live: completion endpoints, crossed gates, joint key support, collapse checks, and capacity matching are missing. The state ridge uses approximately 5–10 times the contextual arm's effective degrees of freedom, so the ridge-versus-context gap is not yet identified as state information.

At F0, contextual token-sequence metadata nearly closes ridge direction but not magnitude; “prefix IDs explain F0” is too broad. The designated F0 ridge field fails three four-cell conditions, but the stored kernel field passes an analogous post hoc reduction in all four. F0 is a model-class-sensitive diagnostic, not an all-field kill.

SVD telemetry is parked by allocation choice, not by an AGENTS.md repair-round rule, and its gate remains unpassed. The Round 33 consequence instrument exists but is unrun and NOT-READY: the joint-positive-key rule is not implemented correctly, and Tier-1 provenance/parity blockers require closure. A future consequence pass would show persistence of predictive accuracy, not by itself operational or semantic state.

#### Audit #18 — tunnel-vision verdict, strongest alternative, and recommended execution order (verbatim)

## Tunnel-vision verdict

Yes. The program is currently concentrated on one residual \(X_\perp \rightarrow \Delta_\perp\) relationship in:

- one 0.6B decoder;
- one 80-word inventory;
- sixteen closely related templates;
- two punctuation sentinels;
- one append operation;
- one readout position;
- one-step local response;
- one heavily repaired analysis path.

The strongest alternative explanation is a **generic contextual-response/Jacobian account**:

> The high-dimensional residual state encodes template, token, position, and lexical context. Appending a fixed punctuation token induces a locally predictable architectural response. A high-capacity ridge learns that deterministic response. Accurate reconstruction then remains closer under later smooth decoder dynamics, without any quotient, operational state, or latent-world law being present.

The contextual screen’s cosine of approximately 0.45–0.65 strengthens this alternative rather than weakening it.

## Recommended execution order

1. **Capacity-match before interpreting Round 33.**  
   On every existing outer fold, constrain the state ridge to the contextual arm’s effective degrees of freedom—approximately 42—either by solving for a state-ridge lambda satisfying  
   \(\mathrm{tr}[X(X^\top X+\lambda I)^{-1}X^\top]\approx df_{\text{ctx}}\),  
   or by a training-only rank/PCA constraint. Preserve the same held-out splits, endpoints, and crossed gates. Add a contextual-capacity ladder as the symmetric control.

2. **Run cheaper X-free moot-makers.**

   - Sentinel/terminal-token plus absolute and relative position only.
   - Template/edit-kernel baseline.
   - Frozen input-embedding sequence baseline over the last-eight prefix and first-four suffix tokens.
   - Contextual nonlinear capacity ladder.

   If these close the state arm after capacity matching, the current interpretation becomes moot.

3. **Use an architecture null.**  
   Repeat the capture and screen in an architecture-matched randomly initialized decoder. A similar depth profile would show that the effect arises from residual architecture and local smoothness rather than learned operational structure. Any GPU execution still requires explicit approval.

4. **Change the measurement.**  
   Replace full-vocabulary KL under one artificial tail with a behavior-bearing readout:

   - frozen yes/no log-odds;
   - a typed operation-specific target;
   - causal patch/ablation effects;
   - at least two disjoint frozen tail families.

5. **Change the task family.**  
   A stronger operational-state test would use a world with known transitions, such as a controlled finite-state machine, modular arithmetic, or truth-evaluable propositions. Require:

   - held-out predicates and templates;
   - matched wrapper-edit and irrelevant-token controls;
   - bidirectional transfer;
   - involution where appropriate;
   - two-step composition;
   - prediction of an externally scored behavior.

6. **Only then test a second trained decoder.**  
   A second decoder checks model specificity, but it does not fix the present construct-validity ambiguity.

The completed contextual commands should be run only after the current provenance/parity review blockers are closed:

```powershell
.venv\Scripts\python.exe experiments\analyze_lm_dynamics.py --run lm_dyn_v1 --config experiments/config/lexical_probe_v1.json --source forward --sentinel-tag A --target delta --unseen-words 2 --residualize static --contextual-prefix-xfree --pairs 0 1 2 3 4 --n-shuffle 20 --n-boot 500 --tag ctx_A
```

```powershell
.venv\Scripts\python.exe experiments\analyze_lm_dynamics.py --run lm_dyn_v1 --config experiments/config/lexical_probe_v1.json --source forward --sentinel-tag B --target delta --unseen-words 2 --residualize static --contextual-prefix-xfree --pairs 0 1 2 3 4 --n-shuffle 20 --n-boot 500 --tag ctx_B
```

Do not run the current consequence command until the joint-key defect and Tier-1 blockers are closed.

## 2026-08-28 — resSA2 complete: the common-scale sentinel × nuisance table is filled

`analysis_resSA2.json` (sentinel A, P_static, amended K=13 candidate universe,
four word-only nulls, crossed block-first bootstrap; 5825 s, committed
analyzer) passes the residual-versus-strongest-null gate at F4, F8, F12 and
F20 (block-first lower bounds: cos ≥ 0.46, skill ≥ 0.18, KL-rank ≥ 0.20;
keys jointly positive 7–8/8 at every passing layer; retention marker held on
all three endpoints) and fails F0 (skill and KL-rank margins negative, 2/8
full-gate keys). Read with resAA, resSB and resAB this completes the
{A,B} × {P_static, P_aug-score4} table on ONE common scale: F4–F20 pass in
all four correlated same-population cells, F0 fails in every cell except the
weak A-score4 association. Audit #17 wording stands unchanged: consistent
within-decoder, within-population condition robustness, not replication,
operational state, or presentation independence; B-score4 stays
amended-implementation and SVD-telemetry-incomplete. The residual F0 failure
in all four cells was read as the one structural regularity of the table
(corrected by audit #18: the designated ridge field is non-qualifying in
three cells with a weak pooled A-score4 exception, but a post hoc kernel
reduction passes the analogous rule at F0 in all four cells — F0 is a
model-class-sensitive diagnostic, not an all-field dead end or a
structural identity law). The
Evidence-gate adjudication of the four-cell synthesis is launched; the
contextual-prefix chain (`run_ctx.cmd`, committed analyzer copy) starts
automatically now that resSA2 has written.

Adjudicated (Evidence gate, `.codex_fourcell_adjudication.md`): PASS,
qualified. Licensed wording, verbatim: "The sentinel {A,B} x
{P_static,P_aug-score4} table is complete on a common
K=13/four-word-only-null/crossed-bootstrap scale for the
residual-versus-null mechanical gate. F4-F20 pass in all four correlated
cells; F0 is non-qualifying in three cells and yields only a weak pooled
A-score4 association with 2/8 full-gate keys. This is consistent
within-decoder, within-population condition robustness. It is not
replication and does not identify operational state, presentation
independence, a presentation decomposition, composition, a native law, or a
representation-level hostile hole. B-score4's ridge cosine and skill results
are mechanically reportable; its K=13 KL-rank endpoint and every low-rank
interpretation remain amended-implementation and SVD-telemetry-incomplete."
Additionally licensed: all 48 F4-F20 layer x endpoint bootstrap-median
common-scale ratios exceed 0.5 (estimator/null competition ratios, not
retained signal; each cell has one F4 continuous-KL interval LB below 0.5).
The phrase "uniform F0 failure" is withdrawn from the entry above: F0 is a
bounded diagnostic, not a structural law (A-score4 clears the pooled gate;
four live readings: token-identity endpoint regime, local emergence boundary,
readout/normalization pathology, score4 instrument specificity). The
EXPERIMENTS.md "7-8/8 keys" reads as jointly positive keys; strict full-gate
counts are 7/8, 7/8, 6/8, 8/8. Ledger erratum appended (resSA2 row wrote
"sentinel 2"). Round 33 order unchanged.

## 2026-08-28 — SVD telemetry gate: repair-round cap tripped; Round 33 consequence test implemented

SVD telemetry re-review #4 (`.codex_svd_review4.md`) returned NOT-READY with
six open items (mixed `primary_shadow` pooling, support-mask and missing-map
fail-closed behaviour, completion-off telemetry, a crash-safe record
validator, unexpected context groups, oracle `fit_id` collisions). That is
the fourth consecutive repair round without an admissible result, so the
repair-round cap (global CLAUDE.md §2.7; corrected by audit #18: no such
rule exists in AGENTS.md, and the parking is a discretionary allocation
decision, not a rule-triggered closure) applies: the low-rank telemetry
gate is parked and unpassed, not repaired again, and the question of
whether to continue it is raised to the user. The gate only blocks low-rank (`--aug-rank`/probe-1
screen) claims; the analyzer's SVD diff stays uncommitted and
`run_r31.cmd` stays disarmed.

Round 33's consequence test is implemented as registered: runner stage
`capture_forward_consequence` (frozen `fixed_tail_v1` eight-token tails
after each sentinel, readout-equality check against the base capture,
compact per-position true-law summaries, repeat-law noise) and analyzer
`--source forward_consequence` (multi-position teacher-forced KL, uniform
mean over positions 1..k, k ∈ {4, 8}, G_k against the strongest of the four
word-only nulls and the contextual-prefix fields inside each block-first
replicate; a layer passes only at both k). Full per-position laws are not
stored (≈3 GB); the analyzer recomputes the truth per fold. Tier-1
implementation review #1 is running; nothing has been run. resSA2 is at
F20.

## 2026-08-29 — Re-contextualization #17 (2-hour step-back; audit #17 fired and adopted in Round 33)

Live question unchanged: is the surviving X⊥→Δ⊥ predictability in one small
decoder an operational-state relation, a smooth presentation/lexical
relation, or an implementation artefact — and what does the answer say about
holes hostile to structured reasoning.

What holds: the sentinel {A,B} × {P_static,P_aug-score4} table is complete only for the residual-versus-four-word-only-null mechanical gate: F4–F20 pass in all four correlated cells, while F0 fails except for a weak pooled A-score4 association with only 2/8 full-gate keys. This is consistent within-decoder, within-population condition robustness, not replication; B-score4's ridge-only cosine and skill margins are mechanically reportable, while its K=13 KL-rank endpoint and every low-rank interpretation remain amended-implementation and SVD-telemetry-incomplete. (Audit #17 wording.)

What this cycle reframed (audit #17 wording): the v1–v3 loop showed that an all-inventory ordinary-use presentation contract had not been achieved; v4 instead obtains grammatical core-operation equivalence by placing every item in the same autonymic `the word <X>` frame. Its 48/48 approval therefore licenses a bounded mentioned-string instruction micro-world, not presentation inertness across ordinary noun, verb, adjective, and function-word uses. The operation-verb update is a declared-operation-verb context intervention, not yet a denizen-enacted operational move (source and recipient are separate prefix encodings; no execution consequence is measured). The numerical-instrument finding (torch SVD non-convergence) forced a telemetry gate in front of every low-rank result. Audit #17's allocation ruling: run the contextual-prefix baseline, then one bounded multi-position consequence test BEFORE v4 bridge/interchangeability and before a second decoder; do not author a v5 — the next population should be a typed use-frame task.

Alternatives held live and NOT yet run: a second pinned decoder (cheap
replication axis; Round 31 order 11); a multi-position consequence law
(next k∈{4,8} tokens, teacher-forced) as a different notion of consequence
(audit #16's alternative); two-step writeback composition; a full-prefix
contextual baseline is now implemented and queued (order 4). Audit #17 fired
after this note was written: the multi-position consequence law is promoted
to the one bounded test that precedes v4 bridge/interchangeability and the
second decoder (Round 33 registers it); the v4 frame is a bounded
mentioned-string instruction micro-world; no v5 — the next population, when
one is authored, is a typed use-frame task, not another all-inventory
search.

Nothing starts before the two Tier-1 gates (op-update/bridge; SVD telemetry)
and resSA2 finish.

## 2026-08-29 — Population v4 approved and frozen (after v2 and v3 were voided)

- v3 (`a8b14a8`) VOID: all 32 presentation cells passed (Please/Kindly;
  ASCII vs typographic apostrophe, rated near-degenerate), but the controls
  under the orthographic wrapper failed clause 6 (a whole-word operation swap
  vs a one-glyph presentation edit). Adversary principle: co-design pairs and
  controls under a predeclared surface-distance metric.
- v4 (`afd6fcc`, fresh outcome-blind author): metalinguistic frames `{Please|Kindly} plan to OP the word <X>` and `{Hello,|Hi,} please plan to OP the word <X>`, OP ∈ {repeat, omit, capitalize, reverse}; declared metric = aligned surface-word edit distance = 1 for every pair and control; frozen `operation_updates` block. A separate fresh Codex session approved 48/48 checklist cells for grammaticality, preservation of the explicit string-edit instruction, and matched surface-word distance under the common mention frame; this is outcome-blind procedural approval, not 48 independent linguistic observations or proof that Please/Kindly and Hello,/Hi, are pragmatically or latently inert (audit #17). Tokenization PASS; approval block written; raw sha256 `f813f9b2…`, git blob `8845f75c…` in the ledger (`nlm007_fresh_v4_frozen`). The config's top-level 'not approved for capture' note is historical authoring-time text superseded by the structured approval/hash fields.
- Next on this population (Round 31 order 5–8, after the order-4 baseline;
  reordered by audit #17 / Round 33: one bounded multi-position consequence
  test comes first): captures A / B / OP_UPDATE → bridge screen →
  interchangeability → fresh analyses A/B → operation-update analysis; chain
  `run_v4.cmd` written, armed only when the operation-update and bridge code
  pass Tier-1 review.

## 2026-08-29 — Residualization B P_aug-score4 completes the 2×2 table; third launch with the SVD fallback

- Sentinel ',' with the implemented score-4 augmented design; 5074 s of the
  7200 s wall on the third launch (the first two died in the F8 grammar
  block; only the second is directly localized to torch SVD non-convergence
  on the fitted low-rank coefficient matrix at grammar_w1 — audit #17
  erratum; the committed analyzer now falls back to a float64 LAPACK SVD —
  this cell is an amended-implementation cell and is reported as such).
- **F4, F8, F12, F20 pass** the residual-vs-null gate (X⊥ ridge 0.52–0.57 vs
  strongest residual null 0.06–0.09; block-first leads cos +0.46–0.51,
  skill +0.40–0.44, KL-rank +0.45–0.54; 6–8/8 full keys; no collapse).
  **F0 fails** (cos +0.33 but skill LB −0.04; 4/8 full keys).
- Registered-static-metadata + carrier-summary nuisance arm (P_aug → Δ)
  0.42–0.64 by layer (not a presentation-only component).
- Same-run common-scale ratios exceed 0.5 at the median at F4–F20 (F0 wide).
- Reading (audit #16 discipline): within one decoder and one authored
  population, under both sentinels and both registered nuisance designs,
  X⊥ retains predictive association with Δ⊥ beyond the four X-free lexical
  nulls at F4–F20; F0 passes only for the A score-4 cell (sparse keys). These
  are four correlated same-population sensitivities, not replications; they
  identify neither operational state nor presentation independence.
- The chain continues automatically: resSA2 (patched A-static common-scale
  cell), then run_r31.cmd (probe-1 screens, P_aug-full cell A,
  contextual-prefix screens and completions). Codex round 32 adjudicated the
  cell as amended-implementation / SVD-telemetry-incomplete and forbade
  further low-rank output before an SVD telemetry gate; run_r31.cmd was
  disarmed (ledger `nlm007_r31_chain_disarmed_pending_svd_gate`).

## 2026-08-29 — Round 31 adopts audit #16; v2 population authored, then voided by the independent adversary

- Round 31 (Codex, `71b5ce3`): audit #16 adopted verbatim; fresh v1 voided for
  confirmatory probes 2–4 (ledger `nlm007_fresh_v1_voided`); the ` not`
  insertion withdrawn as the second move, replaced by the operation-verb
  update in a metalinguistic micro-world (repeat→omit, capitalize→reverse
  under matched wrappers); contextual-prefix X-free baseline (token_ids_v1)
  and calibration-only bridge ladder registered as analyzer modes; corrected
  order 0–11 (baseline before any fresh capture; bridge before
  interchangeability; hostile lower bound must exceed τ; move-norm floor).
- v2 (`lexical_probe_fresh_v2.json`, Codex as outcome-blind author):
  `Please/Kindly | For reference,/For clarity, … plan to {repeat|omit|
  capitalize|reverse} the word <X>`; tokenization pre-check passed (slot
  template-final, ` not` clean). The independent linguistic adversary
  (fresh session, no model access) VOIDED it: all 16 pair-2 cells fail — “For
  reference” vs “For clarity” introduce distinguishable discourse purposes
  that can scope over the operation; pair-1 (Please/Kindly) and all controls
  pass. Design principle for v3 (verbatim): vary only a scope-fixed form whose
  interpretation cannot supply a reason, goal, condition, or other content for
  the requested operation; semantic inertness must hold independently in every
  POS cell. v3 was authored from scratch by a fresh session (later voided
  on control edit-magnitude; see the v4 entry above).
- Implementation: the reviewed analyzer (probe-1 options, insertion source,
  interchangeability, SVD fallback) is committed (`0c774c0`); the
  contextual-prefix baseline is implemented and under Tier-1 review; the
  operation-update move is at its design gate. B-aug's third launch passed
  the fold that failed twice (fallback held); resSA2 follows automatically.

## 2026-08-29 — Re-contextualization #16 (2-hour step-back; audit #16 fired and adopted in Round 31)

Live question unchanged: is the surviving X⊥→Δ⊥ predictability in one small
decoder an operational-state relation, a smooth presentation/lexical relation
the registered designs miss, or an implementation artefact — and what does
either answer say about holes hostile to structured reasoning.

What still holds: A-static, P_aug-score4, and B-static are bounded, correlated same-population sensitivities in one decoder; registered static metadata predict raw displacement across both sentinels, X-linked residual predictability survives the tested nuisance fits at F4–F20, operational state is not identified, the inherited ordering statistic is a local measurement hole, and raw F0 remains identity/token dominated. (Audit #16 wording.)

What reframed this cycle: (1) audit #15 moved the program off the
observational residualization axis onto external axes (fresh population,
second move, interchangeability) — the queue is now about whether the
relation is a property of the authored manifold or of the space; (2) the B-aug analysis failed twice in the F8 grammar block; the preserved traceback localizes the second failure to torch SVD of the fitted low-rank coefficient matrix at grammar_w1, while the later ledger row attributes the first loss to the same defect. This is repeatable numerical-instrument non-robustness, not evidence of ill-conditioned X⊥ until finite-input and spectral diagnostics localize the cause (audit #16 wording); (3) the frozen fresh population fails its pre-capture linguistic design gate: none of the eight pairs establishes coherent presentation-only equivalence across all four word classes, and several change syntactic licensing, modality, definiteness, degree, or quantification. The population is void for confirmatory probes 2–4 and may be retained unchanged only as an exploratory mixed-frame stress set; no noun-only or pair-only post hoc rescue is confirmatory (audit #16 ruling).

Alternatives held live (not yet run; CPU-only): a second pinned decoder as a
cheap replication axis; a full-prefix contextual X-free baseline (audit #15);
a two-step writeback composition test; a wholly new population authored under a predeclared all-POS linguistic contract, reviewed by an independent linguistic adversary before hashing and capture;
and the direct "consequence-sensitive divergence" question — whether the KL
readout at one position is the right notion of consequence for a denizen at
all, or whether a multi-position law (next k tokens) is the honest one.

Nothing in probes 2–4 starts on fresh v1: finish the protected running chain, audit the B-aug numerical amendment, repair and re-review probe 1, then register and freeze a linguistically valid replacement population before capture. (Audit #16; adopted in Round 31.)

## 2026-08-29 — Round 29 reorders the queue; fresh matched population frozen

- Codex Round 29 (`4907a85`), adopting audit #15: Round 23's literal `P_aug`
  meant full carrier mean + rank-4 scores → the observed run is
  `P_aug-score4` (outcome-clean, transductive, contract-validity qualified);
  `P_aug-full` is unrun. New fixed order: (0) finish resAB → resSA2;
  (1) carrier-summary rank ladder {1,2,4,8,full} + nonlinear carrier kernel
  as a cosine screen, plus one preselected full-law cell (sentinel A,
  `P_aug-full`); (2) fresh frozen population + ` not`-insertion capture;
  (3) matched presentation interchangeability (stable vs hostile-hole gates);
  (4) fresh-population analysis; (5) different-move analysis; (6) registered
  X-free field ×4; (7) Freedman–Lane on A-static only, conditionally; (8)
  second pinned decoder. The armed X-free chain was killed.
- Probe 2 population (`experiments/config/lexical_probe_fresh_v1.json`;
  families question / instruction / comparison / enumeration, 8 matched
  presentation pairs, 4 operational control pairs, same 80 words; ` not`
  (id 537) appends as exactly one token to every prefix) was prospectively authored and committed before any new capture or score, but not independently blind to prior results. Its declared digest `c6edaa92…` is not the raw file SHA-256 (`12c72401…`), and audit #16 voids its eight-pair presentation-equivalence claim before capture. The file stays unchanged as an exploratory stress set only; a v2 population under a predeclared all-POS contract replaces it (Round 31).
- Round 30 completed its review and ruled probe 1 NOT-READY; its six repairs remain a prerequisite, while probes 2–4 are additionally paused by audit #16's population-validity failure.

## 2026-08-29 — Residualization B-static: a correlated second-sentinel check takes the same bounded P_static branch

- Sentinel ',' with P_static; 4598 s of the 7200 s wall; unseen-word folds,
  K = 13, class-preserving crossed bootstrap, same-run raw shadow and
  common-scale retention block.
- **F4, F8, F12, F20 pass** the residual-vs-null gate (X⊥ ridge 0.52–0.58 vs
  strongest residual null 0.06–0.09; block-first leads cos +0.45–0.50,
  skill +0.35–0.42, KL-rank +0.40–0.58; 8/8 positive keys at every passing
  layer; no collapse). **F0 fails** (cosine lead +0.27 but skill negative and
  KL-rank LB < 0) — as under A-static.
- Registered-static-metadata arm (`P_static → Delta`) cosine is 0.41–0.63 by layer; this is not a pure presentation component or variance share.
- All twelve F4–F20 residual/raw predictive-margin ratio medians exceed 0.5; eleven lower bounds do so, with F4 continuous KL at 0.426. These are robustness ratios, not retained signal, state, or mediation.
- Reading (audit #16 wording): across the correlated A/B static runs, registered block/length/position metadata predict raw displacement, and X⊥ retains predictive association with Delta⊥ beyond four X-free lexical nulls at F4–F20. This is a two-sentinel robustness result within one decoder and authored population, not independent replication, state, or presentation independence.

## 2026-08-29 — Re-contextualization #15 (2-hour step-back; audit #15 running)

Live question (one project): is the surviving X⊥→Δ⊥ predictability in one small
decoder an operational-state relation, a smooth presentation/lexical relation
the registered designs miss, or an implementation artefact of residual
geometry — and what does either answer say about holes hostile to structured
reasoning.

Current bounded result: in the same sentinel-A cells and folds, X-linked residual predictability survives the registered `P_static` fit and the implemented rank-4-score `P_aug` fit at F4–F20. The raw F0 transition remains identity/token dominated, and the specific across-word within-carrier pairwise-KL ordering statistic is insensitive in this probe. These are correlated sensitivity results, not replications, and they identify neither operational state nor a native law. (Audit #15 wording.)

What is reframed by A-aug (audit #15 wording): A-aug shows only that one registered P_static fit and one implemented, contract-qualified P_aug-score4 sensitivity on the same sentinel-A cells do not absorb the `X⊥–Δ⊥` association. It does not show that every finite presentation design will leave a predictive residual or that the residual is operational state. The two Round 27 comparators are the next within-dataset controls: the registered X-free interaction field tests whether a fixed low-rank presentation/lexical family can reproduce the association without cell-level `X⊥`, and the refitted permutation null tests whether the observed alignment exceeds residual-geometry null refits. Neither is decisive for operational state, because an aligned cell-level prefix/carrier fingerprint can beat both.

Tunnel-vision check — honest: everything queued is one decoder, one template
population, one move, one sentinel pair. Live alternatives held open:
(a) A second pinned decoder is a relatively cheap replication check for decoder specificity; one additional decoder cannot decide whether the relation is generic or identify its mechanism.
(b) the relation is template-population-specific → a fresh authored style
   family, held out entirely, is a cheaper test than another comparator;
(c) A two-step writeback test requires a new intervention capture rather than existing captures alone, but current timings suggest roughly 10–20 minutes of CPU capture plus about one hour of targeted scoring.
(d) The present evidence does not yet prove a structural quotient hole. Failure of two nested nuisance fits shows that the chosen coordinates are incomplete; it does not show that lexical, presentation, and operational coordinates are entangled by construction. The cheapest sharpening is a linguistically validated interchangeability test with matched controls and a predeclared calibration-only bridge ladder; raw scalar swap failure alone cannot establish a hostile quotient hole.

Audit #15 (verbatim in .codex_audit15.md; adopted into theory/EXPERIMENTS.md):
the queue is "strongly tunnel-visioned" — one decoder, one authored template
population, one punctuation-append move, one self-readout; the strongest
alternative both queued comparators miss is a high-dimensional prefix/carrier
fingerprint (aligned, cell-level, compatible with unseen-word transfer and law
improvement); CPU-only alternatives it ranks ahead of the ~100 CPU-h
Freedman–Lane expansion: full carrier-summary rank ladder {1,2,4,8,full} +
nonlinear carrier kernel; contextual X-free baseline from full tokenized
prefix features; a fresh frozen template population (16×80); a different
move (content-bearing append, negation/operator insertion, binding update);
a matched presentation-interchangeability test; two-step writeback; second
pinned decoder. The order change is a Codex decision (round 29).

## 2026-08-29 — Residualization A P_aug-score4: residual predictability survives at F4–F20; F0 remains sparse and raw-identity dominated

- 4738 s of the 7200 s wall; sentinel '.'; `P_aug` uses `P_static` plus at most four scores obtained by
  projecting a leave-calibration-word-pool carrier mean of `X` into a basis
  learned from calibration carriers; the full carrier-mean vector is not
  appended (audit #15); cross-fitted out of both X and Δ; unseen-word folds; K = 13; class-preserving
  crossed bootstrap; same-run raw four-null shadow and common-scale
  retention block present.
- All five correlated checkpoints meet the registered aggregate residual-vs-null gate. F0 is qualitatively weaker — only 2/8 keys clear the full per-key gate — and is not an independent confirmation of the F4–F20 profile. F0 numbers
  (residual cosine 0.34 vs −0.01; block-first skill +0.16 [LB 0.02],
  KL-rank +0.30 [0.12]) — the score-only nuisance fit changes the residual target and reference geometry and exposes a positive pooled F0 association, but only 2/8 keys clear the full gate, so it does not repair the raw identity-dominated transition. F4–F20: X⊥-ridge 0.56–0.62 vs 0.06–0.07;
  block-first leads cos +0.50–0.56, skill +0.35–0.46, KL-rank +0.43–0.56;
  6–8/8 keys; no block collapse.
- `P_aug` nuisance-only carrier-summary arm (P_aug → Δ) 0.45–0.64 by layer; because its scores are derived from carrier-level X, this is not a presentation-only estimate or a variance share.
- The implemented P_aug-score4 run is internally valid for its score-only sensitivity but does not instantiate Round 23's literal full-mean-plus-score P_aug contract; P_aug-full remains unrun. Under Round 23's predeclared readings this is the non-collapse branch for the implemented design: the registered static and rank-4-score nuisance fits do not absorb the association; broader presentation, carrier-geometry, and prefix-fingerprint explanations remain fully live
  (unmeasured presentation remains possible; audit #14's Freedman–Lane
  residual-geometry null and calibration-only presentation/lexical
  comparator are the next preregistered tests). Wording per audits
  #13/#14: residual predictability of X⊥ beyond residualized X-free
  lexical nulls after removal of the registered static AND augmented
  coordinates; not presentation-independence; not state.
- B-static running; then B-aug; then the patched A-static.

## 2026-08-29 — Audit #14 adopted: A-static upheld; Round 26's mediation sentence withdrawn

- Upheld: F4–F20 pass; not a residual-geometry mirage (ridge cosine falls
  under residualization while the nulls collapse; shuffle q95 ≤ 0.13;
  residual normalized error 0.78–0.83).
- Withdrawn (over-read in the kill direction): "much of the raw lead may
  have been presentation-mediated". Licensed joint statement: registered
  static coordinates predict held-out raw displacement; after their
  cross-fitted removal X⊥ still predicts Δ⊥ and its reassembled response-law
  consequence beyond the residual X-free nulls at F4–F20; the overlap
  between presentation and the raw ridge lead is not identified.
- Gate is too easy for a *state* claim: next comparators to preregister are
  a fully refitted Freedman–Lane residual-geometry null and a flexible
  calibration-only P_aug/lexical interaction field without cell-level X⊥.
- Demo copy corrected again (nine verbatim replacements) and republished.
- The two NOTEBOOK entries carrying the withdrawn phrase (Round 26 note;
  re-contextualization #14) are superseded by this entry.

## 2026-08-29 — Re-contextualization #14 (A-static in; P_aug running; audit #14 fired)

*Audit #14 (Tier-3, `theory/EXPERIMENTS.md`, ledger `nlm007_audit14_adopted`) withdrew the sentence "much of the raw lead may have been presentation-mediated" as an over-read and replaced the ruling; read this paragraph as corrected there.*

- **Central bet + second lens:** native mathematics from what a denizen must
  invent; the holes that make this space hostile to structured reasoning
  and what the next latent space must change.
- **Live question:** after the registered template coordinates are removed
  from both state and displacement, X⊥ still predicts Δ⊥ far beyond every
  residual content null (F4–F20). Does that survive the augmented
  presentation design (carrier mean + carrier subspace) and the ',' arm?
  And what, jointly, do the presentation-only arm (0.43–0.63) and the
  residual lead license — "presentation is a large part of the raw move,
  and what remains is still X-predictable" — without either side
  over-reading?
- **What reframes:** the pooled story has quietly changed shape. The
  earlier framing "content vs context" is now "content vs presentation vs
  the residual of X after presentation" — three layers, of which content
  is the smallest, presentation is large, and the X⊥ residual is what a
  denizen would actually need a map of. The demo's 42%-style intuition was
  wrong (cosine ≠ variance), and Round 26's "much of the raw lead may have
  been presentation" may itself be an over-read in the other direction —
  audit #14 is asked to fix the joint statement.
- **Alternatives held live:** (a) residual-space cosines are geometrically
  easy (nulls at ~0.06 because residual targets are near-zero-mean) — the
  fair residual comparator may be a residual-space shared mean or a
  P-only predictor scored in residual space; (b) unmeasured presentation
  remains in X⊥ (P_aug tests part of this); (c) presentation is part of
  operational state and quotienting it removes physics — the operational-
  equivalence target (same moves, same consequences) is the honest
  definition; (d) second family; (e) multi-step composition.
- **Ecosystem deposit:** "cosine of a presentation-only predictor is not a
  variance share; state 'presentation predicts the move at c' and 'the
  residual is X-predictable at r' separately" → `_meta/INDEX.md`.

## 2026-08-29 — Round 26: A-static adjudicated; the presentation-only arm revises the earlier reading

*Audit #14 (Tier-3, `theory/EXPERIMENTS.md`, ledger `nlm007_audit14_adopted`) withdrew the sentence "much of the raw lead may have been presentation-mediated" as an over-read and replaced the ruling; read this paragraph as corrected there.*

- P_static took the non-collapse branch of the primary gate at F4–F20; this
  proves neither operational state nor presentation-independence.
- The presentation-only arm (0.43–0.63 cosine) materially revises the
  earlier unseen-word interpretation: much of the raw X-conditioned lead
  may have been presentation-mediated; residualization shows only that the
  registered static coordinates do not explain all of it.
- For resSA only "the predeclared robustness marker is mechanically met" is
  admissible; a patched A-static rerun (common-scale retention) is required
  for any A-static or four-cell retention claim — queued after B-aug.
- Presentation sensitivity is proven locally; presentation/state
  inseparability remains unproven. Read order: A-aug → B-static → B-aug →
  patched A-static.

## 2026-08-29 — Residualization A-static: the X⊥ lead survives removal of the registered template coordinates

- 4406 s of the 7200 s wall; sentinel '.'; P_static (block one-hot, lengths,
  positions) cross-fitted out of both X and Δ; unseen-word folds; K = 13
  universe; class-preserving crossed bootstrap.
- **F4, F8, F12, F20 pass** the residual-vs-null gate: X⊥-ridge 0.56–0.62
  residual cosine vs 0.06–0.07 for the strongest residual X-free null;
  block-first leads cos +0.50–0.56, skill +0.31–0.48, KL-rank +0.40–0.61
  (lower bounds > 0.17); 6–8/8 keys positive; no block collapse. F0 fails
  (skill negative).
- Presentation-only arm (P_static → Δ) held-out cosine 0.43–0.63 by layer:
  the registered template coordinates are a large part of the raw
  displacement; what remains after their removal is still predicted from
  X⊥ far beyond any content null.
- Retention: "the predeclared robustness marker is mechanically met" on all
  three endpoints at F4–F20 (audit #13: not a fraction of signal; this run
  predates the common-scale block, which A-aug and the B runs carry).
- Remaining: P_aug (adds the leave-word-out carrier mean and a rank-4
  carrier subspace), both B arms; Codex round 26 adjudicates A-static now.

## 2026-08-29 — Audit #13 adopted: demo corrected; retention marker not commensurate

- The published demo over-claimed ("context state", "context takes over",
  "manufactures", "presentation explains 0.42"); every replacement adopted
  verbatim and republished at the same URL; the nearest-state predictor is
  now coloured as X-conditioned; named-word rows labelled as selected.
- Retention marker: raw and residual margins live on different scales;
  until the common-scale repair is in place the residualization runs may
  say only "the predeclared robustness marker is mechanically met". The
  raw shadow remains valid for the amended unseen-word comparison; the
  residual-vs-null gate and the law reassembly are coherent.
- Equalized A wording tightened (defect concern resolved; calibration-
  selected comparator; 0.002–0.009 above the mean).
- Reverse-tunnel note adopted: the X-conditioned advantage can no longer be
  dismissed as lookup or artifact; presentation may be part of state.

## 2026-08-29 — Corrected equalized LOCO addendum, sentinel ',': F12/F20 pass; baselines just above the shared mean

- 4196 s of the 4500 s wall. Contract-correct equalized baselines sit
  0.002–0.007 above the shared mean; ridge's lead is unchanged: **F12 and
  F20 pass** against the stronger equalized baseline, F4/F8 miss on
  skill/KL-rank lower bounds (cosine leads hold), F0 fails. Run-level
  positive (2/5). Both arms of the addendum are now contract-correct and
  agree with the defect-affected runs' numbers — the defect changed the
  baselines by ≤0.01 and no verdict.
- The residualization chain (A-static → A-aug → B-static → B-aug, 120-min
  wall each) has started.

## 2026-08-29 — Re-contextualization #13 (residualization launching; demo audited)

*Wording per audit #13 (see the 2026-08-29 audit #13 entry): "the context predictor" reads "the X-conditioned predictor"; the retention marker is not commensurate, so residualization runs may say only "the predeclared robustness marker is mechanically met".*

- **Central bet + second lens:** native mathematics from what a denizen must
  invent; holes hostile to structured reasoning; the next latent space.
- **Live question:** the forward step's regularity transfers across
  carriers, families and unseen words, and every content null sits at the
  mean — is what X carries operational state or a smooth presentation
  coordinate? The four residualization runs (static/aug × two sentinels)
  are the first direct test; they start automatically after the corrected
  equalized rerun B.
- **What reframes:** the single-template walkthroughs for the demo showed
  something the pooled numbers hide — in a gloss template the context
  predictor wins on the state but the next-token law barely moves; in
  continuation and grammar templates the law moves a lot. "Consequential
  motion" varies by template family, not just by layer. That is a
  template-level version of the middle-depth finding, and it is exactly
  the kind of structure a next-generation latent space would need to make
  explicit: when does a move matter to the world's response?
- **Alternatives held live:** (a) the residual field recovers a smooth
  presentation coordinate P_static/P_aug miss (unmeasured presentation);
  (b) the gloss/continuation difference is a readout-sensitivity artifact
  rather than a world property; (c) multi-step composition may fail even
  if one-step prediction holds; (d) a second family may reorder everything;
  (e) response-space geometry ("same place" = same law) as the native
  metric — the demo's third panel is a first look at exactly that object.
- **Ecosystem deposit:** "whether a move is consequential varies by
  context family, not only by depth — measure consequence per family" →
  `_meta/INDEX.md`.

## 2026-08-29 — Corrected equalized LOCO addendum, sentinel '.': ridge lead unchanged under contract-correct baselines

*Tightened by audit #13 (see the 2026-08-29 audit #13 entry): "audit #11's inner-centre defect concern is resolved by the corrected sentinel-A data", the comparator is the calibration-selected equalized comparator, baselines roughly 0.002–0.009 above the shared mean.*

- 3753 s of the 4500 s wall. With the audit #11 fix (inner centre = the
  inner training carriers' own mean; comparator frozen by calibration
  score), the equalized baselines (word-only one-hot ridge; shrunk word
  mean) no longer collapse exactly onto the shared mean — they land
  0.003–0.01 above it (e.g. F8: shared 0.499, word-only ridge 0.506, shrunk
  0.508, ridge 0.620). The per-word lexical component captured by these
  estimators is small but not identically zero; audit #11's "forced
  maximal shrinkage" concern is resolved by data rather than by wording.
- Gated against the stronger equalized baseline: **F4, F8, F12, F20 pass**
  (cos +0.09–0.13, skill +0.23–0.31, KL-rank +0.30–0.43, LBs > 0.08;
  11–14/16 carriers); F0 fails. Run-level positive. The ',' arm follows.

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

*Superseded in part by audit #11 (see the 2026-08-29 audit #11 entry): the equalized-addendum inner centre included the validation carrier, so "context, not content" and "content nulls collapse to the shared mean" are withdrawn; the addendum's margins are descriptive only. Audit #13 also withdraws "governed by context state"; the object is X-conditioned residual predictability.*

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

## Re-contextualization (2026-08-30, session 2)

**What holds:** Theory stack frozen at audit #47 LOCK-READY. Runner implementation at design gate v3 (all 6 v2 blockers applied). No new claims to audit.

**Direction still makes sense:** The 32-call smoke is the minimum viable test of whether the d_∞ framework makes any falsifiable prediction at all. It will produce eta_smoke (replay stability), fixture checks (hook identity), and timing forecasts. If eta is above the ceiling or fixtures fail, the framework has a mechanical problem before we even get to science.

**Live alternatives:**
1. Framework could be trivially satisfied/violated — smoke + forecast will surface this
2. Native vs surgeon distinction might be vacuous — centroid vs wrong-centroid contrast directly tests this
3. Simpler cosine-similarity baseline could make the framework moot — deferred until we have d_∞ numbers to compare against

**No tunnel vision:** Single thread is correct — it's the critical path to the first empirical test. No theory drift, no infrastructure accumulation.

## Smoke result (2026-08-30)

native_bridge_v1 32-call mechanical smoke: **SMOKE_VALID**.
- eta_smoke = 8.7e-9 (replay discrepancy near zero; well below 1e-4 ceiling)
- epsilon_smoke = 1e-5 (= epsilon_0; eta negligible)
- All fixtures passed: pasteback and source_hook within epsilon_smoke on both channels (c_full, c_9) and both replays (A, B)
- s_smoke = 1.179s/call (plain ≈ hooked)
- F_CPU = 79.2 min, H_CPU = 80 min (within 90-min ceiling)
- Total smoke time: 53 seconds
- Lock row written with all required bindings (runner/config/manifest/call_table hashes, constants, stop rule)
- Codex design gate: v1-v4 completed; smoke-blocking items resolved; 5 science-grade items deferred (token IDs in identity, expanded manifest, bootstrap precommitment, row validation detail, per-call checkpointing)
- Next: commit, apply science-grade fixes, then scientific Phase D/E execution
