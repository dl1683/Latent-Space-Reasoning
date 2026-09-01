# HANDLE-μ: Causal Handle Algebra in a Designed Latent World

**Status:** Locked specification (Codex Rounds 1–3b). Distance-1.
**Date:** 2026-09-01
**Runner:** `experiments/run_handle_mu.py`

## Narrative gate

> Can learned memory be divided into pieces that move between experiences,
> remain silent until the world reaches what they control, and then change
> exactly the right future?

## Distance from central artifact

HANDLE-μ is **distance-1**: it tests causal-handle algebra in a designed world
whose causal structure we control. It is NOT the central artifact (distance-0
would be causal handles in a real model's endogenous state). A HANDLE-μ pass
freezes the evaluator and nulls for later application to real models.

## World

- Grid: 7×7, procedurally generated, all objects reachable.
- Actions: north, south, east, west, use.
- Six unordered observation records: agent, key A, key B, lock A, lock B, goal.
- **Five causal handles:** two keys, two locks, goal. The agent is shared
  control context, excluded from the handle graph (including it connects
  everything and makes disjoint commutation meaningless).
- Goal activates only when both locks are open: {ℓ₁,ℓ₂} → g (higher-order).
- Keys remain held after use (simplifies paired counterfactual construction).

### Causal bijection (Amendment 3)

The hidden key–lock bijection is resampled **independently at every episode
reset**, not per level. Geometry (object positions, walls) remains level-fixed;
the causal wiring does not. This prevents layout-specific causal memorization.

## Observation

Partial visibility, Manhattan radius 2 from agent.

### Identity permutation (Amendment 1)

Sample one uniform object-to-carrier bijection at episode reset and retain it
for the entire trajectory. Maintain two distinct maps:

- `episode_carrier_map`: complete privileged mapping for all six objects,
  including invisible ones. Available to the evaluator but never to the model.
- `slot_maps[t]`: visible projection at timestep t — contains only the subset
  of carriers whose objects are currently visible.

Raw record order may still be shuffled each step as an interface-invariance
check. The event prediction head must be permutation-invariant; a flattened
carrier-position-sensitive event head is not valid.

### Record encoding (Amendment 4b)

Each visible record contains:
- **Type:** 4 classes (key, lock, goal, agent) — categorical.
- **Relative row:** 5 classes [-2, -1, 0, +1, +2] — categorical.
- **Relative column:** 5 classes [-2, -1, 0, +1, +2] — categorical.
- **Status:** 4 classes (idle, held, open, active) — categorical, type-valid
  subset only.
- **Visibility:** binary (1 = visible, 0 = not visible).

Total encoded: 5 fields per carrier. The previous 9-class position encoding
contained four unreachable classes; five classes suffice under Manhattan radius 2.

Record order and identity assignment are independently permuted per step.
No key–lock relation, causal edge, global semantic ID, or simulator adjacency
is exposed. Full visibility would make recurrence decorative — rejected.

## Behavioral object

For handles i, j and interaction context c:

    i →_c j

iff paired interventions differing only in i materially change j's
next-response distribution under c.

First-contact time:

    τ_{i→j} = min{t : TV(R_j^I(t), R_j(t)) > ε}

The claim-bearing structure is:
- Context-indexed edges
- Higher-order hyperedges ({ℓ₁,ℓ₂} → g)
- First-contact times
- Transitive influence closure
- Partial composition laws

No latent-vector metric enters a primary gate.

## Architectures

### 1. Dense typed slots (PRIMARY)
- Six recurrent slots, width 32.
- Shared record encoder and recurrent updater.
- All-pairs learned messaging.
- No architectural light cone.

### 2. Learned-sparse typed slots (SECONDARY)
- Same parameters and state width.
- Learned top-2 incoming message gates.
- No grid distance or simulator graph.
- Its learned mask is never treated as the discovered causal graph.

### 3. Flat GRU — Control A (PREDICTIVE CONTROL, parameter-matched)
- Hidden width chosen for parameter count within 5% of slot models.
- Reports as an efficiency diagnostic.
- No unnatural subvector-swap test.

### 4. Set-aware recurrent — Control B (PREDICTIVE CONTROL, performance-matched) (Amendment 5)

A single global recurrent memory with:
- Shared encoding of carrier-tagged records.
- Permutation-invariant pooling over visible records.
- One global recurrent state.
- Per-carrier decoding conditioned on a carrier query.
- Permutation-invariant event output.

Width ladder: {h_pm, 96, 192} where h_pm is recomputed from the factorized
heads (not hard-coded). Train all three model seeds at each width. Select the
first width satisfying on validation:
- Event and status macro-F1 both ≥ 0.90.
- dense_F1 − control_F1 ≤ 0.03 for both endpoints.
- Median qualification and qualification in at least 2/3 seeds.

Use validation loss only for checkpoint selection within a width. Width
selection uses the registered predictive-matching criteria. If no width
qualifies, eligibility fails.

The margin is one-sided: a control outperforming dense must not fail.

### 5. Historyless (RECURRENCE NECESSITY)
- Same architecture as dense slots but no recurrence.
- Tests whether persistent state is needed.

### 6. Direct-state deterministic oracle (PIPELINE VALIDITY) (Amendment 9)

The oracle is the deterministic simulator transition function, not a learned
MLP. It receives:
- Full episode state.
- Episode-local key–lock bijection.
- Proposed action.

Evaluate on validation episodes. Keep the ≥ 0.99 pipeline gate and report
exact accuracy diagnostically. Any miss triggers pipeline investigation, not
additional oracle training.

A dense-slot pass is scientifically stronger than sparse-only. Sparse-only
success means engineered locality, not discovered structure.

## Training

### Loss (Amendment 4)

Factorized categorical cross-entropy:

    L_record = (L_type + L_row + L_col + L_status + L_visibility) / 5
    L = L_record + L_event

- Type / row / column / status: CE over target-visible carriers only.
- Visibility: BCE over every carrier.
- Event: CE, unchanged.
- Emit class supports and confusion matrices.

Set matching is unnecessary under stable episode carriers.

### Prohibited training signals

**Explicitly prohibited:**
- Swap or interchange training
- Graph supervision
- Causal-edge labels
- Locality losses
- Commutation losses
- Planning or policy optimization
- Privileged next-full-state supervision

Every intervention result is genuinely out of objective.

## Dataset

### Level layout

- 64 training level seeds
- 16 validation level seeds
- 32 held-out test level seeds
- Model seeds: 42, 137, 2026

### Identity permutation partition (Amendment 2)

Of the 6! = 720 possible carrier mappings, lock a manifest-frozen partition:
- 576 train-identity mappings.
- 144 held-out identity mappings.

Training, validation selection, and Rungs 1–3 use only the 576.
Rung 4 uses the 144 held-out mappings while retaining Rung 3's layouts and
contact regime.

### Trajectory generation

- 128 fitting trajectories per training level, length 32.
- 50% randomized behavior, 50% scripted coverage behavior.
- Test uncertainty clustered by level, never by transition.

The scripted policy guarantees coverage but never provides a graph.

### Data manifest and model seed separation (Amendment 8)

Freeze one data manifest (level layouts, trajectory seeds, identity partition)
shared by all model seeds. Model seed controls initialization and training
order only — it must not change generated data.

Rung-1 evaluation uses training layouts but independently seeded episodes not
used for: gradients, early stopping, width selection, or checkpoint selection.

## Intervention protocol

### Patch boundary (Amendment 6)

At intervention time:

1. Assimilate current observation o_t into both memories.
2. Patch `recipient_hidden[r_idx] = donor_hidden[d_idx]`.
3. Given action a_t, predict the next event/observation **before** revealing
   o_{t+1}.
4. Transition both simulator branches.
5. Assimilate each branch's own o_{t+1}.
6. Repeat.

Define:
- τ=0: action a_t produces the first differing registered event.
- τ=1: action a_{t+1} produces it.

The model API must expose this observe–patch–act boundary, or an exactly
equivalent alignment.

### Paired history construction

Construct paired **naturally reached** histories — not simulator-toggled states.

Index prefixes by simulator state and find independently rolled histories with:
- Identical level and agent pose
- Identical non-target world state
- Different target-handle state
- A shared registered future action suffix

Locate donor and recipient target slots independently: patch
`donor[d_idx] → recipient[r_idx]`, not `donor[d] → recipient[d]`.

### Counterfactual branches (Amendment 7)

The two truth branches are:
- Baseline recipient world R.
- Counterfactual recipient world R[target ← D].

Both use the **recipient** episode's carrier mapping and coupled record-order
randomness. The donor episode supplies latent target content only.

Before contact, require identical non-target state, non-target records, and
observation noise. The target record itself may differ; requiring complete
observation identity is too strong.

A missing model TV crossing counts as a timing miss — it is never dropped.

### Pair support requirements (Amendment 8b)

For the four possible labeled key_i → lock_j cells, require before scoring:
- At least 64 eligible zero/one-contact pairs.
- At least 16 distinct level clusters per cell.

Reduce causal consumption macro-wise across those cells. Failure to meet
support is a coverage failure, not a scientific negative.

### Future evaluation

- Score each prediction before revealing the next observation.
- Before causal contact, both counterfactual worlds yield identical observations
  (except at the target record itself).
- After divergence, feed each branch its own resulting observation.
- Compare hybrid prediction with simulator counterfactual.

## Pre-registered gates

95% level-clustered bootstrap intervals. Primary adjudication: median across
model seeds plus directionally qualifying in ≥ 2/3 seeds. All-seed conjunctions
are diagnostic only.

| Gate | Criterion |
|------|-----------|
| Eligibility | Oracle ≥ 0.99 (deterministic, exact accuracy). Dense/sparse next-event and visible-status macro-F1 ≥ 0.90. Recurrent lift over historyless ≥ 0.10 with CI lower bound > 0. Control B within 3 points of dense for both event and status F1. Control A reported as efficiency diagnostic only. |
| Patch integrity | Same-value patch non-target TV upper bound ≤ 0.05. Self-patch exactness diagnostic. |
| Causal consumption | At eligible first contact: hybrid counterfactual-event accuracy ≥ 0.80, LB ≥ 0.70, improvement over unpatched ≥ 0.30 with LB ≥ 0.20. Mere decoding does not count. Macro-reduced across four key→lock cells. |
| Shielding & timing | Pre-contact false-effect rate ≤ 0.10, UB ≤ 0.15. Median onset error 0; 90th-pct absolute timing error ≤ 1 step. Missing TV crossing = timing miss, not dropped. |
| Graph & closure | Held-out context-edge/hyperedge macro-F1 ≥ 0.80, LB ≥ 0.70. Multi-step descendant F1 ≥ 0.80, exceeds edge-only and shuffled-graph nulls by ≥ 0.10, LB ≥ 0.05. |
| Higher-order composition | Single-lock patch activates goal ≤ 0.10; double-lock ≥ 0.80. Double minus best-single effect ≥ 0.60, LB ≥ 0.45. Both execution orders ≥ 0.80 correct, differ by TV ≤ 0.05. |
| Specificity | Wrong-handle and shuffled-donor controls reduce first-contact accuracy by ≥ 0.15, LB ≥ 0.05. |

## Positive-control staircase

Advance one rung at a time:

1. Training levels, zero/one-step contact
2. Training levels, two-step contact
3. Held-out layouts
4. Held-out identity permutations (144 from the frozen partition; same layouts
   and contact regime as Rung 3) (Amendment 2b)
5. Long contact delay (3–6 steps)
6. Joint lock composition

## Smoke test protocol

A 32-trajectory smoke run is approved only if its artifact says `smoke: true`
and forces every scientific gate to `NOT_ADJUDICATED`. It may verify:

- Mapping stability within episodes and diversity across episodes.
- Identity-partition enforcement (576 train / 144 held-out).
- Common-suffix branch simulation.
- Contact indexing.
- Independent d_idx/r_idx.
- Pair support counts.
- Field-loss masking (visible-only CE for type/row/col/status).
- Same-value and self-patch integrity.

It cannot select widths, change thresholds, or support a HANDLE claim.

The final campaign retains 128 fitting trajectories per training level plus a
separately seeded Rung-1 evaluation/pair bank.

## Outcome interpretation

| Outcome | Meaning |
|---------|---------|
| Dense and sparse pass | Autonomous causal swapping feasible in typed designed world |
| Dense passes, sparse matches | Sparsity unnecessary; behavioral law is the result |
| Sparse alone passes | Engineered locality only — do not claim discovery |
| Prediction passes, swaps fail | Repeats FBA negative — readable ≠ causal handle. Stop HANDLE |
| Flat control materially worse | Architecture comparison confounded |
| HANDLE-μ passes | Freeze evaluator, move to real-model eligibility screen |
| Real model fails eligibility | Distance-0 artifact remains unmet |

## Bridge to distance-0

If HANDLE-μ passes:

1. Freeze intervention evaluator and nulls
2. Require real model to pass behavioral eligibility screen first
3. Search full causal state (event-indexed all-layer KV/cache fragments) using
   response effects, not vector clustering
4. Apply same shielding, timing, closure, and composition laws

This does NOT reopen the frozen residual-port lane.

## Provenance

- Codex Round 1: entries e824–e830 on mission board 5df235ea
- Codex Round 2: entries e831–e836 on mission board 5df235ea
- Codex Round 3 (pipeline-invalid verdict): entries e837–e852 on mission board
  5df235ea. Five protocol bugs identified. One bounded repair authorized.
- Codex Round 3b (lock with amendments): entries e853–e867 on mission board
  5df235ea. Nine amendments locked:
  1. Per-episode identity permutation with privileged carrier map
  2. Identity permutation partition (576/144 train/test)
  3. Episode-local key–lock bijection
  4. Factorized CE loss (type/row5/col5/status CE visible-only + vis BCE + event CE)
  5. Control B: set-aware recurrent with invariant pooling, width ladder {h_pm, 96, 192}
  6. Patch boundary: observe → patch → act → predict → transition
  7. Recipient-coordinate counterfactuals with coupled noise
  8. Separate fitting from Rung-1 evaluation (frozen manifest, ≥64 pairs/cell, ≥16 clusters)
  9. Deterministic simulator oracle (not learned MLP)
- FBA-0 transferable residue: separate write paths insufficient for causal
  handles; same-factor preservation insufficient for composition; width
  allocation dominates product factorization
