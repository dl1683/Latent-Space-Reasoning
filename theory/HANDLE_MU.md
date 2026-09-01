# HANDLE-μ: Causal Handle Algebra in a Designed Latent World

**Status:** Locked specification (Codex Rounds 1–2). Distance-1.
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
- A hidden key–lock bijection is resampled per episode and never supplied.
- Goal activates only when both locks are open: {ℓ₁,ℓ₂} → g (higher-order).
- Keys remain held after use (simplifies paired counterfactual construction).

## Observation

Partial visibility, Manhattan radius 2 from agent.

Each visible record contains:
- Episode-local opaque identity
- Coarse type
- Relative position
- Visible mutable status

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

### 3. Flat GRU (PREDICTIVE CONTROL)
- Hidden width chosen for parameter count within 5% of slot models.
- Tests whether slot models merely predict better.
- No unnatural subvector-swap test.

### 4. Historyless (RECURRENCE NECESSITY)
- Same architecture as dense slots but no recurrence.
- Tests whether persistent state is needed.

### 5. Direct-state symbolic oracle (PIPELINE VALIDITY)
- Receives ground-truth simulator state.
- Tests coverage and pipeline correctness.

A dense-slot pass is scientifically stronger than sparse-only. Sparse-only
success means engineered locality, not discovered structure.

## Training

Train solely on:
- Next-observation prediction
- Next-event prediction

Masked categorical cross-entropy, equal event and record loss weights.

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

- 64 training level seeds
- 16 validation level seeds
- 32 held-out test level seeds
- Model seeds: 42, 137, 2026
- 128 trajectories per training level, length 32
- 50% randomized behavior, 50% scripted coverage behavior
- Test uncertainty clustered by level, never by transition

The scripted policy guarantees coverage but never provides a graph.

## Intervention protocol

Construct paired **naturally reached** histories — not simulator-toggled states.

Index prefixes by simulator state and find independently rolled histories with:
- Identical level and agent pose
- Identical non-target world state
- Different target-handle state
- A shared registered future action suffix

After encoding both histories, replace the entire target slot from donor A
in recipient B.

For future evaluation:
- Score each prediction before revealing the next observation
- Before causal contact, both counterfactual worlds yield identical observations
- After divergence, feed each branch its own resulting observation
- Compare hybrid prediction with simulator counterfactual

## Pre-registered gates

95% level-clustered bootstrap intervals. Primary adjudication: median across
model seeds plus directionally qualifying in ≥ 2/3 seeds. All-seed conjunctions
are diagnostic only.

| Gate | Criterion |
|------|-----------|
| Eligibility | Oracle ≥ 0.99. Dense/sparse next-event and visible-status macro-F1 ≥ 0.90. Recurrent lift over historyless ≥ 0.10 with CI lower bound > 0. Flat GRU within 3 points of dense. |
| Patch integrity | Same-value patch non-target TV upper bound ≤ 0.05. Self-patch exactness diagnostic. |
| Causal consumption | At eligible first contact: hybrid counterfactual-event accuracy ≥ 0.80, LB ≥ 0.70, improvement over unpatched ≥ 0.30 with LB ≥ 0.20. Mere decoding does not count. |
| Shielding & timing | Pre-contact false-effect rate ≤ 0.10, UB ≤ 0.15. Median onset error 0; 90th-pct absolute timing error ≤ 1 step. |
| Graph & closure | Held-out context-edge/hyperedge macro-F1 ≥ 0.80, LB ≥ 0.70. Multi-step descendant F1 ≥ 0.80, exceeds edge-only and shuffled-graph nulls by ≥ 0.10, LB ≥ 0.05. |
| Higher-order composition | Single-lock patch activates goal ≤ 0.10; double-lock ≥ 0.80. Double minus best-single effect ≥ 0.60, LB ≥ 0.45. Both execution orders ≥ 0.80 correct, differ by TV ≤ 0.05. |
| Specificity | Wrong-handle and shuffled-donor controls reduce first-contact accuracy by ≥ 0.15, LB ≥ 0.05. |

## Positive-control staircase

Advance one rung at a time:

1. Training levels, zero/one-step contact
2. Training levels, two-step contact
3. Held-out layouts
4. Held-out identity permutations
5. Long contact delay (3–6 steps)
6. Joint lock composition

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
- FBA-0 transferable residue: separate write paths insufficient for causal
  handles; same-factor preservation insufficient for composition; width
  allocation dominates product factorization
