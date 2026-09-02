# HANDLE-mu V2: Latent-Inventory Causal Handles

**Status:** Design gate PASSED. Evidence gate CLEARED.
**Date:** 2026-09-01 (spec), 2026-09-01 (design + evidence gate repairs + final clearance)
**Predecessor:** `theory/HANDLE_MU.md` (v1, locked, eligibility FAIL)
**Runner:** `experiments/run_handle_mu.py` with `--v2` flag

## Relation to V1

HANDLE-mu V1 completed one seed (42) and failed eligibility: recurrent lift
= -0.003 (needs >= 0.10). Root cause: the v1 observation is sufficient for
prediction without memory. Held keys follow the agent and remain visible;
the scripted policy leaks the key-lock bijection through action selection.
V1 is preserved as a completed negative. V2 makes targeted changes
to create genuine memory dependence while preserving all other v1 structure.

## V2 changes

### 1. Latent inventory

Once a key is picked up, it vanishes from observation. Its carrier slot
shows visibility=0 for all subsequent timesteps. The key's world-state
position is set to a sentinel (-99, -99) that falls outside any visibility
check. The model must remember the pickup event to know a key is held.

### 2. Blind USE probes with randomized order

The scripted coverage policy does not consult the key-lock bijection when
choosing lock visit order. It picks up one key (randomly chosen: key 0 or
key 1 with equal probability), visits locks in randomized order, and
attempts USE at each. Successful USE (matching key) produces an unlock
event; failed USE (wrong key) produces no event.

After probing locks, the policy visits the goal for a blind probe (will
produce no event since not all locks are open), then picks up the second
key and completes the episode.

The policy is stateful within each episode (tracks probed locks and
goal probe) to avoid infinite loops.

### 3. Post-identification preflight

The first USE at a lock is a calibration/identification event: it reveals
the bijection (one probe determines the full 2x2 mapping). All subsequent
lock contacts are post-identification.

The preflight verifies memory dependence on post-identification contacts:

- Post-ID canonical observation Bayes ceiling < 0.75 (V1 contract, unmodified)
- Post-ID history Bayes ceiling > 0.99 (full history determines outcome)
- All four key-lock cells balanced (unlock/none) across >= 4 levels
- Goal bank populated (ready/unready events)

Raw observation ceiling (exact model interface) is computed and reported as
diagnostic per e905. No new thresholds: the PASS criterion uses the exact V1
contract (canonical < 0.75, hist > 0.99).

### 4. Pipeline propagation

V2 semantics (sentinel positions, latent inventory) propagate to:
- Oracle evaluation (KeyLockGridWorld with v2_latent_keys)
- Intervention worlds (counterfactual and baseline)
- Counterfactual transplantation (held keys to sentinel, not agent position)

### 5. ControlB width ladder

Three widths are trained per seed: {h_pm, 96, 192}. Per-seed selection:
first width where val event F1 >= 0.90, val status F1 >= 0.90, and both
are within 3 percentage points of Dense (one-sided: outperforming is OK).
If no width qualifies, eligibility fails for that seed. Cross-seed
adjudication uses numerical median F1 values (not boolean flags) and
requires qualification in at least 2/3 seeds.

### 6. Trajectory length

V2 uses traj_length=48 (v1 used 32). The longer trajectories are needed
because the V2 policy performs additional steps (blind goal probe, lock
re-visits after second key pickup). At length 48, scripted episodes
achieve 100% completion for both-locks-open and goal-activation.

### 7. Output separation

V2 results save to v2_seed_{seed}_rung_{rung}.json and
v2_verdict_rung_{rung}.json, preserving V1 results.

### 8. Observation ceiling methodology

The preflight computes two observation ceilings:
- **Canonical** (sorted carrier vectors): the PASS metric with the exact V1
  threshold (< 0.75). The V1 contract was calibrated for canonical-equivalent
  observations (V1 had no raw/canonical distinction). Using the unmodified V1
  threshold eliminates any post-hoc threshold concern.
- **Raw** (exact slot-ordered tensors): diagnostic, computed per e905 for the
  exact model interface. Raw ceiling is higher than canonical because
  per-episode carrier permutations create distinguishable slot orderings.
- Both memory gaps are reported: canonical (hist - canonical) and raw
  (hist - raw).

### 9. Intervention pair bijection filter

`find_paired_histories` requires donor and recipient trajectories to share
the same episode-local key-lock bijection. Pairs crossing different bijections
produce ill-defined interventions (transplanting key state that encodes a
different relation).

### 10. Native pre-update recurrent boundary

This V2 amendment overrides V1 Amendment 6 for the Dense and Sparse slot
models only.

These models expose no observation-only recurrent state. Their native step is

h_t+ = Phi(h_t-, E(o_t) + A(a_t)),

where Phi is the per-slot GRU update followed by learned messaging, and
the event/next-observation prediction is read from h_t+ before
o_{t+1} is revealed. The observation encoding is slot-specific; only the
action embedding is broadcast across slots.

The V2 intervention therefore uses the model's native pre-update hidden-state
boundary:

1. Replay donor and recipient prefixes through
   (o_{t-1}, a_{t-1}), obtaining h_t^{D,-} and h_t^{R,-}.
2. Locate the donor and recipient target carriers independently as
   `d_idx` and `r_idx`.
3. A pair is eligible for causal scoring only if its current raw target-slot
   records are exactly equal:
   `donor_obs[t, d_idx] == recipient_obs[t, r_idx]`.
   Excluded pairs and remaining support must be reported by handle, causal
   cell, and level. Insufficient remaining support is a coverage failure.
4. Form the intervention
   `hybrid_hidden[r_idx] = donor_hidden[d_idx]`.
5. Feed the identical recipient observation o_t^R and registered action
   a_t to the patched and unpatched models, then score their predictions
   before revealing o_{t+1}.
6. Transition the baseline world R and counterfactual world
   R[target<-D]. Thereafter, feed each branch its own
   resulting observation and repeat.

This is a pre-update recurrent-state intervention, not an exactly equivalent
implementation of the V1 observe-then-patch boundary. It tests whether donor
target-slot state encoded by the prefix is causally preserved and consumed
through the trained model's native observation-action update. Attenuation or
overwrite by that update counts as a failure of causal persistence at this
boundary; it is not repaired by changing or retraining the architecture.

Intervention outputs generated before this amendment are exploratory and
remain NOT_ADJUDICATED. Confirmatory scoring requires a re-locked evaluator
run with the amended pair-eligibility and support accounting.

## Preflight result (2026-09-01, V1 contract PASS)

**PASS** (registered seed 9999).
- Post-ID canonical ceiling: 0.6807 (< 0.75, V1 contract PASS)
- Post-ID raw ceiling: 0.7656 (diagnostic, exact model interface)
- Post-ID hist Bayes ceiling: 1.0000 (> 0.99)
- Memory gap: 31.9pp (canonical), 23.4pp (raw)
- All 4 cells balanced (160-194 per outcome per cell, 16 levels each)
- Goal bank: ready_activate=1353, unready_none=1427
- Coverage: 100% scripted completion

## Unchanged from V1

Everything not listed above carries over from HANDLE_MU.md unchanged:

- 7x7 grid, 5 causal handles, episode-local bijection
- Six carriers, partial visibility Manhattan radius 2
- Per-episode identity permutation (Amendment 1)
- Record encoding: type/row/col/status/vis (Amendment 4b)
- All five architectures: Dense, Sparse, FlatGRU, ControlB, Historyless
- Factorized CE loss (Amendment 4)
- Intervention protocol: native pre-update recurrent-state boundary
  (V2 Amendment 10; overrides V1 Amendment 6)
- Counterfactual branches (Amendment 7)
- Data manifest and seed separation (Amendment 8)
- Deterministic oracle (Amendment 9)
- Pre-registered gates (all thresholds unchanged)
- Positive-control staircase (Rungs 1-6)
- Prohibited training signals (no swap/graph/causal/locality losses)
- Smoke test protocol
- Seeds: {42, 137, 2026}
- Campaign: 64/16/32 train/val/test levels, 128 trajs/level, 40 epochs

## Why these changes suffice

With latent inventory, the model cannot know which key(s) it holds from
the current observation. With blind USE probes, the scripted policy
attempts USE at locks regardless of held-key state, producing identical
observations with different outcomes. Memory is necessary to:

1. Remember which key was picked up (from the pickup event)
2. Accumulate contact evidence (failed/successful USE) to resolve the
   episode-local key-lock bijection
3. Predict future USE outcomes using accumulated evidence

The post-identification preflight formally verifies this: observation +
action is insufficient (canonical ceiling 0.681, raw ceiling 0.766); full
history is sufficient (ceiling 1.000). The memory gap is 23-32 percentage
points depending on whether raw or canonical observations are measured.

## Identifiability

With blind probes, the 2x2 key-lock bijection is unidentifiable before
first contact. The first USE at a lock is a calibration event that reveals
the mapping. Post-identification contacts are where memory-dependent
prediction is tested. The model learns the bijection through sequential
contact, not from observation alone.
