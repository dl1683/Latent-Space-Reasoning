# HANDLE-mu V2: Latent-Inventory Causal Handles

**Status:** Codex design gate PASSED (conditional). Repairs implemented.
**Date:** 2026-09-01 (spec), 2026-09-01 (design gate repairs)
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

- Post-ID observation Bayes ceiling < 0.75 (obs alone insufficient)
- Post-ID history Bayes ceiling > 0.99 (full history determines outcome)
- All four key-lock cells balanced (unlock/none) across >= 4 levels
- Goal bank populated (ready/unready events)

### 4. Pipeline propagation

V2 semantics (sentinel positions, latent inventory) propagate to:
- Oracle evaluation (KeyLockGridWorld with v2_latent_keys)
- Intervention worlds (counterfactual and baseline)
- Counterfactual transplantation (held keys to sentinel, not agent position)

### 5. ControlB width ladder

Three widths are trained per seed: {h_pm, 96, 192}. The best is selected
by validation loss. This matches the v1 spec requirement but was not
previously implemented.

### 6. Trajectory length

V2 uses traj_length=48 (v1 used 32). The longer trajectories are needed
because the V2 policy performs additional steps (blind goal probe, lock
re-visits after second key pickup). At length 48, scripted episodes
achieve 100% completion for both-locks-open and goal-activation.

### 7. Output separation

V2 results save to v2_seed_{seed}_rung_{rung}.json and
v2_verdict_rung_{rung}.json, preserving V1 results.

## Preflight result (2026-09-01, repaired)

**PASS.**
- Post-ID obs Bayes ceiling: 0.6807 (< 0.75)
- Post-ID hist Bayes ceiling: 1.0000 (> 0.99)
- Memory gap: 32 percentage points
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
- Intervention protocol: observe-patch-act boundary (Amendment 6)
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
action is insufficient (ceiling 0.68); full history is sufficient (ceiling
1.00). The 32-point gap is the memory dependence signal.

## Identifiability

With blind probes, the 2x2 key-lock bijection is unidentifiable before
first contact. The first USE at a lock is a calibration event that reveals
the mapping. Post-identification contacts are where memory-dependent
prediction is tested. The model learns the bijection through sequential
contact, not from observation alone.
