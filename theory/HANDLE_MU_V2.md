# HANDLE-mu V2: Latent-Inventory Causal Handles

**Status:** Draft specification. Pending Codex design gate.
**Date:** 2026-09-01
**Predecessor:** `theory/HANDLE_MU.md` (v1, locked, eligibility FAIL)
**Runner:** `experiments/run_handle_mu.py` with `--v2` flag

## Relation to V1

HANDLE-mu V1 completed one seed (42) and failed eligibility: recurrent lift
= -0.003 (needs >= 0.10). Root cause: the v1 observation is sufficient for
prediction without memory. Held keys follow the agent and remain visible;
the scripted policy leaks the key-lock bijection through action selection.
V1 is preserved as a completed negative. V2 makes three targeted changes
to create genuine memory dependence while preserving all other v1 structure.

## V2 changes (three modifications)

### 1. Latent inventory

Once a key is picked up, it vanishes from observation. Its carrier slot
shows visibility=0 for all subsequent timesteps. The key's world-state
position is set to a sentinel (-99, -99) that falls outside any visibility
check. The model must remember the pickup event to know a key is held.

### 2. Blind USE probes

The scripted coverage policy does not consult the key-lock bijection when
choosing lock visit order. After picking up one key, it visits each lock
in index order and attempts USE at each. Successful USE (matching key)
produces an unlock event; failed USE (wrong key) produces no event. This
provides balanced contact evidence without leaking hidden state through
action selection.

The policy is stateful within each episode (tracks which locks have been
probed) to avoid infinite loops at non-matching locks.

### 3. Ambiguity bank (preflight gate)

Before any model training, a deterministic preflight verifies the world
creates observation aliasing. Using permutation-invariant observation
matching (sorted carrier vectors), the preflight constructs pairs where:

    o_t = o'_t,   a_t = a'_t,   y_{t+1} != y'_{t+1}

The observation-only Bayes ceiling on ambiguous pairs must be < 0.75.
The full-history oracle ceiling must be 1.00.

**Preflight result (2026-09-01):** PASS. 15 ambiguous groups, Bayes
ceiling 0.591, 8/16 levels with mixed USE outcomes.

## Unchanged from V1

Everything not listed above carries over from HANDLE_MU.md unchanged:

- 7x7 grid, 5 causal handles, episode-local bijection
- Six carriers, partial visibility Manhattan radius 2
- Per-episode identity permutation (Amendment 1)
- Record encoding: type/row/col/status/vis (Amendment 4b)
- All five architectures: Dense, Sparse, FlatGRU, ControlB, Historyless
- ControlB width ladder {h_pm, 96, 192} (to be implemented)
- Factorized CE loss (Amendment 4)
- Intervention protocol: observe-patch-act boundary (Amendment 6)
- Counterfactual branches (Amendment 7)
- Data manifest and seed separation (Amendment 8)
- Deterministic oracle (Amendment 9)
- Pre-registered gates (all thresholds unchanged)
- Positive-control staircase (Rungs 1-6)
- Prohibited training signals (no swap/graph/causal/locality losses)
- Smoke test protocol

## Why these changes suffice

With latent inventory, the model cannot know which key(s) it holds from
the current observation. With blind USE probes, the scripted policy
attempts USE at locks regardless of held-key state, producing identical
observations with different outcomes. Memory is necessary to:

1. Remember which key was picked up (from the pickup event)
2. Accumulate contact evidence (failed/successful USE) to resolve the
   episode-local key-lock bijection
3. Predict future USE outcomes using accumulated evidence

The ambiguity preflight formally verifies this: observation + action is
insufficient; history is necessary.

## Identifiability note

With blind probes, the 2x2 key-lock bijection is unidentifiable before
first contact. Failed USE at a lock reveals that the held key does not
match that lock, narrowing the bijection. Successful USE confirms the
match. The model learns the bijection through sequential contact, not
from observation alone. This is by design: the causal handle test
measures whether memory carries and transmits this learned structure.
