# ARC-AGI-3 LS20 Current State

This is the current working checkpoint for the LS20 ARC-AGI-3 effort.

## Confirmed Official Results

Latest confirmed official score for the verified ordered-subgoal A* plan:

- Game: `ls20-9607627b`
- Levels completed: `7 / 7`
- Score: `100.0`
- Actions: `313`
- Environment completed: `true`
- Summary artifact: `eval_results/arc3_scripted_astar_l7_summary.json`
- Smoke manifest: `eval_results/arc3_scripted_astar_l7_smoke.json`

Latest bounded latent-backend comparison:

- Backend: one local latent call, then state-probe fallback
- Levels completed: `0 / 7`
- Score: `0.0`
- Actions: `66`
- Summary artifact: `eval_results/arc3_latent_one_call_after_fallback_summary.json`
- Smoke manifest: `eval_results/arc3_latent_one_call_after_fallback_smoke.json`
- Diagnostic: first latent response did not mention any legal action and was
  counted as `no_legal_action_in_latent_output`.

Latest latent plus mechanistic-fallback comparison:

- Backend: one local latent call, then verified scripted-plan fallback
- Levels completed: `7 / 7`
- Score: `100.0`
- Actions: `313`
- Summary artifact: `eval_results/arc3_latent_one_call_scripted_fallback_summary.json`
- Smoke manifest: `eval_results/arc3_latent_one_call_scripted_fallback_smoke.json`
- Diagnostic: the first latent response still failed to emit a legal action;
  the verified mechanics recovered the run.

These runs use the official ARC-AGI-3 benchmarking harness with the local
OpenAI-compatible bridge.

Evidence gate: the `100.0` scripted and scripted-fallback results establish the
solved LS20 target behavior and prove the bridge/verifier path. See
`docs/ARC3_REASONING_VALIDATION_GATES.md` for the next bar: showing that
mechanics improve raw model actions, then replacing fixed fallback with learned
online rules.

## Implemented Mechanics

The local LS20 static planner now models:

- walls
- pickups / step-counter reset
- shape, color, and rotation pads
- pusher arrows tagged `gbvqrjtaqo`
- moving modifier pads on hidden tracks

All seven official LS20 levels are solved by
`eval_results/ls20_static_astar_plans_through_l7.json`.

Held-out rule prediction now gives a non-scripted reasoning signal:

- L6 70/30 split: `16 / 16` applicable held-out rule checks supported, `0`
  contradicted, `72.7%` transition coverage.
- L7 70/30 split: `15 / 15` applicable held-out rule checks supported, `0`
  contradicted, `93.8%` transition coverage.

This shows the learned rules are precise on unseen transitions, while also
making the remaining gap clear: uncovered transitions need richer online state
abstraction before they can replace per-step scripts.

## Current Blocker

The official scripted/mechanistic ceiling is solved, and the bridge can now use
that verified policy as a fallback under a local model. The model-driven policy
alone is still not solved. A Qwen3-0.6B latent run with one model call scored
`0.0` because the model spent the bounded completion budget in reasoning text
and did not emit a legal action. The next useful work is to make the verified
mechanics act as an online advisor/validator for model proposals, not just a
fixed fallback after failure.

If a stale runtime search is still running, stop only that process with:

```powershell
powershell -ExecutionPolicy Bypass -File experiments/kill_arc3_runtime_search.ps1
```

## Reproduction Commands

Verify the local replay through level 7:

```powershell
python experiments/replay_ls20_plan.py --plans eval_results/ls20_static_astar_plans_through_l7.json --through-level 7 --require-solved-through 7 --output eval_results/ls20_replay_astar_l7_verified.json
```

Run the official scripted-plan scorecard:

```powershell
python experiments/run_arc3_local_latent_smoke.py --game-id ls20 --server-backend scripted_plan --tags scripted-plan,astar-l7 --harness-output eval_results/arc3_scripted_astar_l7_harness.json --server-log eval_results/arc3_scripted_astar_l7_server.log --trace-jsonl eval_results/arc3_scripted_astar_l7_trace.jsonl --output eval_results/arc3_scripted_astar_l7_smoke.json
```

Run the bounded latent comparison:

```powershell
python experiments/run_arc3_local_latent_smoke.py --game-id ls20 --wait-for-gpu --max-latent-calls 1 --fallback-policy state_probe --max-tokens 64 --chains 1 --generations 1 --tags latent-local,one-call,state-probe,no-action-fallback --harness-output eval_results/arc3_latent_one_call_after_fallback_harness.json --server-log eval_results/arc3_latent_one_call_after_fallback_server.log --trace-jsonl eval_results/arc3_latent_one_call_after_fallback_trace.jsonl --output eval_results/arc3_latent_one_call_after_fallback_smoke.json
```

Run the latent plus mechanistic fallback comparison:

```powershell
python experiments/run_arc3_local_latent_smoke.py --game-id ls20 --wait-for-gpu --max-latent-calls 1 --fallback-policy scripted_plan --max-tokens 64 --chains 1 --generations 1 --tags latent-local,one-call,scripted-fallback --harness-output eval_results/arc3_latent_one_call_scripted_fallback_harness.json --server-log eval_results/arc3_latent_one_call_scripted_fallback_server.log --trace-jsonl eval_results/arc3_latent_one_call_scripted_fallback_trace.jsonl --output eval_results/arc3_latent_one_call_scripted_fallback_smoke.json
```
