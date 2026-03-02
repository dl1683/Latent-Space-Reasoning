# Autonomous Workflow Contract

## Startup Protocol
1. Read `MEMORY.md`.
2. Read `GOALS.md`.
3. Read `TASKS.md`.
4. Read `WORKLOG.md`.
5. Inspect repository state (`git status --short`) before changing code.

## Execution Loop (Default Mode)
1. Choose the highest-priority unchecked task in `TASKS.md` and move it to `Doing`.
2. Implement the smallest safe change that advances the active goal.
3. Run targeted validation (tests, lint, type checks, or script checks relevant to the change).
4. Perform self-review of the diff for bugs, regressions, and missing tests.
5. Re-check `GOALS.md` acceptance criteria against current repository state.
6. Update `TASKS.md` and append a short entry to `WORKLOG.md`.
7. Repeat until all active goal acceptance criteria are complete.

## Autonomy Rules
- Continue without asking for confirmation between subtasks.
- If blocked, pivot to the highest-impact unblocked task that still advances active goals.
- Ask the user only when every meaningful path is blocked by missing credentials, missing external access, or conflicting requirements.
- Do not stop at planning if implementation is feasible in the same session.
- If session context resets, resume by re-running the Startup Protocol.

## Definition of Done
- Every active goal in `GOALS.md` has all acceptance criteria checked.
- Relevant validation commands pass or failures are explicitly documented in `WORKLOG.md`.
- `TASKS.md` has no remaining items in `Doing`.
