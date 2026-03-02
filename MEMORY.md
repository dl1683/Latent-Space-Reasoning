# Persistent Memory

## Owner Preferences
- Prefer autonomous execution: plan, implement, validate, self-review, and continue until goals are complete.
- Avoid unnecessary stop-and-wait prompts during normal execution.
- Maintain explicit task tracking and progress state in repository files.
- Re-evaluate whether the repository has reached the target state after each implementation cycle.
- Prioritize practical accessibility over leaderboard performance.
- Prefer solutions that deliver strong quality at radically lower cost/resource usage.
- Require transparency and auditability in system behavior and evaluation artifacts.

## Operating Constraint
- True always-on runtime is not possible without an active session.
- To preserve continuity, store state in `GOALS.md`, `TASKS.md`, and `WORKLOG.md` and resume from them each new session.
