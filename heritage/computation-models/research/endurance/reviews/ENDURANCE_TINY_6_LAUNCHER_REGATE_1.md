**REPAIR — metadata rejection is incomplete at the current launcher hash.**

Blocking findings in [git_head()](<LOCAL_PROJECTS>/new-computation-model/experiments/run_endurance_tiny.py:26), reproduced with filesystem mocks:

- `ref: refs/..\..\outside` bypasses the slash-only traversal check and reads outside `.git`. Invalid refs containing `..` or empty components also pass when a matching file exists.
- A `.git` directory containing `commondir` and detached HEAD is accepted despite being an unsupported layout.

Before retry, validate symbolic-ref syntax, enforce resolved-path containment, and explicitly reject unsupported metadata layouts.

The remaining targeted checks passed: native Git matched direct metadata before and inside the bound job; job limits and child membership were verified; watchdog cleanup exited **124 in 0.295 seconds**; nine mocked launcher paths checked failure handling and bindings. Four tiny solver controls and stored SAT replays passed. Formula, validator, and protocol bytes match their last-PASS/frozen identities.

Attempt `92052ddb` remains **TINY-VOID**, root exit **1**, worker never started. Any nonzero root launcher exit overrides a transient SAT artifact. The preregistered horizon/configuration/caps remain unchanged. No FULL solver, writes, GPU, or agents were used.

Current git: `671f8047aed0bd691b292460973d8dfe3c5a5878`

```text
SHA256
endurance_tiny.py
1ffdecfcb1b9de5c68a092a925e9f5d185358662df0d67dff8fad9971188e21e
validate_endurance_tiny.py
4f4ab3f2e4771d0bd848eaa0b870f48c2db8dfe9497587148848ea5ad894e7bb
run_endurance_tiny.py
fa3024bc65befa5dd94e72e73215c8b509b24f9a96f796e3af392e7d3dd5228b
ENDURANCE_TINY_6_PROTOCOL.md
e4b63a3b7d94ac8bae649f24aae493d854f5d224c9eadc4505a80a4c17685e08
```