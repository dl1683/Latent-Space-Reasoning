# Re-entry council, round 3 (final; sign or amend)

Read internal/council_gcp/droid_r2.md (Droid's proposed joint pre-registration) and internal/council_gcp/codex_r2.md (Codex: does not sign Droid's text unchanged; signs a consolidated spec with six amendments: exact edge/balancing spec, branch-addressed edit/undo/advance tests, refined capacity grid, PR definition and admissibility, frozen training/selection rules plus a storage/delayed-computation control, and a funded larger-model panel).

This is the last round. Answer in under 600 words:
1. Do you sign codex_r2.md as the frozen pre-registration? Yes, or yes-with-amendments (list each amendment in one line with the reason), or no (give the experiment that would settle the disagreement).
2. Operator implementation notes for your review (amend if wrong): (a) the flagship and the 0.6B gate share one trainer; (b) the storage/delayed-computation control will be implemented as a second trainer variant only if the gate passes, since it is needed for the computation claim, not for admission; (c) frozen Probe A uses the last-layer residual injected as an input embedding into a fresh context, a weak interface whose null is uninformative; the operator proposes to skip frozen A and spend that time on the trained gate. Agree or object.
3. Name the single result that would most change your view of the program's direction.
Write markdown to stdout. No experiments, no spend, no file edits.
