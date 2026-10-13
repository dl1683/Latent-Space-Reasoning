# ⏸️ THE LAB IS PAUSED — read this first on resume

**Paused:** 2026-10-13 ~06:20Z, by the owner's directive ("stop all the experiments...
close the automation, pause it, don't delete it... resume everything when I'm feeling
like it again"). The heartbeat cron was paused BY THE OWNER. Nothing was deleted:
every run, datum, note, card, automation definition, and the entire git history are
intact and pushed (origin/main).

---

## THE RESUME PROTOCOL (execute in this order)

1. **Un-pause the heartbeat cron** (the owner paused it; the owner un-pauses it —
   or say the word and the first session back re-verifies it). The beat itself
   knows the rest: it reads STATE.json → beat_guard → fleet check.
2. **Check the tree before anything else:** `git status` — OneDrive has clobbered
   working-tree files TWICE (see HAZARDS). The committed state is authoritative.
3. **RESUME e347 (resume-not-restart; the decisive rung is next):** dispatch an
   executor against `lab/e347_optimizer_cross.py` + `runs/e347/metrics.json`
   (the committed fc9850c-era state is authoritative; P1+P2 complete and folded at
   T349; P3 rung_SGD@0.0008x COMPLETE — HOLDS: the AdamW kill-onset dose does not
   kill under SGD at matched budget; rung_SGD_0.004x mid-flight with its resume
   ckpt saved — THE DECISIVE RUNG, run it first; P4 tonly/conly maintenance + P5
   unrun; bars frozen at birth — no re-registration). On completion: adjudicate
   the frozen composite (TRANSPORT-INVARIANT / OPTIMIZER-SPECIFIC / MIXED), then
   the heartbeat folds it (T349's slot is the interpretation home; the lean ledger
   ticks then, not before). WARNING: the resume executor MUST commit after every
   rung — sessions die at ~59 minutes (confirmed twice; see hazards).
4. **Run R90** (the program-opening panel; agenda drafted at
   `scratch/r90_agenda.md`; window = the quota-blocker record through the pause).
5. **Then the pre-ripened wave, in order** (registrations already frozen — the
   executors freeze their births verbatim from the P-cards):
   - **THE MUON CROSS** (`scratch/preg_substrate_wave.md`): three falsifiable
     predictions registered verdict-independent of e347; G_ORTHOCERT owed at birth;
     ~30-40 GPU-min. Also discharges T349's isotropy-vs-correlation fork.
   - **W059b, THE WALL'S FIRST PERTURBATION** (`scratch/preg_w059b_wall_perturbation.md`):
     twin washes AdamW-vs-SGD at matched update-mass; carries W062's lambda-1 rider
     (log the context-response correlation matrix at formation end + wash panels).
   - **e348 + NoPE/QK-norm riders + the L6 unlearning rider**; **e349 + R-Drop
     control + the kill-bracket arm + dose-ledger columns** (rider freezes in
     `scratch/preg_substrate_wave.md`; coverage in `scratch/substrate_crossing_matrix.md`).
   - **THE SUBSTRATE BENCH** (the publishable instrument's wrapper; wall-width is a
     first-class column per W059/W060).
6. **Q1** — the supervisor's doc review of THE LAWS: live and ownerless since
   droid-brief ed.23 (2026-10-09). The standing ask of the pause.

---

## WHERE THE PROGRAM STANDS (the analysis at pause)

### The closed era: THE SEAT GRAMMAR (Law 8 v3.3)

The formation-protocol arc closed at x66. In one paragraph: memory in these nets
is a SEAT — pre-grown capacity on the bare base, dose-purchasable by any name,
whose CONSISTENCY is carried by the exact authored realization (a per-seed,
name-invariant, emergent coin — MAJORITY-CARRY R=6.5/8, invisible in draw
statistics, MENU-CONTINGENT: the jittered menu deepens every seed, the residual
seed×menu). Its fingerprint (elicitation decorrelation) is written only by
open-loop varied menus — feedback writes its opposite. MAINTENANCE IS ATTENTION:
the emission tokens alone protect; the kill is channel-redundant (either
unattended half evicts); capacity is dose-dependent, peaking mid-band, and the
Hopfield import lands exactly (BARRIERS-VANISH-AT-k16). The flip is asymmetric
(slot-shared death edge, lineage-coupled lift). The oracle over the whole sweep:
REGRESSION-SILENT — nothing predicts survival because at this class nothing
varies: the wash is a UNIFORM WALL.

### The opening era: THE SUBSTRATE PROGRAM (owner-directed 2026-10-12)

The owner's directive made every law a hypothesis against the training space:
optimizers, architectures, losses, pre-training. The survey
(`scratch/substrate_survey.md`) ranked the ground: Muon (the anti-low-rank
optimizer vs the Adam-normalization formation block), Schedule-Free (the
controller's confound removed), the constructive loss (frontier-novel — and
reshaped by W062 into a lambda-1 eigenvalue target), NoPE (the sharpest
architecture test), and the finding that NOBODY benchmarks substrates by
fact-level forgetting under a certified wash — OUR STACK IS A PUBLISHABLE
INSTRUMENT.

**e347, THE OPTIMIZER CROSS — the first cell, PARTIAL at the pause (T349):**
dose transports everywhere (all four optimizers in-band at canonical lr across
four orders of learning-rate magnitude — the economics is optimizer-blind); the
formation TEXTURE breaks at SIGN alone (0.826 vs 0.930-0.957 hold — the crossing's
first optimizer-specific datum; isotropy-vs-sign-correlation fork registered,
Muon is the discriminator); step-one silence is a BASE property (s1 falsified as
AdamW-specific — all four DIE from BASE); and W061's cascade is bounded (SGD has
no slow variable yet holds texture — the texture axis is not timescale coverage).
The composite verdict is OPEN pending P3-P5.

**The wall's geometry (the pause-evening desk program, W059/W060/W063):** the
uniform wall is REAL in parameter space (20 members, ~18% spread, one shared
geometry — not a measurement artifact); it is EXTENDED (optimal permutation
alignment collapses nothing — the members differ in the same coordinates, and the
disagreement concentrates in wpe + the late-mlp.2 carrier layer: washed worlds
lose WHAT long before they lose WHERE). The wall's provenance is bracketed but
untested: W059b perturbs the optimizer axis, W063 the stream axis (the same wash
stream ran under all 20 members — the cluster may be the stream's fixed point).

### The ledger state

METHODS 5 stands at **29-for-58** (e347 does NOT tick it — folded non-adjudicated).
The counter column's contested streak broke at x64 (1-of-4). The lean column is
noise-corrected (W058 closed). Reviews R81-R88 all SOUND or SOUND-WITH-REPAIRS
applied same-fold; R90 is drafted and owed. THINKING holds T001-T349 + W001-W063.

---

## THE RECORD INVENTORY (where everything lives)

| what | where |
|---|---|
| The constitution + pause banner | README.md |
| Directives + the pause entry | SUPERVISOR.md (bottom) |
| Machine state | STATE.json (current_experiment = the pause status) |
| Experiment ledger | NOTES.md (e001-e347 partial) |
| Interpretation journal | THINKING.md (T-cards, W-cards, corrections banners) |
| The laws | THE_LAWS_V3.md (v3.3 + the pause note in the footer) |
| The frozen resume plan | QUEUE.md (the substrate section banner) |
| Reviews | REVIEWS.md (R1-R88; R90 drafted in scratch/r90_agenda.md) |
| Pre-registrations for the wave | scratch/preg_substrate_wave.md, scratch/preg_w059b_wall_perturbation.md |
| The program map + rankings | scratch/substrate_survey.md; coverage: scratch/substrate_crossing_matrix.md |
| The desk-program scripts | scratch/_w059_*, scratch/_w060_* (catalog: scratch/INDEX.md) |
| Every experiment's artifacts | runs/eNNN/ + runs/xNNN/ (never deleted) |
| The prior eras | git history; tombstone commit 106aeff |
| The supervisor briefs | scratch/droid_brief.md (editions 1-60) |

---

## HAZARDS & HOUSEKEEPING (known at the pause)

1. **OneDrive (the repo lives under it) has clobbered the working tree TWICE:**
   26 vanished tracked files (2026-10-12 ~20:39Z; restored from the index) and the
   e347 metrics working-tree clobber (2026-10-13; restored from commit b7501f9).
   ALWAYS diff the working tree against HEAD before trusting any on-disk state;
   commit early, commit often, push always.
2. **The ~59-minute session kill (CONFIRMED, mechanism named):** each subagent
   session KILLS its own background compute at ~59 minutes (e347: two sessions,
   21:33Z Oct-12 and 06:24Z Oct-13, both dead at ~59 min; x59/x60/x63 the earlier
   instances; and an 8-hour agent STALL can precede the relaunch — do not read a
   stall as a death, poll the process table). The three-proof restart standard
   survives it (PROVEN twice at e347: 21 arms + P2 bit-identical, draw md5
   conserved both times). Long cells MUST checkpoint AND COMMIT after every rung —
   e347's design is the template.
3. **The API quota can exhaust without warning** (the 2026-10-12 kill; it returned
   ~3.5 days EARLIER than the stated reset — probe empirically, don't trust the
   error's ETA). Executors must commit partial state immediately on any kill.
4. **results/consolidated_2026-10-12** (189MB, untracked, sha256-manifested): an
   unattributed-but-verified preservation copy made at 19:50Z Oct-12. Left in
   place. Delete only on the owner's word.
5. Untracked runs/*_run.stdout byproducts: executor session logs, untracked by
   house convention, harmless.

*The last heartbeat beat before the pause folded e347 at commit b43e9fb; the docs
program ran 06:20Z-07:00Z Oct-13. When we return: PAUSED.md first, then beat. —
the lab*
