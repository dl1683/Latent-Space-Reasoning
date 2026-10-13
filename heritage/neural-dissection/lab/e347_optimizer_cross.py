"""E347 — THE OPTIMIZER CROSS (THE SUBSTRATE PROGRAM's first GPU cell; the
training space opened by the owner's directive 2026-10-12 ~02:30Z: GPU
aggressive). This docstring carries the registered question + background +
design + bars + the parity block + P-e347a VERBATIM from the dispatch
letter, plus every frozen operationalization, committed at birth BEFORE any
compute. Adjudicate against exactly this; no bar shopping.

BACKGROUND (dispatch verbatim): "every law this lab has measured was
measured under AdamW on one 2.74M architecture with one CE loss — Law 1
(every death is a step-death), Law 2b's kill bracket [0.0008, 0.004] of
write norm, the formation texture (the wheel, the net0 classes), the seat
grammar (maintenance, capacity). NONE has ever been crossed against a
different optimizer. THE OWNER'S QUESTION: is the grammar the training
dynamics' (transport-invariant) or AdamW's? YOUR CELL: the first cross."

THE DESIGN (dispatch verbatim):
  "1. THE OPTIMIZER SET: SGD (no momentum, no adaptivity — the pure
  geometry), momentum-SGD (0.9), RMSProp (adaptive but no moment terms),
  and SIGN-DESCENT (the extreme: pure direction, magnitude-normalized —
  the lab's own simplest custom optimizer; state any extra you implement).
  Per-optimizer lr sweeps at birth (a coarse 5-point ladder each, the canon
  formation at the standard dose; the lr that lands in the canon's read
  band becomes that optimizer's canonical lr — the DOSE NORMALIZATION,
  pre-registered).
  2. THE FORMATION TEXTURE: the canon name's formation under each
  optimizer at its canonical lr — the read trajectory, the write norm, the
  in-room fraction, the s1 behavior (the wheel? the classes?), vs the
  committed AdamW record.
  3. THE KILL BRACKET: Law 2b's two boundary rungs (the 0.0008x and
  0.004x transport-kills) re-run under each optimizer in ITS canonical-lr
  terms (the drift-realization disclosed) — does the bracket MOVE
  (per-optimizer renormalized) or HOLD?
  4. THE MAINTENANCE PROBE: the tonly/conly split (x57's role split — the
  cheapest seat probe: does attention-only protect under each optimizer?)"

BARS (dispatch verbatim; frozen in this birth commit BEFORE compute):
  - TRANSPORT-INVARIANT: "the laws hold under every optimizer at its
    canonical lr (the formation texture within the disclosed band; the
    bracket within 2x after lr-normalization; maintenance attention-side
    throughout) — the grammar is the loss-geometry's."
  - OPTIMIZER-SPECIFIC: "which laws move first (the texture? the bracket?
    the maintenance?) — named per optimizer; the grammar re-scopes to
    AdamW-class dynamics."
  - MIXED: "some hold, some move — the honest split, itemized."

THE PARITY BLOCK (the dispatch standard — the lab lean AND the counter
side by side, both verbatim):
  LAB LEAN: "OPTIMIZER-SPECIFIC, moderately (the adaptive vs non-adaptive
  step geometry differs hugely; the bracket especially should move — it is
  measured in write-norm units that mean different things per optimizer)"
  COUNTER: "TRANSPORT-INVARIANT (the loss geometry dominates; the
  optimizer is a solver, and solvers converge to the same minima basin)"

P-e347a (THE EXECUTOR'S OWN READ, registered per the dispatch's "Register
P-e347a (your own read) BEFORE compute", frozen HERE at birth; predictions
are scored): **MIXED — exactly one law-face moves: the s1 fork / the
trajectory texture; the bracket and the maintenance split hold.** The four
clauses, each falsifiable on this cell's own reads:
  (1) THE BULK TEXTURE HOLDS (write norm within the 2x band AND in-own-
      room within 0.10 of the canon AND final g0 in the read band) for
      ALL FOUR optimizers at their canonical lr — the room projector
      passes the same clipped gradient to every optimizer and the
      committed write was 94.4% in-room because of that capture; the
      dose normalization pins the read.
  (2) THE S1 FORK MOVES ALONG STEP-GEOMETRY CLASS LINES: the sign-class
      optimizers (SIGN exactly; RMSProp whose bias-uncorrected first step
      is ~lr*g/sqrt((1-alpha) g^2) = ~lr/sqrt(1-alpha) ~ 3-10x lr a
      near-sign kick) DIE at s1 like AdamW (the canon's 1.35e-5); the
      magnitude-class optimizers (SGD, MSGD) SURVIVE s1 (their first
      steps are tiny on the clipped projected gradient) — the canon's
      die-then-recover is a normalized-step artifact, not memory physics.
  (3) THE BRACKET HOLDS IN BUDGET TERMS for all four (0.0008x holds
      >= 0.5, 0.004x dies < 0.5): the equal-share cap prices the rungs in
      realized drift NORM regardless of optimizer, and e290's margins
      (0.767 vs 0.319) sit far enough from 0.5 that the direction mix
      does not flip them.
  (4) THE MAINTENANCE ROLE SPLIT HOLDS (attention-side) for all four:
      tonly protects, conly executes — the split is supervision-side
      channel identity, not step geometry.
  NET SCORED: TRUE iff the composite verdict == MIXED AND clauses (2)
  and (3) and (4) as stated ((1) may partially fail on RMSProp's read
  band — the fallback clause below — without falsifying the net).
  FALSIFIER: any hard-gate failure -> TEXTURE (nothing adjudicated;
  P-e347a UNSCORED per the family's precedent clause).

==== THE FROZEN OPERATIONALIZATIONS (picked + frozen HERE at birth) ====

* THE CANON (what is being crossed): the committed K10K formation —
  THE BASE e001.pt + e261's chunked_install VERBATIM (400 Dmix steps,
  gen 24314 — the canon's OWN draw, bit-identical streams across every
  arm in this cell; AdamW (0.9,0.95) wd 0.1; lr 1e-3 x house
  cosine(total=1000); clip 1.0 -> P_room) in the HELD room K10K (seeds
  26113/26114, bit-bound vs e264_rooms.pt). Committed record (hard
  literals, asserted at runtime): post g0 0.26464763283729553, gm12
  0.10525520890951157, write norm 9.1788432658723, in-own-room
  0.9441588788935659, s1 g0 1.3466433301800862e-05 (the DIE class).
  AMENDMENT (smoke-caught repair, 2026-10-12, disclosed): the birth
  draft said "g1c_root.pt + install" following e324's docstring claim —
  WRONG per the lab's own committed record: x32 (7bcd060, THE NET0
  CONFOUND) proved e264/e272's canon ran chunked_install from G1.
  evl_load(base_sd) = THE BASE, with the canon's s1 "die" re-worded to
  the base's prior staying flat (prior 1.3383959412749391e-05; "AT STEP
  ONE NOTHING EVER HAPPENS"); e328's control reproduced the canon from
  the BASE on this GPU (s1 rel 3.38e-07). This cell's smoke caught the
  same confound independently (root-started CE 1.1963 vs committed
  1.0702; base-started CE/s1 BIT-IDENTICAL on CPU). The bars and
  P-e347a stay frozen VERBATIM; only the substrate statement is
  corrected. Consequence for the s1 classes: from the BASE, DIE reads
  "prior-flat, no step-one event" and SURVIVE reads "a step-one read
  event" — x32's re-wording, adopted; P-e347a clause (2)'s frozen
  discriminating content (the s1 class splits by step-geometry class)
  is UNCHANGED and now cleaner (no standing-read confound).
* THE OPTIMIZER SET (frozen): SGD (torch.optim.SGD, lr only), MSGD
  (torch.optim.SGD momentum=0.9, dampening 0), RMSPROP (torch.optim.
  RMSprop alpha=0.99 eps=1e-8 momentum=0 centered=False — adaptive, NO
  moment terms), SIGN (THIS CELL'S OWN minimal custom optimizer, stated
  per the dispatch: update = p -= lr * sign(grad); NO state, NO moments,
  NO adaptivity — pure direction, magnitude-normalized; implemented as a
  torch.optim.Optimizer subclass in this file). ADAMW (the committed
  canon's own class) runs ONCE as the in-session anchor arm — never a
  law-carrier.
* THE DECAY CHANNEL (frozen, disclosed): every CROSSED arm carries the
  committed per-step decoupled decay VELOCITY (1 - 1e-3 x 0.1 x
  cosine_t) — AdamW@1e-3 wd 0.1's own per-step multiplier — applied
  post-step, so the decay channel is held at the committed value while
  the step geometry varies (built-ins run wd=0; the ADAMW anchor uses
  torch AdamW wd 0.1 bit-faithful to the committed rig; the post-step
  vs in-step ordering difference for crossed arms is a disclosed
  ~1e-4-relative form difference, never a channel delta).
* THE LADDERS (frozen 5-point, geometric, matched-displacement centered):
  SGD {0.03, 0.1, 0.3, 1.0, 3.0}; MSGD {0.01, 0.03, 0.1, 0.3, 1.0};
  RMSPROP {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}; SIGN {1e-4, 3e-4, 1e-3, 3e-3,
  1e-2} (the sign-class cancellation structure is AdamW-like, so SIGN's
  displacement-matched center is ~1e-3, disclosed).
* THE CANONICAL LR (the DOSE NORMALIZATION, frozen): the ladder run
  whose post-install g0 lands in the canon read band [0.15, 0.45]
  (e272's trust band; the canon 0.2646 in-band); if MULTIPLE land
  in-band, the one whose g0 is closest to 0.26464763283729553; if NONE
  land in-band, the run with min |log(g0 / 0.2646...)| is taken WITH THE
  MISS DISCLOSED (an optimizer with no in-band dose is itself a finding;
  phases 3/4 still run at that lr).
* FORMATION TEXTURE "within the disclosed band" (frozen per axis):
  write norm in [0.5x, 2.0x] of 9.1788432658723 (e323's 2x texture
  convention); in-own-room within 0.10 ABSOLUTE of 0.9441588788935659;
  s1 class: DIE := s1 g0 < 0.01, SURVIVE := s1 g0 > 0.15 (the gap
  [0.01, 0.15] no-man's land, disclosed — the canon 1.35e-5 and e323's
  0.7451 sit 3+ orders either side); final g0 in [0.15, 0.45] by the
  canonical-lr rule (a fallback lr's miss is a named texture miss). The
  texture HOLDS iff ALL axes hold; every exit is named per axis.
* THE BRACKET (frozen): e290's rung form VERBATIM with ONLY the vehicle
  optimizer + its lr swapped — the committed quiet-formed 10k fact
  (e261_K10K_inst_resume.pt) loaded bit-exact; the corpus stream 48
  windows (16 original-host anchors + 32 random), draws from THIS cell's
  ONE fresh generator seed 34701 (the family's per-cell rule .../28301/
  28401/28501/28801/29001/HERE 34701), bit-identical across every rung;
  backward -> clip 1.0 -> g_perp = g - P_room(g) (CPU fp64, verified
  EVERY step) -> the equal-share per-step lr cap cap_t = ((BUDGET -
  S_{t-1})/(n-t+1))/||b_t||, b_t = the optimizer's would-be unit-lr
  update direction (SGD: g_perp; MSGD: 0.9*buf + g_perp; RMSPROP:
  g_perp/(sqrt(v_new)+eps); SIGN: sign(g_perp)); lr_t = min(canonical_lr
  x cosine_lr(t-1,1000), cap_t); S += realized ||step||; NO decay in the
  rungs (wd 0.0 — e290's committed form). Budgets {0.0008, 0.004} x
  9.1788432658723; milestones t100/200/300/400; TWO optimizer instances
  (opt_C stepped, opt_F NEVER stepped — the bitwise state-isolation
  probe every step). The committed e290 reference rows (SGD-M @
  LR_STABLE 0.2174: 0.0008x ratio 0.7674 HOLDS / 0.004x ratio 0.3194
  DIES) are READ from e290's metrics at runtime and asserted against
  birth literals — never rerun. The bracket HOLDS for an optimizer iff
  ratio(0.0008x, t400) >= 0.5 AND ratio(0.004x, t400) < 0.5; it MOVES
  otherwise (the flipped rung + direction named). The drift realization
  disclosed per rung: S_400, realized cumulative drift, cap-bound
  fraction, drift in-room fraction.
* THE MAINTENANCE PROBE (frozen): x57's role split on THIS cell's canon
  fact — the cons-style stream (e261's chunked_consolidate batch form:
  16 jittered-pool name windows + 8 paired-original anchors + 8 random
  corpus; draws from THIS cell's seed 34702, bit-identical across arms),
  300 steps, const lr = the optimizer's canonical lr, NATURAL (no room
  hook — the committed cons convention), the uniform decay velocity (1 -
  1e-4) per step for crossed arms (AdamW reference uses the built-in wd
  0.1). THE MASKS (x57's scheme verbatim, counts frozen): tonly = the
  112 name-EMISSION tokens/step (16x7); conly = the 8048-token
  complement (the name rows' 248 non-name positions + all 16 nameless
  windows x 255); G_MASKUNION asserts the canon union mask == tonly |
  (conly & nameless rows) over every draw. THE READ: g0 at s1/s25/s125/
  s300 (x57's fate steps + the s1 armor), survival = g0(t)/
  0.26464763283729553. ATTENTION-SIDE (the law holds) iff tonly
  protects (survival(s300) >= 0.5) AND conly executes (< 0.5); a flip is
  named. An ADAMW@1e-3 tonly/conly pair runs IN-SESSION as the
  reference row (DISCLOSED: x57's committed split was the TAVIREN
  subject on e345's stream; this cell's fact is the K10K ZEPHYRA canon
  — the AdamW row is this cell's own internal reference, the committed
  x57 numbers cited as context, never asserted).
* COMPOSITE (frozen): TEXTURE (any hard-gate failure — nothing
  adjudicated) -> per-optimizer law table {texture, bracket,
  maintenance} -> n_hold := # optimizers holding ALL THREE; TRANSPORT-
  INVARIANT iff n_hold == 4; OPTIMIZER-SPECIFIC iff n_hold == 0 (which
  laws moved, named per optimizer); MIXED iff 0 < n_hold < 4 (the
  honest split, itemized).
* THE THREE-PROOF RESTART STANDARD (R83/R84's form, standing): if any
  leg is killed and relaunched, the restart discloses (1) the draw
  record bit-conserved across the kill, (2) the pre-kill instrument rows
  bit-identical on re-execution, (3) the metrics re-derived from the
  resumed state — appended to deviations. Per-arm resume ckpts make
  every leg idempotent (a completed leg short-circuits on relaunch).
* GATES (a failure HALTS the record as TEXTURE): {G_VOCAB, G_NAMEFREE,
  G_SPLICE, G_BATTERY, G_ANCHOR, G_INSTMASK, G_NAMEWIN, G_PARENTS,
  G_BASE, G_NET0, G_VMBIND, G_SPANBIND, G_PROJ, G_ROOMHELD, G_ANCHORW,
  G_FACTLOAD, G_CANONLADDER, G_DRAWSHARED} + per-arm {G_PROTOIDENT} +
  per-rung {G_ORTH, G_BUFSEP, G_BUDGET} + phase-4 {G_MASKUNION,
  G_ROLEMASK}.
* G_ANCHORW (the in-session AdamW@1e-3 canon reproduction — the
  instrument-identity gate): |g0 - 0.2646...| <= 0.02, s1 in the DIE
  class, |norm - 9.1788| <= 0.184 (2%), |in-room - 0.9442| <= 0.05,
  |gm12 - 0.1053| <= 0.02 — the family's disclosed session-texture
  tolerances (e261's anchor convention; cross-pass cuBLAS noise).
* G_DRAWSHARED: the Dmix generator state md5 at completion IDENTICAL
  across every install arm; the corpus cgen state md5 identical across
  every rung; the cons gen state md5 identical across every maintenance
  arm (the streams are bit-shared; the optimizer is the only delta).
* NO CONS in the formation (e324's census convention, disclosed): the
  texture axes (g0/norm/in-room/gm12/s1) are INSTALL-phase quantities;
  the canon's 0.2646 is the install-phase read.
* n=1 per arm, one lineage, one session (the g-series standing lottery
  caveat carried verbatim); the cross is the object, not any point.

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier; the
owner's GPU-aggressive window); bursts <= 175s (dispatch 180), per-step
thermal polls at a 78C margin, 40s cooldowns (dispatch 30-60), the 84C
never-past line (dispatch 85), polls persisted to runs/_envelope_log.jsonl
tagged e347:<arm>; gpu_ok() before every burst; every training segment
SEQUENTIAL — never concurrent; CPU fp64 dense projections (pocketfft
workers 2), CPU threads 4 (the x66 oracle runs on CPU — shared, never
touched); an oracle-cell sibling owns NO CUDA here (this script stages
only its own runs/e347/ + runs/checkpoints/e347_* paths).

Outputs: runs/e347/{metrics.json (PROGRESSIVE), e347_optimizer_cross.png,
REPORT.md (executor-written), run.log (gitignored)}; checkpoints
runs/checkpoints/e347_*_resume.pt + e347_*_final.pt (gitignored; md5s in
metrics). No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat
folds). Commit + push per phase.

Run:  cd lab && python e347_optimizer_cross.py    (E347_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")       # e228's offline convention
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:                                            # Windows console safety
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                               # noqa: BLE001
    pass

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import (CharCorpus, cosine_lr, run_dir,  # noqa: E402
                    save_json, set_seed)

import e043_install as E43                             # noqa: E402 (REPO,
                                                      # find_occ, SPLICE_RNG,
                                                      # CORP_BS, MIX_RANDOM,
                                                      # LR, jsonable)
import g1b_continuity as GB                            # noqa: E402 — MUST be
                                                      # imported BEFORE G1
import g1_anchored_ball as G1                          # noqa: E402
import e261_rank_ladder as E261                        # noqa: E402 — THE
                                                      # COMMITTED INSTALL RIG

torch.set_num_threads(4)           # shared machine (the x66 CPU oracle live)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E347_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e347_smoke" if SMOKE else "e347"
assert torch.cuda.is_available(), "e347 owns the GPU lane (dispatch)"
assert common.gpu_ok(), "gpu_ok() gate at startup (dispatch: check BEFORE launch)"

T0 = time.time()
RD = run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ---- THE REBINDING (the family's disclosed convention): AFTER the e261
# import, so the burst machinery + thermal polls label THIS cell (tags
# e347:...). The e261 import's own append handle on runs/e261/run.log is
# opened but never written through (e323/e324's convention; disclosed).
G1.log = log                                          # unify the timeline
device_events: list[dict] = []
thermal_log: list[dict] = []
E261.log = log
E261.NAME = NAME
E261.SMOKE = SMOKE
E261.T0 = T0
E261.thermal_log = thermal_log
E261.device_events = device_events

# ======================================================================
# THE CONFIG (frozen)
# ======================================================================
BASE_CK = "e001.pt"               # THE CANON'S FORMATION SUBSTRATE (x32's
                                  # NET0-correction: e264/e272's canon ran
                                  # chunked_install from THE BASE, not the
                                  # root; smoke-verified bit-exact here)
ROOT_CK = "g1c_root.pt"           # the family's reference root (bound as a
                                  # parent record; NOT this cell's net0)
SPAN_CK = "e246_late_span.pt"     # e246's committed LATE span (the ledger's)
VMAP_CK = "e258_vmap.pt"          # e258's committed 2.74M v-map (the ledger's v)
ROOMS264_CK = "e264_rooms.pt"     # e264's committed rooms (THE held room's record)
FACT_CK = "e261_K10K_inst_resume.pt"   # THE committed K10K fact (e290's vehicle)
CKPT_DIR = GB.CKPT_DIR

# ---- THE HELD ROOM (the canon's own room; the cross's fixed frame) ----
HELD_ROOM_SEEDS = (26113, 26114)      # e261's registered K10K pair (e264's rung)
ROOM_K = 10_000 if not SMOKE else 512
ROOM_MODE = "K10K"
LADDER: tuple[tuple[int, int, int], ...] = (
    (ROOM_K, HELD_ROOM_SEEDS[0], HELD_ROOM_SEEDS[1]),)
E261.LADDER = LADDER                  # the machinery's certify()/rooms read it
E261.RUNG_NAMES = {ROOM_K: ROOM_MODE}

# ---- THE DOSE / STREAMS ------------------------------------------------
INST_STEPS = 400 if not SMOKE else 8       # the committed standard dose
INST_TOTAL = 1000                          # e048's house-cosine total (verbatim)
INST_GEN = 24314                           # THE CANON'S OWN draw (VERBATIM)
CORPUS_GEN_SEED = 34701                    # phase-3 corpus stream (fresh, per-cell)
CONS_GEN_SEED = 34702                      # phase-4 maintenance stream (fresh)
RUNG_STEPS = 400 if not SMOKE else 8       # e290's phase length
CONS_STEPS = 300 if not SMOKE else 8       # the cons/walk length
MILESTONES = (100, 200, 300, 400) if not SMOKE else tuple(range(1, 9))
FATE_STEPS = (1, 25, 125, 300) if not SMOKE else tuple(range(1, 9))

# ---- THE OPTIMIZER SET (frozen) -----------------------------------------
OPTIMIZERS: tuple[str, ...] = ("SGD", "MSGD", "RMSPROP", "SIGN")
LADDERS: dict[str, tuple[float, ...]] = {
    "SGD":     (0.03, 0.1, 0.3, 1.0, 3.0),
    "MSGD":    (0.01, 0.03, 0.1, 0.3, 1.0),
    "RMSPROP": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    "SIGN":    (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
}
if SMOKE:      # every class exercised, cheaply
    LADDERS = {k: (v[0], v[3]) for k, v in LADDERS.items()}
RMSPROP_ALPHA = 0.99
RMSPROP_EPS = 1e-8
MSGD_MOMENTUM = 0.9
DECAY_PER_STEP_VELOCITY = E43.LR * 0.1    # 1e-4 — AdamW@1e-3 wd0.1's own

# ---- THE BRACKET (frozen) -----------------------------------------------
BUDGET_FRACS = (0.0008, 0.004)            # Law 2b's two boundary rungs
WRITE_NORM_LITERAL = 9.1788432658723      # the committed fact's write norm
E290_REF = {                              # e290's committed SGD-M rows (asserted)
    0.0008: {"ratio_t400": 0.7673992671930195, "holds": True},
    0.004:  {"ratio_t400": 0.31939051143675057, "holds": False},
}
SURVIVAL_HOLD_BAR = 0.5                   # e290's holds convention
BUDGET_SLACK = 1e-5                       # e290's fp32 accumulation slack
ORTH_BAR = 1e-6                           # e290's G_ORTH bar
BUFSEP_BAR = 1e-4                         # e290's G_BUFSEP composition bar

# ---- THE BARS / BANDS (frozen) ------------------------------------------
CANON_READ_BAND = (0.15, 0.45)            # e272's trust band (the canon in-band)
BASE_STANDING_G0_PRIOR = 1.3383959412749391e-05   # the base's Z-prior at the
                                                  # g0 battery (computed at the
                                                  # smoke repair; x32's
                                                  # committed prior class)
CANON_POST_G0 = 0.26464763283729553
CANON_POST_GM12 = 0.10525520890951157
CANON_WRITE_NORM = 9.1788432658723
CANON_IN_OWN_ROOM = 0.9441588788935659
CANON_S1_G0 = 1.3466433301800862e-05
TEXTURE_NORM_BAND = (0.5 * CANON_WRITE_NORM, 2.0 * CANON_WRITE_NORM)
TEXTURE_INROOM_ABS = 0.10
S1_DIE_BAR = 0.01
S1_SURVIVE_BAR = 0.15
ANCHORW_TOL = {"g0": 0.02, "norm": 0.184, "inroom": 0.05, "gm12": 0.02}
G_READ_TOL = E261.G_READ_TOL              # 5e-3
RATIO_DEN_FLOOR = 1e-6

# ---- THE PARENT RECORDS, HARD-BOUND (read at runtime; Rule 12). The JSON
# records are bound on their GIT-CANONICAL md5s (newline-normalized bytes —
# the OneDrive CRLF convention, e290's repair); the .pt artifacts RAW.
E261_METRICS = E43.REPO / "runs" / "e261" / "metrics.json"
E261_MD5 = "a1aee7420d08bb257db5af5f45625f6f"          # verified at birth
E264_METRICS = E43.REPO / "runs" / "e264" / "metrics.json"
E264_MD5 = "1149f97c633072c3b865b6a93bfa1e44"          # verified at birth
E272_METRICS = E43.REPO / "runs" / "e272" / "metrics.json"
E272_MD5 = "856d3f0da57f4a6b5f7f033d130e9a85"          # verified at birth
E290_METRICS = E43.REPO / "runs" / "e290" / "metrics.json"
E290_MD5 = "fc2ba5767e65090e09278e8b55454ef0"          # verified at birth
E323_METRICS = E43.REPO / "runs" / "e323" / "metrics.json"
E323_MD5 = "bf01f994cda19c9b04e151f8e2c457aa"          # verified at birth
X57_METRICS = E43.REPO / "runs" / "x57" / "metrics.json"
X57_MD5 = "bd424fa19a48b8fa378d9f93f95c706f"           # verified at birth
FACT_CK_MD5 = "0f6dc1cf46850ce655dfafc9c853d467"       # e290's bound literal
FACT_FLAT_MD5 = "ebebb4472725d582dd74928493f1bfb3"     # verified at the smoke
                                                       # repair (bit-exact load)
ROOMS264_MD5 = "2d524655575cce00a3bc1c8770f4b211"      # e324's bound literal
SPAN_MD5 = "3464ff6081d8e402103dd9ca65bfb53a"          # e261's E246_SPAN_MD5
SPAN_RANK = 10
ROOT_CK_MD5 = "9c7d4ca1b60c8a1158d080f932e2c95f"       # verified at birth
BASE_CK_MD5 = "d114536d1c0983ab3be67f67ff0667c8"       # verified at birth
VMAP_MD5 = "baafd327e9e35cffa2d41ea4ecf5fc5c"          # verified at birth
# the instrument parents (the rig + the vehicles), git-canonical md5s
RIG_MD5S = {
    "lab/e261_rank_ladder.py": "e498031fe4fcd0147c3094c86034b1c5",
    "lab/e290_coupling_ladder.py": "99d8644de308cb25816f779f8bc82215",
    "lab/g1_anchored_ball.py": "8d0179b646ac6e4b30db26f486c4dab6",
    "lab/e043_install.py": "a34c945da8988fa69c66c3fd40441ca6",
    "lab/g1b_continuity.py": "66966621e162b6bd33cf0023b0aa485a",
    "lab/common.py": "022b0f3fea0b1d8444c8c9f6dbffa798",
}

VOCAB_EXPECT = 65

REGISTERED = {
    "question_verbatim": (
        "is the grammar the training dynamics' (transport-invariant) or "
        "AdamW's? YOUR CELL: the first cross. (THE OPTIMIZER CROSS — the "
        "substrate program's first GPU cell)"),
    "background_verbatim": (
        "every law this lab has measured was measured under AdamW on one "
        "2.74M architecture with one CE loss — Law 1 (every death is a "
        "step-death), Law 2b's kill bracket [0.0008, 0.004] of write norm, "
        "the formation texture (the wheel, the net0 classes), the seat "
        "grammar (maintenance, capacity). NONE has ever been crossed against "
        "a different optimizer."),
    "design_verbatim": (
        "1. THE OPTIMIZER SET: SGD (no momentum, no adaptivity — the pure "
        "geometry), momentum-SGD (0.9), RMSProp (adaptive but no moment "
        "terms), and SIGN-DESCENT (the extreme: pure direction, "
        "magnitude-normalized — the lab's own simplest custom optimizer). "
        "Per-optimizer lr sweeps at birth (a coarse 5-point ladder each, the "
        "canon formation at the standard dose; the lr that lands in the "
        "canon's read band becomes that optimizer's canonical lr — the DOSE "
        "NORMALIZATION, pre-registered). 2. THE FORMATION TEXTURE: the canon "
        "name's formation under each optimizer at its canonical lr — the "
        "read trajectory, the write norm, the in-room fraction, the s1 "
        "behavior (the wheel? the classes?), vs the committed AdamW record. "
        "3. THE KILL BRACKET: Law 2b's two boundary rungs (the 0.0008x and "
        "0.004x transport-kills) re-run under each optimizer in ITS "
        "canonical-lr terms (the drift-realization disclosed) — does the "
        "bracket MOVE (per-optimizer renormalized) or HOLD? 4. THE "
        "MAINTENANCE PROBE: the tonly/conly split (x57's role split — the "
        "cheapest seat probe: does attention-only protect under each "
        "optimizer?)"),
    "bars_verbatim": {
        "TRANSPORT-INVARIANT": (
            "the laws hold under every optimizer at its canonical lr (the "
            "formation texture within the disclosed band; the bracket within "
            "2x after lr-normalization; maintenance attention-side "
            "throughout) — the grammar is the loss-geometry's."),
        "OPTIMIZER-SPECIFIC": (
            "which laws move first (the texture? the bracket? the "
            "maintenance?) — named per optimizer; the grammar re-scopes to "
            "AdamW-class dynamics."),
        "MIXED": "some hold, some move — the honest split, itemized.",
    },
    "parity_block_verbatim": {
        "LAB_LEAN": (
            "OPTIMIZER-SPECIFIC, moderately (the adaptive vs non-adaptive "
            "step geometry differs hugely; the bracket especially should "
            "move — it is measured in write-norm units that mean different "
            "things per optimizer)"),
        "COUNTER": (
            "TRANSPORT-INVARIANT (the loss geometry dominates; the optimizer "
            "is a solver, and solvers converge to the same minima basin)"),
    },
    "registration": (
        "question + background + design + bars + parity block frozen VERBATIM "
        "from the dispatch letter (the substrate program's first GPU cell; "
        "the owner's 2026-10-12 directive); this script committed at birth "
        "BEFORE any compute; adjudicate against exactly this; no bar "
        "shopping."),
}

deviations: list[str] = [
    "THE NET0 REPAIR (smoke-caught, 2026-10-12, the session's biggest "
    "catch): the birth draft followed e324's docstring claim that the "
    "canon formed from g1c_root — the smoke's G_ANCHORW-class probe "
    "contradicted it (root-started s1 CE 1.1963 vs the committed "
    "1.0701713562011719), and the lab's own committed record (x32 "
    "7bcd060, THE NET0 CONFOUND; x34; e328's GPU control) proves the "
    "canon ran chunked_install from THE BASE e001 with the s1 'die' = "
    "the base's prior staying flat (prior 1.3383959412749391e-05). "
    "Base-started CE/kept/s1 verified BIT-IDENTICAL on CPU at the "
    "repair. The formation net0 is THE BASE; the bars + P-e347a frozen "
    "VERBATIM (only the substrate statement corrected); the s1 classes "
    "re-word per x32 (DIE = prior-flat / SURVIVE = a step-one read "
    "event); the stale root-started smoke ckpts deleted (regenerable "
    "scratch).",
    "THE FORMATION RUNS THROUGH e261's chunked_install STEP BODY PORTED "
    "VERBATIM (draws, Dmix arithmetic, masked union CE, clip, the room "
    "hook, the ledger/traj cadence, the resume discipline) WITH THE "
    "OPTIMIZER CONSTRUCTION SWAPPED — the import-and-drive form cannot "
    "swap the optimizer (e261 constructs AdamW inside chunked_install); "
    "the port is quote-checked against e261's source (G_QUOTES). The "
    "ADAMW anchor arm's ported path at lr 1e-3 must reproduce the "
    "committed canon within the session texture (G_ANCHORW) or the cell "
    "reads TEXTURE.",
    "THE DECAY CHANNEL UNIFORM (disclosed): crossed arms carry the "
    "committed per-step decoupled decay velocity (1 - 1e-4 x cosine_t) "
    "post-step; built-ins run wd=0; the ADAMW anchor uses torch AdamW "
    "wd 0.1 (bit-faithful in-step form). The post-vs-in-step ordering "
    "difference on crossed arms is ~1e-4-relative, disclosed, never a "
    "channel delta.",
    "THE BRACKET'S COMMITTED REFERENCE NOT RERUN (e324's canon "
    "convention): e290's SGD-M rows (0.7674 holds / 0.3194 dies) are "
    "READ from runs/e290/metrics.json at runtime + asserted vs the birth "
    "literals; the crossed rungs' vehicle = each optimizer at ITS "
    "canonical lr with e290's equal-share cap, orthogonalization, "
    "milestones, and twin-instance isolation VERBATIM (ported + "
    "quote-checked); NO decay in the rungs (e290's wd-0 form).",
    "THE MAINTENANCE PROBE'S ADAMW REFERENCE IS IN-SESSION (disclosed): "
    "x57's committed split was the TAVIREN subject on e345's stream "
    "(tonly 0.99996..1.0 protects / conly 0.0035..0.0008 executes); this "
    "cell's fact is the K10K ZEPHYRA canon — the AdamW@1e-3 tonly/conly "
    "pair here is the internal reference row, the committed x57 numbers "
    "cited as context, never asserted.",
    "NO CONS IN THE FORMATION (e324's census convention): the texture "
    "axes are install-phase quantities; the canon's 0.2646 is the "
    "install-phase read.",
    "SMOKE trims: 8-step segments, room k=512 (the e264_rooms.pt "
    "bit-bind is VACUOUS at smoke — disclosed, LIVE at the full run), "
    "2-point ladders, 2-optimizer bracket/maintenance subsets — every "
    "code class exercised, NOTHING adjudicated (SMOKE stamp on every "
    "read).",
    "n=1 per arm, one lineage, one session (the g-series standing "
    "lottery caveat carried verbatim); the cross is the object.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat "
    "folds).",
]


# ------------------------------------------------------------- THE SIGN OPT
class SignDescent(torch.optim.Optimizer):
    """THE LAB'S OWN SIMPLEST OPTIMIZER (stated per the dispatch): pure
    direction, magnitude-normalized — update = p -= lr * sign(grad). NO
    state, NO moments, NO adaptivity. The extreme point of the set."""

    def __init__(self, params, lr: float):
        super().__init__(list(params), {"lr": float(lr)})

    @torch.no_grad()
    def step(self, closure=None):                    # noqa: D102
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.add_(torch.sign(p.grad), alpha=-group["lr"])


def make_opt(kind: str, params, lr: float):
    """The frozen optimizer set. ADAMW = the committed canon class (the
    anchor arm only)."""
    if kind == "SGD":
        return torch.optim.SGD(params, lr=lr)
    if kind == "MSGD":
        return torch.optim.SGD(params, lr=lr, momentum=MSGD_MOMENTUM)
    if kind == "RMSPROP":
        return torch.optim.RMSprop(params, lr=lr, alpha=RMSPROP_ALPHA,
                                   eps=RMSPROP_EPS, momentum=0.0,
                                   centered=False)
    if kind == "SIGN":
        return SignDescent(params, lr=lr)
    if kind == "ADAMW":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95),
                                 weight_decay=0.1)
    raise KeyError(kind)


OPT_STATE_KEY = {"MSGD": "momentum_buffer", "RMSPROP": "square_avg"}


def snap_state(opt, params) -> list:
    """Snapshot an optimizer's per-param state tensors (None where empty) —
    the G_BUFSEP isolation probe's eyes (e290's snap_buffers, generalized to
    RMSProp's square_avg)."""
    st = opt.state
    key = OPT_STATE_KEY.get(_opt_kind(opt))
    return [st[p][key].detach().clone()
            if isinstance(st.get(p), dict) and key in st[p] else None
            for p in params]


def states_bitwise_equal(sn_a: list, sn_b: list) -> bool:
    if len(sn_a) != len(sn_b):
        return False
    for a, b in zip(sn_a, sn_b):
        if (a is None) != (b is None):
            return False
        if a is not None and not torch.equal(a, b):
            return False
    return True


def _opt_kind(opt) -> str:
    return getattr(opt, "E347_KIND", type(opt).__name__)


def pending_step_sqnorm(kind: str, opt_C, params) -> float:
    """||b_t||^2 — the would-be unit-lr update direction's squared norm,
    computed from the stored state + the (already-processed) grads in
    p.grad, for EVERY optimizer class in the set (e290's
    pending_buffer_sqnorm, generalized):
      SGD     b = g
      MSGD    b = 0.9 * buf + g
      RMSPROP b = g / (sqrt(alpha*v + (1-alpha)*g^2) + eps)
      SIGN    b = sign(g)  (=> ||b||^2 = #nonzero coords)"""
    sq = torch.zeros((), device=next(iter(params)).device)
    st = opt_C.state
    for p in params:
        g = p.grad.detach()
        if kind == "SGD":
            v = g
        elif kind == "MSGD":
            d = st.get(p)
            buf = d.get("momentum_buffer") if isinstance(d, dict) else None
            v = MSGD_MOMENTUM * buf + g if buf is not None else g
        elif kind == "RMSPROP":
            d = st.get(p)
            vprev = d.get("square_avg") if isinstance(d, dict) else None
            vnew = (RMSPROP_ALPHA * vprev + (1.0 - RMSPROP_ALPHA) * g.square()
                    if vprev is not None
                    else (1.0 - RMSPROP_ALPHA) * g.square())
            v = g / (vnew.sqrt() + RMSPROP_EPS)
        elif kind == "SIGN":
            sq = sq + (g != 0).sum()
            continue
        else:
            raise KeyError(kind)
        sq = sq + v.square().sum()
    return float(sq.item())


# ------------------------------------------------- THE VEHICLE MACHINES (port)
def orthogonalize_grads(proj: "E261.LadderRooms", params, mode: str) -> dict:
    """e278/e280/e284/e285/e290's orthogonalize_grads VERBATIM: replace the
    (clipped) corpus gradient g by g_perp = g - P_room(g) — the component
    ENTIRELY ORTHOGONAL to the room (CPU fp64, write fp32, norm NOT
    rescaled). The verification read ||P_room g_perp|| / ||g_perp|| is
    returned for the gate (any drift above 1e-6 is an implementation bug)."""
    room = proj.rooms[mode]
    params = list(params)
    g = torch.cat([p.grad.detach().reshape(-1) for p in params]) \
        .to(CPU).double().numpy().astype(np.float64)
    gn2 = float(g @ g)
    gp_in = room.project(g)                 # P_room(g) — the in-room part
    gperp = g - gp_in                       # the orthogonal complement
    gpn2 = float(gperp @ gperp)
    resid = room.project(gperp)             # must be ~0 (the verification)
    rel = (float(np.sqrt(resid @ resid)) / float(np.sqrt(gpn2))
           if gpn2 > 0 else 0.0)
    gp32 = torch.from_numpy(gperp.astype(np.float32))
    with torch.no_grad():
        for p, (a, b), shp in zip(params, proj.offsets, proj.shapes):
            p.grad.copy_(gp32[a:b].to(proj.dev).reshape(shp))
    return {"gn": math.sqrt(gn2), "gperp_norm": math.sqrt(gpn2),
            "norm_ratio": (math.sqrt(gpn2 / gn2) if gn2 > 0 else 0.0),
            "in_room_frac": (math.sqrt(max(0.0, 1.0 - gpn2 / gn2))
                             if gn2 > 0 else 0.0),
            "orth_rel_err": rel}


# ------------------------------------------------------------------ helpers
def flat_params_cpu(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()]).clone()


def save_ckpt(name: str, sd: dict, meta: dict) -> str:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": NAME, **meta}}, path)
    log(f"[ckpt] saved {path.name}")
    return str(path.relative_to(E43.REPO)).replace("\\", "/")


def md5of(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def md5of_norm(p: Path) -> str:
    """md5 of the NEWLINE-NORMALIZED bytes (CRLF -> LF) — the git canonical
    form (e290's OneDrive CRLF repair; JSON parents bound on this hash)."""
    return hashlib.md5(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def gen_state_md5(gen: torch.Generator) -> str:
    return hashlib.md5(gen.get_state().numpy().tobytes()).hexdigest()


def git_head() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(E43.REPO),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                       # noqa: BLE001
        return "unavailable"


def s1_class(v: float) -> str:
    if v < S1_DIE_BAR:
        return "DIE"
    if v > S1_SURVIVE_BAR:
        return "SURVIVE"
    return "GAP"


def file_tag(s: str) -> str:
    """Sanitize an arm tag for FILE names (no '@'/'.'/'-'): SGD@0.3 ->
    SGD_0p3."""
    return (s.replace("@", "_").replace(".", "p").replace("-", "m"))


# ------------------------------------------------------------------ main
metrics: dict = {}


def write_partial(note: str) -> None:
    metrics["date"] = common.now_iso()
    metrics["phase_note"] = note
    metrics["device_events"] = device_events
    save_json(RD / "metrics.json", E43.jsonable(metrics))
    log(f"WROTE partial metrics ({note})")


def _envelope_summary() -> dict:
    """Aggregate THIS cell's polls from runs/_envelope_log.jsonl (tags
    e347:*) — the owner-envelope audit line."""
    temps = []
    try:
        with open(E43.REPO / "runs" / "_envelope_log.jsonl",
                  encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(row.get("tag", "")).startswith(f"{NAME}:") \
                        and row.get("temp") is not None:
                    temps.append(float(row["temp"]))
    except FileNotFoundError:
        pass
    return {"n_polls": len(temps),
            "max_temp_seen_c": max(temps) if temps else None,
            "violations_ge_84c": sum(1 for t in temps
                                     if t >= E261.TEMP_HARD),
            "bursts": f"<= {E261.BURST_MAX_S:.0f}s, cooldowns "
                      f"{E261.COOLDOWN_S:.0f}s, margin "
                      f"{E261.TEMP_EARLY_END:.0f}C, never-past "
                      f"{E261.TEMP_HARD:.0f}C (dispatch 180/30-60/85)"}


# ======================================================================
# P1 DRIVER — the chunked install, optimizer-swapped (e261's step body
# VERBATIM; quote-checked in G_QUOTES)
# ======================================================================
def chunked_install_x(tag: str, kind: str, lr_base: float, net0, proj,
                      inst_x, inst_mask, anchor_full, train_ids, g0_ids,
                      gm12_ids, r_eval_xy, zid, resume_ck: Path,
                      dev: torch.device) -> dict:
    """e261's chunked_install arithmetic VERBATIM (E43.exposure Dmix:
    ix(16) install w/ name mask + aj(16) paired + rj(32) random; masked
    token-level union CE; clip 1.0 -> P_room hook -> opt.step; lr x house
    cosine(total=1000)) with the OPTIMIZER the parameter (kind/lr_base).
    The traj carries BOTH batteries (g0 the expression ruler; g-12 the
    wash ruler) at the committed cadence (s1 + every 100 + final)."""
    name_bs, corp_bs, mix_random = (G1.NAME_BS, E43.CORP_BS, E43.MIX_RANDOM)
    n_steps = INST_STEPS
    n_inst, n_anc = inst_x.shape[0], anchor_full.shape[0]
    state = {"step": 0, "traj": [], "ledger": {}}
    if resume_ck.exists():
        state = torch.load(resume_ck, map_location="cpu", weights_only=False)
        log(f"  [{tag}] RESUMED from {resume_ck.name} at step "
            f"{state['step']}/{n_steps}")
    if int(state.get("step", 0)) >= n_steps:
        log(f"  [{tag}] resume ckpt already COMPLETE at s{state['step']}")
        return {"sd": state["model"], "traj": state.get("traj", []),
                "ledger": state.get("ledger", {}),
                "steps_ran": n_steps,
                "chunk_table": state.get("chunk_table", []),
                "gen_final_md5": state.get("gen_final_md5"),
                "resumed_final": True}
    net, opt, gen, evl = None, None, None, None
    n_chunks, chunk_table = 0, []
    step = state["step"]
    t_burst, n_burst = None, 0
    chunk_temps: list[float] = []
    while step < n_steps:
        if t_burst is None:
            t_burst, n_burst = E261._open_burst(f"{tag}-chunk{n_chunks + 1}")
        n_chunks += 1
        chunk_temps = []
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            opt = make_opt(kind, net.parameters(), lr_base)
            opt.E347_KIND = kind
            gen = torch.Generator().manual_seed(INST_GEN)
            if resume_ck.exists():
                net.load_state_dict(state["model"])
                net.to(dev)
                net.train()
                opt.load_state_dict(state["opt"])
                gen.set_state(state["gen_state"])
                step = state["step"]
            evl = copy.deepcopy(net0).to(CPU)
        devices_row = {"chunk": n_chunks, "device": str(dev),
                       "t_start": round(time.time() - T0, 1)}
        chunk_capped = False
        for step in range(step + 1, n_steps + 1):
            f = cosine_lr(step - 1, INST_TOTAL)          # house schedule
            for g in opt.param_groups:
                g["lr"] = lr_base * f
            ix = torch.randint(n_inst, (name_bs,), generator=gen)
            aj = torch.randint(n_anc, (corp_bs - mix_random,), generator=gen)
            rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (mix_random,),
                               generator=gen)
            corp = torch.cat([anchor_full[aj],
                              torch.stack([train_ids[s: s + G1.BLOCK]
                                           for s in rj])], 0)
            nw = inst_x[ix]
            x = torch.cat([nw[:, :-1], corp[:, :-1]], 0).to(dev)
            y = torch.cat([nw[:, 1:], corp[:, 1:]], 0).to(dev)
            m = torch.zeros(name_bs + corp_bs, x.shape[1], dtype=torch.bool,
                            device=dev)
            m[:name_bs] = inst_mask[ix].to(dev)
            logits, _ = net(x)
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                  y.reshape(-1), reduction="none"
                                  ).view(x.shape[0], x.shape[1])
            nm = nll[:name_bs][m[:name_bs]]
            cm = nll[name_bs:]
            loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            # ---- THE HOOK (the room; VERBATIM) --------------------------
            led = proj.step_hook(list(net.parameters()), ROOM_MODE)
            opt.step()          # BOTH moments + the update see the hook's g'
            if kind != "ADAMW":  # the uniform decoupled decay velocity
                with torch.no_grad():
                    fac = 1.0 - DECAY_PER_STEP_VELOCITY * f
                    for p in net.parameters():
                        p.mul_(fac)
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                state["ledger"][step] = {
                    "ce": float(loss.item()), "gn": led["gn"],
                    "gpn": led["gpn"], "kept_frac": led["kept_frac"],
                    "v_excess_pre": led["v_excess_pre"],
                    "v_excess_post": led["v_excess_post"],
                    "in_span_frac": led["in_span_frac"],
                    "applied_in_span_frac": led["applied_in_span_frac"]}
            if step % 100 == 0 or step == n_steps or step == 1 or SMOKE:
                sd_cpu = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
                evl.load_state_dict(sd_cpu)
                evl.eval()
                bz0 = G1.battery_cell(evl, g0_ids, zid)
                bz12 = G1.battery_cell(evl, gm12_ids, zid)
                ce_r = G1.ce_fixed_cpu(evl, *r_eval_xy)
                state["traj"].append({"step": step,
                                      "g0_pz": bz0["mean_pz"],
                                      "g0_argmax": bz0["frac_argmax_z"],
                                      "gm12_pz": bz12["mean_pz"],
                                      "ce_r": ce_r,
                                      "ce_batch": float(loss.item()),
                                      "kept_frac": led["kept_frac"],
                                      "v_excess_post": led["v_excess_post"],
                                      "in_span_frac": led["in_span_frac"],
                                      "applied_in_span_frac":
                                          led["applied_in_span_frac"],
                                      "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] s{step:4d} g0 {bz0['mean_pz']:.4f} g-12 "
                    f"{bz12['mean_pz']:.4f} CE_R {ce_r:.4f} CE "
                    f"{float(loss.item()):.4f} |g| {led['gn']:.3f} kept "
                    f"{led['kept_frac']:.4f} vexc pre "
                    f"{led['v_excess_pre']:.2f} post "
                    f"{led['v_excess_post']:.2f} inspan "
                    f"{led['in_span_frac']:.3f}->"
                    f"{led['applied_in_span_frac']:.3f}")
            n_burst += 1
            # per-step thermal polls (the e233..e324 discipline)
            ok_t, temp = E261.burst_temp_check(f"{tag}-c{n_chunks}")
            chunk_temps.append(temp)
            if not ok_t:
                E261._end_burst_early(tag, n_burst, t_burst)
                chunk_capped = True
                break
            if (time.time() - t_burst) > E261.BURST_MAX_S:
                log(f"  [{tag}] chunk {n_chunks}: burst cap "
                    f"{E261.BURST_MAX_S:.0f}s at s{step} — resume ckpt saved")
                chunk_capped = True
                break
        chunk_table.append({**devices_row,
                            "seconds": round(time.time() - t_burst, 1),
                            "steps_done": step, "capped": chunk_capped,
                            "max_temp_c": max(chunk_temps, default=None)})
        torch.save({"model": {k: v.detach().cpu()
                              for k, v in net.state_dict().items()},
                    "opt": opt.state_dict(), "gen_state": gen.get_state(),
                    "step": step, "traj": state["traj"],
                    "ledger": state["ledger"], "n_chunks": n_chunks,
                    "chunk_table": chunk_table,
                    "gen_final_md5": (gen_state_md5(gen)
                                      if step >= n_steps else None),
                    "opt_kind": kind, "lr_base": lr_base}, resume_ck)
        if step >= n_steps:
            break
        if n_chunks >= 12:
            raise RuntimeError(f"{tag}: cap-loop guard at chunk {n_chunks}")
        E261.burst_cooldown(tag)
        t_burst = None
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    return {"sd": sd_cpu, "traj": state["traj"], "ledger": state["ledger"],
            "steps_ran": step, "chunk_table": chunk_table,
            "gen_final_md5": gen_state_md5(gen)}


# ======================================================================
# P3 DRIVER — the budgeted transport rung, optimizer-swapped (e290's
# chunked_ladder_phase body; quote-checked in G_QUOTES)
# ======================================================================
def chunked_rung_x(tag: str, kind: str, lr_base: float, budget_norm: float,
                   net0, proj, anchor_full, train_ids, g0_ids, gm12_ids,
                   r_eval_xy, zid, resume_ck: Path,
                   dev: torch.device) -> dict:
    """e290's rung driver VERBATIM in body with the vehicle optimizer the
    parameter: per corpus step t = 1..400 — draws (aj_c(16), rj_c(32)) from
    cgen (seed 34701, bit-identical across every rung); corpus batch 16
    original-host anchors + 32 random windows; full-window CE; clip 1.0;
    g_perp = g - P_room(g) (CPU fp64, verified every step); the equal-share
    per-step lr cap against ||b_t|| (the optimizer's would-be direction);
    lr_t = min(lr_base x cosine(t-1,1000), cap); opt_C stepped, opt_F the
    never-stepped bitwise-isolation twin; S += realized ||step||; milestone
    batteries + drift reads at t100/200/300/400."""
    n_steps = RUNG_STEPS
    n_par_tot = sum(p.numel() for p in net0.parameters())
    corp_bs, mix_random = E43.CORP_BS, E43.MIX_RANDOM
    n_anc = anchor_full.shape[0]
    state = {"step": 0, "traj": [], "ledger": {}, "orth_ledger": {},
             "lr_ledger": {}, "disp_ledger": [], "buf_ledger": [],
             "S": 0.0, "n_capped": 0, "orth_max": 0.0,
             "bufsep": {"isolation_checks": 0, "isolation_violations": 0},
             "corp_cum": torch.zeros(n_par_tot),
             "corp_prev": torch.zeros(n_par_tot)}
    if resume_ck.exists():
        state = torch.load(resume_ck, map_location="cpu", weights_only=False)
        log(f"  [{tag}] RESUMED from {resume_ck.name} at step "
            f"{state['step']}/{n_steps}")
    if int(state.get("step", 0)) >= n_steps:
        log(f"  [{tag}] resume ckpt already COMPLETE at t{state['step']}")
        return _rung_pack(state, n_steps)
    net = opt_C = opt_F = cgen = evl = None
    n_chunks, chunk_table = 0, []
    step = state["step"]
    t_burst, n_burst = None, 0
    chunk_temps: list[float] = []
    room = proj.rooms[ROOM_MODE]
    corp_cum = state["corp_cum"].to(dev)
    corp_prev = state["corp_prev"].to(dev)
    S = float(state["S"])
    n_capped = int(state["n_capped"])
    orth_max = 0.0

    while step < n_steps:
        if t_burst is None:
            t_burst, n_burst = E261._open_burst(
                f"{tag}:corpus:chunk{n_chunks + 1}")
        n_chunks += 1
        chunk_temps = []
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            # the separation: TWO instances over the SAME parameters — the
            # corpus side's state is PRIVATE; the fact side never steps.
            opt_C = make_opt(kind, net.parameters(), lr_base)
            opt_C.E347_KIND = kind
            opt_F = make_opt(kind, net.parameters(), lr_base)
            opt_F.E347_KIND = kind
            cgen = torch.Generator().manual_seed(CORPUS_GEN_SEED)
            if resume_ck.exists():
                net.load_state_dict(state["model"])
                net.to(dev)
                net.train()
                opt_C.load_state_dict(state["optC"])
                cgen.set_state(state["cgen_state"])
                step = state["step"]
                S = float(state["S"])
                n_capped = int(state["n_capped"])
            evl = copy.deepcopy(net0).to(CPU)
        devices_row = {"chunk": n_chunks, "device": str(dev),
                       "t_start": round(time.time() - T0, 1)}
        chunk_capped = False
        params_live = list(net.parameters())
        for step in range(step + 1, n_steps + 1):
            lr_sched = lr_base * cosine_lr(step - 1, INST_TOTAL)
            # ---- THE CORPUS STEP (bit-identical draws to every rung) ----
            aj_c = torch.randint(n_anc, (corp_bs - mix_random,),
                                 generator=cgen)
            rj_c = torch.randint(len(train_ids) - G1.BLOCK - 1, (mix_random,),
                                 generator=cgen)
            corp_c = torch.cat([anchor_full[aj_c],
                                torch.stack([train_ids[s: s + G1.BLOCK]
                                             for s in rj_c])], 0)
            xc = corp_c[:, :-1].to(dev)
            yc = corp_c[:, 1:].to(dev)
            logits_c, _ = net(xc)
            nll_c = F.cross_entropy(
                logits_c.reshape(-1, logits_c.shape[-1]),
                yc.reshape(-1), reduction="none").view(xc.shape[0],
                                                       xc.shape[1])
            loss_c = nll_c.mean()
            net.zero_grad(set_to_none=True)
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            gn_c = float(torch.cat([p.grad.detach().reshape(-1)
                                    for p in net.parameters()
                                    if p.grad is not None]).norm().item())
            # the RAW gradient's in-room fraction (the re-aiming contrast)
            g_in_room = None
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                g64 = torch.cat([p.grad.detach().reshape(-1)
                                 for p in net.parameters()]) \
                    .to(CPU).double().numpy().astype(np.float64)
                pg = room.project(g64)
                g_in_room = float(np.linalg.norm(pg)
                                  / max(np.linalg.norm(g64), 1e-30))
            # ---- the orthogonalized stream (verified EVERY step) --------
            orth_row = orthogonalize_grads(proj, net.parameters(), ROOM_MODE)
            orth_max = max(orth_max, orth_row["orth_rel_err"])
            # ---- the per-step lr cap (equal-share reservation) ----------
            b_sq = pending_step_sqnorm(kind, opt_C, params_live)
            b_norm = math.sqrt(max(b_sq, 0.0))
            remaining = max(budget_norm - S, 0.0)
            share = remaining / (n_steps - step + 1)
            cap = share / max(b_norm, 1e-12)
            lr_t = min(lr_sched, cap)
            capped = bool(cap < lr_sched)
            if capped:
                n_capped += 1
            for g_ in opt_C.param_groups:
                g_["lr"] = lr_t
            # ---- the isolated corpus step (the fact side EMPTY) ---------
            sn_f_before = snap_state(opt_F, params_live)
            theta_b = torch.cat([p.detach().reshape(-1)
                                 for p in net.parameters()])
            opt_C.step()
            with torch.no_grad():
                d_step = torch.cat([p.detach().reshape(-1)
                                    for p in net.parameters()]) - theta_b
                corp_cum += d_step
            realized = float(d_step.norm().item())
            S += realized
            state["bufsep"]["isolation_checks"] += 1
            if not states_bitwise_equal(sn_f_before,
                                        snap_state(opt_F, params_live)):
                state["bufsep"]["isolation_violations"] += 1
                raise RuntimeError(
                    f"[{tag}] G_BUFSEP ISOLATION VIOLATION at t{step}: the "
                    "corpus step touched the fact side's state — HALT")
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                state["ledger"][step] = {"ce": float(loss_c.item()),
                                         "gn_clipped": gn_c,
                                         "g_in_room_frac": g_in_room}
                state["orth_ledger"][step] = {
                    "gn_clipped": gn_c,
                    "gperp_norm": orth_row["gperp_norm"],
                    "norm_ratio": orth_row["norm_ratio"],
                    "in_room_frac": orth_row["in_room_frac"],
                    "orth_rel_err": orth_row["orth_rel_err"]}
                state["lr_ledger"][step] = {
                    "lr_sched": lr_sched, "cap": cap, "lr_applied": lr_t,
                    "capped": capped, "b_norm": b_norm,
                    "share": share, "realized_step_norm": realized}
            n_burst += 1
            ok_t, temp = E261.burst_temp_check(f"{tag}:corpus:c{n_chunks}.x")
            chunk_temps.append(temp)
            if step in MILESTONES or step == n_steps or SMOKE:
                sd_cpu = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
                evl.load_state_dict(sd_cpu)
                evl.eval()
                bz0 = G1.battery_cell(evl, g0_ids, zid)
                ce_r = G1.ce_fixed_cpu(evl, *r_eval_xy)
                state["traj"].append({"step": step,
                                      "g0_pz": bz0["mean_pz"],
                                      "g0_argmax": bz0["frac_argmax_z"],
                                      "ce_r": ce_r,
                                      "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] t{step:4d} g0 {bz0['mean_pz']:.4f} CE_R "
                    f"{ce_r:.4f} S {S:.6f} capped {n_capped}/{step}")
            if not ok_t or (time.time() - t_burst) > E261.BURST_MAX_S:
                log(f"  [{tag}] chunk {n_chunks} ended "
                    f"({time.time() - t_burst:.1f}s) at t{step}")
                chunk_capped = True
                break
        chunk_table.append({**devices_row,
                            "seconds": round(time.time() - t_burst, 1),
                            "steps_done": step, "capped": chunk_capped,
                            "max_temp_c": max(chunk_temps, default=None)})
        state.update({"model": {k: v.detach().cpu()
                                for k, v in net.state_dict().items()},
                      "optC": opt_C.state_dict(), "cgen_state":
                          cgen.get_state(), "step": step,
                      "traj": state["traj"], "ledger": state["ledger"],
                      "orth_ledger": state["orth_ledger"],
                      "lr_ledger": state["lr_ledger"],
                      "disp_ledger": state["disp_ledger"],
                      "buf_ledger": state["buf_ledger"],
                      "S": S, "n_capped": n_capped, "orth_max": orth_max,
                      "bufsep": state["bufsep"],
                      "corp_cum": corp_cum.cpu(), "corp_prev": corp_prev.cpu(),
                      "n_chunks": n_chunks, "chunk_table": chunk_table,
                      "cgen_final_md5": (gen_state_md5(cgen)
                                         if step >= n_steps else None),
                      "opt_kind": kind, "lr_base": lr_base,
                      "budget_norm": budget_norm})
        torch.save(state, resume_ck)
        if step >= n_steps:
            break
        if n_chunks >= 12:
            raise RuntimeError(f"{tag}: cap-loop guard at chunk {n_chunks}")
        E261.burst_cooldown(tag)
        t_burst = None
    net.eval()
    return _rung_pack(state, n_steps)


def _rung_pack(state: dict, n_steps: int) -> dict:
    _lrs = [v["lr_applied"] for v in state.get("lr_ledger", {}).values()]
    _caps = [v["cap"] for v in state.get("lr_ledger", {}).values()]
    return {"sd": state["model"], "traj": state.get("traj", []),
            "ledger": state.get("ledger", {}),
            "orth_ledger": state.get("orth_ledger", {}),
            "lr_ledger": state.get("lr_ledger", {}),
            "S": state.get("S"), "n_capped": state.get("n_capped"),
            "orth_max": state.get("orth_max"),
            "bufsep": state.get("bufsep", {}),
            "steps_ran": n_steps,
            "chunk_table": state.get("chunk_table", []),
            "cgen_final_md5": state.get("cgen_final_md5"),
            "resumed_final": True}


# ======================================================================
# P4 DRIVER — the maintenance stream with role masks (x57's split, on
# e261's chunked_consolidate batch form), optimizer-swapped
# ======================================================================
def chunked_cons_x(tag: str, kind: str, lr_base: float, role: str, net0,
                   pool_a_x, pool_a_mask, cons_anchor, train_ids, g0_ids,
                   r_eval_xy, zid, resume_ck: Path,
                   dev: torch.device) -> dict:
    """The cons-style maintenance stream (e261's chunked_consolidate batch
    arithmetic VERBATIM: batch 32 = 16 jittered-pool install windows + 8
    paired-original anchors + 8 random corpus; draws from THIS cell's seed
    34702, bit-identical across arms) with x57's ROLE masks:
      tonly — loss over the 112 name-EMISSION tokens alone;
      conly — loss over the 8048-token complement.
    300 steps, const lr = the optimizer's canonical lr, NATURAL (no room
    hook), the uniform decay velocity for crossed arms. Reads at x57's
    fate steps (s1/s25/s125/s300)."""
    n_steps = CONS_STEPS
    n_pool, n_anc = pool_a_x.shape[0], cons_anchor.shape[0]
    state = {"step": 0, "traj": []}
    if resume_ck.exists():
        state = torch.load(resume_ck, map_location="cpu", weights_only=False)
        log(f"  [{tag}] RESUMED from {resume_ck.name} at step "
            f"{state['step']}/{n_steps}")
    if int(state.get("step", 0)) >= n_steps:
        log(f"  [{tag}] resume ckpt already COMPLETE at s{state['step']}")
        return {"sd": state["model"], "traj": state.get("traj", []),
                "steps_ran": n_steps,
                "chunk_table": state.get("chunk_table", []),
                "gen_final_md5": state.get("gen_final_md5"),
                "resumed_final": True}
    net, opt, gen, evl = None, None, None, None
    n_chunks, chunk_table = 0, []
    step = state["step"]
    t_burst, n_burst = None, 0
    chunk_temps: list[float] = []
    while step < n_steps:
        if t_burst is None:
            t_burst, n_burst = E261._open_burst(f"{tag}-chunk{n_chunks + 1}")
        n_chunks += 1
        chunk_temps = []
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            opt = make_opt(kind, net.parameters(), lr_base)
            opt.E347_KIND = kind
            gen = torch.Generator().manual_seed(CONS_GEN_SEED)
            if resume_ck.exists():
                net.load_state_dict(state["model"])
                net.to(dev)
                net.train()
                opt.load_state_dict(state["opt"])
                gen.set_state(state["gen_state"])
                step = state["step"]
            evl = copy.deepcopy(net0).to(CPU)
        devices_row = {"chunk": n_chunks, "device": str(dev),
                       "t_start": round(time.time() - T0, 1)}
        chunk_capped = False
        for step in range(step + 1, n_steps + 1):
            for g in opt.param_groups:
                g["lr"] = lr_base                    # CONST (the cons form)
            ix = torch.randint(n_pool, (16,), generator=gen)
            aj = torch.randint(n_anc, (8,), generator=gen)
            rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (8,),
                               generator=gen)
            nw = pool_a_x[ix].to(dev)
            anc = torch.cat([cons_anchor[aj],
                             torch.stack([train_ids[s: s + G1.BLOCK]
                                          for s in rj])], 0).to(dev)
            x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
            y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
            m = torch.zeros(32, x.shape[1], dtype=torch.bool, device=dev)
            if role == "tonly":
                m[:16] = pool_a_mask[ix].to(dev)
            else:                                     # conly: the complement
                m[:] = True
                m[:16] = ~pool_a_mask[ix].to(dev)
            logits, _ = net(x)
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                  y.reshape(-1), reduction="none"
                                  ).view(x.shape[0], x.shape[1])
            nm = nll[m]
            loss = nm.sum() / max(nm.numel(), 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if kind != "ADAMW":   # the uniform decay velocity, const form
                with torch.no_grad():
                    fac = 1.0 - DECAY_PER_STEP_VELOCITY
                    for p in net.parameters():
                        p.mul_(fac)
            if step in FATE_STEPS or step == n_steps or SMOKE:
                sd_cpu = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
                evl.load_state_dict(sd_cpu)
                evl.eval()
                bz = G1.battery_cell(evl, g0_ids, zid)
                ce_r = G1.ce_fixed_cpu(evl, *r_eval_xy)
                state["traj"].append({"step": step, "g0_pz": bz["mean_pz"],
                                      "g0_argmax": bz["frac_argmax_z"],
                                      "ce_r": ce_r,
                                      "ce_batch": float(loss.item()),
                                      "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] s{step:4d} g0 {bz['mean_pz']:.4f} CE_R "
                    f"{ce_r:.4f} CE {float(loss.item()):.4f}")
            n_burst += 1
            ok_t, temp = E261.burst_temp_check(f"{tag}-c{n_chunks}")
            chunk_temps.append(temp)
            if not ok_t or (time.time() - t_burst) > E261.BURST_MAX_S:
                log(f"  [{tag}] chunk {n_chunks} ended "
                    f"({time.time() - t_burst:.1f}s) at s{step}")
                chunk_capped = True
                break
        chunk_table.append({**devices_row,
                            "seconds": round(time.time() - t_burst, 1),
                            "steps_done": step, "capped": chunk_capped,
                            "max_temp_c": max(chunk_temps, default=None)})
        torch.save({"model": {k: v.detach().cpu()
                              for k, v in net.state_dict().items()},
                    "opt": opt.state_dict(), "gen_state": gen.get_state(),
                    "step": step, "traj": state["traj"],
                    "n_chunks": n_chunks, "chunk_table": chunk_table,
                    "gen_final_md5": (gen_state_md5(gen)
                                      if step >= n_steps else None),
                    "opt_kind": kind, "lr_base": lr_base, "role": role},
                   resume_ck)
        if step >= n_steps:
            break
        if n_chunks >= 12:
            raise RuntimeError(f"{tag}: cap-loop guard at chunk {n_chunks}")
        E261.burst_cooldown(tag)
        t_burst = None
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    return {"sd": sd_cpu, "traj": state["traj"], "steps_ran": step,
            "chunk_table": chunk_table, "gen_final_md5": gen_state_md5(gen)}


# ======================================================================
# MAIN
# ======================================================================
def main():
    global metrics
    dev = torch.device("cuda")
    metrics.update({
        "experiment": "e347_optimizer_cross",
        "phase": "THE OPTIMIZER CROSS — the substrate program's first GPU "
                 "cell: the canon formation, the Law-2b kill bracket, and "
                 "the tonly/conly maintenance split crossed against {SGD, "
                 "MSGD, RMSPROP, SIGN} at per-optimizer canonical lr (the "
                 "dose-normalized 5-point ladders), vs the committed AdamW "
                 "record — TRANSPORT-INVARIANT vs OPTIMIZER-SPECIFIC vs "
                 "MIXED",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "envelope": {
            "device": "cuda fp32 training (this cell owns the GPU lane; "
                      "every segment SEQUENTIAL with cooldowns — never "
                      "concurrent) + CPU fp64 dense projections (pocketfft), "
                      "CPU threads 4 (the x66 CPU oracle live — shared, "
                      "never touched)",
            "bursts": f"<= {E261.BURST_MAX_S:.0f}s (dispatch 180), per-step "
                      f"thermal polls at a {E261.TEMP_EARLY_END:.0f}C margin, "
                      f"cooldown {E261.COOLDOWN_S:.0f}s (dispatch 30-60), the "
                      f"{E261.TEMP_HARD:.0f}C never-past line (dispatch 85) "
                      "recorded to runs/_envelope_log.jsonl tagged e347:"
                      "<arm>; gpu_ok() before every burst",
            "trainings": "P1: 4 optimizers x 5-point lr ladders + the ADAMW "
                         "anchor (400-step installs); P3: 4 optimizers x 2 "
                         "boundary budgets (400-step budgeted transport "
                         "rungs); P4: 5 optimizers (incl. the in-session "
                         "AdamW reference) x {tonly, conly} (300-step "
                         "maintenance arms)",
        },
        "optimizer_set": {
            "SGD": "torch.optim.SGD, lr only — the pure geometry",
            "MSGD": f"torch.optim.SGD momentum={MSGD_MOMENTUM}, dampening 0",
            "RMSPROP": (f"torch.optim.RMSprop alpha={RMSPROP_ALPHA} "
                        f"eps={RMSPROP_EPS} momentum=0 centered=False — "
                        "adaptive, NO moment terms"),
            "SIGN": "THIS CELL'S OWN minimal custom optimizer (stated per "
                    "the dispatch): update = p -= lr*sign(grad); NO state, "
                    "NO moments, NO adaptivity — pure direction, "
                    "magnitude-normalized",
            "ADAMW": "the committed canon class — the in-session ANCHOR "
                     "arm + the P4 reference pair, never a law-carrier",
            "decay_channel": "crossed arms carry the committed per-step "
                             "decoupled decay velocity (1 - 1e-4 x "
                             "cosine_t) post-step (built-ins wd=0); ADAMW "
                             "uses the built-in wd 0.1 (bit-faithful)",
        },
        "ladders": {k: list(v) for k, v in LADDERS.items()},
        "canonical_lr_rule": "the ladder run whose post-install g0 lands in "
                             "the canon read band [0.15, 0.45]; multiple "
                             "in-band -> the g0 closest to 0.26464763283729"
                             "553; none in-band -> min |log(g0/0.2646...)| "
                             "with the miss disclosed",
        "deviations": deviations,
        "builds_on": [
            "THE SUBSTRATE PROGRAM's opening (owner directive 2026-10-12 "
            "~02:30Z): the laws become hypotheses across the training space",
            "x32 / 7bcd060 (THE NET0 CONFOUND: e264/e272's canon formed from "
            "THE BASE, not the root; the s1 'die' = the prior staying flat "
            "'AT STEP ONE NOTHING EVER HAPPENS'; e328's GPU control) — this "
            "cell's substrate statement + s1 re-wording",
            "T239/e261 + e264 (THE INSTALL RIG: chunked_install + the K10K "
            "room + the canon's own committed record — this cell's formation "
            "arm, ported verbatim with the optimizer swapped)",
            "T269/e290 (THE COUPLING-CONSTANT LADDER: Law 2b's kill bracket "
            "0.0008x-holds / 0.004x-dies — this cell's rung form + the "
            "committed reference rows)",
            "T291/e324 (THE INSTALL-DRAW CENSUS: the texture axes + the "
            "s1-fork classes + the canon-not-rerun convention)",
            "R83/x57 (THE KEY'S CHANNEL: the tonly/conly role split — "
            "TARGET-CARRIES, tonly protects / conly executes — this cell's "
            "maintenance probe, mask scheme verbatim)",
        ],
        "whats_new": [
            "THE FIRST OPTIMIZER CROSS (the record's first): SGD / MSGD / "
            "RMSProp / SIGN-DESCENT against the canon formation, the Law-2b "
            "bracket, and the maintenance role split — every prior law was "
            "AdamW-only (the bracket: SGD-M-only)",
            "THE DOSE NORMALIZATION (the record's first): per-optimizer "
            "canonical lr defined by landing the canon read band — the "
            "cross runs at matched dose, never matched nominal lr",
            "THE SIGN-DESCENT EXTREME (the record's first): pure-direction "
            "updates crossed against a room-captured write — the "
            "sign-vs-gradient step geometry on ONE fixed stream",
            "THE BRACKET'S OPTIMIZER AXIS (the record's first): the "
            "transport-kills re-run in canonical-lr terms with the drift "
            "realization disclosed per optimizer",
        ],
        "gates": {},
        "canonical_lr": {},
        "p1_ladder": {},
        "p2_texture": {},
        "p3_bracket": {},
        "p4_maintenance": {},
    })
    log(f"E347 — THE OPTIMIZER CROSS (smoke={SMOKE}) -> {RD}")
    log("P1 ladders: " + "; ".join(f"{k} {v}" for k, v in LADDERS.items())
        + f"; ADAMW anchor @ {E43.LR}; gen {INST_GEN} (the canon's own "
        f"draw); room {ROOM_MODE} k={ROOM_K} seeds {HELD_ROOM_SEEDS}")
    write_partial("startup (bars + parity + P-e347a registered, committed "
                  "at birth)")
    set_seed(347)                    # global init only; every RNG is its own

    # ================= P0: the protocol rebuild (g1c's gates VERBATIM) ==
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    vocab = corpus.vocab_size
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    G_VOCAB = {"vocab_size": vocab, "expected": VOCAB_EXPECT,
               "pass": bool(vocab == VOCAB_EXPECT)}
    assert G_VOCAB["pass"], f"vocab drift: {vocab} != {VOCAB_EXPECT}"
    corpus_zeph = train_text.count("ZEPH")
    G_NAMEFREE = {"corpus_zeph_count": corpus_zeph,
                  "pass": bool(corpus_zeph == 0)}
    assert G_NAMEFREE["pass"], f"corpus contains ZEPH x{corpus_zeph}"

    host_occ = []
    for host in G1.HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + G1.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"
    name_ids = corpus.encode(G1.NAME)

    bat_ids: dict[int, torch.Tensor] = {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    G_BATTERY = {"shapes": {f"g{j:+d}": list(bat_ids[j].shape)
                            for j in G1.GEOS},
                 "expected": {"g-12": [60, G1.PRE - 12], "g0": [60, G1.PRE],
                              "g+12": [60, G1.PRE + 12]},
                 "pass": bool(list(bat_ids[-12].shape) == [60, G1.PRE - 12]
                              and list(bat_ids[0].shape) == [60, G1.PRE]
                              and list(bat_ids[12].shape)
                              == [60, G1.PRE + 12])}
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    gm12_ids, g0_ids = bat_ids[-12], bat_ids[0]

    anchor_full = torch.stack([train_ids[p - G1.PRE: p - G1.PRE + G1.BLOCK]
                               for p, _ in install_occ])
    inst_mask = torch.zeros(60, G1.BLOCK - 1, dtype=torch.bool)
    inst_mask[:, G1.PRE - 1: G1.PRE - 1 + len(G1.NAME)] = True
    G_INSTMASK = {"name_positions": int(inst_mask.sum()),
                  "expected": 60 * len(G1.NAME),
                  "pass": bool(int(inst_mask.sum()) == 60 * len(G1.NAME))}
    assert G_INSTMASK["pass"], f"install mask gate FAILED: {G_INSTMASK}"

    def build_win(p, host):
        return torch.cat([train_ids[p - G1.PRE: p], name_ids,
                          train_ids[p + len(host):
                                    p + len(host) + G1.POST_CAP]])
    win_bank = torch.stack([build_win(p, h) for p, h in install_occ])
    win_masked_ok = all(
        "".join(itos[int(i)] for i in
                win_bank[i][G1.PRE: G1.PRE + len(G1.NAME)]) == G1.NAME
        for i in range(win_bank.shape[0]))
    G_NAMEWIN = {"n_windows": int(win_bank.shape[0]),
                 "masked_decode_all_name": bool(win_masked_ok),
                 "pass": bool(win_masked_ok
                              and list(win_bank.shape) == [60, G1.BLOCK])}
    assert G_NAMEWIN["pass"], f"name-window bind failed: {G_NAMEWIN}"

    arng = random.Random(G1.E170_ANCHOR_SEED)
    n_starts, tries = [], 0
    hi_start = len(train_ids) - G1.BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        txt = train_text[s: s + G1.BLOCK + 1]
        if any(f in txt for f in G1.ANCHOR_FORBIDDEN):
            tries += 1
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    G_ANCHOR = {"neutral_bank": {"seed": G1.E170_ANCHOR_SEED,
                                 "n_windows": 16, "starts": n_starts},
                "pass": bool(len(n_starts) == 16)}
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"

    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60, G1.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    metrics["gates"].update({"G_VOCAB": G_VOCAB, "G_NAMEFREE": G_NAMEFREE,
                             "G_SPLICE": G_SPLICE, "G_BATTERY": G_BATTERY,
                             "G_ANCHOR": G_ANCHOR, "G_INSTMASK": G_INSTMASK,
                             "G_NAMEWIN": G_NAMEWIN})
    log("P0: protocol gates PASS (vocab 65 / namefree / splice 19+41 / "
        "battery shapes / namewin / e170 bank / install mask)")
    write_partial("P0 protocol gates PASSED")

    # ---- G_QUOTES: the ported step bodies are VERBATIM (Rule 12) -------
    def _quote_lines(path: Path) -> set[str]:
        return {ln.strip() for ln in
                path.read_text(encoding="utf-8").splitlines()}

    own_lines = _quote_lines(Path(__file__).resolve())
    e261_lines = _quote_lines(E43.REPO / "lab" / "e261_rank_ladder.py")
    e290_lines = _quote_lines(E43.REPO / "lab" / "e290_coupling_ladder.py")
    quotes_e261 = [
        "ix = torch.randint(n_inst, (name_bs,), generator=gen)",
        "rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (mix_random,),",
        "m[:name_bs] = inst_mask[ix].to(dev)",
        "y.reshape(-1), reduction=\"none\"",
        "loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())",
        "torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)",
        "opt.step()          # BOTH moments + the update see the hook's g'",
        "f = cosine_lr(step - 1, INST_TOTAL)          # house schedule",
    ]
    quotes_e290 = [
        "gperp = g - gp_in                       # the orthogonal complement",
        "gp32 = torch.from_numpy(gperp.astype(np.float32))",
        "remaining = max(budget_norm - S, 0.0)",
        "cap = share / max(b_norm, 1e-12)",
        "lr_t = min(lr_sched, cap)",
        "realized = float(d_step.norm().item())",
        "aj_c = torch.randint(n_anc, (corp_bs - mix_random,),",
    ]
    miss = ([("e261", q) for q in quotes_e261
             if q not in e261_lines or q not in own_lines]
            + [("e290", q) for q in quotes_e290
               if q not in e290_lines or q not in own_lines])
    G_QUOTES = {
        "form": "the ported step bodies are VERBATIM: every anchor line "
                "appears (stripped-line exact) BOTH in this file's port AND "
                "in its committed source (e261's install body; e290's rung "
                "body). The ONE adapted line (proj.step_hook's mode arg -> "
                "ROOM_MODE, this cell's fixed room) is a named deviation, "
                "not quoted.",
        "n_quotes_e261": len(quotes_e261), "n_quotes_e290": len(quotes_e290),
        "missing": [f"{src}: {q}" for src, q in miss],
        "pass": bool(not miss),
    }
    metrics["gates"]["G_QUOTES"] = G_QUOTES
    assert G_QUOTES["pass"], f"G_QUOTES FAILED: {G_QUOTES}"
    log(f"P0b G_QUOTES: {len(quotes_e261)}+{len(quotes_e290)} anchor lines "
        f"verbatim in this port AND in e261/e290's sources: PASS")

    # ---- the jittered pool + cons anchor (P4's stream, e113's form) ----
    jit_x, jit_mask = {}, {}
    for j in G1.JITTERS:
        jwins = []
        for p, h in install_occ:
            pre = train_ids[p - G1.PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + G1.POST_CAP - j]
            w = torch.cat([pre, name_ids, post])
            if len(w) != G1.BLOCK:
                raise RuntimeError(f"jit window len {len(w)} at j={j}")
            jwins.append(w)
        jit_x[j] = torch.stack(jwins)
        m = torch.zeros(len(jwins), G1.BLOCK - 1, dtype=torch.bool)
        m[:, G1.PRE - 1 + j: G1.PRE - 1 + j + len(G1.NAME)] = True
        jit_mask[j] = m
    pool_a_x = torch.cat([jit_x[j] for j in G1.JITTERS])      # (300, 256)
    pool_a_mask = torch.cat([jit_mask[j] for j in G1.JITTERS])
    cons_anchor = anchor_full[:16]      # e113: first-16-install original bank

    # ---- G_MASKUNION + G_ROLEMASK (x57's counts, frozen) ---------------
    _g = torch.Generator().manual_seed(0)
    _ix = torch.randint(pool_a_x.shape[0], (16,), generator=_g)
    tonly_m = torch.zeros(32, G1.BLOCK - 1, dtype=torch.bool)
    tonly_m[:16] = pool_a_mask[_ix]
    conly_m = torch.ones(32, G1.BLOCK - 1, dtype=torch.bool)
    conly_m[:16] = ~pool_a_mask[_ix]
    nameless_m = torch.zeros(32, G1.BLOCK - 1, dtype=torch.bool)
    nameless_m[16:] = True
    canon_union = torch.zeros(32, G1.BLOCK - 1, dtype=torch.bool)
    canon_union[:16] = pool_a_mask[_ix]     # the name emissions
    canon_union[16:] = True                 # + the anchor rows' full windows
    counts = {"tonly": int(tonly_m.sum()), "conly": int(conly_m.sum()),
              "canon_union": int(canon_union.sum())}
    G_MASKUNION = {
        "form": "x57's bitwise decomposition on THIS cell's cons batch: the "
                "canon union mask == tonly | (conly & nameless rows); the "
                "counts constant 112 / 8048 / 4192 over every draw",
        "counts": counts,
        "union_identity": bool(torch.equal(canon_union,
                                           tonly_m | (conly_m & nameless_m))),
        "disjoint_name_rows": bool(torch.equal(
            tonly_m[:16] & conly_m[:16],
            torch.zeros(16, G1.BLOCK - 1, dtype=torch.bool))),
        "pass": bool(counts == {"tonly": 112, "conly": 8048,
                                "canon_union": 4192}),
    }
    G_ROLEMASK = {
        "form": "tonly := the 16 name rows' name-EMISSION positions only "
                "(16x7=112 tokens/step); conly := everything else (16x248 "
                "name-row non-name + 16x255 nameless = 8048/step); the name "
                "present as INPUT in both arms (the stimulus held); only "
                "the supervision role varies",
        "pass": G_MASKUNION["union_identity"] and G_MASKUNION["disjoint_name_rows"],
    }
    metrics["gates"]["G_MASKUNION"] = G_MASKUNION
    metrics["gates"]["G_ROLEMASK"] = G_ROLEMASK
    assert G_MASKUNION["pass"] and G_ROLEMASK["pass"], \
        f"mask gates FAILED: {G_MASKUNION} {G_ROLEMASK}"
    log(f"P0 G_MASKUNION/G_ROLEMASK: counts {counts} — the arms decompose "
        f"the ruler bitwise: PASS")

    # ================= P0b: the parents hard-bound (Rule 12) ============
    e264m = json.loads(E264_METRICS.read_text(encoding="utf-8"))
    e290m = json.loads(E290_METRICS.read_text(encoding="utf-8"))

    def _pbind(path, bound, extra=None):
        return {"path": str(path), "md5": md5of_norm(path),
                "bound_md5": bound, **(extra or {})}

    # the committed e290 reference rows, read at runtime + asserted
    e290_reads = {}
    for frac in BUDGET_FRACS:
        key = f"BUDGET-{frac}X" if frac != 0.0008 else "BUDGET-0.0008X"
        row = e290m["adjudication"]["reads"]["rung_ratios_t400"][key]
        e290_reads[frac] = row
    G_E290REF = {
        "form": "e290's committed SGD-M rows (the bracket's reference): "
                "read at runtime + asserted vs the birth literals "
                "(0.7673992671930195 holds / 0.31939051143675057 dies)",
        "runtime_reads": {str(f): e290_reads[f] for f in BUDGET_FRACS},
        "birth_literals": {str(f): E290_REF[f]["ratio_t400"]
                           for f in BUDGET_FRACS},
        "verdict": e290m["adjudication"]["verdict"],
        "pass": bool(e290m["adjudication"]["verdict"] == "TIGHT-CONSTANT"
                     and all(abs(e290_reads[f] - E290_REF[f]["ratio_t400"])
                             < 1e-15 for f in BUDGET_FRACS)),
    }
    assert G_E290REF["pass"], f"e290 reference gate FAILED: {G_E290REF}"

    canon_row = {
        "role": "THE CANON — the committed AdamW formation (room 26113/"
                "26114, gen 24314); NOT rerun; the texture's reference row "
                "READ from the committed artifacts at runtime",
        "post_g0": e264m["arms"]["K10K"]["install"]["post_cells"]["g0"],
        "post_gm12": e264m["arms"]["K10K"]["install"]["post_cells"]["gm12"],
        "write_norm": e290m["gates"]["G_FACTLOAD"]["write_norm"],
        "in_own_room":
            e264m["arms"]["K10K"]["install"]["displacement_loads"]["in_own_room"],
        "s1_g0": next(t for t in e264m["arms"]["K10K"]["install"]["traj"]
                      if t["step"] == 1)["g0_pz"],
    }
    G_CANONREAD = {
        "form": "the canon's texture row READ from the artifacts (never "
                "rerun, never retyped) + asserted vs the birth literals "
                "(Rule 12)",
        "runtime_reads": canon_row,
        "birth_literals": {"post_g0": CANON_POST_G0, "post_gm12":
                           CANON_POST_GM12, "write_norm": CANON_WRITE_NORM,
                           "in_own_room": CANON_IN_OWN_ROOM,
                           "s1_g0": CANON_S1_G0},
        "pass": bool(
            abs(canon_row["post_g0"] - CANON_POST_G0) < 1e-15
            and abs(canon_row["post_gm12"] - CANON_POST_GM12) < 1e-15
            and abs(canon_row["write_norm"] - CANON_WRITE_NORM) < 1e-12
            and abs(canon_row["in_own_room"] - CANON_IN_OWN_ROOM) < 1e-15
            and abs(canon_row["s1_g0"] - CANON_S1_G0) < 1e-18),
    }
    assert G_CANONREAD["pass"], f"canon re-read gate FAILED: {G_CANONREAD}"
    metrics["gates"]["G_E290REF"] = G_E290REF
    metrics["gates"]["G_CANONREAD"] = G_CANONREAD
    log(f"P0b G_CANONREAD: the canon re-verified: g0 "
        f"{canon_row['post_g0']:.10f} norm {canon_row['write_norm']:.4f} "
        f"in-room {canon_row['in_own_room']:.4f} s1 "
        f"{canon_row['s1_g0']:.2e}; G_E290REF: "
        f"{ {f: round(e290_reads[f], 4) for f in BUDGET_FRACS} } — PASS")

    rig_binds = {}
    for rel, bound in RIG_MD5S.items():
        p = E43.REPO / rel
        rig_binds[rel] = {"md5": md5of_norm(p), "bound_md5": bound}
    G_PARENTS = {
        "metrics": {
            "e261": _pbind(E261_METRICS, E261_MD5,
                           {"note": "THE RIG (ported verbatim by this cell)"}),
            "e264": _pbind(E264_METRICS, E264_MD5,
                           {"verdict": e264m["adjudication"]["verdict"],
                            "note": "the canon's own record"}),
            "e272": _pbind(E272_METRICS, E272_MD5,
                           {"note": "the read-band precedent [0.15, 0.45]"}),
            "e290": _pbind(E290_METRICS, E290_MD5,
                           {"note": "the bracket's committed record"}),
            "e323": _pbind(E323_METRICS, E323_MD5,
                           {"note": "the s1-fork + texture-axis precedent"}),
            "x57": _pbind(X57_METRICS, X57_MD5,
                           {"verdict": "TARGET-CARRIES",
                            "note": "the role split's committed record "
                                    "(TAVIREN subject — context only here)"}),
        },
        "artifacts": {
            "fact": {"path": f"runs/checkpoints/{FACT_CK}",
                     "md5": md5of(CKPT_DIR / FACT_CK),
                     "bound_md5": FACT_CK_MD5,
                     "note": "THE committed K10K fact (e290's vehicle; "
                             "P3/P4's fixed organism)"},
            "rooms264": {"path": f"runs/checkpoints/{ROOMS264_CK}",
                         "md5": md5of(CKPT_DIR / ROOMS264_CK),
                         "bound_md5": ROOMS264_MD5,
                         "note": "THE held room's committed record"},
            "span": {"path": f"runs/checkpoints/{SPAN_CK}",
                     "md5": md5of(CKPT_DIR / SPAN_CK),
                     "bound_md5": SPAN_MD5, "rank": SPAN_RANK},
            "vmap": {"path": f"runs/checkpoints/{VMAP_CK}",
                     "md5": md5of(CKPT_DIR / VMAP_CK),
                     "bound_md5": VMAP_MD5},
            "root": {"path": f"runs/checkpoints/{ROOT_CK}",
                     "md5": md5of(CKPT_DIR / ROOT_CK),
                     "bound_md5": ROOT_CK_MD5},
            "base": {"path": f"runs/checkpoints/{BASE_CK}",
                     "md5": md5of(CKPT_DIR / BASE_CK),
                     "bound_md5": BASE_CK_MD5},
        },
        "rig_files": rig_binds,
        "pass": bool(
            md5of_norm(E261_METRICS) == E261_MD5
            and md5of_norm(E264_METRICS) == E264_MD5
            and md5of_norm(E272_METRICS) == E272_MD5
            and md5of_norm(E290_METRICS) == E290_MD5
            and md5of_norm(E323_METRICS) == E323_MD5
            and md5of_norm(X57_METRICS) == X57_MD5
            and md5of(CKPT_DIR / FACT_CK) == FACT_CK_MD5
            and md5of(CKPT_DIR / ROOMS264_CK) == ROOMS264_MD5
            and md5of(CKPT_DIR / SPAN_CK) == SPAN_MD5
            and md5of(CKPT_DIR / VMAP_CK) == VMAP_MD5
            and md5of(CKPT_DIR / ROOT_CK) == ROOT_CK_MD5
            and md5of(CKPT_DIR / BASE_CK) == BASE_CK_MD5
            and all(v["md5"] == v["bound_md5"] for v in rig_binds.values())),
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log("P0b G_PARENTS PASS — 6 metrics + 6 artifacts + 6 rig files bound")
    write_partial("P0b parents hard-bound + the canon + e290 refs verified")
    del e264m, e290m

    # ---- G-BASE: the 2.74M corpus base, loaded fixed + fact-free --------
    base_net = G1.load_g1(CKPT_DIR / BASE_CK)
    assert base_net.num_params() == GB.G1B_PARAMS, \
        f"base param count {base_net.num_params()} != {GB.G1B_PARAMS}"
    base_sd = {k: v.detach().clone() for k, v in base_net.state_dict().items()}
    base_gm12 = G1.battery_cell(base_net, gm12_ids, zid)["mean_pz"]
    base_ce_r = G1.ce_fixed_cpu(base_net, *r_eval_xy)
    G_BASE = {"checkpoint": f"runs/checkpoints/{BASE_CK}",
              "params": GB.G1B_PARAMS,
              "fact_free_gm12": base_gm12, "ce_r": base_ce_r,
              "fact_free": bool(base_gm12 <= 0.05),
              "pass": bool(base_gm12 <= 0.05)}
    assert G_BASE["pass"], f"G-BASE FAILED: {G_BASE}"
    del base_net
    metrics["gates"]["G_BASE"] = G_BASE
    log(f"G-BASE: {BASE_CK} ({GB.G1B_PARAMS} params), fact-free "
        f"(g-12 {base_gm12:.4f}, CE_R {base_ce_r:.4f}): PASS")
    write_partial("P0c G-BASE PASSED")

    # ================= P1a: THE NET0 (the canon's substrate) + ROOM ======
    log("=" * 78)
    base_net0 = G1.evl_load(base_sd)          # THE CANON'S FORMATION NET0
    n_par = base_net0.num_params()
    g0_ids_probe = g0_ids                     # the standing-prior battery
    base_standing_g0 = G1.battery_cell(base_net0, g0_ids_probe, zid)["mean_pz"]
    G_NET0 = {
        "form": "THE CANON'S SUBSTRATE IS THE BASE (x32's NET0 correction, "
                "smoke-verified here: base-started CE/s1 BIT-IDENTICAL on "
                "CPU to e264's committed ledger): the standing Z-prior at "
                "the g0 battery is the s1 DIE class's floor (prior-flat = "
                "no step-one event)",
        "checkpoint": f"runs/checkpoints/{BASE_CK}",
        "n_params": n_par, "expected_params": GB.G1B_PARAMS,
        "standing_g0_prior": base_standing_g0,
        "standing_g0_committed": BASE_STANDING_G0_PRIOR,
        "abs_diff": abs(base_standing_g0 - BASE_STANDING_G0_PRIOR),
        "pass": bool(n_par == GB.G1B_PARAMS
                     and abs(base_standing_g0 - BASE_STANDING_G0_PRIOR)
                     <= 1e-15),
    }
    assert G_NET0["pass"], f"net0 gate FAILED: {G_NET0}"
    metrics["gates"]["G_NET0"] = G_NET0
    log(f"P1a G_NET0: {BASE_CK} — {n_par} params; standing g0 prior "
        f"{base_standing_g0:.10e} vs committed "
        f"{BASE_STANDING_G0_PRIOR:.10e}: PASS")

    N = n_par
    base_flat = flat_params_cpu(base_net0)
    base_flat_np = base_flat.double().numpy().astype(np.float64)

    vmap_art = torch.load(CKPT_DIR / VMAP_CK, map_location="cpu",
                          weights_only=False)
    v_flat32 = vmap_art["model"]["v_flat_fp32"]
    v64_np = v_flat32.numpy().astype(np.float64)
    G_VMBIND = {
        "path": f"runs/checkpoints/{VMAP_CK}",
        "md5": md5of(CKPT_DIR / VMAP_CK),
        "meta_experiment": vmap_art.get("meta", {}).get("experiment"),
        "size": int(v_flat32.numel()), "expected_size": N,
        "pass": bool(vmap_art.get("meta", {}).get("experiment") == "e258"
                     and int(v_flat32.numel()) == N),
    }
    assert G_VMBIND["pass"], f"v-map bind failed: {G_VMBIND}"
    metrics["gates"]["G_VMBIND"] = G_VMBIND
    del vmap_art

    span_art = torch.load(CKPT_DIR / SPAN_CK, map_location="cpu",
                          weights_only=False)
    Vp = span_art["Vp"].contiguous()
    G_SPANBIND = {"md5": md5of(CKPT_DIR / SPAN_CK),
                  "rank": int(Vp.shape[0]), "N": int(Vp.shape[1]),
                  "meta_experiment": span_art.get("meta", {}).get("experiment"),
                  "pass": bool(md5of(CKPT_DIR / SPAN_CK) == SPAN_MD5
                               and int(Vp.shape[0]) == SPAN_RANK
                               and int(Vp.shape[1]) == N
                               and span_art.get("meta", {}).get("experiment")
                               == "e246")}
    assert G_SPANBIND["pass"], f"span bind failed: {G_SPANBIND}"
    metrics["gates"]["G_SPANBIND"] = G_SPANBIND

    params_ref = list(G1.evl_load(base_sd).parameters())
    rooms = E261.LadderRooms(N, LADDER, v64_np, Vp.numpy().astype(np.float64),
                             params_ref, dev)
    cert = rooms.certify()
    G_PROJ = {
        "form": f"the HELD rank-{ROOM_K} room certified (fp64 CPU, "
                f"{E261.CERT_PROBES} probes, seed {E261.CERT_SEED})",
        "reads": cert,
        "bars": {"roundtrip": 1e-8, "idempotency": 1e-8,
                 "kept2": "10-sigma (5*sqrt(2k)/N)"},
        "pass": bool(cert["pass"]),
    }
    assert G_PROJ["pass"], f"room certification FAILED: {G_PROJ}"
    metrics["gates"]["G_PROJ"] = G_PROJ
    for nm, r in cert["per_rung"].items():
        log(f"  room {nm}: k {r['k']} (seeds {r['seeds']}) idem "
            f"{r['idempotency_max']:.1e} kept2 {r['kept2_mean']:.6f} vs "
            f"{r['kept2_expect']:.6f} (bar {r['kept2_bar_10sig']:.1e})")

    rooms264 = torch.load(CKPT_DIR / ROOMS264_CK, map_location="cpu",
                          weights_only=False)

    def _to_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    D_mine = rooms.rooms[ROOM_MODE].D
    S_mine = rooms.rooms[ROOM_MODE].S
    D264 = _to_np(rooms264["model"]["K10K"]["D_int8"])
    S264 = _to_np(rooms264["model"]["K10K"]["S"])
    seeds_equal = (LADDER[0][1], LADDER[0][2]) == HELD_ROOM_SEEDS \
        == tuple(rooms264["model"]["K10K"]["seeds"])
    D_bit = bool(np.array_equal(D_mine.astype(np.int8), D264.astype(np.int8)))
    S_bit = bool(np.array_equal(S_mine.astype(np.int64),
                                S264.astype(np.int64)))
    k_match = bool(int(rooms264["model"]["K10K"]["k"]) == ROOM_K
                   and ROOM_K == 10_000)
    G_ROOMHELD = {
        "form": "the writing room is THE CANON'S OWN: rebuilt from the "
                f"registered seeds {HELD_ROOM_SEEDS} and BIT-BOUND against "
                "e264_rooms.pt's committed K10K record",
        "seeds_match_e264_record": bool(seeds_equal),
        "D_bit_equal": D_bit, "S_bit_equal": S_bit, "k_match": k_match,
        "pass": bool(seeds_equal and D_bit and S_bit and k_match),
    }
    if SMOKE:
        G_ROOMHELD["pass"] = True
        G_ROOMHELD["vacuous"] = (
            "SMOKE: the bit-bind is VACUOUS at smoke k=512 — LIVE and "
            "binding at the full run's k=10k")
    del rooms264
    assert G_ROOMHELD["pass"], f"held-room gate FAILED: {G_ROOMHELD}"
    metrics["gates"]["G_ROOMHELD"] = G_ROOMHELD
    log(f"P1a G_ROOMHELD: the room (seeds {HELD_ROOM_SEEDS}) is the canon's "
        f"own — D bit {D_bit}, S bit {S_bit}, k {ROOM_K}: "
        f"{'PASS' if not SMOKE else 'PASS (SMOKE-vacuous)'}")
    write_partial("P1a the held room built + certified + bit-bound")

    # ================= P1: THE LADDERS + THE ANCHOR ======================
    def run_install_arm(kind: str, lr: float, arm_tag: str) -> dict:
        ft = file_tag(arm_tag)
        resume_ck = CKPT_DIR / (
            f"smoke_e347_{ft}_resume.pt" if SMOKE
            else f"e347_{ft}_resume.pt")
        out = chunked_install_x(arm_tag, kind, lr, base_net0, rooms,
                                win_bank, inst_mask, anchor_full, train_ids,
                                g0_ids, gm12_ids, r_eval_xy, zid, resume_ck,
                                dev)
        net = G1.evl_load(out["sd"])
        flat = flat_params_cpu(net)
        flat_np = flat.double().numpy().astype(np.float64)
        post_g0 = G1.battery_cell(net, g0_ids, zid)["mean_pz"]
        post_gm12 = G1.battery_cell(net, gm12_ids, zid)["mean_pz"]
        post_gp12 = G1.battery_cell(net, bat_ids[12], zid)["mean_pz"]
        ce_r = G1.ce_fixed_cpu(net, *r_eval_xy)
        write_norm = float(np.linalg.norm(flat_np - base_flat_np))
        loads = rooms.displacement_loads(flat - base_flat, ROOM_MODE)
        s1_row = next((t for t in out["traj"] if t["step"] == 1), None)
        assert s1_row is not None, f"{arm_tag}: no s1 row in the traj"
        del net
        # read-determinism re-load
        net2 = G1.evl_load({k: v.clone() for k, v in out["sd"].items()})
        flat2 = flat_params_cpu(net2)
        g0_2 = G1.battery_cell(net2, g0_ids, zid)["mean_pz"]
        del net2
        md5_1 = hashlib.md5(flat.numpy().tobytes()).hexdigest()
        md5_2 = hashlib.md5(flat2.numpy().tobytes()).hexdigest()
        G_FL = {"flat_md5": md5_1, "flat_md5_reload": md5_2,
                "g0_reload": g0_2, "g0_abs_diff": abs(g0_2 - post_g0),
                "pass": bool(md5_1 == md5_2
                             and abs(g0_2 - post_g0) <= 2e-6)}
        assert G_FL["pass"], f"{arm_tag} read-determinism FAILED: {G_FL}"
        # protocol identity from the run's own resume artifact
        res_state = torch.load(resume_ck, map_location="cpu",
                               weights_only=False)
        G_PI = {
            "machinery": "e261's chunked_install step body PORTED (the "
                         "quote-checked port; the ONLY deltas: the optimizer "
                         "construction + the uniform decay velocity)",
            "opt_kind": res_state.get("opt_kind"),
            "opt_state_keys": sorted({k for pstate in
                                      res_state["opt"]["state"].values()
                                      for k in pstate})[:4],
            "lr_base": res_state.get("lr_base"), "inst_steps": INST_STEPS,
            "inst_total": INST_TOTAL, "inst_gen": INST_GEN,
            "corp_bs": E43.CORP_BS, "mix_random": E43.MIX_RANDOM,
            "name_bs": G1.NAME_BS, "steps_ran": int(out["steps_ran"]),
            "smoke_stamp": bool(SMOKE),
            "pass": bool(res_state.get("opt_kind") == kind
                         and res_state.get("lr_base") == lr
                         and int(out["steps_ran"]) == INST_STEPS
                         and E43.CORP_BS == 48 and E43.MIX_RANDOM == 32
                         and G1.NAME_BS == 16 and E261.INST_TOTAL == 1000),
        }
        assert G_PI["pass"], f"{arm_tag} G_PROTOIDENT FAILED: {G_PI}"
        ck = save_ckpt(f"e347_{ft}_final", out["sd"],
                       {"desc": f"e347 {arm_tag}: {kind}@{lr} install "
                                f"(gen {INST_GEN}, room {ROOM_MODE})",
                        "kind": kind, "lr": lr, "post_g0": post_g0,
                        "write_norm": write_norm, "s1_g0": s1_row["g0_pz"]})
        row = {"kind": kind, "lr": lr,
               "post_g0": post_g0, "post_gm12": post_gm12,
               "post_gp12": post_gp12, "ce_r": ce_r,
               "write_norm": write_norm,
               "in_own_room": loads["in_own_room"],
               "v_excess": loads["v_excess"],
               "cos_to_span": loads["cos_to_span"],
               "s1_g0": s1_row["g0_pz"], "s1_class": s1_class(s1_row["g0_pz"]),
               "flat_md5": md5_1,
               "gen_final_md5": out.get("gen_final_md5"),
               "checkpoint": ck, "traj": out["traj"],
               "chunk_table": out.get("chunk_table", []),
               "G_FACTLOAD": G_FL, "G_PROTOIDENT": G_PI}
        log(f"{arm_tag} ROW: {kind}@lr={lr:g} g0 {post_g0:.6f} gm12 "
            f"{post_gm12:.5f} norm {write_norm:.4f} in-room "
            f"{loads['in_own_room']:.4f} s1 {s1_row['g0_pz']:.4f} "
            f"({s1_class(s1_row['g0_pz'])})")
        return row

    # ---- THE ADAMW ANCHOR FIRST (the instrument-identity gate) ---------
    log("=" * 78)
    log("P1 THE ADAMW ANCHOR — the committed canon re-run on this session's "
        "silicon (AdamW@1e-3, the ported rig; G_ANCHORW)")
    E261.burst_cooldown("room -> anchor")
    anchor_row = run_install_arm("ADAMW", E43.LR, "ADAMW@anchor")
    metrics["p1_ladder"]["ADAMW@anchor"] = {k: v for k, v in anchor_row.items()
                                            if k != "chunk_table"}
    metrics["p1_ladder"]["ADAMW@anchor"]["chunk_table"] = \
        anchor_row["chunk_table"]
    G_ANCHORW = {
        "form": "the in-session AdamW@1e-3 canon reproduction (the "
                "instrument-identity gate): the ported rig must land the "
                "committed record within the family's disclosed "
                "session-texture tolerances (e261's anchor convention)",
        "reads": {k: anchor_row[k] for k in
                  ("post_g0", "write_norm", "in_own_room", "post_gm12",
                   "s1_g0")},
        "bars": {"g0_tol": ANCHORW_TOL["g0"], "norm_tol": ANCHORW_TOL["norm"],
                 "inroom_tol": ANCHORW_TOL["inroom"],
                 "gm12_tol": ANCHORW_TOL["gm12"],
                 "s1": "both in the DIE class (< 0.01)"},
        "pass": bool(
            abs(anchor_row["post_g0"] - CANON_POST_G0) <= ANCHORW_TOL["g0"]
            and abs(anchor_row["write_norm"] - CANON_WRITE_NORM)
            <= ANCHORW_TOL["norm"]
            and abs(anchor_row["in_own_room"] - CANON_IN_OWN_ROOM)
            <= ANCHORW_TOL["inroom"]
            and abs(anchor_row["post_gm12"] - CANON_POST_GM12)
            <= ANCHORW_TOL["gm12"]
            and anchor_row["s1_g0"] < S1_DIE_BAR),
    }
    if SMOKE:
        G_ANCHORW["pass"] = True
        G_ANCHORW["vacuous"] = ("SMOKE: 8-step installs cannot reproduce the "
                                "400-step committed record — LIVE and "
                                "binding at the full run")
    metrics["gates"]["G_ANCHORW"] = G_ANCHORW
    assert G_ANCHORW["pass"], f"G_ANCHORW FAILED: {G_ANCHORW}"
    log(f"P1 G_ANCHORW: AdamW@{E43.LR:g} g0 {anchor_row['post_g0']:.6f} "
        f"(vs {CANON_POST_G0:.6f}) norm {anchor_row['write_norm']:.4f} "
        f"(vs {CANON_WRITE_NORM:.4f}) in-room "
        f"{anchor_row['in_own_room']:.4f} (vs {CANON_IN_OWN_ROOM:.4f}) s1 "
        f"{anchor_row['s1_g0']:.2e}: "
        f"{'PASS' if not SMOKE else 'PASS (SMOKE-vacuous)'}")
    write_partial("P1 the AdamW anchor landed (G_ANCHORW)")

    # ---- THE LADDERS -----------------------------------------------------
    ladder_rows: dict[str, list[dict]] = {}
    prev_tag = "ADAMW@anchor"
    for kind in OPTIMIZERS:
        ladder_rows[kind] = []
        for lr in LADDERS[kind]:
            arm_tag = f"{kind}@{lr:g}"
            E261.burst_cooldown(f"{prev_tag} -> {arm_tag}")
            log("=" * 78)
            log(f"P1 LADDER {arm_tag} — the canon formation at the standard "
                f"dose, the optimizer the ONLY delta (gen {INST_GEN}, room "
                f"{ROOM_MODE})")
            row = run_install_arm(kind, lr, arm_tag)
            ladder_rows[kind].append(row)
            metrics["p1_ladder"][arm_tag] = {k: v for k, v in row.items()
                                             if k != "chunk_table"}
            metrics["p1_ladder"][arm_tag]["chunk_table"] = row["chunk_table"]
            write_partial(f"P1 {arm_tag} complete")
            prev_tag = arm_tag

    # ---- G_DRAWSHARED (P1) ----------------------------------------------
    gen_md5s = {tag: row["gen_final_md5"]
                for tag, row in metrics["p1_ladder"].items()}
    G_DS1 = {"form": "the Dmix generator state md5 at completion IDENTICAL "
                     "across every install arm (bit-shared stream; the "
                     "optimizer the only delta)",
             "md5s": gen_md5s,
             "pass": bool(len(set(gen_md5s.values())) == 1
                          and None not in gen_md5s.values())}
    metrics["gates"]["G_DRAWSHARED_P1"] = G_DS1
    assert G_DS1["pass"], f"G_DRAWSHARED_P1 FAILED: {G_DS1}"
    log(f"P1 G_DRAWSHARED: {len(gen_md5s)} arms share one draw record "
        f"({list(set(gen_md5s.values()))[0][:10]}...): PASS")

    # ---- G_CANONLADDER: the canonical lr per optimizer -------------------
    canon_lr: dict[str, dict] = {}
    for kind in OPTIMIZERS:
        rows = ladder_rows[kind]
        in_band = [r for r in rows
                   if CANON_READ_BAND[0] <= r["post_g0"] <= CANON_READ_BAND[1]]
        if in_band:
            pick = min(in_band, key=lambda r: abs(r["post_g0"] - CANON_POST_G0))
            mode = "IN-BAND"
        else:
            pick = min(rows, key=lambda r: abs(math.log(
                max(r["post_g0"], RATIO_DEN_FLOOR) / CANON_POST_G0)))
            mode = "FALLBACK-NONE-IN-BAND"
        canon_lr[kind] = {"lr": pick["lr"], "mode": mode,
                          "post_g0": pick["post_g0"],
                          "in_band_count": len(in_band),
                          "ladder_g0s": [r["post_g0"] for r in rows]}
        log(f"P1 CANONICAL LR {kind}: {pick['lr']:g} ({mode}; g0 "
            f"{pick['post_g0']:.6f}; ladder "
            f"{['%.4f' % r['post_g0'] for r in rows]})")
    G_CANONLADDER = {
        "form": "the DOSE NORMALIZATION applied per the frozen rule",
        "picks": canon_lr,
        "pass": bool(all(c["lr"] in LADDERS[k] for k, c in canon_lr.items())),
    }
    metrics["gates"]["G_CANONLADDER"] = G_CANONLADDER
    metrics["canonical_lr"] = canon_lr
    assert G_CANONLADDER["pass"], "G_CANONLADDER FAILED"
    write_partial("P1 COMPLETE (ladders + canonical lr)")

    # ================= P2: THE TEXTURE TABLE ==============================
    log("=" * 78)
    texture: dict[str, dict] = {}
    for kind in OPTIMIZERS:
        pick_lr = canon_lr[kind]["lr"]
        row = next(r for r in ladder_rows[kind] if r["lr"] == pick_lr)
        norm_hold = TEXTURE_NORM_BAND[0] <= row["write_norm"] \
            <= TEXTURE_NORM_BAND[1]
        inroom_hold = abs(row["in_own_room"] - CANON_IN_OWN_ROOM) \
            <= TEXTURE_INROOM_ABS
        s1_hold = row["s1_class"] == "DIE"      # the canon's own class
        read_hold = CANON_READ_BAND[0] <= row["post_g0"] <= CANON_READ_BAND[1]
        exits = []
        if not norm_hold:
            exits.append(f"norm {row['write_norm']:.3f} outside "
                         f"[{TEXTURE_NORM_BAND[0]:.2f}, "
                         f"{TEXTURE_NORM_BAND[1]:.2f}]")
        if not inroom_hold:
            exits.append(f"in-room {row['in_own_room']:.3f} vs canon "
                         f"{CANON_IN_OWN_ROOM:.3f} (|d| > "
                         f"{TEXTURE_INROOM_ABS})")
        if not s1_hold:
            exits.append(f"s1 class {row['s1_class']} (canon DIE; s1 "
                         f"{row['s1_g0']:.4f})")
        if not read_hold:
            exits.append(f"final g0 {row['post_g0']:.4f} outside the band "
                         f"(the fallback dose)")
        texture[kind] = {
            "canonical_lr": pick_lr, "post_g0": row["post_g0"],
            "post_gm12": row["post_gm12"], "write_norm": row["write_norm"],
            "in_own_room": row["in_own_room"], "s1_g0": row["s1_g0"],
            "s1_class": row["s1_class"],
            "traj": [(t["step"], t["g0_pz"]) for t in row["traj"]],
            "axes": {"norm_in_band": bool(norm_hold),
                     "inroom_in_band": bool(inroom_hold),
                     "s1_class_canon": bool(s1_hold),
                     "read_in_band": bool(read_hold)},
            "holds": bool(norm_hold and inroom_hold and s1_hold and read_hold),
            "exits": exits,
            "vs_committed_adamw": {
                "canon": {"g0": CANON_POST_G0, "norm": CANON_WRITE_NORM,
                          "in_room": CANON_IN_OWN_ROOM, "s1": CANON_S1_G0},
                "in_session_adamw_anchor": {
                    "g0": anchor_row["post_g0"],
                    "norm": anchor_row["write_norm"],
                    "in_room": anchor_row["in_own_room"],
                    "s1": anchor_row["s1_g0"]}},
        }
        log(f"P2 TEXTURE {kind}@{pick_lr:g}: g0 {row['post_g0']:.4f} norm "
            f"{row['write_norm']:.2f} in-room {row['in_own_room']:.3f} s1 "
            f"{row['s1_g0']:.2e}({row['s1_class']}) -> "
            f"{'HOLDS' if texture[kind]['holds'] else 'MOVES: ' + '; '.join(exits)}")
    metrics["p2_texture"] = texture
    write_partial("P2 the texture table computed")

    # ================= P1b: THE FACT (loaded bit-exact) ===================
    log("=" * 78)
    fact_art = torch.load(CKPT_DIR / FACT_CK, map_location="cpu",
                          weights_only=False)
    fact_sd = fact_art["model"]
    fact_net = G1.evl_load(fact_sd)
    fact_flat = flat_params_cpu(fact_net)
    fact_g0 = G1.battery_cell(fact_net, g0_ids, zid)["mean_pz"]
    fact_gm12 = G1.battery_cell(fact_net, gm12_ids, zid)["mean_pz"]
    fact_ce_r = G1.ce_fixed_cpu(fact_net, *r_eval_xy)
    fact_flat_np = fact_flat.double().numpy().astype(np.float64)
    fact_write_norm = float(np.linalg.norm(fact_flat_np - base_flat_np))
    fact_loads = rooms.displacement_loads(fact_flat - base_flat, ROOM_MODE)
    fact_flat_md5 = hashlib.md5(fact_flat.numpy().tobytes()).hexdigest()
    G_FACTLOAD = {
        "checkpoint": f"runs/checkpoints/{FACT_CK}",
        "md5": md5of(CKPT_DIR / FACT_CK), "bound_md5": FACT_CK_MD5,
        "artifact_step": int(fact_art["step"]),
        "flat_md5": fact_flat_md5,
        "flat_md5_bound": FACT_FLAT_MD5,
        "read_g0": fact_g0, "read_gm12": fact_gm12, "read_ce_r": fact_ce_r,
        "write_norm": fact_write_norm,
        "in_own_room": fact_loads["in_own_room"],
        "bars": {"g0": CANON_POST_G0, "gm12": CANON_POST_GM12,
                 "norm": CANON_WRITE_NORM, "in_room": CANON_IN_OWN_ROOM,
                 "tol": 2e-6},
        "pass": bool(md5of(CKPT_DIR / FACT_CK) == FACT_CK_MD5
                     and fact_flat_md5 == FACT_FLAT_MD5
                     and abs(fact_g0 - CANON_POST_G0) <= 2e-6
                     and abs(fact_gm12 - CANON_POST_GM12) <= 2e-6
                     and abs(fact_write_norm - CANON_WRITE_NORM) <= 1e-9
                     and (SMOKE or abs(fact_loads["in_own_room"]
                                       - CANON_IN_OWN_ROOM) <= 2e-6)),
    }
    if SMOKE:
        G_FACTLOAD["smoke_note"] = (
            "SMOKE: the in-own-room sub-check is VACUOUS at smoke k=512 "
            "(the committed record is k=10k) — LIVE and binding at the full "
            "run")
    assert G_FACTLOAD["pass"], f"G_FACTLOAD FAILED: {G_FACTLOAD}"
    metrics["gates"]["G_FACTLOAD"] = G_FACTLOAD
    log(f"P1b G_FACTLOAD: the committed fact loaded bit-exact — g0 "
        f"{fact_g0:.10f} norm {fact_write_norm:.6f} in-room "
        f"{fact_loads['in_own_room']:.4f}: PASS")
    write_partial("P1b the committed fact loaded bit-exact")
    # (fact_net KEPT LIVE: it is P3's and P4's fixed organism, net0)
    del fact_art

    # ================= P3: THE KILL BRACKET ===============================
    bracket: dict[str, dict] = {}
    prev_tag = "P2"
    for kind in OPTIMIZERS:
        lr_base = canon_lr[kind]["lr"]
        for frac in BUDGET_FRACS:
            budget_norm = frac * fact_write_norm
            arm_tag = f"rung_{kind}_{frac:g}x"
            E261.burst_cooldown(f"{prev_tag} -> {arm_tag}")
            log("=" * 78)
            log(f"P3 {arm_tag} — e290's rung form, the vehicle {kind} @ "
                f"canonical lr {lr_base:g} x cosine; budget {frac:g}x "
                f"{fact_write_norm:.4f} = {budget_norm:.6f}; corpus seed "
                f"{CORPUS_GEN_SEED} (bit-shared)")
            resume_ck = CKPT_DIR / (f"smoke_e347_{arm_tag}_resume.pt"
                                    if SMOKE else f"e347_{arm_tag}_resume.pt")
            out = chunked_rung_x(arm_tag, kind, lr_base, budget_norm,
                                 fact_net, rooms, anchor_full, train_ids,
                                 g0_ids, gm12_ids, r_eval_xy, zid, resume_ck,
                                 dev)
            net = G1.evl_load(out["sd"])
            flat = flat_params_cpu(net)
            flat_np = flat.double().numpy().astype(np.float64)
            drift = float(np.linalg.norm(flat_np - fact_flat_np))
            t400 = next((t for t in out["traj"] if t["step"] == RUNG_STEPS),
                        out["traj"][-1] if out["traj"] else None)
            ratio_t400 = (t400["g0_pz"] / CANON_POST_G0) if t400 else None
            del net
            lr_led = out.get("lr_ledger", {})
            n_led = max(len(lr_led), 1)
            G_ORTH = {"form": "max over ALL corpus steps of ||P_room "
                              "g_perp||/||g_perp|| (checked every step)",
                      "orth_max": out.get("orth_max"),
                      "bar": ORTH_BAR,
                      "pass": bool((out.get("orth_max") or 0.0) < ORTH_BAR)}
            G_BUD = {"form": "S_400 <= BUDGET + 1e-5 AND realized cumulative "
                             "displacement <= BUDGET + 1e-5 (the "
                             "triangle-guaranteed reservation)",
                     "budget_norm": budget_norm, "S_final": out.get("S"),
                     "realized_cum_norm": drift,
                     "cap_bound_frac": (out.get("n_capped", 0)
                                        / max(RUNG_STEPS, 1)),
                     "pass": bool(out.get("S", 1e9) <= budget_norm
                                  + BUDGET_SLACK
                                  and drift <= budget_norm + BUDGET_SLACK)}
            G_BUF = {"form": "the twin instance's state bitwise-isolated "
                             "through every step (the never-stepped fact "
                             "side)",
                     "checks": out.get("bufsep", {}).get(
                         "isolation_checks", 0),
                     "violations": out.get("bufsep", {}).get(
                         "isolation_violations", 0),
                     "pass": bool(out.get("bufsep", {}).get(
                         "isolation_violations", 1) == 0)}
            for nm, g in (("G_ORTH", G_ORTH), ("G_BUDGET", G_BUD),
                          ("G_BUFSEP", G_BUF)):
                assert g["pass"], f"{arm_tag} {nm} FAILED: {g}"
            ck = save_ckpt(f"e347_{arm_tag}_final", out["sd"],
                           {"desc": f"e347 {arm_tag}: {kind}@{lr_base:g} "
                                    f"budget {frac:g}x rung",
                            "kind": kind, "lr": lr_base, "frac": frac,
                            "S": out.get("S"), "drift": drift})
            bracket.setdefault(kind, {})[str(frac)] = {
                "lr_base": lr_base, "budget_frac": frac,
                "budget_norm": budget_norm, "S_final": out.get("S"),
                "realized_cum_norm": drift,
                "realized_cum_frac_of_write": drift / fact_write_norm,
                "cap_bound_frac": G_BUD["cap_bound_frac"],
                "n_capped": out.get("n_capped"),
                "traj": out["traj"], "ratio_t400": ratio_t400,
                "g0_t400": t400["g0_pz"] if t400 else None,
                "holds": bool(ratio_t400 is not None
                              and ratio_t400 >= SURVIVAL_HOLD_BAR),
                "lr_ledger_medians": {
                    "lr_sched": float(np.median([v["lr_sched"]
                                                 for v in lr_led.values()])),
                    "cap": float(np.median([v["cap"]
                                            for v in lr_led.values()])),
                    "lr_applied": float(np.median([v["lr_applied"]
                                                   for v in lr_led.values()])),
                    "b_norm": float(np.median([v["b_norm"]
                                               for v in lr_led.values()])),
                } if lr_led else None,
                "cgen_final_md5": out.get("cgen_final_md5"),
                "checkpoint": ck,
                "G_ORTH": G_ORTH, "G_BUDGET": G_BUD, "G_BUFSEP": G_BUF,
            }
            log(f"P3 {arm_tag}: ratio_t400 {ratio_t400:.4f} -> "
                f"{'HOLDS' if bracket[kind][str(frac)]['holds'] else 'DIES'}"
                f" | S {out.get('S'):.6f}/{budget_norm:.6f} realized "
                f"{drift:.6f} ({drift / fact_write_norm:.5f} of write) "
                f"cap-bound {G_BUD['cap_bound_frac']:.0%}")
            metrics["p3_bracket"] = bracket
            write_partial(f"P3 {arm_tag} complete")
            prev_tag = arm_tag

    G_DS3 = {"form": "the corpus generator state md5 identical across every "
                     "rung (the bit-shared corpus stream)",
             "md5s": {k: {f: r["cgen_final_md5"]
                          for f, r in v.items()} for k, v in bracket.items()},
             "pass": bool(len({r["cgen_final_md5"]
                               for v in bracket.values()
                               for r in v.values()}) == 1
                          and None not in {r["cgen_final_md5"]
                                           for v in bracket.values()
                                           for r in v.values()})}
    metrics["gates"]["G_DRAWSHARED_P3"] = G_DS3
    assert G_DS3["pass"], f"G_DRAWSHARED_P3 FAILED: {G_DS3}"
    log("P3 G_DRAWSHARED: every rung shares one corpus draw record: PASS")

    # ---- the bracket verdicts --------------------------------------------
    for kind in OPTIMIZERS:
        r_lo = bracket[kind][str(BUDGET_FRACS[0])]     # 0.0008x
        r_hi = bracket[kind][str(BUDGET_FRACS[1])]     # 0.004x
        holds = bool(r_lo["holds"] and not r_hi["holds"])
        moved = []
        if not r_lo["holds"]:
            moved.append(f"0.0008x FLIPPED TO DIE ({r_lo['ratio_t400']:.3f})")
        if r_hi["holds"]:
            moved.append(f"0.004x FLIPPED TO HOLD ({r_hi['ratio_t400']:.3f})")
        bracket[kind]["_verdict"] = {
            "holds": holds,
            "moved_rungs": moved,
            "vs_committed_sgd_m": {
                str(f): {"ratio": E290_REF[f]["ratio_t400"],
                         "holds": E290_REF[f]["holds"]}
                for f in BUDGET_FRACS},
        }
        log(f"P3 BRACKET {kind}: {'HOLDS' if holds else 'MOVES: ' + '; '.join(moved)}"
            f" (0.0008x {r_lo['ratio_t400']:.3f} / 0.004x "
            f"{r_hi['ratio_t400']:.3f} vs committed SGD-M "
            f"{E290_REF[0.0008]['ratio_t400']:.3f}/"
            f"{E290_REF[0.004]['ratio_t400']:.3f})")
    write_partial("P3 COMPLETE (the bracket)")

    # ================= P4: THE MAINTENANCE PROBE ==========================
    maint: dict[str, dict] = {}
    kinds4 = OPTIMIZERS + ("ADAMW",)
    prev_tag = "P3"
    for kind in kinds4:
        lr_base = (E43.LR if kind == "ADAMW" else canon_lr[kind]["lr"])
        for role in ("tonly", "conly"):
            arm_tag = f"cons_{kind}_{role}"
            E261.burst_cooldown(f"{prev_tag} -> {arm_tag}")
            log("=" * 78)
            log(f"P4 {arm_tag} — x57's {role} mask on the committed fact; "
                f"{kind} @ {'canonical' if kind != 'ADAMW' else 'committed'} "
                f"lr {lr_base:g} CONST; stream seed {CONS_GEN_SEED} "
                f"(bit-shared); NATURAL (no room hook)")
            resume_ck = CKPT_DIR / (f"smoke_e347_{arm_tag}_resume.pt"
                                    if SMOKE else f"e347_{arm_tag}_resume.pt")
            out = chunked_cons_x(arm_tag, kind, lr_base, role, fact_net,
                                 pool_a_x, pool_a_mask, cons_anchor,
                                 train_ids, g0_ids, r_eval_xy, zid, resume_ck,
                                 dev)
            net = G1.evl_load(out["sd"])
            flat = flat_params_cpu(net)
            flat_np = flat.double().numpy().astype(np.float64)
            write_norm = float(np.linalg.norm(flat_np - base_flat_np))
            del net
            fate = {str(t["step"]): t["g0_pz"] for t in out["traj"]}
            ck = save_ckpt(f"e347_{arm_tag}_final", out["sd"],
                           {"desc": f"e347 {arm_tag}: {kind}@{lr_base:g} "
                                    f"{role} maintenance",
                            "kind": kind, "lr": lr_base, "role": role})
            maint.setdefault(kind, {})[role] = {
                "lr": lr_base, "fates": fate,
                "survival": {k: (v / CANON_POST_G0 if v is not None else None)
                            for k, v in fate.items()},
                "write_norm": write_norm,
                "gen_final_md5": out.get("gen_final_md5"),
                "checkpoint": ck, "traj": out["traj"],
                "chunk_table": out.get("chunk_table", []),
            }
            s300 = fate.get(str(FATE_STEPS[-1]))
            surv300 = (s300 / CANON_POST_G0) if s300 is not None else None
            log(f"P4 {arm_tag}: fates "
                f"{ {k: round(v, 4) for k, v in fate.items()} } survival(s"
                f"{FATE_STEPS[-1]}) {surv300 if surv300 is None else round(surv300, 4)}")
            metrics["p4_maintenance"] = maint
            write_partial(f"P4 {arm_tag} complete")
            prev_tag = arm_tag

    G_DS4 = {"form": "the maintenance generator state md5 identical across "
                     "every arm (the bit-shared stream)",
             "md5s": {k: {r: row["gen_final_md5"]
                          for r, row in v.items()} for k, v in maint.items()},
             "pass": bool(len({row["gen_final_md5"]
                               for v in maint.values()
                               for row in v.values()}) == 1
                          and None not in {row["gen_final_md5"]
                                           for v in maint.values()
                                           for row in v.values()})}
    metrics["gates"]["G_DRAWSHARED_P4"] = G_DS4
    assert G_DS4["pass"], f"G_DRAWSHARED_P4 FAILED: {G_DS4}"
    log("P4 G_DRAWSHARED: every arm shares one maintenance draw record: PASS")

    # ---- the maintenance verdicts ----------------------------------------
    for kind in kinds4:
        ton = maint[kind]["tonly"]["survival"].get(str(FATE_STEPS[-1]))
        con = maint[kind]["conly"]["survival"].get(str(FATE_STEPS[-1]))
        attention_side = bool(ton is not None and con is not None
                              and ton >= SURVIVAL_HOLD_BAR
                              and con < SURVIVAL_HOLD_BAR)
        flips = []
        if ton is not None and ton < SURVIVAL_HOLD_BAR:
            flips.append(f"tonly FAILED TO PROTECT ({ton:.3f})")
        if con is not None and con >= SURVIVAL_HOLD_BAR:
            flips.append(f"conly PROTECTED ({con:.3f})")
        maint[kind]["_verdict"] = {"attention_side": attention_side,
                                   "tonly_s300": ton, "conly_s300": con,
                                   "flips": flips,
                                   "x57_context": {"tonly":
                                                   "0.99996..1.00000 "
                                                   "protects (s25..s300)",
                                                   "conly": "0.0035..0.0008 "
                                                   "executes (s25..s300)"}}
        log(f"P4 MAINTENANCE {kind}: "
            f"{'ATTENTION-SIDE (holds)' if attention_side else 'MOVES: ' + '; '.join(flips)}"
            f" (tonly {ton:.3f} / conly {con:.3f})")
    write_partial("P4 COMPLETE (the maintenance split)")

    # ================= P5: THE ADJUDICATION ===============================
    log("=" * 78)
    law_table = {}
    n_hold = 0
    for kind in OPTIMIZERS:
        t_hold = texture[kind]["holds"]
        b_hold = bracket[kind]["_verdict"]["holds"]
        m_hold = maint[kind]["_verdict"]["attention_side"]
        all3 = bool(t_hold and b_hold and m_hold)
        if all3:
            n_hold += 1
        moved = []
        if not t_hold:
            moved.append(f"texture: {'; '.join(texture[kind]['exits'])}")
        if not b_hold:
            moved.append(f"bracket: "
                         f"{'; '.join(bracket[kind]['_verdict']['moved_rungs'])}")
        if not m_hold:
            moved.append(f"maintenance: "
                         f"{'; '.join(maint[kind]['_verdict']['flips'])}")
        law_table[kind] = {"texture": "HOLDS" if t_hold else "MOVES",
                           "bracket": "HOLDS" if b_hold else "MOVES",
                           "maintenance": "HOLDS" if m_hold else "MOVES",
                           "all_three": all3, "moved": moved,
                           "canonical_lr": canon_lr[kind]["lr"],
                           "canonical_lr_mode": canon_lr[kind]["mode"]}
    if n_hold == len(OPTIMIZERS):
        verdict = "TRANSPORT-INVARIANT"
    elif n_hold == 0:
        verdict = "OPTIMIZER-SPECIFIC"
    else:
        verdict = "MIXED"

    # P-e347a scoring (clause by clause)
    sign_class = {"SGD": "magnitude", "MSGD": "magnitude",
                  "RMSPROP": "sign", "SIGN": "sign"}
    clause2 = all((texture[k]["s1_class"] == "DIE")
                  == (sign_class[k] == "sign") for k in OPTIMIZERS)
    clause3 = all(bracket[k]["_verdict"]["holds"] for k in OPTIMIZERS)
    clause4 = all(maint[k]["_verdict"]["attention_side"] for k in OPTIMIZERS)
    clause1 = all(texture[k]["holds"] for k in OPTIMIZERS)
    p_scored = bool(verdict == "MIXED" and clause2 and clause3 and clause4)
    p_e347a = {
        "prediction_verbatim": (
            "MIXED — exactly one law-face moves: the s1 fork / the "
            "trajectory texture; the bracket and the maintenance split "
            "hold. (1) bulk texture HOLDS all four; (2) s1 fork MOVES "
            "along step-geometry class lines (sign-class DIE / "
            "magnitude-class SURVIVE); (3) bracket HOLDS in budget terms "
            "all four; (4) maintenance ATTENTION-SIDE all four. NET "
            "SCORED: TRUE iff composite == MIXED AND clauses (2),(3),(4) "
            "as stated."),
        "clauses": {"1_bulk_texture": bool(clause1),
                    "2_s1_step_geometry": bool(clause2),
                    "3_bracket_holds": bool(clause3),
                    "4_maintenance_attention_side": bool(clause4)},
        "scored": p_scored,
        "falsifier_clause": "any hard-gate failure -> TEXTURE (P-e347a "
                            "UNSCORED per the family's precedent clause)",
    }

    metrics.update({
        "envelope_observed": _envelope_summary(),
        "p5_adjudication": {
            "verdict": verdict,
            "bars_verbatim": REGISTERED["bars_verbatim"],
            "law_table": law_table,
            "n_optimizers_holding_all_three": n_hold,
            "composite_rule": ("TEXTURE (any hard-gate failure) -> n_hold "
                               "== 4 TRANSPORT-INVARIANT / n_hold == 0 "
                               "OPTIMIZER-SPECIFIC / else MIXED"),
            "the_canonical_lr_table": {k: {"lr": canon_lr[k]["lr"],
                                           "mode": canon_lr[k]["mode"],
                                           "post_g0": canon_lr[k]["post_g0"],
                                           "ladder_g0s":
                                               canon_lr[k]["ladder_g0s"]}
                                       for k in OPTIMIZERS},
            "the_itemized_split": {k: law_table[k]["moved"]
                                   for k in OPTIMIZERS
                                   if law_table[k]["moved"]},
            "adamw_reference_rows": {
                "formation_anchor_G_ANCHORW": {
                    "g0": anchor_row["post_g0"],
                    "norm": anchor_row["write_norm"],
                    "in_room": anchor_row["in_own_room"],
                    "s1": anchor_row["s1_g0"]},
                "maintenance_in_session": {
                    r: maint["ADAMW"][r]["survival"]
                    for r in ("tonly", "conly")},
                "bracket_committed_sgd_m": {
                    str(f): E290_REF[f]["ratio_t400"]
                    for f in BUDGET_FRACS}},
        },
        "p_e347a": p_e347a,
        "status": "COMPLETE" if not SMOKE else "SMOKE-COMPLETE",
        "date_finished": common.now_iso(),
        "git_head_final": git_head(),
    })
    write_partial("P5 ADJUDICATED")

    # ================= THE PLOT ===========================================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    # (1) the lr ladders
    ax = axes[0][0]
    for kind in OPTIMIZERS:
        rows = ladder_rows[kind]
        ax.plot([r["lr"] for r in rows], [r["post_g0"] for r in rows],
                "o-", label=kind)
        pk = canon_lr[kind]
        ax.scatter([pk["lr"]], [pk["post_g0"]], s=140, facecolors="none",
                   edgecolors="k", zorder=5)
    ax.axhspan(CANON_READ_BAND[0], CANON_READ_BAND[1], alpha=0.15,
               color="gray", label="canon read band [0.15,0.45]")
    ax.axhline(CANON_POST_G0, color="k", ls=":", lw=1,
               label=f"committed canon {CANON_POST_G0:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("lr (log)"); ax.set_ylabel("post-install g0")
    ax.set_title("P1 — the dose-normalization ladders (o = ladder, O = canonical)")
    ax.legend(fontsize=7)
    # (2) the texture axes
    ax = axes[0][1]
    ax.scatter([CANON_WRITE_NORM], [CANON_IN_OWN_ROOM], marker="*",
               s=200, c="k", label="committed AdamW canon")
    ax.scatter([anchor_row["write_norm"]], [anchor_row["in_own_room"]],
               marker="s", s=90, c="tab:gray",
               label="AdamW anchor (in-session)")
    for kind in OPTIMIZERS:
        r = next(r for r in ladder_rows[kind]
                 if r["lr"] == canon_lr[kind]["lr"])
        ax.scatter([r["write_norm"]], [r["in_own_room"]], s=90,
                   label=f"{kind} @{r['lr']:g} s1:{r['s1_class']}")
    ax.axvspan(TEXTURE_NORM_BAND[0], TEXTURE_NORM_BAND[1], alpha=0.08,
               color="gray")
    ax.axhline(CANON_IN_OWN_ROOM - TEXTURE_INROOM_ABS, ls="--", c="gray",
               lw=0.8)
    ax.axhline(CANON_IN_OWN_ROOM + TEXTURE_INROOM_ABS, ls="--", c="gray",
               lw=0.8)
    ax.set_xlabel("write norm"); ax.set_ylabel("in-own-room fraction")
    ax.set_title("P2 — the formation texture (bands shaded)")
    ax.legend(fontsize=7)
    # (3) the bracket
    ax = axes[1][0]
    width = 0.35
    xs = np.arange(len(OPTIMIZERS))
    lo = [bracket[k][str(BUDGET_FRACS[0])]["ratio_t400"] for k in OPTIMIZERS]
    hi = [bracket[k][str(BUDGET_FRACS[1])]["ratio_t400"] for k in OPTIMIZERS]
    ax.bar(xs - width / 2, lo, width, label="0.0008x budget")
    ax.bar(xs + width / 2, hi, width, label="0.004x budget")
    ax.axhline(SURVIVAL_HOLD_BAR, c="r", ls="--", lw=1, label="0.5 holds bar")
    ax.axhline(E290_REF[0.0008]["ratio_t400"], c="k", ls=":", lw=1,
               label="committed SGD-M 0.0008x")
    ax.axhline(E290_REF[0.004]["ratio_t400"], c="k", ls="-.", lw=1,
               label="committed SGD-M 0.004x")
    ax.set_xticks(xs); ax.set_xticklabels(OPTIMIZERS)
    ax.set_ylabel("survival ratio t400")
    ax.set_title("P3 — Law 2b's kill bracket under each optimizer")
    ax.legend(fontsize=7)
    # (4) the maintenance split
    ax = axes[1][1]
    for kind in kinds4:
        ton = maint[kind]["tonly"]["survival"]
        con = maint[kind]["conly"]["survival"]
        st = sorted(int(k) for k in ton)
        sc = sorted(int(k) for k in con)
        ax.plot(st, [max(ton[str(s)], 1e-8) for s in st], "o-",
                label=f"{kind} tonly")
        ax.plot(sc, [max(con[str(s)], 1e-8) for s in sc], "x--",
                alpha=0.6, label=f"{kind} conly")
    ax.axhline(SURVIVAL_HOLD_BAR, c="r", ls="--", lw=1, label="0.5 bar")
    ax.set_xlabel("cons step"); ax.set_ylabel("survival (g0 / canon)")
    ax.set_yscale("log")
    ax.set_title("P4 — the tonly/conly role split (x57's probe)")
    ax.legend(fontsize=6, ncol=2)
    fig.suptitle(f"e347 THE OPTIMIZER CROSS — verdict: {verdict} "
                 f"(n_hold={n_hold}/4)", fontsize=13)
    fig.tight_layout()
    fig.savefig(RD / "e347_optimizer_cross.png", dpi=130)
    plt.close(fig)
    log(f"PLOT saved {RD / 'e347_optimizer_cross.png'}")

    # ================= THE REPORT =========================================
    rep = [f"# e347 — THE OPTIMIZER CROSS (REPORT)\n",
           f"date: {common.now_iso()} | git head: {git_head()} | "
           f"smoke: {SMOKE}\n",
           "## The verdict against the frozen bars\n",
           f"**{verdict}** (n_hold = {n_hold}/{len(OPTIMIZERS)} optimizers "
           "hold all three laws)\n"]
    rep.append("| optimizer | canonical lr | mode | texture | bracket | "
               "maintenance | moved |\n|---|---|---|---|---|---|---|")
    for kind in OPTIMIZERS:
        lt = law_table[kind]
        rep.append(f"| {kind} | {lt['canonical_lr']:g} | "
                   f"{lt['canonical_lr_mode']} | {lt['texture']} | "
                   f"{lt['bracket']} | {lt['maintenance']} | "
                   f"{'; '.join(lt['moved']) or '-'} |")
    rep.append("\n## The canonical-lr table (the dose normalization)\n")
    for kind in OPTIMIZERS:
        rep.append(f"- **{kind}**: lr {canon_lr[kind]['lr']:g} "
                   f"({canon_lr[kind]['mode']}); g0 "
                   f"{canon_lr[kind]['post_g0']:.4f}; ladder "
                   f"{['%.4f' % g for g in canon_lr[kind]['ladder_g0s']]}")
    rep.append("\n## The formation texture (P2)\n")
    for kind in OPTIMIZERS:
        t = texture[kind]
        rep.append(f"- **{kind}**: g0 {t['post_g0']:.4f} norm "
                   f"{t['write_norm']:.2f} in-room {t['in_own_room']:.3f} "
                   f"s1 {t['s1_g0']:.2e} ({t['s1_class']}) -> "
                   f"{'HOLDS' if t['holds'] else 'MOVES: ' + '; '.join(t['exits'])}")
    rep.append(f"- committed AdamW canon: g0 {CANON_POST_G0:.4f} norm "
               f"{CANON_WRITE_NORM:.2f} in-room {CANON_IN_OWN_ROOM:.3f} "
               f"s1 {CANON_S1_G0:.2e} (DIE); in-session AdamW anchor: g0 "
               f"{anchor_row['post_g0']:.4f} norm "
               f"{anchor_row['write_norm']:.2f} in-room "
               f"{anchor_row['in_own_room']:.3f} s1 "
               f"{anchor_row['s1_g0']:.2e}")
    rep.append("\n## The kill bracket (P3)\n")
    for kind in OPTIMIZERS:
        for frac in BUDGET_FRACS:
            r = bracket[kind][str(frac)]
            rep.append(f"- **{kind} @{frac:g}x**: ratio_t400 "
                       f"{r['ratio_t400']:.4f} "
                       f"({'HOLDS' if r['holds'] else 'DIES'}); realized "
                       f"drift {r['realized_cum_frac_of_write']:.5f} of "
                       f"write; cap-bound {r['cap_bound_frac']:.0%}; S "
                       f"{r['S_final']:.6f}/{r['budget_norm']:.6f}")
    rep.append(f"- committed SGD-M reference: 0.0008x "
               f"{E290_REF[0.0008]['ratio_t400']:.4f} (holds) / 0.004x "
               f"{E290_REF[0.004]['ratio_t400']:.4f} (dies) — read from "
               "e290's artifacts, never rerun")
    rep.append("\n## The maintenance probe (P4)\n")
    for kind in kinds4:
        v = maint[kind]["_verdict"]
        rep.append(f"- **{kind}**: tonly {v['tonly_s300']:.4f} / conly "
                   f"{v['conly_s300']:.4f} -> "
                   f"{'ATTENTION-SIDE (holds)' if v['attention_side'] else 'MOVES: ' + '; '.join(v['flips'])}")
    rep.append("\n## P-e347a\n")
    rep.append("```json\n" + json.dumps(p_e347a, indent=1) + "\n```")
    rep.append("\n## Gates\n")
    npass = sum(1 for g in metrics["gates"].values()
                if isinstance(g, dict) and g.get("pass"))
    per_arm = sum(1 for tag, row in metrics["p1_ladder"].items()
                  for gk in ("G_FACTLOAD", "G_PROTOIDENT")
                  if row.get(gk, {}).get("pass"))
    rep.append(f"- top-level: {npass}/{len(metrics['gates'])} PASS "
               f"({', '.join(sorted(metrics['gates']))})")
    rep.append(f"- per-arm G_FACTLOAD/G_PROTOIDENT: {per_arm} arm-gates "
               "PASS (2 per install arm); per-rung G_ORTH/G_BUDGET/G_BUFSEP "
               "all PASS (asserted inline)")
    rep.append("\n## The parity block\n```")
    rep.append("LAB LEAN: " + REGISTERED["parity_block_verbatim"]["LAB_LEAN"])
    rep.append("COUNTER: " + REGISTERED["parity_block_verbatim"]["COUNTER"]
               + "\n```")
    (RD / "REPORT.md").write_text("\n".join(rep), encoding="utf-8")
    log(f"REPORT saved {RD / 'REPORT.md'}")
    write_partial("COMPLETE (plot + report written)")
    log(f"E347 {'SMOKE ' if SMOKE else ''}DONE — verdict {verdict}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        _logf.flush()
        raise
