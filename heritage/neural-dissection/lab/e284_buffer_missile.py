"""E284 — THE SEPARATE-BUFFER MISSILE (the motel's ownership test, the
morning's last open mechanism question). This docstring carries the
registered question + bars VERBATIM from the dispatch letter, committed at
birth BEFORE any compute. Adjudicate against exactly this; no bar shopping.

THE CONTEXT (the dispatch letter): e278's guided missile (AdamW): an
exactly-orthogonal corpus gradient (4.6e-17) still walked 48-60% in-room —
Adam re-aims. e280's SGD-M missile: STILL funneled (median 0.4553) —
P-x283b's direction refuted; the disclosed mechanical reading: the SHARED
MOMENTUM BUFFER carries install-stream in-room content into the corpus
steps. e284 separates them: SGD-M with SEPARATE momentum buffers per
stream (the install's buffer and the corpus's buffer never mix). If
orthogonality now survives in the APPLIED displacement, the re-aimer was
buffer-sharing (the motel is a momentum-architecture); if it still funnels,
the landscape itself bends steps.

THE ARMS (verbatim; k=10k room, bit-bound; the family's standard protocol;
the concurrent 1:1 interleave):
  (a) SEPARATE-BUFFER-MISSILE: SGD-M (the same stable lr, momentum 0.9,
      wd 0) with TWO momentum buffers — the install steps update buffer I
      only, the corpus steps (the orthogonalized stream, g_perp) update
      buffer C only. No other change.
  (b) SHARED-BUFFER-MISSILE (the control twin, same session): identical
      but ONE shared buffer (this session's replication of e280's missile
      arm — the same-session pair is the discriminator; e280's committed
      numbers cited as the cross-check).
READS on both: the WRITE read (post g0 s400) + milestones; the DISPLACEMENT
LEDGER (the realized corpus displacement's in-room share per milestone —
THE primary read); the orthogonality ledger (machine-exact, the e278
convention); the corpus CE.

FROZEN BARS (verbatim; the primary = the in-room share of the realized
corpus displacement, median over milestones):
  - MOMENTUM-OWNED: "the separate-buffer arm's in-room share < 0.20 (vs
    the shared twin's ~0.45+) — the re-aimer is buffer-sharing; THE MORTel
    IS A MOMENTUM-SHARING ARCHITECTURE; orthogonality is preservable by
    buffer separation." [sic — 'MOTEL' in the dispatch's intent]
  - LANDSCAPE-OWNED: "the separate-buffer arm still >= 0.35 in-room — the
    landscape itself bends applied steps toward the room; the motel is the
    space's, not any optimizer state's."
  - MIXED: "0.20-0.35 or the twins disagree with their committed
    precedents — everything verbatim."
SECONDARY (never the primary; verbatim): "the write's survival in each arm
(does buffer separation spare the write? e273's two-body account predicts
NO — the parameters still collide)."

OPERATIONALIZATIONS (frozen HERE before compute; they fix the clauses,
they do not move the bars):
  * THE VEHICLE := e272's committed K10KR room (k=10,000, seeds
    27215/27216), rebuilt + bit-gated vs runs/checkpoints/e272_rooms.pt's
    stored K10KR D/S (exact equality; G_ROOM10KR); the e001 fact-free
    base; the Dmix install s400 gen 24314, draw order VERBATIM; the hook
    VERBATIM (backward -> clip 1.0 -> project onto the room (CPU fp64,
    write fp32) -> opt.step; norm NOT rescaled).
  * THE OPTIMIZER := SGD(momentum 0.9, wd 0.0) at LR_STABLE = 0.01 x
    21.7385748014537 = 0.21738574801453703 x cosine_lr(s-1, 1000) for
    BOTH streams (the paired lr) — e280's committed lr class EXACTLY
    (e273's SGD001X stable rider; provenance re-derived at runtime from
    the md5-bound runs/e273/lr_calibration.json; G_LR_BIND).
  * THE INTERLEAVE := e268/e278's registered 1:1 form ported VERBATIM:
    after EVERY install step s (bit-identical draws to the family's
    serial installs), ONE corpus step — 16 original-host anchors + 32
    random corpus windows from cgen seed 28401 (THIS cell's ONE fresh
    registered stream; BOTH arms draw the identical sequence, so the
    buffer topology is the arms' ONLY delta); full-window CE; backward ->
    clip 1.0 -> g_perp = g - P_room(g) (CPU fp64, write fp32, verified
    EVERY corpus step) -> the corpus opt step. 800 opt steps per arm.
  * THE BUFFER TOPOLOGY := arm SEP: TWO SGD instances over the SAME
    parameter set — opt_I stepped ONLY by install steps, opt_C ONLY by
    corpus steps, identical lr schedules, momentum 0.9, wd 0 each; arm
    SHA: ONE SGD instance stepped by both streams (e280's committed
    form).
  * THE PRIMARY READ := the median (the family's sorted[n//2]) over
    milestones {100, 200, 300, 400} of the corpus stream's
    realized-displacement INTERVAL in-room fraction (e280's missile-read
    convention: the s1 warm-up interval EXCLUDED); cumulative fractions
    co-reported. Interval = sum over the interval's corpus steps of
    (theta_after - theta_before), GPU fp32 accumulation, CPU fp64
    projection at the milestones (e278's disclosed form).
  * THE TWIN-PRECEDENT CLAUSE (the MIXED branch's
    "twins-disagree-with-precedents" form, frozen): the SHARED twin is the
    only arm with a committed precedent (e280's committed median
    0.45529691870685485; intervals 0.4152-0.4926); the twin "disagrees"
    iff its own primary median < 0.35 (e280's committed MOTEL bar — below
    it the twin failed to funnel this session and the pair's difference
    carries no ownership content) -> the named MIXED branch
    CONTROL-TWIN-OFF-PRECEDENT. The separate arm has NO committed
    precedent (it is the new instrument); its read is adjudicated on the
    letter bars only.
  * COMPOSITE ORDER := TEXTURE (any hard-gate failure — nothing
    adjudicated) -> MIXED (CONTROL-TWIN-OFF-PRECEDENT: shared twin's
    primary < 0.35) -> MOMENTUM-OWNED (separate primary < 0.20) ->
    LANDSCAPE-OWNED (separate primary >= 0.35) -> MIXED (BETWEEN-BARS:
    0.20 <= separate primary < 0.35).
  * THE SURVIVAL SECONDARY (never the primary): each arm's post g0 (the
    WRITE read, s400 + milestones) vs e280's committed cites — the serial
    SGD-M 10k rung post 0.21675553917884827 (the write's undisturbed
    class under this optimizer/lr) and e280's committed shared-missile
    post 0.0006920578307472169 (the missile-under-fire class); SPARED :=
    post >= 0.5x the cited serial; e273's two-body account predicts NOT
    SPARED for both arms (the parameters still collide). Reported
    verbatim, never a bar.
  * HARD GATES (a failure HALTS): {G_NAMEFREE, G_SPLICE, G_BATTERY,
    G_ANCHOR, G_INSTMASK, G_PARENTS, G_BASE, G_ROOT, G_VMBIND, G_SPANBIND,
    G_PROJ, G_ROOM10KR, G_LR_BIND, G_MISSILE_ORTH, G_BUFSEP}. G_MISSILE_ORTH
    is INSTANTIATED this time (e280's bookkeeping omission, disclosed at
    its fold, corrected here): max over ALL corpus steps, BOTH arms, of
    ||P_room g_perp||/||g_perp|| < 1e-6, written into the gates dict.
  * G_BUFSEP (this cell's new gate — the buffer separation's machine
    verification, the smoke's centerpiece made the full-run discipline):
    (i) ISOLATION, every step of the SEP arm: snapshot opt_I's momentum
    buffers immediately before every corpus opt_C.step() and compare
    bitwise (torch.equal) after; snapshot opt_C's before every install
    opt_I.step() and compare after — the install buffer never receives
    corpus content and vice versa, MACHINE-CHECKED (any mismatch HALTS);
    (ii) COMPOSITION, per milestone + final: ||P_room buf_C||/||buf_C||
    < 1e-4 (the corpus buffer carries no in-room content — SGD momentum
    is linear in its grads, every corpus grad is orthogonal to < 1e-6, so
    the buffer must sit at the fp32 accumulation floor ~1e-7-1e-6; the
    bar holds 100x headroom) AND ||P_room buf_I||/||buf_I|| > 0.99 (the
    install buffer IS the in-room stream — install grads are projected
    onto the room before stepping). The SHARED arm's single buffer
    in-room fraction is a READ (the designed mixture; the discrimination
    context), never gated.
  * NO CONS (T259/e281; e278/e280/e283's committed form): the landing
    read is a cons property; the frozen bars read the WRITE and the
    DISPLACEMENT only; both arms' install-final states are CHECKPOINTED
    for any later landing pass.
  * NO SERIAL ARM (budget + cited): e280's committed S10K serial rung
    (post 0.21675553917884827, bit-identical room/protocol/optimizer/lr)
    is the survival secondary's cited denominator, hard-bound in
    G_PARENTS; the primary read (a displacement ratio) needs no serial
    reference by construction.

REGISTERED PREDICTIONS:
  - P-e284a (REGISTERED HERE): under separate buffers the corpus
    displacement's in-room share collapses to the fp noise floor (< 1e-4,
    far below the 0.20 bar) AND the shared twin replicates the funnel
    (primary >= 0.35) — i.e. MOMENTUM-OWNED; the shared momentum buffer
    carrying COHERENT in-room install content into the corpus steps (the
    orthogonal corpus content cancels across steps; the install content
    accumulates — the ratio arithmetic that turns a per-step in-room
    magnitude of ~0.06 into a realized share of ~0.45) is the motel's
    whole mechanism at this optimizer.
  - P-e284s (REGISTERED HERE, the secondary): buffer separation does NOT
    spare the write — the SEP arm's post g0 stays in the missile-under-
    fire class (<< 0.5x the cited serial 0.2168; e273's two-body account:
    the corpus's orthogonal steps still move the same parameters the
    write lives in). DISCLOSED counter-possibility: buf_I under separation
    holds only coherent in-room install content (no orthogonal corpus
    pollution), so the install stream's own steps are cleaner — if the
    two-body account is wrong about the parameter collision, the SEP arm
    could land above the shared twin's post. The read decides; never a
    bar.

CHECKS (the dispatch's, in force): the machinery smoke FIRST (the
buffer-separation correctness is the smoke's centerpiece — verify the
install buffer never receives corpus content and vice versa,
machine-checked, per-step isolation + per-milestone composition); the
room certified AND bit-bound; the lr/momentum provenance bind; the
orthogonality gate instantiated; the first-batch CE identity across arms
(the draw-integrity texture check, non-halting); n=1 per arm (the lottery
note); nothing guaranteed.

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier); bursts
<= 175s (inside the dispatch's 180s), per-step thermal polls (BOTH
streams' opt steps) at a 78C margin, 40s cooldowns (the 30-60s window),
the 84C never-past line (inside the dispatch's 85C), polls persisted to
runs/_envelope_log.jsonl tagged e284:<ARM>:<phase>; CPU fp64 dense
projections (pocketfft workers 2); CPU probing threads 4; NO concurrent
GPU jobs (the two arms run sequentially with cooldowns between).

Outputs: runs/e284/{metrics.json (PROGRESSIVE), e284_buffer_missile.png,
REPORT.md (executor-written), run.log (gitignored)}; checkpoints
runs/checkpoints/e284_*.pt. No NOTES/THINKING/QUEUE/STATE edits (the
coordinator folds). Commit + push per phase.

Run:  cd lab && python e284_buffer_missile.py    (E284_SMOKE=1 shakedown)
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
from common import CharCorpus, cosine_lr, run_dir, save_json, set_seed  # noqa: E402

import e043_install as E43                             # noqa: E402 (REPO,
                                                      # find_occ, SPLICE_RNG,
                                                      # CORP_BS, MIX_RANDOM,
                                                      # LR, jsonable)
import g1b_continuity as GB                            # noqa: E402 — MUST be
                                                      # imported BEFORE G1
import g1_anchored_ball as G1                          # noqa: E402
import e261_rank_ladder as E261                        # noqa: E402 — THE
                                                      # MACHINERY, PORTED
                                                      # WHOLE BY IMPORT (the
                                                      # committed file is
                                                      # NOT modified)

torch.set_num_threads(4)           # shared machine (the CPU lane is shared)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import textwrap                                        # noqa: E402

SMOKE = os.environ.get("E284_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e284_smoke" if SMOKE else "e284"
assert torch.cuda.is_available(), "e284 owns the GPU lane (dispatch)"

T0 = time.time()
RD = run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


G1.log = log                                          # unify the timeline

# the shared instrument ledgers (defined BEFORE the rebinding below so
# e261's drivers land their rows in THIS cell's ledger)
device_events: list[dict] = []
thermal_log: list[dict] = []

# ---- THE REBINDING (e268/e273/e278's disclosed convention): e261's drivers
# resolve their module globals at CALL TIME through e261's namespace —
# rebound HERE so they write THIS cell's log, label THIS cell's envelope
# polls (e284:<tag>), and land their thermal rows in THIS cell's ledger.
E261.log = log
E261.NAME = NAME
E261.SMOKE = SMOKE
E261.T0 = T0
E261.thermal_log = thermal_log
E261.device_events = device_events
if SMOKE:
    E261.INST_STEPS = 8
    E261.CONS_STEPS = 8

# ======================================================================
# THE CONFIG (frozen)
# ======================================================================
BASE_CK = "e001.pt"               # the 2.74M corpus base (e043/e048's own B)
ROOT_CK = "g1c_root.pt"           # the committed fresh root (THE reference)
SPAN_CK = "e246_late_span.pt"     # e246's committed LATE span (the ledger's)
VMAP_CK = "e258_vmap.pt"          # e258's committed 2.74M v-map (the ledger's v)
ROOMS272_CK = "e272_rooms.pt"     # e272's committed rooms (the K10KR bit-bind)
CKPT_DIR = GB.CKPT_DIR

# ---- THE VEHICLE: e272's committed K10KR room (k=10k, seeds 27215/27216)
LADDER_FULL: tuple[tuple[int, int, int], ...] = (
    (10_000, 27215, 27216),       # K10KR — e272's fresh-seed 10k room
)
LADDER_SMOKE: tuple[tuple[int, int, int], ...] = (
    (512, 27215, 27216),
)
LADDER = LADDER_SMOKE if SMOKE else LADDER_FULL
RUNG = {k: ("K10KR" if not SMOKE else f"K{k}") for k, _, _ in LADDER}
ROOM_MODE = RUNG[LADDER[0][0]]       # the hook's mode (smoke names it K512)
E261.LADDER = LADDER            # the machinery's certify()/rooms read these
E261.RUNG_NAMES = RUNG

# the arms (execution order: the question's arm first — the primary; then
# the control twin; the same-session pair IS the discriminator)
ARMS = ("SEP", "SHA")
ARM_DESC = {
    "SEP": "(a) SEPARATE-BUFFER-MISSILE: SGD-M (the same stable lr, "
           "momentum 0.9, wd 0) with TWO momentum buffers — the install "
           "steps update buffer I only, the corpus steps (the "
           "orthogonalized stream, g_perp) update buffer C only. No other "
           "change.",
    "SHA": "(b) SHARED-BUFFER-MISSILE (the control twin, same session): "
           "identical but ONE shared buffer (this session's replication of "
           "e280's missile arm — the same-session pair is the "
           "discriminator; e280's committed numbers cited as the "
           "cross-check).",
}

# THIS cell's ONE fresh registered corpus stream (the e268-family
# convention: one fresh registered stream per concurrent cell — e268
# 26801, e269 26901, e270 27001, e271 27101, e273 27301, e278 27801,
# e280 28001, HERE 28401). BOTH arms draw the identical sequence from it
# (each arm its own generator instance at the same seed) — bit-identical
# corpus batches across arms, so the buffer topology is the arms' ONLY
# delta.
CORPUS_GEN_SEED = 28401

# ---- THE SGD-M CONFIG (frozen; e280's committed lr class VERBATIM) -------
SGD_MOMENTUM = 0.9                # e273's stable rider convention VERBATIM
SGD_WD = 0.0                      # e273's disclosed deviation (wd dropped)
LR_SGD_MATCHED = 21.7385748014537     # e273's committed calibration (md5-bound)
SGD_STABLE_FACTOR = 0.01              # e273's SGD001X rider factor
LR_STABLE = SGD_STABLE_FACTOR * LR_SGD_MATCHED   # 0.21738574801453703

# ---- the committed records, HARD-BOUND (Rule 12; md5-gated at runtime) ----
E278_METRICS = E43.REPO / "runs" / "e278" / "metrics.json"
E278_MD5 = "db14cdff1fd5021a5b255c12127ea9df"
E278_VERDICT = "UNDERTOW-REGARDLESS"
E278_MISSILE_POST_G0 = 0.0005453421035781503     # the AdamW missile's write
E278_MISSILE_ORTH = 4.572246705773293e-17
E278_MISSILE_INT_BAND = (0.48139764670780555,    # t100-400 interval fracs'
                         0.5987770769678085)     # (min, max) — the 48-60%

E280_METRICS = E43.REPO / "runs" / "e280" / "metrics.json"
E280_MD5 = "c1e52fa17144477d88b125147cd005a3"
E280_MISSILE_MEDIAN = 0.45529691870685485        # the SGD-M missile's primary
E280_MISSILE_POST_G0 = 0.0006920578307472169     # the shared missile's write
E280_MISSILE_ORTH = 4.5737387315673036e-17
E280_S10K_SERIAL_POST = 0.21675553917884827      # the cited serial SGD-M rung
E273_LRCAL = E43.REPO / "runs" / "e273" / "lr_calibration.json"
E273_LRCAL_MD5 = "de0b1c3e152c99d7867391c4592e7e24"
E273_LR_SGD = 21.7385748014537                    # the calibration record's value
ROOMS272_MD5 = "066944855b3295e8796c6ca28b2e498c"

# ---- the frozen bars' numbers ------------------------------------------------
MOMENTUM_BAR = 0.20               # MOMENTUM-OWNED: separate primary < 0.20
LANDSCAPE_BAR = 0.35              # LANDSCAPE-OWNED: separate primary >= 0.35
TWIN_PRECEDENT_BAR = 0.35         # the shared twin's funnel-replication bar
SURVIVE_FRAC = 0.5                # the survival secondary's SPARED form
ORTH_BAR = 1e-6                   # the missile's orthogonality gate
BUFSEP_ORTH_BAR = 1e-4            # G_BUFSEP: ||P_room buf_C||/||buf_C|| below
BUFSEP_INROOM_BAR = 0.99          # G_BUFSEP: ||P_room buf_I||/||buf_I|| above
VOCAB_EXPECT = 65
TRAJ_MILE = (1, 100, 200, 300, 400)
PRIMARY_MILE = (100, 200, 300, 400)   # e280's missile-read milestone set

REGISTERED = {
    "question_verbatim": "e284 separates them: SGD-M with SEPARATE "
        "momentum buffers per stream (the install's buffer and the corpus's "
        "buffer never mix). If orthogonality now survives in the APPLIED "
        "displacement, the re-aimer was buffer-sharing (the motel is a "
        "momentum-architecture); if it still funnels, the landscape itself "
        "bends steps.",
    "context_verbatim": "e278's guided missile (AdamW): an "
        "exactly-orthogonal corpus gradient (4.6e-17) still walked 48-60% "
        "in-room — Adam re-aims. e280's SGD-M missile: STILL funneled "
        "(median 0.4553) — P-x283b's direction refuted; the disclosed "
        "mechanical reading: the SHARED MOMENTUM BUFFER carries "
        "install-stream in-room content into the corpus steps.",
    "arms_verbatim": ARM_DESC,
    "bars_verbatim": {
        "MOMENTUM-OWNED": "the separate-buffer arm's in-room share < 0.20 "
            "(vs the shared twin's ~0.45+) — the re-aimer is "
            "buffer-sharing; THE MORTel IS A MOMENTUM-SHARING ARCHITECTURE; "
            "orthogonality is preservable by buffer separation. [sic — "
            "'MOTEL' in the dispatch's intent]",
        "LANDSCAPE-OWNED": "the separate-buffer arm still >= 0.35 in-room "
            "— the landscape itself bends applied steps toward the room; "
            "the motel is the space's, not any optimizer state's.",
        "MIXED": "0.20-0.35 or the twins disagree with their committed "
            "precedents — everything verbatim.",
    },
    "secondary_verbatim": "SECONDARY (never the primary): the write's "
        "survival in each arm (does buffer separation spare the write? "
        "e273's two-body account predicts NO — the parameters still "
        "collide).",
    "operationalizations": (
        "frozen BEFORE compute: THE VEHICLE := e272's committed K10KR room "
        "(k=10000, seeds 27215/27216) rebuilt + bit-gated vs e272_rooms.pt "
        "(G_ROOM10KR); e001 base; Dmix install s400 gen 24314, draw order "
        "VERBATIM; hook = clip 1.0 -> project CPU fp64 -> opt.step, norm "
        "NOT rescaled; THE OPTIMIZER := SGD(momentum 0.9, wd 0.0) at "
        f"LR_STABLE = {SGD_STABLE_FACTOR} x {LR_SGD_MATCHED!r} = "
        f"{LR_STABLE!r} x cosine_lr(s-1,1000) for BOTH streams (the paired "
        "lr; e280's committed lr class; G_LR_BIND re-derives from e273's "
        "md5-bound lr_calibration.json); THE INTERLEAVE := e268/e278's "
        "registered 1:1 form VERBATIM (corpus = 16 anchors + 32 random "
        f"windows from the ONE fresh registered stream seed "
        f"{CORPUS_GEN_SEED} — both arms draw the identical sequence; the "
        "buffer topology is the arms' ONLY delta; corpus grads "
        "orthogonalized g_perp = g - P_room(g), verified EVERY step); THE "
        "BUFFER TOPOLOGY := SEP = TWO SGD instances over the SAME "
        "parameters (opt_I install-only, opt_C corpus-only); SHA = ONE SGD "
        "(e280's committed form); THE PRIMARY READ := median over "
        "milestones {100,200,300,400} of the realized corpus displacement's "
        "INTERVAL in-room fraction (s1 excluded — e280's convention; "
        "cumulative co-reported; GPU fp32 accumulation, CPU fp64 projection "
        "at milestones); THE TWIN-PRECEDENT CLAUSE := the shared twin "
        "disagrees iff its primary median < 0.35 (e280's committed MOTEL "
        "bar; committed median 0.45529691870685485) -> MIXED named "
        "CONTROL-TWIN-OFF-PRECEDENT; COMPOSITE := TEXTURE (hard-gate "
        "failure) -> MIXED (CONTROL-TWIN-OFF-PRECEDENT) -> MOMENTUM-OWNED "
        "(sep < 0.20) -> LANDSCAPE-OWNED (sep >= 0.35) -> MIXED "
        "(BETWEEN-BARS); THE SURVIVAL SECONDARY := post g0 vs e280's "
        f"cited serial SGD-M 10k rung {E280_S10K_SERIAL_POST!r} (SPARED "
        ":= >= 0.5x) and e280's committed shared-missile post "
        f"{E280_MISSILE_POST_G0!r}; never a bar; HARD GATES := {{G_NAMEFREE,"
        " G_SPLICE, G_BATTERY, G_ANCHOR, G_INSTMASK, G_PARENTS, G_BASE, "
        "G_ROOT, G_VMBIND, G_SPANBIND, G_PROJ, G_ROOM10KR, G_LR_BIND, "
        "G_MISSILE_ORTH (INSTANTIATED — e280's omission corrected), "
        "G_BUFSEP}} — a failure HALTS; G_BUFSEP := per-step bitwise "
        "isolation snapshots (opt_I's buffers untouched by every "
        "opt_C.step() and vice versa; any mismatch HALTS) + per-milestone "
        "composition (||P_room buf_C||/||buf_C|| < "
        f"{BUFSEP_ORTH_BAR:.0e}; ||P_room buf_I||/||buf_I|| > "
        f"{BUFSEP_INROOM_BAR}); the SHARED arm's single-buffer in-room "
        "fraction is a READ, never gated; NO CONS; NO SERIAL ARM (e280's "
        "committed S10K rung cited + hard-bound as the survival "
        "denominator)."),
    "registration": "bars + question + arms + secondary frozen VERBATIM "
        "from the dispatch letter (the motel's ownership test); this "
        "script committed at birth BEFORE any compute; adjudicate against "
        "exactly this; no bar shopping.",
    "predictions": {
        "P-e284a_registered": "under separate buffers the corpus "
            "displacement's in-room share collapses to the fp noise floor "
            "(< 1e-4, far below the 0.20 bar) AND the shared twin "
            "replicates the funnel (primary >= 0.35) — MOMENTUM-OWNED; the "
            "shared momentum buffer carrying COHERENT in-room install "
            "content into the corpus steps (orthogonal corpus content "
            "cancels across steps; install content accumulates — the "
            "ratio arithmetic that turns a per-step in-room magnitude "
            "~0.06 into a realized share ~0.45) is the motel's whole "
            "mechanism at this optimizer.",
        "P-e284s_registered": "buffer separation does NOT spare the write "
            "(the SEP arm's post g0 stays in the missile-under-fire class, "
            "<< 0.5x the cited serial 0.2168; e273's two-body account: the "
            "corpus's orthogonal steps still move the same parameters the "
            "write lives in). Disclosed counter-possibility: buf_I under "
            "separation holds only coherent in-room install content, so "
            "the install steps are cleaner — the read decides; never a "
            "bar.",
        "P-x283b_T260_cited_dead": "refuted by e280 (the SGD-M missile "
            "funneled at 0.4553) — the record this cell decomposes.",
    },
}

deviations: list[str] = [
    "THE SMOKE CATCH (pass 1, runs/e284_smoke/; the e260-family record "
    "intact): the figure's primary-panel label formatted the primary "
    "median unconditionally — a None (smoke has no t100+ milestones) "
    "crashed at the figure stage (smoke-only; no bar/gate/arm/read "
    "touched); fixed with an n/a guard. THE SMOKE'S SUBSTANCE "
    "VERIFICATIONS (the centerpiece, all PASS): the buffer separation's "
    "isolation EXACT (16 bitwise snapshot checks, 0 violations — the "
    "install buffer never received corpus content and vice versa); the "
    "corpus buffer's composition at the fp FLOOR (buf_C in-room "
    "3.4e-10 -> 9.9e-10 over 8 steps, vs the 1e-4 bar — the linearity "
    "account confirmed); the install buffer 100.00% in-room (buf_I "
    "1.0000, vs the > 0.99 bar); the SHARED twin's single buffer "
    "already carrying rising in-room content (0.012 -> 0.023) while the "
    "separate arm's displacement read ~0.0000 in-room — the "
    "discination mechanically real from step 1; the first-batch CE "
    "identity across arms EXACT (bit-identical streams); the "
    "orthogonality 1e-17-class on both arms.",
    "THE SEPARATE-BUFFER ARITHMETIC IS LINEAR, AND THE MEASUREMENT IS "
    "STILL THE READ (disclosed at birth, not a bar): SGD momentum is "
    "linear in its stream's gradients — buf_C is a mu-discounted sum of "
    "orthogonalized corpus grads, each orthogonal to < 1e-6 relative, so "
    "P_room(buf_C) sits at the fp32 accumulation floor (~1e-7-1e-6) BY "
    "CONSTRUCTION and the separate arm's primary read is expected "
    "mechanically near zero (P-e284a). The cell's discriminating weight "
    "therefore rests equally on the SHARED twin's same-session funnel "
    "replication (the twin-precedent clause): if the twin fails to funnel "
    "(< 0.35), the verdict is the named MIXED CONTROL-TWIN-OFF-PRECEDENT "
    "and nothing about ownership is claimed. The e280 funnel's own "
    "arithmetic is disclosed: the shared buffer's in-room content is "
    "COHERENT install mass (per-step magnitude only ~kept_frac ~ 0.06) "
    "that accumulates geometrically while the ~1.0-magnitude orthogonal "
    "corpus content cancels across steps — the realized in-room SHARE "
    "(~0.45) is a ratio effect, which is exactly why the primary read is "
    "a share and not a magnitude.",
    "NO SERIAL ARM (budget + cited, frozen at birth): e280's committed "
    "S10K serial rung (post 0.21675553917884827; the bit-identical "
    "room/protocol/optimizer/lr) is the survival secondary's cited "
    "denominator, hard-bound in G_PARENTS; the primary read is a "
    "displacement ratio and needs no serial reference by construction. "
    "The same-session twin pair carries the discrimination.",
    "G_MISSILE_ORTH INSTANTIATED THIS TIME (e280's disclosure honored): "
    "e280 registered the gate in its hard-gate set but never wrote it "
    "into the gates dict (a bookkeeping omission caught at its fold; the "
    "orth ledger itself was recorded, max 4.57e-17). THIS cell writes the "
    "gate for BOTH arms over ALL corpus steps.",
    "NO CONS (T259/e281; e278/e280/e283's committed form): the landing "
    "read is a cons property (0.6508 from a fact-free base); the frozen "
    "bars read the WRITE and the DISPLACEMENT only; both arms' "
    "install-final states are checkpointed (e284_SEP_post.pt / "
    "e284_SHA_post.pt) for any later landing pass.",
    "THE CORPUS STREAM IS THIS CELL'S OWN FRESH SEED 28401 (the family's "
    "fresh-seed precedent: e269/e270/e271/e273/e278/e280 each ran the "
    "form on their own stream); BOTH arms draw the IDENTICAL sequence "
    "(each its own generator instance at the same seed) — bit-identical "
    "corpus batches across arms, the buffer topology the ONLY delta "
    "(e273's shared-stream-across-barrels precedent, tightened to "
    "shared-stream-across-arms).",
    "e261's MACHINERY PORTED WHOLE BY IMPORT: the SRCT projector, the "
    "room certification, the thermal envelope (per-step polls, 78C "
    "margin, 175s bursts — inside the dispatch's 180 — 40s cooldowns, the "
    "84C line inside the dispatch's 85), the progressive-metrics + "
    "resume-ckpt conventions. The ONE new driver is "
    "chunked_missile_buffers (this file): e280's chunked_missile_sgd "
    "(= e278's chunked_install_threenull MISSILE form under SGD) with the "
    "optimizer topology parameterized (ONE or TWO SGD instances) + the "
    "buffer-isolation snapshots + the buffer-composition ledger. The "
    "committed lab/e261_rank_ladder.py, lab/e278_three_null.py and "
    "lab/e280_sgd_ladder.py are NOT modified.",
    "THE V-MAP IS LOADED, NOT RE-RUN (extend, don't repeat): e258's "
    "committed 2.74M v-map feeds the measured v-excess ledger.",
    "n=1 per arm, one lineage, one session (the g-series standing caveat "
    "— the critic's lottery note carried verbatim); the arms' DIFFERENCE "
    "is the registered object, not any single point; nothing guaranteed.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
    "Smoke mode (E284_SMOKE=1): 8 install + 8 interleaved corpus steps "
    "per arm, room k=512 at the same seed pair (G_ROOM10KR vacuous — no "
    "committed record at smoke k; disclosed), both arms + the "
    "adjudication form + the figure exercised, the missile's "
    "orthogonality AND the buffer isolation checked EVERY step (the "
    "smoke's centerpiece), G_BUFSEP composition real (milestone 8), "
    "G_LR_BIND real (the calibration file is session-independent), all "
    "paths smoke_-prefixed, own smoke dir; NOTHING adjudicated or gated "
    "(SMOKE stamp on every read).",
]


# ------------------------------------------------------------------ envelope
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


def git_head() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(E43.REPO),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                       # noqa: BLE001
        return "unavailable"


# --------------------------------------------- THE MISSILE'S PROJECTION (port)
def orthogonalize_grads(proj: "E261.LadderRooms", params, mode: str) -> dict:
    """e278/e280's orthogonalize_grads VERBATIM: replace the (clipped) corpus
    gradient g by g_perp = g - P_room(g) — the component ENTIRELY ORTHOGONAL
    to the room (CPU fp64, write fp32, norm NOT rescaled). The verification
    read ||P_room g_perp|| / ||g_perp|| is returned for the gate (the
    projector is exact: P(I-P) = 0 to fp64 roundoff ~1e-15; any drift above
    1e-6 is an implementation bug, not numerics)."""
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


# --------------------------------------------- THE BUFFER MACHINES (new)
def snap_buffers(opt, params) -> list:
    """Snapshot an optimizer's per-param momentum buffers (None where the
    state has none yet) — the G_BUFSEP isolation probe's eyes."""
    st = opt.state
    return [st[p].get("momentum_buffer").detach().clone()
            if isinstance(st.get(p), dict)
            and "momentum_buffer" in st[p] else None
            for p in params]


def buffers_bitwise_equal(sn_a: list, sn_b: list) -> bool:
    if len(sn_a) != len(sn_b):
        return False
    for a, b in zip(sn_a, sn_b):
        if (a is None) != (b is None):
            return False
        if a is not None and not torch.equal(a, b):
            return False
    return True


def buffer_inroom_frac(opt, params, room) -> tuple:
    """(||P_room buf|| / ||buf||, ||buf||) of an optimizer's momentum
    buffer state — the G_BUFSEP composition probe (CPU fp64 projection)."""
    parts = []
    st = opt.state
    for p in params:
        d = st.get(p)
        if isinstance(d, dict) and "momentum_buffer" in d:
            parts.append(d["momentum_buffer"].detach().reshape(-1).cpu())
    if not parts:
        return None, 0.0
    b = torch.cat(parts).double().numpy().astype(np.float64)
    bn = float(np.linalg.norm(b))
    if bn == 0.0:
        return None, 0.0
    return float(np.linalg.norm(room.project(b)) / bn), bn


# ======================================================================
# THE BUFFER-MISSILE DRIVER — e280's chunked_missile_sgd with the
# optimizer TOPOLOGY parameterized (ONE or TWO SGD instances) + the
# buffer-isolation snapshots + the buffer-composition ledger
# ======================================================================
def chunked_missile_buffers(tag: str, net0, proj: "E261.LadderRooms",
                            base_flat_np: np.ndarray,
                            inst_x, inst_mask, anchor_full, train_ids,
                            g0_ids, gm12_ids, r_eval_xy, zid, mode: str,
                            resume_ck: Path, dev: torch.device,
                            arm: str) -> dict:
    """e278's MISSILE construction VERBATIM under SGD-M (e280's committed
    form) with the arms' ONLY delta = the momentum-buffer topology:

      SEP: opt_I = SGD(momentum 0.9, wd 0) stepped ONLY by install steps;
           opt_C = SGD(momentum 0.9, wd 0) stepped ONLY by corpus steps
           (the orthogonalized stream); identical lr schedules.
      SHA: ONE SGD stepped by both streams (e280's committed missile arm).

    The 1:1 interleave: install step s bit-identical to the family's
    serial installs (draws ix(16)/aj(16)/rj(32) from gen seed 24314; the
    64-window Dmix batch; the masked union CE; lr x cosine_lr(s-1,1000);
    backward -> clip 1.0 -> PROJECT onto the room -> opt_I.step) + ONE
    corpus step (16 anchors + 32 random corpus windows from cgen seed
    28401; full-window CE; the PAIRED lr; backward -> clip 1.0 ->
    g_perp = g - P_room(g), verified -> opt_C.step).

    The ledgers: the orthogonality ledger (every corpus step); the
    displacement ledger (GPU fp32 per-step accumulation; CPU fp64 interval
    + cumulative projections at the milestones); the buffer ledger
    (per-milestone composition of each buffer's in-room fraction); the
    G_BUFSEP isolation snapshots (EVERY step of the SEP arm — a mismatch
    HALTS). Thermal: a poll after EVERY opt step (both streams)."""
    assert arm in ("SEP", "SHA")
    name_bs, corp_bs, mix_random = (G1.NAME_BS, E43.CORP_BS, E43.MIX_RANDOM)
    lr = LR_STABLE
    n_steps = E261.INST_STEPS
    n_inst, n_anc = inst_x.shape[0], anchor_full.shape[0]
    N = int(base_flat_np.size)
    state = {"step": 0, "traj": [], "ledger": {}, "corpus_ledger": {},
             "orth_ledger": {}, "disp_ledger": [], "buf_ledger": [],
             "bufsep": {"isolation_checks": 0, "isolation_violations": 0},
             "corp_cum": torch.zeros(N, dtype=torch.float64),
             "corp_prev": torch.zeros(N, dtype=torch.float64)}
    if resume_ck.exists():
        state = torch.load(resume_ck, map_location="cpu", weights_only=False)
        log(f"  [{tag}] RESUMED from {resume_ck.name} at install step "
            f"{state['step']}/{n_steps}")
    if int(state.get("step", 0)) >= n_steps:
        log(f"  [{tag}] resume ckpt already COMPLETE at s{state['step']}")
        return {"sd": state["model"], "traj": state.get("traj", []),
                "ledger": state.get("ledger", {}),
                "corpus_ledger": state.get("corpus_ledger", {}),
                "orth_ledger": state.get("orth_ledger", {}),
                "disp_ledger": state.get("disp_ledger", []),
                "buf_ledger": state.get("buf_ledger", []),
                "bufsep": state.get("bufsep", {}),
                "orth_max": state.get("orth_max"),   # ALL steps, ckpted
                "orth_norm_ratio_first": None,
                "orth_norm_ratio_mean": None,
                "steps_ran": n_steps, "n_chunks": state.get("n_chunks", 0),
                "chunk_table": state.get("chunk_table", []),
                "resumed_final": True}
    net = opt_I = opt_C = gen = cgen = evl = None
    n_chunks, chunk_table = 0, []
    step = state["step"]
    t_burst, n_burst = None, 0
    chunk_temps: list[float] = []
    # the corpus-displacement accumulators (GPU fp32 per step — e278's
    # smoke performance catch; the projections stay CPU fp64 at milestones)
    corp_cum = state["corp_cum"].to(dev)
    corp_prev = state["corp_prev"].to(dev)
    room = proj.rooms[mode]
    orth_max = 0.0
    orth_ratio_first = None
    orth_ratio_sum, orth_ratio_n = 0.0, 0

    while step < n_steps:
        if t_burst is None:
            t_burst, n_burst = E261._open_burst(f"{tag}-chunk{n_chunks + 1}")
        n_chunks += 1
        chunk_temps = []
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            opt_I = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=SGD_MOMENTUM,
                                    weight_decay=SGD_WD)
            opt_C = (opt_I if arm == "SHA"
                     else torch.optim.SGD(net.parameters(), lr=lr,
                                          momentum=SGD_MOMENTUM,
                                          weight_decay=SGD_WD))
            gen = torch.Generator().manual_seed(E261.FRESH_GEN)
            cgen = torch.Generator().manual_seed(CORPUS_GEN_SEED)
            if resume_ck.exists():
                net.load_state_dict(state["model"])
                net.to(dev)
                net.train()
                opt_I.load_state_dict(state["optI"])
                if arm == "SEP":
                    opt_C.load_state_dict(state["optC"])
                gen.set_state(state["gen_state"])
                cgen.set_state(state["cgen_state"])
                step = state["step"]
            evl = copy.deepcopy(net0).to(CPU)
        devices_row = {"chunk": n_chunks, "device": str(dev),
                       "t_start": round(time.time() - T0, 1)}
        chunk_capped = False
        params_live = list(net.parameters())
        for step in range(step + 1, n_steps + 1):
            f = cosine_lr(step - 1, E261.INST_TOTAL)
            lr_now = lr * f
            for g_ in opt_I.param_groups:
                g_["lr"] = lr_now
            if opt_C is not opt_I:
                for g_ in opt_C.param_groups:
                    g_["lr"] = lr_now
            # ---- 1. THE INSTALL STEP (bit-identical to the family's s) ----
            # G_BUFSEP isolation probe A: opt_C's buffers must be UNTOUCHED
            # by the install step (snapshot -> compare after opt_I.step()).
            if arm == "SEP":
                sn_c_before = snap_buffers(opt_C, params_live)
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
            net.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            led = proj.step_hook(params_live, mode)
            opt_I.step()                         # buffer I only (or shared)
            if arm == "SEP":
                state["bufsep"]["isolation_checks"] += 1
                if not buffers_bitwise_equal(sn_c_before,
                                             snap_buffers(opt_C,
                                                          params_live)):
                    state["bufsep"]["isolation_violations"] += 1
                    raise RuntimeError(
                        f"[{tag}] G_BUFSEP ISOLATION VIOLATION at s{step}: "
                        "the install step modified buffer C — HALT")
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                state["ledger"][step] = {
                    "ce": float(loss.item()), "gn": led["gn"],
                    "gpn": led["gpn"], "kept_frac": led["kept_frac"],
                    "v_excess_pre": led["v_excess_pre"],
                    "v_excess_post": led["v_excess_post"],
                    "in_span_frac": led["in_span_frac"],
                    "applied_in_span_frac": led["applied_in_span_frac"]}
            n_burst += 1
            ok_t, temp = E261.burst_temp_check(f"{tag}-c{n_chunks}.i")
            chunk_temps.append(temp)
            # ---- 2. THE CORPUS STEP (the orthogonalized stream) ----------
            # G_BUFSEP isolation probe B: opt_I's buffers must be UNTOUCHED
            # by the corpus step (snapshot -> compare after opt_C.step()).
            if arm == "SEP":
                sn_i_before = snap_buffers(opt_I, params_live)
            aj_c = torch.randint(n_anc, (corp_bs - mix_random,),
                                 generator=cgen)
            rj_c = torch.randint(len(train_ids) - G1.BLOCK - 1,
                                 (mix_random,), generator=cgen)
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
            orth_row = orthogonalize_grads(proj, net.parameters(), mode)
            orth_max = max(orth_max, orth_row["orth_rel_err"])
            r = orth_row["norm_ratio"]
            orth_ratio_first = r if orth_ratio_first is None \
                else orth_ratio_first
            orth_ratio_sum += r
            orth_ratio_n += 1
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                state["orth_ledger"][step] = {
                    "gn_clipped": gn_c,
                    "gperp_norm": orth_row["gperp_norm"],
                    "norm_ratio": r,
                    "in_room_frac": orth_row["in_room_frac"],
                    "orth_rel_err": orth_row["orth_rel_err"]}
            theta_b = torch.cat([p.detach().reshape(-1)
                                 for p in net.parameters()])
            opt_C.step()                    # buffer C only (or shared)
            if arm == "SEP":
                state["bufsep"]["isolation_checks"] += 1
                if not buffers_bitwise_equal(sn_i_before,
                                             snap_buffers(opt_I,
                                                          params_live)):
                    state["bufsep"]["isolation_violations"] += 1
                    raise RuntimeError(
                        f"[{tag}] G_BUFSEP ISOLATION VIOLATION at s{step}: "
                        "the corpus step modified buffer I — HALT")
            with torch.no_grad():
                corp_cum += torch.cat([p.detach().reshape(-1)
                                       for p in net.parameters()]) - theta_b
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                state["corpus_ledger"][step] = {"ce": float(loss_c.item()),
                                                "gn_clipped": gn_c}
            n_burst += 1
            ok_t2, temp2 = E261.burst_temp_check(f"{tag}-c{n_chunks}.x")
            chunk_temps.append(temp2)
            if (step in TRAJ_MILE or step == n_steps or SMOKE):
                sd_cpu = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
                evl.load_state_dict(sd_cpu)
                evl.eval()
                bz0 = G1.battery_cell(evl, g0_ids, zid)
                bz12 = G1.battery_cell(evl, gm12_ids, zid)
                ce_r = G1.ce_fixed_cpu(evl, *r_eval_xy)
                d_mil = flat_params_cpu(net).double().numpy() - base_flat_np
                ld_mil = proj.displacement_loads(torch.from_numpy(d_mil),
                                                 mode)
                dn = float(np.linalg.norm(d_mil))
                v_int_np = (corp_cum - corp_prev).double().cpu().numpy()
                vn = float(np.linalg.norm(v_int_np))
                if vn > 0:
                    pv = room.project(v_int_np)
                    in_room_frac_c = float(np.linalg.norm(pv) / vn)
                else:
                    in_room_frac_c = None
                corp_prev = corp_cum.clone()
                cum_np = corp_cum.double().cpu().numpy()
                cum_n = float(np.linalg.norm(cum_np))
                pcum = room.project(cum_np)
                in_room_frac_cum = (float(np.linalg.norm(pcum) / cum_n)
                                    if cum_n > 0 else None)
                # the buffer-composition ledger (G_BUFSEP's composition
                # probe + the shared arm's mixture read)
                if arm == "SEP":
                    bufC_ir, bufC_n = buffer_inroom_frac(opt_C,
                                                         params_live, room)
                    bufI_ir, bufI_n = buffer_inroom_frac(opt_I,
                                                         params_live, room)
                    shared_ir = None
                else:
                    bufC_ir = bufI_ir = None
                    shared_ir, bufI_n = buffer_inroom_frac(opt_I,
                                                          params_live, room)
                    bufC_n = bufI_n
                state["disp_ledger"].append({
                    "step": step, "interval_norm": vn,
                    "interval_in_room_frac": in_room_frac_c,
                    "cum_norm": cum_n,
                    "cum_in_room_frac": in_room_frac_cum,
                    "cum_in_room_norm": (float(np.linalg.norm(pcum))
                                         if cum_n > 0 else 0.0)})
                state["buf_ledger"].append({
                    "step": step,
                    "bufC_inroom_frac": bufC_ir, "bufC_norm": bufC_n,
                    "bufI_inroom_frac": bufI_ir, "bufI_norm": bufI_n,
                    "shared_buf_inroom_frac": shared_ir})
                state["traj"].append({
                    "step": step,
                    "g0_pz": bz0["mean_pz"], "g0_argmax": bz0["frac_argmax_z"],
                    "gm12_pz": bz12["mean_pz"], "ce_r": ce_r,
                    "ce_batch": float(loss.item()),
                    "ce_corpus": float(loss_c.item()),
                    "kept_frac": led["kept_frac"],
                    "v_excess_post": led["v_excess_post"],
                    "in_span_frac": led["in_span_frac"],
                    "applied_in_span_frac": led["applied_in_span_frac"],
                    "disp_norm_cum": dn,
                    "in_room_disp_norm_cum": ld_mil["in_own_room"] * dn,
                    "in_own_room_frac_cum": ld_mil["in_own_room"],
                    "v_excess_cum": ld_mil["v_excess"],
                    "corpus_disp_interval_norm": vn,
                    "corpus_disp_interval_in_room_frac": in_room_frac_c,
                    "corpus_disp_cum_in_room_frac": in_room_frac_cum,
                    "orth_max_so_far": orth_max,
                    "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] s{step:4d} g0 {bz0['mean_pz']:.5f} g-12 "
                    f"{bz12['mean_pz']:.5f} CE_R {ce_r:.4f} CE_inst "
                    f"{float(loss.item()):.4f} CE_corp "
                    f"{float(loss_c.item()):.4f} kept "
                    f"{led['kept_frac']:.4f} |d| {dn:.4f} corp|v| "
                    f"{vn if vn else 0.0:.3f} "
                    f"({('in-room %.4f' % in_room_frac_c) if in_room_frac_c is not None else 'n/a'})"
                    + (f" bufC {bufC_ir:.2e}" if bufC_ir is not None
                       else (f" shared-buf {shared_ir:.3f}"
                             if shared_ir is not None else ""))
                    + f" orth {orth_max:.1e}")
            if not (ok_t and ok_t2):
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
                    "optI": opt_I.state_dict(),
                    "optC": (opt_C.state_dict() if opt_C is not opt_I
                             else None),
                    "gen_state": gen.get_state(),
                    "cgen_state": cgen.get_state(),
                    "step": step, "traj": state["traj"],
                    "ledger": state["ledger"],
                    "corpus_ledger": state["corpus_ledger"],
                    "orth_ledger": state["orth_ledger"],
                    "disp_ledger": state["disp_ledger"],
                    "buf_ledger": state["buf_ledger"],
                    "bufsep": state["bufsep"],
                    "corp_cum": corp_cum.cpu(),
                    "corp_prev": corp_prev.cpu(),
                    "orth_max": orth_max,
                    "n_chunks": n_chunks, "chunk_table": chunk_table},
                   resume_ck)
        if step >= n_steps:
            break
        if n_chunks >= 16:
            raise RuntimeError(f"{tag}: cap-loop guard at chunk {n_chunks}")
        E261.burst_cooldown(tag)
        t_burst = None
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    return {"sd": sd_cpu, "traj": state["traj"], "ledger": state["ledger"],
            "corpus_ledger": state["corpus_ledger"],
            "orth_ledger": state["orth_ledger"],
            "disp_ledger": state["disp_ledger"],
            "buf_ledger": state["buf_ledger"],
            "bufsep": state["bufsep"],
            "orth_max": orth_max,
            "orth_norm_ratio_first": orth_ratio_first,
            "orth_norm_ratio_mean": (orth_ratio_sum / orth_ratio_n
                                     if orth_ratio_n else None),
            "steps_ran": step, "n_chunks": n_chunks,
            "chunk_table": chunk_table}


# ------------------------------------------------------------------ main
metrics: dict = {}


def write_partial(note: str) -> None:
    metrics["date"] = common.now_iso()
    metrics["phase_note"] = note
    metrics["device_events"] = device_events
    save_json(RD / "metrics.json", E43.jsonable(metrics))
    log(f"WROTE partial metrics ({note})")


def _envelope_summary() -> dict:
    out = {"burst_cap_s": E261.BURST_MAX_S, "cooldown_s": E261.COOLDOWN_S,
           "per_step_polls": "after EVERY opt step (both streams, both "
                             "arms) — aggregated from "
                             "runs/_envelope_log.jsonl (the persisted "
                             "ledger; survives resume passes)",
           "early_end_margin_c": E261.TEMP_EARLY_END,
           "hard_line_c": E261.TEMP_HARD,
           "dispatch_envelope": "bursts <= 180s, cooldowns 30-60s, never "
                                "past 85C — this cell runs 175/40/84 (all "
                                "inside)"}
    temps = []
    try:
        with open(E43.REPO / "runs" / "_envelope_log.jsonl",
                  encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tag = str(row.get("tag", ""))
                if tag.startswith(f"{NAME}:") and row.get("temp") is not None:
                    temps.append(float(row["temp"]))
    except FileNotFoundError:
        pass
    out["n_polls"] = len(temps)
    out["max_temp_seen_c"] = max(temps) if temps else (
        max((r["temp"] for r in thermal_log), default=None))
    out["violations_ge_84c"] = sum(1 for t in temps if t >= E261.TEMP_HARD)
    out["note"] = ("aggregated across ALL passes of this cell; the smoke's "
                   f"polls are tagged e284_smoke: and excluded")
    return out


def med(xs) -> float:
    xs = sorted(xs)
    return float(xs[len(xs) // 2]) if xs else None


def primary_read(disp_ledger: list) -> tuple:
    """THE PRIMARY READ: median over milestones {100,200,300,400} of the
    realized corpus displacement's INTERVAL in-room fraction (e280's
    missile-read convention; the s1 warm-up interval excluded)."""
    fr = [d["interval_in_room_frac"] for d in disp_ledger
          if d["step"] in PRIMARY_MILE
          and d.get("interval_in_room_frac") is not None]
    cum = [d["cum_in_room_frac"] for d in disp_ledger
           if d["step"] in PRIMARY_MILE
           and d.get("cum_in_room_frac") is not None]
    return med(fr), fr, cum


def main():
    dev = torch.device("cuda")
    metrics.update({
        "experiment": "e284_buffer_missile",
        "phase": "THE SEPARATE-BUFFER MISSILE — the motel's ownership "
                 "test, the morning's last open mechanism question: e280's "
                 "SGD-M missile STILL funneled (median 0.4553) under a "
                 "SHARED momentum buffer; e284 separates the buffers "
                 "(install buffer I / corpus buffer C never mix) vs the "
                 "same-session shared twin — MOMENTUM-OWNED (< 0.20 "
                 "in-room) vs LANDSCAPE-OWNED (>= 0.35) vs MIXED, "
                 "adjudicated on the realized corpus displacement's "
                 "in-room share (median over milestones)",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "no_cons": {"cons_run": False,
                    "why": "T259/e281: the landing read is a cons property "
                           "(0.6508 from a fact-free base; the cons teaches "
                           "from anything); the frozen bars read the WRITE "
                           "and the DISPLACEMENT only; e278/e280/e283's "
                           "committed NO-CONS form; both arms' states are "
                           "checkpointed"},
        "envelope": {
            "device": "cuda fp32 training (this cell owns the GPU lane; "
                      "the arms run SEQUENTIALLY with cooldowns between — "
                      "never concurrent) + CPU fp64 dense projections "
                      "(pocketfft workers 2), CPU probing threads 4",
            "bursts": f"<= {E261.BURST_MAX_S:.0f}s (dispatch 180), "
                      f"per-step thermal polls at a "
                      f"{E261.TEMP_EARLY_END:.0f}C margin, cooldown "
                      f"{E261.COOLDOWN_S:.0f}s (dispatch 30-60), the "
                      f"{E261.TEMP_HARD:.0f}C never-past line (dispatch "
                      "85) recorded to runs/_envelope_log.jsonl tagged "
                      "e284:<ARM>:<phase>",
            "trainings": "2 missile interleaves x 800 SGD-M opt steps "
                         "(SEP: two buffers; SHA: one shared); NO cons; NO "
                         "serial arm (e280's committed S10K rung cited)",
        },
        "arms_desc": ARM_DESC,
        "interleave": {
            "ratio": "1:1 — after EVERY install step s, ONE corpus step "
                     "(e268/e278's registered form, e280's SGD port)",
            "install_step": "bit-identical to the family's serial installs: "
                            "draws ix(16)/aj(16)/rj(32) from gen seed 24314 "
                            "(the same draw order), the 64-window Dmix "
                            "batch, the masked union CE, lr "
                            f"{LR_STABLE:.13f} x cosine_lr(s-1, 1000), "
                            "clip 1.0 -> PROJECT onto the K10KR room -> "
                            "opt_I.step (buffer I)",
            "corpus_step": "48 windows = 16 original-host anchors (the "
                           "same 60-window bank) + 32 random corpus "
                           "windows (the g1c Dmix corpus convention); "
                           "full-window CE; draws from the ONE fresh "
                           f"registered generator seed {CORPUS_GEN_SEED} "
                           "(bit-identical batches across the two arms — "
                           "the buffer topology is the arms' ONLY delta); "
                           "the PAIRED lr; clip 1.0 -> g_perp = g - "
                           "P_room(g) (verified EVERY step) -> opt_C.step "
                           "(buffer C)",
            "buffer_topology": "SEP: opt_I stepped ONLY by install steps, "
                               "opt_C ONLY by corpus steps (TWO SGD "
                               "instances over the SAME parameters, "
                               "momentum 0.9, wd 0, identical schedules); "
                               "SHA: ONE SGD stepped by both (e280's "
                               "committed form)",
            "dose_delta_disclosed": "install dose IDENTICAL across arms "
                                    "(16 masked windows x 400); corpus "
                                    "exposure IDENTICAL and bit-identical "
                                    "(the one fresh stream); ONLY the "
                                    "buffer topology differs",
        },
        "deviations": deviations,
        "builds_on": [
            "T262 / e280 (the SGD-M missile: STILL funneled at median "
            "0.4553 — P-x283b refuted; the disclosed mechanical reading — "
            "the SHARED MOMENTUM BUFFER carries install-stream in-room "
            "content into the corpus steps — is THIS cell's hypothesis; "
            "the lr class LR_STABLE 0.21738574801453703 / momentum 0.9 / "
            "wd 0 is e280's committed record, bound)",
            "T260 / e278 (THE ROACH MOTEL: the AdamW missile's orthogonal "
            "gradient walked 48-60% in-room; the missile construction — "
            "the orthogonal projection, the orthogonality ledger, the "
            "displacement ledger — ORIGINATES here, ported verbatim)",
            "T261 / e283 (the motel is a FORMING-regime phenomenon: the "
            "established write's orthogonal stream walked only 0.07-0.11 "
            "in-room — this cell's forming-regime ownership question is "
            "the motel's last open clause)",
            "T258 / e273 (TRAJECTORY-TWO-BODY: the coupling is in the "
            "parameters — the survival secondary's NO-sparing prediction)",
            "T253/e273 + consult #006 (the lr calibration: matched "
            "diverges; the x0.01 stable rider — this cell's lr)",
            "T239 / e261 (the ladder machinery PORTED WHOLE BY IMPORT)",
            "T259 / e281 (the NO-CONS form)",
        ],
        "whats_new": [
            "THE BUFFER-SEPARATION INTERVENTION ITSELF (the record's "
            "first): the motel's re-aimer localized to optimizer STATE vs "
            "LANDSCAPE by splitting the momentum buffer per stream — the "
            "cleanest one-delta form (bit-identical streams, bit-identical "
            "room, identical lrs; ONLY the buffers differ)",
            "G_BUFSEP (the record's first buffer-topology gate): per-step "
            "bitwise isolation snapshots + per-milestone composition "
            "probes (the corpus buffer must sit at the fp noise floor; "
            "the install buffer must BE the in-room stream)",
            "G_MISSILE_ORTH INSTANTIATED (e280's bookkeeping omission "
            "corrected): the orthogonality gate over ALL corpus steps of "
            "BOTH arms, written into the gates dict",
        ],
        "gates": {},
    })
    log(f"E284 — THE SEPARATE-BUFFER MISSILE (smoke={SMOKE}) -> {RD}")
    log(f"arms: {'/'.join(ARMS)}; vehicle = e272's K10KR room (seeds "
        f"{LADDER[0][1]}/{LADDER[0][2]}, bit-gated vs {ROOMS272_CK}); "
        f"optimizer = SGD(m={SGD_MOMENTUM}, wd={SGD_WD}) at LR_STABLE "
        f"{LR_STABLE:.13f} x cosine; primary = the corpus displacement's "
        f"in-room share (median t100-400): MOMENTUM-OWNED < {MOMENTUM_BAR} "
        f"/ LANDSCAPE-OWNED >= {LANDSCAPE_BAR} / MIXED between-or-"
        f"twin-off-precedent (< {TWIN_PRECEDENT_BAR})")
    write_partial("startup (bars + composite registered, committed at "
                  "birth)")
    set_seed(CORPUS_GEN_SEED)       # global init only; every RNG is its own

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

    bat_ids, held_ids = {}, {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
        hs = [train_text[p - G1.PRE - j: p] for p, _ in held_occ]
        held_ids[j] = torch.stack([corpus.encode(c) for c in hs])
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

    def build_win(p, host):
        return torch.cat([train_ids[p - G1.PRE: p], name_ids,
                          train_ids[p + len(host):
                                    p + len(host) + G1.POST_CAP]])

    win_i = torch.stack([build_win(p, h) for p, h in install_occ])
    inst_x = win_i.clone()
    anchor_full = torch.stack([train_ids[p - G1.PRE: p - G1.PRE + G1.BLOCK]
                               for p, _ in install_occ])
    inst_mask = torch.zeros(60, G1.BLOCK - 1, dtype=torch.bool)
    inst_mask[:, G1.PRE - 1: G1.PRE - 1 + len(G1.NAME)] = True
    G_INSTMASK = {"name_positions": int(inst_mask.sum()),
                  "expected": 60 * len(G1.NAME),
                  "pass": bool(int(inst_mask.sum()) == 60 * len(G1.NAME))}
    assert G_INSTMASK["pass"], f"install mask gate FAILED: {G_INSTMASK}"

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
                                 "n_windows": 16, "starts": n_starts,
                                 "note": "built for protocol identity; NO "
                                         "wash/cons runs in this cell"},
                "pass": bool(len(n_starts) == 16)}
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"

    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60, G1.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    metrics["gates"].update({"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                             "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                             "G_INSTMASK": G_INSTMASK})
    log("P0: protocol gates PASS (namefree / splice 19+41 / battery shapes "
        "/ e170 bank / install mask / vocab 65)")
    write_partial("P0 protocol gates PASSED")

    # ================= P0b: the parents hard-bound (Rule 12) ============
    e278m = json.loads(E278_METRICS.read_text(encoding="utf-8"))
    e278_arm = e278m["arms"]["MISSILE"]["install"]
    e278_post = e278_arm["post_cells"]["g0"]
    e278_orth = e278_arm["orth_max_rel_err"]
    e278_ints = {t["step"]: t["corpus_disp_interval_in_room_frac"]
                 for t in e278_arm["traj"]}
    e278_band = (min(e278_ints[k] for k in PRIMARY_MILE),
                 max(e278_ints[k] for k in PRIMARY_MILE))
    e280m = json.loads(E280_METRICS.read_text(encoding="utf-8"))
    e280_arm = e280m["arms"]["MISSILE_SGD"]["install"]
    e280_med_i = e280_arm["missile_read_interval_median_t100_400"]
    e280_post = e280_arm["post_cells"]["g0"]
    e280_orth = e280_arm["orth_max"]
    e280_s10k = e280m["sgd_ladder"]["posts"]["10000"]
    lrcal = json.loads(E273_LRCAL.read_text(encoding="utf-8"))
    G_PARENTS = {
        "e278_metrics": {"path": str(E278_METRICS),
                         "md5": md5of(E278_METRICS), "bound_md5": E278_MD5,
                         "verdict": e278m["adjudication"]["verdict"],
                         "missile_post_g0": e278_post,
                         "missile_orth_max": e278_orth,
                         "missile_interval_band_t100_400": e278_band,
                         "note": "THE AdamW missile (the 48-60% funnel; "
                                 "orth 4.6e-17) — the motel's Adam-side "
                                 "record"},
        "e280_metrics": {"path": str(E280_METRICS),
                         "md5": md5of(E280_METRICS), "bound_md5": E280_MD5,
                         "missile_median_t100_400": e280_med_i,
                         "missile_post_g0": e280_post,
                         "missile_orth_max": e280_orth,
                         "s10k_serial_post_g0": e280_s10k,
                         "note": "THE SGD-M shared missile (median 0.4553 — "
                                 "the funnel this cell decomposes) + the "
                                 "serial SGD-M 10k rung (the survival "
                                 "secondary's cited denominator)"},
        "e273_lr_calibration": {"path": str(E273_LRCAL),
                                "md5": md5of(E273_LRCAL),
                                "bound_md5": E273_LRCAL_MD5,
                                "lr_sgd": lrcal["lr_sgd"],
                                "note": "LR_STABLE's provenance record"},
        "e272_rooms": {"path": f"runs/checkpoints/{ROOMS272_CK}",
                       "md5": md5of(CKPT_DIR / ROOMS272_CK),
                       "bound_md5": ROOMS272_MD5,
                       "note": "THE ROOM FILE (the K10KR bit-bind itself is "
                               "G_ROOM10KR)"},
        "e246_span": {"path": f"runs/checkpoints/{SPAN_CK}",
                      "md5": md5of(CKPT_DIR / SPAN_CK),
                      "bound_md5": E261.E246_SPAN_MD5},
        "e258_vmap": {"path": f"runs/checkpoints/{VMAP_CK}",
                      "md5": md5of(CKPT_DIR / VMAP_CK)},
        "hardbound": {
            "e278_verdict": E278_VERDICT,
            "e278_missile_post_g0": E278_MISSILE_POST_G0,
            "e278_missile_orth": E278_MISSILE_ORTH,
            "e278_missile_interval_band": E278_MISSILE_INT_BAND,
            "e280_missile_median": E280_MISSILE_MEDIAN,
            "e280_missile_post_g0": E280_MISSILE_POST_G0,
            "e280_missile_orth": E280_MISSILE_ORTH,
            "e280_s10k_serial_post": E280_S10K_SERIAL_POST,
            "lr_sgd": E273_LR_SGD},
        "pass": bool(
            e278m["adjudication"]["verdict"] == E278_VERDICT
            and abs(e278_post - E278_MISSILE_POST_G0) < 1e-12
            and abs(e278_orth - E278_MISSILE_ORTH) < 1e-24
            and abs(e278_band[0] - E278_MISSILE_INT_BAND[0]) < 1e-12
            and abs(e278_band[1] - E278_MISSILE_INT_BAND[1]) < 1e-12
            and abs(e280_med_i - E280_MISSILE_MEDIAN) < 1e-12
            and abs(e280_post - E280_MISSILE_POST_G0) < 1e-12
            and abs(e280_orth - E280_MISSILE_ORTH) < 1e-24
            and abs(e280_s10k - E280_S10K_SERIAL_POST) < 1e-12
            and float(lrcal["lr_sgd"]) == E273_LR_SGD
            and md5of(E278_METRICS) == E278_MD5
            and md5of(E280_METRICS) == E280_MD5
            and md5of(E273_LRCAL) == E273_LRCAL_MD5
            and md5of(CKPT_DIR / ROOMS272_CK) == ROOMS272_MD5
            and md5of(CKPT_DIR / SPAN_CK) == E261.E246_SPAN_MD5),
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log(f"P0b: G_PARENTS PASS — e278 {E278_VERDICT} (missile post "
        f"{e278_post:.7f}, intervals {e278_band[0]:.4f}-"
        f"{e278_band[1]:.4f}, orth {e278_orth:.1e}); e280's SGD missile "
        f"(median {e280_med_i:.4f}, post {e280_post:.7f}, orth "
        f"{e280_orth:.1e}); e280's serial S10K {e280_s10k:.6f}; the lr "
        f"calibration {E273_LR_SGD}")
    write_partial("P0b parents hard-bound")
    del e278m, e280m, lrcal

    # ---- G-BASE: the 2.74M corpus base, loaded fixed + fact-free --------
    base_net = G1.load_g1(CKPT_DIR / BASE_CK)
    assert base_net.num_params() == GB.G1B_PARAMS, \
        f"base param count {base_net.num_params()} != {GB.G1B_PARAMS}"
    base_sd = {k: v.detach().clone() for k, v in base_net.state_dict().items()}
    base_gm12 = G1.battery_cell(base_net, gm12_ids, zid)["mean_pz"]
    base_g0 = G1.battery_cell(base_net, g0_ids, zid)["mean_pz"]
    base_ce_r = G1.ce_fixed_cpu(base_net, *r_eval_xy)
    G_BASE = {"checkpoint": f"runs/checkpoints/{BASE_CK}",
              "params": GB.G1B_PARAMS,
              "fact_free_gm12": base_gm12, "ce_r": base_ce_r,
              "g0_volume_null_floor": base_g0,
              "fact_free": bool(base_gm12 <= 0.05),
              "pass": bool(base_gm12 <= 0.05)}
    assert G_BASE["pass"], f"G-BASE FAILED: {G_BASE}"
    del base_net
    metrics["gates"]["G_BASE"] = G_BASE
    log(f"G-BASE: {BASE_CK} ({GB.G1B_PARAMS} params), fact-free "
        f"(g-12 {base_gm12:.4f}, g0 floor {base_g0:.2e}, CE_R "
        f"{base_ce_r:.4f}): PASS")
    write_partial("P0c G-BASE PASSED")

    # ================= P1: THE ROOM (v-map + span + cert + bit-bind) ====
    log("=" * 78)
    root_net = G1.load_g1(CKPT_DIR / ROOT_CK)
    n_par = root_net.num_params()
    theta_root = flat_params_cpu(root_net)
    root_read = G1.battery_cell(root_net, gm12_ids, zid)["mean_pz"]
    G_ROOT = {
        "checkpoint": f"runs/checkpoints/{ROOT_CK}",
        "n_params": n_par, "expected_params": GB.G1B_PARAMS,
        "battery_read_measured": root_read,
        "battery_read_committed": E261.G1C_ROOT_GM12,
        "abs_diff": abs(root_read - E261.G1C_ROOT_GM12),
        "tol": E261.G_READ_TOL,
        "flat_md5": hashlib.md5(theta_root.numpy().tobytes()).hexdigest(),
        "pass": bool(n_par == GB.G1B_PARAMS
                     and abs(root_read - E261.G1C_ROOT_GM12)
                     < E261.G_READ_TOL)}
    assert G_ROOT["pass"], f"root gate FAILED: {G_ROOT}"
    metrics["gates"]["G_ROOT"] = G_ROOT
    log(f"P1 G_ROOT: {ROOT_CK} — {n_par} params; battery read "
        f"{root_read:.10f} vs committed {E261.G1C_ROOT_GM12:.10f} "
        f"(|d| {abs(root_read - E261.G1C_ROOT_GM12):.1e}): PASS")
    write_partial("P1 G_ROOT PASSED")
    del root_net

    N = n_par
    base_flat = flat_params_cpu(G1.evl_load(base_sd))
    base_flat_np = base_flat.double().numpy().astype(np.float64)

    vmap_art = torch.load(CKPT_DIR / VMAP_CK, map_location="cpu",
                          weights_only=False)
    v_flat32 = vmap_art["model"]["v_flat_fp32"]
    v64_np = v_flat32.numpy().astype(np.float64)
    G_VMBIND = {
        "path": f"runs/checkpoints/{VMAP_CK}", "md5": md5of(CKPT_DIR / VMAP_CK),
        "meta_experiment": vmap_art.get("meta", {}).get("experiment"),
        "meta_k": vmap_art.get("meta", {}).get("k"),
        "size": int(v_flat32.numel()), "expected_size": N,
        "mean_v": float(v64_np.mean()),
        "pass": bool(vmap_art.get("meta", {}).get("experiment") == "e258"
                     and int(v_flat32.numel()) == N
                     and (SMOKE or int(vmap_art["meta"]["k"])
                          == E261.E258_K_HARD)),
    }
    assert G_VMBIND["pass"], f"v-map bind failed: {G_VMBIND}"
    metrics["gates"]["G_VMBIND"] = G_VMBIND
    log(f"P1 G_VMBIND: {VMAP_CK} (md5 {G_VMBIND['md5'][:8]}..., "
        f"{G_VMBIND['size']} coords): PASS")

    span_art = torch.load(CKPT_DIR / SPAN_CK, map_location="cpu",
                          weights_only=False)
    Vp = span_art["Vp"].contiguous()
    G_SPANBIND = {"md5": md5of(CKPT_DIR / SPAN_CK),
                  "rank": int(Vp.shape[0]), "N": int(Vp.shape[1]),
                  "meta_experiment": span_art.get("meta", {}).get("experiment"),
                  "pass": bool(md5of(CKPT_DIR / SPAN_CK) == E261.E246_SPAN_MD5
                               and int(Vp.shape[0]) == E261.E246_SPAN_RANK
                               and int(Vp.shape[1]) == N
                               and span_art.get("meta", {}).get("experiment")
                               == "e246")}
    assert G_SPANBIND["pass"], f"span bind failed: {G_SPANBIND}"
    metrics["gates"]["G_SPANBIND"] = G_SPANBIND
    log(f"P1 G_SPANBIND: {SPAN_CK} (rank {G_SPANBIND['rank']}): PASS")

    params_ref = list(G1.evl_load(base_sd).parameters())
    rooms = E261.LadderRooms(N, LADDER, v64_np, Vp.numpy().astype(np.float64),
                             params_ref, dev)
    cert = rooms.certify()
    G_PROJ = {
        "form": "the K10KR room certified (fp64 CPU, "
                f"{E261.CERT_PROBES} probes, seed {E261.CERT_SEED}): the "
                "DCT roundtrip identity; IDEMPOTENCY and the kept^2 rank "
                "probe (||P x||^2/||x||^2 vs k/N, the 10-sigma bar "
                "5*sqrt(2k)/N); the span-overlap (expect ~sqrt(k/N))",
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
            f"{r['kept2_expect']:.6f} (bar {r['kept2_bar_10sig']:.1e}) "
            f"span-ovl {r['span_overlap_mean']:.4f} "
            f"(expect ~{r['span_overlap_expect']:.4f})")

    # ---- G_ROOM10KR: bit-identity vs e272's committed K10KR room --------
    rooms272 = torch.load(CKPT_DIR / ROOMS272_CK, map_location="cpu",
                          weights_only=False)

    def _to_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    room_name = RUNG[LADDER[0][0]]
    if not SMOKE:
        D272 = _to_np(rooms272["model"]["K10KR"]["D_int8"]).astype(np.float64)
        S272 = _to_np(rooms272["model"]["K10KR"]["S"])
        D_mine = rooms.rooms[room_name].D
        S_mine = rooms.rooms[room_name].S
        G_ROOM10KR = {
            "form": "the vehicle's room == e272's committed K10KR room "
                    "(seeds 27215/27216 at k=10,000): the +-1 diagonal and "
                    "the index set bit-identical to e272_rooms.pt's stored "
                    "K10KR D/S (exact equality)",
            "D_bit_equal": bool(np.array_equal(D_mine, D272)),
            "S_bit_equal": bool(np.array_equal(S_mine, S272)),
            "e272_rooms_md5": md5of(CKPT_DIR / ROOMS272_CK),
            "pass": bool(np.array_equal(D_mine, D272)
                         and np.array_equal(S_mine, S272)
                         and int(rooms272["model"]["K10KR"]["k"])
                         == LADDER[0][0]
                         and list(rooms272["model"]["K10KR"]["seeds"])
                         == [LADDER[0][1], LADDER[0][2]]),
        }
        del rooms272
    else:
        G_ROOM10KR = {
            "form": "SMOKE: the room shares the seed pair (27215/27216) at "
                    "smoke k — no committed record at this k; the bit-bind "
                    "is VACUOUS (explicit pass, disclosed)",
            "pass": True, "vacuous": True,
        }
        del rooms272
    assert G_ROOM10KR["pass"], f"K10KR room bind failed: {G_ROOM10KR}"
    metrics["gates"]["G_ROOM10KR"] = G_ROOM10KR
    log(f"P1 G_ROOM10KR: the vehicle's room "
        f"{('bit-identical to e272_rooms.pt (D/S exact)' if not SMOKE else 'SMOKE-vacuous')}: "
        f"PASS")

    rooms_ck = save_ckpt(
        "e284_rooms",
        {room_name: {"D_int8": rooms.rooms[room_name].D.astype(np.int8),
                     "S": rooms.rooms[room_name].S,
                     "k": LADDER[0][0], "seeds": [LADDER[0][1], LADDER[0][2]]}},
        {"desc": "e284's vehicle room (= e278/e280's): e272's committed "
                 "K10KR room (seeds 27215/27216), rebuilt + bit-gated vs "
                 "e272_rooms.pt",
         "ladder": [LADDER[0][0]], "n": N, "span_rank": rooms.r_span,
         "cert": {kk: vv for kk, vv in cert["per_rung"][room_name].items()
                  if not isinstance(vv, list)}})
    metrics["rooms"] = {
        "vehicle": {"k": LADDER[0][0], "name": room_name,
                    "seeds": [LADDER[0][1], LADDER[0][2]],
                    "k_fraction_of_N": LADDER[0][0] / N,
                    "bit_bound_to": f"runs/checkpoints/{ROOMS272_CK} "
                                    "(e272's committed K10KR room)"},
        "cert_probes_seed": E261.CERT_SEED,
        "certification": cert,
        "vmap_source": f"runs/checkpoints/{VMAP_CK} (e258's committed "
                       "2.74M v-map; LOADED, not re-run)",
        "span_source": f"runs/checkpoints/{SPAN_CK} (e246's committed "
                       f"LATE span; rank {rooms.r_span})",
        "checkpoint": rooms_ck,
    }
    log(f"P1 THE VEHICLE ROOM: {room_name} (k={LADDER[0][0]}): BUILT + "
        f"CERTIFIED + BIT-BOUND")
    write_partial("P1 the vehicle room built (parents bound + v-map loaded "
                  "+ span loaded + certification + bit-bind)")

    # ---- G_LR_BIND: the lr/momentum provenance bind (e280's record) -----
    lr_stable_runtime = SGD_STABLE_FACTOR * float(json.loads(
        E273_LRCAL.read_text(encoding="utf-8"))["lr_sgd"])
    G_LR_BIND = {
        "form": "the SGD-M lr := e273's STABLE RIDER POINT (e280's "
                "committed class) — LR_STABLE = x0.01 x LR_SGD_matched, "
                "re-derived at runtime from the md5-bound "
                "runs/e273/lr_calibration.json and asserted == the frozen "
                "literal; the optimizer hyperparameters match e273's "
                "stable rider / e280's record EXACTLY (momentum 0.9, wd "
                "0.0); schedule LR_STABLE x cosine_lr(s-1, 1000), the "
                "paired lr for BOTH streams and BOTH buffers",
        "lr_sgd_record": E273_LR_SGD,
        "stable_factor": SGD_STABLE_FACTOR,
        "lr_stable": LR_STABLE,
        "lr_stable_runtime": lr_stable_runtime,
        "momentum": SGD_MOMENTUM,
        "wd": SGD_WD,
        "matched_lr_diverges": "e273's committed record: the SGDM barrel "
                               "at LR_SGD_matched collapsed (0/nan); the "
                               "x0.01 stable rider is the family's "
                               "committed stable class",
        "undermatch_disclosed": "inherited from e280 (its central "
                                "disclosure): the s1 applied in-room L2 is "
                                "~x0.01 of AdamW-matched — the motel "
                                "geometry question is lr-scale-robust "
                                "(e280's committed funnel ran at exactly "
                                "this lr)",
        "pass": bool(abs(lr_stable_runtime - LR_STABLE) < 1e-15
                     and float(LR_STABLE) == 0.21738574801453703
                     and SGD_MOMENTUM == 0.9 and SGD_WD == 0.0),
    }
    assert G_LR_BIND["pass"], f"lr bind failed: {G_LR_BIND}"
    metrics["gates"]["G_LR_BIND"] = G_LR_BIND
    log(f"P1 G_LR_BIND: LR_STABLE = {SGD_STABLE_FACTOR} x {E273_LR_SGD} = "
        f"{LR_STABLE!r} (runtime {lr_stable_runtime!r}); momentum "
        f"{SGD_MOMENTUM}, wd {SGD_WD}: PASS")
    write_partial("P1b G_LR_BIND PASSED")

    # ================= P2-P3: THE TWO ARMS ==============================
    arms_rec: dict = {}

    def read_arm_cells(sd: dict) -> tuple:
        net_ = G1.evl_load(sd)
        cells_ = {"gm12": G1.battery_cell(net_, gm12_ids, zid)["mean_pz"],
                  "g0": G1.battery_cell(net_, g0_ids, zid)["mean_pz"],
                  "gp12": G1.battery_cell(net_, bat_ids[12], zid)["mean_pz"],
                  "ce_r": G1.ce_fixed_cpu(net_, *r_eval_xy)}
        return net_, cells_

    for arm in ARMS:
        if arm != ARMS[0]:
            E261.burst_cooldown(f"{ARMS[0]} -> {arm}")
        log("=" * 78)
        log(f"ARM-{arm} — {ARM_DESC[arm]}")
        inst = chunked_missile_buffers(
            f"{arm}-missile", G1.evl_load(base_sd), rooms, base_flat_np,
            inst_x, inst_mask, anchor_full, train_ids, g0_ids, gm12_ids,
            r_eval_xy, zid, ROOM_MODE,
            CKPT_DIR / (f"smoke_e284_{arm}_inst_resume.pt" if SMOKE
                        else f"e284_{arm}_inst_resume.pt"), dev, arm)
        sd_a = inst["sd"]
        net_a, cells_a = read_arm_cells(sd_a)
        d_a = flat_params_cpu(net_a)
        load_a = rooms.displacement_loads(d_a, ROOM_MODE)
        del net_a
        led_kept = [v["kept_frac"] for v in inst["ledger"].values()]
        ce_c = [v["ce"] for v in inst["corpus_ledger"].values()]
        gn_c = [v["gn_clipped"] for v in inst["corpus_ledger"].values()]
        prim, prim_fr, prim_cum = primary_read(inst["disp_ledger"])
        arms_rec[arm] = {
            "desc": ARM_DESC[arm], "install": {
                "traj": inst["traj"], "ledger": inst["ledger"],
                "corpus_ledger": inst["corpus_ledger"],
                "orth_ledger": inst["orth_ledger"],
                "orth_max_rel_err": inst["orth_max"],
                "corpus_ce_median": med(ce_c),
                "corpus_gn_clipped_median": med(gn_c),
                "disp_ledger": inst["disp_ledger"],
                "buf_ledger": inst["buf_ledger"],
                "primary_median_t100_400": prim,
                "primary_interval_fracs_t100_400": prim_fr,
                "primary_cumulative_fracs": prim_cum,
                "ledger_kept_frac_median": med(led_kept),
                "chunk_table": inst["chunk_table"],
                "steps": E261.INST_STEPS,
                "opt_steps": 2 * E261.INST_STEPS,
                "post_cells": cells_a, "displacement_loads": load_a,
                "bufsep": inst["bufsep"],
                "resumed_final": bool(inst.get("resumed_final", False)),
                "checkpoint": save_ckpt(
                    f"e284_{arm}_post", sd_a,
                    {"desc": f"e284 ARM-{arm} post-install state (s400): "
                             f"{ARM_DESC[arm][:110]}...",
                     "arm": arm, "install_seed": E261.FRESH_GEN,
                     "corpus_gen_seed": CORPUS_GEN_SEED,
                     "base": f"runs/checkpoints/{BASE_CK}",
                     "rooms": "runs/checkpoints/e284_rooms.pt",
                     "lr_stable": LR_STABLE,
                     "momentum": SGD_MOMENTUM, "wd": SGD_WD}),
            },
        }
        log(f"ARM-{arm} DONE: post g0 {cells_a['g0']:.7f} g-12 "
            f"{cells_a['gm12']:.7f} CE_R {cells_a['ce_r']:.4f} | PRIMARY "
            f"(in-room med t100-400) {prim if prim is not None else float('nan'):.4f} "
            f"(fracs {[None if x is None else round(x, 4) for x in prim_fr]}) "
            f"| corpus CE med {med(ce_c):.4f} |g_corp| med {med(gn_c):.3f} | "
            f"kept med {med(led_kept):.4f} | ORTH max {inst['orth_max']:.2e} "
            f"(bar {ORTH_BAR:.0e}) | isolation checks "
            f"{inst['bufsep']['isolation_checks']}, violations "
            f"{inst['bufsep']['isolation_violations']}")
        metrics["arms"] = arms_rec
        write_partial(f"ARM-{arm} complete (primary read + ledgers)")

    # ---- the draw-integrity texture check (non-halting) ------------------
    first_inst_ce = {a: arms_rec[a]["install"]["ledger"].get(
        "1", arms_rec[a]["install"]["ledger"].get(1, {})).get("ce")
        for a in ARMS}
    first_corp_ce = {a: arms_rec[a]["install"]["corpus_ledger"].get(
        "1", arms_rec[a]["install"]["corpus_ledger"].get(1, {})).get("ce")
        for a in ARMS}
    draw_ok = (all(v is not None and abs(v - first_inst_ce[ARMS[0]]) < 1e-9
                   for v in first_inst_ce.values())
               and all(v is not None
                       and abs(v - first_corp_ce[ARMS[0]]) < 1e-9
                       for v in first_corp_ce.values()))
    log(f"draw-integrity (non-halting): first-batch install + corpus CE "
        f"identical across arms = {draw_ok} (inst {first_inst_ce}; corp "
        f"{first_corp_ce})")

    # ================= P4: the instantiated gates ========================
    # G_MISSILE_ORTH — INSTANTIATED this time (e280's omission corrected)
    orth_maxes = {a: arms_rec[a]["install"]["orth_max_rel_err"]
                  for a in ARMS}
    G_MISSILE_ORTH = {
        "form": "the stepped corpus gradient is ENTIRELY ORTHOGONAL to the "
                "room on BOTH arms: max over ALL corpus steps of ||P_room "
                "g_perp|| / ||g_perp|| < 1e-6 (checked EVERY corpus step; "
                "the SRCT projector is exact, so any drift above fp64 "
                "roundoff is an implementation bug). INSTANTIATED in the "
                "gates dict this time — e280's bookkeeping omission "
                "(disclosed at its fold) corrected",
        "orth_max_rel_err_per_arm": orth_maxes,
        "bar": ORTH_BAR,
        "n_steps_checked_per_arm": E261.INST_STEPS,
        "norm_ratio_first": {a: arms_rec[a]["install"]["orth_ledger"].get(
            "1", arms_rec[a]["install"]["orth_ledger"].get(1, {})
        ).get("norm_ratio") for a in ARMS},
        "pass": bool(all(v is not None and v < ORTH_BAR
                         for v in orth_maxes.values())),
    }
    assert G_MISSILE_ORTH["pass"], \
        f"missile orthogonality gate FAILED: {G_MISSILE_ORTH}"
    metrics["gates"]["G_MISSILE_ORTH"] = G_MISSILE_ORTH

    # G_BUFSEP — the buffer separation's machine verification
    sep = arms_rec["SEP"]["install"]
    bufC_fr = [b["bufC_inroom_frac"] for b in sep["buf_ledger"]
               if b.get("bufC_inroom_frac") is not None]
    bufI_fr = [b["bufI_inroom_frac"] for b in sep["buf_ledger"]
               if b.get("bufI_inroom_frac") is not None]
    sha_shared_fr = [b["shared_buf_inroom_frac"]
                     for b in arms_rec["SHA"]["install"]["buf_ledger"]
                     if b.get("shared_buf_inroom_frac") is not None]
    G_BUFSEP = {
        "form": "the buffer separation's machine verification: (i) "
                "ISOLATION — opt_I's momentum buffers snapshotted before "
                "EVERY corpus opt_C.step() and compared bitwise after "
                "(and symmetrically for opt_C around every install "
                "opt_I.step()); the install buffer never receives corpus "
                "content and vice versa — any mismatch HALTS at the step; "
                "(ii) COMPOSITION — per milestone, ||P_room buf_C|| / "
                "||buf_C|| < 1e-4 (the corpus buffer carries no in-room "
                "content; SGD momentum is linear in its grads, every "
                "corpus grad orthogonal to < 1e-6, so the floor is the "
                "fp32 accumulation noise ~1e-7-1e-6) AND ||P_room buf_I||"
                "/||buf_I|| > 0.99 (the install buffer IS the in-room "
                "stream). The SHARED arm's single-buffer in-room fraction "
                "is a READ (the designed mixture), never gated",
        "isolation_checks": sep["bufsep"]["isolation_checks"],
        "isolation_violations": sep["bufsep"]["isolation_violations"],
        "bufC_inroom_frac_max": max(bufC_fr) if bufC_fr else None,
        "bufC_inroom_frac_last": bufC_fr[-1] if bufC_fr else None,
        "bufC_bar": BUFSEP_ORTH_BAR,
        "bufI_inroom_frac_min": min(bufI_fr) if bufI_fr else None,
        "bufI_inroom_frac_last": bufI_fr[-1] if bufI_fr else None,
        "bufI_bar": BUFSEP_INROOM_BAR,
        "shared_arm_buf_inroom_frac_read": {
            "first": sha_shared_fr[0] if sha_shared_fr else None,
            "last": sha_shared_fr[-1] if sha_shared_fr else None,
            "note": "the SHARED twin's single buffer (the designed "
                    "mixture — the discrimination context, a READ)"},
        "pass": None,           # computed explicitly below (a hard gate)
    }
    G_BUFSEP["pass"] = bool(
        sep["bufsep"]["isolation_violations"] == 0
        and sep["bufsep"]["isolation_checks"] == 2 * E261.INST_STEPS
        and bufC_fr and max(bufC_fr) < BUFSEP_ORTH_BAR
        and bufI_fr and min(bufI_fr) > BUFSEP_INROOM_BAR)
    assert G_BUFSEP["pass"], f"buffer separation gate FAILED: {G_BUFSEP}"
    metrics["gates"]["G_BUFSEP"] = G_BUFSEP
    log(f"P4 G_MISSILE_ORTH PASS (per-arm max "
        f"{ {a: f'{v:.1e}' for a, v in orth_maxes.items()} }); "
        f"G_BUFSEP PASS ({G_BUFSEP['isolation_checks']} isolation checks, "
        f"0 violations; bufC in-room max "
        f"{max(bufC_fr) if bufC_fr else float('nan'):.2e} < "
        f"{BUFSEP_ORTH_BAR:.0e}; bufI in-room min "
        f"{min(bufI_fr) if bufI_fr else float('nan'):.4f} > "
        f"{BUFSEP_INROOM_BAR}; shared twin's buffer in-room last "
        f"{sha_shared_fr[-1] if sha_shared_fr else float('nan'):.3f})")
    write_partial("P4 gates complete (ORTH instantiated + BUFSEP)")

    # ================= P5: ADJUDICATION (the frozen bars) ================
    prim_sep = arms_rec["SEP"]["install"]["primary_median_t100_400"]
    prim_sha = arms_rec["SHA"]["install"]["primary_median_t100_400"]
    post = {a: arms_rec[a]["install"]["post_cells"]["g0"] for a in ARMS}

    hard = dict(metrics["gates"])
    gates_pass = bool(all(g.get("pass") for g in hard.values()))

    if not gates_pass:
        failed = [k for k, g in hard.items() if not g.get("pass")]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a hard gate failed — nothing adjudicated; the record is "
                  "complete for the autopsy")
    elif prim_sha is None or prim_sep is None:
        verdict = "TEXTURE (PRIMARY READ MISSING)"
        clause = ("a milestone displacement read is missing — nothing "
                  "adjudicated; the ledgers are verbatim")
    elif prim_sha < TWIN_PRECEDENT_BAR:
        verdict = "MIXED (CONTROL-TWIN-OFF-PRECEDENT)"
        clause = (f"the SHARED twin failed to funnel this session (primary "
                  f"{prim_sha:.4f} < {TWIN_PRECEDENT_BAR}; e280's committed "
                  f"median {E280_MISSILE_MEDIAN:.4f}, intervals "
                  f"0.4152-0.4926) — the twins disagree with the committed "
                  f"precedent, so the pair's difference carries no "
                  f"ownership content; the separate arm's read "
                  f"({prim_sep:.4f}) is co-reported verbatim; everything "
                  f"verbatim, no inflation")
    elif prim_sep < MOMENTUM_BAR:
        verdict = "MOMENTUM-OWNED"
        clause = (f"the separate-buffer arm's in-room share "
                  f"{prim_sep:.4f} < {MOMENTUM_BAR} (vs the shared twin's "
                  f"{prim_sha:.4f}, replicating e280's committed "
                  f"{E280_MISSILE_MEDIAN:.4f}) — the re-aimer is "
                  f"buffer-sharing; THE MOTEL IS A MOMENTUM-SHARING "
                  f"ARCHITECTURE; orthogonality is preservable by buffer "
                  f"separation. The funnel's arithmetic: the shared buffer "
                  f"carries COHERENT in-room install content (per-step "
                  f"magnitude ~kept 0.06) into the corpus steps while the "
                  f"~1.0-magnitude orthogonal corpus content cancels "
                  f"across steps — a realized SHARE of ~0.45 from a "
                  f"per-step magnitude of ~0.06")
    elif prim_sep >= LANDSCAPE_BAR:
        verdict = "LANDSCAPE-OWNED"
        clause = (f"the separate-buffer arm still {prim_sep:.4f} in-room "
                  f"(>= {LANDSCAPE_BAR}; shared twin {prim_sha:.4f}) — the "
                  f"landscape itself bends applied steps toward the room; "
                  f"the motel is the space's, not any optimizer state's. "
                  f"NOTE: under exact linear momentum arithmetic the "
                  f"separate corpus buffer is orthogonal by construction "
                  f"— a funnel here means a channel OUTSIDE the gradient "
                  f"composition moves the corpus displacement in-room "
                  f"(the read that would rewrite the motel's story)")
    else:
        verdict = "MIXED (BETWEEN-BARS)"
        clause = (f"the separate-buffer arm's in-room share "
                  f"{prim_sep:.4f} sits between the bars "
                  f"({MOMENTUM_BAR}-{LANDSCAPE_BAR}; shared twin "
                  f"{prim_sha:.4f}) — everything verbatim, all reads, no "
                  f"inflation")

    log("=" * 78)
    log(f"E284 VERDICT: {verdict}")
    log(f"  SEP primary {prim_sep} | SHA primary {prim_sha} "
        f"(e280 committed {E280_MISSILE_MEDIAN}; e278 AdamW band "
        f"{E278_MISSILE_INT_BAND})")
    log(f"  posts: SEP {post['SEP']:.7f}, SHA {post['SHA']:.7f} "
        f"(e280 cites: missile {E280_MISSILE_POST_G0:.7f}, serial S10K "
        f"{E280_S10K_SERIAL_POST:.6f})")
    log(f"  {clause}")
    log("=" * 78)

    # the survival secondary (never the primary)
    survival = {
        a: {"post_g0": post[a],
            "vs_cited_serial": post[a] / max(E280_S10K_SERIAL_POST, 1e-9),
            "spared": bool(post[a] >= SURVIVE_FRAC
                           * E280_S10K_SERIAL_POST),
            "vs_e280_missile_cite": post[a] / max(E280_MISSILE_POST_G0,
                                                  1e-9),
            "traj_g0": {t["step"]: t["g0_pz"]
                        for t in arms_rec[a]["install"]["traj"]}}
        for a in ARMS}
    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "secondary_verbatim": REGISTERED["secondary_verbatim"],
        "composite_order": "TEXTURE -> MIXED (CONTROL-TWIN-OFF-PRECEDENT) "
                           "-> MOMENTUM-OWNED (< 0.20) -> LANDSCAPE-OWNED "
                           "(>= 0.35) -> MIXED (BETWEEN-BARS) — frozen at "
                           "birth",
        "gates_pass": gates_pass,
        "reads": {
            "primary_sep": prim_sep,
            "primary_sha": prim_sha,
            "primary_sep_interval_fracs":
                arms_rec["SEP"]["install"]["primary_interval_fracs_t100_400"],
            "primary_sha_interval_fracs":
                arms_rec["SHA"]["install"]["primary_interval_fracs_t100_400"],
            "primary_sep_cumulative":
                arms_rec["SEP"]["install"]["primary_cumulative_fracs"],
            "primary_sha_cumulative":
                arms_rec["SHA"]["install"]["primary_cumulative_fracs"],
            "posts": post,
            "disp_ledgers": {a: arms_rec[a]["install"]["disp_ledger"]
                             for a in ARMS},
            "buf_ledgers": {a: arms_rec[a]["install"]["buf_ledger"]
                            for a in ARMS},
            "committed_cites": {
                "e278_adamw_missile_band": E278_MISSILE_INT_BAND,
                "e278_adamw_missile_post": E278_MISSILE_POST_G0,
                "e280_sgd_missile_median": E280_MISSILE_MEDIAN,
                "e280_sgd_missile_post": E280_MISSILE_POST_G0,
                "e280_s10k_serial_post": E280_S10K_SERIAL_POST},
            "volume_overlap_sqrt_k_over_n":
                math.sqrt(LADDER[0][0] / 2739072),
        },
        "survival_secondary": {
            "form": f"SPARED := post g0 >= {SURVIVE_FRAC}x e280's cited "
                    f"serial SGD-M 10k rung ({E280_S10K_SERIAL_POST}); "
                    "never the primary",
            "arms": survival,
            "prediction_P-e284s": "NOT SPARED for both arms (e273's "
                                  "two-body account: the parameters still "
                                  "collide)",
        },
        "draw_integrity_first_batch": {"pass": bool(draw_ok),
                                       "install_ce": first_inst_ce,
                                       "corpus_ce": first_corp_ce},
        "verdict": verdict,
        "clause": clause,
        "smoke_stamp": ("SMOKE — nothing adjudicated" if SMOKE else None),
    }
    write_partial("P5 ADJUDICATED (the frozen bars)")

    # ================= P6: honesty + provenance + close ==================
    metrics["honesty"] = {
        "intervention_not_logits": (
            "the two arms share bit-identical install streams (one "
            "generator, seed 24314, one draw order), bit-identical corpus "
            "batches (the ONE fresh stream seed 28401, drawn identically "
            "by each arm), the same room (bit-gated to e272's committed "
            "K10KR room), the same lr schedule, the same optimizer "
            "hyperparameters — the ONLY delta is the momentum-buffer "
            "topology (TWO buffers vs ONE). The rooms' eigenstructures "
            "are identical across arms BY CONSTRUCTION."),
        "the_buffer_disclosure": (
            "the separation acts on OPTIMIZER STATE only — the corpus "
            "gradients are orthogonalized identically in both arms "
            "(verified every step); the install gradients are projected "
            "onto the room identically in both arms; the parameters are "
            "shared within each arm exactly as in e280 (the two-body "
            "medium is untouched). The isolation is machine-checked every "
            "step (bitwise buffer snapshots) and the composition "
            "machine-read per milestone (the corpus buffer at the fp "
            "floor, the install buffer in-room)"),
        "n_and_scope": ("n=1 per arm, one lineage, one session (the "
                        "g-series standing lottery caveat carried "
                        "verbatim); the arms' DIFFERENCE is the registered "
                        "object; nothing guaranteed"),
        "loads_measured_not_nominal": (
            "every read is measured: per-step kept fraction, v-excess "
            "pre/post, in-span pre/applied, the per-milestone realized "
            "displacement projections (interval + cumulative), the "
            "buffer-composition ledger, the orthogonality ledger, the "
            "corpus CE + clipped-grad ledgers — never nominal"),
        "nothing_guaranteed": ("the dispatch's own caveat carried: no "
                               "outcome was promised; the bars cover all "
                               "branches and the trajectories are reported "
                               "verbatim regardless"),
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "machinery_imported_from": str(E261.__file__),
        "checkpoints": {
            "base": f"runs/checkpoints/{BASE_CK}",
            "reference_root": {"file": f"runs/checkpoints/{ROOT_CK}",
                               "flat_md5": G_ROOT["flat_md5"]},
            "span_ledger": f"runs/checkpoints/{SPAN_CK}",
            "vmap": f"runs/checkpoints/{VMAP_CK}",
            "e272_rooms": f"runs/checkpoints/{ROOMS272_CK}",
            "rooms": rooms_ck,
            "post_states": {a: arms_rec[a]["install"]["checkpoint"]
                            for a in ARMS},
        },
        "machinery": {
            "missile_buffers": "THIS file's chunked_missile_buffers: "
                               "e280's chunked_missile_sgd (= e278's "
                               "chunked_install_threenull MISSILE form "
                               "under SGD-M) with the optimizer topology "
                               "parameterized + the buffer isolation/"
                               "composition ledgers; the committed "
                               "lab/e261_rank_ladder.py, "
                               "lab/e278_three_null.py and "
                               "lab/e280_sgd_ladder.py are NOT modified",
            "missile_projection": "orthogonalize_grads ported VERBATIM "
                                  "from e278/e280: g_perp = g - P_room(g) "
                                  "via e261's SRCT projector (exact, fp64, "
                                  "CPU); verified every corpus step",
            "cons": "NONE (registered deviation — the bars read the WRITE "
                    "and the DISPLACEMENT only; T259/e281)",
        },
        "eval": {"device": "cpu fp32 probes / cuda fp32 training / cpu "
                           "fp64 dense projections",
                 "threads": torch.get_num_threads()},
        "thermal_envelope": _envelope_summary(),
        "versions": {"torch": torch.__version__,
                     "numpy": np.__version__,
                     "scipy": __import__("scipy").__version__,
                     "matplotlib": matplotlib.__version__},
    }

    # ================= P7: figures ======================================
    make_buffer_plot(RD, arms_rec, prim_sep, prim_sha, verdict, clause,
                     thermal_log)
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    metrics["outputs"] = [str(RD / "metrics.json"),
                          str(RD / "e284_buffer_missile.png"),
                          str(RD / "REPORT.md")]
    write_partial("P7 DONE (honesty + provenance + figures)")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots
def make_buffer_plot(rd, arms_rec, prim_sep, prim_sha, verdict, clause,
                     thermal_log):
    """THE CELL'S HEADLINE FIGURE: the two arms' in-room walks overlaid
    with e278's AdamW band and e280's committed SGD-M points (the primary
    panel), the write trajectories, the buffer compositions, the
    orthogonality, the corpus CE, the envelope."""
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.6))
    cols = {"SEP": "tab:purple", "SHA": "tab:red"}

    def _f4(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"

    # (0,0) THE PRIMARY PANEL — the two arms' in-room walks + the bands
    ax = axes[0, 0]
    for a in ARMS:
        dl = [d for d in arms_rec[a]["install"]["disp_ledger"]
              if d.get("interval_in_room_frac") is not None]
        ax.plot([d["step"] for d in dl],
                [d["interval_in_room_frac"] for d in dl],
                "o-", lw=1.8, ms=5, color=cols[a],
                label=f"{a} (median "
                f"{_f4(arms_rec[a]['install']['primary_median_t100_400'])})")
    ax.axhspan(E278_MISSILE_INT_BAND[0], E278_MISSILE_INT_BAND[1],
               color="darkorange", alpha=0.14,
               label=f"e278 AdamW missile band "
               f"{E278_MISSILE_INT_BAND[0]:.3f}-{E278_MISSILE_INT_BAND[1]:.3f}")
    e280_pts = {100: 0.41925071220790866, 200: 0.4926210274052686,
                300: 0.45529691870685485, 400: 0.41515870106417924}
    ax.plot(list(e280_pts), list(e280_pts.values()), "s--", lw=1.2, ms=4.5,
            color="tab:gray", alpha=0.85,
            label=f"e280 SGD-M shared missile (median "
                  f"{E280_MISSILE_MEDIAN:.4f})")
    ax.axhline(MOMENTUM_BAR, color="blue", ls="--", lw=1.3,
               label=f"MOMENTUM-OWNED bar < {MOMENTUM_BAR}")
    ax.axhline(LANDSCAPE_BAR, color="crimson", ls="--", lw=1.3,
               label=f"LANDSCAPE-OWNED bar >= {LANDSCAPE_BAR}")
    ax.axhline(math.sqrt(10000 / 2739072), color="gray", ls=":",
               lw=1.1, label=f"volume overlap sqrt(k/N) = "
               f"{math.sqrt(10000 / 2739072):.4f}")
    ax.set_xlabel("milestone s (interval end)")
    ax.set_ylabel("||P_room v|| / ||v|| (the corpus displacement)")
    ax.set_title("THE PRIMARY READ — where each arm's corpus stream "
                 "actually walked", fontsize=9.5)
    ax.legend(fontsize=6.6)
    ax.grid(alpha=0.25)

    # (0,1) THE WRITE TRAJECTORIES (the survival secondary)
    ax = axes[0, 1]
    for a in ARMS:
        tr = arms_rec[a]["install"]["traj"]
        ax.plot([t["step"] for t in tr], [max(t["g0_pz"], 1e-7) for t in tr],
                "o-", lw=1.6, ms=4, color=cols[a], label=f"{a} write")
    ax.axhline(E280_S10K_SERIAL_POST, color="tab:blue", ls=":", lw=1.2,
               label=f"e280 serial SGD-M 10k {E280_S10K_SERIAL_POST:.4f}")
    ax.axhline(E280_MISSILE_POST_G0, color="tab:gray", ls=":", lw=1.2,
               label=f"e280 shared missile post {E280_MISSILE_POST_G0:.6f}")
    ax.axhline(E278_MISSILE_POST_G0, color="darkorange", ls=":", lw=1.0,
               label=f"e278 AdamW missile post {E278_MISSILE_POST_G0:.6f}")
    ax.set_yscale("log")
    ax.set_xlabel("install step s (the corpus step follows each, 1:1)")
    ax.set_ylabel("g0 battery (mean p(Z), log)")
    ax.set_title("THE WRITE's FATE (the survival secondary — never the "
                 "primary)", fontsize=9.5)
    ax.legend(fontsize=6.8)
    ax.grid(alpha=0.25, which="both")

    # (0,2) THE CUMULATIVE IN-ROOM WALKS
    ax = axes[0, 2]
    for a in ARMS:
        dl = [d for d in arms_rec[a]["install"]["disp_ledger"]
              if d.get("cum_in_room_frac") is not None]
        ax.plot([d["step"] for d in dl],
                [d["cum_in_room_frac"] for d in dl],
                "o-", lw=1.6, ms=4.5, color=cols[a], label=f"{a} cumulative")
    ax.axhline(LANDSCAPE_BAR, color="crimson", ls="--", lw=1.0,
               label=f"the {LANDSCAPE_BAR} motel bar")
    ax.set_xlabel("milestone s")
    ax.set_ylabel("cumulative ||P_room v|| / ||v||")
    ax.set_title("THE CUMULATIVE WALK (where each stream's total "
                 "displacement sits)", fontsize=9.5)
    ax.legend(fontsize=7.0)
    ax.grid(alpha=0.25)

    # (1,0) THE BUFFER COMPOSITIONS (G_BUFSEP's read)
    ax = axes[1, 0]
    sep_bl = [b for b in arms_rec["SEP"]["install"]["buf_ledger"]
              if b.get("bufC_inroom_frac") is not None]
    ax.semilogy([b["step"] for b in sep_bl],
                [max(b["bufC_inroom_frac"], 1e-18) for b in sep_bl],
                "o-", lw=1.5, ms=4.5, color=cols["SEP"],
                label="SEP buf_C in-room frac (the corpus buffer)")
    ax.axhline(BUFSEP_ORTH_BAR, color="crimson", ls="--", lw=1.2,
               label=f"G_BUFSEP bar {BUFSEP_ORTH_BAR:.0e}")
    ax2 = ax.twinx()
    ax2.plot([b["step"] for b in sep_bl],
             [b["bufI_inroom_frac"] for b in sep_bl],
             "s--", lw=1.2, ms=3.5, color="tab:blue",
             label="SEP buf_I in-room frac (right)")
    sha_bl = [b for b in arms_rec["SHA"]["install"]["buf_ledger"]
              if b.get("shared_buf_inroom_frac") is not None]
    ax2.plot([b["step"] for b in sha_bl],
             [b["shared_buf_inroom_frac"] for b in sha_bl],
             "^--", lw=1.2, ms=3.5, color=cols["SHA"],
             label="SHA shared buffer in-room frac (right)")
    ax2.axhline(BUFSEP_INROOM_BAR, color="blue", ls=":", lw=1.0)
    ax2.set_ylabel("buf_I / shared-buffer in-room fraction", fontsize=8.5)
    ax2.set_ylim(0.0, 1.05)
    ax.set_xlabel("milestone s")
    ax.set_ylabel("buf_C in-room fraction (log)", color=cols["SEP"])
    ax.set_title("THE BUFFER COMPOSITIONS (the corpus buffer at the fp "
                 "floor; the install buffer in-room; the twin's mixture)",
                 fontsize=9.0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.6)
    ax.grid(alpha=0.25, which="both")

    # (1,1) THE ORTHOGONALITY (both arms)
    ax = axes[1, 1]
    for a in ARMS:
        ol = arms_rec[a]["install"]["orth_ledger"]
        steps = sorted(int(k) for k in ol.keys())
        rels = [max(ol[str(k)]["orth_rel_err"], 1e-18) if str(k) in ol
                else max(ol[k]["orth_rel_err"], 1e-18) for k in steps]
        ax.semilogy(steps, rels, "s-", lw=1.3, ms=3.5, color=cols[a],
                    label=f"{a} (max "
                    f"{arms_rec[a]['install']['orth_max_rel_err']:.1e})")
    ax.axhline(ORTH_BAR, color="crimson", ls="--", lw=1.4,
               label=f"the gate bar {ORTH_BAR:.0e} (INSTANTIATED)")
    ax.set_xlabel("install step s")
    ax.set_ylabel("orthogonality rel err (log)")
    ax.set_title("THE MISSILE'S ORTHOGONALITY (checked EVERY corpus step, "
                 "both arms)", fontsize=9.5)
    ax.legend(fontsize=7.0)
    ax.grid(alpha=0.25, which="both")

    # (1,2) THE CORPUS CE + THE THERMAL ENVELOPE
    ax = axes[1, 2]
    for a in ARMS:
        cl = arms_rec[a]["install"]["corpus_ledger"]
        steps = sorted(int(k) for k in cl.keys())
        ax.plot(steps, [cl[str(k)]["ce"] if str(k) in cl else cl[k]["ce"]
                        for k in steps], "s-", lw=1.3, ms=3.5,
                 color=cols[a], label=f"{a} corpus CE")
    ax.set_xlabel("install step s")
    ax.set_ylabel("corpus batch CE (nats)")
    ax3 = ax.twinx()
    if thermal_log:
        ax3.plot([r["t"] for r in thermal_log],
                 [r["temp"] for r in thermal_log],
                 "-", lw=0.8, color="dimgray", alpha=0.65)
    ax3.axhline(E261.TEMP_EARLY_END, color="crimson", ls=":", lw=1.0)
    ax3.axhline(E261.TEMP_HARD, color="crimson", ls="--", lw=1.2)
    ax3.set_ylabel("GPU temp (C)", fontsize=8.5)
    mx = max((r["temp"] for r in thermal_log), default=float("nan"))
    ax.set_title(f"THE CORPUS CE (semantic, identical draws) + the "
                 f"envelope (max {mx:.1f}C)", fontsize=9.0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.0)
    ax.grid(alpha=0.25)

    fig.suptitle(f"E284 — THE SEPARATE-BUFFER MISSILE (the motel's "
                 f"ownership test) -> {verdict}", fontsize=11)
    fig.text(0.5, 0.005, textwrap.fill(clause, 170), ha="center",
             fontsize=7.2, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(rd / "e284_buffer_missile.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
