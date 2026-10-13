"""E290 — THE COUPLING-CONSTANT LADDER — measuring the program's
load-bearing constant: the orthogonal-drift threshold at which an
established write's read dies. This docstring carries the registered
question + arms + bars VERBATIM from the dispatch letter, committed at
birth BEFORE any compute. Adjudicate against exactly this; no bar shopping.

THE QUESTION (dispatch verbatim): e285 (the sanctuary) found a 20.1%-of-
write orthogonal drift killed the read 330x with the budget held; its
t100 datum (7.2% drift -> 200x down) bounds the threshold at <= 0.07x —
a ONE-DRAW bound (R68's critic: the monotonicity assumed, the threshold
could sit at 0.5%). The ladder measures it. NOTE e288 just landed
(ERROR-GATED-HOLDS — active maintenance works); the constant prices when
maintenance is NECESSARY (below the threshold, no anchor needed).

THE ARMS (dispatch verbatim): "THE ARMS (the established quiet-formed 10k
fact, loaded bit-exact; the corpus stream 1:1; SGD-M, separate buffers,
orthogonal projection — the sanctuary form VERBATIM, only the BUDGET
varying): four rungs at total-drift budgets {0.1x, 0.02x, 0.004x,
0.0008x} of the write's norm 9.1788 (the geometric ladder from e285's
0.5x anchor point; each rung = the equal-share cap form at the smaller
budget). Read post g0 at t100/200/300/400; the realized drift per
milestone (the verification); no maintenance (the pure passive curve at
each budget)."

FROZEN BARS (dispatch verbatim; survival ratio = post g0 at t400 / the
committed loaded baseline 0.26464763283729553; the threshold = the
largest budget whose arm holds >= 0.5x):
  - TIGHT-CONSTANT: "the 0.1x arm dies (< 0.5x) AND some lower rung
    holds — the threshold located in the ladder's span; report the
    bracket + the curve shape (graded vs step)."
  - NO-PASSIVE-THRESHOLD: "ALL rungs die (< 0.5x) even at 0.0008x — the
    coupling is effectively unbounded; NO passive drift is safe;
    maintenance is ALWAYS necessary for a moving organism (the strongest
    fragility statement; e288's controller is then the only
    preservation)."
  - ROBUST-FLOOR: "the 0.1x arm HOLDS (>= 0.5x) — contradicting e285's
    single-datum bound (the draw or the phase timing mattered); the
    threshold is looser than 0.07x after all."
  - MIXED: "non-monotone across rungs — everything verbatim."

OPERATIONALIZATIONS (frozen HERE before compute; they fix the clauses,
they do not move the bars):
  * THE PHASE := 400 CORPUS steps per rung, NO install, NO maintenance
    (e283/e285's convention freeze verbatim); milestones t =
    100/200/300/400 corpus steps; each rung a FRESH instance from the
    same loaded fact.
  * THE VEHICLE := the committed QUIET-FORMED 10k fact (e261's serial cut
    completed by e264; e261_K10K_inst_resume.pt), loaded BIT-EXACT and
    gated THREE ways (artifact md5/size/step/traj/ledger + flat-md5 +
    behavioral read vs the committed literals — G_FACTLOAD); the room :=
    the fact's OWN committed K10K room (seeds 26113/26114), rebuilt +
    bit-gated vs e264_rooms.pt (G_ROOMK10K).
  * THE RUNG FORM := e285's SANCTUARY arm VERBATIM, only BUDGET_FRAC
    varying: SGD-M momentum 0.9 wd 0.0 at LR_STABLE = 0.01 x
    21.7385748014537 = 0.21738574801453703 x cosine_lr(t-1,1000) (the
    nominal schedule, md5-bound provenance), TWO SGD instances over the
    same params — opt_C (the ONLY one stepped) + opt_F (NEVER stepped;
    bitwise isolation probe every step), backward -> clip 1.0 ->
    g_perp = g - P_room(g) (CPU fp64, verified EVERY step) -> the
    equal-share per-step lr cap cap_t = ((BUDGET - S_{t-1})/(n-t+1))/
    ||b_t|| with b_t = 0.9*buf_{t-1} + g_perp; lr_t =
    min(LR_STABLE x cosine, cap_t); S += realized ||step||;
    triangle-guaranteed S_400 <= BUDGET.
  * THE BUDGETS := {0.1, 0.02, 0.004, 0.0008} x ||fact - base|| =
    9.1788432658723 (the geometric 0.2-factor ladder down from e285's
    0.5x anchor); slack 1e-5 absolute on S + cum (fp32 accumulation,
    e285's disclosed form).
  * THE CORPUS STEP := e268's registered form VERBATIM: 48 windows = 16
    original-host anchors (the same 60-window bank) + 32 random corpus
    windows; full-window CE; draws (aj_c(16), rj_c(32)) from THIS cell's
    ONE fresh registered generator seed 29001 (the family's per-cell
    rule: .../28301/28401/28501/28801/HERE 29001); ALL FOUR rungs draw
    the IDENTICAL sequence (each rung its own generator instance at the
    same seed) — bit-identical corpus batches across rungs, the BUDGET
    the rungs' ONLY delta.
  * THE SURVIVAL RATIO := post g0(t) / 0.26464763283729553 (the
    committed loaded baseline, PRIMARY; the session's loaded read
    co-reported).
  * THE ADJUDICATION READ := each rung's t=400 endpoint (all gate
    verdicts at the same endpoints); full milestone trajectories +
    realized-drift ledgers co-reported verbatim.
  * COMPOSITE := TEXTURE (any hard-gate failure — nothing adjudicated)
    -> NO-PASSIVE-THRESHOLD (all four rungs die) -> ROBUST-FLOOR (the
    0.1x rung holds) -> TIGHT-CONSTANT (the 0.1x rung dies AND >= 1
    lower rung holds AND the holds-pattern is monotone — a suffix of the
    rung list) -> MIXED (non-monotone across rungs — everything
    verbatim).
  * THE THRESHOLD BRACKET := in budget terms (largest holding frac,
    smallest dying frac above it]; co-reported in realized-drift terms
    (each bracketing rung's realized cumulative drift at t400 as a
    fraction of the write's norm — the ladder's measured constant).
  * THE CURVE SHAPE (reported, never a bar): GRADED (survival scales
    progressively across rungs) vs STEP (the transition to survival
    lands within one rung gap) — classified from the adjacent-rung
    factors, numbers verbatim.
  * HARD GATES (a failure HALTS): {G_NAMEFREE, G_SPLICE, G_BATTERY,
    G_ANCHOR, G_INSTMASK, G_PARENTS, G_BASE, G_ROOT, G_VMBIND, G_SPANBIND,
    G_PROJ, G_ROOMK10K, G_FACTLOAD, G_CORPUSGEN, G_LR_BIND, G_ORTH,
    G_BUFSEP, G_BUDGET} — G_ORTH/G_BUFSEP/G_BUDGET instantiated PER RUNG
    (all four must pass).
  * G_ORTH := per rung, max over ALL corpus steps of ||P_room g_perp||/
    ||g_perp|| < 1e-6 (checked every step).
  * G_BUFSEP := per rung, (i) isolation: opt_F snapshotted bitwise
    before every opt_C.step() and compared after (any mismatch HALTS);
    (ii) composition: ||P_room buf_C||/||buf_C|| < 1e-4 per milestone.
  * G_BUDGET := per rung, S_400 <= BUDGET + 1e-5 AND realized cumulative
    displacement ||theta_400 - theta_fact|| <= BUDGET + 1e-5; the ledger
    discloses budget_norm, S_400, realized cum, usage fractions, the
    cap's binding count, the lr price.
  * NO CONS (T259/e281); all four post states CHECKPOINTED.
  * NO TWIN (cited + hard-bound instead): e285's same-session
    UNPROTECTED-TWIN (ratio 1.47e-05x) + e283's committed reference
    (3.42e-05x) are the death class's record; the ladder's question is
    the PASSIVE floor, not the unprotected contrast.

REGISTERED PREDICTIONS (registered at birth, before compute):
  - P-e290a (the tight constant, the dispatch's expected reading): the
    0.1x rung's realized cumulative drift will land ~4% of the write
    (e285's cancellation ratio: cum ~ 40% of budget) — inside the lethal
    band e285 already measured at t100 (7.2% -> 0.0051x) — so the 0.1x
    rung DIES; the threshold sits at or below the 0.02x rung's realized
    ~0.8%: TIGHT-CONSTANT with the bracket in-span, or
    NO-PASSIVE-THRESHOLD if even ~0.03% (the 0.0008x rung's realized)
    kills by t400.
  - P-e290b (R68's critic branch): NO-PASSIVE-THRESHOLD — the threshold
    sits under ~0.5% of the write and the read dies at every rung by
    t400; the coupling is effectively unbounded; maintenance is ALWAYS
    necessary (the strongest fragility statement; e288's controller the
    only preservation).
  - P-e290c (the robust branch): ROBUST-FLOOR — e285's kill was
    draw/phase-timing sensitive; the 0.1x rung holds >= 0.5x at t400 and
    the threshold is looser than 0.07x after all.

COMPUTE ENVELOPE: 2,739,072 params (inside the <=100M free tier); bursts
<= 175s (dispatch 180), per-step thermal polls at a 78C margin, 40s
cooldowns (dispatch 30-60), the 84C never-past line (dispatch 85), polls
persisted to runs/_envelope_log.jsonl tagged e290:<ARM>:<phase>; CPU fp64
dense projections (pocketfft workers 2); CPU threads 4; the four rungs
run SEQUENTIALLY with cooldowns between — NO concurrent GPU jobs.

Outputs: runs/e290/{metrics.json (PROGRESSIVE), e290_coupling_ladder.png,
REPORT.md (executor-written), run.log (gitignored)}; checkpoints
runs/checkpoints/e290_*.pt (gitignored; md5s in metrics). No NOTES/
THINKING/QUEUE/STATE edits (dispatch; the coordinator folds). Commit + push
per phase.

Run:  cd lab && python e290_coupling_ladder.py    (E290_SMOKE=1 shakedown)
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

SMOKE = os.environ.get("E290_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e290_smoke" if SMOKE else "e290"
assert torch.cuda.is_available(), "e290 owns the GPU lane (dispatch)"

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

# ---- THE REBINDING (e268/e273/e278/e283/e284/e285's disclosed convention):
# e261's drivers resolve their module globals (log / NAME / LADDER /
# RUNG_NAMES / SMOKE / T0 / thermal ledgers) AT CALL TIME through e261's
# module namespace — rebound HERE so the room build + envelope polls label
# THIS cell. The committed lab/e261_rank_ladder.py is untouched.
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
ROOMS264_CK = "e264_rooms.pt"     # e264's committed rooms (the K10K bit-bind)
CKPT_DIR = GB.CKPT_DIR

# ---- THE ROOM: the committed K10K room (k=10k, seeds 26113/26114) — the
# room the established fact was WRITTEN IN (e285's convention verbatim)
LADDER_FULL: tuple[tuple[int, int, int], ...] = (
    (10_000, 26113, 26114),       # K10K — e261's registered seed pair
)
LADDER_SMOKE: tuple[tuple[int, int, int], ...] = (
    (512, 26113, 26114),
)
LADDER = LADDER_SMOKE if SMOKE else LADDER_FULL
RUNG = {k: ("K10K" if not SMOKE else f"K{k}") for k, _, _ in LADDER}
ROOM_MODE = RUNG[LADDER[0][0]]       # the room's mode key (smoke names it K512)
E261.LADDER = LADDER            # the machinery's certify()/rooms read these
E261.RUNG_NAMES = RUNG

# ---- THE LADDER'S RUNGS (the ONLY delta across arms — frozen) -----------
# the geometric 0.2-factor descent from e285's 0.5x anchor point
BUDGET_FRACS: tuple[float, ...] = (0.1, 0.02, 0.004, 0.0008)
ARMS: tuple[str, ...] = ("BUDGET-0.1X", "BUDGET-0.02X", "BUDGET-0.004X",
                         "BUDGET-0.0008X")
ARM_DESC = {
    arm: (f"THE {frac:g}x RUNG: the sanctuary form VERBATIM (SGD-M m0.9 wd0 "
          f"at LR_STABLE 0.21738574801453703 x cosine_lr(t-1,1000), TWO SGD "
          "instances — opt_C stepped / opt_F never stepped with bitwise "
          "isolation probes, clip 1.0 -> g_perp = g - P_room(g) verified "
          "every step -> the equal-share per-step lr cap) with the TOTAL-"
          f"DRIFT BUDGET := {frac:g} x ||fact - base|| = {frac:g} x "
          "9.1788432658723; NO maintenance — the pure passive curve at this "
          "budget")
    for arm, frac in zip(ARMS, BUDGET_FRACS)}
FRAC_OF = dict(zip(ARMS, BUDGET_FRACS))

# THIS cell's ONE fresh registered corpus stream (the e268-family
# convention: one fresh registered stream per concurrent cell — e283 28301,
# e284 28401, e285 28501, e288 28801, HERE 29001). ALL FOUR rungs draw the
# IDENTICAL sequence (each rung its own generator instance at the same
# seed) — bit-identical corpus batches across rungs, the BUDGET the rungs'
# ONLY delta.
CORPUS_GEN_SEED = 29001

# THE CONVENTION FREEZES (e283/e285's, carried verbatim — no install, no
# maintenance)
PHASE_STEPS = 8 if SMOKE else 400               # CORPUS steps per rung
MILESTONES = tuple(range(1, 9)) if SMOKE else (100, 200, 300, 400)

# THE ESTABLISHED FACT: the committed quiet-formed 10k write — e261's serial
# cut, completed by e264 (the committed threshold rung's final state)
FACT_CK = "e261_K10K_inst_resume.pt"
FACT_MD5 = "0f6dc1cf46850ce655dfafc9c853d467"
FACT_SIZE = 32958479
FACT_STEP = 400
FACT_TRAJ_STEPS = [1, 100, 200, 300, 400]
FACT_LEDGER_MAX = 400

# ---- THE SGD-M CONFIG (frozen; e280/e284/e285's committed lr class) -----
SGD_MOMENTUM = 0.9                # e273's stable rider convention VERBATIM
SGD_WD = 0.0                      # e273's disclosed deviation (wd dropped)
LR_SGD_MATCHED = 21.7385748014537     # e273's committed calibration (md5-bound)
E273_LR_SGD = 21.7385748014537         # the calibration record's literal
SGD_STABLE_FACTOR = 0.01              # e273's SGD001X rider factor
LR_STABLE = SGD_STABLE_FACTOR * LR_SGD_MATCHED   # 0.21738574801453703

# ---- THE BUDGET (frozen; the ladder's rungs + the slack) ----------------
BUDGET_SLACK = 1e-5               # fp32 accumulation slack on S + cum (abs)

# the committed records, HARD-BOUND (read at runtime from their paths and
# asserted against these literals; Rule 12). The JSON records are bound on
# their GIT-CANONICAL md5s (newline-normalized bytes == the committed blob
# — verified at birth; see deviations: the OneDrive CRLF re-serialization
# catch); the .pt artifacts are bound RAW.
E264_METRICS = E43.REPO / "runs" / "e264" / "metrics.json"
E264_MD5 = "1149f97c633072c3b865b6a93bfa1e44"
E264_VERDICT = "SHARP-THRESHOLD"
FACT_BASELINE_G0 = 0.26464763283729553      # e264's committed K10K post g0
FACT_BASELINE_GM12 = 0.10525520890951157    # e264's committed K10K post gm12

E268_METRICS = E43.REPO / "runs" / "e268" / "metrics.json"
E268_MD5 = "1384d6ff1ba3b3362ae333d8504aacda"
E268_VERDICT = "DYNAMICAL-CARRIER"
E268_CONCURRENT_POST = 4.004325455753133e-05  # the FORMING write under fire
E268_RATIO = 0.00015130794937726159          # ~0.0002x — the forming death

E278_METRICS = E43.REPO / "runs" / "e278" / "metrics.json"
E278_MD5 = "db14cdff1fd5021a5b255c12127ea9df"
E278_VERDICT = "UNDERTOW-REGARDLESS"         # the roach-motel record

E283_METRICS = E43.REPO / "runs" / "e283" / "metrics.json"
E283_MD5 = "afd41378c48234e690544c1ba8510a27"
E283_VERDICT = "ESTABLISHED-DIES"
E283_POST = 9.04614535102155e-06             # the unprotected reference
E283_RATIO = 3.4181848724802654e-05          # 0.0000342x — the cite
E283_DRIFT = 14.453935847208887              # the transport read (||d||)
E283_WRITE_NORM = 9.1788432658723            # the write's own norm (||fact-base||)

E284_METRICS = E43.REPO / "runs" / "e284" / "metrics.json"
E284_MD5 = "f3c227e6f50bcfcdd9df00af50e0808a"
E284_VERDICT = "MOMENTUM-OWNED"
E284_SEP_PRIMARY = 2.319773558897833e-07     # the separate-buffer in-room share
E284_SHA_PRIMARY = 0.45027988873803715       # the shared twin's funnel

E285_METRICS = E43.REPO / "runs" / "e285" / "metrics.json"
E285_MD5 = "4a1dfe5a52d0b3def1b5f95db3d6a7ab"
E285_VERDICT = "AIM-ONLY-KILLS"              # THE sanctuary record
E285_POST = 0.0008025270071811974            # the 0.5x-budget arm's t400 read
E285_RATIO_400 = 0.00303243599263398         # 330x down — the anchor
E285_RATIO_100 = 0.005076788048356657        # the t100 datum (7.2% drift)
E285_DRIFT_100 = 0.6609871202355514          # 7.2% of the write
E285_S_FINAL = 4.589421633863822             # S at the 0.5x budget
E285_CUM_FINAL = 1.8445091952850854          # realized cum = 20.1% of write
E285_DRIFT_FINAL = 1.8445091953053947        # ||theta_400 - fact||
E285_TWIN_RATIO = 1.472106033067378e-05      # its same-session twin
E285_BUF_CITE = 0.5                          # the anchor's budget fraction

E288_METRICS = E43.REPO / "runs" / "e288" / "metrics.json"
E288_MD5 = "594129c80fd5df2c015e92c3ea3f95d9"
E288_VERDICT = "ERROR-GATED-HOLDS"           # active maintenance works
E288_POST = 0.9207638502120972
E288_RATIO = 3.4792068243367953
E288_S_TOTAL = 3.999805698171258             # 87.2% of its budget

X14_METRICS = E43.REPO / "runs" / "x14" / "metrics.json"
X14_MD5 = "29deab62eb62de54a86844c7e5efb016"
X14_VERDICT = "MIXED/INCONCLUSIVE"
X14_ARM_A = 0.03915366902947426              # the orthogonal-subtraction read
X14_ARM_B = 5.864363629370928e-05            # the in-room-subtraction read
X14_RESURRECTION_X = 4328.215777016314       # arm A / the dead state

E273_LRCAL = E43.REPO / "runs" / "e273" / "lr_calibration.json"
# REBOUND (pre-compute, disclosed in deviations): the raw-byte md5 this
# family bound at e285 (de0b1c3e152c99d7877391c4592e7e24) no longer
# matches — the worktree copy under OneDrive was re-serialized (CRLF
# flip, one byte; content verified byte-identical to git HEAD after
# newline normalization; the lr_sgd literal is asserted independently by
# G_LR_BIND). The bind below hashes NEWLINE-NORMALIZED bytes (the git
# canonical form — immune to OneDrive's CRLF/LF flips).
E273_LRCAL_MD5 = "f223467ed2e05b5fae45e32dba2c25e1"
ROOMS264_MD5 = "2d524655575cce00a3bc1c8770f4b211"    # e268's G_PARENTS bind

# the frozen bars' numbers ------------------------------------------------
SURVIVE_FRAC = 0.5               # HOLDS: >= 0.5x survival at t400
ORTH_BAR = 1e-6                  # G_ORTH: the missile's orthogonality gate
BUFSEP_ORTH_BAR = 1e-4           # G_BUFSEP: ||P_room buf_C||/||buf_C|| below
FACT_READ_TOL_G0 = 2e-6          # G_FACTLOAD behavioral bars (the family's
FACT_READ_TOL_GM12 = 1e-5        # cross-session read-determinism law)
G_READ_TOL = E261.G_READ_TOL             # 5e-3
RATIO_DEN_FLOOR = 1e-6           # the ladder's floor-guard convention
VOCAB_EXPECT = 65

REGISTERED = {
    "question_verbatim": "THE COUPLING-CONSTANT LADDER — measuring the "
        "program's load-bearing constant: the orthogonal-drift threshold at "
        "which an established write's read dies. e285 (the sanctuary) found "
        "a 20.1%-of-write orthogonal drift killed the read 330x with the "
        "budget held; its t100 datum (7.2% drift -> 200x down) bounds the "
        "threshold at <= 0.07x — a ONE-DRAW bound (R68's critic: the "
        "monotonicity assumed, the threshold could sit at 0.5%). The ladder "
        "measures it. NOTE e288 just landed (ERROR-GATED-HOLDS — active "
        "maintenance works); the constant prices when maintenance is "
        "NECESSARY (below the threshold, no anchor needed).",
    "arms_verbatim": ARM_DESC,
    "bars_verbatim": {
        "TIGHT-CONSTANT": "the 0.1x arm dies (< 0.5x) AND some lower rung "
            "holds — the threshold located in the ladder's span; report the "
            "bracket + the curve shape (graded vs step).",
        "NO-PASSIVE-THRESHOLD": "ALL rungs die (< 0.5x) even at 0.0008x — "
            "the coupling is effectively unbounded; NO passive drift is "
            "safe; maintenance is ALWAYS necessary for a moving organism "
            "(the strongest fragility statement; e288's controller is then "
            "the only preservation).",
        "ROBUST-FLOOR": "the 0.1x arm HOLDS (>= 0.5x) — contradicting "
            "e285's single-datum bound (the draw or the phase timing "
            "mattered); the threshold is looser than 0.07x after all.",
        "MIXED": "non-monotone across rungs — everything verbatim.",
    },
    "reads_verbatim": "the read's survival ratio at t400 (the threshold = "
        "the largest budget whose arm holds >= 0.5x); post g0 at "
        "t100/200/300/400; the realized drift per milestone (the "
        "verification); no maintenance (the pure passive curve at each "
        "budget).",
    "operationalizations": (
        "frozen BEFORE compute: THE PHASE := 400 CORPUS steps per rung, no "
        "install, no maintenance (e283/e285's convention freeze); "
        "milestones t=100/200/300/400; each rung a FRESH instance from the "
        "same loaded fact; THE VEHICLE := the committed quiet-formed 10k "
        "fact loaded BIT-EXACT (three-way G_FACTLOAD); the room := the "
        "fact's OWN K10K room (seeds 26113/26114, bit-gated vs "
        "e264_rooms.pt); THE RUNG FORM := e285's SANCTUARY arm VERBATIM, "
        "only BUDGET_FRAC varying (SGD-M m0.9 wd0 at LR_STABLE "
        "0.21738574801453703 x cosine_lr(t-1,1000); TWO SGD instances — "
        "opt_C stepped / opt_F never stepped, bitwise isolation probe "
        "every step; clip 1.0 -> g_perp = g - P_room(g) verified EVERY "
        "step -> the equal-share per-step lr cap cap_t = ((BUDGET - "
        "S_{t-1})/(n-t+1))/||b_t||, b_t = 0.9*buf_{t-1} + g_perp; lr_t = "
        "min(schedule, cap); S += realized ||step||); THE BUDGETS := "
        "{0.1, 0.02, 0.004, 0.0008} x 9.1788432658723 (slack 1e-5 abs); "
        "THE CORPUS STEP := e268's registered form VERBATIM on THIS cell's "
        f"ONE fresh registered stream seed {CORPUS_GEN_SEED} (ALL FOUR "
        "rungs draw the identical sequence — the budget is the rungs' ONLY "
        "delta); THE SURVIVAL RATIO := post g0(t) / "
        f"{FACT_BASELINE_G0} (committed, PRIMARY); THE ADJUDICATION READ "
        ":= each rung's t=400 endpoint; COMPOSITE := TEXTURE (any "
        "hard-gate failure) -> NO-PASSIVE-THRESHOLD (all four die) -> "
        "ROBUST-FLOOR (the 0.1x rung holds) -> TIGHT-CONSTANT (the 0.1x "
        "rung dies AND >= 1 lower rung holds AND the holds-pattern is "
        "monotone — a suffix of the rung list) -> MIXED (non-monotone — "
        "everything verbatim); THE THRESHOLD BRACKET := budget terms "
        "(largest holding frac, smallest dying frac above it], "
        "co-reported in realized-drift terms; HARD GATES := {G_NAMEFREE, "
        "G_SPLICE, G_BATTERY, G_ANCHOR, G_INSTMASK, G_PARENTS, G_BASE, "
        "G_ROOT, G_VMBIND, G_SPANBIND, G_PROJ, G_ROOMK10K, G_FACTLOAD, "
        "G_CORPUSGEN, G_LR_BIND, G_ORTH, G_BUFSEP, G_BUDGET} (G_ORTH/"
        "G_BUFSEP/G_BUDGET PER RUNG — all four must pass); NO CONS "
        "(T259/e281; all four post states checkpointed); NO TWIN "
        "(e285's same-session twin 1.47e-05x + e283's committed 3.42e-05x "
        "hard-bound as the death class's record)."),
    "registration": "bars + question + arms frozen VERBATIM from the "
        "dispatch letter (the e288 fold's dispatch; the coupling-constant "
        "ladder); this script committed at birth BEFORE any compute; "
        "adjudicate against exactly this; no bar shopping.",
    "predictions": {
        "P-e290a_tight_constant": "the 0.1x rung's realized cumulative "
            "drift lands ~4% of the write (e285's cancellation ratio: cum "
            "~40% of budget) — inside the lethal band e285 already "
            "measured at t100 (7.2% -> 0.0051x) — so the 0.1x rung DIES; "
            "the threshold sits at or below the 0.02x rung's realized "
            "~0.8%: TIGHT-CONSTANT in-span, or NO-PASSIVE-THRESHOLD if "
            "even ~0.03% kills by t400.",
        "P-e290b_no_passive_threshold": "R68's critic branch: the "
            "threshold sits under ~0.5% of the write; the read dies at "
            "every rung by t400; the coupling is effectively unbounded; "
            "maintenance ALWAYS necessary (e288's controller the only "
            "preservation).",
        "P-e290c_robust_floor": "e285's kill was draw/phase-timing "
            "sensitive; the 0.1x rung holds >= 0.5x at t400 and the "
            "threshold is looser than 0.07x after all.",
    },
}

deviations: list[str] = [
    "THE SMOKE PASS (pass 2, runs/e290_smoke/ local per .gitignore; the "
    "e260-family record): ONE environment catch — the G_PARENTS raw-md5 "
    "bind caught an OneDrive CRLF re-serialization of the e273 "
    "lr-calibration record between birth and smoke (fixed pre-compute, "
    "committed separately; see the next entry). THE SUBSTANCE "
    "VERIFICATIONS (all PASS, all four rungs at 8 steps with the "
    "budgets scaled to the FULL RUN'S PER-STEP SHARE): (1) THE "
    "BUDGET-CAP ARITHMETIC AT THE SMALLEST RUNG (the dispatch's "
    "centerpiece) — the 0.0008x rung ran at exactly the full-run share "
    "scale (budget 1.47e-04 over 8 steps = 1.83e-05/step): the "
    "equal-share reservation held S to EXACTLY 100.0% of budget, every "
    "applied lr positive (min 7.28e-06 — no underflow), every realized "
    "step norm positive, realized cum 89.5% of budget (the orthogonal "
    "steps cancel below the triangle bound exactly as designed); the "
    "worst S overage across rungs 0.031% of that rung's budget = 4.5e-08 "
    "absolute (fp32 noise, 200x under the 1e-05 slack); (2) THE BUFFER "
    "SEPARATION — 8/8 bitwise isolation checks per rung (32 total), 0 "
    "violations, opt_F's state EMPTY throughout, buf_C in-room at the "
    "fp floor (3.3e-10 -> 8.4e-10 vs the 1e-4 bar — e284/e285's SEP "
    "arithmetic replicating); (3) the orthogonality machine-exact (max "
    "1.09e-17); (4) G_FACTLOAD's three-way bind EXACT (|d post g0| = "
    "0.0); (5) the draw-integrity check EXACT (the t1 corpus CE + "
    "clipped gn bit-identical across ALL FOUR rungs — the budget the "
    "rungs' only delta); (6) G_PARENTS/e285's anchor + e288's record "
    "hard-bound on the git-canonical md5s. Disclosed smoke-scale "
    "preview (NOT adjudicated, SMOKE stamped): survival scaled "
    "monotonically with the budget (t8 x0.572 / x0.902 / x0.980 / "
    "x0.996 at 0.1x/0.02x/0.004x/0.0008x) — at smoke k=512 the room "
    "captures only 21.7% of the write (e285's smoke note), and each "
    "smoke rung's TOTAL drift is 8/400 of the full rung's; the full "
    "run at k=10k (94.4% capture) with 50x each rung's total drift is "
    "the datum.",
    "THE ONEDRIVE RE-SERIALIZATION CATCH (the smoke's first catch, fixed "
    "pre-compute; no bar, gate, arm, or read touched): G_PARENTS' first "
    "pass FAILED at the e273 lr-calibration md5 — the worktree copy "
    "(under OneDrive) had been re-serialized between this cell's birth "
    "probe and the smoke (a CRLF flip, one byte; the family's e285-era "
    "raw literal no longer matched). The repair: every JSON parent record "
    "is now bound on its GIT-CANONICAL md5 (newline-normalized bytes — "
    "verified byte-identical to the committed blob for all 8 JSON "
    "parents at birth), with the raw worktree md5 co-reported in each "
    "gate row; the .pt artifacts stay RAW-bound (OneDrive does not "
    "re-serialize binaries). The protected content is unchanged — the "
    "lr_sgd literal is asserted independently by G_LR_BIND's runtime "
    "re-derivation.",
    "THE T268 NOTE (disclosed at birth): the dispatch named THINKING.md "
    "T265/T267/T268 as the read-first cards; T268 (e288's fold) is NOT YET "
    "IN THINKING.md at this cell's birth — e288's record was read instead "
    "from its committed metrics (md5-bound here) + NOTES.md's e288 entry + "
    "STATE.json's fold line. The coordinator folds T268; not this cell's "
    "edit (dispatch: NO THINKING edits).",
    "THE RUNG FORM IS e285's SANCTUARY ARM PORTED VERBATIM (extend, don't "
    "repeat): chunked_ladder_phase below is e285's "
    "chunked_sanctuary_phase with (i) the arm tag + budget passed as "
    "parameters (four instances, one per rung), (ii) the resume-ckpt name "
    "per arm — the step body (draws, clip, g_perp, the equal-share cap, "
    "the isolation probe, the milestone battery, every ledger) is "
    "line-for-line e285's. The committed lab/e285_sanctuary.py is NOT "
    "modified.",
    "THE TWIN IS CITED, NOT RE-RUN (disclosed): the dispatch's arms are "
    "the four passive rungs; the unprotected death class is already "
    "committed (e283's record 3.42e-05x + e285's same-session twin "
    "1.47e-05x — both md5-hard-bound in G_PARENTS). No bar reads a "
    "same-session twin; re-running one would double the GPU bill for a "
    "class already replicated twice.",
    "THE ANCHOR IS A DIFFERENT SESSION'S DRAW STREAM (disclosed): e285's "
    "0.5x-budget arm ran seed 28501; this ladder's rungs run seed 29001 "
    "(the family's per-cell rule). The anchor's overlay on the curve is "
    "the committed record's datum (class replication, the family's rule — "
    "bit-replication was never it); the threshold's bracket is adjudicated "
    "on THIS session's four rungs alone (bit-identical draws across "
    "rungs), with the anchor co-reported.",
    "THE BUDGET'S PRICE (disclosed at birth, measured in the CE + lr "
    "ledgers): the equal-share reservation holds the per-step displacement "
    "at ~BUDGET/400 — 0.0023 (0.1x) down to 1.8e-05 (0.0008x) — far under "
    "the nominal schedule's post-warmup ~0.4-0.7, so the cap binds from "
    "the first post-warmup steps at EVERY rung and the lower rungs run at "
    "correspondingly tiny effective lrs (fp32 lr precision verified in "
    "smoke: lr_applied > 0 every step, S monotone, no underflow; the "
    "realized step norm vs the share target disclosed per rung — fp32 "
    "update rounding may shave the smallest rung's realized steps BELOW "
    "the cap's prediction, a conservative direction: realized drift is "
    "MEASURED, never assumed).",
    "NO CONS (T259/e281; the family's committed form): the frozen bars "
    "read the WRITE and the DISPLACEMENT only; all four post states are "
    "checkpointed (e290_<ARM>_post.pt) for any later landing pass.",
    "e261's MACHINERY PORTED WHOLE BY IMPORT: the SRCT projector + "
    "LadderRooms (the room rebuild, certification, displacement loads), "
    "the thermal envelope (per-step polls, 78C margin, 175s bursts inside "
    "the dispatch's 180s, 40s cooldowns, the 84C line inside the "
    "dispatch's 85C), the progressive-metrics + resume-ckpt conventions — "
    "the module-global rebinding (log/NAME/LADDER/RUNG_NAMES/T0/thermal "
    "ledgers, disclosed in-code) retargets the machinery's I/O to this "
    "cell; the committed lab/e261_rank_ladder.py is NOT modified. The "
    "driver is THIS file's: chunked_ladder_phase.",
    "THE V-MAP AND SPAN ARE LOADED, NOT RE-RUN (extend, don't repeat): "
    "e258's committed v-map + e246's committed LATE span feed the measured "
    "displacement loads; no new history is run.",
    "n=1 per rung, one lineage, one session (the g-series standing caveat "
    "— the critic's lottery note carried verbatim); the rungs' MONOTONE "
    "PATTERN across bit-identical draws is the registered object, not any "
    "single point; nothing guaranteed.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
    "Smoke mode (E290_SMOKE=1): 8 corpus steps per rung, milestones at "
    "every step 1..8, room k=512 at the same seed pair (G_ROOMK10K "
    "vacuous — no committed record at smoke k; disclosed), the REAL "
    "committed fact loaded and read (G_FACTLOAD live), G_CORPUSGEN live, "
    "G_ORTH live (every step), G_BUFSEP live (isolation + composition), "
    "G_BUDGET live — each rung's smoke budget is SCALED to the full run's "
    "PER-STEP SHARE (BUDGET x 8/400) so the cap's binding path runs at "
    "exactly the full-run share scale INCLUDING the smallest rung's "
    "1.8e-05/step (the dispatch's fp-precision centerpiece); the "
    "adjudication form + figure exercised; all paths smoke_-prefixed, own "
    "smoke dir; NOTHING adjudicated or gated for the record (SMOKE stamp "
    "on every read).",
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


def md5of_norm(p: Path) -> str:
    """md5 of the NEWLINE-NORMALIZED bytes (CRLF -> LF) — the git canonical
    form. The repo lives under OneDrive, which re-serializes text files'
    line endings at sync time; the JSON parent records are therefore bound
    on the normalized hash (raw md5 co-reported in the gate rows). BINARY
    artifacts (.pt) are always bound RAW (md5of)."""
    return hashlib.md5(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


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
    """e278/e280/e284/e285's orthogonalize_grads VERBATIM: replace the
    (clipped) corpus gradient g by g_perp = g - P_room(g) — the component
    ENTIRELY ORTHOGONAL to the room (CPU fp64, write fp32, norm NOT
    rescaled). The verification read ||P_room g_perp|| / ||g_perp|| is
    returned for the gate (the projector is exact: P(I-P) = 0 to fp64
    roundoff ~1e-15; any drift above 1e-6 is an implementation bug, not
    numerics)."""
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


# --------------------------------------------- THE BUFFER MACHINES (port)
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


def pending_buffer_sqnorm(opt_C, params, momentum: float) -> float:
    """||b_t||^2 for PyTorch SGD-M's exact in-step update b_t = mu * buf_{t-1}
    + g (dampening 0): computed on-GPU fp32 from the optimizer's stored
    buffers + the (orthogonalized) grads now in p.grad — the cap's
    per-step price of the applied displacement (theta -= lr_t * b_t)."""
    sq = torch.zeros((), device=next(iter(params)).device)
    st = opt_C.state
    for p in params:
        d = st.get(p)
        buf_prev = d.get("momentum_buffer") if isinstance(d, dict) else None
        if buf_prev is None:
            v = p.grad.detach()
        else:
            v = momentum * buf_prev + p.grad.detach()
        sq = sq + v.square().sum()
    return float(sq.item())


# ======================================================================
# THE LADDER DRIVER — e285's chunked_sanctuary_phase VERBATIM in body, the
# budget + arm tag the parameters (one instance per rung)
# ======================================================================
def chunked_ladder_phase(tag: str, budget_norm: float, net0,
                         proj: "E261.LadderRooms",
                         fact_flat_np: np.ndarray,
                         base_flat_np: np.ndarray,
                         anchor_full, train_ids, g0_ids, gm12_ids,
                         r_eval_xy, zid, resume_ck: Path,
                         dev: torch.device) -> dict:
    """ONE RUNG'S DRIVER (e285's sanctuary arm, only the budget varying).
    The net starts at the LOADED formed fact (net0 carries it); per corpus
    step t = 1..400:

      draws (aj_c(16), rj_c(32)) from cgen (seed 29001 — bit-identical to
      every other rung's draws); corpus batch = 16 original-host anchors +
      32 random corpus windows; full-window CE; lr_sched = LR_STABLE x
      cosine_lr(t-1, 1000); backward -> clip 1.0 -> the orthogonalization
      g_perp = g - P_room(g) (verified EVERY step) -> the per-step lr cap
      (equal-share reservation against the pending buffer norm) -> the
      isolated opt_C.step() (the corpus side's OWN momentum buffer; opt_F
      snapshotted bitwise around the step — the fact side stays EMPTY).

    Ledgers: the orthogonality ledger (every step checked, rows every 10);
    the budget ledger (per milestone: S, realized cum, usage, the cap's
    binding); the buffer ledger (per milestone: composition + isolation
    counts); the displacement ledger (the drift from the fact with in-room
    share, the remaining-from-base occupancy, the corpus interval/
    cumulative displacement with in-room fractions); the WRITE read
    (g0/gm12 battery + CE_R) at the milestones. Thermal: a poll after
    EVERY corpus opt step."""
    corp_bs, mix_random = E43.CORP_BS, E43.MIX_RANDOM
    n_steps = PHASE_STEPS
    n_anc = anchor_full.shape[0]
    N = int(fact_flat_np.size)
    state = {"step": 0, "traj": [], "corpus_ledger": {}, "orth_ledger": {},
             "disp_ledger": [], "buf_ledger": [], "budget_ledger": [],
             "lr_ledger": {},
             "bufsep": {"isolation_checks": 0, "isolation_violations": 0,
                        "optF_ever_stepped": False},
             "corp_cum": torch.zeros(N, dtype=torch.float64),
             "corp_prev": torch.zeros(N, dtype=torch.float64),
             "S": 0.0, "n_capped": 0}
    if resume_ck.exists():
        state = torch.load(resume_ck, map_location="cpu", weights_only=False)
        log(f"  [{tag}] RESUMED from {resume_ck.name} at corpus step "
            f"{state['step']}/{n_steps}")
    if int(state.get("step", 0)) >= n_steps:
        log(f"  [{tag}] resume ckpt already COMPLETE at t{state['step']}")
        _lrs = [v["lr_applied"] for v in state.get("lr_ledger", {}).values()]
        _caps = [v["cap"] for v in state.get("lr_ledger", {}).values()]
        _scheds = [v["lr_sched"] for v in state.get("lr_ledger", {}).values()]
        _real = [v["realized_step_norm"]
                 for v in state.get("lr_ledger", {}).values()]
        _shares = [v["share"] for v in state.get("lr_ledger", {}).values()]
        return {"sd": state["model"], "traj": state.get("traj", []),
                "corpus_ledger": state.get("corpus_ledger", {}),
                "orth_ledger": state.get("orth_ledger", {}),
                "disp_ledger": state.get("disp_ledger", []),
                "buf_ledger": state.get("buf_ledger", []),
                "budget_ledger": state.get("budget_ledger", []),
                "lr_ledger": state.get("lr_ledger", {}),
                "bufsep": state.get("bufsep", {}),
                "S": state.get("S"), "n_capped": state.get("n_capped"),
                "orth_max": state.get("orth_max"),
                "lr_applied_min": min(_lrs) if _lrs else None,
                "lr_applied_median": float(sorted(_lrs)[len(_lrs) // 2])
                if _lrs else None,
                "lr_sched_median": float(sorted(_scheds)[len(_scheds) // 2])
                if _scheds else None,
                "cap_min": min(_caps) if _caps else None,
                "realized_step_min": min(_real) if _real else None,
                "share_min": min(_shares) if _shares else None,
                "realized_over_share_last": (_real[-1] / _shares[-1]
                                             if _real and _shares
                                             and _shares[-1] > 0 else None),
                "steps_ran": n_steps, "n_chunks": state.get("n_chunks", 0),
                "chunk_table": state.get("chunk_table", []),
                "resumed_final": True}
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
            # the separation: TWO SGD instances over the SAME parameters —
            # the corpus side's buffer is PRIVATE; the fact side never steps.
            opt_C = torch.optim.SGD(net.parameters(), lr=LR_STABLE,
                                    momentum=SGD_MOMENTUM,
                                    weight_decay=SGD_WD)
            opt_F = torch.optim.SGD(net.parameters(), lr=LR_STABLE,
                                    momentum=SGD_MOMENTUM,
                                    weight_decay=SGD_WD)
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
            lr_sched = LR_STABLE * cosine_lr(step - 1, E261.INST_TOTAL)
            # ---- THE CORPUS STEP (the family's standard, bit-identical
            # draws to every other rung)
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
            # ---- the orthogonalized stream (verified EVERY step)
            orth_row = orthogonalize_grads(proj, net.parameters(), ROOM_MODE)
            orth_max = max(orth_max, orth_row["orth_rel_err"])
            # ---- the per-step lr cap (equal-share reservation)
            b_sq = pending_buffer_sqnorm(opt_C, params_live, SGD_MOMENTUM)
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
            # ---- the isolated corpus step (the fact side EMPTY)
            sn_f_before = snap_buffers(opt_F, params_live)
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
            if not buffers_bitwise_equal(sn_f_before,
                                         snap_buffers(opt_F, params_live)):
                state["bufsep"]["isolation_violations"] += 1
                raise RuntimeError(
                    f"[{tag}] G_BUFSEP ISOLATION VIOLATION at t{step}: the "
                    "corpus step touched the fact side's state — HALT")
            if step % 10 == 0 or step == 1 or step == n_steps or SMOKE:
                state["corpus_ledger"][step] = {"ce": float(loss_c.item()),
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
            ok_t, temp = E261.burst_temp_check(
                f"{tag}:corpus:c{n_chunks}.x")
            chunk_temps.append(temp)
            if step in MILESTONES or step == n_steps or SMOKE:
                sd_cpu = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
                evl.load_state_dict(sd_cpu)
                evl.eval()
                bz0 = G1.battery_cell(evl, g0_ids, zid)
                bz12 = G1.battery_cell(evl, gm12_ids, zid)
                ce_r = G1.ce_fixed_cpu(evl, *r_eval_xy)
                theta_t = flat_params_cpu(net).double().numpy() \
                    .astype(np.float64)
                d_fact = theta_t - fact_flat_np          # THE drift read
                dn = float(np.linalg.norm(d_fact))
                pdf = room.project(d_fact)
                d_fact_in_room = float(np.linalg.norm(pdf) / dn) \
                    if dn > 0 else None
                rem = theta_t - base_flat_np             # the write's own
                rn = float(np.linalg.norm(rem))          # remaining disp
                pr = room.project(rem)
                rem_in_room = float(np.linalg.norm(pr) / rn) if rn > 0 \
                    else None
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
                bufC_ir, bufC_n = buffer_inroom_frac(opt_C, params_live,
                                                     room)
                state["disp_ledger"].append({
                    "step": step,
                    "drift_from_fact_norm": dn,
                    "drift_from_fact_in_room_frac": d_fact_in_room,
                    "remaining_from_base_norm": rn,
                    "remaining_from_base_in_room_frac": rem_in_room,
                    "corpus_disp_interval_norm": vn,
                    "corpus_disp_interval_in_room_frac": in_room_frac_c,
                    "corpus_disp_cum_norm": cum_n,
                    "corpus_disp_cum_in_room_frac": in_room_frac_cum})
                state["buf_ledger"].append({
                    "step": step,
                    "bufC_inroom_frac": bufC_ir, "bufC_norm": bufC_n,
                    "optF_state_entries": len(opt_F.state),
                    "optF_ever_stepped": False})
                state["budget_ledger"].append({
                    "step": step, "S": S,
                    "S_usage_frac": S / budget_norm,
                    "cum_norm": cum_n,
                    "cum_usage_frac": cum_n / budget_norm,
                    "n_capped_so_far": n_capped,
                    "lr_sched": lr_sched, "cap": cap, "lr_applied": lr_t,
                    "capped": capped})
                state["traj"].append({
                    "step": step,
                    "g0_pz": bz0["mean_pz"], "g0_argmax": bz0["frac_argmax_z"],
                    "gm12_pz": bz12["mean_pz"], "ce_r": ce_r,
                    "ce_corpus": float(loss_c.item()),
                    "survival_ratio_vs_committed":
                        bz0["mean_pz"] / FACT_BASELINE_G0,
                    "disp_norm": dn,
                    "in_room_frac": d_fact_in_room,
                    "remaining_in_room_frac": rem_in_room,
                    "corpus_disp_interval_in_room_frac": in_room_frac_c,
                    "corpus_disp_cum_in_room_frac": in_room_frac_cum,
                    "budget_S_usage_frac": S / budget_norm,
                    "budget_cum_usage_frac": cum_n / budget_norm,
                    "lr_applied": lr_t, "lr_sched": lr_sched,
                    "capped": capped,
                    "elapsed_s": round(time.time() - T0, 1)})
                log(f"  [{tag}] t{step:4d} g0 {bz0['mean_pz']:.6f} "
                    f"(x{bz0['mean_pz'] / FACT_BASELINE_G0:.4f}) g-12 "
                    f"{bz12['mean_pz']:.5f} CE_R {ce_r:.4f} CE_corp "
                    f"{float(loss_c.item()):.4f} |d| {dn:.3f} drift-in-room "
                    f"{('%.2e' % d_fact_in_room) if d_fact_in_room is not None else 'n/a'} "
                    f"rem-in-room "
                    f"{('%.4f' % rem_in_room) if rem_in_room is not None else 'n/a'} "
                    f"| BUDGET S {S:.4f}/{budget_norm:.4f} "
                    f"({S / budget_norm:.1%}) cum {cum_n:.4f} "
                    f"({cum_n / budget_norm:.1%}) lr {lr_t:.3e} "
                    f"{'CAP' if capped else 'sched'} bufC "
                    f"{('%.*e' % (2, bufC_ir)) if bufC_ir is not None else 'n/a'}")
            if not ok_t:
                E261._end_burst_early(tag, n_burst, t_burst)
                chunk_capped = True
                break
            if (time.time() - t_burst) > E261.BURST_MAX_S:
                log(f"  [{tag}] chunk {n_chunks}: burst cap "
                    f"{E261.BURST_MAX_S:.0f}s at t{step} — resume ckpt saved")
                chunk_capped = True
                break
        chunk_table.append({**devices_row,
                            "seconds": round(time.time() - t_burst, 1),
                            "steps_done": step, "capped": chunk_capped,
                            "max_temp_c": max(chunk_temps, default=None)})
        torch.save({"model": {k: v.detach().cpu()
                              for k, v in net.state_dict().items()},
                    "optC": opt_C.state_dict(),
                    "cgen_state": cgen.get_state(),
                    "step": step, "traj": state["traj"],
                    "corpus_ledger": state["corpus_ledger"],
                    "orth_ledger": state["orth_ledger"],
                    "disp_ledger": state["disp_ledger"],
                    "buf_ledger": state["buf_ledger"],
                    "budget_ledger": state["budget_ledger"],
                    "lr_ledger": state["lr_ledger"],
                    "bufsep": state["bufsep"],
                    "corp_cum": corp_cum.cpu(),
                    "corp_prev": corp_prev.cpu(),
                    "S": S, "n_capped": n_capped, "orth_max": orth_max,
                    "n_chunks": n_chunks, "chunk_table": chunk_table},
                   resume_ck)
        if step >= n_steps:
            break
        if n_chunks >= 12:
            raise RuntimeError(f"{tag}: cap-loop guard at chunk {n_chunks}")
        E261.burst_cooldown(tag)
        t_burst = None
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    lrs = [v["lr_applied"] for v in state["lr_ledger"].values()]
    caps = [v["cap"] for v in state["lr_ledger"].values()]
    scheds = [v["lr_sched"] for v in state["lr_ledger"].values()]
    real = [v["realized_step_norm"] for v in state["lr_ledger"].values()]
    shares = [v["share"] for v in state["lr_ledger"].values()]
    return {"sd": sd_cpu, "traj": state["traj"],
            "corpus_ledger": state["corpus_ledger"],
            "orth_ledger": state["orth_ledger"],
            "disp_ledger": state["disp_ledger"],
            "buf_ledger": state["buf_ledger"],
            "budget_ledger": state["budget_ledger"],
            "lr_ledger": state["lr_ledger"],
            "bufsep": state["bufsep"],
            "S": S, "n_capped": n_capped, "orth_max": orth_max,
            "lr_applied_min": min(lrs) if lrs else None,
            "lr_applied_median": float(sorted(lrs)[len(lrs) // 2])
            if lrs else None,
            "lr_sched_median": float(sorted(scheds)[len(scheds) // 2])
            if scheds else None,
            "cap_min": min(caps) if caps else None,
            "realized_step_min": min(real) if real else None,
            "share_min": min(shares) if shares else None,
            "realized_over_share_last": (real[-1] / shares[-1]
                                         if real and shares
                                         and shares[-1] > 0 else None),
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
           "per_step_polls": "after EVERY corpus opt step (all rungs) — "
                             "aggregated from runs/_envelope_log.jsonl (the "
                             "persisted ledger; survives resume passes; "
                             "tags e290:<ARM>:<phase> per the dispatch)",
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
                   f"polls are tagged e290_smoke: and excluded")
    return out


def med(xs) -> float:
    xs = sorted(xs)
    return float(xs[len(xs) // 2]) if xs else None


def main():
    dev = torch.device("cuda")
    metrics.update({
        "experiment": "e290_coupling_ladder",
        "phase": "THE COUPLING-CONSTANT LADDER — measuring the program's "
                 "load-bearing constant: the orthogonal-drift threshold at "
                 "which an established write's read dies. e285's 0.5x-"
                 "budget sanctuary arm died at 20.1%-of-write realized "
                 "orthogonal drift (330x down) with every protection "
                 "verified; its t100 datum bounds the threshold at <= 0.07x "
                 "— a ONE-DRAW bound. THE LADDER: four rungs at total-drift "
                 "budgets {0.1x, 0.02x, 0.004x, 0.0008x} of the write's "
                 "norm 9.1788, the sanctuary form VERBATIM (SGD-M, separate "
                 "buffers, orthogonal projection, equal-share cap), no "
                 "maintenance — the pure passive curve at each budget; the "
                 "threshold = the largest budget whose arm holds >= 0.5x "
                 "survival at t400",
        "date": common.now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "no_cons": {"cons_run": False,
                    "why": "T259/e281: the landing read is a cons property; "
                           "the frozen bars read the WRITE and the "
                           "DISPLACEMENT only; the family's committed "
                           "NO-CONS form; all four rungs' states are "
                           "checkpointed"},
        "envelope": {
            "device": "cuda fp32 training (this cell owns the GPU lane; the "
                      "four rungs run SEQUENTIALLY with cooldowns between — "
                      "never concurrent) + CPU fp64 dense projections "
                      "(pocketfft workers 2), CPU threads 4",
            "bursts": f"<= {E261.BURST_MAX_S:.0f}s (dispatch 180), "
                      f"per-step thermal polls at a "
                      f"{E261.TEMP_EARLY_END:.0f}C margin, cooldown "
                      f"{E261.COOLDOWN_S:.0f}s (dispatch 30-60), the "
                      f"{E261.TEMP_HARD:.0f}C never-past line (dispatch "
                      "85) recorded to runs/_envelope_log.jsonl tagged "
                      "e290:<ARM>:<phase>",
            "trainings": "4 corpus-only phases x 400 opt steps (the "
                         "sanctuary form at budget {0.1, 0.02, 0.004, "
                         "0.0008} x write norm); NO install (the fact is "
                         "loaded bit-exact); NO maintenance; NO cons",
        },
        "arms_desc": ARM_DESC,
        "convention_freezes": {
            "phase": f"{PHASE_STEPS} CORPUS steps per rung (no install, no "
                     "maintenance — e283/e285's convention freeze carried "
                     "verbatim)",
            "milestones": f"t = {'/'.join(str(m) for m in MILESTONES)} "
                          "corpus steps",
            "budgets": f"{list(BUDGET_FRACS)} x ||fact - base|| "
                       "(the geometric 0.2-factor descent from e285's 0.5x "
                       "anchor)",
            "corpus_step": "e268's registered corpus step VERBATIM: 48 "
                           "windows = 16 original-host anchors (the same "
                           "60-window bank) + 32 random corpus windows; "
                           "full-window CE; THIS cell's ONE fresh registered "
                           f"generator seed {CORPUS_GEN_SEED}; ALL FOUR "
                           "rungs draw the IDENTICAL sequence (the budget "
                           "is the rungs' ONLY delta)",
            "survival_ratio": f"post g0(t) / {FACT_BASELINE_G0} (the "
                              "committed loaded baseline, PRIMARY; the "
                              "session's loaded read co-reported)",
            "adjudication_read": "each rung's t=400 endpoint (all gate "
                                 "verdicts at the same endpoints); "
                                 "trajectories + realized-drift ledgers "
                                 "co-reported verbatim",
        },
        "deviations": deviations,
        "builds_on": [
            "T265 / e285 (THE SANCTUARY: the 0.5x-budget arm's honest "
            "failure — realized cum 1.845 = 20.1% of the write, read "
            "0.0030x, all protections verified; the coupling constant's "
            "<= 0.2x restatement; THE RIG THIS CELL PORTS VERBATIM)",
            "T268 / e288 (ERROR-GATED-HOLDS: the first active survival, "
            "x3.4792 by controller, budget 87.2% — the constant this cell "
            "measures prices when that controller is NECESSARY)",
            "T267 / e287 + T266 / e286 (the maintenance arc's mechanism "
            "cards: the name signal real, the room-frame not operative — "
            "the passive/active division this ladder prices)",
            "R68's critic (the one-draw bound's exposure: the threshold "
            "could sit at 0.5% — the 0.0008x rung's realized ~0.03% probes "
            "17x below it)",
            "T261 / e283 (the established-write rig: the loaded fact, the "
            "400-corpus-step phase, the conventions; the unprotected "
            "reference post 9.04614535102155e-06, ratio 0.0000342x, drift "
            "14.45 vs the write's 9.18)",
            "T264 / e284 (MOMENTUM-OWNED: buffer separation kills the "
            "re-aiming — DIAL machinery + provenance)",
            "T263 / x14 (DIRECTIONAL TRANSPORT: the out-of-room context "
            "carries 4,328x of the kill — the coupling constant's other "
            "side)",
            "T242 / e264 + T239 / e261 (the committed threshold rung: the "
            "loaded fact's baseline; the ladder machinery PORTED WHOLE BY "
            "IMPORT)",
            "T259 / e281 (the NO-CONS form)",
        ],
        "whats_new": [
            "THE COUPLING CONSTANT MEASURED AS A LADDER (the record's "
            "first): e285's single 0.5x datum becomes a four-rung geometric "
            "descent {0.1x, 0.02x, 0.004x, 0.0008x} on bit-identical draws "
            "— the threshold located (or its absence proven) in one "
            "session",
            "THE PASSIVE FLOOR'S EXISTENCE QUESTION (the record's first): "
            "if all four rungs die, NO passive drift is safe and "
            "maintenance is ALWAYS necessary for a moving organism — the "
            "strongest fragility statement the build era can make",
            "THE REALIZED-DRIFT VERIFICATION AT EVERY RUNG (the ladder's "
            "measurement discipline): the constant is reported in realized "
            "displacement (cum + drift-from-fact per milestone), never in "
            "nominal budget alone",
            "THE SMALLEST-RUNG FP PRECISION PASS (the dispatch's smoke "
            "centerpiece): the 1.8e-05-per-step share scale exercised and "
            "disclosed before the full run",
        ],
        "gates": {},
    })
    log(f"E290 — THE COUPLING-CONSTANT LADDER (smoke={SMOKE}) -> {RD}")
    log(f"rungs: {' / '.join(f'{a}={f:g}x' for a, f in zip(ARMS, BUDGET_FRACS))}; "
        f"the fact = {FACT_CK} (md5-bound, post g0 {FACT_BASELINE_G0:.8f}); "
        f"the room = the fact's own committed K10K (seeds "
        f"{LADDER[0][1]}/{LADDER[0][2]}, bit-gated vs {ROOMS264_CK}); the "
        f"phase = {PHASE_STEPS} corpus steps/rung; milestones "
        f"{'/'.join(str(m) for m in MILESTONES)}; the form = the sanctuary "
        f"VERBATIM (SGD-M m{SGD_MOMENTUM} wd{SGD_WD} @ LR_STABLE "
        f"{LR_STABLE:.13f} x cosine, SEPARATE buffers + g_perp + the "
        f"equal-share cap); the corpus stream = seed {CORPUS_GEN_SEED} "
        f"(bit-identical across rungs); the bar: HOLDS >= "
        f"{SURVIVE_FRAC:.0%}x at t400; e285's 0.5x anchor (ratio "
        f"{E285_RATIO_400:.5f}) overlaid")
    write_partial("startup (bars + rungs registered, committed at birth)")
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

    anchor_full = torch.stack([train_ids[p - G1.PRE: p - G1.PRE + G1.BLOCK]
                               for p, _ in install_occ])
    inst_mask = torch.zeros(60, G1.BLOCK - 1, dtype=torch.bool)
    inst_mask[:, G1.PRE - 1: G1.PRE - 1 + len(G1.NAME)] = True
    G_INSTMASK = {"name_positions": int(inst_mask.sum()),
                  "expected": 60 * len(G1.NAME),
                  "note": "the install windows are NOT run in this cell "
                          "(the fact is loaded); the mask gate carries the "
                          "splice/mask convention's identity",
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
        "/ e170 bank / install mask identity / vocab 65)")
    write_partial("P0 protocol gates PASSED")

    # ================= P0b: the parents hard-bound (Rule 12) ============
    e264m = json.loads(E264_METRICS.read_text(encoding="utf-8"))
    e264_post = e264m["arms"]["K10K"]["install"]["post_cells"]["g0"]
    e268m = json.loads(E268_METRICS.read_text(encoding="utf-8"))
    e268_post_c = e268m["arms"]["CONCURRENT"]["install"]["post_cells"]["g0"]
    e283m = json.loads(E283_METRICS.read_text(encoding="utf-8"))
    e283_arm = e283m["arms"]["ESTABLISHED-CONCURRENT"]["phase"]
    e283_post = e283_arm["post_cells"]["g0"]
    e283_drift = e283_arm["disp_ledger"][-1]["drift_from_fact_norm"]
    e283m_ctrl = e283m["arms"]["ESTABLISHED-SERIAL-CONTROL"]["phase"]
    e284m = json.loads(E284_METRICS.read_text(encoding="utf-8"))
    e284_sep = e284m["arms"]["SEP"]["install"]["primary_median_t100_400"]
    e284_sha = e284m["arms"]["SHA"]["install"]["primary_median_t100_400"]
    x14m = json.loads(X14_METRICS.read_text(encoding="utf-8"))
    x14_reads = x14m["adjudication"]["reads"]
    # THE DIRECT PARENT: e285's sanctuary record (the rig + the anchor)
    e285m = json.loads(E285_METRICS.read_text(encoding="utf-8"))
    e285_san = e285m["arms"]["SANCTUARY"]["phase"]
    e285_post = e285_san["post_cells"]["g0"]
    e285_S = e285_san["budget"]["S_final"]
    e285_cum = e285_san["budget"]["cum_final_norm"]
    e285_drift = e285_san["drift_from_fact_final"]["norm"]
    e285_twin = e285m["arms"]["UNPROTECTED-TWIN"]["phase"]["post_cells"]["g0"]
    e285_mile = e285_san["traj"]
    e285_r400 = e285_post / FACT_BASELINE_G0
    e285_r100 = next(t["g0_pz"] for t in e285_mile
                     if t["step"] == 100) / FACT_BASELINE_G0
    e285_d100 = next(d["drift_from_fact_norm"] for d in e285_san["disp_ledger"]
                     if d["step"] == 100)
    # THE CONTEXT PARENT: e288's controller record
    e288m = json.loads(E288_METRICS.read_text(encoding="utf-8"))
    e288_post = e288m["adjudication"]["reads"]["ERROR-GATED"]["post_g0"]
    e288_S = e288m["adjudication"]["reads"]["the_two_gates"]["budget"]["S_total"]
    vehicle = torch.load(CKPT_DIR / FACT_CK, map_location="cpu",
                         weights_only=False)
    vehicle_state = {"step": int(vehicle["step"]),
                     "traj_steps": [t["step"] for t in vehicle["traj"]],
                     "ledger_max": max(int(kk) for kk in
                                       vehicle["ledger"].keys())}
    fact_sd = {k: v.detach().clone() for k, v in vehicle["model"].items()}
    del vehicle
    G_PARENTS = {
        "e264_metrics": {"path": str(E264_METRICS),
                         "md5": md5of_norm(E264_METRICS),
                         "raw_worktree_md5": md5of(E264_METRICS),
                         "bound_md5": E264_MD5,
                         "verdict": e264m["adjudication"]["verdict"],
                         "K10K_post_g0": e264_post,
                         "note": "THE loaded fact's committed record (the "
                                 "quiet-formed threshold rung)"},
        "e268_metrics": {"path": str(E268_METRICS),
                         "md5": md5of_norm(E268_METRICS),
                         "raw_worktree_md5": md5of(E268_METRICS),
                         "bound_md5": E268_MD5,
                         "verdict": e268m["adjudication"]["verdict"],
                         "concurrent_post_g0": e268_post_c,
                         "ratio": E268_RATIO,
                         "note": "the FORMING-concurrent death (~0.0002x)"},
        "e283_metrics": {"path": str(E283_METRICS),
                         "md5": md5of_norm(E283_METRICS),
                         "raw_worktree_md5": md5of(E283_METRICS),
                         "bound_md5": E283_MD5,
                         "verdict": e283m["adjudication"]["verdict"],
                         "concurrent_post_g0": e283_post,
                         "ratio": E283_RATIO,
                         "drift_from_fact_norm": e283_drift,
                         "serial_control_post_g0":
                             e283m_ctrl["post_cells"]["g0"],
                         "note": "the unprotected reference class (the "
                                 "death floor the ladder descends from)"},
        "e284_metrics": {"path": str(E284_METRICS),
                         "md5": md5of_norm(E284_METRICS),
                         "raw_worktree_md5": md5of(E284_METRICS),
                         "bound_md5": E284_MD5,
                         "verdict": e284m["adjudication"]["verdict"],
                         "sep_primary": e284_sep, "sha_primary": e284_sha,
                         "note": "the separation record (the machinery's "
                                 "provenance)"},
        "e285_metrics": {"path": str(E285_METRICS),
                         "md5": md5of_norm(E285_METRICS),
                         "raw_worktree_md5": md5of(E285_METRICS),
                         "bound_md5": E285_MD5,
                         "verdict": e285m["adjudication"]["verdict"],
                         "sanctuary_post_g0": e285_post,
                         "sanctuary_ratio_400": e285_r400,
                         "sanctuary_ratio_100": e285_r100,
                         "sanctuary_drift_100": e285_d100,
                         "sanctuary_S_final": e285_S,
                         "sanctuary_cum_final": e285_cum,
                         "sanctuary_drift_final": e285_drift,
                         "twin_post_g0": e285_twin,
                         "twin_ratio": e285_twin / FACT_BASELINE_G0,
                         "budget_frac": E285_BUF_CITE,
                         "corpus_gen_seed": 28501,
                         "note": "THE DIRECT PARENT (the rig this cell "
                                 "ports verbatim + the 0.5x anchor point: "
                                 "20.1%-of-write realized drift -> 330x "
                                 "down, budget held; its t100 datum 7.2% "
                                 "-> 200x down bounds the threshold at "
                                 "<= 0.07x, one draw)"},
        "e288_metrics": {"path": str(E288_METRICS),
                         "md5": md5of_norm(E288_METRICS),
                         "raw_worktree_md5": md5of(E288_METRICS),
                         "bound_md5": E288_MD5,
                         "verdict": e288m["adjudication"]["verdict"],
                         "error_gated_post_g0": e288_post,
                         "error_gated_ratio": e288_post / FACT_BASELINE_G0,
                         "S_total": e288_S,
                         "budget_norm": 0.5 * 9.1788432658723,
                         "note": "THE CONTEXT PARENT (active maintenance "
                                 "works: x3.4792 by controller, budget "
                                 "87.2% — the constant this cell measures "
                                 "prices when the controller is "
                                 "NECESSARY)"},
        "x14_metrics": {"path": str(X14_METRICS),
                        "md5": md5of_norm(X14_METRICS),
                        "raw_worktree_md5": md5of(X14_METRICS),
                        "bound_md5": X14_MD5,
                        "verdict": x14m["adjudication"]["verdict"],
                        "arm_A_orthogonal_subtraction_g0":
                            x14_reads["arm_A_orthogonal_subtraction_g0"],
                        "arm_B_in_room_subtraction_g0":
                            x14_reads["arm_B_in_room_subtraction_g0"],
                        "resurrection_x_over_dead_state":
                            x14_reads["resurrection_x_over_dead_state"],
                        "note": "the directional-transport record (the "
                                "coupling constant's other side)"},
        "e273_lr_calibration": {"path": str(E273_LRCAL),
                                "md5": md5of_norm(E273_LRCAL),
                                "raw_worktree_md5": md5of(E273_LRCAL),
                                "bound_md5": E273_LRCAL_MD5,
                                "lr_sgd": E273_LR_SGD,
                                "note": "LR_STABLE's provenance record"},
        "the_fact": {"path": f"runs/checkpoints/{FACT_CK}",
                     "md5": md5of(CKPT_DIR / FACT_CK),
                     "bound_md5": FACT_MD5,
                     "size": (CKPT_DIR / FACT_CK).stat().st_size,
                     "bound_size": FACT_SIZE, "state": vehicle_state,
                     "note": "the established fact itself (the committed "
                             "serial 10k rung's final state)"},
        "e264_rooms": {"path": f"runs/checkpoints/{ROOMS264_CK}",
                       "md5": md5of(CKPT_DIR / ROOMS264_CK),
                       "bound_md5": ROOMS264_MD5,
                       "note": "THE ROOM FILE (the fact's own room; the D/S "
                               "bit-bind itself is G_ROOMK10K)"},
        "e246_span": {"path": f"runs/checkpoints/{SPAN_CK}",
                      "md5": md5of(CKPT_DIR / SPAN_CK),
                      "bound_md5": E261.E246_SPAN_MD5},
        "e258_vmap": {"path": f"runs/checkpoints/{VMAP_CK}",
                      "md5": md5of(CKPT_DIR / VMAP_CK)},
        "hardbound": {
            "e283_verdict": E283_VERDICT,
            "e283_post": E283_POST, "e283_ratio": E283_RATIO,
            "e283_drift": E283_DRIFT, "e283_write_norm": E283_WRITE_NORM,
            "e284_verdict": E284_VERDICT,
            "e284_sep_primary": E284_SEP_PRIMARY,
            "e284_sha_primary": E284_SHA_PRIMARY,
            "e285_verdict": E285_VERDICT, "e285_post": E285_POST,
            "e285_ratio_400": E285_RATIO_400,
            "e285_ratio_100": E285_RATIO_100,
            "e285_drift_100": E285_DRIFT_100,
            "e285_S_final": E285_S_FINAL,
            "e285_cum_final": E285_CUM_FINAL,
            "e285_drift_final": E285_DRIFT_FINAL,
            "e288_verdict": E288_VERDICT, "e288_post": E288_POST,
            "e288_ratio": E288_RATIO, "e288_S_total": E288_S_TOTAL,
            "x14_verdict": X14_VERDICT, "x14_arm_A": X14_ARM_A,
            "x14_arm_B": X14_ARM_B, "x14_resurrection_x": X14_RESURRECTION_X,
            "fact_md5": FACT_MD5, "fact_step": FACT_STEP,
            "fact_traj_steps": FACT_TRAJ_STEPS,
            "fact_ledger_max": FACT_LEDGER_MAX},
        "pass": bool(
            e264m["adjudication"]["verdict"] == "SHARP-THRESHOLD"
            and abs(e264_post - FACT_BASELINE_G0) < 1e-12
            and e268m["adjudication"]["verdict"] == E268_VERDICT
            and abs(e268_post_c - E268_CONCURRENT_POST) < 1e-12
            and e283m["adjudication"]["verdict"] == E283_VERDICT
            and abs(e283_post - E283_POST) < 1e-12
            and abs(e283_drift - E283_DRIFT) < 1e-9
            and abs(e283m_ctrl["post_cells"]["g0"]
                    - FACT_BASELINE_G0) < 1e-9
            and e284m["adjudication"]["verdict"] == E284_VERDICT
            and abs(e284_sep - E284_SEP_PRIMARY) < 1e-12
            and abs(e284_sha - E284_SHA_PRIMARY) < 1e-12
            and e285m["adjudication"]["verdict"] == E285_VERDICT
            and abs(e285_post - E285_POST) < 1e-12
            and abs(e285_r400 - E285_RATIO_400) < 1e-12
            and abs(e285_r100 - E285_RATIO_100) < 1e-12
            and abs(e285_d100 - E285_DRIFT_100) < 1e-9
            and abs(e285_S - E285_S_FINAL) < 1e-9
            and abs(e285_cum - E285_CUM_FINAL) < 1e-9
            and abs(e285_drift - E285_DRIFT_FINAL) < 1e-9
            and e288m["adjudication"]["verdict"] == E288_VERDICT
            and abs(e288_post - E288_POST) < 1e-12
            and abs(e288_S - E288_S_TOTAL) < 1e-9
            and x14m["adjudication"]["verdict"] == X14_VERDICT
            and md5of_norm(E264_METRICS) == E264_MD5
            and md5of_norm(E268_METRICS) == E268_MD5
            and md5of_norm(E283_METRICS) == E283_MD5
            and md5of_norm(E284_METRICS) == E284_MD5
            and md5of_norm(E285_METRICS) == E285_MD5
            and md5of_norm(E288_METRICS) == E288_MD5
            and md5of_norm(X14_METRICS) == X14_MD5
            and md5of_norm(E273_LRCAL) == E273_LRCAL_MD5
            and md5of(CKPT_DIR / FACT_CK) == FACT_MD5
            and (CKPT_DIR / FACT_CK).stat().st_size == FACT_SIZE
            and vehicle_state["step"] == FACT_STEP
            and vehicle_state["traj_steps"] == FACT_TRAJ_STEPS
            and vehicle_state["ledger_max"] == FACT_LEDGER_MAX
            and md5of(CKPT_DIR / ROOMS264_CK) == ROOMS264_MD5
            and md5of(CKPT_DIR / SPAN_CK) == E261.E246_SPAN_MD5
            and (SMOKE or LADDER[0][0] == 10_000)),
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log(f"P0b: G_PARENTS PASS — e285 {E285_VERDICT} (post {e285_post:.2e} = "
        f"x{e285_r400:.5f}, S {e285_S:.4f}, cum {e285_cum:.4f} = 20.1% of "
        f"the write, drift {e285_drift:.4f}; t100 x{e285_r100:.5f} at drift "
        f"{e285_d100:.4f} = 7.2%); e288 {E288_VERDICT} (x"
        f"{e288_post / FACT_BASELINE_G0:.4f}, S {e288_S:.4f}); e283 "
        f"{E283_VERDICT} (x{E283_RATIO:.2e}, drift {e283_drift:.2f} vs the "
        f"write's {E283_WRITE_NORM:.2f}); the fact ckpt md5/size/step-bound")
    write_partial("P0b parents hard-bound (e285's anchor + e288's record + "
                  "the fact ckpt loaded)")
    del e264m, e268m, e283m, e284m, x14m, e285m, e288m

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
        "tol": G_READ_TOL,
        "flat_md5": hashlib.md5(theta_root.numpy().tobytes()).hexdigest(),
        "pass": bool(n_par == GB.G1B_PARAMS
                     and abs(root_read - E261.G1C_ROOT_GM12) < G_READ_TOL)}
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
        "form": "the K10K room certified (fp64 CPU, "
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

    # ---- G_ROOMK10K: bit-identity vs e264's committed K10K room ---------
    rooms264 = torch.load(CKPT_DIR / ROOMS264_CK, map_location="cpu",
                          weights_only=False)

    def _to_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    k10k_name = RUNG[LADDER[0][0]]
    if not SMOKE:
        D264 = _to_np(rooms264["model"]["K10K"]["D_int8"]).astype(np.float64)
        S264 = _to_np(rooms264["model"]["K10K"]["S"])
        D_mine = rooms.rooms[k10k_name].D
        S_mine = rooms.rooms[k10k_name].S
        G_ROOMK10K = {
            "form": "the room == the fact's own committed K10K room (seeds "
                    "26113/26114 at k=10,000): the +-1 diagonal and the "
                    "index set bit-identical to e264_rooms.pt's stored "
                    "K10K D/S (exact equality)",
            "D_bit_equal": bool(np.array_equal(D_mine, D264)),
            "S_bit_equal": bool(np.array_equal(S_mine, S264)),
            "e264_rooms_md5": md5of(CKPT_DIR / ROOMS264_CK),
            "pass": bool(np.array_equal(D_mine, D264)
                         and np.array_equal(S_mine, S264)
                         and int(rooms264["model"]["K10K"]["k"])
                         == LADDER[0][0]
                         and list(rooms264["model"]["K10K"]["seeds"])
                         == [LADDER[0][1], LADDER[0][2]]),
        }
        del rooms264
    else:
        G_ROOMK10K = {
            "form": "SMOKE: the room shares the seed pair (26113/26114) at "
                    "smoke k — no committed record at this k; the bit-bind "
                    "is VACUOUS (explicit pass, disclosed)",
            "pass": True, "vacuous": True,
        }
        del rooms264
    assert G_ROOMK10K["pass"], f"K10K room bind failed: {G_ROOMK10K}"
    metrics["gates"]["G_ROOMK10K"] = G_ROOMK10K
    log(f"P1 G_ROOMK10K: the fact's room "
        f"{('bit-identical to e264_rooms.pt (D/S exact)' if not SMOKE else 'SMOKE-vacuous')}: "
        f"PASS")

    rooms_ck = save_ckpt(
        "e290_rooms",
        {k10k_name: {"D_int8": rooms.rooms[k10k_name].D.astype(np.int8),
                     "S": rooms.rooms[k10k_name].S,
                     "k": LADDER[0][0], "seeds": [LADDER[0][1], LADDER[0][2]]}},
        {"desc": "e290's room (the fact's own room): the committed K10K "
                 "room (seeds 26113/26114), rebuilt + bit-gated vs "
                 "e264_rooms.pt",
         "ladder": [LADDER[0][0]], "n": N, "span_rank": rooms.r_span,
         "cert": {kk: vv for kk, vv in cert["per_rung"][k10k_name].items()
                  if not isinstance(vv, list)}})
    metrics["rooms"] = {
        "vehicle": {"k": LADDER[0][0], "name": k10k_name,
                    "seeds": [LADDER[0][1], LADDER[0][2]],
                    "k_fraction_of_N": LADDER[0][0] / N,
                    "bit_bound_to": f"runs/checkpoints/{ROOMS264_CK} "
                                    "(e264's committed K10K room — the "
                                    "fact's own room)"},
        "cert_probes_seed": E261.CERT_SEED,
        "certification": cert,
        "vmap_source": f"runs/checkpoints/{VMAP_CK} (e258's committed "
                       "2.74M v-map; LOADED, not re-run)",
        "span_source": f"runs/checkpoints/{SPAN_CK} (e246's committed "
                       f"LATE span; rank {rooms.r_span})",
        "checkpoint": rooms_ck,
    }
    log(f"P1 THE ROOM: {k10k_name} (k={LADDER[0][0]}): BUILT + CERTIFIED + "
        f"BIT-BOUND (the fact's own writing room)")
    write_partial("P1 the room built (parents bound + v-map loaded + span "
                  "loaded + certification + bit-bind)")

    # ---- G_LR_BIND: the lr/momentum provenance bind (e280/e284/e285) ---
    lr_stable_runtime = SGD_STABLE_FACTOR * float(json.loads(
        E273_LRCAL.read_text(encoding="utf-8"))["lr_sgd"])
    G_LR_BIND = {
        "form": "the rung form's nominal lr := e273's STABLE RIDER POINT "
                "(e280/e284/e285's committed class) — LR_STABLE = x0.01 x "
                "LR_SGD_matched, re-derived at runtime from the md5-bound "
                "runs/e273/lr_calibration.json and asserted == the frozen "
                "literal; momentum 0.9, wd 0.0 EXACTLY (e273's stable "
                "rider); the nominal schedule LR_STABLE x "
                "cosine_lr(t-1,1000); the per-rung budget cap may reduce "
                "the APPLIED lr below the schedule (disclosed per step in "
                "the lr_ledger) — the cap is part of the intervention's "
                "body",
        "lr_sgd_record": E273_LR_SGD,
        "stable_factor": SGD_STABLE_FACTOR,
        "lr_stable": LR_STABLE,
        "lr_stable_runtime": lr_stable_runtime,
        "momentum": SGD_MOMENTUM,
        "wd": SGD_WD,
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

    # ================= P2: THE FACT (loaded bit-exact + read + gated) ===
    log("=" * 78)
    fact_net = G1.evl_load(fact_sd)
    fact_flat = flat_params_cpu(fact_net)
    fact_flat_np = fact_flat.double().numpy().astype(np.float64)
    fact_flat_md5 = hashlib.md5(fact_flat.numpy().tobytes()).hexdigest()
    fact_g0 = G1.battery_cell(fact_net, g0_ids, zid)["mean_pz"]
    fact_gm12 = G1.battery_cell(fact_net, gm12_ids, zid)["mean_pz"]
    fact_gp12 = G1.battery_cell(fact_net, bat_ids[12], zid)["mean_pz"]
    fact_ce_r = G1.ce_fixed_cpu(fact_net, *r_eval_xy)
    d_fact_base = fact_flat - base_flat
    loads_fact = rooms.displacement_loads(d_fact_base, ROOM_MODE)
    write_norm = float(np.linalg.norm(fact_flat_np - base_flat_np))
    G_FACTLOAD = {
        "form": "the established fact, loaded BIT-EXACT and gated THREE "
                "ways: (1) the artifact (md5/size/step/traj/ledger — "
                "G_PARENTS), (2) the loaded state's flat-md5, (3) the "
                "behavioral read (post g0 within "
                f"{FACT_READ_TOL_G0:.0e} / gm12 within "
                f"{FACT_READ_TOL_GM12:.0e} of e264's committed literals — "
                "the family's cross-session read-determinism law)",
        "flat_md5": fact_flat_md5,
        "read_g0": {"mine": fact_g0, "committed": FACT_BASELINE_G0,
                    "abs_diff": abs(fact_g0 - FACT_BASELINE_G0)},
        "read_gm12": {"mine": fact_gm12, "committed": FACT_BASELINE_GM12,
                      "abs_diff": abs(fact_gm12 - FACT_BASELINE_GM12)},
        "read_gp12": fact_gp12, "read_ce_r": fact_ce_r,
        "write_norm": write_norm,
        "survival_ratio_denominator": "the COMMITTED literal (PRIMARY); "
                                      "the session read co-reported",
        "displacement_from_base": {
            "norm": write_norm,
            **loads_fact,
            "note": "the write's own standing displacement (the transport "
                    "channel's reference quantity; the rungs' budgets = "
                    "{0.1, 0.02, 0.004, 0.0008} x this)"},
        "pass": bool(abs(fact_g0 - FACT_BASELINE_G0) <= FACT_READ_TOL_G0
                     and abs(fact_gm12 - FACT_BASELINE_GM12)
                     <= FACT_READ_TOL_GM12),
    }
    assert G_FACTLOAD["pass"], f"G_FACTLOAD FAILED: {G_FACTLOAD}"
    metrics["gates"]["G_FACTLOAD"] = G_FACTLOAD
    log(f"P2 G_FACTLOAD: the formed fact LOADS — read g0 {fact_g0:.10f} vs "
        f"committed {FACT_BASELINE_G0:.10f} (|d| "
        f"{abs(fact_g0 - FACT_BASELINE_G0):.1e}); gm12 |d| "
        f"{abs(fact_gm12 - FACT_BASELINE_GM12):.1e}; flat md5 "
        f"{fact_flat_md5[:10]}...; the write stands "
        f"{loads_fact['in_own_room']:.4f} in-room, norm {write_norm:.4f} "
        f"(the rungs' denominator): PASS")
    metrics["the_fact"] = {
        "checkpoint": f"runs/checkpoints/{FACT_CK}", "md5": FACT_MD5,
        "committed_post_g0": FACT_BASELINE_G0, "session_read_g0": fact_g0,
        "session_read_gm12": fact_gm12, "session_read_ce_r": fact_ce_r,
        "write_norm": write_norm,
        "displacement_from_base": G_FACTLOAD["displacement_from_base"],
        "baseline_used_for_ratios": "committed (PRIMARY)",
    }
    write_partial("P2 the fact loaded bit-exact + gated (the baseline read)")
    del fact_net

    # ---- THE RUNG BUDGETS (DIAL 3's denominators, frozen at load time) --
    budgets = {}
    for arm, frac in FRAC_OF.items():
        bn = frac * write_norm
        if SMOKE:
            # the smoke budget is SCALED to the full run's PER-STEP SHARE
            # so the cap's binding path runs at the full-run share scale
            # (INCLUDING the smallest rung's 1.8e-05/step — the dispatch's
            # fp-precision centerpiece)
            bn = bn * PHASE_STEPS / 400
        budgets[arm] = bn
    metrics["budget"] = {
        "form": "per rung, BUDGET := frac x ||fact - base|| (the write's "
                "own norm, fp64 from the loaded fact); the geometric "
                "0.2-factor descent from e285's 0.5x anchor",
        "write_norm": write_norm,
        "budget_fracs": list(BUDGET_FRACS),
        "budget_norms": {a: budgets[a] for a in ARMS},
        "e285_anchor": {"budget_frac": E285_BUF_CITE,
                        "budget_norm": E285_BUF_CITE * write_norm,
                        "S_final": E285_S_FINAL,
                        "cum_final": E285_CUM_FINAL,
                        "ratio_400": E285_RATIO_400},
        "reservation": "EQUAL-SHARE: cap_t = ((BUDGET - S_{t-1}) / "
                       "(n_steps - t + 1)) / ||b_t||; lr_t = "
                       "min(LR_STABLE x cosine, cap_t); S += realized "
                       "||step||; triangle-guaranteed S_400 <= BUDGET and "
                       "||cum|| <= S_400 <= BUDGET",
        "slack": BUDGET_SLACK,
        "smoke_scaling": ("SMOKE: budget x 8/400 per rung — the full run's "
                          "per-step share preserved so the binding path "
                          "runs (the smallest rung at 1.8e-05/step)"
                          if SMOKE else None),
    }
    log("P2b THE RUNG BUDGETS: "
        + "; ".join(f"{a} {FRAC_OF[a]:g}x = {budgets[a]:.6f}" for a in ARMS)
        + f" (write norm {write_norm:.4f}; e285's 0.5x anchor S "
        f"{E285_S_FINAL:.4f} cum {E285_CUM_FINAL:.4f})")
    write_partial("P2b the rung budgets computed")

    # ---- G_CORPUSGEN: the corpus-generator registration ------------------
    scratch = torch.Generator().manual_seed(CORPUS_GEN_SEED)
    aj_probe = torch.randint(anchor_full.shape[0],
                             (E43.CORP_BS - E43.MIX_RANDOM,),
                             generator=scratch)
    rj_probe = torch.randint(len(train_ids) - G1.BLOCK - 1,
                             (E43.MIX_RANDOM,), generator=scratch)
    G_CORPUSGEN = {
        "form": "the REGISTERED fresh corpus stream: seed "
                f"{CORPUS_GEN_SEED} (the family's per-cell rule — "
                ".../28301/28401/28501/28801/29001), its own generator; per "
                "corpus step the draws (aj_c(16), rj_c(32)) in that order; "
                "the first step's draws logged here from a scratch "
                "generator (the drivers' cgens start identically by "
                "construction); ALL FOUR rungs draw the IDENTICAL sequence",
        "seed": CORPUS_GEN_SEED,
        "first_step_aj": [int(x) for x in aj_probe.tolist()],
        "first_step_rj": [int(x) for x in rj_probe.tolist()],
        "composition": "16 original-host anchors (the 60-window bank) + 32 "
                       "random corpus windows (contiguous train_ids "
                       "slices); full-window CE; clip 1.0 -> g_perp + the "
                       "capped lr + opt_C (the budget the rungs' ONLY "
                       "delta)",
        "pass": True,
    }
    metrics["gates"]["G_CORPUSGEN"] = G_CORPUSGEN
    log(f"P2c G_CORPUSGEN: the corpus stream REGISTERED (seed "
        f"{CORPUS_GEN_SEED}; first draws aj[:4]="
        f"{G_CORPUSGEN['first_step_aj'][:4]} "
        f"rj[:4]={G_CORPUSGEN['first_step_rj'][:4]})")
    write_partial("P2c the corpus generator registered")

    # ================= P3: THE FOUR RUNGS (the ladder) ===================
    rung_out: dict[str, dict] = {}
    metrics["arms"] = {}
    for i_arm, arm in enumerate(ARMS):
        if i_arm:
            E261.burst_cooldown(f"{ARMS[i_arm - 1]} -> {arm}")
        log("=" * 78)
        log(f"RUNG-{arm} — {ARM_DESC[arm]}")
        out = chunked_ladder_phase(
            arm, budgets[arm], G1.evl_load(fact_sd), rooms, fact_flat_np,
            base_flat_np, anchor_full, train_ids, g0_ids, gm12_ids,
            r_eval_xy, zid,
            CKPT_DIR / (f"smoke_e290_{arm}_resume.pt" if SMOKE
                        else f"e290_{arm}_resume.pt"), dev)
        rung_out[arm] = out
        sd_a = out["sd"]
        net_a = G1.evl_load(sd_a)
        cells_a = {"gm12": G1.battery_cell(net_a, gm12_ids, zid)["mean_pz"],
                   "g0": G1.battery_cell(net_a, g0_ids, zid)["mean_pz"],
                   "gp12": G1.battery_cell(net_a, bat_ids[12], zid)["mean_pz"],
                   "ce_r": G1.ce_fixed_cpu(net_a, *r_eval_xy)}
        d_final_a = flat_params_cpu(net_a) - fact_flat
        loads_final_a = rooms.displacement_loads(d_final_a, ROOM_MODE)
        del net_a
        corp_ce_a = [v["ce"] for v in out["corpus_ledger"].values()]
        corp_gn_a = [v["gn_clipped"] for v in out["corpus_ledger"].values()]
        corp_gin_a = [v["g_in_room_frac"] for v in out["corpus_ledger"].values()
                      if v.get("g_in_room_frac") is not None]
        a_ck = save_ckpt(
            f"e290_{arm}_post", sd_a,
            {"desc": f"e290 RUNG-{arm} post-phase state: the committed "
                     f"quiet-formed 10k fact + {PHASE_STEPS} budgeted "
                     f"orthogonal SGD-M corpus steps at budget "
                     f"{FRAC_OF[arm]:g}x write norm (separate buffers, "
                     f"generator {CORPUS_GEN_SEED}) — NO install, NO "
                     f"maintenance, NO cons",
             "arm": arm, "budget_frac": FRAC_OF[arm],
             "budget_norm": budgets[arm], "S": out["S"],
             "fact": f"runs/checkpoints/{FACT_CK} (md5 {FACT_MD5})",
             "rooms": rooms_ck})
        metrics["arms"][arm] = {
            "desc": ARM_DESC[arm],
            "budget_frac": FRAC_OF[arm],
            "phase": {
                "traj": out["traj"], "corpus_ledger": out["corpus_ledger"],
                "orth_ledger": out["orth_ledger"],
                "disp_ledger": out["disp_ledger"],
                "buf_ledger": out["buf_ledger"],
                "budget_ledger": out["budget_ledger"],
                "lr_ledger": out["lr_ledger"],
                "corpus_ce_median": med(corp_ce_a),
                "corpus_gn_clipped_median": med(corp_gn_a),
                "corpus_g_in_room_frac_median": med(corp_gin_a),
                "orth_max_rel_err": out["orth_max"],
                "bufsep": out["bufsep"],
                "budget": {"budget_frac": FRAC_OF[arm],
                           "budget_norm": budgets[arm], "S_final": out["S"],
                           "S_usage_frac": out["S"] / budgets[arm],
                           "cum_final_norm":
                               out["disp_ledger"][-1]["corpus_disp_cum_norm"]
                               if out["disp_ledger"] else None,
                           "n_capped": out["n_capped"],
                           "n_steps": PHASE_STEPS,
                           "capped_frac": out["n_capped"] / PHASE_STEPS,
                           "lr_applied_min": out["lr_applied_min"],
                           "lr_applied_median": out["lr_applied_median"],
                           "lr_sched_median": out["lr_sched_median"],
                           "cap_min": out["cap_min"],
                           "realized_step_min": out["realized_step_min"],
                           "share_min": out["share_min"],
                           "realized_over_share_last":
                               out["realized_over_share_last"]},
                "chunk_table": out["chunk_table"], "steps": PHASE_STEPS,
                "post_cells": cells_a,
                "drift_from_fact_final": {
                    "norm": float(np.linalg.norm(
                        d_final_a.double().numpy())), **loads_final_a},
                "checkpoint": a_ck,
                "resumed_final": bool(out.get("resumed_final", False)),
            }}
        log(f"RUNG-{arm} DONE: post g0 {cells_a['g0']:.7f} "
            f"(x{cells_a['g0'] / FACT_BASELINE_G0:.4f}) g-12 "
            f"{cells_a['gm12']:.7f} CE_R {cells_a['ce_r']:.4f} | BUDGET S "
            f"{out['S']:.6f}/{budgets[arm]:.6f} "
            f"({out['S'] / budgets[arm]:.1%}) | capped "
            f"{out['n_capped']}/{PHASE_STEPS} | lr med "
            f"{out['lr_applied_median'] if out['lr_applied_median'] is not None else float('nan'):.3e} "
            f"(sched med {out['lr_sched_median'] if out['lr_sched_median'] is not None else float('nan'):.3e}) "
            f"| ORTH max {out['orth_max']:.2e} | isolation "
            f"{out['bufsep']['isolation_checks']} checks "
            f"{out['bufsep']['isolation_violations']} violations | corpus "
            f"CE med {med(corp_ce_a):.4f} | drift "
            f"{float(np.linalg.norm(d_final_a.double().numpy())):.5f} = "
            f"{float(np.linalg.norm(d_final_a.double().numpy())) / write_norm:.4%} "
            f"of the write")
        write_partial(f"RUNG-{arm} complete (the {FRAC_OF[arm]:g}x ledger)")

    # ---- the draw-integrity texture check (non-halting) ----------------
    def _first(a):
        cl = metrics["arms"][a]["phase"]["corpus_ledger"]
        return cl.get("1", cl.get(1, {}))
    first_ce = {a: _first(a).get("ce") for a in ARMS}
    first_gn = {a: _first(a).get("gn_clipped") for a in ARMS}
    draw_ok = all(v is not None and abs(v - first_ce[ARMS[0]]) < 1e-9
                  for v in first_ce.values())
    log(f"draw-integrity (non-halting): the t1 corpus CE + clipped ||g|| "
        f"identical across all four rungs = {draw_ok} (ce "
        f"{ {k: round(v, 10) if v is not None else None for k, v in first_ce.items()} }; "
        f"gn { {k: round(v, 10) if v is not None else None for k, v in first_gn.items()} }; "
        f"bit-identical draws — the rungs' ONLY delta is the budget)")

    # ================= P4: the instantiated gates (PER RUNG) =============
    G_ORTH = {"form": "per rung: the stepped corpus gradient is ENTIRELY "
                      "ORTHOGONAL to the room — max over ALL corpus steps "
                      "of ||P_room g_perp|| / ||g_perp|| < 1e-6 (checked "
                      "EVERY corpus step via the family's "
                      "orthogonalize_grads; the SRCT projector is exact)",
              "bar": ORTH_BAR, "per_arm": {}, "pass": None}
    G_BUFSEP = {"form": "per rung: (i) ISOLATION — opt_F snapshotted "
                        "bitwise before EVERY opt_C.step() and compared "
                        "after (it must stay EMPTY; any mismatch HALTS); "
                        "(ii) COMPOSITION — ||P_room buf_C||/||buf_C|| < "
                        "1e-4 per milestone (the fp-floor bar)",
                "bufC_bar": BUFSEP_ORTH_BAR, "per_arm": {}, "pass": None}
    G_BUDGET = {"form": "per rung: the equal-share per-step lr cap holds "
                        "S_400 under the rung's BUDGET "
                        "(triangle-guaranteed) and the realized cumulative "
                        f"displacement <= S <= BUDGET; slack "
                        f"{BUDGET_SLACK:.0e} absolute (fp32 accumulation, "
                        "e285's disclosed form)",
                "slack": BUDGET_SLACK, "per_arm": {}, "pass": None}
    for arm in ARMS:
        out = rung_out[arm]
        ph = metrics["arms"][arm]["phase"]
        # G_ORTH
        row_o = {"orth_max_rel_err": out["orth_max"],
                 "n_steps_checked": PHASE_STEPS,
                 "pass": bool(out["orth_max"] is not None
                              and out["orth_max"] < ORTH_BAR)}
        G_ORTH["per_arm"][arm] = row_o
        # G_BUFSEP
        bufC_fr = [b["bufC_inroom_frac"] for b in out["buf_ledger"]
                   if b.get("bufC_inroom_frac") is not None]
        optF_entries = [b.get("optF_state_entries") for b in out["buf_ledger"]]
        row_b = {
            "isolation_checks": out["bufsep"]["isolation_checks"],
            "isolation_violations": out["bufsep"]["isolation_violations"],
            "optF_state_entries_max": max(optF_entries) if optF_entries else 0,
            "optF_ever_stepped": bool(
                out["bufsep"].get("optF_ever_stepped", False)),
            "bufC_inroom_frac_max": max(bufC_fr) if bufC_fr else None,
            "bufC_inroom_frac_last": bufC_fr[-1] if bufC_fr else None,
            "pass": bool(
                out["bufsep"]["isolation_violations"] == 0
                and out["bufsep"]["isolation_checks"] == PHASE_STEPS
                and max(optF_entries if optF_entries else [0]) == 0
                and not out["bufsep"].get("optF_ever_stepped", False)
                and bufC_fr and max(bufC_fr) < BUFSEP_ORTH_BAR)}
        G_BUFSEP["per_arm"][arm] = row_b
        # G_BUDGET
        S_final = float(out["S"])
        cum_final = (out["disp_ledger"][-1]["corpus_disp_cum_norm"]
                     if out["disp_ledger"] else None)
        row_u = {
            "budget_frac": FRAC_OF[arm],
            "budget_norm": budgets[arm], "S_final": S_final,
            "S_usage_frac": S_final / budgets[arm],
            "cum_disp_final_norm": cum_final,
            "cum_usage_frac": (cum_final / budgets[arm]
                               if cum_final is not None else None),
            "drift_from_fact_final_norm": ph["drift_from_fact_final"]["norm"],
            "drift_frac_of_write":
                ph["drift_from_fact_final"]["norm"] / write_norm,
            "n_capped": out["n_capped"], "n_steps": PHASE_STEPS,
            "capped_frac": out["n_capped"] / PHASE_STEPS,
            "lr_applied_min": out["lr_applied_min"],
            "lr_applied_median": out["lr_applied_median"],
            "realized_step_min": out["realized_step_min"],
            "share_min": out["share_min"],
            "fp_checks": {
                "lr_applied_positive_every_ledger_row":
                    all(v["lr_applied"] > 0.0
                        for v in out["lr_ledger"].values()),
                "realized_step_positive_every_ledger_row":
                    all(v["realized_step_norm"] > 0.0
                        for v in out["lr_ledger"].values()),
                "S_monotone_in_budget_ledger": all(
                    b2["S"] >= b1["S"] - 1e-12
                    for b1, b2 in zip(out["budget_ledger"],
                                      out["budget_ledger"][1:]))},
            "pass": bool(S_final <= budgets[arm] + BUDGET_SLACK
                         and cum_final is not None
                         and cum_final <= budgets[arm] + BUDGET_SLACK
                         and all(v["lr_applied"] > 0.0
                                 for v in out["lr_ledger"].values()))}
        G_BUDGET["per_arm"][arm] = row_u
    G_ORTH["pass"] = bool(all(r["pass"] for r in G_ORTH["per_arm"].values()))
    G_BUFSEP["pass"] = bool(all(r["pass"]
                                for r in G_BUFSEP["per_arm"].values()))
    G_BUDGET["pass"] = bool(all(r["pass"]
                                for r in G_BUDGET["per_arm"].values()))
    metrics["gates"]["G_ORTH"] = G_ORTH
    metrics["gates"]["G_BUFSEP"] = G_BUFSEP
    metrics["gates"]["G_BUDGET"] = G_BUDGET
    assert G_ORTH["pass"], f"orthogonality gate FAILED: {G_ORTH}"
    assert G_BUFSEP["pass"], f"buffer separation gate FAILED: {G_BUFSEP}"
    assert G_BUDGET["pass"], f"budget gate FAILED: {G_BUDGET}"
    log(f"P4 GATES (all four rungs): G_ORTH PASS (worst "
        f"{max(r['orth_max_rel_err'] for r in G_ORTH['per_arm'].values()):.2e} "
        f"< {ORTH_BAR:.0e}); G_BUFSEP PASS (all isolation checks 0 "
        f"violations, opt_F EMPTY everywhere; bufC in-room worst "
        f"{max((r['bufC_inroom_frac_max'] or 0.0) for r in G_BUFSEP['per_arm'].values()):.2e} "
        f"< {BUFSEP_ORTH_BAR:.0e}); G_BUDGET PASS (per-rung S <= budget + "
        f"{BUDGET_SLACK:.0e}; worst overage "
        f"{max(r['S_usage_frac'] for r in G_BUDGET['per_arm'].values()):.6%}"
        f" of budget)")
    write_partial("P4 gates complete (ORTH + BUFSEP + BUDGET, per rung)")

    # ================= P5: ADJUDICATION (the frozen bars) ================
    ratio_400 = {a: metrics["arms"][a]["phase"]["post_cells"]["g0"]
                 / FACT_BASELINE_G0 for a in ARMS}
    ratio_mile = {a: {t["step"]: t["g0_pz"] / FACT_BASELINE_G0
                      for t in rung_out[a]["traj"]} for a in ARMS}
    holds = {a: bool(ratio_400[a] >= SURVIVE_FRAC) for a in ARMS}

    hard = dict(metrics["gates"])
    gates_pass = bool(all(g.get("pass") for g in hard.values()))

    # the rung order (descending budgets): ARMS[0] = 0.1x ... ARMS[-1] =
    # 0.0008x; monotone holds-pattern := the holding set is a SUFFIX of
    # the descending list (all rungs at-or-below the top holder hold)
    holds_seq = [holds[a] for a in ARMS]
    n_hold = sum(holds_seq)
    monotone_suffix = n_hold > 0 and all(holds_seq[i:]
                                         for i in range(len(ARMS)
                                                        - n_hold))
    top_die = ARMS[0]
    bracket_budget = None
    bracket_drift = None

    if not gates_pass:
        failed = [k for k, g in hard.items() if not g.get("pass")]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a hard gate failed — nothing adjudicated; the record is "
                  "complete for the autopsy")
    elif n_hold == 0:
        verdict = "NO-PASSIVE-THRESHOLD"
        worst_drift = min(G_BUDGET["per_arm"][a]["drift_frac_of_write"]
                          for a in ARMS)
        clause = (
            f"ALL FOUR RUNGS DIE (< {SURVIVE_FRAC:.0%}x at t400) — ratios "
            + ", ".join(f"{FRAC_OF[a]:g}x: {ratio_400[a]:.4f}x"
                        for a in ARMS)
            + f" — even at the 0.0008x budget (realized cumulative drift "
            f"{G_BUDGET['per_arm'][ARMS[-1]]['cum_disp_final_norm']:.6f} = "
            f"{G_BUDGET['per_arm'][ARMS[-1]]['cum_disp_final_norm'] / write_norm:.4%} "
            f"of the write; final drift from fact "
            f"{G_BUDGET['per_arm'][ARMS[-1]]['drift_frac_of_write']:.4%}; "
            f"the SMALLEST realized drift any rung produced this session = "
            f"{worst_drift:.4%} of the write). THE COUPLING IS EFFECTIVELY "
            f"UNBOUNDED at this resolution: NO passive drift is safe over "
            f"a 400-step moving phase; maintenance is ALWAYS necessary for "
            f"a moving organism — the strongest fragility statement; "
            f"e288's controller is then the only preservation (its "
            f"x{E288_RATIO:.4f} by active anchoring vs this ladder's best "
            f"passive x{max(ratio_400.values()):.4f})")
    elif holds[top_die]:
        verdict = "ROBUST-FLOOR"
        clause = (
            f"the 0.1x rung HOLDS (ratio {ratio_400[top_die]:.4f}x >= "
            f"{SURVIVE_FRAC:.0%}x at t400; realized drift "
            f"{G_BUDGET['per_arm'][top_die]['drift_frac_of_write']:.4%} of "
            f"the write) — contradicting e285's single-datum bound (the "
            f"draw or the phase timing mattered; e285's own stream: 7.2% "
            f"drift -> x{E285_RATIO_100:.4f} at t100, 20.1% -> "
            f"x{E285_RATIO_400:.4f} at t400); the threshold is looser than "
            f"0.07x after all. The full curve: "
            + ", ".join(f"{FRAC_OF[a]:g}x: {ratio_400[a]:.4f}x"
                        + (" (HOLDS)" if holds[a] else " (dies)")
                        for a in ARMS)
            + " — any non-monotonicity reported verbatim")
    elif n_hold > 0 and monotone_suffix:
        verdict = "TIGHT-CONSTANT"
        # the threshold bracket: largest holding frac .. smallest dying
        # frac above it (budget terms); realized-drift terms co-reported
        i_top_hold = max(i for i, a in enumerate(ARMS) if holds[a])
        arm_hi_hold = ARMS[i_top_hold]
        arm_lo_die = ARMS[i_top_hold - 1]
        bracket_budget = (FRAC_OF[arm_hi_hold], FRAC_OF[arm_lo_die])
        bracket_drift = (
            G_BUDGET["per_arm"][arm_hi_hold]["drift_frac_of_write"],
            G_BUDGET["per_arm"][arm_lo_die]["drift_frac_of_write"])
        # the curve shape (reported, never a bar)
        facs = []
        for a_lo, a_hi in zip(ARMS, ARMS[1:]):
            r_lo, r_hi = max(ratio_400[a_lo], RATIO_DEN_FLOOR), \
                max(ratio_400[a_hi], RATIO_DEN_FLOOR)
            facs.append(r_hi / r_lo)
        step_shape = (max(facs) >= 100.0
                      and max(ratio_400[a] for a in ARMS if holds[a])
                      <= 3.0 * min(ratio_400[a] for a in ARMS if holds[a]))
        shape = ("STEP (the transition to survival lands within one rung "
                 "gap — largest adjacent factor "
                 f"{max(facs):.0f}x)" if step_shape else
                 "GRADED (survival scales progressively across rungs — "
                 f"adjacent factors {', '.join(f'{f:.1f}x' for f in facs)})")
        clause = (
            f"the 0.1x rung DIES (ratio {ratio_400[top_die]:.4f}x < "
            f"{SURVIVE_FRAC:.0%}x; realized drift "
            f"{G_BUDGET['per_arm'][top_die]['drift_frac_of_write']:.4%} of "
            f"the write) AND the lower rungs hold — the threshold LOCATED "
            f"in the ladder's span: THRESHOLD BUDGET in "
            f"({bracket_budget[0]:g}x, {bracket_budget[1]:g}x] of the "
            f"write's norm {write_norm:.4f}, i.e. realized-drift terms in "
            f"({bracket_drift[0]:.4%}, {bracket_drift[1]:.4%}] of the "
            f"write. The full curve: "
            + ", ".join(f"{FRAC_OF[a]:g}x: {ratio_400[a]:.4f}x"
                        + (" (HOLDS)" if holds[a] else " (dies)")
                        for a in ARMS)
            + f". Curve shape: {shape}. e285's 0.5x anchor (x"
            f"{E285_RATIO_400:.5f}, different session's draw stream) "
            f"co-reported")
    else:
        verdict = "MIXED"
        clause = (
            f"non-monotone across rungs — the holds-pattern: "
            + ", ".join(f"{FRAC_OF[a]:g}x: {ratio_400[a]:.4f}x"
                        + (" (HOLDS)" if holds[a] else " (dies)")
                        for a in ARMS)
            + " — everything verbatim, every ledger, no inflation")

    log("=" * 78)
    log(f"E290 VERDICT: {verdict}")
    for a in ARMS:
        log(f"  RUNG-{a}: post g0 "
            f"{metrics['arms'][a]['phase']['post_cells']['g0']:.8f} "
            f"(x{ratio_400[a]:.4f}) | drift "
            f"{G_BUDGET['per_arm'][a]['drift_frac_of_write']:.4%} of write "
            f"| S {G_BUDGET['per_arm'][a]['S_usage_frac']:.1%} of budget | "
            f"milestones "
            + "; ".join(f"t{t} x{r:.4f}"
                        for t, r in ratio_mile[a].items()))
    log(f"  e285's 0.5x anchor: x{E285_RATIO_400:.5f} (S "
        f"{E285_S_FINAL:.4f}, cum {E285_CUM_FINAL:.4f} = 20.1% of write)")
    log(f"  {clause}")
    log("=" * 78)

    # the ladder's measured constant (the rung table, verbatim)
    threshold_read = {
        a: {"budget_frac": FRAC_OF[a],
            "budget_norm": budgets[a],
            "S_final": G_BUDGET["per_arm"][a]["S_final"],
            "S_usage_frac": G_BUDGET["per_arm"][a]["S_usage_frac"],
            "cum_final_norm": G_BUDGET["per_arm"][a]["cum_disp_final_norm"],
            "cum_frac_of_write":
                G_BUDGET["per_arm"][a]["cum_disp_final_norm"] / write_norm,
            "drift_final_norm":
                G_BUDGET["per_arm"][a]["drift_from_fact_final_norm"],
            "drift_frac_of_write":
                G_BUDGET["per_arm"][a]["drift_frac_of_write"],
            "ratio_t400": ratio_400[a],
            "holds": holds[a]}
        for a in ARMS}
    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "composite_order": "TEXTURE (any hard-gate failure) -> "
                           "NO-PASSIVE-THRESHOLD (all four die) -> "
                           "ROBUST-FLOOR (the 0.1x rung holds) -> "
                           "TIGHT-CONSTANT (the 0.1x rung dies AND >= 1 "
                           "lower rung holds AND monotone suffix) -> MIXED "
                           "(non-monotone) — frozen at birth",
        "gates_pass": gates_pass,
        "reads": {
            "rung_ratios_t400": ratio_400,
            "rung_holds": holds,
            "rung_ratio_milestones": ratio_mile,
            "the_threshold_read": threshold_read,
            "e285_anchor": {
                "budget_frac": E285_BUF_CITE, "ratio_400": E285_RATIO_400,
                "ratio_100": E285_RATIO_100, "drift_100": E285_DRIFT_100,
                "S_final": E285_S_FINAL, "cum_final": E285_CUM_FINAL,
                "drift_final": E285_DRIFT_FINAL,
                "note": "the committed same-form record (md5-bound in "
                        "G_PARENTS; its own draw stream seed 28501 — the "
                        "anchor is a class datum, the bracket adjudicated "
                        "on THIS session's bit-identical rungs)"},
            "the_unprotected_class": {
                "e283_committed": {"post_g0": E283_POST,
                                   "ratio": E283_RATIO,
                                   "drift": E283_DRIFT},
                "e285_same_session_twin": {"ratio": E285_TWIN_RATIO},
                "note": "the death class the ladder descends from"},
            "active_maintenance_context": {
                "e288": {"verdict": E288_VERDICT,
                         "ratio": E288_RATIO, "S_total": E288_S_TOTAL,
                         "note": "the controller's record — the constant "
                                 "this ladder measures prices when the "
                                 "controller is NECESSARY"}},
        },
        "threshold_bracket_budget": (list(bracket_budget)
                                     if bracket_budget else None),
        "threshold_bracket_realized_drift": (list(bracket_drift)
                                             if bracket_drift else None),
        "scatter_disclosure": {
            "read_determinism": "G_FACTLOAD's measured delta co-reported "
                                "(cross-session |d post g0| ~ 5e-7-1e-6 on a "
                                "bit-identical artifact — the family law)",
            "the_two_denominators": {
                "committed": FACT_BASELINE_G0,
                "session_loaded": fact_g0},
            "n1_caveat": "n=1 per rung, one lineage, one session (the "
                         "g-series standing lottery note carried verbatim); "
                         "the rungs' MONOTONE PATTERN across bit-identical "
                         "draws is the registered object",
        },
        "draw_integrity_first_batch": {"pass": bool(draw_ok),
                                       "corpus_ce": first_ce,
                                       "corpus_gn": first_gn},
        "verdict": verdict,
        "clause": clause,
        "smoke_stamp": ("SMOKE — nothing adjudicated" if SMOKE else None),
    }
    write_partial("P5 ADJUDICATED (the frozen bars)")

    # ================= P6: honesty + provenance + close ==================
    metrics["honesty"] = {
        "intervention_not_logits": (
            "the four rungs share ONE loaded artifact (the committed "
            "quiet-formed 10k write, md5/flat-md5/behavior-gated), the same "
            "milestone cadence, the same reads, and BIT-IDENTICAL corpus "
            "draws (one generator, seed 29001, one draw order) — the ONLY "
            "delta is the budget constant of the equal-share cap. The "
            "room's eigenstructure is identical across rungs BY "
            "CONSTRUCTION (one bit-gated room)."),
        "the_constant_is_measured_not_nominal": (
            "every rung's realized displacement is MEASURED (S from "
            "realized step norms; cum + drift-from-fact per milestone in "
            "fp64) — the threshold is reported in BOTH budget and "
            "realized-drift terms; the smallest rung's fp32 update "
            "rounding (realized vs share) disclosed in the lr ledger"),
        "the_anchor_is_a_class_datum": (
            "e285's 0.5x anchor ran a different draw stream (seed 28501); "
            "it is overlaid as the committed record's datum, never mixed "
            "into THIS session's bracket arithmetic"),
        "n_and_scope": ("n=1 per rung, one lineage, one session (the "
                        "g-series standing lottery caveat carried "
                        "verbatim); the rungs' MONOTONE PATTERN is the "
                        "registered object; nothing guaranteed"),
        "loads_measured_not_nominal": (
            "every read is measured: the per-step realized displacement "
            "and its norm (S), the per-milestone interval + cumulative "
            "projections (fp64), the buffer-composition ledger, the "
            "orthogonality ledger (every step), the corpus CE + "
            "clipped-grad ledgers — never nominal"),
        "nothing_guaranteed": ("the dispatch's own caveat carried: no "
                               "outcome was promised; the bars cover all "
                               "branches and the trajectories are reported "
                               "verbatim regardless"),
    }
    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "machinery_imported_from": str(E261.__file__),
        "rung_form_ported_from": str((E43.REPO / "lab" / "e285_sanctuary.py")
                                     .resolve()),
        "checkpoints": {
            "base": f"runs/checkpoints/{BASE_CK}",
            "reference_root": {"file": f"runs/checkpoints/{ROOT_CK}",
                               "flat_md5": G_ROOT["flat_md5"]},
            "the_fact": {"file": f"runs/checkpoints/{FACT_CK}",
                         "md5": FACT_MD5,
                         "loaded_flat_md5": fact_flat_md5,
                         "note": "read-only here — the established fact"},
            "span_ledger": f"runs/checkpoints/{SPAN_CK}",
            "vmap": f"runs/checkpoints/{VMAP_CK}",
            "e264_rooms": f"runs/checkpoints/{ROOMS264_CK}",
            "rooms": rooms_ck,
            "post_states": {a: metrics["arms"][a]["phase"]["checkpoint"]
                            for a in ARMS},
        },
        "machinery": {
            "ladder_phase": "THIS file's chunked_ladder_phase: e285's "
                            "chunked_sanctuary_phase VERBATIM in body (the "
                            "budget + arm tag parameterized; the committed "
                            "lab/e285_sanctuary.py NOT modified)",
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
    make_ladder_plot(RD, metrics, verdict, clause, thermal_log, budgets,
                     write_norm)
    metrics["status"] = ("COMPLETE — adjudicated (this write replaces all "
                         "PARTIAL progressive writes)")
    metrics["outputs"] = [str(RD / "metrics.json"),
                          str(RD / "e290_coupling_ladder.png"),
                          str(RD / "REPORT.md")]
    write_partial("P7 DONE (honesty + provenance + figures)")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots
def make_ladder_plot(rd, metrics, verdict, clause, thermal_log,
                     budgets, write_norm):
    """THE CELL'S HEADLINE FIGURE: the survival-vs-budget curve (log-x,
    e285's 0.5x anchor overlaid, the HOLDS bar + any threshold bracket),
    the per-rung survival trajectories, the realized-drift verification,
    the budget ledgers, the lr price + corpus CE, the thermal envelope."""
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.6))
    fr = [FRAC_OF[a] for a in ARMS]
    r400 = [metrics["arms"][a]["phase"]["post_cells"]["g0"]
            / FACT_BASELINE_G0 for a in ARMS]
    gb = metrics["gates"]["G_BUDGET"]["per_arm"]
    cum400 = [gb[a]["cum_disp_final_norm"] / write_norm for a in ARMS]
    drift400 = [gb[a]["drift_frac_of_write"] for a in ARMS]

    # (0,0) THE HEADLINE: survival vs budget (log-x), e285's anchor overlaid
    ax = axes[0, 0]
    ax.plot(fr, [max(r, 1e-5) for r in r400], "o-", lw=2.0, ms=7,
            color="tab:blue", label="this session's rungs (seed 29001)")
    ax.plot([E285_BUF_CITE], [max(E285_RATIO_400, 1e-5)], "*", ms=15,
            color="tab:green",
            label=f"e285's 0.5x anchor x{E285_RATIO_400:.4f} (seed 28501, "
                  f"committed)")
    ax.axhline(SURVIVE_FRAC, color="crimson", ls="--", lw=1.5,
               label=f"the HOLDS bar {SURVIVE_FRAC:.0%}x")
    br = metrics["adjudication"].get("threshold_bracket_budget")
    if br:
        ax.axvspan(br[0], br[1], color="gold", alpha=0.22,
                   label=f"THRESHOLD in ({br[0]:g}x, {br[1]:g}x]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total-drift BUDGET (fraction of the write's norm, log)")
    ax.set_ylabel("survival ratio at t400 (log)")
    ax.set_title("THE COUPLING-CONSTANT LADDER — survival vs budget "
                 "(the passive curve)", fontsize=9.5)
    ax.legend(fontsize=7.0)
    ax.grid(alpha=0.25, which="both")

    # (0,1) THE SURVIVAL TRAJECTORIES per rung
    ax = axes[0, 1]
    for a, col in zip(ARMS, ("tab:blue", "tab:cyan", "tab:purple",
                             "tab:olive")):
        st = metrics["arms"][a]["phase"]["traj"]
        ax.plot([t["step"] for t in st],
                [max(t["g0_pz"], 1e-6) for t in st], "o-", lw=1.6, ms=4.5,
                color=col, label=f"{FRAC_OF[a]:g}x budget")
    ax.axhline(FACT_BASELINE_G0, color="black", ls=":", lw=1.2,
               label=f"the loaded fact {FACT_BASELINE_G0:.4f}")
    ax.axhline(SURVIVE_FRAC * FACT_BASELINE_G0, color="crimson", ls="--",
               lw=1.3, label=f"the {SURVIVE_FRAC:.0%}x HOLDS bar")
    ax.set_yscale("log")
    ax.set_xlabel("corpus step t (no install, no maintenance)")
    ax.set_ylabel("g0 battery (mean p(Z), log)")
    ax.set_title("THE PURE PASSIVE CURVES at each budget "
                 "(bit-identical draws)", fontsize=9.5)
    ax.legend(fontsize=7.0)
    ax.grid(alpha=0.25, which="both")

    # (0,2) THE REALIZED-DRIFT VERIFICATION (the constant's measurement)
    ax = axes[0, 2]
    ax.plot(fr, cum400, "s-", lw=1.7, ms=6, color="tab:orange",
            label="realized cum ||disp|| / write norm at t400")
    ax.plot(fr, drift400, "^--", lw=1.5, ms=6, color="tab:red",
            label="||theta_400 - fact|| / write norm")
    ax.plot(fr, fr, ":", lw=1.2, color="gray",
            label="the nominal budget line (y = x)")
    ax.plot([E285_BUF_CITE], [E285_CUM_FINAL / write_norm], "*", ms=13,
            color="tab:green", label="e285's 0.5x anchor (cum 20.1%)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("BUDGET (fraction of the write's norm, log)")
    ax.set_ylabel("realized displacement (fraction of write norm, log)")
    ax.set_title("THE VERIFICATION — realized drift per rung (cancellation "
                 "below the triangle bound)", fontsize=9.0)
    ax.legend(fontsize=6.8)
    ax.grid(alpha=0.25, which="both")

    # (1,0) THE BUDGET LEDGERS (S vs t per rung)
    ax = axes[1, 0]
    for a, col in zip(ARMS, ("tab:blue", "tab:cyan", "tab:purple",
                             "tab:olive")):
        bl = metrics["arms"][a]["phase"]["budget_ledger"]
        ax.plot([b["step"] for b in bl], [b["S"] for b in bl], "o-", ms=4,
                lw=1.5, color=col,
                label=f"{FRAC_OF[a]:g}x: S (budget {budgets[a]:.4f})")
        ax.axhline(budgets[a], color=col, ls=":", lw=0.9, alpha=0.6)
    ax.set_xlabel("corpus step t (milestone)")
    ax.set_ylabel("S = sum of realized step norms")
    ax.set_title("THE PER-RUNG BUDGET LEDGERS (machine-held; dotted = the "
                 "rung's budget)", fontsize=9.0)
    ax.legend(fontsize=6.6)
    ax.grid(alpha=0.25)

    # (1,1) THE LR PRICE + THE CORPUS CE
    ax = axes[1, 1]
    lr_med = [metrics["arms"][a]["phase"]["budget"]["lr_applied_median"]
              for a in ARMS]
    ce_med = [metrics["arms"][a]["phase"]["corpus_ce_median"] for a in ARMS]
    ax.plot(fr, [v if v else 1e-9 for v in lr_med], "o-", lw=1.7, ms=6,
            color="tab:blue", label="applied lr (median)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("BUDGET (fraction of write norm, log)")
    ax.set_ylabel("applied lr (median, log)", color="tab:blue")
    axb = ax.twinx()
    axb.plot(fr, ce_med, "s--", lw=1.5, ms=5.5, color="tab:red",
             label="corpus CE (median)")
    axb.set_ylabel("corpus CE (nats, median)", color="tab:red")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.0)
    ax.set_title("THE BUDGET'S PRICE at each rung (the smaller the budget, "
                 "the tinier the stream's lr)", fontsize=9.0)
    ax.grid(alpha=0.25, which="both")

    # (1,2) THE THERMAL ENVELOPE + THE VERDICT READOUT
    ax = axes[1, 2]
    if thermal_log:
        ax.plot([r["t"] for r in thermal_log],
                [r["temp"] for r in thermal_log],
                "-", lw=0.8, color="dimgray", alpha=0.7)
    ax.axhline(E261.TEMP_EARLY_END, color="crimson", ls=":", lw=1.0,
               label=f"burst-end margin {E261.TEMP_EARLY_END:.0f}C")
    ax.axhline(E261.TEMP_HARD, color="crimson", ls="--", lw=1.2,
               label=f"never-past line {E261.TEMP_HARD:.0f}C")
    ax.set_xlabel("run seconds")
    ax.set_ylabel("GPU temp (C) per-step polls")
    ax.legend(fontsize=7.0, loc="center left")
    ax.grid(alpha=0.25)
    mx = max((r["temp"] for r in thermal_log), default=float("nan"))
    r_txt = " / ".join(f"{FRAC_OF[a]:g}x:{ratio:.4f}"
                       for a, ratio in zip(ARMS, r400))
    ax.set_title(f"THE THERMAL ENVELOPE (max {mx:.1f}C) + t400 reads: "
                 f"{r_txt}", fontsize=8.2)

    fig.suptitle(f"E290 — THE COUPLING-CONSTANT LADDER (the passive "
                 f"floor's existence question) -> {verdict}", fontsize=11)
    fig.text(0.5, 0.005, textwrap.fill(clause, 170), ha="center",
             fontsize=7.2, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(rd / "e290_coupling_ladder.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
