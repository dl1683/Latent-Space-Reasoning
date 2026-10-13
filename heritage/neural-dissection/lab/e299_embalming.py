"""E299 — THE EMBALMING CURVE, AGE-0 HALF — a desk cell (dispatched in the
b4947c2 wave, "is death ever final?"). This docstring carries the registered
question + bars VERBATIM + the convention freezes + the operationalizations +
the pre-compute predictions, committed at birth BEFORE any compute. Adjudicate
against exactly this; no bar shopping.

THE QUESTION (verbatim): how long does a dead memory stay resuscitable by
x14's subtraction surgery? Age 0 first: the freshest corpses on disk. (This
is the desk phase; the aging continuations ride a later GPU gap.)

FROZEN BARS (verbatim from the dispatch letter):
  - ALWAYS-RESUSCITABLE: at age 0, every corpse above the write-mass floor
    resurrects >= 10x — the surgery works wherever mass survives; the aging
    continuations will set the boundary.
  - MASS-TRACKS: the resurrection factor correlates with the surviving
    in-room mass across corpses, rho >= 0.8 — the corpse keeps until the
    mass erodes; the resuscitation limit = the fill law's erosion.
  - DECOUPLED: resurrection does not track mass — the context's geometry
    matters beyond mass; x14's coupling caveat at full strength.
  - MIXED/INSTRUMENT-MISSING: disclose what states exist; the age-0 answer
    alone is the deliverable.

OPERATIONALIZATIONS (frozen HERE, at birth, before compute — the honest
reading of the bars above; every choice disclosed with its reason):
  * THE ROOM := the fact's own committed K10K room (k=10,000, seeds
    26113/26114 — e264_rooms.pt's D/S, BIT-BOUND; the SRCT projector applied
    exact in fp64 pocketfft CPU, ported by import from the committed
    lab/e261_rank_ladder.py, which is NOT modified) — x14's freeze verbatim.
  * THE STATES := t0 = e261_K10K_inst_resume.pt (the loaded fact; x14's
    G_FACTLOAD three-way bind re-run here); the corpses := the checkpointed
    t400 post states of the fact-lineage whose committed survival ratio
    < 0.5 (the family's holds bar — e290's own definition of "dies"):
      1. e283-CONCURRENT      (free AdamW;     ratio 0.0000342x; drift 14.454) — THE x14 ANCHOR
      2. e285-TWIN            (free AdamW;     ratio 0.0000147x; drift 14.432)
      3. e285-SANCTUARY       (orthogonal SGD-M; ratio 0.003032x; drift 1.845)
      4. e290-0.1X            (orthogonal;     ratio 0.009715x; drift 0.5350)
      5. e290-0.02X           (orthogonal;     ratio 0.059890x; drift 0.1277)
      6. e290-0.004X          (orthogonal;     ratio 0.319391x; drift 0.0297)
      7. e288-NAME-FIXED-TWIN (orthogonal corpus + in-room name maintenance;
                              ratio 0.162340x; drift 1.7946)
    THE ALIVE BOUNDARY := e290-0.0008X (ratio 0.767399x — HOLDS, not a
    corpse): the surgery is co-reported on it, never adjudicated as one.
  * THE AGE := 0 for every state (zero post-death steps have passed — these
    are the freshest corpses; the t100-300 milestone states were never
    saved, disclosed; the age axis is degenerate at 0 in this half and the
    KILL DEPTH axis (the drift norm) is the desk deliverable).
  * THE WRITE-MASS FLOOR := the surviving in-room mass
    ||P_room(theta_corpse - base)|| >= 50% of the t0 in-room mass
    ||P_room(theta_fact - base)|| (x14's EROSION_BAR carried verbatim).
  * THE SUBTRACTION SURGERY := x14's Arm A VERBATIM, per corpse: in fp64,
    delta = theta_corpse - theta0; P_delta = P_room(delta);
    delta_O = delta - P_delta; ARM A = theta_corpse - delta_O
    (== theta0 + P_delta — the corpus's out-of-room transport REMOVED),
    materialized fp32 (quantization residual disclosed), then the probe.
    Arm ID (theta_corpse - delta == theta0) is the per-corpse construction
    closure. NO training, NO stream, NO steps — reads + fp64 projections.
  * THE READS := the family's g0 battery (60 install-splice windows,
    p(Z) at the last position; G_SPLICE 19+41, G_BATTERY shapes — x14's
    P0 verbatim); g0 PRIMARY, gm12/gp12/CE_R co-reported; CPU fp32,
    threads 4.
  * THE RESURRECTION FACTOR := Arm A g0 / corpse g0 (x14's multiplier
    convention — the dispatch's cited 4,328x).
  * THE RECOVERED FRACTION := Arm A g0 / 0.26464763283729553 (the committed
    baseline).
  * ALWAYS-RESUSCITABLE'S SATURATION CLAUSE (frozen BEFORE compute): the
    token ">= 10x" is attainable only where baseline/corpse_g0 >= 10 (full
    restoration caps the multiplier at 1/survival — a corpse at 32%
    survival cannot show 10x even under PERFECT surgery). For corpses
    whose death is shallower than 10x (survival > 0.10), full restoration
    replaces the token: the corpse counts as resurrected iff its recovered
    fraction >= 0.90. Both readings are reported per corpse; this clause
    operationalizes the bar's own parenthetical ("the surgery works
    wherever mass survives") at birth, not after seeing the numbers.
  * MASS-TRACKS' STATISTIC (frozen): Spearman rho across the 7-corpse set
    between the RECOVERED FRACTION and the SURVIVING IN-ROOM MASS RATIO —
    the resuscitation-LIMIT reading. The bar's parenthetical names a LIMIT
    ("the resuscitation limit = the fill law's erosion"), which the
    multiplier cannot express: its denominator diverges as corpses die
    deeper with identical restorability (a corpse at 1e-6 survival with a
    dead Arm A still shows a large factor). The rho on the factor scale is
    CO-REPORTED, and if it disagrees with the primary the disagreement is
    itself disclosed as a finding — no bar moves. Bar: rho >= 0.8.
    Robustness co-report: the same Spearman with mass + fraction rounded
    to 3 decimals before ranking (the mass axis is quasi-binary at age 0 —
    see P-MASS — and the unrounded within-class ordering is 5th-decimal
    noise; the tie-coarsened variant shows whether the verdict survives).
  * DECOUPLED := the primary rho < 0.8.
  * COMPOSITE := TEXTURE (any hard-gate failure) -> the two-axis verdict:
    the headline (ALWAYS-RESUSCITABLE yes/no per the frozen
    operationalization above) + the mechanism (MASS-TRACKS vs DECOUPLED);
    MIXED/INSTRUMENT-MISSING is reserved for missing states — none at
    design time: all 7 named corpses + the boundary exist on disk
    (disclosed in the inventory).

REGISTERED PREDICTIONS (cited from the committed ledgers — derivable
without any new compute; the discriminating observation is the surgery):
  * P-PORT (THE INSTRUMENT CHECK): this session's Arm A on e283's post
    state must reproduce x14's committed 0.03915366902947426 (tol 2e-6)
    and the corpse read 9.04614535102155e-06 — the port is verbatim or
    the run HALTS (G_PORTX14).
  * P-ORTH: the four orthogonal-class corpses' drifts are machine-
    orthogonal to the room (committed in-room fracs 1.2e-6 .. 1.7e-5) ->
    |P(delta)| <= ~5e-6 -> Arm A ~= theta0 -> the read returns to ~the
    fact's own 0.2646 (fraction ~1.0); the factors cap at 1/survival:
    ~330x (e285-SANCTUARY), ~103x (0.1X), ~16.7x (0.02X), ~3.1x (0.004X,
    saturated).
  * P-FREE: the two free-class corpses carry ~1.64 of in-room drift
    (committed fracs 0.113-0.114 at drift ~14.44) -> Arm A = theta0 +
    1.64 in-room -> x14's committed landing 0.0392 (fraction 0.148)
    replicates on e283, and the e285 twin lands in the same class.
  * P-E288: the twin's drift is 0.59% in-room (the name-maintenance walk,
    |P(delta)| ~ 0.0105) -> fraction high (>= 0.9), factor ~5-6x
    (saturated).
  * P-MASS: the mass axis at age 0 is QUASI-BINARY (free ~0.844 vs
    orthogonal ~1.000 — the orthogonal streams preserve the room by
    construction); the primary rho is therefore expected >= 0.8 DRIVEN BY
    THE CLASS SPLIT (a between-class discrimination, not within-class —
    disclosed); the factor-scale rho is expected NEGATIVE (denominator
    saturation). The mass-vs-class collinearity CANNOT be broken at age 0
    on this set — the aging continuations (the GPU half) are what
    de-confound, and this cell registers that debt.
  * THE NULL MODEL (T269's clock, the dispatch's (d)): T269/R69 fitted a
    constant geometric clock on the HOLDING rung's read — survival ~
    2^(-t/1040). The dispatch phrases the null as the WRITE'S IN-ROOM
    MASS decaying with that half-life. This cell measures, per free-class
    corpse, the implied MASS clock T_mass = 400*ln2/ln(m0/m_c) (from the
    committed-ledger-derived masses, re-measured on the loaded states) and
    reports the TWO-CLOCK COMPARISON: if T_mass ~ 1040 the mass and read
    clocks are one; if T_mass >> 1040 the read dies faster than the mass
    erodes and MASS-TRACKS' aging prediction is the SLOWER clock — the
    discriminating prediction handed to the GPU half: under MASS-TRACKS
    the resurrection fraction's aging half-life ~= T_mass; under the read
    clock it is ~1040; at age 0 both predict zero additional decay (every
    corpse measured at its t400 snapshot potential).

HARD GATES := {G_NAMEFREE, G_SPLICE, G_BATTERY, G_PARENTS, G_BASE, G_ROOT,
G_VMBIND, G_SPANBIND, G_PROJ, G_ROOMK10K, G_FACTLOAD, G_CORPSELOAD (all 8
states), G_PORTX14} — a failure HALTS (nothing adjudicated).

COMPUTE ENVELOPE: CPU-ONLY desk cell — torch threads 4, pocketfft workers
2, NO GPU ops (the e261 import carries the family cuda-availability
assert; no CUDA tensor is ever created), NO envelope-log writes, no bursts
(nothing trains); timestamps datetime.now(UTC) only (common.now_iso law
re-implemented locally, x14's form).

Outputs: runs/e299/{metrics.json (PROGRESSIVE), e299_embalming.png,
REPORT.md, run.log (gitignored)}. No NOTES/THINKING/QUEUE/STATE edits
(dispatch; the coordinator folds). Birth commit BEFORE compute; final
commit AND push.

Run:  cd lab && python e299_embalming.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
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

import common                                          # noqa: E402
from common import CharCorpus, run_dir, save_json      # noqa: E402

import e043_install as E43                             # noqa: E402 (REPO,
                                                      # find_occ, SPLICE_RNG,
                                                      # jsonable)
import g1b_continuity as GB                            # noqa: E402 — MUST be
                                                      # imported BEFORE G1
import g1_anchored_ball as G1                          # noqa: E402

import e261_rank_ladder as E261                        # noqa: E402 — THE
                                                      # MACHINERY (SRCT +
                                                      # LadderRooms), PORTED
                                                      # WHOLE BY IMPORT (the
                                                      # committed file is NOT
                                                      # modified; its module
                                                      # import carries the
                                                      # family cuda assert —
                                                      # NO cuda op runs here)

torch.set_num_threads(4)           # the CPU lane's whole budget (dispatch)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from scipy.stats import pearsonr, spearmanr           # noqa: E402

SMOKE = False                     # desk cell: deterministic, cheap — no smoke
CPU = torch.device("cpu")
NAME = "e299"

T0 = time.time()
RD = run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")
HEAD0: str | None = None                  # the birth head (set in __main__)


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


G1.log = log                                          # unify the timeline

# ---- THE REBINDING (the e268/x14 convention): the ported machinery resolves
# its module globals at CALL TIME through e261's namespace — rebound HERE so
# the room build + certify label THIS cell.
E261.log = log
E261.NAME = NAME
E261.SMOKE = SMOKE
E261.T0 = T0

# ======================================================================
# THE CONFIG (frozen)
# ======================================================================
BASE_CK = "e001.pt"               # the 2.74M corpus base (e043/e048's own B)
ROOT_CK = "g1c_root.pt"           # the committed fresh root (THE reference)
SPAN_CK = "e246_late_span.pt"     # e246's committed LATE span (the ledger's)
VMAP_CK = "e258_vmap.pt"          # e258's committed 2.74M v-map (the ledger's)
ROOMS264_CK = "e264_rooms.pt"     # e264's committed rooms (the K10K bit-bind)
CKPT_DIR = GB.CKPT_DIR

LADDER: tuple[tuple[int, int, int], ...] = (
    (10_000, 26113, 26114),       # K10K — the fact's own room (x14's freeze)
)
RUNG = {k: "K10K" for k, _, _ in LADDER}
ROOM_MODE = "K10K"
E261.LADDER = LADDER            # the machinery's certify()/rooms read these
E261.RUNG_NAMES = RUNG

# ---- THE STATES
FACT_CK = "e261_K10K_inst_resume.pt"        # t0: the loaded established fact
FACT_MD5 = "0f6dc1cf46850ce655dfafc9c853d467"
FACT_SIZE = 32958479
FACT_STEP = 400
FACT_TRAJ_STEPS = [1, 100, 200, 300, 400]
FACT_LEDGER_MAX = 400
FACT_FLAT_MD5 = "ebebb4472725d582dd74928493f1bfb3"   # e283/x14's G_FACTLOAD

# ---- THE CORPSES (the frozen 7 + the alive boundary): key, checkpoint,
# class, committed t400 g0, committed drift norm, committed record
CORPSES: tuple[tuple[str, str, str, float, float, str], ...] = (
    ("e283-CONCURRENT", "e283_ESTABLISHED-CONCURRENT_post.pt", "free-adamw",
     9.04614535102155e-06, 14.453935847202613, "runs/e283/metrics.json"),
    ("e285-TWIN", "e285_UNPROTECTED-TWIN_post.pt", "free-adamw",
     3.895893769367831e-06, 14.432213219853237, "runs/e285/metrics.json"),
    ("e285-SANCTUARY", "e285_SANCTUARY_post.pt", "orthogonal-sgdm",
     0.0008025270071811974, 1.844509195284748, "runs/e285/metrics.json"),
    ("e290-0.1X", "e290_BUDGET-0.1X_post.pt", "orthogonal-sgdm",
     0.002571124816313386, 0.5350150074127173, "runs/e290/metrics.json"),
    ("e290-0.02X", "e290_BUDGET-0.02X_post.pt", "orthogonal-sgdm",
     0.015849657356739044, 0.12767011963107458, "runs/e290/metrics.json"),
    ("e290-0.004X", "e290_BUDGET-0.004X_post.pt", "orthogonal-sgdm",
     0.0845259428024292, 0.0296763134218502, "runs/e290/metrics.json"),
    ("e288-NAME-FIXED-TWIN", "e288_NAME-FIXED-TWIN_post.pt",
     "orthogonal+in-room-maint",
     0.04296277463436127, 1.7945658733962344, "runs/e288/metrics.json"),
)
BOUNDARY = ("e290-0.0008X", "e290_BUDGET-0.0008X_post.pt", "orthogonal-sgdm",
            0.20309039950370789, 0.006516891344157901,
            "runs/e290/metrics.json")      # ALIVE (holds 0.767399x)

# committed md5s (this design-time inventory; the file binds gate them)
CORPSE_MD5S = {
    "e283-CONCURRENT": "a98b638cbd52658e63d706cd112b26fb",   # x14's record
    "e285-TWIN": "0f2844b9160705c5628eb696fac61c03",
    "e285-SANCTUARY": "54efad8e339c663edc8297c9698b0b10",
    "e290-0.1X": "799f057b9b80f2404a4aa960dbe29030",
    "e290-0.02X": "7e34a9395fdd23cf861edbb754285ca6",
    "e290-0.004X": "dff12cae812461052fb3911237314641",
    "e288-NAME-FIXED-TWIN": "6f975204e2e1b30dad0a3ceaaf539a79",
}
BOUNDARY_MD5 = "47183720d587ff5bd3c198bc0bc14631"

# ---- the committed records, HARD-BOUND (Rule 12)
E264_METRICS = E43.REPO / "runs" / "e264" / "metrics.json"
E264_MD5 = "a42ff4786784b04cb9819a69b545e343"
FACT_BASELINE_G0 = 0.26464763283729553      # e264's committed K10K post g0
FACT_BASELINE_GM12 = 0.10525520890951157    # e264's committed K10K post gm12

E283_METRICS = E43.REPO / "runs" / "e283" / "metrics.json"
E283_MD5 = "cf5be012f636ede7b953eb9a615c60a5"
E285_METRICS = E43.REPO / "runs" / "e285" / "metrics.json"
E285_MD5 = "3f71c7b209fe9b6e9645d7504ddbb49c"
E288_METRICS = E43.REPO / "runs" / "e288" / "metrics.json"
E288_MD5 = "31bf8df55b51c8a19c48a155388050be"
E290_METRICS = E43.REPO / "runs" / "e290" / "metrics.json"
E290_MD5 = "4ca36a46b04d0045faa833f4f9a49d1f"
X14_METRICS = E43.REPO / "runs" / "x14" / "metrics.json"
X14_MD5 = "4970e27ae8c315df8762e5c3499a2be5"    # e285's committed bind
E268_METRICS = E43.REPO / "runs" / "e268" / "metrics.json"
E268_MD5 = "c1149229b7f0191943a7b8eb0442b494"
E278_METRICS = E43.REPO / "runs" / "e278" / "metrics.json"
E278_MD5 = "db14cdff1fd5021a5b255c12127ea9df"

# ---- the x14 port-check anchor (THE instrument gate)
X14_ARM_A_G0 = 0.03915366902947426           # x14's committed Arm A read
X14_CORPSE_G0 = 9.04614535102155e-06         # x14's committed corpse read
X14_FACTOR = 4328.215777016314               # x14's committed resurrection

# ---- the frozen bars + tolerances
RESURRECT_TOKEN = 10.0           # the >= 10x token (verbatim from the bar)
SATURATED_FRAC_BAR = 0.90        # full restoration where 10x unattainable
MASS_FLOOR = 0.50                # the write-mass floor (x14's EROSION_BAR)
RHO_BAR = 0.8                    # MASS-TRACKS' Spearman bar (verbatim)
FACT_READ_TOL_G0 = 2e-6          # G_FACTLOAD behavioral bars (x14's)
FACT_READ_TOL_GM12 = 1e-5
CORPSE_READ_TOL_G0 = 2e-6        # G_CORPSELOAD behavioral bars (the family
                                 # cross-session read-determinism law;
                                 # e290's own scatter disclosure 5e-7-1e-6)
CORPSE_GEOM_TOL = 1e-6           # G_CORPSELOAD geometric bar (drift norm;
                                 # x14's GEOM_TOL_NORM precedent)
G_READ_TOL = E261.G_READ_TOL     # 5e-3 (the root gate's)
ROOMS264_MD5 = "2d524655575cce00a3bc1c8770f4b211"    # e268/x14's bind

REGISTERED = {
    "question_verbatim": "how long does a dead memory stay resuscitable by "
        "x14's subtraction surgery? Age 0 first: the freshest corpses on "
        "disk.",
    "bars_verbatim": {
        "ALWAYS-RESUSCITABLE": "at age 0, every corpse above the write-mass "
            "floor resurrects >= 10x — the surgery works wherever mass "
            "survives; the aging continuations will set the boundary.",
        "MASS-TRACKS": "the resurrection factor correlates with the "
            "surviving in-room mass across corpses, rho >= 0.8 — the corpse "
            "keeps until the mass erodes; the resuscitation limit = the "
            "fill law's erosion.",
        "DECOUPLED": "resurrection does not track mass — the context's "
            "geometry matters beyond mass; x14's coupling caveat at full "
            "strength.",
        "MIXED/INSTRUMENT-MISSING": "disclose what states exist; the age-0 "
            "answer alone is the deliverable.",
    },
    "operationalizations": "frozen at birth in this script's docstring: the "
        "corpse set (7 named + the alive boundary), the room (x14's K10K "
        "freeze), the write-mass floor (= x14's EROSION_BAR 50%), the "
        "surgery (x14's Arm A verbatim in fp64, fp32 materialization), the "
        "resurrection factor (x14's multiplier) + the recovered fraction "
        "(Arm A / the committed baseline), ALWAYS-RESUSCITABLE's saturation "
        "clause (the 10x token is unattainable where survival > 0.10; full "
        "restoration fraction >= 0.90 replaces it — registered BEFORE "
        "compute), MASS-TRACKS' primary statistic (Spearman rho between the "
        "recovered fraction and the surviving in-room mass ratio — the "
        "resuscitation-LIMIT reading; the factor-scale rho co-reported), "
        "the tie-coarsened robustness co-report, and the two-axis composite "
        "(headline + mechanism).",
    "predictions_pre_compute": {
        "P-PORT": "this session's Arm A on e283's post must reproduce x14's "
            "0.03915366902947426 (tol 2e-6) and the corpse read "
            "9.04614535102155e-06 — the instrument check (G_PORTX14).",
        "P-ORTH": "orthogonal-class drifts are machine-orthogonal (committed "
            "in-room fracs 1.2e-6..1.7e-5) -> Arm A ~= theta0 -> fraction "
            "~1.0; factors cap at 1/survival (~330x, ~103x, ~16.7x, ~3.1x).",
        "P-FREE": "free-class drifts carry ~1.64 in-room -> Arm A = theta0 "
            "+ 1.64 in-room -> x14's 0.0392 landing (fraction 0.148) "
            "replicates on e283; the e285 twin lands in the same class.",
        "P-E288": "0.59% in-room drift (the maintenance walk, |P delta| "
            "~0.0105) -> fraction >= 0.9, factor ~5-6x (saturated).",
        "P-MASS": "the mass axis is QUASI-BINARY at age 0 (free ~0.844 vs "
            "orthogonal ~1.000): the primary rho is expected >= 0.8 driven "
            "by the CLASS SPLIT; the factor-scale rho is expected NEGATIVE "
            "(denominator saturation); the mass-vs-class collinearity is "
            "unbreakable at age 0 — the aging continuations de-confound.",
        "P-NULL": "T269's read clock (half-life ~1,040 steps) vs the "
            "measured MASS clock T_mass = 400*ln2/ln(m0/m_c) per free "
            "corpse: the two-clock comparison registered for the GPU half.",
    },
    "registration": "bars VERBATIM from the b4947c2 dispatch letter; "
        "operationalizations + predictions frozen in the docstring of this "
        "script, committed at birth BEFORE any compute; adjudicate against "
        "exactly this; no bar shopping.",
}

deviations: list[str] = [
    "THE AGE-0 DISCLOSURE (the dispatch's own framing): every corpse here "
    "sits at age 0 (zero post-death steps); the t100-300 milestone states "
    "of the named runs were never saved (only the t400 finals — inspected "
    "at design time), so the AGE axis is degenerate at 0 and the KILL "
    "DEPTH axis (the drift norm) is this half's deliverable. The aging "
    "continuations ride a later GPU gap (the dispatch: 'the aging "
    "continuations ride a later GPU gap').",
    "THE INVENTORY'S OUT-OF-SCOPE STATES (disclosed, not adjudicated): "
    "e286/e287/e289's post states (different arms — re-anchoring, "
    "name-maintenance, contradiction streams), e280/e284's missile posts "
    "(different vehicles/rooms at other k), and the e283 corpus resume "
    "(an optimizer artifact, not a model state) are inventoried but "
    "excluded from the surgery: the dispatch named e285 + e290 + e288 "
    "(+ x14's e283 anchor + e285's own twin, its same-session control).",
    "THE e290 RUNGS' DRIFT IN-ROOM FRACS ARE NOT ~1e-6 AT THE SMALLEST "
    "BUDGETS (design-time read of the committed ledgers): fp32 "
    "accumulation of near-cancelling orthogonal steps leaves 4.2e-6 "
    "(0.1X), 1.8e-5 (0.02X), 1.7e-4 (0.004X), 2.4e-3 (0.0008X) in-room "
    "fractions — the ABSOLUTE in-room drift stays <= 5e-6 for the three "
    "corpse rungs, so P-ORTH's ~theta0 prediction is unchanged; disclosed "
    "because the frac alone reads misleadingly at tiny drift norms.",
    "THE e288 TWIN IS A THIRD CLASS (orthogonal corpus + IN-ROOM name "
    "maintenance): its committed drift is 0.59% in-room (the maintenance "
    "walk 0.061 in-room frac of 0.172 maint displacement) — it is neither "
    "the free class (0.113 in-room) nor machine-orthogonal; carried as "
    "its own class in every table and figure.",
    "CPU-ONLY desk cell (dispatch): torch threads 4, pocketfft workers 2, "
    "no training runs, no corpus stream, no optimizer — reads + fp64 "
    "projections only; NO envelope-log writes; NO cuda tensor is ever "
    "created (the e261 import carries the family cuda-availability assert "
    "— its module-level side effects are the runs/e261/run.log "
    "append-handle open and nothing else, the x14/e283 precedent).",
    "THE CORPSE FILE-MD5s HAVE NO PRIOR COMMITS except e283's post (x14 "
    "recorded it): the other six are bound by md5s recorded HERE at design "
    "time (the inventory commit) + VALUE-BOUND at runtime (the behavioral "
    "g0 read vs each run's committed t400 literal + the geometric drift "
    "norm vs its committed ledger — the x14 value-bind precedent, the "
    "stronger bind).",
    "THE PROBES NEED NO STREAM: no install, no corpus, no optimizer runs "
    "in this cell; the splice/battery convention gates (G_SPLICE / "
    "G_BATTERY) carry the probe identity (x14's disclosed form).",
    "THE ADDITIVITY NULL CARRIES VERBATIM (x14's honesty block): the "
    "subtraction assumes component-wise additivity; LN/softmax coupling "
    "can mask resurrection — NECESSARY-not-SUFFICIENT: resurrection => "
    "the out-of-room context carried the kill; a sub-bar read never "
    "proves overwrite. Every 'resuscitated' verdict here is a lower bound "
    "on the corpse's resuscitability.",
    "n=1 per corpse, one lineage, one session, one draw of history per "
    "kill (the g-series standing lottery note carried verbatim); the "
    "DEPTH LADDER's monotone pattern across bit-identical draw streams "
    "(e290's own disclosure) is the depth axis's strength; nothing "
    "guaranteed.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
]


def md5of(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=str(E43.REPO), capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:                                       # noqa: BLE001
        return "unavailable"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def flat_params_cpu(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()]).clone()


def flat64_of(net) -> np.ndarray:
    return flat_params_cpu(net).double().numpy().astype(np.float64)


def set_flat_from64(net, x64: np.ndarray) -> float:
    """Materialize an fp64 flat into the net's fp32 parameters (the arms);
    returns the fp64->fp32 quantization residual norm (disclosed)."""
    idx, resid2 = 0, 0.0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            seg64 = x64[idx: idx + n]
            seg32 = seg64.astype(np.float32)
            d = seg32.astype(np.float64) - seg64
            resid2 += float((d * d).sum())
            p.copy_(torch.from_numpy(seg32).reshape(p.shape))
            idx += n
    assert idx == int(x64.size), f"flat mismatch {idx} vs {x64.size}"
    return math.sqrt(resid2)


def read_cells(net, g0_ids, gm12_ids, gp12_ids, zid, r_eval_xy) -> dict:
    return {"g0": G1.battery_cell(net, g0_ids, zid)["mean_pz"],
            "gm12": G1.battery_cell(net, gm12_ids, zid)["mean_pz"],
            "gp12": G1.battery_cell(net, gp12_ids, zid)["mean_pz"],
            "ce_r": G1.ce_fixed_cpu(net, *r_eval_xy)}


# ------------------------------------------------------------------ main
metrics: dict = {}


def write_partial(note: str) -> None:
    metrics["date"] = now_iso()
    metrics["phase_note"] = note
    save_json(RD / "metrics.json", E43.jsonable(metrics))
    log(f"WROTE partial metrics ({note})")


def main():
    metrics.update({
        "experiment": "e299_embalming",
        "phase": "THE EMBALMING CURVE, AGE-0 HALF — how long does a dead "
                 "memory stay resuscitable by x14's subtraction surgery? "
                 "Age 0 first: the corpse inventory, x14's Arm A surgery "
                 "per corpse, the resurrection-vs-kill-depth curve, and "
                 "T269's half-life clock as the null model — "
                 "ALWAYS-RESUSCITABLE / MASS-TRACKS / DECOUPLED",
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "envelope": {
            "device": "CPU-ONLY desk cell — torch threads 4, pocketfft "
                      "workers 2, CPU fp64 dense projections; NO GPU ops, "
                      "NO envelope-log writes, no training (reads + "
                      "projections only)",
            "trainings": "NONE — the states are loaded (the fact + the 7 "
                         "corpses + the alive boundary); nothing moves",
            "timestamps": "datetime.now(UTC) only",
        },
        "deviations": deviations,
        "builds_on": [
            "T263 / x14 (THE SUBTRACTION INTERVENTION: Arm A resurrected "
            "e283's corpse 4,328x to 14.8% of baseline; the surgery, the "
            "room freeze, the probe convention — ported VERBATIM; this "
            "cell's instrument)",
            "T269 / e290 (the coupling constant: the orthogonal-drift "
            "ladder at budgets 0.1x/0.02x/0.004x/0.0008x — the corpse "
            "depth ladder; the geometric clock half-life ~1,040 steps — "
            "the null model)",
            "T265 / e285 (the sanctuary's honest failure: the orthogonal "
            "stream preserves the write's mass 93.0% in-room while the "
            "read dies 0.0030x — the mass/read dissociation this cell "
            "measures head-on)",
            "T268 / e288 (the error-gated controller's twin: the "
            "name-maintenance in-room walk — the third corpse class)",
            "T261 / e283 (the established-fact collision: the free-class "
            "kill at drift 14.45; the t400 post state — the anchor corpse)",
            "T239 / e261 (the machinery PORTED WHOLE BY IMPORT: the SRCT "
            "projector, LadderRooms, the certification)",
        ],
        "whats_new": [
            "THE RESURRECTION CURVE ITSELF: x14 measured ONE corpse at ONE "
            "depth; this cell sweeps SEVEN corpses across three kill "
            "classes and 2.6 orders of kill depth (0.030 -> 14.45) at age "
            "0 — the record's first multi-corpse resuscitability map",
            "THE MASS-vs-RESURRECTION QUESTION ASKED INTERVENTIONALLY: "
            "does the surgery's yield track the surviving in-room mass "
            "(MASS-TRACKS) or the kill's geometry/class (DECOUPLED)? — "
            "the question x14's single-corpse coupling caveat could not "
            "ask",
            "THE TWO-CLOCK COMPARISON: T269's read clock (half-life "
            "~1,040 steps) vs the measured MASS clock implied by the "
            "free-class corpses' erosion — the discriminating prediction "
            "registered for the aging (GPU) half of e299",
        ],
        "gates": {},
    })
    log("E299 — THE EMBALMING CURVE, AGE-0 HALF -> " + str(RD))
    log(f"bars: ALWAYS-RESUSCITABLE (every floor-passing corpse: factor "
        f">= {RESURRECT_TOKEN:g}x, or fraction >= "
        f"{SATURATED_FRAC_BAR:.0%} where 10x unattainable); MASS-TRACKS "
        f"(Spearman rho >= {RHO_BAR} fraction-vs-mass); DECOUPLED (<)")
    write_partial("startup (bars + conventions registered, committed at "
                  "birth)")

    # ================= P0: the protocol rebuild (the probe identity) =====
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
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
    install_occ = host_occ[:60]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"

    bat_ids = {}
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

    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60,
                                        G1.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    metrics["gates"].update({"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                             "G_BATTERY": G_BATTERY})
    log("P0: probe gates PASS (namefree / splice 19+41 / battery shapes)")
    write_partial("P0 probe gates PASSED")

    # ================= P0b: the parents hard-bound (Rule 12) ============
    vehicle = torch.load(CKPT_DIR / FACT_CK, map_location="cpu",
                         weights_only=False)
    vehicle_state = {"step": int(vehicle["step"]),
                     "traj_steps": [t["step"] for t in vehicle["traj"]],
                     "ledger_max": max(int(kk) for kk in
                                       vehicle["ledger"].keys())}
    fact_sd = {k: v.detach().clone() for k, v in vehicle["model"].items()}
    del vehicle

    def _ck(name: str) -> Path:
        return CKPT_DIR / name

    corpse_md5_now = {key: md5of(_ck(ck))
                      for key, ck, *_ in CORPSES}
    boundary_md5_now = md5of(_ck(BOUNDARY[1]))
    md5_mismatch = {k: (v, CORPSE_MD5S[k]) for k, v in corpse_md5_now.items()
                    if v != CORPSE_MD5S[k]}
    G_PARENTS = {
        "e264_metrics": {"path": str(E264_METRICS), "md5": md5of(E264_METRICS),
                         "bound_md5": E264_MD5,
                         "K10K_post_g0": FACT_BASELINE_G0,
                         "note": "THE loaded fact's committed record"},
        "e283_metrics": {"path": str(E283_METRICS), "md5": md5of(E283_METRICS),
                         "bound_md5": E283_MD5},
        "e285_metrics": {"path": str(E285_METRICS), "md5": md5of(E285_METRICS),
                         "bound_md5": E285_MD5},
        "e288_metrics": {"path": str(E288_METRICS), "md5": md5of(E288_METRICS),
                         "bound_md5": E288_MD5},
        "e290_metrics": {"path": str(E290_METRICS), "md5": md5of(E290_METRICS),
                         "bound_md5": E290_MD5},
        "x14_metrics": {"path": str(X14_METRICS), "md5": md5of(X14_METRICS),
                        "bound_md5": X14_MD5,
                        "note": "THE SURGERY'S RECORD (the instrument's "
                                "committed literals: Arm A "
                                f"{X14_ARM_A_G0}, factor {X14_FACTOR:.1f}x)"},
        "e268_metrics": {"path": str(E268_METRICS), "md5": md5of(E268_METRICS),
                         "bound_md5": E268_MD5},
        "e278_metrics": {"path": str(E278_METRICS), "md5": md5of(E278_METRICS),
                         "bound_md5": E278_MD5},
        "the_fact": {"path": f"runs/checkpoints/{FACT_CK}",
                     "md5": md5of(_ck(FACT_CK)), "bound_md5": FACT_MD5,
                     "size": _ck(FACT_CK).stat().st_size,
                     "bound_size": FACT_SIZE, "state": vehicle_state},
        "the_corpses": {key: {"path": f"runs/checkpoints/{ck}", "md5": m,
                              "size": _ck(ck).stat().st_size}
                        for (key, ck, *_), m in zip(CORPSES,
                                                    corpse_md5_now.values())},
        "the_boundary": {"path": f"runs/checkpoints/{BOUNDARY[1]}",
                         "md5": boundary_md5_now,
                         "size": _ck(BOUNDARY[1]).stat().st_size,
                         "note": "the ALIVE boundary state (e290-0.0008X "
                                 "holds 0.767x — co-reported, never a "
                                 "corpse)"},
        "e264_rooms": {"path": f"runs/checkpoints/{ROOMS264_CK}",
                       "md5": md5of(_ck(ROOMS264_CK)),
                       "bound_md5": ROOMS264_MD5},
        "e246_span": {"path": f"runs/checkpoints/{SPAN_CK}",
                      "md5": md5of(_ck(SPAN_CK)),
                      "bound_md5": E261.E246_SPAN_MD5},
        "e258_vmap": {"path": f"runs/checkpoints/{VMAP_CK}",
                      "md5": md5of(_ck(VMAP_CK))},
        "hardbound": {
            "fact_baseline_g0": FACT_BASELINE_G0,
            "fact_baseline_gm12": FACT_BASELINE_GM12,
            "x14_arm_A_g0": X14_ARM_A_G0, "x14_corpse_g0": X14_CORPSE_G0,
            "x14_factor": X14_FACTOR},
        "pass": bool(
            md5of(E264_METRICS) == E264_MD5
            and md5of(E283_METRICS) == E283_MD5
            and md5of(E285_METRICS) == E285_MD5
            and md5of(E288_METRICS) == E288_MD5
            and md5of(E290_METRICS) == E290_MD5
            and md5of(X14_METRICS) == X14_MD5
            and md5of(E268_METRICS) == E268_MD5
            and md5of(E278_METRICS) == E278_MD5
            and md5of(_ck(FACT_CK)) == FACT_MD5
            and _ck(FACT_CK).stat().st_size == FACT_SIZE
            and vehicle_state["step"] == FACT_STEP
            and vehicle_state["traj_steps"] == FACT_TRAJ_STEPS
            and vehicle_state["ledger_max"] == FACT_LEDGER_MAX
            and not md5_mismatch
            and boundary_md5_now == BOUNDARY_MD5
            and md5of(_ck(ROOMS264_CK)) == ROOMS264_MD5
            and md5of(_ck(SPAN_CK)) == E261.E246_SPAN_MD5),
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log("P0b: G_PARENTS PASS — 8 parent records md5-bound + the fact "
        "(md5/size/step) + the 7 corpses + the boundary (design-time md5s) "
        "+ the room file + the span")
    write_partial("P0b parents hard-bound")
    del md5_mismatch

    # ---- G-BASE: the 2.74M corpus base, loaded fixed + fact-free --------
    base_net = G1.load_g1(_ck(BASE_CK))
    assert base_net.num_params() == GB.G1B_PARAMS, \
        f"base param count {base_net.num_params()} != {GB.G1B_PARAMS}"
    base_sd = {k: v.detach().clone() for k, v in base_net.state_dict().items()}
    base_gm12 = G1.battery_cell(base_net, bat_ids[-12], zid)["mean_pz"]
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
    root_net = G1.load_g1(_ck(ROOT_CK))
    n_par = root_net.num_params()
    theta_root = flat_params_cpu(root_net)
    root_read = G1.battery_cell(root_net, bat_ids[-12], zid)["mean_pz"]
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
    del root_net
    log(f"P1 G_ROOT: PASS (|d| {G_ROOT['abs_diff']:.1e})")
    write_partial("P1 G_ROOT PASSED")

    N = n_par
    base_flat_np = flat64_of(G1.evl_load(base_sd))

    vmap_art = torch.load(_ck(VMAP_CK), map_location="cpu",
                          weights_only=False)
    v_flat32 = vmap_art["model"]["v_flat_fp32"]
    v64_np = v_flat32.numpy().astype(np.float64)
    G_VMBIND = {
        "path": f"runs/checkpoints/{VMAP_CK}", "md5": md5of(_ck(VMAP_CK)),
        "meta_experiment": vmap_art.get("meta", {}).get("experiment"),
        "meta_k": vmap_art.get("meta", {}).get("k"),
        "size": int(v_flat32.numel()), "expected_size": N,
        "mean_v": float(v64_np.mean()),
        "pass": bool(vmap_art.get("meta", {}).get("experiment") == "e258"
                     and int(v_flat32.numel()) == N
                     and int(vmap_art["meta"]["k"]) == E261.E258_K_HARD),
    }
    assert G_VMBIND["pass"], f"v-map bind failed: {G_VMBIND}"
    metrics["gates"]["G_VMBIND"] = G_VMBIND
    del vmap_art

    span_art = torch.load(_ck(SPAN_CK), map_location="cpu",
                          weights_only=False)
    Vp = span_art["Vp"].contiguous()
    G_SPANBIND = {"md5": md5of(_ck(SPAN_CK)),
                  "rank": int(Vp.shape[0]), "N": int(Vp.shape[1]),
                  "meta_experiment": span_art.get("meta", {}).get("experiment"),
                  "pass": bool(md5of(_ck(SPAN_CK)) == E261.E246_SPAN_MD5
                               and int(Vp.shape[0]) == E261.E246_SPAN_RANK
                               and int(Vp.shape[1]) == N
                               and span_art.get("meta", {}).get("experiment")
                               == "e246")}
    assert G_SPANBIND["pass"], f"span bind failed: {G_SPANBIND}"
    metrics["gates"]["G_SPANBIND"] = G_SPANBIND
    del span_art

    params_ref = list(G1.evl_load(base_sd).parameters())
    rooms = E261.LadderRooms(N, LADDER, v64_np, Vp.numpy().astype(np.float64),
                             params_ref, CPU)
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
            f"{r['kept2_expect']:.6f} span-ovl {r['span_overlap_mean']:.4f}")

    # ---- G_ROOMK10K: bit-identity vs e264's committed K10K room ---------
    rooms264 = torch.load(_ck(ROOMS264_CK), map_location="cpu",
                          weights_only=False)

    def _to_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    D264 = _to_np(rooms264["model"]["K10K"]["D_int8"]).astype(np.float64)
    S264 = _to_np(rooms264["model"]["K10K"]["S"])
    room = rooms.rooms[ROOM_MODE]
    G_ROOMK10K = {
        "form": "the room == the fact's own committed K10K room (seeds "
                "26113/26114 at k=10,000): the +-1 diagonal and the index "
                "set bit-identical to e264_rooms.pt's stored K10K D/S "
                "(exact equality) — x14's freeze",
        "D_bit_equal": bool(np.array_equal(room.D, D264)),
        "S_bit_equal": bool(np.array_equal(room.S, S264)),
        "e264_rooms_md5": md5of(_ck(ROOMS264_CK)),
        "pass": bool(np.array_equal(room.D, D264)
                     and np.array_equal(room.S, S264)
                     and int(rooms264["model"]["K10K"]["k"])
                     == LADDER[0][0]
                     and list(rooms264["model"]["K10K"]["seeds"])
                     == [LADDER[0][1], LADDER[0][2]]),
    }
    del rooms264
    assert G_ROOMK10K["pass"], f"K10K room bind failed: {G_ROOMK10K}"
    metrics["gates"]["G_ROOMK10K"] = G_ROOMK10K
    metrics["rooms"] = {
        "vehicle": {"k": LADDER[0][0], "name": ROOM_MODE,
                    "seeds": [LADDER[0][1], LADDER[0][2]],
                    "k_fraction_of_N": LADDER[0][0] / N,
                    "bit_bound_to": f"runs/checkpoints/{ROOMS264_CK} "
                                    "(e264's committed K10K room — the "
                                    "fact's own room; x14's freeze)"},
        "certification": cert,
    }
    log(f"P1 THE ROOM: {ROOM_MODE} (k={LADDER[0][0]}): BUILT + CERTIFIED + "
        "BIT-BOUND")
    write_partial("P1 the room built (certified + bit-bound)")

    # ================= P2: THE FACT (t0, loaded bit-exact + gated) ======
    log("=" * 78)
    fact_net = G1.evl_load(fact_sd)
    fact_flat = flat_params_cpu(fact_net)
    fact_flat_np = fact_flat.double().numpy().astype(np.float64)
    fact_flat_md5 = hashlib.md5(fact_flat.numpy().tobytes()).hexdigest()
    fact_cells = read_cells(fact_net, bat_ids[0], bat_ids[-12], bat_ids[12],
                            zid, r_eval_xy)
    rem0 = fact_flat_np - base_flat_np                   # the write itself
    P_rem0 = room.project(rem0)
    O_rem0 = rem0 - P_rem0
    m0 = float(np.linalg.norm(P_rem0))                   # the write's t0
    o0 = float(np.linalg.norm(O_rem0))                   # in-room mass
    tot0 = float(np.linalg.norm(rem0))
    G_FACTLOAD = {
        "form": "the established fact (t0), loaded BIT-EXACT and gated "
                "THREE ways (x14's G_FACTLOAD): (1) the artifact "
                "(md5/size/step/traj/ledger — G_PARENTS), (2) the loaded "
                "state's flat-md5 vs the family's recorded bind, (3) the "
                "behavioral read (post g0 within 2e-6 / gm12 within 1e-5 "
                "of e264's committed literals)",
        "flat_md5": fact_flat_md5, "bound_flat_md5": FACT_FLAT_MD5,
        "read_g0": {"mine": fact_cells["g0"],
                    "committed": FACT_BASELINE_G0,
                    "abs_diff": abs(fact_cells["g0"] - FACT_BASELINE_G0)},
        "read_gm12": {"mine": fact_cells["gm12"],
                      "committed": FACT_BASELINE_GM12,
                      "abs_diff": abs(fact_cells["gm12"]
                                      - FACT_BASELINE_GM12)},
        "read_gp12": fact_cells["gp12"], "read_ce_r": fact_cells["ce_r"],
        "write_disposition": {
            "norm": tot0, "in_room_mass": m0, "out_room_mass": o0,
            "in_room_frac": m0 / tot0,
            "note": "the write's own standing t0 ledger — m0 is the "
                    "write-mass floor's denominator"},
        "pass": bool(fact_flat_md5 == FACT_FLAT_MD5
                     and abs(fact_cells["g0"] - FACT_BASELINE_G0)
                     <= FACT_READ_TOL_G0
                     and abs(fact_cells["gm12"] - FACT_BASELINE_GM12)
                     <= FACT_READ_TOL_GM12),
    }
    assert G_FACTLOAD["pass"], f"G_FACTLOAD FAILED: {G_FACTLOAD}"
    metrics["gates"]["G_FACTLOAD"] = G_FACTLOAD
    log(f"P2 G_FACTLOAD: t0 LOADS — read g0 {fact_cells['g0']:.10f} vs "
        f"committed {FACT_BASELINE_G0:.10f} (|d| "
        f"{abs(fact_cells['g0'] - FACT_BASELINE_G0):.1e}); in-room mass "
        f"m0 {m0:.6f}: PASS")
    write_partial("P2 the fact (t0) loaded + gated")
    del fact_net

    # ================= P3: THE INVENTORY + THE SURGERY ===================
    log("=" * 78)
    inventory = {
        "form": "the checkpointed corpse states of the fact-lineage (the "
                "loaded K10K fact's descendants), their kill depths and "
                "their ages — the dispatch's (a)",
        "the_corpses": [],
        "the_alive_boundary": None,
        "out_of_scope": [
            {"key": "e286_RE-ANCHORED / e286_SANCTUARY-TWIN / "
                    "e287_NAME-MAINTAINED / e287_SANCTUARY-TWIN / "
                    "e289_* (6 states)",
             "reason": "different arms (re-anchoring, name-maintenance, "
                       "contradiction streams) — outside the dispatch's "
                       "named set; inventoried, not operated on"},
            {"key": "e280_*_post / e284_SEP/SHA_post",
             "reason": "different vehicles/rooms (other k, other "
                       "conventions) — not this fact's lineage at K10K"},
            {"key": "e283_CONCURRENT_corpus_resume.pt + the e285/e290 "
                    "resume artifacts",
             "reason": "optimizer-state artifacts (model + optimizer), "
                       "not standalone model states; the t400 model states "
                       "are the _post.pt files used here"},
            {"key": "the t100/200/300 milestone states of every named run",
             "reason": "NEVER SAVED (only the t400 finals) — the age axis "
                       "is degenerate at 0 in this half; disclosed"},
        ],
    }
    surgery_records = []
    G_CORPSELOAD = {"form": "every corpse + the boundary bound BEHAVIORALLY "
                    "(this session's g0 read vs each run's committed t400 "
                    "literal, the cross-session read-determinism law) AND "
                    "GEOMETRICALLY (the fp64 drift norm vs each run's "
                    "committed ledger) — the x14 value-bind precedent; the "
                    "file md5s are hard-bound to this cell's design-time "
                    "inventory (G_PARENTS)",
                    "per_state": {}, "pass": None}

    def operate(key: str, ck: str, klass: str, committed_g0: float,
                committed_drift: float, committed_src: str,
                corpse: bool) -> dict:
        art = torch.load(_ck(ck), map_location="cpu", weights_only=False)
        sd = {k: v.detach().clone() for k, v in art["model"].items()}
        meta = {k: v for k, v in art.get("meta", {}).items() if k != "desc"}
        del art
        net = G1.evl_load(sd)
        flat = flat64_of(net)
        cells = read_cells(net, bat_ids[0], bat_ids[-12], bat_ids[12],
                           zid, r_eval_xy)
        del net
        delta = flat - fact_flat_np                    # THE drift (fp64)
        P_delta = room.project(delta)
        delta_O = delta - P_delta
        dn = float(np.linalg.norm(delta))
        pn = float(np.linalg.norm(P_delta))
        on = float(np.linalg.norm(delta_O))
        rem = flat - base_flat_np
        P_rem = room.project(rem)
        O_rem = rem - P_rem
        m_c = float(np.linalg.norm(P_rem))             # surviving in-room
        o_c = float(np.linalg.norm(O_rem))
        tot_c = float(np.linalg.norm(rem))
        cos_mass = float((P_rem @ P_rem0) / (m_c * m0)) if m_c > 0 else None
        id_err = float(np.max(np.abs((flat - delta) - fact_flat_np)))
        # ---- ARM A (x14 verbatim): subtract the out-of-room drift, read
        thetaA = flat - delta_O
        netA = G1.evl_load(sd)                         # buffers ride the sd
        quant = set_flat_from64(netA, thetaA)
        cellsA = read_cells(netA, bat_ids[0], bat_ids[-12], bat_ids[12],
                            zid, r_eval_xy)
        del netA
        g0c, g0A = cells["g0"], cellsA["g0"]
        factor = g0A / g0c
        fraction = g0A / FACT_BASELINE_G0
        survival = g0c / FACT_BASELINE_G0
        max_factor = FACT_BASELINE_G0 / g0c            # 1/survival
        mass_ratio = m_c / m0
        floor_pass = bool(mass_ratio >= MASS_FLOOR)
        saturated = bool(max_factor < RESURRECT_TOKEN)
        if not floor_pass:
            resurrected = False
            basis = "below the write-mass floor"
        elif factor >= RESURRECT_TOKEN:
            resurrected = True
            basis = f"factor {factor:.1f}x >= {RESURRECT_TOKEN:g}x"
        elif saturated and fraction >= SATURATED_FRAC_BAR:
            resurrected = True
            basis = (f"MULTIPLIER SATURATED (max attainable "
                     f"{max_factor:.2f}x < {RESURRECT_TOKEN:g}x at "
                     f"survival {survival:.3f}); full restoration "
                     f"fraction {fraction:.4f} >= "
                     f"{SATURATED_FRAC_BAR:.0%}")
        else:
            resurrected = False
            basis = (f"factor {factor:.2f}x < {RESURRECT_TOKEN:g}x and "
                     f"fraction {fraction:.4f} < {SATURATED_FRAC_BAR:.0%}")
        rec = {
            "key": key, "is_corpse": corpse, "class": klass,
            "checkpoint": f"runs/checkpoints/{ck}",
            "committed_record": committed_src,
            "age_steps_post_death": 0,
            "kill_depth": {"drift_norm": dn,
                           "drift_frac_of_write": dn / tot0,
                           "drift_in_room_norm": pn,
                           "drift_in_room_frac": dn and pn / dn,
                           "drift_out_room_norm": on},
            "corpse_read": cells,
            "survival_ratio": survival,
            "write_mass": {"in_room_mass": m_c, "out_room_mass": o_c,
                           "total": tot_c, "mass_ratio_vs_t0": mass_ratio,
                           "cos_surviving_vs_write_in_room": cos_mass,
                           "floor_pass": floor_pass},
            "arm_A_orthogonal_subtraction": {
                "desc": "x14's Arm A verbatim: theta_corpse - "
                        "(I-P)(theta_corpse - theta0) == theta0 + "
                        "P(delta); read g0",
                "read": cellsA, "fp32_quantization_residual": quant,
                "construction_id_closure_fp64": id_err},
            "resurrection": {
                "factor_over_dead": factor,
                "max_attainable_factor": max_factor,
                "multiplier_saturated": saturated,
                "recovered_fraction_of_baseline": fraction,
                "resurrected": resurrected, "basis": basis},
        }
        # ---- the value bind (behavioral + geometric)
        binds = {
            "read_g0": {"mine": g0c, "committed": committed_g0,
                        "abs_diff": abs(g0c - committed_g0)},
            "drift_norm": {"mine": dn, "committed": committed_drift,
                           "abs_diff": abs(dn - committed_drift)},
            "meta": meta,
            "pass": bool(abs(g0c - committed_g0) <= CORPSE_READ_TOL_G0
                         and abs(dn - committed_drift) <= CORPSE_GEOM_TOL),
        }
        G_CORPSELOAD["per_state"][key] = binds
        log(f"  {key} [{klass}]{' (BOUNDARY)' if not corpse else ''}: "
            f"corpse g0 {g0c:.6e} (committed {committed_g0:.6e}, |d| "
            f"{abs(g0c - committed_g0):.1e}); depth {dn:.4f} "
            f"(in-room {pn / dn:.2e}); mass ratio {mass_ratio:.4f}; "
            f"ARM A g0 {g0A:.6e} -> factor {factor:.1f}x "
            f"(max {max_factor:.1f}x), fraction {fraction:.4f}")
        log(f"    resurrection: {basis}")
        return rec

    for key, ck, klass, g0c, dn, src in CORPSES:
        inventory["the_corpses"].append(
            {"key": key, "checkpoint": f"runs/checkpoints/{ck}",
             "class": klass, "committed_g0": g0c, "committed_drift": dn})
        surgery_records.append(operate(key, ck, klass, g0c, dn, src, True))
        write_partial(f"P3 surgery: {key}")

    bkey, bck, bklass, bg0, bdn, bsrc = BOUNDARY
    inventory["the_alive_boundary"] = {
        "key": bkey, "checkpoint": f"runs/checkpoints/{bck}",
        "class": bklass, "committed_g0": bg0, "committed_drift": bdn,
        "note": "ALIVE (holds 0.767399x >= the family's 0.5x bar) — the "
                "surgery is co-reported on it, never adjudicated as a "
                "corpse"}
    boundary_rec = operate(bkey, bck, bklass, bg0, bdn, bsrc, False)

    G_CORPSELOAD["pass"] = bool(
        all(b["pass"] for b in G_CORPSELOAD["per_state"].values()))
    assert G_CORPSELOAD["pass"], f"G_CORPSELOAD FAILED: {G_CORPSELOAD}"
    metrics["gates"]["G_CORPSELOAD"] = G_CORPSELOAD
    metrics["corpse_inventory"] = inventory
    log(f"P3 G_CORPSELOAD: all {len(G_CORPSELOAD['per_state'])} states "
        "value-bound (behavioral + geometric): PASS")
    write_partial("P3 the surgery done (all corpses + the boundary)")

    # ---- G_PORTX14: the instrument check (e283's corpse reproduces x14) --
    rec283 = next(r for r in surgery_records if r["key"] == "e283-CONCURRENT")
    port_g0_diff = abs(rec283["arm_A_orthogonal_subtraction"]["read"]["g0"]
                       - X14_ARM_A_G0)
    port_corpse_diff = abs(rec283["corpse_read"]["g0"] - X14_CORPSE_G0)
    G_PORTX14 = {
        "form": "THE INSTRUMENT CHECK: this session's Arm A on x14's own "
                "corpse (e283's post state) must reproduce x14's committed "
                "literals — the same state, the same room, the same probe, "
                "the same fp64 arithmetic; the port is verbatim or this "
                "HALTS",
        "arm_A_g0": {"mine": rec283["arm_A_orthogonal_subtraction"]
                                 ["read"]["g0"],
                     "x14_committed": X14_ARM_A_G0,
                     "abs_diff": port_g0_diff},
        "corpse_g0": {"mine": rec283["corpse_read"]["g0"],
                      "x14_committed": X14_CORPSE_G0,
                      "abs_diff": port_corpse_diff},
        "tol": CORPSE_READ_TOL_G0,
        "pass": bool(port_g0_diff <= CORPSE_READ_TOL_G0
                     and port_corpse_diff <= CORPSE_READ_TOL_G0),
    }
    assert G_PORTX14["pass"], f"G_PORTX14 FAILED (the port drifted!): " \
                              f"{G_PORTX14}"
    metrics["gates"]["G_PORTX14"] = G_PORTX14
    log(f"P3b G_PORTX14: Arm A {G_PORTX14['arm_A_g0']['mine']:.10f} vs x14 "
        f"{X14_ARM_A_G0:.10f} (|d| {port_g0_diff:.1e}); corpse "
        f"{G_PORTX14['corpse_g0']['mine']:.6e} (|d| "
        f"{port_corpse_diff:.1e}): PASS — the port is x14 verbatim")
    write_partial("P3b the x14 port check PASSED")
    metrics["surgery"] = {"records": surgery_records, "boundary": boundary_rec}

    # ================= P4: THE CURVE + THE NULL MODEL ====================
    log("=" * 78)
    corps = surgery_records                      # the 7 corpses only
    depths = [r["kill_depth"]["drift_norm"] for r in corps]
    factors = [r["resurrection"]["factor_over_dead"] for r in corps]
    fractions = [r["resurrection"]["recovered_fraction_of_baseline"]
                 for r in corps]
    mass_ratios = [r["write_mass"]["mass_ratio_vs_t0"] for r in corps]
    survivals = [r["survival_ratio"] for r in corps]

    rho_frac, p_frac = spearmanr(fractions, mass_ratios)
    rho_fact, p_fact = spearmanr(factors, mass_ratios)
    r_frac = float(pearsonr(fractions, mass_ratios)[0])
    r_fact = float(pearsonr(factors, mass_ratios)[0])
    # the tie-coarsened robustness variant (mass + fraction to 3 decimals)
    rho_frac_c, _ = spearmanr([round(x, 3) for x in fractions],
                              [round(x, 3) for x in mass_ratios])
    n_distinct_mass = len({round(x, 3) for x in mass_ratios})

    # the two-clock comparison (the null model): T269's read clock vs the
    # implied MASS clock on the free-class corpses (the only class whose
    # mass eroded); T_mass = t * ln2 / ln(m0/m_c)
    free = [r for r in corps if r["class"] == "free-adamw"]
    mass_clocks = []
    for r in free:
        m_c = r["write_mass"]["in_room_mass"]
        ratio = r["write_mass"]["mass_ratio_vs_t0"]
        T = 400.0 * math.log(2.0) / math.log(m0 / m_c)
        mass_clocks.append({"key": r["key"], "m0": m0, "m_c": m_c,
                            "mass_ratio": ratio,
                            "implied_mass_half_life_steps": T,
                            "read_clock_half_life_steps": 1040.0,
                            "read_faster_than_mass_by": T / 1040.0})
    curve = {
        "form": "the resurrection curve at age 0: the factor + the "
                "recovered fraction vs kill depth (the drift norm; the "
                "e290 ladder gives the depth axis for free — its rungs "
                "were killed at different depths), and the "
                "mass-tracking statistic; the age axis is degenerate at 0 "
                "(disclosed — the aging continuations ride the GPU half)",
        "depth_range": [min(depths), max(depths)],
        "depth_span_orders": math.log10(max(depths) / min(depths)),
        "correlations": {
            "primary_spearman_fraction_vs_mass": {
                "rho": float(rho_frac), "p": float(p_frac),
                "bar": RHO_BAR, "n": len(corps)},
            "co_reported_spearman_factor_vs_mass": {
                "rho": float(rho_fact), "p": float(p_fact),
                "note": "the multiplier scale — expected NEGATIVE under "
                        "P-MASS (denominator saturation); never the "
                        "primary (see the operationalization)"},
            "tie_coarsened_spearman_fraction_vs_mass": {
                "rho": float(rho_frac_c),
                "note": "mass + fraction rounded to 3 decimals before "
                        "ranking (the unrounded within-class ordering is "
                        "5th-decimal noise)"},
            "pearson_fraction_vs_mass": r_frac,
            "pearson_factor_vs_mass": r_fact,
            "n_distinct_mass_values_3dp": n_distinct_mass,
            "quasi_binary_disclosure": "the mass axis at age 0 is "
                f"quasi-binary ({n_distinct_mass} distinct values at 3dp: "
                "the free class ~0.844 vs the orthogonal classes ~1.000) "
                "— the primary rho is a BETWEEN-CLASS discrimination; "
                "the collinearity (mass vs kill class) is unbreakable at "
                "age 0; the aging continuations de-confound",
        },
        "inverse_law_context": {
            "cite": "T269: survival ~ 1/drift over the orthogonal ladder "
                    "(refit slope -1.005, R2 0.974)",
            "orthogonal_loglog_slope_this_session": None,  # filled below
        },
    }
    orth = [r for r in corps if r["class"] == "orthogonal-sgdm"]
    ld = [math.log10(r["kill_depth"]["drift_norm"]) for r in orth]
    ls = [math.log10(r["survival_ratio"]) for r in orth]
    if len(orth) >= 3:
        A = np.vstack([ld, np.ones(len(ld))]).T
        slope, icpt = np.linalg.lstsq(A, ls, rcond=None)[0]
        curve["inverse_law_context"]["orthogonal_loglog_slope_this_session"] \
            = float(slope)
        curve["inverse_law_context"]["note"] = (
            f"the orthogonal corpses alone: log-log survival-vs-depth "
            f"slope {slope:.3f} (T269's refit -1.005 on the committed "
            f"rungs incl. the holding boundary); the free-class corpses "
            f"sit ~3 orders BELOW the 1/drift line — the kill classes are "
            f"different animals")
    metrics["curve"] = curve

    null_model = {
        "form": "T269's geometric clock as the null model (the dispatch's "
                "(d)): T269/R69 fitted survival ~ 2^(-t/1040) on the "
                "HOLDING rung's read; the dispatch phrases the null as "
                "the WRITE'S IN-ROOM MASS decaying with that half-life. "
                "At age 0 both clocks predict ZERO additional decay — "
                "every corpse measured at its t400 snapshot potential; "
                "the desk's job is the initial condition + whether "
                "resurrection tracks the mass at all (MASS-TRACKS) or "
                "something else (DECOUPLED)",
        "read_clock_half_life_steps": 1040.0,
        "read_clock_source": "T269 (R69's fit on e290's holding rung)",
        "mass_clock_free_class": mass_clocks,
        "two_clock_comparison": (
            f"the free-class corpses' implied MASS half-life is "
            f"{mass_clocks[0]['implied_mass_half_life_steps']:.0f}-"
            f"{mass_clocks[-1]['implied_mass_half_life_steps']:.0f} steps "
            f"vs the read's 1,040 — THE READ DIES ~"
            f"{mass_clocks[0]['implied_mass_half_life_steps'] / 1040:.2f}x "
            "FASTER THAN THE MASS ERODES under the free stream; the "
            "orthogonal classes' mass clock is INFINITE (mass preserved "
            "by construction while the read dies — e285's T265 "
            "dissociation)"),
        "aging_half_prediction": "the discriminating prediction handed to "
            "the GPU half: under MASS-TRACKS the resurrection fraction's "
            "aging half-life ~= the MASS clock (~1,600 steps on the free "
            "class); under the read-clock coupling it is ~1,040; at age 0 "
            "both predict zero additional decay (measured: the t400 "
            "snapshot potential, this cell)",
    }
    metrics["null_model"] = null_model
    log(f"P4 THE CURVE: depth {min(depths):.4f} -> {max(depths):.4f} "
        f"({curve['depth_span_orders']:.2f} orders); Spearman rho "
        f"fraction-vs-mass {rho_frac:+.4f} (bar >= {RHO_BAR}); factor-"
        f"vs-mass {rho_fact:+.4f} (co-reported); tie-coarsened "
        f"{rho_frac_c:+.4f}; distinct mass values (3dp): "
        f"{n_distinct_mass}")
    log(f"    two-clock: mass half-life "
        f"{mass_clocks[0]['implied_mass_half_life_steps']:.0f}-"
        f"{mass_clocks[-1]['implied_mass_half_life_steps']:.0f} steps vs "
        f"the read's 1,040 (the read dies faster than the mass erodes)")
    write_partial("P4 the curve + the null model")

    # ================= P5: ADJUDICATION (the frozen bars) ===============
    hard = dict(metrics["gates"])
    gates_pass = bool(all(g.get("pass") for g in hard.values()))
    floor_passers = [r for r in corps
                     if r["write_mass"]["floor_pass"]]
    all_resurrected = bool(floor_passers) and all(
        r["resurrection"]["resurrected"] for r in floor_passers)
    failed = [r["key"] for r in floor_passers
              if not r["resurrection"]["resurrected"]]
    mass_tracks = bool(rho_frac >= RHO_BAR)

    if not gates_pass:
        failed_gates = [k for k, g in hard.items() if not g.get("pass")]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed_gates)})"
        headline = mechanism = "TEXTURE"
        clause = ("a hard gate failed — nothing adjudicated; the record "
                  "is complete for the autopsy")
    else:
        headline = ("ALWAYS-RESUSCITABLE" if all_resurrected
                    else "NOT-ALWAYS-RESUSCITABLE")
        mechanism = "MASS-TRACKS" if mass_tracks else "DECOUPLED"
        verdict = f"{headline} + {mechanism}"
        sat = [r["key"] for r in floor_passers
               if r["resurrection"]["multiplier_saturated"]
               and r["resurrection"]["resurrected"]]
        if all_resurrected:
            clause = (f"every corpse above the write-mass floor "
                      f"({MASS_FLOOR:.0%} of t0 in-room mass {m0:.4f}) — "
                      f"all {len(floor_passers)} of 7 — resurrects: "
                      f"the factors run "
                      f"{min(factors):.1f}x-{max(factors):.0f}x over their "
                      f"dead reads and the recovered fractions "
                      f"{min(fractions):.3f}-{max(fractions):.4f} of "
                      "baseline; the surgery works wherever mass survives "
                      "AT AGE 0 — the boundary is the aging "
                      "continuations' to set (the GPU half)")
            if sat:
                clause += (f"; DISCLOSED: {len(sat)} corpse(s) "
                           f"({', '.join(sat)}) had their multiplier "
                           "SATURATED below the 10x token (survival too "
                           "shallow for 10x even under perfect surgery) "
                           "— counted by the registered full-restoration "
                           "clause (fraction >= 90%), both readings in "
                           "the table")
        else:
            clause = (f"corpses above the floor that did NOT resurrect: "
                      f"{', '.join(failed)} — the age-0 boundary is "
                      f"sharper than the mass floor; the trajectories "
                      "verbatim in the table")
        if mass_tracks:
            clause += (f"; the recovered fraction tracks the surviving "
                       f"in-room mass (Spearman rho {rho_frac:+.3f} >= "
                       f"{RHO_BAR}) — BUT the discrimination is "
                       f"BETWEEN-CLASS at age 0 (the mass axis is "
                       f"quasi-binary: {n_distinct_mass} values at 3dp; "
                       "the collinearity disclosure stands; the aging "
                       "continuations de-confound)")
        else:
            clause += (f"; the recovered fraction does NOT track the "
                       f"surviving mass (Spearman rho {rho_frac:+.3f} < "
                       f"{RHO_BAR}) — the context's geometry (the kill "
                       "class: what moved, not how much mass survived) "
                       "carries the resuscitability; x14's coupling "
                       "caveat at full strength")

    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "gates_pass": gates_pass,
        "reads": {
            "n_corpses": len(corps),
            "floor_passing": [r["key"] for r in floor_passers],
            "floor_bar": MASS_FLOOR,
            "all_floor_passers_resurrect": all_resurrected,
            "failed_resurrections": failed,
            "saturated_full_restoration": [
                r["key"] for r in floor_passers
                if r["resurrection"]["multiplier_saturated"]],
            "factors": dict(zip([r["key"] for r in corps], factors)),
            "fractions": dict(zip([r["key"] for r in corps], fractions)),
            "mass_ratios": dict(zip([r["key"] for r in corps],
                                    mass_ratios)),
            "survivals": dict(zip([r["key"] for r in corps], survivals)),
            "depths": dict(zip([r["key"] for r in corps], depths)),
            "spearman_fraction_vs_mass": float(rho_frac),
            "spearman_factor_vs_mass": float(rho_fact),
            "tie_coarsened_rho": float(rho_frac_c),
        },
        "verdict": verdict,
        "headline": headline,
        "mechanism": mechanism,
        "clause": clause,
    }
    log("=" * 78)
    log(f"E299 VERDICT: {verdict}")
    log(f"  {clause}")
    write_partial("P5 adjudicated")

    # ================= P6: honesty + figure + report ====================
    metrics["honesty"] = {
        "intervention_not_logits": "every resurrection is a BEHAVIORAL "
            "probe read on a constructed state: the ONLY delta between a "
            "corpse and its Arm A is the removal of the out-of-room drift "
            "component (fp64 subtraction, fp32 materialization with the "
            "quantization residual measured per corpse); the ID closure "
            "returns the fact exactly (fp64)",
        "the_additivity_null": "the subtraction assumes component-wise "
            "additivity; LN/softmax coupling can mask resurrection — "
            "NECESSARY-not-SUFFICIENT (x14's honesty block carried "
            "verbatim): every 'resuscitated' verdict here is a LOWER "
            "BOUND on the corpse's resuscitability; a sub-bar read never "
            "proves unresuscitability",
        "the_multiplier_saturation": "the factor-over-dead diverges as "
            "corpses die deeper with identical restorability — it cannot "
            "measure a resuscitation LIMIT; the recovered fraction is the "
            "limit-reading (the operationalization's reason, frozen at "
            "birth); both reported per corpse, no bar moved",
        "the_collinearity_debt": "at age 0 the surviving mass and the "
            "kill class are collinear (the orthogonal streams preserve "
            "the room BY CONSTRUCTION): MASS-TRACKS, if it fires, is a "
            "between-class discrimination on this set — the debt is "
            "registered for the aging continuations (the GPU half), "
            "where within-class mass erosion de-confounds",
        "loads_measured_not_nominal": "every mass, depth, and read is a "
            "direct fp64 projection / behavioral read of the loaded "
            "states this session; every corpse value-bound against its "
            "run's committed literals (G_CORPSELOAD) and the surgery "
            "instrument itself reproduced x14's committed Arm A "
            "(G_PORTX14, |d| "
            f"{port_g0_diff:.1e})",
        "n_and_scope": "n=1 per corpse, one lineage, one session, one "
            "draw of history per kill; the age axis is degenerate at 0 "
            "(this is the AGE-0 HALF by dispatch); nothing guaranteed",
    }
    write_partial("P6 honesty block")

    # ---- the figure -----------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5.0))
    CLR = {"free-adamw": "#8b2e2e", "orthogonal-sgdm": "#1a6faf",
           "orthogonal+in-room-maint": "#c2500d"}

    # (a) the curve: fraction + factor vs kill depth
    for r in corps:
        c = CLR[r["class"]]
        d = r["kill_depth"]["drift_norm"]
        ax1.plot(d, r["resurrection"]
                 ["recovered_fraction_of_baseline"], "o", color=c,
                 ms=9 if r["is_corpse"] else 5, zorder=5)
        ax1.annotate(r["key"].replace("-", "\n"), (d, r["resurrection"]
                     ["recovered_fraction_of_baseline"]),
                     textcoords="offset points", xytext=(0, 10),
                     fontsize=6, color=c, ha="center")
        ax1.plot(d, r["survival_ratio"], "o", mfc="none", mec=c, ms=5,
                 alpha=0.6)
    ax1.plot(boundary_rec["kill_depth"]["drift_norm"],
             boundary_rec["survival_ratio"], "o", mfc="none", mec="#777777",
             ms=6)
    ax1.annotate("0.0008X\n(alive)", (boundary_rec["kill_depth"]
                 ["drift_norm"], boundary_rec["survival_ratio"]),
                 textcoords="offset points", xytext=(0, -16), fontsize=6,
                 color="#777777", ha="center")
    ax1.axhline(1.0, color="#2e8b57", ls="--", lw=1.2)
    ax1.text(0.02, 1.03, "the fact (baseline 1.0)", fontsize=7,
             color="#2e8b57", transform=ax1.get_yaxis_transform())
    ax1.set_xscale("log")
    ax1.set_ylim(-0.06, 1.15)
    ax1.set_xlabel("kill depth: ||theta_corpse - theta0|| (log; the e290 "
                   "ladder gives the depth axis)")
    ax1.set_ylabel("recovered fraction of baseline (filled)\n+ the "
                   "corpse's own survival (open)")
    ax1.set_title("(a) THE RESURRECTION CURVE (age 0)\n"
                  f"verdict: {verdict}", fontsize=10)
    ax1.grid(alpha=0.25, which="both")
    for klass, c in CLR.items():
        ax1.plot([], [], "o", color=c, label=klass)
    ax1.legend(fontsize=7, loc="lower left")

    # (b) fraction vs surviving mass ratio
    for r in corps:
        c = CLR[r["class"]]
        ax2.plot(r["write_mass"]["mass_ratio_vs_t0"],
                 r["resurrection"]["recovered_fraction_of_baseline"], "o",
                 color=c, ms=9, zorder=5)
        ax2.annotate(r["key"], (r["write_mass"]["mass_ratio_vs_t0"],
                     r["resurrection"]
                     ["recovered_fraction_of_baseline"]),
                     textcoords="offset points", xytext=(4, 4),
                     fontsize=6.5, color=c)
    ax2.axhline(SATURATED_FRAC_BAR, color="gray", ls=":", lw=1.2)
    ax2.text(0.805, SATURATED_FRAC_BAR + 0.015,
             f"full-restoration bar {SATURATED_FRAC_BAR:.0%}", fontsize=7,
             color="gray")
    ax2.axvline(MASS_FLOOR, color="#8b2e2e", ls=":", lw=1.2)
    ax2.text(MASS_FLOOR + 0.004, 0.05, f"mass floor {MASS_FLOOR:.0%}",
             fontsize=7, color="#8b2e2e", rotation=90, va="bottom")
    ax2.set_xlabel("surviving in-room mass ||P(theta-base)|| / m0")
    ax2.set_ylabel("recovered fraction of baseline")
    ax2.set_title(f"(b) MASS vs RESURRECTION\n"
                  f"Spearman rho {rho_frac:+.3f} (bar >= {RHO_BAR}); "
                  f"factor-scale {rho_fact:+.3f}; {n_distinct_mass} "
                  f"distinct mass values (3dp)", fontsize=10)
    ax2.grid(alpha=0.25)

    # (c) per-corpse bars: the dead read vs Arm A (log)
    keys = [r["key"] for r in corps] + [boundary_rec["key"]]
    ypos = list(range(len(keys)))[::-1]
    for y, r in zip(ypos, corps + [boundary_rec]):
        c = CLR[r["class"]]
        ax3.barh(y + 0.18, max(r["corpse_read"]["g0"], 1e-9), height=0.32,
                 color="#555555", alpha=0.8)
        ax3.barh(y - 0.18,
                 max(r["arm_A_orthogonal_subtraction"]["read"]["g0"],
                     1e-9), height=0.32, color=c, alpha=0.9)
        fx = r["resurrection"]["factor_over_dead"]
        fr = r["resurrection"]["recovered_fraction_of_baseline"]
        ax3.text(max(r["arm_A_orthogonal_subtraction"]["read"]["g0"],
                     1e-9) * 1.4, y - 0.18,
                 f" {fx:.1f}x ({fr:.0%})", va="center", fontsize=6.5,
                 color=c)
    ax3.set_yticks(ypos, [k.replace("-", "\n") for k in keys], fontsize=6.5)
    ax3.axvline(FACT_BASELINE_G0, color="#2e8b57", ls="--", lw=1.2)
    ax3.text(FACT_BASELINE_G0 * 1.1, len(keys) - 0.4,
             "the fact\n0.2646", fontsize=7, color="#2e8b57")
    ax3.set_xscale("log")
    ax3.set_xlim(1e-6, 3)
    ax3.set_xlabel("g0 read (log; gray = the corpse, color = after Arm A)")
    ax3.set_title("(c) THE SUBTRACTION SURGERY PER CORPSE\n"
                  "(the boundary state at top, open verdict)", fontsize=10)
    ax3.grid(alpha=0.25, axis="x")

    fig.suptitle("E299 — THE EMBALMING CURVE, AGE-0 HALF — " + verdict,
                 fontsize=12, fontweight="bold")
    fig.text(0.5, 0.005, "x14's Arm A verbatim (subtract the out-of-room "
             "drift, read the probe); the additivity null governs — "
             "resurrection is a lower bound on resuscitability; the age "
             "axis is 0 everywhere (the GPU half sets it)",
             ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(RD / "e299_embalming.png", dpi=150)
    plt.close(fig)
    log(f"[fig] wrote {RD / 'e299_embalming.png'}")

    # ---- the report -----------------------------------------------------
    def fmt_rec(r):
        res = r["resurrection"]
        kd = r["kill_depth"]
        return (f"| {r['key']} | {r['class']} | "
                f"{kd['drift_norm']:.4f} | {kd['drift_in_room_frac']:.2e} "
                f"| {r['corpse_read']['g0']:.3e} | "
                f"{r['write_mass']['mass_ratio_vs_t0']:.4f} | "
                f"{res['factor_over_dead']:.1f}x | "
                f"{res['max_attainable_factor']:.1f}x | "
                f"{res['recovered_fraction_of_baseline']:.4f} | "
                f"{'YES' if res['resurrected'] else 'no'} |")

    inversion_txt = ("the registered inversion FIRED: the multiplier "
                     "ANTI-correlates with mass — denominator saturation, "
                     "disclosed at birth" if rho_fact < 0 else "positive")
    inv_rows = "\n".join(
        f"| {row['key']} | runs/checkpoints/…{row['checkpoint'].split('_')[-1]}"
        f" | {row['class']} | {row['committed_g0']:.3e} | "
        f"{row['committed_drift']:.4f} |"
        for row in inventory["the_corpses"])
    body_rows = "\n".join(fmt_rec(r) for r in corps)
    brec = boundary_rec
    birth = HEAD0 or git_head()
    rep = f"""# E299 — THE EMBALMING CURVE, AGE-0 HALF — REPORT

**Verdict: {verdict}.** {clause}

THE QUESTION: how long does a dead memory stay resuscitable by x14's
subtraction surgery? This is the age-0 half: seven corpses across three
kill classes and {curve['depth_span_orders']:.1f} orders of kill depth
({min(depths):.4f} -> {max(depths):.4f}), each operated on by x14's Arm A
verbatim (subtract the out-of-room drift component, read the g0 probe),
each value-bound to its run's committed literals, and the instrument
itself reproducing x14's committed Arm A on the anchor corpse
(|d| {port_g0_diff:.1e}; G_PORTX14).

## (a) The corpse inventory (all states on disk; ages all 0)

| corpse | checkpoint | class | committed t400 g0 | kill depth (drift) |
|---|---|---|---|---|
{inv_rows}
| e290-0.0008X (ALIVE boundary, holds 0.767x) | …0.0008X_post.pt | orthogonal-sgdm | {brec['corpse_read']['g0']:.3e} | {brec['kill_depth']['drift_norm']:.4f} |

The t100-300 milestone states were never saved (only the t400 finals) —
the age axis is degenerate at 0 in this half (disclosed); the kill-depth
axis is the deliverable. Out-of-scope states (e286/e287/e289's arms,
e280/e284's other-vehicle posts, the optimizer resume artifacts) are
inventoried in metrics.json and excluded (the dispatch named
e285 + e290 + e288 + x14's e283 anchor).

## (b) The resurrection table (the surgery per corpse)

| corpse | class | depth | drift in-room frac | corpse g0 | mass ratio | factor | max attainable | fraction of baseline | resurrected |
|---|---|---|---|---|---|---|---|---|---|
{body_rows}
| e290-0.0008X (ALIVE) | orthogonal | {brec['kill_depth']['drift_norm']:.4f} | {brec['kill_depth']['drift_in_room_frac']:.2e} | {brec['corpse_read']['g0']:.3e} | {brec['write_mass']['mass_ratio_vs_t0']:.4f} | {brec['resurrection']['factor_over_dead']:.2f}x | {brec['resurrection']['max_attainable_factor']:.2f}x | {brec['resurrection']['recovered_fraction_of_baseline']:.4f} | (boundary) |

The write-mass floor: {MASS_FLOOR:.0%} of the t0 in-room mass {m0:.4f}
(x14's EROSION_BAR carried); every corpse passes it (mass ratios
{min(mass_ratios):.4f}-{max(mass_ratios):.4f}).

## (c) The curve + the mechanism statistic

Kill depth spans {curve['depth_span_orders']:.2f} orders
({min(depths):.4f} -> {max(depths):.4f}). The recovered fraction vs the
surviving in-room mass: **Spearman rho {rho_frac:+.4f}**
(bar >= {RHO_BAR}; tie-coarsened {rho_frac_c:+.4f}; Pearson
{r_frac:+.4f}); the factor-scale co-report **{rho_fact:+.4f}**
({inversion_txt}). The mass axis is quasi-binary at age 0
({n_distinct_mass} distinct values at 3dp): the discrimination is
BETWEEN-CLASS (free ~0.844 vs orthogonal ~1.000) — the collinearity debt
is registered for the aging (GPU) half. The orthogonal corpses' own
survival vs depth reproduces T269's inverse law (log-log slope
{curve['inverse_law_context']['orthogonal_loglog_slope_this_session']:.3f}
vs the committed refit -1.005); the free-class corpses sit ~3 orders
below the 1/drift line — different kill animals.

## (d) The null model — T269's clock vs the mass clock

T269/R69's read clock: survival ~ 2^(-t/1040) (fitted on the holding
rung). The free-class corpses' implied MASS clock: half-life
{mass_clocks[0]['implied_mass_half_life_steps']:.0f}-
{mass_clocks[-1]['implied_mass_half_life_steps']:.0f} steps
(per-corpse arithmetic in metrics.json) — **the read dies ~
{mass_clocks[0]['implied_mass_half_life_steps'] / 1040:.2f}x faster than
the mass erodes** under the free stream, and the orthogonal classes'
mass clock is INFINITE (mass preserved by construction while the read
dies — T265's dissociation). At age 0 both clocks predict zero
additional decay — this cell measures the initial condition. The
discriminating prediction handed to the GPU half: under MASS-TRACKS the
resurrection fraction's aging half-life ~= the MASS clock (~1.6k steps
on the free class); under read-clock coupling it is ~1,040 steps.

## The gates (all PASS)

{len(hard)} hard gates: the probe identity (namefree / splice 19+41 /
battery shapes); 8 parent records md5-bound + the fact (md5/size/step +
flat-md5) + the 7 corpses + the boundary (design-time md5s) + the room
file + the span; the base fact-free; the root read-bound; the room
certified (idem/kept^2/span) and BIT-BOUND to e264_rooms.pt; the fact
behaviorally bound (|d g0| {abs(fact_cells['g0'] - FACT_BASELINE_G0):.1e});
every corpse + the boundary value-bound behaviorally + geometrically
(max |d g0| {max(b['read_g0']['abs_diff'] for b in G_CORPSELOAD['per_state'].values()):.1e};
max |d drift| {max(b['drift_norm']['abs_diff'] for b in G_CORPSELOAD['per_state'].values()):.1e});
and G_PORTX14 — the instrument reproduced x14's committed Arm A
({X14_ARM_A_G0:.10f}) on the anchor corpse.

## Disclosures

- THE SATURATION CLAUSE (registered at birth, before compute): the 10x
  token is unattainable where survival > 0.10 (full restoration caps the
  multiplier at 1/survival); those corpses count by full restoration
  (fraction >= 90%). Both readings are in the table.
- THE COLLINEARITY DEBT: at age 0, surviving mass and kill class are
  collinear (the orthogonal streams preserve the room BY CONSTRUCTION);
  MASS-TRACKS here is a between-class discrimination — the aging
  continuations de-confound.
- THE ADDITIVITY NULL (x14 verbatim): resurrection => the out-of-room
  context carried the kill; a sub-bar read never proves
  unresuscitability — every 'resuscitated' is a LOWER BOUND.
- CPU-ONLY desk cell: no training, no stream, no steps; reads + fp64
  projections; torch threads 4, pocketfft workers 2; NO envelope-log
  writes; NO cuda tensors.
- n=1 per corpse, one lineage, one session; nothing guaranteed.

## Provenance

Birth commit {birth} (bars + conventions + operationalizations +
predictions, BEFORE any compute); full run this commit. Machinery:
e261's SRCT/LadderRooms ported whole by import (the committed file
untouched); the surgery is x14's Arm A arithmetic verbatim in fp64. No
NOTES/THINKING/QUEUE/STATE edits.
"""
    (RD / "REPORT.md").write_text(rep, encoding="utf-8")
    log(f"[report] wrote {RD / 'REPORT.md'}")

    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "birth_commit": birth,
        "script": str(Path(__file__).resolve()),
        "machinery_imported_from": [
            str(Path(__file__).resolve().parent / "e261_rank_ladder.py"),
            str(Path(__file__).resolve().parent
                / "x14_transport_intervention.py") + " (the surgery's "
            "arithmetic, ported verbatim; the FILE is not imported — its "
            "committed literals are hard-bound in G_PORTX14)"],
        "eval": {"device": "cpu fp32 probes / cpu fp64 dense projections",
                 "torch_threads": 4, "pocketfft_workers": E261.DCT_WORKERS,
                 "cuda_tensors_created": 0},
        "versions": {"torch": torch.__version__,
                     "numpy": np.__version__,
                     "scipy": __import__("scipy").__version__,
                     "matplotlib": matplotlib.__version__},
    }
    metrics["status"] = ("COMPLETE — adjudicated (the age-0 half; the "
                         "aging continuations ride the GPU half)")
    metrics["date"] = now_iso()
    metrics["phase_note"] = ("P6 DONE (surgery + curve + null model + "
                             "adjudication + honesty + figure + report)")
    metrics["outputs"] = [str(RD / "metrics.json"),
                          str(RD / "e299_embalming.png"),
                          str(RD / "REPORT.md")]
    save_json(RD / "metrics.json", E43.jsonable(metrics))
    log("E299 COMPLETE — metrics + figure + report written")


if __name__ == "__main__":
    HEAD0 = git_head()
    main()
