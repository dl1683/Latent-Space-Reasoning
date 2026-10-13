"""G1B — THE 2.74M CONTINUITY CELL: g1's registered discharge (W020 g1 §8/6).

Design spec: scratch/g1_design.md — g1's bars are carried VERBATIM, now LIVE.
g1 ran at the 0.84M lineage (the ≤1M size correction, spec §1) and aborted at
G-ROOT (consolidated root g-12 0.2536 < 0.78: the ±8 jitter replay does not
generalize to offset -12 at that scale), so NO wall clause adjudicated. The
spec's own CONTINGENCY CELL (§1, registered as "the preferred continuity
cell") is this file: the identical design on the 2.74M line, reusing the
arc's stored consolidated root DIRECTLY — the root whose g-12 0.9156 cleared
G-ROOT by construction — making the wash bit-comparable to e176N/e185's
stored traces. Adjudicate against g1's frozen bars exactly.

THE CONFIG DELTA from g1 (the whole difference; everything else is
lab/g1_anchored_ball.py VERBATIM via import):
  - family: Cfg 6L/6H/192d block 256 = 2,739,072 trainable params (the
    e131/e048 lineage) instead of 4L/4H/128d = 840,704.
  - root: runs/checkpoints/e131_consolidated_e113.pt loaded DIRECTLY as
    theta0 (phase 0 — install + consolidation — is the arc's own history,
    not re-run); gated bit-exact against the file on disk.
  - basin prior: the RAW 2.5-5 L2 (T119/E180; no sqrt(P) scaling — this IS
    the line the prior was measured on). The R ladder {0.7, 1.4, 4.2} L2 is
    carried VERBATIM from g1 (absolute radii, the continuity requirement):
    {0.28, 0.56, 1.68} x the 2.5 low end — W1/W2 inside any plausible
    basin, W3 beyond e185's stored D_kill 2.489.

THE ARMS (g1's seven, VERBATIM recipes/seeds/instruments at this size):
  C  uncommitted  — the arc's own e176N arm A cell re-run (the clock,
                    D_kill, the CE curve; priors +1 0.678 / +2 0.0271)
  W1 commit(0.7)  — primary: does confinement hold the fact?
  W2 commit(1.4)  — the ladder's middle
  W3 commit(4.2)  — wall-too-far: the kill en route
  N0 uncommitted labels-noise (e185's own organism, seed 18501) — the
     continuity gate for the noise kill
  N1 commit(0.7) labels-noise  — noise under the wall
  N2 commit(0.7) shuffled-target (seed 18502) — arm 2
  All seven share the seed-10902 input stream (md5-gated per step); the
  deltas are the commit and the targets. STRUCTURAL CONTINUITY CHECK: every
  arm's STEP-1 WEIGHTS are bit-identical to C's (the first forward happens
  at d=0 inside the ball; the wall first acts at forward 2). The +1
  READINGS then discriminate: C/N0/W3 read the free step-1 state (W3's +1
  equals C's exactly — 1.65 L2 < 4.2, no projection), while W1/W2/N1/N2's
  light evals run on the ARMED twin (g1's semantics: the settled/projected
  state that enters forward 2) — so W1's +1 measures the step-1 corpus
  displacement REVERSED to the ball, and N1's +1 the noise damage under the
  same reversal: damage-vs-displacement, read directly at +1.

================ LIVE BARS (g1's spec §4-6, VERBATIM — now adjudicable) =====
Conventions: dies = g-12 <= 0.27; maintains = g-12 >= 0.50 at EVERY
checkpoint {1,2,4,10,50,100,200,300}; "dies by +50" = the +50 state.
Order: GATES -> WALL -> NOISE -> PIN -> COSTS; no bar shopping.

GATES (any failure => ABORT TO TEXTURE):
  G-ROOT  root g-12 >= 0.78 (clears by construction at 0.9156; still
          measured in-run, gated, and co-reported vs the stored prior).
  G-CTRL  arm C kills by +50.
  G-PIN   every wall arm's raw displacement <= R + 1.5 L2 at every
          checkpoint (VERBATIM; note the registered one-step fuzz at this
          size is lr*sqrt(P) = 1.66 L2, vs 0.92 at 0.84M — measured, and
          the beyond-wall increments are tangential, so the verbatim bar
          holds mechanically; if it ever failed it would be reported as an
          implementation-gate failure, never re-tuned).
  G-NOISE N0 kills by +10 at displacement-match (e185's own convention).

PREDICTED: WALL-HOLDS — W1 maintains (>= 0.50 through +300, floor band
0.55-0.85); W3 dies by +50 (kill en route); WALL-CLIFF-INSIDE with W2
locating the cliff (predicted maintains => cliff in (1.4, 4.2], which must
bracket the measured D_kill ~ 2.489); NOISE-SPARED-BY-WALL (N1, N2 >= 0.50
pinned, CE_R within root + 0.3; aniso fork if N1 dips >= 0.15 below W1's
floor); FLAT-AT-PIN (|W1 g-12(+300) - g-12(+50)| <= 0.05).
FALSIFIERS: F1 WALL-DEAF | F2 WALL-CLIFF-MISPLACED | F3 NOISE-PENETRATES |
F4 ERODES-AT-PIN. COSTS: WALL-TAXES-ADAPTATION (W1 CE@300 >= C's + 0.05;
WALL-FREE if |dCE| < 0.05); W1 held30_gm12 at +300 >= 0.40; anatomy
(row0 >= 0.5x root, site span >= 0.7, band content=True); memory cost one
fp32 anchor set (10.96 MB) + <5% step compute.

COMPUTE ENVELOPE: 2,739,072 params — above the lab's old 1M default, well
inside the ≤100M free tier (common.py, Devansh 2026-09-27); the stated
reason registered in g1's spec §1 stands: CONTINUITY — the arc's bars are
calibrated on this line. GPU allowed (pick_dev park-once double-poll +
mid-run poll every 25 steps; NO concurrent GPU; park on thermal); torch
threads 8; cooldown 60 s before each training; per-training cap 180 s GPU /
1800 s CPU; 7 trainings + ~25 dial sets.

Outputs: runs/g1b/{metrics.json, continuity.png}; checkpoints
runs/checkpoints/g1b_*.pt. No NOTES/THINKING/QUEUE/STATE edits; single
commit, no push.

Run:  cd lab && python g1b_continuity.py    (G1B_SMOKE=1 shakedown)
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # GPU allowed, gated below

SMOKE = os.environ.get("G1B_SMOKE") == "1"
if SMOKE:
    os.environ["G1_SMOKE"] = "1"     # g1's machinery takes its own smoke trims

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

torch.set_num_threads(8)                              # e152R/e143/e184/e179

import common                                          # noqa: E402
from common import (CharCorpus, cooldown, gpu_ok,     # noqa: E402
                    gpu_status, run_dir, save_json, set_seed)
import e043_install as E43                             # noqa: E402 (REPO,
                                                      # find_occ, SPLICE_RNG,
                                                      # jsonable)
import g1_anchored_ball as G1                          # noqa: E402 — g1's
                                                      # machinery VERBATIM

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)
G1.log = log                                          # unify the timeline

# ======================================================================
# THE CONFIG DELTA (the whole difference from g1)
# ======================================================================
G1B_CFG = common.Cfg(vocab=65, n_layer=6, n_head=6, n_embd=192, block_size=256)
G1B_PARAMS = 2_739_072                                # the e131/e048 line
ROOT_CK = "e131_consolidated_e113.pt"                 # the arc's consolidated
                                                      # root, loaded DIRECTLY
CKPT_DIR = E43.REPO / "runs" / "checkpoints"

R_LADDER = (0.7, 1.4, 4.2)      # VERBATIM from g1 (absolute L2, frozen)
BASIN_PRIOR = (2.5, 5.0)        # T119/E180 RAW (this IS the line; no scaling)
R_HAT = 2.5                     # the raw prior's low end
PRIORS = {                       # same-line stored references (e176N/e185)
    "source": "runs/e176n + runs/e185 metrics (this line's own organism)",
    "root_gm12": 0.9155886769294739,
    "plus1_gm12": 0.6780440807342529,
    "plus2_gm12": 0.027077054604887962,
    "e185_D_kill": 2.4892616271972656,
    "e185_t_kill": 2,
    "g1_discharge": "runs/g1/metrics.json (0.84M, G-ROOT 0.2536 < 0.78 — "
                    "nothing adjudicated; this cell is the discharge)",
}

# g1's machinery constructs CommittedGPT from its module global at call time
# (load_g1 / evl_load / the arm loop): point it at the 2.74M family. This is
# the entire mechanism-level change — CommittedGPT, g1_wash (e185's
# noise_wash + wall bookkeeping), pick_dev/migrate_to_cpu and every
# instrument below run g1's code unmodified.
G1.G1_CFG = G1B_CFG
G1.G1_PARAMS = G1B_PARAMS

G1B_PREDICTION = {
    "gates": "G-ROOT root g-12 >= 0.78 | G-CTRL arm C kills by +50 | "
             "G-PIN every wall arm raw displacement <= R + 1.5 at every "
             "checkpoint | G-NOISE N0 kills by +10 at displacement-match. "
             "Any failure => ABORT TO TEXTURE, nothing adjudicated.",
    "predicted": "WALL-HOLDS (g1's registered prior, now live): (1) W1 "
                 "maintains (g-12 >= 0.50 through +300, floor band 0.55-0.85; "
                 "no checkpoint <= 0.27); (2) W3 dies by +50 (the kill en "
                 "route to the wall); (3) WALL-CLIFF-INSIDE with W2 "
                 "maintaining => cliff in (1.4, 4.2], which must bracket the "
                 "measured D_kill (stored 2.489) — the headline quantitative "
                 "test vs e180's t* extrapolation; (4) NOISE-SPARED-BY-WALL "
                 "(N1, N2 >= 0.50 at {1,2,4,10} pinned; CE_R within root + "
                 "0.3); (5) FLAT-AT-PIN (|W1 g-12(+300) - g-12(+50)| <= 0.05).",
    "falsifiers": "F1 WALL-DEAF (W1 dies by +50 at verified pin) | F2 "
                  "WALL-CLIFF-MISPLACED (survival does not order with R "
                  "against measured D_kill, e.g. W3 maintains) | F3 "
                  "NOISE-PENETRATES (N1/N2 kill at pinned R) | F4 ERODES-AT-PIN "
                  "(W1 declines monotonically >= 0.10 from +50 to +300).",
    "costs": "WALL-TAXES-ADAPTATION predicted FIRES (W1 corpus CE at +300 "
             ">= C's + 0.05); WALL-FREE if |dCE| < 0.05; W1 held30_gm12(+300) "
             ">= 0.40; W1 +300 anatomy: row0 >= 0.5x root, site span >= 0.7, "
             "band rows content=True.",
    "registration": "scratch/g1_design.md sections 4-6 + section 8 VERBATIM "
                    "(committed before g1's implementation; frozen — no bar "
                    "shopping). g1b adds NO bar: the continuity cell of "
                    "spec section 1, the 0.84M discharge's registered "
                    "discharge.",
}

deviations: list[str] = [
    "ROOT: loads the arc's consolidated root (e131_consolidated_e113.pt) "
    "DIRECTLY as theta0 — phase 0 (e043 Dmix install + e113 jitter "
    "consolidation) is the arc's own history, not re-run (spec section 1's "
    "registered contingency cell: 'reuses the stored root directly ... "
    "making g1's wash bit-comparable to e176N/e179's stored traces'). The "
    "root is NOT re-saved under a g1b name (it exists; provenance recorded "
    "in metrics.root); only arm checkpoints are written.",
    "R LADDER VERBATIM: {0.7, 1.4, 4.2} L2 absolute, carried from g1 "
    "(the continuity requirement) — NOT rescaled to the raw 2.74M basin "
    "({0.28, 0.56, 1.68} x the 2.5 low end): W1/W2 sit inside any plausible "
    "basin, W3 beyond e185's stored D_kill 2.489.",
    "ENVELOPE: 2,739,072 trainable params — above the lab's old 1M default, "
    "inside the <=100M free tier (common.py, Devansh 2026-09-27); the stated "
    "reason registered in g1's spec section 1 stands: CONTINUITY — the "
    "arc's bars (D_kill 2.489, root 0.9156, the jitter-to--12 "
    "generalization) are this line's numbers, and the 0.84M discharge "
    "aborted at G-ROOT.",
    "G-PIN kept VERBATIM (raw displacement <= R + 1.5) although the "
    "registered one-step wall fuzz at this size is lr*sqrt(P) = 1.66 L2 "
    "(vs 0.92 at 0.84M): measured, tangential beyond-wall increments keep "
    "the verbatim bar satisfiable; had it failed mechanically it would be "
    "reported as an implementation-gate failure, never re-tuned.",
    "MACHINERY: everything is lab/g1_anchored_ball.py VERBATIM via import "
    "(CommittedGPT, g1_wash, pick_dev/migrate_to_cpu, evl_load's "
    "settle+disarm PIVOT, every instrument); the only patch is "
    "G1.G1_CFG/G1.G1_PARAMS -> the 6/6/192 family (constructed at call "
    "time). G1B_SMOKE sets G1_SMOKE before import so g1's smoke trims "
    "apply. The protocol rebuild, measure()/flat_cells, arm loop, gates, "
    "adjudication and plot are g1's main() copied with the root/labels "
    "delta and phase 0 removed.",
    "DEVICE: g1's policy unchanged (GPU gated, park-once double-poll, "
    "mid-run contention poll every 25 steps, per-training caps 180 s GPU / "
    "1800 s CPU); the stored e176N/e185 references ran CPU-only — GPU float "
    "nondeterminism is co-reported and every bar adjudicates against the "
    "in-run control.",
    "Smoke mode trims: g1's (8-step washes via g1's CK constants, lean "
    "dials, no cooldowns) — nothing adjudicated.",
    "CLAUSE TEMPLATE FIX (after the first full run, re-run end-to-end): "
    "g1's inherited WALL-CLIFF clause ternary labeled W2 'dies' when W2 "
    "lands in the NEITHER branch (min 0.3203 at +4 — below the 0.50 "
    "maintain bar, above the 0.27 death bar — recovered to 0.6501 at "
    "+300), and printed '=> cliff in None'. The template was widened to "
    "report the neither-branch truthfully; NO bar, gate or adjudication "
    "logic changed; the experiment was re-run end-to-end from the stored "
    "root with identical seeds (numbers re-drawn, story unchanged).",
]

CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "g1b", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name}")


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("g1b_smoke" if SMOKE else "g1b")
    log(f"G1B THE 2.74M CONTINUITY CELL (smoke={SMOKE}) -> {rd}")
    set_seed(G1.INSTALL_SEED)          # g1's opening seed (global init only;
                                       # every arm's RNG is its own)

    # ---------------- protocol rebuild (g1's main VERBATIM; the batteries
    # are the arc's — e152's j=54 pool, e043's splice, e119's geometries)
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
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"
    log(f"protocol rebuilt: install60 {mix}, held30 (SPLICE_RNG {E43.SPLICE_RNG})")
    name_ids = corpus.encode(G1.NAME)

    # measurement pool: e152's locked j=54 windows (instrument only)
    wins = []
    for p, h in install_occ:
        pre = train_ids[p - G1.PRE - G1.RETEACH_J: p]
        post = train_ids[p + len(h): p + len(h) + G1.SITE_CONT]
        if len(pre) != G1.PRE + G1.RETEACH_J or len(post) != G1.SITE_CONT:
            raise RuntimeError(f"pool window short at p={p}")
        w = torch.cat([pre, name_ids, post])
        if len(w) != G1.BLOCK:
            raise RuntimeError(f"pool window len {len(w)} != {G1.BLOCK}")
        wins.append(w)
    pool_x = torch.stack(wins)
    G_POOL = {"shape": list(pool_x.shape),
              "name_xcols": [G1.SITE_Z_XCOL, G1.SITE_Z_XCOL + len(G1.NAME) - 1],
              "all_windows_name_in_place": bool(
                  all(torch.equal(w[G1.SITE_Z_XCOL: G1.SITE_Z_XCOL + len(G1.NAME)],
                                  name_ids) for w in pool_x)),
              "note": "MEASUREMENT instrument only (e152's locked j=54 pool); "
                      "NO window from this pool enters any training"}
    G_POOL["pass"] = G_POOL["all_windows_name_in_place"]
    assert G_POOL["pass"], f"pool gate FAILED: {G_POOL}"

    # =====================================================================
    # THE NEUTRAL STREAM (e170's construction VERBATIM via e176n arm A)
    # =====================================================================
    arng = random.Random(G1.E170_ANCHOR_SEED)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - G1.BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        tries += 1
        txt = train_text[s: s + G1.BLOCK + 1]
        if any(f in txt for f in G1.ANCHOR_FORBIDDEN):
            rejections += 1
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    anchor_neutral = torch.stack([train_ids[s: s + G1.BLOCK] for s in n_starts])

    host_positions = [p for p in E43.find_occ(train_text, G1.HOSTS[0])]
    host_positions += [p for p in E43.find_occ(train_text, G1.HOSTS[1])]
    jc_neutral = sum(1 for s in n_starts
                     if any(s <= p < s + G1.BLOCK + 1 for p in host_positions))
    host_occ_total = len(host_positions)
    bg_rate = host_occ_total * (G1.BLOCK + 1) / len(train_ids)
    G_ANCHOR = {
        "neutral_bank": {
            "construction": ("16 plain corpus windows from train_ids, RNG seed "
                             f"{G1.E170_ANCHOR_SEED}, rejection if [s, s+257) "
                             "contains FLORIZEL/ELIZABETH/ZEPH/MIRABEL — "
                             "e170's construction VERBATIM (= e176n arm A's "
                             "stream; FIXED content)"),
            "n_windows": 16, "block": G1.BLOCK, "seed": G1.E170_ANCHOR_SEED,
            "starts": n_starts, "tries": tries, "rejections": rejections,
            "forbidden": list(G1.ANCHOR_FORBIDDEN),
            "windows_with_host_content": sum(
                1 for s in n_starts
                if any(f in train_text[s: s + G1.BLOCK + 1] for f in G1.HOSTS)),
            "junctions_covered": jc_neutral,
        },
        "budget_identical_to_e176n": bool(anchor_neutral.shape == (16, G1.BLOCK)),
        "rng_note": ("g1_wash verbatim; the seed-10902 aj/rj draw sequence is "
                     "IDENTICAL across ALL SEVEN arms (same shapes/moduli, "
                     "drawn BEFORE any noise draw) — the input stream is "
                     "bit-identical (gated by per-step md5); the deltas are "
                     "the wall (wash arms) and each step's targets (noise)"),
        "random_channel_background": {
            "host_occurrences_in_train": host_occ_total,
            "est_window_hit_rate": bg_rate,
        },
    }
    G_ANCHOR["pass"] = bool(
        G_ANCHOR["neutral_bank"]["windows_with_host_content"] == 0
        and G_ANCHOR["neutral_bank"]["junctions_covered"] == 0
        and G_ANCHOR["budget_identical_to_e176n"])
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"
    log(f"G_ANCHOR: neutral bank 16x{G1.BLOCK} (seed {G1.E170_ANCHOR_SEED}, "
        f"{rejections} rejections/{tries} tries) — host 0/16, junctions "
        f"0/16: PASS")

    # ---------------- batteries (e119/e176n verbatim)
    bat_ids, held_ids = {}, {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
        hs = [train_text[p - G1.PRE - j: p] for p, _ in held_occ]
        held_ids[j] = torch.stack([corpus.encode(c) for c in hs])
    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60, G1.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    gm12_ids = bat_ids[-12]
    g0_ids = bat_ids[0]

    gates_surg: dict = {}

    def measure(sd: dict, tag: str, lean: bool = False) -> dict:
        """g1's measure() VERBATIM dial (e176n's = the e131 dial set) on
        evl_load (settle+disarm; g1's PIVOT)."""
        net = G1.evl_load(sd)
        sd_local = {k: v.detach().clone() for k, v in net.state_dict().items()}
        out: dict = {"tag": tag}
        out["base"] = {j: G1.battery_cell(net, bat_ids[j], zid) for j in G1.GEOS}
        out["base_held"] = {j: G1.battery_cell(net, held_ids[j], zid)
                            for j in G1.GEOS}
        out["ce_r"] = G1.ce_fixed_cpu(net, *r_eval_xy)
        log(f"[{tag}] base: " + " ".join(f"g{j:+d} {out['base'][j]['mean_pz']:.4f}"
                                         for j in G1.GEOS)
            + " | held30: " + " ".join(
                f"g{j:+d} {out['base_held'][j]['mean_pz']:.4f}" for j in G1.GEOS)
            + f" | CE_R {out['ce_r']:.4f}")
        out["site_read"] = G1.read_fact_at(net, pool_x, name_ids, zid,
                                           G1.SITE_ADDR_ROW, G1.SITE_Z_XCOL)
        log(f"[{tag}] site read @183: onset "
            f"{out['site_read']['pz_onset_mean']:.4f} span "
            f"{out['site_read']['pname_mean_over7']:.4f}")
        if not lean and not SMOKE:
            out["census_old"] = G1.row_census(net, G1.ROWS_OLD,
                                              lambda n: G1.battery_pz(
                                                  n, bat_ids[0], zid))
            co = out["census_old"]["rows"]
            out["old_band"] = {
                "base_pz": out["census_old"]["base_readout"],
                "row0_strength": co["0"]["strength"],
                "A129": co["129"]["strength"],
                "band121_129_max": max(co[str(r)]["strength"]
                                       for r in range(121, 130)
                                       if str(r) in co)}
            log(f"[{tag}] old band: row0 S "
                f"{out['old_band']['row0_strength']:+.4f} | A(129) "
                f"{out['old_band']['A129']:+.4f}")
            DELS = {"d_all": G1.D_ALL, "d183": (G1.SITE_ADDR_ROW,)}
            out["del_table"] = {}
            for dl, rows_ in DELS.items():
                sd_d, gate = G1.deleted_wpe(sd_local, rows_)
                gates_surg[f"{tag}__{dl}"] = gate
                if not gate["pass"]:
                    raise RuntimeError(f"deletion gate FAILED {tag}/{dl}: "
                                       f"{gate}")
                net.load_state_dict(sd_d)
                cell = {"g0": G1.battery_cell(net, bat_ids[0], zid)["mean_pz"]}
                if dl == "d183":
                    cell["gm12"] = G1.battery_cell(net, bat_ids[-12],
                                                   zid)["mean_pz"]
                out["del_table"][dl] = cell
            net.load_state_dict(sd_local)
            log(f"[{tag}] deletions g0: " + " | ".join(
                f"{dl} {out['del_table'][dl]['g0']:.3f}" for dl in DELS))
        else:
            w = net.wpe.weight.data
            orig = w.clone()
            mean_row = orig.mean(0)
            bp = G1.battery_pz(net, bat_ids[0], zid)
            w[129] = mean_row
            m129 = bp - G1.battery_pz(net, bat_ids[0], zid)
            w.copy_(orig)
            w[129] = 0.0
            z129 = bp - G1.battery_pz(net, bat_ids[0], zid)
            w.copy_(orig)
            assert torch.equal(w, orig), "lean A129 failed to restore wpe"
            out["A129_quick"] = float(min(m129, z129))
        del net
        return out

    def flat_cells(m: dict) -> dict:
        c = {"gm12": m["base"][-12]["mean_pz"],
             "g0": m["base"][0]["mean_pz"],
             "gp12": m["base"][12]["mean_pz"],
             "held30_gm12": m["base_held"][-12]["mean_pz"],
             "held30_g0": m["base_held"][0]["mean_pz"],
             "ce_r": m["ce_r"],
             "site_read_onset": m["site_read"]["pz_onset_mean"],
             "site_read_span": m["site_read"]["pname_mean_over7"]}
        if "old_band" in m:
            c["A129"] = m["old_band"]["A129"]
            c["row0_strength"] = m["old_band"]["row0_strength"]
            c["dall_g0"] = m["del_table"]["d_all"]["g0"]
            c["d183_g0"] = m["del_table"]["d183"]["g0"]
            c["d183_gm12"] = m["del_table"]["d183"]["gm12"]
        else:
            c["A129"] = m["A129_quick"]
        return c

    # =====================================================================
    # THE ROOT — the arc's consolidated root, loaded DIRECTLY (g1's phase 0
    # replaced by the arc's own history; gated bit-exact)
    # =====================================================================
    log("=" * 78)
    root_net = G1.load_g1(CKPT_DIR / ROOT_CK)
    assert root_net.num_params() == G1B_PARAMS, \
        f"root param count {root_net.num_params()} != {G1B_PARAMS}"
    theta0 = {k: v.detach().clone() for k, v in root_net.state_dict().items()}
    raw = torch.load(CKPT_DIR / ROOT_CK, map_location="cpu", weights_only=False)
    raw_sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    md0 = max(float((theta0[k].float() - raw_sd[k].float()).abs().max())
              for k in raw_sd)
    G_BITEXACT = {"checkpoint": f"runs/checkpoints/{ROOT_CK}",
                  "meta": raw.get("meta"),
                  "n_tensors": len(raw_sd),
                  "max_abs_diff_vs_file": md0, "pass": bool(md0 == 0.0)}
    assert G_BITEXACT["pass"], f"root load not bit-exact: {md0}"
    log(f"root loaded bit-exact from {ROOT_CK} ({G1B_PARAMS} params, "
        f"{len(raw_sd)} tensors; meta {raw.get('meta')})")

    root = measure(theta0, "g1b_root", lean=SMOKE)
    root_cells = flat_cells(root)
    G_ROOT0 = {"bar": G1.EXPRESS_BAR, "gm12": root_cells["gm12"],
               "prior_gm12": PRIORS["root_gm12"],
               "delta_vs_prior": root_cells["gm12"] - PRIORS["root_gm12"],
               "pass": bool(root_cells["gm12"] >= G1.EXPRESS_BAR)}
    log(f"G-ROOT: root g-12 {root_cells['gm12']:.4f} "
        f"(bar >= {G1.EXPRESS_BAR}; stored prior {PRIORS['root_gm12']:.4f}, "
        f"delta {G_ROOT0['delta_vs_prior']:+.2e}): "
        f"{'PASS' if G_ROOT0['pass'] else 'FAIL'}")
    if not G_ROOT0["pass"] and not SMOKE:
        log("G-ROOT FAILED — arms still run for the record; verdict will be "
            "TEXTURE (gate failure), nothing adjudicated")

    # =====================================================================
    # THE ARMS (g1's seven VERBATIM; cooldown before each; ALL inputs
    # bit-identical — the deltas are the wall and the targets)
    # =====================================================================
    ARM_SPECS = ([("C", None, "true", 0, G1.CK_WASH,
                   "CONTROL — uncommitted neutral wash (e176N arm A VERBATIM, "
                   "this line's own cell): the clock, D_kill, the CE "
                   "adaptation curve (priors +1 0.678 / +2 0.0271)")]
                  + [(f"W{i+1}", R, "true", 0, G1.CK_WASH,
                      f"WALL R={R} — commit({R}) then the identical neutral "
                      f"wash; step-1 weights bit-identical to C (the wall "
                      f"first acts at forward 2; +1 reads the settled state)")
                      for i, R in enumerate(R_LADDER)]
                  + [("N0", None, "iid", G1.NOISE_SEED_A, G1.CK_NOISE,
                      "NOISE CONTROL — uncommitted labels-noise (e185 "
                      "VERBATIM, its own organism): the continuity gate"),
                     ("N1", 0.7, "iid", G1.NOISE_SEED_A, G1.CK_NOISE,
                      "NOISE UNDER THE WALL — commit(0.7), labels-noise; "
                      "step-1 weights bit-identical to N0; +1 reads the "
                      "settled state"),
                     ("N2", 0.7, "perm", G1.NOISE_SEED_B, G1.CK_NOISE,
                      "NOISE UNDER THE WALL, arm 2 — commit(0.7), "
                      "shuffled-target")])

    arms: dict = {}
    batteries_all: dict = {}
    G_BITROOT = {}
    for tag, R, mode, nseed, cks, desc in ARM_SPECS:
        log("=" * 78)
        if not SMOKE:
            log(f"[thermal] cooldown {G1.COOLDOWN_S:.0f}s before {tag}")
            cooldown(G1.COOLDOWN_S)
        log(f"ARM {tag} — {desc}")
        net0 = G1.evl_load(theta0) if R is None else G1.CommittedGPT(G1B_CFG)
        if R is not None:
            net0.load_state_dict(theta0)
            net0.commit(R)
            # G_BITROOT: the wall root's tensors are bit-identical copies of
            # theta0 (the anchor is a copy; the wall is inert before commit)
            body, _ = G1.split_anchored_sd(net0.state_dict())
            md = max(float((body[k].float() - theta0[k].float()).abs().max())
                     for k in theta0)
            anch_ok = all(torch.equal(net0._anchor(n), p.detach())
                          for n, p in net0.named_parameters())
            G_BITROOT[tag] = {"max_abs_diff": md, "anchors_bit_equal": bool(anch_ok),
                              "n_anchor_tensors": net0._n_anchor_tensors,
                              "pass": bool(md == 0.0 and anch_ok)}
            assert G_BITROOT[tag]["pass"], f"{tag}: wall root != theta0"
            log(f"G_BITROOT[{tag}]: max|diff| {md:.1e}, anchors bit-equal: "
                f"PASS")
        arm = G1.g1_wash(tag, net0, anchor_neutral, train_ids, itos,
                         r_eval_xy, gm12_ids, g0_ids, zid, target_mode=mode,
                         noise_seed=nseed, ckpt_steps=cks)
        G_DRAWFREE = {"zeph_violations": arm["zeph_violations"],
                      "pass": bool(arm["zeph_violations"] == 0)}
        assert G_DRAWFREE["pass"], f"{tag}: name token leaked into a window"
        gates_surg[f"G_DRAWFREE_{tag}"] = G_DRAWFREE
        arms[tag] = arm

        # checkpoints: every noise ckpt + each arm's final
        for s in sorted(arm["sds"]):
            if tag.startswith("N") or s == max(arm["sds"]):
                save_ckpt(f"g1b_{tag}_s{s}", arm["sds"][s],
                          {"desc": f"e131 root + {s}-step {mode}-target neutral "
                                   f"{'wash' if mode == 'true' else 'noise'} "
                                   f"(R={R}, input seed {G1.FREEZE_SEED}, lr "
                                   f"{G1.FT_LR})",
                           "steps": int(s), "R": R, "target_mode": mode,
                           "input_seed": G1.FREEZE_SEED, "lr": G1.FT_LR,
                           "noise_seed": nseed or None,
                           "base": f"runs/checkpoints/{ROOT_CK}"})

        # full dials: wash arms at {2, 50, 300}; noise arms at every ckpt
        full_at = [s for s in cks if s in arm["sds"]
                   and (mode == "true" and s in G1.FULL_DIAL_WASH
                        or mode != "true")]
        batteries = {}
        for s in full_at:
            log(f"{tag} +{s} full dial")
            batteries[str(s)] = measure(arm["sds"][s], f"{tag}{s}", lean=SMOKE)
        batteries_all[tag] = batteries

    # =====================================================================
    # CONSTRUCTION-FIDELITY GATES (g1 VERBATIM)
    # =====================================================================
    G_INPUTS = {"per_step": {}, "pass": None, "note":
                "the seed-10902 aj/rj draw sequence is shared by ALL arms — "
                "per-step input batches are bit-identical (md5-gated) across "
                "wash AND noise arms through +10"}
    noise_tags = [t for t, *_ in ARM_SPECS if t.startswith("N")]
    for step in range(1, G1.CK_NOISE[-1] + 1):
        hs = {t: arms[t]["x_hashes"].get(step) for t in arms}
        same = all(h is not None for h in hs.values()) and \
            len(set(hs.values())) == 1
        G_INPUTS["per_step"][step] = {t: hs[t] for t in arms}
        G_INPUTS["per_step"][step]["identical"] = bool(same)
    G_INPUTS["pass"] = bool(all(v["identical"]
                                for v in G_INPUTS["per_step"].values()))
    assert G_INPUTS["pass"], "input streams diverged across arms"
    log(f"G_INPUTS: per-step inputs bit-identical across all "
        f"{len(arms)} arms through +{G1.CK_NOISE[-1]}: PASS")

    G_TARGETS = {"per_arm": {}}
    for tag in noise_tags:
        ys = arms[tag]["y_stats"]
        G_TARGETS["per_arm"][tag] = {
            "frac_targets_changed_mean": float(
                np.mean([r["frac_targets_changed"] for r in ys])),
            "multiset_equals_true_all_steps": bool(
                all(r["multiset_equals_true"] for r in ys)),
        }
    G_TARGETS["per_arm"]["N0"]["expectation"] = (
        "iid labels change ~64/65 positions; multiset NOT preserved")
    G_TARGETS["per_arm"]["N1"]["expectation"] = (
        "iid labels change ~64/65 positions; multiset NOT preserved")
    G_TARGETS["per_arm"]["N2"]["expectation"] = (
        "permutation changes ~63/65 positions; multiset preserved at EVERY step")
    G_TARGETS["pass"] = bool(
        all(G_TARGETS["per_arm"][t]["frac_targets_changed_mean"] > 0.9
            for t in noise_tags)
        and not G_TARGETS["per_arm"]["N0"]["multiset_equals_true_all_steps"]
        and not G_TARGETS["per_arm"]["N1"]["multiset_equals_true_all_steps"]
        and G_TARGETS["per_arm"]["N2"]["multiset_equals_true_all_steps"])
    assert G_TARGETS["pass"], f"target gate FAILED: {G_TARGETS}"
    log("G_TARGETS: iid arms break the target multiset, perm arm preserves "
        "it at every step: PASS")

    # =====================================================================
    # DISPLACEMENT TABLES + COSINES (e185's currency, measured)
    # =====================================================================
    def disp_table_for(tag):
        rows = []
        for t in arms[tag]["traj"]:
            s = t["step"]
            row = {"step": s, "ce_batch": t["ce_batch"],
                   "cum_disp": t["cum_disp"], "step_disp": t["step_disp"],
                   "d_proj": t["d_proj"]}
            if "g_m12_mean_pz" in t:
                row["g_m12_light"] = t["g_m12_mean_pz"]
                row["ce_r_light"] = t["ce_r"]
            if s in arms[tag]["deltas"] and s in arms["C"]["deltas"]:
                d_a = arms[tag]["deltas"][s]
                d_c = arms["C"]["deltas"][s]
                row["cos_vs_C"] = float(torch.dot(d_a, d_c)
                                        / (torch.norm(d_a) * torch.norm(d_c)
                                           + 1e-30))
            rows.append(row)
        return rows

    disp_table = {tag: disp_table_for(tag) for tag in arms}

    # =====================================================================
    # ADJUDICATION (g1's registered clauses VERBATIM; order GATES -> WALL ->
    # NOISE -> PIN -> COSTS; no shopping)
    # =====================================================================
    def light_gm12(tag):
        return {t["step"]: t["g_m12_mean_pz"] for t in arms[tag]["traj"]
                if "g_m12_mean_pz" in t}

    def full_gm12(tag):
        return {int(s): flat_cells(batteries_all[tag][s])["gm12"]
                for s in batteries_all.get(tag, {})}

    # ---- G-CTRL: arm C kills by +50 (the +50 state)
    c_gm12 = light_gm12("C")
    G_CTRL = {"bar": G1.SHUT_BAR, "gm12_at_50": c_gm12.get(50),
              "earliest_le_bar": next((s for s in G1.CK_WASH if s > 0
                                       and c_gm12.get(s, 1.0) <= G1.SHUT_BAR),
                                      None),
              "pass": bool(c_gm12.get(50, 1.0) <= G1.SHUT_BAR)}
    log(f"G-CTRL: arm C g-12 at +50 = {c_gm12.get(50)} "
        f"(earliest <= {G1.SHUT_BAR}: +{G_CTRL['earliest_le_bar']}): "
        f"{'PASS' if G_CTRL['pass'] else 'FAIL'}")

    # ---- G-PIN: every wall arm's raw displacement <= R + 1.5 at every ckpt
    G_PIN = {"per_arm": {}, "fuzz_formula_at_this_size": float(
        G1.FT_LR * (G1B_PARAMS ** 0.5))}
    for tag in ("W1", "W2", "W3", "N1", "N2"):
        R = arms[tag]["wall_R"]
        rows = [r for r in disp_table[tag] if "g_m12_light" in r]
        mx = max(r["cum_disp"] for r in rows) if rows else None
        G_PIN["per_arm"][tag] = {
            "R": R, "bound": R + G1.PIN_FUZZ_BAR,
            "max_raw_disp_at_ckpt": mx,
            "per_ckpt": {r["step"]: r["cum_disp"] for r in rows},
            "pass": bool(mx is not None and mx <= R + G1.PIN_FUZZ_BAR)}
        log(f"G-PIN[{tag}]: max raw |d| at ckpt {mx:.4f} <= "
            f"{R + G1.PIN_FUZZ_BAR:.2f}: "
            f"{'PASS' if G_PIN['per_arm'][tag]['pass'] else 'FAIL'}")
    G_PIN["pass"] = bool(all(v["pass"] for v in G_PIN["per_arm"].values()))

    # ---- G-NOISE: N0 kills by +10 at displacement-match (e185's logic)
    n0_gm12 = light_gm12("N0")
    t_kill = next((s for s in G1.CK_NOISE if c_gm12.get(s, 1.0) <= G1.SHUT_BAR), None)
    ctrl_kills_fine = t_kill is not None
    if ctrl_kills_fine:
        D_kill = next(r["cum_disp"] for r in disp_table["C"]
                      if r["step"] == t_kill)
    else:
        # the control kills only past the noise horizon: D_kill at the +50
        # state (coarse; co-reported)
        t_kill_eff = next((s for s in G1.CK_WASH
                           if s > 0 and c_gm12.get(s, 1.0) <= G1.SHUT_BAR), None)
        D_kill = next(r["cum_disp"] for r in disp_table["C"]
                      if r["step"] == t_kill_eff) if t_kill_eff else None
    G_NOISE = {"t_kill_C": t_kill, "D_kill": D_kill, "coarse": None,
               "match_step_N0": None, "gm12_at_match": None, "pass": None}
    if D_kill is not None:
        G_NOISE["coarse"] = not ctrl_kills_fine
        M = next((r["step"] for r in disp_table["N0"]
                  if r["cum_disp"] >= D_kill), None)
        G_NOISE["match_step_N0"] = M
        G_NOISE["gm12_at_match"] = n0_gm12.get(M) if M is not None \
            else n0_gm12.get(G1.CK_NOISE[-1])
        G_NOISE["displacement_unmatched"] = M is None
        G_NOISE["pass"] = bool(G_NOISE["gm12_at_match"] is not None
                               and G_NOISE["gm12_at_match"] <= G1.SHUT_BAR)
    log(f"G-NOISE: N0 g-12 at match {G_NOISE['gm12_at_match']} "
        f"(D_kill {D_kill}): "
        f"{'PASS' if G_NOISE['pass'] else 'FAIL'}")

    gates_pass = bool(G_ROOT0["pass"] and G_CTRL["pass"] and G_PIN["pass"]
                      and G_NOISE["pass"])

    # ---- WALL verdicts
    def arm_verdict(tag, cks):
        g = light_gm12(tag)
        vals = [g[s] for s in cks if s in g]
        maintains = bool(vals and all(v >= G1.MAINTAIN_BAR for v in vals))
        dies_by_50 = bool(g.get(50, 1.0) <= G1.SHUT_BAR)
        first_under = next((s for s in cks if g.get(s, 1.0) <= G1.SHUT_BAR), None)
        return {"g_m12": g, "min_gm12": min(vals) if vals else None,
                "maintains": maintains, "dies_by_50": dies_by_50,
                "first_ck_le_bar": first_under}

    wall = {tag: arm_verdict(tag, G1.CK_WASH) for tag in ("C", "W1", "W2", "W3")}
    F1_WALL_DEAF = bool(wall["W1"]["dies_by_50"])
    W3_maintains = wall["W3"]["maintains"]
    WALL_CLIFF_INSIDE = bool(wall["W1"]["maintains"] and
                             wall["W3"]["dies_by_50"])
    if wall["W2"]["maintains"]:
        cliff = (1.4, 4.2)
    elif wall["W2"]["dies_by_50"]:
        cliff = (0.7, 1.4)
    else:
        cliff = None
    # F2: survival must order with R against measured D_kill
    survival_order = [wall[t]["min_gm12"] for t in ("W1", "W2", "W3")]
    orders_with_R = all(survival_order[i] >= survival_order[i + 1] - 1e-12
                        for i in range(2))
    cliff_ok = True
    if cliff is not None and D_kill is not None:
        cliff_ok = bool(cliff[0] <= 2.0 * D_kill and cliff[1] >= 0.5 * D_kill)
    F2_CLIFF_MISPLACED = bool(W3_maintains or not orders_with_R
                              or not cliff_ok)

    # ---- NOISE verdicts (at pinned R)
    noise = {tag: arm_verdict(tag, G1.CK_NOISE) for tag in noise_tags}
    ce_r_root = root_cells["ce_r"]
    for tag in noise_tags:
        g = arms[tag]["traj"]
        noise[tag]["ce_r_at_10"] = next((t["ce_r"] for t in g
                                         if t["step"] == G1.CK_NOISE[-1]), None)
        noise[tag]["ce_r_within_bar"] = bool(
            noise[tag]["ce_r_at_10"] is not None
            and noise[tag]["ce_r_at_10"] <= ce_r_root + G1.CE_NOISE_BAR)
    F3_NOISE_PENETRATES = bool(
        any(light_gm12(t).get(s, 1.0) <= G1.SHUT_BAR for t in ("N1", "N2")
            for s in G1.CK_NOISE))
    NOISE_SPARED = bool(
        all(light_gm12(t).get(s, 0.0) >= G1.MAINTAIN_BAR
            for t in ("N1", "N2") for s in G1.CK_NOISE
            if s in light_gm12(t))
        and noise["N1"]["ce_r_within_bar"] and noise["N2"]["ce_r_within_bar"]
        and G_PIN["per_arm"]["N1"]["pass"] and G_PIN["per_arm"]["N2"]["pass"])
    # the anisotropy fork: N1's floor vs W1's floor over the SAME horizon
    w1_short = [light_gm12("W1").get(s) for s in G1.CK_NOISE
                if s in light_gm12("W1")]
    n1_short = [light_gm12("N1").get(s) for s in G1.CK_NOISE
                if s in light_gm12("N1")]
    aniso = {"W1_floor_h10": min(w1_short) if w1_short else None,
             "N1_floor_h10": min(n1_short) if n1_short else None}
    aniso["dip"] = (aniso["W1_floor_h10"] - aniso["N1_floor_h10"]) \
        if None not in aniso.values() else None
    aniso["direction_thin_for_noise"] = bool(
        aniso["dip"] is not None and aniso["dip"] >= G1.ANISO_BAR)

    # ---- PIN verdict (FLAT-AT-PIN / F4)
    w1g = light_gm12("W1")
    flat_delta = abs(w1g.get(300, float("nan")) - w1g.get(50, float("nan"))) \
        if 50 in w1g and 300 in w1g else None
    FLAT_AT_PIN = bool(flat_delta is not None and flat_delta <= G1.FLAT_BAR)
    seq = [w1g[s] for s in (50, 100, 200, 300) if s in w1g]
    monotone_decline = all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
    F4_ERODES = bool(len(seq) >= 2 and monotone_decline
                     and (seq[0] - seq[-1]) >= G1.ERODE_BAR)

    # ---- COSTS
    def ce_at(tag, step):
        return next((t["ce_batch"] for t in arms[tag]["traj"]
                     if t["step"] == step), None)

    ce_tax = {"W1_ce300": ce_at("W1", 300), "C_ce300": ce_at("C", 300)}
    ce_tax["delta"] = (ce_tax["W1_ce300"] - ce_tax["C_ce300"]) \
        if None not in ce_tax.values() else None
    WALL_TAXES = bool(ce_tax["delta"] is not None
                      and ce_tax["delta"] >= G1.CE_TAX_BAR)
    WALL_FREE = bool(ce_tax["delta"] is not None
                     and abs(ce_tax["delta"]) < G1.CE_TAX_BAR)

    w1_300 = flat_cells(batteries_all["W1"]["300"]) \
        if "300" in batteries_all.get("W1", {}) else None
    anatomy = None
    if w1_300 is not None:
        band = batteries_all["W1"]["300"]["census_old"]["rows"] \
            if "census_old" in batteries_all["W1"]["300"] else {}
        anatomy = {
            "row0_strength": w1_300.get("row0_strength"),
            "row0_ratio_vs_root": (w1_300.get("row0_strength")
                                   / root_cells.get("row0_strength"))
            if w1_300.get("row0_strength") is not None
            and root_cells.get("row0_strength") else None,
            "row0_ge_half_root": bool(
                w1_300.get("row0_strength") is not None
                and root_cells.get("row0_strength") is not None
                and w1_300["row0_strength"]
                >= 0.5 * root_cells["row0_strength"]),
            "site_read_span": w1_300.get("site_read_span"),
            "span_ge_0p7": bool((w1_300.get("site_read_span") or 0)
                                >= 0.7),
            "band121_129_content": {r: band[str(r)]["content"]
                                    for r in range(121, 130)
                                    if str(r) in band} if band else None,
            "band_all_content": bool(band and all(
                band[str(r)]["content"] for r in range(121, 130)
                if str(r) in band)),
            "held30_gm12": w1_300.get("held30_gm12"),
            "held30_ge_bar": bool((w1_300.get("held30_gm12") or 0)
                                  >= G1.HELD30_BAR),
            "ce_r_at_300": w1_300.get("ce_r"),
        }

    # ---- the composed verdict (g1's order; the continuity wording)
    fmt = lambda v: "n/a" if v is None else f"{v:.4f}"

    if not gates_pass:
        failed = [k for k, g in (("G-ROOT", G_ROOT0), ("G-CTRL", G_CTRL),
                                 ("G-PIN", G_PIN), ("G-NOISE", G_NOISE))
                  if not g["pass"]]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a registered gate failed — nothing adjudicated per the "
                  f"spec's abort clause; failed: {failed}; the full record "
                  f"is reported (root g-12 {fmt(root_cells['gm12'])}, "
                  f"C g-12 trace "
                  + " -> ".join(f"+{s}:{fmt(v)}" for s, v in
                                sorted(c_gm12.items())) + ")")
    elif F1_WALL_DEAF:
        verdict = "F1 WALL-DEAF"
        clause = (f"W1 DIED (g-12 {fmt(wall['W1']['g_m12'].get(50))} <= "
                  f"{G1.SHUT_BAR} at +50; first checkpoint under bar: "
                  f"+{wall['W1']['first_ck_le_bar']}) DESPITE G-PIN-verified "
                  f"confinement (max raw |d| "
                  f"{fmt(G_PIN['per_arm']['W1']['max_raw_disp_at_ckpt'])} <= "
                  f"{0.7 + G1.PIN_FUZZ_BAR:.2f}) at R = 0.7 <= a third of "
                  f"the raw basin prior low end (2.5) on the line the prior "
                  f"was measured on — the readout died WITHOUT displacement: "
                  f"the kill is not displacement-limited; the no-basin law "
                  f"upgrades to a stronger architectural necessity.")
    elif F2_CLIFF_MISPLACED:
        verdict = "F2 WALL-CLIFF-MISPLACED"
        clause = (f"survival does not order with R against measured D_kill "
                  f"{fmt(D_kill)}: W1 min {fmt(survival_order[0])}, W2 min "
                  f"{fmt(survival_order[1])}, W3 min {fmt(survival_order[2])} "
                  f"(W3 maintains: {W3_maintains}; orders with R: "
                  f"{orders_with_R}; cliff interval {cliff} vs D_kill) — "
                  f"the L2-position currency fails on its own line.")
    elif WALL_CLIFF_INSIDE:
        verdict = "WALL-HOLDS (WALL-CLIFF-INSIDE)"
        w2_txt = ("MAINTAINS" if wall["W2"]["maintains"]
                  else "DIES" if wall["W2"]["dies_by_50"]
                  else f"NEITHER maintains nor dies (min g-12 "
                       f"{fmt(wall['W2']['min_gm12'])} — a dip-and-recover "
                       f"at pin: below the {G1.MAINTAIN_BAR} maintain bar, "
                       f"above the {G1.SHUT_BAR} death bar; +300 "
                       f"{fmt(wall['W2']['g_m12'].get(300))})")
        clause = (f"W1 maintains (min g-12 {fmt(wall['W1']['min_gm12'])} >= "
                  f"{G1.MAINTAIN_BAR} at every checkpoint; +300 "
                  f"{fmt(wall['W1']['g_m12'].get(300))}) while W3 dies by "
                  f"+50 (g-12 {fmt(wall['W3']['g_m12'].get(50))}; first "
                  f"under +{wall['W3']['first_ck_le_bar']}) — the survival "
                  f"cliff lives inside (0.7, 4.2]; W2 "
                  + w2_txt
                  + (f" => cliff in {cliff}" if cliff is not None
                     else " => the ladder's middle rung does not localize "
                          "the cliff under the maintains/dies dichotomy "
                          "(bracketed only by W1/W3 as inside (0.7, 4.2])")
                  + f"; the control's measured D_kill = "
                  f"{fmt(D_kill)} L2 (the raw 2.74M prior {BASIN_PRIOR}; "
                  f"e185 stored 2.489) — the wall re-measures the basin "
                  f"width dynamically against e180's t* extrapolation, on "
                  f"the line both were measured on.")
    else:
        verdict = "TEXTURE"
        clause = (f"no wall clause fired cleanly: W1 min "
                  f"{fmt(wall['W1']['min_gm12'])}, W2 min "
                  f"{fmt(wall['W2']['min_gm12'])}, W3 min "
                  f"{fmt(wall['W3']['min_gm12'])}, C +50 "
                  f"{fmt(c_gm12.get(50))} — full trajectories reported, no "
                  f"bar shopping.")

    log("=" * 78)
    log(f"G1B VERDICT: {verdict}")
    for tag in ("C", "W1", "W2", "W3"):
        g = wall[tag]["g_m12"]
        d = disp_table[tag]
        log(f"  {tag}: g-12 " + " -> ".join(f"+{s}:{v:.4f}"
                                            for s, v in sorted(g.items())))
        log(f"  {tag}: |d| " + " -> ".join(
            f"+{r['step']}:{r['cum_disp']:.3f}" for r in d if "g_m12_light" in r))
    for tag in noise_tags:
        g = light_gm12(tag)
        log(f"  {tag}: g-12 " + " -> ".join(f"+{s}:{v:.4f}"
                                            for s, v in sorted(g.items()))
            + f" | CE_R@10 {noise[tag]['ce_r_at_10']:.4f} "
              f"(root {ce_r_root:.4f})")
    log(f"  FLAT-AT-PIN: {FLAT_AT_PIN} (|delta(+300,+50)| = {flat_delta})"
        f" | F4 ERODES: {F4_ERODES}")
    log(f"  NOISE-SPARED-BY-WALL: {NOISE_SPARED} | F3 NOISE-PENETRATES: "
        f"{F3_NOISE_PENETRATES} | aniso dip {aniso['dip']}")
    log(f"  WALL-TAXES-ADAPTATION: {WALL_TAXES} (dCE@300 "
        f"{ce_tax['delta']})")
    log(f"  {clause}")
    log("=" * 78)

    # =====================================================================
    # OUTPUTS
    # =====================================================================
    trace = {}
    for tag in arms:
        rows = [{"freeze_steps": 0, **{k: root_cells[k] for k in
                 ("gm12", "g0", "gp12", "held30_gm12", "held30_g0", "ce_r",
                  "site_read_onset", "site_read_span")}}]
        g0_light = {t["step"]: t["g0_mean_pz"] for t in arms[tag]["traj"]
                    if "g0_mean_pz" in t}
        for s in sorted(arms[tag]["sds"]):
            c = None
            if str(s) in batteries_all.get(tag, {}):
                c = flat_cells(batteries_all[tag][str(s)])
            lg = light_gm12(tag).get(s)
            if c is not None:
                rows.append({"freeze_steps": s, **c})
            elif lg is not None:
                rows.append({"freeze_steps": s, "gm12": lg,
                             "g0": g0_light.get(s),
                             "ce_r": next((t["ce_r"] for t in arms[tag]["traj"]
                                           if t["step"] == s), None)})
        trace[tag] = rows

    metrics = {
        "experiment": "g1b_continuity",
        "date": common.now_iso(),
        "design": "scratch/g1_design.md sections 4-6 bars VERBATIM (frozen "
                  "at g1's dispatch; now live) + section 1's CONTINGENCY "
                  "CELL (the registered 2.74M continuity cell); g1's "
                  "registered discharge after the 0.84M G-ROOT abort",
        "registered_prediction": G1B_PREDICTION,
        "question": ("g1's question on the line its bars were calibrated "
                     "on: does commit(R) + a hard L2 wall (flat interior, "
                     "all directions, forward semantics) install a well in "
                     "the 2.74M pre-LN organism whose consolidated root "
                     "actually expresses (g-12 0.9156)? The R ladder "
                     "{0.7, 1.4, 4.2} re-measures the basin width "
                     "dynamically against e180's t* extrapolation and "
                     "e185's stored D_kill 2.489."),
        "continuity": {
            "organism": f"Cfg 6L/6H/192d block 256 = {G1B_PARAMS} params "
                        "(the e131/e048 line — the organism behind every "
                        "stored wash number)",
            "root": {"checkpoint": f"runs/checkpoints/{ROOT_CK}",
                     "bitexact_gate": G_BITEXACT, "root_cells": root_cells,
                     "prior_root_gm12": PRIORS["root_gm12"],
                     "delta_vs_prior": G_ROOT0["delta_vs_prior"]},
            "priors_same_line": PRIORS,
            "basin_prior_raw": list(BASIN_PRIOR),
            "R_ladder_note": "R carried VERBATIM from g1 (absolute L2): "
                             "{0.28, 0.56, 1.68} x the raw 2.5 low end — "
                             "W1/W2 inside, W3 beyond e185's D_kill 2.489",
        },
        "arms": {
            tag: {
                "desc": desc, "R": R, "target_mode": mode,
                "noise_seed": nseed or None,
                "ckpt_steps": list(cks),
                "steps_ran": arms[tag]["steps_ran"],
                "device": arms[tag]["device"],
                "traj": [{k: v for k, v in t.items() if k != "d_proj"}
                         for t in arms[tag]["traj"]],
                "y_stats": arms[tag]["y_stats"],
                "missing_checkpoints": [s for s in cks
                                        if s not in arms[tag]["sds"]],
            } for (tag, R, mode, nseed, cks, desc) in ARM_SPECS
        },
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": G1.PRE, "post_cap": G1.POST_CAP,
                     "neutral_bank": G_ANCHOR["neutral_bank"],
                     "measure_dial": "e176n's measure() (the e131 dial set) "
                                     "on evl_load (settle+disarm PIVOT)"},
        "displacement": {
            "currency": ("cumulative ||theta_t - theta_0||_2 over all "
                         f"{G1B_PARAMS} trainable parameters (fp32, CPU, "
                         "measured per step) + per-step increments + the "
                         "projected displacement min(d, R); cosines vs arm "
                         "C at shared checkpoints"),
            "table": disp_table,
            "theta0_norm": {t: arms[t]["theta0_norm"] for t in arms},
            "wall_fuzz_registered": "one AdamW step ~ lr*sqrt(P) = 1.66 L2 "
                                    "at lr 1e-3, 2.74M params (0.92 at "
                                    "0.84M); the VERBATIM G-PIN bar R + 1.5 "
                                    "held — measured tangential increments",
        },
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_POOL": G_POOL, "G_ANCHOR": G_ANCHOR, "G_BITEXACT":
                  G_BITEXACT, "G_ROOT": G_ROOT0, "G_CTRL": G_CTRL,
                  "G_PIN": G_PIN, "G_NOISE": G_NOISE, "G_BITROOT": G_BITROOT,
                  "G_INPUTS": {k: v for k, v in G_INPUTS.items()
                               if k != "per_step"} | {"per_step": {
                                   s: v["identical"] for s, v in
                                   G_INPUTS["per_step"].items()}},
                  "G_TARGETS": G_TARGETS, "G_SURG": gates_surg},
        "traces": trace,
        "batteries": batteries_all,
        "adjudication": {
            "order": "GATES -> WALL -> NOISE -> PIN -> COSTS (frozen)",
            "gates_pass": gates_pass,
            "wall": wall, "noise": noise,
            "F1_WALL_DEAF": F1_WALL_DEAF,
            "F2_WALL_CLIFF_MISPLACED": F2_CLIFF_MISPLACED,
            "WALL_CLIFF_INSIDE": WALL_CLIFF_INSIDE,
            "cliff_interval": cliff,
            "survival_order_with_R": orders_with_R,
            "F3_NOISE_PENETRATES": F3_NOISE_PENETRATES,
            "NOISE_SPARED_BY_WALL": NOISE_SPARED,
            "anisotropy_fork": aniso,
            "FLAT_AT_PIN": FLAT_AT_PIN, "flat_delta": flat_delta,
            "F4_ERODES_AT_PIN": F4_ERODES,
            "WALL_TAXES_ADAPTATION": WALL_TAXES,
            "WALL_FREE": WALL_FREE, "ce_tax": ce_tax,
            "W1_anatomy_at_300": anatomy,
            "verdict": verdict, "clause": clause,
        },
        "honesty_reflex": {
            "envelope": (f"{G1B_PARAMS} trainable params — above the lab's "
                "old 1M default, inside the <=100M free tier (common.py, "
                "Devansh 2026-09-27); the stated reason registered in g1's "
                "spec section 1 stands: CONTINUITY — the arc's bars "
                "(D_kill 2.489, root 0.9156, the jitter-to--12 "
                "generalization) are this line's numbers, and the 0.84M "
                "discharge aborted at G-ROOT 0.2536 < 0.78."),
            "intervention_not_logits": ("the wall IS the intervention: "
                "committed vs uncommitted arms share bit-identical step-0 "
                "weights (G_BITROOT max|diff| = 0.0), bit-identical "
                "per-step input streams (md5-gated, G_INPUTS) and identical "
                "seeds; the only deltas are the commit event and the noise "
                "targets. G-PIN verifies the geometry did what it claims "
                "(raw |d| <= R + 1.5 at every checkpoint) BEFORE any "
                "behavioral clause is read; the displacement table "
                "co-reports raw and projected L2."),
            "step_1_equality": ("every arm's STEP-1 WEIGHTS are "
                "bit-identical to C's (first forward at d=0 inside the "
                "ball; the wall first acts at forward 2). The +1 READINGS "
                "then differ by arm SEMANTICS, not stream: C/N0/W3 read "
                "the free step-1 state (W3's +1 == C's +1 exactly — no "
                "projection below R=4.2 — re-measuring the arc's stored "
                "+1 0.678), while W1/W2/N1/N2's light evals run on the "
                "ARMED twin, i.e. the settled/projected state entering "
                "forward 2: W1's +1 is the corpus step's displacement "
                "REVERSED onto the ball, N1's +1 the noise damage under "
                "the same reversal — damage-vs-displacement read at +1."),
            "size_matched_control": ("every verdict is adjudicated against "
                "arm C — same class (uncommitted CommittedGPT = bit-identical "
                "TinyGPT), same root, same instruments, same process; the "
                "same-line stored numbers (root 0.9156, +1 0.678, +2 0.0271, "
                "D_kill 2.489) are co-reported priors, now directly "
                "comparable modulo device floats (e176N/e185 ran CPU)."),
            "single_seed": ("one trajectory per arm (wash draws 10902 / "
                "noise 18501-2), n=1 per cell — the arc's honesty "
                "convention (e185's own); the replication debt stands "
                "(g1 spec section 8 priority (1))."),
            "wall_blind_spot": ("the wall never protects the FIRST step: "
                "commit happens at d=0, so step 1's AdamW update (the full "
                "~1.66 L2 at this size) always lands before the first "
                "projection; the wall caps CUMULATIVE displacement at R + "
                "fuzz, it does not shrink any single step — the design's "
                "own registered semantics, not a bug."),
            "gpin_at_this_size": ("the registered one-step fuzz formula "
                f"lr*sqrt(P) = {G1.FT_LR * (G1B_PARAMS ** 0.5):.3f} L2 here "
                "(1.5-bar margin kept VERBATIM); measured beyond-wall "
                "increments are tangential (concentration of measure: "
                "post-projection Adam steps in 2.74M-d are near-orthogonal "
                "to the radial direction), so the verbatim bar held with "
                "margin — reported, not tuned."),
            "bars_anchored": ("dies/maintain reuse the arc's absolute "
                "home-battery bars (0.27 / 0.50) on the e131 dial set's "
                "ruler; the R grid, seeds, checkpoints and adjudication "
                "order were frozen in scratch/g1_design.md before g1's "
                "implementation; g1b adds NO bar — no bar shopping."),
        },
        "trims": G1.trims, "deviations": deviations,
        "device_events": G1.device_events,
        "device_policy": {"parked": G1.GPU_PARKED, "reason": G1.PARK_REASON},
        "ckpt_inventory": CKPT_INVENTORY,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": G1B_PARAMS,
                   "trainable_params": G1B_PARAMS,
                   "anchor_state_bytes": G1B_PARAMS * 4,
                   "R_ladder": list(R_LADDER),
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "continuity.png", trace, disp_table, wall, noise, c_gm12,
         D_kill, verdict, clause, ce_tax, ce_r_root, root_cells,
         wall_taxes=WALL_TAXES, flat=(FLAT_AT_PIN, flat_delta),
         f1=F1_WALL_DEAF, f3=F3_NOISE_PENETRATES, f4=F4_ERODES,
         noise_spared=(NOISE_SPARED, aniso.get("dip")), cliff=cliff)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'continuity.png'}, "
        f"{len(CKPT_INVENTORY)} checkpoints in runs/checkpoints/g1b_*.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot
# g1's plot VERBATIM (reads g1b's globals), with the continuity relabeling:
# G1B title, 2,739,072 params, the raw basin prior, e185's stored D_kill.

def plot(path, trace, disp_table, wall, noise, c_gm12, D_kill, verdict,
         clause, ce_tax, ce_r_root, root_cells, *, wall_taxes=False,
         flat=(False, None), f1=False, f3=False, f4=False,
         noise_spared=(False, None), cliff=None):
    """THE R-DIAL figure (g1's, relabeled): fact survival vs wash steps,
    survival vs R (the ladder) against the raw prior + stored D_kill, the
    cost panel, the displacement trajectories vs the walls, the noise
    panel, and the verdict."""
    import textwrap
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))
    cols = {"C": "crimson", "W1": "seagreen", "W2": "royalblue",
            "W3": "darkorange", "N0": "crimson", "N1": "seagreen",
            "N2": "mediumseagreen"}
    lbls = {"C": "C — control (no wall)", "W1": "W1 — commit(0.7)",
            "W2": "W2 — commit(1.4)", "W3": "W3 — commit(4.2)",
            "N0": "N0 — labels-noise, no wall",
            "N1": "N1 — labels-noise, R=0.7",
            "N2": "N2 — shuffled-target, R=0.7"}
    marks = {"C": "o", "W1": "s", "W2": "^", "W3": "v",
             "N0": "o", "N1": "s", "N2": "^"}

    def gm12_series(tag):
        pts = [(r["freeze_steps"], r["gm12"]) for r in trace[tag]]
        return [p[0] for p in pts], [p[1] for p in pts]

    # (0,0) THE HEADLINE: g-12 vs wash steps
    ax = axes[0, 0]
    for tag in ("C", "W1", "W2", "W3"):
        xs, ys = gm12_series(tag)
        ax.plot(xs, ys, marks[tag] + "-", ms=8, lw=2.2, color=cols[tag],
                alpha=0.9, label=lbls[tag])
    for yv, col in ((G1.MAINTAIN_BAR, "seagreen"), (G1.SHUT_BAR, "tab:purple")):
        ax.axhline(yv, ls="--", lw=1.1, color=col, alpha=0.8)
    ax.axhline(G1.EXPRESS_BAR, ls=":", lw=1.0, color="gray", alpha=0.7)
    ax.annotate(f"root {trace['C'][0]['gm12']:.3f}", (0, trace["C"][0]["gm12"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax.set_xlabel("neutral-wash steps from the committed root")
    ax.set_ylabel("g-12 (absolute mean p(Z), install-60 battery)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title("THE ANCHORED BALL vs the wash — fact survival (2.74M)",
                 fontsize=10)

    # (0,1) THE R LADDER: survival vs R at the three horizons
    ax = axes[0, 1]
    Rs = [0.0] + list(R_LADDER)
    tags_by_R = ["C", "W1", "W2", "W3"]
    for step, col, mk in ((2, "gray", "o"), (50, "tab:purple", "s"),
                          (300, "seagreen", "D")):
        ys = []
        for tag in tags_by_R:
            g = wall[tag]["g_m12"]
            ys.append(g.get(step))
        ax.plot(Rs, ys, mk + "-", ms=9, lw=2.0, color=col,
                label=f"g-12 at +{step}")
    ax.axvspan(BASIN_PRIOR[0], BASIN_PRIOR[1], color="gold", alpha=0.18,
               label=f"basin prior (raw 2.74M) {BASIN_PRIOR}")
    ax.axvline(PRIORS["e185_D_kill"], ls="-.", lw=1.4, color="gray",
               alpha=0.7, label=f"e185 stored D_kill "
                                f"{PRIORS['e185_D_kill']:.3f}")
    if D_kill is not None:
        ax.axvline(D_kill, ls=":", lw=1.8, color="k", alpha=0.8,
                   label=f"measured D_kill {D_kill:.3f}")
    for yv, col in ((G1.MAINTAIN_BAR, "seagreen"), (G1.SHUT_BAR, "tab:purple")):
        ax.axhline(yv, ls="--", lw=1.0, color=col, alpha=0.8)
    ax.set_xticks(Rs)
    ax.set_xticklabels(["C\n(no wall)"] + [f"R={r}" for r in R_LADDER])
    ax.set_xlabel("the wall dial R (L2 over all 2,739,072 params)")
    ax.set_ylabel("g-12")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7, loc="center right")
    ax.set_title("THE R-DIAL: fact survival vs the wall radius", fontsize=10)

    # (0,2) THE COST PANEL: adaptation CE (in-batch corpus CE + CE_R)
    ax = axes[0, 2]
    for tag in ("C", "W1", "W2", "W3"):
        rows = [r for r in disp_table[tag]]
        xs = [r["step"] for r in rows]
        ys = [r["ce_batch"] for r in rows]
        ax.plot(xs, ys, "-", lw=1.6, color=cols[tag], alpha=0.8,
                label=f"{tag} in-batch CE")
    if ce_tax["delta"] is not None:
        ax.annotate(f"WALL-TAX dCE@300 {ce_tax['delta']:+.3f}",
                    (0.03, 0.05), xycoords="axes fraction", fontsize=8,
                    weight="bold",
                    color="darkred" if wall_taxes else "seagreen")
    ax.set_xlabel("wash step")
    ax.set_ylabel("in-batch corpus CE (the wash's own adaptation)")
    ax.legend(fontsize=7.5)
    ax.set_title("WALL-TAXES-ADAPTATION — the stream's learning vs the wall",
                 fontsize=9.5)

    # (1,0) DISPLACEMENT trajectories vs the walls
    ax = axes[1, 0]
    for tag in ("C", "W1", "W2", "W3"):
        rows = [r for r in disp_table[tag] if "g_m12_light" in r]
        ax.plot([r["step"] for r in rows], [r["cum_disp"] for r in rows],
                marks[tag] + "-", ms=6, lw=1.8, color=cols[tag],
                alpha=0.9, label=lbls[tag])
    for R, col in zip(R_LADDER, ("seagreen", "royalblue", "darkorange")):
        ax.axhline(R, ls="--", lw=1.0, color=col, alpha=0.6)
        ax.axhline(R + G1.PIN_FUZZ_BAR, ls=":", lw=0.8, color=col, alpha=0.5)
        ax.annotate(f"R={R} (+pin {R + G1.PIN_FUZZ_BAR})", (0.99, R),
                    xycoords=("axes fraction", "data"), ha="right",
                    fontsize=7, color=col)
    if D_kill is not None:
        ax.axhline(D_kill, ls=":", lw=1.8, color="k", alpha=0.8)
        ax.annotate(f"D_kill {D_kill:.3f}", (0.01, D_kill),
                    xycoords=("axes fraction", "data"), fontsize=7.5)
    ax.set_xlabel("wash step")
    ax.set_ylabel(r"raw $\|\theta_t-\theta_0\|_2$ at checkpoints")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("DISPLACEMENT TRAJECTORIES vs the walls (G-PIN)", fontsize=10)

    # (1,1) THE NOISE PANEL
    ax = axes[1, 1]
    for tag in ("N0", "N1", "N2"):
        xs, ys = gm12_series(tag)
        ax.plot(xs, ys, marks[tag] + "-", ms=9, lw=2.2, color=cols[tag],
                alpha=0.9, label=lbls[tag])
    w1x, w1y = gm12_series("W1")
    w1f = [(x, y) for x, y in zip(w1x, w1y) if x <= 10]
    if w1f:
        ax.plot([p[0] for p in w1f], [p[1] for p in w1f], "kx--", ms=8,
                lw=1.2, alpha=0.6, label="W1 (corpus, same R) — aniso ref")
    for yv, col in ((G1.MAINTAIN_BAR, "seagreen"), (G1.SHUT_BAR, "tab:purple")):
        ax.axhline(yv, ls="--", lw=1.0, color=col, alpha=0.8)
    for tag in ("N1", "N2"):
        c10 = noise[tag].get("ce_r_at_10")
        if c10 is not None:
            ax.annotate(f"{tag} CE_R@10 {c10:.2f} "
                        f"(root {ce_r_root:.2f}+0.3)",
                        (0.03, 0.12 if tag == "N1" else 0.05),
                        xycoords="axes fraction", fontsize=7,
                        color=cols[tag])
    ax.set_xlabel("noise-wash steps")
    ax.set_ylabel("g-12")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title("THE NOISE KILL under the wall (e185's battery, 2.74M)",
                 fontsize=10)

    # (1,2) THE VERDICT PANEL
    ax = axes[1, 2]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, "G1B — THE 2.74M CONTINUITY CELL", fontsize=11, va="top",
            family="monospace", weight="bold")
    y -= 0.052
    gm = {t: wall[t]["g_m12"] for t in ("C", "W1", "W2", "W3")}
    for tag in ("C", "W1", "W2", "W3"):
        seq = " -> ".join(f"+{s}:{v:.4f}" for s, v in sorted(gm[tag].items()))
        ax.text(0.02, y, f"  {tag}: {seq}", fontsize=6.8, va="top",
                family="monospace", color=cols[tag])
        y -= 0.030
    y -= 0.008
    ax.text(0.02, y, f"  FLAT-AT-PIN {flat[0]} (|d| {flat[1]}) | "
            f"F1 {f1} | F4 {f4}", fontsize=7.2, va="top",
            family="monospace")
    y -= 0.034
    ax.text(0.02, y, f"  NOISE-SPARED {noise_spared[0]} | F3 {f3} | aniso "
            f"dip {noise_spared[1]}", fontsize=7.2, va="top",
            family="monospace")
    y -= 0.034
    ax.text(0.02, y, f"  D_kill {D_kill if D_kill is None else round(D_kill, 3)}"
            f" (e185 stored {round(PRIORS['e185_D_kill'], 3)}) | cliff {cliff} "
            f"| WALL-TAX "
            f"{ce_tax['delta'] if ce_tax['delta'] is None else round(ce_tax['delta'], 3)}",
            fontsize=7.2, va="top", family="monospace")
    y -= 0.048
    ax.text(0.02, y, f"VERDICT: {verdict}", fontsize=9.5, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.042
    for wd in textwrap.wrap(clause, width=88, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.6, va="top", family="monospace")
        y -= 0.026

    fig.suptitle("G1B — THE 2.74M CONTINUITY CELL: g1's anchored ball on the "
                 f"line its bars were calibrated on -> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
