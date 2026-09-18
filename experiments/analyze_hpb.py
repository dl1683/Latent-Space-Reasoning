"""Score HP-B results against the frozen v2 predictions (theory/HD_POWER_LAW_ALLOCATION.md).

Reads experiments/results/hp1_spectral_law/*.json, recomputes N1-t controls
from stored norms with the corrected rand_frame (the XE job launched before
the ambient-dimension fix), and emits the verdict table.
"""
import json
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from hp_spectral_lib import zipf_p, rand_frame, fit_window, mf_fixed_point

RES = Path(__file__).parent / "results" / "hp1_spectral_law"
A3 = json.load(open(RES.parent / "hpa_random_frame" / "a3_transient.json"))


def n1t_from_norms(x, d, seed, reps=5):
    rng = np.random.default_rng(seed)
    return float(np.median([fit_window(rand_frame(x, d, rng), 0.9)["alpha"]
                            for _ in range(reps)]))


def tail_mass(n, d, s):
    p = zipf_p(n, s)
    return float(np.sort(p)[:n - d].sum())


def score_qsae(f):
    r = json.load(open(f))
    arm = (r["n"], r["d"], r["s"])
    out = dict(arm=arm, seed=r["seed"], converged=r["converged"],
               ext=r["extensions_used"], capped=r.get("capped_by_wall", False))
    tr = [(rec["t"], rec["bat"]["top90"]["alpha"],
           n1t_from_norms(np.array(rec["norms"]), r["d"], r["seed"] * 31 + rec["t"]))
          for rec in r["records"]]
    out["alpha_t"] = [(t, round(a, 3)) for t, a, _ in tr]
    out["n1t_t"] = [(t, round(a, 3)) for t, _, a in tr]
    out["undershoot_late"] = (tr[-1][1] < tr[-1][2] - 0.10)
    out["pc"] = r.get("pc_scoring")
    out["pc_pass"] = r.get("pc_pass")
    if r["converged"]:
        end = r["records"][-1]
        x = np.array(end["norms"])
        out["P_E"] = dict(alpha_top90=end["bat"]["top90"]["alpha"],
                          nonzero=int((x > 1e-8 * x.max()).sum()),
                          d=r["d"],
                          loss_minus_tailmass=r["final_loss_slope"],
                          loss=end["loss"], tail_mass=tail_mass(r["n"], r["d"], r["s"]))
        out["pa"] = r.get("pa_norm_fit")
        out["pb_adequate"] = r.get("pb_adequate")
    return out


def score_xe(f):
    r = json.load(open(f))
    tr = [(rec["t"], rec["bat"]["top90"]["alpha"],
           n1t_from_norms(np.array(rec["norms"]), r["d"], r["seed"] * 31 + rec["t"]))
          for rec in r["records"]]
    last = r["records"][-1]
    losses = [rec["loss"] for rec in r["records"]]
    slope = (losses[-1] - losses[-2]) / (last["t"] - r["records"][-2]["t"]) if len(losses) > 1 else np.nan
    return dict(seed=r["seed"], alpha_t=[(t, round(a, 3)) for t, a, _ in tr],
                n1t_t=[(t, round(a, 3)) for t, _, a in tr],
                stationary=r.get("gate_stationary"),
                loss_slope_per_step=slope,
                undershoot_late=(tr[-1][1] < tr[-1][2] - 0.10),
                window_stability=abs(last["bat"]["top50"]["alpha"] - last["bat"]["top90"]["alpha"]))


if __name__ == "__main__":
    print("=== QSAE grid ===")
    for f in sorted(RES.glob("qsae_*.json")):
        s = score_qsae(f)
        print(f"{s['arm']} seed{s['seed']} conv={s['converged']} ext={s['ext']} "
              f"alpha_t={s['alpha_t']}")
        print(f"   n1t_t={s['n1t_t']} undershoot_late={s['undershoot_late']} "
              f"pc_pass={s['pc_pass']}")
        if "P_E" in s:
            pe = s["P_E"]
            print(f"   P_E: alpha={pe['alpha_top90']:.3f} nonzero={pe['nonzero']}/d={pe['d']} "
                  f"loss={pe['loss']:.6f} tailmass={pe['tail_mass']:.6f} "
                  f"diff={pe['loss']-pe['tail_mass']:+.2e}")
    print("=== XE ===")
    for f in sorted(RES.glob("xe_*.json")):
        s = score_xe(f)
        print(f"seed{s['seed']} alpha_t={s['alpha_t']}")
        print(f"   n1t_t={s['n1t_t']} stationary={s['stationary']} "
              f"win_stab={s['window_stability']:.3f} undershoot={s['undershoot_late']}")
    print("=== nulls ===")
    for f in sorted(RES.glob("null_*.json")):
        r = json.load(open(f))
        if "records" in r:
            print(f"{f.name}: alpha_end={r['records'][-1]['bat']['top90']['alpha']:.3f}")
        else:
            print(f"{f.name}: max_rel_dev_vs_gd={r.get('max_rel_dev_vs_gd')}")
