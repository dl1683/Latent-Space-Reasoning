"""site_oracle_v1: cross-fitted site-oracle control (Codex round 21). For fold f and tag k, the direction d_{f,k} is the unit
mean, over the 16 names NOT in f, of the gradient w.r.t. the block-12 slot residual of the first-token margin
m_k = l_k - mean_{j!=k} l_j at the query; it is injected ONCE (delta = 0.25 ||h_slot|| d) on the 8 held-out names of fold f.
Arms per name: eight target directions (cross-fitted), cue, zero-hook, eight fixed random unit directions. Reuses LM.

    python experiments/run_site_oracle.py --config experiments/config/site_oracle_v1.json [--preflight]
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from run_onewrite_state import LM, strict


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--preflight", action="store_true"); a = ap.parse_args(); T0 = time.time()
    cfg = json.load(open(a.config, encoding="utf-8")); out = f"experiments/results/{cfg['name']}"; os.makedirs(out, exist_ok=True); tag = "preflight" if a.preflight else "run"
    logf = open(os.path.join(out, f"{tag}.log"), "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    shas = {k: hashlib.sha256(open(v, "rb").read()).hexdigest() for k, v in (("runner", __file__), ("config", a.config), ("machinery", os.path.join(os.path.dirname(__file__), "run_onewrite_state.py")))}
    M = LM(cfg); T = cfg["tags"]; TAGS = [t.upper() for t in T]; names = cfg["names_train"]; K = len(T); G = cfg["gates"]; D = M.m.config.hidden_size
    first = [M.ids(" " + t)[0] for t in T]; assert len(set(first)) == K
    prompt = lambda name: M.ids(cfg["filler"] + cfg["valid_line"] + cfg["query"].format(NAME=name.upper()))
    res = {"config": cfg["name"], "sha256": shas, "revision": M.sp.revision, "fold_hash": hashlib.sha256(json.dumps(cfg["folds"]).encode()).hexdigest()}; save = lambda: json.dump(res, open(os.path.join(out, f"{tag}_result.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else float(o))
    g = torch.Generator().manual_seed(cfg["random_seed"]); Rnd = torch.randn(K, D, generator=g); Rnd = Rnd / Rnd.norm(dim=1, keepdim=True)
    dec = lambda ids, delta: (lambda txt: (txt,) + strict(txt, TAGS))(M.decode(ids, delta, TAGS))
    if a.preflight:                                                                             # no directions: cue, zero-hook identity, slot norms
        rows = []
        for name in names:
            ids = prompt(name); tc, cc, okc = dec(ids, None); tz, cz, okz = dec(ids, torch.zeros(D)); rows.append({"name": name, "cue": cc, "cue_ok": okc, "zero_eq": tz == tc, "slot_norm": M.telemetry["slot_norm"]})
        s = {"cue_completion": float(np.mean([r["cue_ok"] for r in rows])), "zero_hook_matches_cue": all(r["zero_eq"] for r in rows), "slot_norm_mean": float(np.mean([r["slot_norm"] for r in rows])), "cue_choices": [r["cue"] for r in rows]}
        res["preflight"] = s; save(); log(json.dumps(s)); log(f"PREFLIGHT {'PASS' if s['zero_hook_matches_cue'] else 'FAIL'} ({time.time()-T0:.0f}s)"); return
    # --- margin gradients at the slot residual for every name and tag (each name's gradient is used only by folds it is NOT in) ---
    grads = torch.zeros(len(names), K, D)
    for ni, name in enumerate(names):
        ids = prompt(name)
        for k in range(K):
            d0 = torch.zeros(D, requires_grad=True); lg = M._forward(ids, d0, False).logits[0, -1].float(); l = lg[first]
            m = l[k] - (l.sum() - l[k]) / (K - 1); grads[ni, k] = torch.autograd.grad(m, d0)[0].detach()
        if ni % 6 == 0: log(f"gradients {ni + 1}/{len(names)} ({time.time()-T0:.0f}s)")
    dirs = {}
    for fi, fold in enumerate(cfg["folds"]):
        others = [n for n in range(len(names)) if n not in fold]; mean = grads[others].mean(0); dirs[fi] = mean / mean.norm(dim=1, keepdim=True)     # (K, D) unit directions, cross-fitted
    res["direction_hash"] = hashlib.sha256(torch.stack([dirs[f] for f in range(3)]).numpy().tobytes()).hexdigest()
    # --- evaluation: every held-out name x every target direction, cue, zero-hook, random directions ---
    rows = []
    with torch.no_grad():
        for fi, fold in enumerate(cfg["folds"]):
            for ni in fold:
                ids = prompt(names[ni]); tc, cc, okc = dec(ids, None); tz, cz, okz = dec(ids, torch.zeros(D)); h = M.telemetry["slot_norm"]; row = {"name": names[ni], "fold": fi, "cue": cc, "cue_ok": okc, "zero_eq": tz == tc, "target": {}, "random": {}}
                for k in range(K):
                    tt, ct, okt = dec(ids, cfg["norm_clamp_fraction"] * h * dirs[fi][k]); row["target"][k] = {"choice": ct, "ok": okt, "follow": ct == TAGS[k], "tele": dict(M.telemetry)}
                    tr, cr, okr = dec(ids, cfg["norm_clamp_fraction"] * h * Rnd[k]); row["random"][k] = {"choice": cr, "ok": okr, "match": cr == TAGS[k]}
                rows.append(row); log(f"{names[ni]} (fold {fi}): cue={cc} targets=" + "".join("Y" if row["target"][k]["follow"] else "n" for k in range(K)) + " random=" + "".join("Y" if row["random"][k]["match"] else "n" for k in range(K)) + f" ({time.time()-T0:.0f}s)")
    rng = np.random.default_rng(cfg["random_seed"]); F = np.array([[r["target"][k]["follow"] for k in range(K)] for r in rows], float); per_name = F.mean(1)
    boot = lambda v: float(np.quantile([np.mean(rng.choice(v, len(v))) for _ in range(cfg["eval"]["bootstraps"])], 0.025))
    cue_match = np.array([np.mean([r["cue"] == TAGS[k] for k in range(K)]) for r in rows]); rand_match = np.array([np.mean([r["random"][k]["match"] for k in range(K)]) for r in rows])
    s = {"follow": float(F.mean()), "follow_lb": boot(per_name), "per_tag": F.mean(0).tolist(), "per_fold": [float(F[[r["fold"] == fi for r in rows]].mean()) for fi in range(3)], "cue_match": float(cue_match.mean()), "random_match": float(rand_match.mean()),
         "uplift_cue": float((per_name - cue_match).mean()), "uplift_cue_lb": boot(per_name - cue_match), "uplift_random": float((per_name - rand_match).mean()), "uplift_random_lb": boot(per_name - rand_match),
         "completion": {"target": float(np.mean([r["target"][k]["ok"] for r in rows for k in range(K)])), "random": float(np.mean([r["random"][k]["ok"] for r in rows for k in range(K)])), "cue": float(np.mean([r["cue_ok"] for r in rows]))}, "zero_hook_matches_cue": all(r["zero_eq"] for r in rows),
         "cap_active_rate": float(np.mean([r["target"][k]["tele"]["cap_active"] for r in rows for k in range(K)]))}
    if not (s["zero_hook_matches_cue"] and s["completion"]["cue"] >= G["completion_min"]): status = "INVALID — NO VERDICT"
    elif (s["completion"]["target"] >= G["completion_min"] and s["completion"]["random"] >= G["completion_min"] and s["follow"] >= G["follow_min"] and s["follow_lb"] > G["follow_lb_min"] and min(s["per_tag"]) >= G["per_tag_min"] and min(s["per_fold"]) >= G["per_fold_min"]
          and s["uplift_cue"] >= G["uplift_min"] and s["uplift_cue_lb"] > G["uplift_lb_min"] and s["uplift_random"] >= G["uplift_min"] and s["uplift_random_lb"] > G["uplift_lb_min"] and s["random_match"] <= G["random_max"]): status = "SITE-ORACLE PASS — BOUNDED EARLY-SLOT EIGHT-WAY LEXICAL CONTROL"
    else: status = "FAIL — BLOCK-12 SLOT/PROMPT EIGHT-WAY BOUNDED CONTROL"
    res["rows"] = rows; res["summary"] = s; res["status"] = status; res["seconds"] = time.time() - T0; save(); log(json.dumps(s, default=float)); log(f"STATUS: {status}")


if __name__ == "__main__":
    main()
