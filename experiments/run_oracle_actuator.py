"""oracle_actuator_rung0: can a KNOWN hidden code, written once, make the frozen model recall its matching tag? (Codex round 20)
No encoder: a fixed hashed centred-simplex codebook (8 codes in R^16) is injected through a trainable zero-init J at the
block-12 'Internal record:' slot; only J is trained. Arms: own code / cue / zero-hook / wrong codes (7 per entity) / fixed
off-code random vectors; capped primary, uncapped replay diagnostic. Reuses LM (hook, cap telemetry, strict decoding).

    python experiments/run_oracle_actuator.py --config experiments/config/oracle_actuator_rung0.json [--smoke]
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from run_onewrite_state import LM, strict


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args(); T0 = time.time()
    cfg = json.load(open(a.config, encoding="utf-8")); out = f"experiments/results/{cfg['name']}"; os.makedirs(out, exist_ok=True); tag = "smoke" if a.smoke else "run"
    logf = open(os.path.join(out, f"{tag}.log"), "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    if a.smoke: cfg["train"]["steps"] = int(os.environ.get("SMOKE_STEPS", 40)); cfg["train"]["seeds"] = cfg["train"]["seeds"][:1]; cfg["eval"]["bootstraps"] = 200
    shas = {k: hashlib.sha256(open(v, "rb").read()).hexdigest() for k, v in (("runner", __file__), ("config", a.config), ("machinery", os.path.join(os.path.dirname(__file__), "run_onewrite_state.py")))}
    M = LM(cfg); T = cfg["tags"]; TAGS = [t.upper() for t in T]; names = cfg["names_train"]; G = cfg["gates"]; K = len(T)
    C = torch.tensor(np.sqrt(K / (K - 1)) * np.hstack([np.eye(K) - 1 / K, np.zeros((K, cfg["state_dim"] - K))]), dtype=torch.float32)   # centred-simplex codebook
    cb_hash = hashlib.sha256(C.numpy().astype(np.float64).tobytes()).hexdigest(); assert abs(float(C[0] @ C[1]) + 1 / (K - 1)) < 1e-5
    g = torch.Generator().manual_seed(2026); R = torch.randn(cfg["n_random"], cfg["state_dim"], generator=g); R = R / R.norm(dim=1, keepdim=True)     # fixed off-code unit vectors
    prompt = lambda name: M.ids(cfg["filler"] + cfg["valid_line"] + cfg["query"].format(NAME=name.upper()))
    res = {"config": cfg["name"], "sha256": shas, "codebook_hash": cb_hash, "revision": M.sp.revision, "seeds": {}}; save = lambda: json.dump(res, open(os.path.join(out, f"{tag}_result.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else float(o))
    log(f"loaded; codebook {cb_hash[:12]}; slot_pos={M.slot_pos}; {len(names)} entities x {K} codes ({time.time()-T0:.0f}s)")
    deadline = T0 + cfg["train"]["hard_wall_minutes"] * 60
    for seed in cfg["train"]["seeds"]:
        torch.manual_seed(seed); rng = np.random.default_rng(seed); J = nn.Linear(cfg["state_dim"], M.m.config.hidden_size, bias=False); nn.init.zeros_(J.weight)
        opt = torch.optim.AdamW(J.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]); t0 = time.time(); hist = []; tele = []
        order = rng.permutation(len(names) * K)                                                   # balanced entity x code schedule
        for step in range(cfg["train"]["steps"]):
            ei, k = divmod(int(order[step % len(order)]), K); loss = M.label_loss(prompt(names[ei]), " " + T[k], J(C[k]))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(J.parameters(), cfg["train"]["clip"]); opt.step(); hist.append(float(loss.detach())); tele.append({"step": step, "code": k, **M.telemetry})
            if step % 50 == 0 or step == cfg["train"]["steps"] - 1: log(f"seed {seed} step {step}: loss={float(loss):.3f} pre_cap={M.telemetry['pre_cap_norm']:.2f} thr={M.telemetry['threshold']:.2f} cap_active={M.telemetry['cap_active']} ({time.time()-t0:.0f}s)")
        J.eval(); torch.save(J.state_dict(), os.path.join(out, f"J_seed{seed}.pt"))
        def evaluate(uncapped):
            M.uncapped = uncapped; rows = []
            with torch.no_grad():
                for ei, name in enumerate(names):
                    ids = prompt(name); row = {"entity": name, "arms": {}}
                    def arm(label, delta):
                        txt = M.decode(ids, delta, TAGS); c, ok = strict(txt, TAGS); row["arms"][label] = {"choice": c, "completed": ok, "tele": dict(M.telemetry) if delta is not None else None}
                    for k in range(K): arm(f"code{k}", J(C[k]))                                     # own-code arms: every code injected for every entity
                    arm("cue", None); arm("zero", torch.zeros(M.m.config.hidden_size))
                    for r in range(cfg["n_random"]): arm(f"rand{r}", J(R[r]))
                    rows.append(row)
                    if ei % 6 == 0: log(f"  seed {seed} {'UNCAPPED ' if uncapped else ''}{name}: " + " ".join(f"c{k}={row['arms'][f'code{k}']['choice']}" for k in range(K)) + f" cue={row['arms']['cue']['choice']} ({time.time()-T0:.0f}s)")
            M.uncapped = False
            follow = np.array([[r["arms"][f"code{k}"]["choice"] == TAGS[k] for k in range(K)] for r in rows], float)          # entity x code: injected code's tag followed
            comp = lambda keys: float(np.mean([r["arms"][x]["completed"] for r in rows for x in keys]))
            boot = lambda v: float(np.quantile([np.mean(rng.choice(v, len(v))) for _ in range(cfg["eval"]["bootstraps"])], 0.025))
            own = follow.mean(1); own_tag = [T.index(T[ei % K]) for ei in range(len(names))]                                     # 'own' code = entity's assigned code (balanced by index)
            own_acc = np.array([follow[ei, own_tag[ei]] for ei in range(len(names))]); wrong = np.array([np.mean([follow[ei, k] for k in range(K) if k != own_tag[ei]]) for ei in range(len(names))])
            cue_match = np.array([np.mean([r["arms"]["cue"]["choice"] == TAGS[k] for k in range(K) if k != own_tag[ei]]) for ei, r in enumerate(rows)]); cue_true = np.array([rows[ei]["arms"]["cue"]["choice"] == TAGS[own_tag[ei]] for ei in range(len(names))], float)
            rnd_true = np.array([np.mean([rows[ei]["arms"][f"rand{r}"]["choice"] == TAGS[own_tag[ei]] for r in range(cfg["n_random"])]) for ei in range(len(names))]); zero_ok = all(r["arms"]["zero"]["choice"] == r["arms"]["cue"]["choice"] for r in rows)
            s = {"code_follow": float(follow.mean()), "code_follow_lb": boot(own), "per_code": follow.mean(0).tolist(), "own_acc": float(own_acc.mean()), "wrong_follow": float(wrong.mean()), "wrong_uplift": float((wrong - cue_match).mean()), "wrong_uplift_lb": boot(wrong - cue_match),
                 "cue_true": float(cue_true.mean()), "random_true": float(rnd_true.mean()), "own_minus_random": float((own_acc - rnd_true).mean()), "own_minus_random_lb": boot(own_acc - rnd_true), "completion": {"code": comp([f"code{k}" for k in range(K)]), "random": comp([f"rand{r}" for r in range(cfg["n_random"])]), "cue": comp(["cue"])}, "zero_hook_matches_cue": zero_ok,
                 "cap_active_rate": float(np.mean([r["arms"][f"code{k}"]["tele"]["cap_active"] for r in rows for k in range(K)]))}
            s["pass"] = (zero_ok and s["completion"]["code"] >= G["completion_min"] and s["completion"]["random"] >= G["completion_min"] and s["code_follow"] >= G["code_follow_min"] and s["code_follow_lb"] > G["code_follow_lb_min"] and min(s["per_code"]) >= G["per_code_min"] and s["own_acc"] >= G["own_min"]
                         and s["wrong_follow"] >= G["wrong_follow_min"] and s["wrong_uplift"] >= G["wrong_uplift_min"] and s["wrong_uplift_lb"] > G["wrong_uplift_lb_min"] and s["cue_true"] <= G["cue_max"] and s["random_true"] <= G["random_max"] and s["own_minus_random"] >= G["own_minus_random_min"] and s["own_minus_random_lb"] > G["own_minus_random_lb_min"])
            return s, rows
        capped, rows_c = evaluate(False); unc, rows_u = evaluate(True)
        res["seeds"][seed] = {"loss_history": hist, "telemetry": tele, "capped": capped, "uncapped": unc, "rows_capped": rows_c, "rows_uncapped": rows_u}; save()
        log(f"seed {seed} capped: {json.dumps({k: v for k, v in capped.items()}, default=float)}"); log(f"seed {seed} uncapped: {json.dumps({k: v for k, v in unc.items()}, default=float)} ({time.time()-T0:.0f}s)")
        if time.time() > deadline: log("hard wall: stopping seeds"); break
    cp = sum(v["capped"]["pass"] for v in res["seeds"].values()); up = sum(v["uncapped"]["pass"] for v in res["seeds"].values()); n = G["seeds_required"]
    status = "BOUNDED ACTUATOR PASS — ORACLE CODE" if cp >= n else ("CAP-LIMITED ACTUATOR" if up >= n else "FAIL — ORACLE ACTUATOR/SITE/RETRIEVAL CONSTRUCTION") if len(res["seeds"]) >= n else "INCOMPLETE — NO VERDICT"
    res["status"] = status; res["seconds"] = time.time() - T0; save(); log(f"STATUS: {status}; capped passes {cp}, uncapped passes {up}")


if __name__ == "__main__":
    main()
