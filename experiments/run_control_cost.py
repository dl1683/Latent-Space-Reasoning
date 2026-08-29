"""control_cost_v1: minimum-energy span control in a frozen LM (Qwen3-1.7B-Base, CPU). Locked design: .codex_direction_r9.md.

A one-time field  h_12[i] += sigma * v / sqrt(|S|)  is broadcast over the prefix span S (all tokens before the anchor).
Cost = ||v||. The first-order minimum-energy field toward a target three-probe signature is v* = J^T (J J^T)^+ r; the
nonlinear model is evaluated only on a fixed alpha grid. Readouts: optimized signature A, unoptimized signature B.
Controls: within-class targets, calibration-derived shared fields, a direct lexical-gradient field, random fields.

    python experiments/run_control_cost.py --config experiments/config/control_cost_v1.json
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from substitution_probe import SubstitutionProbe
from scipy.stats import spearmanr


class M:
    def __init__(self, cfg):
        self.cfg = cfg; self.sp = SubstitutionProbe(cfg["model_id"], revision=cfg["revision"]); self.m = self.sp.model; self.tok = self.sp.tok
        assert self.sp.revision == cfg["revision"]; torch.set_grad_enabled(True)
        for p in self.m.parameters(): p.requires_grad_(False)
        self.layer = self.m.model.layers[cfg["layer"]]; self.field = None; self.forwards = 0; self.layer.register_forward_hook(self._hook)
        self.A = len(self.tok.encode(cfg["anchor"], add_special_tokens=False)); self.ids = lambda t: self.tok.encode(t, add_special_tokens=False)

    def _hook(self, mod, i, o):
        if self.field is None: return o
        h = o[0] if isinstance(o, tuple) else o; h = h.clone(); n = self.P - self.A
        h[:, :n, :] = h[:, :n, :] + self.field.to(h.dtype); return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h

    def ll(self, ctx, cont):
        c = self.ids(cont); out = self.m(input_ids=torch.tensor([ctx + c])); self.forwards += 1
        lp = torch.log_softmax(out.logits[0, len(ctx) - 1:-1].float(), -1); return lp.gather(1, torch.tensor(c).unsqueeze(1)).mean()

    def sig(self, ctx, probes, v=None):
        """Three signed probe margins (mean LL a - mean LL b); differentiable in v if v requires grad."""
        self.field = None if v is None else (self.sigma * v / np.sqrt(self.P - self.A))
        try: return torch.stack([self.ll(ctx, a) - self.ll(ctx, b) for a, b in probes])
        finally: self.field = None

    def lex(self, ctx, probes, v):
        """First-divergent-token logit differences (a - b) for each probe, differentiable in v."""
        self.field = self.sigma * v / np.sqrt(self.P - self.A); out = []
        try:
            for a, b in probes:
                ta, tb = self.ids(a), self.ids(b); k = next(i for i in range(min(len(ta), len(tb))) if ta[i] != tb[i])
                lg = self.m(input_ids=torch.tensor([ctx + ta[:k]])).logits[0, -1].float(); self.forwards += 1; out.append(lg[ta[k]] - lg[tb[k]])
        finally: self.field = None
        return torch.stack(out)

    def jac(self, fn, ctx, probes):
        v = torch.zeros(self.m.config.hidden_size, requires_grad=True); u = fn(ctx, probes, v)
        J = torch.stack([torch.autograd.grad(u[k], v, retain_graph=k < len(u) - 1)[0] for k in range(len(u))]); return u.detach(), J.detach()


def solve(J, r, rcond):
    JJ = J @ J.T; return J.T @ (torch.linalg.pinv(JJ, rcond=rcond) @ r)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args(); t0 = time.time()
    cfg = json.load(open(a.config, encoding="utf-8")); src = json.load(open(cfg["contexts_from"], encoding="utf-8")); out = f"experiments/results/{cfg['name']}"; os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "run.log"), "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    shas = {k: hashlib.sha256(open(v, "rb").read()).hexdigest() for k, v in (("runner", __file__), ("config", a.config), ("contexts", cfg["contexts_from"]))}
    res = {"config": cfg["name"], "sha256": shas}; save = lambda: json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else float(o))
    deadline = t0 + cfg["hard_wall_minutes"] * 60; W = M(cfg); C = ["cat", "dog"]; sgn = {"cat": 1.0, "dog": -1.0}; G = cfg["gates"]; al = cfg["alphas"]; R = cfg["realize"]
    ctx = {k: {c: [W.ids(t + cfg["anchor"]) for t in src[k][c]] for c in C} for k in ("calibration", "test")}
    W.P = len(ctx["calibration"]["cat"][0]); assert all(len(x) == W.P for k in ctx for c in C for x in ctx[k][c]); log(f"loaded; P={W.P} anchor={W.A} span={W.P-W.A}")
    with torch.no_grad():                                                  # calibration-only per-channel sigma over the prefix span
        H = torch.cat([W.m(input_ids=torch.tensor([x]), output_hidden_states=True).hidden_states[cfg["layer"] + 1][0, :W.P - W.A] for c in C for x in ctx["calibration"][c]])
        sig = H.std(0); W.sigma = torch.maximum(sig, cfg["sigma_floor_fraction"] * sig.median())
        cal = {p: {c: torch.stack([W.sig(x, cfg[p]) for x in ctx["calibration"][c]]) for c in C} for p in ("probes_A", "probes_B")}
    cen = {}
    for p in cal:
        b = (cal[p]["cat"].mean(0) + cal[p]["dog"].mean(0)) / 2; s = torch.sqrt(((cal[p]["cat"] - cal[p]["cat"].mean(0)) ** 2).sum(0) + ((cal[p]["dog"] - cal[p]["dog"].mean(0)) ** 2).sum(0)) / np.sqrt(6)
        s = torch.maximum(s, torch.tensor(1e-3)); ucal = {c: (cal[p][c] - b) / s for c in C}; cen[p] = {"b": b, "s": s, "cent": {c: ucal[c].mean(0) for c in C}}
        cen[p]["delta"] = cen[p]["cent"]["cat"] - cen[p]["cent"]["dog"]; cen[p]["sep"] = float(cen[p]["delta"].norm())
    u = lambda p, m: (m - cen[p]["b"]) / cen[p]["s"]
    with torch.no_grad(): nat = {p: {c: [u(p, W.sig(x, cfg[p])) for x in ctx["test"][c]] for c in C} for p in cal}
    valid = {}
    for p in nat:
        cor = {c: int(sum((torch.sign(x) == sgn[c]).sum() for x in nat[p][c])) for c in C}; valid[p] = sum(cor.values()) >= G["native_total_min"] and all(v >= G["native_per_class_min"] for v in cor.values())
        log(f"native validity {p}: {cor} -> {'PASS' if valid[p] else 'FAIL'}"); res[f"native_{p}"] = {"correct": cor, "passed": valid[p]}
    save()
    if not valid["probes_A"]: res["status"] = "FAIL — FIXED BLOCK-12 SPAN CONTROL CONSTRUCTION (native A validity)"; save(); log(res["status"]); return
    move = lambda p, u1, u0, c: float(((u1 - u0) @ (-sgn[c] * cen[p]["delta"])) / cen[p]["sep"] ** 2)            # fraction of class separation toward the opposite class
    flips = lambda p, u1, c: int(((torch.sign(u1) == -sgn[c])).sum())
    def realize(x, v, c, p="probes_A", target=None, amax=None):
        """Smallest alpha on the grid attaining the criterion; returns (alpha, censored, per-alpha rows)."""
        rows = []; x = list(x); u0 = nat_lookup[tuple(x)]
        with torch.no_grad():
            for alpha in al:
                if amax is not None and alpha > amax: break
                ua = u0 if alpha == 0 else u(p, W.sig(x, cfg[p], alpha * v)); ub = None if alpha == 0 else u("probes_B", W.sig(x, cfg["probes_B"], alpha * v))
                ok = (torch.norm(ua - target) <= cfg["within_class_realize_fraction"] * torch.norm(u0 - target)) if target is not None else (move(p, ua, u0, c) >= R["movement_min"] and flips(p, ua, c) >= R["flips_min"])
                rows.append({"alpha": alpha, "move_A": move(p, ua, u0, c), "flips_A": flips(p, ua, c), "ok": bool(ok), "move_B": None if ub is None else move("probes_B", ub, natB_lookup[tuple(x)], c), "flips_B": None if ub is None else flips("probes_B", ub, c)})
                if ok and alpha > 0: return alpha, False, rows
        return (amax or al[-1]), True, rows
    nat_lookup = {}; natB_lookup = {}
    for c in C:
        for i, x in enumerate(ctx["test"][c]): nat_lookup[tuple(x)] = nat["probes_A"][c][i]; natB_lookup[tuple(x)] = nat["probes_B"][c][i]
    if os.environ.get("SMOKE"): ctx["test"] = {c: ctx["test"][c][:1] for c in C}; cfg["n_random"] = 2; log("SMOKE: 1 recipient per class, 2 random fields")
    # --- per-recipient minimum-energy fields, within-class controls ---
    rec = []
    for c in C:
        o = "dog" if c == "cat" else "cat"
        for i, x in enumerate(ctx["test"][c]):
            if time.time() > deadline: res["status"] = "INCOMPLETE — NO VERDICT"; save(); log(res["status"]); return
            u0, J = W.jac(lambda cc, pp, vv: u("probes_A", W.sig(cc, pp, vv)), x, cfg["probes_A"])
            vs = solve(J, cen["probes_A"]["cent"][o] - u0, cfg["rcond"]); pred = float(vs.norm())
            nat_lookup[tuple(x)] = u0; a_c, cens, rows = realize(x, vs, c); realized = a_c * pred
            tw = u("probes_A", cal["probes_A"][c][i])                                                                   # preassigned same-class calibration signature
            vw = solve(J, tw - u0, cfg["rcond"]); a_w, cens_w, rows_w = realize(x, vw, c, target=tw); within = a_w * float(vw.norm())
            rec.append({"class": c, "i": i, "pred": pred, "realized": realized, "censored": cens, "within": within, "within_censored": cens_w, "rows": rows, "rows_within": rows_w, "J": J})
            log(f"{c}{i}: pred={pred:.2f} realized={realized:.2f}{'(cens)' if cens else ''} within={within:.2f}{'(cens)' if cens_w else ''} ({time.time()-t0:.0f}s)")
    res["recipients"] = [{k: v for k, v in r.items() if k != "J"} for r in rec]; save()
    # --- shared calibration fields, lexical-gradient fields (per direction) ---
    shared = {}
    for c in C:
        o = "dog" if c == "cat" else "cat"; Js, rs, Jl, rl = [], [], [], []
        for x in ctx["calibration"][c]:
            u0, J = W.jac(lambda cc, pp, vv: u("probes_A", W.sig(cc, pp, vv)), x, cfg["probes_A"]); Js.append(J); rs.append(cen["probes_A"]["cent"][o] - u0)
            l0, JL = W.jac(W.lex, x, cfg["probes_A"]); Jl.append(JL); rl.append(l0)
        with torch.no_grad(): lt = torch.stack([W.lex(x, cfg["probes_A"], torch.zeros(W.m.config.hidden_size)) for x in ctx["calibration"][o]]).mean(0)    # opposite-class lexical centroid
        v_sem = solve(torch.cat(Js), torch.cat(rs), cfg["rcond"]); v_lex = solve(torch.cat(Jl), torch.cat([lt - l for l in rl]), cfg["rcond"])
        shared[c] = {"v_sem": v_sem, "v_lex": v_lex / v_lex.norm() * v_sem.norm(), "norm": float(v_sem.norm())}; log(f"shared field {c}->{o}: |v_sem|={shared[c]['norm']:.2f} |v_lex|={float(v_lex.norm()):.2f}")
    evals = []; rng = torch.Generator().manual_seed(2026)
    for c in C:
        med_spec = float(np.median([r["realized"] for r in rec if r["class"] == c]))
        for i, x in enumerate(ctx["test"][c]):
            if time.time() > deadline: res["status"] = "INCOMPLETE — NO VERDICT"; save(); log(res["status"]); return
            a_s, cens_s, rows_s = realize(x, shared[c]["v_sem"], c, amax=G["shared_alpha_max"]); row_s = next((r for r in rows_s if r["alpha"] == a_s), rows_s[-1])
            with torch.no_grad():
                B_lex = u("probes_B", W.sig(x, cfg["probes_B"], a_s * shared[c]["v_lex"])); mB_lex = move("probes_B", B_lex, natB_lookup[tuple(x)], c)
                A_lex = u("probes_A", W.sig(x, cfg["probes_A"], a_s * shared[c]["v_lex"]))
                rnd = []
                for k in range(cfg["n_random"]):
                    g = torch.randn(W.m.config.hidden_size, generator=rng); g = g / g.norm() * shared[c]["norm"]; rnd.append(move("probes_B", u("probes_B", W.sig(x, cfg["probes_B"], a_s * g)), natB_lookup[tuple(x)], c))
            evals.append({"class": c, "i": i, "alpha": a_s, "attained": not cens_s, "cost": a_s * shared[c]["norm"], "cost_ok": a_s * shared[c]["norm"] <= G["shared_cost_factor_max"] * med_spec, "move_B": row_s["move_B"], "flips_B": row_s["flips_B"], "move_A_lex": move("probes_A", A_lex, nat_lookup[tuple(x)], c), "move_B_lex": mB_lex, "random_move_B": rnd})
            log(f"shared {c}{i}: alpha={a_s} attained={not cens_s} move_B={row_s['move_B']} flips_B={row_s['flips_B']} lex_move_B={mB_lex:.2f} rand_max={max(rnd):.2f}")
    res["shared"] = evals; save()
    # --- gates ---
    byc = lambda L, c: [r for r in L if r["class"] == c]; n_real = sum(not r["censored"] for r in rec); per_dir_real = {c: sum(not r["censored"] for r in byc(rec, c)) for c in C}
    rho = float(spearmanr([r["pred"] for r in rec], [r["realized"] for r in rec]).correlation); ratio = float(np.median([r["realized"] / r["pred"] for r in rec]))
    cw = [r["realized"] > r["within"] for r in rec]; cw_ratio = float(np.median([r["realized"] / max(r["within"], 1e-9) for r in rec])); cw_dir = {c: sum(r["realized"] > r["within"] for r in byc(rec, c)) for c in C}
    sh_att = {c: sum(e["attained"] for e in byc(evals, c)) for c in C}; sh_cost = sum(e["cost_ok"] for e in evals); sh_B = {c: sum((e["move_B"] or 0) >= G["shared_B_move_min"] and (e["flips_B"] or 0) >= G["shared_B_flip_min"] for e in byc(evals, c)) for c in C}
    adv = np.array([(e["move_B"] or 0) - e["move_B_lex"] for e in evals]); adv_dir = {c: int(sum(((e["move_B"] or 0) - e["move_B_lex"]) > 0 for e in byc(evals, c))) for c in C}
    obs = adv.mean(); perm = float(np.mean([np.mean(adv * np.array(sg)) >= obs for sg in itertools.product([1, -1], repeat=len(adv))]))
    rand_p = float((1 + sum(np.mean([e["random_move_B"][k] for e in evals]) >= np.mean([(e["move_B"] or 0) for e in evals]) for k in range(cfg["n_random"]))) / (cfg["n_random"] + 1))
    asym = float(np.log(np.median([r["realized"] for r in byc(rec, "cat")]) / np.median([r["realized"] for r in byc(rec, "dog")])))
    gates = {"native_B": valid["probes_B"], "local": n_real >= G["local_realized_min"] and all(v >= G["local_per_direction_min"] for v in per_dir_real.values()) and rho >= G["spearman_min"] and G["ratio_range"][0] <= ratio <= G["ratio_range"][1],
             "cross_vs_within": sum(cw) >= G["cross_gt_within_min"] and cw_ratio >= G["cross_within_ratio_min"] and all(v >= G["per_direction_positive_min"] for v in cw_dir.values()),
             "shared_transfer": all(v >= G["shared_attain_per_direction_min"] for v in sh_att.values()) and sh_cost >= len(evals) / 2 and all(v >= G["shared_B_per_direction_min"] for v in sh_B.values()),
             "lexical_control": float(np.median(adv)) >= G["lex_B_advantage_median_min"] and int((adv > 0).sum()) >= G["lex_positive_min"] and all(v >= G["lex_per_direction_min"] for v in adv_dir.values()) and perm <= G["lex_perm_p_max"],
             "random_control": rand_p <= G["random_p_max"]}
    summary = {"n_realized": n_real, "per_direction_realized": per_dir_real, "spearman": rho, "median_realized_over_pred": ratio, "cross_gt_within": int(sum(cw)), "cross_within_ratio_median": cw_ratio, "cross_gt_within_by_dir": cw_dir,
               "shared_attained_by_dir": sh_att, "shared_cost_ok": sh_cost, "shared_B_by_dir": sh_B, "lex_advantage_median": float(np.median(adv)), "lex_advantage_positive": int((adv > 0).sum()), "lex_advantage_by_dir": adv_dir, "lex_perm_p": perm, "random_p": rand_p,
               "asymmetry_log_ratio": asym, "asymmetry_licensed": abs(asym) >= np.log(G["asymmetry_ratio_min"]), "forwards": W.forwards, "seconds": time.time() - t0}
    if not valid["probes_B"]:                                              # B readout invalid: B-based gates are VOID (spec), positive impossible
        for k in ("shared_transfer", "lexical_control", "random_control"): gates[k] = None
        status = ("PARTIAL — LOCAL SPAN CONTROLLABILITY, NO TRANSFER (B readout void)" if gates["local"] else "FAIL — FIXED BLOCK-12 SPAN CONTROL CONSTRUCTION")
    elif not gates["lexical_control"] and gates["local"]: status = "LEXICAL-GEOMETRY-COMPATIBLE CONTROL"
    elif all(gates.values()): status = "BOUNDED POSITIVE — TRANSFERABLE SPAN CONTROL COST"
    elif gates["local"]: status = "PARTIAL — LOCAL SPAN CONTROLLABILITY, NO TRANSFER"
    else: status = "FAIL — FIXED BLOCK-12 SPAN CONTROL CONSTRUCTION"
    res["summary"] = summary; res["gates"] = gates; res["status"] = status; save(); log(json.dumps(summary, indent=1, default=lambda o: o.item() if hasattr(o, "item") else float(o))); log(f"gates: {gates}"); log(f"STATUS: {status}")


if __name__ == "__main__":
    main()
