"""reachability_v1: nulled reachable-dimension characterization of the closed block-12 slot (Codex round 22). Measurement only.
G_{n,s} = 0.25 ||h_{n,s}|| P (dl_n / dh_{n,s}) for the eight first-token tag logits l_n at the query; shared Jacobian = mean over
names; spectrum, top-mode energy, participation effective rank, name bootstraps; nulls = hash-selected pre-answer positions
and permuted-VALID-TAGS prompt controls; finite-dose validation along the top three shared right-singular directions.

    python experiments/run_reachability.py --config experiments/config/reachability_v1.json
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from run_onewrite_state import LM, strict


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args(); T0 = time.time()
    cfg = json.load(open(a.config, encoding="utf-8")); out = f"experiments/results/{cfg['name']}"; os.makedirs(out, exist_ok=True); logf = open(os.path.join(out, "run.log"), "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    shas = {k: hashlib.sha256(open(v, "rb").read()).hexdigest() for k, v in (("runner", __file__), ("config", a.config), ("machinery", os.path.join(os.path.dirname(__file__), "run_onewrite_state.py")))}
    M = LM(cfg); T = cfg["tags"]; TAGS = [t.upper() for t in T]; K = len(T); names = cfg["names_train"]; D = M.m.config.hidden_size; first = torch.tensor([M.ids(" " + t)[0] for t in T]); P = torch.eye(K) - torch.ones(K, K) / K
    base_order = list(range(K)); prng = np.random.default_rng(cfg["prompt_controls"]["seed"]); perms = [list(prng.permutation(K)) for _ in range(cfg["prompt_controls"]["count"])]
    def prompt(name, order):
        vl = cfg["valid_line_template"].format(tags=" | ".join(T[i] for i in order)); return M.ids(cfg["filler"]) + M.ids(vl) + M.ids(cfg["query"].format(NAME=name.upper())), len(M.ids(cfg["filler"])) + len(M.ids(vl))
    # admissible null offsets (distance before the answer) must land, for EVERY name, on a valid-line token that is not a tag token
    label_toks = {tok for t in T for tok in M.ids(" " + t)} | {tok for t in T for tok in M.ids(t)}
    admissible = None
    for name in names:
        ids, qstart = prompt(name, base_order); L = len(ids); ok = {L - 1 - p for p in range(len(M.ids(cfg["filler"])), qstart) if ids[p] not in label_toks and p != M.slot_pos}
        admissible = ok if admissible is None else admissible & ok
    nr = cfg["null_positions"]; cand = sorted(d for d in admissible if nr["min_before_answer"] <= d <= nr["max_before_answer"]); rng = np.random.default_rng(nr["seed"]); null_offsets = sorted(rng.choice(cand, nr["count"], replace=False).tolist())
    res = {"config": cfg["name"], "sha256": shas, "revision": M.sp.revision, "null_offsets_before_answer": null_offsets, "prompt_control_orders": perms, "admissible_offsets": cand}; save = lambda: json.dump(res, open(os.path.join(out, "run_result.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else float(o))
    log(f"loaded; slot_pos={M.slot_pos}; null offsets {null_offsets} of {len(cand)} admissible; prompt controls {perms} ({time.time()-T0:.0f}s)")
    def jac(ids, pos):
        """Normalized-budget centred Jacobian G (K x D) of the eight first-token logits w.r.t. the residual at `pos`, plus ||h_pos||."""
        M.slot_pos = pos; d0 = torch.zeros(D, requires_grad=True); lg = M._forward(ids, d0, False).logits[0, -1].float()[first]
        rows = [torch.autograd.grad(lg[k], d0, retain_graph=k < K - 1)[0].detach() for k in range(K)]; h = M.telemetry["slot_norm"]; return (cfg["norm_clamp_fraction"] * h * (P @ torch.stack(rows))), h
    sites = {"slot": (base_order, "slot")} | {f"null{i}": (base_order, off) for i, off in enumerate(null_offsets)} | {f"prompt{i}": (perms[i], "slot") for i in range(len(perms))}
    Gs = {}; Hn = {}
    for sname, (order, where) in sites.items():
        Gs[sname] = []; Hn[sname] = []
        for name in names:
            ids, _ = prompt(name, order); pos = (len(M.ids(cfg["slot"])) - 1) if where == "slot" else len(ids) - 1 - where; G, h = jac(ids, pos); Gs[sname].append(G); Hn[sname].append(h)
        Gs[sname] = torch.stack(Gs[sname]); log(f"site {sname}: Jacobians done ({time.time()-T0:.0f}s)")
    def stats(Gstack):
        Gbar = Gstack.mean(0); s = torch.linalg.svdvals(Gbar); e = s ** 2; return {"sv": s.tolist(), "top_energy": float(e[0] / e.sum()), "eff_rank": float(e.sum() ** 2 / (e ** 2).sum()), "s2_over_s1": float(s[1] / s[0]), "s3_over_s1": float(s[2] / s[0])}
    brng = np.random.default_rng(7); B = cfg["eval"]["bootstraps"]
    def boot(Gstack, key):
        vals = [stats(Gstack[torch.tensor(brng.integers(len(Gstack), size=len(Gstack)))])[key] for _ in range(B)]; return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
    summ = {}
    for sname in sites:
        st = stats(Gs[sname]); st["top_energy_ci"] = boot(Gs[sname], "top_energy"); st["eff_rank_ci"] = boot(Gs[sname], "eff_rank"); st["s3_over_s1_ci"] = boot(Gs[sname], "s3_over_s1")
        _, _, Vt = torch.linalg.svd(Gs[sname].mean(0), full_matrices=False); v1 = Vt[0]; st["alignment_across_names"] = float(np.mean([abs(float(torch.linalg.svd(G, full_matrices=False)[2][0] @ v1)) for G in Gs[sname]]))
        summ[sname] = st; log(f"  {sname}: top_energy={st['top_energy']:.3f} {st['top_energy_ci']} eff_rank={st['eff_rank']:.2f} {st['eff_rank_ci']} s2/s1={st['s2_over_s1']:.2f} s3/s1={st['s3_over_s1']:.2f} align={st['alignment_across_names']:.2f}")
    res["spectra"] = summ; save()
    # finite-dose validation at the slot and at the first position null: +/- top-3 shared right-singular directions at exactly 0.25 ||h||
    fd = {}
    with torch.no_grad():
        for sname in ("slot", "null0"):
            order, where = sites[sname]; U, S, Vt = torch.linalg.svd(Gs[sname].mean(0), full_matrices=False); rows = []
            for m in range(cfg["finite_dose_modes"]):
                for sign in (1, -1):
                    for ni, name in enumerate(names):
                        ids, _ = prompt(name, order); pos = (len(M.ids(cfg["slot"])) - 1) if where == "slot" else len(ids) - 1 - where; M.slot_pos = pos; h = Hn[sname][ni]
                        l0 = M._forward(ids, None, False).logits[0, -1].float()[first]; l1 = M._forward(ids, sign * cfg["norm_clamp_fraction"] * h * Vt[m], False).logits[0, -1].float()[first]
                        real = P @ (l1 - l0); pred = sign * (Gs[sname][ni] @ Vt[m]); txt = M.decode(ids, sign * cfg["norm_clamp_fraction"] * h * Vt[m], TAGS); c, ok = strict(txt, TAGS)
                        rows.append({"mode": m, "sign": sign, "name": name, "cos": float(torch.nn.functional.cosine_similarity(real, pred, dim=0)), "norm_ratio": float(real.norm() / (pred.norm() + 1e-9)), "real_norm": float(real.norm()), "choice": c})
            by = lambda m: [r for r in rows if r["mode"] == m]; n1 = np.median([r["real_norm"] for r in by(0)])
            fd[sname] = {"cos_median": [float(np.median([r["cos"] for r in by(m)])) for m in range(cfg["finite_dose_modes"])], "norm_ratio_median": [float(np.median([r["norm_ratio"] for r in by(m)])) for m in range(cfg["finite_dose_modes"])],
                         "rel_norm_to_mode1": [float(np.median([r["real_norm"] for r in by(m)]) / (n1 + 1e-9)) for m in range(cfg["finite_dose_modes"])], "choices": {f"m{m}{'+' if s > 0 else '-'}": dict(zip(*np.unique([str(r["choice"]) for r in rows if r["mode"] == m and r["sign"] == s], return_counts=True))) for m in range(cfg["finite_dose_modes"]) for s in (1, -1)}, "rows": rows}
            log(f"  finite dose {sname}: cos {fd[sname]['cos_median']} norm_ratio {fd[sname]['norm_ratio_median']} rel_norm {fd[sname]['rel_norm_to_mode1']}"); res["finite_dose"] = fd; save()
    M.slot_pos = len(M.ids(cfg["slot"])) - 1
    # classification (measurement, not pass/fail)
    C = cfg["classes"]; s = summ["slot"]; nulls = [summ[k]["top_energy"] for k in summ if k.startswith("null")]; prompts = [summ[k]["top_energy"] for k in summ if k.startswith("prompt")]; f = fd["slot"]
    narrow = s["top_energy_ci"][0] >= C["narrow"]["top_energy_lb_min"] and s["eff_rank_ci"][1] <= C["narrow"]["eff_rank_ub_max"] and s["top_energy"] > np.quantile(nulls, 0.95) and s["top_energy"] > np.quantile(prompts, 0.95) and f["cos_median"][0] >= C["narrow"]["mode1_cos_median_min"] and max(f["rel_norm_to_mode1"][1:]) <= C["narrow"]["modes23_rel_norm_max"]
    multi = s["eff_rank_ci"][0] >= C["multi"]["eff_rank_lb_min"] and s["s3_over_s1_ci"][0] >= C["multi"]["s3_over_s1_lb_min"] and min(f["cos_median"]) >= C["multi"]["modes_cos_median_min"] and min(f["rel_norm_to_mode1"][1:]) >= C["multi"]["modes23_rel_norm_min"]
    res["classification"] = "NARROW SHARED REACHABILITY" if narrow else ("MULTIDIRECTIONAL LOCAL REACHABILITY" if multi else "NO SLOT-SPECIFIC GEOMETRY CONCLUSION"); res["seconds"] = time.time() - T0; save(); log(f"CLASSIFICATION: {res['classification']} ({time.time()-T0:.0f}s)")


if __name__ == "__main__":
    main()
