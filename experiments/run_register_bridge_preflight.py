"""register_bridge_preflight_v1 (Codex rounds 27-28; repaired per audit #38): NONCAUSAL feasibility measurement, not staircase advancement.
Is the explicit-legend episode state (legend + registry record, zero delay) linearly decodable (rank <= 8 ridge, dual form) from frozen
Qwen3-1.7B-Base residuals at the record's tag span, transferring to held-out entities, templates and permutations, above input-embedding,
categorical and PAIRED context-destroyed controls and a balanced entity-clustered label-shuffle null? Every entity is crossed with every
state. A PASS = linear state decodability for an explicit legend lookup; it is not a code-level or causal bridge.

    python experiments/run_register_bridge_preflight.py --config experiments/config/register_bridge_preflight_v1.json [--smoke]
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from substitution_probe import SubstitutionProbe


def banks(K, rng):
    """Two disjoint balanced banks of K permutations (Latin squares from two bases): every tag occurs once per state within a bank."""
    while True:
        b0, b1 = rng.permutation(K), rng.permutation(K); B0 = [tuple(b0[(np.arange(K) + j) % K]) for j in range(K)]; B1 = [tuple(b1[(np.arange(K) + j) % K]) for j in range(K)]
        if not set(B0) & set(B1): return [np.array(p) for p in B0], [np.array(p) for p in B1]


class Ridge:
    """Standardized ridge scores in dual form (n x n solve): W = Xs^T (Xs Xs^T + lam n I)^-1 Yc; fit once, apply to any rows."""
    def __init__(self, Xtr, ytr, lam, K):
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0) + 1e-6; self.Xs = (Xtr - self.mu) / self.sd; Y = np.eye(K)[ytr]; n = len(self.Xs); self.ym = Y.mean(0)
        self.alpha = np.linalg.solve(self.Xs @ self.Xs.T + lam * n * np.eye(n), Y - self.ym)
    def __call__(self, Xte): return ((((Xte - self.mu) / self.sd) @ self.Xs.T) @ self.alpha + self.ym).argmax(1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args(); T0 = time.time()
    P = json.load(open(a.config, encoding="utf-8")); src = json.load(open(P["entities_from"], encoding="utf-8")); names = src["names_train"]; K = len(P["tags"]); tags = P["tags"]; TPL = P["templates"]; E = len(names)
    out = f"experiments/results/{P['name']}"; os.makedirs(out, exist_ok=True); tag = "smoke" if a.smoke else "run"; logf = open(os.path.join(out, f"{tag}.log"), "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    shas = {k: hashlib.sha256(open(v, "rb").read()).hexdigest() for k, v in (("runner", __file__), ("config", a.config), ("entities", P["entities_from"]))}
    rng = np.random.default_rng(P["seed"]); tr_bank, ho_bank = banks(K, rng); layers = P["representation"]["layers_zero_based"]; NS, NB, PPC = P["shuffles"], P["bootstraps"], P["permutation_banks"]["perms_per_cell"]
    if a.smoke: NS, NB, PPC, layers, E = 20, 200, 1, layers[:2], 6                                                                       # smoke: six entities (two per fold group), one permutation per cell, two layers
    dev = P["device"]; sp = SubstitutionProbe(P["model"], dtype=getattr(torch, P["dtype"]), revision=src["revision"]); m, tok = sp.model.to(dev), sp.tok; assert sp.revision == src["revision"]
    ids = lambda t: tok.encode(t, add_special_tokens=False); emb = m.model.embed_tokens.weight; deadline = T0 + P["hard_wall_minutes"] * 60
    res = {"config": P["name"], "sha256": shas, "revision": sp.revision, "device": dev, "dtype": P["dtype"], "train_bank": [p.tolist() for p in tr_bank], "heldout_bank": [p.tolist() for p in ho_bank], "gates": P["gates"], "perms_per_cell": PPC, "layers": layers}
    save = lambda: json.dump(res, open(os.path.join(out, f"{tag}_result.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else float(o))
    def stop(status):
        res["status"] = status; res["seconds"] = time.time() - T0; save(); log(f"STATUS: {status} ({time.time()-T0:.0f}s)")
    def prompt(e, s, t, pi, order, sigma):
        """Legend with clause order `order`; record tag = pi[s]. sigma=None: intact; else legend pairing k -> pi[sigma[k]] (paired arm: same order, permutation, record, token multiset)."""
        leg = pi if sigma is None else pi[sigma]; clause = lambda k: f"state {k} = {tags[leg[k]]}"; legend = "Legend: " + "; ".join(clause(k) for k in order) + ". "; prefix = legend + TPL[t].format(entity=names[e]); word = tags[pi[s]]; full = prefix + " " + word + "."
        a_ids, b_ids, f_ids = ids(prefix), ids(prefix + " " + word), ids(full)
        if f_ids[: len(b_ids)] != b_ids or b_ids[: len(a_ids)] != a_ids or tok.decode(b_ids[len(a_ids):]).strip() != word or full.rfind(word) != len(prefix) + 1: return None
        k_word = int(np.flatnonzero(leg == pi[s])[0]); pos = list(order).index(k_word); lp = ids("Legend: " + "; ".join(clause(k) for k in order[:pos]) + ("; " if pos else "") + f"state {k_word} =")
        return full, f_ids, (len(a_ids), len(b_ids)), (len(lp), len(lp) + len(b_ids) - len(a_ids)), k_word                            # record span; the same tag's legend-occurrence span; the state the record tag denotes under this legend
    rows, feats = [], {l: [] for l in layers} | {"emb": [], "legend": []}
    with torch.no_grad():
        for e, s, t in itertools.product(range(E), range(K), range(len(TPL))):
            bank, bname = (tr_bank, "train") if t in P["templates_train"] else (ho_bank, "heldout"); order = rng.permutation(K); sigma = rng.permutation(K)
            while (sigma == np.arange(K)).any(): sigma = rng.permutation(K)                                                           # derangement: every state's tag changes under the destroyed legend
            for j in range(PPC):
                pidx = (e + j * (K // 2)) % K; pi = bank[pidx]                                                                           # independent of state and template (round 28)
                for destroyed in (False, True):
                    pr = prompt(e, s, t, pi, order, sigma if destroyed else None)
                    if pr is None or tok.decode(pr[1][pr[3][0]: pr[3][1]]).strip() != tags[pi[s]] or (pr[4] != s) != destroyed: res["failed_at"] = [e, s, t, int(pidx), destroyed]; return stop("INVALID — TOKENIZATION/SPLIT MANIFEST")
                    full, toks, (s0, s1), (l0, l1), denoted = pr; hs = m(input_ids=torch.tensor([toks], device=dev), output_hidden_states=True).hidden_states
                    for l in layers: feats[l].append(hs[l + 1][0, s0:s1].float().mean(0).cpu().numpy())
                    feats["emb"].append(emb[torch.tensor(toks[s0:s1], device=dev)].float().mean(0).cpu().numpy()); feats["legend"].append(hs[layers[len(layers) // 2] + 1][0, l0:l1].float().mean(0).cpu().numpy())
                    rows.append({"i": len(rows), "e": e, "s": s, "t": t, "bank": bname, "pidx": int(pidx), "destroyed": destroyed, "tag": int(pi[s]), "denoted_state": denoted, "order": order.tolist(), "sigma": sigma.tolist() if destroyed else None, "span": [s0, s1], "legend_span": [l0, l1], "len": len(toks), "prompt": full, "ids": toks, "span_text": tok.decode(toks[s0:s1])})
            if time.time() > deadline: res["n_rows"] = len(rows); return stop("INCOMPLETE — DEADLINE")
            if s == 0 and t == 0: log(f"entity {e}: rows {len(rows)} ({time.time()-T0:.0f}s)")
    fold = lambda e: e // (E // 3); groups = sorted({fold(e) for e in range(E)})
    for f, s, t, d in itertools.product(groups, range(K), range(len(TPL)), (False, True)):                                              # tag balance within every fold x state x template (round 28)
        c = np.bincount([r["tag"] for r in rows if fold(r["e"]) == f and r["s"] == s and r["t"] == t and r["destroyed"] == d], minlength=K)
        if c.min() != c.max() and not a.smoke: res["imbalance_at"] = [f, s, t, d, c.tolist()]; return stop("INVALID — TOKENIZATION/SPLIT MANIFEST")
    F = {k: np.stack(v).astype(getattr(np, P["feature_dtype"])) for k, v in feats.items()}; np.savez_compressed(os.path.join(out, f"{tag}_features.npz"), **{str(k): v for k, v in F.items()}); log(f"features saved; {len(rows)} rows ({time.time()-T0:.0f}s)")
    y_true = np.array([r["s"] for r in rows]); intact = [r["i"] for r in rows if not r["destroyed"]]; destroyed_rows = [r["i"] for r in rows if r["destroyed"]]
    def X(kind, idx): return F[kind][idx] if kind != "cat" else np.stack([np.concatenate([np.eye(K)[rows[i]["tag"]], np.eye(len(TPL))[rows[i]["t"]], [rows[i]["span"][0] / 100, rows[i]["len"] / 100]]) for i in idx])
    def evaluate(kind_grid, y=None, fixed=None):
        """Outer entity folds (each holds out E/3 entities, all states); inner entity folds select (kind, lambda) unless fixed; decoders fit on intact training rows only and applied to intact AND paired-destroyed held-out rows."""
        y = y_true if y is None else y; pred, dpred, picks = {}, {}, []
        for f in groups:
            if time.time() > deadline: return None, None, None
            tr = [i for i in intact if fold(rows[i]["e"]) != f and rows[i]["bank"] == "train"]; te = [i for i in intact if fold(rows[i]["e"]) == f and rows[i]["bank"] == "heldout"]; dte = [i for i in destroyed_rows if fold(rows[i]["e"]) == f and rows[i]["bank"] == "heldout"]; best = fixed[f] if fixed else None
            if best is None:
                for kind, lam in itertools.product(kind_grid, P["decoder"]["ridge_grid"]):
                    acc = []
                    for g in [x for x in groups if x != f]:
                        itr = [i for i in tr if fold(rows[i]["e"]) != g]; ite = [i for i in tr if fold(rows[i]["e"]) == g]; acc.append(np.mean(Ridge(X(kind, itr), y[itr], lam, K)(X(kind, ite)) == y[ite]))
                    if best is None or np.mean(acc) > best[0]: best = (float(np.mean(acc)), kind, lam)
            dec = Ridge(X(best[1], tr), y[tr], best[2], K); pred |= dict(zip(te, map(int, dec(X(best[1], te))))); dpred |= dict(zip(dte, map(int, dec(X(best[1], dte))))); picks.append({"fold": f, "inner_acc": best[0], "kind": best[1], "lambda": best[2]})
        return pred, dpred, picks
    acc_of = lambda pred, key="s": float(np.mean([p == rows[i][key] for i, p in pred.items()])); by_e = lambda pred: np.array([np.mean([p == y_true[i] for i, p in pred.items() if rows[i]["e"] == e]) for e in range(E)])
    boot = lambda v: float(np.quantile([np.mean(rng.choice(v, len(v))) for _ in range(NB)], 0.025))
    main_pred, destroyed_pred, picks = evaluate(layers)
    if main_pred is None: return stop("INCOMPLETE — DEADLINE")
    acc = acc_of(main_pred); acc_e = by_e(main_pred); acc_lb = boot(acc_e); fixed = {p["fold"]: (p["inner_acc"], p["kind"], p["lambda"]) for p in picks}
    per_fold = [float(np.mean([p == y_true[i] for i, p in main_pred.items() if fold(rows[i]["e"]) == f])) for f in groups]; recall = [float(np.mean([p == y_true[i] for i, p in main_pred.items() if y_true[i] == k])) for k in range(K)]
    ctrl_pred = {"input_embedding": evaluate(["emb"])[0], "categorical": evaluate(["cat"])[0], "context_destroyed_paired": destroyed_pred}; legend_pred = evaluate(["legend"])[0]
    if any(v is None for v in ctrl_pred.values()) or legend_pred is None: return stop("INCOMPLETE — DEADLINE")
    ctrl_e = {k: by_e(v) for k, v in ctrl_pred.items()}; ctrl_max_e = np.max(np.stack(list(ctrl_e.values())), 0); adv = float((acc_e - ctrl_max_e).mean()); adv_lb = boot(acc_e - ctrl_max_e)
    log(f"main acc {acc:.3f} LB {acc_lb:.3f} folds {per_fold} recall {recall}; controls " + json.dumps({k: float(v.mean()) for k, v in ctrl_e.items()}) + f"; destroyed follows denoted {acc_of(destroyed_pred, 'denoted_state'):.3f}; legend reference {acc_of(legend_pred):.3f}; picks {picks} ({time.time()-T0:.0f}s)")
    null = []
    for _ in range(NS):                                                                                                               # balanced entity-clustered label-shuffle null: one state permutation per training entity applied to all its rows; evaluation truth retained
        perm = {e: rng.permutation(K) for e in range(E)}; ys = np.array([perm[rows[i]["e"]][rows[i]["s"]] for i in range(len(rows))]); pn = evaluate(layers, ys, fixed)[0]
        if pn is None: return stop("INCOMPLETE — DEADLINE")
        null.append(acc_of(pn))
    p99 = float(np.quantile(null, 0.99)); G = P["gates"]
    primary = acc >= G["acc_min"] and min(per_fold) >= G["fold_min"] and min(recall) >= G["state_recall_min"] and acc_lb > G["entity_lb_min"] and acc >= p99 + G["shuffle_null_p99_margin"]
    status = ("PREFLIGHT PASS — EXPLICIT-LEGEND STATE LINEARLY DECODABLE" if adv >= G["control_advantage_min"] and adv_lb > G["control_advantage_lb_min"] else "PREFLIGHT PARTIAL — TOKEN/CONTEXT-BOUND DECODABILITY") if primary else "PREFLIGHT FAIL — NO QUALIFYING EXPLICIT-LEGEND STATE DECODER"
    res |= {"accuracy": acc, "entity_lb": acc_lb, "per_fold": per_fold, "state_recall": recall, "entity_acc": acc_e.tolist(), "controls": {k: float(v.mean()) for k, v in ctrl_e.items()}, "control_entity_acc": {k: v.tolist() for k, v in ctrl_e.items()}, "control_advantage": adv, "control_advantage_lb": adv_lb,
            "destroyed_follows_denoted_state": acc_of(destroyed_pred, "denoted_state"), "legend_occurrence_reference": acc_of(legend_pred), "shuffle_null": {"n": NS, "scores": null, "mean": float(np.mean(null)), "p99": p99}, "picks": picks, "n_rows": len(rows)}
    json.dump([r | {"pred_main": main_pred.get(r["i"]), "pred_legend_reference": legend_pred.get(r["i"])} | {f"pred_{k}": v.get(r["i"]) for k, v in ctrl_pred.items()} for r in rows], open(os.path.join(out, f"{tag}_rows.json"), "w")); stop(status)


if __name__ == "__main__":
    main()
