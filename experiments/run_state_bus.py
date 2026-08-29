"""state_bus: a co-developed persistent state interface on a frozen LM (Qwen3-1.7B-Base, CPU).

A 16-d bus at block L: z = E(LN(h_L(anchor))); J z is ADDED to the block-L output at every position after the anchor;
R reads z back from block L+8 at each decision boundary. Trained on two consequences (sound, young) with native,
same-state-swap, cross-state-swap, prototype and persistence losses; the third consequence (taxonomy) is never trained.
Held-out demonstration: arms none / self / same-donor / cross-donor / shuffled / random, scored on candidate-word
tokens only, with identical recipient-conditioned history for every arm; causal adjudication uses matched-baseline
uplift (arm minus none) and donor-versus-recipient difference-in-differences. Own-choice rollouts are secondary.

    python experiments/run_state_bus.py --config experiments/config/state_bus_v1.json --stage smoke|train
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from substitution_probe import SubstitutionProbe


class Bus(nn.Module):
    def __init__(self, d_model, d_state, n_states):
        super().__init__()
        self.ln_e, self.E = nn.LayerNorm(d_model, elementwise_affine=False), nn.Linear(d_model, d_state)
        self.J = nn.Linear(d_state, d_model, bias=False); nn.init.zeros_(self.J.weight)          # starts as a no-op
        self.ln_r, self.R = nn.LayerNorm(d_model, elementwise_affine=False), nn.Linear(d_model, d_state)
        self.proto = nn.Parameter(torch.randn(n_states, d_state))
    def enc(self, h): return self.E(self.ln_e(h))
    def read(self, h): return self.R(self.ln_r(h))


class World:
    def __init__(self, cfg):
        self.cfg = cfg; self.sp = SubstitutionProbe(cfg["model_id"], revision=cfg["revision"]); self.m = self.sp.model; self.tok = self.sp.tok
        assert self.sp.revision == cfg["revision"]; torch.set_grad_enabled(True)
        for p in self.m.parameters(): p.requires_grad_(False)
        self.L = cfg["layer"]; self.read_layer = min(self.L + 8, self.m.config.num_hidden_layers - 1); self.S = cfg["states"]; self.C = cfg["consequences"]
        self.ctx = {s: [self.tok.encode(t + cfg["anchor"], add_special_tokens=False) for t in cfg["contexts"][s]] for s in self.S}
        self.P = len(self.ctx[self.S[0]][0]); assert all(len(x) == self.P for s in self.S for x in self.ctx[s]), "contexts must be length-matched"
        self._inject = None; self._reads = None; self.forwards = 0
        self.m.model.layers[self.L].register_forward_hook(self._hook_inject); self.m.model.layers[self.read_layer].register_forward_hook(self._hook_read)
        with torch.no_grad():
            self.h_anchor = {s: torch.stack([self.m(input_ids=torch.tensor([x]), output_hidden_states=True).hidden_states[self.L + 1][0, -1] for x in self.ctx[s]]) for s in self.S}
        self.tk = lambda t: self.tok.encode(t, add_special_tokens=False)

    def _hook_inject(self, mod, i, o):
        if self._inject is None: return o
        h = o[0] if isinstance(o, tuple) else o; h = h.clone(); h[:, self.P:, :] = h[:, self.P:, :] + self._inject.to(h.dtype)
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h

    def _hook_read(self, mod, i, o):
        if self._reads is not None: self._reads.append(o[0] if isinstance(o, tuple) else o)
        return o

    def build(self, order, words):
        """Continuation token ids for decisions `order` with candidate `words`; returns ids, word spans, boundary indices
        (all relative to the continuation start). Each part is tokenized separately (every part starts with a space)."""
        ids, spans, bounds = [], [], []
        for k, name in enumerate(order):
            c = self.C[name]; pre = self.tk((c["prefix"] if k == 0 else " It" + c["prefix"])); w = self.tk(words[k]); suf = self.tk(c["suffix"])
            ids += pre; spans.append((len(ids), len(ids) + len(w))); ids += w + suf; bounds.append(len(ids) - 1)
        return ids, spans, bounds

    def forward(self, ctx, cont_ids, z=None, reads=False):
        self._inject = None if z is None else self.bus.J(z).view(1, 1, -1); self._reads = [] if reads else None
        out = self.m(input_ids=torch.tensor([ctx + cont_ids])); self.forwards += 1
        r = self._reads; self._inject = None; self._reads = None
        return out.logits[0], (r[0][0] if reads else None)

    def word_ll(self, logits, ctx_len, cont_ids, span):
        """Summed log-likelihood of the candidate-word tokens in span (teacher forced)."""
        a, b = span; lp = torch.log_softmax(logits[ctx_len + a - 1: ctx_len + b - 1].float(), -1)
        return lp.gather(1, torch.tensor(cont_ids[a:b]).unsqueeze(1)).sum()

    def z_of(self, s, i): return self.bus.enc(self.h_anchor[s][i])


def train(W, cfg, seed, log, cap_s):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); tc = cfg["train"]; lw = tc["loss_weights"]
    W.bus = Bus(W.m.config.hidden_size, cfg["state_dim"], len(W.S)); opt = torch.optim.AdamW(W.bus.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    n_par = sum(p.numel() for p in W.bus.parameters()); assert n_par < 100_000, n_par
    tr = cfg["train_indices"]; order = cfg["trained_consequences"]; t0 = time.time(); hist = []
    words = lambda st: [W.C[c]["words"][st] for c in order]
    for step in range(tc["steps"]):
        s = W.S[rng.integers(len(W.S))]; i, j = rng.choice(tr, 2, replace=False); s2 = W.S[(W.S.index(s) + rng.integers(1, len(W.S))) % len(W.S)]; k = rng.choice(tr)
        ctx = W.ctx[s][i]; arms = [("native", W.z_of(s, i), s), ("same_swap", W.z_of(s, j), s), ("cross_swap", W.z_of(s2, k), s2)]
        losses = {"persistence": 0.0}; opt.zero_grad()
        for name, z, target in arms:                                     # three arms accumulated into one optimizer step
            cont, spans, bounds = W.build(order, words(target)); lg, h20 = W.forward(ctx, cont, z, reads=True)
            losses[name] = -sum(W.word_ll(lg, W.P, cont, sp) for sp in spans) / len(spans)
            losses["persistence"] = losses["persistence"] + sum(F.mse_loss(W.bus.read(h20[W.P + b]), z.detach()) for b in bounds) / (len(bounds) * len(arms))
        Z = torch.stack([W.z_of(st, ii) for st in W.S for ii in tr]); lab = torch.tensor([W.S.index(st) for st in W.S for ii in tr])
        d = torch.cdist(Z, W.bus.proto); losses["prototype"] = (d.gather(1, lab.unsqueeze(1)).squeeze(1) ** 2).mean() + F.relu(tc["prototype_margin"] - torch.pdist(W.bus.proto)).mean()
        loss = sum(lw[kk] * v for kk, v in losses.items()); loss.backward(); opt.step()
        hist.append({kk: float(v) for kk, v in losses.items()} | {"step": step})
        if step % 25 == 0 or step == tc["steps"] - 1: log(f"seed {seed} step {step}: " + " ".join(f"{kk}={float(v):.3f}" for kk, v in losses.items()) + f" ({time.time()-t0:.0f}s)")
        if time.time() - t0 > cap_s: log(f"seed {seed}: training wall cap reached at step {step}"); break
    return hist, n_par


def signature_scores(W, cfg, s, i, z):
    """Primary scoring: for each decision k, summed word-LL of each candidate with history = recipient's own words."""
    order = cfg["decision_order"]; out = []
    for k, name in enumerate(order):
        prev = [W.C[c]["words"][s] for c in order[:k]]; lls = {}
        for cand in W.S:
            cont, spans, _ = W.build(order[:k + 1], prev + [W.C[name]["words"][cand]]); lg, _ = W.forward(W.ctx[s][i], cont, z)
            lls[cand] = float(W.word_ll(lg, W.P, cont, spans[k]))
        out.append(lls)
    return out


def rollout(W, cfg, s, i, z):
    """Secondary: the model's own sequential argmax choices (summed word-LL) under z."""
    order = cfg["decision_order"]; chosen = []
    for k, name in enumerate(order):
        lls = {}
        for cand in W.S:
            cont, spans, _ = W.build(order[:k + 1], chosen + [W.C[name]["words"][cand]]); lg, _ = W.forward(W.ctx[s][i], cont, z)
            lls[cand] = float(W.word_ll(lg, W.P, cont, spans[k]))
        chosen.append(W.C[name]["words"][max(lls, key=lls.get)])
    return chosen


def margins(lls_list, s):
    return np.array([d[s] - max(v for c, v in d.items() if c != s) for d in lls_list])


def evaluate(W, cfg, log):
    S, ho, tr = W.S, cfg["heldout_indices"], cfg["train_indices"]; rows = []; reads = []
    with torch.no_grad():
        tau_d = []                                                       # tolerance from TRAINING contexts: same-donor vs self signature distance
        for s in S:
            for i in tr:
                j = tr[(tr.index(i) + 1) % len(tr)]; a = margins(signature_scores(W, cfg, s, i, W.z_of(s, i)), s); b = margins(signature_scores(W, cfg, s, i, W.z_of(s, j)), s)
                tau_d.append(float(np.linalg.norm(a - b)))
        tau = max(1e-3, float(np.percentile(tau_d, 95))); log(f"tau (Q95 of training same-vs-self distances) = {tau:.4f}")
        for s in S:
            other = S[(S.index(s) + 1) % len(S)]
            for i in ho:
                z_own = W.z_of(s, i); j = ho[(ho.index(i) + 1) % len(ho)]; g = torch.Generator().manual_seed(7 + 10 * S.index(s) + i)
                rnd = torch.randn(z_own.shape, generator=g); rnd = rnd / rnd.norm() * z_own.norm()
                arms = {"none": None, "self": z_own, "same": W.z_of(s, j), "cross": W.z_of(other, i), "shuffled": z_own[torch.randperm(z_own.numel(), generator=g)], "random": rnd}
                sc = {a: signature_scores(W, cfg, s, i, z) for a, z in arms.items()}
                roll = {a: rollout(W, cfg, s, i, arms[a]) for a in ("self", "cross")}
                # reader reconstruction on held-out at both trained boundaries (self code)
                cont, spans, bounds = W.build(cfg["trained_consequences"], [W.C[c]["words"][s] for c in cfg["trained_consequences"]]); _, h20 = W.forward(W.ctx[s][i], cont, z_own, reads=True)
                rec = [float(F.mse_loss(W.bus.read(h20[W.P + b]), z_own)) for b in bounds]; reads.append(rec)
                rows.append({"state": s, "i": i, "donor_state": other, "scores": sc, "rollout": roll, "reader_mse": rec, "z_norm": float(z_own.norm())})
                log(f"{s}{i}: self-roll={roll['self']} cross-roll={roll['cross']} reader_mse={[round(x, 3) for x in rec]}")
    return rows, tau


def verdict(rows, tau, cfg, log):
    K, order = cfg["kill"], cfg["decision_order"]; tr_k = [order.index(c) for c in cfg["trained_consequences"]]; ho_k = order.index(cfg["heldout_consequence"]); n = len(rows)
    raw_arg = lambda r, a, k: max(r["scores"][a][k], key=r["scores"][a][k].get)
    uplift = lambda r, a, k: {c: r["scores"][a][k][c] - r["scores"]["none"][k][c] for c in r["scores"][a][k]}
    up_arg = lambda r, a, k: max(uplift(r, a, k), key=uplift(r, a, k).get)
    did = lambda r, a, k: uplift(r, a, k)[r["donor_state"]] - uplift(r, a, k)[r["state"]]
    trained_acc = float(np.mean([raw_arg(r, "self", k) == r["state"] for r in rows for k in tr_k]))
    same_d = [float(np.linalg.norm(margins(r["scores"]["same"], r["state"]) - margins(r["scores"]["self"], r["state"]))) for r in rows]
    cross_two = int(sum(sum(up_arg(r, "cross", k) == r["donor_state"] for k in range(len(order))) >= 2 for r in rows))
    donor_frac = {a: float(np.mean([up_arg(r, a, ho_k) == r["donor_state"] for r in rows])) for a in ("cross", "shuffled", "random")}
    ho_cons = int(sum(up_arg(r, "cross", ho_k) == r["donor_state"] for r in rows)); gain = donor_frac["cross"] - (donor_frac["shuffled"] + donor_frac["random"]) / 2
    mv = [float(np.mean([did(r, "cross", k) for r in rows])) for k in range(len(order))]; m1, m3 = mv[0], mv[ho_k]
    per_state = {s: [float(np.mean([did(r, "cross", k) for r in rows if r["state"] == s])) for k in range(len(order))] for s in cfg["states"]}
    summary = {"n_recipients": n, "trained_consequence_accuracy_self": trained_acc, "tau": tau, "same_out_of_tolerance": int(sum(d > tau for d in same_d)), "same_distances": same_d,
               "cross_two_of_three": cross_two, "heldout_consistent_cross": ho_cons, "donor_fraction_heldout": donor_frac, "heldout_gain": float(gain),
               "movement_did_by_decision": mv, "movement_by_state": per_state, "reader_mse_mean": float(np.mean([r["reader_mse"] for r in rows])),
               "raw_arm_accuracy": {a: [float(np.mean([raw_arg(r, a, k) == (r["donor_state"] if a == "cross" else r["state"]) for r in rows])) for k in range(len(order))] for a in ("none", "self", "same", "cross", "shuffled", "random")},
               "rollout_cross_donor_words": int(sum(all(r["rollout"]["cross"][k] == cfg["consequences"][order[k]]["words"][r["donor_state"]] for k in range(len(order))) for r in rows))}
    fails = []
    if trained_acc < K["trained_min"]: fails.append("trained")
    if summary["same_out_of_tolerance"] > K["same_out_of_tolerance_max"]: fails.append("same_swap")
    if cross_two < K["cross_two_of_three_min"]: fails.append("cross_swap")
    if not (m1 > 0 and m3 > 0 and m3 >= K["third_over_first_min"] * m1): fails.append("persistence")
    tax_fail = ho_cons < K["heldout_consistent_min"] or gain < K["heldout_gain_min"]
    status = "FAIL — " + ", ".join(fails + (["heldout_consequence"] if tax_fail else [])) if fails else ("SUPERVISED RESPONSE CONTROLLER — STOP" if tax_fail else "BOUNDED POSITIVE — PERSISTENT INTERCHANGEABLE STATE BUS (this construction)")   # audit #29: list every failed gate
    summary["fails"] = fails + (["heldout_consequence"] if tax_fail else []); log(json.dumps({k: v for k, v in summary.items() if k != "same_distances"}, indent=1)); log(f"STATUS: {status}")
    return summary, status


FLOOR_NATS = 0.5   # audit #28 magnitude floor on donor-vs-recipient DiD (summed word log-likelihood), declared pre-result


def audit_stage(W, cfg, out, log):
    """Audit-#28 re-adjudication from saved bus weights: raw choice change, magnitude floor, all donor pairs, on-manifold control."""
    S, ho, order = W.S, cfg["heldout_indices"], cfg["decision_order"]; k_tax = order.index(cfg["heldout_consequence"]); res = {}
    for seed in cfg["train"]["seeds"]:
        path = os.path.join(out, f"bus_seed{seed}.pt")
        if not os.path.exists(path): log(f"seed {seed}: no saved bus"); continue
        W.bus = Bus(W.m.config.hidden_size, cfg["state_dim"], len(S)); W.bus.load_state_dict(torch.load(path)); W.bus.eval(); rows = []
        with torch.no_grad():
            for s in S:
                for i in ho:
                    sc = {"none": signature_scores(W, cfg, s, i, None), "self": signature_scores(W, cfg, s, i, W.z_of(s, i))}
                    for d in S:
                        if d != s: sc[f"cross_{d}"] = signature_scores(W, cfg, s, i, W.z_of(d, i))
                    rows.append({"state": s, "i": i, "scores": sc})
        raw = lambda r, a, k: max(r["scores"][a][k], key=r["scores"][a][k].get)
        did = lambda r, a, k, d: (r["scores"][a][k][d] - r["scores"]["none"][k][d]) - (r["scores"][a][k][r["state"]] - r["scores"]["none"][k][r["state"]])
        per_pair = {}; tax_choice = []; tax_floor = []; ctrl = []; per_state = {st: 0 for st in S}
        for r in rows:
            s = r["state"]
            for d in S:
                if d == s: continue
                a = f"cross_{d}"; ch = [raw(r, a, k) == d for k in range(len(order))]; m = did(r, a, k_tax, d)
                per_pair.setdefault(f"{s}->{d}", []).append({"choices": ch, "did_tax": m})
                tax_choice.append(ch[k_tax]); tax_floor.append(ch[k_tax] and m >= FLOOR_NATS)
                for d2 in S:                                   # on-manifold control: third-state code pushing toward d?
                    if d2 not in (s, d): ctrl.append(raw(r, f"cross_{d2}", k_tax) == d)
            per_state[s] += sum(raw(r, f"cross_{d}", k_tax) == d for d in S if d != s) >= 2   # recipient counts if >=2 of 3 donors move its taxonomy choice
        summ = {"n_rows": len(rows), "taxonomy_raw_choice_donor_fraction": float(np.mean(tax_choice)), "taxonomy_raw_choice_with_floor_fraction": float(np.mean(tax_floor)),
                "on_manifold_control_fraction": float(np.mean(ctrl)), "gain_over_on_manifold_control": float(np.mean(tax_choice) - np.mean(ctrl)),
                "trained_raw_choice_donor_fraction": [float(np.mean([raw(r, f"cross_{d}", k) == d for r in rows for d in S if d != r["state"]])) for k in range(len(order))],
                "per_state_recipients_moved": per_state, "did_tax_median": float(np.median([x["did_tax"] for v in per_pair.values() for x in v])),
                "per_pair_taxonomy_fraction": {k: float(np.mean([x["choices"][k_tax] for x in v])) for k, v in per_pair.items()}}
        ok_tax = summ["taxonomy_raw_choice_with_floor_fraction"] >= 10 / 16 and summ["gain_over_on_manifold_control"] >= 0.25 and all(v >= 2 for v in per_state.values())
        ok_trained = all(f >= 0.85 for f in summ["trained_raw_choice_donor_fraction"][:k_tax])
        summ["class"] = "POSITIVE" if (ok_tax and ok_trained) else ("CONTROLLER" if ok_trained else "FAIL"); res[seed] = {"summary": summ, "rows": rows}
        log(f"seed {seed} audit-stage: {json.dumps({k: v for k, v in summ.items() if k != 'per_pair_taxonomy_fraction'})}")
    kinds = [v["summary"]["class"] for v in res.values()]
    overall = "INCOMPLETE - NO VERDICT" if len(kinds) < 3 else (max(set(kinds), key=kinds.count) if max(kinds.count(k) for k in set(kinds)) >= 2 else "SPLIT - NO VERDICT")
    json.dump({"floor_nats": FLOOR_NATS, "seeds": res, "overall": overall, "per_seed": kinds}, open(os.path.join(out, "audit_result.json"), "w"), indent=1, default=float)
    log(f"AUDIT-STAGE OVERALL: {overall} per seed {kinds}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--stage", default="smoke"); a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8")); out = f"experiments/results/{cfg['name']}"; os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, f"{a.stage}.log"), "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    shas = {k: hashlib.sha256(open(v, "rb").read()).hexdigest() for k, v in (("runner", __file__), ("config", a.config), ("substitution_probe", os.path.join(os.path.dirname(__file__), "substitution_probe.py")))}
    T0 = time.time(); deadline = T0 + cfg["train"]["wall_cap_hours"] * 3600; W = World(cfg); log(f"loaded {cfg['model_id']} rev={W.sp.revision}; P={W.P}; read layer {W.read_layer} ({time.time()-T0:.0f}s)")
    if a.stage == "audit": return audit_stage(W, cfg, out, log)
    if a.stage == "smoke":
        cfg = dict(cfg); cfg["train"] = dict(cfg["train"], steps=3); cfg["heldout_indices"] = cfg["heldout_indices"][:1]; cfg["train_indices"] = cfg["train_indices"][:2]
        hist, n = train(W, cfg, 11, log, 600); log(f"bus params {n}"); W.bus.eval(); rows, tau = evaluate(W, cfg, log); verdict(rows, tau, cfg, log); log(f"smoke ok {W.forwards} forwards {time.time()-T0:.0f}s"); return
    result = {"config": cfg["name"], "sha256": shas, "revision": W.sp.revision, "seeds": {}}; seeds = cfg["train"]["seeds"]; statuses = []
    for n_done, seed in enumerate(seeds):
        remaining = deadline - time.time(); cap = max(60, remaining / (len(seeds) - n_done) - 25 * 60)     # leave ~25 min per seed for evaluation
        if remaining < 30 * 60: log(f"global deadline: skipping seed {seed}"); break
        hist, n = train(W, cfg, seed, log, cap); torch.save(W.bus.state_dict(), os.path.join(out, f"bus_seed{seed}.pt")); W.bus.eval()
        te = time.time(); rows, tau = evaluate(W, cfg, log); summary, status = verdict(rows, tau, cfg, log); statuses.append(status)
        result["seeds"][seed] = {"params": n, "history": hist, "rows": rows, "summary": summary, "status": status, "eval_seconds": time.time() - te}
        json.dump(result, open(os.path.join(out, "result.json"), "w"), indent=1, default=float); log(f"seed {seed} done ({time.time()-T0:.0f}s, {W.forwards} forwards)")
    kinds = [s.split(" — ")[0] for s in statuses]; overall = max(set(kinds), key=kinds.count) if kinds else "NO SEED COMPLETED"
    result["overall"] = {"per_seed": statuses, "majority": overall, "unanimous": len(set(kinds)) == 1}; json.dump(result, open(os.path.join(out, "result.json"), "w"), indent=1, default=float)
    log(f"OVERALL (majority of {len(statuses)} seeds): {overall}; per seed: {statuses}; done in {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
