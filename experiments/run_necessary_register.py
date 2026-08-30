"""necessary_register: constructed register substrate (Codex rounds 24-27). Stage rung0 = INSTRUMENT CHECK: eight fixed orthonormal
oracle codes replace the <REG> input embedding of a from-scratch hard-masked two-layer transformer; each episode shows a visible
permutation pi of 8 abstract states onto a panel's 8 labels; train/eval permutations disjoint; readout = pi^-1(label).
Stage writer (rung 1): replay + freeze the rung-0 consumers, crossed oracle regression, then train ONLY a source writer
(<SRC> <E_e> <HAS> <V_s> <WRITE> -> GRU -> unit vector replacing the register embedding) with the answer-only loss.

    python experiments/run_necessary_register.py --config experiments/config/necessary_register_v1.json [--smoke]
    python experiments/run_necessary_register.py --config experiments/config/necessary_register_rung1.json [--smoke]
"""
from __future__ import annotations
import argparse, gzip, hashlib, itertools, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

PAD, BOS, REG, MAP, ANS, EOS = range(6); E0, Q0, L0 = 6, 30, 34                      # entities 6-29, query tokens 30-33, labels 34-65 (4 panels x 8)
TEMPLATES = [lambda e: [Q0, e], lambda e: [e, Q0 + 1], lambda e: [Q0 + 2, e, Q0 + 2], lambda e: [Q0 + 3, Q0 + 3, e]]
VOCAB = L0 + 32; R = 1                                                                  # register position
SRC, HAS, WRITE, MASK, SE0, SV0 = 0, 1, 2, 3, 4, 28                                     # source vocabulary: 4 marks, 24 entities, 8 values
RUNG0_STATUS = "QUALIFIED INSTRUMENT PASS — PERMUTATION-GENERALIZING ORACLE-CODE LABEL SELECTOR"


class RegisterLM(nn.Module):
    def __init__(self, mc, T):
        super().__init__(); d = mc["d_model"]; self.emb = nn.Embedding(VOCAB, d); self.pos = nn.Embedding(T, d)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, mc["n_heads"], mc["d_ff"], mc["dropout"], batch_first=True, norm_first=True) for _ in range(mc["n_layers"])])
        self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, VOCAB); self.register_buffer("mask", self.build_mask(T))
    @staticmethod
    def build_mask(T):
        m = torch.full((T, T), float("-inf")); i = torch.arange(T); m[i[:, None] >= i[None, :]] = 0.0; m[R + 1:, :R] = float("-inf"); return m   # causal; post-register positions cannot see pre-register positions
    def forward(self, ids, code=None):
        x = self.emb(ids) + self.pos(torch.arange(ids.shape[1]))[None]
        if code is not None: x = x.clone(); x[:, R, :] = code                              # write REPLACES the register embedding
        for l in self.layers: x = l(x, src_mask=self.mask[: ids.shape[1], : ids.shape[1]])
        return self.head(self.norm(x))


class Writer(nn.Module):
    def __init__(self, wc, d):
        super().__init__(); self.emb = nn.Embedding(SV0 + 8, wc["src_dim"]); self.gru = nn.GRU(wc["src_dim"], wc["hidden"], batch_first=True); self.lin = nn.Linear(wc["hidden"], d, bias=False)
    def forward(self, src):
        v = self.lin(self.gru(self.emb(src))[1][0]); return F.normalize(v, dim=-1), v.norm(dim=-1)


def source(ents, vals):
    """<SRC> <E_e|MASK> <HAS> <V_s|MASK> <WRITE>; -1 = masked."""
    e = torch.as_tensor(np.asarray(ents)); v = torch.as_tensor(np.asarray(vals)); return torch.stack([torch.full_like(e, SRC), torch.where(e < 0, torch.full_like(e, MASK), SE0 + e), torch.full_like(e, HAS), torch.where(v < 0, torch.full_like(v, MASK), SV0 + v), torch.full_like(e, WRITE)], 1)


def episodes(W, rng, B, perms, states=None, ents=None, tpls=None, panels=None, pidx=None):
    """Batch of episodes: <BOS> <REG> <MAP> labels(pi(0..7)) query <ANS> label(pi(s)) <EOS>. Returns ids, answer positions, meta."""
    K = W["n_states"]; ents = rng.integers(W["n_entities"], size=B) if ents is None else ents; states = rng.integers(K, size=B) if states is None else states
    tpls = rng.integers(W["n_templates"], size=B) if tpls is None else tpls; panels = rng.integers(W["n_panels"], size=B) if panels is None else panels; pidx = rng.integers(len(perms[0]), size=B) if pidx is None else pidx
    seqs, tgt, meta = [], [], []
    for b in range(B):
        pi = perms[panels[b]][pidx[b]]; labels = [L0 + 8 * panels[b] + int(pi[s]) for s in range(K)]
        q = TEMPLATES[tpls[b]](E0 + int(ents[b])); seq = [BOS, REG, MAP] + labels + q + [ANS]; seqs.append(seq); tgt.append([labels[states[b]], EOS]); meta.append((int(ents[b]), int(states[b]), int(tpls[b]), int(panels[b]), int(pidx[b]), pi))
    T = max(map(len, seqs)); ids = torch.full((B, T + 2), PAD, dtype=torch.long)
    for b, s in enumerate(seqs): ids[b, : len(s)] = torch.tensor(s); ids[b, len(s): len(s) + 2] = torch.tensor(tgt[b])
    return ids, torch.tensor([len(s) - 1 for s in seqs]), meta


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args(); T0 = time.time()
    cfg = json.load(open(a.config, encoding="utf-8")); out = f"experiments/results/{cfg['name']}"; os.makedirs(out, exist_ok=True); tag = "smoke" if a.smoke else "run"; logf = open(os.path.join(out, f"{tag}.log"), "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    stage = cfg.get("stage", "rung0"); shas = {k: hashlib.sha256(open(v, "rb").read()).hexdigest() for k, v in (("runner", __file__), ("config", a.config))}
    if stage == "writer":                                                                                                                 # rung 1 binds the rung-0 config, result and runner by hash; consumer hyperparameters come from the rung-0 config
        r0 = cfg["rung0"]; shas |= {"rung0_config": hashlib.sha256(open(r0["config"], "rb").read()).hexdigest(), "rung0_result": hashlib.sha256(open(r0["result"], "rb").read()).hexdigest()}
        base = json.load(open(r0["config"], encoding="utf-8")); prior = json.load(open(r0["result"], encoding="utf-8")); W, mc, tc, G = base["world"], base["model"], base["train"], base["gates"]; WG = cfg["gates"]
        assert shas["rung0_config"] == r0["config_sha256"] and shas["rung0_result"] == r0["result_sha256"] and prior["sha256"]["runner"] == r0["runner_sha256_at_rung0"], "rung-0 binding mismatch"
    else: W, mc, tc, G = cfg["world"], cfg["model"], cfg["train"], cfg["gates"]
    if a.smoke: cfg["train"]["steps"] = int(os.environ.get("SMOKE_STEPS", 200)); cfg["train"]["seeds"] = cfg["train"]["seeds"][:1]; cfg["eval"]["bootstraps"] = 100
    K = W["n_states"]; prng = np.random.default_rng(W["perm_seed"]); allp = {}
    for p in range(W["n_panels"]):
        seen, bank = set(), []
        while len(bank) < W["train_perms_per_panel"] + W["eval_perms_per_panel"]:
            q = tuple(prng.permutation(K))
            if q not in seen: seen.add(q); bank.append(np.array(q))
        allp[p] = bank
    train_perms = [allp[p][: W["train_perms_per_panel"]] for p in range(W["n_panels"])]; eval_perms = [allp[p][W["train_perms_per_panel"]:] for p in range(W["n_panels"])]
    g = torch.Generator().manual_seed(mc["code_seed"]); C = torch.linalg.qr(torch.randn(mc["d_model"], K, generator=g))[0].T.contiguous()          # 8 orthonormal codes in R^128
    Rg = torch.randn(cfg["eval"]["n_random"], mc["d_model"], generator=g); Rg = Rg / Rg.norm(dim=1, keepdim=True)                                   # fixed norm-matched random codes
    res = {"config": cfg["name"], "stage": stage, "sha256": shas, "codebook_hash": hashlib.sha256(C.numpy().tobytes()).hexdigest(), "perm_hash": hashlib.sha256(json.dumps([[q.tolist() for q in allp[p]] for p in allp]).encode()).hexdigest(), "seeds": {}}
    save = lambda: json.dump(res, open(os.path.join(out, f"{tag}_result.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else float(o))
    dump = lambda name, obj: json.dump(obj, gzip.open(os.path.join(out, f"{tag}_{name}.json.gz"), "wt"), default=lambda o: o.item() if hasattr(o, "item") else float(o))
    Tmax = 3 + K + 3 + 1 + 2; log(f"stage {stage}; vocab {VOCAB}; codes {res['codebook_hash'][:10]}; perms {res['perm_hash'][:10]}; Tmax {Tmax}")
    deadline = T0 + cfg["train"]["hard_wall_minutes"] * 60; NR = cfg["eval"]["n_random"]; NB = cfg["eval"]["bootstraps"]; crossed_states = lambda ents: ents % K
    def fit(model, params, steps, code_fn, rng, seed):
        """Answer-label + EOS cross-entropy only; code_fn(meta) -> register write (B x d). Returns (loss history, completed)."""
        opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"]); t0 = time.time(); hist = []
        for step in range(steps):
            ids, ans, meta = episodes(W, rng, tc["batch"], train_perms); lg = model(ids[:, :-1], code_fn(meta)); tgt = ids[:, 1:]; b = torch.arange(len(ids))
            loss = F.cross_entropy(lg[b, ans], tgt[b, ans]) + F.cross_entropy(lg[b, ans + 1], tgt[b, ans + 1])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, tc["clip"]); opt.step(); hist.append(float(loss.detach()))
            if step % 250 == 0 or step == steps - 1: log(f"seed {seed} step {step}: loss={float(loss):.3f} ({time.time()-t0:.0f}s)")
            if time.time() > deadline: log("hard wall"); return hist, False
        return hist, True
    def run(model, perms, n, rng, arms_fn, states=None):
        """Episodes (rung-0 marginal schedule when states is None; else factorial entity x template x panel); arms_fn(ents, states, meta) -> {arm: write}."""
        if states is None: ents = np.repeat(np.arange(W["n_entities"]), n // W["n_entities"])[:n]; states = np.tile(np.arange(K), n // K)[:n]; tpls = np.tile(np.arange(W["n_templates"]), n // W["n_templates"])[:n]; panels = np.tile(np.repeat(np.arange(W["n_panels"]), 4), n // 16 + 1)[:n]
        else: grid = np.array(list(itertools.product(range(W["n_entities"]), range(W["n_templates"]), range(W["n_panels"])))); grid = np.tile(grid, (n // len(grid) + 1, 1))[:n]; ents, tpls, panels = grid[:, 0], grid[:, 1], grid[:, 2]; states = states(ents)
        pidx = rng.integers(len(perms[0]), size=n); ids, ans, meta = episodes(W, rng, n, perms, states, ents, tpls, panels, pidx); rows = []; b = torch.arange(n)
        def decode(code):
            with torch.no_grad(): lg = model(ids[:, :-1], code); return lg[b, ans].argmax(-1).numpy(), (lg[b, ans + 1].argmax(-1) == EOS).numpy()
        arms = arms_fn(ents, states, meta) | {"zero": torch.zeros(n, mc["d_model"])} | {f"rand{r}": Rg[r].expand(n, -1) for r in range(NR)}; outs = {k: decode(v) for k, v in arms.items()}
        with torch.no_grad():                                                                                                         # zero-hook identity: the replacement path fed the learned <REG> input (token + position) vs the untouched path: decoded state, EOS, raw ids, full-logit gap
            LH = model(ids[:, :-1], (model.emb.weight[REG] + model.pos.weight[R]).detach().expand(n, -1)); LU = model(ids[:, :-1], None); zh, zu = LH[b, ans].argmax(-1).numpy(), LU[b, ans].argmax(-1).numpy(); ze, ue = (LH[b, ans + 1].argmax(-1) == EOS).numpy(), (LU[b, ans + 1].argmax(-1) == EOS).numpy(); gap = float((LH - LU).abs().max())
        for i, (e, s, t, p, pi_i, pi) in enumerate(meta):
            inv = {L0 + 8 * p + int(pi[st]): st for st in range(K)}; row = {"e": e, "s": s, "t": t, "p": p, "pi": pi_i}
            for k, (first, term) in outs.items(): row[k] = {"state": inv.get(int(first[i]), -1), "tok": int(first[i]), "term": bool(term[i])}
            row |= {"zero_hook_state": inv.get(int(zh[i]), -1), "untouched_state": inv.get(int(zu[i]), -1), "zero_hook_tok": int(zh[i]), "untouched_tok": int(zu[i]), "zero_hook_eos": bool(ze[i]), "untouched_eos": bool(ue[i]), "zero_hook_logit_gap": gap}; rows.append(row)
        return rows
    def common(rows, trows, rng, main="own"):
        """Shared summary: own / cf1-7 / zero / rand, per-group, agreement, per-arm termination, entity-paired uplift, entity-bootstrap lower bounds."""
        own = np.array([r[main]["state"] == r["s"] for r in rows], float); by_e = lambda v: np.array([v[[r["e"] == e for r in rows]].mean() for e in range(W["n_entities"])])
        boot = lambda pe: float(np.quantile([np.mean(rng.choice(pe, len(pe))) for _ in range(NB)], 0.025))
        cf_ok = np.array([np.mean([r[f"cf{j}"]["state"] == (r["s"] + j) % K for j in range(1, K)]) for r in rows]); paired = own * cf_ok; acc = lambda rs: float(np.mean([r[main]["state"] == r["s"] for r in rs]))
        groups = {"state": [acc([r for r in rows if r["s"] == s]) for s in range(K)], "panel": [acc([r for r in rows if r["p"] == p]) for p in range(W["n_panels"])], "template": [acc([r for r in rows if r["t"] == t]) for t in range(W["n_templates"])]}
        keyf = lambda r: (r["s"], r["e"], r["t"]); agree = []
        for key in {keyf(r) for r in rows}:
            st = [r[main]["state"] for r in rows if keyf(r) == key]
            if len(st) > 1: agree.append(np.mean([st[i] == st[j] for i in range(len(st)) for j in range(i + 1, len(st))]))
        zero_r = np.array([r["zero"]["state"] == r["s"] for r in rows], float); rnd_r = np.array([np.mean([r[f"rand{k}"]["state"] == r["s"] for k in range(NR)]) for r in rows]); ctrl_e = np.maximum(by_e(zero_r), by_e(rnd_r))
        armk = [k for k in rows[0] if isinstance(rows[0][k], dict)]; term_arm = {k: float(np.mean([r[k]["term"] for r in rows])) for k in armk}; dist = {k: np.bincount([r[k]["state"] + 1 for r in rows], minlength=K + 1).tolist() for k in armk if k == "zero" or k.startswith("rand")}
        zh_ok = all(r["zero_hook_state"] == r["untouched_state"] and r["zero_hook_tok"] == r["untouched_tok"] and r["zero_hook_eos"] == r["untouched_eos"] for r in rows)
        return {"heldout_acc": float(own.mean()), "heldout_acc_lb": boot(by_e(own)), "train_perm_acc": float(np.mean([r[main]["state"] == r["s"] for r in trows])), "groups": groups, "agreement": float(np.mean(agree)), "cf_follow": float(cf_ok.mean()), "paired_dir": float(paired.mean()), "paired_dir_lb": boot(by_e(paired)),
                "zero_assigned": float(zero_r.mean()), "random_assigned": float(rnd_r.mean()), "uplift": float(own.mean() - max(zero_r.mean(), rnd_r.mean())), "uplift_paired": float((by_e(own) - ctrl_e).mean()), "uplift_lb": boot(by_e(own) - ctrl_e), "termination": term_arm[main], "termination_min_arm": min(term_arm.values()), "termination_by_arm": term_arm,
                "control_state_distributions": dist, "zero_hook_identity": zh_ok, "zero_hook_logit_gap": rows[0]["zero_hook_logit_gap"], "n": len(rows)}, by_e, boot
    def instrument(model, rng, crossed):
        """Rung-0 gates on the oracle consumer. crossed=False keeps the historical rng order (exact replay); crossed=True is the audit-#37 factorial regression."""
        oracle = lambda ents, states, meta: {"own": C[torch.tensor(states)]} | {f"cf{j}": C[torch.tensor((states + j) % K)] for j in range(1, K)}; st = crossed_states if crossed else None
        rows = run(model, eval_perms, cfg["eval"]["episodes_per_arm"], rng, oracle, st); trows = run(model, train_perms, 768, rng, oracle, st); s, _, _ = common(rows, trows, rng)
        mask_ok = float(model.mask[R + 1, 0]) == float("-inf") and float(model.mask[R, 0]) == 0.0; zero, rnd, train_acc, term = s["zero_assigned"], s["random_assigned"], s["train_perm_acc"], (s["termination_min_arm"] if crossed else s["termination"])
        if not (mask_ok and s["zero_hook_identity"] and s["zero_hook_logit_gap"] <= 1e-4): cls = "INVALID — MASK/HOOK"
        elif train_acc >= G["lookup_train_min"] and (s["heldout_acc"] < G["lookup_heldout_max"] or train_acc - s["heldout_acc"] > G["lookup_gap_max"]): cls = "LOOKUP-BOUND INVALID"
        elif (term >= G["termination_min"] and s["heldout_acc"] >= G["heldout_acc_min"] and s["heldout_acc_lb"] > G["heldout_acc_lb_min"] and min(min(v) for v in s["groups"].values()) >= G["per_group_min"] and s["agreement"] >= G["agreement_min"] and s["paired_dir"] >= G["paired_dir_min"] and s["paired_dir_lb"] > G["paired_dir_lb_min"]
              and s["uplift"] >= G["uplift_min"] and s["uplift_lb"] > G["uplift_lb_min"] and zero <= G["control_max"] and rnd <= G["control_max"]): cls = "VALID"
        else: cls = "INVALID — ORACLE REGISTER CONSUMER"
        s["class"] = cls; return s, rows, trows
    for seed in cfg["train"]["seeds"]:
        torch.manual_seed(seed); rng = np.random.default_rng(seed); model = RegisterLM(mc, Tmax)
        hist, done = fit(model, list(model.parameters()), cfg["train"]["steps"] if (a.smoke or stage == "rung0") else tc["steps"], lambda meta: C[torch.tensor([m[1] for m in meta])], rng, seed); model.eval(); s, rows0, trows0 = instrument(model, rng, False); s["loss_first"], s["loss_last"] = hist[0], float(np.mean(hist[-50:]))
        if not done: s["class"] = "INCOMPLETE — DEADLINE"
        if stage == "rung0":
            res["seeds"][seed] = {"summary": s, "loss_history": hist[::25]}; dump(f"rows_seed{seed}", {"heldout": rows0, "train": trows0}); save(); log(f"seed {seed}: " + json.dumps({k: v for k, v in s.items() if k not in ("groups", "termination_by_arm", "control_state_distributions")}, default=float) + f" groups={s['groups']}"); log(f"seed {seed} class: {s['class']} ({time.time()-T0:.0f}s)"); continue
        # ---- rung 1: exact replay of the stored rung-0 summary, crossed oracle regression, freeze, then train only the writer ----
        p0 = prior["seeds"][str(seed)]["summary"]; replay_ok = a.smoke or (s["class"] == "VALID" and all(abs(float(s[k]) - float(p0[k])) < 1e-9 for k in ("heldout_acc", "paired_dir", "cf_follow", "agreement", "zero_assigned", "random_assigned", "loss_last")))
        sx, rowsx, trowsx = instrument(model, rng, True); dump(f"oracle_crossed_rows_seed{seed}", {"heldout": rowsx, "train": trowsx}); ck = os.path.join(out, f"consumer_seed{seed}.pt"); torch.save(model.state_dict(), ck); ck_sha = hashlib.sha256(open(ck, "rb").read()).hexdigest()
        log(f"seed {seed} consumer replay: class {s['class']} replay_ok={replay_ok}; crossed regression: class {sx['class']} acc={sx['heldout_acc']:.3f} groups={sx['groups']}; ckpt {ck_sha[:10]}")
        pre = {"consumer_replay": s, "consumer_crossed": sx, "consumer_ckpt_sha256": ck_sha}
        if not replay_ok or (sx["class"] != "VALID" and not a.smoke): res["seeds"][seed] = pre | {"summary": {"class": "INVALID — CONSUMER PRECONDITION"}}; save(); continue
        for q in model.parameters(): q.requires_grad_(False)
        torch.manual_seed(seed + 1000); wr = Writer(cfg["writer"], mc["d_model"]); wrng = np.random.default_rng(seed + 1000)
        whist, wdone = fit(model, list(wr.parameters()), cfg["train"]["steps"], lambda meta: wr(source([m[0] for m in meta], [m[1] for m in meta]))[0], wrng, seed); wr.eval()
        def arms(ents, states, meta):
            with torch.no_grad():
                d = {"own": wr(source(ents, states))[0]} | {f"cf{j}": wr(source(ents, (states + j) % K))[0] for j in range(1, K)} | {"masked": wr(source(ents, -np.ones_like(states)))[0], "value_only": wr(source(-np.ones_like(ents), states))[0], "oracle": C[torch.tensor(states)]}
                key = lambda m: (m[3], m[2], m[4]); donor = []
                for i, mi in enumerate(meta):                                                                                          # state-changing donor within the same panel/template/permutation context (fallback: same panel/template)
                    cand = [j for j, m in enumerate(meta) if key(m) == key(mi) and m[1] != mi[1]] or [j for j, m in enumerate(meta) if (m[3], m[2]) == (mi[3], mi[2]) and m[1] != mi[1]]; donor.append(int(wrng.choice(cand)))
                donor = np.array(donor); d["shuffled"] = wr(source(ents[donor], states[donor]))[0]; d["_donor"] = donor; return d
        def run_w(perms, n):
            cap = {}
            def af(ents, states, meta):
                d = arms(ents, states, meta); cap["donor"] = d.pop("_donor"); return d
            rows = run(model, perms, n, wrng, af, crossed_states)
            for i, r in enumerate(rows): j = int(cap["donor"][i]); r["donor_s"] = rows[j]["s"]; r["donor_matched_perm"] = rows[j]["pi"] == r["pi"]
            return rows
        rows = run_w(eval_perms, cfg["eval"]["episodes_per_arm"]); trows = run_w(train_perms, 768); s1, by_e, boot = common(rows, trows, wrng); s1 |= pre; dump(f"rows_seed{seed}", {"heldout": rows, "train": trows})
        wk = os.path.join(out, f"writer_seed{seed}.pt"); torch.save(wr.state_dict(), wk); s1["writer_ckpt_sha256"] = hashlib.sha256(open(wk, "rb").read()).hexdigest()
        s1["loss_first"], s1["loss_last"] = whist[0], float(np.mean(whist[-50:])); s1["donor_context_matched"] = float(np.mean([r["donor_matched_perm"] for r in rows])); s1["value_only_acc"] = float(np.mean([r["value_only"]["state"] == r["s"] for r in rows]))
        dn = np.array([r["shuffled"]["state"] == r["donor_s"] for r in rows], float); rc = np.array([r["shuffled"]["state"] == r["s"] for r in rows], float); mk = float(np.mean([r["masked"]["state"] == r["s"] for r in rows])); orc = float(np.mean([r["oracle"]["state"] == r["s"] for r in rows]))
        ctrl = max(mk, s1["zero_assigned"], s1["random_assigned"]); s1 |= {"shuffled_donor_follow": float(dn.mean()), "shuffled_recipient_follow": float(rc.mean()), "shuffled_effect_lb": boot(by_e(dn) - by_e(rc)), "masked_assigned": mk, "oracle_acc": orc, "oracle_recovery": float((s1["heldout_acc"] - ctrl) / max(orc - ctrl, 1e-9))}
        with torch.no_grad():                                                                                                         # code-space telemetry (diagnostic only)
            ee = np.repeat(np.arange(W["n_entities"]), K); ss = np.tile(np.arange(K), W["n_entities"]); w, nrm = wr(source(ee, ss)); cs = w @ C.T; tss = torch.tensor(ss); true = cs[torch.arange(len(ss)), tss]; top2 = cs.topk(2, dim=1).values; hit = cs.argmax(1) == tss
            s1["telemetry"] = {"write_norm_mean": float(nrm.mean()), "true_code_cos_mean": float(true.mean()), "nearest_code_acc": float(hit.float().mean()), "cos_margin_mean": float((true - torch.where(hit, top2[:, 1], top2[:, 0])).mean()), "euclid_to_true_mean": float((w - C[tss]).norm(dim=1).mean()), "per_state_spread": [float(1 - (w[ss == k] @ w[ss == k].T).mean()) for k in range(K)]}
        gap = s1["train_perm_acc"] - s1["heldout_acc"]; overall = float(np.mean([s1["heldout_acc"], s1["cf_follow"]]))
        core = (s1["termination_min_arm"] >= WG["termination_min"] and s1["heldout_acc"] >= WG["acc_min"] and s1["cf_follow"] >= WG["acc_min"] and s1["heldout_acc_lb"] > WG["acc_lb_min"] and min(min(v) for v in s1["groups"].values()) >= WG["per_group_min"] and s1["agreement"] >= WG["agreement_min"] and s1["paired_dir"] >= WG["paired_dir_min"] and s1["paired_dir_lb"] > WG["paired_dir_lb_min"] and gap <= WG["lookup_gap_max"])
        ctrls = (s1["shuffled_donor_follow"] >= WG["donor_follow_min"] and s1["shuffled_recipient_follow"] <= WG["recipient_follow_max"] and s1["shuffled_effect_lb"] > WG["shuffled_effect_lb_min"] and ctrl <= WG["control_max"] and s1["oracle_recovery"] >= WG["oracle_recovery_min"])
        if not wdone: cls = "INCOMPLETE — DEADLINE"
        elif not s1["zero_hook_identity"] or s1["zero_hook_logit_gap"] > 1e-4: cls = "INVALID — CONSUMER PRECONDITION"
        elif s1["train_perm_acc"] >= WG["lookup_train_min"] and (s1["heldout_acc"] < WG["lookup_heldout_max"] or gap > WG["lookup_gap_max"]): cls = "LOOKUP-BOUND FAIL — OUTPUT MAPPING"
        elif core and ctrls: cls = "PASS"
        elif s1["heldout_acc"] >= WG["acc_min"] and not ctrls: cls = "ENTITY/CONTEXT-BOUND FAIL — SOURCE NOT IDENTIFIED"
        else: cls = "FAIL — SOURCE WRITER CONSTRUCTION"
        s1["class"] = cls; s1["overall"] = overall; res["seeds"][seed] = {"summary": s1, "loss_history": whist[::25]}; save(); log(f"seed {seed} writer: " + json.dumps({k: v for k, v in s1.items() if k not in ("groups", "consumer_replay", "consumer_crossed", "telemetry", "termination_by_arm", "control_state_distributions")}, default=float) + f" groups={s1['groups']} telemetry={s1['telemetry']}"); log(f"seed {seed} class: {cls} ({time.time()-T0:.0f}s)")
    kinds = [v["summary"]["class"] for v in res["seeds"].values()]; done = [v["summary"] for v in res["seeds"].values() if v["summary"]["class"] != "INCOMPLETE — DEADLINE"]
    if stage == "rung0":
        status = RUNG0_STATUS if kinds.count("VALID") >= G["seeds_required"] and all(d["heldout_acc"] >= G["seed_floor"] for d in done) else ("LOOKUP-BOUND INVALID" if "LOOKUP-BOUND INVALID" in kinds else ("INVALID — MASK/HOOK" if "INVALID — MASK/HOOK" in kinds else "INVALID — ORACLE REGISTER CONSUMER"))
    else:
        floor = all(d["overall"] >= WG["seed_floor"] for d in done if "overall" in d)
        status = "PASS — FUNCTIONAL SOURCE WRITER AT ZERO CONFIGURED FILLER" if kinds.count("PASS") >= WG["seeds_required"] and floor else next((k for k in ("INVALID — CONSUMER PRECONDITION", "LOOKUP-BOUND FAIL — OUTPUT MAPPING", "ENTITY/CONTEXT-BOUND FAIL — SOURCE NOT IDENTIFIED") if k in kinds), "FAIL — SOURCE WRITER CONSTRUCTION")
    if len(done) < (G if stage == "rung0" else WG)["seeds_required"]: status = "INCOMPLETE — NO VERDICT"
    res["status"] = status; res["per_seed"] = kinds; res["seconds"] = time.time() - T0; save(); log(f"STATUS: {status}; per seed {kinds}")


if __name__ == "__main__":
    main()
