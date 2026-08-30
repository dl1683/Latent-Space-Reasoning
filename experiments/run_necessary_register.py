"""necessary_register_v1 rung 0 (Codex rounds 24-25): INSTRUMENT CHECK of a from-scratch hard-masked register transformer.
Eight fixed orthonormal oracle codes REPLACE the <REG> input embedding; each episode shows a visible permutation pi of the
eight abstract states onto a panel's eight labels; train/eval permutations are disjoint. Mask: positions before the register
are causal; the register attends everything through itself; every later position attends only the register and later
positions (the register is the only path from the write boundary to the answer). Readout: inferred state = pi^-1(label).

    python experiments/run_necessary_register.py --config experiments/config/necessary_register_v1.json [--smoke]
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

PAD, BOS, REG, MAP, ANS, EOS = range(6); E0, Q0, L0 = 6, 30, 34                      # entities 6-29, query tokens 30-33, labels 34-65 (4 panels x 8)
TEMPLATES = [lambda e: [Q0, e], lambda e: [e, Q0 + 1], lambda e: [Q0 + 2, e, Q0 + 2], lambda e: [Q0 + 3, Q0 + 3, e]]
VOCAB = L0 + 32; R = 1                                                                  # register position


class RegisterLM(nn.Module):
    def __init__(self, mc, T):
        super().__init__(); d = mc["d_model"]; self.emb = nn.Embedding(VOCAB, d); self.pos = nn.Embedding(T, d)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, mc["n_heads"], mc["d_ff"], mc["dropout"], batch_first=True, norm_first=True) for _ in range(mc["n_layers"])])
        self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, VOCAB); self.register_buffer("mask", self.build_mask(T))
    @staticmethod
    def build_mask(T):
        m = torch.full((T, T), float("-inf")); i = torch.arange(T)
        m[i[:, None] >= i[None, :]] = 0.0                                                 # causal
        m[R + 1:, :R] = float("-inf")                                                     # post-register positions cannot see pre-register source positions
        return m
    def forward(self, ids, code=None):
        x = self.emb(ids) + self.pos(torch.arange(ids.shape[1]))[None]
        if code is not None: x = x.clone(); x[:, R, :] = code                              # oracle write REPLACES the register embedding
        for l in self.layers: x = l(x, src_mask=self.mask[: ids.shape[1], : ids.shape[1]])
        return self.head(self.norm(x))


def episodes(W, rng, B, perms, states=None, ents=None, tpls=None, panels=None, pidx=None):
    """Batch of episodes: <BOS> <REG> <MAP> labels(pi(0..7)) query <ANS> label(pi(s)) <EOS>. Returns ids, targets, meta."""
    K = W["n_states"]; ents = rng.integers(W["n_entities"], size=B) if ents is None else ents; states = rng.integers(K, size=B) if states is None else states
    tpls = rng.integers(W["n_templates"], size=B) if tpls is None else tpls; panels = rng.integers(W["n_panels"], size=B) if panels is None else panels; pidx = rng.integers(len(perms[0]), size=B) if pidx is None else pidx
    seqs, tgt, meta = [], [], []
    for b in range(B):
        pi = perms[panels[b]][pidx[b]]; labels = [L0 + 8 * panels[b] + int(pi[s]) for s in range(K)]
        q = TEMPLATES[tpls[b]](E0 + int(ents[b])); seq = [BOS, REG, MAP] + labels + q + [ANS]; seqs.append(seq); tgt.append([labels[states[b]], EOS]); meta.append((int(ents[b]), int(states[b]), int(tpls[b]), int(panels[b]), int(pidx[b]), pi))
    T = max(map(len, seqs)); ids = torch.full((B, T + 2), PAD, dtype=torch.long)
    for b, s in enumerate(seqs): ids[b, : len(s)] = torch.tensor(s); ids[b, len(s): len(s) + 2] = torch.tensor(tgt[b])
    ans_pos = torch.tensor([len(s) - 1 for s in seqs]); return ids, ans_pos, meta


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args(); T0 = time.time()
    cfg = json.load(open(a.config, encoding="utf-8")); out = f"experiments/results/{cfg['name']}"; os.makedirs(out, exist_ok=True); tag = "smoke" if a.smoke else "run"; logf = open(os.path.join(out, f"{tag}.log"), "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    if a.smoke: cfg["train"]["steps"] = int(os.environ.get("SMOKE_STEPS", 200)); cfg["train"]["seeds"] = cfg["train"]["seeds"][:1]; cfg["eval"]["bootstraps"] = 100
    shas = {k: hashlib.sha256(open(v, "rb").read()).hexdigest() for k, v in (("runner", __file__), ("config", a.config))}
    W, mc, tc, G = cfg["world"], cfg["model"], cfg["train"], cfg["gates"]; K = W["n_states"]
    prng = np.random.default_rng(W["perm_seed"]); allp = {}
    for p in range(W["n_panels"]):
        seen, bank = set(), []
        while len(bank) < W["train_perms_per_panel"] + W["eval_perms_per_panel"]:
            q = tuple(prng.permutation(K))
            if q not in seen: seen.add(q); bank.append(np.array(q))
        allp[p] = bank
    train_perms = [allp[p][: W["train_perms_per_panel"]] for p in range(W["n_panels"])]; eval_perms = [allp[p][W["train_perms_per_panel"]:] for p in range(W["n_panels"])]
    g = torch.Generator().manual_seed(mc["code_seed"]); C = torch.linalg.qr(torch.randn(mc["d_model"], K, generator=g))[0].T.contiguous()          # 8 orthonormal codes in R^128
    Rg = torch.randn(cfg["eval"]["n_random"], mc["d_model"], generator=g); Rg = Rg / Rg.norm(dim=1, keepdim=True)                                   # fixed norm-matched random codes
    res = {"config": cfg["name"], "sha256": shas, "codebook_hash": hashlib.sha256(C.numpy().tobytes()).hexdigest(), "perm_hash": hashlib.sha256(json.dumps([[q.tolist() for q in allp[p]] for p in allp]).encode()).hexdigest(), "seeds": {}}
    save = lambda: json.dump(res, open(os.path.join(out, f"{tag}_result.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else float(o))
    Tmax = 3 + K + 3 + 1 + 2; log(f"vocab {VOCAB}; codes {res['codebook_hash'][:10]}; perms {res['perm_hash'][:10]}; Tmax {Tmax}")
    deadline = T0 + tc["hard_wall_minutes"] * 60
    for seed in tc["seeds"]:
        torch.manual_seed(seed); rng = np.random.default_rng(seed); model = RegisterLM(mc, Tmax); opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"]); t0 = time.time(); hist = []
        for step in range(tc["steps"]):
            ids, ans, meta = episodes(W, rng, tc["batch"], train_perms); code = C[torch.tensor([m[1] for m in meta])]
            lg = model(ids[:, :-1], code); tgt = ids[:, 1:]; b = torch.arange(len(ids))
            loss = F.cross_entropy(lg[b, ans], tgt[b, ans]) + F.cross_entropy(lg[b, ans + 1], tgt[b, ans + 1])                      # answer label + EOS only
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), tc["clip"]); opt.step(); hist.append(float(loss.detach()))
            if step % 250 == 0 or step == tc["steps"] - 1: log(f"seed {seed} step {step}: loss={float(loss):.3f} ({time.time()-t0:.0f}s)")
            if time.time() > deadline: log("hard wall"); break
        model.eval()
        # ---- evaluation: balanced held-out-permutation episodes; arms = own / 7 counterfactual / zero / zero-hook / random ----
        def run(perms, n):
            ents = np.repeat(np.arange(W["n_entities"]), n // W["n_entities"])[:n]; states = np.tile(np.arange(K), n // K)[:n]; tpls = np.tile(np.arange(W["n_templates"]), n // W["n_templates"])[:n]; panels = np.tile(np.repeat(np.arange(W["n_panels"]), 4), n // 16 + 1)[:n]
            pidx = rng.integers(len(perms[0]), size=n); ids, ans, meta = episodes(W, rng, n, perms, states, ents, tpls, panels, pidx); rows = []
            def decode(code):
                with torch.no_grad(): lg = model(ids[:, :-1], code); b = torch.arange(n); first = lg[b, ans].argmax(-1); second = lg[b, ans + 1].argmax(-1)
                return first.numpy(), (second == EOS).numpy()
            arms = {"own": C[torch.tensor(states)]} | {f"cf{j}": C[torch.tensor((states + j) % K)] for j in range(1, K)} | {"zero": torch.zeros(n, mc["d_model"])} | {f"rand{r}": Rg[r].expand(n, -1) for r in range(cfg["eval"]["n_random"])}
            outs = {k: decode(v) for k, v in arms.items()}
            with torch.no_grad():                                                                                                         # zero-hook identity: the replacement path fed the learned <REG> input (token + position embedding) must equal the untouched path row-for-row
                zh = model(ids[:, :-1], (model.emb.weight[REG] + model.pos.weight[R]).detach().expand(n, -1))[torch.arange(n), ans].argmax(-1).numpy(); zu = model(ids[:, :-1], None)[torch.arange(n), ans].argmax(-1).numpy()
            for i, (e, s, t, p, pi_i, pi) in enumerate(meta):
                inv = {L0 + 8 * p + int(pi[st]): st for st in range(K)}; row = {"e": e, "s": s, "t": t, "p": p, "pi": pi_i}
                for k, (first, term) in outs.items(): row[k] = {"state": inv.get(int(first[i]), -1), "term": bool(term[i])}
                row["zero_hook_state"] = inv.get(int(zh[i]), -1); row["untouched_state"] = inv.get(int(zu[i]), -1); rows.append(row)
            return rows
        rows = run(eval_perms, cfg["eval"]["episodes_per_arm"]); trows = run(train_perms, 768)
        acc = lambda rs, k, want: float(np.mean([r[k]["state"] == (r["s"] if want == "s" else (r["s"] + int(k[2:])) % K if k.startswith("cf") else r["s"]) for r in rs]))
        own = np.array([r["own"]["state"] == r["s"] for r in rows], float); by_e = lambda v: np.array([v[[r["e"] == e for r in rows]].mean() for e in range(W["n_entities"])])
        boot = lambda pe: float(np.quantile([np.mean(rng.choice(pe, len(pe))) for _ in range(cfg["eval"]["bootstraps"])], 0.025))
        cf_ok = np.array([np.mean([r[f"cf{j}"]["state"] == (r["s"] + j) % K for j in range(1, K)]) for r in rows]); paired = own * cf_ok
        groups = {"state": [acc([r for r in rows if r["s"] == s], "own", "s") for s in range(K)], "panel": [acc([r for r in rows if r["p"] == p], "own", "s") for p in range(W["n_panels"])], "template": [acc([r for r in rows if r["t"] == t], "own", "s") for t in range(W["n_templates"])]}
        keyf = lambda r: (r["s"], r["e"], r["t"]); agree = []
        for key in {keyf(r) for r in rows}:
            st = [r["own"]["state"] for r in rows if keyf(r) == key]
            if len(st) > 1: agree.append(np.mean([st[i] == st[j] for i in range(len(st)) for j in range(i + 1, len(st))]))
        zero = float(np.mean([r["zero"]["state"] == r["s"] for r in rows])); rnd = float(np.mean([r[f"rand{k}"]["state"] == r["s"] for r in rows for k in range(cfg["eval"]["n_random"])])); ctrl = max(zero, rnd)
        term = float(np.mean([r["own"]["term"] for r in rows])); train_acc = float(np.mean([r["own"]["state"] == r["s"] for r in trows])); zh_ok = all(r["zero_hook_state"] == r["untouched_state"] for r in rows)
        s = {"loss_first": hist[0], "loss_last": float(np.mean(hist[-50:])), "heldout_acc": float(own.mean()), "heldout_acc_lb": boot(by_e(own)), "train_perm_acc": train_acc, "groups": groups, "agreement": float(np.mean(agree)), "cf_follow": float(cf_ok.mean()),
             "paired_dir": float(paired.mean()), "paired_dir_lb": boot(by_e(paired)), "zero_assigned": zero, "random_assigned": rnd, "uplift": float(own.mean() - ctrl), "uplift_lb": boot(by_e(own) - ctrl), "termination": term, "zero_hook_identity": zh_ok, "n": len(rows)}
        mask_ok = float(model.mask[R + 1, 0]) == float("-inf") and float(model.mask[R, 0]) == 0.0
        if not (mask_ok and zh_ok): cls = "INVALID — MASK/HOOK"
        elif train_acc >= G["lookup_train_min"] and (s["heldout_acc"] < G["lookup_heldout_max"] or train_acc - s["heldout_acc"] > G["lookup_gap_max"]): cls = "LOOKUP-BOUND INVALID"
        elif (term >= G["termination_min"] and s["heldout_acc"] >= G["heldout_acc_min"] and s["heldout_acc_lb"] > G["heldout_acc_lb_min"] and min(min(v) for v in groups.values()) >= G["per_group_min"] and s["agreement"] >= G["agreement_min"] and s["paired_dir"] >= G["paired_dir_min"] and s["paired_dir_lb"] > G["paired_dir_lb_min"]
              and s["uplift"] >= G["uplift_min"] and s["uplift_lb"] > G["uplift_lb_min"] and zero <= G["control_max"] and rnd <= G["control_max"]): cls = "VALID"
        else: cls = "INVALID — ORACLE REGISTER CONSUMER"
        s["class"] = cls; res["seeds"][seed] = {"summary": s, "loss_history": hist[::25]}; save(); log(f"seed {seed}: " + json.dumps({k: v for k, v in s.items() if k != "groups"}, default=float) + f" groups={groups}"); log(f"seed {seed} class: {cls} ({time.time()-T0:.0f}s)")
    kinds = [v["summary"]["class"] for v in res["seeds"].values()]; ok = [v["summary"]["heldout_acc"] >= G["seed_floor"] for v in res["seeds"].values()]
    status = "INSTRUMENT VALID — COMPOSITIONAL ORACLE REGISTER CONSUMER" if kinds.count("VALID") >= G["seeds_required"] and all(ok) else ("LOOKUP-BOUND INVALID" if "LOOKUP-BOUND INVALID" in kinds else ("INVALID — MASK/HOOK" if "INVALID — MASK/HOOK" in kinds else "INVALID — ORACLE REGISTER CONSUMER"))
    if len(kinds) < G["seeds_required"]: status = "INCOMPLETE — NO VERDICT"
    res["status"] = status; res["per_seed"] = kinds; res["seconds"] = time.time() - T0; save(); log(f"STATUS: {status}; per seed {kinds}")


if __name__ == "__main__":
    main()
