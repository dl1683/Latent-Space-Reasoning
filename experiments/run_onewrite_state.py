"""onewrite_state_v1: a state written ONCE into a frozen LM (Qwen3-1.7B-Base, CPU). Locked design: .codex_direction_r13.md.

E encodes z (16-d) from the block-12 residual at the final token of a neutral source anchor; J adds Jz once, at block 12,
to the final token of the early 'Internal record:' slot during the prefill of a NEW-WORDING target; the hook is disabled
before greedy continuation. Training: token cross-entropy on the correct decoded label for three trained consequence
families only. Heldout families H1 (XOR) and H2 (pairing) - labels and templates - never enter optimization.

    python experiments/run_onewrite_state.py --config experiments/config/onewrite_state_v1.json [--smoke]
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, os, re, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from substitution_probe import SubstitutionProbe


class Iface(nn.Module):
    def __init__(self, d, k):
        super().__init__(); self.ln = nn.LayerNorm(d, elementwise_affine=False); self.E = nn.Linear(d, k); self.J = nn.Linear(k, d, bias=False); nn.init.zeros_(self.J.weight)
    def enc(self, h): return self.E(self.ln(h))


class LM:
    def __init__(self, cfg):
        self.cfg = cfg; self.sp = SubstitutionProbe(cfg["model_id"], revision=cfg["revision"]); self.m = self.sp.model; self.tok = self.sp.tok
        assert self.sp.revision == cfg["revision"]; torch.set_grad_enabled(True)
        for p in self.m.parameters(): p.requires_grad_(False)
        self.layer = self.m.model.layers[cfg["layer"]]; self.layer.register_forward_hook(self._hook); self.delta = None; self.slot_pos = None; self.writes = 0
        self.ids = lambda t: self.tok.encode(t, add_special_tokens=False); self.slot_len = len(self.ids(cfg["slot"])); self.eos = {self.tok.eos_token_id, self.ids("\n")[0]}

    def _hook(self, mod, i, o):
        if self.delta is None: return o
        h = o[0] if isinstance(o, tuple) else o; h = h.clone(); p = self.slot_pos; d = self.delta
        scale = torch.clamp(self.cfg["norm_clamp_fraction"] * h[0, p].norm().detach() / (d.norm() + 1e-6), max=1.0)   # ||Jz|| <= 0.25 ||h_slot||
        h[0, p, :] = h[0, p, :] + scale * d; self.writes += 1; return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h

    @torch.no_grad()
    def source_state(self, text):
        out = self.m(input_ids=torch.tensor([self.ids(text + self.cfg["anchor"])]), output_hidden_states=True); return out.hidden_states[self.cfg["layer"] + 1][0, -1].clone()

    def prefill(self, prompt_ids, delta):
        """Single prefill with the one write (if delta is not None) at the slot; returns logits and past. Asserts exactly one write."""
        self.delta, self.slot_pos, self.writes = delta, self.slot_len - 1, 0
        try: out = self.m(input_ids=torch.tensor([prompt_ids]), use_cache=True)
        finally: self.delta = None
        assert self.writes == (0 if delta is None else 1), self.writes; return out

    def label_loss(self, prompt_ids, label, delta):
        lab = self.ids(label)
        self.delta, self.slot_pos, self.writes = delta, self.slot_len - 1, 0
        try: lg = self.m(input_ids=torch.tensor([prompt_ids + lab])).logits[0]
        finally: self.delta = None
        assert self.writes == 1; lp = torch.log_softmax(lg[len(prompt_ids) - 1:-1].float(), -1); return -lp.gather(1, torch.tensor(lab).unsqueeze(1)).mean()

    @torch.no_grad()
    def decode(self, prompt_ids, delta):
        out = self.prefill(prompt_ids, delta); past = out.past_key_values; nxt = int(out.logits[0, -1].argmax()); toks = []; ended = False
        for _ in range(self.cfg["decode"]["max_new_tokens"]):
            if nxt in self.eos: ended = True; break
            toks.append(nxt); o = self.m(input_ids=torch.tensor([[nxt]]), past_key_values=past, use_cache=True); past = o.past_key_values; nxt = int(o.logits[0, -1].argmax())
        return self.tok.decode(toks), ended


def build_world(cfg):
    A = cfg["attributes"]; states = list(itertools.product([0, 1], repeat=3)); facts = []
    for split, names, per in (("train", cfg["names_train"], 3), ("eval", cfg["names_eval"], 2)):
        for si, st in enumerate(states):
            for k in range(per): facts.append({"split": split, "name": names[si * per + k], "bits": st, "vals": [A[i][st[i]] for i in range(3)]})
    return facts


def fill(t, f, cfg, labels, rot):
    A = cfg["attributes"]; L = [labels[(i + rot) % len(labels)] for i in range(len(labels))]
    return t.format(name=f["name"], a1=f["vals"][0], a2=f["vals"][1], a3=f["vals"][2], v1a=A[0][0], v1b=A[0][1], v2a=A[1][0], v2b=A[1][1], v3a=A[2][0], v3b=A[2][1], x1=A[0][0], x2=A[1][0], L0=L[0], L1=L[1], L2=L[2] if len(L) > 2 else "", L3=L[3] if len(L) > 3 else ""), L


def answer(family, f, L):
    """Correct label under rotation L for a trained family (0..2) or heldout family ('H1'/'H2')."""
    b = f["bits"]
    if family == "H1": return L[0] if (b[0] == 0) != (b[1] == 0) else L[1]
    if family == "H2": return L[2 * b[1] + b[2]]
    return L[b[family]]


def parse(text, labels):
    words = re.findall(r"[A-Za-z]+", text.upper()); return next((w for w in words if w in labels), None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args(); T0 = time.time()
    cfg = json.load(open(a.config, encoding="utf-8")); out = f"experiments/results/{cfg['name']}"; os.makedirs(out, exist_ok=True); tag = "smoke" if a.smoke else "run"
    logf = open(os.path.join(out, f"{tag}.log"), "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    if a.smoke: cfg["train"]["steps"] = int(os.environ.get("SMOKE_STEPS", 40)); cfg["train"]["seeds"] = cfg["train"]["seeds"][:1]; cfg["eval"]["bootstraps"] = 200
    shas = {k: hashlib.sha256(open(v, "rb").read()).hexdigest() for k, v in (("runner", __file__), ("config", a.config))}
    M = LM(cfg); facts = build_world(cfg); train = [f for f in facts if f["split"] == "train"]; ev = [f for f in facts if f["split"] == "eval"]; G = cfg["gates"]; H = cfg["heldout"]
    log(f"loaded; slot_len={M.slot_len}; train facts {len(train)}; eval facts {len(ev)} ({time.time()-T0:.0f}s)")
    for f in facts:                                                                            # cached source states (frozen model => constants)
        tpl = cfg["source_templates_train"] if f["split"] == "train" else cfg["source_templates_eval"]; f["src"] = [M.source_state(fill(t, f, cfg, ["", ""], 0)[0]) for t in tpl]
    res = {"config": cfg["name"], "sha256": shas, "revision": M.sp.revision, "seeds": {}}; save = lambda: json.dump(res, open(os.path.join(out, f"{tag}_result.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else float(o))
    deadline = T0 + cfg["train"]["hard_wall_minutes"] * 60
    for seed in cfg["train"]["seeds"]:
        torch.manual_seed(seed); rng = np.random.default_rng(seed); iface = nn.ModuleDict({"i": Iface(M.m.config.hidden_size, cfg["state_dim"])})["i"]; opt = torch.optim.AdamW(iface.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
        n_par = sum(p.numel() for p in iface.parameters()); t0 = time.time(); hist = []
        for step in range(cfg["train"]["steps"]):
            f = train[rng.integers(len(train))]; fam = int(rng.integers(3)); tpl = cfg["target_templates_train"][fam]; rot = int(rng.integers(2)); src = f["src"][rng.integers(len(f["src"]))]
            prompt, L = fill(tpl, f, cfg, cfg["labels_train"][fam], rot); z = iface.enc(src); loss = M.label_loss(M.ids(prompt), " " + answer(fam, f, L), iface.J(z))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(iface.parameters(), cfg["train"]["clip"]); opt.step(); hist.append(float(loss.detach()))
            if step % 50 == 0 or step == cfg["train"]["steps"] - 1: log(f"seed {seed} step {step}: loss={float(loss):.3f} ({time.time()-t0:.0f}s)")
            if time.time() - t0 > cfg["train"]["seed_wall_minutes"] * 60 * 0.6: log("seed training wall reached"); break
        iface.eval(); torch.save(iface.state_dict(), os.path.join(out, f"iface_seed{seed}.pt"))
        with torch.no_grad():
            ztr = torch.stack([iface.enc(s) for f in train for s in f["src"]]); zr = torch.randn(cfg["state_dim"], generator=torch.Generator().manual_seed(999 + seed)); zr = ztr.mean(0) + (zr - zr.mean()) / zr.std() * ztr.std(0).mean()
            rows = []
            # heldout families: 16 facts x {H1, H2} x 2 unseen wordings x 5 arms
            for fi, f in enumerate(ev):
                z = iface.enc(f["src"][0]); dz = iface.J(z); dr = iface.J(zr)
                for fam in ("H1", "H2"):
                    labs = H[fam]["labels"]
                    for ti, tpl in enumerate(H[fam]["templates"]):
                        rot = (fi + ti) % len(labs); prompt, L = fill(tpl, f, cfg, labs, rot); ids = M.ids(prompt); correct = answer(fam, f, L)
                        donors = [g for g in ev if answer(fam, g, L) != correct]; d = donors[(fi + ti) % len(donors)]; dd = iface.J(iface.enc(d["src"][0]))
                        vis_prompt = prompt.replace("Internal record:", "Internal record: " + fill(cfg["source_templates_eval"][1], f, cfg, ["", ""], 0)[0], 1)
                        arms = {"write": (ids, dz), "cue": (ids, None), "wrong": (ids, dd), "random": (ids, dr), "visible": (M.ids(vis_prompt), None)}
                        row = {"fact": fi, "family": fam, "template": ti, "correct": correct, "donor_correct": answer(fam, d, L)}
                        for name, (pid, delta) in arms.items():
                            txt, ended = M.decode(pid, delta); row[name] = {"text": txt, "choice": parse(txt, labs), "ended": ended}
                        rows.append(row)
                if fi % 4 == 0: log(f"seed {seed} eval fact {fi}: " + " | ".join(f"{k}:{rows[-1][k]['choice']}" for k in ("write", "cue", "wrong", "random", "visible")) + f" correct={rows[-1]['correct']} ({time.time()-T0:.0f}s)")
            # trained families on heldout facts (controller status): write vs cue
            trows = []
            for fi, f in enumerate(ev):
                dz = iface.J(iface.enc(f["src"][0]))
                for fam in range(3):
                    rot = fi % 2; prompt, L = fill(cfg["target_templates_train"][fam], f, cfg, cfg["labels_train"][fam], rot); ids = M.ids(prompt); correct = answer(fam, f, L)
                    trows.append({"fact": fi, "family": fam, "correct": correct, **{k: parse(M.decode(ids, dl)[0], cfg["labels_train"][fam]) for k, dl in (("write", dz), ("cue", None))}})
        # --- gates (facts are the units) ---
        acc = lambda arm, rs=rows: float(np.mean([r[arm]["choice"] == r["correct"] for r in rs])); term = lambda arm: float(np.mean([r[arm]["ended"] for r in rows]))
        by_fact = lambda arm: np.array([np.mean([r[arm]["choice"] == r["correct"] for r in rows if r["fact"] == fi]) for fi in range(len(ev))])
        diff = by_fact("write") - by_fact("cue"); lb = float(np.quantile([np.mean(rng.choice(diff, len(diff))) for _ in range(cfg["eval"]["bootstraps"])], 0.025))
        sf = float(np.mean([np.mean(diff * rng.choice([-1, 1], len(diff))) >= diff.mean() for _ in range(cfg["eval"]["randomizations"])]))
        fam_acc = {fam: acc("write", [r for r in rows if r["family"] == fam]) for fam in ("H1", "H2")}; tpl_acc = [acc("write", [r for r in rows if r["template"] == t]) for t in (0, 1)]
        wrong_follow = float(np.mean([r["wrong"]["choice"] == r["donor_correct"] for r in rows])); wrong_base = float(np.mean([r["cue"]["choice"] == r["donor_correct"] for r in rows]))
        recov = (acc("write") - acc("cue")) / max(acc("visible") - acc("cue"), 1e-6)
        s = {"params": n_par, "loss_history": hist, "acc": {k: acc(k) for k in ("write", "cue", "wrong", "random", "visible")}, "termination": {k: term(k) for k in ("write", "visible")}, "family_acc": fam_acc, "template_acc": tpl_acc,
             "write_minus_cue": float(diff.mean()), "write_minus_cue_lb": lb, "signflip_p": sf, "write_minus_random": acc("write") - acc("random"), "recovery": recov, "wrong_follows_donor": wrong_follow, "wrong_baseline": wrong_base,
             "trained_write": acc("write", [dict(r, write={"choice": r["write"]}) for r in trows]), "trained_cue": acc("cue", [dict(r, cue={"choice": r["cue"]}) for r in trows])}
        gates = {"instrument": s["acc"]["visible"] >= G["visible_min"] and s["termination"]["write"] >= G["termination_min"] and s["termination"]["visible"] >= G["termination_min"] and s["acc"]["cue"] <= G["cue_max"],
                 "transfer": s["acc"]["write"] >= G["write_min"] and all(v >= G["write_family_min"] for v in fam_acc.values()) and abs(tpl_acc[0] - tpl_acc[1]) <= G["wording_diff_max"] and diff.mean() >= G["write_minus_cue_min"] and lb > G["write_minus_cue_lb_min"] and s["write_minus_random"] >= G["write_minus_random_min"] and recov >= G["recovery_min"],
                 "specificity": wrong_follow >= G["wrong_follows_donor_min"] and wrong_follow - wrong_base >= G["wrong_over_baseline_min"],
                 "controller": s["trained_write"] >= G["controller_trained_min"] and s["trained_write"] - s["trained_cue"] >= G["controller_uplift_min"]}
        s["gates"] = gates; s["class"] = "POSITIVE" if all(gates[k] for k in ("instrument", "transfer", "specificity")) else ("CONTROLLER" if gates["controller"] else "FAIL")
        res["seeds"][seed] = {"summary": s, "rows": rows, "trained_rows": trows}; save(); log(f"seed {seed}: " + json.dumps({k: v for k, v in s.items() if k not in ("loss_history",)}, default=float)); log(f"seed {seed} class: {s['class']} ({time.time()-T0:.0f}s)")
        if time.time() > deadline: log("hard wall: stopping seeds"); break
    kinds = [v["summary"]["class"] for v in res["seeds"].values()]; n = len(kinds)
    if n < G["seeds_required"]: status = "INCOMPLETE — NO VERDICT"
    elif kinds.count("POSITIVE") >= G["seeds_required"]: status = "BOUNDED POSITIVE — ONE-WRITE PERSISTENT CAUSAL STATE (this construction)"
    elif kinds.count("POSITIVE") + kinds.count("CONTROLLER") >= G["seeds_required"]: status = "SUPERVISED CONTROLLER"
    else: status = "FAIL — ONE-WRITE STATE CONSTRUCTION"
    res["status"] = status; res["per_seed"] = kinds; res["seconds"] = time.time() - T0; save(); log(f"STATUS: {status}; per seed {kinds}")


if __name__ == "__main__":
    main()
