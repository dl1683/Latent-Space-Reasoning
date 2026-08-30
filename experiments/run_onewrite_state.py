"""onewrite_state_v1: a state written ONCE into a frozen LM (Qwen3-1.7B-Base, CPU). Locked design: .codex_direction_r13.md,
instrument repair and pre-lock validation: .codex_direction_r14.md (the ONLY instrument repair; fail => kill pre-lock).

E encodes z (16-d) from the block-12 residual at the final token of a neutral source anchor; J adds Jz once, at block 12,
to the final token of the 'Internal record:' slot during the prefill of a NEW-WORDING target; the hook is disabled before
greedy continuation. Training: token cross-entropy on the correct label for three trained families (TAGS: STORED).
Heldout families H1 (XOR) and H2 (pairing) - labels, tables and wordings - never enter optimization. Strict protocol:
the FIRST lexical item of the decode must be an allowed label; everything else counts as incorrect.

    python experiments/run_onewrite_state.py --config experiments/config/onewrite_state_v1.json --stage validate|train [--smoke]
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, os, re, sys, time
import numpy as np, torch, torch.nn as nn
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
        self.layer = self.m.model.layers[cfg["layer"]]; self.layer.register_forward_hook(self._hook); self.delta = None; self.writes = 0
        self.ids = lambda t: self.tok.encode(t, add_special_tokens=False); self.slot_pos = len(self.ids(cfg["slot"])) - 1; self.nl = self.ids("\n")[0]

    def _hook(self, mod, i, o):
        if self.delta is None: return o
        h = o[0] if isinstance(o, tuple) else o; h = h.clone(); p = self.slot_pos; d = self.delta
        scale = torch.clamp(self.cfg["norm_clamp_fraction"] * h[0, p].norm().detach() / (d.norm() + 1e-6), max=1.0)   # ||Jz|| <= 0.25 ||h_slot||
        h[0, p, :] = h[0, p, :] + scale * d; self.writes += 1; return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h

    @torch.no_grad()
    def source_state(self, text):
        out = self.m(input_ids=torch.tensor([self.ids(text + self.cfg["anchor"])]), output_hidden_states=True); return out.hidden_states[self.cfg["layer"] + 1][0, -1].clone()

    def _forward(self, ids, delta, cache):
        self.delta, self.writes = delta, 0
        try: out = self.m(input_ids=torch.tensor([ids]), use_cache=cache)
        finally: self.delta = None
        assert self.writes == (0 if delta is None else 1), self.writes; return out

    def label_loss(self, prompt_ids, label, delta):
        lab = self.ids(label); lg = self._forward(prompt_ids + lab, delta, False).logits[0]
        lp = torch.log_softmax(lg[len(prompt_ids) - 1:-1].float(), -1); return -lp.gather(1, torch.tensor(lab).unsqueeze(1)).mean()

    @torch.no_grad()
    def decode(self, prompt_ids, delta, labels):
        """Greedy, at most 6 tokens; stop on newline/EOS or once the first lexical item completes an allowed label."""
        out = self._forward(prompt_ids, delta, True); past = out.past_key_values; nxt = int(out.logits[0, -1].argmax()); toks = []
        for _ in range(self.cfg["decode"]["max_new_tokens"]):
            if nxt in (self.tok.eos_token_id, self.nl): break
            toks.append(nxt); text = self.tok.decode(toks); first = re.findall(r"[A-Za-z]+", text)
            if first and first[0].upper() in labels and (len(first) > 1 or not text.rstrip().endswith(first[0]) or len(toks) >= 2): break
            o = self.m(input_ids=torch.tensor([[nxt]]), past_key_values=past, use_cache=True); past = o.past_key_values; nxt = int(o.logits[0, -1].argmax())
        return self.tok.decode(toks)


def build_world(cfg):
    A = cfg["attributes"]; states = list(itertools.product([0, 1], repeat=3)); facts = []
    for split, names, per in (("train", cfg["names_train"], 3), ("eval", cfg["names_eval"], 2)):
        for si, st in enumerate(states):
            for k in range(per): facts.append({"split": split, "name": names[si * per + k], "bits": st, "vals": [A[i][st[i]] for i in range(3)]})
    return facts


def rotate(labels, rot): return [labels[(i + rot) % len(labels)] for i in range(len(labels))]


def prompt(cfg, f, family, wording, rot, tags):
    """family: 0..2 trained or 'H1'/'H2'; wording: 'train', 'A' or 'B'; tags: 'STORED' or the visible three tags."""
    P = cfg["prompt"]; labels = cfg["labels_train"][family] if isinstance(family, int) else cfg["heldout"][family]["labels"]; L = rotate(labels, rot)
    table = P["tables"][f"T{family + 1}" if isinstance(family, int) else family].format(L0=L[0], L1=L[1], L2=L[2] if len(L) > 2 else "", L3=L[3] if len(L) > 3 else "")
    body = P["wordings"][wording].format(name=f["name"], tags=tags, table=table, options=" | ".join(L))
    return P["demo"] + body, L, labels


def answer(family, f, L):
    b = f["bits"]
    if family == "H1": return L[0] if (b[0] == 0) != (b[1] == 0) else L[1]      # exactly one of marn (bit0==0) and vep (bit1==0)
    if family == "H2": return L[2 * b[1] + b[2]]
    return L[b[family]]


def strict(text, labels):
    """Protocol: the FIRST lexical item must be an allowed label. Returns (choice or None, completed)."""
    w = re.findall(r"[A-Za-z]+", text); ok = bool(w) and w[0].upper() in labels; return (w[0].upper() if ok else None), ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--stage", default="validate"); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args(); T0 = time.time()
    cfg = json.load(open(a.config, encoding="utf-8")); out = f"experiments/results/{cfg['name']}"; os.makedirs(out, exist_ok=True); tag = a.stage + ("_smoke" if a.smoke else "")
    logf = open(os.path.join(out, f"{tag}.log"), "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    if a.smoke: cfg["train"]["steps"] = int(os.environ.get("SMOKE_STEPS", 40)); cfg["train"]["seeds"] = cfg["train"]["seeds"][:1]; cfg["eval"]["bootstraps"] = 200
    shas = {k: hashlib.sha256(open(v, "rb").read()).hexdigest() for k, v in (("runner", __file__), ("config", a.config))}
    M = LM(cfg); facts = build_world(cfg); train = [f for f in facts if f["split"] == "train"]; ev = [f for f in facts if f["split"] == "eval"]; G = cfg["gates"]; V = cfg["validation_gate"]
    vis = lambda f: " | ".join(f["vals"]); log(f"loaded; slot_pos={M.slot_pos}; train facts {len(train)}; eval facts {len(ev)} ({time.time()-T0:.0f}s)")
    res = {"config": cfg["name"], "sha256": shas, "revision": M.sp.revision}; save = lambda: json.dump(res, open(os.path.join(out, f"{tag}_result.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else float(o))
    # ---------------- sole pre-lock validation (no state, no Iface): visible vs cue on all 64 heldout cases ----------------
    if a.stage == "validate":
        rows = []
        for fi, f in enumerate(ev):
            for fam in ("H1", "H2"):
                for wi, w in enumerate(("A", "B")):
                    rot = (fi + wi) % len(cfg["heldout"][fam]["labels"]); row = {"fact": fi, "family": fam, "wording": w}
                    for arm, tags in (("visible", vis(f)), ("cue", "STORED")):
                        p, L, labels = prompt(cfg, f, fam, w, rot, tags); txt = M.decode(M.ids(p), None, labels); c, ok = strict(txt, labels); row[arm] = {"text": txt, "choice": c, "completed": ok, "correct": c == answer(fam, f, L)}
                    rows.append(row)
            log(f"fact {fi}: " + " | ".join(f"{r['family']}{r['wording']} vis={r['visible']['text'].strip()[:12]!r}{'+' if r['visible']['correct'] else '-'} cue={r['cue']['text'].strip()[:12]!r}" for r in rows[-4:]))
        acc = lambda arm, rs=rows: float(np.mean([r[arm]["correct"] for r in rs])); comp = lambda arm: float(np.mean([r[arm]["completed"] for r in rows]))
        s = {"visible_acc": acc("visible"), "cue_acc": acc("cue"), "visible_completion": comp("visible"), "cue_completion": comp("cue"),
             "visible_by_family": {fam: acc("visible", [r for r in rows if r["family"] == fam]) for fam in ("H1", "H2")}, "visible_by_wording": {w: acc("visible", [r for r in rows if r["wording"] == w]) for w in ("A", "B")}}
        passed = (s["visible_acc"] >= V["visible_min"] and all(v >= V["visible_sub_min"] for v in list(s["visible_by_family"].values()) + list(s["visible_by_wording"].values())) and s["visible_completion"] >= V["completion_min"]
                  and s["cue_completion"] >= V["completion_min"] and s["cue_acc"] <= V["cue_max"] and s["visible_acc"] - s["cue_acc"] >= V["visible_minus_cue_min"])
        res["validation"] = {"summary": s, "passed": passed, "rows": rows}; save(); log(json.dumps(s, indent=1)); log(f"PRE-LOCK VALIDATION: {'PASS' if passed else 'FAIL - KILL PRE-LOCK'} ({time.time()-T0:.0f}s)"); return
    # ---------------- locked run ----------------
    for f in facts:
        tpl = cfg["source_templates_train"] if f["split"] == "train" else cfg["source_templates_eval"]; f["src"] = [M.source_state(t.format(name=f["name"], a1=f["vals"][0], a2=f["vals"][1], a3=f["vals"][2])) for t in tpl]
    res["seeds"] = {}; deadline = T0 + cfg["train"]["hard_wall_minutes"] * 60
    for seed in cfg["train"]["seeds"]:
        torch.manual_seed(seed); rng = np.random.default_rng(seed); iface = Iface(M.m.config.hidden_size, cfg["state_dim"]); opt = torch.optim.AdamW(iface.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
        n_par = sum(p.numel() for p in iface.parameters()); t0 = time.time(); hist = []
        for step in range(cfg["train"]["steps"]):
            f = train[rng.integers(len(train))]; fam = int(rng.integers(3)); rot = int(rng.integers(2)); src = f["src"][rng.integers(len(f["src"]))]
            p, L, _ = prompt(cfg, f, fam, "train", rot, "STORED"); loss = M.label_loss(M.ids(p), " " + answer(fam, f, L), iface.J(iface.enc(src)))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(iface.parameters(), cfg["train"]["clip"]); opt.step(); hist.append(float(loss.detach()))
            if step % 50 == 0 or step == cfg["train"]["steps"] - 1: log(f"seed {seed} step {step}: loss={float(loss):.3f} ({time.time()-t0:.0f}s)")
            if time.time() - t0 > cfg["train"]["seed_wall_minutes"] * 60 * 0.6: log("seed training wall reached"); break
        iface.eval(); torch.save(iface.state_dict(), os.path.join(out, f"iface_seed{seed}.pt"))
        with torch.no_grad():
            ztr = torch.stack([iface.enc(s) for f in train for s in f["src"]]); g = torch.Generator().manual_seed(999 + seed); zr = torch.randn(cfg["state_dim"], generator=g); zr = ztr.mean(0) + (zr - zr.mean()) / zr.std() * ztr.std(0).mean()
            rows = []
            for fi, f in enumerate(ev):
                dz = iface.J(iface.enc(f["src"][0])); dr = iface.J(zr)
                for fam in ("H1", "H2"):
                    for wi, w in enumerate(("A", "B")):
                        rot = (fi + wi) % len(cfg["heldout"][fam]["labels"]); p, L, labels = prompt(cfg, f, fam, w, rot, "STORED"); pv, _, _ = prompt(cfg, f, fam, w, rot, vis(f)); correct = answer(fam, f, L)
                        donors = [d for d in ev if answer(fam, d, L) != correct]; d = donors[(fi + wi) % len(donors)]; dd = iface.J(iface.enc(d["src"][0]))
                        arms = {"write": (p, dz), "cue": (p, None), "zero_hooked": (p, torch.zeros_like(dz)), "wrong": (p, dd), "random": (p, dr), "visible": (pv, None)}
                        row = {"fact": fi, "family": fam, "wording": w, "correct": correct, "donor_correct": answer(fam, d, L)}
                        for name, (pp, delta) in arms.items():
                            txt = M.decode(M.ids(pp), delta, labels); c, ok = strict(txt, labels); row[name] = {"text": txt, "choice": c, "completed": ok, "correct": c == correct}
                        rows.append(row)
                if fi % 4 == 0: log(f"seed {seed} fact {fi}: " + " ".join(f"{k}={rows[-1][k]['choice']}" for k in ("write", "cue", "wrong", "random", "visible")) + f" correct={rows[-1]['correct']} ({time.time()-T0:.0f}s)")
            trows = []
            for fi, f in enumerate(ev):
                dz = iface.J(iface.enc(f["src"][0]))
                for fam in range(3):
                    p, L, labels = prompt(cfg, f, fam, "train", fi % 2, "STORED"); correct = answer(fam, f, L)
                    trows.append({"fact": fi, "family": fam, "correct": correct, **{k: strict(M.decode(M.ids(p), dl, labels), labels)[0] == correct for k, dl in (("write", dz), ("cue", None))}})
        acc = lambda arm, rs=rows: float(np.mean([r[arm]["correct"] for r in rs])); comp = lambda arm: float(np.mean([r[arm]["completed"] for r in rows]))
        by_fact = lambda arm: np.array([np.mean([r[arm]["correct"] for r in rows if r["fact"] == fi]) for fi in range(len(ev))]); diff = by_fact("write") - by_fact("cue")
        lb = float(np.quantile([np.mean(rng.choice(diff, len(diff))) for _ in range(cfg["eval"]["bootstraps"])], 0.025)); sf = float(np.mean([np.mean(diff * rng.choice([-1, 1], len(diff))) >= diff.mean() for _ in range(cfg["eval"]["randomizations"])]))
        fam_acc = {fam: acc("write", [r for r in rows if r["family"] == fam]) for fam in ("H1", "H2")}; w_acc = {w: acc("write", [r for r in rows if r["wording"] == w]) for w in ("A", "B")}
        wrong_follow = float(np.mean([r["wrong"]["choice"] == r["donor_correct"] for r in rows])); wrong_base = float(np.mean([r["cue"]["choice"] == r["donor_correct"] for r in rows])); recov = (acc("write") - acc("cue")) / max(acc("visible") - acc("cue"), 1e-6)
        s = {"params": n_par, "loss_history": hist, "acc": {k: acc(k) for k in ("write", "cue", "zero_hooked", "wrong", "random", "visible")}, "completion": {k: comp(k) for k in ("write", "cue", "zero_hooked", "wrong", "random", "visible")}, "family_acc": fam_acc, "wording_acc": w_acc,
             "write_minus_cue": float(diff.mean()), "write_minus_cue_lb": lb, "signflip_p": sf, "write_minus_random": acc("write") - acc("random"), "recovery": recov, "wrong_follows_donor": wrong_follow, "wrong_baseline": wrong_base,
             "trained_write": float(np.mean([r["write"] for r in trows])), "trained_cue": float(np.mean([r["cue"] for r in trows]))}
        gates = {"instrument": s["acc"]["visible"] >= G["visible_min"] and all(s["completion"][k] >= G["termination_min"] for k in ("write", "cue", "wrong", "random", "visible")) and s["acc"]["cue"] <= G["cue_max"],
                 "transfer": s["acc"]["write"] >= G["write_min"] and all(v >= G["write_family_min"] for v in fam_acc.values()) and abs(w_acc["A"] - w_acc["B"]) <= G["wording_diff_max"] and diff.mean() >= G["write_minus_cue_min"] and lb > G["write_minus_cue_lb_min"] and s["write_minus_random"] >= G["write_minus_random_min"] and recov >= G["recovery_min"],
                 "specificity": wrong_follow >= G["wrong_follows_donor_min"] and wrong_follow - wrong_base >= G["wrong_over_baseline_min"],
                 "controller": s["trained_write"] >= G["controller_trained_min"] and s["trained_write"] - s["trained_cue"] >= G["controller_uplift_min"]}
        s["gates"] = gates; s["class"] = "POSITIVE" if all(gates[k] for k in ("instrument", "transfer", "specificity")) else ("CONTROLLER" if gates["controller"] else "FAIL")
        res["seeds"][seed] = {"summary": s, "rows": rows, "trained_rows": trows}; save(); log(f"seed {seed}: " + json.dumps({k: v for k, v in s.items() if k != "loss_history"}, default=float)); log(f"seed {seed} class: {s['class']} ({time.time()-T0:.0f}s)")
        if time.time() > deadline: log("hard wall: stopping seeds"); break
    kinds = [v["summary"]["class"] for v in res["seeds"].values()]
    if len(kinds) < G["seeds_required"]: status = "INCOMPLETE — NO VERDICT"
    elif kinds.count("POSITIVE") >= G["seeds_required"]: status = "BOUNDED POSITIVE — ONE-WRITE PERSISTENT CAUSAL STATE (this construction)"
    elif kinds.count("POSITIVE") + kinds.count("CONTROLLER") >= G["seeds_required"]: status = "SUPERVISED CONTROLLER"
    else: status = "FAIL — ONE-WRITE STATE CONSTRUCTION"
    res["status"] = status; res["per_seed"] = kinds; res["seconds"] = time.time() - T0; save(); log(f"STATUS: {status}; per seed {kinds}")


if __name__ == "__main__":
    main()
