"""onewrite_recall_v1: a single fact written ONCE into a frozen LM and recalled after unseen wording (Qwen3-1.7B-Base, CPU).
Design: .codex_direction_r15.md (+ round 16 gates). Reuses the one-write machinery (Iface, LM, strict parsing) of run_onewrite_state.py.

    python experiments/run_onewrite_recall.py --config experiments/config/onewrite_recall_v1.json --stage validate|train [--smoke]
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from run_onewrite_state import Iface, LM, strict


def build_facts(cfg):
    rng = np.random.default_rng(cfg["assignment_seed"]); T = cfg["tags"]; facts = []
    for split, names in (("train", cfg["names_train"]), ("eval", cfg["names_eval"])):
        tags = [T[i % len(T)] for i in range(len(names))]; rng.shuffle(tags)                   # balanced assignment, one tag per entity
        facts += [{"split": split, "name": n, "tag": t} for n, t in zip(names, tags)]
    return facts


def target(cfg, f, wording, visible):
    q = cfg["wordings"][wording].format(NAME=f["name"].upper(), name=f["name"]); note = cfg["visible_note"].format(name=f["name"], tag=f["tag"]) if visible else ""
    return cfg["filler"] + note + q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--stage", default="validate"); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args(); T0 = time.time()
    cfg = json.load(open(a.config, encoding="utf-8")); out = f"experiments/results/{cfg['name']}"; os.makedirs(out, exist_ok=True); tag = a.stage + ("_smoke" if a.smoke else "")
    logf = open(os.path.join(out, f"{tag}.log"), "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    if a.smoke: cfg["train"]["steps"] = int(os.environ.get("SMOKE_STEPS", 40)); cfg["train"]["seeds"] = cfg["train"]["seeds"][:1]; cfg["eval"]["bootstraps"] = 200
    shas = {k: hashlib.sha256(open(v, "rb").read()).hexdigest() for k, v in (("runner", __file__), ("config", a.config), ("machinery", os.path.join(os.path.dirname(__file__), "run_onewrite_state.py")))}
    M = LM(cfg); TAGS = [t.upper() for t in cfg["tags"]]; facts = build_facts(cfg); train = [f for f in facts if f["split"] == "train"]; ev = [f for f in facts if f["split"] == "eval"]; G = cfg["gates"]; V = cfg["validation_gate"]
    first_tok = [M.ids(" " + t)[0] for t in cfg["tags"]]; assert len(set(first_tok)) == len(first_tok), "tags must have distinct first tokens"
    assert cfg["filler"].startswith(cfg["slot"]) and not any(t in cfg["filler"].lower() for t in cfg["tags"]); log(f"loaded; slot_pos={M.slot_pos}; {len(train)} train / {len(ev)} eval facts; tag first tokens distinct ({time.time()-T0:.0f}s)")
    res = {"config": cfg["name"], "sha256": shas, "revision": M.sp.revision}; save = lambda: json.dump(res, open(os.path.join(out, f"{tag}_result.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else float(o))
    dec = lambda p, delta: (lambda txt: (txt,) + strict(txt, TAGS))(M.decode(M.ids(p), delta, TAGS))
    if a.stage == "validate":                                                                  # no state: visible-copy vs cue-only on heldout entities x wordings
        rows = []
        for fi, f in enumerate(ev):
            for w in ("A", "B"):
                row = {"fact": fi, "wording": w, "tag": f["tag"].upper()}
                for arm, vis in (("visible", True), ("cue", False)):
                    txt, c, ok = dec(target(cfg, f, w, vis), None); row[arm] = {"text": txt, "choice": c, "completed": ok, "correct": c == f["tag"].upper()}
                rows.append(row)
            log(f"{f['name']} ({f['tag']}): " + " | ".join(f"{r['wording']} vis={r['visible']['text'].strip()[:10]!r}{'+' if r['visible']['correct'] else '-'} cue={r['cue']['text'].strip()[:10]!r}" for r in rows[-2:]))
        s = {"visible_acc": float(np.mean([r["visible"]["correct"] for r in rows])), "cue_acc": float(np.mean([r["cue"]["correct"] for r in rows])), "visible_completion": float(np.mean([r["visible"]["completed"] for r in rows])), "cue_completion": float(np.mean([r["cue"]["completed"] for r in rows])),
             "visible_by_wording": {w: float(np.mean([r["visible"]["correct"] for r in rows if r["wording"] == w])) for w in ("A", "B")}}
        passed = s["visible_acc"] >= V["visible_min"] and s["visible_completion"] >= V["completion_min"] and s["cue_completion"] >= V["completion_min"] and s["cue_acc"] <= V["cue_max"]
        res["validation"] = {"summary": s, "passed": passed, "rows": rows}; save(); log(json.dumps(s, indent=1)); log(f"PRE-LOCK VALIDATION: {'PASS' if passed else 'FAIL - KILL PRE-LOCK'} ({time.time()-T0:.0f}s)"); return
    for f in facts:
        tpl = cfg["source_templates_train"] if f["split"] == "train" else cfg["source_templates_eval"]; f["src"] = [M.source_state(t.format(name=f["name"], tag=f["tag"])) for t in tpl]
    res["seeds"] = {}; deadline = T0 + cfg["train"]["hard_wall_minutes"] * 60
    for seed in cfg["train"]["seeds"]:
        torch.manual_seed(seed); rng = np.random.default_rng(seed); iface = Iface(M.m.config.hidden_size, cfg["state_dim"]); opt = torch.optim.AdamW(iface.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]); t0 = time.time(); hist = []
        for step in range(cfg["train"]["steps"]):
            f = train[rng.integers(len(train))]; src = f["src"][rng.integers(len(f["src"]))]; loss = M.label_loss(M.ids(target(cfg, f, "train", False)), " " + f["tag"], iface.J(iface.enc(src)))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(iface.parameters(), cfg["train"]["clip"]); opt.step(); hist.append(float(loss.detach()))
            if step % 50 == 0 or step == cfg["train"]["steps"] - 1: log(f"seed {seed} step {step}: loss={float(loss):.3f} ({time.time()-t0:.0f}s)")
            if time.time() - t0 > cfg["train"]["seed_wall_minutes"] * 60: log("seed training wall reached"); break
        iface.eval(); torch.save(iface.state_dict(), os.path.join(out, f"iface_seed{seed}.pt"))
        with torch.no_grad():
            ztr = torch.stack([iface.enc(s) for f in train for s in f["src"]]); g = torch.Generator().manual_seed(999 + seed); zr = torch.randn(cfg["state_dim"], generator=g); zr = ztr.mean(0) + (zr - zr.mean()) / zr.std() * ztr.std(0).mean()
            rows = []
            for fi, f in enumerate(ev):
                dz = iface.J(iface.enc(f["src"][0])); d = ev[(fi + 7) % len(ev)]; assert d["tag"] != f["tag"] or True; dd = iface.J(iface.enc(d["src"][0])); dr = iface.J(zr)
                for w in ("A", "B"):
                    p = target(cfg, f, w, False); arms = {"write": (p, dz), "cue": (p, None), "zero_hooked": (p, torch.zeros_like(dz)), "wrong": (p, dd), "random": (p, dr), "visible": (target(cfg, f, w, True), None)}
                    row = {"fact": fi, "wording": w, "tag": f["tag"].upper(), "donor_tag": d["tag"].upper()}
                    for name, (pp, delta) in arms.items():
                        txt, c, ok = dec(pp, delta); row[name] = {"text": txt, "choice": c, "completed": ok, "correct": c == row["tag"]}
                    rows.append(row)
                log(f"seed {seed} {f['name']} ({f['tag']}->donor {d['tag']}): " + " ".join(f"{k}={rows[-1][k]['choice']}" for k in ("write", "cue", "wrong", "random", "visible")) + f" ({time.time()-T0:.0f}s)")
        acc = lambda arm, rs=rows: float(np.mean([r[arm]["correct"] for r in rs])); comp = lambda arm: float(np.mean([r[arm]["completed"] for r in rows]))
        by_fact = lambda arm: np.array([np.mean([r[arm]["correct"] for r in rows if r["fact"] == fi]) for fi in range(len(ev))]); diff = by_fact("write") - by_fact("cue")
        lb = float(np.quantile([np.mean(rng.choice(diff, len(diff))) for _ in range(cfg["eval"]["bootstraps"])], 0.025)); sf = float(np.mean([np.mean(diff * rng.choice([-1, 1], len(diff))) >= diff.mean() for _ in range(cfg["eval"]["randomizations"])]))
        w_acc = {w: acc("write", [r for r in rows if r["wording"] == w]) for w in ("A", "B")}; wrong_follow = float(np.mean([r["wrong"]["choice"] == r["donor_tag"] for r in rows])); cue_donor = float(np.mean([r["cue"]["choice"] == r["donor_tag"] for r in rows]))
        recov = (acc("write") - acc("cue")) / max(acc("visible") - acc("cue"), 1e-6)
        s = {"loss_history": hist, "acc": {k: acc(k) for k in ("write", "cue", "zero_hooked", "wrong", "random", "visible")}, "completion": {k: comp(k) for k in ("write", "cue", "zero_hooked", "wrong", "random", "visible")}, "wording_acc": w_acc,
             "write_minus_cue": float(diff.mean()), "write_minus_cue_lb": lb, "signflip_p": sf, "write_minus_random": acc("write") - acc("random"), "recovery": recov, "wrong_follows_donor": wrong_follow, "cue_donor_rate": cue_donor}
        gates = {"recall": s["acc"]["write"] >= G["write_min"] and all(v >= G["write_wording_min"] for v in w_acc.values()) and abs(w_acc["A"] - w_acc["B"]) <= G["wording_diff_max"] and diff.mean() >= G["write_minus_cue_min"] and lb > G["write_minus_cue_lb_min"] and recov >= G["recovery_min"],
                 "specificity": wrong_follow >= G["wrong_follows_donor_min"] and wrong_follow - cue_donor >= G["wrong_over_cue_donor_min"] and s["write_minus_random"] >= G["write_minus_random_min"] and s["acc"]["random"] <= G["random_max"],
                 "instrument": all(s["completion"][k] >= G["completion_min"] for k in ("write", "cue", "wrong", "random", "visible"))}
        s["gates"] = gates; s["class"] = "POSITIVE" if all(gates.values()) else ("PARTIAL" if gates["instrument"] and gates["recall"] else "FAIL")
        res["seeds"][seed] = {"summary": s, "rows": rows}; save(); log(f"seed {seed}: " + json.dumps({k: v for k, v in s.items() if k != "loss_history"}, default=float)); log(f"seed {seed} class: {s['class']} ({time.time()-T0:.0f}s)")
        if time.time() > deadline: log("hard wall: stopping seeds"); break
    kinds = [v["summary"]["class"] for v in res["seeds"].values()]
    if len(kinds) < G["seeds_required"]: status = "INCOMPLETE — NO VERDICT"
    elif kinds.count("POSITIVE") >= G["seeds_required"]: status = "BOUNDED POSITIVE — ONE-WRITE CAUSAL MEMORY CHANNEL (this construction)"
    elif kinds.count("POSITIVE") + kinds.count("PARTIAL") >= G["seeds_required"]: status = "PARTIAL — RECALL WITHOUT DONOR SPECIFICITY"
    else: status = "FAIL — ONE-WRITE RECALL CONSTRUCTION"
    res["status"] = status; res["per_seed"] = kinds; res["seconds"] = time.time() - T0; save(); log(f"STATUS: {status}; per seed {kinds}")


if __name__ == "__main__":
    main()
