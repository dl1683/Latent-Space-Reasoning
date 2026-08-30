"""onewrite_recall_v1: a single fact written ONCE into a frozen LM and recalled after unseen wording (Qwen3-1.7B-Base, CPU).
Design: .codex_direction_r15.md; locked amendments and gates: .codex_direction_r16.md. Reuses the one-write machinery
(Iface, LM, strict parsing) of run_onewrite_state.py. Wrong-state = same entity with a counterfactual tag; the nonvisible
target is byte-identical across write / cue / zero-hook / wrong / random arms; zero-hook must reproduce cue row-for-row.

    python experiments/run_onewrite_recall.py --config experiments/config/onewrite_recall_v1.json --stage validate|train [--smoke]
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from run_onewrite_state import Iface, LM, strict


def build_facts(cfg):
    rng = np.random.default_rng(cfg["assignment_seed"]); T = cfg["tags"]; facts = []
    for split, names in (("train", cfg["names_train"]), ("eval", cfg["names_eval"])):
        tags = [T[i % len(T)] for i in range(len(names))]; rng.shuffle(tags)                   # balanced: 3 train / 2 heldout facts per tag
        facts += [{"split": split, "name": n, "tag": t} for n, t in zip(names, tags)]
    for i, f in enumerate(facts): f["cf_tag"] = T[(T.index(f["tag"]) + 1 + (i % (len(T) - 1))) % len(T)]   # balanced derangement, never the true tag
    return facts


def target(cfg, f, wording, filler_i, visible):
    q = cfg["wordings"][wording].format(NAME=f["name"].upper(), name=f["name"]); note = cfg["visible_note"].format(name=f["name"], tag=f["tag"]) if visible else ""
    return cfg["fillers"][filler_i] + cfg["valid_line"] + note + q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--stage", default="validate"); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args(); T0 = time.time()
    cfg = json.load(open(a.config, encoding="utf-8")); out = f"experiments/results/{cfg['name']}"; os.makedirs(out, exist_ok=True); tag = a.stage + ("_smoke" if a.smoke else "")
    logf = open(os.path.join(out, f"{tag}.log"), "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    if a.smoke: cfg["train"]["steps"] = int(os.environ.get("SMOKE_STEPS", 40)); cfg["train"]["seeds"] = cfg["train"]["seeds"][:1]; cfg["eval"]["bootstraps"] = 200
    shas = {k: hashlib.sha256(open(v, "rb").read()).hexdigest() for k, v in (("runner", __file__), ("config", a.config), ("machinery", os.path.join(os.path.dirname(__file__), "run_onewrite_state.py")))}
    M = LM(cfg); TAGS = [t.upper() for t in cfg["tags"]]; facts = build_facts(cfg); train = [f for f in facts if f["split"] == "train"]; G = cfg["gates"]; V = cfg["validation_gate"]
    ev = [f for f in facts if f["split"] == cfg.get("eval_split", "eval")]; WORDS = cfg.get("eval_wordings", ["A", "B"])                # rung 1: TRAINING entities under the training wording
    # leakage assertions
    assert len({M.ids(" " + t)[0] for t in cfg["tags"]}) == len(cfg["tags"]), "tags must have distinct first tokens"
    lows = [t.lower() for t in cfg["tags"]]; texts = cfg["fillers"] + list(cfg["wordings"].values()) + cfg["names_train"] + cfg["names_eval"]
    assert all(not any(tg in x.lower() for tg in lows) for x in texts), "no tag may occur in fillers, wordings or names"
    assert all(fl.startswith(cfg["slot"]) for fl in cfg["fillers"]); fl_len = [len(M.ids(fl)) for fl in cfg["fillers"]]; assert abs(fl_len[0] - fl_len[1]) <= 2 and min(fl_len) >= cfg.get("min_filler_tokens", 64), fl_len
    log(f"loaded; slot_pos={M.slot_pos}; {len(train)} train / {len(ev)} eval facts; filler tokens {fl_len} ({time.time()-T0:.0f}s)")
    res = {"config": cfg["name"], "sha256": shas, "revision": M.sp.revision}; save = lambda: json.dump(res, open(os.path.join(out, f"{tag}_result.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else float(o))
    dec = lambda ids, delta: (lambda txt: (txt,) + strict(txt, TAGS))(M.decode(ids, delta, TAGS))
    if a.stage == "validate":                                                                  # no state: visible-copy vs cue-only on heldout entities x wordings (filler = wording index)
        rows = []
        for fi, f in enumerate(ev):
            for wi, w in enumerate(WORDS):
                row = {"fact": fi, "wording": w, "tag": f["tag"].upper()}
                for arm, vis in (("visible", True), ("cue", False)):
                    txt, c, ok = dec(M.ids(target(cfg, f, w, wi, vis)), None); row[arm] = {"text": txt, "choice": c, "completed": ok, "correct": c == row["tag"]}
                rows.append(row)
            log(f"{f['name']} ({f['tag']}): " + " | ".join(f"{r['wording']} vis={r['visible']['text'].strip()[:8]!r}{'+' if r['visible']['correct'] else '-'} cue={r['cue']['text'].strip()[:8]!r}" for r in rows[-2:]))
        acc = lambda arm, rs=rows: float(np.mean([r[arm]["correct"] for r in rs])); comp = lambda arm: float(np.mean([r[arm]["completed"] for r in rows])); byw = lambda arm: {w: acc(arm, [r for r in rows if r["wording"] == w]) for w in WORDS}
        s = {"visible_acc": acc("visible"), "cue_acc": acc("cue"), "visible_completion": comp("visible"), "cue_completion": comp("cue"), "visible_by_wording": byw("visible"), "cue_by_wording": byw("cue")}
        zero_ok = all((lambda t, c2, ok2: t == r["cue"]["text"] and c2 == r["cue"]["choice"] and ok2 == r["cue"]["completed"])(*dec(M.ids(target(cfg, ev[r["fact"]], r["wording"], 0 if r["wording"] == "A" else 1, False)), torch.zeros(M.m.config.hidden_size))) for r in rows)
        s["zero_hook_matches_cue"] = zero_ok                                                     # round 17: cue completion is diagnostic only
        passed = (s["visible_acc"] >= V["visible_min"] and all(v >= V["visible_wording_min"] for v in s["visible_by_wording"].values()) and s["visible_completion"] >= V["completion_min"] and s["cue_acc"] <= V["cue_max"]
                  and all(v <= V["cue_wording_max"] for v in s["cue_by_wording"].values()) and s["visible_acc"] - s["cue_acc"] >= V["visible_minus_cue_min"] and zero_ok)
        res["validation"] = {"summary": s, "passed": passed, "rows": rows}; save(); log(json.dumps(s, indent=1)); log(f"PRE-LOCK VALIDATION: {'PASS' if passed else 'FAIL - KILL PRE-LOCK'} ({time.time()-T0:.0f}s)"); return
    for f in facts:                                                                            # cached source states; heldout facts also cache the counterfactual-tag sources
        tpl = cfg["source_templates_train"] if f["split"] == "train" else cfg["source_templates_eval"]; f["src"] = [M.source_state(t.format(name=f["name"], tag=f["tag"])) for t in tpl]
        f["src_cf"] = [M.source_state(t.format(name=f["name"], tag=f["cf_tag"])) for t in tpl]
    res["seeds"] = {}; deadline = T0 + cfg["train"]["hard_wall_minutes"] * 60
    for seed in cfg["train"]["seeds"]:
        torch.manual_seed(seed); rng = np.random.default_rng(seed); iface = Iface(M.m.config.hidden_size, cfg["state_dim"]); opt = torch.optim.AdamW(iface.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]); t0 = time.time(); hist = []
        for step in range(cfg["train"]["steps"]):
            f = train[rng.integers(len(train))]; src = f["src"][rng.integers(len(f["src"]))]; loss = M.label_loss(M.ids(target(cfg, f, "train", int(rng.integers(2)), False)), " " + f["tag"], iface.J(iface.enc(src)))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(iface.parameters(), cfg["train"]["clip"]); opt.step(); hist.append(float(loss.detach()))
            if step % 50 == 0 or step == cfg["train"]["steps"] - 1: log(f"seed {seed} step {step}: loss={float(loss):.3f} ({time.time()-t0:.0f}s)")
            if time.time() - t0 > cfg["train"]["seed_wall_minutes"] * 60: log("seed training wall reached"); break
        iface.eval(); torch.save(iface.state_dict(), os.path.join(out, f"iface_seed{seed}.pt"))
        with torch.no_grad():
            ztr = torch.stack([iface.enc(s) for f in train for s in f["src"]]); g = torch.Generator().manual_seed(999 + seed); zr = torch.randn(cfg["state_dim"], generator=g); zr = ztr.mean(0) + (zr - zr.mean()) / zr.std() * ztr.std(0).mean(); dr = iface.J(zr)
            rows = []
            for fi, f in enumerate(ev):
                for si in range(2):                                                             # both heldout source phrasings
                    dz = iface.J(iface.enc(f["src"][si])); dcf = iface.J(iface.enc(f["src_cf"][si]))
                    for wi, w in enumerate(WORDS):
                        ids = M.ids(target(cfg, f, w, wi, False)); vids = M.ids(target(cfg, f, w, wi, True))
                        arms = {"write": (ids, dz), "cue": (ids, None), "zero_hooked": (ids, torch.zeros_like(dz)), "wrong": (ids, dcf), "random": (ids, dr), "visible": (vids, None)}
                        row = {"fact": fi, "source": si, "wording": w, "tag": f["tag"].upper(), "cf_tag": f["cf_tag"].upper()}
                        for name, (pid, delta) in arms.items():
                            txt, c, ok = dec(pid, delta); row[name] = {"text": txt, "choice": c, "completed": ok, "correct": c == row["tag"]}
                        rows.append(row)
                log(f"seed {seed} {f['name']} ({f['tag']}, cf {f['cf_tag']}): " + " ".join(f"{k}={rows[-1][k]['choice']}" for k in ("write", "cue", "wrong", "random", "visible")) + f" ({time.time()-T0:.0f}s)")
        acc = lambda arm, rs=rows: float(np.mean([r[arm]["correct"] for r in rs])); comp = lambda arm: float(np.mean([r[arm]["completed"] for r in rows]))
        by_fact = lambda arm: np.array([np.mean([r[arm]["correct"] for r in rows if r["fact"] == fi]) for fi in range(len(ev))]); boot = lambda d: float(np.quantile([np.mean(rng.choice(d, len(d))) for _ in range(cfg["eval"]["bootstraps"])], 0.025))
        dc = by_fact("write") - by_fact("cue"); drn = by_fact("write") - by_fact("random"); sub = {f"source{si}": acc("write", [r for r in rows if r["source"] == si]) for si in range(2)} | {f"wording{w}": acc("write", [r for r in rows if r["wording"] == w]) for w in WORDS}
        cf_follow = float(np.mean([r["wrong"]["choice"] == r["cf_tag"] for r in rows])); cue_cf = float(np.mean([r["cue"]["choice"] == r["cf_tag"] for r in rows])); recov = (acc("write") - acc("cue")) / max(acc("visible") - acc("cue"), 1e-6)
        zero_match = all(r["zero_hooked"]["text"] == r["cue"]["text"] and r["zero_hooked"]["choice"] == r["cue"]["choice"] and r["zero_hooked"]["completed"] == r["cue"]["completed"] for r in rows)
        s = {"loss_history": hist, "acc": {k: acc(k) for k in ("write", "cue", "zero_hooked", "wrong", "random", "visible")}, "completion": {k: comp(k) for k in ("write", "cue", "zero_hooked", "wrong", "random", "visible")}, "write_sub": sub,
             "write_minus_cue": float(dc.mean()), "write_minus_cue_lb": boot(dc), "write_minus_random": float(drn.mean()), "write_minus_random_lb": boot(drn), "recovery": recov, "cf_follow": cf_follow, "cue_cf_rate": cue_cf, "zero_hook_matches_cue": zero_match}
        gates = {"core": s["acc"]["write"] >= G["write_min"] and dc.mean() >= G["write_minus_cue_min"] and s["write_minus_cue_lb"] > G["write_minus_cue_lb_min"] and drn.mean() >= G["write_minus_random_min"] and s["write_minus_random_lb"] > G["write_minus_random_lb_min"] and s["acc"]["random"] <= G["random_max"] and recov >= G["recovery_min"],
                 "wording": all(v >= G["write_sub_min"] for v in sub.values()) and abs(sub["source0"] - sub["source1"]) <= G["wording_gap_max"] and (len(WORDS) < 2 or abs(sub[f"wording{WORDS[0]}"] - sub[f"wording{WORDS[1]}"]) <= G["wording_gap_max"]),
                 "specificity": cf_follow >= G["wrong_follows_cf_min"] and cf_follow - cue_cf >= G["wrong_over_cue_cf_min"],
                 "completion": all(s["completion"][k] >= G["completion_min"] for k in G["completion_arms"])}          # round 17: cue/zero-hook completion diagnostic only
        s["gates"] = gates; s["class"] = "INVALID" if not zero_match else ("POSITIVE" if all(gates.values()) else ("PARTIAL" if gates["core"] and gates["completion"] else "FAIL"))   # random-write completion failure => completion gate fails => FAIL
        res["seeds"][seed] = {"summary": s, "rows": rows}; save(); log(f"seed {seed}: " + json.dumps({k: v for k, v in s.items() if k != "loss_history"}, default=float)); log(f"seed {seed} class: {s['class']} ({time.time()-T0:.0f}s)")
        if time.time() > deadline: log("hard wall: stopping seeds"); break
    kinds = [v["summary"]["class"] for v in res["seeds"].values()]
    if "INVALID" in kinds: status = "INVALID — NO VERDICT (zero-hook did not reproduce cue)"
    elif len(kinds) < G["seeds_required"]: status = "INCOMPLETE — NO VERDICT"
    elif kinds.count("POSITIVE") >= G["seeds_required"]: status = "BOUNDED POSITIVE — ONE-WRITE SHORT-CONTEXT CAUSAL RECALL CHANNEL (this construction)"
    elif kinds.count("POSITIVE") + kinds.count("PARTIAL") >= G["seeds_required"]: status = "PARTIAL — RECALL WITHOUT WORDING ROBUSTNESS OR COUNTERFACTUAL SPECIFICITY"
    else: status = "FAIL — ONE-WRITE RECALL CONSTRUCTION"
    res["status"] = status; res["per_seed"] = kinds; res["seconds"] = time.time() - T0; save(); log(f"STATUS: {status}; per seed {kinds}")


if __name__ == "__main__":
    main()
