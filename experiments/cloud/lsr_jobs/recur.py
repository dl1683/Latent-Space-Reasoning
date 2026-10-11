"""Probe C: frozen recurrent depth. Repeat a middle decoder block at inference (no training) and measure
held-out exact accuracy against the unlooped model and equal-FLOP controls.

Phase dev: sweep blocks x loops on dev items (per mode). Phase test: freeze the best dev config per mode
(only if it beat the dev baseline) and evaluate it, the baseline and the equal-FLOP control on test items.
Equal-FLOP controls: direct mode -> P filler tokens appended to the prompt so prompt FLOPs match;
cot mode -> self-consistency with round(mult) samples when mult >= 1.5, else the baseline with budget x mult.
"""
import math
from collections import Counter

from . import common, tasks

PREFILL = "Final answer:"  # direct mode: assistant prefill, generation stops at the first newline
PROMPTS = {
    "direct": "{q}\nRespond with only the final answer (a number or an expression), no explanation.",
    "cot": "{q}\nThink step by step, then give the final answer on the last line as 'Answer: <answer>'.",
}


def items_for(cfg, split):
    n = cfg["n_dev"] if split == "dev" else cfg["n_test"]
    return [it for fam in cfg["families"] for it in tasks.make(fam, split, cfg.get("task_seed", 0), n)]


def run_arm(model, tok, cfg, items, mode, filler=0, budget=None, samples=1):
    budget = budget or (cfg["cot_budget"] if mode == "cot" else cfg.get("direct_budget", 64))
    prompts = [common.chat(tok, PROMPTS[mode].format(q=it["prompt"]) + (" ." * filler)) + (PREFILL if mode == "direct" else "")
               for it in items]
    outs = []
    for s in range(samples):
        outs.append(common.generate(model, tok, prompts, budget, temperature=0.7 if samples > 1 else 0.0,
                                    bs=cfg["bs"], seed=1000 * s if samples > 1 else None,
                                    stop=["\n"] if mode == "direct" else None))
    rows = []
    for i, it in enumerate(items):
        preds = [tasks.extract_final(o[i][0], it["check"]) for o in outs]
        if samples > 1:
            pred = Counter(p for p in preds if p is not None).most_common(1)
            pred = pred[0][0] if pred else None
        else:
            pred = preds[0]
        rows.append({"id": it["id"], "ok": tasks.verify(pred, it), "term": all(o[i][2] for o in outs),
                     "ntok": sum(o[i][1] for o in outs), "pred": pred, "text": outs[0][i][0][-300:]})
    return rows


def summarize(rows):
    n = len(rows)
    return {"n": n, "acc": sum(r["ok"] for r in rows) / n, "term": sum(r["term"] for r in rows) / n,
            "mean_tok": sum(r["ntok"] for r in rows) / n}


def main(cfg, out):
    model, tok, info = common.load(cfg["model"])
    info["n_params_nonembed"] = common.nonembed_params(model)
    out.meta("model_info.json", info)
    looper = common.Looper(model)
    L = info["n_layers"]
    bw = cfg.get("block_width", 4)
    blocks = [(max(0, round(c * L) - bw // 2), min(L, max(0, round(c * L) - bw // 2) + bw)) for c in cfg["centers"]]
    variants = [tuple(v) for v in cfg["variants"]]  # (mode, gamma)
    grid = [(s, e, n, vm, g) for s, e in blocks for n in cfg["loops"] for vm, g in variants]
    dev = items_for(cfg, "dev")
    probe_prompts = [common.chat(tok, PROMPTS["cot"].format(q=it["prompt"])) for it in dev[:8]]
    best, scores = {}, {}

    def cfg_key(c):
        s, e, n, vm, g = c
        return "base" if n == 1 else f"{s}-{e}x{n}-{vm}{g}"

    def run_dev(mode, c):
        s, e, n, vm, g = c
        key = f"dev|{mode}|{cfg_key(c)}"
        if key not in out.done:
            mult = looper.set(s, e, n, vm, g) if n > 1 else 1.0
            trace = None
            if n > 1:
                looper.record = True
                with __import__("torch").no_grad():
                    enc = tok(probe_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                    model(**enc)
                looper.record = False
                trace = looper.trace
            t = common.timer()
            rows = run_arm(model, tok, cfg, dev, mode)
            looper.reset()
            out.write({"key": key, "phase": "dev", "mode": mode, "block": [s, e], "loops": n, "variant": [vm, g],
                       "mult": mult, "divergence_per_application": trace, "summary": summarize(rows),
                       "by_family": {f: summarize([r for r in rows if r["id"].startswith(f)]) for f in cfg["families"]},
                       "rows": [{k: r[k] for k in ("id", "ok", "term", "ntok")} for r in rows], "sec": t()})
        import json as _j
        for line in out.path.read_text().splitlines():
            r = _j.loads(line)
            if r["key"] == key:
                scores[(mode, c)] = r["summary"]["acc"]
        return scores[(mode, c)]

    base_c = (0, 0, 1, "plain", 1.0)
    # direct mode: full dev grid (cheap). cot mode: base + 3 best direct configs + plain x2 at each block
    # (declared before data: CoT at 2048 tokens is too costly for the full grid).
    for mode in cfg["modes"]:
        if mode == "direct" or "direct" not in cfg["modes"]:
            cands = grid
        else:
            ranked = sorted(grid, key=lambda c: -scores.get(("direct", c), -1))[:3]
            cands = list(dict.fromkeys(ranked + [(s, e, 2, "plain", 1.0) for s, e in blocks]))
        base = run_dev(mode, base_c)
        for c in cands:
            run_dev(mode, c)
        cand = max(cands, key=lambda c: scores[(mode, c)])
        best[mode] = cand if scores[(mode, cand)] > base else None
        out.meta(f"selection_{mode}.json", {"baseline_dev_acc": base, "best": cand, "best_dev_acc": scores[(mode, cand)],
                                            "admitted": best[mode] is not None})
    test = items_for(cfg, "test")
    for mode in cfg["modes"]:
        arms = [("base", base_c)]
        if best[mode]:
            arms.append(("loop", best[mode]))
        for name, (s, e, n, vm, g) in arms:
            key = f"test|{mode}|{name}"
            if key in out.done:
                continue
            mult = looper.set(s, e, n, vm, g) if n > 1 else 1.0
            rows = run_arm(model, tok, cfg, test, mode)
            looper.reset()
            out.write({"key": key, "phase": "test", "mode": mode, "arm": name, "block": [s, e], "loops": n,
                       "mult": mult, "summary": summarize(rows), "rows": rows})
        if best[mode]:
            s, e, n = best[mode][:3]
            mult = (L + (n - 1) * (e - s)) / L
            key = f"test|{mode}|flop_control"
            if key not in out.done:
                if mode == "direct":
                    plen = sum(len(tok(common.chat(tok, PROMPTS[mode].format(q=it["prompt"])) + PREFILL)["input_ids"]) for it in test) / len(test)
                    filler = int(math.ceil(plen * (mult - 1)))
                    rows = run_arm(model, tok, cfg, test, mode, filler=filler)
                    desc = f"{filler} filler tokens"
                elif mult >= 1.5:
                    rows = run_arm(model, tok, cfg, test, mode, samples=int(round(mult)))
                    desc = f"self-consistency x{int(round(mult))}"
                else:
                    rows = run_arm(model, tok, cfg, test, mode, budget=int(cfg["cot_budget"] * mult))
                    desc = f"budget x{mult:.2f}"
                out.write({"key": key, "phase": "test", "mode": mode, "arm": "flop_control", "desc": desc,
                           "mult": mult, "summary": summarize(rows), "rows": rows})
