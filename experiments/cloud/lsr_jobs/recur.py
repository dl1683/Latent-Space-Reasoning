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
    budget = budget or (cfg["cot_budget"] if mode == "cot" else 16)
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
    blocks = [(round(a * L), round(b * L)) for a, b in cfg["blocks"]]
    dev = items_for(cfg, "dev")
    best = {}
    for mode in cfg["modes"]:
        configs = [(0, 0, 1)] + [(s, e, n) for s, e in blocks for n in cfg["loops"]]
        scores = {}
        for s, e, n in configs:
            key = f"dev|{mode}|{s}-{e}x{n}"
            mult = looper.set(s, e, n) if n > 1 else 1.0
            if key not in out.done:
                t = common.timer()
                rows = run_arm(model, tok, cfg, dev, mode)
                out.write({"key": key, "phase": "dev", "mode": mode, "block": [s, e], "loops": n, "mult": mult,
                           "summary": summarize(rows), "by_family": {f: summarize([r for r in rows if r["id"].startswith(f)])
                                                                     for f in cfg["families"]},
                           "rows": [{k: r[k] for k in ("id", "ok", "term", "ntok")} for r in rows], "sec": t()})
            looper.reset()
            scores[(s, e, n)] = next(r for r in [__import__("json").loads(l) for l in out.path.read_text().splitlines()]
                                     if r["key"] == key)["summary"]["acc"]
        base = scores[(0, 0, 1)]
        cand = max((c for c in configs if c[2] > 1), key=lambda c: scores[c])
        best[mode] = cand if scores[cand] > base else None
        out.meta(f"selection_{mode}.json", {"baseline_dev_acc": base, "best": cand, "best_dev_acc": scores[cand],
                                            "admitted": best[mode] is not None})
    test = items_for(cfg, "test")
    for mode in cfg["modes"]:
        arms = [("base", (0, 0, 1))]
        if best[mode]:
            arms.append(("loop", best[mode]))
        for name, (s, e, n) in arms:
            key = f"test|{mode}|{name}"
            if key in out.done:
                continue
            mult = looper.set(s, e, n) if n > 1 else 1.0
            rows = run_arm(model, tok, cfg, test, mode)
            looper.reset()
            out.write({"key": key, "phase": "test", "mode": mode, "arm": name, "block": [s, e], "loops": n,
                       "mult": mult, "summary": summarize(rows), "rows": rows})
        if best[mode]:
            s, e, n = best[mode]
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
