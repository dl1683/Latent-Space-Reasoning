"""Runner smoke test: load a small model, verify layer looping is cache-safe (loops=1 reproduces the
unmodified model exactly; looped generate runs), measure the identical-input noise floor."""
import torch

from . import common, tasks


def main(cfg, out):
    model, tok, info = common.load(cfg["model"])
    out.meta("model_info.json", info)
    items = tasks.make("arith", "dev", 0, 8)
    prompts = [common.chat(tok, it["prompt"] + "\nRespond with only the final answer.") for it in items]
    a = common.generate(model, tok, prompts, 16)
    b = common.generate(model, tok, prompts, 16)
    lp = common.Looper(model)
    lp.set(0, 0, 1)
    c = common.generate(model, tok, prompts, 16)
    lp.reset()
    L = info["n_layers"]
    mult = lp.set(L // 3, 2 * L // 3, 2)
    d = common.generate(model, tok, prompts, 16)
    lp.reset()
    enc = tok(prompts[:1], return_tensors="pt").to("cuda")
    with torch.no_grad():
        l1 = model(**enc).logits[0, -1]
        lp.set(0, 0, 1)
        l2 = model(**enc).logits[0, -1]
        lp.reset()
    out.write({"key": "smoke", "repeat_identical": [x[0] for x in a] == [x[0] for x in b],
               "loop1_identical": [x[0] for x in a] == [x[0] for x in c], "logit_maxdiff_loop1": float((l1 - l2).abs().max()),
               "looped_mult": mult, "base": [x[0] for x in a], "looped": [x[0] for x in d],
               "answers": [it["answer"] for it in items]})
