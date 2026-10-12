"""Registration F (step-back direction 3): formation collisions in a pretrained model.

4 families x N fictional entities, each with one of 8 options; LoRA on a frozen base learns them from training
wordings; retained answerability is scored on held-out question wordings (argmax over the 8 option first-tokens).
Arms (identical examples and total steps, single-family batches, common balanced consolidation phase):
staged, reversed, interleaved (round-robin, one AdamW), interleaved_moments (round-robin, one AdamW state per family).
"""
import random

import torch
import torch.nn.functional as F

from . import common

FAMILIES = {
    "color": {"opts": ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"],
              "train": ["The {e} flag is {o}.", "{e} has a {o} flag.", "The flag of {e} is colored {o}."],
              "test": [" Question: What color is the {e} flag? Answer: The {e} flag is",
                       " The color of the flag of {e} is"]},
    "city": {"opts": ["Paris", "Tokyo", "Lima", "Cairo", "Oslo", "Delhi", "Quito", "Rome"],
             "train": ["{e} lives in {o}.", "The home city of {e} is {o}.", "{e} resides in {o}."],
             "test": [" Question: Which city does {e} live in? Answer: {e} lives in",
                      " The city where {e} lives is"]},
    "instrument": {"opts": ["piano", "violin", "drum", "flute", "guitar", "harp", "cello", "trumpet"],
                   "train": ["{e} plays the {o}.", "The instrument of {e} is the {o}.", "{e} is a {o} player."],
                   "test": [" Question: Which instrument does {e} play? Answer: {e} plays the",
                            " The instrument that {e} plays is the"]},
    "number": {"opts": ["two", "three", "four", "five", "six", "seven", "eight", "nine"],
               "train": ["{e} owns {o} boats.", "The number of boats {e} owns is {o}.", "{e} has {o} boats."],
               "test": [" Question: How many boats does {e} own? Answer: {e} owns",
                        " The number of boats that {e} owns is"]},
}
SYL = ["ka", "ro", "vi", "len", "dar", "mo", "zu", "tel", "qua", "nis", "bor", "fe", "lo", "sar", "wen", "ix"]


def entities(rng, n):
    seen, out = set(), []
    while len(out) < n:
        e = "".join(rng.choice(SYL) for _ in range(3)).capitalize()
        if e not in seen:
            seen.add(e); out.append(e)
    return out


def build(cfg):
    rng = random.Random(20261012)
    fam_names = list(FAMILIES)
    data = {}
    ents = entities(rng, cfg["n_per_family"] * len(fam_names))
    for fi, f in enumerate(fam_names):
        es = ents[fi * cfg["n_per_family"]:(fi + 1) * cfg["n_per_family"]]
        ans = [i % 8 for i in range(len(es))]
        rng.shuffle(ans)  # balanced options
        data[f] = {"ents": es, "ans": ans}
    return fam_names, data


def train_texts(f, d):
    w = FAMILIES[f]
    return [w["train"][j].format(e=e, o=w["opts"][a]) for e, a in zip(d["ents"], d["ans"]) for j in range(len(w["train"]))]


@torch.no_grad()
def evaluate(model, tok, fam_names, data, opt_ids, freq=False):
    tok.padding_side = "right"
    res = {}
    model.eval()
    for f in fam_names:
        w, d = FAMILIES[f], data[f]
        prompts = [q.format(e=e) for e in d["ents"] for q in w["test"]]
        gold = [a for a in d["ans"] for _ in w["test"]]
        preds = []
        for i in range(0, len(prompts), 64):
            enc = tok(prompts[i:i + 64], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            lg = model(**enc).logits
            last = enc["attention_mask"].sum(1) - 1
            preds += lg[torch.arange(len(last)), last][:, opt_ids[f]].argmax(-1).tolist()
        res[f] = sum(int(p == g) for p, g in zip(preds, gold)) / len(gold)
        res[f + "_option_freq"] = [preds.count(o) / len(preds) for o in range(8)]
    model.train()
    acc = {f: res[f] for f in fam_names}
    return {"family_acc": acc, "mean_acc": sum(acc.values()) / len(acc),
            "option_freq": {f: res[f + "_option_freq"] for f in fam_names}} if freq else acc


def main(cfg, out):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(cfg["seed"])
    model, tok, info = common.load(cfg["model"])
    model = get_peft_model(model, LoraConfig(r=cfg.get("lora_r", 16), lora_alpha=32,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    out.meta("model_info.json", info)
    fam_names, data = build(cfg)
    opt_ids = {f: [tok(" " + o, add_special_tokens=False)["input_ids"][0] for o in FAMILIES[f]["opts"]] for f in fam_names}
    for f in fam_names:
        assert len(set(opt_ids[f])) == 8, f"option first tokens not distinct for {f}"
    texts = {f: train_texts(f, data[f]) for f in fam_names}
    bs, total = cfg["batch"], cfg["steps"]
    cons = int(0.1 * total)
    main_steps = total - cons
    per_fam = main_steps // len(fam_names)
    main_steps = per_fam * len(fam_names)
    cons = total - main_steps
    # identical per-family batch CONTENT across arms (seeded by family, not arm); arms differ only in ORDER
    def batches_for(f, n, salt):
        r = random.Random(f"{cfg['data_seed']}|{f}|{salt}")
        pool, out_ = [], []
        while len(out_) < n:
            if len(pool) < bs:
                pool = r.sample(texts[f], len(texts[f])) + pool
            out_.append([pool.pop() for _ in range(bs)])
        return out_
    fam_batches = {f: batches_for(f, per_fam, "main") for f in fam_names}
    cons_batches = [batches_for(f, cons // len(fam_names) + 1, "cons") for f in fam_names]
    cons_list = [cons_batches[i % len(fam_names)][i // len(fam_names)] for i in range(cons)]
    arm = cfg["arm"]
    if arm in ("staged", "reversed"):
        order = fam_names if arm == "staged" else list(reversed(fam_names))
        sched = [(f, fam_batches[f][i]) for f in order for i in range(per_fam)]
    else:
        sched = [(fam_names[s % len(fam_names)], fam_batches[fam_names[s % len(fam_names)]][s // len(fam_names)])
                 for s in range(main_steps)]
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    b1, b2, eps = 0.9, 0.999, 1e-8

    def new_state():
        return [(torch.zeros_like(p_), torch.zeros_like(p_)) for p_ in params]
    # family-specific moment buffers for interleaved_moments; one shared buffer set otherwise.
    # bias correction and lr schedule use the GLOBAL step in every arm (amendment).
    states = {f: new_state() for f in fam_names} if arm == "interleaved_moments" else {"shared": new_state()}

    def adam_step(buf, t, lr):
        with torch.no_grad():
            for p_, (m_, v_) in zip(params, buf):
                if p_.grad is None:
                    continue
                g = p_.grad
                m_.mul_(b1).add_(g, alpha=1 - b1)
                v_.mul_(b2).addcmul_(g, g, value=1 - b2)
                mh = m_ / (1 - b1 ** t)
                vh = v_ / (1 - b2 ** t)
                p_.add_(-lr * mh / (vh.sqrt() + eps))
    tok.padding_side = "right"
    t0 = common.timer()
    model.train()
    log_every = max(1, total // 10)
    for step in range(total):
        if step < main_steps:
            f, batch = sched[step]
            buf = states[f] if arm == "interleaved_moments" else states["shared"]
        else:
            if step == main_steps:  # consolidation: fresh optimizer for every arm, identical balanced batches
                ev = evaluate(model, tok, fam_names, data, opt_ids, freq=True)
                out.write({"key": "f|pre_consolidation", "arm": arm, "seed": cfg["seed"], **ev})
                cons_buf = new_state()
            batch, buf = cons_list[step - main_steps], cons_buf
        lr = cfg["lr"] * min(1.0, (step + 1) / 50)
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**enc, labels=labels).loss
        for p_ in params:
            p_.grad = None
        loss.backward()
        adam_step(buf, step + 1 if step < main_steps else step - main_steps + 1, lr)
        if (step + 1) % log_every == 0:
            ev = evaluate(model, tok, fam_names, data, opt_ids, freq=True)
            out.write({"key": f"f|step{step + 1}", "arm": arm, "seed": cfg["seed"], "step": step + 1,
                       "phase": "consolidation" if step >= main_steps else "main", "loss": loss.item(), "sec": t0(), **ev})
    ev = evaluate(model, tok, fam_names, data, opt_ids, freq=True)
    out.write({"key": "f|final", "arm": arm, "seed": cfg["seed"], "chance": 1 / 8, "steps": total,
               "main_steps": main_steps, "per_family_batches": per_fam, **ev})
