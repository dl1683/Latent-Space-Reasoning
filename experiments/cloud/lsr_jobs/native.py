"""Direction 1 (step-back council): native selectable information in a frozen pretrained residual state.

Donor: k four-choice facts in text, ending with a fixed closing word; the residual at the LAST position after layer
l is the "state". Receiver: a fact-free question with a placeholder token at position 0 whose layer-l residual is
replaced by the donor state (no donor tokens, no donor KV cache). Answer = argmax over the 4 option tokens.
Controls: FULL (donor text + question in one context, no patch), NOPATCH (receiver alone), SHUFFLED (state of a
different donor), and the selective test (donor' differs in exactly one fact: targeted change vs collateral).
"""
import random

import torch

from . import common

NAMES = ["Bex", "Tarn", "Quill", "Mova", "Senn", "Dace", "Lorn", "Pim", "Vash", "Kell", "Rua", "Fen", "Joss", "Nim",
         "Oda", "Ryl", "Sable", "Tov", "Wren", "Yara", "Zed", "Cass", "Brin", "Hale", "Ivo", "Juno", "Kip", "Lux",
         "Mara", "Nox", "Orin", "Pell"]
COLORS = ["red", "blue", "green", "yellow"]


def collection(rng, k, names_pool):
    names = rng.sample(names_pool, k)
    return {"names": names, "colors": [rng.randrange(4) for _ in names]}


def donor_text(c):
    return "Remember these facts.\n" + "\n".join(f"The {n} box is {COLORS[v]}." for n, v in zip(c["names"], c["colors"])) + \
        "\nRemembered."


def question(name):
    return f" Question: What color is the {name} box? Answer: The {name} box is"


class Patcher:
    def __init__(self, model, layer):
        self.state = None
        self.model = model
        model.model.layers[layer].register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if self.state is None or hs.shape[1] == 1:
            return out
        hs = hs.clone()
        hs[:, 0] = self.state.to(hs.dtype)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs


@torch.no_grad()
def donor_states(model, tok, texts, layer, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        hs = model(**enc, output_hidden_states=True).hidden_states[layer + 1]
        out.append(hs[:, -1].float())  # left padding: last position is the closing word
    return torch.cat(out)


@torch.no_grad()
def answer(model, tok, prompts, opt_ids, patcher=None, states=None, bs=32):
    preds = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        if patcher is not None:
            if tok.padding_side == "left":
                raise ValueError("receiver must be right-padded so the placeholder is position 0")
            patcher.state = states[i:i + bs]
        logits = model(**enc).logits
        if patcher is not None:
            patcher.state = None
        last = enc["attention_mask"].sum(1) - 1
        lg = logits[torch.arange(len(last)), last][:, opt_ids]
        preds += lg.argmax(-1).tolist()
    return preds


def main(cfg, out):
    model, tok, info = common.load(cfg["model"])
    out.meta("model_info.json", info)
    L = info["n_layers"]
    opt_ids = [tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in COLORS]
    assert len(set(opt_ids)) == 4, "option first tokens must be distinct"
    rng = random.Random(cfg["seed"])
    split_names = {"dev": NAMES[:16], "test": NAMES[16:]}  # held-out names for test
    for split in cfg["splits"]:
        for lf in cfg["layer_fracs"]:
            layer = int(round(lf * L))
            patcher = Patcher(model, layer)
            for k in cfg["ks"]:
                key = f"native|{split}|l{layer}|k{k}"
                if key in out.done:
                    continue
                cols = [collection(rng, k, split_names[split]) for _ in range(cfg["n"])]
                # full-context ceiling
                tok.padding_side = "left"
                full_p = [donor_text(c) + question(c["names"][j]) for c in cols for j in range(k)]
                gold = [c["colors"][j] for c in cols for j in range(k)]
                full = answer(model, tok, full_p, opt_ids)
                S = donor_states(model, tok, [donor_text(c) for c in cols], layer)
                tok.padding_side = "right"
                recv = [" ." + question(c["names"][j]) for c in cols for j in range(k)]
                st = S.repeat_interleave(k, 0)
                patched = answer(model, tok, recv, opt_ids, patcher, st)
                nopatch = answer(model, tok, recv, opt_ids)
                shuf = answer(model, tok, recv, opt_ids, patcher, S.roll(1, 0).repeat_interleave(k, 0))
                # selective test: change fact i in the donor
                cols2 = []
                for c in cols:
                    c2 = {"names": c["names"], "colors": list(c["colors"])}
                    i = rng.randrange(k)
                    c2["colors"][i] = (c2["colors"][i] + 1 + rng.randrange(3)) % 4
                    c2["changed"] = i
                    cols2.append(c2)
                tok.padding_side = "left"
                S2 = donor_states(model, tok, [donor_text(c) for c in cols2], layer)
                tok.padding_side = "right"
                patched2 = answer(model, tok, recv, opt_ids, patcher, S2.repeat_interleave(k, 0))
                tgt_follow = coll_keep = coll_n = 0
                for ci, c2 in enumerate(cols2):
                    for j in range(k):
                        idx = ci * k + j
                        if j == c2["changed"]:
                            tgt_follow += patched2[idx] == c2["colors"][j]
                        else:
                            coll_n += 1
                            coll_keep += patched2[idx] == patched[idx]
                acc = lambda p: sum(int(a == b) for a, b in zip(p, gold)) / len(gold)
                per_sel = [sum(int(patched[ci * k + j] == gold[ci * k + j]) for ci in range(len(cols))) / len(cols)
                           for j in range(k)]
                out.write({"key": key, "split": split, "layer": layer, "k": k, "n_collections": len(cols),
                           "full_acc": acc(full), "patched_acc": acc(patched), "patched_min_selector": min(per_sel),
                           "nopatch_acc": acc(nopatch), "shuffled_acc": acc(shuf),
                           "targeted_follow": tgt_follow / len(cols2), "collateral_keep": coll_keep / coll_n if coll_n else None})
                out.beat()
            del patcher
            model.model.layers[layer]._forward_hooks.clear()
