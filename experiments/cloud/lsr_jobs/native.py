"""Registration Q (step-back directions 1 and 2) on a frozen pretrained model.

Facts: k labelled four-choice facts, balanced colors, names disjoint dev/test, two wording families; a fixed closing
word ends the fact block; the selector question comes afterward. Answers = argmax over the 4 option first-tokens.
Q1 (causal effective dimension): donor and receiver contexts differ in exactly one fact; at layer l and the closing
position, patch into the receiver (its own context retained) the full donor-minus-receiver difference, its
projection on the top-r dev principal subspace of single-fact differences, the complement, or a norm-matched random
r-dim projection. Targeted effect: the receiver's answer about the changed fact follows the donor; collateral:
accuracy change on other facts.
Q2 (single-position capacity, spatial ladder): fact-free receiver with m placeholder positions whose layer-l
residuals are replaced by the donor's last m positions (no donor KV, no other donor positions).
"""
import random

import torch

from . import common

NAMES = ["Bex", "Tarn", "Quill", "Mova", "Senn", "Dace", "Lorn", "Pim", "Vash", "Kell", "Rua", "Fen", "Joss", "Nim",
         "Oda", "Ryl", "Sable", "Tov", "Wren", "Yara", "Zed", "Cass", "Brin", "Hale", "Ivo", "Juno", "Kip", "Lux",
         "Mara", "Nox", "Orin", "Pell", "Quor", "Rhee", "Sil", "Tay", "Ulm", "Vey", "Wyn", "Xan"]
COLORS = ["red", "blue", "green", "yellow"]
FAM = {
    "W1": {"head": "Remember these facts.\n", "fact": "The {n} box is {c}.", "close": "\nRemembered. Keep all of these facts in mind for later questions.",
           "q": " Question: What color is the {n} box? Answer: The {n} box is"},
    "W2": {"head": "Inventory notes:\n", "fact": "- {n}'s crate: painted {c}", "close": "\nEnd of notes. The record above is complete and will be queried later.",
           "q": " Q: Which color was {n}'s crate painted? A: It was painted"},
}


def collection(rng, k, pool):
    return {"names": rng.sample(pool, k), "colors": [rng.randrange(4) for _ in range(k)]}


def block(c, f):
    w = FAM[f]
    return w["head"] + "\n".join(w["fact"].format(n=n, c=COLORS[v]) for n, v in zip(c["names"], c["colors"])) + w["close"]


def changed(rng, c):
    c2 = {"names": c["names"], "colors": list(c["colors"])}
    i = rng.randrange(len(c["names"]))
    c2["colors"][i] = (c2["colors"][i] + 1 + rng.randrange(3)) % 4
    return c2, i


class Hook:
    """Replace/add residual at given (row, pos) after decoder layer `layer` during the prefill pass."""

    def __init__(self, model, layer):
        self.edits = None  # list of (pos_tensor[B], value[B, D]) ; mode 'add' or 'set'
        self.h = model.model.layers[layer].register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if self.edits is None or hs.shape[1] == 1:
            return out
        hs = hs.clone()
        rows = torch.arange(hs.shape[0], device=hs.device)
        for pos, val, mode in self.edits:
            pos, val = pos.to(hs.device), val.to(hs.device)
            if mode == "add":
                hs[rows, pos] = hs[rows, pos] + val.to(hs.dtype)
            else:
                hs[rows, pos] = val.to(hs.dtype)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

    def remove(self):
        self.h.remove()


@torch.no_grad()
def run_prompts(model, tok, prompts, opt_ids, hook=None, edits_fn=None, bs=24):
    """Right-padded batched scoring; edits_fn(batch_slice, enc) -> list of (pos, val, mode) for this batch."""
    tok.padding_side = "right"
    preds = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        if hook is not None and edits_fn is not None:
            hook.edits = edits_fn(slice(i, i + bs), enc)
        logits = model(**enc).logits
        if hook is not None:
            hook.edits = None
        last = (enc["attention_mask"].sum(1) - 1).to(logits.device)
        preds += logits[torch.arange(len(last), device=logits.device), last][:, opt_ids].argmax(-1).tolist()
    return preds


@torch.no_grad()
def states_at(model, tok, texts, layer, positions=None, bs=24):
    """Layer-`layer` residuals at each text's last token (or last m tokens if positions=m)."""
    tok.padding_side = "right"
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        hs = model(**enc, output_hidden_states=True).hidden_states[layer + 1].float()
        last = enc["attention_mask"].sum(1) - 1
        if positions is None:
            out.append(hs[torch.arange(len(last)), last])
        else:
            out.append(torch.stack([hs[torch.arange(len(last)), last - (positions - 1 - j)] for j in range(positions)], 1))
    return torch.cat(out)


def close_pos(tok, texts):
    return torch.tensor([len(tok(t, add_special_tokens=False)["input_ids"]) - 1 for t in texts], device="cuda")


def q1_cell(model, tok, opt_ids, layer, rng, cfg, k, fam, pool, n, basis):
    """Returns per-variant targeted-follow and collateral-keep rates (lists per collection for clustering)."""
    hook = Hook(model, layer)
    variants = [("none", None), ("sham", None), ("full", None)] + [(f"{kind}{r}", r) for r in cfg["ranks"]
                                                                 for kind in ("proj", "comp", "rand")]
    res = {v: {"tgt": [], "coll": []} for v, _ in variants}
    base_preds = None
    n_coll_q = min(cfg["collateral_queries"], k - 1)
    gen = torch.Generator(device="cpu").manual_seed(cfg["seed"] + 991)
    for start in range(0, n, cfg["q1_batch"]):
        cs = [collection(rng, k, pool) for _ in range(min(cfg["q1_batch"], n - start))]
        pairs = [changed(rng, c) for c in cs]  # receiver = counterfactual, donor = original
        donor_blocks = [block(c, fam) for c in cs]
        recv_blocks = [block(c2, fam) for c2, _ in pairs]
        hd = states_at(model, tok, donor_blocks, layer)
        hr = states_at(model, tok, recv_blocks, layer)
        delta = hd - hr
        pos = close_pos(tok, recv_blocks)
        # queries: changed fact (targeted) + n_coll_q other facts
        qs, meta = [], []
        for ci, ((c2, i), c) in enumerate(zip(pairs, cs)):
            others = [j for j in range(k) if j != i]
            rng.shuffle(others)
            for j in [i] + others[:n_coll_q]:
                qs.append(recv_blocks[ci] + FAM[fam]["q"].format(n=c["names"][j]))
                meta.append((ci, j == i, c["colors"][j], c2["colors"][j]))
        idx = torch.tensor([m[0] for m in meta], device="cuda")
        for name, r in variants:
            if name == "none":
                dv = None
            elif name == "sham":
                z_ = torch.randn(delta.shape, generator=gen).cuda()
                dv = z_ / z_.norm(dim=-1, keepdim=True) * delta.norm(dim=-1, keepdim=True)
            elif name == "full":
                dv = delta
            else:
                kind = name[:4]
                if kind == "proj":
                    U = basis[:, :r]
                    dv = (delta @ U) @ U.T
                elif kind == "comp":
                    U = basis[:, :r]
                    dv = delta - (delta @ U) @ U.T
                else:
                    R = torch.linalg.qr(torch.randn(delta.shape[1], r, generator=gen).cuda())[0]
                    p = (delta @ R) @ R.T
                    pu = (delta @ basis[:, :r]) @ basis[:, :r].T
                    dv = p * (pu.norm(dim=-1, keepdim=True) / p.norm(dim=-1, keepdim=True).clamp_min(1e-6))
            if dv is None:
                preds = run_prompts(model, tok, qs, opt_ids)
            else:
                dvq, posq = dv[idx], pos[idx]
                preds = run_prompts(model, tok, qs, opt_ids, hook,
                                    lambda sl, enc: [(posq[sl], dvq[sl], "add")])
            if name == "none":
                base_preds = preds
            per_c_t, per_c_k = {}, {}
            for (ci, is_t, donor_col, recv_col), p, p0 in zip(meta, preds, base_preds):
                if is_t:
                    per_c_t[ci] = int(p == donor_col)
                else:
                    per_c_k.setdefault(ci, []).append(int(p != p0))  # collateral = paired answer flip vs unpatched
            res[name]["tgt"] += [per_c_t[ci] for ci in sorted(per_c_t)]
            res[name]["coll"] += [sum(v) / len(v) for _, v in sorted(per_c_k.items())]
        cfg["_out"].beat()
    hook.remove()
    return res


def fit_basis(model, tok, layer, rng, k, fam, pool, n):
    deltas = []
    for start in range(0, n, 64):
        cs = [collection(rng, k, pool) for _ in range(min(64, n - start))]
        hd = states_at(model, tok, [block(c, fam) for c in cs], layer)
        hr = states_at(model, tok, [block(changed(rng, c)[0], fam) for c in cs], layer)
        deltas.append(hd - hr)
    X = torch.cat(deltas)
    X = X - X.mean(0)
    U, S, V = torch.linalg.svd(X, full_matrices=False)
    ev = S ** 2
    pr = (ev.sum() ** 2 / (ev ** 2).sum()).item()
    return V.T.contiguous(), pr, (ev[:32] / ev.sum()).tolist()


def q2_cell(model, tok, opt_ids, layer, rng, cfg, k, fam, pool, n, m):
    hook = Hook(model, layer)
    cs = [collection(rng, k, pool) for _ in range(n)]
    blocks = [block(c, fam) for c in cs]
    full = run_prompts(model, tok, [b + FAM[fam]["q"].format(n=c["names"][j]) for b, c in zip(blocks, cs) for j in range(k)],
                       opt_ids)
    gold = [c["colors"][j] for c in cs for j in range(k)]
    st = states_at(model, tok, blocks, layer, positions=m)  # [n, m, D]
    assert len(tok(" ." * m, add_special_tokens=False)["input_ids"]) == m, "placeholder must be m tokens"
    recv = [" ." * m + FAM[fam]["q"].format(n=c["names"][j]) for c in cs for j in range(k)]
    stq = st.repeat_interleave(k, 0)

    def edits(states):
        return lambda sl, enc: [(torch.full((enc["input_ids"].shape[0],), j, device="cuda", dtype=torch.long),
                                 states[sl][:, j], "set") for j in range(m)]
    patched = run_prompts(model, tok, recv, opt_ids, hook, edits(stq))
    nopatch = run_prompts(model, tok, recv, opt_ids)
    shuf = run_prompts(model, tok, recv, opt_ids, hook, edits(st.roll(1, 0).repeat_interleave(k, 0)))
    # counterfactual donors
    pairs = [changed(rng, c) for c in cs]
    st2 = states_at(model, tok, [block(c2, fam) for c2, _ in pairs], layer, positions=m).repeat_interleave(k, 0)
    patched2 = run_prompts(model, tok, recv, opt_ids, hook, edits(st2))
    hook.remove()
    per_sel = [sum(int(patched[ci * k + j] == gold[ci * k + j]) for ci in range(n)) / n for j in range(k)]
    tgt, coll = [], []
    for ci, (c2, i) in enumerate(pairs):
        tgt.append(int(patched2[ci * k + i] == c2["colors"][i]))
        others = [int(patched2[ci * k + j] != patched[ci * k + j]) for j in range(k) if j != i]  # flip rate
        if others:
            coll.append(sum(others) / len(others))
    acc = lambda p: sum(int(a == b) for a, b in zip(p, gold)) / len(gold)
    return {"full_acc": acc(full), "patched_acc": acc(patched), "patched_min_selector": min(per_sel),
            "nopatch_acc": acc(nopatch), "shuffled_acc": acc(shuf), "targeted_follow": sum(tgt) / len(tgt),
            "nopatch_follow_changed": sum(int(nopatch[ci * k + i] == c2["colors"][i]) for ci, (c2, i) in enumerate(pairs)) / n,
            "collateral_flip": (sum(coll) / len(coll)) if coll else None, "n_collections": n,
            "constant_answer_baseline": max(gold.count(v) for v in range(4)) / len(gold),
            "paired": {"gold": gold, "patched": patched, "patched_cf": patched2}}


def main(cfg, out):
    cfg["_out"] = out
    model, tok, info = common.load(cfg["model"])
    out.meta("model_info.json", info)
    L = info["n_layers"]
    opt_ids = [tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in COLORS]
    assert len(set(opt_ids)) == 4
    for f_ in FAM:  # registered amendment: fact-independent suffix of >= max(8, max m) tokens after the last fact
        n_suffix = len(tok(FAM[f_]["close"], add_special_tokens=False)["input_ids"])
        assert n_suffix >= max(8, max(cfg.get("ladder", [1]))), f"suffix too short for {f_}: {n_suffix} tokens"
    out.meta("suffix_tokens.json", {f_: len(tok(FAM[f_]["close"], add_special_tokens=False)["input_ids"]) for f_ in FAM})
    pools = {"dev": NAMES[:20], "test": NAMES[20:]}
    rng = random.Random(cfg["seed"])
    # layer choice on dev by Q1 full-patch targeted effect (W1, k=4)
    key = "q|layer_choice"
    if key not in out.done:
        scores = {}
        for lf in cfg["layer_fracs"]:
            layer = int(round(lf * L))
            basis = torch.eye(info["d_model"], device="cuda")[:, :max(cfg["ranks"])]
            r_ = q1_cell(model, tok, opt_ids, layer, random.Random(f"dev{layer}"), {**cfg, "ranks": [1]}, 4, "W1",
                         pools["dev"], cfg["dev_n"], basis)
            scores[layer] = (sum(r_["full"]["tgt"]) / len(r_["full"]["tgt"])) - (sum(r_["none"]["tgt"]) / len(r_["none"]["tgt"]))
        out.write({"key": key, "scores": scores, "layer": max(scores, key=scores.get)})
    layer = [__import__("json").loads(l) for l in out.path.read_text().splitlines()
             if __import__("json").loads(l)["key"] == key][0]["layer"]
    for fam in cfg["families"]:
        if "q1" in cfg["parts"]:
            for k in cfg["q1_ks"]:
                key = f"q1|{fam}|k{k}"
                if key in out.done:
                    continue
                basis, pr, evf = fit_basis(model, tok, layer, random.Random(f"fit{fam}{k}"), k, fam, pools["dev"], cfg["fit_n"])
                r_ = q1_cell(model, tok, opt_ids, layer, random.Random(f"test{fam}{k}"), cfg, k, fam, pools["test"],
                             cfg["test_n"], basis)
                summ = {v: {"targeted_follow": sum(d["tgt"]) / len(d["tgt"]),
                            "collateral_flip": (sum(d["coll"]) / len(d["coll"])) if d["coll"] else None,
                            "per_collection_tgt": d["tgt"]} for v, d in r_.items()}
                sham = summ["sham"]["targeted_follow"]
                fulle = summ["full"]["targeted_follow"] - sham
                for v in summ:
                    summ[v]["retention_vs_full"] = ((summ[v]["targeted_follow"] - sham) / fulle) if fulle > 0 else None
                out.write({"key": key, "layer": layer, "family": fam, "k": k, "dev_delta_PR": pr,
                           "dev_delta_energy_top32": evf, "variants": summ})
        if "q2" in cfg["parts"]:
            for k in cfg["q2_ks"]:
                for m in cfg["ladder"]:
                    key = f"q2|{fam}|k{k}|m{m}"
                    if key in out.done:
                        continue
                    r_ = q2_cell(model, tok, opt_ids, layer, random.Random(f"q2{fam}{k}{m}"), cfg, k, fam, pools["test"],
                                 cfg["test_n"], m)
                    out.write({"key": key, "layer": layer, "family": fam, "k": k, "m": m, **r_})
                    out.beat()


def diag(cfg, out):
    """Non-claim diagnostic for Q1: delta norm ratio at the closing position, and a hook positive control
    (adding a random vector of 10x the hidden norm at the closing position must change some answers)."""
    model, tok, info = common.load(cfg["model"])
    L = info["n_layers"]
    opt_ids = [tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in COLORS]
    layer = int(round(cfg["layer_frac"] * L))
    rng = random.Random(5)
    for k in (1, 4):
        cs = [collection(rng, k, NAMES[:20]) for _ in range(64)]
        pairs = [changed(rng, c) for c in cs]
        db, rb = [block(c, "W1") for c in cs], [block(c2, "W1") for c2, _ in pairs]
        hd, hr = states_at(model, tok, db, layer), states_at(model, tok, rb, layer)
        ratio = ((hd - hr).norm(dim=-1) / hr.norm(dim=-1)).tolist()
        pos = close_pos(tok, rb)
        qs = [rb[i] + FAM["W1"]["q"].format(n=pairs[i][0]["names"][pairs[i][1]]) for i in range(len(cs))]
        base = run_prompts(model, tok, qs, opt_ids)
        hook = Hook(model, layer)
        big = torch.randn(len(qs), hr.shape[1], device="cuda")
        big = big / big.norm(dim=-1, keepdim=True) * hr.norm(dim=-1, keepdim=True) * 10
        pc = run_prompts(model, tok, qs, opt_ids, hook, lambda sl, enc: [(pos[sl], big[sl], "add")])
        # also: SET the closing state to the donor's (equivalent to full patch) and patch ALL positions of the
        # changed fact line via a full donor-state swap at every position (upper bound: everything from the donor)
        hook.remove()
        out.write({"key": f"diag|k{k}", "layer": layer, "delta_over_h_mean": sum(ratio) / len(ratio),
                   "delta_over_h_max": max(ratio), "hook_pc_flip_rate": sum(int(a != b) for a, b in zip(base, pc)) / len(qs)})


def value_positions(tok, text, c, fam):
    """Token positions (in `text`) of each fact's value word; asserts the block is an exact token prefix."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    pos = []
    tmpl = FAM[fam]["fact"]
    upto_value = tmpl[:tmpl.index("{c}") + 3]  # fact text up to and including the value word
    for j in range(len(c["names"])):
        prev = [tmpl.format(n=n, c=COLORS[v]) for n, v in zip(c["names"][:j], c["colors"][:j])]
        prefix = FAM[fam]["head"] + "".join(x + "\n" for x in prev) + upto_value.format(n=c["names"][j],
                                                                                      c=COLORS[c["colors"][j]])
        p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        assert ids[:len(p_ids)] == p_ids, "fact prefix is not a token prefix of the full prompt"
        pos.append(len(p_ids) - 1)  # the value token
    return pos


def validate(cfg, out):
    """Non-claim instrument validation on the frozen model (registration Q follow-up, Codex hunt): token alignment,
    post-hook residual change, logit change, flip rates for large perturbations at the closing position / at all
    positions, and a localized causal positive control (SET the changed fact's value-token residual to the donor's)."""
    model, tok, info = common.load(cfg["model"])
    out.meta("model_info.json", info)
    L = info["n_layers"]
    opt_ids = [tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in COLORS]
    rng = random.Random(7)
    for fam in ("W1", "W2"):
        for lf in cfg["layer_fracs"]:
            layer = int(round(lf * L))
            k = 4
            cs = [collection(rng, k, NAMES[:20]) for _ in range(cfg["n"])]
            pairs = [changed(rng, c) for c in cs]
            rb = [block(c2, fam) for c2, _ in pairs]
            db = [block(c, fam) for c in cs]
            qs = [rb[i] + FAM[fam]["q"].format(n=pairs[i][0]["names"][pairs[i][1]]) for i in range(len(cs))]
            # token alignment: closing position and value positions are token prefixes of the FULL prompt
            cpos, vpos, dvpos = [], [], []
            for i in range(len(cs)):
                full_ids = tok(qs[i], add_special_tokens=False)["input_ids"]
                b_ids = tok(rb[i], add_special_tokens=False)["input_ids"]
                assert full_ids[:len(b_ids)] == b_ids, "receiver block is not a token prefix of the full prompt"
                cpos.append(len(b_ids) - 1)
                vpos.append(value_positions(tok, qs[i], pairs[i][0], fam)[pairs[i][1]])
                dvpos.append(value_positions(tok, db[i], cs[i], fam)[pairs[i][1]])
            cpos_t = torch.tensor(cpos, device="cuda"); vpos_t = torch.tensor(vpos, device="cuda")
            base = run_prompts(model, tok, qs, opt_ids)
            hook = Hook(model, layer)
            hr = states_at(model, tok, rb, layer)
            big = torch.randn(len(qs), hr.shape[1], device="cuda")
            big = big / big.norm(dim=-1, keepdim=True) * hr.norm(dim=-1, keepdim=True) * 10
            # measured post-hook residual change at the hooked position (capture layer output)
            captured = {}

            def cap(mod, inp, o):
                captured["h"] = (o[0] if isinstance(o, tuple) else o).detach()
            hcap = model.model.layers[layer].register_forward_hook(cap)  # registered after Hook -> sees edited output
            enc = tok([qs[0]], return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.no_grad():
                hook.edits = None; model(**enc); h0 = captured["h"][0, cpos[0]].float().clone()
                l0 = model(**enc).logits[0, -1].float()
                hook.edits = [(cpos_t[:1], big[:1], "add")]; l1 = model(**enc).logits[0, -1].float()
                h1 = captured["h"][0, cpos[0]].float().clone(); hook.edits = None
            hcap.remove()
            pc_close = run_prompts(model, tok, qs, opt_ids, hook, lambda sl, enc: [(cpos_t[sl], big[sl], "add")])
            # all positions: add the big vector at every position up to the closing token
            pc_all = run_prompts(model, tok, qs, opt_ids, hook, lambda sl, enc: [
                (torch.full((enc["input_ids"].shape[0],), p_, device="cuda", dtype=torch.long), big[sl], "add")
                for p_ in range(0, int(cpos_t[sl].min().item()) + 1, 3)])
            # localized causal positive control: SET value-token residual to the donor's value-token residual
            dv_states = []
            tok.padding_side = "right"
            for i in range(0, len(db), 24):
                e = tok(db[i:i + 24], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    hs = model(**e, output_hidden_states=True).hidden_states[layer + 1].float()
                dv_states.append(hs[torch.arange(hs.shape[0]), torch.tensor(dvpos[i:i + 24], device="cuda")])
            dv_states = torch.cat(dv_states)
            vpatch = run_prompts(model, tok, qs, opt_ids, hook, lambda sl, enc: [(vpos_t[sl], dv_states[sl], "set")])
            hook.remove()
            follow = sum(int(p == cs[i]["colors"][pairs[i][1]]) for i, p in enumerate(vpatch)) / len(cs)
            base_follow = sum(int(p == cs[i]["colors"][pairs[i][1]]) for i, p in enumerate(base)) / len(cs)
            out.write({"key": f"validate|{fam}|l{layer}", "family": fam, "layer": layer, "n": len(cs),
                       "post_hook_residual_change_over_h": ((h1 - h0).norm() / h0.norm()).item(),
                       "max_abs_logit_change_close_add": (l1 - l0).abs().max().item(),
                       "flip_rate_close_add_10x": sum(int(a != b) for a, b in zip(base, pc_close)) / len(base),
                       "flip_rate_all_positions_add_10x": sum(int(a != b) for a, b in zip(base, pc_all)) / len(base),
                       "value_token_patch_donor_follow": follow, "unpatched_donor_follow": base_follow,
                       "base_acc_recv": sum(int(p == pairs[i][0]["colors"][pairs[i][1]]) for i, p in enumerate(base)) / len(base)})
            out.beat()


@torch.no_grad()
def states_at_pos(model, tok, texts, layer, positions, bs=16):
    """Layer-`layer` residual of each text at its given token position."""
    tok.padding_side = "right"
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        hs = model(**enc, output_hidden_states=True).hidden_states[layer + 1].float()
        out.append(hs[torch.arange(hs.shape[0], device=hs.device),
                      torch.tensor(positions[i:i + bs], device=hs.device)].to("cuda"))
    return torch.cat(out)


def q1p_pairs(tok, rng, k, fam, pool, n):
    """Donor/receiver blocks differing in one fact, with the changed fact's value-token position (asserted equal)."""
    rows = []
    while len(rows) < n:
        c = collection(rng, k, pool)
        c2, i = changed(rng, c)
        db, rb = block(c, fam), block(c2, fam)
        pd, pr = value_positions(tok, db, c, fam)[i], value_positions(tok, rb, c2, fam)[i]
        if pd != pr:  # multi-token values would misalign positions: reject
            continue
        rows.append((c, c2, i, db, rb, pr))
    return rows


def q1p_cell(model, tok, opt_ids, layer, rng, cfg, k, fam, pool, n, basis):
    hook = Hook(model, layer)
    gen = torch.Generator(device="cpu").manual_seed(cfg["seed"] + 4242)
    variants = [("none", None), ("sham", None), ("full", None)] + [(f"{kd}{r}", r) for r in cfg["ranks"]
                                                                 for kd in ("proj", "comp")] + \
        [(f"rand{r}s{s_}", r) for r in cfg["ranks"] for s_ in range(cfg.get("rand_seeds", 5))]
    frozen_R = {}  # frozen random subspaces (fixed before test), one per (rank, seed)
    D_ = basis.shape[0]
    for r in cfg["ranks"]:
        for s_ in range(cfg.get("rand_seeds", 5)):
            g_ = torch.Generator(device="cpu").manual_seed(10_000 + 97 * r + s_)
            frozen_R[(r, s_)] = torch.linalg.qr(torch.randn(D_, r, generator=g_))[0].cuda()
    res = {v: {"tgt": [], "coll": [], "coll_pairs": [], "tgt_pred": []} for v, _ in variants}
    recv_gold = []
    absorbed = []
    n_coll_q = min(cfg["collateral_queries"], k - 1)
    for start in range(0, n, cfg["q1_batch"]):
        rows = q1p_pairs(tok, rng, k, fam, pool, min(cfg["q1_batch"], n - start))
        hd = states_at_pos(model, tok, [r[3] for r in rows], layer, [r[5] for r in rows])
        hr = states_at_pos(model, tok, [r[4] for r in rows], layer, [r[5] for r in rows])
        delta = hd - hr
        qs, meta = [], []
        for ri, (c, c2, i, db, rb, p) in enumerate(rows):
            others = [j for j in range(k) if j != i]
            rng.shuffle(others)
            for j in [i] + others[:n_coll_q]:
                qs.append(rb + FAM[fam]["q"].format(n=c["names"][j]))
                meta.append((ri, j == i, c["colors"][j]))
            recv_gold.append(c2["colors"][i])
        for q_, m_ in zip(qs, meta):  # value token is an exact token prefix position of the FULL query prompt
            b_ids = tok(rows[m_[0]][4], add_special_tokens=False)["input_ids"]
            assert tok(q_, add_special_tokens=False)["input_ids"][:rows[m_[0]][5] + 1] == b_ids[:rows[m_[0]][5] + 1]
        idx = torch.tensor([m_[0] for m_ in meta], device="cuda")
        pos_q = torch.tensor([rows[m_[0]][5] for m_ in meta], device="cuda")
        base_preds = None
        for name, r in variants:
            if name == "none":
                dv = None
            elif name == "sham":
                z_ = torch.randn(delta.shape, generator=gen).cuda()
                dv = z_ / z_.norm(dim=-1, keepdim=True) * delta.norm(dim=-1, keepdim=True)
            elif name == "full":
                dv = delta
            else:
                kd = name[:4]
                U = basis[:, :r]
                if kd == "proj":
                    dv = (delta @ U) @ U.T
                elif kd == "comp":
                    dv = delta - (delta @ U) @ U.T
                else:
                    R = frozen_R[(r, int(name.split("s")[-1]))]
                    p_ = (delta @ R) @ R.T
                    pu = (delta @ U) @ U.T
                    dv = p_ * (pu.norm(dim=-1, keepdim=True) / p_.norm(dim=-1, keepdim=True).clamp_min(1e-6))
            if name == "full":  # bf16 absorption: fraction of the patch lost when added to the bf16 residual
                hb = hr.to(torch.bfloat16)
                absorbed += (((hb + dv.to(torch.bfloat16)).float() - hb.float() - dv).norm(dim=-1)
                             / dv.norm(dim=-1).clamp_min(1e-9)).tolist()
            if dv is None:
                preds = run_prompts(model, tok, qs, opt_ids)
                base_preds = preds
            else:
                dvq = dv[idx]
                preds = run_prompts(model, tok, qs, opt_ids, hook, lambda sl, enc: [(pos_q[sl], dvq[sl], "add")])
            t_, c_, tp_ = {}, {}, {}
            for (ri, is_t, donor_col), pr_, p0 in zip(meta, preds, base_preds):
                if is_t:
                    t_[ri] = int(pr_ == donor_col)
                    tp_[ri] = pr_
                else:
                    c_.setdefault(ri, []).append(int(pr_ != p0))
            res[name]["tgt"] += [t_[x] for x in sorted(t_)]
            res[name]["tgt_pred"] += [tp_[x] for x in sorted(tp_)]
            res[name]["coll"] += [sum(v) / len(v) for _, v in sorted(c_.items())]
            res[name]["coll_pairs"] += [v for _, v in sorted(c_.items())]
        cfg["_out"].beat()
    hook.remove()
    res["_absorbed_mean"] = sum(absorbed) / len(absorbed) if absorbed else None
    res["_recv_gold"] = recv_gold
    return res


def fit_basis_value(model, tok, layer, rng, k, fam, pool, n):
    rows = q1p_pairs(tok, rng, k, fam, pool, n)
    X = torch.cat([states_at_pos(model, tok, [r[3] for r in rows[i:i + 64]], layer, [r[5] for r in rows[i:i + 64]])
                   - states_at_pos(model, tok, [r[4] for r in rows[i:i + 64]], layer, [r[5] for r in rows[i:i + 64]])
                   for i in range(0, len(rows), 64)])
    Xc = X - X.mean(0)
    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    ev = S ** 2
    return Vh.T.contiguous(), (ev.sum() ** 2 / (ev ** 2).sum()).item(), (ev[:64] / ev.sum()).tolist()


def q1prime(cfg, out):
    """Registration Q1-prime main entry (value-token causal effective dimension)."""
    cfg["_out"] = out
    model, tok, info = common.load(cfg["model"], device_map=cfg.get("device_map", "cuda"))
    out.meta("model_info.json", info)
    L = info["n_layers"]
    layer = int(round(cfg["layer_frac"] * L))
    opt_ids = [tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in COLORS]
    assert len(set(opt_ids)) == 4
    pools = {"dev": NAMES[:20], "test": NAMES[20:]}
    for fam in cfg["families"]:
        for k in cfg["ks"]:
            key = f"q1p|{fam}|k{k}"
            if key in out.done:
                continue
            basis, pr, evf = fit_basis_value(model, tok, layer, random.Random(f"fitv{fam}{k}"), k, fam, pools["dev"],
                                             cfg["fit_n"])
            r_ = q1p_cell(model, tok, opt_ids, layer, random.Random(f"testv{fam}{k}"), cfg, k, fam, pools[cfg["split"]],
                          cfg["test_n"], basis)
            absorbed = r_.pop("_absorbed_mean")
            gold = r_.pop("_recv_gold")
            recv_acc = sum(int(a == b) for a, b in zip(r_["none"]["tgt_pred"], gold)) / len(gold)
            summ = {v: {"targeted_follow": sum(d["tgt"]) / len(d["tgt"]),
                        "collateral_flip": (sum(d["coll"]) / len(d["coll"])) if d["coll"] else None,
                        "per_collection_tgt": d["tgt"], "per_collection_collateral": d["coll_pairs"],
                        "tgt_pred": d["tgt_pred"], "option_freq": [d["tgt_pred"].count(o) / len(d["tgt_pred"])
                                                                   for o in range(len(opt_ids))]}
                    for v, d in r_.items()}
            sham = summ["sham"]["targeted_follow"]
            fulle = summ["full"]["targeted_follow"] - sham
            for v in summ:
                summ[v]["retention_vs_full"] = ((summ[v]["targeted_follow"] - sham) / fulle) if fulle > 0 else None
            nseed = cfg.get("rand_seeds", 5)
            rand_ret = {r: [summ[f"rand{r}s{s_}"]["retention_vs_full"] for s_ in range(nseed)] for r in cfg["ranks"]}
            gate = (summ["full"]["targeted_follow"] >= 0.80 and fulle >= 0.80 and
                    (k == 1 or (summ["full"]["collateral_flip"] or 0) <= 0.05))
            r90 = None
            if gate:
                hit = [r for r in cfg["ranks"] if (summ["proj" + str(r)]["retention_vs_full"] or 0) >= 0.90]
                r90 = hit[0] if hit else "censored_above_" + str(max(cfg["ranks"]))
            out.write({"key": key, "model": cfg["model"], "d_model": info["d_model"], "layer": layer, "family": fam,
                       "k": k, "split": cfg["split"], "n": cfg["test_n"], "dev_value_delta_PR": pr,
                       "dev_energy_top64": evf, "gate_pass": gate, "r90": r90,
                       "full_follow": summ["full"]["targeted_follow"], "gate_full_targeted_minus_sham": fulle,
                       "bf16_absorbed_frac": absorbed, "receiver_unpatched_acc": recv_acc, "rand_retention_by_rank": rand_ret, "equivalence_margin": 0.10,
                       "variants": summ, "tokenizer_revision": info.get("tokenizer_revision"),
                       "model_revision": info.get("revision")})
            if cfg.get("smoke") and not gate:
                out.write({"key": "smoke_fail|" + fam + "|k" + str(k), "note": "instrument gate failed; no full sweep"})
