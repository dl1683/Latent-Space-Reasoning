"""Flagship + admission gate: causal, selectable latent branch capacity (theory/LATENT_BRANCH_CAPACITY.md).

k labelled DAGs (16 edge bits each) are encoded from text into ONE state z; the prompt and its KV cache are then
discarded. Each update is a FRESH forward context [state token, instruction] -> new z. The answer comes from a
fresh context [state token, query] through a binary head. Only z crosses between contexts (by construction:
every context is a separate forward call on inputs_embeds with use_cache=False).

variant="state"  : updates pass through the state (the flagship).
variant="storage": updates are NOT applied to the state; they are appended as text to the answer context
                   (storage / delayed-computation control, same parameters, data and allowance).
"""
import json
import math
import random

import torch
import torch.nn.functional as F

from . import common

# node ids: s=0, a=(1,2), b=(3,4), c=(5,6), d=(7,8), t=9; edge index order is lexicographic within groups
EDGES = [(0, 1), (0, 2)] + [(x, y) for x in (1, 2) for y in (3, 4)] + [(x, y) for x in (3, 4) for y in (5, 6)] + \
        [(x, y) for x in (5, 6) for y in (7, 8)] + [(7, 9), (8, 9)]
assert len(EDGES) == 16

_pairs = [(a, b) for a in range(16) for b in range(a + 1, 16)]
random.Random(20261011).shuffle(_pairs)
PAIRS = {"train": _pairs[:72], "dev": _pairs[72:96], "test": _pairs[96:]}


def reachable(bits):
    seen = {0}
    for (a, b), on in zip(EDGES, bits):  # topological order
        if on and a in seen:
            seen.add(b)
    return 9 in seen


def apply(graphs, ops):
    """ops: list of (edge, target) with target None = every graph, else branch index."""
    gs = [g[:] for g in graphs]
    for e, tgt in ops:
        for i in range(len(gs)):
            if tgt is None or tgt == i:
                gs[i][e] ^= 1
    return gs


def answers(ex):
    return [reachable(g) for g in apply(ex["graphs"], ex["ops"])]


QUERY = {"mode": "reach"}  # "reach": selector = branch j (reachability); "edge": selector = 16*j + e (edge bit, P1)


def n_sel(k):
    return k if QUERY["mode"] == "reach" else 16 * k


def branch_of(s_):
    return s_ if QUERY["mode"] == "reach" else s_ // 16


def gold_of(ex, s_):
    if QUERY["mode"] == "reach":
        return answers(ex)[s_]
    return bool(apply(ex["graphs"], ex["ops"])[s_ // 16][s_ % 16])


def query_text(s_):
    if QUERY["mode"] == "reach":
        return f"Graph G{s_ + 1}: is the sink reachable?"
    return f"Graph G{s_ // 16 + 1}: is edge {s_ % 16} present?"


def is_test_pattern(bits):
    return (int("".join(map(str, bits)), 2) * 2654435761 % 2 ** 32) % 5 == 0  # 20% of 16-bit patterns: test only


def sample_example(r, k, u, split, addressed_frac=0.5):
    if QUERY["mode"] == "edge":  # P1: Bernoulli(1/2) bits, train/test graph patterns disjoint, no balancing needed
        pair = list(r.choice(PAIRS[split]))
        r.shuffle(pair)
        ops = [(pair[i], (r.randrange(k) if (k > 1 and r.random() < addressed_frac) else None)) for i in range(u)]
        graphs = []
        while len(graphs) < k:
            g = [int(r.random() < 0.5) for _ in range(16)]
            if is_test_pattern(g) == (split == "test"):
                graphs.append(g)
        return {"graphs": graphs, "ops": ops, "k": k}
    """u update ops drawn from this split's edge pairs; each op global or branch-addressed (50/50).
    Each graph is rejection-sampled so its final answer is balanced (target drawn uniformly per graph)."""
    pair = list(r.choice(PAIRS[split]))
    r.shuffle(pair)
    ops = [(pair[i], (r.randrange(k) if (k > 1 and r.random() < addressed_frac) else None)) for i in range(u)]
    graphs = []
    for i in range(k):
        want = r.random() < 0.5
        for _ in range(500):
            g = [int(r.random() < 0.5) for _ in range(16)]
            if reachable(apply([g], [(e, None if t is None else (0 if t == i else -1)) for e, t in ops])[0]) == want:
                break
        graphs.append(g)
    return {"graphs": graphs, "ops": ops, "k": k}


def encode_text(ex):
    return "Graphs (16 edge bits each):\n" + "\n".join(
        f"G{i + 1}: {''.join(map(str, g))}" for i, g in enumerate(ex["graphs"])) + "\nState:"


def op_text(op):
    e, t = op
    return f"Toggle edge {e} in every graph." if t is None else f"Toggle edge {e} in graph G{t + 1}."


def sham_text(op):
    e, t = op
    return "Keep every edge in every graph." if t is None else f"Keep every edge in graph G{t + 1}."


class BranchModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        self.variant = cfg.get("variant", "state")
        self.lm, self.tok, self.info = common.load(cfg["model"])
        self.tok.padding_side = "right"
        self.lm.config.use_cache = False
        if cfg.get("grad_ckpt", True):
            self.lm.gradient_checkpointing_enable()
        self.lm.enable_input_require_grads()
        self.lm = get_peft_model(self.lm, LoraConfig(r=cfg.get("lora_r", 8), lora_alpha=2 * cfg.get("lora_r", 8),
                                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                     "gate_proj", "up_proj", "down_proj"]))
        D, S = self.info["d_model"], cfg["state_dim"]
        self.S = S
        g = torch.Generator().manual_seed(cfg["seed"])
        Q, _ = torch.linalg.qr(torch.randn(S, S, generator=g))
        self.register_buffer("Q", Q.cuda())
        self.E = torch.nn.Linear(D, S).cuda()
        self.D = torch.nn.Linear(S, D).cuda()
        self.head = torch.nn.Linear(D, 1).cuda()
        self.emb = self.lm.get_input_embeddings()

    def base(self):
        return self.lm.get_base_model().model

    def last_hidden(self, embeds, am):
        hs = self.base()(inputs_embeds=embeds, attention_mask=am, use_cache=False).last_hidden_state
        idx = am.sum(1) - 1  # right padding: last real position
        return hs[torch.arange(hs.shape[0], device=hs.device), idx]

    def compress(self, h, r):
        y = self.E(h.float()) @ self.Q.T  # rotate into the fixed seed basis
        y = y[:, :r]
        y = y / (y.norm(dim=-1, keepdim=True) + 1e-6)  # unit norm at every width
        z = F.pad(y, (0, self.S - r))
        return quantize(z, r, self.qbits_pass) if getattr(self, "qbits_pass", None) else z

    def expand(self, z):
        return self.D(z @ self.Q)

    def ctx(self, z, texts):
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        e = self.emb(enc["input_ids"])
        embeds = torch.cat([self.expand(z).to(e.dtype)[:, None, :], e], 1)
        am = torch.cat([torch.ones_like(enc["attention_mask"][:, :1]), enc["attention_mask"]], 1)
        return self.last_hidden(embeds, am)

    def encode(self, exs, r):
        if self.variant == "text":  # positive control: no bottleneck; the answer context sees the full text
            return torch.zeros(len(exs), self.S, device="cuda")
        enc = self.tok([encode_text(x) for x in exs], return_tensors="pt", padding=True,
                       add_special_tokens=False).to("cuda")
        return self.compress(self.last_hidden(self.emb(enc["input_ids"]), enc["attention_mask"]), r)

    def step(self, z, op_texts, r):
        return self.compress(self.ctx(z, op_texts), r)

    def advance(self, z, exs, r, sham=False):
        """Apply each example's ops in order through the state (variant=state). Storage variant: no-op."""
        if self.variant in ("storage", "text"):
            return z
        for s in range(max(len(x["ops"]) for x in exs)):
            idx = [i for i, x in enumerate(exs) if len(x["ops"]) > s]  # ragged: rows with an s-th op
            zn = self.step(z[idx], [(sham_text if sham else op_text)(exs[i]["ops"][s]) for i in idx], r)
            z = z.index_copy(0, torch.tensor(idx, device=z.device), zn.to(z.dtype))
        return z

    def answer(self, z, exs, js):
        if self.variant == "text":
            texts = [encode_text(x) + " " + " ".join(op_text(o) for o in x["ops"]) + f" {query_text(j)}"
                     for x, j in zip(exs, js)]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            return self.head(self.last_hidden(self.emb(enc["input_ids"]), enc["attention_mask"]).float()).squeeze(-1)
        if self.variant == "storage":
            pre = [" ".join(op_text(o) for o in x["ops"]) + " " if x["ops"] else "" for x in exs]
        else:
            pre = [""] * len(exs)
        h = self.ctx(z, [f"{p}{query_text(j)}" for p, j in zip(pre, js)])
        return self.head(h.float()).squeeze(-1)

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]


def collection(seed, i, split, ks, u_choices):
    r = random.Random(f"{seed}|{split}|{i}")
    k = r.choice(ks)
    return sample_example(r, k, r.choice(u_choices), split)


def forward_loss(m, exs, r_w, n_ans, rng):
    z = m.advance(m.encode(exs, r_w), exs, r_w)
    rep, js, gold = [], [], []
    for x in exs:
        pick = rng.sample(range(n_sel(x["k"])), min(n_sel(x["k"]), n_ans))
        rep += [x] * len(pick); js += pick; gold += [gold_of(x, p) for p in pick]
    zz = z.repeat_interleave(torch.tensor([min(n_sel(x["k"]), n_ans) for x in exs], device="cuda"), 0)
    logit = m.answer(zz, rep, js)
    return F.binary_cross_entropy_with_logits(logit, torch.tensor(gold, device="cuda", dtype=torch.float))


@torch.no_grad()
def quantize(z, r_w, bits):
    """Eval-only precision arm (Amendment 1). Fixed, sample-independent rule declared before any data:
    symmetric uniform quantizer on the active r coordinates, clip at c = 4/sqrt(r) (z is unit-norm, so a
    coordinate has rms 1/sqrt(r)), 2^bits - 1 levels, round-to-nearest; inactive coordinates stay exactly zero;
    no renormalization after quantization. Metadata cost: none (scale is a function of r only)."""
    if not bits:
        return z
    c = 4.0 / math.sqrt(r_w)
    lv = 2 ** (bits - 1) - 1 if bits > 1 else 1
    a = z[:, :r_w].clamp(-c, c)
    return F.pad(torch.round(a / c * lv) * c / lv, (0, z.shape[1] - r_w))


def acc_cell(m, cfg, split, r_w, u, k, n, seed, bits=None, qmode="read"):
    """Balanced per-branch accuracy with selectors exhausted; custody controls on the same collections."""
    rng = random.Random(f"eval|{seed}|{split}|{u}|{k}")
    ok = ok0 = oksh = okshuf = tot = 0
    un_ok = un_tot = ad_ok = ad_tot = 0
    S = n_sel(k)
    pos_ok = [0] * S
    un_pos_ok, un_pos_n = [0] * S, [0] * S
    jointok = 0
    zs = []
    bs = max(1, min(cfg["eval_tokens"], 8000) // (k * 22 + 40))  # hard cap: eval OOM at 32k tokens on 80 GB
    done = 0
    while done < n:
        torch.cuda.empty_cache()
        exs = [sample_example(rng, k, u, split) for _ in range(min(bs, n - done))]
        done += len(exs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m.qbits_pass = bits if qmode == "pass" else None
            z0 = m.encode(exs, r_w)
            z = m.advance(z0, exs, r_w)
            m.qbits_pass = None
            z = quantize(z, r_w, bits) if qmode == "read" else z
            zsham = m.advance(z0, exs, r_w, sham=True) if (u > 0 and m.variant == "state" and not bits) else None
            rep = [x for x in exs for _ in range(S)]
            js = [j for _ in exs for j in range(S)]
            gold = torch.tensor([gold_of(x, j) for x in exs for j in range(S)], device="cuda")
            for zz, name in ((z, "main"), (torch.zeros_like(z), "zero"), (z.roll(1, 0), "shuf"), (zsham, "sham")):
                if zz is None:
                    continue
                pred = []
                for b in range(0, len(rep), 512):
                    zr = zz.repeat_interleave(S, 0)[b:b + 512]
                    pred.append(m.answer(zr, rep[b:b + 512], js[b:b + 512]) > 0)
                pred = torch.cat(pred)
                c = (pred == gold)
                if name == "main":
                    ok += c.sum().item(); tot += len(c)
                    jointok += c.view(len(exs), S).all(1).sum().item()
                    cv = c.view(len(exs), S)
                    for j in range(S):
                        pos_ok[j] += cv[:, j].sum().item()
                    addr = torch.tensor([[any(t == branch_of(j) for _, t in x["ops"]) for j in range(S)] for x in exs], device="cuda")
                    un_ok += cv[~addr].sum().item(); un_tot += (~addr).sum().item()
                    for j in range(S):
                        un_pos_ok[j] += cv[:, j][~addr[:, j]].sum().item(); un_pos_n[j] += (~addr[:, j]).sum().item()
                    ad_ok += cv[addr].sum().item(); ad_tot += addr.sum().item()
                elif name == "zero":
                    ok0 += c.sum().item()
                elif name == "shuf":
                    okshuf += c.sum().item()
                else:
                    oksh += c.sum().item()
        zs.append(z[:, :r_w].float().cpu())
    return {"acc": ok / tot, "acc_unaddressed": un_ok / max(1, un_tot), "acc_addressed": (ad_ok / ad_tot) if ad_tot else None,
            "acc_by_selector": [x / n for x in pos_ok], "min_selector_acc": min(pos_ok) / n,
            "unaddressed_acc_by_selector": [a / b if b else None for a, b in zip(un_pos_ok, un_pos_n)],
            "min_unaddressed_selector_acc": min(a / b for a, b in zip(un_pos_ok, un_pos_n) if b) if any(un_pos_n) else None,
            "joint_acc": jointok / n, "acc_zero": ok0 / tot, "acc_shuffled": okshuf / tot,
            "acc_sham_updates": (oksh / tot) if (zsham is not None) else None, "n_collections": n, "bits": bits,
            "qmode": qmode if bits else None}


@torch.no_grad()
def causal_tests(m, cfg, split, r_w, k, n, seed):
    """Branch-addressed edit / undo, then two further global updates (edit-then-advance, undo-then-advance).
    Edits are oracle-certified answer-flip witnesses on the edited branch (declared evaluation stratification)."""
    rng = random.Random(f"causal|{seed}|{split}|{k}")
    agg = {k_: 0.0 for k_ in ("tgt_follow", "tgt_base", "coll_before", "coll_after", "undo_tgt", "undo_coll",
                              "sham_tgt", "adv_edit_tgt", "adv_undo_tgt", "adv_edit_coll", "n_coll")}
    cnt = 0
    while cnt < n:
        exs, edits = [], []
        while len(exs) < 16:
            x = sample_example(rng, k, 2, split, addressed_frac=0.0)  # two global updates (advance phase)
            b = rng.randrange(k)
            es = [e for e in range(16) if reachable(apply([x["graphs"][b]], [(e, None)])[0]) != reachable(x["graphs"][b])]
            if not es:
                continue
            exs.append(x); edits.append((rng.choice(es), b))
        B = len(exs)
        js_t = [b for _, b in edits]
        others = [[j for j in range(k) if j != b] for _, b in edits]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z0 = m.encode(exs, r_w)
            ze = m.step(z0, [op_text(e) for e in edits], r_w)
            zs = m.step(z0, [sham_text(e) for e in edits], r_w)
            zu = m.step(ze, [op_text(e) for e in edits], r_w)

            def q(z, xs_graph_ops, js):
                return (m.answer(z, xs_graph_ops, js) > 0)

            base_ex = [{"graphs": x["graphs"], "ops": [], "k": k} for x in exs]
            gold0 = torch.tensor([reachable(x["graphs"][b]) for x, b in zip(exs, js_t)], device="cuda")
            p_base = q(z0, base_ex, js_t)
            p_edit = q(ze, base_ex, js_t)
            p_sham = q(zs, base_ex, js_t)
            p_undo = q(zu, base_ex, js_t)
            agg["tgt_base"] += (p_base == gold0).sum().item()
            agg["tgt_follow"] += (p_edit == ~gold0).sum().item()
            agg["sham_tgt"] += (p_sham == gold0).sum().item()
            agg["undo_tgt"] += (p_undo == gold0).sum().item()
            # collateral: every other branch, before vs after the edit, and after undo
            rep = [be for be, o in zip(base_ex, others) for _ in o]
            jo = [j for o in others for j in o]
            go = torch.tensor([reachable(x["graphs"][j]) for x, o in zip(exs, others) for j in o], device="cuda")
            idx = torch.tensor([i for i, o in enumerate(others) for _ in o], device="cuda")
            agg["coll_before"] += (q(z0[idx], rep, jo) == go).float().mean().item() * B
            agg["coll_after"] += (q(ze[idx], rep, jo) == go).float().mean().item() * B
            agg["undo_coll"] += (q(zu[idx], rep, jo) == go).float().mean().item() * B
            # advance: two global updates after edit / after undo
            ge = [{"graphs": apply(x["graphs"], [e]), "ops": x["ops"], "k": k} for x, e in zip(exs, edits)]
            za = m.advance(ze, exs, r_w)
            zua = m.advance(zu, exs, r_w)
            gold_e = torch.tensor([answers(g)[b] for g, b in zip(ge, js_t)], device="cuda")
            gold_u = torch.tensor([answers(x)[b] for x, b in zip(exs, js_t)], device="cuda")
            agg["adv_edit_tgt"] += (q(za, base_ex, js_t) == gold_e).sum().item()
            agg["adv_undo_tgt"] += (q(zua, base_ex, js_t) == gold_u).sum().item()
            go2 = torch.tensor([answers(g)[j] for g, o in zip(ge, others) for j in o], device="cuda")
            agg["adv_edit_coll"] += (q(za[idx], rep, jo) == go2).float().mean().item() * B
        cnt += B
    res = {k_: v / cnt for k_, v in agg.items() if k_ != "n_coll"}
    res["targeted_effect_pp"] = 100 * (res["tgt_follow"] - (1 - res["tgt_base"]))
    res["collateral_loss_pp"] = 100 * (res["coll_before"] - res["coll_after"])
    induced = res["tgt_base"] - (1 - res["tgt_follow"])
    res["undo_recovery"] = (res["undo_tgt"] - (1 - res["tgt_follow"])) / induced if induced > 0 else None
    res["n"] = cnt
    return res


@torch.no_grad()
def pr_stats(X):
    """Sample PR, Gaussian bias-corrected PR (E tr S^2 = (1+1/n) tr Sig^2 + (tr Sig)^2/n), and the isotropic
    null PR at the same (n, r)."""
    n, r = X.shape
    X = X - X.mean(0)
    C = X.T @ X / (n - 1)
    t1, t2 = torch.trace(C).item(), (C @ C).trace().item()
    t2c = (t2 - t1 ** 2 / n) / (1 + 1 / n)
    G = torch.randn(n, r)
    G = G - G.mean(0)
    Cg = G.T @ G / (n - 1)
    return {"PR": t1 ** 2 / t2, "PR_bias_corrected": t1 ** 2 / t2c if t2c > 0 else None,
            "PR_isotropic_null": (torch.trace(Cg) ** 2 / (Cg @ Cg).trace()).item(), "PR_over_r": t1 ** 2 / t2 / r,
            "top_pc_share": torch.linalg.eigvalsh(C)[-1].item() / t1, "n": n}


def branch_pr(m, r_w, n=None, seed=0, k=4, u=2):
    """PR of branch-variation state differences: pairs differing in one graph, k=4, two global updates.
    Amendment 1: 8,192 independent pairs at every width, paired across widths (same seeded collections);
    centered uncorrected PR is primary; bias-corrected, isotropic-null and top-PC-removed are diagnostics."""
    n = n or 8192
    rng = random.Random(f"pr|{seed}")
    diffs = []
    while len(diffs) * 16 < n:
        A = [sample_example(rng, k, u, "test", addressed_frac=0.0) for _ in range(16)]
        Bx = json.loads(json.dumps(A))
        for b in Bx:
            b["graphs"][rng.randrange(k)] = sample_example(rng, 1, 0, "test")["graphs"][0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d = m.advance(m.encode(Bx, r_w), Bx, r_w) - m.advance(m.encode(A, r_w), A, r_w)
        diffs.append(d[:, :r_w].float().cpu())
    X = torch.cat(diffs)
    out = pr_stats(X)
    Xc = X - X.mean(0)
    out["kurtosis_mean"] = ((Xc ** 4).mean(0) / (Xc ** 2).mean(0) ** 2).mean().item()
    Xr = Xc - (Xc @ torch.linalg.eigh(Xc.T @ Xc)[1][:, -1:]) @ torch.linalg.eigh(Xc.T @ Xc)[1][:, -1:].T
    out["PR_top_pc_removed"] = pr_stats(Xr)["PR"]
    return out


def save(m, opt, step, path):
    torch.save({"model": {n: p.detach().cpu() for n, p in m.state_dict().items()
                          if "lora" in n or n.split(".")[0] in ("E", "D", "head")},
                "opt": opt.state_dict(), "step": step}, path)


def main(cfg, out):
    QUERY["mode"] = cfg.get("query", "reach")
    torch.manual_seed(cfg["seed"])
    m = BranchModel(cfg)
    out.meta("model_info.json", m.info)
    ckpt = out.dir / "ckpt.pt"
    opt = torch.optim.AdamW(m.trainable(), lr=cfg["lr"], weight_decay=0.0)
    step0 = 0
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cuda")
        m.load_state_dict({k: v.cuda() for k, v in st["model"].items()}, strict=False)
        opt.load_state_dict(st["opt"])
        step0 = st["step"]
    if cfg.get("init_ckpt"):  # eval-only: evaluate a trained (dev-selected) checkpoint with the current code
        import subprocess
        subprocess.run(["gcloud", "storage", "cp", cfg["init_ckpt"], str(out.dir / "best.pt")], check=True)
        step0 = total = 0
    ks = cfg["train_ks"]
    pool, eff_bs = cfg["pool"], cfg["batch"]
    total = 0 if cfg.get("init_ckpt") else cfg["passes"] * pool // eff_bs
    sched = lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / total))
    rng = random.Random(cfg["seed"] * 7919)
    order = list(range(pool))
    t = common.timer()
    hist, best = [], -1.0
    m.train()
    for step in range(step0, total):
        p, off = divmod(step * eff_bs, pool)
        if off == 0 or step == step0:
            order = list(range(pool)); random.Random(cfg["seed"] * 31 + p).shuffle(order)
        ids = order[off:off + eff_bs]
        cur = cfg.get("curriculum")  # Amendment 2: staged arm, [[start_frac, ks, updates], ...]
        if cur:
            stage = [c for c in cur if step / total >= c[0]][-1]
            exs = [collection(cfg["seed"] + 1000, step * eff_bs + j, "train", stage[1], stage[2]) for j in range(eff_bs)]
        else:
            exs = [collection(cfg["seed"], i, "train", ks, cfg["train_updates"]) for i in ids]
        r_w = rng.choice(cfg["widths"])
        for g in opt.param_groups:
            g["lr"] = cfg["lr"] * sched(step)
        opt.zero_grad(set_to_none=True)
        # micro-batches grouped by k (equal prompt lengths), token-budgeted; loss averaged over the batch
        exs.sort(key=lambda x: x["k"])
        mb, lsum = [], 0.0
        groups, cur, cur_tok = [], [], 0
        for x in exs:
            tl = x["k"] * 22 + 40
            if cur and cur_tok + tl > cfg["train_tokens"]:
                groups.append(cur); cur, cur_tok = [], 0
            cur.append(x); cur_tok += tl
        groups.append(cur)
        for gexs in groups:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = forward_loss(m, gexs, r_w, cfg["answers_per_ex"], rng) * len(gexs) / len(exs)
            loss.backward()
            lsum += loss.item()
        torch.nn.utils.clip_grad_norm_(m.trainable(), 1.0)
        opt.step()
        hist.append(lsum)
        if step % 25 == 0:
            out.beat()
            out.meta("train_progress.json", {"step": step, "total": total, "loss25": sum(hist[-25:]) / len(hist[-25:]),
                                             "sec": t()})
        if (step + 1) % cfg["eval_every"] == 0 or step == total - 1:
            # dev selection: mean balanced accuracy over a fixed small dev grid
            cells = {f"r{r}|u{u}|k{k}": acc_cell(m, cfg, "dev", r, u, k, cfg["dev_n"], 0)["acc"]
                     for r in cfg["widths"] for u in cfg.get("dev_updates", (0, 1, 2)) for k in cfg["dev_ks"]}
            uu = max(cfg.get("dev_updates", (0, 1, 2)))
            score = sum(v for c, v in cells.items() if f"|u{uu}|" in c) / sum(1 for c in cells if f"|u{uu}|" in c)
            out.write({"key": f"dev|step{step + 1}", "step": step + 1, "dev_score": score, "dev_cells": cells,
                       "loss_recent": sum(hist[-cfg["eval_every"]:]) / len(hist[-cfg["eval_every"]:]), "sec": t()})
            save(m, opt, step + 1, ckpt)
            sel_ok = not cfg.get("curriculum") or (step + 1) / total >= cfg["curriculum"][-1][0]
            if sel_ok and score > best:
                best = score
                save(m, opt, step + 1, out.dir / "best.pt")
            m.train()
    # final evaluation from the dev-selected checkpoint
    st = torch.load(out.dir / "best.pt", map_location="cuda")
    m.load_state_dict({k: v.cuda() for k, v in st["model"].items()}, strict=False)
    m.eval()
    split = cfg.get("final_split", "dev")
    if cfg.get("bracket"):  # P1: capacity bracket on dev (coarse grid), then freeze the integer test grid
        grid = sorted(cfg["eval_ks"])
        for r_w in cfg["widths"]:
            key = f"bracket|r{r_w}"
            if key in out.done:
                continue
            dev = {k: acc_cell(m, cfg, "dev", r_w, 0, k, cfg["bracket_n"], 3)["min_selector_acc"] for k in grid}
            last = max([0] + [k for k in grid if all(dev[j] >= 0.90 for j in grid if j <= k)])
            first_fail = min([k for k in grid if k > last] + [grid[-1]])
            test_grid = sorted({k for k in grid if k <= first_fail} | set(range(max(1, last), first_fail + 1)))
            out.write({"key": key, "r": r_w, "dev_min_selector_acc": dev, "last_pass": last, "first_fail": first_fail,
                       "test_grid": test_grid})
        brackets = {json.loads(l)["r"]: json.loads(l)["test_grid"] for l in out.path.read_text().splitlines()
                    if json.loads(l)["key"].startswith("bracket|")}
    for r_w in cfg["widths"]:
        for u in cfg["eval_updates"]:
            for k in (brackets[r_w] if cfg.get("bracket") else cfg["eval_ks"]):
                key = f"final|{split}|r{r_w}|u{u}|k{k}"
                if key not in out.done:
                    out.write({"key": key, "split": split, "r": r_w, "u": u, "k": k, "best_step": st["step"],
                               **acc_cell(m, cfg, split, r_w, u, k, cfg["eval_n"], 1)})
        for b in cfg.get("quant_bits", []):
            for qm, u in [tuple(x) for x in cfg.get("quant_modes", [["read", 0], ["pass", 2]])]:
                for k in cfg.get("quant_ks", cfg["eval_ks"]):
                    key = f"quant|{split}|r{r_w}|b{b}|{qm}|u{u}|k{k}"
                    if key not in out.done:
                        out.write({"key": key, "split": split, "r": r_w, "u": u, "k": k, "best_step": st["step"],
                                   **acc_cell(m, cfg, split, r_w, u, k, cfg["eval_n"], 1, bits=b, qmode=qm)})
        if m.variant == "state" and cfg.get("causal_ks"):
            for k in cfg["causal_ks"]:
                key = f"causal|{split}|r{r_w}|k{k}"
                if key not in out.done:
                    out.write({"key": key, "split": split, "r": r_w, "k": k, **causal_tests(m, cfg, split, r_w, k, cfg["causal_n"], 2)})
        if m.variant == "state":
            key = f"pr|r{r_w}"
            if key not in out.done and not cfg.get("skip_pr"):
                out.write({"key": key, "r": r_w, **branch_pr(m, r_w, u=cfg.get("pr_updates", 2))})
