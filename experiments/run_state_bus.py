"""state_bus_v1: a co-developed persistent state interface on a frozen LM (Qwen3-1.7B-Base, CPU).

A 16-d bus at block L: z = E(h_L(anchor)); J z is ADDED to the block-L output at every position after the anchor;
R reads z back from a later block at decision boundaries. Trained on two consequences (sound, young) with native,
same-state-swap, cross-state-swap, prototype and persistence losses; the third consequence (taxonomy) is never trained.
Demonstration on held-out paraphrases: arms none / self / same-donor / cross-donor / shuffled / random.

    python experiments/run_state_bus.py --config experiments/config/state_bus_v1.json --stage smoke|train|eval
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from substitution_probe import SubstitutionProbe


class Bus(nn.Module):
    def __init__(self, d_model, d_state, n_states):
        super().__init__()
        self.E = nn.Linear(d_model, d_state, bias=True); self.J = nn.Linear(d_state, d_model, bias=False); self.R = nn.Linear(d_model, d_state, bias=True)
        self.proto = nn.Parameter(torch.randn(n_states, d_state))
        nn.init.zeros_(self.J.weight)                                    # start as a no-op on the frozen model


class World:
    def __init__(self, cfg):
        self.cfg = cfg; self.sp = SubstitutionProbe(cfg["model_id"], revision=cfg["revision"]); self.m = self.sp.model; self.tok = self.sp.tok
        assert self.sp.revision == cfg["revision"]
        torch.set_grad_enabled(True)                                     # SubstitutionProbe disables grad globally
        for p in self.m.parameters(): p.requires_grad_(False)
        self.L = cfg["layer"]; self.read_layer = min(self.L + 8, self.m.config.num_hidden_layers - 1); self.states = cfg["states"]
        self.ctx_ids = {s: [self.tok.encode(t + cfg["anchor"], add_special_tokens=False) for t in cfg["contexts"][s]] for s in self.states}
        self.P = len(self.ctx_ids[self.states[0]][0]); assert all(len(x) == self.P for s in self.states for x in self.ctx_ids[s]), "contexts must be length-matched"
        self._inject = None; self._reads = None
        self.m.model.layers[self.L].register_forward_hook(self._hook_inject); self.m.model.layers[self.read_layer].register_forward_hook(self._hook_read)
        with torch.no_grad():                                            # anchor activations of every context (frozen model => constants)
            self.h_anchor = {s: torch.stack([self.m(input_ids=torch.tensor([x]), output_hidden_states=True).hidden_states[self.L + 1][0, -1] for x in self.ctx_ids[s]]) for s in self.states}

    def _hook_inject(self, mod, i, o):
        if self._inject is None: return o
        h = o[0] if isinstance(o, tuple) else o; h = h.clone(); h[:, self.P:, :] = h[:, self.P:, :] + self._inject.unsqueeze(1).to(h.dtype)
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h

    def _hook_read(self, mod, i, o):
        if self._reads is not None: self._reads.append(o[0] if isinstance(o, tuple) else o)
        return o

    def cont(self, order, words):
        """Continuation text for a sequence of decisions (consequence names) with the given state words per decision."""
        C = self.cfg["consequences"]; parts = []
        for k, name in enumerate(order):
            c = C[name]; parts.append((c["prefix"] if k == 0 else " It" + c["prefix"]) + words[k] + c["suffix"])
        return "".join(parts)

    def forward(self, ctx, text, z=None, want_reads=False):
        """Teacher-forced forward over ctx + text with bus injection J z after the anchor. Returns logits, ids, reads."""
        ids = ctx + self.tok.encode(text, add_special_tokens=False); self._inject = None if z is None else self.bus.J(z).unsqueeze(0)
        self._reads = [] if want_reads else None
        out = self.m(input_ids=torch.tensor([ids])); reads = self._reads; self._inject = None; self._reads = None
        return out.logits[0], ids, reads

    def ll_span(self, logits, ids, start):
        """Mean log-likelihood of ids[start:] under logits (teacher forced)."""
        lp = torch.log_softmax(logits[start - 1:-1].float(), -1); tgt = torch.tensor(ids[start:])
        return lp.gather(1, tgt.unsqueeze(1)).mean()

    def z_of(self, s, i): return self.bus.E(self.h_anchor[s][i])


def train(W, cfg, seed, log):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); tc = cfg["train"]; lw = tc["loss_weights"]
    W.bus = Bus(W.m.config.hidden_size, cfg["state_dim"], len(W.states)); opt = torch.optim.AdamW(W.bus.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    n_par = sum(p.numel() for p in W.bus.parameters()); assert n_par < 100_000, n_par
    tr = cfg["train_indices"]; order = cfg["trained_consequences"]; C = cfg["consequences"]; t0 = time.time(); hist = []
    for step in range(tc["steps"]):
        s = W.states[rng.integers(len(W.states))]; i, j = rng.choice(tr, 2, replace=False); s2 = W.states[(W.states.index(s) + rng.integers(1, len(W.states))) % len(W.states)]; k = rng.choice(tr)
        ctx = W.ctx_ids[s][i]; words = lambda st: [C[c]["words"][st] for c in order]
        z_own, z_same, z_cross = W.z_of(s, i), W.z_of(s, j), W.z_of(s2, k)
        losses = {}
        lg, ids, reads = W.forward(ctx, W.cont(order, words(s)), z_own, want_reads=True); losses["native"] = -W.ll_span(lg, ids, W.P)
        losses["persistence"] = F.mse_loss(W.bus.R(reads[0][0, -1]), z_own)      # read back at the end of the decisions
        lg, ids, _ = W.forward(ctx, W.cont(order, words(s)), z_same); losses["same_swap"] = -W.ll_span(lg, ids, W.P)
        lg, ids, _ = W.forward(ctx, W.cont(order, words(s2)), z_cross); losses["cross_swap"] = -W.ll_span(lg, ids, W.P)
        Z = torch.stack([W.z_of(st, ii) for st in W.states for ii in tr]); lab = torch.tensor([W.states.index(st) for st in W.states for ii in tr])
        d = torch.cdist(Z, W.bus.proto); losses["prototype"] = (d.gather(1, lab.unsqueeze(1)).squeeze(1) ** 2).mean() + F.relu(tc["prototype_margin"] - torch.pdist(W.bus.proto)).mean()
        loss = sum(lw[k] * v for k, v in losses.items()); opt.zero_grad(); loss.backward(); opt.step()
        hist.append({k: float(v) for k, v in losses.items()}); hist[-1]["step"] = step
        if step % 20 == 0 or step == tc["steps"] - 1: log(f"seed {seed} step {step}: " + " ".join(f"{k}={float(v):.3f}" for k, v in losses.items()) + f" ({time.time()-t0:.0f}s)")
        if time.time() - t0 > tc["wall_cap_hours"] * 3600 / len(tc["seeds"]): log("per-seed wall cap reached"); break
    return hist, n_par


def evaluate(W, cfg, log):
    """Held-out demonstration. For each recipient and arm, score the three decisions in order with 4-way candidate LL."""
    C = cfg["consequences"]; order = cfg["decision_order"]; ho = cfg["heldout_indices"]; S = W.states; rows = []
    def score(ctx, z, expect_state):
        """Sequential 4-way scoring; earlier decisions are filled with the expected state's words."""
        out = []
        for k, name in enumerate(order):
            prev = [C[c]["words"][expect_state] for c in order[:k]]; lls = {}
            for cand in S:
                text = W.cont(order[:k + 1], prev + [C[name]["words"][cand]]); lg, ids, _ = W.forward(ctx, text, z)
                start = len(ctx) + len(W.tok.encode(W.cont(order[:k], prev), add_special_tokens=False)) if k else len(ctx)
                lls[cand] = float(W.ll_span(lg, ids, start))
            out.append(lls)
        return out
    with torch.no_grad():
        for s in S:
            other = S[(S.index(s) + 1) % len(S)]
            for i in ho:
                ctx = W.ctx_ids[s][i]; z_own = W.z_of(s, i); j = ho[(ho.index(i) + 1) % len(ho)]
                g = torch.Generator().manual_seed(7 + i); rnd = torch.randn(z_own.shape, generator=g); rnd = rnd / rnd.norm() * z_own.norm()
                arms = {"none": (None, s), "self": (z_own, s), "same": (W.z_of(s, j), s), "cross": (W.z_of(other, i), other),
                        "shuffled": (z_own[torch.randperm(z_own.numel(), generator=g)], s), "random": (rnd, s)}
                for name, (z, exp) in arms.items():
                    dec = score(ctx, z, exp); rows.append({"state": s, "i": i, "arm": name, "expect": exp, "decisions": dec,
                                                            "choice": [max(d, key=d.get) for d in dec], "consistent": [max(d, key=d.get) == exp for d in dec]})
                log(f"{s}{i}: " + " | ".join(f"{r['arm']}:{''.join('Y' if c else 'n' for c in r['consistent'])}" for r in rows[-6:]))
    return rows


def verdict(rows, cfg, log):
    K = cfg["kill"]; order = cfg["decision_order"]; ho_n = len(cfg["heldout_indices"]) * len(cfg["states"]); tr_idx = [order.index(c) for c in cfg["trained_consequences"]]; ho_idx = order.index(cfg["heldout_consequence"])
    by = lambda arm: [r for r in rows if r["arm"] == arm]
    trained_acc = np.mean([r["consistent"][k] for r in by("self") for k in tr_idx])
    margin = lambda r, k: r["decisions"][k][r["expect"]] - max(v for c, v in r["decisions"][k].items() if c != r["expect"])
    sig = lambda r: np.array([margin(r, k) for k in range(len(order))])
    same_d = [float(np.linalg.norm(sig(a) - sig(b))) for a, b in zip(by("same"), by("self"))]; tau = float(np.percentile(same_d, 50) * 2 + 1e-6)
    cross_two = sum(sum(r["consistent"]) >= 2 for r in by("cross")); ho_cons = sum(r["consistent"][ho_idx] for r in by("cross"))
    ctrl = np.mean([r["consistent"][ho_idx] for a in ("shuffled", "random") for r in by(a)])
    # movement toward the donor at decision 1 vs 3: donor-consistent margin under cross minus under self, standardized by self spread
    mv = lambda k: float(np.mean([(rc["decisions"][k][rc["expect"]] - rc["decisions"][k][rs["expect"]]) - (rs["decisions"][k][rc["expect"]] - rs["decisions"][k][rs["expect"]]) for rc, rs in zip(by("cross"), by("self"))]))
    m1, m3 = mv(0), mv(ho_idx)
    summary = {"trained_consequence_accuracy_self": float(trained_acc), "same_out_of_tolerance": int(sum(d > tau for d in same_d)), "tau_2x_median": tau,
               "cross_two_of_three": int(cross_two), "heldout_consistent_cross": int(ho_cons), "heldout_consistent_controls": float(ctrl), "heldout_gain": float(ho_cons / ho_n - ctrl),
               "movement_first": m1, "movement_third": m3, "n_recipients": ho_n,
               "arm_accuracy": {a: [float(np.mean([r["consistent"][k] for r in by(a)])) for k in range(len(order))] for a in ("none", "self", "same", "cross", "shuffled", "random")}}
    fails = []
    if trained_acc < K["trained_min"]: fails.append("trained consequences below 85%")
    if summary["same_out_of_tolerance"] > K["same_out_of_tolerance_max"]: fails.append("same-state swaps out of tolerance")
    if cross_two < K["cross_two_of_three_min"]: fails.append("cross-state donors not consistent on 2/3")
    if ho_cons < K["heldout_consistent_min"] or summary["heldout_gain"] < K["heldout_gain_min"]: fails.append("never-trained consequence does not move")
    if m1 > 0 and m3 < K["third_over_first_min"] * m1: fails.append("movement decays by the third decision")
    status = "BOUNDED POSITIVE — PERSISTENT INTERCHANGEABLE STATE BUS (this construction)" if not fails else ("SUPERVISED RESPONSE CONTROLLER — STOP" if fails == ["never-trained consequence does not move"] else "FAIL — " + "; ".join(fails))
    log(json.dumps(summary, indent=1)); log(f"STATUS: {status}"); return summary, status


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--stage", default="smoke"); a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8")); out = f"experiments/results/{cfg['name']}"; os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, f"{a.stage}.log"), "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    shas = {k: hashlib.sha256(open(v, "rb").read()).hexdigest() for k, v in (("runner", __file__), ("config", a.config))}
    t0 = time.time(); W = World(cfg); log(f"loaded {cfg['model_id']} rev={W.sp.revision}; P={W.P}; read layer {W.read_layer} ({time.time()-t0:.0f}s)")
    if a.stage == "smoke":
        cfg = dict(cfg); cfg["train"] = dict(cfg["train"], steps=3); hist, n = train(W, cfg, 11, log); log(f"bus params {n}; 3 steps ok; {time.time()-t0:.0f}s")
        W.bus.eval(); cfg["heldout_indices"] = cfg["heldout_indices"][:1]; rows = evaluate(W, cfg, log); verdict(rows, cfg, log); return
    result = {"config": cfg["name"], "sha256": shas, "revision": W.sp.revision, "seeds": {}}
    for seed in cfg["train"]["seeds"]:
        hist, n = train(W, cfg, seed, log); torch.save(W.bus.state_dict(), os.path.join(out, f"bus_seed{seed}.pt")); W.bus.eval()
        rows = evaluate(W, cfg, log); summary, status = verdict(rows, cfg, log)
        result["seeds"][seed] = {"params": n, "history": hist, "rows": rows, "summary": summary, "status": status}
        json.dump(result, open(os.path.join(out, "result.json"), "w"), indent=1, default=float); log(f"seed {seed} done ({time.time()-t0:.0f}s)")
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
