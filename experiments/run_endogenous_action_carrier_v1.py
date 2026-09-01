"""EAC-1: Endogenous Action Carrier v1.
Theory: Section 14, PREDICTIVE_FIBER_ACTION_ALGEBRA.md. Budget: 150K seq-fwd, 120 CPU min.
Co-designed carrier trained only on final-state prediction, then evaluated via
causal transplantation for compositional action laws.

Architecture: structured world encoding → differentiable table lookup.
Each transition triple → key from (state, action) embeddings, value = next_state embedding.
Carrier = attention output over transition values. Readout = carrier vs state embeddings."""
import torch, torch.nn as nn, torch.nn.functional as F
import json, hashlib, os, sys, time, math
import numpy as np
from datetime import datetime

RDIR = os.path.join(os.path.dirname(__file__), "results", "endogenous_action_carrier_v1")

def _cfg(p):
    with open(p) as f: return json.load(f)

def _save(name, obj):
    os.makedirs(RDIR, exist_ok=True)
    with open(os.path.join(RDIR, name), "w") as f: json.dump(obj, f, indent=2)


class EACModel(nn.Module):
    def __init__(self, vocab, d, n_states):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.key_proj = nn.Linear(d * 2, d)
        self.q_proj = nn.Linear(d * 2, d)
        self.state_init = nn.Linear(d, d)
        self.d = d

    def _encode_world(self, world_ids):
        n_trans = world_ids.size(1) // 3
        e = self.embed(world_ids).view(-1, n_trans, 3, self.d)
        keys = self.key_proj(torch.cat([e[:, :, 0], e[:, :, 1]], -1))
        vals = e[:, :, 2, :]
        return keys, vals

    def _step(self, carrier, action_emb, keys, vals):
        q = self.q_proj(torch.cat([carrier, action_emb], -1))
        s = torch.bmm(q.unsqueeze(1), keys.transpose(1, 2)) / math.sqrt(self.d)
        return torch.bmm(F.softmax(s, -1), vals).squeeze(1)

    def _readout(self, carrier, state_ids):
        se = self.embed(state_ids)
        return torch.bmm(carrier.unsqueeze(1), se.transpose(1, 2)).squeeze(1) / math.sqrt(self.d)

    def forward(self, world_ids, start_id, action_ids, state_ids, return_carriers=False):
        keys, vals = self._encode_world(world_ids)
        carrier = self.state_init(self.embed(start_id))
        carriers = [carrier]
        for i in range(action_ids.size(1)):
            carrier = self._step(carrier, self.embed(action_ids[:, i]), keys, vals)
            carriers.append(carrier)
        logits = self._readout(carrier, state_ids)
        return (logits, carriers) if return_carriers else logits

    def forward_with_inject(self, world_ids, start_id, action_ids, state_ids, inj, inj_step):
        keys, vals = self._encode_world(world_ids)
        carrier = self.state_init(self.embed(start_id))
        for i in range(action_ids.size(1)):
            if i == inj_step:
                carrier = inj
            carrier = self._step(carrier, self.embed(action_ids[:, i]), keys, vals)
        return self._readout(carrier, state_ids)


class WorldGen:
    def __init__(self, ns, na, pool_size, rng):
        self.ns, self.na, self.ps, self.rng = ns, na, pool_size, rng

    def make_world(self):
        names = self.rng.choice(self.ps, self.ns + self.na, replace=False).tolist()
        return dict(sn=names[:self.ns], an=names[self.ns:],
                    tab=self.rng.randint(0, self.ns, (self.na, self.ns)).tolist())

    def episode(self, w, wl):
        s = self.rng.randint(0, self.ns)
        acts = self.rng.randint(0, self.na, wl).tolist()
        cur = s
        for a in acts: cur = w["tab"][a][cur]
        return dict(start=s, actions=acts, final=cur)

    def encode(self, w):
        order = list(range(self.ns * self.na))
        self.rng.shuffle(order)
        toks = []
        for idx in order:
            a, s = idx // self.ns, idx % self.ns
            toks.extend([w["sn"][s], w["an"][a], w["sn"][w["tab"][a][s]]])
        return toks


def _batch(gen, w, n, wl, dev, w_enc=None):
    eps = [gen.episode(w, wl) for _ in range(n)]
    if w_enc is None: w_enc = gen.encode(w)
    wids = torch.tensor([w_enc] * n, device=dev)
    sids = torch.tensor([w["sn"][e["start"]] for e in eps], device=dev)
    aids = torch.tensor([[w["an"][a] for a in e["actions"]] for e in eps], device=dev)
    snids = torch.tensor([w["sn"]] * n, device=dev)
    tgts = torch.tensor([e["final"] for e in eps], device=dev)
    return wids, sids, aids, snids, tgts, eps, w_enc


def preflight(cp):
    c = _cfg(cp)
    model = EACModel(c["vocab_pool_size"], c["hidden_dim"], c["n_states"])
    n_params = sum(p.numel() for p in model.parameters())
    rng = np.random.RandomState(42)
    gen = WorldGen(c["n_states"], c["n_actions"], c["vocab_pool_size"], rng)
    w = gen.make_world(); ep = gen.episode(w, 3); we = gen.encode(w)
    wids = torch.tensor([we]); sid = torch.tensor([w["sn"][ep["start"]]])
    aids = torch.tensor([[w["an"][a] for a in ep["actions"]]]); snids = torch.tensor([w["sn"]])
    out = model(wids, sid, aids, snids)
    checks = []
    if n_params > 2_000_000: checks.append(f"params {n_params} > 2M")
    if out.shape != (1, c["n_states"]): checks.append(f"shape {out.shape}")
    total = (c["train_forwards_per_seed"] + c["eval_forwards_per_seed"] +
             c["integrity_forwards_per_seed"]) * len(c["seeds"])
    if total > 150_000: checks.append(f"total {total} > 150K")
    _save("preflight.json", dict(valid=not checks, n_params=n_params, total_fwd=total,
        world_tokens=len(we), checks=checks))
    print(f"Preflight: {'PASS' if not checks else 'FAIL'} -- {n_params:,} params, {total:,} fwd")
    for ch in checks: print(f"  ! {ch}")


def _train_seed(c, seed):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    gen = WorldGen(c["n_states"], c["n_actions"], c["vocab_pool_size"], rng)
    model = EACModel(c["vocab_pool_size"], c["hidden_dim"], c["n_states"])
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
    bs, n_bat = c["batch_size"], c["train_forwards_per_seed"] // c["batch_size"]
    losses = []; t0 = time.time()
    for i in range(n_bat):
        w = gen.make_world()
        wl = int(rng.choice(c["train_word_lengths"]))
        wids, sids, aids, snids, tgts, _, _ = _batch(gen, w, bs, wl, "cpu")
        loss = F.cross_entropy(model(wids, sids, aids, snids), tgts)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.detach().item())
        if (i + 1) % 250 == 0:
            print(f"  [seed {seed}] {i+1}/{n_bat} loss={np.mean(losses[-250:]):.4f} ({time.time()-t0:.0f}s)")
    return model, dict(seed=seed, n_bat=n_bat, elapsed=round(time.time()-t0, 1),
                       final_loss=round(float(np.mean(losses[-100:])), 6))


def _eval_seed(c, model, seed):
    rng = np.random.RandomState(seed + 10000)
    gen = WorldGen(c["n_states"], c["n_actions"], c["vocab_pool_size"], rng)
    model.eval(); bs = c["batch_size"]; r = dict(seed=seed)
    with torch.no_grad():
        acc = {l: [] for l in c["test_word_lengths"]}
        for _ in range(c["eval_forwards_per_seed"] // (bs * len(c["test_word_lengths"]))):
            w = gen.make_world()
            for wl in c["test_word_lengths"]:
                wids, sids, aids, snids, tgts, _, _ = _batch(gen, w, bs, wl, "cpu")
                acc[wl].extend((model(wids, sids, aids, snids).argmax(-1) == tgts).tolist())
        r["accuracy"] = {str(l): round(float(np.mean(acc[l])), 6) for l in c["test_word_lengths"]}
        r["accuracy_overall"] = round(float(np.mean([v for vs in acc.values() for v in vs])), 6)

        ni = c["integrity_forwards_per_seed"]

        sp = []
        for _ in range(ni // (bs * 4)):
            w = gen.make_world(); we = gen.encode(w)
            wids, sids, aids, snids, tgts, _, _ = _batch(gen, w, bs, 2, "cpu", we)
            clean, cars = model(wids, sids, aids, snids, return_carriers=True)
            patched = model.forward_with_inject(wids, sids, aids, snids, cars[1], 1)
            sp.append(float(torch.max(torch.abs(F.softmax(clean, -1) - F.softmax(patched, -1)))))
        r["self_patch_max"] = round(max(sp) if sp else 1.0, 8)

        same_p, same_j = [], []
        for _ in range(ni // (bs * 4)):
            w = gen.make_world(); we = gen.encode(w)
            e1, e2 = gen.episode(w, 2), gen.episode(w, 2)
            for _ in range(20):
                if e1["final"] == e2["final"]: break
                e2 = gen.episode(w, 2)
            if e1["final"] != e2["final"]: continue
            wids = torch.tensor([we]); snids = torch.tensor([w["sn"]])
            s1 = torch.tensor([w["sn"][e1["start"]]]); s2 = torch.tensor([w["sn"][e2["start"]]])
            a1 = torch.tensor([[w["an"][a] for a in e1["actions"]]])
            a2 = torch.tensor([[w["an"][a] for a in e2["actions"]]])
            _, c1 = model(wids, s1, a1, snids, return_carriers=True)
            _, c2 = model(wids, s2, a2, snids, return_carriers=True)
            for ai in range(c["n_actions"]):
                suf = torch.tensor([[w["an"][ai]]])
                full_a = torch.cat([a1, suf], 1)
                o1 = F.softmax(model.forward_with_inject(wids, s1, full_a, snids, c1[-1], len(e1["actions"])), -1)
                o2 = F.softmax(model.forward_with_inject(wids, s1, full_a, snids, c2[-1], len(e1["actions"])), -1)
                same_p.append(int(o1.argmax() == o2.argmax()))
                m = 0.5 * (o1 + o2)
                jsd = 0.5 * float(F.kl_div(m.log(), o1, reduction='sum') +
                                  F.kl_div(m.log(), o2, reduction='sum'))
                same_j.append(math.sqrt(max(0, jsd)))
        r["same_place_rate"] = round(float(np.mean(same_p)), 6) if same_p else 0.0
        r["same_place_jsd_pass"] = round(float(np.mean([1 if j <= c["gates"]["same_place_jsd_ub"] else 0
                                                         for j in same_j])), 6) if same_j else 0.0

        desc = []
        for _ in range(ni // (bs * 4)):
            w = gen.make_world(); we = gen.encode(w)
            e1, e2 = gen.episode(w, 2), gen.episode(w, 2)
            for _ in range(20):
                if e1["final"] == e2["final"]: break
                e2 = gen.episode(w, 2)
            if e1["final"] != e2["final"]: continue
            wids = torch.tensor([we]); snids = torch.tensor([w["sn"]])
            s1 = torch.tensor([w["sn"][e1["start"]]]); s2 = torch.tensor([w["sn"][e2["start"]]])
            a1 = torch.tensor([[w["an"][a] for a in e1["actions"]]])
            a2 = torch.tensor([[w["an"][a] for a in e2["actions"]]])
            _, c1 = model(wids, s1, a1, snids, return_carriers=True)
            _, c2 = model(wids, s2, a2, snids, return_carriers=True)
            for ai in range(c["n_actions"]):
                expected = w["tab"][ai][e1["final"]]
                suf = torch.tensor([[w["an"][ai]]])
                full_a = torch.cat([a1, suf], 1)
                o1 = model.forward_with_inject(wids, s1, full_a, snids, c1[-1], len(e1["actions"])).argmax(-1)
                o2 = model.forward_with_inject(wids, s1, full_a, snids, c2[-1], len(e1["actions"])).argmax(-1)
                desc.append(int(o1.item() == expected and o2.item() == expected))
        r["action_descent"] = round(float(np.mean(desc)), 6) if desc else 0.0

        tw_t, tw_eh, tw_ed = [], [], []
        for _ in range(ni // (bs * 2)):
            w = gen.make_world(); we = gen.encode(w)
            he, de = gen.episode(w, 2), gen.episode(w, 2)
            if he["final"] == de["final"]: continue
            for ai in range(c["n_actions"]):
                tgt = w["tab"][ai][de["final"]]
                h_nxt = w["tab"][ai][he["final"]]
                d_cur = de["final"]
                if len({tgt, h_nxt, d_cur}) < 3: continue
                wids = torch.tensor([we]); snids = torch.tensor([w["sn"]])
                sh = torch.tensor([w["sn"][he["start"]]]); sd = torch.tensor([w["sn"][de["start"]]])
                ah = torch.tensor([[w["an"][a] for a in he["actions"]]])
                ad = torch.tensor([[w["an"][a] for a in de["actions"]]])
                _, ch = model(wids, sh, ah, snids, return_carriers=True)
                _, cd = model(wids, sd, ad, snids, return_carriers=True)
                suf = torch.tensor([[w["an"][ai]]])
                out = F.softmax(model.forward_with_inject(wids, sh, torch.cat([ah, suf], 1),
                    snids, cd[-1], len(he["actions"])), -1)
                tw_t.append(1 if out.argmax().item() == tgt else 0)
                tw_eh.append(float(out[0, tgt]) - float(out[0, h_nxt]))
                tw_ed.append(float(out[0, tgt]) - float(out[0, d_cur]))
        r["three_way_target"] = round(float(np.mean(tw_t)), 6) if tw_t else 0.0
        r["three_way_exc_host"] = round(float(np.mean(tw_eh)), 6) if tw_eh else 0.0
        r["three_way_exc_donor"] = round(float(np.mean(tw_ed)), 6) if tw_ed else 0.0
        r["three_way_n"] = len(tw_t)
    return r


def train(cp):
    c = _cfg(cp); all_t = []; all_e = []; t0 = time.time()
    os.makedirs(RDIR, exist_ok=True)
    for seed in c["seeds"]:
        model, ti = _train_seed(c, seed)
        ei = _eval_seed(c, model, seed)
        all_t.append(ti); all_e.append(ei)
        torch.save(model.state_dict(), os.path.join(RDIR, f"model_s{seed}.pt"))
        print(f"  Seed {seed}: acc={ei['accuracy_overall']:.4f} sp={ei['self_patch_max']:.8f} "
              f"same={ei['same_place_rate']:.4f} desc={ei['action_descent']:.4f} "
              f"3w={ei['three_way_target']:.4f}")
    _save("evidence.json", dict(exp=c["experiment"], ts=datetime.now().isoformat(),
        elapsed_s=round(time.time()-t0, 1), training=all_t, evaluation=all_e))
    print(f"\nDone in {time.time()-t0:.1f}s")


def reduce(cp):
    c = _cfg(cp); g = c["gates"]
    with open(os.path.join(RDIR, "evidence.json")) as f: ev = json.load(f)
    evals = ev["evaluation"]; gates = {}
    for gn, fn in [
        ("held_out_acc", lambda e: all(e["accuracy"][str(l)] >= g["held_out_accuracy"]
                                       for l in c["test_word_lengths"])),
        ("self_patch", lambda e: e["self_patch_max"] <= g["self_patch_max"]),
        ("same_place", lambda e: e["same_place_rate"] >= g["same_place_interchangeability"]),
        ("same_place_jsd", lambda e: e["same_place_jsd_pass"] >= g["same_place_jsd_rate"]),
        ("action_descent", lambda e: e["action_descent"] >= g["action_descent"]),
        ("three_way_follow", lambda e: e["three_way_target"] >= g["three_way_target_following"]),
        ("three_way_exceed", lambda e: (e["three_way_exc_host"] >= g["three_way_target_exceeds"]
                                        and e["three_way_exc_donor"] >= g["three_way_target_exceeds"])),
    ]:
        ps = {e["seed"]: fn(e) for e in evals}
        gates[gn] = dict(ok=all(ps.values()), per_seed=ps)
    cap = gates["held_out_acc"]["ok"]; inst = gates["self_patch"]["ok"]
    if not (cap and inst): vr = "INVALID_CARRIER_CONSTRUCTION"
    elif all(g["ok"] for g in gates.values()): vr = "ENDOGENOUS_ACTION_CARRIER_PASS"
    else: vr = "VALID_CARRIER_FAIL"
    _save("verdict.json", dict(exp=c["experiment"], ts=datetime.now().isoformat(),
        verdict=vr, gates=gates, per_seed=[dict(
            seed=e["seed"], accuracy=e["accuracy"], acc_all=e["accuracy_overall"],
            self_patch=e["self_patch_max"], same_place=e["same_place_rate"],
            jsd_pass=e["same_place_jsd_pass"], descent=e["action_descent"],
            tw_target=e["three_way_target"], tw_eh=e["three_way_exc_host"],
            tw_ed=e["three_way_exc_donor"], tw_n=e["three_way_n"]) for e in evals],
        elapsed_s=ev["elapsed_s"]))
    P = lambda b: "PASS" if b else "FAIL"
    print(f"\nVERDICT: {vr}")
    for gn, gr in gates.items(): print(f"  {gn}: {P(gr['ok'])} {gr['per_seed']}")
    for e in evals:
        print(f"\n  Seed {e['seed']}: acc={e['accuracy_overall']:.4f} sp={e['self_patch_max']:.8f} "
              f"same={e['same_place_rate']:.4f} desc={e['action_descent']:.4f} "
              f"3w={e['three_way_target']:.4f} eh={e['three_way_exc_host']:.4f} ed={e['three_way_exc_donor']:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or "--config" not in sys.argv:
        print("Usage: run_endogenous_action_carrier_v1.py <preflight|train|reduce> --config <path>"); sys.exit(1)
    {"preflight": preflight, "train": train, "reduce": reduce}[sys.argv[1]](
        sys.argv[sys.argv.index("--config") + 1])
