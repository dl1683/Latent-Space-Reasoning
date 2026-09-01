"""ERQ-1: Endogenous Response-Quotient Selector v1.
Theory: Section 10, PREDICTIVE_FIBER_ACTION_ALGEBRA.md. Budget: 144 forwards."""
import torch, torch.nn.functional as F, math, json, hashlib, os, sys
import numpy as np
from datetime import datetime
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

RDIR = os.path.join(os.path.dirname(__file__), "results", "endogenous_response_quotient_v1")

def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""): h.update(c)
    return h.hexdigest()

def _cfg(p):
    with open(p) as f: return json.load(f)

def _model(c):
    tok = AutoTokenizer.from_pretrained(c["model_id"], revision=c["model_revision"], trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(c["model_id"], revision=c["model_revision"],
        torch_dtype=torch.float32, device_map=c["device"], trust_remote_code=True)
    m.eval(); return m, tok

def _prompts(c):
    out = []
    for f in c["families"]:
        A, B = f["entities"]; v = f["values"]
        for ord, t in [("std", c["template_standard"]), ("rev", c["template_reversed"])]:
            for ai in range(2):
                for bi in range(2):
                    for qi, Q in enumerate([A, B]):
                        out.append(dict(family=f["name"], world=f"w{ai}{bi}", query=Q,
                            query_idx=qi, order=ord,
                            prompt=t.format(A=A, B=B, va=v[ai], vb=v[bi], Q=Q), vals=v))
    return out

def _states(m, tok, prompt, dev):
    st = {}; hooks = []
    for i, layer in enumerate(m.model.layers):
        def mh(idx):
            def hf(_, __, out):
                st[idx] = (out[0] if isinstance(out, tuple) else out).detach().clone()
            return hf
        hooks.append(layer.register_forward_hook(mh(i)))
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad(): o = m(ids, use_cache=False)
    for h in hooks: h.remove()
    return st, F.softmax(o.logits[0, -1], dim=-1)

def _hooked(m, tok, prompt, blk, mode, dev):
    ok = [False]
    def hf(_, inp, out):
        ok[0] = True
        if mode == "bypass":
            return (inp[0],) + out[1:] if isinstance(out, tuple) else inp[0]
        return out
    hnd = m.model.layers[blk].register_forward_hook(hf)
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad(): o = m(ids, use_cache=False)
    hnd.remove(); assert ok[0]
    return F.softmax(o.logits[0, -1], dim=-1)

def _ll(m, h):
    return F.softmax(m.lm_head(m.model.norm(h[None, None]))[0, 0].detach(), dim=-1)

def _jsd(p, q):
    mid = (p + q) / 2; eps = 1e-10
    j = .5*((p+eps)*((p+eps)/(mid+eps)).log()).sum() + .5*((q+eps)*((q+eps)/(mid+eps)).log()).sum()
    return math.sqrt(max(0, float(j) / math.log(2)))

def _edges(c):
    out = []
    for f in c["families"]:
        n = f["name"]
        out += [dict(id=f"{n}_A0", fam=n, w1="w00", w2="w10", cf=0),
                dict(id=f"{n}_A1", fam=n, w1="w01", w2="w11", cf=0),
                dict(id=f"{n}_B0", fam=n, w1="w00", w2="w01", cf=1),
                dict(id=f"{n}_B1", fam=n, w1="w10", w2="w11", cf=1)]
    return out

def _save(name, obj):
    os.makedirs(RDIR, exist_ok=True)
    with open(os.path.join(RDIR, name), "w") as f: json.dump(obj, f, indent=2)

def _metrics(cl):
    ao, co, af, cf = [], [], [], []
    for c in cl.values():
        for e in c["rel"]: ao.append(e["dBO"]-e["dIO"]); af.append(e["dBF"]-e["dIF"])
        for e in c["irr"]: co.append(e["dIO"]-e["dBO"]); cf.append(e["dIF"]-e["dBF"])
    if not ao: return {k: 0.0 for k in ["AO","CO","SO","AF","CF","SF"]}
    return dict(AO=np.mean(ao), CO=np.mean(co), SO=np.mean(ao)+np.mean(co),
                AF=np.mean(af), CF=np.mean(cf), SF=np.mean(af)+np.mean(cf))

def preflight(cp):
    c = _cfg(cp)
    tok = AutoTokenizer.from_pretrained(c["model_id"], revision=c["model_revision"], trust_remote_code=True)
    ps = _prompts(c); r = dict(valid=True, n=len(ps), fwd=len(ps)*3, checks=[], vtok={})
    for f in c["families"]:
        for v in f["values"]:
            t = tok.encode(" " + v)
            if len(t) != 1: r["valid"] = False; r["checks"].append(f"'{v}' multi-token: {t}")
            else: r["vtok"][v] = t[0]
    grp = defaultdict(set)
    for p in ps: grp[(p["family"], p["query"], p["order"])].add(len(tok.encode(p["prompt"])))
    for k, lens in grp.items():
        if len(lens) > 1: r["valid"] = False; r["checks"].append(f"len mismatch {k}: {lens}")
    _save("preflight.json", r)
    print(f"Preflight: {'PASS' if r['valid'] else 'FAIL'} -- {r['n']} cells, {r['fwd']} fwd")
    print(f"  vtok: {r['vtok']}")
    for ch in r["checks"]: print(f"  ! {ch}")

def lock(cp):
    c = _cfg(cp); m, tok = _model(c); checks = []
    for f in c["families"]:
        A, B = f["entities"]; v = f["values"]
        for qi, Q in enumerate([A, B]):
            for ai in range(2):
                for bi in range(2):
                    p = c["template_standard"].format(A=A, B=B, va=v[ai], vb=v[bi], Q=Q)
                    ids = tok(p, return_tensors="pt").input_ids.to(c["device"])
                    with torch.no_grad(): g = tok.decode([int(torch.argmax(m(ids).logits[0,-1]))]).strip()
                    exp = v[ai] if qi == 0 else v[bi]
                    checks.append(dict(f=f["name"], w=f"w{ai}{bi}", q=Q, g=g, e=exp, ok=g==exp))
    nc = sum(1 for x in checks if x["ok"])
    _save("manifest.json", dict(exp=c["experiment"], ts=datetime.now().isoformat(),
        model=c["model_id"], rev=c["model_revision"], blk=c["bypass_block"],
        nl=m.config.num_hidden_layers, vs=m.config.vocab_size,
        rh=_sha(os.path.abspath(__file__)), ch=_sha(os.path.abspath(cp)),
        bl=checks, ok=f"{nc}/{len(checks)}"))
    del m; print(f"Locked: {nc}/{len(checks)} baselines correct")

def produce(cp):
    c = _cfg(cp); m, tok = _model(c); blk = c["bypass_block"]; dev = c["device"]
    ps = _prompts(c); data = {}
    for i, p in enumerate(ps):
        k = (p["family"], p["query"], p["order"], p["world"])
        print(f"  [{i+1}/{len(ps)}] {p['family']} {p['world']} q={p['query']} {p['order']}")
        st, nF = _states(m, tok, p["prompt"], dev)
        noF = _hooked(m, tok, p["prompt"], blk, "noop", dev)
        bF = _hooked(m, tok, p["prompt"], blk, "bypass", dev)
        nO = _ll(m, st[blk][0, -1]); bO = _ll(m, st[blk-1][0, -1])
        nd = _jsd(nF, noF)
        vmn = vmb = 0.0
        for f in c["families"]:
            if f["name"] == p["family"]:
                for v in f["values"]:
                    tid = tok.encode(" " + v)[0]; vmn += float(nF[tid]); vmb += float(bF[tid])
        data[k] = dict(nF=nF, bF=bF, nO=nO, bO=bO, nd=nd, vmn=vmn, vmb=vmb); del st
    ev = []
    for e in _edges(c):
        ents = next(f["entities"] for f in c["families"] if f["name"] == e["fam"])
        for qi, Q in enumerate(ents):
            for o in ["std", "rev"]:
                d1, d2 = data[(e["fam"], Q, o, e["w1"])], data[(e["fam"], Q, o, e["w2"])]
                ev.append(dict(edge=e["id"], fam=e["fam"], query=Q, order=o, cf=e["cf"],
                    rel=e["cf"]==qi, dBF=round(_jsd(d1["nF"], d2["nF"]), 6),
                    dIF=round(_jsd(d1["bF"], d2["bF"]), 6),
                    dBO=round(_jsd(d1["nO"], d2["nO"]), 6),
                    dIO=round(_jsd(d1["bO"], d2["bO"]), 6)))
    mn = max(d["nd"] for d in data.values())
    bv = sum(1 for d in data.values() if d["vmb"] >= c["gates"]["bypass_viability_mass_fraction"]*d["vmn"])
    _save("evidence.json", dict(exp=c["experiment"], ts=datetime.now().isoformat(), blk=blk,
        nc=len(ps), nf=len(ps)*3, instr=dict(mnoop=round(mn,8)),
        bypass=dict(v=bv, t=len(ps)), edges=ev))
    del m; print(f"Evidence: {len(ev)} meas, noop={mn:.8f}, bypass {bv}/{len(ps)}")

def reduce(cp):
    c = _cfg(cp); g = c["gates"]
    with open(os.path.join(RDIR, "evidence.json")) as f: ev = json.load(f)
    ed = ev["edges"]; mn = ev["instr"]["mnoop"]
    g1 = mn <= g["noop_tolerance"]
    eids = sorted(set(e["edge"] for e in ed))
    mat = sum(1 for eid in eids if any(e["dBF"]>=g["material_separation_floor"]
              for e in ed if e["edge"]==eid and e["rel"]))
    g2 = mat >= g["min_eligible_edges"]
    g3 = ev["bypass"]["v"] >= g["bypass_viability_min_cells"]
    cl = {}
    for e in ed:
        cl.setdefault(e["edge"], {"rel":[], "irr":[]})["rel" if e["rel"] else "irr"].append(e)
    pt = _metrics(cl)
    rng = np.random.RandomState(c["bootstrap_seed"]); nb = c["bootstrap_resamples"]
    boots = {k: [] for k in pt}
    for _ in range(nb):
        si = rng.choice(eids, size=len(eids), replace=True)
        m = _metrics({f"s{i}": cl[eid] for i, eid in enumerate(si)})
        for k in boots: boots[k].append(m[k])
    ci = {}
    for k in pt:
        s = sorted(boots[k]); n = len(s)
        ci[k] = dict(est=round(float(pt[k]),6), lb=round(float(s[int(.025*n)]),6),
                     ub=round(float(s[int(.975*n)]),6))
    pex = []
    for eid in eids:
        ae = cl[eid]["rel"] + cl[eid]["irr"]
        std = sorted([e for e in ae if e["order"]=="std"], key=lambda x: x["query"])
        rev = sorted([e for e in ae if e["order"]=="rev"], key=lambda x: x["query"])
        for s, r in zip(std, rev):
            pex.append(abs(s["dBF"]-r["dBF"]) - abs(s["dIF"]-r["dIF"]))
    pb = sorted(float(np.mean(rng.choice(pex, size=len(pex), replace=True))) for _ in range(nb))
    pub = pb[int(.975*len(pb))]
    fsf = {}
    for fc in c["families"]:
        fn = fc["name"]; fsf[fn] = float(_metrics({e: cl[e] for e in eids if e.startswith(fn)})["SF"])
    g4 = ci["AO"]["lb"]>=g["A_O_lb"] and ci["CO"]["lb"]>=g["C_O_lb"] and ci["SO"]["lb"]>=g["Sigma_O_lb"]
    g5 = (ci["AF"]["est"]>=g["A_F_est_min"] and ci["CF"]["est"]>=g["C_F_est_min"]
          and ci["AF"]["lb"]>0 and ci["CF"]["lb"]>0
          and ci["SF"]["est"]>=g["Sigma_F_est_min"] and ci["SF"]["lb"]>=g["Sigma_F_lb"])
    g6 = pub <= g["presentation_stability_ub"]; g7 = all(v>=g["family_Sigma_F_min"] for v in fsf.values())
    if not (g1 and g2 and g3): vr = "INVALID_OR_NO_PROPAGATION_CONTROL"
    elif g4 and g5 and g6 and g7: vr = "ENDOGENOUS_RESPONSE_QUOTIENT_REGISTERED"
    elif g4 and not g5: vr = "LOCAL_OBSERVER_ONLY"
    elif not g4 and not g5: vr = "ORDINARY_PROPAGATION_SUFFICIENT"
    elif g4 and g5 and not g6: vr = "QUERY_SELECTIVE_BUT_PRESENTATION_UNSTABLE"
    elif (ci["AO"]["lb"]>=g["A_O_lb"]) != (ci["CO"]["lb"]>=g["C_O_lb"]): vr = "ONE_SIDED_BLOCK_ACTION"
    else: vr = "INCONCLUSIVE_ALLOCATION_STOP"
    _save("verdict.json", dict(exp=c["experiment"], ts=datetime.now().isoformat(), verdict=vr,
        gates=dict(instrument=dict(ok=g1, mn=round(mn,8)), material=dict(ok=g2, n=mat),
            bypass=dict(ok=g3, v=ev["bypass"]["v"]),
            immediate=dict(ok=g4, AO=ci["AO"], CO=ci["CO"], SO=ci["SO"]),
            suffix=dict(ok=g5, AF=ci["AF"], CF=ci["CF"], SF=ci["SF"]),
            presentation=dict(ok=g6, ub=round(pub,6)),
            family=dict(ok=g7, sf={k: round(v2,6) for k, v2 in fsf.items()})),
        bootstrap=dict(seed=c["bootstrap_seed"], n=nb)))
    P = lambda b: "PASS" if b else "FAIL"
    print(f"\nVERDICT: {vr}")
    print(f"  Instrument: {P(g1)} (noop={mn:.8f})")
    print(f"  Material:   {P(g2)} ({mat}/12)")
    print(f"  Bypass:     {P(g3)} ({ev['bypass']['v']}/{ev['bypass']['t']})")
    print(f"  Immediate:  {P(g4)} AO={ci['AO']['est']:.4f}[{ci['AO']['lb']:.4f}] "
          f"CO={ci['CO']['est']:.4f}[{ci['CO']['lb']:.4f}] SO={ci['SO']['est']:.4f}[{ci['SO']['lb']:.4f}]")
    print(f"  Suffix:     {P(g5)} AF={ci['AF']['est']:.4f}[{ci['AF']['lb']:.4f}] "
          f"CF={ci['CF']['est']:.4f}[{ci['CF']['lb']:.4f}] SF={ci['SF']['est']:.4f}[{ci['SF']['lb']:.4f}]")
    print(f"  Stability:  {P(g6)} (ub={pub:.4f})")
    print(f"  Family:     {P(g7)} {fsf}")

if __name__ == "__main__":
    if len(sys.argv) < 3 or "--config" not in sys.argv:
        print("Usage: run_endogenous_response_quotient_v1.py <preflight|lock|produce|reduce> --config <path>")
        sys.exit(1)
    {"preflight": preflight, "lock": lock, "produce": produce, "reduce": reduce}[sys.argv[1]](
        sys.argv[sys.argv.index("--config") + 1])
