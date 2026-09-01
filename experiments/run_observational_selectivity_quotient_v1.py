"""OSQ-1: Observational Selectivity Quotient v1.
Theory: Section 11, PREDICTIVE_FIBER_ACTION_ALGEBRA.md. Budget: 48 forwards."""
import torch, torch.nn.functional as F, math, json, hashlib, os, sys
import numpy as np
from datetime import datetime
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

RDIR = os.path.join(os.path.dirname(__file__), "results", "observational_selectivity_quotient_v1")

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
        for on, t in [("std", c["template_standard"]), ("rev", c["template_reversed"])]:
            for ai in range(2):
                for bi in range(2):
                    for qi, Q in enumerate([A, B]):
                        out.append(dict(family=f["name"], world=f"w{ai}{bi}", query=Q,
                            query_idx=qi, order=on,
                            prompt=t.format(A=A, B=B, va=v[ai], vb=v[bi], Q=Q), vals=v))
    return out

def _cap(m, tok, prompt, dev):
    hs = {}; hooks = []
    for i, layer in enumerate(m.model.layers):
        def mh(idx):
            def hf(_, __, out):
                hs[idx] = (out[0] if isinstance(out, tuple) else out)[0, -1].detach().clone()
            return hf
        hooks.append(layer.register_forward_hook(mh(i)))
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad(): o = m(ids, use_cache=False)
    for h in hooks: h.remove()
    return hs, F.softmax(o.logits[0, -1], dim=-1)

def _ll(m, h):
    return F.softmax(m.lm_head(m.model.norm(h[None]))[0].detach(), dim=-1)

def _jsd(p, q):
    mid = (p + q) / 2; eps = 1e-10
    j = .5*((p+eps)*((p+eps)/(mid+eps)).log()).sum() + .5*((q+eps)*((q+eps)/(mid+eps)).log()).sum()
    return math.sqrt(max(0, float(j) / math.log(2)))

def _j3(p, q, tids):
    def b(d): s = sum(float(d[t]) for t in tids); return torch.tensor([float(d[tids[0]]), float(d[tids[1]]), 1-s])
    return _jsd(b(p), b(q))

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

def _bci(vals, rng, nb):
    b = sorted(float(np.mean(rng.choice(vals, len(vals), True))) for _ in range(nb))
    return dict(est=round(float(np.mean(vals)), 6), lb=round(b[int(.025*nb)], 6), ub=round(b[int(.975*nb)], 6))

def preflight(cp):
    c = _cfg(cp); tok = AutoTokenizer.from_pretrained(c["model_id"], revision=c["model_revision"], trust_remote_code=True)
    ps = _prompts(c); r = dict(valid=True, n=len(ps), fwd=len(ps), checks=[], vtok={})
    for f in c["families"]:
        for v in f["values"]:
            t = tok.encode(" " + v)
            if len(t) != 1: r["valid"] = False; r["checks"].append(f"'{v}' multi-token: {t}")
            else: r["vtok"][v] = t[0]
    grp = defaultdict(set)
    for p in ps: grp[(p["family"], p["query"], p["order"])].add(len(tok.encode(p["prompt"])))
    for k, lens in grp.items():
        if len(lens) > 1: r["valid"] = False; r["checks"].append(f"len mismatch {k}: {lens}")
    _save("preflight.json", r); print(f"Preflight: {'PASS' if r['valid'] else 'FAIL'} -- {r['n']} cells, {r['fwd']} fwd")
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
        model=c["model_id"], rev=c["model_revision"], nl=m.config.num_hidden_layers,
        vs=m.config.vocab_size, rh=_sha(os.path.abspath(__file__)),
        ch=_sha(os.path.abspath(cp)), bl=checks, ok=f"{nc}/{len(checks)}"))
    del m; print(f"Locked: {nc}/{len(checks)} baselines correct")

def produce(cp):
    c = _cfg(cp); m, tok = _model(c); dev = c["device"]; nl = m.config.num_hidden_layers
    vtok = {}
    for f in c["families"]:
        for v in f["values"]: vtok[v] = tok.encode(" " + v)[0]
    ps = _prompts(c); data = {}
    for i, p in enumerate(ps):
        k = (p["family"], p["query"], p["order"], p["world"])
        print(f"  [{i+1}/{len(ps)}] {p['family']} {p['world']} q={p['query']} {p['order']}")
        data[k] = _cap(m, tok, p["prompt"], dev)
    ev = []; integ = []
    for e in _edges(c):
        fam = next(f for f in c["families"] if f["name"] == e["fam"])
        tids = [vtok[v] for v in fam["values"]]
        for qi, Q in enumerate(fam["entities"]):
            for o in ["std", "rev"]:
                hs1, f1 = data[(e["fam"], Q, o, e["w1"])]; hs2, f2 = data[(e["fam"], Q, o, e["w2"])]
                ly = {}; ly3 = {}
                for l in range(nl):
                    p1 = _ll(m, hs1[l]); p2 = _ll(m, hs2[l])
                    ly[str(l)] = round(_jsd(p1, p2), 6); ly3[str(l)] = round(_j3(p1, p2, tids), 6)
                ev.append(dict(edge=e["id"], fam=e["fam"], query=Q, order=o,
                    cf=e["cf"], rel=e["cf"]==qi, layers=ly, layers3=ly3))
    for k, (hs, fin) in data.items():
        integ.append(dict(cell=str(k), d=round(_jsd(_ll(m, hs[nl-1]), fin), 8)))
    _save("evidence.json", dict(exp=c["experiment"], ts=datetime.now().isoformat(),
        nl=nl, nc=len(ps), nf=len(ps), edges=ev, integrity=integ))
    del m; print(f"Evidence: {len(ev)} meas, {nl} layers, integrity max={max(x['d'] for x in integ):.8f}")

def reduce(cp):
    c = _cfg(cp); g = c["gates"]; ew = c["early_window"]; a = c["anchor_layer"]
    with open(os.path.join(RDIR, "evidence.json")) as f: ev = json.load(f)
    ed = ev["edges"]; nl = ev["nl"]; eids = sorted(set(e["edge"] for e in ed))
    mx_int = max(x["d"] for x in ev["integrity"]); gi = mx_int <= g["logit_lens_integrity"]
    cl = {}; cl3 = {}; clR = {}; clI = {}
    for eid in eids:
        ee = [e for e in ed if e["edge"] == eid]
        rel = [e for e in ee if e["rel"]]; irr = [e for e in ee if not e["rel"]]
        d = {}; d3 = {}; dR = {}; dI = {}
        for l in range(nl):
            ls = str(l)
            dR[ls] = float(np.mean([e["layers"][ls] for e in rel]))
            dI[ls] = float(np.mean([e["layers"][ls] for e in irr]))
            d[ls] = dR[ls] - dI[ls]
            d3[ls] = float(np.mean([e["layers3"][ls] for e in rel]) - np.mean([e["layers3"][ls] for e in irr]))
        cl[eid] = d; cl3[eid] = d3; clR[eid] = dR; clI[eid] = dI
    rng = np.random.RandomState(c["bootstrap_seed"]); nb = c["bootstrap_resamples"]
    S = {l: float(np.mean([cl[e][str(l)] for e in eids])) for l in range(nl)}
    R = {l: float(np.mean([clR[e][str(l)] for e in eids])) for l in range(nl)}
    I_v = {l: float(np.mean([clI[e][str(l)] for e in eids])) for l in range(nl)}
    B = float(np.mean([S[l] for l in range(ew[0], ew[1]+1)]))
    G = {l: S[l] - B for l in range(nl)}
    OSQ = {l: (R[l]-I_v[l])/(R[l]+I_v[l]) if R[l]+I_v[l]>1e-10 else 0 for l in range(nl)}
    S3a = float(np.mean([cl3[e][str(a)] for e in eids]))
    pk = max(range(nl), key=lambda l: G[l])
    ci = {l: _bci([cl[e][str(l)] for e in eids], rng, nb) for l in range(nl)}
    ciG = {l: dict(est=round(G[l],6), lb=round(ci[l]["lb"]-B,6), ub=round(ci[l]["ub"]-B,6)) for l in range(nl)}
    ciRa = _bci([clR[e][str(a)] for e in eids], rng, nb)
    ciOa = _bci([(clR[e][str(a)]-clI[e][str(a)])/(clR[e][str(a)]+clI[e][str(a)])
                 if clR[e][str(a)]+clI[e][str(a)]>1e-10 else 0 for e in eids], rng, nb)
    ge = all(-g["early_null_band"]<=ci[l]["lb"] and ci[l]["ub"]<=g["early_null_band"] for l in range(ew[0],ew[1]+1))
    rw = c["resolution_window"]
    Gw = float(np.mean([G[l] for l in range(rw[0],rw[1]+1)]))
    Gw_lb = float(np.mean([ciG[l]["lb"] for l in range(rw[0],rw[1]+1)]))
    gw = Gw_lb >= g["G_window_lb"]; ga = ciG[a]["lb"] >= g["G_anchor_lb"]
    gR = ciRa["lb"] >= g["R_anchor_lb"]; gO = ciOa["lb"] >= g["OSQ_anchor_lb"]
    gt = g["peak_range"][0] <= pk <= g["peak_range"][1]
    onset = next((l for l in range(nl-1) if ciG[l]["lb"]>0 and ciG[l+1]["lb"]>0
                  and G[l]>=.10 and G[l+1]>=.10), None)
    go = onset is not None and g["onset_range"][0] <= onset <= g["onset_range"][1]
    gp = ci[nl-1]["lb"] >= g["persistence_lb"]
    for eid in eids:
        ee = [e for e in ed if e["edge"]==eid]
        for o in ["std","rev"]:
            ro=[e["layers"][str(a)] for e in ee if e["rel"] and e["order"]==o]
            io=[e["layers"][str(a)] for e in ee if not e["rel"] and e["order"]==o]
            cl[eid][f"S_{o}"] = float(np.mean(ro)-np.mean(io)) if ro and io else 0.0
    cis = _bci([cl[e]["S_std"] for e in eids], rng, nb)
    cir = _bci([cl[e]["S_rev"] for e in eids], rng, nb)
    gpr = cis["lb"]>=g["presentation_per_order_lb"] and cir["lb"]>=g["presentation_per_order_lb"]
    fsS = {fc["name"]: float(np.mean([cl[e][str(a)] for e in eids if e.startswith(fc["name"])]))
           for fc in c["families"]}
    gf = all(v>=g["family_S_lb"] for v in fsS.values())
    V = S3a/S[a] if abs(S[a])>1e-10 else 1.0
    ciV = _bci([cl3[e][str(a)] for e in eids], rng, nb)
    gv = (V < g["verbalizer_V_ub"] and ci[a]["lb"]-ciV["ub"] >= g["verbalizer_residual_lb"])
    mat = sum(1 for eid in eids if any(e["layers"][str(l)]>=g["material_floor"]
              for e in ed if e["edge"]==eid and e["rel"] for l in range(nl)))
    gm = mat >= g["min_eligible_edges"]
    core = gi and gm and ge and gw and ga and gR and gO and gt and go and gp and gpr and gf
    if not gi: vr = "INVALID_MEASUREMENT"
    elif core and gv: vr = "OBSERVATIONAL_SELECTIVITY_BROAD"
    elif core: vr = "OBSERVATIONAL_SELECTIVITY_VERBALIZER_SUFFICIENT"
    elif not gpr: vr = "PRESENTATION_SENSITIVE_SELECTIVITY"
    elif not gf: vr = "FAMILY_SPECIFIC_SELECTIVITY"
    elif not (ge and gt and go): vr = "SELECTIVITY_NOT_LATE_EMERGENT"
    elif not gp: vr = "TRANSIENT_LOGIT_LENS_SELECTIVITY"
    elif not (ga and gw and gR and gO): vr = "NO_REGISTERED_LATE_SELECTIVITY"
    else: vr = "INCONCLUSIVE_ALLOCATION_STOP"
    _save("verdict.json", dict(exp=c["experiment"], ts=datetime.now().isoformat(), verdict=vr,
        peak_layer=pk, onset_layer=onset, baseline=round(B,6),
        gates=dict(integrity=dict(ok=gi, max_d=round(mx_int,8)), material=dict(ok=gm, n=mat),
            early_null=dict(ok=ge), window=dict(ok=gw, Gw=round(Gw,6), lb=round(Gw_lb,6)),
            anchor=dict(ok=ga, G=ciG[a], R=ciRa, OSQ=ciOa, matR=gR, osq=gO),
            timing=dict(ok=gt and go, peak=pk, onset=onset),
            persistence=dict(ok=gp, S27=ci[nl-1]),
            presentation=dict(ok=gpr, std=cis, rev=cir),
            family=dict(ok=gf, S={k:round(v,6) for k,v in fsS.items()}),
            verbalizer=dict(ok=gv, V=round(V,6), S3=round(S3a,6), ciV=ciV)),
        profile={str(l):dict(S=ci[l],G=ciG[l],R=round(R[l],6),I=round(I_v[l],6),
                              OSQ=round(OSQ[l],6)) for l in range(nl)},
        bootstrap=dict(seed=c["bootstrap_seed"], n=nb)))
    P = lambda b: "PASS" if b else "FAIL"
    print(f"\nVERDICT: {vr}")
    print(f"  Integrity: {P(gi)} (d={mx_int:.8f})  Material: {P(gm)} ({mat}/12)")
    print(f"  Early null: {P(ge)} (B={B:.4f})  Window: {P(gw)} Gw={Gw:.4f}[{Gw_lb:.4f}]")
    print(f"  G(25): {P(ga)} {ciG[a]['est']:.4f}[{ciG[a]['lb']:.4f}]  R(25): {P(gR)} {ciRa['est']:.4f}[{ciRa['lb']:.4f}]")
    print(f"  OSQ(25): {P(gO)} {ciOa['est']:.4f}[{ciOa['lb']:.4f}]  peak={pk} onset={onset}")
    print(f"  S(27): {P(gp)} {ci[nl-1]['est']:.4f}[{ci[nl-1]['lb']:.4f}]  V: {P(gv)} {V:.4f}")
    print(f"  Pres: {P(gpr)} std={cis['est']:.4f}[{cis['lb']:.4f}] rev={cir['est']:.4f}[{cir['lb']:.4f}]")
    print(f"  Family: {P(gf)} {fsS}")

if __name__ == "__main__":
    if len(sys.argv) < 3 or "--config" not in sys.argv:
        print("Usage: run_observational_selectivity_quotient_v1.py <preflight|lock|produce|reduce> --config <path>"); sys.exit(1)
    {"preflight": preflight, "lock": lock, "produce": produce, "reduce": reduce}[sys.argv[1]](
        sys.argv[sys.argv.index("--config") + 1])
