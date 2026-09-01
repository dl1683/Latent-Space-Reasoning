"""QPC-1: Query-Port Composition v1.
Theory: Section 12, PREDICTIVE_FIBER_ACTION_ALGEBRA.md. Budget: 144 forwards.
Three-way clash: transplanted query state must select recipient's answer to the
transplanted query (t), not the donor's clean answer (d) or host inertia (s)."""
import torch, torch.nn.functional as F, json, hashlib, os, sys
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

RDIR = os.path.join(os.path.dirname(__file__), "results", "query_port_composition_v1")

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

def _save(name, obj):
    os.makedirs(RDIR, exist_ok=True)
    with open(os.path.join(RDIR, name), "w") as f: json.dump(obj, f, indent=2)

def _bci(vals, rng, nb):
    a = np.array(vals, dtype=float)
    b = sorted(float(np.mean(rng.choice(a, len(a), True))) for _ in range(nb))
    return dict(est=round(float(np.mean(a)), 6), lb=round(b[int(.025*nb)], 6), ub=round(b[int(.975*nb)], 6))

def _cells(c):
    """Build 36-cell grid. Each cell: host prompt, donor prompt, s/t/d answers.
    Host: x(q)=v_r, x(q̄)=v_{r+1}, asks q.  Donor: y(q)=v_r, y(q̄)=v_{r+2}, asks q̄.
    s=v_r (host clean), t=v_{r+1} (host answers transplanted query), d=v_{r+2} (donor clean)."""
    cells = []
    for fam in c["families"]:
        A, B = fam["entities"]; v = fam["values"]
        for on, tmpl in [("std", c["template_standard"]), ("rev", c["template_reversed"])]:
            for qi in range(2):
                Q, Qb = fam["entities"][qi], fam["entities"][1-qi]
                for r in range(3):
                    if qi == 0:
                        hva, hvb = v[r], v[(r+1)%3]
                        dva, dvb = v[r], v[(r+2)%3]
                    else:
                        hva, hvb = v[(r+1)%3], v[r]
                        dva, dvb = v[(r+2)%3], v[r]
                    cells.append(dict(
                        fam=fam["name"], order=on, qi=qi, r=r, Q=Q, Qb=Qb,
                        host=tmpl.format(A=A, B=B, va=hva, vb=hvb, Q=Q),
                        donor=tmpl.format(A=A, B=B, va=dva, vb=dvb, Q=Qb),
                        s=v[r], t=v[(r+1)%3], d=v[(r+2)%3],
                        cluster=f"{fam['name']}_r{r}"))
    return cells

def _clean_fwd(m, tok, prompt, layers, dev):
    cap = {}; hooks = []
    for L in layers:
        def mh(idx):
            def hf(_, __, out):
                h = out[0] if isinstance(out, tuple) else out
                cap[idx] = h[0, -1].detach().clone()
            return hf
        hooks.append(m.model.layers[L].register_forward_hook(mh(L)))
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad(): o = m(ids, use_cache=False)
    for h in hooks: h.remove()
    return cap, F.softmax(o.logits[0, -1], dim=-1)

def _patch_fwd(m, tok, prompt, layer, state, dev):
    fired = [False]
    def hf(_, __, out):
        o = out[0] if isinstance(out, tuple) else out
        h = o.clone(); h[0, -1] = state; fired[0] = True
        return (h,) + out[1:] if isinstance(out, tuple) else h
    hook = m.model.layers[layer].register_forward_hook(hf)
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad(): o = m(ids, use_cache=False)
    hook.remove()
    assert fired[0], f"Hook at layer {layer} did not fire"
    return F.softmax(o.logits[0, -1], dim=-1)

def preflight(cp):
    c = _cfg(cp)
    tok = AutoTokenizer.from_pretrained(c["model_id"], revision=c["model_revision"], trust_remote_code=True)
    cells = _cells(c); checks = []; vtok = {}
    for fam in c["families"]:
        for v in fam["values"]:
            t = tok.encode(" " + v)
            if len(t) != 1: checks.append(f"'{v}' multi-token: {t}")
            else: vtok[v] = t[0]
    for cell in cells:
        hl, dl = len(tok.encode(cell["host"])), len(tok.encode(cell["donor"]))
        if hl != dl: checks.append(f"len mismatch {cell['fam']} r{cell['r']} q{cell['qi']} {cell['order']}: h={hl} d={dl}")
    for cell in cells:
        if len({cell["s"], cell["t"], cell["d"]}) != 3:
            checks.append(f"non-distinct triplet {cell['fam']} r{cell['r']}")
    hosts = {cell["host"] for cell in cells}; donors = {cell["donor"] for cell in cells}
    if not donors.issubset(hosts): checks.append(f"donor prompts not subset of hosts ({len(donors-hosts)} extra)")
    valid = len(checks) == 0
    _save("preflight.json", dict(valid=valid, n_cells=len(cells), n_fwd=len(cells)*4,
                                  vtok=vtok, checks=checks))
    print(f"Preflight: {'PASS' if valid else 'FAIL'} -- {len(cells)} cells, {len(cells)*4} fwd")
    for ch in checks: print(f"  ! {ch}")

def lock(cp):
    c = _cfg(cp); m, tok = _model(c); dev = c["device"]; cells = _cells(c); checks = []
    for cell in cells:
        ids = tok(cell["host"], return_tensors="pt").input_ids.to(dev)
        with torch.no_grad(): g = tok.decode([int(torch.argmax(m(ids).logits[0,-1]))]).strip()
        checks.append(dict(fam=cell["fam"], r=cell["r"], qi=cell["qi"],
                          order=cell["order"], Q=cell["Q"], got=g, exp=cell["s"], ok=g==cell["s"]))
    nc = sum(1 for x in checks if x["ok"])
    _save("manifest.json", dict(exp=c["experiment"], ts=datetime.now().isoformat(),
        model=c["model_id"], rev=c["model_revision"], nl=m.config.num_hidden_layers,
        vs=m.config.vocab_size, rh=_sha(os.path.abspath(__file__)),
        ch=_sha(os.path.abspath(cp)), bl=checks, ok=f"{nc}/{len(checks)}"))
    del m; print(f"Locked: {nc}/{len(checks)} baselines correct")

def produce(cp):
    c = _cfg(cp); m, tok = _model(c); dev = c["device"]
    layers = c["intervention_layers"]; cells = _cells(c)
    vtok = {}
    for fam in c["families"]:
        for v in fam["values"]: vtok[v] = tok.encode(" " + v)[0]
    cache = {}
    for i, cell in enumerate(cells):
        if cell["host"] not in cache:
            cap, soft = _clean_fwd(m, tok, cell["host"], layers, dev)
            cache[cell["host"]] = (cap, soft)
            print(f"  [clean {len(cache)}/36] {cell['fam']} q{cell['qi']} r{cell['r']} {cell['order']}")
    assert len(cache) == 36, f"Expected 36 unique prompts, got {len(cache)}"
    evidence = []
    for i, cell in enumerate(cells):
        hcap, hsoft = cache[cell["host"]]; dcap, _ = cache[cell["donor"]]
        si, ti, di = vtok[cell["s"]], vtok[cell["t"]], vtok[cell["d"]]
        def ex(sf):
            return dict(probs={cell["s"]: round(float(sf[si]),8), cell["t"]: round(float(sf[ti]),8),
                               cell["d"]: round(float(sf[di]),8)},
                        top1=tok.decode([int(torch.argmax(sf))]).strip(),
                        triplet_mass=round(float(sf[si]+sf[ti]+sf[di]), 6))
        sp = _patch_fwd(m, tok, cell["host"], layers[0], hcap[layers[0]], dev)
        dp21 = _patch_fwd(m, tok, cell["host"], layers[0], dcap[layers[0]], dev)
        dp25 = _patch_fwd(m, tok, cell["host"], layers[1], dcap[layers[1]], dev)
        e = ex(sp); e["max_diff"] = round(float(torch.max(torch.abs(hsoft - sp))), 8)
        evidence.append(dict(fam=cell["fam"], order=cell["order"], qi=cell["qi"], r=cell["r"],
            Q=cell["Q"], Qb=cell["Qb"], s=cell["s"], t=cell["t"], d=cell["d"],
            cluster=cell["cluster"], clean=ex(hsoft), self_patch=e,
            donor_L21=ex(dp21), donor_L25=ex(dp25)))
        if (i+1) % 6 == 0: print(f"  [patch {i+1}/36]")
    _save("evidence.json", dict(exp=c["experiment"], ts=datetime.now().isoformat(),
        n_cells=len(cells), n_fwd=144, layers=layers, evidence=evidence))
    del m; print(f"Evidence: {len(evidence)} cells, 144 forwards")

def reduce(cp):
    c = _cfg(cp); g = c["gates"]
    with open(os.path.join(RDIR, "evidence.json")) as f: ev = json.load(f)
    cells = ev["evidence"]; rng = np.random.RandomState(c["bootstrap_seed"]); nb = c["bootstrap_resamples"]
    clusters = sorted(set(cell["cluster"] for cell in cells))
    families = sorted(set(cell["fam"] for cell in cells))
    g_carrier = ev["n_cells"] == 36
    sp_max = max(cell["self_patch"]["max_diff"] for cell in cells)
    g_self = sp_max <= g["self_patch_max"]
    cc = [1 if cell["clean"]["top1"]==cell["s"] else 0 for cell in cells]
    cc_rate = sum(cc)/len(cc)
    cl_rates = {cl: np.mean([cc[i] for i,c2 in enumerate(cells) if c2["cluster"]==cl]) for cl in clusters}
    cc_ci = _bci(list(cl_rates.values()), rng, nb)
    fam_cc = {f: np.mean([cc[i] for i,c2 in enumerate(cells) if c2["fam"]==f]) for f in families}
    g_clean = cc_rate>=g["clean_interface_rate"] and cc_ci["lb"]>=g["clean_interface_cluster_lb"] and all(v>=g["clean_interface_family_lb"] for v in fam_cc.values())
    via = [1 if cell["clean"]["triplet_mass"]>0 and cell["donor_L21"]["triplet_mass"]/cell["clean"]["triplet_mass"]>=0.5 else 0 for cell in cells]
    via_rate = sum(via)/len(via)
    via_cl = {cl: np.mean([via[i] for i,c2 in enumerate(cells) if c2["cluster"]==cl]) for cl in clusters}
    via_ci = _bci(list(via_cl.values()), rng, nb)
    g_via = via_rate>=g["l21_viability_mass_rate"] and via_ci["lb"]>=g["l21_viability_mass_lb"]
    def tn(pr, s, t, d):
        total = pr[s]+pr[t]+pr[d]
        if total<1e-10: return {s:1/3, t:1/3, d:1/3}
        return {s:pr[s]/total, t:pr[t]/total, d:pr[d]/total}
    F21=[]; C21=[]; W21=[]; F25=[]; C25=[]; W25=[]; D21=[]; S21=[]
    for cell in cells:
        s,t,d = cell["s"], cell["t"], cell["d"]
        r21 = tn(cell["donor_L21"]["probs"], s, t, d)
        F21.append(1 if r21[t]>=r21[s] and r21[t]>=r21[d] else 0)
        C21.append(r21[t]-r21[d]); W21.append(r21[t]-r21[s])
        D21.append(1 if r21[d]>=r21[s] and r21[d]>=r21[t] else 0)
        S21.append(1 if r21[s]>=r21[t] and r21[s]>=r21[d] else 0)
        r25 = tn(cell["donor_L25"]["probs"], s, t, d)
        F25.append(1 if r25[t]>=r25[s] and r25[t]>=r25[d] else 0)
        C25.append(r25[t]-r25[d]); W25.append(r25[t]-r25[s])
    def cl_mean(arr): return {cl: float(np.mean([arr[i] for i,c2 in enumerate(cells) if c2["cluster"]==cl])) for cl in clusters}
    ciF21 = _bci(list(cl_mean(F21).values()), rng, nb)
    ciC21 = _bci(list(cl_mean(C21).values()), rng, nb)
    ciW21 = _bci(list(cl_mean(W21).values()), rng, nb)
    ciC25 = _bci(list(cl_mean(C25).values()), rng, nb)
    ciL = _bci([cl_mean(C21)[cl]-cl_mean(C25)[cl] for cl in clusters], rng, nb)
    ciD21 = _bci(list(cl_mean(D21).values()), rng, nb)
    ciS21 = _bci(list(cl_mean(S21).values()), rng, nb)
    ciF25 = _bci(list(cl_mean(F25).values()), rng, nb)
    ciW25 = _bci(list(cl_mean(W25).values()), rng, nb)
    g_target = ciF21["est"]>=g["target_following_est"] and ciF21["lb"]>=g["target_following_lb"]
    g_donor = ciC21["est"]>=g["beats_donor_est"] and ciC21["lb"]>g["beats_donor_lb"]
    g_host = ciW21["est"]>=g["beats_host_est"] and ciW21["lb"]>g["beats_host_lb"]
    oF = {}; oC = {}
    for o in ["std","rev"]:
        oi = [i for i,c2 in enumerate(cells) if c2["order"]==o]
        oF[o] = round(float(np.mean([F21[i] for i in oi])), 6)
        oC[o] = round(float(np.mean([C21[i] for i in oi])), 6)
    g_pres = all(oF[o]>=g["presentation_F_lb"] and oC[o]>0 for o in ["std","rev"])
    fF = {}; fC = {}
    for f in families:
        fi = [i for i,c2 in enumerate(cells) if c2["fam"]==f]
        fF[f] = round(float(np.mean([F21[i] for i in fi])), 6)
        fC[f] = round(float(np.mean([C21[i] for i in fi])), 6)
    g_fam = all(fF[f]>=g["family_F_lb"] and fC[f]>=g["family_C_lb"] for f in families)
    g_loc = ciL["est"]>=g["localization_est"] and ciL["lb"]>g["localization_lb"]
    primary = g_carrier and g_self and g_clean and g_via and g_target and g_donor and g_host and g_pres and g_fam
    if not (g_carrier and g_self): vr = "INVALID_INSTRUMENT"
    elif not (g_clean and g_via): vr = "NO_VIABLE_QUERY_PORT"
    elif primary and g_loc: vr = "QUERY_PORT_COMPOSITION_LOCALIZED"
    elif primary: vr = "QUERY_PORT_COMPOSITION_REGISTERED"
    elif ciD21["est"]>=0.70 and ciD21["lb"]>=0.50 and ciC21["ub"]<=0: vr = "DONOR_VERBALIZER_COPY"
    elif ciS21["est"]>=0.70 and ciS21["lb"]>=0.50 and ciW21["ub"]<=0: vr = "HOST_INERTIA"
    elif (ciF25["est"]>=g["target_following_est"] and ciF25["lb"]>=g["target_following_lb"]
          and ciC25["est"]>=g["beats_donor_est"] and ciC25["lb"]>0
          and ciW25["est"]>=g["beats_host_est"] and ciW25["lb"]>0): vr = "LATE_PORT_ONLY"
    else: vr = "INCONCLUSIVE_ALLOCATION_STOP"
    _save("verdict.json", dict(exp=c["experiment"], ts=datetime.now().isoformat(), verdict=vr,
        gates=dict(carrier=dict(ok=g_carrier, n=ev["n_cells"]),
            self_patch=dict(ok=g_self, max_diff=round(sp_max,8)),
            clean_interface=dict(ok=g_clean, rate=round(cc_rate,4), cluster_ci=cc_ci,
                family={k:round(float(v),4) for k,v in fam_cc.items()}),
            l21_viability=dict(ok=g_via, rate=round(via_rate,4), cluster_ci=via_ci),
            target_following=dict(ok=g_target, F21=ciF21),
            beats_donor=dict(ok=g_donor, C21=ciC21),
            beats_host=dict(ok=g_host, W21=ciW21),
            presentation=dict(ok=g_pres, F21=oF, C21=oC),
            family_robustness=dict(ok=g_fam, F21=fF, C21=fC),
            localization=dict(ok=g_loc, L=ciL)),
        estimands=dict(F21=ciF21, C21=ciC21, W21=ciW21, F25=ciF25, C25=ciC25,
            L=ciL, D21=ciD21, S21=ciS21),
        bootstrap=dict(seed=c["bootstrap_seed"], n=nb)))
    P = lambda b: "PASS" if b else "FAIL"
    print(f"\nVERDICT: {vr}")
    print(f"  Carrier: {P(g_carrier)}  Self-patch: {P(g_self)} (max={sp_max:.8f})")
    print(f"  Clean: {P(g_clean)} ({cc_rate:.2%})  L21 viable: {P(g_via)} ({via_rate:.2%})")
    print(f"  F(21): {P(g_target)} {ciF21['est']:.4f}[{ciF21['lb']:.4f}]")
    print(f"  C(21): {P(g_donor)} {ciC21['est']:.4f}[{ciC21['lb']:.4f}]")
    print(f"  W(21): {P(g_host)} {ciW21['est']:.4f}[{ciW21['lb']:.4f}]")
    print(f"  Pres: {P(g_pres)} F_std={oF['std']:.4f} F_rev={oF['rev']:.4f}")
    print(f"  Family: {P(g_fam)} {fF}")
    print(f"  Local: {P(g_loc)} L={ciL['est']:.4f}[{ciL['lb']:.4f}]")

if __name__ == "__main__":
    if len(sys.argv) < 3 or "--config" not in sys.argv:
        print("Usage: run_query_port_composition_v1.py <preflight|lock|produce|reduce> --config <path>"); sys.exit(1)
    {"preflight": preflight, "lock": lock, "produce": produce, "reduce": reduce}[sys.argv[1]](
        sys.argv[sys.argv.index("--config") + 1])
