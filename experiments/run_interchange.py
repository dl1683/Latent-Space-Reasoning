"""Operational interchangeability of paraphrases in a real residual stream (Qwen3-1.7B-Base, CPU).

Donor block-L output at the anchor token REPLACES the recipient's during prefill; the readout is a signature of three
downstream probe margins (mean log-likelihood of continuation A minus B). v2 (locked, direction round 8): probe margins
are centred by a calibration-only midpoint and scaled by a calibration-only pooled-within SD floored by the self-swap
numerical floor; fractional donor movement T = ((u_arm - u_native) . d) / |delta|^2; arms native / self-sham /
same-state donor / opposite-state donor / on-manifold third-state donor; exact 2^8 recipient sign-flip test.

    python experiments/run_interchange.py --config experiments/config/interchange_v2.json
"""
from __future__ import annotations
import argparse, copy, hashlib, itertools, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from substitution_probe import SubstitutionProbe


class Model:
    def __init__(self, cfg):
        self.cfg = cfg; self.sp = SubstitutionProbe(cfg["model_id"], revision=cfg["revision"]); self.m = self.sp.model; self.tok = self.sp.tok
        assert self.sp.revision == cfg["revision"]; self.layer = self.m.model.layers[cfg["layer"]]; self.forwards = 0

    def ids(self, text): return self.tok.encode(text, add_special_tokens=False)

    def capture(self, ctx_ids):
        out = self.m(input_ids=torch.tensor([ctx_ids]), output_hidden_states=True); self.forwards += 1
        return out.hidden_states[self.cfg["layer"] + 1][0, -1].clone()

    def signature(self, ctx_ids, donor=None):
        """Three signed probe margins after prefill with the anchor activation optionally replaced by `donor`."""
        handle = None
        if donor is not None:
            def hook(mod, i, o):
                h = o[0] if isinstance(o, tuple) else o; h = h.clone(); h[:, -1, :] = donor.to(h.dtype)
                return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
            handle = self.layer.register_forward_hook(hook)
        try: out = self.m(input_ids=torch.tensor([ctx_ids]), use_cache=True); self.forwards += 1
        finally:
            if handle: handle.remove()
        past, last = out.past_key_values, out.logits[0, -1]
        def ll(cont):
            toks = self.ids(cont); lp = torch.log_softmax(last.float(), -1)[toks[0]].item()
            o = self.m(input_ids=torch.tensor([toks]), past_key_values=copy.deepcopy(past), use_cache=True); self.forwards += 1
            lps = torch.log_softmax(o.logits[0, :-1].float(), -1); lp += sum(lps[i, toks[i + 1]].item() for i in range(len(toks) - 1))
            return lp / len(toks)
        return np.array([ll(a) - ll(b) for a, b in self.cfg["probes"]])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8")); out_dir = f"experiments/results/{cfg['name']}"; os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "run.log"), "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    shas = {k: hashlib.sha256(open(v, "rb").read()).hexdigest() for k, v in (("runner", __file__), ("config", a.config))}
    t0 = time.time(); M = Model(cfg); log(f"loaded {cfg['model_id']} rev={M.sp.revision} ({time.time()-t0:.0f}s)")
    classes = ["cat", "dog"]; pos = cfg["positive_class"]; sign = {c: (1 if c == pos else -1) for c in classes}
    ctx = lambda text: M.ids(text + cfg["anchor"])
    lengths = {k: [len(ctx(t)) for c in classes for t in cfg[k][c]] for k in ("calibration", "test")}; lengths["third"] = [len(ctx(t)) for t in cfg["third"]]
    log(f"context token lengths: {lengths}"); assert len({l for v in lengths.values() for l in v}) == 1, "contexts must be exactly length-matched"
    res = {"config": cfg["name"], "sha256": shas, "revision": M.sp.revision, "lengths": lengths}
    save = lambda: json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1, default=float)
    with torch.no_grad():
        # --- calibration-only centring and scale ---
        cal = {c: np.array([M.signature(ctx(t)) for t in cfg["calibration"][c]]) for c in classes}
        b = (cal["cat"].mean(0) + cal["dog"].mean(0)) / 2; within = np.sqrt(((cal["cat"] - cal["cat"].mean(0)) ** 2).sum(0) + ((cal["dog"] - cal["dog"].mean(0)) ** 2).sum(0)) / np.sqrt(6)
        floor = []
        for c in classes:                                                   # self-swap numerical floor on calibration contexts
            for t in cfg["calibration"][c]:
                r = ctx(t); floor.append(np.abs(M.signature(r, M.capture(r)) - M.signature(r)))
        eta = np.median(np.array(floor), 0) + 1e-6; s = np.maximum(within, eta); u = lambda m: (m - b) / s
        ucal = {c: np.array([u(m) for m in cal[c]]) for c in classes}; delta = ucal["cat"].mean(0) - ucal["dog"].mean(0); sep2 = float(delta @ delta)
        tau_pairs = [float(np.linalg.norm(x - y)) for c in classes for x, y in itertools.combinations(ucal[c], 2)]
        tau = float(np.percentile(tau_pairs, cfg["gates"]["tau_percentile"]) + np.median([np.linalg.norm(f / s) for f in floor]))
        res["calibration"] = {"midpoint": b.tolist(), "scale": s.tolist(), "eta": eta.tolist(), "delta": delta.tolist(), "sep2": sep2, "tau": tau}; save()
        log(f"midpoint={np.round(b,3).tolist()} scale={np.round(s,3).tolist()} delta={np.round(delta,2).tolist()} tau={tau:.3f}")
        # --- native held-out validity (centred decisions) ---
        nat = {c: [u(M.signature(ctx(t))) for t in cfg["test"][c]] for c in classes}
        correct = {c: int(sum((np.sign(x) == sign[c]).sum() for x in nat[c])) for c in classes}
        g = cfg["baseline_gate"]; ok = sum(correct.values()) >= g["total_min"] and all(v >= g["per_class_min"] for v in correct.values())
        res["native"] = {"u": {c: [x.tolist() for x in nat[c]] for c in classes}, "correct": correct, "passed": ok}; save()
        log(f"native held-out centred probe decisions correct: {correct} (of 12 each) -> BASELINE {'PASS' if ok else 'FAIL'}")
        if not ok: res["status"] = "FAIL — FIXED BLOCK-12 SINGLE-ANCHOR INTERCHANGE CONSTRUCTION (native validity)"; save(); log(res["status"]); return
        # --- arms on held-out recipients ---
        donors = {c: [M.capture(ctx(t)) for t in cfg["calibration"][c]] for c in classes}; third = [M.capture(ctx(t)) for t in cfg["third"]]
        rows = []
        for ci, c in enumerate(classes):
            other = classes[1 - ci]; d = delta * sign[c] * -1                        # desired donor direction: toward the opposite class
            for i, t in enumerate(cfg["test"][c]):
                r = ctx(t); own = M.capture(r); native = nat[c][i]
                arms = {"self": u(M.signature(r, own)), "same": u(M.signature(r, donors[c][i])), "cross": u(M.signature(r, donors[other][i])), "third": u(M.signature(r, third[ci * 4 + i]))}
                T = {k: float(((v - native) @ d) / sep2) for k, v in arms.items()}
                flips = int(((np.sign(arms["cross"]) != np.sign(native)) & (np.sign(arms["cross"]) == sign[other])).sum())
                row = {"class": c, "i": i, "native": native.tolist(), **{k: v.tolist() for k, v in arms.items()}, "T": T, "d_same": float(np.linalg.norm(arms["same"] - native)), "d_self": float(np.linalg.norm(arms["self"] - native)), "cross_flips": flips}
                rows.append(row); log(f"{c}{i}: d_self={row['d_self']:.3f} d_same={row['d_same']:.3f} T_same={T['same']:.2f} T_cross={T['cross']:.2f} T_third={T['third']:.2f} flips={flips}"); res["rows"] = rows; save()
        G = cfg["gates"]; d_same = [r["d_same"] for r in rows]; Tc = [r["T"]["cross"] for r in rows]; Tt = [r["T"]["third"] for r in rows]; diff = np.array(Tc) - np.array(Tt)
        obs = diff.mean(); perm = float(np.mean([np.mean(diff * np.array(sg)) >= obs for sg in itertools.product([1, -1], repeat=len(diff))]))
        summary = {"same_median": float(np.median(d_same)), "same_within_tau": int(sum(x <= tau for x in d_same)), "tau": tau,
                   "cross_T_median": float(np.median(Tc)), "cross_flip_two": int(sum(r["cross_flips"] >= 2 for r in rows)),
                   "specificity_median": float(np.median(diff)), "specificity_paired_positive": int((diff > 0).sum()), "perm_p": perm,
                   "by_class": {c: {"T_cross": [r["T"]["cross"] for r in rows if r["class"] == c], "T_third": [r["T"]["third"] for r in rows if r["class"] == c]} for c in classes}}
        gates = {"same_state": summary["same_median"] <= tau and summary["same_within_tau"] >= G["same_within_tau_min"],
                 "cross_state": summary["cross_T_median"] >= G["cross_T_median_min"] and summary["cross_flip_two"] >= G["cross_flip_two_min"],
                 "specificity": summary["specificity_median"] >= G["specificity_median_min"] and summary["specificity_paired_positive"] >= G["specificity_paired_positive_min"] and perm <= G["perm_p_max"]}
        status = ("BOUNDED POSITIVE — OPERATIONAL INTERCHANGEABILITY (this construction)" if all(gates.values()) else
                  "STATE-DIRECTED STEERING WITHOUT INTERCHANGEABILITY" if (gates["cross_state"] and gates["specificity"] and not gates["same_state"]) else
                  "FAIL — FIXED BLOCK-12 SINGLE-ANCHOR INTERCHANGE CONSTRUCTION")
        res["summary"] = summary; res["gates"] = gates; res["status"] = status; save()
        log(json.dumps(summary, indent=1)); log(f"gates: {gates}"); log(f"STATUS: {status} ({M.forwards} forwards, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
