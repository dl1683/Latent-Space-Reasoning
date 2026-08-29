"""Operational interchangeability of paraphrases in a real residual stream (Qwen3-1.7B-Base, CPU).

Donor block-L output at the anchor token REPLACES the recipient's during prefill; the readout is a signature of three
downstream probe margins (mean log-likelihood of continuation A minus B). Arms: self-swap sham, same-class donor,
matched cross-class donor, unrelated-class donor. All text and gates are fixed in the config before any result.

    python experiments/run_interchange.py --config experiments/config/interchange_v1.json
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from substitution_probe import SubstitutionProbe


class Model:
    def __init__(self, cfg):
        self.cfg = cfg; self.sp = SubstitutionProbe(cfg["model_id"], revision=cfg["revision"]); self.m = self.sp.model; self.tok = self.sp.tok
        assert self.sp.revision == cfg["revision"]; self.layer = self.m.model.layers[cfg["layer"]]; self.forwards = 0

    def ids(self, text): return self.tok.encode(text, add_special_tokens=False)

    def capture(self, ctx_ids):
        """Block-L output at the final (anchor) token of a context."""
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
            toks = self.ids(cont); lp = torch.log_softmax(last.float(), -1)[toks[0]].item(); p = _clone(past)
            o = self.m(input_ids=torch.tensor([toks]), past_key_values=p, use_cache=True); self.forwards += 1
            lps = torch.log_softmax(o.logits[0, :-1].float(), -1); lp += sum(lps[i, toks[i + 1]].item() for i in range(len(toks) - 1))
            return lp / len(toks)
        return np.array([ll(a) - ll(b) for a, b in self.cfg["probes"]])


def _clone(past):
    import copy; return copy.deepcopy(past)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8")); out_dir = f"experiments/results/{cfg['name']}"; os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "run.log"), "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    shas = {k: hashlib.sha256(open(v, "rb").read()).hexdigest() for k, v in (("runner", __file__), ("config", a.config))}
    t0 = time.time(); M = Model(cfg); log(f"loaded {cfg['model_id']} rev={M.sp.revision} ({time.time()-t0:.0f}s)")
    classes = ["cat", "dog"]; pos = cfg["positive_class"]; sign = {c: (1 if c == pos else -1) for c in classes}
    ctx = lambda text: M.ids(text + cfg["anchor"])
    lengths = {k: [len(ctx(t)) for c in classes for t in cfg[k][c]] for k in ("calibration", "test")}; lengths["unrelated"] = [len(ctx(t)) for t in cfg["unrelated"]]
    log(f"context token lengths: {lengths}")
    res = {"config": cfg["name"], "sha256": shas, "revision": M.sp.revision, "lengths": lengths}
    save = lambda: json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1, default=float)
    # native signatures
    nat = {k: {c: [M.signature(ctx(t)) for t in cfg[k][c]] for c in classes} for k in ("calibration", "test")}
    res["native"] = {k: {c: [s.tolist() for s in v[c]] for c in classes} for k, v in nat.items()}
    correct = {c: int(sum((np.sign(s) == sign[c]).sum() for s in nat["test"][c])) for c in classes}
    g = cfg["baseline_gate"]; ok = sum(correct.values()) >= g["total_min"] and all(v >= g["per_class_min"] for v in correct.values())
    log(f"native held-out probe decisions correct: {correct} (of 12 each) -> BASELINE {'PASS' if ok else 'FAIL - ABANDON TASK'}"); res["baseline"] = {"correct": correct, "passed": ok}; save()
    if not ok: return
    # calibration-only scale and tolerance
    cal = np.array([s for c in classes for s in nat["calibration"][c]]); scale = cal.std(0, ddof=1) + 1e-6
    z = lambda s: s / scale
    within = [np.linalg.norm(z(x) - z(y)) for c in classes for x, y in itertools.combinations(nat["calibration"][c], 2)]
    cent = {c: np.mean([z(s) for s in nat["calibration"][c]], 0) for c in classes}; sep = float(np.linalg.norm(cent["cat"] - cent["dog"]))
    # arms on held-out recipients
    donors = {c: [M.capture(ctx(t)) for t in cfg["calibration"][c]] for c in classes}; unrel = [M.capture(ctx(t)) for t in cfg["unrelated"]]
    rows = []; floor = []
    for c in classes:
        other = classes[1 - classes.index(c)]
        for i, t in enumerate(cfg["test"][c]):
            r = ctx(t); own = M.capture(r); native = z(nat["test"][c][i])
            selfswap = z(M.signature(r, own)); floor.append(float(np.linalg.norm(selfswap - native)))
            same = z(M.signature(r, donors[c][i])); cross = z(M.signature(r, donors[other][i])); un = z(M.signature(r, unrel[i]))
            row = {"class": c, "i": i, "native": native.tolist(), "self": selfswap.tolist(), "same": same.tolist(), "cross": cross.tolist(), "unrelated": un.tolist(),
                   "d_same": float(np.linalg.norm(same - native)), "d_cross": float(np.linalg.norm(cross - native)), "d_unrelated": float(np.linalg.norm(un - native)),
                   "cross_flips": int((np.sign(cross) != np.sign(native)).sum()), "cross_toward_other": float(np.dot(native - cross, cent[c] - cent[other]) / sep)}
            rows.append(row); log(f"{c}{i}: d_self={floor[-1]:.3f} d_same={row['d_same']:.3f} d_cross={row['d_cross']:.3f} d_unrel={row['d_unrelated']:.3f} flips={row['cross_flips']} toward_other={row['cross_toward_other']:.2f}"); res["rows"] = rows; save()
    tau = float(np.percentile(within, cfg["gates"]["tau_percentile"]) + np.median(floor))
    G = cfg["gates"]; d_same = [r["d_same"] for r in rows]; frac = [r["cross_toward_other"] for r in rows]
    summary = {"tau": tau, "within_calibration_90pct": float(np.percentile(within, 90)), "self_swap_floor_median": float(np.median(floor)), "class_separation": sep,
               "same_median": float(np.median(d_same)), "same_within_tau": int(sum(d <= tau for d in d_same)),
               "cross_median_fraction": float(np.median(frac)), "cross_flip_two": int(sum(r["cross_flips"] >= 2 for r in rows)),
               "cross_minus_unrelated_median": float(np.median([(r["d_cross"] - r["d_unrelated"]) / sep for r in rows]))}
    passed = {"same_class_interchangeable": summary["same_median"] <= tau and summary["same_within_tau"] >= G["same_within_tau_min"],
              "cross_class_moves": summary["cross_median_fraction"] >= G["cross_median_fraction_min"] and summary["cross_flip_two"] >= G["cross_flip_two_min"],
              "cross_beats_unrelated": summary["cross_minus_unrelated_median"] >= G["cross_minus_unrelated_min"]}
    # exact permutation of donor-class labels over the 8 recipients (paired same-vs-cross distance)
    diffs = np.array([r["d_cross"] - r["d_same"] for r in rows]); obs = diffs.mean()
    perm = np.mean([np.mean(diffs * np.array(sgn)) >= obs for sgn in itertools.product([1, -1], repeat=len(diffs))])
    summary["perm_p_cross_gt_same"] = float(perm); res["summary"] = summary; res["gates"] = passed
    status = "BOUNDED POSITIVE — OPERATIONAL INTERCHANGEABILITY (this construction)" if all(passed.values()) else "FAIL — " + ", ".join(k for k, v in passed.items() if not v)
    res["status"] = status; save(); log(json.dumps(summary, indent=1)); log(f"gates: {passed}"); log(f"STATUS: {status} ({M.forwards} forwards, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
