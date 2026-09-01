"""
Commitment Hysteresis v1: Does a prefix transplant at L21 leave a causal
trace that survives restoring the original prefix after commitment (L25)?

Design gate: Codex design gate (2026-08-31).
Theory: PREDICTIVE_FIBER_ACTION_ALGEBRA.md Section 9.
Budget: 456 CPU forward passes.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import json
import hashlib
import os
import sys
import time
from datetime import datetime

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DEVICE = "cpu"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "commitment_hysteresis_v1")


def load_model():
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION,
        torch_dtype=torch.float32, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()
    return model, tok


def sha256_str(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompts(cfg):
    """Build all prompts for all families, worlds, and queries."""
    template = cfg["template"]
    prompts = []
    for fam in cfg["families"]:
        ents = fam["entities"]
        vals = fam["values"]
        for va_idx in range(2):
            for vb_idx in range(2):
                va = vals[va_idx]
                vb = vals[vb_idx]
                world_id = f"w{va_idx}{vb_idx}"
                for qi, qent in enumerate(ents):
                    prompt = template.format(
                        A=ents[0], B=ents[1], va=va, vb=vb, Q=qent
                    )
                    expected = va if qi == 0 else vb
                    prompts.append({
                        "family": fam["name"],
                        "world": world_id,
                        "query_entity": qent,
                        "query_idx": qi,
                        "expected": expected,
                        "prompt": prompt,
                        "entity_a": ents[0],
                        "entity_b": ents[1],
                        "val_a": va,
                        "val_b": vb,
                    })
    return prompts


def find_prefix_mask(tok, prompt, query_entity):
    """Find token indices wholly before the query entity on the second line.
    Uses offset mapping to avoid substring-first-occurrence bugs."""
    enc = tok(prompt, return_offsets_mapping=True, return_tensors="pt")
    ids = enc.input_ids[0]
    offsets = enc.offset_mapping[0]

    query_line_start = prompt.index("\n") + 1
    query_entity_start = prompt.index(query_entity, query_line_start)

    prefix_mask = []
    for i, (s, e) in enumerate(offsets):
        s, e = int(s), int(e)
        if e <= query_entity_start and s < e:
            prefix_mask.append(i)

    boundary_ok = True
    for i, (s, e) in enumerate(offsets):
        s, e = int(s), int(e)
        if s < query_entity_start < e and s != e:
            boundary_ok = False
            break

    return prefix_mask, boundary_ok, ids


def preflight(cfg, tok):
    """Tokenizer-only integrity checks."""
    prompts = build_prompts(cfg)
    results = {"valid": True, "checks": [], "n_prompts": len(prompts)}

    for p in prompts:
        mask, boundary_ok, ids = find_prefix_mask(tok, p["prompt"], p["query_entity"])
        check = {
            "family": p["family"],
            "world": p["world"],
            "query": p["query_entity"],
            "n_tokens": int(ids.shape[0]),
            "n_prefix": len(mask),
            "boundary_ok": boundary_ok,
            "prompt_hash": sha256_str(p["prompt"])[:16],
        }
        if not boundary_ok:
            check["error"] = "token straddles declaration/query boundary"
            results["valid"] = False
        results["checks"].append(check)

    families = {}
    for p in prompts:
        key = (p["family"], p["query_entity"])
        _, _, ids = find_prefix_mask(tok, p["prompt"], p["query_entity"])
        n = int(ids.shape[0])
        if key not in families:
            families[key] = []
        families[key].append(n)

    for key, lengths in families.items():
        if len(set(lengths)) > 1:
            results["valid"] = False
            results["checks"].append({
                "error": f"length mismatch in family {key}: {set(lengths)}"
            })

    for fam in cfg["families"]:
        for qi, qent in enumerate(fam["entities"]):
            worlds = [p for p in prompts
                      if p["family"] == fam["name"] and p["query_entity"] == qent]
            if len(worlds) < 2:
                continue
            ref_mask, _, ref_ids = find_prefix_mask(tok, worlds[0]["prompt"], qent)
            ref_suffix = ref_ids[max(ref_mask) + 1:] if ref_mask else ref_ids
            for w in worlds[1:]:
                w_mask, _, w_ids = find_prefix_mask(tok, w["prompt"], qent)
                w_suffix = w_ids[max(w_mask) + 1:] if w_mask else w_ids
                if not torch.equal(ref_suffix, w_suffix):
                    results["valid"] = False
                    results["checks"].append({
                        "error": f"suffix mismatch: {worlds[0]['world']} vs {w['world']} "
                                 f"for {fam['name']}/{qent}"
                    })

    val_tokens = set()
    for fam in cfg["families"]:
        for v in fam["values"]:
            toks = tok.encode(" " + v)
            if len(toks) != 1:
                results["valid"] = False
                results["checks"].append({
                    "error": f"value '{v}' not single-token: {toks}"
                })
            else:
                val_tokens.add((v, toks[0]))

    results["val_tokens"] = {v: t for v, t in val_tokens}
    return results


def capture_all_states(model, tok, prompt):
    """Run the model and capture post-block hidden states at every layer."""
    states = {}
    hooks = []
    for i, layer in enumerate(model.model.layers):
        def make_hook(idx):
            def hook_fn(module, inp, output):
                if isinstance(output, tuple):
                    states[idx] = output[0].detach().clone()
                else:
                    states[idx] = output.detach().clone()
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(i)))

    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(ids, use_cache=False)
    for h in hooks:
        h.remove()

    final_logits = out.logits[0, -1]
    final_dist = F.softmax(final_logits, dim=-1)
    return states, final_dist, final_logits


def run_with_prefix_swap(model, tok, host_prompt, host_states, donor_states,
                         prefix_mask, swap_layer, restore_layer=None,
                         restore_states=None, full_restore_layer=None,
                         full_restore_states=None):
    """Run the host prompt with prefix positions replaced by donor at swap_layer,
    optionally restored at restore_layer."""
    actions_done = {"swap": False, "restore": False, "full_restore": False}
    captured = {}

    def make_swap_hook(layer_idx):
        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                h = output[0].clone()
            else:
                h = output.clone()

            if layer_idx == swap_layer and not actions_done["swap"]:
                actions_done["swap"] = True
                for pos in prefix_mask:
                    h[0, pos, :] = donor_states[layer_idx][0, pos, :]

            if (restore_layer is not None and layer_idx == restore_layer
                    and restore_states is not None and not actions_done["restore"]):
                actions_done["restore"] = True
                for pos in prefix_mask:
                    h[0, pos, :] = restore_states[layer_idx][0, pos, :]

            if (full_restore_layer is not None and layer_idx == full_restore_layer
                    and full_restore_states is not None
                    and not actions_done["full_restore"]):
                actions_done["full_restore"] = True
                h = full_restore_states[layer_idx].clone()

            captured[layer_idx] = h.detach().clone()

            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        return hook_fn

    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_swap_hook(i)))

    ids = tok(host_prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(ids, use_cache=False)
    for h in hooks:
        h.remove()

    final_logits = out.logits[0, -1]
    final_dist = F.softmax(final_logits, dim=-1)
    return final_dist, final_logits, captured


def jsd_normalized(p, q):
    """Normalized sqrt(JSD / log2), range [0,1]."""
    m = (p + q) / 2
    eps = 1e-10
    jsd = 0.5 * ((p + eps) * ((p + eps) / (m + eps)).log()).sum() \
        + 0.5 * ((q + eps) * ((q + eps) / (m + eps)).log()).sum()
    return math.sqrt(max(0, float(jsd) / math.log(2)))


def run_edge(model, tok, cfg, host, donor, family,
             host_cache, donor_cache):
    """Run all arms for one directed edge: host -> donor.
    host_cache/donor_cache = (states, dist, logits) from pre-cached clean runs."""
    fwd_layer = cfg["forward_layer"]
    late_layer = cfg["late_layer"]
    restore_layers = cfg["intervention_layers"]

    prefix_mask_h, _, _ = find_prefix_mask(tok, host["prompt"], host["query_entity"])

    host_states, host_dist, host_logits = host_cache
    donor_states, donor_dist, donor_logits = donor_cache

    b = jsd_normalized(host_dist, donor_dist)

    host_greedy = tok.decode([int(torch.argmax(host_logits))]).strip()
    donor_greedy = tok.decode([int(torch.argmax(donor_logits))]).strip()

    self_dist, _, _ = run_with_prefix_swap(
        model, tok, host["prompt"], host_states, host_states,
        prefix_mask_h, fwd_layer
    )
    self_d = jsd_normalized(self_dist, host_dist)

    f21_dist, f21_logits, _ = run_with_prefix_swap(
        model, tok, host["prompt"], host_states, donor_states,
        prefix_mask_h, fwd_layer
    )
    f21_d_host = jsd_normalized(f21_dist, host_dist)
    f21_d_donor = jsd_normalized(f21_dist, donor_dist)

    f25_dist, f25_logits, _ = run_with_prefix_swap(
        model, tok, host["prompt"], host_states, donor_states,
        prefix_mask_h, late_layer
    )
    f25_d_host = jsd_normalized(f25_dist, host_dist)
    f25_d_donor = jsd_normalized(f25_dist, donor_dist)

    restore_results = {}
    for rl in restore_layers:
        r_dist, _, _ = run_with_prefix_swap(
            model, tok, host["prompt"], host_states, donor_states,
            prefix_mask_h, fwd_layer, restore_layer=rl,
            restore_states=host_states
        )
        r_d_host = jsd_normalized(r_dist, host_dist)
        r_d_donor = jsd_normalized(r_dist, donor_dist)
        restore_results[rl] = {
            "d_host": round(r_d_host, 6),
            "d_donor": round(r_d_donor, 6),
        }

    c25_dist, _, _ = run_with_prefix_swap(
        model, tok, host["prompt"], host_states, donor_states,
        prefix_mask_h, fwd_layer, full_restore_layer=late_layer,
        full_restore_states=host_states
    )
    c25_d = jsd_normalized(c25_dist, host_dist)

    f21_greedy = tok.decode([int(torch.argmax(f21_logits))]).strip()

    M = f21_d_host / b if b > 0 else 0
    T = (b - f21_d_donor) / b if b > 0 else 0
    H = restore_results[late_layer]["d_host"] / b if b > 0 else 0
    U = (b - restore_results[late_layer]["d_donor"]) / b if b > 0 else 0
    T25 = (b - f25_d_donor) / b if b > 0 else 0
    L = T - T25

    return {
        "host_world": host["world"],
        "donor_world": donor["world"],
        "query": host["query_entity"],
        "family": family["name"],
        "host_expected": host["expected"],
        "donor_expected": donor["expected"],
        "host_greedy": host_greedy,
        "donor_greedy": donor_greedy,
        "f21_greedy": f21_greedy,
        "b": round(b, 6),
        "self_d": round(self_d, 6),
        "c25_d": round(c25_d, 6),
        "f21_d_host": round(f21_d_host, 6),
        "f21_d_donor": round(f21_d_donor, 6),
        "f25_d_host": round(f25_d_host, 6),
        "f25_d_donor": round(f25_d_donor, 6),
        "restore_profile": restore_results,
        "M": round(M, 6),
        "T": round(T, 6),
        "H": round(H, 6),
        "U": round(U, 6),
        "T25": round(T25, 6),
        "L": round(L, 6),
    }


def build_directed_edges(cfg):
    """Build all directed edges: host->donor differs in exactly one entity value."""
    prompts = build_prompts(cfg)
    edges = []
    for fam in cfg["families"]:
        fam_prompts = [p for p in prompts if p["family"] == fam["name"]]
        for host in fam_prompts:
            for donor in fam_prompts:
                if host["query_entity"] != donor["query_entity"]:
                    continue
                if host["world"] == donor["world"]:
                    continue
                h_vals = (host["val_a"], host["val_b"])
                d_vals = (donor["val_a"], donor["val_b"])
                diffs = sum(1 for a, b in zip(h_vals, d_vals) if a != b)
                if diffs == 1:
                    edges.append((host, donor, fam))
    return edges


def cluster_edges(edge_results):
    """Group edges by undirected binary-world edge (12 clusters)."""
    clusters = {}
    for r in edge_results:
        worlds = tuple(sorted([r["host_world"], r["donor_world"]]))
        key = (r["family"], worlds[0], worlds[1])
        if key not in clusters:
            clusters[key] = []
        clusters[key].append(r)
    return clusters


def bootstrap_ci(values, n_resamples=10000, seed=51203):
    """Deterministic percentile bootstrap CI over cluster means."""
    import numpy as np
    rng = np.random.RandomState(seed)
    n = len(values)
    if n == 0:
        return {"mean": 0, "lb": 0, "ub": 0, "n": 0}
    arr = np.array(values)
    boot = np.array([arr[rng.randint(0, n, n)].mean() for _ in range(n_resamples)])
    return {
        "mean": round(float(arr.mean()), 6),
        "lb": round(float(np.percentile(boot, 2.5)), 6),
        "ub": round(float(np.percentile(boot, 97.5)), 6),
        "n": n,
    }


def adjudicate(clusters, cfg):
    """Evaluate all gates against locked thresholds."""
    gates = cfg["gates"]
    kill = cfg["kill_thresholds"]
    min_elig = cfg["min_eligible_edges"]
    floor = cfg.get("material_separation_floor", 0.05)

    eta = 0.0
    for cl in clusters.values():
        for r in cl:
            eta = max(eta, r["self_d"], r["c25_d"])
    elig_threshold = max(floor, 8 * eta)

    eligible = {}
    for key, cl in clusters.items():
        b_mean = sum(r["b"] for r in cl) / len(cl)
        if b_mean >= elig_threshold:
            eligible[key] = cl

    n_eligible = len(eligible)

    control_ok = True
    max_self = max(r["self_d"] for cl in clusters.values() for r in cl)
    max_c25 = max(r["c25_d"] for cl in clusters.values() for r in cl)
    if max_self > cfg["control_tolerance"] or max_c25 > cfg["control_tolerance"]:
        control_ok = False

    if n_eligible < min_elig or not control_ok:
        return {
            "verdict": "INVALID_CARRIER",
            "n_eligible": n_eligible,
            "eta": round(eta, 8),
            "max_self": round(max_self, 8),
            "max_c25": round(max_c25, 8),
            "control_ok": control_ok,
        }

    cluster_M = []
    cluster_T = []
    cluster_H = []
    cluster_U = []
    cluster_L = []
    for key, cl in eligible.items():
        cluster_M.append(sum(r["M"] for r in cl) / len(cl))
        cluster_T.append(sum(r["T"] for r in cl) / len(cl))
        cluster_H.append(sum(r["H"] for r in cl) / len(cl))
        cluster_U.append(sum(r["U"] for r in cl) / len(cl))
        cluster_L.append(sum(r["L"] for r in cl) / len(cl))

    seed = cfg["bootstrap_seed"]
    n_boot = cfg["bootstrap_resamples"]
    M_ci = bootstrap_ci(cluster_M, n_boot, seed)
    T_ci = bootstrap_ci(cluster_T, n_boot, seed)
    H_ci = bootstrap_ci(cluster_H, n_boot, seed)
    U_ci = bootstrap_ci(cluster_U, n_boot, seed)
    L_ci = bootstrap_ci(cluster_L, n_boot, seed)

    if M_ci["ub"] < kill["no_action_ub"] or T_ci["ub"] < kill["no_action_ub"]:
        verdict = "NO_ACTION_INTERFACE"
    elif H_ci["ub"] < kill["reversible_ub"] and U_ci["ub"] < kill["reversible_ub"]:
        verdict = "REVERSIBLE_ACTION"
    elif (M_ci["lb"] >= gates["move_size_lb"] and T_ci["lb"] >= gates["donor_progress_lb"]
          and H_ci["lb"] >= gates["restored_residue_lb"]
          and U_ci["lb"] >= gates["donor_residue_lb"]):
        if L_ci["lb"] >= gates["localization_lb"]:
            verdict = "COMMITMENT_HYSTERESIS_REGISTERED"
        elif L_ci["ub"] < kill["not_localized_ub"]:
            verdict = "HYSTERESIS_NOT_COMMITMENT_LOCALIZED"
        else:
            verdict = "INCONCLUSIVE_ALLOCATION_STOP"
    else:
        verdict = "INCONCLUSIVE_ALLOCATION_STOP"

    return {
        "verdict": verdict,
        "n_eligible": n_eligible,
        "eta": round(eta, 8),
        "max_self": round(max_self, 8),
        "max_c25": round(max_c25, 8),
        "control_ok": control_ok,
        "M": M_ci,
        "T": T_ci,
        "H": H_ci,
        "U": U_ci,
        "L": L_ci,
    }


def produce(cfg, model, tok):
    """Run all directed edges and collect results."""
    prompts = build_prompts(cfg)
    edges = build_directed_edges(cfg)
    print(f"Total directed edges: {len(edges)}")
    print(f"Unique prompts: {len(prompts)}")
    n_fwd = len(prompts) + len(edges) * 9
    print(f"Forward passes: {len(prompts)} clean + {len(edges)}×9 intervention = {n_fwd}")

    print("Pre-caching clean states...")
    t0 = time.time()
    clean_cache = {}
    for pi, p in enumerate(prompts):
        key = p["prompt"]
        if key not in clean_cache:
            states, dist, logits = capture_all_states(model, tok, key)
            clean_cache[key] = (states, dist, logits)
            print(f"  [{pi+1}/{len(prompts)}] {p['family']} {p['world']} "
                  f"q={p['query_entity']} "
                  f"greedy={tok.decode([int(torch.argmax(logits))]).strip()}")
    print(f"Clean states cached: {len(clean_cache)} ({time.time()-t0:.1f}s)")

    all_results = []
    for i, (host, donor, fam) in enumerate(edges):
        print(f"  [{i+1}/{len(edges)}] {fam['name']} "
              f"{host['world']}->{donor['world']} q={host['query_entity']}")
        host_cache = clean_cache[host["prompt"]]
        donor_cache = clean_cache[donor["prompt"]]
        result = run_edge(model, tok, cfg, host, donor, fam,
                          host_cache, donor_cache)
        all_results.append(result)
        elapsed = time.time() - t0
        rate = elapsed / (i + 1)
        eta_s = rate * (len(edges) - i - 1)
        print(f"    b={result['b']:.4f} M={result['M']:.4f} T={result['T']:.4f} "
              f"H={result['H']:.4f} L={result['L']:.4f} "
              f"[{elapsed:.0f}s, ETA {eta_s:.0f}s]")

    total_time = time.time() - t0
    return all_results, total_time


def reduce(edge_results, cfg):
    """Cluster, bootstrap, and adjudicate."""
    clusters = cluster_edges(edge_results)
    print(f"Clusters: {len(clusters)}")
    for key, cl in sorted(clusters.items()):
        b_mean = sum(r["b"] for r in cl) / len(cl)
        M_mean = sum(r["M"] for r in cl) / len(cl)
        H_mean = sum(r["H"] for r in cl) / len(cl)
        print(f"  {key}: n={len(cl)} b={b_mean:.4f} M={M_mean:.4f} H={H_mean:.4f}")

    verdict = adjudicate(clusters, cfg)
    return verdict, clusters


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "produce", "reduce", "all"])
    parser.add_argument("--config", default="experiments/config/commitment_hysteresis_v1.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.command == "preflight":
        tok = AutoTokenizer.from_pretrained(
            cfg["model_id"], revision=cfg["model_revision"], trust_remote_code=True
        )
        result = preflight(cfg, tok)
        out_path = os.path.join(RESULTS_DIR, "preflight.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Preflight: {'PASS' if result['valid'] else 'FAIL'}")
        print(f"  {result['n_prompts']} prompts checked")
        if not result["valid"]:
            for c in result["checks"]:
                if "error" in c:
                    print(f"  ERROR: {c['error']}")
        print(f"Saved to {out_path}")
        return

    model, tok = load_model()

    if args.command in ("produce", "all"):
        pf = preflight(cfg, tok)
        if not pf["valid"]:
            print("PREFLIGHT FAILED — aborting")
            sys.exit(1)
        pf_path = os.path.join(RESULTS_DIR, "preflight.json")
        with open(pf_path, "w", encoding="utf-8") as f:
            json.dump(pf, f, indent=2)

        edge_results, total_time = produce(cfg, model, tok)

        evidence = {
            "experiment": cfg["experiment"],
            "timestamp": datetime.now().isoformat(),
            "model_id": cfg["model_id"],
            "model_revision": cfg["model_revision"],
            "config_hash": sha256_file(args.config)[:16],
            "runner_hash": sha256_file(__file__)[:16],
            "total_time_s": round(total_time, 2),
            "n_edges": len(edge_results),
            "edges": edge_results,
        }

        evidence_path = os.path.join(RESULTS_DIR, "evidence.json")
        with open(evidence_path, "w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2)
        print(f"\nEvidence saved to {evidence_path}")

    if args.command in ("reduce", "all"):
        if args.command == "reduce":
            evidence_path = os.path.join(RESULTS_DIR, "evidence.json")
            with open(evidence_path, "r", encoding="utf-8") as f:
                evidence = json.load(f)
            edge_results = evidence["edges"]

        verdict, clusters = reduce(edge_results, cfg)

        print(f"\n=== VERDICT: {verdict['verdict']} ===")
        for k in ["M", "T", "H", "U", "L"]:
            if k in verdict:
                v = verdict[k]
                print(f"  {k}: mean={v['mean']:.4f} "
                      f"95%CI=[{v['lb']:.4f}, {v['ub']:.4f}] n={v['n']}")
        print(f"  eta={verdict['eta']:.2e} max_self={verdict['max_self']:.2e} "
              f"max_c25={verdict['max_c25']:.2e}")

        verdict_path = os.path.join(RESULTS_DIR, "verdict.json")
        with open(verdict_path, "w", encoding="utf-8") as f:
            json.dump(verdict, f, indent=2)
        print(f"\nVerdict saved to {verdict_path}")


if __name__ == "__main__":
    main()
