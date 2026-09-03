"""PMO-0R: Path-Memory Observability (Revised) — terminal closeout experiment.

Locked specification from Codex R1+R2 (2026-09-02). Tests whether
same-endpoint, last-two-matched histories retain entity-conditioned path
traces under common suffixes on Finch-3B.

Key design corrections from PFC-0:
- Asymmetric endpoints (entities end at DIFFERENT locations)
- 4-bin response law: {kitchen, garden, office, OTHER} — no renormalization
- Suffix injection via saved recurrent states
- Competence staircase
- Entity interaction statistic (iota) alongside commutation defect (kappa)
"""
import copy
import hashlib
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch

LOCS = ["kitchen", "garden", "office"]
ENTS = ["Avery", "Blake"]

# ── Data structures ──────────────────────────────────────────────────

@dataclass
class Root:
    id: str
    a_location: str
    b_location: str
    fold: int
    phrasing: str

@dataclass
class Panel:
    id: str
    actions: dict          # A1/A2/B1/B2 -> (entity, location)
    final_avery: str
    final_blake: str

@dataclass
class Observation:
    key: str
    dist: np.ndarray       # 4-bin: [kitchen, garden, office, other]

# ── Model adapter ────────────────────────────────────────────────────

class FinchAdapter:
    def __init__(self, cfg):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(
            cfg["model_id"], trust_remote_code=True,
            revision=cfg["model_revision"]
        )
        self.mdl = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"], trust_remote_code=True,
            revision=cfg["model_revision"],
            torch_dtype=torch.float32
        )
        self.mdl.eval()
        self.loc_token_ids = {}
        for loc in cfg["locations"]:
            toks = self.tok.encode(f" {loc}", add_special_tokens=False)
            assert len(toks) == 1, f"'{loc}' not a single token: {toks}"
            tid = toks[0]
            assert tid == cfg["location_tokens"][loc], \
                f"Token ID mismatch for '{loc}': {tid} vs {cfg['location_tokens'][loc]}"
            self.loc_token_ids[loc] = tid
        self.call_count = 0
        self.query_template = cfg["query_template"]
        suf_macro = cfg["suffix_macro"]
        suf_ids = self.tok.encode(suf_macro, add_special_tokens=False)
        assert suf_ids == cfg["suffix_token_ids"], \
            f"Suffix token mismatch: {suf_ids} vs {cfg['suffix_token_ids']}"
        for n in [1, 2, 4]:
            repeated = self.tok.encode(suf_macro * n, add_special_tokens=False)
            assert repeated == suf_ids * n, \
                f"Suffix tokenization not linear at n={n}: {len(repeated)} vs {len(suf_ids)*n}"
        self.suffix_token_ids = suf_ids

    def _extract_4bin(self, logits):
        probs = torch.softmax(logits, dim=0).numpy().astype(np.float64)
        bins = np.zeros(4, dtype=np.float64)
        for i, loc in enumerate(LOCS):
            bins[i] = probs[self.loc_token_ids[loc]]
        bins[3] = 1.0 - bins[:3].sum()
        return bins

    def get_dist(self, text, entity):
        query = self.query_template.replace("{entity}", entity)
        full = text + query
        ids = self.tok.encode(full, return_tensors="pt")
        with torch.no_grad():
            out = self.mdl(ids)
        self.call_count += 1
        return self._extract_4bin(out.logits[0, -1, :])

    def get_state_after_prefix(self, text):
        ids = self.tok.encode(text, return_tensors="pt")
        with torch.no_grad():
            out = self.mdl(ids)
        self.call_count += 1
        return copy.deepcopy(out.state)

    def get_dist_from_state(self, state, continuation_text):
        ids = self.tok.encode(continuation_text, return_tensors="pt")
        state_copy = copy.deepcopy(state)
        with torch.no_grad():
            out = self.mdl(ids, state=state_copy)
        self.call_count += 1
        return self._extract_4bin(out.logits[0, -1, :]), copy.deepcopy(out.state)

    def advance_state(self, state, token_ids_list):
        ids = torch.tensor([token_ids_list], dtype=torch.long)
        state_copy = copy.deepcopy(state)
        with torch.no_grad():
            out = self.mdl(ids, state=state_copy)
        self.call_count += 1
        return copy.deepcopy(out.state)

# ── Population construction ──────────────────────────────────────────

def make_roots(cfg):
    fold_map = {}
    for fi, (fname, members) in enumerate(cfg["folds"].items()):
        for m in members:
            fold_map[tuple(m)] = fi
    roots = []
    for a in cfg["locations"]:
        for b in cfg["locations"]:
            fid = fold_map[(a, b)]
            phrasing = f"Avery is in the {a}. Blake is in the {b}."
            roots.append(Root(f"{a}_{b}", a, b, fid, phrasing))
    return roots

def make_panels(cfg):
    panels = []
    for pid, pdef in cfg["panels"].items():
        actions = {}
        for role in ["A1", "A2", "B1", "B2"]:
            ent, loc = pdef[role]
            actions[role] = (ent, loc)
        panels.append(Panel(pid, actions, pdef["final_avery"], pdef["final_blake"]))
    return panels

def action_sentence(entity, location):
    return f" {entity} moved to the {location}."

def build_4action_prefix(root, panel, extension):
    parts = [root.phrasing]
    for role in extension:
        ent, loc = panel.actions[role]
        parts.append(action_sentence(ent, loc))
    return "".join(parts)

def expected_location(panel, entity):
    if entity == "Avery":
        return panel.final_avery
    return panel.final_blake

# ── Competence staircase ─────────────────────────────────────────────

def run_competence(adapter, roots, panels, cfg):
    print("\n=== COMPETENCE STAIRCASE ===", flush=True)
    results = {}

    print("  Rung 1: Direct facts...", flush=True)
    direct_rows = []
    for r in roots:
        for order in ["AB", "BA"]:
            if order == "AB":
                text = f"Avery is in the {r.a_location}. Blake is in the {r.b_location}."
            else:
                text = f"Blake is in the {r.b_location}. Avery is in the {r.a_location}."
            for ent in ENTS:
                dist = adapter.get_dist(text, ent)
                correct = r.a_location if ent == "Avery" else r.b_location
                correct_idx = LOCS.index(correct)
                top1 = np.argmax(dist[:3]) == correct_idx and dist[correct_idx] > dist[3]
                direct_rows.append({
                    "root": r.id, "order": order, "entity": ent,
                    "correct": correct, "top1": top1,
                    "p_correct": float(dist[correct_idx]),
                    "p_other": float(dist[3]),
                    "dist": dist.tolist()
                })
    results["direct"] = _score_rung(direct_rows, "Direct facts")

    if not results["direct"]["pass"]:
        return results

    print("  Rung 2: Two actions...", flush=True)
    two_rows = []
    for r in roots:
        for p in panels:
            for first_pair in [("A1", "B1"), ("B1", "A1")]:
                ent1, loc1 = p.actions[first_pair[0]]
                ent2, loc2 = p.actions[first_pair[1]]
                text = r.phrasing + action_sentence(ent1, loc1) + action_sentence(ent2, loc2)
                for ent in ENTS:
                    dist = adapter.get_dist(text, ent)
                    correct = p.actions["A1"][1] if ent == "Avery" else p.actions["B1"][1]
                    correct_idx = LOCS.index(correct)
                    top1 = np.argmax(dist[:3]) == correct_idx and dist[correct_idx] > dist[3]
                    two_rows.append({
                        "root": r.id, "panel": p.id,
                        "order": f"{first_pair[0]}{first_pair[1]}",
                        "entity": ent, "correct": correct, "top1": top1,
                        "p_correct": float(dist[correct_idx]),
                        "p_other": float(dist[3]),
                        "dist": dist.tolist()
                    })
    results["two_action"] = _score_rung(two_rows, "Two actions")

    if not results["two_action"]["pass"]:
        return results

    print("  Rung 3: Four actions (suffix 0)...", flush=True)
    four_rows = []
    for r in roots:
        for p in panels:
            for ext in cfg["extensions"]:
                text = build_4action_prefix(r, p, ext)
                for ent in ENTS:
                    dist = adapter.get_dist(text, ent)
                    correct = expected_location(p, ent)
                    correct_idx = LOCS.index(correct)
                    top1 = np.argmax(dist[:3]) == correct_idx and dist[correct_idx] > dist[3]
                    four_rows.append({
                        "root": r.id, "panel": p.id,
                        "ext": "_".join(ext), "entity": ent,
                        "correct": correct, "top1": top1,
                        "p_correct": float(dist[correct_idx]),
                        "p_other": float(dist[3]),
                        "dist": dist.tolist()
                    })
    results["four_action_s0"] = _score_rung(four_rows, "Four actions (suffix 0)")

    return results

def _score_rung(rows, label):
    n = len(rows)
    correct = sum(1 for r in rows if r["top1"])
    overall_acc = correct / n if n > 0 else 0

    arm_accs = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in rows:
        key = (r["entity"], r["correct"])
        arm_accs[key]["total"] += 1
        if r["top1"]:
            arm_accs[key]["correct"] += 1

    min_arm_acc = 1.0
    arm_details = {}
    for key, counts in arm_accs.items():
        acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        arm_details[f"{key[0]}_{key[1]}"] = acc
        min_arm_acc = min(min_arm_acc, acc)

    passed = overall_acc >= 0.95 and min_arm_acc >= 0.90
    print(f"    {label}: overall={overall_acc:.4f}, min_arm={min_arm_acc:.4f} "
          f"{'PASS' if passed else 'FAIL'}", flush=True)
    return {
        "n": n, "correct": correct, "overall_accuracy": overall_acc,
        "min_arm_accuracy": min_arm_acc, "arm_details": arm_details,
        "pass": passed
    }

# ── Science observations ─────────────────────────────────────────────

def run_science(adapter, roots, panels, cfg):
    print("\n=== SCIENCE OBSERVATIONS ===", flush=True)
    extensions = cfg["extensions"]
    suffix_lengths = cfg["suffix_lengths"]
    suffix_tids = cfg["suffix_token_ids"]
    query_template = cfg["query_template"]

    states = {}
    observations = {}

    print("  Building 4-action states...", flush=True)
    t0 = time.time()
    for r in roots:
        for p in panels:
            for ext in extensions:
                ext_id = "_".join(ext)
                key_base = f"{r.id}_{p.id}_{ext_id}"
                prefix = build_4action_prefix(r, p, ext)
                state = adapter.get_state_after_prefix(prefix)
                states[key_base] = state

                for ent in ENTS:
                    query = query_template.replace("{entity}", ent)
                    dist, _ = adapter.get_dist_from_state(state, query)
                    obs_key = f"{key_base}_{ent}_s0"
                    observations[obs_key] = dist
    print(f"    {len(states)} states built, {len(observations)} s0 observations "
          f"({time.time()-t0:.1f}s)", flush=True)

    for slen in suffix_lengths:
        if slen == 0:
            continue
        print(f"  Suffix length {slen}...", flush=True)
        t0 = time.time()
        suffix_ids = suffix_tids * slen
        for r in roots:
            for p in panels:
                for ext in extensions:
                    ext_id = "_".join(ext)
                    key_base = f"{r.id}_{p.id}_{ext_id}"
                    base_state = states[key_base]
                    suf_state = adapter.advance_state(base_state, suffix_ids)
                    for ent in ENTS:
                        query = query_template.replace("{entity}", ent)
                        dist, _ = adapter.get_dist_from_state(suf_state, query)
                        obs_key = f"{key_base}_{ent}_s{slen}"
                        observations[obs_key] = dist
        print(f"    {time.time()-t0:.1f}s", flush=True)

    return observations, states

# ── Replay/integrity checks ──────────────────────────────────────────

def run_replay_checks(adapter, roots, panels, cfg, states, observations):
    print("\n=== REPLAY/INTEGRITY CHECKS ===", flush=True)
    n_checks = min(36, len(states))
    rng = np.random.RandomState(cfg["seed"])
    check_keys = rng.choice(list(states.keys()), size=n_checks, replace=False)

    max_tv = 0.0
    max_oneshot_discrepancy = 0.0

    for key_base in check_keys:
        parts = key_base.split("_")
        r_id = f"{parts[0]}_{parts[1]}"
        p_id = parts[2]
        ext = parts[3:]

        root = next(r for r in roots if r.id == r_id)
        panel = next(p for p in panels if p.id == p_id)
        prefix = build_4action_prefix(root, panel, ext)

        fresh_state = adapter.get_state_after_prefix(prefix)
        for ent in ENTS:
            query = cfg["query_template"].replace("{entity}", ent)
            obs_key = f"{key_base}_{ent}_s0"
            cached = observations[obs_key]

            fresh_dist, _ = adapter.get_dist_from_state(fresh_state, query)
            tv = 0.5 * np.abs(cached - fresh_dist).sum()
            max_tv = max(max_tv, tv)

            oneshot_dist = adapter.get_dist(prefix, ent)
            discrepancy = 0.5 * np.abs(cached - oneshot_dist).sum()
            max_oneshot_discrepancy = max(max_oneshot_discrepancy, discrepancy)

    print(f"  Replay max TV: {max_tv:.6e}", flush=True)
    print(f"  One-shot discrepancy: {max_oneshot_discrepancy:.6e}", flush=True)
    return {
        "n_checks": n_checks,
        "replay_max_tv": float(max_tv),
        "oneshot_max_discrepancy": float(max_oneshot_discrepancy),
        "replay_pass": max_tv <= cfg["gates"]["replay_max_tv"],
        "oneshot_pass": max_oneshot_discrepancy <= cfg["gates"]["oneshot_state_discrepancy"]
    }

# ── Observables: kappa and iota ──────────────────────────────────────

def tv(a, b):
    return 0.5 * np.abs(a - b).sum()

def compute_kappa_iota(observations, roots, panels, cfg):
    print("\n=== COMPUTING KAPPA AND IOTA ===", flush=True)
    matched_pairs = cfg["matched_pairs"]
    suffix_lengths = cfg["suffix_lengths"]
    results = {}

    for slen in suffix_lengths:
        kappas = []
        iotas = []
        kappas_by_root = defaultdict(list)
        kappas_by_location = defaultdict(list)

        for r in roots:
            for p in panels:
                for src_ext, tgt_ext in matched_pairs:
                    src_id = "_".join(src_ext)
                    tgt_id = "_".join(tgt_ext)

                    tvs_per_entity = {}
                    deltas = {}
                    for ent in ENTS:
                        src_key = f"{r.id}_{p.id}_{src_id}_{ent}_s{slen}"
                        tgt_key = f"{r.id}_{p.id}_{tgt_id}_{ent}_s{slen}"
                        if src_key not in observations or tgt_key not in observations:
                            continue
                        d_src = observations[src_key]
                        d_tgt = observations[tgt_key]
                        tvs_per_entity[ent] = tv(d_src, d_tgt)
                        deltas[ent] = d_src - d_tgt

                    if len(tvs_per_entity) == 2:
                        kappa = sum(tvs_per_entity.values()) / 2
                        kappas.append(kappa)
                        kappas_by_root[r.id].append(kappa)
                        for ent in ENTS:
                            loc = expected_location(p, ent)
                            kappas_by_location[loc].append(tvs_per_entity[ent])

                        delta_diff = deltas["Avery"] - deltas["Blake"]
                        iota = 0.25 * np.abs(delta_diff).sum()
                        iotas.append(iota)

        if not kappas:
            results[f"s{slen}"] = {"n": 0, "kappa_mean": 0, "iota_mean": 0}
            continue

        kappas = np.array(kappas)
        iotas = np.array(iotas)

        k_ci = _root_cluster_ci(kappas_by_root, roots, cfg)
        i_by_root = defaultdict(list)
        idx = 0
        for r in roots:
            for p in panels:
                for _ in matched_pairs:
                    if idx < len(iotas):
                        i_by_root[r.id].append(iotas[idx])
                        idx += 1
        i_ci = _root_cluster_ci(i_by_root, roots, cfg)

        loc_pass_count = 0
        for loc in LOCS:
            if loc in kappas_by_location:
                loc_mean = np.mean(kappas_by_location[loc])
                if loc_mean >= cfg["gates"]["registered_witness_kappa"]:
                    loc_pass_count += 1

        results[f"s{slen}"] = {
            "n": len(kappas),
            "kappa_mean": float(kappas.mean()),
            "kappa_std": float(kappas.std()),
            "kappa_ci95": k_ci,
            "iota_mean": float(iotas.mean()),
            "iota_std": float(iotas.std()),
            "iota_ci95": i_ci,
            "location_replication": loc_pass_count,
            "kappas_by_location": {
                loc: float(np.mean(vs)) for loc, vs in kappas_by_location.items()
            }
        }
        print(f"  s{slen}: kappa={kappas.mean():.6f} CI={k_ci}, "
              f"iota={iotas.mean():.6f} CI={i_ci}, loc_rep={loc_pass_count}",
              flush=True)

    agg_kappas = []
    agg_iotas = []
    agg_by_root = defaultdict(list)
    agg_i_by_root = defaultdict(list)
    for slen in suffix_lengths:
        for r in roots:
            for p in panels:
                for src_ext, tgt_ext in matched_pairs:
                    src_id = "_".join(src_ext)
                    tgt_id = "_".join(tgt_ext)
                    tvs_per_entity = {}
                    deltas = {}
                    for ent in ENTS:
                        src_key = f"{r.id}_{p.id}_{src_id}_{ent}_s{slen}"
                        tgt_key = f"{r.id}_{p.id}_{tgt_id}_{ent}_s{slen}"
                        if src_key in observations and tgt_key in observations:
                            d_src = observations[src_key]
                            d_tgt = observations[tgt_key]
                            tvs_per_entity[ent] = tv(d_src, d_tgt)
                            deltas[ent] = d_src - d_tgt
                    if len(tvs_per_entity) == 2:
                        kappa = sum(tvs_per_entity.values()) / 2
                        agg_kappas.append(kappa)
                        agg_by_root[r.id].append(kappa)
                        delta_diff = deltas["Avery"] - deltas["Blake"]
                        iota = 0.25 * np.abs(delta_diff).sum()
                        agg_iotas.append(iota)
                        agg_i_by_root[r.id].append(iota)

    agg_kappas = np.array(agg_kappas)
    agg_iotas = np.array(agg_iotas)
    results["aggregate"] = {
        "n": len(agg_kappas),
        "kappa_mean": float(agg_kappas.mean()) if len(agg_kappas) else 0,
        "kappa_ci95": _root_cluster_ci(agg_by_root, roots, cfg),
        "iota_mean": float(agg_iotas.mean()) if len(agg_iotas) else 0,
        "iota_ci95": _root_cluster_ci(agg_i_by_root, roots, cfg),
    }
    print(f"  Aggregate: kappa={agg_kappas.mean():.6f}, iota={agg_iotas.mean():.6f}",
          flush=True)

    return results

def _root_cluster_ci(by_root, roots, cfg):
    root_ids = sorted(set(r.id for r in roots))
    per_root = []
    for rid in root_ids:
        if rid in by_root and len(by_root[rid]) > 0:
            per_root.append(np.mean(by_root[rid]))
    if len(per_root) < 3:
        return [0.0, 0.0]
    per_root = np.array(per_root)
    rng = np.random.RandomState(cfg["bootstrap_seed"])
    boots = np.array([
        np.mean(rng.choice(per_root, len(per_root), True))
        for _ in range(cfg["bootstrap_resamples"])
    ])
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

# ── Null ladder ──────────────────────────────────────────────────────

def run_null_ladder(observations, roots, panels, cfg):
    print("\n=== NULL LADDER ===", flush=True)
    matched_pairs = cfg["matched_pairs"]
    suffix_lengths = cfg["suffix_lengths"]
    extensions = cfg["extensions"]
    folds = cfg["folds"]

    results = {}

    for fold_id in range(3):
        train_roots = [r for r in roots if r.fold != fold_id]
        test_roots = [r for r in roots if r.fold == fold_id]

        for slen in suffix_lengths:
            train_data = _collect_matched_data(
                observations, train_roots, panels, matched_pairs, slen)
            test_data = _collect_matched_data(
                observations, test_roots, panels, matched_pairs, slen)

            train_contrasts = defaultdict(list)
            for item in train_data:
                pid = item["panel"]
                contrast = item["tgt_dist"] - item["src_dist"]
                train_contrasts[pid].append(contrast)

            panel_means = {}
            for pid, contrasts in train_contrasts.items():
                panel_means[pid] = np.mean(contrasts, axis=0)

            discounted_contrasts = _fit_discounted(train_data, cfg)
            additive_contrasts = _fit_panel_additive(train_data)

            for item in test_data:
                src = item["src_dist"]
                tgt = item["tgt_dist"]
                key = (fold_id, slen, item["root"], item["panel"],
                       item["pair_idx"], item["entity"])

                results.setdefault("identity", []).append(
                    {"key": key, "tv": tv(src, tgt)})

                panel_pred = _clip_dist(src + panel_means.get(item["panel"], np.zeros(4)))
                results.setdefault("panel_centroid", []).append(
                    {"key": key, "tv": tv(panel_pred, tgt)})

                ext_src = item["src_ext"]
                _add_feature_null(results, "last_1", train_data, test_data,
                                  item, key, lambda x: x["src_ext"][-1])
                _add_feature_null(results, "last_2", train_data, test_data,
                                  item, key, lambda x: tuple(x["src_ext"][-2:]))

                disc_key = (item["panel"], tuple(ext_src))
                if disc_key in discounted_contrasts:
                    disc_pred = _clip_dist(src + discounted_contrasts[disc_key])
                    results.setdefault("discounted", []).append(
                        {"key": key, "tv": tv(disc_pred, tgt)})

                add_key = (item["panel"], tuple(ext_src))
                if add_key in additive_contrasts:
                    add_pred = _clip_dist(src + additive_contrasts[add_key])
                    results.setdefault("panel_additive", []).append(
                        {"key": key, "tv": tv(add_pred, tgt)})

                ext_key = tuple(ext_src)
                _add_lookup_null(results, "path_id", train_data, item, key, ext_key)

                rng = np.random.RandomState(42)
                shuffled_ext = list(ext_src)
                rng.shuffle(shuffled_ext)
                _add_lookup_null(results, "shuffled", train_data, item, key,
                                 tuple(shuffled_ext))

                geo = _geometric_mean(src, tgt)
                results.setdefault("geometric_mean", []).append(
                    {"key": key, "tv": tv(geo, tgt)})

    null_summary = {}
    for method, items in results.items():
        tvs = np.array([it["tv"] for it in items])
        null_summary[method] = {
            "mean_tv": float(tvs.mean()),
            "std_tv": float(tvs.std()),
            "n": len(tvs)
        }
        print(f"  {method}: mean_tv={tvs.mean():.6f} (n={len(tvs)})", flush=True)

    return null_summary

def _collect_matched_data(observations, roots, panels, matched_pairs, slen):
    data = []
    for r in roots:
        for p in panels:
            for pair_idx, (src_ext, tgt_ext) in enumerate(matched_pairs):
                src_id = "_".join(src_ext)
                tgt_id = "_".join(tgt_ext)
                for ent in ENTS:
                    src_key = f"{r.id}_{p.id}_{src_id}_{ent}_s{slen}"
                    tgt_key = f"{r.id}_{p.id}_{tgt_id}_{ent}_s{slen}"
                    if src_key in observations and tgt_key in observations:
                        data.append({
                            "root": r.id, "panel": p.id, "pair_idx": pair_idx,
                            "entity": ent, "src_ext": list(src_ext),
                            "tgt_ext": list(tgt_ext),
                            "src_dist": observations[src_key],
                            "tgt_dist": observations[tgt_key],
                        })
    return data

def _clip_dist(d):
    d = np.clip(d, 0, None)
    s = d.sum()
    if s > 0:
        d = d / s
    return d

def _add_feature_null(results, name, train_data, test_data, item, key, feat_fn):
    feat = feat_fn(item)
    train_contrasts = [
        td["tgt_dist"] - td["src_dist"]
        for td in train_data if feat_fn(td) == feat
    ]
    if train_contrasts:
        mean_contrast = np.mean(train_contrasts, axis=0)
        pred = _clip_dist(item["src_dist"] + mean_contrast)
        results.setdefault(name, []).append({"key": key, "tv": tv(pred, item["tgt_dist"])})

def _add_lookup_null(results, name, train_data, item, key, lookup_key):
    matches = [td for td in train_data
               if tuple(td["src_ext"]) == lookup_key and td["panel"] == item["panel"]]
    if matches:
        mean_contrast = np.mean([td["tgt_dist"] - td["src_dist"] for td in matches], axis=0)
        pred = _clip_dist(item["src_dist"] + mean_contrast)
        results.setdefault(name, []).append({"key": key, "tv": tv(pred, item["tgt_dist"])})

def _fit_discounted(train_data, cfg):
    lambdas = cfg.get("discounted_lambdas", [0.0, 0.25, 0.5, 0.75, 0.9, 0.97, 1.0])
    ROLES = {"A1": 0, "A2": 1, "B1": 2, "B2": 3}
    best_lambda = 0.5
    best_err = float("inf")
    for lam in lambdas:
        err = 0.0
        for item in train_data:
            weights = np.array([lam ** (3 - i) for i in range(4)])
            feat_key = (item["panel"], tuple(item["src_ext"]))
            same_feat = [td for td in train_data
                         if (td["panel"], tuple(td["src_ext"])) == feat_key
                         and td is not item]
            if same_feat:
                mean_c = np.mean([td["tgt_dist"] - td["src_dist"] for td in same_feat], axis=0)
                pred = _clip_dist(item["src_dist"] + mean_c * weights.mean())
                err += tv(pred, item["tgt_dist"])
        if err < best_err:
            best_err = err
            best_lambda = lam

    out = {}
    by_key = defaultdict(list)
    for item in train_data:
        k = (item["panel"], tuple(item["src_ext"]))
        by_key[k].append(item["tgt_dist"] - item["src_dist"])
    for k, contrasts in by_key.items():
        weights = np.array([best_lambda ** (3 - i) for i in range(4)])
        out[k] = np.mean(contrasts, axis=0) * weights.mean()
    return out

def _fit_panel_additive(train_data):
    ROLES = ["A1", "A2", "B1", "B2"]
    by_panel = defaultdict(list)
    for item in train_data:
        by_panel[item["panel"]].append(item)
    out = {}
    for pid, items in by_panel.items():
        pos_effects = defaultdict(list)
        for item in items:
            contrast = item["tgt_dist"] - item["src_dist"]
            for pos_i, role in enumerate(item["src_ext"]):
                pos_effects[(pos_i, role)].append(contrast)
        mean_effects = {}
        for k, vs in pos_effects.items():
            mean_effects[k] = np.mean(vs, axis=0)
        for item in items:
            total = np.zeros(4)
            for pos_i, role in enumerate(item["src_ext"]):
                if (pos_i, role) in mean_effects:
                    total += mean_effects[(pos_i, role)]
            total /= max(1, len(item["src_ext"]))
            out[(pid, tuple(item["src_ext"]))] = total
    return out

def _geometric_mean(dist_a, dist_b):
    eps = 1e-10
    geo = np.sqrt((dist_a + eps) * (dist_b + eps))
    return geo / geo.sum()

# ── Suffix competence ────────────────────────────────────────────────

def run_suffix_competence(observations, roots, panels, cfg, slen):
    rows = []
    extensions = cfg["extensions"]
    for r in roots:
        for p in panels:
            for ext in extensions:
                ext_id = "_".join(ext)
                for ent in ENTS:
                    obs_key = f"{r.id}_{p.id}_{ext_id}_{ent}_s{slen}"
                    if obs_key not in observations:
                        continue
                    dist = observations[obs_key]
                    correct = expected_location(p, ent)
                    correct_idx = LOCS.index(correct)
                    top1 = np.argmax(dist[:3]) == correct_idx and dist[correct_idx] > dist[3]
                    rows.append({
                        "root": r.id, "panel": p.id, "ext": ext_id,
                        "entity": ent, "correct": correct, "top1": top1,
                        "p_correct": float(dist[correct_idx]),
                        "p_other": float(dist[3]),
                    })
    return _score_rung(rows, f"Suffix {slen}")

# ── Adjudication ─────────────────────────────────────────────────────

def adjudicate(competence, replay, kappa_iota, null_ladder, cfg):
    gates = cfg["gates"]

    if not replay["replay_pass"] or not replay["oneshot_pass"]:
        return "INVALID_IMPLEMENTATION", "Replay or one-shot integrity check failed"

    for rung_name in ["direct", "two_action", "four_action_s0"]:
        if rung_name in competence and not competence[rung_name]["pass"]:
            return "TASK_POPULATION_VOID", f"Competence failed at {rung_name}"

    agg = kappa_iota.get("aggregate", {})
    kappa_mean = agg.get("kappa_mean", 0)
    kappa_ci = agg.get("kappa_ci95", [0, 0])
    iota_mean = agg.get("iota_mean", 0)
    iota_ci = agg.get("iota_ci95", [0, 0])

    kappa_lb = kappa_ci[0] if kappa_ci else 0
    iota_lb = iota_ci[0] if iota_ci else 0

    s0 = kappa_iota.get("s0", {})
    s0_kappa = s0.get("kappa_mean", 0)
    s0_ci = s0.get("kappa_ci95", [0, 0])
    s0_lb = s0_ci[0] if s0_ci else 0
    s0_loc_rep = s0.get("location_replication", 0)

    if s0_kappa < gates["registered_witness_kappa"] or \
       s0_lb <= gates["registered_witness_lb"] or \
       s0_loc_rep < gates["location_replication_min"]:
        return ("NO_REGISTERED_PATH_WITNESS",
                f"Suffix-0 kappa={s0_kappa:.6f} lb={s0_lb:.6f} loc_rep={s0_loc_rep}")

    if iota_mean < gates["entity_interaction_iota"] or iota_lb <= gates["entity_interaction_lb"]:
        return ("GLOBAL_TRACE_ONLY",
                f"kappa passes but iota={iota_mean:.6f} lb={iota_lb:.6f} insufficient")

    path_id_tv = null_ladder.get("path_id", {}).get("mean_tv", 1.0)
    for null_name in ["identity", "panel_centroid", "last_1", "last_2", "shuffled",
                       "geometric_mean"]:
        null_tv = null_ladder.get(null_name, {}).get("mean_tv", 1.0)
        advantage = null_tv - path_id_tv
        if advantage < gates["null_advantage"]:
            return ("LOW_ORDER_PRESENTATION_SUFFICIENT",
                    f"path_id advantage over {null_name}: {advantage:.6f} < {gates['null_advantage']}")

    witness_band = "strong" if kappa_mean >= gates["strong_witness_kappa"] and \
                               kappa_lb > gates["strong_witness_lb"] else "registered"

    return ("BOUNDED_TRACE_MEMORY_WITNESS",
            f"All gates pass. Band: {witness_band}. "
            f"kappa={kappa_mean:.6f}, iota={iota_mean:.6f}")

# ── Main ─────────────────────────────────────────────────────────────

def main():
    config_path = "experiments/config/pmo_0r.json"
    with open(config_path, "rb") as f:
        config_bytes = f.read()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    cfg = json.loads(config_bytes)

    result_dir = Path(cfg["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"PMO-0R Runner", flush=True)
    print(f"Config hash: {config_hash}", flush=True)
    print(f"CPU abort: {cfg['cpu_abort_minutes']} min", flush=True)
    t_start = time.time()

    adapter = FinchAdapter(cfg)
    print(f"Model loaded. Location tokens: {adapter.loc_token_ids}", flush=True)

    roots = make_roots(cfg)
    panels = make_panels(cfg)
    print(f"Population: {len(roots)} roots, {len(panels)} panels, "
          f"{len(cfg['extensions'])} extensions", flush=True)

    competence = run_competence(adapter, roots, panels, cfg)

    competence_pass = all(
        competence.get(k, {}).get("pass", False)
        for k in ["direct", "two_action", "four_action_s0"]
    )
    if not competence_pass:
        verdict, detail = "TASK_POPULATION_VOID", "Competence staircase failed"
        _write_result(result_dir, cfg, config_hash, t_start, adapter,
                      competence, {}, {}, {}, verdict, detail)
        return

    observations, states = run_science(adapter, roots, panels, cfg)

    for slen in cfg["suffix_lengths"]:
        if slen == 0:
            continue
        suf_comp = run_suffix_competence(observations, roots, panels, cfg, slen)
        competence[f"suffix_{slen}"] = suf_comp
        if not suf_comp["pass"]:
            print(f"  Competence failed at suffix {slen}, stopping staircase", flush=True)
            break

    elapsed = time.time() - t_start
    if elapsed / 60 > cfg["cpu_abort_minutes"]:
        verdict = "INVALID_IMPLEMENTATION"
        detail = f"CPU time {elapsed/60:.1f} min exceeds {cfg['cpu_abort_minutes']} min"
        _write_result(result_dir, cfg, config_hash, t_start, adapter,
                      competence, observations, {}, {}, verdict, detail)
        return

    replay = run_replay_checks(adapter, roots, panels, cfg, states, observations)

    kappa_iota = compute_kappa_iota(observations, roots, panels, cfg)

    null_ladder = run_null_ladder(observations, roots, panels, cfg)

    verdict, detail = adjudicate(competence, replay, kappa_iota, null_ladder, cfg)

    _write_result(result_dir, cfg, config_hash, t_start, adapter,
                  competence, observations, kappa_iota, null_ladder,
                  verdict, detail, replay)

def _write_result(result_dir, cfg, config_hash, t_start, adapter,
                  competence, observations, kappa_iota, null_ladder,
                  verdict, detail, replay=None):
    elapsed = time.time() - t_start
    result = {
        "manifest": {
            "config_hash": config_hash,
            "model_id": cfg["model_id"],
            "model_revision": cfg["model_revision"],
            "total_calls": adapter.call_count,
            "elapsed_seconds": elapsed,
        },
        "competence": competence,
        "replay": replay or {},
        "kappa_iota": kappa_iota,
        "null_ladder": null_ladder,
        "verdict": verdict,
        "detail": detail,
        "measurement_to_artifact_ratio": "1:1",
    }

    result_path = result_dir / "result.json"

    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.bool_):
                return bool(o)
            return super().default(o)

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, cls=_Enc)
    print(f"\n{'='*60}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"Detail: {detail}", flush=True)
    print(f"Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"Model calls: {adapter.call_count}", flush=True)
    print(f"Result: {result_path}", flush=True)

    if observations:
        obs_path = result_dir / "obs_cache.npz"
        np.savez_compressed(obs_path, **{k: v for k, v in observations.items()})
        print(f"Observations cached: {obs_path}", flush=True)

if __name__ == "__main__":
    main()
