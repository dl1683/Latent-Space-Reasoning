"""RCQ-0: Real Causal Quotient — first experiment.

Discovers the response quotient and action law of Finch-3B's recurrent state
on a two-entity/three-location state-tracking task. Uses only behavioral
equivalence — no R^n geometry.

Target: 250-350 lines. Hard halt at 450.
"""
import json
import os
import sys
import time
import copy
import hashlib
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CONFIG_PATH = Path(__file__).parent / "config" / "rcq0_v1.json"

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

def deep_clone_state(state):
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.clone()
    if isinstance(state, (list, tuple)):
        return type(state)(deep_clone_state(s) for s in state)
    return state

class RCQModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"], trust_remote_code=True, dtype=torch.float32
        )
        self.model.eval()
        self.answer_token_ids = {
            loc: self.tokenizer.encode(f" {loc}", add_special_tokens=False)[0]
            for loc in cfg["locations"]
        }

    def get_distribution(self, text=None, input_ids=None, state=None):
        if input_ids is None:
            input_ids = self.tokenizer.encode(text, return_tensors="pt")
        with torch.no_grad():
            kw = {"state": state} if state is not None else {}
            out = self.model(input_ids, **kw)
        logits = out.logits[0, -1, :]
        vals = torch.tensor([logits[self.answer_token_ids[l]].item() for l in self.cfg["locations"]])
        probs = torch.softmax(vals, dim=0).numpy()
        return dict(zip(self.cfg["locations"], probs.tolist())), out.state

    def capture_state(self, text):
        ids = self.tokenizer.encode(text, return_tensors="pt")
        with torch.no_grad():
            out = self.model(ids)
        return deep_clone_state(out.state)


def tv(d1, d2, locs):
    return 0.5 * sum(abs(d1[l] - d2[l]) for l in locs)


def quantize_dist(d, locs, delta):
    return tuple(round(d[l] / delta) * delta for l in locs)


def make_histories(cfg):
    ents = cfg["entities"]
    locs = cfg["locations"]
    n = cfg["histories_per_state"]
    histories = {}
    for a_loc in locs:
        for b_loc in locs:
            state = (a_loc, b_loc)
            base = f"{ents[0]} is in the {a_loc}. {ents[1]} is in the {b_loc}."
            histories[state] = [base] * n
    return histories


def make_actions(cfg):
    return [f"{e} moved to the {l}." for e in cfg["entities"] for l in cfg["locations"]]


def make_queries(cfg):
    return [f"\nQuestion: Where is {e}?\nAnswer: The" for e in cfg["entities"]]


def run_phase1(model, histories, queries, cfg):
    """Capability screen + state extraction."""
    print("Phase 1: Capability screen + state extraction")
    locs = cfg["locations"]
    n_con = cfg["construction_split"]
    data = {}
    for state, paths in histories.items():
        data[state] = []
        for pi, prefix in enumerate(paths):
            row = {"prefix": prefix, "split": "con" if pi < n_con else "held"}
            row["state_obj"] = model.capture_state(prefix)
            for qi, query in enumerate(queries):
                dist, _ = model.get_distribution(text=prefix + query)
                row[f"q{qi}"] = dist
            data[state].append(row)
    # Entity discrimination check
    asym_tvs = []
    for state, rows in data.items():
        if state[0] != state[1]:
            for r in rows:
                asym_tvs.append(tv(r["q0"], r["q1"], locs))
    mean_disc = float(np.mean(asym_tvs))
    print(f"  Entity discrimination: mean asym TV = {mean_disc:.4f} "
          f"(gate: >{cfg['gates']['entity_discrimination_tv']})")
    return data, mean_disc


def response_vector(row, n_q, locs):
    """Extract the flat response vector from a data row."""
    return np.array([row[f"q{qi}"][l] for qi in range(n_q) for l in locs])


def response_tv(v1, v2, n_q, n_locs):
    """Mean TV across all queries between two response vectors."""
    total = 0.0
    for qi in range(n_q):
        s = qi * n_locs
        total += 0.5 * np.sum(np.abs(v1[s:s+n_locs] - v2[s:s+n_locs]))
    return total / n_q


def run_phase2(data, cfg):
    """Quotient construction: cluster or assign ground-truth classes."""
    mode = cfg.get("quotient_mode", "cluster")
    print(f"Phase 2: Quotient construction (mode={mode})")
    locs = cfg["locations"]
    n_q = len(cfg["entities"])
    n_locs = len(locs)

    keys = []
    vecs = []
    for state, rows in data.items():
        for ri, r in enumerate(rows):
            keys.append((state, ri))
            vecs.append(response_vector(r, n_q, locs))
    vecs = np.array(vecs)
    n = len(keys)

    if mode == "ground_truth":
        state_to_class = {}
        next_class = 0
        labels = []
        for state, ri in keys:
            if state not in state_to_class:
                state_to_class[state] = next_class
                next_class += 1
            labels.append(state_to_class[state])
    else:
        threshold = cfg["quantization_delta"]
        labels = [-1] * n
        next_class = 0
        for i in range(n):
            if labels[i] >= 0:
                continue
            labels[i] = next_class
            for j in range(i + 1, n):
                if labels[j] >= 0:
                    continue
                d = response_tv(vecs[i], vecs[j], n_q, n_locs)
                if d <= threshold:
                    labels[j] = next_class
            next_class += 1

    class_map = {}
    classes = defaultdict(list)
    centroids = {}
    for idx, key in enumerate(keys):
        ci = labels[idx]
        class_map[key] = ci
        classes[ci].append(key)

    for ci, members in classes.items():
        centroids[ci] = np.mean([vecs[keys.index(m)] for m in members], axis=0)

    n_classes = len(classes)
    sizes = sorted([len(v) for v in classes.values()], reverse=True)
    print(f"  Found {n_classes} equivalence classes from {n} states")
    print(f"  Class sizes: {sizes}")

    # Within-class TV diagnostic
    within_tvs = []
    for ci, members in classes.items():
        if len(members) < 2:
            continue
        mvecs = [vecs[keys.index(m)] for m in members]
        for i in range(len(mvecs)):
            for j in range(i+1, len(mvecs)):
                within_tvs.append(response_tv(mvecs[i], mvecs[j], n_q, n_locs))
    if within_tvs:
        print(f"  Within-class TV: mean={np.mean(within_tvs):.4f} max={np.max(within_tvs):.4f}")

    return class_map, dict(classes), centroids


def run_phase3(model, data, class_map, centroids, actions, queries, cfg):
    """Action law: learn transitions from construction data."""
    print("Phase 3: Action law (transitions from construction data)")
    locs = cfg["locations"]
    n_q = len(cfg["entities"])
    n_locs = len(locs)
    transition_vecs = defaultdict(list)

    for state, rows in data.items():
        for ri, r in enumerate(rows):
            if r["split"] != "con":
                continue
            src_class = class_map.get((state, ri))
            if src_class is None:
                continue
            for ai, action in enumerate(actions):
                action_text = " " + action
                for qi, query in enumerate(queries):
                    full = r["prefix"] + action_text + query
                    dist, _ = model.get_distribution(text=full)
                    r[f"a{ai}_q{qi}"] = dist
            for ai in range(len(actions)):
                vec = np.array([r[f"a{ai}_q{qi}"][l] for qi in range(n_q) for l in locs])
                transition_vecs[(src_class, ai)].append(vec)

    # Build transition table: mean vector -> nearest class
    transition_table = {}
    consistency_scores = []
    for (src, ai), vecs_list in transition_vecs.items():
        mean_vec = np.mean(vecs_list, axis=0)
        # Find nearest existing class by TV
        best_ci, best_d = None, float('inf')
        for ci, centroid in centroids.items():
            d = response_tv(mean_vec, centroid, n_q, n_locs)
            if d < best_d:
                best_d = d
                best_ci = ci
        transition_table[(src, ai)] = best_ci
        # Consistency: do all members of this class transition to the same target?
        target_classes = []
        for v in vecs_list:
            bc, bd = None, float('inf')
            for ci, centroid in centroids.items():
                d = response_tv(v, centroid, n_q, n_locs)
                if d < bd:
                    bd = d
                    bc = ci
            target_classes.append(bc)
        if target_classes:
            from collections import Counter
            counts = Counter(target_classes)
            consistency = counts.most_common(1)[0][1] / len(target_classes)
            consistency_scores.append(consistency)

    mean_cons = float(np.mean(consistency_scores)) if consistency_scores else 0.0
    print(f"  Transition table: {len(transition_table)} entries")
    print(f"  Mean within-class consistency: {mean_cons:.4f} "
          f"(gate: >={cfg['gates']['within_class_transition_consistency']})")
    return transition_table, mean_cons


def run_phase4(model, data, class_map, centroids, transition_table, actions, queries, cfg):
    """Held-out composition: predict two-action outcomes using centroid-based matching."""
    print("Phase 4: Held-out composition test")
    locs = cfg["locations"]
    n_q = len(cfg["entities"])
    n_locs = len(locs)
    rng = np.random.RandomState(cfg["seed"])

    top1_hits = 0
    tv_scores = []
    n_tested = 0

    use_con = cfg["held_out_split"] == 0
    for state, rows in data.items():
        for ri, r in enumerate(rows):
            if not use_con and r["split"] != "held":
                continue
            src_class = class_map.get((state, ri))
            if src_class is None:
                continue
            n_actions = len(actions)
            pairs = [(a1, a2) for a1 in range(n_actions) for a2 in range(n_actions)]
            rng.shuffle(pairs)
            for a1, a2 in pairs[:6]:
                mid_class = transition_table.get((src_class, a1))
                if mid_class is None:
                    continue
                pred_class = transition_table.get((mid_class, a2))
                if pred_class is None:
                    continue
                pred_centroid = centroids[pred_class]
                text = r["prefix"] + " " + actions[a1] + " " + actions[a2]
                actual_vec = []
                actual_dists = {}
                for qi, query in enumerate(queries):
                    dist, _ = model.get_distribution(text=text + query)
                    actual_dists[qi] = dist
                    actual_vec.extend([dist[l] for l in locs])
                actual_vec = np.array(actual_vec)
                # Check top-1: predicted class == nearest class of actual
                best_ci, best_d = None, float('inf')
                for ci, centroid in centroids.items():
                    d = response_tv(actual_vec, centroid, n_q, n_locs)
                    if d < best_d:
                        best_d = d
                        best_ci = ci
                if best_ci == pred_class:
                    top1_hits += 1
                comp_tv = response_tv(actual_vec, pred_centroid, n_q, n_locs)
                tv_scores.append(comp_tv)
                n_tested += 1

    top1_rate = top1_hits / max(n_tested, 1)
    mean_tv = float(np.mean(tv_scores)) if tv_scores else 1.0
    print(f"  Tested {n_tested} two-action compositions")
    print(f"  Top-1 class agreement: {top1_rate:.4f} (gate: >={cfg['gates']['composition_top1_agreement']})")
    print(f"  Mean response TV: {mean_tv:.4f} (gate: <={cfg['gates']['composition_tv_max']})")
    return top1_rate, mean_tv, n_tested


def run_phase5(model, data, class_map, classes, queries, cfg):
    """State substitution: inject same-class states, compare continuations."""
    print("Phase 5: State substitution (causal test)")
    locs = cfg["locations"]
    same_tvs = []
    cross_tvs = []

    # Group rows by class
    class_rows = defaultdict(list)
    for state, rows in data.items():
        for ri, r in enumerate(rows):
            c = class_map.get((state, ri))
            if c is not None:
                class_rows[c].append((state, ri, r))

    # Same-class substitution
    for ci, members in class_rows.items():
        if len(members) < 2:
            continue
        for i in range(min(len(members), 3)):
            for j in range(i + 1, min(len(members), 3)):
                _, _, r1 = members[i]
                _, _, r2 = members[j]
                s1 = deep_clone_state(r1["state_obj"])
                for qi, query in enumerate(queries):
                    nat_dist = r2[f"q{qi}"]
                    sub_dist, _ = model.get_distribution(text=query, state=deep_clone_state(s1))
                    same_tvs.append(tv(nat_dist, sub_dist, locs))

    # Cross-class substitution (should differ)
    class_ids = list(class_rows.keys())
    for i in range(min(len(class_ids), 5)):
        for j in range(i + 1, min(len(class_ids), 5)):
            r_src = class_rows[class_ids[i]][0][2]
            r_tgt = class_rows[class_ids[j]][0][2]
            s_src = deep_clone_state(r_src["state_obj"])
            for qi, query in enumerate(queries):
                nat_dist = r_tgt[f"q{qi}"]
                sub_dist, _ = model.get_distribution(text=query, state=deep_clone_state(s_src))
                cross_tvs.append(tv(nat_dist, sub_dist, locs))

    mean_same = float(np.mean(same_tvs)) if same_tvs else 1.0
    mean_cross = float(np.mean(cross_tvs)) if cross_tvs else 0.0
    print(f"  Same-class substitution: mean TV = {mean_same:.4f} (gate: <={cfg['gates']['substitution_same_class_tv_max']})")
    print(f"  Cross-class substitution: mean TV = {mean_cross:.4f} (gate: >={cfg['gates']['substitution_cross_class_tv_min']})")
    return mean_same, mean_cross, len(same_tvs), len(cross_tvs)


def run_parser_null(data, cfg):
    """Text parser baseline: parse entity locations from prefix text."""
    print("Parser null: predict from text alone")
    locs = cfg["locations"]
    ents = cfg["entities"]
    parser_tvs = []
    for state, rows in data.items():
        a_loc, b_loc = state
        for r in rows:
            for qi in range(len(ents)):
                correct = a_loc if qi == 0 else b_loc
                parser_dist = {l: (1.0 if l == correct else 0.0) for l in locs}
                model_dist = r[f"q{qi}"]
                parser_tvs.append(tv(parser_dist, model_dist, locs))
    mean_parser_tv = float(np.mean(parser_tvs))
    print(f"  Parser-to-model mean TV: {mean_parser_tv:.4f}")
    return mean_parser_tv


def main():
    cfg = load_config()
    np.random.seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    t_start = time.time()

    print(f"RCQ-0 experiment: {cfg['experiment_id']}")
    print(f"Model: {cfg['model_id']}")
    print(f"Loading model...")
    model = RCQModel(cfg)
    print(f"Model loaded in {time.time()-t_start:.1f}s\n")

    histories = make_histories(cfg)
    actions = make_actions(cfg)
    queries = make_queries(cfg)
    print(f"Task: {len(histories)} joint states, {sum(len(v) for v in histories.values())} histories")
    print(f"Actions: {len(actions)}, Queries: {len(queries)}\n")

    # Phase 1
    data, entity_disc = run_phase1(model, histories, queries, cfg)
    disc_pass = entity_disc > cfg["gates"]["entity_discrimination_tv"]
    print(f"  GATE: {'PASS' if disc_pass else 'FAIL'}\n")
    if not disc_pass:
        print("ABORT: Entity discrimination gate failed.")
        return

    # Phase 2
    class_map, classes, centroids = run_phase2(data, cfg)

    # Phase 3
    transition_table, trans_cons = run_phase3(model, data, class_map, centroids, actions, queries, cfg)
    trans_pass = trans_cons >= cfg["gates"]["within_class_transition_consistency"]
    print(f"  GATE: {'PASS' if trans_pass else 'FAIL'}\n")

    # Phase 4
    top1_rate, comp_tv, n_comp = run_phase4(model, data, class_map, centroids, transition_table, actions, queries, cfg)
    comp_pass = (top1_rate >= cfg["gates"]["composition_top1_agreement"]
                 and comp_tv <= cfg["gates"]["composition_tv_max"])
    print(f"  GATE: {'PASS' if comp_pass else 'FAIL'}\n")

    # Phase 5
    same_tv, cross_tv, n_same, n_cross = run_phase5(model, data, class_map, classes, queries, cfg)
    sub_pass = (same_tv <= cfg["gates"]["substitution_same_class_tv_max"]
                and cross_tv >= cfg["gates"]["substitution_cross_class_tv_min"])
    print(f"  GATE: {'PASS' if sub_pass else 'FAIL'}\n")

    # Parser null
    parser_tv = run_parser_null(data, cfg)

    # Overall verdict
    elapsed = time.time() - t_start
    all_pass = disc_pass and trans_pass and comp_pass and sub_pass
    result = {
        "experiment_id": cfg["experiment_id"],
        "model_id": cfg["model_id"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "n_joint_states": cfg["n_joint_states"],
        "n_histories": sum(len(v) for v in histories.values()),
        "n_classes": len(classes),
        "class_sizes": sorted([len(v) for v in classes.values()], reverse=True),
        "gates": {
            "entity_discrimination": {"value": round(entity_disc, 4), "threshold": cfg["gates"]["entity_discrimination_tv"], "pass": disc_pass},
            "transition_consistency": {"value": round(trans_cons, 4), "threshold": cfg["gates"]["within_class_transition_consistency"], "pass": trans_pass},
            "composition_top1": {"value": round(top1_rate, 4), "threshold": cfg["gates"]["composition_top1_agreement"], "pass": comp_pass},
            "composition_tv": {"value": round(comp_tv, 4), "threshold": cfg["gates"]["composition_tv_max"]},
            "substitution_same": {"value": round(same_tv, 4), "threshold": cfg["gates"]["substitution_same_class_tv_max"], "pass": sub_pass},
            "substitution_cross": {"value": round(cross_tv, 4), "threshold": cfg["gates"]["substitution_cross_class_tv_min"]},
        },
        "parser_null_tv": round(parser_tv, 4),
        "overall_pass": all_pass,
        "status": "PASS" if all_pass else "FAIL",
    }

    print(f"\n{'='*60}")
    print(f"RCQ-0 RESULT: {result['status']}")
    print(f"{'='*60}")
    print(f"  Classes found: {result['n_classes']}")
    print(f"  Entity discrimination: {entity_disc:.4f} {'PASS' if disc_pass else 'FAIL'}")
    print(f"  Transition consistency: {trans_cons:.4f} {'PASS' if trans_pass else 'FAIL'}")
    print(f"  Composition top-1: {top1_rate:.4f} {'PASS' if comp_pass else 'FAIL'}")
    print(f"  Composition TV: {comp_tv:.4f}")
    print(f"  Substitution same-class: {same_tv:.4f} {'PASS' if sub_pass else 'FAIL'}")
    print(f"  Substitution cross-class: {cross_tv:.4f}")
    print(f"  Parser null TV: {parser_tv:.4f}")
    print(f"  Elapsed: {elapsed:.1f}s")

    outpath = Path(cfg["output_dir"]) / "result.json"
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {outpath}")

    # Save transition table
    tt_out = {}
    for (src, ai), tgt_class in transition_table.items():
        tt_out[f"{src}_{ai}"] = tgt_class
    with open(Path(cfg["output_dir"]) / "transition_table.json", "w") as f:
        json.dump(tt_out, f, indent=2)

    return result


if __name__ == "__main__":
    main()
