"""PFC-0: Path-Fiber Calculus v1 — four-corner path-square experiment.

Locked specification from Codex R3 (2026-09-02). Implements:
- Three panels (KG_O, GO_K, OK_G) with cyclic relative coordinates
- Four corners (p00, pL, pR, pLR) with washout tail
- Cross-fitted stochastic-channel transports (SLSQP)
- Seven baselines: parser, last-1, last-2, multiset, discounted, primitive, causal-kNN
- State replay/injection validation
- Root-clustered bootstrap CIs
- Full adjudication tree
"""
import copy
import hashlib
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

LOCS = ["kitchen", "garden", "office"]
ENTS = ["Avery", "Blake"]

@dataclass
class Root:
    id: str
    a_location: str
    b_location: str
    fold: int

@dataclass
class Panel:
    id: str
    first: str
    second: str
    reset: str
    canonical_order: list  # [first, second, reset]

@dataclass
class PathSpec:
    id: str
    action_roles: list  # e.g. ["A1","B1","A2","B2"]
    washed: bool

@dataclass
class PromptSpec:
    key: str
    root_id: str
    panel_id: str
    path_id: str
    query_entity: str
    prefix: str
    query: str
    expected_location: str

@dataclass
class Observation:
    key: str
    distribution: np.ndarray

@dataclass
class Prediction:
    fold: int
    root_id: str
    panel_id: str
    query_entity: str
    method: str
    distribution: np.ndarray
    target: np.ndarray
    tv: float

# ---------------------------------------------------------------------------
# Config / lock
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return json.load(f)

def validate_lock(config_bytes):
    return hashlib.sha256(config_bytes).hexdigest()

# ---------------------------------------------------------------------------
# Model adapter
# ---------------------------------------------------------------------------

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
        self.aid = {}
        for loc in cfg["locations"]:
            toks = self.tok.encode(f" {loc}", add_special_tokens=False)
            assert len(toks) == 1, f"'{loc}' is not a single token: {toks}"
            self.aid[loc] = toks[0]
        self.answer_ids = [self.aid[l] for l in cfg["locations"]]
        self.model_revision = cfg["model_revision"]
        self.call_count = 0

    def get_dist(self, text):
        ids = self.tok.encode(text, return_tensors="pt")
        with torch.no_grad():
            out = self.mdl(ids)
        logits = out.logits[0, -1, :]
        vals = torch.tensor([logits[self.aid[l]].item() for l in LOCS])
        dist = torch.softmax(vals, dim=0).numpy().astype(np.float64)
        self.call_count += 1
        return dist

    def get_dist_from_state(self, state, continuation_ids):
        with torch.no_grad():
            out = self.mdl(continuation_ids, state=state)
        logits = out.logits[0, -1, :]
        vals = torch.tensor([logits[self.aid[l]].item() for l in LOCS])
        dist = torch.softmax(vals, dim=0).numpy().astype(np.float64)
        self.call_count += 1
        return dist

    def get_state_and_dist(self, text, query):
        full = text + query
        full_ids = self.tok.encode(full, return_tensors="pt")
        prefix_ids = self.tok.encode(text, return_tensors="pt")
        prefix_len = prefix_ids.shape[1]
        assert (full_ids[0, :prefix_len] == prefix_ids[0]).all(), \
            "Prefix tokens must be exact prefix of full sequence"
        with torch.no_grad():
            prefix_out = self.mdl(prefix_ids)
        state = prefix_out.state
        state_clone = copy.deepcopy(state)
        continuation = full_ids[:, prefix_len:]
        with torch.no_grad():
            cont_out = self.mdl(continuation, state=state_clone)
        logits = cont_out.logits[0, -1, :]
        vals = torch.tensor([logits[self.aid[l]].item() for l in LOCS])
        dist = torch.softmax(vals, dim=0).numpy().astype(np.float64)
        self.call_count += 2
        return copy.deepcopy(prefix_out.state), dist, continuation

def load_model(cfg):
    print("Loading model...", flush=True)
    t0 = time.time()
    adapter = FinchAdapter(cfg)
    print(f"Model loaded in {time.time()-t0:.1f}s. Answer token IDs: {adapter.answer_ids}", flush=True)
    return adapter

# ---------------------------------------------------------------------------
# Population construction
# ---------------------------------------------------------------------------

def make_roots(cfg):
    roots = []
    fold_map = {}
    for fi, (fname, members) in enumerate(cfg["folds"].items()):
        for m in members:
            fold_map[tuple(m)] = fi
    for a in cfg["locations"]:
        for b in cfg["locations"]:
            rid = f"{a[0]}{b[0]}"
            roots.append(Root(id=rid, a_location=a, b_location=b, fold=fold_map[(a, b)]))
    return roots

def make_panels(cfg):
    panels = []
    for pid, spec in cfg["panels"].items():
        panels.append(Panel(
            id=pid,
            first=spec["first"],
            second=spec["second"],
            reset=spec["reset"],
            canonical_order=[spec["first"], spec["second"], spec["reset"]]
        ))
    return panels

def make_paths():
    washed = [
        PathSpec(id="p00",  action_roles=["A1","B1","A2","B2"], washed=True),
        PathSpec(id="pL",   action_roles=["B1","A1","A2","B2"], washed=True),
        PathSpec(id="pR",   action_roles=["A1","B1","B2","A2"], washed=True),
        PathSpec(id="pLR",  action_roles=["B1","A1","B2","A2"], washed=True),
    ]
    raw = [
        PathSpec(id="p00_raw",  action_roles=["A1","B1","A2","B2"], washed=False),
        PathSpec(id="pLR_raw",  action_roles=["B1","A1","B2","A2"], washed=False),
    ]
    return washed + raw

def render_action(entity, location):
    return f" {entity} moved to the {location}."

def render_prefix(root, panel, path, entities, washed):
    base = f"{entities[0]} is in the {root.a_location}. {entities[1]} is in the {root.b_location}."
    role_to_action = {
        "A1": render_action(entities[0], panel.first),
        "A2": render_action(entities[0], panel.second),
        "B1": render_action(entities[1], panel.first),
        "B2": render_action(entities[1], panel.second),
    }
    actions = "".join(role_to_action[r] for r in path.action_roles)
    if washed:
        rA = render_action(entities[0], panel.reset)
        rB = render_action(entities[1], panel.reset)
        return base + actions + rA + rB + " Now,"
    else:
        return base + actions + " Now,"

def compute_expected_location(root, panel, path, query_entity, entities, washed):
    if washed:
        return panel.reset
    state = {entities[0]: root.a_location, entities[1]: root.b_location}
    role_to_move = {
        "A1": (entities[0], panel.first),
        "A2": (entities[0], panel.second),
        "B1": (entities[1], panel.first),
        "B2": (entities[1], panel.second),
    }
    for r in path.action_roles:
        ent, loc = role_to_move[r]
        state[ent] = loc
    return state[query_entity]

def make_prompt_specs(roots, panels, paths, cfg):
    entities = cfg["entities"]
    specs = []
    for root in roots:
        for panel in panels:
            for path in paths:
                for qe in entities:
                    query = f"\nQuestion: Where is {qe}?\nAnswer: The"
                    prefix = render_prefix(root, panel, path, entities, path.washed)
                    expected = compute_expected_location(root, panel, path, qe, entities, path.washed)
                    key = f"{root.id}_{panel.id}_{path.id}_{qe}"
                    specs.append(PromptSpec(
                        key=key,
                        root_id=root.id,
                        panel_id=panel.id,
                        path_id=path.id,
                        query_entity=qe,
                        prefix=prefix,
                        query=query,
                        expected_location=expected,
                    ))
    return specs

# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def collect_base(adapter, specs, cache_path):
    cache = {}
    if cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=True)
        for k in loaded.files:
            cache[k] = loaded[k]
        print(f"Loaded {len(cache)} cached observations", flush=True)

    observations = {}
    total = len(specs)
    for i, spec in enumerate(specs):
        if spec.key in cache:
            observations[spec.key] = Observation(key=spec.key, distribution=cache[spec.key])
        else:
            text = spec.prefix + spec.query
            dist = adapter.get_dist(text)
            observations[spec.key] = Observation(key=spec.key, distribution=dist)
            cache[spec.key] = dist
            if (i + 1) % 18 == 0 or i == total - 1:
                np.savez(cache_path, **cache)
                print(f"  collected {i+1}/{total} (calls: {adapter.call_count})", flush=True)

    if len(cache) > len(observations):
        np.savez(cache_path, **{k: v for k, v in cache.items() if k in {s.key for s in specs}})

    return observations

def collect_replay(adapter, observations, roots, panels, cfg):
    entities = cfg["entities"]
    diagonal_roots = [r for r in roots if r.fold == 0]
    results = []

    for root in diagonal_roots:
        for panel in panels:
            for corner_id in ["p00", "pLR"]:
                path = PathSpec(id=corner_id,
                                action_roles=["A1","B1","A2","B2"] if corner_id == "p00"
                                else ["B1","A1","B2","A2"],
                                washed=True)
                prefix = render_prefix(root, panel, path, entities, True)
                for qe in entities:
                    query = f"\nQuestion: Where is {qe}?\nAnswer: The"
                    obs_key = f"{root.id}_{panel.id}_{corner_id}_{qe}"
                    oneshot_dist = observations[obs_key].distribution
                    state, replay_dist, cont_ids = adapter.get_state_and_dist(prefix, query)
                    tv_val = tv(oneshot_dist, replay_dist)
                    results.append({
                        "root": root.id, "panel": panel.id,
                        "corner": corner_id, "query": qe,
                        "oneshot": oneshot_dist.tolist(),
                        "replay": replay_dist.tolist(),
                        "tv": float(tv_val),
                    })
    return results

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def tv(a, b):
    return 0.5 * np.sum(np.abs(a - b))

def canonicalize(dist, panel):
    idx = [LOCS.index(panel.first), LOCS.index(panel.second), LOCS.index(panel.reset)]
    return dist[idx]

def uncanonicalize(canon_dist, panel):
    out = np.zeros(3)
    idx = [LOCS.index(panel.first), LOCS.index(panel.second), LOCS.index(panel.reset)]
    for i, j in enumerate(idx):
        out[j] = canon_dist[i]
    return out

# ---------------------------------------------------------------------------
# Transport fitting (SLSQP)
# ---------------------------------------------------------------------------

def fit_stochastic_channel(source, target, ridge, cfg):
    n = source.shape[1]
    K0 = np.eye(n).flatten()

    def objective(k_flat):
        K = k_flat.reshape(n, n)
        pred = source @ K
        mse = np.mean(np.sum((pred - target) ** 2, axis=1))
        reg = ridge * np.sum((K - np.eye(n)) ** 2)
        return mse + reg

    def grad(k_flat):
        K = k_flat.reshape(n, n)
        pred = source @ K
        residual = pred - target
        g_mse = 2.0 / source.shape[0] * (source.T @ residual)
        g_reg = 2.0 * ridge * (K - np.eye(n))
        return (g_mse + g_reg).flatten()

    constraints = []
    for i in range(n):
        def row_sum(k_flat, i=i):
            K = k_flat.reshape(n, n)
            return np.sum(K[i]) - 1.0
        constraints.append({"type": "eq", "fun": row_sum})

    bounds = [(0.0, None)] * (n * n)

    result = minimize(
        objective, K0, jac=grad, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"ftol": cfg["slsqp_ftol"], "maxiter": cfg["slsqp_maxiter"],
                 "disp": False}
    )
    if not result.success:
        return None
    return result.x.reshape(n, n)

def compose_square(mu00, k_left, k_right):
    pred_lr = mu00 @ k_left @ k_right
    pred_rl = mu00 @ k_right @ k_left
    return pred_lr, pred_rl, 0.5 * (pred_lr + pred_rl)

def crossfit_path_calculus(observations, roots, panels, folds_cfg, cfg):
    fold_assignments = {}
    for fi, (fname, members) in enumerate(folds_cfg.items()):
        for m in members:
            fold_assignments[tuple(m)] = fi

    fits = {}
    predictions = []

    for test_fold in range(3):
        train_roots = [r for r in roots if r.fold != test_fold]
        test_roots = [r for r in roots if r.fold == test_fold]

        for qe in ENTS:
            src_L, tgt_L = [], []
            src_R, tgt_R = [], []
            for root in train_roots:
                for panel in panels:
                    key00 = f"{root.id}_{panel.id}_p00_{qe}"
                    keyL  = f"{root.id}_{panel.id}_pL_{qe}"
                    keyR  = f"{root.id}_{panel.id}_pR_{qe}"
                    d00 = canonicalize(observations[key00].distribution, panel)
                    dL  = canonicalize(observations[keyL].distribution, panel)
                    dR  = canonicalize(observations[keyR].distribution, panel)
                    src_L.append(d00); tgt_L.append(dL)
                    src_R.append(d00); tgt_R.append(dR)

            src_L = np.array(src_L)
            tgt_L = np.array(tgt_L)
            src_R = np.array(src_R)
            tgt_R = np.array(tgt_R)

            K_L = fit_stochastic_channel(src_L, tgt_L, cfg["ridge"], cfg)
            K_R = fit_stochastic_channel(src_R, tgt_R, cfg["ridge"], cfg)

            fit_key = f"fold{test_fold}_{qe}"
            fits[fit_key] = {
                "K_L": K_L.tolist() if K_L is not None else None,
                "K_R": K_R.tolist() if K_R is not None else None,
                "n_train": len(src_L),
            }

            if K_L is None or K_R is None:
                for root in test_roots:
                    for panel in panels:
                        predictions.append(Prediction(
                            fold=test_fold, root_id=root.id, panel_id=panel.id,
                            query_entity=qe, method="pfc",
                            distribution=None, target=None, tv=float("inf")
                        ))
                continue

            for root in test_roots:
                for panel in panels:
                    key00 = f"{root.id}_{panel.id}_p00_{qe}"
                    keyLR = f"{root.id}_{panel.id}_pLR_{qe}"
                    d00 = canonicalize(observations[key00].distribution, panel)
                    dLR = canonicalize(observations[keyLR].distribution, panel)

                    pred_lr, pred_rl, pred_avg = compose_square(
                        d00.reshape(1, -1), K_L, K_R
                    )
                    pred_avg = pred_avg.flatten()
                    pred_avg = np.clip(pred_avg, 0, None)
                    if pred_avg.sum() > 0:
                        pred_avg /= pred_avg.sum()

                    pred_uncan = uncanonicalize(pred_avg, panel)
                    target_uncan = uncanonicalize(dLR, panel)

                    predictions.append(Prediction(
                        fold=test_fold, root_id=root.id, panel_id=panel.id,
                        query_entity=qe, method="pfc",
                        distribution=pred_uncan, target=target_uncan,
                        tv=float(tv(pred_uncan, target_uncan))
                    ))

                    pred_lr_uncan = uncanonicalize(np.clip(pred_lr.flatten(), 0, None), panel)
                    if pred_lr_uncan.sum() > 0:
                        pred_lr_uncan /= pred_lr_uncan.sum()
                    pred_rl_uncan = uncanonicalize(np.clip(pred_rl.flatten(), 0, None), panel)
                    if pred_rl_uncan.sum() > 0:
                        pred_rl_uncan /= pred_rl_uncan.sum()

                    predictions.append(Prediction(
                        fold=test_fold, root_id=root.id, panel_id=panel.id,
                        query_entity=qe, method="pfc_lr",
                        distribution=pred_lr_uncan, target=target_uncan,
                        tv=float(tv(pred_lr_uncan, target_uncan))
                    ))
                    predictions.append(Prediction(
                        fold=test_fold, root_id=root.id, panel_id=panel.id,
                        query_entity=qe, method="pfc_rl",
                        distribution=pred_rl_uncan, target=target_uncan,
                        tv=float(tv(pred_rl_uncan, target_uncan))
                    ))

    return fits, predictions

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def get_semantic_actions(panel, path):
    role_to_semantic = {
        "A1": ("Avery", panel.first),
        "A2": ("Avery", panel.second),
        "B1": ("Blake", panel.first),
        "B2": ("Blake", panel.second),
    }
    return [role_to_semantic[r] for r in path.action_roles]

def fit_parser_baseline(observations, roots, panels, folds_cfg, cfg):
    predictions = []
    for test_fold in range(3):
        train_roots = [r for r in roots if r.fold != test_fold]
        test_roots = [r for r in roots if r.fold == test_fold]
        for qe in ENTS:
            residuals_by_endpoint = defaultdict(list)
            for root in train_roots:
                for panel in panels:
                    key00 = f"{root.id}_{panel.id}_p00_{qe}"
                    keyLR = f"{root.id}_{panel.id}_pLR_{qe}"
                    d00 = observations[key00].distribution
                    dLR = observations[keyLR].distribution
                    endpoint = panel.reset
                    residuals_by_endpoint[endpoint].append(dLR - d00)
            mean_res = {ep: np.mean(rs, axis=0) for ep, rs in residuals_by_endpoint.items()}
            for root in test_roots:
                for panel in panels:
                    key00 = f"{root.id}_{panel.id}_p00_{qe}"
                    keyLR = f"{root.id}_{panel.id}_pLR_{qe}"
                    d00 = observations[key00].distribution
                    dLR = observations[keyLR].distribution
                    endpoint = panel.reset
                    pred = d00 + mean_res.get(endpoint, np.zeros(3))
                    pred = np.clip(pred, 0, None)
                    if pred.sum() > 0:
                        pred /= pred.sum()
                    predictions.append(Prediction(
                        fold=test_fold, root_id=root.id, panel_id=panel.id,
                        query_entity=qe, method="parser",
                        distribution=pred, target=dLR, tv=float(tv(pred, dLR))
                    ))
    return predictions

def get_last_k_actions(panel, path, k):
    actions = get_semantic_actions(panel, path)
    washed_actions = actions + [("Avery", panel.reset), ("Blake", panel.reset)]
    return tuple(washed_actions[-k:])

def fit_last_k_baseline(observations, roots, panels, k, cfg):
    predictions = []
    for test_fold in range(3):
        train_roots = [r for r in roots if r.fold != test_fold]
        test_roots = [r for r in roots if r.fold == test_fold]
        for qe in ENTS:
            washed_paths = [p for p in make_paths() if p.washed]
            residuals_by_key = defaultdict(list)
            for root in train_roots:
                for panel in panels:
                    for path in washed_paths:
                        last_k = get_last_k_actions(panel, path, k)
                        obs_key = f"{root.id}_{panel.id}_{path.id}_{qe}"
                        key00 = f"{root.id}_{panel.id}_p00_{qe}"
                        d = observations[obs_key].distribution
                        d00 = observations[key00].distribution
                        residuals_by_key[(last_k, qe)].append(d - d00)

            for root in test_roots:
                for panel in panels:
                    pLR_path = [p for p in washed_paths if p.id == "pLR"][0]
                    last_k = get_last_k_actions(panel, pLR_path, k)
                    key00 = f"{root.id}_{panel.id}_p00_{qe}"
                    keyLR = f"{root.id}_{panel.id}_pLR_{qe}"
                    d00 = observations[key00].distribution
                    dLR = observations[keyLR].distribution
                    rkey = (last_k, qe)
                    if rkey in residuals_by_key:
                        pred = d00 + np.mean(residuals_by_key[rkey], axis=0)
                    else:
                        pred = d00.copy()
                    pred = np.clip(pred, 0, None)
                    if pred.sum() > 0:
                        pred /= pred.sum()
                    predictions.append(Prediction(
                        fold=test_fold, root_id=root.id, panel_id=panel.id,
                        query_entity=qe, method=f"last_{k}",
                        distribution=pred, target=dLR, tv=float(tv(pred, dLR))
                    ))
    return predictions

def fit_multiset_baseline(observations, roots, panels, cfg):
    predictions = []
    washed_paths = [p for p in make_paths() if p.washed]
    for test_fold in range(3):
        train_roots = [r for r in roots if r.fold != test_fold]
        test_roots = [r for r in roots if r.fold == test_fold]
        for qe in ENTS:
            residuals_by_mset = defaultdict(list)
            for root in train_roots:
                for panel in panels:
                    for path in washed_paths:
                        actions = get_semantic_actions(panel, path)
                        actions += [("Avery", panel.reset), ("Blake", panel.reset)]
                        mset = tuple(sorted(actions))
                        obs_key = f"{root.id}_{panel.id}_{path.id}_{qe}"
                        key00 = f"{root.id}_{panel.id}_p00_{qe}"
                        d = observations[obs_key].distribution
                        d00 = observations[key00].distribution
                        residuals_by_mset[(mset, qe)].append(d - d00)

            for root in test_roots:
                for panel in panels:
                    pLR_path = [p for p in washed_paths if p.id == "pLR"][0]
                    actions = get_semantic_actions(panel, pLR_path)
                    actions += [("Avery", panel.reset), ("Blake", panel.reset)]
                    mset = tuple(sorted(actions))
                    key00 = f"{root.id}_{panel.id}_p00_{qe}"
                    keyLR = f"{root.id}_{panel.id}_pLR_{qe}"
                    d00 = observations[key00].distribution
                    dLR = observations[keyLR].distribution
                    rkey = (mset, qe)
                    if rkey in residuals_by_mset:
                        pred = d00 + np.mean(residuals_by_mset[rkey], axis=0)
                    else:
                        pred = d00.copy()
                    pred = np.clip(pred, 0, None)
                    if pred.sum() > 0:
                        pred /= pred.sum()
                    predictions.append(Prediction(
                        fold=test_fold, root_id=root.id, panel_id=panel.id,
                        query_entity=qe, method="multiset",
                        distribution=pred, target=dLR, tv=float(tv(pred, dLR))
                    ))
    return predictions

def encode_action_role(entity, location):
    ent_idx = ENTS.index(entity)
    loc_idx = LOCS.index(location)
    vec = np.zeros(6)
    vec[ent_idx * 3 + loc_idx] = 1.0
    return vec

def _discounted_features(actions, lam):
    x = np.zeros(6)
    for t, (ent, loc) in enumerate(actions):
        x += (lam ** (len(actions) - 1 - t)) * encode_action_role(ent, loc)
    return x

def _discounted_fit_predict(observations, train_roots, panels, washed_paths, qe, lam, cfg):
    X, Y = [], []
    for root in train_roots:
        for panel in panels:
            for path in washed_paths:
                actions = get_semantic_actions(panel, path)
                actions += [("Avery", panel.reset), ("Blake", panel.reset)]
                X.append(_discounted_features(actions, lam))
                key = f"{root.id}_{panel.id}_{path.id}_{qe}"
                key00 = f"{root.id}_{panel.id}_p00_{qe}"
                Y.append(observations[key].distribution - observations[key00].distribution)
    X = np.array(X)
    Y = np.array(Y)
    XtX = X.T @ X + cfg["ridge"] * np.eye(6)
    return np.linalg.solve(XtX, X.T @ Y)

def fit_discounted_history_baseline(observations, roots, panels, cfg):
    washed_paths = [p for p in make_paths() if p.washed]
    lambdas = cfg["discounted_lambdas"]
    all_predictions = []
    selected_lambdas = []

    for test_fold in range(3):
        train_roots = [r for r in roots if r.fold != test_fold]
        test_roots = [r for r in roots if r.fold == test_fold]

        best_lam, best_inner_tv = None, float("inf")
        for lam in lambdas:
            inner_tvs = []
            for loo_root in train_roots:
                loo_train = [r for r in train_roots if r.id != loo_root.id]
                for qe in ENTS:
                    W = _discounted_fit_predict(observations, loo_train, panels, washed_paths, qe, lam, cfg)
                    for panel in panels:
                        pLR_path = [p for p in washed_paths if p.id == "pLR"][0]
                        actions = get_semantic_actions(panel, pLR_path)
                        actions += [("Avery", panel.reset), ("Blake", panel.reset)]
                        x = _discounted_features(actions, lam)
                        k00 = f"{loo_root.id}_{panel.id}_p00_{qe}"
                        kLR = f"{loo_root.id}_{panel.id}_pLR_{qe}"
                        d00 = observations[k00].distribution
                        dLR = observations[kLR].distribution
                        pred = d00 + x @ W
                        pred = np.clip(pred, 0, None)
                        if pred.sum() > 0:
                            pred /= pred.sum()
                        inner_tvs.append(tv(pred, dLR))
            mean_inner = np.mean(inner_tvs)
            if mean_inner < best_inner_tv or (mean_inner == best_inner_tv and lam < best_lam):
                best_inner_tv = mean_inner
                best_lam = lam

        selected_lambdas.append(best_lam)
        for qe in ENTS:
            W = _discounted_fit_predict(observations, train_roots, panels, washed_paths, qe, best_lam, cfg)
            for root in test_roots:
                for panel in panels:
                    pLR_path = [p for p in washed_paths if p.id == "pLR"][0]
                    actions = get_semantic_actions(panel, pLR_path)
                    actions += [("Avery", panel.reset), ("Blake", panel.reset)]
                    x = _discounted_features(actions, best_lam)
                    key00 = f"{root.id}_{panel.id}_p00_{qe}"
                    keyLR = f"{root.id}_{panel.id}_pLR_{qe}"
                    d00 = observations[key00].distribution
                    dLR = observations[keyLR].distribution
                    pred = d00 + x @ W
                    pred = np.clip(pred, 0, None)
                    if pred.sum() > 0:
                        pred /= pred.sum()
                    all_predictions.append(Prediction(
                        fold=test_fold, root_id=root.id, panel_id=panel.id,
                        query_entity=qe, method="discounted",
                        distribution=pred, target=dLR, tv=float(tv(pred, dLR))
                    ))
    return all_predictions, selected_lambdas

def get_primitive_action_type(entity, location, panel):
    ent_idx = ENTS.index(entity)
    if location == panel.first:
        loc_type = 0
    elif location == panel.second:
        loc_type = 1
    else:
        loc_type = 2
    return ent_idx * 3 + loc_type

def fit_primitive_composition_baseline(observations, roots, panels, cfg):
    torch.manual_seed(cfg["primitive_seed"])
    np.random.seed(cfg["primitive_seed"])
    washed_paths = [p for p in make_paths() if p.washed]

    predictions = []
    for test_fold in range(3):
        train_roots = [r for r in roots if r.fold != test_fold]
        test_roots = [r for r in roots if r.fold == test_fold]

        for qe in ENTS:
            channels = torch.nn.ParameterList([
                torch.nn.Parameter(torch.eye(3) * cfg["primitive_init_diag"]
                                   + (1 - cfg["primitive_init_diag"]) / 3)
                for _ in range(6)
            ])
            optimizer = torch.optim.Adam(channels.parameters(), lr=cfg["primitive_lr"])

            train_data = []
            for root in train_roots:
                for panel in panels:
                    key00 = f"{root.id}_{panel.id}_p00_{qe}"
                    d00 = canonicalize(observations[key00].distribution, panel)
                    for path in washed_paths:
                        obs_key = f"{root.id}_{panel.id}_{path.id}_{qe}"
                        target = canonicalize(observations[obs_key].distribution, panel)
                        actions = get_semantic_actions(panel, path)
                        actions += [("Avery", panel.reset), ("Blake", panel.reset)]
                        action_types = [get_primitive_action_type(e, l, panel) for e, l in actions]
                        train_data.append((d00, target, action_types, panel))

            for step in range(cfg["primitive_steps"]):
                total_loss = 0.0
                for d00, target, atypes, pan in train_data:
                    x = torch.tensor(d00, dtype=torch.float32).unsqueeze(0)
                    for at in atypes:
                        ch = torch.softmax(channels[at], dim=1)
                        x = x @ ch
                    t = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
                    loss = 0.5 * torch.sum(torch.abs(x - t))
                    total_loss += loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            for root in test_roots:
                for panel in panels:
                    pLR_path = [p for p in washed_paths if p.id == "pLR"][0]
                    key00 = f"{root.id}_{panel.id}_p00_{qe}"
                    keyLR = f"{root.id}_{panel.id}_pLR_{qe}"
                    d00 = canonicalize(observations[key00].distribution, panel)
                    dLR = canonicalize(observations[keyLR].distribution, panel)

                    actions = get_semantic_actions(panel, pLR_path)
                    actions += [("Avery", panel.reset), ("Blake", panel.reset)]
                    action_types = [get_primitive_action_type(e, l, panel) for e, l in actions]

                    x = torch.tensor(d00, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        for at in action_types:
                            ch = torch.softmax(channels[at], dim=1)
                            x = x @ ch
                    pred_canon = x.numpy().flatten()
                    pred_canon = np.clip(pred_canon, 0, None)
                    if pred_canon.sum() > 0:
                        pred_canon /= pred_canon.sum()
                    pred = uncanonicalize(pred_canon, panel)
                    target = uncanonicalize(dLR, panel)
                    predictions.append(Prediction(
                        fold=test_fold, root_id=root.id, panel_id=panel.id,
                        query_entity=qe, method="primitive",
                        distribution=pred, target=target, tv=float(tv(pred, target))
                    ))
    return predictions

def predict_causal_knn(observations, roots, panels, cfg):
    predictions = []
    for test_fold in range(3):
        train_roots = [r for r in roots if r.fold != test_fold]
        test_roots = [r for r in roots if r.fold == test_fold]

        source_cells = []
        for root in train_roots:
            for panel in panels:
                d00_a = observations[f"{root.id}_{panel.id}_p00_Avery"].distribution
                d00_b = observations[f"{root.id}_{panel.id}_p00_Blake"].distribution
                rep = np.concatenate([
                    canonicalize(d00_a, panel),
                    canonicalize(d00_b, panel)
                ])
                dLR_a = observations[f"{root.id}_{panel.id}_pLR_Avery"].distribution
                dLR_b = observations[f"{root.id}_{panel.id}_pLR_Blake"].distribution
                source_cells.append({
                    "root_id": root.id, "panel_id": panel.id,
                    "rep": rep,
                    "pLR_Avery": dLR_a, "pLR_Blake": dLR_b,
                })

        for root in test_roots:
            for panel in panels:
                d00_a = observations[f"{root.id}_{panel.id}_p00_Avery"].distribution
                d00_b = observations[f"{root.id}_{panel.id}_p00_Blake"].distribution
                query_rep = np.concatenate([
                    canonicalize(d00_a, panel),
                    canonicalize(d00_b, panel)
                ])

                best_dist = float("inf")
                best_cell = None
                for cell in source_cells:
                    d = 0.5 * (
                        tv(query_rep[:3], cell["rep"][:3]) +
                        tv(query_rep[3:], cell["rep"][3:])
                    )
                    if d < best_dist or (d == best_dist and
                        (cell["root_id"], cell["panel_id"]) <
                        (best_cell["root_id"], best_cell["panel_id"])):
                        best_dist = d
                        best_cell = cell

                for qe in ENTS:
                    keyLR = f"{root.id}_{panel.id}_pLR_{qe}"
                    dLR = observations[keyLR].distribution
                    pred = best_cell[f"pLR_{qe}"]
                    predictions.append(Prediction(
                        fold=test_fold, root_id=root.id, panel_id=panel.id,
                        query_entity=qe, method="causal_knn",
                        distribution=pred, target=dLR, tv=float(tv(pred, dLR))
                    ))
    return predictions

# ---------------------------------------------------------------------------
# Bootstrap & metrics
# ---------------------------------------------------------------------------

def root_cluster_bootstrap(values_by_root, n_resamples, seed):
    rng = np.random.RandomState(seed)
    root_ids = sorted(values_by_root.keys())
    root_means = {rid: np.mean(vals) for rid, vals in values_by_root.items()}
    boot_means = []
    for _ in range(n_resamples):
        sampled = rng.choice(root_ids, size=len(root_ids), replace=True)
        boot_means.append(np.mean([root_means[r] for r in sampled]))
    boot_means = np.array(boot_means)
    return {
        "mean": float(np.mean(boot_means)),
        "ci95_lower": float(np.percentile(boot_means, 2.5)),
        "ci95_upper": float(np.percentile(boot_means, 97.5)),
    }

def compute_defects(observations, roots, panels, washed):
    defects = []
    for root in roots:
        for panel in panels:
            corner_prefix = "" if washed else "_raw"
            id00 = "p00" + ("" if washed else "_raw")
            idLR = "pLR" + ("" if washed else "_raw")
            kappas = []
            for qe in ENTS:
                key00 = f"{root.id}_{panel.id}_{id00}_{qe}"
                keyLR = f"{root.id}_{panel.id}_{idLR}_{qe}"
                if key00 in observations and keyLR in observations:
                    d00 = observations[key00].distribution
                    dLR = observations[keyLR].distribution
                    kappas.append(tv(d00, dLR))
            if kappas:
                defects.append({
                    "root_id": root.id, "panel_id": panel.id,
                    "kappa": float(np.mean(kappas)),
                    "kappas_by_query": [float(k) for k in kappas],
                })
    return defects

def compute_accuracy(observations, specs, washed):
    correct = 0
    total = 0
    arm_correct = defaultdict(int)
    arm_total = defaultdict(int)
    for spec in specs:
        if washed and not any(spec.path_id == p for p in ["p00", "pL", "pR", "pLR"]):
            continue
        if not washed and not any(spec.path_id == p for p in ["p00_raw", "pLR_raw"]):
            continue
        if spec.key in observations:
            dist = observations[spec.key].distribution
            pred_loc = LOCS[np.argmax(dist)]
            is_correct = pred_loc == spec.expected_location
            correct += int(is_correct)
            total += 1
            arm_key = f"{spec.panel_id}_{spec.query_entity}"
            arm_correct[arm_key] += int(is_correct)
            arm_total[arm_key] += 1
    overall = correct / total if total > 0 else 0
    arm_accs = {k: arm_correct[k] / arm_total[k] for k in arm_total}
    return overall, arm_accs

# ---------------------------------------------------------------------------
# Adjudication
# ---------------------------------------------------------------------------

def adjudicate(metrics, gates):
    if not metrics["integrity_pass"]:
        return "INVALID_IMPLEMENTATION"
    if not metrics["competence_pass"]:
        return "TASK_POPULATION_VOID"
    if not metrics["raw_order_pass"]:
        return "NO_ORDER_SIGNAL"
    if not metrics["wash_eligibility_pass"]:
        return "SHORT_MEMORY_ORDER_ONLY"
    if not metrics["pfc_coverage_pass"] or not metrics["pfc_mean_tv_pass"]:
        return "PATH_MEMORY_UNCOMPRESSED"
    if not metrics["pfc_advantage_pass"]:
        worst = metrics["worst_baseline"]
        baseline_verdicts = {
            "parser": "LOW_ORDER_HISTORY_SUFFICIENT",
            "last_1": "LOW_ORDER_HISTORY_SUFFICIENT",
            "last_2": "LOW_ORDER_HISTORY_SUFFICIENT",
            "multiset": "LOW_ORDER_HISTORY_SUFFICIENT",
            "discounted": "LOW_ORDER_HISTORY_SUFFICIENT",
            "primitive": "PRIMITIVE_DYNAMICS_SUFFICIENT",
            "causal_knn": "LOCAL_RETRIEVAL_SUFFICIENT",
        }
        return baseline_verdicts.get(worst, "LOW_ORDER_HISTORY_SUFFICIENT")
    if not metrics["pfc_ci_pass"]:
        return "INCONCLUSIVE_PREDICTIVE_ADVANTAGE"
    if not metrics["pfc_coherence_pass"] or not metrics["pfc_order_pass"]:
        return "NONCOHERENT_TRANSPORT"
    return "PFC0_PASS_BOUNDED_PATH_CALCULUS"

# ---------------------------------------------------------------------------
# Atomic save
# ---------------------------------------------------------------------------

def atomic_save(result_dir, data, filename):
    path = Path(result_dir) / filename
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(path)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config_path = Path("experiments/config/path_fiber_v1.json")
    with open(config_path, "rb") as f:
        config_bytes = f.read()
    cfg = json.loads(config_bytes)
    config_hash = validate_lock(config_bytes)
    print(f"Config hash: {config_hash}", flush=True)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    result_dir = Path(cfg["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    cache_path = result_dir / "obs_cache.npz"

    adapter = load_model(cfg)

    roots = make_roots(cfg)
    panels = make_panels(cfg)
    paths = make_paths()
    specs = make_prompt_specs(roots, panels, paths, cfg)

    washed_specs = [s for s in specs if any(s.path_id == p for p in ["p00","pL","pR","pLR"])]
    raw_specs = [s for s in specs if any(s.path_id == p for p in ["p00_raw","pLR_raw"])]

    print(f"\nPopulation: {len(washed_specs)} washed + {len(raw_specs)} raw = {len(specs)} total specs", flush=True)
    print(f"Roots: {len(roots)}, Panels: {len(panels)}, Paths: {len(paths)}", flush=True)

    # --- Collect ---
    print("\n=== Collecting observations ===", flush=True)
    observations = collect_base(adapter, specs, cache_path)
    print(f"Collected {len(observations)} observations ({adapter.call_count} model calls)", flush=True)

    # --- Validate distributions ---
    integrity_issues = []
    for key, obs in observations.items():
        d = obs.distribution
        if not np.all(np.isfinite(d)):
            integrity_issues.append(f"{key}: non-finite")
        if np.any(d < -1e-6):
            integrity_issues.append(f"{key}: negative")
        if abs(d.sum() - 1.0) > 1e-6:
            integrity_issues.append(f"{key}: sum={d.sum()}")

    expected_washed = 9 * 3 * 4 * 2  # 216
    expected_raw = 9 * 3 * 2 * 2     # 108
    expected_total = expected_washed + expected_raw  # 324
    if len(observations) != expected_total:
        integrity_issues.append(f"Expected {expected_total} observations, got {len(observations)}")

    print(f"Integrity issues: {len(integrity_issues)}", flush=True)
    for issue in integrity_issues[:5]:
        print(f"  {issue}", flush=True)

    # --- Accuracy ---
    washed_acc, washed_arm_accs = compute_accuracy(observations, specs, washed=True)
    raw_acc, raw_arm_accs = compute_accuracy(observations, specs, washed=False)
    print(f"\nWashed accuracy: {washed_acc:.4f} (arms: {min(washed_arm_accs.values()):.4f}-{max(washed_arm_accs.values()):.4f})", flush=True)
    print(f"Raw accuracy: {raw_acc:.4f} (arms: {min(raw_arm_accs.values()):.4f}-{max(raw_arm_accs.values()):.4f})", flush=True)

    # --- Replay ---
    print("\n=== State replay validation ===", flush=True)
    replay_results = collect_replay(adapter, observations, roots, panels, cfg)
    max_replay_tv = max(r["tv"] for r in replay_results)
    print(f"Max replay TV: {max_replay_tv:.6f} (gate: {cfg['gates']['replay_max_tv']})", flush=True)

    # --- Defects ---
    print("\n=== Computing defects ===", flush=True)
    raw_defects = compute_defects(observations, roots, panels, washed=False)
    wash_defects = compute_defects(observations, roots, panels, washed=True)

    raw_kappas = [d["kappa"] for d in raw_defects]
    wash_kappas = [d["kappa"] for d in wash_defects]
    print(f"Raw defect: mean={np.mean(raw_kappas):.4f}, min={np.min(raw_kappas):.4f}, max={np.max(raw_kappas):.4f}", flush=True)
    print(f"Wash defect: mean={np.mean(wash_kappas):.4f}, min={np.min(wash_kappas):.4f}, max={np.max(wash_kappas):.4f}", flush=True)

    raw_by_root = defaultdict(list)
    wash_by_root = defaultdict(list)
    wash_by_panel = defaultdict(list)
    for d in raw_defects:
        raw_by_root[d["root_id"]].append(d["kappa"])
    for d in wash_defects:
        wash_by_root[d["root_id"]].append(d["kappa"])
        wash_by_panel[d["panel_id"]].append(d["kappa"])

    raw_boot = root_cluster_bootstrap(raw_by_root, cfg["bootstrap_resamples"], cfg["bootstrap_seed"])
    wash_boot = root_cluster_bootstrap(wash_by_root, cfg["bootstrap_resamples"], cfg["bootstrap_seed"])
    print(f"Raw bootstrap: {raw_boot}", flush=True)
    print(f"Wash bootstrap: {wash_boot}", flush=True)

    wash_panel_means = {pid: np.mean(vals) for pid, vals in wash_by_panel.items()}
    panels_above_threshold = sum(1 for v in wash_panel_means.values() if v >= cfg["gates"]["wash_panel_min"])
    print(f"Wash panel means: {wash_panel_means}, panels >= {cfg['gates']['wash_panel_min']}: {panels_above_threshold}", flush=True)

    # --- Cross-fitted PFC ---
    print("\n=== Cross-fitted path-fiber calculus ===", flush=True)
    fits, pfc_predictions = crossfit_path_calculus(observations, roots, panels, cfg["folds"], cfg)
    pfc_main = [p for p in pfc_predictions if p.method == "pfc"]
    pfc_lr = [p for p in pfc_predictions if p.method == "pfc_lr"]
    pfc_rl = [p for p in pfc_predictions if p.method == "pfc_rl"]

    pfc_coverage = sum(1 for p in pfc_main if p.distribution is not None) / len(pfc_main)
    pfc_tvs = [p.tv for p in pfc_main if p.distribution is not None]
    pfc_mean_tv = np.mean(pfc_tvs) if pfc_tvs else float("inf")
    print(f"PFC coverage: {pfc_coverage:.4f}, mean TV: {pfc_mean_tv:.4f}", flush=True)

    # Coherence
    coherence_tvs = []
    for lr_pred, rl_pred in zip(pfc_lr, pfc_rl):
        if lr_pred.distribution is not None and rl_pred.distribution is not None:
            coherence_tvs.append(tv(lr_pred.distribution, rl_pred.distribution))
    coherence_mean = np.mean(coherence_tvs) if coherence_tvs else float("inf")
    coherence_p90 = np.percentile(coherence_tvs, 90) if coherence_tvs else float("inf")
    print(f"Coherence: mean={coherence_mean:.4f}, p90={coherence_p90:.4f}", flush=True)

    # --- Baselines ---
    print("\n=== Fitting baselines ===", flush=True)
    parser_preds = fit_parser_baseline(observations, roots, panels, cfg["folds"], cfg)
    last1_preds = fit_last_k_baseline(observations, roots, panels, 1, cfg)
    last2_preds = fit_last_k_baseline(observations, roots, panels, 2, cfg)
    multiset_preds = fit_multiset_baseline(observations, roots, panels, cfg)
    discounted_preds, selected_lambdas = fit_discounted_history_baseline(observations, roots, panels, cfg)
    print(f"Discounted: selected lambdas={selected_lambdas}", flush=True)
    primitive_preds = fit_primitive_composition_baseline(observations, roots, panels, cfg)
    knn_preds = predict_causal_knn(observations, roots, panels, cfg)

    all_baselines = {
        "parser": parser_preds,
        "last_1": last1_preds,
        "last_2": last2_preds,
        "multiset": multiset_preds,
        "discounted": discounted_preds,
        "primitive": primitive_preds,
        "causal_knn": knn_preds,
    }

    baseline_mean_tvs = {}
    for name, preds in all_baselines.items():
        tvs = [p.tv for p in preds]
        baseline_mean_tvs[name] = np.mean(tvs)
        print(f"  {name}: mean TV = {baseline_mean_tvs[name]:.4f}", flush=True)

    # --- Advantages ---
    print("\n=== Advantage analysis ===", flush=True)
    advantages = {}
    advantage_cis = {}
    pfc_order_advantages = {}

    for bname, bpreds in all_baselines.items():
        bmap = {(p.root_id, p.panel_id, p.query_entity): p.tv for p in bpreds}
        paired_diffs_by_root = defaultdict(list)
        for p in pfc_main:
            if p.distribution is None:
                continue
            bkey = (p.root_id, p.panel_id, p.query_entity)
            if bkey in bmap:
                diff = bmap[bkey] - p.tv
                paired_diffs_by_root[p.root_id].append(diff)

        advantages[bname] = np.mean([np.mean(vs) for vs in paired_diffs_by_root.values()])
        advantage_cis[bname] = root_cluster_bootstrap(
            paired_diffs_by_root, cfg["bootstrap_resamples"], cfg["bootstrap_seed"]
        )
        print(f"  vs {bname}: advantage={advantages[bname]:.4f}, CI={advantage_cis[bname]}", flush=True)

        for order_name, order_preds in [("lr", pfc_lr), ("rl", pfc_rl)]:
            order_tvs = []
            for op in order_preds:
                if op.distribution is None:
                    continue
                bkey = (op.root_id, op.panel_id, op.query_entity)
                if bkey in bmap:
                    order_tvs.append(bmap[bkey] - op.tv)
            key = f"{bname}_{order_name}"
            pfc_order_advantages[key] = np.mean(order_tvs) if order_tvs else float("-inf")

    # --- Gates ---
    gates = cfg["gates"]
    min_washed_arm = min(washed_arm_accs.values()) if washed_arm_accs else 0
    min_raw_arm = min(raw_arm_accs.values()) if raw_arm_accs else 0

    integrity_pass = (len(integrity_issues) == 0 and
                      max_replay_tv <= gates["replay_max_tv"])
    competence_pass = (washed_acc >= gates["washed_top1_overall"] and
                       min_washed_arm >= gates["washed_top1_arm"] and
                       raw_acc >= gates["raw_top1_overall"] and
                       min_raw_arm >= gates["raw_top1_arm"])
    raw_order_pass = (np.mean(raw_kappas) >= gates["raw_mean_kappa"] and
                      raw_boot["ci95_lower"] > gates["raw_ci95_lower"])
    wash_eligibility_pass = (np.mean(wash_kappas) >= gates["wash_mean_kappa"] and
                             wash_boot["ci95_lower"] > gates["wash_ci95_lower"] and
                             panels_above_threshold >= gates["wash_panel_count"])
    pfc_coverage_pass = pfc_coverage >= gates["pfc_coverage"]
    pfc_mean_tv_pass = pfc_mean_tv <= gates["pfc_mean_tv"]
    pfc_advantage_pass = all(adv >= gates["pfc_advantage"] for adv in advantages.values())
    pfc_ci_pass = all(ci["ci95_lower"] > 0 for ci in advantage_cis.values())
    pfc_order_pass = all(
        pfc_order_advantages.get(f"{bn}_lr", float("-inf")) >= gates["pfc_order_advantage"] and
        pfc_order_advantages.get(f"{bn}_rl", float("-inf")) >= gates["pfc_order_advantage"]
        for bn in all_baselines
    )
    pfc_coherence_pass = (coherence_mean <= gates["pfc_coherence_mean"] and
                          coherence_p90 <= gates["pfc_coherence_p90"])

    worst_baseline = min(advantages, key=lambda k: advantages[k]) if advantages else None

    metrics = {
        "integrity_pass": integrity_pass,
        "competence_pass": competence_pass,
        "raw_order_pass": raw_order_pass,
        "wash_eligibility_pass": wash_eligibility_pass,
        "pfc_coverage_pass": pfc_coverage_pass,
        "pfc_mean_tv_pass": pfc_mean_tv_pass,
        "pfc_advantage_pass": pfc_advantage_pass,
        "pfc_ci_pass": pfc_ci_pass,
        "pfc_order_pass": pfc_order_pass,
        "pfc_coherence_pass": pfc_coherence_pass,
        "worst_baseline": worst_baseline,
    }

    verdict = adjudicate(metrics, gates)
    print(f"\n=== VERDICT: {verdict} ===", flush=True)

    # --- Licensed sentences ---
    if verdict == "PFC0_PASS_BOUNDED_PATH_CALCULUS":
        licensed = ("On this fixed Finch-3B entity-location world, a two-generator "
                    "response-law transport predicted an unseen composed path across "
                    "held-out initial states better than endpoint, recency, "
                    "primitive-composition, and causal-retrieval controls.")
    else:
        licensed = f"PFC-0 verdict is {verdict}; no affirmative claim licensed."

    never_say = [
        "intrinsic latent fiber bundle",
        "intrinsic curvature",
        "unique model-native base",
        "general law of language-model latent space",
    ]

    # --- Result ---
    result = {
        "lock": {
            "config_hash": config_hash,
            "config_path": str(config_path),
        },
        "manifest": {
            "model_id": cfg["model_id"],
            "model_revision": cfg["model_revision"],
            "answer_ids": adapter.answer_ids,
            "total_calls": adapter.call_count,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "population": {
            "n_roots": len(roots),
            "n_panels": len(panels),
            "n_paths": len(paths),
            "n_specs": len(specs),
            "n_observations": len(observations),
        },
        "capability": {
            "washed_accuracy": washed_acc,
            "washed_arm_accuracies": washed_arm_accs,
            "raw_accuracy": raw_acc,
            "raw_arm_accuracies": raw_arm_accs,
        },
        "raw_defect": {
            "mean": float(np.mean(raw_kappas)),
            "min": float(np.min(raw_kappas)),
            "max": float(np.max(raw_kappas)),
            "bootstrap": raw_boot,
            "by_cell": raw_defects,
        },
        "washed_defect": {
            "mean": float(np.mean(wash_kappas)),
            "min": float(np.min(wash_kappas)),
            "max": float(np.max(wash_kappas)),
            "bootstrap": wash_boot,
            "panel_means": {k: float(v) for k, v in wash_panel_means.items()},
            "panels_above_threshold": panels_above_threshold,
            "by_cell": wash_defects,
        },
        "replay": {
            "max_tv": max_replay_tv,
            "results": replay_results,
        },
        "crossfit_fits": {k: v for k, v in fits.items()},
        "predictions": {
            "pfc_coverage": pfc_coverage,
            "pfc_mean_tv": pfc_mean_tv,
            "pfc_tvs": pfc_tvs,
            "coherence_mean": coherence_mean,
            "coherence_p90": coherence_p90,
            "coherence_tvs": [float(c) for c in coherence_tvs],
        },
        "baselines": {
            name: {
                "mean_tv": float(baseline_mean_tvs[name]),
                "tvs": [float(p.tv) for p in preds],
            }
            for name, preds in all_baselines.items()
        },
        "discounted_lambdas_selected": selected_lambdas,
        "advantages": {k: float(v) for k, v in advantages.items()},
        "advantage_cis": advantage_cis,
        "order_advantages": {k: float(v) for k, v in pfc_order_advantages.items()},
        "clustered_intervals": {
            "raw": raw_boot,
            "washed": wash_boot,
            "advantages": advantage_cis,
        },
        "gates": {
            "integrity": integrity_pass,
            "competence": competence_pass,
            "raw_order": raw_order_pass,
            "wash_eligibility": wash_eligibility_pass,
            "pfc_coverage": pfc_coverage_pass,
            "pfc_mean_tv": pfc_mean_tv_pass,
            "pfc_advantage": pfc_advantage_pass,
            "pfc_ci": pfc_ci_pass,
            "pfc_order": pfc_order_pass,
            "pfc_coherence": pfc_coherence_pass,
        },
        "verdict": verdict,
        "licensed_sentence": licensed,
        "never_say": never_say,
        "measurement_to_artifact_ratio": "1:1",
    }

    atomic_save(result_dir, result, "result.json")
    print(f"\nResult saved to {result_dir / 'result.json'}", flush=True)
    print(f"Total model calls: {adapter.call_count}", flush=True)

    # Save detailed predictions
    pred_records = []
    for p in pfc_main + list(pfc_lr) + list(pfc_rl):
        pred_records.append({
            "fold": p.fold, "root_id": p.root_id, "panel_id": p.panel_id,
            "query_entity": p.query_entity, "method": p.method,
            "distribution": p.distribution.tolist() if p.distribution is not None else None,
            "target": p.target.tolist() if p.target is not None else None,
            "tv": p.tv,
        })
    for name, preds in all_baselines.items():
        for p in preds:
            pred_records.append({
                "fold": p.fold, "root_id": p.root_id, "panel_id": p.panel_id,
                "query_entity": p.query_entity, "method": p.method,
                "distribution": p.distribution.tolist() if p.distribution is not None else None,
                "target": p.target.tolist() if p.target is not None else None,
                "tv": p.tv,
            })
    atomic_save(result_dir, pred_records, "predictions.jsonl")
    atomic_save(result_dir, {k: v for k, v in fits.items()}, "fits.json")

    return verdict

if __name__ == "__main__":
    verdict = main()
    sys.exit(0 if verdict == "PFC0_PASS_BOUNDED_PATH_CALCULUS" else 1)
