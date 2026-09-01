"""
FBA-0: Fiber-Bottleneck Architecture Constructive Program
Pre-registered experiment per Codex direction rounds 11-15.

Conditional claim (bounded, requires GO verdict): In this synthetic Z_8 x Z_4
POMDP, does the 16/16 independently updated recurrent architecture improve
held-out response-class accuracy over registered controls and support matched
branch-interchange behavior above the historyless null?

Distance from claim: 1 (engineered-factorization control for the native math program).
The central artifact is intervenable structure in a real model's latent space
(per AGENTS.md). This is a one-round toy-world control providing empirical
grounding for the definitions in theory/FBA_BRIDGE.md.

POMDP Contract (locked):
- State space: n_locations x n_states = 8 x 4 = 32 cells
- Action space: n_move_types x n_transform_types = 4 x 4 = 16 opaque tokens
  Moves: cyclic shifts 0,1,2,3 on Z/8.  Transforms: cyclic shifts 0,1,2,3 on Z/4.
- Emission: C[true, obs] = 0.85 if true==obs, else 0.15/(n-1). Independent per dim.
- T=3 actions, T+1=4 observations. Target: true terminal CELL (32 classes).
- Historyless oracle ~73%. Recurrent oracle ~96%.

Split: response-law equivalence classes (32 classes, 21/3/8 train/val/test).

Six-way comparison:
1. FBA (16/16 symmetric independent-update bottleneck)
2. Flat GRU (~40K params)
3. Flat GRU matched (~33K params, capacity-matched to FBA)
4. Asymmetric split (24/8 independent-update, non-conjugate width ablation)
5. Modular (4 groups of 8 dims)
6. Flat bottleneck (32 dims, single update path)

Kill gates (joint predicate per seed, majority of seeds):
K4: Training accuracy >= 90% for ALL models
K6: FBA > best_flat on test by >= 20pp
K7a: FBA > asymmetric_split on test by >= 15pp
K7b: Branch interchange beats historyless null with wrong-channel controls
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import itertools
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Config:
    n_locations: int = 8
    n_states: int = 4
    n_move_types: int = 4
    n_transform_types: int = 4
    episode_length: int = 3
    loc_noise: float = 0.15
    state_noise: float = 0.15
    holdout_frac: float = 0.25
    val_frac: float = 0.1
    d_model: int = 64
    d_place: int = 16
    d_fiber: int = 16
    n_epochs: int = 2000
    batch_size: int = 256
    lr: float = 1e-3
    lr_min: float = 1e-5
    seeds: list = field(default_factory=lambda: [42, 137, 2026])


class GridWorld:
    def __init__(self, cfg: Config, seed: int = 42):
        self.cfg = cfg
        self.n_cells = cfg.n_locations * cfg.n_states
        self.n_actions = cfg.n_move_types * cfg.n_transform_types
        self.rng = np.random.RandomState(seed)

        self.moves = [[(loc + i) % cfg.n_locations
                       for loc in range(cfg.n_locations)]
                      for i in range(cfg.n_move_types)]

        self.transforms = [[(s + i) % cfg.n_states
                            for s in range(cfg.n_states)]
                           for i in range(cfg.n_transform_types)]

        tokens = list(range(self.n_actions))
        self.rng.shuffle(tokens)
        self.token_to_factors = {}
        idx = 0
        for mi in range(cfg.n_move_types):
            for ti in range(cfg.n_transform_types):
                self.token_to_factors[tokens[idx]] = (mi, ti)
                idx += 1

        self.loc_confusion = self._make_confusion(cfg.n_locations, cfg.loc_noise)
        self.state_confusion = self._make_confusion(cfg.n_states, cfg.state_noise)

        self._build_response_classes()

    @staticmethod
    def _make_confusion(n, noise):
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                C[i, j] = (1.0 - noise) if i == j else noise / (n - 1)
        return C

    def apply_action(self, cell, tok):
        loc = cell // self.cfg.n_states
        state = cell % self.cfg.n_states
        mi, ti = self.token_to_factors[tok]
        return self.moves[mi][loc] * self.cfg.n_states + self.transforms[ti][state]

    def apply_sequence(self, cell, actions):
        for tok in actions:
            cell = self.apply_action(cell, tok)
        return cell

    def observe(self, cell, rng):
        loc = cell // self.cfg.n_states
        state = cell % self.cfg.n_states
        noisy_loc = rng.choice(self.cfg.n_locations, p=self.loc_confusion[loc])
        noisy_state = rng.choice(self.cfg.n_states, p=self.state_confusion[state])
        return (int(noisy_loc), int(noisy_state))

    def _build_response_classes(self):
        all_seqs = list(itertools.product(range(self.n_actions),
                                          repeat=self.cfg.episode_length))
        trans_to_seqs = defaultdict(list)
        self.seq_to_class_id = {}

        for seq in all_seqs:
            transitions = tuple(self.apply_sequence(c, list(seq))
                                for c in range(self.n_cells))
            trans_to_seqs[transitions].append(seq)

        classes = sorted(trans_to_seqs.keys(), key=str)
        self.rng.shuffle(classes)

        n_test = max(1, int(len(classes) * self.cfg.holdout_frac))
        n_val = max(1, int(len(classes) * self.cfg.val_frac))

        self.test_seqs = []
        self.val_seqs = []
        self.train_seqs = []
        self.class_manifests = []

        for i, cls_trans in enumerate(classes):
            seqs = trans_to_seqs[cls_trans]
            for seq in seqs:
                self.seq_to_class_id[seq] = i

            manifest = {"id": i, "transition": list(cls_trans),
                        "n_seqs": len(seqs),
                        "members": [list(s) for s in seqs]}

            if i < n_test:
                self.test_seqs.extend(seqs)
                manifest["split"] = "test"
            elif i < n_test + n_val:
                self.val_seqs.extend(seqs)
                manifest["split"] = "val"
            else:
                self.train_seqs.extend(seqs)
                manifest["split"] = "train"

            self.class_manifests.append(manifest)

        self.n_response_classes = len(classes)
        self.class_counts = {"train": len(classes) - n_test - n_val,
                             "val": n_val, "test": n_test}

        train_set = set(self.train_seqs)
        val_set = set(self.val_seqs)
        test_set = set(self.test_seqs)
        assert not (train_set & val_set), "Train/val overlap!"
        assert not (train_set & test_set), "Train/test overlap!"
        assert not (val_set & test_set), "Val/test overlap!"

    def generate_episodes(self, batch_size, split, rng):
        seqs = {"train": self.train_seqs, "val": self.val_seqs,
                "test": self.test_seqs}[split]
        obs_list, act_list, tgt_list = [], [], []

        for _ in range(batch_size):
            seq = seqs[rng.randint(len(seqs))]
            cell = rng.randint(self.n_cells)
            observations = [self.observe(cell, rng)]
            for tok in seq:
                cell = self.apply_action(cell, tok)
                observations.append(self.observe(cell, rng))
            obs_list.append(observations)
            act_list.append(list(seq))
            tgt_list.append(cell)

        return (torch.tensor(obs_list, dtype=torch.long),
                torch.tensor(act_list, dtype=torch.long),
                torch.tensor(tgt_list, dtype=torch.long))


def historyless_oracle(world, n_samples=10000, seed=99):
    rng = np.random.RandomState(seed)
    correct = 0
    for _ in range(n_samples):
        seq = world.train_seqs[rng.randint(len(world.train_seqs))]
        cell = rng.randint(world.n_cells)
        for tok in seq:
            cell = world.apply_action(cell, tok)
        obs_loc, obs_state = world.observe(cell, rng)
        guess = obs_loc * world.cfg.n_states + obs_state
        correct += (guess == cell)
    return correct / n_samples


def recurrent_oracle(world, n_samples=10000, seed=99):
    cfg = world.cfg
    rng = np.random.RandomState(seed)
    correct = 0

    for _ in range(n_samples):
        seq = world.train_seqs[rng.randint(len(world.train_seqs))]
        start_cell = rng.randint(world.n_cells)

        belief = np.ones(world.n_cells) / world.n_cells
        cell = start_cell
        obs = world.observe(cell, rng)
        for c in range(world.n_cells):
            c_loc = c // cfg.n_states
            c_state = c % cfg.n_states
            belief[c] *= world.loc_confusion[c_loc, obs[0]]
            belief[c] *= world.state_confusion[c_state, obs[1]]
        belief /= belief.sum()

        for tok in seq:
            new_belief = np.zeros(world.n_cells)
            for c in range(world.n_cells):
                new_c = world.apply_action(c, tok)
                new_belief[new_c] += belief[c]
            belief = new_belief
            cell = world.apply_action(cell, tok)
            obs = world.observe(cell, rng)
            for c in range(world.n_cells):
                c_loc = c // cfg.n_states
                c_state = c % cfg.n_states
                belief[c] *= world.loc_confusion[c_loc, obs[0]]
                belief[c] *= world.state_confusion[c_state, obs[1]]
            s = belief.sum()
            if s > 0:
                belief /= s

        correct += (belief.argmax() == cell)

    return correct / n_samples


# ============================================================
# ARCHITECTURES
# ============================================================

class RecurrentFBA(nn.Module):
    """Two branches (16/16) updated independently."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        d_p, d_f = cfg.d_place, cfg.d_fiber

        self.loc_embed = nn.Embedding(cfg.n_locations, d // 2)
        self.state_embed = nn.Embedding(cfg.n_states, d // 2)
        self.obs_enc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

        self.act_embed = nn.Embedding(cfg.n_move_types * cfg.n_transform_types, d)
        self.act_enc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

        self.init_place = nn.Linear(d, d_p)
        self.init_fiber = nn.Linear(d, d_f)

        self.obs_up_p = nn.Sequential(nn.Linear(d + d_p, d_p * 2), nn.ReLU(),
                                      nn.Linear(d_p * 2, d_p))
        self.obs_up_f = nn.Sequential(nn.Linear(d + d_f, d_f * 2), nn.ReLU(),
                                      nn.Linear(d_f * 2, d_f))

        self.act_p = nn.Sequential(nn.Linear(d + d_p, d_p * 2), nn.ReLU(),
                                   nn.Linear(d_p * 2, d_p))
        self.act_f = nn.Sequential(nn.Linear(d + d_f, d_f * 2), nn.ReLU(),
                                   nn.Linear(d_f * 2, d_f))

        self.head = nn.Linear(d_p + d_f, cfg.n_locations * cfg.n_states)

    def _encode_obs(self, obs_step):
        return self.obs_enc(torch.cat([self.loc_embed(obs_step[:, 0]),
                                       self.state_embed(obs_step[:, 1])], dim=-1))

    def forward(self, obs, actions):
        _, _, logits = self.forward_with_branches(obs, actions)
        return logits

    def forward_with_branches(self, obs, actions):
        B, Tp1, _ = obs.shape
        T = Tp1 - 1
        o0 = self._encode_obs(obs[:, 0])
        place = self.init_place(o0)
        fiber = self.init_fiber(o0)

        for t in range(T):
            a = self.act_enc(self.act_embed(actions[:, t]))
            place = self.act_p(torch.cat([a, place], dim=-1))
            fiber = self.act_f(torch.cat([a, fiber], dim=-1))
            ot = self._encode_obs(obs[:, t + 1])
            place = self.obs_up_p(torch.cat([ot, place], dim=-1))
            fiber = self.obs_up_f(torch.cat([ot, fiber], dim=-1))

        return place, fiber, self.head(torch.cat([place, fiber], dim=-1))


class RecurrentFlat(nn.Module):
    """Flat GRU."""

    def __init__(self, cfg, d_hidden=None):
        super().__init__()
        d = cfg.d_model
        d_h = d_hidden if d_hidden else cfg.d_place + cfg.d_fiber

        self.loc_embed = nn.Embedding(cfg.n_locations, d // 2)
        self.state_embed = nn.Embedding(cfg.n_states, d // 2)
        self.obs_enc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

        self.act_embed = nn.Embedding(cfg.n_move_types * cfg.n_transform_types, d)
        self.act_enc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

        self.init_h = nn.Linear(d, d_h)
        self.act_gru = nn.GRUCell(d, d_h)
        self.obs_gru = nn.GRUCell(d, d_h)

        self.head = nn.Linear(d_h, cfg.n_locations * cfg.n_states)

    def _encode_obs(self, obs_step):
        return self.obs_enc(torch.cat([self.loc_embed(obs_step[:, 0]),
                                       self.state_embed(obs_step[:, 1])], dim=-1))

    def forward(self, obs, actions):
        B, Tp1, _ = obs.shape
        T = Tp1 - 1
        h = self.init_h(self._encode_obs(obs[:, 0]))
        for t in range(T):
            a = self.act_enc(self.act_embed(actions[:, t]))
            h = self.act_gru(a, h)
            h = self.obs_gru(self._encode_obs(obs[:, t + 1]), h)
        return self.head(h)


class RecurrentAsymmetricSplit(nn.Module):
    """Asymmetric two-branch bottleneck (24/8 split).
    Non-conjugate to FBA's 16/16 by dimension mismatch.
    Same independent update structure, same total dims (32)."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        d_a, d_b = 24, 8

        self.loc_embed = nn.Embedding(cfg.n_locations, d // 2)
        self.state_embed = nn.Embedding(cfg.n_states, d // 2)
        self.obs_enc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

        self.act_embed = nn.Embedding(cfg.n_move_types * cfg.n_transform_types, d)
        self.act_enc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

        self.init_a = nn.Linear(d, d_a)
        self.init_b = nn.Linear(d, d_b)

        self.obs_up_a = nn.Sequential(nn.Linear(d + d_a, d_a * 2), nn.ReLU(),
                                      nn.Linear(d_a * 2, d_a))
        self.obs_up_b = nn.Sequential(nn.Linear(d + d_b, d_b * 2), nn.ReLU(),
                                      nn.Linear(d_b * 2, d_b))

        self.act_a = nn.Sequential(nn.Linear(d + d_a, d_a * 2), nn.ReLU(),
                                   nn.Linear(d_a * 2, d_a))
        self.act_b = nn.Sequential(nn.Linear(d + d_b, d_b * 2), nn.ReLU(),
                                   nn.Linear(d_b * 2, d_b))

        self.head = nn.Linear(d_a + d_b, cfg.n_locations * cfg.n_states)

    def _encode_obs(self, obs_step):
        return self.obs_enc(torch.cat([self.loc_embed(obs_step[:, 0]),
                                       self.state_embed(obs_step[:, 1])], dim=-1))

    def forward(self, obs, actions):
        B, Tp1, _ = obs.shape
        T = Tp1 - 1
        o0 = self._encode_obs(obs[:, 0])
        branch_a = self.init_a(o0)
        branch_b = self.init_b(o0)

        for t in range(T):
            a = self.act_enc(self.act_embed(actions[:, t]))
            branch_a = self.act_a(torch.cat([a, branch_a], dim=-1))
            branch_b = self.act_b(torch.cat([a, branch_b], dim=-1))
            ot = self._encode_obs(obs[:, t + 1])
            branch_a = self.obs_up_a(torch.cat([ot, branch_a], dim=-1))
            branch_b = self.obs_up_b(torch.cat([ot, branch_b], dim=-1))

        return self.head(torch.cat([branch_a, branch_b], dim=-1))

    def forward_with_branches(self, obs, actions):
        B, Tp1, _ = obs.shape
        T = Tp1 - 1
        o0 = self._encode_obs(obs[:, 0])
        branch_a = self.init_a(o0)
        branch_b = self.init_b(o0)

        for t in range(T):
            a = self.act_enc(self.act_embed(actions[:, t]))
            branch_a = self.act_a(torch.cat([a, branch_a], dim=-1))
            branch_b = self.act_b(torch.cat([a, branch_b], dim=-1))
            ot = self._encode_obs(obs[:, t + 1])
            branch_a = self.obs_up_a(torch.cat([ot, branch_a], dim=-1))
            branch_b = self.obs_up_b(torch.cat([ot, branch_b], dim=-1))

        return (branch_a, branch_b,
                self.head(torch.cat([branch_a, branch_b], dim=-1)))


class RecurrentModular(nn.Module):
    """4 groups of 8 dims each, same total (32)."""

    def __init__(self, cfg, n_groups=4):
        super().__init__()
        d = cfg.d_model
        d_total = cfg.d_place + cfg.d_fiber
        self.n_groups = n_groups
        self.d_g = d_total // n_groups

        self.loc_embed = nn.Embedding(cfg.n_locations, d // 2)
        self.state_embed = nn.Embedding(cfg.n_states, d // 2)
        self.obs_enc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

        self.act_embed = nn.Embedding(cfg.n_move_types * cfg.n_transform_types, d)
        self.act_enc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

        self.init_state = nn.Linear(d, d_total)
        self.obs_up = nn.Sequential(nn.Linear(d + d_total, d_total * 2), nn.ReLU(),
                                    nn.Linear(d_total * 2, d_total))

        self.group_fx = nn.ModuleList([
            nn.Sequential(nn.Linear(d + self.d_g, self.d_g * 2), nn.ReLU(),
                          nn.Linear(self.d_g * 2, self.d_g))
            for _ in range(n_groups)])

        self.head = nn.Linear(d_total, cfg.n_locations * cfg.n_states)

    def _encode_obs(self, obs_step):
        return self.obs_enc(torch.cat([self.loc_embed(obs_step[:, 0]),
                                       self.state_embed(obs_step[:, 1])], dim=-1))

    def forward(self, obs, actions):
        B, Tp1, _ = obs.shape
        T = Tp1 - 1
        state = self.init_state(self._encode_obs(obs[:, 0]))

        for t in range(T):
            a = self.act_enc(self.act_embed(actions[:, t]))
            groups = state.chunk(self.n_groups, dim=-1)
            state = torch.cat([fx(torch.cat([a, g], dim=-1))
                               for fx, g in zip(self.group_fx, groups)], dim=-1)
            state = self.obs_up(torch.cat([self._encode_obs(obs[:, t + 1]),
                                            state], dim=-1))
        return self.head(state)


class RecurrentFlatBN(nn.Module):
    """Flat bottleneck: 32 dims, single update path."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        d_bn = cfg.d_place + cfg.d_fiber

        self.loc_embed = nn.Embedding(cfg.n_locations, d // 2)
        self.state_embed = nn.Embedding(cfg.n_states, d // 2)
        self.obs_enc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

        self.act_embed = nn.Embedding(cfg.n_move_types * cfg.n_transform_types, d)
        self.act_enc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

        self.init_state = nn.Linear(d, d_bn)
        self.act_fx = nn.Sequential(nn.Linear(d + d_bn, d_bn * 2), nn.ReLU(),
                                    nn.Linear(d_bn * 2, d_bn))
        self.obs_up = nn.Sequential(nn.Linear(d + d_bn, d_bn * 2), nn.ReLU(),
                                    nn.Linear(d_bn * 2, d_bn))

        self.head = nn.Linear(d_bn, cfg.n_locations * cfg.n_states)

    def _encode_obs(self, obs_step):
        return self.obs_enc(torch.cat([self.loc_embed(obs_step[:, 0]),
                                       self.state_embed(obs_step[:, 1])], dim=-1))

    def forward(self, obs, actions):
        B, Tp1, _ = obs.shape
        T = Tp1 - 1
        state = self.init_state(self._encode_obs(obs[:, 0]))

        for t in range(T):
            a = self.act_enc(self.act_embed(actions[:, t]))
            state = self.act_fx(torch.cat([a, state], dim=-1))
            state = self.obs_up(torch.cat([self._encode_obs(obs[:, t + 1]),
                                            state], dim=-1))
        return self.head(state)


# ============================================================
# TRAINING & EVALUATION
# ============================================================

def train_model(model, world, cfg, seed):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs, eta_min=cfg.lr_min)
    criterion = nn.CrossEntropyLoss()
    best_val, best_state = -1, None

    for epoch in range(cfg.n_epochs):
        model.train()
        obs, acts, tgts = world.generate_episodes(cfg.batch_size, "train", rng)
        loss = criterion(model(obs, acts), tgts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            val = evaluate(model, world, "val", seed + 1000)
            if val > best_val:
                best_val = val
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 200 == 0:
                print(f"  ep {epoch+1}: loss={loss.item():.4f} val={val:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


def evaluate(model, world, split, seed, n_eval=2048):
    model.eval()
    rng = np.random.RandomState(seed)
    with torch.no_grad():
        obs, acts, tgts = world.generate_episodes(n_eval, split, rng)
        return (model(obs, acts).argmax(dim=-1) == tgts).float().mean().item()


def evaluate_detailed(model, world, split, seed, n_eval=2048):
    """Return per-episode predictions with response class IDs."""
    model.eval()
    rng = np.random.RandomState(seed)
    seqs_pool = {"train": world.train_seqs, "val": world.val_seqs,
                 "test": world.test_seqs}[split]

    obs_list, act_list, tgt_list, cls_list = [], [], [], []
    for _ in range(n_eval):
        seq = seqs_pool[rng.randint(len(seqs_pool))]
        cell = rng.randint(world.n_cells)
        observations = [world.observe(cell, rng)]
        for tok in seq:
            cell = world.apply_action(cell, tok)
            observations.append(world.observe(cell, rng))
        obs_list.append(observations)
        act_list.append(list(seq))
        tgt_list.append(cell)
        cls_list.append(world.seq_to_class_id.get(seq, -1))

    obs_t = torch.tensor(obs_list, dtype=torch.long)
    acts_t = torch.tensor(act_list, dtype=torch.long)
    tgts_t = torch.tensor(tgt_list, dtype=torch.long)

    with torch.no_grad():
        preds = model(obs_t, acts_t).argmax(dim=-1)

    episodes = []
    for i in range(n_eval):
        episodes.append({
            "true_cell": tgt_list[i],
            "pred_cell": preds[i].item(),
            "correct": preds[i].item() == tgt_list[i],
            "class_id": cls_list[i],
        })

    acc = sum(e["correct"] for e in episodes) / len(episodes)
    return acc, episodes


def evaluate_k7b(model, world, seed, n_episodes=1024):
    """Branch interchange: disjoint episode pairs for valid bootstrap."""
    model.eval()
    rng = np.random.RandomState(seed)
    n_s = world.cfg.n_states
    hl_cross_null = (1 - world.cfg.loc_noise) * (1 - world.cfg.state_noise)
    hl_pres_null = 1 - max(world.cfg.loc_noise, world.cfg.state_noise)

    obs, acts, tgts = world.generate_episodes(n_episodes, "test", rng)

    with torch.no_grad():
        branch_a, branch_b, logits = model.forward_with_branches(obs, acts)

    self_logits = model.head(torch.cat([branch_a, branch_b], dim=-1))
    self_exact = bool(torch.allclose(logits, self_logits, atol=1e-5))

    locs = (tgts // n_s).numpy()
    states = (tgts % n_s).numpy()
    cls_ids = np.array([world.seq_to_class_id.get(tuple(acts[k].tolist()), -1)
                        for k in range(n_episodes)])

    indices = rng.permutation(n_episodes)
    all_pairs = [(int(indices[2*k]), int(indices[2*k+1]))
                 for k in range(n_episodes // 2)]

    matched_pairs, wc_loc_pairs, wc_state_pairs = [], [], []
    for i, j in all_pairs:
        if locs[i] != locs[j] and states[i] != states[j]:
            matched_pairs.append((i, j))
        elif locs[i] == locs[j] and states[i] != states[j]:
            wc_loc_pairs.append((i, j))
        elif states[i] == states[j] and locs[i] != locs[j]:
            wc_state_pairs.append((i, j))

    bonf_alpha = 0.05 / 2

    best_result = None
    best_cross = -1

    for orient in ["A=place,B=fiber", "A=fiber,B=place"]:
        a_is_place = orient.startswith("A=place")
        per_pair_cross = []
        per_pair_place = []
        per_pair_fiber = []
        per_pair_cls = []

        for i, j in matched_pairs:
            expected = int(locs[i]) * n_s + int(states[j])
            with torch.no_grad():
                if a_is_place:
                    hybrid = model.head(torch.cat(
                        [branch_a[i:i+1], branch_b[j:j+1]], dim=-1))
                else:
                    hybrid = model.head(torch.cat(
                        [branch_a[j:j+1], branch_b[i:i+1]], dim=-1))
                pred = hybrid.argmax(dim=-1).item()
            per_pair_cross.append(int(pred == expected))
            per_pair_place.append(int(pred // n_s == locs[i]))
            per_pair_fiber.append(int(pred % n_s == states[j]))
            per_pair_cls.append(cls_ids[i])

        wc_loc_scores = []
        for i, j in wc_loc_pairs:
            with torch.no_grad():
                if a_is_place:
                    pred = model.head(torch.cat(
                        [branch_a[i:i+1], branch_b[j:j+1]], dim=-1)
                    ).argmax(dim=-1).item()
                else:
                    pred = model.head(torch.cat(
                        [branch_a[j:j+1], branch_b[i:i+1]], dim=-1)
                    ).argmax(dim=-1).item()
            wc_loc_scores.append(int(pred // n_s == locs[i]))

        wc_state_scores = []
        for i, j in wc_state_pairs:
            with torch.no_grad():
                if a_is_place:
                    pred = model.head(torch.cat(
                        [branch_a[i:i+1], branch_b[j:j+1]], dim=-1)
                    ).argmax(dim=-1).item()
                else:
                    pred = model.head(torch.cat(
                        [branch_a[j:j+1], branch_b[i:i+1]], dim=-1)
                    ).argmax(dim=-1).item()
            wc_state_scores.append(int(pred % n_s == states[i]))

        if per_pair_cross:
            cross_acc = np.mean(per_pair_cross)
            if cross_acc > best_cross:
                best_cross = cross_acc
                cross_m, cross_lo, cross_hi = bootstrap_ci(
                    per_pair_cross, alpha=bonf_alpha)
                place_m, place_lo, place_hi = bootstrap_ci(
                    per_pair_place, alpha=bonf_alpha)
                fiber_m, fiber_lo, fiber_hi = bootstrap_ci(
                    per_pair_fiber, alpha=bonf_alpha)

                wc_loc_m, wc_loc_lo, wc_loc_hi = bootstrap_ci(
                    wc_loc_scores, alpha=bonf_alpha) if wc_loc_scores else (0, 0, 0)
                wc_state_m, wc_state_lo, wc_state_hi = bootstrap_ci(
                    wc_state_scores, alpha=bonf_alpha) if wc_state_scores else (0, 0, 0)

                by_cls = defaultdict(list)
                for k, c in enumerate(per_pair_cls):
                    by_cls[c].append(per_pair_cross[k])
                cls_means = [np.mean(v) for v in by_cls.values()]
                cls_m, cls_lo, cls_hi = bootstrap_ci(cls_means, alpha=bonf_alpha)

                best_result = {
                    "orientation": orient,
                    "cross_acc": cross_m,
                    "cross_ci": [cross_lo, cross_hi],
                    "place_preserved": place_m,
                    "place_ci": [place_lo, place_hi],
                    "fiber_preserved": fiber_m,
                    "fiber_ci": [fiber_lo, fiber_hi],
                    "n_matched_pairs": len(matched_pairs),
                    "wc_same_loc": wc_loc_m,
                    "wc_loc_ci": [wc_loc_lo, wc_loc_hi],
                    "wc_same_state": wc_state_m,
                    "wc_state_ci": [wc_state_lo, wc_state_hi],
                    "n_wc_loc_pairs": len(wc_loc_pairs),
                    "n_wc_state_pairs": len(wc_state_pairs),
                    "self_exact_match": self_exact,
                    "hl_cross_null": hl_cross_null,
                    "hl_pres_null": hl_pres_null,
                    "bonferroni_alpha": bonf_alpha,
                    "cross_by_class": {"mean": cls_m, "ci_lo": cls_lo, "ci_hi": cls_hi},
                }

    return best_result if best_result else {
        "orientation": None, "cross_acc": 0, "cross_ci": [0, 0],
        "place_preserved": 0, "place_ci": [0, 0],
        "fiber_preserved": 0, "fiber_ci": [0, 0],
        "n_matched_pairs": 0, "self_exact_match": self_exact,
        "hl_cross_null": hl_cross_null, "hl_pres_null": hl_pres_null,
    }


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_boot):
        sample = arr[rng.randint(len(arr), size=len(arr))]
        means.append(sample.mean())
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(np.mean(arr)), float(lo), float(hi)


def per_class_accuracy(episodes):
    by_class = defaultdict(list)
    for ep in episodes:
        by_class[ep["class_id"]].append(ep["correct"])
    results = {}
    for cid, corrects in sorted(by_class.items()):
        mean, lo, hi = bootstrap_ci(corrects)
        results[cid] = {"acc": mean, "ci_lo": lo, "ci_hi": hi, "n": len(corrects)}
    return results


def paired_class_effect(fba_episodes, control_episodes):
    """Paired FBA-minus-control effect, clustered by response class."""
    fba_by = defaultdict(list)
    ctrl_by = defaultdict(list)
    for ep in fba_episodes:
        fba_by[ep["class_id"]].append(int(ep["correct"]))
    for ep in control_episodes:
        ctrl_by[ep["class_id"]].append(int(ep["correct"]))

    common = sorted(set(fba_by.keys()) & set(ctrl_by.keys()))
    if not common:
        return {"mean_effect": 0, "ci_lo": 0, "ci_hi": 0, "n_classes": 0}

    effects = [np.mean(fba_by[c]) - np.mean(ctrl_by[c]) for c in common]
    mean_eff, lo, hi = bootstrap_ci(effects)
    return {"mean_effect": mean_eff, "ci_lo": lo, "ci_hi": hi,
            "n_classes": len(effects), "per_class": [float(e) for e in effects]}


# ============================================================
# MAIN
# ============================================================

def run_seed(seed, cfg):
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    world = GridWorld(cfg, seed=seed)
    print(f"World: {cfg.n_locations}x{cfg.n_states}={world.n_cells} cells, "
          f"{world.n_actions} actions")
    print(f"Response classes: {world.n_response_classes}")
    print(f"Split: {world.class_counts}")
    print(f"Sequences: train={len(world.train_seqs)} val={len(world.val_seqs)} "
          f"test={len(world.test_seqs)}")

    hl = historyless_oracle(world, 5000, seed)
    rec = recurrent_oracle(world, 5000, seed)
    print(f"Oracle ceilings: historyless={hl:.3f} recurrent={rec:.3f}")

    oracle_valid = (hl < 0.85) and (rec >= 0.95)
    if not oracle_valid:
        print(f"FATAL: Oracle validity failed")

    models = {
        "fba": RecurrentFBA(cfg),
        "flat": RecurrentFlat(cfg),
        "flat_matched": RecurrentFlat(cfg, d_hidden=24),
        "asym_split": RecurrentAsymmetricSplit(cfg),
        "modular": RecurrentModular(cfg),
        "flat_bn": RecurrentFlatBN(cfg),
    }

    results = {"seed": seed,
               "oracle": {"historyless": hl, "recurrent": rec,
                           "valid": oracle_valid},
               "models": {}, "k7b": {}, "paired_effects": {}}

    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n--- {name} ({n_params:,} params) ---")
        torch.manual_seed(seed)
        train_model(model, world, cfg, seed)

        tr, tr_eps = evaluate_detailed(model, world, "train", seed + 2000)
        va = evaluate(model, world, "val", seed + 3000)
        te, te_eps = evaluate_detailed(model, world, "test", seed + 4000)
        print(f"  train={tr:.3f} val={va:.3f} test={te:.3f}")

        results["models"][name] = {
            "params": n_params, "train": tr, "val": va, "test": te,
            "train_per_class": per_class_accuracy(tr_eps),
            "test_per_class": per_class_accuracy(te_eps),
            "test_episodes": te_eps,
        }

    k7b_fba = evaluate_k7b(models["fba"], world, seed + 5000)
    k7b_asym = evaluate_k7b(models["asym_split"], world, seed + 5000)
    results["k7b"] = {"fba": k7b_fba, "asym_split": k7b_asym}

    print(f"\n--- K7b interchange ---")
    print(f"  FBA: orient={k7b_fba['orientation']} cross={k7b_fba['cross_acc']:.3f} "
          f"place={k7b_fba['place_preserved']:.3f} fiber={k7b_fba['fiber_preserved']:.3f} "
          f"wc_loc={k7b_fba.get('wc_same_loc',0):.3f} wc_state={k7b_fba.get('wc_same_state',0):.3f}")
    print(f"  Asym: orient={k7b_asym['orientation']} cross={k7b_asym['cross_acc']:.3f}")
    print(f"  Null: cross={k7b_fba['hl_cross_null']:.3f} pres={k7b_fba['hl_pres_null']:.3f}")

    fba_te_eps = results["models"]["fba"]["test_episodes"]
    for ctrl in ["flat", "flat_matched", "asym_split", "modular", "flat_bn"]:
        ctrl_te_eps = results["models"][ctrl]["test_episodes"]
        pe = paired_class_effect(fba_te_eps, ctrl_te_eps)
        results["paired_effects"][f"fba_vs_{ctrl}"] = pe
        print(f"  FBA vs {ctrl}: effect={pe['mean_effect']:+.3f} "
              f"CI=[{pe['ci_lo']:+.3f}, {pe['ci_hi']:+.3f}]")

    return results


def evaluate_gates(all_results, cfg):
    print(f"\n{'='*60}\nGATE EVALUATION\n{'='*60}")

    hl_cross_null = (1 - cfg.loc_noise) * (1 - cfg.state_noise)
    hl_pres_null = 1 - max(cfg.loc_noise, cfg.state_noise)

    per_seed = {}
    all_oracle_valid = True

    for r in all_results:
        seed = r["seed"]
        m = r["models"]

        if not r["oracle"]["valid"]:
            all_oracle_valid = False
            print(f"  seed={seed}: ORACLE INVALID")
            per_seed[seed] = {"joint_pass": False, "oracle_invalid": True}
            continue

        k4 = all(m[n]["train"] >= 0.90 for n in m)
        fba_test = m["fba"]["test"]
        best_flat = max(m["flat"]["test"], m["flat_matched"]["test"],
                        m["flat_bn"]["test"])
        asym_test = m["asym_split"]["test"]
        mod_test = m["modular"]["test"]

        k6 = (fba_test - best_flat) >= 0.20
        k7a = (fba_test - asym_test) >= 0.15
        fba_beats_mod = fba_test > mod_test

        k7b = r["k7b"]["fba"]
        k7b_self = k7b["self_exact_match"]
        k7b_ci = k7b.get("cross_ci", [0, 0])
        k7b_cross = k7b_ci[0] > hl_cross_null + 0.03
        k7b_place_ci = k7b.get("place_ci", [0, 0])
        k7b_place = k7b_place_ci[0] > hl_pres_null + 0.02
        k7b_fiber_ci = k7b.get("fiber_ci", [0, 0])
        k7b_fiber = k7b_fiber_ci[0] > hl_pres_null + 0.02
        wc_loc_ci = k7b.get("wc_loc_ci", [0, 0])
        wc_state_ci = k7b.get("wc_state_ci", [0, 0])
        k7b_wc = (wc_loc_ci[0] > hl_pres_null and
                  wc_state_ci[0] > hl_pres_null)
        cls_ci = k7b.get("cross_by_class", {})
        k7b_cls = cls_ci.get("ci_lo", 0) > hl_cross_null
        k7b_pass = (k7b_self and k7b_cross and k7b_place and k7b_fiber
                    and k7b_wc and k7b_cls)

        pe = r.get("paired_effects", {})
        pe_pass = True
        for ctrl_key in ["fba_vs_flat", "fba_vs_flat_matched", "fba_vs_flat_bn"]:
            eff = pe.get(ctrl_key, {})
            if eff.get("ci_lo", 0) <= 0:
                pe_pass = False
                break

        joint = k4 and k6 and k7a and fba_beats_mod and k7b_pass and pe_pass

        print(f"  seed={seed}:")
        print(f"    K4={'PASS' if k4 else 'FAIL'}")
        if not k4:
            for n in m:
                if m[n]["train"] < 0.90:
                    print(f"      {n}: train={m[n]['train']:.3f}")
        print(f"    K6={fba_test-best_flat:+.3f} ({'PASS' if k6 else 'FAIL'}) "
              f"fba={fba_test:.3f} best_flat={best_flat:.3f}")
        print(f"    K7a={fba_test-asym_test:+.3f} ({'PASS' if k7a else 'FAIL'}) "
              f"asym={asym_test:.3f}")
        print(f"    FBA>mod={'PASS' if fba_beats_mod else 'FAIL'} "
              f"mod={mod_test:.3f}")
        print(f"    K7b={'PASS' if k7b_pass else 'FAIL'} "
              f"cross_ci=[{k7b_ci[0]:.3f},{k7b_ci[1]:.3f}](>{hl_cross_null+.03:.3f}) "
              f"place_ci=[{k7b_place_ci[0]:.3f},{k7b_place_ci[1]:.3f}] "
              f"fiber_ci=[{k7b_fiber_ci[0]:.3f},{k7b_fiber_ci[1]:.3f}] "
              f"wc_loc_ci=[{wc_loc_ci[0]:.3f},{wc_loc_ci[1]:.3f}](>{hl_pres_null:.3f}) "
              f"wc_state_ci=[{wc_state_ci[0]:.3f},{wc_state_ci[1]:.3f}] "
              f"cls_ci=[{cls_ci.get('ci_lo',0):.3f},{cls_ci.get('ci_hi',0):.3f}](>{hl_cross_null:.3f})")
        print(f"    Paired={'PASS' if pe_pass else 'FAIL'} "
              + " ".join(f"{k}:[{pe[k]['ci_lo']:+.3f},{pe[k]['ci_hi']:+.3f}]"
                         for k in sorted(pe.keys()) if k.startswith("fba_vs_flat")))
        print(f"    JOINT: {'PASS' if joint else 'FAIL'}")

        per_seed[seed] = {
            "K4": k4, "K6_gap": fba_test - best_flat, "K7a_gap": fba_test - asym_test,
            "fba_vs_mod": fba_test - mod_test, "K7b": k7b_pass,
            "paired_effect_pass": pe_pass,
            "joint_pass": joint,
            "all_test": {n: m[n]["test"] for n in m},
        }

    if not all_oracle_valid:
        print(f"\n  OVERALL: FAIL (oracle violation)")
        return per_seed, False

    n_seeds = len(per_seed)
    n_joint_pass = sum(1 for g in per_seed.values() if g["joint_pass"])
    majority = (n_seeds + 1) // 2
    overall = n_joint_pass >= majority

    print(f"\n  Joint-pass seeds: {n_joint_pass}/{n_seeds} "
          f"(need {majority}) {'PASS' if overall else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")

    return per_seed, overall


if __name__ == "__main__":
    cfg = Config()
    t0 = time.time()

    print("FBA-0: Fiber-Bottleneck Architecture Constructive Program")
    print(f"Config: {cfg.n_locations}x{cfg.n_states} grid, T={cfg.episode_length}, "
          f"noise={cfg.loc_noise}/{cfg.state_noise}, epochs={cfg.n_epochs}")

    all_results = []
    for seed in cfg.seeds:
        all_results.append(run_seed(seed, cfg))

    gates, overall = evaluate_gates(all_results, cfg)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    out_dir = Path("experiments/results/fba_0")
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        return obj

    class_data = {}
    token_maps = {}
    for sr in all_results:
        s = sr["seed"]
        world = GridWorld(cfg, seed=s)
        class_data[str(s)] = world.class_manifests
        token_maps[str(s)] = {str(k): list(v)
                               for k, v in world.token_to_factors.items()}

    verdict = {
        "experiment": "FBA-0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {k: v for k, v in cfg.__dict__.items()
                   if not k.startswith('_')},
        "results": make_serializable(all_results),
        "gates": make_serializable(gates),
        "overall": overall,
        "elapsed_s": elapsed,
        "class_manifests": make_serializable(class_data),
        "token_maps": token_maps,
    }

    with open(out_dir / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, default=str)

    print(f"\nResults saved to {out_dir / 'verdict.json'}")
