"""
HANDLE-μ: Causal Handle Algebra in a Designed Latent World

Distance-1 constructive control. Non-visual key-lock grid on CPU.
Spec: theory/HANDLE_MU.md (Codex Rounds 1-2, entries e824-e836).

Single file: environment, models, training, paired-history generator,
intervention evaluator, clustered reduction, and verdict.

Usage:
    python experiments/run_handle_mu.py [--seed 42] [--staircase-rung 1]
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "experiments" / "results" / "handle_mu"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    grid_size: int = 7
    visibility_radius: int = 2
    n_handles: int = 5          # 2 keys + 2 locks + goal
    n_obs_records: int = 6      # + agent
    slot_width: int = 32
    n_slots: int = 6
    n_actions: int = 5          # N, S, E, W, use
    n_object_types: int = 4     # key, lock, goal, agent
    n_status_values: int = 4    # idle, held, open, active
    max_pos_offset: int = 4     # visibility diameter - 1 = 4
    sparse_top_k: int = 2

    n_train_levels: int = 64
    n_val_levels: int = 16
    n_test_levels: int = 32
    trajs_per_level: int = 32
    traj_length: int = 32
    scripted_fraction: float = 0.5

    model_seeds: tuple = (42, 137, 2026)
    lr: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 40
    grad_clip: float = 1.0
    event_loss_weight: float = 1.0

    intervention_n_pairs: int = 512
    bootstrap_n: int = 2000
    bootstrap_ci: float = 0.95

    epsilon_tv: float = 0.05  # TV threshold for causal contact
    staircase_rung: int = 1   # 1-6, advance one at a time

    device: str = "cpu"


# ---------------------------------------------------------------------------
# Environment: Key-Lock Grid World
# ---------------------------------------------------------------------------

NORTH, SOUTH, EAST, WEST, USE = range(5)
DELTAS = {NORTH: (-1, 0), SOUTH: (1, 0), EAST: (0, 1), WEST: (0, -1), USE: (0, 0)}

OBJ_KEY, OBJ_LOCK, OBJ_GOAL, OBJ_AGENT = range(4)
STATUS_IDLE, STATUS_HELD, STATUS_OPEN, STATUS_ACTIVE = range(4)

@dataclass
class ObjectState:
    obj_type: int
    row: int
    col: int
    status: int
    handle_id: int  # -1 for agent
    identity: int   # episode-local opaque ID

@dataclass
class LevelConfig:
    grid_size: int
    key_positions: list       # [(r,c), (r,c)]
    lock_positions: list      # [(r,c), (r,c)]
    goal_position: tuple      # (r,c)
    agent_start: tuple        # (r,c)
    key_lock_bijection: list  # [0->?, 1->?] which key opens which lock

@dataclass
class WorldState:
    agent_row: int
    agent_col: int
    keys: list           # [ObjectState, ObjectState]
    locks: list          # [ObjectState, ObjectState]
    goal: ObjectState
    key_held: list       # [bool, bool]
    lock_open: list      # [bool, bool]
    goal_active: bool
    step: int


class KeyLockGridWorld:

    def __init__(self, level_config: LevelConfig):
        self.cfg = level_config
        self.gs = level_config.grid_size

    def reset(self) -> WorldState:
        cfg = self.cfg
        keys = [
            ObjectState(OBJ_KEY, cfg.key_positions[i][0], cfg.key_positions[i][1],
                        STATUS_IDLE, handle_id=i, identity=-1)
            for i in range(2)
        ]
        locks = [
            ObjectState(OBJ_LOCK, cfg.lock_positions[i][0], cfg.lock_positions[i][1],
                        STATUS_IDLE, handle_id=2+i, identity=-1)
            for i in range(2)
        ]
        goal = ObjectState(OBJ_GOAL, cfg.goal_position[0], cfg.goal_position[1],
                           STATUS_IDLE, handle_id=4, identity=-1)
        return WorldState(
            agent_row=cfg.agent_start[0], agent_col=cfg.agent_start[1],
            keys=keys, locks=locks, goal=goal,
            key_held=[False, False], lock_open=[False, False],
            goal_active=False, step=0
        )

    def step(self, state: WorldState, action: int) -> tuple:
        """Returns (new_state, events). Events are list of (event_type, handle_id)."""
        dr, dc = DELTAS[action]
        nr = max(0, min(self.gs - 1, state.agent_row + dr))
        nc = max(0, min(self.gs - 1, state.agent_col + dc))

        new_state = WorldState(
            agent_row=nr, agent_col=nc,
            keys=[ObjectState(k.obj_type, k.row, k.col, k.status, k.handle_id, k.identity) for k in state.keys],
            locks=[ObjectState(l.obj_type, l.row, l.col, l.status, l.handle_id, l.identity) for l in state.locks],
            goal=ObjectState(state.goal.obj_type, state.goal.row, state.goal.col,
                             state.goal.status, state.goal.handle_id, state.goal.identity),
            key_held=list(state.key_held), lock_open=list(state.lock_open),
            goal_active=state.goal_active, step=state.step + 1
        )

        events = []

        for ki, key in enumerate(new_state.keys):
            if key.row == nr and key.col == nc and not new_state.key_held[ki]:
                if action == USE:
                    new_state.key_held[ki] = True
                    key.status = STATUS_HELD
                    events.append(("pickup", key.handle_id))

        bij = self.cfg.key_lock_bijection
        for ki in range(2):
            if new_state.key_held[ki]:
                new_state.keys[ki].row = nr
                new_state.keys[ki].col = nc

            li = bij[ki]
            lock = new_state.locks[li]
            if (new_state.key_held[ki] and lock.row == nr and lock.col == nc
                    and not new_state.lock_open[li] and action == USE):
                new_state.lock_open[li] = True
                lock.status = STATUS_OPEN
                events.append(("unlock", lock.handle_id))

        if all(new_state.lock_open) and not new_state.goal_active:
            if new_state.goal.row == nr and new_state.goal.col == nc and action == USE:
                new_state.goal_active = True
                new_state.goal.status = STATUS_ACTIVE
                events.append(("activate", new_state.goal.handle_id))

        return new_state, events

    def get_objects(self, state: WorldState) -> list:
        """All 6 objects: agent, key0, key1, lock0, lock1, goal."""
        agent = ObjectState(OBJ_AGENT, state.agent_row, state.agent_col,
                            STATUS_IDLE, handle_id=-1, identity=-1)
        return [agent] + state.keys + state.locks + [state.goal]

    def observe(self, state: WorldState, rng: random.Random) -> list:
        """Partial-visibility observation: list of visible record dicts."""
        objects = self.get_objects(state)
        identities = list(range(len(objects)))
        rng.shuffle(identities)
        for i, obj in enumerate(objects):
            obj.identity = identities[i]

        visible = []
        for obj in objects:
            dr = abs(obj.row - state.agent_row)
            dc = abs(obj.col - state.agent_col)
            if dr + dc <= 2:
                visible.append({
                    "identity": obj.identity,
                    "type": obj.obj_type,
                    "rel_row": obj.row - state.agent_row,
                    "rel_col": obj.col - state.agent_col,
                    "status": obj.status,
                    "handle_id": obj.handle_id,
                })

        order = list(range(len(visible)))
        rng.shuffle(order)
        return [visible[i] for i in order]

    def observe_encoded(self, state: WorldState, rng: random.Random,
                        n_slots: int, record_dim: int, max_pos_offset: int,
                        n_object_types: int, n_status_values: int) -> np.ndarray:
        """Directly encode observation as numpy array. Returns (n_slots, record_dim)."""
        result = np.zeros((n_slots, record_dim), dtype=np.float32)
        objects = self.get_objects(state)
        identities = list(range(len(objects)))
        rng.shuffle(identities)

        visible_ids = []
        for i, obj in enumerate(objects):
            dr = abs(obj.row - state.agent_row)
            dc = abs(obj.col - state.agent_col)
            if dr + dc <= 2:
                slot = identities[i]
                if slot < n_slots:
                    offset = 0
                    result[slot, offset + obj.obj_type] = 1.0
                    offset += n_object_types
                    row_idx = (obj.row - state.agent_row) + max_pos_offset
                    row_idx = max(0, min(2 * max_pos_offset, row_idx))
                    result[slot, offset + row_idx] = 1.0
                    offset += 2 * max_pos_offset + 1
                    col_idx = (obj.col - state.agent_col) + max_pos_offset
                    col_idx = max(0, min(2 * max_pos_offset, col_idx))
                    result[slot, offset + col_idx] = 1.0
                    offset += 2 * max_pos_offset + 1
                    result[slot, offset + obj.status] = 1.0
                    offset += n_status_values
                    result[slot, offset] = 1.0  # visibility flag
                visible_ids.append((identities[i], obj.handle_id))

        return result, visible_ids

    def encode_events(self, events: list) -> list:
        """Encode events as integer codes."""
        event_types = {"pickup": 0, "unlock": 1, "activate": 2, "none": 3}
        if not events:
            return [event_types["none"]]
        return [event_types[e[0]] for e in events]


def generate_level(seed: int, grid_size: int = 7) -> LevelConfig:
    """Generate a random level configuration."""
    rng = random.Random(seed)
    positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    rng.shuffle(positions)
    key_pos = [positions[0], positions[1]]
    lock_pos = [positions[2], positions[3]]
    goal_pos = positions[4]
    agent_start = positions[5]
    bijection = [0, 1]
    rng.shuffle(bijection)
    return LevelConfig(
        grid_size=grid_size,
        key_positions=key_pos, lock_positions=lock_pos,
        goal_position=goal_pos, agent_start=agent_start,
        key_lock_bijection=bijection,
    )


def scripted_policy(state: WorldState, level_cfg: LevelConfig, rng: random.Random) -> int:
    """Coverage policy: pick up keys, open locks, activate goal. Mixes with random."""
    ar, ac = state.agent_row, state.agent_col

    for ki in range(2):
        if not state.key_held[ki]:
            kr, kc = state.keys[ki].row, state.keys[ki].col
            if ar == kr and ac == kc:
                return USE
            if ar > kr: return NORTH
            if ar < kr: return SOUTH
            if ac > kc: return WEST
            if ac < kc: return EAST

    bij = level_cfg.key_lock_bijection
    for ki in range(2):
        li = bij[ki]
        if state.key_held[ki] and not state.lock_open[li]:
            lr, lc = state.locks[li].row, state.locks[li].col
            if ar == lr and ac == lc:
                return USE
            if ar > lr: return NORTH
            if ar < lr: return SOUTH
            if ac > lc: return WEST
            if ac < lc: return EAST

    if all(state.lock_open) and not state.goal_active:
        gr, gc = level_cfg.goal_position
        if ar == gr and ac == gc:
            return USE
        if ar > gr: return NORTH
        if ar < gr: return SOUTH
        if ac > gc: return WEST
        if ac < gc: return EAST

    return rng.randint(0, 4)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _record_dim(cfg: Config) -> int:
    return cfg.n_object_types + (2 * cfg.max_pos_offset + 1) * 2 + cfg.n_status_values + 1


@dataclass
class Trajectory:
    level_seed: int
    obs_encoded: np.ndarray   # (T+1, n_slots, record_dim) pre-encoded
    slot_maps: list           # list of [(slot_idx, handle_id), ...] per timestep
    actions: list             # list of int
    events: list              # list of event code lists
    states: list              # list of WorldState (ground truth for intervention)
    level_config: object      # LevelConfig


def generate_trajectories(level_seeds: list, cfg: Config, data_rng_seed: int = 0) -> list:
    """Generate trajectories for a set of level seeds."""
    rng = random.Random(data_rng_seed)
    all_trajs = []
    rd = _record_dim(cfg)

    for ls in level_seeds:
        level = generate_level(ls, cfg.grid_size)
        world = KeyLockGridWorld(level)

        for t_idx in range(cfg.trajs_per_level):
            obs_rng = random.Random(rng.randint(0, 2**31))
            state = world.reset()
            obs_encoded = np.zeros((cfg.traj_length + 1, cfg.n_slots, rd), dtype=np.float32)
            slot_maps = []
            actions = []
            events_list = []
            states = [state]

            use_scripted = t_idx < int(cfg.trajs_per_level * cfg.scripted_fraction)

            for step in range(cfg.traj_length):
                enc, vis_ids = world.observe_encoded(
                    state, obs_rng, cfg.n_slots, rd,
                    cfg.max_pos_offset, cfg.n_object_types, cfg.n_status_values)
                obs_encoded[step] = enc
                slot_maps.append(vis_ids)

                if use_scripted:
                    action = scripted_policy(state, level, rng)
                else:
                    action = rng.randint(0, cfg.n_actions - 1)

                state, evts = world.step(state, action)
                actions.append(action)
                events_list.append(world.encode_events(evts))
                states.append(state)

            enc, vis_ids = world.observe_encoded(
                state, obs_rng, cfg.n_slots, rd,
                cfg.max_pos_offset, cfg.n_object_types, cfg.n_status_values)
            obs_encoded[cfg.traj_length] = enc
            slot_maps.append(vis_ids)

            all_trajs.append(Trajectory(
                level_seed=ls, obs_encoded=obs_encoded,
                slot_maps=slot_maps,
                actions=actions, events=events_list,
                states=states, level_config=level
            ))

    return all_trajs


# ---------------------------------------------------------------------------
# Observation encoding (fixed, no learning)
# ---------------------------------------------------------------------------

def encode_observation(obs_records: list, cfg: Config) -> torch.Tensor:
    """Encode a single timestep's observation into a fixed-size tensor.

    Returns: (n_slots, record_dim) where record_dim encodes type, rel_pos, status.
    Invisible slots get a zero vector with a visibility flag = 0.
    """
    record_dim = cfg.n_object_types + (2 * cfg.max_pos_offset + 1) * 2 + cfg.n_status_values + 1
    result = torch.zeros(cfg.n_slots, record_dim)

    slot_assignment = {}
    for rec in obs_records:
        identity = rec["identity"]
        if identity < cfg.n_slots:
            slot_assignment[identity] = rec

    for slot_idx in range(cfg.n_slots):
        if slot_idx in slot_assignment:
            rec = slot_assignment[slot_idx]
            offset = 0
            type_vec = torch.zeros(cfg.n_object_types)
            type_vec[rec["type"]] = 1.0
            result[slot_idx, offset:offset + cfg.n_object_types] = type_vec
            offset += cfg.n_object_types

            row_vec = torch.zeros(2 * cfg.max_pos_offset + 1)
            row_idx = rec["rel_row"] + cfg.max_pos_offset
            row_idx = max(0, min(len(row_vec) - 1, row_idx))
            row_vec[row_idx] = 1.0
            result[slot_idx, offset:offset + len(row_vec)] = row_vec
            offset += len(row_vec)

            col_vec = torch.zeros(2 * cfg.max_pos_offset + 1)
            col_idx = rec["rel_col"] + cfg.max_pos_offset
            col_idx = max(0, min(len(col_vec) - 1, col_idx))
            col_vec[col_idx] = 1.0
            result[slot_idx, offset:offset + len(col_vec)] = col_vec
            offset += len(col_vec)

            status_vec = torch.zeros(cfg.n_status_values)
            status_vec[rec["status"]] = 1.0
            result[slot_idx, offset:offset + cfg.n_status_values] = status_vec
            offset += cfg.n_status_values

            result[slot_idx, offset] = 1.0  # visibility flag

    return result


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RecordEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        rd = _record_dim(cfg)
        self.net = nn.Sequential(
            nn.Linear(rd, cfg.slot_width),
            nn.ReLU(),
            nn.Linear(cfg.slot_width, cfg.slot_width),
        )

    def forward(self, x):
        return self.net(x)


class SlotGRUCell(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gru = nn.GRUCell(cfg.slot_width, cfg.slot_width)

    def forward(self, x, h):
        return self.gru(x, h)


class DenseMessaging(nn.Module):
    """All-pairs message passing between slots."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.query = nn.Linear(cfg.slot_width, cfg.slot_width)
        self.key = nn.Linear(cfg.slot_width, cfg.slot_width)
        self.value = nn.Linear(cfg.slot_width, cfg.slot_width)
        self.out = nn.Linear(cfg.slot_width, cfg.slot_width)
        self.scale = cfg.slot_width ** 0.5

    def forward(self, slots):
        """slots: (B, N, D) -> (B, N, D)"""
        Q = self.query(slots)
        K = self.key(slots)
        V = self.value(slots)
        attn = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        attn = F.softmax(attn, dim=-1)
        msg = torch.matmul(attn, V)
        return self.out(msg)


class SparseMessaging(nn.Module):
    """Learned top-k message gating."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.query = nn.Linear(cfg.slot_width, cfg.slot_width)
        self.key = nn.Linear(cfg.slot_width, cfg.slot_width)
        self.value = nn.Linear(cfg.slot_width, cfg.slot_width)
        self.out = nn.Linear(cfg.slot_width, cfg.slot_width)
        self.scale = cfg.slot_width ** 0.5
        self.top_k = cfg.sparse_top_k

    def forward(self, slots):
        Q = self.query(slots)
        K = self.key(slots)
        V = self.value(slots)
        attn = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        # top-k masking
        topk_vals, topk_idx = attn.topk(self.top_k, dim=-1)
        mask = torch.zeros_like(attn).scatter_(-1, topk_idx, 1.0)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        msg = torch.matmul(attn, V)
        return self.out(msg)


class PredictionHead(nn.Module):
    """Predict next observation records and events from slot states."""
    def __init__(self, cfg: Config):
        super().__init__()
        rd = _record_dim(cfg)
        self.obs_head = nn.Sequential(
            nn.Linear(cfg.slot_width, cfg.slot_width),
            nn.ReLU(),
            nn.Linear(cfg.slot_width, rd),
        )
        total_slot = cfg.n_slots * cfg.slot_width
        self.event_head = nn.Sequential(
            nn.Linear(total_slot, cfg.slot_width),
            nn.ReLU(),
            nn.Linear(cfg.slot_width, 4),  # pickup, unlock, activate, none
        )

    def forward(self, slots):
        """slots: (B, N, D). Returns obs_pred (B, N, rd), event_pred (B, 4)."""
        obs_pred = self.obs_head(slots)
        B, N, D = slots.shape
        event_pred = self.event_head(slots.reshape(B, N * D))
        return obs_pred, event_pred


class DenseSlotModel(nn.Module):
    """Primary: dense typed slots with all-pairs messaging."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = RecordEncoder(cfg)
        self.action_embed = nn.Embedding(cfg.n_actions, cfg.slot_width)
        self.gru = SlotGRUCell(cfg)
        self.messaging = DenseMessaging(cfg)
        self.head = PredictionHead(cfg)
        self.n_slots = cfg.n_slots

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.n_slots, self.cfg.slot_width,
                           device=next(self.parameters()).device)

    def forward_step(self, obs_encoded, action, hidden):
        """Single step. obs_encoded: (B, N, D), action: (B,), hidden: (B, N, D)."""
        B, N, D = hidden.shape
        act_emb = self.action_embed(action).unsqueeze(1).expand(B, N, D)
        combined = obs_encoded + act_emb
        # Vectorize GRU across slots: reshape (B,N,D) -> (B*N,D)
        hidden = self.gru(combined.reshape(B*N, D), hidden.reshape(B*N, D)).reshape(B, N, D)
        hidden = hidden + self.messaging(hidden)
        return hidden

    def forward(self, obs_seq, action_seq):
        B, Tp1, N, rd = obs_seq.shape
        T = Tp1 - 1
        hidden = self.init_hidden(B)
        obs_preds = []
        event_preds = []
        for t in range(T):
            enc = self.encoder(obs_seq[:, t])
            hidden = self.forward_step(enc, action_seq[:, t], hidden)
            op, ep = self.head(hidden)
            obs_preds.append(op)
            event_preds.append(ep)
        return torch.stack(obs_preds, 1), torch.stack(event_preds, 1), hidden

    def forward_step_with_hidden(self, obs_t, action_t, hidden):
        enc = self.encoder(obs_t)
        hidden = self.forward_step(enc, action_t, hidden)
        op, ep = self.head(hidden)
        return op, ep, hidden


class SparseSlotModel(nn.Module):
    """Secondary: learned-sparse typed slots."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = RecordEncoder(cfg)
        self.action_embed = nn.Embedding(cfg.n_actions, cfg.slot_width)
        self.gru = SlotGRUCell(cfg)
        self.messaging = SparseMessaging(cfg)
        self.head = PredictionHead(cfg)
        self.n_slots = cfg.n_slots

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.n_slots, self.cfg.slot_width,
                           device=next(self.parameters()).device)

    def forward_step(self, obs_encoded, action, hidden):
        B, N, D = hidden.shape
        act_emb = self.action_embed(action).unsqueeze(1).expand(B, N, D)
        combined = obs_encoded + act_emb
        hidden = self.gru(combined.reshape(B*N, D), hidden.reshape(B*N, D)).reshape(B, N, D)
        hidden = hidden + self.messaging(hidden)
        return hidden

    def forward(self, obs_seq, action_seq):
        B, Tp1, N, rd = obs_seq.shape
        T = Tp1 - 1
        hidden = self.init_hidden(B)
        obs_preds, event_preds = [], []
        for t in range(T):
            enc = self.encoder(obs_seq[:, t])
            hidden = self.forward_step(enc, action_seq[:, t], hidden)
            op, ep = self.head(hidden)
            obs_preds.append(op)
            event_preds.append(ep)
        return torch.stack(obs_preds, 1), torch.stack(event_preds, 1), hidden

    def forward_step_with_hidden(self, obs_t, action_t, hidden):
        enc = self.encoder(obs_t)
        hidden = self.forward_step(enc, action_t, hidden)
        op, ep = self.head(hidden)
        return op, ep, hidden


class FlatGRUModel(nn.Module):
    """Predictive control: flat GRU, parameter-matched."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        rd = _record_dim(cfg)
        input_dim = cfg.n_slots * rd + cfg.n_actions
        # Match parameter count to slot models (~within 5%)
        slot_params = self._estimate_slot_params(cfg)
        # Solve for hidden dim: GRU params ~ 3 * hidden * (input + hidden + 1) + head
        # Approximate
        self.hidden_dim = self._solve_hidden(input_dim, slot_params, cfg)
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.obs_head = nn.Sequential(
            nn.Linear(self.hidden_dim, cfg.slot_width),
            nn.ReLU(),
            nn.Linear(cfg.slot_width, cfg.n_slots * rd),
        )
        self.event_head = nn.Sequential(
            nn.Linear(self.hidden_dim, cfg.slot_width),
            nn.ReLU(),
            nn.Linear(cfg.slot_width, 4),
        )

    @staticmethod
    def _estimate_slot_params(cfg):
        rd = _record_dim(cfg)
        enc = rd * cfg.slot_width + cfg.slot_width + cfg.slot_width * cfg.slot_width + cfg.slot_width
        gru = 3 * (cfg.slot_width * cfg.slot_width + cfg.slot_width * cfg.slot_width + cfg.slot_width * 2)
        msg = 4 * (cfg.slot_width * cfg.slot_width + cfg.slot_width)
        head_obs = cfg.slot_width * cfg.slot_width + cfg.slot_width + cfg.slot_width * rd + rd
        head_evt = (cfg.n_slots * cfg.slot_width) * cfg.slot_width + cfg.slot_width + cfg.slot_width * 4 + 4
        act = cfg.n_actions * cfg.slot_width
        return enc + gru + msg + head_obs + head_evt + act

    @staticmethod
    def _solve_hidden(input_dim, target_params, cfg):
        rd = _record_dim(cfg)
        for h in range(32, 512):
            proj = input_dim * h + h
            gru = 3 * (h * h + h * h + h * 2)
            obs = h * cfg.slot_width + cfg.slot_width + cfg.slot_width * (cfg.n_slots * rd) + cfg.n_slots * rd
            evt = h * cfg.slot_width + cfg.slot_width + cfg.slot_width * 4 + 4
            total = proj + gru + obs + evt
            if total >= target_params * 0.95:
                return h
        return 256

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim,
                           device=next(self.parameters()).device)

    def forward(self, obs_seq, action_seq):
        B, Tp1, N, rd_dim = obs_seq.shape
        T = Tp1 - 1
        hidden = self.init_hidden(B)
        obs_preds, event_preds = [], []
        for t in range(T):
            flat_obs = obs_seq[:, t].reshape(B, -1)
            act_onehot = F.one_hot(action_seq[:, t], self.cfg.n_actions).float()
            inp = torch.cat([flat_obs, act_onehot], dim=-1)
            inp = self.input_proj(inp)
            hidden = self.gru(inp, hidden)
            op = self.obs_head(hidden).reshape(B, N, rd_dim)
            ep = self.event_head(hidden)
            obs_preds.append(op)
            event_preds.append(ep)
        return torch.stack(obs_preds, 1), torch.stack(event_preds, 1), hidden

    def forward_step_with_hidden(self, obs_t, action_t, hidden):
        B = obs_t.shape[0]
        flat_obs = obs_t.reshape(B, -1)
        act_onehot = F.one_hot(action_t, self.cfg.n_actions).float()
        inp = torch.cat([flat_obs, act_onehot], dim=-1)
        inp = self.input_proj(inp)
        hidden = self.gru(inp, hidden)
        N = self.cfg.n_slots
        rd_dim = _record_dim(self.cfg)
        op = self.obs_head(hidden).reshape(B, N, rd_dim)
        ep = self.event_head(hidden)
        return op, ep, hidden


class HistorylessModel(nn.Module):
    """No recurrence — tests whether persistent state is needed."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        rd = _record_dim(cfg)
        self.encoder = RecordEncoder(cfg)
        self.head = PredictionHead(cfg)

    def forward(self, obs_seq, action_seq):
        B, Tp1, N, rd = obs_seq.shape
        T = Tp1 - 1
        obs_preds, event_preds = [], []
        for t in range(T):
            enc = self.encoder(obs_seq[:, t])
            op, ep = self.head(enc)
            obs_preds.append(op)
            event_preds.append(ep)
        return torch.stack(obs_preds, 1), torch.stack(event_preds, 1), None


class OracleModel(nn.Module):
    """Direct-state oracle for pipeline validity."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        rd = _record_dim(cfg)
        state_dim = 2 + 2*3 + 2*3 + 3 + 2 + 2 + 1  # agent_pos + keys + locks + goal + held + open + active
        self.net = nn.Sequential(
            nn.Linear(state_dim + cfg.n_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.obs_head = nn.Linear(128, cfg.n_slots * rd)
        self.event_head = nn.Linear(128, 4)

    @staticmethod
    def encode_state(state: WorldState, action: int, cfg: Config) -> torch.Tensor:
        feats = [
            state.agent_row / cfg.grid_size, state.agent_col / cfg.grid_size,
        ]
        for k in state.keys:
            feats.extend([k.row / cfg.grid_size, k.col / cfg.grid_size, k.status / 3.0])
        for l in state.locks:
            feats.extend([l.row / cfg.grid_size, l.col / cfg.grid_size, l.status / 3.0])
        feats.extend([state.goal.row / cfg.grid_size, state.goal.col / cfg.grid_size, state.goal.status / 3.0])
        feats.extend([float(h) for h in state.key_held])
        feats.extend([float(o) for o in state.lock_open])
        feats.append(float(state.goal_active))
        act = [0.0] * cfg.n_actions
        act[action] = 1.0
        feats.extend(act)
        return torch.tensor(feats, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_batches(trajs: list, cfg: Config, shuffle: bool = True) -> list:
    """Convert trajectories to tensor batches using pre-encoded observations."""
    n = len(trajs)
    rd = _record_dim(cfg)

    obs_all = np.stack([t.obs_encoded for t in trajs])  # (N, T+1, slots, rd)
    act_all = np.array([t.actions for t in trajs], dtype=np.int64)  # (N, T)
    evt_all = np.array([[e[0] for e in t.events] for t in trajs], dtype=np.int64)  # (N, T)
    level_ids = [t.level_seed for t in trajs]

    obs_t = torch.from_numpy(obs_all)
    act_t = torch.from_numpy(act_all)
    tgt_obs_t = obs_t[:, 1:]  # next obs
    tgt_evt_t = torch.from_numpy(evt_all)

    indices = list(range(n))
    if shuffle:
        random.shuffle(indices)

    batches = []
    for i in range(0, n, cfg.batch_size):
        idx = indices[i:i + cfg.batch_size]
        batches.append({
            "obs": obs_t[idx],
            "actions": act_t[idx],
            "target_obs": tgt_obs_t[idx],
            "target_events": tgt_evt_t[idx],
            "level_ids": [level_ids[j] for j in idx],
        })

    return batches


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_loss(obs_pred, event_pred, target_obs, target_events, cfg: Config):
    """Masked cross-entropy for obs + events."""
    B, T, N, rd = target_obs.shape

    vis_mask = target_obs[:, :, :, -1]  # visibility flag
    obs_loss = F.mse_loss(obs_pred * vis_mask.unsqueeze(-1),
                          target_obs * vis_mask.unsqueeze(-1))

    event_loss = F.cross_entropy(event_pred.reshape(-1, 4), target_events.reshape(-1))

    return obs_loss + cfg.event_loss_weight * event_loss, obs_loss.item(), event_loss.item()


def train_model(model, train_batches, val_batches, cfg: Config, model_name: str):
    """Train a model and return metrics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best_val_loss = float('inf')
    best_state = None
    patience = 5
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(cfg.n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_batches:
            obs = batch["obs"].to(cfg.device)
            actions = batch["actions"].to(cfg.device)
            tgt_obs = batch["target_obs"].to(cfg.device)
            tgt_evt = batch["target_events"].to(cfg.device)

            optimizer.zero_grad()
            obs_pred, event_pred, _ = model(obs, actions)
            loss, ol, el = compute_loss(obs_pred, event_pred, tgt_obs, tgt_evt, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_batches:
                obs = batch["obs"].to(cfg.device)
                actions = batch["actions"].to(cfg.device)
                tgt_obs = batch["target_obs"].to(cfg.device)
                tgt_evt = batch["target_events"].to(cfg.device)
                obs_pred, event_pred, _ = model(obs, actions)
                loss, _, _ = compute_loss(obs_pred, event_pred, tgt_obs, tgt_evt, cfg)
                val_loss += loss.item()
                n_val += 1

        avg_val = val_loss / max(n_val, 1)
        val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == cfg.n_epochs - 1:
            print(f"  [{model_name}] Epoch {epoch:3d}  train={avg_train:.4f}  val={avg_val:.4f}")

        if patience_counter >= patience:
            print(f"  [{model_name}] Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"train_losses": train_losses, "val_losses": val_losses,
            "best_val_loss": best_val_loss}


# ---------------------------------------------------------------------------
# Evaluation: next-event and visible-status accuracy
# ---------------------------------------------------------------------------

def evaluate_prediction(model, batches, cfg: Config) -> dict:
    """Compute next-event and visible-status macro-F1."""
    model.eval()
    all_event_preds = []
    all_event_targets = []
    all_obs_preds = []
    all_obs_targets = []
    all_vis_masks = []
    all_level_ids = []

    with torch.no_grad():
        for batch in batches:
            obs = batch["obs"].to(cfg.device)
            actions = batch["actions"].to(cfg.device)
            tgt_obs = batch["target_obs"].to(cfg.device)
            tgt_evt = batch["target_events"].to(cfg.device)

            obs_pred, event_pred, _ = model(obs, actions)
            all_event_preds.append(event_pred.argmax(dim=-1).cpu())
            all_event_targets.append(tgt_evt.cpu())
            all_obs_preds.append(obs_pred.cpu())
            all_obs_targets.append(tgt_obs.cpu())
            all_vis_masks.append(tgt_obs[:, :, :, -1].cpu())
            all_level_ids.extend(batch["level_ids"])

    event_preds = torch.cat(all_event_preds, dim=0).reshape(-1).numpy()
    event_targets = torch.cat(all_event_targets, dim=0).reshape(-1).numpy()

    # Event macro-F1
    event_classes = set(event_targets) | set(event_preds)
    f1s = []
    for c in event_classes:
        tp = ((event_preds == c) & (event_targets == c)).sum()
        fp = ((event_preds == c) & (event_targets != c)).sum()
        fn = ((event_preds != c) & (event_targets == c)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        f1s.append(f1)
    event_macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    # Visible-status accuracy (status field in observation)
    obs_preds_cat = torch.cat(all_obs_preds, dim=0)
    obs_targets_cat = torch.cat(all_obs_targets, dim=0)
    vis_masks_cat = torch.cat(all_vis_masks, dim=0)

    rd = _record_dim(cfg)
    status_start = cfg.n_object_types + (2 * cfg.max_pos_offset + 1) * 2
    status_end = status_start + cfg.n_status_values

    pred_status = obs_preds_cat[:, :, :, status_start:status_end].argmax(dim=-1)
    true_status = obs_targets_cat[:, :, :, status_start:status_end].argmax(dim=-1)
    vis = vis_masks_cat > 0.5

    status_correct = ((pred_status == true_status) & vis).sum().item()
    status_total = vis.sum().item()
    status_acc = status_correct / max(status_total, 1)

    # Status macro-F1
    pred_s = pred_status[vis].numpy()
    true_s = true_status[vis].numpy()
    s_classes = set(true_s) | set(pred_s)
    s_f1s = []
    for c in s_classes:
        tp = ((pred_s == c) & (true_s == c)).sum()
        fp = ((pred_s == c) & (true_s != c)).sum()
        fn = ((pred_s != c) & (true_s == c)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        s_f1s.append(f1)
    status_macro_f1 = float(np.mean(s_f1s)) if s_f1s else 0.0

    return {
        "event_macro_f1": event_macro_f1,
        "status_accuracy": status_acc,
        "status_macro_f1": status_macro_f1,
        "n_event_samples": len(event_preds),
        "n_status_samples": status_total,
        "level_ids": list(set(all_level_ids)),
    }


# ---------------------------------------------------------------------------
# Paired history construction for intervention
# ---------------------------------------------------------------------------

@dataclass
class PairedHistories:
    """A pair of naturally-reached histories differing only in target handle."""
    level_seed: int
    target_handle: int
    donor_traj: Trajectory
    recipient_traj: Trajectory
    donor_step: int       # step where target state differs
    recipient_step: int
    shared_suffix: list   # shared future action sequence
    first_contact_step: Optional[int]  # simulator ground truth


def find_paired_histories(trajs: list, cfg: Config, target_handle: int,
                          n_pairs: int, rng: random.Random) -> list:
    """Find naturally-reached paired histories for intervention."""
    by_level = defaultdict(list)
    for t in trajs:
        by_level[t.level_seed].append(t)

    pairs = []
    level_seeds = list(by_level.keys())
    rng.shuffle(level_seeds)

    for ls in level_seeds:
        if len(pairs) >= n_pairs:
            break
        level_trajs = by_level[ls]
        if len(level_trajs) < 2:
            continue

        for attempt in range(min(100, len(level_trajs) * 2)):
            if len(pairs) >= n_pairs:
                break
            t1, t2 = rng.sample(level_trajs, 2)

            for step in range(1, min(len(t1.states) - 2, len(t2.states) - 2)):
                s1, s2 = t1.states[step], t2.states[step]

                if s1.agent_row != s2.agent_row or s1.agent_col != s2.agent_col:
                    continue

                target_differs = False
                non_target_same = True

                all_handles = list(range(5))  # 0-1: keys, 2-3: locks, 4: goal
                for h in all_handles:
                    h_same = _handle_state_equal(s1, s2, h)
                    if h == target_handle:
                        target_differs = not h_same
                    else:
                        if not h_same:
                            non_target_same = False

                if target_differs and non_target_same:
                    suffix_len = min(8, len(t1.actions) - step, len(t2.actions) - step)
                    if suffix_len < 2:
                        continue
                    shared_suffix = t1.actions[step:step + suffix_len]

                    fc = _find_first_contact(t1, t2, step, target_handle, cfg)

                    pairs.append(PairedHistories(
                        level_seed=ls, target_handle=target_handle,
                        donor_traj=t1, recipient_traj=t2,
                        donor_step=step, recipient_step=step,
                        shared_suffix=shared_suffix,
                        first_contact_step=fc,
                    ))
                    break

    return pairs


def _handle_state_equal(s1: WorldState, s2: WorldState, handle: int) -> bool:
    """Check if two world states have the same state for a given handle."""
    if handle < 2:  # keys
        k1, k2 = s1.keys[handle], s2.keys[handle]
        return (k1.row == k2.row and k1.col == k2.col and
                k1.status == k2.status and s1.key_held[handle] == s2.key_held[handle])
    elif handle < 4:  # locks
        li = handle - 2
        l1, l2 = s1.locks[li], s2.locks[li]
        return (l1.row == l2.row and l1.col == l2.col and
                l1.status == l2.status and s1.lock_open[li] == s2.lock_open[li])
    else:  # goal
        return (s1.goal.row == s2.goal.row and s1.goal.col == s2.goal.col and
                s1.goal.status == s2.goal.status and s1.goal_active == s2.goal_active)


def _find_first_contact(t1: Trajectory, t2: Trajectory, step: int,
                        target_handle: int, cfg: Config) -> Optional[int]:
    """Find the first step where intervention on target would change another handle's future."""
    for future_step in range(step + 1, min(len(t1.states), len(t2.states))):
        s1, s2 = t1.states[future_step], t2.states[future_step]
        for h in range(5):
            if h == target_handle:
                continue
            if not _handle_state_equal(s1, s2, h):
                return future_step - step
    return None


# ---------------------------------------------------------------------------
# Intervention evaluation
# ---------------------------------------------------------------------------

def evaluate_intervention(model, pairs: list, cfg: Config, model_type: str) -> dict:
    """Run slot-swap intervention and measure causal consumption."""
    if not pairs or model_type in ("flat_gru", "historyless"):
        return {"skipped": True, "reason": f"no swap for {model_type}"}

    model.eval()
    results = {
        "n_pairs": len(pairs),
        "patch_integrity": [],      # same-value patch non-target TV
        "causal_consumption": [],   # hybrid counterfactual accuracy at first contact
        "pre_contact_effects": [],  # false effect rate before contact
        "timing_errors": [],        # onset timing error
        "by_level": defaultdict(list),
    }

    with torch.no_grad():
        for pair in pairs:
            dt, rt = pair.donor_traj, pair.recipient_traj
            ds, rs = pair.donor_step, pair.recipient_step
            suffix = pair.shared_suffix

            # Use pre-encoded observations
            donor_obs = torch.from_numpy(dt.obs_encoded).unsqueeze(0)
            recip_obs = torch.from_numpy(rt.obs_encoded).unsqueeze(0)
            donor_acts = torch.tensor(dt.actions, dtype=torch.long).unsqueeze(0)
            recip_acts = torch.tensor(rt.actions, dtype=torch.long).unsqueeze(0)

            # Run both to intervention point
            _, _, donor_hidden = model(donor_obs[:, :ds+1], donor_acts[:, :ds])
            _, _, recip_hidden = model(recip_obs[:, :rs+1], recip_acts[:, :rs])

            if donor_hidden is None or recip_hidden is None:
                continue

            target_slot = _find_target_slot_from_map(
                dt.slot_maps[ds], pair.target_handle, cfg.n_slots)

            if target_slot is None:
                continue

            # Create hybrid: recipient hidden with donor's target slot
            hybrid_hidden = recip_hidden.clone()
            hybrid_hidden[0, target_slot] = donor_hidden[0, target_slot]

            # Run forward with shared suffix, comparing predictions
            h_recip = recip_hidden.clone()
            h_hybrid = hybrid_hidden.clone()
            h_donor = donor_hidden.clone()

            pre_contact_false = 0
            pre_contact_total = 0
            contact_found = False
            contact_step = None

            for si, action in enumerate(suffix):
                act_t = torch.tensor([action], dtype=torch.long)

                if si + ds < dt.obs_encoded.shape[0] and si + rs < rt.obs_encoded.shape[0]:
                    obs_d = torch.from_numpy(dt.obs_encoded[si + ds]).unsqueeze(0)
                    obs_r = torch.from_numpy(rt.obs_encoded[si + rs]).unsqueeze(0)
                else:
                    break

                _, ep_recip, h_recip = model.forward_step_with_hidden(obs_r, act_t, h_recip)
                _, ep_hybrid, h_hybrid = model.forward_step_with_hidden(obs_r, act_t, h_hybrid)
                _, ep_donor, h_donor = model.forward_step_with_hidden(obs_d, act_t, h_donor)

                # TV between hybrid and recipient event predictions
                p_recip = F.softmax(ep_recip, dim=-1)
                p_hybrid = F.softmax(ep_hybrid, dim=-1)
                p_donor = F.softmax(ep_donor, dim=-1)

                tv_hr = 0.5 * (p_hybrid - p_recip).abs().sum().item()

                if pair.first_contact_step is not None:
                    if si < pair.first_contact_step:
                        if tv_hr > cfg.epsilon_tv:
                            pre_contact_false += 1
                        pre_contact_total += 1
                    elif si == pair.first_contact_step and not contact_found:
                        contact_found = True
                        contact_step = si

                        # Causal consumption: does hybrid predict donor's event?
                        donor_event = dt.events[ds + si][0] if ds + si < len(dt.events) else 3
                        hybrid_event = ep_hybrid.argmax(dim=-1).item()
                        recip_event = ep_recip.argmax(dim=-1).item()

                        results["causal_consumption"].append({
                            "correct": int(hybrid_event == donor_event),
                            "recip_correct": int(recip_event == donor_event),
                            "level": pair.level_seed,
                            "handle": pair.target_handle,
                        })

            if pre_contact_total > 0:
                results["pre_contact_effects"].append(
                    pre_contact_false / pre_contact_total)

            if pair.first_contact_step is not None and contact_step is not None:
                results["timing_errors"].append(
                    abs(contact_step - pair.first_contact_step))

            results["by_level"][pair.level_seed].append({
                "target_handle": pair.target_handle,
                "contact_found": contact_found,
            })

    return results


def _find_target_slot_from_map(slot_map: list, target_handle: int, n_slots: int) -> Optional[int]:
    """Find which slot corresponds to target handle using slot_map from observe_encoded."""
    for slot_idx, handle_id in slot_map:
        if handle_id == target_handle and slot_idx < n_slots:
            return slot_idx
    return None


# ---------------------------------------------------------------------------
# Composition test (higher-order: both locks -> goal)
# ---------------------------------------------------------------------------

def evaluate_composition(model, trajs: list, cfg: Config, model_type: str) -> dict:
    """Test higher-order composition: {lock0, lock1} -> goal."""
    if model_type in ("flat_gru", "historyless"):
        return {"skipped": True}

    rng = random.Random(9999)
    single_lock_activations = []
    double_lock_activations = []

    # Find trajectories where we can construct single/double lock patches
    by_level = defaultdict(list)
    for t in trajs:
        by_level[t.level_seed].append(t)

    model.eval()
    n_tested = 0

    with torch.no_grad():
        for ls, level_trajs in by_level.items():
            if n_tested >= 64:
                break

            # Find a trajectory where both locks are open and one where neither is
            open_trajs = [t for t in level_trajs
                          if any(s.lock_open[0] and s.lock_open[1] for s in t.states)]
            closed_trajs = [t for t in level_trajs
                            if any(not s.lock_open[0] and not s.lock_open[1] for s in t.states)]

            if not open_trajs or not closed_trajs:
                continue

            donor = rng.choice(open_trajs)
            recipient = rng.choice(closed_trajs)

            # Find steps where locks are open (donor) / closed (recipient)
            d_step = None
            for si, s in enumerate(donor.states):
                if s.lock_open[0] and s.lock_open[1] and si < len(donor.actions):
                    d_step = si
                    break
            r_step = None
            for si, s in enumerate(recipient.states):
                if not s.lock_open[0] and not s.lock_open[1] and si < len(recipient.actions):
                    r_step = si
                    break

            if d_step is None or r_step is None or d_step < 1 or r_step < 1:
                continue

            donor_obs = torch.from_numpy(donor.obs_encoded).unsqueeze(0)
            recip_obs = torch.from_numpy(recipient.obs_encoded).unsqueeze(0)
            donor_acts = torch.tensor(donor.actions, dtype=torch.long).unsqueeze(0)
            recip_acts = torch.tensor(recipient.actions, dtype=torch.long).unsqueeze(0)

            _, _, d_hidden = model(donor_obs[:, :d_step+1], donor_acts[:, :d_step])
            _, _, r_hidden = model(recip_obs[:, :r_step+1], recip_acts[:, :r_step])

            if d_hidden is None or r_hidden is None:
                continue

            lock0_slot = _find_target_slot_from_map(donor.slot_maps[d_step], 2, cfg.n_slots)
            lock1_slot = _find_target_slot_from_map(donor.slot_maps[d_step], 3, cfg.n_slots)
            goal_slot = _find_target_slot_from_map(donor.slot_maps[d_step], 4, cfg.n_slots)

            if lock0_slot is None or lock1_slot is None or goal_slot is None:
                continue

            # Single lock 0 patch
            h_single0 = r_hidden.clone()
            h_single0[0, lock0_slot] = d_hidden[0, lock0_slot]

            # Single lock 1 patch
            h_single1 = r_hidden.clone()
            h_single1[0, lock1_slot] = d_hidden[0, lock1_slot]

            # Double lock patch
            h_double = r_hidden.clone()
            h_double[0, lock0_slot] = d_hidden[0, lock0_slot]
            h_double[0, lock1_slot] = d_hidden[0, lock1_slot]

            # Run forward and check if goal activates
            for h, label in [(h_single0, "single0"), (h_single1, "single1"), (h_double, "double")]:
                remaining = min(8, len(donor.actions) - d_step)
                activated = False
                h_curr = h.clone()
                for si in range(remaining):
                    act_idx = d_step + si
                    if act_idx >= len(donor.actions) or act_idx >= donor.obs_encoded.shape[0]:
                        break
                    obs_t = torch.from_numpy(donor.obs_encoded[act_idx]).unsqueeze(0)
                    act_t = torch.tensor([donor.actions[act_idx]], dtype=torch.long)
                    _, ep, h_curr = model.forward_step_with_hidden(obs_t, act_t, h_curr)
                    if ep.argmax(dim=-1).item() == 2:  # activate event
                        activated = True
                        break

                if label.startswith("single"):
                    single_lock_activations.append(float(activated))
                else:
                    double_lock_activations.append(float(activated))

            n_tested += 1

    return {
        "single_lock_activation_rate": float(np.mean(single_lock_activations)) if single_lock_activations else None,
        "double_lock_activation_rate": float(np.mean(double_lock_activations)) if double_lock_activations else None,
        "n_tested": n_tested,
    }


# ---------------------------------------------------------------------------
# Clustered bootstrap
# ---------------------------------------------------------------------------

def clustered_bootstrap_ci(values: list, cluster_ids: list, n_boot: int = 2000,
                           ci: float = 0.95) -> dict:
    """Level-clustered bootstrap confidence interval."""
    if not values:
        return {"mean": None, "ci_lower": None, "ci_upper": None}

    by_cluster = defaultdict(list)
    for v, c in zip(values, cluster_ids):
        by_cluster[c].append(v)

    cluster_means = {c: np.mean(vs) for c, vs in by_cluster.items()}
    clusters = list(cluster_means.keys())

    rng = np.random.RandomState(42)
    boot_means = []
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        boot_means.append(np.mean([cluster_means[c] for c in sampled]))

    boot_means = sorted(boot_means)
    alpha = (1 - ci) / 2
    lo = boot_means[int(alpha * n_boot)]
    hi = boot_means[int((1 - alpha) * n_boot)]

    return {
        "mean": float(np.mean(list(cluster_means.values()))),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "n_clusters": len(clusters),
    }


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------

def evaluate_gates(results: dict, cfg: Config) -> dict:
    """Evaluate all pre-registered gates."""
    gates = {}

    # Eligibility gate
    dense = results.get("dense_slots", {})
    sparse = results.get("sparse_slots", {})
    flat = results.get("flat_gru", {})
    historyless = results.get("historyless", {})
    oracle = results.get("oracle", {})

    oracle_acc = oracle.get("prediction", {}).get("event_macro_f1", 0)
    dense_event = dense.get("prediction", {}).get("event_macro_f1", 0)
    dense_status = dense.get("prediction", {}).get("status_macro_f1", 0)
    sparse_event = sparse.get("prediction", {}).get("event_macro_f1", 0)
    sparse_status = sparse.get("prediction", {}).get("status_macro_f1", 0)
    flat_event = flat.get("prediction", {}).get("event_macro_f1", 0)
    hist_event = historyless.get("prediction", {}).get("event_macro_f1", 0)

    recurrent_lift = dense_event - hist_event if hist_event else 0

    eligibility = {
        "oracle_acc": oracle_acc,
        "oracle_pass": oracle_acc >= 0.99,
        "dense_event_f1": dense_event,
        "dense_status_f1": dense_status,
        "sparse_event_f1": sparse_event,
        "sparse_status_f1": sparse_status,
        "dense_pass": dense_event >= 0.90 and dense_status >= 0.90,
        "sparse_pass": sparse_event >= 0.90 and sparse_status >= 0.90,
        "recurrent_lift": recurrent_lift,
        "recurrent_lift_pass": recurrent_lift >= 0.10,
        "flat_within_3": abs(flat_event - dense_event) <= 0.03,
        "overall": False,
    }
    eligibility["overall"] = (eligibility["dense_pass"] and eligibility["sparse_pass"]
                              and eligibility["recurrent_lift_pass"])
    gates["eligibility"] = eligibility

    # Intervention gates (only if eligible)
    dense_interv = dense.get("intervention", {})
    if not dense_interv.get("skipped", False):
        cc = dense_interv.get("causal_consumption", [])
        if cc:
            cc_correct = [c["correct"] for c in cc]
            cc_levels = [c["level"] for c in cc]
            cc_boot = clustered_bootstrap_ci(cc_correct, cc_levels, cfg.bootstrap_n, cfg.bootstrap_ci)

            recip_correct = [c["recip_correct"] for c in cc]
            recip_boot = clustered_bootstrap_ci(recip_correct, cc_levels, cfg.bootstrap_n, cfg.bootstrap_ci)

            improvement = (cc_boot["mean"] or 0) - (recip_boot["mean"] or 0)

            gates["causal_consumption"] = {
                "hybrid_accuracy": cc_boot,
                "recipient_accuracy": recip_boot,
                "improvement": improvement,
                "pass": ((cc_boot["mean"] or 0) >= 0.80
                         and (cc_boot["ci_lower"] or 0) >= 0.70
                         and improvement >= 0.30),
            }

        pre = dense_interv.get("pre_contact_effects", [])
        if pre:
            false_rate = float(np.mean(pre))
            gates["shielding"] = {
                "false_effect_rate": false_rate,
                "pass": false_rate <= 0.10,
            }

        timing = dense_interv.get("timing_errors", [])
        if timing:
            gates["timing"] = {
                "median_error": float(np.median(timing)),
                "p90_error": float(np.percentile(timing, 90)),
                "pass": np.median(timing) == 0 and np.percentile(timing, 90) <= 1,
            }

    # Composition gate
    dense_comp = dense.get("composition", {})
    if not dense_comp.get("skipped", False):
        single_rate = dense_comp.get("single_lock_activation_rate")
        double_rate = dense_comp.get("double_lock_activation_rate")
        if single_rate is not None and double_rate is not None:
            gates["composition"] = {
                "single_lock_rate": single_rate,
                "double_lock_rate": double_rate,
                "double_minus_single": double_rate - single_rate,
                "pass": single_rate <= 0.10 and double_rate >= 0.80,
            }

    return gates


# ---------------------------------------------------------------------------
# Oracle training (special case: uses ground-truth state)
# ---------------------------------------------------------------------------

def train_oracle(trajs: list, cfg: Config) -> tuple:
    """Train oracle model on ground-truth states (batched)."""
    model = OracleModel(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rd = _record_dim(cfg)

    # Pre-encode all state-action pairs and targets into tensors
    all_x = []
    all_tgt_obs = []
    all_tgt_evt = []
    for traj in trajs:
        for step in range(len(traj.actions)):
            all_x.append(OracleModel.encode_state(traj.states[step], traj.actions[step], cfg))
            all_tgt_obs.append(torch.from_numpy(traj.obs_encoded[step + 1]))
            all_tgt_evt.append(traj.events[step][0])

    X = torch.stack(all_x).to(cfg.device)
    T_obs = torch.stack(all_tgt_obs).to(cfg.device)
    T_evt = torch.tensor(all_tgt_evt, dtype=torch.long).to(cfg.device)
    n_samples = len(all_x)
    bs = min(256, n_samples)

    for epoch in range(40):
        model.train()
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n_samples, bs):
            idx = perm[i:i+bs]
            x_b = X[idx]
            tgt_o = T_obs[idx]
            tgt_e = T_evt[idx]

            h = model.net(x_b)
            op = model.obs_head(h).reshape(-1, cfg.n_slots, rd)
            ep = model.event_head(h)

            vis = tgt_o[:, :, -1:]
            loss = F.mse_loss(op * vis, tgt_o * vis) + F.cross_entropy(ep, tgt_e)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0:
            print(f"  [oracle] Epoch {epoch}  loss={total_loss/max(n_batches,1):.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        h = model.net(X)
        ep = model.event_head(h)
        preds = ep.argmax(dim=-1)
        correct = (preds == T_evt).sum().item()

    acc = correct / max(n_samples, 1)
    return model, {"event_macro_f1": acc, "n_samples": n_samples}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_seed(cfg: Config, model_seed: int) -> dict:
    """Run full HANDLE-μ evaluation for one model seed."""
    print(f"\n{'='*60}")
    print(f"HANDLE-mu  seed={model_seed}  staircase_rung={cfg.staircase_rung}")
    print(f"{'='*60}")

    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    random.seed(model_seed)

    # Generate data
    print("\nGenerating trajectories...")
    t0 = time.time()
    train_levels = list(range(cfg.n_train_levels))
    val_levels = list(range(1000, 1000 + cfg.n_val_levels))
    test_levels = list(range(2000, 2000 + cfg.n_test_levels))

    train_trajs = generate_trajectories(train_levels, cfg, data_rng_seed=model_seed)
    val_trajs = generate_trajectories(val_levels, cfg, data_rng_seed=model_seed + 1000)
    test_trajs = generate_trajectories(test_levels, cfg, data_rng_seed=model_seed + 2000)
    print(f"  Generated {len(train_trajs)} train, {len(val_trajs)} val, {len(test_trajs)} test trajectories")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Prepare batches
    print("Preparing batches...")
    train_batches = prepare_batches(train_trajs, cfg)
    val_batches = prepare_batches(val_trajs, cfg, shuffle=False)
    test_batches = prepare_batches(test_trajs, cfg, shuffle=False)

    results = {"seed": model_seed, "config": asdict(cfg)}

    # Train oracle
    print("\nTraining oracle...")
    oracle_model, oracle_pred = train_oracle(train_trajs[:512], cfg)
    results["oracle"] = {"prediction": oracle_pred}
    print(f"  Oracle event accuracy: {oracle_pred['event_macro_f1']:.4f}")

    # Train and evaluate each model
    model_configs = [
        ("dense_slots", DenseSlotModel),
        ("sparse_slots", SparseSlotModel),
        ("flat_gru", FlatGRUModel),
        ("historyless", HistorylessModel),
    ]

    for model_name, ModelClass in model_configs:
        print(f"\nTraining {model_name}...")
        torch.manual_seed(model_seed)

        if model_name == "historyless":
            model = ModelClass(cfg).to(cfg.device)
        else:
            model = ModelClass(cfg).to(cfg.device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        train_metrics = train_model(model, train_batches, val_batches, cfg, model_name)
        pred_metrics = evaluate_prediction(model, test_batches, cfg)

        model_results = {
            "n_params": n_params,
            "training": train_metrics,
            "prediction": pred_metrics,
        }

        # Intervention tests (only for slot models)
        if model_name in ("dense_slots", "sparse_slots"):
            print(f"  Running intervention tests for {model_name}...")
            interv_rng = random.Random(model_seed + 7777)
            all_interv = {}
            for target_h in range(5):
                pairs = find_paired_histories(test_trajs, cfg, target_h,
                                              cfg.intervention_n_pairs // 5, interv_rng)
                if pairs:
                    interv = evaluate_intervention(model, pairs, cfg, model_name)
                    all_interv[f"handle_{target_h}"] = interv

            # Aggregate
            all_cc = []
            all_pre = []
            all_timing = []
            for h_key, h_res in all_interv.items():
                if isinstance(h_res, dict) and not h_res.get("skipped"):
                    all_cc.extend(h_res.get("causal_consumption", []))
                    all_pre.extend(h_res.get("pre_contact_effects", []))
                    all_timing.extend(h_res.get("timing_errors", []))

            model_results["intervention"] = {
                "causal_consumption": all_cc,
                "pre_contact_effects": all_pre,
                "timing_errors": all_timing,
                "per_handle": all_interv,
                "n_pairs_total": sum(len(h.get("causal_consumption", []))
                                     for h in all_interv.values()
                                     if isinstance(h, dict) and not h.get("skipped")),
            }

            # Composition test
            print(f"  Running composition test for {model_name}...")
            comp = evaluate_composition(model, test_trajs, cfg, model_name)
            model_results["composition"] = comp

        results[model_name] = model_results

    # Evaluate gates
    gates = evaluate_gates(results, cfg)
    results["gates"] = gates

    # Print summary
    print(f"\n{'='*60}")
    print("GATE SUMMARY")
    print(f"{'='*60}")
    for gate_name, gate_val in gates.items():
        if isinstance(gate_val, dict):
            passed = gate_val.get("pass", gate_val.get("overall", "N/A"))
            print(f"  {gate_name}: {'PASS' if passed else 'FAIL'}")
            for k, v in gate_val.items():
                if k != "pass" and k != "overall":
                    if isinstance(v, dict):
                        print(f"    {k}: mean={v.get('mean', 'N/A'):.4f}" if v.get("mean") else f"    {k}: {v}")
                    elif isinstance(v, float):
                        print(f"    {k}: {v:.4f}")
                    else:
                        print(f"    {k}: {v}")

    return results


def main():
    parser = argparse.ArgumentParser(description="HANDLE-μ experiment")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed to run (default: all three)")
    parser.add_argument("--staircase-rung", type=int, default=1,
                        help="Staircase rung (1-6)")
    parser.add_argument("--n-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = Config(
        staircase_rung=args.staircase_rung,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

    seeds = [args.seed] if args.seed else list(cfg.model_seeds)

    all_results = {}
    t_start = time.time()

    for seed in seeds:
        result = run_seed(cfg, seed)
        all_results[str(seed)] = result

        # Save per-seed result
        out_path = RESULTS_DIR / f"seed_{seed}_rung_{cfg.staircase_rung}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")

    elapsed = time.time() - t_start

    # Save combined verdict
    verdict = {
        "experiment": "handle_mu",
        "staircase_rung": cfg.staircase_rung,
        "seeds": seeds,
        "elapsed_seconds": elapsed,
        "per_seed": all_results,
    }

    verdict_path = RESULTS_DIR / f"verdict_rung_{cfg.staircase_rung}.json"
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2, default=str)
    print(f"\nVerdict saved: {verdict_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
