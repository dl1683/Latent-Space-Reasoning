"""
FBA-0 architectures — proper recurrent bottleneck with non-isomorphic controls.

Key design: z_t IS the GRU hidden state. No wider transient state.
For FBA, z_t is split into two independent GRU-16 cells.

1. FBAAgent: Two INDEPENDENT GRU-16 cells (factored, no cross-half flow)
2. FlatAgent: One GRU-32 cell (unfactored, full cross-dim mixing)
3. CrossFBAAgent: Two GRU-16 cells with BIDIRECTIONAL connections
4. ReactiveAgent: No recurrence (memoryless baseline)

All agents include a delta prediction head for movement-direction auxiliary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from experiments.env_cnw import obs_dim, N_ACTIONS


D_OBS = 64
D_ACT = 32
D_GOAL = 8
D_INPUT = D_OBS + D_ACT + D_GOAL  # 104
D_HALF = 16
D_BN = D_HALF * 2  # 32
D_EXPAND = 128


class FBAAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_enc = nn.Linear(obs_dim(), D_OBS)
        self.act_enc = nn.Embedding(N_ACTIONS, D_ACT)
        self.goal_enc = nn.Embedding(3, D_GOAL)
        self.gru_a = nn.GRUCell(D_INPUT, D_HALF)
        self.gru_b = nn.GRUCell(D_INPUT, D_HALF)
        self.expand = nn.Linear(D_BN, D_EXPAND)
        self.policy_head = nn.Linear(D_EXPAND, N_ACTIONS)
        self.pred_head = nn.Linear(D_EXPAND, obs_dim())
        self.value_head = nn.Linear(D_EXPAND, 1)
        self.delta_head = nn.Linear(D_EXPAND, N_ACTIONS)

    def initial_state(self, batch_size, device):
        return {
            "za": torch.zeros(batch_size, D_HALF, device=device),
            "zb": torch.zeros(batch_size, D_HALF, device=device),
        }

    def forward(self, obs, action, goal, state, training=True):
        obs_e = F.relu(self.obs_enc(obs))
        act_e = self.act_enc(action)
        goal_e = self.goal_enc(goal)
        x = torch.cat([obs_e, act_e, goal_e], dim=-1)

        za = self.gru_a(x, state["za"])
        zb = self.gru_b(x, state["zb"])
        z = torch.cat([za, zb], dim=-1)

        h = F.relu(self.expand(z))
        logits = self.policy_head(h)
        pred = self.pred_head(h)
        value = self.value_head(h).squeeze(-1)
        delta_logits = self.delta_head(h)
        new_state = {"za": za, "zb": zb}
        return logits, pred, value, new_state, {"place": za, "fiber": zb, "delta_logits": delta_logits}

    def arch_name(self):
        return "fba"


class FlatAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_enc = nn.Linear(obs_dim(), D_OBS)
        self.act_enc = nn.Embedding(N_ACTIONS, D_ACT)
        self.goal_enc = nn.Embedding(3, D_GOAL)
        self.gru = nn.GRUCell(D_INPUT, D_BN)
        self.expand = nn.Linear(D_BN, D_EXPAND)
        self.policy_head = nn.Linear(D_EXPAND, N_ACTIONS)
        self.pred_head = nn.Linear(D_EXPAND, obs_dim())
        self.value_head = nn.Linear(D_EXPAND, 1)
        self.delta_head = nn.Linear(D_EXPAND, N_ACTIONS)

    def initial_state(self, batch_size, device):
        return {"z": torch.zeros(batch_size, D_BN, device=device)}

    def forward(self, obs, action, goal, state, training=True):
        obs_e = F.relu(self.obs_enc(obs))
        act_e = self.act_enc(action)
        goal_e = self.goal_enc(goal)
        x = torch.cat([obs_e, act_e, goal_e], dim=-1)

        z = self.gru(x, state["z"])

        h = F.relu(self.expand(z))
        logits = self.policy_head(h)
        pred = self.pred_head(h)
        value = self.value_head(h).squeeze(-1)
        delta_logits = self.delta_head(h)
        new_state = {"z": z}
        return logits, pred, value, new_state, {"delta_logits": delta_logits}

    def arch_name(self):
        return "flat"


class CrossFBAAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_enc = nn.Linear(obs_dim(), D_OBS)
        self.act_enc = nn.Embedding(N_ACTIONS, D_ACT)
        self.goal_enc = nn.Embedding(3, D_GOAL)
        self.gru_a = nn.GRUCell(D_INPUT + D_HALF, D_HALF)
        self.gru_b = nn.GRUCell(D_INPUT + D_HALF, D_HALF)
        self.expand = nn.Linear(D_BN, D_EXPAND)
        self.policy_head = nn.Linear(D_EXPAND, N_ACTIONS)
        self.pred_head = nn.Linear(D_EXPAND, obs_dim())
        self.value_head = nn.Linear(D_EXPAND, 1)
        self.delta_head = nn.Linear(D_EXPAND, N_ACTIONS)

    def initial_state(self, batch_size, device):
        return {
            "za": torch.zeros(batch_size, D_HALF, device=device),
            "zb": torch.zeros(batch_size, D_HALF, device=device),
        }

    def forward(self, obs, action, goal, state, training=True):
        obs_e = F.relu(self.obs_enc(obs))
        act_e = self.act_enc(action)
        goal_e = self.goal_enc(goal)
        x = torch.cat([obs_e, act_e, goal_e], dim=-1)

        x_a = torch.cat([x, state["zb"]], dim=-1)
        x_b = torch.cat([x, state["za"]], dim=-1)

        za = self.gru_a(x_a, state["za"])
        zb = self.gru_b(x_b, state["zb"])
        z = torch.cat([za, zb], dim=-1)

        h = F.relu(self.expand(z))
        logits = self.policy_head(h)
        pred = self.pred_head(h)
        value = self.value_head(h).squeeze(-1)
        delta_logits = self.delta_head(h)
        new_state = {"za": za, "zb": zb}
        return logits, pred, value, new_state, {"place": za, "fiber": zb, "delta_logits": delta_logits}

    def arch_name(self):
        return "cross"


class ReactiveAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_enc = nn.Linear(obs_dim(), D_OBS)
        self.goal_enc = nn.Embedding(3, D_GOAL)
        self.hidden = nn.Linear(D_OBS + D_GOAL, D_EXPAND)
        self.policy_head = nn.Linear(D_EXPAND, N_ACTIONS)
        self.pred_head = nn.Linear(D_EXPAND, obs_dim())
        self.value_head = nn.Linear(D_EXPAND, 1)
        self.delta_head = nn.Linear(D_EXPAND, N_ACTIONS)

    def initial_state(self, batch_size, device):
        return {}

    def forward(self, obs, action, goal, state, training=True):
        obs_e = F.relu(self.obs_enc(obs))
        goal_e = self.goal_enc(goal)
        h = F.relu(self.hidden(torch.cat([obs_e, goal_e], dim=-1)))
        logits = self.policy_head(h)
        pred = self.pred_head(h)
        value = self.value_head(h).squeeze(-1)
        delta_logits = self.delta_head(h)
        return logits, pred, value, {}, {"delta_logits": delta_logits}

    def arch_name(self):
        return "reactive"


def make_agent(arch_name, **kwargs):
    agents = {
        "fba": FBAAgent,
        "flat": FlatAgent,
        "cross": CrossFBAAgent,
        "reactive": ReactiveAgent,
    }
    return agents[arch_name](**kwargs)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
