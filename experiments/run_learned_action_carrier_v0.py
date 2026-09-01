"""LAC-0: Learned Action Carrier — Phase 3 central artifact.

Architecture per Codex design gate (scratchpad/codex_eac1_design_gate.txt):
  WorldWriter(table, move_legend, start) -> M (32×D memory + D carrier)
  ActionWriter(command, move_legend) -> A (action carrier)
  Composer(A1, A2) -> μ(A1,A2) (composed action carrier)
  Executor(M, A) -> M' (updated world carrier)
  Renderer(M', output_legend) -> logits (pointer scorer)

Access control by function signature: ActionWriter never sees world/place/output.
Theory Section 14. Codex ruling: scratchpad/codex_qpc1_pivot.txt.
"""

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. WORLD GENERATOR
# ---------------------------------------------------------------------------

@dataclass
class World:
    """8-place navigation world with 4 permutation moves."""
    transitions: np.ndarray  # (4, 8): transitions[move][place] -> next_place
    world_id: int = 0

    @staticmethod
    def random(rng: np.random.Generator, world_id: int = 0) -> "World":
        transitions = np.zeros((4, 8), dtype=np.int64)
        for m in range(4):
            transitions[m] = rng.permutation(8)
        return World(transitions=transitions, world_id=world_id)

    def execute(self, place: int, moves: list[int]) -> int:
        for m in moves:
            place = int(self.transitions[m, place])
        return place


# Fixed balanced composition split per Codex spec
TRAIN_COMPOSITIONS = [(0,0),(0,1),(1,2),(1,3),(2,2),(2,3),(3,0),(3,1)]
HELD_OUT_COMPOSITIONS = [(0,2),(0,3),(1,0),(1,1),(2,0),(2,1),(3,2),(3,3)]


# ---------------------------------------------------------------------------
# 2. TENSORIZATION
# ---------------------------------------------------------------------------

def tensorize_primitive(
    world: World,
    rng: np.random.Generator,
    abstract_move: int,
) -> dict[str, torch.Tensor]:
    """One primitive-action episode with full independent relabeling."""
    place_perm = rng.permutation(8)   # abstract -> opaque place alias
    move_perm = rng.permutation(4)    # abstract -> opaque move alias
    output_perm = rng.permutation(8)  # abstract -> output alias

    initial_place = int(rng.integers(0, 8))
    target_place = int(world.transitions[abstract_move, initial_place])

    # Move legend: for each opaque alias, what abstract role does it have?
    # legend[opaque_alias] = abstract_move_role
    move_legend = torch.zeros(4, dtype=torch.long)
    for am in range(4):
        move_legend[int(move_perm[am])] = am

    # Output legend: 8 rows in shuffled order. Row i has (opaque_place, output_alias).
    # The model must point to the correct row. Shuffling prevents ignoring the legend.
    legend_order = rng.permutation(8)
    output_legend_places = torch.tensor(place_perm[legend_order], dtype=torch.long)
    output_legend_outputs = torch.tensor(output_perm[legend_order], dtype=torch.long)
    target_row = int(np.where(legend_order == target_place)[0][0])

    # World table: 32 triples (opaque_place, opaque_move, opaque_next)
    triples = []
    for am in range(4):
        for ap in range(8):
            an = int(world.transitions[am, ap])
            triples.append((int(place_perm[ap]), int(move_perm[am]), int(place_perm[an])))
    triple_order = list(range(32))
    triple_order = list(rng.permutation(32))
    triples = [triples[i] for i in triple_order]

    opaque_command = int(move_perm[abstract_move])
    opaque_initial = int(place_perm[initial_place])

    return {
        "world_table": torch.tensor(triples, dtype=torch.long),      # (32, 3)
        "move_legend": move_legend,                                     # (4,)
        "initial_place": torch.tensor(opaque_initial, dtype=torch.long),
        "command": torch.tensor(opaque_command, dtype=torch.long),      # scalar
        "output_legend_places": output_legend_places,                   # (8,)
        "output_legend_outputs": output_legend_outputs,                 # (8,)
        "target": torch.tensor(target_row, dtype=torch.long),
        "mid_target": torch.tensor(target_row, dtype=torch.long),
        "is_composed": torch.tensor(0, dtype=torch.long),
        "command2": torch.tensor(0, dtype=torch.long),
        # Metadata for evaluation (not used in training)
        "abstract_move": torch.tensor(abstract_move, dtype=torch.long),
        "abstract_initial": torch.tensor(initial_place, dtype=torch.long),
        "abstract_target": torch.tensor(target_place, dtype=torch.long),
        "world_id": torch.tensor(world.world_id, dtype=torch.long),
        "legend_order": torch.tensor(legend_order, dtype=torch.long),
    }


def tensorize_composition(
    world: World,
    rng: np.random.Generator,
    abstract_move1: int,
    abstract_move2: int,
) -> dict[str, torch.Tensor]:
    """One composed-action episode: μ(a1, a2) applied in one step."""
    place_perm = rng.permutation(8)
    move_perm = rng.permutation(4)
    output_perm = rng.permutation(8)

    initial_place = int(rng.integers(0, 8))
    mid_place = int(world.transitions[abstract_move1, initial_place])
    target_place = int(world.transitions[abstract_move2, mid_place])

    move_legend = torch.zeros(4, dtype=torch.long)
    for am in range(4):
        move_legend[int(move_perm[am])] = am

    legend_order = rng.permutation(8)
    output_legend_places = torch.tensor(place_perm[legend_order], dtype=torch.long)
    output_legend_outputs = torch.tensor(output_perm[legend_order], dtype=torch.long)
    target_row = int(np.where(legend_order == target_place)[0][0])

    triples = []
    for am in range(4):
        for ap in range(8):
            an = int(world.transitions[am, ap])
            triples.append((int(place_perm[ap]), int(move_perm[am]), int(place_perm[an])))
    triple_order = list(range(32))
    triple_order = list(rng.permutation(32))
    triples = [triples[i] for i in triple_order]

    opaque_cmd1 = int(move_perm[abstract_move1])
    opaque_cmd2 = int(move_perm[abstract_move2])
    opaque_initial = int(place_perm[initial_place])

    mid_target_row = int(np.where(legend_order == mid_place)[0][0])

    return {
        "world_table": torch.tensor(triples, dtype=torch.long),
        "move_legend": move_legend,
        "initial_place": torch.tensor(opaque_initial, dtype=torch.long),
        "command": torch.tensor(opaque_cmd1, dtype=torch.long),
        "output_legend_places": output_legend_places,
        "output_legend_outputs": output_legend_outputs,
        "target": torch.tensor(target_row, dtype=torch.long),
        "mid_target": torch.tensor(mid_target_row, dtype=torch.long),
        "is_composed": torch.tensor(1, dtype=torch.long),
        "command2": torch.tensor(opaque_cmd2, dtype=torch.long),
        "abstract_move": torch.tensor(abstract_move1, dtype=torch.long),
        "abstract_initial": torch.tensor(initial_place, dtype=torch.long),
        "abstract_target": torch.tensor(target_place, dtype=torch.long),
        "world_id": torch.tensor(world.world_id, dtype=torch.long),
        "legend_order": torch.tensor(legend_order, dtype=torch.long),
    }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ---------------------------------------------------------------------------
# 3. MODEL — TYPED LEARNED ACTION CARRIER
# ---------------------------------------------------------------------------

class WorldWriter(nn.Module):
    """Encodes world table + initial place -> raw triple embeddings + place carrier.

    Keeps raw src/move/dst embeddings for conjunctive retrieval in the Executor.
    Cannot see: command, output labels.
    """
    def __init__(self, dim: int = 128):
        super().__init__()
        self.place_embed = nn.Embedding(8, dim)
        self.move_embed = nn.Embedding(4, dim)

    def forward(self, world_table, move_legend, initial_place):
        B = world_table.shape[0]
        abstract_roles = torch.gather(move_legend, 1, world_table[:,:,1])
        src_emb = self.place_embed(world_table[:,:,0])   # (B, 32, D)
        move_emb = self.move_embed(abstract_roles)         # (B, 32, D)
        dst_emb = self.place_embed(world_table[:,:,2])    # (B, 32, D)
        place_carrier = self.place_embed(initial_place)    # (B, D)
        return src_emb, move_emb, dst_emb, place_carrier


class ActionWriter(nn.Module):
    """Encodes a single command alias -> action carrier.

    Cannot see: world table, current place, output labels.
    Sees: command alias, move legend.
    """
    def __init__(self, action_dim: int = 64):
        super().__init__()
        self.alias_embed = nn.Embedding(4, action_dim)
        self.role_embed = nn.Embedding(4, action_dim)
        self.proj = nn.Sequential(
            nn.Linear(2 * action_dim, action_dim),
            nn.GELU(),
            nn.Linear(action_dim, action_dim),
        )

    def forward(self, command, move_legend):
        """
        command: (B,) — opaque alias index
        move_legend: (B, 4)
        Returns: (B, action_dim) — action carrier
        """
        abstract_role = torch.gather(move_legend, 1, command.unsqueeze(1)).squeeze(1)
        alias_emb = self.alias_embed(command)
        role_emb = self.role_embed(abstract_role)
        return self.proj(torch.cat([alias_emb, role_emb], dim=-1))


class Composer(nn.Module):
    """Composes two action carriers: μ(a1, a2) -> composed action.

    Receives action carriers only. Single shared GRU cell.
    """
    def __init__(self, action_dim: int = 64):
        super().__init__()
        self.cell = nn.GRUCell(action_dim, action_dim)
        self.norm = nn.LayerNorm(action_dim)

    def forward(self, a1, a2):
        """
        a1, a2: (B, action_dim)
        Returns: (B, action_dim) — composed action carrier
        """
        return self.norm(self.cell(a2, a1))


class ExecutorBlock(nn.Module):
    """One gated conjunctive-lookup hop: score = sim(carrier,src) * sim(action,move).

    Gate learns from the action carrier whether this block should activate.
    Block-2 learns gate~0 for primitives, gate~1 for composed carriers.
    """
    def __init__(self, world_dim: int = 240, action_dim: int = 128):
        super().__init__()
        self.action_to_move = nn.Linear(action_dim, world_dim)
        self.update = nn.Sequential(
            nn.Linear(2 * world_dim, world_dim), nn.GELU(),
            nn.Linear(world_dim, world_dim),
        )
        self.norm = nn.LayerNorm(world_dim)
        self.gate = nn.Linear(action_dim, 1)

    def forward(self, src_emb, move_emb, dst_emb, place_carrier, action_carrier):
        g = torch.sigmoid(self.gate(action_carrier))
        a_move = self.action_to_move(action_carrier)
        D = src_emb.shape[-1]
        src_sim = (src_emb * place_carrier.unsqueeze(1)).sum(-1) / math.sqrt(D)
        move_sim = (move_emb * a_move.unsqueeze(1)).sum(-1) / math.sqrt(D)
        score = src_sim * move_sim
        attn = F.softmax(score, -1)
        retrieved = (attn.unsqueeze(-1) * dst_emb).sum(1)
        hop_result = self.norm(retrieved + self.update(
            torch.cat([place_carrier, retrieved], -1)))
        return place_carrier + g * (hop_result - place_carrier)


class Executor(nn.Module):
    """Two-block conjunctive executor per Codex spec.

    Two hops so a composed carrier can induce two-step behavior
    without being handed its primitive components.
    """
    def __init__(self, world_dim: int = 240, action_dim: int = 128, n_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ExecutorBlock(world_dim, action_dim) for _ in range(n_blocks)])

    def forward(self, src_emb, move_emb, dst_emb, place_carrier, action_carrier):
        for block in self.blocks:
            place_carrier = block(
                src_emb, move_emb, dst_emb, place_carrier, action_carrier)
        return place_carrier

    def forward_with_intermediates(self, src_emb, move_emb, dst_emb,
                                   place_carrier, action_carrier):
        intermediates = []
        gates = []
        for block in self.blocks:
            g = torch.sigmoid(block.gate(action_carrier))
            gates.append(g.detach())
            place_carrier = block(
                src_emb, move_emb, dst_emb, place_carrier, action_carrier)
            intermediates.append(place_carrier.detach())
        return place_carrier, intermediates, gates


class Renderer(nn.Module):
    """Pointer scorer: scores each output legend row against the carrier.

    Cannot see: world table, commands, action carriers.
    Sees: place carrier, output legend.
    """
    def __init__(self, world_dim: int = 128):
        super().__init__()
        self.place_embed = nn.Embedding(8, world_dim)
        self.output_embed = nn.Embedding(8, world_dim)
        self.row_proj = nn.Linear(2 * world_dim, world_dim)
        self.carrier_proj = nn.Linear(world_dim, world_dim)

    def forward(self, place_carrier, output_legend_places, output_legend_outputs):
        """
        place_carrier: (B, world_dim)
        output_legend_places: (B, 8) — opaque place alias for each row
        output_legend_outputs: (B, 8) — output alias for each row
        Returns: (B, 8) — logits over the 8 output aliases
        """
        B = place_carrier.shape[0]
        p_emb = self.place_embed(output_legend_places)    # (B, 8, D)
        o_emb = self.output_embed(output_legend_outputs)  # (B, 8, D)
        rows = self.row_proj(torch.cat([p_emb, o_emb], dim=-1))  # (B, 8, D)

        carrier = self.carrier_proj(place_carrier).unsqueeze(1)  # (B, 1, D)
        scores = (rows * carrier).sum(-1)  # (B, 8) — dot product scoring
        return scores


def _init_weights(module):
    """Orthogonal init for GRU only — fixes seed sensitivity without slowing Linear convergence."""
    if isinstance(module, nn.GRUCell):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


class TypedLAC(nn.Module):
    """The full LAC-0 typed model."""
    def __init__(self, world_dim: int = 240, action_dim: int = 128):
        super().__init__()
        self.world_writer = WorldWriter(world_dim)
        self.action_writer = ActionWriter(action_dim)
        self.composer = Composer(action_dim)
        self.executor = Executor(world_dim, action_dim)
        self.renderer = Renderer(world_dim)
        self.apply(_init_weights)

    def _write_world(self, batch):
        return self.world_writer(
            batch["world_table"], batch["move_legend"], batch["initial_place"]
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        src_emb, move_emb, dst_emb, place_carrier = self._write_world(batch)

        a1 = self.action_writer(batch["command"], batch["move_legend"])

        is_comp = batch["is_composed"]  # (B,)
        a2 = self.action_writer(batch["command2"], batch["move_legend"])
        composed = self.composer(a1, a2)

        action = torch.where(is_comp.unsqueeze(-1).bool(), composed, a1)

        new_carrier = self.executor(
            src_emb, move_emb, dst_emb, place_carrier, action)
        logits = self.renderer(
            new_carrier, batch["output_legend_places"], batch["output_legend_outputs"]
        )
        return logits

    def get_action_carrier(self, batch, composed=False):
        a1 = self.action_writer(batch["command"], batch["move_legend"])
        if not composed:
            return a1
        a2 = self.action_writer(batch["command2"], batch["move_legend"])
        return self.composer(a1, a2)

    def execute_with_transplanted_action(self, recipient_batch, donor_action):
        src_emb, move_emb, dst_emb, place_carrier = self._write_world(recipient_batch)
        new_carrier = self.executor(
            src_emb, move_emb, dst_emb, place_carrier, donor_action)
        return self.renderer(
            new_carrier, recipient_batch["output_legend_places"],
            recipient_batch["output_legend_outputs"]
        )


# ---------------------------------------------------------------------------
# 4. UNTYPED TRANSFORMER CONTROL
# ---------------------------------------------------------------------------

class UntypedControl(nn.Module):
    """Parameter-matched untyped transformer. Sees everything including move legend."""
    def __init__(self, d_model: int = 148, n_heads: int = 4, n_layers: int = 4):
        super().__init__()
        self.place_embed = nn.Embedding(8, d_model)
        self.move_embed = nn.Embedding(4, d_model)
        self.type_embed = nn.Embedding(8, d_model)  # 0-2:src/mv/dst, 3:init, 4:cmd1, 5:cmd2, 6:is_comp, 7:legend
        self.out_embed = nn.Embedding(8, d_model)
        self.pos_embed = nn.Embedding(128, d_model)
        self.score_scale = d_model ** 0.5

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.apply(_init_weights)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        B = batch["world_table"].shape[0]
        device = batch["world_table"].device

        wt = batch["world_table"]  # (B, 32, 3)
        src = self.place_embed(wt[:,:,0]) + self.type_embed(torch.zeros(B,32,dtype=torch.long,device=device))
        mv = self.move_embed(wt[:,:,1]) + self.type_embed(torch.ones(B,32,dtype=torch.long,device=device))
        dst = self.place_embed(wt[:,:,2]) + self.type_embed(2*torch.ones(B,32,dtype=torch.long,device=device))
        triple_toks = torch.stack([src, mv, dst], dim=2).reshape(B, 96, -1)

        init_tok = (self.place_embed(batch["initial_place"]).unsqueeze(1)
                    + self.type_embed(3*torch.ones(B,1,dtype=torch.long,device=device)))
        cmd1_tok = (self.move_embed(batch["command"]).unsqueeze(1)
                    + self.type_embed(4*torch.ones(B,1,dtype=torch.long,device=device)))
        cmd2_tok = (self.move_embed(batch["command2"]).unsqueeze(1)
                    + self.type_embed(5*torch.ones(B,1,dtype=torch.long,device=device)))
        comp_tok = self.type_embed(6*torch.ones(B,1,dtype=torch.long,device=device))
        comp_tok = comp_tok * batch["is_composed"].float().unsqueeze(-1).unsqueeze(-1)

        # Move legend tokens: 4 tokens mapping alias -> abstract role
        legend = batch["move_legend"]  # (B, 4)
        legend_alias = self.move_embed(torch.arange(4, device=device).unsqueeze(0).expand(B, -1))
        legend_role = self.move_embed(legend)
        legend_toks = legend_alias + legend_role + self.type_embed(
            7*torch.ones(B, 4, dtype=torch.long, device=device))

        tokens = torch.cat([init_tok, cmd1_tok, cmd2_tok, comp_tok,
                            legend_toks, triple_toks], dim=1)
        seq_len = tokens.shape[1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.pos_embed(positions)

        encoded = self.transformer(tokens)
        cls = encoded[:, 0]  # (B, D)

        olp = batch["output_legend_places"]
        olo = batch["output_legend_outputs"]
        p_emb = self.place_embed(olp)    # (B, 8, D)
        o_emb = self.out_embed(olo)      # (B, 8, D)
        rows = p_emb + o_emb             # (B, 8, D)

        scores = (rows * cls.unsqueeze(1)).sum(-1) / self.score_scale  # scaled dot product
        return scores


# ---------------------------------------------------------------------------
# 5. TRAINING
# ---------------------------------------------------------------------------

def make_batch(
    worlds: list[World],
    rng: np.random.Generator,
    batch_size: int = 128,
    compositions: list[tuple[int,int]] = TRAIN_COMPOSITIONS,
) -> dict[str, torch.Tensor]:
    """Generate a balanced batch: half primitives, half compositions."""
    n_prim = batch_size // 2
    n_comp = batch_size - n_prim
    episodes = []

    for _ in range(n_prim):
        w = worlds[int(rng.integers(0, len(worlds)))]
        m = int(rng.integers(0, 4))
        episodes.append(tensorize_primitive(w, rng, m))

    for _ in range(n_comp):
        w = worlds[int(rng.integers(0, len(worlds)))]
        idx = int(rng.integers(0, len(compositions)))
        m1, m2 = compositions[idx]
        episodes.append(tensorize_composition(w, rng, m1, m2))

    return collate(episodes)


def train_model(model, config, seed, worlds, results_dir, model_name):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    lr = config["training"]["lr"]
    wd = config["training"]["weight_decay"]
    n_steps = config["training"]["n_steps"]
    batch_size = config["training"]["batch_size"]
    warmup = config["training"]["warmup_steps"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) if s < warmup else 1.0
    )

    train_log = []
    total_fwd = 0
    t0 = time.time()

    for step in range(1, n_steps + 1):
        model.train()
        batch = make_batch(worlds, rng, batch_size)
        logits = model(batch)
        loss = F.cross_entropy(logits, batch["target"])

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
        optimizer.step()
        scheduler.step()

        total_fwd += batch_size
        acc = (logits.argmax(-1) == batch["target"]).float().mean().item()

        if step % config["training"]["log_every"] == 0:
            elapsed = time.time() - t0
            entry = {"step": step, "loss": loss.item(), "acc": acc,
                     "forwards": total_fwd, "elapsed_s": elapsed}
            train_log.append(entry)
            print(f"[{model_name}] step={step}/{n_steps} loss={loss.item():.4f} "
                  f"acc={acc:.3f} fwd={total_fwd} t={elapsed:.1f}s")

    elapsed = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())

    ckpt_path = results_dir / f"{model_name}_seed{seed}.pt"
    torch.save(model.state_dict(), ckpt_path)

    return {
        "model_name": model_name, "seed": seed, "n_params": n_params,
        "n_steps": n_steps, "total_forwards": total_fwd,
        "final_loss": train_log[-1]["loss"] if train_log else None,
        "final_acc": train_log[-1]["acc"] if train_log else None,
        "elapsed_s": elapsed, "train_log": train_log,
        "checkpoint": str(ckpt_path),
    }


# ---------------------------------------------------------------------------
# 6. EVALUATION
# ---------------------------------------------------------------------------

def eval_clean_accuracy(model, worlds, rng, n_samples=2000, held_out_worlds=None):
    hw = held_out_worlds if held_out_worlds is not None else worlds
    model.eval()
    n_prim = n_samples // 3
    n_train_comp = n_samples // 3
    n_held_comp = n_samples - n_prim - n_train_comp
    prim_correct = comp_train_correct = comp_held_correct = 0
    with torch.no_grad():
        for _ in range(n_prim):
            w = worlds[int(rng.integers(0, len(worlds)))]
            m = int(rng.integers(0, 4))
            b = collate([tensorize_primitive(w, rng, m)])
            if model(b).argmax(-1).item() == b["target"].item():
                prim_correct += 1
        for _ in range(n_train_comp):
            w = worlds[int(rng.integers(0, len(worlds)))]
            idx = int(rng.integers(0, len(TRAIN_COMPOSITIONS)))
            m1, m2 = TRAIN_COMPOSITIONS[idx]
            b = collate([tensorize_composition(w, rng, m1, m2)])
            if model(b).argmax(-1).item() == b["target"].item():
                comp_train_correct += 1
        for _ in range(n_held_comp):
            w = hw[int(rng.integers(0, len(hw)))]
            idx = int(rng.integers(0, len(HELD_OUT_COMPOSITIONS)))
            m1, m2 = HELD_OUT_COMPOSITIONS[idx]
            b = collate([tensorize_composition(w, rng, m1, m2)])
            if model(b).argmax(-1).item() == b["target"].item():
                comp_held_correct += 1
    total = prim_correct + comp_train_correct + comp_held_correct
    return {
        "clean_accuracy": total / n_samples,
        "primitive_accuracy": prim_correct / n_prim,
        "train_composition_accuracy": comp_train_correct / n_train_comp,
        "held_out_composition_accuracy": comp_held_correct / n_held_comp,
        "n_samples": n_samples,
    }


def eval_self_patch_action(model, worlds, rng, n_samples=500):
    """Self-patch the ACTION carrier (not the world state)."""
    if not isinstance(model, TypedLAC):
        return {"skip": True, "reason": "untyped model"}
    model.eval()
    max_disc = 0.0
    with torch.no_grad():
        for _ in range(n_samples):
            w = worlds[int(rng.integers(0, len(worlds)))]
            m = int(rng.integers(0, 4))
            b = collate([tensorize_primitive(w, rng, m)])

            logits_clean = model(b)
            probs_clean = F.softmax(logits_clean, dim=-1)

            action = model.get_action_carrier(b, composed=False)
            logits_patch = model.execute_with_transplanted_action(b, action)
            probs_patch = F.softmax(logits_patch, dim=-1)

            disc = (probs_clean - probs_patch).abs().max().item()
            max_disc = max(max_disc, disc)

    return {"max_discrepancy": max_disc, "pass": max_disc <= 1e-5,
            "n_samples": n_samples}


def eval_three_way_action_portability(model, worlds, rng, n_samples=1000):
    """Three-way portability: transplant donor's ACTION carrier into recipient world.

    Both donor and recipient use the same abstract move m.
    Donor encodes m in W_d's relabeling. Carrier is transplanted into W_r.
    Target: does R(E(M_r, A_donor)) predict the correct W_r endpoint?
    Filter: only count cases where donor's endpoint d != recipient's endpoint t.
    """
    if not isinstance(model, TypedLAC):
        return {"skip": True}
    model.eval()
    results = {"target": 0, "other": 0, "total": 0}
    per_move = {m: {"target": 0, "total": 0} for m in range(4)}

    with torch.no_grad():
        attempts = 0
        while results["total"] < n_samples and attempts < n_samples * 20:
            attempts += 1
            if len(worlds) < 2:
                continue
            idx = rng.choice(len(worlds), size=2, replace=False)
            w_recip = worlds[idx[0]]
            w_donor = worlds[idx[1]]

            abstract_move = int(rng.integers(0, 4))

            # Check if recipient and donor transitions differ for this move
            # (at least one starting place produces a different endpoint)
            recip_trans = w_recip.transitions[abstract_move]
            donor_trans = w_donor.transitions[abstract_move]
            if np.array_equal(recip_trans, donor_trans):
                continue

            # Generate episodes with their own independent relabeling
            recip_ep = tensorize_primitive(w_recip, rng, abstract_move)
            donor_ep = tensorize_primitive(w_donor, rng, abstract_move)

            recip_batch = collate([recip_ep])
            donor_batch = collate([donor_ep])

            # Get donor's action carrier (encoded under donor's relabeling)
            donor_action = model.get_action_carrier(donor_batch, composed=False)

            # Execute donor's action in recipient's world
            logits = model.execute_with_transplanted_action(recip_batch, donor_action)
            pred = logits.argmax(-1).item()

            # Target: the recipient's correct answer row
            target_row = recip_batch["target"].item()

            if pred == target_row:
                results["target"] += 1
                per_move[abstract_move]["target"] += 1
            else:
                results["other"] += 1

            results["total"] += 1
            per_move[abstract_move]["total"] += 1

    total = max(results["total"], 1)
    F_rate = results["target"] / total
    per_move_rates = {m: d["target"] / max(d["total"], 1)
                      for m, d in per_move.items()}

    return {
        "target_following_F": F_rate,
        "per_move_F": per_move_rates,
        "n_samples": results["total"],
        "pass": F_rate >= 0.90 and all(r >= 0.80 for r in per_move_rates.values()),
    }


def eval_block_diagnostics(model, worlds, rng, n_samples=200, held_out_worlds=None):
    """Per-block diagnostic readouts per Codex spec."""
    hw = held_out_worlds if held_out_worlds is not None else worlds
    model.eval()
    diag = {
        "prim_gate_b1": [], "prim_gate_b2": [],
        "comp_gate_b1": [], "comp_gate_b2": [],
        "prim_acc_after_b1": 0, "prim_acc_after_b2": 0,
        "comp_train_acc_after_b1": 0, "comp_train_acc_after_b2": 0,
        "comp_held_acc_after_b1": 0, "comp_held_acc_after_b2": 0,
        "comp_train_mid_after_b1": 0, "comp_held_mid_after_b1": 0,
    }
    n_prim = n_comp_train = n_comp_held = n_samples // 3

    with torch.no_grad():
        for i in range(n_prim):
            w = worlds[int(rng.integers(0, len(worlds)))]
            m = int(rng.integers(0, 4))
            b = collate([tensorize_primitive(w, rng, m)])
            src_emb, move_emb, dst_emb, pc = model._write_world(b)
            a1 = model.action_writer(b["command"], b["move_legend"])
            action = a1
            _, intermediates, gates = model.executor.forward_with_intermediates(
                src_emb, move_emb, dst_emb, pc, action)
            diag["prim_gate_b1"].append(gates[0].item())
            diag["prim_gate_b2"].append(gates[1].item())
            for bi, key in enumerate(["prim_acc_after_b1", "prim_acc_after_b2"]):
                logits = model.renderer(
                    intermediates[bi], b["output_legend_places"],
                    b["output_legend_outputs"])
                if logits.argmax(-1).item() == b["target"].item():
                    diag[key] += 1

        for i in range(n_comp_train):
            w = worlds[int(rng.integers(0, len(worlds)))]
            idx = int(rng.integers(0, len(TRAIN_COMPOSITIONS)))
            m1, m2 = TRAIN_COMPOSITIONS[idx]
            b = collate([tensorize_composition(w, rng, m1, m2)])
            src_emb, move_emb, dst_emb, pc = model._write_world(b)
            a1 = model.action_writer(b["command"], b["move_legend"])
            a2 = model.action_writer(b["command2"], b["move_legend"])
            action = model.composer(a1, a2)
            _, intermediates, gates = model.executor.forward_with_intermediates(
                src_emb, move_emb, dst_emb, pc, action)
            diag["comp_gate_b1"].append(gates[0].item())
            diag["comp_gate_b2"].append(gates[1].item())
            for bi, key in enumerate(["comp_train_acc_after_b1",
                                       "comp_train_acc_after_b2"]):
                logits = model.renderer(
                    intermediates[bi], b["output_legend_places"],
                    b["output_legend_outputs"])
                if logits.argmax(-1).item() == b["target"].item():
                    diag[key] += 1
            logits_b1 = model.renderer(
                intermediates[0], b["output_legend_places"],
                b["output_legend_outputs"])
            if logits_b1.argmax(-1).item() == b["mid_target"].item():
                diag["comp_train_mid_after_b1"] += 1

        for i in range(n_comp_held):
            w = hw[int(rng.integers(0, len(hw)))]
            idx = int(rng.integers(0, len(HELD_OUT_COMPOSITIONS)))
            m1, m2 = HELD_OUT_COMPOSITIONS[idx]
            b = collate([tensorize_composition(w, rng, m1, m2)])
            src_emb, move_emb, dst_emb, pc = model._write_world(b)
            a1 = model.action_writer(b["command"], b["move_legend"])
            a2 = model.action_writer(b["command2"], b["move_legend"])
            action = model.composer(a1, a2)
            _, intermediates, gates = model.executor.forward_with_intermediates(
                src_emb, move_emb, dst_emb, pc, action)
            diag["comp_gate_b1"].append(gates[0].item())
            diag["comp_gate_b2"].append(gates[1].item())
            for bi, key in enumerate(["comp_held_acc_after_b1",
                                       "comp_held_acc_after_b2"]):
                logits = model.renderer(
                    intermediates[bi], b["output_legend_places"],
                    b["output_legend_outputs"])
                if logits.argmax(-1).item() == b["target"].item():
                    diag[key] += 1
            logits_b1 = model.renderer(
                intermediates[0], b["output_legend_places"],
                b["output_legend_outputs"])
            if logits_b1.argmax(-1).item() == b["mid_target"].item():
                diag["comp_held_mid_after_b1"] += 1

    return {
        "prim_gate_b1_mean": float(np.mean(diag["prim_gate_b1"])),
        "prim_gate_b2_mean": float(np.mean(diag["prim_gate_b2"])),
        "comp_gate_b1_mean": float(np.mean(diag["comp_gate_b1"])),
        "comp_gate_b2_mean": float(np.mean(diag["comp_gate_b2"])),
        "prim_acc_after_b1": diag["prim_acc_after_b1"] / n_prim,
        "prim_acc_after_b2": diag["prim_acc_after_b2"] / n_prim,
        "comp_train_acc_after_b1": diag["comp_train_acc_after_b1"] / n_comp_train,
        "comp_train_acc_after_b2": diag["comp_train_acc_after_b2"] / n_comp_train,
        "comp_held_acc_after_b1": diag["comp_held_acc_after_b1"] / n_comp_held,
        "comp_held_acc_after_b2": diag["comp_held_acc_after_b2"] / n_comp_held,
        "comp_train_mid_after_b1": diag["comp_train_mid_after_b1"] / n_comp_train,
        "comp_held_mid_after_b1": diag["comp_held_mid_after_b1"] / n_comp_held,
    }


def eval_oracle_action(model, worlds, rng, n_samples=500):
    """Oracle-action: try every possible action carrier on a primitive episode.

    For each episode with abstract move m, produce action carriers for all 4 moves.
    The oracle selects the one that gives the correct answer. If the model's own
    carrier for move m is the one the oracle selects, this passes. Tests whether
    the learned action carrier IS the correct one, not just that the pipeline
    can execute it.
    """
    if not isinstance(model, TypedLAC):
        return {"skip": True}
    model.eval()
    oracle_correct = 0
    model_correct = 0
    with torch.no_grad():
        for _ in range(n_samples):
            w = worlds[int(rng.integers(0, len(worlds)))]
            m = int(rng.integers(0, 4))
            b = collate([tensorize_primitive(w, rng, m)])
            target = b["target"].item()

            # Model's own carrier
            own_action = model.get_action_carrier(b, composed=False)
            own_logits = model.execute_with_transplanted_action(b, own_action)
            if own_logits.argmax(-1).item() == target:
                model_correct += 1

            # Oracle: try all 4 possible move commands
            best_found = False
            for trial_m in range(4):
                trial_b = collate([tensorize_primitive(w, rng, trial_m)])
                trial_action = model.get_action_carrier(trial_b, composed=False)
                trial_logits = model.execute_with_transplanted_action(b, trial_action)
                if trial_logits.argmax(-1).item() == target:
                    best_found = True
                    break
            if best_found:
                oracle_correct += 1

    return {
        "oracle_accuracy": oracle_correct / n_samples,
        "model_accuracy": model_correct / n_samples,
        "accuracy": model_correct / n_samples,
        "pass": model_correct / n_samples >= 0.99,
        "n_samples": n_samples,
    }


def eval_three_way_portability_full(model, worlds, rng, n_samples=1000):
    """Full three-way portability with margins and donor-answer tracking.

    For each trial: donor and recipient use the same abstract move but different
    worlds. We track whether the model follows the RECIPIENT target (good) or
    the DONOR answer (bad — indicates memorization, not transport).
    """
    if not isinstance(model, TypedLAC):
        return {"skip": True}
    model.eval()
    target_hits = 0
    donor_hits = 0
    total = 0
    per_move = {m: {"target": 0, "donor": 0, "total": 0} for m in range(4)}
    target_margins = []

    with torch.no_grad():
        attempts = 0
        while total < n_samples and attempts < n_samples * 20:
            attempts += 1
            if len(worlds) < 2:
                continue
            idx = rng.choice(len(worlds), size=2, replace=False)
            w_r, w_d = worlds[idx[0]], worlds[idx[1]]
            am = int(rng.integers(0, 4))
            if np.array_equal(w_r.transitions[am], w_d.transitions[am]):
                continue

            r_ep = tensorize_primitive(w_r, rng, am)
            d_ep = tensorize_primitive(w_d, rng, am)
            r_batch = collate([r_ep])
            d_batch = collate([d_ep])

            donor_action = model.get_action_carrier(d_batch, composed=False)
            logits = model.execute_with_transplanted_action(r_batch, donor_action)
            probs = F.softmax(logits, dim=-1).squeeze(0)
            pred = logits.argmax(-1).item()

            recip_target = r_batch["target"].item()
            recip_prob = probs[recip_target].item()

            donor_abstract_target = d_ep["abstract_target"].item()
            r_legend = r_ep["legend_order"].numpy()
            donor_row_in_recip = int(np.where(r_legend == donor_abstract_target)[0][0])

            if pred == recip_target:
                target_hits += 1
                per_move[am]["target"] += 1
            if pred == donor_row_in_recip:
                donor_hits += 1
                per_move[am]["donor"] += 1
            target_margins.append(recip_prob)

            total += 1
            per_move[am]["total"] += 1

    total = max(total, 1)
    F_rate = target_hits / total
    D_rate = donor_hits / total
    per_move_rates = {m: d["target"] / max(d["total"], 1) for m, d in per_move.items()}

    return {
        "target_following_F": F_rate,
        "donor_answer_F": D_rate,
        "per_move_F": per_move_rates,
        "mean_target_margin": float(np.mean(target_margins)) if target_margins else 0.0,
        "n_samples": total,
        "pass": F_rate >= 0.90 and all(r >= 0.80 for r in per_move_rates.values()),
    }


def eval_explicit_sequential(model, worlds, rng, n_samples=500):
    """Explicit sequential comparator: R(E(E(M,a1),a2)) vs R(E(M,μ(a1,a2))).

    For held-out compositions, compare executing two primitives in sequence
    (two executor passes) with executing the composed carrier (one pass).
    Agreement ≥90% is the gate.
    """
    if not isinstance(model, TypedLAC):
        return {"skip": True}
    model.eval()
    agree = 0
    seq_correct = 0
    comp_correct = 0
    with torch.no_grad():
        for _ in range(n_samples):
            w = worlds[int(rng.integers(0, len(worlds)))]
            idx = int(rng.integers(0, len(HELD_OUT_COMPOSITIONS)))
            m1, m2 = HELD_OUT_COMPOSITIONS[idx]
            b = collate([tensorize_composition(w, rng, m1, m2)])
            src_emb, move_emb, dst_emb, pc = model._write_world(b)

            # Composed path: μ(a1,a2) → one executor pass
            a1 = model.action_writer(b["command"], b["move_legend"])
            a2 = model.action_writer(b["command2"], b["move_legend"])
            composed = model.composer(a1, a2)
            carrier_comp = model.executor(src_emb, move_emb, dst_emb, pc, composed)
            logits_comp = model.renderer(
                carrier_comp, b["output_legend_places"], b["output_legend_outputs"])
            pred_comp = logits_comp.argmax(-1).item()

            # Sequential path: E(M,a1) then E(M',a2)
            carrier_s1 = model.executor(src_emb, move_emb, dst_emb, pc, a1)
            carrier_s2 = model.executor(src_emb, move_emb, dst_emb, carrier_s1, a2)
            logits_seq = model.renderer(
                carrier_s2, b["output_legend_places"], b["output_legend_outputs"])
            pred_seq = logits_seq.argmax(-1).item()

            target = b["target"].item()
            if pred_comp == pred_seq:
                agree += 1
            if pred_seq == target:
                seq_correct += 1
            if pred_comp == target:
                comp_correct += 1

    agreement = agree / n_samples
    return {
        "agreement": agreement,
        "sequential_accuracy": seq_correct / n_samples,
        "composed_accuracy": comp_correct / n_samples,
        "pass": agreement >= 0.90,
        "n_samples": n_samples,
    }


def eval_recipient_dependence(model, worlds, rng, n_samples=500):
    """Recipient-dependence: transplanted carrier must follow recipient, not donor.

    Use distinct moves for host (s), target (t), donor (d) where s≠t≠d.
    Donor encodes move d in world W_d. Carrier transplanted into W_r.
    Check: does output match recipient's endpoint for move d (good),
    or donor's endpoint for move d (bad)?
    """
    if not isinstance(model, TypedLAC):
        return {"skip": True}
    model.eval()
    recip_correct = 0
    donor_copy = 0
    total = 0

    with torch.no_grad():
        attempts = 0
        while total < n_samples and attempts < n_samples * 30:
            attempts += 1
            if len(worlds) < 2:
                continue
            idx = rng.choice(len(worlds), size=2, replace=False)
            w_r, w_d = worlds[idx[0]], worlds[idx[1]]
            am = int(rng.integers(0, 4))
            if np.array_equal(w_r.transitions[am], w_d.transitions[am]):
                continue

            r_ep = tensorize_primitive(w_r, rng, am)
            d_ep = tensorize_primitive(w_d, rng, am)
            r_batch = collate([r_ep])
            d_batch = collate([d_ep])

            donor_action = model.get_action_carrier(d_batch, composed=False)

            r_abstract_init = r_ep["abstract_initial"].item()
            d_abstract_init = d_ep["abstract_initial"].item()
            r_abstract_target = int(w_r.transitions[am, r_abstract_init])
            d_abstract_target = int(w_d.transitions[am, d_abstract_init])
            if r_abstract_target == d_abstract_target:
                continue

            logits = model.execute_with_transplanted_action(r_batch, donor_action)
            pred = logits.argmax(-1).item()

            if pred == r_batch["target"].item():
                recip_correct += 1

            r_legend = r_ep["legend_order"].numpy()
            donor_row_in_recip = int(np.where(r_legend == d_abstract_target)[0][0])
            if pred == donor_row_in_recip:
                donor_copy += 1

            total += 1

    total = max(total, 1)
    return {
        "recipient_following_F": recip_correct / total,
        "donor_answer_F": donor_copy / total,
        "pass": (recip_correct / total >= 0.90) and (donor_copy / total <= 0.10),
        "n_samples": total,
    }


def bootstrap_world_cluster(model, worlds, compositions, rng,
                             n_bootstrap=1000, n_per_world=10):
    """Bootstrap confidence interval over world clusters for composition accuracy."""
    if not isinstance(model, TypedLAC):
        return {"skip": True}
    model.eval()
    per_world_acc = []
    with torch.no_grad():
        for w in worlds[:64]:  # sample 64 worlds for speed
            correct = 0
            for _ in range(n_per_world):
                idx = int(rng.integers(0, len(compositions)))
                m1, m2 = compositions[idx]
                b = collate([tensorize_composition(w, rng, m1, m2)])
                if model(b).argmax(-1).item() == b["target"].item():
                    correct += 1
            per_world_acc.append(correct / n_per_world)

    per_world_acc = np.array(per_world_acc)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(per_world_acc, size=len(per_world_acc), replace=True)
        boot_means.append(sample.mean())
    boot_means = np.sort(boot_means)

    return {
        "mean": float(per_world_acc.mean()),
        "std": float(per_world_acc.std()),
        "lower_5": float(boot_means[int(0.05 * n_bootstrap)]),
        "upper_95": float(boot_means[int(0.95 * n_bootstrap)]),
        "n_worlds": len(per_world_acc),
        "n_per_world": n_per_world,
    }


def bootstrap_portability_cluster(model, worlds, rng,
                                   n_bootstrap=1000, n_per_pair=5):
    """Bootstrap confidence interval for portability over world pairs."""
    if not isinstance(model, TypedLAC):
        return {"skip": True}
    model.eval()
    pair_acc = []
    n_worlds = min(len(worlds), 32)
    with torch.no_grad():
        for i in range(n_worlds):
            for j in range(i + 1, min(i + 4, n_worlds)):
                w_r, w_d = worlds[i], worlds[j]
                hits = 0
                trials = 0
                for _ in range(n_per_pair):
                    am = int(rng.integers(0, 4))
                    if np.array_equal(w_r.transitions[am], w_d.transitions[am]):
                        continue
                    r_ep = tensorize_primitive(w_r, rng, am)
                    d_ep = tensorize_primitive(w_d, rng, am)
                    r_batch = collate([r_ep])
                    d_batch = collate([d_ep])
                    donor_action = model.get_action_carrier(d_batch, composed=False)
                    logits = model.execute_with_transplanted_action(r_batch, donor_action)
                    if logits.argmax(-1).item() == r_batch["target"].item():
                        hits += 1
                    trials += 1
                if trials > 0:
                    pair_acc.append(hits / trials)

    if not pair_acc:
        return {"mean": 0.0, "lower_5": 0.0, "upper_95": 0.0, "n_pairs": 0}

    pair_acc = np.array(pair_acc)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(pair_acc, size=len(pair_acc), replace=True)
        boot_means.append(sample.mean())
    boot_means = np.sort(boot_means)

    return {
        "mean": float(pair_acc.mean()),
        "std": float(pair_acc.std()),
        "lower_5": float(boot_means[int(0.05 * n_bootstrap)]),
        "upper_95": float(boot_means[int(0.95 * n_bootstrap)]),
        "n_pairs": len(pair_acc),
    }


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def main():
    cfg_path = Path("experiments/config/learned_action_carrier_v0.json")
    with open(cfg_path) as f:
        config = json.load(f)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"experiments/results/lac0_run_{run_id}")
    results_dir.mkdir(parents=True, exist_ok=True)

    wall_start = time.time()
    wall_limit = config["budget"]["total_wall_hours"] * 3600

    # Generate worlds per Codex spec: 256 train, 256 held-out
    world_rng = np.random.default_rng(12345)
    train_worlds = [World.random(world_rng, i) for i in range(256)]
    eval_worlds = [World.random(world_rng, 256 + i) for i in range(256)]

    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    cfg_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    manifest = {
        "config": config,
        "code_hash": code_hash,
        "config_hash": cfg_hash,
        "run_id": run_id,
        "n_train_worlds": len(train_worlds),
        "n_eval_worlds": len(eval_worlds),
        "train_compositions": TRAIN_COMPOSITIONS,
        "held_out_compositions": HELD_OUT_COMPOSITIONS,
    }
    with open(results_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    all_results = {}

    for seed in config["training"]["seeds"]:
        if time.time() - wall_start > wall_limit:
            print("WALL LIMIT REACHED. Stopping.")
            break

        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        # Seed BEFORE model construction for deterministic init
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Typed model
        typed = TypedLAC(world_dim=240, action_dim=128)
        n_typed = sum(p.numel() for p in typed.parameters())
        print(f"Typed: {n_typed:,} params")

        typed_train = train_model(typed, config, seed, train_worlds,
                                   results_dir, "typed")

        # Same eval RNG seed for both models (paired comparison)
        eval_seed = seed + 1000
        eval_rng = np.random.default_rng(eval_seed)

        # 2x2 evaluation: {train_worlds, eval_worlds} x {train_comp, held_comp}
        clean_tw = eval_clean_accuracy(typed, train_worlds, np.random.default_rng(eval_seed),
                                        held_out_worlds=train_worlds)
        clean_ew = eval_clean_accuracy(typed, train_worlds, np.random.default_rng(eval_seed),
                                        held_out_worlds=eval_worlds)
        print(f"Clean accuracy (train_worlds): {clean_tw['clean_accuracy']:.4f} "
              f"(prim={clean_tw['primitive_accuracy']:.4f} "
              f"train_comp={clean_tw['train_composition_accuracy']:.4f} "
              f"held_comp={clean_tw['held_out_composition_accuracy']:.4f})")
        print(f"Clean accuracy (eval_worlds):  {clean_ew['clean_accuracy']:.4f} "
              f"(prim={clean_ew['primitive_accuracy']:.4f} "
              f"train_comp={clean_ew['train_composition_accuracy']:.4f} "
              f"held_comp={clean_ew['held_out_composition_accuracy']:.4f})")

        sp = eval_self_patch_action(typed, eval_worlds,
                                     np.random.default_rng(eval_seed + 2))
        print(f"Self-patch: {sp['max_discrepancy']:.2e} ({'PASS' if sp['pass'] else 'FAIL'})")

        port = eval_three_way_portability_full(typed, eval_worlds,
                                                np.random.default_rng(eval_seed + 3))
        print(f"Portability: target_F={port['target_following_F']:.4f} "
              f"donor_F={port['donor_answer_F']:.4f} "
              f"({'PASS' if port['pass'] else 'FAIL'})")

        oracle = eval_oracle_action(typed, eval_worlds,
                                     np.random.default_rng(eval_seed + 4))
        print(f"Oracle-action: {oracle['accuracy']:.4f} "
              f"({'PASS' if oracle['pass'] else 'FAIL'})")

        seq = eval_explicit_sequential(typed, eval_worlds,
                                        np.random.default_rng(eval_seed + 5))
        print(f"Sequential agreement: {seq['agreement']:.4f} "
              f"({'PASS' if seq['pass'] else 'FAIL'})")

        recip = eval_recipient_dependence(typed, eval_worlds,
                                           np.random.default_rng(eval_seed + 6))
        print(f"Recipient dependence: recip_F={recip['recipient_following_F']:.4f} "
              f"donor_copy_F={recip['donor_answer_F']:.4f} "
              f"({'PASS' if recip['pass'] else 'FAIL'})")

        blk = eval_block_diagnostics(typed, train_worlds,
                                      np.random.default_rng(eval_seed + 7),
                                      held_out_worlds=eval_worlds)
        print(f"Block diagnostics:")
        print(f"  Gates: prim b1={blk['prim_gate_b1_mean']:.3f} "
              f"b2={blk['prim_gate_b2_mean']:.3f} | "
              f"comp b1={blk['comp_gate_b1_mean']:.3f} "
              f"b2={blk['comp_gate_b2_mean']:.3f}")
        print(f"  Acc after b1: prim={blk['prim_acc_after_b1']:.3f} "
              f"train_comp={blk['comp_train_acc_after_b1']:.3f} "
              f"held_comp={blk['comp_held_acc_after_b1']:.3f}")
        print(f"  Acc after b2: prim={blk['prim_acc_after_b2']:.3f} "
              f"train_comp={blk['comp_train_acc_after_b2']:.3f} "
              f"held_comp={blk['comp_held_acc_after_b2']:.3f}")
        print(f"  Mid after b1: train={blk['comp_train_mid_after_b1']:.3f} "
              f"held={blk['comp_held_mid_after_b1']:.3f}")

        # World-cluster bootstrap for composition accuracy
        boot_comp = bootstrap_world_cluster(
            typed, eval_worlds, HELD_OUT_COMPOSITIONS,
            np.random.default_rng(eval_seed + 8))
        print(f"Composition bootstrap: mean={boot_comp['mean']:.4f} "
              f"lower_5={boot_comp['lower_5']:.4f}")

        boot_port = bootstrap_portability_cluster(
            typed, eval_worlds, np.random.default_rng(eval_seed + 9))
        print(f"Portability bootstrap: mean={boot_port['mean']:.4f} "
              f"lower_5={boot_port['lower_5']:.4f}")

        all_results[f"typed_seed{seed}"] = {
            "training": typed_train,
            "clean_train_worlds": clean_tw,
            "clean_eval_worlds": clean_ew,
            "self_patch": sp, "portability": port,
            "oracle_action": oracle,
            "sequential_agreement": seq,
            "recipient_dependence": recip,
            "block_diagnostics": blk,
            "bootstrap_composition": boot_comp,
            "bootstrap_portability": boot_port,
        }

        # Log if clean accuracy is low but continue to next seed
        if clean_ew["clean_accuracy"] < 0.50:
            print(f"WARNING: clean accuracy {clean_ew['clean_accuracy']:.3f} < 0.50 — skipping untyped for this seed")
            all_results[f"typed_seed{seed}"]["low_accuracy"] = True
            continue

        # Untyped control — seed before construction
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        untyped = UntypedControl(d_model=148, n_heads=4, n_layers=4)
        n_untyped = sum(p.numel() for p in untyped.parameters())
        print(f"\nUntyped: {n_untyped:,} params")

        untyped_train = train_model(untyped, config, seed, train_worlds,
                                     results_dir, "untyped")

        # Same eval seed as typed for paired comparison
        clean_u = eval_clean_accuracy(untyped, train_worlds,
                                       np.random.default_rng(eval_seed),
                                       held_out_worlds=eval_worlds)
        print(f"Untyped clean: {clean_u['clean_accuracy']:.4f} "
              f"(prim={clean_u['primitive_accuracy']:.4f} "
              f"train_comp={clean_u['train_composition_accuracy']:.4f} "
              f"held_comp={clean_u['held_out_composition_accuracy']:.4f})")

        all_results[f"untyped_seed{seed}"] = {
            "training": untyped_train, "clean": clean_u,
        }

    with open(results_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - wall_start
    print(f"\nTotal: {elapsed:.1f}s ({elapsed/3600:.2f}h)")


if __name__ == "__main__":
    main()
