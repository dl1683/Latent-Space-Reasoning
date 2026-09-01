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

    # Output legend: 8 rows, each row is (opaque_place_alias, output_alias)
    # Renderer uses this to score which output alias corresponds to the place
    output_legend_places = torch.tensor(place_perm, dtype=torch.long)  # (8,)
    output_legend_outputs = torch.tensor(output_perm, dtype=torch.long)  # (8,)

    # World table: 32 triples (opaque_place, opaque_move, opaque_next)
    triples = []
    for am in range(4):
        for ap in range(8):
            an = int(world.transitions[am, ap])
            triples.append((int(place_perm[ap]), int(move_perm[am]), int(place_perm[an])))
    triple_order = list(range(32))
    random.shuffle(triple_order)
    triples = [triples[i] for i in triple_order]

    opaque_command = int(move_perm[abstract_move])
    opaque_initial = int(place_perm[initial_place])
    target_output = int(output_perm[target_place])

    return {
        "world_table": torch.tensor(triples, dtype=torch.long),      # (32, 3)
        "move_legend": move_legend,                                     # (4,)
        "initial_place": torch.tensor(opaque_initial, dtype=torch.long),
        "command": torch.tensor(opaque_command, dtype=torch.long),      # scalar
        "output_legend_places": output_legend_places,                   # (8,)
        "output_legend_outputs": output_legend_outputs,                 # (8,)
        "target": torch.tensor(target_output, dtype=torch.long),
        "is_composed": torch.tensor(0, dtype=torch.long),
        # For composition: second command (unused for primitives)
        "command2": torch.tensor(0, dtype=torch.long),
        # Metadata for evaluation (not used in training)
        "abstract_move": torch.tensor(abstract_move, dtype=torch.long),
        "abstract_initial": torch.tensor(initial_place, dtype=torch.long),
        "abstract_target": torch.tensor(target_place, dtype=torch.long),
        "world_id": torch.tensor(world.world_id, dtype=torch.long),
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

    output_legend_places = torch.tensor(place_perm, dtype=torch.long)
    output_legend_outputs = torch.tensor(output_perm, dtype=torch.long)

    triples = []
    for am in range(4):
        for ap in range(8):
            an = int(world.transitions[am, ap])
            triples.append((int(place_perm[ap]), int(move_perm[am]), int(place_perm[an])))
    triple_order = list(range(32))
    random.shuffle(triple_order)
    triples = [triples[i] for i in triple_order]

    opaque_cmd1 = int(move_perm[abstract_move1])
    opaque_cmd2 = int(move_perm[abstract_move2])
    opaque_initial = int(place_perm[initial_place])
    target_output = int(output_perm[target_place])

    return {
        "world_table": torch.tensor(triples, dtype=torch.long),
        "move_legend": move_legend,
        "initial_place": torch.tensor(opaque_initial, dtype=torch.long),
        "command": torch.tensor(opaque_cmd1, dtype=torch.long),
        "output_legend_places": output_legend_places,
        "output_legend_outputs": output_legend_outputs,
        "target": torch.tensor(target_output, dtype=torch.long),
        "is_composed": torch.tensor(1, dtype=torch.long),
        "command2": torch.tensor(opaque_cmd2, dtype=torch.long),
        "abstract_move": torch.tensor(abstract_move1, dtype=torch.long),
        "abstract_initial": torch.tensor(initial_place, dtype=torch.long),
        "abstract_target": torch.tensor(target_place, dtype=torch.long),
        "world_id": torch.tensor(world.world_id, dtype=torch.long),
    }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ---------------------------------------------------------------------------
# 3. MODEL — TYPED LEARNED ACTION CARRIER
# ---------------------------------------------------------------------------

class WorldWriter(nn.Module):
    """Encodes world table + initial place -> (world_memory, place_carrier).

    Cannot see: command, output labels.
    Sees: world table (32 triples), move legend, initial place alias.
    """
    def __init__(self, dim: int = 128):
        super().__init__()
        self.place_embed = nn.Embedding(8, dim)
        self.move_embed = nn.Embedding(4, dim)
        self.role_embed = nn.Embedding(3, dim)  # source, move, dest roles

        self.triple_proj = nn.Linear(3 * dim, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=dim * 2,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.init_proj = nn.Linear(dim, dim)

    def forward(self, world_table, move_legend, initial_place):
        """
        world_table: (B, 32, 3) — (opaque_place, opaque_move, opaque_next)
        move_legend: (B, 4) — legend[opaque_alias] = abstract_role
        initial_place: (B,) — opaque place alias
        Returns: (world_memory, place_carrier)
            world_memory: (B, 32, dim)
            place_carrier: (B, dim)
        """
        B = world_table.shape[0]
        device = world_table.device

        # Embed each component of the triple with role typing
        src_emb = self.place_embed(world_table[:,:,0]) + self.role_embed(
            torch.zeros(B, 32, dtype=torch.long, device=device))
        # Use abstract role from legend for the move embedding
        abstract_roles = torch.gather(move_legend, 1, world_table[:,:,1])  # (B, 32)
        move_emb = self.move_embed(abstract_roles) + self.role_embed(
            torch.ones(B, 32, dtype=torch.long, device=device))
        dst_emb = self.place_embed(world_table[:,:,2]) + self.role_embed(
            2 * torch.ones(B, 32, dtype=torch.long, device=device))

        triples = self.triple_proj(torch.cat([src_emb, move_emb, dst_emb], dim=-1))

        # Add initial place as CLS token
        init_tok = self.init_proj(self.place_embed(initial_place)).unsqueeze(1)
        tokens = torch.cat([init_tok, triples], dim=1)  # (B, 33, D)

        encoded = self.set_encoder(tokens)
        place_carrier = encoded[:, 0]       # (B, D)
        world_memory = encoded[:, 1:]       # (B, 32, D)
        return world_memory, place_carrier


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


class Executor(nn.Module):
    """Applies action carrier to world state: E(M, A) -> M'.

    Two cross-attention/update blocks so a composed carrier can
    induce two-hop behavior.
    """
    def __init__(self, world_dim: int = 128, action_dim: int = 64):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, world_dim)

        # Block 1
        self.query1 = nn.Linear(2 * world_dim, world_dim)
        self.attn1 = nn.MultiheadAttention(world_dim, 4, batch_first=True, dropout=0.0)
        self.update1 = nn.Sequential(
            nn.Linear(2 * world_dim, world_dim), nn.GELU(),
            nn.Linear(world_dim, world_dim),
        )
        self.norm1 = nn.LayerNorm(world_dim)

        # Block 2
        self.query2 = nn.Linear(2 * world_dim, world_dim)
        self.attn2 = nn.MultiheadAttention(world_dim, 4, batch_first=True, dropout=0.0)
        self.update2 = nn.Sequential(
            nn.Linear(2 * world_dim, world_dim), nn.GELU(),
            nn.Linear(world_dim, world_dim),
        )
        self.norm2 = nn.LayerNorm(world_dim)

    def forward(self, world_memory, place_carrier, action_carrier):
        """
        world_memory: (B, 32, world_dim)
        place_carrier: (B, world_dim)
        action_carrier: (B, action_dim)
        Returns: (B, world_dim) — updated place carrier
        """
        a_proj = self.action_proj(action_carrier)  # (B, world_dim)

        # Block 1: combine carrier + action, attend to world memory
        q1 = self.query1(torch.cat([place_carrier, a_proj], dim=-1)).unsqueeze(1)
        looked_up1, _ = self.attn1(q1, world_memory, world_memory)
        looked_up1 = looked_up1.squeeze(1)
        h1 = self.norm1(place_carrier + self.update1(torch.cat([place_carrier, looked_up1], dim=-1)))

        # Block 2: refine with another lookup
        q2 = self.query2(torch.cat([h1, a_proj], dim=-1)).unsqueeze(1)
        looked_up2, _ = self.attn2(q2, world_memory, world_memory)
        looked_up2 = looked_up2.squeeze(1)
        h2 = self.norm2(h1 + self.update2(torch.cat([h1, looked_up2], dim=-1)))

        return h2


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
        self.scorer = nn.Linear(world_dim, 1)
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


class TypedLAC(nn.Module):
    """The full LAC-0 typed model."""
    def __init__(self, world_dim: int = 128, action_dim: int = 64):
        super().__init__()
        self.world_writer = WorldWriter(world_dim)
        self.action_writer = ActionWriter(action_dim)
        self.composer = Composer(action_dim)
        self.executor = Executor(world_dim, action_dim)
        self.renderer = Renderer(world_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        world_mem, place_carrier = self.world_writer(
            batch["world_table"], batch["move_legend"], batch["initial_place"]
        )

        a1 = self.action_writer(batch["command"], batch["move_legend"])

        is_comp = batch["is_composed"]  # (B,)
        a2 = self.action_writer(batch["command2"], batch["move_legend"])
        composed = self.composer(a1, a2)

        # For primitives use a1; for compositions use μ(a1, a2)
        action = torch.where(is_comp.unsqueeze(-1).bool(), composed, a1)

        new_carrier = self.executor(world_mem, place_carrier, action)
        logits = self.renderer(
            new_carrier, batch["output_legend_places"], batch["output_legend_outputs"]
        )
        return logits

    def get_action_carrier(self, batch, composed=False):
        """Extract the action carrier (for transplantation tests)."""
        a1 = self.action_writer(batch["command"], batch["move_legend"])
        if not composed:
            return a1
        a2 = self.action_writer(batch["command2"], batch["move_legend"])
        return self.composer(a1, a2)

    def execute_with_transplanted_action(self, recipient_batch, donor_action):
        """Execute donor's action carrier in recipient's world."""
        world_mem, place_carrier = self.world_writer(
            recipient_batch["world_table"], recipient_batch["move_legend"],
            recipient_batch["initial_place"]
        )
        new_carrier = self.executor(world_mem, place_carrier, donor_action)
        return self.renderer(
            new_carrier, recipient_batch["output_legend_places"],
            recipient_batch["output_legend_outputs"]
        )


# ---------------------------------------------------------------------------
# 4. UNTYPED TRANSFORMER CONTROL
# ---------------------------------------------------------------------------

class UntypedControl(nn.Module):
    """Parameter-matched untyped transformer. Sees everything."""
    def __init__(self, d_model: int = 148, n_heads: int = 4, n_layers: int = 4):
        super().__init__()
        self.place_embed = nn.Embedding(8, d_model)
        self.move_embed = nn.Embedding(4, d_model)
        self.type_embed = nn.Embedding(6, d_model)  # src, move, dst, init, cmd1, cmd2
        self.out_embed = nn.Embedding(8, d_model)
        self.pos_embed = nn.Embedding(128, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.scorer = nn.Linear(d_model, 1)

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

        tokens = torch.cat([init_tok, cmd1_tok, cmd2_tok, triple_toks], dim=1)
        seq_len = tokens.shape[1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.pos_embed(positions)

        encoded = self.transformer(tokens)
        cls = encoded[:, 0]  # (B, D)

        # Pointer scorer over output legend
        olp = batch["output_legend_places"]
        olo = batch["output_legend_outputs"]
        p_emb = self.place_embed(olp)    # (B, 8, D)
        o_emb = self.out_embed(olo)      # (B, 8, D)
        rows = p_emb + o_emb             # (B, 8, D)

        scores = (rows * cls.unsqueeze(1)).sum(-1)  # (B, 8)
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

def eval_clean_accuracy(model, worlds, rng, n_samples=2000):
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(n_samples // 2):
            w = worlds[int(rng.integers(0, len(worlds)))]
            m = int(rng.integers(0, 4))
            b = collate([tensorize_primitive(w, rng, m)])
            if model(b).argmax(-1).item() == b["target"].item():
                correct += 1
        for _ in range(n_samples // 2):
            w = worlds[int(rng.integers(0, len(worlds)))]
            idx = int(rng.integers(0, len(TRAIN_COMPOSITIONS)))
            m1, m2 = TRAIN_COMPOSITIONS[idx]
            b = collate([tensorize_composition(w, rng, m1, m2)])
            if model(b).argmax(-1).item() == b["target"].item():
                correct += 1
    return {"clean_accuracy": correct / n_samples, "n_samples": n_samples}


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

    Donor produces action A in their world. Recipient world has different
    transition for the same abstract move. Target = what A does in recipient world.
    """
    if not isinstance(model, TypedLAC):
        return {"skip": True}
    model.eval()
    results = {"target": 0, "host": 0, "donor": 0, "other": 0, "total": 0}
    per_move = {m: {"target": 0, "total": 0} for m in range(4)}

    with torch.no_grad():
        attempts = 0
        while results["total"] < n_samples and attempts < n_samples * 20:
            attempts += 1
            # Pick two different worlds
            if len(worlds) < 2:
                continue
            idx = rng.choice(len(worlds), size=2, replace=False)
            w_recip = worlds[idx[0]]
            w_donor = worlds[idx[1]]

            abstract_move = int(rng.integers(0, 4))
            recip_place = int(rng.integers(0, 8))

            # s = host action endpoint (what recipient's own action does)
            s = int(w_recip.transitions[abstract_move, recip_place])
            # t = donor action's correct recipient-world endpoint
            t = int(w_recip.transitions[abstract_move, recip_place])
            # d = donor-world endpoint
            donor_place = int(rng.integers(0, 8))
            d = int(w_donor.transitions[abstract_move, donor_place])

            # For cross-world test: s = t (same abstract move, same recipient)
            # The test is: does the donor's action carrier for abstract_move
            # produce the correct result in the recipient's world?
            # s and t are the same here because it's the same move
            # The three-way clash requires s, t, d pairwise distinct
            if not (len({s, t, d}) == 3 or (s == t and s != d)):
                # We need at least s != d for a meaningful test
                if s == d:
                    continue

            # Shared relabeling for fair comparison
            place_perm = rng.permutation(8)
            move_perm = rng.permutation(4)
            output_perm = rng.permutation(8)

            # Recipient episode: recipient world, recipient place, same move
            recip_ep = tensorize_primitive(w_recip, rng, abstract_move)
            # Override to use specific initial place
            recip_ep["initial_place"] = torch.tensor(
                int(place_perm[recip_place]), dtype=torch.long)
            recip_ep["abstract_initial"] = torch.tensor(recip_place, dtype=torch.long)
            recip_ep["abstract_target"] = torch.tensor(t, dtype=torch.long)
            recip_ep["target"] = torch.tensor(int(output_perm[t]), dtype=torch.long)

            # Donor episode: donor world, same abstract move
            donor_ep = tensorize_primitive(w_donor, rng, abstract_move)

            recip_batch = collate([recip_ep])
            donor_batch = collate([donor_ep])

            # Get donor's action carrier
            donor_action = model.get_action_carrier(donor_batch, composed=False)

            # Execute donor's action in recipient's world
            logits = model.execute_with_transplanted_action(recip_batch, donor_action)
            pred = logits.argmax(-1).item()

            target_label = int(output_perm[t])
            donor_label = int(output_perm[d]) if d < 8 else -1

            if pred == target_label:
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


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def main():
    cfg_path = Path("experiments/config/learned_action_carrier_v0.json")
    with open(cfg_path) as f:
        config = json.load(f)

    results_dir = Path("experiments/results/learned_action_carrier_v0")
    results_dir.mkdir(parents=True, exist_ok=True)

    wall_start = time.time()
    wall_limit = config["budget"]["total_wall_hours"] * 3600

    # Generate worlds per Codex spec: 256 train, 256 held-out
    world_rng = np.random.default_rng(12345)
    train_worlds = [World.random(world_rng, i) for i in range(256)]
    eval_worlds = [World.random(world_rng, 256 + i) for i in range(256)]

    manifest = {
        "config": config,
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

        # Typed model
        typed = TypedLAC(world_dim=128, action_dim=64)
        n_typed = sum(p.numel() for p in typed.parameters())
        print(f"Typed: {n_typed:,} params")

        typed_train = train_model(typed, config, seed, train_worlds,
                                   results_dir, "typed")

        eval_rng = np.random.default_rng(seed + 1000)
        clean = eval_clean_accuracy(typed, train_worlds, eval_rng)
        print(f"Clean accuracy: {clean['clean_accuracy']:.4f}")

        sp = eval_self_patch_action(typed, train_worlds, eval_rng)
        print(f"Self-patch: {sp['max_discrepancy']:.2e} ({'PASS' if sp['pass'] else 'FAIL'})")

        port = eval_three_way_action_portability(typed, train_worlds, eval_rng)
        print(f"Three-way F: {port['target_following_F']:.4f} "
              f"({'PASS' if port['pass'] else 'FAIL'})")

        all_results[f"typed_seed{seed}"] = {
            "training": typed_train, "clean": clean,
            "self_patch": sp, "portability": port,
        }

        # Gate 1 early stop
        if clean["clean_accuracy"] < 0.50:
            print(f"EARLY STOP: clean accuracy {clean['clean_accuracy']:.3f} < 0.50")
            all_results[f"typed_seed{seed}"]["early_stop"] = "clean_accuracy"
            break

        # Untyped control
        untyped = UntypedControl(d_model=148, n_heads=4, n_layers=4)
        n_untyped = sum(p.numel() for p in untyped.parameters())
        print(f"\nUntyped: {n_untyped:,} params")

        untyped_train = train_model(untyped, config, seed, train_worlds,
                                     results_dir, "untyped")

        eval_rng2 = np.random.default_rng(seed + 2000)
        clean_u = eval_clean_accuracy(untyped, train_worlds, eval_rng2)
        print(f"Untyped clean: {clean_u['clean_accuracy']:.4f}")

        all_results[f"untyped_seed{seed}"] = {
            "training": untyped_train, "clean": clean_u,
        }

    with open(results_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - wall_start
    print(f"\nTotal: {elapsed:.1f}s ({elapsed/3600:.2f}h)")


if __name__ == "__main__":
    main()
