"""
Commit-Seal Carrier Experiment (CR-14 revised, Rung 1)

Tests whether a learned update U, constrained by a hard causal cut
(zeroing the history-bypass path at COMMIT), learns to carry the live
proposal through the carrier while ignoring the obsolete first proposal
at the registered immediate reader.

Rung 1: 1 address, 2 values, 2 commits.
Architecture per Codex design gate (2026-09-04):
  - Transient encoder: GRU, reset per statement
  - Proposal: MLP (SiLU + tanh), with supervised visible head
  - Carrier update U: GRUCell (carrier as hidden state)
  - History bypass: GRUCell, zeroed at COMMIT in sealed arm
  - Reader: candidate scorer with per-episode output legend

The seal is a causal cut (not an information bottleneck: continuous
m can still encode both values). The learned update U is the
falsifiable object. Five seeds; RUNG1_MECHANISM_PASS requires 4/5.
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


N_VALUES = 2
SEEDS = [11, 23, 37, 53, 71]


# -------------------------------------------------------------------
# Data
# -------------------------------------------------------------------

def make_all_episodes():
    """All 8 factorial cells: (u, v, legend_order)."""
    episodes = []
    for u in range(N_VALUES):
        for v in range(N_VALUES):
            for lo in range(2):
                legend = [0, 1] if lo == 0 else [1, 0]
                episodes.append({
                    "u": u, "v": v,
                    "legend": legend,
                    "target_u": legend.index(u),
                    "target_v": legend.index(v),
                })
    return episodes


def episodes_to_tensors(episodes, device="cpu"):
    """Convert episode dicts to batched tensors."""
    n = len(episodes)
    write_vals = torch.tensor([[e["u"], e["v"]] for e in episodes],
                              dtype=torch.long, device=device)
    legends = torch.tensor([e["legend"] for e in episodes],
                           dtype=torch.long, device=device)
    target_u = torch.tensor([e["target_u"] for e in episodes],
                            dtype=torch.long, device=device)
    target_v = torch.tensor([e["target_v"] for e in episodes],
                            dtype=torch.long, device=device)
    return write_vals, legends, target_u, target_v


def make_training_data(n_train, seed, device="cpu"):
    """Sample uniformly from the 8 cells."""
    rng = random.Random(seed)
    all_eps = make_all_episodes()
    eps = [rng.choice(all_eps) for _ in range(n_train)]
    return episodes_to_tensors(eps, device)


# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------

class CommitSealRung1(nn.Module):
    """Commit-seal carrier model for rung 1."""

    def __init__(self, d_embed=16, d_transient=32, d_proposal=16,
                 d_carrier=16, d_addr=8, d_bypass=32, seal=True):
        super().__init__()
        self.seal = seal
        self.d_transient = d_transient
        self.d_bypass = d_bypass

        self.value_embed = nn.Embedding(N_VALUES, d_embed)
        self.addr_embed = nn.Embedding(1, d_addr)

        self.transient_gru = nn.GRUCell(d_embed, d_transient)

        self.proposal_mlp = nn.Sequential(
            nn.Linear(d_transient, d_transient), nn.SiLU(),
            nn.Linear(d_transient, d_proposal), nn.Tanh(),
        )
        self.visible_head = nn.Linear(d_proposal, N_VALUES)

        self.update_gru = nn.GRUCell(d_addr + d_proposal, d_carrier)
        self.m0 = nn.Parameter(torch.zeros(d_carrier))

        self.bypass_gru = nn.GRUCell(d_transient, d_bypass)

        reader_in = d_carrier + d_bypass + d_addr + d_embed
        self.reader = nn.Sequential(
            nn.Linear(reader_in, d_transient), nn.SiLU(),
            nn.Linear(d_transient, 1),
        )

    def _encode(self, val_idx):
        emb = self.value_embed(val_idx)
        h0 = torch.zeros(val_idx.shape[0], self.d_transient,
                         device=val_idx.device)
        return self.transient_gru(emb, h0)

    def _propose(self, s):
        return self.proposal_mlp(s)

    def _update(self, m, proposal, addr):
        addr_emb = self.addr_embed(addr)
        return self.update_gru(torch.cat([addr_emb, proposal], 1), m)

    def _score(self, m, b, addr, legend):
        """Score each legend slot. Returns (batch, 2) logits."""
        addr_emb = self.addr_embed(addr)
        scores = []
        for j in range(N_VALUES):
            cand = self.value_embed(legend[:, j])
            inp = torch.cat([m, b, addr_emb, cand], 1)
            scores.append(self.reader(inp))
        return torch.cat(scores, 1)

    def forward_full(self, write_vals, legends):
        """Full forward returning all intermediates for loss + measurement.

        Args:
            write_vals: (batch, 2) — values u, v
            legends:    (batch, 2) — legend mapping

        Returns dict with keys:
            final_logits, inter_logits, prop_logits (list of 2),
            m1, m2, b1, b2, p1, p2
        """
        batch = write_vals.shape[0]
        dev = write_vals.device
        addr = torch.zeros(batch, dtype=torch.long, device=dev)

        m = self.m0.unsqueeze(0).expand(batch, -1).contiguous()
        b = torch.zeros(batch, self.d_bypass, device=dev)

        intermediates = {}
        for i in range(2):
            s = self._encode(write_vals[:, i])
            p = self._propose(s)
            intermediates[f"p{i+1}"] = p
            intermediates[f"prop_logits_{i+1}"] = self.visible_head(p)

            m = self._update(m, p, addr)
            b_new = self.bypass_gru(s, b)
            b = torch.zeros_like(b) if self.seal else b_new

            intermediates[f"m{i+1}"] = m
            intermediates[f"b{i+1}"] = b

            if i == 0:
                intermediates["inter_logits"] = self._score(m, b, addr, legends)

        intermediates["final_logits"] = self._score(m, b, addr, legends)
        return intermediates

    def forward(self, write_vals, legends):
        return self.forward_full(write_vals, legends)["final_logits"]

    def read_from(self, m, b, legends, device="cpu"):
        """Score candidates from given carrier + bypass states."""
        batch = m.shape[0]
        addr = torch.zeros(batch, dtype=torch.long, device=device)
        return self._score(m, b, addr, legends)


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------

def train_model(model, write_vals, legends, target_u, target_v,
                n_epochs, lr, batch_size, weight_decay=1e-4,
                grad_clip=1.0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    n = write_vals.shape[0]
    dev = write_vals.device
    history = {"loss": [], "acc": []}

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        eloss, ecorrect = 0.0, 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            out = model.forward_full(write_vals[idx], legends[idx])

            l_prop = 0.5 * (
                F.cross_entropy(out["prop_logits_1"], write_vals[idx, 0])
                + F.cross_entropy(out["prop_logits_2"], write_vals[idx, 1])
            )
            l_inter = F.cross_entropy(out["inter_logits"], target_u[idx])
            l_final = F.cross_entropy(out["final_logits"], target_v[idx])

            loss = 0.25 * l_prop + 0.50 * l_inter + 1.00 * l_final

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            bs = len(idx)
            eloss += loss.item() * bs
            ecorrect += (out["final_logits"].argmax(1) == target_v[idx]).sum().item()

        history["loss"].append(eloss / n)
        history["acc"].append(ecorrect / n)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}: loss={eloss/n:.6f} "
                  f"acc={ecorrect/n:.3f}", flush=True)

    return history


# -------------------------------------------------------------------
# Measurements
# -------------------------------------------------------------------

def eval_accuracy(model, write_vals, legends, target_u, target_v):
    """Compute visible-proposal, single-write, and two-write accuracy."""
    model.eval()
    with torch.no_grad():
        out = model.forward_full(write_vals, legends)
    prop1_acc = (out["prop_logits_1"].argmax(1) == write_vals[:, 0]).float().mean().item()
    prop2_acc = (out["prop_logits_2"].argmax(1) == write_vals[:, 1]).float().mean().item()
    inter_acc = (out["inter_logits"].argmax(1) == target_u).float().mean().item()
    final_acc = (out["final_logits"].argmax(1) == target_v).float().mean().item()
    prop1_prob = F.softmax(out["prop_logits_1"], 1).gather(1, write_vals[:, 0:1]).mean().item()
    prop2_prob = F.softmax(out["prop_logits_2"], 1).gather(1, write_vals[:, 1:2]).mean().item()
    inter_prob = F.softmax(out["inter_logits"], 1).gather(1, target_u.unsqueeze(1)).mean().item()
    final_prob = F.softmax(out["final_logits"], 1).gather(1, target_v.unsqueeze(1)).mean().item()
    return {
        "proposal_acc": (prop1_acc + prop2_acc) / 2,
        "proposal_prob": (prop1_prob + prop2_prob) / 2,
        "single_write_acc": inter_acc,
        "single_write_prob": inter_prob,
        "two_write_acc": final_acc,
        "two_write_prob": final_prob,
    }


def measure_numerical_floor(model, write_vals, legends, target_v, device):
    """Replay under different batch orderings to establish noise floor."""
    model.eval()
    probs_list = []
    for bs in [1, 4, 8]:
        all_p = []
        with torch.no_grad():
            for i in range(0, write_vals.shape[0], bs):
                sl = slice(i, i + bs)
                logits = model(write_vals[sl], legends[sl])
                all_p.append(F.softmax(logits, 1))
        probs_list.append(torch.cat(all_p, 0))

    max_tv = 0.0
    for i in range(len(probs_list)):
        for j in range(i + 1, len(probs_list)):
            tv = 0.5 * (probs_list[i] - probs_list[j]).abs().sum(1).max().item()
            max_tv = max(max_tv, tv)
    return max_tv


def extract_carriers(model, write_vals, legends):
    """Extract carriers and bypasses after each commit on the 8 cells."""
    model.eval()
    with torch.no_grad():
        out = model.forward_full(write_vals, legends)
    return {
        "m1": out["m1"], "m2": out["m2"],
        "b1": out["b1"], "b2": out["b2"],
    }


def measure_paired_erasure(model, device):
    """Max TV between counterfactual pairs sharing the same v."""
    model.eval()
    eps = make_all_episodes()
    wv, lg, _, tv = episodes_to_tensors(eps, device)

    with torch.no_grad():
        logits = model(wv, lg)
        probs = F.softmax(logits, 1)

    # Group by (v, legend_order) — same v & legend → same target
    max_tv = 0.0
    for v in range(N_VALUES):
        for lo in range(2):
            idxs = [i for i, e in enumerate(eps)
                    if e["v"] == v and e["legend"] == ([0, 1] if lo == 0 else [1, 0])]
            if len(idxs) == 2:
                tv_val = 0.5 * (probs[idxs[0]] - probs[idxs[1]]).abs().sum().item()
                max_tv = max(max_tv, tv_val)

    return max_tv


def measure_transplant_matrix(model, device):
    """Full 4×4 transplant matrix (donor × recipient) for the carrier."""
    model.eval()
    eps = make_all_episodes()
    wv, lg, _, tv = episodes_to_tensors(eps, device)

    with torch.no_grad():
        out = model.forward_full(wv, lg)
        m2_all = out["m2"]
        b2_all = out["b2"]
        baseline_probs = F.softmax(out["final_logits"], 1)

    results = {"baseline": baseline_probs.cpu().tolist(), "transplants": []}

    for donor_i in range(len(eps)):
        for recip_i in range(len(eps)):
            m_donor = m2_all[donor_i:donor_i + 1]
            b_recip = b2_all[recip_i:recip_i + 1]
            lg_recip = lg[recip_i:recip_i + 1]

            with torch.no_grad():
                logits = model.read_from(m_donor, b_recip, lg_recip, device)
                probs = F.softmax(logits, 1)

            results["transplants"].append({
                "donor": (eps[donor_i]["u"], eps[donor_i]["v"]),
                "recipient": (eps[recip_i]["u"], eps[recip_i]["v"]),
                "donor_legend": eps[donor_i]["legend"],
                "recip_legend": eps[recip_i]["legend"],
                "probs": probs[0].cpu().tolist(),
                "pred": probs[0].argmax().item(),
            })

    return results


def measure_proposal_shuffle(model, device):
    """Shuffle first or second proposal and measure effect."""
    model.eval()
    eps = make_all_episodes()
    wv, lg, tu, tv = episodes_to_tensors(eps, device)

    with torch.no_grad():
        baseline_out = model.forward_full(wv, lg)
        base_probs = F.softmax(baseline_out["final_logits"], 1)

    # First-proposal shuffle: swap u → 1-u
    wv_shuf1 = wv.clone()
    wv_shuf1[:, 0] = 1 - wv_shuf1[:, 0]
    with torch.no_grad():
        shuf1_out = model.forward_full(wv_shuf1, lg)
        shuf1_probs = F.softmax(shuf1_out["final_logits"], 1)
    first_max_tv = 0.5 * (base_probs - shuf1_probs).abs().sum(1).max().item()

    # Second-proposal shuffle: swap v → 1-v
    wv_shuf2 = wv.clone()
    wv_shuf2[:, 1] = 1 - wv_shuf2[:, 1]
    with torch.no_grad():
        shuf2_out = model.forward_full(wv_shuf2, lg)
        shuf2_probs = F.softmax(shuf2_out["final_logits"], 1)
    second_max_tv = 0.5 * (base_probs - shuf2_probs).abs().sum(1).max().item()

    # Second shuffle should flip the answer: check if new answer is correct
    # for the flipped v
    new_targets = []
    for i, e in enumerate(eps):
        new_v = 1 - e["v"]
        new_targets.append(e["legend"].index(new_v))
    new_targets_t = torch.tensor(new_targets, device=device)
    shuf2_acc = (shuf2_out["final_logits"].argmax(1) == new_targets_t).float().mean().item()
    shuf2_prob = F.softmax(shuf2_out["final_logits"], 1).gather(
        1, new_targets_t.unsqueeze(1)).mean().item()

    return {
        "first_shuffle_max_tv": first_max_tv,
        "second_shuffle_max_tv": second_max_tv,
        "second_shuffle_acc": shuf2_acc,
        "second_shuffle_prob": shuf2_prob,
    }


def measure_quotient(model, device, tau):
    """Transitivity-checked quotient over response profiles.

    Per Codex spec: each history (u,v) has a response profile that
    concatenates probability vectors from both legend orders. This gives
    4 profiles (one per history), and the quotient is computed on those.
    """
    model.eval()
    eps = make_all_episodes()
    wv, lg, _, _ = episodes_to_tensors(eps, device)

    with torch.no_grad():
        probs = F.softmax(model(wv, lg), 1).cpu().numpy()

    # Group cells by (u,v), keyed by legend order
    history_by_legend = {}
    for i, e in enumerate(eps):
        key = (e["u"], e["v"])
        lg_key = tuple(e["legend"])
        history_by_legend.setdefault(key, {})[lg_key] = probs[i]

    histories = sorted(history_by_legend.keys())
    legend_orders = sorted(set(tuple(e["legend"]) for e in eps))

    # Distance = max TV over legend orders (Codex spec)
    n = len(histories)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            max_tv = 0.0
            for lg in legend_orders:
                pi = history_by_legend[histories[i]].get(lg)
                pj = history_by_legend[histories[j]].get(lg)
                if pi is not None and pj is not None:
                    tv = 0.5 * np.abs(pi - pj).sum()
                    max_tv = max(max_tv, tv)
            dist[i, j] = max_tv

    equiv = dist <= tau
    transitive = True
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if equiv[i, j] and equiv[j, k] and not equiv[i, k]:
                    transitive = False

    if not transitive:
        return {"transitive": False, "n_classes": None, "clusters": None,
                "histories": [list(h) for h in histories]}

    visited = [False] * n
    clusters = [-1] * n
    c = 0
    for i in range(n):
        if visited[i]:
            continue
        for j in range(i, n):
            if equiv[i, j]:
                clusters[j] = c
                visited[j] = True
        c += 1

    # Within-live max TV and between-live min TV (on histories)
    live_groups = {}
    for i, h in enumerate(histories):
        live_groups.setdefault(h[1], []).append(i)

    within_max = 0.0
    for idxs in live_groups.values():
        for a in idxs:
            for b in idxs:
                within_max = max(within_max, dist[a, b])

    between_min = float("inf")
    groups = list(live_groups.values())
    if len(groups) == 2:
        for a in groups[0]:
            for b in groups[1]:
                between_min = min(between_min, dist[a, b])
    else:
        between_min = 0.0

    return {
        "transitive": True,
        "n_classes": len(set(clusters)),
        "clusters": clusters,
        "histories": [list(h) for h in histories],
        "within_live_max_tv": within_max,
        "between_live_min_tv": between_min,
        "dist_matrix": dist.tolist(),
    }


# -------------------------------------------------------------------
# Gate logic
# -------------------------------------------------------------------

def analyze_transplant(transplant):
    """Extract gate-relevant metrics from the full transplant matrix."""
    eps = make_all_episodes()
    baseline = transplant["baseline"]
    entries = transplant["transplants"]

    self_max_tv = 0.0
    same_live_max_tv = 0.0
    wrong_live_probs = []
    donor_obsolete_max_tv = 0.0

    for t in entries:
        d_u, d_v = t["donor"]
        r_u, r_v = t["recipient"]
        d_lg = tuple(t["donor_legend"])
        r_lg = tuple(t["recip_legend"])
        t_probs = t["probs"]

        # Find the baseline for this recipient
        r_idx = next(i for i, e in enumerate(eps)
                     if e["u"] == r_u and e["v"] == r_v
                     and tuple(e["legend"]) == r_lg)
        b_probs = baseline[r_idx]

        tv = 0.5 * sum(abs(a - b) for a, b in zip(t_probs, b_probs))

        if d_u == r_u and d_v == r_v and d_lg == r_lg:
            self_max_tv = max(self_max_tv, tv)
        elif d_v == r_v and d_u != r_u and d_lg == r_lg:
            same_live_max_tv = max(same_live_max_tv, tv)
        elif d_v != r_v and d_lg == r_lg:
            target_slot = list(r_lg).index(d_v)
            wrong_live_probs.append(t_probs[target_slot])

    # Donor-obsolete effect: among wrong-live donors sharing the same
    # donor v but different donor u, max TV between their transplant outputs
    wrong_live_by_key = {}
    for t in entries:
        d_u, d_v = t["donor"]
        r_u, r_v = t["recipient"]
        d_lg = tuple(t["donor_legend"])
        r_lg = tuple(t["recip_legend"])
        if d_v != r_v and d_lg == r_lg:
            key = (d_v, r_u, r_v, r_lg)
            wrong_live_by_key.setdefault(key, []).append(t["probs"])
    for probs_list in wrong_live_by_key.values():
        for i in range(len(probs_list)):
            for j in range(i + 1, len(probs_list)):
                tv = 0.5 * sum(abs(a - b) for a, b in
                               zip(probs_list[i], probs_list[j]))
                donor_obsolete_max_tv = max(donor_obsolete_max_tv, tv)

    mean_wrong_live = (sum(wrong_live_probs) / len(wrong_live_probs)
                       if wrong_live_probs else 0.0)

    return {
        "self_max_tv": self_max_tv,
        "same_live_max_tv": same_live_max_tv,
        "wrong_live_mean_prob": mean_wrong_live,
        "donor_obsolete_max_tv": donor_obsolete_max_tv,
    }


def check_mechanism_gate(accs, erasure_tv, transplant, prop_shuf,
                         quotient, num_floor, tau, tau_self):
    """Check all rung-1 mechanism gate conditions. Returns (pass, reasons)."""
    reasons = []

    if num_floor > 1e-3:
        reasons.append(f"INVALID: numerical floor {num_floor:.6f} > 1e-3")

    if accs["proposal_acc"] < 0.95 or accs["proposal_prob"] < 0.95:
        reasons.append(f"proposal visibility: acc={accs['proposal_acc']:.3f} "
                       f"prob={accs['proposal_prob']:.3f}")
    if accs["single_write_acc"] < 0.95 or accs["single_write_prob"] < 0.95:
        reasons.append(f"single-write: acc={accs['single_write_acc']:.3f} "
                       f"prob={accs['single_write_prob']:.3f}")
    if accs["two_write_acc"] < 0.95 or accs["two_write_prob"] < 0.95:
        reasons.append(f"two-write: acc={accs['two_write_acc']:.3f} "
                       f"prob={accs['two_write_prob']:.3f}")
    if erasure_tv > tau:
        reasons.append(f"paired erasure: TV={erasure_tv:.6f} > tau={tau:.6f}")

    tx = analyze_transplant(transplant)
    if tx["self_max_tv"] > tau_self:
        reasons.append(f"self-transplant: TV={tx['self_max_tv']:.6f} > tau_self={tau_self:.6f}")
    if tx["same_live_max_tv"] > tau:
        reasons.append(f"same-live transplant: TV={tx['same_live_max_tv']:.6f}")
    if tx["wrong_live_mean_prob"] < 0.95:
        reasons.append(f"wrong-live donor prob={tx['wrong_live_mean_prob']:.3f}")
    if tx["donor_obsolete_max_tv"] > tau:
        reasons.append(f"donor-obsolete effect: TV={tx['donor_obsolete_max_tv']:.6f}")

    if prop_shuf["first_shuffle_max_tv"] > tau:
        reasons.append(f"first-proposal shuffle: TV={prop_shuf['first_shuffle_max_tv']:.6f}")
    if prop_shuf["second_shuffle_prob"] < 0.95:
        reasons.append(f"second-proposal shuffle prob={prop_shuf['second_shuffle_prob']:.3f}")

    if quotient["transitive"] and quotient["n_classes"] != 2:
        reasons.append(f"quotient: {quotient['n_classes']} classes, expected 2")
    if not quotient["transitive"]:
        reasons.append("quotient: transitivity failed")
    if quotient.get("within_live_max_tv", 0) > tau:
        reasons.append(f"quotient within-live TV={quotient['within_live_max_tv']:.6f}")
    if quotient.get("between_live_min_tv", 1) < 0.50:
        reasons.append(f"quotient between-live TV={quotient['between_live_min_tv']:.6f} < 0.50")

    return len(reasons) == 0, reasons


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def run_one_seed(seed, config, device="cpu"):
    """Run sealed + no-seal for one seed. Returns results dict."""
    print(f"\n====== Seed {seed} ======")
    d_embed = config["d_embed"]
    d_transient = config["d_transient"]
    d_proposal = config["d_proposal"]
    d_carrier = config["d_carrier"]
    d_addr = config["d_addr"]
    d_bypass = config["d_bypass"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    wv_train, lg_train, tu_train, tv_train = make_training_data(
        config["n_train"], seed, device)

    eps = make_all_episodes()
    wv_eval, lg_eval, tu_eval, tv_eval = episodes_to_tensors(eps, device)

    # Create sealed model, clone for no-seal
    torch.manual_seed(seed)
    sealed = CommitSealRung1(d_embed, d_transient, d_proposal,
                             d_carrier, d_addr, d_bypass, seal=True)
    torch.manual_seed(seed)
    noseal = CommitSealRung1(d_embed, d_transient, d_proposal,
                             d_carrier, d_addr, d_bypass, seal=False)
    # Ensure identical initial weights
    noseal.load_state_dict(sealed.state_dict())

    results = {}

    for tag, model in [("sealed", sealed), ("noseal", noseal)]:
        print(f"\n--- {tag.upper()} ---")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params}")

        # Train with same minibatch order
        torch.manual_seed(seed + 1000)
        history = train_model(
            model, wv_train, lg_train, tu_train, tv_train,
            n_epochs=config["n_epochs"], lr=config["lr"],
            batch_size=config["batch_size"],
            weight_decay=config.get("weight_decay", 1e-4),
            grad_clip=config.get("grad_clip", 1.0),
        )

        accs = eval_accuracy(model, wv_eval, lg_eval, tu_eval, tv_eval)
        num_floor = measure_numerical_floor(model, wv_eval, lg_eval, tv_eval, device)
        tau = max(0.02, 20 * num_floor)
        tau_self = max(1e-6, 10 * num_floor)

        if tau > 0.02:
            print(f"  WARNING: numerical floor {num_floor:.6f} yields tau={tau:.4f} > 0.02")

        erasure_tv = measure_paired_erasure(model, device)
        transplant = measure_transplant_matrix(model, device)
        prop_shuf = measure_proposal_shuffle(model, device)
        quotient = measure_quotient(model, device, tau)

        gate_pass, gate_reasons = check_mechanism_gate(
            accs, erasure_tv, transplant, prop_shuf, quotient,
            num_floor, tau, tau_self)
        tx_analysis = analyze_transplant(transplant)

        print(f"  Proposal acc: {accs['proposal_acc']:.3f} (prob {accs['proposal_prob']:.3f})")
        print(f"  Single-write acc: {accs['single_write_acc']:.3f}")
        print(f"  Two-write acc: {accs['two_write_acc']:.3f}")
        print(f"  Numerical floor: {num_floor:.8f}, tau={tau:.6f}")
        print(f"  Paired erasure max TV: {erasure_tv:.6f}")
        print(f"  Quotient: {quotient['n_classes']} classes "
              f"(transitive={quotient['transitive']})")
        print(f"  First-prop shuffle TV: {prop_shuf['first_shuffle_max_tv']:.6f}")
        print(f"  Second-prop shuffle acc: {prop_shuf['second_shuffle_acc']:.3f}")
        print(f"  Transplant: self={tx_analysis['self_max_tv']:.6f}, "
              f"same-live={tx_analysis['same_live_max_tv']:.6f}, "
              f"wrong-live={tx_analysis['wrong_live_mean_prob']:.3f}, "
              f"obsolete={tx_analysis['donor_obsolete_max_tv']:.6f}")
        if quotient.get("within_live_max_tv") is not None:
            print(f"  Quotient within-live={quotient['within_live_max_tv']:.6f}, "
                  f"between-live={quotient['between_live_min_tv']:.6f}")
        print(f"  Gate: {'PASS' if gate_pass else 'FAIL'}")
        if gate_reasons:
            for r in gate_reasons:
                print(f"    - {r}")

        results[tag] = {
            "n_params": n_params,
            "final_loss": history["loss"][-1],
            "final_acc": history["acc"][-1],
            "accuracy": accs,
            "numerical_floor": num_floor,
            "tau": tau,
            "tau_self": tau_self,
            "paired_erasure_max_tv": erasure_tv,
            "quotient": quotient,
            "proposal_shuffle": prop_shuf,
            "transplant_analysis": tx_analysis,
            "transplant_matrix": transplant,
            "gate_pass": gate_pass,
            "gate_reasons": gate_reasons,
        }

    return results


def run_experiment(config, out_dir=None):
    device = "cpu"
    seeds = config.get("seeds", SEEDS)

    if out_dir is None:
        out_dir = f"experiments/results/{config.get('experiment_id', 'commit_seal_rung1')}"
    os.makedirs(out_dir, exist_ok=True)
    partial_path = os.path.join(out_dir, "partial.json")

    # Resume from partial results if they exist
    all_results = {}
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            all_results = json.load(f)
        done = [s for s in seeds if f"seed_{s}" in all_results]
        print(f"Resuming: seeds {done} already complete, skipping them.")

    print("=== Commit-Seal Carrier (CR-14 revised, Rung 1) ===")
    print(f"Values: {N_VALUES}, Commits: 2, Address: 1")
    print(f"Seeds: {seeds}")
    print(f"d_carrier={config['d_carrier']}, d_embed={config['d_embed']}")

    sealed_passes = 0
    noseal_competent = 0

    for seed in seeds:
        key = f"seed_{seed}"
        if key in all_results:
            res = all_results[key]
        else:
            res = run_one_seed(seed, config, device)
            all_results[key] = res
            with open(partial_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            sys.stdout.flush()

        if res["sealed"]["gate_pass"]:
            sealed_passes += 1
        if res["noseal"]["accuracy"]["two_write_acc"] >= 0.95:
            noseal_competent += 1

    # Overall verdict
    print(f"\n=== OVERALL ({sealed_passes}/{len(seeds)} sealed seeds pass) ===")

    if sealed_passes >= 4:
        verdict = "RUNG1_MECHANISM_PASS"
    elif sealed_passes <= len(seeds) - 2:
        verdict = "RUNG1_FAIL"
    else:
        verdict = "INCONCLUSIVE"

    # Seal advantage (separate from mechanism pass)
    seal_advantage_count = 0
    for seed in seeds:
        sr = all_results[f"seed_{seed}"]
        sq = sr["sealed"].get("quotient", {}).get("n_classes")
        nq = sr["noseal"].get("quotient", {}).get("n_classes")
        if sq == 2 and nq is not None and nq > 2:
            seal_advantage_count += 1

    seal_verdict = ("SEAL_ADVANTAGE_SUPPORTED" if seal_advantage_count >= 4
                    else "SEAL_ADVANTAGE_NOT_ESTABLISHED")

    print(f"  Mechanism: {verdict}")
    print(f"  Seal advantage: {seal_verdict} ({seal_advantage_count}/{len(seeds)})")
    print(f"  No-seal competent: {noseal_competent}/{len(seeds)}")

    all_results["verdict"] = verdict
    all_results["seal_verdict"] = seal_verdict
    all_results["sealed_passes"] = sealed_passes
    all_results["noseal_competent"] = noseal_competent

    # Clean up partial file now that we have the full result
    if os.path.exists(partial_path):
        os.remove(partial_path)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Commit-Seal Carrier (Rung 1)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    out_dir = args.output_dir or (
        f"experiments/results/{config.get('experiment_id', 'commit_seal_rung1')}")
    results = run_experiment(config, out_dir)

    output = {"config": config, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    output.update(results)

    path = os.path.join(out_dir, "result.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
