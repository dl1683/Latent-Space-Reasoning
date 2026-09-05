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


# -------------------------------------------------------------------
# Single-cell interval certificate (CR-15.3 on one trained sealed cell)
#
# Design gate: internal/codex_transcripts/analytic_r4_design.md (round 4).
# Sound outward interval arithmetic (mpmath.iv, 128-bit). Every reported
# bound is an upper bound; every inequality is checked on 128-bit
# endpoints. The certificate concerns the real-arithmetic network defined
# by the saved binary weights, not floating-point inference.
# -------------------------------------------------------------------

def _cert_backend():
    from mpmath import iv, mp
    iv.prec = 128
    mp.prec = 128
    return iv


def _I(iv, x):
    return iv.mpf(float(x))


def _ivec(iv, arr):
    return [iv.mpf(float(a)) for a in np.asarray(arr, dtype=np.float64).ravel()]


def _matvec(iv, W, x):
    """W: numpy (n,m) float; x: list of n-intervals length m."""
    W = np.asarray(W, dtype=np.float64)
    out = []
    for i in range(W.shape[0]):
        acc = iv.mpf(0)
        row = W[i]
        for j in range(W.shape[1]):
            if row[j] != 0.0:
                acc += iv.mpf(float(row[j])) * x[j]
        out.append(acc)
    return out


def _isig(iv, t):
    return 1 / (1 + iv.exp(-t))


def _itanh(iv, t):
    return 1 - 2 / (iv.exp(2 * t) + 1)


def _isilu(iv, t):
    return t * _isig(iv, t)


def _isilu_d(iv, t):
    s = _isig(iv, t)
    return s + t * s * (1 - s)


def _unimodal_range(iv, t, f, fmax):
    """Range of a function f that is increasing on (-inf,0], decreasing on
    [0,inf), with maximum fmax at 0, over the interval t. Sound because the
    monotone pieces are evaluated at endpoints."""
    a, b = t.a, t.b
    if a <= 0 <= b:
        hi = iv.mpf(fmax)
        lo_candidates = [f(iv.mpf(a)), f(iv.mpf(b))]
        lo = min(c.a for c in lo_candidates)
        return iv.mpf([lo, hi.b])
    fa, fb = f(iv.mpf(a)), f(iv.mpf(b))
    return iv.mpf([min(fa.a, fb.a), max(fa.b, fb.b)])


def _sigd_range(iv, t):
    return _unimodal_range(iv, t, lambda u: _isig(iv, u) * (1 - _isig(iv, u)), 0.25)


def _tanhd_range(iv, t):
    return _unimodal_range(iv, t, lambda u: 1 - _itanh(iv, u) ** 2, 1.0)


def _sup_abs(v):
    return max(abs(v.a), abs(v.b))


def _inf_norm_sup(vec):
    return max(_sup_abs(x) for x in vec)


def _pt_hex(t):
    a = t.detach().cpu().contiguous().numpy()
    return {"dtype": str(a.dtype), "shape": list(a.shape), "hex": a.tobytes().hex()}


class _CellIV:
    """Interval evaluator for the update GRUCell of a CommitSealRung1 model."""

    def __init__(self, iv, model):
        self.iv = iv
        g = model.update_gru
        W_ih = g.weight_ih.detach().double().numpy()
        W_hh = g.weight_hh.detach().double().numpy()
        b_ih = g.bias_ih.detach().double().numpy()
        b_hh = g.bias_hh.detach().double().numpy()
        H = g.hidden_size
        self.H = H
        self.W_ir, self.W_iz, self.W_in = W_ih[:H], W_ih[H:2 * H], W_ih[2 * H:]
        self.H_r, self.H_z, self.H_n = W_hh[:H], W_hh[H:2 * H], W_hh[2 * H:]
        self.b_ir, self.b_iz, self.b_in = b_ih[:H], b_ih[H:2 * H], b_ih[2 * H:]
        self.b_hr, self.b_hz, self.b_hn = b_hh[:H], b_hh[H:2 * H], b_hh[2 * H:]
        self.model = model

    def input_intervals(self, v):
        """x_v = [addr_emb(0), p_v] enclosed from exact weights (real arithmetic)."""
        iv = self.iv
        m = self.model
        e = _ivec(iv, m.value_embed.weight[v].detach().double().numpy())
        tg = m.transient_gru
        Wih = tg.weight_ih.detach().double().numpy()
        Whh_b = tg.bias_hh.detach().double().numpy()
        bih = tg.bias_ih.detach().double().numpy()
        Ht = tg.hidden_size
        pre_r = [a + _I(iv, bih[i]) + _I(iv, Whh_b[i]) for i, a in enumerate(_matvec(iv, Wih[:Ht], e))]
        pre_z = [a + _I(iv, bih[Ht + i]) + _I(iv, Whh_b[Ht + i]) for i, a in enumerate(_matvec(iv, Wih[Ht:2 * Ht], e))]
        r = [_isig(iv, t) for t in pre_r]
        z = [_isig(iv, t) for t in pre_z]
        pre_n = [a + _I(iv, bih[2 * Ht + i]) + r[i] * _I(iv, Whh_b[2 * Ht + i])
                 for i, a in enumerate(_matvec(iv, Wih[2 * Ht:], e))]
        n = [_itanh(iv, t) for t in pre_n]
        s = [(1 - z[i]) * n[i] for i in range(Ht)]  # h0 = 0
        l1, l2 = m.proposal_mlp[0], m.proposal_mlp[2]
        h1 = [a + _I(iv, b) for a, b in zip(_matvec(iv, l1.weight.detach().double().numpy(), s),
                                            l1.bias.detach().double().numpy())]
        h1 = [_isilu(iv, t) for t in h1]
        h2 = [a + _I(iv, b) for a, b in zip(_matvec(iv, l2.weight.detach().double().numpy(), h1),
                                            l2.bias.detach().double().numpy())]
        p = [_itanh(iv, t) for t in h2]
        addr = _ivec(iv, m.addr_embed.weight[0].detach().double().numpy())
        return addr + p, p

    def affine_inputs(self, x):
        iv = self.iv
        a_r = [a + _I(iv, self.b_ir[i]) + _I(iv, self.b_hr[i]) for i, a in enumerate(_matvec(iv, self.W_ir, x))]
        a_z = [a + _I(iv, self.b_iz[i]) + _I(iv, self.b_hz[i]) for i, a in enumerate(_matvec(iv, self.W_iz, x))]
        a_n = [a + _I(iv, self.b_in[i]) for i, a in enumerate(_matvec(iv, self.W_in, x))]
        return a_r, a_z, a_n

    def step(self, m, aff):
        """Interval GRU update (PyTorch convention). Returns dict of parts."""
        iv = self.iv
        a_r, a_z, a_n = aff
        pre_r = [a_r[i] + h for i, h in enumerate(_matvec(iv, self.H_r, m))]
        pre_z = [a_z[i] + h for i, h in enumerate(_matvec(iv, self.H_z, m))]
        t = [h + _I(iv, self.b_hn[i]) for i, h in enumerate(_matvec(iv, self.H_n, m))]
        r = [_isig(iv, u) for u in pre_r]
        z = [_isig(iv, u) for u in pre_z]
        pre_n = [a_n[i] + r[i] * t[i] for i in range(self.H)]
        n = [_itanh(iv, u) for u in pre_n]
        out = [z[i] * m[i] + (1 - z[i]) * n[i] for i in range(self.H)]
        return {"out": out, "r": r, "z": z, "n": n, "t": t,
                "pre_r": pre_r, "pre_z": pre_z, "pre_n": pre_n}

    def jacobian_rowsum_sup(self, m, aff):
        """Sup over the box m of the infinity-norm (max abs row sum) of dU/dm."""
        iv = self.iv
        H = self.H
        p = self.step(m, aff)
        sd_r = [_sigd_range(iv, u) for u in p["pre_r"]]
        sd_z = [_sigd_range(iv, u) for u in p["pre_z"]]
        td_n = [_tanhd_range(iv, u) for u in p["pre_n"]]
        worst_b = iv.mpf(0).b
        for i in range(H):
            # rows of J_r, J_z restricted as needed
            row = []
            for j in range(H):
                Jr_ij = sd_r[i] * _I(iv, self.H_r[i, j])
                Jz_ij = sd_z[i] * _I(iv, self.H_z[i, j])
                Jn_ij = td_n[i] * (p["r"][i] * _I(iv, self.H_n[i, j]) + p["t"][i] * Jr_ij)
                JU_ij = (p["z"][i] if i == j else iv.mpf(0)) \
                    + (m[i] - p["n"][i]) * Jz_ij + (1 - p["z"][i]) * Jn_ij
                row.append(iv.mpf(_sup_abs(JU_ij)))   # interval so the sum rounds outward
            s = sum(row, iv.mpf(0))
            # endpoint-safe comparison (mpmath interval comparisons are three-valued)
            if s.b > worst_b:
                worst_b = s.b
        return worst_b


def _reader_iv(iv, model, m, legend):
    """Interval reader: returns (p_slot0, g=s0-s1, gradient-1-norm-sup)."""
    l1, l2 = model.reader[0], model.reader[2]
    W1 = l1.weight.detach().double().numpy()
    b1 = l1.bias.detach().double().numpy()
    w2 = l2.weight.detach().double().numpy().ravel()
    b2 = float(l2.bias.detach().double().numpy().ravel()[0])
    dc, db, da = model.update_gru.hidden_size, model.d_bypass, model.addr_embed.embedding_dim
    A = W1[:, :dc]
    W_addr = W1[:, dc + db:dc + db + da]
    W_cand = W1[:, dc + db + da:]
    addr = _ivec(iv, model.addr_embed.weight[0].detach().double().numpy())
    Am = _matvec(iv, A, m)
    c_addr = _matvec(iv, W_addr, addr)
    pre = []
    for j in range(2):
        cand = _ivec(iv, model.value_embed.weight[legend[j]].detach().double().numpy())
        c_cand = _matvec(iv, W_cand, cand)
        # fixed offsets enclosed in interval arithmetic (evidence-gate fix)
        c = [c_addr[i] + c_cand[i] + _I(iv, b1[i]) for i in range(len(b1))]
        pre.append([Am[i] + c[i] for i in range(len(c))])
    scores = []
    for j in range(2):
        acc = iv.mpf(b2)
        for i in range(len(w2)):
            acc += _I(iv, w2[i]) * _isilu(iv, pre[j][i])
        scores.append(acc)
    g = scores[0] - scores[1]
    p0 = _isig(iv, g)
    # gradient of g w.r.t. m: A^T (w2 * (phi'(pre0) - phi'(pre1)))
    d = [_I(iv, w2[i]) * (_isilu_d(iv, pre[0][i]) - _isilu_d(iv, pre[1][i])) for i in range(len(w2))]
    grad = _matvec(iv, A.T, d)
    grad1 = sum(_sup_abs(x) for x in grad)
    return p0, g, iv.mpf(grad1).b


def single_cell_certificate(model, config, device="cpu"):
    """Run the ten-step certificate on a trained sealed model."""
    iv = _cert_backend()
    t0 = time.time()
    limit = config.get("certificate_time_limit_s", 60)
    rep = {"backend": "mpmath.iv", "prec_bits": 128, "norm": "infinity",
           "steps": {}, "first_unmet": None}

    def unmet(step, msg):
        if rep["first_unmet"] is None:
            rep["first_unmet"] = f"{step}: {msg}"

    cell = _CellIV(iv, model)
    H = cell.H
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        unmet("1", "non-finite coefficient")
    rep["reporting_note"] = ("reported floats are round-to-nearest images of 128-bit endpoints; "
                             "inequalities were decided on the 128-bit endpoints, not on these floats")
    # Step 1: interface, inputs, intended laws
    xs, props = {}, {}
    for v in range(N_VALUES):
        xs[v], props[v] = cell.input_intervals(v)
    rep["steps"]["1_interface"] = {
        "actions": ["write(0)", "write(1)", "read"],
        "bypass_sealed_zero": bool(model.seal),
        "proposal_enclosure_width_max": max(float(p.delta.b) for v in props for p in props[v]),
        "intended_laws": "nu_{legend,v} = delta_{index_legend(v)}", "Delta": 1.0,
    }
    if not model.seal:
        unmet("1", "model is not sealed")

    # Step 2: domain
    m0 = model.m0.detach().double().numpy()
    Hbox = max(1.0, float(np.max(np.abs(m0))))
    Hbox_iv = iv.mpf(Hbox)
    m_box = [iv.mpf([-Hbox, Hbox]) for _ in range(H)]
    rep["steps"]["2_domain"] = {"H": Hbox, "m0_inf_norm": float(np.max(np.abs(m0)))}

    # Step 3: invariance U_v(D) subset D
    inv = {}
    affs = {v: cell.affine_inputs(xs[v]) for v in range(N_VALUES)}
    for v in range(N_VALUES):
        p = cell.step(m_box, affs[v])
        zr = [(float(z.a), float(z.b)) for z in p["z"]]
        nr = [(float(n.a), float(n.b)) for n in p["n"]]
        ok = all(0 <= z.a and z.b <= 1 for z in p["z"]) and all(-1 <= n.a and n.b <= 1 for n in p["n"])
        # hull of convex combination with shared z: extremes at endpoint triples
        hull_ok = True
        for i in range(H):
            zi, ni = p["z"][i], p["n"][i]
            ext = []   # endpoint triples evaluated in interval arithmetic (outward)
            for zz in (zi.a, zi.b):
                for xx in (-Hbox_iv, Hbox_iv):
                    for yy in (ni.a, ni.b):
                        ext.append(iv.mpf(zz) * xx + (1 - iv.mpf(zz)) * iv.mpf(yy))
            lo = min(e_.a for e_ in ext); hi = max(e_.b for e_ in ext)
            if lo < -Hbox_iv.a or hi > Hbox_iv.b:
                hull_ok = False
        inv[v] = {"gates_in_unit": ok, "hull_in_box": hull_ok,
                  "z_range_min_max": [min(a for a, _ in zr), max(b for _, b in zr)],
                  "n_range_min_max": [min(a for a, _ in nr), max(b for _, b in nr)]}
        if not (ok and hull_ok):
            unmet("3", f"invariance failed for v={v}")
    rep["steps"]["3_invariance"] = inv

    # Step 4: contraction (interval Jacobian, infinity norm)
    kappa = iv.mpf(0)
    per_v = {}
    for v in range(N_VALUES):
        k_v = cell.jacobian_rowsum_sup(m_box, affs[v])
        per_v[v] = float(k_v)
        if k_v > kappa:
            kappa = iv.mpf(k_v)
    kappa_hat = kappa.b
    rep["steps"]["4_contraction"] = {"kappa_hat": float(kappa_hat), "per_value": per_v,
                                     "contractive": bool(kappa_hat < 1),
                                     "recipe_needs_kappa_below_one_third": bool(kappa_hat < iv.mpf(1) / 3)}
    if not kappa_hat < 1:
        unmet("4", f"kappa_hat={float(kappa_hat):.4f} >= 1 (inconclusive: interval bound)")
    if time.time() - t0 > limit:
        unmet("4", "time limit")

    # Step 5: fixed points
    fx = {}
    q = {}
    # Candidate centers only (float32 iteration); enclosure is proved below
    # by the interval residual, so the iteration's precision is immaterial.
    with torch.no_grad():
        for v in range(N_VALUES):
            x = torch.tensor([[v]], dtype=torch.long)
            s = model._encode(x[:, 0]); p = model._propose(s)
            addr = torch.zeros(1, dtype=torch.long)
            m = model.m0.unsqueeze(0).clone()
            for _ in range(config.get("certificate_fixed_point_iters", 64)):
                m = model._update(m, p, addr)
            q[v] = m[0].numpy().astype(np.float64)
    e = {}
    for v in range(N_VALUES):
        qv = _ivec(iv, q[v])
        out = cell.step(qv, affs[v])["out"]
        resid = iv.mpf(max(_sup_abs(out[i] - qv[i]) for i in range(H))).b
        ev = (iv.mpf(resid) / (1 - iv.mpf(kappa_hat))).b if kappa_hat < 1 else None
        e[v] = ev
        fx[v] = {"residual_hat": float(resid), "e_v": (float(ev) if ev is not None else None)}
    if kappa_hat < 1:
        Dstar = (iv.mpf(float(np.max(np.abs(q[0] - q[1])))) + iv.mpf(e[0]) + iv.mpf(e[1])).b
        Rstar = (iv.mpf(kappa_hat) * iv.mpf(Dstar) / (1 - iv.mpf(kappa_hat))).b
    else:
        Dstar = Rstar = None
    rep["steps"]["5_fixed_points"] = {"per_value": fx,
                                      "D_star_hat": (float(Dstar) if Dstar is not None else None),
                                      "R_star_hat": (float(Rstar) if Rstar is not None else None)}

    # Step 6: initialization coverage and tube closure
    R = None
    if kappa_hat < 1:
        m0_iv = _ivec(iv, m0)
        Rinit = iv.mpf(0)
        init = {}
        for v in range(N_VALUES):
            E_v = (iv.mpf(float(np.max(np.abs(m0 - q[v])))) + iv.mpf(e[v])).b
            out = cell.step(m0_iv, affs[v])["out"]
            B_v = (iv.mpf(max(_sup_abs(out[i] - _I(iv, q[v][i])) for i in range(H))) + iv.mpf(e[v])).b
            cand = min(B_v, (iv.mpf(kappa_hat) * iv.mpf(E_v)).b)
            init[v] = {"E_v": float(E_v), "B_v": float(B_v), "first_write_radius": float(cand)}
            if cand > Rinit:
                Rinit = iv.mpf(cand)
        R = max(iv.mpf(Rstar).b, Rinit.b)
        closure_ok = bool((iv.mpf(kappa_hat) * (iv.mpf(R) + iv.mpf(Dstar))).b <= R)
        rep["steps"]["6_initialization"] = {"per_value": init, "R_init": float(Rinit.b),
                                            "R": float(R), "tube_closure_ok": closure_ok}
        if not closure_ok:
            unmet("6", "tube closure inequality failed")
    else:
        rep["steps"]["6_initialization"] = {"skipped": "kappa_hat >= 1"}

    # Step 7: reader Lipschitz constant on D (both legends)
    L_hat = iv.mpf(0)
    for legend in ([0, 1], [1, 0]):
        _, _, g1 = _reader_iv(iv, model, m_box, legend)
        Lc = (iv.mpf(g1) / 4).b
        if Lc > L_hat:
            L_hat = iv.mpf(Lc)
    L_hat = L_hat.b
    rep["steps"]["7_reader_lipschitz"] = {"L_hat": float(L_hat), "softmax_factor": 0.25,
                                          "note": "SiLU not globally 1-Lipschitz; gradient enclosed directly"}

    # Step 8: calibration at fixed-point enclosures
    beta = iv.mpf(0)
    cal = {}
    if kappa_hat < 1:
        for v in range(N_VALUES):
            # enclosure endpoints in interval arithmetic, intersected with D (evidence-gate fix)
            Pv = [iv.mpf([max((_I(iv, q[v][i]) - iv.mpf(e[v])).a, -Hbox_iv.a),
                          min((_I(iv, q[v][i]) + iv.mpf(e[v])).b, Hbox_iv.b)]) for i in range(H)]
            if any(abs(q[v][i]) > Hbox for i in range(H)):
                unmet("8", f"fixed-point candidate for v={v} outside D")
            for legend in ([0, 1], [1, 0]):
                p0, _, _ = _reader_iv(iv, model, Pv, legend)
                idx = legend.index(v)
                p_correct = p0 if idx == 0 else 1 - p0
                b = (1 - iv.mpf(p_correct.a)).b
                cal[f"v{v}_legend{legend}"] = {"p_correct_lower": float(p_correct.a), "tv_to_intended": float(b)}
                if b > beta:
                    beta = iv.mpf(b)
    beta_hat = beta.b
    rep["steps"]["8_calibration"] = {"beta_hat": float(beta_hat), "per_cell": cal, "Delta": 1.0}

    # Step 9: usefulness
    if kappa_hat < 1 and R is not None:
        eta = (iv.mpf(L_hat) * iv.mpf(R) + iv.mpf(beta_hat)).b
        useful = bool((2 * iv.mpf(eta)).b < 1)
        rep["steps"]["9_usefulness"] = {"eta": float(eta), "two_eta": float((2 * iv.mpf(eta)).b),
                                        "two_eta_below_one": useful,
                                        "read_closure": "identity transition; automatic"}
        if not useful:
            unmet("9", f"2*eta = {float(2*eta):.4f} >= 1")
    else:
        eta = None
        rep["steps"]["9_usefulness"] = {"skipped": "no contraction certificate"}

    rep["elapsed_s"] = time.time() - t0
    if rep["first_unmet"] is None:
        rep["verdict"] = "CERTIFICATE_PASS"
        rep["licensed"] = {
            "correct_slot_probability_lower_bound_any_history": float((1 - iv.mpf(eta)).a),
            "same_last_value_histories_d_infinity_upper_bound": float((2 * iv.mpf(eta)).b),
            "different_live_values_immediate_separation_lower_bound": float((1 - 2 * iv.mpf(eta)).a),
            "scope": "real-arithmetic network with saved weights; single address; two values; "
                     "training wording; registered actions write(0), write(1), read; "
                     "not address selection, two registers, unseen wording, seal advantage, "
                     "or native structure of a pretrained model",
        }
    else:
        rep["verdict"] = "CERTIFICATE_INCONCLUSIVE"
        rep["licensed"] = {"note": "sufficient certificate not established; this does not refute "
                                   "behavioral overwrite (audit #40)"}
    return rep


def run_single_cell_certificate(config, out_dir):
    """Train one sealed cell (seed from config) and certify it."""
    import hashlib
    device = "cpu"
    seed = config["seeds"][0]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "result.json")

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    wv_train, lg_train, tu_train, tv_train = make_training_data(config["n_train"], seed, device)
    eps = make_all_episodes()
    wv_eval, lg_eval, tu_eval, tv_eval = episodes_to_tensors(eps, device)

    torch.manual_seed(seed)
    model = CommitSealRung1(config["d_embed"], config["d_transient"], config["d_proposal"],
                            config["d_carrier"], config["d_addr"], config["d_bypass"], seal=True)
    t0 = time.time()
    torch.manual_seed(seed + 1000)
    history = train_model(model, wv_train, lg_train, tu_train, tv_train,
                          n_epochs=config["n_epochs"], lr=config["lr"],
                          batch_size=config["batch_size"],
                          weight_decay=config.get("weight_decay", 1e-4),
                          grad_clip=config.get("grad_clip", 1.0))
    train_s = time.time() - t0
    accs = eval_accuracy(model, wv_eval, lg_eval, tu_eval, tv_eval)
    erasure = measure_paired_erasure(model, device)

    weights = {k: _pt_hex(v) for k, v in model.state_dict().items()}
    whash = hashlib.sha256("".join(weights[k]["hex"] for k in sorted(weights)).encode()).hexdigest()
    with open(__file__, "rb") as f:
        src_hash = hashlib.sha256(f.read()).hexdigest()
    cfg_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    output = {"config": config, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "mode": "single_cell_certificate", "seed": seed, "arm": "sealed",
              "train_elapsed_s": train_s, "final_loss": history["loss"][-1],
              "diagnostic_accuracy": accs, "diagnostic_paired_erasure_max_tv": erasure,
              "weights": weights, "weights_sha256": whash, "source_sha256": src_hash,
              "config_sha256": cfg_hash,
              "versions": {"torch": torch.__version__, "numpy": np.__version__}}
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)   # weights saved BEFORE certification
    print(f"  Trained seed {seed} sealed in {train_s:.1f}s; acc={accs['two_write_acc']:.3f}; "
          f"erasure TV={erasure:.2e}; weights saved.", flush=True)

    cert = single_cell_certificate(model, config, device)
    output["certificate"] = cert
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Certificate: {cert['verdict']} in {cert['elapsed_s']:.1f}s; "
          f"first unmet: {cert['first_unmet']}", flush=True)
    for k, v in cert["steps"].items():
        print(f"    {k}: {json.dumps(v, default=str)[:300]}")
    return output


def load_model_from_result(path):
    """Rebuild the model from the hex weights stored in a certificate result."""
    with open(path) as f:
        res = json.load(f)
    c = res["config"]
    model = CommitSealRung1(c["d_embed"], c["d_transient"], c["d_proposal"],
                            c["d_carrier"], c["d_addr"], c["d_bypass"], seal=True)
    sd = {}
    for k, w in res["weights"].items():
        a = np.frombuffer(bytes.fromhex(w["hex"]), dtype=np.dtype(w["dtype"])).reshape(w["shape"])
        sd[k] = torch.from_numpy(a.copy())
    model.load_state_dict(sd)
    model.eval()
    return model, res


def witness_search(path, n_pairs=20000, seed=0):
    """POST-HOC DIAGNOSTIC (not a certificate). On the certificate's fixed
    domain D=[-H,H]^d and infinity norm, search for a pair x,y in D with
    ||U_v(x)-U_v(y)|| > ||x-y|| and confirm any witness with interval
    arithmetic. A confirmed witness shows the contraction PREMISE fails on
    D for that value (the interval bound was not merely loose). Also reports
    sampled Jacobian norms at the fixed points (descriptive only)."""
    iv = _cert_backend()
    model, res = load_model_from_result(path)
    cell = _CellIV(iv, model)
    Hbox = res["certificate"]["steps"]["2_domain"]["H"]
    rng = np.random.default_rng(seed)
    d = cell.H
    out = {"n_pairs": n_pairs, "H": Hbox, "witness": {}, "sampled_jacobian_inf_norm": {}}
    with torch.no_grad():
        for v in range(N_VALUES):
            x_in = torch.tensor([v]); s = model._encode(x_in); p = model._propose(s)
            addr = torch.zeros(1, dtype=torch.long)
            X = torch.tensor(rng.uniform(-Hbox, Hbox, size=(n_pairs, d)), dtype=torch.float32)
            Y = torch.tensor(rng.uniform(-Hbox, Hbox, size=(n_pairs, d)), dtype=torch.float32)
            # local pairs too (small perturbations expose derivative > 1)
            Y[n_pairs // 2:] = X[n_pairs // 2:] + 0.01 * torch.tensor(
                rng.uniform(-1, 1, size=(n_pairs - n_pairs // 2, d)), dtype=torch.float32)
            Y = Y.clamp(-Hbox, Hbox)
            pp = p.expand(n_pairs, -1); aa = addr.expand(n_pairs)
            UX = model._update(X, pp, aa); UY = model._update(Y, pp, aa)
            num = (UX - UY).abs().max(1).values; den = (X - Y).abs().max(1).values
            ratio = num / den.clamp_min(1e-12)
            k = int(torch.argmax(ratio))
            best = float(ratio[k])
            # interval confirmation of the best pair
            aff = cell.affine_inputs(cell.input_intervals(v)[0])
            xi = _ivec(iv, X[k].numpy()); yi = _ivec(iv, Y[k].numpy())
            ux = cell.step(xi, aff)["out"]; uy = cell.step(yi, aff)["out"]
            num_lo = max((ux[i] - uy[i]).a if (ux[i] - uy[i]).a > 0 else -(ux[i] - uy[i]).b
                         for i in range(d))
            num_lo = max(num_lo, iv.mpf(0).a)
            den_hi = max(_sup_abs(xi[i] - yi[i]) for i in range(d))
            confirmed = bool(num_lo > den_hi)
            out["witness"][v] = {"best_sampled_ratio_float32": best, "confirmed_expansion": confirmed,
                                 "interval_num_lower": float(num_lo), "interval_den_upper": float(den_hi),
                                 "pair_index": k, "local_pair": bool(k >= n_pairs // 2),
                                 "x_hex": X[k].numpy().tobytes().hex(), "y_hex": Y[k].numpy().tobytes().hex(),
                                 "search_seed": seed}
            # sampled Jacobian norms at fixed point and m0 (descriptive)
            def jn(m):
                m = m.clone().requires_grad_(True)
                J = torch.autograd.functional.jacobian(lambda mm: model._update(mm.unsqueeze(0), p, addr)[0], m)
                return float(J.abs().sum(1).max())
            m = model.m0.detach().clone()
            for _ in range(64):
                m = model._update(m.unsqueeze(0), p, addr)[0]
            out["sampled_jacobian_inf_norm"][v] = {"at_64step_candidate": jn(m.detach()), "at_m0": jn(model.m0.detach())}
    import hashlib
    with open(__file__, "rb") as f:
        out["source_sha256"] = hashlib.sha256(f.read()).hexdigest()
    res["post_hoc_diagnostics"] = {"witness_search": out,
                                   "note": "post hoc; descriptive; not part of the certificate; "
                                           "a confirmed pair disproves nonexpansiveness on D in the inf-norm"}
    with open(path, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(out, indent=1, default=str))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Commit-Seal Carrier (Rung 1)")
    parser.add_argument("--config", required=False)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--witness-search", default=None,
                        help="Post-hoc diagnostic on an existing certificate result.json")
    args = parser.parse_args()

    if args.witness_search:
        witness_search(args.witness_search)
        return

    with open(args.config) as f:
        config = json.load(f)

    out_dir = args.output_dir or (
        f"experiments/results/{config.get('experiment_id', 'commit_seal_rung1')}")
    if config.get("mode") == "single_cell_certificate":
        run_single_cell_certificate(config, out_dir)
        return
    results = run_experiment(config, out_dir)

    output = {"config": config, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    output.update(results)

    path = os.path.join(out_dir, "result.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
