"""
Write-Boundary Construction Experiment (CR-14)

Tests whether an explicit overwrite mechanism produces compositional
compression that a matched append-only carrier does not.

Two architectures, same task, same capacity:
1. Overwrite: GRU controller + hard-masked register file
2. Append-only: GRU controller + append-only log buffer

Task: fact-tracking with overwrites. k registers, |V| values each.
N-step write sequences + 1 query. Correct answer = last write to
queried register.

Primary measurement: behavioral quotient |Q_H| for each model.
Decision gate F1: if both quotients are equal, the write boundary
doesn't help.
"""

import argparse
import gc
import itertools
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


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_write_sequence(n_steps, n_registers, n_values, rng):
    """Generate a random write sequence and the ground-truth store state."""
    writes = []
    store = [0] * n_registers  # initial state: all registers = 0
    for _ in range(n_steps):
        reg = rng.randint(0, n_registers - 1)
        val = rng.randint(0, n_values - 1)
        writes.append((reg, val))
        store[reg] = val
    return writes, store


def encode_sequence(writes, query_reg, n_registers, n_values):
    """Encode a write sequence + query as integer tokens.

    Token encoding:
    - Write tokens: reg * n_values + val  (range [0, n_registers * n_values))
    - Query tokens: n_registers * n_values + query_reg
    """
    n_write_tokens = n_registers * n_values
    tokens = []
    for reg, val in writes:
        tokens.append(reg * n_values + val)
    tokens.append(n_write_tokens + query_reg)
    return tokens


def make_dataset(n_samples, n_steps, n_registers, n_values, seed):
    """Generate a dataset of (token_sequence, target_value) pairs."""
    rng = random.Random(seed)
    sequences = []
    targets = []
    for _ in range(n_samples):
        writes, store = generate_write_sequence(n_steps, n_registers, n_values, rng)
        query_reg = rng.randint(0, n_registers - 1)
        tokens = encode_sequence(writes, query_reg, n_registers, n_values)
        target = store[query_reg]
        sequences.append(tokens)
        targets.append(target)
    return sequences, targets


def enumerate_all_histories(n_steps, n_registers, n_values):
    """Enumerate all possible write histories for quotient measurement.

    Returns list of (writes, store) tuples.
    """
    all_writes = list(itertools.product(range(n_registers), range(n_values)))
    histories = []
    for combo in itertools.product(all_writes, repeat=n_steps):
        writes = list(combo)
        store = [0] * n_registers
        for reg, val in writes:
            store[reg] = val
        histories.append((writes, list(store)))
    return histories


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class OverwriteModel(nn.Module):
    """GRU controller with hard-masked register file."""

    def __init__(self, n_registers, n_values, d_embed, d_hidden):
        super().__init__()
        self.n_registers = n_registers
        self.n_values = n_values
        self.d_hidden = d_hidden

        n_tokens = n_registers * n_values + n_registers  # writes + queries
        self.embed = nn.Embedding(n_tokens, d_embed)

        # GRU input: embedding + register file (flattened)
        reg_size = n_registers * n_values
        self.gru = nn.GRUCell(d_embed + reg_size, d_hidden)

        # Write head: predict (register, value) from hidden state
        self.write_reg_head = nn.Linear(d_hidden, n_registers)
        self.write_val_head = nn.Linear(d_hidden, n_values)

        # Output head: predict value from hidden state + register readout
        self.output_head = nn.Linear(d_hidden + n_values, n_values)

    def forward(self, token_seqs):
        """Process a batch of token sequences.

        Args:
            token_seqs: (batch, seq_len) integer tensor

        Returns:
            logits: (batch, n_values) — prediction for the query
        """
        batch_size, seq_len = token_seqs.shape
        device = token_seqs.device

        # Initialize register file: all zeros (one-hot for value 0)
        reg_file = torch.zeros(batch_size, self.n_registers, self.n_values,
                               device=device)
        reg_file[:, :, 0] = 1.0  # initial value = 0 for all registers

        # Initialize hidden state
        h = torch.zeros(batch_size, self.d_hidden, device=device)

        n_write_tokens = self.n_registers * self.n_values

        for t in range(seq_len):
            tok = token_seqs[:, t]
            emb = self.embed(tok)  # (batch, d_embed)

            # Flatten register file for GRU input
            reg_flat = reg_file.reshape(batch_size, -1)  # (batch, n_reg * n_val)
            gru_input = torch.cat([emb, reg_flat], dim=1)

            h = self.gru(gru_input, h)

            # Determine if this is a write step (not a query)
            is_write = (tok < n_write_tokens).float().unsqueeze(1)  # (batch, 1)

            if t < seq_len - 1:  # all but the last step are writes
                reg_idx = tok // self.n_values  # (batch,)
                val_idx = tok % self.n_values   # (batch,)

                # Hard write: overwrite the register (vectorized)
                val_onehot = F.one_hot(val_idx, self.n_values).float()
                reg_onehot = F.one_hot(reg_idx, self.n_registers).float()

                # mask: (batch, n_reg, 1) — 1 for the written register
                mask = reg_onehot.unsqueeze(2)
                # new_vals: (batch, n_reg, n_val) — broadcast new value
                new_vals = val_onehot.unsqueeze(1).expand_as(reg_file)
                reg_file = reg_file * (1 - mask) + new_vals * mask

        # Last step is the query — extract query register
        query_tok = token_seqs[:, -1]
        query_reg = query_tok - n_write_tokens  # (batch,)

        # Read the queried register (vectorized)
        query_idx = query_reg.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.n_values)
        reg_readout = reg_file.gather(1, query_idx).squeeze(1)  # (batch, n_values)

        # Predict from hidden state + register readout
        logits = self.output_head(torch.cat([h, reg_readout], dim=1))
        return logits

    def get_behavioral_state(self, token_seqs):
        """Get the behavioral response profile for quotient measurement.

        Returns softmax distributions for ALL possible queries.
        """
        batch_size = token_seqs.shape[0]
        device = token_seqs.device
        n_write_tokens = self.n_registers * self.n_values

        # Process write steps only (all but the last token)
        write_seqs = token_seqs[:, :-1]

        # Initialize
        reg_file = torch.zeros(batch_size, self.n_registers, self.n_values,
                               device=device)
        reg_file[:, :, 0] = 1.0
        h = torch.zeros(batch_size, self.d_hidden, device=device)

        for t in range(write_seqs.shape[1]):
            tok = write_seqs[:, t]
            emb = self.embed(tok)
            reg_flat = reg_file.reshape(batch_size, -1)
            gru_input = torch.cat([emb, reg_flat], dim=1)
            h = self.gru(gru_input, h)

            reg_idx = tok // self.n_values
            val_idx = tok % self.n_values
            val_onehot = F.one_hot(val_idx, self.n_values).float()
            reg_onehot = F.one_hot(reg_idx, self.n_registers).float()
            mask = reg_onehot.unsqueeze(2)
            new_vals = val_onehot.unsqueeze(1).expand_as(reg_file)
            reg_file = reg_file * (1 - mask) + new_vals * mask

        # Now query each register and collect response profiles
        profiles = []
        for qr in range(self.n_registers):
            query_tok = torch.full((batch_size,), n_write_tokens + qr,
                                   dtype=torch.long, device=device)
            emb = self.embed(query_tok)
            reg_flat = reg_file.reshape(batch_size, -1)
            gru_input = torch.cat([emb, reg_flat], dim=1)
            h_q = self.gru(gru_input, h)

            reg_readout = reg_file[:, qr]  # (batch, n_values)
            logits = self.output_head(torch.cat([h_q, reg_readout], dim=1))
            probs = F.softmax(logits, dim=1)  # (batch, n_values)
            profiles.append(probs)

        # Stack: (batch, n_registers, n_values)
        return torch.stack(profiles, dim=1)


class AppendOnlyModel(nn.Module):
    """GRU controller with append-only log buffer (no overwrite)."""

    def __init__(self, n_registers, n_values, d_embed, d_hidden, max_steps):
        super().__init__()
        self.n_registers = n_registers
        self.n_values = n_values
        self.d_hidden = d_hidden
        self.max_steps = max_steps

        n_tokens = n_registers * n_values + n_registers
        self.embed = nn.Embedding(n_tokens, d_embed)

        # GRU input: embedding + log summary (attention over log entries)
        log_entry_size = n_registers + n_values  # one-hot reg + one-hot val
        self.log_proj = nn.Linear(log_entry_size, d_embed)
        self.attn_query = nn.Linear(d_hidden, d_embed)

        self.gru = nn.GRUCell(d_embed + d_embed, d_hidden)  # emb + log_summary

        # Output head: predict value from hidden state
        self.output_head = nn.Linear(d_hidden, n_values)

    def _attend_log(self, h, log_entries, log_mask):
        """Attend over the log buffer using the hidden state as query."""
        # h: (batch, d_hidden)
        # log_entries: (batch, max_steps, log_entry_size)
        # log_mask: (batch, max_steps) — 1 for valid entries, 0 for empty

        batch_size = h.shape[0]

        # Project log entries
        log_proj = self.log_proj(log_entries)  # (batch, max_steps, d_embed)

        # Attention query from hidden state
        q = self.attn_query(h).unsqueeze(1)  # (batch, 1, d_embed)

        # Dot-product attention
        scores = (q * log_proj).sum(dim=2)  # (batch, max_steps)
        scores = scores.masked_fill(log_mask == 0, float('-inf'))

        has_entries = log_mask.any(dim=1)  # (batch,)
        weights = torch.zeros_like(scores)
        if has_entries.any():
            safe_scores = scores.clone()
            safe_scores[~has_entries] = 0.0
            weights[has_entries] = F.softmax(scores[has_entries], dim=1)

        weights = weights.unsqueeze(2)  # (batch, max_steps, 1)
        summary = (weights * log_proj).sum(dim=1)  # (batch, d_embed)
        return summary

    def _build_log(self, write_tokens, device):
        """Pre-build log entries from write tokens (no in-place ops)."""
        # write_tokens: (batch, n_write_steps)
        batch_size, n_write_steps = write_tokens.shape

        reg_idx = write_tokens // self.n_values  # (batch, n_write_steps)
        val_idx = write_tokens % self.n_values

        reg_onehot = F.one_hot(reg_idx, self.n_registers).float()
        val_onehot = F.one_hot(val_idx, self.n_values).float()
        log_entries = torch.cat([reg_onehot, val_onehot], dim=2)
        # log_entries: (batch, n_write_steps, n_registers + n_values)
        return log_entries

    def forward(self, token_seqs):
        batch_size, seq_len = token_seqs.shape
        device = token_seqs.device
        n_write_steps = seq_len - 1

        # Pre-build the full log from write tokens
        write_tokens = token_seqs[:, :n_write_steps]
        full_log = self._build_log(write_tokens, device)

        h = torch.zeros(batch_size, self.d_hidden, device=device)

        for t in range(seq_len):
            tok = token_seqs[:, t]
            emb = self.embed(tok)

            # Build incremental log mask (only entries up to current step)
            if t > 0:
                log_slice = full_log[:, :t]
                log_mask = torch.ones(batch_size, t, device=device)
            else:
                log_slice = full_log[:, :1]  # dummy, will be masked
                log_mask = torch.zeros(batch_size, 1, device=device)

            log_summary = self._attend_log(h, log_slice, log_mask)
            gru_input = torch.cat([emb, log_summary], dim=1)
            h = self.gru(gru_input, h)

        logits = self.output_head(h)
        return logits

    def get_behavioral_state(self, token_seqs):
        batch_size = token_seqs.shape[0]
        device = token_seqs.device
        n_write_tokens = self.n_registers * self.n_values

        write_seqs = token_seqs[:, :-1]
        n_write_steps = write_seqs.shape[1]

        # Pre-build the full log
        full_log = self._build_log(write_seqs, device)

        h = torch.zeros(batch_size, self.d_hidden, device=device)

        for t in range(n_write_steps):
            tok = write_seqs[:, t]
            emb = self.embed(tok)

            if t > 0:
                log_slice = full_log[:, :t]
                log_mask = torch.ones(batch_size, t, device=device)
            else:
                log_slice = full_log[:, :1]
                log_mask = torch.zeros(batch_size, 1, device=device)

            log_summary = self._attend_log(h, log_slice, log_mask)
            gru_input = torch.cat([emb, log_summary], dim=1)
            h = self.gru(gru_input, h)

        # Full log available for query step
        full_mask = torch.ones(batch_size, n_write_steps, device=device)

        profiles = []
        for qr in range(self.n_registers):
            query_tok = torch.full((batch_size,), n_write_tokens + qr,
                                   dtype=torch.long, device=device)
            emb = self.embed(query_tok)
            log_summary = self._attend_log(h, full_log, full_mask)
            gru_input = torch.cat([emb, log_summary], dim=1)
            h_q = self.gru(gru_input, h)
            logits = self.output_head(h_q)
            probs = F.softmax(logits, dim=1)
            profiles.append(probs)

        return torch.stack(profiles, dim=1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, train_seqs, train_targets, val_seqs, val_targets,
                n_epochs, lr, batch_size, device):
    """Train a model and return training history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_x = torch.tensor(train_seqs, dtype=torch.long, device=device)
    train_y = torch.tensor(train_targets, dtype=torch.long, device=device)
    val_x = torch.tensor(val_seqs, dtype=torch.long, device=device)
    val_y = torch.tensor(val_targets, dtype=torch.long, device=device)

    n_train = len(train_seqs)
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        epoch_correct = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            x_batch = train_x[idx]
            y_batch = train_y[idx]

            logits = model(x_batch)
            loss = F.cross_entropy(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
            epoch_correct += (logits.argmax(1) == y_batch).sum().item()

        avg_loss = epoch_loss / n_train
        train_acc = epoch_correct / n_train

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_acc = (val_logits.argmax(1) == val_y).float().mean().item()

        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}: loss={avg_loss:.4f} "
                  f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}",
                  flush=True)

    return history


# ---------------------------------------------------------------------------
# Quotient measurement
# ---------------------------------------------------------------------------

def measure_quotient(model, histories, n_registers, n_values, tolerance,
                     device, batch_size=256):
    """Compute the behavioral quotient |Q_H| for a trained model.

    Groups write histories by their response profile across all queries.
    Two histories are equivalent if their response distributions differ
    by less than `tolerance` (max TV distance across all queries).
    """
    model.eval()
    n_write_tokens = n_registers * n_values

    # Encode all histories as token sequences with a dummy query
    all_profiles = []

    for i in range(0, len(histories), batch_size):
        batch_histories = histories[i:i+batch_size]
        batch_tokens = []
        for writes, _ in batch_histories:
            tokens = []
            for reg, val in writes:
                tokens.append(reg * n_values + val)
            tokens.append(n_write_tokens)  # dummy query (will be overridden)
            batch_tokens.append(tokens)

        token_tensor = torch.tensor(batch_tokens, dtype=torch.long, device=device)

        with torch.no_grad():
            profiles = model.get_behavioral_state(token_tensor)
            # profiles: (batch, n_registers, n_values)

        all_profiles.append(profiles.cpu().numpy())

    all_profiles = np.concatenate(all_profiles, axis=0)
    # all_profiles: (n_histories, n_registers, n_values)

    # Flatten profiles for comparison
    flat_profiles = all_profiles.reshape(len(histories), -1)
    # flat_profiles: (n_histories, n_registers * n_values)

    # Cluster by similarity
    n = len(histories)
    cluster_ids = [-1] * n
    n_clusters = 0
    cluster_centers = []

    for i in range(n):
        assigned = False
        for c_idx, center in enumerate(cluster_centers):
            # Max TV distance across queries
            diff = np.abs(flat_profiles[i] - center)
            # Reshape to (n_registers, n_values) and compute TV per register
            diff_reshaped = diff.reshape(n_registers, n_values)
            tv_per_reg = 0.5 * diff_reshaped.sum(axis=1)
            max_tv = tv_per_reg.max()

            if max_tv < tolerance:
                cluster_ids[i] = c_idx
                assigned = True
                break

        if not assigned:
            cluster_ids[i] = n_clusters
            cluster_centers.append(flat_profiles[i].copy())
            n_clusters += 1

    # Verify: check how many distinct store states map to each cluster
    store_to_clusters = {}
    for i, (writes, store) in enumerate(histories):
        store_key = tuple(store)
        if store_key not in store_to_clusters:
            store_to_clusters[store_key] = set()
        store_to_clusters[store_key].add(cluster_ids[i])

    # Store alignment: each store state should map to exactly one cluster
    store_alignment = sum(1 for s, cs in store_to_clusters.items() if len(cs) == 1)
    store_alignment_rate = store_alignment / len(store_to_clusters)

    return {
        'n_quotient_classes': n_clusters,
        'n_histories': len(histories),
        'n_store_states': len(store_to_clusters),
        'store_alignment_rate': store_alignment_rate,
        'compression_ratio': len(histories) / max(n_clusters, 1),
        'tolerance': tolerance,
    }


# ---------------------------------------------------------------------------
# Generalization test (longer sequences)
# ---------------------------------------------------------------------------

def test_generalization(model, n_registers, n_values, n_steps_test,
                        n_samples, seed, device):
    """Test model on longer sequences than training."""
    seqs, targets = make_dataset(n_samples, n_steps_test, n_registers,
                                 n_values, seed)
    x = torch.tensor(seqs, dtype=torch.long, device=device)
    y = torch.tensor(targets, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        acc = (logits.argmax(1) == y).float().mean().item()

    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Write-Boundary Construction Experiment (CR-14)')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # Extract config
    n_registers = cfg['n_registers']
    n_values = cfg['n_values']
    n_steps = cfg['n_steps']
    d_embed = cfg['d_embed']
    d_hidden = cfg['d_hidden']
    n_train = cfg['n_train']
    n_val = cfg['n_val']
    n_epochs = cfg['n_epochs']
    lr = cfg['lr']
    batch_size = cfg['batch_size']
    tolerance = cfg['quotient_tolerance']
    seed = cfg['seed']

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cpu')

    store_cardinality = n_values ** n_registers
    history_space = (n_registers * n_values) ** n_steps

    print(f"=== Write-Boundary Construction Experiment (CR-14) ===")
    print(f"Registers: {n_registers}, Values: {n_values}, Steps: {n_steps}")
    print(f"Store cardinality |S|: {store_cardinality}")
    print(f"History space: {history_space}")
    print(f"Expected compression: {history_space}:{store_cardinality} "
          f"= {history_space/store_cardinality:.0f}:1")
    print(f"Device: {device}")
    print(flush=True)

    # Generate datasets
    print("\nGenerating datasets...", flush=True)
    train_seqs, train_targets = make_dataset(n_train, n_steps, n_registers,
                                              n_values, seed)
    val_seqs, val_targets = make_dataset(n_val, n_steps, n_registers,
                                          n_values, seed + 1000)

    # Enumerate all histories for quotient measurement
    print(f"Enumerating all {history_space} histories for quotient...",
          flush=True)
    all_histories = enumerate_all_histories(n_steps, n_registers, n_values)
    print(f"  Enumerated {len(all_histories)} histories", flush=True)

    results = {
        'config': cfg,
        'store_cardinality': store_cardinality,
        'history_space': history_space,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    # -----------------------------------------------------------------------
    # Train and evaluate Overwrite model
    # -----------------------------------------------------------------------
    print("\n--- Overwrite Model ---", flush=True)
    overwrite_model = OverwriteModel(n_registers, n_values, d_embed, d_hidden)
    n_params_ow = sum(p.numel() for p in overwrite_model.parameters())
    print(f"  Parameters: {n_params_ow}", flush=True)

    ow_history = train_model(overwrite_model, train_seqs, train_targets,
                              val_seqs, val_targets, n_epochs, lr, batch_size,
                              device)

    print("  Measuring quotient...", flush=True)
    ow_quotient = measure_quotient(overwrite_model, all_histories, n_registers,
                                    n_values, tolerance, device)
    print(f"  |Q_H| = {ow_quotient['n_quotient_classes']} "
          f"(store states: {ow_quotient['n_store_states']}, "
          f"compression: {ow_quotient['compression_ratio']:.1f}:1, "
          f"store alignment: {ow_quotient['store_alignment_rate']:.3f})")

    # Generalization to longer sequences
    gen_results_ow = {}
    for test_len in cfg.get('generalization_lengths', [7, 10, 15]):
        acc = test_generalization(overwrite_model, n_registers, n_values,
                                  test_len, 1000, seed + 2000 + test_len,
                                  device)
        gen_results_ow[str(test_len)] = acc
        print(f"  Generalization (len={test_len}): {acc:.3f}", flush=True)

    results['overwrite'] = {
        'n_params': n_params_ow,
        'final_train_acc': ow_history['train_acc'][-1],
        'final_val_acc': ow_history['val_acc'][-1],
        'final_train_loss': ow_history['train_loss'][-1],
        'quotient': ow_quotient,
        'generalization': gen_results_ow,
    }

    gc.collect()

    # -----------------------------------------------------------------------
    # Train and evaluate Append-Only model
    # -----------------------------------------------------------------------
    print("\n--- Append-Only Model ---", flush=True)
    max_steps = max(n_steps, max(cfg.get('generalization_lengths', [n_steps])))
    appendonly_model = AppendOnlyModel(n_registers, n_values, d_embed,
                                       d_hidden, max_steps)
    n_params_ao = sum(p.numel() for p in appendonly_model.parameters())
    print(f"  Parameters: {n_params_ao}", flush=True)

    ao_history = train_model(appendonly_model, train_seqs, train_targets,
                              val_seqs, val_targets, n_epochs, lr, batch_size,
                              device)

    print("  Measuring quotient...", flush=True)
    ao_quotient = measure_quotient(appendonly_model, all_histories, n_registers,
                                    n_values, tolerance, device)
    print(f"  |Q_H| = {ao_quotient['n_quotient_classes']} "
          f"(store states: {ao_quotient['n_store_states']}, "
          f"compression: {ao_quotient['compression_ratio']:.1f}:1, "
          f"store alignment: {ao_quotient['store_alignment_rate']:.3f})")

    gen_results_ao = {}
    for test_len in cfg.get('generalization_lengths', [7, 10, 15]):
        acc = test_generalization(appendonly_model, n_registers, n_values,
                                  test_len, 1000, seed + 2000 + test_len,
                                  device)
        gen_results_ao[str(test_len)] = acc
        print(f"  Generalization (len={test_len}): {acc:.3f}", flush=True)

    results['append_only'] = {
        'n_params': n_params_ao,
        'final_train_acc': ao_history['train_acc'][-1],
        'final_val_acc': ao_history['val_acc'][-1],
        'final_train_loss': ao_history['train_loss'][-1],
        'quotient': ao_quotient,
        'generalization': gen_results_ao,
    }

    gc.collect()

    # -----------------------------------------------------------------------
    # Comparative analysis
    # -----------------------------------------------------------------------
    print("\n=== Comparative Analysis ===", flush=True)

    ow_q = ow_quotient['n_quotient_classes']
    ao_q = ao_quotient['n_quotient_classes']

    results['comparison'] = {
        'overwrite_quotient': ow_q,
        'appendonly_quotient': ao_q,
        'quotient_ratio': ao_q / max(ow_q, 1),
        'store_cardinality': store_cardinality,
        'overwrite_at_bound': ow_q <= store_cardinality * 1.5,
        'f1_gate': 'PASS' if ao_q > ow_q * 1.5 else 'FAIL',
    }

    print(f"  Overwrite |Q_H|: {ow_q} (bound: {store_cardinality})")
    print(f"  Append-only |Q_H|: {ao_q}")
    print(f"  Quotient ratio (AO/OW): {ao_q/max(ow_q,1):.2f}")
    print(f"  Overwrite at bound: {results['comparison']['overwrite_at_bound']}")
    print(f"  F1 gate: {results['comparison']['f1_gate']}")

    if ow_q <= store_cardinality * 1.5 and ao_q > ow_q * 1.5:
        verdict = "PASS — write boundary produces compositional compression"
    elif ow_q <= store_cardinality * 1.5 and ao_q <= ow_q * 1.5:
        verdict = "F4 — both compress; task may be too simple"
    elif ow_q > store_cardinality * 1.5:
        verdict = "F2 — overwrite model leaks history despite boundary"
    else:
        verdict = "INCONCLUSIVE"

    results['verdict'] = verdict
    print(f"\n  VERDICT: {verdict}")

    # Save results
    out_path = os.path.join(args.output_dir, 'result.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
