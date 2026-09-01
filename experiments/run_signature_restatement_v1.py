"""
Signature-indexed restatement S^G_g (v1).

Tests whether a restatement constructed from the model's own observable
greedy signature — not the experimenter-known world — retains the algebraic
properties established for S^W_w in v2:
  1. Approximate idempotence: (S^G_g)^2 ~= S^G_g
  2. Descent to greedy quotient G: all fiber members map to same target
  3. Non-naturality with correction: typed square S^G_{g'} . C vs C . S^G_g

The prediction: S^G should fix the one descent failure in v2 (fiber spanning
worlds w101/w111 with different hidden worlds but same greedy signature),
because S^G uses only the shared observable, not the divergent hidden world.

For the typed square, S^G_{g'} after correction requires running the
corrected history through the model to get its NEW greedy signature g',
then building the restatement from g'. This is the key construction.

Phase 4d is the terminal anti-echo factorial. Run its tokenizer-only checks
with --phase4d-preflight. The scientific run is isolated behind
--phase4d-only and refuses to overwrite its terminal result artifact.
"""

import argparse
import sys
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import json
import os
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
import hashlib

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DEVICE = "cpu"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "signature_restatement_v1")

PHASE4D_ALIASES = ("Q7", "V4", "J2")
PHASE4D_DOMAINS = {
    "registered": ("big", "small", "hot", "cold", "red", "blue"),
    "heldout": ("fast", "slow", "tall", "short", "loud", "quiet"),
}
PHASE4D_COUNTERFACTUAL = {
    "big": "small", "small": "big",
    "hot": "cold", "cold": "hot",
    "red": "blue", "blue": "red",
    "fast": "slow", "slow": "fast",
    "tall": "short", "short": "tall",
    "loud": "quiet", "quiet": "loud",
}
PHASE4D_LOCK = {
    "registration_id": "signature-restatement-phase4d-terminal-v1",
    "status": "PRE_REGISTERED_UNRUN",
    "primary_estimand": "world-cluster mean target-minus-counterfactual logit contrast for counterfactual-then-target minus target-then-counterfactual",
    "bootstrap": {"resamples": 10000, "seed": 43117, "cluster": "entity_set + semantic_world"},
    "integrity_interface": {
        "domain_valid_rate_each_set_min": 0.95,
        "target_only_follow_each_set_min": 0.90,
        "counterfactual_only_follow_each_set_min": 0.60,
        "counterfactual_only_cluster_ci_low_each_set_min": 0.40,
        "require_all_ordered_prompt_token_multisets_equal": True,
        "require_all_counterfactuals_fixed_point_free": True,
        "require_exact_alias_counterbalance": True,
        "require_single_token_value_verbalizers": True,
    },
    "direct_recency": {
        "last_block_follow_each_set_min": 0.70,
        "last_block_follow_ci_low_each_set_min": 0.50,
        "target_rate_order_effect_each_set_min": 0.30,
        "target_rate_order_effect_ci_low_each_set_strictly_above": 0.0,
        "logit_order_effect_each_set_min": 1.0,
        "logit_order_effect_ci_low_each_set_strictly_above": 0.0,
    },
    "alias_necessity": {
        "counterfactual_follow_each_set_min": 0.60,
        "counterfactual_follow_ci_low_each_set_min": 0.40,
        "target_rate_drop_each_set_min": 0.30,
        "target_rate_drop_ci_low_each_set_strictly_above": 0.0,
        "logit_shift_each_set_min": 1.0,
        "logit_shift_ci_low_each_set_strictly_above": 0.0,
        "require_every_alias_map_point_shift_positive": True,
    },
    "alias_anti_echo": {
        "matched_target_rate_effect_each_set_min": 0.30,
        "matched_target_rate_effect_ci_low_each_set_strictly_above": 0.0,
        "matched_logit_effect_each_set_min": 1.0,
        "matched_logit_effect_ci_low_each_set_strictly_above": 0.0,
        "discordant_alias_last_follow_each_set_min": 0.60,
        "discordant_alias_last_follow_ci_low_each_set_min": 0.40,
        "require_every_alias_map_point_shift_positive": True,
    },
    "terminal_rule": "Any non-pass ends alias/renderer tuning. RECENCY_EXPLAINS takes precedence over alias outcomes. A narrow anti-echo pass is not evidence of a latent invariant or native mathematics.",
}

REGISTERED_ENTITIES = {
    "ZOG": ("big", "small"),
    "MIP": ("hot", "cold"),
    "PLIM": ("red", "blue"),
}

HELDOUT_ENTITIES = {
    "KROT": ("fast", "slow"),
    "HESK": ("tall", "short"),
    "VORN": ("loud", "quiet"),
}


def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )


def load_model():
    tok = load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32,
        device_map=DEVICE, trust_remote_code=True
    )
    model.eval()
    return model, tok


def get_dist(model, tok, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    return F.softmax(logits, dim=-1)


def js_dist(p, q):
    m = (p + q) / 2
    eps = 1e-10
    jsd = (
        0.5 * ((p + eps) * ((p + eps) / (m + eps)).log()).sum()
        + 0.5 * ((q + eps) * ((q + eps) / (m + eps)).log()).sum()
    )
    return math.sqrt(max(0, float(jsd)))


def shannon_entropy(p):
    eps = 1e-10
    return float(-(p * (p + eps).log2()).sum())


def task_kernel(dist, tok, correct_val, wrong_val):
    correct_ids = set(tok.encode(f" {correct_val}", add_special_tokens=False))
    correct_ids.update(tok.encode(correct_val, add_special_tokens=False))
    wrong_ids = set(tok.encode(f" {wrong_val}", add_special_tokens=False))
    wrong_ids.update(tok.encode(wrong_val, add_special_tokens=False))
    wrong_ids -= correct_ids
    correct_mass = sum(float(dist[tid]) for tid in correct_ids)
    wrong_mass = sum(float(dist[tid]) for tid in wrong_ids)
    other_mass = 1.0 - correct_mass - wrong_mass
    return {
        "correct": round(correct_mass, 6),
        "wrong": round(wrong_mass, 6),
        "other": round(other_mass, 6),
    }


def make_worlds(entities):
    names = list(entities.keys())
    n = len(names)
    worlds = {}
    for i in range(2**n):
        bits = format(i, f'0{n}b')
        world = {}
        for j, name in enumerate(names):
            world[name] = entities[name][int(bits[j])]
        key = f"w{''.join(bits)}"
        worlds[key] = world
    return worlds


def make_history(world, entities, order="std"):
    names = list(entities.keys())
    if order == "rev":
        names = names[::-1]
    parts = [f"{n}: {world[n]}" for n in names]
    return ". ".join(parts) + "."


def make_restatement_from_world(world, entities):
    """S^W_w — world-conditioned (kept for comparison only)."""
    names = list(entities.keys())
    parts = [f"{n}: {world[n]}" for n in names]
    return " To be clear: " + ". ".join(parts) + "."


def make_restatement_from_signature(sig, entity_names):
    """S^G_g — signature-indexed restatement from observable greedy answers.

    sig: tuple of (entity_name, greedy_answer) pairs, sorted by entity name.
    entity_names: list of entity names (for ordering).
    """
    sig_dict = dict(sig)
    parts = [f"{n}: {sig_dict[n]}" for n in entity_names]
    return " To be clear: " + ". ".join(parts) + "."


def make_shuffled_restatement(sig, entity_names):
    """Control: same format, same tokens, but values rotated among entities."""
    sig_dict = dict(sig)
    values = [sig_dict[n] for n in entity_names]
    rotated = values[1:] + values[:1]
    parts = [f"{n}: {rotated[i]}" for i, n in enumerate(entity_names)]
    return " To be clear: " + ". ".join(parts) + "."


ENTITY_ALIASES = {
    "registered": {"ZOG": "cedar", "MIP": "amber", "PLIM": "violet"},
    "heldout": {"KROT": "maple", "HESK": "coral", "VORN": "slate"},
}


def make_decoy(sig, entity_names, entities):
    """Append wrong direct assignments for every entity (anti-signature)."""
    sig_dict = dict(sig)
    parts = []
    for n in entity_names:
        vals = entities[n]
        decoy_val = [v for v in vals if v != sig_dict[n]]
        if decoy_val:
            parts.append(f"{n}: {decoy_val[0]}")
        else:
            parts.append(f"{n}: {sig_dict[n]}")
    return " To be clear: " + ". ".join(parts) + "."


def make_alias_clause(entity_names, set_name):
    """Define the nonce alias mapping."""
    aliases = ENTITY_ALIASES[set_name]
    parts = [f"{aliases[n]} means {n}" for n in entity_names]
    return " In the coded record, " + ", ".join(parts[:-1]) + ", and " + parts[-1] + "."


def make_alias_restatement(sig, entity_names, set_name):
    """Faithful alias rendering: correct values via nonce aliases."""
    aliases = ENTITY_ALIASES[set_name]
    sig_dict = dict(sig)
    parts = [f"{aliases[n]} has value {sig_dict[n]}" for n in entity_names]
    return " The coded record says: " + ". ".join(parts) + "."


def make_shuffled_alias_restatement(sig, entity_names, set_name):
    """Shuffled alias rendering: same aliases, wrong value-alias pairing."""
    aliases = ENTITY_ALIASES[set_name]
    sig_dict = dict(sig)
    values = [sig_dict[n] for n in entity_names]
    rotated = values[1:] + values[:1]
    parts = [f"{aliases[n]} has value {rotated[i]}" for i, n in enumerate(entity_names)]
    return " The coded record says: " + ". ".join(parts) + "."


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def phase4d_signature_string(sig):
    return "|".join(f"{key}={value}" for key, value in sig)


def make_phase4d_history(world, entities, order, set_name):
    """Base carrier with an explicit shared value type for every entity."""
    domain = ", ".join(PHASE4D_DOMAINS[set_name])
    declaration = (
        "Every named item has exactly one value from this shared value "
        f"vocabulary: {domain}. "
    )
    return declaration + make_history(world, entities, order)


def make_phase4d_record(sig, entity_names=None, label_for_entity=None):
    """Common direct/alias assignment grammar used by every Phase 4d arm."""
    sig_dict = dict(sig)
    if entity_names is None:
        entity_names = [name for name, _ in sig]
    if label_for_entity is None:
        ordered_labels = [(name, name) for name in entity_names]
    else:
        entity_for_label = {label: entity for entity, label in label_for_entity.items()}
        ordered_labels = [(label, entity_for_label[label]) for label in PHASE4D_ALIASES]
    lines = [f"{label}: {sig_dict[entity]}" for label, entity in ordered_labels]
    return "\nRecord:\n" + "\n".join(lines) + "\nEnd record."


def make_phase4d_alias_maps(entity_names):
    """Three Latin-square maps: every entity occupies every alias/position."""
    maps = []
    n = len(entity_names)
    if n != len(PHASE4D_ALIASES):
        raise ValueError("Phase 4d requires exactly three entities and aliases")
    for shift in range(n):
        entity_to_alias = {
            entity_names[(index + shift) % n]: alias
            for index, alias in enumerate(PHASE4D_ALIASES)
        }
        maps.append({
            "alias_map_id": f"latin_shift_{shift}",
            "entity_to_alias": entity_to_alias,
        })
    return maps


def make_phase4d_alias_key(entity_to_alias):
    entity_for_alias = {alias: entity for entity, alias in entity_to_alias.items()}
    lines = [f"{alias} means {entity_for_alias[alias]}." for alias in PHASE4D_ALIASES]
    return "\nKey:\n" + "\n".join(lines) + "\nEnd key."


def phase4d_token_multiset(tok, prompt):
    token_ids = tok(prompt, add_special_tokens=False).input_ids
    counts = Counter(int(token_id) for token_id in token_ids)
    canonical = json.dumps(sorted(counts.items()), separators=(",", ":"))
    return {
        "token_count": len(token_ids),
        "multiset_sha256": sha256_text(canonical),
        "counts": counts,
    }


def phase4d_value_token_ids(tok, set_name):
    token_ids = {}
    failures = []
    for value in PHASE4D_DOMAINS[set_name]:
        ids = tok.encode(f" {value}", add_special_tokens=False)
        token_ids[value] = [int(token_id) for token_id in ids]
        if len(ids) != 1:
            failures.append({"value": value, "token_ids": token_ids[value]})
    return token_ids, failures


def phase4d_alias_balance(entity_names):
    maps = make_phase4d_alias_maps(entity_names)
    counts = {entity: {alias: 0 for alias in PHASE4D_ALIASES} for entity in entity_names}
    positions = {entity: {str(position): 0 for position in range(len(PHASE4D_ALIASES))} for entity in entity_names}
    for mapping in maps:
        entity_to_alias = mapping["entity_to_alias"]
        for entity, alias in entity_to_alias.items():
            counts[entity][alias] += 1
            positions[entity][str(PHASE4D_ALIASES.index(alias))] += 1
    exact = all(
        count == 1
        for table in (counts, positions)
        for entity_counts in table.values()
        for count in entity_counts.values()
    )
    return {"exact": exact, "alias_counts": counts, "position_counts": positions, "maps": maps}


def evaluate_phase4d_prompt(model, tok, prompt, target_value, counterfactual_value,
                            target_token_id, counterfactual_token_id):
    encoded = tok(prompt, return_tensors="pt")
    ids = encoded.input_ids.to(DEVICE)
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    probs = F.softmax(logits, dim=-1)
    greedy_id = int(torch.argmax(logits).item())
    greedy_raw = tok.decode([greedy_id])
    greedy = greedy_raw.strip()
    if greedy == target_value:
        output_class = "target"
    elif greedy == counterfactual_value:
        output_class = "counterfactual"
    else:
        output_class = "other"
    top_values, top_indices = torch.topk(probs, 5)
    top5 = []
    for probability, token_id in zip(top_values.tolist(), top_indices.tolist()):
        top5.append({
            "token_id": int(token_id),
            "token_text": tok.decode([int(token_id)]),
            "probability": round(float(probability), 9),
            "logit": round(float(logits[int(token_id)]), 9),
        })
    target_logit = float(logits[target_token_id])
    counterfactual_logit = float(logits[counterfactual_token_id])
    prompt_ids = encoded.input_ids[0].tolist()
    return {
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "prompt_token_count": len(prompt_ids),
        "greedy_token_id": greedy_id,
        "greedy_token_raw": greedy_raw,
        "greedy_output": greedy,
        "output_class": output_class,
        "other_output": greedy_raw if output_class == "other" else None,
        "target_token_id": int(target_token_id),
        "counterfactual_token_id": int(counterfactual_token_id),
        "target_logit": round(target_logit, 9),
        "counterfactual_logit": round(counterfactual_logit, 9),
        "target_minus_counterfactual_logit": round(target_logit - counterfactual_logit, 9),
        "target_probability": round(float(probs[target_token_id]), 9),
        "counterfactual_probability": round(float(probs[counterfactual_token_id]), 9),
        "top5": top5,
    }


def phase4d_cluster_stat(records, label):
    """Equal-world estimate and deterministic percentile interval."""
    grouped = defaultdict(list)
    for record in records:
        grouped[record["cluster"]].append(float(record["value"]))
    cluster_means = {
        cluster: statistics.mean(values) for cluster, values in grouped.items()
    }
    if not cluster_means:
        return {
            "label": label, "estimate": None, "ci_low": None, "ci_high": None,
            "n_clusters": 0, "n_nested_rows": 0, "cluster_means": {},
        }
    ordered = sorted(cluster_means)
    point = statistics.mean(cluster_means.values())
    seed_offset = int(sha256_text(label)[:8], 16)
    rng = random.Random(PHASE4D_LOCK["bootstrap"]["seed"] + seed_offset)
    boot = []
    for _ in range(PHASE4D_LOCK["bootstrap"]["resamples"]):
        sampled = [cluster_means[rng.choice(ordered)] for _ in ordered]
        boot.append(statistics.mean(sampled))
    boot.sort()
    lower_index = int(0.025 * (len(boot) - 1))
    upper_index = int(0.975 * (len(boot) - 1))
    return {
        "label": label,
        "estimate": round(point, 9),
        "ci_low": round(boot[lower_index], 9),
        "ci_high": round(boot[upper_index], 9),
        "n_clusters": len(cluster_means),
        "n_nested_rows": sum(len(values) for values in grouped.values()),
        "cluster_means": {key: round(value, 9) for key, value in sorted(cluster_means.items())},
    }


def phase4d_arm_stat(rows, arm, value_fn, label, alias_map_id=None):
    selected = []
    for row in rows:
        if row["arm"] != arm:
            continue
        if alias_map_id is not None and row["alias_map_id"] != alias_map_id:
            continue
        selected.append({"cluster": row["cluster"], "value": value_fn(row)})
    return phase4d_cluster_stat(selected, label)


def phase4d_paired_records(rows, high_arm, low_arm, value_fn, pair_label,
                           alias_map_id=None):
    by_arm = {high_arm: {}, low_arm: {}}
    for row in rows:
        if row["arm"] not in by_arm:
            continue
        if alias_map_id is not None and row["alias_map_id"] != alias_map_id:
            continue
        key = (row["source"], row["entity"], row["alias_map_id"])
        by_arm[row["arm"]][key] = row
    if set(by_arm[high_arm]) != set(by_arm[low_arm]):
        missing_high = sorted(set(by_arm[low_arm]) - set(by_arm[high_arm]))
        missing_low = sorted(set(by_arm[high_arm]) - set(by_arm[low_arm]))
        raise RuntimeError(
            f"Unpaired Phase 4d rows for {pair_label}: "
            f"missing_high={missing_high[:3]}, missing_low={missing_low[:3]}"
        )
    paired = []
    for key in sorted(by_arm[high_arm]):
        high = by_arm[high_arm][key]
        low = by_arm[low_arm][key]
        paired.append({
            "cluster": high["cluster"],
            "value": float(value_fn(high)) - float(value_fn(low)),
            "alias_map_id": high["alias_map_id"],
            "pair": pair_label,
        })
    return paired


def phase4d_preflight(tok):
    checks = {
        "registration_id": PHASE4D_LOCK["registration_id"],
        "model_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "sets": {},
        "counterfactual_involution": {},
    }
    all_values = [value for values in PHASE4D_DOMAINS.values() for value in values]
    checks["counterfactual_involution"] = {
        "covers_all_values": all(value in PHASE4D_COUNTERFACTUAL for value in all_values),
        "fixed_point_free": all(PHASE4D_COUNTERFACTUAL[value] != value for value in all_values),
        "involutive": all(
            PHASE4D_COUNTERFACTUAL[PHASE4D_COUNTERFACTUAL[value]] == value
            for value in all_values
        ),
    }
    for set_name, entities in [
        ("registered", REGISTERED_ENTITIES), ("heldout", HELDOUT_ENTITIES)
    ]:
        entity_names = list(entities)
        token_ids, token_failures = phase4d_value_token_ids(tok, set_name)
        balance = phase4d_alias_balance(entity_names)
        multiset_checks = []
        for world_key, world in make_worlds(entities).items():
            target_sig = tuple((entity, world[entity]) for entity in entity_names)
            counterfactual_sig = tuple(
                (entity, PHASE4D_COUNTERFACTUAL[world[entity]])
                for entity in entity_names
            )
            target_record = make_phase4d_record(target_sig, entity_names=entity_names)
            counterfactual_record = make_phase4d_record(counterfactual_sig, entity_names=entity_names)
            for order in ("std", "rev"):
                base = make_phase4d_history(world, entities, order, set_name)
                for entity in entity_names:
                    query = f"\n{entity}:"
                    cf_then_target = base + counterfactual_record + target_record + query
                    target_then_cf = base + target_record + counterfactual_record + query
                    first = phase4d_token_multiset(tok, cf_then_target)
                    second = phase4d_token_multiset(tok, target_then_cf)
                    multiset_checks.append({
                        "source": f"{world_key}_{order}",
                        "entity": entity,
                        "equal": first["counts"] == second["counts"],
                        "counterfactual_then_target_sha256": first["multiset_sha256"],
                        "target_then_counterfactual_sha256": second["multiset_sha256"],
                    })
        checks["sets"][set_name] = {
            "value_token_ids": token_ids,
            "single_token_value_verbalizers": len(token_failures) == 0,
            "single_token_failures": token_failures,
            "alias_balance": balance,
            "ordered_prompt_multisets_all_equal": all(item["equal"] for item in multiset_checks),
            "ordered_prompt_multiset_checks": multiset_checks,
            "alias_token_ids": {
                alias: tok.encode(alias, add_special_tokens=False)
                for alias in PHASE4D_ALIASES
            },
        }
    checks["passed"] = (
        all(checks["counterfactual_involution"].values())
        and all(
            set_checks["single_token_value_verbalizers"]
            and set_checks["alias_balance"]["exact"]
            and set_checks["ordered_prompt_multisets_all_equal"]
            for set_checks in checks["sets"].values()
        )
    )
    return checks


def run_phase4d_set(model, tok, entities, set_name, preflight):
    worlds = make_worlds(entities)
    entity_names = list(entities)
    alias_maps = make_phase4d_alias_maps(entity_names)
    value_token_ids = {
        value: ids[0]
        for value, ids in preflight["sets"][set_name]["value_token_ids"].items()
        if len(ids) == 1
    }
    domain = set(PHASE4D_DOMAINS[set_name])
    rows = []
    sources = []
    integrity = {
        "counterfactual_coordinate_checks": 0,
        "counterfactual_fixed_points": [],
        "ordered_prompt_multiset_checks": 0,
        "ordered_prompt_multiset_failures": [],
        "domain_valid_coordinates": 0,
        "domain_total_coordinates": 0,
        "eligible_sources": 0,
        "total_sources": 0,
    }

    for world_key, world in worlds.items():
        for order in ("std", "rev"):
            source = f"{world_key}_{order}"
            cluster = f"{set_name}:{world_key}"
            base = make_phase4d_history(world, entities, order, set_name)
            sig, margins = get_greedy_signature(model, tok, base, entity_names)
            sig_dict = dict(sig)
            invalid = {
                entity: value for entity, value in sig if value not in domain
            }
            integrity["total_sources"] += 1
            integrity["domain_total_coordinates"] += len(entity_names)
            integrity["domain_valid_coordinates"] += len(entity_names) - len(invalid)
            source_record = {
                "source": source,
                "world_key": world_key,
                "cluster": cluster,
                "presentation_order": order,
                "world": world,
                "base_prompt": base,
                "base_prompt_sha256": sha256_text(base),
                "observed_signature": phase4d_signature_string(sig),
                "observed_signature_margins": margins,
                "domain_invalid_outputs": invalid,
                "eligible": not invalid,
            }
            if invalid:
                sources.append(source_record)
                continue

            integrity["eligible_sources"] += 1
            counterfactual_sig = tuple(
                (entity, PHASE4D_COUNTERFACTUAL[sig_dict[entity]])
                for entity in entity_names
            )
            counterfactual_dict = dict(counterfactual_sig)
            source_record["counterfactual_signature"] = phase4d_signature_string(counterfactual_sig)
            sources.append(source_record)
            for entity in entity_names:
                integrity["counterfactual_coordinate_checks"] += 1
                if counterfactual_dict[entity] == sig_dict[entity]:
                    integrity["counterfactual_fixed_points"].append({
                        "source": source, "entity": entity, "value": sig_dict[entity]
                    })

            target_record = make_phase4d_record(sig, entity_names=entity_names)
            counterfactual_record = make_phase4d_record(
                counterfactual_sig, entity_names=entity_names
            )
            direct_arms = {
                "D0_base": "",
                "D1_target_only": target_record,
                "D2_counterfactual_only": counterfactual_record,
                "D3_counterfactual_then_target": counterfactual_record + target_record,
                "D4_target_then_counterfactual": target_record + counterfactual_record,
            }
            for entity in entity_names:
                query = f"\n{entity}:"
                prompt_d3 = base + direct_arms["D3_counterfactual_then_target"] + query
                prompt_d4 = base + direct_arms["D4_target_then_counterfactual"] + query
                multiset_d3 = phase4d_token_multiset(tok, prompt_d3)
                multiset_d4 = phase4d_token_multiset(tok, prompt_d4)
                integrity["ordered_prompt_multiset_checks"] += 1
                if multiset_d3["counts"] != multiset_d4["counts"]:
                    integrity["ordered_prompt_multiset_failures"].append({
                        "source": source,
                        "entity": entity,
                        "d3_multiset_sha256": multiset_d3["multiset_sha256"],
                        "d4_multiset_sha256": multiset_d4["multiset_sha256"],
                    })
                for arm, suffix in direct_arms.items():
                    prompt = base + suffix + query
                    evaluated = evaluate_phase4d_prompt(
                        model, tok, prompt,
                        sig_dict[entity], counterfactual_dict[entity],
                        value_token_ids[sig_dict[entity]],
                        value_token_ids[counterfactual_dict[entity]],
                    )
                    rows.append({
                        "entity_set": set_name,
                        "source": source,
                        "world_key": world_key,
                        "cluster": cluster,
                        "presentation_order": order,
                        "entity": entity,
                        "alias_map_id": None,
                        "arm": arm,
                        "target_value": sig_dict[entity],
                        "counterfactual_value": counterfactual_dict[entity],
                        **evaluated,
                    })

            for alias_mapping in alias_maps:
                alias_map_id = alias_mapping["alias_map_id"]
                entity_to_alias = alias_mapping["entity_to_alias"]
                key_block = make_phase4d_alias_key(entity_to_alias)
                alias_target = make_phase4d_record(
                    sig, entity_names=entity_names, label_for_entity=entity_to_alias
                )
                alias_counterfactual = make_phase4d_record(
                    counterfactual_sig, entity_names=entity_names,
                    label_for_entity=entity_to_alias
                )
                alias_arms = {
                    "A0_alias_target_only": key_block + alias_target,
                    "A1_alias_counterfactual_only": key_block + alias_counterfactual,
                    "A2_counterfactual_then_alias_target": counterfactual_record + key_block + alias_target,
                    "A3_counterfactual_then_alias_counterfactual": counterfactual_record + key_block + alias_counterfactual,
                    "A4_target_then_alias_target": target_record + key_block + alias_target,
                    "A5_target_then_alias_counterfactual": target_record + key_block + alias_counterfactual,
                }
                for entity in entity_names:
                    query = f"\n{entity}:"
                    for arm, suffix in alias_arms.items():
                        prompt = base + suffix + query
                        evaluated = evaluate_phase4d_prompt(
                            model, tok, prompt,
                            sig_dict[entity], counterfactual_dict[entity],
                            value_token_ids[sig_dict[entity]],
                            value_token_ids[counterfactual_dict[entity]],
                        )
                        rows.append({
                            "entity_set": set_name,
                            "source": source,
                            "world_key": world_key,
                            "cluster": cluster,
                            "presentation_order": order,
                            "entity": entity,
                            "alias_map_id": alias_map_id,
                            "alias_map": entity_to_alias,
                            "arm": arm,
                            "target_value": sig_dict[entity],
                            "counterfactual_value": counterfactual_dict[entity],
                            **evaluated,
                        })
    return {"sources": sources, "rows": rows, "integrity": integrity}


def analyze_phase4d_set(set_result, set_name):
    rows = set_result["rows"]
    integrity = set_result["integrity"]
    is_target = lambda row: 1.0 if row["output_class"] == "target" else 0.0
    is_counterfactual = lambda row: 1.0 if row["output_class"] == "counterfactual" else 0.0
    is_other = lambda row: 1.0 if row["output_class"] == "other" else 0.0
    logit_contrast = lambda row: row["target_minus_counterfactual_logit"]

    direct = {
        "target_only_follow": phase4d_arm_stat(
            rows, "D1_target_only", is_target, f"{set_name}:D1 target follow"
        ),
        "counterfactual_only_follow": phase4d_arm_stat(
            rows, "D2_counterfactual_only", is_counterfactual,
            f"{set_name}:D2 counterfactual follow"
        ),
        "other_rate_by_arm": {},
    }
    for arm in (
        "D0_base", "D1_target_only", "D2_counterfactual_only",
        "D3_counterfactual_then_target", "D4_target_then_counterfactual",
    ):
        direct["other_rate_by_arm"][arm] = phase4d_arm_stat(
            rows, arm, is_other, f"{set_name}:{arm} other rate"
        )
    order_target_records = phase4d_paired_records(
        rows, "D3_counterfactual_then_target", "D4_target_then_counterfactual",
        is_target, "D3 minus D4 target follow"
    )
    order_logit_records = phase4d_paired_records(
        rows, "D3_counterfactual_then_target", "D4_target_then_counterfactual",
        logit_contrast, "D3 minus D4 target-counterfactual logit"
    )
    direct["target_rate_order_effect"] = phase4d_cluster_stat(
        order_target_records, f"{set_name}:direct target-rate order effect"
    )
    direct["logit_order_effect"] = phase4d_cluster_stat(
        order_logit_records, f"{set_name}:direct logit order effect"
    )
    last_block_records = []
    for row in rows:
        if row["arm"] == "D3_counterfactual_then_target":
            last_block_records.append({"cluster": row["cluster"], "value": is_target(row)})
        elif row["arm"] == "D4_target_then_counterfactual":
            last_block_records.append({"cluster": row["cluster"], "value": is_counterfactual(row)})
    direct["last_block_follow"] = phase4d_cluster_stat(
        last_block_records, f"{set_name}:direct final-block follow"
    )

    necessity_target_drop_records = phase4d_paired_records(
        rows, "A0_alias_target_only", "A1_alias_counterfactual_only",
        is_target, "A0 minus A1 target follow"
    )
    necessity_logit_records = phase4d_paired_records(
        rows, "A0_alias_target_only", "A1_alias_counterfactual_only",
        logit_contrast, "A0 minus A1 target-counterfactual logit"
    )
    alias = {
        "necessity_counterfactual_follow": phase4d_arm_stat(
            rows, "A1_alias_counterfactual_only", is_counterfactual,
            f"{set_name}:A1 alias counterfactual follow"
        ),
        "necessity_target_rate_drop": phase4d_cluster_stat(
            necessity_target_drop_records, f"{set_name}:alias necessity target drop"
        ),
        "necessity_logit_shift": phase4d_cluster_stat(
            necessity_logit_records, f"{set_name}:alias necessity logit shift"
        ),
        "other_rate_by_arm": {},
    }
    for arm in (
        "A0_alias_target_only", "A1_alias_counterfactual_only",
        "A2_counterfactual_then_alias_target",
        "A3_counterfactual_then_alias_counterfactual",
        "A4_target_then_alias_target", "A5_target_then_alias_counterfactual",
    ):
        alias["other_rate_by_arm"][arm] = phase4d_arm_stat(
            rows, arm, is_other, f"{set_name}:{arm} other rate"
        )

    anti_target_records = []
    anti_logit_records = []
    for high_arm, low_arm, pair_label in (
        ("A2_counterfactual_then_alias_target", "A3_counterfactual_then_alias_counterfactual", "counterfactual direct context"),
        ("A4_target_then_alias_target", "A5_target_then_alias_counterfactual", "target direct context"),
    ):
        anti_target_records.extend(phase4d_paired_records(
            rows, high_arm, low_arm, is_target, f"anti target: {pair_label}"
        ))
        anti_logit_records.extend(phase4d_paired_records(
            rows, high_arm, low_arm, logit_contrast, f"anti logit: {pair_label}"
        ))
    alias["anti_echo_matched_target_rate_effect"] = phase4d_cluster_stat(
        anti_target_records, f"{set_name}:alias anti-echo matched target-rate effect"
    )
    alias["anti_echo_matched_logit_effect"] = phase4d_cluster_stat(
        anti_logit_records, f"{set_name}:alias anti-echo matched logit effect"
    )
    discordant_follow_records = []
    for row in rows:
        if row["arm"] == "A2_counterfactual_then_alias_target":
            discordant_follow_records.append({"cluster": row["cluster"], "value": is_target(row)})
        elif row["arm"] == "A5_target_then_alias_counterfactual":
            discordant_follow_records.append({"cluster": row["cluster"], "value": is_counterfactual(row)})
    alias["discordant_alias_last_follow"] = phase4d_cluster_stat(
        discordant_follow_records, f"{set_name}:discordant final-alias follow"
    )

    alias["necessity_logit_shift_by_alias_map"] = {}
    alias["anti_echo_logit_effect_by_alias_map"] = {}
    for alias_map_id in ("latin_shift_0", "latin_shift_1", "latin_shift_2"):
        necessity_map = phase4d_paired_records(
            rows, "A0_alias_target_only", "A1_alias_counterfactual_only",
            logit_contrast, f"necessity {alias_map_id}", alias_map_id=alias_map_id
        )
        alias["necessity_logit_shift_by_alias_map"][alias_map_id] = phase4d_cluster_stat(
            necessity_map, f"{set_name}:necessity {alias_map_id}"
        )
        anti_map = []
        for high_arm, low_arm in (
            ("A2_counterfactual_then_alias_target", "A3_counterfactual_then_alias_counterfactual"),
            ("A4_target_then_alias_target", "A5_target_then_alias_counterfactual"),
        ):
            anti_map.extend(phase4d_paired_records(
                rows, high_arm, low_arm, logit_contrast,
                f"anti {alias_map_id}", alias_map_id=alias_map_id
            ))
        alias["anti_echo_logit_effect_by_alias_map"][alias_map_id] = phase4d_cluster_stat(
            anti_map, f"{set_name}:anti-echo {alias_map_id}"
        )

    domain_valid_rate = (
        integrity["domain_valid_coordinates"] / integrity["domain_total_coordinates"]
        if integrity["domain_total_coordinates"] else 0.0
    )
    return {
        "entity_set": set_name,
        "support": {
            "domain_valid_coordinates": integrity["domain_valid_coordinates"],
            "domain_total_coordinates": integrity["domain_total_coordinates"],
            "domain_valid_rate": round(domain_valid_rate, 9),
            "eligible_sources": integrity["eligible_sources"],
            "total_sources": integrity["total_sources"],
            "semantic_world_clusters": len({row["cluster"] for row in rows}),
            "scored_rows": len(rows),
        },
        "direct": direct,
        "alias": alias,
    }


def adjudicate_phase4d(preflight, set_results, analyses):
    integrity_by_set = {}
    for set_name, set_result in set_results.items():
        integrity = set_result["integrity"]
        integrity_by_set[set_name] = {
            "no_counterfactual_fixed_points": not integrity["counterfactual_fixed_points"],
            "ordered_prompt_multisets_all_equal": not integrity["ordered_prompt_multiset_failures"],
            "alias_balance_exact": preflight["sets"][set_name]["alias_balance"]["exact"],
            "single_token_value_verbalizers": preflight["sets"][set_name]["single_token_value_verbalizers"],
        }
    gate0_integrity = preflight["passed"] and all(
        all(checks.values()) for checks in integrity_by_set.values()
    )

    interface_by_set = {}
    recency_by_set = {}
    necessity_by_set = {}
    anti_echo_by_set = {}
    for set_name, analysis in analyses.items():
        support = analysis["support"]
        direct = analysis["direct"]
        alias = analysis["alias"]
        interface_checks = {
            "domain_valid_rate": support["domain_valid_rate"] >= PHASE4D_LOCK["integrity_interface"]["domain_valid_rate_each_set_min"],
            "target_only_follow": direct["target_only_follow"]["estimate"] is not None and direct["target_only_follow"]["estimate"] >= PHASE4D_LOCK["integrity_interface"]["target_only_follow_each_set_min"],
            "counterfactual_only_follow": direct["counterfactual_only_follow"]["estimate"] is not None and direct["counterfactual_only_follow"]["estimate"] >= PHASE4D_LOCK["integrity_interface"]["counterfactual_only_follow_each_set_min"],
            "counterfactual_only_ci_low": direct["counterfactual_only_follow"]["ci_low"] is not None and direct["counterfactual_only_follow"]["ci_low"] >= PHASE4D_LOCK["integrity_interface"]["counterfactual_only_cluster_ci_low_each_set_min"],
        }
        interface_by_set[set_name] = {"passed": all(interface_checks.values()), "checks": interface_checks}

        recency_checks = {
            "last_block_follow": direct["last_block_follow"]["estimate"] is not None and direct["last_block_follow"]["estimate"] >= PHASE4D_LOCK["direct_recency"]["last_block_follow_each_set_min"],
            "last_block_follow_ci_low": direct["last_block_follow"]["ci_low"] is not None and direct["last_block_follow"]["ci_low"] >= PHASE4D_LOCK["direct_recency"]["last_block_follow_ci_low_each_set_min"],
            "target_rate_order_effect": direct["target_rate_order_effect"]["estimate"] is not None and direct["target_rate_order_effect"]["estimate"] >= PHASE4D_LOCK["direct_recency"]["target_rate_order_effect_each_set_min"],
            "target_rate_order_effect_ci_low": direct["target_rate_order_effect"]["ci_low"] is not None and direct["target_rate_order_effect"]["ci_low"] > PHASE4D_LOCK["direct_recency"]["target_rate_order_effect_ci_low_each_set_strictly_above"],
            "logit_order_effect": direct["logit_order_effect"]["estimate"] is not None and direct["logit_order_effect"]["estimate"] >= PHASE4D_LOCK["direct_recency"]["logit_order_effect_each_set_min"],
            "logit_order_effect_ci_low": direct["logit_order_effect"]["ci_low"] is not None and direct["logit_order_effect"]["ci_low"] > PHASE4D_LOCK["direct_recency"]["logit_order_effect_ci_low_each_set_strictly_above"],
        }
        recency_by_set[set_name] = {"passed": all(recency_checks.values()), "checks": recency_checks}

        necessity_map_positive = all(
            stat["estimate"] is not None and stat["estimate"] > 0.0
            for stat in alias["necessity_logit_shift_by_alias_map"].values()
        )
        necessity_checks = {
            "counterfactual_follow": alias["necessity_counterfactual_follow"]["estimate"] is not None and alias["necessity_counterfactual_follow"]["estimate"] >= PHASE4D_LOCK["alias_necessity"]["counterfactual_follow_each_set_min"],
            "counterfactual_follow_ci_low": alias["necessity_counterfactual_follow"]["ci_low"] is not None and alias["necessity_counterfactual_follow"]["ci_low"] >= PHASE4D_LOCK["alias_necessity"]["counterfactual_follow_ci_low_each_set_min"],
            "target_rate_drop": alias["necessity_target_rate_drop"]["estimate"] is not None and alias["necessity_target_rate_drop"]["estimate"] >= PHASE4D_LOCK["alias_necessity"]["target_rate_drop_each_set_min"],
            "target_rate_drop_ci_low": alias["necessity_target_rate_drop"]["ci_low"] is not None and alias["necessity_target_rate_drop"]["ci_low"] > PHASE4D_LOCK["alias_necessity"]["target_rate_drop_ci_low_each_set_strictly_above"],
            "logit_shift": alias["necessity_logit_shift"]["estimate"] is not None and alias["necessity_logit_shift"]["estimate"] >= PHASE4D_LOCK["alias_necessity"]["logit_shift_each_set_min"],
            "logit_shift_ci_low": alias["necessity_logit_shift"]["ci_low"] is not None and alias["necessity_logit_shift"]["ci_low"] > PHASE4D_LOCK["alias_necessity"]["logit_shift_ci_low_each_set_strictly_above"],
            "all_alias_maps_positive": necessity_map_positive,
        }
        necessity_by_set[set_name] = {"passed": all(necessity_checks.values()), "checks": necessity_checks}

        anti_map_positive = all(
            stat["estimate"] is not None and stat["estimate"] > 0.0
            for stat in alias["anti_echo_logit_effect_by_alias_map"].values()
        )
        anti_checks = {
            "matched_target_rate_effect": alias["anti_echo_matched_target_rate_effect"]["estimate"] is not None and alias["anti_echo_matched_target_rate_effect"]["estimate"] >= PHASE4D_LOCK["alias_anti_echo"]["matched_target_rate_effect_each_set_min"],
            "matched_target_rate_effect_ci_low": alias["anti_echo_matched_target_rate_effect"]["ci_low"] is not None and alias["anti_echo_matched_target_rate_effect"]["ci_low"] > PHASE4D_LOCK["alias_anti_echo"]["matched_target_rate_effect_ci_low_each_set_strictly_above"],
            "matched_logit_effect": alias["anti_echo_matched_logit_effect"]["estimate"] is not None and alias["anti_echo_matched_logit_effect"]["estimate"] >= PHASE4D_LOCK["alias_anti_echo"]["matched_logit_effect_each_set_min"],
            "matched_logit_effect_ci_low": alias["anti_echo_matched_logit_effect"]["ci_low"] is not None and alias["anti_echo_matched_logit_effect"]["ci_low"] > PHASE4D_LOCK["alias_anti_echo"]["matched_logit_effect_ci_low_each_set_strictly_above"],
            "discordant_alias_last_follow": alias["discordant_alias_last_follow"]["estimate"] is not None and alias["discordant_alias_last_follow"]["estimate"] >= PHASE4D_LOCK["alias_anti_echo"]["discordant_alias_last_follow_each_set_min"],
            "discordant_alias_last_follow_ci_low": alias["discordant_alias_last_follow"]["ci_low"] is not None and alias["discordant_alias_last_follow"]["ci_low"] >= PHASE4D_LOCK["alias_anti_echo"]["discordant_alias_last_follow_ci_low_each_set_min"],
            "all_alias_maps_positive": anti_map_positive,
        }
        anti_echo_by_set[set_name] = {"passed": all(anti_checks.values()), "checks": anti_checks}

    gate1_interface = gate0_integrity and all(item["passed"] for item in interface_by_set.values())
    recency_explains = gate1_interface and all(item["passed"] for item in recency_by_set.values())
    alias_necessity = gate1_interface and all(item["passed"] for item in necessity_by_set.values())
    alias_anti_echo = alias_necessity and all(item["passed"] for item in anti_echo_by_set.values())
    if not gate1_interface:
        verdict = "NO_INTERFACE_OR_INVALID__TERMINAL_DEMOTION"
        interpretation = "Mechanism gates are uninterpretable; terminal allocation ends renderer tuning and S^G receives no semantic upgrade."
    elif recency_explains:
        verdict = "RECENCY_EXPLAINS__DEMOTE_SG"
        interpretation = "Identical-token direct blocks follow crossed final-block order; demote S^G to a sequence-sensitive syntactic append operator regardless of alias outcomes."
    elif not alias_necessity:
        verdict = "ALIASES_UNINTERPRETABLE__TERMINAL_DEMOTION"
        interpretation = "The alias instrument did not move answers away from the base signature; do not interpret later alias arms, do not tune aliases again, and demote S^G by terminal allocation."
    elif not alias_anti_echo:
        verdict = "ANTI_ECHO_NONPASS__DEMOTE_SG"
        interpretation = "Aliases were behaviorally usable but did not clear the matched conflict gate; demote S^G and pivot."
    else:
        verdict = "NARROW_ANTI_ECHO_PASS"
        interpretation = "Keyed re-encoding beats the registered verbatim/recency explanations; this does not establish a latent invariant, semantic retraction, or native mathematics."
    return {
        "verdict": verdict,
        "interpretation": interpretation,
        "gate_0_integrity": {"passed": gate0_integrity, "by_set": integrity_by_set},
        "gate_1_interface": {"passed": gate1_interface, "by_set": interface_by_set},
        "gate_2_direct_recency": {"passed": recency_explains, "by_set": recency_by_set},
        "gate_3_alias_necessity": {"passed": alias_necessity, "by_set": necessity_by_set},
        "gate_4_alias_anti_echo": {"passed": alias_anti_echo, "by_set": anti_echo_by_set},
        "termination_rate": "not_applicable_one_step_next_token_scoring",
    }


def run_phase4d(model, tok):
    preflight = phase4d_preflight(tok)
    if not preflight["passed"]:
        raise RuntimeError("Phase 4d tokenizer/integrity preflight failed; no model factorial was run")
    set_results = {}
    analyses = {}
    for set_name, entities in (
        ("registered", REGISTERED_ENTITIES), ("heldout", HELDOUT_ENTITIES)
    ):
        print(f"\n--- Phase 4d terminal factorial: {set_name} ---")
        set_result = run_phase4d_set(model, tok, entities, set_name, preflight)
        set_results[set_name] = set_result
        analyses[set_name] = analyze_phase4d_set(set_result, set_name)
        print(json.dumps({
            "support": analyses[set_name]["support"],
            "direct": {
                "last_block_follow": analyses[set_name]["direct"]["last_block_follow"],
                "logit_order_effect": analyses[set_name]["direct"]["logit_order_effect"],
            },
            "alias": {
                "necessity_counterfactual_follow": analyses[set_name]["alias"]["necessity_counterfactual_follow"],
                "anti_echo_matched_logit_effect": analyses[set_name]["alias"]["anti_echo_matched_logit_effect"],
            },
        }, indent=2))
    adjudication = adjudicate_phase4d(preflight, set_results, analyses)
    return {
        "schema_version": "signature-restatement-phase4d-v1",
        "registration": PHASE4D_LOCK,
        "registration_sha256": sha256_text(json.dumps(PHASE4D_LOCK, sort_keys=True, separators=(",", ":"))),
        "preflight": preflight,
        "sets": set_results,
        "analysis": analyses,
        "adjudication": adjudication,
    }


def make_corrected_world(world, target_entity, new_value):
    corrected = dict(world)
    corrected[target_entity] = new_value
    return corrected


def get_greedy_signature(model, tok, base_prompt, entity_names):
    sig = {}
    margins = {}
    for ent in entity_names:
        prompt = base_prompt + f"\n{ent}:"
        dist = get_dist(model, tok, prompt)
        topk = torch.topk(dist, 2)
        greedy = tok.decode([topk.indices[0].item()]).strip()
        second = tok.decode([topk.indices[1].item()]).strip()
        margin = float(topk.values[0] - topk.values[1])
        sig[ent] = greedy
        margins[ent] = {"top1": greedy, "top2": second, "margin": round(margin, 6)}
    return tuple(sorted(sig.items())), margins


def run_sg_validation(model, tok, entities, entity_set_name):
    worlds = make_worlds(entities)
    entity_names = list(entities.keys())
    results = {
        "entity_set": entity_set_name,
        "entities": {k: list(v) for k, v in entities.items()},
        "n_worlds": len(worlds),
        "greedy_signatures": {},
        "sw_vs_sg_comparison": [],
        "idempotence_test": [],
        "typed_square_test": [],
        "descent_test": [],
    }

    print(f"\n{'=' * 60}")
    print(f"Entity set: {entity_set_name} ({entity_names})")
    print(f"{'=' * 60}")

    # Phase 1: Enumerate all states and compute full greedy signatures
    print("\n--- Phase 1: Enumerate states ---")
    all_states = {}

    for wname, world in worlds.items():
        for order in ["std", "rev"]:
            history = make_history(world, entities, order)
            base = history
            sig, sig_margins = get_greedy_signature(model, tok, base, entity_names)
            key = f"{wname}_{order}"
            per_entity = {}
            for ent in entity_names:
                prompt = base + f"\n{ent}:"
                dist = get_dist(model, tok, prompt)
                greedy = tok.decode([torch.argmax(dist).item()]).strip()
                correct_val = world[ent]
                wrong_val = entities[ent][1 - entities[ent].index(correct_val)]
                tk = task_kernel(dist, tok, correct_val, wrong_val)
                per_entity[ent] = {
                    "dist": dist,
                    "greedy": greedy,
                    "correct": correct_val,
                    "wrong": wrong_val,
                    "task_kernel": tk,
                }
            all_states[key] = {
                "world_key": wname,
                "world": world,
                "order": order,
                "base_prompt": base,
                "signature": sig,
                "sig_margins": sig_margins,
                "per_entity": per_entity,
            }

    # Group by greedy signature
    sig_groups = {}
    baseline_correct = 0
    baseline_total = 0
    for key, state in all_states.items():
        sig = state["signature"]
        sig_str = "|".join(f"{k}={v}" for k, v in sig)
        if sig_str not in sig_groups:
            sig_groups[sig_str] = []
        sig_groups[sig_str].append(key)
        for ent in entity_names:
            baseline_total += 1
            pe = state["per_entity"][ent]
            if pe["greedy"] == pe["correct"] or pe["correct"].lower() in pe["greedy"].lower():
                baseline_correct += 1

    baseline_rate = baseline_correct / max(baseline_total, 1)
    results["baseline_accuracy"] = round(baseline_rate, 4)
    print(f"Baseline accuracy: {baseline_correct}/{baseline_total} = {baseline_rate:.1%}")
    print(f"Greedy signatures found: {len(sig_groups)}")
    for sig_str, members in sorted(sig_groups.items()):
        print(f"  {sig_str}: {len(members)} members ({', '.join(members[:4])})")

    results["greedy_signatures"] = {k: v for k, v in sig_groups.items()}

    # Phase 2: S^W vs S^G comparison — how often do they produce different text?
    print("\n--- Phase 2: S^W vs S^G comparison ---")
    sw_sg_comparison = []
    for key, state in all_states.items():
        base = state["base_prompt"]
        world = state["world"]
        sig = state["signature"]

        sw = make_restatement_from_world(world, entities)
        sg = make_restatement_from_signature(sig, entity_names)
        match = sw == sg

        for ent in entity_names:
            sw_prompt = base + sw + f"\n{ent}:"
            sg_prompt = base + sg + f"\n{ent}:"
            sw_dist = get_dist(model, tok, sw_prompt)
            sg_dist = get_dist(model, tok, sg_prompt)
            jsd = js_dist(sw_dist, sg_dist)
            sw_greedy = tok.decode([torch.argmax(sw_dist).item()]).strip()
            sg_greedy = tok.decode([torch.argmax(sg_dist).item()]).strip()

            sw_sg_comparison.append({
                "source": key,
                "entity": ent,
                "sw_text_equals_sg": match,
                "sw_greedy": sw_greedy,
                "sg_greedy": sg_greedy,
                "greedy_match": sw_greedy == sg_greedy,
                "jsd_sw_vs_sg": round(jsd, 6),
            })

    results["sw_vs_sg_comparison"] = sw_sg_comparison
    text_matches = sum(1 for r in sw_sg_comparison if r["sw_text_equals_sg"]) // len(entity_names)
    greedy_matches = sum(1 for r in sw_sg_comparison if r["greedy_match"])
    jsds_comp = [r["jsd_sw_vs_sg"] for r in sw_sg_comparison]
    n_total = len(sw_sg_comparison)
    print(f"  Text identical (S^W == S^G): {text_matches}/{len(all_states)}")
    print(f"  Greedy identical: {greedy_matches}/{n_total}")
    if jsds_comp:
        divergent = [r for r in sw_sg_comparison if not r["sw_text_equals_sg"]]
        if divergent:
            div_jsds = [r["jsd_sw_vs_sg"] for r in divergent]
            print(f"  When S^W != S^G: JSD mean={statistics.mean(div_jsds):.4f}, max={max(div_jsds):.4f}")

    # Phase 3: Idempotence of S^G
    print("\n--- Phase 3: Idempotence of S^G ---")
    idemp_results = []
    for key, state in all_states.items():
        base = state["base_prompt"]
        sig = state["signature"]
        sg = make_restatement_from_signature(sig, entity_names)

        for ent in entity_names:
            pe = state["per_entity"][ent]
            sg1_prompt = base + sg + f"\n{ent}:"
            sg2_prompt = base + sg + sg + f"\n{ent}:"
            sg1_dist = get_dist(model, tok, sg1_prompt)
            sg2_dist = get_dist(model, tok, sg2_prompt)

            jsd = js_dist(sg1_dist, sg2_dist)
            sg1_greedy = tok.decode([torch.argmax(sg1_dist).item()]).strip()
            sg2_greedy = tok.decode([torch.argmax(sg2_dist).item()]).strip()
            sg1_tk = task_kernel(sg1_dist, tok, pe["correct"], pe["wrong"])
            sg2_tk = task_kernel(sg2_dist, tok, pe["correct"], pe["wrong"])

            idemp_results.append({
                "source": key,
                "entity": ent,
                "jsd_SG_vs_SG2": round(jsd, 6),
                "greedy_SG": sg1_greedy,
                "greedy_SG2": sg2_greedy,
                "greedy_match": sg1_greedy == sg2_greedy,
                "tk_SG": sg1_tk,
                "tk_SG2": sg2_tk,
            })

    results["idempotence_test"] = idemp_results
    jsds_i = [r["jsd_SG_vs_SG2"] for r in idemp_results]
    matches_i = sum(1 for r in idemp_results if r["greedy_match"])
    print(f"  n={len(idemp_results)}, JSD range [{min(jsds_i):.4f}, {max(jsds_i):.4f}], mean={statistics.mean(jsds_i):.4f}")
    print(f"  Greedy idempotence: {matches_i}/{len(idemp_results)} = {matches_i/len(idemp_results):.1%}")

    # Phase 4: Place preservation under S^G
    print("\n--- Phase 4: Place preservation under S^G ---")
    preservation_results = []
    for key, state in all_states.items():
        base = state["base_prompt"]
        sig = state["signature"]
        sg = make_restatement_from_signature(sig, entity_names)
        after_prompt = base + sg
        sig_after, _ = get_greedy_signature(model, tok, after_prompt, entity_names)
        preserved = sig == sig_after
        preservation_results.append({
            "source": key,
            "sig_before": "|".join(f"{k}={v}" for k, v in sig),
            "sig_after": "|".join(f"{k}={v}" for k, v in sig_after),
            "preserved": preserved,
        })
        if not preserved:
            print(f"  CHANGED: {key}: {sig} -> {sig_after}")

    results["place_preservation"] = preservation_results
    pres_rate = sum(1 for r in preservation_results if r["preserved"]) / len(preservation_results)
    print(f"  S^G place preservation: {sum(1 for r in preservation_results if r['preserved'])}/{len(preservation_results)} = {pres_rate:.1%}")

    # Phase 4b: Shuffled-renderer control (Codex confound: textual echo)
    print("\n--- Phase 4b: Shuffled-renderer control ---")
    shuffled_results = []
    for key, state in all_states.items():
        base = state["base_prompt"]
        sig = state["signature"]
        shuffled = make_shuffled_restatement(sig, entity_names)
        after_prompt = base + shuffled
        sig_after, _ = get_greedy_signature(model, tok, after_prompt, entity_names)
        preserved = sig == sig_after
        shuffled_results.append({
            "source": key,
            "sig_before": "|".join(f"{k}={v}" for k, v in sig),
            "sig_after": "|".join(f"{k}={v}" for k, v in sig_after),
            "preserved": preserved,
            "shuffled_text": shuffled[:60],
        })
        if not preserved:
            print(f"  DISRUPTED: {key}: {dict(sig)} -> {dict(sig_after)}")

    results["shuffled_control"] = shuffled_results
    shuf_pres = sum(1 for r in shuffled_results if r["preserved"]) / len(shuffled_results)
    print(f"  Shuffled place preservation: {sum(1 for r in shuffled_results if r['preserved'])}/{len(shuffled_results)} = {shuf_pres:.1%}")
    print(f"  (S^G was {pres_rate:.1%} — shuffled should be lower if content matters)")

    # Phase 4c: Anti-echo alias control (Codex design gate)
    # Decoy (wrong direct assignment) + faithful alias rendering via nonce aliases.
    # A copier predicts the decoy; semantic use predicts the original via alias resolution.
    print("\n--- Phase 4c: Anti-echo alias control ---")
    alias_clause = make_alias_clause(entity_names, entity_set_name)
    alias_results = []

    for key, state in all_states.items():
        base = state["base_prompt"]
        sig = state["signature"]
        sig_dict = dict(sig)

        decoy = make_decoy(sig, entity_names, entities)
        faithful_alias = make_alias_restatement(sig, entity_names, entity_set_name)
        shuffled_alias = make_shuffled_alias_restatement(sig, entity_names, entity_set_name)
        direct_faithful = make_restatement_from_signature(sig, entity_names)

        # Arm 1: Decoy only
        arm1_prompt = base + decoy
        arm1_sig, _ = get_greedy_signature(model, tok, arm1_prompt, entity_names)

        # Arm 2: Decoy + direct faithful R(g)
        arm2_prompt = base + decoy + direct_faithful
        arm2_sig, _ = get_greedy_signature(model, tok, arm2_prompt, entity_names)

        # Arm 3: Decoy + alias clause + faithful alias rendering (DECISIVE)
        arm3_prompt = base + decoy + alias_clause + faithful_alias
        arm3_sig, _ = get_greedy_signature(model, tok, arm3_prompt, entity_names)

        # Arm 4: Decoy + alias clause + shuffled alias rendering
        arm4_prompt = base + decoy + alias_clause + shuffled_alias
        arm4_sig, _ = get_greedy_signature(model, tok, arm4_prompt, entity_names)

        # Arm 5: Alias clause + faithful alias only (no decoy)
        arm5_prompt = base + alias_clause + faithful_alias
        arm5_sig, _ = get_greedy_signature(model, tok, arm5_prompt, entity_names)

        # Score per-coordinate recovery and decoy adoption
        decoy_sig = {}
        for n in entity_names:
            vals = entities[n]
            decoy_sig[n] = [v for v in vals if v != sig_dict[n]][0] if len([v for v in vals if v != sig_dict[n]]) > 0 else sig_dict[n]

        def score_arm(arm_sig):
            arm_dict = dict(arm_sig)
            recovery = sum(1 for n in entity_names if arm_dict.get(n) == sig_dict[n])
            decoy_adopt = sum(1 for n in entity_names if arm_dict.get(n) == decoy_sig[n])
            return recovery, decoy_adopt

        r1, d1 = score_arm(arm1_sig)
        r2, d2 = score_arm(arm2_sig)
        r3, d3 = score_arm(arm3_sig)
        r4, d4 = score_arm(arm4_sig)
        r5, d5 = score_arm(arm5_sig)

        n_ent = len(entity_names)
        alias_results.append({
            "source": key,
            "sig_original": "|".join(f"{k}={v}" for k, v in sig),
            "arm1_decoy_only": {"sig": "|".join(f"{k}={v}" for k, v in arm1_sig), "recovery": r1, "decoy_adopt": d1},
            "arm2_decoy_direct": {"sig": "|".join(f"{k}={v}" for k, v in arm2_sig), "recovery": r2, "decoy_adopt": d2},
            "arm3_decoy_alias": {"sig": "|".join(f"{k}={v}" for k, v in arm3_sig), "recovery": r3, "decoy_adopt": d3},
            "arm4_decoy_shuffled_alias": {"sig": "|".join(f"{k}={v}" for k, v in arm4_sig), "recovery": r4, "decoy_adopt": d4},
            "arm5_alias_only": {"sig": "|".join(f"{k}={v}" for k, v in arm5_sig), "recovery": r5, "decoy_adopt": d5},
            "n_entities": n_ent,
        })

    results["anti_echo_alias"] = alias_results
    n_total = sum(r["n_entities"] for r in alias_results)
    for arm_key, arm_name in [
        ("arm1_decoy_only", "Decoy only"),
        ("arm2_decoy_direct", "Decoy + direct R(g)"),
        ("arm3_decoy_alias", "Decoy + faithful alias"),
        ("arm4_decoy_shuffled_alias", "Decoy + shuffled alias"),
        ("arm5_alias_only", "Alias only (no decoy)"),
    ]:
        rec = sum(r[arm_key]["recovery"] for r in alias_results)
        dec = sum(r[arm_key]["decoy_adopt"] for r in alias_results)
        print(f"  {arm_name}: recovery={rec}/{n_total} ({rec/n_total:.1%}), decoy_adopt={dec}/{n_total} ({dec/n_total:.1%})")

    # Phase 5a: Correction descent — does C_{e<-v} produce same g' for all fiber members?
    # (Codex: prerequisite for a genuine quotient-level typed square)
    print("\n--- Phase 5a: Correction descent ---")
    correction_descent = []
    for sig_str, members in sig_groups.items():
        if len(members) < 2:
            continue
        for ent in entity_names:
            # Use a fixed correction value: flip to the "wrong" value of the first member
            first_pe = all_states[members[0]]["per_entity"][ent]
            correction_val = first_pe["wrong"]
            correction = f" Actually, {ent}: {correction_val}."

            post_correction_sigs = {}
            for mk in members:
                mstate = all_states[mk]
                corrected_base = mstate["base_prompt"] + correction
                sig_after, _ = get_greedy_signature(model, tok, corrected_base, entity_names)
                sig_str_after = "|".join(f"{k}={v}" for k, v in sig_after)
                post_correction_sigs[mk] = sig_str_after

            unique_targets = set(post_correction_sigs.values())
            descends = len(unique_targets) == 1
            correction_descent.append({
                "fiber": sig_str,
                "entity": ent,
                "correction_val": correction_val,
                "n_members": len(members),
                "post_correction_sigs": post_correction_sigs,
                "descends": descends,
            })
            if not descends:
                print(f"  FAIL: {sig_str}/{ent}->'{correction_val}': {post_correction_sigs}")

    results["correction_descent"] = correction_descent
    if correction_descent:
        cd_pass = sum(1 for r in correction_descent if r["descends"])
        print(f"  Correction descent: {cd_pass}/{len(correction_descent)} = {cd_pass/len(correction_descent):.1%}")
    else:
        print(f"  No multi-member fibers to test")

    # Phase 6: TYPED SQUARE with S^G
    # Path CS: base + correction + S^G_{g'} + query
    #   where g' = greedy signature of (base + correction)
    # Path SC: base + S^G_g + correction + query
    print("\n--- Phase 6: Typed correction/S^G square ---")
    square_results = []

    for key, state in all_states.items():
        base = state["base_prompt"]
        sig_g = state["signature"]

        for ent in entity_names:
            pe = state["per_entity"][ent]
            correct_val = pe["correct"]
            wrong_val = pe["wrong"]

            correction = f" Actually, {ent}: {wrong_val}."

            # Get the greedy signature AFTER correction (Codex: g'_x = γ(C_{e←v}x))
            corrected_base = base + correction
            sig_g_prime, _ = get_greedy_signature(model, tok, corrected_base, entity_names)

            sg_g = make_restatement_from_signature(sig_g, entity_names)
            sg_g_prime = make_restatement_from_signature(sig_g_prime, entity_names)

            # Path 1: C then S^G_{g'} — correct, then restate from post-correction signature
            cs_prompt = corrected_base + sg_g_prime + f"\n{ent}:"
            cs_dist = get_dist(model, tok, cs_prompt)

            # Path 2: S^G_g then C — restate from pre-correction signature, then correct
            sc_prompt = base + sg_g + correction + f"\n{ent}:"
            sc_dist = get_dist(model, tok, sc_prompt)

            jsd = js_dist(cs_dist, sc_dist)
            cs_greedy = tok.decode([torch.argmax(cs_dist).item()]).strip()
            sc_greedy = tok.decode([torch.argmax(sc_dist).item()]).strip()
            cs_tk = task_kernel(cs_dist, tok, wrong_val, correct_val)
            sc_tk = task_kernel(sc_dist, tok, wrong_val, correct_val)

            # Also compare with S^W square for the same pair
            corrected_world = make_corrected_world(state["world"], ent, wrong_val)
            sw_old = make_restatement_from_world(state["world"], entities)
            sw_new = make_restatement_from_world(corrected_world, entities)
            cs_sw_prompt = base + correction + sw_new + f"\n{ent}:"
            sc_sw_prompt = base + sw_old + correction + f"\n{ent}:"
            cs_sw_dist = get_dist(model, tok, cs_sw_prompt)
            sc_sw_dist = get_dist(model, tok, sc_sw_prompt)
            jsd_sw = js_dist(cs_sw_dist, sc_sw_dist)

            sq = {
                "source": key,
                "entity": ent,
                "corrected_value": wrong_val,
                "original_value": correct_val,
                "sig_g": "|".join(f"{k}={v}" for k, v in sig_g),
                "sig_g_prime": "|".join(f"{k}={v}" for k, v in sig_g_prime),
                "sg_text": sg_g[:60],
                "sg_prime_text": sg_g_prime[:60],
                "jsd_CS_vs_SC_SG": round(jsd, 6),
                "jsd_CS_vs_SC_SW": round(jsd_sw, 6),
                "greedy_CS": cs_greedy,
                "greedy_SC": sc_greedy,
                "greedy_match": cs_greedy == sc_greedy,
                "tk_CS": cs_tk,
                "tk_SC": sc_tk,
                "tk_correct_diff": round(abs(cs_tk["correct"] - sc_tk["correct"]), 6),
            }
            square_results.append(sq)
            delta_label = "SG<SW" if jsd < jsd_sw else ("SG>SW" if jsd > jsd_sw else "SG==SW")
            print(f"  {key}/{ent}: JSD_SG={jsd:.4f}, JSD_SW={jsd_sw:.4f} [{delta_label}], greedy={cs_greedy}/{sc_greedy}")

    results["typed_square_test"] = square_results
    sq_jsds_sg = [r["jsd_CS_vs_SC_SG"] for r in square_results]
    sq_jsds_sw = [r["jsd_CS_vs_SC_SW"] for r in square_results]
    sq_matches = sum(1 for r in square_results if r["greedy_match"])
    sq_tk_diffs = [r["tk_correct_diff"] for r in square_results]
    print(f"\n  n={len(square_results)}")
    print(f"  S^G JSD: [{min(sq_jsds_sg):.4f}, {max(sq_jsds_sg):.4f}], mean={statistics.mean(sq_jsds_sg):.4f}")
    print(f"  S^W JSD: [{min(sq_jsds_sw):.4f}, {max(sq_jsds_sw):.4f}], mean={statistics.mean(sq_jsds_sw):.4f}")
    print(f"  Greedy commutativity (S^G): {sq_matches}/{len(square_results)} = {sq_matches/len(square_results):.1%}")
    print(f"  Task kernel diff: mean={statistics.mean(sq_tk_diffs):.4f}")
    sg_less = sum(1 for s, w in zip(sq_jsds_sg, sq_jsds_sw) if s < w)
    print(f"  S^G < S^W (less non-naturality): {sg_less}/{len(square_results)}")

    # Phase 5: Descent test with S^G (Codex: test FULL signature, not entity-by-entity)
    print("\n--- Phase 5: Descent test (S^G vs S^W) ---")
    descent_results = []
    for sig_str, members in sig_groups.items():
        if len(members) < 2:
            continue
        sig_tuple = all_states[members[0]]["signature"]
        sg = make_restatement_from_signature(sig_tuple, entity_names)

        for ent in entity_names:
            targets_empty = set()
            targets_sg = set()
            targets_sw = set()
            for mk in members:
                mstate = all_states[mk]
                base = mstate["base_prompt"]
                world = mstate["world"]
                # Empty
                prompt = base + f"\n{ent}:"
                dist = get_dist(model, tok, prompt)
                g = tok.decode([torch.argmax(dist).item()]).strip()
                targets_empty.add(g)
                # S^G (same for all members — that's the point)
                prompt = base + sg + f"\n{ent}:"
                dist = get_dist(model, tok, prompt)
                g = tok.decode([torch.argmax(dist).item()]).strip()
                targets_sg.add(g)
                # S^W (for comparison — uses hidden world, differs across members)
                sw = make_restatement_from_world(world, entities)
                prompt = base + sw + f"\n{ent}:"
                dist = get_dist(model, tok, prompt)
                g = tok.decode([torch.argmax(dist).item()]).strip()
                targets_sw.add(g)

            descent_results.append({
                "signature": sig_str,
                "entity": ent,
                "n_members": len(members),
                "member_worlds": [all_states[mk]["world_key"] for mk in members],
                "empty_targets": list(targets_empty),
                "empty_descends": len(targets_empty) == 1,
                "sg_targets": list(targets_sg),
                "sg_descends": len(targets_sg) == 1,
                "sw_targets": list(targets_sw),
                "sw_descends": len(targets_sw) == 1,
            })

    results["descent_test"] = descent_results
    empty_desc = sum(1 for r in descent_results if r["empty_descends"])
    sg_desc = sum(1 for r in descent_results if r["sg_descends"])
    sw_desc = sum(1 for r in descent_results if r["sw_descends"])
    total_desc = len(descent_results)
    print(f"  Fiber groups tested: {total_desc}")
    print(f"  Empty descent:       {empty_desc}/{total_desc} = {empty_desc/max(total_desc,1):.1%}")
    print(f"  S^G descent:         {sg_desc}/{total_desc} = {sg_desc/max(total_desc,1):.1%}")
    print(f"  S^W descent:         {sw_desc}/{total_desc} = {sw_desc/max(total_desc,1):.1%}")
    failures_sg = [r for r in descent_results if not r["sg_descends"]]
    failures_sw = [r for r in descent_results if not r["sw_descends"]]
    if failures_sg:
        print(f"  S^G failures:")
        for f in failures_sg:
            print(f"    {f['signature']}/{f['entity']}: worlds={f['member_worlds']}, targets={f['sg_targets']}")
    if failures_sw:
        print(f"  S^W failures:")
        for f in failures_sw:
            print(f"    {f['signature']}/{f['entity']}: worlds={f['member_worlds']}, targets={f['sw_targets']}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--phase4d-only", action="store_true",
        help="Run the terminal Phase 4d factorial once and write phase4d_results.json",
    )
    mode.add_argument(
        "--phase4d-preflight", action="store_true",
        help="Run tokenizer-only Phase 4d integrity checks; no model inference",
    )
    args = parser.parse_args()

    if args.phase4d_preflight:
        tok = load_tokenizer()
        print(json.dumps(phase4d_preflight(tok), indent=2))
        return

    if args.phase4d_only:
        out_path = os.path.join(RESULTS_DIR, "phase4d_results.json")
        if os.path.exists(out_path):
            raise FileExistsError(
                f"Terminal Phase 4d artifact already exists: {out_path}. "
                "The runner refuses to overwrite or silently rerun it."
            )
        print("Loading model for terminal Phase 4d...")
        model, tok = load_model()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        result = run_phase4d(model, tok)
        runner_bytes = open(__file__, "rb").read()
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        git_status = subprocess.run(
            ["git", "status", "--short"], capture_output=True, text=True
        ).stdout
        result["generated_at"] = datetime.now().isoformat()
        result["provenance"] = {
            "model_id": MODEL_ID,
            "model_revision_requested": MODEL_REVISION,
            "model_revision_loaded": getattr(model.config, "_commit_hash", None),
            "tokenizer_revision_loaded": tok.init_kwargs.get("_commit_hash"),
            "runner_sha256": hashlib.sha256(runner_bytes).hexdigest(),
            "git_commit": git_commit,
            "git_status_porcelain": git_status,
            "git_status_sha256": sha256_text(git_status),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": str(next(model.parameters()).device),
            "tokenizer_vocab_size": tok.vocab_size,
            "argv": sys.argv,
        }
        payload = json.dumps(result, sort_keys=True, separators=(",", ":"), default=str)
        result["provenance"]["payload_sha256_before_hash_field"] = sha256_text(payload)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, default=str)
        print(json.dumps(result["adjudication"], indent=2))
        print(f"\nTerminal Phase 4d results saved to {out_path}")
        return

    print("Loading model...")
    model, tok = load_model()
    print(f"Model: {MODEL_ID}")
    print(f"Tokenizer vocab: {tok.vocab_size}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    _rh = hashlib.md5(open(__file__, "rb").read()).hexdigest()[:12]
    _gc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    all_results = {
        "model": MODEL_ID,
        "timestamp": datetime.now().isoformat(),
        "provenance": {
            "runner_hash": _rh,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "vocab_size": tok.vocab_size,
            "device": str(next(model.parameters()).device),
            "git_commit": _gc,
        },
        "sets": {},
    }

    for set_name, entities in [("registered", REGISTERED_ENTITIES), ("heldout", HELDOUT_ENTITIES)]:
        print(f"\n{'#' * 60}")
        print(f"# SET {set_name}")
        print(f"{'#' * 60}")
        res = run_sg_validation(model, tok, entities, set_name)
        all_results["sets"][set_name] = res

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for sn in ["registered", "heldout"]:
        r = all_results["sets"][sn]
        print(f"\n  SET {sn}")
        print(f"    Baseline: {r['baseline_accuracy']:.1%}")

        comp = r["sw_vs_sg_comparison"]
        text_same = sum(1 for c in comp if c["sw_text_equals_sg"])
        greedy_same = sum(1 for c in comp if c["greedy_match"])
        print(f"    S^W vs S^G text identical: {text_same}/{len(comp)}")
        print(f"    S^W vs S^G greedy identical: {greedy_same}/{len(comp)}")

        idemp = r["idempotence_test"]
        ij = [x["jsd_SG_vs_SG2"] for x in idemp]
        im = sum(1 for x in idemp if x["greedy_match"])
        print(f"    S^G idempotence: n={len(idemp)}, JSD [{min(ij):.4f},{max(ij):.4f}], greedy {im}/{len(idemp)}")

        sq = r["typed_square_test"]
        sj_sg = [x["jsd_CS_vs_SC_SG"] for x in sq]
        sj_sw = [x["jsd_CS_vs_SC_SW"] for x in sq]
        sm = sum(1 for x in sq if x["greedy_match"])
        stk = [x["tk_correct_diff"] for x in sq]
        print(f"    S^G square: JSD [{min(sj_sg):.4f},{max(sj_sg):.4f}], mean={statistics.mean(sj_sg):.4f}")
        print(f"    S^W square: JSD [{min(sj_sw):.4f},{max(sj_sw):.4f}], mean={statistics.mean(sj_sw):.4f}")
        print(f"    Greedy commutativity (S^G): {sm}/{len(sq)} = {sm/len(sq):.1%}")

        desc = r["descent_test"]
        if desc:
            sg_d = sum(1 for d in desc if d["sg_descends"])
            sw_d = sum(1 for d in desc if d["sw_descends"])
            print(f"    Descent S^G: {sg_d}/{len(desc)}, S^W: {sw_d}/{len(desc)}")

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
