"""Confound-killing curvature control: isochronous position-matched test.
Codex design gate (session 01a06cea).
"""
import copy
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np


def build_prefix(cfg, tmpl, prof, depth):
    schema = cfg["template_schema"][f"depth{depth}"]
    var = cfg["task"]["variable"]
    s = schema
    s = s.replace("{var}", var)
    s = s.replace("{outer_value}", str(cfg["task"]["outer_value"]))
    s = s.replace("{fn0}", tmpl["fn0"])
    s = s.replace("{fn1}", tmpl["fn1"])
    s = s.replace("{fn2}", tmpl["fn2"])
    s = s.replace("{fn3}", tmpl["fn3"])
    s = s.replace("{inner1}", prof["inner1"])
    s = s.replace("{inner2}", prof["inner2"])
    s = s.replace("{inner3}", prof["inner3"])
    s = s.replace("{inner4}", prof["inner4"])
    return s


def build_padding(pad_cfg, delta):
    if pad_cfg["builder"] == "repeat":
        return pad_cfg["unit"] * delta
    q, r = divmod(delta, pad_cfg["primary_unit_tokens"])
    return pad_cfg["primary_unit"] * q + pad_cfg["remainder_unit"] * r


def build_query(cfg, tmpl):
    var = cfg["task"]["variable"]
    return cfg["task"]["query_template"].replace("{var}", var).replace("{fn0}", tmpl["fn0"])


def get_all_surfaces(cfg):
    var = cfg["task"]["variable"]
    surfaces = []
    for pair in cfg["semantic_surface_pairs"]:
        surfaces.append({
            "id": pair["assert_id"], "text": pair["assert_text"].replace("{var}", var),
            "role": "assert", "pair_id": pair["id"], "split": pair["split"]
        })
        surfaces.append({
            "id": pair["mislead_id"], "text": pair["mislead_text"].replace("{var}", var),
            "role": "mislead", "pair_id": pair["id"], "split": pair["split"]
        })
    for n in cfg["neutral_surfaces"]:
        surfaces.append({
            "id": n["id"], "text": n["text"].replace("{var}", var),
            "role": "neutral", "pair_id": None, "split": n["split"]
        })
    none = cfg["positioned_none"]
    surfaces.append({
        "id": none["id"], "text": none["text"],
        "role": "none", "pair_id": None, "split": "train"
    })
    return surfaces


def preflight(cfg):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        cfg["model"]["model_id"], revision=cfg["model"]["revision"],
        local_files_only=True)
    surfaces = get_all_surfaces(cfg)
    pad_cfgs = {p["id"]: p for p in cfg["padding"]["families"]}

    for tmpl in cfg["template_instances"]:
        query = build_query(cfg, tmpl)
        q_toks = tok.encode(query, add_special_tokens=False)
        for surf in surfaces:
            s_toks = tok.encode(surf["text"], add_special_tokens=False)
            assert len(s_toks) == 5, f"{surf['id']} on {tmpl['id']}: {len(s_toks)} tokens"
            sq_toks = tok.encode(surf["text"] + query, add_special_tokens=False)
            assert sq_toks == s_toks + q_toks, f"Boundary merge: {surf['id']} on {tmpl['id']}"

        for prof in cfg["inner_literal_profiles"]:
            d3 = build_prefix(cfg, tmpl, prof, 3)
            d4 = build_prefix(cfg, tmpl, prof, 4)
            n3 = len(tok.encode(d3, add_special_tokens=False))
            n4 = len(tok.encode(d4, add_special_tokens=False))
            delta = n4 - n3
            assert delta > 0, f"{tmpl['id']}::{prof['id']} delta={delta}"
            exp = cfg["padding"]["expected_verified_counts"].get(tmpl["id"])
            if exp:
                assert n3 == exp["d3"] and n4 == exp["d4"] and delta == exp["delta"], \
                    f"{tmpl['id']} tokens: got {n3}/{n4}/{delta}, expected {exp}"
            for pc in cfg["padding"]["families"]:
                pad = build_padding(pc, delta)
                pt = len(tok.encode(pad, add_special_tokens=False))
                assert pt == delta, f"Pad {pc['id']} for {tmpl['id']}: {pt} != {delta}"

    n_ctx = len(cfg["template_instances"]) * len(cfg["inner_literal_profiles"])
    n_arms = len(surfaces)
    n_place = len(cfg["placements"])
    n_replay = cfg["replay"]["expected_extra_calls"]
    expected = n_ctx * (2 + n_place * n_arms) + n_replay
    assert expected <= cfg["execution"]["max_model_calls"], \
        f"Budget {expected} > {cfg['execution']['max_model_calls']}"
    print(f"G0 PASS: {expected} expected calls, all token counts verified.", flush=True)
    return {"expected_calls": expected, "pass": True}


def collect(adapter, cfg):
    surfaces = get_all_surfaces(cfg)
    pad_cfgs = {p["id"]: p for p in cfg["padding"]["families"]}
    replay_set = set(cfg["replay"]["context_ids"])
    replay_placements = set(cfg["replay"]["placements"])
    replay_surfaces = set(cfg["replay"]["surface_ids"])
    result_dir = Path(cfg["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    obs = []
    t0 = time.time()
    call_idx = 0

    for tmpl in cfg["template_instances"]:
        for prof in cfg["inner_literal_profiles"]:
            ctx_id = f"{tmpl['id']}::{prof['id']}"
            query = build_query(cfg, tmpl)
            is_replay_ctx = ctx_id in replay_set

            d3_prefix = build_prefix(cfg, tmpl, prof, 3)
            d4_prefix = build_prefix(cfg, tmpl, prof, 4)
            d3_n = len(adapter.tok.encode(d3_prefix, add_special_tokens=False))
            d4_n = len(adapter.tok.encode(d4_prefix, add_special_tokens=False))
            delta = d4_n - d3_n

            d3_state = adapter.get_state_after_prefix(d3_prefix)
            call_idx += 1

            for placement in cfg["placements"]:
                if placement["depth"] != 3:
                    continue
                pad_id = placement["padding"]
                if pad_id:
                    pad_str = build_padding(pad_cfgs[pad_id], delta)
                else:
                    pad_str = ""

                for surf in surfaces:
                    full_suffix = pad_str + surf["text"] + query
                    dist = adapter.get_dist_from_state(d3_state, full_suffix, deepcopy=True)
                    call_idx += 1
                    obs.append(_make_obs(
                        tmpl, prof, 3, placement, pad_id, surf, dist, call_idx, False,
                        d3_n, delta if pad_id else 0))

                    if (is_replay_ctx and placement["id"] in replay_placements
                            and surf["id"] in replay_surfaces):
                        dist2 = adapter.get_dist_from_state(d3_state, full_suffix, deepcopy=True)
                        call_idx += 1
                        obs.append(_make_obs(
                            tmpl, prof, 3, placement, pad_id, surf, dist2, call_idx, True,
                            d3_n, delta if pad_id else 0))

            del d3_state
            gc.collect()

            d4_state = adapter.get_state_after_prefix(d4_prefix)
            call_idx += 1

            placement = [p for p in cfg["placements"] if p["id"] == "D4_NATURAL"][0]
            for surf in surfaces:
                full_suffix = surf["text"] + query
                dist = adapter.get_dist_from_state(d4_state, full_suffix, deepcopy=True)
                call_idx += 1
                obs.append(_make_obs(
                    tmpl, prof, 4, placement, None, surf, dist, call_idx, False, d4_n, 0))

                if (is_replay_ctx and "D4_NATURAL" in replay_placements
                        and surf["id"] in replay_surfaces):
                    dist2 = adapter.get_dist_from_state(d4_state, full_suffix, deepcopy=True)
                    call_idx += 1
                    obs.append(_make_obs(
                        tmpl, prof, 4, placement, None, surf, dist2, call_idx, True, d4_n, 0))

            del d4_state
            gc.collect()

            elapsed = time.time() - t0
            if elapsed > cfg["execution"]["max_wall_seconds"]:
                print(f"WALL LIMIT at {elapsed:.0f}s, {call_idx} calls", flush=True)
                break
            if call_idx > cfg["execution"]["max_model_calls"]:
                print(f"CALL LIMIT at {call_idx} calls", flush=True)
                break

            _checkpoint(obs, cfg, result_dir, elapsed)
            print(f"  {ctx_id}: {call_idx} calls, {elapsed:.1f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nCollection done: {call_idx} calls, {elapsed:.1f}s\n", flush=True)
    return obs


def _make_obs(tmpl, prof, depth, placement, pad_id, surf, dist, call_idx, is_replay,
              prefix_tokens, padding_tokens):
    top = int(np.argmax(dist[:10]))
    return {
        "template_id": tmpl["id"], "template_family": tmpl["family"],
        "template_split": tmpl["split"], "literal_profile": prof["id"],
        "depth": depth, "placement": placement["id"], "padding_family": pad_id,
        "prefix_tokens": prefix_tokens, "padding_tokens": padding_tokens,
        "surface_id": surf["id"], "surface_role": surf["role"],
        "surface_pair_id": surf["pair_id"], "surface_split": surf["split"],
        "dist": dist.tolist(), "top_digit": top, "call_index": call_idx,
        "is_replay": is_replay,
    }


def _checkpoint(obs, cfg, result_dir, elapsed):
    cp = {"config_name": cfg["experiment_name"], "observations": len(obs), "elapsed_s": elapsed}
    with open(result_dir / "result.partial.json", "w") as f:
        json.dump(cp, f)


def compute_z(dist, correct_digit, shadow_digit):
    p_c = max(dist[correct_digit], 1e-30)
    p_s = max(dist[shadow_digit], 1e-30)
    return np.log(p_c) - np.log(p_s)


def analyze(obs, cfg):
    correct = cfg["task"]["outer_value"]
    shadows = cfg["task"]["analysis_shadow_digits"]

    for o in obs:
        o["z"] = {}
        for sd in shadows:
            o["z"][str(sd)] = compute_z(o["dist"], correct, sd)

    results = {"observations": obs, "gates": {}}

    g1 = gate_competence(obs, cfg)
    results["gates"]["G1"] = g1
    print_gate("G1_competence", g1)

    g2 = gate_semantic_control(obs, cfg, shadows)
    results["gates"]["G2"] = g2
    print_gate("G2_semantic_positive_control", g2)

    g3 = gate_replay_noise(obs, cfg, shadows)
    results["gates"]["G3"] = g3
    print_gate("G3_replay_noise", g3)

    effects = compute_effects(obs, cfg, shadows)
    results["effects"] = effects

    g4, null_models = gate_position_null(effects, cfg, shadows)
    results["gates"]["G4"] = g4
    results["null_models"] = null_models
    print_gate("G4_position_gain_validity", g4)

    kappa_iso = compute_isochronous_curvature(effects, cfg, shadows)
    results["kappa_iso"] = kappa_iso
    g5 = gate_isochronous(kappa_iso, cfg)
    results["gates"]["G5"] = g5
    print_gate("G5_isochronous_curvature", g5)

    kappa_gain = compute_gain_residual(effects, null_models, cfg, shadows)
    results["kappa_gain"] = kappa_gain
    g6 = gate_gain_residual(kappa_gain, cfg)
    results["gates"]["G6"] = g6
    print_gate("G6_gain_null_residual", g6)

    return results


def gate_competence(obs, cfg):
    g = cfg["gates"]["G1_competence"]
    correct = cfg["task"]["outer_value"]
    none_obs = [o for o in obs if o["surface_id"] == "NONE5" and not o["is_replay"]]
    n_correct = sum(1 for o in none_obs if o["top_digit"] == correct)
    acc = n_correct / len(none_obs) if none_obs else 0

    by_placement = {}
    for o in none_obs:
        by_placement.setdefault(o["placement"], []).append(o["top_digit"] == correct)
    placement_accs = {k: np.mean(v) for k, v in by_placement.items()}
    each_pass = all(a >= g["candidate_digit_accuracy_each_placement_min"] for a in placement_accs.values())

    ok = acc >= g["candidate_digit_accuracy_overall_min"] and each_pass
    return {"pass": ok, "accuracy": acc, "placement_accuracies": placement_accs,
            "n": len(none_obs)}


def gate_semantic_control(obs, cfg, shadows):
    g = cfg["gates"]["G2_semantic_positive_control"]
    correct = cfg["task"]["outer_value"]
    main_obs = [o for o in obs if not o["is_replay"]]

    gaps = []
    for o in main_obs:
        if o["surface_role"] not in ("assert", "mislead"):
            continue
        for sd in shadows:
            z = o["z"][str(sd)]
            none_key = (o["template_id"], o["literal_profile"], o["placement"])
            z_none = None
            for o2 in main_obs:
                if (o2["surface_id"] == "NONE5" and o2["template_id"] == none_key[0]
                        and o2["literal_profile"] == none_key[1]
                        and o2["placement"] == none_key[2] and not o2["is_replay"]):
                    z_none = o2["z"][str(sd)]
                    break
            if z_none is None:
                continue
            e = z - z_none
            gaps.append({"role": o["surface_role"], "effect": e,
                         "template_split": o["template_split"],
                         "surface_split": o["surface_split"]})

    a_effects = [g["effect"] for g in gaps if g["role"] == "assert"]
    m_effects = [g["effect"] for g in gaps if g["role"] == "mislead"]
    mean_gap = np.mean(a_effects) - np.mean(m_effects) if a_effects and m_effects else 0

    ok = mean_gap >= g["mean_nat_min"]
    return {"pass": ok, "mean_assert_minus_mislead": float(mean_gap),
            "n_assert": len(a_effects), "n_mislead": len(m_effects)}


def gate_replay_noise(obs, cfg, shadows):
    g = cfg["gates"]["G3_replay_noise"]
    replays = [o for o in obs if o["is_replay"]]
    originals = {(o["template_id"], o["literal_profile"], o["placement"],
                  o["surface_id"]): o for o in obs if not o["is_replay"]}

    tv_deltas = []
    z_deltas = []
    for r in replays:
        key = (r["template_id"], r["literal_profile"], r["placement"], r["surface_id"])
        orig = originals.get(key)
        if orig is None:
            continue
        d1 = np.array(orig["dist"])
        d2 = np.array(r["dist"])
        tv = 0.5 * np.sum(np.abs(d1 - d2))
        tv_deltas.append(tv)
        for sd in shadows:
            z1 = orig["z"][str(sd)]
            z2 = r["z"][str(sd)]
            z_deltas.append(abs(z1 - z2))

    if not tv_deltas:
        return {"pass": False, "reason": "no replays found"}

    q99_tv = float(np.percentile(tv_deltas, 99))
    q99_z = float(np.percentile(z_deltas, 99)) if z_deltas else 0
    noise_floor = max(q99_z, g["noise_floor_nat"])

    ok = q99_tv <= g["q99_tv_max"] and q99_z <= g["q99_abs_z_delta_max"]
    return {"pass": ok, "q99_tv": q99_tv, "q99_abs_z_delta": q99_z,
            "noise_floor_nat": noise_floor, "n_replay_pairs": len(tv_deltas)}


def compute_effects(obs, cfg, shadows):
    main_obs = [o for o in obs if not o["is_replay"]]
    none_lookup = {}
    for o in main_obs:
        if o["surface_id"] == "NONE5":
            for sd in shadows:
                key = (o["template_id"], o["literal_profile"], o["placement"], sd)
                none_lookup[key] = o["z"][str(sd)]

    effects = []
    for o in main_obs:
        if o["surface_id"] == "NONE5":
            continue
        for sd in shadows:
            none_key = (o["template_id"], o["literal_profile"], o["placement"], sd)
            z_none = none_lookup.get(none_key)
            if z_none is None:
                continue
            e = o["z"][str(sd)] - z_none
            effects.append({
                "template_id": o["template_id"], "template_family": o["template_family"],
                "template_split": o["template_split"], "literal_profile": o["literal_profile"],
                "depth": o["depth"], "placement": o["placement"],
                "padding_family": o["padding_family"], "shadow_digit": sd,
                "surface_id": o["surface_id"], "surface_role": o["surface_role"],
                "surface_pair_id": o["surface_pair_id"], "surface_split": o["surface_split"],
                "z": o["z"][str(sd)], "z_none": z_none, "effect": e,
            })
    return effects


def gate_position_null(effects, cfg, shadows):
    g = cfg["gates"]["G4_position_gain_validity"]
    null_models = {}
    all_ok = True

    for pad_fam in ["PAD_BARE", "PAD_WORD"]:
        for sd in shadows:
            train_raw = [e for e in effects
                         if e["placement"] == "D3_RAW" and e["template_split"] == "train"
                         and e["surface_split"] == "train" and e["shadow_digit"] == sd
                         and e["surface_role"] != "neutral"]
            train_raw.extend([e for e in effects
                              if e["placement"] == "D3_RAW" and e["template_split"] == "train"
                              and e["shadow_digit"] == sd
                              and e["surface_id"] in ("N_NOTE", "N_READ", "N_CHECK")])
            train_matched = [e for e in effects
                             if e["placement"] == f"D3_MATCH_{pad_fam.split('_')[1]}"
                             and e["template_split"] == "train"
                             and e["surface_split"] == "train" and e["shadow_digit"] == sd
                             and e["surface_role"] != "neutral"]
            train_matched.extend([e for e in effects
                                  if e["placement"] == f"D3_MATCH_{pad_fam.split('_')[1]}"
                                  and e["template_split"] == "train"
                                  and e["shadow_digit"] == sd
                                  and e["surface_id"] in ("N_NOTE", "N_READ", "N_CHECK")])

            placement_id = f"D3_MATCH_{pad_fam.split('_')[1]}"

            raw_by_key = {}
            for e in train_raw:
                key = (e["template_id"], e["literal_profile"], e["surface_id"])
                raw_by_key[key] = e["effect"]
            matched_by_key = {}
            for e in train_matched:
                key = (e["template_id"], e["literal_profile"], e["surface_id"])
                matched_by_key[key] = e["effect"]

            common_keys = sorted(set(raw_by_key) & set(matched_by_key))
            if len(common_keys) < 3:
                null_models[(pad_fam, sd)] = {"valid": False, "reason": "too few points"}
                all_ok = False
                continue

            X_raw = np.array([raw_by_key[k] for k in common_keys])
            Y_matched = np.array([matched_by_key[k] for k in common_keys])

            if np.std(X_raw) < g["min_train_effect_sd_nat"]:
                null_models[(pad_fam, sd)] = {"valid": False, "reason": "effect range too narrow"}
                all_ok = False
                continue

            A = np.column_stack([np.ones(len(X_raw)), X_raw])
            coeffs = np.linalg.lstsq(A, Y_matched, rcond=None)[0]
            alpha, gain = float(coeffs[0]), float(coeffs[1])

            holdout_raw = [e for e in effects
                           if e["placement"] == "D3_RAW" and e["template_split"] == "holdout"
                           and e["shadow_digit"] == sd and e["surface_id"] == "N_COMMENT"]
            holdout_matched = [e for e in effects
                               if e["placement"] == placement_id
                               and e["template_split"] == "holdout"
                               and e["shadow_digit"] == sd and e["surface_id"] == "N_COMMENT"]

            hraw_by_key = {(e["template_id"], e["literal_profile"]): e["effect"]
                           for e in holdout_raw}
            hmatched_by_key = {(e["template_id"], e["literal_profile"]): e["effect"]
                               for e in holdout_matched}
            hkeys = sorted(set(hraw_by_key) & set(hmatched_by_key))

            if hkeys:
                preds = [alpha + gain * hraw_by_key[k] for k in hkeys]
                actuals = [hmatched_by_key[k] for k in hkeys]
                errors = [a - p for a, p in zip(actuals, preds)]
                mae = float(np.mean(np.abs(errors)))
                mean_err = float(np.mean(errors))
                p90 = float(np.percentile(np.abs(errors), 90))
                val_ok = (mae <= g["max_holdout_neutral_mae_nat"]
                          and abs(mean_err) <= g["max_abs_holdout_neutral_mean_error_nat"]
                          and p90 <= g["max_holdout_neutral_p90_abs_error_nat"])
            else:
                mae, mean_err, p90, val_ok = 0, 0, 0, True

            null_models[(pad_fam, sd)] = {
                "valid": val_ok, "alpha": alpha, "gain": gain,
                "n_train": len(common_keys), "train_r2": _r2(X_raw, Y_matched, alpha, gain),
                "holdout_mae": mae, "holdout_mean_error": mean_err, "holdout_p90": p90,
            }
            if not val_ok:
                all_ok = False

    return {"pass": all_ok, "models": {f"{k[0]}_sd{k[1]}": v for k, v in null_models.items()}}, null_models


def _r2(x, y, alpha, gain):
    pred = alpha + gain * x
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0


def compute_isochronous_curvature(effects, cfg, shadows):
    blocks = []
    holdout_templates = [t for t in cfg["template_instances"] if t["split"] == "holdout"]
    holdout_pairs = [p for p in cfg["semantic_surface_pairs"] if p["split"] == "holdout"]

    for tmpl in cfg["template_instances"]:
        for pair in cfg["semantic_surface_pairs"]:
            for pad_fam in ["PAD_BARE", "PAD_WORD"]:
                placement_id = f"D3_MATCH_{pad_fam.split('_')[1]}"
                for prof in cfg["inner_literal_profiles"]:
                    for sd in shadows:
                        e_a_d3 = _get_effect(effects, tmpl["id"], prof["id"],
                                             placement_id, pair["assert_id"], sd)
                        e_m_d3 = _get_effect(effects, tmpl["id"], prof["id"],
                                             placement_id, pair["mislead_id"], sd)
                        e_a_d4 = _get_effect(effects, tmpl["id"], prof["id"],
                                             "D4_NATURAL", pair["assert_id"], sd)
                        e_m_d4 = _get_effect(effects, tmpl["id"], prof["id"],
                                             "D4_NATURAL", pair["mislead_id"], sd)
                        if None in (e_a_d3, e_m_d3, e_a_d4, e_m_d4):
                            continue
                        rho_a = e_a_d4 - e_a_d3
                        rho_m = e_m_d4 - e_m_d3
                        kappa = rho_m - rho_a
                        blocks.append({
                            "template_id": tmpl["id"], "template_split": tmpl["split"],
                            "pair_id": pair["id"], "pair_split": pair["split"],
                            "padding_family": pad_fam, "literal_profile": prof["id"],
                            "shadow_digit": sd, "rho_a": rho_a, "rho_m": rho_m, "kappa": kappa,
                        })

    return blocks


def compute_gain_residual(effects, null_models, cfg, shadows):
    blocks = []
    for tmpl in cfg["template_instances"]:
        for pair in cfg["semantic_surface_pairs"]:
            for pad_fam in ["PAD_BARE", "PAD_WORD"]:
                for prof in cfg["inner_literal_profiles"]:
                    for sd in shadows:
                        nm = null_models.get((pad_fam, sd))
                        if nm is None or not nm.get("valid"):
                            continue
                        alpha, gain = nm["alpha"], nm["gain"]

                        for role, sid in [("assert", pair["assert_id"]),
                                          ("mislead", pair["mislead_id"])]:
                            e_raw = _get_effect(effects, tmpl["id"], prof["id"],
                                                "D3_RAW", sid, sd)
                            e_d4 = _get_effect(effects, tmpl["id"], prof["id"],
                                               "D4_NATURAL", sid, sd)
                            if e_raw is None or e_d4 is None:
                                continue
                            e_hat = alpha + gain * e_raw
                            residual = e_d4 - e_hat
                            blocks.append({
                                "template_id": tmpl["id"], "template_split": tmpl["split"],
                                "pair_id": pair["id"], "pair_split": pair["split"],
                                "padding_family": pad_fam, "literal_profile": prof["id"],
                                "shadow_digit": sd, "role": role, "surface_id": sid,
                                "e_raw": e_raw, "e_d4": e_d4, "e_hat": e_hat,
                                "residual": residual,
                            })

    kappa_blocks = []
    keyed = {}
    for b in blocks:
        key = (b["template_id"], b["pair_id"], b["padding_family"],
               b["literal_profile"], b["shadow_digit"])
        keyed.setdefault(key, {})[b["role"]] = b["residual"]

    for key, roles in keyed.items():
        if "assert" in roles and "mislead" in roles:
            kappa = roles["mislead"] - roles["assert"]
            kappa_blocks.append({
                "template_id": key[0], "pair_id": key[1], "padding_family": key[2],
                "literal_profile": key[3], "shadow_digit": key[4],
                "template_split": next(t["split"] for t in cfg["template_instances"]
                                       if t["id"] == key[0]),
                "pair_split": next(p["split"] for p in cfg["semantic_surface_pairs"]
                                   if p["id"] == key[1]),
                "kappa_gain": kappa,
            })
    return kappa_blocks


def _get_effect(effects, tmpl_id, prof_id, placement, surface_id, shadow_digit):
    for e in effects:
        if (e["template_id"] == tmpl_id and e["literal_profile"] == prof_id
                and e["placement"] == placement and e["surface_id"] == surface_id
                and e["shadow_digit"] == shadow_digit):
            return e["effect"]
    return None


def _holdout_kappa_stats(blocks, key="kappa"):
    ho = [b[key] for b in blocks if b["template_split"] == "holdout" and b["pair_split"] == "holdout"]
    if not ho:
        return {"n": 0}
    arr = np.array(ho)

    ho_tmpl_ids = list(set(b["template_id"] for b in blocks
                           if b["template_split"] == "holdout" and b["pair_split"] == "holdout"))
    ho_pair_ids = list(set(b["pair_id"] for b in blocks
                           if b["template_split"] == "holdout" and b["pair_split"] == "holdout"))

    block_vals = []
    for tid in ho_tmpl_ids:
        for pid in ho_pair_ids:
            cell = [b[key] for b in blocks
                    if b["template_id"] == tid and b["pair_id"] == pid
                    and b["template_split"] == "holdout" and b["pair_split"] == "holdout"]
            if cell:
                block_vals.append(np.mean(cell))

    block_arr = np.array(block_vals)
    mean_k = float(np.mean(block_arr))
    sign_agree = float(np.mean(np.sign(block_arr) == np.sign(mean_k))) if len(block_arr) > 0 else 0

    rng = np.random.RandomState(42017)
    boot_means = []
    for _ in range(100000):
        t_idx = rng.choice(len(ho_tmpl_ids), size=len(ho_tmpl_ids), replace=True)
        p_idx = rng.choice(len(ho_pair_ids), size=len(ho_pair_ids), replace=True)
        vals = []
        for ti in t_idx:
            for pi in p_idx:
                tid = ho_tmpl_ids[ti]
                pid = ho_pair_ids[pi]
                cell = [b[key] for b in blocks
                        if b["template_id"] == tid and b["pair_id"] == pid
                        and b["template_split"] == "holdout" and b["pair_split"] == "holdout"]
                if cell:
                    vals.append(np.mean(cell))
        if vals:
            boot_means.append(np.mean(vals))
    boot = np.array(boot_means)
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    marginals = {}
    for tid in ho_tmpl_ids:
        vals = [b[key] for b in blocks if b["template_id"] == tid
                and b["template_split"] == "holdout" and b["pair_split"] == "holdout"]
        marginals[f"tmpl_{tid}"] = float(np.mean(vals)) if vals else None
    for pid in ho_pair_ids:
        vals = [b[key] for b in blocks if b["pair_id"] == pid
                and b["template_split"] == "holdout" and b["pair_split"] == "holdout"]
        marginals[f"pair_{pid}"] = float(np.mean(vals)) if vals else None

    for factor in ["padding_family", "shadow_digit", "literal_profile"]:
        levels = set(b[factor] for b in blocks
                     if b["template_split"] == "holdout" and b["pair_split"] == "holdout")
        for lv in levels:
            vals = [b[key] for b in blocks if b[factor] == lv
                    and b["template_split"] == "holdout" and b["pair_split"] == "holdout"]
            marginals[f"{factor}_{lv}"] = float(np.mean(vals)) if vals else None

    return {"mean": mean_k, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "sign_agreement": sign_agree, "n_blocks": len(block_vals),
            "block_values": [float(v) for v in block_vals], "marginals": marginals}


def gate_isochronous(kappa_blocks, cfg):
    g = cfg["gates"]["G5_isochronous_curvature"]
    stats = _holdout_kappa_stats(kappa_blocks, "kappa")
    if stats["n_blocks"] == 0:
        return {"pass": False, "reason": "no holdout blocks", "stats": stats}

    checks = {
        "mean_ok": stats["mean"] <= g["heldout_mean_kappa_nat_max"],
        "ci_ok": stats["ci_hi"] <= g["heldout_ci_upper_max"],
        "sign_ok": stats["sign_agreement"] >= g["heldout_block_sign_agreement_min"],
        "retention_ok": abs(stats["mean"]) >= g["min_retention_fraction"] * g["original_abs_kappa_nat"],
    }

    m = stats["marginals"]
    pad_means = [v for k, v in m.items() if k.startswith("padding_family_") and v is not None]
    checks["each_pad_ok"] = all(v <= g["each_padding_family_mean_max"] for v in pad_means)
    if len(pad_means) >= 2:
        checks["pad_diff_ok"] = abs(pad_means[0] - pad_means[1]) <= g["max_abs_pad_family_diff"]
    else:
        checks["pad_diff_ok"] = True

    sd_means = [v for k, v in m.items() if k.startswith("shadow_digit_") and v is not None]
    checks["each_sd_ok"] = all(v <= g["each_shadow_digit_mean_max"] for v in sd_means)

    lp_means = [v for k, v in m.items() if k.startswith("literal_profile_") and v is not None]
    checks["each_lp_ok"] = all(v <= g["each_literal_profile_mean_max"] for v in lp_means)

    tmpl_marginals = [v for k, v in m.items() if k.startswith("tmpl_") and v is not None]
    checks["every_tmpl_neg"] = all(v < 0 for v in tmpl_marginals)

    pair_marginals = [v for k, v in m.items() if k.startswith("pair_") and v is not None]
    checks["every_pair_neg"] = all(v < 0 for v in pair_marginals)

    ok = all(checks.values())
    return {"pass": ok, "checks": checks, "stats": stats}


def gate_gain_residual(kappa_blocks, cfg):
    g = cfg["gates"]["G6_gain_null_residual"]
    stats = _holdout_kappa_stats(kappa_blocks, "kappa_gain")
    if stats.get("n_blocks", 0) == 0:
        return {"pass": False, "reason": "no holdout blocks", "stats": stats}

    checks = {
        "mean_ok": stats["mean"] <= g["heldout_mean_kappa_gain_nat_max"],
        "ci_ok": stats["ci_hi"] <= g["heldout_ci_upper_max"],
        "sign_ok": stats["sign_agreement"] >= g["heldout_block_sign_agreement_min"],
    }
    ok = all(checks.values())
    return {"pass": ok, "checks": checks, "stats": stats}


def print_gate(name, result):
    v = "PASS" if result["pass"] else "FAIL"
    print(f"--- {name}: {v} ---")
    for k, val in result.items():
        if k in ("pass", "stats", "models"):
            continue
        print(f"  {k}: {val}")
    if "stats" in result:
        s = result["stats"]
        for k in ("mean", "ci_lo", "ci_hi", "sign_agreement", "n_blocks"):
            if k in s:
                print(f"  {k}: {s[k]:.6f}" if isinstance(s.get(k), float) else f"  {k}: {s.get(k)}")
        if "marginals" in s:
            for mk, mv in s["marginals"].items():
                if mv is not None:
                    print(f"    {mk}: {mv:+.6f}")
    print(flush=True)


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/curvature_control_qwen3.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    result_dir = Path(cfg["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]

    print(f"=== {cfg['experiment_name']} ===")
    print(f"Config hash: {cfg_hash}\n")

    g0 = preflight(cfg)
    if not g0["pass"]:
        print("VERDICT: FAIL_INVALID_DESIGN_NO_MODEL_RUN")
        return

    from run_svb_0 import ModelAdapter
    adapter_cfg = {"model_id": cfg["model"]["model_id"], "device": "cpu", "dtype": "float32"}
    adapter = ModelAdapter(adapter_cfg)
    print(f"Model loaded.\n", flush=True)

    obs = collect(adapter, cfg)
    del adapter
    gc.collect()

    results = analyze(obs, cfg)
    results["config_sha256"] = cfg_hash
    results["call_count"] = len([o for o in obs if not o.get("is_replay")])

    all_pass = all(g.get("pass", False) for g in results["gates"].values())
    verdict = "PASS_CONTROLLED_CURVATURE" if all_pass else "FAIL_NO_CONTROLLED_CURVATURE"
    first_fail = next((k for k, v in results["gates"].items() if not v.get("pass")), None)
    results["verdict"] = verdict
    results["first_failing_gate"] = first_fail

    print(f"\n{'='*60}")
    print(f"  VERDICT: {verdict}")
    if first_fail:
        print(f"  First failing gate: {first_fail}")
    print(f"{'='*60}\n")

    save_obs = [{k: v for k, v in o.items() if k != "dist"} for o in obs]
    out = {
        "config": cfg, "verdict": verdict, "first_failing_gate": first_fail,
        "gates": results["gates"], "config_sha256": cfg_hash,
        "kappa_iso_summary": _holdout_kappa_stats(results.get("kappa_iso", []), "kappa"),
        "kappa_gain_summary": _holdout_kappa_stats(results.get("kappa_gain", []), "kappa_gain"),
        "null_models": results.get("gates", {}).get("G4", {}).get("models", {}),
    }
    with open(result_dir / "result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved to {result_dir / 'result.json'}", flush=True)

    dist_data = {f"{o['template_id']}_{o['literal_profile']}_{o['placement']}_{o['surface_id']}_sd{sd}":
                 o["z"][str(sd)] for o in obs for sd in cfg["task"]["analysis_shadow_digits"]
                 if not o["is_replay"]}
    with open(result_dir / "z_values.json", "w") as f:
        json.dump(dist_data, f, indent=2)


if __name__ == "__main__":
    main()
