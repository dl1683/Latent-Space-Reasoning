"""Frozen PT6 receiving panel, using production envelope/resampler/decoder APIs.

Run with -B, --offline and --device cpu. Stdout is one terminal JSON receipt;
stderr prints the hash-bound registration BEFORE model construction/evaluation.
Persist stdout outside the source tree to call panel_passed. Its default refuses
random-tiny receipts for an actual 31B fit. Explicit expected_model_scope="random-tiny"
allows a matching CPU mechanics receipt only, never a real-model authorization.

No S0 assembly is copied here. A mandatory gate imports current s0_memory_port and
calls its production forward with typed attachments, masks, labels and caches.
Only executed helpers are hash-bound; unrelated concurrent edits are irrelevant.
Callers own exact address selection and policy, not learned prefix positions/PLE.
Random logits are a numerical oracle, not semantic answers. Receipts are unkeyed
custody records from a trusted executor, not authentication of an arbitrary caller.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

from . import s0_envelope as envelope

FIXTURE_PATH = Path(__file__).resolve().with_name("s0_receiving_panel_v1.json")
FIXTURE_SHA256 = "24152b002bd52bddd5f9a87b81d2cb4c6dc3ea46c9de26d70da43751dfabdf3c"
MODULE_NAMES = ("s0_receiving_checks.py", "gemma4_receiver.py", "s0_envelope.py",
                "s0_memory_port.py")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _modules():
    root = Path(__file__).resolve().parent
    return {name: _sha(root / name) for name in MODULE_NAMES}


def _seal(receipt):
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _fixture(path=FIXTURE_PATH):
    if _sha(path) != FIXTURE_SHA256:
        raise ValueError("fixture SHA-256 differs from frozen config")
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    if (len(fixture["competence_cases"]) != 20 or len(fixture["paired_controls"]) != 8
            or len({c["id"] for c in fixture["competence_cases"]}) != 20
            or len({c["id"] for c in fixture["paired_controls"]}) != 8):
        raise ValueError("invalid frozen panel cardinality")
    return fixture


def _number(value, *, nonnegative=True):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (nonnegative and value < 0)):
        raise ValueError("measurement must be a finite number in its registered domain")
    return value


def _hash(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("invalid state SHA-256")
    return value


def _assert_verdict(detail, verdict):
    if detail["passed"] is not verdict:
        raise ValueError("asserted verdict contradicts measurements")
    return verdict


def _failed_measurement(detail):
    if "reason" not in detail:
        return False
    if (detail["passed"] is not False or not isinstance(detail["reason"], str)
            or not detail["reason"] or _number(detail["noise_floor"]) != 0):
        raise ValueError("invalid failed-instrument record")
    return True


def _arm_noise(detail):
    floor = max(_number(detail[key]) for key in
                ("repeat_max_abs", "batch_max_abs", "reordered_batch_max_abs"))
    if _number(detail["noise_floor"]) != floor:
        raise ValueError("arm noise floor contradicts measurements")
    return floor


def _pair_verdict(detail, expect, tol, *, check_flag=True):
    if _failed_measurement(detail):
        return False
    floor = max(_arm_noise(detail["left_noise"]), _arm_noise(detail["right_noise"]))
    effect = _number(detail["answer_max_abs_difference"])
    threshold = (0.0 if expect == "exact" else
                 (tol["effect_min_abs"] if expect == "different" else tol["invariance_abs"])
                 + tol["noise_multiplier"] * floor)
    if (detail["expected"] != expect or _number(detail["threshold"]) != threshold
            or _number(detail["noise_floor"]) != floor):
        raise ValueError("pair differs from frozen expectation/noise/threshold")
    verdict = bool(floor <= tol["noise_max_abs"] and
                   (effect > threshold if expect == "different" else effect <= threshold))
    return _assert_verdict(detail, verdict) if check_flag else verdict


def _gate_verdict(name, detail, fixture, expected_state=None):
    if _failed_measurement(detail):
        return False
    tol = fixture["tolerances"]
    if name == "exact_none_empty_bypass_logits_and_loss":
        errors = detail["max_abs_by_observable"]
        if set(errors) != {f"{mode}_{batch}_{observable}" for mode in ("none", "empty")
                           for batch in (1, 2) for observable in ("logits", "loss")}:
            raise ValueError("missing bypass measurements")
        verdict = all(_number(value) <= tol["bypass_abs"] for value in errors.values())
    elif name in ("returned_first_token_loss_gradient", "returned_answer_loss_gradient",
                  "returned_supplied_target_loss_gradient"):
        norms = detail["per_parameter_gradient_norms"]
        errors = detail["per_parameter_loss_gradient_max_abs"]
        required = set(fixture["measurement"]["connector_parameters"])
        if set(norms) != required or set(errors) != required:
            raise ValueError("missing intended connector gradient measurements")
        positions = ([0] if name == "returned_first_token_loss_gradient" else
                     list(range(*fixture["answer_slice"])))
        shift = (fixture["interface_controls"]["supplied_label_shift"]
                 if name == "returned_supplied_target_loss_gradient" else 0)
        targets = [(fixture["inputs"][i] + shift) % fixture["model"]["vocab_size"]
                   for i in positions]
        if (detail["supervised_text_positions"] != positions
                or detail["supervised_target_ids"] != targets
                or detail["memory_rows"] != fixture["content_recipe"]["rows"]
                or detail["loss_shape"] != [] or detail["optimizer_steps"] != 0):
            raise ValueError("wrong backward supervision/state/operation")
        loss, reference = _number(detail["loss"]), _number(detail["shifted_cross_entropy"])
        difference = _number(detail["loss_max_abs_difference"])
        if difference != abs(loss - reference):
            raise ValueError("loss difference contradicts measurements")
        norm_values = [_number(n) for n in norms.values()]
        gradient_errors = [_number(e) for e in errors.values()]
        memory_norm = _number(detail["memory_gradient_norm"])
        memory_error = _number(detail["memory_loss_gradient_max_abs"])
        for key in ("loss_requires_grad", "intended_gradients_finite",
                    "stem_frozen_eval_and_no_gradients"):
            if type(detail[key]) is not bool:
                raise ValueError("invalid backward gate state field")
        verdict = (detail["loss_requires_grad"] is True
                   and detail["intended_gradients_finite"] is True
                   and detail["stem_frozen_eval_and_no_gradients"] is True
                   and all(n > tol["gradient_min_norm"] for n in norm_values)
                   and memory_norm > tol["gradient_min_norm"]
                   and difference <= tol["loss_abs"]
                   and all(e <= tol["loss_gradient_abs"] for e in gradient_errors)
                   and memory_error <= tol["loss_gradient_abs"])
        before, after = _hash(detail["before_sha256"]), _hash(detail["after_sha256"])
        verdict = bool(verdict and before == after
                       and (expected_state is None or before == expected_state))
    elif name in ("attached_text_attention_mask", "attached_memory_mask"):
        registered = fixture["interface_controls"]
        masks = registered["text_masks" if name == "attached_text_attention_mask" else "memory_masks"]
        trials = detail["trials"]
        if (detail["mask_rows"] != masks
                or set(trials) != {"singleton_0", "singleton_1", "mixed_batch"}):
            raise ValueError("missing registered singleton/mixed-mask trials")
        results = []
        floors = []
        for trial in trials.values():
            if set(trial) != {"masked_invariance", "visible_effect"}:
                raise ValueError("missing masked invariance or matched visible effect")
            results.extend((_pair_verdict(trial["masked_invariance"], "equal", tol),
                            _pair_verdict(trial["visible_effect"], "different", tol)))
            floors.extend(p["noise_floor"] for p in trial.values())
        if _number(detail["noise_floor"]) != max(floors):
            raise ValueError("mask gate noise summary contradicts measurements")
        verdict = all(results)
    elif name == "nonzero_explicit_prefix_use":
        pairs = detail["pairs"]
        if (detail["seed"] != fixture["interface_controls"]["explicit_prefix_seed"]
                or set(pairs) != {"prefix_reference", "changed_prefix_reference", "prefix_effect"}):
            raise ValueError("missing registered nonzero explicit-prefix trials")
        results = [_pair_verdict(pairs[key], expect, tol) for key, expect in (
            ("prefix_reference", "equal"), ("changed_prefix_reference", "equal"),
            ("prefix_effect", "different"))]
        if _number(detail["noise_floor"]) != max(p["noise_floor"] for p in pairs.values()):
            raise ValueError("explicit-prefix noise contradicts measurements")
        verdict = all(results)
    elif name == "categorical_modality_use":
        return _pair_verdict(detail, "different", tol)
    elif name == "bounded_time_offsets":
        registered = fixture["interface_controls"]
        offsets = [t - fixture["content_recipe"]["host"]["as_of"]
                   for t in registered["older_event_times"]]
        if (detail["offsets_seconds"] != offsets
                or detail["supported_range_seconds"] != registered["supported_time_offsets_seconds"]):
            raise ValueError("wrong registered time domain")
        _, _, selection = _time_arms(fixture)
        _selection_equal(detail["selection"], selection, fixture)
        pairs = detail["pairs"]
        if set(pairs) != {"older_reference", "time_effect"}:
            raise ValueError("missing older time oracle or matched effect")
        results = [_pair_verdict(pairs["older_reference"], "equal", tol),
                   _pair_verdict(pairs["time_effect"], "different", tol)]
        if _number(detail["noise_floor"]) != max(p["noise_floor"] for p in pairs.values()):
            raise ValueError("time gate noise contradicts measurements")
        verdict = all(results)
    elif name == "appended_decode_token_masks":
        if detail["decode_masks"] != fixture["interface_controls"]["decode_masks"]:
            raise ValueError("wrong registered appended token masks")
        trials = detail["trials"]
        if set(trials) != {"singleton_0", "singleton_1", "mixed_batch"}:
            raise ValueError("missing decode singleton/mixed trials")
        results, floors = [], []
        for trial in trials.values():
            results.extend((_pair_verdict(trial["masked_invariance"], "equal", tol),
                            _pair_verdict(trial["visible_effect"], "different", tol)))
            floors.extend(trial[k]["noise_floor"] for k in ("masked_invariance", "visible_effect"))
            if type(trial["retained_mask_exact"]) is not bool:
                raise ValueError("invalid retained decode mask state")
            results.append(_number(trial["cached_recompute_max_abs"]) <= tol["cache_abs"]
                           and trial["retained_mask_exact"] is True)
        if _number(detail["noise_floor"]) != max(floors):
            raise ValueError("decode gate noise contradicts measurements")
        verdict = all(results)
    elif name == "production_s0_integration":
        trials = detail["trials"]
        if set(trials) != {"singleton_0", "singleton_1", "mixed_batch", "none", "empty"}:
            raise ValueError("missing production S0 trials")
        results = []
        for trial in trials.values():
            results.extend(_number(trial[k]) <= tol["loss_abs" if k == "loss_max_abs" else "cache_abs"]
                           for k in ("logits_max_abs", "loss_max_abs", "decode_max_abs"))
            if type(trial["returned_receiver_object"]) is not bool:
                raise ValueError("invalid S0 return identity")
            results.append(trial["returned_receiver_object"] is True)
        verdict = all(results)
    elif name in ("fresh_decode_cache_and_ownership", "masked_prefill_decode_cache_and_ownership"):
        error = _number(detail["cached_recompute_max_abs"])
        floor = _number(detail["repeat_batch_noise_floor"])
        effect = _number(detail["fresh_donor_effect"])
        steps = len(fixture["decode_tokens"])
        if (detail["decode_steps"] != steps or detail["trials_per_memory"] != 2
                or detail["batch_sizes"] != [1, 2]
                or detail["consumed_cache_refusals"] != 8 * steps
                or detail["foreign_cache_refusals"] != 8):
            raise ValueError("missing registered cache trials/refusals")
        if (type(detail["window_crossed"]) is not bool
                or type(detail["consumed_and_foreign_caches_rejected"]) is not bool):
            raise ValueError("invalid cache gate state field")
        masks = (fixture["interface_controls"]["text_masks"]
                 if name == "masked_prefill_decode_cache_and_ownership" else
                 [[1] * len(fixture["inputs"])] * 2)
        if detail["prefill_text_masks"] != masks:
            raise ValueError("cache gate differs from registered prefill masks")
        verdict = (error <= tol["cache_abs"] and floor <= tol["noise_max_abs"]
                   and effect > tol["effect_min_abs"] + tol["noise_multiplier"] * floor
                   and detail["window_crossed"] is True
                   and detail["consumed_and_foreign_caches_rejected"] is True)
    elif name == "unchanged_parameters":
        before, after = _hash(detail["before_sha256"]), _hash(detail["after_sha256"])
        verdict = before == after and (expected_state is None or before == expected_state)
    else:
        raise ValueError("unregistered gate")
    return _assert_verdict(detail, bool(verdict))


def _control_verdict(detail, case, fixture):
    if _failed_measurement(detail):
        return False
    tol = fixture["tolerances"]
    if case["kind"] == "shuffled_role_time":
        subpairs = detail["subpairs"]
        if set(subpairs) != set(case["require"]):
            raise ValueError("missing independent role/time controls")
        results = [_pair_verdict(subpairs[name], "different", tol) for name in case["require"]]
        if _number(detail["noise_floor"]) != max(p["noise_floor"] for p in subpairs.values()):
            raise ValueError("role/time noise summary contradicts measurements")
        verdict = all(results)
    else:
        _selection_verdict(detail, case, fixture)
        verdict = _pair_verdict(detail, case["expect"], tol, check_flag=False)
        if case["kind"] == "donor_swap":
            follow = _pair_verdict(detail["donor_reload_oracle"], "equal", tol)
            verdict = verdict and follow
        elif case["kind"] == "non_bypass_none":
            bypass = _gate_verdict("exact_none_empty_bypass_logits_and_loss",
                                   detail["bypass"], fixture)
            verdict = verdict and bypass
    return _assert_verdict(detail, bool(verdict))


def _control_noise(detail):
    return max(detail["noise_floor"],
               detail.get("donor_reload_oracle", {}).get("noise_floor", 0.0))


def _gate_noise(detail):
    return max(_number(detail.get("noise_floor", 0.0)),
               _number(detail.get("repeat_batch_noise_floor", 0.0)))


def _core_verdict(core, fixture, expected_state=None):
    cases, controls = core["per_case"], core["paired_controls"]
    expected_cases = fixture["competence_cases"]
    expected_controls = [c for c in fixture["paired_controls"] if c["expect"] != "reject"]
    for actual, expected in ((cases, expected_cases), (controls, expected_controls)):
        if [(c["id"], c["kind"]) for c in actual] != [(c["id"], c["kind"]) for c in expected]:
            raise ValueError("missing/reordered registered cases or controls")
    results = []
    for c, spec in zip(cases, expected_cases):
        _selection_verdict(c, spec, fixture)
        results.append(_pair_verdict(c, spec["expect"], fixture["tolerances"]))
    live = [_control_verdict(c, spec, fixture) for c, spec in zip(controls, expected_controls)]
    gates = core["critical_gates"]
    if set(gates) != set(fixture["measurement"]["critical_gates"]) - {"stable_module_hashes"}:
        raise ValueError("missing/unregistered critical gate")
    gate_results = [_gate_verdict(name, detail, fixture, expected_state)
                    for name, detail in gates.items()]
    floor = max(*(c["noise_floor"] for c in cases), *(_control_noise(c) for c in controls),
                *(_gate_noise(g) for g in gates.values()))
    if (type(core["competence_pass_count"]) is not int
            or core["competence_pass_count"] != sum(results)
            or _number(core["noise_floor"]) != floor):
        raise ValueError("core summary contradicts measurements")
    # A missing/nonfinite instrument cannot consume the one-small-effect allowance.
    return (sum(results) >= fixture["pass_rule"]["competence_required"] and all(live)
            and all(gate_results) and floor <= fixture["tolerances"]["noise_max_abs"]
            and all("reason" not in c for c in cases + controls))


def _core_pass(core, fixture):
    try:
        return _core_verdict(core, fixture)
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        return False


def panel_passed(receipt_path, *, expected_model_scope="actual-31B",
                 expected_model_state_sha256=None):
    """Fail closed on missing/stale/edited receipts and on tiny -> 31B promotion.

    An explicit random-tiny scope AND caller-supplied expected state hash can gate
    a CPU fit on this exact random instrument. There is no actual-31B check here. The real
    fit must additionally bind its actual model/checkpoint/config and gate its
    separate actual-port receipt; this function never upgrades tiny evidence.
    """
    try:
        receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
        if not _receipt_verdict(receipt, expected_model_scope, expected_model_state_sha256):
            return False
        fixture = _fixture()
        probes = receipt["receipt_adversaries"]
        return (probes["genuine_accepted"] is True
                and set(probes["variants"]) == set(fixture["pass_rule"]["must_fail_receipts"])
                and all(v is False for v in probes["variants"].values()))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _receipt_verdict(receipt, expected_model_scope, expected_model_state_sha256):
    """Shared measured receipt consumer; negative probes do not approve themselves."""
    try:
        fixture = _fixture()
        if (expected_model_scope != "random-tiny"
                or expected_model_state_sha256 is None
                or receipt["model_scope"] != expected_model_scope
                or receipt["schema"] != "s0-receiving-receipt/1"
                or receipt["check"] != fixture["check"]
                or receipt["status"] != "PASS" or receipt["exit_code"] != 0
                or receipt["fixture_sha256"] != FIXTURE_SHA256
                or receipt["registered_fixture_sha256"] != FIXTURE_SHA256
                or receipt["module_sha256"] != _modules()
                or receipt["config"] != fixture
                or receipt["tolerances"] != fixture["tolerances"]
                or receipt["receipt_sha256"] != _seal(receipt)
                or receipt["stable_module_hashes"] is not True):
            return False
        state = _hash(expected_model_state_sha256)
        if (_hash(receipt["model_state_sha256"]) != state
                or receipt["device"] != "cpu" or receipt["threads"] != 2
                or receipt["s0_integrated"] is not True):
            return False
        if (receipt["dependencies"] != {name: importlib.metadata.version(name)
                                       for name in ("torch", "transformers")}
                or receipt["per_case"] != receipt["production"]["per_case"]
                or receipt["production"]["passed"] is not True
                or not _core_verdict(receipt["production"], fixture, state)):
            return False
        controls = receipt["paired_controls"]
        if [(c["id"], c["kind"]) for c in controls] != [
                (c["id"], c["kind"]) for c in fixture["paired_controls"]]:
            return False
        if set(receipt["adversaries"]) != set(fixture["pass_rule"]["must_fail_receivers"]):
            return False
        for name in fixture["pass_rule"]["must_fail_receivers"]:
            null = receipt["adversaries"][name]
            if not _adversary_rejected(null, name, fixture, state):
                return False
        live = {c["id"]: c for c in receipt["production"]["paired_controls"]}
        for detail, spec in zip(controls, fixture["paired_controls"]):
            if spec["expect"] != "reject":
                if detail != live[spec["id"]] or not _control_verdict(detail, spec, fixture):
                    return False
                continue
            null = receipt["adversaries"][spec["kind"]]
            paired = detail["paired_measurement"]
            if (not _pair_verdict(paired, "equal", fixture["tolerances"])
                    or detail["adversary_core_passed"] is not False
                    or detail["adversary_competence_pass_count"] != null["competence_pass_count"]
                    or _number(detail["paired_logits_difference"]) != 0
                    or _number(detail["noise_floor"]) != paired["noise_floor"]
                    or type(detail["distinct_digests_echoed"]) is not bool):
                return False
            verdict = (_adversary_rejected(null, spec["kind"], fixture, state)
                       and paired["passed"] is True)
            if spec["kind"] == "digest_echo_receiver":
                verdict = verdict and detail["distinct_digests_echoed"]
            if not _assert_verdict(detail, bool(verdict)):
                return False
        floor = max(receipt["production"]["noise_floor"], *(_control_noise(c) for c in controls))
        return (_number(receipt["noise_floor"]) == floor
                and floor <= fixture["tolerances"]["noise_max_abs"])
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError,
            importlib.metadata.PackageNotFoundError):
        return False


def _state_hash(receiver):
    import torch
    h = hashlib.sha256()
    for name, tensor in sorted(receiver.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(str((tensor.dtype, tuple(tensor.shape))).encode("ascii"))
        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _build(fixture):
    import torch
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    from .gemma4_receiver import Gemma4Receiver, MemoryResampler, ResamplerConfig
    for name, version in fixture["dependencies"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"frozen dependency mismatch: {name}")
    torch.random.default_generator.manual_seed(fixture["random_seed"])
    config = Gemma4TextConfig(**fixture["model"])
    config._attn_implementation = "eager"
    with torch.device("cpu"):
        stem = Gemma4ForCausalLM(config).float().eval()
        connector = MemoryResampler(ResamplerConfig(**fixture["connector"])).float()
    return Gemma4Receiver(stem, connector).eval()


def _rows(fixture, seed):
    import torch
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(fixture["content_recipe"]["rows"], envelope.WIDTH,
                       generator=generator, dtype=torch.float32, device="cpu")


def _origins(fixture):
    recipe = fixture["content_recipe"]
    return [{"origin_type": role, "modality": recipe["modalities"][i],
             "source_sha256": hashlib.sha256(f"fixture-row-{i}".encode()).hexdigest(),
             "locator": f"fixture:row:{i}", "event_time": recipe["event_times"][i],
             "ingestion_time": recipe["ingestion_times"][i],
             "valid_from": recipe["valid_from"], "valid_until": recipe["valid_until"],
             "branch": recipe["branch"], "dependencies": [],
             "permissions": recipe["permissions"][i]}
            for i, role in enumerate(recipe["roles"])]


def _attachment(fixture, rows, *, origins=None, host=None, addresses=None,
                order=None, reload=False):
    """Translate typed production selection to row-level receiver arguments only."""
    import torch
    origins = copy.deepcopy(_origins(fixture) if origins is None else origins)
    indices = list(range(rows.shape[0]))
    if order is not None:
        rows, origins = rows[order], [origins[i] for i in order]
        indices = [indices[i] for i in order]
    mask = [int(addresses is None or i in addresses) for i in indices]
    env = envelope.make_envelope(rows, origin=origins, mask=mask, index_map=indices)
    if reload:
        env = envelope.deserialize_envelope(envelope.serialize_envelope(env))
    selected, refusals = envelope.select(
        [env], copy.deepcopy(fixture["content_recipe"]["host"] if host is None else host))
    roles, offsets = envelope.role_time_tensors(selected)
    if not torch.isfinite(offsets.float()).all():
        raise ValueError("conditioning overflows receiver float32")
    memory = torch.cat([envelope.to_torch(e["payload"]) for e in selected], 0) if selected else (
        torch.empty(0, envelope.WIDTH, dtype=torch.float32))
    modalities = [r["modality"] for e in selected for r in e["meta"]["origin"]]
    types = torch.tensor([("text", "image", "video", "audio").index(m) for m in modalities],
                         dtype=torch.long)
    args = {"input_ids": torch.tensor([fixture["inputs"]], dtype=torch.long),
            "memory": memory.unsqueeze(0), "role_ids": roles.unsqueeze(0),
            "type_ids": types.unsqueeze(0), "time_offsets": offsets.float().unsqueeze(0),
            "memory_mask": torch.ones((1, memory.shape[0]), dtype=torch.bool),
            "assembly_digest": env["digest"]}
    detail = {"digest": env["digest"],
              "selected_indices": [i for e in selected for i in e["index_map"]],
              "refusals": refusals}
    return args, detail


def _arms(fixture, case):
    import torch
    rows = _rows(fixture, case["seed"])
    kind = case["kind"]
    left, left_info = _attachment(fixture, rows)
    right, right_info = _attachment(fixture, -rows + 2)
    if kind == "equal":
        right, right_info = _attachment(fixture, rows.clone())
    elif kind == "reload":
        right, right_info = _attachment(fixture, rows, reload=True)
    elif kind == "address":
        left, left_info = _attachment(fixture, rows, addresses=case["addresses"][:1])
        right, right_info = _attachment(fixture, rows, addresses=case["addresses"][1:])
    elif kind in ("permission", "max_age", "as_of", "branch"):
        host = copy.deepcopy(fixture["content_recipe"]["host"])
        if kind == "permission":
            host["permissions"] = ["read"]
        else:
            host[kind] = case[kind]
        right, right_info = _attachment(fixture, rows, host=host)
    elif kind == "masked":
        right, right_info = _attachment(fixture, rows)
        mask = torch.tensor([case["mask"]], dtype=torch.bool)
        left["memory_mask"] = right["memory_mask"] = mask
        right["memory"] = right["memory"].clone()
        right["memory"][:, ~mask[0]] += 1000
        right_info["intervention"] = "changed only production receiver-masked rows by +1000"
    elif kind in ("roles", "types", "times"):
        origins = _origins(fixture)
        field = {"roles": "origin_type", "types": "modality", "times": "event_time"}[kind]
        values = [r[field] for r in origins][::-1]
        for rec, value in zip(origins, values):
            rec[field] = value
            if kind == "times":
                rec["ingestion_time"] = value
        right, right_info = _attachment(fixture, rows, origins=origins)
    elif kind in ("row_binding", "wrong_address"):
        left, left_info = _attachment(fixture, rows, addresses=case["addresses"])
        right, right_info = _attachment(fixture, rows.flip(0), addresses=case["addresses"])
    elif kind == "aligned_permutation":
        right, right_info = _attachment(fixture, rows, order=[3, 2, 1, 0])
    elif kind == "omission":
        right = {"input_ids": left["input_ids"], "memory": None}
        right_info = {"selected_indices": [], "digest": None, "refusals": []}
    elif kind == "zero_prefix":
        right = {"input_ids": left["input_ids"], "prefix_states": torch.zeros(
            1, 8, fixture["model"]["hidden_size"])}
        right_info = {"selected_indices": [], "digest": None, "refusals": [],
                      "intervention": "eight literal zero prefix embeddings, NOT bypass"}
    return left, right, {"left": left_info, "right": right_info}


def _selection_equal(actual, expected, fixture):
    if not isinstance(actual, dict) or set(actual) != {"left", "right"}:
        raise ValueError("missing selection arms")
    for arm in actual.values():
        indices = arm["selected_indices"]
        if (not isinstance(indices, list)
                or any(type(i) is not int or not 0 <= i < fixture["content_recipe"]["rows"]
                       for i in indices) or len(set(indices)) != len(indices)):
            raise ValueError("illegal selected index")
        if arm["digest"] is not None:
            _hash(arm["digest"])
        if not isinstance(arm["refusals"], list):
            raise ValueError("missing selection refusals")
    # Canonical JSON also distinguishes bool from int; compare reconstructible
    # digest bindings, refusal reasons/order and declared intervention text.
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("selection contradicts deterministic fixture policy/binding")


def _selection_verdict(detail, case, fixture):
    if _failed_measurement(detail) or case["kind"] == "non_bypass_none":
        return
    _, _, expected = _arms(fixture, case)
    _selection_equal(detail["selection"], expected, fixture)


def _batch(args, reverse=False):
    import torch
    result = dict(args)
    for key, value in args.items():
        if isinstance(value, torch.Tensor):
            companion = value.clone()
            if key == "input_ids":
                companion = value.flip(1)
            elif key in ("memory", "prefix_states"):
                companion = -value + 1
            result[key] = torch.cat((value, companion), 0)
            if reverse:
                result[key] = result[key].flip(0)
    return result


def _difference(a, b):
    import torch
    if a.shape != b.shape or not a.numel() or not all(torch.isfinite(x).all() for x in (a, b)):
        raise AssertionError("unaligned, empty or nonfinite observable")
    return (a - b).abs().max().item()


def _measure(receiver, fixture, args):
    import torch
    start, stop = fixture["answer_slice"]
    samples = args["input_ids"].shape[0]

    def answer(options, index=0, reverse=False):
        out = receiver(**options)
        logits = out["logits"]
        expected = (options["input_ids"].shape[0], options["input_ids"].shape[1],
                    fixture["model"]["vocab_size"])
        if logits.shape != expected or not torch.isfinite(logits).all():
            raise AssertionError("invalid/nonfinite full returned logits")
        result = logits[index:index+samples, start:stop].detach().clone()
        return result.flip(0) if reverse else result

    with torch.no_grad():
        first = answer(args)
        repeat = _difference(first, answer(args))
        batch = _difference(first, answer(_batch(args)))
        reordered = _difference(first, answer(_batch(args, True), samples, True))
    return first, {"repeat_max_abs": repeat, "batch_max_abs": batch,
                   "reordered_batch_max_abs": reordered,
                   "noise_floor": max(repeat, batch, reordered)}


def _pair(receiver, fixture, left, right, expect, *, right_receiver=None):
    a, a_noise = _measure(receiver, fixture, left)
    b, b_noise = _measure(receiver if right_receiver is None else right_receiver, fixture, right)
    effect = _difference(a, b)
    tol = fixture["tolerances"]
    floor = max(a_noise["noise_floor"], b_noise["noise_floor"])
    threshold = (tol["effect_min_abs"] if expect == "different" else
                 tol["invariance_abs"]) + tol["noise_multiplier"] * floor
    passed = (effect > threshold if expect == "different" else
              effect == 0 if expect == "exact" else effect <= threshold)
    return {"passed": bool(passed and floor <= tol["noise_max_abs"]),
            "expected": expect, "answer_max_abs_difference": effect,
            "threshold": 0 if expect == "exact" else threshold, "noise_floor": floor,
            "left_noise": a_noise, "right_noise": b_noise}


def _bypass_gate(receiver, native, fixture):
    import torch
    ids = torch.tensor([fixture["inputs"]], dtype=torch.long)
    labels = torch.full_like(ids, -100)
    start, stop = fixture["answer_slice"]
    labels[:, start:stop] = ids[:, start:stop]
    errors = {}
    with torch.no_grad():
        for batch in (1, 2):
            batch_ids = ids.expand(batch, -1).clone()
            batch_labels = labels.expand(batch, -1).clone()
            mask = torch.ones_like(batch_ids)
            if batch == 2:
                mask[1, :2] = 0
            direct = native.stem(input_ids=batch_ids, attention_mask=mask, labels=batch_labels,
                                 use_cache=False, return_dict=True)
            for name, memory in (("none", None), ("empty", torch.empty(batch, 0, envelope.WIDTH))):
                out = receiver(input_ids=batch_ids, memory=memory, attention_mask=mask,
                               labels=batch_labels)
                errors[f"{name}_{batch}_logits"] = _difference(out["logits"], direct.logits)
                errors[f"{name}_{batch}_loss"] = _difference(out["loss"], direct.loss)
    return {"passed": all(x == fixture["tolerances"]["bypass_abs"] for x in errors.values()),
            "max_abs_by_observable": errors}


def _mask_gate(receiver, fixture, *, text):
    import torch
    registered = fixture["interface_controls"]
    masks = registered["text_masks" if text else "memory_masks"]
    seed = registered["text_mask_seed" if text else "memory_mask_seed"]
    base, _ = _attachment(fixture, _rows(fixture, seed))
    mixed = _batch(base)
    key = "attention_mask" if text else "memory_mask"
    mixed[key] = torch.tensor(masks, dtype=torch.long if text else torch.bool)
    trials = {}
    for name, left in (
            ("singleton_0", {k: v[:1] if isinstance(v, torch.Tensor) else v
                             for k, v in mixed.items()}),
            ("singleton_1", {k: v[1:] if isinstance(v, torch.Tensor) else v
                             for k, v in mixed.items()}),
            ("mixed_batch", mixed)):
        hidden, visible = dict(left), dict(left)
        if text:
            changed = (left["input_ids"] + registered["text_token_shift"]) % fixture["model"]["vocab_size"]
            hidden["input_ids"] = torch.where(left[key].bool(), left["input_ids"], changed)
            visible["input_ids"] = left["input_ids"].clone()
            position = registered["text_visible_position"]
            if not left[key][:, position].all():
                raise ValueError("registered text intervention is not visible")
            visible["input_ids"][:, position] = changed[:, position]
        else:
            shift = registered["memory_row_shift"]
            hidden["memory"] = left["memory"] + (~left[key]).unsqueeze(-1) * shift
            visible["memory"] = left["memory"] + left[key].unsqueeze(-1) * shift
        trials[name] = {
            "masked_invariance": _pair(receiver, fixture, left, hidden, "equal"),
            "visible_effect": _pair(receiver, fixture, left, visible, "different")}
    return {"passed": all(p["passed"] for trial in trials.values() for p in trial.values()),
            "mask_rows": masks, "trials": trials,
            "noise_floor": max(p["noise_floor"] for trial in trials.values() for p in trial.values())}


def _modality_gate(receiver, fixture):
    left, right, _ = _arms(fixture, {
        "seed": fixture["interface_controls"]["modality_seed"], "kind": "types"})
    return _pair(receiver, fixture, left, right, "different")


def _prefix_gate(receiver, native, fixture):
    import torch
    seed = fixture["interface_controls"]["explicit_prefix_seed"]
    explicit = []
    with torch.no_grad():
        for rows in (_rows(fixture, seed), -_rows(fixture, seed) + 2):
            args, _ = _attachment(fixture, rows, reload=True)
            prefix = native.connector(args["memory"], args["memory_mask"], args["role_ids"],
                                      args["type_ids"], args["time_offsets"])
            if not torch.isfinite(prefix).all() or not torch.count_nonzero(prefix):
                raise AssertionError("nonzero explicit-prefix control unavailable")
            explicit.append({"input_ids": args["input_ids"], "prefix_states": prefix,
                             "attention_mask": torch.tensor(
                                 fixture["interface_controls"]["text_masks"][:1], dtype=torch.long)})
    left, right = explicit
    pairs = {"prefix_reference": _pair(receiver, fixture, left, left, "equal", right_receiver=native),
             "changed_prefix_reference": _pair(
                 receiver, fixture, right, right, "equal", right_receiver=native),
             "prefix_effect": _pair(receiver, fixture, left, right, "different")}
    return {"passed": all(p["passed"] for p in pairs.values()), "seed": seed, "pairs": pairs,
            "noise_floor": max(p["noise_floor"] for p in pairs.values())}


def _time_arms(fixture):
    rows = _rows(fixture, fixture["interface_controls"]["time_seed"])
    origins = _origins(fixture)
    for rec, event in zip(origins, fixture["interface_controls"]["older_event_times"]):
        rec["event_time"] = rec["ingestion_time"] = event
    older, older_info = _attachment(fixture, rows, origins=origins)
    recent, recent_info = _attachment(fixture, rows)
    return older, recent, {"left": older_info, "right": recent_info}


def _time_gate(receiver, native, fixture):
    older, recent, selection = _time_arms(fixture)
    pairs = {"older_reference": _pair(receiver, fixture, older, older, "equal",
                                       right_receiver=native),
             "time_effect": _pair(receiver, fixture, older, recent, "different")}
    return {"passed": all(p["passed"] for p in pairs.values()), "pairs": pairs,
            "selection": selection, "offsets_seconds": older["time_offsets"][0].tolist(),
            "supported_range_seconds": fixture["interface_controls"]["supported_time_offsets_seconds"],
            "noise_floor": max(p["noise_floor"] for p in pairs.values())}


def _decode_measure(receiver, fixture, base, tokens, masks):
    """Fresh cache each time; retain positions and compare every step to recompute."""
    import torch
    errors, retained = [], []

    def trace(args, token_rows, mask_rows):
        args = dict(args)
        current = receiver(**args, use_cache=True)
        for step in range(tokens.shape[1]):
            token, mask = token_rows[:, step:step+1], mask_rows[:, step:step+1]
            current = receiver.decode(token, current["past_key_values"], token_mask=mask)
            args["input_ids"] = torch.cat((args["input_ids"], token), 1)
            args["attention_mask"] = torch.cat((args["attention_mask"], mask), 1)
            full = receiver(**args, use_cache=False)
            errors.append(_difference(current["logits"][:, -1:], full["logits"][:, -1:]))
            expected = torch.cat((torch.ones(mask.shape[0], 8, dtype=torch.long),
                                  args["attention_mask"]), 1)
            retained.append(torch.equal(current["past_key_values"].attention_mask, expected))
        return current["logits"][:, -1:].detach().clone()

    with torch.no_grad():
        first = trace(base, tokens, masks)
        repeat = _difference(first, trace(base, tokens, masks))
        doubled_tokens, doubled_masks = torch.cat((tokens, tokens)), torch.cat((masks, masks))
        batch = _difference(first, trace(_batch(base), doubled_tokens, doubled_masks)[:len(tokens)])
        reordered = _difference(first, trace(
            _batch(base, True), doubled_tokens.flip(0), doubled_masks.flip(0)
        )[len(tokens):].flip(0))
    noise = {"repeat_max_abs": repeat, "batch_max_abs": batch,
             "reordered_batch_max_abs": reordered, "noise_floor": max(repeat, batch, reordered)}
    return first, noise, max(errors), all(retained)


def _decode_mask_gate(receiver, fixture):
    import torch
    registered = fixture["interface_controls"]
    base, _ = _attachment(fixture, _rows(fixture, registered["decode_mask_seed"]))
    mixed = _batch(base)
    mixed["attention_mask"] = torch.tensor(registered["text_masks"], dtype=torch.long)
    masks = torch.tensor(registered["decode_masks"], dtype=torch.long)
    tokens = torch.tensor([fixture["decode_tokens"]] * 2, dtype=torch.long)
    changed = torch.where(masks.bool(), tokens,
                          (tokens + registered["decode_token_shift"]) % fixture["model"]["vocab_size"])
    trials = {}
    for name, index in (("singleton_0", slice(0, 1)), ("singleton_1", slice(1, 2)),
                        ("mixed_batch", slice(None))):
        args = {k: v[index] if isinstance(v, torch.Tensor) else v for k, v in mixed.items()}
        selected_tokens, selected_changed, selected_masks = tokens[index], changed[index], masks[index]
        measurements = [
            _decode_measure(receiver, fixture, args, selected_tokens, selected_masks),
            _decode_measure(receiver, fixture, args, selected_changed, selected_masks),
            _decode_measure(receiver, fixture, args, selected_tokens, torch.ones_like(selected_masks)),
            _decode_measure(receiver, fixture, args, selected_changed, torch.ones_like(selected_masks))]
        pairs = {}
        for key, expect, left, right in (("masked_invariance", "equal", 0, 1),
                                         ("visible_effect", "different", 2, 3)):
            a, a_noise, _, _ = measurements[left]
            b, b_noise, _, _ = measurements[right]
            floor = max(a_noise["noise_floor"], b_noise["noise_floor"])
            effect = _difference(a, b)
            tol = fixture["tolerances"]
            threshold = (tol["invariance_abs"] if expect == "equal" else
                         tol["effect_min_abs"]) + tol["noise_multiplier"] * floor
            pairs[key] = {"passed": floor <= tol["noise_max_abs"] and
                          (effect <= threshold if expect == "equal" else effect > threshold),
                          "expected": expect, "answer_max_abs_difference": effect,
                          "threshold": threshold, "noise_floor": floor,
                          "left_noise": a_noise, "right_noise": b_noise}
        trials[name] = {**pairs, "cached_recompute_max_abs": max(m[2] for m in measurements),
                        "retained_mask_exact": all(m[3] for m in measurements)}
    detail = {"decode_masks": registered["decode_masks"], "trials": trials,
              "noise_floor": max(t[k]["noise_floor"] for t in trials.values()
                                 for k in ("masked_invariance", "visible_effect"))}
    detail["passed"] = all(
        t["masked_invariance"]["passed"] and t["visible_effect"]["passed"]
        and t["cached_recompute_max_abs"] <= fixture["tolerances"]["cache_abs"]
        and t["retained_mask_exact"] for t in trials.values())
    return detail


def _integration_gate(receiver, native, fixture):
    """Exercise S0 itself, not a duplicated forward/selection implementation."""
    import torch
    from . import s0_memory_port as s0
    from .gemma4_receiver import Gemma4Receiver

    class RecordingReceiver(Gemma4Receiver):
        def forward(self, **args):
            self.returned = receiver(**args)
            return self.returned

        def decode(self, token, cache, token_mask=None):
            return receiver.decode(token, cache, token_mask=token_mask)

    port = RecordingReceiver(native.stem, native.connector)
    handle = {"receiver": port, "connector": native.connector, "stem_info": {
        "D_memory": fixture["connector"]["memory_dim"], "D_stem": fixture["model"]["hidden_size"]}}
    host = fixture["content_recipe"]["host"]
    rows = _rows(fixture, 4008)
    specs = []
    for sample, addresses in enumerate(([0, 2], [1, 3])):
        content = rows if sample == 0 else -rows + 1
        stored = envelope.make_envelope(content, origin=_origins(fixture),
                                       mask=[int(i in addresses) for i in range(len(rows))])
        stored = envelope.deserialize_envelope(envelope.serialize_envelope(stored))
        direct, _ = _attachment(fixture, content, addresses=addresses)
        specs.append((stored, direct))
    trials = {}
    for name, samples in (("singleton_0", [0]), ("singleton_1", [1]), ("mixed_batch", [0, 1]),
                          ("none", []), ("empty", [])):
        batch = len(samples) or 1
        if samples:
            direct = {k: torch.cat([specs[i][1][k] for i in samples], 0)
                      for k in specs[0][1] if isinstance(specs[0][1][k], torch.Tensor)}
            memory = [specs[i][0] for i in samples]
        else:
            direct = {"input_ids": torch.tensor([fixture["inputs"]], dtype=torch.long), "memory": None}
            memory = None if name == "none" else envelope.empty_envelope()
        direct["attention_mask"] = torch.tensor(
            [fixture["interface_controls"]["text_masks"][i] for i in (samples or [0])],
            dtype=torch.long)
        labels = torch.full_like(direct["input_ids"], -100)
        start, stop = fixture["answer_slice"]
        labels[:, start:stop] = direct["input_ids"][:, start:stop]
        inputs = {k: direct[k] for k in ("input_ids", "attention_mask")}
        out = s0.forward(handle, {**inputs, "labels": labels}, memory=memory, conditioning=host)
        returned = out is port.returned
        reference = native(**direct, labels=labels)
        logit_error, loss_error = (_difference(out["logits"], reference["logits"]),
                                  _difference(out["loss"], reference["loss"]))
        with torch.no_grad():
            cached = s0.forward(handle, {**inputs, "use_cache": True}, memory=memory, conditioning=host)
            oracle = native(**direct, use_cache=True)
            decode_errors = []
            for token_id, visible in zip(fixture["decode_tokens"][:2], (0, 1)):
                token = torch.full((batch, 1), token_id, dtype=torch.long)
                mask = torch.full_like(token, visible)
                cached = port.decode(token, cached["past_key_values"], token_mask=mask)
                oracle = native.decode(token, oracle["past_key_values"], token_mask=mask)
                decode_errors.append(_difference(cached["logits"], oracle["logits"]))
        trials[name] = {"logits_max_abs": logit_error, "loss_max_abs": loss_error,
                        "decode_max_abs": max(decode_errors), "returned_receiver_object": returned}
    detail = {"trials": trials}
    detail["passed"] = _integration_pass(trials, fixture)
    return detail


def _integration_pass(trials, fixture):
    tol = fixture["tolerances"]
    return all(t["returned_receiver_object"] and t["logits_max_abs"] <= tol["cache_abs"]
               and t["loss_max_abs"] <= tol["loss_abs"] and t["decode_max_abs"] <= tol["cache_abs"]
               for t in trials.values())


def _gradient_gate(receiver, native, fixture, *, answer_slice, supplied_targets=False):
    import torch
    from torch.nn import functional as F
    before = _state_hash(native)
    native.zero_grad(set_to_none=True)
    seed = fixture["interface_controls"]["supplied_label_seed"] if supplied_targets else 4001
    args, _ = _attachment(fixture, _rows(fixture, seed))
    args["memory"] = args["memory"].clone().requires_grad_(True)
    labels = torch.full_like(args["input_ids"], -100)
    positions = list(range(*fixture["answer_slice"])) if answer_slice else [0]
    shift = fixture["interface_controls"]["supplied_label_shift"] if supplied_targets else 0
    labels[:, positions] = (args["input_ids"][:, positions] + shift) % fixture["model"]["vocab_size"]
    out = receiver(**args, labels=labels, use_cache=False)
    loss = out["loss"]
    if (not isinstance(loss, torch.Tensor) or loss.ndim != 0
            or not torch.isfinite(loss)):
        raise AssertionError("returned answer loss is not a finite scalar tensor")
    if answer_slice:
        # Labels name target tokens; their predictors are the preceding text logits.
        start, stop = fixture["answer_slice"]
        prediction = out["logits"][:, start - 1:stop - 1]
    else:
        # Keep the separate last-prefix-slot -> text-token-zero regression.
        prediction = out["model_output"].logits[:, 7:8]
    reference = F.cross_entropy(prediction.float().reshape(-1, fixture["model"]["vocab_size"]),
                                labels[:, positions].reshape(-1))
    if not torch.isfinite(reference):
        raise AssertionError("shifted cross entropy is not finite")
    parameters = list(native.connector.named_parameters())
    targets = [p for _, p in parameters] + [args["memory"]]
    # Observe a detached/missing gradient as a measured contract violation.
    # Execution exceptions still belong to _guarded's failed instrument records.
    returned_grads = (torch.autograd.grad(loss, targets, retain_graph=True, allow_unused=True)
                      if loss.requires_grad else [None] * len(targets))
    reference_grads = (torch.autograd.grad(reference, targets, retain_graph=True, allow_unused=True)
                       if reference.requires_grad else [None] * len(targets))
    gradient_errors = [_difference(torch.zeros_like(target) if a is None else a,
                                   torch.zeros_like(target) if b is None else b)
                       for a, b, target in zip(returned_grads, reference_grads, targets)]
    if loss.requires_grad:
        loss.backward()
    if (any(p.grad is not None and not torch.isfinite(p.grad).all()
            for p in native.connector.parameters())
            or (args["memory"].grad is not None and
                not torch.isfinite(args["memory"].grad).all())):
        raise AssertionError("nonfinite intended-path gradient")
    minimum = fixture["tolerances"]["gradient_min_norm"]
    norms = {name: 0.0 if p.grad is None else p.grad.float().norm().item()
             for name, p in native.connector.named_parameters()}
    memory_grad = args["memory"].grad
    memory_norm = 0.0 if memory_grad is None else memory_grad.float().norm().item()
    frozen = (not native.stem.training and all(
        not p.requires_grad and p.grad is None for p in native.stem.parameters()))
    finite = (all(a is not None and b is not None and torch.isfinite(a).all()
                  and torch.isfinite(b).all() for a, b in zip(returned_grads, reference_grads))
              and all(p.grad is not None and torch.isfinite(p.grad).all()
                      for p in native.connector.parameters()))
    loss_value, reference_value = loss.detach().item(), reference.detach().item()
    loss_error = abs(loss_value - reference_value)
    after = _state_hash(native)
    passed = (finite and frozen and memory_grad is not None and torch.isfinite(memory_grad).all()
              and all(math.isfinite(n) and n > minimum for n in norms.values())
              and math.isfinite(memory_norm) and memory_norm > minimum
              and loss_error <= fixture["tolerances"]["loss_abs"]
              and max(gradient_errors) <= fixture["tolerances"]["loss_gradient_abs"]
              and before == after)
    return {"passed": bool(passed), "loss": loss_value, "shifted_cross_entropy": reference_value,
            "loss_max_abs_difference": loss_error, "loss_shape": list(loss.shape),
            "loss_requires_grad": loss.requires_grad, "intended_gradients_finite": bool(finite),
            "per_parameter_loss_gradient_max_abs": {
                name: error for (name, _), error in zip(parameters, gradient_errors[:-1])},
            "memory_loss_gradient_max_abs": gradient_errors[-1],
            "supervised_text_positions": positions, "memory_rows": args["memory"].shape[1],
            "supervised_target_ids": labels[0, positions].tolist(),
            "per_parameter_gradient_norms": norms, "memory_gradient_norm": memory_norm,
            "stem_frozen_eval_and_no_gradients": frozen, "optimizer_steps": 0,
            "before_sha256": before, "after_sha256": after}


def _cache_gate(receiver, native, fixture, *, masked=False):
    import torch
    tol = fixture["tolerances"]
    terminal, errors, repeats = {}, [], []
    consumed_refusals, foreign_refusals = 0, 0
    masks = (fixture["interface_controls"]["text_masks"] if masked else
             [[1] * len(fixture["inputs"])] * 2)
    with torch.no_grad():
        for name, rows in (("original", _rows(fixture, 4002)),
                           ("donor", -_rows(fixture, 4002) + 2)):
            base, _ = _attachment(fixture, rows)
            base["attention_mask"] = torch.tensor(masks[:1], dtype=torch.long)
            fresh = []
            for trial in range(2):
                for batch in (1, 2):
                    args = base if batch == 1 else _batch(base, reverse=True)
                    args = dict(args)
                    if batch == 2:
                        args["attention_mask"] = torch.tensor(masks[::-1], dtype=torch.long)
                    current = receiver(**args, use_cache=True)
                    for token_id in fixture["decode_tokens"]:
                        token = torch.full((batch, 1), token_id, dtype=torch.long)
                        old = current["past_key_values"]
                        current = receiver.decode(token, old)
                        args["input_ids"] = torch.cat((args["input_ids"], token), 1)
                        args["attention_mask"] = torch.cat((args["attention_mask"], torch.ones_like(token)), 1)
                        recompute = receiver(**args, use_cache=False)
                        errors.append(_difference(current["logits"][:, -1:],
                                                  recompute["logits"][:, -1:]))
                        try:
                            receiver.decode(token, old)
                        except ValueError:
                            consumed_refusals += 1
                        else:
                            raise AssertionError("consumed cache accepted")
                    index = 0 if batch == 1 else 1
                    final = current["logits"][index:index+1, -1:].clone()
                    if batch == 1:
                        fresh.append(final)
                        terminal[name] = final
                    else:
                        repeats.append(_difference(fresh[-1], final))
                    foreign = replace(current["past_key_values"], owner=object())
                    try:
                        receiver.decode(token, foreign)
                    except ValueError:
                        foreign_refusals += 1
                    else:
                        raise AssertionError("foreign cache accepted")
            repeats.append(_difference(*fresh))
    effect = _difference(terminal["original"], terminal["donor"])
    floor = max(repeats)
    return {"passed": max(errors) <= tol["cache_abs"] and floor <= tol["noise_max_abs"]
            and effect > tol["effect_min_abs"] + tol["noise_multiplier"] * floor,
            "cached_recompute_max_abs": max(errors), "repeat_batch_noise_floor": floor,
            "fresh_donor_effect": effect, "decode_steps": len(fixture["decode_tokens"]),
            "trials_per_memory": 2, "batch_sizes": [1, 2],
            "prefill_text_masks": masks,
            "window_crossed": (len(fixture["inputs"]) + 8 + len(fixture["decode_tokens"])
                               > fixture["model"]["sliding_window"]),
            "consumed_and_foreign_caches_rejected": True,
            "consumed_cache_refusals": consumed_refusals,
            "foreign_cache_refusals": foreign_refusals}


def _guarded(fn):
    try:
        return fn()
    except Exception as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}", "noise_floor": 0.0}


def _control(receiver, native, fixture, case):
    import torch
    kind = case["kind"]
    if kind == "non_bypass_none":
        ids = torch.tensor([fixture["inputs"]], dtype=torch.long)
        args = {"input_ids": ids, "memory": None}

        def original(**kwargs):
            out = native.stem(input_ids=kwargs["input_ids"], use_cache=False, return_dict=True)
            return {"logits": out.logits}

        detail = _pair(receiver, fixture, args, args, "exact", right_receiver=original)
        detail["bypass"] = _bypass_gate(receiver, native, fixture)
        detail["passed"] = detail["passed"] and detail["bypass"]["passed"]
        return detail
    if kind == "shuffled_role_time":
        subpairs = {}
        for change in ("role_only", "time_only", "joint"):
            rows = _rows(fixture, case["seed"])
            left, _ = _attachment(fixture, rows)
            origins = _origins(fixture)
            roles = [r["origin_type"] for r in origins][::-1]
            times = [r["event_time"] for r in origins][::-1]
            for i, rec in enumerate(origins):
                if change != "time_only":
                    rec["origin_type"] = roles[i]
                if change != "role_only":
                    rec["event_time"] = rec["ingestion_time"] = times[i]
            right, _ = _attachment(fixture, rows, origins=origins)
            subpairs[change] = _pair(receiver, fixture, left, right, "different")
        return {"passed": all(p["passed"] for p in subpairs.values()), "subpairs": subpairs,
                "noise_floor": max(p["noise_floor"] for p in subpairs.values())}
    left, right, selection = _arms(fixture, case)
    detail = _pair(receiver, fixture, left, right, case["expect"])
    detail["selection"] = selection
    if kind == "donor_swap":
        donor, _ = _attachment(fixture, -_rows(fixture, case["seed"]) + 2, reload=True)
        follow = _pair(receiver, fixture, right, donor, "equal")
        detail["donor_reload_oracle"] = follow
        detail["passed"] = detail["passed"] and follow["passed"]
    return detail


def _evaluate(receiver, native, fixture):
    before = _state_hash(native)
    cases, controls = [], []
    for case in fixture["competence_cases"]:
        def evaluate(case=case):
            left, right, selection = _arms(fixture, case)
            return {**_pair(receiver, fixture, left, right, case["expect"]), "selection": selection}
        cases.append({"id": case["id"], "kind": case["kind"], **_guarded(evaluate)})
    for case in fixture["paired_controls"]:
        if case["expect"] != "reject":
            controls.append({"id": case["id"], "kind": case["kind"], **_guarded(
                lambda case=case: _control(receiver, native, fixture, case))})
    gates = {
        "exact_none_empty_bypass_logits_and_loss": _guarded(
            lambda: _bypass_gate(receiver, native, fixture)),
        "returned_first_token_loss_gradient": _guarded(
            lambda: _gradient_gate(receiver, native, fixture, answer_slice=False)),
        "returned_answer_loss_gradient": _guarded(
            lambda: _gradient_gate(receiver, native, fixture, answer_slice=True)),
        "returned_supplied_target_loss_gradient": _guarded(
            lambda: _gradient_gate(receiver, native, fixture, answer_slice=True, supplied_targets=True)),
        "nonzero_explicit_prefix_use": _guarded(lambda: _prefix_gate(receiver, native, fixture)),
        "attached_text_attention_mask": _guarded(lambda: _mask_gate(receiver, fixture, text=True)),
        "attached_memory_mask": _guarded(lambda: _mask_gate(receiver, fixture, text=False)),
        "categorical_modality_use": _guarded(lambda: _modality_gate(receiver, fixture)),
        "bounded_time_offsets": _guarded(lambda: _time_gate(receiver, native, fixture)),
        "appended_decode_token_masks": _guarded(lambda: _decode_mask_gate(receiver, fixture)),
        "production_s0_integration": _guarded(lambda: _integration_gate(receiver, native, fixture)),
        "fresh_decode_cache_and_ownership": _guarded(lambda: _cache_gate(receiver, native, fixture)),
        "masked_prefill_decode_cache_and_ownership": _guarded(
            lambda: _cache_gate(receiver, native, fixture, masked=True))
    }
    after = _state_hash(native)
    gates["unchanged_parameters"] = {
        "passed": before == after, "before_sha256": before, "after_sha256": after}
    core = {"per_case": cases, "paired_controls": controls, "critical_gates": gates,
            "competence_pass_count": sum(c["passed"] for c in cases),
            "noise_floor": max(*(c["noise_floor"] for c in cases),
                               *(_control_noise(c) for c in controls),
                               *(_gate_noise(g) for g in gates.values()))}
    core["passed"] = _core_pass(core, fixture)
    return core


class _NullReceiver:
    """Built-in adversaries: real native bypass logits, no prefix-content use."""

    def __init__(self, native, echo):
        self.native, self.echo = native, echo

    def __call__(self, input_ids, memory=None, **kwargs):
        labels = kwargs.get("labels")
        valid_labels = labels is not None and (labels[:, 1:] != -100).any()
        out = self.native(input_ids, memory=None, attention_mask=kwargs.get("attention_mask"),
                          labels=labels if valid_labels else None,
                          use_cache=kwargs.get("use_cache", False))
        if labels is not None and not valid_labels:
            out["loss"] = self.native.connector.queries.sum() * 0
        out["received_digest"] = kwargs.get("assembly_digest") if self.echo else None
        return out

    def decode(self, token, cache, token_mask=None):
        return self.native.decode(token, cache, token_mask=token_mask)


class _BrokenReceiver:
    """Review reproductions delegate every unaffected operation to production."""

    def __init__(self, native, kind):
        self.native, self.kind = native, kind

    def __call__(self, **args):
        import torch
        args = dict(args)
        if self.kind == "zero_explicit_prefix" and args.get("prefix_states") is not None:
            args["prefix_states"] = torch.zeros_like(args["prefix_states"])
        if self.kind == "replace_supplied_labels" and args.get("labels") is not None:
            args["labels"] = torch.where(args["labels"] == -100, args["labels"], args["input_ids"])
        memory = args.get("memory")
        nonempty = memory is not None and memory.shape[1] > 0
        if nonempty:
            if self.kind == "first_row_only":
                # Keep a valid visible row even when the caller masks row zero.
                args["memory_mask"] = torch.zeros_like(args["memory_mask"])
                args["memory_mask"][:, 0] = True
            if self.kind in ("ignore_roles", "ignore_role_time"):
                args["role_ids"] = torch.zeros_like(args["role_ids"])
            if self.kind in ("ignore_time", "ignore_role_time"):
                args["time_offsets"] = torch.zeros_like(args["time_offsets"])
            if self.kind == "clip_old_time":
                args["time_offsets"] = args["time_offsets"].clamp(min=-4)
            if self.kind == "drop_attached_text_mask":
                args.pop("attention_mask", None)
            if self.kind == "ignore_memory_mask":
                args["memory_mask"] = torch.ones_like(args["memory_mask"])
            if self.kind == "ignore_types":
                args["type_ids"] = torch.zeros_like(args["type_ids"])
        out = self.native(**args)
        labels = args.get("labels")
        answer_only = nonempty and labels is not None and (labels[:, 0] == -100).all()
        if self.kind == "detached_loss" and out["loss"] is not None:
            out["loss"] = out["loss"].detach()
        elif answer_only:
            if self.kind == "detached_answer_loss":
                out["loss"] = out["loss"].detach()
            elif self.kind == "surrogate_answer_loss":
                out["loss"] = out["loss"] + 0.5 * out["loss"].detach()
            elif self.kind == "wrong_answer_loss_gradient":
                # Identical scalar value, twice the actual answer-loss gradient.
                out["loss"] = 2 * out["loss"] - out["loss"].detach()
        return out

    def decode(self, token, cache, token_mask=None):
        return self.native.decode(token, cache, token_mask=(
            None if self.kind == "drop_decode_mask" else token_mask))


def _adversary(native, name):
    if name in ("constant_receiver", "digest_echo_receiver"):
        return _NullReceiver(native, name == "digest_echo_receiver")
    if name not in ("first_row_only", "ignore_roles", "ignore_time", "ignore_role_time",
                    "detached_answer_loss", "detached_loss", "surrogate_answer_loss",
                    "wrong_answer_loss_gradient", "drop_attached_text_mask",
                    "ignore_memory_mask", "ignore_types", "drop_decode_mask", "clip_old_time",
                    "zero_explicit_prefix", "replace_supplied_labels"):
        raise ValueError("unregistered adversary")
    return _BrokenReceiver(native, name)


def _reject_controls(native, fixture, neg):
    """Reject-expectation paired controls shared by every check: a
    negative receiver must FAIL the core gates while its paired
    content-effect control still measures what it claims."""
    import torch
    controls = []
    for case in fixture["paired_controls"]:
        if case["expect"] != "reject":
            continue
        null = neg[case["kind"]]
        left, right, _ = _arms(fixture, {**case, "kind": "content"})
        adversary = _NullReceiver(native, case["kind"] == "digest_echo_receiver")
        paired = _pair(adversary, fixture, left, right, "equal")
        with torch.no_grad():
            a, b = adversary(**left), adversary(**right)
        effect = _difference(a["logits"], b["logits"])
        echo = (a["received_digest"] == left["assembly_digest"]
                and b["received_digest"] == right["assembly_digest"]
                and a["received_digest"] != b["received_digest"])
        detail = {"id": case["id"], "kind": case["kind"],
                  "passed": _adversary_rejected(null, case["kind"], fixture,
                                                _state_hash(native)) and paired["passed"],
                  "adversary_core_passed": null["passed"],
                  "adversary_competence_pass_count": null["competence_pass_count"],
                  "paired_logits_difference": effect, "paired_measurement": paired,
                  "noise_floor": paired["noise_floor"], "distinct_digests_echoed": echo}
        if case["kind"] == "digest_echo_receiver":
            detail["passed"] = detail["passed"] and echo and effect == 0
        controls.append(detail)
    return controls


class _UnavailableReceiver:
    """Actual failed executions, never successful receiver rejections."""

    def __init__(self, timeout=False):
        self.error = TimeoutError if timeout else RuntimeError

    def __call__(self, **args):
        raise self.error("instrument unavailable")

    def decode(self, token, cache, token_mask=None):
        raise self.error("instrument unavailable")


def _receipt_controls(fixture, receipt, native):
    """Resealed semantic attacks exercise the same consumer used by panel_passed."""
    candidate = copy.deepcopy(receipt)
    candidate.update(status="PASS", exit_code=0, per_case=candidate["production"]["per_case"])
    candidate["receipt_sha256"] = _seal(candidate)
    state = candidate["model_state_sha256"]
    genuine = _receipt_verdict(candidate, "random-tiny", state)
    unavailable = {
        name: _evaluate_adversary(_UnavailableReceiver(timeout=timeout), native, fixture,
                                 "first_row_only")
        for name, timeout in (("crashed_adversary", False), ("timed_out_adversary", True))}
    variants = {}
    for name in fixture["pass_rule"]["must_fail_receipts"]:
        bad = copy.deepcopy(candidate)
        cases = bad["production"]["per_case"]
        if name == "missing_selection":
            for case in cases:
                case.pop("selection", None)
        elif name == "illegal_selected_index":
            cases[6]["selection"]["left"]["selected_indices"] = [999]
        elif name == "stale_selection_digest":
            cases[6]["selection"]["left"]["digest"] = "0" * 64
        elif name == "forged_refusals":
            cases[6]["selection"]["left"]["refusals"] = []
        elif name == "stale_helper_hash":
            bad["module_sha256"]["gemma4_receiver.py"] = "0" * 64
        elif name == "stale_fixture_hash":
            bad["fixture_sha256"] = "0" * 64
        elif name == "wrong_model_state":
            bad["model_state_sha256"] = "0" * 64
        elif name == "unsealed_edit":
            bad["noise_floor"] += 1
        elif name == "missing_decode_mask_measurement":
            bad["production"]["critical_gates"]["appended_decode_token_masks"]["trials"][
                "singleton_0"].pop("cached_recompute_max_abs")
        elif name == "contradictory_time_measurement":
            bad["production"]["critical_gates"]["bounded_time_offsets"]["pairs"][
                "older_reference"]["answer_max_abs_difference"] = 1.0
        elif name == "missing_receiver_adversary":
            bad["adversaries"].pop("drop_decode_mask")
        elif name in unavailable:
            bad["adversaries"]["first_row_only"] = unavailable[name]
        elif name == "partial_exception_adversary":
            core = bad["adversaries"]["first_row_only"]
            core["critical_gates"]["attached_memory_mask"] = {
                "passed": False, "reason": "RuntimeError: instrument unavailable", "noise_floor": 0.0}
            # Keep the old explicit rejection to test semantic reconstruction.
        elif name == "missing_adversary_result":
            bad["adversaries"]["first_row_only"]["critical_gates"].pop("attached_memory_mask")
        elif name == "misattributed_rejection":
            bad["adversaries"]["first_row_only"]["adversary_name"] = "ignore_roles"
        elif name == "missing_explicit_rejection":
            bad["adversaries"]["first_row_only"].pop("rejection")
        else:
            raise ValueError("unregistered receipt adversary")
        # Synchronize duplicated records so attacks must fail semantic checks,
        # not only the duplicate-table or public unkeyed custody seal.
        bad["per_case"] = cases
        if name != "unsealed_edit":
            bad["receipt_sha256"] = _seal(bad)
        variants[name] = _receipt_verdict(bad, "random-tiny", state)
    return {"genuine_accepted": genuine, "variants": variants,
            "unavailable_evaluations": {
                name: {"rejection": core["rejection"],
                       "evidence_valid": _adversary_evidence(core, fixture, state),
                       "competence_pass_count": core["competence_pass_count"],
                       "failed_measurements": sum("reason" in d for d in (
                           core["per_case"] + core["paired_controls"]
                           + list(core["critical_gates"].values())))}
                for name, core in unavailable.items()}}


def _run(fixture, receipt):
    import torch
    native = _build(fixture)
    receipt["model_state_sha256"] = _state_hash(native)
    receipt["production"] = _evaluate(native, native, fixture)
    receipt["adversaries"] = {
        name: _evaluate_adversary(_adversary(native, name), native, fixture, name)
        for name in fixture["pass_rule"]["must_fail_receivers"]}
    live = {c["id"]: c for c in receipt["production"]["paired_controls"]}
    rejects = {c["id"]: c for c in _reject_controls(
        native, fixture, receipt["adversaries"])}
    all_controls = []
    for case in fixture["paired_controls"]:
        if case["id"] in rejects:
            all_controls.append(rejects[case["id"]])
        else:
            all_controls.append(live[case["id"]])
    receipt["paired_controls"] = all_controls
    receipt["noise_floor"] = max(
        receipt["production"]["noise_floor"], *(_control_noise(c) for c in all_controls))
    receipt["stable_module_hashes"] = (receipt["module_sha256"] == _modules()
                                      and _sha(FIXTURE_PATH) == FIXTURE_SHA256)
    receipt["receipt_adversaries"] = _receipt_controls(fixture, receipt, native)
    return (receipt["production"]["passed"] and all(c["passed"] for c in all_controls)
            and all(_adversary_rejected(core, name, fixture, receipt["model_state_sha256"])
                    for name, core in receipt["adversaries"].items())
            and receipt["stable_module_hashes"]
            and receipt["receipt_adversaries"]["genuine_accepted"]
            and all(not v for v in receipt["receipt_adversaries"]["variants"].values()))


def _instrument_failed(value):
    if isinstance(value, dict):
        return "reason" in value or any(_instrument_failed(v) for v in value.values())
    if isinstance(value, list):
        return any(_instrument_failed(v) for v in value)
    return False


def _adversary_evidence(core, fixture, expected_state=None):
    """Complete, reconstructible observations, not exceptions or positive noise."""
    try:
        verdict = _core_verdict(core, fixture, expected_state)
        unchanged = core["critical_gates"]["unchanged_parameters"]
        return (not _instrument_failed(core) and core["passed"] is verdict
                and unchanged["passed"] is True
                and (expected_state is None or
                     unchanged["before_sha256"] == unchanged["after_sha256"] == expected_state))
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        return False


def _rejection_result(core, name, fixture, expected_state=None):
    """The same attributed rejection decision for execution and receipt consumers."""
    failure = {"status": "CHECK_FAILED", "failed_observations": []}
    if not _adversary_evidence(core, fixture, expected_state):
        return failure
    if core["passed"]:
        return {"status": "NOT_REJECTED", "failed_observations": []}
    observations = {f"case:{d['id']}": d for d in core["per_case"]}
    observations.update({f"gate:{key}": d for key, d in core["critical_gates"].items()})
    witnesses = fixture["pass_rule"]["rejection_witnesses"].get(name, [])
    failed = [key for key in witnesses if observations[key]["passed"] is False]
    return {"status": "REJECTED", "failed_observations": failed} if failed else failure


def _adversary_rejected(core, name, fixture, expected_state=None):
    try:
        result = _rejection_result(core, name, fixture, expected_state)
        return (core["adversary_name"] == name and core["rejection"] == result
                and result["status"] == "REJECTED")
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        return False


def _evaluate_adversary(receiver, native, fixture, name):
    core = _evaluate(receiver, native, fixture)
    core["adversary_name"] = name
    core["rejection"] = _rejection_result(core, name, fixture, _state_hash(native))
    return core


def _run_negatives(fixture, receipt):
    """negative-controls check (Brief C addition): the must-fail
    receivers and reject-expectation controls ALONE. PASS iff every
    negative receiver FAILS the core gates with COMPLETE measurement
    evidence (a crashed evaluation never counts as a failure) and
    every reject control passes; production competence is not
    re-asserted here (tiny-production owns it) and no negative
    receiver may PASS."""
    native = _build(fixture)
    receipt["model_state_sha256"] = _state_hash(native)
    receipt["negative_scope"] = True
    neg = receipt["adversaries"] = {
        name: _evaluate_adversary(_adversary(native, name), native, fixture, name)
        for name in fixture["pass_rule"]["must_fail_receivers"]}
    controls = receipt["paired_controls"] = _reject_controls(
        native, fixture, neg)
    receipt["noise_floor"] = max(*(_control_noise(c) for c in controls), 0.0)
    receipt["negative_receivers_all_fail"] = all(
        _adversary_rejected(core, name, fixture, receipt["model_state_sha256"])
        for name, core in neg.items())
    receipt["negative_evidence_complete"] = all(
        _adversary_evidence(v, fixture, receipt["model_state_sha256"]) for v in neg.values())
    receipt["negative_controls_all_pass"] = all(
        c["passed"] for c in controls)
    receipt["stable_module_hashes"] = (receipt["module_sha256"] == _modules()
                                      and _sha(FIXTURE_PATH) == FIXTURE_SHA256)
    return (receipt["negative_receivers_all_fail"]
            and receipt["negative_evidence_complete"]
            and receipt["negative_controls_all_pass"]
            and receipt["stable_module_hashes"])


def _cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    args = parser.parse_args(argv)
    started = time.monotonic()
    receipt = {"schema": "s0-receiving-receipt/1", "check": args.check,
               "status": "FAIL", "exit_code": 1, "model_scope": "random-tiny",
               "registered_fixture_sha256": FIXTURE_SHA256, "module_sha256": _modules(),
               "s0_integrated": True, "per_case": [], "paired_controls": [], "noise_floor": 0.0,
               "tolerances": {}, "fit_authorization": "none; CPU mechanics only"}
    try:
        receipt["fixture_sha256"] = _sha(args.fixture)
        if args.check not in ("tiny-production", "negative-controls"):
            receipt.update(status="UNKNOWN_CHECK", exit_code=2)
        else:
            fixture = _fixture(args.fixture)
            receipt.update(config=fixture, tolerances=fixture["tolerances"])
            registration = {"check": args.check, "fixture_sha256": FIXTURE_SHA256,
                            "module_sha256": receipt["module_sha256"],
                            "pass_rule": fixture["pass_rule"], "tolerances": fixture["tolerances"]}
            print("PANEL_CONFIG " + json.dumps(registration, sort_keys=True), file=sys.stderr, flush=True)
            if not args.offline:
                raise ValueError("--offline is required")
            os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                              HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                              HF_DATASETS_OFFLINE="1")
            with ExitStack() as stack:
                for target in ("socket.socket.connect", "socket.socket.connect_ex",
                               "socket.create_connection"):
                    stack.enter_context(patch(target, side_effect=AssertionError("network forbidden")))
                import torch
                import transformers
                stack.enter_context(patch.object(transformers.PreTrainedModel, "from_pretrained",
                                                side_effect=AssertionError("weights forbidden")))
                torch.set_num_threads(2)
                if torch.get_num_interop_threads() != 2:
                    torch.set_num_interop_threads(2)
                receipt["dependencies"] = {"torch": torch.__version__,
                                           "transformers": transformers.__version__}
                receipt["device"], receipt["threads"] = "cpu", 2
                runner = _run if args.check == "tiny-production" else _run_negatives
                passed = runner(fixture, receipt)
            if args.check == "tiny-production":
                receipt["per_case"] = receipt["production"]["per_case"]
            receipt.update(status="PASS" if passed else "FAIL", exit_code=0 if passed else 1)
    except Exception as exc:
        receipt["reason"] = f"{type(exc).__name__}: {exc}"
    receipt["elapsed_seconds"] = time.monotonic() - started
    receipt["receipt_sha256"] = _seal(receipt)
    print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False), flush=True)
    return receipt["exit_code"]


if __name__ == "__main__":
    sys.exit(_cli())
