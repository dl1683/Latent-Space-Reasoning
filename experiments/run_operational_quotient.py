"""Round 36 minimal operational-quotient world.

This is the reusable constructive-program boundary registered in Round 36:
behavior -> learned latent transition system -> operational quotient.  The
``produce`` trains and evaluates in one non-claiming process.  The ``reduce``
command is a separate, fail-closed consumer of serialized evidence and never
receives a live model object.  The no-Torch ``fixture`` command exercises the
same reducer schema without making a scientific claim.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SCHEMA_CONFIG = "round36-operational-quotient-config-v1"
SCHEMA_MANIFEST = "round36-operational-quotient-manifest-v1"
SCHEMA_EVIDENCE = "round36-operational-quotient-evidence-v1"
SCHEMA_VERDICT = "round36-operational-quotient-verdict-v1"
REGISTRATION_ID = "round36-minimal-operational-quotient-v1"
ROUND36B_REGISTRATION_ID = "round36b-behavior-fit-ladder-v1"
PASS_STATUS = "PASS — MINIMAL OPERATIONAL QUOTIENT WORLD"
UNDERFIT_STATUS = "FAIL — BEHAVIOR UNDERFIT; QUOTIENT INELIGIBLE"
SIGNATURE_UNSUPPORTED_STATUS = (
    "FAIL — BEHAVIOR FIT; OPERATIONAL SIGNATURE UNSUPPORTED"
)
SUPPORTED_NONCONGRUENT_PREFIX = (
    "FAIL — BEHAVIOR AND SIGNATURE SUPPORTED; NON-CONGRUENT ACTIONS"
)
DIAGNOSTIC_LABEL = (
    "DIAGNOSTIC ONLY — p>0.5; cannot alter the primary 0.10/0.90 status"
)
ROUND36B_DEPTH_TRACE_INTERVAL = 1000
INTEGRITY_SCOPE = (
    "integrity relative to retained manifest; producer authenticity out of scope"
)

ACTION_NAMES = [
    "no-op",
    "toggle(1)",
    "toggle(2)",
    "toggle(3)",
    "toggle(4)",
    "swap(1,2)",
    "swap(1,3)",
    "swap(1,4)",
    "swap(2,3)",
    "swap(2,4)",
    "swap(3,4)",
]
TOGGLE_NAMES = ACTION_NAMES[1:5]
SWAP_NAMES = ACTION_NAMES[5:]
SIGNATURE_RE = re.compile(r"^[01?]{12}$")
HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(RuntimeError):
    """The frozen config or a serialized artifact violates the contract."""


class BudgetExceeded(RuntimeError):
    """The registered full producer wall was exceeded."""


class ProducerDeadline:
    def __init__(self, expires_at: float, hard_wall_seconds: int) -> None:
        self.expires_at = expires_at
        self.hard_wall_seconds = hard_wall_seconds


def _check_deadline(deadline: float | ProducerDeadline) -> None:
    expires_at = (
        deadline.expires_at if isinstance(deadline, ProducerDeadline) else deadline
    )
    hard_wall_seconds = (
        deadline.hard_wall_seconds
        if isinstance(deadline, ProducerDeadline)
        else 900
    )
    if time.monotonic() > expires_at:
        raise BudgetExceeded(
            f"registered {hard_wall_seconds}-second full producer wall exceeded"
        )


@contextlib.contextmanager
def _temporary_directory(prefix: str) -> Iterator[Path]:
    directory = Path(tempfile.gettempdir()) / f"{prefix}{uuid.uuid4().hex}"
    directory.mkdir(parents=False, exist_ok=False)
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonical finite JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_bytes(value))


def _reject_constant(token: str) -> None:
    raise ContractError(f"non-finite JSON constant {token!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
    except ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read strict JSON {path}: {exc}") from exc


def _require_keys(value: Any, expected: Iterable[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{where} must be an object")
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ContractError(f"{where} keys mismatch; missing={missing}, extra={extra}")
    return value


def _expect(actual: Any, expected: Any, where: str) -> None:
    if actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
        raise ContractError(f"{where} must equal {expected!r}; got {actual!r}")


def _expect_int(value: Any, where: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{where} must be an integer")
    if minimum is not None and value < minimum:
        raise ContractError(f"{where} must be >= {minimum}")
    return value


def _expect_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"{where} must be finite")
    return number


def _expect_sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or not HEX_RE.fullmatch(value):
        raise ContractError(f"{where} must be a lowercase SHA-256 hex digest")
    return value


def _registration_id(config: dict[str, Any]) -> str:
    return config.get("registration_id", REGISTRATION_ID)


def _is_round36b(config: dict[str, Any]) -> bool:
    return _registration_id(config) == ROUND36B_REGISTRATION_ID


def _validate_config(config: Any) -> dict[str, Any]:
    is_round36b = (
        isinstance(config, dict)
        and config.get("registration_id") == ROUND36B_REGISTRATION_ID
    )
    root_keys = [
        "schema_version",
        "name",
        "registration",
        "world",
        "actions",
        "split",
        "model",
        "training",
        "thresholds",
    ]
    if is_round36b:
        root_keys.append("registration_id")
    root = _require_keys(
        config,
        root_keys,
        "config",
    )
    _expect(root["schema_version"], SCHEMA_CONFIG, "config.schema_version")
    _expect(root["name"], "operational_quotient_v1", "config.name")
    _expect(
        root["registration"],
        "Round 36 — minimal operational-quotient world",
        "config.registration",
    )
    if is_round36b:
        _expect(
            root["registration_id"],
            ROUND36B_REGISTRATION_ID,
            "config.registration_id",
        )

    world = _require_keys(
        root["world"],
        ["bit_count", "state_count", "response_bit", "data_seed", "handle_permutation"],
        "config.world",
    )
    for key, expected in {
        "bit_count": 4,
        "state_count": 16,
        "response_bit": 1,
        "data_seed": 3601,
        "handle_permutation": "numpy-pcg64-permutation-v1",
    }.items():
        _expect(world[key], expected, f"config.world.{key}")

    actions = root["actions"]
    if not isinstance(actions, list) or len(actions) != 11:
        raise ContractError("config.actions must contain exactly 11 ordered actions")
    expected_actions = [
        {"name": "no-op", "kind": "no-op", "indices": []},
        *[
            {"name": f"toggle({i})", "kind": "toggle", "indices": [i]}
            for i in range(1, 5)
        ],
        *[
            {"name": f"swap({i},{j})", "kind": "swap", "indices": [i, j]}
            for i in range(1, 5)
            for j in range(i + 1, 5)
        ],
    ]
    _expect(actions, expected_actions, "config.actions")

    split = _require_keys(
        root["split"],
        [
            "max_word_depth",
            "identity_response_depth",
            "h2_salt",
            "h2_per_first_action",
            "h3_salt",
            "h3_per_first_action",
            "expected_training_words",
            "expected_training_rows",
            "expected_h2_words",
            "expected_h3_words",
            "expected_heldout_rows",
            "representative_prefix_depths",
            "expected_representatives_per_seed",
        ],
        "config.split",
    )
    expected_split = {
        "max_word_depth": 3,
        "identity_response_depth": 1,
        "h2_salt": "round36-h2-v1",
        "h2_per_first_action": 2,
        "h3_salt": "round36-h3-v1",
        "h3_per_first_action": 6,
        "expected_training_words": 1324,
        "expected_training_rows": 21184,
        "expected_h2_words": 74,
        "expected_h3_words": 66,
        "expected_heldout_rows": 2240,
        "representative_prefix_depths": [0, 1, 2],
        "expected_representatives_per_seed": 944,
    }
    for key, expected in expected_split.items():
        _expect(split[key], expected, f"config.split.{key}")

    model = _require_keys(
        root["model"],
        [
            "latent_dim",
            "transition_hidden_width",
            "state_table_rows",
            "shared_transition",
            "response_outputs",
        ],
        "config.model",
    )
    expected_model = {
        "latent_dim": 8,
        "state_table_rows": 16,
        "shared_transition": True,
        "response_outputs": 1,
    }
    for key, expected in expected_model.items():
        _expect(model[key], expected, f"config.model.{key}")

    training = _require_keys(
        root["training"],
        [
            "model_seeds",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "betas",
            "epsilon",
            "batch_size",
            "optimizer_steps_per_seed",
            "loss",
            "device",
            "threads",
            "deterministic_algorithms",
            "evaluation_batch_size",
            "target_cpu_minutes",
            "hard_wall_seconds",
        ],
        "config.training",
    )
    expected_training = {
        "model_seeds": [11, 23, 37, 53, 71],
        "optimizer": "AdamW",
        "weight_decay": 0.00001,
        "betas": [0.9, 0.999],
        "epsilon": 1e-08,
        "batch_size": 512,
        "loss": "binary_cross_entropy_with_logits",
        "device": "cpu",
        "threads": 1,
        "deterministic_algorithms": True,
        "evaluation_batch_size": 4096,
        "target_cpu_minutes": [3, 8],
    }
    for key, expected in expected_training.items():
        _expect(training[key], expected, f"config.training.{key}")

    registered_knobs = (
        training["optimizer_steps_per_seed"],
        training["learning_rate"],
        model["transition_hidden_width"],
        training["hard_wall_seconds"],
    )
    if is_round36b:
        allowed_round36b_knobs = {
            (16000, 0.003, 32, 480),
            (64000, 0.003, 32, 1200),
            (64000, 0.001, 32, 1200),
            (64000, 0.003, 64, 1800),
        }
        if registered_knobs not in allowed_round36b_knobs:
            raise ContractError(
                "Round 36b config knobs do not match a registered S16/S64/LR64/W64 cell"
            )
    else:
        _expect(
            registered_knobs,
            (4000, 0.003, 32, 900),
            "Round 36 v1 registered knobs",
        )

    thresholds = _require_keys(
        root["thresholds"],
        [
            "signature_low",
            "signature_high",
            "quotient_availability_fraction",
            "quotient_well_definedness_fraction",
            "toggle_involution_fraction",
            "swap_toggle_table_fraction",
            "heldout_depth2_fraction",
            "heldout_depth3_fraction",
            "interchangeability_fraction",
            "cross_seed_action_table_fraction",
            "required_seed_fraction",
        ],
        "config.thresholds",
    )
    expected_thresholds = {
        "signature_low": 0.10,
        "signature_high": 0.90,
        "quotient_availability_fraction": 1.0,
        "quotient_well_definedness_fraction": 1.0,
        "toggle_involution_fraction": 1.0,
        "swap_toggle_table_fraction": 1.0,
        "heldout_depth2_fraction": 1.0,
        "heldout_depth3_fraction": 1.0,
        "interchangeability_fraction": 1.0,
        "cross_seed_action_table_fraction": 1.0,
        "required_seed_fraction": 1.0,
    }
    for key, expected in expected_thresholds.items():
        _expect(thresholds[key], expected, f"config.thresholds.{key}")
    return root


def _module_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _states() -> list[tuple[int, int, int, int]]:
    return list(itertools.product((0, 1), repeat=4))


def _state_index(state: Sequence[int]) -> int:
    return sum(int(bit) << (3 - position) for position, bit in enumerate(state))


def _apply_action(state: Sequence[int], action: str) -> tuple[int, int, int, int]:
    result = list(state)
    if action == "no-op":
        pass
    elif action.startswith("toggle("):
        index = int(action[7:-1]) - 1
        result[index] = 1 - result[index]
    elif action.startswith("swap("):
        left, right = (int(part) - 1 for part in action[5:-1].split(","))
        result[left], result[right] = result[right], result[left]
    else:
        raise ContractError(f"unknown action {action!r}")
    return tuple(result)  # type: ignore[return-value]


def _apply_word(state: Sequence[int], word: Sequence[str]) -> tuple[int, int, int, int]:
    result = tuple(state)
    for action in word:
        result = _apply_action(result, action)
    return result  # type: ignore[return-value]


def _oracle_signature(state: Sequence[int]) -> str:
    probes: list[tuple[str, ...]] = [tuple()] + [(action,) for action in ACTION_NAMES]
    return "".join(str(_apply_word(state, word)[0]) for word in probes)


def _word_spelling(word: Sequence[str]) -> str:
    return ">".join(word)


def _spelling_word(spelling: str) -> tuple[str, ...]:
    if spelling == "":
        return tuple()
    parts = tuple(spelling.split(">"))
    if any(part not in ACTION_NAMES for part in parts):
        raise ContractError(f"invalid canonical word spelling {spelling!r}")
    return parts


def _salted_word_hash(salt: str, word: Sequence[str]) -> str:
    return _sha256_bytes(f"{salt}|{_word_spelling(word)}".encode("utf-8"))


def _plain_word_hash(word: Sequence[str]) -> str:
    return _sha256_bytes(_word_spelling(word).encode("utf-8"))


def _word_universe(config: dict[str, Any]) -> dict[str, list[tuple[str, ...]]]:
    by_depth = {
        depth: list(itertools.product(ACTION_NAMES, repeat=depth))
        for depth in range(4)
    }
    forced_h2: set[tuple[str, ...]] = {
        (f"toggle({index})", f"toggle({index})") for index in range(1, 5)
    }
    for swap in SWAP_NAMES:
        for toggle in TOGGLE_NAMES:
            forced_h2.add((swap, toggle))
            forced_h2.add((toggle, swap))

    remaining = [word for word in by_depth[2] if word not in forced_h2]
    selected_h2 = set(forced_h2)
    for first in ACTION_NAMES:
        stratum = [word for word in remaining if word[0] == first]
        stratum.sort(key=lambda word: (_salted_word_hash(config["split"]["h2_salt"], word), word))
        selected_h2.update(stratum[: config["split"]["h2_per_first_action"]])

    selected_h3: set[tuple[str, ...]] = set()
    for first in ACTION_NAMES:
        stratum = [word for word in by_depth[3] if word[0] == first]
        stratum.sort(key=lambda word: (_salted_word_hash(config["split"]["h3_salt"], word), word))
        selected_h3.update(stratum[: config["split"]["h3_per_first_action"]])

    h2 = [word for word in by_depth[2] if word in selected_h2]
    h3 = [word for word in by_depth[3] if word in selected_h3]
    training = (
        by_depth[0]
        + by_depth[1]
        + [word for word in by_depth[2] if word not in selected_h2]
        + [word for word in by_depth[3] if word not in selected_h3]
    )
    heldout = h2 + h3
    expected = config["split"]
    if len(h2) != expected["expected_h2_words"]:
        raise ContractError(f"derived H_2 count {len(h2)} != registered count")
    if len(h3) != expected["expected_h3_words"]:
        raise ContractError(f"derived H_3 count {len(h3)} != registered count")
    if len(training) != expected["expected_training_words"]:
        raise ContractError(f"derived training count {len(training)} != registered count")
    if len(heldout) * 16 != expected["expected_heldout_rows"]:
        raise ContractError("derived held-out row count is not registered 2,240")
    return {"training": training, "h2": h2, "h3": h3, "heldout": heldout}


def _word_pack(config: dict[str, Any]) -> dict[str, Any]:
    universe = _word_universe(config)
    result: dict[str, Any] = {}
    for name in ("training", "h2", "h3"):
        words = universe[name]
        spellings = [_word_spelling(word) for word in words]
        if name == "h2":
            hashes = [_salted_word_hash(config["split"]["h2_salt"], word) for word in words]
        elif name == "h3":
            hashes = [_salted_word_hash(config["split"]["h3_salt"], word) for word in words]
        else:
            hashes = [_plain_word_hash(word) for word in words]
        result[name] = spellings
        result[f"{name}_hashes"] = hashes
        result[f"{name}_list_sha256"] = _sha256_bytes(_canonical_bytes(spellings))
    return result


def _handle_to_state(config: dict[str, Any]) -> list[int]:
    import numpy as np

    generator = np.random.Generator(np.random.PCG64(config["world"]["data_seed"]))
    return [int(value) for value in generator.permutation(16)]


def _representative_layout(config: dict[str, Any]) -> dict[str, Any]:
    universe = _word_universe(config)
    prefixes = [tuple()] + [word for word in universe["training"] if len(word) in (1, 2)]
    if len(prefixes) != 59:
        raise ContractError(f"representative prefix count {len(prefixes)} != 59")
    handle_to_state = _handle_to_state(config)
    states = _states()
    ids: list[str] = []
    prefix_spellings: list[str] = []
    oracle_states: list[int] = []
    for handle, state_index in enumerate(handle_to_state):
        for prefix in prefixes:
            spelling = _word_spelling(prefix)
            ids.append(f"h{handle:02d}|{spelling}")
            prefix_spellings.append(spelling)
            oracle_states.append(_state_index(_apply_word(states[state_index], prefix)))
    if len(ids) != config["split"]["expected_representatives_per_seed"]:
        raise ContractError("representative count does not match the registration")
    return {
        "prefixes": prefixes,
        "ids": ids,
        "prefix_spellings": prefix_spellings,
        "oracle_states": oracle_states,
        "handle_to_state": handle_to_state,
    }


def _world_evidence(config: dict[str, Any]) -> dict[str, Any]:
    states = _states()
    oracle_signatures = [_oracle_signature(state) for state in states]
    if len(set(oracle_signatures)) != 16:
        raise ContractError("depth-1 oracle signatures are not 16 distinct classes")
    oracle_action_table = [
        [_oracle_signature(_apply_action(state, action)) for action in ACTION_NAMES]
        for state in states
    ]
    return {
        "states": [list(state) for state in states],
        "handle_to_state": _handle_to_state(config),
        "oracle_signatures": oracle_signatures,
        "oracle_action_table": oracle_action_table,
    }


def _signature_from_probabilities(probabilities: Sequence[Any], low: float, high: float) -> str:
    if len(probabilities) != 12:
        raise ContractError("a depth-1 signature requires exactly 12 probabilities")
    cells: list[str] = []
    for value in probabilities:
        number = _expect_number(value, "response probability")
        if number < 0.0 or number > 1.0:
            raise ContractError("response probabilities must lie in [0,1]")
        if number <= low:
            cells.append("0")
        elif number >= high:
            cells.append("1")
        else:
            cells.append("?")
    return "".join(cells)


def _supported(signature: str) -> bool:
    return "?" not in signature


def _support_flags_for_seed(seed: dict[str, Any]) -> dict[str, Any]:
    return {
        "representatives": [_supported(signature) for signature in seed["representative_signatures"]],
        "primitive_successors": [
            [_supported(signature) for signature in row]
            for row in seed["primitive_successor_signatures"]
        ],
        "toggle_twice": [
            [_supported(signature) for signature in row]
            for row in seed["toggle_twice_signatures"]
        ],
        "law_endpoints": [
            [
                _supported(row["swap_after_toggle"]),
                _supported(row["toggle_after_swap"]),
                _supported(row["conjugate_after_swap"]),
            ]
            for row in seed["law_rows"]
        ],
        "heldout_endpoints": [
            [_supported(signature) for signature in row]
            for row in seed["heldout_endpoint_signatures"]
        ],
        "representative_continuations": [
            [_supported(signature) for signature in row]
            for row in seed["representative_continuation_signatures"]
        ],
        "canonical_continuations": [
            [_supported(signature) for signature in row]
            for row in seed["canonical_continuation_signatures"]
        ],
    }


def _make_model(torch: Any, config: dict[str, Any]) -> Any:
    nn = torch.nn

    class OperationalQuotientModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            latent_dim = config["model"]["latent_dim"]
            width = config["model"]["transition_hidden_width"]
            self.encoder = nn.Embedding(16, latent_dim)
            self.w_z = nn.Linear(latent_dim, width, bias=False)
            self.action_embedding = nn.Embedding(11, width)
            self.b1 = nn.Parameter(torch.zeros(width))
            self.w2 = nn.Linear(width, latent_dim, bias=True)
            self.response = nn.Linear(latent_dim, 1)

        def transition(self, latent: Any, action_ids: Any) -> Any:
            hidden = torch.tanh(
                self.w_z(latent) + self.action_embedding(action_ids) + self.b1
            )
            return latent + self.w2(hidden)

        def run_word(self, handles: Any, action_ids: Any, lengths: Any) -> Any:
            latent = self.encoder(handles)
            for position in range(action_ids.shape[1]):
                proposed = self.transition(latent, action_ids[:, position])
                latent = torch.where((lengths > position).unsqueeze(1), proposed, latent)
            return self.response(latent).squeeze(1)

    return OperationalQuotientModel()


def _assert_cpu_model(model: Any) -> list[str]:
    devices = {str(parameter.device) for parameter in model.parameters()}
    if devices != {"cpu"}:
        raise ContractError(f"CUDA/non-CPU parameter use is forbidden; devices={sorted(devices)}")
    return sorted(devices)


def _import_producer_dependencies(config: dict[str, Any]) -> tuple[Any, Any]:
    inherited_cuda_visibility = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if inherited_cuda_visibility != "":
        raise ContractError(
            "Round 36 requires CUDA_VISIBLE_DEVICES to be absent or exactly empty"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch

    torch.set_num_threads(config["training"]["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(config["training"]["deterministic_algorithms"])
    if config["training"]["device"] != "cpu":
        raise ContractError("the registered producer is CPU-only")
    if torch.cuda.is_initialized():
        raise ContractError("CUDA was initialized in a CPU-only Round 36 producer")
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise ContractError("Round 36 requires exactly one intra-op and one inter-op CPU thread")
    if not torch.are_deterministic_algorithms_enabled():
        raise ContractError("Round 36 requires deterministic algorithms")
    return np, torch


def _producer_cpu_settings(torch: Any, parameter_devices: Sequence[str]) -> dict[str, Any]:
    settings = {
        "device": "cpu",
        "parameter_devices": sorted(set(parameter_devices)),
        "intra_op_threads": int(torch.get_num_threads()),
        "inter_op_threads": int(torch.get_num_interop_threads()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }
    if settings != {
        "device": "cpu",
        "parameter_devices": ["cpu"],
        "intra_op_threads": 1,
        "inter_op_threads": 1,
        "deterministic_algorithms": True,
        "cuda_visible_devices": "",
        "cuda_initialized": False,
    }:
        raise ContractError(f"producer CPU settings violate the frozen contract: {settings!r}")
    return settings


def _training_arrays(np: Any, config: dict[str, Any]) -> dict[str, Any]:
    universe = _word_universe(config)
    handle_to_state = _handle_to_state(config)
    states = _states()
    action_ids = {name: index for index, name in enumerate(ACTION_NAMES)}
    handles: list[int] = []
    words: list[list[int]] = []
    lengths: list[int] = []
    targets: list[float] = []
    for word in universe["training"]:
        padded = [action_ids[action] for action in word] + [0] * (3 - len(word))
        for handle, state_index in enumerate(handle_to_state):
            handles.append(handle)
            words.append(padded)
            lengths.append(len(word))
            targets.append(float(_apply_word(states[state_index], word)[0]))
    if len(handles) != config["split"]["expected_training_rows"]:
        raise ContractError("training row count is not 21,184")
    return {
        "handles": np.asarray(handles, dtype=np.int64),
        "words": np.asarray(words, dtype=np.int64),
        "lengths": np.asarray(lengths, dtype=np.int64),
        "targets": np.asarray(targets, dtype=np.float32),
    }


def _behavior_arrays(
    np: Any,
    config: dict[str, Any],
    words_to_score: Sequence[Sequence[str]],
) -> dict[str, Any]:
    handle_to_state = _handle_to_state(config)
    states = _states()
    action_ids = {name: index for index, name in enumerate(ACTION_NAMES)}
    handles: list[int] = []
    words: list[list[int]] = []
    lengths: list[int] = []
    targets: list[float] = []
    for word in words_to_score:
        padded = [action_ids[action] for action in word] + [0] * (3 - len(word))
        for handle, state_index in enumerate(handle_to_state):
            handles.append(handle)
            words.append(padded)
            lengths.append(len(word))
            targets.append(float(_apply_word(states[state_index], word)[0]))
    return {
        "handles": np.asarray(handles, dtype=np.int64),
        "words": np.asarray(words, dtype=np.int64),
        "lengths": np.asarray(lengths, dtype=np.int64),
        "targets": np.asarray(targets, dtype=np.float32),
    }


def _behavior_target_bits(
    config: dict[str, Any],
    words_to_score: Sequence[Sequence[str]],
) -> list[int]:
    """Reconstruct the frozen word-major, handle-minor behavioral targets."""
    handle_to_state = _handle_to_state(config)
    states = _states()
    return [
        int(_apply_word(states[state_index], word)[0])
        for word in words_to_score
        for state_index in handle_to_state
    ]


def _sigmoid_from_logit(logit: float) -> float:
    if logit >= 0.0:
        exp_negative = math.exp(-logit)
        return 1.0 / (1.0 + exp_negative)
    exp_positive = math.exp(logit)
    return exp_positive / (1.0 + exp_positive)


def _behavior_metric_from_logits(
    config: dict[str, Any],
    words_to_score: Sequence[Sequence[str]],
    raw_logits: Sequence[Any],
    *,
    include_row_logits: bool,
) -> dict[str, Any]:
    targets = _behavior_target_bits(config, words_to_score)
    if not isinstance(raw_logits, list) or len(raw_logits) != len(targets):
        raise ContractError(
            "behavior row_logits must be a list in frozen word-major, "
            f"handle-minor order with exactly {len(targets)} rows"
        )
    logits = [
        _expect_number(value, f"behavior row_logits[{index}]")
        for index, value in enumerate(raw_logits)
    ]
    low = config["thresholds"]["signature_low"]
    high = config["thresholds"]["signature_high"]
    correct = 0
    supported = 0
    correct_and_supported = 0
    losses: list[float] = []
    for logit, target in zip(logits, targets):
        is_correct = (logit > 0.0 and target == 1) or (
            logit < 0.0 and target == 0
        )
        probability = _sigmoid_from_logit(logit)
        is_supported = probability <= low or probability >= high
        correct += int(is_correct)
        supported += int(is_supported)
        correct_and_supported += int(is_correct and is_supported)
        losses.append(
            max(logit, 0.0)
            - logit * target
            + math.log1p(math.exp(-abs(logit)))
        )
    total = len(logits)
    try:
        full_dataset_bce = math.fsum(losses) / total
    except OverflowError as exc:
        raise ContractError("behavior row_logits yield non-finite BCE") from exc
    if not math.isfinite(full_dataset_bce):
        raise ContractError("behavior row_logits yield non-finite BCE")
    metric = {
        "full_dataset_bce": full_dataset_bce,
        "correct_responses": correct,
        "total_responses": total,
        "response_accuracy": correct / total,
        "supported_responses": supported,
        "support_fraction": supported / total,
        "correct_and_supported_responses": correct_and_supported,
        "correct_and_supported_fraction": correct_and_supported / total,
    }
    if include_row_logits:
        metric["row_logits"] = logits
    return metric


def _score_behavior_rows(
    np: Any,
    torch: Any,
    model: Any,
    config: dict[str, Any],
    words_to_score: Sequence[Sequence[str]],
    deadline: float | ProducerDeadline,
    *,
    serialize_row_logits: bool = False,
) -> dict[str, Any]:
    arrays = _behavior_arrays(np, config, words_to_score)
    handles = torch.from_numpy(arrays["handles"])
    words = torch.from_numpy(arrays["words"])
    lengths = torch.from_numpy(arrays["lengths"])
    targets = torch.from_numpy(arrays["targets"])
    batch_size = config["training"]["evaluation_batch_size"]
    correct = 0
    supported = 0
    correct_and_supported = 0
    loss_sum = 0.0
    row_logits: list[float] = []
    with torch.no_grad():
        for start in range(0, handles.shape[0], batch_size):
            _check_deadline(deadline)
            stop = min(handles.shape[0], start + batch_size)
            batch_targets = targets[start:stop]
            logits = model.run_word(
                handles[start:stop], words[start:stop], lengths[start:stop]
            )
            if serialize_row_logits:
                row_logits.extend(float(value) for value in logits.cpu().tolist())
            loss_sum += float(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, batch_targets, reduction="sum"
                ).cpu()
            )
            correct_mask = ((logits > 0.0) & (batch_targets == 1.0)) | (
                (logits < 0.0) & (batch_targets == 0.0)
            )
            probabilities = torch.sigmoid(logits)
            supported_mask = (
                (probabilities <= config["thresholds"]["signature_low"])
                | (probabilities >= config["thresholds"]["signature_high"])
            )
            correct += int(correct_mask.sum().cpu())
            supported += int(supported_mask.sum().cpu())
            correct_and_supported += int((correct_mask & supported_mask).sum().cpu())
    total = int(handles.shape[0])
    if serialize_row_logits:
        return _behavior_metric_from_logits(
            config,
            words_to_score,
            row_logits,
            include_row_logits=True,
        )
    return {
        "full_dataset_bce": loss_sum / total,
        "correct_responses": correct,
        "total_responses": total,
        "response_accuracy": correct / total,
        "supported_responses": supported,
        "support_fraction": supported / total,
        "correct_and_supported_responses": correct_and_supported,
        "correct_and_supported_fraction": correct_and_supported / total,
    }


def _behavior_fit_record(
    np: Any,
    torch: Any,
    model: Any,
    loss_trace: Any,
    training_by_word_depth: list[dict[str, Any]],
    config: dict[str, Any],
    deadline: float,
) -> dict[str, Any]:
    universe = _word_universe(config)
    training = _score_behavior_rows(
        np,
        torch,
        model,
        config,
        universe["training"],
        deadline,
        serialize_row_logits=True,
    )
    h2 = _score_behavior_rows(
        np,
        torch,
        model,
        config,
        universe["h2"],
        deadline,
        serialize_row_logits=True,
    )
    h3 = _score_behavior_rows(
        np,
        torch,
        model,
        config,
        universe["h3"],
        deadline,
        serialize_row_logits=True,
    )
    heldout = _behavior_metric_from_logits(
        config,
        universe["heldout"],
        h2["row_logits"] + h3["row_logits"],
        include_row_logits=False,
    )
    return {
        "loss_trace": [float(value) for value in loss_trace],
        "training_by_word_depth": training_by_word_depth,
        "training": training,
        "h2": h2,
        "h3": h3,
        "heldout": heldout,
    }


def _training_depth_checkpoint(
    np: Any,
    torch: Any,
    model: Any,
    config: dict[str, Any],
    optimizer_step: int,
    deadline: float | ProducerDeadline,
) -> dict[str, Any]:
    training_words = _word_universe(config)["training"]
    return {
        "optimizer_step": optimizer_step,
        "by_word_depth": {
            str(depth): _score_behavior_rows(
                np,
                torch,
                model,
                config,
                [word for word in training_words if len(word) == depth],
                deadline,
            )
            for depth in range(4)
        },
    }


def _prepare_empty_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ContractError(f"refusing to overwrite non-empty output directory {path}")
    path.mkdir(parents=True, exist_ok=True)


def _loss_trace_sha(np: Any, trace: Sequence[float]) -> str:
    return _sha256_bytes(np.asarray(trace, dtype="<f8").tobytes(order="C"))


def _dependency_versions(np: Any | None = None, torch: Any | None = None) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": "not-loaded" if np is None else str(np.__version__),
        "torch": "not-loaded" if torch is None else str(torch.__version__),
    }


def _platform_info() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _append_ledger(entry: dict[str, Any]) -> None:
    ledger = Path(__file__).resolve().with_name("ledger.jsonl")
    with ledger.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_canonical_bytes(entry).decode("utf-8"))


def _ledger_entry(
    event_id: str,
    purpose: str,
    config_path: Path,
    command: str,
    metrics: dict[str, Any],
    artifacts: list[str],
    status: str,
    data_ref: str | None,
    notes: str,
) -> dict[str, Any]:
    return {
        "timestamp": _utc_now(),
        "id": event_id,
        "purpose": purpose,
        "git_commit": _git_commit(),
        "config_path": str(config_path),
        "data_ref": data_ref,
        "command": command,
        "metrics": metrics,
        "artifacts": artifacts,
        "notes": notes,
        "status": status,
    }


def _train(
    config: dict[str, Any],
    output_dir: Path,
    np: Any,
    torch: Any,
    *,
    producer_started: float,
    deadline: float,
) -> dict[str, Any]:
    _check_deadline(deadline)
    arrays = _training_arrays(np, config)
    handles = torch.from_numpy(arrays["handles"])
    words = torch.from_numpy(arrays["words"])
    lengths = torch.from_numpy(arrays["lengths"])
    targets = torch.from_numpy(arrays["targets"])
    saved: dict[str, Any] = {}
    seed_summaries: list[dict[str, Any]] = []
    parameter_devices: set[str] = set()

    for seed in config["training"]["model_seeds"]:
        _check_deadline(deadline)
        torch.manual_seed(seed)
        model = _make_model(torch, config).cpu()
        parameter_devices.update(_assert_cpu_model(model))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            betas=tuple(config["training"]["betas"]),
            eps=config["training"]["epsilon"],
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        trace: list[float] = []
        depth_trace: list[dict[str, Any]] = []
        if _is_round36b(config):
            depth_trace.append(
                _training_depth_checkpoint(
                    np, torch, model, config, 0, deadline
                )
            )
        for step in range(config["training"]["optimizer_steps_per_seed"]):
            if step % 25 == 0:
                _check_deadline(deadline)
            indices = torch.randint(
                0,
                handles.shape[0],
                (config["training"]["batch_size"],),
                generator=generator,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model.run_word(handles[indices], words[indices], lengths[indices])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets[indices]
            )
            loss.backward()
            optimizer.step()
            trace.append(float(loss.detach().cpu()))
            completed_steps = step + 1
            if (
                _is_round36b(config)
                and completed_steps % ROUND36B_DEPTH_TRACE_INTERVAL == 0
            ):
                depth_trace.append(
                    _training_depth_checkpoint(
                        np,
                        torch,
                        model,
                        config,
                        completed_steps,
                        deadline,
                    )
                )
        parameter_devices.update(_assert_cpu_model(model))
        for name, tensor in model.state_dict().items():
            key = f"seed_{seed}__{name.replace('.', '_')}"
            saved[key] = tensor.detach().cpu().numpy()
        saved[f"seed_{seed}__loss_trace"] = np.asarray(trace, dtype=np.float64)
        seed_summary = {
            "seed": seed,
            "steps": len(trace),
            "final_loss": trace[-1],
            "loss_trace_sha256": _loss_trace_sha(np, trace),
        }
        if _is_round36b(config):
            seed_summary["training_by_word_depth"] = depth_trace
        seed_summaries.append(seed_summary)

    weights_path = output_dir / "weights.npz"
    np.savez_compressed(weights_path, **saved)
    _check_deadline(deadline)
    return {
        "wall_seconds": time.monotonic() - producer_started,
        "weights_sha256": _sha256_file(weights_path),
        "parameter_devices": sorted(parameter_devices),
        "seeds": seed_summaries,
    }


def _state_dict_keys(seed: int) -> dict[str, str]:
    names = [
        "encoder.weight",
        "w_z.weight",
        "action_embedding.weight",
        "b1",
        "w2.weight",
        "w2.bias",
        "response.weight",
        "response.bias",
    ]
    return {name: f"seed_{seed}__{name.replace('.', '_')}" for name in names}


def _load_model_from_npz(torch: Any, model: Any, archive: Any, seed: int) -> None:
    state = {}
    for name, key in _state_dict_keys(seed).items():
        if key not in archive.files:
            raise ContractError(f"weights archive missing {key}")
        state[name] = torch.from_numpy(archive[key])
    model.load_state_dict(state, strict=True)
    model.eval()
    _assert_cpu_model(model)


def _transition_word(torch: Any, model: Any, latent: Any, word: Sequence[str]) -> Any:
    result = latent
    for action in word:
        action_id = ACTION_NAMES.index(action)
        ids = torch.full((result.shape[0],), action_id, dtype=torch.long)
        result = model.transition(result, ids)
    return result


def _signature_batches(
    torch: Any,
    model: Any,
    latent: Any,
    config: dict[str, Any],
    include_probabilities: bool,
    deadline: float | None = None,
) -> tuple[list[str], list[list[float]] | None]:
    low = config["thresholds"]["signature_low"]
    high = config["thresholds"]["signature_high"]
    batch_size = config["training"]["evaluation_batch_size"]
    signatures: list[str] = []
    probability_rows: list[list[float]] | None = [] if include_probabilities else None
    with torch.no_grad():
        for start in range(0, latent.shape[0], batch_size):
            if deadline is not None:
                _check_deadline(deadline)
            batch = latent[start : start + batch_size]
            base = torch.sigmoid(model.response(batch)).squeeze(1)
            expanded = batch[:, None, :].expand(-1, 11, -1).reshape(-1, batch.shape[1])
            action_ids = torch.arange(11, dtype=torch.long).repeat(batch.shape[0])
            moved = model.transition(expanded, action_ids)
            moved_probs = torch.sigmoid(model.response(moved)).reshape(batch.shape[0], 11)
            probabilities = torch.cat([base[:, None], moved_probs], dim=1).cpu().tolist()
            for row in probabilities:
                signatures.append(_signature_from_probabilities(row, low, high))
                if probability_rows is not None:
                    probability_rows.append([float(value) for value in row])
    return signatures, probability_rows


def _roll_grid_signatures(
    torch: Any,
    model: Any,
    initial: Any,
    words: list[tuple[str, ...]],
    config: dict[str, Any],
    deadline: float | None = None,
    probability_sink: list[list[list[float]]] | None = None,
) -> tuple[list[list[str]], list[list[int | None]]]:
    batch_size = config["training"]["evaluation_batch_size"]
    total = initial.shape[0] * len(words)
    flat_signatures: list[str] = []
    flat_probabilities: list[list[float]] = []
    action_ids = {name: index for index, name in enumerate(ACTION_NAMES)}
    word_ids = [[action_ids[action] for action in word] for word in words]
    with torch.no_grad():
        for offset in range(0, total, batch_size):
            if deadline is not None:
                _check_deadline(deadline)
            stop = min(total, offset + batch_size)
            flat = torch.arange(offset, stop, dtype=torch.long)
            source_indices = torch.div(flat, len(words), rounding_mode="floor")
            word_indices = (flat % len(words)).tolist()
            latent = initial[source_indices].clone()
            for position in range(3):
                ids = []
                active = []
                for word_index in word_indices:
                    row = word_ids[word_index]
                    active.append(position < len(row))
                    ids.append(row[position] if position < len(row) else 0)
                proposed = model.transition(latent, torch.tensor(ids, dtype=torch.long))
                latent = torch.where(torch.tensor(active).unsqueeze(1), proposed, latent)
            batch_signatures, batch_probabilities = _signature_batches(
                torch,
                model,
                latent,
                config,
                include_probabilities=probability_sink is not None,
                deadline=deadline,
            )
            flat_signatures.extend(batch_signatures)
            if batch_probabilities is not None:
                flat_probabilities.extend(batch_probabilities)
    rows = [
        flat_signatures[index * len(words) : (index + 1) * len(words)]
        for index in range(initial.shape[0])
    ]
    bits = [[int(sig[0]) if _supported(sig) else None for sig in row] for row in rows]
    if probability_sink is not None:
        probability_sink.extend(
            [
                flat_probabilities[index * len(words) : (index + 1) * len(words)]
                for index in range(initial.shape[0])
            ]
        )
    return rows, bits


def _law_rows_from_model(
    torch: Any,
    model: Any,
    encoder_latent: Any,
    config: dict[str, Any],
    deadline: float | None = None,
    probability_sink: list[list[list[float]]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    starts = []
    words_a: list[tuple[str, ...]] = []
    words_b: list[tuple[str, ...]] = []
    words_c: list[tuple[str, ...]] = []
    for handle in range(16):
        for swap in SWAP_NAMES:
            i, j = (int(part) for part in swap[5:-1].split(","))
            for toggle in TOGGLE_NAMES:
                k = int(toggle[7:-1])
                mapped = j if k == i else i if k == j else k
                starts.append(handle)
                words_a.append((toggle, swap))
                words_b.append((swap, toggle))
                words_c.append((swap, f"toggle({mapped})"))
                rows.append({
                    "handle": handle,
                    "swap": swap,
                    "toggle": toggle,
                    "swap_after_toggle": "",
                    "toggle_after_swap": "",
                    "conjugate_after_swap": "",
                    "registered_cell_pass": False,
                })
    latent = encoder_latent[torch.tensor(starts, dtype=torch.long)]
    endpoints = []
    endpoint_probabilities: list[list[list[float]]] = []
    for words in (words_a, words_b, words_c):
        pieces = []
        with torch.no_grad():
            for index, word in enumerate(words):
                if deadline is not None and index % 16 == 0:
                    _check_deadline(deadline)
                pieces.append(_transition_word(torch, model, latent[index : index + 1], word))
        combined = torch.cat(pieces, dim=0)
        signatures, probabilities = _signature_batches(
            torch,
            model,
            combined,
            config,
            probability_sink is not None,
            deadline=deadline,
        )
        endpoints.append(signatures)
        if probabilities is not None:
            endpoint_probabilities.append(probabilities)
    handle_to_state = _handle_to_state(config)
    states = _states()
    for index, row in enumerate(rows):
        row["swap_after_toggle"] = endpoints[0][index]
        row["toggle_after_swap"] = endpoints[1][index]
        row["conjugate_after_swap"] = endpoints[2][index]
        swap = row["swap"]
        toggle = row["toggle"]
        i, j = (int(part) for part in swap[5:-1].split(","))
        k = int(toggle[7:-1])
        state = states[handle_to_state[row["handle"]]]
        mapped = j if k == i else i if k == j else k
        expected = (
            _oracle_signature(_apply_word(state, (toggle, swap))),
            _oracle_signature(_apply_word(state, (swap, toggle))),
            _oracle_signature(_apply_word(state, (swap, f"toggle({mapped})"))),
        )
        relation = (
            row["swap_after_toggle"] == row["toggle_after_swap"]
            if k not in (i, j)
            else row["swap_after_toggle"] != row["toggle_after_swap"]
        )
        row["registered_cell_pass"] = (
            all(_supported(row[key]) for key in (
                "swap_after_toggle", "toggle_after_swap", "conjugate_after_swap"
            ))
            and (
                row["swap_after_toggle"],
                row["toggle_after_swap"],
                row["conjugate_after_swap"],
            ) == expected
            and relation
            and row["swap_after_toggle"] == row["conjugate_after_swap"]
        )
    if probability_sink is not None:
        probability_sink.extend(
            [
                [endpoint_probabilities[column][row] for column in range(3)]
                for row in range(len(rows))
            ]
        )
    return rows


def _evaluate_seed(
    np: Any,
    torch: Any,
    model: Any,
    archive: Any,
    seed: int,
    config: dict[str, Any],
    deadline: float,
    training_summary: dict[str, Any],
) -> dict[str, Any]:
    _check_deadline(deadline)
    layout = _representative_layout(config)
    universe = _word_universe(config)
    world = _world_evidence(config)
    is_round36b = _is_round36b(config)
    with torch.no_grad():
        handles = torch.arange(16, dtype=torch.long)
        encoder_latent = model.encoder(handles)
        representative_parts = []
        for handle in range(16):
            _check_deadline(deadline)
            base = encoder_latent[handle : handle + 1]
            for prefix in layout["prefixes"]:
                representative_parts.append(_transition_word(torch, model, base, prefix))
        representatives = torch.cat(representative_parts, dim=0)
    signatures, probabilities = _signature_batches(
        torch,
        model,
        representatives,
        config,
        include_probabilities=True,
        deadline=deadline,
    )
    assert probabilities is not None

    with torch.no_grad():
        expanded = representatives[:, None, :].expand(-1, 11, -1).reshape(-1, 8)
        action_ids = torch.arange(11, dtype=torch.long).repeat(representatives.shape[0])
        primitive_latent = model.transition(expanded, action_ids)
    primitive_flat, primitive_probability_flat = _signature_batches(
        torch,
        model,
        primitive_latent,
        config,
        is_round36b,
        deadline=deadline,
    )
    primitive = [primitive_flat[index * 11 : (index + 1) * 11] for index in range(944)]
    primitive_probabilities = (
        [
            primitive_probability_flat[index * 11 : (index + 1) * 11]
            for index in range(944)
        ]
        if primitive_probability_flat is not None
        else []
    )

    double_parts = []
    with torch.no_grad():
        for toggle_id in range(1, 5):
            ids = torch.full((representatives.shape[0],), toggle_id, dtype=torch.long)
            once = model.transition(representatives, ids)
            double_parts.append(model.transition(once, ids))
    double_latent = torch.stack(double_parts, dim=1).reshape(-1, 8)
    double_flat, double_probability_flat = _signature_batches(
        torch,
        model,
        double_latent,
        config,
        is_round36b,
        deadline=deadline,
    )
    toggle_twice = [double_flat[index * 4 : (index + 1) * 4] for index in range(944)]
    toggle_twice_probabilities = (
        [
            double_probability_flat[index * 4 : (index + 1) * 4]
            for index in range(944)
        ]
        if double_probability_flat is not None
        else []
    )

    encoder_signatures = [signatures[handle * 59] for handle in range(16)]
    signature_to_handle = {
        signature: handle
        for handle, signature in enumerate(encoder_signatures)
        if _supported(signature)
    }
    canonical_handles = [signature_to_handle.get(signature, -1) for signature in signatures]
    canonical_latent = torch.cat(
        [
            encoder_latent[handle : handle + 1] if handle >= 0 else encoder_latent[0:1]
            for handle in canonical_handles
        ],
        dim=0,
    )

    heldout_probabilities: list[list[list[float]]] = []
    heldout_signatures, _ = _roll_grid_signatures(
        torch,
        model,
        encoder_latent,
        universe["heldout"],
        config,
        deadline,
        heldout_probabilities if is_round36b else None,
    )
    representative_continuation_probabilities: list[list[list[float]]] = []
    representative_endpoints, representative_bits = _roll_grid_signatures(
        torch,
        model,
        representatives,
        universe["heldout"],
        config,
        deadline,
        representative_continuation_probabilities if is_round36b else None,
    )
    canonical_endpoints, canonical_bits = _roll_grid_signatures(
        torch, model, canonical_latent, universe["heldout"], config, deadline
    )
    for index, handle in enumerate(canonical_handles):
        if handle < 0:
            canonical_endpoints[index] = ["?" * 12] * len(universe["heldout"])
            canonical_bits[index] = [None] * len(universe["heldout"])

    recovered_table: list[list[str]] = []
    for oracle_signature in world["oracle_signatures"]:
        handle = signature_to_handle.get(oracle_signature, -1)
        recovered_table.append(primitive[handle * 59] if handle >= 0 else ["?" * 12] * 11)

    loss_key = f"seed_{seed}__loss_trace"
    if loss_key not in archive.files:
        raise ContractError(f"weights archive missing {loss_key}")
    loss_trace = archive[loss_key]
    if loss_trace.shape != (config["training"]["optimizer_steps_per_seed"],):
        raise ContractError(f"loss trace for seed {seed} has wrong shape")
    law_probabilities: list[list[list[float]]] = []
    law_rows = _law_rows_from_model(
        torch,
        model,
        encoder_latent,
        config,
        deadline,
        law_probabilities if is_round36b else None,
    )
    record = {
        "seed": seed,
        "loss_trace_sha256": _sha256_bytes(np.asarray(loss_trace, dtype="<f8").tobytes()),
        "representative_ids": layout["ids"],
        "representative_prefixes": layout["prefix_spellings"],
        "representative_oracle_states": layout["oracle_states"],
        "representative_response_probabilities": probabilities,
        "representative_signatures": signatures,
        "primitive_successor_signatures": primitive,
        "toggle_twice_signatures": toggle_twice,
        "law_rows": law_rows,
        "heldout_endpoint_signatures": heldout_signatures,
        "canonical_handles": canonical_handles,
        "representative_continuation_signatures": representative_endpoints,
        "canonical_continuation_signatures": canonical_endpoints,
        "representative_terminal_bits": representative_bits,
        "canonical_terminal_bits": canonical_bits,
        "recovered_action_table": recovered_table,
    }
    record["support_flags"] = _support_flags_for_seed(record)
    if is_round36b:
        record["diagnostic_response_probabilities"] = {
            "primitive_successors": primitive_probabilities,
            "toggle_twice": toggle_twice_probabilities,
            "law_endpoints": law_probabilities,
            "heldout_endpoints": heldout_probabilities,
            "representative_continuations": (
                representative_continuation_probabilities
            ),
        }
        record["behavior_fit"] = _behavior_fit_record(
            np,
            torch,
            model,
            loss_trace,
            training_summary["training_by_word_depth"],
            config,
            deadline,
        )
    return record


def _learned_construction() -> dict[str, Any]:
    return {
        "kind": "learned",
        "latent_points": [],
        "action_affine_maps": [],
        "response_readout": [],
    }


def _expected_counts() -> dict[str, int]:
    return {
        "training_words": 1324,
        "training_rows": 21184,
        "h2_words": 74,
        "h3_words": 66,
        "heldout_rows": 2240,
        "representatives_per_seed": 944,
        "class_action_cells": 176,
        "swap_toggle_cells": 384,
        "heldout_depth2_cells": 1184,
        "heldout_depth3_cells": 1056,
        "interchangeability_cells_per_seed": 132160,
    }


def _evidence(
    run_dir: Path,
    config_bytes: bytes,
    config: dict[str, Any],
    np: Any,
    torch: Any,
    *,
    started_at: str,
    producer_started: float,
    deadline: float,
    training: dict[str, Any],
) -> dict[str, Any]:
    _check_deadline(deadline)
    config_sha = _sha256_bytes(config_bytes)
    copied_config = run_dir / "config.json"
    weights_path = run_dir / "weights.npz"
    for required in (copied_config, weights_path):
        if not required.is_file():
            raise ContractError(f"missing in-process producer artifact {required}")
    if copied_config.read_bytes() != config_bytes:
        raise ContractError("provided config bytes differ from the producer config copy")
    if training["weights_sha256"] != _sha256_file(weights_path):
        raise ContractError("weights hash mismatch")

    parameter_devices = set(training["parameter_devices"])
    with np.load(weights_path, allow_pickle=False) as archive:
        expected_archive_keys = {f"seed_{seed}__loss_trace" for seed in config["training"]["model_seeds"]}
        for seed in config["training"]["model_seeds"]:
            expected_archive_keys.update(_state_dict_keys(seed).values())
        if set(archive.files) != expected_archive_keys:
            raise ContractError("weights archive has missing or extra arrays")
        seed_evidence = []
        for seed_index, seed in enumerate(config["training"]["model_seeds"]):
            _check_deadline(deadline)
            model = _make_model(torch, config).cpu()
            _load_model_from_npz(torch, model, archive, seed)
            parameter_devices.update(_assert_cpu_model(model))
            seed_record = _evaluate_seed(
                np,
                torch,
                model,
                archive,
                seed,
                config,
                deadline,
                training["seeds"][seed_index],
            )
            if seed_record["loss_trace_sha256"] != training["seeds"][seed_index]["loss_trace_sha256"]:
                raise ContractError(f"loss trace checksum mismatch for seed {seed}")
            seed_evidence.append(seed_record)
    evidence = {
        "schema_version": SCHEMA_EVIDENCE,
        "registration_id": _registration_id(config),
        "producer_kind": "learned",
        "config_sha256": config_sha,
        "code_sha256": _module_sha256(),
        "weights_sha256": _sha256_file(weights_path),
        "action_order": ACTION_NAMES,
        "data_seed": config["world"]["data_seed"],
        "model_seeds": config["training"]["model_seeds"],
        "expected_counts": _expected_counts(),
        "world": _world_evidence(config),
        "words": _word_pack(config),
        "construction": _learned_construction(),
        "seeds": seed_evidence,
        "producer_counts": {},
    }
    _attach_producer_counts(evidence, config, deadline)
    _check_deadline(deadline)
    evidence_path = run_dir / "evidence.json"
    _write_json(evidence_path, evidence)
    _check_deadline(deadline)
    total_seconds = time.monotonic() - producer_started
    train_seconds = float(training["wall_seconds"])
    evidence_seconds = total_seconds - train_seconds
    manifest = {
        "schema_version": SCHEMA_MANIFEST,
        "registration_id": _registration_id(config),
        "producer_status": "complete_nonclaiming",
        "producer_kind": "learned",
        "integrity_scope": INTEGRITY_SCOPE,
        "command": " ".join(sys.argv),
        "config_sha256": config_sha,
        "config_copy_sha256": _sha256_file(copied_config),
        "code_sha256": _module_sha256(),
        "weights_sha256": _sha256_file(weights_path),
        "evidence_sha256": _sha256_file(evidence_path),
        "git_commit": _git_commit(),
        "started_at": started_at,
        "ended_at": _utc_now(),
        "wall_seconds": {
            "train": train_seconds,
            "evidence": evidence_seconds,
            "total": total_seconds,
        },
        "platform": _platform_info(),
        "dependencies": _dependency_versions(np, torch),
        "cpu_settings": _producer_cpu_settings(torch, sorted(parameter_devices)),
        "data_seed": config["world"]["data_seed"],
        "model_seeds": config["training"]["model_seeds"],
        "action_order": ACTION_NAMES,
        "word_list_hashes": {
            key: value
            for key, value in evidence["words"].items()
            if key.endswith("_list_sha256")
        },
        "expected_counts": _expected_counts(),
    }
    _write_json(run_dir / "manifest.json", manifest)
    _check_deadline(deadline)
    return manifest


def _produce(config_path: Path, output_dir: Path) -> dict[str, Any]:
    started_at = _utc_now()
    producer_started = time.monotonic()
    config_bytes = config_path.read_bytes()
    config = _validate_config(_read_json(config_path))
    deadline = ProducerDeadline(
        producer_started + config["training"]["hard_wall_seconds"],
        config["training"]["hard_wall_seconds"],
    )
    _prepare_empty_directory(output_dir)
    (output_dir / "config.json").write_bytes(config_bytes)
    _check_deadline(deadline)
    np, torch = _import_producer_dependencies(config)
    training = _train(
        config,
        output_dir,
        np,
        torch,
        producer_started=producer_started,
        deadline=deadline,
    )
    manifest = _evidence(
        output_dir,
        config_bytes,
        config,
        np,
        torch,
        started_at=started_at,
        producer_started=producer_started,
        deadline=deadline,
        training=training,
    )
    _append_ledger(
        _ledger_entry(
            "round36_operational_quotient_produce",
            "Round 36 non-claiming CPU training and evidence producer",
            config_path,
            " ".join(sys.argv),
            {"wall_seconds": manifest["wall_seconds"], "seeds": config["training"]["model_seeds"]},
            [str(output_dir / "manifest.json"), str(output_dir / "evidence.json")],
            "producer_complete_nonclaiming",
            str(output_dir),
            f"No scientific verdict is emitted; {INTEGRITY_SCOPE}.",
        )
    )
    return manifest


def _fixture_affine_construction(config: dict[str, Any]) -> dict[str, Any]:
    states = _states()
    handle_to_state = _handle_to_state(config)
    latent_points = [list(states[state_index]) + [0.0] * 4 for state_index in handle_to_state]
    maps = []
    for action in ACTION_NAMES:
        matrix = [[float(row == column) for column in range(8)] for row in range(8)]
        bias = [0.0] * 8
        if action.startswith("toggle("):
            index = int(action[7:-1]) - 1
            matrix[index][index] = -1.0
            bias[index] = 1.0
        elif action.startswith("swap("):
            left, right = (int(part) - 1 for part in action[5:-1].split(","))
            matrix[left] = [0.0] * 8
            matrix[right] = [0.0] * 8
            matrix[left][right] = 1.0
            matrix[right][left] = 1.0
        maps.append({"name": action, "matrix": matrix, "bias": bias})
    return {
        "kind": "fixture",
        "latent_points": latent_points,
        "action_affine_maps": maps,
        "response_readout": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }


def _fixture_law_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    states = _states()
    handle_to_state = _handle_to_state(config)
    rows = []
    for handle, state_index in enumerate(handle_to_state):
        state = states[state_index]
        for swap in SWAP_NAMES:
            i, j = (int(part) for part in swap[5:-1].split(","))
            for toggle in TOGGLE_NAMES:
                k = int(toggle[7:-1])
                mapped = j if k == i else i if k == j else k
                rows.append(
                    {
                        "handle": handle,
                        "swap": swap,
                        "toggle": toggle,
                        "swap_after_toggle": _oracle_signature(_apply_word(state, (toggle, swap))),
                        "toggle_after_swap": _oracle_signature(_apply_word(state, (swap, toggle))),
                        "conjugate_after_swap": _oracle_signature(
                            _apply_word(state, (swap, f"toggle({mapped})"))
                        ),
                        "registered_cell_pass": True,
                    }
                )
    return rows


def _fixture_behavior_fit(config: dict[str, Any]) -> dict[str, Any]:
    def exact(total: int) -> dict[str, Any]:
        return {
            "full_dataset_bce": 0.0,
            "correct_responses": total,
            "total_responses": total,
            "response_accuracy": 1.0,
            "supported_responses": total,
            "support_fraction": 1.0,
            "correct_and_supported_responses": total,
            "correct_and_supported_fraction": 1.0,
        }

    universe = _word_universe(config)
    split_metrics = {
        split_name: _behavior_metric_from_logits(
            config,
            universe[split_name],
            [
                1000.0 if target == 1 else -1000.0
                for target in _behavior_target_bits(config, universe[split_name])
            ],
            include_row_logits=True,
        )
        for split_name in ("training", "h2", "h3")
    }
    heldout = _behavior_metric_from_logits(
        config,
        universe["heldout"],
        split_metrics["h2"]["row_logits"]
        + split_metrics["h3"]["row_logits"],
        include_row_logits=False,
    )
    return {
        "loss_trace": [],
        "training_by_word_depth": [
            {
                "optimizer_step": 0,
                "by_word_depth": {
                    "0": exact(16),
                    "1": exact(176),
                    "2": exact(752),
                    "3": exact(20240),
                },
            }
        ],
        "training": split_metrics["training"],
        "h2": split_metrics["h2"],
        "h3": split_metrics["h3"],
        "heldout": heldout,
    }


def _replace_behavior_summary(
    destination: dict[str, Any], source: dict[str, Any]
) -> None:
    for key in (
        "full_dataset_bce",
        "correct_responses",
        "total_responses",
        "response_accuracy",
        "supported_responses",
        "support_fraction",
        "correct_and_supported_responses",
        "correct_and_supported_fraction",
    ):
        destination[key] = source[key]


def _refresh_fixture_behavior_summaries_from_rows(
    seed: dict[str, Any], config: dict[str, Any]
) -> None:
    behavior = seed["behavior_fit"]
    universe = _word_universe(config)
    recomputed: dict[str, dict[str, Any]] = {}
    for split_name in ("training", "h2", "h3"):
        recomputed[split_name] = _behavior_metric_from_logits(
            config,
            universe[split_name],
            behavior[split_name]["row_logits"],
            include_row_logits=False,
        )
        _replace_behavior_summary(behavior[split_name], recomputed[split_name])
    recomputed["heldout"] = _behavior_metric_from_logits(
        config,
        universe["heldout"],
        behavior["h2"]["row_logits"] + behavior["h3"]["row_logits"],
        include_row_logits=False,
    )
    _replace_behavior_summary(behavior["heldout"], recomputed["heldout"])

    cursor = 0
    final_depths = behavior["training_by_word_depth"][-1]["by_word_depth"]
    for depth in range(4):
        depth_words = [word for word in universe["training"] if len(word) == depth]
        depth_rows = len(depth_words) * 16
        depth_logits = behavior["training"]["row_logits"][
            cursor : cursor + depth_rows
        ]
        cursor += depth_rows
        depth_metric = _behavior_metric_from_logits(
            config,
            depth_words,
            depth_logits,
            include_row_logits=False,
        )
        _replace_behavior_summary(final_depths[str(depth)], depth_metric)


def _fixture_seed(seed: int, config: dict[str, Any]) -> dict[str, Any]:
    layout = _representative_layout(config)
    universe = _word_universe(config)
    states = _states()
    world = _world_evidence(config)
    signatures = [world["oracle_signatures"][index] for index in layout["oracle_states"]]
    probabilities = [[float(cell) for cell in signature] for signature in signatures]
    primitive = [
        [
            _oracle_signature(_apply_action(states[state_index], action))
            for action in ACTION_NAMES
        ]
        for state_index in layout["oracle_states"]
    ]
    toggle_twice = [[signatures[index]] * 4 for index in range(len(signatures))]
    heldout = [
        [
            _oracle_signature(_apply_word(states[state_index], word))
            for word in universe["heldout"]
        ]
        for state_index in _handle_to_state(config)
    ]
    canonical_handles = []
    state_to_handle = {state_index: handle for handle, state_index in enumerate(_handle_to_state(config))}
    for state_index in layout["oracle_states"]:
        canonical_handles.append(state_to_handle[state_index])
    continuation = [
        [
            _oracle_signature(_apply_word(states[state_index], word))
            for word in universe["heldout"]
        ]
        for state_index in layout["oracle_states"]
    ]
    terminal_bits = [[int(signature[0]) for signature in row] for row in continuation]
    law_rows = _fixture_law_rows(config)
    record = {
        "seed": seed,
        "loss_trace_sha256": _sha256_bytes(f"fixture-loss-trace|{seed}".encode("utf-8")),
        "representative_ids": layout["ids"],
        "representative_prefixes": layout["prefix_spellings"],
        "representative_oracle_states": layout["oracle_states"],
        "representative_response_probabilities": probabilities,
        "representative_signatures": signatures,
        "primitive_successor_signatures": primitive,
        "toggle_twice_signatures": toggle_twice,
        "law_rows": law_rows,
        "heldout_endpoint_signatures": heldout,
        "canonical_handles": canonical_handles,
        "representative_continuation_signatures": continuation,
        "canonical_continuation_signatures": [list(row) for row in continuation],
        "representative_terminal_bits": terminal_bits,
        "canonical_terminal_bits": [list(row) for row in terminal_bits],
        "recovered_action_table": world["oracle_action_table"],
    }
    record["support_flags"] = _support_flags_for_seed(record)
    if _is_round36b(config):
        record["diagnostic_response_probabilities"] = {
            "primitive_successors": [
                [[float(cell) for cell in signature] for signature in row]
                for row in primitive
            ],
            "toggle_twice": [
                [[float(cell) for cell in signature] for signature in row]
                for row in toggle_twice
            ],
            "law_endpoints": [
                [
                    [float(cell) for cell in row[key]]
                    for key in (
                        "swap_after_toggle",
                        "toggle_after_swap",
                        "conjugate_after_swap",
                    )
                ]
                for row in law_rows
            ],
            "heldout_endpoints": [
                [[float(cell) for cell in signature] for signature in row]
                for row in heldout
            ],
            "representative_continuations": [
                [[float(cell) for cell in signature] for signature in row]
                for row in continuation
            ],
        }
        record["behavior_fit"] = _fixture_behavior_fit(config)
    return record


def _fixture_evidence(config: dict[str, Any], config_sha: str) -> dict[str, Any]:
    construction = _fixture_affine_construction(config)
    weights_sha = _sha256_bytes(_canonical_bytes(construction))
    evidence = {
        "schema_version": SCHEMA_EVIDENCE,
        "registration_id": _registration_id(config),
        "producer_kind": "fixture",
        "config_sha256": config_sha,
        "code_sha256": _module_sha256(),
        "weights_sha256": weights_sha,
        "action_order": ACTION_NAMES,
        "data_seed": config["world"]["data_seed"],
        "model_seeds": config["training"]["model_seeds"],
        "expected_counts": _expected_counts(),
        "world": _world_evidence(config),
        "words": _word_pack(config),
        "construction": construction,
        "seeds": [_fixture_seed(seed, config) for seed in config["training"]["model_seeds"]],
        "producer_counts": {},
    }
    _attach_producer_counts(evidence, config)
    return evidence


def _fixture_manifest(config: dict[str, Any], config_sha: str, evidence: dict[str, Any]) -> dict[str, Any]:
    now = _utc_now()
    word_hashes = {
        key: value for key, value in evidence["words"].items() if key.endswith("_list_sha256")
    }
    return {
        "schema_version": SCHEMA_MANIFEST,
        "registration_id": _registration_id(config),
        "producer_status": "complete_nonclaiming",
        "producer_kind": "fixture",
        "integrity_scope": INTEGRITY_SCOPE,
        "command": " ".join(sys.argv),
        "config_sha256": config_sha,
        "config_copy_sha256": config_sha,
        "code_sha256": evidence["code_sha256"],
        "weights_sha256": evidence["weights_sha256"],
        "evidence_sha256": "0" * 64,
        "git_commit": _git_commit(),
        "started_at": now,
        "ended_at": now,
        "wall_seconds": {"train": 0.0, "evidence": 0.0, "total": 0.0},
        "platform": _platform_info(),
        "dependencies": _dependency_versions(),
        "cpu_settings": {
            "device": "cpu",
            "parameter_devices": [],
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "deterministic_algorithms": True,
            "cuda_visible_devices": "",
            "cuda_initialized": False,
        },
        "data_seed": config["world"]["data_seed"],
        "model_seeds": config["training"]["model_seeds"],
        "action_order": ACTION_NAMES,
        "word_list_hashes": word_hashes,
        "expected_counts": _expected_counts(),
    }


def _write_fixture_artifact(
    directory: Path, config_path: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    _prepare_empty_directory(directory)
    config_bytes = config_path.read_bytes()
    (directory / "config.json").write_bytes(config_bytes)
    evidence = _fixture_evidence(config, _sha256_bytes(config_bytes))
    _write_json(directory / "evidence.json", evidence)
    manifest = _fixture_manifest(config, _sha256_bytes(config_bytes), evidence)
    manifest["evidence_sha256"] = _sha256_file(directory / "evidence.json")
    _write_json(directory / "manifest.json", manifest)
    return evidence, manifest


def _validate_manifest(value: Any, config: dict[str, Any]) -> dict[str, Any]:
    manifest = _require_keys(
        value,
        [
            "schema_version",
            "registration_id",
            "producer_status",
            "producer_kind",
            "integrity_scope",
            "command",
            "config_sha256",
            "config_copy_sha256",
            "code_sha256",
            "weights_sha256",
            "evidence_sha256",
            "git_commit",
            "started_at",
            "ended_at",
            "wall_seconds",
            "platform",
            "dependencies",
            "cpu_settings",
            "data_seed",
            "model_seeds",
            "action_order",
            "word_list_hashes",
            "expected_counts",
        ],
        "manifest",
    )
    _expect(manifest["schema_version"], SCHEMA_MANIFEST, "manifest.schema_version")
    _expect(
        manifest["registration_id"],
        _registration_id(config),
        "manifest.registration_id",
    )
    _expect(manifest["producer_status"], "complete_nonclaiming", "manifest.producer_status")
    if manifest["producer_kind"] not in ("learned", "fixture"):
        raise ContractError("manifest.producer_kind must be learned or fixture")
    _expect(manifest["integrity_scope"], INTEGRITY_SCOPE, "manifest.integrity_scope")
    if not isinstance(manifest["command"], str):
        raise ContractError("manifest.command must be a string")
    for key in (
        "config_sha256",
        "config_copy_sha256",
        "code_sha256",
        "weights_sha256",
        "evidence_sha256",
    ):
        _expect_sha(manifest[key], f"manifest.{key}")
    wall = _require_keys(manifest["wall_seconds"], ["train", "evidence", "total"], "manifest.wall_seconds")
    train_wall = _expect_number(wall["train"], "manifest.wall_seconds.train")
    evidence_wall = _expect_number(wall["evidence"], "manifest.wall_seconds.evidence")
    total_wall = _expect_number(wall["total"], "manifest.wall_seconds.total")
    if train_wall < 0.0 or evidence_wall < 0.0 or total_wall < 0.0:
        raise ContractError("manifest wall components must be nonnegative")
    if abs((train_wall + evidence_wall) - total_wall) > 1e-6:
        raise ContractError("manifest total wall does not equal train + evidence wall")
    if total_wall > config["training"]["hard_wall_seconds"]:
        raise ContractError(
            f"registered {config['training']['hard_wall_seconds']}-second hard wall was exceeded"
        )
    _expect(manifest["data_seed"], 3601, "manifest.data_seed")
    _expect(manifest["model_seeds"], [11, 23, 37, 53, 71], "manifest.model_seeds")
    _expect(manifest["action_order"], ACTION_NAMES, "manifest.action_order")
    _expect(manifest["expected_counts"], _expected_counts(), "manifest.expected_counts")
    _require_keys(manifest["platform"], ["platform", "machine", "processor"], "manifest.platform")
    _require_keys(manifest["dependencies"], ["python", "numpy", "torch"], "manifest.dependencies")
    cpu = _require_keys(
        manifest["cpu_settings"],
        [
            "device",
            "parameter_devices",
            "intra_op_threads",
            "inter_op_threads",
            "deterministic_algorithms",
            "cuda_visible_devices",
            "cuda_initialized",
        ],
        "manifest.cpu_settings",
    )
    _expect(cpu["device"], "cpu", "manifest.cpu_settings.device")
    expected_parameter_devices = ["cpu"] if manifest["producer_kind"] == "learned" else []
    _expect(
        cpu["parameter_devices"],
        expected_parameter_devices,
        "manifest.cpu_settings.parameter_devices",
    )
    _expect(cpu["intra_op_threads"], 1, "manifest.cpu_settings.intra_op_threads")
    _expect(cpu["inter_op_threads"], 1, "manifest.cpu_settings.inter_op_threads")
    _expect(cpu["deterministic_algorithms"], True, "manifest.cpu_settings.deterministic_algorithms")
    _expect(cpu["cuda_visible_devices"], "", "manifest.cpu_settings.cuda_visible_devices")
    _expect(cpu["cuda_initialized"], False, "manifest.cpu_settings.cuda_initialized")
    return manifest


def _validate_signature(value: Any, where: str) -> str:
    if not isinstance(value, str) or not SIGNATURE_RE.fullmatch(value):
        raise ContractError(f"{where} must be a 12-cell 0/1/? signature")
    return value


def _validate_matrix(
    value: Any,
    rows: int,
    columns: int,
    where: str,
    validator: Any,
) -> list[list[Any]]:
    if not isinstance(value, list) or len(value) != rows:
        raise ContractError(f"{where} must have exactly {rows} rows")
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != columns:
            raise ContractError(f"{where}[{row_index}] must have exactly {columns} cells")
        for column_index, cell in enumerate(row):
            validator(cell, f"{where}[{row_index}][{column_index}]")
    return value


def _validate_probability(value: Any, where: str) -> None:
    number = _expect_number(value, where)
    if not 0.0 <= number <= 1.0:
        raise ContractError(f"{where} must lie in [0,1]")


def _validate_probability_signature_table(
    value: Any,
    expected_signatures: list[list[str]],
    where: str,
    low: float,
    high: float,
) -> list[list[list[float]]]:
    if not isinstance(value, list) or len(value) != len(expected_signatures):
        raise ContractError(f"{where} row count does not match its signature table")
    for row_index, (row, signature_row) in enumerate(
        zip(value, expected_signatures)
    ):
        if not isinstance(row, list) or len(row) != len(signature_row):
            raise ContractError(
                f"{where}[{row_index}] column count does not match its signature table"
            )
        for column_index, (probabilities, signature) in enumerate(
            zip(row, signature_row)
        ):
            cell_where = f"{where}[{row_index}][{column_index}]"
            if not isinstance(probabilities, list) or len(probabilities) != 12:
                raise ContractError(f"{cell_where} must contain 12 probabilities")
            for component_index, probability in enumerate(probabilities):
                _validate_probability(
                    probability, f"{cell_where}[{component_index}]"
                )
            _expect(
                _signature_from_probabilities(probabilities, low, high),
                signature,
                f"{cell_where} primary-signature replay",
            )
    return value


def _validate_bit(value: Any, where: str) -> None:
    if value is not None and (isinstance(value, bool) or value not in (0, 1)):
        raise ContractError(f"{where} must be 0, 1, or null")


def _validate_words(value: Any, config: dict[str, Any]) -> dict[str, Any]:
    words = _require_keys(
        value,
        [
            "training",
            "training_hashes",
            "training_list_sha256",
            "h2",
            "h2_hashes",
            "h2_list_sha256",
            "h3",
            "h3_hashes",
            "h3_list_sha256",
        ],
        "evidence.words",
    )
    expected = _word_pack(config)
    _expect(words, expected, "evidence.words")
    return words


def _validate_world(value: Any, config: dict[str, Any]) -> dict[str, Any]:
    world = _require_keys(
        value,
        ["states", "handle_to_state", "oracle_signatures", "oracle_action_table"],
        "evidence.world",
    )
    _expect(world, _world_evidence(config), "evidence.world")
    return world


def _validate_construction(
    value: Any, producer_kind: str, config: dict[str, Any]
) -> dict[str, Any]:
    construction = _require_keys(
        value,
        ["kind", "latent_points", "action_affine_maps", "response_readout"],
        "evidence.construction",
    )
    _expect(construction["kind"], producer_kind, "evidence.construction.kind")
    if producer_kind == "learned":
        _expect(construction["latent_points"], [], "learned construction.latent_points")
        _expect(construction["action_affine_maps"], [], "learned construction.action_affine_maps")
        _expect(construction["response_readout"], [], "learned construction.response_readout")
    else:
        _expect(
            construction,
            _fixture_affine_construction(config),
            "evidence.construction fixture realization",
        )
        if not isinstance(construction["latent_points"], list) or len(construction["latent_points"]) != 16:
            raise ContractError("fixture construction must contain 16 latent points")
        for row in construction["latent_points"]:
            if not isinstance(row, list) or len(row) != 8:
                raise ContractError("fixture latent points must be 8-dimensional")
            for value in row:
                _expect_number(value, "fixture latent coordinate")
        maps = construction["action_affine_maps"]
        if not isinstance(maps, list) or len(maps) != 11:
            raise ContractError("fixture must contain 11 affine action maps")
        for index, raw in enumerate(maps):
            row = _require_keys(raw, ["name", "matrix", "bias"], f"fixture action map {index}")
            _expect(row["name"], ACTION_NAMES[index], f"fixture action map {index}.name")
            _validate_matrix(row["matrix"], 8, 8, f"fixture action map {index}.matrix", _expect_number)
            if not isinstance(row["bias"], list) or len(row["bias"]) != 8:
                raise ContractError("fixture action bias must be 8-dimensional")
            for value in row["bias"]:
                _expect_number(value, "fixture action bias")
        if not isinstance(construction["response_readout"], list) or len(construction["response_readout"]) != 9:
            raise ContractError("fixture response readout must contain eight weights and a bias")
        for value in construction["response_readout"]:
            _expect_number(value, "fixture response readout")
    return construction


def _validate_behavior_metric(
    value: Any,
    where: str,
    expected_total: int,
    *,
    require_row_logits: bool = False,
) -> dict[str, Any]:
    expected_keys = [
        "full_dataset_bce",
        "correct_responses",
        "total_responses",
        "response_accuracy",
        "supported_responses",
        "support_fraction",
        "correct_and_supported_responses",
        "correct_and_supported_fraction",
    ]
    if require_row_logits:
        expected_keys.append("row_logits")
    metric = _require_keys(
        value,
        expected_keys,
        where,
    )
    total = _expect_int(metric["total_responses"], f"{where}.total_responses", minimum=1)
    _expect(total, expected_total, f"{where}.total_responses")
    correct = _expect_int(metric["correct_responses"], f"{where}.correct_responses", minimum=0)
    supported = _expect_int(
        metric["supported_responses"], f"{where}.supported_responses", minimum=0
    )
    correct_supported = _expect_int(
        metric["correct_and_supported_responses"],
        f"{where}.correct_and_supported_responses",
        minimum=0,
    )
    if (
        correct > total
        or supported > total
        or correct_supported > correct
        or correct_supported > supported
    ):
        raise ContractError(f"{where} response counts are inconsistent")
    bce = _expect_number(metric["full_dataset_bce"], f"{where}.full_dataset_bce")
    if bce < 0.0:
        raise ContractError(f"{where}.full_dataset_bce must be nonnegative")
    if require_row_logits:
        row_logits = metric["row_logits"]
        if not isinstance(row_logits, list) or len(row_logits) != expected_total:
            raise ContractError(
                f"{where}.row_logits must contain exactly {expected_total} rows"
            )
        for index, logit in enumerate(row_logits):
            _expect_number(logit, f"{where}.row_logits[{index}]")
    for key, expected in (
        ("response_accuracy", correct / total),
        ("support_fraction", supported / total),
        ("correct_and_supported_fraction", correct_supported / total),
    ):
        _expect(
            _expect_number(metric[key], f"{where}.{key}"),
            expected,
            f"{where}.{key} replay",
        )
    return metric


def _expect_behavior_summary_replay(
    supplied: dict[str, Any],
    recomputed: dict[str, Any],
    where: str,
) -> None:
    for key in (
        "full_dataset_bce",
        "correct_responses",
        "total_responses",
        "response_accuracy",
        "supported_responses",
        "support_fraction",
        "correct_and_supported_responses",
        "correct_and_supported_fraction",
    ):
        _expect(supplied[key], recomputed[key], f"{where}.{key} row replay")


def _validate_behavior_fit(
    value: Any,
    where: str,
    config: dict[str, Any],
    producer_kind: str,
    loss_trace_sha256: str,
) -> dict[str, Any]:
    behavior = _require_keys(
        value,
        [
            "loss_trace",
            "training_by_word_depth",
            "training",
            "h2",
            "h3",
            "heldout",
        ],
        where,
    )
    trace = behavior["loss_trace"]
    if not isinstance(trace, list):
        raise ContractError(f"{where}.loss_trace must be a list")
    expected_trace_length = (
        config["training"]["optimizer_steps_per_seed"]
        if producer_kind == "learned"
        else 0
    )
    if len(trace) != expected_trace_length:
        raise ContractError(
            f"{where}.loss_trace must contain exactly {expected_trace_length} values"
        )
    trace_values = [
        _expect_number(item, f"{where}.loss_trace[{index}]")
        for index, item in enumerate(trace)
    ]
    if producer_kind == "learned":
        trace_bytes = b"".join(struct.pack("<d", item) for item in trace_values)
        _expect(
            _sha256_bytes(trace_bytes),
            loss_trace_sha256,
            f"{where}.loss_trace checksum replay",
        )

    expected_totals = {
        "training": config["split"]["expected_training_rows"],
        "h2": config["split"]["expected_h2_words"] * 16,
        "h3": config["split"]["expected_h3_words"] * 16,
        "heldout": config["split"]["expected_heldout_rows"],
    }
    for split_name, expected_total in expected_totals.items():
        _validate_behavior_metric(
            behavior[split_name],
            f"{where}.{split_name}",
            expected_total,
            require_row_logits=split_name != "heldout",
        )

    universe = _word_universe(config)
    recomputed: dict[str, dict[str, Any]] = {}
    for split_name in ("training", "h2", "h3"):
        recomputed[split_name] = _behavior_metric_from_logits(
            config,
            universe[split_name],
            behavior[split_name]["row_logits"],
            include_row_logits=False,
        )
    recomputed["heldout"] = _behavior_metric_from_logits(
        config,
        universe["heldout"],
        behavior["h2"]["row_logits"] + behavior["h3"]["row_logits"],
        include_row_logits=False,
    )
    for split_name in ("training", "h2", "h3", "heldout"):
        _expect_behavior_summary_replay(
            behavior[split_name],
            recomputed[split_name],
            f"{where}.{split_name}",
        )

    depth_trace = behavior["training_by_word_depth"]
    if not isinstance(depth_trace, list):
        raise ContractError(f"{where}.training_by_word_depth must be a list")
    expected_steps = (
        list(
            range(
                0,
                config["training"]["optimizer_steps_per_seed"] + 1,
                ROUND36B_DEPTH_TRACE_INTERVAL,
            )
        )
        if producer_kind == "learned"
        else [0]
    )
    if len(depth_trace) != len(expected_steps):
        raise ContractError(
            f"{where}.training_by_word_depth must contain exactly "
            f"{len(expected_steps)} checkpoints"
        )
    depth_totals = {"0": 16, "1": 176, "2": 752, "3": 20240}
    for checkpoint_index, (checkpoint, expected_step) in enumerate(
        zip(depth_trace, expected_steps)
    ):
        checkpoint_where = (
            f"{where}.training_by_word_depth[{checkpoint_index}]"
        )
        row = _require_keys(
            checkpoint,
            ["optimizer_step", "by_word_depth"],
            checkpoint_where,
        )
        _expect(row["optimizer_step"], expected_step, f"{checkpoint_where}.optimizer_step")
        depths = _require_keys(
            row["by_word_depth"], depth_totals, f"{checkpoint_where}.by_word_depth"
        )
        for depth, total in depth_totals.items():
            _validate_behavior_metric(
                depths[depth], f"{checkpoint_where}.by_word_depth.{depth}", total
            )

    final_depths = depth_trace[-1]["by_word_depth"]
    for count_key in (
        "correct_responses",
        "supported_responses",
        "correct_and_supported_responses",
    ):
        _expect(
            sum(final_depths[depth][count_key] for depth in depth_totals),
            behavior["training"][count_key],
            f"{where}.final depth trace {count_key}",
        )
    _expect(
        behavior["heldout"]["correct_responses"],
        behavior["h2"]["correct_responses"] + behavior["h3"]["correct_responses"],
        f"{where}.heldout correct response count",
    )
    _expect(
        behavior["heldout"]["correct_and_supported_responses"],
        behavior["h2"]["correct_and_supported_responses"]
        + behavior["h3"]["correct_and_supported_responses"],
        f"{where}.heldout correct-and-supported response count",
    )
    _expect(
        behavior["heldout"]["supported_responses"],
        behavior["h2"]["supported_responses"] + behavior["h3"]["supported_responses"],
        f"{where}.heldout supported response count",
    )
    return recomputed


def _validate_seed_record(
    value: Any,
    index: int,
    config: dict[str, Any],
    layout: dict[str, Any],
    producer_kind: str,
) -> dict[str, dict[str, Any]] | None:
    where = f"evidence.seeds[{index}]"
    seed_keys = [
        "seed",
        "loss_trace_sha256",
        "representative_ids",
        "representative_prefixes",
        "representative_oracle_states",
        "representative_response_probabilities",
        "representative_signatures",
        "primitive_successor_signatures",
        "toggle_twice_signatures",
        "law_rows",
        "heldout_endpoint_signatures",
        "canonical_handles",
        "representative_continuation_signatures",
        "canonical_continuation_signatures",
        "representative_terminal_bits",
        "canonical_terminal_bits",
        "recovered_action_table",
        "support_flags",
    ]
    if _is_round36b(config):
        seed_keys.extend(["behavior_fit", "diagnostic_response_probabilities"])
    seed = _require_keys(
        value,
        seed_keys,
        where,
    )
    _expect(seed["seed"], config["training"]["model_seeds"][index], f"{where}.seed")
    _expect_sha(seed["loss_trace_sha256"], f"{where}.loss_trace_sha256")
    _expect(seed["representative_ids"], layout["ids"], f"{where}.representative_ids")
    _expect(seed["representative_prefixes"], layout["prefix_spellings"], f"{where}.representative_prefixes")
    _expect(seed["representative_oracle_states"], layout["oracle_states"], f"{where}.representative_oracle_states")
    _validate_matrix(seed["representative_response_probabilities"], 944, 12, f"{where}.representative_response_probabilities", _validate_probability)
    if not isinstance(seed["representative_signatures"], list) or len(seed["representative_signatures"]) != 944:
        raise ContractError(f"{where}.representative_signatures must have 944 rows")
    low = config["thresholds"]["signature_low"]
    high = config["thresholds"]["signature_high"]
    for row_index, signature in enumerate(seed["representative_signatures"]):
        _validate_signature(signature, f"{where}.representative_signatures[{row_index}]")
        recomputed = _signature_from_probabilities(
            seed["representative_response_probabilities"][row_index], low, high
        )
        if signature != recomputed:
            raise ContractError(f"{where} representative signature {row_index} does not replay")
    _validate_matrix(seed["primitive_successor_signatures"], 944, 11, f"{where}.primitive_successor_signatures", _validate_signature)
    _validate_matrix(seed["toggle_twice_signatures"], 944, 4, f"{where}.toggle_twice_signatures", _validate_signature)
    law_rows = seed["law_rows"]
    if not isinstance(law_rows, list) or len(law_rows) != 384:
        raise ContractError(f"{where}.law_rows must have 384 rows")
    expected_law_keys = [
        "handle",
        "swap",
        "toggle",
        "swap_after_toggle",
        "toggle_after_swap",
        "conjugate_after_swap",
        "registered_cell_pass",
    ]
    cursor = 0
    for handle in range(16):
        for swap in SWAP_NAMES:
            for toggle in TOGGLE_NAMES:
                row = _require_keys(law_rows[cursor], expected_law_keys, f"{where}.law_rows[{cursor}]")
                _expect(row["handle"], handle, f"{where}.law_rows[{cursor}].handle")
                _expect(row["swap"], swap, f"{where}.law_rows[{cursor}].swap")
                _expect(row["toggle"], toggle, f"{where}.law_rows[{cursor}].toggle")
                for key in ("swap_after_toggle", "toggle_after_swap", "conjugate_after_swap"):
                    _validate_signature(row[key], f"{where}.law_rows[{cursor}].{key}")
                if not isinstance(row["registered_cell_pass"], bool):
                    raise ContractError(f"{where}.law_rows[{cursor}].registered_cell_pass must be boolean")
                cursor += 1
    _validate_matrix(seed["heldout_endpoint_signatures"], 16, 140, f"{where}.heldout_endpoint_signatures", _validate_signature)
    if not isinstance(seed["canonical_handles"], list) or len(seed["canonical_handles"]) != 944:
        raise ContractError(f"{where}.canonical_handles must have 944 rows")
    for row_index, handle in enumerate(seed["canonical_handles"]):
        if isinstance(handle, bool) or not isinstance(handle, int) or not -1 <= handle < 16:
            raise ContractError(f"{where}.canonical_handles[{row_index}] is invalid")
    for key in ("representative_continuation_signatures", "canonical_continuation_signatures"):
        _validate_matrix(seed[key], 944, 140, f"{where}.{key}", _validate_signature)
    for key in ("representative_terminal_bits", "canonical_terminal_bits"):
        _validate_matrix(seed[key], 944, 140, f"{where}.{key}", _validate_bit)
    _validate_matrix(seed["recovered_action_table"], 16, 11, f"{where}.recovered_action_table", _validate_signature)
    support_flags = _require_keys(
        seed["support_flags"],
        [
            "representatives",
            "primitive_successors",
            "toggle_twice",
            "law_endpoints",
            "heldout_endpoints",
            "representative_continuations",
            "canonical_continuations",
        ],
        f"{where}.support_flags",
    )
    _expect(support_flags, _support_flags_for_seed(seed), f"{where}.support_flags replay")
    if _is_round36b(config):
        diagnostic_probabilities = _require_keys(
            seed["diagnostic_response_probabilities"],
            [
                "primitive_successors",
                "toggle_twice",
                "law_endpoints",
                "heldout_endpoints",
                "representative_continuations",
            ],
            f"{where}.diagnostic_response_probabilities",
        )
        _validate_probability_signature_table(
            diagnostic_probabilities["primitive_successors"],
            seed["primitive_successor_signatures"],
            f"{where}.diagnostic_response_probabilities.primitive_successors",
            low,
            high,
        )
        _validate_probability_signature_table(
            diagnostic_probabilities["toggle_twice"],
            seed["toggle_twice_signatures"],
            f"{where}.diagnostic_response_probabilities.toggle_twice",
            low,
            high,
        )
        law_signatures = [
            [
                row["swap_after_toggle"],
                row["toggle_after_swap"],
                row["conjugate_after_swap"],
            ]
            for row in seed["law_rows"]
        ]
        _validate_probability_signature_table(
            diagnostic_probabilities["law_endpoints"],
            law_signatures,
            f"{where}.diagnostic_response_probabilities.law_endpoints",
            low,
            high,
        )
        _validate_probability_signature_table(
            diagnostic_probabilities["heldout_endpoints"],
            seed["heldout_endpoint_signatures"],
            f"{where}.diagnostic_response_probabilities.heldout_endpoints",
            low,
            high,
        )
        _validate_probability_signature_table(
            diagnostic_probabilities["representative_continuations"],
            seed["representative_continuation_signatures"],
            (
                f"{where}.diagnostic_response_probabilities."
                "representative_continuations"
            ),
            low,
            high,
        )
        recomputed_behavior_fit = _validate_behavior_fit(
            seed["behavior_fit"],
            f"{where}.behavior_fit",
            config,
            producer_kind,
            seed["loss_trace_sha256"],
        )
        return recomputed_behavior_fit
    return None


def _gate_result(numerator: int, denominator: int, details: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "numerator": numerator,
        "denominator": denominator,
        "fraction": numerator / denominator if denominator else 0.0,
        "passed": denominator > 0 and numerator == denominator,
    }
    if details:
        result["details"] = details
    return result


def _confidence_free_signature(probabilities: Sequence[Any]) -> str:
    if len(probabilities) != 12:
        raise ContractError("a confidence-free signature requires 12 probabilities")
    bits: list[str] = []
    for value in probabilities:
        probability = _expect_number(value, "diagnostic response probability")
        if not 0.0 <= probability <= 1.0:
            raise ContractError("diagnostic response probabilities must lie in [0,1]")
        bits.append("1" if probability > 0.5 else "0")
    return "".join(bits)


def _confidence_free_table(
    probabilities: list[list[list[float]]],
) -> list[list[str]]:
    return [
        [_confidence_free_signature(cell) for cell in row]
        for row in probabilities
    ]


def _diagnostic_seed_view(seed: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    probability_tables = seed["diagnostic_response_probabilities"]
    representatives = [
        _confidence_free_signature(row)
        for row in seed["representative_response_probabilities"]
    ]
    primitive = _confidence_free_table(
        probability_tables["primitive_successors"]
    )
    toggle_twice = _confidence_free_table(probability_tables["toggle_twice"])
    law_endpoints = _confidence_free_table(probability_tables["law_endpoints"])
    heldout = _confidence_free_table(probability_tables["heldout_endpoints"])
    representative_continuations = _confidence_free_table(
        probability_tables["representative_continuations"]
    )

    law_rows: list[dict[str, Any]] = []
    for row, endpoint_row in zip(seed["law_rows"], law_endpoints):
        diagnostic_row = dict(row)
        diagnostic_row["swap_after_toggle"] = endpoint_row[0]
        diagnostic_row["toggle_after_swap"] = endpoint_row[1]
        diagnostic_row["conjugate_after_swap"] = endpoint_row[2]
        law_rows.append(diagnostic_row)

    encoder_indices = [handle * 59 for handle in range(16)]
    signature_to_handle = {
        representatives[index]: handle
        for handle, index in enumerate(encoder_indices)
    }
    canonical_handles = [
        signature_to_handle.get(signature, -1) for signature in representatives
    ]
    unknown_row = ["?" * 12] * 140
    canonical_continuations = [
        heldout[handle] if handle >= 0 else list(unknown_row)
        for handle in canonical_handles
    ]
    representative_terminal_bits = [
        [int(signature[0]) for signature in row]
        for row in representative_continuations
    ]
    canonical_terminal_bits = [
        [int(signature[0]) if _supported(signature) else None for signature in row]
        for row in canonical_continuations
    ]
    recovered_action_table: list[list[str]] = []
    for oracle_signature in evidence["world"]["oracle_signatures"]:
        handle = signature_to_handle.get(oracle_signature, -1)
        recovered_action_table.append(
            primitive[handle * 59] if handle >= 0 else ["?" * 12] * 11
        )

    return {
        **seed,
        "representative_signatures": representatives,
        "primitive_successor_signatures": primitive,
        "toggle_twice_signatures": toggle_twice,
        "law_rows": law_rows,
        "heldout_endpoint_signatures": heldout,
        "canonical_handles": canonical_handles,
        "representative_continuation_signatures": representative_continuations,
        "canonical_continuation_signatures": canonical_continuations,
        "representative_terminal_bits": representative_terminal_bits,
        "canonical_terminal_bits": canonical_terminal_bits,
        "recovered_action_table": recovered_action_table,
    }


def _signature_support_summary(signatures: Sequence[str]) -> dict[str, Any]:
    signature_count = len(signatures)
    supported_signatures = sum(int(_supported(signature)) for signature in signatures)
    supported_components = sum(
        sum(int(component != "?") for component in signature)
        for signature in signatures
    )
    total_components = signature_count * 12
    return {
        "supported_signatures": supported_signatures,
        "total_signatures": signature_count,
        "supported_components": supported_components,
        "total_components": total_components,
        "unsupported_component_count": total_components - supported_components,
        "passed": total_components > 0 and supported_components == total_components,
    }


def _flatten_signature_rows(rows: Sequence[Sequence[str]]) -> list[str]:
    return [signature for row in rows for signature in row]


def _operational_signature_support(evidence: dict[str, Any]) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    passed_seeds = 0
    for seed in evidence["seeds"]:
        representatives = seed["representative_signatures"]
        encoder = [representatives[handle * 59] for handle in range(16)]
        primitive = _flatten_signature_rows(seed["primitive_successor_signatures"])
        toggles = _flatten_signature_rows(seed["toggle_twice_signatures"])
        law = [
            row[key]
            for row in seed["law_rows"]
            for key in (
                "swap_after_toggle",
                "toggle_after_swap",
                "conjugate_after_swap",
            )
        ]
        heldout = seed["heldout_endpoint_signatures"]
        h2 = [signature for row in heldout for signature in row[:74]]
        h3 = [signature for row in heldout for signature in row[74:]]
        representative_continuations = _flatten_signature_rows(
            seed["representative_continuation_signatures"]
        )
        canonical_continuations = _flatten_signature_rows(
            seed["canonical_continuation_signatures"]
        )
        action_table = _flatten_signature_rows(seed["recovered_action_table"])
        domains = {
            "quotient_availability": _signature_support_summary(encoder),
            "quotient_well_definedness": _signature_support_summary(
                list(representatives) + primitive
            ),
            "toggle_involution": _signature_support_summary(
                list(representatives) + toggles
            ),
            "swap_toggle_table": _signature_support_summary(law),
            "heldout_depth2_closure": _signature_support_summary(h2),
            "heldout_depth3_closure": _signature_support_summary(h3),
            "interchangeability": _signature_support_summary(
                list(representatives)
                + representative_continuations
                + canonical_continuations
            ),
            "action_table_truth": _signature_support_summary(action_table),
            "cross_seed_action_table": _signature_support_summary(action_table),
        }
        seed_passed = all(domain["passed"] for domain in domains.values())
        passed_seeds += int(seed_passed)
        per_seed[str(seed["seed"])] = {
            "passed": seed_passed,
            "domains": domains,
        }
    return {
        "passed_seeds": passed_seeds,
        "required_seeds": 5,
        "passed": passed_seeds == 5,
        "per_seed": per_seed,
    }


def _cross_seed_cellwise_accounting(
    recovered_tables: list[list[list[str]]],
    truth_table: list[list[str]],
) -> dict[str, Any]:
    identical_cells = 0
    identical_supported_cells = 0
    all_five_truthful_cells = 0
    majority_truthful_cells = 0
    majority_truthful_bits = 0
    majority_resolved_bits = 0
    for state_index in range(16):
        for action_index in range(11):
            values = [
                table[state_index][action_index] for table in recovered_tables
            ]
            truth = truth_table[state_index][action_index]
            identical = all(value == values[0] for value in values[1:])
            identical_cells += int(identical)
            identical_supported_cells += int(identical and _supported(values[0]))
            all_five_truthful_cells += int(
                all(_supported(value) and value == truth for value in values)
            )
            majority_bits: list[str] = []
            for component_index in range(12):
                zeros = sum(value[component_index] == "0" for value in values)
                ones = sum(value[component_index] == "1" for value in values)
                majority = "0" if zeros >= 3 else "1" if ones >= 3 else "?"
                majority_bits.append(majority)
                if majority != "?":
                    majority_resolved_bits += 1
                    majority_truthful_bits += int(
                        majority == truth[component_index]
                    )
            majority_truthful_cells += int("".join(majority_bits) == truth)
    return {
        "identical_cells": {"numerator": identical_cells, "denominator": 176},
        "identical_supported_cells": {
            "numerator": identical_supported_cells,
            "denominator": 176,
        },
        "all_five_truthful_cells": {
            "numerator": all_five_truthful_cells,
            "denominator": 176,
        },
        "bitwise_majority_truthful_cells": {
            "numerator": majority_truthful_cells,
            "denominator": 176,
        },
        "bitwise_majority_truthful_bits": {
            "numerator": majority_truthful_bits,
            "denominator": 2112,
            "resolved_bits": majority_resolved_bits,
        },
        "note": (
            "Literal cellwise accounting; the inherited whole-table exact gate "
            "remains separate and all-or-none."
        ),
    }


def _iter_probabilities(value: Any) -> Iterator[float]:
    if isinstance(value, list):
        for item in value:
            yield from _iter_probabilities(item)
    else:
        yield float(value)


def _diagnostic_margin_table(evidence: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for seed in evidence["seeds"]:
        domains = {
            "representatives": seed["representative_response_probabilities"],
            **seed["diagnostic_response_probabilities"],
        }
        seed_result: dict[str, Any] = {}
        for name, raw in domains.items():
            margins = sorted(abs(value - 0.5) for value in _iter_probabilities(raw))
            count = len(margins)
            midpoint = count // 2
            median = (
                margins[midpoint]
                if count % 2
                else (margins[midpoint - 1] + margins[midpoint]) / 2.0
            )
            seed_result[name] = {
                "component_count": count,
                "minimum_absolute_margin": margins[0],
                "median_absolute_margin": median,
                "mean_absolute_margin": math.fsum(margins) / count,
                "maximum_absolute_margin": margins[-1],
                "zero_margin_count": sum(margin == 0.0 for margin in margins),
            }
        result[str(seed["seed"])] = seed_result
    return result


def _scientific_gates(
    evidence: dict[str, Any],
    config: dict[str, Any],
    deadline: float | None = None,
    *,
    enforce_registered_law_flag: bool = True,
) -> dict[str, Any]:
    states = _states()
    handle_to_state = evidence["world"]["handle_to_state"]
    oracle_signatures = evidence["world"]["oracle_signatures"]
    universe = _word_universe(config)
    gates_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    recovered_tables = []

    for seed in evidence["seeds"]:
        if deadline is not None:
            _check_deadline(deadline)
        seed_id = seed["seed"]
        signatures = seed["representative_signatures"]
        oracle_states = seed["representative_oracle_states"]
        encoder_indices = [handle * 59 for handle in range(16)]

        quotient_ok = 0
        encoder_seen = []
        for handle, representative_index in enumerate(encoder_indices):
            signature = signatures[representative_index]
            oracle = oracle_signatures[handle_to_state[handle]]
            cell_ok = _supported(signature) and signature == oracle
            quotient_ok += int(cell_ok)
            if cell_ok:
                encoder_seen.append(signature)
        quotient_gate = _gate_result(
            quotient_ok + int(len(set(encoder_seen)) == 16),
            17,
            {"encoder_cells": quotient_ok, "distinct_classes": len(set(encoder_seen))},
        )

        descent_ok = 0
        descent_total = 944 * 11
        class_targets: dict[tuple[str, int], set[str]] = {}
        for representative_index, source_signature in enumerate(signatures):
            if deadline is not None and representative_index % 64 == 0:
                _check_deadline(deadline)
            oracle_state = states[oracle_states[representative_index]]
            for action_index, action in enumerate(ACTION_NAMES):
                target = seed["primitive_successor_signatures"][representative_index][action_index]
                expected = _oracle_signature(_apply_action(oracle_state, action))
                cell_ok = _supported(source_signature) and _supported(target) and target == expected
                descent_ok += int(cell_ok)
                class_targets.setdefault((source_signature, action_index), set()).add(target)
        congruence_ok = 0
        for source_signature in oracle_signatures:
            for action_index in range(11):
                targets = class_targets.get((source_signature, action_index), set())
                congruence_ok += int(len(targets) == 1 and _supported(next(iter(targets)))) if targets else 0
        descent_gate = _gate_result(
            descent_ok + congruence_ok,
            descent_total + 176,
            {"representative_cells": descent_ok, "class_action_cells": congruence_ok},
        )

        involution_ok = 0
        for representative_index, source_signature in enumerate(signatures):
            for target in seed["toggle_twice_signatures"][representative_index]:
                involution_ok += int(
                    _supported(source_signature) and _supported(target) and target == source_signature
                )
        involution_gate = _gate_result(involution_ok, 944 * 4)

        law_ok = 0
        for row in seed["law_rows"]:
            state = states[handle_to_state[row["handle"]]]
            swap = row["swap"]
            toggle = row["toggle"]
            i, j = (int(part) for part in swap[5:-1].split(","))
            k = int(toggle[7:-1])
            mapped = j if k == i else i if k == j else k
            expected_a = _oracle_signature(_apply_word(state, (toggle, swap)))
            expected_b = _oracle_signature(_apply_word(state, (swap, toggle)))
            expected_c = _oracle_signature(_apply_word(state, (swap, f"toggle({mapped})")))
            supported = all(
                _supported(row[key])
                for key in ("swap_after_toggle", "toggle_after_swap", "conjugate_after_swap")
            )
            exact = (
                row["swap_after_toggle"] == expected_a
                and row["toggle_after_swap"] == expected_b
                and row["conjugate_after_swap"] == expected_c
            )
            relation = (
                row["swap_after_toggle"] == row["toggle_after_swap"]
                if k not in (i, j)
                else row["swap_after_toggle"] != row["toggle_after_swap"]
            )
            conjugacy = row["swap_after_toggle"] == row["conjugate_after_swap"]
            replayed_cell = supported and exact and relation and conjugacy
            law_ok += int(
                replayed_cell
                and (
                    not enforce_registered_law_flag
                    or row["registered_cell_pass"] == replayed_cell
                )
            )
        law_gate = _gate_result(law_ok, 384)

        h2_ok = 0
        h3_ok = 0
        for handle, row in enumerate(seed["heldout_endpoint_signatures"]):
            start_state = states[handle_to_state[handle]]
            for word_index, word in enumerate(universe["heldout"]):
                target = row[word_index]
                expected = _oracle_signature(_apply_word(start_state, word))
                cell_ok = _supported(target) and target == expected
                if word_index < len(universe["h2"]):
                    h2_ok += int(cell_ok)
                else:
                    h3_ok += int(cell_ok)
        h2_gate = _gate_result(h2_ok, 74 * 16)
        h3_gate = _gate_result(h3_ok, 66 * 16)

        interchange_ok = 0
        for representative_index in range(944):
            if deadline is not None and representative_index % 64 == 0:
                _check_deadline(deadline)
            source_signature = signatures[representative_index]
            handle = seed["canonical_handles"][representative_index]
            expected_handle = -1
            if _supported(source_signature):
                for candidate_handle, encoder_index in enumerate(encoder_indices):
                    if signatures[encoder_index] == source_signature:
                        expected_handle = candidate_handle
                        break
            canonical_binding_ok = handle == expected_handle and handle >= 0
            for word_index in range(140):
                left = seed["representative_continuation_signatures"][representative_index][word_index]
                right = seed["canonical_continuation_signatures"][representative_index][word_index]
                left_bit = seed["representative_terminal_bits"][representative_index][word_index]
                right_bit = seed["canonical_terminal_bits"][representative_index][word_index]
                bits_replay = (
                    left_bit == (int(left[0]) if _supported(left) else None)
                    and right_bit == (int(right[0]) if _supported(right) else None)
                )
                interchange_ok += int(
                    canonical_binding_ok
                    and _supported(left)
                    and _supported(right)
                    and left == right
                    and left_bit == right_bit
                    and bits_replay
                )
        interchange_gate = _gate_result(interchange_ok, 944 * 140)

        recovered = seed["recovered_action_table"]
        table_ok = 0
        for state_index in range(16):
            for action_index in range(11):
                target = recovered[state_index][action_index]
                table_ok += int(
                    _supported(target)
                    and target == evidence["world"]["oracle_action_table"][state_index][action_index]
                )
        recovered_tables.append(recovered)
        seed_table_gate = _gate_result(table_ok, 176)

        gates_by_seed[seed_id] = {
            "quotient_availability": quotient_gate,
            "quotient_well_definedness": descent_gate,
            "toggle_involution": involution_gate,
            "swap_toggle_table": law_gate,
            "heldout_depth2_closure": h2_gate,
            "heldout_depth3_closure": h3_gate,
            "interchangeability": interchange_gate,
            "action_table_truth": seed_table_gate,
        }

    gate_names = [
        "quotient_availability",
        "quotient_well_definedness",
        "toggle_involution",
        "swap_toggle_table",
        "heldout_depth2_closure",
        "heldout_depth3_closure",
        "interchangeability",
        "action_table_truth",
    ]
    joint: dict[str, Any] = {}
    for gate_name in gate_names:
        passed_seeds = sum(int(gates_by_seed[seed][gate_name]["passed"]) for seed in gates_by_seed)
        joint[gate_name] = {
            "passed_seeds": passed_seeds,
            "required_seeds": 5,
            "passed": passed_seeds == 5,
            "per_seed": {str(seed): gates_by_seed[seed][gate_name] for seed in gates_by_seed},
        }
    identical = all(table == recovered_tables[0] for table in recovered_tables[1:])
    truth = recovered_tables[0] == evidence["world"]["oracle_action_table"]
    joint["cross_seed_action_table"] = {
        "numerator": 176 if identical and truth else 0,
        "denominator": 176,
        "identical_across_five_seeds": identical,
        "equals_behavioral_truth": truth,
        "passed": identical and truth,
    }
    return joint


def _diagnostic_component_error_counts(
    gates: dict[str, Any],
    cellwise: dict[str, Any],
) -> dict[str, Any]:
    errors: dict[str, Any] = {}
    for gate_name, gate in gates.items():
        if gate_name == "cross_seed_action_table":
            truthful = cellwise["all_five_truthful_cells"]["numerator"]
            errors[gate_name] = {
                "truth_cell_error_count": 176 - truthful,
                "whole_table_gate_failed": not gate["passed"],
                "note": "Cell errors are literal; the primary gate is all-or-none.",
            }
            continue
        per_seed = {
            seed: {
                "error_count": row["denominator"] - row["numerator"],
                "component_count": row["denominator"],
            }
            for seed, row in gate["per_seed"].items()
        }
        errors[gate_name] = {
            "error_count": sum(row["error_count"] for row in per_seed.values()),
            "component_count": sum(
                row["component_count"] for row in per_seed.values()
            ),
            "failed_seed_count": gate["required_seeds"] - gate["passed_seeds"],
            "per_seed": per_seed,
        }
    return errors


def _confidence_free_diagnostic_table(
    evidence: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    diagnostic_evidence = {
        **evidence,
        "seeds": [
            _diagnostic_seed_view(seed, evidence) for seed in evidence["seeds"]
        ],
    }
    gates = _scientific_gates(
        diagnostic_evidence,
        config,
        enforce_registered_law_flag=False,
    )
    recovered_tables = [
        seed["recovered_action_table"] for seed in diagnostic_evidence["seeds"]
    ]
    cellwise = _cross_seed_cellwise_accounting(
        recovered_tables, evidence["world"]["oracle_action_table"]
    )
    return {
        "label": DIAGNOSTIC_LABEL,
        "decision_rule": "1 iff p > 0.5; p == 0.5 decodes as 0",
        "can_alter_primary_status": False,
        "gates": gates,
        "component_error_counts": _diagnostic_component_error_counts(
            gates, cellwise
        ),
        "decision_margins_from_0_5": _diagnostic_margin_table(evidence),
        "cross_seed_cellwise_accounting": cellwise,
    }


def _producer_counts_from_gates(gates: dict[str, Any]) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    for seed in (11, 23, 37, 53, 71):
        per_seed[str(seed)] = {}
        for gate_name, gate in gates.items():
            if gate_name == "cross_seed_action_table":
                continue
            seed_gate = gate["per_seed"][str(seed)]
            per_seed[str(seed)][gate_name] = {
                "numerator": seed_gate["numerator"],
                "denominator": seed_gate["denominator"],
            }
    cross_seed = gates["cross_seed_action_table"]
    return {
        "per_seed": per_seed,
        "cross_seed_action_table": {
            "numerator": cross_seed["numerator"],
            "denominator": cross_seed["denominator"],
        },
    }


def _attach_producer_counts(
    evidence: dict[str, Any],
    config: dict[str, Any],
    deadline: float | None = None,
) -> None:
    evidence["producer_counts"] = _producer_counts_from_gates(
        _scientific_gates(evidence, config, deadline)
    )


def _validate_evidence(
    value: Any, manifest: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    evidence = _require_keys(
        value,
        [
            "schema_version",
            "registration_id",
            "producer_kind",
            "config_sha256",
            "code_sha256",
            "weights_sha256",
            "action_order",
            "data_seed",
            "model_seeds",
            "expected_counts",
            "world",
            "words",
            "construction",
            "seeds",
            "producer_counts",
        ],
        "evidence",
    )
    _expect(evidence["schema_version"], SCHEMA_EVIDENCE, "evidence.schema_version")
    _expect(
        evidence["registration_id"],
        _registration_id(config),
        "evidence.registration_id",
    )
    _expect(evidence["producer_kind"], manifest["producer_kind"], "evidence.producer_kind")
    for key in ("config_sha256", "code_sha256", "weights_sha256"):
        _expect_sha(evidence[key], f"evidence.{key}")
        _expect(evidence[key], manifest[key], f"evidence/manifest {key}")
    _expect(evidence["action_order"], ACTION_NAMES, "evidence.action_order")
    _expect(evidence["data_seed"], 3601, "evidence.data_seed")
    _expect(evidence["model_seeds"], [11, 23, 37, 53, 71], "evidence.model_seeds")
    _expect(evidence["expected_counts"], _expected_counts(), "evidence.expected_counts")
    _validate_world(evidence["world"], config)
    _validate_words(evidence["words"], config)
    _validate_construction(evidence["construction"], evidence["producer_kind"], config)
    seeds = evidence["seeds"]
    if not isinstance(seeds, list) or len(seeds) != 5:
        raise ContractError("evidence.seeds must contain exactly five seed records")
    layout = _representative_layout(config)
    recomputed_behavior_fit: dict[str, dict[str, Any]] = {}
    for index, seed in enumerate(seeds):
        recomputed = _validate_seed_record(
            seed, index, config, layout, evidence["producer_kind"]
        )
        if _is_round36b(config):
            if recomputed is None:
                raise ContractError("Round 36b behavior rows were not recomputed")
            recomputed_behavior_fit[str(seed["seed"])] = recomputed
    gates = _scientific_gates(evidence, config)
    _expect(
        evidence["producer_counts"],
        _producer_counts_from_gates(gates),
        "evidence.producer_counts replay",
    )
    return evidence, gates, recomputed_behavior_fit


def _input_hashes(config_path: Path, evidence_dir: Path) -> dict[str, str]:
    paths = {
        "config_argument": config_path,
        "config_copy": evidence_dir / "config.json",
        "manifest": evidence_dir / "manifest.json",
        "evidence": evidence_dir / "evidence.json",
    }
    return {name: _sha256_file(path) for name, path in paths.items() if path.is_file()}


def _behavior_fit_eligibility(
    recomputed_behavior_fit: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    passed_seeds = 0
    for seed in (11, 23, 37, 53, 71):
        behavior = recomputed_behavior_fit[str(seed)]
        training = behavior["training"]
        heldout = behavior["heldout"]
        training_exact = (
            training["correct_responses"] == training["total_responses"] == 21184
        )
        heldout_exact = (
            heldout["correct_responses"] == heldout["total_responses"] == 2240
        )
        passed = training_exact and heldout_exact
        passed_seeds += int(passed)
        per_seed[str(seed)] = {
            "training_correct": training["correct_responses"],
            "training_total": training["total_responses"],
            "heldout_correct": heldout["correct_responses"],
            "heldout_total": heldout["total_responses"],
            "passed": passed,
        }
    return {
        "passed_seeds": passed_seeds,
        "required_seeds": 5,
        "passed": passed_seeds == 5,
        "per_seed": per_seed,
    }


def _reduce_directory(
    config_path: Path,
    evidence_dir: Path,
    *,
    write_verdict: bool,
    record_ledger: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    gates: dict[str, Any] = {}
    behavior_fit_eligibility: dict[str, Any] = {}
    operational_signature_support: dict[str, Any] = {}
    primary_cellwise_accounting: dict[str, Any] = {}
    confidence_free_diagnostic: dict[str, Any] = {}
    status = "INVALID"
    config: dict[str, Any] | None = None
    producer_kind: str | None = None
    try:
        config_bytes = config_path.read_bytes()
        config = _validate_config(_read_json(config_path))
        copied_config = evidence_dir / "config.json"
        manifest_path = evidence_dir / "manifest.json"
        evidence_path = evidence_dir / "evidence.json"
        for required in (copied_config, manifest_path, evidence_path):
            if not required.is_file():
                raise ContractError(f"missing reducer input {required}")
        if copied_config.read_bytes() != config_bytes:
            raise ContractError("reducer config argument and stored config copy differ byte-for-byte")
        manifest = _validate_manifest(_read_json(manifest_path), config)
        producer_kind = manifest["producer_kind"]
        config_sha = _sha256_bytes(config_bytes)
        _expect(manifest["config_sha256"], config_sha, "manifest.config_sha256")
        _expect(manifest["config_copy_sha256"], _sha256_file(copied_config), "manifest.config_copy_sha256")
        _expect(manifest["evidence_sha256"], _sha256_file(evidence_path), "manifest.evidence_sha256")
        word_pack = _word_pack(config)
        expected_word_hashes = {
            key: value for key, value in word_pack.items() if key.endswith("_list_sha256")
        }
        _expect(manifest["word_list_hashes"], expected_word_hashes, "manifest.word_list_hashes")
        evidence, gates, recomputed_behavior_fit = _validate_evidence(
            _read_json(evidence_path), manifest, config
        )
        if _is_round36b(config):
            behavior_fit_eligibility = _behavior_fit_eligibility(
                recomputed_behavior_fit
            )
            operational_signature_support = _operational_signature_support(evidence)
            primary_cellwise_accounting = _cross_seed_cellwise_accounting(
                [seed["recovered_action_table"] for seed in evidence["seeds"]],
                evidence["world"]["oracle_action_table"],
            )
            confidence_free_diagnostic = _confidence_free_diagnostic_table(
                evidence, config
            )
        failed = [name for name, gate in gates.items() if not gate["passed"]]
        if _is_round36b(config) and not behavior_fit_eligibility["passed"]:
            status = UNDERFIT_STATUS
        elif _is_round36b(config) and not operational_signature_support["passed"]:
            status = SIGNATURE_UNSUPPORTED_STATUS
        elif not failed:
            status = PASS_STATUS
        elif _is_round36b(config):
            status = SUPPORTED_NONCONGRUENT_PREFIX + " — " + ", ".join(
                name.upper().replace("_", " ") for name in failed
            )
        else:
            status = "FAIL — " + ", ".join(name.upper().replace("_", " ") for name in failed)
        del evidence
    except (ContractError, OSError) as exc:
        errors.append(str(exc))
        status = "INVALID"

    verdict = {
        "schema_version": SCHEMA_VERDICT,
        "registration_id": (
            _registration_id(config) if config is not None else REGISTRATION_ID
        ),
        "reducer_code_sha256": _module_sha256(),
        "reduced_at": _utc_now(),
        "input_sha256": _input_hashes(config_path, evidence_dir),
        "integrity_scope": INTEGRITY_SCOPE,
        "result_scope": (
            "FIXTURE-ONLY"
            if producer_kind == "fixture"
            else "SCIENTIFIC"
            if producer_kind == "learned"
            else "UNKNOWN"
        ),
        "status": status,
        "gates": gates,
        "errors": errors,
        "claim_boundary": (
            "One tiny learned finite world recovered the registered operational quotient and action algebra."
            if status == PASS_STATUS and producer_kind == "learned"
            else None
        ),
    }
    if config is not None and _is_round36b(config):
        verdict["behavior_fit_eligibility"] = behavior_fit_eligibility
        verdict["operational_signature_support"] = operational_signature_support
        verdict["cross_seed_cellwise_accounting"] = primary_cellwise_accounting
        verdict["confidence_free_diagnostic"] = confidence_free_diagnostic
    if write_verdict:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        _write_json(evidence_dir / "verdict.json", verdict)
    if record_ledger and config is not None:
        _append_ledger(
            _ledger_entry(
                "round36_operational_quotient_reduce",
                "Round 36 declarative reduction of stored evidence",
                config_path,
                " ".join(sys.argv),
                {"status": status, "gates": {name: gate["passed"] for name, gate in gates.items()}},
                [str(evidence_dir / "verdict.json")],
                status,
                str(evidence_dir),
                (
                    "Reducer read only config, manifest, and evidence; weights were not "
                    f"reducer input; {INTEGRITY_SCOPE}."
                ),
            )
        )
    return verdict


def _refresh_fixture_hash(directory: Path, manifest: dict[str, Any]) -> None:
    manifest["evidence_sha256"] = _sha256_file(directory / "evidence.json")
    _write_json(directory / "manifest.json", manifest)


def _run_fixture(config_path: Path, output_dir: Path | None) -> dict[str, Any]:
    config = _validate_config(_read_json(config_path))
    torch_loaded_before = "torch" in sys.modules

    def execute(directory: Path) -> dict[str, Any]:
        _write_fixture_artifact(directory, config_path, config)
        results: dict[str, str] = {}

        pass_verdict = _reduce_directory(
            config_path, directory, write_verdict=True, record_ledger=False
        )
        results["exact_fixture"] = pass_verdict["status"]
        if pass_verdict["claim_boundary"] is not None:
            raise ContractError("fixture reducer emitted learned-world claim text")
        if pass_verdict["result_scope"] != "FIXTURE-ONLY":
            raise ContractError("fixture reducer did not label its result FIXTURE-ONLY")
        if _is_round36b(config):
            diagnostic = pass_verdict.get("confidence_free_diagnostic")
            if (
                not isinstance(diagnostic, dict)
                or diagnostic.get("label") != DIAGNOSTIC_LABEL
                or diagnostic.get("can_alter_primary_status") is not False
                or set(diagnostic.get("gates", {})) != set(pass_verdict["gates"])
            ):
                raise ContractError(
                    "Round 36b fixture did not emit the complete DIAGNOSTIC table"
                )
            for accounting in (
                pass_verdict["cross_seed_cellwise_accounting"],
                diagnostic["cross_seed_cellwise_accounting"],
            ):
                expected_cellwise = {
                    "identical_cells": 176,
                    "identical_supported_cells": 176,
                    "all_five_truthful_cells": 176,
                    "bitwise_majority_truthful_cells": 176,
                    "bitwise_majority_truthful_bits": 2112,
                }
                for key, expected_count in expected_cellwise.items():
                    if accounting[key]["numerator"] != expected_count:
                        raise ContractError(
                            f"Round 36b fixture cellwise accounting failed for {key}"
                        )

        with _temporary_directory("round36_fixture_missing_") as temporary:
            branch = temporary / "artifact"
            shutil.copytree(directory, branch)
            evidence = _read_json(branch / "evidence.json")
            manifest = _read_json(branch / "manifest.json")
            evidence["seeds"][0]["heldout_endpoint_signatures"][0].pop()
            _write_json(branch / "evidence.json", evidence)
            _refresh_fixture_hash(branch, manifest)
            missing_verdict = _reduce_directory(
                config_path, branch, write_verdict=True, record_ledger=False
            )
            results["missing_required_row"] = missing_verdict["status"]

        with _temporary_directory("round36_fixture_nonfinite_") as temporary:
            branch = temporary / "artifact"
            shutil.copytree(directory, branch)
            evidence = _read_json(branch / "evidence.json")
            manifest = _read_json(branch / "manifest.json")
            evidence["seeds"][0]["representative_response_probabilities"][0][0] = float("nan")
            raw = json.dumps(
                evidence,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=True,
            ).encode("utf-8") + b"\n"
            (branch / "evidence.json").write_bytes(raw)
            _refresh_fixture_hash(branch, manifest)
            nonfinite_verdict = _reduce_directory(
                config_path, branch, write_verdict=True, record_ledger=False
            )
            results["nonfinite_response"] = nonfinite_verdict["status"]

        with _temporary_directory("round36_fixture_successor_") as temporary:
            branch = temporary / "artifact"
            shutil.copytree(directory, branch)
            evidence = _read_json(branch / "evidence.json")
            manifest = _read_json(branch / "manifest.json")
            original_successor = evidence["seeds"][0]["primitive_successor_signatures"][1][0]
            alternatives = [
                signature
                for signature in evidence["world"]["oracle_signatures"]
                if signature != original_successor
            ]
            evidence["seeds"][0]["primitive_successor_signatures"][1][0] = alternatives[0]
            if _is_round36b(config):
                evidence["seeds"][0]["diagnostic_response_probabilities"][
                    "primitive_successors"
                ][1][0] = [float(component) for component in alternatives[0]]
            evidence["seeds"][0]["support_flags"] = _support_flags_for_seed(
                evidence["seeds"][0]
            )
            _attach_producer_counts(evidence, config)
            _write_json(branch / "evidence.json", evidence)
            _refresh_fixture_hash(branch, manifest)
            mutation_verdict = _reduce_directory(
                config_path, branch, write_verdict=True, record_ledger=False
            )
            results["rehashed_representative_successor_mutation"] = mutation_verdict[
                "status"
            ]

        if _is_round36b(config):
            with _temporary_directory("round36b_fixture_unsupported_") as temporary:
                branch = temporary / "artifact"
                shutil.copytree(directory, branch)
                evidence = _read_json(branch / "evidence.json")
                manifest = _read_json(branch / "manifest.json")
                seed = evidence["seeds"][0]
                truth_bit = seed["representative_signatures"][0][0]
                seed["representative_response_probabilities"][0][0] = (
                    0.51 if truth_bit == "1" else 0.49
                )
                seed["representative_signatures"][0] = (
                    _signature_from_probabilities(
                        seed["representative_response_probabilities"][0],
                        config["thresholds"]["signature_low"],
                        config["thresholds"]["signature_high"],
                    )
                )
                seed["support_flags"] = _support_flags_for_seed(seed)
                _attach_producer_counts(evidence, config)
                _write_json(branch / "evidence.json", evidence)
                _refresh_fixture_hash(branch, manifest)
                unsupported_verdict = _reduce_directory(
                    config_path, branch, write_verdict=True, record_ledger=False
                )
                results["behavior_fit_signature_unsupported"] = (
                    unsupported_verdict["status"]
                )
                diagnostic_gates = unsupported_verdict[
                    "confidence_free_diagnostic"
                ]["gates"]
                if not all(gate["passed"] for gate in diagnostic_gates.values()):
                    raise ContractError(
                        "correct low-confidence fixture did not pass the p>0.5 diagnostic"
                    )

            with _temporary_directory("round36b_fixture_underfit_") as temporary:
                branch = temporary / "artifact"
                shutil.copytree(directory, branch)
                evidence = _read_json(branch / "evidence.json")
                manifest = _read_json(branch / "manifest.json")
                seed = evidence["seeds"][0]
                training_logits = seed["behavior_fit"]["training"]["row_logits"]
                training_logits[-1] = -training_logits[-1]
                _refresh_fixture_behavior_summaries_from_rows(seed, config)
                _write_json(branch / "evidence.json", evidence)
                _refresh_fixture_hash(branch, manifest)
                underfit_verdict = _reduce_directory(
                    config_path, branch, write_verdict=True, record_ledger=False
                )
                results["behavior_underfit_quotient_ineligible"] = underfit_verdict[
                    "status"
                ]
                first_seed = str(config["training"]["model_seeds"][0])
                replayed_fit = underfit_verdict["behavior_fit_eligibility"][
                    "per_seed"
                ][first_seed]
                if replayed_fit != {
                    "training_correct": 21183,
                    "training_total": 21184,
                    "heldout_correct": 2240,
                    "heldout_total": 2240,
                    "passed": False,
                }:
                    raise ContractError(
                        "raw-row underfit fixture did not drive recomputed eligibility"
                    )

            with _temporary_directory("round36b_fixture_summary_tamper_") as temporary:
                branch = temporary / "artifact"
                shutil.copytree(directory, branch)
                evidence = _read_json(branch / "evidence.json")
                manifest = _read_json(branch / "manifest.json")
                evidence["seeds"][0]["behavior_fit"]["training"][
                    "full_dataset_bce"
                ] = 1.0
                _write_json(branch / "evidence.json", evidence)
                _refresh_fixture_hash(branch, manifest)
                tamper_verdict = _reduce_directory(
                    config_path, branch, write_verdict=True, record_ledger=False
                )
                results["behavior_summary_only_tamper"] = tamper_verdict["status"]
                if not any(
                    "full_dataset_bce row replay" in error
                    for error in tamper_verdict["errors"]
                ):
                    raise ContractError(
                        "summary-only tamper was not rejected by behavioral-row replay"
                    )

        expected = {
            "exact_fixture": PASS_STATUS,
            "missing_required_row": "INVALID",
            "nonfinite_response": "INVALID",
        }
        if _is_round36b(config):
            expected["behavior_underfit_quotient_ineligible"] = UNDERFIT_STATUS
            expected["behavior_summary_only_tamper"] = "INVALID"
            expected["behavior_fit_signature_unsupported"] = (
                SIGNATURE_UNSUPPORTED_STATUS
            )
        for key, expected_status in expected.items():
            if results[key] != expected_status:
                raise ContractError(f"fixture branch {key} returned {results[key]!r}")
        expected_successor_status = (
            SUPPORTED_NONCONGRUENT_PREFIX + " — QUOTIENT WELL DEFINEDNESS"
            if _is_round36b(config)
            else "FAIL — QUOTIENT WELL DEFINEDNESS"
        )
        if results["rehashed_representative_successor_mutation"] != expected_successor_status:
            raise ContractError(
                "schema-valid rehashed successor mutation did not return the exact "
                "quotient-well-definedness failure"
            )
        torch_imported_by_fixture = not torch_loaded_before and "torch" in sys.modules
        if torch_imported_by_fixture:
            raise ContractError("the no-model fixture imported torch")
        branches = {
            "exact_fixture": "FIXTURE-ONLY — exact synthetic branch accepted",
            "missing_required_row": "FIXTURE-ONLY — missing row rejected as INVALID",
            "nonfinite_response": "FIXTURE-ONLY — non-finite response rejected as INVALID",
            "rehashed_representative_successor_mutation": (
                "FIXTURE-ONLY — successor mutation rejected by QUOTIENT WELL DEFINEDNESS"
            ),
        }
        if _is_round36b(config):
            branches["behavior_fit_signature_unsupported"] = (
                "FIXTURE-ONLY — exact behavior with correct low-confidence "
                "probabilities rejected as OPERATIONAL SIGNATURE UNSUPPORTED"
            )
            branches["behavior_underfit_quotient_ineligible"] = (
                "FIXTURE-ONLY — a raw behavioral-row error made quotient "
                "interpretation ineligible"
            )
            branches["behavior_summary_only_tamper"] = (
                "FIXTURE-ONLY — rehashed producer-summary tamper rejected as INVALID"
            )
        return {
            "fixture_status": "FIXTURE-ONLY",
            "torch_imported_by_fixture": False,
            "branches": branches,
        }

    if output_dir is not None:
        return execute(output_dir)
    with _temporary_directory("round36_fixture_") as temporary:
        return execute(temporary)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    produce = subcommands.add_parser(
        "produce", help="train and serialize evidence in one non-claiming CPU process"
    )
    produce.add_argument("--config", type=Path, required=True)
    produce.add_argument("--out", type=Path, required=True)

    reduce = subcommands.add_parser("reduce", help="declaratively reduce stored evidence")
    reduce.add_argument("--config", type=Path, required=True)
    reduce.add_argument("--evidence", type=Path, required=True)

    fixture = subcommands.add_parser(
        "fixture", help="exercise synthetic reducer branches without importing Torch"
    )
    fixture.add_argument("--config", type=Path, required=True)
    fixture.add_argument("--out", type=Path, help="optional persistent exact fixture directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "produce":
            result = _produce(args.config, args.out)
        elif args.command == "reduce":
            result = _reduce_directory(
                args.config, args.evidence, write_verdict=True, record_ledger=True
            )
        else:
            result = _run_fixture(args.config, args.out)
        print(_canonical_bytes(result).decode("utf-8"), end="")
        if args.command == "reduce":
            if result["status"] == PASS_STATUS:
                return 0
            if result["status"].startswith("FAIL —"):
                return 1
            return 2
        return 0
    except BudgetExceeded as exc:
        if args.command == "produce":
            try:
                _append_ledger(
                    _ledger_entry(
                        "round36_operational_quotient_budget_exceeded",
                        "Round 36 producer exceeded the registered CPU wall",
                        args.config,
                        " ".join(sys.argv),
                        {},
                        [],
                        "producer_incomplete_budget_exceeded",
                        str(args.out),
                        "Non-claiming producer stopped; no scientific verdict was issued.",
                    )
                )
            except OSError:
                pass
        print(f"producer incomplete: {exc}", file=sys.stderr)
        return 2
    except (ContractError, OSError) as exc:
        print(f"invalid invocation/artifact: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
