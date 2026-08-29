"""Round 37 presentation-duplicated operational-quotient world.

This sibling is the reusable boundary for worlds whose hidden-state identity is
a genuinely non-trivial quotient: 32 opaque presentations but 16 operational
places.  ``produce`` trains one carrier across both predeclared presentation
role folds and writes non-claiming evidence.  ``reduce`` is a separate,
fail-closed consumer that recomputes both carriers' certificates from JSON and
never reads weights.  ``fixture`` exercises the same reducer without importing
Torch.

The Round 36 runner is intentionally not imported or modified: a Round 36d
producer may be executing while this module is installed.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SCHEMA_CONFIG = "round37-presentation-quotient-config-v1"
SCHEMA_MANIFEST = "round37-presentation-quotient-manifest-v1"
SCHEMA_EVIDENCE = "round37-presentation-quotient-evidence-v2"
SCHEMA_VERDICT = "round37-presentation-quotient-verdict-v2"
REGISTRATION_ID = "round37-presentation-duplicated-quotient-v1"

FACTORED = "factored"
UNRESTRICTED = "unrestricted"
CARRIERS = (FACTORED, UNRESTRICTED)

PASS_STATUS = "PASS — PRESENTATION-DUPLICATED OPERATIONAL QUOTIENT"
UNDERFIT_STATUS = "FAIL — BEHAVIOR UNDERFIT OR BASE SIGNATURE UNSUPPORTED"
FAIL_PAIRED = "FAIL — PAIRED PRESENTATIONS DO NOT NAME ONE PLACE"
FAIL_DESCENT = "FAIL — TASK ACTIONS DO NOT DESCEND INDEPENDENTLY OF PRESENTATION"
FAIL_TRANSFER = "FAIL — HELD-OUT PRESENTATION TRANSFER"
FAIL_INTERCHANGE = "FAIL — ROLLED INTERCHANGEABILITY"
FAIL_PRESENT = "FAIL — PRESENTATION MOVE CHANGES OPERATIONAL PLACE"
INVALID_STATUS = "INVALID"

FACTORED_ADVANTAGE = "FACTORED ADVANTAGE IN THIS MATCH"
SOLVED_BY_BOTH = "SOLVED BY BOTH — NO FACTORIZATION ADVANTAGE SHOWN"
UNRESTRICTED_ADVANTAGE = "UNRESTRICTED ADVANTAGE IN THIS MATCH"
NO_ARCHITECTURAL_WIN = "NO ARCHITECTURAL WIN"

TASK_ACTIONS = [
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
PRESENT_ACTION = "present"
ALL_ACTIONS = TASK_ACTIONS + [PRESENT_ACTION]
TOGGLE_ACTIONS = TASK_ACTIONS[1:5]
SWAP_ACTIONS = TASK_ACTIONS[5:]
TRACE_STEPS = list(range(0, 32001, 1000))
HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(RuntimeError):
    """A config or serialized artifact violates the frozen contract."""


class BudgetExceeded(RuntimeError):
    """A registered producer wall was exceeded."""


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


def _module_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


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


def _expect_probability(value: Any, where: str) -> float:
    number = _expect_number(value, where)
    if not 0.0 <= number <= 1.0:
        raise ContractError(f"{where} must lie in [0,1]")
    return number


def _expect_sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or not HEX_RE.fullmatch(value):
        raise ContractError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _expect_list(value: Any, length: int, where: str) -> list[Any]:
    if not isinstance(value, list) or len(value) != length:
        actual = len(value) if isinstance(value, list) else type(value).__name__
        raise ContractError(f"{where} must be a list of length {length}; got {actual}")
    return value


def _expect_signature(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 12 or any(
        component not in "01?" for component in value
    ):
        raise ContractError(f"{where} must be a 12-component signature")
    return value


def _validate_config(value: Any) -> dict[str, Any]:
    root = _require_keys(
        value,
        [
            "schema_version",
            "registration_id",
            "name",
            "registration",
            "world",
            "actions",
            "split",
            "model",
            "training",
            "thresholds",
        ],
        "config",
    )
    _expect(root["schema_version"], SCHEMA_CONFIG, "config.schema_version")
    _expect(root["registration_id"], REGISTRATION_ID, "config.registration_id")
    _expect(root["name"], "presentation_quotient_v1", "config.name")
    _expect(
        root["registration"],
        "Round 37 — presentation-duplicated quotient world",
        "config.registration",
    )

    world = _require_keys(
        root["world"],
        [
            "operational_bit_count",
            "presentation_bit_count",
            "hidden_state_count",
            "operational_place_count",
            "representatives_per_place",
            "response_bit",
            "data_seed",
            "handle_permutation",
        ],
        "config.world",
    )
    expected_world = {
        "operational_bit_count": 4,
        "presentation_bit_count": 1,
        "hidden_state_count": 32,
        "operational_place_count": 16,
        "representatives_per_place": 2,
        "response_bit": 1,
        "data_seed": 3701,
        "handle_permutation": "numpy-pcg64-permutation-v1",
    }
    for key, expected in expected_world.items():
        _expect(world[key], expected, f"config.world.{key}")

    actions = _expect_list(root["actions"], 12, "config.actions")
    expected_actions = [
        ("no-op", "no-op", []),
        ("toggle(1)", "toggle", [1]),
        ("toggle(2)", "toggle", [2]),
        ("toggle(3)", "toggle", [3]),
        ("toggle(4)", "toggle", [4]),
        ("swap(1,2)", "swap", [1, 2]),
        ("swap(1,3)", "swap", [1, 3]),
        ("swap(1,4)", "swap", [1, 4]),
        ("swap(2,3)", "swap", [2, 3]),
        ("swap(2,4)", "swap", [2, 4]),
        ("swap(3,4)", "swap", [3, 4]),
        ("present", "presentation", [1]),
    ]
    for index, (name, kind, indices) in enumerate(expected_actions):
        action = _require_keys(actions[index], ["name", "kind", "indices"], f"config.actions[{index}]")
        _expect(action["name"], name, f"config.actions[{index}].name")
        _expect(action["kind"], kind, f"config.actions[{index}].kind")
        _expect(action["indices"], indices, f"config.actions[{index}].indices")

    split = _require_keys(
        root["split"],
        [
            "max_task_word_depth",
            "identity_response_depth",
            "h2_salt",
            "h2_per_first_action",
            "h3_salt",
            "h3_per_first_action",
            "expected_base_training_words",
            "expected_h2_words",
            "expected_h3_words",
            "expected_heldout_task_words",
            "expected_base_training_rows",
            "expected_role_selected_rows",
            "expected_presentation_training_words",
            "expected_presentation_training_rows",
            "expected_training_rows_per_role",
            "expected_primary_heldout_rows_per_role",
            "role_folds",
            "canonical_row_order",
            "training_list_sha256",
            "h2_list_sha256",
            "h3_list_sha256",
        ],
        "config.split",
    )
    expected_split = {
        "max_task_word_depth": 3,
        "identity_response_depth": 1,
        "h2_salt": "round36-h2-v1",
        "h2_per_first_action": 2,
        "h3_salt": "round36-h3-v1",
        "h3_per_first_action": 6,
        "expected_base_training_words": 1324,
        "expected_h2_words": 74,
        "expected_h3_words": 66,
        "expected_heldout_task_words": 140,
        "expected_base_training_rows": 42368,
        "expected_role_selected_rows": 2240,
        "expected_presentation_training_words": 23,
        "expected_presentation_training_rows": 736,
        "expected_training_rows_per_role": 45344,
        "expected_primary_heldout_rows_per_role": 2240,
        "role_folds": [0, 1],
        "canonical_row_order": "section-word-major-handle-minor-v1",
        "training_list_sha256": "4ae7deba9133776503a0c7cb30811c595c09eb04fea6f56c266ed2d916929926",
        "h2_list_sha256": "679a9ec46e134b1163916751ac056d27de85f61b677b924861fdc0ed2273e5fe",
        "h3_list_sha256": "6fb5e7f366faba21c47f9e4249c0f43ade85acd02a29a1b7d91e4b6b2b42c784",
    }
    for key, expected in expected_split.items():
        _expect(split[key], expected, f"config.split.{key}")

    model = _require_keys(
        root["model"],
        [
            "latent_dim",
            "state_table_rows",
            "response_outputs",
            "transition_family",
            "factored",
            "unrestricted",
        ],
        "config.model",
    )
    for key, expected in {
        "latent_dim": 8,
        "state_table_rows": 32,
        "response_outputs": 1,
        "transition_family": "residual-tanh-v1",
    }.items():
        _expect(model[key], expected, f"config.model.{key}")
    factored = _require_keys(
        model["factored"],
        [
            "operational_latent_dim",
            "presentation_latent_dim",
            "task_hidden_width",
            "presentation_hidden_width",
            "active_trainable_parameters",
        ],
        "config.model.factored",
    )
    for key, expected in {
        "operational_latent_dim": 6,
        "presentation_latent_dim": 2,
        "task_hidden_width": 48,
        "presentation_hidden_width": 16,
        "active_trainable_parameters": 1503,
    }.items():
        _expect(factored[key], expected, f"config.model.factored.{key}")
    unrestricted = _require_keys(
        model["unrestricted"],
        ["transition_hidden_width", "active_trainable_parameters"],
        "config.model.unrestricted",
    )
    _expect(unrestricted["transition_hidden_width"], 64, "config.model.unrestricted.transition_hidden_width")
    _expect(unrestricted["active_trainable_parameters"], 2129, "config.model.unrestricted.active_trainable_parameters")

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
            "component_trace_interval",
            "target_cpu_minutes_full_matrix",
            "cell_hard_wall_seconds",
            "total_hard_wall_seconds",
        ],
        "config.training",
    )
    expected_training = {
        "model_seeds": [11, 23, 37, 53, 71],
        "optimizer": "AdamW",
        "learning_rate": 0.003,
        "weight_decay": 0.00001,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "batch_size": 512,
        "optimizer_steps_per_seed": 32000,
        "loss": "binary_cross_entropy_with_logits",
        "device": "cpu",
        "threads": 1,
        "deterministic_algorithms": True,
        "evaluation_batch_size": 4096,
        "component_trace_interval": 1000,
        "target_cpu_minutes_full_matrix": [24, 32],
        "cell_hard_wall_seconds": 720,
        "total_hard_wall_seconds": 2700,
    }
    for key, expected in expected_training.items():
        _expect(training[key], expected, f"config.training.{key}")

    thresholds = _require_keys(
        root["thresholds"],
        [
            "behavior_decision_strictly_greater_than",
            "signature_low",
            "signature_high",
            "paired_place_fraction",
            "task_descent_fraction",
            "heldout_transfer_signature_fraction",
            "heldout_transfer_response_fraction",
            "rolled_interchangeability_fraction",
            "presentation_place_fraction",
            "required_seed_role_fraction",
        ],
        "config.thresholds",
    )
    expected_thresholds = {
        "behavior_decision_strictly_greater_than": 0.5,
        "signature_low": 0.10,
        "signature_high": 0.90,
        "paired_place_fraction": 1.0,
        "task_descent_fraction": 1.0,
        "heldout_transfer_signature_fraction": 1.0,
        "heldout_transfer_response_fraction": 1.0,
        "rolled_interchangeability_fraction": 1.0,
        "presentation_place_fraction": 1.0,
        "required_seed_role_fraction": 1.0,
    }
    for key, expected in expected_thresholds.items():
        _expect(thresholds[key], expected, f"config.thresholds.{key}")
    return root


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


def _operational_states() -> list[tuple[int, int, int, int]]:
    return list(itertools.product((0, 1), repeat=4))


def _hidden_states() -> list[tuple[tuple[int, int, int, int], int]]:
    return [(q, p) for q in _operational_states() for p in (0, 1)]


def _q_index(q: Sequence[int]) -> int:
    return sum(int(bit) << (3 - position) for position, bit in enumerate(q))


def _apply_task_action(q: Sequence[int], action: str) -> tuple[int, int, int, int]:
    result = list(q)
    if action == "no-op":
        pass
    elif action.startswith("toggle("):
        index = int(action[7:-1]) - 1
        result[index] = 1 - result[index]
    elif action.startswith("swap("):
        left, right = (int(part) - 1 for part in action[5:-1].split(","))
        result[left], result[right] = result[right], result[left]
    else:
        raise ContractError(f"unknown task action {action!r}")
    return tuple(result)  # type: ignore[return-value]


def _apply_word(
    state: tuple[Sequence[int], int], word: Sequence[str]
) -> tuple[tuple[int, int, int, int], int]:
    q = tuple(state[0])
    p = int(state[1])
    for action in word:
        if action == PRESENT_ACTION:
            p = 1 - p
        else:
            q = _apply_task_action(q, action)
    return q, p  # type: ignore[return-value]


def _oracle_signature(q: Sequence[int]) -> str:
    probes: list[tuple[str, ...]] = [tuple()] + [(action,) for action in TASK_ACTIONS]
    return "".join(str(_apply_word((q, 0), word)[0][0]) for word in probes)


def _word_spelling(word: Sequence[str]) -> str:
    return ">".join(word)


def _spelling_word(spelling: str) -> tuple[str, ...]:
    if spelling == "":
        return tuple()
    word = tuple(spelling.split(">"))
    if any(action not in ALL_ACTIONS for action in word):
        raise ContractError(f"invalid canonical word spelling {spelling!r}")
    return word


def _salted_word_hash(salt: str, word: Sequence[str]) -> str:
    return _sha256_bytes(f"{salt}|{_word_spelling(word)}".encode("utf-8"))


def _word_universe(config: dict[str, Any]) -> dict[str, list[tuple[str, ...]]]:
    by_depth = {
        depth: list(itertools.product(TASK_ACTIONS, repeat=depth))
        for depth in range(4)
    }
    forced_h2: set[tuple[str, ...]] = {
        (f"toggle({index})", f"toggle({index})") for index in range(1, 5)
    }
    for swap in SWAP_ACTIONS:
        for toggle in TOGGLE_ACTIONS:
            forced_h2.add((swap, toggle))
            forced_h2.add((toggle, swap))
    remaining = [word for word in by_depth[2] if word not in forced_h2]
    h2_set = set(forced_h2)
    for first in TASK_ACTIONS:
        stratum = [word for word in remaining if word[0] == first]
        stratum.sort(key=lambda word: (_salted_word_hash(config["split"]["h2_salt"], word), word))
        h2_set.update(stratum[: config["split"]["h2_per_first_action"]])
    h3_set: set[tuple[str, ...]] = set()
    for first in TASK_ACTIONS:
        stratum = [word for word in by_depth[3] if word[0] == first]
        stratum.sort(key=lambda word: (_salted_word_hash(config["split"]["h3_salt"], word), word))
        h3_set.update(stratum[: config["split"]["h3_per_first_action"]])
    h2 = [word for word in by_depth[2] if word in h2_set]
    h3 = [word for word in by_depth[3] if word in h3_set]
    training = (
        by_depth[0]
        + by_depth[1]
        + [word for word in by_depth[2] if word not in h2_set]
        + [word for word in by_depth[3] if word not in h3_set]
    )
    heldout = h2 + h3
    expected = config["split"]
    _expect(len(training), expected["expected_base_training_words"], "derived training-word count")
    _expect(len(h2), expected["expected_h2_words"], "derived H_2 count")
    _expect(len(h3), expected["expected_h3_words"], "derived H_3 count")
    _expect(len(heldout), expected["expected_heldout_task_words"], "derived held-out count")
    return {"training": training, "h2": h2, "h3": h3, "heldout": heldout}


def _word_pack(config: dict[str, Any]) -> dict[str, Any]:
    universe = _word_universe(config)
    result: dict[str, Any] = {}
    for name in ("training", "h2", "h3"):
        spellings = [_word_spelling(word) for word in universe[name]]
        result[name] = spellings
        result[f"{name}_list_sha256"] = _sha256_bytes(_canonical_bytes(spellings))
    for name in ("training", "h2", "h3"):
        _expect(
            result[f"{name}_list_sha256"],
            config["split"][f"{name}_list_sha256"],
            f"derived {name} list SHA-256",
        )
    return result


def _handle_to_hidden(config: dict[str, Any]) -> list[int]:
    import numpy as np

    generator = np.random.Generator(np.random.PCG64(config["world"]["data_seed"]))
    return [int(value) for value in generator.permutation(32)]


def _world_truth(config: dict[str, Any]) -> dict[str, Any]:
    states = _hidden_states()
    handle_to_hidden = _handle_to_hidden(config)
    hidden_to_handle = {state_index: handle for handle, state_index in enumerate(handle_to_hidden)}
    pairs: list[list[int]] = []
    for q_index in range(16):
        pairs.append([hidden_to_handle[2 * q_index], hidden_to_handle[2 * q_index + 1]])
    oracle_signatures = [_oracle_signature(q) for q in _operational_states()]
    if len(set(oracle_signatures)) != 16:
        raise ContractError("oracle task signatures do not separate the 16 places")
    oracle_task_successors = [
        [_q_index(_apply_task_action(q, action)) for action in TASK_ACTIONS]
        for q in _operational_states()
    ]
    return {
        "hidden_states": [list(q) + [p] for q, p in states],
        "handle_to_hidden_state": handle_to_hidden,
        "operational_pairs": pairs,
        "oracle_signatures": oracle_signatures,
        "oracle_task_successor_places": oracle_task_successors,
    }


def _row_key(handle: int, word: Sequence[str]) -> str:
    return f"h{handle:02d}|{_word_spelling(word)}"


def _parse_row_key(value: Any) -> tuple[int, tuple[str, ...]]:
    if not isinstance(value, str) or not re.fullmatch(r"h\d{2}\|.*", value):
        raise ContractError(f"invalid row key {value!r}")
    handle = int(value[1:3])
    if not 0 <= handle < 32:
        raise ContractError(f"row-key handle out of range: {handle}")
    return handle, _spelling_word(value[4:])


def _role_splits(config: dict[str, Any], world: dict[str, Any]) -> list[dict[str, Any]]:
    universe = _word_universe(config)
    presentation_words = [(PRESENT_ACTION,)]
    for action in TASK_ACTIONS:
        presentation_words.append((PRESENT_ACTION, action))
        presentation_words.append((action, PRESENT_ACTION))
    _expect(len(presentation_words), 23, "presentation training-word count")
    states = _hidden_states()
    result: list[dict[str, Any]] = []
    for role in config["split"]["role_folds"]:
        train_handles = [
            handle
            for handle, state_index in enumerate(world["handle_to_hidden_state"])
            if states[state_index][1] == role
        ]
        heldout_handles = [
            handle
            for handle, state_index in enumerate(world["handle_to_hidden_state"])
            if states[state_index][1] == 1 - role
        ]
        training_rows = [
            _row_key(handle, word)
            for word in universe["training"]
            for handle in range(32)
        ]
        training_rows.extend(
            _row_key(handle, word)
            for word in universe["heldout"]
            for handle in train_handles
        )
        training_rows.extend(
            _row_key(handle, word)
            for word in presentation_words
            for handle in range(32)
        )
        heldout_rows = [
            _row_key(handle, word)
            for word in universe["heldout"]
            for handle in heldout_handles
        ]
        _expect(len(training_rows), config["split"]["expected_training_rows_per_role"], f"role {role} training rows")
        _expect(len(heldout_rows), config["split"]["expected_primary_heldout_rows_per_role"], f"role {role} held-out rows")
        if len(set(training_rows)) != len(training_rows):
            raise ContractError(f"role {role} training rows contain duplicates")
        if set(training_rows).intersection(heldout_rows):
            raise ContractError(f"role {role} training and held-out rows overlap")
        result.append(
            {
                "role": role,
                "training_rows": training_rows,
                "training_rows_sha256": _sha256_bytes(_canonical_bytes(training_rows)),
                "heldout_rows": heldout_rows,
                "heldout_rows_sha256": _sha256_bytes(_canonical_bytes(heldout_rows)),
            }
        )
    return result


def _target_for_row(row_key: str, world: dict[str, Any]) -> int:
    handle, word = _parse_row_key(row_key)
    state_index = world["handle_to_hidden_state"][handle]
    q, p = _hidden_states()[state_index]
    endpoint_q, _ = _apply_word((q, p), word)
    return int(endpoint_q[0])


def _prepare_empty_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ContractError(f"output path exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise ContractError(f"output directory must be absent or empty: {path}")
    else:
        path.mkdir(parents=True)


@contextlib.contextmanager
def _temporary_directory(prefix: str) -> Iterator[Path]:
    directory = Path(tempfile.gettempdir()) / f"{prefix}{uuid.uuid4().hex}"
    directory.mkdir(parents=False, exist_ok=False)
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def _import_producer_dependencies(config: dict[str, Any]) -> tuple[Any, Any]:
    inherited_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if inherited_cuda != "":
        raise ContractError("CUDA_VISIBLE_DEVICES must be absent or exactly empty")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    import numpy as np
    import torch

    torch.set_num_threads(config["training"]["threads"])
    try:
        torch.set_num_interop_threads(config["training"]["threads"])
    except RuntimeError as exc:
        raise ContractError(f"cannot lock Torch interop threads: {exc}") from exc
    torch.use_deterministic_algorithms(True)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise ContractError("Torch did not accept the registered single-thread lock")
    return np, torch


def _make_model(torch: Any, config: dict[str, Any], carrier: str) -> Any:
    nn = torch.nn

    class FactoredModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Embedding(32, 8)
            self.task_w_z = nn.Linear(6, 48, bias=False)
            self.task_action_embedding = nn.Embedding(11, 48)
            self.task_b1 = nn.Parameter(torch.zeros(48))
            self.task_w2 = nn.Linear(48, 6, bias=True)
            self.presentation_w_z = nn.Linear(2, 16, bias=False)
            self.presentation_b1 = nn.Parameter(torch.zeros(16))
            self.presentation_w2 = nn.Linear(16, 2, bias=True)
            self.response = nn.Linear(6, 1)

        def transition(self, latent: Any, action_ids: Any) -> Any:
            q_latent, p_latent = latent[:, :6], latent[:, 6:]
            safe_task_ids = torch.clamp(action_ids, max=10)
            task_hidden = torch.tanh(
                self.task_w_z(q_latent)
                + self.task_action_embedding(safe_task_ids)
                + self.task_b1
            )
            task_next = torch.cat(
                [q_latent + self.task_w2(task_hidden), p_latent], dim=1
            )
            presentation_hidden = torch.tanh(
                self.presentation_w_z(p_latent) + self.presentation_b1
            )
            presentation_next = torch.cat(
                [q_latent, p_latent + self.presentation_w2(presentation_hidden)],
                dim=1,
            )
            return torch.where(
                (action_ids == 11).unsqueeze(1), presentation_next, task_next
            )

        def response_logits(self, latent: Any) -> Any:
            return self.response(latent[:, :6]).squeeze(1)

        def run_latent(self, latent: Any, action_ids: Any, lengths: Any) -> Any:
            for position in range(action_ids.shape[1]):
                proposed = self.transition(latent, action_ids[:, position])
                latent = torch.where(
                    (lengths > position).unsqueeze(1), proposed, latent
                )
            return latent

        def run_word(self, handles: Any, action_ids: Any, lengths: Any) -> Any:
            return self.response_logits(
                self.run_latent(self.encoder(handles), action_ids, lengths)
            )

    class UnrestrictedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Embedding(32, 8)
            self.w_z = nn.Linear(8, 64, bias=False)
            self.action_embedding = nn.Embedding(12, 64)
            self.b1 = nn.Parameter(torch.zeros(64))
            self.w2 = nn.Linear(64, 8, bias=True)
            self.response = nn.Linear(8, 1)

        def transition(self, latent: Any, action_ids: Any) -> Any:
            hidden = torch.tanh(
                self.w_z(latent) + self.action_embedding(action_ids) + self.b1
            )
            return latent + self.w2(hidden)

        def response_logits(self, latent: Any) -> Any:
            return self.response(latent).squeeze(1)

        def run_latent(self, latent: Any, action_ids: Any, lengths: Any) -> Any:
            for position in range(action_ids.shape[1]):
                proposed = self.transition(latent, action_ids[:, position])
                latent = torch.where(
                    (lengths > position).unsqueeze(1), proposed, latent
                )
            return latent

        def run_word(self, handles: Any, action_ids: Any, lengths: Any) -> Any:
            return self.response_logits(
                self.run_latent(self.encoder(handles), action_ids, lengths)
            )

    if carrier == FACTORED:
        return FactoredModel()
    if carrier == UNRESTRICTED:
        return UnrestrictedModel()
    raise ContractError(f"unknown carrier {carrier!r}")


def _parameter_count(model: Any) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)


def _assert_cpu_model(model: Any) -> list[str]:
    devices = sorted({str(parameter.device) for parameter in model.parameters()})
    if devices != ["cpu"]:
        raise ContractError(f"all parameters must remain on CPU; got {devices}")
    return devices


def _state_dict_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(_canonical_bytes(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _arrays_for_rows(np: Any, rows: Sequence[str], world: dict[str, Any]) -> dict[str, Any]:
    action_ids = {action: index for index, action in enumerate(ALL_ACTIONS)}
    handles: list[int] = []
    words: list[list[int]] = []
    lengths: list[int] = []
    targets: list[float] = []
    for row in rows:
        handle, word = _parse_row_key(row)
        handles.append(handle)
        words.append([action_ids[action] for action in word] + [0] * (3 - len(word)))
        lengths.append(len(word))
        targets.append(float(_target_for_row(row, world)))
    return {
        "handles": np.asarray(handles, dtype=np.int64),
        "words": np.asarray(words, dtype=np.int64),
        "lengths": np.asarray(lengths, dtype=np.int64),
        "targets": np.asarray(targets, dtype=np.float32),
    }


def _check_deadline(deadline: float, label: str) -> None:
    if time.monotonic() > deadline:
        raise BudgetExceeded(f"registered 12-minute wall exceeded for {label}")


def _predict_rows(
    torch: Any,
    model: Any,
    arrays: dict[str, Any],
    batch_size: int,
    deadline: float,
    label: str,
) -> list[float]:
    handles = torch.from_numpy(arrays["handles"])
    words = torch.from_numpy(arrays["words"])
    lengths = torch.from_numpy(arrays["lengths"])
    probabilities: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(handles), batch_size):
            _check_deadline(deadline, label)
            end = min(start + batch_size, len(handles))
            logits = model.run_word(handles[start:end], words[start:end], lengths[start:end])
            probabilities.extend(float(value) for value in torch.sigmoid(logits).cpu())
    model.train()
    return probabilities


def _signature_probabilities(
    torch: Any,
    model: Any,
    latents: Any,
    batch_size: int,
    deadline: float,
    label: str,
) -> list[list[float]]:
    rows: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(latents.shape[0]), batch_size):
            _check_deadline(deadline, label)
            batch = latents[start : start + batch_size]
            columns = [torch.sigmoid(model.response_logits(batch))]
            for action_id in range(11):
                ids = torch.full((batch.shape[0],), action_id, dtype=torch.long)
                moved = model.transition(batch, ids)
                columns.append(torch.sigmoid(model.response_logits(moved)))
            matrix = torch.stack(columns, dim=1).cpu().tolist()
            rows.extend([[float(value) for value in row] for row in matrix])
    model.train()
    return rows


def _component_trace_record(
    torch: Any,
    model: Any,
    config: dict[str, Any],
    world: dict[str, Any],
    step: int,
    batch_loss: float | None,
    deadline: float,
    label: str,
) -> dict[str, Any]:
    handles = torch.arange(32, dtype=torch.long)
    latents = model.encoder(handles)
    probabilities = _signature_probabilities(
        torch,
        model,
        latents,
        config["training"]["evaluation_batch_size"],
        deadline,
        label,
    )
    low = config["thresholds"]["signature_low"]
    high = config["thresholds"]["signature_high"]
    losses: list[float] = []
    supported: list[bool] = []
    flattened: list[float] = []
    for handle, row in enumerate(probabilities):
        q_index = world["handle_to_hidden_state"][handle] // 2
        target_signature = world["oracle_signatures"][q_index]
        for probability, bit in zip(row, target_signature):
            clipped = min(max(probability, 1e-12), 1.0 - 1e-12)
            target = int(bit)
            losses.append(float(-(target * math.log(clipped) + (1 - target) * math.log(1.0 - clipped))))
            supported.append(probability <= low or probability >= high)
            flattened.append(probability)
    return {
        "step": step,
        "batch_loss": batch_loss,
        "base_signature_probabilities": flattened,
        "base_signature_component_bce": losses,
        "base_signature_component_supported": supported,
    }


def _heldout_step_evaluation(
    torch: Any,
    model: Any,
    arrays: dict[str, Any],
    batch_size: int,
    low: float,
    high: float,
    deadline: float,
    label: str,
) -> tuple[list[list[list[float]]], list[list[list[str]]]]:
    handles = torch.from_numpy(arrays["handles"])
    words = torch.from_numpy(arrays["words"])
    lengths = torch.from_numpy(arrays["lengths"])
    length_values = [int(value) for value in arrays["lengths"]]
    signatures_by_row: list[list[str]] = [[] for _ in length_values]
    final_probabilities: list[list[float] | None] = [None] * len(length_values)
    model.eval()
    with torch.no_grad():
        latents = model.encoder(handles)
        configured_max_depth = int(words.shape[1])
        if configured_max_depth != 3:
            raise ContractError("held-out step evaluation requires max depth 3")
        for position in range(configured_max_depth):
            _check_deadline(deadline, label)
            proposed = model.transition(latents, words[:, position])
            active_mask = lengths > position
            latents = torch.where(active_mask.unsqueeze(1), proposed, latents)
            active_rows = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
            step_probabilities = _signature_probabilities(
                torch, model, latents[active_rows], batch_size, deadline, label
            )
            for active_index, row_index_value in enumerate(active_rows.tolist()):
                row_index = int(row_index_value)
                row_length = length_values[row_index]
                probabilities = step_probabilities[active_index]
                signatures_by_row[row_index].append(
                    _signature(probabilities, low, high)
                )
                if row_length == position + 1:
                    final_probabilities[row_index] = probabilities
    model.train()
    if any(value is None for value in final_probabilities):
        raise ContractError("held-out step evaluation did not reach every endpoint")
    final_rows = [value for value in final_probabilities if value is not None]
    if len(final_rows) % 2 != 0:
        raise ContractError("held-out step evaluation requires presentation pairs")
    endpoint_pairs: list[list[list[float]]] = []
    step_pairs: list[list[list[str]]] = []
    for row_index in range(0, len(final_rows), 2):
        if len(signatures_by_row[row_index]) != len(signatures_by_row[row_index + 1]):
            raise ContractError("paired presentations have unequal continuation depths")
        endpoint_pairs.append([final_rows[row_index], final_rows[row_index + 1]])
        step_pairs.append(
            [
                [
                    signatures_by_row[row_index][step],
                    signatures_by_row[row_index + 1][step],
                ]
                for step in range(len(signatures_by_row[row_index]))
            ]
        )
    return endpoint_pairs, step_pairs


def _evaluate_seed(
    np: Any,
    torch: Any,
    model: Any,
    config: dict[str, Any],
    world: dict[str, Any],
    split: dict[str, Any],
    training_arrays: dict[str, Any],
    deadline: float,
    label: str,
) -> dict[str, Any]:
    batch_size = config["training"]["evaluation_batch_size"]
    training_probabilities = _predict_rows(
        torch, model, training_arrays, batch_size, deadline, label
    )

    model.eval()
    with torch.no_grad():
        base_latents = model.encoder(torch.arange(32, dtype=torch.long))
    base_probabilities = _signature_probabilities(
        torch, model, base_latents, batch_size, deadline, label
    )

    descent_latents: list[Any] = []
    with torch.no_grad():
        for q_index in range(16):
            for action_id in range(11):
                for presentation in (0, 1):
                    handle = world["operational_pairs"][q_index][presentation]
                    latent = model.encoder(torch.tensor([handle], dtype=torch.long))
                    task_ids = torch.tensor([action_id], dtype=torch.long)
                    present_ids = torch.tensor([11], dtype=torch.long)
                    direct = model.transition(latent, task_ids)
                    before = model.transition(model.transition(latent, present_ids), task_ids)
                    after = model.transition(model.transition(latent, task_ids), present_ids)
                    descent_latents.extend([direct, before, after])
    descent_matrix = torch.cat(descent_latents, dim=0)
    flat_descent = _signature_probabilities(
        torch, model, descent_matrix, batch_size, deadline, label
    )
    task_descent: list[list[list[list[list[float]]]]] = []
    cursor = 0
    for _q_index in range(16):
        q_rows: list[list[list[list[float]]]] = []
        for _action_id in range(11):
            action_rows: list[list[list[float]]] = []
            for _presentation in (0, 1):
                action_rows.append(flat_descent[cursor : cursor + 3])
                cursor += 3
            q_rows.append(action_rows)
        task_descent.append(q_rows)

    heldout_words = _word_universe(config)["heldout"]
    pair_rows = [
        _row_key(world["operational_pairs"][q_index][presentation], word)
        for word in heldout_words
        for q_index in range(16)
        for presentation in (0, 1)
    ]
    pair_arrays = _arrays_for_rows(np, pair_rows, world)
    heldout_pair_probabilities, heldout_pair_step_signatures = (
        _heldout_step_evaluation(
            torch,
            model,
            pair_arrays,
            batch_size,
            config["thresholds"]["signature_low"],
            config["thresholds"]["signature_high"],
            deadline,
            label,
        )
    )

    with torch.no_grad():
        present_ids = torch.full((32,), 11, dtype=torch.long)
        presented_latents = model.transition(base_latents, present_ids)
    presentation_move_probabilities = _signature_probabilities(
        torch, model, presented_latents, batch_size, deadline, label
    )
    model.train()
    evaluation = {
        "training_probabilities": training_probabilities,
        "base_signature_probabilities": base_probabilities,
        "task_descent_probabilities": task_descent,
        "heldout_pair_probabilities": heldout_pair_probabilities,
        "heldout_pair_step_signatures": heldout_pair_step_signatures,
        "presentation_move_probabilities": presentation_move_probabilities,
    }
    evaluation["endpoint_signatures"] = _reported_endpoint_signatures(
        evaluation,
        config["thresholds"]["signature_low"],
        config["thresholds"]["signature_high"],
    )
    return evaluation


def _train_carrier(
    config: dict[str, Any],
    carrier: str,
    output_dir: Path,
    np: Any,
    torch: Any,
    world: dict[str, Any],
    splits: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[float]]:
    saved_arrays: dict[str, Any] = {}
    role_records: list[dict[str, Any]] = []
    cell_walls: list[float] = []
    parameter_devices: set[str] = set()
    for split in splits:
        role = split["role"]
        cell_started = time.monotonic()
        deadline = cell_started + config["training"]["cell_hard_wall_seconds"]
        label = f"{carrier}/role-{role}"
        training_arrays = _arrays_for_rows(np, split["training_rows"], world)
        seed_records: list[dict[str, Any]] = []
        for seed in config["training"]["model_seeds"]:
            _check_deadline(deadline, label)
            torch.manual_seed(seed)
            model = _make_model(torch, config, carrier).cpu()
            parameter_devices.update(_assert_cpu_model(model))
            active_parameters = _parameter_count(model)
            _expect(
                active_parameters,
                config["model"][carrier]["active_trainable_parameters"],
                f"{carrier} active parameter count",
            )
            initial_weight_sha256 = _state_dict_sha256(model)
            initial_encoder_sha256 = _tensor_sha256(model.encoder.weight)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config["training"]["learning_rate"],
                weight_decay=config["training"]["weight_decay"],
                betas=tuple(config["training"]["betas"]),
                eps=config["training"]["epsilon"],
            )
            handles = torch.from_numpy(training_arrays["handles"])
            words = torch.from_numpy(training_arrays["words"])
            lengths = torch.from_numpy(training_arrays["lengths"])
            targets = torch.from_numpy(training_arrays["targets"])
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            loss_trace: list[float] = []
            minibatch_index_digest = hashlib.sha256()
            component_trace = [
                _component_trace_record(
                    torch, model, config, world, 0, None, deadline, label
                )
            ]
            for step in range(config["training"]["optimizer_steps_per_seed"]):
                if step % 25 == 0:
                    _check_deadline(deadline, label)
                indices = torch.randint(
                    0,
                    handles.shape[0],
                    (config["training"]["batch_size"],),
                    generator=generator,
                )
                minibatch_index_digest.update(
                    indices.detach().cpu().contiguous().numpy().tobytes(order="C")
                )
                optimizer.zero_grad(set_to_none=True)
                logits = model.run_word(handles[indices], words[indices], lengths[indices])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets[indices]
                )
                loss.backward()
                optimizer.step()
                loss_trace.append(float(loss.detach().cpu()))
                completed = step + 1
                if completed % config["training"]["component_trace_interval"] == 0:
                    component_trace.append(
                        _component_trace_record(
                            torch,
                            model,
                            config,
                            world,
                            completed,
                            loss_trace[-1],
                            deadline,
                            label,
                        )
                    )
            final_weight_sha256 = _state_dict_sha256(model)
            evaluation = _evaluate_seed(
                np,
                torch,
                model,
                config,
                world,
                split,
                training_arrays,
                deadline,
                label,
            )
            for name, tensor in model.state_dict().items():
                key = f"role_{role}__seed_{seed}__{name.replace('.', '_')}"
                saved_arrays[key] = tensor.detach().cpu().numpy()
            loss_array = np.asarray(loss_trace, dtype=np.float64)
            saved_arrays[f"role_{role}__seed_{seed}__loss_trace"] = loss_array
            seed_records.append(
                {
                    "seed": seed,
                    "steps": len(loss_trace),
                    "active_trainable_parameter_count": active_parameters,
                    "initial_weight_sha256": initial_weight_sha256,
                    "initial_encoder_sha256": initial_encoder_sha256,
                    "final_weight_sha256": final_weight_sha256,
                    "minibatch_index_stream_sha256": minibatch_index_digest.hexdigest(),
                    "loss_trace_sha256": _sha256_bytes(loss_array.tobytes(order="C")),
                    "component_trace": component_trace,
                    "evaluation": evaluation,
                }
            )
            del optimizer, model
        cell_wall = time.monotonic() - cell_started
        if cell_wall > config["training"]["cell_hard_wall_seconds"]:
            raise BudgetExceeded(f"registered 12-minute wall exceeded for {label}")
        cell_walls.append(cell_wall)
        role_records.append(
            {
                "role": role,
                "cell_wall_seconds": cell_wall,
                "seeds": seed_records,
            }
        )
    weights_path = output_dir / "weights.npz"
    np.savez_compressed(weights_path, **saved_arrays)
    return role_records, {
        "parameter_devices": sorted(parameter_devices),
        "weights_sha256": _sha256_file(weights_path),
    }, cell_walls


def _dependency_versions(np: Any | None = None, torch: Any | None = None) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": "not-imported" if np is None else str(np.__version__),
        "torch": "not-imported" if torch is None else str(torch.__version__),
    }


def _platform_info() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _append_ledger(event: dict[str, Any]) -> None:
    ledger = Path(__file__).resolve().parent / "ledger.jsonl"
    with ledger.open("ab") as handle:
        handle.write(_canonical_bytes(event))


def _producer_manifest(
    config: dict[str, Any],
    config_bytes: bytes,
    carrier: str,
    output_dir: Path,
    evidence: dict[str, Any],
    training_meta: dict[str, Any],
    cell_walls: list[float],
    np: Any,
    torch: Any,
    started_at: str,
    wall_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_MANIFEST,
        "registration_id": REGISTRATION_ID,
        "artifact_kind": "learned",
        "claiming": False,
        "carrier": carrier,
        "producer_code_sha256": _module_sha256(),
        "config_sha256": _sha256_bytes(config_bytes),
        "evidence_sha256": _sha256_file(output_dir / "evidence.json"),
        "weights_sha256": training_meta["weights_sha256"],
        "git_commit": _git_commit(),
        "command": " ".join(sys.argv),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "wall_seconds": wall_seconds,
        "role_cell_wall_seconds": cell_walls,
        "platform": _platform_info(),
        "dependencies": _dependency_versions(np, torch),
        "cpu_settings": {
            "device": config["training"]["device"],
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "parameter_devices": training_meta["parameter_devices"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "action_order": ALL_ACTIONS,
        "word_list_sha256": {
            key: evidence["word_lists"][key]
            for key in ("training_list_sha256", "h2_list_sha256", "h3_list_sha256")
        },
        "role_split_sha256": [
            {
                "role": split["role"],
                "training_rows_sha256": split["training_rows_sha256"],
                "heldout_rows_sha256": split["heldout_rows_sha256"],
            }
            for split in evidence["role_splits"]
        ],
        "data_seed": config["world"]["data_seed"],
        "model_seeds": config["training"]["model_seeds"],
        "expected_counts": {
            "hidden_states": 32,
            "operational_places": 16,
            "role_folds": 2,
            "seeds_per_role": 5,
            "training_rows_per_role": 45344,
            "heldout_rows_per_role": 2240,
        },
        "active_trainable_parameter_count": config["model"][carrier]["active_trainable_parameters"],
        "seed_weight_hashes": [
            {
                "role": role["role"],
                "seeds": [
                    {
                        "seed": seed["seed"],
                        "initial": seed["initial_weight_sha256"],
                        "initial_encoder": seed["initial_encoder_sha256"],
                        "final": seed["final_weight_sha256"],
                        "minibatch_indices": seed["minibatch_index_stream_sha256"],
                    }
                    for seed in role["seeds"]
                ],
            }
            for role in evidence["roles"]
        ],
        "complete": True,
    }


def _produce(config_path: Path, carrier: str, output_dir: Path) -> dict[str, Any]:
    if carrier not in CARRIERS:
        raise ContractError(f"carrier must be one of {CARRIERS}")
    started_at = _utc_now()
    started = time.monotonic()
    config_bytes = config_path.read_bytes()
    config = _validate_config(_read_json(config_path))
    _prepare_empty_directory(output_dir)
    (output_dir / "config.json").write_bytes(config_bytes)
    world = _world_truth(config)
    splits = _role_splits(config, world)
    word_pack = _word_pack(config)
    np, torch = _import_producer_dependencies(config)
    roles, training_meta, cell_walls = _train_carrier(
        config, carrier, output_dir, np, torch, world, splits
    )
    evidence = {
        "schema_version": SCHEMA_EVIDENCE,
        "registration_id": REGISTRATION_ID,
        "artifact_kind": "learned",
        "claiming": False,
        "carrier": carrier,
        "producer_code_sha256": _module_sha256(),
        "config_sha256": _sha256_bytes(config_bytes),
        "word_lists": word_pack,
        "world": world,
        "role_splits": splits,
        "roles": roles,
    }
    _write_json(output_dir / "evidence.json", evidence)
    wall_seconds = time.monotonic() - started
    manifest = _producer_manifest(
        config,
        config_bytes,
        carrier,
        output_dir,
        evidence,
        training_meta,
        cell_walls,
        np,
        torch,
        started_at,
        wall_seconds,
    )
    _write_json(output_dir / "manifest.json", manifest)
    _append_ledger(
        {
            "event_id": "round37_presentation_quotient_produce",
            "timestamp": _utc_now(),
            "registration_id": REGISTRATION_ID,
            "purpose": "Round 37 non-claiming CPU training and evidence producer",
            "carrier": carrier,
            "status": "producer_complete_nonclaiming",
            "wall_seconds": wall_seconds,
            "artifacts": [
                str(output_dir / "manifest.json"),
                str(output_dir / "evidence.json"),
                str(output_dir / "weights.npz"),
            ],
            "claim": "No scientific verdict is emitted by the producer.",
        }
    )
    return manifest


def _validate_manifest(
    value: Any,
    config: dict[str, Any],
    config_sha256: str,
    directory: Path,
) -> dict[str, Any]:
    manifest = _require_keys(
        value,
        [
            "schema_version",
            "registration_id",
            "artifact_kind",
            "claiming",
            "carrier",
            "producer_code_sha256",
            "config_sha256",
            "evidence_sha256",
            "weights_sha256",
            "git_commit",
            "command",
            "started_at",
            "completed_at",
            "wall_seconds",
            "role_cell_wall_seconds",
            "platform",
            "dependencies",
            "cpu_settings",
            "action_order",
            "word_list_sha256",
            "role_split_sha256",
            "data_seed",
            "model_seeds",
            "expected_counts",
            "active_trainable_parameter_count",
            "seed_weight_hashes",
            "complete",
        ],
        "manifest",
    )
    _expect(manifest["schema_version"], SCHEMA_MANIFEST, "manifest.schema_version")
    _expect(manifest["registration_id"], REGISTRATION_ID, "manifest.registration_id")
    if manifest["artifact_kind"] not in ("learned", "fixture"):
        raise ContractError("manifest.artifact_kind must be learned or fixture")
    _expect(manifest["claiming"], False, "manifest.claiming")
    if manifest["carrier"] not in CARRIERS:
        raise ContractError("manifest.carrier is not registered")
    _expect_sha(manifest["producer_code_sha256"], "manifest.producer_code_sha256")
    _expect(manifest["config_sha256"], config_sha256, "manifest.config_sha256")
    _expect_sha(manifest["evidence_sha256"], "manifest.evidence_sha256")
    _expect_sha(manifest["weights_sha256"], "manifest.weights_sha256")
    if not isinstance(manifest["git_commit"], str) or not manifest["git_commit"]:
        raise ContractError("manifest.git_commit must be a nonempty string")
    for key in ("command", "started_at", "completed_at"):
        if not isinstance(manifest[key], str) or not manifest[key]:
            raise ContractError(f"manifest.{key} must be a nonempty string")
    wall = _expect_number(manifest["wall_seconds"], "manifest.wall_seconds")
    if wall < 0.0:
        raise ContractError("manifest.wall_seconds must be nonnegative")
    cell_walls = _expect_list(manifest["role_cell_wall_seconds"], 2, "manifest.role_cell_wall_seconds")
    for index, value in enumerate(cell_walls):
        cell_wall = _expect_number(value, f"manifest.role_cell_wall_seconds[{index}]")
        if cell_wall < 0.0:
            raise ContractError("role cell wall must be nonnegative")
        if manifest["artifact_kind"] == "learned" and cell_wall > config["training"]["cell_hard_wall_seconds"]:
            raise ContractError("learned role cell exceeded the registered 12-minute wall")
    platform_record = _require_keys(
        manifest["platform"], ["system", "release", "machine", "processor"], "manifest.platform"
    )
    if any(not isinstance(value, str) for value in platform_record.values()):
        raise ContractError("manifest.platform values must be strings")
    dependencies = _require_keys(
        manifest["dependencies"], ["python", "numpy", "torch"], "manifest.dependencies"
    )
    if any(not isinstance(value, str) or not value for value in dependencies.values()):
        raise ContractError("manifest.dependencies values must be nonempty strings")
    cpu = _require_keys(
        manifest["cpu_settings"],
        [
            "device",
            "threads",
            "interop_threads",
            "deterministic_algorithms",
            "parameter_devices",
            "cuda_visible_devices",
        ],
        "manifest.cpu_settings",
    )
    _expect(cpu["device"], "cpu", "manifest.cpu_settings.device")
    _expect(cpu["threads"], 1, "manifest.cpu_settings.threads")
    _expect(cpu["interop_threads"], 1, "manifest.cpu_settings.interop_threads")
    _expect(cpu["deterministic_algorithms"], True, "manifest.cpu_settings.deterministic_algorithms")
    expected_devices = ["cpu"] if manifest["artifact_kind"] == "learned" else []
    _expect(cpu["parameter_devices"], expected_devices, "manifest.cpu_settings.parameter_devices")
    _expect(cpu["cuda_visible_devices"], "", "manifest.cpu_settings.cuda_visible_devices")
    _expect(manifest["action_order"], ALL_ACTIONS, "manifest.action_order")
    word_hashes = _require_keys(
        manifest["word_list_sha256"],
        ["training_list_sha256", "h2_list_sha256", "h3_list_sha256"],
        "manifest.word_list_sha256",
    )
    for key, digest in word_hashes.items():
        _expect(digest, config["split"][key], f"manifest.word_list_sha256.{key}")
    role_hashes = _expect_list(manifest["role_split_sha256"], 2, "manifest.role_split_sha256")
    for role, record in enumerate(role_hashes):
        row = _require_keys(
            record,
            ["role", "training_rows_sha256", "heldout_rows_sha256"],
            f"manifest.role_split_sha256[{role}]",
        )
        _expect(row["role"], role, f"manifest.role_split_sha256[{role}].role")
        _expect_sha(row["training_rows_sha256"], f"manifest.role_split_sha256[{role}].training_rows_sha256")
        _expect_sha(row["heldout_rows_sha256"], f"manifest.role_split_sha256[{role}].heldout_rows_sha256")
    _expect(manifest["data_seed"], 3701, "manifest.data_seed")
    _expect(manifest["model_seeds"], [11, 23, 37, 53, 71], "manifest.model_seeds")
    _expect(
        manifest["expected_counts"],
        {
            "hidden_states": 32,
            "operational_places": 16,
            "role_folds": 2,
            "seeds_per_role": 5,
            "training_rows_per_role": 45344,
            "heldout_rows_per_role": 2240,
        },
        "manifest.expected_counts",
    )
    _expect(
        manifest["active_trainable_parameter_count"],
        config["model"][manifest["carrier"]]["active_trainable_parameters"],
        "manifest.active_trainable_parameter_count",
    )
    seed_hashes = _expect_list(manifest["seed_weight_hashes"], 2, "manifest.seed_weight_hashes")
    for role, role_record in enumerate(seed_hashes):
        typed_role = _require_keys(role_record, ["role", "seeds"], f"manifest.seed_weight_hashes[{role}]")
        _expect(typed_role["role"], role, f"manifest.seed_weight_hashes[{role}].role")
        seeds = _expect_list(typed_role["seeds"], 5, f"manifest.seed_weight_hashes[{role}].seeds")
        for index, seed in enumerate(seeds):
            typed_seed = _require_keys(
                seed,
                ["seed", "initial", "initial_encoder", "final", "minibatch_indices"],
                f"manifest.seed_weight_hashes[{role}].seeds[{index}]",
            )
            _expect(typed_seed["seed"], config["training"]["model_seeds"][index], f"manifest seed order role {role}")
            _expect_sha(typed_seed["initial"], "manifest seed initial hash")
            _expect_sha(typed_seed["initial_encoder"], "manifest seed initial encoder hash")
            _expect_sha(typed_seed["final"], "manifest seed final hash")
            _expect_sha(typed_seed["minibatch_indices"], "manifest seed minibatch-index hash")
    _expect(manifest["complete"], True, "manifest.complete")
    if _sha256_file(directory / "evidence.json") != manifest["evidence_sha256"]:
        raise ContractError("manifest/evidence SHA-256 mismatch")
    weights_path = directory / "weights.npz"
    if not weights_path.is_file() or weights_path.stat().st_size <= 0:
        raise ContractError("weights.npz reproduction material is missing or empty")
    return manifest


def _validate_probability_matrix(value: Any, rows: int, columns: int, where: str) -> list[list[float]]:
    matrix = _expect_list(value, rows, where)
    for row_index, row in enumerate(matrix):
        typed = _expect_list(row, columns, f"{where}[{row_index}]")
        for column_index, probability in enumerate(typed):
            _expect_probability(probability, f"{where}[{row_index}][{column_index}]")
    return matrix  # type: ignore[return-value]


def _validate_evaluation(
    value: Any, config: dict[str, Any], where: str
) -> dict[str, Any]:
    evaluation = _require_keys(
        value,
        [
            "training_probabilities",
            "base_signature_probabilities",
            "task_descent_probabilities",
            "heldout_pair_probabilities",
            "heldout_pair_step_signatures",
            "presentation_move_probabilities",
            "endpoint_signatures",
        ],
        where,
    )
    training = _expect_list(evaluation["training_probabilities"], 45344, f"{where}.training_probabilities")
    for index, probability in enumerate(training):
        _expect_probability(probability, f"{where}.training_probabilities[{index}]")
    _validate_probability_matrix(evaluation["base_signature_probabilities"], 32, 12, f"{where}.base_signature_probabilities")
    descent = _expect_list(evaluation["task_descent_probabilities"], 16, f"{where}.task_descent_probabilities")
    for q_index, q_rows in enumerate(descent):
        actions = _expect_list(q_rows, 11, f"{where}.task_descent_probabilities[{q_index}]")
        for action_index, action_rows in enumerate(actions):
            presentations = _expect_list(action_rows, 2, f"{where}.task_descent_probabilities[{q_index}][{action_index}]")
            for presentation, variants in enumerate(presentations):
                _validate_probability_matrix(
                    variants,
                    3,
                    12,
                    f"{where}.task_descent_probabilities[{q_index}][{action_index}][{presentation}]",
                )
    pairs = _expect_list(evaluation["heldout_pair_probabilities"], 2240, f"{where}.heldout_pair_probabilities")
    for index, pair in enumerate(pairs):
        _validate_probability_matrix(pair, 2, 12, f"{where}.heldout_pair_probabilities[{index}]")
    step_signatures = _expect_list(
        evaluation["heldout_pair_step_signatures"],
        2240,
        f"{where}.heldout_pair_step_signatures",
    )
    heldout_words = _word_universe(config)["heldout"]
    for index, steps_value in enumerate(step_signatures):
        word = heldout_words[index // 16]
        steps = _expect_list(
            steps_value,
            len(word),
            f"{where}.heldout_pair_step_signatures[{index}]",
        )
        for step_index, pair_value in enumerate(steps):
            pair = _expect_list(
                pair_value,
                2,
                f"{where}.heldout_pair_step_signatures[{index}][{step_index}]",
            )
            for presentation, signature in enumerate(pair):
                _expect_signature(
                    signature,
                    f"{where}.heldout_pair_step_signatures[{index}][{step_index}][{presentation}]",
                )
    _validate_probability_matrix(evaluation["presentation_move_probabilities"], 32, 12, f"{where}.presentation_move_probabilities")
    expected_signatures = _reported_endpoint_signatures(evaluation, 0.10, 0.90)
    _expect(evaluation["endpoint_signatures"], expected_signatures, f"{where}.endpoint_signatures replay")
    for index, steps in enumerate(step_signatures):
        _expect(
            steps[-1],
            expected_signatures["heldout_pairs"][index],
            f"{where}.heldout_pair_step_signatures[{index}] final-step replay",
        )
    return evaluation


def _validate_evidence(
    value: Any,
    config: dict[str, Any],
    config_sha256: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    evidence = _require_keys(
        value,
        [
            "schema_version",
            "registration_id",
            "artifact_kind",
            "claiming",
            "carrier",
            "producer_code_sha256",
            "config_sha256",
            "word_lists",
            "world",
            "role_splits",
            "roles",
        ],
        "evidence",
    )
    _expect(evidence["schema_version"], SCHEMA_EVIDENCE, "evidence.schema_version")
    _expect(evidence["registration_id"], REGISTRATION_ID, "evidence.registration_id")
    _expect(evidence["artifact_kind"], manifest["artifact_kind"], "evidence.artifact_kind")
    _expect(evidence["claiming"], False, "evidence.claiming")
    _expect(evidence["carrier"], manifest["carrier"], "evidence.carrier")
    _expect(evidence["producer_code_sha256"], manifest["producer_code_sha256"], "evidence.producer_code_sha256")
    _expect(evidence["config_sha256"], config_sha256, "evidence.config_sha256")
    expected_word_pack = _word_pack(config)
    _expect(evidence["word_lists"], expected_word_pack, "evidence.word_lists")
    expected_world = _world_truth(config)
    _expect(evidence["world"], expected_world, "evidence.world")
    expected_splits = _role_splits(config, expected_world)
    _expect(evidence["role_splits"], expected_splits, "evidence.role_splits")
    for role, manifest_hashes in enumerate(manifest["role_split_sha256"]):
        _expect(manifest_hashes["training_rows_sha256"], expected_splits[role]["training_rows_sha256"], f"manifest/evidence role {role} training hash")
        _expect(manifest_hashes["heldout_rows_sha256"], expected_splits[role]["heldout_rows_sha256"], f"manifest/evidence role {role} heldout hash")
    roles = _expect_list(evidence["roles"], 2, "evidence.roles")
    for role, role_value in enumerate(roles):
        role_record = _require_keys(role_value, ["role", "cell_wall_seconds", "seeds"], f"evidence.roles[{role}]")
        _expect(role_record["role"], role, f"evidence.roles[{role}].role")
        wall = _expect_number(role_record["cell_wall_seconds"], f"evidence.roles[{role}].cell_wall_seconds")
        if wall < 0.0:
            raise ContractError("evidence role wall must be nonnegative")
        _expect(wall, manifest["role_cell_wall_seconds"][role], f"manifest/evidence role {role} wall")
        seeds = _expect_list(role_record["seeds"], 5, f"evidence.roles[{role}].seeds")
        for seed_index, seed_value in enumerate(seeds):
            where = f"evidence.roles[{role}].seeds[{seed_index}]"
            seed = _require_keys(
                seed_value,
                [
                    "seed",
                    "steps",
                    "active_trainable_parameter_count",
                    "initial_weight_sha256",
                    "initial_encoder_sha256",
                    "final_weight_sha256",
                    "minibatch_index_stream_sha256",
                    "loss_trace_sha256",
                    "component_trace",
                    "evaluation",
                ],
                where,
            )
            expected_seed = config["training"]["model_seeds"][seed_index]
            _expect(seed["seed"], expected_seed, f"{where}.seed")
            _expect(seed["steps"], 32000, f"{where}.steps")
            _expect(seed["active_trainable_parameter_count"], config["model"][manifest["carrier"]]["active_trainable_parameters"], f"{where}.active_trainable_parameter_count")
            for hash_key in (
                "initial_weight_sha256",
                "initial_encoder_sha256",
                "final_weight_sha256",
                "minibatch_index_stream_sha256",
                "loss_trace_sha256",
            ):
                _expect_sha(seed[hash_key], f"{where}.{hash_key}")
            manifest_seed = manifest["seed_weight_hashes"][role]["seeds"][seed_index]
            _expect(seed["initial_weight_sha256"], manifest_seed["initial"], f"{where} initial hash binding")
            _expect(seed["initial_encoder_sha256"], manifest_seed["initial_encoder"], f"{where} initial encoder hash binding")
            _expect(seed["final_weight_sha256"], manifest_seed["final"], f"{where} final hash binding")
            _expect(seed["minibatch_index_stream_sha256"], manifest_seed["minibatch_indices"], f"{where} minibatch-index hash binding")
            trace = _expect_list(seed["component_trace"], 33, f"{where}.component_trace")
            for trace_index, trace_value in enumerate(trace):
                trace_where = f"{where}.component_trace[{trace_index}]"
                record = _require_keys(
                    trace_value,
                    [
                        "step",
                        "batch_loss",
                        "base_signature_probabilities",
                        "base_signature_component_bce",
                        "base_signature_component_supported",
                    ],
                    trace_where,
                )
                _expect(record["step"], TRACE_STEPS[trace_index], f"{trace_where}.step")
                if trace_index == 0:
                    _expect(record["batch_loss"], None, f"{trace_where}.batch_loss")
                else:
                    _expect_number(record["batch_loss"], f"{trace_where}.batch_loss")
                probabilities = _expect_list(record["base_signature_probabilities"], 384, f"{trace_where}.base_signature_probabilities")
                losses = _expect_list(record["base_signature_component_bce"], 384, f"{trace_where}.base_signature_component_bce")
                supported = _expect_list(record["base_signature_component_supported"], 384, f"{trace_where}.base_signature_component_supported")
                for component in range(384):
                    probability = _expect_probability(probabilities[component], f"{trace_where}.probability[{component}]")
                    loss = _expect_number(losses[component], f"{trace_where}.loss[{component}]")
                    if loss < 0.0:
                        raise ContractError(f"{trace_where}.loss[{component}] must be nonnegative")
                    if not isinstance(supported[component], bool):
                        raise ContractError(f"{trace_where}.supported[{component}] must be boolean")
                    expected_support = probability <= config["thresholds"]["signature_low"] or probability >= config["thresholds"]["signature_high"]
                    _expect(supported[component], expected_support, f"{trace_where}.supported[{component}] replay")
            _validate_evaluation(seed["evaluation"], config, f"{where}.evaluation")
    return evidence


def _load_artifact(config_path: Path, directory: Path) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = _validate_config(_read_json(config_path))
    config_sha256 = _sha256_bytes(config_bytes)
    for name in ("config.json", "manifest.json", "evidence.json", "weights.npz"):
        if not (directory / name).is_file():
            raise ContractError(f"required artifact missing: {directory / name}")
    if _sha256_file(directory / "config.json") != config_sha256:
        raise ContractError(f"artifact config does not match reducer config: {directory}")
    _expect(_read_json(directory / "config.json"), config, f"{directory}/config.json")
    manifest = _validate_manifest(_read_json(directory / "manifest.json"), config, config_sha256, directory)
    evidence = _validate_evidence(_read_json(directory / "evidence.json"), config, config_sha256, manifest)
    return {
        "directory": directory,
        "config": config,
        "config_sha256": config_sha256,
        "manifest": manifest,
        "evidence": evidence,
        "input_hashes": {
            "manifest_sha256": _sha256_file(directory / "manifest.json"),
            "evidence_sha256": _sha256_file(directory / "evidence.json"),
        },
    }


def _signature(probabilities: Sequence[Any], low: float, high: float) -> str:
    if len(probabilities) != 12:
        raise ContractError("operational signatures require exactly 12 components")
    result: list[str] = []
    for index, value in enumerate(probabilities):
        probability = _expect_probability(value, f"signature probability {index}")
        if probability <= low:
            result.append("0")
        elif probability >= high:
            result.append("1")
        else:
            result.append("?")
    return "".join(result)


def _reported_endpoint_signatures(
    evaluation: dict[str, Any], low: float, high: float
) -> dict[str, Any]:
    return {
        "base": [
            _signature(probabilities, low, high)
            for probabilities in evaluation["base_signature_probabilities"]
        ],
        "task_descent": [
            [
                [
                    [
                        _signature(probabilities, low, high)
                        for probabilities in variants
                    ]
                    for variants in presentations
                ]
                for presentations in actions
            ]
            for actions in evaluation["task_descent_probabilities"]
        ],
        "heldout_pairs": [
            [
                _signature(probabilities, low, high)
                for probabilities in pair
            ]
            for pair in evaluation["heldout_pair_probabilities"]
        ],
        "presentation_move": [
            _signature(probabilities, low, high)
            for probabilities in evaluation["presentation_move_probabilities"]
        ],
    }


def _signed_margins(probabilities: Sequence[Any], oracle: str) -> list[float]:
    if len(probabilities) != len(oracle):
        raise ContractError("margin probability/oracle length mismatch")
    return [
        _expect_probability(probability, "margin probability") - 0.90
        if bit == "1"
        else 0.10 - _expect_probability(probability, "margin probability")
        for probability, bit in zip(probabilities, oracle)
    ]


def _margin_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ContractError("cannot summarize an empty margin population")
    checked = [_expect_number(value, "signed margin") for value in values]
    return {
        "minimum": min(checked),
        "median": statistics.median(checked),
        "mean": statistics.fmean(checked),
        "maximum": max(checked),
    }


def _rate(correct: int, unsupported: int, wrong: int) -> dict[str, Any]:
    denominator = correct + unsupported + wrong
    if denominator <= 0:
        raise ContractError("certificate denominator must be positive")
    return {
        "numerator": correct,
        "denominator": denominator,
        "rate": correct / denominator,
        "unsupported": unsupported,
        "wrong": wrong,
        "exact": correct == denominator,
    }


def _classify_signatures(signatures: Sequence[str], oracle: str) -> str:
    if any("?" in signature for signature in signatures):
        return "unsupported"
    if any(signature != oracle for signature in signatures):
        return "wrong"
    if len(set(signatures)) != 1:
        return "wrong"
    return "correct"


def _diagnostic_component_status(
    signatures: Sequence[str], oracle_component: str, component: int
) -> str:
    observed = [signature[component] for signature in signatures]
    if any(value == "?" for value in observed):
        return "unsupported"
    if any(value != oracle_component for value in observed) or len(set(observed)) != 1:
        return "wrong_supported"
    return "correct"


def _heldout_contexts(config: dict[str, Any]) -> list[dict[str, Any]]:
    universe = _word_universe(config)
    contexts: list[dict[str, Any]] = []
    for horizon, key in (("H2", "h2"), ("H3", "h3")):
        for word in universe[key]:
            for q_index, q in enumerate(_operational_states()):
                contexts.append(
                    {
                        "horizon": horizon,
                        "continuation_depth": len(word),
                        "word": word,
                        "q_index": q_index,
                        "q": q,
                    }
                )
    _expect(len(contexts), 2240, "diagnostic held-out context count")
    return contexts


def _diagnostic_margin_vectors(
    config: dict[str, Any],
    world: dict[str, Any],
    seed_record: dict[str, Any],
) -> dict[str, dict[str, list[float]]]:
    result = {
        horizon: {
            "all_components": [],
            "terminal_response_components": [],
            "future_signature_components": [],
        }
        for horizon in ("H2", "H3")
    }
    probabilities = seed_record["evaluation"]["heldout_pair_probabilities"]
    for index, context in enumerate(_heldout_contexts(config)):
        endpoint_q, _ = _apply_word((context["q"], 0), context["word"])
        oracle = world["oracle_signatures"][_q_index(endpoint_q)]
        margins: list[float] = []
        for presentation_probabilities in probabilities[index]:
            margins.extend(_signed_margins(presentation_probabilities, oracle))
        horizon = context["horizon"]
        result[horizon]["all_components"].extend(margins)
        result[horizon]["terminal_response_components"].extend(
            [margins[0], margins[12]]
        )
        result[horizon]["future_signature_components"].extend(
            margins[1:12] + margins[13:24]
        )
    return result


def _diagnostic_unit(
    config: dict[str, Any],
    world: dict[str, Any],
    split: dict[str, Any],
    seed_record: dict[str, Any],
) -> dict[str, Any]:
    accumulators: dict[str, dict[str, Any]] = {}
    for horizon, continuation_depth, expected_cells in (
        ("H2", 2, 1184),
        ("H3", 3, 1056),
    ):
        accumulators[horizon] = {
            "continuation_depth": continuation_depth,
            "expected_cells": expected_cells,
            "final_cells": {"correct": 0, "unsupported": 0, "wrong_supported": 0},
            "first_divergence_step": {
                "none": 0,
                "step_0": 0,
                "step_1": 0,
                "step_2": 0,
                **({"step_3": 0} if continuation_depth == 3 else {}),
            },
            "failure_scope": {
                "neither": 0,
                "terminal_only": 0,
                "future_signature_only": 0,
                "terminal_and_future_signature": 0,
            },
            "bad_components": {
                "terminal_response": {"unsupported": 0, "wrong_supported": 0},
                "future_signature": {"unsupported": 0, "wrong_supported": 0},
            },
        }

    evaluation = seed_record["evaluation"]
    base_signatures = evaluation["endpoint_signatures"]["base"]
    step_signatures = evaluation["heldout_pair_step_signatures"]
    contexts = _heldout_contexts(config)
    for index, context in enumerate(contexts):
        accumulator = accumulators[context["horizon"]]
        q_index = context["q_index"]
        oracle_start = world["oracle_signatures"][q_index]
        handles = world["operational_pairs"][q_index]
        start_pair = [base_signatures[handle] for handle in handles]
        first_divergence: int | None = None
        if _classify_signatures(start_pair, oracle_start) != "correct":
            first_divergence = 0
        for step, pair in enumerate(step_signatures[index], start=1):
            prefix_q, _ = _apply_word((context["q"], 0), context["word"][:step])
            oracle_step = world["oracle_signatures"][_q_index(prefix_q)]
            if (
                first_divergence is None
                and _classify_signatures(pair, oracle_step) != "correct"
            ):
                first_divergence = step
        divergence_key = (
            "none" if first_divergence is None else f"step_{first_divergence}"
        )
        accumulator["first_divergence_step"][divergence_key] += 1

        endpoint_q, _ = _apply_word((context["q"], 0), context["word"])
        oracle = world["oracle_signatures"][_q_index(endpoint_q)]
        final_pair = step_signatures[index][-1]
        final_classification = _classify_signatures(final_pair, oracle)
        final_key = (
            "wrong_supported"
            if final_classification == "wrong"
            else final_classification
        )
        accumulator["final_cells"][final_key] += 1

        component_statuses = [
            _diagnostic_component_status(final_pair, bit, component)
            for component, bit in enumerate(oracle)
        ]
        terminal_failed = component_statuses[0] != "correct"
        future_failed = any(status != "correct" for status in component_statuses[1:])
        if terminal_failed and future_failed:
            scope = "terminal_and_future_signature"
        elif terminal_failed:
            scope = "terminal_only"
        elif future_failed:
            scope = "future_signature_only"
        else:
            scope = "neither"
        accumulator["failure_scope"][scope] += 1
        for component, status in enumerate(component_statuses):
            if status == "correct":
                continue
            component_scope = (
                "terminal_response" if component == 0 else "future_signature"
            )
            accumulator["bad_components"][component_scope][status] += 1

    margin_vectors = _diagnostic_margin_vectors(config, world, seed_record)
    by_continuation: dict[str, Any] = {}
    for horizon in ("H2", "H3"):
        accumulator = accumulators[horizon]
        total_cells = sum(accumulator["final_cells"].values())
        _expect(total_cells, accumulator["expected_cells"], f"{horizon} diagnostic cell count")
        _expect(
            sum(accumulator["first_divergence_step"].values()),
            total_cells,
            f"{horizon} first-divergence accounting",
        )
        _expect(
            sum(accumulator["failure_scope"].values()),
            total_cells,
            f"{horizon} failure-scope accounting",
        )
        bad = accumulator["bad_components"]
        by_continuation[horizon] = {
            "history_depth": {
                "representative_presentation_start_task_depth": 0,
                "continuation_task_depth": accumulator["continuation_depth"],
                "continuation_steps_are_one_indexed": True,
            },
            "final_cell_classification": {
                "total": total_cells,
                "supported_truthful_equal": accumulator["final_cells"]["correct"],
                "unsupported": accumulator["final_cells"]["unsupported"],
                "wrong_supported": accumulator["final_cells"]["wrong_supported"],
            },
            "first_divergence_step_histogram": accumulator["first_divergence_step"],
            "terminal_vs_future_signature_failure_cells": accumulator["failure_scope"],
            "unsupported_vs_wrong_supported_bad_components": {
                "terminal_response": bad["terminal_response"],
                "future_signature": bad["future_signature"],
                "total": {
                    "unsupported": bad["terminal_response"]["unsupported"]
                    + bad["future_signature"]["unsupported"],
                    "wrong_supported": bad["terminal_response"]["wrong_supported"]
                    + bad["future_signature"]["wrong_supported"],
                },
            },
            "signed_certificate_margins": {
                scope: _margin_summary(values)
                for scope, values in margin_vectors[horizon].items()
            },
        }
    return {
        "role": split["role"],
        "seed": seed_record["seed"],
        "by_continuation": by_continuation,
    }


def _diagnostic_block(
    config: dict[str, Any], artifacts: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    carrier_units: dict[str, list[dict[str, Any]]] = {}
    for carrier in CARRIERS:
        evidence = artifacts[carrier]["evidence"]
        units: list[dict[str, Any]] = []
        for role in range(2):
            split = evidence["role_splits"][role]
            for seed_record in evidence["roles"][role]["seeds"]:
                units.append(
                    _diagnostic_unit(config, evidence["world"], split, seed_record)
                )
        carrier_units[carrier] = units

    matched_margins: list[dict[str, Any]] = []
    for role in range(2):
        for seed_index, seed in enumerate(config["training"]["model_seeds"]):
            factored_vectors = _diagnostic_margin_vectors(
                config,
                artifacts[FACTORED]["evidence"]["world"],
                artifacts[FACTORED]["evidence"]["roles"][role]["seeds"][seed_index],
            )
            unrestricted_vectors = _diagnostic_margin_vectors(
                config,
                artifacts[UNRESTRICTED]["evidence"]["world"],
                artifacts[UNRESTRICTED]["evidence"]["roles"][role]["seeds"][seed_index],
            )
            by_continuation: dict[str, Any] = {}
            for horizon in ("H2", "H3"):
                by_scope: dict[str, Any] = {}
                for scope in (
                    "all_components",
                    "terminal_response_components",
                    "future_signature_components",
                ):
                    factored_values = factored_vectors[horizon][scope]
                    unrestricted_values = unrestricted_vectors[horizon][scope]
                    _expect(
                        len(factored_values),
                        len(unrestricted_values),
                        f"matched diagnostic margin length role={role} seed={seed} {horizon} {scope}",
                    )
                    by_scope[scope] = {
                        FACTORED: _margin_summary(factored_values),
                        UNRESTRICTED: _margin_summary(unrestricted_values),
                        "factored_minus_unrestricted_per_component": _margin_summary(
                            [
                                factored_value - unrestricted_value
                                for factored_value, unrestricted_value in zip(
                                    factored_values, unrestricted_values
                                )
                            ]
                        ),
                    }
                by_continuation[horizon] = by_scope
            matched_margins.append(
                {
                    "role": role,
                    "seed": seed,
                    "by_continuation": by_continuation,
                }
            )
    return {
        "status_effect": "none",
        "gate_threshold_or_verdict_logic_changes": False,
        "primary_structural_question": "rolled-history interchangeability across opaque presentations",
        "definitions": {
            "history_depth": "Presentation starts are encoder representatives at task-history depth 0; H2/H3 continuation actions are numbered from 1.",
            "first_divergence_step": "Earliest task-prefix step at which the two presentation signatures are not both supported, equal, and oracle-truthful; step 0 is the presentation start and none means no divergence through the endpoint.",
            "terminal_vs_future_signature": "Signature component 0 is the immediate endpoint response; components 1 through 11 are one-task-action future probes.",
            "unsupported_vs_wrong_supported": "Unsupported means at least one paired component lies inside (0.10,0.90); wrong-supported means both are supported but disagree with each other or oracle truth.",
            "margins": "Signed certificate margins use p-0.90 for oracle 1 and 0.10-p for oracle 0; matched deltas are paired factored minus unrestricted per component within seed and role.",
        },
        "carrier_units": {
            carrier: {"units": carrier_units[carrier]} for carrier in CARRIERS
        },
        "factored_vs_unrestricted_margins_per_seed_role": matched_margins,
    }


def _unit_certificates(
    config: dict[str, Any],
    world: dict[str, Any],
    split: dict[str, Any],
    seed_record: dict[str, Any],
) -> dict[str, Any]:
    low = config["thresholds"]["signature_low"]
    high = config["thresholds"]["signature_high"]
    role = split["role"]
    evaluation = seed_record["evaluation"]

    targets = [_target_for_row(row, world) for row in split["training_rows"]]
    training_predictions = [
        int(_expect_probability(probability, "training probability") > 0.5)
        for probability in evaluation["training_probabilities"]
    ]
    training_wrong = sum(prediction != target for prediction, target in zip(training_predictions, targets))
    behavior = {
        "numerator": len(targets) - training_wrong,
        "denominator": len(targets),
        "accuracy": (len(targets) - training_wrong) / len(targets),
        "wrong": training_wrong,
        "exact": training_wrong == 0,
        "decision_rule": "p>0.5",
    }

    base_signatures: list[str] = []
    base_margins: list[float] = []
    base_unsupported_components = 0
    base_wrong_components = 0
    for handle, probabilities in enumerate(evaluation["base_signature_probabilities"]):
        q_index = world["handle_to_hidden_state"][handle] // 2
        oracle = world["oracle_signatures"][q_index]
        signature = _signature(probabilities, low, high)
        base_signatures.append(signature)
        base_margins.extend(_signed_margins(probabilities, oracle))
        base_unsupported_components += signature.count("?")
        base_wrong_components += sum(
            observed != "?" and observed != expected
            for observed, expected in zip(signature, oracle)
        )
    base_support = {
        "supported_components": 384 - base_unsupported_components,
        "total_components": 384,
        "unsupported_components": base_unsupported_components,
        "wrong_supported_components": base_wrong_components,
        "exact_supported_truthful": base_unsupported_components == 0 and base_wrong_components == 0,
        "signed_margins": _margin_summary(base_margins),
    }
    behavior_eligible = bool(behavior["exact"] and base_support["exact_supported_truthful"])

    paired_counts = {"correct": 0, "unsupported": 0, "wrong": 0}
    paired_margins: list[float] = []
    for q_index, handles in enumerate(world["operational_pairs"]):
        oracle = world["oracle_signatures"][q_index]
        signatures = [base_signatures[handle] for handle in handles]
        classification = _classify_signatures(signatures, oracle)
        paired_counts[classification] += 1
        for handle in handles:
            paired_margins.extend(
                _signed_margins(evaluation["base_signature_probabilities"][handle], oracle)
            )
    paired_places = {
        **_rate(**paired_counts),
        "signed_margins": _margin_summary(paired_margins),
    }

    descent_counts = {"correct": 0, "unsupported": 0, "wrong": 0}
    variant_margins: dict[str, list[float]] = {
        f"{variant}_p{presentation}": []
        for presentation in (0, 1)
        for variant in ("direct", "present_before", "present_after")
    }
    for q_index in range(16):
        for action_id in range(11):
            successor_place = world["oracle_task_successor_places"][q_index][action_id]
            oracle = world["oracle_signatures"][successor_place]
            signatures: list[str] = []
            for presentation in (0, 1):
                for variant_index, variant in enumerate(("direct", "present_before", "present_after")):
                    probabilities = evaluation["task_descent_probabilities"][q_index][action_id][presentation][variant_index]
                    signatures.append(_signature(probabilities, low, high))
                    variant_margins[f"{variant}_p{presentation}"].extend(
                        _signed_margins(probabilities, oracle)
                    )
            descent_counts[_classify_signatures(signatures, oracle)] += 1
    task_descent = {
        **_rate(**descent_counts),
        "endpoint_signed_margins": {
            key: _margin_summary(values) for key, values in variant_margins.items()
        },
    }

    heldout_words = _word_universe(config)["heldout"]
    transfer_signature_counts = {"correct": 0, "unsupported": 0, "wrong": 0}
    transfer_response_correct = 0
    transfer_response_wrong = 0
    transfer_margins: list[float] = []
    rolled_counts = {"correct": 0, "unsupported": 0, "wrong": 0}
    rolled_margins = {"presentation_0": [], "presentation_1": []}
    pair_index = 0
    heldout_presentation = 1 - role
    for word in heldout_words:
        for q_index, q in enumerate(_operational_states()):
            endpoint_q, _ = _apply_word((q, 0), word)
            oracle = world["oracle_signatures"][_q_index(endpoint_q)]
            pair = evaluation["heldout_pair_probabilities"][pair_index]
            signatures = [_signature(pair[presentation], low, high) for presentation in (0, 1)]
            heldout_signature = signatures[heldout_presentation]
            if "?" in heldout_signature:
                transfer_signature_counts["unsupported"] += 1
            elif heldout_signature != oracle:
                transfer_signature_counts["wrong"] += 1
            else:
                transfer_signature_counts["correct"] += 1
            response_probability = _expect_probability(pair[heldout_presentation][0], "held-out terminal response")
            response_target = int(endpoint_q[0])
            if int(response_probability > 0.5) == response_target:
                transfer_response_correct += 1
            else:
                transfer_response_wrong += 1
            transfer_margins.extend(_signed_margins(pair[heldout_presentation], oracle))
            rolled_counts[_classify_signatures(signatures, oracle)] += 1
            for presentation in (0, 1):
                rolled_margins[f"presentation_{presentation}"].extend(
                    _signed_margins(pair[presentation], oracle)
                )
            pair_index += 1
    heldout_signature_rate = _rate(**transfer_signature_counts)
    heldout_response = {
        "numerator": transfer_response_correct,
        "denominator": transfer_response_correct + transfer_response_wrong,
        "accuracy": transfer_response_correct / (transfer_response_correct + transfer_response_wrong),
        "wrong": transfer_response_wrong,
        "exact": transfer_response_wrong == 0,
    }
    heldout_transfer = {
        "signature": heldout_signature_rate,
        "terminal_response": heldout_response,
        "signed_margins": _margin_summary(transfer_margins),
        "exact": bool(heldout_signature_rate["exact"] and heldout_response["exact"]),
    }
    rolled = {
        **_rate(**rolled_counts),
        "side_signed_margins": {
            key: _margin_summary(values) for key, values in rolled_margins.items()
        },
    }

    presentation_counts = {"correct": 0, "unsupported": 0, "wrong": 0}
    presentation_margins: list[float] = []
    for handle, moved_probabilities in enumerate(evaluation["presentation_move_probabilities"]):
        q_index = world["handle_to_hidden_state"][handle] // 2
        oracle = world["oracle_signatures"][q_index]
        moved_signature = _signature(moved_probabilities, low, high)
        base_signature = base_signatures[handle]
        if "?" in moved_signature or "?" in base_signature:
            presentation_counts["unsupported"] += 1
        elif moved_signature != oracle or base_signature != oracle or moved_signature != base_signature:
            presentation_counts["wrong"] += 1
        else:
            presentation_counts["correct"] += 1
        presentation_margins.extend(_signed_margins(moved_probabilities, oracle))
    presentation_move = {
        **_rate(**presentation_counts),
        "signed_margins": _margin_summary(presentation_margins),
    }

    certificates = {
        "paired_presentations_name_one_place": paired_places,
        "task_actions_descend_independently": task_descent,
        "heldout_presentation_transfer": heldout_transfer,
        "rolled_interchangeability": rolled,
        "presentation_move_preserves_place": presentation_move,
    }
    all_certificates_exact = all(
        certificate["exact"] for certificate in certificates.values()
    )
    comparison_margin = min(
        heldout_transfer["signed_margins"]["minimum"],
        rolled["side_signed_margins"]["presentation_0"]["minimum"],
        rolled["side_signed_margins"]["presentation_1"]["minimum"],
    )
    return {
        "role": role,
        "seed": seed_record["seed"],
        "behavior_fit": behavior,
        "base_signature_support": base_support,
        "behavior_eligible": behavior_eligible,
        "certificates": certificates,
        "all_certificates_exact": all_certificates_exact,
        "all_cell_exact": bool(behavior_eligible and all_certificates_exact),
        "comparison_metrics": {
            "heldout_transfer_rate": heldout_signature_rate["rate"],
            "rolled_interchangeability_rate": rolled["rate"],
            "minimum_signed_margin": comparison_margin,
            "heldout_transfer_minimum_signed_margin": heldout_transfer["signed_margins"]["minimum"],
            "rolled_minimum_signed_margin": min(
                rolled["side_signed_margins"]["presentation_0"]["minimum"],
                rolled["side_signed_margins"]["presentation_1"]["minimum"],
            ),
        },
    }


def _carrier_result(artifact: dict[str, Any]) -> dict[str, Any]:
    evidence = artifact["evidence"]
    config = artifact["config"]
    units: list[dict[str, Any]] = []
    for role in range(2):
        for seed_record in evidence["roles"][role]["seeds"]:
            units.append(
                _unit_certificates(
                    config,
                    evidence["world"],
                    evidence["role_splits"][role],
                    seed_record,
                )
            )
    gate_names = [
        "paired_presentations_name_one_place",
        "task_actions_descend_independently",
        "heldout_presentation_transfer",
        "rolled_interchangeability",
        "presentation_move_preserves_place",
    ]
    exact_units = {
        gate: sum(bool(unit["certificates"][gate]["exact"]) for unit in units)
        for gate in gate_names
    }
    eligible_units = sum(bool(unit["behavior_eligible"]) for unit in units)
    return {
        "carrier": evidence["carrier"],
        "artifact_kind": evidence["artifact_kind"],
        "active_trainable_parameter_count": artifact["manifest"]["active_trainable_parameter_count"],
        "behavior_eligible_units": {
            "numerator": eligible_units,
            "denominator": 10,
            "exact": eligible_units == 10,
        },
        "certificate_exact_units": {
            gate: {
                "numerator": count,
                "denominator": 10,
                "exact": count == 10,
            }
            for gate, count in exact_units.items()
        },
        "units": units,
        "all_cell_exact": all(unit["all_cell_exact"] for unit in units),
    }


def _primary_status(factored: dict[str, Any]) -> str:
    if not factored["behavior_eligible_units"]["exact"]:
        return UNDERFIT_STATUS
    failure_order = [
        ("paired_presentations_name_one_place", FAIL_PAIRED),
        ("task_actions_descend_independently", FAIL_DESCENT),
        ("heldout_presentation_transfer", FAIL_TRANSFER),
        ("rolled_interchangeability", FAIL_INTERCHANGE),
        ("presentation_move_preserves_place", FAIL_PRESENT),
    ]
    for gate, status in failure_order:
        if not factored["certificate_exact_units"][gate]["exact"]:
            return status
    return PASS_STATUS


def _comparison_label(factored_exact: bool, unrestricted_exact: bool) -> str:
    if factored_exact and not unrestricted_exact:
        return FACTORED_ADVANTAGE
    if factored_exact and unrestricted_exact:
        return SOLVED_BY_BOTH
    if not factored_exact and unrestricted_exact:
        return UNRESTRICTED_ADVANTAGE
    return NO_ARCHITECTURAL_WIN


def _paired_differences(
    factored: dict[str, Any], unrestricted: dict[str, Any]
) -> list[dict[str, Any]]:
    unrestricted_units = {
        (unit["role"], unit["seed"]): unit for unit in unrestricted["units"]
    }
    differences: list[dict[str, Any]] = []
    for unit in factored["units"]:
        key = (unit["role"], unit["seed"])
        peer = unrestricted_units[key]
        factored_metrics = unit["comparison_metrics"]
        peer_metrics = peer["comparison_metrics"]
        differences.append(
            {
                "role": unit["role"],
                "seed": unit["seed"],
                "heldout_transfer_rate_factored_minus_unrestricted": factored_metrics["heldout_transfer_rate"] - peer_metrics["heldout_transfer_rate"],
                "rolled_interchangeability_rate_factored_minus_unrestricted": factored_metrics["rolled_interchangeability_rate"] - peer_metrics["rolled_interchangeability_rate"],
                "minimum_signed_margin_factored_minus_unrestricted": factored_metrics["minimum_signed_margin"] - peer_metrics["minimum_signed_margin"],
                "heldout_transfer_minimum_signed_margin_factored_minus_unrestricted": factored_metrics["heldout_transfer_minimum_signed_margin"] - peer_metrics["heldout_transfer_minimum_signed_margin"],
                "rolled_minimum_signed_margin_factored_minus_unrestricted": factored_metrics["rolled_minimum_signed_margin"] - peer_metrics["rolled_minimum_signed_margin"],
            }
        )
    return differences


def _invalid_verdict(
    config_path: Path,
    target_carrier: str,
    reason: str,
    target_directory: Path,
    peer_directory: Path,
) -> dict[str, Any]:
    input_hashes: dict[str, Any] = {"config_sha256": None, FACTORED: None, UNRESTRICTED: None}
    try:
        input_hashes["config_sha256"] = _sha256_file(config_path)
    except OSError:
        pass
    for directory in (target_directory, peer_directory):
        try:
            manifest = _read_json(directory / "manifest.json")
            carrier = manifest.get("carrier") if isinstance(manifest, dict) else None
            if carrier in CARRIERS:
                input_hashes[carrier] = {
                    "manifest_sha256": _sha256_file(directory / "manifest.json"),
                    "evidence_sha256": _sha256_file(directory / "evidence.json"),
                }
        except (ContractError, OSError):
            continue
    return {
        "schema_version": SCHEMA_VERDICT,
        "registration_id": REGISTRATION_ID,
        "claiming": False,
        "artifact_kind": None,
        "target_carrier": target_carrier,
        "status": INVALID_STATUS,
        "reason": reason,
        "primary_world_verdict": None,
        "comparison_label": None,
        "carrier_results": None,
        "paired_differences": None,
        "diagnostic_only": None,
        "reducer_code_sha256": _module_sha256(),
        "input_hashes": input_hashes,
        "generated_at": _utc_now(),
    }


def _reduce_directories(
    config_path: Path,
    target_carrier: str,
    target_directory: Path,
    peer_directory: Path,
    *,
    write_verdict: bool,
    record_ledger: bool,
) -> dict[str, Any]:
    try:
        if target_carrier not in CARRIERS:
            raise ContractError(f"target carrier must be one of {CARRIERS}")
        target = _load_artifact(config_path, target_directory)
        peer = _load_artifact(config_path, peer_directory)
        _expect(target["manifest"]["carrier"], target_carrier, "target artifact carrier")
        if target["manifest"]["carrier"] == peer["manifest"]["carrier"]:
            raise ContractError("target and peer artifacts must be opposite carriers")
        _expect(
            target["manifest"]["artifact_kind"],
            peer["manifest"]["artifact_kind"],
            "matched artifact kinds",
        )
        artifacts = {
            target["manifest"]["carrier"]: target,
            peer["manifest"]["carrier"]: peer,
        }
        if set(artifacts) != set(CARRIERS):
            raise ContractError("reduction requires exactly factored and unrestricted artifacts")
        if (
            artifacts[UNRESTRICTED]["manifest"]["active_trainable_parameter_count"]
            < artifacts[FACTORED]["manifest"]["active_trainable_parameter_count"]
        ):
            raise ContractError("unrestricted baseline is parameter-starved relative to factored")
        for role in range(2):
            for seed_index, seed in enumerate(target["config"]["training"]["model_seeds"]):
                factored_seed = artifacts[FACTORED]["evidence"]["roles"][role]["seeds"][seed_index]
                unrestricted_seed = artifacts[UNRESTRICTED]["evidence"]["roles"][role]["seeds"][seed_index]
                _expect(
                    factored_seed["initial_encoder_sha256"],
                    unrestricted_seed["initial_encoder_sha256"],
                    f"matched initial encoder hash role={role} seed={seed}",
                )
                _expect(
                    factored_seed["minibatch_index_stream_sha256"],
                    unrestricted_seed["minibatch_index_stream_sha256"],
                    f"matched minibatch-index stream role={role} seed={seed}",
                )
        total_wall = sum(
            artifact["manifest"]["wall_seconds"] for artifact in artifacts.values()
        )
        if (
            target["manifest"]["artifact_kind"] == "learned"
            and total_wall > target["config"]["training"]["total_hard_wall_seconds"]
        ):
            raise ContractError("full four-cell matrix exceeded the registered 45-minute wall")
        factored = _carrier_result(artifacts[FACTORED])
        unrestricted = _carrier_result(artifacts[UNRESTRICTED])
        primary = _primary_status(factored)
        comparison = _comparison_label(
            factored["all_cell_exact"], unrestricted["all_cell_exact"]
        )
        diagnostic_only = _diagnostic_block(target["config"], artifacts)
        verdict = {
            "schema_version": SCHEMA_VERDICT,
            "registration_id": REGISTRATION_ID,
            "claiming": False,
            "artifact_kind": target["manifest"]["artifact_kind"],
            "target_carrier": target_carrier,
            "status": primary,
            "reason": None,
            "primary_world_verdict": primary,
            "comparison_label": comparison,
            "carrier_results": {
                FACTORED: factored,
                UNRESTRICTED: unrestricted,
            },
            "paired_differences": _paired_differences(factored, unrestricted),
            "diagnostic_only": diagnostic_only,
            "reducer_code_sha256": _module_sha256(),
            "input_hashes": {
                "config_sha256": target["config_sha256"],
                FACTORED: {
                    **artifacts[FACTORED]["input_hashes"],
                    "weights_reproduction_sha256_not_reduced": artifacts[FACTORED]["manifest"]["weights_sha256"],
                },
                UNRESTRICTED: {
                    **artifacts[UNRESTRICTED]["input_hashes"],
                    "weights_reproduction_sha256_not_reduced": artifacts[UNRESTRICTED]["manifest"]["weights_sha256"],
                },
            },
            "generated_at": _utc_now(),
        }
    except (ContractError, OSError) as exc:
        verdict = _invalid_verdict(
            config_path, target_carrier, str(exc), target_directory, peer_directory
        )
    if write_verdict and target_directory.is_dir():
        _write_json(target_directory / "verdict.json", verdict)
    if record_ledger:
        _append_ledger(
            {
                "event_id": "round37_presentation_quotient_reduce",
                "timestamp": _utc_now(),
                "registration_id": REGISTRATION_ID,
                "purpose": "Round 37 declarative fail-closed reduction",
                "target_carrier": target_carrier,
                "status": verdict["status"],
                "comparison_label": verdict["comparison_label"],
                "artifact": str(target_directory / "verdict.json"),
                "claim": "Reducer verdict remains bounded by the Round 37 claim wall.",
            }
        )
    return verdict


def _fixture_probability_row(signature: str) -> list[float]:
    return [0.99 if bit == "1" else 0.01 for bit in signature]


def _fixture_evaluation(
    config: dict[str, Any], world: dict[str, Any], split: dict[str, Any]
) -> dict[str, Any]:
    training_probabilities = [
        0.99 if _target_for_row(row, world) == 1 else 0.01
        for row in split["training_rows"]
    ]
    base = [
        _fixture_probability_row(
            world["oracle_signatures"][world["handle_to_hidden_state"][handle] // 2]
        )
        for handle in range(32)
    ]
    descent: list[Any] = []
    for q_index in range(16):
        q_rows: list[Any] = []
        for action_id in range(11):
            successor = world["oracle_task_successor_places"][q_index][action_id]
            row = _fixture_probability_row(world["oracle_signatures"][successor])
            q_rows.append(
                [
                    [list(row), list(row), list(row)],
                    [list(row), list(row), list(row)],
                ]
            )
        descent.append(q_rows)
    heldout_pairs: list[Any] = []
    heldout_pair_step_signatures: list[Any] = []
    for word in _word_universe(config)["heldout"]:
        for q in _operational_states():
            endpoint_q, _ = _apply_word((q, 0), word)
            endpoint_signature = world["oracle_signatures"][_q_index(endpoint_q)]
            row = _fixture_probability_row(endpoint_signature)
            heldout_pairs.append([list(row), list(row)])
            heldout_pair_step_signatures.append(
                [
                    [
                        world["oracle_signatures"][
                            _q_index(_apply_word((q, 0), word[:step])[0])
                        ],
                        world["oracle_signatures"][
                            _q_index(_apply_word((q, 0), word[:step])[0])
                        ],
                    ]
                    for step in range(1, len(word) + 1)
                ]
            )
    presentation_move = [list(row) for row in base]
    evaluation = {
        "training_probabilities": training_probabilities,
        "base_signature_probabilities": base,
        "task_descent_probabilities": descent,
        "heldout_pair_probabilities": heldout_pairs,
        "heldout_pair_step_signatures": heldout_pair_step_signatures,
        "presentation_move_probabilities": presentation_move,
    }
    evaluation["endpoint_signatures"] = _reported_endpoint_signatures(
        evaluation,
        config["thresholds"]["signature_low"],
        config["thresholds"]["signature_high"],
    )
    return evaluation


def _fixture_component_trace(
    config: dict[str, Any], world: dict[str, Any]
) -> list[dict[str, Any]]:
    probabilities: list[float] = []
    losses: list[float] = []
    for handle in range(32):
        signature = world["oracle_signatures"][world["handle_to_hidden_state"][handle] // 2]
        for bit in signature:
            probability = 0.99 if bit == "1" else 0.01
            probabilities.append(probability)
            losses.append(float(-math.log(0.99)))
    return [
        {
            "step": step,
            "batch_loss": None if step == 0 else 0.0,
            "base_signature_probabilities": list(probabilities),
            "base_signature_component_bce": list(losses),
            "base_signature_component_supported": [True] * 384,
        }
        for step in TRACE_STEPS
    ]


def _fixture_seed_record(
    config: dict[str, Any],
    carrier: str,
    world: dict[str, Any],
    split: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    role = split["role"]
    token = f"round37-fixture|{carrier}|role={role}|seed={seed}"
    matched_token = f"round37-fixture-matched|role={role}|seed={seed}"
    return {
        "seed": seed,
        "steps": 32000,
        "active_trainable_parameter_count": config["model"][carrier]["active_trainable_parameters"],
        "initial_weight_sha256": _sha256_bytes(f"{token}|initial".encode("utf-8")),
        "initial_encoder_sha256": _sha256_bytes(f"{matched_token}|initial-encoder".encode("utf-8")),
        "final_weight_sha256": _sha256_bytes(f"{token}|final".encode("utf-8")),
        "minibatch_index_stream_sha256": _sha256_bytes(f"{matched_token}|minibatch-indices".encode("utf-8")),
        "loss_trace_sha256": _sha256_bytes(f"{token}|loss".encode("utf-8")),
        "component_trace": _fixture_component_trace(config, world),
        "evaluation": _fixture_evaluation(config, world, split),
    }


def _fixture_manifest(
    config: dict[str, Any],
    config_bytes: bytes,
    carrier: str,
    directory: Path,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_MANIFEST,
        "registration_id": REGISTRATION_ID,
        "artifact_kind": "fixture",
        "claiming": False,
        "carrier": carrier,
        "producer_code_sha256": _module_sha256(),
        "config_sha256": _sha256_bytes(config_bytes),
        "evidence_sha256": _sha256_file(directory / "evidence.json"),
        "weights_sha256": _sha256_file(directory / "weights.npz"),
        "git_commit": _git_commit(),
        "command": "fixture-no-torch",
        "started_at": "fixture",
        "completed_at": "fixture",
        "wall_seconds": 0.0,
        "role_cell_wall_seconds": [0.0, 0.0],
        "platform": _platform_info(),
        "dependencies": _dependency_versions(__import__("numpy"), None),
        "cpu_settings": {
            "device": "cpu",
            "threads": 1,
            "interop_threads": 1,
            "deterministic_algorithms": True,
            "parameter_devices": [],
            "cuda_visible_devices": "",
        },
        "action_order": ALL_ACTIONS,
        "word_list_sha256": {
            key: evidence["word_lists"][key]
            for key in ("training_list_sha256", "h2_list_sha256", "h3_list_sha256")
        },
        "role_split_sha256": [
            {
                "role": split["role"],
                "training_rows_sha256": split["training_rows_sha256"],
                "heldout_rows_sha256": split["heldout_rows_sha256"],
            }
            for split in evidence["role_splits"]
        ],
        "data_seed": 3701,
        "model_seeds": [11, 23, 37, 53, 71],
        "expected_counts": {
            "hidden_states": 32,
            "operational_places": 16,
            "role_folds": 2,
            "seeds_per_role": 5,
            "training_rows_per_role": 45344,
            "heldout_rows_per_role": 2240,
        },
        "active_trainable_parameter_count": config["model"][carrier]["active_trainable_parameters"],
        "seed_weight_hashes": [
            {
                "role": role["role"],
                "seeds": [
                    {
                        "seed": seed["seed"],
                        "initial": seed["initial_weight_sha256"],
                        "initial_encoder": seed["initial_encoder_sha256"],
                        "final": seed["final_weight_sha256"],
                        "minibatch_indices": seed["minibatch_index_stream_sha256"],
                    }
                    for seed in role["seeds"]
                ],
            }
            for role in evidence["roles"]
        ],
        "complete": True,
    }


def _write_fixture_artifact(
    config_path: Path, carrier: str, directory: Path
) -> None:
    import numpy as np

    directory.mkdir(parents=True, exist_ok=False)
    config_bytes = config_path.read_bytes()
    config = _validate_config(_read_json(config_path))
    (directory / "config.json").write_bytes(config_bytes)
    world = _world_truth(config)
    splits = _role_splits(config, world)
    evidence = {
        "schema_version": SCHEMA_EVIDENCE,
        "registration_id": REGISTRATION_ID,
        "artifact_kind": "fixture",
        "claiming": False,
        "carrier": carrier,
        "producer_code_sha256": _module_sha256(),
        "config_sha256": _sha256_bytes(config_bytes),
        "word_lists": _word_pack(config),
        "world": world,
        "role_splits": splits,
        "roles": [
            {
                "role": split["role"],
                "cell_wall_seconds": 0.0,
                "seeds": [
                    _fixture_seed_record(config, carrier, world, split, seed)
                    for seed in config["training"]["model_seeds"]
                ],
            }
            for split in splits
        ],
    }
    _write_json(directory / "evidence.json", evidence)
    np.savez_compressed(
        directory / "weights.npz",
        fixture_marker=np.asarray([37], dtype=np.int64),
    )
    _write_json(
        directory / "manifest.json",
        _fixture_manifest(config, config_bytes, carrier, directory, evidence),
    )


def _refresh_manifest_evidence_hash(directory: Path) -> None:
    manifest = _read_json(directory / "manifest.json")
    manifest["evidence_sha256"] = _sha256_file(directory / "evidence.json")
    _write_json(directory / "manifest.json", manifest)


def _fixture_case_pair(
    exact_root: Path, case_root: Path
) -> tuple[Path, Path]:
    factored = case_root / FACTORED
    unrestricted = case_root / UNRESTRICTED
    shutil.copytree(exact_root / FACTORED, factored)
    shutil.copytree(exact_root / UNRESTRICTED, unrestricted)
    return factored, unrestricted


def _run_fixture(config_path: Path, carrier: str, output_root: Path | None) -> dict[str, Any]:
    if carrier not in CARRIERS:
        raise ContractError(f"carrier must be one of {CARRIERS}")
    config = _validate_config(_read_json(config_path))
    torch_preloaded = "torch" in sys.modules

    def execute(exact_root: Path, tamper_root: Path) -> dict[str, Any]:
        exact_root.mkdir(parents=True, exist_ok=False)
        _write_fixture_artifact(config_path, FACTORED, exact_root / FACTORED)
        _write_fixture_artifact(config_path, UNRESTRICTED, exact_root / UNRESTRICTED)
        target = exact_root / carrier
        peer = exact_root / (UNRESTRICTED if carrier == FACTORED else FACTORED)
        exact = _reduce_directories(
            config_path,
            carrier,
            target,
            peer,
            write_verdict=False,
            record_ledger=False,
        )
        if exact["status"] != PASS_STATUS or exact["comparison_label"] != SOLVED_BY_BOTH:
            raise ContractError("exact fixture did not pass both carriers and all certificates")
        diagnostic = exact.get("diagnostic_only")
        if not isinstance(diagnostic, dict) or diagnostic.get("status_effect") != "none":
            raise ContractError("exact fixture did not emit the diagnostic-only block")
        if diagnostic.get("gate_threshold_or_verdict_logic_changes") is not False:
            raise ContractError("diagnostic-only block claims a gate or verdict change")
        for diagnostic_carrier in CARRIERS:
            units = diagnostic["carrier_units"][diagnostic_carrier]["units"]
            if len(units) != 10:
                raise ContractError("diagnostic-only carrier unit count is not 10")
            for unit in units:
                for horizon, expected_cells in (("H2", 1184), ("H3", 1056)):
                    horizon_record = unit["by_continuation"][horizon]
                    final_cells = horizon_record["final_cell_classification"]
                    if final_cells != {
                        "total": expected_cells,
                        "supported_truthful_equal": expected_cells,
                        "unsupported": 0,
                        "wrong_supported": 0,
                    }:
                        raise ContractError("exact fixture diagnostic cell accounting failed")
                    first = horizon_record["first_divergence_step_histogram"]
                    if first["none"] != expected_cells or any(
                        value != 0 for key, value in first.items() if key != "none"
                    ):
                        raise ContractError("exact fixture diagnostic divergence accounting failed")
        matched_margins = diagnostic["factored_vs_unrestricted_margins_per_seed_role"]
        if len(matched_margins) != 10:
            raise ContractError("diagnostic-only matched margin unit count is not 10")
        for unit in matched_margins:
            for horizon in ("H2", "H3"):
                for scope in (
                    "all_components",
                    "terminal_response_components",
                    "future_signature_components",
                ):
                    delta = unit["by_continuation"][horizon][scope][
                        "factored_minus_unrestricted_per_component"
                    ]
                    if any(not math.isclose(value, 0.0, abs_tol=1e-15) for value in delta.values()):
                        raise ContractError("exact fixture paired diagnostic margins are nonzero")

        cases: dict[str, dict[str, Any]] = {}

        case = tamper_root / "missing_row"
        factored, unrestricted = _fixture_case_pair(exact_root, case)
        missing = _read_json(factored / "evidence.json")
        missing["roles"][0]["seeds"][0]["evaluation"]["training_probabilities"].pop()
        _write_json(factored / "evidence.json", missing)
        _refresh_manifest_evidence_hash(factored)
        cases["missing_required_row"] = _reduce_directories(
            config_path, carrier, factored if carrier == FACTORED else unrestricted, unrestricted if carrier == FACTORED else factored, write_verdict=False, record_ledger=False
        )

        case = tamper_root / "nonfinite"
        factored, unrestricted = _fixture_case_pair(exact_root, case)
        evidence_path = factored / "evidence.json"
        raw = evidence_path.read_text(encoding="utf-8")
        replacement_count = 0
        for token in ("0.01", "0.99"):
            if token in raw:
                raw = raw.replace(token, "NaN", 1)
                replacement_count = 1
                break
        if replacement_count != 1:
            raise ContractError("fixture could not create non-finite JSON tamper")
        evidence_path.write_text(raw, encoding="utf-8", newline="\n")
        _refresh_manifest_evidence_hash(factored)
        cases["nonfinite_response"] = _reduce_directories(
            config_path, carrier, factored if carrier == FACTORED else unrestricted, unrestricted if carrier == FACTORED else factored, write_verdict=False, record_ledger=False
        )

        case = tamper_root / "hash_corruption"
        factored, unrestricted = _fixture_case_pair(exact_root, case)
        with (factored / "evidence.json").open("ab") as handle:
            handle.write(b" ")
        cases["unrehash_corruption"] = _reduce_directories(
            config_path, carrier, factored if carrier == FACTORED else unrestricted, unrestricted if carrier == FACTORED else factored, write_verdict=False, record_ledger=False
        )

        case = tamper_root / "duplicate_key"
        factored, unrestricted = _fixture_case_pair(exact_root, case)
        evidence_path = factored / "evidence.json"
        raw = evidence_path.read_text(encoding="utf-8")
        needle = '"carrier":"factored"'
        if raw.count(needle) != 1:
            raise ContractError("fixture could not create duplicate-key tamper")
        evidence_path.write_text(raw.replace(needle, f"{needle},{needle}", 1), encoding="utf-8", newline="\n")
        _refresh_manifest_evidence_hash(factored)
        cases["duplicate_json_key"] = _reduce_directories(
            config_path, carrier, factored if carrier == FACTORED else unrestricted, unrestricted if carrier == FACTORED else factored, write_verdict=False, record_ledger=False
        )

        case = tamper_root / "rehashed_scientific"
        factored, unrestricted = _fixture_case_pair(exact_root, case)
        mutated = _read_json(factored / "evidence.json")
        row = mutated["roles"][0]["seeds"][0]["evaluation"]["heldout_pair_probabilities"][0][1]
        mutated_row = [
            0.01 if value >= 0.90 else 0.99 for value in row
        ]
        mutated["roles"][0]["seeds"][0]["evaluation"]["heldout_pair_probabilities"][0][1] = mutated_row
        mutated_evaluation = mutated["roles"][0]["seeds"][0]["evaluation"]
        mutated_evaluation["heldout_pair_step_signatures"][0][-1][1] = _signature(
            mutated_row, 0.10, 0.90
        )
        mutated_evaluation["endpoint_signatures"] = _reported_endpoint_signatures(
            mutated_evaluation, 0.10, 0.90
        )
        _write_json(factored / "evidence.json", mutated)
        _refresh_manifest_evidence_hash(factored)
        cases["schema_valid_rehashed_scientific_mutation"] = _reduce_directories(
            config_path, carrier, factored if carrier == FACTORED else unrestricted, unrestricted if carrier == FACTORED else factored, write_verdict=False, record_ledger=False
        )

        case = tamper_root / "rehashed_diagnostic_only"
        factored, unrestricted = _fixture_case_pair(exact_root, case)
        diagnostic_mutation = _read_json(factored / "evidence.json")
        diagnostic_evaluation = diagnostic_mutation["roles"][0]["seeds"][0]["evaluation"]
        first_h3_cell = config["split"]["expected_h2_words"] * 16
        original_signature = diagnostic_evaluation[
            "heldout_pair_step_signatures"
        ][first_h3_cell][0][1]
        diagnostic_evaluation["heldout_pair_step_signatures"][first_h3_cell][0][1] = "".join(
            "1" if component == "0" else "0" if component == "1" else "?"
            for component in original_signature
        )
        _write_json(factored / "evidence.json", diagnostic_mutation)
        _refresh_manifest_evidence_hash(factored)
        cases["schema_valid_rehashed_diagnostic_only_mutation"] = _reduce_directories(
            config_path, carrier, factored if carrier == FACTORED else unrestricted, unrestricted if carrier == FACTORED else factored, write_verdict=False, record_ledger=False
        )

        for name in (
            "missing_required_row",
            "nonfinite_response",
            "unrehash_corruption",
            "duplicate_json_key",
        ):
            if cases[name]["status"] != INVALID_STATUS:
                raise ContractError(f"fixture branch {name} did not fail closed as INVALID")
            if cases[name]["diagnostic_only"] is not None:
                raise ContractError(f"fixture branch {name} did not serialize diagnostic_only as null")
        scientific_status = cases["schema_valid_rehashed_scientific_mutation"]["status"]
        if not isinstance(scientific_status, str) or not scientific_status.startswith("FAIL —"):
            raise ContractError("rehashed scientific mutation did not produce a scientific FAIL")
        mutated_diagnostic = cases["schema_valid_rehashed_scientific_mutation"][
            "diagnostic_only"
        ]["carrier_units"][FACTORED]["units"][0]["by_continuation"]["H2"]
        if mutated_diagnostic["first_divergence_step_histogram"]["step_2"] != 1:
            raise ContractError("diagnostic fixture did not localize the H2 final-step divergence")
        bad_components = mutated_diagnostic[
            "unsupported_vs_wrong_supported_bad_components"
        ]["total"]
        if bad_components != {"unsupported": 0, "wrong_supported": 12}:
            raise ContractError("diagnostic fixture did not separate wrong-supported components")
        failure_scope = mutated_diagnostic[
            "terminal_vs_future_signature_failure_cells"
        ]
        if failure_scope["terminal_and_future_signature"] != 1:
            raise ContractError("diagnostic fixture did not separate terminal and future failure")
        diagnostic_only_case = cases[
            "schema_valid_rehashed_diagnostic_only_mutation"
        ]
        if (
            diagnostic_only_case["status"] != PASS_STATUS
            or diagnostic_only_case["comparison_label"] != SOLVED_BY_BOTH
        ):
            raise ContractError("diagnostic-only fixture mutation changed scientific status")
        diagnostic_only_h3 = diagnostic_only_case["diagnostic_only"][
            "carrier_units"
        ][FACTORED]["units"][0]["by_continuation"]["H3"]
        if diagnostic_only_h3["first_divergence_step_histogram"]["step_1"] != 1:
            raise ContractError("diagnostic-only fixture did not report the H3 step-1 divergence")
        if diagnostic_only_h3["final_cell_classification"] != {
            "total": 1056,
            "supported_truthful_equal": 1056,
            "unsupported": 0,
            "wrong_supported": 0,
        }:
            raise ContractError("diagnostic-only fixture mutation changed final-cell accounting")
        if not torch_preloaded and "torch" in sys.modules:
            raise ContractError("fixture imported Torch")
        return {
            "status": "FIXTURE PASS",
            "claiming": False,
            "carrier_cli_target": carrier,
            "exact_primary_status": exact["status"],
            "exact_comparison_label": exact["comparison_label"],
            "tamper_statuses": {name: result["status"] for name, result in cases.items()},
            "diagnostic_only_exercised": {
                "history_depth_and_h2_h3": True,
                "first_divergence_step": "H2 step_2",
                "terminal_vs_future_signature": True,
                "unsupported_vs_wrong_supported": True,
                "matched_seed_role_margins": True,
                "diagnostic_mutation_preserved_verdict": True,
            },
            "torch_imported": False,
        }

    with _temporary_directory("round37_fixture_tamper_") as tamper_root:
        if output_root is not None:
            _prepare_empty_directory(output_root)
            exact_root = output_root / "exact_pair"
            return execute(exact_root, tamper_root)
        with _temporary_directory("round37_fixture_exact_") as temporary:
            return execute(temporary / "exact_pair", tamper_root)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    produce = subcommands.add_parser(
        "produce",
        help="train one carrier across both role folds and serialize non-claiming evidence",
    )
    produce.add_argument("--config", type=Path, required=True)
    produce.add_argument("--carrier", choices=CARRIERS, required=True)
    produce.add_argument("--out", type=Path, required=True)

    reduce = subcommands.add_parser(
        "reduce",
        help="declaratively reduce matched factored and unrestricted evidence",
    )
    reduce.add_argument("--config", type=Path, required=True)
    reduce.add_argument("--carrier", choices=CARRIERS, required=True)
    reduce.add_argument("--evidence", type=Path, required=True)
    reduce.add_argument("--peer-evidence", type=Path, required=True)

    fixture = subcommands.add_parser(
        "fixture",
        help="exercise exact and fail-closed synthetic branches without Torch",
    )
    fixture.add_argument("--config", type=Path, required=True)
    fixture.add_argument("--carrier", choices=CARRIERS, required=True)
    fixture.add_argument("--out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "produce":
            result = _produce(args.config, args.carrier, args.out)
        elif args.command == "reduce":
            result = _reduce_directories(
                args.config,
                args.carrier,
                args.evidence,
                args.peer_evidence,
                write_verdict=True,
                record_ledger=True,
            )
        else:
            result = _run_fixture(args.config, args.carrier, args.out)
        print(_canonical_bytes(result).decode("utf-8"), end="")
        if args.command == "reduce":
            if result["status"] == PASS_STATUS:
                return 0
            if isinstance(result["status"], str) and result["status"].startswith("FAIL —"):
                return 1
            return 2
        return 0
    except BudgetExceeded as exc:
        print(f"producer incomplete: {exc}", file=sys.stderr)
        return 2
    except (ContractError, OSError) as exc:
        print(f"invalid invocation/artifact: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
