"""Run a bounded pretrained native-memory addressing pilot on a revision stream.

The publish command sees only the source stream. The read command is a separate
process and receives tasks only after the source-only publications exist. The
gold file is opened only by score. This is a visible synthetic pilot, not a
held-task or general-capability result.
"""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.memory.linear_feedback import LinearFeedbackKernel
from experiments.memory.native_latent_recurrence import NativeLatentModel
from experiments.memory.native_publication import read_native_publication, write_native_publication
from experiments.memory.source_revision_feed import write_source_delta
from experiments.memory.verified_source_tool import sha256


MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_PATH = Path(os.environ.get(
    "LSR_QWEN3_0_6B_SNAPSHOT",
    Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-0.6B"
    / "snapshots" / MODEL_REVISION,
))
HERE = Path(__file__).resolve().parent
DEFAULT_SOURCES = HERE / "native_world_addressing_sources.json"
DEFAULT_TASKS = HERE / "native_world_addressing_tasks.json"
DEFAULT_GOLD = HERE / "native_world_addressing_gold.json"
THREADS = 12
STEPS = 2
OUTPUT_TOKENS = 2
CONTEXT_TOKENS = 8192
SOURCE_CHARS = 512
TOP_K = 1
TIE_LIMIT = 16
SOURCE_BUDGET = 1 << 20
PUBLICATION_BUDGET = 64 << 20
IDENTITY_GAIN = 1.0


class NoThinkingTokenizer:
    def __init__(self, inner):
        self.inner = inner

    def apply_chat_template(self, messages, **kwargs):
        kwargs["enable_thinking"] = False
        return self.inner.apply_chat_template(messages, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _load_json(path, label):
    try:
        value = json.loads(Path(path).read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {label}: {error}") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf8")


def _runtime_digest():
    modules = (
        "latent_feedback.py", "linear_feedback.py", "native_forecast_set.py",
        "native_latent_recurrence.py", "native_publication.py", "selective_continuations.py",
        "source_revision_feed.py", "verified_reader_bridge.py", "verified_source_tool.py",
    )
    code = {name: sha256((HERE / name).read_bytes()) for name in modules}
    settings = {
        "schema": "lsr-native-world-addressing-runtime-v1",
        "model_revision": MODEL_REVISION,
        "implementation_sha256": sha256(Path(__file__).read_bytes()),
        "module_sha256": code,
        "kernel": "identity-query-key-value-feedback",
        "latent_steps": STEPS,
        "output_tokens": OUTPUT_TOKENS,
        "context_tokens": CONTEXT_TOKENS,
        "source_chars": SOURCE_CHARS,
        "top_k": TOP_K,
        "tie_limit": TIE_LIMIT,
        "representation_scale": IDENTITY_GAIN,
        "memory_gain": 1.0,
        "null_weight": 0.1,
        "relay_gain": 1.0,
        "derived_gain": 0.5,
    }
    return sha256(_canonical(settings)), settings


def _preflight_resources(output=None):
    import psutil

    available = psutil.virtual_memory().available
    if available < 12 * (1 << 30):
        raise MemoryError(f"native CPU pilot requires 12 GiB available RAM; found {available}")
    if output is not None:
        free = __import__("shutil").disk_usage(Path(output).resolve().parent).free
        if free < 2 * (1 << 30):
            raise OSError(f"pilot output needs 2 GiB free disk; found {free}")
    return {"available_ram_bytes": available, "logical_cpus": psutil.cpu_count(logical=True)}


def _model_path():
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"pinned local model snapshot is unavailable: {MODEL_PATH}")
    return MODEL_PATH


def _load_reader():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _preflight_resources()
    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    start = perf_counter()
    base_tokenizer = AutoTokenizer.from_pretrained(
        _model_path(), local_files_only=True, trust_remote_code=False,
    )
    tokenizer = NoThinkingTokenizer(base_tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        _model_path(), local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cpu").eval()
    model_load_seconds = perf_counter() - start
    width = int(model.config.hidden_size)
    identity = np.eye(width, dtype=np.float64)
    kernel = LinearFeedbackKernel(
        query=identity, carry=None,
        observation_key=identity, observation_value=identity,
        feedback_key=identity, feedback_value=identity,
        max_steps=STEPS,
    )
    reader = NativeLatentModel(
        model, tokenizer, kernel,
        source_chars=SOURCE_CHARS, context_tokens=CONTEXT_TOKENS,
        latent_steps=STEPS, output_tokens=OUTPUT_TOKENS,
        representation_scale=IDENTITY_GAIN, memory_gain=1.0, null_weight=0.1,
        eos_ids=(int(base_tokenizer.eos_token_id),), relay_gain=1.0,
        derived_gain=0.5, top_k=TOP_K, tie_limit=TIE_LIMIT, temperature=0.1,
        task_template=base_tokenizer.get_chat_template(),
    )
    return reader, base_tokenizer, model_load_seconds


def _read_source_stream(path):
    data = _load_json(path, "source-only stream")
    if (set(data) != {"schema", "stream_id", "worlds"}
            or data["schema"] != "lsr-native-world-source-stream-v1"
            or type(data["stream_id"]) is not str or type(data["worlds"]) is not list
            or not data["worlds"]):
        raise ValueError("source-only stream has an invalid shape")
    ids, snapshots, previous = set(), {}, {}
    pins = []
    for row in data["worlds"]:
        if type(row) is not dict or set(row) != {"pin", "changes"}:
            raise ValueError("source-only world rows may contain only a pin and source changes")
        pin, changes = row["pin"], row["changes"]
        if type(pin) is not str or pin in snapshots or type(changes) is not dict or not changes:
            raise ValueError("source-only pin or change set is invalid")
        current = dict(previous)
        for source_id, text in changes.items():
            if (type(source_id) is not str or len(source_id) != 32
                    or any(char not in "0123456789abcdef" for char in source_id)):
                raise ValueError("source IDs must be opaque lowercase 128-bit hex values")
            if text is None:
                if source_id not in current:
                    raise ValueError("source stream withdraws a source that is not present")
                del current[source_id]
            elif type(text) is str and text.strip():
                current[source_id] = text
            else:
                raise ValueError("source changes must contain nonempty UTF-8 text or null")
        if not current:
            raise ValueError("source stream cannot publish an empty corpus")
        snapshots[pin] = current
        pins.append((pin, changes))
        previous = current
        ids.update(changes)
    if len(ids) != 5 or len(pins) != 3:
        raise ValueError("the registered pilot requires five sources across three source pins")
    return data, pins, snapshots


def _outside_repo(path):
    path = Path(path).resolve()
    repo = HERE.parents[1].resolve()
    if path == repo or repo in path.parents:
        raise ValueError("model artifacts must be written outside the source checkout")
    return path


def publish(source_path, output_path):
    """Build the source-only world without opening the task or evaluator files."""
    output = _outside_repo(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite a prior pilot directory: {output}")
    resource = _preflight_resources(output)
    sources, pin_changes, _ = _read_source_stream(source_path)
    reader, tokenizer, model_load_seconds = _load_reader()
    runtime_sha, runtime = _runtime_digest()
    feed_root, publication_root = output / "source_feeds", output / "publications"
    feed_root.mkdir(parents=True)
    publication_root.mkdir()
    previous_delta, memory, pins = None, None, []
    totals = {"source_encode_seconds": 0.0, "maintenance_seconds": 0.0,
              "publication_seconds": 0.0, "backbone_calls": 0, "backbone_input_positions": 0}
    base_model = reader._model.base_model

    def count_backbone(_module, args, kwargs):
        tensors = kwargs.get("inputs_embeds")
        if tensors is None and args:
            tensors = args[0]
        if tensors is not None:
            totals["backbone_calls"] += 1
            totals["backbone_input_positions"] += int(tensors.shape[1])

    handle = base_model.register_forward_pre_hook(count_backbone, with_kwargs=True)
    try:
        for pin, changes in pin_changes:
            feed = feed_root / pin
            encoded = {source_id: None if body is None else body.encode("utf8")
                       for source_id, body in changes.items()}
            feed_start = perf_counter()
            delta = write_source_delta(feed, encoded, previous=previous_delta, max_bytes=SOURCE_BUDGET)
            before_calls, before_positions = totals["backbone_calls"], totals["backbone_input_positions"]
            start = perf_counter()
            memory = reader.publish(feed, delta.seal, previous=memory, max_source_bytes=SOURCE_BUDGET)
            totals["source_encode_seconds"] += perf_counter() - start
            encode_calls = totals["backbone_calls"] - before_calls
            encode_positions = totals["backbone_input_positions"] - before_positions
            before_pending = memory.pending_seeds
            before_calls, before_positions = totals["backbone_calls"], totals["backbone_input_positions"]
            start = perf_counter()
            memory = reader.maintain(memory, max_seeds=max(1, memory.rows))
            totals["maintenance_seconds"] += perf_counter() - start
            maintenance_calls = totals["backbone_calls"] - before_calls
            maintenance_positions = totals["backbone_input_positions"] - before_positions
            pub_dir = publication_root / pin
            start = perf_counter()
            publication_sha = write_native_publication(
                pub_dir, reader, memory, runtime_sha256=runtime_sha,
                max_bytes=PUBLICATION_BUDGET,
            )
            totals["publication_seconds"] += perf_counter() - start
            pins.append({
                "pin": pin,
                "source_seal": memory.seal,
                "publication_seal": publication_sha,
                "rows": memory.rows,
                "pending_before_maintenance": before_pending,
                "pending_after_maintenance": memory.pending_seeds,
                "installed_continuations": len(memory.derived_seeds),
                "source_encode_backbone_calls": encode_calls,
                "source_encode_backbone_input_positions": encode_positions,
                "maintenance_backbone_calls": maintenance_calls,
                "maintenance_backbone_input_positions": maintenance_positions,
                "pin_wall_seconds": perf_counter() - feed_start,
            })
            previous_delta = delta
    finally:
        handle.remove()

    index = {
        "schema": "lsr-native-world-publication-index-v1",
        "source_stream_sha256": sha256(Path(source_path).read_bytes()),
        "runtime_sha256": runtime_sha,
        "runtime": runtime,
        "model_revision": MODEL_REVISION,
        "model_load_seconds": model_load_seconds,
        "torch": __import__("torch").__version__,
        "transformers": __import__("transformers").__version__,
        "tokenizer_template_sha256": sha256(reader._task_template.encode("utf8")),
        "device": "cpu (NativeLatentModel enforces CPU for its retained NumPy state)",
        "cpu_threads": THREADS,
        "limits": {"source_bytes_per_epoch": SOURCE_BUDGET,
                   "publication_bytes_per_pin": PUBLICATION_BUDGET,
                   "latent_steps": STEPS, "maintenance_seed_budget": "all current source rows"},
        "resource_preflight": resource,
        "usage": totals,
        "pins": pins,
    }
    temporary = output / "publication_index.pending"
    temporary.write_bytes(_canonical(index))
    temporary.replace(output / "publication_index.json")
    print(json.dumps(index, ensure_ascii=False, indent=2))


def _choice_constraint(prompt_ids, choice_ids, eos_id):
    def allow(prefix):
        prefix = tuple(prefix)
        if prefix[:len(prompt_ids)] != tuple(prompt_ids):
            raise ValueError("decoder prefix differs from the frozen task input")
        suffix = prefix[len(prompt_ids):]
        if not suffix:
            return choice_ids
        if len(suffix) == 1 and suffix[0] in choice_ids:
            return (eos_id,)
        raise ValueError("choice decoder received an invalid generated prefix")
    return allow


def _evidence_rows(memory, read):
    spans = memory.evidence_spans(read)
    return [dict(source_id=span.source_id, source_version=span.version_sha256,
                 start_byte=span.start_byte, end_byte=span.end_byte,
                 span_sha256=span.span_sha256)
            for span in spans if span is not None]


def read_tasks(source_path, task_path, publication_path, output_path, *, task_id=None):
    """Read published source memory only after opening the separate task file."""
    output = _outside_repo(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prior reader output: {output}")
    resource = _preflight_resources(output)
    source_data, _, snapshots = _read_source_stream(source_path)
    task_data = _load_json(task_path, "late task set")
    if (set(task_data) != {"schema", "tasks"}
            or task_data["schema"] != "lsr-native-world-task-set-v1"
            or type(task_data["tasks"]) is not list):
        raise ValueError("late task set has an invalid shape")
    tasks = task_data["tasks"]
    if any(type(task) is not dict or set(task) != {"task_id", "pin", "task"}
           for task in tasks):
        raise ValueError("late tasks may contain only task_id, pin and task")
    if task_id is not None:
        tasks = [task for task in tasks if task["task_id"] == task_id]
        if not tasks:
            raise ValueError(f"unknown task ID {task_id}")

    index = _load_json(Path(publication_path) / "publication_index.json", "publication index")
    if (index.get("schema") != "lsr-native-world-publication-index-v1"
            or index.get("source_stream_sha256") != sha256(Path(source_path).read_bytes())):
        raise ValueError("publication index does not match this source stream")
    reader, tokenizer, model_load_seconds = _load_reader()
    runtime_sha, runtime = _runtime_digest()
    if runtime_sha != index.get("runtime_sha256") or runtime != index.get("runtime"):
        raise ValueError("reader runtime differs from the published world")
    token_ids = [tokenizer.encode(label, add_special_tokens=False) for label in ("1", "2", "3")]
    if any(len(ids) != 1 for ids in token_ids):
        raise ValueError("fixed answer labels must each be one token")
    choice_ids = tuple(ids[0] for ids in token_ids)
    if len(set(choice_ids)) != 3:
        raise ValueError("answer label token IDs must be distinct")
    eos = int(tokenizer.eos_token_id)
    pin_rows = {row["pin"]: row for row in index["pins"]}
    results_root = output / "reads.jsonl"
    output.mkdir(parents=True)
    totals = {"backbone_calls": 0, "backbone_input_positions": 0}
    generation_seconds = 0.0
    arms = ("active", "derived_off", "memory_omitted", "full_text")

    def count_backbone(_module, args, kwargs):
        tensors = kwargs.get("inputs_embeds")
        if tensors is None and args:
            tensors = args[0]
        if tensors is not None:
            totals["backbone_calls"] += 1
            totals["backbone_input_positions"] += int(tensors.shape[1])

    hook = reader._model.base_model.register_forward_pre_hook(count_backbone, with_kwargs=True)

    try:
        with results_root.open("x", encoding="utf8") as handle:
            for task in tasks:
                pin_row = pin_rows.get(task["pin"])
                if pin_row is None or task["pin"] not in snapshots:
                    raise ValueError("task pin is absent from the source publication")
                publication_dir = Path(publication_path) / "publications" / task["pin"]
                memory = read_native_publication(
                    publication_dir, reader, seal=pin_row["publication_seal"],
                    source_seal=pin_row["source_seal"], runtime_sha256=runtime_sha,
                    max_bytes=PUBLICATION_BUDGET,
                )
                variants = {
                    "active": memory,
                    "derived_off": reader.without_derived(memory),
                    "memory_omitted": None,
                    "full_text": None,
                }
                for arm in arms:
                    task_text = task["task"]
                    if arm == "full_text":
                        corpus = "\n".join(
                            f"[source {source_id}] {body}"
                            for source_id, body in sorted(snapshots[task["pin"]].items())
                        )
                        task_text = f"{task_text}\n\nPinned corpus sources:\n{corpus}"
                    prompt_ids = reader.task_token_ids(task_text, steps=STEPS)
                    decoded_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
                    if task_text not in decoded_prompt:
                        raise ValueError("tokenized prompt does not preserve the exact task and corpus text")
                    call_before = totals["backbone_calls"]
                    positions_before = totals["backbone_input_positions"]
                    start = perf_counter()
                    read = reader.read_token_ids(
                        prompt_ids, memory=variants[arm], steps=STEPS,
                        max_new_tokens=OUTPUT_TOKENS,
                        allowed_next_tokens=_choice_constraint(prompt_ids, choice_ids, eos),
                    )
                    seconds = perf_counter() - start
                    generation_seconds += seconds
                    record = {
                        "task_id": task["task_id"], "pin": task["pin"], "arm": arm,
                        "task": task["task"], "input_tokens": len(prompt_ids),
                        "output_token_ids": list(read.token_ids), "prediction": read.text.strip(),
                        "terminated": read.terminated, "wall_seconds": seconds,
                        "backbone_calls": totals["backbone_calls"] - call_before,
                        "backbone_input_positions": totals["backbone_input_positions"] - positions_before,
                        "read": read.json_record(),
                        "selected_source_spans": (
                            _evidence_rows(variants[arm], read) if variants[arm] is not None else []
                        ),
                    }
                    record["delivered_source_ids"] = (
                        sorted(snapshots[task["pin"]]) if arm == "full_text" else
                        sorted({row["source_id"] for row in record["selected_source_spans"]})
                    )
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
    finally:
        hook.remove()

    run = {
        "schema": "lsr-native-world-addressing-read-v1",
        "source_stream_sha256": sha256(Path(source_path).read_bytes()),
        "task_set_sha256": sha256(Path(task_path).read_bytes()),
        "publication_index_sha256": sha256((Path(publication_path) / "publication_index.json").read_bytes()),
        "runtime_sha256": runtime_sha,
        "model_revision": MODEL_REVISION,
        "model_load_seconds": model_load_seconds,
        "torch": __import__("torch").__version__,
        "transformers": __import__("transformers").__version__,
        "device": "cpu",
        "cpu_threads": THREADS,
        "decoder": "greedy; first token constrained to labels 1/2/3, then EOS",
        "limits": {"calls_per_task": len(arms), "latent_steps": STEPS,
                   "output_tokens": OUTPUT_TOKENS, "context_tokens": CONTEXT_TOKENS},
        "resource_preflight": resource,
        "usage": {"backbone_calls": totals["backbone_calls"],
                  "backbone_input_positions": totals["backbone_input_positions"],
                  "generation_wall_seconds": generation_seconds},
        "read_records": results_root.name,
    }
    (output / "run.json").write_bytes(_canonical(run))
    print(json.dumps(run, ensure_ascii=False, indent=2))


def score(read_path, gold_path, output_path):
    gold = _load_json(gold_path, "evaluator gold")
    if gold.get("schema") != "lsr-native-world-evaluator-gold-v1":
        raise ValueError("evaluator gold schema is invalid")
    expected = gold["answers"]
    critical = set(gold["critical_source_ids"])
    records = [json.loads(line) for line in Path(read_path).read_text(encoding="utf8").splitlines() if line]
    rows, summary = [], {}
    for arm in ("active", "derived_off", "memory_omitted", "full_text"):
        subset = [row for row in records if row["arm"] == arm]
        correct = sum(row["prediction"] == expected[row["task_id"]] for row in subset)
        evidence_covered = sum(critical <= set(row["delivered_source_ids"])
                               for row in subset)
        summary[arm] = {
            "tasks": len(subset), "correct": correct,
            "task_ids": sorted(row["task_id"] for row in subset),
            "accuracy": None if not subset else correct / len(subset),
            "critical_source_coverage": evidence_covered,
            "routed_derived_steps": sum(row["read"]["routed_derived_steps"] for row in subset),
            "applied_derived_steps": sum(row["read"]["applied_derived_steps"] for row in subset),
            "mean_wall_seconds": (None if not subset else
                                  sum(row["wall_seconds"] for row in subset) / len(subset)),
        }
    by_task = {task_id: {row["arm"]: row for row in records if row["task_id"] == task_id}
               for task_id in expected}
    version_sequence = []
    for task_id in ("glass-arch-p0", "glass-arch-p1", "glass-arch-p2"):
        active = by_task.get(task_id, {}).get("active")
        if active:
            version_sequence.append(active["prediction"])
    sequence_task_ids = [task_id for task_id in
                         ("glass-arch-p0", "glass-arch-p1", "glass-arch-p2")
                         if by_task.get(task_id, {}).get("active")]
    sequence_complete = len(sequence_task_ids) == 3
    causal = {
        task_id: {
            "active": by_task.get(task_id, {}).get("active", {}).get("prediction"),
            "derived_off": by_task.get(task_id, {}).get("derived_off", {}).get("prediction"),
            "memory_omitted": by_task.get(task_id, {}).get("memory_omitted", {}).get("prediction"),
            "full_text": by_task.get(task_id, {}).get("full_text", {}).get("prediction"),
        }
        for task_id in expected if task_id in by_task
    }
    result = {
        "schema": "lsr-native-world-addressing-score-v1",
        "read_records_sha256": sha256(Path(read_path).read_bytes()),
        "gold_sha256": sha256(Path(gold_path).read_bytes()),
        "critical_source_ids": sorted(critical),
        "arms": summary,
        "active_version_sequence": version_sequence,
        "active_version_sequence_task_ids": sequence_task_ids,
        "active_version_sequence_complete": sequence_complete,
        "active_version_sequence_matches_gold": (
            None if not sequence_complete else
            version_sequence == [expected[key] for key in ("glass-arch-p0", "glass-arch-p1", "glass-arch-p2")]
        ),
        "paired_predictions": causal,
        "interpretation_limit": "One visible synthetic three-hop relation family; no general capability or latent-specific advantage follows from this pilot.",
    }
    target = Path(output_path).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite score output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_canonical(result))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    pub = sub.add_parser("publish", help="build source-only publications")
    pub.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    pub.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("read", help="read the published world with late tasks")
    run.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    run.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    run.add_argument("--publication", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--task-id")
    grade = sub.add_parser("score", help="open evaluator gold after inference")
    grade.add_argument("--reads", type=Path, required=True)
    grade.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    grade.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "publish":
        publish(args.sources, args.output)
    elif args.mode == "read":
        read_tasks(args.sources, args.tasks, args.publication, args.output, task_id=args.task_id)
    else:
        score(args.reads, args.gold, args.output)


if __name__ == "__main__":
    main()
