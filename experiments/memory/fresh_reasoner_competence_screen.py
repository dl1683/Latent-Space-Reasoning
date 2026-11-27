"""Blinded U/N/E reader-competence diagnostic on fresh, pinned task inputs.

U sees only the broad task and complete source packet. N also sees a neutral
obligation list. E sees N plus verbatim highlighted spans in the same source
packet. Every session is stateless, uses the same 0.6B reader and requested
JSON format, and is recorded without loading evaluator gold. JSON is
validated after generation; the sampler does not mask output tokens.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time

from experiments.memory.verified_reader_bridge import encode_messages
from experiments.memory.verified_source_tool import sha256


HERE = Path(__file__).resolve().parent
SOURCE_PATH = HERE / "fresh_reasoner_diagnostic_sources.json"
TASK_PATH = HERE / "fresh_reasoner_diagnostic_tasks.json"
NEUTRAL_PATH = HERE / "fresh_reasoner_diagnostic_neutral.json"
MARKER_PATH = HERE / "fresh_reasoner_diagnostic_markers.json"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_PATH = Path(os.environ.get(
    "LSR_QWEN3_0_6B_SNAPSHOT",
    Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-0.6B"
    / "snapshots" / MODEL_REVISION,
))
THREADS = 12
CONTEXT_TOKENS = 8192
OUTPUT_TOKENS = 1024
ORDER_SEED = 20261127
THERMAL_STOP_C = 82
ARMS = ("U", "N", "E")
SAFE_RE = re.compile(r"[-A-Za-z0-9 _.,:;!?()+/:]+\Z")
NEED_FIELDS = (
    "id", "target", "relation", "scope", "evidence_sought", "support",
    "defeat", "alternatives", "unresolved",
)
FINDING_FIELDS = ("need_id", "stance", "claim", "citations")
STANCES = ("supports", "defeats", "unknown", "conflicted", "hypothetical")


def _canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("ascii")


def _load_json(path, expected_schema):
    value = json.loads(Path(path).read_text(encoding="utf8"))
    if type(value) is not dict or value.get("schema") != expected_schema:
        raise ValueError(f"unexpected schema in {path.name}")
    return value


def _input_hashes():
    return {
        path.name: sha256(path.read_bytes())
        for path in (Path(__file__), SOURCE_PATH, TASK_PATH, NEUTRAL_PATH, MARKER_PATH)
    }


def _outside_repo(path):
    path = Path(path).resolve()
    repo = HERE.parents[1].resolve()
    if path == repo or repo in path.parents:
        raise ValueError("model artifacts must be written outside the source checkout")
    return path


def _maps():
    source = _load_json(SOURCE_PATH, "lsr-fresh-reasoner-source-packets-v1")
    tasks = _load_json(TASK_PATH, "lsr-fresh-reasoner-task-pins-v1")
    neutral = _load_json(NEUTRAL_PATH, "lsr-fresh-reasoner-neutral-obligations-v1")
    markers = _load_json(MARKER_PATH, "lsr-fresh-reasoner-verbatim-markers-v1")

    corpora = {}
    for corpus in source.get("corpora", ()):
        case_id = corpus.get("case_id")
        if type(case_id) is not str or case_id in corpora:
            raise ValueError("source corpus IDs must be unique strings")
        pins = {}
        for version in corpus.get("pins", ()):
            pin = version.get("pin")
            rows = version.get("sources")
            if type(pin) is not str or pin in pins or type(rows) is not list or not rows:
                raise ValueError("each source pin needs one unique pin and a nonempty ordered source list")
            ids = [row.get("source_id") for row in rows]
            if len(ids) != len(set(ids)) or any(
                type(row.get("title")) is not str or type(row.get("text")) is not str
                or not row["title"] or not row["text"] for row in rows
            ):
                raise ValueError("source IDs must be unique and source text/title nonempty")
            pins[pin] = rows
        corpora[case_id] = pins

    task_rows = tasks.get("tasks")
    if type(task_rows) is not list or len(task_rows) != 6:
        raise ValueError("the frozen screen must contain exactly six task-pin cases")
    task_by_id = {}
    for task in task_rows:
        task_id = task.get("task_id")
        case_id, pin = task.get("case_id"), task.get("pin")
        if (type(task_id) is not str or task_id in task_by_id or case_id not in corpora
                or pin not in corpora[case_id] or task.get("kind") not in ("direct", "composed")
                or type(task.get("task")) is not str or not task["task"]):
            raise ValueError("task rows must bind unique IDs to an existing source pin")
        task_by_id[task_id] = task
    counts = {kind: sum(row["kind"] == kind for row in task_rows)
              for kind in ("direct", "composed")}
    if counts != {"direct": 2, "composed": 4}:
        raise ValueError("the frozen set must contain two direct and four composed task-pins")

    neutral_by_id = {}
    for row in neutral.get("obligations", ()):
        task_id, items = row.get("task_id"), row.get("items")
        if (task_id in neutral_by_id or task_id not in task_by_id or type(items) is not list
                or not items or any(type(item) is not str or not item for item in items)):
            raise ValueError("neutral obligations must be nonempty and unique per task-pin")
        neutral_by_id[task_id] = items

    marker_by_id = {}
    for row in markers.get("markers", ()):
        task_id, spans = row.get("task_id"), row.get("spans")
        if (task_id in marker_by_id or task_id not in task_by_id or type(spans) is not list
                or not spans):
            raise ValueError("verbatim markers must be nonempty and unique per task-pin")
        marker_by_id[task_id] = spans

    if set(neutral_by_id) != set(task_by_id) or set(marker_by_id) != set(task_by_id):
        raise ValueError("each task-pin needs one neutral list and one E-arm span list")
    return task_rows, task_by_id, corpora, neutral_by_id, marker_by_id


def _marked_source_rows(task_id, source_rows, spans):
    source_order = {row["source_id"]: index for index, row in enumerate(source_rows)}
    previous_key = (-1, -1)
    by_id = {}
    for span in spans:
        source_id, text = span.get("source_id"), span.get("text")
        if source_id not in source_order or type(text) is not str or not text:
            raise ValueError(f"invalid verbatim span in {task_id}")
        body = next(row["text"] for row in source_rows if row["source_id"] == source_id)
        if body.count(text) != 1:
            raise ValueError(f"span must occur exactly once in its source: {task_id}/{source_id}")
        start = body.index(text)
        key = (source_order[source_id], start)
        if key <= previous_key:
            raise ValueError(f"E-arm spans must be in source and text order: {task_id}")
        previous_key = key
        by_id.setdefault(source_id, []).append((start, start + len(text)))

    result = []
    for row in source_rows:
        body = row["text"]
        ranges = by_id.get(row["source_id"], ())
        cursor, pieces = 0, []
        for start, end in ranges:
            if start < cursor:
                raise ValueError(f"overlapping marked spans: {task_id}/{row['source_id']}")
            pieces.extend((body[cursor:start], "[[", body[start:end], "]]"))
            cursor = end
        pieces.append(body[cursor:])
        result.append({**row, "text": "".join(pieces)})
    return result


def _system_text():
    return (
        "You are a source-grounded reasoner. Use only this task and its complete pinned source packet. "
        "First declare one to four material evidence needs, before writing findings. A need states "
        "its target, relation, time or version scope, evidence sought, what would support it, what "
        "would defeat it, plausible alternatives, and unresolved conditions. Then write separate "
        "findings linked to those need IDs, with a stance, concise claim, and source IDs from the packet. "
        "Do not treat a source read or a matching phrase as proof. Preserve conflicts and unknowns. "
        "Do not invent facts, evidence, or citations. "
        "Return exactly one JSON object with keys needs, findings, answer, open_questions. Each need "
        "has keys id,target,relation,scope,evidence_sought,support,defeat,alternatives,unresolved. "
        "Need IDs are n1 through n4. Each finding has keys need_id,stance,claim,citations. Stance is "
        "supports, defeats, unknown, conflicted, or hypothetical. Answer is a concise response to the "
        "task; open_questions lists only material facts still unknown. Use plain ASCII text."
    )


def _prompt(task, sources, arm, neutral_items):
    chunks = [_system_text(), f"Task: {task['task']}", f"Pinned version: {task['pin']}"]
    if arm in ("N", "E"):
        obligations = "\n".join(f"- {item}" for item in neutral_items)
        chunks.append("Neutral obligations to consider (not facts or answers):\n" + obligations)
    corpus = "\n".join(
        f"[source {row['source_id']} | {row['title']}] {row['text']}"
        for row in sources
    )
    if arm == "E":
        chunks.append("Relevant spans are bracketed [[like this]] in source order. They can support or defeat a finding; they do not state the answer.")
    chunks.append("Complete pinned corpus:\n" + corpus)
    return "\n\n".join(chunks)


def _all_prompts():
    task_rows, task_by_id, corpora, neutral, markers = _maps()
    prompts = {}
    for task in task_rows:
        task_id = task["task_id"]
        source_rows = corpora[task["case_id"]][task["pin"]]
        for arm in ARMS:
            view = source_rows
            if arm == "E":
                view = _marked_source_rows(task_id, source_rows, markers[task_id])
            prompts[(task_id, arm)] = _prompt(task, view, arm, neutral[task_id])
    return task_rows, task_by_id, corpora, neutral, markers, prompts


def _validate_text(value, maximum, *, allow_empty=False):
    return (type(value) is str and value.isascii() and len(value) <= maximum
            and (allow_empty or bool(value))
            and (allow_empty and value == "" or bool(SAFE_RE.fullmatch(value))))


def _validate_reply(value, source_ids):
    if type(value) is not dict or set(value) != {"needs", "findings", "answer", "open_questions"}:
        raise ValueError("output must contain only needs, findings, answer, and open_questions")
    needs, findings = value["needs"], value["findings"]
    if type(needs) is not list or not 1 <= len(needs) <= 4:
        raise ValueError("need count must be one to four")
    need_ids = []
    for need in needs:
        if type(need) is not dict or set(need) != set(NEED_FIELDS):
            raise ValueError("need does not match the registered fields")
        if need["id"] not in ("n1", "n2", "n3", "n4") or need["id"] in need_ids:
            raise ValueError("need IDs must be unique members of n1 through n4")
        need_ids.append(need["id"])
        for field, maximum in (("target", 48), ("relation", 48), ("scope", 48)):
            if not _validate_text(need[field], maximum):
                raise ValueError(f"invalid need text in {field}")
        for field, maximum, max_items, min_items in (
            ("evidence_sought", 36, 2, 1), ("support", 36, 1, 0),
            ("defeat", 36, 1, 0), ("alternatives", 36, 2, 0),
            ("unresolved", 36, 1, 0),
        ):
            items = need[field]
            if (type(items) is not list or not min_items <= len(items) <= max_items
                    or len(items) != len(set(items))
                    or any(not _validate_text(item, maximum) for item in items)):
                raise ValueError(f"invalid need list in {field}")
    if type(findings) is not list or not 1 <= len(findings) <= 4:
        raise ValueError("finding count must be one to four")
    for finding in findings:
        if type(finding) is not dict or set(finding) != set(FINDING_FIELDS):
            raise ValueError("finding does not match the registered fields")
        if finding["need_id"] not in need_ids or finding["stance"] not in STANCES:
            raise ValueError("finding must reference a declared need and known stance")
        if not _validate_text(finding["claim"], 112):
            raise ValueError("invalid finding claim")
        citations = finding["citations"]
        if (type(citations) is not list or len(citations) > 4
                or len(citations) != len(set(citations))
                or any(value not in source_ids for value in citations)):
            raise ValueError("finding citations must be unique IDs in this pinned corpus")
    if not _validate_text(value["answer"], 240):
        raise ValueError("invalid answer text")
    questions = value["open_questions"]
    if (type(questions) is not list or len(questions) > 3
            or len(questions) != len(set(questions))
            or any(not _validate_text(item, 96) for item in questions)):
        raise ValueError("invalid open_questions list")
    return value


def _fixture(source_id):
    needs = []
    for index in range(1, 5):
        needs.append({
            "id": f"n{index}", "target": "target object current version",
            "relation": "material relation to task conclusion", "scope": "current pinned source version",
            "evidence_sought": [f"direct source evidence {index}"],
            "support": [f"support condition {index}"], "defeat": [f"defeat condition {index}"],
            "alternatives": [f"plausible alternative {index}"],
            "unresolved": [f"unresolved condition {index}"],
        })
    findings = [{
        "need_id": f"n{index}", "stance": "supports",
        "claim": f"Grounded conclusion for material need {index}", "citations": [source_id],
    } for index in range(1, 5)]
    return {
        "needs": needs, "findings": findings,
        "answer": "Concise source-grounded answer with a relevant condition.",
        "open_questions": ["A material record remains unresolved."],
    }


def _nvidia_sample():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    query = "temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw"
    result = subprocess.run(
        [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"error": result.stderr.strip() or "nvidia-smi returned no sample"}
    row = [cell.strip() for cell in result.stdout.splitlines()[0].split(",")]
    if len(row) != 5:
        return {"error": "unexpected nvidia-smi sample shape", "raw": result.stdout.strip()}
    return {
        "temperature_c": int(row[0]), "utilization_percent": int(row[1]),
        "memory_used_mib": int(row[2]), "memory_total_mib": int(row[3]),
        "power_w": float(row[4]),
    }


def _gpu_compute_processes():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    result = subprocess.run(
        [executable, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _tokenizer():
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"pinned local tokenizer snapshot is unavailable: {MODEL_PATH}")
    from transformers import AutoTokenizer

    base = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=False)

    class NoThinkingTokenizer:
        def __init__(self, inner):
            self.inner = inner

        def apply_chat_template(self, messages, **kwargs):
            kwargs["enable_thinking"] = False
            return self.inner.apply_chat_template(messages, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    return base, NoThinkingTokenizer(base)


def _build_plan(output, prompts, tokenizer, wrapper):
    task_rows, task_by_id, corpora, neutral, markers, _ = _all_prompts()
    sessions = [
        {"task_id": task["task_id"], "pin": task["pin"], "arm": arm}
        for task in task_rows for arm in ARMS
    ]
    random.Random(ORDER_SEED).shuffle(sessions)
    plan_rows = []
    token_counts = []
    for index, session in enumerate(sessions, 1):
        key = (session["task_id"], session["arm"])
        prompt = prompts[key]
        tokens = tuple(encode_messages(
            wrapper, [{"role": "user", "content": prompt}],
            chat_template=tokenizer.get_chat_template(),
        ))
        if prompt not in wrapper.decode(tokens, skip_special_tokens=False):
            raise ValueError(f"chat template lost exact prompt text for {key}")
        if len(tokens) + OUTPUT_TOKENS > CONTEXT_TOKENS:
            raise ValueError(f"prompt plus output cap exceeds context for {key}: {len(tokens)}")
        plan_rows.append({
            "session_id": f"r{index:02d}", **session,
            "prompt_sha256": sha256(prompt.encode("ascii")),
            "input_tokens": len(tokens),
        })
        token_counts.append(len(tokens))

    source_ids_by_task = {
        task["task_id"]: [row["source_id"] for row in corpora[task["case_id"]][task["pin"]]]
        for task in task_rows
    }
    first_task = task_rows[0]
    source_ids = source_ids_by_task[first_task["task_id"]]
    fixture_text = json.dumps(_fixture(source_ids[0]), ensure_ascii=True, separators=(",", ":"))
    _validate_reply(json.loads(fixture_text), source_ids)
    fixture_ids = tuple(tokenizer.encode(fixture_text, add_special_tokens=False))
    if tokenizer.decode(fixture_ids, skip_special_tokens=True) != fixture_text:
        raise ValueError("legal response fixture did not round-trip through the pinned tokenizer")
    fixture_tokens = len(fixture_ids) + 1
    if fixture_tokens > OUTPUT_TOKENS:
        raise ValueError(f"legal response fixture needs {fixture_tokens} tokens")
    report = {
        "schema": "lsr-fresh-reasoner-preflight-v1",
        "qualification": "prompt, requested JSON format, tokenizer and budget feasibility only; no model weights or output token mask",
        "model_revision": MODEL_REVISION,
        "model_snapshot": str(MODEL_PATH),
        "inputs_sha256": _input_hashes(),
        "tokenizer_sha256": sha256(tokenizer.backend_tokenizer.to_str().encode("utf8")),
        "chat_template_sha256": sha256(tokenizer.get_chat_template().encode("utf8")),
        "limits": {
            "task_pins": len(task_rows), "arms": list(ARMS),
            "sessions": len(plan_rows), "context_tokens": CONTEXT_TOKENS,
            "output_tokens_per_session": OUTPUT_TOKENS,
            "maximum_generated_tokens": len(plan_rows) * OUTPUT_TOKENS,
            "order_seed": ORDER_SEED, "thermal_stop_c": THERMAL_STOP_C,
        },
        "input_token_min": min(token_counts), "input_token_max": max(token_counts),
        "input_tokens_by_task_and_arm": [
            {"session_id": row["session_id"], "task_id": row["task_id"], "pin": row["pin"],
             "arm": row["arm"], "input_tokens": row["input_tokens"]}
            for row in plan_rows
        ],
        "legal_response_fixture_tokens_including_eos": fixture_tokens,
        "json_output_is_sampler_enforced": False,
        "device_sample": _nvidia_sample(),
        "available_gpu_compute_processes": _gpu_compute_processes(),
        "gold_opened": False,
        "model_weights_loaded": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_bytes(_canonical({
        "schema": "lsr-fresh-reasoner-session-plan-v1",
        "input_hashes": report["inputs_sha256"], "sessions": plan_rows,
    }))
    (output / "assignment.json").write_bytes(_canonical({
        "schema": "lsr-fresh-reasoner-assignment-v1",
        "sessions": [{k: row[k] for k in ("session_id", "task_id", "pin", "arm")}
                     for row in plan_rows],
    }))
    (output / "preflight.json").write_bytes(_canonical(report))
    return report


class _GpuLease:
    def __init__(self):
        local_data_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        local_data_dir.mkdir(parents=True, exist_ok=True)
        self.path = local_data_dir / "LSR" / "fresh_reader_gpu.lock"
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        if os.name == "nt":
            import msvcrt
            self.handle.seek(0)
            try:
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("another fresh-reader GPU run holds the local lease") from error
        else:
            import fcntl
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("another fresh-reader GPU run holds the local lease") from error
        return self

    def __exit__(self, *_args):
        if self.handle is not None:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def _safe_session(lease, output, expected_hashes):
    del lease
    preflight_path = output / "preflight.json"
    plan_path = output / "plan.json"
    if not preflight_path.is_file() or not plan_path.is_file():
        raise FileNotFoundError("run requires the exact completed tokenizer preflight")
    preflight = json.loads(preflight_path.read_text(encoding="ascii"))
    plan = json.loads(plan_path.read_text(encoding="ascii"))
    if preflight.get("inputs_sha256") != expected_hashes or plan.get("input_hashes") != expected_hashes:
        raise ValueError("source, task, span, or runner inputs changed after preflight")
    return preflight, plan


def _run(output):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    output = _outside_repo(output)
    if not output.is_dir():
        raise FileNotFoundError("run mode requires an output directory created by preflight")
    task_rows, task_by_id, corpora, neutral, markers, prompts = _all_prompts()
    del task_by_id, markers
    expected_hashes = _input_hashes()
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"pinned local model snapshot is unavailable: {MODEL_PATH}")

    pending = output / "sessions.jsonl.pending"
    completed = output / "sessions.jsonl"
    if completed.exists():
        raise FileExistsError("this preflight already has a completed run")
    with _GpuLease() as lease:
        preflight, plan = _safe_session(lease, output, expected_hashes)
        session_rows = plan.get("sessions")
        if type(session_rows) is not list or len(session_rows) != 18:
            raise ValueError("the session plan must contain exactly 18 randomized task-arm sessions")
        old = []
        if pending.exists():
            for line in pending.read_text(encoding="ascii").splitlines():
                if line.strip():
                    old.append(json.loads(line))
        completed_ids = {row["session_id"] for row in old}
        allowed_ids = {row["session_id"] for row in session_rows}
        if len(completed_ids) != len(old) or not completed_ids <= allowed_ids:
            raise ValueError("existing checkpoint has duplicate or unknown sessions")

        processes = _gpu_compute_processes()
        if processes:
            raise RuntimeError(f"GPU is already occupied by another compute process: {processes}")
        import psutil
        if psutil.virtual_memory().available < 12 * (1 << 30):
            raise MemoryError("fresh-reader run requires at least 12 GiB available system RAM")

        base, wrapper = _tokenizer()
        device = "cpu"
        cuda_preflight = None
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            bf16 = bool(torch.cuda.is_bf16_supported())
            sample = _nvidia_sample()
            safe_temp = sample is None or sample.get("temperature_c", 0) < THERMAL_STOP_C
            if free >= 5 * (1 << 30) and bf16 and safe_temp:
                device = "cuda"
            cuda_preflight = {
                "free_bytes_before_load": int(free), "total_bytes": int(total),
                "device_name": torch.cuda.get_device_name(), "bf16_supported": bf16,
                "nvidia_sample": sample, "minimum_free_bytes": 5 * (1 << 30),
                "thermal_stop_c": THERMAL_STOP_C,
            }
        torch.set_num_threads(THREADS)
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass

        model_start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(device).eval()
        model_load_seconds = time.perf_counter() - model_start
        if device == "cuda":
            free_after_load, _ = torch.cuda.mem_get_info()
            if free_after_load < 1 * (1 << 30):
                raise MemoryError("less than 1 GiB free VRAM remains after model load")

        started = time.perf_counter()
        with pending.open("a", encoding="ascii", newline="\n") as stream:
            for row in session_rows:
                if row["session_id"] in completed_ids:
                    continue
                task = next(value for value in task_rows if value["task_id"] == row["task_id"])
                source_rows = corpora[task["case_id"]][task["pin"]]
                source_ids = tuple(source["source_id"] for source in source_rows)
                prompt = prompts[(row["task_id"], row["arm"])]
                prompt_ids = tuple(encode_messages(
                    wrapper, [{"role": "user", "content": prompt}],
                    chat_template=base.get_chat_template(),
                ))
                if (sha256(prompt.encode("ascii")) != row["prompt_sha256"]
                        or len(prompt_ids) != row["input_tokens"]):
                    raise ValueError("prompt changed since the token preflight")
                eos = int(base.eos_token_id)

                ids_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                mask = torch.ones_like(ids_tensor)
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                call_start = time.perf_counter()
                error = None
                generated_ids = ()
                try:
                    with torch.inference_mode():
                        generated = model.generate(
                            input_ids=ids_tensor, attention_mask=mask,
                            max_new_tokens=OUTPUT_TOKENS, do_sample=False,
                            num_beams=1, use_cache=True, eos_token_id=eos,
                            pad_token_id=eos,
                        )
                    if device == "cuda":
                        torch.cuda.synchronize()
                    generated_ids = tuple(map(int, generated[0, len(prompt_ids):].tolist()))
                except Exception as caught:
                    error = {"type": type(caught).__name__, "message": str(caught)[:500]}
                    if "out of memory" in str(caught).lower():
                        if device == "cuda":
                            torch.cuda.empty_cache()
                        record = {
                            "session_id": row["session_id"], "task_id": row["task_id"],
                            "pin": row["pin"], "arm": row["arm"], "status": "resource_error",
                            "error": error, "prompt_text": prompt,
                            "prompt_sha256": row["prompt_sha256"],
                            "input_tokens": len(prompt_ids), "gold_opened": False,
                        }
                        stream.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        break
                call_seconds = time.perf_counter() - call_start
                raw = base.decode(
                    generated_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                terminated = bool(generated_ids and generated_ids[-1] == eos)
                parsed = None
                validation_error = None
                if terminated:
                    try:
                        parsed = _validate_reply(json.loads(raw), source_ids)
                    except (ValueError, TypeError, json.JSONDecodeError) as caught:
                        validation_error = {"type": type(caught).__name__, "message": str(caught)}
                record = {
                    "session_id": row["session_id"], "task_id": row["task_id"],
                    "pin": row["pin"], "arm": row["arm"],
                    "status": "generated" if error is None else "generation_error",
                    "prompt_text": prompt, "prompt_sha256": row["prompt_sha256"],
                    "prompt_token_ids_sha256": sha256(_canonical(list(prompt_ids))),
                    "input_tokens": len(prompt_ids),
                    "output_token_ids": list(generated_ids), "output_tokens": len(generated_ids),
                    "output_text": raw, "terminated": terminated,
                    "schema_valid": parsed is not None, "validation_error": validation_error,
                    "generation_error": error, "source_ids": list(source_ids),
                    "device": device, "call_seconds": call_seconds,
                    "cuda_peak_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
                    ),
                    "gpu_sample": _nvidia_sample() if device == "cuda" else None,
                    "gold_opened": False,
                }
                stream.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                completed_ids.add(row["session_id"])
                del generated_ids, ids_tensor, mask
                if device == "cuda":
                    torch.cuda.empty_cache()
                    sample = _nvidia_sample()
                    if sample and sample.get("temperature_c", 0) >= THERMAL_STOP_C:
                        break

        run_seconds = time.perf_counter() - started
        records = [json.loads(line) for line in pending.read_text(encoding="ascii").splitlines() if line.strip()]
        if len({row["session_id"] for row in records}) != len(records):
            raise ValueError("checkpoint contains duplicate session IDs")
        if len(records) == len(session_rows):
            pending.replace(completed)
            plan_rows = plan["sessions"]
            assignment = {
                row["session_id"]: {k: row[k] for k in ("task_id", "pin", "arm")}
                for row in plan_rows
            }
            blind = [
                {k: record.get(k) for k in (
                    "session_id", "task_id", "pin", "status", "output_text",
                    "terminated", "schema_valid", "validation_error", "generation_error",
                )}
                for record in records
            ]
            (output / "blind_outputs.jsonl").write_text("".join(
                json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n"
                for row in blind
            ), encoding="ascii")
            (output / "run_manifest.json").write_bytes(_canonical({
                "schema": "lsr-fresh-reasoner-run-manifest-v1",
                "model_revision": MODEL_REVISION, "model_snapshot": str(MODEL_PATH),
                "runner_sha256": _input_hashes()[Path(__file__).name],
                "inputs_sha256": expected_hashes,
                "tokenizer_sha256": preflight["tokenizer_sha256"],
                "chat_template_sha256": preflight["chat_template_sha256"],
                "order_seed": ORDER_SEED, "device": device,
                "cuda_preflight": cuda_preflight,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "threads": THREADS, "context_tokens": CONTEXT_TOKENS,
                "output_tokens_per_session": OUTPUT_TOKENS,
                "maximum_generated_tokens": len(session_rows) * OUTPUT_TOKENS,
                "session_count": len(records), "valid_schema_count": sum(r.get("schema_valid") is True for r in records),
                "terminated_count": sum(r.get("terminated") is True for r in records),
                "decoder": "greedy ordinary full-forward; JSON requested in prompt; no output token mask",
                "model_load_seconds": model_load_seconds, "run_seconds": run_seconds,
                "gold_opened": False, "model_weights_loaded": True,
            }))
            (output / "assignment.json").write_bytes(_canonical({
                "schema": "lsr-fresh-reasoner-assignment-v1", "sessions": assignment,
            }))
        print(json.dumps({
            "output_directory": str(output), "sessions_completed": len(records),
            "sessions_planned": len(session_rows), "device": device,
            "model_load_seconds": model_load_seconds, "run_seconds": run_seconds,
            "thermal_sample": _nvidia_sample(),
        }, ensure_ascii=True, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = _outside_repo(args.output)
    if args.mode == "preflight":
        if output.exists():
            raise FileExistsError(f"refusing to overwrite preflight directory: {output}")
        _, _, _, _, _, prompts = _all_prompts()
        base, wrapper = _tokenizer()
        report = _build_plan(output, prompts, base, wrapper)
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return
    _run(output)


if __name__ == "__main__":
    main()
