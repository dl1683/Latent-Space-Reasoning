"""Input-only execution for the independent four-case ordinary-reader gate."""

from dataclasses import dataclass
from inspect import signature
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from experiments.memory.iterative_text_competence import (
    MAX_INPUT_TOKENS, MAX_NEW_TOKENS, read_iterative_text_competence, run_iterative_text_competence,
)
from experiments.memory.source_revision_feed import write_source_delta
from experiments.memory.task_episode import PolicyCall, PolicyReply, _Ledger
from experiments.memory.task_evidence import TaskBudget, TaskWorld
from experiments.memory.verified_source_tool import sha256


_CASE_COUNT = 4
_SOURCE_LIMITS = dict(max_feed_bytes=65_536, max_source_bytes=65_536, max_passages=16, passage_bytes=512)
_READ_LIMITS = TaskBudget(steps=8, candidates_per_search=1, evidence_bytes=2048, source_bytes=2048)
_FREE_RAM_FLOOR = 12 * 2**30
_RESOURCE_ERRORS = (TimeoutError, MemoryError, RuntimeError)


@dataclass(frozen=True, slots=True)
class OrdinaryCalibrationCase:
    case_id: str
    question: str
    source_deltas: tuple[dict[str, str | None], ...]
    read_epoch: int  # One-based position in the normalized revision list.


class CpuOrdinaryPolicy:
    """Run a loaded CPU causal model with a fresh greedy decode for each call."""

    def __init__(self, model, tokenizer, *, deadline):
        from math import isfinite

        if not isfinite(deadline):
            raise ValueError("ordinary calibration requires a finite monotonic deadline")
        if (any(module.training for module in model.modules())
                or any(tensor.device.type != "cpu" for tensor in (*model.parameters(), *model.buffers()))):
            raise ValueError("ordinary calibration requires evaluation mode on CPU")
        self._model = model
        self.tokenizer = tokenizer
        self.deadline = deadline
        self.resource_stop = None
        self._legacy_generate_options = (
            {"use_model_defaults": False} if "use_model_defaults" in signature(model.generate).parameters else {}
        )
        self._eos = tokenizer.eos_token_id
        self._pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self._eos
        if (type(self._eos) is not int or self._eos < 0 or type(self._pad) is not int or self._pad < 0
                or self._eos != model.generation_config.eos_token_id
                or self._pad != model.generation_config.pad_token_id):
            raise ValueError("ordinary calibration requires matching single EOS and padding IDs")

    def runtime_record(self):
        import psutil
        import torch
        import transformers

        return {
            "torch": str(torch.__version__), "transformers": transformers.__version__,
            "model_class": type(self._model).__qualname__,
            "parameter_dtypes": sorted({str(tensor.dtype) for tensor in self._model.parameters()}),
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "eos_id": self._eos, "pad_id": self._pad,
            "free_ram_bytes": psutil.virtual_memory().available, "min_free_ram_bytes": _FREE_RAM_FLOOR,
            "remaining_seconds": max(0.0, self.deadline - monotonic()),
        }

    def check_resources(self):
        import psutil
        import torch

        if self.resource_stop is None:
            if self.deadline <= monotonic():
                self.resource_stop = (TimeoutError, "ordinary calibration reached its wall-clock allowance")
            elif torch.get_num_threads() > 4 or torch.get_num_interop_threads() > 4:
                self.resource_stop = (RuntimeError, "ordinary calibration exceeded its CPU thread profile")
            elif psutil.virtual_memory().available < _FREE_RAM_FLOOR:
                self.resource_stop = (MemoryError, "ordinary calibration crossed its free-RAM floor")
        if self.resource_stop is not None:
            error, message = self.resource_stop
            raise error(message)

    def __call__(self, call):
        import torch
        from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

        if (type(call) is not PolicyCall or not 0 < len(call.input_ids) <= MAX_INPUT_TOKENS
                or any(type(token) is not int or token < 0 for token in call.input_ids)
                or type(call.max_new_tokens) is not int or not 1 <= call.max_new_tokens <= MAX_NEW_TOKENS):
            raise ValueError("ordinary calibration generation request exceeds its contract")
        self.check_resources()
        policy = self

        class ResourceStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                try:
                    policy.check_resources()
                except _RESOURCE_ERRORS:
                    stopped = True
                else:
                    stopped = False
                return input_ids.new_full((input_ids.shape[0],), stopped, dtype=torch.bool)

        inputs = torch.tensor([call.input_ids], dtype=torch.long, device="cpu")
        config = GenerationConfig(max_new_tokens=call.max_new_tokens, do_sample=False, num_beams=1,
                                  use_cache=True, eos_token_id=self._eos, pad_token_id=self._pad)
        try:
            with torch.inference_mode():
                output = self._model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                              generation_config=config, stopping_criteria=StoppingCriteriaList([ResourceStop()]),
                                              **self._legacy_generate_options)
        except Exception as error:
            if self.resource_stop is None:
                kind = type(error) if type(error) in _RESOURCE_ERRORS else RuntimeError
                self.resource_stop = (kind, "ordinary calibration generation failed")
            raise
        prefix = len(call.input_ids)
        if (not isinstance(output, torch.Tensor) or output.device.type != "cpu" or output.dtype != torch.long
                or output.ndim != 2 or output.shape[0] != 1
                or not prefix <= output.shape[1] <= prefix + call.max_new_tokens
                or tuple(output[0, :prefix].tolist()) != call.input_ids):
            raise ValueError("ordinary generation changed its input or output envelope")
        ids = output[0, prefix:].tolist()
        terminated = bool(ids and ids[-1] == self._eos and self.resource_stop is None)
        return PolicyReply(self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                           len(ids), terminated)


def _prepare_case(case, directory):
    if type(case.read_epoch) is not int or not 1 <= case.read_epoch <= len(case.source_deltas):
        raise ValueError("ordinary calibration read epoch is outside its source history")
    directory.mkdir()
    world = delta = None
    setup_bytes = feed_bytes = 0
    for index, changes in enumerate(case.source_deltas[:case.read_epoch], start=1):
        if type(changes) is not dict or any(value is not None and type(value) is not str for value in changes.values()):
            raise ValueError("ordinary calibration needs text source deltas and explicit withdrawals")
        encoded = {source: None if value is None else value.encode("utf-8") for source, value in changes.items()}
        epoch_dir = directory / str(index)
        delta = write_source_delta(epoch_dir, encoded, previous=delta, max_bytes=_SOURCE_LIMITS["max_feed_bytes"])
        feed_bytes += sum((epoch_dir / filename).stat().st_size for filename in ("manifest.json", "updates.json"))
        if feed_bytes > _SOURCE_LIMITS["max_feed_bytes"]:
            raise ValueError("ordinary calibration source history exceeds its feed budget")
        world = TaskWorld.from_source_epoch(epoch_dir, delta.seal, previous=world, **_SOURCE_LIMITS)
        setup_bytes += world.setup_source_bytes
    return world, setup_bytes, feed_bytes


def run_ordinary_cpu_calibration(packet, *, packet_sha256, decode_cases, backend, output, backend_id, tokenizer_id):
    """Execute a confirmed input-only packet using its custodian's normalization."""
    if type(packet) is not bytes or sha256(packet) != packet_sha256:
        raise ValueError("ordinary calibration input differs from the frozen custody digest")
    if (type(backend) is not CpuOrdinaryPolicy
            or any(type(item) is not str or not item.strip() for item in (backend_id, tokenizer_id))):
        raise ValueError("ordinary calibration requires its CPU policy and pinned identities")
    cases = tuple(islice(decode_cases(packet), _CASE_COUNT + 1))
    if (len(cases) != _CASE_COUNT or any(type(case) is not OrdinaryCalibrationCase
            or any(type(item) is not str or not item.strip() for item in (case.case_id, case.question))
            for case in cases) or len({case.case_id for case in cases}) != _CASE_COUNT):
        raise ValueError("ordinary calibration needs four distinct case IDs and nonempty questions")
    path = Path(output)
    with TemporaryDirectory(prefix="lsr-ordinary-feeds-") as scratch:
        worlds = tuple(_prepare_case(case, Path(scratch) / str(index)) for index, case in enumerate(cases, start=1))
    stop_reason = None
    try:
        runtime = backend.runtime_record()
    except Exception as error:
        runtime = None
        stop_reason = type(error).__name__
    path.mkdir(mode=0o700)
    with (path / "batch.jsonl").open("xb") as stream:
        ledger = _Ledger(stream)
        ledger.write("start", {"input_sha256": packet_sha256, "cases": [case.case_id for case in cases],
                               "backend_id": backend_id, "tokenizer_id": tokenizer_id, "runtime": runtime,
                               "runner_sha256": sha256(Path(__file__).read_bytes())})
        for index, (case, (world, setup, feed)) in enumerate(zip(cases, worlds, strict=True), start=1):
            details = {"case": case.case_id, "revision_setup_source_bytes": setup,
                       "revision_feed_bytes": feed, "source_seal": world.source_seal}
            if stop_reason is None:
                try:
                    backend.check_resources()
                except Exception as error:
                    stop_reason = type(error).__name__
            if stop_reason is not None:
                ledger.write("not_run", {**details, "reason": stop_reason})
                continue
            transcript = f"case-{index}.jsonl"
            try:
                artifact = run_iterative_text_competence(
                    case.question, world, backend.tokenizer, backend, path / transcript,
                    backend_id=backend_id, tokenizer_id=tokenizer_id, budget=_READ_LIMITS,
                )
                rows = read_iterative_text_competence(artifact)
            except Exception as error:
                stop_reason = type(error).__name__
                ledger.write("case_error", {**details, "path": transcript, "reason": stop_reason})
                continue
            if backend.resource_stop is not None:
                stop_reason = backend.resource_stop[0].__name__
            elif artifact.status in ("backend_error", "tokenizer_error"):
                stop_reason = artifact.status
            ledger.write("case", {**details, "path": transcript, "sha256": artifact.sha256,
                                  "status": artifact.status, "calls": rows[-1]["data"]["calls"],
                                  "stop_reason": stop_reason})
        ledger.write("finish", {"scheduled_cases": _CASE_COUNT, "stop_reason": stop_reason})
        return ledger.head
