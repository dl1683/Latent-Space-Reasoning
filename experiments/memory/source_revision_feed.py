"""A replayable source-only boundary for workers on versioned corpora."""

from base64 import b64decode, b64encode
from binascii import Error as Base64Error
from dataclasses import dataclass
import json
from pathlib import Path
import re
from types import MappingProxyType

from experiments.memory.verified_source_tool import sha256


_SCHEMA = "lsr-source-delta-v1"
_BINARY_SCHEMA = "lsr-source-delta-v2"
_OPAQUE_ID = re.compile(r"[a-f0-9]{32}")


def _encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf8")


@dataclass(frozen=True, slots=True)
class SourceVersion:
    sha256: str
    generation: int


@dataclass(frozen=True, slots=True)
class SourceDelta:
    seal: str
    epoch: int
    manifest: MappingProxyType
    changes: MappingProxyType
    descriptor: bytes | None = None
    schema: str = _SCHEMA
    binary_sources: frozenset[str] = frozenset()


def _advance(previous, changes):
    epoch = 1 if previous is None else previous.epoch + 1
    manifest = {} if previous is None else dict(previous.manifest)
    for source, raw in changes.items():
        if type(source) is not str or not _OPAQUE_ID.fullmatch(source):
            raise ValueError("source feed needs an opaque document identity")
        if raw is None:
            if source not in manifest:
                raise ValueError("source feed cannot withdraw an absent document")
            del manifest[source]
        else:
            if type(raw) is not bytes:
                raise TypeError("source feed needs committed bytes")
            digest = sha256(raw)
            current = manifest.get(source)
            if current is not None and current.sha256 == digest:
                raise ValueError("source update repeats the current bytes")
            manifest[source] = SourceVersion(digest, epoch)
    return epoch, manifest


def _wire_value(raw, *, binary):
    if raw is None:
        return None
    if binary:
        return {"base64": b64encode(raw).decode("ascii")}
    return raw.decode("utf8")


def _raw_value(value):
    if value is None:
        return None
    if type(value) is str:
        return value.encode("utf8")
    if type(value) is dict and set(value) == {"base64"} and type(value["base64"]) is str:
        try:
            return b64decode(value["base64"], validate=True)
        except (Base64Error, ValueError) as error:
            raise ValueError("invalid binary source encoding") from error
    raise ValueError("source updates need text, binary data or withdrawals")


def _prepare(previous, changes, *, binary_sources):
    epoch, manifest = _advance(previous, changes)
    binary = frozenset(binary_sources)
    if any(source not in changes or not changes[source] for source in binary):
        raise ValueError("binary sources must name nonempty updates")
    prior = frozenset() if previous is None else previous.binary_sources
    current = prior.difference(changes) | binary
    schema = _BINARY_SCHEMA if binary or (previous and previous.schema == _BINARY_SCHEMA) else _SCHEMA
    updates = _encode({source: _wire_value(raw, binary=source in binary) for source, raw in changes.items()})
    record = dict(schema=schema, epoch=epoch, parent=None if previous is None else previous.seal,
                  changes_sha256=sha256(updates),
                  documents={key: [value.sha256, value.generation] for key, value in manifest.items()})
    if schema == _BINARY_SCHEMA:
        record["binary_sources"] = sorted(current)
    descriptor = _encode(record)
    return SourceDelta(sha256(descriptor), epoch, MappingProxyType(manifest),
                       MappingProxyType(changes), descriptor, schema, current), updates


def _validate_limit(max_bytes, schema):
    if max_bytes is None:
        if schema == _BINARY_SCHEMA:
            raise ValueError("binary source feed requires a byte budget")
    elif type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("source feed byte budget must be a positive integer")


def _read_file(path, limit):
    if limit is None:
        return path.read_bytes()
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError("source feed exceeds the byte budget")
    return data


def write_source_delta(directory, changes, *, previous=None, binary_sources=(), max_bytes=None):
    """Publish one source revision, optionally including declared binary assets."""
    binary_sources = frozenset(binary_sources)
    _validate_limit(max_bytes, _BINARY_SCHEMA if binary_sources else getattr(previous, "schema", None))
    delta, updates = _prepare(previous, dict(changes), binary_sources=binary_sources)
    if max_bytes is not None and len(delta.descriptor) + len(updates) > max_bytes:
        raise ValueError("source feed exceeds the byte budget")
    root = Path(directory)
    root.mkdir(mode=0o700)
    (root / "updates.json").write_bytes(updates)
    pending = root / "manifest.pending"
    pending.write_bytes(delta.descriptor)
    pending.rename(root / "manifest.json")
    return delta


def read_source_delta(directory, seal, *, previous=None, max_bytes=None, allow_binary=False):
    """Verify externally sealed source bytes and the ordered manifest transition."""
    _validate_limit(max_bytes, _BINARY_SCHEMA if allow_binary else getattr(previous, "schema", None))
    root = Path(directory)
    descriptor = _read_file(root / "manifest.json", max_bytes)
    if sha256(descriptor) != seal:
        raise ValueError("source manifest seal differs")
    receipt = json.loads(descriptor)
    if type(receipt) is not dict:
        raise ValueError("source manifest needs an object")
    schema = receipt.get("schema")
    if schema == _BINARY_SCHEMA and allow_binary is not True:
        raise ValueError("binary source feed is not enabled")
    _validate_limit(max_bytes, schema)
    remaining = None if max_bytes is None else max_bytes - len(descriptor)
    updates = _read_file(root / "updates.json", remaining)
    if sha256(updates) != receipt.get("changes_sha256"):
        raise ValueError("source update bytes differ from the manifest")
    decoded = json.loads(updates)
    if type(decoded) is not dict:
        raise ValueError("source updates need an object")
    binary = tuple(source for source, value in decoded.items() if type(value) is dict)
    if binary and schema != _BINARY_SCHEMA:
        raise ValueError("binary updates require a binary source descriptor")
    changes = {key: _raw_value(value) for key, value in decoded.items()}
    expected, canonical = _prepare(previous, changes, binary_sources=binary)
    if expected.descriptor != descriptor or canonical != updates:
        raise ValueError("source projection does not match the retained scope and parent")
    return expected
