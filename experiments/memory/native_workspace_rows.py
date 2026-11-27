"""Store exact canonical native numeric groups once per workspace."""

from hashlib import sha256
from os import fsync
from struct import Struct

from experiments.memory.native_publication import _read_bounded


REFERENCES = "rows.refs"
_HEADER = b"lsr-native-rows-v1\n"
_REFERENCE = Struct("<32sQ")


class SharedNativeRows:
    def __init__(self, blocks, max_bytes, *, draft=None):
        self._blocks = blocks
        self._max_bytes = max_bytes
        self._draft = draft

    @staticmethod
    def _write_bytes(handle, raw):
        handle.write(raw)
        handle.flush()
        fsync(handle.fileno())

    @staticmethod
    def _verified_block(path, digest, size):
        raw = _read_bounded(path, size)
        if len(raw) != size or sha256(raw).hexdigest() != digest:
            raise ValueError("native shared numeric block differs")
        return raw

    def write_rows(self, directory, groups, size):
        total = 0
        payload = sha256()
        with (directory / REFERENCES).open("xb") as index:
            index.write(_HEADER)
            for group in groups:
                raw = b"".join(memoryview(array).cast("B") for array in group)
                if not raw:
                    continue
                total += len(raw)
                if total > size or index.tell() + _REFERENCE.size > self._max_bytes:
                    raise ValueError("native shared numeric rows exceed their byte allowance")
                digest = sha256(raw).digest()
                name = digest.hex()
                staged = self._draft / name
                if not (self._blocks / name).exists() and not staged.exists():
                    with staged.open("xb") as handle:
                        self._write_bytes(handle, raw)
                index.write(_REFERENCE.pack(digest, len(raw)))
                payload.update(raw)
            if total != size:
                raise ValueError("native publication numeric sizes differ")
            index.flush()
            fsync(index.fileno())
        return payload.hexdigest()

    def read_rows(self, directory, size):
        raw = _read_bounded(directory / REFERENCES, self._max_bytes)
        if not raw.startswith(_HEADER) or (len(raw) - len(_HEADER)) % _REFERENCE.size:
            raise ValueError("native shared numeric references are malformed")
        references = tuple(_REFERENCE.iter_unpack(memoryview(raw)[len(_HEADER):]))
        if any(length == 0 for _, length in references) or sum(length for _, length in references) != size:
            raise ValueError("native shared numeric reference sizes differ")
        blocks = []
        for digest, length in references:
            name = digest.hex()
            path = self._blocks / name
            if self._draft is not None and (self._draft / name).exists():
                path = self._draft / name
            blocks.append(self._verified_block(path, name, length))
        return b"".join(blocks)

    def install(self):
        # The workspace's writer lock is held. Only previously absent staged
        # blocks can reach this loop.
        for staged in self._draft.iterdir():
            destination = self._blocks / staged.name
            if destination.exists():
                self._verified_block(destination, staged.name, staged.stat().st_size)
            else:
                staged.rename(destination)
