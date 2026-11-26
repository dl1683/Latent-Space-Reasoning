"""Linux sealed-source custody control; no model or repository runtime required."""
import errno
import fcntl
import hashlib
import os
import tempfile
from pathlib import Path

SEALS = (fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK
         | fcntl.F_SEAL_SEAL)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def partition(raw, cuts):
    return tuple((a, b, digest(raw[a:b])) for a, b in zip(cuts, cuts[1:]))


class Custody:
    """Trusted process owns immutable descriptors and source-local proofs."""
    def __init__(self):
        self.files, self.proofs, self.bytes_checked = {}, set(), 0

    def add(self, raw):
        version = digest(raw)
        if version not in self.files:
            fd = os.memfd_create('source', os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
            with os.fdopen(os.dup(fd), 'wb') as writer:
                writer.write(raw)
            fcntl.fcntl(fd, fcntl.F_ADD_SEALS, SEALS)
            self.files[version] = fd
        return version

    def verify(self, version, spans):
        fd = self.files[version]
        if fcntl.fcntl(fd, fcntl.F_GET_SEALS) & SEALS != SEALS:
            raise ValueError('unsealed source')
        key = (version, spans)
        if key in self.proofs:
            return True
        size = os.fstat(fd).st_size
        raw = os.pread(fd, size, 0)
        self.bytes_checked += len(raw)
        if len(raw) != size or digest(raw) != version:
            return False
        end = 0
        for a, b, h in spans:
            if a != end or not a < b <= size or digest(raw[a:b]) != h:
                return False
            end = b
        if end != size:
            return False
        self.proofs.add(key)
        return True

    def close(self):
        for fd in self.files.values():
            os.close(fd)
        self.files.clear()
        self.proofs.clear()


def main():
    # Equal-size, same inode, restored mtime: not evidence of immutability.
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / 'source'
        path.write_bytes(b'old!')
        old = path.stat()
        path.write_bytes(b'new!')
        os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
        new = path.stat()
        assert (old.st_ino, old.st_size, old.st_mtime_ns) == (new.st_ino, new.st_size, new.st_mtime_ns)
        assert digest(path.read_bytes()) != digest(b'old!')
    c = Custody()
    try:
        a, b, revised = b'a' * 4096, b'b' * 8192, b'c' * 4096
        va, vb, vr = [c.add(raw) for raw in (a, b, revised)]
        pa, pb, pr = [partition(raw, (0, len(raw)//2, len(raw))) for raw in (a,b,revised)]
        assert c.verify(va, pa) and c.verify(vb, pb)
        opening = c.bytes_checked
        for _ in range(100):
            assert c.verify(va, pa) and c.verify(vb, pb)
        warm = c.bytes_checked - opening
        assert warm == 0
        # Revisions and historical reads: source-local content proofs only.
        assert c.verify(vr, pr) and c.verify(vb, pb)
        update = c.bytes_checked - opening - warm
        assert update == len(revised)
        assert c.verify(va, pa) and c.verify(vb, pb)
        historical = c.bytes_checked - opening - warm - update
        assert historical == 0
        before = c.bytes_checked
        assert not c.verify(vr, pa)  # same size, wrong bytes
        assert not c.verify(va, ((0, 4096, digest(b'z'*4096)),))
        assert not c.verify(va, ((0, 2048, digest(a[:2048])),))  # missing suffix
        assert c.verify(va, partition(a, (0, 4096)))  # different valid partition
        assert c.bytes_checked - before == 4 * 4096
        fd = c.files[va]
        dup = os.dup(fd)
        try:
            for handle in (fd, dup):
                for action in (lambda: os.pwrite(handle, b'Z', 0),
                               lambda: os.ftruncate(handle, 0),
                               lambda: os.ftruncate(handle, 8192)):
                    try:
                        action()
                    except OSError as e:
                        assert e.errno == errno.EPERM
                    else:
                        raise AssertionError('sealed write or resize succeeded')
            assert os.pread(fd, len(a), 0) == a
        finally:
            os.close(dup)
        print(dict(opening=opening, warm_100=warm, update=update,
                   historical=historical, rejected_invalid_partitions=3,
                   checked_new_partition=4096, blocked_mutations=6))
    finally:
        c.close()


if __name__ == '__main__':
    main()
