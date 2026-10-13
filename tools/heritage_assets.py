"""List or recover explicitly selected files from local, verified heritage archives.

This is custody tooling, not a research result. It never launches archived code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('matrix-native-math', 'llm-genome', 'new-computation-model')
DEFAULT_ARCHIVES = Path.home() / 'lsr-research-archives' / '2026-10-13'


def safe_relative(name: str) -> Path:
    p = PurePosixPath(name)
    if (not name or p.is_absolute() or any(x in ('..', '.') for x in p.parts)
            or p.as_posix() != name or '\\' in name or ':' in name):
        raise ValueError(f'Unsafe archive path: {name!r}')
    for part in p.parts:
        if part.endswith((' ', '.')) or part.split('.')[0].upper() in {
            'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]
        }:
            raise ValueError(f'Windows alias/reserved path: {name!r}')
    return Path(*p.parts)


def load_receipt(source: str, archives: Path) -> tuple[dict, Path]:
    receipt = json.loads((archives / f'{source}.receipt.json').read_text(encoding='utf-8'))
    if receipt.get('state') != 'verified':
        raise ValueError('The archive has no completed verification receipt.')
    archive = archives / f'{source}.zip'
    if archive.stat().st_size != receipt['archive_bytes']:
        raise ValueError('Archive size differs from its verification receipt.')
    records = receipt['files']
    if len(records) != len({r['path'] for r in records}):
        raise ValueError('Duplicate paths in receipt.')
    for r in records:
        safe_relative(r['path'])
    return receipt, archive


def choose(records: list[dict], paths: list[str], prefix: str | None) -> list[dict]:
    by_path = {r['path']: r for r in records}
    for path in paths:
        safe_relative(path)
        if path not in by_path:
            raise ValueError(f'Not present in archive: {path}')
    selected = set(paths)
    if prefix is not None:
        clean = prefix.rstrip('/')
        safe_relative(clean)
        selected.update(p for p in by_path if p == clean or p.startswith(clean + '/'))
    return [by_path[p] for p in sorted(selected)]


def restore(archive: Path, records: list[dict], destination: Path) -> int:
    """Verify each member before installing; refuse overwrites and symlink escapes."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    targets = []
    aliases = set()
    for r in records:
        target = root / safe_relative(r['path'])
        if not target.resolve().is_relative_to(root):
            raise ValueError('Destination escapes asset root through a link.')
        if target.exists():
            raise FileExistsError(f'Already restored: {target}')
        alias = str(target).casefold()
        if alias in aliases:
            raise ValueError('Multiple selected records map to one Windows target.')
        aliases.add(alias)
        targets.append((r, target))
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members.')
        for r, target in targets:
            info = z.getinfo(r['path'])
            if info.file_size != r['bytes'] or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Invalid ZIP member: {r["path"]}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.parent.resolve().is_relative_to(root):
                raise ValueError('Destination parent escapes asset root.')
            h = hashlib.sha256()
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.restore-', delete=False) as sink:
                    temp_path = Path(sink.name)
                    with z.open(info) as stream:
                        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                            sink.write(chunk); h.update(chunk)
                if h.hexdigest() != r['sha256']:
                    raise ValueError(f'SHA256 mismatch: {r["path"]}')
                # A hard link installs atomically and refuses a concurrently created target.
                target.hardlink_to(temp_path)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
    return sum(r['bytes'] for r in records)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('list', 'restore'))
    p.add_argument('source', choices=SOURCES)
    p.add_argument('paths', nargs='*', help='Exact archived paths; restore requires paths or --prefix.')
    p.add_argument('--prefix', help='Select a directory prefix without restoring the full archive.')
    p.add_argument('--archives', type=Path, default=DEFAULT_ARCHIVES,
                   help='Directory containing ZIPs and verified receipts (never uploaded to Git).')
    p.add_argument('--limit', type=int, default=50, help='Listing limit; 0 lists every selected record.')
    a = p.parse_args()
    if a.limit < 0:
        p.error('--limit must be nonnegative')
    receipt, archive = load_receipt(a.source, a.archives)
    records = choose(receipt['files'], a.paths, a.prefix)
    if a.action == 'list':
        if not a.paths and a.prefix is None:
            records = receipt['files']
        print(f'{len(records)} files / {sum(r["bytes"] for r in records):,} bytes')
        for r in records[:a.limit] if a.limit else records:
            print(f'{r["bytes"]:>14,}  {r["path"]}')
    else:
        if not a.paths and a.prefix is None:
            p.error('Restore requires exact paths or --prefix; full restoration is never implicit.')
        if not records:
            p.error('No files matched.')
        destination = ROOT / 'assets' / 'heritage' / a.source
        size = restore(archive, records, destination)
        print(f'Restored {len(records)} files / {size:,} bytes into {destination}')


if __name__ == '__main__':
    main()
