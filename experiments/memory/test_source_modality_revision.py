"""A modality correction is a revision even when the source bytes do not change."""
import json

import pytest

from experiments.memory.source_revision_feed import read_source_delta, write_source_delta
from experiments.memory.verified_source_tool import sha256


LIMIT = 4096
ASSET, OTHER = "a" * 32, "b" * 32
DATA = b"P6\n1 1\n255\n\x00\x00d"  # A UTF-8-valid binary image.


@pytest.mark.parametrize("start_binary", [False, True])
def test_same_bytes_modality_revision_is_atomic_and_replayable(tmp_path, start_binary):
    prior = None
    pins = []
    for epoch in range(1, 5):
        binary = start_binary if epoch % 2 else not start_binary
        path = tmp_path / str(epoch)
        changes = {ASSET: DATA} if prior else {ASSET: DATA, OTHER: b"unchanged"}
        delta = write_source_delta(path, changes, previous=prior,
                                   binary_sources=(ASSET,) if binary else (), max_bytes=LIMIT)
        reopened = read_source_delta(path, delta.seal, previous=prior,
                                     allow_binary=True, max_bytes=LIMIT)
        assert delta == reopened
        assert reopened.changes == changes
        assert reopened.manifest[ASSET].sha256 == sha256(DATA)
        assert reopened.manifest[ASSET].generation == epoch
        assert reopened.manifest[OTHER].generation == 1
        assert (ASSET in reopened.binary_sources) == binary
        if prior is not None:
            assert reopened.seal != prior.seal
            assert json.loads(reopened.descriptor)["parent"] == prior.seal
        with pytest.raises(ValueError, match="repeats the current bytes"):
            write_source_delta(tmp_path / "duplicate", {ASSET: DATA}, previous=reopened,
                               binary_sources=(ASSET,) if binary else (), max_bytes=LIMIT)
        pins.append(reopened)
        prior = reopened
    assert len({pin.seal for pin in pins}) == 4
    # Historical type, generation and bytes remain accessible after later corrections.
    prior = None
    for epoch, expected in enumerate(pins, 1):
        prior = read_source_delta(tmp_path / str(epoch), expected.seal, previous=prior,
                                  allow_binary=True, max_bytes=LIMIT)
        assert prior == expected


def test_type_only_revision_requires_binary_authority_and_sealed_declaration(tmp_path):
    first = write_source_delta(tmp_path / "text", {ASSET: DATA})
    with pytest.raises(ValueError, match="requires a byte budget"):
        write_source_delta(tmp_path / "unbounded", {ASSET: DATA}, previous=first,
                           binary_sources=(ASSET,))
    path = tmp_path / "typed"
    delta = write_source_delta(path, {ASSET: DATA}, previous=first,
                               binary_sources=(ASSET,), max_bytes=LIMIT)
    with pytest.raises(ValueError, match="not enabled"):
        read_source_delta(path, delta.seal, previous=first, max_bytes=LIMIT)
    descriptor = json.loads(delta.descriptor)
    descriptor["binary_sources"] = []
    altered = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    (path / "manifest.json").write_bytes(altered)
    # Even with a matching digest for the altered descriptor, its transition
    # disagrees with the canonical typed update and must fail.
    with pytest.raises(ValueError, match="retained scope and parent"):
        read_source_delta(path, sha256(altered), previous=first,
                          allow_binary=True, max_bytes=LIMIT)
