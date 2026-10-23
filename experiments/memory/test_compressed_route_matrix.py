"""Tiny deterministic software checks; no actual model, source corpus or service.

The Main integration check is skipped outside its private checkout; a standalone
fresh real CLI tester must independently validate this builder's work.
"""
from dataclasses import dataclass, FrozenInstanceError
import gc
import hashlib
import importlib.util
from pathlib import Path
import sys
import tracemalloc
import unittest
import weakref

import numpy as np

from experiments.memory.compressed_route_matrix import CompressedRouteMatrix
from experiments.memory.compressed_routing_map import normalize, pack_rows, unpack_rows
from experiments.memory.corpus_size_ladder import canonical_order
from experiments.memory.packed_map_scan import Packed8Scan


MAIN = (Path(__file__).resolve().parents[2] / "workspace"
        / "optional_route_fixture" / "component.py")


def decoded_reference64(codes, scales, query):
    # Format levels round in FP32; decode ORIGINAL scales in FP64 so neither
    # their products nor squared norms overflow across the finite FP32 range.
    levels = (2 * codes.astype(np.float32) / 255 - 1).astype(np.float64)
    decoded = levels * scales.astype(np.float64)[:, None]
    denominator = np.maximum(np.linalg.norm(decoded, axis=1, keepdims=True), 1e-12)
    return (decoded / denominator) @ query.astype(np.float64)


class CompressedRouteMatrixTests(unittest.TestCase):
    def test_large_finite_float32_rows(self):
        maximum = np.finfo(np.float32).max
        below_maximum = np.nextafter(maximum, np.float32(0))
        for dim in (2, 65):
            x = np.empty((6, dim), dtype=np.float32)
            x[0] = np.float32(1e20)  # Original R3-F1, dim2 axis score=1/sqrt2.
            x[1] = maximum
            x[2] = below_maximum
            x[3] = maximum
            x[3, 1::2] = -maximum
            x[4] = np.linspace(-1, 1, dim, dtype=np.float32) * maximum
            x[5] = 0
            x[5, 0] = -maximum
            self.assertTrue(np.isfinite(x).all())
            codes, scales = pack_rows(x, 8)
            axis = np.zeros(dim, np.float32)
            axis[0] = 1
            queries = (axis, normalize(np.ones(dim, np.float32)),
                       normalize(np.linspace(-1, 1, dim, dtype=np.float32)))
            for block_rows in (1, 4):
                with self.subTest(dim=dim, block_rows=block_rows), np.errstate(over="raise", invalid="raise"):
                    matrix = CompressedRouteMatrix.from_float32(x, block_rows=block_rows)
                    np.testing.assert_array_equal(matrix.codes, codes)
                    for query in queries:
                        actual = matrix @ query
                        self.assertTrue(np.isfinite(actual).all())
                        np.testing.assert_allclose(actual, decoded_reference64(codes, scales, query),
                                                   atol=2e-6, rtol=0)
                    if dim == 2:
                        self.assertAlmostEqual(float((matrix @ axis)[0]), 2**-0.5, delta=2e-6)
                    self.assertEqual(matrix.nbytes, codes.nbytes + 8 * len(codes))

    def test_large_direct_scales_against_fp64_reference(self):
        rng = np.random.default_rng(1846)
        codes = rng.integers(0, 256, (7, 65), dtype=np.uint8)
        codes[0] = 128  # Smallest positive format level; floor still inactive.
        codes[1] = 127
        scales = np.array([np.nextafter(np.float32(1), np.float32(2)), 2, 1e10,
                           1e20, np.finfo(np.float32).max, 1e-30, 1e-12], np.float32)
        queries = (normalize(rng.normal(size=65).astype(np.float32)),
                   normalize(np.ones(65, np.float32)))
        for block_rows in (1, 3, 64):
            with self.subTest(block_rows=block_rows), np.errstate(over="raise", invalid="raise"):
                matrix = CompressedRouteMatrix(codes, scales, block_rows=block_rows)
                for query in queries:
                    np.testing.assert_allclose(matrix @ query,
                                               decoded_reference64(codes, scales, query),
                                               atol=2e-6, rtol=0)

    def test_small_scales_preserve_scanner_floor_exactly(self):
        scales = np.array([np.nextafter(np.float32(0), np.float32(1)), 1e-30,
                           1e-12, 1e-6, np.nextafter(np.float32(1), np.float32(0)),
                           1], dtype=np.float32)
        codes = np.array([[128, 128, 128], [127, 128, 127], [255, 0, 128],
                          [0, 255, 127], [128, 128, 128], [255, 127, 0]], np.uint8)
        matrix = CompressedRouteMatrix(codes, scales, block_rows=2)
        unchanged = Packed8Scan.build(codes, scales, block_rows=2)
        np.testing.assert_array_equal(matrix.multipliers, unchanged.multipliers)
        query = np.array([1, 0, 0], np.float32)
        np.testing.assert_array_equal(matrix @ query, unchanged.scores(query))

    def test_invalid_original_scales_cannot_be_clipped_valid(self):
        codes = np.full((2, 3), 128, dtype=np.uint8)
        for invalid in (np.nan, np.inf, -np.inf, 0, -1):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                CompressedRouteMatrix(codes, np.array([1e20, invalid], np.float32))
        for dtype in (np.float64, np.int64, np.complex64, object):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                CompressedRouteMatrix(codes, np.array([2, 3], dtype=dtype))

    def test_decoded_reference_and_block_construction(self):
        rng = np.random.default_rng(1845)
        for shape in ((31, 17), (73, 65)):
            x = normalize(rng.normal(size=shape).astype(np.float32))
            x[0] = 0
            x[1] *= np.float32(1e-12)
            x[2] = x[3]
            c, s = pack_rows(x, 8)
            decoded = normalize(unpack_rows(c, s, shape[1], 8))
            for rows in (1, 7, 64):
                matrix = CompressedRouteMatrix.from_float32(x, block_rows=rows)
                np.testing.assert_array_equal(matrix.codes, c)
                for _ in range(3):
                    q = normalize(rng.normal(size=shape[1]).astype(np.float32))
                    np.testing.assert_allclose(matrix @ q, decoded @ q, atol=2e-6, rtol=0)
                    self.assertEqual((matrix @ q)[2], (matrix @ q)[3])
                self.assertEqual(matrix.shape, shape)
                self.assertEqual(matrix.nbytes, c.nbytes + 8 * len(c))

    def test_zero_and_near_zero_are_existing_affine_format(self):
        x = np.array([[0, 0, 0], [1e-30, -1e-30, 0]], dtype=np.float32)
        c, s = pack_rows(x, 8)
        decoded = normalize(unpack_rows(c, s, 3, 8))
        matrix = CompressedRouteMatrix(c, s)
        q = np.array([1, 0, 0], np.float32)
        np.testing.assert_allclose(matrix @ q, decoded @ q, atol=2e-6, rtol=0)
        self.assertGreater((matrix @ q)[0], 0)  # Known affine zero bias, not exact zero.
        self.assertTrue(np.isfinite(matrix @ q).all())

    def test_owned_immutable_buffers_and_float_release(self):
        x = np.eye(5, dtype=np.float32)
        reference = weakref.ref(x)
        matrix = CompressedRouteMatrix.from_float32(x)
        del x
        gc.collect()
        self.assertIsNone(reference())
        c, s = pack_rows(np.eye(5, dtype=np.float32), 8)
        matrix = CompressedRouteMatrix(c, s)
        q = np.ones(5, np.float32)
        before = matrix @ q
        c[:] = 0
        s[:] = 99
        np.testing.assert_array_equal(matrix @ q, before)
        for buffer in (matrix.codes, matrix.multipliers):
            with self.assertRaises(ValueError):
                buffer.setflags(write=True)
            with self.assertRaises(ValueError):
                buffer.flat[0] = 1
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            matrix._scan = None

    def test_invalid_float_inputs_and_blocks(self):
        for x in (np.empty((0, 3), np.float32), np.empty((3, 0), np.float32),
                  np.ones(3, np.float32), np.ones((2, 3), np.float64),
                  np.ones((2, 3), np.int8), np.ones((2, 3), np.complex64)):
            with self.subTest(shape=x.shape, dtype=x.dtype), self.assertRaises(ValueError):
                CompressedRouteMatrix.from_float32(x)
        for value in (np.nan, np.inf, -np.inf):
            x = np.ones((67, 3), np.float32)
            x[-1, 0] = value
            with self.assertRaises(ValueError):
                CompressedRouteMatrix.from_float32(x, block_rows=7)
        for block in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                CompressedRouteMatrix.from_float32(np.ones((2, 3), np.float32), block_rows=block)

    def test_invalid_packed_inputs_queries_and_signed_format(self):
        c, s = pack_rows(np.eye(3, dtype=np.float32), 8)
        for codes, scales in ((c.astype(np.int8), s), (c.astype(np.int16), s),
                              (c, s.astype(np.float64)), (c, s[:-1]),
                              (c, -s), (c, np.zeros_like(s)),
                              (c, np.full_like(s, np.inf)), (c, np.full_like(s, np.nan)),
                              (c.reshape(-1), s), (c[:0], s[:0])):
            with self.assertRaises(ValueError):
                CompressedRouteMatrix(codes, scales)
        for block in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                CompressedRouteMatrix(c, s, block_rows=block)
        matrix = CompressedRouteMatrix(c, s)
        for q in (np.zeros(3, np.float32), np.ones(2, np.float32), np.ones((1, 3), np.float32),
                  np.ones(3, np.float64), np.ones(3, np.complex64),
                  np.full(3, np.nan, np.float32), np.full(3, np.inf, np.float32)):
            with self.assertRaises(ValueError):
                matrix @ q

    def test_query_temporary_peak_below_full_float_bank(self):
        # Traced allocations only: not a total-process RAM or latency claim.
        matrix = CompressedRouteMatrix.from_float32(np.ones((2048, 384), np.float32))
        q = normalize(np.ones(384, np.float32))
        tracemalloc.start()
        try:
            matrix @ q
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 2048 * 384 * 4)

    def test_quantizer_receives_only_bounded_row_blocks(self):
        sizes = []
        def observe(frame, event, arg):
            if event == "call" and frame.f_code is pack_rows.__code__:
                sizes.append(len(frame.f_locals["x"]))
        previous = sys.getprofile()
        sys.setprofile(observe)
        try:
            CompressedRouteMatrix.from_float32(np.ones((73, 65), np.float32), block_rows=7)
        finally:
            sys.setprofile(previous)
        self.assertEqual(sizes, [7] * 10 + [3])

    @unittest.skipUnless(MAIN.is_file(), "private Main RouteIndex unavailable")
    def test_main_routeindex_and_unchanged_selection_verifier(self):
        # Load the actual unchanged implementation, never copy it or patch globals.
        before = hashlib.sha256(MAIN.read_bytes()).hexdigest()
        spec = importlib.util.spec_from_file_location("compression_main_fixture", MAIN)
        main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main)
        from experiments.memory.ladder_text_memory import LadderTextMemory

        x = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        matrix = CompressedRouteMatrix.from_float32(x)
        c, s = pack_rows(x, 8)
        decoded = normalize(unpack_rows(c, s, 3, 8))
        pointers = [dict(source_id=key, start=0, end=5) for key in ("c", "a", "b")]

        class Lexical:
            lengths = np.ones(3, np.int32)
            def score(self, query):
                return np.array([0, 2, 1], np.float32)
            def query_work(self, query):
                return dict(matching_terms=1)

        for arm in ("dense", "hybrid"):
            calls = []
            def encode(texts):
                calls.append(list(texts))
                return np.array([[1, 0, 0]], np.float32)
            route = main.RouteIndex(Lexical(), matrix, encode, arm, pointers)
            control = main.RouteIndex(Lexical(), decoded, encode, arm, pointers)
            actual = route.score("synthetic query")
            expected = control.score("synthetic query")
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
            np.testing.assert_array_equal(canonical_order(actual, pointers),
                                          canonical_order(expected, pointers))
            if arm == "dense":
                self.assertEqual(canonical_order(actual, pointers).tolist(), [1, 0, 2])
            route.score("synthetic query")
            self.assertEqual(route.actual_query_encodes, 1)
            self.assertEqual(route.work["actual_query_encodes"], 0)
            self.assertEqual(calls, [["synthetic query"], ["synthetic query"]])
            self.assertEqual(route.work["vector_bytes_scanned"], matrix.nbytes)

        @dataclass(frozen=True)
        class Span:
            source_id: str
            start_byte: int = 0
            end_byte: int = 5

        class SyntheticPacket:
            spans = [Span(key) for key in ("c", "a", "b")]
            preparation_sha256 = "0" * 64
            rung = dict(target_tokens=1)
            def __init__(self):
                self.verified = []
            def verify(self, pointer):
                self.verified.append(pointer.source_id)
                # Explicit fake verifier: source 'b' rejected, not real evidence.
                return dict(status=("SOURCE_CHANGED" if pointer.source_id == "b"
                                    else "VERIFIED_SOURCE_BYTES"),
                            pointer=dict(source_id=pointer.source_id), text="fake!",
                            packed_start_byte=0, packed_end_byte=5, source_bytes_read=5)

        # Exercise unchanged selection/novelty plumbing using synthetic packet only.
        route = main.RouteIndex(Lexical(), matrix, encode, "dense", pointers)
        base = LadderTextMemory.__new__(LadderTextMemory)
        base._index = route
        base._pointers = tuple(pointers)
        base._document_ids = ("c", "a", "b")
        base._seen = set()
        base._policy = "flat"
        base._archive_sha256 = "0" * 64
        base._packet = SyntheticPacket()
        self.assertEqual(base.search("synthetic query", 1)["selected_rows"], [1])
        self.assertEqual(base.search("synthetic query", 1, novelty=True)["selected_rows"], [0])
        rejected = base.search("synthetic query", 1, novelty=True)
        self.assertEqual(rejected["rejected_rows"], [2])
        self.assertEqual(rejected["passages"], [])
        self.assertEqual(base._seen, {0, 1})
        self.assertEqual(base._packet.verified, ["a", "c", "b"])
        self.assertEqual(route.actual_query_encodes, 1)
        self.assertEqual(base.search("source:fake", 1)["passages"], [])
        self.assertEqual(hashlib.sha256(MAIN.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
