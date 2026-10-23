import hashlib
import tempfile
from pathlib import Path
import unittest

import numpy as np

from experiments.memory.refined_routing_map import (
    RefinedMap, feature_scores, rank, verified_spans,
)


class RefinedMapTests(unittest.TestCase):
    def test_all_rates_preserve_exact_random_and_zero_ties(self):
        rng = np.random.default_rng(1750)
        x = rng.normal(size=(79, 17)).astype(np.float32)
        x[5] = x[3]
        queries = [*rng.normal(size=(8, 17)).astype(np.float32),
                   np.zeros(17, dtype=np.float32)]
        for bits in (1, 2, 4, 8):
            memory = RefinedMap.build(x, bits)
            for q in queries:
                out = memory.search(q, lambda ids: x[ids], k=10)
                expected = feature_scores(x, q)
                np.testing.assert_array_equal(out["indices"], rank(expected)[:10])
                self.assertTrue((expected >= out["lower"]).all())
                self.assertTrue((expected <= out["upper"]).all())
                self.assertEqual(out["semantic_absence"], "UNKNOWN")

    def test_actual_roundtrip_and_nonaligned_bit_width(self):
        x = np.array([[.3, .2, .1], [-.3, .5, -.8]], dtype=np.float32)
        for bits in (1, 2, 4, 8):
            a = RefinedMap.build(x, bits)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "map.npz"
                a.save(path)
                b = RefinedMap.load(path)
                np.testing.assert_array_equal(a.restore(), b.restore())
                np.testing.assert_array_equal(a.errors, b.errors)
                np.testing.assert_array_equal(
                    a.search(x[0], lambda ids: x[ids], 2)["scores"],
                    b.search(x[0], lambda ids: x[ids], 2)["scores"])
                with self.assertRaises(FileExistsError):
                    a.save(path)

    def test_refinement_repairs_a_quantized_order_error(self):
        rng = np.random.default_rng(18)
        x = rng.normal(size=(32, 7)).astype(np.float32)
        q = rng.normal(size=7).astype(np.float32)
        expected = rank(x.astype(np.float64) @ q.astype(np.float64))[:3]
        memory = RefinedMap.build(x, 1)
        out = memory.search(q, lambda ids: x[ids], k=3)
        self.assertFalse(np.array_equal(out["approximate_indices"], expected))
        np.testing.assert_array_equal(out["indices"], expected)

    def test_invalid_input_and_exact_reader_refused(self):
        x = np.eye(3, dtype=np.float32)
        for value in (x.astype(np.float64), x*np.nan, np.zeros((0, 3), dtype=np.float32)):
            with self.assertRaises(ValueError):
                RefinedMap.build(value, 4)
        memory = RefinedMap.build(x, 4)
        for q, k in ((x[0].astype(np.float64), 1), (x[0], 0), (x[0], True)):
            with self.assertRaises(ValueError):
                memory.search(q, lambda ids: x[ids], k)
        with self.assertRaises(ValueError):
            memory.search(x[0], lambda ids: x[ids].astype(np.float64), 1)

    def test_source_span_identity_and_mutation_refusal(self):
        data = "A clause. Un café. Last clause.".encode("utf8")
        digest = hashlib.sha256(data).hexdigest()
        pointer = dict(document=digest, source_sha256=digest,
                       file=f"leaves/{digest}.txt", start=10, end=17)
        original = dict(pointer)
        with tempfile.TemporaryDirectory() as directory:
            leaf = Path(directory) / pointer["file"]
            leaf.parent.mkdir()
            leaf.write_bytes(data)
            result = verified_spans(directory, [pointer])
            self.assertEqual(result["passages"][0]["text"], "Un café")
            self.assertEqual(pointer, original)
            self.assertEqual(result["passages"][0]["pointer"], original)
            self.assertEqual(result["logical_source_bytes_read"], len(data))
            for override in (dict(file="../foreign"), dict(start=-1), dict(end=999),
                             dict(start=True), dict(document="foreign")):
                with self.assertRaises(ValueError):
                    verified_spans(directory, [dict(pointer, **override)])
            leaf.write_bytes(data + b" mutated")
            with self.assertRaisesRegex(ValueError, "source changed"):
                verified_spans(directory, [pointer])

    def test_rowwise_scoring_is_batch_shape_independent(self):
        rng = np.random.default_rng(21)
        x = rng.normal(size=(1814, 384)).astype(np.float32)
        q = rng.normal(size=384).astype(np.float32)
        ids = np.array([0, 1, 5, 399, 1800])
        np.testing.assert_array_equal(feature_scores(x, q)[ids],
                                      feature_scores(x[ids], q))


if __name__ == "__main__":
    unittest.main()
