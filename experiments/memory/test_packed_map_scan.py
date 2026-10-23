import tempfile
from pathlib import Path
import unittest

import numpy as np

from experiments.memory.compressed_routing_map import normalize, pack_rows, unpack_rows
from experiments.memory.packed_map_scan import Packed8Scan


class PackedScanTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(1845)
        x = rng.normal(size=(31, 17)).astype(np.float32)
        x[0] = 0
        x[1] *= 1e-12
        codes, scales = pack_rows(x, 8)
        return codes, scales, rng.normal(size=17).astype(np.float32)

    def test_normalization_and_small_scales(self):
        codes, scales, query = self.fixture()
        reference = normalize(unpack_rows(codes, scales, 17, 8)) @ query
        scan = Packed8Scan.build(codes, scales)
        for block in (1, 7, 32):
            np.testing.assert_allclose(scan.scores(query, block_columns=block),
                                       reference, atol=2e-6, rtol=0)
        self.assertEqual(scan.payload_bytes, codes.nbytes + 8*len(codes))

    def test_constant_query_duplicate_rows_and_cold_state(self):
        codes, scales, query = self.fixture()
        codes[2], scales[2] = codes[3], scales[3]
        scan = Packed8Scan.build(codes, scales)
        scores = scan.scores(query)
        self.assertEqual(scores[2], scores[3])
        np.testing.assert_array_equal(scan.scores(np.zeros(17, np.float32)), np.zeros(31))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"map.npz"
            scan.save(path)
            cold = Packed8Scan.load(path)
            np.testing.assert_array_equal(cold.scores(query), scores)
            self.assertEqual(set(np.load(path, allow_pickle=False).files),
                             {"codes", "multipliers"})
            with self.assertRaises(FileExistsError):
                scan.save(path)

    def test_bad_inputs(self):
        codes, scales, query = self.fixture()
        with self.assertRaises(ValueError):
            Packed8Scan.build(codes.astype(np.int16), scales)
        with self.assertRaises(ValueError):
            Packed8Scan.build(codes, np.zeros_like(scales))
        scan = Packed8Scan.build(codes, scales)
        for bad in (query.astype(np.float64), query[:-1], np.full(17, np.nan, np.float32)):
            with self.assertRaises(ValueError):
                scan.scores(bad)
        with self.assertRaises(ValueError):
            scan.scores(query, block_columns=0)


if __name__ == "__main__":
    unittest.main()
