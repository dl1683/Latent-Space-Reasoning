import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.memory.hierarchical_routing_map import HierarchicalMap, rank, scores


class HierarchyTests(unittest.TestCase):
    def test_pruning_and_roundtrip(self):
        x = np.asarray([[1, i/1000] for i in range(32)]
                       + [[-1, i/1000] for i in range(32)], dtype=np.float32)
        q = np.asarray([1, .1], dtype=np.float32)
        for boxes in (False, True):
            memory = HierarchicalMap.build(x, leaf_size=8, boxes=boxes)
            expected = rank(scores(memory.decode(np.arange(64)), q))[:3].tolist()
            out = memory.search(q, k=3)
            self.assertEqual(out["indices"], expected)
            self.assertLess(out["decoded_rows"], 64)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"map.npz"
                memory.save(path)
                restored = HierarchicalMap.load(path)
                self.assertEqual(restored.search(q, 3), out)

    def test_random_exact_and_ties(self):
        rng = np.random.default_rng(37)
        x = rng.normal(size=(125, 9)).astype(np.float32)
        x[9:12] = x[0]
        for boxes in (False, True):
            memory = HierarchicalMap.build(x, 8, boxes)
            values = memory.decode(np.arange(len(x)))
            for q in [np.zeros(9, np.float32), x[0], *rng.normal(size=(12, 9)).astype(np.float32)]:
                expected = rank(scores(values, q))[:10].tolist()
                a = memory.search(q)
                self.assertEqual(a["indices"], expected)
                self.assertEqual(memory.search(q), a)
                self.assertEqual(len(set(a["decoded_ids"])), a["decoded_rows"])
                self.assertEqual(a["semantic_absence"], "UNKNOWN")

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            HierarchicalMap.build(np.full((2, 3), np.nan, np.float32))
        memory = HierarchicalMap.build(np.eye(3, dtype=np.float32))
        for q in (np.zeros(3), np.zeros(2, np.float32), np.full(3, np.inf, np.float32)):
            with self.assertRaises(ValueError):
                memory.search(q, 1)
        for k in (0, 4, True):
            with self.assertRaises(ValueError):
                memory.search(np.zeros(3, np.float32), k)
        memory.nodes[0, 0] = 1
        with self.assertRaises(ValueError):
            memory.validate()


if __name__ == "__main__":
    unittest.main()
