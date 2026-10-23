"""Host persistence invariants; these are not semantic model tests."""
import tempfile
import unittest
from pathlib import Path

from experiments.memory.local_native_kv import NativeMemory, qualify_binary, finalize_binary


class NativeMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.torch = torch

    def memory(self, dtype=None):
        t = self.torch
        k = t.arange(24, dtype=t.float32).reshape(1, 2, 3, 4)
        if dtype is not None:
            k = k.to(dtype)
        return NativeMemory(((k, k + 1),), 3)

    def test_query_cache_has_no_alias_to_retained_memory(self):
        memory = self.memory()
        before = memory.digest()
        cache = memory.cache()
        cache.layers[0].keys.add_(100)
        self.assertEqual(memory.digest(), before)
        self.assertFalse(self.torch.equal(cache.layers[0].keys, memory.layers[0][0]))

    def test_roundtrip_both_cpu_precisions(self):
        for dtype in (self.torch.float32, self.torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                memory = self.memory(dtype)
                path = Path(directory) / 'memory.pt'
                memory.save(path)
                loaded = NativeMemory.load(path)
                self.assertEqual(loaded.digest(), memory.digest())
                self.assertEqual(loaded.payload_bytes, 48 * dtype.itemsize)

    def test_reject_nonfinite_wrong_length_and_mixed_precision(self):
        t = self.torch
        for failure in ('length', 'nonfinite', 'dtype'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                memory = self.memory()
                k, v = memory.layers[0]
                obj = dict(layers=((k, v),), length=3)
                if failure == 'length':
                    obj['length'] = 2
                elif failure == 'nonfinite':
                    v[0, 0, 0, 0] = float('nan')
                else:
                    obj['layers'] = ((k, v.to(t.bfloat16)),)
                path = Path(directory) / 'invalid.pt'
                t.save(obj, path)
                with self.assertRaises(ValueError):
                    NativeMemory.load(path)

    def test_uncertain_is_never_a_negative_answer(self):
        self.assertEqual(qualify_binary([0., 2.], [0., 2.])['answer'], 'B')
        for cached, reference in [([0., .01], [0., .01]),
                                  ([0., 2.], [0., 1.8]),
                                  ([float('nan'), 2.], [0., 2.]),
                                  ([0., 2.], [2., 0.])]:
            self.assertEqual(qualify_binary(cached, reference)['answer'], 'UNCERTAIN')

    def test_binary_api_does_not_expose_uncalibrated_no(self):
        cold = finalize_binary(dict(arms=dict(cached=dict(answer='B', logits=[-2., 0.]))))
        self.assertEqual(cold['answer'], 'UNCERTAIN')
        self.assertEqual(cold['arms']['cached']['raw_answer'], 'B')
        self.assertNotIn('answer', cold['arms']['cached'])
        warm = finalize_binary(dict(arms=dict(cached=dict(answer='B', logits=[-2., 0.]),
                                             full=dict(answer='B', logits=[-2., 0.]))))
        self.assertEqual(warm['answer'], 'B')
        bad = finalize_binary(dict(arms=dict(cached=dict(answer='B', logits=[-2., .2]),
                                            full=dict(answer='B', logits=[-2., 0.]))))
        self.assertEqual(bad['answer'], 'UNCERTAIN')


if __name__ == '__main__':
    unittest.main()
