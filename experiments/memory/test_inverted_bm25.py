"""Software equivalence tests, not memory quality or scale evidence."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.memory.compressed_routing_map import BM25, ranked
from experiments.memory.inverted_bm25 import InvertedBM25


class InvertedBM25Tests(unittest.TestCase):
    def assert_equivalent(self, texts, queries):
        reference, index = BM25(texts), InvertedBM25(iter(texts))
        for query in queries:
            np.testing.assert_array_equal(index.score(query), reference.score(query))
            np.testing.assert_array_equal(ranked(index.score(query)),
                                          ranked(reference.score(query)))
        return index

    def test_literal_repeats_unicode_and_missing_words(self):
        self.assert_equivalent(
            ["Cedar cedar; birch.", "birch: maple maple", "", "naïve café 東京"],
            ["CEDAR cedar", "birch maple", "東京 café", "unseen", "", "cedar unseen"])

    def test_empty_corpus(self):
        index = InvertedBM25([])
        self.assertEqual(index.score("anything").tolist(), [])
        self.assertEqual(index.query_work("anything")["postings_visited"], 0)

    def test_zero_length_documents_and_ties(self):
        self.assert_equivalent(["", "", "same", "same"], ["", "same", "missing"])

    def test_randomized_reference(self):
        rng = np.random.default_rng(18062026)
        vocabulary = ["alpha", "beta", "gamma", "delta", "café", "東京"]
        texts = [" ".join(rng.choice(vocabulary, size=int(rng.integers(0, 40))))
                 for _ in range(120)]
        queries = [" ".join(rng.choice(vocabulary + ["unknown"], size=8))
                   for _ in range(30)]
        self.assert_equivalent(texts, queries)

    def test_save_reload_and_refuse_overwrite(self):
        index = InvertedBM25(["red blue red", "blue green", "東京 café"])
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "index.npz"
            index.save(path)
            loaded = InvertedBM25.load(path)
            self.assertEqual(index.vocabulary, loaded.vocabulary)
            for query in ("red", "green blue", "東京 café", "missing"):
                np.testing.assert_array_equal(index.score(query), loaded.score(query))
            with self.assertRaises(FileExistsError):
                index.save(path)

    def test_empty_vocabulary_roundtrip(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "empty.npz"
            InvertedBM25(["", ""]).save(path)
            self.assertEqual(InvertedBM25.load(path).score("missing").tolist(), [0., 0.])

    def test_read_only_state(self):
        index = InvertedBM25(["one two", "two three"])
        for array in (index.offsets, index.documents, index.frequencies,
                      index.lengths, index.norm):
            with self.assertRaises(ValueError):
                array[0] = 999
        with self.assertRaises(TypeError):
            index.lookup["injected"] = 0

    def test_posting_work_counts_no_reader(self):
        index = InvertedBM25(["a b a", "a c", "c"])
        self.assertEqual(index.query_work("a a c missing"),
                         dict(matching_terms=2, postings_visited=4,
                              score_vector_bytes=12, reader_calls=0))

    def test_reject_inconsistent_frequency_state(self):
        index = InvertedBM25(["one one", "two"])
        bad = index.frequencies.copy()
        bad[0] += 1
        with self.assertRaises(ValueError):
            InvertedBM25.__new__(InvertedBM25)._initialize(
                index.vocabulary, index.offsets, index.documents, bad, index.lengths)

    def test_reject_duplicate_posting_rows(self):
        index = InvertedBM25(["same", "same"])
        with self.assertRaises(ValueError):
            InvertedBM25.__new__(InvertedBM25)._initialize(
                index.vocabulary, index.offsets, np.array([0, 0], dtype=np.int64),
                index.frequencies, index.lengths)


if __name__ == "__main__":
    unittest.main()
