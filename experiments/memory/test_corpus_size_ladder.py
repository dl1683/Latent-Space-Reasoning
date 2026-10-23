"""Model-free tests for the actual ladder's invariants."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.memory.corpus_size_ladder import (
    SENSITIVE, byte_chunks, canonical_order, evaluate, file_sha, graph_route, hybrid_scores,
    rung_counts, save, support_chunk_score, support_score, verify_span,
)


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


class CorpusSizeLadderTests(unittest.TestCase):
    def evaluation_fixture(self, root):
        raw = b"alpha evidence"
        key = hashlib.sha256(raw).hexdigest()
        (root / "leaves.bin").write_bytes(raw)
        save(root / "pointers.json", [dict(
            source_id=key, start=0, end=len(raw), span_sha256=key, tokens=2)])
        save(root / "sources.json", [])
        save(root / "questions.json", [dict(id="q", question="alpha", type="title_lookup")])
        save(root / "evaluation_only.json", [dict(id="q", question="alpha",
                                                    type="title_lookup", gold=[key])])
        save(root / "rungs.json", [dict(target_tokens=2, actual_tokens=2,
                                         chunks=1, status="AVAILABLE")])
        names = ("leaves.bin", "pointers.json", "sources.json", "questions.json",
                 "evaluation_only.json", "rungs.json")
        save(root / "prepare.json", dict(
            pins={name: file_sha(root / name) for name in names}, reader_window=12288))

    def test_utf8_coverage_is_exact_and_nonoverlapping(self):
        text = "é中 Alpha\nBeta"
        spans = byte_chunks(text, CharacterTokenizer(), width=3)
        raw = text.encode("utf8")
        self.assertEqual(b"".join(raw[a:b] for a, b in spans), raw)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(raw))
        for a, b in spans:
            raw[a:b].decode("utf8")
        self.assertTrue(all(a[1] == b[0] for a, b in zip(spans, spans[1:])))

    def test_empty_source_has_no_chunks(self):
        self.assertEqual(byte_chunks("", CharacterTokenizer()), [])

    def test_chunk_width_refuses_bad_values(self):
        for width in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                byte_chunks("a", CharacterTokenizer(), width)

    def test_rungs_are_nested_whole_chunks(self):
        rows = rung_counts([6, 5, 7, 8], sizes=(10, 20, 100))
        self.assertEqual([r["chunks"] for r in rows], [2, 4, 4])
        self.assertEqual([r["actual_tokens"] for r in rows], [11, 26, 26])
        self.assertEqual(rows[-1]["status"], "INSUFFICIENT_SOURCE")

    def test_rungs_refuse_invalid_counts(self):
        for counts in ([0], [-1], [[2]]):
            with self.assertRaises(ValueError):
                rung_counts(counts)

    def test_graph_never_crosses_source_or_budget(self):
        documents = ["a", "a", "b", "b", "c", "c"]
        selected, edges = graph_route([1, 4, 3, 0, 2, 5], documents, budget=4)
        self.assertEqual(selected, [1, 4, 0, 5])
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(edges, 2)

    def test_graph_same_exposure_is_deterministic(self):
        args = ([3, 0, 1, 2], ["a", "b", "c", "d"])
        self.assertEqual(graph_route(*args), graph_route(*args))

    def test_large_graph_fill_remains_budget_bounded(self):
        selected, edges = graph_route(range(100_000), list(range(100_000)))
        self.assertEqual(selected, list(range(10)))
        self.assertEqual(edges, 0)

    def test_verification_rejects_mutated_and_missing_spans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "leaves.bin"
            raw = "évidence".encode("utf8")
            path.write_bytes(raw)
            pointer = dict(start=0, end=len(raw),
                           span_sha256=hashlib.sha256(raw).hexdigest())
            self.assertEqual(verify_span(root, pointer), ("évidence", len(raw)))
            path.write_bytes(b"x" + raw[1:])
            with self.assertRaises(ValueError):
                verify_span(root, pointer)
            path.write_bytes(raw[:-1])
            with self.assertRaises(ValueError):
                verify_span(root, pointer)

    def test_support_recall_is_not_chunk_recall_or_absence(self):
        pointers = [{"source_id": key} for key in ("a", "a", "b")]
        self.assertEqual(support_score([0, 1], pointers, ["a", "b"])["recall"], .5)
        self.assertTrue(support_score([0, 2], pointers, ["a", "b"])["all_sources_found"])
        self.assertFalse(support_chunk_score([0, 2], pointers, ["a", "b"])["all_support_chunks_found"])
        self.assertAlmostEqual(support_chunk_score([0, 2], pointers, ["a", "b"])["support_chunk_recall"], 2 / 3)
        with self.assertRaises(ValueError):
            support_score([], pointers, [])

    def test_zero_scores_do_not_prefer_support_enriched_row_order(self):
        pointers = [dict(source_id=key, start=i, end=i + 1)
                    for i, key in enumerate(("z", "a", "m"))]
        self.assertEqual(canonical_order(np.zeros(3), pointers).tolist(), [1, 2, 0])
        dense = np.array([.9, .1, .5])
        np.testing.assert_array_equal(hybrid_scores(dense, np.zeros(3), pointers), dense)

    def test_private_material_filter(self):
        self.assertIsNotNone(SENSITIVE.search("person" + chr(64) + "example.invalid"))
        self.assertIsNotNone(SENSITIVE.search("-----BEGIN " + "PRIVATE" + " KEY-----"))
        self.assertIsNone(SENSITIVE.search("A cobalt observatory."))

    def test_final_cleanup_failure_never_admits_dense(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.evaluation_fixture(root)
            save(root / "panel_resources.json", dict(status="COMPLETED", cleanup_verified=True))
            save(root / "operation.json", dict(status="ABORTED_CLEANUP_UNVERIFIED",
                                                 safe_cleanup=False, own_lease_released=False))
            result = evaluate(root, root / "out")
            self.assertFalse(result["corpus_encoding_admitted"])
            self.assertEqual({r["arm"] for r in result["aggregate"]}, {"bm25"})

    def test_shape_preserving_vector_modification_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.evaluation_fixture(root)
            np.save(root / "bank.npy", np.zeros((1, 384), dtype=np.float32))
            np.save(root / "query_vectors.npy", np.zeros((1, 384), dtype=np.float32))
            save(root / "panel_measurements.json", dict(
                execution_id="fixture", pins_sha256="pins",
                bank_sha256=file_sha(root / "bank.npy"),
                query_vectors_sha256=file_sha(root / "query_vectors.npy")))
            save(root / "panel_resources.json", dict(
                status="COMPLETED", cleanup_verified=True, execution_id="fixture",
                pins_sha256="pins", artifact_sha256={
                    "panel_measurements.json": file_sha(root / "panel_measurements.json")}))
            save(root / "operation.json", dict(
                status="COMPLETED", safe_cleanup=True, own_lease_released=True))
            np.save(root / "bank.npy", np.ones((1, 384), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "vector artifact changed"):
                evaluate(root, root / "out")


if __name__ == "__main__":
    unittest.main()
