"""Small CPU/software fixtures only; no evaluator assets, model or cloud access.

Run explicitly: python -m unittest experiments.test_dogfood_audit
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.dogfood_audit import (
    audit_manifest,
    audit_payloads,
    make_canary_reference,
)


class FakeEncoder:
    """Deterministic whole-span hashes, plus one deliberately semantic equivalence."""

    def __init__(self, semantic=False):
        self.semantic = semantic
        self.batch_sizes = []
        self.encoded_texts = []

    def __call__(self, texts):
        self.batch_sizes.append(len(texts))
        self.encoded_texts.extend(texts)
        result = []
        for text in texts:
            words = set(text.split())
            if self.semantic and (
                {"secret", "amber"} <= words or {"hidden", "yellow"} <= words
            ):
                vector = np.zeros(256)
                vector[0] = 1.0
            else:
                # Dense signed fingerprint; unlike a bag-of-words encoder this does
                # not create unrelated short-window semantic positives in fixtures.
                raw = hashlib.sha256(text.encode()).digest()
                vector = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).astype(float)
                vector = 2 * vector - 1
            result.append(vector)
        return np.asarray(result)


class PayloadAuditTests(unittest.TestCase):
    def audit(self, text, gold, source="", **kwargs):
        return audit_payloads(
            [{"label": "reader-prompt", "text": text}],
            [{"id": "gold-1", "text": gold}],
            [{"label": "permitted-source", "text": source}] if source else [],
            **kwargs,
        )

    def test_exclusive_exact_and_ngram(self):
        text = "violet bronze copper silver cobalt nickel jade quartz ruby pearl opal topaz"
        result = self.audit("before " + text + " after", text, embedder=FakeEncoder())
        self.assertEqual(result["admission"], "FAIL")
        self.assertEqual(result["checks"]["ngram"], "FAIL")
        self.assertTrue(result["leakage"])
        self.assertEqual(result["leakage"][0]["payload_word_span"], [1, 13])
        json.dumps(result, allow_nan=False)

    def test_unicode_words_are_normalized(self):
        result = self.audit("ＡＭＢＥＲ—SIGNAL: ready!", "amber signal READY",
                            embedder=FakeEncoder())
        self.assertEqual(result["checks"]["exact"], "FAIL")

    def test_actual_source_quote_is_legitimate(self):
        source = "The measured river level remains below the old stone bridge."
        result = self.audit(source, source, source, embedder=FakeEncoder())
        self.assertEqual(result["admission"], "PASS", result)
        self.assertEqual(result["leakage"], [])
        self.assertTrue(result["source_supported"])
        for row in result["source_supported"]:
            self.assertEqual(row["source_labels"], ["permitted-source"])

    def test_partial_quote_does_not_exempt_exclusive_remainder(self):
        quote = "The measured river level remains below the old stone bridge"
        novel = "secret evaluator preference is amber"
        result = self.audit(quote + ". " + novel, quote + ". " + novel, quote,
                            embedder=FakeEncoder())
        self.assertEqual(result["admission"], "FAIL")
        self.assertTrue(result["source_supported"])
        novel_hash = hashlib.sha256(novel.encode()).hexdigest()
        self.assertTrue(any(row["span_sha256"] == novel_hash for row in result["leakage"]))

    def test_short_novel_tail_after_same_sentence_source_quote(self):
        quote = "The measured river level remains below the old stone bridge"
        result = self.audit("secret amber", quote + " secret amber", quote,
                            embedder=FakeEncoder())
        self.assertEqual(result["admission"], "FAIL")
        self.assertTrue(any(row["payload_word_span"] == [0, 2]
                            for row in result["leakage"]))

    def test_one_word_source_prefix_does_not_hide_short_novel_tail(self):
        result = self.audit("secret amber", "published secret amber", "published",
                            embedder=FakeEncoder())
        self.assertEqual(result["admission"], "FAIL")
        self.assertEqual(result["checks"]["exact"], "FAIL")

    def test_mosaic_of_source_words_does_not_exempt_complete_gold(self):
        result = self.audit("amber secret criterion", "amber secret criterion",
                            "amber unrelated secret other criterion",
                            embedder=FakeEncoder())
        self.assertEqual(result["checks"]["exact"], "FAIL")
        self.assertEqual(result["admission"], "FAIL")

    def test_source_quote_alone_does_not_leak_mixed_gold(self):
        quote = "The measured river level remains below the old stone bridge"
        result = self.audit(quote, quote + ". Secret amber criterion.", quote,
                            embedder=FakeEncoder())
        self.assertEqual(result["admission"], "PASS", result)
        self.assertTrue(result["source_supported"])

    def test_fuzzy_exclusive_overlap(self):
        gold = "Confidential evaluator scoring requires the bright amber compliance marker"
        text = "Confidential evaluator scoring requires the bright ambar compliance marker"
        result = self.audit(text, gold, embedder=FakeEncoder())
        self.assertEqual(result["admission"], "FAIL")
        self.assertEqual(result["checks"]["fuzzy"], "FAIL")
        self.assertTrue(any(row["method"] == "fuzzy" for row in result["leakage"]))

    def test_semantic_exclusive_overlap(self):
        result = self.audit("hidden yellow condition", "secret amber requirement",
                            "The public source concerns rainfall.", embedder=FakeEncoder(True))
        self.assertEqual(result["checks"]["exact"], "PASS")
        self.assertEqual(result["checks"]["ngram"], "PASS")
        self.assertEqual(result["checks"]["fuzzy"], "PASS")
        self.assertEqual(result["checks"]["embedding"], "FAIL")
        self.assertEqual(result["admission"], "FAIL")

    def test_semantic_paraphrase_with_different_word_count(self):
        result = self.audit("hidden yellow",
                            "the secret amber evaluation condition has priority",
                            embedder=FakeEncoder(True))
        self.assertEqual(result["checks"]["embedding"], "FAIL")
        self.assertEqual(result["admission"], "FAIL")

    def test_missing_embedding_is_unrun_not_pass(self):
        result = self.audit("ordinary rainfall report", "secret amber requirement")
        self.assertEqual(result["checks"]["embedding"], "UNRUN")
        self.assertEqual(result["admission"], "NOT_PASS")
        self.assertEqual(result["errors"], [])

    def test_all_canary_hits_reject_even_when_source_supported(self):
        ref = make_canary_reference("synthetic only", nonce="a" * 32)
        before = dict(ref)
        result = audit_payloads(
            [{"label": "prompt", "text": ref["text"]}], [ref],
            [{"label": "source", "text": ref["text"]}], embedder=FakeEncoder(),
        )
        self.assertEqual(result["admission"], "FAIL")
        self.assertEqual(result["checks"]["canary"], "FAIL")
        self.assertEqual(ref, before)
        self.assertTrue(any(row["method"] == "canary" for row in result["leakage"]))

    def test_unknown_canary_is_also_rejected(self):
        result = self.audit("DOGFOODCANARY" + "b" * 32, "unrelated criterion",
                            embedder=FakeEncoder())
        self.assertEqual(result["checks"]["canary"], "FAIL")

    def test_encoder_failures_fail_closed_without_echoing_gold(self):
        for output in (np.zeros((1, 3)), np.array([[np.nan, 1]]), np.ones(3)):
            with self.subTest(shape=output.shape):
                result = self.audit("ordinary rainfall", "secret amber",
                                    embedder=lambda _: output)
                self.assertEqual(result["admission"], "FAIL")
                self.assertEqual(result["checks"]["embedding"], "FAIL")

        def broken(_):
            raise RuntimeError("private gold MUST NOT appear in receipt")

        result = self.audit("ordinary rainfall", "secret amber", embedder=broken)
        self.assertEqual(result["admission"], "FAIL")
        self.assertNotIn("private gold", json.dumps(result))

    def test_empty_or_duplicate_references_fail_closed(self):
        for refs in ([], [{"id": "x", "text": ""}],
                     [{"id": "x", "text": "one"}, {"id": "x", "text": "two"}]):
            result = audit_payloads([{"label": "p", "text": "hello"}], refs, [],
                                    embedder=FakeEncoder())
            self.assertEqual(result["admission"], "FAIL")

    def test_invalid_threshold_receipt_is_json_serializable(self):
        result = self.audit("ordinary", "secret", fuzzy_threshold=float("nan"))
        self.assertEqual(result["admission"], "FAIL")
        json.dumps(result, allow_nan=False)

    def test_resource_limit_fails_closed(self):
        with patch("experiments.dogfood_audit.MAX_COMPARISONS", 1):
            result = self.audit("ordinary rainfall report", "secret amber requirement",
                                embedder=FakeEncoder())
        self.assertEqual(result["admission"], "FAIL")
        self.assertEqual(result["errors"][0]["kind"], "resource_limit")

    def test_embedding_batches_are_bounded(self):
        encoder = FakeEncoder()
        result = self.audit(" ".join("payload" + str(i) for i in range(50)),
                            "secret amber requirement", embedder=encoder)
        self.assertEqual(result["admission"], "PASS", result)
        self.assertLessEqual(max(encoder.batch_sizes), 32)

    def test_receipt_does_not_include_reference_text(self):
        result = self.audit("unrelated public material",
                            "secret amber evaluation condition",
                            embedder=FakeEncoder())
        self.assertNotIn("secret amber evaluation condition", json.dumps(result))

    def test_indexed_ngram_catches_partial_long_reference(self):
        words = ["privateword" + str(i) for i in range(80)]
        result = self.audit(
            "public prefix " + " ".join(words[20:32]) + " public suffix",
            " ".join(words), embedder=FakeEncoder(),
        )
        self.assertEqual(result["checks"]["ngram"], "FAIL")
        self.assertTrue(any(row["reference_word_span"] == [20, 32]
                            for row in result["leakage"] if row["method"] == "ngram"))

    def test_embedding_texts_encoded_once_across_duplicate_payloads(self):
        encoder = FakeEncoder()
        result = audit_payloads(
            [{"label": "first", "text": "ordinary rainfall report"},
             {"label": "second", "text": "ordinary rainfall report"}],
            [{"id": "gold", "text": "secret amber criterion"}], [],
            embedder=encoder,
        )
        self.assertEqual(result["admission"], "PASS", result)
        self.assertEqual(len(encoder.encoded_texts), len(set(encoder.encoded_texts)))
        self.assertEqual(result["work"]["embedding_unique_texts"], 2)

    def test_100kb_source_screening_has_bounded_work(self):
        source = "".join(
            f"Record {i:05d} reports ordinary rainfall near the eastern river gauge "
            "and confirms that the public bridge remains open.\n"
            for i in range(900)
        )
        self.assertGreater(len(source.encode()), 100_000)
        self.assertLess(len(source.encode()), 120_000)
        gold = (
            "secret amber evaluator preference requires confidential bronze rubric "
            "scoring only after clearance"
        )
        encoder = FakeEncoder()
        result = self.audit(
            "Read the following records.\n" + source + "\nExplain seasonal precipitation.",
            gold, source, embedder=encoder,
        )
        self.assertEqual(result["admission"], "PASS", result)
        self.assertEqual(result["errors"], [])
        self.assertGreater(result["work"]["ngram_windows"], 15_000)
        self.assertLess(result["comparisons"], 50_000)
        self.assertLess(result["work"]["embedding_unique_texts"], 32)
        self.assertEqual(len(encoder.encoded_texts), len(set(encoder.encoded_texts)))
        self.assertLessEqual(max(encoder.batch_sizes), 32)
        self.assertEqual(len(result["source_supported"]), 32)
        self.assertTrue(result["source_supported_summary"]["examples_truncated"])
        self.assertFalse(result["parameters"]["exhaustive_subsentence_fuzzy"])

    def test_full_source_witness_skips_redundant_encoder_work(self):
        # >100k words supplied both as source and payload; >200k total words.
        source = "Public river gauge remains open.\n" * 20_001
        encoder = FakeEncoder()
        result = self.audit(source, "secret amber criterion", source, embedder=encoder)
        self.assertEqual(result["admission"], "PASS", result)
        self.assertEqual(result["work"]["source_pure_payloads"], 1)
        self.assertEqual(result["work"]["embedding_unique_texts"], 0)
        self.assertEqual(encoder.encoded_texts, [])
        self.assertEqual(result["source_supported"][0]["method"], "source_witness")

    def test_embedding_cache_exhaustion_fails_closed(self):
        with patch("experiments.dogfood_audit.MAX_EMBEDDING_CACHE_BYTES", 1):
            result = self.audit("ordinary rainfall report", "secret amber criterion",
                                embedder=FakeEncoder())
        self.assertEqual(result["admission"], "FAIL")
        self.assertEqual(result["errors"][0]["kind"], "resource_limit")


class ManifestAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.file = self.root / "source.txt"
        self.file.write_bytes(b"pinned source bytes\n")
        self.entry = {"path": str(self.file),
                      "sha256": hashlib.sha256(self.file.read_bytes()).hexdigest()}

    def test_exact_allowlist_passes(self):
        result = audit_manifest([self.entry], [str(self.file)])
        self.assertEqual(result["admission"], "PASS", result)
        self.assertEqual(result["files"][0]["size_bytes"], 20)
        json.dumps(result, allow_nan=False)

    def test_tampered_hash_fails(self):
        self.file.write_bytes(b"changed bytes")
        result = audit_manifest([self.entry], [str(self.file)])
        self.assertEqual(result["admission"], "FAIL")
        self.assertEqual(result["files"][0]["status"], "FAIL")

    def test_cross_api_timestamps_do_not_replace_identity_and_hash_checks(self):
        from types import SimpleNamespace
        before = self.file.stat()
        descriptor = SimpleNamespace(st_dev=before.st_dev, st_ino=before.st_ino,
            st_size=before.st_size, st_mtime_ns=before.st_mtime_ns - 100,
            st_ctime_ns=before.st_ctime_ns - 100)
        with patch("experiments.dogfood_audit.os.fstat", return_value=descriptor):
            result = audit_manifest([self.entry], [str(self.file)])
        self.assertEqual(result["admission"], "PASS", result)

    def test_descriptor_metadata_change_during_hash_fails(self):
        from types import SimpleNamespace
        before = self.file.stat()
        common = dict(st_dev=before.st_dev, st_ino=before.st_ino,
                      st_size=before.st_size, st_ctime_ns=before.st_ctime_ns)
        opened = SimpleNamespace(**common, st_mtime_ns=before.st_mtime_ns)
        changed = SimpleNamespace(**common, st_mtime_ns=before.st_mtime_ns + 1)
        with patch("experiments.dogfood_audit.os.fstat", side_effect=[opened, changed]):
            result = audit_manifest([self.entry], [str(self.file)])
        self.assertEqual(result["admission"], "FAIL", result)
        self.assertIn("during hashing", result["errors"][0]["detail"])

    def test_outside_allowlist_is_not_opened_or_resolved(self):
        outside = self.root / "not-allowed.txt"
        entry = {"path": str(outside), "sha256": "0" * 64}
        with patch("experiments.dogfood_audit._checked_file") as check:
            result = audit_manifest([entry], [str(self.file)])
        self.assertEqual(result["admission"], "FAIL")
        check.assert_not_called()

    def test_parent_traversal_escape_is_rejected(self):
        entry = {"path": str(self.root / "child" / ".." / "source.txt"),
                 "sha256": self.entry["sha256"]}
        result = audit_manifest([entry], [str(self.file)])
        self.assertEqual(result["admission"], "FAIL")

    def test_duplicate_and_missing_entries_fail(self):
        other = self.root / "other.txt"
        other.write_text("other", encoding="utf-8")
        for manifest, allowed in (
            ([self.entry, self.entry], [str(self.file)]),
            ([self.entry], [str(self.file), str(self.file)]),
            ([self.entry], [str(self.file), str(other)]),
            ([], []),
        ):
            with self.subTest(manifest=manifest, allowed=allowed):
                self.assertEqual(audit_manifest(manifest, allowed)["admission"], "FAIL")

    def test_relative_path_and_invalid_hash_fail(self):
        self.assertEqual(audit_manifest(
            [{"path": "relative.txt", "sha256": "0" * 64}], ["relative.txt"]
        )["admission"], "FAIL")
        self.assertEqual(audit_manifest(
            [{"path": str(self.file), "sha256": "invalid"}], [str(self.file)]
        )["admission"], "FAIL")

    def test_directory_is_not_a_file(self):
        result = audit_manifest(
            [{"path": str(self.root), "sha256": "0" * 64}], [str(self.root)]
        )
        self.assertEqual(result["admission"], "FAIL")

    def test_symlink_file_and_parent_rejected(self):
        link = self.root / "link.txt"
        directory_link = self.root / "linked-directory"
        real_directory = self.root / "real-directory"
        real_directory.mkdir()
        (real_directory / "child.txt").write_bytes(b"child")
        try:
            link.symlink_to(self.file)
            directory_link.symlink_to(real_directory, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest("OS does not permit symlink creation: " + type(exc).__name__)
        for path in (link, directory_link / "child.txt"):
            with self.subTest(path=path):
                result = audit_manifest(
                    [{"path": str(path), "sha256": "0" * 64}], [str(path)]
                )
                self.assertEqual(result["admission"], "FAIL")

    def test_hardlink_duplicate_identity_rejected(self):
        alias = self.root / "hardlink.txt"
        try:
            os.link(self.file, alias)
        except (OSError, NotImplementedError) as exc:
            self.skipTest("OS does not permit hardlinks: " + type(exc).__name__)
        result = audit_manifest(
            [self.entry, {"path": str(alias), "sha256": self.entry["sha256"]}],
            [str(self.file), str(alias)],
        )
        self.assertEqual(result["admission"], "FAIL")

    def test_reparse_junction_attribute_rejected_without_following(self):
        from types import SimpleNamespace

        fake = SimpleNamespace(st_mode=0, st_file_attributes=0x400)
        with patch.object(Path, "lstat", return_value=fake):
            result = audit_manifest([self.entry], [str(self.file)])
        self.assertEqual(result["admission"], "FAIL")
        self.assertIn("reparse", result["errors"][0]["detail"])


if __name__ == "__main__":
    unittest.main()
