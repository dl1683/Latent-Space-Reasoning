"""Pure finite checks; no torch import, CUDA operation or model downloads."""
import unittest

from experiments.memory.attention_completeness import (
    attention_query, bm25, choose_heads, ranked, summarize,
)


class AttentionProbeTests(unittest.TestCase):
    def test_dev_only_selection_and_test_reversal(self):
        # DEV selects head 0; TEST prefers head 1. A scorer that accidentally
        # chooses on TEST would pass, whereas the frozen shared head must fail.
        dev = {'dev': [[1, 0, 0, 0], [0, 1, 0, 0]]}
        heads = choose_heads(dev, {'dev': 0}, count=1)
        self.assertEqual(heads, [0])
        test = {'test': [[0, 1, 1, 1], [1, 0, 0, 0]]}
        self.assertFalse(summarize(test, {'test': 0}, heads)[0]['recall3'])

    def test_missing_question_is_not_vacuous_pass(self):
        with self.assertRaises(ValueError):
            summarize({}, {'test': 0}, [0])
        with self.assertRaises(ValueError):
            choose_heads({'dev': [[1, 0]]}, {'other': 0})

    def test_nan_and_zero_mass_refused(self):
        with self.assertRaises(ValueError):
            ranked([float('nan'), 0])
        with self.assertRaises(ValueError):
            summarize({'test': [[0, 0]]}, {'test': 0}, [0])

    def test_bm25_entity_disambiguation(self):
        docs = ['The code for RedStation is green.',
                'The code for BlueStation is yellow.']
        self.assertEqual(ranked(bm25(docs, 'code for BlueStation'))[0], 1)

    def test_no_unguarded_model_operation(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'supervision'):
                attention_query(None, None, None, None)

    def test_empty_pin_manifest_rejected_before_model_import(self):
        import hashlib
        import json
        import os
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'pins.json').write_text('{}')
            (root/'config.json').write_text(json.dumps(dict(reader=folder,dense=folder)))
            env=dict(os.environ,LSR_ATTENTION_SUPERVISED='1',
                     LSR_EXECUTION_ID='negative-fixture',
                     LSR_PIN_DIGEST=hashlib.sha256(b'{}').hexdigest())
            result=subprocess.run([sys.executable,str(Path(__file__).with_name(
                'attention_completeness_run.py')),folder,'tiny'],
                env=env,capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('missing required pin coverage',result.stderr)


if __name__ == '__main__':
    unittest.main()
