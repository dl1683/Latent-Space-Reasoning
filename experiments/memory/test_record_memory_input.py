import json
import unittest

from experiments.memory.record_memory_input import encode_records, decode_records, digest


class EvidenceInputTests(unittest.TestCase):
    def test_metadata_only_update_changes_input_and_keeps_history(self):
        old = [dict(text='same instruction', revision=1, state='active', role='instruction'),
               dict(text='same instruction', revision=2, state='tombstoned', role='instruction')]
        new = [dict(x) for x in old]; new[1]['state'] = 'active'
        first, second = encode_records(old), encode_records(new)
        self.assertNotEqual(first, second)
        self.assertEqual(decode_records(first, expected_sha256=digest(first)), old)
        self.assertEqual(json.loads(first)['role'], 'evidence')
        self.assertEqual(encode_records([dict(reversed(list(x.items()))) for x in old]), first)
        with self.assertRaises(ValueError):
            decode_records(second, expected_sha256=digest(first))

    def test_record_binding_checked_inside_valid_outer_checksum(self):
        envelope = json.loads(encode_records([dict(text='body', provenance='agent')]))
        envelope['records'][0]['record']['provenance'] = 'user'
        raw = json.dumps(envelope).encode()
        with self.assertRaises(ValueError):
            decode_records(raw, expected_sha256=digest(raw))


if __name__ == '__main__':
    unittest.main()
