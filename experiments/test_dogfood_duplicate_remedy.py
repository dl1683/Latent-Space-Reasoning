"""CPU-only interface checks; AST-loads selected pure code, never imports the SUT."""
import ast
import copy
import json
from pathlib import Path
import re
import time
import unittest

HERE = Path(__file__).resolve()
NAMES = {
    'navigation_enabled', 'explicit_references_enabled', 'plain_file', 'config_at',
    'validate_typed_source_ids',
    'typed_navigation_metadata', 'accept_typed_navigation_answer',
    'validate_navigation_source_ids', 'accept_navigation_answer',
    'action_schema', 'reader_evidence', 'run_episode', '_run_episode_impl',
    'ANSWER_BODY_INSTRUCTION', 'FORCED_ANSWER_INSTRUCTION', 'SYSTEM', 'SYSTEM_V2',
    'SYSTEM_TYPED_NAVIGATION', 'ACTION_SCHEMA', 'PHASE_SCHEMAS', 'MAX_CITATIONS',
    'CONFIG_KEYS', 'TASK_KEYS',
}


def pure_load(filename):
    source = filename.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(filename))
    nodes = []
    for node in tree.body:
        names = ({node.name} if isinstance(node, ast.FunctionDef) else
                 {t.id for t in node.targets if isinstance(t, ast.Name)}
                 if isinstance(node, ast.Assign) else set())
        if names & NAMES:
            nodes.append(node)
    ns = dict(json=json, re=re, time=time, Path=Path)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(filename), 'exec'), ns)
    return ns


C = pure_load(HERE.with_name('dogfood.py'))


def config(profile='typed_v3', enabled=True):
    result = dict(schema=1, protocol='v2', source_navigation=enabled,
                  seed=1729, num_ctx=8192, max_output_tokens=1024,
                  max_answer_chars=2400, synthesis_max_words=128,
                  top_k=1, caps=[0, 1], chunk_chars=512, tasks=[], readers=[])
    if profile is not None:
        result['source_navigation_protocol'] = profile
    return result


class Memory:
    def __init__(self, size=0):
        self.count = 0
        self.size = size

    def search(self, query, k, **kwargs):
        self.count += 1
        name = 'A' if self.count == 1 else 'B'
        return dict(passages=[dict(source_id=name, handle='S' + str(self.count),
            verification_status='VERIFIED_SOURCE_BYTES', verification='BYTE_IDENTITY_ONLY',
            navigation=dict(next='next:S' + str(self.count), prev=None),
            pointer=dict(source=name, page=None), text='x' * self.size)])


def episode(ns, cfg, raw, cap=1, memory=None):
    calls = []
    def adapter(prompt, phase, schema):
        calls.append((copy.deepcopy(prompt), phase, copy.deepcopy(schema)))
        text = raw if phase == 'answer' else '{"query":"find"}'
        return dict(message=dict(content=text), done_reason='stop',
                    prompt_eval_count=100, eval_count=10)
    result = ns['run_episode'](cfg, dict(id='task', benchmark='synthetic'),
        dict(id='q', type='synthetic', question='Describe the source.'),
        dict(id='stub', origin='synthetic'), cap, memory or Memory(),
        lambda _: (_ for _ in ()).throw(AssertionError('unexpected call')),
        schema_call=adapter)
    result.pop('seconds')
    return result, calls


class DuplicateRemedy(unittest.TestCase):
    def test_strict_default_and_v2(self):
        action = dict(answer='text', citations=['A', 'A'])
        with self.assertRaises(ValueError):
            C['accept_typed_navigation_answer'](action, ['A'])
        out, _ = episode(C, config('typed_v2'), json.dumps(action))
        self.assertFalse(out['terminated'])
        self.assertIn('CITATION_REFUSAL', out['errors'][0])

    def test_v3_raw_bytes_order_and_literal_text(self):
        raw = ' {"citations" : ["B", "A", "B"], "answer" : "text"}  '
        out, calls = episode(C, config(), raw, cap=2)
        self.assertTrue(out['terminated'], out['errors'])
        self.assertEqual(out['native_final_response'].encode(), raw.encode())
        self.assertEqual(out['trace'][-1]['raw_text'], raw)
        self.assertEqual(out['trace'][-1]['response']['message']['content'], raw)
        self.assertEqual(out['native_citations'], ['B', 'A', 'B'])
        self.assertEqual(out['answer'], 'text')
        self.assertEqual(out['citation_duplicate_count'], 1)
        self.assertEqual(out['trace'][-1]['citation_duplicate_count'], 1)
        instruction = calls[-1][0][-1]['content']
        self.assertNotIn('"answer":"text"', instruction)
        self.assertIn('own substantive final', instruction)

    def test_visible_ids_malformed_and_caps(self):
        for action in [dict(answer='ok', citations=['foreign', 'foreign']),
                       dict(answer='ok', citations=['evicted']),
                       dict(answer='ok', citations=[1]),
                       dict(answer='ok', citations='A'),
                       dict(answer='ok', citations=['A'] * 65),
                       dict(answer=1, citations=['A']),
                       dict(answer='ok', citations=['A'], extra=True),
                       dict(answer=' ', citations=['A'])]:
            with self.subTest(action=action), self.assertRaises(ValueError):
                C['accept_typed_navigation_answer'](action, ['A'], allow_duplicates=True)
        action = dict(answer='ok', citations=['A'] * 64)
        before = json.dumps(action).encode()
        C['accept_typed_navigation_answer'](action, ['A'], allow_duplicates=True)
        self.assertEqual(json.dumps(action).encode(), before)

    def test_abstention(self):
        for action in [dict(answer='ok', citations=[]),
                       dict(answer='UNRESOLVED', citations=['A']),
                       dict(answer=' UNRESOLVED', citations=[])]:
            with self.assertRaises(ValueError):
                C['accept_typed_navigation_answer'](action, ['A'], allow_duplicates=True)
        out, _ = episode(C, config(), '{"answer":"UNRESOLVED","citations":[]}', cap=0)
        self.assertTrue(out['terminated'])
        self.assertEqual(out['native_citations'], [])
        self.assertEqual(out['citation_duplicate_count'], 0)

    def test_schema_profiles_and_config(self):
        for profile in ['typed_v2', 'typed_v3']:
            cfg = config(profile)
            self.assertTrue(C['navigation_enabled'](cfg))
            local = dict(C, load=lambda _: cfg)
            exec(compile(ast.Module(body=[next(n for n in ast.parse(
                HERE.with_name('dogfood.py').read_text(encoding='utf-8')).body
                if isinstance(n, ast.FunctionDef) and n.name == 'config_at')],
                type_ignores=[]), '<config>', 'exec'), local)
            self.assertEqual(local['config_at']('unused'), cfg)
            local['load'] = lambda _: dict(cfg, explicit_references=True)
            self.assertTrue(local['config_at']('unused')['explicit_references'])
            local['load'] = lambda _: dict(cfg, explicit_references="invalid")
            with self.assertRaisesRegex(ValueError, "explicit_references must be boolean"):
                local['config_at']('unused')
            for change in [dict(source_navigation=False), dict(protocol='v1'),
                           dict(dependency_evidence=True)]:
                with self.assertRaises(ValueError):
                    C['navigation_enabled'](dict(cfg, **change))
            for phase in ['query', 'choose', 'answer']:
                schema = C['action_schema'](cfg, phase, visible_ids=['A'],
                                           navigation_addresses=['source:A'])
                if phase == 'answer':
                    citations = schema['properties']['citations']
                    self.assertEqual(citations['items']['enum'], ['A'])
                    self.assertEqual(citations.get('uniqueItems'),
                                     True if profile == 'typed_v2' else None)
                    self.assertEqual(citations['maxItems'], 1 if profile == 'typed_v2' else 64)
            empty = C['action_schema'](cfg, 'answer', visible_ids=[], navigation_addresses=[])
            self.assertEqual(empty['properties']['citations']['maxItems'], 0)
        with self.assertRaises(ValueError):
            C['navigation_enabled'](config('unknown'))

    def test_v2_forced_message_unchanged(self):
        out, calls = episode(C, config('typed_v2'), '{"answer":"ok","citations":["A"]}')
        self.assertTrue(out['terminated'])
        self.assertEqual(out['native_final_response'], '{"answer":"ok","citations":["A"]}')
        self.assertNotIn('citation_duplicate_count', out)
        self.assertEqual(calls[-1][0][-1]['content'],
                         'Return only your final {"answer":"text","citations":["source_id"]} JSON now.')

    def test_eviction_refuses_old_id(self):
        cfg = config()
        cfg.update(num_ctx=2048, max_output_tokens=128)
        out, _ = episode(C, cfg, '{"answer":"ok","citations":["A","A"]}',
                         cap=4, memory=Memory(500))
        self.assertTrue(out['context_evictions'])
        self.assertNotIn('A', out['trace'][-1]['final_visible_source_ids'])
        self.assertFalse(out['terminated'])
        self.assertIn('CITATION_REFUSAL', out['errors'][0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
