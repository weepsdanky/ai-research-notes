"""Evaluation integrity fixtures, not benchmark/model-quality evidence."""
import contextlib
import io
import json
import sys
import signal
import tempfile
import types
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
import evaluate_batched as engine
from eval_tasks import Task, correct
from model import Model
from test_recipe import tiny
from eval_code_tasks import load_tasks, stopping, completion


class BatchedTests(unittest.TestCase):
    def test_partial_batch_replay_matches_uninterrupted(self):
        torch.manual_seed(83)
        c = tiny(vocab_size=248320, eos=248044, width=8, rank=1)
        tokenizer = Tokenizer.from_file('.cache/references/tokenizer.json')
        prompts = ['hello '*20+s for s in ['hello world', 'hello there', 'good morning', 'good night']]
        self.assertEqual(len({len(tokenizer.encode(s).ids) for s in prompts}), 1)
        with tempfile.TemporaryDirectory(prefix='batch-unit-', dir='runs') as directory:
            root = Path(directory);checkpoint = root/'fixture.pt'
            torch.save(dict(model=Model(c).state_dict(), config=asdict(c), args=dict(stage='sparse', steps=1, seq=8), step=1,
                            source_hashes={name: engine.digest(name) for name in ('model.py', 'optim.py', 'polar_coefficients.json', 'train.py')}), checkpoint)
            adapter = types.SimpleNamespace(__file__=__file__, correct=correct,
                load_tasks=lambda: [Task('fixture', 0, False, 'choice', [(s, 'A') for s in prompts])])
            actual_import = engine.importlib.import_module

            def run(out, resume=False, prefix=False, pressure=0):
                argv = ['evaluate_batched.py', str(checkpoint), '--out', str(out), '--tasks', 'fixture', '--full',
                        '--tokens', '4', '--context', '64', '--batch-size', '4', '--prefill-chunk', '16','--stop-pressure',str(pressure)]
                if resume:argv.append('--resume')
                if prefix:argv.append('--prefix-cache')
                with patch.object(sys, 'argv', argv), patch.object(engine.importlib, 'import_module',
                        side_effect=lambda name: adapter if name == 'eval_tasks' else actual_import(name)), contextlib.redirect_stdout(io.StringIO()):
                    engine.main()

            full = root/'full.jsonl';resumed = root/'resumed.jsonl';run(full)
            baseline = full.read_text().splitlines();self.assertEqual(len(baseline), 4)
            cached = root/'cached.jsonl';run(cached, prefix=True)
            lines = cached.read_text().splitlines()
            self.assertEqual([json.loads(s)['output_ids'] for s in lines], [json.loads(s)['output_ids'] for s in baseline])
            # Simulate interruption after one durable item of the four-item batch.
            resumed.write_text(lines[0]+'\n')
            meta = json.loads(Path(str(cached)+'.meta.json').read_text());meta['status'] = 'time_budget'
            Path(str(resumed)+'.meta.json').write_text(json.dumps(meta))
            run(resumed, True, prefix=True)
            expected = [json.loads(s) for s in lines];actual = [json.loads(s) for s in resumed.read_text().splitlines()]
            self.assertEqual([(r['index'], r['output_ids'], r['prediction']) for r in actual],
                             [(r['index'], r['output_ids'], r['prediction']) for r in expected])
            final = json.loads(Path(str(resumed)+'.meta.json').read_text())
            self.assertTrue(final['full_dataset_generated'])
            self.assertEqual(final['completed_items'], 4)
            adapter.load_tasks=lambda:[Task('fixture',0,False,'choice',[(s,'A') for s in prompts*2])]
            guarded=root/'guarded.jsonl';control=root/'control8.jsonl'
            with patch.object(engine,'system_pressure',return_value=1):run(control,pressure=4)
            with patch.object(engine,'system_pressure',return_value=4):run(guarded,pressure=4)
            stopped=json.loads(Path(str(guarded)+'.meta.json').read_text())
            self.assertEqual((stopped['status'],stopped['completed_items']),('memory_pressure',4))
            with patch.object(engine,'system_pressure',return_value=1):run(guarded,True,pressure=4)
            with guarded.open() as stream:actual=[json.loads(s)['output_ids'] for s in stream]
            with control.open() as stream:expected=[json.loads(s)['output_ids'] for s in stream]
            self.assertEqual(actual,expected)
            # Two distinct subject prefixes must get distinct seeds. Mixed
            # subject batches must use the uncached path with identical IDs.
            other=[s.replace('hello ','world ',20) for s in prompts]
            for mixed in (False,True):
                pairs=list(zip(prompts,other))
                questions=[s for pair in pairs for s in pair] if mixed else prompts+other
                tags=['first','second']*4 if mixed else ['first']*4+['second']*4
                self.assertEqual(len({len(tokenizer.encode(s).ids) for s in questions}),1)
                def tasks():
                    task=Task('fixture',0,False,'choice',[(s,'A') for s in questions]);task.prefix_groups=tags
                    return [task]
                adapter.load_tasks=tasks
                cold=root/f'groups-{mixed}-cold.jsonl';cached=root/f'groups-{mixed}-cached.jsonl'
                run(cold);run(cached,prefix=True)
                with cold.open() as stream:expected=[json.loads(s)['output_ids'] for s in stream]
                with cached.open() as stream:actual=[json.loads(s)['output_ids'] for s in stream]
                self.assertEqual(actual,expected)
                plans=json.loads(Path(str(cached)+'.meta.json').read_text())['identity']['prefix_cache']['plans']
                self.assertEqual(set(plans),{'fixture/first','fixture/second'})
                self.assertNotEqual(plans['fixture/first']['sha256'],plans['fixture/second']['sha256'])
            # A changed source input must be rejected, not silently appended.
            adapter.load_tasks = lambda: [Task('fixture', 0, False, 'choice', [('changed question', 'A')])]
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):run(resumed, True, prefix=True)

    def test_signal_requests_are_deferred_and_handlers_restored(self):
        previous=signal.getsignal(signal.SIGINT)
        with engine.stop_requests() as requested:
            signal.raise_signal(signal.SIGINT)
            self.assertEqual(requested,['SIGINT'])
        self.assertEqual(signal.getsignal(signal.SIGINT),previous)

    def test_code_coverage_and_official_completion_stops(self):
        tasks = list(load_tasks())
        self.assertEqual([(t.name, len(t.rows)) for t in tasks], [('evalplus-humaneval', 164), ('evalplus-mbpp', 378)])
        self.assertIn('\ndef ', stopping('evalplus-humaneval'))
        self.assertNotIn('\ndef ', stopping('evalplus-mbpp'))
        self.assertEqual(completion('    return 1\ndef another(): pass', stopping('evalplus-humaneval')), '    return 1')
