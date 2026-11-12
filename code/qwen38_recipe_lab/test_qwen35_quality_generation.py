"""Greedy/cache/stop integration on small controls, not downstream accuracy."""
import os
import json
from pathlib import Path
import unittest

import torch
from qwen35_controls import ControlConfig, Qwen35Control
from qwen35_inference import ControlInference
from qwen35_quality_generation import ControlGreedy
from run_dense_text_guarded import digest


class ControlGreedyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.device = os.environ.get('TEST_DEVICE', 'cpu')
        self.length = 131 if self.device == 'cpu' else 19
        self.tokens = 8 if self.device == 'cpu' else 4

    def fixture(self, arm):
        c = ControlConfig(vocab_size=97, width=16, heads=8, head_dim=16,
                          rotary_dim=4, gdn_k_dim=16, gdn_v_dim=16,
                          hidden=8, experts=4, top_k=2)
        model = Qwen35Control(c, arm).to(self.device).eval()
        engine = ControlInference(model, query_chunk=16, max_context=256)
        return model, ControlGreedy(engine, eos=None, prefill_chunk=11, prefix_slots=2)

    @torch.no_grad()
    def teacher(self, model, prompts, count):
        ids = torch.tensor(prompts, device=self.device)
        output = []
        for _ in range(count):
            hidden, _ = model.hidden(ids)
            tokens = model.head(hidden[:, -1]).argmax(-1)
            output.append(tokens.cpu())
            ids = torch.cat((ids, tokens[:, None]), -1)
        return torch.stack(output, 1).tolist()

    def prompts(self):
        generator = torch.Generator().manual_seed(73)
        prompts = torch.randint(0, 97, (2, self.length), generator=generator).tolist()
        n = self.length // 2
        prompts[1][:n] = prompts[0][:n]
        return prompts, prompts[0][:n]

    def test_prefix_batch_and_reuse_equal_full_teacher(self):
        prompts, prefix = self.prompts()
        for arm in ['full', 'swa', 'gdn']:
            with self.subTest(arm=arm):
                model, runner = self.fixture(arm)
                expected = self.teacher(model, prompts, self.tokens)
                actual, reasons = runner.generate(prompts, self.tokens)
                self.assertEqual(actual, expected)
                self.assertEqual(reasons, ['max_new_tokens'] * 2)
                actual, _ = runner.generate(prompts, self.tokens, prefix=prefix)
                self.assertEqual(actual, expected)
                cached = runner.prefixes[tuple(prefix)]
                before = [{k: v.clone() for k, v in layer.items()} for layer in cached[1]['layers']]
                actual, _ = runner.generate(prompts, self.tokens, prefix=prefix)
                self.assertEqual(actual, expected)
                for layer, snapshot in zip(cached[1]['layers'], before):
                    for key in snapshot:self.assertTrue(torch.equal(layer[key], snapshot[key]))
                self.assertEqual(cached[1]['lengths'].tolist(), [len(prefix)])
                # CPU length131 crosses SWA128; fixture tests the actual window.
                if arm == 'swa' and self.device == 'cpu':
                    self.assertTrue(all(len(row) == 8 for row in actual))

    def test_whole_prompt_prefix_and_bounded_lru(self):
        _, runner = self.fixture('full')
        prompts = [[1, 2, 3], [1, 2, 3]]
        plain, _ = runner.generate(prompts, 3)
        cached, _ = runner.generate(prompts, 3, prefix=prompts[0])
        self.assertEqual(plain, cached)
        runner.generate([[2, 3, 4]], 2, prefix=[2])
        runner.generate([[3, 4, 5]], 2, prefix=[3])
        self.assertEqual(list(runner.prefixes), [(2,), (3,)])

    def test_batch_eight_all_arms_against_full_teacher(self):
        rng = torch.Generator().manual_seed(901)
        prompts = torch.randint(0, 97, (8, 19), generator=rng).tolist()
        for row in prompts:row[:7] = prompts[0][:7]
        for arm in ['full', 'swa', 'gdn']:
            with self.subTest(arm=arm):
                model, runner = self.fixture(arm)
                expected = self.teacher(model, prompts, 3)
                actual, reasons = runner.generate(prompts, 3, prefix=prompts[0][:7])
                self.assertEqual(actual, expected)
                self.assertEqual(reasons, ['max_new_tokens'] * 8)

    def test_eos_and_row_specific_stop_callback(self):
        _, runner = self.fixture('full')
        prompts = [[1, 2, 3], [1, 2, 3]]
        reference, _ = runner.generate(prompts, 3)
        runner.eos = reference[0][0]
        actual, reasons = runner.generate(prompts, 3)
        self.assertEqual(actual, [[runner.eos], [runner.eos]])
        self.assertEqual(reasons, ['eos', 'eos'])
        runner.eos = None
        actual, reasons = runner.generate(prompts, 3, stopping=lambda row, ids: len(ids) == row + 1)
        self.assertEqual(actual, [reference[0][:1], reference[1][:2]])
        self.assertEqual(reasons, ['stop_string', 'stop_string'])

    def test_refuse_padding_capacity_and_prefix_drift(self):
        _, runner = self.fixture('full')
        for prompts, tokens, prefix in [([], 1, ()), ([[]], 1, ()), ([[1], [1, 2]], 1, ()),
                                        ([[1, 2]], 255, ()), ([[1, 2]], 1, (2,)),
                                        ([[1, 2]], 1, (1, 2, 3)), ([[97]], 1, ())]:
            with self.subTest(prompts=prompts, tokens=tokens, prefix=prefix):
                with self.assertRaises(ValueError):runner.generate(prompts, tokens, prefix=prefix)
        self.assertEqual(len(runner.prefixes), 0)

    def test_selected_trained_checkpoints_cpu_integration(self):
        manifest = os.environ.get('QWEN35_QUAL_MANIFEST')
        if self.device != 'cpu' or manifest is None:
            self.skipTest('Opt-in actual three-weight CPU integration')
        from tokenizers import Tokenizer
        spec = json.loads(Path('qwen35_tokenizer_sources.lock.json').read_text())[0]
        self.assertEqual(digest(spec['file']), spec['sha256'])
        tokenizer = Tokenizer.from_file(spec['file'])
        pool = tokenizer.encode(('ROMEO: The world is full of words.\nScience explains observations with evidence.\n' * 100), add_special_tokens=False).ids
        prompts = [pool[:130] + [pool[131 + row]] for row in range(8)]
        prefix = prompts[0][:65]
        result = dict(status='running',scope='Actual selected80+2 feasibility weights, CPU B8/131-token4-token greedy+prefix integration only; not formal800/20 or benchmark quality',checks=[])
        for entry in json.loads(Path(manifest).read_text())['selected']:
            with self.subTest(arm=entry['arm']):
                self.assertEqual(digest(entry['checkpoint']), entry['checkpoint_sha256'])
                self.assertEqual(digest(entry['audit']), entry['audit_sha256'])
                proof = json.loads(Path(entry['audit']).read_text())
                self.assertTrue(proof['phase_complete'])
                self.assertEqual(proof['checkpoint_sha256'], entry['checkpoint_sha256'])
                saved = torch.load(entry['checkpoint'], weights_only=True, map_location='cpu', mmap=True)
                self.assertEqual(saved['profile_sha256'], proof['profile_sha256'])
                self.assertTrue(all(digest(p) == sha for p, sha in saved['source_hashes'].items()))
                with torch.device('meta'):
                    model = Qwen35Control(ControlConfig(**saved['config']), saved['arm'], seed=saved['profile']['seed'])
                model.load_state_dict(saved['model'], strict=True, assign=True)
                model.eval();del saved
                engine = ControlInference(model, query_chunk=16, max_context=256)
                runner = ControlGreedy(engine, eos=248044, prefill_chunk=31)
                expected = self.teacher(model, prompts, 4)
                actual, reasons = runner.generate(prompts, 4, prefix=prefix)
                self.assertEqual(actual, expected)
                replay, _ = runner.generate(prompts, 4, prefix=prefix)
                self.assertEqual(actual, replay)
                self.assertEqual(reasons, ['max_new_tokens'] * 8)
                self.assertEqual(digest(entry['checkpoint']), entry['checkpoint_sha256'])
                result['checks'].append(dict(arm=entry['arm'],checkpoint=entry['checkpoint'],checkpoint_sha256=entry['checkpoint_sha256'],audit_sha256=entry['audit_sha256'],input_ids=prompts,output_ids=actual,raw_predictions=[tokenizer.decode(row, skip_special_tokens=False) for row in actual],status='passed'))
                del model, engine, runner
        self.assertEqual(len(result['checks']), 3)
        result['status'] = 'passed'
        path = Path(os.environ['QWEN35_QUAL_PROBE_OUT'])
        self.assertFalse(path.exists())
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':unittest.main()
