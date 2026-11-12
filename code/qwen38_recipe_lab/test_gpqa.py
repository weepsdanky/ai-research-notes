"""Independent pinned-author controls; no network calls or model allocation."""
import ast
from collections import namedtuple
import csv
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
from typing import List
import unittest
from unittest.mock import patch
from tokenizers import Tokenizer
import eval_gpqa_tasks as gpqa


class Row(dict):
    __getattr__ = dict.__getitem__


class Frame:
    def __init__(self, path):
        with Path(path).open(newline='') as stream:
            self.rows = list(csv.DictReader(stream))

    def iterrows(self):
        return enumerate(Row(row) for row in self.rows)


def reference():
    data, examples = gpqa.sources()  # validates all pinned bytes before compilation
    root = Path('.cache/references/gpqa-author')
    tree = ast.parse((root/'baselines__utils.py').read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('generate_prompt_from_examples', 'chain_of_thought_prompt', 'load_examples')]
    book = ast.parse((root/'baselines__closed_book.py').read_text())
    cls = next(node for node in book.body if isinstance(node, ast.ClassDef) and node.name == 'AnswerPredictor')
    parser = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'parse_sampled_answer')
    parser.decorator_list = []
    namespace = dict(Example=namedtuple('Example', 'question choice1 choice2 choice3 choice4 correct_index'), List=List,
                     pd=SimpleNamespace(read_csv=Frame), random=random, re=gpqa.re,
                     AnswerPredictor=SimpleNamespace(LETTER_TO_INDEX=dict(zip('ABCD', range(4)))))
    exec(compile(ast.Module(body=nodes+[parser], type_ignores=[]), '<pinned GPQA pure functions>', 'exec'), namespace)
    return namespace


class GPQAControls(unittest.TestCase):
    def test_all_author_prompts_shuffle_and_gold(self):
        oracle = reference()
        for task in gpqa.load_tasks():
            mode, name = task.name.removeprefix('gpqa-').split('-')
            fixtures = gpqa.fewshot(mode)
            oracle['load_from_json'] = lambda path: {'questions': fixtures}
            expected = oracle['load_examples'](f'data/gpqa/gpqa_{name}.csv', seed=42)
            self.assertEqual(len(expected), 448 if name == 'main' else 198)
            self.assertEqual(task.shots, 5)
            for (prompt, gold), example in zip(task.rows, expected):
                self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(), hashlib.sha256(oracle['chain_of_thought_prompt'](example).encode()).hexdigest())
                self.assertEqual(gold, 'ABCD'[example.correct_index])

    def test_clean_examples_disjoint_and_only_two_replaced(self):
        data, original = gpqa.sources()
        clean = gpqa.fewshot('clean')
        tests = {gpqa.normalize(r['Question']) for name in ('main', 'diamond') for r in data[name]}
        self.assertEqual([i for i, row in enumerate(original) if gpqa.normalize(row['question']) in tests], [2, 3])
        self.assertEqual([i for i in range(5) if clean[i] != original[i]], [2, 3])
        self.assertTrue(all(gpqa.normalize(row['question']) not in tests for row in clean))
        for row in clean:
            self.assertTrue(row['explanation'].strip())
            self.assertIn(row['correct_answer'], row['choices'])

    def test_author_parser_order_and_invalid_answers(self):
        oracle = reference()['parse_sampled_answer']
        for value in ['The correct answer is (A)', 'Answer: (B)', 'answer: (C)', '(D)', 'nothing', '(E)', '(A) then answer is (D)', 'answer is (E) then (B)', 'answer is (a)', 'Answer: (B) and Answer: (C)']:
            self.assertEqual(gpqa.extract(value), oracle(value))
            for gold in 'ABCD':
                self.assertEqual(gpqa.correct(value, gold, 'gpqa_choice'), oracle(value) == gold)

    def test_source_drift_rejected(self):
        original = Path.read_bytes
        def changed(path):
            raw = original(path)
            return raw+b' ' if path.as_posix() == 'data/gpqa/gpqa_main.csv' else raw
        gpqa.sources.cache_clear()
        try:
            with patch.object(Path, 'read_bytes', changed):
                with self.assertRaisesRegex(ValueError, 'GPQA source drift'):
                    gpqa.sources()
        finally:
            gpqa.sources.cache_clear()

    def test_full_context_without_truncation(self):
        tokenizer = Tokenizer.from_file('.cache/references/tokenizer.json')
        lengths = [len(tokenizer.encode(prompt, add_special_tokens=False).ids) for task in gpqa.load_tasks() for prompt, _ in task.rows]
        self.assertEqual(len(lengths), 1292)
        self.assertLessEqual(max(lengths)+600, 8192)
        Path('runs/gpqa-context-audit.json').write_text(json.dumps(dict(items=len(lengths), max_prompt_tokens=max(lengths), new_tokens=600, context=8192, no_truncation=True), indent=2)+'\n')


if __name__ == '__main__':
    unittest.main()
