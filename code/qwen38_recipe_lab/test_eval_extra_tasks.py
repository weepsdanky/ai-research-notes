import unittest
from pathlib import Path
from eval_extra_tasks import load_tasks, boxed, correct, include_examples


class ExtraTasksTests(unittest.TestCase):
    def test_full_categories_and_public_rationales(self):
        tasks = [t for t in load_tasks() if t.name.startswith('mmlu-pro-')]
        self.assertEqual(len(tasks), 14)
        self.assertEqual(sum(len(t.rows) for t in tasks), 12032)
        for task in tasks:
            self.assertEqual((task.shots, task.cot, task.kind), (5, True, 'choice'))
            prompt, gold = task.rows[0]
            self.assertEqual(prompt.count('Question: '), 6)
            self.assertGreaterEqual(prompt.count("Let's think step by step."), 6)
            self.assertIn(gold, 'ABCDEFGHIJ')

    def test_complete_additional_coverage_and_shot_boundaries(self):
        tasks = list(load_tasks())
        self.assertEqual(sum(len(t.rows) for t in tasks if t.name.startswith('math-')), 5000)
        redux = [t for t in tasks if t.name.startswith('redux2-ok-')]
        self.assertEqual(len(redux), 57)
        self.assertEqual(sum(len(t.rows) for t in redux), 5330)
        include = [t for t in tasks if t.name.startswith('include-')]
        self.assertEqual(len(include), 44)
        self.assertEqual(sum(len(t.rows) for t in include), 22639)
        supergpqa = next(t for t in tasks if t.name == 'supergpqa')
        self.assertEqual(len(supergpqa.rows), 26529)
        self.assertEqual(supergpqa.rows[0][0].count('Question:'), 6)
        self.assertEqual(include[0].rows[0][0].count('\nAnswer:'), 6)

    def test_math_nested_box_and_primary_normalization(self):
        self.assertEqual(boxed(r'First \boxed{2}, finally \boxed{\frac{1}{2}}'), r'\frac{1}{2}')
        self.assertIsNone(boxed(r'\boxed{\frac{1}{2}'))
        self.assertTrue(correct(r'Work. \boxed{0.5}', r'\frac{1}{2}', 'boxed_math'))
        self.assertTrue(correct('The answer is: 1/2', r'\frac{1}{2}', 'boxed_math'))
        self.assertTrue(correct('Answer: 2\nFinal answer: 1/2', r'\frac{1}{2}', 'boxed_math'))
        self.assertFalse(correct(r'\boxed{2}', r'\frac{1}{2}', 'boxed_math'))
        self.assertFalse(correct('', '', 'boxed_math'))

    def test_include_examples_are_disjoint_from_full_test(self):
        fallback = {}
        for path in Path('data/eval-source').glob('include-*-test-*.parquet'):
            test, examples, provenance = include_examples(path)
            norm = lambda text: ' '.join(text.split()).casefold()
            self.assertEqual(len(examples), 5)
            self.assertFalse({norm(r['question']) for r in test} & {norm(r['question']) for r in examples})
            count = sum(p['fallback'] for p in provenance)
            if count:
                fallback[path.name.split('-test-')[0]] = count
        self.assertEqual(fallback, {'include-Estonian': 5, 'include-Georgian': 2, 'include-Malayalam': 1})
