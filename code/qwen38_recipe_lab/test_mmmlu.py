import unittest
from eval_mmmlu_tasks import load_tasks, validate_selection


class MMMLUTests(unittest.TestCase):
    def test_complete_14_language_coverage_and_selection_bound(self):
        names = set();total = 0
        for task in load_tasks():
            names.add(task.name);total += len(task.rows)
            self.assertEqual(len(task.rows), 14042)
            self.assertEqual(len(task.prefix_groups),14042)
            self.assertEqual(len(set(task.prefix_groups)),57)
            self.assertEqual((task.shots, task.cot, task.kind), (5, False, 'choice'))
            self.assertEqual(task.rows[0][0].count('\nAnswer:'), 6)
            self.assertTrue(all(gold in 'ABCD' and len(gold) == 1 for _, gold in task.rows))
            validate_selection([task])
            with self.assertRaisesRegex(ValueError, 'one complete MMMLU language'):
                validate_selection([task, task])
        self.assertEqual(len(names), 14)
        self.assertEqual(total, 196588)
