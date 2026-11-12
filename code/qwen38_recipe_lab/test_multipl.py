import unittest
from eval_multipl_tasks import datasets, load_tasks, stopping, completion


class MultiPLTests(unittest.TestCase):
    def test_full_eight_language_coverage_and_prompt_only_input(self):
        counts = {'py':161,'cpp':161,'java':158,'php':161,'ts':159,'cs':158,'sh':158,'js':161}
        self.assertEqual({lang:len(rows) for lang,rows in datasets().items()}, counts)
        self.assertEqual(sum(len(t.rows) for t in load_tasks()), 1277)
        for task in load_tasks():
            self.assertEqual((task.shots,task.cot,task.kind),(0,False,'code_pending'))
            rows=datasets()[task.name.removeprefix('multipl-')]
            for (prompt,gold),row in zip(task.rows,rows):
                self.assertEqual((prompt,gold),(row['prompt'],row['name']))
                self.assertNotIn('### Canonical solution below ###',prompt)
            self.assertEqual(completion('body'+stopping(task.name)[0]+'later',stopping(task.name)), 'body')
