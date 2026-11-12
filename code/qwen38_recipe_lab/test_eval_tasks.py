"""Coverage and nontrivial answer-extraction checks for the public core suite."""
import unittest
from eval_tasks import answer,correct,load_tasks


class EvaluationTests(unittest.TestCase):
    def test_extraction_respects_explicit_answers_and_does_not_score_prose_letters(self):
        self.assertEqual(answer('Answer: (C).\nI am sure.','choice'),'C')
        self.assertEqual(answer(' B\n\nQuestion: next','choice'),'B')
        self.assertEqual(answer('The answer is (J).','choice'),'J')
        self.assertIsNone(answer('Some A appears here. I cannot decide.','choice'))
        self.assertTrue(correct('The answer is 1,800.0','1800','number'))
        self.assertTrue(correct('So the answer is False.','False','boolean'))
        self.assertTrue(correct('So the answer is ] } ).','] } )','text'))
        self.assertFalse(correct('','0','number'))
        self.assertFalse(correct('No','Yes','yesno'))

    def test_all_subjects_subtasks_languages_and_shot_counts_are_present(self):
        tasks=load_tasks()
        for prefix,count,total,shots,cot in (('mmlu-',57,14042,5,False),('bbh-',27,6511,3,True),('mgsm-',11,2750,8,True)):
            group=[t for t in tasks if t.name.startswith(prefix)]
            self.assertEqual(len(group),count);self.assertEqual(sum(len(t.rows) for t in group),total)
            self.assertTrue(all((t.shots,t.cot)==(shots,cot) for t in group))
        self.assertEqual(len({t.name for t in tasks}),len(tasks))


if __name__=='__main__':unittest.main()
