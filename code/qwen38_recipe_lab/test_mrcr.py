"""No model calls: author scoring, actual released chat format and source controls."""
import unittest
from unittest.mock import patch
from pathlib import Path
from jinja2.exceptions import TemplateError
from mrcr_tasks import render,author_grade,locked_text,validate_source_row


class MRCRTests(unittest.TestCase):
    def test_author_prefix_and_sequence_ratio_without_whitespace_repair(self):
        grade=author_grade()
        self.assertEqual(grade('hashabc','hashabc','hash'),1.)
        self.assertEqual(grade('abchash','hashabc','hash'),0.)
        self.assertEqual(grade(' hashabc','hashabc','hash'),0.)
        self.assertEqual(grade('hashab','hashabc','hash'),.8)
        self.assertEqual(grade('hash','hashabc','hash'),0.)
    def test_released_single_user_non_thinking_generation_prefix(self):
        self.assertEqual(render([dict(role='user',content=' hello ')]),
                         '<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    def test_multiround_assistant_content_and_invalid_roles(self):
        text=render([dict(role='user',content='first'),dict(role='assistant',content='reply'),dict(role='user',content='again')])
        self.assertIn('<|im_start|>assistant\n<think>\n\n</think>\n\nreply<|im_end|>\n',text)
        self.assertEqual(text.count('<|im_start|>user\n'),2)
        self.assertLess(text.index('first'),text.index('reply'));self.assertLess(text.index('reply'),text.index('again'))
        with self.assertRaises(TemplateError):render([dict(role='assistant',content='no user query')])
    def test_source_answer_binds_to_exact_assistant_and_eight_requests(self):
        messages=[m for i in range(8) for m in (dict(role='user',content='same request'),dict(role='assistant',content=f'answer{i}'))]+[dict(role='user',content='return instance4')]
        row=dict(n_needles=8,desired_msg_index=6,answer='hashanswer3',random_string_to_prepend='hash')
        validate_source_row(row,messages)
        with self.assertRaises(AssertionError):validate_source_row(dict(row,answer='hashanswer4'),messages)
        with self.assertRaises(AssertionError):validate_source_row(row,messages[:-3]+messages[-1:])
    def test_locked_source_drift_rejected(self):
        original=Path.read_bytes
        def changed(path):
            data=original(path)
            return data+b' ' if path.name=='mrcr-readme.md' else data
        with patch.object(Path,'read_bytes',changed):
            with self.assertRaises(AssertionError):locked_text('mrcr-readme.md')


if __name__=='__main__':unittest.main()
