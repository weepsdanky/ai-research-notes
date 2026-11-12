"""Validate the byte proof against actual tokenizer IDs and reject unsafe specs."""
import copy,json,unittest
from pathlib import Path
from tokenizers import Tokenizer
from token_capacity import CapacityBound

class CapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads(Path('.cache/references/tokenizer.json').read_text())
        cls.bound=CapacityBound(cls.config);cls.codec=Tokenizer.from_file('.cache/references/tokenizer.json')
    def test_nfc_added_boundaries_and_byte_coverage_match_real_encoder(self):
        cases=['abc '*1000,' '*50000,'e\u0301'*500,'😀 漢字 Ελληνικά\n'*50,'<|im_start|>\u0338e\u0301<|im_end|>','<think>\n\n</think>\nAnswer:']
        for text in cases:
            encoded=self.codec.encode(text,add_special_tokens=False);lower,size=self.bound.measure(text,self.codec.normalizer)
            covered=sum(len(t.encode()) if t in self.bound.added else len(t) for t in encoded.tokens)
            self.assertEqual(size,covered);self.assertLessEqual(lower,len(encoded.ids))
            self.assertGreaterEqual(len(encoded.ids)*self.bound.maximum,size)
    def test_unsupported_normalization_or_consuming_added_tokens_rejected(self):
        for path,value in [('normalizer',None),('lstrip',True),('normalized',True),('single_word',True)]:
            config=copy.deepcopy(self.config)
            if path=='normalizer':config[path]=value
            else:config['added_tokens'][0][path]=value
            with self.assertRaises(AssertionError):CapacityBound(config)

if __name__=='__main__':unittest.main()
