"""Grouped KV contraction, including the cold-cache path, against old oracle."""
import os
import unittest
import torch
from model import Model
from prefill_inference import PrefillInference
from test_recipe import tiny


class GroupedGQATests(unittest.TestCase):
    def test_cold_prefill_tail_multiple_kv_heads_and_decode_state(self):
        torch.manual_seed(227);device=os.environ.get('LAB_TEST_DEVICE','cpu')
        for kv in (1,2):
            model=Model(tiny(heads=12,kv_heads=kv,head_dim=256,rotary_dim=64,index_dim=128)).to(device).eval()
            data=torch.randint(0,model.config.vocab_size,(2,145),device=device);data[:,17]=model.config.eos
            for stage in ('pretrain','sparse'):
                for length,chunk in [(1,0),(4,0),(37,31),(145,128)]:
                    legacy=PrefillInference(model,stage,prefill_chunk=chunk)
                    grouped=PrefillInference(model,stage,prefill_chunk=chunk,grouped_gqa=True)
                    a={};b={};left=legacy(data[:,:length],a);right=grouped(data[:,:length],b)
                    torch.testing.assert_close(left,right,atol=5e-5,rtol=5e-4)
                    for _ in range(8):
                        self.assertTrue(torch.equal(left.argmax(-1),right.argmax(-1)))
                        token=left.argmax(-1)[:,None];left=legacy(token,a);right=grouped(token,b)
                        torch.testing.assert_close(left,right,atol=5e-5,rtol=5e-4)
                    def same(x,y):
                        if isinstance(x,dict):
                            self.assertEqual(x.keys(),y.keys())
                            for key in x:same(x[key],y[key])
                        else:torch.testing.assert_close(x,y,atol=5e-5,rtol=5e-4)
                    same(a,b)
