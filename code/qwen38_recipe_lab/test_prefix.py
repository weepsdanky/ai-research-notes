import os
import unittest
import torch
from model import Model
from prefill_inference import PrefillInference
from prefix_inference import PrefixInference,common_prefix,clone_batch
from test_recipe import tiny


class PrefixTests(unittest.TestCase):
    def test_prefix_cache_batch_expansion_state_and_seed_immutability(self):
        torch.manual_seed(223);device=os.environ.get('LAB_TEST_DEVICE','cpu')
        model=Model(tiny(heads=12,head_dim=256,rotary_dim=64,index_dim=128,gdn_k_dim=128,gdn_v_dim=128)).to(device).eval()
        ids=torch.randint(0,model.config.vocab_size,(2,273),device=device);ids[1,:256]=ids[0,:256];ids[:,125]=model.config.eos
        prefix=common_prefix(ids.tolist(),128);self.assertEqual(len(prefix),256)
        for stage in ('pretrain','sparse'):
            runner=PrefillInference(model,stage,prefill_chunk=128);cached=PrefixInference(runner,prefix)
            original=clone_batch(cached.seed,1);a={};b={}
            left=runner(ids,a);right=cached(ids,b)
            torch.testing.assert_close(left,right,atol=5e-5,rtol=5e-4)
            for _ in range(16):
                self.assertTrue(torch.equal(left.argmax(-1),right.argmax(-1)))
                token=left.argmax(-1)[:,None];left=runner(token,a);right=cached(token,b)
                torch.testing.assert_close(left,right,atol=5e-5,rtol=5e-4)
            def same(x,y):
                if isinstance(x,dict):
                    self.assertEqual(x.keys(),y.keys())
                    for key in x:same(x[key],y[key])
                else:self.assertTrue(torch.equal(x,y))
            same(original,cached.seed)
            # Reusing a seed again must not inherit the prior continuation.
            c={};again=cached(ids,c);torch.testing.assert_close(again,runner(ids,{}),atol=5e-5,rtol=5e-4)
            wrong=ids.clone();wrong[0,0]=(wrong[0,0]+1)%model.config.vocab_size
            with self.assertRaises(ValueError):cached(wrong,{})
