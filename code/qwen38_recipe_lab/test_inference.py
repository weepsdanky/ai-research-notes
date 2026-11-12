"""Numerical proof of incremental KV against the training mathematical oracle."""
import os,unittest
import torch
from model import Model
from inference import Inference
from test_recipe import tiny


class InferenceTests(unittest.TestCase):
    def test_dense_sparse_incremental_logits_and_block_boundaries(self):
        torch.manual_seed(101); device=os.environ.get('LAB_TEST_DEVICE','cpu')
        c=tiny(heads=12,head_dim=256,rotary_dim=64,index_dim=128,gdn_k_dim=128,gdn_v_dim=128)
        model=Model(c).to(device).eval()
        ids=torch.randint(0,c.vocab_size,(2,25),device=device);ids[:,8]=c.eos
        with torch.no_grad():
            for stage in ('pretrain','sparse'):
                runner=Inference(model,stage)
                for prefix in (1,3,4,7):
                    cache={};actual=runner(ids[:,:prefix],cache)
                    torch.testing.assert_close(actual,model(ids[:,:prefix],stage)[0][:,-1],atol=3e-5,rtol=3e-4)
                    for end in range(prefix+1,26):
                        actual=runner(ids[:,end-1:end],cache)
                        expected=model(ids[:,:end],stage)[0][:,-1]
                        torch.testing.assert_close(actual,expected,atol=3e-5,rtol=3e-4)
    def test_zero_index_scores_keep_stable_ties_and_prefill_cache(self):
        torch.manual_seed(103);device=os.environ.get('LAB_TEST_DEVICE','cpu')
        c=tiny();model=Model(c).to(device).eval()
        with torch.no_grad():
            model.layers[3].mix.indexer.qk.weight.zero_()
            ids=torch.randint(0,c.vocab_size,(1,27),device=device)
            runner=Inference(model,'sparse');cache={};runner(ids[:,:11],cache)
            for end in range(12,28):
                actual=runner(ids[:,end-1:end],cache)
                torch.testing.assert_close(actual,model(ids[:,:end],'sparse')[0][:,-1],atol=3e-5,rtol=3e-4)
            with self.assertRaises(ValueError):runner(ids[:,:0],cache)

    def test_chunked_prefill_matches_full_prefix_across_partial_blocks(self):
        torch.manual_seed(107);device=os.environ.get('LAB_TEST_DEVICE','cpu')
        c=tiny(heads=12,head_dim=256,index_dim=128,rotary_dim=64,gdn_k_dim=128,gdn_v_dim=128)
        model=Model(c).to(device).eval();ids=torch.randint(0,c.vocab_size,(1,67),device=device);ids[:,8]=c.eos
        with torch.no_grad():
            for stage in ('pretrain','sparse'):
                expected=model(ids,stage)[0][:,-1]
                for chunk in (3,8,13):
                    runner=Inference(model,stage,prefill_chunk=chunk);cache={}
                    torch.testing.assert_close(runner(ids,cache),expected,atol=3e-5,rtol=3e-4)
                    more=ids[:,:3]
                    torch.testing.assert_close(runner(more,cache),model(torch.cat((ids,more),1),stage)[0][:,-1],atol=3e-5,rtol=3e-4)


if __name__=='__main__':unittest.main()
