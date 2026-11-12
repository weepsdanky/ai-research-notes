"""Incremental component oracles, not MTP training or accepted-length scores."""
import os,unittest
import torch
from model import rope
from mtp import MultiTokenPredictor
from mtp_inference import IncrementalMTP,pack_selection,unpack_selection
from test_recipe import tiny


class MTPInferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91);self.device=os.environ.get('LAB_TEST_DEVICE','cpu')
        self.m=MultiTokenPredictor(tiny()).to(self.device).eval()
        self.e=torch.randn(2,21,self.m.c.width,device=self.device)
        self.h=torch.randn(2,21,4*self.m.c.width,device=self.device)
        self.assertEqual(self.e.device.type,self.device)

    @torch.no_grad()
    def test_dense_and_sparse_chunk_cache_match_full(self):
        for stage in ('sparse','pretrain'):
            with self.subTest(stage=stage):
                runner=IncrementalMTP(self.m,stage);cache={};parts=[];multi=[];pos=0
                for count in (3,5,1,12):
                    out,residual,_=runner(self.e[:,pos:pos+count],self.h[:,pos:pos+count],cache)
                    parts.append(out);multi.append(residual);pos+=count
                expected=self.m(self.e,self.h,stage)
                torch.testing.assert_close(torch.cat(parts,1),expected[0],atol=3e-5,rtol=3e-4)
                torch.testing.assert_close(torch.cat(multi,1),expected[1],atol=3e-5,rtol=3e-4)

    @torch.no_grad()
    def test_frozen_rows_still_update_index_and_main_caches(self):
        m=self.m;c=m.c;runner=IncrementalMTP(m);cache={};_,_,packed=runner(self.e[:,:7],self.h[:,:7],cache);history=unpack_selection(packed,c.budget+c.block-1,torch.arange(7,device=self.device));frozen=cache['selection'].clone();calls=[]
        hook=m.layer.mix.indexer.qk.register_forward_hook(lambda *args:calls.append(1))
        for pos in range(7,10):
            actual,multi,packed=runner(self.e[:,pos:pos+1],self.h[:,pos:pos+1],cache,reuse=True)
            self.assertTrue(torch.equal(packed[:,0],frozen))
            tokens=unpack_selection(packed,c.budget+c.block-1,torch.tensor([pos],device=self.device));history=torch.cat((history,tokens),1)
            reference=m(self.e[:,:pos+1],self.h[:,:pos+1],reuse_tokens=history)
            torch.testing.assert_close(actual,reference[0][:,-1:],atol=3e-5,rtol=3e-4)
            torch.testing.assert_close(multi,reference[1][:,-1:],atol=3e-5,rtol=3e-4)
            self.assertEqual(cache['raw'].shape[1],pos+1);self.assertEqual(cache['pooled'].shape[1],(pos+1)//c.block)
        hook.remove();self.assertEqual(len(calls),3)
        x,_=m.layer.mix_residual(m.fusion(self.e[:,:10],self.h[:,:10]));expected=m.layer.mix.indexer.qk(x)[...,-c.index_dim:]
        torch.testing.assert_close(cache['raw'],expected,atol=2e-6,rtol=2e-5)

    @torch.no_grad()
    def test_rollback_refill_and_request_compaction(self):
        runner=IncrementalMTP(self.m);cache={};runner(self.e[:,:13],self.h[:,:13],cache);runner.rollback(cache,7)
        with self.assertRaises(ValueError):runner(self.e[:,7:8],self.h[:,7:8],cache,reuse=True)
        self.assertEqual(cache['k'].shape[1],7)
        expected=self.m(self.e[:,:17],self.h[:,:17])
        actual,multi,_=runner(self.e[:,7:17],self.h[:,7:17],cache)
        torch.testing.assert_close(actual,expected[0][:,7:],atol=3e-5,rtol=3e-4)
        torch.testing.assert_close(multi,expected[1][:,7:],atol=3e-5,rtol=3e-4)
        runner.compact_requests(cache,torch.tensor([1],device=self.device));out,_,_=runner(self.e[1:2,17:18],self.h[1:2,17:18],cache)
        expected=self.m(self.e[1:2,:18],self.h[1:2,:18]);torch.testing.assert_close(out,expected[0][:,-1:],atol=3e-5,rtol=3e-4)

    def test_packed_count_is_not_token_and_invalid_rows_rejected(self):
        tokens=torch.tensor([[[3,-1,1,-1,2]]],device=self.device);packed=pack_selection(tokens,7)
        self.assertEqual(packed[0,0,-1].item(),3);self.assertEqual(packed[0,0,:3].tolist(),[3,1,2])
        unpacked=unpack_selection(packed,7,torch.tensor([3],device=self.device));self.assertEqual(unpacked[0,0,-1].item(),-1)
        malformed=packed.clone();malformed[0,0,-1]=8
        with self.assertRaises(ValueError):unpack_selection(malformed,7,torch.tensor([3],device=self.device))
        with self.assertRaises(ValueError):unpack_selection(packed,7,torch.tensor([2],device=self.device))


if __name__=='__main__':unittest.main()
