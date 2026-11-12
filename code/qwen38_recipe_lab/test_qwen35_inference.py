"""Teacher-oracle checks for chunked, ragged and forked control caches."""
import os,unittest
import torch
from model import rope
from qwen35_controls import ControlConfig,Qwen35Control,GDN35
from qwen35_inference import ControlInference,batch_rope


def config(**kw):
    values=dict(vocab_size=32,width=8,heads=8,kv_heads=1,head_dim=8,rotary_dim=2,gdn_k_dim=8,gdn_v_dim=8,hidden=4,experts=4,top_k=2,query_chunk=17,scan_chunk=17,activation_checkpoint=False);values.update(kw);return ControlConfig(**values)


class InferenceTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(498);self.device=os.environ.get('LAB_TEST_DEVICE','cpu')
    def model(self,arm,c=None):return Qwen35Control(c or config(),arm).to(self.device).eval()
    def test_batch_rope_matches_original_positions(self):
        x=torch.randn(2,13,8,8,device=self.device);pos=torch.arange(29,42,device=self.device)
        actual=batch_rope(x,pos[None].expand(2,-1),2,1e7)
        torch.testing.assert_close(actual,rope(x,pos,2,1e7),atol=1e-6,rtol=1e-6)
    def test_chunked_prefill_and_greedy_match_teacher_all_arms(self):
        for arm in ['full','swa','gdn']:
            with self.subTest(arm=arm):
                model=self.model(arm);engine=ControlInference(model,query_chunk=7,max_context=512);ids=torch.randint(0,32,(1,259),device=self.device)
                logits,cache=engine.prefill(ids,chunk=31)
                with torch.no_grad():expected=model(ids)[0][:,-1]
                torch.testing.assert_close(logits,expected,atol=2e-5,rtol=2e-4)
                for _ in range(5):
                    token=expected.argmax(-1)[:,None];self.assertTrue(torch.equal(logits.argmax(-1),token[:,0]));ids=torch.cat((ids,token),-1)
                    logits=engine.forward(token,cache)
                    with torch.no_grad():expected=model(ids)[0][:,-1]
                    torch.testing.assert_close(logits,expected,atol=2e-5,rtol=2e-4)
                if arm=='swa':
                    for layer,state in zip(model.layers,cache['layers'],strict=True):
                        if getattr(layer.mixer,'window',None) is not None:self.assertEqual(state['k'].shape[-1],127)
    def test_ragged_decode_and_prefix_fork_isolation(self):
        for arm in ['full','swa','gdn']:
            with self.subTest(arm=arm):
                model=self.model(arm);engine=ControlInference(model,query_chunk=11,max_context=512);prompts=[torch.randint(0,32,(1,n),device=self.device) for n in [3,131,259]]
                caches=[]
                for prompt in prompts:_,cache=engine.prefill(prompt,chunk=43);caches.append(cache)
                snapshots=[{key:value.clone() for key,value in layer.items()} for layer in caches[0]['layers']]
                branch=engine.fork(caches[0]);engine.forward(torch.tensor([[7]],device=self.device),branch)
                for original,snapshot in zip(caches[0]['layers'],snapshots,strict=True):
                    for key in snapshot:self.assertTrue(torch.equal(original[key],snapshot[key]),key)
                packed=engine.pack(caches)
                for step in range(3):
                    tokens=torch.tensor([[step+1],[step+2],[step+3]],device=self.device);actual=engine.forward(tokens,packed)
                    for row in range(3):
                        prompts[row]=torch.cat((prompts[row],tokens[row:row+1]),-1)
                        with torch.no_grad():expected=model(prompts[row])[0][:,-1]
                        torch.testing.assert_close(actual[row:row+1],expected,atol=2e-5,rtol=2e-4)
    def test_multiple_kv_groups_and_capacity(self):
        model=self.model('swa',config(heads=16,kv_heads=2,window=1));engine=ControlInference(model,max_context=20);ids=torch.randint(0,32,(1,17),device=self.device);actual,cache=engine.prefill(ids,chunk=5)
        with torch.no_grad():expected=model(ids)[0][:,-1]
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4)
        for layer,state in zip(model.layers,cache['layers'],strict=True):
            if getattr(layer.mixer,'window',None) is not None:self.assertEqual(state['k'].shape[-1],0)
        with self.assertRaises(AssertionError):engine.forward(torch.zeros(1,4,dtype=torch.long,device=self.device),cache)


if __name__=='__main__':unittest.main()
