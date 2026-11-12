"""Check saved masks and full/SWA gradients against independent dense SDPA."""
import copy,os,unittest
import torch
from torch.nn import functional as F
from model import rope
from qwen35_controls import ControlConfig,PlainAttention,causal_window_mask


class CheckpointMaskTests(unittest.TestCase):
    def test_checkpoint_recomputes_mask_and_matches_sdpa_gradients(self):
        torch.set_num_threads(1);torch.manual_seed(941);device=os.environ.get('LAB_TEST_DEVICE','cpu')
        c=ControlConfig(vocab_size=32,width=16,heads=8,kv_heads=1,head_dim=16,rotary_dim=4,gdn_k_dim=16,gdn_v_dim=16,hidden=4,experts=4,top_k=2,query_chunk=17)
        for window in [None,128]:
            with self.subTest(window=window):
                model=PlainAttention(c,window).to(device).train();reference=copy.deepcopy(model)
                x=torch.randn(1,259,16,device=device,requires_grad=True);xr=x.detach().clone().requires_grad_();saved=[]
                def pack(tensor):saved.append((tensor.dtype,tensor.numel()));return tensor
                with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):actual=model(x)
                self.assertFalse(any(dtype==torch.bool for dtype,_ in saved),'Checkpoint must not save quadratic bool masks')
                t=xr.shape[1];pos=torch.arange(t,device=device)
                q,gate=reference.q_gate(xr).reshape(1,t,c.heads,2*c.head_dim).chunk(2,-1)
                q=rope(reference.qnorm(q),pos,c.rotary_dim,c.rope_theta).transpose(1,2)
                k=rope(reference.knorm(reference.k(xr).reshape(1,t,c.kv_heads,c.head_dim)),pos,c.rotary_dim,c.rope_theta).transpose(1,2).repeat_interleave(8,1)
                v=reference.v(xr).reshape(1,t,c.kv_heads,c.head_dim).transpose(1,2).repeat_interleave(8,1)
                attended=F.scaled_dot_product_attention(q,k,v,attn_mask=causal_window_mask(pos,pos,window))
                expected=reference.out((attended.transpose(1,2)*gate.sigmoid()).flatten(-2))
                torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
                actual.square().mean().backward();expected.square().mean().backward()
                for ga,gb in [(x.grad,xr.grad),*[(p.grad,r.grad) for p,r in zip(model.parameters(),reference.parameters(),strict=True)]]:
                    torch.testing.assert_close(ga,gb,atol=2e-6,rtol=2e-4)


if __name__=='__main__':unittest.main()
