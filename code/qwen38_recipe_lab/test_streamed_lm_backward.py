"""Independent full-projection CE and backbone/optimizer equivalence checks."""
import os,unittest
import torch
from torch.nn import functional as F
from streamed_lm_backward import streamed_lm_backward
from chunked_lm_loss import chunked_cross_entropy
from qwen35_controls import ControlConfig
from qwen35_layer_checkpoint import LayerCheckpointControl
from train_qwen35_controls import balance_loss
from optim import RecipeOptimizer,ownership


class StreamedBackwardTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(290);self.device=os.environ.get('LAB_TEST_DEVICE','cpu')
    def test_uneven_noncontiguous_loss_and_both_gradients(self):
        x=torch.randn(2,7,11,device=self.device).transpose(0,1).requires_grad_();w=torch.randn(97,11,device=self.device,requires_grad=True);y=torch.randint(0,97,(7,2),device=self.device)
        expected=F.cross_entropy(F.linear(x,w).reshape(-1,97),y.reshape(-1));expected.backward();gx=x.grad.clone();gw=w.grad.clone()
        for chunk in [1,4,100]:
            w.grad=None;actual,gradient=streamed_lm_backward(x,w,y,chunk)
            torch.testing.assert_close(actual,expected.detach(),atol=3e-6,rtol=3e-6)
            torch.testing.assert_close(gradient,gx,atol=3e-6,rtol=2e-5)
            torch.testing.assert_close(w.grad,gw,atol=3e-6,rtol=2e-5)
    def test_all_arms_backbone_and_cpu_recipe_update(self):
        c=ControlConfig(vocab_size=32,width=8,heads=2,kv_heads=1,head_dim=8,rotary_dim=2,gdn_k_dim=8,gdn_v_dim=8,hidden=4,experts=4,top_k=2,query_chunk=17,scan_chunk=17)
        ids=torch.randint(0,32,(1,129),device=self.device);labels=torch.randint(0,32,(1,129),device=self.device)
        for arm in ['full','swa','gdn']:
            with self.subTest(arm=arm):
                a=LayerCheckpointControl(c,arm).to(self.device).train();b=LayerCheckpointControl(c,arm).to(self.device).train();b.load_state_dict(a.state_dict())
                ha,ra=a.hidden(ids);hb,rb=b.hidden(ids);ba=balance_loss(ra,c.experts,c.top_k);bb=balance_loss(rb,c.experts,c.top_k)
                la,g=streamed_lm_backward(ha,a.head.weight,labels,7);lb=chunked_cross_entropy(hb,b.head.weight,labels,7)
                torch.autograd.backward((ha,ba),(g,ba.new_tensor(.001)));(lb+.001*bb).backward()
                torch.testing.assert_close(la,lb.detach(),atol=3e-6,rtol=3e-6)
                for pa,pb in zip(a.parameters(),b.parameters(),strict=True):
                    if pa.grad is None or pb.grad is None:self.assertIs(pa.grad,pb.grad)
                    else:torch.testing.assert_close(pa.grad,pb.grad,atol=3e-6,rtol=2e-4)
                if self.device=='cpu':
                    for model in [a,b]:torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True,foreach=False)
                    oa=RecipeOptimizer(ownership(a),3e-4);ob=RecipeOptimizer(ownership(b),3e-4);oa.step();ob.step()
                    for pa,pb in zip(a.parameters(),b.parameters(),strict=True):torch.testing.assert_close(pa,pb,atol=3e-6,rtol=2e-4)


if __name__=='__main__':unittest.main()
