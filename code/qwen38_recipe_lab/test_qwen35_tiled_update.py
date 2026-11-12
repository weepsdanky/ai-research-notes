"""Compare query32/head8 execution with query128/head16 before long runs."""
import os,unittest
import torch
from qwen35_controls import ControlConfig
from qwen35_layer_checkpoint import LayerCheckpointControl
from qwen35_tiled_update import tiled_control
from streamed_lm_backward import streamed_lm_backward
from train_qwen35_controls import balance_loss
from optim import RecipeOptimizer,ownership


class TilingTests(unittest.TestCase):
    def test_all_arms_loss_gradients_and_cpu_update(self):
        torch.set_num_threads(1);torch.manual_seed(824);device=os.environ.get('LAB_TEST_DEVICE','cpu')
        c=ControlConfig(vocab_size=97,width=16,heads=8,kv_heads=1,head_dim=16,rotary_dim=4,gdn_k_dim=16,gdn_v_dim=16,hidden=8,experts=4,top_k=2,query_chunk=128,scan_chunk=128)
        ids=torch.randint(0,97,(1,259),device=device);targets=torch.randint(0,97,(1,259),device=device)
        for arm in ['full','swa','gdn']:
            with self.subTest(arm=arm):
                original=LayerCheckpointControl(c,arm,seed=42).to(device).train();tiled=tiled_control(c,arm,42,32).to(device).train();tiled.load_state_dict(original.state_dict())
                losses=[]
                for model,chunk in [(original,16),(tiled,8)]:
                    hidden,routers=model.hidden(ids);balance=balance_loss(routers,c.experts,c.top_k);lm,g=streamed_lm_backward(hidden,model.head.weight,targets,chunk)
                    torch.autograd.backward((hidden,balance),(g,balance.new_tensor(.001)));losses.append(lm)
                torch.testing.assert_close(losses[0],losses[1],atol=3e-6,rtol=3e-6)
                for a,b in zip(original.parameters(),tiled.parameters(),strict=True):
                    if a.grad is None or b.grad is None:self.assertIs(a.grad,b.grad)
                    else:torch.testing.assert_close(a.grad,b.grad,atol=3e-6,rtol=2e-4)
                if device=='cpu':
                    for model in [original,tiled]:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True,foreach=False)
                        optimizer=RecipeOptimizer(ownership(model),3e-4);optimizer.step()
                    for a,b in zip(original.parameters(),tiled.parameters(),strict=True):torch.testing.assert_close(a,b,atol=3e-6,rtol=2e-4)


if __name__=='__main__':unittest.main()
