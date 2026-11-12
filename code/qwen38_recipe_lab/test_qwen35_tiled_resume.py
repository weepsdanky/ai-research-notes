"""CPU uninterrupted8 versus serialized4+4, with the selected tiling/math."""
import io,unittest
import torch
from qwen35_controls import ControlConfig
from qwen35_tiled_update import tiled_control
from qwen35_long_phase import exact
from streamed_lm_backward import streamed_lm_backward
from train_qwen35_controls import balance_loss
from optim import RecipeOptimizer,ownership


class TiledResumeTests(unittest.TestCase):
    def test_all_arms_eight_steps_bitwise(self):
        torch.set_num_threads(1)
        c=ControlConfig(vocab_size=97,width=16,heads=8,kv_heads=1,head_dim=16,rotary_dim=4,gdn_k_dim=16,gdn_v_dim=16,hidden=8,experts=4,top_k=2,query_chunk=128,scan_chunk=128)
        for arm in ['full','swa','gdn']:
            with self.subTest(arm=arm):
                def initialize():
                    model=tiled_control(c,arm,42,32).train();optimizer=RecipeOptimizer(ownership(model),3e-4,weight_decay=.1,ns_dtype='bfloat16');rng=torch.Generator().manual_seed(867)
                    return model,optimizer,rng
                def advance(model,optimizer,rng,count):
                    rows=[]
                    for _ in range(count):
                        data=torch.randint(0,97,(1,260),generator=rng);optimizer.zero_grad(set_to_none=True)
                        hidden,routers=model.hidden(data[:,:-1]);balance=balance_loss(routers,c.experts,c.top_k);lm,g=streamed_lm_backward(hidden,model.head.weight,data[:,1:],8)
                        torch.autograd.backward((hidden,balance),(g,balance.new_tensor(.001)));norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True,foreach=False);optimizer.step();rows.append((lm.item(),balance.item(),norm.item()))
                    return rows
                a,oa,ra=initialize();expected=advance(a,oa,ra,8)
                b,ob,rb=initialize();actual=advance(b,ob,rb,4)
                buffer=io.BytesIO();torch.save(dict(model=b.state_dict(),optimizer=ob.state_dict(),rng=rb.get_state()),buffer);del b,ob,rb;buffer.seek(0)
                checkpoint=torch.load(buffer,map_location='cpu',weights_only=True);b,ob,rb=initialize();b.load_state_dict(checkpoint['model']);ob.load_state_dict(checkpoint['optimizer']);rb.set_state(checkpoint['rng'])
                exact(b.state_dict(),checkpoint['model']);exact(ob.state_dict(),checkpoint['optimizer']);exact(rb.get_state(),checkpoint['rng'])
                actual+=advance(b,ob,rb,4);self.assertEqual(actual,expected)
                exact(a.state_dict(),b.state_dict());exact(oa.state_dict(),ob.state_dict());exact(ra.get_state(),rb.get_state())


if __name__=='__main__':unittest.main()
