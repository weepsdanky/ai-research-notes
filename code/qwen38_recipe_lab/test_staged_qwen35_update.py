"""Exact reference gradients and serialized stage continuation; CPU only."""
import copy
import io
import unittest
import torch
from qwen35_controls import ControlConfig
from qwen35_tiled_update import tiled_control
from streamed_lm_backward import streamed_lm_backward
from staged_qwen35_update import StagedControlUpdate
from train_qwen35_controls import balance_loss
from qwen35_long_phase import exact
from optim import RecipeOptimizer,ownership


def config():
    return ControlConfig(vocab_size=97,width=16,layers=4,heads=2,kv_heads=1,head_dim=16,rotary_dim=4,gdn_k_heads=1,gdn_v_heads=2,gdn_k_dim=16,gdn_v_dim=16,hidden=4,experts=4,top_k=2,window=4,query_chunk=2,scan_chunk=4,activation_checkpoint=True)


def restored(value):
    stream=io.BytesIO();torch.save(value,stream);stream.seek(0)
    return torch.load(stream,weights_only=True)


class StagedUpdateTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def pair(self,arm):
        return [tiled_control(config(),arm,42,2).train() for _ in range(2)]

    def test_all_three_arms_exact_reference_and_every_boundary(self):
        for arm in ['full','swa','gdn']:
            with self.subTest(arm=arm):
                reference,staged=self.pair(arm);rng=torch.Generator().manual_seed(77)
                x=torch.randint(97,(1,11),generator=rng);y=torch.randint(97,(1,11),generator=rng)
                hidden,routers=reference.hidden(x);balance=balance_loss(routers,4,2)
                loss,adjoint=streamed_lm_backward(hidden,reference.head.weight,y,3)
                torch.autograd.backward((hidden,balance),(adjoint,balance.new_tensor(.001)))
                run=StagedControlUpdate(staged,x,y,binding='fixture-owner',chunk=3)
                phases=set()
                while not run.done:
                    phases.add(run.s['phase']);run.advance()
                    state=restored(run.snapshot());new=tiled_control(config(),arm,42,2).train();new.load_state_dict(staged.state_dict())
                    staged=new;run=StagedControlUpdate(staged,x,y,binding='fixture-owner',chunk=3,state=state)
                self.assertEqual(phases,{'embedding_forward','layer_forward','norm_forward','head','aux','norm_backward','layer_backward','embedding_backward'})
                self.assertTrue(torch.equal(loss,run.s['lm']));self.assertTrue(torch.equal(balance,run.s['balance']))
                for (name,a),(other,b) in zip(reference.named_parameters(),staged.named_parameters(),strict=True):
                    self.assertEqual(name,other)
                    self.assertEqual(a.grad is None,b.grad is None,name)
                    if a.grad is not None:self.assertTrue(torch.equal(a.grad,b.grad),name)

    def test_muon_eight_updates_and_serialized_four_plus_four(self):
        models=self.pair('gdn');opts=[RecipeOptimizer(ownership(m),3e-4,weight_decay=.1,ns_dtype='bfloat16') for m in models]
        rngs=[torch.Generator().manual_seed(16) for _ in models]
        for step in range(8):
            for index in range(2):
                m=models[index];opt=opts[index];rng=rngs[index];opt.zero_grad(set_to_none=True)
                x=torch.randint(97,(1,11),generator=rng);y=torch.randint(97,(1,11),generator=rng)
                if index==0:
                    h,r=m.hidden(x);balance=balance_loss(r,4,2);loss,adj=streamed_lm_backward(h,m.head.weight,y,3)
                    torch.autograd.backward((h,balance),(adj,balance.new_tensor(.001)))
                else:
                    run=StagedControlUpdate(m,x,y,binding=f'update-{step}',chunk=3)
                    while not run.done:run.advance()
                torch.nn.utils.clip_grad_norm_(m.parameters(),.5,error_if_nonfinite=True,foreach=False);opt.step()
            exact(models[0].state_dict(),models[1].state_dict());exact(opts[0].state_dict(),opts[1].state_dict());exact(rngs[0].get_state(),rngs[1].get_state())
            if step==3:
                saved=restored(dict(model=models[1].state_dict(),optimizer=opts[1].state_dict(),rng=rngs[1].get_state()))
                models[1]=tiled_control(config(),'gdn',42,2).train();models[1].load_state_dict(saved['model'])
                opts[1]=RecipeOptimizer(ownership(models[1]),3e-4,weight_decay=.1,ns_dtype='bfloat16');opts[1].load_state_dict(saved['optimizer']);rngs[1].set_state(saved['rng'])

    def test_owner_data_and_progress_corruption_rejected(self):
        m=self.pair('full')[0];x=torch.arange(11)[None];y=x+1
        run=StagedControlUpdate(m,x,y,binding='owner',chunk=3)
        for _ in range(8):run.advance()
        state=restored(run.snapshot())
        for owner,inputs,change in [('wrong',x,None),('owner',x+1,None),('owner',x,'progress')]:
            bad=copy.deepcopy(state)
            if change:bad['ce_done']=100
            fresh=self.pair('full')[0]
            with self.assertRaises(ValueError):StagedControlUpdate(fresh,inputs,y,binding=owner,chunk=3,state=bad)


if __name__=='__main__':unittest.main()
