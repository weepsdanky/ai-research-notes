"""Direct public MoE expert/gate and global balance value/gradient comparisons."""
import os, unittest
import torch
from model import Model, MoE
from reference import load_reference
from test_recipe import tiny, upstream

REF=load_reference()


class MoETests(unittest.TestCase):
    def test_moe_outputs_input_and_fused_parameter_gradients(self):
        torch.manual_seed(31); c=tiny(); device=os.environ.get('LAB_TEST_DEVICE','cpu')
        model=MoE(c).to(device)
        config=upstream(c)
        config.num_experts=c.experts; config.num_experts_per_tok=c.top_k
        config.norm_topk_prob=True; config.moe_intermediate_size=c.hidden
        config.shared_expert_intermediate_size=c.hidden
        ref=REF['Qwen4ExpTextSparseMoeBlock'](config)
        with torch.no_grad():
            ref.gate.weight.copy_(model.router.weight.cpu())
            ref.shared_expert_gate.weight.copy_(model.shared_expert_gate.weight.cpu())
            for i,expert in enumerate(model.experts):
                ref.experts.gate_up_proj[i].copy_(torch.cat((expert.gate.weight,expert.up.weight)).cpu())
                ref.experts.down_proj[i].copy_(expert.down.weight.cpu())
            for key in ('gate','up','down'):
                getattr(ref.shared_expert,key+'_proj').weight.copy_(getattr(model.shared,key).weight.cpu())
        x=torch.randn(2,13,c.width,requires_grad=True); actual_x=x.detach().to(device).requires_grad_()
        expected=ref(x); actual=model(actual_x)[0]
        torch.testing.assert_close(actual.cpu(),expected,atol=3e-6,rtol=3e-5)
        cotangent=torch.randn_like(expected)
        (expected*cotangent).sum().backward(); (actual*cotangent.to(device)).sum().backward()
        torch.testing.assert_close(actual_x.grad.cpu(),x.grad,atol=3e-6,rtol=3e-5)
        torch.testing.assert_close(model.router.weight.grad.cpu(),ref.gate.weight.grad,atol=3e-6,rtol=3e-5)
        torch.testing.assert_close(model.shared_expert_gate.weight.grad.cpu(),ref.shared_expert_gate.weight.grad,atol=3e-6,rtol=3e-5)
        for i,expert in enumerate(model.experts):
            torch.testing.assert_close(torch.cat((expert.gate.weight.grad,expert.up.weight.grad)).cpu(),ref.experts.gate_up_proj.grad[i],atol=3e-6,rtol=3e-5)
            torch.testing.assert_close(expert.down.weight.grad.cpu(),ref.experts.down_proj.grad[i],atol=3e-6,rtol=3e-5)

    def test_global_auxiliary_balance_matches_public_formula(self):
        torch.manual_seed(33); c=tiny(); device=os.environ.get('LAB_TEST_DEVICE','cpu')
        model=Model(c).to(device); captured=[]
        handles=[m.register_forward_hook(lambda m,i,o:captured.append(o[1])) for m in model.modules() if isinstance(m,MoE)]
        ids=torch.randint(0,c.vocab_size,(2,17),device=device)
        balance=model(ids)[2]
        independent=tuple(x.detach().cpu().requires_grad_() for x in captured)
        expected=REF['load_balancing_loss_func'](independent,c.experts,c.top_k)
        torch.testing.assert_close(balance.cpu(),expected,atol=2e-6,rtol=2e-5)
        actual_grad=torch.autograd.grad(balance,captured)
        reference_grad=torch.autograd.grad(expected,independent)
        for a,b in zip(actual_grad,reference_grad,strict=True):
            torch.testing.assert_close(a.cpu(),b,atol=2e-6,rtol=2e-5)
        for handle in handles: handle.remove()


if __name__=='__main__': unittest.main()
