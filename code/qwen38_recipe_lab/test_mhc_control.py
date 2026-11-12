"""Check mHC operators and cached inference across residual controls."""
import unittest

import torch

from mhc_control import MHCResidualControl, ManifoldHyperConnection, sinkhorn
from optim import ownership
from residual_controls import ResidualConfig, ResidualControl, VARIANTS


class ManifoldControlTests(unittest.TestCase):
    def test_sinkhorn_matches_direct_equations_and_backpropagates(self):
        logits = torch.tensor([[0.2, -0.1, 0.7], [1.3, -0.3, 0.5], [0.8, 0.4, -1.2]], requires_grad=True)
        direct = logits.exp()
        for _ in range(20):
            direct = direct/direct.sum(-2, keepdim=True)
            direct = direct/direct.sum(-1, keepdim=True)
        actual = sinkhorn(logits)
        torch.testing.assert_close(actual, direct)
        torch.testing.assert_close(actual.sum(-1), torch.ones(3))
        torch.testing.assert_close(actual.sum(-2), torch.ones(3))
        probe = torch.arange(9, dtype=torch.float32).reshape(3, 3).square()
        a = torch.autograd.grad((probe*actual).sum(), logits)[0]
        b = torch.autograd.grad((probe*direct).sum(), logits)[0]
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-6)

    def test_static_maps_and_branch_orientation(self):
        control = ManifoldHyperConnection(2, 3, 1, dynamic=False)
        x = torch.tensor([[1., 2.], [3., 5.], [7., 11.]])
        torch.testing.assert_close(control(x)[0], x[1], rtol=0.002, atol=0.001)
        for a,b in zip(control.maps(x),control.maps(x*2),strict=True):
            self.assertTrue(torch.equal(a,b))
        mix = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
        output = torch.tensor([13., 17.])
        write = torch.tensor([0., 1., 2.])
        expected = torch.stack([x[1], x[2]+output, x[0]+2*output])
        torch.testing.assert_close(control.update(x,output,write,mix),expected)

    def test_dynamic_parameters_and_input_have_finite_gradients(self):
        torch.manual_seed(71)
        control = ManifoldHyperConnection(4, 3, 0, dynamic=True)
        x = torch.randn(2, 3, 4, requires_grad=True)
        value, write, mix = control(x)
        self.assertFalse(torch.equal(control.maps(x)[0],control.maps(x.flip(-1))[0]))
        control.update(x,value,write,mix).square().mean().backward()
        for parameter in [x,*control.parameters()]:
            self.assertTrue(parameter.grad is not None and torch.isfinite(parameter.grad).all())

    def test_backbone_shared_with_prenorm_and_mhc_gates_use_adamw(self):
        torch.set_num_threads(1)
        config = ResidualConfig(vocab_size=97, eos=0, width=16, rank=2, heads=2,
            head_dim=16, rotary_dim=4, index_dim=16, gdn_k_dim=16, gdn_v_dim=16,
            hidden=8, experts=4, top_k=2, activation_checkpoint=False)
        reference = ResidualControl(config, 'prenorm').state_dict()
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        for arm in ('mhc_static', 'mhc_dynamic'):
            model = MHCResidualControl(config, arm)
            actual = model.state_dict()
            for name, weight in reference.items():
                self.assertTrue(torch.equal(weight, actual[name]), name)
            for name, _, pieces in ownership(model):
                if name.startswith('maps.'):
                    self.assertEqual([piece['kind'] for piece in pieces], ['adamw'])
            logits, routers = model(ids)
            self.assertEqual(tuple(logits.shape), (1, 7, 97))
            self.assertEqual(len(routers), 4)
            logits.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.maps.parameters()))


    def test_cpu_residual_cache_matches_full_prefix(self):
        from dataclasses import asdict
        from prefix_inference import PrefixInference
        from residual_inference import ResidualInference
        from train_residual_control import make_model

        torch.set_num_threads(1)
        config = ResidualConfig(vocab_size=97, eos=0, width=16, rank=2, heads=2,
            head_dim=16, rotary_dim=4, index_dim=16, gdn_k_dim=16, gdn_v_dim=16,
            hidden=8, experts=4, top_k=2, activation_checkpoint=False)
        sequence = torch.tensor([[1, 7, 3, 11, 9, 2, 5, 5, 2, 9, 11, 3, 7, 1]])
        inputs = torch.cat((sequence, sequence.roll(1, dims=1)), dim=0)
        inputs[1, :4] = inputs[0, :4]
        for variant in (*VARIANTS, 'mhc_static', 'mhc_dynamic'):
            with self.subTest(variant=variant), torch.no_grad():
                model = make_model(dict(config=asdict(config), seed=42), variant).eval()
                runner = ResidualInference(model, prefill_chunk=4)
                prefix = PrefixInference(runner, inputs[0, :4].tolist())
                cache = {}
                actual = prefix(inputs[:, :11], cache)
                expected = model(inputs[:, :11])[0][:, -1]
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
                for end in range(12, inputs.shape[1]+1):
                    actual = runner(inputs[:, end-1:end], cache)
                    expected = model(inputs[:, :end])[0][:, -1]
                    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


if __name__ == '__main__':
    unittest.main()
