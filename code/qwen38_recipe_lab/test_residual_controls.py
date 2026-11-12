"""Mac residual construction, math and serialized CPU resume fixtures."""
from dataclasses import asdict
import io
import os
import unittest

import torch
from torch import nn
from attention_residual import AttentionResidual
from model import GatedResidual, GDN
from optim import RecipeOptimizer, ownership
from residual_controls import ResidualConfig, ResidualControl, VARIANTS
from train_qwen35_controls import balance_loss
from train_residual_control import make_model


class ResidualControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.device = os.environ.get('TEST_DEVICE', 'cpu')
        self.config = ResidualConfig(vocab_size=97, eos=0, width=16, rank=2,
            heads=12, head_dim=16, rotary_dim=4, gdn_k_dim=16, gdn_v_dim=16,
            hidden=8, experts=4, top_k=2, index_dim=16, activation_checkpoint=False)
        self.ids = torch.tensor([[1, 7, 3, 11, 9, 2, 5]], device=self.device)

    def make(self, variant):
        return ResidualControl(self.config, variant).to(self.device)

    def test_shared_backbone_bitwise_and_norm_query_initialization(self):
        reference = self.make('prenorm').state_dict()
        shared = [name for name in reference if name.startswith(('embedding.', 'head.')) or '.mix.' in name or '.moe.' in name]
        self.assertTrue(shared)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = self.make(variant)
                for name in shared:self.assertTrue(torch.equal(reference[name], model.state_dict()[name]), name)
                for module in model.modules():
                    if isinstance(module, AttentionResidual):self.assertEqual(torch.count_nonzero(module.query).item(), 0)
                    if isinstance(module, nn.RMSNorm):self.assertTrue(torch.equal(module.weight, torch.ones_like(module.weight)))
                self.assertEqual(sum(isinstance(m, GDN) for m in model.modules()), 3)
                plans = {name: specs for name, _, specs in ownership(model)}
                q_gate = plans['layers.3.mix.q_gate.weight']
                self.assertEqual(sum(s['kind'] == 'muon' for s in q_gate), 12)
                self.assertEqual(sum(s['kind'] == 'adamw' for s in q_gate), 12)
                self.assertEqual(len(plans['layers.0.mix.in_proj_qkv.weight']), 5)
                self.assertEqual(plans['layers.0.mix.in_proj_z.weight'][0]['kind'], 'adamw')
                if variant.endswith('_gn'):
                    self.assertEqual(plans['layers.0.mix_input.down.weight'][0]['kind'], 'adamw')

    def test_nine_arms_finite_loss_and_gradients(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = self.make(variant).train()
                logits, routers = model(self.ids)
                self.assertEqual(tuple(logits.shape), (1, 7, 97))
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), self.ids.flatten()) + .001 * balance_loss(routers, 4, 2)
                loss.backward()
                self.assertTrue(torch.isfinite(loss).item())
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                for module in model.modules():
                    if isinstance(module, AttentionResidual):self.assertIsNotNone(module.query.grad)

    def test_zero_outputs_preserve_embedding_once_with_correct_depth_divisor(self):
        for variant, (mode, _, size) in VARIANTS.items():
            with self.subTest(variant=variant):
                model = self.make(variant).eval()
                with torch.no_grad():
                    for layer in model.layers:
                        (layer.mix.out_proj if isinstance(layer.mix, GDN) else layer.mix.out).weight.zero_()
                        for expert in [*layer.moe.experts, layer.moe.shared]:expert.down.weight.zero_()
                    embedding = model.embedding(self.ids)
                    if mode == 'gr':
                        expected = model.readout(embedding.repeat(1, 1, 4))
                    else:
                        model.readout = nn.Identity()
                        expected = embedding if mode == 'plain' else embedding / (1 + 8 // size)
                    hidden, _ = model.hidden(self.ids)
                torch.testing.assert_close(hidden, expected, rtol=2e-5, atol=2e-6)

    def test_causal_prefix_all_arms(self):
        extended = torch.cat((self.ids, self.ids[:, :2]), 1)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = self.make(variant).eval()
                with torch.no_grad():a, _ = model(self.ids);b, _ = model(extended)
                torch.testing.assert_close(a, b[:, :7], rtol=2e-4, atol=2e-5)

    def test_cpu_all_arms_eight_updates_equal_serialized_four_plus_four(self):
        if self.device != 'cpu':self.skipTest('Exact full small-model optimizer resume is CPU only')
        def initialize(variant):
            model = make_model(dict(config=asdict(self.config), seed=42), variant).train()
            optimizer = RecipeOptimizer(ownership(model), lr=3e-4)
            return model, optimizer, torch.Generator().manual_seed(707)
        def advance(model, optimizer, rng, steps):
            history = []
            for _ in range(steps):
                data = torch.randint(0, 97, (1, 8), generator=rng)
                optimizer.zero_grad(set_to_none=True)
                logits, routers = model(data[:, :-1])
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), data[:, 1:].flatten()) + .001 * balance_loss(routers, 4, 2)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True, foreach=False)
                optimizer.step();history.append((loss.item(), norm.item()))
            return history
        def exact(a, b):
            if isinstance(a, torch.Tensor):self.assertTrue(torch.equal(a, b))
            elif isinstance(a, dict):
                self.assertEqual(a.keys(), b.keys())
                for key in a:exact(a[key], b[key])
            elif isinstance(a, (list, tuple)):
                self.assertEqual(len(a), len(b))
                for x, y in zip(a, b):exact(x, y)
            else:self.assertEqual(a, b)
        for variant in (*VARIANTS, 'mhc_static', 'mhc_dynamic'):
            with self.subTest(variant=variant):
                a, oa, ra = initialize(variant);expected = advance(a, oa, ra, 8)
                b, ob, rb = initialize(variant);actual = advance(b, ob, rb, 4)
                buffer = io.BytesIO();torch.save(dict(model=b.state_dict(), optimizer=ob.state_dict(), rng=rb.get_state()), buffer);buffer.seek(0)
                saved = torch.load(buffer, weights_only=True);b, ob, rb = initialize(variant)
                b.load_state_dict(saved['model']);ob.load_state_dict(saved['optimizer']);rb.set_state(saved['rng'])
                actual += advance(b, ob, rb, 4)
                self.assertEqual(actual, expected)
                exact(a.state_dict(), b.state_dict());exact(oa.state_dict(), ob.state_dict());exact(ra.get_state(), rb.get_state())


if __name__ == '__main__':unittest.main()
