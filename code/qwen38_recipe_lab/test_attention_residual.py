"""Equation, history, gradient and serialization checks; not an ablation result."""
import copy
import io
import os
import unittest

import torch
from attention_residual import AttentionResidual, DepthHistory, GatedNorm
from optim import RecipeOptimizer, ownership


class AttentionResidualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.device = os.environ.get('TEST_DEVICE', 'cpu')
        self.x = torch.randn(2, 3, 8, device=self.device)

    def test_zero_query_raw_value_average_and_single_source(self):
        read = AttentionResidual(8).to(self.device)
        h = DepthHistory.start(self.x)
        torch.testing.assert_close(read(h), self.x, rtol=0, atol=0)
        h = h.append(self.x * 100, 1)
        torch.testing.assert_close(read(h), (self.x + self.x * 100) / 2)
        self.assertEqual(torch.count_nonzero(read.query).item(), 0)

    def test_block_boundaries_no_zero_or_duplicate_embedding(self):
        for size in [1, 2, 4]:
            with self.subTest(size=size):
                h = DepthHistory.start(self.x)
                outputs = []
                for step in range(1, 8):
                    y = self.x * step
                    old = h
                    h = h.append(y, size)
                    outputs.append(y)
                    expected = [self.x]
                    for j in range(0, step, size):
                        expected.append(torch.stack(outputs[j:j + size]).sum(0))
                    self.assertEqual(len(h.sources()), len(expected))
                    for actual, reference in zip(h.sources(), expected):
                        torch.testing.assert_close(actual, reference)
                    self.assertEqual(old.outputs, step - 1)
                self.assertEqual(h.outputs, 7)
        for bad in [0, -1, 1.5]:
            with self.assertRaises(ValueError):
                DepthHistory.start(self.x).append(self.x, bad)

    def test_nonuniform_scalar_reference_and_all_gradients(self):
        read = AttentionResidual(8).to(self.device)
        with torch.no_grad():
            read.query.copy_(torch.linspace(-.7, .8, 8, device=self.device))
            read.key_norm.weight.copy_(torch.linspace(.6, 1.4, 8, device=self.device))
        other = copy.deepcopy(read)
        values = [torch.randn_like(self.x, requires_grad=True) for _ in range(3)]
        references = [x.detach().clone().requires_grad_() for x in values]
        actual = read(DepthHistory(tuple(values)))
        scores = []
        for value in references:
            key = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
            scores.append((key * other.key_norm.weight * other.query).sum(-1))
        # Independent pairwise exp/sum rather than the production stack/softmax.
        maximum = torch.stack(scores).max(0).values
        exponents = [(score - maximum).exp() for score in scores]
        total = sum(exponents)
        expected = sum(v * (e / total).unsqueeze(-1) for v, e in zip(references, exponents))
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        actual.square().sum().backward();expected.square().sum().backward()
        for a, b in zip(values + list(read.parameters()), references + list(other.parameters())):
            self.assertIsNotNone(a.grad)
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=2e-5)

    def test_full_equals_block_one_for_outputs_and_gradients(self):
        read = AttentionResidual(8).to(self.device)
        with torch.no_grad():read.query.normal_(std=.2)
        copies = [[torch.randn_like(self.x, requires_grad=True) for _ in range(6)]]
        copies.append([x.detach().clone().requires_grad_() for x in copies[0]])
        h = DepthHistory.start(copies[0][0])
        for value in copies[0][1:]:h = h.append(value, 1)
        a = read(h);b = read(DepthHistory(tuple(copies[1])))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.sum().backward();b.sum().backward()
        for x, y in zip(*copies):torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)

    def test_gated_norm_equation_and_roundtrip(self):
        norm = GatedNorm(8).to(self.device)
        x = self.x.clone().requires_grad_()
        normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * norm.norm.weight
        z = normalized @ norm.down.weight.T
        expected = normalized * ((z * z.sigmoid()) @ norm.up.weight.T).sigmoid()
        torch.testing.assert_close(norm(x), expected, rtol=2e-5, atol=2e-6)
        norm(x).square().sum().backward()
        for p in [x, *norm.parameters()]:self.assertTrue(torch.isfinite(p.grad).all().item())
        read = AttentionResidual(8).to(self.device)
        with torch.no_grad():read.query.normal_()
        buffer = io.BytesIO();torch.save(read.state_dict(), buffer);buffer.seek(0)
        restored = AttentionResidual(8).to(self.device)
        restored.load_state_dict(torch.load(buffer, weights_only=True, map_location=self.device))
        h = DepthHistory.start(self.x).append(self.x * 2, 2)
        torch.testing.assert_close(read(h), restored(h), rtol=0, atol=0)

    def test_cpu_eight_updates_equal_serialized_four_plus_four(self):
        if self.device != 'cpu':self.skipTest('Exact component optimizer resume is CPU only')
        def initialize():
            torch.manual_seed(19)
            module = torch.nn.ModuleDict(dict(read=AttentionResidual(8), gn=GatedNorm(8)))
            plans = ownership(module)
            self.assertTrue(all(s['kind'] == 'adamw' for _, _, specs in plans for s in specs))
            optimizer = RecipeOptimizer(plans, lr=3e-4)
            rng = torch.Generator().manual_seed(77)
            return module, optimizer, rng
        def advance(module, optimizer, rng, steps):
            losses = []
            for _ in range(steps):
                values = [torch.randn(2, 3, 8, generator=rng) for _ in range(5)]
                target = torch.randn(2, 3, 8, generator=rng)
                history = DepthHistory.start(values[0])
                for value in values[1:]:history = history.append(value, 2)
                optimizer.zero_grad(set_to_none=True)
                loss = (module['gn'](module['read'](history)) - target).square().mean()
                loss.backward();optimizer.step();losses.append(loss.item())
            return losses
        a, oa, ra = initialize();expected = advance(a, oa, ra, 8)
        b, ob, rb = initialize();actual = advance(b, ob, rb, 4)
        buffer = io.BytesIO()
        torch.save(dict(model=b.state_dict(), optimizer=ob.state_dict(), rng=rb.get_state()), buffer)
        buffer.seek(0);saved = torch.load(buffer, weights_only=True)
        b, ob, rb = initialize()
        b.load_state_dict(saved['model']);ob.load_state_dict(saved['optimizer']);rb.set_state(saved['rng'])
        actual += advance(b, ob, rb, 4)
        self.assertEqual(actual, expected)
        def exact(x, y):
            if isinstance(x, torch.Tensor):self.assertTrue(torch.equal(x, y))
            elif isinstance(x, dict):
                self.assertEqual(x.keys(), y.keys())
                for key in x:exact(x[key], y[key])
            elif isinstance(x, (list, tuple)):
                self.assertEqual(len(x), len(y))
                for left, right in zip(x, y):exact(left, right)
            else:self.assertEqual(x, y)
        exact(a.state_dict(), b.state_dict());exact(oa.state_dict(), ob.state_dict());exact(ra.get_state(), rb.get_state())


if __name__ == '__main__':unittest.main()
