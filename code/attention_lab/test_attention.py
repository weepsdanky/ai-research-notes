"""Semantic checks; run on CPU or set LAB_DEVICE=mps."""
import os
import unittest

import torch
from attention import CausalMHA, TinyLM


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.device = os.environ.get("LAB_DEVICE", "cpu")

    def test_weights_and_future_independence(self):
        module = CausalMHA(8, 2).to(self.device)
        x = torch.randn(1, 4, 8, device=self.device)
        y, weights = module(x, return_weights=True)
        self.assertEqual(tuple(weights.shape), (1, 2, 4, 4))
        torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
        self.assertTrue((weights.triu(1) == 0).all().item())
        changed = x.clone()
        changed[:, 3] += 20
        torch.testing.assert_close(module(changed)[:, :3], y[:, :3], atol=1e-5, rtol=1e-5)

    def test_loss_backpropagates_and_updates_qkv(self):
        model = TinyLM(7, context=4, width=8, heads=2).to(self.device)
        x = torch.tensor([[0, 1, 2, 3]], device=self.device)
        y = torch.tensor([[1, 2, 3, 4]], device=self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        before = {name: p.detach().clone() for name, p in model.attention.named_parameters()}
        _, loss = model(x, y)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        for name in ("q.weight", "k.weight", "v.weight"):
            grad = dict(model.attention.named_parameters())[name].grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all().item())
            self.assertGreater(grad.abs().sum().item(), 0)
        optimizer.step()
        for name in ("q.weight", "k.weight", "v.weight"):
            self.assertFalse(torch.equal(before[name], dict(model.attention.named_parameters())[name]))


if __name__ == "__main__":
    unittest.main()
