"""Run on Mac mini: LAB_TEST_DEVICE=mps uv run python -m unittest -v."""
import os
import unittest

import torch

from model import Config, Model, Indexer, RMSNorm, delta_scan
from optim import Muon, make_optimizers
from prepare import document_split, encode


class LabTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.device = os.environ.get("LAB_TEST_DEVICE", "cpu")
        self.config = Config(width=32, hidden=48, budget=8)

    def test_zero_centered_gain_is_not_mean_centering(self):
        x = torch.tensor([3.0, 4.0], device=self.device)
        norm = RMSNorm(2).to(self.device)
        expected = x / (x.square().mean() + 1e-6).sqrt()
        torch.testing.assert_close(norm(x), expected)
        self.assertGreater(norm(x).mean().item(), 0)

    def test_delta_overwrites_repeated_unit_key(self):
        # alpha=beta=1 is the limiting case: overwrite, not additive accumulation.
        k = torch.tensor([[[[1., 0.]], [[1., 0.]]]], device=self.device)
        v = torch.tensor([[[[2., 3.]], [[5., 7.]]]], device=self.device)
        gates = torch.ones(1, 2, 1, device=self.device)
        torch.testing.assert_close(delta_scan(k, k, v, gates, gates), v)

    def test_indexer_never_reads_future_and_keeps_incomplete_tail(self):
        indexer = Indexer(self.config).to(self.device)
        x = torch.randn(2, 15, 32, device=self.device)
        _, valid, selected, mask = indexer(x)
        self.assertFalse(mask.triu(1).any().item())
        self.assertFalse((selected & ~valid).any().item())
        self.assertTrue((selected.sum(-1) <= 2).all().item())
        for i in range(15):
            start = ((i + 1) // 4) * 4
            self.assertTrue(mask[:, i, start:i + 1].all().item())
            self.assertTrue(mask[:, i].any(-1).all().item())

    def test_model_causality_in_all_phases(self):
        model = Model(self.config).to(self.device).eval()
        ids = torch.randint(0, 257, (1, 12), device=self.device)
        changed = ids.clone()
        changed[:, 6:] = (changed[:, 6:] + 37) % 257
        with torch.no_grad():
            for stage in ("pretrain", "distill", "sparse"):
                before = model(ids, stage)[0][:, :6]
                after = model(changed, stage)[0][:, :6]
                prefix = model(ids[:, :6], stage)[0]
                torch.testing.assert_close(before, after, atol=2e-5, rtol=2e-4)
                torch.testing.assert_close(before, prefix, atol=2e-5, rtol=2e-4)

    def test_sparse_equals_dense_when_all_blocks_fit(self):
        model = Model(self.config).to(self.device).eval()
        ids = torch.randint(0, 257, (1, 7), device=self.device)
        with torch.no_grad():
            torch.testing.assert_close(model(ids, "pretrain")[0], model(ids, "sparse")[0])

    def test_distillation_changes_only_indexer(self):
        model = Model(self.config).to(self.device)
        optimizers = make_optimizers(model, "distill", "muon", 1e-3)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        ids = torch.randint(0, 257, (1, 12), device=self.device)
        _, loss, _ = model(ids, "distill")
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        changed = [name for name, p in model.named_parameters() if not torch.equal(p, before[name])]
        self.assertTrue(changed)
        self.assertTrue(all("indexer" in name for name in changed))

    def test_backward_and_optimizer_partition(self):
        for stage in ("pretrain", "sparse"):
            model = Model(self.config).to(self.device)
            optimizers = make_optimizers(model, stage, "muon", 3e-4)
            params = [p for o in optimizers for g in o.param_groups for p in g["params"]]
            self.assertEqual(len(params), len({id(p) for p in params}))
            self.assertEqual({id(p) for p in params}, {id(p) for p in model.parameters() if p.requires_grad})
            ids = torch.randint(0, 257, (1, 12), device=self.device)
            logits, kl, balance = model(ids, stage)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), ids.flatten()) + kl + .01 * balance
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None))
            for optimizer in optimizers:
                optimizer.step()
            self.assertTrue(all(torch.isfinite(p).all().item() for p in params))

    def test_muon_zero_gradient_is_safe(self):
        p = torch.nn.Parameter(torch.ones(3, 5, device=self.device))
        optimizer = Muon([p], lr=1e-3, weight_decay=0)
        p.grad = torch.zeros_like(p)
        optimizer.step()
        torch.testing.assert_close(p, torch.ones_like(p))

    def test_byte_tokenizer_and_deterministic_document_split(self):
        self.assertEqual(encode("你好 Shakespeare").astype("uint8").tobytes().decode(), "你好 Shakespeare")
        self.assertEqual(document_split("document"), document_split("document"))


if __name__ == "__main__":
    unittest.main()
