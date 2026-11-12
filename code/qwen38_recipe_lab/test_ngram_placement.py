"""Checks for the controlled Mac N-gram placement experiment."""
import unittest

import torch

from ngram_placement import NGramPlacement, PLACEMENTS
from optim import ownership
from test_recipe import tiny


class PlacementTests(unittest.TestCase):
    def test_placement_budget_backbone_and_optimizer(self):
        cfg = tiny()
        baseline = NGramPlacement(cfg, 'none')
        for arm, slots in PLACEMENTS.items():
            with self.subTest(arm=arm):
                model = NGramPlacement(cfg, arm)
                active = tuple(index for index, layer in enumerate(model.layers, 1) if layer.ple is not None)
                self.assertEqual(active, tuple(layer for layer, _ in slots))
                self.assertEqual(model.ngram_parameters(), 4096*cfg.width//4 if slots else 0)
                for name, value in baseline.state_dict().items():
                    self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
                plans = {name: pieces for name, _, pieces in ownership(model)}
                for layer, _ in slots:
                    plan = plans[f'layers.{layer-1}.ple.embedding.ngram_embedding.weight']
                    self.assertEqual([piece['kind'] for piece in plan], ['adam'])
                self.assertFalse(any('.indexer.' in name for name in plans))

    def test_single_and_dual_placement_forward_and_causality(self):
        cfg = tiny()
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        for arm in ('layer2','layer2_3'):
            model = NGramPlacement(cfg, arm)
            hidden, _ = model.hidden(ids)
            logits, _, _ = model(ids, 'pretrain')
            self.assertTrue(torch.equal(model.head(hidden), logits))
            prefix, _ = model.hidden(ids[:, :5])
            torch.testing.assert_close(prefix, hidden[:, :5], rtol=2e-4, atol=2e-5)
            logits.square().mean().backward()
            for layer, _ in PLACEMENTS[arm]:
                gradient = model.layers[layer-1].ple.embedding.ngram_embedding.weight.grad
                self.assertTrue(torch.isfinite(gradient).all() and torch.any(gradient != 0))


if __name__ == '__main__':
    unittest.main()
